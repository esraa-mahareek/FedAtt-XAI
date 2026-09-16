"""Client-side optimisation (Algorithm 1, lines 14-19) and the method-specific
local objectives of the baselines (FedProx proximal term, MOON contrastive term,
FedAS parameter alignment and Fisher-information trace)."""
from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.backbones import features, is_head, param_groups


def cosine_factor(round_idx: int, total_rounds: int) -> float:
    """Cosine decay of the learning rate over communication rounds."""
    return 0.5 * (1.0 + math.cos(math.pi * round_idx / max(total_rounds, 1)))


def make_optimizer(model, cfg, lr_scale=1.0, freeze_encoder=False, lr_head=None, lr_encoder=None):
    o = cfg["optim"]
    groups = param_groups(model,
                          lr_head=(lr_head if lr_head is not None else o["lr_head"]) * lr_scale,
                          lr_encoder=(lr_encoder if lr_encoder is not None else o["lr_encoder"]) * lr_scale,
                          weight_decay=o["weight_decay"], freeze_encoder=freeze_encoder)
    return torch.optim.AdamW(groups, betas=tuple(o.get("betas", (0.9, 0.999))), eps=o.get("eps", 1e-8))


def _autocast(device, enabled):
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def train_epochs(model: nn.Module, loader, optimizer, device, epochs: int, cfg: dict,
                 method: str = "fedavg", global_model: Optional[nn.Module] = None,
                 prev_model: Optional[nn.Module] = None, scaler=None) -> Dict[str, float]:
    """Generic local loop with gradient accumulation to an effective batch of
    cfg.optim.batch_size and optional FedProx / MOON terms."""
    amp = cfg["train"].get("amp", True) and device.type == "cuda"
    eff_bs = cfg["optim"]["batch_size"]
    micro_bs = loader.batch_size
    accum = max(1, eff_bs // micro_bs)
    mu = cfg["methods"].get("fedprox", {}).get("mu", 0.01)
    moon_mu = cfg["methods"].get("moon", {}).get("mu", 1.0)
    moon_t = cfg["methods"].get("moon", {}).get("temperature", 0.5)
    if method == "fedprox" and global_model is not None:
        gparams = {n: p.detach() for n, p in global_model.named_parameters()}
    model.train()
    tot_loss, tot_n = 0.0, 0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with _autocast(device, amp):
                if method == "moon":
                    z = features(model, x)
                    logits = model.get_classifier()(z)
                    with torch.no_grad():
                        zg = features(global_model, x)
                        zp = features(prev_model, x)
                    pos = F.cosine_similarity(z, zg, dim=-1) / moon_t
                    neg = F.cosine_similarity(z, zp, dim=-1) / moon_t
                    con = F.cross_entropy(torch.stack([pos, neg], 1),
                                          torch.zeros(len(x), dtype=torch.long, device=device))
                    loss = F.cross_entropy(logits, y) + moon_mu * con
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
            if method == "fedprox" and global_model is not None:
                prox = sum(((p.float() - gparams[n].float()) ** 2).sum()
                           for n, p in model.named_parameters() if p.requires_grad)
                loss = loss + 0.5 * mu * prox
            loss_s = loss / accum
            if scaler is not None:
                scaler.scale(loss_s).backward()
            else:
                loss_s.backward()
            last = (step + 1) == len(loader)
            if (step + 1) % accum == 0 or last:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"].get("grad_clip", 1.0))
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            tot_loss += float(loss.detach()) * len(y)
            tot_n += len(y)
    return {"train_loss": tot_loss / max(tot_n, 1), "n": tot_n}


# --------------------------------------------------------------------------- #
# FedAS helpers (Yang et al., CVPR 2024)
# --------------------------------------------------------------------------- #
def fedas_parameter_alignment(model, loader, device, cfg):
    """Align the client's personalised head with the freshly received shared
    encoder by training the head only (encoder frozen) before local training."""
    epochs = cfg["methods"].get("fedas", {}).get("align_epochs", 1)
    opt = make_optimizer(model, cfg, freeze_encoder=True)
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and cfg["train"].get("amp", True)) else None
    train_epochs(model, loader, opt, device, epochs, cfg, method="fedavg", scaler=scaler)
    for p in model.parameters():
        p.requires_grad_(True)


def fisher_information_trace(model, loader, device, max_batches: int = 10) -> float:
    """Trace of the empirical Fisher information of the shared (encoder)
    parameters, used by FedAS client synchronisation as the aggregation weight."""
    model.eval()
    trace, n = 0.0, 0
    for b, (x, y) in enumerate(loader):
        if b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).float(), y)
        loss.backward()
        trace += sum(float((p.grad.detach() ** 2).sum()) for nme, p in model.named_parameters()
                     if p.grad is not None and not is_head(nme, model)) * len(y)
        n += len(y)
    model.zero_grad(set_to_none=True)
    return trace / max(n, 1)
