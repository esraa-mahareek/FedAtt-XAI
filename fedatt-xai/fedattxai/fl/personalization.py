"""Phase 2 of Algorithm 1 — validation-guided client personalisation (Eq. 8).

|D_i^tr| < τ  -> encoder frozen, head trained at η_p = 3e-3
otherwise    -> all parameters, head 3e-3 / encoder 3e-5
Ep is selected per client from {1,2,3,5,10} on the client's validation loss,
with early stopping at patience 2. The test partition is never touched.

Implementation note: because η_p is constant (no schedule) and the data order
is seeded, training once for up to max(grid) epochs while recording the
validation loss after every epoch is equivalent to training each candidate
from W_g* separately; the checkpoint of the best grid epoch is kept.
"""
from __future__ import annotations

import copy
from typing import Dict

import torch

from ..evaluation.metrics import predict
from ..models.backbones import state_to_cpu
from .local_training import make_optimizer, train_epochs


def personalize_client(model, client, cfg, device, seed: int, tau: float | None = None) -> Dict:
    pc = cfg["personalization"]
    tau = pc["tau"] if tau is None else tau
    grid = sorted(pc["epoch_grid"])
    patience = pc.get("patience", 2)
    n_train = client.n("train")
    head_only = n_train < tau
    regime = "head_only" if head_only else "full"

    opt = make_optimizer(model, cfg, freeze_encoder=head_only,
                         lr_head=pc["lr_head"], lr_encoder=pc["lr_encoder"])
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and cfg["train"].get("amp", True)) else None
    tl = client.loader("train", cfg["train"]["micro_batch"], seed=seed, workers=cfg["train"]["workers"])
    vl = client.loader("val", cfg["train"]["eval_batch"], shuffle=False, augment=False,
                       workers=cfg["train"]["workers"])

    history, best = [], {"loss": float("inf"), "epoch": 0, "state": state_to_cpu(model)}
    v0 = predict(model, vl, device)
    history.append({"epoch": 0, "val_loss": v0["loss_sum"] / max(v0["n"], 1)})
    best_any, bad = history[0]["val_loss"], 0
    for e in range(1, grid[-1] + 1):
        train_epochs(model, tl, opt, device, 1, cfg, scaler=scaler)
        v = predict(model, vl, device)
        vloss = v["loss_sum"] / max(v["n"], 1)
        history.append({"epoch": e, "val_loss": vloss})
        if e in grid and vloss < best["loss"]:
            best = {"loss": vloss, "epoch": e, "state": state_to_cpu(model)}
        if vloss < best_any - 1e-6:
            best_any, bad = vloss, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best["epoch"] == 0:  # no grid epoch evaluated (should not happen) -> 1 epoch default
        best["epoch"] = history[1]["epoch"] if len(history) > 1 else 0
    model.load_state_dict(best["state"])
    for p in model.parameters():
        p.requires_grad_(True)
    return {"regime": regime, "n_train": n_train, "selected_epochs": best["epoch"],
            "val_history": history, "state": best["state"]}
