"""ViT-Small/16 (ImageNet-21k) backbone with attention capture (Section 3.3),
plus the ResNet-50 backbone used only in the ablation (Table 17)."""
from __future__ import annotations

import re
import types
from typing import Dict, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

VIT_NAME = "vit_small_patch16_224.augreg_in21k"
RESNET_NAME = "resnet50.a1_in1k"


def _attn_forward_with_capture(self, x, *args, **kwargs):
    """Explicit (non-fused) scaled dot-product attention, Eq. (4)-(5).

    Stores the post-softmax attention matrix in `self.last_attn`
    (B, heads, 197, 197) when `self.capture` is True. The explicit form also
    supports the double backward needed by the DLG attack."""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    attn = (q * self.scale) @ k.transpose(-2, -1)          # / sqrt(d_k), d_k = 64
    attn = attn.softmax(dim=-1)
    if getattr(self, "capture", False):
        self.last_attn = attn
    attn = self.attn_drop(attn)
    x = attn @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    if hasattr(self, "norm"):
        x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def create_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if backbone == "vit_small":
        model = timm.create_model(VIT_NAME, pretrained=pretrained, num_classes=num_classes)
        for blk in model.blocks:
            blk.attn.forward = types.MethodType(_attn_forward_with_capture, blk.attn)
            blk.attn.capture = False
    elif backbone == "resnet50":
        model = timm.create_model(RESNET_NAME, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(backbone)
    model.backbone_name = backbone
    return model


# --------------------------------------------------------------------------- #
# Parameter bookkeeping
# --------------------------------------------------------------------------- #
def head_prefix(model: nn.Module) -> str:
    return "head." if getattr(model, "backbone_name", "vit_small") == "vit_small" else "fc."


def is_head(name: str, model: nn.Module) -> bool:
    return name.startswith(head_prefix(model))


def layer_key(name: str) -> str:
    """Encoder layer index ℓ used by the layer-wise FedAtt weights (Eq. 6).
    ViT : cls_token / pos_embed / patch_embed / blocks.<i> / norm / head
    ResNet: conv1 / bn1 / layer<k>.<j> / fc"""
    m = re.match(r"^(blocks\.\d+|layer\d+\.\d+)\.", name)
    if m:
        return m.group(1)
    return name.split(".")[0]


def is_layernorm_param(name: str, model: nn.Module) -> bool:
    mod_name = name.rsplit(".", 1)[0]
    mod = dict(model.named_modules()).get(mod_name)
    return isinstance(mod, nn.LayerNorm)


def param_groups(model: nn.Module, lr_head: float, lr_encoder: float, weight_decay: float,
                 freeze_encoder: bool = False) -> List[Dict]:
    head, enc = [], []
    for n, p in model.named_parameters():
        if is_head(n, model):
            p.requires_grad_(True)
            head.append(p)
        else:
            p.requires_grad_(not freeze_encoder)
            if not freeze_encoder:
                enc.append(p)
    groups = [{"params": head, "lr": lr_head, "weight_decay": weight_decay, "name": "head"}]
    if enc:
        groups.append({"params": enc, "lr": lr_encoder, "weight_decay": weight_decay, "name": "encoder"})
    return groups


def features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Pre-logit representation (CLS token for ViT) — used by MOON."""
    return model.forward_head(model.forward_features(x), pre_logits=True)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    enc = sum(p.numel() for n, p in model.named_parameters() if not is_head(n, model))
    hd = sum(p.numel() for n, p in model.named_parameters() if is_head(n, model))
    return {"encoder": enc, "head": hd, "total": enc + hd}


def set_attention_capture(model: nn.Module, on: bool) -> None:
    for blk in getattr(model, "blocks", []):
        blk.attn.capture = on
        if not on and hasattr(blk.attn, "last_attn"):
            del blk.attn.last_attn


def state_to_cpu(model_or_state) -> Dict[str, torch.Tensor]:
    sd = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    return {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}
