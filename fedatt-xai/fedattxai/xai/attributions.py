"""Attribution maps at the 14×14 ViT patch grid (Section 3.6).

* raw attention      final block, head-averaged, CLS row without CLS->CLS
* attention rollout  Abnar & Zuidema: (A + I) row-normalised, multiplied from
                     the first block to the last, CLS row
* ViT-Grad-CAM       output of final block (before the final norm / head);
                     patch tokens as 14×14×384 feature map; gradient weights
                     averaged over positions; ReLU of the weighted sum
* GradientSHAP       Captum, 128 samples, client-local background (≤64 images),
                     |channel-wise contributions| summed within each 16×16 patch

Target = logit of the model-predicted class unless given explicitly.
All maps are returned at 14×14 and min–max normalised to [0, 1].
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import GradientShap

from ..models.backbones import set_attention_capture

GRID = 14
PATCH = 16


def minmax(a: torch.Tensor) -> torch.Tensor:
    a = a - a.min()
    return a / a.max().clamp_min(1e-12)


def predicted_class(model, x) -> int:
    with torch.no_grad():
        return int(model(x).argmax(-1).item())


# --------------------------------------------------------------------------- #
def raw_attention(model, x, target=None) -> torch.Tensor:
    set_attention_capture(model, True)
    with torch.no_grad():
        model(x)
    A = model.blocks[-1].attn.last_attn.mean(1)[0]        # (197,197), mean over 6 heads
    set_attention_capture(model, False)
    return minmax(A[0, 1:].reshape(GRID, GRID).float())


def attention_rollout(model, x, target=None) -> torch.Tensor:
    set_attention_capture(model, True)
    with torch.no_grad():
        model(x)
    rollout = None
    for blk in model.blocks:                                  # first -> last
        A = blk.attn.last_attn.mean(1)[0].float()
        A = A + torch.eye(A.shape[-1], device=A.device)      # residual connection
        A = A / A.sum(-1, keepdim=True)
        rollout = A if rollout is None else A @ rollout
    set_attention_capture(model, False)
    return minmax(rollout[0, 1:].reshape(GRID, GRID))


def vit_gradcam(model, x, target=None) -> torch.Tensor:
    store = {}

    def hook(_m, _i, out):
        out.retain_grad()
        store["act"] = out

    h = model.blocks[-1].register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(False)
    with torch.enable_grad():
        logits = model(x)
        t = int(logits.argmax(-1)) if target is None else int(target)
        logits[0, t].backward()
    h.remove()
    act = store["act"][0, 1:]                                 # (196, 384) patch tokens
    grad = store["act"].grad[0, 1:]
    weights = grad.mean(0)                                    # average over positions
    cam = F.relu((act * weights).sum(-1)).reshape(GRID, GRID).detach().float()
    model.zero_grad(set_to_none=True)
    return minmax(cam)


def patch_pool(attr: torch.Tensor) -> torch.Tensor:
    """(1,3,224,224) pixel attributions -> (14,14): sum |channel contributions|."""
    a = attr.abs().sum(1, keepdim=True)
    return F.avg_pool2d(a, PATCH, PATCH)[0, 0] * PATCH * PATCH


def gradient_shap(model, x, background: torch.Tensor, target=None, n_samples: int = 128,
                  chunk: int = 32, stdevs: float = 0.0, seed: int = 0, normalize: bool = True,
                  return_raw: bool = False):
    """GradientSHAP with a client-local background set. The 128 samples are
    drawn in chunks (4 × 32 by default) and averaged, which is the same
    Monte-Carlo estimator with bounded memory."""
    model.zero_grad(set_to_none=True)
    t = predicted_class(model, x) if target is None else int(target)
    gs = GradientShap(model)
    torch.manual_seed(seed)
    total, done = None, 0
    while done < n_samples:
        k = min(chunk, n_samples - done)
        a = gs.attribute(x, baselines=background, target=t, n_samples=k, stdevs=stdevs)
        total = a.detach() * k if total is None else total + a.detach() * k
        done += k
    attr = total / n_samples
    grid = patch_pool(attr).float()
    if return_raw:
        return grid
    return minmax(grid) if normalize else grid


METHODS = {"raw_attention": raw_attention, "attention_rollout": attention_rollout,
           "vit_gradcam": vit_gradcam, "gradient_shap": gradient_shap}


def draw_background(dataset, size: int, seed: int) -> torch.Tensor:
    """≤64 images uniformly at random without replacement from the client's own
    (non-augmented) training partition; the full partition if it is smaller."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(size, len(dataset)), replace=False)
    return torch.stack([dataset[i][0] for i in idx])


def stratified_background(datasets, size: int, seed: int) -> torch.Tensor:
    """Common background (Table 28 only): class-stratified sample from the pooled
    client training partitions. Offline analysis — requires pooling."""
    rng = np.random.default_rng(seed)
    items = [(d, i, int(d.labels[i])) for d in datasets for i in range(len(d))]
    labels = np.array([l for _, _, l in items])
    classes = np.unique(labels)
    per = size // len(classes)
    chosen = []
    for c in classes:
        pool = np.where(labels == c)[0]
        chosen += list(rng.choice(pool, size=min(per, len(pool)), replace=False))
    return torch.stack([items[j][0][items[j][1]][0] for j in chosen])
