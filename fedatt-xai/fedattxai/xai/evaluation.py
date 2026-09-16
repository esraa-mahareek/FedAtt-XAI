"""Explanation evaluation protocol (Section 4.5.2, Tables 23–29)."""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import spearmanr

GRID, PATCH, N_PATCH = 14, 16, 196
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# Perturbation baselines
# --------------------------------------------------------------------------- #
class PerturbationBaseline:
    """Builds the image used to replace (deletion) or start from (insertion).

    mean      channel-wise mean of the source client's training partition
    blur      Gaussian blur, 15×15 kernel, σ = 5.0
    inpaint   OpenCV Telea inpainting of the removed patches (mask-dependent)
    Operates in model-input (normalised) space; `norm` / `denorm` convert."""

    def __init__(self, kind: str, channel_mean: torch.Tensor, norm: Callable, denorm: Callable):
        self.kind, self.mean, self.norm, self.denorm = kind, channel_mean, norm, denorm

    def full(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "mean":
            return self.mean.view(1, 3, 1, 1).to(x).expand_as(x).clone()
        if self.kind == "blur":
            u = self.denorm(x)[0].permute(1, 2, 0).cpu().numpy()
            b = cv2.GaussianBlur(u, (15, 15), 5.0)
            return self.norm(torch.from_numpy(b).permute(2, 0, 1)[None].to(x))
        raise ValueError("inpaint baseline is mask-dependent; use `apply`")

    def apply(self, x: torch.Tensor, patch_mask: np.ndarray) -> torch.Tensor:
        """Return x with the patches in `patch_mask` (bool, 196) replaced."""
        pix = torch.from_numpy(np.kron(patch_mask.reshape(GRID, GRID), np.ones((PATCH, PATCH)))) \
            .to(x.device).bool()[None, None]
        if self.kind in ("mean", "blur"):
            return torch.where(pix, self.full(x), x)
        u = (self.denorm(x)[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        m = pix[0, 0].cpu().numpy().astype(np.uint8)
        if m.any():
            u = cv2.inpaint(u, m, 3, cv2.INPAINT_TELEA)
        return self.norm(torch.from_numpy(u.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(x))


def channel_mean_of(dataset) -> torch.Tensor:
    """Mean of the (normalised) training images of a client, per channel."""
    s, n = torch.zeros(3), 0
    for i in range(len(dataset)):
        x, _ = dataset[i]
        s += x.mean((1, 2))
        n += 1
    return s / max(n, 1)


# --------------------------------------------------------------------------- #
# Faithfulness: deletion / insertion (20 steps × 5% of 196 patches)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def deletion_insertion(model, x, attr: torch.Tensor, target: int, baseline: PerturbationBaseline,
                       steps: int = 20) -> Dict[str, float]:
    order = torch.argsort(attr.flatten(), descending=True).cpu().numpy()
    counts = [int(round(k * N_PATCH / steps)) for k in range(steps + 1)]
    del_imgs, ins_imgs = [], []
    for c in counts:
        m = np.zeros(N_PATCH, bool)
        m[order[:c]] = True
        del_imgs.append(baseline.apply(x, m))              # remove top-c patches
        ins_imgs.append(baseline.apply(x, ~m))             # restore top-c patches
    probs = {}
    for name, imgs in (("deletion", del_imgs), ("insertion", ins_imgs)):
        batch = torch.cat(imgs)
        p = torch.cat([model(b).softmax(-1)[:, target] for b in batch.split(32)]).float().cpu().numpy()
        probs[name] = p
    xs = np.linspace(0, 1, steps + 1)
    return {"deletion_auc": float(_trapz(probs["deletion"], xs)),
            "insertion_auc": float(_trapz(probs["insertion"], xs)),
            "deletion_curve": probs["deletion"].tolist(), "insertion_curve": probs["insertion"].tolist()}


# --------------------------------------------------------------------------- #
# Stability: Spearman between original and 10 noisy copies (σ in [0,1] domain)
# --------------------------------------------------------------------------- #
def stability(model, x, attr_fn: Callable, sigma: float, norm, denorm, n_copies: int = 10,
              seed: int = 0) -> Dict[str, float]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        pred = int(model(x).argmax(-1))
    base = attr_fn(x, target=pred).flatten().cpu().numpy()
    u = denorm(x)
    rhos, excluded = [], 0
    for _ in range(n_copies):
        noise = torch.randn(u.shape, generator=g).to(u) * sigma
        xp = norm((u + noise).clamp(0, 1))
        with torch.no_grad():
            if int(model(xp).argmax(-1)) != pred:
                excluded += 1
                continue
        a = attr_fn(xp, target=pred).flatten().cpu().numpy()
        rhos.append(spearmanr(base, a).statistic)
    return {"stability": float(np.nanmean(rhos)) if rhos else float("nan"),
            "excluded": excluded, "evaluated": len(rhos)}


# --------------------------------------------------------------------------- #
# Localization: top-p% mask IoU and pointing game
# --------------------------------------------------------------------------- #
def upsample_map(attr: torch.Tensor, shape) -> np.ndarray:
    a = F.interpolate(attr[None, None].float(), size=tuple(shape), mode="bilinear", align_corners=False)
    return a[0, 0].cpu().numpy()


def localization(attr: torch.Tensor, gt_mask: np.ndarray, top_fraction: float = 0.20) -> Dict[str, float]:
    up = upsample_map(attr, gt_mask.shape)
    # exactly the top `top_fraction` of pixels (80th-percentile threshold,
    # ties broken by rank so flat maps cannot inflate the mask)
    flat = up.ravel()
    k = max(1, int(round(top_fraction * flat.size)))
    pm = np.zeros(flat.size, bool)
    pm[np.argpartition(-flat, k - 1)[:k]] = True
    pm = pm.reshape(up.shape)
    inter = np.logical_and(pm, gt_mask).sum()
    union = np.logical_or(pm, gt_mask).sum()
    iy, ix = np.unravel_index(np.argmax(up), up.shape)
    lab, ncomp = ndimage.label(pm)
    return {"iou": float(inter / max(union, 1)), "pointing": float(gt_mask[iy, ix]),
            "mask_area_frac": float(pm.mean()), "n_components": int(ncomp),
            "lesion_area_frac": float(gt_mask.mean())}


# --------------------------------------------------------------------------- #
# Cross-client agreement (GradientSHAP only)
# --------------------------------------------------------------------------- #
def cross_client_agreement(attr_vectors: List[np.ndarray]) -> float:
    """Mean pairwise Spearman of L1-normalised 196-vectors computed for the same
    image with each client's background."""
    vs = [v / max(np.abs(v).sum(), 1e-12) for v in attr_vectors]
    rhos = [spearmanr(a, b).statistic for a, b in itertools.combinations(vs, 2)]
    return float(np.nanmean(rhos)) if rhos else float("nan")


# --------------------------------------------------------------------------- #
# Failure modes (Table 29)
# --------------------------------------------------------------------------- #
def failure_modes(correct: Optional[bool], loc: Dict[str, float]) -> Dict[str, bool]:
    """Operational definitions of Section 4.10.

    NOTE: a top-20% mask covers ~20% of the image by construction, so the
    literal Mode-1 criterion "single component covering < 10% of the image"
    can only be met with heavily tied attribution values. Set
    `compact_area_max` / the mask fraction in the config to match the
    definition actually used in the manuscript."""
    mode1 = (correct is False) and loc["n_components"] == 1 and loc["mask_area_frac"] < 0.10
    ok = True if correct is None else bool(correct)          # BreCaHAD: no label
    mode2 = ok and loc["pointing"] == 0.0
    mode3 = ok and loc["pointing"] == 1.0 and loc["iou"] < 0.20 and loc["lesion_area_frac"] < 0.05
    return {"mode1": bool(mode1), "mode2": bool(mode2), "mode3": bool(mode3)}
