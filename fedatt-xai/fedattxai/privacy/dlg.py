"""Deep Leakage from Gradients stress test (Zhu et al., 2019; Section 4.9).

Honest-but-curious server observes the gradient of ONE client batch (32 images)
at communication round 10. Dummy images (uniform noise in [0,1]) and dummy soft
labels are optimised with L-BFGS for 300 iterations to match that gradient.
PSNR is computed in the [0,1] image domain after optimal (Hungarian) matching of
reconstructed to true images, since batch order is not identifiable.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def dlg_attack(model, x_true: torch.Tensor, y_true: torch.Tensor, norm, denorm,
               iterations: int = 300, lr: float = 1.0, seed: int = 0) -> Dict[str, float]:
    device = x_true.device
    model.eval()                                   # dropout-free ViT; eval avoids stochastic layers
    params = [p for p in model.parameters() if p.requires_grad]
    loss = F.cross_entropy(model(x_true), y_true)
    true_grads = [g.detach() for g in torch.autograd.grad(loss, params)]

    g = torch.Generator(device="cpu").manual_seed(seed)
    dummy_u = torch.rand(denorm(x_true).shape, generator=g).to(device).requires_grad_(True)  # [0,1]
    k = model.get_classifier().out_features
    dummy_y = torch.randn((x_true.shape[0], k), generator=g).to(device).requires_grad_(True)
    opt = torch.optim.LBFGS([dummy_u, dummy_y], lr=lr, max_iter=1, history_size=100,
                            line_search_fn="strong_wolfe")
    history: List[float] = []

    def closure():
        opt.zero_grad()
        pred = model(norm(dummy_u.clamp(0, 1)))
        dl = torch.sum(-F.softmax(dummy_y, -1) * F.log_softmax(pred, -1), dim=-1).mean()
        dgrads = torch.autograd.grad(dl, params, create_graph=True)
        diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(dgrads, true_grads))
        diff.backward()
        return diff

    for _ in range(iterations):
        history.append(float(opt.step(closure)))

    with torch.no_grad():
        rec = dummy_u.clamp(0, 1)
        tru = denorm(x_true).clamp(0, 1)
        B = rec.shape[0]
        cost = torch.cdist(rec.view(B, -1), tru.view(B, -1)).cpu().numpy()
        r, c = linear_sum_assignment(cost)
        vals = [psnr(rec[i], tru[j]) for i, j in zip(r, c)]
    return {"psnr_mean": float(np.mean(vals)), "psnr_per_image": vals, "grad_match_final": history[-1]}
