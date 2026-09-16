"""Predictive performance and calibration metrics (Section 4.5.1).

* threshold-dependent: accuracy, macro-P/R/F1, sensitivity, specificity —
  computed at a validation-selected Youden-J operating point (binary tasks) or
  argmax (BACH);
* threshold-free: AUROC, PR-AUC (one-vs-rest macro for BACH), ECE (15 equal-
  width bins), Brier score;
* task-level metrics pool the held-out predictions of all clients into ONE
  confusion matrix (per-client metrics are not averaged).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize


@torch.no_grad()
def predict(model, loader, device, amp=True) -> Dict[str, np.ndarray]:
    model.eval()
    probs, labels, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                            enabled=amp):
            logits = model(x)
        logits = logits.float()
        losses.append(F.cross_entropy(logits, y, reduction="none").cpu())
        probs.append(logits.softmax(-1).cpu())
        labels.append(y.cpu())
    if not labels:
        return {"prob": np.zeros((0, 1)), "label": np.zeros(0, int), "loss_sum": 0.0, "n": 0}
    return {"prob": torch.cat(probs).numpy(), "label": torch.cat(labels).numpy(),
            "loss_sum": float(torch.cat(losses).sum()), "n": int(sum(len(l) for l in labels))}


def youden_threshold(y: np.ndarray, p1: np.ndarray) -> float:
    """Operating point maximising J = sensitivity + specificity - 1."""
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p1)
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    return float(np.clip(t, 0.0, 1.0))


def decide(prob: np.ndarray, threshold: Optional[float]) -> np.ndarray:
    if prob.shape[1] == 2 and threshold is not None:
        return (prob[:, 1] >= threshold).astype(int)
    return prob.argmax(1)


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = prob.max(1)
    pred = prob.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(prob: np.ndarray, y: np.ndarray) -> float:
    """Binary: mean (p1 - y)^2. Multi-class: mean over samples of the
    squared error summed over classes."""
    if prob.shape[1] == 2:
        return float(np.mean((prob[:, 1] - y) ** 2))
    onehot = np.eye(prob.shape[1])[y]
    return float(np.mean(np.sum((prob - onehot) ** 2, axis=1)))


def compute_metrics(prob: np.ndarray, y: np.ndarray, threshold: Optional[float] = None,
                    pred: Optional[np.ndarray] = None) -> Dict[str, float]:
    k = prob.shape[1]
    if pred is None:
        pred = decide(prob, threshold)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=list(range(k)),
                                                  average="macro", zero_division=0)
    out = {"accuracy": accuracy_score(y, pred) * 100, "macro_precision": p * 100,
           "macro_recall": r * 100, "macro_f1": f1 * 100,
           "ece": expected_calibration_error(prob, y), "brier": brier_score(prob, y), "n": int(len(y))}
    try:
        if k == 2:
            out["auroc"] = roc_auc_score(y, prob[:, 1]) * 100
            out["pr_auc"] = average_precision_score(y, prob[:, 1]) * 100
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            out["sensitivity"] = tp / max(tp + fn, 1) * 100
            out["specificity"] = tn / max(tn + fp, 1) * 100
        else:
            yb = label_binarize(y, classes=list(range(k)))
            out["auroc"] = roc_auc_score(yb, prob, average="macro", multi_class="ovr") * 100
            out["pr_auc"] = average_precision_score(yb, prob, average="macro") * 100
    except ValueError:  # a class missing from y
        out.setdefault("auroc", float("nan"))
        out.setdefault("pr_auc", float("nan"))
    if threshold is not None:
        out["threshold"] = float(threshold)
    return out


def pooled_task_metrics(per_client: Dict[int, Dict], binary: bool) -> Dict[str, float]:
    """per_client[i] = {"val": predict(...), "test": predict(...), "threshold": t or None}
    Each client's test predictions are thresholded with that client's operating
    point, then all decisions are pooled into one confusion matrix."""
    probs, ys, preds, thrs = [], [], [], []
    for i, d in per_client.items():
        t = d.get("threshold")
        probs.append(d["test"]["prob"])
        ys.append(d["test"]["label"])
        preds.append(decide(d["test"]["prob"], t if binary else None))
        if t is not None:
            thrs.append(t)
    prob, y, pred = np.concatenate(probs), np.concatenate(ys), np.concatenate(preds)
    out = compute_metrics(prob, y, pred=pred)
    if thrs:
        out["threshold"] = float(np.mean(thrs))
    return out
