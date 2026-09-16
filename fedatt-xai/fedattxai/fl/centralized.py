"""Centralized reference (Section 4.6.1): union of client training partitions,
same backbone / augmentation / optimiser / learning rates; 200 epochs
(= rounds × local epochs); model selected on the pooled validation loss;
preprocessing fitted on the pooled training partition. Upper reference only."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from ..evaluation.metrics import compute_metrics, predict, youden_threshold
from ..models.backbones import create_model, state_to_cpu
from ..utils.common import JsonlLogger, save_json
from .local_training import cosine_factor, make_optimizer, train_epochs


def train_centralized(pooled, cfg, seed, out_dir, device):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = JsonlLogger(out / "epochs.jsonl")
    torch.manual_seed(seed)
    k = len(cfg["label_space"])
    model = create_model(cfg["model"]["backbone"], k, cfg["model"].get("pretrained", True)).to(device)
    epochs = cfg["fl"]["rounds"] * cfg["fl"]["local_epochs"]
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and cfg["train"].get("amp", True)) else None
    t = cfg["train"]
    vl = pooled.loader("val", t["eval_batch"], shuffle=False, augment=False, workers=t["workers"])
    best = {"loss": float("inf"), "state": None, "epoch": -1}
    for e in range(epochs):
        opt = make_optimizer(model, cfg, lr_scale=cosine_factor(e, epochs))
        tl = pooled.loader("train", t["micro_batch"], seed=seed * 100003 + e, workers=t["workers"])
        st = train_epochs(model, tl, opt, device, 1, cfg, scaler=scaler)
        v = predict(model, vl, device)
        vloss = v["loss_sum"] / max(v["n"], 1)
        log.log(epoch=e, train_loss=st["train_loss"], val_loss=vloss)
        if vloss < best["loss"]:
            best = {"loss": vloss, "state": state_to_cpu(model), "epoch": e}
    model.load_state_dict(best["state"])
    v = predict(model, vl, device)
    te = predict(model, pooled.loader("test", t["eval_batch"], shuffle=False, augment=False,
                                      workers=t["workers"]), device)
    thr = youden_threshold(v["label"], v["prob"][:, 1]) if k == 2 else None
    res = {"pooled": compute_metrics(te["prob"], te["label"], thr), "selected_epoch": best["epoch"]}
    save_json(res, out / "metrics_centralized.json")
    return res
