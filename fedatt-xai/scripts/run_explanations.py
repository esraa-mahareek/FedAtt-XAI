"""Phase 3 + explanation evaluation (Sections 3.6, 4.5.2, 4.10; Tables 23–29).

In-domain tasks:
  python scripts/run_explanations.py --config configs/inbreast.yaml --seed 0
BreCaHAD transfer (BreaKHis global model, malignant logit, no fine-tuning):
  python scripts/run_explanations.py --config configs/brecahad.yaml --seed 0

All four attribution methods are computed on the SAME global ViT checkpoint
(W_g* of the fedatt_xai run) for each task and seed.
"""
import argparse
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedattxai.data.datasets import build_index  # noqa: E402
from fedattxai.data.preprocessing import load_annotation_mask  # noqa: E402
from fedattxai.models.backbones import create_model  # noqa: E402
from fedattxai.pipeline import get_clients, run_dir  # noqa: E402
from fedattxai.utils.common import (JsonlLogger, get_device, load_config,  # noqa: E402
                                    parse_overrides, save_json, set_seed)
from fedattxai.xai import attributions as A  # noqa: E402
from fedattxai.xai import evaluation as X  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--method", default="fedatt_xai", help="run whose global checkpoint is explained")
ap.add_argument("--no_sweeps", action="store_true", help="skip Tables 25–27 sweeps")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()

cfg = load_config(a.config, overrides=parse_overrides(a.set))
xc = cfg["xai"]
dev = get_device()
set_seed(a.seed)
transfer = cfg["task"] == "brecahad"
src_cfg = load_config(cfg["source_task_config"], overrides=parse_overrides(a.set)) if transfer else cfg
src_run = run_dir(src_cfg, a.method, a.seed)
out = run_dir(cfg, "explanations", a.seed)
out.mkdir(parents=True, exist_ok=True)
log = JsonlLogger(out / "per_image.jsonl", verbose=False)

part, clients = get_clients(src_cfg, a.seed)
model = create_model(src_cfg["model"]["backbone"], len(src_cfg["label_space"]), pretrained=False).to(dev)
ck = torch.load(src_run / "global_final.pt")
model.load_state_dict(ck["global"])
model.eval()
for p in model.parameters():
    p.requires_grad_(True)

# ---------------------------------------------------------------- per client
train_sets = {c.cid: c.dataset("train", augment=False) for c in clients}
backgrounds = {cid: A.draw_background(ds, xc["background_size"], seed=a.seed * 1000 + cid).to(dev)
               for cid, ds in train_sets.items()}
means = {cid: X.channel_mean_of(ds) for cid, ds in train_sets.items()}


def attr_fns(cid):
    bg = backgrounds[cid]
    return {
        "raw_attention": lambda x, target=None: A.raw_attention(model, x, target),
        "attention_rollout": lambda x, target=None: A.attention_rollout(model, x, target),
        "vit_gradcam": lambda x, target=None: A.vit_gradcam(model, x, target),
        "gradient_shap": lambda x, target=None: A.gradient_shap(
            model, x, bg, target, xc["n_samples"], xc["shap_chunk"], seed=a.seed),
    }


# ---------------------------------------------------------------- image list
items = []  # (cid, dataset, index, row)
if transfer:
    src_cid = cfg.get("source_client_for_preprocessing", 0)
    c = clients[src_cid]
    bdf = build_index("brecahad", cfg["dataset"]["root"])
    arr = np.stack([c.pp.to_uint8(p, cfg["preprocessing"]) for p in bdf["path"]])
    from fedattxai.data.loaders import ArrayDataset
    ds = ArrayDataset(arr, np.zeros(len(bdf), int), c.pp)
    items = [(src_cid, ds, i, bdf.iloc[i].to_dict()) for i in range(len(bdf))]
else:
    for c in clients:
        ds = c.dataset("test", augment=False)
        fr = c.frames["test"]
        lim = xc.get("max_test_images_per_client") or len(fr)
        items += [(c.cid, ds, i, fr.iloc[i].to_dict()) for i in range(min(lim, len(fr)))]

summary = defaultdict(list)
sigmas = xc["stability_sigma_sweep"] if not a.no_sweeps else [xc["stability_sigma"]]
baselines = xc["baselines"] if not a.no_sweeps else ["mean"]
fracs = xc["top_fraction_sweep"] if not a.no_sweeps else [xc["top_fraction"]]

for k, (cid, ds, i, row) in enumerate(items):
    x = ds[i][0][None].to(dev)
    with torch.no_grad():
        prob = model(x).softmax(-1)[0]
    pred = int(prob.argmax())
    target = 1 if transfer else pred                      # malignant logit for BreCaHAD
    correct = None if transfer else bool(pred == int(row["label"]))
    mask = load_annotation_mask({**row, "task": cfg["task"]}, cfg["preprocessing"])
    fns = attr_fns(cid)
    bl_objs = {b: X.PerturbationBaseline(b, means[cid], ds.normalize, ds.denormalize) for b in baselines}
    rec = {"client": cid, "image_id": row["image_id"], "pred": pred, "correct": correct}
    for mname, fn in fns.items():
        attr = fn(x, target=target)
        for b, bo in bl_objs.items():
            fi = X.deletion_insertion(model, x, attr, target, bo, xc["deletion_steps"])
            rec[f"{mname}/{b}/deletion_auc"] = fi["deletion_auc"]
            rec[f"{mname}/{b}/insertion_auc"] = fi["insertion_auc"]
        for s in sigmas:
            st = X.stability(model, x, fn, s, ds.normalize, ds.denormalize, xc["stability_copies"],
                             seed=a.seed * 7919 + k)
            rec[f"{mname}/sigma{s}/stability"] = st["stability"]
            rec[f"{mname}/sigma{s}/excluded"] = st["excluded"]
        if mask is not None:
            for fr in fracs:
                loc = X.localization(attr, mask, fr)
                rec[f"{mname}/top{fr}/iou"] = loc["iou"]
                rec[f"{mname}/top{fr}/pointing"] = loc["pointing"]
                if mname == "gradient_shap" and fr == xc["top_fraction"]:
                    for mk, mv in X.failure_modes(correct, loc).items():
                        rec[f"failure/{mk}"] = mv
    log.log(**rec)
    for kk, v in rec.items():
        if "/" in kk and v is not None and not (isinstance(v, float) and np.isnan(v)):
            summary[kk].append(float(v))
    if k % 10 == 0:
        print(f"[{k + 1}/{len(items)}]", flush=True)

# ------------------------------------------------ cross-client agreement (GradientSHAP)
if not transfer:
    rng = np.random.default_rng(a.seed)
    ref = [items[j] for j in rng.choice(len(items), size=min(xc["reference_subset"], len(items)), replace=False)]
    common_bg = A.stratified_background(list(train_sets.values()), xc["background_size"], a.seed).to(dev)
    agree, local_vs_common, iou_local, iou_common = [], [], [], []
    for cid, ds, i, row in ref:
        x = ds[i][0][None].to(dev)
        tgt = A.predicted_class(model, x)
        vecs = [A.gradient_shap(model, x, backgrounds[c.cid], tgt, xc["n_samples"], xc["shap_chunk"],
                                seed=a.seed, return_raw=True).flatten().cpu().numpy() for c in clients]
        agree.append(X.cross_client_agreement(vecs))
        common = A.gradient_shap(model, x, common_bg, tgt, xc["n_samples"], xc["shap_chunk"],
                                 seed=a.seed, return_raw=True).flatten().cpu().numpy()
        local_vs_common.append(np.mean([X.spearmanr(v, common).statistic for v in vecs]))
        mask = load_annotation_mask({**row, "task": cfg["task"]}, cfg["preprocessing"])
        if mask is not None:
            own = torch.from_numpy(vecs[cid].reshape(14, 14))
            iou_local.append(X.localization(A.minmax(own), mask, xc["top_fraction"])["iou"])
            iou_common.append(X.localization(A.minmax(torch.from_numpy(common.reshape(14, 14))), mask,
                                             xc["top_fraction"])["iou"])
    summary["gradient_shap/cross_client_agreement"] = agree
    summary["gradient_shap/local_vs_common_spearman"] = local_vs_common
    if iou_local:
        summary["gradient_shap/iou_client_local_bg"] = iou_local
        summary["gradient_shap/iou_common_bg"] = iou_common

save_json({k: {"mean": float(np.nanmean(v)), "n": len(v)} for k, v in summary.items()}, out / "summary.json")
print("saved", out / "summary.json")
