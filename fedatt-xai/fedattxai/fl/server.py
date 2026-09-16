"""Phase 1 (federated training), Phase 2 (personalisation) and Phase 4
(evaluation) of Algorithm 1 for every compared method.

Methods
-------
fedavg, fedprox, fedbn (LayerNorm-adapted), moon, fedas, fedatt   external baselines
fedatt_xai    FedAtt aggregation + validation-guided global checkpoint
              selection (line 23) + adaptive personalisation (lines 26-34)
fedas_xai     FedAS aggregation + the same personalisation stage (Table 19)

All methods share the backbone, initialisation, augmentation, client
partitions, rounds, local epochs and full participation (Section 4.3).
Baselines report the global model of the final round; *_xai methods retain the
checkpoint with minimum sample-weighted validation loss.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ..evaluation.metrics import compute_metrics, pooled_task_metrics, predict, youden_threshold
from ..models.backbones import create_model, is_head, is_layernorm_param, state_to_cpu
from ..utils.common import JsonlLogger, save_json, sha256_state_dict
from . import aggregation as agg
from .local_training import (cosine_factor, fedas_parameter_alignment, fisher_information_trace,
                             make_optimizer, train_epochs)
from .personalization import personalize_client

BASE_METHOD = {"fedatt_xai": "fedatt", "fedas_xai": "fedas"}


class FederatedRun:
    def __init__(self, method: str, clients, cfg: dict, seed: int, out_dir: str | Path, device):
        self.method, self.clients, self.cfg, self.seed = method, clients, cfg, seed
        self.base = BASE_METHOD.get(method, method)
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_classes = len(cfg["label_space"])
        self.binary = self.num_classes == 2
        self.logger = JsonlLogger(self.out / "rounds.jsonl")

        torch.manual_seed(seed)
        self.model = create_model(cfg["model"]["backbone"], self.num_classes,
                                  pretrained=cfg["model"].get("pretrained", True)).to(device)
        self.global_state = state_to_cpu(self.model)
        self.head_keys = {k for k in self.global_state if is_head(k, self.model)}

        if self.base == "fedbn":
            self.local_keys = {k for k in self.global_state if is_layernorm_param(k, self.model)}
        elif self.base == "fedas":
            self.local_keys = set(self.head_keys)
        else:
            self.local_keys = set()
        self.local_states = [{k: self.global_state[k].clone() for k in self.local_keys}
                             for _ in clients]
        self.prev_local = ([copy.deepcopy(self.global_state) for _ in clients]
                           if self.base == "moon" else None)

    # ------------------------------------------------------------------ utils
    def client_state(self, i: int, state=None) -> Dict[str, torch.Tensor]:
        s = dict(state if state is not None else self.global_state)
        s.update(self.local_states[i])
        return s

    def _loader(self, c, split, train=False):
        t = self.cfg["train"]
        if train:
            return c.loader("train", t["micro_batch"], seed=self.seed * 100003 + self._round,
                            workers=t["workers"])
        return c.loader(split, t["eval_batch"], shuffle=False, augment=False, workers=t["workers"])

    def federated_val_loss(self, state) -> float:
        """Line 23: each client evaluates the candidate on its own validation
        partition and returns a scalar; server takes the sample-weighted mean."""
        tot, n = 0.0, 0
        for i, c in enumerate(self.clients):
            self.model.load_state_dict(self.client_state(i, state))
            p = predict(self.model, self._loader(c, "val"), self.device)
            tot += p["loss_sum"]
            n += p["n"]
        return tot / max(n, 1)

    # ----------------------------------------------------------------- train
    def train(self) -> Dict:
        cfg, dev = self.cfg, self.device
        R, E = cfg["fl"]["rounds"], cfg["fl"]["local_epochs"]
        select_best = self.method.endswith("_xai") or cfg["fl"].get("force_val_checkpoint", False)
        best = {"loss": float("inf"), "round": -1, "state": None, "local": None}
        scaler = torch.amp.GradScaler("cuda") if (dev.type == "cuda" and cfg["train"].get("amp", True)) else None
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        n = [c.n("train") for c in self.clients]
        global_model = copy.deepcopy(self.model) if self.base in ("fedprox", "moon") else None
        prev_model = copy.deepcopy(self.model) if self.base == "moon" else None

        for t in range(R):
            self._round = t
            lr_scale = cosine_factor(t, R)
            states, fims, losses, up = [], [], [], 0
            t_round = time.time()
            if global_model is not None:
                global_model.load_state_dict(self.global_state)
                global_model.eval()
            for i, c in enumerate(self.clients):                      # "in parallel"
                self.model.load_state_dict(self.client_state(i))        # W_i <- W_g^(t)
                loader = self._loader(c, "train", train=True)
                if self.base == "fedas":
                    fedas_parameter_alignment(self.model, loader, dev, cfg)
                if self.base == "moon":
                    prev_model.load_state_dict(self.prev_local[i])
                    prev_model.eval()
                opt = make_optimizer(self.model, cfg, lr_scale=lr_scale)
                stats = train_epochs(self.model, loader, opt, dev, E, cfg, method=self.base,
                                     global_model=global_model, prev_model=prev_model, scaler=scaler)
                s = state_to_cpu(self.model)
                if self.base == "fedas":
                    fims.append(fisher_information_trace(self.model, loader, dev,
                                                         cfg["methods"]["fedas"].get("fim_batches", 10)))
                if self.local_keys:
                    self.local_states[i] = {k: s[k].clone() for k in self.local_keys}
                if self.prev_local is not None:
                    self.prev_local[i] = s
                states.append(s)                                        # transmit W_i
                losses.append(stats["train_loss"])
                up += agg.uplink_bytes(s, self.local_keys)

            t_agg = time.time()
            alphas = None
            if self.base in ("fedatt",):
                self.global_state, alphas = agg.fedatt(
                    self.global_state, states, n, self.head_keys,
                    temperature=cfg["methods"]["fedatt"]["temperature"],
                    normalize_by_size=cfg["methods"]["fedatt"].get("normalize_by_size", False),
                    local_keys=self.local_keys, return_weights=True)
            elif self.base == "fedas":
                self.global_state = agg.fedas_aggregate(self.global_state, states, fims, self.head_keys)
            else:                                                       # fedavg / fedprox / moon / fedbn
                self.global_state = agg.fedavg(self.global_state, states, n, self.local_keys)
            agg_time = time.time() - t_agg
            del states

            rec = dict(round=t, lr_scale=lr_scale, train_loss=float(np.average(losses, weights=n)),
                       uplink_mb_per_client=up / len(self.clients) / 2 ** 20, agg_time_s=agg_time,
                       round_time_s=time.time() - t_round)
            if select_best or cfg["fl"].get("log_val_every_round", True):
                vloss = self.federated_val_loss(self.global_state)
                rec["val_loss"] = vloss
                if select_best and vloss < best["loss"]:
                    best = {"loss": vloss, "round": t, "state": copy.deepcopy(self.global_state),
                            "local": copy.deepcopy(self.local_states)}
            if alphas is not None and cfg["fl"].get("log_alphas", True):
                rec["alpha_blocks0"] = alphas.get("blocks.0")
            self.logger.log(**rec)

            if t + 1 in cfg["fl"].get("save_rounds", [10]):             # DLG attack checkpoint
                torch.save({"global": self.global_state, "local": self.local_states},
                           self.out / f"round{t + 1}.pt")

        if select_best and best["state"] is not None:
            self.global_state, self.local_states = best["state"], best["local"]
        self.selected_round = best["round"] if select_best else R - 1
        cost = {"wall_clock_h": (time.time() - t_start) / 3600,
                "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None),
                "selected_round": self.selected_round}
        torch.save({"global": self.global_state, "local": self.local_states},
                   self.out / "global_final.pt")
        cost["checkpoint_sha256"] = sha256_state_dict(self.global_state)
        save_json(cost, self.out / "cost.json")
        return cost

    # -------------------------------------------------------------- evaluate
    def _predict_clients(self, states: List[Dict]) -> Dict[int, Dict]:
        res = {}
        for i, c in enumerate(self.clients):
            self.model.load_state_dict(states[i])
            res[i] = {"val": predict(self.model, self._loader(c, "val"), self.device),
                      "test": predict(self.model, self._loader(c, "test"), self.device)}
        return res

    def evaluate_global(self) -> Dict:
        """One validation-selected operating point per method and seed
        (Youden J on the pooled client validation predictions)."""
        states = [self.client_state(i) for i in range(len(self.clients))]
        res = self._predict_clients(states)
        thr = None
        if self.binary:
            yv = np.concatenate([r["val"]["label"] for r in res.values()])
            pv = np.concatenate([r["val"]["prob"][:, 1] for r in res.values()])
            thr = youden_threshold(yv, pv)
        for r in res.values():
            r["threshold"] = thr
        metrics = pooled_task_metrics(res, self.binary)
        per_client = {i: compute_metrics(r["test"]["prob"], r["test"]["label"], thr) for i, r in res.items()
                      if r["test"]["n"] > 0}
        self._dump_predictions(res, "global")
        return {"pooled": metrics, "per_client": per_client}

    def personalize_and_evaluate(self, tau: float | None = None, tag: str = "personalized") -> Dict:
        res, info = {}, {}
        for i, c in enumerate(self.clients):
            self.model.load_state_dict(self.client_state(i))
            p = personalize_client(self.model, c, self.cfg, self.device, self.seed, tau=tau)
            info[i] = {k: v for k, v in p.items() if k != "state"}
            v = predict(self.model, self._loader(c, "val"), self.device)
            te = predict(self.model, self._loader(c, "test"), self.device)
            thr = youden_threshold(v["label"], v["prob"][:, 1]) if self.binary else None   # per client
            res[i] = {"val": v, "test": te, "threshold": thr}
            if self.cfg["fl"].get("save_personalized", False):
                torch.save(p["state"], self.out / f"{tag}_client{i}.pt")
        metrics = pooled_task_metrics(res, self.binary)
        per_client = {i: {**compute_metrics(r["test"]["prob"], r["test"]["label"], r["threshold"]),
                          **{k: info[i][k] for k in ("regime", "n_train", "selected_epochs")}}
                      for i, r in res.items() if r["test"]["n"] > 0}
        self._dump_predictions(res, tag)
        head_only_pct = 100.0 * np.mean([info[i]["regime"] == "head_only" for i in info])
        return {"pooled": metrics, "per_client": per_client, "head_only_pct": head_only_pct,
                "personalization": info}

    def _dump_predictions(self, res, tag):
        """Raw per-client predictions released with the package."""
        rows = []
        for i, r in res.items():
            ids = self.clients[i].frames["test"]["image_id"].tolist()
            for j, iid in enumerate(ids):
                rows.append({"client": i, "image_id": iid, "label": int(r["test"]["label"][j]),
                             "prob": r["test"]["prob"][j].tolist(), "threshold": r.get("threshold")})
        save_json(rows, self.out / f"predictions_{tag}.json")
