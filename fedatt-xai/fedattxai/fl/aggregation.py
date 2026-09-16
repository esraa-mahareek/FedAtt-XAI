"""Server-side aggregation rules.

* FedAvg   : sample-count weighted average of all transmitted tensors.
* FedAtt   : layer-wise attentive aggregation (Ji et al., 2019) — Eq. (6)-(7):
                 α_i^(ℓ) = softmax_i( -‖W_i^(ℓ) − W_g^(ℓ)‖_F / T )
                 W_g^(t+1,ℓ) = Σ_i α_i^(ℓ) W_i^(t,ℓ)
             The classification head (and any BN buffers) are averaged by
             sample count, as in FedAvg.
* FedAS    : shared encoder weighted by the clients' Fisher-information trace;
             heads are client-specific and not aggregated.
Keys listed in `local_keys` (FedBN-LN: LayerNorm parameters; FedAS: head) are
never transmitted and never aggregated.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Dict, List, Sequence, Set

import torch

from ..models.backbones import layer_key

State = Dict[str, torch.Tensor]


def _weighted(states: Sequence[State], weights: Sequence[float], key: str) -> torch.Tensor:
    ref = states[0][key]
    if not torch.is_floating_point(ref):          # e.g. num_batches_tracked
        return ref.clone()
    out = torch.zeros_like(ref, dtype=torch.float32)
    for s, w in zip(states, weights):
        out += float(w) * s[key].float()
    return out.to(ref.dtype)


def fedavg(global_state: State, states: List[State], n: List[int], local_keys: Set[str] = frozenset()) -> State:
    w = torch.tensor(n, dtype=torch.float64)
    w = (w / w.sum()).tolist()
    new = OrderedDict()
    for k in global_state:
        new[k] = global_state[k].clone() if k in local_keys else _weighted(states, w, k)
    return new


def fedatt(global_state: State, states: List[State], n: List[int], head_keys: Set[str],
           temperature: float, normalize_by_size: bool = False,
           local_keys: Set[str] = frozenset(), return_weights: bool = False):
    """Layer-wise attentive aggregation, Eq. (6)-(7)."""
    N = len(states)
    ws = torch.tensor(n, dtype=torch.float64)
    ws = (ws / ws.sum()).tolist()

    layers: Dict[str, List[str]] = defaultdict(list)
    for k, v in global_state.items():
        if k in local_keys or k in head_keys or not torch.is_floating_point(v):
            continue
        if "running_" in k:                       # BN statistics (ResNet ablation)
            continue
        layers[layer_key(k)].append(k)

    new = OrderedDict()
    alphas = {}
    for lk, keys in layers.items():
        dist = torch.zeros(N, dtype=torch.float64)
        numel = 0
        for k in keys:
            g = global_state[k].double()
            numel += g.numel()
            for i, s in enumerate(states):
                dist[i] += ((s[k].double() - g) ** 2).sum()
        dist = dist.sqrt()                         # Frobenius norm over layer ℓ
        if normalize_by_size:
            dist = dist / (numel ** 0.5)
        alpha = torch.softmax(-dist / temperature, dim=0)
        alphas[lk] = alpha.tolist()
        for k in keys:
            new[k] = _weighted(states, alpha.tolist(), k)

    for k in global_state:
        if k in new:
            continue
        if k in local_keys:
            new[k] = global_state[k].clone()
        else:                                      # head, BN buffers: sample count
            new[k] = _weighted(states, ws, k)
    new = OrderedDict((k, new[k]) for k in global_state)
    return (new, alphas) if return_weights else new


def fedas_aggregate(global_state: State, states: List[State], fim_traces: List[float],
                    head_keys: Set[str]) -> State:
    f = torch.tensor(fim_traces, dtype=torch.float64).clamp_min(1e-12)
    w = (f / f.sum()).tolist()
    new = OrderedDict()
    for k in global_state:
        new[k] = global_state[k].clone() if k in head_keys else _weighted(states, w, k)
    return new


def uplink_bytes(state: State, local_keys: Set[str] = frozenset()) -> int:
    return int(sum(v.numel() * v.element_size() for k, v in state.items() if k not in local_keys))
