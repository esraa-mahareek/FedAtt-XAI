"""Phase 0 of Algorithm 1 — leakage-controlled client construction.

1. Images are grouped by source unit (patient / case / image). A group is
   never divided.
2. Groups are allocated to clients under a Dirichlet distribution over class
   proportions with concentration beta (label-skew non-IID protocol).
   Group disjointness across clients is asserted at allocation time.
3. Within each client, groups are split 70/10/20 into train/val/test,
   stratified by the group label.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def group_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group with its image count and group label.

    The group label is the majority image label (ties -> the higher class,
    i.e. malignant). For BreaKHis / BACH every group is single-class.
    """
    def _lab(s):
        vc = s.value_counts()
        top = vc[vc == vc.max()].index
        return int(max(top))

    g = df.groupby("group").agg(n_images=("image_id", "size"), label=("label", _lab))
    return g.reset_index()


def dirichlet_group_allocation(groups: pd.DataFrame, num_clients: int, beta: float,
                               rng: np.random.Generator, min_groups: int = 3,
                               max_tries: int = 1000) -> Dict[int, List[str]]:
    """Allocate whole groups to clients.

    For each class c a client-proportion vector q_c ~ Dir(beta * 1_N) is drawn,
    and the groups of class c are assigned greedily to the client whose realised
    image count is furthest below its target q_c * n_c. This is the standard
    Dirichlet label-skew protocol (Hsu et al., 2019) applied at group level, so
    that every client's class composition follows the sampled proportions.
    The draw is repeated until every client holds at least `min_groups` groups
    (needed for a non-empty 70/10/20 split).
    """
    classes = sorted(groups["label"].unique())
    for _ in range(max_tries):
        assign: Dict[int, List[str]] = {i: [] for i in range(num_clients)}
        for c in classes:
            gc = groups[groups["label"] == c].sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
            q = rng.dirichlet(beta * np.ones(num_clients))
            target = q * gc["n_images"].sum()
            realised = np.zeros(num_clients)
            # largest groups first gives a closer match to the target proportions
            for _, row in gc.sort_values("n_images", ascending=False).iterrows():
                i = int(np.argmax(target - realised))
                assign[i].append(row["group"])
                realised[i] += row["n_images"]
        if min(len(v) for v in assign.values()) >= min_groups:
            break
    else:
        raise RuntimeError("Could not satisfy min_groups per client; lower num_clients or raise beta.")

    # ---- Algorithm 1, line 5: assert G_i ∩ G_j = ∅ for all j ≠ i --------------
    seen = set()
    for i, gl in assign.items():
        s = set(gl)
        assert not (s & seen), f"group allocated to more than one client (client {i})"
        seen |= s
    assert seen == set(groups["group"]), "partition is not exhaustive"
    return assign


def stratified_group_split(groups: pd.DataFrame, rng: np.random.Generator,
                           ratios=(0.7, 0.1, 0.2)) -> Dict[str, List[str]]:
    """GroupSplit(G_i, 70/10/20): per class, shuffle groups and fill train/val/test
    by image count. Val and test receive at least one group whenever the client
    holds >= 3 groups."""
    out = {"train": [], "val": [], "test": []}
    for c in sorted(groups["label"].unique()):
        gc = groups[groups["label"] == c]
        order = rng.permutation(len(gc))
        gc = gc.iloc[order]
        total = gc["n_images"].sum()
        cum = 0
        for _, row in gc.iterrows():
            frac = cum / total
            if frac < ratios[0]:
                out["train"].append(row["group"])
            elif frac < ratios[0] + ratios[1]:
                out["val"].append(row["group"])
            else:
                out["test"].append(row["group"])
            cum += row["n_images"]
    # guarantee non-empty val / test (move smallest train groups)
    for split in ("test", "val"):
        if not out[split] and len(out["train"]) > 1:
            tr = groups[groups["group"].isin(out["train"])].sort_values("n_images")
            g = tr["group"].iloc[0]
            out["train"].remove(g)
            out[split].append(g)
    return out


def make_partition(df: pd.DataFrame, num_clients: int, beta: float, seed: int,
                   ratios=(0.7, 0.1, 0.2), min_groups: int = 3) -> pd.DataFrame:
    """Returns df with added columns `client` and `split`."""
    rng = np.random.default_rng(seed)
    gt = group_table(df)
    assign = dirichlet_group_allocation(gt, num_clients, beta, rng, min_groups=min_groups)
    g2client, g2split = {}, {}
    for i, gl in assign.items():
        sub = gt[gt["group"].isin(gl)]
        sp = stratified_group_split(sub, rng, ratios)
        for split, glist in sp.items():
            for g in glist:
                g2client[g] = i
                g2split[g] = split
    out = df.copy()
    out["client"] = out["group"].map(g2client).astype(int)
    out["split"] = out["group"].map(g2split)
    verify_no_leakage(out)
    return out


def verify_no_leakage(part: pd.DataFrame) -> None:
    """Every group appears in exactly one (client, split)."""
    per_group = part.groupby("group")[["client", "split"]].nunique()
    bad = per_group[(per_group["client"] > 1) | (per_group["split"] > 1)]
    if len(bad):
        raise AssertionError(f"Leakage: {len(bad)} groups span clients/splits, e.g. {bad.index[:5].tolist()}")


def partition_report(part: pd.DataFrame) -> pd.DataFrame:
    """Table-5 style report: images (groups) per client/split/class."""
    rep = (part.groupby(["client", "split", "label"])
               .agg(images=("image_id", "size"), groups=("group", "nunique"))
               .reset_index())
    return rep
