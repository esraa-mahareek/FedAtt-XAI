"""Glue shared by the scripts: index -> partition -> client stores."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data.datasets import build_index, summarize_index
from .data.loaders import build_all_clients
from .data.partition import make_partition, partition_report
from .utils.common import save_json


def run_dir(cfg: dict, method: str, seed: int, tag: str = "") -> Path:
    p = cfg["partition"]
    name = f"{method}{('_' + tag) if tag else ''}"
    return Path(cfg["paths"]["results"]) / cfg["task"] / f"beta{p['beta']}_N{p['num_clients']}" / name / f"seed{seed}"


def partition_path(cfg: dict, seed: int) -> Path:
    p = cfg["partition"]
    return Path(cfg["paths"]["results"]) / "partitions" / cfg["task"] / \
        f"beta{p['beta']}_N{p['num_clients']}" / f"seed{seed}.csv"


def load_index(cfg: dict) -> pd.DataFrame:
    kw = {k: v for k, v in cfg["dataset"].items() if k != "root"}
    return build_index(cfg["task"], cfg["dataset"]["root"], **kw)


def get_partition(cfg: dict, seed: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Seed governs Phase 0: each repetition draws a fresh client partition."""
    path = partition_path(cfg, seed)
    if path.exists():
        return pd.read_csv(path)
    df = load_index(cfg) if df is None else df
    p = cfg["partition"]
    part = make_partition(df, p["num_clients"], p["beta"], seed, tuple(p["ratios"]),
                          p.get("min_groups_per_client", 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    part.to_csv(path, index=False)
    partition_report(part).to_csv(path.with_suffix(".report.csv"), index=False)
    save_json({"index": summarize_index(df), "excluded_birads3": df.attrs.get("excluded_birads3")},
              path.with_suffix(".summary.json"))
    return part


def get_clients(cfg: dict, seed: int):
    part = get_partition(cfg, seed)
    return part, build_all_clients(part, cfg, seed)
