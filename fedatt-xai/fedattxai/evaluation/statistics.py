"""Paired comparisons (Section 4.7, Table 16): mean paired macro-F1 difference,
95% BCa bootstrap CI over 10,000 resamples of the seed-level differences, and
the matched-pairs rank-biserial correlation."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.stats import bootstrap, rankdata


def matched_pairs_rank_biserial(diffs: Sequence[float]) -> float:
    d = np.asarray(diffs, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired_comparison(a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000,
                      seed: int = 0) -> Dict[str, float]:
    """a, b: per-seed macro-F1 of the proposed model and a comparator on identical partitions."""
    d = np.asarray(a, float) - np.asarray(b, float)
    out = {"mean_diff": float(d.mean()), "diffs": d.tolist(),
           "rank_biserial": matched_pairs_rank_biserial(d)}
    if np.allclose(d, d[0]):
        out["ci_low"] = out["ci_high"] = float(d[0])
        return out
    res = bootstrap((d,), np.mean, n_resamples=n_resamples, confidence_level=0.95,
                    method="BCa", random_state=np.random.default_rng(seed))
    out["ci_low"], out["ci_high"] = float(res.confidence_interval.low), float(res.confidence_interval.high)
    return out


def mean_std(values: Sequence[float]) -> str:
    v = np.asarray(values, float)
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v.mean():.1f}"
