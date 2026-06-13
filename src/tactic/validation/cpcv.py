"""Combinatorial Purged Cross-Validation (de Prado sec15.1).

Split the ordered trading-day axis into `n_groups` contiguous, roughly-equal groups; hold out
every combination of `k_test` groups as OOS -> C(n_groups, k_test) folds. In each fold the other
groups train. Purge training rows whose forward-label window overlaps a test block, and embargo
`embargo` days after each test block. Every group is OOS in C(n_groups-1, k_test-1) folds, so the
union of per-fold OOS predictions covers the whole date axis (each session OOS that many times).

`cpcv_paths` operates purely on the ordered date axis (positions), so it is data-agnostic; the
training driver maps dates->positions and applies the returned masks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np


@dataclass
class Fold:
    fold_id: int
    test_groups: tuple[int, ...]
    train_pos: np.ndarray        # positions (into the ordered date axis) used for training
    test_pos: np.ndarray         # positions held out as OOS
    test_groups_pos: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class CPCVPlan:
    n_groups: int
    k_test: int
    purge: int
    embargo: int
    n_dates: int
    group_bounds: list[tuple[int, int]]   # [start, end) position bounds per group
    folds: list[Fold]

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def oos_count_per_group(self) -> int:
        return comb(self.n_groups - 1, self.k_test - 1)


def _contiguous_groups(n_dates: int, n_groups: int) -> list[tuple[int, int]]:
    """Split [0, n_dates) into n_groups contiguous near-equal [start, end) blocks."""
    edges = np.linspace(0, n_dates, n_groups + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_groups)]


def cpcv_paths(returns, n_groups: int = 8, k_test: int = 2,
               purge: int = 2, embargo: int = 5) -> CPCVPlan:
    """Build the CPCV fold plan over an ordered date axis.

    `returns` may be an int (number of ordered dates) or any sequence whose length is the number
    of ordered, de-duplicated trading dates. Returns a CPCVPlan with per-fold train/test position
    masks (purge+embargo applied).
    """
    n_dates = int(returns) if np.isscalar(returns) else len(returns)
    if n_groups < k_test or k_test < 1:
        raise ValueError("require n_groups >= k_test >= 1")
    if n_dates < n_groups:
        raise ValueError(f"need >= n_groups dates, got {n_dates} < {n_groups}")

    bounds = _contiguous_groups(n_dates, n_groups)
    group_pos = [np.arange(s, e) for (s, e) in bounds]

    folds: list[Fold] = []
    for fid, test_combo in enumerate(combinations(range(n_groups), k_test)):
        test_pos = np.sort(np.concatenate([group_pos[g] for g in test_combo]))
        test_set = set(test_pos.tolist())

        # candidate train = all positions not in test
        train_mask = np.ones(n_dates, dtype=bool)
        train_mask[test_pos] = False

        # purge: a train decision at p has its forward label over (p, p+purge]; drop it if that
        # window touches the test set (i.e. some position in [p+1, p+purge] is a test position).
        # embargo: drop train within `embargo` positions AFTER any test position.
        for p in range(n_dates):
            if not train_mask[p]:
                continue
            if any((p + h) in test_set for h in range(1, purge + 1)):
                train_mask[p] = False
                continue
            if any((p - h) in test_set for h in range(1, embargo + 1)):
                train_mask[p] = False

        train_pos = np.flatnonzero(train_mask)
        folds.append(Fold(
            fold_id=fid,
            test_groups=test_combo,
            train_pos=train_pos,
            test_pos=test_pos,
            test_groups_pos={g: group_pos[g] for g in test_combo},
        ))

    return CPCVPlan(n_groups=n_groups, k_test=k_test, purge=purge, embargo=embargo,
                    n_dates=n_dates, group_bounds=bounds, folds=folds)


def combined_oos_coverage(plan: CPCVPlan) -> np.ndarray:
    """Count, per date position, in how many folds it is OOS. Should equal
    C(n_groups-1, k_test-1) for every position (full, uniform coverage)."""
    cov = np.zeros(plan.n_dates, dtype=int)
    for f in plan.folds:
        cov[f.test_pos] += 1
    return cov
