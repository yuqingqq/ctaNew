"""Walk-forward cross-validation with embargo and label purging.

Per the Program plan: 5 folds × (train 50d / calibration 10d / test 20d).
Embargo = max(label_horizon, 1 ATR-time-bars) on both ends of test.
Purging removes train/cal labels whose [entry_time, exit_time] window
overlaps the test slice or its embargo zones.

The fold layout is sliding (not expanding) so each fold gets the same
amount of train data — important for stable comparisons across folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class FoldSpec:
    """Date ranges for one walk-forward fold. All ranges are right-exclusive."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp        # exclusive
    cal_start: pd.Timestamp
    cal_end: pd.Timestamp          # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp         # exclusive
    embargo: pd.Timedelta          # applied symmetrically around test
    fold_id: int

    def label(self) -> str:
        return f"fold-{self.fold_id}: train {self.train_start.date()}→{self.train_end.date()}, cal {self.cal_start.date()}→{self.cal_end.date()}, test {self.test_start.date()}→{self.test_end.date()}"


def make_walk_forward_folds(
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    *,
    n_folds: int = 5,
    train_days: int = 50,
    cal_days: int = 10,
    test_days: int = 20,
    embargo_days: float = 1.0,
) -> list[FoldSpec]:
    """Generate `n_folds` non-overlapping walk-forward folds covering the
    available data.

    Each fold spans `train_days + cal_days + test_days = 80d` by default.
    Folds are placed end-to-end starting from `data_start`. The function
    raises if the requested span exceeds the available data.
    """
    fold_span = train_days + cal_days + test_days
    total_needed = pd.Timedelta(days=fold_span * n_folds)
    avail = data_end - data_start
    if avail < total_needed:
        raise ValueError(
            f"insufficient data: have {avail.days}d, need ≥ {total_needed.days}d "
            f"for {n_folds} folds × {fold_span}d each"
        )

    folds = []
    cursor = data_start
    embargo = pd.Timedelta(days=embargo_days)
    for fid in range(n_folds):
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days)
        cal_start = train_end
        cal_end = cal_start + pd.Timedelta(days=cal_days)
        test_start = cal_end + embargo  # left-side embargo
        test_end = test_start + pd.Timedelta(days=test_days)
        folds.append(FoldSpec(
            train_start=train_start, train_end=train_end,
            cal_start=cal_start, cal_end=cal_end,
            test_start=test_start, test_end=test_end,
            embargo=embargo, fold_id=fid,
        ))
        # Next fold starts after this fold's test slice + right-side embargo.
        cursor = test_end + embargo
    return folds


def split_features_by_fold(
    feats: pd.DataFrame,
    labels: pd.DataFrame,
    fold: FoldSpec,
    *,
    label_exit_col: str = "exit_time",
) -> dict[str, pd.DataFrame]:
    """Slice features + labels into train/cal/test for one fold, applying
    label purging.

    Purging rules:
      - `train_set`: keep entries whose label window does not overlap the
        test slice or its embargo zones.
      - `cal_set`:   same purge rule against test.
      - `test_set`:  no further restriction.

    Parameters
    ----------
    feats : DataFrame indexed by `open_time`. Must contain the candidate entry
        timestamps (one row per bar).
    labels : DataFrame indexed by `entry_time` with at least column `exit_time`.
        Used both to filter feats to labeled entries and to apply purging.
    fold : the FoldSpec produced by `make_walk_forward_folds`.

    Returns
    -------
    dict with keys train, cal, test → each a DataFrame of features at labeled
        entry timestamps, plus the merged label columns.
    """
    if not feats.index.is_monotonic_increasing:
        feats = feats.sort_index()

    test_left = fold.test_start - fold.embargo
    test_right = fold.test_end + fold.embargo

    # Labeled entries only — drop any row in feats without a label.
    joined = feats.join(labels, how="inner")

    def _slice(start: pd.Timestamp, end: pd.Timestamp, *, purge_test_overlap: bool) -> pd.DataFrame:
        m = (joined.index >= start) & (joined.index < end)
        sub = joined.loc[m]
        if purge_test_overlap and label_exit_col in sub.columns:
            # Drop entries whose label window crosses into [test_left, test_right).
            overlap = (sub[label_exit_col] >= test_left) & (sub.index < test_right)
            sub = sub.loc[~overlap]
        return sub

    return {
        "train": _slice(fold.train_start, fold.train_end, purge_test_overlap=True),
        "cal":   _slice(fold.cal_start, fold.cal_end, purge_test_overlap=True),
        "test":  _slice(fold.test_start, fold.test_end, purge_test_overlap=False),
    }


def fold_iter(
    feats: pd.DataFrame,
    labels: pd.DataFrame,
    folds: list[FoldSpec],
) -> Iterator[tuple[FoldSpec, dict[str, pd.DataFrame]]]:
    """Yield (fold, splits) tuples; convenience for training loops."""
    for f in folds:
        yield f, split_features_by_fold(feats, labels, f)
