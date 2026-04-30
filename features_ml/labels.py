"""Triple-barrier labeler (López de Prado).

For each entry timestamp t, define three barriers:
    upper = entry_price * (1 + k_up * atr_pct[t])
    lower = entry_price * (1 - k_down * atr_pct[t])
    vertical = t + max_horizon bars

The first barrier touched by the price path determines the label:
    +1 → upper hit (TP for long, SL for short)
    -1 → lower hit (SL for long, TP for short)
     0 → vertical / timeout

Outputs a DataFrame with entry_time, exit_time, exit_reason, label, return_pct,
holding_bars. Used both for training labels and as the cost-model target horizon.

Two evaluation modes:
    - close-only: each bar contributes only its close price to the path. Matches
      the live-bot config (`--close-only`). Default.
    - intrabar:   each bar's high and low can both touch barriers. The order
      within the bar is unknown; if both barriers would be hit, we conservatively
      label as the side adverse to the position (SL for both directions). This
      matches what López de Prado calls the "vertical-barrier-first" tie-break
      rule applied conservatively.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class ExitReason(str, Enum):
    TP = "tp"
    SL = "sl"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class TripleBarrierConfig:
    k_up: float = 1.5            # upper barrier in ATR-multiples
    k_down: float = 1.0          # lower barrier in ATR-multiples
    max_horizon: int = 36        # vertical barrier in bars
    intrabar: bool = False       # use bar high/low (else close-only)


def triple_barrier_labels(
    klines: pd.DataFrame,
    atr_pct: pd.Series,
    entry_times: pd.DatetimeIndex,
    side: Side = Side.LONG,
    config: TripleBarrierConfig = TripleBarrierConfig(),
) -> pd.DataFrame:
    """Apply the triple-barrier method to a set of entry timestamps.

    Parameters
    ----------
    klines : DataFrame indexed by `open_time` with columns close, high, low.
    atr_pct : Series indexed by `open_time` — ATR/close at each bar (point-in-time).
    entry_times : timestamps to label. Must be a subset of `klines.index`.
    side : long or short.
    config : TripleBarrierConfig.

    Returns
    -------
    DataFrame indexed by `entry_time` with columns:
        exit_time, exit_reason, label (+1/-1/0), return_pct, holding_bars.
    """
    if not isinstance(klines.index, pd.DatetimeIndex):
        raise TypeError("klines must be DatetimeIndex-indexed")
    if not entry_times.isin(klines.index).all():
        raise ValueError("entry_times must be a subset of klines.index")

    # Use positional indexing for the forward scan — much faster than label lookups.
    idx_pos = {ts: i for i, ts in enumerate(klines.index)}
    closes = klines["close"].to_numpy()
    highs = klines["high"].to_numpy() if config.intrabar else closes
    lows = klines["low"].to_numpy() if config.intrabar else closes
    atr_arr = atr_pct.reindex(klines.index).to_numpy()
    n = len(closes)

    rows = []
    for t in entry_times:
        i0 = idx_pos[t]
        entry_price = closes[i0]
        a = atr_arr[i0]
        if not np.isfinite(a) or a <= 0:
            continue  # ATR not yet warmed up; skip
        upper = entry_price * (1.0 + config.k_up * a)
        lower = entry_price * (1.0 - config.k_down * a)
        end = min(i0 + config.max_horizon, n - 1)

        exit_pos = end
        exit_reason = ExitReason.TIMEOUT
        # Forward scan from the bar AFTER entry (entry bar can't trigger its own exit).
        for j in range(i0 + 1, end + 1):
            hi, lo = highs[j], lows[j]
            up_hit = hi >= upper
            dn_hit = lo <= lower
            if up_hit and dn_hit:
                # Conservative tie-break: assume the adverse barrier hits first.
                exit_pos = j
                exit_reason = ExitReason.SL if side == Side.LONG else ExitReason.TP
                break
            if up_hit:
                exit_pos = j
                exit_reason = ExitReason.TP if side == Side.LONG else ExitReason.SL
                break
            if dn_hit:
                exit_pos = j
                exit_reason = ExitReason.SL if side == Side.LONG else ExitReason.TP
                break

        exit_price = closes[exit_pos]
        ret = (exit_price - entry_price) / entry_price
        if side == Side.SHORT:
            ret = -ret

        if exit_reason == ExitReason.TP:
            label = 1
        elif exit_reason == ExitReason.SL:
            label = -1
        else:
            label = 0

        rows.append({
            "entry_time": t,
            "exit_time": klines.index[exit_pos],
            "exit_reason": exit_reason.value,
            "label": label,
            "return_pct": ret,
            "holding_bars": exit_pos - i0,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.set_index("entry_time")


def label_summary(labels: pd.DataFrame) -> pd.Series:
    """Quick summary of label distribution and economics."""
    if labels.empty:
        return pd.Series(dtype=float)
    counts = labels["label"].value_counts().to_dict()
    by_reason = labels["exit_reason"].value_counts().to_dict()
    return pd.Series({
        "n": len(labels),
        "frac_tp": counts.get(1, 0) / len(labels),
        "frac_sl": counts.get(-1, 0) / len(labels),
        "frac_timeout": counts.get(0, 0) / len(labels),
        "by_reason_tp": by_reason.get("tp", 0),
        "by_reason_sl": by_reason.get("sl", 0),
        "by_reason_timeout": by_reason.get("timeout", 0),
        "mean_return": labels["return_pct"].mean(),
        "median_return": labels["return_pct"].median(),
        "mean_holding_bars": labels["holding_bars"].mean(),
    })
