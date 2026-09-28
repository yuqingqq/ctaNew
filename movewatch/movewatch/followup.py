"""Forward-test instruments (docs/FORWARD_TEST.md), pure functions so they are unit-tested.

- breadth(): share of liquid coins moving >= 2% (log) the same way over the trigger horizon,
  excluding the alerted asset — the definition used in docs/SIGNAL_FEATURES.md (`mkt_breadth`).
- volume_ratio(): USD volume in minutes [m_a+60, m_a+240) / [m_a-180, m_a) summed over the
  asset's instruments, where m_a is the alert minute — `vol_persist` of
  docs/PERSISTENCE_STUDY.md (hours 1-4 after the alert vs the 3 h up to it), in live minutes.
- labels from the studies' own tercile cutoffs (config `followup` / `breadth`).
- coverage() / gap_minutes(): completeness of the volume windows in the minute ring (a minute
  with no price for an instrument = service down, venue down, or not yet listed there).
- code_fingerprint(): the forward test's freeze identity, recomputed by research/forward_eval.py
  from the freeze commit; every service start logs it.
Nothing here is a validated rule: breadth passed the 4 h study's declared test narrowly;
heavy post-alert volume is an unvalidated candidate. Messages say so.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

BREADTH_MOVE = 0.02   # log move that counts a coin as "moving the same way" (study definition)
GAP_TOL_MIN = 5       # minutes without any price allowed in each 180-min volume window (declared)


def breadth(r_h: np.ndarray, liq: np.ndarray, ok: np.ndarray, direction: int, exclude: np.ndarray,
            liq_min: float = 5e6, min_coins: int = 10) -> float:
    """r_h: log returns of each asset's primary instrument over the trigger horizon."""
    keep = ok & (liq >= liq_min) & np.isfinite(r_h)
    keep[exclude] = False
    if keep.sum() < min_coins:
        return float("nan")
    return float((direction * r_h[keep] >= BREADTH_MOVE).mean())


def breadth_label(b: float, coin_specific_max: float, market_wide_min: float) -> str:
    if b != b:
        return "unknown"
    if b <= coin_specific_max:
        return "coin-specific"
    if b > market_wide_min:
        return "market-wide"
    return "partly shared"


def volume_ratio(min_qv: np.ndarray, M: int, members: np.ndarray, alert_minute: int) -> tuple[float, float, float]:
    """(ratio, after_usd, before_usd) from a minute ring min_qv[n, M] indexed by absolute minute % M.
    after = minutes [m_a+60, m_a+240), before = [m_a-180, m_a)."""
    after_pos = [(alert_minute + k) % M for k in range(60, 240)]
    before_pos = [(alert_minute - k) % M for k in range(1, 181)]
    after = float(min_qv[np.ix_(members, after_pos)].sum(dtype=np.float64))
    before = float(min_qv[np.ix_(members, before_pos)].sum(dtype=np.float64))
    ratio = after / before if after > 0 and before > 0 else float("nan")
    return ratio, after, before


def volume_label(ratio: float, quiet_max: float, heavy_min: float) -> str:
    if ratio != ratio:
        return "unknown"
    if ratio < quiet_max:
        return "quiet"
    if ratio > heavy_min:
        return "heavy"
    return "mid"


def hold(direction: int, px_alert: float, px_now: float) -> float:
    return direction * math.log(px_now / px_alert) if px_alert > 0 and px_now > 0 else float("nan")


def _positions(M: int, m0: int, m1: int) -> list[int]:
    return [m % M for m in range(m0, m1)]


def coverage(min_px: np.ndarray, M: int, ids: np.ndarray, m0: int, m1: int) -> np.ndarray:
    """Per instrument: share of minutes in [m0, m1) with a price."""
    return np.isfinite(min_px[np.ix_(ids, _positions(M, m0, m1))]).mean(axis=1)


def gap_minutes(min_px: np.ndarray, M: int, ids: np.ndarray, m0: int, m1: int) -> int:
    """Minutes in [m0, m1) in which none of `ids` has a price."""
    return int((~np.isfinite(min_px[np.ix_(ids, _positions(M, m0, m1))])).all(axis=0).sum())


def fingerprint(files: list[tuple[str, bytes]], cfg: dict) -> str:
    """sha256 over (relative path, bytes) of the package sources and the loaded config."""
    h = hashlib.sha256()
    for rel, data in sorted(files):
        h.update(rel.encode() + b"\0" + data + b"\0")
    h.update(json.dumps(cfg, sort_keys=True, default=str).encode())
    return h.hexdigest()


def code_fingerprint(cfg: dict, pkg: Path | None = None) -> str:
    pkg = pkg or Path(__file__).resolve().parent
    return fingerprint([(p.relative_to(pkg).as_posix(), p.read_bytes()) for p in pkg.rglob("*.py")], cfg)
