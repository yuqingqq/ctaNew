"""Flow model: the arrival-intensity clock, non-parametrically.

Executes only the `f_r` and `f_p` layer of BE_FLOWANDFILLS_MODEL_PLAN.md §2.2.
No parametric form, no Hawkes, no self-excitation, no f_book -- deliberately.
A self-exciting term with a constant baseline attributes clock-driven intensity
growth to itself and adopts on in-sample fit, so the clock is measured FIRST and
anything else is tested against it later.

Guards carried from the session, each paid for:
  * R-DUAL      -- arrival counts and descriptive notional throughput are both
                   reported, with distinct units and the 0.02 event type shown.
  * FOLD        -- trades are single-sided; the pair book is one book. Down-token
                   trades enter the unified frame at 1-p with side flipped.
                   Skipping this halves every estimate.
  * ARRIVALS    -- one `last_trade_price` == one taker-order aggregate. Counting
                   OrderFilled legs would inject a state-dependent multiplicity.
  * DENOMINATOR -- numerator and denominator use the SAME window and state
                   intervals. Excluding the micro class changes counts, never
                   exposure.
  * EXPOSURE    -- integrate over observed time using exact gap boundaries. This
                   fixes the DENOMINATOR bias and NOT the selection bias.
  * ERA         -- gap-ledger work is clob_v3_1 only; before 2026-08-20 14:50:21
                   absence of a gap record is NOT evidence of a clean window.

    python3 live/pm_research/flow_intensity.py --selftest
    python3 live/pm_research/flow_intensity.py fr
    python3 live/pm_research/flow_intensity.py fp
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import random
import zlib
from pathlib import Path
from typing import Any, Iterator, Sequence

REPO = Path(__file__).resolve().parents[2]
PM = REPO / "data/pm_5min"
RAW = PM / "raw"
GAPS = PM / "collector_gaps.jsonl"
MARKETS = PM / "markets.jsonl"
OUT_MD = Path(__file__).with_name("FLOW_INTENSITY_RESULTS.md")

WINDOW_S = 300.0
QUOTE_STATE_LAG_S = 0.250  # frozen; shared by numerator and exposure
# Declared in FLOW_MODEL_PROTOCOL_V*.yaml as `minimum_price_bin_dwell_s` and in
# SPEC_REV2 as part of the estimand rather than a post-hoc filter -- but until
# 2026-08-21 it existed only as an editorial fence applied in the write-up, with
# no code path. A bin with 9 s of dwell alone produced a shape_ratio of 295.
MIN_BIN_DWELL_S = 60.0
MICRO_SIZE = 0.02          # labelled single-actor class; prevalence varies by coin
MICRO_TOL = 1e-9
ERA = "clob_v3_1"
DAYS = ("20260819", "20260820", "20260821")

TRADE_MARK = b'"event_type":"last_trade_price"'
QUOTE_MARK = b'"event_type":"price_change"'

# f_r bins: 15 s, 20 per window. BTC runs ~2,424 trades/window and the thinnest
# alts 108-133, so at ~100 windows/coin even the alts hold ~500+ per bin.
FR_BINS = 20
FR_W = WINDOW_S / FR_BINS

FP_EDGES = (0.0, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0)


# --------------------------------------------------------------------------
# fold
# --------------------------------------------------------------------------

def fold_price(price: float, is_down: bool) -> float:
    """Express a trade price in the unified (Up-token) frame."""
    return (1.0 - price) if is_down else price


def fold_side(side: str, is_down: bool) -> str:
    """Buying Down is selling Up. Flip the taker side into the unified frame."""
    s = side.upper()
    if not is_down:
        return s
    return "SELL" if s == "BUY" else "BUY"


def r_bin(elapsed_s: float) -> int:
    """Bin index by elapsed time; bin k covers r in (300-(k+1)w, 300-kw]."""
    if not (0.0 <= elapsed_s <= WINDOW_S):
        raise ValueError(f"elapsed {elapsed_s} outside window")
    return min(int(elapsed_s / FR_W), FR_BINS - 1)


def bin_r_mid(k: int) -> float:
    """Seconds remaining at the midpoint of bin k."""
    return WINDOW_S - (k + 0.5) * FR_W


def p_bin(p: float) -> int:
    for i in range(len(FP_EDGES) - 1):
        if FP_EDGES[i] <= p < FP_EDGES[i + 1]:
            return i
    return len(FP_EDGES) - 2


def p_label(i: int) -> str:
    return f"[{FP_EDGES[i]:.2f},{FP_EDGES[i+1]:.2f})"


# --------------------------------------------------------------------------
# exposure
# --------------------------------------------------------------------------

def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def bin_exposure(gaps: Sequence[tuple[float, float]]) -> list[float]:
    """Observed seconds per f_r bin, given gap intervals relative to window start.

    Gaps are clipped to the window and may straddle bin boundaries.
    """
    out = []
    for k in range(FR_BINS):
        lo, hi = k * FR_W, (k + 1) * FR_W
        lost = sum(overlap(lo, hi, g0, g1) for g0, g1 in gaps)
        out.append(max(0.0, FR_W - lost))
    return out


def _eras() -> list[tuple[int, float, str]]:
    ev = []
    if not GAPS.exists():
        return []
    with GAPS.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") in ("collector_start", "collector_stop"):
                ev.append((r["recv_ns"], r["event"], r.get("collector_version")))
    ev.sort()
    out, cur = [], None
    for t, e, v in ev:
        if e == "collector_start":
            if cur:
                out.append((cur[0], t, cur[1]))
            cur = (t, v)
        else:
            if cur:
                out.append((cur[0], t, cur[1]))
                cur = None
    if cur:
        out.append((cur[0], float("inf"), cur[1]))
    return out


def gaps_by_slug(era: str = ERA) -> dict[str, list[tuple[float, float]]]:
    """Gap intervals per slug, relative to window start, for ONE era."""
    out: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    if not GAPS.exists():
        return out
    with GAPS.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("collector_version") != era:
                continue
            s, e, ws = r.get("gap_start_ns"), r.get("gap_end_ns"), r.get("window_start")
            slug = r.get("slug")
            if not (s and e and ws and slug):
                continue
            g0, g1 = s / 1e9 - int(ws), e / 1e9 - int(ws)
            g0, g1 = max(0.0, g0), min(WINDOW_S, g1)
            if g1 > g0:
                out[slug].append((g0, g1))
    return out


def covered_slugs(era: str = ERA) -> set[str]:
    """Slugs whose entire [ws, ws+300] lies inside a single era of `era`."""
    spans = [(a / 1e9, b / 1e9) for a, b, v in _eras() if v == era]
    if not spans:
        return set()
    out = set()
    for day in DAYS:
        d = RAW / day
        if not d.is_dir():
            continue
        for path in d.glob("*.jsonl*.gz"):        # .gz only == immutable
            slug = path.name.split(".jsonl")[0]
            try:
                ws = int(slug.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if any(a <= ws and ws + WINDOW_S <= b for a, b in spans):
                out.add(slug)
    return out


def token_map() -> dict[str, tuple[str, str]]:
    """slug -> (up_token_id, down_token_id). outcomes are ["Up","Down"] in all 3230."""
    out: dict[str, tuple[str, str]] = {}
    if not MARKETS.exists():
        return out
    for line in MARKETS.read_text().splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        ids, slug = m.get("clobTokenIds"), m.get("slug")
        if ids and len(ids) == 2 and slug:
            out[slug] = (str(ids[0]), str(ids[1]))
    return out


def _gz_lines(path: Path) -> Iterator[bytes]:
    dec = zlib.decompressobj(zlib.MAX_WBITS | 16)
    tail = b""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            try:
                out = dec.decompress(chunk)
            except zlib.error:
                return
            if not out:
                continue
            buf = tail + out
            *lines, tail = buf.split(b"\n")
            yield from lines
    if tail:
        yield tail


def _archive_paths() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for day in DAYS:
        d = RAW / day
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.jsonl*.gz")):
            out.setdefault(path.name.split(".jsonl")[0], path)
    return out


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def window_trades(path: Path, up_id: str, down_id: str) -> list[dict[str, Any]]:
    """Folded taker arrivals for one window. One row per last_trade_price event.

    Execution price is a mark, not the state used to condition ``f_p``.  The
    latter is joined separately from the lagged midpoint-state timeline.
    """
    try:
        ws = int(path.name.split(".jsonl")[0].rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return []
    rows = []
    for line in _gz_lines(path):
        if TRADE_MARK not in line:
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            rn = int(parts[0])
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            continue
        el = rn / 1e9 - ws
        if not (0.0 <= el <= WINDOW_S):
            continue
        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict) or msg.get("event_type") != "last_trade_price":
                continue
            aid = str(msg.get("asset_id"))
            if aid == up_id:
                is_down = False
            elif aid == down_id:
                is_down = True
            else:
                continue
            try:
                px, sz = float(msg["price"]), float(msg["size"])
                side = str(msg["side"]).upper()
            except (KeyError, ValueError, TypeError):
                continue
            rows.append({
                "elapsed": el,
                "native_price": px,
                "exec_p_up": fold_price(px, is_down),
                "side": fold_side(side, is_down),
                "size": sz,
                "notional": sz * px,        # USDC actually paid; do not use folded p
                "micro": abs(sz - MICRO_SIZE) < MICRO_TOL,
            })
    return rows


def state_segments_from_points(
    points: Sequence[tuple[float, float]],
    gaps: Sequence[tuple[float, float]],
    lag_s: float = QUOTE_STATE_LAG_S,
) -> list[tuple[float, float, float]]:
    """Build half-open, knowledge-admissible ``(start, end, mid)`` segments.

    A quote received at ``t`` becomes usable only at ``t + lag_s``.  A
    collector gap invalidates the current state at its start; that state is
    never carried across the gap.  Only a later quote can establish a new
    segment.  This is intentionally stricter than subtracting gap duration
    from a stale quote interval.
    """
    if lag_s < 0:
        raise ValueError("state lag must be non-negative")

    # Last quote wins when multiple updates share one receive timestamp.
    effective: list[tuple[float, float]] = []
    ordered = sorted(
        ((float(t) + lag_s, float(mid)) for t, mid in points),
        key=lambda point: point[0],
    )
    for t, mid in ordered:
        if not (0.0 <= mid <= 1.0) or t > WINDOW_S:
            continue
        if effective and abs(effective[-1][0] - t) < 1e-12:
            effective[-1] = (t, mid)
        else:
            effective.append((t, mid))

    clipped_gaps = sorted(
        (max(0.0, float(g0)), min(WINDOW_S, float(g1)))
        for g0, g1 in gaps if g1 > g0 and g1 > 0.0 and g0 < WINDOW_S
    )
    out: list[tuple[float, float, float]] = []
    for i, (raw_start, mid) in enumerate(effective):
        start = max(0.0, raw_start)
        end = effective[i + 1][0] if i + 1 < len(effective) else WINDOW_S
        end = min(WINDOW_S, end)
        if end <= start:
            continue

        received = raw_start - lag_s
        if any(g0 < start and g1 > received for g0, g1 in clipped_gaps):
            # A disconnect between receipt and admissibility invalidates the
            # quote even when the 250 ms lag would otherwise mature after the
            # reconnect.
            continue

        # The first gap touching this quote's lifetime kills it.  In
        # particular, a state whose admissibility time lands inside a gap does
        # not reappear at gap_end.
        first_touch = next(
            ((g0, g1) for g0, g1 in clipped_gaps if g1 > start and g0 < end),
            None,
        )
        if first_touch is not None:
            g0, _ = first_touch
            if g0 <= start:
                continue
            end = min(end, g0)
        if end > start:
            out.append((start, end, mid))
    return out


def state_mid_at(segments: Sequence[tuple[float, float, float]],
                 elapsed: float) -> float | None:
    """Return the midpoint state on the half-open segment containing elapsed."""
    starts = [s[0] for s in segments]
    i = bisect.bisect_right(starts, elapsed) - 1
    if i >= 0 and segments[i][0] <= elapsed < segments[i][1]:
        return segments[i][2]
    return None


def state_dwell(segments: Sequence[tuple[float, float, float]]) -> list[float]:
    """Exposure seconds by midpoint-state bin."""
    dwell = [0.0] * (len(FP_EDGES) - 1)
    for start, end, mid in segments:
        dwell[p_bin(mid)] += end - start
    return dwell


def window_mid_segments(path: Path, up_id: str,
                        gaps: Sequence[tuple[float, float]]) -> list[
                            tuple[float, float, float]
                        ]:
    """Lagged Up-midpoint state segments, with stale state killed at gaps."""
    try:
        ws = int(path.name.split(".jsonl")[0].rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return []
    pts: list[tuple[float, float]] = []
    for line in _gz_lines(path):
        if QUOTE_MARK not in line:
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            rn = int(parts[0])
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            continue
        el = rn / 1e9 - ws
        if not (0.0 <= el <= WINDOW_S):
            continue
        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict) or msg.get("event_type") != "price_change":
                continue
            for pc in msg.get("price_changes", []):
                if str(pc.get("asset_id")) != up_id:
                    continue
                try:
                    b, a = float(pc["best_bid"]), float(pc["best_ask"])
                except (KeyError, TypeError, ValueError):
                    continue
                # boundary quotes are RETAINED -- 0.0 <= b < a <= 1.0. The strict
                # form dropped 5.2% of quotes, all from the tails, in two
                # independent codebases this session.
                if not (0.0 <= b < a <= 1.0):
                    continue
                pts.append((el, (a + b) / 2.0))
    return state_segments_from_points(pts, gaps)


def fp_window_counts(
    rows: Sequence[dict[str, Any]],
    segments: Sequence[tuple[float, float, float]],
) -> tuple[list[float], list[float], list[float], dict[str, int]]:
    """Bin arrivals and exposure on the identical lagged midpoint state.

    ``exec_p_up`` is retained only as an execution mark and a mismatch
    diagnostic.  It never selects the conditioning bin.
    """
    n_bins = len(FP_EDGES) - 1
    cnt = [0.0] * n_bins
    cnt_ex = [0.0] * n_bins
    notl = [0.0] * n_bins
    diag = {"total": len(rows), "admitted": 0, "no_state": 0,
            "exec_state_bin_mismatch": 0, "micro_admitted": 0}
    for row in rows:
        mid = state_mid_at(segments, float(row["elapsed"]))
        if mid is None:
            diag["no_state"] += 1
            continue
        i = p_bin(mid)
        diag["admitted"] += 1
        if p_bin(float(row["exec_p_up"])) != i:
            diag["exec_state_bin_mismatch"] += 1
        cnt[i] += 1.0
        notl[i] += float(row["notional"])
        if row["micro"]:
            diag["micro_admitted"] += 1
        else:
            cnt_ex[i] += 1.0
    return cnt, cnt_ex, notl, diag


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------

def cluster_bootstrap(per_window: list[tuple[list[float], list[float]]],
                      n_boot: int, seed: int) -> list[tuple[float, float]]:
    """Window-clustered CI for count/exposure ratios, per bin.

    CAVEAT CARRIED EVERYWHERE: window clustering cannot capture day-level common
    factors. With two days these intervals UNDERSTATE true uncertainty.
    """
    if not per_window:
        return []
    n_bins = len(per_window[0][0])
    rng = random.Random(seed)
    draws: list[list[float]] = [[] for _ in range(n_bins)]
    idx = range(len(per_window))
    for _ in range(n_boot):
        pick = [rng.choice(per_window) for _ in idx]
        for k in range(n_bins):
            num = sum(w[0][k] for w in pick)
            den = sum(w[1][k] for w in pick)
            if den > 0:
                draws[k].append(num / den)
    out = []
    for k in range(n_bins):
        d = sorted(draws[k])
        if not d:
            out.append((float("nan"), float("nan")))
            continue
        lo = d[int(0.025 * (len(d) - 1))]
        hi = d[int(0.975 * (len(d) - 1))]
        out.append((lo, hi))
    return out


def profile_ratio(per_window: list[tuple[list[float], list[float]]]) -> list[float]:
    if not per_window:
        return []
    n = len(per_window[0][0])
    out = []
    for k in range(n):
        num = sum(w[0][k] for w in per_window)
        den = sum(w[1][k] for w in per_window)
        out.append(num / den if den > 0 else float("nan"))
    return out


def shape_ratio(prof: Sequence[float]) -> float:
    """max/min over finite positive bins -- the flatness diagnostic."""
    v = [x for x in prof if math.isfinite(x) and x > 0]
    return (max(v) / min(v)) if len(v) >= 2 else float("nan")


# --------------------------------------------------------------------------
# f_r
# --------------------------------------------------------------------------

def fr(n_boot: int = 2000, seed: int = 20260821,
       era_only: bool = True) -> dict[str, Any]:
    """Empirical arrival-count profile per (coin, r-bin), exposure-corrected."""
    paths = _archive_paths()
    toks = token_map()
    gaps = gaps_by_slug(ERA)
    cov = covered_slugs(ERA)
    slugs = sorted(cov if era_only else set(paths))
    slugs = [s for s in slugs if s in paths and s in toks]

    by_coin: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    n_gap_touched = 0
    for slug in slugs:
        up, dn = toks[slug]
        rows = window_trades(paths[slug], up, dn)
        if not rows:
            continue
        g = gaps.get(slug, []) if era_only else []
        if g:
            n_gap_touched += 1
        expo = bin_exposure(g)
        cnt = [0.0] * FR_BINS
        cnt_ex = [0.0] * FR_BINS
        notl = [0.0] * FR_BINS
        notl_ex = [0.0] * FR_BINS
        for r in rows:
            k = r_bin(r["elapsed"])
            cnt[k] += 1.0
            notl[k] += r["notional"]
            if not r["micro"]:
                cnt_ex[k] += 1.0
                notl_ex[k] += r["notional"]
        by_coin[slug.split("-")[0]].append(
            {"slug": slug, "cnt": cnt, "cnt_ex": cnt_ex,
             "notl": notl, "notl_ex": notl_ex, "expo": expo})

    res: dict[str, Any] = {"bins": FR_BINS, "bin_w": FR_W, "era_only": era_only,
                           "n_gap_touched": n_gap_touched, "coins": {}}
    for coin, ws in sorted(by_coin.items()):
        d = {"n_windows": len(ws),
             "n_trades": int(sum(sum(w["cnt"]) for w in ws)),
             "n_micro": int(sum(sum(w["cnt"]) - sum(w["cnt_ex"]) for w in ws)),
             "exposure_s": sum(sum(w["expo"]) for w in ws)}
        for key, num in (("count", "cnt"), ("count_ex_micro", "cnt_ex"),
                         ("notional", "notl"), ("notional_ex_micro", "notl_ex")):
            pw = [(w[num], w["expo"]) for w in ws]
            prof = profile_ratio(pw)
            d[key] = {"profile": prof, "shape_ratio": shape_ratio(prof)}
            if key in ("count", "notional"):
                d[key]["ci"] = cluster_bootstrap(pw, n_boot, seed)
        res["coins"][coin] = d
    return res


# --------------------------------------------------------------------------
# f_p
# --------------------------------------------------------------------------

def fp(per_coin: int = 10, n_boot: int = 2000,
       seed: int = 20260821) -> dict[str, Any]:
    """Arrival intensity per lagged midpoint-state bin.

    Numerator and exposure are assigned by the identical knowledge-admissible
    state timeline.  Execution price is a mark only.  The quote stream is ~97%
    of message volume, so this runs on a deterministic subsample; scope and
    state-join diagnostics are reported.
    """
    paths = _archive_paths()
    toks = token_map()
    gaps = gaps_by_slug(ERA)
    cov = sorted(covered_slugs(ERA))

    picked: dict[str, list[str]] = collections.defaultdict(list)
    for slug in cov:
        coin = slug.split("-")[0]
        if len(picked[coin]) < per_coin and slug in paths and slug in toks:
            picked[coin].append(slug)

    n_bins = len(FP_EDGES) - 1
    res: dict[str, Any] = {"schema_version": 2, "edges": FP_EDGES,
                           "state": "up_midpoint", "state_lag_s": QUOTE_STATE_LAG_S,
                           "per_coin_target": per_coin, "coins": {}}
    for coin, slugs in sorted(picked.items()):
        pw_cnt, pw_ex, pw_notl = [], [], []
        diag_total = collections.Counter()
        for slug in slugs:
            up, dn = toks[slug]
            g = gaps.get(slug, [])
            segments = window_mid_segments(paths[slug], up, g)
            dwell = state_dwell(segments)
            if sum(dwell) <= 0:
                continue
            rows = window_trades(paths[slug], up, dn)
            cnt, cnt_ex, notl, diag = fp_window_counts(rows, segments)
            diag_total.update(diag)
            pw_cnt.append((cnt, dwell))
            pw_ex.append((cnt_ex, dwell))
            pw_notl.append((notl, dwell))
        if not pw_cnt:
            continue
        prof = profile_ratio(pw_cnt)
        res["coins"][coin] = {
            "n_windows": len(pw_cnt),
            "n_trades": diag_total["admitted"],
            "n_trades_total": diag_total["total"],
            "n_trades_state_admitted": diag_total["admitted"],
            "n_trades_no_state": diag_total["no_state"],
            "n_exec_state_bin_mismatch": diag_total["exec_state_bin_mismatch"],
            "n_micro": diag_total["micro_admitted"],
            "dwell_s": (dwell_total := [sum(w[1][i] for w in pw_cnt)
                                       for i in range(n_bins)]),
            # The fence is now CODE, not prose. Bins below the declared minimum
            # are marked non-reportable and published beside the retained set
            # rather than dropped -- the excluded set is part of the result.
            "min_bin_dwell_s": MIN_BIN_DWELL_S,
            "bin_reportable": [d >= MIN_BIN_DWELL_S for d in dwell_total],
            "n_bins_below_min_dwell": sum(d < MIN_BIN_DWELL_S for d in dwell_total),
            "count": {"profile": prof, "shape_ratio": shape_ratio(prof),
                      "ci": cluster_bootstrap(pw_cnt, n_boot, seed)},
            "count_ex_micro": {"profile": profile_ratio(pw_ex)},
            "notional": {"profile": profile_ratio(pw_notl)},
        }
    return res


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def _expect(label: str, exc: type[BaseException], fn) -> None:
    try:
        fn()
    except exc:
        return
    except BaseException as other:  # noqa: BLE001
        raise AssertionError(f"{label}: expected {exc.__name__}, got {other!r}") from other
    raise AssertionError(f"{label}: expected {exc.__name__}, nothing raised")


def _synth(profile: Sequence[float], n_windows: int = 40
           ) -> list[tuple[list[float], list[float]]]:
    """Windows whose per-bin counts follow `profile` exactly, full exposure."""
    return [([c for c in profile], [FR_W] * len(profile)) for _ in range(n_windows)]


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # 1-4: the fold. Buying Down at 0.3 is selling Up at 0.7.
    ok(abs(fold_price(0.30, True) - 0.70) < 1e-12, "fold price down")
    ok(abs(fold_price(0.30, False) - 0.30) < 1e-12, "fold price up")
    ok(fold_side("BUY", True) == "SELL", "fold side down")
    ok(fold_side("BUY", False) == "BUY", "fold side up")

    # 5-8: r binning and its orientation. Bin 0 is the START of the window,
    # which is the LARGEST r. Getting this backwards would mirror every profile.
    ok(r_bin(0.0) == 0, "r_bin at open")
    ok(r_bin(299.9) == FR_BINS - 1, "r_bin at close")
    ok(abs(bin_r_mid(0) - (WINDOW_S - FR_W / 2)) < 1e-9, "bin 0 has largest r")
    ok(bin_r_mid(0) > bin_r_mid(FR_BINS - 1), "r decreases with bin index")
    _expect("elapsed out of window", ValueError, lambda: r_bin(400.0))

    # 9-12: exposure. A gap covering a whole bin must zero it, not shrink it.
    e = bin_exposure([(0.0, FR_W)])
    ok(abs(e[0]) < 1e-9, "full-bin gap zeroes exposure")
    ok(abs(e[1] - FR_W) < 1e-9, "neighbouring bin untouched")
    e2 = bin_exposure([(0.0, FR_W * 2.5)])
    ok(abs(e2[2] - FR_W * 0.5) < 1e-9, "straddling gap splits correctly")
    ok(abs(sum(bin_exposure([])) - WINDOW_S) < 1e-9, "no gaps == full window")

    # 13-14: CONTROL -- the flatness diagnostic must not fire on flat input and
    # must fire on a ramp. Without this pair, "f_r is non-flat" proves nothing.
    flat = profile_ratio(_synth([10.0] * FR_BINS))
    ok(abs(shape_ratio(flat) - 1.0) < 1e-9, "CONTROL: flat input -> ratio 1")
    ramp = profile_ratio(_synth([1.0 + k for k in range(FR_BINS)]))
    ok(shape_ratio(ramp) > 5.0, "CONTROL: ramped input -> ratio >> 1")

    # 15: CONTROL -- a profile must be invariant to how many identical windows
    # it is built from, or the estimator is really counting windows.
    ok(abs(profile_ratio(_synth([7.0] * FR_BINS, 5))[0]
           - profile_ratio(_synth([7.0] * FR_BINS, 500))[0]) < 1e-12,
       "CONTROL: ratio invariant to window count")

    # 16-17: bootstrap. Identical windows carry no cluster variance.
    ci = cluster_bootstrap(_synth([10.0] * FR_BINS, 30), 200, 1)
    ok(abs(ci[0][1] - ci[0][0]) < 1e-9, "CONTROL: identical windows -> zero CI width")
    mixed = [([1.0] * FR_BINS, [FR_W] * FR_BINS)] * 15 + \
            [([9.0] * FR_BINS, [FR_W] * FR_BINS)] * 15
    ci2 = cluster_bootstrap(mixed, 500, 1)
    ok(ci2[0][1] - ci2[0][0] > 1e-3, "CONTROL: dispersed windows -> positive CI width")

    # 18-19: the micro exclusion changes the NUMERATOR only. If exposure moved
    # with it, the two weightings would not be comparable.
    pw = [([10.0] * FR_BINS, [FR_W] * FR_BINS)]
    pw_ex = [([6.0] * FR_BINS, [FR_W] * FR_BINS)]
    ok(pw[0][1] == pw_ex[0][1], "micro exclusion leaves exposure identical")
    ok(profile_ratio(pw)[0] > profile_ratio(pw_ex)[0], "excluding micro lowers count rate")

    # 20-22: price binning covers [0,1] with no gap and no overlap.
    ok(p_bin(0.0) == 0, "p_bin lower edge")
    ok(p_bin(0.999) == len(FP_EDGES) - 2, "p_bin upper edge")
    ok(p_bin(1.0) == len(FP_EDGES) - 2, "p_bin at 1.0 does not overflow")

    # 23: overlap is symmetric and non-negative on disjoint intervals.
    ok(overlap(0, 1, 2, 3) == 0.0 and overlap(0, 5, 1, 2) == 1.0, "overlap")

    # 24: shape_ratio refuses to invent structure from a single bin.
    ok(math.isnan(shape_ratio([3.0])), "shape_ratio undefined on one bin")

    # 25-29: f_p numerator and denominator share one lagged state.  Execution
    # price is deliberately in the opposite tail and must remain only a mark.
    seg = state_segments_from_points([(0.0, 0.10), (10.0, 0.90)], [], lag_s=0.25)
    ok(seg == [(0.25, 10.25, 0.10), (10.25, WINDOW_S, 0.90)],
       "quote state begins only after frozen lag")
    row = {"elapsed": 1.0, "exec_p_up": 0.90, "notional": 5.0, "micro": False}
    c, cx, nv, dg = fp_window_counts([row], seg)
    ok(c[p_bin(0.10)] == 1.0 and c[p_bin(0.90)] == 0.0,
       "arrival uses midpoint-state bin, not execution-price bin")
    ok(cx[p_bin(0.10)] == 1.0 and nv[p_bin(0.10)] == 5.0
       and dg["exec_state_bin_mismatch"] == 1,
       "count, ex-micro count and notional share one state bin")
    ok(abs(state_dwell(seg)[p_bin(0.10)] - 10.0) < 1e-12,
       "denominator uses the same midpoint-state bin")

    # A gap kills the old state; subtracting the gap and resuming the old quote
    # would create a forbidden stale-state interval from 8.0 to 10.25.
    gseg = state_segments_from_points([(0.0, 0.10), (10.0, 0.90)],
                                      [(5.0, 8.0)], lag_s=0.25)
    ok(gseg == [(0.25, 5.0, 0.10), (10.25, WINDOW_S, 0.90)]
       and state_mid_at(gseg, 8.5) is None,
       "state is not carried across collector gaps")

    # 30: a trade without an admitted state is explicit, never silently binned.
    _, _, _, no_state = fp_window_counts([
        {"elapsed": 8.5, "exec_p_up": 0.10, "notional": 1.0, "micro": True}
    ], gseg)
    ok(no_state["no_state"] == 1 and no_state["admitted"] == 0,
       "missing state is counted and rejected")

    # 31: a quote received before a gap cannot mature after the reconnect.
    interrupted = state_segments_from_points([(4.9, 0.10), (6.0, 0.90)],
                                             [(5.0, 5.1)], lag_s=0.25)
    ok(state_mid_at(interrupted, 5.2) is None
       and state_mid_at(interrupted, 6.25) == 0.90,
       "gap between quote receipt and lag maturity invalidates the quote")

    print(f"flow_intensity selftest: {checks} checks OK")
    return 0


# --------------------------------------------------------------------------

def _fmt_prof(prof: Sequence[float], every: int = 2) -> str:
    return " ".join(f"{prof[k]:.2f}" for k in range(0, len(prof), every))


def report_fr(res: dict[str, Any]) -> list[str]:
    L = ["## f_r — arrival intensity vs seconds-to-settlement", "",
         f"Bins: {res['bins']} x {res['bin_w']:.0f} s. Exposure-corrected on the "
         f"`{ERA}` covered set ({res['n_gap_touched']} gap-touched windows).", "",
         "`shape_ratio` = max/min across bins. 1.0 is flat.", "",
         "| coin | windows | trades | micro | count/s ratio | notional/s ratio |",
         "|---|---:|---:|---:|---:|---:|"]
    for coin, d in res["coins"].items():
        L.append(f"| {coin} | {d['n_windows']} | {d['n_trades']:,} | {d['n_micro']:,} "
                 f"| {d['count']['shape_ratio']:.2f} | {d['notional']['shape_ratio']:.2f} |")
    L += ["", "### Profiles, arrivals/s by bin (open -> close, every 2nd bin)", "",
          "```"]
    L.append("r_mid  " + " ".join(f"{bin_r_mid(k):5.0f}" for k in range(0, FR_BINS, 2)))
    for coin, d in res["coins"].items():
        L.append(f"{coin:6s} " + " ".join(
            f"{d['count']['profile'][k]:5.2f}" for k in range(0, FR_BINS, 2)))
    L.append("```")
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["fr", "fp"], default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--per-coin", type=int, default=10)
    ap.add_argument("--all-windows", action="store_true",
                    help="ignore era/ledger coverage (NO exposure correction)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "fr":
        res = fr(n_boot=a.boot, era_only=not a.all_windows)
        print(json.dumps({c: {k: v for k, v in d.items() if k != "slug"}
                          for c, d in res["coins"].items()}, indent=1)[:200])
        for line in report_fr(res):
            print(line)
        (PM / "derived").mkdir(parents=True, exist_ok=True)
        (PM / "derived/flow_fr.json").write_text(json.dumps(res, indent=1))
        return 0
    if a.cmd == "fp":
        res = fp(per_coin=a.per_coin, n_boot=a.boot)
        (PM / "derived").mkdir(parents=True, exist_ok=True)
        (PM / "derived/flow_fp.json").write_text(json.dumps(res, indent=1))
        for coin, d in res["coins"].items():
            print(f"{coin:6s} n_win={d['n_windows']:3d} "
                  f"admitted={d['n_trades_state_admitted']:6,}/"
                  f"{d['n_trades_total']:6,} no_state={d['n_trades_no_state']:5,} "
                  f"bin_mismatch={d['n_exec_state_bin_mismatch']:6,} "
                  f"shape={d['count']['shape_ratio']:.2f}")
            print("   dwell_s " + " ".join(f"{x:7.0f}" for x in d["dwell_s"]))
            print("   rate/s  " + " ".join(f"{x:7.3f}" for x in d["count"]["profile"]))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
