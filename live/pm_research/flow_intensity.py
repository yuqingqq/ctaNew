"""Flow model: the arrival-intensity clock, non-parametrically.

Executes only the `f_r` and `f_p` layer of BE_FLOWANDFILLS_MODEL_PLAN.md §2.2.
No parametric form, no Hawkes, no self-excitation, no f_book -- deliberately.
A self-exciting term with a constant baseline attributes clock-driven intensity
growth to itself and adopts on in-sample fit, so the clock is measured FIRST and
anything else is tested against it later.

Guards carried from the session, each paid for:
  * R-DUAL      -- every intensity reported count- AND notional-weighted, with
                   the 0.02 single-actor class separated and published beside.
  * FOLD        -- trades are single-sided; the pair book is one book. Down-token
                   trades enter the unified frame at 1-p with side flipped.
                   Skipping this halves every estimate.
  * ARRIVALS    -- one `last_trade_price` == one taker-order aggregate. Counting
                   OrderFilled legs would inject a state-dependent multiplicity.
  * DENOMINATOR -- numerator and denominator are computed over the SAME window
                   set. Excluding the micro class changes counts, never exposure.
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
MICRO_SIZE = 0.02          # the single-actor class: 16.3% of events, 0.0145% of notional
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
    """Folded taker arrivals for one window. One row per last_trade_price event."""
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
                "p": fold_price(px, is_down),
                "side": fold_side(side, is_down),
                "size": sz,
                "notional": sz * px,        # USDC actually paid, frame-invariant
                "micro": abs(sz - MICRO_SIZE) < MICRO_TOL,
            })
    return rows


def window_mid_dwell(path: Path, up_id: str,
                     gaps: Sequence[tuple[float, float]]) -> list[float]:
    """Seconds the Up-token mid spent in each p-bin, gap intervals removed."""
    try:
        ws = int(path.name.split(".jsonl")[0].rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return [0.0] * (len(FP_EDGES) - 1)
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
    if not pts:
        return [0.0] * (len(FP_EDGES) - 1)
    pts.sort()
    dwell = [0.0] * (len(FP_EDGES) - 1)
    for i, (t0, mid) in enumerate(pts):
        t1 = pts[i + 1][0] if i + 1 < len(pts) else WINDOW_S
        seg = t1 - t0
        if seg <= 0:
            continue
        lost = sum(overlap(t0, t1, g0, g1) for g0, g1 in gaps)
        dwell[p_bin(mid)] += max(0.0, seg - lost)
    return dwell


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
    """Arrival intensity per unified-price bin, with dwell time as exposure.

    Needs the quote stream for dwell, which is ~97% of message volume, so this
    runs on a deterministic subsample of windows per coin. Scope is reported.
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
    res: dict[str, Any] = {"edges": FP_EDGES, "per_coin_target": per_coin,
                           "coins": {}}
    for coin, slugs in sorted(picked.items()):
        pw_cnt, pw_ex, pw_notl = [], [], []
        n_tr = n_mic = 0
        for slug in slugs:
            up, dn = toks[slug]
            g = gaps.get(slug, [])
            dwell = window_mid_dwell(paths[slug], up, g)
            if sum(dwell) <= 0:
                continue
            rows = window_trades(paths[slug], up, dn)
            cnt = [0.0] * n_bins
            cnt_ex = [0.0] * n_bins
            notl = [0.0] * n_bins
            for r in rows:
                i = p_bin(r["p"])
                cnt[i] += 1.0
                notl[i] += r["notional"]
                if not r["micro"]:
                    cnt_ex[i] += 1.0
            n_tr += len(rows)
            n_mic += sum(1 for r in rows if r["micro"])
            pw_cnt.append((cnt, dwell))
            pw_ex.append((cnt_ex, dwell))
            pw_notl.append((notl, dwell))
        if not pw_cnt:
            continue
        prof = profile_ratio(pw_cnt)
        res["coins"][coin] = {
            "n_windows": len(pw_cnt), "n_trades": n_tr, "n_micro": n_mic,
            "dwell_s": [sum(w[1][i] for w in pw_cnt) for i in range(n_bins)],
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
            print(f"{coin:6s} n_win={d['n_windows']:3d} trades={d['n_trades']:6,} "
                  f"shape={d['count']['shape_ratio']:.2f}")
            print("   dwell_s " + " ".join(f"{x:7.0f}" for x in d["dwell_s"]))
            print("   rate/s  " + " ".join(f"{x:7.3f}" for x in d["count"]["profile"]))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
