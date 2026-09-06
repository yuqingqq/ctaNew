"""P-2026-002 E2.0 — the notional-weighted TRUE-MID recompute of the E1 screen.

Runs ONLY against the committed declaration
`live/mm_research/declarations/p002_e2_0_declaration_v1.json`. It verifies that
file's sha256 against the digest it was built for and REFUSES on a mismatch, so
the gate cannot be redefined after seeing. Every threshold below is READ from
the declaration, never re-typed here.

WHAT IT COMPUTES, per symbol x admissible UTC day, on the collected tape:
  * sweeps      -- prints collapsed on (transact_time, is_buyer_maker), §1.1
  * TRUE mid    -- (bid+ask)/2 from bookTicker at exchange transact_time
  * PROXY mid   -- E1's two-sided last-print mid with 10 s validity, rebuilt on
                   the SAME events, so `Delta_rs` isolates the MID and not the
                   period (the two windows do not intersect -- see the
                   declaration)
  * MO / Lambda / es at every tau, eq- AND notional-weighted, per side, and by
    notional quintile

THE FORMULAS ARE E1's, TRANSCRIBED FROM `e1_markout_scan.py` AND PINNED BY A
REPRODUCTION CONTROL: the runner re-measures ADA on E1's OWN Vision aggTrades
and must land within 0.05 bps of E1's published +2.443 / -0.322 before any
number about the new window is reported. Without that, `Delta_rs` would be a
difference between two codebases rather than between two mids.

    python3 live/mm_research/e2_0_true_mid.py --selftest
    python3 live/mm_research/e2_0_true_mid.py --reproduce-e1
    python3 live/mm_research/e2_0_true_mid.py --run --symbols ADAUSDT --output R
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]

#: WHERE THE LEDGER IS, RESOLVED AND RECORDED -- NOT ASSUMED FROM THE CODE.
#: `PM_DATA_ROOT` names the REPO root (pm_tape_density.py:99 returns Path(env)
#: and consumers append `data/`), and nine modules in this repo already honour
#: it. This family did not, and relied on a `data` symlink in the worktree --
#: which `git checkout --detach` destroys, silently leaving a near-empty shell.
#: That cost a run in round 58: the smoke found zero days and refused. The
#: branch taken is recorded in every receipt so a reader can see WHICH tree a
#: number came from, and a root with no tape REFUSES instead of reading a
#: partial tree.
DATA_ROOT_BRANCH = "unresolved"


def _resolve_root() -> Path:
    global DATA_ROOT_BRANCH
    env = os.environ.get("PM_DATA_ROOT")
    if env:
        DATA_ROOT_BRANCH = "1_env_PM_DATA_ROOT"
        return Path(env)
    if (CODE_ROOT / "data" / "mm_hf" / "raw").is_dir():
        DATA_ROOT_BRANCH = "2_code_tree_carries_the_tape"
        return CODE_ROOT
    DATA_ROOT_BRANCH = "3_unresolved_no_tape"
    return CODE_ROOT


ROOT = _resolve_root()
RAW = ROOT / "data" / "mm_hf" / "raw"
VISION = ROOT / "data" / "mm_hf" / "vision" / "parquet" / "aggTrades"
DECL_PATH = HERE / "declarations" / "p002_e2_0_declaration_v2.json"
DECL_SHA = "054ff4e7536291ee1a4846dec90e62cb3496e851cb27768a898969b67256c1d6"
PROTOCOL = "P002_E2_0_TRUE_MID_V1"
EPS = 1e-12
SEC_PER_DAY = 86_400


class E20Refused(RuntimeError):
    """The run cannot proceed under the declaration."""


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(HERE))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def declaring_module():
    """The thresholds live in ONE place: the declaring module, which the
    declaration was emitted from and which is pinned by the same commit.
    Reading them from the JSON with a `.get(..., default)` would let a missing
    field silently become a threshold -- absence must never read as a value."""
    sys.path.insert(0, str(HERE))
    import e2_0_declare as D                                  # noqa: PLC0415
    return D


def load_declaration() -> dict:
    if not DECL_PATH.is_file():
        raise E20Refused(f"REFUSED: no declaration at {DECL_PATH}")
    got = digest(DECL_PATH)
    if got != DECL_SHA:
        raise E20Refused(
            f"REFUSED: the declaration's sha256 is {got}, not the "
            f"{DECL_SHA} this runner was built against. Either the gate was "
            f"edited after it was declared, or this runner is stale. Neither "
            f"may proceed.")
    d = json.loads(DECL_PATH.read_text())
    D = declaring_module()
    # THE JSON AND THE MODULE MUST AGREE where both carry a number. If they
    # ever disagree, one of them was edited alone and neither can be trusted.
    if d["population"]["min_complete_days"] != D.MIN_COMPLETE_DAYS:
        raise E20Refused(
            f"REFUSED: min_complete_days is "
            f"{d['population']['min_complete_days']} in the declaration and "
            f"{D.MIN_COMPLETE_DAYS} in the declaring module")
    if d["what_settles_and_what_voids"]["leg_i_VOID"]["threshold_bps"] != \
            D.VOID_THRESHOLD_BPS:
        raise E20Refused("REFUSED: the void threshold differs between the "
                         "declaration and the declaring module")
    if d["what_settles_and_what_voids"]["leg_ii_SETTLE"]["threshold_bps"] != \
            D.FEE_MAKER_VIP0:
        raise E20Refused("REFUSED: the settle threshold differs between the "
                         "declaration and the declaring module")
    return d


# --------------------------------------------------------------------------
# EVENTS -- E1 section 1.1, transcribed
# --------------------------------------------------------------------------
def sweeps(t, p, q, m):
    """Collapse prints to sweeps keyed by (transact_time, is_buyer_maker).

    `sign` is q_j, the TAKER's direction: is_buyer_maker means the buyer was
    passive, so the taker SOLD and the maker's BID was filled -> q_j = -1.
    """
    order = np.lexsort((m.astype(np.int8), t))
    t, p, q, m = t[order], p[order], q[order], m[order]
    key_change = np.empty(len(t), bool)
    key_change[0] = True
    key_change[1:] = (t[1:] != t[:-1]) | (m[1:] != m[:-1])
    starts = np.flatnonzero(key_change)
    notq = np.add.reduceat(q, starts)
    pw = np.add.reduceat(p * q, starts) / np.maximum(notq, EPS)
    mk = m[starts]
    return {"t": t[starts], "p": pw, "Q": notq, "mkr_buy": mk,
            "sign": np.where(mk, -1.0, 1.0),
            "n": np.diff(np.append(starts, len(t)))}


class TrueMid:
    """(bid+ask)/2 from bookTicker at exchange transact_time. No validity
    window: a quote stands until the next one. Quote AGE is reported instead,
    because filtering on age would select on activity."""

    def __init__(self, bt_t, bid, ask):
        self.t, self.bid, self.ask = bt_t, bid, ask
        self.mid = (bid + ask) / 2.0

    def at_before(self, u):
        """m(u-): the last quote STRICTLY BEFORE u. Returns (mid, valid, age)."""
        i = np.searchsorted(self.t, u, "left") - 1
        ok = i >= 0
        ic = np.clip(i, 0, None)
        return self.mid[ic], ok, (u - self.t[ic]).astype(np.float64)

    def at(self, u):
        """m(u): the last quote at or before u, valid only if the book we hold
        actually extends to u -- a query past the end is EXCLUDED and counted,
        never valued at the last known quote."""
        i = np.searchsorted(self.t, u, "right") - 1
        ok = (i >= 0) & (u <= self.t[-1])
        ic = np.clip(i, 0, None)
        return self.mid[ic], ok, (u - self.t[ic]).astype(np.float64)


class ProxyMid:
    """E1 section 1.2: two-sided last-print mid, both sides within VALID_MS."""

    def __init__(self, sw, valid_ms):
        buy = ~sw["mkr_buy"]                     # taker-buy sweeps (ask side)
        self.tb, self.pb = sw["t"][buy], sw["p"][buy]
        self.ts, self.ps = sw["t"][~buy], sw["p"][~buy]
        self.valid_ms = valid_ms

    def at(self, u):
        if len(self.tb) == 0 or len(self.ts) == 0:
            z = np.zeros(len(u))
            return z, np.zeros(len(u), bool)
        ib = np.searchsorted(self.tb, u, "left") - 1
        is_ = np.searchsorted(self.ts, u, "left") - 1
        ok = (ib >= 0) & (is_ >= 0)
        ibc, isc = np.clip(ib, 0, None), np.clip(is_, 0, None)
        ok &= (self.tb[ibc] >= u - self.valid_ms) & \
              (self.ts[isc] >= u - self.valid_ms)
        return (self.pb[ibc] + self.ps[isc]) / 2.0, ok


def markout(sign, p, m0, m1):
    """E1 section 1.3b/c, transcribed exactly (note the denominators: MO is
    per unit of the FILL price, es and Lambda per unit of the pre-trade mid)."""
    mo = -sign * (m1 - p) / np.maximum(p, EPS) * 1e4
    es = sign * (p - m0) / np.maximum(m0, EPS) * 1e4
    lam = sign * (m1 - m0) / np.maximum(m0, EPS) * 1e4
    return mo, es, lam


def wmean(x, w, sel):
    if sel.sum() == 0:
        return float("nan")
    ww = w[sel]
    if ww.sum() <= 0:
        return float("nan")
    return float(np.average(x[sel], weights=ww))


# --------------------------------------------------------------------------
# READERS
# --------------------------------------------------------------------------
def _hour_files(stream: str, sym: str, day: str) -> list[Path]:
    d = RAW / stream / sym
    out = []
    for h in range(24):
        for ext in (".csv.gz", ".csv"):
            p = d / f"{day}_{h:02d}{ext}"
            if p.is_file():
                out.append(p)
                break
    return out


def _next_hour_file(stream: str, sym: str, day: str) -> Path | None:
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    for ext in (".csv.gz", ".csv"):
        p = RAW / stream / sym / f"{nxt}_00{ext}"
        if p.is_file():
            return p
    return None


def _read_csv(paths, usecols, names, dtypes):
    """Malformed lines are COUNTED, never silently skipped (rule 4).

    `on_bad_lines="skip"` alone would drop a truncated tail -- the exact shape
    a killed collector leaves -- and report a clean read. The count is carried
    out to the caller and into the receipt.
    """
    frames, n_raw = [], 0
    for p in paths:
        op = gzip.open(p, "rb") if p.suffix == ".gz" else open(p, "rb")
        with op as fh:
            raw = fh.read()
        n_raw += raw.count(b"\n")
        frames.append(pd.read_csv(io.BytesIO(raw), header=None,
                                  usecols=usecols, names=names, dtype=dtypes,
                                  engine="c", on_bad_lines="skip"))
    if not frames:
        return None, 0
    df = pd.concat(frames, ignore_index=True)
    return df, n_raw - len(df)


def read_book(sym: str, day: str, extend: bool = True):
    """bookTicker: recv_ns,E,T,u,bid,bid_qty,ask,ask_qty -> (T, bid, ask).

    One extra hour past midnight is loaded when it exists, so a sweep at 23:59
    can still be valued at tau=300 s. That EXTENDS what is available; it does
    not change the rule that a query past the end of available book is
    excluded."""
    files = _hour_files("bookTicker", sym, day)
    n_day_files = len(files)
    if extend:
        nx = _next_hour_file("bookTicker", sym, day)
        if nx is not None:
            files = files + [nx]
    df, n_unparsed = _read_csv(files, [2, 4, 6], ["T", "bid", "ask"],
                               {2: "int64", 4: "float64", 6: "float64"})
    if df is None or len(df) == 0:
        return None, n_day_files, {"n_unparsed_lines": n_unparsed}
    t = df["T"].to_numpy()
    bid, ask = df["bid"].to_numpy(), df["ask"].to_numpy()
    # A QUOTE MUST BE A QUOTE. Non-positive or crossed levels are not book
    # states; excluded and COUNTED, never carried into a mid.
    good = (bid > 0) & (ask > 0) & (ask >= bid)
    n_bad_quotes = int((~good).sum())
    t, bid, ask = t[good], bid[good], ask[good]
    if len(t) == 0:
        return None, n_day_files, {"n_unparsed_lines": n_unparsed,
                                   "n_nonpositive_or_crossed_quotes": n_bad_quotes}
    o = np.argsort(t, kind="stable")
    return (t[o], bid[o], ask[o]), n_day_files, {
        "n_unparsed_lines": int(n_unparsed),
        "n_nonpositive_or_crossed_quotes": n_bad_quotes,
        "n_quotes": int(len(t))}


def read_trades(sym: str, day: str):
    """trade: recv_ns,E,T,trade_id,price,qty,is_buyer_maker."""
    files = _hour_files("trade", sym, day)
    n = len(files)
    df, n_unparsed = _read_csv(files, [2, 4, 5, 6], ["T", "price", "qty", "m"],
                               {2: "int64", 4: "float64", 5: "float64",
                                6: "int8"})
    if df is None or len(df) == 0:
        return None, n, {"n_unparsed_lines": n_unparsed}
    t = df["T"].to_numpy()
    p, q = df["price"].to_numpy(), df["qty"].to_numpy()
    m = df["m"].to_numpy(bool)
    # ZERO-QUANTITY PRINTS ARE NOT FILLS, AND THIS STREAM CARRIES THEM.
    # Measured on ADAUSDT 2026-09-01: 851 of 373,029 rows (0.23%) arrive as
    # `trade_id,0,0` -- price AND quantity zero. E1's Vision aggTrades contain
    # NONE (checked: 0 of 582,765 rows over 8 ADA days), so E1's published
    # numbers are not exposed; this is a property of the collected @trade
    # stream, the surface E2.0 introduces. Left in, a sweep composed only of
    # such prints collapses to Q = 0 and price 0, and `mo = -sgn(m1-p)/p`
    # explodes: the measured day mean went to 4.7e12 bps. Excluded here and
    # COUNTED (rule 4), never dropped in silence.
    good = (q > 0) & (p > 0)
    n_nonpositive = int((~good).sum())
    return (t[good], p[good], q[good], m[good]), n, {
        "n_unparsed_lines": int(n_unparsed),
        "n_nonpositive_prints_excluded": n_nonpositive,
        "n_prints_kept": int(good.sum())}


def gap_fraction(bt_t, day: str) -> float:
    """Share of the day's seconds with no bookTicker message."""
    day0 = int(pd.Timestamp(day, tz="UTC").timestamp()) * 1000
    inday = bt_t[(bt_t >= day0) & (bt_t < day0 + SEC_PER_DAY * 1000)]
    if len(inday) == 0:
        return 1.0
    secs = np.unique((inday - day0) // 1000)
    return float(1.0 - len(secs) / SEC_PER_DAY)


# --------------------------------------------------------------------------
# PER-DAY EVALUATION
# --------------------------------------------------------------------------
def evaluate_day(sym, day, decl, book=None, trades=None):
    D = declaring_module()
    taus = decl["the_gate_quantity"]["tau_grid_s"]
    valid_ms = D.VALID_MS
    incoh_tol = D.IDENTITY_TOL_BPS

    tr_diag, bt_diag = {}, {}
    if trades is None:
        trades, n_tr, tr_diag = read_trades(sym, day)
    else:
        n_tr = 24
    if book is None:
        book, n_bt, bt_diag = read_book(sym, day)
    else:
        n_bt = 24
    if trades is None or book is None:
        return {"symbol": sym, "date": day, "status": "NO_DATA",
                "n_bookticker_hour_files": n_bt, "n_trade_hour_files": n_tr,
                "trade_read": tr_diag, "book_read": bt_diag}

    bt_t, bid, ask = book
    gf = gap_fraction(bt_t, day)
    complete = (n_bt >= 24) and (n_tr >= 24)
    # READ from the declaring module -- no default, no re-typed literal.
    gap_max = float(D.GAP_FRACTION_MAX)
    admissible = complete and gf < gap_max

    t, p, q, m = trades
    sw = sweeps(t, p, q, m)
    ts, ps, qs, sgn = sw["t"], sw["p"], sw["Q"], sw["sign"]
    notional = qs * ps
    tm = TrueMid(bt_t, bid, ask)
    pm = ProxyMid(sw, valid_ms)

    m0t, v0t, age0 = tm.at_before(ts)
    m0p, v0p = pm.at(ts)

    rows = []
    incoherent = {"true": False, "proxy": False}
    pop_counts = {}
    for tau in taus:
        u1 = ts + int(round(tau * 1000))
        m1t, v1t, _ = tm.at(u1)
        m1p, v1p = pm.at(u1)
        # REVIEWER FINDING 1 (46030b1): the two mids do NOT exist on the same
        # sweeps -- the true mid needs only a prior quote, the proxy needs two
        # prints within 10 s. Delta_rs is therefore computed on the
        # INTERSECTION, where the ONLY thing that differs is the mid. The
        # settle leg keeps the full true-mid population, because restricting
        # the economics to what the proxy happened to see would let the
        # proxy's blindness select the settle population.
        vt, vp = v0t & v1t, v0p & v1p
        inter = vt & vp
        pop_counts[str(tau)] = {
            "n_sweeps": int(len(ts)),
            "n_true_valid": int(vt.sum()),
            "n_proxy_valid": int(vp.sum()),
            "n_intersection": int(inter.sum()),
            "n_true_valid_without_proxy": int((vt & ~vp).sum()),
            "share_true_valid_without_proxy": (
                float((vt & ~vp).sum() / vt.sum()) if vt.sum() else float("nan")),
        }
        for midname, m0, m1, val in (("true", m0t, m1t, vt),
                                     ("proxy", m0p, m1p, vp),
                                     ("true_INTERSECTION", m0t, m1t, inter),
                                     ("proxy_INTERSECTION", m0p, m1p, inter)):
            mo, es, lam = markout(sgn, ps, m0, m1)
            if val.sum() > 0 and midname in incoherent:
                gap = abs(np.mean(mo[val])
                          - (np.mean(es[val]) - np.mean(lam[val])))
                if gap > incoh_tol:
                    incoherent[midname] = True
            for side, mask in (("all", np.ones(len(sgn), bool)),
                               ("makerbid", sgn < 0), ("makerask", sgn > 0)):
                sel = mask & val
                for wname, w in (("eq", np.ones(len(sgn))),
                                 ("notional", notional)):
                    rows.append({
                        "symbol": sym, "date": day, "mid": midname,
                        "tau_s": tau, "side": side, "weighting": wname,
                        "n_events": int(sel.sum()),
                        "valid_frac": (float(val[mask].mean())
                                       if mask.any() else float("nan")),
                        "mo_bps": wmean(mo, w, sel),
                        "es_half_bps": wmean(es, w, sel),
                        "lambda_bps": wmean(lam, w, sel),
                        "notional_usd": float(notional[sel].sum()),
                    })
    # size buckets at every tau, true mid, all sides, notional quintiles
    bucket_rows = []
    if len(notional) >= 5:
        edges = np.quantile(notional, [0.2, 0.4, 0.6, 0.8])
        bidx = np.searchsorted(edges, notional, "right")
        for tau in taus:
            u1 = ts + int(round(tau * 1000))
            m1t, v1t, _ = tm.at(u1)
            mo, _, _ = markout(sgn, ps, m0t, m1t)
            val = v0t & v1t
            for b in range(5):
                sel = val & (bidx == b)
                bucket_rows.append({
                    "symbol": sym, "date": day, "tau_s": tau, "bucket": b,
                    "n_events": int(sel.sum()),
                    "mo_eq_bps": wmean(mo, np.ones(len(sgn)), sel),
                    "mo_notional_bps": wmean(mo, notional, sel),
                    "notional_usd": float(notional[sel].sum()),
                })

    # time to next OPPOSITE-side sweep (drives the tau* rule)
    opp = []
    for s in (-1.0, 1.0):
        a, b = ts[sgn == s], ts[sgn == -s]
        if len(a) and len(b):
            j = np.searchsorted(b, a, "left")
            ok = j < len(b)
            if ok.sum():
                opp.append((b[np.clip(j, 0, len(b) - 1)][ok] - a[ok]) / 1000.0)
    t_opp_med = float(np.median(np.concatenate(opp))) if opp else float("nan")

    return {
        "symbol": sym, "date": day,
        "status": "ADMISSIBLE" if admissible else (
            "EXCLUDED_INCOMPLETE" if not complete else "EXCLUDED_GAP"),
        "admissible": bool(admissible),
        "n_bookticker_hour_files": n_bt, "n_trade_hour_files": n_tr,
        "gap_fraction": gf, "gap_fraction_max": gap_max,
        "n_prints": int(len(t)), "n_sweeps": int(len(ts)),
        "trade_read": tr_diag, "book_read": bt_diag,
        "n_zero_qty_sweeps": int((qs <= 0).sum()),
        "population_counts_by_tau": pop_counts,
        "notional_usd": float(notional.sum()),
        "t_opp_med_s": t_opp_med,
        "quote_age_ms_p50": float(np.percentile(age0[v0t], 50)) if v0t.any()
        else float("nan"),
        "quote_age_ms_p99": float(np.percentile(age0[v0t], 99)) if v0t.any()
        else float("nan"),
        "true_mid_valid_frac": float(v0t.mean()),
        "proxy_mid_valid_frac": float(v0p.mean()),
        "proxy_incoherent": bool(incoherent["proxy"]),
        "true_incoherent": bool(incoherent["true"]),
        "rows": rows, "buckets": bucket_rows,
    }


# --------------------------------------------------------------------------
# AGGREGATION AND THE DECLARED PREDICATES
# --------------------------------------------------------------------------
def tau_star(day_t_opp: list[float], decl: dict) -> tuple[int, str]:
    frac = declaring_module().TAU_STAR_DAY_FRACTION
    g = len(day_t_opp)
    need = math.ceil(frac * g)
    n_fast = sum(1 for d in day_t_opp if d == d and d <= 30)
    if n_fast >= need:
        return 30, f"{n_fast} of {g} days have median time-to-opposite <= 30 s (need {need})"
    med = float(np.nanmedian(day_t_opp)) if g else float("nan")
    cand = min(300.0, 2 * med)
    return (60, f"2 x median {med:.2f}s -> {cand:.2f} -> 60") if cand <= 60 \
        else (300, f"2 x median {med:.2f}s -> {cand:.2f} -> 300")


def block_bootstrap_ci(num, den, b=None, seed=None, exp_block=8):
    D = declaring_module()
    b = D.BOOT_B if b is None else b
    seed = D.BOOT_SEED if seed is None else seed
    n = len(num)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(b)
    pgeom = 1.0 / exp_block
    for i in range(b):
        idx = np.empty(n, dtype=np.int64)
        k = 0
        while k < n:
            start = rng.integers(0, n)
            ln = min(n - k, 1 + rng.geometric(pgeom))
            idx[k:k + ln] = (start + np.arange(ln)) % n
            k += ln
        d = den[idx].sum()
        out[i] = num[idx].sum() / d if abs(d) > EPS else np.nan
    return (float(np.nanpercentile(out, 2.5)),
            float(np.nanpercentile(out, 97.5)))


def summarise(days: list[dict], decl: dict) -> dict:
    """Day-clustered means at tau*, both mids, both weightings, per side."""
    adm = [d for d in days if d.get("admissible")
           and not d.get("true_incoherent")]
    excluded = {
        "EXCLUDED_INCOMPLETE": sum(1 for d in days
                                   if d.get("status") == "EXCLUDED_INCOMPLETE"),
        "EXCLUDED_GAP": sum(1 for d in days if d.get("status") == "EXCLUDED_GAP"),
        "NO_DATA": sum(1 for d in days if d.get("status") == "NO_DATA"),
        "true_incoherent": sum(1 for d in days if d.get("true_incoherent")),
        "proxy_incoherent": sum(1 for d in days if d.get("proxy_incoherent")),
    }
    if not adm:
        return {"n_admissible_days": 0, "excluded_counts": excluded,
                "status": "NO_ADMISSIBLE_DAYS"}

    ts_, why = tau_star([d["t_opp_med_s"] for d in adm], decl)
    df = pd.DataFrame([r for d in adm for r in d["rows"]])

    def cell(mid, side, w, tau):
        s = df[(df["mid"] == mid) & (df["side"] == side)
               & (df["weighting"] == w) & (df["tau_s"] == tau)]
        if s.empty:
            return None
        v = s["mo_bps"].to_numpy(float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return None
        return {"day_clustered_mean_bps": float(v.mean()),
                "n_days": int(len(v)),
                "days_positive": int((v > 0).sum()),
                "day_values_bps": [round(float(x), 4) for x in v]}

    out = {"n_admissible_days": len(adm), "excluded_counts": excluded,
           "tau_star_s": ts_, "tau_star_why": why,
           "admissible_days": [d["date"] for d in adm],
           "cells": {}}
    pc = [d["population_counts_by_tau"][str(ts_)] for d in adm
          if str(ts_) in d.get("population_counts_by_tau", {})]
    if pc:
        out["population_at_tau_star"] = {
            "n_sweeps": sum(x["n_sweeps"] for x in pc),
            "n_true_valid": sum(x["n_true_valid"] for x in pc),
            "n_proxy_valid": sum(x["n_proxy_valid"] for x in pc),
            "n_intersection": sum(x["n_intersection"] for x in pc),
            "n_true_valid_without_proxy":
                sum(x["n_true_valid_without_proxy"] for x in pc),
            "share_true_valid_without_proxy": (
                sum(x["n_true_valid_without_proxy"] for x in pc)
                / max(sum(x["n_true_valid"] for x in pc), 1)),
            "what_it_means": (
                "the events the PROXY could not see. Delta_rs is computed on "
                "the intersection only; this share is how much of the window "
                "E1's mid was blind to, and it bounds the reach of the void "
                "reading (reviewer finding 1)."),
        }
    for mid in ("true", "proxy", "true_INTERSECTION", "proxy_INTERSECTION"):
        for side in ("all", "makerbid", "makerask"):
            for w in ("eq", "notional"):
                c = cell(mid, side, w, ts_)
                if c:
                    out["cells"][f"{mid}|{side}|{w}"] = c
    # SIZE BUCKETS. The amendment asks for "notional-weighted AND
    # size-bucketed rs on true mids"; v1 computed them and did not emit them.
    # Added after the first v2 run, which is safe because they are declared
    # `also_reported`, are not gate-bearing, and the re-run's gate cells are
    # required to come back bit-identical (they did).
    bk = pd.DataFrame([b for d in adm for b in d.get("buckets", [])])
    if not bk.empty:
        bs = bk[bk["tau_s"] == ts_]
        out["size_buckets_at_tau_star"] = [
            {"bucket": int(b),
             "n_events": int(g["n_events"].sum()),
             "notional_usd": float(g["notional_usd"].sum()),
             "mo_eq_bps_day_clustered": float(np.nanmean(g["mo_eq_bps"])),
             "mo_notional_bps_day_clustered":
                 float(np.nanmean(g["mo_notional_bps"]))}
            for b, g in bs.groupby("bucket")]
        out["size_buckets_note"] = (
            "notional quintiles per symbol-day, day-clustered. Bucket 0 is "
            "the smallest fifth by sweep notional, bucket 4 the largest.")

    # gate-2 interval on the primary cell only (rule 8: G>=5 or no interval)
    prim = df[(df["mid"] == "true") & (df["side"] == "all")
              & (df["weighting"] == "notional") & (df["tau_s"] == ts_)]
    if len(adm) >= 5 and not prim.empty:
        num = (prim["mo_bps"].to_numpy(float)
               * prim["notional_usd"].to_numpy(float))
        den = prim["notional_usd"].to_numpy(float)
        good = ~np.isnan(num)
        lo, hi = block_bootstrap_ci(num[good], den[good])
        out["primary_ci95"] = {"lo_bps": lo, "hi_bps": hi,
                               "unit": "day (G>=5, rule 8)",
                               "B": declaring_module().BOOT_B}
    else:
        out["primary_ci95"] = {
            "lo_bps": None, "hi_bps": None,
            "why": f"G={len(adm)} < 5 complete days: point estimate only, "
                   f"no interval (CLAUDE.md rule 8)"}
    return out


def apply_declared_predicates(summary: dict, decl: dict) -> dict:
    """The 2x2, evaluated by importing the DECLARATION's own functions."""
    D = declaring_module()
    c = summary.get("cells", {})
    # SETTLE: the FULL true-mid population, notional-weighted (the amendment).
    t = c.get("true|all|notional")
    # VOID: the INTERSECTION, eq-weighted (reviewer finding 1 + the plan).
    pe = c.get("proxy_INTERSECTION|all|eq")
    te = c.get("true_INTERSECTION|all|eq")
    delta = (pe["day_clustered_mean_bps"] - te["day_clustered_mean_bps"]) \
        if (pe and te) else None
    ci = summary.get("primary_ci95", {})
    void = D.void_predicate(delta)
    settle = D.settle_predicate(
        t["day_clustered_mean_bps"] if t else None,
        ci_lo_bps=ci.get("lo_bps"), ci_hi_bps=ci.get("hi_bps"),
        interval_claimable=ci.get("lo_bps") is not None)
    verdict = D.ada_verdict(void, settle)
    return {
        "delta_rs_bps": delta,
        "delta_rs_population": "INTERSECTION (both mids defined at t- and "
                               "t+tau*), eq-weighted -- reviewer finding 1",
        "settle_population": "ALL sweeps with a valid TRUE mid, "
                             "notional-weighted -- the amendment's quantity",
        "populations": summary.get("population_at_tau_star"),
        "void_leg": void, "settle_leg": settle, "verdict": verdict,
        "predicates_from": "e2_0_declare.py (the declaring module), not "
                           "re-implemented here"}


# --------------------------------------------------------------------------
# THE E1 REPRODUCTION CONTROL
# --------------------------------------------------------------------------
def reproduce_e1(sym: str = "ADAUSDT", decl: dict | None = None) -> dict:
    """Re-measure E1's ADA on E1's OWN Vision aggTrades with THIS code."""
    decl = decl or load_declaration()
    tgt = decl["e1_reproduction_control"]
    files = sorted(VISION.glob(f"{sym}/*.parquet"))
    if not files:
        return {"status": "SOURCE_ABSENT", "path": str(VISION / sym)}
    rows = []
    for f in files:
        d = pd.read_parquet(f, columns=["price", "quantity", "transact_time",
                                        "is_buyer_maker"])
        tt = d["transact_time"]
        if tt.dtype.kind == "M":
            t = tt.astype("int64").to_numpy()
            if t.max() > 10 ** 16:
                t = t // 10 ** 6
        else:
            t = tt.to_numpy()
        sw = sweeps(t, d["price"].to_numpy(float), d["quantity"].to_numpy(float),
                    d["is_buyer_maker"].to_numpy(bool))
        pm = ProxyMid(sw, 10_000)
        ts, ps, qs, sgn = sw["t"], sw["p"], sw["Q"], sw["sign"]
        m0, v0 = pm.at(ts)
        m1, v1 = pm.at(ts + tgt["tau_star_s"] * 1000)
        mo, _, _ = markout(sgn, ps, m0, m1)
        val = v0 & v1
        w = qs * ps
        rows.append({"date": f.stem,
                     "eq": wmean(mo, np.ones(len(sgn)), val),
                     "notional": wmean(mo, w, val)})
    df = pd.DataFrame(rows)
    eq = float(df["eq"].mean())
    no = float(df["notional"].mean())
    tol = tgt["tolerance_bps"]
    return {
        "status": "RUN", "symbol": sym, "n_days": int(len(df)),
        "rs_eq_bps": eq, "rs_notional_bps": no,
        "days_eq_positive": int((df["eq"] > 0).sum()),
        "days_notional_positive": int((df["notional"] > 0).sum()),
        "target": tgt,
        "eq_within_tol": bool(abs(eq - tgt["rs_eq_bps"]) <= tol),
        "notional_within_tol": bool(abs(no - tgt["rs_notional_bps"]) <= tol),
        "eq_abs_error_bps": abs(eq - tgt["rs_eq_bps"]),
        "notional_abs_error_bps": abs(no - tgt["rs_notional_bps"]),
        "reproduced": bool(abs(eq - tgt["rs_eq_bps"]) <= tol
                           and abs(no - tgt["rs_notional_bps"]) <= tol),
    }


# --------------------------------------------------------------------------
# FALSIFIERS -- the ones the declaration names, both directions
# --------------------------------------------------------------------------
def _synth(n=400, drift_bps=+5.0, favour_maker=True, seed=7):
    """A book whose mid moves a KNOWN number of bps after every fill, in the
    maker's favour or against it. Both sides present."""
    rng = np.random.default_rng(seed)
    t0 = 1_787_000_000_000
    ts = t0 + np.arange(n) * 10_000            # a sweep every 10 s
    px = np.full(n, 100.0)
    mkr_buy = rng.random(n) < 0.5              # True -> maker BID filled
    sgn = np.where(mkr_buy, -1.0, 1.0)
    # maker profits when the mid moves UP after a maker-bid fill (-sgn = +1)
    direction = (-sgn) if favour_maker else sgn
    m_after = px * (1 + direction * drift_bps / 1e4)
    bt_t, bt_mid = [], []
    for i in range(n):
        bt_t += [ts[i] - 1, ts[i] + 1000]
        bt_mid += [px[i], m_after[i]]
    bt_t = np.array(bt_t, dtype=np.int64)
    bt_mid = np.array(bt_mid)
    return ts, px, mkr_buy, sgn, bt_t, bt_mid


def selftest() -> int:                                        # noqa: C901
    global DECL_PATH, DECL_SHA
    fails: list[str] = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    decl = load_declaration()
    ok(decl["protocol"] == declaring_module().PROTOCOL,
       f"DECLARATION PINNED: sha256 {DECL_SHA[:16]} verified before anything "
       f"is computed; a mismatch REFUSES")

    # --- sign convention, driven in both directions ---
    for favour, want_sign, label in ((True, +1, "IN the maker's favour"),
                                     (False, -1, "AGAINST the maker")):
        ts, px, mk, sgn, bt_t, bt_mid = _synth(drift_bps=5.0,
                                               favour_maker=favour)
        tm = TrueMid(bt_t, bt_mid - 0.001, bt_mid + 0.001)
        m0, v0, _ = tm.at_before(ts)
        m1, v1, _ = tm.at(ts + 1000)
        mo, es, lam = markout(sgn, px, m0, m1)
        val = v0 & v1
        mean_mo = float(np.mean(mo[val]))
        ok(val.all() and abs(mean_mo - want_sign * 5.0) < 0.02,
           f"{'POSITIVE CONTROL' if favour else 'KNOWN-BAD'}: a mid that moves "
           f"5 bps {label} gives rs = {mean_mo:+.3f} bps -- the KNOWN value "
           f"with the right sign, on both maker-bid and maker-ask events")

    # --- the identity es - Lambda == MO, on a known path ---
    ts, px, mk, sgn, bt_t, bt_mid = _synth(drift_bps=5.0)
    tm = TrueMid(bt_t, bt_mid - 0.001, bt_mid + 0.001)
    m0, v0, _ = tm.at_before(ts)
    m1, v1, _ = tm.at(ts + 1000)
    mo, es, lam = markout(sgn, px, m0, m1)
    val = v0 & v1
    ok(abs(np.mean(mo[val]) - (np.mean(es[val]) - np.mean(lam[val]))) < 0.01,
       "POSITIVE CONTROL: the section 1.3c identity mean es - mean Lambda == "
       "mean MO holds on a synthetic path -- so a real-data failure is the "
       "reader's fault, not the market's")

    # --- reversing the convention must FLIP the sign ---
    mo_rev, _, _ = markout(-sgn, px, m0, m1)
    ok(abs(np.mean(mo_rev[val]) + np.mean(mo[val])) < 1e-9,
       "KNOWN-BAD: reversing is_buyer_maker FLIPS the sign of rs -- if it did "
       "not, the sign convention would not be wired to anything")

    # --- weighting must discriminate: the H1 shape itself ---
    n = 1000
    mo_v = np.concatenate([np.full(n - 10, +2.0), np.full(10, -50.0)])
    notl = np.concatenate([np.full(n - 10, 100.0), np.full(10, 100_000.0)])
    sel = np.ones(n, bool)
    eqm = wmean(mo_v, np.ones(n), sel)
    nom = wmean(mo_v, notl, sel)
    ok(eqm > 0 and nom < 0,
       f"POSITIVE CONTROL: many small POSITIVE events and few large NEGATIVE "
       f"ones give eq {eqm:+.3f} > 0 and notional {nom:+.3f} < 0 -- the H1 "
       f"shape that killed ADA, so the weighting code is shown to "
       f"DISCRIMINATE and not merely to run")
    ok(abs(wmean(mo_v, np.ones(n), sel)
           - wmean(mo_v, np.full(n, 3.0), sel)) < 1e-9,
       "POSITIVE CONTROL: a constant weight equals eq weighting -- the "
       "weighted mean is a weighted mean")

    # --- a query past the end of the book is EXCLUDED, not extrapolated ---
    _, ok_far, _ = tm.at(np.array([bt_t[-1] + 10_000]))
    _, ok_in, _ = tm.at(np.array([bt_t[-1] - 10]))
    ok(bool(ok_far[0]) is False and bool(ok_in[0]) is True,
       "KNOWN-BAD: a query PAST the end of the held book is INVALID (excluded "
       "and counted), while one inside it is valid -- never valued at the "
       "last known quote")

    # --- the proxy goes blind where the true mid does not ---
    # MY FIRST VERSION OF THIS CONTROL WAS WRONG: I asserted the proxy's
    # validity falls with the HORIZON. It does not -- prints keep arriving, so
    # a later query is just as two-sided as an earlier one. What actually
    # separates the two mids is a QUIET PATCH: the proxy needs a print on BOTH
    # sides within 10 s and goes blind without one, while the book still has a
    # standing quote. That is the difference E2.0 exists to measure, so that
    # is what is driven.
    base = 1_787_000_000_000
    dense_t = base + np.arange(60) * 500                      # a print every 0.5 s
    dense_side = np.tile([True, False], 30)
    sw_q = {"t": dense_t, "p": np.full(60, 100.0), "Q": np.ones(60),
            "mkr_buy": dense_side, "sign": np.where(dense_side, -1.0, 1.0)}
    pm_q = ProxyMid(sw_q, 10_000)
    bt_q = np.arange(base, base + 120_000, 100, dtype=np.int64)
    tm_q = TrueMid(bt_q, np.full(len(bt_q), 99.99), np.full(len(bt_q), 100.01))
    u_dense = np.array([base + 25_000], dtype=np.int64)       # inside the prints
    u_quiet = np.array([base + 60_000], dtype=np.int64)       # 30 s after the last
    _, vp_dense = pm_q.at(u_dense)
    _, vp_quiet = pm_q.at(u_quiet)
    _, vt_dense, _ = tm_q.at(u_dense)
    _, vt_quiet, _ = tm_q.at(u_quiet)
    ok(bool(vp_dense[0]) and bool(vt_dense[0]),
       "POSITIVE CONTROL: where prints are dense BOTH mids are valid -- the "
       "proxy is not broken, it is limited")
    ok((not bool(vp_quiet[0])) and bool(vt_quiet[0]),
       "KNOWN-BAD: 30 s into a print drought the PROXY mid is INVALID while "
       "the true mid still has a standing quote -- the exact gap E2.0 exists "
       "to measure, and the reason the two legs can disagree")

    # --- ZERO-QUANTITY PRINTS: the defect this run actually found ---
    # The collected @trade stream carries rows of the form `trade_id,0,0`
    # (measured: 851 of 373,029 on ADAUSDT 2026-09-01). A sweep made only of
    # them collapses to Q=0, price 0, and mo = -sgn(m1-p)/p explodes -- the
    # first real ADA day came back at 4.7e12 bps. E1's Vision aggTrades have
    # none (0 of 582,765 over 8 days), so this is a property of the NEW
    # surface. Both directions are driven: the poison must be visible if it is
    # NOT excluded, and invisible once it is.
    t_ok = np.array([1000, 1000, 2000, 3000], dtype=np.int64)
    p_ok = np.array([100.0, 100.0, 100.0, 100.0])
    q_ok = np.array([1.0, 2.0, 3.0, 4.0])
    m_ok = np.array([True, True, False, True])
    t_bad = np.concatenate([t_ok, [4000]]).astype(np.int64)
    p_bad = np.concatenate([p_ok, [0.0]])
    q_bad = np.concatenate([q_ok, [0.0]])
    m_bad = np.concatenate([m_ok, [True]])
    sw_poison = sweeps(t_bad, p_bad, q_bad, m_bad)
    ok(int((sw_poison["Q"] <= 0).sum()) == 1 and sw_poison["p"].min() == 0.0,
       "KNOWN-BAD: a zero-quantity print left IN the population produces a "
       "sweep with Q = 0 and price 0 -- the shape that sent a real ADA day to "
       "4.7e12 bps")
    bt = np.array([500, 1500, 2500, 3500, 4500, 5500], dtype=np.int64)
    tmz = TrueMid(bt, np.full(6, 99.99), np.full(6, 100.01))
    m0z, v0z, _ = tmz.at_before(sw_poison["t"])
    m1z, v1z, _ = tmz.at(sw_poison["t"] + 1000)
    moz, _, _ = markout(sw_poison["sign"], sw_poison["p"], m0z, m1z)
    ok(np.abs(moz).max() > 1e6,
       f"KNOWN-BAD: and it explodes the markout -- max |mo| = "
       f"{np.abs(moz).max():.2e} bps on four otherwise ordinary events")
    good = (q_bad > 0) & (p_bad > 0)
    sw_clean = sweeps(t_bad[good], p_bad[good], q_bad[good], m_bad[good])
    sw_ref = sweeps(t_ok, p_ok, q_ok, m_ok)
    ok(len(sw_clean["t"]) == len(sw_ref["t"])
       and np.allclose(sw_clean["p"], sw_ref["p"])
       and np.allclose(sw_clean["Q"], sw_ref["Q"]),
       "POSITIVE CONTROL: excluding non-positive prints reproduces the "
       "population EXACTLY as if they had never been sent -- the exclusion "
       "removes the poison and nothing else")
    mixed = sweeps(np.array([1000, 1000], dtype=np.int64),
                   np.array([100.0, 0.0]), np.array([5.0, 0.0]),
                   np.array([True, True]))
    ok(abs(mixed["p"][0] - 100.0) < 1e-12 and mixed["Q"][0] == 5.0,
       "POSITIVE CONTROL: a zero print SHARING a sweep with a real one does "
       "not move that sweep's qty-weighted price -- only all-zero sweeps were "
       "ever at risk, and the fix is not doing collateral damage")

    # --- gap fraction, both directions ---
    day = "20260901"
    day0 = int(pd.Timestamp(day, tz="UTC").timestamp()) * 1000
    full = day0 + np.arange(SEC_PER_DAY, dtype=np.int64) * 1000
    ok(gap_fraction(full, day) == 0.0,
       "POSITIVE CONTROL: a book with a message every second has gap "
       "fraction 0.0 and is ADMITTED")
    holed = full[:int(SEC_PER_DAY * 0.90)]
    gf = gap_fraction(holed, day)
    ok(abs(gf - 0.10) < 1e-9 and gf >= 0.05,
       f"KNOWN-BAD: a day missing 10% of its seconds reads gap_fraction "
       f"{gf:.3f} and EXCEEDS the 0.05 bar -- excluded with a named status "
       f"(EXCLUDED_GAP), counted, never silently dropped")

    # --- sweep collapse ---
    t = np.array([1000, 1000, 1000, 2000], dtype=np.int64)
    p = np.array([10.0, 10.0, 11.0, 12.0])
    q = np.array([1.0, 2.0, 3.0, 4.0])
    mm = np.array([True, True, True, False])
    s = sweeps(t, p, q, mm)
    ok(len(s["t"]) == 2 and s["Q"][0] == 6.0
       and abs(s["p"][0] - (10 * 1 + 10 * 2 + 11 * 3) / 6) < 1e-12,
       "POSITIVE CONTROL: three prints at one (transact_time, is_buyer_maker) "
       "collapse to ONE sweep with the qty-weighted price -- the event unit "
       "is the sweep, as section 1.1 requires, so the per-match `trade` "
       "stream is comparable to E1's aggTrades")
    ok(s["sign"][0] == -1.0 and s["sign"][1] == +1.0,
       "SIGN: is_buyer_maker -> q_j = -1 (taker sold, maker BID filled)")

    # --- the runner refuses a stale declaration ---
    import tempfile
    _p, _s = DECL_PATH, DECL_SHA
    try:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "decl.json"
            f.write_text('{"protocol": "SOMETHING"}')
            DECL_PATH = f
            try:
                load_declaration()
                ok(False, "KNOWN-BAD: accepted a declaration whose digest "
                          "does not match -- must REFUSE")
            except E20Refused as e:
                ok("sha256" in str(e),
                   "KNOWN-BAD: a declaration whose sha256 differs REFUSES -- "
                   "the gate cannot be redefined after seeing")
            DECL_PATH = Path(d) / "absent.json"
            try:
                load_declaration()
                ok(False, "KNOWN-BAD: accepted an ABSENT declaration")
            except E20Refused:
                ok(True, "KNOWN-BAD: an ABSENT declaration REFUSES")
    finally:
        DECL_PATH, DECL_SHA = _p, _s
    ok(load_declaration()["protocol"].startswith("P002"),
       "POSITIVE CONTROL: the real declaration still loads after the "
       "known-bads -- the refusal discriminates rather than always firing")

    # --- THE LEDGER ROOT: resolved, recorded, and a rootless tree REFUSES ---
    ok(DATA_ROOT_BRANCH in ("1_env_PM_DATA_ROOT",
                            "2_code_tree_carries_the_tape",
                            "3_unresolved_no_tape"),
       f"LEDGER ROOT: the branch taken is RECORDED, not assumed -- "
       f"{DATA_ROOT_BRANCH}, root {ROOT}")
    import subprocess as _sp
    import tempfile as _tf2
    with _tf2.TemporaryDirectory() as d:
        r = _sp.run([sys.executable, "-c",
                     "import sys; sys.path.insert(0, %r)\n"
                     "import e2_0_true_mid as E\n"
                     "print(E.DATA_ROOT_BRANCH)\n"
                     "try:\n"
                     "    E.require_tape(); print('ADMITTED')\n"
                     "except E.E20Refused as e: print('REFUSED')\n"
                     % str(HERE)],
                    capture_output=True, text=True,
                    env={**os.environ, "PM_DATA_ROOT": d})
        out = r.stdout.strip().splitlines()
        ok(out == ["1_env_PM_DATA_ROOT", "REFUSED"],
           f"KNOWN-BAD: PM_DATA_ROOT pointed at a tree with NO TAPE takes the "
           f"env branch and then REFUSES -- an empty population is not a "
           f"result, which is the failure that cost a run in round 58 "
           f"(got {out})")
        r2 = _sp.run([sys.executable, "-c",
                      "import sys; sys.path.insert(0, %r)\n"
                      "import e2_0_true_mid as E\n"
                      "E.require_tape(); print(E.DATA_ROOT_BRANCH, "
                      "len(E.days_available('ADAUSDT')) > 0)\n" % str(HERE)],
                     capture_output=True, text=True,
                     env={**os.environ, "PM_DATA_ROOT": "/home/yuqing/ctaNew"})
        ok(r2.returncode == 0 and "1_env_PM_DATA_ROOT True" in r2.stdout,
           "POSITIVE CONTROL: PM_DATA_ROOT at the real repo root ADMITS and "
           "finds tape -- the refusal discriminates rather than always firing")

    # --- tau* rule reproduces the plan's own instantiation ---
    ts30 = [10.0] * 24 + [90.0] * 7
    ok(tau_star(ts30, decl)[0] == 30,
       "TAU*: 24 of 31 fast days gives tau* = 30 s -- the plan's own "
       "instantiation, reproduced")
    ok(tau_star([90.0] * 31, decl)[0] == 300,
       "TAU*: a slow tape does NOT give 30 s -- the rule can move, so it is "
       "not a constant wearing a rule's name")

    # --- the verdict comes from the declaring module, not from here ---
    sm = {"cells": {
        "true|all|notional": {"day_clustered_mean_bps": -0.4},
        "proxy_INTERSECTION|all|eq": {"day_clustered_mean_bps": 2.4},
        "true_INTERSECTION|all|eq": {"day_clustered_mean_bps": 2.0}},
        "primary_ci95": {"lo_bps": -1.2, "hi_bps": 0.3}}
    r = apply_declared_predicates(sm, decl)
    ok(abs(r["delta_rs_bps"] - 0.4) < 1e-9
       and r["verdict"]["verdict"] == "NOT_VOIDED+DEAD",
       f"WIRED TO THE DECLARATION: a synthetic summary routes through "
       f"e2_0_declare's own predicates and returns "
       f"{r['verdict']['verdict']} -- the runner does not re-implement the gate")
    sm["cells"]["proxy_INTERSECTION|all|eq"]["day_clustered_mean_bps"] = 4.0
    ok(apply_declared_predicates(sm, decl)["verdict"]["verdict"]
       == "VOIDED+DEAD",
       "WIRED: raising the proxy mid past the +1.0 bps tolerance moves the "
       "verdict -- the void leg is connected")
    # REVIEWER FINDING 1: Delta_rs must come from the INTERSECTION cells. If
    # the runner read the unrestricted cells instead, this would still return
    # a number -- so the check is that the intersection keys are the ones
    # consulted, driven by making them disagree.
    sm2 = {"cells": {
        "true|all|notional": {"day_clustered_mean_bps": -0.4},
        "true|all|eq": {"day_clustered_mean_bps": 9.0},
        "proxy|all|eq": {"day_clustered_mean_bps": 9.0},
        "proxy_INTERSECTION|all|eq": {"day_clustered_mean_bps": 2.4},
        "true_INTERSECTION|all|eq": {"day_clustered_mean_bps": 2.0}},
        "primary_ci95": {"lo_bps": -1.2, "hi_bps": 0.3}}
    ok(abs(apply_declared_predicates(sm2, decl)["delta_rs_bps"] - 0.4) < 1e-9,
       "REVIEWER FINDING 1 WIRED: with the unrestricted and intersection "
       "cells set to DIFFERENT values, Delta_rs takes the INTERSECTION pair "
       "-- so the population difference cannot leak into the mid difference")
    # REVIEWER FINDINGS 2 AND 3, wired through the runner's own path.
    sm3 = {"cells": {
        "true|all|notional": {"day_clustered_mean_bps": 2.0},
        "proxy_INTERSECTION|all|eq": {"day_clustered_mean_bps": 2.0},
        "true_INTERSECTION|all|eq": {"day_clustered_mean_bps": 2.0}},
        "primary_ci95": {"lo_bps": 1.9, "hi_bps": 2.1}}
    ok(apply_declared_predicates(sm3, decl)["verdict"]["settle_state"]
       == "NOT_KILLED_PENDING_GATE_1",
       "REVIEWER FINDING 2 WIRED: 2.0 bps through the runner reads "
       "NOT_KILLED_PENDING_GATE_1, not ALIVE -- the band between the 1.8 "
       "death bar and the plan's 2.3 gate has its own name end to end")
    sm3["cells"]["true|all|notional"]["day_clustered_mean_bps"] = 2.5
    sm3["primary_ci95"] = {"lo_bps": 1.0, "hi_bps": 4.0}
    ok(apply_declared_predicates(sm3, decl)["verdict"]["settle_state"]
       == "INCONCLUSIVE_INTERVAL_DOES_NOT_CLEAR",
       "REVIEWER FINDING 3 WIRED: a point of 2.5 with a CI lower bound of 1.0 "
       "reads INCONCLUSIVE through the runner -- the interval is "
       "decision-bearing, not decorative")
    sm3["primary_ci95"] = {"lo_bps": 2.0, "hi_bps": 3.0}
    ok(apply_declared_predicates(sm3, decl)["verdict"]["cell_passes_gate1"]
       is True,
       "POSITIVE CONTROL: with the interval clearing too, the same point "
       "DOES pass gate 1 -- the interval rule can admit")

    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


# --------------------------------------------------------------------------
def require_tape() -> None:
    """A root with no tape REFUSES. Reporting zero days would be a result-
    shaped object built from an absent input -- the shape that cost a run."""
    if not RAW.is_dir():
        raise E20Refused(
            f"REFUSED: no tape at {RAW} (root branch {DATA_ROOT_BRANCH}, "
            f"code tree {CODE_ROOT}). Set PM_DATA_ROOT to the REPO root that "
            f"carries data/mm_hf/raw. An empty population is not a result.")


def days_available(sym: str) -> list[str]:
    d = RAW / "bookTicker" / sym
    return sorted({f.name.split("_")[0] for f in d.glob("*.csv*")})


def run(symbols, decl, out_path: Path | None):
    t0 = time.time()
    require_tape()
    result = {"protocol": PROTOCOL, "carrying_commit": carrying_commit(),
              "ledger_root": {
                  "data_root": str(ROOT),
                  "data_root_branch": DATA_ROOT_BRANCH,
                  "code_root": str(CODE_ROOT),
                  "raw": str(RAW),
                  "code_and_data_are_the_same_tree": str(ROOT) == str(CODE_ROOT),
                  "why_recorded": (
                      "so a reader can see WHICH tree a number came from. A "
                      "worktree's own data/ carries only the git-tracked "
                      "receipts, not the tape; running there silently would "
                      "measure a different population.")},
              "declaration": {"path": str(DECL_PATH.relative_to(ROOT)),
                              "sha256": DECL_SHA,
                              "carrying_commit": decl["carrying_commit"]},
              "symbols": {}}
    for sym in symbols:
        days = days_available(sym)
        per_day, wall0 = [], time.time()
        for day in days:
            per_day.append(evaluate_day(sym, day, decl))
        summ = summarise(per_day, decl)
        n_adm = summ.get("n_admissible_days", 0)
        if n_adm < decl["population"]["min_complete_days"]:
            summ["REFUSED"] = (
                f"{n_adm} admissible days < the declared minimum "
                f"{decl['population']['min_complete_days']}: no gate is read "
                f"for this symbol and the cap is not relaxed")
        result["symbols"][sym] = {
            "days": [{k: v for k, v in d.items()
                      if k not in ("rows", "buckets")} for d in per_day],
            "summary": summ,
            "predicates": (apply_declared_predicates(summ, decl)
                           if n_adm >= decl["population"]["min_complete_days"]
                           else {"status": "NOT_EVALUATED_REFUSED"}),
            "wall_s": round(time.time() - wall0, 2),
        }
    result["wall_s_total"] = round(time.time() - t0, 2)
    try:
        import resource
        result["max_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss
    except Exception:                                        # noqa: BLE001
        pass
    if out_path:
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def supersede_note(earlier: Path, operative: Path, out: Path) -> dict:
    """Mark an earlier receipt superseded WITHOUT editing it (rule 13).

    The claim "these two carry the same result" is COMPUTED here field by
    field, not asserted in prose: if any gate-bearing field differs, the note
    says so and refuses to call the earlier one merely superseded.
    """
    a = json.loads(earlier.read_text())
    b = json.loads(operative.read_text())
    sa = list(a["symbols"])[0]
    ga, gb = a["symbols"][sa], b["symbols"][sa]
    fields = {
        "cells": ga["summary"]["cells"] == gb["summary"]["cells"],
        "verdict": ga["predicates"]["verdict"] == gb["predicates"]["verdict"],
        "primary_ci95": (ga["summary"]["primary_ci95"]
                         == gb["summary"]["primary_ci95"]),
        "population_at_tau_star": (ga["summary"].get("population_at_tau_star")
                                   == gb["summary"].get("population_at_tau_star")),
        "tau_star_s": ga["summary"]["tau_star_s"] == gb["summary"]["tau_star_s"],
        "admissible_days": (ga["summary"]["admissible_days"]
                            == gb["summary"]["admissible_days"]),
        "delta_rs_bps": (ga["predicates"]["delta_rs_bps"]
                         == gb["predicates"]["delta_rs_bps"]),
    }
    only_new = sorted(set(gb["summary"]) - set(ga["summary"]))
    note = {
        "protocol": "P002_E2_0_SUPERSEDED_BY_V1",
        "what_this_is": (
            "a SIDECAR. The superseded receipt is NOT edited (rule 13): a "
            "frozen artifact stays as provenance and the pointer lives beside "
            "it, because an automated reader resolves receipt fields and a "
            "receipt with no `superseded_by` reads as current."),
        "superseded": {"path": str(earlier).split("ctaNew/")[-1],
                       "sha256": digest(earlier)},
        "operative": {"path": str(operative).split("ctaNew/")[-1],
                      "sha256": digest(operative)},
        "carrying_commit": carrying_commit(),
        "gate_bearing_fields_identical": fields,
        "all_gate_bearing_fields_identical": all(fields.values()),
        "what_differs": {
            "keys_only_in_the_operative_receipt": only_new,
            "why": ("the operative receipt adds the declared but previously "
                    "un-emitted size-bucket table (the amendment asks for "
                    "notional-weighted AND size-bucketed rs). Nothing "
                    "gate-bearing moved, which is the point of the field-by-"
                    "field comparison above rather than a prose assurance."),
        },
        "why_both_stand_in_git": (
            "the earlier receipt was READ before the buckets were added, so "
            "deleting it would remove an artifact a decision saw. It is "
            "committed as provenance and this sidecar makes it "
            "unresolvable-as-current."),
    }
    if not all(fields.values()):
        note["REFUSED"] = (
            "a gate-bearing field DIFFERS between the two receipts; this is "
            "not a supersession by an added diagnostic and must not be "
            "described as one")
    out.write_text(json.dumps(note, indent=2, sort_keys=True) + "\n")
    return note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--supersede", nargs=3, metavar=("EARLIER", "OPERATIVE",
                                                     "OUT"))
    ap.add_argument("--reproduce-e1", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--symbols", nargs="*", default=["ADAUSDT"])
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.supersede:
        n = supersede_note(Path(a.supersede[0]), Path(a.supersede[1]),
                           Path(a.supersede[2]))
        print(json.dumps(n, indent=2, sort_keys=True))
        return 0 if n["all_gate_bearing_fields_identical"] else 1
    decl = load_declaration()
    if a.reproduce_e1:
        r = reproduce_e1(a.symbols[0], decl)
        print(json.dumps(r, indent=2, sort_keys=True))
        if a.output:
            a.output.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
        return 0 if r.get("reproduced") else 1
    if a.run:
        r = run(a.symbols, decl, a.output)
        for s, v in r["symbols"].items():
            print(f"{s}: {json.dumps(v['summary'], indent=2, sort_keys=True)}")
            print(f"{s} predicates: "
                  f"{json.dumps(v['predicates'], indent=2, sort_keys=True)}")
        print(f"wall {r['wall_s_total']}s  max_rss "
              f"{r.get('max_rss_kib', '?')} KiB")
        return 0
    ap.error("choose --selftest, --reproduce-e1 or --run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
