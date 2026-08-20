"""E1 — spread-economics & markout universe scan (P-2026-002).

Implements EXPERIMENT_PLAN.md §1 (pre-registered 2026-08-19) on the 16-sym × 31-day
Vision aggTrades parquets. Outputs the five §1.6 CSVs to data/mm_hf/e1/.

Protocol deviations (logged per prereg discipline):
  D-a. XS-overlap ADV: plan says trailing-ADV top-40 "as of 2026-08-17"; the only
       full-universe notional source (data/ml/cache/flow_*.parquet) ends 2026-05-30,
       so ADV = trailing-30d mean daily notional ending 2026-05-30. ~2.5-month
       staleness, flagged in e1a_universe.csv.
  D-b. proxy_incoherent: plan §1.3c is ambiguous over τ; implemented as "flag the
       day if the identity fails at ANY τ" (strictest reading).
  D-c. E1-A bootstrap: §1.5 says "block-bootstrap CI" without a bin spec; implemented
       as stationary bootstrap on the cross-symbol day-mean sequence, expected block
       3 days, B=2000 (day-level object; the §1.4 30-min-bin spec is for markouts).
  Post-review amendments (E1_CODE_REVIEW.md, applied BEFORE gates were read):
  D-e. e1a_overlay_daily gains n_skipped (episodes dropped on invalid decision- or
       chase-mid were silent → undeclared optimistic bias, MUST FIX 2).
  D-f. e1b_final distinguishes "no_estimate" (days_used < 10 after exclusions) from
       measured "fail" — a no-estimate is NOT final under §1.2 (SHOULD FIX 4).
  D-g. bootstrap bins stored for τ ∈ {30,60,300} so slow-tape (τ*≠30) symbols use
       the §1.4 bin bootstrap rather than an unlogged day-level fallback (SHOULD FIX 3).
  D-h. shadow accounting eff_rt_stale_bps: skipped chases re-booked at the stale
       (validity-free) two-sided mid to BOUND the D-e drop bias (SHOULD FIX 5).
  D-i. (post-first-read, before acceptance) E1-A touch/sweep compared on INTEGER
       tick indices — float `floor(L/tick)*tick` arithmetic made px==L never fire,
       collapsing the bracket to one rule (observed: touch ≡ sweep to 6 decimals).
  D-j. gate condition 3 uses the prereg's LITERAL count (rs>0 on ≥22 days), not
       fraction-of-used-days — after the D-f/MUST-FIX-1 exclusions a thin name
       (ATOM: 3 usable days) could otherwise pass on a 3-day sample.

Run: python3 -u -m live.mm_research.e1_markout_scan
"""
from __future__ import annotations

import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/mm_hf/vision/parquet/aggTrades"
OUT = REPO / "data/mm_hf/e1"

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT",
        "ADAUSDT", "AVAXUSDT", "LTCUSDT", "APTUSDT", "ARBUSDT", "FILUSDT",
        "ATOMUSDT", "AAVEUSDT", "GMXUSDT", "ICPUSDT"]

TAUS_S = (1, 5, 15, 30, 60, 300)         # markout horizons (§1.3b; 30 s = gate point)
VALID_MS = 10_000                         # two-sided mid validity window (§1.2)
FLIP_MS = (1000, 100)                     # ES flip-pair Δt_max: primary, robustness (§1.3a)
FEE_MAKER = {"vip0": 1.8, "vip1": 1.44}   # bps, +BNB (§0)
FEE_TAKER_VIP0 = 4.5                      # bps (E1-A chase branch)
C_SAFE = 0.5                              # bps safety margin (§0)
TP_GRID_S = (60, 600, 3600)               # E1-A patience grid; primary = 600 (§1.5)
INCOHERENT_BPS = 0.2                      # §1.3c reconciliation threshold
BOOT_B = 2000
BOOT_SEED = 20260819
EPS = 1e-12


# ---------------------------------------------------------------- data access
def day_files(sym: str) -> list[Path]:
    return sorted((SRC / sym).glob("*.parquet"))


def load_day(p: Path):
    df = pd.read_parquet(p, columns=["price", "quantity", "transact_time", "is_buyer_maker"])
    t = df["transact_time"].astype("int64").to_numpy() // 10**6 \
        if df["transact_time"].dtype.kind == "M" else df["transact_time"].to_numpy()
    # tz-aware datetime64[ms] → int64 is ms already when unit is ms; guard both
    if df["transact_time"].dtype.kind == "M":
        t = df["transact_time"].astype("int64").to_numpy()
        if t.max() > 10**16:          # ns-scaled
            t = t // 10**6
    return t, df["price"].to_numpy(float), df["quantity"].to_numpy(float), \
        df["is_buyer_maker"].to_numpy(bool)


def sweeps(t, p, q, m):
    """Collapse prints → sweep events keyed by (transact_time, is_buyer_maker). §1.1"""
    order = np.lexsort((m.astype(np.int8), t))
    t, p, q, m = t[order], p[order], q[order], m[order]
    key_change = np.empty(len(t), bool)
    key_change[0] = True
    key_change[1:] = (t[1:] != t[:-1]) | (m[1:] != m[:-1])
    starts = np.flatnonzero(key_change)
    notq = np.add.reduceat(q, starts)
    pw = np.add.reduceat(p * q, starts) / np.maximum(notq, EPS)
    out = {
        "t": t[starts],
        "p": pw,                                   # qty-weighted sweep price
        "Q": notq,
        "pmin": np.minimum.reduceat(p, starts),
        "pmax": np.maximum.reduceat(p, starts),
        "n": np.diff(np.append(starts, len(t))),
        "mkr_buy": m[starts],                      # True = taker sold (maker BID filled)
    }
    out["sign"] = np.where(out["mkr_buy"], -1.0, 1.0)   # q_j: +1 taker buy (§1.1)
    return out


# ---------------------------------------------------------------- mid proxies
class MidProxy:
    """Two-sided last-print mid with 10 s validity (§1.2) + forward trade VWAP."""

    def __init__(self, sw):
        buy = ~sw["mkr_buy"]                       # taker-buy sweeps (ask-side prints)
        self.tb, self.pb = sw["t"][buy], sw["p"][buy]
        self.ts, self.ps = sw["t"][~buy], sw["p"][~buy]
        self.ta, self.pa, self.Qa = sw["t"], sw["p"], sw["Q"]
        self._cpq = np.concatenate([[0.0], np.cumsum(self.pa * self.Qa)])
        self._cq = np.concatenate([[0.0], np.cumsum(self.Qa)])

    def mid(self, u):
        """m̂(u⁻): strictly-before two-sided mid; returns (mid, valid)."""
        ib = np.searchsorted(self.tb, u, "left") - 1
        is_ = np.searchsorted(self.ts, u, "left") - 1
        ok = (ib >= 0) & (is_ >= 0)
        ibc, isc = np.clip(ib, 0, None), np.clip(is_, 0, None)
        ok &= (self.tb[ibc] >= u - VALID_MS) & (self.ts[isc] >= u - VALID_MS)
        return (self.pb[ibc] + self.ps[isc]) / 2.0, ok

    def mid_stale(self, u):
        """Validity-FREE two-sided mid (any age) — shadow accounting only (D-h)."""
        ib = np.searchsorted(self.tb, u, "left") - 1
        is_ = np.searchsorted(self.ts, u, "left") - 1
        ok = (ib >= 0) & (is_ >= 0)
        ibc, isc = np.clip(ib, 0, None), np.clip(is_, 0, None)
        return (self.pb[ibc] + self.ps[isc]) / 2.0, ok

    def fwd_vwap(self, u, win_ms):
        """m̃: VWAP of sweeps in [u, u+win); returns (vwap, valid). §1.2 secondary."""
        i0 = np.searchsorted(self.ta, u, "left")
        i1 = np.searchsorted(self.ta, u + win_ms, "left")
        num = self._cpq[i1] - self._cpq[i0]
        den = self._cq[i1] - self._cq[i0]
        return num / np.maximum(den, EPS), den > EPS


# ---------------------------------------------------------------- tick size
def tick_size(sym: str) -> float:
    """Grid modulus of unique prices (§1.1). MODE of positive diffs, not min —
    results-audit finding: a handful of off-grid prints (liquidation-style, 81 on
    FIL) poison the min (FIL 1e-6 vs true 1e-4). The modal diff is the tick on any
    real tape; verified by the ≥99.9% integer-multiple check, GCD fallback kept."""
    uniq = set()
    for f in day_files(sym):
        uniq.update(np.unique(pd.read_parquet(f, columns=["price"]).to_numpy(float).ravel()))
    u = np.sort(np.fromiter(uniq, float))
    d = np.diff(u)
    d = d[d > EPS]
    scaled = np.round(d * 1e8).astype(np.int64)
    vals, cnts = np.unique(scaled, return_counts=True)
    tick = vals[cnts.argmax()] / 1e8                      # modal diff
    mult = d / tick
    frac_int = np.mean(np.abs(mult - np.round(mult)) < 1e-6 * np.maximum(mult, 1))
    if frac_int < 0.999:   # fall back to GCD of integer-scaled diffs
        g = 0
        for v in vals:
            g = math.gcd(g, int(v))
        tick = g / 1e8
    return float(tick)


# ---------------------------------------------------------------- bootstrap
def stationary_boot_ci(num, den, exp_block, B=BOOT_B, seed=BOOT_SEED, q=(2.5, 97.5)):
    """Politis–Romano stationary bootstrap (circular) on (num, den) pairs;
    percentile CI of the ratio-of-sums (weighted mean)."""
    n = len(num)
    if n < 4 or np.sum(den) <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    p_geo = 1.0 / exp_block
    stats = np.empty(B)
    for b in range(B):
        idx = np.empty(n, np.int64)
        i = 0
        while i < n:
            start = rng.integers(n)
            length = min(int(rng.geometric(p_geo)), n - i)
            idx[i:i + length] = (start + np.arange(length)) % n
            i += length
        s_den = num[idx].sum(), den[idx].sum()
        stats[b] = s_den[0] / s_den[1] if s_den[1] > 0 else np.nan
    lo, hi = np.nanpercentile(stats, q)
    return float(lo), float(hi)


# ---------------------------------------------------------------- per-day core
def process_day(sym, day, tick, mo_rows, sp_rows, in_rows, bin_store):
    t, p, qty, m = load_day(SRC / sym / f"{day}.parquet")
    sw = sweeps(t, p, qty, m)
    mp = MidProxy(sw)
    ts_, ps_, qs_, sgn = sw["t"], sw["p"], sw["Q"], sw["sign"]
    day0 = (ts_[0] // 86_400_000) * 86_400_000

    m0, v0 = mp.mid(ts_)
    es_half = sgn * (ps_ - m0) / np.maximum(m0, EPS) * 1e4          # §1.3c

    day_incoherent = False
    mo_taustore = {}
    for tau in TAUS_S:
        u1 = ts_ + tau * 1000
        m1, v1 = mp.mid(u1)
        mo = -sgn * (m1 - ps_) / ps_ * 1e4                           # §1.3b: rs ≡ MO
        lam = sgn * (m1 - m0) / np.maximum(m0, EPS) * 1e4
        vw, vvw = mp.fwd_vwap(u1, min(tau, 5) * 1000)
        mo_vw = -sgn * (vw - ps_) / ps_ * 1e4
        val = v0 & v1
        # identity check (§1.3c) on the common valid set, eq-weighted, all-side
        if val.sum() > 0:
            gap = abs(np.mean(mo[val]) - (np.mean(es_half[val]) - np.mean(lam[val])))
            if gap > INCOHERENT_BPS:
                day_incoherent = True
        mo_taustore[tau] = (mo, lam, val, mo_vw, vvw & v0)
        for side_name, mask in (("makerbid", sgn < 0), ("makerask", sgn > 0),
                                ("all", np.ones(len(sgn), bool))):
            for wname in ("eq", "notional"):
                w = (qs_ * ps_) if wname == "notional" else np.ones(len(sgn))
                sel = mask & val
                selv = mask & (vvw & v0)
                mo_rows.append({
                    "symbol": sym, "date": day, "side": side_name, "tau_s": tau,
                    "weighting": wname, "n_events": int(sel.sum()),
                    "valid_frac": float(val[mask].mean()) if mask.any() else np.nan,
                    "mo_bps": _wm(mo, w, sel), "es_half_bps": _wm(es_half, w, sel),
                    "lambda_bps": _wm(lam, w, sel), "mo_vwapmid_bps": _wm(mo_vw, w, selv),
                })
        if tau in (30, 60, 300):   # 30-min bin sums at all τ*-candidate points (D-g)
            bins = ((ts_ - day0) // 1_800_000).astype(int)
            sel = val
            for b in range(48):
                bsel = sel & (bins == b)
                bin_store.append((sym, day, tau, b, float(mo[bsel].sum()), float(bsel.sum())))

    # ---- (a) effective-spread flip pairs + Roll + pinned -----------------------
    flip = sgn[1:] != sgn[:-1]
    dt = ts_[1:] - ts_[:-1]
    es_stats = {}
    for dtmax in FLIP_MS:
        sel = flip & (dt <= dtmax)
        espair = sgn[:-1][sel] * (ps_[:-1][sel] - ps_[1:][sel])      # price units
        mflip, vflip = m0[:-1][sel], v0[:-1][sel]
        es_bps = espair[vflip] / np.maximum(mflip[vflip], EPS) * 1e4
        es_stats[dtmax] = (espair, es_bps)
    dp = np.diff(ps_)
    cov = np.cov(dp[:-1], dp[1:])[0, 1] if len(dp) > 2 else np.nan
    roll_bps = 2 * math.sqrt(-cov) / ps_.mean() * 1e4 if cov < 0 else np.nan
    ep100 = es_stats[100][0]
    pinned = bool(len(ep100) > 10 and np.median(ep100) <= 1.0 * tick
                  and np.percentile(ep100, 75) <= 2.0 * tick)
    e1s, e1bps = es_stats[1000]
    sp_rows.append({
        "symbol": sym, "date": day, "n_trades": len(t), "n_sweeps": len(ts_),
        "n_sweeps_buy": int((sgn > 0).sum()), "n_sweeps_sell": int((sgn < 0).sum()),
        "notional_usd": float((qs_ * ps_).sum()), "tick_size": tick,
        "tick_bps": tick / ps_.mean() * 1e4,
        "es_med_bps_1s": float(np.median(e1bps)) if len(e1bps) else np.nan,
        "es_trim_bps_1s": _trim_mean(e1bps, 0.01), "n_pairs_1s": len(e1bps),
        "es_med_bps_100ms": float(np.median(es_stats[100][1])) if len(es_stats[100][1]) else np.nan,
        "n_pairs_100ms": len(es_stats[100][1]), "roll_bps": roll_bps,
        "pinned_flag": pinned, "midvalid_frac_10s": float(v0.mean()),
        "proxy_incoherent_flag": day_incoherent,
        "es_med_price_1s": float(np.median(e1s)) if len(e1s) else np.nan,  # E1-A input
    })

    # ---- (d) intensity + size stats -------------------------------------------
    for side_name, mask in (("takerbuy", sgn > 0), ("takersell", sgn < 0)):
        tt = ts_[mask]
        notl = (qs_ * ps_)[mask]
        dts = np.diff(tt)
        opp = ts_[~mask]
        nxt = np.searchsorted(opp, ts_[mask], "right")
        topp = np.where(nxt < len(opp), opp[np.clip(nxt, 0, len(opp) - 1)] - tt, np.nan)
        Qs = qs_[mask]
        vals, cnts = np.unique(Qs, return_counts=True)
        modal = vals[cnts.argmax()] if len(vals) else np.nan
        ratio = Qs / max(modal, EPS)
        in_rows.append({
            "symbol": sym, "date": day, "side": side_name,
            "sweeps_per_s": len(tt) / 86_400,
            "dt_p50_ms": _pct(dts, 50), "dt_p90_ms": _pct(dts, 90), "dt_p99_ms": _pct(dts, 99),
            "burst_frac_100ms": float((dts < 100).mean()) if len(dts) else np.nan,
            "notional_p50": _pct(notl, 50), "notional_p90": _pct(notl, 90),
            "notional_p99": _pct(notl, 99),
            "round_atom_share": float(np.mean(np.abs(ratio - np.round(ratio)) < 1e-9)),
            "max_sweep_share": float(notl.max() / max(notl.sum(), EPS)) if len(notl) else np.nan,
            "t_opp_med_s": float(np.nanmedian(topp) / 1000) if len(topp) else np.nan,
        })
    return sw, mp


def _wm(x, w, sel):
    return float(np.average(x[sel], weights=w[sel])) if sel.sum() > 0 else np.nan


def _pct(x, p_):
    return float(np.percentile(x, p_)) if len(x) else np.nan


def _trim_mean(x, frac):
    if len(x) == 0:
        return np.nan
    lo, hi = np.percentile(x, [100 * frac, 100 * (1 - frac)])
    xx = x[(x >= lo) & (x <= hi)]
    return float(xx.mean()) if len(xx) else np.nan


# ---------------------------------------------------------------- E1-A overlay
def xs_overlap() -> pd.DataFrame:
    """Trailing-30d ADV top-40 from flow caches (deviation D-a: as of cache end)."""
    rows = []
    for f in glob.glob(str(REPO / "data/ml/cache/flow_*.parquet")):
        s = os.path.basename(f).replace("flow_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            d = d.iloc[-30 * 288:]                       # trailing 30 d of 5-min bars
            rows.append({"symbol": s,
                         "adv_usd": float((d["total_volume"] * d["vwap"]).sum() / 30)})
        except Exception:
            pass
    adv = pd.DataFrame(rows).sort_values("adv_usd", ascending=False).reset_index(drop=True)
    adv["adv_rank"] = adv.index + 1
    adv["in_top40"] = adv["adv_rank"] <= 40
    adv["in_pilot"] = adv["symbol"].isin(SYMS)
    return adv


def e1a_day(sym, day, sw, mp, es_day_price, tick, rows):
    ts_all, sgn = sw["t"], sw["sign"]
    day0 = (ts_all[0] // 86_400_000) * 86_400_000
    sells = sgn < 0        # taker-sell sweeps hit resting BIDS → can fill our buy
    buys = sgn > 0
    t_s, pmin_s = ts_all[sells], sw["pmin"][sells]
    t_b, pmax_b = ts_all[buys], sw["pmax"][buys]
    for tp in TP_GRID_S:
        hours = range(24 if tp * 1000 <= 3_600_000 else 23)
        for rule in ("touch", "sweep"):
            rec = {"fill": [], "chase": [], "drift": [], "as60": [], "n_skip": 0,
                   "chase_stale": []}
            for hh in hours:
                t0 = day0 + hh * 3_600_000
                mhat0, ok0 = mp.mid(np.array([t0]))
                if not ok0[0]:
                    rec["n_skip"] += 2
                    continue
                mhat0 = float(mhat0[0])
                for sign in (1.0, -1.0):                      # +1 buy, −1 sell
                    L_raw = mhat0 - sign * es_day_price / 2.0
                    kL = math.floor(L_raw / tick + 1e-9) if sign > 0 \
                        else math.ceil(L_raw / tick - 1e-9)
                    L = kL * tick
                    # D-i: compare on integer tick indices, not floats — else the
                    # touch (≤) and sweep-through (<) rules are indistinguishable.
                    if sign > 0:
                        tt = t_s
                        kpx = np.round(pmin_s / tick).astype(np.int64)
                        hit = (kpx <= kL) if rule == "touch" else (kpx < kL)
                    else:
                        tt = t_b
                        kpx = np.round(pmax_b / tick).astype(np.int64)
                        hit = (kpx >= kL) if rule == "touch" else (kpx > kL)
                    i0 = np.searchsorted(tt, t0, "right")
                    i1 = np.searchsorted(tt, t0 + tp * 1000, "right")
                    seg = hit[i0:i1]
                    if seg.any():
                        t_f = tt[i0 + int(np.argmax(seg))]
                        cost = sign * (L - mhat0) / mhat0 * 1e4 + FEE_MAKER["vip0"]
                        rec["fill"].append(cost)
                        m60, ok60 = mp.mid(np.array([t_f + 60_000]))
                        if ok60[0]:
                            rec["as60"].append(sign * (float(m60[0]) - L) / L * 1e4)
                    else:
                        mTp, okTp = mp.mid(np.array([t0 + tp * 1000]))
                        if not okTp[0]:
                            rec["n_skip"] += 1
                            # D-h shadow: book the skipped chase at the stale mid so the
                            # drop bias is bounded rather than silent.
                            mS, okS = mp.mid_stale(np.array([t0 + tp * 1000]))
                            if okS[0]:
                                p_xs = float(mS[0]) + sign * es_day_price / 2.0
                                rec["chase_stale"].append(
                                    sign * (p_xs - mhat0) / mhat0 * 1e4 + FEE_TAKER_VIP0)
                            continue
                        mTp = float(mTp[0])
                        p_x = mTp + sign * es_day_price / 2.0
                        rec["chase"].append(sign * (p_x - mhat0) / mhat0 * 1e4 + FEE_TAKER_VIP0)
                        rec["drift"].append(sign * (mTp - mhat0) / mhat0 * 1e4)
            nf, nc = len(rec["fill"]), len(rec["chase"])
            if nf + nc == 0:
                continue
            fr = nf / (nf + nc)
            cf = float(np.mean(rec["fill"])) if nf else 0.0
            cc = float(np.mean(rec["chase"])) if nc else 0.0
            eff_leg = fr * cf + (1 - fr) * cc
            # D-h shadow: same estimator with skipped chases included at stale mid
            ncs = nc + len(rec["chase_stale"])
            ccs = float(np.mean(rec["chase"] + rec["chase_stale"])) if ncs else 0.0
            frs = nf / (nf + ncs)
            eff_leg_stale = frs * cf + (1 - frs) * ccs
            rows.append({
                "symbol": sym, "date": day, "tp_s": tp, "fill_rule": rule,
                "n_episodes": nf + nc, "n_skipped": rec["n_skip"],
                "fill_rate": fr, "cost_fill_bps": cf if nf else np.nan,
                "chase_frac": 1 - fr, "cost_chase_bps": cc if nc else np.nan,
                "drift_nofill_bps": float(np.mean(rec["drift"])) if rec["drift"] else np.nan,
                "eff_leg_bps": eff_leg, "eff_rt_bps": 2 * eff_leg,
                "eff_rt_stale_bps": 2 * eff_leg_stale,
                "as60_fill_bps": float(np.mean(rec["as60"])) if rec["as60"] else np.nan,
            })


# ---------------------------------------------------------------- gates
def tau_star(intens: pd.DataFrame, sym: str) -> int:
    d = intens[intens.symbol == sym].groupby("date")["t_opp_med_s"].median()
    if (d <= 30).sum() >= 24:
        return 30
    cand = min(300, 2 * d.median())
    return 60 if cand <= 60 else 300


def build_gate_summary(mo: pd.DataFrame, sp: pd.DataFrame, intens: pd.DataFrame,
                       bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym in SYMS:
        ts = tau_star(intens, sym)
        excl = set(sp[(sp.symbol == sym) & (sp.proxy_incoherent_flag)]["date"])
        # MUST FIX 1: <50%-valid days excluded from ALL five gate conditions
        # (per-side means, bootstrap bins, VWAP check), not only mean/sign-frac.
        # ~(x >= 0.5) also catches NaN.
        cell = mo[(mo.symbol == sym) & (mo.side == "all") & (mo.weighting == "eq")
                  & (mo.tau_s == ts)]
        excl |= set(cell[~(cell.valid_frac >= 0.5)]["date"])
        sub = mo[(mo.symbol == sym) & (mo.side == "all") & (mo.weighting == "eq")
                 & (mo.tau_s == ts) & (~mo.date.isin(excl))]
        days = sub["date"].nunique()
        rs = sub.groupby("date")["mo_bps"].mean()
        per_side = {
            s: mo[(mo.symbol == sym) & (mo.side == s) & (mo.weighting == "eq")
                  & (mo.tau_s == ts) & (~mo.date.isin(excl))].groupby("date")["mo_bps"].mean().mean()
            for s in ("makerbid", "makerask")}
        vw = sub.groupby("date")["mo_vwapmid_bps"].mean().mean()
        # §1.4 bin bootstrap at the symbol's own τ* (bins stored for 30/60/300, D-g)
        bs = bins[(bins.symbol == sym) & (bins.tau == ts) & (~bins.date.isin(excl))]
        ci_lo, ci_hi = stationary_boot_ci(bs["num"].to_numpy(), bs["den"].to_numpy(),
                                          exp_block=8)
        mean_rs = float(rs.mean()) if days else np.nan
        sign_frac = float((rs > 0).mean()) if days else np.nan
        n_pos_days = int((rs > 0).sum()) if days else 0
        vw_agree = bool(np.sign(vw) == np.sign(mean_rs)) if np.isfinite(vw) and np.isfinite(mean_rs) else False
        conds = {
            tier: (mean_rs >= FEE_MAKER[tier] + C_SAFE) and (ci_lo >= FEE_MAKER[tier])
            and (n_pos_days >= 22)   # D-j: literal prereg count (≥22 of 31)
            and (per_side["makerbid"] > 0) and (per_side["makerask"] > 0) and vw_agree
            for tier in ("vip0", "vip1")}
        rows.append({
            "symbol": sym, "days_used": days, "tau_star_s": ts,
            "rs_taustar_bps": mean_rs, "rs_ci_lo": ci_lo, "rs_ci_hi": ci_hi,
            "rs_bid_bps": per_side["makerbid"], "rs_ask_bps": per_side["makerask"],
            "sign_frac_days": sign_frac, "vwap_sign_agree": vw_agree,
            "pinned_flag": bool(sp[sp.symbol == sym]["pinned_flag"].mean() > 0.5),
            "pass_vip0": conds["vip0"], "pass_vip1": conds["vip1"],
            "e1x_quarters_pos": "",   # E1x pending (§1.4)
            # D-f: a symbol with <10 usable days is a NO-ESTIMATE (mid-proxy
            # starvation), not a measured economic fail — §1.2 finality doesn't apply.
            "e1b_final": ("pending_e1x" if (conds["vip0"] or conds["vip1"])
                          else ("no_estimate" if days < 10 else "fail")),
        })
    return pd.DataFrame(rows)


def build_e1a_summary(e1a: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tp in TP_GRID_S:
        res = {}
        for rule in ("touch", "sweep"):
            sub = e1a[(e1a.tp_s == tp) & (e1a.fill_rule == rule)]
            if sub.empty:
                res[rule] = (np.nan, np.nan, np.nan)
                continue
            daymean = sub.groupby("date")["eff_rt_bps"].mean()      # eq-weight syms
            ci = stationary_boot_ci(daymean.to_numpy(),
                                    np.ones(len(daymean)), exp_block=3)
            res[rule] = (float(daymean.mean()), ci[0], ci[1])
        touch, swp = res["touch"][0], res["sweep"][0]
        verdict = ("PASS" if swp <= 8 else
                   "MARGINAL" if touch <= 8 < swp else "FAIL") if np.isfinite(touch) else "NA"
        rows.append({
            "tp_s": tp, "n_symbols": e1a[e1a.tp_s == tp]["symbol"].nunique(),
            "symbols_list": ";".join(sorted(e1a[e1a.tp_s == tp]["symbol"].unique())),
            "eff_rt_touch_bps": touch, "eff_rt_sweep_bps": swp,
            "ci_touch_lo": res["touch"][1], "ci_touch_hi": res["touch"][2],
            "ci_sweep_lo": res["sweep"][1], "ci_sweep_hi": res["sweep"][2],
            "verdict": verdict,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # E1-A universe written BEFORE any cost is computed (§1.5 prereg order)
    adv = xs_overlap()
    adv.to_csv(OUT / "e1a_universe.csv", index=False)
    overlap = sorted(set(adv[adv.in_top40]["symbol"]) & set(SYMS))
    print(f"[e1] XS-overlap (top-40 ∩ pilot, ADV as of flow-cache end — deviation D-a): "
          f"{len(overlap)}: {overlap}", flush=True)

    mo_rows, sp_rows, in_rows, e1a_rows = [], [], [], []
    bin_store = []
    for sym in SYMS:
        tick = tick_size(sym)
        files = day_files(sym)
        print(f"[e1] {sym}: {len(files)} days, tick={tick:g}", flush=True)
        for f in files:
            day = f.stem
            sw, mp = process_day(sym, day, tick, mo_rows, sp_rows, in_rows, bin_store)
            if sym in overlap:
                es_day = sp_rows[-1]["es_med_price_1s"]
                if np.isfinite(es_day):
                    e1a_day(sym, day, sw, mp, es_day, tick, e1a_rows)

    mo = pd.DataFrame(mo_rows)
    sp = pd.DataFrame(sp_rows)
    intens = pd.DataFrame(in_rows)
    bins = pd.DataFrame(bin_store, columns=["symbol", "date", "tau", "bin", "num", "den"])
    e1a = pd.DataFrame(e1a_rows)

    mo.drop(columns=[], inplace=True)
    sp.drop(columns=["es_med_price_1s"]).to_csv(OUT / "e1_spread_daily.csv", index=False)
    mo.to_csv(OUT / "e1_markout_daily.csv", index=False)
    intens.to_csv(OUT / "e1_intensity_daily.csv", index=False)
    e1a.to_csv(OUT / "e1a_overlay_daily.csv", index=False)

    gate = build_gate_summary(mo, sp, intens, bins)
    gate.to_csv(OUT / "e1_gate_summary.csv", index=False)
    e1a_sum = build_e1a_summary(e1a)
    e1a_sum.to_csv(OUT / "e1a_gate_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== E1-B gate summary ===")
    print(gate.to_string(index=False))
    print("\n=== E1-A overlay summary (gate row: tp_s=600) ===")
    print(e1a_sum.to_string(index=False))


if __name__ == "__main__":
    main()
