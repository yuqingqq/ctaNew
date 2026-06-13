"""R1 — Honest curated baseline + concentration frontier.

Reuses the validated V3.1 production machinery from
scripts/phase_ah_sleeve.py (rolling-IC universe, K=3 conv_gate +
filter_refill + PM_M2 protocol, 6 equal-weight overlapping 24h sleeves)
and adds, per PLAN.md v3:
  - per-name PnL attribution (gross & cost)
  - concentration CAP SWEEP c in {inf, 1/2, 1/3, 1/5, 1/8 of book gross},
    truncate-and-redistribute -> Sharpe-vs-c frontier (primary deliverable)
  - secondary vol-normalized sizing variant
  - 3 cost modes: flat 4.5, flat 9 (stress), realized sqrt(ADV)
  - moving-block bootstrap (block = ceil(hold/4h)+(n_sleeves-1)=11),
    one-sided 95% LCB, N_eff, 80%-power MDE  (reported as info)
  - drop-5 random-symbol robustness (30 draws)  (reported as info)

Pre-registered prediction (PLAN.md R1): uncapped Sharpe in [+1.5,+2.6],
Herfindahl >= 0.40; frontier monotone-decreasing in tightness; at c=1/3
Sharpe in [+0.6,+1.8]; drop-5 mean <= +1.2. Deployable iff some c clears
criteria 1-6 (see PLAN.md). Misses rewrite the Diagnosis, not the gate.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import phase_ah_sleeve as PA  # validated V3.1 machinery
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUT = REPO / "research/portable_alpha_2026-05-19/results"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = REPO / "research/portable_alpha_2026-05-19/results/_cache"
CACHE.mkdir(parents=True, exist_ok=True)

APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES = REPO / "data/ml/test/parquet/klines"
HE, HB, NSL = PA.HORIZON_ENTRY, PA.HOLD_BARS, PA.N_SLEEVES   # 48, 288, 6
CPY = PA.CYCLES_PER_YEAR
COST_UNIT = PA.COST_PER_UNIT_ABS_DELTA                       # 2.25 bps
BLOCK = (HB // HE) + (NSL - 1)                               # 6 + 5 = 11
CAPS = [np.inf, 0.5, 1/3, 0.2, 0.125]
SIG_WIN = 288        # trailing realized-vol window for vol-norm sizing
N_DROP5 = 30


def _sharpe(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.mean() / x.std() * np.sqrt(CPY)) if len(x) > 1 and x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net); return float((cum - np.maximum.accumulate(cum)).min())


def _herfindahl(sym_pnl: dict) -> float:
    a = np.array([abs(v) for v in sym_pnl.values()], float)
    s = a.sum()
    return float(((a / s) ** 2).sum()) if s > 0 else np.nan


def _gini(sym_pnl: dict) -> float:
    a = np.sort(np.array([abs(v) for v in sym_pnl.values()], float))
    n = len(a); s = a.sum()
    if n == 0 or s == 0: return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * s))


def _apply_cap(tw: dict, cap_frac: float) -> dict:
    """Cap |w_s| <= cap_frac * gross, truncate-and-redistribute pro-rata to
    uncapped names on the SAME side. Gross held at the uncapped value."""
    if not np.isfinite(cap_frac) or not tw:
        return dict(tw)
    gross = sum(abs(v) for v in tw.values())
    if gross <= 0:
        return dict(tw)
    cap = cap_frac * gross
    out = dict(tw)
    for sign in (+1, -1):
        side = {s: w for s, w in out.items() if (w > 0) == (sign > 0) and w != 0}
        for _ in range(50):
            over = {s: w for s, w in side.items() if abs(w) > cap + 1e-12}
            if not over:
                break
            excess = sum(abs(w) - cap for w in over.values())
            for s in over:
                side[s] = sign * cap
            free = {s: w for s, w in side.items() if abs(w) < cap - 1e-12}
            base = sum(abs(w) for w in free.values())
            if base <= 1e-12:                       # nowhere to put it -> drop
                break
            for s, w in free.items():
                side[s] = w + sign * excess * (abs(w) / base)
        out.update(side)
    return out


def aggregate_capped(records, fwd_rets, sigma_wide, adv_unit_cost,
                     cap_frac=np.inf, sizing="equal", cost_mode="flat45"):
    """V3.1 6-sleeve overlap with optional per-name cap, sizing, cost mode.
    Returns (per_cycle_df, sym_gross_pnl, sym_cost)."""
    bar = pd.Timedelta(minutes=5)
    queue = deque(maxlen=NSL)
    prev = {}
    rows = []
    sym_g = defaultdict(float)
    sym_c = defaultdict(float)
    flat9 = (cost_mode == "flat9")
    realized = (cost_mode == "realized")
    for _, rec in records.iterrows():
        t, fold = rec["time"], rec["fold"]
        lb, sb = rec["long_basket"], rec["short_basket"]
        if lb and sb:
            queue.append({"entry_time": t, "longs": lb, "shorts": sb})
        queue = deque([s for s in queue if (t - s["entry_time"]) < HB * bar],
                      maxlen=NSL)
        tw = defaultdict(float)
        sw = 1.0 / NSL
        for sl in queue:
            L, S = sl["longs"], sl["shorts"]
            if not L or not S:
                continue
            if sizing == "volnorm":
                sg = sigma_wide.loc[sl["entry_time"]] if sl["entry_time"] in sigma_wide.index else None
                def _w(names):
                    iv = {}
                    for s in names:
                        v = (sg.get(s, np.nan) if sg is not None else np.nan)
                        iv[s] = 1.0 / v if (v == v and v > 0) else np.nan
                    if all(np.isnan(list(iv.values()))):
                        return {s: 1.0 / len(names) for s in names}
                    md = np.nanmedian([x for x in iv.values() if x == x])
                    iv = {s: (x if x == x else md) for s, x in iv.items()}
                    tot = sum(iv.values())
                    return {s: iv[s] / tot for s in names}
                wl, ws = _w(L), _w(S)
                for s in L: tw[s] += sw * wl[s]
                for s in S: tw[s] -= sw * ws[s]
            else:
                for s in L: tw[s] += sw * (1.0 / len(L))
                for s in S: tw[s] -= sw * (1.0 / len(S))
        tw = _apply_cap(dict(tw), cap_frac)
        gross_pnl = 0.0
        if t in fwd_rets.index:
            r = fwd_rets.loc[t]
            for s, w in prev.items():
                rv = r.get(s, np.nan)
                if rv == rv:
                    p = w * rv * 1e4
                    gross_pnl += p
                    sym_g[s] += p
        alls = set(tw) | set(prev)
        cost = 0.0
        for s in alls:
            d = abs(tw.get(s, 0.0) - prev.get(s, 0.0))
            if d == 0:
                continue
            if realized:
                u = adv_unit_cost.get(s, COST_UNIT)
            elif flat9:
                u = COST_UNIT * 2.0
            else:
                u = COST_UNIT
            c = d * u
            cost += c
            sym_c[s] += c
        net = gross_pnl - cost
        rows.append({"time": t, "fold": fold, "gross_pnl_bps": gross_pnl,
                     "cost_bps": cost, "net_pnl_bps": net,
                     "gross_exposure": sum(abs(v) for v in tw.values()),
                     "n_symbols": len(tw)})
        prev = dict(tw)
    return pd.DataFrame(rows), dict(sym_g), dict(sym_c)


def metrics(df, sym_g):
    net = df["net_pnl_bps"].to_numpy()
    sh = _sharpe(net)
    try:
        _, lo, hi = block_bootstrap_ci(net, statistic=_sharpe,
                                       block_size=BLOCK, n_boot=2000)
    except Exception:
        lo = hi = np.nan
    fold_sum = df.groupby("fold")["net_pnl_bps"].sum()
    fp = int((fold_sum > 0).sum())
    n = len(net)
    n_eff = n / BLOCK
    mde = 2.49 * np.sqrt((1 + 0.5 * sh ** 2) / max(n_eff, 1)) * np.sqrt(CPY) \
        if n_eff > 0 else np.nan      # ~80% power, one-sided-ish heuristic
    return {"sharpe": round(sh, 3), "boot_lcb": round(float(lo), 3),
            "boot_hcb": round(float(hi), 3), "folds_pos": fp,
            "n_folds": int(df["fold"].nunique()),
            "totPnL": round(float(net.sum()), 0),
            "maxDD": round(_max_dd(net), 0),
            "gross_avg": round(float(df["gross_pnl_bps"].mean()), 3),
            "cost_avg": round(float(df["cost_bps"].mean()), 3),
            "herfindahl": round(_herfindahl(sym_g), 4),
            "gini": round(_gini(sym_g), 4),
            "n_eff": round(n_eff, 1), "mde_sharpe": round(float(mde), 3),
            "top1_pnl_share": round(float(max(
                (abs(v) for v in sym_g.values()), default=0) /
                max(sum(abs(v) for v in sym_g.values()), 1e-9)), 4)}


def build_caches(apd, panel_syms):
    cw_p = CACHE / "close_wide.parquet"
    if cw_p.exists():
        close_wide = pd.read_parquet(cw_p)
    else:
        frames = []
        for sym in panel_syms:
            d = KLINES / sym / "5m"
            if not d.exists():
                continue
            fs = sorted(d.glob("*.parquet"))
            if not fs:
                continue
            parts = []
            for f in fs:
                try:
                    parts.append(pd.read_parquet(f, columns=["open_time", "close",
                                                             "quote_volume"]))
                except Exception:
                    pass
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True)
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
            df = (df.dropna(subset=["open_time"]).drop_duplicates("open_time")
                    .set_index("open_time"))
            frames.append(df.rename(columns={"close": f"c_{sym}",
                                              "quote_volume": f"q_{sym}"}))
        close_wide = pd.concat(frames, axis=1).sort_index()
        close_wide.to_parquet(cw_p)
    cs = [c for c in close_wide.columns if c.startswith("c_")]
    qs = [c for c in close_wide.columns if c.startswith("q_")]
    px = close_wide[cs].rename(columns=lambda x: x[2:])
    qv = close_wide[qs].rename(columns=lambda x: x[2:])
    fwd = (px.shift(-HE) - px) / px
    ret5 = px.pct_change()
    sigma = ret5.rolling(SIG_WIN, min_periods=48).std().shift(1)
    flo = sigma.quantile(0.20)
    sigma = sigma.clip(lower=flo, axis=1)
    # realized per-unit cost: leg_cost_s = max(0.5, k/sqrt(ADV30_$M)); calibrate
    adv = qv.rolling(288 * 30, min_periods=288).mean().shift(1) / 1e6  # $M ADV
    inv = 1.0 / np.sqrt(adv.replace(0, np.nan))
    med = np.nanmedian(inv.to_numpy())
    k = 4.5 / med if med and med == med else 4.5
    leg = (k * inv).clip(lower=0.5)
    # per-symbol per-unit-abs-delta cost = 0.5 * leg_cost (matches COST_UNIT calib)
    unit = 0.5 * leg
    adv_unit_cost = {s: float(np.nanmedian(unit[s])) for s in unit.columns}
    return fwd, sigma, adv_unit_cost


def main():
    t0 = time.time()
    print("R1 — honest baseline + concentration frontier", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = PA.get_listings()

    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= ts}

    tt = sorted(apd[apd["fold"].isin(PA.OOS_FOLDS)]["open_time"].unique())
    sampled = tt[::HE]
    print(f"  building universe + base records ...", flush=True)
    universe = PA.build_rolling_ic_universe(apd, sampled, PA.TOP_N, elig)
    records = PA.run_production_protocol_save_sleeves(apd, universe)
    records.to_parquet(CACHE / "base_records.parquet", index=False)
    print(f"  {len(records)} cycles, {records['traded'].sum()} traded", flush=True)
    fwd, sigma, advc = build_caches(apd, panel_syms)
    print(f"  caches ready ({time.time()-t0:.0f}s); BLOCK={BLOCK}", flush=True)

    # ---- frontier: caps x sizing x cost ---------------------------------
    rows = []
    cap_name = {np.inf: "inf", 0.5: "1/2", 1/3: "1/3", 0.2: "1/5", 0.125: "1/8"}
    grid = ([("equal", "flat45", c) for c in CAPS] +
            [("equal", "flat9", c) for c in CAPS] +
            [("equal", "realized", c) for c in CAPS] +
            [("volnorm", "flat45", c) for c in CAPS])
    unc = {}
    for sizing, cost, c in grid:
        df, sg, sc = aggregate_capped(records, fwd, sigma, advc,
                                      cap_frac=c, sizing=sizing, cost_mode=cost)
        m = metrics(df, sg)
        m.update({"sizing": sizing, "cost_mode": cost, "cap": cap_name[c]})
        if c == np.inf:
            unc[(sizing, cost)] = m["sharpe"]
        base = unc.get((sizing, cost), np.nan)
        m["cap_retention"] = round(m["sharpe"] / base, 3) if base and base > 0 else np.nan
        rows.append(m)
        print(f"  [{sizing:>7}|{cost:>8}|cap {cap_name[c]:>3}] "
              f"Sh={m['sharpe']:+.2f} LCB={m['boot_lcb']:+.2f} "
              f"f+={m['folds_pos']}/{m['n_folds']} DD={m['maxDD']:+.0f} "
              f"H={m['herfindahl']:.3f} top1={m['top1_pnl_share']:.2f} "
              f"ret={m['cap_retention']}", flush=True)
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "R1_frontier.csv", index=False)

    # ---- drop-5 robustness (equal, flat45) on each cap ------------------
    print(f"\n  drop-5 robustness ({N_DROP5} draws) ...", flush=True)
    rng = np.random.RandomState(20260519)
    d5 = {cap_name[c]: [] for c in CAPS}
    for i in range(N_DROP5):
        drop = set(rng.choice(panel_syms, 5, replace=False))
        keep = [s for s in panel_syms if s not in drop]
        a2 = apd[apd["symbol"].isin(keep)].copy()
        u2 = PA.build_rolling_ic_universe(a2, sampled, PA.TOP_N, elig)
        r2 = PA.run_production_protocol_save_sleeves(a2, u2)
        for c in CAPS:
            df, sg, _ = aggregate_capped(r2, fwd, sigma, advc, cap_frac=c,
                                         sizing="equal", cost_mode="flat45")
            d5[cap_name[c]].append(_sharpe(df["net_pnl_bps"].to_numpy()))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{N_DROP5} ({time.time()-t0:.0f}s)", flush=True)
    d5stat = {k: {"mean": round(float(np.mean(v)), 3),
                  "worst": round(float(np.min(v)), 3),
                  "std": round(float(np.std(v)), 3)} for k, v in d5.items()}

    out = {"block": BLOCK, "frontier": rows, "drop5": d5stat,
           "uncapped_sharpe": {f"{s}|{c}": v for (s, c), v in unc.items()},
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "R1_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  drop5(equal,flat45): " +
          " ".join(f"cap{kk}: mean{vv['mean']:+.2f}/worst{vv['worst']:+.2f}"
                   for kk, vv in d5stat.items()), flush=True)
    print(f"\nR1 done [{out['elapsed_s']}s] -> {OUT}/R1_frontier.csv", flush=True)


if __name__ == "__main__":
    main()
