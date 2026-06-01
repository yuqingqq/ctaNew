"""LONG-PRED iter-032 — CLEAN gate calibration with 3-way split + transaction cost.

Split (no retrain — model trained <= 2025-10-02, all OOS):
  VAL     (calibrate T): H1a  2025-10-04 -> 2025-12-01
  INTERIM (check #1):     H1b  2025-12-01 -> 2026-01-22
  FINAL   (check #2):     H2   2026-01-22 -> 2026-05-26

Procedure:
  1. Sweep gate grid on VAL only. Objective = net-PnL Sharpe (after cost),
     subject to "position taken on >= 70% of cycles" (no degenerate tiny configs).
  2. Freeze the single best config.
  3. Report frozen config on VAL / INTERIM / FINAL.
  4. Show baseline K=5/K=5 on all three for reference.
  5. Show top-8 VAL configs ranked, with their INTERIM+FINAL — so we can SEE
     whether the VAL-optimal choice generalizes or is a VAL fluke.

Gate families:
  - z-gate:    include long if pred_z > +T_long; short if pred_z < -T_short
  - pct-gate:  include long if cycle-rank in top X%; short if in bottom Y%

Transaction cost: per-cycle cost = COST_RT_BPS * turnover_notional, where
turnover is the equal-weight book notional that rotates vs previous cycle.
Held names incur no cost (cost amortization is captured honestly).
Net-long cycles are hedged with an equal-weight basket short (market median).
"""
import sys, time, itertools
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"agents_system/research/outputs/iter025/preds_L1_V0_pooled.parquet"
OUT_DIR = REPO/"agents_system/research/outputs/iter032"; OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_S,  VAL_E  = pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2025-12-01",tz="UTC")
INT_S,  INT_E  = pd.Timestamp("2025-12-01",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC")
FIN_S,  FIN_E  = pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC")

COST_RT_BPS = 9.0          # round-trip cost on rotated notional (4.5/leg)
CYCLES_PER_YEAR = 6 * 365  # 4h cycles; annualization (same for all variants -> fair ranking)
MIN_ACTIVE_FRAC = 0.70     # config must take a position on >=70% of cycles
K_BASELINE = 5

def build_selection(gc, rule):
    """Return (long_df, short_df) for one cycle given a rule dict."""
    if rule["family"] == "K":
        return gc.nlargest(rule["K"], "pred"), gc.nsmallest(rule["K"], "pred")
    if rule["family"] == "z":
        return gc[gc["pred_z"] > rule["T_long"]], gc[gc["pred_z"] < -rule["T_short"]]
    if rule["family"] == "pct":
        n = len(gc)
        lo_rank = gc["pred"].rank(pct=True)
        longs = gc[lo_rank > (1 - rule["pct_long"])]
        shorts = gc[lo_rank < rule["pct_short"]]
        return longs, shorts
    raise ValueError(rule)

def evaluate(df_period, rule):
    """Per-cycle net PnL (bps) with turnover cost + basket hedge for net-long cycles."""
    rows = []
    prev_long_w, prev_short_w = {}, {}
    for ot, gc in df_period.groupby("open_time"):
        if len(gc) < 10:
            continue
        market_med = gc["return_pct"].median()
        longs, shorts = build_selection(gc, rule)
        nL, nS = len(longs), len(shorts)

        # equal-weight target books (total $1 each side)
        cur_long_w  = {s: 1.0/nL for s in longs["symbol"]}  if nL else {}
        cur_short_w = {s: 1.0/nS for s in shorts["symbol"]} if nS else {}

        # turnover notional (sum of positive weight deltas) on each side
        def turn(cur, prev):
            keys = set(cur) | set(prev)
            return sum(max(0.0, cur.get(k,0.0) - prev.get(k,0.0)) for k in keys)
        to_long  = turn(cur_long_w,  prev_long_w)
        to_short = turn(cur_short_w, prev_short_w)
        prev_long_w, prev_short_w = cur_long_w, cur_short_w

        long_ret  = longs["return_pct"].mean()  if nL else 0.0
        short_ret = shorts["return_pct"].mean() if nS else 0.0

        # gross PnL (dollar-neutral where both sides present; hedge net-long with basket)
        if nL and nS:
            gross = long_ret - short_ret
            hedge_notional = 0.0
        elif nL and not nS:
            gross = long_ret - market_med     # basket-short hedge
            hedge_notional = 1.0              # the hedge leg also turns over (approx full)
        elif nS and not nL:
            gross = -short_ret
            hedge_notional = 0.0
        else:
            gross = 0.0; hedge_notional = 0.0

        cost = COST_RT_BPS * (to_long + to_short + hedge_notional) / 1e4  # in return units
        net = gross - cost
        rows.append({"open_time": ot, "nL": nL, "nS": nS,
                     "gross": gross, "net": net,
                     "long_alpha": (long_ret-market_med) if nL else np.nan,
                     "short_alpha": (market_med-short_ret) if nS else np.nan,
                     "active": 1 if (nL or nS) else 0,
                     "turnover": to_long+to_short})
    return pd.DataFrame(rows)

def stats(res):
    net = res["net"].values * 1e4
    n = len(res)
    sharpe = (net.mean()/net.std()*np.sqrt(CYCLES_PER_YEAR)) if net.std()>0 else 0.0
    la = res["long_alpha"].dropna().values * 1e4
    sa = res["short_alpha"].dropna().values * 1e4
    return {
        "n": n,
        "active_frac": res["active"].mean(),
        "net_bps": net.mean(),
        "net_t": net.mean()/(net.std()/np.sqrt(n)) if net.std()>0 else 0.0,
        "sharpe": sharpe,
        "gross_bps": res["gross"].mean()*1e4,
        "avg_nL": res[res["nL"]>0]["nL"].mean(),
        "avg_nS": res[res["nS"]>0]["nS"].mean(),
        "long_alpha": la.mean() if len(la) else np.nan,
        "short_alpha": sa.mean() if len(sa) else np.nan,
        "avg_turnover": res["turnover"].mean(),
    }

def main():
    t0 = time.time()
    print("=== iter-032: CLEAN gate calibration (3-way split + cost) ===\n", flush=True)
    print(f"  VAL     {VAL_S.date()} -> {VAL_E.date()}")
    print(f"  INTERIM {INT_S.date()} -> {INT_E.date()}")
    print(f"  FINAL   {FIN_S.date()} -> {FIN_E.date()}")
    print(f"  cost={COST_RT_BPS} bps RT on rotated notional; min active frac={MIN_ACTIVE_FRAC}\n")

    preds = pd.read_parquet(PREDS)
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    preds["pred_z"] = preds.groupby("open_time")["pred"].transform(
        lambda x: (x - x.mean())/x.std() if x.std()>0 else 0.0)
    val = preds[(preds.open_time>=VAL_S)&(preds.open_time<VAL_E)].copy()
    inter = preds[(preds.open_time>=INT_S)&(preds.open_time<INT_E)].copy()
    fin = preds[(preds.open_time>=FIN_S)&(preds.open_time<FIN_E)].copy()
    print(f"  cycles: VAL={val.open_time.nunique()}  INT={inter.open_time.nunique()}  FIN={fin.open_time.nunique()}\n", flush=True)

    # ---- Build candidate grid ----
    rules = []
    for tl, ts in itertools.product([0.3,0.5,0.7,1.0], [1.0,1.5,2.0,2.5]):
        rules.append({"family":"z","T_long":tl,"T_short":ts,
                      "name":f"z(L>{tl},S<-{ts})"})
    for pl, ps in itertools.product([0.30,0.20,0.10], [0.10,0.05,0.025]):
        rules.append({"family":"pct","pct_long":pl,"pct_short":ps,
                      "name":f"pct(L top{int(pl*100)}%,S bot{ps*100:.1f}%)"})

    # ---- Calibrate on VAL ----
    print("=== Sweeping grid on VAL ===\n", flush=True)
    val_rows = []
    for r in rules:
        st = stats(evaluate(val, r))
        st["rule"] = r; st["name"] = r["name"]
        val_rows.append(st)
    vdf = pd.DataFrame(val_rows)
    # eligible = meets min active frac
    elig = vdf[vdf["active_frac"] >= MIN_ACTIVE_FRAC].copy()
    elig = elig.sort_values("sharpe", ascending=False)
    print(f"  {len(elig)}/{len(vdf)} configs meet active>={MIN_ACTIVE_FRAC:.0%}")
    print(f"\n  Top-8 VAL configs by net Sharpe:")
    print(f"  {'config':<26} {'actv':>5} {'#L':>5} {'#S':>5} {'net bps':>8} {'Sharpe':>7} {'L_a':>6} {'S_a':>6}")
    for _, r in elig.head(8).iterrows():
        print(f"  {r['name']:<26} {r['active_frac']*100:>4.0f}% {r['avg_nL']:>4.1f} {r['avg_nS']:>4.1f} "
              f"{r['net_bps']:>+7.2f} {r['sharpe']:>+6.2f} {r['long_alpha']:>+5.1f} {r['short_alpha']:>+5.1f}")

    if len(elig)==0:
        print("  No eligible config; aborting."); return
    best = elig.iloc[0]["rule"]
    print(f"\n  >>> FROZEN config (VAL argmax): {best['name']}\n", flush=True)

    # ---- Baseline + frozen on all three slices ----
    baseline = {"family":"K","K":K_BASELINE,"name":f"K={K_BASELINE}/{K_BASELINE}"}
    print("=== Out-of-sample verification ===\n")
    print(f"  {'slice':<9} {'config':<26} {'#L':>5} {'#S':>5} {'net bps':>8} {'net_t':>6} {'Sharpe':>7} {'L_a':>6} {'S_a':>6}")
    print("  "+"-"*92)
    for slabel, sl in [("VAL",val),("INTERIM",inter),("FINAL",fin)]:
        for r in [baseline, best]:
            st = stats(evaluate(sl, r))
            tag = "★" if abs(st["net_t"])>1.96 else " "
            print(f"  {slabel:<9} {r['name']:<26} {st['avg_nL']:>4.1f} {st['avg_nS']:>4.1f} "
                  f"{st['net_bps']:>+7.2f}{tag} {st['net_t']:>+5.2f} {st['sharpe']:>+6.2f} "
                  f"{st['long_alpha']:>+5.1f} {st['short_alpha']:>+5.1f}")
        print()

    # ---- Stability: top-8 VAL configs across all slices ----
    print("=== Stability: do top VAL configs generalize? (net Sharpe per slice) ===\n")
    print(f"  {'config':<26} {'VAL':>8} {'INTERIM':>9} {'FINAL':>8}")
    print("  "+"-"*54)
    for _, vr in elig.head(8).iterrows():
        r = vr["rule"]
        sv = stats(evaluate(val, r))["sharpe"]
        si = stats(evaluate(inter, r))["sharpe"]
        sf = stats(evaluate(fin, r))["sharpe"]
        mark = "  <== frozen" if r["name"]==best["name"] else ""
        print(f"  {r['name']:<26} {sv:>+7.2f} {si:>+8.2f} {sf:>+7.2f}{mark}")
    # baseline row
    bv = stats(evaluate(val, baseline))["sharpe"]
    bi = stats(evaluate(inter, baseline))["sharpe"]
    bf = stats(evaluate(fin, baseline))["sharpe"]
    print(f"  {'K=5/5 baseline':<26} {bv:>+7.2f} {bi:>+8.2f} {bf:>+7.2f}")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
