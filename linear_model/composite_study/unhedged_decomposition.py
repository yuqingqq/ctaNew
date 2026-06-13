"""Composite Study — does NOT beta-hedging (holding market risk) make sense?
(LOCKED, descriptive.) The D1 leak-free alpha model, traded as a portfolio:
  HEDGED   pnl = pos·alpha_beta            (market-neutral; the residual)
  UNHEDGED pnl = pos·raw_fwd               (hold market risk)
           decomposed = pos·alpha_beta(IDIO) + pos·beta·btc_fwd(MARKET)
Report Sharpe + maxDD for each, the % of unhedged PnL from MARKET vs IDIO,
the basket net-beta, and a passive-market benchmark over the same window.
Honest framing: any positive UNHEDGED number = sample market drift, not
skill (model has no market-timing edge — Steps 99/100/101). Production
LGBM unaffected.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
s94b = _imp("s94b", "linear_model/scripts/94b_info_ceiling_d1_grouped.py")
ANN = np.sqrt(365.0 * 6.0)
COST = s94.COST
OUTD = REPO / "linear_model/composite_study/results"


def stats(x):
    x = np.asarray(x, float)
    sh = x.mean()/x.std(ddof=1)*ANN if x.std(ddof=1) > 1e-12 else np.nan
    eq = np.cumsum(x)
    dd = (eq - np.maximum.accumulate(eq)).min()
    return float(sh), float(eq[-1]), float(dd)


def main():
    print("=" * 100, flush=True)
    print("  COMPOSITE STUDY — does NOT hedging beta (holding market risk) "
          "make sense?  (descriptive)", flush=True)
    print("=" * 100, flush=True)
    dec, syms, btc, pan = s94.build(universe_oi=False)
    LEAK = s94.LEAK
    FEATS = [c for c in dec.columns if c not in LEAK and
             pd.api.types.is_numeric_dtype(dec[c])]
    if "s_t" not in FEATS:
        FEATS.append("s_t")
    d = dec.dropna(subset=FEATS + ["tz", "alpha_beta"]).reset_index(drop=True)
    rid, _ = s94b.grouped_oof(d, FEATS)
    d["pred"] = rid
    d = d[~d["pred"].isna()].reset_index(drop=True)

    # raw fwd-4h + BTC fwd-4h → market part = beta·btc_fwd
    parts = []
    for sym in d.symbol.unique():
        c = s94.load_close(sym)
        if c is None:
            continue
        c = c.set_index("open_time")["close"]
        parts.append(pd.DataFrame({"symbol": sym, "open_time": c.index,
                                   "fwd4_raw": (c.shift(-48)/c-1.0).values}))
    fr = pd.concat(parts, ignore_index=True)
    fr["open_time"] = pd.to_datetime(fr["open_time"], utc=True)
    bdf = pd.DataFrame({"open_time": btc.index,
                        "btc4": (btc.shift(-48)/btc-1.0).values})
    d = d.merge(fr, on=["symbol", "open_time"], how="left").merge(
        bdf, on="open_time", how="left").dropna(subset=["fwd4_raw", "btc4"])

    d["pos"] = np.sign(d["pred"])
    d = d.sort_values(["symbol", "open_time"])
    d["dp"] = d.groupby("symbol")["pos"].diff().abs().fillna(d["pos"].abs())
    d["mkt"] = d["pos"] * d["beta_btc_pit"] * d["btc4"]      # market/beta part
    d["idio"] = d["pos"] * d["alpha_beta"]                    # idiosyncratic
    d["raw"] = d["pos"] * d["fwd4_raw"]                       # unhedged total
    d["c"] = d["dp"] * COST / 1e4

    # per-cycle equal-weight portfolio
    pt = d.groupby("open_time").agg(
        hedged=("idio", "mean"), market=("mkt", "mean"),
        unhedged=("raw", "mean"), cost=("c", "mean"),
        netbeta=("pos", lambda s: (d.loc[s.index, "pos"] *
                                   d.loc[s.index, "beta_btc_pit"]).mean()),
        btc4=("btc4", "first")).reset_index()
    H = (pt["hedged"] - pt["cost"]).to_numpy() * 1e4         # bps/cyc
    U = (pt["unhedged"] - pt["cost"]).to_numpy() * 1e4
    Mk = pt["market"].to_numpy() * 1e4
    Id = (pt["hedged"]).to_numpy() * 1e4
    # passive benchmark: just hold the basket's net-beta in BTC each cycle
    BM = (pt["netbeta"] * pt["btc4"]).to_numpy() * 1e4

    hsh, heq, hdd = stats(H)
    ush, ueq, udd = stats(U)
    msh, meq, mdd = stats(Mk)
    bsh, beq, bdd = stats(BM)
    print(f"\n  rows={len(d)} cycles={len(pt)} "
          f"avg basket net-beta = {pt['netbeta'].mean():+.3f} "
          f"(0=neutral; ≠0 = accidental, drifting market exposure)", flush=True)
    print(f"  sample BTC over window: cum 4h-fwd ≈ {pt['btc4'].sum()*100:+.0f}%"
          f" (the drift this 'strategy' would inherit)", flush=True)
    print(f"\n  {'book':24s} {'Sharpe':>7s} {'cumPnL_bps':>11s} "
          f"{'maxDD_bps':>10s}", flush=True)
    print(f"  {'HEDGED (market-neutral)':24s} {hsh:+7.2f} {heq:+11.0f} "
          f"{hdd:+10.0f}", flush=True)
    print(f"  {'UNHEDGED (hold mkt risk)':24s} {ush:+7.2f} {ueq:+11.0f} "
          f"{udd:+10.0f}", flush=True)
    print(f"  {'  ├ idiosyncratic part':24s} {hsh:+7.2f} {heq:+11.0f} "
          f"(= HEDGED — the real, sub-cost edge)", flush=True)
    print(f"  {'  └ MARKET/beta part':24s} {msh:+7.2f} {meq:+11.0f} "
          f"{mdd:+10.0f}  (undirected exposure)", flush=True)
    print(f"  {'passive net-beta in BTC':24s} {bsh:+7.2f} {beq:+11.0f} "
          f"{bdd:+10.0f}  (zero-skill benchmark)", flush=True)
    share = meq / ueq * 100 if ueq != 0 else float('nan')
    v = (f"UNHEDGED Sharpe {ush:+.2f} (vs HEDGED {hsh:+.2f}); {share:.0f}% of "
         f"unhedged cumPnL is the MARKET/beta part (idiosyncratic adds the "
         f"same sub-cost ≈{hsh:+.2f} either way). The unhedged 'edge' is "
         f"≈ the zero-skill passive-net-beta benchmark ({bsh:+.2f}) ⇒ NOT "
         f"skill — it is uncompensated, drifting market exposure = this "
         f"sample's BTC drift, which inverts in a down market (maxDD "
         f"{udd:+.0f} bps). Not hedging does NOT rescue the strategy: it "
         f"adds undirected market risk to an already-sub-cost idiosyncratic "
         f"signal. 'Hold market risk' only makes sense as a separate, "
         f"explicit beta-allocation decision (then you'd just hold the index "
         f"— the model is irrelevant and cannot time it: Steps 99/100/101). "
         f"Production LGBM unaffected.")
    print(f"\n  VERDICT: {v}", flush=True)
    pd.DataFrame([dict(hedged_sh=hsh, unhedged_sh=ush, market_sh=msh,
                       passive_beta_sh=bsh, mkt_share_pct=share,
                       avg_netbeta=pt['netbeta'].mean(),
                       unhedged_maxdd=udd, verdict=v)]).to_csv(
        OUTD/"unhedged_decomposition.csv", index=False)
    print(f"\nSaved {OUTD}/unhedged_decomposition.csv", flush=True)


if __name__ == "__main__":
    main()
