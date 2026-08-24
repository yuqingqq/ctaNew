"""User's objection (fair): absorp_net is NOT return_1d — it carries the book's depth dynamics, info return_1d lacks.
So "redundant" must be proven, not asserted. DIRECT test: strip the price/momentum component OUT of absorp_net
(per-bar cross-sectional OLS residual), then measure the rank-IC of the PURE ob-reaction residual on forward alpha,
both eras. Three controls sets:
  raw            absorp_net itself (baseline: ~-0.006 both eras)
  | momentum     residualize on [return_1d, ret_3d]  -> "ob reaction beyond price performance" (the user's exact ask)
  | full V0      residualize on all 14 V0_LEAN feats  -> does ANYTHING survive the whole model's price/vol/beta span?
If | momentum -> 0 : the signal WAS momentum; the pure reaction has no alpha (redundancy confirmed at the user's level).
If | full V0 stays same-sign + CI-off-zero BOTH eras : the reaction carries alpha orthogonal to V0 -> user is right,
   the Ridge incremental null needs reconciling (escalate).
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_absorption import build_sym
from live.bookdepth_persist import day_boot
import live.train_twobook_models as tt
rng = np.random.default_rng(41)
V0 = list(tt.V0_LEAN); MOM = ["return_1d", "ret_3d"]; TGT = "alpha_vs_btc_realized"

def _perbar_resid_ic(g, feat, controls, tgt):
    gg = g[[feat, tgt] + controls].dropna()
    if len(gg) < len(controls) + 6: return np.nan
    X = np.column_stack([np.ones(len(gg))] + [gg[c].values for c in controls])
    y = gg[feat].values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ beta
    except Exception:
        return np.nan
    return pd.Series(r, index=gg.index).corr(gg[tgt], method="spearman")

def ic(sub, feat, controls, tgt):
    if controls is None:
        s = sub.groupby("open_time").apply(lambda g: g[feat].corr(g[tgt], method="spearman")
                                           if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()
    else:
        s = sub.groupby("open_time").apply(lambda g: _perbar_resid_ic(g, feat, controls, tgt)).dropna()
    if len(s) < 5: return (np.nan, np.nan, np.nan)
    lo, up = day_boot(s); return (s.mean(), lo, up)

def main():
    rows = []
    files = [f for f in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in f]
    for i, f in enumerate(files):
        o = build_sym(f)
        if o is not None: rows.append(o[["symbol", "open_time", "absorp_net"]])
        if (i + 1) % 40 == 0: print(f"  built {i+1}/{len(files)}", flush=True)
    L = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet(tt.PANEL, columns=["symbol", "open_time", TGT] + V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = pan.merge(L, on=["symbol", "open_time"], how="inner")
    cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"\nmerged {len(m)} | {m.symbol.nunique()} syms | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print("Does the PURE ob-reaction (absorp_net, price-performance stripped out) predict fwd ALPHA?\n")
    print(f"{'control set':16s} | {'RECENT resid-IC [CI]':26s} | {'OOS resid-IC [CI]':26s} | survives?")
    for lab, ctrl in [("raw (none)", None), ("| momentum(ret1,3)", MOM), ("| full V0 (14)", V0)]:
        (ra, rl, ru) = ic(eras["RECENT"], "absorp_net", ctrl, TGT)
        (oa, ol, ou) = ic(eras["OOS"], "absorp_net", ctrl, TGT)
        surv = "YES" if (np.sign(ra) == np.sign(oa) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
        print(f"{lab:16s} | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {surv}")
    print("\nread: if | momentum -> ~0, the pure reaction has no alpha (was momentum). If | full V0 survives both eras,")
    print("the reaction carries orthogonal alpha and the incremental null needs reconciling. ABSPARTIALDONE")

if __name__ == "__main__":
    main()
