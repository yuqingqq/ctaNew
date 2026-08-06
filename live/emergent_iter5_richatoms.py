"""iter5 (capstone): the RICHER atom set — do trade-microstructure features, TOGETHER with book/flow,
add a stable orthogonal factor or any both-era incremental signal? (the user's literal "many features".)

Book/flow atoms were used in iter1-4. Here add the more-orthogonal TRADE-side atoms (tfi=order-flow
imbalance, kyle_lambda=impact, vpin=toxicity, signed_volume_z, avg_trade_size) from the ext panel.
(1) STRUCTURE: effdim of book/flow-11 vs full-16 — do trade atoms open new dimensions or collapse in?
(2) INCREMENTAL: partial-IC of each trade atom vs fwd (5m,30m) controlling the WHOLE price+book/flow set,
    both eras, day-cluster CI, Bonferroni. If null => the orthogonal atoms add no harvestable info together.

Run:  python3 -m live.emergent_iter5_richatoms
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from live.flow_harness import CUT, ci, partial_xsic
from live.emergent_harness import EXT, clean_std, corr_spectrum, participation_ratio

BOOKFLOW = ["imb1", "ask_bid_ratio", "imb_change_5min", "bid_change_5min", "ask_change_5min",
            "buy_to_ask_5min", "sell_to_bid_5min", "signed_pressure_5min",
            "impact_bps_per_pressure_5min", "ask_depth_residual_5min", "bid_depth_residual_5min"]
TRADE = ["tfi", "kyle_lambda", "vpin", "signed_volume_z", "avg_trade_size"]
CONTROLS = ["tr_5m", "tr_30m", "imb1", "signed_pressure_5min",
            "ask_depth_residual_5min", "bid_depth_residual_5min",
            "buy_to_ask_5min", "sell_to_bid_5min"]
CUTv = np.datetime64(CUT.tz_convert(None))
MINR = 800


def load_ext(cols):
    files = sorted(glob.glob(f"{EXT}/*.parquet"))
    fr = []
    for f in files:
        x = pd.read_parquet(f, columns=cols)
        for c in x.columns:
            if x[c].dtype == np.float64:
                x[c] = x[c].astype(np.float32)
        fr.append(x)
    d = pd.concat(fr, ignore_index=True)
    d["bar_time"] = pd.to_datetime(d["bar_time"], utc=True)
    return d


def structure():
    print("=== (1) STRUCTURE: does adding 5 trade atoms open new dimensions? ===", flush=True)
    files = sorted(glob.glob(f"{EXT}/*.parquet"))
    full = BOOKFLOW + TRADE
    rows = []
    for f in files:
        d = pd.read_parquet(f, columns=["bar_time", *full])
        bt = pd.to_datetime(d["bar_time"], utc=True).to_numpy("datetime64[ns]")
        for era in ("OOS", "REC"):
            m = (bt < CUTv) if era == "OOS" else (bt >= CUTv)
            Xbf = d.loc[m, BOOKFLOW].to_numpy(float)
            Xfl = d.loc[m, full].to_numpy(float)
            ok_bf = np.isfinite(Xbf).all(axis=1)
            ok_fl = np.isfinite(Xfl).all(axis=1)
            if ok_bf.sum() < MINR or ok_fl.sum() < MINR:
                continue
            _, wbf, _ = corr_spectrum(clean_std(Xbf[ok_bf]))
            _, wfl, Vfl = corr_spectrum(clean_std(Xfl[ok_fl]))
            rows.append({"era": era, "eff_bf": participation_ratio(wbf),
                         "eff_full": participation_ratio(wfl)})
    R = pd.DataFrame(rows)
    for era in ("OOS", "REC"):
        e = R[R.era == era]
        print(f"  {era}: effdim book/flow-11 {e['eff_bf'].median():.2f} → full-16 "
              f"{e['eff_full'].median():.2f}  (Δ {e['eff_full'].median()-e['eff_bf'].median():+.2f}; "
              f"+5 would mean fully independent, ~0 fully redundant)", flush=True)


def incremental():
    print("\n=== (2) INCREMENTAL: partial-IC of each trade atom vs fwd, controlling price+book/flow ===",
          flush=True)
    D = load_ext(["symbol", "bar_time", *set(TRADE + CONTROLS), "fwd_5m", "fwd_30m"])
    print(f"  panel {len(D):,} rows | Bonferroni {len(TRADE)*2} tests → per-test α≈{0.05/(len(TRADE)*2):.4f}\n",
          flush=True)
    m_oos = (D["bar_time"] < CUT).to_numpy()
    m_rec = (D["bar_time"] >= CUT).to_numpy()
    print(f"  {'trade atom':<18}{'h':<6}{'OOS partial [95% CI]':<30}{'REC partial [95% CI]':<30}both", flush=True)
    for feat in TRADE:
        ctrl = [c for c in CONTROLS if c != feat]
        for h in ("fwd_5m", "fwd_30m"):
            ao, lo, uo = ci(partial_xsic(D, feat, ctrl, h, row_mask=m_oos))
            ar, lr, ur = ci(partial_xsic(D, feat, ctrl, h, row_mask=m_rec))
            bo = (lo > 0 or uo < 0); br = (lr > 0 or ur < 0)
            both = "YES" if (bo and br and np.sign(ao) == np.sign(ar)) else "no"
            print(f"  {feat:<18}{h:<6}{f'{ao:+.4f}[{lo:+.4f},{uo:+.4f}]':<30}"
                  f"{f'{ar:+.4f}[{lr:+.4f},{ur:+.4f}]':<30}{both}", flush=True)


def main():
    structure()
    incremental()


if __name__ == "__main__":
    main()
