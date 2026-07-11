"""event_switch_test (user Q: can we DETECT the era/event and SWITCH to avoid the losing era?).
Two questions, both eras, on the clean 1L/2S book (daily net, pinned cost):
 (1) PREDICTABILITY — does any OBSERVABLE, point-in-time state (trailing book perf, btc_ret_30d,
     btc_rvol_7d, cross-sectional dispersion) predict the NEXT day's book PnL? If not, no detector can
     switch us out of a loss ahead of time. (Spearman corr per feature + combined OLS R² + sign hit-rate.)
 (2) SWITCH — does a reactive "go flat when the detector says bad" rule beat always-on in BOTH eras
     without killing the good era? (trailing-perf momentum switch; high-vol risk-off switch.)
"""
import numpy as np, pandas as pd, glob, sys
from pathlib import Path
sys.path.insert(0, "live")
from attribution_v4_regime import btc_reg, load, COST
import warnings; warnings.filterwarnings("ignore")
KD = Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines"); WL = WS = 0.5

def btc_ctx():
    fs = sorted(glob.glob(str(KD / "BTCUSDT" / "5m" / "*.parquet")))
    b = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in fs], ignore_index=True)
    b["open_time"] = pd.to_datetime(b["open_time"], utc=True)
    b = b.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")["close"]
    d = b.resample("1D").last(); ret = d.pct_change()
    out = pd.DataFrame({"btc_30d": (d / d.shift(30) - 1), "btc_rvol7": ret.rolling(7).std()})
    out.index = out.index.normalize(); return out

def book_daily(base, long, reg):
    lg = long.groupby("open_time"); rows = []; prevL = None; prevS = set()
    for t, g in base.groupby("open_time"):
        if len(g) < 5 or reg.get(t) in (None, "deepbull"): continue
        try: gl = lg.get_group(t)
        except KeyError: continue
        Lp = gl.nlargest(1, "pred"); S = g.nsmallest(2, "pred")
        if len(Lp) < 1 or len(S) < 2: continue
        la = Lp.iloc[0]["alpha_A"] * 1e4; sa = S["alpha_A"].mean() * 1e4
        disp = float(g["return_pct"].std() * 1e4)
        Ln, Ss = Lp.iloc[0]["symbol"], set(S["symbol"])
        lt = 1.0 if (prevL is None or Ln != prevL) else 0.0
        st = (len(Ss - prevS) / 2.0) if prevS else 1.0
        net = WL * la - WS * sa - lt * 0.5 * WL * COST / 0.5 - st * 0.5 * WS * COST / 0.5
        rows.append((pd.Timestamp(t).normalize(), net, disp)); prevL, prevS = Ln, Ss
    return pd.DataFrame(rows, columns=["day", "net", "disp"]).groupby("day").agg(net=("net", "sum"), disp=("disp", "mean"))

def sharpe(x):
    x = x[np.isfinite(x)]; return x.mean() / x.std() * np.sqrt(365) if len(x) and x.std() > 0 else np.nan

def main():
    reg = btc_reg(); ctx = btc_ctx()
    for era, bp, lp in (("RECENT", "hl_tgt_res_base_cleanfix", "hl_tgt_res_long_cleanfix"),
                        ("OOS", "hl_v4base_oos_cleanfix", "hl_v4long_oos_cleanfix")):
        d = book_daily(*load(bp, lp), reg).join(ctx, how="left")
        # PIT observable detectors (all lagged 1 day → knowable before today's book)
        d["trail10"] = d["net"].rolling(10, min_periods=5).mean().shift(1)
        d["btc30"] = d["btc_30d"].shift(1); d["rvol"] = d["btc_rvol7"].shift(1); d["disp_l"] = d["disp"].shift(1)
        feats = ["trail10", "btc30", "rvol", "disp_l"]
        dd = d.dropna(subset=feats + ["net"]).copy()
        print(f"\n===== {era}: {len(dd)} days | book Sharpe {sharpe(dd['net'].values):+.2f} =====")
        print("  (1) PREDICT next-day book PnL from observable state:")
        for f in feats:
            print(f"        Spearman({f:>7} → next-day net) = {dd[f].corr(dd['net'], method='spearman'):+.3f}")
        Z = np.column_stack([np.ones(len(dd))] + [((dd[f] - dd[f].mean()) / dd[f].std()).values for f in feats])
        y = dd["net"].values; beta = np.linalg.lstsq(Z, y, rcond=None)[0]; yhat = Z @ beta
        r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # sign hit-rate of the combined predictor
        hit = (np.sign(yhat - yhat.mean()) == np.sign(y)).mean()
        print(f"        combined OLS R² = {r2:.4f}   |   sign hit-rate = {hit*100:.1f}%  (50% = coin flip)")
        print("  (2) SWITCH rules — go flat next day when detector says bad (both-era test):")
        base_sh = sharpe(dd["net"].values)
        for name, mask in [("flat when trailing-10d book < 0", dd["trail10"] >= 0),
                           ("flat when btc_rvol7 in top-20%", dd["rvol"] <= dd["rvol"].quantile(0.8)),
                           ("flat when btc_ret_30d < -10% (bear)", dd["btc30"] >= -0.10)]:
            sw = dd["net"].where(mask, 0.0).values
            frac_flat = (~mask).mean() * 100
            print(f"        {name:<38}: Sharpe {sharpe(sw):+.2f} (vs {base_sh:+.2f})  | net {sw.sum():+.0f} vs {dd['net'].sum():+.0f} bps | flat {frac_flat:.0f}% of days")
    print("SWITCHTESTDONE")

if __name__ == "__main__":
    main()
