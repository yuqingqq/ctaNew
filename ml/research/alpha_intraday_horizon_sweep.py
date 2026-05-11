"""Stage 1b: sweep intraday forward horizons. Maybe 30min is too fast."""
from __future__ import annotations
import logging, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parents[2] / "data" / "ml" / "cache"
TIER_AB = ["AAPL","AMZN","GOOGL","META","MSFT","MU","NFLX","NVDA","ORCL","PLTR","TSLA"]
SP100 = sorted({p.name.split("_")[1] for p in CACHE.glob("poly_*_5m.parquet")})

def is_rth_utc(ts):
    minutes = ts.dt.hour * 60 + ts.dt.minute
    return (minutes >= 13 * 60 + 30) & (minutes <= 21 * 60)


def load_5m():
    rows = []
    for s in SP100:
        p = CACHE / f"poly_{s}_5m.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if "symbol" not in df.columns: df["symbol"] = s
            rows.append(df[["ts","symbol","open","high","low","close","volume","vwap"]])
    panel = pd.concat(rows, ignore_index=True)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    return panel.sort_values(["symbol","ts"]).reset_index(drop=True)


def resample_to(panel5m, freq):
    panel5m = panel5m[is_rth_utc(panel5m["ts"])].copy()
    rows = []
    for s, g in panel5m.groupby("symbol"):
        g = g.set_index("ts").sort_index()
        agg = g.resample(freq, origin="start_day").agg({
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"
        }).dropna(subset=["close"])
        agg["symbol"] = s
        agg = agg.reset_index()
        agg = agg[is_rth_utc(agg["ts"])]
        rows.append(agg)
    return pd.concat(rows, ignore_index=True).sort_values(["symbol","ts"]).reset_index(drop=True)


def add_residual_and_features(panel, freq_label, beta_bars=60):
    g = panel.groupby("symbol", group_keys=False)
    panel["ret"] = g["close"].apply(lambda s: np.log(s / s.shift(1)))
    grp = panel.groupby("ts")["ret"]
    total = grp.transform("sum"); n = grp.transform("count")
    panel["bk_ret"] = (total - panel["ret"].fillna(0)) / (n - 1).replace(0, np.nan)
    def _b(g):
        cov = ((g["ret"]*g["bk_ret"]).rolling(beta_bars).mean()
               - g["ret"].rolling(beta_bars).mean()*g["bk_ret"].rolling(beta_bars).mean())
        var = g["bk_ret"].rolling(beta_bars).var().replace(0, np.nan)
        return (cov/var).clip(-5,5).shift(1)
    panel["beta"] = panel.groupby("symbol", group_keys=False).apply(_b).values
    panel["resid"] = panel["ret"] - panel["beta"] * panel["bk_ret"]
    return panel


def per_bar_rank_ic(df, feat, target):
    valid = df.dropna(subset=[feat, target])
    if valid.empty: return float("nan"), 0
    ics = []
    for ts, g in valid.groupby("ts"):
        if len(g) < 3: continue
        ic = g[feat].rank().corr(g[target].rank())
        if not np.isnan(ic): ics.append(ic)
    return (float(np.mean(ics)) if ics else float("nan")), len(ics)


def main():
    panel5m = load_5m()
    log.info("loaded 5m panel: %d rows", len(panel5m))

    for cadence_label, freq in [("30min","30min"), ("1h","1h"), ("2h","2h")]:
        log.info("\n========== Cadence: %s bars ==========", cadence_label)
        panel = resample_to(panel5m, freq)
        log.info("  bars: %d", len(panel))
        panel = add_residual_and_features(panel, cadence_label)
        g = panel.groupby("symbol", group_keys=False)

        # Test multiple forward horizons (in bars)
        # And test FORWARD HORIZONS as MULTI-bar (e.g., next 3 bars for 30min cadence = next 90min)
        bars_per_h = {"30min": 2, "1h": 1, "2h": 0.5}[cadence_label]
        for fwd_h_label, fwd_bars in [
            (f"fwd_1bar (~{int(60/bars_per_h)}min)" if bars_per_h>=1 else f"fwd_1bar (~2h)", 1),
            ("fwd_2bars", 2),
            ("fwd_4bars", 4),
            ("fwd_8bars", 8),
        ]:
            tgt = f"fwd_resid_{fwd_h_label}"
            panel[tgt] = g["resid"].apply(lambda s, h=fwd_bars: s.rolling(h).sum().shift(-h))

            # Test simple feature: trailing 1-bar idio reversal (-1 × prev resid)
            feat = "f_idio_rev"
            panel[feat] = -1 * g["resid"].apply(lambda s: s.shift(1))

            sub = panel[panel["symbol"].isin(TIER_AB)]
            ic, nb = per_bar_rank_ic(sub, feat, tgt)
            log.info(f"  cadence={cadence_label:<5} target={fwd_h_label:<25} "
                     f"naive_reversal_IC={ic:>+.4f} n_bars={nb}")

        # Also test hour-of-day binned IC for 30min/1h cadence
        if cadence_label in ("30min", "1h"):
            sub = panel[panel["symbol"].isin(TIER_AB)].copy()
            tgt = f"fwd_resid_fwd_2bars"
            sub["hour_utc"] = sub["ts"].dt.hour
            log.info(f"\n  IC by hour (cadence={cadence_label}, target=fwd_2bars):")
            log.info(f"  {'hour_utc':<10} {'IC':>8} {'n_bars':>8}")
            for hour, g_h in sub.groupby("hour_utc"):
                ic, nb = per_bar_rank_ic(g_h, "f_idio_rev", tgt)
                if nb > 100:
                    log.info(f"    {hour:>2d}:00 UTC  IC={ic:>+.4f}  n_bars={nb}")


if __name__ == "__main__":
    main()
