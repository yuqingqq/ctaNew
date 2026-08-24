"""Model-free adversarial cross-check for the interaction hypothesis (C1). The Ridge preproc (x6) quantile-clips at
1/99% and standardizes, which could blunt an interaction living in the high-dispersion tail. So, independent of the
pipeline, split each bar's cross-section by market regime (xs-dispersion of trailing 1d ret; btc_rvol_7d) into
HIGH/LOW halves and measure the raw cross-sectional rank-IC of l2_imb1 (and imb_ewma) -> realized alpha inside each
half, both eras, day-clustered CI. If imbalance has NO conditional edge even raw/unclipped in the HIGH-regime half,
the C1 interaction null is robust (not a preproc artifact). imb1 is CONTINUATION-signed vs a reversion target, so a
real conditional continuation edge would show as a *positive* IC that strengthens in the HIGH half.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
import live.l2_alpha_constructions as L

rng = np.random.default_rng(11)
CUT = pd.Timestamp("2025-10-01", tz="UTC")


def day_boot(ic):
    s = pd.DataFrame({"ic": ic.values}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    g = [x["ic"].values for _, x in s.groupby("d")]
    if len(g) < 4: return (np.nan, np.nan)
    o = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2500)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))


def main():
    PAN = L.build_panel()
    PAN = PAN.rename(columns={"alpha_vs_btc_realized": "alpha_A"})
    sub_all = PAN[PAN.cov_full & PAN["alpha_A"].notna()].copy()
    # regime scalars per bar (market-wide): xs_disp already per-bar; btc_rvol_7d is ~per-bar (BTC series)
    bar = sub_all.groupby("open_time").agg(xs_disp=("xs_disp", "first"), btc_rvol=("btc_rvol_7d", "first"))
    for reg in ["xs_disp", "btc_rvol"]:
        med = bar[reg].expanding(min_periods=30).median()          # PIT trailing median (no look-ahead in the split)
        bar[reg + "_hi"] = (bar[reg] > med)
    sub_all = sub_all.merge(bar[[c for c in bar.columns if c.endswith("_hi")]], left_on="open_time", right_index=True)

    for feat in ["l2_imb1", "imb_ewma"]:
        print(f"\n============ conditional cross-sectional rank-IC of {feat} -> realized alpha ============")
        print("(imb1 is continuation-signed; a real conditional edge = IC grows in the HIGH-regime half)")
        for reg in ["xs_disp", "btc_rvol"]:
            print(f"  -- regime split = {reg} --")
            for era, m in [("RECENT", sub_all[sub_all.open_time >= CUT]), ("OOS", sub_all[sub_all.open_time < CUT])]:
                for half, flagcol, want in [("HIGH", reg + "_hi", True), ("LOW", reg + "_hi", False)]:
                    d = m[m[flagcol] == want]
                    ic = d.groupby("open_time").apply(
                        lambda g: spearmanr(g[feat], g["alpha_A"]).correlation if len(g) >= 6 else np.nan).dropna()
                    if len(ic) < 4:
                        print(f"    {era:6s} {half:4s}: n/a"); continue
                    lo, up = day_boot(ic)
                    off = "off-0" if (lo > 0 or up < 0) else "  0  "
                    print(f"    {era:6s} {half:4s}: IC {ic.mean():+.4f} [{lo:+.4f},{up:+.4f}] {off}  (bars={len(ic)})")
    print("\nL2ALPHADIAG_DONE")


if __name__ == "__main__":
    main()
