"""Probe #3 — Trailing-skew SYMBOL-CLASS signature, PIT OOS-symbol.

Probe #2 finding: systematic-winner symbols have positive idio_skew at the
symbol-mean level; systematic-loser symbols have negative. Per-trade
instantaneous idio_skew (C0pre) gave no directional edge.

Question: does a TRAILING-mean idio_skew (a slower symbol-class signature,
structurally different from the per-row feature WINNER_21/C0pre tested)
predict the SIGN of next-4h beta-neutral residual, OOS-symbol, vs placebo?
Also: does it work BETTER within the primed (high-atr) cohort?

If trailing-skew OOS-symbol directional accuracy materially exceeds (a) the
instantaneous idio_skew_1d baseline, (b) label-shuffled placebo, AND (c) the
51.5% directional ceiling from the lifecycle probe — that's a genuinely
new symbol-class selection mechanism. Otherwise, Probe #2's pattern was
likely driven by 1–2 outlier symbols (VVV) and isn't a stable lever.

PIT: trailing-30d-mean idio_skew_1d is computed strictly PIT (.shift(1) after
rolling). Label = sign(alpha_vs_btc_realized) — realized next-4h residual,
legitimate.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/mechanism_probes_2026-05-20"; OUT.mkdir(parents=True, exist_ok=True)
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
SEED = 20260519
WIN_BARS = 288 * 30   # 30-day trailing


def oos_sym_dir_acc(D, feat, label_col):
    """Train sign of (feat→label) on groups ≠g, apply to g; average over g."""
    syms = sorted(D["symbol"].unique())
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    gmap = {s: i % 5 for i, s in enumerate(shf)}
    D = D.assign(g=D["symbol"].map(gmap))
    accs = []
    for g in range(5):
        tr = D[(D["g"] != g) & D[feat].notna() & D[label_col].notna()]
        te = D[(D["g"] == g) & D[feat].notna() & D[label_col].notna()]
        if len(tr) < 5000 or len(te) < 2000: continue
        rel = np.sign(np.corrcoef(tr[feat].rank(), tr[label_col])[0, 1])
        if rel == 0: rel = 1.0
        pred_sign = np.sign(rel * (te[feat] - te[feat].median()))
        lab_sign = np.sign(te[label_col].to_numpy())
        m = (pred_sign != 0) & (lab_sign != 0)
        if m.sum() < 200: continue
        accs.append(float((pred_sign[m] == lab_sign[m]).mean()))
    return float(np.mean(accs)) if accs else np.nan, accs


def main():
    t0 = time.time()
    cols = ["symbol", "open_time", "alpha_vs_btc_realized", "idio_skew_1d",
            "atr_pct"]
    p = pd.read_parquet(PANEL, columns=cols)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p.dropna(subset=["alpha_vs_btc_realized"])

    # Trailing-30d mean idio_skew per symbol, PIT
    p = p.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    p["skew_30d_mean"] = (p.groupby("symbol")["idio_skew_1d"]
                          .transform(lambda s: s.rolling(WIN_BARS, min_periods=288*7).mean().shift(1)))
    print(f"panel rows {len(p):,}  with trailing-30d skew non-null "
          f"{p['skew_30d_mean'].notna().mean()*100:.1f}%", flush=True)

    # Build cohort indicator: per-cycle top-decile atr_pct (PIT) = "primed"
    p["primed"] = p.groupby("open_time")["atr_pct"].transform(
        lambda s: s >= s.quantile(0.90))

    # Full panel directional accuracy
    full_trail, gA = oos_sym_dir_acc(p, "skew_30d_mean", "alpha_vs_btc_realized")
    full_inst, gB = oos_sym_dir_acc(p, "idio_skew_1d", "alpha_vs_btc_realized")
    # within primed cohort only
    C = p[p["primed"]].copy()
    primed_trail, gC = oos_sym_dir_acc(C, "skew_30d_mean", "alpha_vs_btc_realized")
    primed_inst, gD = oos_sym_dir_acc(C, "idio_skew_1d", "alpha_vs_btc_realized")

    # Placebo: shuffled label
    rng = np.random.RandomState(SEED)
    D = p.copy(); D["lab_sh"] = np.sign(D["alpha_vs_btc_realized"]).to_numpy()
    sh = D["lab_sh"].to_numpy().copy(); rng.shuffle(sh); D["lab_sh"] = sh
    plac_full, _ = oos_sym_dir_acc(D, "skew_30d_mean", "lab_sh")
    Dc = D[D["primed"]]
    plac_primed, _ = oos_sym_dir_acc(Dc, "skew_30d_mean", "lab_sh")

    # Magnitude check — does trailing-skew predict the |move| too?
    p["abs_resid"] = p["alpha_vs_btc_realized"].abs()
    full_corr_mag = float(p[["skew_30d_mean", "abs_resid"]].dropna()
                          .corr().iloc[0, 1])

    # SYS_WINNER vs SYS_LOSER trailing-skew quintile contribution test:
    # rank symbols (per cycle) by trailing-skew, compute Q5−Q1 mean residual
    p["sk_q"] = p.groupby("open_time")["skew_30d_mean"].transform(
        lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    by_q = p.groupby("sk_q")["alpha_vs_btc_realized"].mean()

    out = {
        "trailing_skew_30d_OOS_symbol_dir_acc_full":   round(full_trail, 4),
        "trailing_skew_30d_OOS_symbol_dir_acc_primed": round(primed_trail, 4),
        "instantaneous_skew_OOS_symbol_dir_acc_full":   round(full_inst, 4),
        "instantaneous_skew_OOS_symbol_dir_acc_primed": round(primed_inst, 4),
        "placebo_label_shuffled_full":   round(plac_full, 4),
        "placebo_label_shuffled_primed": round(plac_primed, 4),
        "per_group_acc_trailing_full":   [round(x, 3) for x in gA],
        "per_group_acc_trailing_primed": [round(x, 3) for x in gC],
        "skew_30d_vs_|resid|_corr_full": round(full_corr_mag, 4),
        "per_cycle_trailing_skew_quintile_mean_residual": {
            int(k): round(float(v), 6) for k, v in by_q.items()},
        "lifecycle_probe_ceiling_for_reference": "r24 0.515, all others ~0.50",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "probe3_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str), flush=True)
    print("PROBE3_DONE", flush=True)


if __name__ == "__main__":
    main()
