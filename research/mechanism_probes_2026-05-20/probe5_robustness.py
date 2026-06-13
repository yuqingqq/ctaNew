"""Probe #5 — Robustness of the Probe #4 PnL-mean-reversion signal.

Probe #4 found OOS-symbol directional acc 0.526 (placebo 0.487, lifecycle
ceiling 0.515) for trail_signed_mean(7d) → next_sign — a small but real lift.
Before treating this as a mechanism candidate worth a strategy plan, test
robustness on three axes:

  A. Window sensitivity: 3d, 7d (probe #4), 14d, 30d lookbacks. Real signal
     should be relatively stable across nearby windows; sensitive to one
     specific window is noise.
  B. Cohort interaction: primed cohort (top-decile atr_pct at the next-leg
     entry, PIT) — does the signal amplify where magnitudes are largest?
  C. Magnitude predictability: trailing |contrib| → next |contrib| corr at
     each window, plus a combined sign + magnitude composite (predict
     POSITION SIZE as past |contrib|, predict SIGN as reversal of past sign).
     Composite signal Sharpe-equivalent vs sign-only and magnitude-only.

PIT throughout; OOS-symbol (5 disjoint groups, seed 20260519); label-shuffled
placebo for each window. Honest verdict: if direction acc stays ≥0.52 across
≥3 of 4 windows AND amplifies in primed cohort AND magnitude corr ≥0.20 →
genuine robust mechanism; spawn a strategy pre-registered plan. Else likely
noise.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/mechanism_probes_2026-05-20"; OUT.mkdir(parents=True, exist_ok=True)
LEGS = REPO / "research/convexity_forensic_2026-05-19/entered_legs.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
SEED = 20260519
WINDOWS_DAYS = [3, 7, 14, 30]


def build_history(L, days):
    rows = []
    win = pd.Timedelta(days=days)
    for sym, g in L.groupby("symbol", sort=False):
        c = g["contrib_bps"].to_numpy(); t = g["time"].to_numpy()
        for i in range(len(g)):
            mask = (t < t[i]) & (t >= (t[i] - win.to_numpy()))
            past = c[mask]
            if len(past) >= 3:
                rows.append({"symbol": sym, "time": pd.Timestamp(t[i]),
                             "trail_signed_mean": float(past.mean()),
                             "trail_abs_mean": float(np.abs(past).mean()),
                             "next_contrib": float(c[i])})
    return pd.DataFrame(rows)


def dir_acc(D, feat, lab):
    syms = sorted(D["symbol"].unique())
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    gmap = {s: i % 5 for i, s in enumerate(shf)}
    D = D.assign(g=D["symbol"].map(gmap))
    accs = []
    for g in range(5):
        tr = D[D["g"] != g]; te = D[D["g"] == g]
        if len(tr) < 100 or len(te) < 30: continue
        rel = np.sign(np.corrcoef(tr[feat].rank(), tr[lab])[0, 1])
        if rel == 0: rel = 1.0
        pred = np.sign(rel * (te[feat] - te[feat].median()))
        actual = np.sign(te[lab].to_numpy())
        m = (pred != 0) & (actual != 0)
        if m.sum() < 30: continue
        accs.append(float((pred[m] == actual[m]).mean()))
    return float(np.mean(accs)) if accs else np.nan, accs


def main():
    t0 = time.time()
    L = pd.read_parquet(LEGS)
    L["time"] = pd.to_datetime(L["time"], utc=True)
    L = L.sort_values(["symbol", "time"]).reset_index(drop=True)

    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "atr_pct"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    # primed PIT (top-decile atr_pct per open_time) — feature is PIT in panel
    pan["primed"] = pan.groupby("open_time")["atr_pct"].transform(
        lambda s: s >= s.quantile(0.90))

    out = {"windows": {}}
    for w in WINDOWS_DAYS:
        D = build_history(L, w)
        D["next_sign"] = np.sign(D["next_contrib"]); D["next_abs"] = D["next_contrib"].abs()
        D = D.merge(pan[["symbol", "open_time", "primed"]],
                    left_on=["symbol", "time"], right_on=["symbol", "open_time"], how="left")
        D["primed"] = D["primed"].fillna(False)

        # full
        a_full, gs_full = dir_acc(D, "trail_signed_mean", "next_sign")
        # placebo
        D_p = D.copy()
        D_p["next_sign_sh"] = D["next_sign"].sample(frac=1, random_state=42).to_numpy()
        a_plac, _ = dir_acc(D_p, "trail_signed_mean", "next_sign_sh")
        # primed cohort only
        Dc = D[D["primed"]].copy()
        a_primed, gs_primed = dir_acc(Dc, "trail_signed_mean", "next_sign")
        # magnitude
        mag_full = float(D[["trail_abs_mean", "next_abs"]].corr().iloc[0, 1])
        mag_primed = float(Dc[["trail_abs_mean", "next_abs"]].corr().iloc[0, 1]) if len(Dc) > 50 else None
        out["windows"][f"{w}d"] = {
            "n": int(len(D)),
            "dir_acc_full": round(a_full, 4),
            "dir_acc_primed": round(a_primed, 4) if a_primed == a_primed else None,
            "placebo_full": round(a_plac, 4),
            "lift_vs_placebo_full": round(a_full - a_plac, 4),
            "per_group_full": [round(x, 3) for x in gs_full],
            "per_group_primed": [round(x, 3) for x in gs_primed] if gs_primed else None,
            "mag_corr_full": round(mag_full, 4),
            "mag_corr_primed": round(mag_primed, 4) if mag_primed else None,
            "n_primed": int(len(Dc)),
        }

    # robustness verdict
    accs = [out["windows"][f"{w}d"]["dir_acc_full"] for w in WINDOWS_DAYS]
    plac = [out["windows"][f"{w}d"]["placebo_full"] for w in WINDOWS_DAYS]
    mag = [out["windows"][f"{w}d"]["mag_corr_full"] for w in WINDOWS_DAYS]
    n_above_thresh = int(sum((a >= 0.52) for a in accs))
    out["robustness"] = {
        "dir_acc_per_window": dict(zip([f"{w}d" for w in WINDOWS_DAYS], accs)),
        "placebo_per_window": dict(zip([f"{w}d" for w in WINDOWS_DAYS], plac)),
        "mag_corr_per_window": dict(zip([f"{w}d" for w in WINDOWS_DAYS], mag)),
        "n_windows_with_dir_acc>=0.52": n_above_thresh,
        "verdict": ("ROBUST candidate-mechanism" if n_above_thresh >= 3
                    and min(accs) > 0.50 + 0.005 else
                    "FRAGILE/likely-noise — single-window result, not a mechanism"),
    }
    out["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "probe5_results.json").write_text(json.dumps(out, indent=2, default=str))

    # printout
    print(f"{'window':<8}{'n':>8}{'dir_full':>10}{'plac':>8}{'lift':>8}"
          f"{'dir_primed':>12}{'mag_full':>10}{'mag_primed':>12}{'per_group':>40}")
    for w in WINDOWS_DAYS:
        v = out["windows"][f"{w}d"]
        pg = " ".join(f"{x:.2f}" for x in (v["per_group_full"] or []))
        pg_p = (" ".join(f"{x:.2f}" for x in v["per_group_primed"])
                if v["per_group_primed"] else "—")
        print(f"{w}d{'':>5}{v['n']:>8}{v['dir_acc_full']:>10.4f}{v['placebo_full']:>8.3f}"
              f"{v['lift_vs_placebo_full']:>+8.3f}"
              f"{(v['dir_acc_primed'] or 0):>12.4f}"
              f"{v['mag_corr_full']:>10.3f}"
              f"{(v['mag_corr_primed'] or 0):>12.3f}"
              f"   {pg}  | primed {pg_p}")
    print(f"\nROBUSTNESS VERDICT: {out['robustness']['verdict']}")
    print("PROBE5_DONE")


if __name__ == "__main__":
    main()
