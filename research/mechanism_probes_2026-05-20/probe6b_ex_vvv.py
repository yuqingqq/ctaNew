"""Probe #6b — Does the PnL-mean-reversion signal survive without VVV?

Probe #6 found g3's apparent 0.55 acc is built on 161 rows of VVV (40% of
the entire 405-row primed-cohort dataset). VVV's own acc 0.56 is mid-pack;
the question is whether VVV is propping up the cross-group mean or whether
the signal is broadly distributed. Probe #6 also showed strong per-symbol
disagreement — PENDLE (n=21) at acc 0.29, HBAR (n=8) at 0.38, BIO (n=53)
at 0.45 — directly refuting the "mean-reversion" interpretation for several
top-contributor non-VVV names.

This probe re-runs Probe #4's full and primed-cohort OOS-symbol directional
accuracy with VVV (and separately, with all top-3 contributors by leg count)
excluded. Two reads:
  ex_vvv: drop VVVUSDT only
  ex_top3: drop VVV + BIO + PENGU (3 highest non-trivial leg counts)
For each cut, run windows {3, 7, 14} (primed cohort), report dir acc and
placebo.

Decisive check: if ex_vvv full-cohort acc drops to ~0.50 and ex_vvv
primed-cohort drops to <0.52 across all windows, the "PnL mean-reversion"
mechanism is VVV-only and not portable. KILL the direction.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/mechanism_probes_2026-05-20"; OUT.mkdir(parents=True, exist_ok=True)
LEGS = REPO / "research/convexity_forensic_2026-05-19/entered_legs.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
SEED = 20260519


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


def oos_dir_acc(D, feat, lab):
    syms = sorted(D["symbol"].unique())
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    gmap = {s: i % 5 for i, s in enumerate(shf)}
    D = D.assign(g=D["symbol"].map(gmap))
    accs = []; ns = []
    for g in range(5):
        tr = D[D["g"] != g]; te = D[D["g"] == g]
        if len(tr) < 50 or len(te) < 30: continue
        rel = np.sign(np.corrcoef(tr[feat].rank(), tr[lab])[0, 1])
        if rel == 0: rel = 1.0
        pred = np.sign(rel * (te[feat] - te[feat].median()))
        actual = np.sign(te[lab].to_numpy())
        m = (pred != 0) & (actual != 0)
        if m.sum() < 30: continue
        accs.append(float((pred[m] == actual[m]).mean()))
        ns.append(int(m.sum()))
    return (float(np.mean(accs)) if accs else np.nan, accs, ns)


def main():
    t0 = time.time()
    L = pd.read_parquet(LEGS)
    L["time"] = pd.to_datetime(L["time"], utc=True)
    L = L.sort_values(["symbol", "time"]).reset_index(drop=True)

    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "atr_pct"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan["primed"] = pan.groupby("open_time")["atr_pct"].transform(
        lambda s: s >= s.quantile(0.90))

    cuts = {
        "all":       set(),
        "ex_vvv":    {"VVVUSDT"},
        "ex_top3":   {"VVVUSDT", "BIOUSDT", "PENGUUSDT"},
    }
    windows = [3, 7, 14]

    results = {}
    for cut_name, excl in cuts.items():
        L_c = L[~L["symbol"].isin(excl)]
        for w in windows:
            D = build_history(L_c, w)
            D = D.merge(pan[["symbol", "open_time", "primed"]],
                        left_on=["symbol", "time"], right_on=["symbol", "open_time"], how="left")
            D["primed"] = D["primed"].fillna(False)
            D["next_sign"] = np.sign(D["next_contrib"])

            a_full, gs_f, ns_f = oos_dir_acc(D, "trail_signed_mean", "next_sign")
            Dc = D[D["primed"]].copy()
            a_pri, gs_p, ns_p = oos_dir_acc(Dc, "trail_signed_mean", "next_sign")

            # placebo
            D_p = D.copy()
            D_p["lab_sh"] = D["next_sign"].sample(frac=1, random_state=42).to_numpy()
            a_pl, _, _ = oos_dir_acc(D_p, "trail_signed_mean", "lab_sh")

            key = f"{cut_name}_{w}d"
            results[key] = {
                "n_full":    int(len(D)),
                "n_primed":  int(len(Dc)),
                "dir_full":  round(a_full, 4) if a_full == a_full else None,
                "dir_primed": round(a_pri, 4) if a_pri == a_pri else None,
                "placebo_full": round(a_pl, 4) if a_pl == a_pl else None,
                "per_group_full":    [round(x, 3) for x in gs_f],
                "per_group_primed":  [round(x, 3) for x in gs_p],
                "n_per_group_primed": ns_p,
            }
            print(f"{cut_name:<10} {w:>3}d  n_full={len(D):>5}  n_primed={len(Dc):>4}  "
                  f"dir_full={a_full:.4f}  dir_primed={(a_pri if a_pri==a_pri else 0):.4f}  "
                  f"placebo={a_pl:.4f}  per_g_primed={[round(x,2) for x in gs_p]}")

    # verdict
    full_drop = (results["all_7d"]["dir_full"] or 0) - (results["ex_vvv_7d"]["dir_full"] or 0)
    primed_drop = (results["all_7d"]["dir_primed"] or 0) - (results["ex_vvv_7d"]["dir_primed"] or 0)
    verdict = ("VVV-driven (single-name); NOT a portable mechanism"
               if (results["ex_vvv_7d"]["dir_primed"] or 0) < 0.515
               else "Survives ex-VVV — real residual mechanism worth strategy plan")

    out = {
        "scope": "PnL-mean-reversion robustness without VVV / top-3 names",
        "cuts": list(cuts.keys()),
        "windows_days": windows,
        "results": results,
        "drop_dir_full_all_minus_exvvv_7d":   round(full_drop, 4),
        "drop_dir_primed_all_minus_exvvv_7d": round(primed_drop, 4),
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "probe6b_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nFull-drop (all − ex_vvv) at 7d: {full_drop:+.4f} dir_full, {primed_drop:+.4f} dir_primed")
    print(f"VERDICT: {verdict}")
    print("PROBE6B_DONE")


if __name__ == "__main__":
    main()
