"""Probe #6 — Why the cross-group variance? Per-symbol decomposition within
each R3c group + sub-period stability check.

Probe #5 within primed cohort: per-group accs at 7d were [0.58, 0.62, 0.55, 0.45]
(spread of 0.17). Is each group's accuracy driven by 1-2 symbol idiosyncrasies
(signal fragile, name-specific) or by most symbols contributing similarly
(signal robust, name-class)? And is the signal sample-period stable (first
half vs second half)?

Decomposition:
  A. Per-symbol directional accuracy: for each symbol s in held-out group g,
     compute accuracy of the trail-PnL→next-sign rule on s's rows. Rule sign
     learned on the OTHER 4 groups (as in Probe #4/5). Report symbol counts
     and per-symbol acc within each group.
  B. Per-group, distribution of per-symbol acc: how concentrated vs spread.
  C. Sub-period: first half vs second half of sample, overall and per group.

Honest verdict: if most symbols in each group contribute mildly positive
acc (e.g., 0.52-0.58 for most syms in groups that scored 0.60) → robust
name-class signal. If 1-2 syms at 0.75+ pull a group while others at 0.45 →
fragile name-specific (the closed pattern the K=3/R3c arcs kept catching).
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
WIN_DAYS = 7   # the "best" middle window from probe #5


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


def main():
    t0 = time.time()
    L = pd.read_parquet(LEGS)
    L["time"] = pd.to_datetime(L["time"], utc=True)
    L = L.sort_values(["symbol", "time"]).reset_index(drop=True)

    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "atr_pct"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan["primed"] = pan.groupby("open_time")["atr_pct"].transform(
        lambda s: s >= s.quantile(0.90))

    D = build_history(L, WIN_DAYS)
    D = D.merge(pan[["symbol", "open_time", "primed"]],
                left_on=["symbol", "time"], right_on=["symbol", "open_time"], how="left")
    D["primed"] = D["primed"].fillna(False)
    Dc = D[D["primed"]].copy()
    Dc["next_sign"] = np.sign(Dc["next_contrib"])
    Dc["next_abs"] = Dc["next_contrib"].abs()

    syms = sorted(L["symbol"].unique())
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    gmap = {s: i % 5 for i, s in enumerate(shf)}
    Dc["g"] = Dc["symbol"].map(gmap)

    # A. per-symbol acc within each held-out group
    per_group = {}
    for g in range(5):
        tr = Dc[Dc["g"] != g]
        te = Dc[Dc["g"] == g]
        if len(tr) < 50 or len(te) < 30:
            per_group[f"g{g}"] = {"status": "insufficient", "n": int(len(te))}
            continue
        rel = np.sign(np.corrcoef(tr["trail_signed_mean"].rank(), tr["next_sign"])[0, 1])
        if rel == 0: rel = 1.0
        te = te.copy()
        te["pred"] = np.sign(rel * (te["trail_signed_mean"] - te["trail_signed_mean"].median()))
        te["hit"] = (te["pred"] == te["next_sign"]).astype(int)
        te = te[te["pred"] != 0]
        sym_stats = te.groupby("symbol").agg(
            n=("hit", "size"),
            acc=("hit", "mean"),
            mean_next=("next_contrib", "mean"),
        ).reset_index().sort_values("acc", ascending=False)
        per_group[f"g{g}"] = {
            "n_total": int(len(te)),
            "overall_acc": round(float(te["hit"].mean()), 4),
            "rel_learned_sign": int(rel),
            "per_symbol": [{"sym": r["symbol"], "n": int(r["n"]),
                            "acc": round(float(r["acc"]), 3),
                            "mean_next_bps": round(float(r["mean_next"]), 1)}
                           for _, r in sym_stats.iterrows()],
        }

    # B. distribution stats: in each group, fraction of symbols with acc >= 0.55
    for g in range(5):
        if "per_symbol" not in per_group[f"g{g}"]: continue
        ps = per_group[f"g{g}"]["per_symbol"]
        accs = [s["acc"] for s in ps if s["n"] >= 5]
        per_group[f"g{g}"]["n_syms_evaluable"] = len(accs)
        per_group[f"g{g}"]["frac_syms_acc>=0.55"] = round(
            float(sum(a >= 0.55 for a in accs) / len(accs)), 3) if accs else None
        per_group[f"g{g}"]["frac_syms_acc>=0.50"] = round(
            float(sum(a >= 0.50 for a in accs) / len(accs)), 3) if accs else None
        per_group[f"g{g}"]["med_sym_acc"] = round(float(np.median(accs)), 3) if accs else None

    # C. sub-period stability: split sample on time median
    tmed = Dc["time"].median()
    sub_period_results = {}
    for half, sub in (("first_half", Dc[Dc["time"] <= tmed]),
                       ("second_half", Dc[Dc["time"] > tmed])):
        gs = []
        for g in range(5):
            tr = sub[sub["g"] != g]
            te = sub[sub["g"] == g]
            if len(tr) < 50 or len(te) < 30: continue
            rel = np.sign(np.corrcoef(tr["trail_signed_mean"].rank(), tr["next_sign"])[0, 1])
            if rel == 0: rel = 1.0
            pred = np.sign(rel * (te["trail_signed_mean"] - te["trail_signed_mean"].median()))
            actual = np.sign(te["next_sign"].to_numpy())
            m = (pred != 0) & (actual != 0)
            if m.sum() < 30: continue
            gs.append(float((pred[m] == actual[m]).mean()))
        sub_period_results[half] = {
            "n_rows": int(len(sub)),
            "per_group_acc": [round(x, 3) for x in gs],
            "mean_acc": round(float(np.mean(gs)), 4) if gs else None,
        }

    out = {
        "window_days": WIN_DAYS,
        "n_total_primed_history_rows": int(len(Dc)),
        "median_time_split": str(tmed),
        "per_group": per_group,
        "sub_period_stability": sub_period_results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "probe6_results.json").write_text(json.dumps(out, indent=2, default=str))

    # printout
    print(f"primed cohort 7d window: n={len(Dc):,} rows\n")
    for g in range(5):
        v = per_group[f"g{g}"]
        if "per_symbol" not in v:
            print(f"g{g}: insufficient"); continue
        print(f"g{g}: overall acc {v['overall_acc']:.4f}, "
              f"n_syms_evaluable {v['n_syms_evaluable']}, "
              f"frac_syms_acc≥0.55 {v['frac_syms_acc>=0.55']}, "
              f"frac_syms_acc≥0.50 {v['frac_syms_acc>=0.50']}, "
              f"med_sym_acc {v['med_sym_acc']}")
        print(f"  per-symbol (top/bottom 4):")
        ps = v["per_symbol"]
        for r in ps[:4]:
            print(f"    {r['sym']:<14} n={r['n']:>3} acc={r['acc']:>.3f}  mean_next={r['mean_next_bps']:>+7.1f} bps")
        if len(ps) > 8:
            print("    ...")
        for r in ps[-4:]:
            print(f"    {r['sym']:<14} n={r['n']:>3} acc={r['acc']:>.3f}  mean_next={r['mean_next_bps']:>+7.1f} bps")
    print(f"\nSub-period:")
    for h, v in sub_period_results.items():
        print(f"  {h}: mean acc {v['mean_acc']}, per_group {v['per_group_acc']}, n={v['n_rows']}")
    print("\nPROBE6_DONE")


if __name__ == "__main__":
    main()
