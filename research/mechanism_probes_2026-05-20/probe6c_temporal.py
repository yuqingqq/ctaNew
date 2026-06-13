"""Probe #6c — Temporal stability of the PnL-mean-reversion signal ex-VVV.

Probe #6 found first_half 0.495 / second_half 0.566 — strong period instability.
Probe #6b found ex-VVV the cross-symbol signal survives at lift +0.05 over
placebo in primed cohort. Open question: is the temporal instability a
VVV-pump-window artifact (which would be resolved by ex-VVV), or a genuine
regime issue that would also kill the strategy forward?

Decisive test: split the ex-VVV dataset by time (median split, plus a
two-third / one-third split), report per-half / per-third dir_primed and
placebo. If both halves still show acc ≥ 0.52 over placebo → the signal is
temporally stable. If only the second half does → forward expected lift
is roughly half of the in-sample lift, and the strategy is fragile.

Three windows {3, 7, 14}d primed cohort. ex_vvv only — Probe #6b already
confirmed VVV is not the structural carrier.
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
                             "next_contrib": float(c[i])})
    return pd.DataFrame(rows)


def oos_dir_acc(D, feat, lab):
    syms = sorted(D["symbol"].unique())
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    gmap = {s: i % 5 for i, s in enumerate(shf)}
    D = D.assign(g=D["symbol"].map(gmap))
    accs = []
    for g in range(5):
        tr = D[D["g"] != g]; te = D[D["g"] == g]
        if len(tr) < 30 or len(te) < 20: continue
        rel = np.sign(np.corrcoef(tr[feat].rank(), tr[lab])[0, 1])
        if rel == 0: rel = 1.0
        pred = np.sign(rel * (te[feat] - te[feat].median()))
        actual = np.sign(te[lab].to_numpy())
        m = (pred != 0) & (actual != 0)
        if m.sum() < 20: continue
        accs.append(float((pred[m] == actual[m]).mean()))
    return (float(np.mean(accs)) if accs else np.nan, accs)


def main():
    t0 = time.time()
    L = pd.read_parquet(LEGS)
    L["time"] = pd.to_datetime(L["time"], utc=True)
    L = L[L["symbol"] != "VVVUSDT"].copy()
    L = L.sort_values(["symbol", "time"]).reset_index(drop=True)

    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "atr_pct"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan["primed"] = pan.groupby("open_time")["atr_pct"].transform(
        lambda s: s >= s.quantile(0.90))

    windows = [3, 7, 14]
    res = {}
    for w in windows:
        D = build_history(L, w)
        D = D.merge(pan[["symbol", "open_time", "primed"]],
                    left_on=["symbol", "time"], right_on=["symbol", "open_time"], how="left")
        D["primed"] = D["primed"].fillna(False)
        D["next_sign"] = np.sign(D["next_contrib"])
        Dc = D[D["primed"]].copy()

        # median time split
        tmed = Dc["time"].median()
        H1 = Dc[Dc["time"] <= tmed]
        H2 = Dc[Dc["time"] >  tmed]

        # tercile split
        t33, t66 = Dc["time"].quantile([1/3, 2/3])
        T1 = Dc[Dc["time"] <= t33]
        T2 = Dc[(Dc["time"] > t33) & (Dc["time"] <= t66)]
        T3 = Dc[Dc["time"] > t66]

        out_w = {}
        for name, sub in (("all", Dc), ("half1", H1), ("half2", H2),
                          ("tercile1", T1), ("tercile2", T2), ("tercile3", T3)):
            a, gs = oos_dir_acc(sub, "trail_signed_mean", "next_sign")
            # placebo
            sub_p = sub.copy()
            if len(sub_p) > 0:
                sub_p["lab_sh"] = sub_p["next_sign"].sample(frac=1, random_state=42).to_numpy()
            ap, _ = oos_dir_acc(sub_p, "trail_signed_mean", "lab_sh")
            out_w[name] = {"n": int(len(sub)),
                           "dir_primed": round(a, 4) if a == a else None,
                           "placebo": round(ap, 4) if ap == ap else None,
                           "lift": round(a - ap, 4) if (a == a and ap == ap) else None,
                           "per_group": [round(x, 3) for x in gs]}
            t_str = f"{sub['time'].min().date() if len(sub)>0 else '-'}..{sub['time'].max().date() if len(sub)>0 else '-'}"
            print(f"{w:>3}d {name:<10}  n={len(sub):>4}  "
                  f"dir={(a if a==a else 0):.4f}  plac={(ap if ap==ap else 0):.4f}  "
                  f"lift={((a-ap) if (a==a and ap==ap) else 0):+.4f}  range {t_str}  per_g={[round(x,2) for x in gs]}")
        res[f"{w}d"] = out_w

    # verdict
    lifts_h1 = [res[f"{w}d"]["half1"]["lift"] for w in windows]
    lifts_h2 = [res[f"{w}d"]["half2"]["lift"] for w in windows]
    h1_mean = float(np.nanmean([x for x in lifts_h1 if x is not None]))
    h2_mean = float(np.nanmean([x for x in lifts_h2 if x is not None]))

    if h1_mean >= 0.03 and h2_mean >= 0.03:
        verdict = ("Temporally stable ex-VVV — mechanism survives "
                   "VVV de-confound and time split; worth strategy plan")
    elif h1_mean < 0.01 and h2_mean >= 0.04:
        verdict = ("Second-half-only signal — period-fragile; "
                   "forward expected lift halved at best; do not adopt")
    else:
        verdict = ("Mixed/inconclusive — temporal stability marginal; "
                   "needs operational forward test, not bigger backtest")

    out = {"ex_vvv": True, "windows": res,
           "mean_lift_half1": round(h1_mean, 4),
           "mean_lift_half2": round(h2_mean, 4),
           "verdict": verdict,
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "probe6c_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nmean lift half1 = {h1_mean:+.4f}, half2 = {h2_mean:+.4f}")
    print(f"VERDICT: {verdict}")
    print("PROBE6C_DONE")


if __name__ == "__main__":
    main()
