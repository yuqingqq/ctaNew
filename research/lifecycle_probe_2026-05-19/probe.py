"""Data-driven lifecycle / mechanism probe (exploratory; rigorous on
look-ahead so any PREDICTIVE claim is trustworthy).

Answers the user's two points with data, not assertion:
 A. MECHANISM (descriptive, post-hoc, NOT tradeable): event-time price path
    around entered LONG winners vs losers — is the profitable pattern
    momentum-into-pump (rise→continue→reverse) or reversal?
 B. CONDITIONAL DIRECTION (the decisive tradeable test, PIT + OOS-symbol):
    WITHIN the "big-move-coming" cohort (PIT top-decile atr_pct per cycle),
    does a PIT *lifecycle* feature predict the SIGN of the next-4h
    beta-neutral residual (alpha_vs_btc_realized)? Feature→sign direction
    LEARNED on train groups, applied to held-out group (proper OOS-symbol,
    5 disjoint groups seed 20260519), vs label-shuffled placebo.
    Closed result = static cross-sectional rank gives ~50% (IC≈0.02). New
    question: does pump-PHASE/timing beat 50% where rank didn't?
 C. EX-VVV MECHANISM: of profitable non-VVV entered legs, concentration by
    the best lifecycle bucket — is the VVV→AXS→PENDLE rotation "system
    rotates into whatever name is currently in <phase X>"?
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/lifecycle_probe_2026-05-19"; OUT.mkdir(parents=True, exist_ok=True)
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
CW = REPO / "research/portable_alpha_2026-05-19/results/_cache/close_wide.parquet"
LEGS = REPO / "research/convexity_forensic_2026-05-19/entered_legs.parquet"
SEED = 20260519


def _dirhit(feat, sgn):
    """Directional accuracy of 'go with sign(feat)' vs realized sign."""
    m = (~np.isnan(feat)) & (~np.isnan(sgn)) & (sgn != 0)
    if m.sum() < 200: return np.nan, 0
    return float((np.sign(feat[m]) == np.sign(sgn[m])).mean()), int(m.sum())


def main():
    t0 = time.time()
    cw = pd.read_parquet(CW)
    px = cw[[c for c in cw.columns if c.startswith("c_")]].rename(columns=lambda x: x[2:]).sort_index()
    # PIT lifecycle features (strictly prior; .shift(1) after rolling)
    r24 = px.pct_change(288).shift(1)
    r72 = px.pct_change(864).shift(1)
    r7d = px.pct_change(2016).shift(1)
    dist_hi = (px / px.rolling(5760, min_periods=288).max() - 1.0).shift(1)  # 20d high
    runup_z = ((px.pct_change(288) - px.pct_change(288).rolling(2016, min_periods=288).mean())
               / px.pct_change(288).rolling(2016, min_periods=288).std()).shift(1)

    def melt(df, name):
        m = df.reset_index().melt("open_time", var_name="symbol", value_name=name)
        return m
    life = melt(r24, "r24")
    for d, n in ((r72, "r72"), (r7d, "r7d"), (dist_hi, "dist_hi"), (runup_z, "runup_z")):
        life = life.merge(melt(d, n), on=["open_time", "symbol"], how="outer")
    life["open_time"] = pd.to_datetime(life["open_time"], utc=True)

    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "alpha_vs_btc_realized",
                                           "atr_pct", "funding_rate_z_7d", "idio_max_abs_12b"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    D = pan.merge(life, on=["symbol", "open_time"], how="left").dropna(
        subset=["alpha_vs_btc_realized", "atr_pct"])

    # ---- B: conditional direction within the big-move-coming cohort ----
    # cohort = per-cycle top-decile atr_pct (PIT "primed for a big move")
    thr = D.groupby("open_time")["atr_pct"].transform(lambda s: s.quantile(0.90))
    C = D[D["atr_pct"] >= thr].copy()
    C["sgn"] = np.sign(C["alpha_vs_btc_realized"])
    syms = sorted(C["symbol"].unique())
    rng = np.random.RandomState(SEED); shf = syms.copy(); rng.shuffle(shf)
    gmap = {s: i % 5 for i, s in enumerate(shf)}
    C["g"] = C["symbol"].map(gmap)
    LIFE_FEATS = ["r24", "r72", "r7d", "dist_hi", "runup_z", "funding_rate_z_7d",
                  "idio_max_abs_12b"]
    res_B = {}
    for f in LIFE_FEATS:
        accs = []
        for g in range(5):
            tr = C[(C["g"] != g) & C[f].notna()]
            te = C[(C["g"] == g) & C[f].notna()]
            if len(tr) < 500 or len(te) < 200: continue
            # learn sign of feature->residual relationship on TRAIN groups
            rel = np.sign(np.corrcoef(tr[f].rank(), tr["sgn"])[0, 1])
            if rel == 0: rel = 1.0
            pred_sign = rel * (te[f] - te[f].median())     # +→predict up
            a, n = _dirhit(pred_sign.to_numpy(), te["sgn"].to_numpy())
            if not np.isnan(a): accs.append(a)
        # placebo: shuffled residual sign
        pl = []
        Cs = C[C[f].notna()].copy()
        ysh = Cs["sgn"].to_numpy().copy(); rng.shuffle(ysh)
        for g in range(5):
            trm = (Cs["g"] != g).to_numpy(); tem = (Cs["g"] == g).to_numpy()
            if trm.sum() < 500 or tem.sum() < 200: continue
            rel = np.sign(np.corrcoef(Cs[f][trm].rank(), ysh[trm])[0, 1]) or 1.0
            ps = rel * (Cs[f][tem] - Cs[f][tem].median())
            a, _ = _dirhit(ps.to_numpy(), ysh[tem])
            if not np.isnan(a): pl.append(a)
        res_B[f] = {"oos_dir_acc": round(float(np.mean(accs)), 4) if accs else None,
                    "per_group": [round(x, 3) for x in accs],
                    "placebo_acc": round(float(np.mean(pl)), 4) if pl else None,
                    "n_cohort": int(C[f].notna().sum())}

    # ---- A: event-time path of entered LONG winners vs losers ----
    legs = pd.read_parquet(LEGS)
    legs["time"] = pd.to_datetime(legs["time"], utc=True)
    L = legs[legs["side"] == "long"].copy()
    L["win"] = L["contrib_bps"] > 0
    pxd = {c: px[c] for c in px.columns}
    offs = list(range(-288, 289, 24))    # ±24h, 2h steps
    def path(rows):
        accP = {o: [] for o in offs}
        for _, r in rows.sample(min(len(rows), 800), random_state=1).iterrows():
            s = r["symbol"]
            if s not in pxd: continue
            ser = pxd[s]
            try: i = ser.index.get_indexer([r["time"]], method="nearest")[0]
            except Exception: continue
            if i < 300 or i > len(ser) - 300: continue
            base = ser.iloc[i]
            if not np.isfinite(base) or base <= 0: continue
            for o in offs:
                j = i + o
                if 0 <= j < len(ser):
                    accP[o].append(ser.iloc[j] / base - 1.0)
        return {int(o): round(float(np.nanmean(v)) * 100, 3) for o, v in accP.items() if v}
    res_A = {"long_winners_pathpct": path(L[L.win]),
             "long_losers_pathpct": path(L[~L.win])}

    # ---- C: ex-VVV winners by best lifecycle bucket ----
    best = max((k for k in res_B if res_B[k]["oos_dir_acc"]),
               key=lambda k: res_B[k]["oos_dir_acc"] or 0, default="r24")
    exv = L[(L["symbol"] != "VVVUSDT") & L.win].merge(
        life[["symbol", "open_time", best]].rename(columns={"open_time": "time"}),
        on=["symbol", "time"], how="left").dropna(subset=[best])
    if len(exv):
        q = pd.qcut(exv[best], 5, labels=False, duplicates="drop")
        bucket_share = (exv.groupby(q)["contrib_bps"].sum()
                        / exv["contrib_bps"].sum()).round(3).to_dict()
    else:
        bucket_share = {}
    out = {"B_conditional_direction": res_B,
           "A_event_path": res_A,
           "C_exVVV_winner_share_by_%s_quintile" % best: {str(k): v for k, v in bucket_share.items()},
           "interpretation_keys": {
             "B": "oos_dir_acc materially >0.50 AND > placebo within the "
                  "primed cohort => lifecycle gives CONDITIONAL DIRECTION "
                  "(point 1 validated, NEW lever). ~0.50 => still no direction.",
             "A": "winners path rising into t=0 then falling after = "
                  "momentum-into-pump then dump (point 1 structure); flat = "
                  "no exploitable shape at 4h cadence."},
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "probe_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str), flush=True)
    print("LIFECYCLE_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
