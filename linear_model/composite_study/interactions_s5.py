"""§5-INT — interaction features (owner-authorized, pre-registered LOCKED).

Tests the 26 locked interactions (spot-perp / price-volume / order-flow /
short×long / OI), the lever the §4 binding constraint (standalone signal
too weak) actually points at. Pre-reg: FEATURE_REENGINEERING_PLAN.md
§5-INT. Reuses harness_v3's 3-agent-cleared pieces UNCHANGED; adds a
BLOCKING interaction-specific PIT self-check (interactions are the
classic look-ahead trap; CLAUDE.md: |IC|>0.10 ⇒ probable leak).

Two tiers (honest about the universe-shrink trap):
  A  42 syms, NO shrink — 13 interactions needing only the base panel
     (price-volume, short×long, funding/dom/beta).
  B  19 syms perp∩spot∩oi — all 26 (adds spot-perp/order-flow/OI),
     gate WITHIN-universe (vs 19-sym base-only; Success-B vs the fixed
     production V3.1 per-cycle series on the matched cycle grid).

Gate per tier: Success-A standalone net Sh>+1.5 @MAKER (CI excl 0,
≥⅔ folds+, no fold>60%, beats P2) OR Success-B nested var-min blend
lift≥+0.30 vs matched-grid V3.1 (paired CI excl 0, no fold>60%); AND
mandatory marginal-lift: base+ENG must beat base-only by ≥+0.30 within
the SAME universe/cycles. Both cost rates. Production LGBM untouched.
"""
from __future__ import annotations
import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


H = _imp("harness_v3", "linear_model/composite_study/harness_v3.py")
S4 = _imp("ensemble_s4", "linear_model/composite_study/ensemble_s4.py")
ANN, MAKER, COST, OOS, PIT_T = H.ANN, H.MAKER, H.COST, H.OOS, H.PIT_THRESH
OUTD = REPO / "linear_model/composite_study/results"
OUTD.mkdir(parents=True, exist_ok=True)
V31_CSV = REPO / "outputs/vBTC_sleeve_horizon/per_cycle_V3.1_equal6_baseline.csv"
OI = REPO / "outputs/vBTC_features_oi/oi_panel.parquet"
SPOT = REPO / "outputs/vBTC_features_spot/spot_panel.parquet"
OFLOW = REPO / "outputs/vBTC_features_oflow/oflow_panel.parquet"
SA_GATE, SB_GATE, MARGIN_GATE, WIN_CAP = 1.5, 0.30, 0.30, 0.60
sh = S4.sh

# LOCKED 26 interactions (feature_reengineering.py Stage 1). TIER-A = the
# 13 needing ONLY the base panel; TIER-B adds the 13 spot/oflow/oi ones.
TIER_A = ["x_r1d_r8h", "x_r1d_sq", "x_st_r1d", "x_st_autoc", "x_r1d_vws",
          "x_r1d_volz", "x_st_volz", "x_r1d_obv", "x_st_idiov",
          "x_fz_r1d", "x_st_corrb", "x_st_betac", "x_r1d_domz"]
TIER_B_EXTRA = ["x_basis_r1d", "x_spvr_r1d", "x_spti_st", "x_basis_fz",
                "x_ofi_r1d", "x_ofi_st", "x_oftfi_volz", "x_ofkyle_st",
                "x_ofi_oichg", "x_ofi_spti", "x_oic_r1d", "x_oiz_st",
                "x_lst_r1d"]


def build_interactions(d):
    sgn = np.sign
    I = {
        "x_r1d_r8h": d.return_1d * d.return_8h,
        "x_r1d_sq": sgn(d.return_1d) * d.return_1d ** 2,
        "x_st_r1d": d.s_t * d.return_1d,
        "x_st_autoc": d.s_t * d.autocorr_pctile_7d,
        "x_r1d_vws": d.return_1d * d.vwap_slope_96,
        "x_r1d_volz": d.return_1d * d.vol_zscore_4h_over_7d,
        "x_st_volz": d.s_t * d.vol_zscore_4h_over_7d,
        "x_r1d_obv": d.return_1d * d.obv_z_1d,
        "x_st_idiov": d.s_t * d.idio_vol_to_btc_1d,
        "x_fz_r1d": d.funding_rate_z_7d * sgn(d.return_1d),
        "x_st_corrb": d.s_t * d.corr_to_btc_1d,
        "x_st_betac": d.s_t * d.beta_to_btc_change_5d,
        "x_r1d_domz": d.return_1d * d.dom_btc_z_1d,
    }
    if "sp_basis_z1d" in d.columns:
        I.update({
            "x_basis_r1d": d.sp_basis_z1d * d.return_1d,
            "x_spvr_r1d": d.sp_volratio_z1d * d.return_1d,
            "x_spti_st": d.sp_taker_imb_1d * d.s_t,
            "x_basis_fz": d.sp_basis_z1d * d.funding_rate_z_7d,
            "x_ofi_r1d": d.of_imb_1d * d.return_1d,
            "x_ofi_st": d.of_imb_1d * d.s_t,
            "x_oftfi_volz": d.of_tfi_z1d * d.vol_zscore_4h_over_7d,
            "x_ofkyle_st": d.of_kyle_1d * d.s_t,
            "x_ofi_oichg": d.of_imb_1d * d.oi_chg_1d,
            "x_ofi_spti": d.of_imb_1d * d.sp_taker_imb_1d,
            "x_oic_r1d": d.oi_chg_1d * sgn(d.return_1d),
            "x_oiz_st": d.oi_z_7d * d.s_t,
            "x_lst_r1d": d.ls_taker_z_1d * sgn(d.return_1d),
        })
    out = d.copy()
    for k, v in I.items():
        out[k] = v.astype("float64")
    return out, list(I.keys())


LEAK_ASYM_TOL = 0.02   # G3 sampling-noise band (LOCKED, pre-reg §5-INT-v2)


def selfcheck_interaction_guard(dec, icols):
    """BLOCKING leak guard (pre-reg §5-INT-v2 CORRECTED — leak-specific,
    not the mis-calibrated magnitude-abort):
      G1 signed-PIT : max |spearman(xsr(feat)_t, αβ_{t+1})| < 0.10
                      (project-canonical signed look-ahead sniff).
      G3 leak-asym  : a strictly-past PIT feature must NOT predict the
                      FUTURE better than the CONTEMPORANEOUS bar — for
                      every interaction, signed AND magnitude, abort iff
                      |corr(feat_t, αβ_{t+1})| > |corr(feat_t, αβ_t)|
                      + 0.02. Benign persistence decays (next≤same)⇒PASS;
                      a genuine forward leak has next≳same ⇒ ABORT.
    (G2 prefix-causal = selfcheck_interaction_causal, unchanged.)"""
    F = H._xsrank_invnorm(dec, icols)
    d = dec.sort_values(["symbol", "open_time"]).copy()
    d["ab_t"] = d["alpha_beta"]
    d["ab_t1"] = d.groupby("symbol")["alpha_beta"].shift(-1)
    F = F.reindex(d.index)
    keep = d["ab_t1"].notna() & F.notna().all(axis=1)
    abt = pd.Series(d.loc[keep, "ab_t"].values)
    abt1 = pd.Series(d.loc[keep, "ab_t1"].values)
    g1, g1f, g3, g3f = 0.0, None, -9.0, None
    for c in icols:
        v = pd.Series(F.loc[keep, c].values)
        s1 = v.corr(abt1, method="spearman")           # G1 next-cycle
        if s1 is not None and abs(s1) > g1:
            g1, g1f = abs(s1), c
        for lab, x, yt, yt1 in ((c, v, abt, abt1),
                                (f"|{c}|", v.abs(), abt.abs(),
                                 abt1.abs())):           # G3 signed+mag
            cs = x.corr(yt, method="spearman")
            cn = x.corr(yt1, method="spearman")
            if cs is None or cn is None:
                continue
            asym = abs(cn) - abs(cs)
            if asym > g3:
                g3, g3f = asym, lab
    ok = bool(g1 < PIT_T and g3 <= LEAK_ASYM_TOL)
    return ok, dict(g1=g1, g1f=g1f, g3=g3, g3f=g3f)


def selfcheck_interaction_causal(dec, icols, n=3, seed=0):
    """BLOCKING: the interaction xsr is prefix-causal — recompute on a
    strictly-past prefix at interior cuts; value at the cut == whole-panel
    value (per-cycle rank uses only the contemporaneous cross-section, so
    truncating future rows cannot change a past row). Δ must be 0."""
    full = H._xsrank_invnorm(dec, icols)
    rng = np.random.default_rng(seed)
    syms = rng.choice(dec.symbol.unique(),
                      min(n, dec.symbol.nunique()), replace=False)
    worst = 0.0
    for s in syms:
        ds = dec[dec.symbol == s].sort_values("open_time")
        if len(ds) < 80:
            continue
        for fr in (0.5, 0.75, 0.95):
            i = int(len(ds) * fr)
            t = ds.open_time.iloc[i]
            ridx = ds.index[i]
            pref = dec[dec.open_time <= t]
            fp = H._xsrank_invnorm(pref, icols)
            if ridx not in fp.index:
                continue
            a = full.loc[ridx, icols].to_numpy(float)
            b = fp.loc[ridx, icols].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.any():
                worst = max(worst, float(np.max(np.abs(a[m] - b[m]))))
    return worst < 1e-9, worst


def make_feature_matrix(dec, icols):
    """Base feats via harness frozen map; interactions via xsr (plan
    §2a). Returns (F, feats, drop)."""
    Fb, dropb = H.preprocess(dec)
    Fi = H._xsrank_invnorm(dec, icols) if icols else pd.DataFrame(
        index=dec.index)
    F = pd.concat([Fb, Fi], axis=1)
    feats_base = list(Fb.columns)
    drop = dropb | F.isna().any(axis=1)
    return F, feats_base, list(Fi.columns), drop


def wf(dec, F, feats, folds, drop):
    """harness_v3.walk_forward masks/target/envelope, arbitrary feats."""
    sig = H.strict_sigma_idio(dec)
    b = dec.copy()
    b["sig"] = sig
    b["tz"] = (b.alpha_beta / b.sig).clip(-5, 5)
    elig = (~drop) & b.sig.notna() & (b.sig > 1e-12) & b.tz.notna()
    recs = []
    for fid in OOS:
        fo = folds[fid]
        tr = (dec.open_time < fo["cal_start"]) & elig
        ca = ((dec.open_time >= fo["cal_start"]) &
              (dec.open_time < fo["cal_end"]) &
              (dec.exit_time < fo["test_start"]) & elig)
        te = (dec.fold == fid) & elig
        if tr.sum() < 600 or ca.sum() < 40 or te.sum() < 15:
            continue
        pr = H.model_envelope(
            F.loc[tr, feats].to_numpy(float), b.loc[tr, "tz"].to_numpy(float),
            F.loc[ca, feats].to_numpy(float), b.loc[ca, "tz"].to_numpy(float),
            F.loc[te, feats].to_numpy(float))
        r = b.loc[te, ["symbol", "open_time", "fold", "alpha_beta"]].copy()
        r["ridge"] = pr["ridge_best"]
        r["lgbm"] = pr["lgbm_es"]
        recs.append(r)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def eval_book(pf, predcol, v):
    """Standalone (both costs) + Success-B nested var-min vs matched
    V3.1. Returns dict per cost."""
    res = {}
    for c, cl in ((MAKER, "MAKER"), (COST, "COST2.25")):
        lb = H.linear_book(pf, predcol, cost=c).rename(
            columns={"net": "lin"})
        m = lb.merge(v, on="open_time", how="inner").sort_values(
            "open_time").reset_index(drop=True)
        if len(m) < 50:
            res[cl] = None
            continue
        lin_sh = sh(m.lin.to_numpy())
        lo, hi = H.block_bootstrap_ci(m.lin.to_numpy(), sh,
                                      block=S4.BOOT_BLOCK, n_boot=S4.N_BOOT)
        ef = sorted(m.fold.unique())
        fsh = m.groupby("fold").lin.apply(lambda g: sh(g.to_numpy()))
        fpos = int((fsh > 0).sum())
        fp = m.groupby("fold").lin.sum()
        share = float(fp.abs().max() / fp.abs().sum()) \
            if fp.abs().sum() > 1e-9 else np.nan
        w = np.zeros(len(m))
        for i, f in enumerate(ef):
            mk = (m.fold == f).to_numpy()
            if i == 0:
                continue
            pr = m[m.fold.isin(ef[:i])]
            L0, V0 = pr.lin.to_numpy(), pr.v31.to_numpy()
            den = L0.var() + V0.var() - 2 * np.cov(L0, V0)[0, 1]
            w[mk] = float(np.clip((V0.var() - np.cov(L0, V0)[0, 1]) / den
                                  if abs(den) > 1e-12 else 0.0, 0, 1))
        bl = w * m.lin.to_numpy() + (1 - w) * m.v31.to_numpy()
        v31m = sh(m.v31.to_numpy())
        lift = sh(bl) - v31m
        blo, bhi = S4.paired_lift_ci(m.lin.to_numpy(), m.v31.to_numpy(), w)
        res[cl] = dict(lin_sh=lin_sh, ci_lo=lo, ci_hi=hi, fpos=fpos,
                       nf=len(ef), share=share, v31m=v31m, lift=lift,
                       blo=blo, bhi=bhi, n=len(m))
    return res


def main():
    print("=" * 94, flush=True)
    print("  §5-INT — interaction features (owner-authorized; "
          "pre-registered LOCKED)", flush=True)
    print("=" * 94, flush=True)
    t0 = time.time()
    dec, folds, btc, pan = H.load_panel()
    if not H.run_selfchecks(dec):
        print("\n  ABORT — base self-checks failed.", flush=True)
        return
    v = pd.read_csv(V31_CSV)
    v["open_time"] = pd.to_datetime(v["time"], utc=True)
    v = v[["open_time", "net_pnl_bps"]].rename(
        columns={"net_pnl_bps": "v31"})

    rows = []
    for tier in ("A", "B"):
        print(f"\n{'='*70}\n  TIER {tier}\n{'='*70}", flush=True)
        d = dec
        if tier == "B":
            for pth in (OI, SPOT, OFLOW):
                p = pd.read_parquet(pth)
                p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
                d = d.merge(p, on=["symbol", "open_time"], how="inner")
            d = d.sort_values(["symbol", "open_time"]).reset_index(
                drop=True)
        d, all_ic = build_interactions(d)
        icols = TIER_A if tier == "A" else TIER_A + TIER_B_EXTRA
        icols = [c for c in icols if c in d.columns]
        print(f"  universe={d.symbol.nunique()} syms  rows={len(d)}  "
              f"cycles={d.open_time.nunique()}  interactions={len(icols)}",
              flush=True)
        # BLOCKING interaction leak guard (pre-reg §5-INT-v2: G1+G2+G3)
        cG, gi = selfcheck_interaction_guard(d, icols)
        c2, w2 = selfcheck_interaction_causal(d, icols)
        print(f"  [leak guard] G1 signed-PIT |corr|={gi['g1']:.4f} on "
              f"'{gi['g1f']}' ({'PASS' if gi['g1']<PIT_T else 'FAIL'}, "
              f"<{PIT_T}) | G3 leak-asym (next−same)={gi['g3']:+.4f} on "
              f"'{gi['g3f']}' ({'PASS' if gi['g3']<=LEAK_ASYM_TOL else 'FAIL'}"
              f", ≤{LEAK_ASYM_TOL}) | G2 prefix-causal Δ={w2:.2e} "
              f"({'PASS' if c2 else 'FAIL'}, <1e-9)", flush=True)
        if not (cG and c2):
            print(f"  ⛔ TIER {tier} ABORT — interaction leak guard "
                  f"FAILED. Not scored.", flush=True)
            rows.append(dict(tier=tier, status="LEAK-ABORT",
                             g1=gi['g1'], g3=gi['g3'], causal=w2))
            continue
        F, fb, fi, drop = make_feature_matrix(d, icols)
        base_pf = wf(d, F, fb, folds, drop)
        treat_pf = wf(d, F, fb + fi, folds, drop)
        if not len(base_pf) or not len(treat_pf):
            print(f"  TIER {tier}: insufficient folds — skip", flush=True)
            continue
        for member in ("ridge", "lgbm"):
            be = eval_book(base_pf, member, v)
            te = eval_book(treat_pf, member, v)
            for cl in ("MAKER", "COST2.25"):
                b_, t_ = be.get(cl), te.get(cl)
                if b_ is None or t_ is None:
                    continue
                margin = t_["lin_sh"] - b_["lin_sh"]
                gA = bool(cl == "MAKER" and t_["lin_sh"] > SA_GATE
                          and t_["ci_lo"] > 0
                          and t_["fpos"] >= int(np.ceil(t_["nf"]*2/3))
                          and t_["share"] <= WIN_CAP)
                if gA:   # P2 placebo (pre-reg §5-INT-v2; lazy — only if
                         # the other Success-A conditions already hold)
                    p2 = H.p2_placebo(treat_pf, member, n_perm=1000,
                                      cost=MAKER)
                    p2p95 = float(np.percentile(p2, 95))
                    gA = bool(t_["lin_sh"] > p2p95)
                    print(f"      [P2] Success-A pre-pass → real "
                          f"{t_['lin_sh']:+.3f} vs placebo p95 "
                          f"{p2p95:+.3f} ⇒ {'PASS' if gA else 'FAIL'}",
                          flush=True)
                gB = bool(t_["lift"] >= SB_GATE and t_["blo"] > 0
                          and t_["share"] <= WIN_CAP)
                gM = bool(margin >= MARGIN_GATE)
                pas = bool((gA or gB) and gM)
                print(f"  [{member}/{cl:8}] base Sh={b_['lin_sh']:+.3f} "
                      f"→ base+ENG Sh={t_['lin_sh']:+.3f} "
                      f"(margin {margin:+.3f}, need ≥+{MARGIN_GATE}) | "
                      f"CI[{t_['ci_lo']:+.2f},{t_['ci_hi']:+.2f}] "
                      f"f+={t_['fpos']}/{t_['nf']} sh={t_['share']:.0%} | "
                      f"nested lift={t_['lift']:+.3f} "
                      f"CI[{t_['blo']:+.2f},{t_['bhi']:+.2f}] "
                      f"(V31m {t_['v31m']:+.2f}) | A={'P' if gA else '·'} "
                      f"B={'P' if gB else '·'} M={'P' if gM else '·'} "
                      f"⇒ {'PASS' if pas else 'fail'}", flush=True)
                rows.append(dict(tier=tier, member=member, cost=cl,
                                 base_sh=b_["lin_sh"],
                                 treat_sh=t_["lin_sh"], margin=margin,
                                 ci_lo=t_["ci_lo"], ci_hi=t_["ci_hi"],
                                 fpos=t_["fpos"], nf=t_["nf"],
                                 share=t_["share"], lift=t_["lift"],
                                 blo=t_["blo"], bhi=t_["bhi"],
                                 v31m=t_["v31m"], passA=gA, passB=gB,
                                 passM=gM, PASS=pas))
    R = pd.DataFrame(rows)
    R.to_csv(OUTD / "interactions_s5.csv", index=False)
    won = R[R.get("PASS", False) == True] if "PASS" in R else pd.DataFrame()
    print("\n" + "=" * 94, flush=True)
    if len(won):
        verdict = ("§5-INT SIGNAL — interaction cell(s) clear the locked "
                   "gate: " + "; ".join(
                       f"T{r.tier}/{r.member}@{r.cost} "
                       f"(margin{r.margin:+.2f},A={r.passA},B={r.passB})"
                       for _, r in won.iterrows())
                   + ". The interaction lever is real. PROCEED to §7 "
                   "strict nested forward validation before any claim. "
                   "Do NOT over-claim — §7 is the bar.")
    else:
        verdict = ("§5-INT NEGATIVE — no interaction tier/member/cost "
                   "clears Success-A or Success-B AND none lifts "
                   "base-only by ≥+0.30 (the mandatory marginal gate). "
                   "Spot-perp / price-volume / order-flow / short×long "
                   "interactions do NOT add tradeable signal on the "
                   "trustworthy harness. Combined with §4 + §3.5: the "
                   "linear β-residual line is closed INCLUDING the "
                   "feature-interaction lever — the owner's specific "
                   "question answered honestly. A NARROW negative (free "
                   "4h scope); orthogonal PAID data / longer horizon "
                   "remain the only untested, out-of-scope levers. "
                   "Production LGBM unaffected.")
    print(f"  VERDICT: {verdict}", flush=True)
    pd.DataFrame([{"any_pass": bool(len(won)), "verdict": verdict}]
                 ).to_csv(OUTD / "interactions_s5_verdict.csv",
                          index=False)
    print(f"\nSaved {OUTD}/interactions_s5*.csv  "
          f"Total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
