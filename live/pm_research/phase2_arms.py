"""PHASE 2 — three heads on the top-up, per the committed declaration.

AUTHORISATION (R-126, in-file): R-170/R-173. Governed by `phase2_declaration`,
which was committed BEFORE any Phase-2 number existed (rule 11). Nothing here
chooses a threshold, window, feature set or hyperparameter -- if this file and
the declaration ever disagree, the declaration wins and this file is wrong.

THE ARCHITECTURE, and the reason it is not the obvious one:
  FIT on the CONSUMED FRAGMENT   (already spent; fitting costs nothing new)
  SCORE on the TOP-UP            (held out; R-166(3) reserved it)
Fitting on the top-up would consume the only held-out tape Phase 2 has and
make all three heads in-sample at once.

  A  PM_PLUS_FINE       the FROZEN artifact APPLIED UNCHANGED. No refit, no
                        weighting. R-157(2): the incumbent is not rewritten
                        mid-comparison, so arm A does not even load a fitter.
  B  PLUS_PRED_STATE_V1 A's features + DA's 21 state features, fitted on the
                        fragment WITH w = 1/n_rows(generation).
  C  LGBM_PINNED        same features and weighting, capacity pinned in the
                        declaration, seed pinned, no grid, no early stopping.

Every arm scores the SAME top-up rows. A row any arm cannot feature is dropped
from ALL arms, so the comparison stays paired -- an arm scoring a different
population is not a comparison at all.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

sys.path.insert(0, "/home/yuqing/ctaNew/live/pm_research")
import phase2_declaration as D

DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
TOPUP = DERIVED / "harmful_exposure_rows_v3_topup.json"
FRAGMENT = DERIVED / "harmful_exposure_rows_v3_eraB.json"
FROZEN = DERIVED / "harmful_reduced_fine_candidate_v1.json"
OUT = DERIVED / "phase2_three_arm_v1.json"


class PopulationLeak(RuntimeError):
    """A scoring row came from the fitting population."""


def assert_disjoint(fit_slugs: set, score_slugs: set) -> None:
    """The two populations must not intersect. AT ALL.

    This is the check that would have caught the 808-window build before any
    arm ran: the test set contained every fitting slug, and nothing errored."""
    overlap = fit_slugs & score_slugs
    if overlap:
        raise PopulationLeak(
            f"{len(overlap)} slug(s) appear in BOTH the fitting and scoring "
            f"populations (e.g. {sorted(overlap)[:3]}). Arms B and C are fitted "
            f"on the fragment; scoring them on rows they were fitted on is "
            f"in-sample and FLATTERS them. Refusing.")


def _feature_pass(src: Path, population: str) -> dict:
    """Build PM+fine+state features for every OK row of one population.

    Returns per-coin parallel lists. A row missing ANY family is dropped from
    ALL arms (paired comparison); drops are counted, never silent."""
    import harmful_hazard_model as hm
    import harmful_state_features as sf

    data = json.loads(src.read_text())
    rows = [r for r in data["rows"] if r["status"] == "OK"]
    paths = hm.fi._archive_paths(); tokens = hm.fi.token_map()
    out: dict = {}
    for coin in ("btc", "eth"):
        crows = [r for r in rows if r["coin"] == coin]
        streams: dict = {}; tapes: dict = {}
        PM = []; FN = []; ST = []; kept = []
        drops = {"pm": 0, "fine": 0, "state": 0}
        bywin: dict = {}
        for r in crows:
            bywin.setdefault(r["slug"], []).append(r)
        for slug, wrows in bywin.items():
            up, dn = tokens[slug]
            streams[slug] = hm.window_streams(paths[slug], up, dn)
            try:
                tape = sf.build_tape(paths[slug], up, dn)
            except Exception:
                drops["state"] += len(wrows); continue
            sfeats = sf.features_for_window(tape, wrows)
            for r, sfe in zip(wrows, sfeats):
                fp = hm.features(streams[slug], r["t_start"], r["side"],
                                 r["level"], r["resting"], r["qahead"])
                if fp is None:
                    drops["pm"] += 1; continue
                ff = hm.fine_feats(r["t0"] + r["t_start"], r["side"], coin)
                if ff is None:
                    drops["fine"] += 1; continue
                sv = [float(sfe.get(k) or 0.0) for k in D.PRED_STATE_V1]
                PM.append(fp); FN.append(ff); ST.append(sv)
                kept.append({k: r.get(k) for k in
                             ("slug", "day", "t0", "t_start", "side", "gen",
                              "latency", "coin")})
            streams.pop(slug, None)
        out[coin] = {"PM": PM, "FN": FN, "ST": ST, "kept": kept, "drops": drops}
        print(f"  [{population}/{coin}] kept {len(kept)} rows, drops {drops}",
              flush=True)
    return out


def _labels(kept: list):
    Lh = str(D.TARGET_LATENCY_MS)
    y = [1 if (r.get("latency") or {}).get(Lh, {}).get(
             "preventable_shares", 0.0) > 0 else 0 for r in kept]
    tgt = [(r.get("latency") or {}).get(Lh, {}).get(
               "preventable_value_cents", 0.0) for r in kept]
    return y, tgt


FITDIR = DERIVED / "phase2_fits"


def stage_fit() -> None:
    """STAGE 1: fit B and C on the fragment, persist, exit.

    The single-process version held fragment features + top-up features + LGBM
    training matrices simultaneously and was oom-killed at 14G AFTER all four
    feature passes had succeeded. Splitting is not only a memory fix: it is the
    daily-refit shape -- fit once, persist, apply -- so the scoring stage never
    needs the fitting population in memory at all."""
    import harmful_fast_compute as fc
    import lightgbm as lgb
    import numpy as np
    FITDIR.mkdir(parents=True, exist_ok=True)
    FIT = _feature_pass(FRAGMENT, "fragment")
    for coin in ("btc", "eth"):
        f = FIT[coin]
        yF, tF = _labels(f["kept"])
        XF = [f["PM"][i] + f["FN"][i] + f["ST"][i] for i in range(len(f["kept"]))]
        Xf, mu, sd = fc.fast_zscale(XF, XF)
        sw = fc.fast_generation_weights(f["kept"])
        W = fc.fast_fit_logistic_w(Xf, yF, sw)
        ft = [i for i in range(len(yF)) if yF[i]]
        WM = (fc.fast_fit_ridge_w([Xf[i] for i in ft], [tF[i] for i in ft],
                                  [sw[i] for i in ft], lam=10.0)
              if len(ft) >= 100 else None)
        (FITDIR / f"linear_{coin}.json").write_text(json.dumps(
            {"hazard_weights": list(W), "value_weights": list(WM) if WM else None,
             "norm_mu": list(mu), "norm_sd": list(sd),
             "n_rows": len(f["kept"]), "n_positive": sum(yF),
             "n_actions": len({(r["slug"], r["side"], r["gen"]) for r in f["kept"]}),
             "drops": f["drops"]}))
        A = np.asarray(Xf, dtype=np.float64); swa = np.asarray(sw)
        clf = lgb.LGBMClassifier(**D.LGBM_PARAMS)
        clf.fit(A, np.asarray(yF), sample_weight=swa)
        clf.booster_.save_model(str(FITDIR / f"lgbm_haz_{coin}.txt"))
        ftm = np.asarray(yF) == 1
        if ftm.sum() >= 100:
            reg = lgb.LGBMRegressor(**D.LGBM_VALUE_PARAMS)
            reg.fit(A[ftm], np.asarray(tF)[ftm], sample_weight=swa[ftm])
            reg.booster_.save_model(str(FITDIR / f"lgbm_val_{coin}.txt"))
        print(f"  [fit/{coin}] persisted linear + lgbm; rows {len(f['kept'])}, "
              f"positive {sum(yF)}", flush=True)
        del XF, Xf, A, FIT[coin]["PM"], FIT[coin]["FN"], FIT[coin]["ST"]
    (FITDIR / "fit_slugs.json").write_text(json.dumps(sorted(
        {r["slug"] for c in FIT.values() for r in c["kept"]})))
    print("STAGE FIT COMPLETE", flush=True)


def stage_score() -> dict:
    """STAGE 2: score all three arms on the top-up. Never loads the fragment."""
    import harmful_hazard_model as hm
    import harmful_action_eval as ae
    import harmful_fast_compute as fc
    import lightgbm as lgb
    import numpy as np

    frozen = json.loads(FROZEN.read_text())
    fit_slugs = set(json.loads((FITDIR / "fit_slugs.json").read_text()))
    SC = _feature_pass(TOPUP, "topup")
    assert_disjoint(fit_slugs, {r["slug"] for c in SC.values() for r in c["kept"]})
    print("  populations asserted DISJOINT (fitted slugs read from stage 1)",
          flush=True)

    out = {"protocol": "PHASE2_THREE_ARM_V1", "arms": {}, "population": {},
           "declaration_commit": "d7082b6",
           "multiplicity_before": D.MULTIPLICITY_BEFORE,
           "multiplicity_after": D.MULTIPLICITY_AFTER,
           "n_random": D.N_RANDOM, "decision_metric": D.DECISION_METRIC,
           "lgbm_params": D.LGBM_PARAMS,
           "staged_because": "the single-process run was oom-killed at 14G after "
                             "all four feature passes succeeded; fit and score "
                             "are now separate processes (the daily-refit shape)",
           "da_caveat_field": "RESERVED for Q-DA-79 post-gap queue-validity finding"}

    for coin in ("btc", "eth"):
        sc = SC[coin]
        lin = json.loads((FITDIR / f"linear_{coin}.json").read_text())
        srows = [hm.keptrow(r) for r in sc["kept"]]
        nA = len({(r["slug"], r["side"], r["gen"]) for r in sc["kept"]})
        out["population"][coin] = {
            "score_rows": len(sc["kept"]), "score_actions": nA,
            "score_windows": len({r["slug"] for r in sc["kept"]}),
            "score_drops": sc["drops"], "fit_rows": lin["n_rows"],
            "fit_actions": lin["n_actions"], "fit_positive": lin["n_positive"],
            "fit_drops": lin["drops"]}
        # R-174: NO duplicate materialization. The previous version built
        # XS_lin (a full concatenated copy of 638k x 81) AND a float64 matrix
        # for LGBM, on top of the feature-pass lists already resident -- three
        # copies of the same data, which is the shape that oom-killed the
        # single-process run. Rows are now concatenated ON DEMAND inside each
        # arm, and LGBM is fed in CHUNKS so no full matrix ever exists.
        n_sc = len(sc["kept"])

        def _raw(i):
            return sc["PM"][i] + sc["FN"][i] + sc["ST"][i]

        for arm in D.ARMS:
            if arm == "PM_PLUS_FINE":
                fz = frozen["fits"][coin]
                mu, sd = fz["norm_mu"], fz["norm_sd"]
                W, WM = fz["hazard_weights"], fz["value_weights"]
                ecv = []
                for j in range(len(sc["kept"])):
                    raw = sc["PM"][j] + sc["FN"][j]
                    x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
                    ecv.append(fc.fast_predict_p(W, x) *
                               float(sum(a * b for a, b in zip(WM, x))))
            elif arm == "PLUS_PRED_STATE_V1":
                mu, sd = lin["norm_mu"], lin["norm_sd"]
                W, WM = lin["hazard_weights"], lin["value_weights"]
                ecv = []
                for j in range(n_sc):
                    raw = _raw(j)                       # transient, freed each loop
                    x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
                    ecv.append(fc.fast_predict_p(W, x) *
                               (float(sum(a * b for a, b in zip(WM, x))) if WM else 0.0))
            else:
                mu, sd = lin["norm_mu"], lin["norm_sd"]
                hb = lgb.Booster(model_file=str(FITDIR / f"lgbm_haz_{coin}.txt"))
                vf = FITDIR / f"lgbm_val_{coin}.txt"
                vb = lgb.Booster(model_file=str(vf)) if vf.exists() else None
                CH = 50_000                             # ~32MB per chunk, not ~2GB
                ecv = []
                for lo in range(0, n_sc, CH):
                    hi = min(lo + CH, n_sc)
                    S = np.empty((hi - lo, len(mu) + 1), dtype=np.float64)
                    S[:, 0] = 1.0
                    for j in range(lo, hi):
                        raw = _raw(j)
                        S[j - lo, 1:] = [(raw[i] - mu[i]) / sd[i]
                                         for i in range(len(mu))]
                    p = hb.predict(S)
                    v = vb.predict(S) if vb is not None else np.zeros(hi - lo)
                    ecv.extend((p * v).tolist())
                    del S
            gate = ae.evaluate_policy(srows, ecv, latency_ms=D.TARGET_LATENCY_MS,
                                      budgets=D.BUDGETS, n_random=D.N_RANDOM)
            out["arms"].setdefault(coin, {})[arm] = {"gate": gate}
            print(f"  {coin} {arm:<20} n_actions={gate.get('n_actions')}", flush=True)
            for b, g in gate["budgets"].items():
                print(f"      @{b}: net {g['net_cents']:+9.1f}c  "
                      f"rand_max {g['random_net_max']:+8.1f}  "
                      f"beats_NET={g['beats_random_max_on_NET']}", flush=True)
        # free this coin's features before the next coin is built
        SC[coin]["PM"] = []; SC[coin]["FN"] = []; SC[coin]["ST"] = []
        del sc, srows
    import os, tempfile
    fd, tmp = tempfile.mkstemp(dir=str(OUT.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, OUT)
    print(f"\nWROTE {OUT.name}", flush=True)
    return out


def run() -> dict:
    """SUPERSEDED. The single-process path was oom-killed at 14G and R-174
    mandates the staged pattern. Kept as a refusal rather than deleted so a
    caller reaching for it gets told why, instead of finding it missing and
    reinventing it."""
    raise SystemExit(
        "REFUSED: the single-process path is superseded. It held the fragment "
        "features, the top-up features and the LGBM matrices at once and was "
        "oom-killed at 14G after all four feature passes had succeeded. "
        "Use --stage-fit then --stage-score (R-174: restructure, never raise "
        "the cap).")


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    assert_disjoint({"a", "b"}, {"c", "d"})
    ok(True, "KNOWN-GOOD: disjoint populations pass")
    try:
        assert_disjoint({"a", "b"}, {"b", "c"})
        ok(False, "an overlapping slug must be refused")
    except PopulationLeak as e:
        ok("FLATTERS" in str(e),
           "POSITIVE CONTROL: a single shared slug is REFUSED, naming the "
           "direction of the harm -- this is the 808-window failure, caught "
           "before any arm runs rather than after")
    ok(D.ARMS[0] == "PM_PLUS_FINE" and "PM_PLUS_FINE" not in D.WEIGHTED_ARMS,
       "arm A is the unweighted frozen incumbent, per the declaration")
    ok(D.N_RANDOM >= 200 and D.DECISION_METRIC == "net_cents",
       "declared null and decision metric come from the declaration, not "
       "from this file")
    ok(D.MULTIPLICITY_AFTER == 4, "multiplicity 4 after scoring two candidates")
    ok(len(D.PRED_STATE_V1) == 21, "PRED_STATE_V1 is the declared 21 features")
    ok("early_stopping" not in D.LGBM_PARAMS,
       "arm C has no early stopping -- nothing about it is chosen after "
       "seeing the scoring side")
    print(f"phase2_arms selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    selftest()
    if not TOPUP.exists():
        raise SystemExit(f"REFUSED: {TOPUP.name} does not exist. Phase 2 has "
                         f"no test surface; build it before scoring.")
    if "--stage-fit" in sys.argv:
        stage_fit(); return 0
    if "--stage-score" in sys.argv:
        stage_score(); return 0
    raise SystemExit("specify --stage-fit or --stage-score (staged after the "
                     "single-process run was oom-killed at 14G)")


if __name__ == "__main__":
    raise SystemExit(main())
