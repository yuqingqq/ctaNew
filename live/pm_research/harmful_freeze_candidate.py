"""Build the frozen reduced-fine candidate. RUNS ONLY ON THE USER'S YES.

AUTHORISATION (R-126, in-file): R-166(2) instructs BE to draft the freeze
receipt as an ASK. The FREEZE DECISION IS THE USER'S (R-162(3)); this module
exists so the ASK describes something concrete and so the freeze act is one
reviewable command rather than an improvisation after the yes.

WHY A SEPARATE MODULE. `harmful_hazard_model.py` just passed a cent-exact
reproduction gate. Adding a freeze mode to it would change the file whose
hash the manifest pins and whose output was verified to the cent. So the
freeze builder imports the gated pipeline and does not edit it.

WHAT THE REFIT IS, AND WHAT IT IS NOT (R-166(3)):
  IS  : one fit over the CONSUMED FRAGMENT ONLY -- both days, through slug
        1787650200 -- producing the coefficient vector that would be deployed.
        `run --fine` fits on days[:-1] (08-24) and scores 08-25; a deployed
        candidate should use every consumed row it is allowed to use.
  NOT : any use of the R-145(3) top-up. That tape is Phase 2's untouched
        development-test surface. Training on it here would consume it and
        change the incumbent mid-race.

NO IN-SAMPLE VERDICT IS STORED. The refit's own fit-quality numbers are not
evidence of anything -- they are in-sample by construction. The evidence for
this candidate is the ALREADY-COMPLETED paired comparison; the freeze does not
re-argue it. Rule 14: this estimates, it does not decide.
"""
from __future__ import annotations

import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"
LAST_CONSUMED_SLUG_T0 = 1787650200          # R-166(3) boundary, inclusive
FREEZE_COINS = ("btc", "eth")               # explicit; matches the ASK
OUT = DERIVED / "harmful_reduced_fine_candidate_v1.json"


class TopUpLeak(RuntimeError):
    """A row outside the consumed fragment reached the freeze refit."""


def assert_consumed_fragment_only(slug_t0s) -> None:
    """REFUSE if any row post-dates the consumed fragment.

    The top-up is Phase 2's development-test surface. If it leaked into the
    freeze refit, the incumbent would have been trained on the tape it is about
    to be tested against, and nobody would see it in the numbers -- the fit
    would simply look slightly better. So this refuses loudly instead."""
    late = sorted({t for t in slug_t0s if t > LAST_CONSUMED_SLUG_T0})
    if late:
        raise TopUpLeak(
            f"{len(late)} slug(s) after the consumed fragment reached the "
            f"freeze refit (first {late[0]}, last {late[-1]}; boundary "
            f"{LAST_CONSUMED_SLUG_T0}). R-166(3) excludes the top-up from the "
            f"freeze: training on Phase 2's test surface would consume it and "
            f"move the incumbent mid-race.")


def atomic_write_json(path: Path, obj: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True); raise


def refit(user_approved: bool = False) -> dict:
    """ONE fit over the consumed fragment, both days. Mirrors the gated
    pipeline exactly; the only difference is that the TRAIN SET IS EVERY
    CONSUMED ROW rather than day one."""
    if not user_approved:
        raise RuntimeError("refit() requires explicit user approval")
    import harmful_hazard_model as hm

    data = json.loads(hm.ROWS_ERA.read_text())
    if data.get("schema") != hm.EXPECTED_SCHEMA:
        raise SystemExit(f"REFUSED: schema {data.get('schema')!r}")
    rows = [r for r in data["rows"] if r["status"] == "OK"]
    assert_consumed_fragment_only(
        int(r["slug"].rsplit("-", 1)[1]) for r in rows)      # armed, not decorative
    paths = hm.fi._archive_paths(); tokens = hm.fi.token_map()
    Lh = str(hm.TARGET_LATENCY_MS)
    out: dict = {}

    # COINS is a LOCAL inside run_fine, not a module attribute -- the first
    # draft assumed it was importable and died in 15s. Declared here instead,
    # because the freeze's coin scope should be explicit in the freeze builder
    # rather than inherited from another function's internals.
    for coin in FREEZE_COINS:
        crows = [r for r in rows if r["coin"] == coin]
        streams: dict = {}
        PM: list = []; FN: list = []; kept: list = []
        for r in crows:
            slug = r["slug"]
            if slug not in streams:
                up, dn = tokens[slug]
                streams[slug] = hm.window_streams(paths[slug], up, dn)
                if len(streams) > 4:
                    streams.pop(next(iter(streams)))
            fp = hm.features(streams[slug], r["t_start"], r["side"],
                             r["level"], r["resting"], r["qahead"])
            if fp is None:
                continue
            ff = hm.fine_feats(r["t0"] + r["t_start"], r["side"], coin)
            if ff is None:
                continue
            PM.append(fp); FN.append(ff)
            kept.append({k: r.get(k) for k in
                         ("slug", "day", "t0", "t_start", "side", "gen",
                          "latency")})
        if not kept:
            raise hm.EmptyFeatureSet(
                f"{coin}: ZERO rows survived feature construction in the "
                f"freeze refit. Check the era floor before anything else.")
        XA = [PM[i] + FN[i] for i in range(len(kept))]
        allidx = list(range(len(kept)))                  # <-- the whole point
        y = [1 if (kept[i].get("latency") or {}).get(Lh, {}).get(
                 "preventable_shares", 0.0) > 0 else 0 for i in allidx]
        tgt = lambda i: kept[i]["latency"][Lh]["preventable_value_cents"]
        Xs, mu, sd = hm.zscale(XA, XA)                   # train == all rows
        w = hm.fit_logistic(Xs, y)
        ft = [i for i in allidx if y[i]]
        wm = (hm.fit_ridge([Xs[i] for i in ft], [tgt(i) for i in ft], lam=10.0)
              if len(ft) >= 100 else None)
        days = sorted({r["day"] for r in kept})
        out[coin] = {
            "n_rows_fitted": len(kept),
            "n_actions_fitted": len({(r["slug"], r["side"], r["gen"])
                                     for r in kept}),
            "n_positive": sum(y),
            "days_fitted": days,
            "hazard_weights": w,
            "value_weights": wm,
            "norm_mu": mu, "norm_sd": sd,
            "feature_vector_contract": {
                "layout": "[1.0] + [(raw[i] - norm_mu[i]) / norm_sd[i] "
                          "for i in 0..n-1]",
                "n_weights": len(w),
                "n_norm_params": len(mu),
                "intercept_is_position_0": True,
                "norm_applies_to_positions": "1..%d" % len(mu),
                "block_order": "54 PM features, then 6 reduced-fine features "
                               "in FINE_NAMES order",
                "fine_names": list(hm.FINE_NAMES),
                "why_recorded": "a reader who assumes norm_mu[0] pairs with "
                                "weight[0] misaligns EVERY coefficient by one, "
                                "silently, with plausible-looking output",
            },
            "first_slug_t0": min(int(r["slug"].rsplit("-", 1)[1]) for r in kept),
            "last_slug_t0": max(int(r["slug"].rsplit("-", 1)[1]) for r in kept),
        }
        print(f"  {coin}: fitted {len(kept)} rows / "
              f"{out[coin]['n_actions_fitted']} actions over {days}, "
              f"{sum(y)} positive, {len(w)} hazard weights")
        del XA, Xs, PM, FN
    return out


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    assert_consumed_fragment_only([1787579400, LAST_CONSUMED_SLUG_T0])
    ok(True, "KNOWN-GOOD: the consumed fragment's own slugs pass, boundary "
             "inclusive")
    try:
        assert_consumed_fragment_only([1787579400, LAST_CONSUMED_SLUG_T0 + 300])
        ok(False, "a top-up slug must be REFUSED")
    except TopUpLeak as e:
        ok("consume it" in str(e),
           "POSITIVE CONTROL: a single post-boundary slug is REFUSED, and the "
           "message names WHY it matters (consuming Phase 2's test surface), "
           "not merely that a bound was crossed")
    try:
        assert_consumed_fragment_only([1787702100])
        ok(False, "the declared top-up's last slug must be refused")
    except TopUpLeak:
        ok(True, "the declared top-up range is refused wholesale")
    ok(OUT.name.endswith("_v1.json"),
       "the candidate artifact is versioned, so a correction supersedes as v2 "
       "rather than editing a frozen file (rule 13)")
    print(f"harmful_freeze_candidate selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if "--user-approved-freeze" not in sys.argv:
        print("REFUSED: the freeze is the USER's decision (R-162(3)).\n"
              "This builder runs only after an explicit yes, via:\n"
              "  python3 harmful_freeze_candidate.py --user-approved-freeze")
        return 2
    selftest()
    fits = refit(user_approved=True)
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True,
                                  text=True).stdout.strip()
    ask = json.loads((DERIVED /
                      "harmful_reduced_fine_FREEZE_ASK_v1.json").read_text())
    man = DERIVED / "harmful_candidate_manifest_v1.json"
    cand = {
        "protocol": "HARMFUL_REDUCED_FINE_CANDIDATE_V1",
        "status": "FROZEN",
        "spec": "PM_PLUS_FINE (reduced fine)",
        "frozen_at_utc": sh("date -u +%Y-%m-%dT%H:%M:%SZ"),
        "user_approval": "explicit yes in BE's pane, 2026-08-26; recorded "
                         "coordinator-side as R-168",
        "authorising_ask": "harmful_reduced_fine_FREEZE_ASK_v1.json",
        "target_latency_ms": hm_target(),
        "fits": fits,
        "trained_on": {
            "population": "v3_4_consumed_fragment",
            "days": ask["population"]["days"],
            "last_consumed_slug_t0": LAST_CONSUMED_SLUG_T0,
            "topup_excluded": True,
            "guard": "assert_consumed_fragment_only",
        },
        "declared_nulls": ask["declared_nulls"],
        "race_multiplicity_at_freeze": 2,
        "race_members": ["PM_PLUS_FINE (PRIMARY, this artifact)",
                         "PM_FINE_EXTENDED (HELD)"],
        "consumed_specs_disclosed": 5,
        "era": ask["era"],
        "manifest": man.name,
        "manifest_sha256": hashlib.sha256(man.read_bytes()).hexdigest(),
        "builder": "live/pm_research/harmful_hazard_model.py",
        "builder_sha256": hashlib.sha256(
            (REPO / "live/pm_research/harmful_hazard_model.py").read_bytes()
        ).hexdigest(),
        "freeze_builder_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "git_commit_at_refit": sh("git rev-parse HEAD"),
        "NO_IN_SAMPLE_VERDICT":
            "The refit is in-sample by construction. No score, AUC or gate "
            "number from it is stored or claimed. The evidence for this "
            "candidate is the completed paired comparison; the freeze does "
            "not re-argue it.",
        "decision_eligible": False,
        "forward_validation":
            "Begins at the freeze COMMIT instant. Requires >=5 complete "
            "untouched UTC days (R-109 G>=5). The consumed fragment and the "
            "R-145(3) top-up can never serve as forward validation.",
    }
    atomic_write_json(OUT, cand)
    print(f"\nWROTE {OUT.name}  status FROZEN  multiplicity 2")
    return 0


def hm_target() -> int:
    import harmful_hazard_model as hm
    return hm.TARGET_LATENCY_MS


if __name__ == "__main__":
    raise SystemExit(main())
