#!/usr/bin/env python3
"""BE fragment diagnostic — R-293 scope, R-295 authorisation.

DIAGNOSTIC_NEVER_EVIDENCE. Scores the FROZEN btc candidate (LGBM_PINNED,
hash-verified against freeze receipt v3 DIRECTLY) on the post-freeze fragments
DA admitted, against the incumbent counterpart and a matched-random control on
the IDENTICAL rows.

WHAT THIS IS NOT. It is not a validation, not evidence, and cannot move the
race. DA's receipt states three UNCONDITIONAL inadmissibility reasons: neither
fragment is a complete UTC day; fragment A begins mid-day at the freeze epoch
and is therefore a selected slice; and both carry quantified burst-concentrated
feed loss. Those hold whatever this reads. The pre-registered readings (R-293)
are: POSITIVE is weak comfort only, NEGATIVE is ambiguous, and NEITHER may
change the candidate, the race, or multiplicity.

WHY A SEPARATE HARNESS. harmful_forward_scorer.py exists but has no run path
(its main() prints usage and returns 0) AND its CANDIDATE constant points at
harmful_reduced_fine_candidate_v1.json -- arm A's frozen LINEAR, not
LGBM_PINNED. Pointed at this diagnostic it would score the WRONG MODEL and
return clean-looking numbers. That trap is filed race-critical and is
deliberately NOT repaired here.

ZERO IDENTITY FILES. Every upstream function is IMPORTED, never edited:
importing binds no identity, only editing moves the hash. The identity is
measured before and after and asserted equal.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

import phase2_arms as PA                              # noqa: E402
import harmful_exposure_rows as HER                   # noqa: E402
import harmful_action_eval as HAE                     # noqa: E402

DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
DA_RECEIPT = DERIVED / "da_fragment_censoring_v1.json"
FREEZE = DERIVED / "harmful_phase2_lgbm_btc_freeze_v3.json"
OUT = DERIVED / "be_fragment_diagnostic_v1.json"
ROWS_OUT = DERIVED / "be_fragment_exposure_rows_v1.json"
COIN = "btc"
IDENTITY_AT_BUILD = "3d0b6c8c6dfe9466"   # measured before this file existed


class DiagnosticRefused(RuntimeError):
    """Refuse at the producer, before a number exists to be believed."""


# ---------------------------------------------------------------------------
# VACUUM PAIR (rule-18 candidate, R-290)
# ---------------------------------------------------------------------------
# A filter that reads a field NOTHING CARRIES selects nothing and reports zero.
# Zero is a legitimate answer, so the failure is invisible: "no rows matched"
# and "the field I filtered on does not exist" print identically. Every filter
# here is therefore paired with an assertion that the field EXISTS and READS AS
# ITS TYPE on a non-empty sample, and with an asserted population count. A
# renamed field must REFUSE, never quietly find nothing.
def assert_field_readable(rows, field: str, typ, why: str, sample: int = 200):
    if not rows:
        raise DiagnosticRefused(
            f"REFUSED: cannot verify {field!r} on an EMPTY population. An empty "
            f"input makes every field assertion vacuously true, which is the "
            f"hole this check exists to close ({why}).")
    look = rows[:sample]
    missing = sum(1 for r in look if field not in r)
    if missing:
        raise DiagnosticRefused(
            f"REFUSED: {missing}/{len(look)} sampled rows carry no {field!r}. "
            f"Filtering on a field the rows do not have selects NOTHING and "
            f"reports zero, which is indistinguishable from a real zero. "
            f"({why})")
    wrong = [type(r[field]).__name__ for r in look
             if not isinstance(r[field], typ) or isinstance(r[field], bool)
             and typ is not bool]
    if wrong:
        raise DiagnosticRefused(
            f"REFUSED: {field!r} reads as {sorted(set(wrong))} on "
            f"{len(wrong)}/{len(look)} sampled rows, not {typ.__name__}. A "
            f"comparison against the wrong type is False everywhere and empties "
            f"the population silently. ({why})")
    return {"field": field, "type": typ.__name__, "sampled": len(look)}


def assert_population(got: int, expected: int, what: str):
    if got != expected:
        raise DiagnosticRefused(
            f"REFUSED: {what} produced {got}, expected {expected}. A population "
            f"that silently differs from the one declared makes every number "
            f"downstream describe a different question.")
    if got == 0:
        raise DiagnosticRefused(
            f"REFUSED: {what} is EMPTY. Zero is a real answer only when the "
            f"instrument could have produced a non-zero one.")
    return {"what": what, "n": got}


# ---------------------------------------------------------------------------
# (a) the fragment driver — imports only
# ---------------------------------------------------------------------------
def da_bounds() -> dict:
    """DA's receipt is the authority on WHICH rows exist. Read, never guessed."""
    if not DA_RECEIPT.exists():
        raise DiagnosticRefused(
            f"REFUSED: {DA_RECEIPT.name} absent. The score read is gated on DA's "
            f"admissibility/censoring receipt (R-293); scoring without it would "
            f"be scoring an unbounded population.")
    d = json.loads(DA_RECEIPT.read_text())
    if d.get("status") != "DIAGNOSTIC_NEVER_EVIDENCE":
        raise DiagnosticRefused(
            f"REFUSED: DA receipt status is {d.get('status')!r}. This harness "
            f"produces a diagnostic and must not consume a receipt that claims "
            f"more.")
    if not d.get("INADMISSIBLE_FOR_THE_RACE"):
        raise DiagnosticRefused(
            "REFUSED: DA's receipt does not mark the fragments inadmissible. "
            "This harness exists only for an inadmissible read; if that changed, "
            "the change is a ruling, not a default.")
    if d.get("coin") != COIN:
        raise DiagnosticRefused(f"REFUSED: DA receipt is for {d.get('coin')!r}.")
    return d


def select_fragment_windows(bounds: list, coin: str = COIN) -> list:
    """Windows whose slug t0 lies in [lo, hi). `days` is a PARAMETER of
    select_stratified; build_rows simply never passes it, which is why this
    driver exists rather than an edit to a fit-identity module."""
    import datetime as dt
    days = sorted({dt.datetime.fromtimestamp(t, dt.timezone.utc)
                   .strftime("%Y-%m-%d")
                   for lo, hi in bounds for t in (lo, hi - 1)})
    sel = HER.select_stratified(10 ** 6, days=tuple(days), coins=(coin,))
    keep = []
    for ent in sel:
        try:
            t0 = int(ent[0].rsplit("-", 1)[1])
        except ValueError:
            continue
        if any(lo <= t0 < hi for lo, hi in bounds):
            keep.append(ent)
    keep.sort(key=lambda e: (int(e[0].rsplit("-", 1)[1]), e[0]))
    return keep


def build_fragment_rows(windows: list) -> dict:
    """build_rows' per-window loop, replicated by IMPORT.

    The seven upstream functions are called exactly as build_rows calls them,
    including the STRICT reconciliation condition -- a window failing any of
    join/boundary/clock/generation/unhooked has EVERY row marked
    RECONCILIATION_FAILED rather than dropped, so exclusions stay counted
    statuses (rule 4) instead of silent absences."""
    import datetime as _dt
    spec = HER.qr._qr_spec(HER.qr.QR_SKEW, latency_ms=0, cancel=False)
    rows, stats = [], {
        "windows_in": len(windows), "windows_replayed": 0,
        "windows_reconciliation_failed": 0, "wrong_generation_assignments": 0,
        "boundary_time_violations": 0, "consume_clock_violations": 0,
        "unhooked_state_changes": 0, "windows_replay_returned_none": 0}
    for ent in windows:
        slug = ent[0]
        out = HER.replay_with_recorder(ent[1], ent[2], ent[3], ent[4], spec)
        if out is None:
            stats["windows_replay_returned_none"] += 1
            continue
        arm, wf = out
        stats["windows_replayed"] += 1
        t0 = int(slug.rsplit("-", 1)[1])
        day = _dt.datetime.fromtimestamp(t0, _dt.timezone.utc).strftime("%Y-%m-%d")
        joined, jrec = HER.join_fills(arm.fill_log, arm.fills)
        n_boundary_bad = HER.verify_boundary_times(arm.segments, joined)
        ttimes = HER.trade_receipt_times(ent[1], ent[2], ent[3])
        n_clock_bad = HER.verify_consume_clock(arm.consume_times, ttimes)
        gens, recon = HER.generation_table(arm.segments, joined, wf,
                                           HER.qr.base.fi.WINDOW_S)
        wrows = HER.label_rows(arm.segments, gens, wf, HER.qr.base.fi.WINDOW_S)
        bad = (jrec["count_mismatch"] or jrec["tuple_mismatches"]
               or recon["orphan_fills"]
               or recon["wrong_generation_assignments"]
               or arm.unhooked_changes or n_boundary_bad or n_clock_bad)
        stats["wrong_generation_assignments"] += recon["wrong_generation_assignments"]
        stats["boundary_time_violations"] += n_boundary_bad
        stats["consume_clock_violations"] += n_clock_bad
        stats["unhooked_state_changes"] += arm.unhooked_changes
        if bad:
            stats["windows_reconciliation_failed"] += 1
            for r in wrows:
                r["status"] = "RECONCILIATION_FAILED"
        for r in wrows:
            r["slug"] = slug; r["coin"] = slug.split("-")[0]
            r["day"] = day; r["t0"] = t0
        rows.extend(wrows)
    stats["rows_total"] = len(rows)
    return {"rows": rows, "stats": stats}


def canonical(o) -> str:
    return hashlib.sha256(json.dumps(o, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _sha16(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# (b) the score path — the FROZEN candidate, hash-verified against freeze v2
# ---------------------------------------------------------------------------
def load_frozen_candidate() -> dict:
    """LGBM_PINNED, verified against FREEZE RECEIPT v2 DIRECTLY.

    R-293's mechanics, stated not laundered: the R-225 fit7-manifest binding
    refusal STANDS for the race scorer. This diagnostic verifies the CANDIDATE
    against the freeze receipt, and the gate-binding step is superseded FOR THIS
    DIAGNOSTIC ONLY by the R-277 determination (drift noticed, determined
    identical). The mechanical chain closure still waits for the fit-time
    re-stamp; nothing here closes it."""
    fz = json.loads(FREEZE.read_text())
    if fz.get("candidate", {}).get("arm") != "LGBM_PINNED":
        raise DiagnosticRefused(
            f"REFUSED: freeze v2's candidate is "
            f"{fz.get('candidate', {}).get('arm')!r}, not LGBM_PINNED.")
    want = fz["provenance"]["model_artifacts_sha256_prefix"]
    checked = {}
    for name, w in sorted(want.items()):
        p = PA.FITDIR / name
        if not p.exists():
            raise DiagnosticRefused(
                f"REFUSED: freeze v2 names {name}, which is absent.")
        got = _sha16(p)
        if got != w:
            raise DiagnosticRefused(
                f"REFUSED: {name} is not the FROZEN artifact "
                f"(freeze={w} live={got}). A different model produces a "
                f"different diagnostic and the numbers cannot show which ran.")
        checked[name] = got
    # THE SCALER IS NOT PINNED BY THE FREEZE RECEIPT. LGBM_PINNED is applied on
    # the z-scaled full design, and the scaler lives in linear_<coin>.json --
    # which freeze v2 does NOT hash. It is verified against the FIT MANIFEST
    # instead, and the gap is reported rather than papered over: the frozen
    # candidate cannot be applied without an input its own freeze receipt does
    # not bind.
    # v3 CLOSED THIS GAP. v2 pinned the three model artifacts but not the
    # SCALER the model is applied through, so the frozen candidate could not be
    # reproduced from its own receipt. v3 (R-297) adds the binding, and it is
    # now the PRIMARY check; the fit manifest remains an independent SECOND
    # binding and both must agree.
    scaler = PA.FITDIR / f"linear_{COIN}.json"
    fw = (fz["provenance"].get("scaler_sha256_prefix") or {}).get(scaler.name)
    man = json.loads((PA.FITDIR / PA.FIT_MANIFEST).read_text())
    mw = (man.get("file_hashes") or {}).get(scaler.name)
    if fw is None:
        raise DiagnosticRefused(
            f"REFUSED: this freeze receipt does not bind {scaler.name}. "
            f"LGBM_PINNED cannot be applied without it, and a receipt that "
            f"cannot reproduce an application is incomplete (R-297).")
    sg = _sha16(scaler)
    for src, w in (("freeze receipt", fw), ("fit manifest", mw)):
        if w is None:
            raise DiagnosticRefused(f"REFUSED: {src} does not bind {scaler.name}.")
        if sg != w:
            raise DiagnosticRefused(
                f"REFUSED: {scaler.name} ({src}={w} live={sg}). A different "
                f"scaler produces different scores from identical model "
                f"artifacts.")
    lin = json.loads(scaler.read_text())
    thr = json.loads((PA.FITDIR / f"lgbm_thresholds_{COIN}.json").read_text())
    return {"arm": "LGBM_PINNED", "coin": COIN,
            "norm_mu": lin["norm_mu"], "norm_sd": lin["norm_sd"],
            "haz": str(PA.FITDIR / f"lgbm_haz_{COIN}.txt"),
            "val": str(PA.FITDIR / f"lgbm_val_{COIN}.txt"),
            "causal_thresholds": thr,
            "verified": {
                "against": "harmful_phase2_lgbm_btc_freeze_v3 (DIRECT)",
                "model_artifacts": checked,
                "scaler": {"file": scaler.name, "sha256_prefix": sg,
                           "bound_by": ["freeze_receipt_v3 (primary)",
                                        "fit_manifest.file_hashes (second, independent"
                                        " — both must agree)"],
                           "NOT_bound_by_freeze_receipt": False,
                           "gap_closed_by": "R-297 / freeze receipt v3. This "
                           "diagnostic surfaced the gap (v2 pinned the model "
                           "artifacts but not the scaler they are applied "
                           "through); it is now closed AT ITS SOURCE rather "
                           "than annotated here."},
                "gate_binding": "SUPERSEDED FOR THIS DIAGNOSTIC ONLY by the "
                                "R-277 determination (drift noticed, determined "
                                "IDENTICAL). The R-225 refusal stands for the "
                                "race scorer and the mechanical chain closure "
                                "still awaits the fit-time re-stamp."}}


def score_rows(model: dict, block: dict, idx) -> list:
    """ecv per row. Arithmetic COPIED from the four-arm LGBM_PINNED branch:
    full PM+FN+ST design, z-scaled with the fit's own scaler, ecv = p * v."""
    import lightgbm as lgb
    import numpy as np
    mu, sd = model["norm_mu"], model["norm_sd"]
    hb = lgb.Booster(model_file=model["haz"])
    vb = lgb.Booster(model_file=model["val"])
    out = []
    CH = 50_000
    idx = list(idx)
    for lo in range(0, len(idx), CH):
        chunk = idx[lo:lo + CH]
        S = np.empty((len(chunk), len(mu) + 1), dtype=np.float64)
        S[:, 0] = 1.0
        for k, j in enumerate(chunk):
            raw = block["PM"][j] + block["FN"][j] + block["ST"][j]
            if len(raw) != len(mu):
                raise DiagnosticRefused(
                    f"REFUSED: row {j} has {len(raw)} features, the frozen "
                    f"candidate was fitted on {len(mu)}.")
            S[k, 1:] = [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
        out.extend((hb.predict(S) * vb.predict(S)).tolist())
        del S
    return out


# ---------------------------------------------------------------------------
def build_rows_stage() -> dict:
    """STAGE 1: fragment exposure rows. Needs no edit to any module."""
    da = da_bounds()
    bounds = [tuple(f["bounds_epoch"]) for f in da["fragments"]]
    expected = sum(f["contained_windows"] for f in da["fragments"])
    print(f"[frag] DA bounds {bounds} expecting {expected} windows", flush=True)
    wins = select_fragment_windows(bounds)
    assert_population(len(wins), expected,
                      "fragment window selection vs DA's contained_windows")
    print(f"[frag] {len(wins)} windows selected, building rows", flush=True)
    built = build_fragment_rows(wins)
    rows, stats = built["rows"], built["stats"]

    # VACUUM PAIR. Everything downstream filters on these; a rename makes the
    # filter select nothing and report a clean zero.
    checks = [assert_field_readable(rows, "status", str, "OK-row filtering"),
              assert_field_readable(rows, "coin", str, "per-coin split"),
              assert_field_readable(rows, "slug", str, "window identity"),
              assert_field_readable(rows, "t0", int, "window epoch"),
              assert_field_readable(rows, "day", str, "UTC day grouping"),
              assert_field_readable(rows, "side", str, "state-tape join key"),
              assert_field_readable(rows, "gen", int, "action identity"),
              assert_field_readable(rows, "t_start", float, "state-tape join key")]
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    if n_ok == 0:
        raise DiagnosticRefused(
            "REFUSED: zero OK rows. Every downstream stage filters on "
            "status=='OK'; a zero here would propagate as an empty population "
            "and read as a null result rather than an absent one.")
    by_status: dict = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    payload = {"schema": "be_fragment_exposure_v1",
               "produced_by": "be_fragment_diagnostic.build_rows_stage",
               "status": "DIAGNOSTIC_NEVER_EVIDENCE",
               "da_receipt": {"file": DA_RECEIPT.name,
                              "sha256_prefix": _sha16(DA_RECEIPT),
                              "declared_cutoff_epoch": da["declared_cutoff_epoch"],
                              "freeze_epoch": da["freeze_epoch"]},
               "bounds_epoch": [list(b) for b in bounds],
               "windows_expected_from_da": expected,
               "field_assertions": checks,
               "rows_by_status": dict(sorted(by_status.items())),
               "stats": stats,
               "days": sorted({r["day"] for r in rows}),
               "rows": rows}
    if ROWS_OUT.exists():
        snap = ROWS_OUT.with_suffix(".json.prev")
        snap.write_bytes(ROWS_OUT.read_bytes())
        print(f"[frag] snapshotted {ROWS_OUT.name} -> {snap.name}", flush=True)
    ROWS_OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"[frag] wrote {ROWS_OUT} ({ROWS_OUT.stat().st_size/1e6:.1f} MB)",
          flush=True)
    return {k: v for k, v in payload.items() if k != "rows"}


def _equivalence_fixture(d: Path) -> Path:
    """A tiny VALID tape carrying both an OK row and a non-OK one, so the
    comparison exercises the status branch rather than only the happy path."""
    import phase2_state_schema_freeze as _PIN
    feats = _PIN.build_pin()["features_in_order"]
    rows = []
    for k in range(4):
        rows.append({
            "slug": f"btc-updown-5m-{1787897400 + 300 * k}",
            "side": "BUY_UP" if k % 2 else "SELL_UP",
            "gen": k + 1, "t_start": float(k) - 1.0,
            "t0": float(1787897400 + 300 * k), "split": "train",
            "state_status": "OK" if k < 3 else "GAP_AT_CUTOFF",
            "state": {f: float(k) + i * 0.5 for i, f in enumerate(feats)}})
    rows.append(dict(rows[0], split="score", gen=99))   # a DIFFERENT split
    fx = d / "fixture_tape.json"
    fx.write_text(json.dumps({"protocol": "PHASE2_STATE_TAPE_V5", "rows": rows}))
    return fx


def selftest() -> int:
    fails: list = []

    def ok(c, label):
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    good = [{"status": "OK", "coin": "btc", "slug": "s", "t0": 1,
             "day": "2026-08-28", "side": "BUY_UP", "gen": 1, "t_start": 0.5}
            for _ in range(10)]

    # ---- VACUUM PAIR (rule-18 bar, R-290) --------------------------------
    ok(assert_field_readable(good, "status", str, "x")["sampled"] == 10,
       "POSITIVE CONTROL: a well-formed population passes the field assertion "
       "(an assertion that refused everything would pass every known-bad)")

    renamed = [dict(r) for r in good]
    for r in renamed:
        r["state"] = r.pop("status")          # the field, DELIBERATELY RENAMED
    naive = [r for r in renamed if r.get("status") == "OK"]
    try:
        assert_field_readable(renamed, "status", str, "OK-row filtering")
        got = ""
    except DiagnosticRefused as e:
        got = str(e)
    ok(len(naive) == 0 and "carry no 'status'" in got,
       f"KNOWN-BAD, THE VACUUM PAIR: a RENAMED field makes the naive filter "
       f"select {len(naive)} rows and report a clean zero; the assertion "
       f"REFUSES instead of finding nothing")

    wrongtype = [dict(r, t0="1") for r in good]
    try:
        assert_field_readable(wrongtype, "t0", int, "window epoch"); g2 = ""
    except DiagnosticRefused as e:
        g2 = str(e)
    ok("reads as ['str']" in g2,
       "KNOWN-BAD: a field of the WRONG TYPE is refused — a comparison against "
       "the wrong type is False everywhere and empties the population silently")

    try:
        assert_field_readable([], "status", str, "x"); g3 = ""
    except DiagnosticRefused as e:
        g3 = str(e)
    ok("EMPTY population" in g3,
       "KNOWN-BAD: an EMPTY population is refused — it makes every field "
       "assertion vacuously true, which is the hole being closed")

    for got_n, exp_n, want in ((5, 7, "expected 7"), (0, 0, "EMPTY")):
        try:
            assert_population(got_n, exp_n, "probe"); g4 = ""
        except DiagnosticRefused as e:
            g4 = str(e)
        ok(want in g4, f"KNOWN-BAD: population {got_n} vs {exp_n} is refused")
    ok(assert_population(7, 7, "probe")["n"] == 7,
       "POSITIVE CONTROL: a matching, non-zero population passes")

    # ---- the frozen candidate --------------------------------------------
    try:
        m = load_frozen_candidate()
        ok(m["arm"] == "LGBM_PINNED" and len(m["verified"]["model_artifacts"]) == 3,
           "POSITIVE CONTROL: the FROZEN candidate verifies against freeze "
           "receipt v2 DIRECTLY (all three artifacts)")
        ok(m["verified"]["scaler"]["NOT_bound_by_freeze_receipt"] is False,
           "R-297: the scaler is now bound by the FREEZE RECEIPT itself (v3), "
           "verified against it AND the fit manifest independently — the gap "
           "this diagnostic surfaced is closed at its source")
    except DiagnosticRefused as e:
        ok(False, f"frozen candidate failed to load: {e}")

    # ---- DA's receipt gates the read -------------------------------------
    try:
        d = da_bounds()
        ok(d["declared_cutoff_epoch"] == 1787973300,
           "the cutoff is DA's committed instant, read from the receipt, never "
           "a local default")
        ok(sum(f["contained_windows"] for f in d["fragments"]) == 253,
           "DA's contained-window total is 253, which my independent selection "
           "must reproduce exactly")
    except DiagnosticRefused as e:
        ok(False, f"DA receipt gate failed: {e}")

    # ---- R-301(1): the FORK AGREES WITH PA, measured every run -----------
    import tempfile as _tf
    _d = Path(_tf.mkdtemp())
    _fx = _equivalence_fixture(_d)
    _mine = _index_tape(_fx, split="train")
    _orig = PA.TAPE_PATH
    try:
        # TEST SCOPE ONLY, permitted by R-301 for exactly this comparison and
        # visible right here. Production never patches it -- that is the whole
        # reason the fork exists instead of a monkeypatch.
        PA.TAPE_PATH = _fx
        _theirs = PA.tape_index("train")
    finally:
        PA.TAPE_PATH = _orig
    ok(_mine == _theirs,
       f"R-301 EQUIVALENCE: the eight-line fork and PA.tape_index produce "
       f"IDENTICAL output on the fixture ({len(_mine)} vs {len(_theirs)} keys)")
    ok(len(_mine) == 4 and all(k[0].startswith("btc") for k in _mine),
       "the fixture exercises a real index: 4 train rows, the score-split row "
       "correctly excluded")
    ok(any(v["vec"] is None and v["status"] != "OK" for v in _mine.values())
       and any(v["vec"] is not None for v in _mine.values()),
       "the comparison covers BOTH branches — an OK row carrying a vector and a "
       "non-OK row carrying None with its status")
    ok(PA.TAPE_PATH == _orig,
       "the test-scope patch is REVERTED; a leaked TAPE_PATH would point "
       "production at a fixture")
    try:
        _index_tape(_fx, split="nonesuch"); _g5 = ""
    except DiagnosticRefused as e:
        _g5 = str(e)
    ok("ZERO entries" in _g5,
       "KNOWN-BAD: an index that comes back EMPTY is REFUSED — every row would "
       "then drop as state_join_failed and the read would be clean, empty and "
       "wrong (the vacuum shape at the index level)")
    import shutil as _sh; _sh.rmtree(_d, ignore_errors=True)

    ok(PA.measured_code_identity()["combined"] == IDENTITY_AT_BUILD,
       f"ZERO IDENTITY FILES TOUCHED: the lattice identity is still "
       f"{IDENTITY_AT_BUILD} (checkable, not asserted)")

    print(f"\n{'BE FRAGMENT DIAGNOSTIC SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    if "--build-rows" in argv:
        print(json.dumps(build_rows_stage(), indent=1, sort_keys=True)[:1600])
        return 0
    print("usage: be_fragment_diagnostic.py --selftest | --build-rows")
    return 0




# ---------------------------------------------------------------------------
# STAGE 3: score the frozen candidate on the fragment rows
# ---------------------------------------------------------------------------
def score_stage(tape_path: Path, latency_ms: int = None) -> dict:
    """The diagnostic read. Bound to DA's receipt and cutoff throughout.

    Both comparators run on the IDENTICAL rows (R-293): the incumbent
    counterpart is arm D applied to the same population, and matched-random is
    evaluate_policy's own (side x hour)-matched control on the same rows. The
    candidate's thresholds are its FROZEN causal thresholds, never re-derived
    here -- re-deriving them on the scored data would make the read
    retrospective and it would still look normal."""
    import phase2_iter011_run as R11        # reuse, never re-implement
    da = da_bounds()
    model = load_frozen_candidate()
    L = int(getattr(__import__("phase2_declaration"), "TARGET_LATENCY_MS")
            if latency_ms is None else latency_ms)

    TAPE = PA.tape_index("score") if tape_path is None else _index_tape(tape_path)
    block_all = PA._feature_pass(ROWS_OUT, "be_fragment", TAPE=TAPE)
    if COIN not in block_all:
        raise DiagnosticRefused(
            f"REFUSED: the feature pass produced no {COIN} block. Every row was "
            f"excluded, which reads as a null result and is an absent one.")
    blk = block_all[COIN]
    kept = blk["kept"]
    assert_field_readable(kept, "status", str, "post-feature-pass rows")
    if not kept:
        raise DiagnosticRefused(
            f"REFUSED: zero rows survived the feature pass. drops={blk['drops']}. "
            f"A state tape that does not cover these rows drops every one of "
            f"them as state_join_failed and yields a clean, empty, wrong answer.")
    idx = range(len(kept))

    cand = score_rows(model, blk, idx)
    inc_model = R11.load_verified_incumbent(COIN)
    inc = R11.apply_incumbent(inc_model, blk, idx)["expected_cancel_value"]
    if not (len(cand) == len(inc) == len(kept)):
        raise DiagnosticRefused(
            f"REFUSED: {len(cand)} candidate / {len(inc)} incumbent scores "
            f"against {len(kept)} rows. The comparison is only defined on the "
            f"IDENTICAL population.")

    thetas = model["causal_thresholds"]
    budgets = [float(b.rstrip('%')) / 100.0 for b in sorted(thetas)]
    ev_c = HAE.evaluate_policy(kept, cand, latency_ms=L, budgets=budgets,
                               theta_frozen=thetas)
    ev_i = HAE.evaluate_policy(kept, inc, latency_ms=L, budgets=budgets)
    cells = {}
    for b in sorted(ev_c):
        c, i = ev_c[b], ev_i.get(b, {})
        cells[b] = {
            "budget": b,
            "candidate_net_cents": c.get("net_cents"),
            "incumbent_net_cents": i.get("net_cents"),
            "increment_vs_incumbent_cents": (
                None if i.get("net_cents") is None or c.get("net_cents") is None
                else c["net_cents"] - i["net_cents"]),
            "beats_matched_random_on_NET": c.get("beats_random_max_on_NET"),
            "random_net_max": c.get("random_net_max"),
            "random_net_p95": c.get("random_net_p95"),
            "n_cancelled_generations": c.get("n_cancelled_generations"),
            "harm_avoided_cents": c.get("harm_avoided_cents"),
            "sacrifice_cents": c.get("sacrifice_cents"),
            "rho_captured_over_sacrificed": c.get("rho_captured_over_sacrificed"),
            "concentration": c.get("concentration"),
            "threshold_mode": c.get("threshold_mode"),
        }
    return {
        "artifact": "be_fragment_diagnostic_v1",
        "status": "DIAGNOSTIC_NEVER_EVIDENCE",
        "what_this_cannot_do": (
            "This cannot admit, re-freeze, re-parameterise or re-schedule "
            "anything. R-293 pre-registered the readings BEFORE any number "
            "existed: a POSITIVE result is WEAK COMFORT ONLY because the "
            "censoring plausibly flatters; a NEGATIVE result is AMBIGUOUS "
            "because censoring artifacts are indistinguishable from real "
            "failure at this coverage, and it specifically must NOT trigger a "
            "candidate change, which would be selection on a contaminated "
            "read. Under EVERY outcome the race admission rule, the frozen "
            "candidate and multiplicity (1) are untouched."),
        "inadmissibility_is_unconditional": da["inadmissibility_reasons"],
        "censoring_statement": da["censoring_statement"],
        "censoring_measured": da.get("censoring_measured_not_asserted"),
        "da_receipt": {"file": DA_RECEIPT.name,
                       "sha256_prefix": _sha16(DA_RECEIPT),
                       "declared_cutoff_epoch": da["declared_cutoff_epoch"],
                       "declared_cutoff_utc": da["declared_cutoff_utc"]},
        "candidate": model["verified"],
        "incumbent": inc_model.get("_verified"),
        "population": {"rows_scored": len(kept),
                       "n_actions": len({(r.get("slug"), r.get("side"),
                                          r.get("gen")) for r in kept}),
                       "drops_are_counted_statuses": blk["drops"],
                       "days": sorted({r.get("day") for r in kept})},
        "latency_ms": L,
        "cells": cells,
        "one_run": "R-293 permits ONE run; a re-run requires its own written "
                   "reason recorded before it.",
    }


def _index_tape(path: Path, split: str = "train") -> dict:
    """Index a tape at an EXPLICIT path.

    A NAMED, MINIMAL FORK, declared rather than hidden. PA.tape_index reads the
    module constant TAPE_PATH and takes no path argument, and phase2_arms IS an
    identity file -- adding a parameter there would move fit_code_sha256_prefix.
    So the eight-line loop below is copied from it VERBATIM.

    Everything that could be reused IS reused: the row stream is PA's own
    path-parameterised _stream_tape_rows, the feature order and the encoder are
    the pinned ones. Only the loop is duplicated, and the duplication is stated
    here rather than discovered later. Monkeypatching PA.TAPE_PATH was the
    alternative and was refused for the same reason it was refused for the tape
    builder: a reader of phase2_arms could not see what ran.

    RETIREMENT IS DECLARED, NOT HOPED (R-301 condition 2): when the post-release
    cycle moves fit identity anyway, tape_index gains its path parameter and
    this function is DELETED. It is temporary by declaration, with a named
    trigger, so it cannot quietly become the second permanent implementation --
    which is the whole objection to a fork.

    AGREEMENT IS MEASURED ON EVERY SUITE RUN (R-301 condition 1): the selftest
    runs BOTH indexers over one fixture and asserts identical output, so the day
    the eight lines drift from PA's, the suite says so."""
    import phase2_state_schema_freeze as _PIN
    feats = _PIN.build_pin()["features_in_order"]
    idx = {}
    for r in PA._stream_tape_rows(Path(path)):
        if r.get("split") != split:
            continue
        _st = str(r.get("state_status", "OK"))
        state = r.get("state") or {}
        idx[(r["slug"], r["side"], r["gen"], r["t_start"])] = {
            "vec": tuple(_PIN.encode_row(state, feats)) if _st == "OK" else None,
            "status": _st,
            "t0": float(r["t0"]), "t_start": float(r["t_start"])}
    if not idx:
        raise DiagnosticRefused(
            f"REFUSED: indexing {Path(path).name} for split {split!r} produced "
            f"ZERO entries. Every downstream row would then drop as "
            f"state_join_failed and the diagnostic would report a clean, empty, "
            f"wrong answer.")
    return idx


if __name__ == "__main__":
    raise SystemExit(main())
