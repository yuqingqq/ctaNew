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
import inspect
import json
import math
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

import phase2_arms as PA                              # noqa: E402
import phase2_declaration as D                        # noqa: E402
import harmful_exposure_rows as HER                   # noqa: E402
import harmful_action_eval as HAE                     # noqa: E402

DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
DA_RECEIPT = DERIVED / "da_fragment_censoring_v1.json"
FREEZE = DERIVED / "harmful_phase2_lgbm_btc_freeze_v3.json"
OUT = DERIVED / "be_fragment_diagnostic_v1.json"
ROWS_OUT = DERIVED / "be_fragment_exposure_rows_v1.json"
COIN = "btc"
# The lattice identity this harness was built against. It MOVED ONCE, under an
# explicit ruling (FD-R7): phase2_arms._stream_tape_rows was a fail-open reader
# that accepted a truncated tape as complete, and repairing it necessarily moved
# fit_code_sha256_prefix. Both values are kept and the rebind artifact is named,
# because silently re-pinning this constant is exactly how an identity guard
# stops guarding -- the next unexplained move must still go RED.
# R-310: THE POLICY-FIXED PREDICATE EXCLUSIONS for this diagnostic binding.
#
# REGISTER POLICY, hard-coded here — never read from the tape's self-declaration
# and never widened for a consumer's convenience. Two shapes were refused in
# pure form when this was ruled: naming predicates alone risks a SHRINKABLE
# ALLOWLIST (a consumer quietly excluding whatever it needs), and consuming a
# failing verdict with the failures merely NAMED would let a conformance failure
# through. So: every applicable predicate must PASS except exactly these two,
# each carrying the ruling that admits it, and ANY OTHER failing predicate
# REFUSES.
#
# SCOPE: this DIAGNOSTIC_NEVER_EVIDENCE binding only. A result-bearing binding
# requires all_pass, full stop.
# FD4: the MINIMUM predicate universe this consumer requires, versioned.
# Additional predicates a newer gate introduces are ALLOWED (they must pass, per
# R-310). An OMITTED governed predicate REFUSES -- otherwise a gate that simply
# stopped emitting a check would look like a gate whose check passed, and the
# consumer would never know the difference.
GATE_UNIVERSE_VERSION = "fragment_v2"
# BIND TO THE SET'S IDENTITY, not to my transcription of its names and not to
# the gate FILE's hash. DA publishes load_bearing_identity(): a sorted-set
# sha256 over the load-bearing names, so it is order-independent by
# construction. Why this and not the alternatives, from DA's own reasoning:
#   * a transcribed name list goes STALE SILENTLY -- the load-bearing set moved
#     TWICE TODAY (both_splits_populated under R-306, per_row_conformance_exact
#     under R-312), so a pin written yesterday would already be wrong;
#   * gate_code_identity hashes the FILE, so it moves on any comment and cannot
#     distinguish "the asserted set changed" from "anything changed". A consumer
#     bound to it either refuses every edit or learns to ignore it, and both
#     habits end the same way.
GATE_LOAD_BEARING_IDENTITY = "c499e4efd214a89f"
# TRANSCRIBED from DA's LOAD_BEARING, not imported — the consumer declares what
# it requires, and the suite PROVES agreement against DA's live list so drift
# goes red instead of silently shrinking what this harness checks. Three names
# was bypassable: DA declares eight load-bearing, so omitting five was ACCEPTED.
GATE_LOAD_BEARING = (
    "per_row_conformance_exact",
    "both_splits_populated",
    "gap_count_matches_expected",
    "provenance_matches_expected",
    "dataset_non_empty",
    "no_rows_skipped_by_builder",
    "absorption_within_bound",
    "half_open_containment_landed",
)
GATE_GOVERNED_EMBARGO = "embargo_respected"

# The EXACT state each governed name must be in. An exclusion permits a
# predicate to FAIL; it never permits it to disappear, to be waived as N/A, or
# to quietly PASS — the failed set must be exactly the ruled pair, no more and
# no fewer. DA declares both_splits_populated ALWAYS-APPLICABLE and
# NON-WAIVABLE, so accepting it as N/A was accepting a disappearance.
#
# embargo_respected's state is DA'S TO DECLARE, not mine to infer. Codex expects
# applicable-and-failing; my v3 tape printed embargo NOT_APPLICABLE — but that
# is the TAPE's embargo field, a different thing from the GATE predicate's
# emitted state, and resolving the two by inference is exactly what R-302
# forbids. Left None: the consumer REFUSES until DA declares it, because an
# undeclared expectation defaulted into code is a guess wearing a constant's
# clothes.
GOVERNED_STATES = {
    "both_splits_populated": {
        "applicable": True, "pass": False,
        "why": ("R-303 ruled the empty train split, so this predicate FAILS by "
                "design. DA declares it always-applicable and non-waivable: "
                "R-310 permits its FAILURE, never its disappearance.")},
    GATE_GOVERNED_EMBARGO: {
        "applicable": True, "pass": False,
        "why": ("DECLARED BY DA (Q-DA-164), executed from the writer code, not "
                "inferred. My tape header says 'embargo NOT_APPLICABLE' and I "
                "would have guessed that state — WRONGLY. They are two "
                "different objects wearing one phrase: the header is the "
                "TAPE'S SELF-DECLARATION, while the gate keys applicability on "
                "the substring 'unpurged' alone and does not honour a tape's "
                "self-declared status (R-306: a tape may not earn a waiver by "
                "declaring its own). My string lacks 'unpurged', so the gate "
                "takes the APPLICABLE branch and the predicate fails.")},
}

# DA's structural pins for the v4 fragment population (empty train / score-only).
# SCOPE, stated by DA and kept here: the applicability vector and these two named
# states are STRUCTURAL and pinnable. DA is NOT pre-declaring pass/fail for the
# DATA-DEPENDENT predicates (gap_count, at_g0, ledger_pin, containment) — a
# declaration that guesses is not a pin. Those still must pass, but because the
# failed set is pinned to exactly the two exclusions, not because their outcome
# was predicted.
GATE_NOT_APPLICABLE_MUST_BE_EMPTY = True
GATE_REQUIRED_PREDICATES = frozenset(GATE_LOAD_BEARING) | {GATE_GOVERNED_EMBARGO}

DIAGNOSTIC_PREDICATE_EXCLUSIONS = {
    "both_splits_populated":
        "R-303 RULED the empty train split. This diagnostic TRAINS NOTHING — "
        "the candidate is frozen and the fragment population is SCORE-ONLY — so "
        "an empty train split is the truthful shape of what exists. Pointing "
        "both splits at the fragment rows was refused as double-counting the "
        "population and asserting training that never happened.",
    "embargo_respected":
        "R-306 admits a NOT_APPLICABLE embargo for a DIAGNOSTIC_NEVER_EVIDENCE "
        "artifact ONLY. With no train rows there is no train label to embargo "
        "AGAINST, so the comparison is UNDEFINED rather than satisfied — which "
        "is why the builder now emits NOT_APPLICABLE instead of the vacuous "
        "CERTIFIED the rehearsal tape carries.",
}

IDENTITY_BEFORE_FDR7 = "3d0b6c8c6dfe9466"
IDENTITY_AT_BUILD = "e27cab9e5f6ce8e5"       # post-FD-R7 rebind
IDENTITY_REBIND_ARTIFACT = "be_fitcode_rebind_v1.json"


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

    _idn = PA.measured_code_identity()["combined"]
    # ---- FD1 residual: the valuation INPUTS, validated before the gate ----
    _LV = str(D.TARGET_LATENCY_MS)

    def _row(lat):
        return {"slug": "w", "side": "BUY_UP", "gen": 1, "t_start": 0.0,
                "latency": lat}

    def _cell(**kw):
        c = {"preventable_value_cents": 10.0, "preventable_shares": 1.0,
             "stale_shares": 0.0}
        c.update(kw)
        return {_LV: c}

    # F3 SEAM (R-319): the validator must run BEFORE the reconstruction it
    # guards. Every unit falsifier below calls assert_valuation_inputs DIRECTLY,
    # so not one of them can see WIRING ORDER — and the validator was in fact
    # wired AFTER the keptrow loop, so a scalar latency cell raised a raw
    # AttributeError before the controlled refusal could fire. This pushes the
    # malformed row through the REAL entry point.
    import tempfile as _tf8
    _d8 = Path(_tf8.mkdtemp())
    _L8 = str(D.TARGET_LATENCY_MS)
    _bad8 = {"slug": "w", "side": "BUY_UP", "gen": 1, "t_start": 0.0,
             "status": "OK", "latency": {_L8: 7}}          # scalar cell
    _e8 = _d8 / "e.json"
    _e8.write_text(json.dumps({"schema": "be_fragment_exposure_v1",
                               "rows": [_bad8]}))
    _kept8 = [{k: _bad8.get(k) for k in ("slug", "day", "t0", "t_start",
                                         "side", "gen", "latency", "coin")}]
    try:
        rejoin_source_fields(_kept8, _e8)
        _r8 = "ACCEPTED"
    except DiagnosticRefused as e:
        _r8 = f"REFUSED:{e}"
    except Exception as e:                                  # noqa: BLE001
        _r8 = f"RAW:{type(e).__name__}"
    ok(_r8.startswith("REFUSED:") and "malformed valuation input" in _r8,
       f"F3 SEAM: a malformed row pushed through the REAL entry point raises a "
       f"CONTROLLED DiagnosticRefused, not a raw AttributeError — the check now "
       f"runs before the thing it guards ({_r8[:44]})")
    import shutil as _sh8; _sh8.rmtree(_d8, ignore_errors=True)

    # THE STRUCTURAL CONTROL, first: a LEGITIMATE no-fill must still PASS.
    # Strictness that redefines absence-of-a-fill as malformed data would refuse
    # real populations — 29,129 PRE_WINDOW rows in this very tape are genuine.
    _zero = _row(_cell(preventable_value_cents=0.0, preventable_shares=0.0))
    _vz = assert_valuation_inputs([_zero], D.TARGET_LATENCY_MS)
    ok(_vz["rows_validated"] == 1,
       "FD1 STRUCTURAL CONTROL: a LEGITIMATE zero-fill row (value 0, shares 0) "
       "PASSES — absence of a fill is an outcome, not malformed data, and a "
       "check that refused it would refuse real populations")
    import harmful_hazard_model as _hm7
    ok(_hm7.keptrow(_zero)[VALUATION_GATE] is False,
       "FD1 and that legitimate row reconstructs to gate FALSE — the SAME "
       "answer every malformed shape gave, which is exactly why the inputs must "
       "be validated rather than the output inspected")

    for _lbl, _r in (
            ("latency = None", _row(None)),
            ("latency missing", {"slug": "w", "side": "BUY_UP", "gen": 1,
                                 "t_start": 0.0}),
            ("target bucket missing", _row({"999": {}})),
            ("latency cell not a mapping", _row({_LV: 7})),
            ("subfield missing", _row({_LV: {"preventable_shares": 1.0}})),
            ("value is a BOOL", _row(_cell(preventable_value_cents=True))),
            ("shares is NaN", _row(_cell(preventable_shares=float("nan")))),
            ("value is Infinity", _row(_cell(
                preventable_value_cents=float("inf")))),
            ("shares NEGATIVE", _row(_cell(preventable_shares=-5.0)))):
        try:
            assert_valuation_inputs([_r], D.TARGET_LATENCY_MS); _gv = ""
        except DiagnosticRefused as e:
            _gv = str(e)
        ok("malformed valuation input" in _gv,
           f"FD1 KNOWN-BAD: {_lbl} is REFUSED — it would otherwise reconstruct "
           f"to the same gate a genuine no-fill produces")

    # a PARTIALLY malformed population: the all-false guard cannot see this
    _mixed = [_row(_cell()), _row(None)]
    try:
        assert_valuation_inputs(_mixed, D.TARGET_LATENCY_MS); _gp = ""
    except DiagnosticRefused as e:
        _gp = str(e)
    ok("1 malformed" in _gp,
       "FD1 KNOWN-BAD: ONE malformed row beside a valid one is REFUSED — the "
       "all-false guard only fires when EVERY row is false, so a partially "
       "malformed population was invisible to it")

    # ---- R-314(3): THE END-TO-END POSITIVE CONTROL -----------------------
    # Nothing called score_stage, so its entire downstream — scoring, the
    # incumbent, the economics, the cells, the receipt — had never executed. A
    # suite can be green over a path that cannot run, and this one was.
    _syn = score_stage(tape_path=None, synthetic=True)
    ok(bool(_syn.get("cells")) and len(_syn["cells"]) == 3,
       f"R-314(3) END-TO-END: score_stage runs to RECEIPT CELLS "
       f"({len(_syn.get('cells', {}))} budgets) — the path that had never "
       f"executed now executes every suite run")
    _cands = [c["candidate_net_cents"] for c in _syn["cells"].values()]
    ok(all(v is not None and v != 0.0 for v in _cands),
       f"R-314(3) THE CELLS ARE NON-ZERO {_cands} — a positive control that "
       f"reached 'receipt cells' with every cell zero would PASS while proving "
       f"nothing, which is the silent-zero failure it exists to detect")
    ok(all(c["threshold_mode_candidate"] == CAUSAL_MODE
           and c["threshold_mode_incumbent"] == CAUSAL_MODE
           for c in _syn["cells"].values()),
       f"R-314(3) BOTH ARMS ran {CAUSAL_MODE} end-to-end — the FD-R3 look-ahead "
       f"cannot reappear silently")
    ok(all(c["increment_vs_incumbent_cents"] is not None
           for c in _syn["cells"].values()),
       "R-314(3) the incumbent counterpart is computed on the SAME rows, so "
       "every cell carries a real increment rather than a None")
    # R-318(2): THE CONTROL PASSED WITH A DEAD INCUMBENT. threshold_mode proves
    # the PATH ran, not that the ARM did, and a non-None increment is perfectly
    # compatible with subtracting a column of zeros. Both arms must be shown
    # ALIVE: finite, NON-CONSTANT score vectors and NON-ZERO economics.
    _incs = [c["incumbent_net_cents"] for c in _syn["cells"].values()]
    ok(all(v is not None and v != 0.0 for v in _incs),
       f"R-318(2) THE INCUMBENT ARM IS ALIVE: its economics are non-zero "
       f"{_incs} — an all-zero incumbent left every previous assertion green "
       f"while the arm did nothing")
    ok(all(c["candidate_net_cents"] != c["incumbent_net_cents"]
           for c in _syn["cells"].values()),
       "R-318(2) the two arms DIFFER at every budget — identical columns would "
       "mean one vector was standing in for both")
    _sv = _syn.get("score_vectors_alive")
    ok(_sv and _sv["candidate"]["distinct"] > 1 and _sv["incumbent"]["distinct"] > 1
       and _sv["candidate"]["all_finite"] and _sv["incumbent"]["all_finite"],
       f"R-318(2) BOTH score vectors are finite and NON-CONSTANT {_sv} — a "
       f"constant vector ranks nothing and would still produce cells")
    ok(_syn["status"] == "SYNTHETIC_SELFTEST_NOT_A_RESULT"
       and _syn["provenance"]["real_data_read"] is False,
       "R-314(3) the synthetic receipt CANNOT masquerade as a measurement: it "
       "says so in the status and in real_data_read, the fields a reader looks "
       "at first")

    # ---- FD1 / R-315 / R-316: the SILENT ZERO cannot come back -----------
    import tempfile as _tf6, shutil as _sh6
    import harmful_hazard_model as _hm6
    _d6 = Path(_tf6.mkdtemp())
    _LK = str(D.TARGET_LATENCY_MS)

    def _exp(rows):
        q = _d6 / f"e{abs(hash(json.dumps(rows, sort_keys=True)))}.json"
        q.write_text(json.dumps({"schema": "be_fragment_exposure_v1",
                                 "rows": rows}))
        return q

    def _src(gen, v, shares=1.0, status="OK"):
        return {"slug": "w", "side": "BUY_UP", "gen": gen, "t_start": float(gen),
                "day": "d", "t0": 1787897400, "coin": "btc", "status": status,
                "any_fill_ahead": bool(shares),
                "latency": {_LK: {"preventable_value_cents": v,
                                  "preventable_shares": shares,
                                  "stale_shares": 0.0}}}

    def _proj(r):                      # what _feature_pass actually hands back
        return {k: r.get(k) for k in ("slug", "day", "t0", "t_start", "side",
                                      "gen", "latency", "coin")}

    _rows = [_src(1, 50.0), _src(2, -20.0)]
    _kept = [_proj(r) for r in _rows]
    ok(all("status" not in k and VALUATION_GATE not in k for k in _kept),
       "FD1 the projection genuinely lacks BOTH fields — the fixture reproduces "
       "_feature_pass's own output shape rather than assuming it")
    _rj = rejoin_source_fields(_kept, _exp(_rows))
    ok(all(k.get("status") == "OK" and isinstance(k.get(VALUATION_GATE), bool)
           for k in _kept),
       f"FD1 POSITIVE CONTROL: the re-join restores status AND a BOOLEAN "
       f"valuation gate on every kept row ({_rj['valuation_gate_true']} true)")
    _canon = _hm6.keptrow(_rows[0])[VALUATION_GATE]
    ok(_kept[0][VALUATION_GATE] == _canon,
       "FD1 the gate is the CANONICAL reconstruction (hm.keptrow), the same "
       "composition stage_score runs — joining the raw field would have made a "
       "THIRD rule for one quantity")

    # THE SILENT ZERO ITSELF: without the gate every cent is 0.0.
    _ev0 = HAE.evaluate_policy(_kept, [0.9, 0.1], latency_ms=D.TARGET_LATENCY_MS,
                               budgets=(0.5,), n_random=200)
    _b0 = list(_ev0["budgets"])[0]
    _stripped = [{k: v for k, v in r.items() if k != VALUATION_GATE}
                 for r in _kept]
    _evz = HAE.evaluate_policy(_stripped, [0.9, 0.1],
                               latency_ms=D.TARGET_LATENCY_MS,
                               budgets=(0.5,), n_random=200)
    ok(_ev0["budgets"][_b0]["net_cents"] != 0.0
       and _evz["budgets"][_b0]["net_cents"] == 0.0,
       f"FD1 THE MECHANISM, executed: with the gate net is "
       f"{_ev0['budgets'][_b0]['net_cents']}, WITHOUT it exactly "
       f"{_evz['budgets'][_b0]['net_cents']} — a receipt of zeros that reads as "
       f"a measured negative")
    try:
        rejoin_source_fields([_proj(_src(1, 50.0, shares=0.0))],
                             _exp([_src(1, 50.0, shares=0.0)])); _gz = ""
    except DiagnosticRefused as e:
        _gz = str(e)
    ok("False on ALL" in _gz,
       "FD1 KNOWN-BAD: a population whose valuation gate is False on EVERY row "
       "REFUSES — uniformly zero cents is indistinguishable from a broken join "
       "and must not be published as a negative result")
    try:
        rejoin_source_fields(_kept + [_proj(_src(9, 1.0))], _exp(_rows)); _gm = ""
    except DiagnosticRefused as e:
        _gm = str(e)
    ok("NO source row" in _gm,
       "FD1 KNOWN-BAD: a kept row with NO source row REFUSES rather than being "
       "silently valued at zero")
    try:
        rejoin_source_fields([_kept[0], dict(_kept[0])], _exp(_rows)); _gd = ""
    except DiagnosticRefused as e:
        _gd = str(e)
    ok("MORE THAN" in _gd,
       "FD1/R-316 KNOWN-BAD: a DUPLICATED kept identity REFUSES — exactly one "
       "source row per kept row, duplicates as well as misses")
    try:
        rejoin_source_fields(_kept, _exp(_rows + [_src(1, 99.0)])); _gs = ""
    except DiagnosticRefused as e:
        _gs = str(e)
    ok("AMBIGUOUS" in _gs,
       "FD1 KNOWN-BAD: a duplicated SOURCE identity REFUSES — attaching either "
       "one would be a coin flip recorded as a measurement")
    _sh6.rmtree(_d6, ignore_errors=True)

    # ---- R-313: ragged rows refused INDEPENDENTLY of DA's gate -----------
    import tempfile as _tf5, shutil as _sh5
    import phase2_state_schema_freeze as _PIN5
    _d5 = Path(_tf5.mkdtemp())
    _fe = _PIN5.build_pin()["features_in_order"]

    def _tape_with(state):
        q = _d5 / "r.json"
        q.write_text(json.dumps({"protocol": "PHASE2_STATE_TAPE_V5", "rows": [{
            "slug": "btc-updown-5m-1787897400", "side": "BUY_UP", "gen": 1,
            "t_start": 0.0, "t0": 1787897400.0, "split": "score",
            "state_status": "OK", "state": state}]}))
        return q

    _ok = _index_tape(_tape_with({f: 1.0 for f in _fe}), split="score")
    ok(len(_ok) == 1,
       f"R-313 POSITIVE CONTROL: a row declaring all {len(_fe)} pinned fields is "
       f"indexed — a check that refused everything would pass the known-bads")
    for _label, _state in (("ONE field of 45", {_fe[0]: 1.0}),
                           ("44 of 45 (one missing)",
                            {f: 1.0 for f in _fe[:-1]}),
                           ("46 (an undeclared extra)",
                            dict({f: 1.0 for f in _fe}, not_a_feature=1.0))):
        try:
            _index_tape(_tape_with(_state), split="score"); _gr = ""
        except DiagnosticRefused as e:
            _gr = str(e)
        ok("declares" in _gr and "not the pinned" in _gr,
           f"R-313 KNOWN-BAD: a row declaring {_label} is REFUSED by MY indexer, "
           f"independently of DA's gate")
    # THE COUNT/IDENTITY GAP — none of the three above exercises it, because
    # every one of them moves the field COUNT.
    _sub_keep_guard = {f: 1.0 for f in _fe if f != "bn_feed_age_s"}
    _sub_keep_guard["not_a_feature"] = 1.0
    _sub_drop_guard = {f: 1.0 for f in _fe
                       if f not in ("bn_feed_age_s", "bn_feed_missing")}
    _sub_drop_guard["x1"] = 1.0
    _sub_drop_guard["x2"] = 1.0
    for _lbl, _st in (("value substituted, guard kept", _sub_keep_guard),
                      ("value AND ITS GUARD substituted", _sub_drop_guard)):
        ok(len(_st) == len(_fe), f"the {_lbl} fixture PRESERVES the count "
                                 f"({len(_st)}) — otherwise it would be caught "
                                 f"by the count alone and prove nothing")
        try:
            _index_tape(_tape_with(_st), split="score"); _gs = ""
        except DiagnosticRefused as e:
            _gs = str(e)
        ok("MISSING" in _gs and "EXTRA" in _gs,
           f"R-313/FD5 KNOWN-BAD: a COUNT-PRESERVING substitution ({_lbl}) is "
           f"REFUSED, naming the missing and extra fields — counting the fields "
           f"is not checking them")
    _rag = _PIN5.encode_row({_fe[0]: 7.0}, _fe)
    ok(_rag[1] == 0.0 and len(set(_rag)) == 2,
       "R-313 the reason it matters, asserted not assumed: encode_row turns the "
       "44 absent fields into 0.0 — and their guard flags into 0.0 too, i.e. "
       "NOT MISSING. A ragged row does not degrade the score, it lies to it")
    _sh5.rmtree(_d5, ignore_errors=True)

    # ---- R-310: the exclusion-list binding -------------------------------
    import tempfile as _tf4, shutil as _sh4
    _d4 = Path(_tf4.mkdtemp())
    # a SEPARATE exposure fixture: a file cannot contain its own hash, and
    # making the tape its own exposure input was circular by construction.
    _expf = _d4 / "exposure.json"
    _expf.write_text(json.dumps({"schema": "be_fragment_exposure_v1",
                                 "rows_by_status": {"OK": 0}, "rows": []}))
    _expsha = hashlib.sha256(_expf.read_bytes()).hexdigest()
    _tape = _d4 / "t.json"
    _tape.write_text(json.dumps({"protocol": "PHASE2_STATE_TAPE_V5",
                                 "clock_basis": {},
                                 "input_sha256": {"score": _expsha},
                                 "rows": []}))
    _tsha = _sha16(_tape)

    def _full_states(over=None):
        rows = [{"predicate": n, "applicable": True,
                 "pass": n not in DIAGNOSTIC_PREDICATE_EXCLUSIONS}
                for n in sorted(GATE_REQUIRED_PREDICATES)]
        for r in rows:
            if over and r["predicate"] in over:
                r.update(over[r["predicate"]])
        return rows

    def _try_preds_raw(rows, extra=None):
        (_d4 / "da_verdict_probe.json").write_text(json.dumps(
            {"predicates": rows + list(extra or []), "all_pass": False,
             "tape_path": str(_tape), "tape_sha256_prefix": _tsha}))
        _o = globals()["DERIVED"]
        try:
            globals()["DERIVED"] = _d4
            load_gate_verdict(_tape, _expf); return ""
        except DiagnosticRefused as e:
            return str(e)
        finally:
            globals()["DERIVED"] = _o

    def _verdict(fails, names=None, extra=None):
        names = names or (sorted(GATE_REQUIRED_PREDICATES)
                          + ["whole_stream_conformance"])
        preds = [{"predicate": n, "applicable": True, "pass": n not in fails}
                 for n in names]
        preds += list(extra or [])
        return {"predicates": preds,
                "all_pass": not fails,
                "tape_path": str(_tape), "tape_sha256_prefix": _tsha}

    def _try(fails, name="da_verdict_probe.json"):
        vp = _d4 / name
        vp.write_text(json.dumps(_verdict(fails)))
        _orig = globals()["DERIVED"]
        _keep = GOVERNED_STATES[GATE_GOVERNED_EMBARGO]
        try:
            globals()["DERIVED"] = _d4          # test scope, visible here
            GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = {
                "applicable": True, "pass": False, "why": "TEST-SCOPE ONLY"}
            return load_gate_verdict(_tape, _expf), ""
        except DiagnosticRefused as e:
            return None, str(e)
        finally:
            globals()["DERIVED"] = _orig
            GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = _keep

    _g, _e = _try(["both_splits_populated", "embargo_respected"])
    ok(_g is not None and set(_g["predicates_failed_and_EXCUSED_by_policy"]) ==
       {"both_splits_populated", "embargo_respected"},
       f"R-310 POSITIVE CONTROL: a verdict failing EXACTLY the ruled pair is "
       f"consumed, and both appear in the receipt WITH their citations "
       f"({_e[:50]})")
    ok(_g and all(len(v) > 40 for v in
                  _g["predicates_failed_and_EXCUSED_by_policy"].values()),
       "R-310 each excused failure carries its RULING verbatim, not a bare name "
       "— an exclusion without its citation is indistinguishable from a "
       "convenience")
    _g2, _e2 = _try(["whole_stream_conformance"])
    ok(_g2 is None and "whole_stream_conformance" in _e2 and "REFUSED" in _e2,
       f"R-310 KNOWN-BAD: a NON-EXCLUDED failing predicate REFUSES — consuming "
       f"a verdict with a conformance failure merely 'named' would let it "
       f"through ({_e2[:50]})")
    _g3, _e3 = _try(["both_splits_populated", "whole_stream_conformance"])
    ok(_g3 is None and "whole_stream_conformance" in _e3,
       "R-310 KNOWN-BAD: an excluded failure alongside a non-excluded one still "
       "REFUSES — the exclusion list does not widen to cover its neighbours")
    ok("DIAGNOSTIC_PREDICATE_EXCLUSIONS" in inspect.getsource(load_gate_verdict)
       and set(DIAGNOSTIC_PREDICATE_EXCLUSIONS) ==
           {"both_splits_populated", "embargo_respected"},
       "R-310 the exclusions are a POLICY CONSTANT of exactly the ruled two — "
       "read from the consumer's own declaration, never from the verdict or the "
       "tape's self-declaration (a shrinkable allowlist is a consumer choosing "
       "what it needs)")
    # FD2 consumer: COMPARE the exposure stamp, do not merely compute it
    _other = _d4 / "other_exposure.json"
    _other.write_text(json.dumps({"schema": "be_fragment_exposure_v1",
                                  "rows_by_status": {"OK": 1}, "rows": []}))
    (_d4 / "da_verdict_probe.json").write_text(json.dumps(_verdict(
        ["both_splits_populated", "embargo_respected"])))
    _o = globals()["DERIVED"]
    _k2 = GOVERNED_STATES[GATE_GOVERNED_EMBARGO]
    try:
        globals()["DERIVED"] = _d4
        GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = {
            "applicable": True, "pass": False, "why": "TEST-SCOPE ONLY"}
        try:
            load_gate_verdict(_tape, _other); _ge = ""
        except DiagnosticRefused as e:
            _ge = str(e)
    finally:
        globals()["DERIVED"] = _o
        GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = _k2
    ok("DIFFERENT exposure file" in _ge,
       f"PR3-FD2 KNOWN-BAD: a DIFFERENT exposure file than the tape was built "
       f"from is REFUSED — hashing whatever the caller supplies and comparing "
       f"it against nothing accepted any file ({_ge[:44]})")

    # FD4: an OMITTED governed predicate refuses; a NEW passing one is fine
    # The embargo state is DA's to declare and production REFUSES without it.
    # The suite supplies a test-scope declaration so the rest of the machinery
    # stays exercised — visible right here, never in production.
    _TEST_EMBARGO_STATE = {"applicable": True, "pass": False,
                           "why": "TEST-SCOPE ONLY — DA declares the real one"}

    def _with_declared(fn):
        _keep = GOVERNED_STATES[GATE_GOVERNED_EMBARGO]
        GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = _TEST_EMBARGO_STATE
        try:
            return fn()
        finally:
            GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = _keep

    _emb = GOVERNED_STATES[GATE_GOVERNED_EMBARGO]
    ok(_emb is not None and _emb["applicable"] is True and _emb["pass"] is False,
       "R-302 the embargo predicate's state is DECLARED BY DA (applicable, "
       "failing) — and NOT inferred: my own tape header reads 'embargo "
       "NOT_APPLICABLE', which is the TAPE'S self-declaration, a different "
       "object from the GATE predicate's emitted state. Guessing from my "
       "artifact would have pinned the wrong value")
    ok("unpurged" in _emb["why"] and "self-declar" in _emb["why"].lower(),
       "R-302 the declaration carries its MECHANISM — the gate keys "
       "applicability on 'unpurged' and does not honour a tape's self-declared "
       "status — so a later reader can check the reasoning, not just the value")
    _keep_e = GOVERNED_STATES[GATE_GOVERNED_EMBARGO]
    GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = None
    try:
        _und = _try_preds_raw(_full_states())
    finally:
        GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = _keep_e
    ok("has not been DECLARED" in _und,
       "R-302 KNOWN-BAD: with a governed state UNDECLARED, even a PERFECT "
       "verdict is REFUSED — the refusal is the declaration's absence, not the "
       "verdict's fault, and it keeps a future governed name from defaulting "
       "silently")

    def _try_preds(preds, extra=None):
        (_d4 / "da_verdict_probe.json").write_text(json.dumps(
            {"predicates": [{"predicate": n, "applicable": True, "pass": q}
                            for n, q in preds] + list(extra or []),
             "all_pass": all(q for _, q in preds),
             "tape_path": str(_tape), "tape_sha256_prefix": _tsha}))
        _oo = globals()["DERIVED"]
        _kk = GOVERNED_STATES[GATE_GOVERNED_EMBARGO]
        try:
            globals()["DERIVED"] = _d4
            GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = {
                "applicable": True, "pass": False, "why": "TEST-SCOPE ONLY"}
            load_gate_verdict(_tape, _expf); return ""
        except DiagnosticRefused as e:
            return str(e)
        finally:
            globals()["DERIVED"] = _oo
            GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = _kk
    _EXC = set(DIAGNOSTIC_PREDICATE_EXCLUSIONS)
    _full = [(n, n not in _EXC) for n in sorted(GATE_REQUIRED_PREDICATES)]
    ok(_try_preds(_full) == "",
       "FD4 POSITIVE CONTROL: the governed universe present -> consumed")
    ok(_try_preds(_full + [("brand_new_check", True)]) == "",
       "FD4 a NEW predicate that PASSES is allowed — the universe is a MINIMUM, "
       "not a closed set")
    _om = _try_preds([p for p in _full if p[0] != "embargo_respected"])
    ok("does not contain governed predicate" in _om,
       f"FD4 KNOWN-BAD: an OMITTED governed predicate REFUSES — a gate that "
       f"stops emitting a check is indistinguishable from one whose check "
       f"passed ({_om[:44]})")
    (_d4 / "da_verdict_probe.json").write_text(json.dumps(
        {"predicates": [{"predicate": n, "applicable": True, "pass": q}
                        for n, q in _full],
         "all_pass": False, "tape_path": str(_tape),
         "tape_sha256_prefix": "short"}))
    _o2 = globals()["DERIVED"]
    _k3 = GOVERNED_STATES[GATE_GOVERNED_EMBARGO]
    try:
        globals()["DERIVED"] = _d4
        GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = {
            "applicable": True, "pass": False, "why": "TEST-SCOPE ONLY"}
        try:
            load_gate_verdict(_tape, _expf); _gh = ""
        except DiagnosticRefused as e:
            _gh = str(e)
    finally:
        globals()["DERIVED"] = _o2
        GOVERNED_STATES[GATE_GOVERNED_EMBARGO] = _k3
    ok("well-formed tape_sha256_prefix" in _gh,
       "FD4 KNOWN-BAD: a malformed subject hash REFUSES — an unbound subject "
       "means the verdict could be about any tape")

    # FD4 (R-318): the pin AGREES WITH DA's live declaration, or goes red
    import da_state_tape_verify as _G
    _lbi = _G.load_bearing_identity()
    ok(_lbi["sha256"] == GATE_LOAD_BEARING_IDENTITY,
       f"FD4 AGREEMENT, RECOMPUTED LIVE: DA's load_bearing_identity is "
       f"{_lbi['sha256']} and I am pinned to {GATE_LOAD_BEARING_IDENTITY}. A "
       f"mismatch means the ASSERTED SET moved and my expectations are stale — "
       f"which is a different fact from 'the gate file changed', and the reason "
       f"binding to the file hash could not serve")
    ok(set(GATE_LOAD_BEARING) == set(_G.LOAD_BEARING),
       f"FD4 my own versioned name list still agrees with DA's live set "
       f"({len(GATE_LOAD_BEARING)} vs {len(_G.LOAD_BEARING)}) — the identity "
       f"hash says THAT it moved, the name list says WHICH names I require")
    ok(_lbi["n"] == len(_G.LOAD_BEARING) == len(GATE_LOAD_BEARING),
       "FD4 the identity's own count agrees with both lists")
    import hashlib as _h4
    _perm = _h4.sha256("\n".join(sorted(reversed(list(_G.LOAD_BEARING))))
                       .encode()).hexdigest()[:16]
    ok(_perm == _lbi["sha256"],
       "FD4 the identity is ORDER-INDEPENDENT by construction (a reversed tuple "
       "hashes the same), so I need not care how DA writes it")
    ok(len(GATE_REQUIRED_PREDICATES) == len(_G.LOAD_BEARING) + 1,
       "FD4 the governed universe is DA's load-bearing set PLUS the governed "
       "embargo name")
    _dup = _try_preds(_full, extra=[{"predicate": "dataset_non_empty",
                                     "applicable": True, "pass": True}])
    ok("MORE THAN ONCE" in _dup,
       "FD4 KNOWN-BAD: a DUPLICATED governed predicate REFUSES — two rows for "
       "one check means one of them decided and nothing says which")
    _na = _try_preds([(n, q) for n, q in _full
                      if n != "gap_count_matches_expected"],
                     extra=[{"predicate": "gap_count_matches_expected",
                             "applicable": False, "pass": False}])
    ok("not applicable" in _na.lower(),
       "FD4 KNOWN-BAD: a required check marked applicable=false REFUSES — DA "
       "pins the applicability vector ALL-TRUE for this population, so a "
       "waiver is not a pass and relabelling cannot skip a check")
    _noapp = _try_preds([(n, q) for n, q in _full if n != "dataset_non_empty"],
                        extra=[{"predicate": "dataset_non_empty",
                                "pass": False}])
    ok(_noapp != "",
       "FD4 KNOWN-BAD: a predicate emitted WITHOUT the applicable key defaults "
       "to TRUE as DA's writer does — my reader defaulted it to False, silently "
       "dropping a governed check from the denominator")

    # F1 (R-319): an exclusion permits FAILURE, never disappearance or a pass
    ok(_with_declared(lambda: _try_preds_raw(_full_states())) == "",
       "F1 POSITIVE CONTROL: the exact ruled shape — both exclusions "
       "applicable-and-FAILING, everything else applicable-and-passing — is "
       "consumed")
    _bs = _with_declared(lambda: _try_preds_raw(_full_states(
        {"both_splits_populated": {"applicable": False, "pass": False}})))
    ok("applicable=False" in _bs or "NOT APPLICABLE" in _bs or "declared state" in _bs,
       f"F1 KNOWN-BAD: an EXCLUSION marked N/A is REFUSED — DA declares "
       f"both_splits_populated always-applicable and non-waivable, so R-310 "
       f"permits its FAILURE, not its DISAPPEARANCE ({_bs[:44]})")
    _bp = _with_declared(lambda: _try_preds_raw(_full_states(
        {"both_splits_populated": {"applicable": True, "pass": True}})))
    ok(_bp != "",
       "F1 KNOWN-BAD: an EXCLUSION that PASSES is REFUSED — the ruling "
       "describes a specific expected shape, not a ceiling, and a surprise "
       "pass is as much a surprise as a missing failure")
    _ef = _with_declared(lambda: _try_preds_raw(_full_states(
        {"dataset_non_empty": {"applicable": True, "pass": False}})))
    ok("NOT on the policy exclusion list" in _ef,
       "F1 KNOWN-BAD: an EXTRA failure beyond the ruled pair is REFUSED (by the "
       "unexcused check, which reaches it first — the failed-set equality "
       "catches the other direction, an exclusion that PASSES)")
    ok(set(DIAGNOSTIC_PREDICATE_EXCLUSIONS) == {"both_splits_populated",
                                                GATE_GOVERNED_EMBARGO},
       "F1 the two checks together pin the failed set from BOTH sides: no extra "
       "failures (unexcused) and no missing ones (failed-set equality)")

    # F2: a NEW predicate declared N/A slipped through every earlier check
    _nna = _with_declared(lambda: _try_preds_raw(
        _full_states(), extra=[{"predicate": "brand_new_check",
                                "applicable": False, "pass": False}]))
    ok("N/A or" in _nna or "inert" in _nna,
       "F2 KNOWN-BAD: a NON-GOVERNED predicate declared N/A is REFUSED — it is "
       "not governed, so it was excluded from `applicable` and never reached "
       "the failure check; additional may not mean inert")
    _nnf = _with_declared(lambda: _try_preds_raw(
        _full_states(), extra=[{"predicate": "brand_new_check",
                                "applicable": True, "pass": False}]))
    ok(_nnf != "",
       "F2 KNOWN-BAD: a NON-GOVERNED predicate that FAILS is REFUSED")
    ok(_with_declared(lambda: _try_preds_raw(
        _full_states(), extra=[{"predicate": "brand_new_check",
                                "applicable": True, "pass": True}])) == "",
       "F2 POSITIVE CONTROL: a new predicate that is applicable AND passing is "
       "allowed — the universe is a minimum, not a closed set")

    # FD3: the reconcile boolean is READ, not merely reported
    ok("recon[\"reconciles\"]" in inspect.getsource(score_stage),
       "FD3 score_stage READS the reconciliation boolean; a receipt field "
       "nobody checks is a decoration")
    _r3 = reconcile_population(
        [{"slug": "s", "side": "BUY_UP", "gen": 1, "t_start": 0.0}],
        {"state_join_failed": 0}, 2)
    ok(_r3["reconciles"] is False,
       "FD3 the boolean is FALSE when kept+drops (1) misses the declared OK "
       "count (2) — the case score_stage now refuses on")

    _sh4.rmtree(_d4, ignore_errors=True)

    # ---- FD-R5: the CLI refuses rather than exiting clean ----------------
    ok(main([]) == 2, "FD-R5 no arguments EXITS NON-ZERO")
    ok(main(["--not-a-mode"]) == 2,
       "FD-R5 an UNKNOWN MODE exits non-zero — a mode typo that returns 0 is a "
       "run that looks like it happened")
    for bad, want in ((["--score"], "requires --tape"),
                      (["--score", "--tape", "/nope", "--out", "/tmp/x.json",
                        "--reason", "r"], "no tape at")):
        try:
            main(bad); _g = ""
        except SystemExit as e:
            _g = str(e)
        ok(want in _g, f"FD-R5 KNOWN-BAD: {' '.join(bad)[:34]} -> REFUSED")
    import tempfile as _tf3
    _d3 = Path(_tf3.mkdtemp()); _ex = _d3 / "exists.json"; _ex.write_text("{}")
    try:
        main(["--score", "--tape", str(_ex), "--out", str(_ex),
              "--reason", "r"]); _g2 = ""
    except SystemExit as e:
        _g2 = str(e)
    ok("already exists" in _g2,
       "FD-R5 KNOWN-BAD: an EXISTING output is refused — R-293 permits one run "
       "and overwriting destroys the evidence the previous one happened")
    try:
        main(["--score", "--tape", str(_ex), "--out", str(_d3 / "fresh.json")])
        _g3 = ""
    except SystemExit as e:
        _g3 = str(e)
    ok("--reason is required" in _g3,
       "FD-R5 KNOWN-BAD: a run with NO WRITTEN REASON is refused; a reason "
       "supplied after the numbers is chosen with them in view")
    import shutil as _sh3; _sh3.rmtree(_d3, ignore_errors=True)

    # ---- FD-R6: the reconciliation refuses what it must ------------------
    _rows = [{"slug": "s", "side": "BUY_UP", "gen": 1, "t_start": 0.0}]
    try:
        reconcile_population(_rows, {"state_join_failed": 3}, 4); _g4 = ""
    except DiagnosticRefused as e:
        _g4 = str(e)
    ok("state_join_failed=3" in _g4,
       "FD-R6 KNOWN-BAD: a non-zero state_join_failed REFUSES — the tape does "
       "not cover the population, and a partial join reads as a clean small "
       "result rather than a missing input")
    try:
        reconcile_population(_rows + _rows, {"state_join_failed": 0}, 2); _g5 = ""
    except DiagnosticRefused as e:
        _g5 = str(e)
    ok("DUPLICATE" in _g5,
       "FD-R6 KNOWN-BAD: duplicate decision rows REFUSE (rule 2: one outcome "
       "attributed to several rows inflates every count)")
    _r6 = reconcile_population(_rows, {"pm": 2, "state_join_failed": 0}, 3)
    ok(_r6["reconciles"] and _r6["n_actions"] == 1,
       "FD-R6 POSITIVE CONTROL: a clean population reconciles kept+drops to the "
       "exposure OK count, with the ACTION count reported")

    ok(_idn == IDENTITY_AT_BUILD,
       f"the lattice identity is the POST-REBIND value {IDENTITY_AT_BUILD} "
       f"(was {IDENTITY_BEFORE_FDR7}; moved ONCE under FD-R7, evidenced in "
       f"{IDENTITY_REBIND_ARTIFACT}). Measured {_idn}")
    ok(_idn != IDENTITY_BEFORE_FDR7,
       "the pre-rebind identity is NOT silently still in force — if the repair "
       "were reverted this would catch it rather than passing quietly")

    print(f"\n{'BE FRAGMENT DIAGNOSTIC SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


USAGE = ("usage: be_fragment_diagnostic.py --selftest\n"
         "       be_fragment_diagnostic.py --build-rows\n"
         "       be_fragment_diagnostic.py --score --tape PATH "
         "[--exposure PATH] --out PATH --reason TEXT")


def main(argv=None) -> int:
    """FD-R5: a REAL cli. An unknown mode EXITS NON-ZERO rather than printing
    usage and returning 0 -- a mode typo that exits clean is a run that looks
    like it happened."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(USAGE, file=sys.stderr)
        return 2
    mode = argv[0]

    def opt(name, required=True):
        if name in argv:
            k = argv.index(name)
            if k + 1 < len(argv):
                return argv[k + 1]
        if required:
            raise SystemExit(f"REFUSED: {mode} requires {name} PATH.\n{USAGE}")
        return None

    if mode == "--selftest":
        return selftest()
    if mode == "--build-rows":
        print(json.dumps(build_rows_stage(), indent=1, sort_keys=True)[:1600])
        return 0
    if mode == "--score":
        tape = Path(opt("--tape"))
        out = Path(opt("--out"))
        exposure = opt("--exposure", required=False)
        reason = opt("--reason", required=False)
        if not tape.exists():
            raise SystemExit(f"REFUSED: no tape at {tape}.")
        if out.exists():
            raise SystemExit(
                f"REFUSED: {out} already exists. R-293 permits ONE run; "
                f"overwriting the artifact of a previous one destroys the "
                f"evidence that it happened. Choose a fresh path.")
        if not reason:
            raise SystemExit(
                "REFUSED: --reason is required. R-293 permits ONE run and a "
                "re-run requires its own WRITTEN REASON RECORDED BEFORE IT; a "
                "reason supplied after the numbers is chosen with them in view.")
        res = score_stage(tape, exposure_path=exposure)
        res["run_reason"] = reason
        res["invocation"] = {"argv": argv, "tape": str(tape),
                             "exposure": str(exposure or ROWS_OUT),
                             "out": str(out)}
        out.write_text(json.dumps(res, indent=1, sort_keys=True,
                                  allow_nan=False))
        print(f"[frag] wrote {out}", flush=True)
        print(json.dumps(res["cells"], indent=1, sort_keys=True)[:1800])
        return 0
    print(f"REFUSED: unknown mode {mode!r}.\n{USAGE}", file=sys.stderr)
    return 2




# ---------------------------------------------------------------------------
# STAGE 3: score the frozen candidate on the fragment rows
# ---------------------------------------------------------------------------
def load_gate_verdict(tape_path: Path, exposure_path: Path) -> dict:
    """FD-R1: DA's gate verdict is a REQUIRED INPUT, and it is RECOMPUTED.

    The verdict file says so itself: "ALL_PASS is recomputed from the predicate
    table in this file, not carried in. A consumer should re-derive it." A
    consumer that reads the carried boolean trusts a field instead of the
    evidence beside it, which is the shape that lets a stale or hand-edited
    verdict authorise a score.

    THREE BINDINGS, because a valid verdict about a DIFFERENT artifact is still
    the wrong verdict: the verdict's subject must be THIS tape by path AND by
    content hash, and the tape must have been built from THE EXACT exposure file
    being scored."""
    cands = sorted(Path(DERIVED).glob("da_*verdict*.json"))
    hits = []
    for c in cands:
        try:
            d = json.loads(c.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        if Path(str(d.get("tape_path", ""))).name == Path(tape_path).name:
            hits.append((c, d))
    if not hits:
        raise DiagnosticRefused(
            f"REFUSED: no DA gate verdict names {Path(tape_path).name}. The "
            f"score read is GATED on DA's verdict (R-293); scoring an ungated "
            f"tape is scoring a population nobody certified.")
    if len(hits) > 1:
        raise DiagnosticRefused(
            f"REFUSED: {len(hits)} verdicts name this tape "
            f"({[c.name for c, _ in hits]}). Choosing among them is a decision, "
            f"not a lookup.")
    vpath, v = hits[0]

    # RECOMPUTE all_pass from the predicate table.
    preds = v.get("predicates")
    if not isinstance(preds, list) or not preds:
        raise DiagnosticRefused(
            f"REFUSED: {vpath.name} carries no predicate table; a verdict with "
            f"no evidence cannot be re-derived, only believed.")
    # DA's writer DEFAULTS APPLICABLE TO TRUE; my reader defaulted it to False,
    # so a predicate emitted without the key was silently dropped from the
    # denominator — a contract mismatch that made a governed check vanish.
    applicable = [p for p in preds if p.get("applicable", True)]
    failed = [p.get("predicate") for p in applicable if not p.get("pass")]
    recomputed = bool(applicable) and not failed
    if recomputed != bool(v.get("all_pass")):
        raise DiagnosticRefused(
            f"REFUSED: {vpath.name} carries all_pass={v.get('all_pass')!r} but "
            f"its own predicate table recomputes to {recomputed}. The table is "
            f"the evidence; the field is a claim about it.")
    # R-310: all_pass is NOT the bar for this binding; the bar is "every
    # applicable predicate passes EXCEPT the policy-fixed pair".
    unexcused = [f for f in failed if f not in DIAGNOSTIC_PREDICATE_EXCLUSIONS]
    if unexcused:
        raise DiagnosticRefused(
            f"REFUSED: DA's gate reports failing predicate(s) {unexcused} that "
            f"are NOT on the policy exclusion list "
            f"{sorted(DIAGNOSTIC_PREDICATE_EXCLUSIONS)}. A diagnostic may be "
            f"inadmissible as evidence and still may not be computed over a "
            f"tape that failed a conformance check. Excusing this would be a "
            f"consumer choosing what it needs.")
    excused = {f: DIAGNOSTIC_PREDICATE_EXCLUSIONS[f] for f in failed}
    # FD4: an OMITTED governed predicate refuses. A gate that stops emitting a
    # check is indistinguishable from one whose check passed, unless the
    # consumer declares what it requires to be present.
    from collections import Counter as _C
    counts = _C(p.get("predicate") for p in preds)
    absent = sorted(GATE_REQUIRED_PREDICATES - set(counts))
    if absent:
        raise DiagnosticRefused(
            f"REFUSED: the verdict does not contain governed predicate(s) "
            f"{absent} (universe {GATE_UNIVERSE_VERSION}). A predicate that is "
            f"not emitted is not a predicate that passed.")
    dupes = sorted(n for n in GATE_REQUIRED_PREDICATES if counts[n] > 1)
    if dupes:
        raise DiagnosticRefused(
            f"REFUSED: governed predicate(s) {dupes} appear MORE THAN ONCE. "
            f"Two rows for one check means one of them decided and nothing says "
            f"which.")
    # Each governed name must be in its PERMITTED applicability state. For this
    # fragment: the two ruled exclusions may fail; every other governed name
    # must be APPLICABLE and PASSING. A required check marked applicable=false
    # is a waiver, and a waiver is not a pass.
    by = {p.get("predicate"): p for p in preds}
    # F1: EXACT STATES. The previous form skipped the exclusions entirely when
    # checking applicability, so an exclusion marked N/A was ACCEPTED — the list
    # was acting as a WAIVER rather than as permission-to-fail. An exclusion
    # that PASSED was accepted too, because the failed set was never pinned.
    undeclared = sorted(n for n, st in GOVERNED_STATES.items() if st is None)
    if undeclared:
        raise DiagnosticRefused(
            f"REFUSED: the required state of governed predicate(s) {undeclared} "
            f"has not been DECLARED. Defaulting it here would be inference "
            f"standing in for a declaration (R-302); the owner declares it.")
    for n, st in GOVERNED_STATES.items():
        got_app = by[n].get("applicable", True)
        got_pass = by[n].get("pass")
        if got_app != st["applicable"] or bool(got_pass) != st["pass"]:
            raise DiagnosticRefused(
                f"REFUSED: governed predicate {n!r} is "
                f"applicable={got_app!r} pass={got_pass!r}, but its declared "
                f"state is applicable={st['applicable']} pass={st['pass']}. "
                f"{st['why']}")
    # DA's pin: not_applicable must be an EMPTY LIST. A name parked there is a
    # governed check that ran nowhere.
    if GATE_NOT_APPLICABLE_MUST_BE_EMPTY and (v.get("not_applicable") or []):
        raise DiagnosticRefused(
            f"REFUSED: the verdict declares not_applicable="
            f"{v.get('not_applicable')!r}, but DA pins it EMPTY for this "
            f"population. A predicate parked in not_applicable is a governed "
            f"check that ran nowhere.")
    # DA's pin: EVERY governed predicate is applicable for this population.
    non_applicable = sorted(n for n in GATE_REQUIRED_PREDICATES
                            if not by[n].get("applicable", True))
    if non_applicable:
        raise DiagnosticRefused(
            f"REFUSED: governed predicate(s) {non_applicable} are not "
            f"applicable, but DA pins the applicability vector ALL-TRUE for the "
            f"v4 fragment population. Both exclusions are APPLICABLE FAILURES, "
            f"not waivers.")
    non_excl_na = sorted(
        n for n in GATE_REQUIRED_PREDICATES
        if n not in GOVERNED_STATES and not by[n].get("applicable", True))
    if non_excl_na:
        raise DiagnosticRefused(
            f"REFUSED: governed predicate(s) {non_excl_na} are marked NOT "
            f"APPLICABLE. A waiver is not a pass, and relabelling would let any "
            f"check be skipped.")
    # The failed set must be EXACTLY the ruled pair — no more, and no fewer.
    if set(failed) != set(DIAGNOSTIC_PREDICATE_EXCLUSIONS):
        raise DiagnosticRefused(
            f"REFUSED: the recomputed failed set is {sorted(failed)}, not "
            f"exactly the ruled exclusions "
            f"{sorted(DIAGNOSTIC_PREDICATE_EXCLUSIONS)}. An exclusion that "
            f"PASSES is as much a surprise as one that is missing: the ruling "
            f"describes a specific expected shape, not a ceiling.")
    # F2: ADDITIONAL predicates must be unique, applicable and passing. One
    # declared N/A slipped through every check — not governed, so excluded from
    # `applicable`, so never reaching `unexcused`.
    extra_bad = sorted(
        p.get("predicate") for p in preds
        if p.get("predicate") not in GATE_REQUIRED_PREDICATES
        and (not p.get("applicable", True) or not p.get("pass")))
    if extra_bad:
        raise DiagnosticRefused(
            f"REFUSED: non-governed predicate(s) {extra_bad} are N/A or "
            f"failing. A predicate outside the governed set is still a check "
            f"the gate chose to run; it may be additional, it may not be "
            f"inert.")
    extra_dupes = sorted(n for n, c in counts.items()
                         if c > 1 and n not in GATE_REQUIRED_PREDICATES)
    if extra_dupes:
        raise DiagnosticRefused(
            f"REFUSED: non-governed predicate(s) {extra_dupes} appear more than "
            f"once.")
    # FD4: a well-formed subject hash is MANDATORY, not optional.
    if not isinstance(v.get("tape_sha256_prefix"), str) or \
            len(v["tape_sha256_prefix"]) < 16:
        raise DiagnosticRefused(
            f"REFUSED: the verdict carries no well-formed tape_sha256_prefix "
            f"({v.get('tape_sha256_prefix')!r}). An unbound subject means the "
            f"verdict could be about any tape.")

    # BIND the verdict's subject to THIS tape, by path and by content.
    if Path(str(v.get("tape_path"))).resolve() != Path(tape_path).resolve():
        raise DiagnosticRefused(
            f"REFUSED: the verdict certifies {v.get('tape_path')}, this run "
            f"consumes {tape_path}.")
    want = str(v.get("tape_sha256_prefix", ""))
    got = _sha16(tape_path)
    if want and want != got:
        raise DiagnosticRefused(
            f"REFUSED: verdict certifies tape {want}, on disk it is {got}. A "
            f"valid verdict about different bytes is still the wrong verdict.")

    # BIND the tape to the EXACT exposure input it was built from.
    head = Path(tape_path).open().read(1 << 16)
    meta = json.loads(head[:head.index('"rows"')].rstrip().rstrip(",") + "}")
    built_from = json.dumps(meta.get("per_split", {}))
    exp_sha = _sha16(exposure_path)
    # PR3-FD2: COMPARE, do not merely compute. Hashing whatever exposure file
    # the caller supplied and recording it proves nothing -- two different
    # exposure files would both be "verified". The tape stamps the inputs it was
    # actually built from; this asserts ours is that one.
    stamped = (meta.get("input_sha256") or {}).get("score")
    if stamped is None:
        raise DiagnosticRefused(
            f"REFUSED: {Path(tape_path).name} does not stamp the exposure input "
            f"it was built from (input_sha256.score). Without it this harness "
            f"could hash ANY exposure file and compare it against nothing — "
            f"which is what it did before PR3-FD2.")
    _full = hashlib.sha256()
    with Path(exposure_path).open("rb") as _fh:
        for _b in iter(lambda: _fh.read(1 << 22), b""):
            _full.update(_b)
    if _full.hexdigest() != stamped:
        raise DiagnosticRefused(
            f"REFUSED: the tape was built from exposure input {stamped[:16]} "
            f"but this run supplies {_full.hexdigest()[:16]} "
            f"({Path(exposure_path).name}). Scoring a tape against a DIFFERENT "
            f"exposure file than it was built from joins two populations that "
            f"were never the same one.")
    return {"verdict_file": vpath.name, "verdict_sha256_prefix": _sha16(vpath),
            "all_pass_recomputed": recomputed,
            "binding_rule": ("R-310: every applicable predicate must PASS "
                             "except the policy-fixed exclusions below; any "
                             "other failure REFUSES. Scope: this "
                             "DIAGNOSTIC_NEVER_EVIDENCE binding only — a "
                             "result-bearing binding requires all_pass."),
            "predicates_failed_and_EXCUSED_by_policy": excused,
            "n_predicates_failed": len(failed),
            "n_predicates_passed": len(applicable) - len(failed),
            "exclusions_are": ("REGISTER POLICY, hard-coded in the consumer — "
                               "not read from the verdict, not from the tape's "
                               "self-declaration"),
            "n_applicable": len(applicable),
            "not_applicable": v.get("not_applicable"),
            "tape_path": str(tape_path), "tape_sha256_prefix": got,
            "tape_per_split": built_from,
            "exposure_input": Path(exposure_path).name,
            "exposure_sha256_prefix": exp_sha,
            "exposure_matches_tape_stamp": True,
            "gate_universe_version": GATE_UNIVERSE_VERSION,
            "gate_load_bearing_identity": GATE_LOAD_BEARING_IDENTITY,
            "governed_predicates_required": sorted(GATE_REQUIRED_PREDICATES),
            "tape_header_clock_basis": meta.get("clock_basis"),
            "tape_header_ledger_sha256": meta.get("ledger_sha256"),
            "how_all_pass_was_obtained": "RE-DERIVED from the predicate table, "
                                         "not read from the carried field"}


def reconcile_population(kept: list, drops: dict, expected_rows: int) -> dict:
    """FD-R6: every row accounted for, every status declared, no duplicates.

    state_join_failed MUST be zero: a row dropped because the tape lacks its key
    is not an exclusion, it is a tape that does not cover the population -- the
    exact failure that would otherwise present as a clean, smaller answer."""
    if drops.get("state_join_failed", 0) != 0:
        raise DiagnosticRefused(
            f"REFUSED: state_join_failed={drops['state_join_failed']}. The state "
            f"tape does not cover these rows, so the scored population is not "
            f"the population -- and a partial join reads as a clean small "
            f"result rather than as a missing input.")
    keys = [(r.get("slug"), r.get("side"), r.get("gen"), r.get("t_start"))
            for r in kept]
    if len(set(keys)) != len(keys):
        from collections import Counter
        dup = [k for k, n in Counter(keys).items() if n > 1]
        raise DiagnosticRefused(
            f"REFUSED: {len(keys) - len(set(keys))} DUPLICATE decision rows "
            f"(e.g. {dup[:3]}). One outcome attributed to several rows inflates "
            f"every count taken from them (rule 2).")
    accounted = len(kept) + sum(drops.values())
    return {"rows_kept": len(kept), "drops_by_named_status": dict(drops),
            "rows_accounted": accounted,
            "rows_expected_from_exposure_OK": expected_rows,
            "reconciles": accounted == expected_rows,
            "n_actions": len({(r.get("slug"), r.get("side"), r.get("gen"))
                              for r in kept}),
            "state_join_failed": drops.get("state_join_failed", 0)}


VALUATION_GATE = "any_fill_ahead"
VALUATION_SUBFIELDS = ("preventable_value_cents", "preventable_shares",
                       "stale_shares")


def assert_valuation_inputs(kept: list, latency_ms: int) -> dict:
    """Validate the valuation INPUTS before the gate is reconstructed.

    hm.keptrow computes `_any_fill_ahead(r.get("latency") or {})`. That `or {}`
    turns EVERY malformed shape into a clean gate=False, which is
    INDISTINGUISHABLE FROM A GENUINE NO-FILL. Executed on the real composition:

        latency = None          -> gate False
        latency missing         -> gate False
        target key missing      -> gate False
        shares = NaN            -> gate False
        shares negative         -> gate False
        LEGITIMATE zero-fill    -> gate False   <- the same answer

    So a missing valuation INPUT is silently recorded as a no-fill OUTCOME, and
    neither the all-false guard (which only fires when EVERY row is false) nor
    the boolean assertion (keptrow always returns a bool) can see a PARTIALLY
    malformed population. Worse in the other direction: a row missing
    preventable_value_cents reconstructs to gate TRUE and would be valued.

    Enforced, never repaired. A default here would re-create the defect one
    layer up. The legitimate zero-fill row MUST still pass -- strictness that
    redefines absence-of-a-fill as malformed data would refuse real
    populations, which is why the suite keeps it as a structural control."""
    L = str(latency_ms)
    bad: list = []
    for i, r in enumerate(kept):
        ident = (r.get("slug"), r.get("side"), r.get("gen"), r.get("t_start"))
        lat = r.get("latency")
        if not isinstance(lat, dict):
            bad.append(f"{ident}: latency is {type(lat).__name__}, not a mapping")
            continue
        if L not in lat:
            bad.append(f"{ident}: no latency bucket {L!r} (has {sorted(lat)[:4]})")
            continue
        cell = lat[L]
        if not isinstance(cell, dict):
            bad.append(f"{ident}: latency[{L}] is {type(cell).__name__}")
            continue
        for f in VALUATION_SUBFIELDS:
            if f not in cell:
                bad.append(f"{ident}: latency[{L}] has no {f!r}")
                continue
            v = cell[f]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                bad.append(f"{ident}: {f} is {type(v).__name__} {v!r}, not a "
                           f"number (a bool would arithmetic as 1/0)")
            elif not math.isfinite(v):
                bad.append(f"{ident}: {f} is {v!r}; a non-finite valuation "
                           f"input compares False against every threshold")
            elif f.endswith("shares") and v < 0:
                bad.append(f"{ident}: {f} is negative ({v!r})")
        if len(bad) > 40:
            break
    if bad:
        raise DiagnosticRefused(
            f"REFUSED: {len(bad)} malformed valuation input(s) among "
            f"{len(kept):,} kept rows. A malformed input reconstructs to the "
            f"SAME gate=False a genuine no-fill produces, so it would be scored "
            f"as an outcome rather than refused as an absence. First: "
            f"{bad[:4]}")
    return {"rows_validated": len(kept), "latency_bucket": L,
            "subfields_checked": list(VALUATION_SUBFIELDS),
            "enforced_never_repaired": True}
# The evaluator's OWN mode constants, transcribed. Comparing against a string
# that does not exist refuses everything, which looks like strictness.
CAUSAL_MODE = "CAUSAL_FROZEN_FROM_TRAIN"
RETRO_MODE = "RETROSPECTIVE_TOPK"


def rejoin_source_fields(kept: list, exposure_path: Path) -> dict:
    """FD1 (R-315): restore the fields _feature_pass drops.

    _feature_pass returns a PROJECTION --
      (slug, day, t0, t_start, side, gen, latency, coin)
    -- which drops BOTH `status` and `any_fill_ahead`. The first is LOUD: the
    status assertion refuses and the run stops. The second is SILENT and far
    worse: harmful_action_eval's val() reads

        r.get("any_fill_ahead") and "latency" in r

    so WITHOUT that key every valuation returns 0.0. Measured on a projected
    row: net_cents 0.0 against 123.0 for the same row carrying the gate. A run
    in that state completes, writes a receipt, and reports NET ZERO AT EVERY
    BUDGET -- which reads as "the candidate captured nothing", a clean negative
    result born from a dropped dictionary key.

    The loud half is what saved the run. This function exists so the quiet half
    cannot come back.

    HARNESS-SIDE RE-JOIN by row identity from our OWN exposure file: no identity
    file moves and no rebind is needed. Soundness was MEASURED before this was
    written -- 472,413 OK rows, all identity keys distinct, none missing either
    field; only the excluded statuses (GAP_IN_HORIZON, TRUNCATED_HORIZON) lack
    the gate, and those never reach the join because _feature_pass keeps
    status=='OK' only.
    """
    import harmful_hazard_model as _hm
    # F3: VALIDATE BEFORE ANY RECONSTRUCTION. The validator was wired AFTER the
    # keptrow loop it exists to protect, so a malformed scalar latency cell
    # raised a raw AttributeError ('int' object has no attribute 'get') before
    # the controlled refusal could fire. Rule 17 verbatim in my own code: ten
    # green falsifiers, every one calling the validator DIRECTLY, so not one of
    # them could see the wiring order. A check placed after the thing it guards
    # is not a check.
    valinputs = assert_valuation_inputs(kept, D.TARGET_LATENCY_MS)
    src: dict = {}
    for r in PA._stream_tape_rows(Path(exposure_path)):
        if r.get("status") != "OK":
            continue
        k = (r.get("slug"), r.get("side"), r.get("gen"), r.get("t_start"))
        if k in src:
            raise DiagnosticRefused(
                f"REFUSED: exposure row identity {k} occurs more than once, so "
                f"a re-join by identity is AMBIGUOUS. Attaching either one "
                f"would be a coin flip recorded as a measurement.")
        src[k] = r.get("status")
    missed = []
    for row in kept:
        k = (row.get("slug"), row.get("side"), row.get("gen"),
             row.get("t_start"))
        if k not in src:
            missed.append(k)
            continue
        row["status"] = src[k]
        # THE GATE IS RECONSTRUCTED, NOT JOINED. hm.keptrow is the CANONICAL
        # composition -- the same one stage_score runs, which is why the
        # committed receipts were never affected by this defect. Its own comment
        # records the prior finding: "ONE definition... this expression and the
        # builder's bool(fut) were two rules for the same valuation gate."
        # Joining the raw field would have made a THIRD rule. `latency` IS in
        # the projection, so the canonical derivation is available here.
        row[VALUATION_GATE] = _hm.keptrow(row)[VALUATION_GATE]
    seen_keys: dict = {}
    for row in kept:
        k = (row.get("slug"), row.get("side"), row.get("gen"),
             row.get("t_start"))
        seen_keys[k] = seen_keys.get(k, 0) + 1
    dup_kept = sorted(k for k, c in seen_keys.items() if c > 1)
    if dup_kept:
        raise DiagnosticRefused(
            f"REFUSED: {len(dup_kept)} kept row identity/ies occur MORE THAN "
            f"ONCE (e.g. {dup_kept[:3]}). Each kept row must match EXACTLY ONE "
            f"source row; a duplicated identity means one outcome is attached "
            f"to several rows (rule 2).")
    if missed:
        raise DiagnosticRefused(
            f"REFUSED: {len(missed)} kept row(s) found NO source row in "
            f"{Path(exposure_path).name} (e.g. {missed[:3]}). Every kept row "
            f"must re-join to exactly one source row; a kept row with no source "
            f"cannot be valued and must not be silently valued at zero.")
    # THE SILENT-ZERO KNOWN-BAD, as a live assertion rather than a test-only one.
    absent = sum(1 for r in kept if VALUATION_GATE not in r)
    if absent:
        raise DiagnosticRefused(
            f"REFUSED: {absent} kept row(s) carry no {VALUATION_GATE!r} after "
            f"the re-join. val() returns 0.0 without it, so the run would "
            f"report NET ZERO at every budget and that zero would read as a "
            f"measured negative rather than as an absent field.")
    n_true = sum(1 for r in kept if r.get(VALUATION_GATE))
    if n_true == 0:
        raise DiagnosticRefused(
            f"REFUSED: the valuation gate is False on ALL {len(kept):,} kept "
            f"rows, so every cent would be 0.0 by construction. A uniformly "
            f"zero receipt is indistinguishable from a broken join and must not "
            f"be published as a negative result.")
    return {"rejoined": len(kept), "source_rows_indexed": len(src),
            "valuation_inputs": valinputs,
            "fields_restored": ["status (re-joined by identity)",
                                VALUATION_GATE + " (RECONSTRUCTED via "
                                "hm.keptrow — the canonical composition "
                                "stage_score uses, not a second rule)"],
            "valuation_gate_true": n_true,
            "valuation_gate_false": len(kept) - n_true,
            "why": ("_feature_pass returns a projection that drops both; "
                    "without the gate every valuation is 0.0 and the receipt "
                    "reads as a clean negative")}


def synthetic_population(n_actions: int = 240, seed: int = 20260829) -> tuple:
    """A block shaped exactly like _feature_pass's output, no real data.

    R-314(3): nothing called score_stage, so its whole downstream — scoring, the
    incumbent, the economics, the cells, the receipt — had never executed. A
    suite can be green over a path that cannot run. This exercises that path on
    substituted inputs, and the receipt it produces is stamped so it can never
    be mistaken for a measurement.

    Values are built so the cells are NON-ZERO: a positive control that reached
    'receipt cells' with every cell zero would pass while proving nothing —
    which is exactly the silent-zero failure it exists to detect.
    """
    import random as _r
    rr = _r.Random(seed)
    PM, FN, ST, kept = [], [], [], []
    L = str(D.TARGET_LATENCY_MS)
    for i in range(n_actions):
        PM.append([rr.gauss(0, 1) for _ in range(45)])
        FN.append([rr.gauss(0, 1) for _ in range(15)])
        ST.append([rr.gauss(0, 1) for _ in range(45)])
        v = (60.0 if i % 3 == 0 else -25.0)          # both signs, NON-ZERO
        kept.append({
            "slug": f"btc-updown-5m-{1787897400 + 300 * (i // 12)}",
            "side": "BUY_UP" if i % 2 else "SELL_UP", "gen": i,
            "t_start": float(i % 12) - 6.0, "t0": 1787897400 + 300 * (i // 12),
            "day": "2026-08-28", "coin": COIN, "status": "OK",
            VALUATION_GATE: True,
            "latency": {L: {"preventable_value_cents": v,
                            "preventable_shares": 1.0, "stale_shares": 0.0}}})
    return {"PM": PM, "FN": FN, "ST": ST, "kept": kept,
            "drops": {"state_join_failed": 0}}, len(kept)


def score_stage(tape_path: Path, exposure_path: Path = None,
                latency_ms: int = None, synthetic: bool = False) -> dict:
    """The diagnostic read. Gated on DA's verdict; both comparators on the
    IDENTICAL rows; both held to their OWN frozen causal thresholds."""
    import phase2_iter011_run as R11        # reuse, never re-implement
    import phase2_declaration as _D
    exposure_path = Path(exposure_path or ROWS_OUT)
    da = da_bounds()
    gate = ({"SYNTHETIC": "no gate consumed; no real tape read"} if synthetic
            else load_gate_verdict(Path(tape_path), exposure_path))
    model = load_frozen_candidate()
    L = int(_D.TARGET_LATENCY_MS if latency_ms is None else latency_ms)

    # FD: split is EXPLICIT. The fragments were built into the SCORE split
    # (R-303); indexing 'train' would silently return an empty index.
    if synthetic:
        blk, _n_syn = synthetic_population()
        block_all = {COIN: blk}
        exp_meta = {"rows_by_status": {"OK": _n_syn}}
    else:
        TAPE = _index_tape(Path(tape_path), split="score")
        block_all = PA._feature_pass(exposure_path, "be_fragment", TAPE=TAPE)
    if COIN not in block_all:
        raise DiagnosticRefused(
            f"REFUSED: the feature pass produced no {COIN} block; every row was "
            f"excluded, which reads as a null result and is an absent one.")
    blk = block_all[COIN]
    kept = blk["kept"]
    # FD1: restore what the projection dropped BEFORE asserting on it.
    rejoin = ({"SYNTHETIC": "fields present by construction"} if synthetic
              else rejoin_source_fields(kept, exposure_path))
    assert_field_readable(kept, "status", str, "post-feature-pass rows")
    assert_field_readable(kept, VALUATION_GATE, bool,
                          "the VALUATION GATE — without it every cent is 0.0")
    assert_field_readable(kept, "t0", int, "ABSOLUTE clock component")
    assert_field_readable(kept, "t_start", float, "window-relative component")

    if not synthetic:
        exp_head = exposure_path.open().read(1 << 16)
        exp_meta = json.loads(
            exp_head[:exp_head.index('"rows"')].rstrip().rstrip(",") + "}")
    recon = reconcile_population(
        kept, blk["drops"], exp_meta["rows_by_status"].get("OK", -1))
    # FD3: the boolean was COMPUTED AND NEVER READ. A reconciliation that
    # reports `reconciles: false` in a receipt nobody checks is a decoration.
    if not recon["reconciles"]:
        raise DiagnosticRefused(
            f"REFUSED: the population does not reconcile — kept "
            f"{recon['rows_kept']:,} + drops "
            f"{sum(recon['drops_by_named_status'].values()):,} = "
            f"{recon['rows_accounted']:,}, but the exposure file declares "
            f"{recon['rows_expected_from_exposure_OK']:,} OK rows. Rows that "
            f"are neither scored nor named are unaccounted for, and a total "
            f"that does not add up cannot be a population statement.")

    # CANONICAL ORDER (R-306(2)). evaluate_policy breaks gmax TIES by ARRIVAL
    # ORDER, so the decision metric moves with row order -- measured, a 110c
    # swing on a 12-generation fixture. Sorting here makes THIS run
    # reproducible. It does NOT make the function order-independent, and the
    # receipt says so rather than letting one be read as the other.
    order = sorted(range(len(kept)),
                   key=lambda i: (kept[i].get("slug"), kept[i].get("side"),
                                  kept[i].get("gen"), kept[i].get("t_start"), i))
    kept = [kept[i] for i in order]
    for fam in ("PM", "FN", "ST"):
        blk[fam] = [blk[fam][i] for i in order]
    idx = range(len(kept))

    cand = score_rows(model, blk, idx)
    inc_model = R11.load_verified_incumbent(COIN)
    inc = R11.apply_incumbent(inc_model, blk, idx)["expected_cancel_value"]
    if not (len(cand) == len(inc) == len(kept)):
        raise DiagnosticRefused(
            f"REFUSED: {len(cand)} candidate / {len(inc)} incumbent scores "
            f"against {len(kept)} rows; the comparison is defined only on the "
            f"IDENTICAL population.")

    # FD-R3: BOTH ARMS ON THEIR OWN FROZEN THRESHOLDS. The incumbent was
    # previously evaluated with NO theta_frozen, so it ran RETROSPECTIVE_TOPK --
    # a cutoff resolved from the scored data, i.e. a look-ahead the candidate
    # was not given. That is not a fair comparison; it is a handicap match whose
    # numbers look entirely normal.
    # R-318(2): a DEAD ARM produces cells too. A constant or non-finite score
    # vector ranks nothing, yet threshold_mode still reports the frozen mode and
    # the increment is still a number. Liveness is asserted on the VECTORS.
    def _alive(v, nm):
        fin = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
        return {"n": len(v), "all_finite": len(fin) == len(v),
                "distinct": len(set(fin)), "arm": nm}
    alive = {"candidate": _alive(cand, "candidate"),
             "incumbent": _alive(inc, INCUMBENT_ARM if False else "incumbent")}
    for nm, a in alive.items():
        if not a["all_finite"]:
            raise DiagnosticRefused(
                f"REFUSED: the {nm} score vector contains non-finite values; "
                f"they compare False against every threshold and the arm would "
                f"silently rank nothing.")
        if a["distinct"] <= 1:
            raise DiagnosticRefused(
                f"REFUSED: the {nm} score vector is CONSTANT "
                f"({a['distinct']} distinct value over {a['n']:,} rows). A "
                f"constant score ranks nothing, so its cells would be an "
                f"artifact of tie-breaking rather than a measurement.")
    thetas_c = model["causal_thresholds"]
    thetas_i = inc_model.get("causal_thresholds")
    if not thetas_i:
        raise DiagnosticRefused(
            "REFUSED: the incumbent artifact carries no causal_thresholds, so "
            "it could only be evaluated retrospectively. A look-ahead cutoff "
            "for one arm and a frozen one for the other is not a comparison.")
    budgets = [float(b.rstrip("%")) / 100.0 for b in sorted(thetas_c)]
    ev_c = HAE.evaluate_policy(kept, cand, latency_ms=L, budgets=budgets,
                               theta_frozen=thetas_c)
    ev_i = HAE.evaluate_policy(kept, inc, latency_ms=L, budgets=budgets,
                               theta_frozen=thetas_i)
    for nm, ev in (("candidate", ev_c), ("incumbent", ev_i)):
        for b, d in ev["budgets"].items():
            # The evaluator's own constants are CAUSAL_FROZEN_FROM_TRAIN and
            # RETROSPECTIVE_TOPK. My first version compared against a string
            # that does not exist ("CAUSAL_FROZEN"), so it would have REFUSED
            # THE REAL RUN — found by the end-to-end synthetic on its first
            # execution, which is the entire reason that control exists. The
            # check now names both real constants: the frozen mode is required
            # and the retrospective one is refused explicitly.
            _mode = d.get("threshold_mode")
            if _mode != CAUSAL_MODE:
                raise DiagnosticRefused(
                    f"REFUSED: {nm} budget {b} ran threshold_mode {_mode!r}, "
                    f"not {CAUSAL_MODE!r}"
                    + (f" — {RETRO_MODE!r} resolves the cutoff FROM THE SCORED "
                       f"DATA, which is a look-ahead granted to one arm."
                       if _mode == RETRO_MODE else
                       f". An unrecognised mode is not evidence of a frozen "
                       f"threshold."))

    # TIE COUNT AT THE BUDGET BOUNDARY, measured from the real score vector.
    gens: dict = {}
    for i, r in enumerate(kept):
        gens.setdefault((r.get("slug"), r.get("side"), r.get("gen")),
                        []).append(i)
    gmax = sorted((max(cand[i] for i in ix) for ix in gens.values()),
                  reverse=True)
    ties = {}
    for b in budgets:
        k = max(1, int(len(gmax) * b))
        ties[f"{int(b*100)}%"] = {
            "k": k,
            "tie_at_boundary": (k < len(gmax) and gmax[k - 1] == gmax[k]),
            "n_equal_to_boundary": sum(1 for g in gmax if g == gmax[k - 1])}

    cells = {}
    for b in sorted(ev_c["budgets"]):                 # FD: ev['budgets'], not top
        c, i = ev_c["budgets"][b], ev_i["budgets"].get(b, {})
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
            "threshold_mode_candidate": c.get("threshold_mode"),
            "threshold_mode_incumbent": i.get("threshold_mode"),
            "gmax_tie_at_budget_boundary": ties.get(b)}
    return {
        "artifact": "be_fragment_diagnostic_v1",
        "status": ("SYNTHETIC_SELFTEST_NOT_A_RESULT" if synthetic
                   else "DIAGNOSTIC_NEVER_EVIDENCE"),
        "provenance": {"synthetic": bool(synthetic),
                       "real_data_read": not synthetic,
                       "why": ("a synthetic run must be unable to masquerade as "
                               "a measurement, so it says so in the field a "
                               "reader looks at first")},
        "this_is_a_MODEL_DIAGNOSTIC_not_strategy_performance": (
            "These cents are a MODEL DIAGNOSTIC on a censored, inadmissible "
            "fragment. They are NOT strategy performance, not a P&L, and not a "
            "forward result. Reading them as what the policy 'would have made' "
            "is the single most likely misreading of this artifact."),
        "what_this_cannot_do": (
            "R-293 pre-registered the readings BEFORE any number existed: a "
            "POSITIVE result is WEAK COMFORT ONLY because the censoring "
            "plausibly flatters; a NEGATIVE result is AMBIGUOUS because "
            "censoring artifacts are indistinguishable from real failure at "
            "this coverage, and it specifically must NOT trigger a candidate "
            "change, which would be selection on a contaminated read. Under "
            "EVERY outcome the race admission rule, the frozen candidate and "
            "multiplicity (1) are untouched."),
        "inadmissibility_is_unconditional": da["inadmissibility_reasons"],
        "censoring_statement": da["censoring_statement"],
        "censoring_measured": da.get("censoring_measured_not_asserted"),
        "da_receipt": {"file": DA_RECEIPT.name,
                       "sha256_prefix": _sha16(DA_RECEIPT),
                       "declared_cutoff_epoch": da["declared_cutoff_epoch"],
                       "declared_cutoff_utc": da["declared_cutoff_utc"]},
        "gate": gate,
        "candidate": model["verified"],
        "incumbent": inc_model.get("_verified"),
        "population": recon,
        "rejoin": rejoin,
        "score_vectors_alive": alive,
        "clock_basis_consumed": {
            "tape_declares": gate.get("tape_header_clock_basis"),
            "strata_hour": "ABSOLUTE — harmful_action_eval._hour reads "
                           "(t0 + t_start), verified by measurement",
            "within_generation_order": "t_start, window-relative, valid because "
                                       "a generation never spans a window",
            "canonical_sort_key": "(slug, side, gen, t_start, index)"},
        "determinism": {
            "canonically_sorted": True,
            "deterministic": True,
            "order_independent": False,
            "why": "evaluate_policy breaks gmax TIES by arrival order (measured: "
                   "a 110c swing across row-order shuffles on an all-tied "
                   "fixture). Canonical sorting makes THIS run reproducible; it "
                   "does NOT make the function order-independent. The "
                   "boundary-tie count above says whether the sensitivity could "
                   "have bitten this population at all.",
            "underlying_fix_blocked_by": "harmful_action_eval is in "
                                         "CODE_IDENTITY_FILES"},
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
    _N_PINNED = len(feats)
    _PINNED_SET = set(feats)
    idx = {}
    for r in PA._stream_tape_rows(Path(path)):
        if r.get("split") != split:
            continue
        _st = str(r["state_status"])
        state = r.get("state") or {}
        # R-313: DECLARED-FIELD COMPLETENESS, checked HERE and not delegated.
        #
        # A RAGGED ROW DOES NOT DEGRADE THE SCORE, IT LIES TO IT. encode_row
        # appends 0.0 for any absent field -- and the GUARD FLAG for that field
        # is itself absent, so it also encodes 0.0, meaning NOT MISSING. The
        # model is then told the value is genuinely zero AND present, which is
        # the exact distinction the guard pair exists to preserve. encode_row's
        # docstring says such a row "is a schema break and it raises"; the code
        # does not raise (filed to the identity queue -- the fix moves
        # fit_code_sha256_prefix).
        #
        # This is the SECOND independent line on the point DA's gate is being
        # repaired for. It is cheap, it is local, and it means the diagnostic
        # stops trusting a certification for something it reads anyway. Measured
        # on the real tape: every row of EVERY status declares all 45.
        # EXACT SET EQUALITY, not a count. The first version of this check
        # compared len(state) against the pinned count, and a COUNT-PRESERVING
        # SUBSTITUTION walked straight through it: drop bn_feed_age_s AND
        # bn_feed_missing, add two unknown keys, total still 45 -- ACCEPTED,
        # with the value encoding 0.0 and its guard ALSO 0.0, i.e. "genuinely
        # zero and present". That is precisely the anti-safe case this check
        # exists to stop, reached through the gap between COUNT and IDENTITY.
        # My three original falsifiers (1/45, 44/45, 46/45) all moved the COUNT,
        # so none of them ever exercised identity: the green was real and the
        # claim was wider than the green.
        # FD5: the state_status must be DECLARED, never defaulted. Reading a
        # missing status as "OK" invents a population statement.
        if "state_status" not in r:
            raise DiagnosticRefused(
                f"REFUSED: tape row "
                f"{(r.get('slug'), r.get('side'), r.get('gen'))} declares no "
                f"state_status. Defaulting an absent status to OK invents a "
                f"population statement the tape never made.")
        _idk = (r["slug"], r["side"], r["gen"], r["t_start"])
        if _idk in idx:
            raise DiagnosticRefused(
                f"REFUSED: tape row identity {_idk} occurs MORE THAN ONCE. A "
                f"duplicated key silently overwrites the earlier row, so the "
                f"index would be smaller than the tape and nothing would say "
                f"so.")
        _have = set(state)
        if _have != _PINNED_SET:
            _missing = sorted(_PINNED_SET - _have)
            _extra = sorted(_have - _PINNED_SET)
            raise DiagnosticRefused(
                f"REFUSED: tape row "
                f"{(r.get('slug'), r.get('side'), r.get('gen'), r.get('t_start'))} "
                f"declares a state field set that is not the pinned schema. "
                f"MISSING {_missing}; EXTRA {_extra}. A missing required field "
                f"encodes as 0.0 and its guard, if also missing, encodes 0.0 "
                f"meaning NOT MISSING -- the model then reads it as genuinely "
                f"zero and present. Counting the fields is not checking them.")
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
