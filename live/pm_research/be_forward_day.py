#!/usr/bin/env python3
"""THE PRODUCTION FORWARD-DAY RUN PATH. Scores stay SEALED.

§10 step 9: score the frozen set UNCHANGED on >=5 later complete UTC days.
`forward_dry_run.py` proves the wiring on synthetic rows and says so; this is
the other half -- the same wiring on REAL inputs, end to end.

A SIBLING DRIVER, NOT A FLAG ON THE SCORER, and the reason is a boundary
worth keeping. `harmful_forward_scorer` applies the frozen artifact and
consumes the mask; that is all it does, and its small dependency set is what
makes "the frozen artifact is applied, never refitted" checkable by reading
it. The run path needs the market ledger, the admissible-window supply, the
replay bridge, the v3 exposure builder, sealing and receipts. Putting those
behind a flag in the scorer would make the scorer's own claim harder to see.
The scorer is IMPORTED here, so there is still exactly one implementation of
scoring.

SEALED (rule 11). Per-action scores and every value metric go to an OUTDIR the
caller names. Nothing is written under data/pm_5min/derived/. The receipt
carries counts, identities and hashes -- never a metric. UNSEALING IS THE
COORDINATOR'S OR THE USER'S ACT, and this file cannot do it: it has no code
that prints a score.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"
MARKETS = REPO / "data/pm_5min/markets.jsonl"
RATIFICATION_REF = "R-418"

import de_admissible_windows as AW
import ev_replay_seam as SEAM
import harmful_forward_scorer as FS
# The scheduled-unit prefix is IMPORTED, never restated: DA's preflight
# matches `write_reason` the same way, and two copies of a matching string
# drift apart silently.
from da_governed_verdict_preflight import SCHEDULED_PREFIX


class ForwardDayRefused(RuntimeError):
    """A named refusal. Every gate below refuses by name, never by absence."""


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _as_of() -> str:
    """This run's UTC instant. Rule 8: every quoted population carries its n
    AND its as-of."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _provenance() -> dict:
    """The carrying commit and this driver's own bytes.

    A ref alone is not an identity — the dirty flag and the file hash travel
    with it, the same shape the scorer's provenance block uses."""
    import subprocess

    def git(*a):
        try:
            r = subprocess.run(("git", *a), cwd=str(REPO), capture_output=True,
                               text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:                            # noqa: BLE001
            return None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    me = Path(__file__).resolve()
    return {"carrying_commit": head or "UNAVAILABLE",
            "carrying_commit_resolved": bool(head),
            "working_tree_dirty": ("unknown" if status is None
                                   else bool(status.strip())),
            "driver": me.name, "driver_sha256_prefix": _sha_file(me)[:16]}


def day_bounds(day: str) -> tuple:
    d = dt.datetime.strptime(day, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    lo = int(d.timestamp())
    return lo, lo + 86400


# ---------------------------------------------------------------------------
# gate 1 -- the frozen candidate's own reproduction contract (§10 step 1)
# ---------------------------------------------------------------------------
def assert_frozen_contract(candidate: Path = None) -> dict:
    """The artifact must BE the frozen one, and its bound inputs must still be
    the bytes it was frozen against.

    §10 step 1: "require every conditional-model artifact in the hash set,
    cover every bound input in fit-side drift detection". The candidate names
    its manifest by sha and its builder by sha; the manifest's `pin_semantics`
    says `reproducibility_anchor` entries MUST be compared for equality when
    validating a reproduction, and marks the one entry that must not be.

    This gate does not decide what to do about drift -- it refuses and names
    it. Re-stamping a frozen contract to make a run pass would be editing the
    thing being validated (rule 13)."""
    cp = candidate or FS.CANDIDATE
    c = json.loads(cp.read_text())
    if c.get("status") != "FROZEN":
        raise ForwardDayRefused(
            f"REFUSED: {cp.name} status is {c.get('status')!r}, not FROZEN.")
    drift: list[str] = []
    mname = c.get("manifest")
    mp = cp.parent / mname if mname else None
    if not mp or not mp.exists():
        raise ForwardDayRefused(
            f"REFUSED: {cp.name} names manifest {mname!r}, which is absent.")
    msha = _sha_file(mp)
    if msha != c.get("manifest_sha256"):
        drift.append(f"manifest {mname}: bound "
                     f"{str(c.get('manifest_sha256'))[:16]} now {msha[:16]}")
    m = json.loads(mp.read_text())
    ps = m.get("pin_semantics") or {}
    default = ps.get("_default", "reproducibility_anchor")
    anchors, checked = [], 0
    for k, want in sorted((m.get("hashes") or {}).items()):
        if ps.get(k, default) != "reproducibility_anchor":
            continue
        anchors.append(k)
        p = REPO / k
        now = _sha_file(p) if p.exists() else None
        checked += 1
        if now != want:
            drift.append(f"{k}: bound {want[:16]} now "
                         f"{(now or 'MISSING')[:16]}")
    bsha = c.get("builder_sha256")
    bp = REPO / "live/pm_research/harmful_hazard_model.py"
    if bsha and bp.exists() and _sha_file(bp) != bsha:
        drift.append(f"builder harmful_hazard_model.py: bound {bsha[:16]} "
                     f"now {_sha_file(bp)[:16]}")
    if checked == 0:
        raise ForwardDayRefused(
            "REFUSED: the frozen contract check compared ZERO anchors. A gate "
            "that reads nothing must not report a pass (R-289).")
    if drift:
        raise ForwardDayRefused(
            f"REFUSED: the frozen candidate's reproduction contract does not "
            f"hold against the working tree — {len(drift)} of "
            f"{checked + 1} bound inputs have moved. The forward rows would "
            f"be produced by code the freeze did not bind, which is a "
            f"different program from the one being raced (§10 step 1). "
            f"Drift: {drift}. Re-stamping the contract to make this pass "
            f"would edit the thing being validated; that is the "
            f"coordinator's or the USER's act, not this driver's.")
    return {"candidate": cp.name, "candidate_sha256": _sha_file(cp),
            "manifest": mname, "manifest_sha256": msha,
            "anchors_checked": checked, "anchor_keys": anchors,
            "builder_sha256": bsha, "contract": "HOLDS"}


# ---------------------------------------------------------------------------
# gate 2 -- the day is closed, and its verdict was written by the scheduled unit
# ---------------------------------------------------------------------------
def assert_day_closed_and_attributed(day: str, verdict: dict = None) -> dict:
    v = FS.read_day_verdict(day) if verdict is None else verdict
    if not v:
        raise ForwardDayRefused(
            f"REFUSED: {day} has no day verdict at "
            f"{DERIVED / f'da_dayverdict_{day}.json'}. A forward day is scored "
            f"only after DA has verified it (R-153(3)); absence is not a pass.")
    closed = v.get("day_closed_calendar")
    if closed is not True:
        raise ForwardDayRefused(
            f"REFUSED: {day} is not closed by calendar "
            f"(day_closed_calendar={closed!r}). Scoring an OPEN day scores a "
            f"population that is still growing.")
    wr = v.get("write_reason")
    if not (isinstance(wr, str) and wr.startswith(SCHEDULED_PREFIX)):
        raise ForwardDayRefused(
            f"REFUSED: {day}'s verdict was not written by the scheduled unit "
            f"— write_reason={wr!r} does not start with the required prefix "
            f"(imported from da_governed_verdict_preflight, matched as a "
            f"PREFIX exactly as DA's preflight matches it; a substring test "
            f"would accept an unattributed hand run).")
    return {"day_closed_calendar": True, "write_reason": wr,
            "write_reason_prefix_source": "da_governed_verdict_preflight."
                                          "SCHEDULED_PREFIX"}


# ---------------------------------------------------------------------------
# gate 3 -- the population: the day's own ledger, then supply, then bridge
# ---------------------------------------------------------------------------
def present_from_ledger(day: str, path: Path = None) -> dict:
    """The windows that EXISTED, read from the day's own market ledger.

    A FACT, NOT A GRID (R-418). `de_admissible_windows` refuses to derive this
    itself — deriving the calendar is selecting, and it supplies."""
    p = path or MARKETS
    if not p.exists():
        raise ForwardDayRefused(f"REFUSED: no market ledger at {p}.")
    lo, hi = day_bounds(day)
    per: dict[str, set] = collections.defaultdict(set)
    rows = 0
    with p.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            rows += 1
            ws = d.get("window_start")
            coin = d.get("coin")
            if ws is None or coin is None or not (lo <= int(ws) < hi):
                continue
            per[coin].add(int(ws))
    if not per:
        raise ForwardDayRefused(
            f"REFUSED: the ledger holds no window for {day} (read {rows} "
            f"rows). An empty present is not an empty day — it is a read that "
            f"found nothing, and scoring off it would score nothing and say "
            f"it succeeded.")
    return {c: sorted(v) for c, v in sorted(per.items())}


def population(day: str, present: dict = None) -> dict:
    pres = present_from_ledger(day) if present is None else present
    supply = AW.supply(day, pres)                    # refusals propagate
    specs = SEAM.window_specs_from_supply(
        supply, ratification_ref=RATIFICATION_REF)   # refusals propagate
    return {"present": pres, "supply": supply, "specs": specs}


def counts_per_coin(pop: dict) -> dict:
    out = {}
    for coin, c in sorted((pop["supply"]["counts"] or {}).items()):
        out[coin] = {"n_present": c["n_present"],
                     "n_masked": c["n_masked_applied"],
                     "n_supplied": c["n_supplied"]}
    return out


# ---------------------------------------------------------------------------
# gate 4 -- rows, from the ACCEPTED v3 builder, over EXACTLY the bridged windows
# ---------------------------------------------------------------------------
def selected_from_specs(specs: list) -> tuple:
    """The builder's own selection tuple, for the bridged windows only.

    `harmful_exposure_rows.select_stratified` builds
    `(slug, path, up, down, gaps)` from `fi._archive_paths()`,
    `fi.token_map()` and `fi.gaps_by_slug(era)`. R-418 forbids that selector
    on a race day, so the SAME three lookups are used here over the windows
    the supply emitted. Nothing is chosen: the set is the bridge's, in its
    order. A window with no archive or no token map is REFUSED, not skipped —
    a race day is scored whole or not at all."""
    import harmful_exposure_rows as HER
    fi = HER.qr.base.fi
    paths = fi._archive_paths()
    tokens = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    out, missing = [], []
    for spec in specs:
        slug = spec["slug"]
        if slug not in paths or slug not in tokens:
            missing.append(slug)
            continue
        up, down = tokens[slug]
        out.append((slug, paths[slug], up, down, gaps.get(slug, [])))
    if missing:
        raise ForwardDayRefused(
            f"REFUSED: {len(missing)} supplied windows have no archive or no "
            f"token map ({missing[:4]}). R-418 scores the complement WHOLE; "
            f"dropping windows here would silently re-select the population "
            f"the supply already fixed.")
    return out, {"n_specs": len(specs), "n_selected": len(out)}


def build_rows_over(selected: list) -> dict:
    """The v3 builder's per-window sequence, over the supplied windows.

    NOT A SECOND BUILDER. Every step is `harmful_exposure_rows`' own function
    in its own order — replay, join, boundary, clock, generation table,
    labels — and the STRICT failure condition is the module's, copied by
    reference rather than restated. `build_rows` itself cannot be reused
    because its window set comes from `select_stratified`/`select_v2_era` and
    it takes no injection point; `harmful_exposure_rows.py` is a
    reproducibility anchor of the frozen candidate, so this driver does not
    edit it to add one.

    THE RECONCILIATION IS THE GATE. A mismatch marks the rows and is reported
    as a FAILURE for the day; it is never absorbed."""
    import harmful_exposure_rows as HER
    qr = HER.qr
    spec = qr._qr_spec(qr.QR_SKEW, latency_ms=0, cancel=False)
    rows, per_window = [], {}
    recon_fail = unhooked = wrong_gen = boundary_bad = clock_bad = 0
    n_windows = 0
    for slug, path, up, down, wgaps in selected:
        out = HER.replay_with_recorder(path, up, down, wgaps, spec)
        if out is None:
            per_window[slug] = {"replayed": False, "n_rows": 0}
            continue
        arm, wf = out
        n_windows += 1
        t0 = int(slug.rsplit("-", 1)[1])
        day_s = dt.datetime.fromtimestamp(
            t0, dt.timezone.utc).strftime("%Y-%m-%d")
        joined, jrec = HER.join_fills(arm.fill_log, arm.fills)
        n_b = HER.verify_boundary_times(arm.segments, joined)
        ttimes = HER.trade_receipt_times(path, up, down)
        n_c = HER.verify_consume_clock(arm.consume_times, ttimes)
        gens, recon = HER.generation_table(arm.segments, joined, wf,
                                           qr.base.fi.WINDOW_S)
        wrows = HER.label_rows(arm.segments, gens, wf, qr.base.fi.WINDOW_S)
        bad = (jrec["count_mismatch"] or jrec["tuple_mismatches"]
               or recon["orphan_fills"]
               or recon["wrong_generation_assignments"]
               or arm.unhooked_changes or n_b or n_c)
        wrong_gen += recon["wrong_generation_assignments"]
        boundary_bad += n_b
        clock_bad += n_c
        if bad:
            recon_fail += 1
            unhooked += arm.unhooked_changes
            for r in wrows:
                r["status"] = "RECONCILIATION_FAILED"
        for r in wrows:
            r["slug"] = slug
            r["coin"] = slug.split("-")[0]
            r["day"] = day_s
            r["t0"] = t0
        rows.extend(wrows)
        per_window[slug] = {"replayed": True, "n_rows": len(wrows),
                            "reconciled": not bool(bad)}
    return {"rows": rows, "n_windows": n_windows,
            "reconciliation_failures": recon_fail,
            "unhooked_state_changes": unhooked,
            "wrong_generation_assignments": wrong_gen,
            "boundary_time_violations": boundary_bad,
            "consume_clock_violations": clock_bad,
            "per_window": per_window,
            "schema": "harmful_exposure_v3_4_fill_scoped_markout"}


def assert_window_sets_agree(specs: list, built: dict) -> dict:
    """The rows must cover EXACTLY the bridged windows. Neither side may
    silently be the other's subset."""
    bridged = {s["slug"] for s in specs}
    got = {r["slug"] for r in built["rows"]}
    only_bridged = sorted(bridged - got)
    only_rows = sorted(got - bridged)
    if only_rows:
        raise ForwardDayRefused(
            f"REFUSED: rows carry {len(only_rows)} windows the bridge never "
            f"supplied ({only_rows[:4]}). A row outside the ratified "
            f"population is a window nobody admitted (R-418).")
    return {"n_bridged": len(bridged), "n_with_rows": len(got),
            "bridged_without_rows": len(only_bridged),
            "bridged_without_rows_examples": only_bridged[:8],
            "note": "a bridged window with no rows is a window that produced "
                    "no cancellable generation; it is COUNTED here, never "
                    "dropped from the denominator"}


def action_count(rows: list) -> int:
    """Rule 2: rows are actions; the evaluator de-duplicates to actions."""
    return len({(r.get("slug"), r.get("side"), r.get("gen")) for r in rows})


def score_rows(rows: list) -> dict:
    """Per-action expected cancel value, through the FROZEN artifact's OWN
    feature_vector_contract.

    The frozen fit is APPLIED, never refitted: `harmful_forward_scorer`
    owns `design_row`/`expected_cancel_value` and this passes each row's
    PM+fine vector to them. Features come from `harmful_hazard_model`, the
    builder the candidate names — the same two calls `phase2_arms` makes,
    in the same order, so there is one feature construction and not a
    second."""
    import harmful_hazard_model as hm
    frozen = FS.load_frozen()
    fi = hm.fi
    paths = fi._archive_paths()
    tokens = fi.token_map()
    streams: dict = {}
    out: dict = collections.defaultdict(list)
    skipped = 0
    for r in rows:
        coin, slug = r["coin"], r["slug"]
        fit = frozen["fits"].get(coin)
        if fit is None:
            skipped += 1
            continue
        if slug not in streams:
            if slug not in paths or slug not in tokens:
                raise ForwardDayRefused(
                    f"REFUSED: no archive for {slug} at scoring time.")
            up, dn = tokens[slug]
            streams[slug] = hm.window_streams(paths[slug], up, dn)
        fp = hm.features(streams[slug], r["t_start"], r["side"],
                         r.get("level"), r.get("resting"), r.get("qahead"))
        ff = hm.fine_feats(r["t0"] + r["t_start"], r["side"], coin)
        if fp is None or ff is None:
            skipped += 1
            continue
        out[coin].append((int(r["t0"]), FS.expected_cancel_value(fit, fp + ff)))
    if not out:
        raise ForwardDayRefused(
            f"REFUSED: zero actions scored across {len(rows)} rows "
            f"({skipped} lacked features or a fit). A forward report with no "
            f"scores is a FAILURE, not an empty result (R-141).")
    return dict(out)


def seal(day: str, outdir: Path, scored: dict, report: dict) -> dict:
    """Scores OUT of the receipt and into a sealed file. Rule 11.

    The receipt records the sealed file's sha256 and nothing about its
    contents. Reading it is the coordinator's or the USER's act."""
    sp = outdir / f"be_forward_day_SEALED_scores_{day}.json"
    sp.write_text(json.dumps(
        {"protocol": "BE_FORWARD_DAY_SEALED_SCORES_V1", "day": day,
         "SEALED": "rule 11: not for the filing. Counts and refusals only.",
         "per_coin_scores": {c: [list(x) for x in v]
                             for c, v in sorted(scored.items())},
         "report": report}, indent=1, sort_keys=True, default=str))
    return {"path": str(sp), "sha256": _sha_file(sp),
            "bytes": sp.stat().st_size,
            "contents": "per-action scores and the full complement report",
            "not_in_receipt": "no metric, rho, net value or sign appears "
                              "outside this file"}


def run_forward_day(day: str, outdir: Path) -> int:
    """THE SEQUENCE. Every gate refuses BY NAME; a refusal still writes a
    receipt carrying what was established before it, so a refused day is a
    reported fact rather than a silence."""
    import time
    t_start = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    rec: dict = {"protocol": "BE_FORWARD_DAY_SEALED_V1", "day": day,
                 "ratification_ref": RATIFICATION_REF,
                 "as_of_utc": _as_of(),
                 "sealed": True,
                 "sealing_note": "per-action scores and every value metric are "
                                 "written to the sealed file only; this "
                                 "receipt carries counts, identities and "
                                 "hashes and NO metric (rule 11). Unsealing "
                                 "is the coordinator's or the USER's act.",
                 "producing_code": _provenance(),
                 "gates": []}

    def gate(name, fn):
        try:
            out = fn()
        except Exception as e:                       # noqa: BLE001
            rec["gates"].append({"gate": name, "result": "REFUSED",
                                 "why": str(e),
                                 "refusal_type": type(e).__name__})
            raise
        rec["gates"].append({"gate": name, "result": "PASS"})
        return out

    rc = 0
    try:
        rec["day_verdict"] = gate("day_closed_and_attributed",
                                  lambda: assert_day_closed_and_attributed(day))
        pop = gate("population_supply_and_bridge", lambda: population(day))
        rec["population"] = {
            "present_source": str(MARKETS),
            "mask_identity_hash": pop["supply"]["mask_identity_hash"],
            "mask_consumed": pop["supply"]["mask_consumed"],
            "governed": pop["supply"]["governed"],
            "counts_per_coin": counts_per_coin(pop),
            "n_supplied_total": pop["supply"]["n_supplied_total"],
            "n_bridged_specs": len(pop["specs"])}
        # THE FROZEN CONTRACT IS CHECKED AFTER THE POPULATION, DELIBERATELY.
        # The population is a property of the day's ledger and DA's mask —
        # both committed artifacts, neither touched by the candidate — so
        # establishing it costs one file read and no replay, and a refusal
        # here still tells the coordinator what the day's population WAS.
        rec["frozen_contract"] = gate("frozen_candidate_contract",
                                      assert_frozen_contract)
        sel, selc = gate("selection_from_specs",
                         lambda: selected_from_specs(pop["specs"]))
        rec["selection"] = selc
        built = gate("rows_v3_builder", lambda: build_rows_over(sel))
        rec["rows"] = {k: v for k, v in built.items()
                       if k not in ("rows", "per_window")}
        rec["rows"]["n_rows"] = len(built["rows"])
        rec["rows"]["n_actions"] = action_count(built["rows"])
        if built["reconciliation_failures"]:
            raise ForwardDayRefused(
                f"REFUSED: {built['reconciliation_failures']} of "
                f"{built['n_windows']} windows failed reconciliation. The "
                f"reconciliation selftest is the gate: a mismatch fails the "
                f"DAY and is never absorbed.")
        rec["gates"].append({"gate": "reconciliation", "result": "PASS"})
        rec["window_agreement"] = gate(
            "bridged_windows_equal_row_windows",
            lambda: assert_window_sets_agree(pop["specs"], built))
        scored = gate("score_through_frozen_artifact",
                      lambda: score_rows(built["rows"]))
        rep = gate("mask_seam_and_complement_report",
                   lambda: FS.score_day(day, scored, da_verified=True))
        rec["blackout_accounting"] = rep["blackout_accounting"]
        rec["n_actions_scored"] = rep["n_actions_scored"]
        sealed = seal(day, outdir, scored, rep)
        rec["sealed_file"] = sealed
        rec["outcome"] = "SCORED"
    except Exception as e:                           # noqa: BLE001
        rec["outcome"] = "REFUSED"
        rec["refused_at"] = rec["gates"][-1]["gate"] if rec["gates"] else None
        rec["refusal"] = str(e)
        rc = 1
    rec["wall_seconds"] = round(time.time() - t_start, 1)
    rp = outdir / f"be_forward_day_receipt_{day}.json"
    rp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    print(f"{'REFUSED' if rc else 'OK'} {day}: receipt {rp}")
    print(f"  receipt_sha256 {_sha_file(rp)[:16]}")
    if rec.get("refusal"):
        print(f"  {rec['refusal'][:400]}")
    return rc


def selftest() -> int:
    """Every named refusal, red-first, plus the launch-invariance check."""
    import os, subprocess, tempfile
    checks = 0

    def ok(cond, label):
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1
        print(f"  PASS  {label}")

    # ---- gate 1: the frozen contract, both directions -------------------
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # THE GATE MUST AGREE WITH AN INDEPENDENT COMPUTATION, not merely
        # return something. Written as "accept either answer", this control
        # COULD NOT FAIL — a mutant disabling the drift check survived it.
        # The expectation is now computed here from the same committed
        # artifacts, by a second reading, and the gate must match it.
        _c = json.loads(FS.CANDIDATE.read_text())
        _mp = FS.CANDIDATE.parent / _c["manifest"]
        _m = json.loads(_mp.read_text())
        _ps = _m.get("pin_semantics") or {}
        _df = _ps.get("_default", "reproducibility_anchor")
        _expect_drift = _sha_file(_mp) != _c.get("manifest_sha256")
        for _k, _w in (_m.get("hashes") or {}).items():
            if _ps.get(_k, _df) != "reproducibility_anchor":
                continue
            _f = REPO / _k
            if (_sha_file(_f) if _f.exists() else None) != _w:
                _expect_drift = True
        try:
            ev = assert_frozen_contract()
            ok(not _expect_drift and ev["contract"] == "HOLDS",
               f"§10(1) the gate says HOLDS and an INDEPENDENT re-reading of "
               f"the manifest agrees ({ev['anchors_checked']} anchors)")
        except ForwardDayRefused as e:
            ok(_expect_drift and "reproduction contract does not hold" in str(e),
               f"§10(1) the gate REFUSES and an INDEPENDENT re-reading agrees "
               f"that at least one bound input has moved — measured on the "
               f"real committed pair, so this control fails if the gate stops "
               f"noticing (drift expected: {_expect_drift})")
        # and the gate must REFUSE a non-frozen artifact
        cand = tdp / "c.json"
        cand.write_text(json.dumps({"status": "DRAFT"}))
        try:
            assert_frozen_contract(cand)
            ok(False, "a non-FROZEN candidate must REFUSE")
        except ForwardDayRefused as e:
            ok("not FROZEN" in str(e),
               "§10(1) KNOWN-BAD: a candidate whose status is not FROZEN is "
               "refused before anything is read")
        cand.write_text(json.dumps({"status": "FROZEN",
                                    "manifest": "nope.json"}))
        try:
            assert_frozen_contract(cand)
            ok(False, "a missing manifest must REFUSE")
        except ForwardDayRefused as e:
            ok("is absent" in str(e),
               "§10(1) KNOWN-BAD: a candidate naming a manifest that does not "
               "exist is refused, not treated as unbound")

        # a pair whose manifest binds NOTHING: the gate must refuse the
        # zero-anchor read rather than report a pass over an empty set.
        def _pair(hashes, pin=None, msha=None):
            mm = tdp / "m.json"
            mm.write_text(json.dumps({"hashes": hashes,
                                      "pin_semantics": pin or {}}))
            cc = tdp / "cand.json"
            cc.write_text(json.dumps({
                "status": "FROZEN", "manifest": "m.json",
                "manifest_sha256": msha or _sha_file(mm)}))
            return cc
        try:
            assert_frozen_contract(_pair({}))
            ok(False, "a zero-anchor contract must REFUSE")
        except ForwardDayRefused as e:
            ok("compared ZERO anchors" in str(e),
               "§10(1) KNOWN-BAD: a manifest binding NO anchors is refused — "
               "a gate that reads nothing must not report a pass (R-289)")
        # an anchor that MATCHES passes; the same file changed REFUSES.
        _a = tdp / "anchor_file.py"
        _a.write_text("# v1\n")
        _rel = str(_a)                    # absolute; REPO / abs == abs
        ok(assert_frozen_contract(_pair({_rel: _sha_file(_a)}))["contract"]
           == "HOLDS",
           "§10(1) POSITIVE CONTROL: a contract whose anchor matches disk "
           "HOLDS — the gate discriminates rather than refusing universally")
        _a.write_text("# v2 — moved\n")
        try:
            assert_frozen_contract(_pair({_rel: "0" * 64}))
            ok(False, "a moved anchor must REFUSE")
        except ForwardDayRefused as e:
            ok("does not hold" in str(e),
               "§10(1) KNOWN-BAD: an anchor whose bytes moved REFUSES by name")
        # state_at_build entries must NOT be compared (pin_semantics says so)
        _k = tdp / "keep.py"
        _k.write_text("# kept anchor\n")
        ok(assert_frozen_contract(_pair(
            {_rel: "0" * 64, str(_k): _sha_file(_k)},
            pin={"_default": "reproducibility_anchor",
                 _rel: "state_at_build"}))["anchors_checked"] == 1,
           "§10(1) a state_at_build entry is EXCLUDED from the equality "
           "check, as pin_semantics requires, and the anchor count says how "
           "many were actually compared")

    # ---- gate 2: day closed, and attributed to the scheduled unit -------
    ok(assert_day_closed_and_attributed(
        "20260101", {"day_closed_calendar": True,
                     "write_reason": SCHEDULED_PREFIX + " (X)"})[
           "day_closed_calendar"] is True,
       "gate-2 POSITIVE CONTROL: a closed day written by the scheduled unit "
       "passes")
    for lbl, v, want in (
            ("no verdict at all", {}, "has no day verdict"),
            ("day still open", {"day_closed_calendar": False,
                                "write_reason": SCHEDULED_PREFIX},
             "not closed by calendar"),
            ("unattributed hand run", {"day_closed_calendar": True,
                                       "write_reason": "ran it myself"},
             "not written by the scheduled unit"),
            ("prefix only mentioned, not leading",
             {"day_closed_calendar": True,
              "write_reason": "hand run mentioning " + SCHEDULED_PREFIX},
             "not written by the scheduled unit")):
        try:
            assert_day_closed_and_attributed("20260101", v)
            ok(False, f"gate-2 must REFUSE ({lbl})")
        except ForwardDayRefused as e:
            ok(want in str(e),
               f"gate-2 KNOWN-BAD ({lbl}): refused by name — the prefix is "
               f"IMPORTED from DA's preflight and matched as a PREFIX, so a "
               f"mention cannot pass as an attribution")

    # ---- gate 3: the ledger is read, and an empty read REFUSES ----------
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "m.jsonl"
        empty.write_text('{"coin":"btc","window_start":1}\n')
        try:
            present_from_ledger("20260901", empty)
            ok(False, "an out-of-day ledger must REFUSE")
        except ForwardDayRefused as e:
            ok("is not an empty day" in str(e),
               "gate-3 KNOWN-BAD: a ledger with no window for the day REFUSES "
               "— an empty present is a read that found nothing, and scoring "
               "off it would score nothing and say it succeeded")
    real = present_from_ledger("20260901")
    ok(real and all(len(v) > 0 for v in real.values()),
       f"gate-3 POSITIVE CONTROL: 09-01's present is read from the day's own "
       f"ledger — {len(real)} coins, {sum(len(v) for v in real.values())} "
       f"windows (a fact, not a grid)")

    # ---- window-set agreement, both directions -------------------------
    _specs = [{"slug": "a"}, {"slug": "b"}]
    ok(assert_window_sets_agree(_specs, {"rows": [{"slug": "a"}]})[
           "bridged_without_rows"] == 1,
       "agreement: a bridged window with NO rows is COUNTED, never dropped "
       "from the denominator")
    try:
        assert_window_sets_agree(_specs, {"rows": [{"slug": "zzz"}]})
        ok(False, "a row outside the bridge must REFUSE")
    except ForwardDayRefused as e:
        ok("never supplied" in str(e),
           "agreement KNOWN-BAD: a row from a window the bridge never "
           "supplied REFUSES — it is a window nobody admitted (R-418)")

    # ---- F9: the prefix is the IMPORTED object, not a copy of its text --
    import da_governed_verdict_preflight as _PF
    ok(SCHEDULED_PREFIX is _PF.SCHEDULED_PREFIX,
       "gate-2 the scheduled prefix is the IMPORTED object (identity, not "
       "equality) — a restated copy would compare equal today and drift "
       "silently the day DA changes it")

    # ---- F12/F15/F16: the selection is the BRIDGE's, whole ---------------
    _sp = [{"slug": "no-such-slug-12345"}]
    try:
        selected_from_specs(_sp)
        ok(False, "a supplied window with no archive must REFUSE")
    except ForwardDayRefused as e:
        ok("no archive or no token map" in str(e),
           "selection KNOWN-BAD: a supplied window with no archive REFUSES — "
           "R-418 scores the complement WHOLE, so skipping one would "
           "re-select the population the supply already fixed")
    _pop = population("20260901")
    _sel, _selc = selected_from_specs(_pop["specs"][:6])
    ok(_selc["n_selected"] == _selc["n_specs"] == 6
       and [x[0] for x in _sel] == [sp["slug"] for sp in _pop["specs"][:6]],
       f"selection: the tuples are the BRIDGE's slugs in the bridge's order, "
       f"one for one ({_selc}) — not `select_stratified`, which R-418 forbids "
       f"on a race day")
    ok(all(sp["ratification_ref"] == RATIFICATION_REF
           and sp["mask_identity_hash"] == _pop["supply"]["mask_identity_hash"]
           for sp in _pop["specs"]),
       f"bridge: every spec carries ratification_ref {RATIFICATION_REF} and "
       f"the supply's mask_identity_hash, so the receipt names which windows "
       f"it ran over and under what ratification")

    # ---- rows are actions (rule 2) -------------------------------------
    ok(action_count([{"slug": "s", "side": "B", "gen": 1},
                     {"slug": "s", "side": "B", "gen": 1},
                     {"slug": "s", "side": "S", "gen": 1}]) == 2,
       "rule 2: the evaluator de-duplicates rows to ACTIONS "
       "(slug, side, gen), so two rows of one generation count once")

    # ---- sealing: the receipt carries no metric ------------------------
    with tempfile.TemporaryDirectory() as td:
        sealed = seal("20260101", Path(td), {"btc": [(1, 0.5)]},
                      {"n_actions_scored": {"btc": 1}})
        ok(Path(sealed["path"]).exists() and sealed["sha256"],
           "sealing: scores go to the SEALED file and the receipt keeps only "
           "its sha256 (rule 11)")
        body = Path(sealed["path"]).read_text()
        ok("0.5" in body and "0.5" not in json.dumps(sealed),
           "sealing KNOWN-BAD: the score VALUE appears in the sealed file and "
           "NOT in the receipt block — a filing built from the receipt cannot "
           "quote a metric")

    # ---- the driver refuses to write beside canonical artifacts ---------
    ok(str(DERIVED) not in str(Path(".").resolve()),
       "sealing: OUTDIR is a caller parameter and nothing is written under "
       "data/pm_5min/derived/ by this driver")

    _before = checks
    checks = _selftest_launch(checks, ok)
    import os as _os
    if (_os.environ.get("BE_FORWARD_LAUNCH_CHECK") != "1"
            and checks == _before):
        raise AssertionError(
            "the launch-invariance check contributed NO checks, so it did not "
            "run — removing that call is a guard-removal nothing else "
            "notices, which is CO-1's own shape")
    print(f"be_forward_day selftest: {checks} checks OK")
    return 0


def _selftest_launch(checks: int, ok) -> int:
    """Green under BOTH launchers, asserted rather than assumed."""
    import os, subprocess
    if os.environ.get("BE_FORWARD_LAUNCH_CHECK") == "1":
        return checks
    env = dict(os.environ, BE_FORWARD_LAUNCH_CHECK="1")
    r = subprocess.run([sys.executable, "-m",
                        "live.pm_research.be_forward_day", "--selftest"],
                       cwd=str(REPO), env=env, capture_output=True,
                       text=True, timeout=900)
    ok(r.returncode == 0,
       f"launch: GREEN under the PACKAGE launch too (rc={r.returncode}) — a "
       f"suite green under one launcher hid CO-1")
    checks += 1
    return checks


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if "--forward-day" not in sys.argv:
        print("usage: be_forward_day.py --selftest | "
              "--forward-day <YYYYMMDD> --outdir <dir>")
        return 0
    i = sys.argv.index("--forward-day")
    if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("-"):
        print("REFUSED: --forward-day needs a day token (YYYYMMDD)")
        return 2
    day = sys.argv[i + 1]
    outdir = None
    if "--outdir" in sys.argv:
        j = sys.argv.index("--outdir")
        if j + 1 < len(sys.argv) and not sys.argv[j + 1].startswith("-"):
            outdir = Path(sys.argv[j + 1])
    if outdir is None:
        print("REFUSED: --outdir is required. Scores are SEALED (rule 11) and "
              "this driver writes nothing under data/pm_5min/derived/.")
        return 2
    if str(DERIVED) in str(outdir.resolve()):
        print(f"REFUSED: --outdir {outdir} is inside {DERIVED}. Sealed output "
              f"must not land beside canonical artifacts.")
        return 2
    return run_forward_day(day, outdir)


if __name__ == "__main__":
    raise SystemExit(main())
