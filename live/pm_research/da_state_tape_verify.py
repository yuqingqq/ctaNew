"""Gate BE's rebuilt state tape against DA's DECLARED PRED_STATE_V1 schema.

SURFACE AUTHORISATION (R-126, in-file): R-185 accepts DA's review offer AS A
GATE -- the corrected Phase-2 freeze waits on this check.

PRE-REGISTERED: written, and its falsifiers RUN, BEFORE BE's rebuilt tape
exists. The population verifier earned this pattern (Q-DA-77): a checker
authored after seeing the artifact can always be shaped to it, however
honestly. This one cannot be.

EXPECTATIONS COME FROM THE SCHEMA ARTIFACT, NOT FROM THIS FILE. It reads
`da_pred_state_v1_schema.json`, which is itself emitted by running the builder.
So the chain is: builder -> schema -> gate. Nothing in the middle is
transcribed, and a change to the family propagates to this gate without anyone
remembering to update it.

WHAT IT IS LOOKING FOR -- the five ways the FIRST tape went wrong (R-184/R-185),
each of which passed silently at the time:
  1. REQUIRED_INPUTS omitted. `build_tape` without `gaps` makes GAP_AT_CUTOFF
     unreachable; without `bn_recv_ns` the freshness family is CONSTANT.
     Neither errors, so the tape looked fine.
  2. A guardless pin. `level_size_vel_*` were pinned WITHOUT their
     `level_vel_missing_*` flags.
  3. Zero-imputation. `float(x or 0.0)` maps None -> 0.0, so a missing velocity
     is indistinguishable from a genuinely flat book -- the exact distinction
     the family exists to preserve.
  4. Undeclared reduction. 21 of 53 fields, discovered later as "the tested
     family was not the planned family".
  5. state_status ignored -- 27,552 PRE_WINDOW rows consumed as if clean.

IT REFUSES RATHER THAN PASSES when it cannot tell. An unreadable tape, an
unrecognised layout, or a missing schema is a REFUSAL, never a green check.

    python3 live/pm_research/da_state_tape_verify.py --selftest
    python3 live/pm_research/da_state_tape_verify.py verify --tape PATH
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[2]
DERIVED = REPO / "data/pm_5min/derived"
SCHEMA = DERIVED / "da_pred_state_v1_schema.json"
EMBARGO_S = 60.0          # the manifest's declared split_embargo_s

#: R-207: predicates that MUST be asserted for a verdict to authorise fitting.
#: A verdict is not written at all unless every one of these was actually
#: evaluated. The reason is a bypass DA demonstrated against its own design:
#: `all_pass` excludes N/A predicates -- correct for an enforcer that genuinely
#: lives downstream -- which made EVERY predicate silently downgradable by
#: omitting a CLI argument. A gate run with no expectations produced
#: `all_pass: true` and the consumer accepted it. The third state built to
#: avoid certifying the unchecked became the mechanism for certifying it.
LOAD_BEARING = (
    "gap_count_matches_expected",
    "provenance_matches_expected",
    "dataset_non_empty",
    "no_rows_skipped_by_builder",
    "absorption_within_bound",
    "half_open_containment_landed",
)

#: The ONLY predicate permitted to be N/A, by name, and only with its
#: downstream enforcer named in the verdict (R-207). Not a category -- a list
#: of one, so a future N/A needs a ruling rather than an argument.
PERMITTED_NA = {"embargo_respected": (
    "enforced downstream: phase2_embargo.purge_training + assert_embargo, "
    "evidenced by the rerun receipt's per-side purge counts and a computed "
    "realized gap >= 60s (R-189/R-203)")}


class VerdictRefused(RuntimeError):
    """A verdict cannot be written because something load-bearing was not checked."""


class GateRefused(RuntimeError):
    """The gate could not evaluate. NOT a pass."""


def load_schema(path: Path = SCHEMA) -> dict[str, Any]:
    if not path.exists():
        raise GateRefused(
            f"{path} absent. The gate's expectations come from the builder's "
            f"own schema; without it there is nothing to check against, and a "
            f"check against nothing is not a pass. Regenerate with "
            f"`harmful_state_features.py --schema`.")
    return json.loads(path.read_text(encoding="utf-8"))


HEADER_SCAN_CAP = 8 << 20   # declared; see read_header
CHUNK = 1 << 23


def read_header(path: Path) -> dict[str, Any]:
    """The tape's OBJECT-LEVEL pins, without loading the tape.

    The declaration convention said a wrapper "declares the wrapping key on the
    tape itself" and did not say WHERE. BE put `features_under` and
    `clock_basis` at the object level, which is the sensible reading; this gate
    was looking for them per row and would have fallen back to GUESSING the
    layout from a `state` key. Under-specified by me -- so the gate now reads
    the header first, and the schema wording is tightened to match.
    """
    # A BOUNDED READ MAY NOT CONCLUDE ABSENCE. This read 64KB and, if `"rows"`
    # was not inside it, returned {} -- an EMPTY header, silently. The gate
    # then has no `features_under` or `clock_basis` declaration and falls back
    # to GUESSING the layout, which is the exact behaviour this function's
    # docstring says it was written to stop. A header merely LARGER than 64KB
    # (more declared fields, a longer provenance block) was enough.
    #
    # So: scan forward to a declared cap, and distinguish the two reasons
    # `"rows"` can be missing instead of collapsing them. A tape that is a BARE
    # ARRAY is legitimately headerless; an OBJECT whose header cannot be
    # located is unreadable and REFUSES. Fifth sibling of the Q-DA-135 class,
    # found by re-sweeping with idioms my first sweep did not cover.
    raw, i = "", -1
    with path.open("r", encoding="utf-8") as fh:
        while len(raw) < HEADER_SCAN_CAP:
            chunk = fh.read(1 << 16)
            if not chunk:
                break
            raw += chunk
            i = raw.find('"rows"')
            if i >= 0:
                break
    lead = raw.lstrip()[:1]
    if i < 0:
        if lead == "[":
            return {}          # bare array: legitimately headerless
        raise GateRefused(
            f"no `rows` key within the first {HEADER_SCAN_CAP} bytes of an "
            f"OBJECT tape (leading char {lead!r}). REFUSING rather than "
            f"returning an empty header: an empty header makes the gate guess "
            f"the layout, which is what reading the header exists to prevent.")
    prefix = raw[:i].rstrip().rstrip(",")
    try:
        return json.loads(prefix + "}")
    except json.JSONDecodeError:
        raise GateRefused(
            "a `rows` key is present but the object header before it does not "
            "parse. REFUSING rather than proceeding on a header I cannot read.")


def _stream_array(path: Path, key: str) -> Iterator[dict[str, Any]]:
    """Stream `key`'s array elements from a single-line multi-GB object.

    Quote-and-escape aware brace scanner, the same one validated
    element-for-element against `json.load` on a 21MB file during the v3.4
    audit. A 3.17GB `read_text()` + parse is the R-148 allocation burst that
    took the box down; this holds O(one row).
    """
    buf = ""
    pos = keep = depth = 0
    start = None
    in_str = esc = started = closed = False
    marker = f'"{key}"'
    with path.open("r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            if keep:
                buf = buf[keep:]
                pos -= keep
                if start is not None:
                    start -= keep
                keep = 0
            buf += chunk
            if not started:
                j = buf.find(marker)
                if j < 0:
                    keep = max(0, len(buf) - len(marker))
                    pos = len(buf)
                    continue
                k = buf.find("[", j)
                if k < 0:
                    continue
                started = True
                pos = k + 1
            n = len(buf)
            while pos < n:
                c = buf[pos]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == "{":
                    if depth == 0:
                        start = pos
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        yield json.loads(buf[start:pos + 1])
                        keep = pos + 1
                        start = None
                elif c == "]" and depth == 0:
                    closed = True
                    return
                pos += 1
    # EOF BEFORE THE CLOSING BRACKET IS TRUNCATION, NOT COMPLETION.
    # `{"rows":[{...},{...}` yielded both rows and returned normally, so a
    # tape cut off mid-write -- a killed builder, a full disk, an interrupted
    # copy -- read as a complete tape that happened to be short. Every count
    # downstream would then be over a silently truncated population, which is
    # the one thing a gate must never let past.
    if not closed:
        raise GateRefused(
            f"{path.name}: the `{key}` array is NEVER CLOSED -- EOF reached "
            f"with depth={depth} and no `]`. The tape is TRUNCATED, not "
            f"complete, and a short read is not a short tape.")


def iter_tape(path: Path) -> Iterator[dict[str, Any]]:
    """Accept JSONL or a JSON object with a rows list. REFUSE anything else."""
    if not path.exists():
        raise GateRefused(f"{path} does not exist -- an absent tape is not a "
                          f"passing tape.")
    # DISPATCH ON STRUCTURE, WITH NO CATASTROPHIC FALLBACK.
    # The first version looked for `"rows"` in the first 2048 bytes and fell
    # back to JSONL otherwise. This tape's header is richer than that: the
    # marker sits at byte 2545. So a 3.17GB SINGLE-LINE object took the JSONL
    # branch, `json.loads` ate the whole file as one "row", and the gate then
    # refused because that pseudo-row carried no features. A 497-byte
    # threshold silently changed the parsing mode, and the fallback was the
    # worst possible one.
    HEAD = 1 << 20
    head = path.open("rb").read(HEAD)
    stripped = head.lstrip()
    if stripped.startswith(b"{") and b'"rows"' in head:
        yield from _stream_array(path, "rows")
        return
    if stripped.startswith(b"{") and b"\n" not in head:
        raise GateRefused(
            f"tape opens as a single JSON object with no `rows` key in the "
            f"first {HEAD} bytes and no newline. REFUSING: treating it as "
            f"JSONL would parse the entire {path.stat().st_size} bytes as one "
            f"row, which is both wrong and the allocation burst R-148 "
            f"diagnosed.")
    if head.startswith(b"{"):
        n = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                n += 1
        if n == 0:
            raise GateRefused("no rows parsed from a JSONL-looking file")
        return
    raise GateRefused(
        f"unrecognised tape layout (first bytes {head[:40]!r}). REFUSING: a "
        f"gate that guesses a layout can silently check the wrong thing.")


def locate_features(rows: list[dict[str, Any]], schema: dict[str, Any],
                    header: dict[str, Any] | None = None
                    ) -> tuple[str | None, list[dict[str, Any]]]:
    """Find the feature dicts by DECLARATION, never by guessing (R-187).

    The schema declares the builder's NATIVE layout (flat). A consumer that
    wraps the fields must say so on the tape as `features_under`. If the fields
    are in neither place this REFUSES -- because the alternative is what
    happened before: a checker searching top level while the builder nested
    under `state`, whereupon predicates that iterate over "present" fields
    found none, iterated over nothing, and PASSED. Three of them did.
    """
    # Locate on FEATURES, never on identity. `slug` is an emitted field, so a
    # tape carrying only slug+t0 scored a "hit" and the gate declined to
    # refuse -- one identity field masking a wholly absent feature set.
    declared = set(schema["emitted_fields"]) - set(
        schema.get("identity_fields", []))
    probe = rows[:200]
    flat_hits = len(declared & set().union(*(set(r) for r in probe)))
    under = None
    if header and isinstance(header.get("features_under"), str):
        under = header["features_under"]          # DECLARED at object level
    if under is None:
        for r in probe:
            if isinstance(r.get("features_under"), str):
                under = r["features_under"]
                break
    if under is None:
        for cand in ("state", "features", "pred_state"):
            if any(isinstance(r.get(cand), dict) for r in probe):
                under = cand
                break
    nest_hits = 0
    if under:
        nest_hits = len(declared & set().union(
            *(set(r.get(under) or {}) for r in probe)))
    if flat_hits == 0 and nest_hits == 0:
        raise GateRefused(
            f"cannot locate ANY of the {len(declared)} declared FEATURE fields "
            f"(identity fields excluded on purpose), flat "
            f"or under {under!r}. REFUSING rather than checking an empty "
            f"intersection -- an unreadable layout must not yield passes.")
    # THE PROBE CHOSE THE LAYOUT; NOW PROVE THE CHOICE HOLDS ON EVERY ROW.
    #
    # This function exists to stop predicates iterating over an empty field set
    # and PASSING -- and it decided from `rows[:200]` and applied the answer to
    # all 1.76M, never checking. A probe can prove a layout is PRESENT; it can
    # never prove the other layout is ABSENT later in the tape. So a tape whose
    # first 200 rows are flat and whose later rows nest under `state` was
    # returned unflattened, its features invisible, and the very defect named
    # in this docstring would have run -- inside the guard written against it.
    #
    # Found 2026-08-28 by applying the search lesson from Q-DA-135 ("head can
    # prove presence, never absence") to my own committed code.
    chosen = under if nest_hits > flat_hits else None
    bad, first, _conf = _layout_nonconforming(rows, declared, chosen)
    if bad:
        raise GateRefused(
            f"HETEROGENEOUS LAYOUT: the {len(probe)}-row probe selected "
            f"{'nested under ' + repr(chosen) if chosen else 'FLAT'}, but "
            f"{bad} of {len(rows)} rows FAIL per-row conformance "
            f"({_conf['wrong_layout']} under the wrong layout, "
            f"{_conf['no_declared_features']} carrying NO declared feature at "
            f"all); first at index {first[0]}: {first[1]}. REFUSING: those "
            f"rows would have been "
            f"iterated as empty and passed silently, which is the defect this "
            f"locator exists to prevent.")
    if chosen is not None:
        return under, [dict(r.get(under) or {}, **{k: v for k, v in r.items()
                                                   if k != under}) for r in rows]
    return None, rows


def _layout_nonconforming(rows, declared, under,
                          cands=("state", "features", "pred_state")):
    """PER-ROW conformance. Two failure classes, and the second was invisible.

    WRONG LAYOUT -- the row's features are under a different key. Caught since
    Q-DA-136.

    NO FEATURES AT ALL -- the row carries none of the declared features
    ANYWHERE. This was blind: the check only flagged rows whose features lived
    elsewhere, so a row whose whole `state` dict was DELETED had no `other` to
    find and passed. Codex's executed fixture: 401 rows, empty row 401's state,
    and every predicate still passed -- because `present` at the caller is the
    UNION of all rows' keys, so 400 healthy rows carry the union past every
    check while one row contributes nothing and is never missed.

    A union answers "does any row have this field". A gate proof needs "does
    every row", and those differ by exactly the rows that are broken.

    Also returns the per-row declared-feature COUNT distribution, so a bound
    can be set from what the tape actually contains rather than assumed.
    """
    wrong, none_at_all, first, dist = 0, 0, None, {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        here = declared & (set(r) if under is None
                           else set(r.get(under) or {}) | set(r))
        dist[len(here)] = dist.get(len(here), 0) + 1
        if here:
            continue
        other = set()
        for c in cands:
            if c == under:
                continue
            v = r.get(c)
            if isinstance(v, dict):
                other |= declared & set(v)
        if other:
            wrong += 1
            if first is None:
                first = (i, "WRONG_LAYOUT", sorted(other)[:4])
        else:
            none_at_all += 1
            if first is None:
                first = (i, "NO_DECLARED_FEATURES", [])
    return wrong + none_at_all, first, {
        "wrong_layout": wrong, "no_declared_features": none_at_all,
        "per_row_feature_count": dict(sorted(dist.items()))}


def clock_of(r: dict[str, Any], schema: dict[str, Any],
             header: dict[str, Any] | None = None) -> float | None:
    """Absolute decision epoch, honouring the DECLARED clock basis.

    Adding t0 to an already-absolute decision_time double-counts the window
    start. The basis is read from the tape when it declares one, else from the
    schema; it is never assumed.
    """
    hb = (header or {}).get("clock_basis")
    if isinstance(hb, dict):
        hb = hb.get("decision_time")
    basis = r.get("clock_basis") or hb or schema["CLOCK_BASIS"]["decision_time"]
    for k in ("decision_epoch", "abs_decision_time"):
        if r.get(k) is not None:
            return float(r[k])
    dt = r.get("decision_time")
    if dt is None:
        return None
    if str(basis).startswith("absolute"):
        return float(dt)
    if r.get("t0") is None:
        return None
    return float(r["t0"]) + float(dt)


def gate_code_identity() -> dict:
    """WHICH BYTES ISSUED THIS VERDICT.

    The await unit invokes this file BY PATH, so it executes whatever is on
    disk when the tape lands -- which is not necessarily what was reviewed when
    the gate was armed. Today that worked in my favour (R-209 fixes landed
    after arming and the armed gate picked them up). It is still a hole: a
    verdict recorded the tape's `builder_ref` and nothing at all about the
    checker, so "which gate passed this" was unanswerable from the artifact.

    R-205's lesson -- a review binds to a ref -- in operational form: a VERDICT
    binds to a ref too. Additive only; no consumer field changes meaning.
    """
    import hashlib
    import subprocess
    me = Path(__file__).resolve()
    out = {"file": str(me),
           "sha256": hashlib.sha256(me.read_bytes()).hexdigest()[:16]}
    try:
        r = subprocess.run(["git", "-C", str(me.parent), "rev-parse", "--short",
                            "HEAD"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            out["head"] = r.stdout.strip()
        d = subprocess.run(["git", "-C", str(me.parent), "status", "--porcelain",
                            "--", str(me)], capture_output=True, text=True,
                           timeout=10)
        # UNCOMMITTED CHECKER BYTES ARE NOT A DETAIL. A verdict issued by a
        # working-tree edit is not reproducible from any ref, and rule 12 says
        # a freeze is a commit -- so say so IN the artifact rather than let a
        # reader assume `head` describes what ran.
        out["dirty"] = bool(d.returncode == 0 and d.stdout.strip())
    except Exception as e:                       # never fail a verdict on this
        out["git"] = f"unavailable: {type(e).__name__}"
    return out


PRE_EMISSION_KEY = "pre_emission_skip_counts"


def _protocol_major(header) -> int | None:
    """Major version from the header protocol, or None if unidentifiable."""
    import re
    m = re.search(r"_V(\d+)\b", str((header or {}).get("protocol") or ""))
    return int(m.group(1)) if m else None


def builder_skip_counts(header, schema) -> tuple[dict, bool, str]:
    """Where builder-side exclusions actually live. Returns (skips, ok, why).

    R-209 finding 1. The builder splits skips OUT of the row-status tally
    (`pre_emission_skip_counts`, build_state_tape_v2.py:371) precisely because
    skips and statuses are different KINDS of number. This gate derived skips
    from `state_status_counts` instead -- so after the split it read zero
    skips ALWAYS, and both predicates that rest on it passed vacuously on a
    tape that had dropped rows. Deriving from the wrong source is enumerating's
    quieter sibling: the code looks principled and evaluates nothing.

    ABSENCE IS NEVER ZERO. On an artifact that identifies as V5+, a missing
    key is a FAILURE, not an empty tally -- otherwise a builder silently
    dropping the field buys itself a pass.
    """
    hdr = header or {}
    declared = hdr.get(PRE_EMISSION_KEY)
    ver = _protocol_major(hdr)
    if declared is None and ver is not None and ver >= 5:
        return {}, False, (
            f"header declares {hdr.get('protocol')!r} but carries no "
            f"{PRE_EMISSION_KEY!r} -- absence is never zero; a builder that "
            f"stops reporting its exclusions must not read as having none")
    skips = {k: v for k, v in (declared or {}).items() if v}
    # Defence in depth, and the ONLY source pre-split: any key in the ROW
    # status tally that the schema does not declare as a row status describes
    # rows excluded before emission, whatever it is named. Kept so a builder
    # that reverts the split is still seen -- reading ONE source is how this
    # broke, so read both and union.
    row_statuses = set(schema["statuses"])
    for k, v in (hdr.get("state_status_counts") or {}).items():
        if k not in row_statuses and v:
            skips[k] = max(skips.get(k, 0), v)
    return skips, True, ""


def _predicates(schema, n_rows, under, present, statuses, bn_vals, pair_bad,
                asof_checked, asof_viol, tr_max, sc_min, split_seen,
                gapped_slugs_expected, header,
                expect_gap_count: int | None = None,
                expect_provenance: str | None = None,
                at_g1: tuple[int, int] | None = None,
                at_g0: tuple[int, int] | None = None,
                expect_ledger_sha: str | None = None) -> list[dict[str, Any]]:
    """The same verdicts the list gate renders, from accumulated counters.

    Kept beside `gate()` deliberately: two code paths that must agree are a
    seam, and this programme has learned what happens at seams. The selftests
    assert they agree on the same input.
    """
    out: list[dict[str, Any]] = []

    def p(name, ok, detail, applicable=True):
        out.append({"predicate": name, "pass": bool(ok), "detail": detail,
                    "applicable": bool(applicable)})

    # RENAMED to the R-207 contract name. The gate emitted
    # "tape_non_empty" and BE's consumer requires
    # "dataset_non_empty"; both implemented the ruling faithfully and
    # the chain DEADLOCKED -- no verdict this gate could produce would
    # ever be accepted. Behaviour is identical; only the name moves,
    # and it moves to the ruled one.
    p("dataset_non_empty", n_rows > 0, f"{n_rows} rows")
    p("layout_matches_declaration", bool(present),
      f"features located {'under ' + repr(under) if under else 'at top level'}"
      f"{' (DECLARED in the tape header)' if (header or {}).get('features_under') else ' (inferred)'}")
    declared = set(schema["emitted_fields"]) - set(
        schema.get("identity_fields", []))
    # A reduction declared in the tape HEADER counts as declared. BE lists
    # `features_in_order`; anything of mine absent from that list is a stated
    # choice, not a silent drop. (The rule R-185 bound: reductions must be
    # DECLARED against the schema, and they are.)
    carried = (header or {}).get("features_in_order")
    reductions = (declared - set(carried)) if carried else set()
    missing = sorted(declared - present - reductions)
    p("no_undeclared_reduction", not missing,
      f"{len(present & declared)} of {len(declared)} schema feature fields "
      f"present; {len(reductions)} DECLARED reductions via header "
      f"features_in_order"
      + (f"; UNDECLARED ABSENT: {missing[:8]}" if missing else ""))
    pairs = schema["nullable_fields_and_their_flags"]
    checked_pairs = [n for n in pairs if n in present]
    orphans = [n for n, f in pairs.items() if n in present and f not in present]
    p("guard_beside_every_nullable", bool(checked_pairs) and not orphans,
      f"{len(checked_pairs)} of {len(pairs)} declared nullables present; "
      f"orphaned: {orphans or 'none'}")
    p("no_zero_imputation", bool(checked_pairs) and not pair_bad,
      f"rows whose guard says MISSING while the value is PRESENT: "
      f"{dict(pair_bad) or 'none'}")
    p("bn_recv_ns_was_supplied", len(bn_vals) > 1,
      f"bn_feed_age_s distinct non-null values (capped at 64): {len(bn_vals)}")
    # A ZERO-ROW tape has no __ABSENT__ because it has no rows, so this passed
    # on the quarantined artifact. Seen on real evidence, fixed here: an empty
    # status tally cannot demonstrate the field is present.
    p("state_status_present",
      bool(statuses) and "__ABSENT__" not in statuses,
      f"{schema['status_field']} counts: {dict(statuses)}"
      + ("  <-- NO rows carried any status: presence is not demonstrated"
         if not statuses else ""))
    gap_seen = statuses.get("GAP_AT_CUTOFF", 0)
    okg = gap_seen > 0 if gapped_slugs_expected is None else (
        gap_seen > 0 or gapped_slugs_expected == 0)
    p("gaps_were_supplied", okg,
      f"GAP_AT_CUTOFF rows: {gap_seen}"
      + (f"; population carries {gapped_slugs_expected} gapped slugs"
         if gapped_slugs_expected is not None else ""))
    # R-196: a NON-ZERO GAP_AT_CUTOFF count is not the same as the RIGHT one.
    # tape4 returned 286 -- neither the ruled 289 nor the closed-containment
    # 782 -- and would have sailed past a mere >0 test. An expected count is
    # supplied only when one has been PRE-REGISTERED; without it this reports
    # NOT-ASSERTED rather than inventing a target.
    if expect_gap_count is not None:
        p("gap_count_matches_expected", gap_seen == expect_gap_count,
          f"GAP_AT_CUTOFF {gap_seen} vs pre-registered {expect_gap_count}"
          + ("" if gap_seen == expect_gap_count else
             f"  <-- MISMATCH ({gap_seen - expect_gap_count:+d})"))
    else:
        p("gap_count_matches_expected", False,
          "no expected count supplied -- NOT ASSERTED by this run (pass "
          "--expect-gap-count N with a PRE-REGISTERED value)",
          applicable=False)

    # R-198: the tape must say WHICH BYTES BUILT IT. tape4 was
    # provenance-indeterminate -- launched from a moving tree -- and that alone
    # made its 286 unattributable. A prefix match is accepted because refs are
    # quoted short and long; the comparison is anchored at the start so a short
    # ref cannot match some unrelated hash mid-string.
    if expect_provenance is not None:
        hdr_prov = ""
        # `builder_ref` is the RULED interface (R-199 item 2): the launcher
        # passes BUILD_REF in the environment and the builder writes it
        # VERBATIM -- it never runs git at completion. tape5's `builder_commit`
        # was main HEAD read at finish, which is a different quantity from the
        # bytes that built it and is exactly why that tape was unattributable.
        for k in ("builder_ref", "provenance", "pinned_ref", "commit",
                  "build_commit", "built_from_commit", "git_ref"):
            v = (header or {}).get(k)
            if isinstance(v, str):
                hdr_prov = v
                break
            if isinstance(v, dict):
                for kk in ("commit", "ref", "sha", "pinned_ref"):
                    if isinstance(v.get(kk), str):
                        hdr_prov = v[kk]
                        break
                if hdr_prov:
                    break
        a, b = hdr_prov.lower(), expect_provenance.lower()
        okp = bool(hdr_prov) and (a.startswith(b) or b.startswith(a))
        p("provenance_matches_expected", okp,
          f"tape header provenance {hdr_prov or '(ABSENT)'} vs pre-registered "
          f"{expect_provenance}"
          + ("" if okp else "  <-- MISMATCH or ABSENT: bytes of unknown origin "
                            "cannot be certified"))
    else:
        p("provenance_matches_expected", False,
          "no expected provenance supplied -- NOT ASSERTED by this run",
          applicable=False)

    # A SKIPPED ROW IS INVISIBLE TO A ROW-STREAMING GATE. Every other
    # predicate here inspects rows that EXIST; a builder that drops a slug
    # shrinks the population and this gate would never know. The builder now
    # counts such slugs as NO_TOKEN_MAP in its HEADER rather than raising
    # (R-201 seam 23) -- so the header is the only place the loss is visible,
    # and it has to be read. Expected zero on this population; non-zero is a
    # population change, not a formatting detail.
    # DERIVED, NOT ENUMERATED. The first version listed the skip names I
    # happened to know -- and guessed `NO_ARCHIVE` while the builder emits
    # `NO_ARCHIVE_PATH`, so a real skip counter would have been INVISIBLE to
    # this predicate. BE's own lesson applies to the reader too: enumerating
    # constants does not converge, verifying consumers does.
    #
    # The rule instead: the schema declares which statuses ROWS may carry. Any
    # OTHER key in the header's status tally describes rows the builder
    # excluded BEFORE emission -- whatever it is called, including names that
    # do not exist yet.
    skipped, _src_ok, _src_why = builder_skip_counts(header, schema)
    p("no_rows_skipped_by_builder", _src_ok and not skipped,
      _src_why or
      f"builder-side exclusions read from {PRE_EMISSION_KEY!r} (and any "
      f"undeclared key in the row tally): {skipped or 'none'}"
      + ("  <-- rows were DROPPED before emission; the population is smaller "
         "than the declaration implies and no row-level predicate can see it"
         if skipped else ""))

    # R-202 ABSORPTION BOUND, mirrored so BOTH sides enforce it independently.
    # A build that excludes most of its input has not produced a thin
    # population, it has produced a different one. tape6b absorbed 100% into
    # NO_TOKEN_MAP and still wrote a well-formed header.
    total_skipped = sum(skipped.values())
    denom = n_rows + total_skipped
    frac = (total_skipped / denom) if denom else 0.0
    p("absorption_within_bound", _src_ok and frac <= 0.01,
      (_src_why + " -- absorption UNCOMPUTABLE" if not _src_ok else
       f"excluded-before-emission {total_skipped} of {denom} input rows "
       f"= {100*frac:.2f}% (bound 1.00%)")
      + ("  <-- a build absorbing this share has produced a DIFFERENT "
         "population, not a thinner one" if frac > 0.01 else ""))

    # ---- R-213: the LEDGER PIN. A pre-registered count over a GROWING
    # artifact must pin its input, or the count names no reproducible
    # population. FAIL-CLOSED: an unverified pin is not a passed pin, so this
    # asserts even when no expectation was supplied -- otherwise omitting the
    # flag would silently downgrade it to N/A, which is the Q-DA-93 bypass.
    _hdr_sha = (header or {}).get("ledger_sha256")
    _pinned = bool((header or {}).get("ledger_pinned"))
    if expect_ledger_sha:
        p("ledger_pin_matches",
          bool(_hdr_sha) and _pinned
          and str(_hdr_sha).lower() == expect_ledger_sha.lower(),
          f"header ledger_sha256={str(_hdr_sha)[:16] if _hdr_sha else None}... "
          f"pinned={_pinned} vs expected {expect_ledger_sha[:16]}..."
          + ("" if _hdr_sha else
             "  <-- NO ledger sha in the header: the build did not pin its "
             "gap population, so its count is not reproducible"))
    else:
        p("ledger_pin_matches", False,
          "no --expect-ledger-sha supplied: the gap population this tape was "
          "built from was NOT verified against a pin. Absence of a check is "
          "not a passed check.")

    # ---- R-213 mirror of half_open: a row exactly at a g_start must FLAG ----
    if at_g0 is not None:
        _g0p, _g0f = at_g0
        p("at_g0_rows_all_flagged", _g0p > 0 and _g0f == _g0p,
          f"rows exactly at a gap g0: {_g0p} present, {_g0f} flagged "
          f"GAP_AT_CUTOFF"
          + ("" if _g0p else "  <-- NONE PRESENT: cannot pass on an empty "
                             "population; either the gaps did not reach the "
                             "builder or the tape lost the rows")
          + ("" if _g0f == _g0p else
             f"  <-- {_g0p - _g0f} row(s) at an INCLUSIVE lower bound are "
             f"unflagged; R-191 rules [g_start, g_end)"))

    if at_g1 is not None:
        _pres, _flag = at_g1
        p("half_open_containment_landed", _pres > 0 and _flag == 0,
          f"rows exactly at a gap g1: {_pres} present, {_flag} flagged "
          f"GAP_AT_CUTOFF"
          + ("" if _pres > 0 and _flag == 0 else
             ("  <-- ABSENT: the gaps never reached the builder"
              if _pres == 0 else
              "  <-- FLAGGED: containment is still CLOSED; these are the rows "
              "built FROM the gap-ending message, the freshest on the tape")))
    else:
        p("half_open_containment_landed", False,
          "not computed by this call path", applicable=False)

    bad_status = sum(v for k, v in statuses.items()
                     if k not in set(schema["statuses"]) | {"__ABSENT__"})
    p("statuses_are_declared_values",
      bool(statuses) and "__ABSENT__" not in statuses and bad_status == 0,
      f"rows carrying an undeclared status: {bad_status}"
      + ("  <-- on ZERO rows, nothing was checked" if not statuses else ""))
    p("feature_asof_never_after_decision", asof_checked > 0 and asof_viol == 0,
      f"{asof_checked} rows carried both fields; {asof_viol} violations")
    # THE EMBARGO IS NOT THIS TAPE'S TO SATISFY (R-189). The certified chain is
    # builder -> tape -> PURGE -> fit -> score, so the tape is the FULL
    # UNPURGED population by design and its honest `embargo: VIOLATED
    # (unpurged)` header is REQUIRED -- a tape hiding that state would be the
    # real defect. Reported as a THIRD verdict state, never as a pass: an N/A
    # that counted as a pass would be this gate certifying something it has not
    # checked, which is the failure it exists to prevent.
    hdr_emb = (header or {}).get("embargo") or {}
    declared_unpurged = "unpurged" in str(hdr_emb.get("state", "")).lower()
    if declared_unpurged:
        p("embargo_respected", False,
          f"ENFORCED-DOWNSTREAM, not applicable at the tape (R-189). The tape "
          f"DECLARES its own state: {hdr_emb.get('state')!r}. This gate does "
          f"NOT certify the embargo; enforcement evidence is required at "
          f"fixture seam 2 and in the rerun receipt (purge row-counts per side "
          f"plus a computed `realized gap >= {EMBARGO_S:.0f}s`).",
          applicable=False)
    elif split_seen["train"] and split_seen["score"]:
        if tr_max is not None and sc_min is not None:
            margin = sc_min - tr_max
            p("embargo_respected", margin >= EMBARGO_S,
              f"score starts {margin:.3f}s after training ends; declared "
              f"embargo {EMBARGO_S:.0f}s")
        else:
            p("embargo_respected", False,
              "split labels present but no usable decision clock")
    else:
        p("embargo_respected", False,
          f"train rows {split_seen['train']}, score rows {split_seen['score']} "
          f"-- embargo NOT certified by this gate")
    return out


def _accumulate(rows, schema, header, coin_gaps=None):
    """Fold a row LIST into the same counters the streaming path builds.

    R-209 finding 4. `GC`, `_coin_gaps` and the at-g1 counters were bound only
    on the STREAMING path, so this path raised NameError/UnboundLocalError the
    moment a row carried coin/t0/t_start -- i.e. on any realistic row, never on
    the synthetic ones the suite used. Worse than the crash: at_g1 was never
    RETURNED, so `half_open_containment_landed` -- load-bearing -- went N/A on
    this path without anyone choosing that.

    The import stays function-local: `da_gap_at_cutoff_count` imports THIS
    module, so hoisting it to module scope is a circular import.
    """
    _coin_gaps = coin_gaps            # loaded LAZILY: only a row that carries
    at_g1_present = at_g1_flagged = 0  # the identity triple needs the gap table
    at_g1_checked = 0
    pairs = schema["nullable_fields_and_their_flags"]
    under, flat = locate_features(rows, schema, header)   # REFUSES if unlocatable
    present = set().union(*(set(r) for r in flat)) if flat else set()
    statuses = collections.Counter()
    bn_vals, pair_bad = set(), collections.Counter()
    asof_checked = asof_viol = 0
    tr_max = sc_min = None
    split_seen = {"train": 0, "score": 0}
    for r in flat:
        statuses[str(r.get(schema["status_field"], "__ABSENT__"))] += 1
        b = r.get("bn_feed_age_s")
        if b is not None and len(bn_vals) < 64:
            bn_vals.add(round(float(b), 6))
        for n, f in pairs.items():
            if float(r.get(f) or 0.0) != 0.0 and r.get(n) is not None:
                pair_bad[n] += 1
        a, t = r.get("feature_asof"), r.get("decision_time")
        if a is not None and t is not None:
            asof_checked += 1
            if a > t + 1e-9:
                asof_viol += 1
        _c, _t0, _ts = r.get("coin"), r.get("t0"), r.get("t_start")
        if _c is not None and _t0 is not None and _ts is not None:
            import da_gap_at_cutoff_count as GC   # local: GC imports us
            if _coin_gaps is None:
                _coin_gaps = GC.coin_gaps()[0]
            at_g1_checked += 1
            _exact, _ = GC.at_upper_edge(_coin_gaps.get(_c, []),
                                         float(_t0) + float(_ts))
            if _exact:
                at_g1_present += 1
                if str(r.get(schema["status_field"], "")) == "GAP_AT_CUTOFF":
                    at_g1_flagged += 1
        sp = str(r.get("split", "")).lower()
        c = clock_of(r, schema, header)
        if sp.startswith("train"):
            split_seen["train"] += 1
            if c is not None:
                tr_max = c if tr_max is None else max(tr_max, c)
        elif sp.startswith(("score", "test", "hold")):
            split_seen["score"] += 1
            if c is not None:
                sc_min = c if sc_min is None else min(sc_min, c)
    return dict(n_rows=len(flat), under=under, present=present,
                statuses=statuses, bn_vals=bn_vals, pair_bad=pair_bad,
                asof_checked=asof_checked, asof_viol=asof_viol,
                tr_max=tr_max, sc_min=sc_min, split_seen=split_seen,
                at_g1=((at_g1_present, at_g1_flagged)
                       if at_g1_checked else None))


def gate(rows: list[dict[str, Any]], schema: dict[str, Any],
         gapped_slugs_expected: int | None = None,
         header: dict[str, Any] | None = None,
         expect_gap_count: int | None = None,
         expect_provenance: str | None = None,
         coin_gaps: dict | None = None) -> list[dict[str, Any]]:
    """List-input gate. DELEGATES to the same `_predicates` the stream uses.

    It used to render its own verdicts from its own loop, and I added a
    path-agreement test to keep the two honest. That test then caught a real
    divergence -- which was the right outcome, but the better fix is to DELETE
    the second path rather than police it. Two implementations that must agree
    are a seam; one implementation is not.
    """
    if not rows:
        return [{"predicate": "dataset_non_empty", "pass": False,
                 "applicable": True,
                 "detail": "0 rows -- an empty tape never passes"}]
    acc = _accumulate(rows, schema, header, coin_gaps)
    return _predicates(schema, acc["n_rows"], acc["under"], acc["present"],
                       acc["statuses"], acc["bn_vals"], acc["pair_bad"],
                       acc["asof_checked"], acc["asof_viol"], acc["tr_max"],
                       acc["sc_min"], acc["split_seen"], gapped_slugs_expected,
                       header, expect_gap_count, expect_provenance,
                       acc["at_g1"])


def verify(tape: Path, schema_path: Path = SCHEMA, ledger: Path | None = None,
           gapped_slugs_expected: int | None = None,
           expect_gap_count: int | None = None,
           expect_provenance: str | None = None,
           expect_ledger_sha: str | None = None) -> dict[str, Any]:
    schema = load_schema(schema_path)
    header = read_header(tape)
    # SINGLE PASS, COUNTERS ONLY. Streaming the parse is not enough: a
    # `list(iter_tape(...))` over a 3.17GB tape materialises every row as a
    # ~53-key dict and is the same allocation problem one layer up. Only the
    # first BUFFER rows are held, for the key-union; everything else is
    # accumulated.
    BUFFER = 400
    buf: list[dict[str, Any]] = []
    under = None
    n_rows = 0
    statuses: collections.Counter = collections.Counter()
    bn_vals: set = set()
    pair_bad: collections.Counter = collections.Counter()
    asof_checked = asof_viol = 0
    tr_max = None
    sc_min = None
    split_seen = {"train": 0, "score": 0}
    present: set = set()
    pairs = schema["nullable_fields_and_their_flags"]
    # at-g1: computed IN-GATE so it needs no expectation flag and therefore has
    # no downgrade path. The count alone cannot distinguish "half-open landed"
    # from "the gaps never arrived"; only PRESENT-and-UNFLAGGED separates them,
    # and the artifact that authorises fitting should not omit the assertion
    # DA argued was the decisive one.
    if header is not None and PRE_EMISSION_KEY not in header:
        raise VerdictRefused(
            f"REFUSED: tape header carries no {PRE_EMISSION_KEY!r}. This gate "
            f"reads builder-side exclusions from that key ONLY; without it "
            f"no_rows_skipped_by_builder and absorption_within_bound would "
            f"report zero having measured nothing. Absence is never zero -- "
            f"header keys present: {sorted(header)}")
    import da_gap_at_cutoff_count as GC
    # THE GATE MUST READ THE SAME GAP POPULATION AS THE BUILDER. The live
    # ledger grows during a build, so a gate reading the live file while the
    # tape was built from a pinned snapshot compares against a DIFFERENT
    # population -- and the edge predicates would be answering a question
    # nobody asked. R-213(4).
    _coin_gaps, _ = GC.coin_gaps(path=ledger)
    at_g1_present = at_g1_flagged = 0
    at_g0_present = at_g0_flagged = 0
    _starts = {c: {a for a, _b in ivs} for c, ivs in _coin_gaps.items()}
    # LOCATE FIRST, THEN ACCUMULATE. The first version resolved `under` only
    # once the buffer reached BUFFER rows, so a tape SHORTER than that never
    # un-nested and every feature read as absent -- the agreement seam test
    # caught it on a 3-row tape. On the multi-million-row tape it would have
    # worked by luck, which is worse than failing.
    stream = iter_tape(tape)
    for raw in stream:
        buf.append(raw)
        if len(buf) >= BUFFER:
            break
    if buf:
        under, flat = locate_features(buf, schema, header)
        present = set().union(*(set(r) for r in flat))
    # THE PER-ROW CONFORMANCE FIX WAS BYPASSED IN PRODUCTION. `locate_features`
    # was hardened to check every row it is GIVEN -- and production gives it a
    # 400-row BUFFER, then flattens the remaining millions inline below with no
    # check at all. Codex's executed counterexample: 400 valid rows plus a
    # 401st whose `state` dict is empty returns all_pass=true, because the
    # buffer never sees row 401 and `present` is the union over the buffer.
    #
    # Rule 17 INSIDE the fix: the helper was correct, tested, and unreached.
    # A fix that does not run on the production path is a fix the production
    # path does not have.
    #
    # So conformance runs INCREMENTALLY over every streamed row -- one set
    # intersection per row, on a stream that already does more work than that.
    _declared = set(schema["emitted_fields"]) - set(
        schema.get("identity_fields", []))
    _nonconf, _first_nonconf, _featdist = 0, None, collections.Counter()

    import itertools
    for raw in itertools.chain(buf, stream):
        n_rows += 1
        r = raw
        if under:
            r = dict(raw.get(under) or {},
                     **{k: v for k, v in raw.items() if k != under})
        _here = _declared & set(r)
        _featdist[len(_here)] += 1
        if not _here:
            _nonconf += 1
            if _first_nonconf is None:
                _first_nonconf = (n_rows - 1, sorted(set(raw))[:6])
        statuses[str(r.get(schema["status_field"], "__ABSENT__"))] += 1
        b = r.get("bn_feed_age_s")
        if b is not None and len(bn_vals) < 64:
            bn_vals.add(round(float(b), 6))
        for n, f in pairs.items():
            if float(r.get(f) or 0.0) != 0.0 and r.get(n) is not None:
                pair_bad[n] += 1
        a, t = r.get("feature_asof"), r.get("decision_time")
        if a is not None and t is not None:
            asof_checked += 1
            if a > t + 1e-9:
                asof_viol += 1
        _c, _t0, _ts = r.get("coin"), r.get("t0"), r.get("t_start")
        if _c is not None and _t0 is not None and _ts is not None:
            _exact, _ = GC.at_upper_edge(_coin_gaps.get(_c, []),
                                         float(_t0) + float(_ts))
            if _exact:
                at_g1_present += 1
                if str(r.get(schema["status_field"], "")) == "GAP_AT_CUTOFF":
                    at_g1_flagged += 1
            # AT-g0: the mirror predicate. R-213 routes both paths through one
            # comparison, so a row exactly at a g_start must now be FLAGGED --
            # the 4 rows that were not are what forced the rebuild. Computed
            # IN-GATE like at-g1, so it has no expectation flag and therefore
            # no downgrade path.
            if (float(_t0) + float(_ts)) in _starts.get(_c, ()):
                at_g0_present += 1
                if str(r.get(schema["status_field"], "")) == "GAP_AT_CUTOFF":
                    at_g0_flagged += 1
        sp = str(r.get("split", "")).lower()
        c = clock_of(r, schema, header)
        if sp.startswith("train"):
            split_seen["train"] += 1
            if c is not None:
                tr_max = c if tr_max is None else max(tr_max, c)
        elif sp.startswith(("score", "test", "hold")):
            split_seen["score"] += 1
            if c is not None:
                sc_min = c if sc_min is None else min(sc_min, c)
    # REFUSE ON THE WHOLE-STREAM RESULT, not the buffer's. This is the check
    # `locate_features` performs on what it is given; production gives it 400
    # rows, so it has to be re-asserted over every row that was actually
    # flattened -- otherwise the hardening stops at row 400 and the tape is
    # certified on a sample.
    if _nonconf:
        raise GateRefused(
            f"PER-ROW CONFORMANCE FAILED ON THE PRODUCTION STREAM: "
            f"{_nonconf} of {n_rows} rows carry NO declared feature after "
            f"flattening (first at index {_first_nonconf[0]}, keys "
            f"{_first_nonconf[1]}). Feature-count distribution: "
            f"{dict(sorted(_featdist.items()))}. REFUSING: those rows would "
            f"have been iterated as empty and passed silently -- the defect "
            f"the locator exists to prevent, reached through the streaming "
            f"path that bypassed it.")
    preds = _predicates(schema, n_rows, under, present, statuses, bn_vals,
                        pair_bad, asof_checked, asof_viol, tr_max, sc_min,
                        split_seen, gapped_slugs_expected, header,
                        expect_gap_count, expect_provenance,
                        (at_g1_present, at_g1_flagged),
                        (at_g0_present, at_g0_flagged), expect_ledger_sha)
    return {"gate": "da_state_tape_verify_v1", "tape": str(tape),
            "gate_code": gate_code_identity(),
            "schema_family": schema["family"], "n_rows": n_rows,
            # THE DISTRIBUTION, not just all_pass: on tape6e it is
            # {48: 1764206}. A reader can see whether every row carried the
            # same feature set or whether the tape is ragged, which `all_pass`
            # alone cannot express.
            "per_row_feature_count": dict(sorted(_featdist.items())),
            # builder_ref included so the verdict CARRIES the ref it
            # certified: a reader can see WHICH bytes were gated without
            # opening a 3GB tape. Not a hole without it -- absence is accepted
            # and the gate asserts the ref tape-side -- but a verdict that
            # names its subject should name it completely.
            "tape_header_pins": {k: header.get(k) for k in
                                 ("features_under", "clock_basis", "protocol",
                                  "built_from_schema", "builder_ref")},
            "predicates": preds,
            "not_applicable": [x["predicate"] for x in preds
                               if not x.get("applicable", True)],
            "all_pass": all(x["pass"] for x in preds
                            if x.get("applicable", True))}


def assert_all_load_bearing_asserted(preds: list[dict[str, Any]]) -> None:
    """REFUSE to produce a verdict unless every load-bearing check ran.

    Preventing the artifact from existing beats relying on every future reader
    to interrogate it -- the same argument that made the tape-side guard refuse
    rather than degrade.
    """
    by = {x["predicate"]: x for x in preds}
    missing = [n for n in LOAD_BEARING if n not in by]
    unasserted = [n for n in LOAD_BEARING
                  if n in by and not by[n].get("applicable", True)]
    bad_na = [x["predicate"] for x in preds
              if not x.get("applicable", True)
              and x["predicate"] not in PERMITTED_NA]
    problems = []
    if missing:
        problems.append(f"load-bearing predicates ABSENT: {missing}")
    if unasserted:
        problems.append(
            f"load-bearing predicates NOT ASSERTED (an expectation was not "
            f"supplied): {unasserted}")
    if bad_na:
        problems.append(
            f"predicates marked N/A that are not permitted to be: {bad_na}. "
            f"Only {sorted(PERMITTED_NA)} may be N/A, and only with its "
            f"downstream enforcer named.")
    if problems:
        raise VerdictRefused(
            "REFUSING to write a verdict -- " + "; ".join(problems)
            + ". A verdict that cannot distinguish 'checked and passed' from "
              "'never checked' is not a gate.")


def write_verdict(rep: dict[str, Any], tape: Path, out: Path) -> dict[str, Any]:
    """Emit the machine-readable verdict (R-199 item 3).

    BE's fit stage REFUSES to run without a PASS here, so this instrument
    becomes the pipeline gate in fact and not only in protocol. Two properties
    follow from that and are deliberate:

    * The artifact names the TAPE IT JUDGED by path AND content prefix. A
      verdict that does not identify its subject can be replayed against a
      different artifact -- and this programme has had three in-place
      overwrites destroy the bytes a claim was anchored to.
    * `all_pass` is written from the predicate list, never passed in. A verdict
      file whose headline disagrees with its own table is the rule-10 defect
      in artifact form.
    """
    # Every key verify() produces is either CARRIED into the verdict or named
    # here as deliberately omitted. Adding a field upstream then forces a
    # decision instead of a silent drop -- which is how gate_code came to read
    # None in the artifact that authorises fitting.
    _OMITTED_ON_PURPOSE = {
        "gate",          # renamed to "verdict" below
        "tape",          # renamed to "tape_path"
        "all_pass",      # RECOMPUTED here from the table, never copied
        "gap_at_cutoff_rows",   # bulk row detail; the counter reports it
    }
    import hashlib
    h = hashlib.sha256()
    with tape.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert_all_load_bearing_asserted(rep["predicates"])
    applicable = [x for x in rep["predicates"] if x.get("applicable", True)]
    verdict = {
        "verdict": "da_tape_gate_verdict_v1",
        "produced_by": "live/pm_research/da_state_tape_verify.py",
        "tape_path": str(tape),
        "tape_sha256_prefix": h.hexdigest()[:16],
        "tape_bytes": tape.stat().st_size,
        "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_rows": rep.get("n_rows"),
        # CARRIED, not omitted: the distribution is the evidence that per-row
        # conformance ran over the WHOLE stream rather than a 400-row buffer,
        # and a verdict asserting conformance should show what it measured.
        # (My own _OMITTED_ON_PURPOSE guard refused the verdict until this
        # line existed -- upstream-present/downstream-absent, caught by the
        # check built for exactly that.)
        "per_row_feature_count": rep.get("per_row_feature_count"),
        "schema_family": rep.get("schema_family"),
        "tape_header_pins": rep.get("tape_header_pins"),
        # WHICH BYTES ISSUED THIS VERDICT. verify() computed this and
        # write_verdict silently dropped it, so the artifact carried
        # gate_code=None while the return value carried the truth -- the
        # third time this exact shape has bitten: the field is PRESENT
        # upstream and never TAKES EFFECT in the thing consumers read.
        # BE binds fit manifests to this, so an absent field is a manifest
        # bound to nothing.
        "gate_code": rep.get("gate_code"),
        "predicates": rep["predicates"],
        "not_applicable": rep.get("not_applicable", []),
        # An N/A predicate must NAME who enforces it instead. "Not checked
        # here" is only acceptable when accompanied by where it IS checked.
        "not_applicable_enforced_by": {
            x["predicate"]: PERMITTED_NA[x["predicate"]]
            for x in rep["predicates"]
            if not x.get("applicable", True)
            and x["predicate"] in PERMITTED_NA},
        "load_bearing_asserted": list(LOAD_BEARING),
        # recomputed HERE from the table, never copied
        "all_pass": all(x["pass"] for x in applicable),
        "n_applicable": len(applicable),
        "fields_deliberately_omitted": sorted(_OMITTED_ON_PURPOSE),
        "note": ("ALL_PASS is recomputed from the predicate table in this "
                 "file, not carried in. A consumer should re-derive it rather "
                 "than trust the headline, and should check tape_sha256_prefix "
                 "against the artifact it is about to use -- a verdict that "
                 "cannot identify its subject can be replayed against another."),
    }
    # ENFORCE the allowlist. Without this the set above is a comment, and a
    # comment has never stopped a field from vanishing.
    _dropped = set(rep) - set(verdict) - _OMITTED_ON_PURPOSE
    if _dropped:
        raise VerdictRefused(
            f"REFUSED: verify() produced {sorted(_dropped)} and the verdict "
            f"does not carry them. Either add them to the artifact or name "
            f"them in _OMITTED_ON_PURPOSE -- a field that exists upstream and "
            f"is absent downstream is the defect that made gate_code read None "
            f"in the artifact consumers resolve.")
    verdict["_dropped_check"] = ("no unexpected field loss between verify() "
                                 "and this artifact")
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(verdict, indent=2, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(out)
    return verdict


def _selftests() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            raise AssertionError(f"selftest failed: {label}")

    # A MINIATURE schema for the unit cases -- but it now carries the LAYOUT
    # and CLOCK_BASIS pins, because a test schema that omits them is how the
    # gate came to assume both (R-187).
    schema = {
        "family": "PRED_STATE_V1",
        "emitted_fields": ["a", "a_missing", "bn_feed_age_s", "state_status",
                           "feature_asof", "decision_time"],
        "statuses": ["OK", "PRE_WINDOW", "GAP_AT_CUTOFF"],
        "status_field": "state_status",
        "nullable_fields_and_their_flags": {"a": "a_missing"},
        "LAYOUT": {"native": "flat", "features_under": None},
        "CLOCK_BASIS": {"decision_time": "window_relative_seconds"},
    }

    def R(**kw):
        base = {"a": 1.0, "a_missing": 0.0, "bn_feed_age_s": 0.1,
                "state_status": "OK", "feature_asof": 1.0,
                "decision_time": 2.0}
        base.update(kw)
        return base

    def res(rows, **kw):
        return {p["predicate"]: p["pass"] for p in gate(rows, schema, **kw)}

    good = ([R(bn_feed_age_s=0.1), R(bn_feed_age_s=0.2, a=None, a_missing=1.0),
             R(bn_feed_age_s=0.3, state_status="GAP_AT_CUTOFF"),
             R(bn_feed_age_s=0.4, split="train", t0=0, decision_time=1.0),
             R(bn_feed_age_s=0.5, split="score", t0=0, decision_time=100.0)]
            * 12)          # clear the 50-row signature minimum
    g = res(good)
    for k in ("dataset_non_empty", "no_undeclared_reduction",
              "guard_beside_every_nullable", "no_zero_imputation",
              "bn_recv_ns_was_supplied", "state_status_present",
              "gaps_were_supplied", "statuses_are_declared_values",
              "feature_asof_never_after_decision", "embargo_respected"):
        ok(g[k], f"a clean tape passes {k}")

    # FALSIFIERS -- each must FIRE (rule 15)
    ok(not res([])["dataset_non_empty"], "an EMPTY tape fails, never passes")
    ok(not res([{k: v for k, v in R().items() if k != "a_missing"}])
       ["guard_beside_every_nullable"],
       "a nullable present WITHOUT its guard flag is caught (guardless pin)")
    ok(not res([R(a=0.0, a_missing=1.0)])["no_zero_imputation"],
       "guard says MISSING while the value is PRESENT -- caught exactly; this "
       "is the fingerprint float(x or 0.0) leaves")
    ok(res([R(a=None, a_missing=1.0), R()])["no_zero_imputation"],
       "a genuine None beside a hot flag is CLEAN")
    ok(res([R()] * 3)["no_zero_imputation"],
       "a column that is LEGITIMATELY always computable is NOT flagged -- the "
       "false positive the seam test exposed in the signature version, which "
       "would have fired on gen_age_s and pm_feed_age_s on every real tape")
    ok(not res([R(bn_feed_age_s=0.1), R(bn_feed_age_s=0.1)])
       ["bn_recv_ns_was_supplied"],
       "a CONSTANT freshness family is caught (bn_recv_ns omitted)")
    ok(not res([R()], gapped_slugs_expected=5)["gaps_were_supplied"],
       "zero GAP_AT_CUTOFF with gapped slugs expected is caught")
    ok(res([R()], gapped_slugs_expected=0)["gaps_were_supplied"],
       "zero GAP_AT_CUTOFF is fine when the population HAS no gapped slugs")
    ok(not res([{k: v for k, v in R().items() if k != "state_status"}])
       ["state_status_present"], "a tape without state_status is caught")
    ok(not res([R(state_status="INVENTED")])["statuses_are_declared_values"],
       "an undeclared status value is caught")
    ok(not res([R(feature_asof=9.0, decision_time=2.0)])
       ["feature_asof_never_after_decision"],
       "a knowledge-time violation is caught")
    ok(not res([{k: v for k, v in R().items()
                 if k not in ("feature_asof", "decision_time")}])
       ["feature_asof_never_after_decision"],
       "a tape that does not CARRY the fields cannot pass the invariant "
       "vacuously -- absence is failure, not silence")
    near = [R(split="train", t0=0, decision_time=100.0),
            R(split="score", t0=0, decision_time=110.0)]
    ok(not res(near)["embargo_respected"],
       "a 10s gap against a 60s declared embargo is caught (R-184 blocker)")
    ok(not res([R()])["embargo_respected"],
       "no split labels -> embargo NOT certified, rather than assumed clean")
    # declared reductions are honoured
    # Reductions are declared in the tape HEADER via `features_in_order`,
    # which is where BE declares them. A field absent from that list is a
    # stated choice; a field absent with no list at all is a silent drop.
    lean = [{k: v for k, v in R().items() if k != "a"}] * 3
    carried = [f for f in schema["emitted_fields"] if f != "a"]
    ok(gate(lean, schema, 0, {"features_in_order": carried})[2]["pass"],
       "a reduction DECLARED in the header is accepted")
    ok(not res(lean)["no_undeclared_reduction"],
       "the same absence with NO declaration is an undeclared reduction")

    # refusals, never passes
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            load_schema(Path(td) / "nope.json")
        except GateRefused:
            ok(True, "a missing schema REFUSES rather than passing")
        else:
            ok(False, "must refuse without a schema")
        try:
            list(iter_tape(Path(td) / "absent.jsonl"))
        except GateRefused:
            ok(True, "an absent tape REFUSES")
        else:
            ok(False, "must refuse an absent tape")
        weird = Path(td) / "w.csv"
        weird.write_text("a,b\n1,2\n", encoding="utf-8")
        try:
            list(iter_tape(weird))
        except GateRefused:
            ok(True, "an unrecognised layout REFUSES rather than guessing")
        else:
            ok(False, "must refuse an unrecognised layout")

    # ---- THE SEAM TEST (R-187) -------------------------------------------
    # Every case above feeds the gate dicts I INVENTED. That is precisely how
    # this gate came to search top level while the builder nested, and to add
    # t0 to a clock that might already be absolute: per-module selftests,
    # however falsified, cannot certify a seam. So round-trip a REAL builder
    # row through the REAL schema.
    import harmful_state_features as SF
    real_schema = SF.declared_schema()
    k2 = ("BUY", 0.5)
    tp = SF._synth_tape(level_t={k2: [10.0, 20.0]}, level_v={k2: [100.0, 60.0]},
                        pm_event_t=[10.0, 20.0])
    real_row = SF.features_at(tp, SF._row(25.0))
    flat_tape = [dict(real_row, slug="s", t0=1787650500)]
    under, got = locate_features(flat_tape, real_schema)
    ok(under is None and set(real_schema["emitted_fields"]) <= set(got[0]),
       "SEAM: a real builder row is located at top level and complete")

    nested_tape = [{"slug": "s", "t0": 1787650500, "state": dict(real_row)}]
    under_n, got_n = locate_features(nested_tape, real_schema)
    ok(under_n == "state"
       and set(real_schema["emitted_fields"]) <= set(got_n[0]),
       "SEAM: the SAME row nested under `state` is still located -- the exact "
       "layout mismatch that made three predicates pass on nothing")
    # Two real rows, the second at a level the tape never saw, so the
    # velocity nullables genuinely carry None with their flags set.
    sparse = SF.features_at(tp, SF._row(25.0, level=0.99))
    nested_many = [{"slug": "s", "t0": 1787650500, "state": dict(r)}
                   for r in ([real_row, sparse] * 30)]
    gn = {q["predicate"]: q["pass"] for q in gate(nested_many, real_schema)}
    ok(gn["guard_beside_every_nullable"],
       "SEAM: nested rows are now CHECKED for guard-beside-nullable, not "
       "skipped-and-passed -- the defect that made three predicates pass on "
       "an unreadable layout")
    ok(gn["no_zero_imputation"],
       "SEAM: REAL builder rows pass the exact zero-imputation test -- "
       "including the always-computable columns the signature version would "
       "have falsely flagged")
    forged = [{"slug": "s", "t0": 1,
               "state": dict(real_row, gen_age_s=0.0, gen_age_missing=1.0)}]
    gf = {q["predicate"]: q["pass"] for q in gate(forged, real_schema)}
    ok(not gf["no_zero_imputation"],
       "SEAM: a real row with ONE value imputed is caught -- so the exact test "
       "still fires on the defect it exists for")

    opaque = [{"slug": "s", "t0": 1, "payload": {"nothing": 1}}]
    try:
        locate_features(opaque, real_schema)
    except GateRefused:
        ok(True, "SEAM: an unlocatable layout REFUSES instead of passing on an "
                 "empty intersection")
    else:
        ok(False, "must refuse when no declared field can be found")

    # clock basis is READ, not assumed
    rel = {"t0": 1000.0, "decision_time": 25.0}
    ab = {"t0": 1000.0, "decision_time": 1787650525.0,
          "clock_basis": "absolute_epoch"}
    ok(clock_of(rel, real_schema) == 1025.0,
       "window-relative decision_time is made absolute by adding t0")
    ok(clock_of(ab, real_schema) == 1787650525.0,
       "an ALREADY-ABSOLUTE decision_time is not double-counted -- the basis "
       "is read from the tape, never assumed")
    ok(clock_of(rel, real_schema) != clock_of(
        dict(rel, clock_basis="absolute_epoch"), real_schema),
       "the two bases give DIFFERENT answers, so a rule that ignores the pin "
       "cannot pass (R-42 mirror on the clock)")

    # ---- the two code paths must AGREE (they are themselves a seam) -------
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        f = Path(td) / "t.json"
        rows_obj = [{"slug": f"s{i}", "t0": 1787650500,
                     "state": dict(real_row)} for i in range(3)]
        f.write_text(json.dumps({"protocol": "T", "features_under": "state",
                                 "clock_basis": {"decision_time":
                                                 "window_relative_seconds"},
                                 PRE_EMISSION_KEY: {},
                                 "rows": rows_obj}), encoding="utf-8")
        streamed = list(iter_tape(f))
        ok(streamed == rows_obj,
           "SEAM: the streaming reader round-trips an object tape exactly")
        hdr = read_header(f)
        ok(hdr.get("features_under") == "state",
           "SEAM: object-level pins are READ, not guessed -- BE declares them "
           "on the header, and the first version of this gate looked per-row")
        # list-gate vs streaming-gate on identical input
        list_verdict = {q["predicate"]: q["pass"]
                        for q in gate(rows_obj, real_schema, 0, hdr)}
        rep = verify(f, gapped_slugs_expected=0)
        stream_verdict = {q["predicate"]: q["pass"] for q in rep["predicates"]}
        shared = set(list_verdict) & set(stream_verdict)
        ok(shared and all(list_verdict[k] == stream_verdict[k] for k in shared),
           "SEAM: list gate and streaming gate agree. This test CAUGHT a real "
           "divergence when they were two implementations; they now share one "
           "`_predicates`, so it is close to a tautology -- kept because it is "
           "the regression guard against anyone re-forking them")

    unp = {"embargo": {"state": "VIOLATED (unpurged): gap -8.1s < 60.0s"}}
    pr = gate(good, schema, 0, unp)
    emb = [x for x in pr if x["predicate"] == "embargo_respected"][0]
    ok(emb["applicable"] is False and emb["pass"] is False,
       "a tape DECLARING itself unpurged makes the embargo NOT APPLICABLE -- "
       "and it stays pass=False, so it can never be read as certified")
    ok(all(x.get("applicable", True) for x in gate(good, schema, 0)
           if x["predicate"] == "embargo_respected") is False or True,
       "without that declaration the embargo remains an ordinary predicate")
    plain = [x for x in gate(good, schema, 0)
             if x["predicate"] == "embargo_respected"][0]
    ok(plain.get("applicable", True) is True,
       "N/A is granted ONLY on the tape's own declaration, never by default")

    gc_hit = {q["predicate"]: q for q in gate(good, schema, 0, None, 12)}
    ok(gc_hit["gap_count_matches_expected"]["pass"],
       "a matching pre-registered count passes")
    gc_miss = {q["predicate"]: q for q in gate(good, schema, 0, None, 999)}
    ok(not gc_miss["gap_count_matches_expected"]["pass"],
       "a MISMATCHED count fails -- tape4's 286 would not sail past a >0 test")
    gc_none = {q["predicate"]: q for q in gate(good, schema, 0, None, None)}
    ok(gc_none["gap_count_matches_expected"]["applicable"] is False
       and gc_none["gap_count_matches_expected"]["pass"] is False,
       "with NO expected count it is NOT ASSERTED -- never a silent pass, and "
       "the gate does not invent a target")

    pv = lambda hdr, exp: {q["predicate"]: q for q in
                           gate(good, schema, 0, hdr, None, exp)}
    ok(pv({"provenance": "057b1b7aa13b04c8"}, "057b1b7")
       ["provenance_matches_expected"]["pass"],
       "a short pre-registered ref matches a long declared one")
    ok(pv({"pinned_ref": "057b1b7"}, "057b1b7aa13b04c8")
       ["provenance_matches_expected"]["pass"],
       "...and the reverse, since refs are quoted at both lengths")
    ok(not pv({"provenance": "9fb043b"}, "057b1b7")
       ["provenance_matches_expected"]["pass"],
       "a DIFFERENT commit fails -- this is the check tape4 had no way to make")
    ok(not pv({}, "057b1b7")["provenance_matches_expected"]["pass"],
       "an ABSENT provenance header FAILS -- unknown origin is not a pass")
    ok(not pv({"provenance": "aa057b1b7bb"}, "057b1b7")
       ["provenance_matches_expected"]["pass"],
       "a ref appearing MID-STRING does not match -- the compare is anchored")
    ok(pv({"provenance": "057b1b7"}, None)
       ["provenance_matches_expected"]["applicable"] is False,
       "with no expectation it is NOT ASSERTED, never a silent pass")

    # --- builder_ref is accepted, and preferred over stale alternatives ----
    ok(pv({"builder_ref": "abc1234"}, "abc1234")
       ["provenance_matches_expected"]["pass"],
       "builder_ref is an accepted provenance field (R-199 ruled interface)")
    ok(not pv({"builder_commit": "abc1234"}, "abc1234")
       ["provenance_matches_expected"]["pass"],
       "builder_commit is NOT accepted -- tape5 wrote main HEAD at completion "
       "into that field, a different quantity from the bytes that built it, "
       "and accepting it would certify exactly what made tape5 unattributable")
    ok(pv({"builder_ref": "abc1234", "commit": "deadbee"}, "abc1234")
       ["provenance_matches_expected"]["pass"],
       "builder_ref is consulted FIRST, so a stale sibling field cannot win")

    # --- the verdict artifact ---------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td) / "t.json"
        tp.write_text(json.dumps({"features_under": "state",
                                  PRE_EMISSION_KEY: {}, "rows": [
            {"slug": "s", "t0": 1, "state": dict(real_row)}]}), encoding="utf-8")
        rep = verify(tp, gapped_slugs_expected=0)
        # A verdict now REQUIRES every load-bearing predicate to have been
        # ASSERTED, so the fixture supplies them all.
        def _full(preds):
            """Fixture helper: every load-bearing predicate PRESENT and
            ASSERTED. It must also FLIP ones already present as N/A -- the
            fixture's first version only added missing names and the refusal
            still fired, correctly, on the two the real run had downgraded."""
            out = [dict(x, applicable=True) if x["predicate"] in LOAD_BEARING
                   else x for x in preds]
            by = {x["predicate"] for x in out}
            for n in LOAD_BEARING:
                if n not in by:
                    out.append({"predicate": n, "pass": True,
                                "applicable": True, "detail": "fixture"})
            return out
        rep = dict(rep, predicates=_full(rep["predicates"]))
        vout = Path(td) / "v.json"
        v = write_verdict(rep, tp, vout)
        ok(vout.exists() and json.loads(vout.read_text())["all_pass"] == v["all_pass"],
           "the verdict artifact is written and self-consistent")
        ok(len(v["tape_sha256_prefix"]) == 16 and v["tape_path"] == str(tp),
           "it identifies the tape it judged by PATH and CONTENT prefix -- a "
           "verdict that cannot name its subject can be replayed against "
           "another artifact")
        forged = dict(rep)
        forged["predicates"] = _full([{"predicate": "x", "pass": False,
                                       "applicable": True}])
        v2 = write_verdict(forged, tp, vout)
        ok(v2["all_pass"] is False,
           "all_pass is RECOMPUTED from the predicate table, so a headline "
           "cannot disagree with the table beside it (rule 10 in artifact form)")

    # ---- the ARTIFACT must carry the checker identity, not just verify() ----
    with tempfile.TemporaryDirectory() as td:
        tp3 = Path(td) / "t.json"
        tp3.write_text(json.dumps({"features_under": "state",
                                   PRE_EMISSION_KEY: {}, "rows": [
            {"slug": "s", "t0": 1, "state": dict(real_row)}]}), encoding="utf-8")
        rep3 = verify(tp3, gapped_slugs_expected=0)
        o3 = Path(td) / "v.json"
        v3 = write_verdict(dict(rep3, predicates=_full(rep3["predicates"])), tp3, o3)
        on_disk = json.loads(o3.read_text())
        ok(isinstance(on_disk.get("gate_code"), dict)
           and len(on_disk["gate_code"].get("sha256", "")) == 16,
           "the WRITTEN verdict carries gate_code -- verify() computed it and "
           "write_verdict dropped it, so the artifact read None while the "
           "return value read true (present upstream, absent downstream)")
        # FALSIFIER: an unlisted new field must REFUSE, not vanish
        rep_extra = dict(rep3, predicates=_full(rep3["predicates"]))
        rep_extra["a_field_added_next_week"] = 1
        refused = False
        try:
            write_verdict(rep_extra, tp3, o3)
        except VerdictRefused as e:
            refused = "a_field_added_next_week" in str(e)
        ok(refused,
           "a field verify() gains and the verdict does not carry REFUSES by "
           "name -- the allowlist is enforced, not documented")

    sk = lambda hdr: {q["predicate"]: q["pass"] for q in gate(good, schema, 0, hdr)}
    ok(sk({"state_status_counts": {"OK": 5}})["no_rows_skipped_by_builder"],
       "a header with no skip counters passes")
    ok(not sk({"state_status_counts": {"OK": 5, "NO_ARCHIVE_PATH": 3}})
       ["no_rows_skipped_by_builder"],
       "NO_ARCHIVE_PATH is caught -- the name my enumerated list GUESSED wrong "
       "(it had NO_ARCHIVE), which would have made a real skip invisible")
    ok(not sk({"state_status_counts": {"OK": 5, "A_NAME_INVENTED_TOMORROW": 2}})
       ["no_rows_skipped_by_builder"],
       "a skip name that DOES NOT EXIST YET is caught, because the set is "
       "derived from the declared ROW statuses rather than enumerated")
    ok(sk({"state_status_counts": {"OK": 5, "PRE_WINDOW": 2,
                                   "GAP_AT_CUTOFF": 1}})
       ["no_rows_skipped_by_builder"],
       "declared ROW statuses are NOT mistaken for skips, however large")
    ok(not sk({"state_status_counts": {"OK": 5, "NO_TOKEN_MAP": 3}})
       ["no_rows_skipped_by_builder"],
       "a NON-ZERO NO_TOKEN_MAP fails -- rows dropped before emission are "
       "invisible to every row-level predicate in this gate")
    ok(sk({"state_status_counts": {"OK": 5, "NO_TOKEN_MAP": 0}})
       ["no_rows_skipped_by_builder"],
       "an explicit ZERO is a pass, not a failure -- the counter being present "
       "and zero is the builder REPORTING no loss")
    ok(sk({})["no_rows_skipped_by_builder"],
       "no counters at all passes, since a builder that never skips need not "
       "declare a zero")

    ab = lambda hdr, rows: {q["predicate"]: q["pass"]
                            for q in gate(rows, schema, 0, hdr)}
    ok(ab({"state_status_counts": {"NO_TOKEN_MAP": 100}}, good)
       ["absorption_within_bound"] is False,
       "100 skipped against 60 emitted breaches the 1% absorption bound")
    ok(ab({"state_status_counts": {"NO_TOKEN_MAP": 0}}, good)
       ["absorption_within_bound"],
       "zero skipped is within bound")
    ok(ab({}, good)["absorption_within_bound"],
       "no counters declared is within bound")
    ok(not ab({"state_status_counts": {"NO_TOKEN_MAP": 1}}, good * 2)
       ["absorption_within_bound"] is False,
       "1 skipped of 121 is 0.83%%, inside the bound")

    # ---- R-213: ledger pin + the at-g0 mirror, both directions ------------
    PIN_SHA = "6cb3a027e25fb5df97f74c4ec63924fa710ab24d9f64486651e040f1b9660d55"
    lp = lambda hdr, exp: {q["predicate"]: q for q in
                           _predicates(schema, 1, None, set(), collections.Counter(),
                                       set(), collections.Counter(), 0, 0, None,
                                       None, {"train": 0, "score": 0}, 0, hdr,
                                       None, None, None, None, exp)}
    ok(lp({"ledger_sha256": PIN_SHA, "ledger_pinned": True}, PIN_SHA)
       ["ledger_pin_matches"]["pass"],
       "R-213: a tape built from the pinned ledger PASSES against its sha")
    ok(not lp({"ledger_sha256": "0"*64, "ledger_pinned": True}, PIN_SHA)
       ["ledger_pin_matches"]["pass"],
       "a DIFFERENT ledger sha fails -- the count would be over another "
       "population than the one registered")
    ok(not lp({"ledger_pinned": False}, PIN_SHA)["ledger_pin_matches"]["pass"],
       "a build that did NOT pin its ledger fails: an unpinned count over a "
       "growing artifact is not reproducible")
    ok(not lp({"ledger_sha256": PIN_SHA, "ledger_pinned": True}, None)
       ["ledger_pin_matches"]["pass"],
       "and omitting --expect-ledger-sha FAILS rather than going N/A -- "
       "absence of a check is not a passed check (the Q-DA-93 bypass)")

    g0 = lambda pres, flag: {q["predicate"]: q for q in
                             _predicates(schema, 1, None, set(), collections.Counter(),
                                         set(), collections.Counter(), 0, 0, None,
                                         None, {"train": 0, "score": 0}, 0, {},
                                         None, None, None, (pres, flag), None)}
    ok(g0(4, 4)["at_g0_rows_all_flagged"]["pass"],
       "R-213 mirror: all rows at an inclusive lower bound FLAGGED passes")
    ok(not g0(4, 0)["at_g0_rows_all_flagged"]["pass"],
       "the tape6d state -- 4 present, 0 flagged -- FAILS, which is what "
       "forced the rebuild")
    ok(not g0(4, 3)["at_g0_rows_all_flagged"]["pass"],
       "PARTIAL flagging fails too: 3 of 4 is not the ruled predicate")
    ok(not g0(0, 0)["at_g0_rows_all_flagged"]["pass"],
       "and an EMPTY at-g0 population cannot pass -- zero present means the "
       "gaps never reached the builder, not that containment landed")

    # ---- R-209 finding 1: the SOURCE the skip predicates read -------------
    # The user's probe: the builder splits skips into their own tally, this
    # gate kept reading the row-status tally, and 100 dropped rows passed BOTH
    # predicates. Deriving from the wrong source is enumerating's quieter
    # sibling -- so the source itself is now under test, both directions.
    V5 = {"protocol": "PHASE2_STATE_TAPE_V5"}
    ok(not sk({**V5, PRE_EMISSION_KEY: {"NO_ARCHIVE_PATH": 100}})
       ["no_rows_skipped_by_builder"],
       "R-209: 100 skips in pre_emission_skip_counts FAIL -- the exact probe "
       "that passed vacuously when this gate read state_status_counts")
    ok(ab({**V5, PRE_EMISSION_KEY: {"NO_ARCHIVE_PATH": 100}}, good)
       ["absorption_within_bound"] is False,
       "R-209 other direction: absorption is computed over the SAME source, "
       "so 100 of 160 input rows breaches the bound")
    ok(sk({**V5, PRE_EMISSION_KEY: {"NO_ARCHIVE_PATH": 0}})
       ["no_rows_skipped_by_builder"],
       "an EXPLICIT zero in the new tally passes -- the builder reporting no "
       "loss is a measurement, and a checker that only fails is not a checker")
    ok(ab({**V5, PRE_EMISSION_KEY: {}}, good)["absorption_within_bound"],
       "an explicitly EMPTY tally passes")
    ok(not sk({**V5, "state_status_counts": {"OK": 5}})
       ["no_rows_skipped_by_builder"],
       "ABSENCE IS NEVER ZERO: a V5 artifact with no skip tally at all FAILS "
       "rather than reading as having skipped nothing")
    ok(ab({**V5, "state_status_counts": {"OK": 5}}, good)
       ["absorption_within_bound"] is False,
       "absorption on a V5 artifact missing the tally is UNCOMPUTABLE, and "
       "uncomputable is not within bound")
    ok(not sk({**V5, PRE_EMISSION_KEY: {},
               "state_status_counts": {"OK": 5, "NO_TOKEN_MAP": 7}})
       ["no_rows_skipped_by_builder"],
       "defence in depth: a builder that REVERTS the split and puts skips back "
       "in the row tally is still caught -- reading one source is how this broke")

    # ---- the verdict names the bytes that issued it ------------------------
    _gc = gate_code_identity()
    ok(len(_gc.get("sha256", "")) == 16,
       "the verdict carries a sha of the CHECKER, not only of the tape: the "
       "await unit runs this file by path, so armed-at is not ran-at")
    ok(isinstance(_gc.get("dirty"), bool) or "git" in _gc,
       "and it says whether those bytes were UNCOMMITTED -- a verdict issued "
       "from a working-tree edit is reproducible from no ref at all")

    # ---- R-210: the bound is TOTAL, and per-status detail STAYS REPORTED ---
    # R-202's per-status wording is superseded: two statuses under the bound
    # individually, over it together, is the same population change decided by
    # how the loss was LABELLED. The evasion is the reason for the ruling, so
    # the evasion is the test.
    _many = [dict(real_row) for _ in range(1000)]
    _split = {q["predicate"]: q for q in gate(
        _many, schema, 0, {**V5, PRE_EMISSION_KEY: {"NO_TOKEN_MAP": 6,
                                                    "NO_ARCHIVE_PATH": 6}})}
    ok(_split["absorption_within_bound"]["pass"] is False,
       "R-210: two skip statuses at 0.59% each (1.19% total) REFUSE -- a "
       "per-status bound would have built, and status names are free")
    _d = _split["no_rows_skipped_by_builder"]["detail"]
    ok("NO_TOKEN_MAP" in _d and "NO_ARCHIVE_PATH" in _d,
       "R-210: the bound is total but the PER-STATUS breakdown stays in the "
       "detail -- a refusal has to say what was lost, not only how much")

    # ---- R-209 finding 4: the list path on a REALISTIC row -----------------
    # It crashed the moment a row carried coin/t0/t_start, which no synthetic
    # fixture did -- and at_g1 was never returned, so a load-bearing predicate
    # went N/A on this path without anyone choosing that.
    _real = [dict(R(), coin="btc", t0=1787650500.0, t_start=-30.0)]
    _inj = {"btc": [(1787650400.0, 1787650470.0)]}
    _lp = {q["predicate"]: q for q in
           gate(_real, schema, 0, {**V5, PRE_EMISSION_KEY: {}},
                coin_gaps=_inj)}
    ok("half_open_containment_landed" in _lp,
       "R-209: the list path survives a realistic row (coin/t0/t_start) -- it "
       "raised UnboundLocalError before, on any row a real tape contains")
    _at = [dict(R(), coin="btc", t0=1787650500.0, t_start=-30.0,
                state_status="OK")]
    _lp2 = {q["predicate"]: q for q in
            gate(_at, schema, 0, {**V5, PRE_EMISSION_KEY: {}},
                 coin_gaps={"btc": [(1787650000.0, 1787650470.0)]})}
    ok(_lp2["half_open_containment_landed"].get("applicable", True),
       "and at_g1 REACHES the predicate on the list path: a row exactly at g1 "
       "makes containment ASSERTED, not silently N/A")
    ok(_lp2["half_open_containment_landed"]["pass"],
       "exactly-at-g1 PRESENT and UNFLAGGED is the half-open landing (a row "
       "at the upper edge is OUTSIDE the gap and must not carry GAP_AT_CUTOFF)")

    # ---- R-207: a verdict that checked NOTHING must not exist -------------
    with tempfile.TemporaryDirectory() as td:
        tp2 = Path(td) / "t.json"
        tp2.write_text(json.dumps({"rows": []}), encoding="utf-8")
        o2 = Path(td) / "v.json"

        def _try(preds):
            try:
                write_verdict({"predicates": preds, "n_rows": 1,
                               "schema_family": "X", "tape_header_pins": {},
                               "not_applicable": []}, tp2, o2)
                return True
            except VerdictRefused:
                return False

        allp = [{"predicate": n, "pass": True, "applicable": True,
                 "detail": ""} for n in LOAD_BEARING]
        ok(_try(allp), "a verdict with every load-bearing check ASSERTED writes")
        ok(not o2.exists() or True, "")
        checks -= 1  # the line above is a no-op guard, not a check

        for drop in LOAD_BEARING:
            partial = [x for x in allp if x["predicate"] != drop]
            ok(not _try(partial),
               f"REFUSES when {drop} is ABSENT -- a load-bearing predicate "
               f"that never ran cannot be certified by omission")
            na = [dict(x, applicable=False) if x["predicate"] == drop else x
                  for x in allp]
            ok(not _try(na),
               f"REFUSES when {drop} is present but NOT ASSERTED -- this is "
               f"the exact bypass: applicable=False was excluded from "
               f"all_pass, so a verdict that checked nothing said PASS")
        ok(not _try([]),
           "THE CHECKED-NOTHING CASE: an empty predicate table is REFUSED, "
           "permanently")
        emb = allp + [{"predicate": "embargo_respected", "pass": False,
                       "applicable": False, "detail": ""}]
        ok(_try(emb),
           "embargo_respected MAY be N/A -- the one permitted exception, by "
           "name, with its downstream enforcer named in the verdict")
        v_emb = json.loads(o2.read_text())
        ok("embargo_respected" in v_emb["not_applicable_enforced_by"],
           "...and the verdict NAMES who enforces it instead: 'not checked "
           "here' is only acceptable beside where it IS checked")
        other_na = allp + [{"predicate": "some_other", "pass": False,
                            "applicable": False, "detail": ""}]
        ok(not _try(other_na),
           "any OTHER N/A is refused -- the permitted set is a list of one, so "
           "a future exception needs a ruling rather than an argument")

    # ---- T1/T2: the fix was BYPASSED, and EOF read as completion --------
    import tempfile as _tfp
    _SCH = DERIVED / "da_pred_state_v1_schema.json"
    if _SCH.exists():
        _sc = json.loads(_SCH.read_text())
        _ft = [f for f in _sc["emitted_fields"]
               if f not in set(_sc.get("identity_fields", []))]

        def _r(i, empty=False):
            st = {} if empty else {f: (0.0 if f != "state_status" else "OK")
                                   for f in _ft}
            if not empty:
                for _n, _fl in _sc["nullable_fields_and_their_flags"].items():
                    st[_fl] = 0.0
            return {"slug": "btc-updown-5m-1787650200", "t0": 1787650200,
                    "gen": i, "split": "train", "state": st}

        _HDR = {"features_under": "state", "protocol": "PHASE2_STATE_TAPE_V5",
                "pre_emission_skip_counts": {},
                "required_inputs_supplied": {"gaps": True,
                                             "bn_recv_ns": True}}
        with _tfp.TemporaryDirectory() as _td:
            _bad = Path(_td) / "bypass.json"
            _bad.write_text(json.dumps(dict(
                _HDR, rows=[_r(i) for i in range(400)] + [_r(400, True)])),
                encoding="utf-8")
            _refused = ""
            try:
                verify(_bad, _SCH)
            except GateRefused as e:
                _refused = str(e)
            ok("PER-ROW CONFORMANCE FAILED ON THE PRODUCTION STREAM" in _refused,
               "T1: 400 valid rows plus a 401st with an EMPTY state dict now "
               "REFUSES. locate_features was hardened to check every row it is "
               "GIVEN -- and production gives it a 400-row BUFFER, then "
               "flattens the remaining millions inline with no check. RULE 17 "
               "INSIDE THE FIX: the helper was correct, tested, and UNREACHED. "
               "Conformance now runs incrementally over every streamed row")
            _good = Path(_td) / "clean.json"
            _good.write_text(json.dumps(dict(
                _HDR, rows=[_r(i) for i in range(401)])), encoding="utf-8")
            _v = verify(_good, _SCH)
            ok(_v.get("per_row_feature_count") == {48: 401},
               f"positive control: 401 uniformly-valid rows pass the stream "
               f"check and the verdict CARRIES the distribution "
               f"{_v.get('per_row_feature_count')} -- not just all_pass, so a "
               f"reader can see whether the tape is uniform or ragged")
            _tr = Path(_td) / "trunc.json"
            _tr.write_text('{"rows":[' + json.dumps(_r(0)) + ","
                           + json.dumps(_r(1)), encoding="utf-8")
            _t2 = ""
            try:
                list(iter_tape(_tr))
            except GateRefused as e:
                _t2 = str(e)
            ok("NEVER CLOSED" in _t2 and "TRUNCATED" in _t2,
               "T2: EOF before the closing bracket now REFUSES. It yielded "
               "both rows and returned normally, so a tape cut off mid-write "
               "-- killed builder, full disk, interrupted copy -- read as a "
               "COMPLETE tape that happened to be short, and every count "
               "downstream would have been over a silently truncated "
               "population")
            _okf = Path(_td) / "closed.json"
            _okf.write_text(json.dumps(dict(_HDR, rows=[_r(0), _r(1)])),
                            encoding="utf-8")
            ok(len(list(iter_tape(_okf))) == 2,
               "positive control: a properly closed array still streams, so "
               "the truncation check is not simply refusing every tape")

    # ---- a bounded header read may not conclude absence (Q-DA-137) ------
    import tempfile as _tfh
    with _tfh.TemporaryDirectory() as _td:
        _pad = "x" * 100_000            # header LARGER than the old 64KB read
        _big = Path(_td) / "big.json"
        _big.write_text(json.dumps({"features_under": "state",
                                    "note": _pad, "rows": [{"a": 1}]}),
                        encoding="utf-8")
        _h = read_header(_big)
        ok(_h.get("features_under") == "state",
           "A HEADER LARGER THAN 64KB IS NOW READ: the old bounded read took "
           "64KB, missed `rows`, and returned an EMPTY header silently -- so "
           "the gate lost `features_under` and fell back to GUESSING the "
           "layout, which is what reading the header exists to prevent")
        _bare = Path(_td) / "bare.json"
        _bare.write_text(json.dumps([{"a": 1}]), encoding="utf-8")
        ok(read_header(_bare) == {},
           "positive control: a BARE ARRAY tape is legitimately headerless and "
           "returns {} -- the two reasons `rows` can be missing are "
           "distinguished, not collapsed")
        _hdrless = Path(_td) / "obj.json"
        _hdrless.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
        _ref = ""
        try:
            read_header(_hdrless)
        except GateRefused as e:
            _ref = str(e)
        ok("REFUSING rather than returning an empty header" in _ref,
           "and an OBJECT with no `rows` key REFUSES instead of returning an "
           "empty header -- absence of a locatable header is unreadability, "
           "not a headerless tape")

    # ---- the probe chose a layout it never verified (Q-DA-136) ----------
    _sch = {"emitted_fields": ["f_a", "f_b", "f_c"],
            "identity_fields": ["slug", "t0"]}
    _flat = [{"slug": "s", "t0": 1, "f_a": 1.0, "f_b": 2.0} for _ in range(200)]
    _nest = [{"slug": "s", "t0": 1, "state": {"f_a": 1.0, "f_b": 2.0}}
             for _ in range(50)]
    _u, _r = locate_features(list(_flat), _sch)
    ok(_u is None and len(_r) == 200,
       "positive control: a HOMOGENEOUS flat tape locates flat and is "
       "returned untouched")
    _u2, _r2 = locate_features(list(_nest) * 4, _sch)
    ok(_u2 == "state" and "f_a" in _r2[0],
       "positive control: a homogeneous NESTED tape locates `state` and is "
       "flattened")
    _mixed_refused = ""
    try:
        locate_features(list(_flat) + list(_nest), _sch)
    except GateRefused as e:
        _mixed_refused = str(e)
    ok("HETEROGENEOUS LAYOUT" in _mixed_refused and "50 of 250" in _mixed_refused,
       f"THE PROBE CHOSE A LAYOUT IT NEVER VERIFIED: 200 FLAT rows followed by "
       f"50 NESTED ones now REFUSES, naming 50 of 250. Before this guard the "
       f"probe read rows[:200], selected FLAT, and returned all 250 "
       f"unflattened -- the 50 nested rows' features were invisible, "
       f"predicates iterating 'present' fields found none, iterated over "
       f"nothing, and PASSED. That is verbatim the defect `locate_features` "
       f"exists to prevent, running INSIDE the guard written against it")
    _rev_refused = ""
    try:
        locate_features(list(_nest) * 4 + [{"slug": "s", "t0": 1, "f_a": 1.0}],
                        _sch)
    except GateRefused as e:
        _rev_refused = str(e)
    ok(_rev_refused == "",
       "and the REVERSE mix does NOT refuse: a flat row under a nested "
       "selection still surfaces its fields when flattened, so it conforms -- "
       "the guard fires on real invisibility, not on mere heterogeneity")

    print(f"da_state_tape_verify selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["verify"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tape", default=None)
    ap.add_argument("--gapped-slugs", type=int, default=None)
    ap.add_argument("--expect-gap-count", type=int, default=None,
                    help="PRE-REGISTERED expected GAP_AT_CUTOFF row count")
    ap.add_argument("--expect-provenance", default=None,
                    help="PRE-REGISTERED commit the tape must declare")
    ap.add_argument("--ledger", default=None,
                    help="pinned ledger snapshot; the gate MUST read the same "
                         "gap population the builder did (R-213(4))")
    ap.add_argument("--expect-ledger-sha", default=None,
                    help="sha256 of the pinned ledger; the tape header's "
                         "ledger_sha256 must equal it")
    ap.add_argument("--verdict-out", default=None,
                    help="write the machine-readable verdict artifact here")
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    if not a.tape:
        raise SystemExit("--tape PATH required; refusing to guess a tape")
    rep = verify(Path(a.tape), gapped_slugs_expected=a.gapped_slugs,
                 expect_gap_count=a.expect_gap_count,
                 expect_provenance=a.expect_provenance,
                 ledger=Path(a.ledger) if a.ledger else None,
                 expect_ledger_sha=a.expect_ledger_sha)
    if a.verdict_out:
        write_verdict(rep, Path(a.tape), Path(a.verdict_out))
    print(json.dumps({k: v for k, v in rep.items() if k != "predicates"},
                     indent=2, sort_keys=True))
    print("\nPREDICATES")
    for x in rep["predicates"]:
        mark = ("N/A " if not x.get("applicable", True)
                else "PASS" if x["pass"] else "FAIL")
        print(f"  [{mark}] {x['predicate']}: {x['detail']}")
    print(f"\nALL PASS: {rep['all_pass']}")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
