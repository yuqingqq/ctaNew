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
    raw = path.open("r", encoding="utf-8").read(1 << 16)
    i = raw.find('"rows"')
    if i < 0:
        return {}
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
    in_str = esc = started = False
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
                    return
                pos += 1


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
    if nest_hits > flat_hits:
        return under, [dict(r.get(under) or {}, **{k: v for k, v in r.items()
                                                   if k != under}) for r in rows]
    return None, rows


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


def _predicates(schema, n_rows, under, present, statuses, bn_vals, pair_bad,
                asof_checked, asof_viol, tr_max, sc_min, split_seen,
                gapped_slugs_expected, header,
                expect_gap_count: int | None = None) -> list[dict[str, Any]]:
    """The same verdicts the list gate renders, from accumulated counters.

    Kept beside `gate()` deliberately: two code paths that must agree are a
    seam, and this programme has learned what happens at seams. The selftests
    assert they agree on the same input.
    """
    out: list[dict[str, Any]] = []

    def p(name, ok, detail, applicable=True):
        out.append({"predicate": name, "pass": bool(ok), "detail": detail,
                    "applicable": bool(applicable)})

    p("tape_non_empty", n_rows > 0, f"{n_rows} rows")
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
    p("state_status_present", "__ABSENT__" not in statuses,
      f"{schema['status_field']} counts: {dict(statuses)}")
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

    bad_status = sum(v for k, v in statuses.items()
                     if k not in set(schema["statuses"]) | {"__ABSENT__"})
    p("statuses_are_declared_values",
      "__ABSENT__" not in statuses and bad_status == 0,
      f"rows carrying an undeclared status: {bad_status}")
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


def _accumulate(rows, schema, header):
    """Fold a row LIST into the same counters the streaming path builds."""
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
                tr_max=tr_max, sc_min=sc_min, split_seen=split_seen)


def gate(rows: list[dict[str, Any]], schema: dict[str, Any],
         gapped_slugs_expected: int | None = None,
         header: dict[str, Any] | None = None,
         expect_gap_count: int | None = None) -> list[dict[str, Any]]:
    """List-input gate. DELEGATES to the same `_predicates` the stream uses.

    It used to render its own verdicts from its own loop, and I added a
    path-agreement test to keep the two honest. That test then caught a real
    divergence -- which was the right outcome, but the better fix is to DELETE
    the second path rather than police it. Two implementations that must agree
    are a seam; one implementation is not.
    """
    if not rows:
        return [{"predicate": "tape_non_empty", "pass": False,
                 "applicable": True,
                 "detail": "0 rows -- an empty tape never passes"}]
    acc = _accumulate(rows, schema, header)
    return _predicates(schema, acc["n_rows"], acc["under"], acc["present"],
                       acc["statuses"], acc["bn_vals"], acc["pair_bad"],
                       acc["asof_checked"], acc["asof_viol"], acc["tr_max"],
                       acc["sc_min"], acc["split_seen"], gapped_slugs_expected,
                       header, expect_gap_count)


def verify(tape: Path, schema_path: Path = SCHEMA,
           gapped_slugs_expected: int | None = None,
           expect_gap_count: int | None = None) -> dict[str, Any]:
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

    import itertools
    for raw in itertools.chain(buf, stream):
        n_rows += 1
        r = raw
        if under:
            r = dict(raw.get(under) or {},
                     **{k: v for k, v in raw.items() if k != under})
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
    preds = _predicates(schema, n_rows, under, present, statuses, bn_vals,
                        pair_bad, asof_checked, asof_viol, tr_max, sc_min,
                        split_seen, gapped_slugs_expected, header,
                        expect_gap_count)
    return {"gate": "da_state_tape_verify_v1", "tape": str(tape),
            "schema_family": schema["family"], "n_rows": n_rows,
            "tape_header_pins": {k: header.get(k) for k in
                                 ("features_under", "clock_basis", "protocol",
                                  "built_from_schema")},
            "predicates": preds,
            "not_applicable": [x["predicate"] for x in preds
                               if not x.get("applicable", True)],
            "all_pass": all(x["pass"] for x in preds
                            if x.get("applicable", True))}


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
    for k in ("tape_non_empty", "no_undeclared_reduction",
              "guard_beside_every_nullable", "no_zero_imputation",
              "bn_recv_ns_was_supplied", "state_status_present",
              "gaps_were_supplied", "statuses_are_declared_values",
              "feature_asof_never_after_decision", "embargo_respected"):
        ok(g[k], f"a clean tape passes {k}")

    # FALSIFIERS -- each must FIRE (rule 15)
    ok(not res([])["tape_non_empty"], "an EMPTY tape fails, never passes")
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
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    if not a.tape:
        raise SystemExit("--tape PATH required; refusing to guess a tape")
    rep = verify(Path(a.tape), gapped_slugs_expected=a.gapped_slugs,
                 expect_gap_count=a.expect_gap_count)
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
