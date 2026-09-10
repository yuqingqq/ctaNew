#!/usr/bin/env python3
"""EVERY NAMED REFUSAL MUST BE EXERCISED. A REPOSITORY INSTRUMENT.

Built to REV 168's specification (DA 200), superseding DA 198's
`p003_refusal_coverage`, which had TWO defects this one exists to close:

  * IT SAW ONE IDIOM. It matched inline `REFUSED <NAME>` literals only and was
    blind to `NAME = "SOME_REFUSAL"` referenced inside a `raise`. REV measured
    the split at 14/24, so either idiom alone misses more than half.
  * IT SILENTLY EXCLUDED THE UNNAMED ONES. Its regex floor dropped
    `REFUSED DAY {day}: ...` sites on the grounds that they were "sentences,
    not tokens" -- 29 of them in `de_multiday_gate1_runner.py` alone -- and a
    dropped population reads as a clean one. They are now a NAMED STATUS.

THE DECLARED LIMIT, REPEATED ON EVERY OUTPUT AND NOT ONLY THE FIRST: the
predicate is TEXTUAL. A cell that drives a refusal without naming it -- a bare
`except`, a message fragment, a call through a helper -- reads as
never-exercised. **THE ERROR INFLATES RATHER THAN DEFLATES.** That is the safe
direction for a floor, and the count is a FLOOR ON WHAT HAS BEEN SEEN TO BE
DRIVEN, never an exact defect count.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOTS = (HERE,)

NEVER_EXERCISED = "REFUSAL_NEVER_EXERCISED"
HAS_NO_NAME = "REFUSAL_HAS_NO_NAME"
NO_REFUSALS = "EXAMINED_NO_REFUSALS_FOUND"

#: REV 168 idiom (2). `-{0,2}` admits `REFUSED -- NAME`, which this repo writes.
INLINE = re.compile(r"REFUSED\s*-{0,2}\s*([A-Z][A-Z0-9_]{5,})")
#: REV 168 idiom (1). A module-level constant whose VALUE is a refusal name.
CONST_VALUE = re.compile(r"^[A-Z][A-Z0-9_]{5,}$")
#: The exercised set: selftest | falsify | _falsify | *_cells | test_*
EXERCISER = re.compile(r"(^|_)(selftest|falsify)([_0-9]|$)|_cells$|^test_", re.I)

LIMIT = ("TEXTUAL PREDICATE. A cell that drives a refusal WITHOUT NAMING it "
         "reads as never-exercised, so THE ERROR INFLATES RATHER THAN "
         "DEFLATES. This is a FLOOR on what has been SEEN to be driven, never "
         "an exact defect count.")


#: DA 201 / REV 168. THE DISCRIMINATOR IS POSITION, NOT PRESENCE.
#: REV proposed separating a refusal token from a protocol constant by whether
#: it appears "in the message". MEASURED, THAT DOES NOT SEPARATE THEM: at
#: `da_early_read_verify.py:411` the message is
#:     f"NOT_AN_EARLY_READ_ARTIFACT: {name} carries protocol "
#:     f"{doc.get('protocol')!r}, not {EARLY_PROTOCOL!r}."
#: -- the refusal token is an inline literal AND `EARLY_PROTOCOL` is also in
#: the message, interpolated as a VALUE.
#:
#: What separates them is WHERE: a refusal token LEADS the message (optionally
#: behind a bare `REFUSED` prefix); a protocol constant is interpolated
#: somewhere inside it. `f"action protocol must be {ACTION_PROTOCOL!r}"`
#: (`de_v2_acting_matched_control.py:138`) starts with lowercase prose, so its
#: constant is not a refusal name.
#:
#: THIS IS NOT THE REGEX TIGHTENED UNTIL THE COUNT LOOKED RIGHT -- the failure
#: this tool exists to prevent. It is a structural predicate, and the tokens
#: it removes are ENUMERATED in the falsifier so the change is reviewable
#: rather than merely smaller.
_LEAD_OK = re.compile(r"^\s*(REFUSED)?\s*-{0,2}\s*$")


def _name_resolved_at_runtime(raise_node, consts) -> bool:
    """Does this raise interpolate a NON-LITERAL in the leading slot?

    IDIOM (3). If the module declares refusal-name constants at all AND the
    message's first interpolated slot is an expression rather than a name or
    a literal, the refusal is very likely named from data. The checker cannot
    resolve it and must not pretend either way."""
    if not consts:
        return False
    for arg in _message_exprs(raise_node):
        if not isinstance(arg, ast.JoinedStr):
            continue
        seen = ""
        for part in arg.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                seen += part.value
                if not _LEAD_OK.match(seen):
                    break
                continue
            if isinstance(part, ast.FormattedValue):
                v = part.value
                # DA 208b: an ATTRIBUTE (`R.SOME_REFUSAL`) is a CROSS-MODULE
                # CONSTANT and is RESOLVABLE -- "I did not look" is not the
                # same claim as "I cannot tell", and only the second earns
                # this bucket. A SUBSCRIPT or a CALL is genuinely computed
                # from data and is undecidable textually.
                if _LEAD_OK.match(seen) and isinstance(
                        v, (ast.Subscript, ast.Call)):
                    return True
            break
    return False


def _cross_module_constant(raise_node, all_consts):
    """A leading `MODULE.SOME_REFUSAL` resolved against every module's
    constants -- idiom (1) extended across the import boundary."""
    out = []
    for arg in _message_exprs(raise_node):
        if not isinstance(arg, ast.JoinedStr):
            continue
        seen = ""
        for part in arg.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                seen += part.value
                if not _LEAD_OK.match(seen):
                    break
                continue
            if isinstance(part, ast.FormattedValue):
                v = part.value
                if _LEAD_OK.match(seen) and isinstance(v, ast.Attribute) \
                        and v.attr in all_consts:
                    out.append(all_consts[v.attr])
            break
    return out


def _leading_constants(raise_node, consts):
    """Module constants that LEAD the raise's message expression."""
    out = []
    for arg in _message_exprs(raise_node):
        if isinstance(arg, ast.Name) and arg.id in consts:
            out.append(arg.id)                     # raise X(SOME_REFUSAL)
            continue
        if not isinstance(arg, ast.JoinedStr):
            continue
        seen_text = ""
        for part in arg.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                seen_text += part.value
                if not _LEAD_OK.match(seen_text):
                    break                          # real prose came first
                continue
            if isinstance(part, ast.FormattedValue):
                v = part.value
                if isinstance(v, ast.Name) and v.id in consts \
                        and _LEAD_OK.match(seen_text):
                    out.append(v.id)
                break                              # only the FIRST slot counts
            break
    return out


def _message_exprs(raise_node):
    """The expressions that become the exception's message."""
    exc = raise_node.exc
    if exc is None:
        return []
    if isinstance(exc, ast.Call):
        args = list(exc.args)
    else:
        args = [exc]
    out = []
    for a in args:
        # an implicitly concatenated f-string is a BinOp/JoinedStr chain
        if isinstance(a, ast.JoinedStr):
            out.append(a)
        elif isinstance(a, ast.Name):
            out.append(a)
        elif isinstance(a, ast.BinOp):
            # DA 201: THIS WAS NOT RECURSIVE AND IT SILENTLY DROPPED REAL
            # TOKENS. `f"REFUSED {X}: ..." + tail + f" | {LIMIT}"` parses
            # left-associative, so `a.left` is itself a BinOp and the
            # JoinedStr holding the token sits one level deeper. TWO real
            # refusals vanished from the population -- including this
            # module's own REFUSAL_NEVER_EXERCISED -- and the COUNT alone
            # would have looked like a clean improvement. Caught only by
            # enumerating WHICH tokens moved.
            out += _flatten_binop(a)
    return out


def _flatten_binop(node):
    out = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.BinOp):
            stack += [cur.left, cur.right]
        elif isinstance(cur, (ast.JoinedStr, ast.Name)):
            out.append(cur)
        elif isinstance(cur, ast.IfExp):
            stack += [cur.body, cur.orelse]
    return out


def is_exerciser(name: str) -> bool:
    return bool(EXERCISER.search(name or ""))


def _module_refusal_constants(tree, src):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str) \
                and CONST_VALUE.match(node.value.value):
            out[node.targets[0].id] = node.value.value
    return out


def scan(roots=DEFAULT_ROOTS) -> dict:
    files = []
    for r in roots:
        files += sorted(p for p in Path(r).rglob("*.py") if p.is_file())

    per_file, unparsable, runtime_named = {}, [], []
    # DA 208b: every module's refusal constants, so a leading
    # `OTHERMODULE.SOME_REFUSAL` resolves instead of being called undecidable.
    all_consts = {}
    for p in files:
        try:
            all_consts.update(_module_refusal_constants(ast.parse(p.read_text()),
                                                        p.read_text()))
        except Exception:
            pass
    population = {}          # token -> {files, idioms}
    unnamed = []             # (file, line, excerpt)
    const_names = {}         # token -> set of constant identifiers

    for p in files:
        try:
            src = p.read_text()
            tree = ast.parse(src)
        except Exception as e:
            unparsable.append({"file": str(p), "error": type(e).__name__})
            continue
        consts = _module_refusal_constants(tree, src)
        n_raise = n_tok = n_unnamed = n_runtime = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise):
                continue
            seg = ast.get_source_segment(src, node) or ""
            n_raise += 1
            toks = set(INLINE.findall(seg))                       # idiom (2)
            for cname in _leading_constants(node, consts):        # idiom (1)
                toks.add(consts[cname])
                const_names.setdefault(consts[cname], set()).add(cname)
            for val in _cross_module_constant(node, all_consts):  # idiom (1b)
                toks.add(val)
            if toks:
                n_tok += len(toks)
                for t in toks:
                    e = population.setdefault(t, {"files": set(), "idioms": set()})
                    e["files"].add(p.name)
                    e["idioms"].add("inline" if t in INLINE.findall(seg)
                                    else "named_constant")
            elif "REFUSED" in seg and _name_resolved_at_runtime(node, consts):
                # DA 208, IDIOM (3): the refusal NAME is computed at run time.
                # `be_reserved_days.py:141` -- raise R(f"REFUSED
                # {v['refusal_name']}: {v['why']}") -- is NAMED, dynamically,
                # from a dict whose values are this module's own declared
                # refusal constants. A TEXTUAL predicate cannot read it and
                # MUST NOT call it unnamed: that is a false alarm, and it fired
                # on BE 161 within a minute of the ratchet going live.
                # It gets its OWN bucket rather than being forced into either
                # of the other two, because the checker genuinely CANNOT TELL.
                n_runtime += 1
                runtime_named.append({"file": p.name,
                                      "line": getattr(node, "lineno", None),
                                      "excerpt": " ".join(seg.split())[:110]})
            elif "REFUSED" in seg:
                n_unnamed += 1
                unnamed.append({"file": p.name,
                                "line": getattr(node, "lineno", None),
                                "excerpt": " ".join(seg.split())[:110]})
        per_file[p.name] = {"n_raise_sites": n_raise,
                            "n_named_refusals": n_tok,
                            "n_unnamed_refusals": n_unnamed,
                            "n_runtime_named_refusals": n_runtime,
                            "status": (NO_REFUSALS if n_raise == 0
                                       else "EXAMINED")}

    # ---- exercised set, decided only after the population is complete
    exercised, exerciser_fns = set(), 0
    for p in files:
        try:
            src = p.read_text()
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and is_exerciser(node.name):
                exerciser_fns += 1
                seg = ast.get_source_segment(src, node) or ""
                for t in population:
                    if t in seg or any(c in seg for c in const_names.get(t, ())):
                        exercised.add(t)

    never = sorted(t for t in population if t not in exercised)
    out = {
        "n_files_enumerated": len(files),
        "n_unparsable": len(unparsable), "unparsable": unparsable,
        "POPULATION": len(population),
        "population_by_idiom": {
            "inline": sum(1 for v in population.values() if "inline" in v["idioms"]),
            "named_constant": sum(1 for v in population.values()
                                  if "named_constant" in v["idioms"])},
        "n_exercised": len(exercised),
        "n_never_exercised": len(never),
        "never_exercised": [{"token": t,
                             "files": sorted(population[t]["files"])}
                            for t in never],
        "REFUSAL_HAS_NO_NAME": {"n": len(unnamed), "sites": unnamed[:40],
                                "n_shown": min(40, len(unnamed))},
        "REFUSAL_NAME_RESOLVED_AT_RUNTIME": {
            "n": len(runtime_named), "sites": runtime_named[:20],
            "what_it_is": ("the refusal NAME is computed at run time from "
                           "data, so a TEXTUAL predicate cannot read it. NOT "
                           "unnamed and NOT verifiably named -- the checker "
                           "cannot tell, and says so rather than guessing."),
            "why_it_has_its_own_bucket": ("forcing it into `unnamed` is a "
                                          "FALSE ALARM; forcing it into "
                                          "`named` is a false clean. Neither "
                                          "is honest, so it is neither.")},
        "n_exerciser_functions": exerciser_fns,
        "files_with_no_refusals": sorted(
            f for f, v in per_file.items() if v["status"] == NO_REFUSALS),
        "per_file": per_file,
        "DECLARED_LIMIT": LIMIT,
    }
    out["verdict"] = (
        "UNPARSABLE_FILES_PRESENT" if unparsable else
        NEVER_EXERCISED if never else
        HAS_NO_NAME if unnamed else "EXERCISE_FLOOR_HELD")
    return out


def assert_floor(roots=DEFAULT_ROOTS, strict: bool = True) -> dict:
    """REFUSES, publishing the POPULATION COUNT beside every failure.

    REV 168 criterion 4: a checker reporting '0 never-exercised' out of a
    population of 0 is the `BINANCE_GAP_EXCLUDED: 0` shape -- a zero with no
    denominator beside it."""
    r = scan(roots)
    if r["unparsable"]:
        raise AssertionError(
            f"REFUSED UNPARSABLE_FILES_PRESENT: {r['n_unparsable']} file(s) "
            f"unparsable; their refusals are UNKNOWN, not clean. "
            f"POPULATION={r['POPULATION']}. {LIMIT}")
    if r["never_exercised"]:
        listed = "; ".join(f"{u['files'][0]}:{u['token']}"
                           for u in r["never_exercised"][:10])
        raise AssertionError(
            f"REFUSED {NEVER_EXERCISED}: {r['n_never_exercised']} of "
            f"POPULATION={r['POPULATION']} named refusals appear in no "
            f"exerciser: {listed}"
            + (" ..." if r["n_never_exercised"] > 10 else "")
            + f" | {LIMIT}")
    if r["REFUSAL_HAS_NO_NAME"]["n"] and strict:
        # REV 168 criterion 3. `strict=False` gates ONLY on the actionable
        # subset -- named refusals never exercised -- and is the mode a build
        # can hold green today. IT IS NOT A WEAKER SPEC: the unnamed count
        # still travels on every output, and a gate nobody can ever turn green
        # is a gate people route around, which is how a class goes quiet.
        raise AssertionError(
            f"REFUSED {HAS_NO_NAME}: {r['REFUSAL_HAS_NO_NAME']['n']} raise "
            f"site(s) say REFUSED and carry NO TOKEN, so they cannot be "
            f"exercised by name. POPULATION={r['POPULATION']}. {LIMIT}")
    return r



# ------------------------------------------------------- THE RATCHET
#
# REV 168, second round: `--named-only` will become the default invocation,
# and then the 1,100-odd unnamed refusals go quiet by a different door. The
# fix for that shape is a RATCHET -- the existing ones are grandfathered, a
# NEW one is blocked at the commit that adds it, and the number can only move
# DOWN. A gate that is green today, cannot be routed around, and makes the
# class shrink monotonically instead of sitting permanently amber.
#
# PER FILE, not just the total: a single total lets a new unnamed refusal in
# file A hide behind a deletion in file B, and hiding inside an aggregate is
# the shape this programme keeps paying for.

BASELINE = HERE / "declarations" / "p003_unnamed_refusal_baseline_v1.json"
RATCHET_ROSE = "UNNAMED_REFUSAL_COUNT_ROSE"


def write_baseline(roots=DEFAULT_ROOTS, path=None) -> dict:
    r = scan(roots)
    doc = {"protocol": "P003_UNNAMED_REFUSAL_BASELINE_V1",
           "what_this_is": ("a RATCHET, not a target. Existing unnamed "
                            "refusals are grandfathered; a NEW one fails at "
                            "the commit that adds it; the number may only "
                            "move DOWN."),
           "why_per_file": ("a single total lets a new unnamed refusal in one "
                            "file hide behind a deletion in another"),
           "how_to_lower_it": ("give the refusal a NAME, re-run "
                               "`--write-baseline`, and commit the lower "
                               "number with the change that earned it"),
           "never_raise_it": ("a baseline raised to accommodate new unnamed "
                              "refusals is the ratchet disarmed. If a rise is "
                              "deliberate it is a USER/coordinator decision "
                              "and belongs in the register, not in a quiet "
                              "regeneration."),
           "DECLARED_LIMIT": LIMIT,
           "total": r["REFUSAL_HAS_NO_NAME"]["n"],
           "per_file": {f: v["n_unnamed_refusals"]
                        for f, v in sorted(r["per_file"].items())
                        if v["n_unnamed_refusals"]}}
    Path(path or BASELINE).write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def assert_ratchet(roots=DEFAULT_ROOTS, baseline=None) -> dict:
    """REFUSES if the unnamed-refusal count ROSE in ANY file."""
    bpath = Path(baseline or BASELINE)
    if not bpath.is_file():
        raise AssertionError(
            f"REFUSED {RATCHET_ROSE}: no baseline at {bpath}. An absent "
            f"baseline is not a pass -- write one with --write-baseline.")
    base = json.loads(bpath.read_text())
    r = scan(roots)
    now = {f: v["n_unnamed_refusals"] for f, v in r["per_file"].items()
           if v["n_unnamed_refusals"]}
    rose = {f: (base["per_file"].get(f, 0), n) for f, n in now.items()
            if n > base["per_file"].get(f, 0)}
    if rose:
        detail = "; ".join(f"{f} {was}->{is_}" for f, (was, is_) in
                           sorted(rose.items())[:8])
        raise AssertionError(
            f"REFUSED {RATCHET_ROSE}: unnamed refusals ROSE in "
            f"{len(rose)} file(s): {detail}"
            + (" ..." if len(rose) > 8 else "")
            + f" | baseline total {base['total']}, now "
            f"{r['REFUSAL_HAS_NO_NAME']['n']} | {LIMIT}")
    fell = sum(base["per_file"].get(f, 0) - now.get(f, 0)
               for f in base["per_file"])
    return {"verdict": "RATCHET_HELD",
            "baseline_total": base["total"],
            "now_total": r["REFUSAL_HAS_NO_NAME"]["n"],
            "net_reduction_since_baseline": fell,
            "n_files_at_or_below_baseline": len(now),
            "POPULATION": r["POPULATION"], "DECLARED_LIMIT": LIMIT}


# ------------------------------------------------------------- falsifier

_N = {"n": 0, "bad": 0}


def _ok(cond, label):
    _N["n"] += 1
    if not cond:
        _N["bad"] += 1
        print(f"  FAIL {label}")
    return cond


def selftest(quiet: bool = False) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ---- CELL 1, POSITIVE CONTROL: a named refusal with no cell -> FLAG
        (tmp / "a_named_no_cell.py").write_text(
            'def go(x):\n'
            '    if not x:\n'
            '        raise ValueError("REFUSED FIXTURE_ALPHA_TOKEN: nope")\n')
        r1 = scan((tmp,))
        _ok(r1["POPULATION"] == 1 and r1["n_never_exercised"] == 1
            and r1["never_exercised"][0]["token"] == "FIXTURE_ALPHA_TOKEN"
            and r1["verdict"] == NEVER_EXERCISED,
            f"CELL 1 positive control: a named refusal with no cell is FLAGGED "
            f"and NAMED (POPULATION={r1['POPULATION']})")

        # ---- CELL 2, KNOWN-BAD: the same token inside a selftest -> NOT flagged
        (tmp / "a_named_with_cell.py").write_text(
            'def selftest():\n'
            '    assert "FIXTURE_ALPHA_TOKEN"\n')
        r2 = scan((tmp,))
        _ok(r2["POPULATION"] == 1 and r2["n_never_exercised"] == 0,
            "CELL 2 known-bad: the same refusal named in a selftest is NOT "
            "flagged -- the check does not simply flag everything")

        # ---- CELL 3, THE SILENT-REGEX CONTROL. The one that matters.
        (tmp / "a_named_with_cell.py").unlink()
        (tmp / "a_named_no_cell.py").unlink()
        (tmp / "b_unmatched_spelling.py").write_text(
            'def go(day):\n'
            '    raise ValueError(f"REFUSED DAY {day}: the book moved")\n')
        r3 = scan((tmp,))
        _ok(r3["POPULATION"] == 0
            and r3["REFUSAL_HAS_NO_NAME"]["n"] == 1
            and r3["verdict"] == HAS_NO_NAME,
            f"CELL 3 SILENT-REGEX CONTROL: a refusal the token regex does NOT "
            f"match is reported under {HAS_NO_NAME} -- NOT returned as a clean "
            f"zero (verdict={r3['verdict']}, population={r3['POPULATION']})")
        try:
            assert_floor((tmp,))
            fired = False
            msg = ""
        except AssertionError as e:
            fired, msg = True, str(e)
        _ok(fired and HAS_NO_NAME in msg and "POPULATION=0" in msg,
            "CELL 3: assert_floor REFUSES on it AND publishes POPULATION=0 "
            "beside the failure -- a zero never travels without its "
            "denominator (criterion 4)")

        # ---- IDIOM (1): a named CONSTANT referenced in a raise
        (tmp / "b_unmatched_spelling.py").unlink()
        (tmp / "c_const_idiom.py").write_text(
            'FIXTURE_CONST_REFUSAL = "FIXTURE_GAMMA_TOKEN"\n'
            '\n'
            'def go(x):\n'
            '    if not x:\n'
            '        raise ValueError(f"REFUSED {FIXTURE_CONST_REFUSAL}: no")\n')
        r4 = scan((tmp,))
        _ok(r4["POPULATION"] == 1
            and r4["never_exercised"][0]["token"] == "FIXTURE_GAMMA_TOKEN"
            and r4["population_by_idiom"]["named_constant"] == 1,
            f"IDIOM (1): a refusal held in a module CONSTANT and referenced in "
            f"a raise is in the population -- the idiom DA 198 was blind to "
            f"({r4['population_by_idiom']})")
        (tmp / "c_cell.py").write_text(
            'def falsify_gamma():\n'
            '    assert "FIXTURE_CONST_REFUSAL"\n')
        r5 = scan((tmp,))
        _ok(r5["n_never_exercised"] == 0,
            "IDIOM (1): naming the CONSTANT in a `falsify_*` cell exercises "
            "it -- the exerciser set is REV 168's, not DA 198's narrower one")

        # ---- CRITERION 5: show it examined its target.
        (tmp / "d_no_refusals.py").write_text('def plain():\n    return 1\n')
        r6 = scan((tmp,))
        _ok("d_no_refusals.py" in r6["files_with_no_refusals"]
            and r6["per_file"]["d_no_refusals.py"]["status"] == NO_REFUSALS
            and r6["per_file"]["c_const_idiom.py"]["n_raise_sites"] >= 1,
            f"CRITERION 5: a file yielding zero raise-sites is reported as "
            f"{NO_REFUSALS} rather than silently contributing 0/0, and a file "
            f"that WAS examined publishes its raise-site count")

        # ---- PARTIAL INPUT
        (tmp / "e_broken.py").write_text('def (:\n')
        r7 = scan((tmp,))
        _ok(r7["n_unparsable"] == 1
            and r7["verdict"] == "UNPARSABLE_FILES_PRESENT",
            "PARTIAL INPUT: an unparsable file makes the verdict UNKNOWN and "
            "is NAMED -- a failed parse is never read as coverage")

    # ---- IDIOM (3) AND (1b), DA 208. Driven, because the ratchet's FIRST
    # ---- LIVE FIRING was a FALSE ALARM on one of them.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "rt.py").write_text(
            'SOME_REFUSAL = "FIXTURE_RUNTIME_TOKEN"\n'
            'def go(v):\n'
            '    raise ValueError(f"REFUSED {v[\'refusal_name\']}: {v[\'why\']}")\n')
        r_rt = scan((tmp,))
        _ok(r_rt["REFUSAL_NAME_RESOLVED_AT_RUNTIME"]["n"] == 1
            and r_rt["REFUSAL_HAS_NO_NAME"]["n"] == 0,
            "IDIOM (3): a refusal NAMED FROM DATA lands in its OWN bucket -- "
            "NOT unnamed (a false alarm) and NOT named (a false clean). The "
            "checker cannot tell and says so.")
        (tmp / "other.py").write_text('FAR_REFUSAL = "FIXTURE_CROSS_TOKEN"\n')
        (tmp / "xm.py").write_text(
            'import other as O\n'
            'def go():\n'
            '    raise ValueError(f"REFUSED {O.FAR_REFUSAL}: no")\n')
        r_xm = scan((tmp,))
        _ok("FIXTURE_CROSS_TOKEN" in {u["token"] for u in r_xm["never_exercised"]}
            and r_xm["REFUSAL_NAME_RESOLVED_AT_RUNTIME"]["n"] == 1,
            "IDIOM (1b): a CROSS-MODULE constant RESOLVES to a named refusal "
            "-- 'I did not look' is not the same claim as 'I cannot tell', "
            "and only the second earns the runtime bucket")

    # ---- THE RATCHET, driven both ways (REV 168 second round)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "r_one.py").write_text(
            'def go(d):\n    raise ValueError(f"REFUSED DAY {d}: one")\n')
        bfile = tmp / "baseline.json"
        b = write_baseline((tmp,), bfile)
        _ok(b["total"] == 1 and b["per_file"]["r_one.py"] == 1,
            f"RATCHET baseline records the count PER FILE ({b['per_file']})")
        _ok(assert_ratchet((tmp,), bfile)["verdict"] == "RATCHET_HELD",
            "RATCHET positive control: unchanged surface HOLDS")
        # a NEW unnamed refusal in the SAME file -> must refuse
        (tmp / "r_one.py").write_text(
            'def go(d):\n    raise ValueError(f"REFUSED DAY {d}: one")\n'
            'def go2(d):\n    raise ValueError(f"REFUSED DAY {d}: two")\n')
        try:
            assert_ratchet((tmp,), bfile)
            fired_r = False
            msg_r = ""
        except AssertionError as e:
            fired_r, msg_r = True, str(e)
        _ok(fired_r and RATCHET_ROSE in msg_r and "1->2" in msg_r,
            "RATCHET known-bad: a NEW unnamed refusal REFUSES and names the "
            "file and the rise")
        # a new unnamed refusal in a DIFFERENT file, offset by a deletion in
        # the first -- the total is unchanged and the ratchet must STILL fire
        (tmp / "r_one.py").write_text('def go():\n    return 1\n')
        (tmp / "r_two.py").write_text(
            'def go(d):\n    raise ValueError(f"REFUSED DAY {d}: elsewhere")\n')
        try:
            assert_ratchet((tmp,), bfile)
            fired_h = False
            msg_h = ""
        except AssertionError as e:
            fired_h, msg_h = True, str(e)
        _ok(fired_h and "r_two.py 0->1" in msg_h,
            "RATCHET, THE CASE A TOTAL WOULD MISS: one file drops to 0 while "
            "another gains one, TOTAL UNCHANGED -- and it still REFUSES, "
            "because hiding inside an aggregate is the shape being prevented")
        # and it may move DOWN freely
        (tmp / "r_two.py").unlink()
        got = assert_ratchet((tmp,), bfile)
        _ok(got["verdict"] == "RATCHET_HELD"
            and got["net_reduction_since_baseline"] == 1,
            f"RATCHET: the number may move DOWN freely "
            f"(net reduction {got['net_reduction_since_baseline']})")
        # an ABSENT baseline is not a pass
        bfile.unlink()
        try:
            assert_ratchet((tmp,), bfile)
            fired_a = False
        except AssertionError as e:
            fired_a = RATCHET_ROSE in str(e)
        _ok(fired_a, "RATCHET: an ABSENT baseline REFUSES -- it is not a pass")

    real = scan()
    _ok(real["DECLARED_LIMIT"] == LIMIT,
        "the declared limit ships ON THE OUTPUT, every output, not only the "
        "first (REV 168)")
    _ok(real["POPULATION"] > 0,
        f"the real surface has a NON-ZERO population "
        f"({real['POPULATION']}) -- a zero here would be the "
        f"BINANCE_GAP_EXCLUDED: 0 shape")

    if not quiet:
        print(f"[p003_refusal_exercise_check] {_N['n'] - _N['bad']}/{_N['n']} "
              f"checks, {_N['bad']} failures | POPULATION={real['POPULATION']} "
              f"(inline {real['population_by_idiom']['inline']} / const "
              f"{real['population_by_idiom']['named_constant']}), "
              f"exercised {real['n_exercised']}, NEVER EXERCISED "
              f"{real['n_never_exercised']}, {HAS_NO_NAME} "
              f"{real['REFUSAL_HAS_NO_NAME']['n']}")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    if "--write-baseline" in sys.argv:
        b = write_baseline()
        print("baseline written: total=%d across %d files"
              % (b["total"], len(b["per_file"])))
    elif "--ratchet" in sys.argv:
        try:
            print(json.dumps(assert_ratchet(), indent=1))
        except AssertionError as e:
            print(e)
            sys.exit(1)
    elif "--assert" in sys.argv:
        try:
            r = assert_floor(strict="--named-only" not in sys.argv)
            print("EXERCISE_FLOOR_HELD  POPULATION=%d" % r["POPULATION"])
        except AssertionError as e:
            print(e)
            sys.exit(1)
    else:
        r = scan()
        r.pop("per_file", None)
        print(json.dumps(r, indent=1, default=str)[:6000])
