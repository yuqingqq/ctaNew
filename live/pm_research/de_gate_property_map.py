"""THE PROPERTY-TO-CELL MAP, AS AN ARTIFACT A DRIVER RESOLVES.

REVIEW 203: my property-to-cell map lived in a COMMIT MESSAGE, so nothing
resolved it and nothing drove it. A map nobody runs is prose beside a
table -- the shape rule 10 names -- and the class it was written to answer
(`CELLS_PASSING_IS_NOT_PROPERTIES_COVERED`) had already appeared INSIDE
it: "every other non-fair-value input" was mapped to a cell that covered
only dataclass fields, while the action list was an input and not a field.

So the map is data here, and `resolve()` REFUSES:

  * `PROPERTY_MAPS_TO_NOTHING`      -- a declared property with no cell;
  * `MAPPED_CELL_DOES_NOT_EXIST`    -- a cell name no falsifier emits;
  * `MAPPED_CELL_DID_NOT_PASS`      -- it exists and is red;
  * `UNIVERSAL_PROPERTY_ENUMERATED_COVER` -- a property quantified over
    "every X" mapped to a cell whose coverage is ENUMERATED rather than
    DERIVED. That is the fourth-instance rule made mechanical: a universal
    claim cannot be covered by a hand-listed cell, because the list is
    exactly what goes stale.

Usage:  de_gate_property_map.py            -> resolve, print the table
        de_gate_property_map.py --falsify  -> the driver's own controls
Exit:   0 every property covered, 3 a refusal, 4 a falsifier absent.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "P003_DE_GATE_PROPERTY_MAP_V1"

NOTHING = "PROPERTY_MAPS_TO_NOTHING"
NO_CELL = "MAPPED_CELL_DOES_NOT_EXIST"
RED_CELL = "MAPPED_CELL_DID_NOT_PASS"
ENUMERATED = "UNIVERSAL_PROPERTY_ENUMERATED_COVER"
NO_FALSIFIER = "GATE_MODULE_HAS_NO_FALSIFIER"
#: REVIEW 204's recorded limitation: `universal` and `coverage` are
#: AUTHOR-SET, so the narrowness check fires on a label -- relabelling the
#: genuine universal line `universal: False` made its refusal disappear
#: while the map still resolved. Not fully removable (some declaration of
#: what a property quantifies over is unavoidable), so it is MITIGATED the
#: way REV suggests: a property whose TEXT quantifies over an open set
#: must be marked universal, checked mechanically against the text.
MISLABELLED = "QUANTIFIED_PROPERTY_NOT_MARKED_UNIVERSAL"
QUANTIFIERS = ("every", " any ", "all ", "each ")

#: gate -> module, then one entry per DECLARED property.
#: `universal` marks a property quantified over an open set ("every other
#: non-fair-value input"); such a property requires `coverage: derived`.
MAP = {
    "5.3 estimator wrapper": {
        "module": "de_fair_price_wrapper.py",
        "properties": [
            {"property": "source-event and local-knowledge time stay "
                         "distinct at every hop",
             "cell": "a COLLAPSED pair (source == local) REFUSES",
             "coverage": "derived", "universal": True},
            {"property": "an inverted pair is its own fault",
             "cell": "an INVERTED pair refuses under its OWN name",
             "coverage": "enumerated"},
            {"property": "C1 consumes Identity's admissibility decision",
             "cell": "C1 REFUSES if its admissibility ever diverges",
             "coverage": "derived", "universal": True},
            {"property": "C2 consumes X60(t0) only after local receipt",
             "cell": "a FUTURE-KNOWLEDGE reference REFUSES",
             "coverage": "enumerated"},
            {"property": "the estimator never substitutes Identity",
             "cell": "and the refusal is NOT a substitution of Identity",
             "coverage": "enumerated"},
            {"property": "DOWN is the mechanical complement",
             "cell": "DOWN is the MECHANICAL complement",
             "coverage": "enumerated"},
            {"property": "a sign flip is detected",
             "cell": "a UP/DOWN SIGN FLIP is DETECTED",
             "coverage": "enumerated"},
            {"property": "the two regimes are carried, not assumed",
             "cell": "the point-in-time record carries its regime",
             "coverage": "enumerated"},
        ]},
    "5.4 canonical actions + scorer": {
        "module": "de_fair_value_actions.py",
        "properties": [
            {"property": "one row per consumption decision",
             "cell": "two quote sides consuming ONE value at one key are "
                     "ONE action", "coverage": "enumerated"},
            {"property": "duplicate keys refuse the build",
             "cell": "two VALUES at one key REFUSE the build",
             "coverage": "enumerated"},
            {"property": "membership on the neutral reference path is "
                         "DERIVED for every row",
             "cell": "a row whose FLAG contradicts the population REFUSES",
             "coverage": "derived", "universal": True},
            {"property": "the population is attributable",
             "cell": "an ANONYMOUS population refuses",
             "coverage": "enumerated"},
            {"property": "one epsilon clips both sides",
             "cell": "both sides are clipped by the SAME epsilon",
             "coverage": "enumerated"},
            {"property": "DOWN is carried only as the complement",
             "cell": "DOWN is never scored separately",
             "coverage": "enumerated"},
            {"property": "a non-OK challenger is scored on the Identity "
                         "fallback",
             "cell": "a NON-OK challenger is scored on the POLICY'S "
                     "IDENTITY FALLBACK", "coverage": "enumerated"},
            {"property": "a non-OK Identity counts and scores neither",
             "cell": "when IDENTITY is non-OK the action is COUNTED",
             "coverage": "enumerated"},
            {"property": "no labelled score is readable",
             "cell": "an action with NO SUPPLIED OUTCOME refuses",
             "coverage": "enumerated"},
        ]},
    "5.5 policy seam": {
        "module": "de_fair_value_policy_seam.py",
        "properties": [
            {"property": "Identity substitution is bit-identical",
             "cell": "substituting Identity for Identity is BIT-IDENTICAL",
             "coverage": "enumerated"},
            {"property": "a non-Identity value moves the quote anchor",
             "cell": "a NON-IDENTITY value MOVES the quote anchor",
             "coverage": "enumerated"},
            {"property": "an absent challenger falls back and is counted",
             "cell": "an ABSENT challenger falls back to Identity, and "
                     "the fallback is COUNTED", "coverage": "enumerated"},
            {"property": "a metadata-only change is decorative",
             "cell": "changing ONLY the estimator metadata is DETECTED",
             "coverage": "enumerated"},
        ]},
    "5.6 replay seam": {
        "module": "de_fair_value_replay_seam.py",
        "properties": [
            {"property": "shared non-fair-value parameters",
             "cell": "a differing non_fair_value_params REFUSES",
             "coverage": "enumerated"},
            {"property": "shared input snapshot",
             "cell": "a challenger on a FLAT 0.99 TAPE is REFUSED",
             "coverage": "enumerated"},
            {"property": "shared initial state",
             "cell": "a differing initial_state REFUSES",
             "coverage": "enumerated"},
            {"property": "EVERY OTHER non-fair-value input is shared",
             "cell": "a NEW field is digested with NO edit to any list",
             "coverage": "derived", "universal": True},
            {"property": "the action population is a shared input",
             "cell": "two arms on DIFFERENT ACTION POPULATIONS are REFUSED",
             "coverage": "enumerated"},
            {"property": "both legs keep their own resulting order paths",
             "cell": "while the legs remain FREE to produce different "
                     "order paths", "coverage": "enumerated"},
            {"property": "a pinned outcome path is refused",
             "cell": "a replay that PINS the outcome path is REFUSED",
             "coverage": "enumerated"},
            {"property": "the seam is not over-constrained",
             "cell": "and the real-tape pair still PASSES",
             "coverage": "enumerated"},
        ]},
}

CELL = re.compile(r"^\s*\[(PASS|FAIL)\]\s+(.*?)(?:\s\s+|$)")


class MapRefused(ValueError):
    """A property is uncovered, or covered by something narrower."""


def drive(module: str, tree: Path = HERE) -> dict:
    """Run a gate's falsifier and return {cell text: passed}."""
    f = Path(tree) / module
    if not f.is_file():
        raise MapRefused(f"REFUSED {NO_FALSIFIER}: {module} is not at "
                         f"{tree}")
    r = subprocess.run([sys.executable, str(f), "--falsify"],
                       capture_output=True, text=True, cwd="/tmp",
                       env={"PATH": "/usr/bin:/bin",
                            "HOME": "/home/yuqing",
                            "DE_VALUATION_PREFLIGHT_OFF": "1"})
    cells = {}
    for line in r.stdout.splitlines():
        m = CELL.match(line)
        if m:
            cells[m.group(2).strip()] = (m.group(1) == "PASS")
    if not cells:
        raise MapRefused(
            f"REFUSED {NO_FALSIFIER}: {module} emitted no cell lines "
            f"(rc={r.returncode}). A gate whose falsifier does not run "
            f"cannot cover any property.")
    return cells


def resolve(mapping=None, tree: Path = HERE) -> dict:
    """Resolve EVERY entry, refusing on the four failures."""
    mapping = mapping or MAP
    rows, problems = [], []
    for gate, spec in mapping.items():
        cells = drive(spec["module"], tree)
        if not spec.get("properties"):
            problems.append(f"{NOTHING}: {gate} declares no properties")
            continue
        for entry in spec["properties"]:
            prop, want = entry.get("property"), entry.get("cell")
            if not want:
                problems.append(f"{NOTHING}: {gate} / {prop!r}")
                continue
            hits = [c for c in cells if want.lower() in c.lower()]
            if not hits:
                problems.append(f"{NO_CELL}: {gate} / {prop!r} -> {want!r}")
                continue
            if not all(cells[h] for h in hits):
                problems.append(f"{RED_CELL}: {gate} / {prop!r} -> {want!r}")
                continue
            text = f" {str(prop).lower()} "
            if (any(q in text for q in QUANTIFIERS)
                    and not entry.get("universal")):
                problems.append(
                    f"{MISLABELLED}: {gate} / {prop!r} quantifies over an "
                    f"open set in its own text and is not marked "
                    f"universal. The label is author-set, so the TEXT is "
                    f"checked against it (REVIEW 204).")
                continue
            if entry.get("universal") and entry.get("coverage") != "derived":
                problems.append(
                    f"{ENUMERATED}: {gate} / {prop!r} is quantified over "
                    f"an open set and its cell's coverage is "
                    f"{entry.get('coverage')!r}")
                continue
            rows.append({"gate": gate, "property": prop, "cell": hits[0],
                         "coverage": entry.get("coverage"),
                         "universal": bool(entry.get("universal"))})
    if problems:
        raise MapRefused("REFUSED:\n  " + "\n  ".join(problems))
    return {"protocol": PROTOCOL, "n_properties": len(rows),
            "n_gates": len(mapping), "rows": rows,
            "every_declared_property_maps_to_a_passing_cell": True}


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    real = None
    try:
        real = resolve()
        err = ""
    except MapRefused as exc:
        err = str(exc)
    ck("every declared property maps to a PASSING cell",
       real is not None,
       f"{real['n_properties']} properties over {real['n_gates']} gates"
       if real else err[:120])

    def one(entry, module="de_fair_value_replay_seam.py"):
        try:
            resolve({"g": {"module": module, "properties": [entry]}})
            return ""
        except MapRefused as exc:
            return str(exc)

    ck("a property mapping to NOTHING refuses",
       NOTHING in one({"property": "p", "cell": None}))
    ck("a cell name NO falsifier emits refuses",
       NO_CELL in one({"property": "p", "cell": "a cell nobody wrote",
                       "coverage": "enumerated"}))
    ck("a UNIVERSAL property with an ENUMERATED cover refuses -- the "
       "fourth-instance rule, made mechanical",
       ENUMERATED in one({"property": "every other input",
                          "cell": "a differing initial_state REFUSES",
                          "coverage": "enumerated", "universal": True}))
    ck("  and the same property with a DERIVED cover resolves",
       one({"property": "every other input",
            "cell": "a NEW field is digested with NO edit to any list",
            "coverage": "derived", "universal": True}) == "")
    ck("a property whose TEXT quantifies but is NOT marked universal "
       "refuses -- the label cannot be quietly downgraded",
       MISLABELLED in one({"property": "every other non-fair-value input "
                                       "is shared",
                           "cell": "a differing initial_state REFUSES",
                           "coverage": "enumerated", "universal": False}),
       "relabelling universal:False no longer hides it")
    ck("  and a property with no quantifier is unaffected",
       one({"property": "a pinned outcome path is refused",
            "cell": "a replay that PINS the outcome path is REFUSED",
            "coverage": "enumerated"}) == "")
    ck("a module with no falsifier refuses rather than covering nothing",
       NO_FALSIFIER in one({"property": "p", "cell": "x",
                            "coverage": "enumerated"},
                           module="de_data_root.py"))
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    try:
        out = resolve()
    except MapRefused as exc:
        print(str(exc))
        return 3
    for r in out["rows"]:
        mark = "U" if r["universal"] else " "
        print(f"  {mark} {r['gate']:28s} {r['property'][:46]:46s} "
              f"-> {r['cell'][:44]}")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
