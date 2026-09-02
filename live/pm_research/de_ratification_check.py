"""Is the ratification a supply cites REAL, and does it name THIS population?

SURFACE AUTHORISATION (R-126, in-file): coordinator DE round-8 dispatch;
EV_REPLAY_PLAN section 2 (window selection is an R-ADMISS act the coordinator
ratifies); R-418 (the ratification this checker is first exercised against).
RESEARCH-ONLY, OFFLINE.

WHY IT EXISTS.  `window_specs_from_supply` refuses a ref that is not `R-<n>`,
which stops a typo and a Q-row.  It cannot tell whether `R-999` exists, nor
whether the entry it names ratifies THIS population -- a well-formed ref to
an entry that ratifies something else is exactly as wrong as a malformed one
and looks perfectly correct.  Rule 10: the ratification becomes a computed
predicate instead of a string a receipt carries.

WHAT IT DOES NOT DO.  It performs no ratification, admits no day, and decides
nothing.  It reads a committed register entry and reports what it could
VERIFY, what it REFUSED, and -- the part that matters most -- what it could
NOT BIND from prose at all.  An unbindable field is reported by name and is
never silently counted as satisfied (rule 16: absence must not read as a
pass).

THE FORMAT PROPOSAL IS A PROPOSAL.  Section `PROPOSED_BLOCK` below is a
fenced `ratification:` block for FUTURE entries; adopting it is the
coordinator's act, not this module's.  R-418 is prose, so the checker binds
what prose supports and NAMES what it cannot -- which is itself the argument
for the block.

    python3 live/pm_research/de_ratification_check.py --selftest
    python3 live/pm_research/de_ratification_check.py check --ref R-418
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROTOCOL = "de_ratification_check_v1"
ROOT = Path(__file__).resolve().parents[2]
REGISTER = (ROOT / "orchestrator/PROGRAMS/P-2026-003-polymarket-5min"
                   "/workspace/COORDINATION.md")

REF_RE = re.compile(r"^R-\d+$")
ENTRY_RE = "^### {ref} "                      # a register entry heading
#: A new entry heading or a new section ends the one being read.
ENTRY_END_RE = re.compile(r"^(### R-\d+ |## )")

#: The populations this checker can evaluate.  A ratification naming anything
#: else is REPORTED as unknown and REFUSED -- never treated as "probably the
#: full one".
POP_FULL = "FULL_SUPPLIED_COMPLEMENT"
POP_SAMPLED = "SAMPLED_OR_CAPPED"
KNOWN_POPULATIONS = (POP_FULL, POP_SAMPLED)

#: The fields a ratification must pin for a supply to be checkable against it.
RATIFICATION_FIELDS = ("ref", "kind", "population", "sampling",
                       "present_source", "scope_days")

#: PROPOSED, NOT ADOPTED -- the coordinator's to take or leave (Q-DE-26).
PROPOSED_BLOCK = """```ratification
ref: R-418
kind: R-ADMISS
population: FULL_SUPPLIED_COMPLEMENT
sampling: NONE
present_source: data/pm_5min/markets.jsonl
scope_days: FORWARD_RACE_DAYS
scope_from: 20260901          # the field prose cannot supply
scope_to: null                # open-ended, stated as null rather than absent
revocable_by: USER
supersedes: null
```"""


class RatificationRefused(RuntimeError):
    """The cited ratification is absent, is not one, or names a different
    population.  Refusal is the product."""


# ---------------------------------------------------------------------------
# 1. parse the entry -- a bounded object, not a grep over the file
# ---------------------------------------------------------------------------

def parse_entry(register_text: str, ref: str) -> dict | None:
    """The register entry for `ref` as a STRUCTURED object, or None.

    The whole file is never searched for vocabulary: the entry's boundaries
    are found first (its own heading, up to the next entry or section), and
    every predicate below runs over THAT text.  A word appearing in some
    other entry -- or in the Q-table, which carries no `### R-` heading at
    all -- can therefore never satisfy a claim about this one."""
    if not REF_RE.match(ref or ""):
        return None
    lines = register_text.split("\n")
    start = None
    head = re.compile(ENTRY_RE.format(ref=re.escape(ref)))
    for i, line in enumerate(lines):
        if head.match(line):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if ENTRY_END_RE.match(lines[j]):
            end = j
            break
    return {"ref": ref, "heading": lines[start],
            "body": "\n".join(lines[start + 1:end]),
            "line_start": start + 1, "line_end": end}


# ---------------------------------------------------------------------------
# 2. bind fields -- from the fenced block if present, else from prose
# ---------------------------------------------------------------------------

def bind_from_block(entry: dict) -> dict | None:
    """The PROPOSED machine-readable form, if the entry carries one."""
    m = re.search(r"```ratification\n(.*?)```", entry["heading"] + "\n"
                  + entry["body"], re.S)
    if not m:
        return None
    out: dict[str, Any] = {}
    for line in m.group(1).split("\n"):
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


#: Prose anchors, scoped to the parsed entry.  Each is a phrase the entry
#: itself uses; the EVIDENCE is recorded beside the value so a reader can see
#: what the binding rested on rather than trusting the field.
_PROSE = {
    "kind": [("R-ADMISS", "R-ADMISS")],
    "population": [("FULL supplied complement", POP_FULL),
                   ("full supplied complement", POP_FULL),
                   ("stratified", POP_SAMPLED),
                   ("capped selection", POP_SAMPLED)],
    "sampling": [("no sampling", "NONE"),
                 ("no stratified or capped selection", "NONE")],
    "present_source": [("data/pm_5min/markets.jsonl",
                        "data/pm_5min/markets.jsonl")],
    "scope_days": [("forward-race day", "FORWARD_RACE_DAYS"),
                   ("FORWARD-RACE DAYS", "FORWARD_RACE_DAYS")],
}
#: What prose CANNOT supply, however carefully it is read.  Named here rather
#: than discovered per-entry, because the gap is a property of prose.
UNBINDABLE_FROM_PROSE = ("scope_from", "scope_to")


def bind_from_prose(entry: dict) -> tuple[dict, dict, list[str]]:
    """(fields, evidence, unbindable).  Sampled beats full when both appear:
    an entry that mentions a cap at all is not a full-population entry until
    someone says which sentence governs."""
    text = entry["heading"] + "\n" + entry["body"]
    fields: dict[str, Any] = {"ref": entry["ref"]}
    evidence: dict[str, str] = {"ref": "heading"}
    for field, anchors in _PROSE.items():
        for phrase, value in anchors:
            if phrase in text:
                if field == "population" and value == POP_SAMPLED:
                    # a mention of sampling only counts as the POPULATION if
                    # the entry is not explicitly negating it
                    neg = (f"no {phrase}" in text
                           or f"no stratified or {phrase}" in text)
                    if neg:
                        continue
                fields.setdefault(field, value)
                evidence.setdefault(field, phrase)
    return fields, evidence, list(UNBINDABLE_FROM_PROSE)


# ---------------------------------------------------------------------------
# 3. what the supply itself computes
# ---------------------------------------------------------------------------

def supply_population(supplied: dict) -> dict:
    """The population the SUPPLY computes, from its own counts."""
    per = {c: v["n_present"] - v["n_masked_applied"]
           for c, v in supplied["counts"].items()}
    return {"n_supplied_total": supplied["n_supplied_total"],
            "sum_present_minus_masked": sum(per.values()),
            "per_coin": per,
            "counts_sum_matches": sum(per.values())
            == supplied["n_supplied_total"],
            "selection_fields_present": sorted(
                k for k in supplied
                if k in ("sampled", "sample_size", "cap", "stratified",
                         "selection", "per_coin_cap"))}


# ---------------------------------------------------------------------------
# 4. the check
# ---------------------------------------------------------------------------

def check(supplied: dict, ratification_ref: str,
          register_text: str | None = None) -> dict:
    """VERIFY / REFUSE / report-unbindable.  Decides nothing else."""
    if register_text is None:
        register_text = REGISTER.read_text()
    refusals: list[str] = []
    entry = parse_entry(register_text, ratification_ref)
    if entry is None:
        raise RatificationRefused(
            f"REFUSED: no register entry `### {ratification_ref} ` in "
            f"{REGISTER.name}. A well-formed ref to an entry that does not "
            f"exist looks exactly like a valid one, which is why the bridge's "
            f"shape check cannot be the last word.")
    block = bind_from_block(entry)
    fields, evidence, unbindable = bind_from_prose(entry)
    source = "PROSE"
    if block:
        fields = {**fields, **block}
        evidence = {**evidence, **{k: "ratification block" for k in block}}
        unbindable = [f for f in unbindable if f not in block]
        source = "BLOCK"
    if fields.get("kind") != "R-ADMISS":
        raise RatificationRefused(
            f"REFUSED: {ratification_ref} does not declare itself an R-ADMISS "
            f"ratification (bound kind: {fields.get('kind')!r}). An entry can "
            f"be real, recent and about something else entirely.")
    pop = supply_population(supplied)
    if not pop["counts_sum_matches"]:
        raise RatificationRefused(
            f"REFUSED: the supply's n_supplied_total "
            f"({pop['n_supplied_total']}) is not the sum of its per-coin "
            f"(n_present - n_masked_applied) ({pop['sum_present_minus_masked']})"
            f" -- whatever population the entry names, this supply does not "
            f"describe itself consistently")
    named = fields.get("population")
    if named not in KNOWN_POPULATIONS:
        raise RatificationRefused(
            f"REFUSED: {ratification_ref} names population {named!r}, which "
            f"this checker cannot evaluate (known: {KNOWN_POPULATIONS}). "
            f"Reported as unknown rather than assumed to be the full one.")
    if named == POP_FULL:
        if pop["selection_fields_present"]:
            raise RatificationRefused(
                f"REFUSED: {ratification_ref} ratifies the FULL complement "
                f"but the supply carries selection field(s) "
                f"{pop['selection_fields_present']}")
    else:
        raise RatificationRefused(
            f"REFUSED: {ratification_ref} ratifies a {named} population while "
            f"this supply is the full complement "
            f"({pop['n_supplied_total']} windows, no selection field). A "
            f"ratification for a sampled population does not cover a full one.")
    checks = {
        "entry_exists": True,
        "declares_r_admiss": True,
        "population_named_is_full": named == POP_FULL,
        "supply_is_full_complement": pop["counts_sum_matches"]
        and not pop["selection_fields_present"],
        "sampling_declared_none": fields.get("sampling") == "NONE",
        # None = UNBINDABLE, never True. Prose names a CLASS of day
        # ("a forward-race day D") and no range, so whether THIS day is in
        # scope cannot be decided from it. This is the field the proposed
        # block exists for.
        "day_in_scope": None if "scope_from" in unbindable else
        (str(supplied["day"]) >= str(fields.get("scope_from", "")) ),
    }
    return {
        "protocol": PROTOCOL,
        "ratification_ref": ratification_ref,
        "binding_source": source,
        "entry_heading": entry["heading"][:160],
        "bound_fields": fields,
        "binding_evidence": evidence,
        "unbindable_from_prose": unbindable,
        "supply_population": pop,
        "checks": checks,
        "verified": all(v for v in checks.values() if v is not None)
        and not refusals,
        "unverifiable": sorted(k for k, v in checks.items() if v is None),
        "decides": "nothing -- this reports; admission is the coordinator's "
                   "act and accrual is decided elsewhere (R-418)",
    }


# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 24


def selftest() -> int:
    import de_admissible_windows as daw
    import ev_replay_seam as seam
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_ratification_check] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle=None):
        try:
            fn()
        except RatificationRefused as exc:
            if needle and needle not in str(exc):
                raise SystemExit(f"[de_ratification_check] FAIL: {label} -- "
                                 f"refused for another reason ({exc})")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_ratification_check] FAIL (no refusal): {label}")

    day = daw.REAL_DAY
    mask = daw.load_mask(day)
    sup = daw.supply(day, {c: list(daw._grid(day)) for c in mask["coins"]},
                     mask)

    # ---- the REAL R-418, in the REAL register --------------------------
    e = parse_entry(REGISTER.read_text(), "R-418")
    ok(e is not None and "R-ADMISS ratification" in e["heading"],
       f"R-418 parses out of the committed register as a BOUNDED entry "
       f"(lines {e['line_start']}-{e['line_end']}), heading first")
    ok(ENTRY_END_RE.match("### R-419 x") and ENTRY_END_RE.match("## 6. x")
       and not ENTRY_END_RE.match("**bold**"),
       "and its end is the next entry or section heading, so the body is "
       "this entry's text and not the rest of the file")
    res = check(sup, "R-418")
    ok(res["verified"] and res["binding_source"] == "PROSE",
       f"R-418 VERIFIES against the real 09-01 supply, bound from PROSE: "
       f"{ {k: v for k, v in res['checks'].items()} }")
    ok(res["supply_population"]["n_supplied_total"] == 1875
       and res["supply_population"]["counts_sum_matches"],
       f"the population predicate is COMPUTED: n_supplied_total 1875 == the "
       f"sum of per-coin (n_present - n_masked_applied)")
    ok(res["bound_fields"]["population"] == POP_FULL
       and res["bound_fields"]["sampling"] == "NONE"
       and res["bound_fields"]["present_source"]
       == "data/pm_5min/markets.jsonl",
       f"and the entry's own fields bound: population {POP_FULL}, sampling "
       f"NONE, present_source named")
    ok(res["unverifiable"] == ["day_in_scope"]
       and res["checks"]["day_in_scope"] is None,
       "THE HONEST GAP, NAMED: `day_in_scope` is UNBINDABLE from prose -- "
       "R-418 names a CLASS of day ('a forward-race day D') and no range, so "
       "the checker reports None, never True. An unbindable field silently "
       "counted as satisfied is the absence-reads-as-pass failure")
    ok(res["unbindable_from_prose"] == list(UNBINDABLE_FROM_PROSE),
       f"and the fields prose cannot supply are named as a property of "
       f"prose, not discovered per entry: {res['unbindable_from_prose']}")

    # ---- refusals, on FIXTURE register text ----------------------------
    good = ("### R-900 — coordinator: R-ADMISS ratification for X — the "
            "population is the FULL supplied complement, no sampling\n\n"
            "Body naming data/pm_5min/markets.jsonl for a forward-race day "
            "D.\n\n## next\n")
    ok(check(sup, "R-900", good)["verified"],
       "FIXTURE POSITIVE CONTROL: a valid entry in a fixture register "
       "VERIFIES, so the refusals below are not a blanket")
    refuses(lambda: check(sup, "R-901", good),
            "KNOWN-BAD: a ref with NO entry REFUSES by name",
            needle="no register entry")
    notrat = ("### R-900 — coordinator: a status note about the collector\n\n"
              "Nothing is ratified here.\n\n## next\n")
    refuses(lambda: check(sup, "R-900", notrat),
            "KNOWN-BAD: an entry that is NOT a ratification REFUSES -- an "
            "entry can be real, recent and about something else",
            needle="does not declare itself an R-ADMISS")
    sampled = ("### R-900 — coordinator: R-ADMISS ratification — the "
               "population is a stratified sample of the complement\n\n"
               "Body for a forward-race day D.\n\n## next\n")
    refuses(lambda: check(sup, "R-900", sampled),
            "KNOWN-BAD: a ratification naming a SAMPLED population REFUSES "
            "against a FULL supply -- the ref is real and the population is "
            "not this one",
            needle="does not cover a full one")
    bad_counts = json.loads(json.dumps(sup))
    bad_counts["n_supplied_total"] += 7
    refuses(lambda: check(bad_counts, "R-418"),
            "KNOWN-BAD: a supply whose counts do NOT sum REFUSES before any "
            "population question is reached",
            needle="does not describe itself consistently")
    withsel = dict(sup, sampled=True)
    refuses(lambda: check(withsel, "R-418"),
            "KNOWN-BAD: a FULL ratification against a supply carrying a "
            "selection field REFUSES")

    # ---- the two refusals COMPOSE -------------------------------------
    try:
        seam.window_specs_from_supply(sup, ratification_ref="Q-DE-25")
        bridge_refused = False
    except seam.SeamRefused:
        bridge_refused = True
    ok(bridge_refused and parse_entry(REGISTER.read_text(), "Q-DE-25") is None,
       "THE TWO REFUSALS COMPOSE: a Q-row is refused at the BRIDGE on shape "
       "AND has no `### R-` entry to find here -- neither layer relies on "
       "the other")
    try:
        seam.window_specs_from_supply(sup, ratification_ref="R-99999")
        shape_ok = False
    except seam.SeamRefused:
        shape_ok = True
    refuses(lambda: check(sup, "R-99999"),
            "AND THEY DIVIDE THE WORK: `R-99999` is well-FORMED, so the "
            "bridge admits its shape; only this checker can say the entry "
            "does not exist")
    ok(not shape_ok,
       "confirmed at the bridge: it accepts R-99999's shape, which is "
       "exactly the gap this module closes")

    # ---- the proposed block, read when present ------------------------
    blocked = ("### R-902 — coordinator: R-ADMISS ratification\n\n"
               + PROPOSED_BLOCK.replace("R-418", "R-902")
               + "\n\n## next\n")
    rb = check(sup, "R-902", blocked)
    ok(rb["binding_source"] == "BLOCK" and rb["verified"],
       "THE PROPOSED FORMAT, EXERCISED: an entry carrying a fenced "
       "`ratification:` block binds from the BLOCK rather than from prose")
    ok(rb["checks"]["day_in_scope"] is True and rb["unverifiable"] == [],
       "and the gap CLOSES: `scope_from` makes `day_in_scope` decidable, "
       "which is the whole argument for adopting the block")
    ok(rb["bound_fields"]["scope_from"] == "20260901",
       "with the field bound from the block, evidence recorded as such")
    ok("adopt" not in PROTOCOL and PROPOSED_BLOCK.startswith("```ratification"),
       "the format is PROPOSED here and adopted nowhere -- adopting it is "
       "the coordinator's act (Q-DE-26)")

    # ---- the checker decides nothing ----------------------------------
    ok(res["decides"].startswith("nothing"),
       "and the emission says what it decides: nothing -- admission is the "
       "coordinator's act and accrual is decided elsewhere")

    audit = mutation_audit(sup)
    ok(audit["all_load_bearing"],
       f"MUTATION AUDIT: {audit['n_guards']} refusal paths, live and disabled "
       f"as distinct calls, {audit['survivors']} survivors")
    ok(set(audit["per_guard"]) == set(audit["expected"]),
       f"covering every refusal by name: {sorted(audit['per_guard'])}")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_ratification_check] selftest OK -- {n[0]} checks")
    return 0


def mutation_audit(sup: dict) -> dict:
    """Each refusal driven LIVE (must refuse) and against an input that
    should NOT trip it (must not) -- visibly different calls, round 5's
    lesson. There is no `skip_guard` here: the refusals are sequential in
    `check`, so the disabled arm is a DIFFERENT INPUT that must reach past
    that refusal rather than the same input with a switch thrown."""
    good = ("### R-900 — coordinator: R-ADMISS ratification for X — the "
            "population is the FULL supplied complement, no sampling\n\n"
            "Body naming data/pm_5min/markets.jsonl for a forward-race day "
            "D.\n\n## next\n")
    notrat = ("### R-900 — coordinator: a status note\n\nNothing.\n\n## next\n")
    sampled = ("### R-900 — coordinator: R-ADMISS ratification — the "
               "population is a stratified sample\n\nBody for a "
               "forward-race day D.\n\n## next\n")
    bad_counts = json.loads(json.dumps(sup))
    bad_counts["n_supplied_total"] += 7
    cases = {
        "entry_absent": ((sup, "R-901", good), (sup, "R-900", good)),
        "not_a_ratification": ((sup, "R-900", notrat), (sup, "R-900", good)),
        "sampled_population": ((sup, "R-900", sampled), (sup, "R-900", good)),
        "counts_do_not_sum": ((bad_counts, "R-900", good),
                              (sup, "R-900", good)),
        "selection_field": ((dict(sup, sampled=True), "R-900", good),
                            (sup, "R-900", good)),
    }
    per: dict[str, dict] = {}
    for name, (bad, ctrl) in cases.items():
        try:
            check(*bad)
            live = False
        except RatificationRefused:
            live = True
        try:
            check(*ctrl)
            disabled = False
        except RatificationRefused:
            disabled = True
        per[name] = {"refuses_on_its_known_bad": live,
                     "refuses_on_the_control": disabled,
                     "load_bearing": live and not disabled}
    survivors = sorted(k for k, v in per.items() if not v["load_bearing"])
    return {"n_guards": len(per), "per_guard": per, "survivors": survivors,
            "expected": sorted(cases), "all_load_bearing": not survivors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("cmd", nargs="?", choices=["check"])
    ap.add_argument("--ref", default="R-418")
    ap.add_argument("--day", default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "check":
        if selftest() != 0:
            return 1
        import de_admissible_windows as daw
        day = a.day or daw.REAL_DAY
        mask = daw.load_mask(day)
        sup = daw.supply(day, {c: list(daw._grid(day)) for c in mask["coins"]},
                         mask)
        print(json.dumps(check(sup, a.ref), indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
