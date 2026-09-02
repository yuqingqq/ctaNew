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


def fixture_register(ref: str = "R-900", **over) -> str:
    """A fixture register entry in the ADOPTED block form.

    Shared by the selftest and the mutation audit so the two cannot drift --
    and in block form because prose is no longer admissible for anything but
    the grandfathered ref."""
    f = {"ref": ref, "kind": "R-ADMISS", "population": POP_FULL,
         "sampling": "NONE",
         "present_source": "data/pm_5min/markets.jsonl",
         "scope_days": "FORWARD_RACE_DAYS", "scope_from": "20260901",
         "scope_to": "null", "revocable_by": "USER", "supersedes": "null"}
    f.update({k: v for k, v in over.items() if v is not None})
    body = "\n".join(f"{k}: {v}" for k, v in f.items())
    return (f"### {ref} — coordinator: R-ADMISS ratification for a fixture\n\n"
            "```ratification\n" + body + "\n```\n\n## next\n")


class RatificationRefused(RuntimeError):
    """The cited ratification is absent, is not one, or names a different
    population.  Refusal is the product."""


# ---------------------------------------------------------------------------
# 1. parse the entry -- a bounded object, not a grep over the file
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^### (R-\d+) ")


def all_entries(register_text: str) -> list[dict]:
    """Every `### R-<n> ` entry, in FILE ORDER, with its body.

    Order is what makes supersession decidable: a later entry may supersede
    an earlier one, never the reverse."""
    lines = register_text.split("\n")
    starts = [(i, m.group(1)) for i, line in enumerate(lines)
              if (m := HEADING_RE.match(line))]
    out = []
    for k, (i, ref) in enumerate(starts):
        end = len(lines)
        for j in range(i + 1, len(lines)):
            if ENTRY_END_RE.match(lines[j]):
                end = j
                break
        out.append({"ref": ref, "index": k, "line": i,
                    "heading": lines[i], "body": "\n".join(lines[i + 1:end])})
    return out


def superseded_by(register_text: str, ref: str) -> list[str]:
    """Refs of LATER entries whose block declares it supersedes `ref`."""
    entries = all_entries(register_text)
    pos = {e["ref"]: e["line"] for e in entries}
    if ref not in pos:
        return []
    out = []
    for e in entries:
        if e["line"] <= pos[ref]:
            continue
        blk = bind_from_block(e)
        if blk and str(blk.get("supersedes", "")).strip() == ref:
            out.append(e["ref"])
    return sorted(set(out))


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

#: CO-4: PROSE BINDING IS NOW ADMISSIBLE FOR EXACTLY ONE REF.
#: An entry ABOUT a ratification verified -- a coordinator sweep whose recap
#: sentence carried R-418's vocabulary ("R-ADMISS", "FULL supplied
#: complement", "no sampling", the ledger path) bound five fields and came
#: back VERIFIED, while ending "Nothing here ratifies anything." Vocabulary
#: hits are not references (rule 16), and from R-419 on EVERY coordinator
#: sweep carries that vocabulary because it quotes the ratification it is
#: reporting. So prose binding survives only for the one PRE-FORMAT entry it
#: was built to read, and every other block-less entry refuses by name.
GRANDFATHERED_PROSE_REFS = ("R-418",)
#: `scope_to: null` means OPEN. Written as a string because the block is
#: read as text; an absent field is NOT the same as null and does not open
#: the scope.
SCOPE_OPEN_TOKENS = ("null", "none", "")


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

def day_in_scope(day: str, fields: dict, unbindable: Sequence[str]):
    """True / False / None(UNBINDABLE). Both ends evaluated."""
    if "scope_from" in unbindable or "scope_from" not in fields:
        return None
    if day < str(fields["scope_from"]).strip():
        return False
    to = fields.get("scope_to")
    if to is None:
        return None                       # absent is NOT null
    to = str(to).strip()
    if to.lower() in SCOPE_OPEN_TOKENS:
        return True                       # `null` = open, explicitly
    return day <= to


def check(supplied: dict, ratification_ref: str,
          register_text: str | None = None) -> dict:
    """VERIFY / REFUSE / report-unbindable.  Decides nothing else."""
    if register_text is None:
        register_text = REGISTER.read_text()
    entry = parse_entry(register_text, ratification_ref)
    if entry is None:
        raise RatificationRefused(
            f"REFUSED: no register entry `### {ratification_ref} ` in "
            f"{REGISTER.name}. A well-formed ref to an entry that does not "
            f"exist looks exactly like a valid one, which is why the bridge's "
            f"shape check cannot be the last word.")

    # SUPERSESSION FIRST -- before any field is read, because a superseded
    # ratification's fields may be perfectly valid and still not the one in
    # force. FOR NEW RUNS ONLY: a receipt written BEFORE the superseding
    # entry existed carries its ref as PROVENANCE and is not invalidated by
    # this; that distinction is the coordinator's to keep and this refusal
    # only ever speaks about a run being started now.
    supers = superseded_by(register_text, ratification_ref)
    if supers:
        raise RatificationRefused(
            f"REFUSED FOR A NEW RUN: {ratification_ref} is SUPERSEDED by "
            f"{', '.join(supers)}. A receipt already carrying "
            f"{ratification_ref} is provenance and stays valid -- this "
            f"refusal is about starting a run under a ratification that is "
            f"no longer the one in force, not about rewriting history.")

    block = bind_from_block(entry)
    if block is None:
        # CO-4: prose binding survives for exactly one grandfathered ref.
        if ratification_ref not in GRANDFATHERED_PROSE_REFS:
            raise RatificationRefused(
                f"REFUSED: {ratification_ref} carries no ratification block; "
                f"prose binding is not admissible after R-419. An entry that "
                f"merely QUOTES a ratification carries all of its vocabulary "
                f"and would bind every field from it -- and from R-419 on "
                f"every coordinator sweep quotes the ratification it "
                f"reports. Only {list(GRANDFATHERED_PROSE_REFS)} predates the "
                f"format and may be read from prose.")
        fields, evidence, unbindable = bind_from_prose(entry)
        source = "PROSE_GRANDFATHERED"
    else:
        heading_ref = HEADING_RE.match(entry["heading"])
        heading_ref = heading_ref.group(1) if heading_ref else None
        if str(block.get("ref", "")).strip() != heading_ref:
            raise RatificationRefused(
                f"REFUSED: the block declares ref "
                f"{block.get('ref')!r} while the entry heading is "
                f"{heading_ref!r}. A block copied from another entry would "
                f"otherwise ratify under the wrong number.")
        fields, evidence, unbindable = bind_from_prose(entry)
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
    if named != POP_FULL:
        raise RatificationRefused(
            f"REFUSED: {ratification_ref} ratifies a {named} population while "
            f"this supply is the full complement "
            f"({pop['n_supplied_total']} windows, no selection field). A "
            f"ratification for a sampled population does not cover a full one.")
    if pop["selection_fields_present"]:
        raise RatificationRefused(
            f"REFUSED: {ratification_ref} ratifies the FULL complement "
            f"but the supply carries selection field(s) "
            f"{pop['selection_fields_present']}")
    # A FULL ratification whose own sampling field is not NONE contradicts
    # itself. It REFUSES rather than lowering `verified`: a self-contradictory
    # ratification is not a weaker one.
    if fields.get("sampling") != "NONE":
        raise RatificationRefused(
            f"REFUSED: {ratification_ref} ratifies {POP_FULL} but declares "
            f"sampling={fields.get('sampling')!r}. A ratification that "
            f"contradicts itself is refused, not scored lower.")
    checks = {
        "entry_exists": True,
        "declares_r_admiss": True,
        "population_named_is_full": named == POP_FULL,
        "supply_is_full_complement": pop["counts_sum_matches"]
        and not pop["selection_fields_present"],
        "sampling_declared_none": True,          # refused above otherwise
        # None = UNBINDABLE, never True. Prose names a CLASS of day and no
        # range, so whether THIS day is in scope cannot be decided from it --
        # which is what the block exists for. With a block: BOTH ends are
        # evaluated, and `scope_to: null` means OPEN. An ABSENT scope_to is
        # not null and does not open the scope.
        "day_in_scope": day_in_scope(str(supplied["day"]), fields,
                                     unbindable),
    }
    return {
        "protocol": PROTOCOL,
        "refusal_scope": "a refusal here is about STARTING A RUN; a receipt "
                         "already carrying a ref keeps it as provenance",
        "ratification_ref": ratification_ref,
        "binding_source": source,
        "entry_heading": entry["heading"][:160],
        "bound_fields": fields,
        "binding_evidence": evidence,
        "unbindable_from_prose": unbindable,
        "supply_population": pop,
        "checks": checks,
        "verified": all(v for v in checks.values() if v is not None),
        "unverifiable": sorted(k for k, v in checks.items() if v is None),
        "decides": "nothing -- this reports; admission is the coordinator's "
                   "act and accrual is decided elsewhere (R-418)",
    }


# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 42


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
    res = check(sup, "R-419")
    ok(res["verified"] and res["binding_source"] == "BLOCK",
       f"R-419 VERIFIES against the real 09-01 supply, bound from its "
       f"adopted BLOCK: { {k: v for k, v in res['checks'].items()} }")
    ok(res["checks"]["day_in_scope"] is True,
       f"and `day_in_scope` is now DECIDABLE and True — scope_from "
       f"{res['bound_fields']['scope_from']}, scope_to "
       f"{res['bound_fields']['scope_to']} (null = open). The gap round 8 "
       f"could only name is closed by the adopted format")
    ok(superseded_by(REGISTER.read_text(), "R-418") == ["R-419"],
       "SUPERSESSION IS A PREDICATE over the real register: R-419's block "
       "declares it supersedes R-418, found by forward-scanning later "
       "entries' blocks")
    refuses(lambda: check(sup, "R-418"),
            "AND R-418 NOW REFUSES FOR A NEW RUN, naming its superseder -- "
            "while a receipt already carrying it stays provenance, which the "
            "refusal says in its own words",
            needle="SUPERSEDED by R-419")
    ok("provenance" in res["refusal_scope"],
       "the emission states the scope of any refusal: about STARTING a run, "
       "never about rewriting a receipt already written")
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
    ok(res["unverifiable"] == [],
       "nothing is left unverifiable on R-419 -- every check is decided")

    # ---- CO-4: an entry ABOUT a ratification -----------------------------
    recap = (REGISTER.read_text()
             + "\n### R-9001 — 2026-09-02T11:00Z — coordinator: MEM round "
               "14 verified; recap of state\n\n"
               "MEM's true-up carries R-418: the coordinator's R-ADMISS "
               "ratification for FORWARD-RACE DAYS — the population is the "
               "FULL supplied complement, no sampling, present read from "
               "data/pm_5min/markets.jsonl for a forward-race day D. "
               "Nothing here ratifies anything.\n")
    ok(bind_from_prose(parse_entry(recap, "R-9001"))[0].get("population")
       == POP_FULL,
       "CO-4 REPRODUCED: the recap entry's prose binds population "
       f"{POP_FULL} from vocabulary it is merely QUOTING -- five fields would "
       f"have bound, and the entry ends 'Nothing here ratifies anything'")
    refuses(lambda: check(sup, "R-9001", recap),
            "AND IT NOW REFUSES BY NAME: no ratification block, and prose "
            "binding is not admissible after R-419. Vocabulary hits are not "
            "references (rule 16), and every coordinator sweep from now on "
            "carries that vocabulary because it quotes what it reports",
            needle="prose binding is not admissible")
    ok(GRANDFATHERED_PROSE_REFS == ("R-418",),
       "prose survives for EXACTLY ONE grandfathered ref, named with its "
       "reason: R-418 is the pre-format ratification")
    grand = ("### R-418 — coordinator: R-ADMISS ratification — the population "
             "is the FULL supplied complement, no sampling\n\nBody naming "
             "data/pm_5min/markets.jsonl for a forward-race day D.\n\n## x\n")
    gres = check(sup, "R-418", grand)
    ok(gres["binding_source"] == "PROSE_GRANDFATHERED"
       and gres["checks"]["day_in_scope"] is None,
       "POSITIVE CONTROL on the grandfather: with no superseder present "
       "R-418 still reads from prose, and still cannot decide day_in_scope -- "
       "the reason the format was adopted")

    # ---- refusals, on FIXTURE register text ----------------------------
    good = fixture_register()
    ok(check(sup, "R-900", good)["verified"],
       "FIXTURE POSITIVE CONTROL: a valid block entry in a fixture register "
       "VERIFIES, so the refusals below are not a blanket")
    refuses(lambda: check(sup, "R-901", good),
            "KNOWN-BAD: a ref with NO entry REFUSES by name",
            needle="no register entry")
    refuses(lambda: check(sup, "R-900", fixture_register(kind="STATUS_NOTE")),
            "KNOWN-BAD: a block declaring a kind that is not R-ADMISS "
            "REFUSES -- an entry can be real, recent and about something else",
            needle="does not declare itself an R-ADMISS")
    refuses(lambda: check(sup, "R-900",
                          fixture_register(population=POP_SAMPLED)),
            "KNOWN-BAD: a ratification naming a SAMPLED population REFUSES "
            "against a FULL supply -- the ref is real and the population is "
            "not this one",
            needle="does not cover a full one")
    bad_counts = json.loads(json.dumps(sup))
    bad_counts["n_supplied_total"] += 7
    refuses(lambda: check(bad_counts, "R-900", good),
            "KNOWN-BAD: a supply whose counts do NOT sum REFUSES before any "
            "population question is reached",
            needle="does not describe itself consistently")
    withsel = dict(sup, sampled=True)
    refuses(lambda: check(withsel, "R-900", good),
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
            "does not exist",
            needle="no register entry")
    ok(not shape_ok,
       "confirmed at the bridge: it accepts R-99999's shape, which is "
       "exactly the gap this module closes")

    # ---- scope_to, the heading ref, and self-contradiction --------------
    def _blk(**over):
        f = {"ref": "R-902", "kind": "R-ADMISS",
             "population": POP_FULL, "sampling": "NONE",
             "present_source": "data/pm_5min/markets.jsonl",
             "scope_days": "FORWARD_RACE_DAYS", "scope_from": "20260901",
             "scope_to": "null", "revocable_by": "USER", "supersedes": "null"}
        f.update(over)
        body = "\n".join(f"{k}: {v}" for k, v in f.items())
        return ("### R-902 — coordinator: R-ADMISS ratification\n\n"
                "```ratification\n" + body + "\n```\n\n## next\n")

    ok(check(sup, "R-902", _blk())["checks"]["day_in_scope"] is True,
       "scope_to `null` means OPEN: 09-01 is in scope")
    ok(check(sup, "R-902", _blk(scope_to="20260902"))["checks"]["day_in_scope"]
       is True,
       "and a CLOSED scope that still contains the day reads True")
    closed = check(sup, "R-902", _blk(scope_to="20260901"))
    ok(closed["checks"]["day_in_scope"] is True,
       "a day exactly ON the closing bound is IN scope (<=, not <)")
    sup902 = dict(sup, day="20260902")
    past = check(sup902, "R-902", _blk(scope_to="20260901"))
    ok(past["checks"]["day_in_scope"] is False and not past["verified"],
       "KNOWN-BAD: a day PAST a closed scope reads day_in_scope FALSE and "
       "the run is not verified (scope_to 20260901, day 20260902)")
    early = check(sup, "R-902", _blk(scope_from="20260902"))
    ok(early["checks"]["day_in_scope"] is False,
       "and a day BEFORE scope_from is likewise False")
    noto = _blk()
    noto = noto.replace("scope_to: null\n", "")
    ok(check(sup, "R-902", noto)["checks"]["day_in_scope"] is None,
       "AN ABSENT scope_to IS NOT `null`: it reads UNBINDABLE, not open -- "
       "absence must never be the permissive answer")
    refuses(lambda: check(sup, "R-902", _blk(ref="R-418")),
            "KNOWN-BAD: a block whose `ref` is not the heading's REFUSES -- "
            "a block copied from another entry would ratify under the wrong "
            "number",
            needle="while the entry heading is")
    refuses(lambda: check(sup, "R-902", _blk(sampling="STRATIFIED")),
            "KNOWN-BAD: a FULL ratification declaring sampling != NONE "
            "REFUSES rather than scoring lower -- a ratification that "
            "contradicts itself is not a weaker one",
            needle="contradicts itself")
    sup_chain = ("### R-902 — coordinator: R-ADMISS ratification\n\n"
                 "```ratification\nref: R-902\nkind: R-ADMISS\n"
                 f"population: {POP_FULL}\nsampling: NONE\n"
                 "present_source: data/pm_5min/markets.jsonl\n"
                 "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
                 "scope_to: null\nrevocable_by: USER\nsupersedes: null\n"
                 "```\n\n"
                 "### R-903 — coordinator: R-ADMISS ratification\n\n"
                 "```ratification\nref: R-903\nkind: R-ADMISS\n"
                 f"population: {POP_FULL}\nsampling: NONE\n"
                 "present_source: data/pm_5min/markets.jsonl\n"
                 "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
                 "scope_to: null\nrevocable_by: USER\n"
                 "supersedes: R-902\n```\n\n## next\n")
    refuses(lambda: check(sup, "R-902", sup_chain),
            "KNOWN-BAD on a fixture chain: R-902 refuses once a LATER entry's "
            "block supersedes it")
    ok(check(sup, "R-903", sup_chain)["verified"],
       "POSITIVE CONTROL: the superseding entry itself VERIFIES, so the "
       "predicate is directional -- later supersedes earlier, never the "
       "reverse")
    ok(superseded_by(sup_chain, "R-903") == [],
       "and the superseder is not itself superseded by the entry it replaced")

    # ---- the proposed block, read when present ------------------------
    blocked = ("### R-904 — coordinator: R-ADMISS ratification\n\n"
               + PROPOSED_BLOCK.replace("R-418", "R-904")
               + "\n\n## next\n")
    rb = check(sup, "R-904", blocked)
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


def sup_chain_fixture() -> str:
    """R-902 followed by R-903, whose block supersedes it."""
    return (fixture_register("R-902").replace("\n\n## next\n", "\n\n")
            + fixture_register("R-903", supersedes="R-902"))


def mutation_audit(sup: dict) -> dict:
    """Each refusal driven LIVE (must refuse) and against an input that
    should NOT trip it (must not) -- visibly different calls, round 5's
    lesson. There is no `skip_guard` here: the refusals are sequential in
    `check`, so the disabled arm is a DIFFERENT INPUT that must reach past
    that refusal rather than the same input with a switch thrown."""
    good = fixture_register()
    bad_counts = json.loads(json.dumps(sup))
    bad_counts["n_supplied_total"] += 7
    cases = {
        "entry_absent": ((sup, "R-901", good), (sup, "R-900", good)),
        "not_a_ratification": ((sup, "R-900",
                                fixture_register(kind="STATUS_NOTE")),
                               (sup, "R-900", good)),
        "sampled_population": ((sup, "R-900",
                                fixture_register(population=POP_SAMPLED)),
                               (sup, "R-900", good)),
        "no_block_not_grandfathered": ((sup, "R-905",
                                        "### R-905 — coordinator: R-ADMISS "
                                        "ratification, FULL supplied "
                                        "complement, no sampling\n\nx\n"),
                                       (sup, "R-900", good)),
        "block_ref_mismatch": ((sup, "R-900",
                                fixture_register(ref="R-900").replace(
                                    "ref: R-900", "ref: R-777")),
                               (sup, "R-900", good)),
        "self_contradicting_sampling": ((sup, "R-900",
                                         fixture_register(
                                             sampling="STRATIFIED")),
                                        (sup, "R-900", good)),
        "superseded": ((sup, "R-902", sup_chain_fixture()),
                       (sup, "R-903", sup_chain_fixture())),
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
