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

#: EVERY field the ADOPTED block must carry (R-419 section 4). A block
#: missing any of them REFUSES BY NAME.
#:
#: CO-5: this constant was DECLARED AND NEVER USED, and a block missing
#: `scope_to` therefore came back `verified: True` with `day_in_scope: None`.
#: The check was right -- absent is UNBINDABLE, never open -- but `verified`
#: is the conjunction of DECIDED checks, so a consumer reading `verified`
#: alone read the absence as a pass. A malformed block is now REFUSED rather
#: than left undecided: undecided is a state a caller can mishandle, refused
#: is not.
RATIFICATION_FIELDS = ("ref", "kind", "population", "sampling",
                       "present_source", "scope_days", "scope_from",
                       "scope_to", "revocable_by", "supersedes")

#: DE-R3: the ADOPTED VOCABULARY each field's VALUE must come from.
#: Round 10 made a MISSING field refuse; a NONSENSE VALUE still verified
#: clean -- `present_source: /etc/passwd`, `scope_days: WHATEVER`,
#: `revocable_by: DE` all returned verified True with unverifiable [].
#: "A field nobody supplied", "a field this checker cannot decide" and "a
#: field with a wrong value" are three different things and must never look
#: alike; the refusal below says VALUE, never MISSING.
LEDGER_PATH = "data/pm_5min/markets.jsonl"       # named once, R-419's own
#: `kind` is DELIBERATELY NOT HERE. It already has its own refusal, and that
#: refusal says something different: an entry whose kind is not R-ADMISS is
#: NOT A RATIFICATION, which is not the same complaint as a ratification
#: carrying a wrong value. Folding it in unified the message and lost the
#: distinction -- caught by the reason-check on the existing known-bad, which
#: is what those needles are for.
FIELD_VOCABULARY: dict[str, tuple] = {
    "population": KNOWN_POPULATIONS,
    # LEGITIMATE sampling values, not just the one this programme uses today.
    # Restricting this to ("NONE",) would have hardcoded that no sampled
    # ratification can ever exist -- which contradicts KNOWN_POPULATIONS
    # carrying SAMPLED_OR_CAPPED -- and it swallowed the SEMANTIC refusal
    # below, whose complaint is different: a FULL population declaring
    # sampling is CONTRADICTING ITSELF, not carrying an unknown word.
    "sampling": ("NONE", "STRATIFIED", "CAPPED"),
    "present_source": (LEDGER_PATH,),
    "scope_days": ("FORWARD_RACE_DAYS",),
    "revocable_by": ("USER",),
}

#: `### R-419 — 2026-09-02T11:03Z — coordinator: …`
HEADING_TS_RE = re.compile(
    r"^### R-\d+ [—-]+ (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z)")

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


class NotVerified(RuntimeError):
    """A consumer asked for a verified ratification and did not get one."""


#: DE10-R1: EVERY temporal comparison in this module was LEXICOGRAPHIC on
#: strings. One root cause, three symptoms, and none of them surfaced:
#:   now_utc="zzzz"          -> day_closed True,  verified True   (permissive)
#:   scope_to="not-a-date"   -> day_in_scope True, verified True  (permissive)
#:   scope_from="not-a-date" -> day_in_scope False                (restrictive)
#:   now_utc=123             -> TypeError, which is not a refusal
#: Garbage sorted after "2026-…" reads as the future and garbage sorting
#: before it reads as the past, so the SAME defect was permissive in two
#: fields and restrictive in a third. Everything temporal is parsed to a
#: datetime BEFORE any comparison now, and an unparsable or non-string value
#: REFUSES naming the field AND the value.
INSTANT_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ")
DAY_FORMAT = "%Y%m%d"


def parse_instant(value, field: str):
    """An ISO-8601 Z instant, or a REFUSAL naming the field and the value."""
    import datetime as _dt
    if not isinstance(value, str):
        raise RatificationRefused(
            f"REFUSED: {field} is {value!r} ({type(value).__name__}), not a "
            f"string. A TypeError from a comparison is not a refusal -- it "
            f"is a crash wearing one (DE10-R1).")
    for fmt in INSTANT_FORMATS:
        try:
            return _dt.datetime.strptime(value.strip(), fmt).replace(
                tzinfo=_dt.timezone.utc)
        except ValueError:
            continue
    raise RatificationRefused(
        f"REFUSED: {field} carries {value!r}, which is not an instant in "
        f"{list(INSTANT_FORMATS)}. Compared as a STRING it would sort "
        f"against real timestamps and read as the future or the past "
        f"depending only on its first character (DE10-R1).")


def parse_day(value, field: str):
    """A YYYYMMDD day, or a REFUSAL naming the field and the value."""
    import datetime as _dt
    if not isinstance(value, str):
        raise RatificationRefused(
            f"REFUSED: {field} is {value!r} ({type(value).__name__}), not a "
            f"string")
    try:
        return _dt.datetime.strptime(value.strip(), DAY_FORMAT).replace(
            tzinfo=_dt.timezone.utc)
    except ValueError:
        raise RatificationRefused(
            f"REFUSED: {field} carries {value!r}, which is not a day in "
            f"{DAY_FORMAT}. Compared as a STRING it would sort against real "
            f"days and silently open or close the scope (DE10-R1).")


def entry_timestamp(heading: str) -> str | None:
    """The register timestamp in an entry's heading, or None."""
    m = HEADING_TS_RE.match(heading)
    return m.group(1) if m else None


def _norm_ts(ts: str, field: str = "timestamp"):
    """PARSED, not normalised. The name is kept so every call site changes
    meaning at once: this used to pad seconds and return a string for a
    lexical compare, which is the DE10-R1 defect."""
    return parse_instant(ts, field)


def day_end_instant(day: str, field: str = "supply.day"):
    """The instant a UTC day is FINISHED: the next day's 00:00:00Z."""
    import datetime as dt
    return parse_day(str(day), field) + dt.timedelta(days=1)


def day_end_utc(day: str) -> str:
    """The same instant, rendered. Kept for readers and receipts; nothing
    compares this string any more."""
    return day_end_instant(day).strftime("%Y-%m-%dT%H:%M:%SZ")


def require_verified(res: dict) -> dict:
    """The CONSUMER's gate. Raises unless the result is verified AND nothing
    was left unverifiable.

    CO-5's other half: `verified` alone is not the contract, because a check
    this module could not decide is reported in `unverifiable` and a caller
    reading one field would miss it. The third conjunct is mine and is
    flagged as such -- a PROVENANCE result describes a run that already
    happened and must never be used to start a new one, so it is refused
    here too rather than left to the caller to notice."""
    bad = []
    if not res.get("verified"):
        bad.append("verified is False: "
                   + str(sorted(k for k, v in res.get("checks", {}).items()
                                if v is False)))
    if res.get("unverifiable"):
        bad.append(f"unverifiable checks remain: {res['unverifiable']} -- "
                   f"each is a question this module could not decide, not a "
                   f"question it answered yes")
    if res.get("provenance"):
        bad.append("this result is PROVENANCE for a run stamped before the "
                   "superseding entry; it may not start a new run")
    if bad:
        raise NotVerified(
            f"REFUSED for {res.get('ratification_ref')!r}: " + "; ".join(bad))
    return res


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
    """True / False / None(UNBINDABLE). Both ends evaluated, both PARSED.

    The open-ended token is read BEFORE parsing -- `null` is a declared word,
    not a date -- and everything else is a day or a refusal."""
    if "scope_from" in unbindable or "scope_from" not in fields:
        return None
    d = parse_day(str(day), "supply.day")
    if d < parse_day(fields["scope_from"], "block.scope_from"):
        return False
    to = fields.get("scope_to")
    if to is None:
        return None                       # absent is NOT null
    if isinstance(to, str) and to.strip().lower() in SCOPE_OPEN_TOKENS:
        return True                       # `null` = open, explicitly
    return d <= parse_day(to, "block.scope_to")


def check(supplied: dict, ratification_ref: str,
          register_text: str | None = None, *,
          now_utc: str | None = None,
          stamped_at: str | None = None) -> dict:
    """VERIFY / REFUSE / report-unbindable.  Decides nothing else.

    `now_utc` is the clock the closure check reads -- injectable so a test
    can place itself either side of a day boundary rather than waiting for
    one.  `stamped_at` is the `as_of_utc` of an EXISTING receipt: supplied,
    the supersession question becomes "was this ratification in force WHEN
    THE RUN WAS STAMPED", and a run that predates its superseder is
    PROVENANCE rather than a refusal.  Omitted, the run is a NEW one and a
    superseded ref refuses."""
    import datetime as _dt
    # AN EMPTY now_utc MAY MEAN THE WALL CLOCK ONLY IF THE EMISSION SAYS SO.
    # Absent (None) is the wall clock and is RECORDED as such; anything else
    # is injected and is parsed. An empty STRING is not absent -- it is an
    # unparsable instant, and it refuses.
    if now_utc is None:
        now_dt = _dt.datetime.now(_dt.timezone.utc)
        now_utc = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        now_utc_source = "wall_clock"
    else:
        now_dt = parse_instant(now_utc, "now_utc")
        now_utc_source = "injected"
    # CO-6: PARSE THE STAMP WHENEVER ONE IS SUPPLIED, before any branch.
    # It used to be parsed only on the SUPERSEDED path, so the same garbage
    # refused on R-418 and was echoed verbatim into a verified result on
    # R-419. A stamp is a claim about a receipt whether or not a superseder
    # exists TODAY -- and a superseder that lands tomorrow would then be
    # weighed against a value nobody ever parsed. `None` stays "no receipt".
    stamped_dt = None
    if stamped_at is not None:
        stamped_dt = parse_instant(stamped_at, "stamped_at")
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
    provenance = False
    superseder_times: dict[str, str] = {}
    if supers:
        entries = {e["ref"]: e for e in all_entries(register_text)}
        for sref in supers:
            ts = entry_timestamp(entries[sref]["heading"])
            if ts is None:
                raise RatificationRefused(
                    f"REFUSED: {sref} supersedes {ratification_ref} but its "
                    f"heading carries no parsable register timestamp, so "
                    f"WHEN it took force cannot be computed. A supersession "
                    f"whose instant is unknown cannot be weighed against a "
                    f"receipt's stamp, and guessing the order is exactly "
                    f"what a stamp exists to avoid.")
            superseder_times[sref] = _norm_ts(
                ts, f"heading timestamp of {sref}")
        if stamped_at is None:
            raise RatificationRefused(
                f"REFUSED FOR A NEW RUN: {ratification_ref} is SUPERSEDED by "
                f"{', '.join(supers)}. A receipt already carrying "
                f"{ratification_ref} is provenance -- pass its `as_of_utc` as "
                f"`stamped_at` and this becomes a COMPUTED provenance "
                f"finding rather than a sentence in a report (CO-R3).")
        stamp = stamped_dt          # parsed at entry (CO-6)
        later = {r: t for r, t in superseder_times.items() if t > stamp}
        if len(later) == len(superseder_times):
            provenance = True          # every superseder postdates the stamp
        else:
            in_force = sorted(r for r, t in superseder_times.items()
                              if t <= stamp)
            raise RatificationRefused(
                f"REFUSED: {ratification_ref} was ALREADY superseded by "
                f"{in_force} at the stamped instant {stamp} -- the run did "
                f"not predate its superseder, so this is not provenance")

    block = bind_from_block(entry)
    if block is not None:
        # CO-5: a MALFORMED block is refused BY NAME, not left undecided.
        missing = [f for f in RATIFICATION_FIELDS if f not in block]
        if missing:
            raise RatificationRefused(
                f"REFUSED: {ratification_ref}'s ratification block is MISSING "
                f"{missing}. A missing field left the check UNDECIDED and "
                f"`verified` -- the conjunction of DECIDED checks -- still "
                f"read True, so a consumer reading that one field read an "
                f"absence as a pass (CO-5). A malformed block is refused.")
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

    # DE-R3: VALUES against the adopted vocabulary, before any of them is
    # believed. `population` is checked here for MEMBERSHIP and again below
    # for which member -- the two questions are different and their refusals
    # say so.
    for _f, _allowed in FIELD_VOCABULARY.items():
        if _f not in fields:
            continue
        if fields[_f] not in _allowed:
            raise RatificationRefused(
                f"REFUSED: {ratification_ref} field {_f!r} carries the VALUE "
                f"{fields[_f]!r}, which is not in the adopted vocabulary "
                f"{list(_allowed)}. This is a WRONG VALUE, not a missing "
                f"field and not an undecidable one -- round 10 made absence "
                f"refuse and left nonsense verifying clean (DE-R3).")
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
        # CO-R1: the ratified `present_source` is the market LEDGER, which
        # runs AHEAD of the tape on an open day (measured 09-02 11:16Z:
        # ledger 137 vs tape 135 per coin, the 14 ledger-only windows being
        # the 11:15Z/11:20Z starts). The driver already refuses an open day
        # and the bridge already refuses ledger-only windows; this module had
        # NO notion of closure at all, so it would certify a population that
        # is still growing. DECIDED, never None: a day is closed iff its own
        # end instant has passed.
        "day_closed": day_end_instant(str(supplied["day"])) <= now_dt,
    }
    return {
        "protocol": PROTOCOL,
        "refusal_scope": "a refusal here is about STARTING A RUN; a receipt "
                         "already carrying a ref keeps it as provenance",
        "now_utc": now_utc,
        "now_utc_source": now_utc_source,
        "stamped_at": (stamped_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                       if stamped_dt else None),
        "stamped_at_raw": stamped_at,
        "provenance": provenance,
        "superseded_by": sorted(superseder_times),
        "superseder_times": {r: t.strftime("%Y-%m-%dT%H:%M:%SZ")
                             for r, t in superseder_times.items()},
        "ratification_ref": ratification_ref,
        "binding_source": source,
        "entry_heading": entry["heading"][:160],
        "bound_fields": fields,
        "binding_evidence": evidence,
        "unbindable_from_prose": unbindable,
        "supply_population": pop,
        "checks": checks,
        "verified": all(v for v in checks.values() if v is not None),
        "verified_for_new_run": (all(v for v in checks.values()
                                     if v is not None)
                                 and not [k for k, v in checks.items()
                                          if v is None]
                                 and not provenance),
        "unverifiable": sorted(k for k, v in checks.items() if v is None),
        "decides": "nothing -- this reports; admission is the coordinator's "
                   "act and accrual is decided elsewhere (R-418)",
    }


# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 84


def selftest() -> int:
    import de_admissible_windows as daw
    import ev_replay_seam as seam
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_ratification_check] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses_nv(fn, label):
        try:
            fn()
        except NotVerified:
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_ratification_check] FAIL (no refusal): {label}")

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

    # ---- CO-5: a malformed block REFUSES, and the consumer gate ---------
    noto2 = fixture_register().replace("scope_to: null\n", "")
    refuses(lambda: check(sup, "R-900", noto2),
            "CO-5 CLOSED: a block MISSING a required field REFUSES BY NAME "
            "-- it used to leave the check undecided while `verified`, the "
            "conjunction of DECIDED checks, still read True, so a consumer "
            "reading that one field read an ABSENCE as a PASS",
            needle="is MISSING ['scope_to']")
    ok(len(RATIFICATION_FIELDS) == 10 and "scope_to" in RATIFICATION_FIELDS,
       f"and the required-field list is now USED rather than declared: "
       f"{len(RATIFICATION_FIELDS)} fields, the adopted format's own "
       f"(it was a dead constant, which is why the gap existed)")
    ok(require_verified(check(sup, "R-419")) is not None,
       "CONSUMER GATE: `require_verified` ADMITS the real R-419 result")
    refuses_nv(lambda: require_verified(
        {"ratification_ref": "X", "verified": True,
         "unverifiable": ["day_in_scope"], "checks": {}}),
        "KNOWN-BAD: it REFUSES a result that is `verified` but leaves a "
        "check UNVERIFIABLE -- the pair is the contract, not the one field")
    refuses_nv(lambda: require_verified(
        {"ratification_ref": "X", "verified": False, "unverifiable": [],
         "checks": {"day_closed": False}}),
        "and it refuses an unverified result, NAMING the failed check")
    refuses_nv(lambda: require_verified(
        {"ratification_ref": "X", "verified": True, "unverifiable": [],
         "checks": {}, "provenance": True}),
        "and a PROVENANCE result may not start a new run -- my own third "
        "conjunct, flagged as an addition to the dispatched contract")

    # ---- DE-R3: nonsense VALUES refuse, and the message says VALUE ------
    for _f, _v in (("present_source", "/etc/passwd"),
                   ("scope_days", "WHATEVER"),
                   ("revocable_by", "DE")):
        refuses(lambda f=_f, v=_v: check(sup, "R-900",
                                         fixture_register(**{f: v})),
                f"DE-R3 CLOSED ({_f}={_v!r}): a NONSENSE VALUE REFUSES -- it "
                f"used to return verified True with unverifiable [], because "
                f"round 10 made a MISSING field refuse and left a wrong one "
                f"alone",
                needle="carries the VALUE")
    try:
        check(sup, "R-900", fixture_register(present_source="/etc/passwd"))
        _msg = ""
    except RatificationRefused as _e:
        _msg = str(_e)
    ok("VALUE" in _msg and "MISSING" not in _msg,
       "and the message says VALUE, never MISSING: 'a field nobody "
       "supplied', 'a field this checker cannot decide' and 'a field with a "
       "wrong value' are three different things that must not look alike")
    ok(FIELD_VOCABULARY["present_source"] == (LEDGER_PATH,)
       and LEDGER_PATH == "data/pm_5min/markets.jsonl"
       and "kind" not in FIELD_VOCABULARY,
       f"the ledger path is named ONCE ({LEDGER_PATH}); `kind` is "
       f"deliberately NOT in the vocabulary loop, because 'this is not a "
       f"ratification' is a different complaint from 'this value is wrong'")
    ok(check(sup, "R-419")["verified"],
       "POSITIVE CONTROL: the REAL R-419 still verifies against the "
       "vocabulary -- the values it carries are the adopted ones")
    ok(set(FIELD_VOCABULARY["sampling"]) == {"NONE", "STRATIFIED", "CAPPED"},
       "and `sampling` admits the LEGITIMATE values, not just the one this "
       "programme uses: restricting it to NONE would hardcode that no "
       "sampled ratification can ever exist, and it swallowed the SEMANTIC "
       "self-contradiction refusal whose complaint is different")

    # ---- DE-R4: the unknown-population branch, driven -------------------
    refuses(lambda: check(sup, "R-900",
                          fixture_register(population="SOMETHING_NEW")),
            "DE-R4: an UNKNOWN population REFUSES when driven -- the branch "
            "existed and had NO control in the audit; it has one now",
            needle="carries the VALUE")

    # ---- CO-R1: closure as a DECIDED check ------------------------------
    ok(check(sup, "R-419")["checks"]["day_closed"] is True,
       "CO-R1: 09-01 reads day_closed TRUE today -- the ratified "
       "present_source is the market LEDGER, which runs AHEAD of the tape on "
       "an open day, and this module had no notion of closure at all")
    ok(day_end_utc("20260901") == "2026-09-02T00:00:00Z"
       and day_end_utc("20260902") == "2026-09-03T00:00:00Z",
       f"a day is FINISHED at its own end instant: "
       f"{day_end_utc('20260901')}")
    sup02 = dict(sup, day="20260902")
    open_day = check(sup02, "R-419", now_utc="2026-09-02T11:16:00Z")
    ok(open_day["checks"]["day_closed"] is False
       and not open_day["verified"],
       "KNOWN-BAD: 09-02 at 11:16Z reads day_closed FALSE and the result is "
       "NOT verified -- the exact instant the reviewer measured ledger 137 "
       "against tape 135")
    closed_later = check(sup02, "R-419", now_utc="2026-09-03T00:00:00Z")
    ok(closed_later["checks"]["day_closed"] is True,
       "POSITIVE CONTROL: the same day at its own end instant reads TRUE "
       "(<=, so the boundary itself closes it) -- the clock is injectable, "
       "so the control does not wait for midnight")
    refuses_nv(lambda: require_verified(open_day),
               "and the consumer gate REFUSES the open day rather than "
               "leaving closure to the caller to notice")

    # ---- DE10-R1: every temporal comparison PARSED, both directions -----
    # GARBAGE THAT WOULD HAVE SORTED PERMISSIVE and GARBAGE THAT WOULD HAVE
    # SORTED RESTRICTIVE, per field. "zzzz" sorts AFTER "2026-…" and read as
    # the future; "aaaa" sorts BEFORE and read as the past. The same defect
    # was permissive in two fields and restrictive in a third, and neither
    # face of it surfaced.
    for _val in ("zzzz", "aaaa"):
        refuses(lambda v=_val: check(sup, "R-419", now_utc=v),
                f"DE10-R1 ({'permissive' if _val > '2' else 'restrictive'} "
                f"direction): now_utc={_val!r} REFUSES -- lexically it read "
                f"as {'the future, so day_closed was True' if _val > '2' else 'the past'}",
                needle="not an instant")
    for _val in ("zzzz", "aaaa"):
        refuses(lambda v=_val: check(sup, "R-900",
                                     fixture_register(scope_to=v)),
                f"scope_to={_val!r} REFUSES in {'the permissive' if _val > '2' else 'the restrictive'} "
                f"direction too",
                needle="block.scope_to")
        refuses(lambda v=_val: check(sup, "R-900",
                                     fixture_register(scope_from=v)),
                f"and scope_from={_val!r} likewise -- the field that used to "
                f"go silently RESTRICTIVE",
                needle="block.scope_from")
    for _bad in (123, 20260901, ["2026-09-02T00:00:00Z"]):
        refuses(lambda v=_bad: check(sup, "R-419", now_utc=v),
                f"KNOWN-BAD: a NON-STRING now_utc ({_bad!r}) REFUSES BY NAME "
                f"-- a TypeError from a comparison is a crash wearing a "
                f"refusal's clothes",
                needle="not a string")
    ok(check(sup, "R-419")["now_utc_source"] == "wall_clock"
       and check(sup, "R-419",
                 now_utc="2026-09-02T00:00:00Z")["now_utc_source"]
       == "injected",
       "AN ABSENT now_utc MAY MEAN THE WALL CLOCK ONLY BECAUSE THE EMISSION "
       "SAYS SO: `now_utc_source` reads wall_clock when it was defaulted and "
       "injected when it was given")
    refuses(lambda: check(sup, "R-419", now_utc=""),
            "and an EMPTY STRING is NOT absent -- it is an unparsable "
            "instant and refuses, rather than quietly becoming the clock",
            needle="not an instant")
    refuses(lambda: check(sup, "R-418", stamped_at="not-a-time"),
            "KNOWN-BAD: an unparsable `stamped_at` REFUSES BY NAME",
            needle="stamped_at")
    badhead = ("### R-902 — 2026-09-02T09:00Z — coordinator: R-ADMISS\n\n"
               + fixture_register("R-902").split("\n\n", 1)[1]
               ).replace("\n\n## next\n", "\n\n") + (
        "### R-903 — 2026-99-99T99:99Z — coordinator: R-ADMISS\n\n"
        + fixture_register("R-903", supersedes="R-902").split("\n\n", 1)[1])
    refuses(lambda: check(sup, "R-902", badhead,
                          stamped_at="2026-09-02T10:00:00Z"),
            "KNOWN-BAD: a superseder heading whose timestamp is well-SHAPED "
            "but not a real instant (2026-99-99T99:99Z) REFUSES -- the "
            "regex admits it and only parsing rejects it",
            needle="not an instant")

    # ---- the boundary, unchanged -----------------------------------------
    ok(day_end_utc("20260901") == "2026-09-02T00:00:00Z",
       f"the day-end instant is unchanged: {day_end_utc('20260901')}")
    for _n, _want in (("2026-09-01T23:59:59Z", False),
                      ("2026-09-02T00:00:00Z", True),
                      ("2026-09-02T00:00:01Z", True)):
        ok(check(sup, "R-419", now_utc=_n)["checks"]["day_closed"] is _want,
           f"boundary held: now_utc {_n} -> day_closed {_want} "
           f"(day+1 00:00:00Z <=, so -1s False, 0 True, +1s True)")
    refuses(lambda: check(dict(sup, day="2026-09-01"), "R-419"),
            "and the SUPPLY's own day is parsed too: a day that is not "
            "YYYYMMDD refuses rather than sorting",
            needle="supply.day")

    # ---- CO-R3: supersession against a STAMP ----------------------------
    prov = check(sup, "R-418", stamped_at="2026-09-02T10:30:00Z")
    ok(prov["provenance"] is True and prov["verified_for_new_run"] is False
       and prov["binding_source"] == "PROSE_GRANDFATHERED",
       f"CO-R3: R-418 stamped 10:30Z -- BEFORE R-419's "
       f"{prov['superseder_times']['R-419']} -- is PROVENANCE, computed "
       f"rather than asserted, and NEVER a new-run pass")
    refuses(lambda: check(sup, "R-418", stamped_at="2026-09-02T11:30:00Z"),
            "KNOWN-BAD: the same ref stamped 11:30Z, AFTER the superseder, "
            "REFUSES -- the run did not predate its superseder, so it is not "
            "provenance",
            needle="ALREADY superseded")
    refuses(lambda: check(sup, "R-418"),
            "and with NO stamp the current refusal stands: a new run under a "
            "superseded ratification, refused by name",
            needle="SUPERSEDED by R-419")
    ok(entry_timestamp("### R-419 — 2026-09-02T11:03Z — coordinator: x")
       == "2026-09-02T11:03Z"
       and entry_timestamp("### R-419 — coordinator: no timestamp") is None,
       "the entry timestamp is PARSED from the heading, and a heading "
       "without one parses to None rather than to a guess")
    nots = (fixture_register("R-902").replace("\n\n## next\n", "\n\n")
            .replace("### R-902 — coordinator",
                     "### R-902 — coordinator")
            + fixture_register("R-903", supersedes="R-902"))
    refuses(lambda: check(sup, "R-902", nots,
                          stamped_at="2026-09-02T10:00:00Z"),
            "KNOWN-BAD: a superseder whose heading carries NO parsable "
            "timestamp REFUSES -- a supersession whose instant is unknown "
            "cannot be weighed against a stamp, and guessing the order is "
            "what the stamp exists to avoid",
            needle="no parsable register timestamp")

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
    noto = _blk().replace("scope_to: null\n", "")
    refuses(lambda: check(sup, "R-902", noto),
            "AN ABSENT scope_to IS NOT `null` -- and after CO-5 it does not "
            "even reach the UNBINDABLE reading: a malformed block REFUSES, "
            "which is the stronger form of 'absence is never the permissive "
            "answer'",
            needle="is MISSING")
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
    chain = sup_chain_fixture()
    nots = (fixture_register("R-902").replace("\n\n## next\n", "\n\n")
            + fixture_register("R-903", supersedes="R-902"))
    stamped_chain = (
        "### R-902 — 2026-09-02T09:00Z — coordinator: R-ADMISS\n\n"
        "```ratification\nref: R-902\nkind: R-ADMISS\n"
        f"population: {POP_FULL}\nsampling: NONE\n"
        "present_source: data/pm_5min/markets.jsonl\n"
        "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
        "scope_to: null\nrevocable_by: USER\nsupersedes: null\n```\n\n"
        "### R-903 — 2026-09-02T10:00Z — coordinator: R-ADMISS\n\n"
        "```ratification\nref: R-903\nkind: R-ADMISS\n"
        f"population: {POP_FULL}\nsampling: NONE\n"
        "present_source: data/pm_5min/markets.jsonl\n"
        "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
        "scope_to: null\nrevocable_by: USER\nsupersedes: R-902\n```\n\n"
        "## next\n")
    # (bad args/kwargs, control args/kwargs)
    cases = {
        "entry_absent": (((sup, "R-901", good), {}),
                         ((sup, "R-900", good), {})),
        "not_a_ratification": (((sup, "R-900",
                                 fixture_register(kind="STATUS_NOTE")), {}),
                               ((sup, "R-900", good), {})),
        "sampled_population": (((sup, "R-900",
                                 fixture_register(population=POP_SAMPLED)),
                                {}),
                               ((sup, "R-900", good), {})),
        "counts_do_not_sum": (((bad_counts, "R-900", good), {}),
                              ((sup, "R-900", good), {})),
        "selection_field": (((dict(sup, sampled=True), "R-900", good), {}),
                            ((sup, "R-900", good), {})),
        "no_block_not_grandfathered": (
            ((sup, "R-905", "### R-905 — coordinator: R-ADMISS ratification, "
                            "FULL supplied complement, no sampling\n\nx\n"),
             {}),
            ((sup, "R-900", good), {})),
        "block_ref_mismatch": (
            ((sup, "R-900", fixture_register().replace("ref: R-900",
                                                       "ref: R-777")), {}),
            ((sup, "R-900", good), {})),
        "self_contradicting_sampling": (
            ((sup, "R-900", fixture_register(sampling="STRATIFIED")), {}),
            ((sup, "R-900", good), {})),
        "superseded": (((sup, "R-902", chain), {}),
                       ((sup, "R-903", chain), {})),
        # --- round 10's three new refusals -----------------------------
        "malformed_block_missing_field": (
            ((sup, "R-900", fixture_register().replace("scope_to: null\n",
                                                       "")), {}),
            ((sup, "R-900", good), {})),
        "superseder_timestamp_unparsable": (
            ((sup, "R-902", nots), {"stamped_at": "2026-09-02T10:00:00Z"}),
            ((sup, "R-902", stamped_chain),
             {"stamped_at": "2026-09-02T09:30:00Z"})),
        "unknown_population_value": (
            ((sup, "R-900", fixture_register(population="SOMETHING_NEW")), {}),
            ((sup, "R-900", good), {})),
        "nonsense_field_value": (
            ((sup, "R-900", fixture_register(present_source="/etc/passwd")),
             {}),
            ((sup, "R-900", good), {})),
        "unparsable_now_utc": (((sup, "R-419", None), {"now_utc": "zzzz"}),
                               ((sup, "R-419", None),
                                {"now_utc": "2026-09-02T00:00:00Z"})),
        "non_string_now_utc": (((sup, "R-419", None), {"now_utc": 123}),
                               ((sup, "R-419", None),
                                {"now_utc": "2026-09-02T00:00:00Z"})),
        "unparsable_scope_to": (
            ((sup, "R-900", fixture_register(scope_to="zzzz")), {}),
            ((sup, "R-900", good), {})),
        "unparsable_scope_from": (
            ((sup, "R-900", fixture_register(scope_from="aaaa")), {}),
            ((sup, "R-900", good), {})),
        "unparsable_stamped_at": (
            ((sup, "R-418", None), {"stamped_at": "nope"}),
            ((sup, "R-418", None),
             {"stamped_at": "2026-09-02T10:30:00Z"})),
        "already_superseded_at_stamp": (
            ((sup, "R-902", stamped_chain),
             {"stamped_at": "2026-09-02T10:30:00Z"}),
            ((sup, "R-902", stamped_chain),
             {"stamped_at": "2026-09-02T09:30:00Z"})),
    }
    per: dict[str, dict] = {}
    for name, ((bad_a, bad_k), (ctl_a, ctl_k)) in cases.items():
        try:
            check(*bad_a, **bad_k)
            live = False
        except RatificationRefused:
            live = True
        try:
            check(*ctl_a, **ctl_k)
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
