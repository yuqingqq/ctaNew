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
supersedes: null              # the prior ref or `null`; SINGULAR
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
        # SITE: parse_instant#1
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
    # SITE: parse_instant#2
    raise RatificationRefused(
        f"REFUSED: {field} carries {value!r}, which is not an instant in "
        f"{list(INSTANT_FORMATS)}. Compared as a STRING it would sort "
        f"against real timestamps and read as the future or the past "
        f"depending only on its first character (DE10-R1).")


def parse_day(value, field: str):
    """A YYYYMMDD day, or a REFUSAL naming the field and the value."""
    import datetime as _dt
    if not isinstance(value, str):
        # SITE: parse_day#1
        raise RatificationRefused(
            f"REFUSED: {field} is {value!r} ({type(value).__name__}), not a "
            f"string")
    try:
        return _dt.datetime.strptime(value.strip(), DAY_FORMAT).replace(
            tzinfo=_dt.timezone.utc)
    except ValueError:
        # SITE: parse_day#2
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


#: DE14-R1: `supersedes` is the one field whose VALUE decides another
#: entry's check, and it was matched by raw string equality and validated
#: nowhere. On a two-entry chain, a superseder declaring `supersedes:` empty
#: (or `r-902`, `R-9O2`, `R-902 (partial)`, `R-902, R-901`) left the entry it
#: names verifying for NEW RUNS -- the DE12-R2 shape one entry over, in the
#: field that drives this module's strongest refusal. The register could hold
#: an entry simultaneously "refused if you check it" and "invisible as a
#: superseder".
#:
#: SINGULAR STAYS SINGULAR. R-419 section 4 defines `supersedes` as "the
#: prior ref or null" -- one ref. A comma list REFUSES by name; declaring a
#: plural spelling is a spec change and the coordinator's, not this module's.
SUPERSEDES_NULL = "null"


def validate_supersedes(value, where: str) -> str | None:
    """The shape of a `supersedes` field, or a REFUSAL naming where it came
    from.  Returns the ref it names, or None for the declared `null`."""
    if value is None:
        # SITE: validate_supersedes#1
        raise RatificationRefused(
            f"REFUSED: {where} carries no `supersedes`. The adopted block "
            f"requires it -- `null` when nothing is superseded -- and an "
            f"absent one cannot be told from a supersession nobody wrote.")
    if not isinstance(value, str):
        # SITE: validate_supersedes#2
        raise RatificationRefused(
            f"REFUSED: {where} `supersedes` is {value!r} "
            f"({type(value).__name__}), not a string")
    v = value.strip()
    if not v:
        # SITE: validate_supersedes#3
        raise RatificationRefused(
            f"REFUSED: {where} `supersedes` is EMPTY. An empty value is "
            f"absence in place -- and here it silently means 'supersedes "
            f"nothing', which is how an entry stays invisible as a "
            f"superseder while refusing if anyone checks it (DE14-R1).")
    if v == SUPERSEDES_NULL:
        return None
    if "," in v:
        # SITE: validate_supersedes#4
        raise RatificationRefused(
            f"REFUSED: {where} `supersedes` names MORE THAN ONE ref "
            f"({v!r}). R-419 section 4 defines it as the prior ref or "
            f"`null`, singular. A plural spelling is a SPEC CHANGE and the "
            f"coordinator's to declare; this module refuses rather than "
            f"inventing one.")
    if not REF_RE.match(v):
        # SITE: validate_supersedes#5
        raise RatificationRefused(
            f"REFUSED: {where} `supersedes` is {v!r}, which is neither "
            f"`{SUPERSEDES_NULL}` nor a well-shaped ref "
            f"({REF_RE.pattern}). Matched by raw equality it would simply "
            f"fail to match and supersede nothing, silently.")
    return v


#: DE20-R1: "AN ENTRY EXISTS" WAS DERIVED THREE TIMES -- a line map here,
#: an entry map in the supersession branch, and a set comprehension at
#: `check#16` -- over the same unfiltered call on the same text. They could
#: not disagree, and that is the point: the finding is the DRIFT SURFACE.
#: If one site ever gains a filter (skip malformed entries, ignore a
#: section, restrict to R-ADMISS) the others do not follow, and the two
#: ends of ONE rule -- the later entry's `supersedes` and the entry's own --
#: would answer "exists" differently. That asymmetry is what DE16-R2 and
#: DE18-R1 were both about, from opposite ends. One implementation, and it
#: carries the LINE as well as the ref, because existence without direction
#: is DE20-R2 one line down.
#: DE22-R1: AND A DICT ANSWERS BY LAST-WINS. A ref heading two entries
#: resolved to the LATER one, refused by nothing and reported by nothing --
#: and the register HAS one: R-6 heads an entry ABOUT R-6 (the CO-4 shape,
#: this time the ref itself in a heading) as well as the real one. Measured
#: on a fixture where it matters (R-902 at lines 0 and 30, R-903 at 15
#: declaring `supersedes: R-902`): the supersession was LOST, R-902
#: verified for a new run, and R-903 was REFUSED at `check#18` quoting line
#: 30 -- BOTH ENDS WRONG AT ONCE, and the refusal quoting a line its author
#: never wrote.
#:
#: The register is append-only and only 217 of its 437 headings carry the
#: stamped form, so "unstamped is not an entry" is not available and the
#: existing duplicate cannot be edited away. So (R-446 section 3): REFUSE
#: where a duplicate can REACH AN ANSWER -- it is the subject of the check,
#: it is named by any `supersedes:` in the register, or any occurrence of
#: it carries a ratification block -- and REPORT it otherwise, keeping the
#: FIRST occurrence. Which occurrence is kept is a RULE, not a judgement:
#: nothing here names a ref and there is no allowlist, so the day R-6 gains
#: a block or a superseder the answer becomes a refusal by itself.
class EntryIndex(dict):
    """ref -> entry, plus what the parse found duplicated."""


def entry_index(register_text: str, *, subject: str | None = None
                ) -> EntryIndex:
    """ref -> entry, for every entry in the register. The one place that
    answers whether an entry exists and where it stands.

    `line` is the parser's 0-BASED index into `register_text.split("\n")`
    -- the same number `check#18` prints, so a reader comparing a message
    with an editor's 1-based gutter adds one."""
    entries = all_entries(register_text)
    at: dict[str, list[int]] = {}
    for e in entries:
        at.setdefault(e["ref"], []).append(e["line"])
    dups = {r: ls for r, ls in sorted(at.items()) if len(ls) > 1}
    idx = EntryIndex()
    for e in entries:
        idx.setdefault(e["ref"], e)          # the FIRST occurrence, by rule
    idx.duplicate_refs = dups
    idx.kept = ("FIRST occurrence, by rule -- computed from the parse; no "
                "ref is named in this module and there is no allowlist")
    if dups:
        if subject in dups:
            # SITE: entry_index#1
            raise RatificationRefused(
                f"REFUSED: {subject} heads {len(dups[subject])} entries in "
                f"this register, at 0-based lines {dups[subject]}, and it "
                f"is the SUBJECT of this check. Which one is asked about "
                f"cannot be decided by a reader of the file, and taking "
                f"either silently is how a supersession is lost at one end "
                f"while a direction error is manufactured at the other "
                f"(DE22-R1).")
        # DE24-R1: this read EVERY fenced block, owned or QUOTED, so a
        # non-ratifying sweep entry quoting a block that named a
        # duplicated ref made every OTHER entry's check refuse -- a narrow
        # return of DE16-R1, with the quotation refusing a check instead
        # of superseding an entry. That path can decide NOTHING:
        # `superseded_by` reads own blocks only, and round 18 settled that
        # a quoted block is not that entry's ratification. By R-446 §3's
        # own criterion -- refuse where the duplication CAN reach an
        # answer, report where it cannot -- the case belongs on the
        # reporting side, and the rule stops leaning on R-432 §1, a FORMAT
        # convention about prose, when the module has an ownership
        # predicate of its own. RULED in band (R-450 §3): "(ii) is named
        # by the `supersedes:` of any entry's OWN ratification block".
        #
        # ORDERING, unchanged: `own_ratification_blocks` raises on a
        # malformed entry, and it is already called on this same path (by
        # `superseded_by` for every later entry, and by `check()` on the
        # entry under check), so nothing is refused earlier than before.
        named = {str(blk.get("supersedes", "")).strip()
                 for e in entries for blk, _ in own_blocks_quiet(e)}
        reached = [r for r in dups if r in named]
        if reached:
            # SITE: entry_index#2
            raise RatificationRefused(
                f"REFUSED: {reached} head more than one entry "
                f"({ {r: dups[r] for r in reached} }, 0-based lines) AND "
                f"are named by a `supersedes:` in this register. WHERE the "
                f"target stands decides whether the supersession is read "
                f"at all, so a duplicate an OWN block points at reaches "
                f"an answer (DE22-R1). Only entries' OWN ratification "
                f"blocks are read here (R-450 §3): a block QUOTED inside "
                f"a non-ratifying entry supersedes nothing, so it cannot "
                f"make a duplication reach an answer either, and refusing "
                f"on it was the narrow return of DE16-R1 -- a quotation "
                f"deciding another entry's check.")
        # R-451 §3: OWN blocks, the same criterion as (ii). A QUOTED block
        # under a duplicated heading reaches no answer -- `superseded_by`
        # reads the kept occurrence's OWN blocks, and a quotation is not
        # own whichever occurrence were kept -- so it belongs on the
        # reporting side. A SELF-quotation counts as own and refuses:
        # that is the predicate's definition, and fail-closed.
        blocky = [r for r in dups
                  if any(own_blocks_quiet(e) for e in entries
                         if e["ref"] == r)]
        if blocky:
            # SITE: entry_index#3
            raise RatificationRefused(
                f"REFUSED: {blocky} head more than one entry "
                f"({ {r: dups[r] for r in blocky} }, 0-based lines) and at "
                f"least one occurrence carries an OWN ratification block "
                f"(ref == the heading, kind R-ADMISS; a QUOTED block is "
                f"somebody else's and reaches no answer here -- R-451 §3). "
                f"Two headings under one ref with an own block among them "
                f"is the "
                f"heading-level form of `own_ratification_blocks#1`: a "
                f"corrected ratification would be shadowed by the one it "
                f"corrects, and which is read would depend on the order "
                f"they were appended (DE22-R1).")
    return idx


def superseded_by(register_text: str, ref: str) -> list[str]:
    """Refs of LATER entries whose block declares it supersedes `ref`."""
    idx = entry_index(register_text, subject=ref)
    entries = list(idx.values())
    pos = {r: e["line"] for r, e in idx.items()}
    if ref not in pos:
        return []
    out = []
    for e in entries:
        if e["line"] <= pos[ref]:
            continue
        # DE16-R1: the entry's OWN block(s) only. `bind_from_block` reads the
        # FIRST fence, so a sweep entry QUOTING a spelling was read as that
        # sweep's ratification -- which is how a quoted `supersedes: R-419`
        # made R-419 read superseded by an entry whose next sentence says it
        # ratifies nothing, and how a quoted malformed one made every
        # earlier ref REFUSE for a reason about somebody else's block.
        for blk in own_ratification_blocks(e):
            # THE LATER ENTRY'S OWN FIELD IS VALIDATED. It is never otherwise
            # checked, and its value decides whether THIS ref may start a run.
            named = validate_supersedes(blk.get("supersedes"),
                                        f"{e['ref']} (a later entry)")
            # DE16-R2: SHAPE IS NOT EXISTENCE. `R-9021` -- one digit from
            # `R-902` -- is perfectly well-shaped, matched nothing, and left
            # the ratification it meant to supersede verifying for new runs
            # in silence. That is DE14-R1's own sentence ("a failed match
            # says nothing") surviving the fix that quoted it, and `check#1`
            # already refuses a ref that names no entry FOR THE SAME REASON:
            # a well-formed ref to a missing entry looks exactly like a
            # valid one. `pos` is already built over every entry.
            #
            # SCOPE, stated: this is the loop where the value DECIDES
            # something. The entry under check's own `supersedes` is
            # validated for SHAPE (check-side) and its target's existence
            # becomes this question when someone checks that target.
            if named is not None and named not in pos:
                # SITE: superseded_by#1
                raise RatificationRefused(
                    f"REFUSED: {e['ref']} (a later entry) declares "
                    f"`supersedes: {named}`, and NO ENTRY {named} exists in "
                    f"this register. A well-shaped ref that names nothing "
                    f"supersedes nothing SILENTLY -- the ratification it "
                    f"was written to retire keeps verifying for new runs, "
                    f"which is the fail-OPEN direction (DE16-R2).")
            if named == ref:
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

def _undef_note(empty: list) -> str:
    """NAME AN UNDEFINED FIELD AS UNDEFINED. The EMPTY message ranges over
    `block.items()`, so an empty `notes:` line would otherwise be reported as
    though `notes` were a ratification field."""
    undefined = [f for f in empty if f not in RATIFICATION_FIELDS]
    if not undefined:
        return ""
    return (f" Of these, {undefined} are NOT ratification fields at all -- "
            f"they are undefined in the adopted block, so an empty undefined "
            f"line is absence in place, not a missing ratification field.")


#: DE16-R3: a block is parsed into a dict by `k, v = line.split(":", 1)`,
#: so a REPEATED key was silently LAST-WINS. Two `supersedes:` lines naming
#: `R-902` then `R-901` left the first target DROPPED WITHOUT A WORD and the
#: earlier ratification verifying for new runs -- fail-OPEN, the direction
#: DE14-R1 was about. Absence in place and presence twice are the two shapes
#: of one defect and this module already refused the first, so the parse
#: REPORTS the duplicates and the callers refuse them by name. It reports
#: rather than raises because a QUOTED block's duplicates are not this
#: module's business (DE16-R1) -- only an entry's OWN block is.
def _parse_block(body: str) -> tuple[dict, list[str]]:
    """(fields, keys that appeared MORE THAN ONCE) for one fenced block."""
    out: dict[str, Any] = {}
    dups: list[str] = []
    for line in body.split("\n"):
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        if k in out and k not in dups:
            dups.append(k)
        out[k] = v.strip()
    return out, sorted(dups)


def _fenced_blocks(entry: dict) -> list[tuple[dict, list[str]]]:
    """EVERY fenced ratification block in the entry, in order.

    `bind_from_block` reads the FIRST fence; that is what made a QUOTED
    block indistinguishable from an entry's own (DE16-R1). Ownership needs
    all of them, because the quotation may come first."""
    text = entry["heading"] + "\n" + entry["body"]
    return [_parse_block(m.group(1))
            for m in re.finditer(r"```ratification\n(.*?)```", text, re.S)]


def _heading_ref(entry: dict) -> str | None:
    m = HEADING_RE.match(entry["heading"])
    return m.group(1) if m else None


#: DE16-R1: A BLOCK QUOTED INSIDE AN ENTRY IS NOT THAT ENTRY'S RATIFICATION.
#: `superseded_by` validated the first fence of every later entry, so a
#: coordinator sweep SHOWING a spelling -- which is exactly where these
#: spellings get documented -- was read as the sweep's own ratification: a
#: quoted `supersedes: R-419` made R-419 read SUPERSEDED BY THE SWEEP, and a
#: quoted malformed one made every earlier ref's check REFUSE for a reason
#: with nothing to do with the ref being checked. The rule is the one
#: `check#8` already applies to the entry under check, one loop over: the
#: block's `ref` must be the entry's OWN heading ref, and its `kind` must be
#: R-ADMISS. Anything else is a quotation and is not read.
#: CO-9: THE SCANS MUST NOT ADJUDICATE. Round 26 moved (ii) from
#: `_fenced_blocks` -- which raises nowhere -- to
#: `own_ratification_blocks`, which RAISES (`#1` two own blocks, `#2` a
#: duplicated key), and the scan runs over EVERY entry. So a malformed own
#: block in an entry standing EARLIER than the subject refused the
#: subject's check: an entry before the subject can supersede nothing, and
#: the refusal named a defect with nothing to do with the ref being
#: checked -- the DE16-R1 sentence, one reader over. It fired only while
#: the register carried an UNRELATED duplicate, because the scan sits
#: inside `if dups:`; whether one entry refused another depended on R-6.
#:
#: My round-26 ordering note ("nothing is refused earlier than before") was
#: true of ORDER and silent on the SET, and the suite had no
#: earlier-malformed fixture, so its green was not the check on that claim.
#: It is now: fixtures C and C3 stand on either side of the subject.
#:
#: So ownership is asked TWICE, by two readers with different jobs. This
#: one only ANSWERS -- same predicate, no raise -- and is what the two
#: scans use; `own_ratification_blocks` stays the ADJUDICATING reader on
#: the path (`check()` on the subject, `superseded_by` on later entries),
#: where a malformed entry is the answer rather than an obstacle to one.
def own_blocks_quiet(entry: dict) -> list[tuple[dict, list[str]]]:
    """The entry's own ratification blocks as `(block, duplicated keys)`,
    ANSWERED not adjudicated.

    DE27-R1: the two conjuncts were spelled out HERE and again in
    `own_ratification_blocks` -- one predicate, two texts, edited under
    different pressures (this one by the scans, that one by the path).
    They agreed and every conjunct drop was red, so it was a DRIFT SURFACE
    rather than a gap -- the same shape DE20-R1 removed for "an entry
    exists", and asserted from the AST for the same reason: a predicate
    stated twice drifts without either copy noticing. The pairs are
    returned rather than the blocks alone so the adjudicating reader can
    consume this output and add ONLY its two raises."""
    ref = _heading_ref(entry)
    return [(blk, dups) for blk, dups in _fenced_blocks(entry)
            if str(blk.get("ref", "")).strip() == ref
            and str(blk.get("kind", "")).strip() == "R-ADMISS"]


def own_ratification_blocks(entry: dict) -> list[dict]:
    """The entry's OWN ratification block(s) -- ownership is asked of
    `own_blocks_quiet`, the one text that spells the two conjuncts. This
    reader adds only what ADJUDICATION means: more than one own block
    REFUSES, because two ratifications under one heading is a malformed
    entry rather than a choice between them, and a duplicated key inside
    one REFUSES because the parse is last-wins."""
    ref = _heading_ref(entry)
    own = own_blocks_quiet(entry)
    if len(own) > 1:
        # SITE: own_ratification_blocks#1
        raise RatificationRefused(
            f"REFUSED: {ref} carries {len(own)} ratification blocks of its "
            f"OWN (ref == the heading, kind R-ADMISS). Two ratifications "
            f"under one heading is a MALFORMED ENTRY, not a choice between "
            f"them -- and taking the first is how a corrected block would "
            f"be shadowed by the one it corrects. A correction supersedes "
            f"in band, as its own entry (rule 13).")
    for blk, dups in own:
        if dups:
            # SITE: own_ratification_blocks#2
            raise RatificationRefused(
                f"REFUSED: {ref}'s ratification block carries the key(s) "
                f"{dups} MORE THAN ONCE. The parse is last-wins, so the "
                f"earlier line is dropped in silence: `supersedes: R-902` "
                f"followed by `supersedes: R-901` supersedes NEITHER "
                f"(DE16-R3).")
    return [blk for blk, _ in own]


def bind_from_block(entry: dict) -> dict | None:
    """The PROPOSED machine-readable form, if the entry carries one.

    The FIRST fence, which is what the entry under check is bound from --
    a foreign block there is refused by `check#8` rather than skipped,
    because the entry under check is the one making the claim and a
    fail-closed refusal is the right answer to an entry whose first
    ratification block is somebody else's."""
    blocks = _fenced_blocks(entry)
    if not blocks:
        return None
    out, dups = blocks[0]
    if dups:
        # SITE: bind_from_block#1
        raise RatificationRefused(
            f"REFUSED: a ratification block carries the key(s) {dups} MORE "
            f"THAN ONCE. `k, v = line.split(':', 1)` into a dict is "
            f"LAST-WINS, so the earlier value is dropped without a word -- "
            f"two `supersedes:` lines named two refs and superseded NEITHER, "
            f"and the ratification kept verifying for new runs (DE16-R3). "
            f"Presence twice is absence in place wearing the other face.")
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
#: `scope_to: null` means OPEN -- and `null` is the ONLY spelling.
#:
#: DE12-R2: the set was `("null", "none", "")`, so `scope_to:` with NOTHING
#: after the colon read as OPEN-ENDED, `verified True`, `unverifiable []`,
#: silently. An empty value is ABSENCE IN PLACE, and this module already
#: refuses the line when it is missing entirely -- so the two shapes of the
#: same absence were treated oppositely, one refused and one permissive.
#:
#: `none` went with it, and that is a decision rather than tidying: R-419
#: section 4 adopted the block with `scope_to` (`null` = open) and nothing
#: else. A synonym I kept would be a SECOND SPELLING NOBODY RATIFIED, and
#: adding one to a coordinator-adopted format is not mine to do. If a synonym
#: is wanted it belongs in the block spec first.
SCOPE_OPEN_TOKENS = ("null",)


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
    # DE14-R3: EXACT, not case-folded. The constant said one spelling and
    # the comparison admitted four (`NULL`, `Null`, `nUlL`), so a user who
    # lowercases any other field is refused while one who uppercases this one
    # is silently granted an unbounded scope. The module case-folds nowhere
    # else. R-419 section 4 says `null`; restoring the exactness is not a
    # spec change, it is the code agreeing with its own constant.
    if isinstance(to, str) and to.strip() in SCOPE_OPEN_TOKENS:
        return True                       # `null` = open, explicitly
    return d <= parse_day(to, "block.scope_to")


def check(supplied: dict, ratification_ref: str,
          register_text: str | None = None, *,
          now_utc: str | None = None,
          stamped_at: str | None = None) -> dict:
    """VERIFY / REFUSE / report-unbindable.  Decides nothing else.

    `now_utc` is the clock the closure check reads -- injectable so a test
    can place itself either side of a day boundary rather than waiting for
    one.

    THE EMISSION ECHOES THE STAMP TWICE: `stamped_at` is the CANONICAL PARSE
    and `stamped_at_raw` is the value exactly as supplied -- so a receipt can
    be matched against its own field while a reader still sees what was
    handed in.  Both read None when no receipt was supplied.

    `stamped_at` is the `as_of_utc` of an EXISTING receipt: supplied,
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
        # SITE: check#1
        raise RatificationRefused(
            f"REFUSED: no register entry `### {ratification_ref} ` in "
            f"{REGISTER.name}. A well-formed ref to an entry that does not "
            f"exist looks exactly like a valid one, which is why the bridge's "
            f"shape check cannot be the last word.")

    # ONE index for this call, and it knows its SUBJECT: a duplicated ref
    # that reaches an answer refuses here (DE22-R1), and one that cannot is
    # carried into the emission as a reported fact.
    idx = entry_index(register_text, subject=ratification_ref)

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
        entries = idx                              # DE20-R1/DE22-R1: one
        for sref in supers:
            ts = entry_timestamp(entries[sref]["heading"])
            if ts is None:
                # SITE: check#2
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
            # SITE: check#3
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
            # SITE: check#4
            raise RatificationRefused(
                f"REFUSED: {ratification_ref} was ALREADY superseded by "
                f"{in_force} at the stamped instant {stamp} -- the run did "
                f"not predate its superseder, so this is not provenance")

    block = bind_from_block(entry)
    # DE16-R1's other half, on the entry under check: `bind_from_block`
    # reads the FIRST fence, so an entry carrying two ratifications of its
    # own would be bound from one of them with nothing said about the
    # other. Ownership is evaluated here purely for that refusal; the
    # BINDING stays the first fence, because a foreign block THERE is
    # `check#8`'s fail-closed refusal rather than something to skip past --
    # the entry under check is the one making the claim.
    own_ratification_blocks(entry)
    if block is not None:
        # CO-5: a MALFORMED block is refused BY NAME, not left undecided.
        empty = sorted(f for f, v in block.items()
                       if isinstance(v, str) and not v.strip())
        if empty:
            # SITE: check#5
            raise RatificationRefused(
                f"REFUSED: {ratification_ref}'s ratification block carries "
                f"EMPTY value(s) for {empty}. An empty value is ABSENCE IN "
                f"PLACE: the line is there and says nothing. It is neither a "
                f"MISSING field nor a wrong VALUE, and it used to read as "
                f"open-ended for `scope_to` while the same absence written as "
                f"a missing line refused (DE12-R2).{_undef_note(empty)}")
        # DE18-R2: the `supersedes` shape rule USED TO RUN HERE, on the
        # FIRST fence, whoever it belonged to -- so an entry whose first
        # fence was a QUOTATION carrying `supersedes: R-902, R-901` was
        # refused with "R-999's block names MORE THAN ONE ref" while
        # R-999's own block was well-formed. The refusal was right and the
        # message sent a reader to fix a block that was fine. It now runs
        # in the OWN-BLOCK branch below, AFTER `check#8`, so a foreign
        # first fence is spoken for by the refusal that can name the
        # mismatch and nothing else attributes it to an owner.
        missing = [f for f in RATIFICATION_FIELDS if f not in block]
        if missing:
            # SITE: check#6
            raise RatificationRefused(
                f"REFUSED: {ratification_ref}'s ratification block is MISSING "
                f"{missing}. A missing field left the check UNDECIDED and "
                f"`verified` -- the conjunction of DECIDED checks -- still "
                f"read True, so a consumer reading that one field read an "
                f"absence as a pass (CO-5). A malformed block is refused.")
    if block is None:
        # CO-4: prose binding survives for exactly one grandfathered ref.
        if ratification_ref not in GRANDFATHERED_PROSE_REFS:
            # SITE: check#7
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
            # SITE: check#8
            raise RatificationRefused(
                f"REFUSED: the block declares ref "
                f"{block.get('ref')!r} while the entry heading is "
                f"{heading_ref!r}. A block copied from another entry would "
                f"otherwise ratify under the wrong number.")
        # The shape rule on the entry's OWN block: its `supersedes` was
        # unvalidated too (`WHATEVER`, `/etc/passwd`, `R-418` all verified).
        if "supersedes" in block:
            own_named = validate_supersedes(block["supersedes"],
                                            f"{ratification_ref}'s block")
            # DE18-R1: SHAPE WAS NOT EXISTENCE HERE EITHER. `supersedes:
            # R-777` in the entry's own block verified True, [] while the
            # SAME STRING one entry over refused at `superseded_by#1`. The
            # deferral -- "the target's existence becomes that question
            # when someone checks the target" -- holds only if someone
            # does, and nothing makes that happen: a run stamping the NEW
            # ratification never causes the old one to be checked, so a
            # supersession written as a typo is examined by nobody while
            # both ends read clean. `check#1` already refuses this exact
            # shape one field over, for the reason it states in its own
            # message: a well-formed ref to an entry that does not exist
            # looks exactly like a valid one.
            own_idx = idx                          # DE20-R1/DE22-R1: one
            if own_named is not None and own_named not in own_idx:
                # SITE: check#16
                raise RatificationRefused(
                    f"REFUSED: {ratification_ref}'s own block declares "
                    f"`supersedes: {own_named}`, and NO ENTRY {own_named} "
                    f"exists in this register. A well-shaped ref that names "
                    f"nothing supersedes nothing SILENTLY -- and this is "
                    f"the end nobody checks later: the entry it meant to "
                    f"retire keeps verifying for new runs and no run on "
                    f"THIS one ever asks (DE18-R1).")
            # DE20-R2: EXISTENCE IS NECESSARY, NOT SUFFICIENT. Both shapes
            # below are well-formed, name entries that exist, and supersede
            # NOTHING while the entry claiming the supersession verifies --
            # DE18-R1's own argument one step over. `superseded_by` scans
            # FORWARD only, and rightly: the direction is the module's rule
            # and its control asserts it. So a claim pointing at itself or
            # backwards can never take effect, and saying so is the only
            # way the author finds out.
            if own_named == ratification_ref:
                # SITE: check#17
                raise RatificationRefused(
                    f"REFUSED: {ratification_ref}'s own block declares "
                    f"`supersedes: {own_named}` -- ITSELF. A ratification "
                    f"cannot supersede itself: the entry would have to "
                    f"postdate itself for the supersession to be read, and "
                    f"it verified clean while superseding nothing.")
            # `own_named in own_idx` is not redundant with `check#16`
            # above: written without it, neutralising THAT guard turns THIS
            # line into a KeyError, and a crash is neither a refusal nor a
            # named failure. Each guard stands on its own input.
            if (own_named is not None and own_named in own_idx
                    and own_idx[own_named]["line"]
                    > own_idx[ratification_ref]["line"]):
                # SITE: check#18
                raise RatificationRefused(
                    f"REFUSED: {ratification_ref} (0-based register line "
                    f"{own_idx[ratification_ref]['line']}) declares "
                    f"`supersedes: {own_named}`, which stands LATER in the "
                    f"register (0-based line "
                    f"{own_idx[own_named]['line']}). "
                    f"Supersession is read forward only, so this one never "
                    f"takes effect: the entry it names keeps verifying for "
                    f"new runs and this one verifies too. It is a "
                    f"supersession written on the wrong entry (DE20-R2).")
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
            # SITE: check#9
            raise RatificationRefused(
                f"REFUSED: {ratification_ref} field {_f!r} carries the VALUE "
                f"{fields[_f]!r}, which is not in the adopted vocabulary "
                f"{list(_allowed)}. This is a WRONG VALUE, not a missing "
                f"field and not an undecidable one -- round 10 made absence "
                f"refuse and left nonsense verifying clean (DE-R3).")
    if fields.get("kind") != "R-ADMISS":
        # SITE: check#10
        raise RatificationRefused(
            f"REFUSED: {ratification_ref} does not declare itself an R-ADMISS "
            f"ratification (bound kind: {fields.get('kind')!r}). An entry can "
            f"be real, recent and about something else entirely.")
    pop = supply_population(supplied)
    if not pop["counts_sum_matches"]:
        # SITE: check#11
        raise RatificationRefused(
            f"REFUSED: the supply's n_supplied_total "
            f"({pop['n_supplied_total']}) is not the sum of its per-coin "
            f"(n_present - n_masked_applied) ({pop['sum_present_minus_masked']})"
            f" -- whatever population the entry names, this supply does not "
            f"describe itself consistently")
    named = fields.get("population")
    if named not in KNOWN_POPULATIONS:
        # SITE: check#12
        raise RatificationRefused(
            f"REFUSED: {ratification_ref} names population {named!r}, which "
            f"this checker cannot evaluate (known: {KNOWN_POPULATIONS}). "
            f"Reported as unknown rather than assumed to be the full one.")
    if named != POP_FULL:
        # SITE: check#13
        raise RatificationRefused(
            f"REFUSED: {ratification_ref} ratifies a {named} population while "
            f"this supply is the full complement "
            f"({pop['n_supplied_total']} windows, no selection field). A "
            f"ratification for a sampled population does not cover a full one.")
    if pop["selection_fields_present"]:
        # SITE: check#14
        raise RatificationRefused(
            f"REFUSED: {ratification_ref} ratifies the FULL complement "
            f"but the supply carries selection field(s) "
            f"{pop['selection_fields_present']}")
    # A FULL ratification whose own sampling field is not NONE contradicts
    # itself. It REFUSES rather than lowering `verified`: a self-contradictory
    # ratification is not a weaker one.
    if fields.get("sampling") != "NONE":
        # SITE: check#15
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
        "stamp_fields": "`stamped_at` is the CANONICAL PARSE of the supplied "
                        "receipt stamp; `stamped_at_raw` is that value "
                        "exactly as supplied; both are None when no receipt "
                        "was supplied",
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
        # DE22-R1: a duplication that could not reach an answer is REPORTED
        # rather than swallowed -- with the line numbers, so a reader can
        # see both headings, and with which occurrence the index kept.
        "duplicate_refs": dict(idx.duplicate_refs),
        "duplicate_refs_kept": idx.kept,
        "decides": "nothing -- this reports; admission is the coordinator's "
                   "act and accrual is decided elsewhere (R-418)",
    }


# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 183


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
    # The module's anchor claim, and the one an added rule is most likely
    # to break: a refusal here is CAUGHT and reported by name rather than
    # ending the run in a traceback. (A rule with the comparison the wrong
    # way round refuses R-419 itself, because its block supersedes the
    # EARLIER R-418 -- which is the shape this catch was written for.)
    _saw419 = ""
    try:
        res = check(sup, "R-419")
    except RatificationRefused as _exc:
        res = {"verified": False, "binding_source": None, "checks": {},
               "bound_fields": {}, "unverifiable": ["<REFUSED>"],
               "superseded_by": []}
        _saw419 = f" REFUSED INSTEAD: {str(_exc)[:110]}"
    ok(res["verified"] and res["binding_source"] == "BLOCK" and not _saw419,
       f"R-419 VERIFIES against the real 09-01 supply, bound from its "
       f"adopted BLOCK: { {k: v for k, v in res['checks'].items()} }"
       f"{_saw419}")
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

    # ---- DE12-R2: an empty value is ABSENCE IN PLACE --------------------
    refuses(lambda: check(sup, "R-900", fixture_register(scope_to="")),
            "DE12-R2 CLOSED: `scope_to:` with NOTHING after the colon "
            "REFUSES -- it used to read as OPEN-ENDED with verified True and "
            "unverifiable [], while the SAME absence written as a missing "
            "line refused. Two shapes of one absence, treated oppositely",
            needle="EMPTY value(s)")
    ok(check(sup, "R-900", fixture_register(scope_to="null"))
       ["checks"]["day_in_scope"] is True,
       "and `null` is still the declared open spelling")
    ok(check(sup, "R-900", fixture_register(scope_to="20260930"))
       ["checks"]["day_in_scope"] is True,
       "a bounded scope still bounds (09-01 <= 09-30)")
    refuses(lambda: check(sup, "R-900", fixture_register(scope_to="~")),
            "and `~` still refuses as a VALUE, not as an absence",
            needle="not a day")
    refuses(lambda: check(sup, "R-900",
                          fixture_register().replace("scope_to: null\n", "")),
            "while the ABSENT line still refuses as MISSING -- the three "
            "cases now have three distinct messages",
            needle="MISSING")
    refuses(lambda: check(sup, "R-900", fixture_register(scope_to="none")),
            "SCOPE_OPEN_TOKENS is `null` ALONE: `none` was an undeclared "
            "SYNONYM and is gone. R-419 section 4 adopted `null` and nothing "
            "else, and adding a second spelling to a coordinator-adopted "
            "format is not mine to do -- it belongs in the block spec first",
            needle="not a day")
    ok("the prior ref or `null`; SINGULAR" in PROPOSED_BLOCK
       and validate_supersedes(SUPERSEDES_NULL, "the documented spelling")
       is None,
       "and the documented block SAYS the shape the validator enforces -- "
       "the doc line and the code are checked against each other, because "
       "a documentation claim that nothing executes is the shape Q-DE-33 "
       "corrected in band")
    ok(SCOPE_OPEN_TOKENS == ("null",),
       f"the open-token set is exactly {SCOPE_OPEN_TOKENS}")
    # ---- DE14-R3: the constant and the comparison must say the SAME
    # thing.  `.lower()` on the value admitted three spellings the declared
    # set does not contain, so the tuple read as one token and behaved as
    # four. DECIDED as a RESTORATION: exact `null`, matching R-419 section 4.
    for _case in ("NULL", "Null", "nUlL"):
        refuses(lambda v=_case: check(sup, "R-900",
                                      fixture_register(scope_to=v)),
                f"KNOWN-BAD: scope_to {_case!r} REFUSES as a VALUE -- "
                f"`.lower()` made it OPEN-ENDED and `verified True`, while "
                f"the constant beside it declared one spelling. Each of "
                f"these is its own control: none is a typo for the other",
                needle="not a day")
    ok(check(sup, "R-900",
             fixture_register(scope_to="null"))["checks"]["day_in_scope"]
       is True,
       "POSITIVE CONTROL: the declared spelling `null` is still OPEN, so "
       "the restoration removed the undeclared spellings and nothing else")

    # ---- the EMPTY message named a field the format does not define -----
    _und = fixture_register().replace("supersedes: null\n",
                                      "supersedes: null\nnotes: \n")
    refuses(lambda: check(sup, "R-900", _und),
            "an EMPTY value on a field the format does not DEFINE refuses "
            "and says so: the message iterates the block's own rows, so an "
            "undefined `notes:` used to be listed beside real fields as "
            "though R-419 section 4 had a `notes` field",
            needle="are NOT ratification fields at all")
    refuses(lambda: check(sup, "R-900", fixture_register(revocable_by="")),
            "and the empty-value refusal is GENERAL, not a scope_to special "
            "case: any block field that is present and says nothing refuses",
            needle="EMPTY value(s)")

    # ---- CO-7: the CO-6 fix ships its falsifier -------------------------
    # Round 13 changed behaviour and added NO check: 84 -> 84. The refusal
    # worked and nothing asserted it, which is rule 15 unmet in my own batch.
    for _bad in ("not-a-time", "", "2026-13-45T99:99Z"):
        refuses(lambda v=_bad: check(sup, "R-419", stamped_at=v),
                f"CO-7: a garbage `stamped_at` ({_bad!r}) REFUSES on R-419, "
                f"which is NOT superseded -- the branch the parse used to "
                f"skip entirely",
                needle="stamped_at")
    for _bad in (123, 20260902, ["x"]):
        refuses(lambda v=_bad: check(sup, "R-419", stamped_at=v),
                f"and a NON-STRING stamp ({_bad!r}) refuses there too",
                needle="stamped_at")
    # DE13-R2: THE CHECK THAT WOULD HAVE CAUGHT A FALSE FILING.
    # Round 14 reported `stamped_at_raw` as "documented where check()
    # describes its emission". It was not: the patch's anchor did not match,
    # `str.replace` silently did nothing, and nothing asserted the result. A
    # docstring assertion is cheap and it is the thing that fails when a
    # claim about documentation stops being true.
    # DE15-R4: TOKEN PRESENCE IS NOT A BINDING. The first version asserted
    # `"stamped_at_raw" in doc` and `"CANONICAL PARSE" in doc` separately,
    # so a docstring that SWAPS the two meanings -- "`stamped_at_raw` is the
    # CANONICAL PARSE and `stamped_at` is the value exactly as supplied" --
    # kept both tokens and passed (the reviewer ran it: OK, 104 checks).
    # This module's own rule about the register is the rule here: vocabulary
    # hits are not references (rule 16). So the assertion is the PHRASE that
    # binds each field to its meaning, as one string.
    _doc = check.__doc__ or ""
    ok("`stamped_at` is the CANONICAL PARSE" in _doc
       and "`stamped_at_raw` is the value exactly as supplied" in _doc,
       "the emission's stamp fields are DOCUMENTED in check()'s docstring "
       "as BINDING PHRASES -- each field named together with what it is -- "
       "so a docstring that keeps both tokens while reversing their "
       "meanings goes red (DE15-R4)")
    _sf = check(sup, "R-419").get("stamp_fields", "")
    ok("`stamped_at` is the CANONICAL PARSE" in _sf
       and "`stamped_at_raw` is that value exactly as supplied" in _sf,
       "and the EMISSION carries the same BINDING, in the refusal_scope/"
       "decides idiom, so a reader of the artifact alone is told which "
       "field is which and not merely that both exist")

    _st = check(sup, "R-419", stamped_at="2026-09-02T10:30Z")
    ok(_st["verified"] and _st["stamped_at"] == "2026-09-02T10:30:00Z"
       and _st["stamped_at_raw"] == "2026-09-02T10:30Z",
       f"POSITIVE CONTROL: a WELL-FORMED stamp on a non-superseded ref "
       f"verifies, and the emission echoes BOTH -- the PARSED value "
       f"({_st['stamped_at']}) and the raw one ({_st['stamped_at_raw']}), so "
       f"a receipt can be matched against its own field while a reader still "
       f"sees what was handed in")
    ok(check(sup, "R-419")["stamped_at"] is None
       and check(sup, "R-419")["stamped_at_raw"] is None,
       "and with no receipt supplied both read None -- `no receipt` is not "
       "an unparsable one")

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

    # ---- DE14-R1: `supersedes` was matched by RAW EQUALITY and validated
    # nowhere.  The field that decides whether an entry may start a run was
    # the one field nothing checked, and the LATER entry's copy -- the one
    # that actually does the deciding -- was never even read as a value.
    def _chain(sup_val, ref="R-903"):
        """R-902 at 09:00Z, then a later entry whose block carries
        `sup_val` as its `supersedes`."""
        blocks = []
        for r, ts, sv in (("R-902", "09:00Z", "null"), (ref, "10:00Z",
                                                        sup_val)):
            rows = ["ref: " + r, "kind: R-ADMISS", f"population: {POP_FULL}",
                    "sampling: NONE",
                    "present_source: data/pm_5min/markets.jsonl",
                    "scope_days: FORWARD_RACE_DAYS", "scope_from: 20260901",
                    "scope_to: null", "revocable_by: USER"]
            if sv is not None:
                rows.append(f"supersedes: {sv}")
            blocks.append(f"### {r} — 2026-09-02T{ts} — coordinator: "
                          f"R-ADMISS\n\n```ratification\n"
                          + "\n".join(rows) + "\n```\n\n")
        return "".join(blocks) + "## next\n"

    _before = "2026-09-02T09:30:00Z"        # predates the superseder
    ok(superseded_by(_chain("R-902"), "R-902") == ["R-903"],
       "POSITIVE CONTROL FIRST: the EXACT row `supersedes: R-902` is still "
       "found, so the shape rule below is a filter on a working matcher "
       "and not a wall in front of a broken one")
    ok(superseded_by(_chain("null"), "R-902") == [],
       "and a later entry declaring `null` is READ, ADMITTED and does not "
       "supersede -- `null` is a value here, not a hole")
    for _bad, _why, _needle in (
            ("", "an EMPTY value -- absence in place, the exact shape "
                 "DE12-R2 closed one field over", "EMPTY"),
            ("   ", "WHITESPACE, which is the empty case wearing a "
                    "different spelling", "EMPTY"),
            ("r-902", "the right ref in the wrong CASE", "neither"),
            ("R-9O2", "a LETTER O for a ZERO -- indistinguishable by eye "
                      "and never equal", "neither"),
            ("R-902 (partial)", "a ref with a PARENTHETICAL, which is how "
                                "a coordinator would naturally qualify one",
             "neither"),
            ("R-902, R-901", "TWO refs, which R-419 section 4 does not "
                             "define -- singular stays singular, and a "
                             "plural spelling is a SPEC CHANGE that is the "
                             "coordinator's to declare, not mine to invent",
             "MORE THAN ONE")):
        refuses(lambda v=_bad: check(sup, "R-902", _chain(v),
                                     stamped_at=_before),
                f"KNOWN-BAD ON THE LATER ENTRY: `supersedes: {_bad!r}` "
                f"REFUSES BY NAME -- {_why}. Every one of these left "
                f"R-902 reading `verified_for_new_run: True` while a "
                f"supersession sat one entry away, because raw equality "
                f"simply fails to match and says nothing (DE14-R1)",
                needle=_needle)
    refuses(lambda: check(sup, "R-902", _chain(None), stamped_at=_before),
            "and an ABSENT `supersedes` on a later entry REFUSES too: the "
            "adopted block requires the field, and an absent one cannot be "
            "told from a supersession nobody wrote",
            needle="carries no `supersedes`")
    # DE18-R3: `parse_day#1` was reached by NOTHING -- neutralising it left
    # the suite green at 150, alone among the six markers the audit does
    # not drive. It is unreachable through `check()` (block and prose
    # values are always strings), so the choice was to drive it or to
    # annotate it unreachable. DRIVEN, and the reason is the difference
    # from `de_admissible_windows`'s C-extension entry: THAT one cannot be
    # addressed by any in-process assertion, while this guard defends an
    # EXPORTED function's contract against a direct caller -- and a direct
    # call is exactly how the module already drives `validate_supersedes`'s
    # non-string guard on the line below. An unreachable annotation would
    # have declared a limit the module does not have (DE15's own lesson).
    refuses(lambda: parse_day(20260901, "scope_from"),
            "KNOWN-BAD, DIRECT CALL: `parse_day` refuses a NON-STRING day "
            "naming the field and the value -- reachable only by a direct "
            "caller, driven here because the guard defends the function's "
            "contract rather than `check()`'s path (DE18-R3)",
            needle="scope_from")
    ok(parse_day("20260901", "scope_from").strftime("%Y%m%d") == "20260901",
       "POSITIVE CONTROL on the same call: a well-formed day parses, so "
       "the guard above is a filter and not a wall")
    refuses(lambda: validate_supersedes(902, "a fixture"),
            "a NON-STRING refuses naming its type -- the block is read "
            "from text today, but the validator is the contract and a "
            "future JSON reader is where an int would arrive",
            needle="not a string")
    ok(validate_supersedes("null", "x") is None
       and validate_supersedes("R-902", "x") == "R-902",
       "and in the admitting direction it RETURNS the ref it names, `null` "
       "as None -- so the caller matches on a validated value")
    for _u, _lab in (("WHATEVER", "free text"),
                     ("/etc/passwd", "a path"),
                     ("R-902 (partial)", "a qualified ref")):
        refuses(lambda v=_u: check(sup, "R-900",
                                   fixture_register(supersedes=v)),
                f"KNOWN-BAD ON THE ENTRY UNDER CHECK: `supersedes: {_u!r}` "
                f"({_lab}) REFUSES -- its own field was unvalidated too, "
                f"and all three verified clean")
    # DE18-R1: THIS FIXTURE MOVED, and the move is the finding. It used to
    # read "a WELL-SHAPED ref still verifies" -- and it did, because
    # `R-418` is not in the fixture register at all. The same string one
    # entry over had refused since round 18.
    refuses(lambda: check(sup, "R-900", fixture_register(supersedes="R-418")),
            "KNOWN-BAD (the fixture that used to be a positive control): "
            "`supersedes: R-418` in the entry's OWN block, against a "
            "register that holds no R-418, now REFUSES -- it verified True, "
            "[] while the identical string in a later entry refused, and "
            "the deferral that permitted it ('someone will check the "
            "target') rests on a check nothing performs (DE18-R1)",
            needle="exists in this register")
    _pair = (fixture_register("R-899").replace("\n\n## next\n", "\n\n")
             + fixture_register("R-900", supersedes="R-899"))
    _psup = check(sup, "R-900", _pair)
    ok(_psup["verified"] and _psup["unverifiable"] == [],
       f"POSITIVE CONTROL, REBUILT SO IT CAN FAIL: a well-shaped ref whose "
       f"target EXISTS in the register still verifies "
       f"({_psup['verified']}, {_psup['unverifiable']}) -- the rule is "
       f"shape AND existence, and it does not quietly require `null`")
    # ---- DE20-R2: existence is NECESSARY, NOT SUFFICIENT ---------------
    refuses(lambda: check(sup, "R-902",
                          fixture_register("R-902", supersedes="R-902")),
            "KNOWN-BAD: a block declaring `supersedes: <ITSELF>` REFUSES -- "
            "it verified True with superseded_by [], superseding nothing: "
            "the entry would have to postdate itself for the supersession "
            "to be read at all (DE20-R2)",
            needle="ITSELF")
    _backwards = (fixture_register("R-902", supersedes="R-903")
                  .replace("\n\n## next\n", "\n\n")
                  + fixture_register("R-903"))
    refuses(lambda: check(sup, "R-902", _backwards),
            "KNOWN-BAD: an EARLIER entry declaring it supersedes a LATER "
            "one REFUSES -- `superseded_by` scans forward only, so the "
            "claim never takes effect: both entries verified and neither "
            "was superseded. A supersession written on the wrong entry",
            needle="stands LATER in the register")
    # The STAMPED chain, because `fixture_register` headings carry no
    # register timestamp and a correctly-directed supersession then dies at
    # `check#2` before the new-run refusal can be reached -- which would
    # make this control pass on the wrong refusal.
    _fwd = _chain("R-902")
    # A REFUSAL on the correctly-directed entry is the defect this control
    # exists for, so it is CAUGHT and reported by name: uncaught, a flipped
    # comparison would end the run in a traceback instead.
    _f903, _saw903 = None, ""
    try:
        _f903 = check(sup, "R-903", _fwd)
    except RatificationRefused as _exc:
        _f903 = {"verified": False, "superseded_by": []}
        _saw903 = f" R-903 REFUSED INSTEAD: {str(_exc)[:100]}"
    _f902 = None
    try:
        check(sup, "R-902", _fwd)
    except RatificationRefused as _exc:
        _f902 = str(_exc)
    ok(_f903["verified"] and _f903["superseded_by"] == [] and not _saw903
       and _f902 is not None and "SUPERSEDED by R-903" in _f902,
       f"POSITIVE CONTROL ON THE DIRECTION: written the RIGHT way round -- "
       f"the LATER entry superseding the earlier -- R-903 verifies "
       f"({_f903['verified']}, superseded_by {_f903['superseded_by']}) and "
       f"R-902 then refuses FOR A NEW RUN: {str(_f902)[:64]!r}... So the "
       f"two refusals above are about DIRECTION and not about "
       f"supersession.{_saw903}")

    # ---- DE22-R1: a duplicated ref, refused where it can reach an
    # answer and REPORTED where it cannot ------------------------------
    def _dup(ref_lines, extra=""):
        """A register with `ref_lines` = [(ref, supersedes|None), ...] in
        file order, headings stamped so nothing dies at `check#2` first."""
        out, hh = [], 9
        for r, sv in ref_lines:
            rows = ["ref: " + r, "kind: R-ADMISS", f"population: {POP_FULL}",
                    "sampling: NONE",
                    "present_source: data/pm_5min/markets.jsonl",
                    "scope_days: FORWARD_RACE_DAYS", "scope_from: 20260901",
                    "scope_to: null", "revocable_by: USER",
                    f"supersedes: {sv if sv else 'null'}"]
            body = ("\n```ratification\n" + "\n".join(rows) + "\n```\n"
                    if sv != "NO_BLOCK" else "\nprose only, no block\n")
            out.append(f"### {r} — 2026-09-02T{hh:02d}:00Z — coordinator: "
                       f"R-ADMISS\n{body}\n")
            hh += 1
        return "".join(out) + extra + "## next\n"

    # THE REVIEWER'S FIXTURE: R-902 twice with R-903 declaring it superseded
    # in between. Before this round: superseded_by(R-902) == [] (the real
    # supersession LOST), check(R-902) VERIFIED, and check(R-903) REFUSED
    # at check#18 quoting a line its author never wrote -- both ends wrong
    # at once, from one dict lookup.
    _rev = _dup([("R-902", "NO_BLOCK"), ("R-903", "R-902"),
                 ("R-902", None)])
    _seen = {}
    for _who in ("R-902", "R-903"):
        try:
            _r = check(sup, _who, _rev)
            _seen[_who] = f"VERIFIED {_r['verified']}"
        except RatificationRefused as _exc:
            _seen[_who] = str(_exc)
    # The lines are COMPUTED from the fixture, never pinned: the geometry
    # is the parser's, and a literal here would be a number to update.
    _revlines = [e["line"] for e in all_entries(_rev) if e["ref"] == "R-902"]
    ok(len(_revlines) == 2
       and all(str(_revlines) in v for v in _seen.values()),
       f"DE22-R1, THE REVIEWER'S FIXTURE: a ref heading TWO entries with a "
       f"supersession pointing at it now REFUSES at BOTH ends, naming the "
       f"ref and both 0-based lines {_revlines} -- R-902 refuses as the "
       f"SUBJECT, R-903 because a `supersedes:` names it: "
       f"{_seen['R-902'][:76]!r}. "
       f"Before, the dict answered LAST-WINS: the supersession was lost, "
       f"R-902 verified for a new run, and R-903 was refused at `check#18` "
       f"quoting the LAST line ({_revlines[-1]}), which nobody wrote")
    refuses(lambda: check(sup, "R-902", _dup([("R-902", "NO_BLOCK"),
                                              ("R-902", "NO_BLOCK")])),
            "KNOWN-BAD: the SUBJECT of the check duplicated REFUSES -- "
            "which of the two is being asked about cannot be decided by a "
            "reader of the file, so it is not decided here either",
            needle="is the SUBJECT of this check")
    refuses(lambda: check(sup, "R-903",
                          _dup([("R-902", None), ("R-902", None),
                                ("R-903", None)])),
            "KNOWN-BAD: two duplicated entries each CARRYING a ratification "
            "block REFUSE even when neither is the subject -- the "
            "heading-level form of `own_ratification_blocks#1`, where a "
            "corrected ratification is shadowed by the one it corrects",
            needle="carries an OWN ratification block")
    refuses(lambda: check(sup, "R-903",
                          _dup([("R-902", "NO_BLOCK"), ("R-902", "NO_BLOCK"),
                                ("R-903", "R-902")])),
            "KNOWN-BAD: a duplicated ref NAMED BY A `supersedes:` refuses "
            "too, with neither occurrence carrying a block and neither "
            "being the subject -- where the target stands is what decides "
            "whether the supersession is read at all",
            needle="named by a `supersedes:`")

    # ---- DE24-R1: (ii) reads OWN blocks, so a QUOTATION cannot refuse
    # another entry's check ---------------------------------------------
    _quote = (
        "\n### R-99901 — 2026-09-02T16:00Z — coordinator: showing a "
        "spelling\n\n```ratification\nref: R-903\nkind: R-ADMISS\n"
        f"population: {POP_FULL}\nsampling: NONE\n"
        "present_source: data/pm_5min/markets.jsonl\n"
        "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
        "scope_to: null\nrevocable_by: USER\nsupersedes: {dup}\n```\n\n"
        "Nothing here ratifies anything.\n\n## next\n")
    _rtxt0 = REGISTER.read_text()
    _dupref = next(iter(entry_index(_rtxt0).duplicate_refs))
    _quoted = _rtxt0 + _quote.replace("{dup}", _dupref)
    _saw_q = ""
    try:
        _qres = check(sup, "R-419", _quoted)
    except RatificationRefused as _exc:
        _qres, _saw_q = None, f" REFUSED INSTEAD: {str(_exc)[:110]}"
    ok(_qres is not None and _qres["verified"]
       and _qres["unverifiable"] == []
       and _dupref in _qres["duplicate_refs"] and not _saw_q,
       f"POSITIVE CONTROL (DE24-R1): a later, NON-RATIFYING entry QUOTING "
       f"a block whose `supersedes:` names the duplicated {_dupref} does "
       f"NOT refuse R-419's check -- it verifies "
       f"{_qres['verified'] if _qres else False}, "
       f"{_qres['unverifiable'] if _qres else '<refused>'}, and the "
       f"duplication still STANDS AS A REPORT "
       f"({_qres['duplicate_refs'] if _qres else {}}). A quotation "
       f"supersedes nothing, so it cannot make a duplication reach an "
       f"answer; refusing on it was DE16-R1 returning narrow -- a "
       f"quotation deciding another entry's check{_saw_q}")
    _own_names = _dup([("R-902", "NO_BLOCK"), ("R-902", "NO_BLOCK"),
                       ("R-903", "R-902")])
    _dl = [e["line"] for e in all_entries(_own_names) if e["ref"] == "R-902"]
    _msg2 = ""
    try:
        check(sup, "R-903", _own_names)
    except RatificationRefused as _exc:
        _msg2 = str(_exc)
    ok("R-902" in _msg2 and str(_dl) in _msg2 and "0-based lines" in _msg2,
       f"KNOWN-BAD (DE24-R1's other half): an entry's OWN block naming a "
       f"duplicated ref STILL refuses at `entry_index#2`, naming the ref "
       f"and both 0-based lines {_dl} -- every reachable case still "
       f"refuses, which is what the refinement had to preserve: "
       f"{_msg2[:76]!r}")

    # ---- CO-9: the scans ANSWER, they do not adjudicate ---------------
    # C / C2 / C3 stand on either side of the subject, which is the
    # fixture my round-26 ordering note lacked: it was true of ORDER and
    # silent on the SET, and a suite with no earlier-malformed entry could
    # not have caught that.
    def _malformed(ref):
        """An entry carrying TWO blocks of its OWN -- `#1`'s known-bad."""
        blk = ("```ratification\nref: " + ref + "\nkind: R-ADMISS\n"
               f"population: {POP_FULL}\nsampling: NONE\n"
               "present_source: data/pm_5min/markets.jsonl\n"
               "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
               "scope_to: null\nrevocable_by: USER\nsupersedes: null\n"
               "```\n")
        return (f"\n### {ref} — 2026-09-02T16:30Z — coordinator: R-ADMISS\n\n"
                + blk + "\nand again, malformed:\n\n" + blk + "\n")

    _r419_line = entry_index(_rtxt0)["R-419"]["line"]
    _lines0 = _rtxt0.split("\n")
    _before = ("\n".join(_lines0[:_r419_line]) + _malformed("R-99900")
               + "\n".join(_lines0[_r419_line:]))
    _saw_c = ""
    try:
        _c = check(sup, "R-419", _before)
    except RatificationRefused as _exc:
        _c, _saw_c = None, f" REFUSED INSTEAD: {str(_exc)[:110]}"
    ok(_c is not None and _c["verified"] and not _saw_c,
       f"POSITIVE CONTROL C (CO-9): a MALFORMED entry standing EARLIER "
       f"than the subject -- two own blocks under one heading -- no longer "
       f"refuses the subject's check: R-419 verifies "
       f"{_c['verified'] if _c else False}. An entry before the subject "
       f"can supersede nothing, so it reaches no answer, and the scan "
       f"ANSWERS ownership rather than adjudicating it{_saw_c}")
    # The SECOND occurrence is renamed, by its parsed line rather than by
    # a text replace -- and to a ref the register cannot hold, since
    # appending a digit to a real ref lands on another real one.
    _bl = _before.split("\n")
    _dup_lines = [e["line"] for e in all_entries(_before)
                  if e["ref"] == _dupref]
    _bl[max(_dup_lines)] = _bl[max(_dup_lines)].replace(
        f"### {_dupref} ", "### R-990006 ", 1)
    _no_dup = "\n".join(_bl)
    _saw_c2 = ""
    try:
        _c2 = check(sup, "R-419", _no_dup)
    except RatificationRefused as _exc:
        _c2, _saw_c2 = None, f" REFUSED INSTEAD: {str(_exc)[:110]}"
    ok(_c2 is not None and _c2["verified"]
       and _c2["duplicate_refs"] == {} and not _saw_c2,
       f"POSITIVE CONTROL C2: the SAME register with the duplicate renamed "
       f"away verifies too ({_c2['duplicate_refs'] if _c2 else None} "
       f"duplicates) -- and that pair is the point of the finding: before "
       f"this round C refused and C2 returned, so whether one entry "
       f"refused another depended on {_dupref}, a duplicate with nothing "
       f"to do with either{_saw_c2}")
    _after = _rtxt0 + _malformed("R-99900")
    refuses(lambda: check(sup, "R-419", _after),
            "KNOWN-BAD C3: the same malformed entry standing LATER than "
            "the subject STILL refuses -- it is on the path, read by "
            "`superseded_by` as a possible superseder, and round 18 made "
            "that adjudication the answer rather than an obstacle to one",
            needle="ratification blocks of its OWN")

    # ---- R-451 §3: (iii) reads OWN blocks -----------------------------
    def _dupe_pair(second_block_ref, sup_val="null", kind="R-ADMISS"):
        """A ref heading two entries; the second carries a block whose
        `ref` is `second_block_ref` -- own when it equals the heading."""
        head = ("### R-99903 — 2026-09-02T16:40Z — coordinator: R-ADMISS\n"
                "\nprose only, no block\n\n")
        blk = (f"### R-99903 — 2026-09-02T16:45Z — coordinator: R-ADMISS\n\n"
               f"```ratification\nref: {second_block_ref}\n"
               f"kind: {kind}\npopulation: {POP_FULL}\nsampling: NONE\n"
               "present_source: data/pm_5min/markets.jsonl\n"
               "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
               f"scope_to: null\nrevocable_by: USER\n"
               f"supersedes: {sup_val}\n```\n\n")
        return _rtxt0 + "\n" + head + blk + "## next\n"

    _saw_d = ""
    try:
        _d = check(sup, "R-419", _dupe_pair("R-777"))
    except RatificationRefused as _exc:
        _d, _saw_d = None, f" REFUSED INSTEAD: {str(_exc)[:110]}"
    ok(_d is not None and _d["verified"]
       and _d["duplicate_refs"].get("R-99903") is not None and not _saw_d,
       f"POSITIVE CONTROL D (R-451 §3): a duplicated heading whose second "
       f"occurrence carries a QUOTED block (`ref: R-777`) is REPORTED, not "
       f"refused -- {_d['duplicate_refs'] if _d else {}} -- because "
       f"`superseded_by` reads the kept occurrence's OWN blocks and a "
       f"quotation is not own whichever occurrence were kept. It reached "
       f"no answer, so it belongs on the reporting side{_saw_d}")
    # D2 is the control for the filter's OTHER conjunct: a block whose ref
    # IS the heading but whose kind is not R-ADMISS is not a ratification,
    # so it is not own -- the same pair `check#8` and `check#10` apply to
    # the subject. Without this, dropping `kind` from the quiet filter
    # passes: D turns on the REF conjunct and nothing turned on kind.
    _saw_d2 = ""
    try:
        _d2 = check(sup, "R-419", _dupe_pair("R-99903", kind="STATUS_NOTE"))
    except RatificationRefused as _exc:
        _d2, _saw_d2 = None, f" REFUSED INSTEAD: {str(_exc)[:110]}"
    ok(_d2 is not None and _d2["verified"]
       and _d2["duplicate_refs"].get("R-99903") is not None and not _saw_d2,
       f"POSITIVE CONTROL D2: and a block under its OWN heading that does "
       f"NOT declare itself R-ADMISS is not a ratification either, so the "
       f"duplication is REPORTED "
       f"({_d2['duplicate_refs'] if _d2 else {}}) -- ownership is ref AND "
       f"kind in the quiet filter exactly as in the adjudicating "
       f"reader{_saw_d2}")
    refuses(lambda: check(sup, "R-419", _dupe_pair("R-99903", "R-419")),
            "KNOWN-BAD E: the same shape with an OWN block on the second "
            "occurrence -- ref == the heading, carrying `supersedes: "
            "R-419` -- still REFUSES at `entry_index#3`: keeping the first "
            "occurrence would drop that supersession, which is an answer "
            "reached",
            needle="carries an OWN ratification block")

    # ---- DE27-R1: ONE ownership text, and a control aimed at the
    # conjunct that had none ------------------------------------------
    # Three of the four conjunct drops died at a control written for that
    # conjunct; the ADJUDICATING reader's `kind` drop died as an UNCAUGHT
    # refusal inside a positive control ("R-419 is SUPERSEDED by R-999") --
    # red and loud, but not aimed at it, so an edit that changed which
    # fixture broke would turn a named red into a puzzling one. After the
    # refactor the drop is a SINGLE site, and this is the control on it.
    _two_kinds = {"heading": "### R-99904 — 2026-09-02T17:00Z — coordinator: "
                             "R-ADMISS",
                  "body": ("\n```ratification\nref: R-99904\n"
                           "kind: STATUS_NOTE\n"
                           "supersedes: null\n```\n\nand its real one:\n\n"
                           "```ratification\nref: R-99904\n"
                           "kind: R-ADMISS\nsupersedes: null\n```\n"),
                  "ref": "R-99904"}
    _own_two, _saw_k = None, ""
    try:
        _own_two = own_ratification_blocks(_two_kinds)
    except RatificationRefused as _exc:
        _saw_k = f" REFUSED INSTEAD: {str(_exc)[:110]}"
    ok(_own_two is not None and len(_own_two) == 1
       and _own_two[0].get("kind") == "R-ADMISS" and not _saw_k,
       f"NAMED CONTROL for the ownership filter's `kind` conjunct: an "
       f"entry (`R-99904`) carrying TWO fenced blocks under its own ref, "
       f"one `kind: STATUS_NOTE` and one `kind: R-ADMISS`, has exactly ONE "
       f"own block ({len(_own_two) if _own_two is not None else '<refused>'}"
       f") -- so the adjudicating reader does NOT refuse it. Drop the "
       f"`kind` conjunct and both count as own, and the message a "
       f"maintainer meets is `own_ratification_blocks#1`: 'R-99904 carries "
       f"2 ratification blocks of its OWN'. That drop used to surface as "
       f"an uncaught refusal inside another control (DE27-R1){_saw_k}")

    # ---- and the filter is written in ONE place, asserted from the AST --
    # CO-11: the first version of this census keyed on the VARIABLE NAME
    # (`blk.get("kind", ...)`), so the same filter pasted back with the
    # loop variable RENAMED -- semantically the second text DE27-R1
    # removed -- left it GREEN, and its known-bad exercised exactly the
    # idiom the census keyed on, so the falsifier could not fail by any
    # other spelling. The message claimed the predicate; the check
    # asserted the idiom. It is now keyed on the CONSTANT and the SHAPE:
    # any `==` against "R-ADMISS" whose left side reaches a `kind` lookup,
    # by `.get("kind")` or by `["kind"]`, on ANY receiver.
    import ast as _ast_own
    _own_src = Path(__file__).read_text()

    def _ownership_sites(src: str, _ast=_ast_own) -> list[str]:
        """Functions asking a block's `kind` whether it EQUALS R-ADMISS --
        the ownership conjunct, whatever the receiver is called.

        `check#10` stays out by its operator: it asks
        `fields.get("kind") != "R-ADMISS"` of the BOUND kind of the entry
        under check, a different question about a different object.

        `selftest` and everything nested in it are excluded because the
        suite's own fixtures ASSERT this constant rather than deciding
        with it -- the named control above reads
        `_own_two[0].get("kind") == "R-ADMISS"` to show the filter kept
        the right block. A census that counted its own assertions would
        report two texts the moment it was written."""
        tree = _ast.parse(src)
        suite = [f for f in _ast.walk(tree)
                 if isinstance(f, _ast.FunctionDef) and f.name == "selftest"]
        in_suite = {id(n) for f in suite for n in _ast.walk(f)}
        out = []
        for fn in [n for n in _ast.walk(tree)
                   if isinstance(n, _ast.FunctionDef)
                   and id(n) not in in_suite]:
            for cmp_ in [n for n in _ast.walk(fn)
                         if isinstance(n, _ast.Compare)]:
                if not any(isinstance(o, _ast.Eq) for o in cmp_.ops):
                    continue
                if not any(isinstance(c, _ast.Constant)
                           and c.value == "R-ADMISS"
                           for c in cmp_.comparators):
                    continue
                reaches_kind = False
                for n in _ast.walk(cmp_.left):
                    if (isinstance(n, _ast.Call)
                            and isinstance(n.func, _ast.Attribute)
                            and n.func.attr == "get" and n.args
                            and isinstance(n.args[0], _ast.Constant)
                            and n.args[0].value == "kind"):
                        reaches_kind = True
                    if (isinstance(n, _ast.Subscript)
                            and isinstance(n.slice, _ast.Constant)
                            and n.slice.value == "kind"):
                        reaches_kind = True
                if reaches_kind:
                    out.append(fn.name)
        return sorted(set(out))

    def _paste_into_reader(src: str, filter_text: str, _ast=_ast_own):
        """Put `filter_text` where `own_ratification_blocks` asks for
        ownership, LOCATED BY THE AST rather than by a text anchor.

        CO-11's second half: the old known-bad replaced the code line by
        its text, and that same text is a STRING LITERAL in this suite --
        so with the code line gone the replace hit the literal, left the
        copy with an unterminated string, and `ast.parse` raised a
        SyntaxError: a traceback where a refusal by name belongs, and
        `_pasted != _own_src` was satisfied by mangling the suite. Returns
        None when the assignment is not exactly where it should be, and a
        copy is never built by guessing."""
        tree = _ast.parse(src)
        fns = [f for f in _ast.walk(tree)
               if isinstance(f, _ast.FunctionDef)
               and f.name == "own_ratification_blocks"]
        if len(fns) != 1:
            return None
        tgt = [n for n in fns[0].body
               if isinstance(n, _ast.Assign)
               and getattr(n.targets[0], "id", "") == "own"]
        if len(tgt) != 1:
            return None
        lines = src.split("\n")
        return "\n".join(lines[:tgt[0].lineno - 1] + filter_text.split("\n")
                          + lines[tgt[0].end_lineno:])

    _osites = _ownership_sites(_own_src)
    ok(_osites == ["own_blocks_quiet"],
       f"ONE OWNERSHIP TEXT, asserted from the AST on the CONSTANT AND THE "
       f"SHAPE: an `== \"R-ADMISS\"` whose left side reaches a `kind` "
       f"lookup -- `.get(\"kind\")` or `[\"kind\"]`, on any receiver -- "
       f"appears in {_osites} and nowhere else outside the suite. Keyed on "
       f"the VARIABLE NAME it read the idiom rather than the predicate, "
       f"and a renamed copy passed (CO-11)")
    # The same filter, three spellings, each pasted back into the
    # adjudicating reader: the idiom the census used to key on, the
    # RENAMED variable that defeated it, and a SUBSCRIPT lookup.
    _variants = {
        "the same idiom (`blk.get`)":
            '    own = [(blk, dups) for blk, dups in _fenced_blocks(entry)\n'
            "           if str(blk.get('ref', '')).strip() == ref\n"
            "           and str(blk.get('kind', '')).strip() == 'R-ADMISS']",
        "the loop variable RENAMED (`b.get`)":
            '    own = [(b, d) for b, d in _fenced_blocks(entry)\n'
            "           if str(b.get('ref', '')).strip() == ref\n"
            "           and str(b.get('kind', '')).strip() == 'R-ADMISS']",
        "a SUBSCRIPT lookup (`b[\"kind\"]`)":
            '    own = [(b, d) for b, d in _fenced_blocks(entry)\n'
            "           if str(b.get('ref', '')).strip() == ref\n"
            '           and str(b["kind"]).strip() == "R-ADMISS"]',
    }
    for _label, _text in _variants.items():
        _copy = _paste_into_reader(_own_src, _text)
        _sites = _ownership_sites(_copy) if _copy else ["<not located>"]
        ok(_copy is not None
           and _sites == ["own_blocks_quiet", "own_ratification_blocks"],
           f"KNOWN-BAD, DRIVEN THROUGH THE SAME PARSE ({_label}): the "
           f"filter pasted back into the adjudicating reader reads "
           f"{_sites} -- two texts again -- and the census goes red. The "
           f"renamed one is CO-11's mutant: it passed until this round, "
           f"because the check was keyed on what the variable was called")
    _no_reader = _own_src.replace("def own_ratification_blocks(entry: dict)",
                                  "def _reader_renamed_by_the_mutant(entry: dict)", 1)
    ok(_paste_into_reader(_no_reader, "    own = []") is None
       and _paste_into_reader(_own_src, "    own = []") is not None,
       "AND THE PASTE IS LOCATED BY THE AST, NOT BY A TEXT ANCHOR: with "
       "the reader's own assignment absent, the helper returns None rather "
       "than replacing the first matching TEXT -- which in this file is a "
       "string literal in this very suite, so the old anchor built a copy "
       "with an unterminated string and `ast.parse` raised a SyntaxError: "
       "a traceback where a refusal by name belongs (CO-11)")

    # ---- DE24-R2: `check#18` names the convention its number uses ------
    _msg18 = ""
    try:
        check(sup, "R-902", _backwards)
    except RatificationRefused as _exc:
        _msg18 = str(_exc)
    ok("0-based register line" in _msg18 and "0-based line" in _msg18
       and "stands LATER" in _msg18,
       f"DE24-R2: `check#18` now names the convention at BOTH numbers it "
       f"prints -- it was the one message of five printing a 0-based field "
       f"under the bare words 'register line' / '(line ...)', in the "
       f"message whose whole job is to send an author to two specific "
       f"places in a 20,000-line file, one line above each heading: "
       f"{_msg18[:96]!r}")

    # ---- and the real register, where it is REPORTED ------------------
    _rtxt = REGISTER.read_text()
    _ridx = entry_index(_rtxt)
    _rlines = _rtxt.split("\n")
    _expect = {r: [e["line"] for e in all_entries(_rtxt) if e["ref"] == r]
               for r in _ridx.duplicate_refs}
    ok(_ridx.duplicate_refs == _expect and len(_expect) == 1,
       f"ON THE REAL REGISTER the duplication is REPORTED, not refused: "
       f"{_ridx.duplicate_refs} -- computed from the parse and compared "
       f"with an independent recount, never a literal. It reaches no "
       f"answer here: it is nobody's subject, no `supersedes:` names it, "
       f"and no occurrence carries a block")
    _dref, _dlines = next(iter(_ridx.duplicate_refs.items()))
    ok(_ridx[_dref]["line"] == min(_dlines)
       and _rlines[_ridx[_dref]["line"]].startswith(f"### {_dref}")
       and _rlines[max(_dlines)].startswith(f"### {_dref}"),
       f"and the index keeps the FIRST occurrence by rule -- line "
       f"{_ridx[_dref]['line']} of {_dlines} -- with BOTH headings "
       f"reachable in the file at those 0-BASED indices: "
       f"{_rlines[min(_dlines)][:44]!r} and {_rlines[max(_dlines)][:44]!r}. "
       f"The convention is the parser's own and the one `check#18` prints; "
       f"an editor's 1-based gutter shows these as "
       f"{[n + 1 for n in _dlines]}")
    _rres = check(sup, "R-419")
    ok(_rres["verified"] and _rres["unverifiable"] == []
       and _rres["superseded_by"] == []
       and _rres["duplicate_refs"] == _ridx.duplicate_refs
       and "FIRST occurrence" in _rres["duplicate_refs_kept"],
       f"AND EVERY LIVE ANSWER IS UNCHANGED THROUGH IT: R-419 verifies "
       f"{_rres['verified']}, {_rres['unverifiable']}, superseded_by "
       f"{_rres['superseded_by']}, and the emission now CARRIES the "
       f"duplication as a reported fact ({_rres['duplicate_refs']}) with "
       f"the occurrence it kept -- a reader of the artifact alone is told")
    require_verified(_rres)
    ok(True,
       "and BE's consumer path is exercised on the real register: "
       "`require_verified(check(sup, 'R-419'))` returns rather than "
       "raising, so the reported duplication does not reach the gate that "
       "reads this result")

    # ---- DE20-R1: ONE implementation of "an entry exists" --------------
    import ast as _ast

    def _entry_calls(src: str) -> tuple[int, int]:
        """(calls to all_entries in the module's LOGIC, calls inside
        entry_index). The suite's own calls are excluded, deliberately:
        the rule is that the module answers "an entry exists" one way, and
        this selftest RECOUNTS independently to check the index against a
        fresh parse. Forbidding that would forbid verification by
        independent derivation -- and the mutant that matters injects a
        second derivation into `check()`, which this still catches."""
        tree = _ast.parse(src)
        suite = [f for f in _ast.walk(tree)
                 if isinstance(f, _ast.FunctionDef) and f.name == "selftest"]
        in_suite = {id(n) for f in suite for n in _ast.walk(f)}
        every = [n for n in _ast.walk(tree)
                 if isinstance(n, _ast.Call)
                 and getattr(n.func, "id", "") == "all_entries"
                 and id(n) not in in_suite]
        helper = [f for f in _ast.walk(tree)
                  if isinstance(f, _ast.FunctionDef)
                  and f.name == "entry_index"]
        inside = [n for f in helper for n in _ast.walk(f)
                  if isinstance(n, _ast.Call)
                  and getattr(n.func, "id", "") == "all_entries"]
        return len(every), len(inside)

    _src = Path(__file__).read_text()
    _every, _inside = _entry_calls(_src)
    ok(_every == 1 and _inside == 1,
       f"ONE IMPLEMENTATION OF 'AN ENTRY EXISTS': `all_entries` is called "
       f"{_every} time(s) in this module's LOGIC (the suite's own "
       f"independent recounts excluded) and {_inside} of them is inside "
       f"`entry_index` -- read from the AST, not asserted. It was derived "
       f"THREE ways (a line map, an entry map, a set comprehension) over "
       f"the same unfiltered call; they could not disagree, which is why "
       f"the finding is the DRIFT SURFACE: a filter added at one site "
       f"would leave the two ends of one rule answering differently "
       f"(DE20-R1)")
    _extra = _src.replace("    idx = entry_index(register_text, "
                          "subject=ratification_ref)",
                          "    idx = {e['ref']: e for e in "
                          "all_entries(register_text)}", 1)
    ok(_extra != _src and _entry_calls(_extra) == (2, 1),
       f"KNOWN-BAD, DRIVEN THROUGH THE SAME READER: re-deriving the index "
       f"at `check#16` in a COPY makes it {_entry_calls(_extra)} -- two "
       f"call sites, one of them outside the helper -- and this check goes "
       f"red. The predicate is over the parse, so a second derivation "
       f"cannot arrive as a comment or a string")

    # ---- DE16-R1: A QUOTED BLOCK IS NOT THE QUOTING ENTRY'S -----------
    # The register is exactly where these spellings get documented, and one
    # fence deeper the checker read the documentation as a ratification: a
    # sweep entry quoting `supersedes: R-419` made R-419 read SUPERSEDED BY
    # THE SWEEP, and a quoted malformed one made every earlier ref's check
    # REFUSE for a reason about somebody else's block.
    def _sweep(block_ref, sup_val, heading="R-999"):
        """A later, NON-ratifying entry that QUOTES a block -- the shape a
        coordinator sweep naturally takes when it reports a spelling."""
        rows = [f"ref: {block_ref}", "kind: R-ADMISS",
                f"population: {POP_FULL}", "sampling: NONE",
                "present_source: data/pm_5min/markets.jsonl",
                "scope_days: FORWARD_RACE_DAYS", "scope_from: 20260901",
                "scope_to: null", "revocable_by: USER"]
        if sup_val is not None:
            rows.append(f"supersedes: {sup_val}")
        return (f"\n### {heading} — 2026-09-02T14:00Z — coordinator: the "
                f"spelling that used to pass silently\n\n```ratification\n"
                + "\n".join(rows) + "\n```\n\nNothing here ratifies "
                "anything.\n\n## next\n")

    _real = REGISTER.read_text()
    for _sv, _was in (("R-419", "made R-419 read SUPERSEDED BY THE SWEEP"),
                      ("", "made R-419's own check REFUSE as EMPTY"),
                      ("R-902, R-901", "REFUSED as MORE THAN ONE")):
        # A REFUSAL here is the defect, so it is CAUGHT and turned into a
        # named failure: an uncaught RatificationRefused would end the run
        # in a traceback, and a crash is not a refusal -- nor a report of
        # one (this module's own standard, applied to its own suite).
        try:
            _r = check(sup, "R-419", _real + _sweep("R-903", _sv))
            _saw = ""
        except RatificationRefused as _exc:
            _r = {"verified": False, "unverifiable": ["<REFUSED>"],
                  "superseded_by": []}
            _saw = f" REFUSED INSTEAD: {str(_exc)[:110]}"
        ok(_r["verified"] and _r["unverifiable"] == []
           and _r["superseded_by"] == [] and not _saw,
           f"DE16-R1 ON THE REAL REGISTER: a later sweep entry QUOTING a "
           f"block (`ref: R-903`, `supersedes: {_sv!r}`) is NOT read as its "
           f"own ratification -- R-419 still verifies "
           f"{_r['verified']}, {_r['unverifiable']}, superseded_by "
           f"{_r['superseded_by']}. Before this it {_was}{_saw}")
    _own = superseded_by(_real + _sweep("R-999", "R-419"), "R-419")
    ok(_own == ["R-999"],
       f"POSITIVE CONTROL, SAME FIXTURE: when the later entry's block "
       f"declares ITS OWN heading ref, the supersession IS read ({_own}) "
       f"-- the rule skips QUOTATIONS, not supersessions, and without this "
       f"control the three above could pass by seeing nothing at all")
    refuses(lambda: check(sup, "R-419", _real + _sweep("R-999", "R-419")),
            f"and the end-to-end consequence follows: with {_own} really "
            f"superseding it, R-419 refuses FOR A NEW RUN, which is the "
            "verdict the quoted block was producing on no authority",
            needle="is SUPERSEDED by R-999")
    _note = _sweep("R-999", "R-419").replace("kind: R-ADMISS",
                                             "kind: STATUS_NOTE")
    ok(check(sup, "R-419", _real + _note)["superseded_by"] == [],
       "and a block under its own heading that does NOT declare itself "
       "R-ADMISS is not a ratification either -- ownership is ref AND "
       "kind, the same pair `check#8` and `check#10` already apply to the "
       "entry under check")
    _two = (_real + _sweep("R-999", "R-419")
            .replace("## next\n", "")
            + "```ratification\nref: R-999\nkind: R-ADMISS\n"
            f"population: {POP_FULL}\nsampling: NONE\n"
            "present_source: data/pm_5min/markets.jsonl\n"
            "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
            "scope_to: null\nrevocable_by: USER\nsupersedes: null\n"
            "```\n\n## next\n")
    refuses(lambda: check(sup, "R-419", _two),
            "KNOWN-BAD: an entry carrying TWO blocks of its OWN REFUSES by "
            "name -- two ratifications under one heading is a malformed "
            "entry, not a choice between them, and taking the first is how "
            "a correction would be shadowed by the block it corrects",
            needle="ratification blocks of its")

    # ---- DE18-R2: a malformed QUOTATION is not the entry's own block ---
    _quoted_bad = (
        "### R-999 — 2026-09-02T14:00Z — coordinator: showing a spelling\n\n"
        "```ratification\nref: R-903\nkind: R-ADMISS\n"
        f"population: {POP_FULL}\nsampling: NONE\n"
        "present_source: data/pm_5min/markets.jsonl\n"
        "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
        "scope_to: null\nrevocable_by: USER\n"
        "supersedes: R-902, R-901\n```\n\nand R-999's own, below:\n\n"
        "```ratification\nref: R-999\nkind: R-ADMISS\n"
        f"population: {POP_FULL}\nsampling: NONE\n"
        "present_source: data/pm_5min/markets.jsonl\n"
        "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
        "scope_to: null\nrevocable_by: USER\nsupersedes: null\n```\n\n"
        "## next\n")
    try:
        check(sup, "R-999", _quoted_bad)
        _msg = "<NO REFUSAL>"
    except RatificationRefused as _exc:
        _msg = str(_exc)
    ok("declares ref 'R-903'" in _msg and "R-999's block" not in _msg
       and "MORE THAN ONE" not in _msg,
       f"DE18-R2: an entry whose FIRST fence is a QUOTATION carrying "
       f"`supersedes: R-902, R-901` still refuses -- fail-closed is the "
       f"right answer to an entry whose first fence is somebody else's -- "
       f"but the refusal now names THE FENCE: {_msg[:96]!r}... It used to "
       f"read `R-999's block names MORE THAN ONE ref` while R-999's own "
       f"block was WELL-FORMED, sending a reader to fix a block that was "
       f"fine. The shape rule moved into the own-block branch, after "
       f"`check#8`, so the only message that can fire here is the one that "
       f"can name the mismatch")
    # The other direction: with NO quotation in front of it, the entry's
    # own malformed block is still caught AND still attributed to it. The
    # move narrows which block the shape rule reads; it does not stop it
    # reading. (A foreign first fence never reaches this rule at all now --
    # `check#8` refuses first, which is the whole closure.)
    try:
        check(sup, "R-902", fixture_register("R-902",
                                             supersedes="R-902, R-901"))
        _omsg = "<NO REFUSAL>"
    except RatificationRefused as _exc:
        _omsg = str(_exc)
    ok("MORE THAN ONE" in _omsg and "R-902's block" in _omsg,
       f"and the OWN block's malformation is still caught and still "
       f"attributed to ITS OWNER: {_omsg[:88]!r}...")

    # ---- DE16-R2: shape is not existence -------------------------------
    for _dangling, _why in (
            ("R-9021", "one digit from R-902 and perfectly well-shaped"),
            ("R-99999", "well-shaped and naming nothing at all"),
            ("R-418", "a REAL ref elsewhere, absent from THIS register")):
        refuses(lambda v=_dangling: check(sup, "R-902", _chain(v)),
                f"KNOWN-BAD: a later entry declaring `supersedes: "
                f"{_dangling}` REFUSES -- {_why}. It matched nothing and "
                f"left R-902 verifying for new runs in SILENCE, which is "
                f"DE14-R1's own sentence surviving the fix that quoted it: "
                f"a failed match says nothing (DE16-R2)",
                needle="exists in this register")
    _found = superseded_by(_chain("R-902"), "R-902")
    _sup903 = check(sup, "R-903", _chain("R-902"))
    ok(_found == ["R-903"] and _sup903["verified"],
       f"POSITIVE CONTROL: an EXISTING target still supersedes ({_found}) "
       f"and the superseder itself still verifies ({_sup903['verified']}, "
       f"{_sup903['unverifiable']}) -- the existence rule refuses dangling "
       f"refs, not supersession")

    # ---- DE16-R3: duplicate keys, no last-wins -------------------------
    _dup = fixture_register("R-902", supersedes="R-902").replace(
        "supersedes: R-902", "supersedes: R-902\nsupersedes: R-901")
    refuses(lambda: check(sup, "R-902", _dup),
            "KNOWN-BAD: TWO `supersedes:` lines in one block REFUSE -- the "
            "parse is `k, v = line.split(':', 1)` into a dict, so "
            "`R-902` then `R-901` superseded NEITHER: the first was "
            "dropped in silence and the second matched nothing, while the "
            "ratification kept verifying for new runs (fail-OPEN, DE16-R3)",
            needle="MORE THAN ONCE")
    refuses(lambda: check(sup, "R-900", fixture_register().replace(
        "scope_to: null", "scope_to: null\nscope_to: 20260901")),
            "and it is a property of the PARSE, not of `supersedes`: a "
            "repeated `scope_to` refuses the same way, which is why the "
            "guard reads keys rather than one field",
            needle="MORE THAN ONCE")
    ok(check(sup, "R-900", fixture_register())["verified"],
       "POSITIVE CONTROL: a block with each key ONCE still verifies, so "
       "the duplicate rule is a filter and not a wall")
    # The ownership guard's OTHER refusal needs a quoted-first-fence entry:
    # with two OWN blocks the count refuses first, so a duplicated key in
    # an own block is only reachable when the first fence belongs to
    # somebody else. Driven rather than left to be inferred.
    _quoted_then_own = (
        "### R-902 — 2026-09-02T09:00Z — coordinator: R-ADMISS, quoting "
        "R-903 first\n\n```ratification\nref: R-903\nkind: R-ADMISS\n"
        f"population: {POP_FULL}\nsampling: NONE\n"
        "present_source: data/pm_5min/markets.jsonl\n"
        "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
        "scope_to: null\nrevocable_by: USER\nsupersedes: null\n```\n\n"
        "and its own, below:\n\n```ratification\nref: R-902\n"
        f"kind: R-ADMISS\npopulation: {POP_FULL}\nsampling: NONE\n"
        "present_source: data/pm_5min/markets.jsonl\n"
        "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
        "scope_to: null\nscope_to: 20260901\nrevocable_by: USER\n"
        "supersedes: null\n```\n\n## next\n")
    refuses(lambda: check(sup, "R-902", _quoted_then_own),
            "KNOWN-BAD: a duplicated key in the entry's OWN block refuses "
            "even when the FIRST fence is a quotation -- the ownership "
            "guard reads the entry's own blocks, not the first one it "
            "finds, so the two refusals cover different entries rather "
            "than one of them shadowing the other",
            needle="MORE THAN ONCE")

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
       f"MUTATION AUDIT: {audit['n_cases']} CASES, live and control as "
       f"distinct calls, {audit['survivors']} survivors")
    ok(audit["n_raise_sites"] < audit["n_cases"]
       and audit["n_raise_sites"] >= 1,
       f"AND THE TWO NUMBERS ARE REPORTED SEPARATELY, computed from the "
       f"tracebacks rather than narrated: {audit['n_cases']} cases reach "
       f"{audit['n_raise_sites']} RAISE SITES. For a shared parser that is "
       f"call-site coverage -- several inputs through one refusal -- and a "
       f"reader must not read the case count as a guard count "
       f"(REVIEW_DE_ROUND_12 section 6, accepted)")
    ok(all(len(v) >= 1 for v in audit["cases_per_site"].values())
       and sum(len(v) for v in audit["cases_per_site"].values())
       == audit["n_cases"],
       f"every case is attributed to the site that refused it, and the "
       f"attribution is total: {audit['cases_per_site']}")
    # ---- DE14-R2: the coverage is now ASSERTED, against a map the
    # producer RECORDED rather than derived from the run ----------------
    _reached = audit["site_reached_by_case"]
    ok(audit["coverage_matches_expected"] and _reached == EXPECTED_SITE,
       f"COVERAGE IS ASSERTED, NOT REPORTED: every one of the "
       f"{audit['n_cases']} cases reaches the raise site EXPECTED_SITE "
       f"records for it. The old assertion compared per_guard with "
       f"`expected`, which mutation_audit built as sorted(cases) -- derived "
       f"from the very dict it was compared to, so it could not fail "
       f"(DE14-R2)")
    # ---- the marker NAMES must be unique, or two sites merge into one --
    # A site is identified by its `# SITE:` name, so two raises carrying the
    # SAME name are indistinguishable and a migration between them is
    # invisible to the coverage map -- the map would be satisfied by the
    # wrong raise. The 24 markers were unique and nothing said so; a
    # uniqueness that holds by luck is not a property (the reviewer's
    # residual on DE14-R2).
    _names = _site_names()
    ok(len(set(_names.values())) == len(_names),
       f"THE {len(_names)} `# SITE:` MARKER NAMES ARE UNIQUE, asserted -- "
       f"two raises under one name would merge into a single site key and "
       f"a case migrating between them would satisfy EXPECTED_SITE while "
       f"reaching the wrong raise")
    import tempfile as _tf
    _src = Path(__file__).read_text()
    _dupname = _src.replace("# SITE: check#3", "# SITE: check#2", 1)
    ok(_dupname != _src, "(the marker-rename mutant really changed the "
                         "source it is built from)")
    with _tf.TemporaryDirectory() as _d:
        _cp = Path(_d) / "copy.py"
        _cp.write_text(_dupname)
        _bad = _site_names(_cp)
    ok(len(set(_bad.values())) < len(_bad)
       and sorted(_bad.values()).count("check#2") == 2,
       f"KNOWN-BAD, ON A REAL COPY: renaming one marker to another's makes "
       f"`check#2` name TWO raises ({len(_bad)} markers, "
       f"{len(set(_bad.values()))} names) and the uniqueness assertion "
       f"above goes red -- run through `_site_names` itself, on a file, "
       f"not by editing the dict it returned (DE16-R4's lesson applied to "
       f"its neighbour)")
    ok(all(v["file"] == "de_ratification_check.py" and v["lineno"] > 0
           and v["site"] != "<untagged>"
           for v in audit["raise_site_by_case"].values()),
       "and each site is keyed on (filename, lineno) READ FROM THE "
       "TRACEBACK and resolved to its `# SITE:` name, so a raise that "
       "moves down the file keeps its identity while a raise that is "
       "deleted loses it")
    # ---- THE FALSIFIERS, DRIVEN THROUGH THE AUDIT (DE16-R4) -----------
    # These three used to take `_reached`, edit a COPY and assert it was
    # `!= EXPECTED_SITE` -- two lines after `_reached == EXPECTED_SITE` was
    # asserted and the suite exited if that failed. Evaluated under a
    # correct map, a migrated one and a lost one they returned True, True,
    # True: they asserted a property of dict equality and never drove the
    # coverage assertion on a mutated input. They wore KNOWN-BAD labels in
    # the round that closed "derived from the very dict it was compared to,
    # so it could not fail" -- the same defect, one artefact over.
    #
    # Now the HARNESS is mutated and the map is recomputed from real
    # tracebacks, which is what the copy-mutant outside the suite does. The
    # hook is chosen over deleting the three and citing that mutant because
    # rule 15 asks the checker to SHIP its falsifier: a mutant that lives
    # in a filing is one nobody re-runs.
    _dropped = mutation_audit(sup, _drop_case="superseded_new_run")
    ok(not _dropped["coverage_matches_expected"]
       and _dropped["n_cases"] == audit["n_cases"] - 1
       and "superseded_new_run" not in _dropped["site_reached_by_case"],
       f"KNOWN-BAD, DRIVEN: DELETING A CASE from the harness leaves "
       f"{_dropped['n_cases']} cases and `coverage_matches_expected` "
       f"{_dropped['coverage_matches_expected']} -- the reviewer's own test, "
       f"run here rather than approximated by editing a dict the assertion "
       f"had already passed")
    _migrated = mutation_audit(
        sup, _migrate_case=("superseded", "unknown_population_value"))
    ok(not _migrated["coverage_matches_expected"]
       and _migrated["site_reached_by_case"]["superseded"] == "check#9",
       f"KNOWN-BAD, DRIVEN: a case given ANOTHER case's input lands at "
       f"another site under its own name "
       f"({_migrated['site_reached_by_case']['superseded']} where "
       f"{EXPECTED_SITE['superseded']} is recorded) and the coverage "
       f"assertion sees it -- which is exactly what `superseded` and "
       f"`unknown_population_value` did unnoticed before the map existed")
    _added = mutation_audit(sup, _add_case="a_case_nobody_recorded")
    ok(not _added["coverage_matches_expected"]
       and _added["n_cases"] == audit["n_cases"] + 1,
       f"KNOWN-BAD, DRIVEN: a case nobody RECORDED is not silently absorbed "
       f"({_added['n_cases']} cases, coverage "
       f"{_added['coverage_matches_expected']}) -- the map is a producer "
       f"act, so adding a case without recording where it must land is a "
       f"change to the expectation and reads as one")
    ok(_reached["superseded"] == "check#2"
       and _reached["superseded_new_run"] == "check#3",
       f"and the two are now DISTINCT cases at DISTINCT sites: "
       f"`superseded` refuses where the superseder's heading carries no "
       f"parsable timestamp ({_reached['superseded']}), and "
       f"`superseded_new_run` -- new this round, on a well-formed stamped "
       f"chain -- reaches the NEW-RUN refusal itself "
       f"({_reached['superseded_new_run']})")
    ok(_reached["unknown_population_value"] == "check#9"
       and _reached["population_unbindable_from_prose"] == "check#12",
       f"and the population site the audit never reached HAS a driver "
       f"rather than a story: an unknown VALUE dies in the vocabulary "
       f"check ({_reached['unknown_population_value']}) since round 10 "
       f"gave `population` a vocabulary, and the KNOWN_POPULATIONS guard "
       f"({_reached['population_unbindable_from_prose']}) is reachable "
       f"ONLY through the grandfathered prose path, where the field can "
       f"bind to NOTHING -- a block missing it refuses as MALFORMED first. "
       f"Reachable, so it stays")
    ok("n_guards" not in audit
       and audit["note"].count("n_cases") >= 1
       and "n_raise_sites" in audit["note"],
       f"DE14-R4: `n_guards` is GONE from the emission (it carried "
       f"len(per) = the CASE count, in the round whose point was that the "
       f"two differ; nothing in the repo read it), and the distinction "
       f"travels WITH the numbers for a machine reading the dict: "
       f"{audit['note']!r}")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_ratification_check] selftest OK -- {n[0]} checks")
    return 0


def sup_chain_fixture() -> str:
    """R-902 followed by R-903, whose block supersedes it."""
    return (fixture_register("R-902").replace("\n\n## next\n", "\n\n")
            + fixture_register("R-903", supersedes="R-902"))


#: DE14-R2: THE AUDIT REPORTED ITS COVERAGE AND ASSERTED NONE OF IT.
#: Two of 21 cases refused at a guard other than the one their name claimed
#: (`superseded` died at the no-parsable-timestamp guard because
#: `fixture_register()` headings carry none; `unknown_population_value`
#: migrated to the VALUE guard when round 10 gave `population` a vocabulary
#: entry). Neither showed as a number: `n_raise_sites` was asserted only as
#: `1 <= n < n_cases`, and `expected` was `sorted(cases)` -- derived from the
#: dict it was compared to, so it could not fail. Adding two cases or
#: DELETING round 14's own new case both left the suite green.
#:
#: So the expectation is PRODUCER-RECORDED (R-230): case name -> the raise
#: site it must reach. Sites are named `<function>#<ordinal>` rather than by
#: line number, so the map survives an edit elsewhere in the file; the
#: (filename, lineno) is still recorded beside it for a reader.
EXPECTED_SITE: dict[str, str] = {
    "entry_absent": "check#1",
    "superseder_timestamp_unparsable": "check#2",
    "superseded": "check#2",
    "superseded_new_run": "check#3",
    "already_superseded_at_stamp": "check#4",
    "no_block_not_grandfathered": "check#7",
    "empty_block_value": "check#5",
    "malformed_block_missing_field": "check#6",
    "block_ref_mismatch": "check#8",
    "nonsense_field_value": "check#9",
    "unknown_population_value": "check#9",
    "population_unbindable_from_prose": "check#12",
    "later_entry_bad_supersedes": "validate_supersedes#5",
    "later_entry_dangling_supersedes": "superseded_by#1",
    "duplicate_block_key": "bind_from_block#1",
    "two_own_blocks": "own_ratification_blocks#1",
    "under_check_bad_supersedes": "validate_supersedes#5",
    "under_check_dangling_supersedes": "check#16",
    "under_check_self_supersedes": "check#17",
    "under_check_backwards_supersedes": "check#18",
    "duplicate_subject": "entry_index#1",
    "duplicate_named_by_supersedes": "entry_index#2",
    "duplicate_carrying_a_block": "entry_index#3",
    "not_a_ratification": "check#10",
    "counts_do_not_sum": "check#11",
    "sampled_population": "check#13",
    "selection_field": "check#14",
    "self_contradicting_sampling": "check#15",
    "unparsable_now_utc": "parse_instant#2",
    "non_string_now_utc": "parse_instant#1",
    "unparsable_stamped_at": "parse_instant#2",
    "unparsable_stamped_at_not_superseded": "parse_instant#2",
    "unparsable_scope_to": "parse_day#2",
    "unparsable_scope_from": "parse_day#2",
}


def _site_names(path: Path | None = None) -> dict:
    """lineno of each `raise RatificationRefused` -> its `# SITE:` name,
    read from this module's own source. Computed, never transcribed."""
    src = (path or Path(__file__)).read_text().split("\n")
    out, pending = {}, None
    for i, ln in enumerate(src, 1):
        st = ln.strip()
        if st.startswith("# SITE: "):
            pending = st[len("# SITE: "):]
        elif st.startswith("raise RatificationRefused(") and pending:
            out[i] = pending
            pending = None
    return out


def mutation_audit(sup: dict, *, _drop_case: str | None = None,
                   _migrate_case: tuple[str, str] | None = None,
                   _add_case: str | None = None) -> dict:
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
    # The grandfathered prose pair. The bad one carries the KIND vocabulary
    # and no population sentence, so `population` binds to nothing at all --
    # `fields.get("population")` is None, which is not an unknown value but
    # an unbound field, and the two refuse at different sites.
    prose_no_population = (
        "### R-418 — coordinator: R-ADMISS ratification\n\n"
        "Body naming data/pm_5min/markets.jsonl for a forward-race day.\n\n"
        "## next\n")
    prose_full = (
        "### R-418 — coordinator: R-ADMISS ratification — the population "
        "is the FULL supplied complement, no sampling\n\nBody naming "
        "data/pm_5min/markets.jsonl for a forward-race day.\n\n## next\n")
    # DE16-R1/R2/R3's three refusals, MEASURED here rather than only
    # asserted in the selftest: each needs an input that trips it and one
    # that must reach past it.
    dangling = stamped_chain.replace("supersedes: R-902", "supersedes: R-9021")
    dup_key = fixture_register("R-902").replace(
        "scope_to: null", "scope_to: null\nscope_to: 20260901")
    def _stamped(rows):
        """entries in file order, headings stamped; sv=NO_BLOCK = prose."""
        out, hh = [], 9
        for r, sv in rows:
            fields = ["ref: " + r, "kind: R-ADMISS", f"population: {POP_FULL}",
                      "sampling: NONE",
                      "present_source: data/pm_5min/markets.jsonl",
                      "scope_days: FORWARD_RACE_DAYS",
                      "scope_from: 20260901", "scope_to: null",
                      "revocable_by: USER",
                      f"supersedes: {sv if sv else 'null'}"]
            body = ("\n```ratification\n" + "\n".join(fields) + "\n```\n"
                    if sv != "NO_BLOCK" else "\nprose only, no block\n")
            out.append(f"### {r} — 2026-09-02T{hh:02d}:00Z — coordinator: "
                       f"R-ADMISS\n{body}\n")
            hh += 1
        return "".join(out) + "## next\n"

    dup_subject = _stamped([("R-902", "NO_BLOCK"), ("R-902", "NO_BLOCK")])
    dup_named = _stamped([("R-902", "NO_BLOCK"), ("R-902", "NO_BLOCK"),
                          ("R-903", "R-902")])
    dup_block = _stamped([("R-902", None), ("R-902", None), ("R-903", None)])
    backwards = (fixture_register("R-902", supersedes="R-903")
                 .replace("\n\n## next\n", "\n\n")
                 + fixture_register("R-903"))
    two_own = (fixture_register("R-902").replace("\n\n## next\n", "\n\n")
               + "```ratification\nref: R-902\nkind: R-ADMISS\n"
               f"population: {POP_FULL}\nsampling: NONE\n"
               "present_source: data/pm_5min/markets.jsonl\n"
               "scope_days: FORWARD_RACE_DAYS\nscope_from: 20260901\n"
               "scope_to: null\nrevocable_by: USER\nsupersedes: null\n"
               "```\n\n## next\n")
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
        # CO-7: the SAME refusal driven on the NON-superseded branch -- the
        # one the parse used to skip entirely. Driving it only on R-418
        # exercised the path that already refused and left the blind one
        # uncovered, which is how a fix without a falsifier looks from the
        # audit's side.
        "unparsable_stamped_at_not_superseded": (
            ((sup, "R-419", None), {"stamped_at": "nope"}),
            ((sup, "R-419", None),
             {"stamped_at": "2026-09-02T10:30:00Z"})),
        "empty_block_value": (
            ((sup, "R-900", fixture_register(scope_to="")), {}),
            ((sup, "R-900", good), {})),
        "already_superseded_at_stamp": (
            ((sup, "R-902", stamped_chain),
             {"stamped_at": "2026-09-02T10:30:00Z"}),
            ((sup, "R-902", stamped_chain),
             {"stamped_at": "2026-09-02T09:30:00Z"})),
        # --- DE14-R2: sites the audit reported nothing about ------------
        # The NEW-RUN refusal on a WELL-FORMED chain. `superseded` above
        # never reached it: its fixture headings carry no timestamp, so it
        # died one guard earlier and the audit still printed its name.
        "superseded_new_run": (((sup, "R-902", stamped_chain), {}),
                               ((sup, "R-903", stamped_chain), {})),
        # The UNBINDABLE population, which is a different fact from an
        # unknown VALUE and refuses at a different site. Only the
        # grandfathered prose path can produce it: a block missing the
        # field refuses as MALFORMED first.
        "population_unbindable_from_prose": (
            ((sup, "R-418", prose_no_population), {}),
            ((sup, "R-418", prose_full), {})),
        # --- DE14-R1's two call sites, measured rather than promised ----
        "later_entry_bad_supersedes": (
            ((sup, "R-902",
              stamped_chain.replace("supersedes: R-902", "supersedes: R-9O2")),
             {"stamped_at": "2026-09-02T09:30:00Z"}),
            ((sup, "R-902", stamped_chain),
             {"stamped_at": "2026-09-02T09:30:00Z"})),
        "under_check_bad_supersedes": (
            ((sup, "R-900", fixture_register(supersedes="WHATEVER")), {}),
            ((sup, "R-900", good), {})),
        # DE18-R1: the same field, the other question -- shape passes and
        # the target names no entry.
        "under_check_dangling_supersedes": (
            ((sup, "R-900", fixture_register(supersedes="R-777")), {}),
            ((sup, "R-900", good), {})),
        # DE20-R2: the same field again -- shape passes, the target exists,
        # and the claim still supersedes nothing.
        "under_check_self_supersedes": (
            ((sup, "R-902", fixture_register("R-902", supersedes="R-902")),
             {}),
            ((sup, "R-900", good), {})),
        "under_check_backwards_supersedes": (
            ((sup, "R-902", backwards), {}),
            ((sup, "R-900", good), {})),
        # DE22-R1: the three ways a duplicated ref reaches an answer.
        "duplicate_subject": (((sup, "R-902", dup_subject), {}),
                              ((sup, "R-900", good), {})),
        "duplicate_named_by_supersedes": (
            ((sup, "R-903", dup_named), {}), ((sup, "R-900", good), {})),
        "duplicate_carrying_a_block": (
            ((sup, "R-903", dup_block), {}), ((sup, "R-900", good), {})),
        # --- DE16-R1/R2/R3 -----------------------------------------------
        "later_entry_dangling_supersedes": (
            ((sup, "R-902", dangling),
             {"stamped_at": "2026-09-02T09:30:00Z"}),
            ((sup, "R-902", stamped_chain),
             {"stamped_at": "2026-09-02T09:30:00Z"})),
        "duplicate_block_key": (((sup, "R-902", dup_key), {}),
                                ((sup, "R-900", good), {})),
        "two_own_blocks": (((sup, "R-902", two_own), {}),
                           ((sup, "R-900", good), {})),
    }
    # ---- DE16-R4: the TEST HOOKS, so the coverage assertion is driven
    # from inside the suite rather than compared to a copy of itself.
    # The three lines that used to follow the assertion took `_reached`,
    # edited a copy and asserted it was `!= EXPECTED_SITE` -- two lines
    # after `_reached == EXPECTED_SITE` had been asserted and the suite had
    # exited if it failed. They were properties of dict equality: True
    # under the correct map, True under a migrated one, True under a lost
    # one. The exact wording of the finding is what the round they were
    # written for had closed -- "derived from the very dict it was compared
    # to, so it could not fail" -- and they carried KNOWN-BAD labels.
    #
    # These hooks MUTATE THE HARNESS ITSELF, so `raise_site_by_case` is
    # recomputed from real tracebacks over mutated input and the real
    # assertion sees a genuinely wrong map. `_drop_case` removes a case
    # (the reviewer's test), `_migrate_case=(name, other)` gives one case
    # ANOTHER case's input so it lands at another site under its own name,
    # and `_add_case` runs a case nobody recorded.
    if _drop_case is not None:
        assert _drop_case in cases, _drop_case      # the hook itself fails
        cases.pop(_drop_case)                       # loudly on a stale name
    if _migrate_case is not None:
        _from, _to = _migrate_case
        assert _from in cases and _to in cases, _migrate_case
        cases[_from] = (cases[_to][0], cases[_from][1])
    if _add_case is not None:
        assert _add_case not in cases, _add_case
        cases[_add_case] = cases["entry_absent"]
    import traceback as _tb
    per: dict[str, dict] = {}
    sites: dict[str, tuple] = {}
    for name, ((bad_a, bad_k), (ctl_a, ctl_k)) in cases.items():
        try:
            check(*bad_a, **bad_k)
            live = False
        except RatificationRefused:
            # WHERE it raised, computed. The reviewer's note (accepted, not a
            # finding): these are (input, expected-refusal) CASES over fewer
            # raise SITES -- for a shared parser that is call-site coverage,
            # which is the right design, and a reader must not read the case
            # count as a guard count. So both numbers are reported and
            # neither is narrated.
            _f = _tb.extract_tb(sys.exc_info()[2])[-1]
            sites[name] = (_f.filename.rsplit("/", 1)[-1], _f.lineno)
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
    names = _site_names()
    by_case = {k: {"file": f, "lineno": ln, "site": names.get(ln, "<untagged>")}
               for k, (f, ln) in sites.items()}
    reached = {k: v["site"] for k, v in by_case.items()}
    return {
        "n_cases": len(per),
        "n_raise_sites": len({v["site"] for v in by_case.values()}),
        # THE DISTINCTION LIVES BESIDE THE NUMBERS (DE14-R4), not only in a
        # selftest label a JSON reader never sees.
        "note": ("n_cases counts (input, expected-refusal) CASES; "
                 "n_raise_sites counts the distinct raises they reach. "
                 "Several inputs through one refusal is call-site coverage "
                 "for a shared parser -- do NOT read n_cases as a guard "
                 "count."),
        "raise_site_by_case": by_case,
        "site_reached_by_case": reached,
        "expected_site": dict(EXPECTED_SITE),
        "coverage_matches_expected": reached == EXPECTED_SITE,
        "cases_per_site": {
            st: sorted(k for k, v in reached.items() if v == st)
            for st in sorted({v for v in reached.values()})},
        "per_guard": per, "survivors": survivors,
        "all_load_bearing": not survivors}


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
