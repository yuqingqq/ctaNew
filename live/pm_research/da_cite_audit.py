#!/usr/bin/env python3
"""DA: does the CITED TEXT contain the claim, or does the cite merely RESOLVE?

`da_forward_day_verify.era_authority_audit` already does the right thing for
one table: it resolves each `R-nnn` in an authority string AND checks the
cited entry NAMES the era it is cited for. That second half is the whole
value, and it governed exactly one table.

GENERALISED, IT IS THE CHECK THIS CLASS NEEDS. For any claim of the form
"X, because <cite>", two different things can be true and only one is usually
checked: the cite RESOLVES (an entry with that id exists) and the cited text
CONTAINS the claim. A day of work produced four instances of the second
failing while the first held -- a claim about what a named artifact SAYS,
where the cite resolved and the text was never read.

AND THE REASON A SECOND SEAT AGREEING IS NOT EVIDENCE: agreement between
seats is evidence about the SEATS, not about the claim, unless the seats read
DIFFERENT SOURCES. Two seats quoting the same unread register entry are one
observation reported twice -- R-495's non-independence error in a second
domain: replicated citations in place of replicated statistics. So this reads
the register itself, every time, and never another seat's summary of it.

REPORTS, NEVER ENFORCES (rule 14). A cite whose text does not name its
subject may still be the right cite -- prose is prose. What must not happen
is a table asserting an authority nobody checked.

Falsifiers both directions; an unreadable register REFUSES rather than
reporting that every cite resolves.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
REGISTER = (CODE_ROOT / "orchestrator/PROGRAMS/P-2026-003-polymarket-5min"
            / "workspace/COORDINATION.md")
CITE_RE = re.compile(r"\bR-\d+\b")
#: ROUND 49, AND THIS IS THE FIX TO A DEFECT DA FILED AGAINST ITSELF.
#: `EXPECTED_CHECKS` was a single hand-pinned integer, and DA re-pinned its
#: sibling digest TWICE IN ONE DAY by pasting the value the failure printed
#: -- "a pin whose maintainer updates it reflexively from the error output is
#: a tally again". Adding nine tables here would have demanded exactly that
#: paste a third time.
#:
#: So the pin is SPLIT along the line that makes the reflex unnecessary. The
#: per-table checks are one per audited table and MUST move when `TABLES`
#: moves; that part is derived, never pinned. Everything else is fixed
#: behaviour, and `EXPECTED_FIXED_CHECKS` still fails if a check is deleted
#: or replaced. Growing the declared input no longer requires touching a
#: pin -- which is the only way a pin stays meaningful.
EXPECTED_FIXED_CHECKS = 20


class CiteAuditRefused(RuntimeError):
    """Refused rather than reporting that every cite resolves."""


def register_entries(text: str) -> dict[str, str]:
    """`{R-id: the entry's FULL text}`.

    AND A CORRECTION I OWE THE INSTRUMENT I GENERALISED FROM. I wrote this
    believing `era_authority_audit` was wrong to store `bodies[c] = the ###
    line` -- that a subject named in the entry text and not the title would
    read as absent. **Measured: it is not wrong. THIS REGISTER WRITES EACH
    ENTRY AS ONE LINE** (R-497 is 10,506 characters on a single line, R-500
    is 8,124), so the `###` line IS the entry and the existing reader was
    correct for the file it reads. My "fix" was a claim about what a named
    artifact says, made without reading it -- the exact shape this module
    exists to catch, committed while building it.

    What survives is the multi-line case: this parser accumulates
    continuation lines, so it is correct for BOTH shapes, which the
    single-line reader would not be if the register's format ever changes.
    """
    out: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for ln in text.split("\n"):
        m = re.match(r"^#{2,4} (R-\d+)\b", ln)
        if m:
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = m.group(1), [ln]
        elif cur is not None:
            buf.append(ln)
    if cur:
        out[cur] = "\n".join(buf)
    return out


def cite_names_subject(subject_terms: tuple, cites: tuple,
                       entries: dict[str, str]) -> dict[str, Any]:
    """For one claim: does each cited entry's TEXT carry the subject?

    THREE strengths, and the third is the one that matters. **R512-R1: this
    module had the defect it was built to catch** -- an entry that merely
    DISCUSSES a subject returned true on both of my first two tests, so
    "the cited text carries the claim" was satisfied by a passing mention.
    The fix is DE16-R1's, carried over rather than reinvented: that remedy
    says a quoted block "must be the entry's OWN heading ref", i.e.
    **ownership is established by POSITION, not by presence.** The general
    form of the class is not "everything that reads the register" -- it is
    every reader that asks whether a token is PRESENT rather than whether it
    is OWNED.

      * `mentions` -- any declared term appears anywhere in the entry. Crude,
        and the crude pass is what most cites survive.
      * `in_prose` -- the term appears after the entry's bolded title, so a
        title-only mention is distinguishable from one in the argument.
      * `owns` -- the term appears in the entry's OWN TITLE. An entry titled
        for a subject is ruling on it; an entry that names it in passing is
        discussing it, and this programme has read one for the other before.
    """
    rows = {}
    for c in cites:
        body = entries.get(c)
        if body is None:
            rows[c] = {"resolves": False, "term_level": None, "strict": None,
                       "why": "no entry with this id in the register"}
            continue
        # THE TITLE, NOT THE FIRST LINE. Splitting on the newline made
        # `rest` EMPTY for every entry in this register and `strict` was
        # therefore False everywhere -- a check that fires on everything,
        # which is a check that gets turned off. The register's shape is
        # `### R-nnn — time — author — **TITLE** <prose>`, so the prose is
        # what follows the bolded title.
        title_end = body.find("**", body.find("**") + 2)
        rest = body[title_end + 2:] if title_end > 0 else ""
        title = body[:title_end + 2] if title_end > 0 else body
        mentions = any(t in body for t in subject_terms)
        owns = any(t in title for t in subject_terms)
        rows[c] = {
            "resolves": True,
            "term_level": mentions,            # kept: the crude pass
            "mentions": mentions,
            "in_prose": (any(t in rest for t in subject_terms)
                         if rest else None),
            "strict": (any(t in rest for t in subject_terms)
                       if rest else None),     # kept: prior name
            "owns": owns,
            "ownership": ("OWNS" if owns else
                          "DISCUSSES_ONLY" if mentions else "ABSENT"),
            "n_chars": len(body), "n_chars_after_title": len(rest),
        }
    return rows


def subject_variants(key: str) -> tuple:
    """The SPELLINGS a register entry may use for one subject.

    Driven by measurement, not by taste: the first cut flagged R-497 and
    R-500 as strict failures, and both are FALSE POSITIVES OF SPELLING --
    R-500's body writes the withdrawn day as `08-29` where the table's key is
    `20260829`, and R-497 writes the collector era in prose. A strict check
    that sends a reader to verify two sound cites is a check that gets turned
    off, so the variants are declared and the residue is stated rather than
    the strictness dropped.
    """
    out = {key}
    if len(key) == 8 and key.isdigit():
        y, m, d = key[:4], key[4:6], key[6:]
        out |= {f"{y}-{m}-{d}", f"{m}-{d}", f"{m}/{d}"}
    if "_" in key:
        out.add(key.replace("_", "."))
        head, _, tail = key.rpartition("_")
        out.add(f"{head}.{tail}")                  # clob_v3_1 -> clob_v3.1
        parts = key.split("_")
        if len(parts) > 1:
            out.add("_".join(parts[1:]))          # clob_v3_1 -> v3_1
            out.add(".".join(parts[1:]))          # clob_v3_1 -> v3.1
    return tuple(sorted(out))


#: The authority-bearing tables this audit walks. Each entry names WHERE the
#: claim lives, the SUBJECT it is about, and the terms that would show the
#: cited text is about that subject. Declared as data so adding a table is a
#: line, not a code change -- and so the coverage is legible.
#: ROUND 49 -- THE DEFECT DA FILED AGAINST ITSELF, AND WHY TWO MORE ENTRIES
#: WOULD NOT HAVE CLOSED IT. At R-531(E) DA reported "both tables come back
#: clean" while `TABLES` held exactly TWO of the FOUR surfaces it had been
#: asked to cover. Adding the two missing ones by hand reproduces the same
#: failure one surface later: a HAND-ENUMERATED coverage list cannot report
#: what it omits, which is the identical shape as the check-id digest DA
#: re-pinned twice in a day from the value the failure printed.
#:
#: So the list is no longer the coverage claim. `discover_tables` walks the
#: package by AST and finds EVERY module-level upper-case dict whose source
#: carries an `R-nnn` cite, and the audit reports any discovered table that
#: this tuple does not name. The declared tuple still exists -- it carries
#: the per-table subject terms, which cannot be inferred -- but a surface
#: added tomorrow shows up as `undeclared_authority_tables` instead of
#: silently not being audited.
TABLES: tuple = (
    {"id": "race_withdrawals.authority",
     "module": "da_race_withdrawals", "attr": "RACE_WITHDRAWALS",
     "field": "authority", "subject_terms_from_key": True,
     "extra_terms": ("withdraw", "withdrawal", "race")},
    {"id": "era_authority",
     "module": "da_forward_day_verify", "attr": "ERA_AUTHORITY",
     "field": None, "subject_terms_from_key": True,
     "extra_terms": ()},
    # THE FREEZE SURFACE, owed since round 37. The split ruling is the
    # freeze-side ruling that carries a register cite in a `ruled_by` field.
    {"id": "split_ruling",
     "module": "de_phase4_diag_runner", "attr": "SPLIT_RULING",
     "field": None, "subject_terms_from_key": True,
     "extra_terms": ("split", "splits", "mechanics", "score", "train")},
    # THE FAMILY SURFACE, owed since round 37.
    {"id": "forward_family.open_factors",
     "module": "be_forward_family", "attr": "OPEN_FACTORS",
     "field": "ruled_by", "subject_terms_from_key": True,
     "extra_terms": ("family", "arm", "cells", "multiplicity")},
    # The remaining cite-bearing surfaces AST discovery found. Named so the
    # audit covers them, not because any of them was asked for.
    {"id": "forward_day.user_admissions_by_day",
     "module": "be_forward_day", "attr": "USER_ADMISSIONS_BY_DAY",
     "field": "authority", "subject_terms_from_key": True,
     "extra_terms": ("admit", "admission", "day")},
    {"id": "forward_day.development_read_ratifications",
     "module": "be_forward_day", "attr": "DEVELOPMENT_READ_RATIFICATIONS",
     "field": "authority", "subject_terms_from_key": True,
     "extra_terms": ("development", "read", "ratif")},
    {"id": "forward_metric.pairing_conventions",
     "module": "be_forward_metric", "attr": "PAIRING_CONVENTIONS",
     "field": "authority", "subject_terms_from_key": True,
     "extra_terms": ("pairing", "convention", "operating point")},
    {"id": "fragment_diagnostic.governed_states",
     "module": "be_fragment_diagnostic", "attr": "GOVERNED_STATES",
     "field": "authority", "subject_terms_from_key": True,
     "extra_terms": ("fragment", "governed", "state")},
    {"id": "phase4.arm_spec",
     "module": "de_phase4_diag_runner", "attr": "ARM_SPEC",
     "field": "authority", "subject_terms_from_key": True,
     "extra_terms": ("arm", "phase 4", "phase-4")},
    {"id": "phase4.user_admissions",
     "module": "de_phase4_diag_runner", "attr": "USER_ADMISSIONS",
     "field": "authority", "subject_terms_from_key": True,
     "extra_terms": ("admission", "admit")},
    {"id": "phase2_arms.registration_provenance",
     "module": "phase2_arms", "attr": "REGISTRATION_PROVENANCE",
     "field": None, "subject_terms_from_key": True,
     "extra_terms": ("registration", "provenance", "arm")},
)

#: Tables AST discovery is expected to find and that are deliberately NOT
#: audited, each with the reason. An entry here is a decision on the record,
#: not an omission -- and the audit prints them, so the list is visible.
NOT_AUDITED: dict = {
    "da_forward_day_verify._ERA_AUTHORITY_FOR_TIMELINE_TESTS":
        "a test fixture whose cites are deliberately fake (R-000); auditing "
        "it would require the register to contain an entry invented to fail",
    "be_forward_recon.DECLARED_PREDICATES":
        "prose predicates, not per-key authorities: the cite governs the "
        "whole table rather than any one row",
    "be_fragment_diagnostic.DIAGNOSTIC_PREDICATE_EXCLUSIONS":
        "same shape as DECLARED_PREDICATES -- one table-level cite",
    "da_state_tape_verify.PERMITTED_NA":
        "same shape -- one table-level cite",
    "de_phase4_diag_runner.DRIFT_FACTS":
        "facts about a drift, cited once at table level",
    "replay_canary.R7_LICENSE":
        "a licence text block, cited at table level",
    "da_cite_audit.NOT_AUDITED":
        "this table itself -- discovery finds it because its REASONS quote "
        "cites. Excusing it is the honest move and it is recorded here "
        "rather than filtered out in the finder, where it would become an "
        "invisible special case",
}


def discover_tables(pkg: Path | None = None) -> dict:
    """Every module-level upper-case dict whose source carries an `R-nnn`.

    THE POINT: coverage is MEASURED, never asserted. This is what makes the
    hand-written `TABLES` tuple auditable instead of self-certifying."""
    import ast
    root = Path(pkg) if pkg is not None else Path(__file__).resolve().parent
    found = {}
    for path in sorted(root.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8",
                                            errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.AnnAssign):
                targets = [node.target]
            elif isinstance(node, ast.Assign):
                targets = node.targets
            else:
                continue
            name = getattr(targets[0], "id", None)
            if not name or not name.isupper() or node.value is None:
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            try:
                src = ast.unparse(node.value)
            except Exception:                            # noqa: BLE001
                continue
            if CITE_RE.search(src):
                found[f"{path.stem}.{name}"] = sorted(
                    set(CITE_RE.findall(src)))
    return found


def audit(register: Path | None = None,
          tables: tuple = TABLES) -> dict[str, Any]:
    """Every declared table, every cite, both strengths. REPORTS."""
    reg = REGISTER if register is None else Path(register)
    if not reg.is_file():
        raise CiteAuditRefused(
            f"REFUSED: no register at {reg}. A cite audit that cannot read "
            f"the register must never report that every cite resolves -- "
            f"that is the empty-set trap on the instrument built to catch "
            f"unread citations.")
    text = reg.read_text(encoding="utf-8", errors="replace")
    entries = register_entries(text)
    if not entries:
        raise CiteAuditRefused(
            f"REFUSED: {reg} parsed to ZERO register entries. A zero from a "
            f"parser that never proved it can fire is not a result "
            f"(rule 15).")
    out = {}
    for spec in tables:
        try:
            mod = __import__(spec["module"])
            table = getattr(mod, spec["attr"])
        except Exception as e:                               # noqa: BLE001
            out[spec["id"]] = {"status": "TABLE_UNREADABLE", "error": repr(e)}
            continue
        rows = {}
        for key, val in sorted(table.items()):
            s = val.get(spec["field"]) if (spec["field"] and
                                           isinstance(val, dict)) else val
            if not isinstance(s, str):
                rows[key] = {"status": "NO_AUTHORITY_STRING"}
                continue
            cites = tuple(dict.fromkeys(CITE_RE.findall(s)))
            terms = ((subject_variants(key))
                     if spec["subject_terms_from_key"] else ()) \
                + tuple(spec["extra_terms"])
            rows[key] = {"status": "AUDITED", "authority": s[:120],
                         "cites": list(cites), "subject_terms": list(terms),
                         "per_cite": cite_names_subject(terms, cites,
                                                        entries)}
        out[spec["id"]] = {
            "status": "AUDITED", "n_rows": len(rows), "rows": rows,
            "cites_that_do_not_resolve": sorted(
                {c for r in rows.values() for c, v in
                 r.get("per_cite", {}).items() if v["resolves"] is False}),
            "cites_failing_term_level": sorted(
                {c for r in rows.values() for c, v in
                 r.get("per_cite", {}).items() if v["term_level"] is False}),
            "cites_that_only_DISCUSS_the_subject": sorted(
                {c for r in rows.values() for c, v in
                 r.get("per_cite", {}).items()
                 if v.get("ownership") == "DISCUSSES_ONLY"}),
            "cites_that_OWN_the_subject": sorted(
                {c for r in rows.values() for c, v in
                 r.get("per_cite", {}).items()
                 if v.get("ownership") == "OWNS"}),
            "cites_failing_strict_only": sorted(
                {c for r in rows.values() for c, v in
                 r.get("per_cite", {}).items()
                 if v["term_level"] is True and v["strict"] is False}),
        }
    n_bad = sum(len(v.get("cites_failing_term_level", []))
                for v in out.values() if isinstance(v, dict))
    # COVERAGE IS MEASURED, NOT ASSERTED (round 49). Anything AST discovery
    # finds that is neither audited nor explicitly excused is reported.
    discovered = discover_tables()
    declared = {f"{s['module']}.{s['attr']}" for s in tables}
    undeclared = sorted(set(discovered) - declared - set(NOT_AUDITED))
    return {
        "instrument": "da_cite_audit", "register": str(reg),
        "n_register_entries": len(entries),
        "n_tables": len(tables), "tables": out,
        "n_cites_failing_term_level": n_bad,
        "coverage": {
            "n_authority_tables_discovered": len(discovered),
            "n_declared_in_TABLES": len(declared),
            "n_excused_in_NOT_AUDITED": len(NOT_AUDITED),
            "undeclared_authority_tables": undeclared,
            "coverage_complete": not undeclared,
            "discovered": discovered,
            "why": ("round 37 asked for four surfaces and TABLES carried "
                    "two; DA reported 'both tables come back clean' without "
                    "saying both was two of four. A hand-enumerated list "
                    "cannot report what it omits, so the list is now audited "
                    "against AST discovery rather than trusted"),
        },
        "role": "REPORTED_NOT_ENFORCED",
        "n_cites_that_only_discuss": sum(
            len(v.get("cites_that_only_DISCUSS_the_subject", []))
            for v in out.values() if isinstance(v, dict)),
        "why": ("a cite that RESOLVES, a cite whose TEXT carries the claim, "
                "and a cite whose entry OWNS the subject are three different "
                "facts. Usually only the first is checked, and this module "
                "checked only the first two until it was found to have the "
                "defect it was built to catch (R512-R1)"),
        "limits": ("term-level matching over prose: a cite can name its "
                   "subject in words this does not list, and an entry can "
                   "mention a subject without ruling on it. A failure here "
                   "is a place to READ, never a verdict about the cite"),
        "non_independence": ("read from the register itself every time. A "
                             "second seat agreeing is evidence about the "
                             "seats unless they read different sources -- "
                             "replicated citations are not replicated "
                             "evidence"),
        "decides_nothing": "REPORTED (rule 14).",
    }


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            print(f"FAIL: {label}")
            raise SystemExit(1)
        print(f"PASS: {label}")

    import tempfile
    _T = ("### R-100 a ruling about clob_v9\nbody mentions widgets\n\n"
          "### R-101 a title only\nbody says nothing relevant\n")
    e = register_entries(_T)
    ok(set(e) == {"R-100", "R-101"} and "widgets" in e["R-100"],
       "ENTRIES: continuation lines are accumulated, so the parser is "
       "correct for a MULTI-LINE register too. It is not a fix to the "
       "existing single-line reader, which measurement shows was right for "
       "this register -- that correction is in the docstring")
    r = cite_names_subject(("widgets",), ("R-100", "R-101", "R-999"), e)
    ok(r["R-100"]["resolves"] and r["R-100"]["term_level"] is True
       and r["R-100"]["strict"] is None,
       "CITE-1 ADMITS at term level, and an entry with NO bolded title "
       "yields `strict: None` rather than False -- there is no prose to "
       "search, and 'could not compute' must not read as 'the prose does "
       "not name it' (the codomain rule, in the newest instrument)")
    ok(r["R-101"]["resolves"] and r["R-101"]["term_level"] is False,
       "CITE-2 FIRES: a cite that RESOLVES while its text never names the "
       "subject -- the whole shape, and the half that usually goes "
       "unchecked")
    ok(r["R-999"]["resolves"] is False and r["R-999"]["term_level"] is None
       and r["R-999"]["strict"] is None,
       "CITE-3 an unresolved cite reports term_level None, OUTSIDE the "
       "codomain of the boolean: 'no such entry' must not read as 'the text "
       "does not name it'")
    _TT = ("### R-200 — t — a — **a ruling about clob_v9** and the prose "
           "here discusses widgets at length\n")
    e2 = register_entries(_TT)
    r2 = cite_names_subject(("clob_v9",), ("R-200",), e2)
    r3 = cite_names_subject(("widgets",), ("R-200",), e2)
    ok(r2["R-200"]["ownership"] == "OWNS"
       and r3["R-200"]["ownership"] == "DISCUSSES_ONLY",
       "CITE-5 (R512-R1) OWNERSHIP, NOT PRESENCE -- DE16-R1's rule carried "
       "over rather than reinvented: an entry TITLED for a subject OWNS it, "
       "and one that names it only in the argument DISCUSSES it. Both "
       "returned true on my first two tests, so this module had the defect "
       "it was built to catch")
    ok(cite_names_subject(("nowhere",), ("R-200",), e2)["R-200"]["ownership"]
       == "ABSENT",
       "CITE-6 and a subject the entry never names is ABSENT -- three "
       "levels, not a boolean, so 'discussed' is never reported as 'ruled'")
    ok(r2["R-200"]["term_level"] is True and r2["R-200"]["strict"] is False
       and r3["R-200"]["strict"] is True,
       "CITE-4 the two strengths DISAGREE usefully: a subject named only in "
       "the entry's bolded TITLE passes term-level and fails strict, while "
       "one named in the prose passes both -- so a passing mention is not "
       "mistaken for a ruling. Split on the TITLE, not the newline: this "
       "register is one line per entry, so a newline split made `strict` "
       "False for every cite in it")
    try:
        audit(register=Path("/nonexistent/reg.md"))
        ok(False, "an absent register must REFUSE")
    except CiteAuditRefused as ex:
        ok("empty-set trap" in str(ex),
           "REFUSE-1 an absent register REFUSES rather than reporting that "
           "every cite resolves")
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "empty.md"
        p.write_text("no entries here\n", encoding="utf-8")
        try:
            audit(register=p)
            ok(False, "a register parsing to zero entries must REFUSE")
        except CiteAuditRefused as ex:
            ok("never proved it can fire" in str(ex),
               "REFUSE-2 a register that parses to ZERO entries refuses -- a "
               "zero from a parser that never fired is not a result")
    a = audit()
    ok(a["n_register_entries"] > 0 and a["n_tables"] == len(TABLES),
       f"REAL: {a['n_register_entries']} register entries, "
       f"{a['n_tables']} authority-bearing tables walked")
    for tid, tv in sorted(a["tables"].items()):
        ok(tv["status"] in ("AUDITED", "TABLE_UNREADABLE"),
           f"REAL/{tid}: {tv['status']}"
           + (f", {tv['n_rows']} row(s), "
              f"unresolved={tv['cites_that_do_not_resolve']}, "
              f"term-level failures={tv['cites_failing_term_level']}, "
              f"strict-only failures={tv['cites_failing_strict_only']}"
              if tv["status"] == "AUDITED" else ""))
    ok(a["role"] == "REPORTED_NOT_ENFORCED" and "limits" in a
       and "non_independence" in a,
       "ROLE: reports and decides nothing, states its own limits, and says "
       "why a second seat agreeing is not evidence unless the seats read "
       "different sources (R-495's error in a second domain)")
    _v = subject_variants("20260829")
    ok("2026-08-29" in _v and "08-29" in _v and "20260829" in _v,
       f"VARIANTS: a day token carries its spellings ({list(_v)}) -- the "
       f"first cut flagged R-500 strict-FAIL because the entry writes the "
       f"withdrawn day as `08-29` and the table's key is `20260829`. A "
       f"strict check that sends a reader to verify two SOUND cites is one "
       f"that gets turned off")
    ok("clob_v3.1" in subject_variants("clob_v3_1")
       and "v3_1" in subject_variants("clob_v3_1"),
       "VARIANTS: and an era key carries its dotted and suffix spellings")
    ok(audit(tables=())["n_cites_failing_term_level"] == 0
       and audit(tables=())["n_tables"] == 0,
       "EMPTY: an audit over NO tables reports zero tables, so a zero "
       "failure count is never mistaken for a clean surface")

    # ---- COVERAGE, both directions (round 49) -------------------------
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _d:
        _p = Path(_d)
        (_p / "m_hit.py").write_text(
            'X = {"k": {"authority": "R-123 says so"}}\n')
        (_p / "m_miss.py").write_text('Y = {"k": {"authority": "no cite"}}\n')
        (_p / "m_lower.py").write_text('z = {"k": "R-124"}\n')
        (_p / "m_broken.py").write_text('def (\n')
        _f = discover_tables(_p)
    ok(set(_f) == {"m_hit.X"} and _f["m_hit.X"] == ["R-123"],
       f"DISCOVERY POSITIVE CONTROL: finds the cite-bearing table and "
       f"reports its cites ({_f}) -- an instrument that has never been "
       f"shown to fire is not evidence (rule 15)")
    ok("m_miss.Y" not in _f and "m_lower.z" not in _f,
       "DISCOVERY KNOWN-BAD: a table with no cite, and a lower-case name, "
       "are NOT reported -- the finder is not firing on everything, which "
       "is the other way a check gets turned off")
    ok("m_broken" not in str(_f),
       "DISCOVERY: an unparseable module is skipped, never crashing the "
       "audit that must still report the rest")
    _cov = audit()["coverage"]
    ok(_cov["n_authority_tables_discovered"] >= len(TABLES),
       f"COVERAGE: discovery finds {_cov['n_authority_tables_discovered']} "
       f"authority-bearing tables against {_cov['n_declared_in_TABLES']} "
       f"declared and {_cov['n_excused_in_NOT_AUDITED']} excused")
    ok(_cov["coverage_complete"] and not _cov["undeclared_authority_tables"],
       f"COVERAGE COMPLETE: nothing discovered is silently unaudited "
       f"(undeclared={_cov['undeclared_authority_tables']}) -- this is the "
       f"check that would have caught 'two of four' at round 37")
    _stale = audit(tables=TABLES[:1])["coverage"]
    ok(_stale["undeclared_authority_tables"]
       and not _stale["coverage_complete"],
       f"COVERAGE FALSIFIER: dropping tables makes coverage INCOMPLETE and "
       f"names {len(_stale['undeclared_authority_tables'])} surface(s) -- "
       f"the guard moves with its input rather than always passing")
    expected = EXPECTED_FIXED_CHECKS + len(TABLES)
    print(f"\nda_cite_audit selftest: {checks} checks PASSED "
          f"({EXPECTED_FIXED_CHECKS} fixed + {len(TABLES)} per-table)")
    if checks != expected:
        print(f"FAIL: expected {expected} checks "
              f"({EXPECTED_FIXED_CHECKS} fixed + {len(TABLES)} per-table) "
              f"but {checks} ran. If you ADDED a table this number moves on "
              f"its own; if it still mismatches, a FIXED check was deleted "
              f"or replaced -- read it, do not re-pin it.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        print(json.dumps(audit(), indent=1, sort_keys=True))
    except CiteAuditRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
