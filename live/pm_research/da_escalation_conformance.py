"""Does every escalation a plan marks OPEN actually exist in the coordinator's inbox?

THE DEFECT THIS EXISTS FOR, committed by DA on 2026-08-23: SP_PLANE_PLAN's §10
status table marked items 10.17, 10.18, 10.21 and 10.22 "NEW -- OPEN", the plan
prose said they were "escalated, not guessed", and **not one of them had a §0a
register row.**  The plane believed it had escalated; the inbox was empty.  One
of the four was a hard build blocker (two Class-D rows unwritable as keys).

That is the exact failure §0a was created to end -- `COORDINATION.md:20-21`,
*"A request buried in a prose report does not count as asked"* -- committed by
the plane that files into it, and invisible to eight review iterations because
every one of them audited whether the plan's CONTENT was right, never whether
its DELIVERY happened.

R-42's rule applies: the check does not ASK the plan whether it escalated
something; it makes the plan REVEAL it, by requiring a matching row in a file
the plan does not control.  A plan cannot mark itself delivered.

Deliberately generic -- any plane with a status table and a register namespace
can run it.  DA tooling; adopts nothing, changes no CHOSEN value.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

CONFORMANCE_VERSION = "escalation_conformance_v1_r42"

# `| **10.17** | **NEW at iteration 7 -- OPEN.** ... |`
# Key cell capture is deliberately WIDE ([^|]+), not a digit class.  The
# narrow version silently skipped any row whose key it could not match --
# "10.43-10.44" contains a hyphen, so the row never reached the parser and
# BOTH items vanished while the tool reported conformance.  Parsing is now
# scoped to the status table, so a wide capture is safe, and a key that
# looks like item numbers but yields none is REPORTED rather than dropped.
_STATUS_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*$")
_ITEM = re.compile(r"\d+\.\d+")
# a §0a row: `| Q-DA-21 | DA | ... | OPEN |`
_REG_ROW = re.compile(r"^\|\s*(Q-[A-Z]+-\d+)\s*\|\s*([A-Z]+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$")

# "SELF-RESOLVED" is terminal for the COORDINATOR's queue: R-33 makes plane
# self-resolution a legitimate close, and such a row owes no ruling.  Added
# deliberately rather than by accident -- the "IN PART" guard above still keeps
# "SELF-RESOLVED IN PART" open, which is the case that must not close.
CLOSED_WORDS = ("CLOSED", "RULED", "WITHDRAWN", "ANSWERED", "UPHELD",
                "CONFIRMED", "SELF-RESOLVED")


# The status vocabulary, CLOSED.  A status closes an item only if its first
# bolded token matches one of these EXACTLY, after normalising case, spaces and
# punctuation.  Everything else is OPEN.
CLOSED_STATUSES = frozenset({
    "CLOSED", "RULED", "RULED CLOSED", "WITHDRAWN", "WITHDRAWN BY DA",
    "ANSWERED", "UPHELD", "CONFIRMED", "SELF RESOLVED", "RESOLVED",
    "SUPERSEDED", "VOID", "MOOT",
})


def _first_bold_token(status: str) -> str:
    """The first **bolded** run, normalised.  Falls back to the leading words."""
    m = re.search(r"\*\*(.+?)\*\*", status, re.S)
    raw = m.group(1) if m else status
    raw = re.sub(r"[\u2013\u2014,;:.!?()\[\]]", " ", raw)
    raw = raw.replace("-", " ").replace("_", " ")
    return " ".join(raw.upper().split())


def _prefix_closes(prefix: str) -> bool:
    """Does this status close the item?  EXACT MATCH ONLY.

    THIRTEEN FAIL-OPENS were found in this module before this rewrite, and the
    pattern behind them was finally named at iteration 13: every previous fix
    was a SAME-SHAPE REPLACEMENT -- substring for substring, adjacency guard for
    adjacency guard, delimiter list for delimiter list -- and each guard was a
    hand-enumerated vocabulary tested only against its own enumeration.  So the
    tests always passed and the next unenumerated phrasing always got through:
    "UNANSWERED" (substring), "NOT YET RULED" (adjacency), "WITHDRAWN IN
    SUBSTANCE" (partial-guard vocabulary), a comma instead of an em-dash
    (delimiter list).  Selftests caught NONE of the thirteen; all came from
    adversarial use.

    Enumerating what CLOSES is finite and auditable.  Enumerating what does NOT
    close is infinite, and that is what the previous versions kept attempting.
    An unrecognised status is OPEN and therefore demands a register row -- the
    rule this module's docstring always claimed and its code never implemented.
    """
    token = _first_bold_token(prefix)
    if not token:
        return False
    words = token.split()
    # Split STATUS from ATTRIBUTION.  A closure may name who closed it and
    # under what -- "ANSWERED R 37", "SELF RESOLVED BY DA UNDER R 33" -- and
    # that attribution must not defeat the match.  But a QUALIFIER must:
    # "RULED IN SUBSTANCE", "UPHELD IN PART" are partial and stay OPEN.  So the
    # remainder after an exactly-matched status must be attribution and nothing
    # else.  Attribution is a closed, tiny vocabulary; qualifiers are open-ended,
    # which is why this tests the former and not the latter.
    for n in range(len(words), 0, -1):
        if " ".join(words[:n]) in CLOSED_STATUSES:
            return _is_attribution(words[n:])
    return False


def _is_attribution(rest: list[str]) -> bool:
    """Is the remainder only 'who closed it and under what'?

    Empty, or opening with BY/UNDER/PER/VIA, or a bare ruling reference.
    Anything else -- notably IN PART, IN SUBSTANCE, PENDING, EXCEPT -- means the
    closure is qualified, so the item stays open.  Fail-closed by construction:
    an unrecognised remainder is NOT attribution.
    """
    if not rest:
        return True
    if rest[0] in {"BY", "UNDER", "PER", "VIA"}:
        return True
    if rest[0] == "R" and len(rest) > 1 and rest[1].isdigit():
        return True
    return False


def _row_is_open(status: str) -> str:
    """Is a §0a register row still open?  Same fail-closed rule as Item.is_open,
    shared deliberately so the two can never drift apart."""
    return not _prefix_closes(status)


@dataclass(frozen=True)
class Item:
    item: str
    status_text: str

    @property
    def is_open(self) -> bool:
        """OPEN unless the status's first bolded token EXACTLY names a closure.

        No prose is read, no substring is matched, no delimiter is guessed.
        See `_prefix_closes` for why: thirteen fail-opens came from trying to
        enumerate the ways a status might NOT close.
        """
        return not _prefix_closes(self.status_text)


def parse_status_table(text: str, table_marker: str = "STATUS after iteration") -> list[Item]:
    """Parse ONLY the status table, not every two-column table in the document.

    The first version scanned all pipe-rows and picked up an arithmetic table's
    numeric keys ("0.110", "0.150") as escalation items, reporting them as
    orphans forever.  A checker that cries wolf trains its reader to ignore it,
    which is the same end state as not having one.
    """
    items: list[Item] = []
    lines = text.splitlines()
    start = next((i for i, L in enumerate(lines) if table_marker in L), None)
    if start is None:
        # FAIL CLOSED.  Returning [] here would mean "no open items", hence
        # "conforms" -- so renaming the heading would silently switch the
        # checker off while it kept printing a pass.  That is the fail-open
        # shape this module exists to detect, so it must not be its own.
        raise LookupError(
            f"status table not found (marker {table_marker!r}); "
            "cannot certify conformance")
    # Join WRAPPED rows before parsing.  A row that starts with "|" but does not
    # end with one continues on the next line; the regex requires a terminal "|",
    # so such a row was SILENTLY SKIPPED -- and with no matching body the item
    # vanished entirely and the tool reported conformance.  Erasure is the worst
    # failure available to this instrument, so wrapping is joined, not tolerated.
    window, blanks, pending = [], 0, None
    for L in lines[start:]:
        st = L.strip()
        if pending is not None:
            pending += " " + st
            if st.endswith("|"):
                window.append(pending); pending = None
            continue
        if st.startswith("|"):
            blanks = 0
            if st.endswith("|") and st.count("|") >= 2:
                window.append(st)
            else:
                pending = st
        elif window or pending:
            blanks += 1
            if blanks >= 2:
                break
    if pending is not None:
        window.append(pending)
    for line in window:
        m = _STATUS_ROW.match(line)
        if not m:
            continue
        key, status = m.group(1), m.group(2)
        # Only the KEY cell, and only tokens that are wholly a section number.
        # The first version ran `findall` over the key without anchoring, which
        # let decimals appearing in a status cell ("0.110", "0.150") parse as
        # items and report as orphans -- a checker whose own false positives
        # train the reader to ignore it is worse than no checker.
        key = key.replace("*", "").strip()
        toks = [x for x in re.split(r"[,\s]+", key) if x]
        found = [x for x in toks if _ITEM.fullmatch(x)]
        if not found:
            # A key cell that LOOKS like item numbers but parses to none is an
            # erasure, not a non-row: "10.43-10.44" contains no fullmatch token,
            # so both items vanished and the tool certified conformance.  Record
            # it so it fails closed instead of disappearing.
            if any(ch.isdigit() for ch in key) and "." in key:
                items.append(Item(f"UNPARSEABLE_KEY:{key.strip()[:40]}", status))
            continue
        for it in found:
            items.append(Item(it, status))
    return items


def parse_register(text: str, namespace: str) -> list[tuple[str, str, str]]:
    """(row_id, body, status) for rows in this plane's namespace.

    Split from the RIGHT for the status cell.  The regex version matched cells
    left-to-right and mis-split any row whose BODY contained a pipe -- Q-DA-20's
    body quotes `Known[IncentiveContract] | Unavailable`, so its status parsed as
    prose and the row read OPEN after it had been withdrawn.  The status is
    always the last cell, so the last delimiter is the reliable one; the body is
    everything between the plane column and it, pipes and all.
    """
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = s[1:-1]
        head = cells.split("|", 2)
        if len(head) < 3:
            continue
        rid, plane = head[0].strip(), head[1].strip()
        if not rid.startswith(namespace):
            continue
        rest = head[2]
        body, _, status = rest.rpartition("|")
        out.append((rid, body.strip() or rest.strip(), status.strip()))
    return out


_BODY = re.compile(r"^\*\*(\d+\.\d+)[\s\u2014-]", re.M)


def parse_bodies(text: str, section_heading: str = "## 10.") -> set[str]:
    """Escalation BODIES, found independently of the status table.

    Added 2026-08-23 after §10.29 and §10.30 were written as bodies with no
    table row while the checker reported the plan CONFORMING -- it only ever
    validated table -> register, so an item missing from the TABLE was invisible
    to the instrument built to catch missing items.  Both directions are checked
    now: every body needs a row, every open row needs a register entry.

    Scoped to the escalation section and to that section's own numbering,
    because the first version matched any bolded decimal at a line start and
    reported `**3.50 c/share ...**` as an unfiled escalation.  A checker's own
    false positives are how it gets ignored, which is the same end state as not
    having one.
    """
    start = text.find(section_heading)
    if start < 0:
        raise LookupError(
            f"escalation section not found (heading {section_heading!r}); "
            "cannot certify body coverage")
    major = section_heading.strip("# .")
    return {m for m in _BODY.findall(text[start:]) if m.split(".")[0] == major}


def check(plan_text: str, register_text: str, namespace: str) -> dict:
    items = parse_status_table(plan_text)
    rows = parse_register(register_text, namespace)
    open_items = [i for i in items if i.is_open]

    covered, orphaned = {}, []
    for it in open_items:
        # Require an explicit section citation WITH A RIGHT BOUNDARY.
        #
        # THE FOURTH FAIL-OPEN IN THIS MODULE, found 2026-08-23: `f"§{item}" in
        # body` is a SUBSTRING test, so "§10.1" matched "§10.16", "§10.17",
        # "§10.19", "§10.21"...  Item 10.1 reported as covered by FOURTEEN rows
        # when exactly one cited it, and deleting that one row still reported it
        # covered.  The checker written after "four escalations marked OPEN and
        # never asked" would have certified an unasked item as filed.
        #
        # Every previous fix to this module introduced or exposed another hole
        # of the same family, which is why `_selftests` now includes a MUTATION
        # test: remove a row and the tool MUST notice.  A checker nobody can
        # break on demand is a checker nobody has tested.
        pat = re.compile(rf"§{re.escape(it.item)}(?!\d)")
        hits = [rid for rid, body, _ in rows if pat.search(body)]
        if hits:
            covered[it.item] = hits
        else:
            orphaned.append(it.item)
    # THE FIFTH FAIL-OPEN, found at iteration 12: the tool checked
    # table -> register and body -> table, but NOTHING checked that an item the
    # plan marks CLOSED/SELF-RESOLVED had its §0a row closed too.  DA
    # self-resolved §10.20 in the plan and left Q-DA-20 reading OPEN for a full
    # iteration, so the coordinator would have spent a ruling on a withdrawn ask.
    # A withdrawal buried in a plan does not count as withdrawn -- the mirror of
    # §0a's own rule -- and this direction is now checked.
    closed_items = {i.item for i in items if not i.is_open}
    stale_rows = []
    for it in sorted(closed_items):
        pat = re.compile(rf"§{re.escape(it)}(?!\d)")
        for rid, body, status in rows:
            if pat.search(body) and _row_is_open(status):
                stale_rows.append(f"{rid} (open) cites closed §{it}")

    malformed = sorted({i.item for i in items if i.item.startswith("UNPARSEABLE_KEY:")})
    tabled = {i.item for i in items if not i.item.startswith("UNPARSEABLE_KEY:")}
    # No try/except here on purpose.  An earlier draft swallowed the LookupError
    # so test fixtures would pass, which silently disabled the body check
    # whenever the section heading was missing -- a fail-open introduced to make
    # a test go green.  Fixtures carry the heading instead.
    bodies = parse_bodies(plan_text)
    untabled = sorted(bodies - tabled)
    return {
        "conformance_version": CONFORMANCE_VERSION,
        "BODIES_with_no_status_row": untabled,
        "MALFORMED_status_keys": malformed,
        # ADVISORY, not blocking.  Shipped blocking at iteration 12 and it
        # fired on six rows of which THREE were legitimate -- an open row may
        # track a SUCCESSOR obligation under a closed item (Q-DA-27 under
        # §10.12, Q-DA-26 under §10.13, Q-DA-33 under §10.3).  Blocking on an
        # ambiguous signal is how a checker earns being ignored, which is the
        # specificity failure this programme ruled on one tick earlier -- and I
        # shipped it without the control again.  Report, do not block.
        "ADVISORY_open_rows_citing_closed_items": stale_rows,
        "items_total": len(tabled),
        "items_open": sorted({i.item for i in open_items}),
        "register_rows": len(rows),
        "covered": covered,
        "ORPHANED_open_items_with_no_register_row": sorted(set(orphaned)),
        "conforms": not orphaned and not untabled and not malformed,
    }


def _selftests() -> int:
    checks = 0

    def ok(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            raise AssertionError(f"selftest failed: {label}")

    HDR = "## 10. escalations\n**STATUS after iteration 9 -- read this before ruling.**\n"
    plan = (
        HDR
        + "| **10.1** | **RULED, CLOSED** by R-9 |\n"
        "| **10.2** | **NEW -- OPEN.** something unfiled |\n"
        "| **10.3** | **UPHELD IN PART** -- half survives |\n"
    )
    reg = ("| Q-DA-1 | DA | covers §10.3 somehow | OPEN |\n"
           "| Q-BE-1 | BE | mentions 10.2 but wrong plane | OPEN |\n")

    r = check(plan, reg, "Q-DA")
    # 1. the real defect: an OPEN item with no row of THIS plane's namespace
    ok(r["ORPHANED_open_items_with_no_register_row"] == ["10.2"],
       "an open item with no same-plane register row is reported")
    ok(not r["conforms"], "a plan with an unfiled open item does not conform")
    # 2. another plane's row does not discharge this plane's duty
    ok("10.2" not in r["covered"], "a Q-BE row does not cover a DA item")
    # 3. UPHELD IN PART is OPEN -- a surviving half still needs an inbox row
    ok("10.3" in r["items_open"], "'UPHELD IN PART' counts as open")
    ok("10.3" in r["covered"], "and it is covered by its row")
    # 4. closed items are not demanded
    ok("10.1" not in r["items_open"], "RULED/CLOSED is not open")
    # 5. fail-CLOSED: an unrecognised status word must count as OPEN
    ok(check(HDR + "| **10.9** | probably fine? |\n", "", "Q-DA")
       ["ORPHANED_open_items_with_no_register_row"] == ["10.9"],
       "an unparseable status counts as OPEN and demands a row")
    # 6. NEGATIVE CONTROL -- a conforming plan must come back clean, or the
    #    checker would just always fire and mean nothing
    clean = check(HDR + "| **10.4** | **OPEN.** |\n",
                  "| Q-DA-2 | DA | fixes §10.4 | OPEN |\n", "Q-DA")
    ok(clean["conforms"] and not clean["ORPHANED_open_items_with_no_register_row"],
       "NEGATIVE CONTROL: a conforming plan reports clean")
    # 7. a row with several items in one key cell covers each of them
    multi = check(HDR + "| **10.2, 10.3** | **OPEN** |\n", "", "Q-DA")
    ok(multi["ORPHANED_open_items_with_no_register_row"] == ["10.2", "10.3"],
       "a bundled status row is unpacked into its items")

    # 8. FAIL CLOSED on a missing table -- a renamed heading must raise, never
    #    report clean.  This is the check that keeps the checker honest.
    try:
        check("| **10.1** | **OPEN** |\n", "", "Q-DA")
    except LookupError:
        ok(True, "a missing status table raises rather than certifying")
    else:
        ok(False, "a missing table must NOT be reported as conforming")

    # 9. an arithmetic table's numeric keys are not escalation items
    noise = HDR + "| **10.5** | **OPEN** |\n\n\n| 0.110 | 14.6 |\n| 0.150 | 18.9 |\n"
    r9 = check(noise, "", "Q-DA")
    ok(r9["ORPHANED_open_items_with_no_register_row"] == ["10.5"],
       "decimals in a different table are not parsed as items")

    # 10. THE REGRESSION: a closed-sounding word in the EXPLANATION must not
    #     close the item.  This exact cell was live and read CLOSED.
    live = ("**PARTLY DISCHARGED** -- the counts re-derived; the "
            "\"exercises both branches\" claim withdrawn")
    ok(Item("10.14", live).is_open,
       "an explanation containing 'withdrawn' must NOT close a PARTLY item")
    ok(Item("10.9", "**UPHELD IN PART -- R-38** -- RE-OPENED at iteration 7").is_open,
       "'IN PART' stays open")
    ok(Item("10.3", "**RULED, CLOSED** by R-20 (cores verified sound)").is_open is False,
       "a genuine closure still closes")
    ok(Item("10.7", "**WITHDRAWN by DA** after iteration 3").is_open is False,
       "a genuine withdrawal still closes")
    # 11. NEGATIVE CONTROL on the prefix rule: prose mentioning a ruling must
    #     not close an item whose status is open.
    ok(Item("10.5", "**NEW at iteration 9 -- OPEN.** supersedes the ANSWERED "
                    "half of R-35 and the CLOSED 10.2").is_open,
       "NEGATIVE CONTROL: closed-words in prose cannot close an OPEN item")

    # 11b. a bolded MEASUREMENT is not an escalation body
    ok(parse_bodies("## 10. x\n**3.50 c/share against 1.75 c**\n"
                    "**10.4 A real one.**\n") == {"10.4"},
       "a bolded decimal that is not a section number is not a body")
    ok(parse_bodies("## 10. x\n**10.7 real.**\n") == {"10.7"}, "a real body is found")
    try:
        parse_bodies("no section heading here\n")
    except LookupError:
        ok(True, "a missing escalation section RAISES rather than certifying")
    else:
        ok(False, "missing section must not read as zero bodies")

    # 12. THE MIRROR: an escalation body with no status row must be reported.
    #     This is how 10.29/10.30 slipped through as "conforming".
    body_only = (HDR + "| **10.1** | **RULED, CLOSED** |\n\n\n"
                 "**10.9 A new escalation nobody tabled.**\nsome text\n")
    r12 = check(body_only, "", "Q-DA")
    ok(r12["BODIES_with_no_status_row"] == ["10.9"],
       "MIRROR: a body with no status-table row is reported")
    ok(not r12["conforms"], "and it blocks conformance")
    ok(check(HDR + "| **10.9** | **OPEN** |\n\n\n**10.9 tabled.**\n",
             "| Q-DA-1 | DA | §10.9 | OPEN |\n", "Q-DA")["conforms"],
       "NEGATIVE CONTROL: a tabled, filed body still conforms")

    # 13. THE SUBSTRING BUG, pinned.  "§10.1" must not be satisfied by "§10.16".
    plan13 = HDR + "| **10.1** | **OPEN** |\n| **10.16** | **OPEN** |\n"
    reg13 = "| Q-DA-1 | DA | covers §10.16 only | OPEN |\n"
    r13 = check(plan13, reg13, "Q-DA")
    ok("10.1" in r13["ORPHANED_open_items_with_no_register_row"],
       "a row citing §10.16 must NOT cover §10.1")
    ok("10.16" not in r13["ORPHANED_open_items_with_no_register_row"],
       "and it must still cover §10.16")

    # 14. MUTATION TEST -- the property that makes the tool trustworthy: for a
    #     covered item, deleting its ONLY citing row must produce an orphan.
    #     The substring bug survived three prior fixes because nothing ever
    #     tried to break the tool on purpose.
    plan14 = HDR + "| **10.4** | **OPEN** |\n"
    reg14 = ("| Q-DA-1 | DA | fixes §10.4 | OPEN |\n"
             "| Q-DA-2 | DA | unrelated §10.40 §10.41 | OPEN |\n")
    ok(check(plan14, reg14, "Q-DA")["conforms"], "covered item conforms")
    mutated = "| Q-DA-2 | DA | unrelated §10.40 §10.41 | OPEN |\n"
    ok(not check(plan14, mutated, "Q-DA")["conforms"],
       "MUTATION: deleting the only citing row MUST break conformance")

    # 15. SELF-RESOLVED closes; SELF-RESOLVED IN PART does not.
    ok(Item("10.33", "**SELF-RESOLVED BY DA under R-33 clause 3 — NO RULING OWED**")
       .is_open is False, "a self-resolution closes the coordinator's queue")
    ok(Item("10.26", "**SELF-RESOLVED IN PART** — residue needs a ruling").is_open,
       "but a partial self-resolution stays open")

    # 16. THE FIFTH DIRECTION: an OPEN row citing a CLOSED item is reported.
    plan16 = HDR + "| **10.20** | **SELF-RESOLVED BY DA — NO RULING OWED** |\n"
    r16 = check(plan16, "| Q-DA-20 | DA | covers §10.20 | OPEN |\n", "Q-DA")
    ok(r16["ADVISORY_open_rows_citing_closed_items"],
       "an open row under a closed item is REPORTED")
    ok(r16["conforms"],
       "but it is ADVISORY -- an open row may track a successor obligation")
    r16b = check(plan16, "| Q-DA-20 | DA | covers §10.20 | **WITHDRAWN BY DA** |\n", "Q-DA")
    ok(not r16b["ADVISORY_open_rows_citing_closed_items"] and r16b["conforms"],
       "NEGATIVE CONTROL: once the row is withdrawn it conforms")

    # 17. NEGATIONS must not close.  "UNANSWERED" contains "ANSWERED".
    for word in ("**UNANSWERED**", "**UNRULED**",
                 "**NOT WITHDRAWN — the ask stands**",
                 "**NOT CONFIRMED — DA disputes it**"):
        ok(Item("10.9", word).is_open, f"{word!r} must stay OPEN")
    ok(Item("10.9", "**ANSWERED — R-37**").is_open is False, "a real closure closes")

    # 18. ERASURE: a wrapped status row must be joined, not skipped.
    wrapped = (HDR + "| **10.8** | **NEW at iteration 13 — OPEN,\n"
               "  and this row wraps onto a second line** |\n")
    r18 = check(wrapped, "", "Q-DA")
    ok("10.8" in r18["items_open"], "a wrapped status row is parsed, not erased")
    ok("10.8" in r18["ORPHANED_open_items_with_no_register_row"],
       "and still demands a register row")

    # 19. ERASURE: a range key must fail closed, not vanish.
    rng = HDR + "| **10.43-10.44** | **NEW — OPEN** |\n"
    r19 = check(rng, "", "Q-DA")
    ok(r19["MALFORMED_status_keys"], "an unparseable item key is reported")
    ok(not r19["conforms"], "and blocks conformance rather than disappearing")

    print(f"da_escalation_conformance selftests: {checks} checks passed")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", default="plans/SP_PLANE_PLAN.md")
    ap.add_argument("--register", default=str(
        Path(__file__).resolve().parents[2]
        / "orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/COORDINATION.md"))
    ap.add_argument("--namespace", default="Q-DA")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftests())
    rep = check(Path(a.plan).read_text(encoding="utf-8"),
                Path(a.register).read_text(encoding="utf-8"), a.namespace)
    import json
    print(json.dumps(rep, indent=2, sort_keys=True))
    raise SystemExit(0 if rep["conforms"] else 1)


if __name__ == "__main__":
    main()
