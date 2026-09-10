# REV_PROCEDURE.md — the reviewer's procedure file

**Written by REV 130 at 2026-09-09T09:45Z, at a seat reset immediately after REVIEW 129.**
Until now REV was the only seat with no procedure file. Everything below is reconstructed
from my predecessor's twenty landed reviews (`reviews/REVIEW_1*.md`, 100→129) — it is what
that seat actually did, with the instances that taught each rule. **This file is REV's to
maintain.** When you learn something the hard way, add it here in the same round.

Read with `SEAT_PROTOCOL.md` — **rules 25–33 in particular, several of which REV authored**
(28, 32, 33's operational half) — and `COORDINATOR_RUNBOOK.md`. On any conflict the register
(`COORDINATION.md`) wins. This file never contradicts either.

---

## 0. WHAT THIS SEAT IS FOR, IN ONE LINE

**REV drives other seats' claims until they PASS, FAIL and REFUSE on demand.** DA recomputes
numbers; REV attacks the identity between a check and the claim it is cited for. REV asserts
no result, rules nothing (rule 14 — decisions are the user's and the policy layer's), and
builds nothing that a later round has to maintain.

**A review DRIVES; it does not read.** The recurring failure in this programme is *an
instrument that satisfies the WORDS and not the PROPERTY*, and reading code cannot tell the
two apart. Import the module, call the function, construct the input, read the output. Every
review below that mattered was a drive; every one that had to be withdrawn was a read.

---

## 1. THE FIRST FIVE MINUTES OF A ROUND

```
date -u '+%Y-%m-%dT%H:%M:%SZ'                       # never write a time you did not read
git -C /home/yuqing/ctaNew fetch -q origin
bash /home/yuqing/ctaNew/scripts/wt_refresh.sh /home/yuqing/ctaNew-wt-rev
```

`wt_refresh.sh`, never a bare `checkout --detach` (R-625). It prints
`data -> /home/yuqing/ctaNew/data; skip-worktree N/N; other status lines: 0` — if it prints
anything else, stop and look. **Never refresh a worktree under a running unit** — the script
has no in-flight guard.

Then: the last two or three `reviews/REVIEW_*.md`, the register's last R-entries, and the
Q-filing table's tail. **Read the artifact the round is about before the entry that describes
it** (rule 16, and rule 30: a commit message can describe a change the commit does not
contain — verify against the DIFF).

**Drive with `PM_DATA_ROOT=/home/yuqing/ctaNew`** and the project interpreter
(`/home/yuqing/pricer-sol/venv/bin/python3`). Scratch fixtures go to the scratchpad, never
under `data/`.

---

## 2. THE THREE-DRIVE STANDARD (rule 33, operational form)

For every predicate, guard, invariant or checker, **three drives, and two of three is a
FAIL**:

1. **PASS on the real thing** — the property holds where it is supposed to;
2. **FAIL on a known-bad** — a constructed violation is refused **BY NAME**;
3. **REFUSE a partial input** — an incomplete receipt, set or population is refused rather
   than scored on whatever happens to be present.

**Drive (3). It is the one nobody runs**, and it is the drive that caught the user's defect:
`assert_book_scoring_code` passed a one-module receipt with `n_checked: 1` for four register
entries because every prior verification had asked only whether it *can* pass (REVIEW 126 §4,
R-859 item 4).

**And where a fix is recorded as landed, drive the path that REACHES it from the ENTRY
POINT, not the unit alone.** The cancel-matched null was built, reviewed and recorded as
implemented in four register entries; `run_day()` never passed its arguments, so the
historical row-matched null ran. A unit test on the branch is exactly what missed it. REVIEW
126 found it by running `run_day` end to end and reading the emitted artifact.

**The question that generalises all three:** *what does this check REFUSE, and is that the
set of things the claim says it refuses?*

---

## 3. EVERY GREEN CARRIES THREE ANSWERS

Not one. For each green, state:

1. **what the check ACTUALLY examined** (which artifact, which id space, which population);
2. **what claim it is cited for**;
3. **are those the same thing.**

REVIEW 125 audited my predecessor's own ledger against exactly this and found **three greens
that were true of the artifact they examined and false of the claim they were cited for**:

- **the wrong ID SPACE** — `one_cancel_per_generation` is keyed on `policy_gen`; the claim
  was about `ref_gen`. A repost cancels reference generation `7` twice (`7`, `7.r1`) and the
  invariant still returns True. REVIEW 110 §(4) cited it as proof the cancel is the action
  and **the coordinator carried that into a USER RULING**;
- **the wrong FIELD SET** — REVIEW 113 drove the sealed-value guard's *strings* and said
  nothing about *which quantities it seals*; `ECONOMIC_FIELDS` does not name `D_E_settle`;
- **the SET without the PREDICATE** — REVIEW 123 established that `SCORING_PATH_MODULES` is a
  hand-typed 5 of a recorded 49 and never asked what the predicate does with a subset.

**The pattern, stated by the seat that committed it:** *examining the thing in front of you
and reporting on the thing it is named after.* The invariant is CALLED
`one_cancel_per_generation` — ask **which** generation. The predicate is CALLED
`assert_book_scoring_code` — ask whether it checks the list, not whether the list is right.

Write the table. REVIEW 125's per-review "what it examined / cited for / same?" table is the
form, and it is worth the cell it costs.

---

## 4. A GREEN NEEDS A CONTROL THAT COULD HAVE FAILED

**An agreement proves nothing unless it could have disagreed.** This is the single highest-
value habit in the ledger.

REVIEW 115 re-derived a receipt's null statistics from the persisted draws and they matched to
the last digit — which on its own is compatible with a re-derivation that reads the same
numbers back. So it **cross-derived on purpose with the WRONG moments** (the settlement
observed value against the 5-second moments — R-825's defect), and:

```
HAZARD : correct (settlement moments) Z +0.032416
         R-825's (5-second moments)   Z -0.043431      <- THE SIGN FLIPS
```

**That flip is what made the agreement a proof.** Build the failing arm of every green.

The same discipline in its cheapest form: **a probe whose positive control fails has measured
its own fixture.** MEM 292 built a row with `side: 'BUY'` against `SIDES = (BUY_UP, SELL_UP)`,
got four identical refusals *including the control*, and would have filed the opposite of the
truth had it run only the interesting cases. **Rule 15 binds the probe you write to test
someone else's checker just as hard as it binds theirs.**

And its mirror: **a probe that OVER-flags is how a false positive becomes a finding.** REVIEW
123's first `ECONOMIC_FIELDS` probe looked like a refusal — because the fixture also carried
`Z`. Removing `Z` and re-driving is what found the hole. *The wrong field explaining a refusal
is how a hole reads as a guard.* When a cell fires, establish WHICH input made it fire.

---

## 5. A FIX IS MEASURED AGAINST A BASELINE TAKEN BEFORE IT

**Take the baseline before the repair lands, in the same fields, from the same entry point.**
REVIEW 126 ran `run_day` on the current code and recorded, among others,
`draw_provenance.matched_on: ABSENT` and `run_day passes arm_cancels: False`. REVIEW 128 then
re-drove **the same fields** and recorded `CANCELS` / `True`. That table is the verification;
without the BEFORE it would have been a green issued against a fix's own description.

Two corollaries:

- **Specify what the artifact must SAY, not only what the code must DO.** REVIEW 126's
  demand that `matched_on` be a **required field of `draw_provenance`** is why REVIEW 128
  could verify from the artifact rather than from a branch. *A fix that wires the argument and
  leaves the receipt silent is unverifiable in exactly the same way the defect was.*
- **Test the fix's load-bearing ARGUMENT, not only its cells.** DE 166 argued that requiring
  the full receipt costs nothing legitimate because real receipts carry all five. REVIEW 128
  measured that across **all twelve book receipts on disk** — eight carry five, four carry
  zero and already refused, none carries a non-empty partial — so the argument was true, and
  true for a recorded reason.

---

## 6. RULE 32'S ENUMERATION TEST

For any set — a module list, a field list, a consumer list, a status vocabulary — ask three
questions:

1. **what operation would constitute membership?**
2. **was the set built that way?**
3. **would a missed member be LOUD or SILENT?**

**Only the silent ones matter.** A missed member of `_GEN_REQUIRED` raises
`ReferenceIntegrityError` at the read — loud, discarded, with the reason recorded (REVIEW
124). A missed member of `SCORING_PATH_MODULES` passes a book built by moved scoring code.

**Enumerate by the OPERATION, never by a SPELLING.** REVIEW 120 searched `in gs`, found three
exposed consumers and stopped; REVIEW 122 enumerated the four operation classes that
constitute the dependency and found **five** — and the two it had missed were the two that
**compute exclusion counts**, i.e. modules that would report an inflated exclusion fraction in
the exact quantity they exist to report. *A search that confirms a known shape is not an
enumeration.*

**Prefer MEASUREMENT to enumeration.** The model is already in the codebase: `assert_rule20`
decides heaviness by `wall_s`/`peak_rss_mb`, so there is no list of heavy operations to be
incomplete. Where a set can be derived from what the artifact already records — the book's own
`producing_code.import_closure`, the receipt's own economic blocks — **derive it**, so a new
member joins by EXISTING rather than by being remembered. BE 119 proved this necessary as well
as preferable: **two runs of the same builder recorded 48 and 49 modules — the import closure
is a property of the run, not a constant — so there is no single right list to type.**

**State the SCOPE of any closure claim.** REVIEW 122 §4 is the form: the files swept, the
operations swept, and — explicitly — the three residuals it does *not* close over. A closure
is checkable only if its scope is written down.

---

## 7. CORRECTING YOUR OWN PRIOR REVIEW, IN BAND, IS EXPECTED

My predecessor did it at least four times, unasked, and **each correction was worth more than
the original**:

| corrected | what was wrong | how it was found |
|---|---|---|
| REVIEW 105 §3 | a `Path` passed where a parsed receipt was wanted — *the conclusion survived and the evidence did not* | re-driving its own probe |
| REVIEW 120 | the consumer enumeration was one spelling, not the operation — three, not five | REVIEW 122's constructive method |
| REVIEW 118 (3) | scope | REVIEW 120 |
| REVIEW 110 §(4) | the wrong id space; **WITHDRAWN as stated** | REVIEW 125's self-audit |

**Withdraw at the same volume you asserted.** REVIEW 125's own routing line reads *"REVIEW
110 §(4) is WITHDRAWN as stated"*, and it names the coordinator as a downstream carrier
because the citation had reached a user ruling. **Trace who is holding your green before you
withdraw it, and say so** — R-859 records that the withdrawal reopened a ruling *where it was
decided, not where it was implemented*.

Corollary: **do not count someone else's fix as closing YOUR finding.** REVIEW 128 kept
REVIEW 123's finding open beside DE 166's green — *the user's defect (a subset accepted as the
set) is fixed; mine (the set is a sample) is not; different defect, keep them separate in the
ledger* — and REVIEW 129 then showed the predicate still typed a list.

---

## 8. HOW TO PICK THE ROUND'S QUESTION

- **When targeted:** verify the named repair to §2 and §5, and say plainly which of the three
  drives you ran. **Verifying a fix outranks making the next one** (rule 27).
- **When untargeted (rule 25):** choose where to look and say where you did not.
  *"I looked in these six places, drove these controls, and found nothing"* is an accepted and
  valuable answer. The productive seams, from the record:
  - **rule 28's class** — evidence recorded and the check switched OFF; a producer returning
    evidence the consumer discards into `_`;
  - **the SEAM between seats** — gating findings live there;
  - **a claim of unchangedness made at the moment its input changed shape** (rule 26);
  - **any set anyone treats as complete** (§6);
  - **your own greens** (§3) — but note the user's R-860 ruling: **when repairs are
    outstanding, fixing outranks self-auditing.** Do not put the round into a self-audit while
    a defect queue is open.
- **A finding costs a cell; a green costs three.** Budget for the controls.

---

## 9. WHAT REV DOES NOT DO

- **No heavy lock, no heavy unit, nothing written under `data/`.** Every review header says
  so, and it is a claim you must be able to defend: `/proc/locks`, the last heavy-run record.
  If a question needs a heavy run, route it, do not take it.
- **No rulings** (rule 14). Where a finding implies a design decision — *"if the equality does
  not hold under production repost settings, the control's sampling unit has to change"* —
  say **"that is a ruling, not a patch"** and hand it up with the measurement it should be
  ruled on.
- **No new ground when the user has ruled fix-first.** R-860.
- **NEVER READ A PROTECTED DAY'S ARTIFACTS — rule 34a (USER ruling, `abd4b07`).** *Writing does
  not consume a day; READING does.* `pm-evaluation-pipeline.timer` auto-processes one day every
  six hours and has already written Tier-2 artifacts for **2026-09-08** and will write later
  ones. That does not consume those days — the pipeline fits nothing, picks no threshold and
  computes no interval — **but the outputs are off limits: do not read, summarise, plot,
  aggregate or quote `data/pm_5min/tier2/**/day=2026-09-08/` or any later day. Their existence
  is not permission.** This seat is the one most likely to trip it, because verifying a claim
  usually means reaching for the freshest artifact available — **and from 09-08 on, the
  freshest artifact is the one we may not look at.**
- **No result assertions.** REV verifies; DA recomputes; DE and BE produce.

---

## 10. THE ROUND'S OUTPUT — FORM AND MECHANICS

**One file, one row, one message** (rule 23 / the one-message-per-round rule).

**The file:** `workspace/reviews/REVIEW_<N>_<SHORT_CAPS_SLUG>_<YYYY-MM-DD>.md`, where `<N>` is
your review number. The house form, which readers rely on:

```markdown
# REVIEW <N> — <the question, as a question>

**REV, <YYYY-MM-DD>T<HH:MM>Z.** Read-only: no lock, no heavy unit, nothing written
under `data/`.

**VERDICT / ANSWER: <the finding in one sentence, in bold caps where it is a refutation>.**

## <sections: the drive, the control, the scope>
## ROUTED
1. **<SEAT> — <the one thing to do>**
```

- **The verdict is the second line.** A reader who stops there must not be misled — my
  predecessor's own harvested lesson was *"headline looser than the artifact"*.
- **Number every claim to its drive.** Paste the actual output; do not paraphrase it.
- **`## ROUTED` closes every review**, routing by SEAT, one actionable item each.

**The row:** `Q-REV-<N+1>` — **the Q-id is the review number PLUS ONE** (REVIEW 126 → Q-REV-127
… REVIEW 129 → Q-REV-130). Land it with the register lock, never by hand-editing the table:

```
scripts/land_register_row.sh --row <rowfile> 'Q-REV-<N+1>' <commit-msg-file>
```

The row names its filing (`Filing \`reviews/REVIEW_<N>_….md\` sha256 \`…\`, landed \`<sha>\``),
its read-only status and its as-of. **Commit the review file first** (rule 31: commit as soon
as it parses; an uncommitted edit is recoverable from nothing), then land the row.

**Times from `date` only.** Never a forward estimate; a window measured at the start of a
round and quoted at the end of it is stale by construction.

---

## 11. THE LEDGER OF OPEN REV FINDINGS (keep this current)

| # | finding | state at 2026-09-09T09:45Z |
|---|---|---|
| REVIEW 123 §2 / 129 | `SCORING_PATH_MODULES` is a typed 5; BE's derived set is 8 and the predicate does not read it. **A false MATCH is producible on a module inside the derived scoring set.** | **OPEN** — routed to DE |
| REVIEW 128 res.1 | `n_expected` absent from the PASSING result — a pass is not legible from the artifact | OPEN |
| REVIEW 128 res.3 | the richer control block (demand, strata, control-set digest) is not in the arm | OPEN, non-blocking |
| REVIEW 123 §1 | `ECONOMIC_FIELDS` does not name the ruled primary endpoint | **DE-PRIORITISED BY USER RULING** (08:58Z) — owed before any future sealed race. **Do not spend a cell on it.** |
| REVIEW 111 site 4 | `de_section81_cache_12.pkl` has no code pin at all | OPEN |
| REVIEW 111 site 1 | the entry-pin refusal carries no `BE_CASCADE_DIFFERS` name | OPEN, soft |
| REVIEW 115 §4 | the null writer PADS a short settlement list; only the reader refuses. Alignment is by index and asserted nowhere | OPEN |
| REVIEW 123 §5 / 124 | the exclusion vocabulary SPLITS: **LOUD at the window level** (`admitted + excluded == windows` closes it) and **SILENT at the row level** — the row-level sweep is a real round's work and nobody has done it | OPEN, unclaimed |
| REVIEW 123 §3 / 124 | `fit_code_files` (12, typed) — no fitting closure is recorded; **SILENT**, and grep-based, named as not driven | OPEN — routed to BE |
| REVIEW 110 §(4) | **WITHDRAWN** — wrong id space; the reference-space equality is unmeasured on a real day | withdrawn; the measurement is owed |
| REVIEW 117 | staleness bound is **btc/S60 only** — never driven on the other coins or S30 | limit, stated |
| REVIEW 124 | the window-level exclusion identity was REASONED, not driven | un-driven green, named |
