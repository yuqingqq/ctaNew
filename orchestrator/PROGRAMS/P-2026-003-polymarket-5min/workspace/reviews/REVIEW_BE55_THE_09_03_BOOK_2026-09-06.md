# REVIEW — THE 09-03 BOOK: **GO.** The book stands with check #1 only, and I did not rule that on principle — **I measured the window shut.** `asm` is what DE's R1 requires and set equality is exactly the precondition the null rests on. Three reporting gaps: `state_join_failed` is a register-only number, the round-49 correction is not in the receipt, and the playbook's ONE COMMAND as written REFUSES

**Filed** 2026-09-06T08:15Z (clock read before composing) · reviewer seat (pm-codex)
· tip `4718064` (BE 55 `f301731`, an ancestor) · **no arm scored, no null drawn, no
sealed file opened, no `--open`.** I hashed the book, the tape and the fragment and
drove DE's seam against the real artifacts; **I did not re-stream the 991 MB tape.**

**ROUTING — CHECKED.** Every number below is recomputed or driven by me.

**Rule 20.** The lock was **FREE** throughout (checked 08:11:53Z and 08:15:12Z). My
heaviest steps are light by measurement and I did not take it: the book digest
**0.18 s / 3.8 MB**, the tape digest **0.65 s / 3.8 MB**, the fragment **0.38 s**, DE's
R6 on the real book **0.34 s**, `rehearse_smoke` **0.01 s**. One heavy thing I
deliberately did NOT do: re-drive `build_tape_index` over the tape (§2.3).

## VERDICT — **GO for the sealed smoke on this book.**

**(a) THE BOOK STANDS with check #1 only, disclosed — and the disclosure is already in
the artifact.** `seam.index: "build_tape_index(splits, tape_path=…)"` is recorded in
the receipt, so my REV 45 §1.3 finding is visible to a future reader rather than
hidden. And I did not rule on principle: **both inputs still hash to their pinned
digests after the assembly, with mtimes that predate the run** — the window between
the front-door digest and the stream is *measured* shut, not argued shut. **§2.**
A re-assembly under `inputs=` would buy no evidence this book lacks. **Do NOT spend
35 minutes behind the lock on it**; fix the call form for 09-04 onward.

**(b) `asm` is what R1 requires and set equality is precisely the precondition.**
Verified at both sides of the seam: `be_cancel_axis_null.py:217` builds the decision
population for **both arms** from the **CONDVALUE head's** key set alone — so unequal
sets would evaluate HAZARD on a population defined by the other head's coverage, and
nothing downstream would show it. 297,379 shared keys, `sets_are_equal: true`. The
15,735 uncovered are a **counted status** (5.03%, identical for both heads, and
313,114 − 297,379 = 15,735 checks out) — **but the receipt records the count and not
the reason.** **§3.**

**(c) `state_join_failed: 0 across 42 chunks` IS NOT IN THE RECEIPT.** My scan:
`state_join` ×0, `chunk` ×0. `be_daybook_build` never records it. The claim lives only
in Q-BE-297. It is corroborated indirectly — a wrong tape gives ~0 coverage, and
coverage is 0.9497 — but the direct evidence for the parameterised seam's whole point
is unrecorded. **§4.**

**(d) The round-49 budget correction is right, and the receipt does not carry it.**
3.190 GB is the day's 991 MB tape indexed; 5.971 GB was the live v5 tape. The receipt
has no `5.971`, no `artefact`, no mention of round 49. **§5.** And my REV 43 §2.2
stands: the release is still **measured and never asserted** — it freed 1.156 GB here,
which is exactly why an unasserted check is dangerous.

**(e) DE's rehearsal resolves everything and reads `status: READY`, blocking `[]`** —
driven, with the builder receipt resolved to the file BE actually wrote and R6
admitting end-to-end on the real book. **But the playbook's §2 command, as written,
REFUSES: `--day requires --book`** (driven, rc 1). Use the rehearsal's
`THE_ONE_COMMAND`, not the playbook's. **§6.**

---

# 1. THE BOOK'S IDENTITY — recomputed by me

```
sha256  aad816d637f8445abd53f8e69e44eee058f8d4e196eb9a434da20d0938470323   [0.18 s / 3.8 MB]
bytes   290,758,834
```

Identical to the receipt's `book.sha256` **and** to its independent `readback_sha256`.
The receipt's discipline here is right and worth naming: `digest_is_of_the_buffer_as_
written: true` with the read-back *"a SECOND, independent statement of the same bytes,
reported beside it, not instead"* — the B-1 lesson applied on the **write** side,
which is where it is cheap and total.

---

# 2. (a) THE RULING — CHECK #1 ONLY, AND THE WINDOW IS MEASURED SHUT

## 2.1 What actually happened, from the receipt and the code

```
be_daybook_build.py:396   inp = R.day_assembly_inputs(_hy, tape={path,sha256}, fragment={path,sha256})
                          -> CHECK #1: both digests recomputed at read time, ledger-rooted,
                             regime RULED_DAY_INPUTS_SUPPLIED, is_the_consumed_era_constant false
   :406                   tape = R.build_tape_index(splits, tape_path=inp["tape"]["path"])
                          -> no `day`, no `expect_sha256`: CHECK #2 cannot fire (REV 45 §1.3)
```

**Between them: two statements** — a dict comprehension into `obs` and a dict lookup
for `splits`. No I/O, no subprocess, no yield.

## 2.2 **The measurement that settles it**

Check #2 exists to catch bytes that change between the digest and the use. So I
measured whether they did — at **2026-09-06T08:12:39Z**, after the assembly finished:

| input | mtime | digest now | pinned in the receipt |
|---|---|---|---|
| tape | **2026-09-06 07:01:11** | `9de88da950598e86…` | `9de88da950598e86…` ✓ |
| fragment | **2026-09-06 05:58:23** | `2860832a66e820ca…` | `2860832a66e820ca…` ✓ |

**Both mtimes predate the run** (BE took the lock at 07:29:26Z; wall 2,115.7 s; the
book was written at 08:04) **and both digests still equal their pins.** A rewrite —
even one restoring identical content — moves mtime; identical content that never moved
mtime is the same bytes. **So for THIS book the window was empty, and that is measured
rather than assumed.**

## 2.3 What I did NOT do, and why

I did not re-drive `build_tape_index` over the tape to reproduce the index. That is a
991 MB stream and heavy by rule 20's bar; the lock is free but the run buys nothing the
two digests above do not already give. **Recorded so the absence is a decision, not a
gap.**

## 2.4 The ruling I recommend

**The book STANDS for the sealed smoke, with the disclosure that is already in
`seam.index`.** Three things carry it:

1. the front-door check ran, on both inputs, with the digests recomputed at read time;
2. the window is two statements wide **and measured empty** (§2.2);
3. the receipt **already discloses the weaker call form**, so nothing is being waved
   through quietly — a future reader resolving `seam.index` sees exactly which call
   opened the stream.

**Against re-assembly:** 35 minutes behind the lock would produce a book with the same
digest from the same bytes, and would delay the only Gate-1 day that exists. The
correct place for the fix is **09-04 onward**, where BE 56's `inputs=` form (already in
flight) closes it before the next book is built. If the coordinator wants belt-and-
braces on THIS book, the cheap form is not a rebuild: it is to record the two
after-digests above in the smoke's own receipt as an input-stability statement.

---

# 3. (b) THE POPULATION

## 3.1 The reference, and its statuses

```
247 windows · 247 slugs · 313,114 generations · terminal_marks_present true
ADMITTED 247 · TERMINAL_MARK_OK 247 · TERMINAL_MARK_ENDED_IN_GAP 63 · TRANCHE_KEPT 46,439
TERMINAL_MARK_MISSING 0 · RECONCILIATION_FAILED 0 · BINANCE_GAP_EXCLUDED 0
NO_REPLAY 0 · TRANCHE_NO_MARKOUT 0
```

Every counter is present including the zeros, which is the right shape — a zero that is
printed is a zero that was computed. The 63 `ENDED_IN_GAP` sit **inside** the 247 marks
that are OK, as a sub-status; both are disclosed rather than one masking the other.

**Selection era `clob_v3_1`, with the reason computed and stated:** *"the declared
population intervals end 2026-08-26T00:00, so no September day passes them"* — which
is the same fact BE's own battery check 1 measures, and it is why the day path exists
at all.

## 3.2 **Set equality is exactly what DE's R1 needs — verified at both sides**

Design v12's R1 states the requirement and names where DE checked it. I checked the
same line myself:

```python
be_cancel_axis_null.py:217
    scored = asm["by_arm"][(COIN, ARMS["CONDVALUE_X_SKEW"]["head"])][0]
```

**The decision population for BOTH arms is built from the CONDVALUE head's key set
alone.** So if the two heads had scored different sets, the HAZARD arm would be
evaluated over a population defined by the *other* head's coverage — and no field
anywhere would record it. That is why `sets_are_equal` is not bookkeeping: it is the
precondition the shared draw pool rests on.

```
n_shared_keys 297,379 · set_equality_asserted true · sets_are_equal true
both heads at their PINNED thetas: incumbent_linear_d 0.435259 · q1_arrival_composed_lgbm 0.324506
```

## 3.3 The uncovered are a status — with the count, not the reason

```
n_reference_generations 313,114 · n_covered 297,379 · n_uncovered 15,735 · coverage 0.949747
313,114 − 297,379 = 15,735          (checked)
identical for both heads            (implied by set equality, and stated separately)
```

**A counted status, not a silent drop** — rule 4 satisfied on the count. Two things to
note honestly:

* `assert_coverage` refuses only when coverage is **0 on every head**. 5.03% uncovered
  passes with no bar, which is the right design (there is no declared coverage floor)
  but means the number is reported, never gated.
* **The reason is not in the artifact.** Q-BE-297 says they are *"generations the
  feature pass dropped"*; the receipt records four numbers and no reason class. Rule 4
  asks for exclusions as statuses **with their counts reported beside every table** —
  the count is there; a reader cannot tell a feature-pass drop from a join failure
  from the receipt alone. One field.

---

# 4. (c) `state_join_failed` — **THE NUMBER IS NOT IN THE RECEIPT**

```
scan of be_daybook_receipt_20260903_btc.json:   'state_join' ×0   'chunk' ×0
grep state_join_failed be_daybook_build.py:      no match
```

`state_join_failed` is produced inside `de_phase4_diag_runner`'s assembly and appears
in its docstrings; **`be_daybook_build` never records it**, so BE's *"0 on every one of
the 42 chunks"* is a reading taken from the run's own output at the time and preserved
only in the register entry.

**What it would have caught:** a row whose `(slug, side, gen, t_start)` key found no
entry in the tape index — i.e. the day's rows failing to join the day's tape. That is
precisely the failure the parameterised seam exists to prevent, and it is the failure
mode whose symptom is an **empty or thin `asm`** rather than an error.

**It is corroborated indirectly and I say which way:** a wrong tape would give coverage
near zero; coverage is **0.9497** on 313,114 generations, which no wrong-tape join
produces. So the claim is almost certainly true — and *almost certainly true from a
side-effect* is not the same as recorded. **One field in the receipt**
(`state_join_failed` and `n_chunks`), and the seam's own evidence stops living in a
register entry.

---

# 5. (d) RESOURCES, BUDGETS, AND BE'S CORRECTION OF ITSELF

```
stage             wall_s    peak_gb   current_gb   budget   within
A0_reference       555.9      2.006      2.007       3.0     true
A1_index           118.0      3.190      3.172       6.5     true
A2_assemble      1,420.4      5.317      4.096       7.5     true      <- asm_peak_gb_PUBLISHED
A3_release_index     1.5      5.317      2.940       7.5     true
A4_write_book        6.2      5.317      3.182       7.5     true
total wall 2,115.7 s · peak 5.317 GB of 8 · no budget refused · cap not raised
index_released: 4.096 -> 2.940 VmRSS, freed 1.156 GB, "ru_maxrss cannot show a release"
```

**The instruments are the right ones.** `peak_gb` is the process high-water (correct
for the cap); `current_gb` falls, which is what makes the release visible; and A2's
`current_gb` 4.096 against `peak_gb` 5.317 shows the assembly's transient — the shape
my REV 43 §4.3 asked DE to be able to see.

**The round-49 correction is right.** 3.190 GB is the index built from the day's
991 MB tape; **5.971 GB was the same stage indexing the live v5 tape**, which is a
different and much larger object. So the 8.713 GB resident floor that made BE's
round-49 budget compute to *NO* was an artefact of indexing the wrong tape — the
same defect the parameterised seam fixed. **The arithmetic is consistent with every
other number here** and I have no reason to doubt it.

**But the receipt does not say so.** `'5.971' ×0`, `'artefact' ×0`, `'round 49' ×0`.
The correction — including the fact that a previously published budget verdict is
withdrawn — lives only in Q-BE-297. A reader who finds the round-49 declaration and
this receipt has no way to learn from the artifacts that the first is superseded.
**One field beside `stage_budgets_gb`.**

**And my REV 43 §2.2 is unchanged:** `freed_gb` occurs exactly once in `build()`,
inside the recorded dict, with no `raise` after it. It freed 1.156 GB here — **which
is the argument, not the refutation**: the release passed, and nothing in the code
would have stopped the book being written if it had not.

**The wrapper is measured and my exclusivity fix is live in a result-bearing run:**
`cgroup_leaf be55book.scope`, `exclusive: true`, `flock_modes_on_the_inode: ["WRITE"]`,
`fresh_probe.probes ["LOCK_EX|LOCK_NB", "LOCK_SH|LOCK_NB"]`, holder pid 2996707 present
in `ancestor_pids`, `delegated_to de_multiday_gate1_runner.wrapper_observed`. That is
REV 41 §2.5 closed and then exercised on the real thing.

---

# 6. (e) WHAT THE SMOKE COMMAND RESOLVES — DRIVEN

## 6.1 DE's resolver reaches BE's receipt, and R6 admits on the real book

```
builder_receipt_for(book, "2026-09-03", "btc")  ->  be_daybook_receipt_20260903_btc.json   exists
the naive with_suffix(".json") would be         ->  be_daybook_20260903_btc.json           DOES NOT EXIST

verify_book_against_builder_receipt("2026-09-03", book, receipt)   ADMITS  [0.34 s]
   digest_read_from_field  book.sha256
   receipt_day 20260903 · day_forms_matched ['2026-09-03','20260903']
   digest_recomputed_at_read_time True · digest_source "BE's builder receipt, not a DE constant"

KNOWN-BAD: the same book against 2026-09-04  ->  REFUSES, "BE's receipt is for day '20260903'"
```

Both naming defects DE's rehearsal found are real and both are closed: the receipt is
not the book with a `.json` suffix, and BE stamps the day compact while DE names it
dashed. **Either would have refused a correct book at GO for a reason that had nothing
to do with the book.**

## 6.2 The rehearsal reads READY

```
rehearse_smoke("2026-09-03")  ->  status READY, blocking []          [0.01 s, opens no book, scores no arm]
  P2_book_exists                    HOLDS   blocks_go True
  P2_builder_receipt_exists         HOLDS   blocks_go True
  P3_params    v5  306bfdb0…        HOLDS   blocks_go True
  P3_design    v12 c32c7245… declared == on_disk   HOLDS   blocks_go True
  P4_data_root_is_the_ledger        HOLDS   blocks_go True
  P7_cascade_digest 93332a45…       HOLDS   blocks_go True
  day_is_in_the_ruled_set           HOLDS   blocks_go True
  P5_lock_free_now                  HOLDS   blocks_go False (informational)
```

## 6.3 **FINDING — the playbook's ONE COMMAND refuses as written**

The rehearsal composes the command **with `--book`**. The playbook's §2 does not:

```
$ python3 de_multiday_gate1_runner.py --day 2026-09-03 --output <path>
  -> rc 1   RunnerRefused: REFUSED: --day requires --book
```

**Use the rehearsal's `THE_ONE_COMMAND`, not the playbook's §2 block.** And the
playbook's precondition table is stale in both blocking rows: **P1** still cites REV 38
and says the three wiring items *"need re-driving"* (driven at REV 41 and REV 45);
**P2** still reads *"NOT MET and further away than it looked"* while the book exists and
its receipt verifies. At GO, the playbook is the document a human reads.

*One small thing in the rehearsal itself:* `P5_lock_free_now`'s detail is a static
sentence — *"Held right now by BE 55's assembly, which is the correct state while a
book is being built"* — printed regardless of the measured state. The lock is **free**
and the field still says it is held. `blocks_go` is `false` so nothing turns on it, but
it is the same hardcoded-prose-beside-a-computed-value class as the "23", one line from
being computed.

---

# 7. VERDICT

**GO for the sealed smoke on `be_daybook_20260903_btc.pkl`, digest
`aad816d637f8445a…`, recomputed by me.**

Everything the smoke needs is present and driven: the book verifies against BE's own
receipt through DE's resolver; `asm` carries both pinned heads at their pinned thetas
with equal key sets, which is the precondition the null actually rests on; every stage
stayed inside its declared budget with the peak published; and the wrapper was measured
exclusive.

**On the check-#1 question: the book stands.** Not because one check is enough in
general — it is not, and BE 56's `inputs=` form should land before 09-04 — but because
for **this** book the thing check #2 protects against is measured not to have happened.

**Before GO (minutes, not a rebuild):**

1. Use `rehearse_smoke`'s `THE_ONE_COMMAND`; correct the playbook's §2 and its P1/P2
   rows (§6.3).

**In the smoke's or the next book's receipt, not blocking GO:**

2. `state_join_failed` and `n_chunks` as fields (§4).
3. The uncovered generations' reason class (§3.3).
4. The round-49 budget withdrawal recorded beside `stage_budgets_gb` (§5).
5. The release **asserted**, not only measured (§5, REV 43 §2.2) — before the next
   assembly, because it passed here by luck of arithmetic rather than by a guard.
6. `inputs=` on `build_tape_index` for 09-04 onward (BE 56, in flight).

Nothing here scored an arm, drew a null, opened a sealed file, or took the lock.

---

## CONTEXT

Approximately 72%. Below the reset threshold; I will report the 80% crossing.
