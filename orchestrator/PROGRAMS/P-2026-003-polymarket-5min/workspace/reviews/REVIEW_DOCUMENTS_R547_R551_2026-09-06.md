# REVIEW — the document round: R-550(D)'s day sets cannot be COMPUTED from the ledger, DE's design still carries a sentence R-549(A) withdrew, and rule 20 serialises (driven)

**Filed** 2026-09-06T03:58Z (clock read before composing) · reviewer seat
(pm-codex) · **tip `1d66589`**, worktree clean · no code fixed · no write under
`data/` · **nothing sealed opened** — I read day verdicts and RESULTS §3 as
permitted, and no sealed score.

**ROUTING — every finding CHECKED**, read, recomputed or driven by me.

**THREE FINDINGS, plus one correction of my own:**

1. **(A-2) R-550(D) states a day-set rule with a `day-quality` conjunct that has
   no evaluable input for half of either set** — the `da_dayverdict_` series in
   the repository stops at 09-02.
2. **(A-1) DE's design declaration still carries the sentence R-549(A) withdrew** —
   *"sealed and unread"* on days now recorded as opened and consumed.
3. **(B-3) The slice's `MemoryMax` is 14 GiB, not 8 G** — consistent with rule 20,
   but it means the **lock**, not the slice, is what serialises.
4. **My own correction:** my design filing's "≈20 hours of null" repeated DE's
   mislabelled per-arm figure. **The correct figure is ≈9.7 h**, and R-551(2) is
   right.

---

# (A) The coordinator's entries and the three headline amendments

## A.1 R-549(A) is a genuine self-correction, and the artifact it cites says so

**CHECKED at `RESULTS.md:681`, verbatim:**

> *"**09-01 and 09-02 were scored and read under the interim declaration.** They
> are now consumed and cannot be reused as untouched forward validation."*

So R-547(C)'s *"every score is SEALED and unread, so nothing has been chosen on
them"* **was false for two of the five named days**, and R-549(A) says so plainly,
identifies how it happened (*"BE 44's '09-01..09-04 not opened' was true of THIS
session's run and I generalised it"*), and — correctly — **refuses to decide after
seeing** whether "consumed by a read of a different object" counts against a
Gate-1 test of the arms. It puts two options to the USER instead. That is the
right handling of a correction that changes what a run may use.

**And RESULTS.md:685 supports the other half**: *"09-03 remains an accrued race
day, not an opened economic result."*

## A.2 **FINDING — DE's design declaration still carries the withdrawn sentence**

`p003_de_multiday_gate1_design__20260906T031853Z.json`,
`days.nothing_has_been_chosen_on_them`, read just now:

> *"the thetas were fixed on the consumed 08-24 hour and the race scored a
> different object on these days, **sealed and unread**"*

**"sealed and unread" is now known false for 09-01 and 09-02.** The register is
corrected; **the artifact that will be cited is not.** The substantive part of
DE's sentence survives — the thetas were fixed on 08-24 and nothing about them was
chosen on 09-01/02, which R-549(A) affirms — but the phrase overstates it.

**DE 70's v2 must not inherit the sentence**, and v1 stays as provenance (rule 13).
This is exactly the shape of the A-5/A-2 family: a claim that was true of one thing
generalised to another.

## A.3 **FINDING — R-550(D)'s two day sets are stated as computed, but the ledger cannot evaluate the rule**

The declared rule is **four conjuncts ∧ day-quality ∧ NOT previously opened**, and
both sets are described as *"computed and listed in DE's v2"*. I went to the
ledger:

```
da_dayverdict_20260828.json   present
da_dayverdict_20260829.json   present   era_pure=True
da_dayverdict_20260830.json   present   era_pure=False
da_dayverdict_20260901.json   present   era_pure=True
da_dayverdict_20260902.json   present   era_pure=True
20260831, 20260903, 20260904, 20260905  —  NO VERDICT FILE
```

**The series stops at 09-02.** So:

* **Set A** (08-29 + 09-01..09-05) contains **three days with no verdict**
  (09-03/04/05).
* **Set B** (09-03/04/05 + 09-06/07/08) is **six days of which three have no
  verdict and three do not yet exist**.

**Neither set's `day-quality` conjunct can be evaluated today for the days that
matter most.** BE 45's receipts show 09-03 and 09-04 gate-complete and sealed
(12/12), and that is real — but it is evidence of **scoring health**, not the
**day-quality verdict** the rule names, and the two are different objects (the
programme's own cadence note says *"accrual ≠ day quality"*, `SEAT_PROTOCOL.md:138`).

**This must be closed before G is fixed**, because G is the whole question: DE's
own arithmetic makes six the smallest G that clears Holm at m = 2. **A rule that
cannot be evaluated on its own inputs cannot fix G.** Concretely: the day-verdict
producer has to run for 09-03..09-05 (and 09-06+ as they accrue) before either set
is more than a name.

*(For completeness and in its favour: R-550(D) is the correct **rule**. It drops
the version bar my design filing showed was imported — 08-29 returns on quality,
as R-497(F)(1) ruled — and it makes both candidate sets **six days**, so either
branch is significance-bearing. That is the right shape; only its inputs are
missing.)*

## A.4 R-551(2)'s resource correction — **VERIFIED, and I recompute it exactly**

The field name settles it: `be_null_500_draws_one_hour_two_arms.wall_s = 290.9`
— **one hour, BOTH arms**. My arithmetic:

```
null  per day (both arms) : 290.9 x 24 / 3600 = 1.9393 h
replay per day            :  47.0 x 24 / 3600 = 0.3133 h
five days : 9.697 h null + 1.567 h replay = 11.26 h
six  days :                                  13.52 h
```

**Matching R-551(2)'s 1.94 / ≈9.7 / ≈11.3 / ≈13.5 exactly.** And DE's own
declaration text is the source of the error — `resources.per_day_estimate` reads
*"~2 hours of null per **arm-day**"*, which double-counts, because 290.9 s already
covers both arms.

### **And I repeated it.** My design filing said *"2 arms × 5 days ≈ 20 hours of
null"*. **That is wrong by 2×; the figure is ≈9.7 h.** My conclusion — the 8G cap
is protected while the 500-draw minimum is not, so a time overrun must refuse the
day rather than cut draws — **survives at half the magnitude, which makes it less
alarming than I made it sound.** R-551(2) corrects the coordinator and me in the
same stroke.

## A.5 R-551(1) — verified, and the cause is a convention, not an arithmetic slip

`de_multiday_design_declaration.py`:

```
:31   EXPECTED_CHECKS = 20
:518  ok(n[0] + 1 == EXPECTED_CHECKS, ...)          <- the count-assertion itself
:544  payload["battery"] = {... "n_checks": EXPECTED_CHECKS - 1 ...}
```

**The receipt's 19 is `EXPECTED_CHECKS - 1` by construction** — the count *before*
the count-assertion fires. So 19 and 20 are both "right" under different
conventions, and a reader hits the discrepancy with no way to resolve it from the
receipt. **DE 70's fix — carry the produced count AND the source's expected count
with a computed equality — closes the convention rather than patching the
number**, which is the correct direction.

## A.6 The three headline amendments — internally consistent, and one is ahead of the artifact

`HARMFUL_FILL_HAZARD_TOXICITY_PLAN_V2.md:3`, `PROGRAM.md:3` and `RESULTS.md:14`
each carry the same current state (V2 RESUMED, fee blocker dissolved, replay null,
≥5-day run decides §7, design under review, **no data touched**) and each keeps
the superseded 09-05 status **below, labelled as provenance**. No contradiction
among them.

**Notably, `PROGRAM.md:3` already carries R-549(A)'s correction inline** —
*"the forward race has G = 5 with all five days sealed (09-01/09-02 opened earlier
under the interim read)"* — so **the headline is ahead of DE's artifact** (§A.2).
The parenthetical does sit awkwardly beside "all five days sealed"; *sealed* there
means each day carries a sealed score, while *opened* means two were read. The
distinction is real and the sentence would be safer as "all five carry a sealed
score; 09-01/09-02 were additionally opened under the interim read."

## A.7 SEAT_PROTOCOL rule 20 — the repair is complete and the text is executable

After `1d66589`, rule 20 (`SEAT_PROTOCOL.md:124–131`) carries the wrapper line
intact at `:127`, backticks and all. **I ran the exact command uncontended:
`rc = 0`.** The rule is complete as written and executable. Its "heavy" definition
(>60 s wall or >1 GiB RSS) and its two never-clauses are present.

*(One small tension worth naming so no seat reports itself in violation for
obeying it: the lock lives at `/home/yuqing/ctaNew/data/.heavy_run.lock` — it
exists, 0 bytes — and every seat's standing prohibition is "never write under
`data/` except result-bearing artifacts". An ops lock is neither an artifact nor a
violation in spirit; it should be named as an explicit exception in the rule.)*

---

# (B) The five USER review items — verifying the verification

| item | verdict |
|---|---|
| receipt count at `:544` | **CHECKED** — 19 is `EXPECTED_CHECKS - 1`; cause is a convention (§A.5) |
| the 290.9 s label | **CHECKED** — recomputed exactly (§A.4) |
| slice properties now | **CHECKED with one correction** (§B.3) |
| rule 20 executable as written | **CHECKED** — uncontended `rc = 0` (§A.7) |
| does the flock serialise | **CHECKED BY DRIVING IT** (§B.4) |

## B.3 Slice properties — CPU confirmed, memory is NOT 8 G

```
$ systemctl --user show research.slice -p CPUQuotaPerSecUSec -p MemoryMax
CPUQuotaPerSecUSec=2s
MemoryMax=15032385536
```

* **`CPUQuotaPerSecUSec=2s` = 200 %** ✓ — exactly what R-551 claims, so light
  suites can overlap a heavy run.
* **`MemoryMax=15032385536` = 14 GiB**, not 8 G. This is *consistent* with rule 20
  — the 8 G is applied per **scope**, not to the slice — but it has a consequence
  worth stating: **the slice does not enforce one-heavy-run-at-a-time.** Two 8 G
  scopes sum to 16 G against a 14 GiB ceiling, so under contention they would be
  reclaimed or OOM-killed rather than politely refused. **The serialisation
  guarantee rests entirely on the flock**, which is why §B.4 matters.

## B.4 **The flock DOES serialise — driven with two sleeps, not argued**

Running the rule-20 wrapper verbatim, A holding for 5 s and B attempting at
t+1.5 s:

```
B while A holds : rc = 1
A               : rc = 0
uncontended     : rc = 0
```

And flock's conflict exit on this box is **1** (`util-linux 2.39.3`, `-E` default),
so **rule 20's documented "exit 1" is correct as written.**

**A correction of my own measurement, made in the same round:** my first attempt
reported `rc=69`. That was an artifact of my shell construction — a backgrounded
subshell inside a compound `&&` chain — not of the wrapper. **I re-drove it in a
clean script and got 1.** The earlier number should be disregarded; I record it
because a reviewer's wrong measurement that reaches a report is the same defect
class as any other, and this one did not.

---

## CONTEXT

Far below the 80% reset threshold; I will report the crossing when it happens.
**Standing by to file on DE's design v2 and BE's book declaration the moment they
land — nothing runs before that filing.**
