# REVIEW — BE 50: **the race reader must NOT be run.** Its statistic is not the declared estimand, and on the writer's real shape it returns `+1` by construction — a 1e-9 perturbation flips a collapsing day to positive

**Filed** 2026-09-06T06:42Z (clock read before composing) · reviewer seat (pm-codex)
· tip `b474e34` · **no sealed file opened** — every drive below is on synthetic
sealed files I built to the writer's shape, and the shape itself was read at the
WRITER (`be_forward_day.score_rows`), never by opening one · no data written outside
the scratchpad.

**ROUTING — CHECKED.** Everything driven by me.

## VERDICT

**(A) THE READER IS NOT APPROVED.** Nine checks reproduce (0.04 s, 19 MB) and four
of its five properties are right and driven. But two findings block it, and running
it would **consume all five days** — rule 11, and there is no sixth day.

1. **The statistic is not the declared estimand.** The declaration says, in four
   places, `NET CENTS against the INCUMBENT` at the unit of `the ACTION (slug, side,
   gen), de-duplicated`, with `L = 50 ms` and `BY_THRESHOLD` pairing. The reader
   computes the net of **sign-flips between consecutive score values within a coin**.
   No incumbent, no action unit, no latency, no pairing. **§A.2.**
2. **On the writer's real shape the day sign is arithmetically forced to `+1`.**
   Driven: 288 windows × 2,100 rows → `up 604,512 / down 287 / net +604,225`,
   **identical across three independent random draws and identical on a strictly
   collapsing day.** The decisive pair: the same collapsing series gives **−1** when
   within-window values are exactly tied and **+1 when they differ by 1e-9.** **§A.3.**

**(B)** The split assignment: BE routes it as PROVISIONAL and its reasoning is right.
My view by the code — the label is assigned by **input file**, its only behavioural
consequence in the builder is an embargo comparison an empty `score` split makes
undefined (and BE correctly made **no** embargo claim), and it is **not** inert
downstream. **§B.**

**(C)** Two of my three builder findings are closed and driven; **the third is not** —
the worktree selftest now reports SKIPs with reasons, and then fails on its own
count arithmetic. **§C.**

---

# (A) THE RACE READER

## A.1 What is right, and driven

```
be_race_reader.py --selftest   9 checks passed   [0.04 s, 19 MB]
```

Driven end-to-end on **five synthetic sealed files I wrote to the writer's shape**:

* **Both floors, conservative resolved** — `optimistic {G 5, 0.0625}`,
  `pessimistic {G 3, 0.25}`, `resolved_best_possible_adjusted_p 0.25`. **It is the
  0.25 reading**, as declaration v2 requires, and `max(o, p)` makes it structural
  rather than a choice. ✓
* **The byte-identity void is a code path, not an instruction** — my mutated-between-
  passes case raises `ReadVoid` and **no result is emitted**. The declaration's
  `REQUIRED_AFTER_THE_READ` now has code to run in — my REVIEW_BE48 §C.2 residual. ✓
* **The separation is checked against the paths this run OPENED** —
  `haystack: "the paths THIS RUN opened, not a module constant"`, and a planted
  Gate-1 path refuses. **My §C.3 finding — constant-vs-constant — is closed.** ✓
* **It writes the ONE declared artifact and nothing else.** Driven: the outdir
  contains exactly `be_race_read_result_v1.json`; the directory holding the sealed
  inputs contains **only the five inputs, unchanged**. ✓
* **No path writes under the run dirs.** `read()` writes to `outdir` or
  `_BDR.derived()` — the resolver's path, not the code tree. `ROOT = HERE.parents[1]`
  is defined at `:49` and **never used** (full grep: one line, the definition). ✓
* The shape guard refuses a row that is not `[t0, value]`, and a day with no
  increment is a **STATUS**, never a zero entering the sign count. ✓

## A.2 **FINDING — the statistic is not the estimand the declaration commits to**

`be_race_read_declaration_v2.json` states the estimand four times over:

```
per_day_quantity          : "NET CENTS against the INCUMBENT"
per_day_unit_of_analysis  : "the ACTION (slug, side, gen), de-duplicated"
estimand.comparator       : "the frozen per-coin incumbent, not a base rate"
estimand.latency_axis     : "only tranches after t + L are valued, L = 50 ms"
estimand.tranche_valuation: "each tranche at ITS OWN time and level (rule 3)"
estimand.pairing_convention: "BY_THRESHOLD"     inherited_verbatim_from: be_read_declaration.estimand()
```

`day_statistic()` computes: sort a coin's `(t0, value)` rows, difference consecutive
pairs, count `up − down`, take the sign. **There is no incumbent in it, no action
key, no latency, no tranche and no pairing convention** — and there could not be:
full grep over `be_race_reader.py` for `net cents|incumbent|estimand|action|latency|
BY_THRESHOLD` returns **0 matches**.

**And the sealed file cannot carry the declared estimand — by BE's own account.** I
established the shape at the WRITER in REVIEW_BE48 §A.4: `be_forward_day.seal()`
writes `per_coin_scores` from `score_rows`, whose one appending line is

```python
out[coin].append((int(r["t0"]), FS.expected_cancel_value(fit, fp + ff)))
```

— a value keyed on the **window** `t0`, with no slug, side or gen. The `FeedWriter`
docstring in the same module names this as the gap it exists for: *"a scored forward
day emitted `(window_start, value)` pairs and nothing the action-level estimand could
consume."* The file that **does** carry `slug, side, gen, t0, t_start` and the
latency-resolved value is `SEALED_feed_<DAY>.jsonl` — which the declaration
deliberately leaves **sealed**.

**So the declaration commits to an estimand, and the reader opens the one file that
cannot express it, and substitutes a different statistic without saying so.** No
field in the reader or in v2 records the substitution.

## A.3 **FINDING — and on the writer's real shape the statistic is degenerate**

`per_coin_scores[coin]` holds **one tuple per ACTION**, each stamped with its window
`t0` — hundreds of thousands per day (the 09-01 receipt records 610,064 btc actions
over 288 windows, ≈2,100 rows per window). `vals.sort()` orders by `(t0, value)`, so
**within a window consecutive values are non-decreasing** and every within-window
increment is counted `up`.

**Driven, on random data:**

```
rows/window   n_rows     up        down    net        sign
      1          288      136       151      -15       -1     <- the selftest's shape
      1          288      144       143       +1       +1        (varies with the data)
     10        2,880    2,592       287    2,305       +1
     10        2,880    2,592       287    2,305       +1     <- IDENTICAL, another draw
  2,100      604,800  604,512       287  604,225       +1
  2,100      604,800  604,512       287  604,225       +1     <- IDENTICAL, another draw
```

**At the real shape the numbers do not move with the data at all.** The arithmetic:
288 × 2,099 = **604,512** within-window pairs (all `up`) against **287**
between-window pairs (the only ones that can be negative), so `net ≥ 604,225 > 0`
on any input.

**The decisive test — the same strictly collapsing day, three ways:**

```
window series falls by 1,000 every window in ALL THREE rows
  within-window values exactly TIED      up 0        down 287  flat 604,512  net   -287  sign -1
  distinct by 1e-9                       up 604,512  down 287  flat 0        net +604,225 sign +1
  a whisker of noise                     up 604,512  down 287  flat 0        net +604,225 sign +1
```

**A 1e-9 perturbation flips a collapsing day from −1 to +1.** Real scores are
continuous model outputs; they will be distinct.

**End-to-end on five synthetic sealed files of the real shape, random values:**

```
day_signs {20260901: 1, 20260902: 1, 20260903: 1, 20260904: 1, 20260905: 1}
n_positive 5   n_negative 0
```

**Five-day unanimity, produced by the sort.** And unanimity is exactly what the
permutation floor would be applied to.

**Why the battery cannot see it.** Its controls are *"three rising windows"* and
*"three falling windows"* — **one row per window**, the single shape where the
statistic tracks the data. The real shape is the interior case, and it is untested.
Same lesson as the queue-model orientation two days ago: the extreme case
discriminates, the case the data is actually in does not.

## A.4 A third, smaller finding — the byte-identity check does not digest the bytes it parsed

`read()` touches each file **three times**: `_sha(p)` → `json.loads(Path(p).read_text())`
→ `_sha(p)`. The statistic comes from the middle read, which **neither digest
covers**. That is BE's own B-1 lesson, from `be_cancel_axis_null.load()`: *"the digest
must be of THE BYTES THAT WERE UNPICKLED, not of a second read of the same path. Two
reads can differ — a writer mid-flight, a symlink repointed, a filesystem that
lies."* One line: read once into a buffer, hash **that** buffer, parse **that**
buffer, then re-read for the `after` digest.

## A.5 Verdict on the reader

**NOT APPROVED for the coordinator to run.** The opening is the coordinator's act on
GO, and on GO it consumes 09-01..09-05 irreversibly. On the evidence above the read
would return five `+1` signs produced by a sort, under a floor of 0.25, and would be
reported as the race's direction. **§A.2 and §A.3 must both close first**, and §A.3
is the one that cannot be argued about: it is arithmetic.

**What closing them looks like.** Either the estimand is computed from the file that
carries it — which means opening `SEALED_feed_<DAY>.jsonl` and amending the
declaration's `STAYS_SEALED` clause, a USER-facing change — or the substituted
statistic is declared, justified against R-529(A), and **aggregated to the window
before the sign is taken** so that the unit is the window and not the row. Either
way the fix is a declaration act before it is a code act, and the days stay unread
until it lands.

---

# (B) THE 09-03 STATE TAPE AND THE SPLIT ASSIGNMENT

**The build is sound and its guard is real:** `refuses_any_existing_path: true`,
`required_stem: phase2_state_tape_gate1_`, four existing state tapes enumerated as
refused. 544,286 rows, sha `7206101d…`, 4.74 GB peak of 8, 1,475.8 s. **Both missing
inputs from my REVIEW_BE48 §A.3 now exist for one day.**

**BE already routes the split question rather than absorbing it** —
`THE_SPLIT_QUESTION_IS_NOT_MINE`, `status: PROVISIONAL`, routed to DE — and its
reasoning is the right one: *"those are two POPULATIONS (eraB, top-up), not two
feature sets."* **My view by the code, which is what was asked:**

1. **The label is assigned by INPUT FILE, not by any fit-time role.**
   `build_state_tape_v2.py:426` writes `"split": split` inside the loop over the two
   inputs, which the builder maps `(('train', FRAG), ('score', TOPUP))`. With one
   population the assignment is **arbitrary**: the day's fragment went to `train`
   because it was passed first, and the opposite assignment would be equally
   defensible.
2. **Its only behavioural consequence inside the builder is the EMBARGO pair.**
   `:447` — `if split == "train": tr_last_exit = max(...)` else `sc_first_feat =
   min(...)`. With an **empty** score split that comparison has no right-hand side.
   **BE made no embargo claim: zero `embargo` fields anywhere in the receipt** —
   which is the honest handling, and exactly what the comment twenty lines below
   warns against (*"a header claiming 'embargo CERTIFIED' over no data"*).
3. **It is NOT inert downstream.** `de_phase4_diag_runner.generation_scores` carries
   `split_of` per generation under R-496(E)'s *"splits labelled per cell"*. Every
   09-03 cell would be labelled **`train`** — the name of the partition a fit was
   trained on — for a **post-freeze forward day scored by frozen heads**, where
   nothing is fitted and every row is a row the pinned heads SCORE.

**So: the assignment changes no number** (`build_tape_index` unions both indices and
the join is by key; `assemble_streaming`'s split partition merely makes one pass
empty) **and it does mislabel every cell.** If one of the two existing names must be
used, `score` is the honest one. Better is a third declared value for post-freeze
forward rows, or a receipt field saying the `train` label is a mechanical artifact of
the builder's input order and carries no fit-time meaning. **Cheap either way — BE
records the rebuild as "one day and cheap relative to the fragment."**

---

# (C) MY THREE BUILDER FINDINGS — two closed, one not

## C.1 The tautological known-bad — **CLOSED, and driven from my worktree**

```
PASS: KNOWN-BAD, AND IT NOW REACHES THIS MODULE'S OWN REFUSAL: a supply carrying
      eth windows and NO btc raises BookRefused here, not an upstream refusal --
      the previous check was carried by `isinstance(e, Exception)` and tested
      be_forward_day instead
```

**The refusal at `day_slugs`, which had zero driven coverage, now has a case that
reaches it**, and the label states the defect it replaces. ✓

## C.2 The two roots — **CLOSED, and the new audit answers the blind spot I filed**

`audit_literals()` now reports **clean, 0 offenders** (it was 4). And the new
`audit_derived_roots()` is the positive-form audit my §B.4 asked for:

> *"a MODULE-LEVEL assignment whose target names a data location and whose value does
> NOT come from the resolver — **the ACT, not the spelling**."*

`n_offending 0`, with **one declared exemption**, visible rather than silent:
`be_forward_preflight.py:33 LOCAL_DERIVED` — *"the mirror TARGET must be the local
tree by definition."* And an intentional second root must carry
`be_data_root: allow-second-root <reason>` on its own line.

**What its own first version missed, in the field's own words:** *"the first version
looked for `parents` and missed `DERIVED = ROOT / 'data/...'`, which is the same
defect one variable removed."* **That is precisely the blind spot I filed** — the
audit that greps for the spelling cannot see the act — found by BE against its own
instrument, one variable further along than I had looked.

## C.3 **The worktree-drivable selftest — NOT closed**

Driven from `~/ctaNew-wt-rev`:

```
PASS  (population intervals)
SKIP  the day supply returns 247 btc slugs        [mask not present ...]
SKIP  the selector returns entries in the 5-tuple shape
PASS  KNOWN-BAD (the fixed one, C.1)
SKIP  the upstream ForwardDayRefused case
3 check(s) SKIPPED — real ledger data not reachable from this tree (BE48 §B.5).
FAIL: ran 2 checks, expected 6 (EXPECTED_CHECKS=9 minus 3 skipped)
```

**The half that matters is right** — the real-data checks are now statuses with
their reason (rule 4), not a hard refusal at check 1. **The half that was the point
is not:** the count reconciliation is wrong (2 ran, 3 skipped, **4 unaccounted** of
9), so the module still **exits non-zero from exactly the tree the fix was written
for.** A reviewer in an R-397 worktree still cannot get a green battery, now for an
arithmetic reason instead of a refusal. One line: make the expected count
`EXPECTED_CHECKS − n_skipped` agree with what actually ran, or count the
fixture-driven checks explicitly.

---

# WHAT I WOULD DO NEXT, IN ORDER

1. **Hold the race read.** §A.3 is arithmetic and does not need adjudication; §A.2 is
   a declaration question and does. Neither is expensive; both are irreversible if
   skipped.
2. **The reader's statistic**, per §A.5 — aggregate to the window before the sign, or
   open the feed and amend the declaration. A USER-facing choice either way.
3. **The B-1 single-read fix** in `read()` (§A.4), which is one line and closes the
   only remaining gap in a safeguard that is otherwise now real.
4. **The split label** (§B) — DE's ruling, cheap to act on, and it should land before
   the assembly rather than after.
5. **The selftest count** (§C.3).

---

## CONTEXT

**I have crossed 80% — I estimate ~82%, and I am reporting it as this round's
filing lands, per the standing instruction.** This filing is complete; I have taken
nothing further. **Recommend a seat reset before the next round.** My open findings,
conventions and intended attacks are the ones R-572(C) records, plus: the ordering
predicate (REVIEW_DA61 §A.5), the legacy-stamped population (§B.3 there), the three
`--day` wiring items (REVIEW_DAY_PATH_DE78 §5), and the two reader findings above.
