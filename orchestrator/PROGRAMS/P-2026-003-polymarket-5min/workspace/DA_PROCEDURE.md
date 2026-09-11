# DA seat — the reader's procedure and standards

**Harvested verbatim in substance from the DA seat's stop answer at 2026-09-07T20:11Z,
before its context was cleared (R-820). None of it was in a file. DA maintains this file
from here on — it is DA's, not the coordinator's.**

## What this seat is for

DA reads what other seats produce and RECOMPUTES it independently. A number DA prints has
been re-derived by a second implementation, or it is labelled as read. DA interprets
nothing and asserts no result.

## Driving both modes

- **Mode one — the per-day read** (`verify` → `print_table`): the arm table, the ledger
  recompute verdict, the settlement block, POPULATION / FINALITY / PLACEMENT, and the
  materiality refusal.
- **Mode two — the four-day table** (`four_day_table`): resolves each day's head and
  prints the family.
- R-764 is orthogonal to both.
- **Both modes must run with `PM_DATA_ROOT=/home/yuqing/ctaNew`.** From a worktree without
  it, `da_gate1_day_verdict` and `da_process_budget_audit` go red BY DESIGN (they check
  they are reading the ledger tree). **A correct gate run is 11–12 green / 2 declared-red,
  rc 0** — do not read a correct gate as a failure.

## Head resolution (DA 130)

`SUPERSEDES_FIELD = "supersedes"`, `SUPERSEDES_PAIR_KEYS = ("path", "sha256")`. A day's
head is the one artifact no other artifact supersedes. **Supersession is honoured only by
the PAIR**: the named path must exist and hash to the named sha, and the sha must be 64
lowercase hex, else `SUPERSESSION_PAIR_MISMATCH`. Two unchained artifacts →
`EARLY_READ_HEAD_AMBIGUOUS`; none → `EARLY_READ_HEAD_ABSENT`. **Why pair-only:** this
mirrors R-729 / REVIEW 86 §8 — history is resolved by the pair the act recorded, never by
today's head. (That reason was written nowhere; it is written here now.) 09-03 now has two
artifacts and resolves through a chain.

## Reading the settlement rows (DA 131/132)

`SETTLEMENT_SCALARS` carries the per-arm `D_E_settle` and its legs; `SETTLEMENT_SLUG`
carries the per-slug values. `SETTLE_RECONCILE_TOL = 1e-9`;
`WINNER_STATUS_REQUIRED = "VERIFIED_AGREE"`. **`D_E_settle` is PRIMARY; the 5-second
markout `D_E0` is DIAGNOSTIC and is labelled so on every line.** DA 131's filed finding:
the two new row kinds arrived under an UNCHANGED `schema_version` 2, and a pre-DE-136 v2
reader **silently skips them** and reports a complete-looking day that omits the primary
endpoint. (The schema is v3 as of DE 139.)

## The standard this seat paid for

**A guard keyed on an optional field is SILENT, not green, and must report its own
denominator.** `SETTLEMENT_WINNER_NOT_VERIFIED` refuses a `SETTLEMENT_SLUG` row whose
status is not `VERIFIED_AGREE` — but **no v3 slug row carries a status field at all (0 of
492)**, so the check had nothing to fire on, and DA's first print of 09-03 said "every
slug's winner is VERIFIED_AGREE" **on a day with one DISAGREE**. The fix counts
`n_slug_rows_carrying_a_status` beside `n_slug_rows`, states that the per-slug
verification lives in the RECEIPT, and prints the receipt's own counts. Rule 15 in its
sharpest form.

## What DA 133 (09-04) paid for — read this before the next day

**The settlement null's MOMENTS are READ and cannot be re-derived by anyone.** The
artifact summarises it as `{"n": 500}`; the ledger's 1,000 NULL_DRAW rows carry
`['arm','cancels','i','row','value']` and their mean and sd reproduce the **5-SECOND
MARKOUT** null to the last digit on both days read so far — they match the settlement
null on neither. The settlement-valued draws reach no file. So: recompute `D_E_settle`
from the FILL rows (a real second implementation), recompute `Z` from the published
moments, and label the moments READ. **Measure which null the persisted draws carry;
never assume it** — the reader does this now and prints it per arm.

**Four shapes this seat's own instrument wore, all found by one day (fixed `70bd9f8`).
Every one is a literal or a guard that had to track something moving:**

1. **The PRIMARY endpoint had no test beside it.** The settlement block printed
   `D_E_settle` and its legs and nothing else while the DIAGNOSTIC's Z and p printed
   in full three lines above. R-819's 09-03 statistics were lifted from the artifact
   BY HAND and labelled "the artifact's own statistics" — the instrument never had
   them, and nobody noticed for a whole day's read. **When the estimand changes, audit
   what the PRINT carries, not only what the recompute covers.**
2. **The denominator line reported half its denominator.** `n_slug_rows` counts ONE
   arm's two books; the file holds that per arm. It said 576 of a 1,152-row file, and
   492 of 09-03's 984 — which is what Q-DA-357 carries. This is the line built at DA
   132 *to report a denominator*. The standard bites its own instrument first.
3. **A typed literal in a printed clause.** "not over a 288-window day", written on
   09-03 at 246, printed verbatim on a 288-window day beside the two numbers that
   contradicted it. Derive it (`WINDOW_SECONDS`, checked against the `window_s` the
   artifacts pin) and name the shortfall.
4. **A battery cell asserting a CENSUS instead of a property.** The four-days cell
   named 09-03 as the only chain and went red the moment 09-04's re-run landed — it
   was measuring the tree's history. Assert the property (one head per day,
   `n_artifacts == links + 1`, `sole` agreeing with its own count); the known-bad (two
   artifacts, no `supersedes` → `EARLY_READ_HEAD_AMBIGUOUS`) is what keeps it able to
   fire. **R3/R4 will make 09-05 and 09-06 chains too — a census would have gone red
   twice more.**

Also from that round: `settlement_statistics` crashed where no ledger exists
(`recompute` is a status STRING there) and, once guarded, would have reported
`matches_the_5s_markout_null: False` for a null it never measured — **absence gets its
own name** (`NO_LEDGER_DRAWS_TO_MEASURE`), never a False that reads as a mismatch. And
a bare `except: pass` in a new cell swallowed an `AttributeError` and handed back an
empty set that read as "the artifacts pin nothing".

**Finality on 09-04, and the sentence to keep saying.** 09-04 is the first day whose
`is_final_for_quotation` reads **True** (288/288 VERIFIED_AGREE). `f_provenance` is
still a **RECORDED BOOLEAN**: a bare `files` key appears **0 times** in the artifact —
check the KEY, not the substring, because `stream_files_digest` contains the word. The
check ran in the producer and its verdict is recorded; it cannot be re-run from the
artifact. R-818 allows that for DESIGN data only, so **`is_final_for_quotation: True`
is not R-818's "quotable as final"** until DE 142 lands.

**Populations so far: 09-03 = 246 slugs (42 of its 288 windows absent), 09-04 = 288
(full).** Never one column.

## A sibling key nobody surfaces is not an answer (DA 150)

BE 116 put `BINANCE_GAP_EXCLUDED_STATUS` beside the count rather than in it —
correct, because a string in the count slot raises `TypeError` in this file's sum
(driven here, not taken from BE's receipt). **But my own verifier published only the
SUM**: its block mentioned neither the count nor the status, so a reader met the zero
folded inside `admitted_plus_excluded` and was never led to the sibling. **DA 147's
question was still open on MY surface.**

**The general form: when another seat adds a field to answer a finding of yours, ask
whether YOUR output leads a reader to it.** A producer-side fix and a reader-side fix
are different fixes. Every exclusion summand now carries its count, its sibling, and
what the number means — and a zero with **no** sibling is named `NOT COMPUTABLE`,
which is what all twelve landed receipts carry.

## Nothing found: the seed and the draw pool (DA 150) — do not re-till

Rule 10's "the seed must pin the data, not just the RNG", driven both directions.
`seed_for(book_sha, arm)` pins the book's bytes; the pool is derived from them by
`be_cancel_axis_null.load()`, whose logic changed at BE 107 — **but
`verify_draw_provenance` binds `module_sha256` to that very module**, and all five
bindings (module, book, seed, arm, block-present) **refuse alone** with the positive
control admitting. `load()` reaches `de_phase4_diag_runner`, one of the ten modules
params pins under `be_cascade.modules`. **The closure is pinned, just elsewhere. No
gap.** Not driven: `draw_null` generating draws.

## Times: I committed the fourth instance of the placeholder slip (DA 150)

I wrote `as-of 09:00:0xZ` in a row header — an estimate with a placeholder digit —
while holding an `08:58:17Z` reading I had already taken. **The rule is not "estimate
carefully": read the clock in a separate call BEFORE composing, and quote that**
(rule 12). A placeholder digit is the tell. Corrected in band at Q-DA-374.

## Enumerate by OPERATION, never by SPELLING (DA 149) — I got this wrong

DA 148 closed DA's consumer set at **three**. It is **four**. `da_de53_exclusion`
spells the test `key = (slug, side, float(g["t0"]))` then `key in gen_scores`, which
my regex — which required the tuple inline — could not see. **It appeared in my own
sweep output marked `['-']` and I read "my pattern did not match" as "this module is
clean."**

REV 122 enumerated by **operation class** (a start-keyed lookup, `len(assembly)` as a
generation count, iterating the assembly as generations, reading the key's third
element as a start time). That finds modules a spelling search cannot. **A grep is a
sample. Say "I searched this spelling" and never "the set is closed."**

**And the module a spelling search misses is likely to be the worse one.**
`da_de53_exclusion` publishes `n_excluded` / `excluded_fraction` and feeds the split
to a 400-permutation audit: on a per-row book it would not mis-count, it would run a
statistical test on a population that is an artefact of its own membership test.

**A classifier needs MEMBERSHIP, not a stream.** `scored_stream_rows` returns
`covered_generation_keys` for exactly that; do not build a second traversal.

## Gate item 4 is CLOSED and verified (DA 149) — both halves

DE 164 routes by `gen`. Verified on my own DA 143 constructions: the event labelled
gen 0 at `t1_0 = t0_1` gives **0 cancels and `crossings_for_another_generation: 1`**
(counted, not dropped); a genuine gen-1 event at that instant **still** cancels gen 1
(narrows the misrouted event only); `validate_scores` now **requires** `gen`. And the
half I found by driving rather than naming — `_head_scorer` serving one generation's
score to another's event — now refuses **`SCORE_BELONGS_TO_ANOTHER_GENERATION`**.
Rule 27: a correctly-labelled event timed after its generation ended is **not**
cancelled retroactively. No regression.

## Uncommitted work in the SHARED TREE is not safe (DA 148) — the one that cost a round

I completed DA 148 once — builder, three wirings, five cells, battery green at 38 —
then found all four files **clean at HEAD with every edit gone**. The reflog showed
no revert, which is consistent: **`git checkout -- <path>` does not move HEAD and
leaves no reflog entry.** Six other-seat commits landed in that window. R-557
forbids that command in the shared tree precisely because seats keep uncommitted
work there; something did it anyway.

**My share, and the rule that follows: I held four edited files uncommitted while a
battery ran.** Test-then-land is right, but the gap between them is the exposure.

> **On the shared tree, commit each edited file as soon as it parses, and let the
> battery gate the PUSH, not the commit.** A local commit is recoverable from the
> reflog; an uncommitted edit is not recoverable from anything.

## The score-event stream: three DA probes had the shape bug too (DA 148)

`da_elementwise`, `da_elem_grid`, `da_elementwise_hz` each built their event rows
inline with a PER_GENERATION membership test **and stamped every event at the
generation's start** — so on a corrected book they would drop most of the
population *and* re-create DE 155's look-ahead inside DA's own code. Fixed through
one shared builder, `da_book_verify.scored_stream_rows`, imported by all three.

**The fix is not conditional on anyone's first-row claim (REV 121):** even under
PER_GENERATION, if the feature pass dropped the row at the generation's start, the
key is absent and the generation reads UNSCORED though its other rows scored.

Standing cells for this class: (a) PER_GENERATION byte-for-byte as before,
(b) PER_ROW one event per row at its own time **with the old expression computed on
the same map as the known-bad**, (c) REV 121's dropped-start-row case,
(d) **a genuinely under-covered map still reported under-covered**, (e) rule 17
wiring checked at the **comment-stripped** bytes, so a comment quoting the old form
does not read as wiring.

**Enumerations of "which modules have this" are samples until swept.** REV 122
said five where the brief said three; my own regex sweep over comment-stripped
`live/` independently surfaced `de_section81_arms.py:526` — same test, same
stamping, and it *counts* the misses, so it would publish a bogus exclusion
fraction. DE's module: report, never edit (R-235).

## Verify the COMMIT, not the commit message (DA 146)

DE 162's message stated mode C's message no longer blames the feature pass.
**`git show <sha> | grep -c "<the phrase>"` returned 0** — the phrase is neither
added nor removed in that commit — and `git log <sha>..HEAD -S "<phrase>"` was
empty, so no later commit touched it. The old text is live, and I reproduced it
three ways at the tip.

**Make this a standing step when verifying any fix: grep the commit's own diff for
the string the message claims to have changed.** A commit message is a claim like
any other; a green battery does not test it, and a careful reader agrees with it.
Cost: one command.

## "Can it PASS?" is the question that separates a guard from a nuisance

`BOOK_BUILT_BY_DIFFERENT_SCORING_CODE` refuses all twelve books on disk. That alone
is indistinguishable from a predicate that always refuses. **Construct the input
that SHOULD pass** — a receipt recording the current sha256 of every
`SCORING_PATH_MODULES` entry — and confirm `BOOK_SCORING_CODE_MATCHES`. Only then
do the refusals mean something about the books.

**And when rule 27 is vacuous, say so and check robustness instead.** The predicate
was new, so no old behaviour could regress. Probing the comparison
(`actual.startswith(str(declared)[:16])`) instead found that an **empty declared
digest is admitted** — `"".startswith` is always True — absence reading as a pass
inside a guard whose docstring forbids exactly that. Low reachability; state it.

## The era fix is verified (DA 145) — gate item 1 closed

`be_daybook_build.day_selector("20260903","btc")` runs in **1.9 s** read-only and
returns 247 entries, **160 gap-bearing, 2,294.7 s** — REV 112's numbers to the
decimal, and the population Q-DA-364 priced at 60.3 % of the day's settled money.
`sel.era = clob_v4_1` while `fi.ERA` stays `clob_v3_1`. **Re-run that one call after
any change to the selector; it is the cheapest end-to-end check the seat has.**

Verified and not to be re-audited: `be_era_for_day.resolve` has one return and six
raises, no default (its only `fi.ERA` mention is inside the returned *evidence*
dict); resolution is day-dependent by measurement; six of twenty-one days refuse
and all six are correct; every day the programme builds admits; the mask closes
287 − 40 = 247 with a both-directions known-bad.

**Known bound, disclosed by the artifact itself** (`era_span_open_ended: true`): the
last era span is open, so a day *after* it admits rather than refuses. Unreachable
through the builder — `day_slugs` refuses a day the ledger has no window for.

**Sweep for the site nobody named (REV 111's shape).** After a two-site fix, grep
the whole tree for the old form. I found a third live call
(`de_v2_local_selector.py:151`) and established it **correct** — smoke-only, bound
to `population == "v3_4_consumed_fragment"`, where the module literal *is* the right
era. **Record a checked-and-correct third site, or the next seat re-finds it and
reads it as a miss.**

## My own verifier had the shape bug too (DA 144) — and what it teaches

`da_book_verify` asserted `n_scored_keys == n_covered` and recomputed coverage as
`len(keys)/n_generations`. **Both hold only under PER_GENERATION.** On a per-row
book the key count is a ROW count — 1,080 rows against 360 generations, coverage
2.7 — so this seat would have raised a **false population alarm on every EV20
book**, on the first one built. BE found it in my module and correctly did not
touch it (R-235).

**The fixture is why it survived: it built a LIST of string keys** — neither real
shape — so no cell of mine could ever have exercised the per-row case. DA 125's
lesson, in my own module. **A fixture that does not reproduce the producer's shape
proves nothing about the producer.** It now emits `(slug, side, t)` triples with
bare floats or dicts carrying `gen`, and `rows_per_gen` makes rows outnumber
generations so a wrong divisor gives a wrong answer instead of an accidentally
right one.

**Condition the predicate; never drop it.** PER_GENERATION keeps the strict
equality; PER_ROW gets `n_scored_keys >= n_covered` (rows can never be fewer than
the generations they cover); an undetermined shape gets the weak form **reported as
the weak form**, never as a pass.

**Resolve the shape yourself where you can (R-235).** The book tier holds the
pickle, so it derives the shape from the assembly's own values and *compares* it to
the receipt's declared field — a receipt declaring the wrong shape is visible
rather than believed. The receipt-only tier has no values: it takes the declared
field, else infers per-generation **with the reason recorded**.

**The third cell is the one that matters.** Removing a false alarm is how a real
alarm gets through. Always drive: a book that genuinely does not cover what its
receipt claims — 60 generations dropped from the assembly and not the receipt —
and confirm it is STILL flagged under BOTH shapes. And make the known-bad the
*old expression computed on the same book*, so the green is a measurement of the
fix rather than of an easy input.

**Operational note:** checkers that resolve "the newest params present" change
verdict the moment anyone lands a params version. `da_gate1_day_verdict` and
`da_accrual_report` went red on `params_v25` landing mid-round. **Every params
landing needs its DA-side re-run in the same round** — and when the gate refuses,
prove the reds are outside your import closure by AST before landing, and say so.

## The abutting boundary (DA 143) — the open edge of 361/362

**Status: 361 CLOSED, 362 OPEN.** DE 162 bounded score times by the generation's
`t1`. It excludes `t > t1 + 1e-9`, so **`t == t1` is admitted**, and
`validate_reference` refuses only `t0 < prev_t1 - EPS` — **strict** overlap — so
**abutting generations are legal**. `harmful_stateful_policy` calls abutting the
normal case in its own comments and names a real instance (2026-09-01,
`btc-updown-5m-1787580000/BUY_UP/gen 449`, a tranche at exactly its `t1` == the
next `t0`).

At that shared instant, three things happen and none of them is visible:
1. a gen-N score event is admitted;
2. the engine routes by time, and GEN_START outranks GEN_END, so **N+1 is live** —
   the cancel records `ref_gen: N+1`;
3. `score_events_for` also emits N+1's *unscored fallback* at its `t0` — the same
   instant — so `_head_scorer` serves **N's score to N+1's event**, and the refusal
   DE 155 deliberately preserved ("a generation with no assembled score … still
   REFUSES by name") is **unreachable**, because another generation occupies its key.

**Test any future fix here with ABUTTING generations.** DE's own cell uses
`[100,200]` and `[400,500]` — a 200 s hole — and cannot see any of this. Drive:
`t = t1` exactly, plus a control one tick earlier (must be correct) and a genuine
next-generation label (must give the identical outcome, proving the label inert).

**Two candidate closures, both with costs I could not price:** exclude `t >= t1`
(one character, but drops every row landing at a generation's close, and HSP says
those exist); or require `gen` on the event and route by it (drops nothing, costs a
24-literal fixture migration in a BE-shared module). That choice is DE's and the
USER's.

## Verifying someone else's fix (DA 142, rule 27) — the four moves

1. **Build your own fixture.** A fix driven only against its author's fixture is
   the class REV 113 caught. Mine was 3 slugs x 2 sides x 2 generations with every
   row at the generation MIDPOINT, so not one sat at `t0` — the exact case the
   pre-fix test got wrong.
2. **Make the corrected count falsifiable.** Run the same generations with rows AT
   `t0` too: if new == old == n there, the corrected count is not just always-n and
   the pre-fix recomputation is not inert.
3. **Answer the dispatch's question, including the half that is NO.** "Does it now
   refuse real under-coverage?" — structurally yes (five predicates fire by name),
   on the LEVEL no: 8.3 % coverage is ADMITTED because `MIN_COVERAGE_DEFAULT = None`.
   Confirm the omission is honest rather than hidden by supplying the floor
   explicitly and watching it refuse.
4. **For "did it narrow something correct", name what you found even when it
   exonerates.** I found two; both were corrections — an old FALSE PASS (two
   generations "covered" by one key) now refused by the right name, and an
   unreachable one that fails loud. Check reachability at the validator before
   calling a narrowing real.

**And run the positive control with the environment check OFF as well as on.** Mine
refused with the disk check on, which looked like a fix defect and was not: a pinned
module had moved in a LATER commit. Isolating the logic from the environment
separated "the fix is wrong" from "the tree moved", and the second was a blocker
worth more than the round's actual question.

## Bounding an upstream data defect (DA 141) — the move that made it answerable

REV 112 found the builder resolving the wrong collector era, so every September
window got `gaps=[]`. The question "is the surviving finding safe?" looked binary
(small share = safe, large share = unsafe). **It was not, and the move that
dissolved it generalises:**

1. **Find the defect's ONE channel and prove it is the only one.** AST census of
   `day_selector` + signatures: `_archive_paths()` and `token_map()` take no era;
   `gaps_by_slug(era)` is the sole era-dependent input.
2. **That licenses a clean subpopulation.** For a window with no gaps in the
   CORRECT table, `gaps=[]` is the RIGHT input — so the defect is *inert* there,
   not merely absent. Without step 1 the clean set is just a subset.
3. **Recompute the STATISTIC on it.** The four latency percentages moved
   +1.251 / +2.807 / +4.126 / +1.297 pp — all one direction, all under 4.2 pp.
4. **Report both halves separately.** The SHARE was large (60.3 % of 09-03's
   settled money in gapped windows) so the ABSOLUTES are exposed; the PERCENTAGE
   survives. A question posed as binary often has two answers about two quantities.

**State the limits or the bound is oversold:** a clean subset is not a random
subsample (gap incidence correlates with market conditions), 09-03's clean set is
only 87 of 246 windows — thinnest on the most exposed day — and **no corrected
number exists without a rebuild.** Say that rather than approximate it.

**Reconcile with the finder's own counts before filing.** `day_slugs('20260903')`
= 247, of which 160 gapped over 2,294.7 s — REV 112 exactly; my 159 / 2,183.4 s is
the same fact scoped to the 246 settled windows. Numbers that differ for a stated
reason read as agreement; unexplained they read as contradiction.

## The placement latency is DRIVEN and clean (DA 140) — do not re-audit it

Verified end to end 2026-09-09, five checks each with a control; **nothing wrong**.
Do not spend another round here unless a book is rebuilt at a new L.

- `apply_placement_latency` keeps `(t − t0)·1000 >= L`; units ms vs ms, identity at L ≤ 0.
- **The L = 250 books contain EXACTLY the L = 0 fills surviving that predicate** — set
  equality on all four days. **Control: the predicate matches at NO other L** (200/240/
  260/300 all fail), so the match is a measurement.
- Timestamps are real: 0 offsets exactly 0.0 of 57,850; 30,975 distinct values among the
  31,471 dropped; no negatives.
- Boundaries are not fill-induced: first-fill p50 172 ms vs later 555 ms, no pile at zero.
- **The DECISION population is identical across L** (delta 0 on three days, per arm too)
  while generations-with-fills halves — the unit is the generation, so the null's matching
  variable is stable across L.

**The open item left behind:** `settlement_endpoint.require_book_declares_L` is absent from
params v19–v23 and can never fire. Its justification — "every landed book predates BE 101"
— **expires at the first rebuild**, because `build_reference` now writes a
`placement_latency` block on every call including L = 0. The fallback is LABELLED
(`source: THE BOOK'S BUILDER RECEIPT DECLARES NONE`), so it is not silent, which is why it
rates below a real defect. **A guard whose stated reason has expired is a control that
cannot fire wearing a justification that no longer holds** — check this class whenever a
guard is left unarmed "for now".

## Hunting untargeted (DA 138/139) — what actually worked

Two defects in two rounds, both in code declared complete. Neither came from
reading documents. The method that found them:

- **Ask what a change made INDISTINGUISHABLE.** DE 155 re-keyed scores from
  per-generation to per-row. The question that found 361 was not "is the new
  key right" but "what can this key no longer tell apart?" — it dropped `gen`.
- **Find the sentence that says a thing was NOT re-tested.** DE 155's own
  commit says "THE ENGINE WAS ALREADY CORRECT … Nothing in the policy is
  changed" — while handing that engine an input shape it had never seen. That
  sentence is where 362 was.
- **Read the fixture, not just the assertion.** DE's case (c) fixture has ONE
  generation, so it can only show a late event vanishing. Add a successor and
  the same event cancels it. A fixture's missing dimension is where the
  untested case lives.
- **Ask what the artifacts would show if it happened.** The cancel record
  carries the RECORD's `gen`, never the score event's — so 362 is invisible in
  every receipt. Defects that leave no trace are the ones no review finds.

## The instrument failure I committed (DA 139) — read this before quoting a count

**Q-DA-361 cited "2.29 % of generations have fills past the next generation's
t0". It was wrong and it was mine.** `last_fill = defaultdict(float)` defaults to
**0.0** while `fill_ns` on this tape is **NEGATIVE**, so `max(0.0, negative)`
stayed 0.0 for every generation and `0.0 > next_t0` was trivially true. Truth:
**0 of 49,240.**

**The rule this cost, stated so the next DA does not repeat it: a NUMBER offered
as evidence needs a control exactly as much as a checker does.** I applied that
standard to the code I was auditing and not to the statistic I was quoting in
the same row. The repaired measurement carries both directions — plant one
overlap and the detector reports 1; re-run with the old initialiser and it
reproduces 1129, which *identifies* the error instead of merely fixing it.

**A defaultdict's default is a silent assumption about the sign of your data.**
On this tape times are negative up to the window close. Check the sign before
any `max`/`min` accumulator.

## The baseline/arm seam (DA 137) — where a retraction stops

When an ARM result is retracted, the question is always whether the 0-cancel
baseline goes with it. It does not, and the reason is five links, each checkable
in seconds — check them, don't recite them:

1. `run_day:7469` builds the baseline as `mod.replay(bk, mod.flagged_stream(bk["rows"], []), 0.5)`
   — **the flag list is EMPTY**, so every score is 0.0. Drive `flagged_stream(rows, [1])`
   as the positive control: it emits 1.0, so the zeros are the empty list, not a
   builder that can only emit zeros.
2. The policy cancels on "first score crossing >= theta_cancel"; 0.0 >= 0.5 is False.
   `BASELINE_CANCELS = 0` is pinned in `be_cancel_axis_null.py` and its
   `reproduction_gate` refuses otherwise.
3. **The defect surfaces are `arm_stream`-only.** AST census: `flagged_stream`
   touches `['gen','side','slug','t']` and no attributes; `arm_stream` touches
   `_head_scorer` and `bk["asm"]["by_arm"]`. Scoring-model defects (a wrong or
   unloaded head) and book-`asm` defects (look-ahead) both live behind those two.
4. **The one real channel — do not skip it.** `bk["rows"]` IS built from the
   contaminated `asm`, and its row COUNT changes between a pre-fix
   (`PER_GENERATION_SCORES`) and a corrected (`PER_ROW_SCORES`) book. It reaches
   the baseline only via `_decision_times(scores)` → the `gen_start_ns` FIELD.
   It selects no fills: `replay_policy` iterates `reference.items()`, never the
   score stream.
5. `settle_value_cents` / `settlement_legs_by_slug` read `px_cents`, `size`,
   `side`, `slug` — `gen_start_ns` appears **zero** times in the settlement
   valuation. So the one channel moves a field the ruled number never reads.

**The empirical companion, and it is the cheapest confirmation available:** recompute
the baseline from EVERY ledger of a (day, L) chain. On 09-05 L=250 all five gave
1,974.57551 c over 25,721 fills — the fix rounds moved the arm side and the metadata
and left the baseline untouched.

## Recomputing a day's baseline — the four things to state every time

- **Deduplicate, and prove it lossless.** BASELINE fill rows are emitted ONCE PER
  ARM. Reading one arm's copy is right, but hash the sorted
  `(slug, side, px_cents, size, fill_ns)` tuples per arm and show the digests equal
  — otherwise it is a silent choice that happens to be correct.
- **Name the denominator.** The loss % is against THAT DAY'S OWN L=0 total, so it is
  within-day. Verify the L=0 and L=250 slug SETS are identical (they were, all four
  days) — that is what makes it a matched-population measurement rather than two
  different books compared.
- **Populations still differ ACROSS days** — 09-03 is 246 windows, the rest 288. Four
  within-day ratios may sit side by side; they may not be averaged or pooled.
- **Say which days have two implementations and which have one.** 09-06 L=0 is
  SCHEMA 2 with no settlement rows: recompute-only. The other seven cross-check
  against their own `SETTLEMENT_SLUG` BASELINE rows to 1e-9. A uniform table that
  did not say so would imply a check that day never had.

## Two battery cells whose premise changed with the v3 ledger

1. **The missing-status known-bad** must now expect `EARLY_READ_STATUS_UNACCOUNTED`, and
   **its fixture must also name the field nowhere in `where_the_five_live_now`**, or the
   accounted-for route absorbs it and the cell cannot fire.
2. **The four-days cell** now asserts 09-03 resolves through a CHAIN (two artifacts, the
   08:54 one superseded by pair), not one artifact per day. **Anyone regenerating fixtures
   from a single-artifact day silently defeats it.**

## Artifact shapes that are not written down elsewhere

- DE's unsealed early-read emission **nests** the six census fields under `economic`;
  sealed receipts carry them **flat**. The reader reads both and reports
  `where_the_six_were_found`. This cost a false refusal at DA 125 because DA's own fixture
  had reproduced DA's assumption — **fixtures written by the same hand as the reader prove
  nothing about the producer.**
- `inventory_leg` is not a field of the ledger (measured False over every row kind).
  R-803 renamed it `trades_cash_flow_cents`, the exact negative of the R-801 trades leg.
  Its inputs exist but no aggregation rule has been declared, so an inventory leg cannot
  be computed today; every day value is **fills leg only** (R-795) and each line says so.
- The artifact does **not** carry the hourly-files list, so `f_provenance_complete` is a
  RECORDED BOOLEAN, not a check DA ran (R-818 accepts this for design data; DE 142 fixes
  it for quotable days). 09-03 read 54 hourly files.
- **09-03's totals are over 246 slugs; 09-04/05/06 are over 288 windows. Any four-day
  table must carry that or it silently compares different populations.**
- The three counts (`n_fills_arm`, `n_fills_baseline`, `n_cancels_issued`) have been
  visible in the open since the 2026-09-06T14:01Z sealed run landed — that run's own
  eight-name `sealed_field_names` does not include them.
- From DA 129: **sigma has no producer anywhere in the tree**, so a challenger fair series
  cannot be produced today. The Chainlink S60 stream: cadence p50 ≈ 0.93–0.96 s, world→us
  ≈ 1.68 s p50, per-window coverage p50 ≈ 0.997.

## A modified file in a worktree is not always the tip's (DA 157)

My predecessor disclosed ONE uncommitted file in `wt-da`, "byte-identical to what
landed". There were **two**, and the undisclosed one was identical to **`8cea440`
(DA 132, two days back)**, not to the tip -- 2,648 lines against 2,966.

**Identify such a file by BLOB ID against history, never by "is it the tip".** Content
sha256 and `git rev-parse <sha>:<path>` both landed on the same commit in one pass; that
is what turns "an unexplained modified file" into "a recoverable landed copy" and
licenses the discard.

**And a stale copy of an INSTRUMENT is not inert.** That copy predates DA 133, so it has
no `settlement_statistics` at all: it is this reader without the primary endpoint's test
beside `D_E_settle` -- the exact defect this seat's own file records paying for. The risk
in a stale checkout is not losing work, it is *reading through the pre-fix instrument*.

**`wt_refresh.sh` is a real falsifier here and it is free:** it restores files whose bytes
equal `$REF`'s blob and REFUSES the checkout (rc 3, naming the file) on one that differs,
so it both admits and refuses in one run. **Its bound: it compares against ONE ref**, so a
file identical to an OLDER landed commit reads to it as a real edit. Conservative in the
safe direction -- but the identification is still DA's to do.

## Verifying a tree-dependent cell: drive the STATE, not the tree count (DA 157)

`committed_state`'s fix was claimed "green from both trees". Two trees is not the test --
**the reported state must MOVE, or the cell may be passing on a constant.** Three drives on
the same bytes: main -> `COMMITTED_IN_THIS_TREE`, `wt-da` dirty at a stale HEAD ->
`PRESENT_BUT_NOT_COMMITTED`, `wt-da` **spotless** -> `COMMITTED_IN_THIS_TREE`. 23/0 each.

**The spotless worktree is the drive that matters, because it is the input the rival
explanation named.** REV 131's red was blamed on rule-31 dirtiness; DA 156 answered that a
spotless worktree fails identically. Driving the OLD expression there -- `returncode=128 ->
NOT_IN_THIS_TREE` with zero uncommitted bytes, cell conjunct False under OLD and True under
NEW -- **refutes the narrative at the input rather than by argument.** REV 134 drove the
same fix the other way (re-introducing both defects from `wt-rev`); backwards-through-the-
code and forwards-through-the-input are complementary, and neither substitutes.

**Check that a new provenance field is EMITTED, not merely present in the source.** Ran
`--sweep` and read the artifact: `committed_state_asked_of` on 12 of 12 rows, each equal to
`read_from_tree`. A field added to answer a finding is a source edit until an artifact
carries it.

**Residual, checked and correct -- do not re-find it as a miss:** `committed_in_that_tree`
(`:1408`) asks the FILE'S OWN tree via `_is_committed` while sitting beside
`read_from_tree: <AUDIT_ROOT>` and naming *that* tree -- the same conflation one field left
of where it was fixed. **Unreachable by construction**, established by AST over
`audit_module` call sites rather than by grep (rule 32): every path is either
`AUDIT_ROOT / rel` or a `/tmp` scratch path, and both questions agree on each. **LOUD if
reached, driven rather than reasoned**: a `wt-da` path emits one row carrying
`committed_state: NOT_IN_THIS_TREE` beside `committed_in_that_tree: True`. Rule 32's
loud-or-silent test answered by measurement.

## Open conditions

- **CLOSED at the artifact by the coordinator, 2026-09-07T20:1xZ:** DA carried
  `producer_exit_maps_v8` as having a broken pair (`DECLARATION_LINK_CORRUPTED`) after the
  coordinator's revert of DE's v7 (R-760). **It is repaired today**: every pair in the
  chain resolves — v6→v5, v7→v6, v8→v7, v9→v8, each PAIR_OK against the file on disk. The
  revert-of-revert restored v7's bytes and v9 supersedes v8 correctly. **A fresh DA does
  not owe this.**
- The four-day table must be re-run as the 09-04/05/06 settlement re-runs land, and must
  carry the population difference. Those are coordinator GOs.
- **`wt-da`'s HEAD goes stale** and then DA's own landed work looks uncommitted there.
  **Refresh before believing its status.** (Refreshed to `5c7d220` at DA 157; spotless.)
- No factual correction is outstanding against a landed row (DA 117's mistyped digest
  corrected in band at Q-DA-343; DA 128's repetition of DE's `inventory_leg` claim
  corrected in band; DA 132's silent control reported in DA's own report before any of its
  numbers).


## The 09-13 mask, by hand at 2026-09-14T00:00Z (DA 251)

**Why by hand at all.** `da-midnight-verify.timer` fires at **00:06Z**, so it
produces 09-13's mask at **2026-09-14T00:06Z** — six minutes *after* the
2026-09-14T00:00Z boundary. If 09-13 must be valued before that boundary, the
timer is too late and no valuation may wait on a mask.

**The command**, run the moment 09-13 closes (2026-09-14T00:00:00Z), from
`/home/yuqing/ctaNew/live/pm_research`:

```
/home/yuqing/pricer-sol/venv/bin/python3 da_blackout_mask.py --day 20260913 --write
sha256sum /home/yuqing/ctaNew/data/pm_5min/derived/da_blackout_mask_20260913.json
```

Record that digest. It is the thing the 00:06Z check compares against.

**THE CHECK IS NOT "THE TIMER LEAVES IT ALONE" — I DROVE THAT AND IT IS FALSE.**
`da_forward_day_verify.days_needing_verdict` returns
`base = [(closed_token, "closed_today"), (opened_token, "open_today")]`
*unconditionally*: the "already has a closed artifact → skip" rule governs only
the **catch-up range behind the floor**, never today's closed day. Driven in
all three states — no verdict, a CLOSED verdict, an OPEN-written verdict —
`20260913` is in the list **every time**. So on 2026-09-14 the timer **will**
re-verdict 09-13 and **will** rewrite the mask.

**The check that actually protects the artifact** is therefore a digest
comparison, not an expectation of a skip:

```
# after the 00:06Z run
sha256sum /home/yuqing/ctaNew/data/pm_5min/derived/da_blackout_mask_20260913.json
```

It **must equal the digest recorded at 00:00Z**. The mask is a deterministic
function of the raw tape and the gap ledger, so a re-run on a closed day
reproduces it byte-for-byte; an unchanged digest means the rewrite changed
nothing and the freeze is undisturbed. **A digest that moved is a finding** —
it means an input changed between 00:00Z and 00:06Z, which on a closed day
should be impossible.

`da_population_freeze_verify.py` is the mechanical form of that check: the mask
is a listed file, so a rewrite with different bytes refuses
`POPULATION_FREEZE_FILE_DRIFTED` **naming it**, and an identical rewrite passes.

**Why this is safe for 09-07..09-10.** Those days are *behind* the floor, in
the catch-up range, where the closed-artifact rule does apply — all four carry
`day_closed_calendar=True` and are skipped. Only *today's* closed day is
unconditionally re-verdicted. The distinction is the whole reason this note
exists.
