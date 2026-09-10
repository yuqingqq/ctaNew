# REVIEW 168 — the twelve freeze-bearing guards, driven to their failing side

**VERDICT: eleven of twelve fire on their failing side; the twelfth is MEASURABLY
DOMINATED. Two hours ago none of the twelve had ever been seen to refuse.**
Plan step 4 commits builder, scorer, thresholds, settlement convention, null
construction and decision rule — these twelve are the guards that footing rests on.

Read-only except `live/pm_research/p003_rule6_floor.py` (fixed, `2501d4c`) and my own
scratch. No lock taken, no book loaded, no heavy unit. Rule 34a honoured: nothing
under `data/pm_5min/tier2/**/day=2026-09-08/` or later was opened. as-of 2026-09-10T17:11Z.

## 1. THE FOURTEEN BRANCHES (twelve guards; `verify_book_against_builder_receipt` has five)

| guard | driven |
|---|---|
| `verify_book_against_builder_receipt` — DIGEST MISMATCH | YES |
| — receipt absent / no book digest / wrong day / book absent | YES ×4 |
| `POINT_ESTIMATE_PRIOR_UNREADABLE` | YES |
| `POINT_ESTIMATE_PRIOR_IDENTITY_MISMATCH` | YES |
| `POINT_ESTIMATE_PRIOR_LATENCY_MISMATCH` | YES |
| `POINT_ESTIMATE_OUTPUT_EXISTS` | YES |
| `POINT_ESTIMATE_OUTPUT_NOT_DIRECTORY` | YES |
| `POINT_ESTIMATE_DRIVER_CHANGED_DURING_RUN` | YES |
| `POINT_ESTIMATE_DRIVER_NOT_COMMITTED` | YES |
| `HOLM_FAMILY_SMALLER_THAN_THE_TESTS` | YES (`family 2, 3 tests -> REFUSED 2 < 3`) |
| `SETTLEMENT_CONTROL_BELOW_THE_FAIL_CLOSED_FLOOR` | YES (5/199 refuse, 200 passes) |
| `p003_rule6_floor` — the floor | **fired for the WRONG PURPOSE; FIXED** |
| `POINT_ESTIMATE_PLACEMENT_LATENCY_ABSENT` | **DOMINATED (measured)** |

The most load-bearing one is alive: `verify_book_against_builder_receipt` has two
occurrences in 281 files, is called once inside `run_day`, and sits on the path of
every armed number tonight. All five branches refuse, including
`reference-book digest mismatch -- BE's receipt declares ffffffffffffffff, the bytes
on disk are b344b0bad34c`.

## 2. THE FIX (`2501d4c`)

DRIVEN, before: `THE_NUMBER = 150` — below CLAUDE.md rule 6's minimum — was RETURNED.
The predicate was `not isinstance(n, int) or isinstance(n, bool) or n < 1`: it tested
that the value is A number, never that it is THE number. **The guard was green and its
name promised a guarantee nobody was providing** — the subtlest of the three failure
kinds and the one a passing test hides.

And `RULE6_FLOOR_DIVERGED` was doing two jobs: the declaration gives that name to
CARRIER DIVERGENCE (`refusal_name`; known_bad "a carrier whose value is 199"), and
`floor()` had borrowed it for a TYPE fault. Rule 16. Each fault now has its own name:
`RULE6_FLOOR_NOT_A_USABLE_NUMBER`, `RULE6_FLOOR_BELOW_THE_RULE`, and `DIVERGED`
unchanged for `reconcile()`.

RE-DRIVEN, eight cells: 200 RETURNS; **500 (a RAISED floor) RETURNS**; 199/150/1 REFUSE
`BELOW_THE_RULE`; `"200"`/None/True REFUSE `NOT_A_USABLE_NUMBER`. `reconcile()` on the
real tree still returns AGREED; all four DA carriers import clean.

`RULE_6_ABSOLUTE_MINIMUM = 200` is NOT a fourteenth copy of `THE_NUMBER`. They are
different quantities: `THE_NUMBER` is this programme's floor, which the declaration says
"may be RAISED ... may never be lowered"; this is the bound it may never be lowered
THROUGH.

**SAFE AGAINST THE LIVE RUN, ESTABLISHED NOT ASSUMED.** `p003_rule6_floor` IS in step 2's
static import closure (80 modules), so I checked where the run reads from: both draw
processes have cwd `/home/yuqing/ctaNew-wt-arms/live/pm_research` with
`sys.path.insert(0, '.')`. They read the ARMS worktree; the edit is in the main tree.

## 3. THE DOMINATED ONE — and why its scope is narrower than `LEGS_DO_NOT_CLOSE`

Last round I recorded it as unreached and refused to call it dead. It is now MEASURED.
Two shadows: `assert_driver_source` at `:1010` (unrelated, stubbed to reach the branch
under test), and behind it the real one — `run()` calls `assert_point_estimate_result`
at **`:1019`**, which itself calls `R.assert_one_placement_latency` and folds its refusal
into `problems`. Driven on a result satisfying every other contract term:

    full result, L = 250                -> CONTRACT SATISFIED
    L_place_ms key REMOVED              -> REFUSED POINT_ESTIMATE_RESULT_CONTRACT_VIOLATION
    L_place_ms = None                   -> REFUSED POINT_ESTIMATE_RESULT_CONTRACT_VIOLATION
    assert_one_placement_latency alone  -> REFUSED SETTLEMENT_PLACEMENT_LATENCY_ABSENT_FROM_THE_DOCUMENT

Every input that would trigger `:1054` is caught at `:1019`. **Second measured-unreachable
guard tonight — but DOMINATED BY AN ORDERING, not by construction.** `LEGS_DO_NOT_CLOSE`
could not diverge arithmetically; this one becomes live again if 1019/1054 ever swap.
The two must not be filed as the same thing.

## 4. THE LIMIT, WHICH SHIPS WITH EVERY NUMBER

The sweep predicate is TEXTUAL: a cell that drives a refusal without naming it — bare
exception, message fragment, via a helper — reads as never-exercised. **The error
INFLATES rather than deflates.** I hand-checked six of fifty-three.

**AND THE RATIO MUST NOT BE EXTRAPOLATED.** Roughly one defective in twelve is the only
basis anyone has for guessing what the other thirty-nine hold, **which is a weak basis
and should be said as one.** These twelve were SELECTED as freeze-bearing — the guards
most likely to have been exercised in anger and least likely to be representative.
Multiplying one-in-twelve across thirty-nine produces a tidy number carrying no
information.

## 5. A FINDING I DID NOT GO LOOKING FOR

`verify_book_against_builder_receipt`'s refusals are UNNAMED — `REFUSED DAY 2026-09-04: …`
with no token. **29 such sites in `de_multiday_gate1_runner.py`.** An unnamed refusal
cannot be asserted by a cell matching on names, cannot be resolved by an automated
reader, and was invisible to my own sweep — which is why this guard surfaced only in the
verdict-function pass. A real defect and a measured limit of my instrument.

## ROUTED

1. **DA** — the standing-check spec is sent (five acceptance criteria, three falsifier
   cells, the silent-regex control, and the fifth criterion added on DA's own `_ok(`/`ok(`
   finding: the checker must demonstrate it examined what it claims).
2. **DE** — `POINT_ESTIMATE_PLACEMENT_LATENCY_ABSENT` is dominated by the ordering at
   1019/1054; either remove it or record it as dominated with that reason.
3. **DE** — the 29 unnamed `REFUSED DAY` sites are a distinct uncounted population.
4. **coordinator** — the twelve are settled; the ratio caveat travels with the count.
