# Review — DE round 40 at `35452c0` (set identity named as such; the null asserted on values; `null_status` and `_by_ca`)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `35452c0`** (row Q-DE-58, same commit). Verified at the blob: runner
**`3f4bf21da2dfa188`** (3,329 lines, `EXPECTED_CHECKS = 119`), v2 DRAFT **`cb693000880c3d94`**
(307 lines, **+17 / −0** against the `cd93663` blob I read for conditions (i)–(iv)), score-stream
**`f85be3354610e2ce`** untouched.
**Request of record:** `REQUEST_DE_ROUND_40_2026-09-02.md`. **Composed 2026-09-02T23:38:04Z.** One filing, per R-377.
**This round is behind the package**, per my own ruling: nothing below re-opens the forwarding.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach 35452c0` (`data/pm_5min`
mirrored); `~/ctaNew-wt-de` / `~/ctaNew-wt-be` never read. Five mutants applied to my worktree
copy and **restored byte-identical** (`git status --short` **0** after). The declared OUTDIR never
passed to `--run`; `derived/` **173 before and after**; nothing written under `data/`; **no plan
file edited** — the DRAFT is in the USER's hands; no unit, timer, scope or anchor;
`DA_MIDNIGHT_MODE` never set; `git worktree list` **34 at quiescence**, unchanged.

## 1. Counts — CONFIRMED (item 1)

Eight modules, both launchers, PASS = summary = rc 0, zero stderr:
**31 / 119 / 26 / 21 / 26 / 21 / 184 / 92** — R-484 reproduces; the row's prior counts (115 / 26)
are the `cd93663` blob's, which I measured myself last round. `--run --outdir <scratch>` →
**rc 2**, no traceback, nothing created; `preflight()` refuses at the scorer.

## 2. Ruling (i) — **YES**, the clause is what I asked for (item 2)

The source comment (`:1401-1414`) now names **SET identity**, says when set and stream identity
coincide (the above values descend in time), gives the reason the set is the decision unit (with
`enable_reduce` False a score is read against `theta_cancel` / `theta_repost` / `theta_reduce`
and nothing else), and states the re-read condition under a reduce band.

The DRAFT's added paragraph is one paragraph, **+17 / −0**: it **sets no number, admits nothing,
and widens nothing**. It states a limit of the definition under a named configuration, cites the
measurement, and says what must happen if the configuration changes. That is exactly the shape I
asked for, and it is the shape this programme's own history calls for — it says so itself.

**Does §5 as landed still describe the null my condition (i) accepted? Yes.** The diff is purely
additive, so the definition, the admission of the identity, the retirement of the guard and the
degeneracy remedy are unchanged; the clause bounds their validity rather than altering them.

## 3. Rulings (ii) and (iii) — **CONFIRMED CLOSED**, driven (item 3)

At the blob: the assertion is on the **logged accepted values** (`_val_differs`, `:1867-1872`)
and it also asserts `drawn` is present on every accepted row; `drawn` is logged at `:1432`; the
stream-map count survives as `n_accepted_stream_differs` with `stream_differs_note`
(`:1485-1486`) and is asserted only as a **bound** (`>= _val_differs`, `:1883-1887`); **zero**
`MRC.draw(_fpool` recomputations remain.

Driven by me on the free fixture, both orderings:

| | DESC (as shipped) | ASC (swapped) |
|---|---|---|
| accepted values | `{16.0: 13, −8.0: 3, 40.0: 4}` | **identical** |
| q50 / net_diff | 16.0 / 24.0 | **identical** |
| distinct / identity | 6 / 4 | **identical** |
| `_val_differs` | **16 of 20** | **16 of 20** |
| `n_accepted_stream_differs` | **16** | **20** |
| `drawn` on every accepted row | yes | yes |

**And I checked the invariance more broadly than the suite does:** of the **21** fields in
`null_population`, **exactly one moves** under the reordering — `n_accepted_stream_differs`.
Nothing else in the block, and nothing in the predicate row, depends on the order of decision-inert
inputs. See **DE40-R3** for the consequence.

## 4. DE39-R1 and DE39-R2 (item 4)

**R1 — CLOSED, in the form I asked.** `null_status` (`:798-802`) is a three-state field and I
drove all three on real cells:

| cell | `interval` | `null_status` | `beats_null_q95` |
|---|---|---|---|
| C1 fixture (collapsed) | `POINT_ESTIMATE_NO_INTERVAL` | **`NULL_COLLAPSED`** | `None` |
| free fixture (sampled) | `NULL_QUANTILES` | **`NULL_SAMPLED`** | `False` |
| free fixture, `draws=0` | `POINT_ESTIMATE_NO_INTERVAL` | **`NO_NULL_REQUESTED`** | `None` |

A reader of the predicate table can now tell a collapsed null from a cell that never ran one, which
is what I filed. One residual on the derivation — **DE40-R2**, below.

**R2 — CLOSED in form.** `len(_ca) == 1` is gone; `_by_ca` (`:2607-2622`) groups the declarations
by `changed_at` and checks **each group** at its own commit and parent. But see **DE40-R1**: the
closure has no falsifier while one group exists, and I measured that.

## 5. What the coordinator missed — the class (item 5)

**(a) Order-dependent statistics.** Measured above: one field of 21. Nothing downstream reads
`n_accepted_stream_differs` — its only references are the receipt field and the suite's two
assertions (`:1883`, `:1930`), so it cannot be mistaken for a decision count by any consumer that
exists. **What the invariance should cover:** not six named fields but **the whole
`null_population` block, with `n_accepted_stream_differs` excluded by name** — the enumeration is
a maintenance surface that will fall behind the block's next field, and the measurement shows the
stronger assertion holds today. Filed **DE40-R3**.

**(b) The exhausted-budget state.** It **cannot arise** at this tip: `null#2` refuses the run when
`accepted < draws`, so a cell that requested draws and accepted none never reaches the receipt —
I drove that at `DRAW_ATTEMPT_BUDGET = 1` last round ("only 3 of 20 … 17 rejected"). So the three
states are exhaustive **because of `null#2`'s policy**, not because of anything the cell records:
`null_status` is derived from the *absence of quantiles*, and the cell carries no
`n_draws_requested`. If `null#2` is ever softened to "report what was accepted", a cell that
requested 200 and accepted 3 would read **`NO_NULL_REQUESTED`** — the one state that says nothing
was asked. **No fourth state is needed; the derivation should come from what was requested.**
Filed **DE40-R2**.

**(c) The six mutants.** They are the coordinator's drives, not a shipped table, so I could not
inspect them; I built five of my own against this round's own claims instead:

| mutant | result |
|---|---|
| `null_status`'s COLLAPSED branch returns `NO_NULL_REQUESTED` | **red at the DE39-R1 three-state check**, by name |
| `n_accepted_stream_differs` hardcoded to 0 | **red at the bound check** (`:1883`), by name |
| `_by_ca` checks only the first group | **GREEN — survivor** |
| the ruling-(ii) assertion weakened (`_val_differs >= 0`) | **GREEN — survivor** |
| (a malformed fourth, discarded: it died on an undefined name, not at a check) |

The first two die **for the reason their names say**. The fourth is the known-unkillable class (a
weakened conjunct in an assertion cannot be caught by the assertion it weakens) — not a defect, and
the reason BE's audit keeps an explicit `AUDIT_KNOWN_UNKILLABLE` list. **The third is a real gap**
and is DE40-R1.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DE40-R1 | LOW-MEDIUM | `:2607-2622` | `_by_ca`'s grouping has **no falsifier** — "check only the first group" survives the suite while one `changed_at` group exists |
| DE40-R2 | LOW | `:798-802` | `null_status` is derived from the absence of quantiles, not from what the cell requested; the cell records no `n_draws_requested` |
| DE40-R3 | LOW | `:1895-1934` | the reordering invariance enumerates six fields where the measured invariant is the whole block minus one named statistic |

**DE39-C1 (i), (ii), (iii): CONFIRMED CLOSED. DE39-R1: CONFIRMED CLOSED. DE39-R2: CLOSED in form,
with DE40-R1 outstanding.** Item 2: **yes**.

## 6. Round 41 (item 6)

**RELEASE `35452c0` as round 41's base.** Every part of my last round's order is built and driven:
set identity is named where a maintainer meets it and bounded where the USER reads it, the null is
asserted on values with the stream-map count demoted to a labelled bound, `drawn` is logged and the
coincidental recomputation is gone, `null_status` separates the three states, and the declaration
check groups by the commit that changed each function. Nothing can run — `preflight()` refuses at
the scorer — so no finding here can reach an artifact.

**Round 41's order (independent of the USER):** 1. DE40-R1 (a two-group fixture for `_by_ca`);
2. DE40-R2 (`n_draws_requested` on the cell, `null_status` derived from it); 3. DE40-R3 (the
block-wide invariance).

**And for ask (5), so round 41 can be dispatched without a second review** — what must move in the
runner for each possible answer:

**If the USER rules MECHANICS on the consumed population** (score everything; the cell is a
mechanics diagnostic): the population and the selection are unchanged, and the work is **reporting
plus one falsifier**. (i) The receipt must carry, per cell, the **computed split composition** of
the generations scored (`n_rows_train` / `n_rows_score` from the tape index) — the DRAFT's "the run
declares which splits it consumed, per cell" must become a computed field, not a sentence, with a
known-bad that a cell whose composition is absent refuses; (ii) the evidence class already carries
`DIAGNOSTIC_NEVER_EVIDENCE`, so nothing there changes; (iii) no change to `build_reference`, the
null, or the grid. **One field, one refusal, one falsifier.**

**If the USER rules the `score`-split restriction**: the change is structural and it reaches the
declaration. (i) `build_reference` must filter generations to the `score` split, which means **the
tape index — the expensive half — must run BEFORE the reference is built**, inverting today's order
(`preflight` → feed → score); (ii) §3's population statement ("471 windows, btc 234 / eth 237")
would no longer describe what is scored, so the receipt must record **both** the declared §3
population and the scored subset with its counts, and **the addendum's §a must be re-read** — that
is a change to a declaration the USER has already been handed, so it needs to travel back to them
rather than be absorbed in a round; (iii) the pool, the strata and the demand all shrink with the
population, so degeneracy becomes materially more likely and the `NULL_COLLAPSED` branch becomes
load-bearing rather than defensive — the per-stratum accepted block is what will carry the reading;
(iv) `is_a_validation` stays False either way. **This answer is the one that costs a round and a
re-declaration; the other costs a field.**
