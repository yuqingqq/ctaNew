# Review — DE round 36 at `92c7da4` (the permuted control, the computed pin, the corrected compute)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `92c7da4`** (row Q-DE-54 `2432aa5`). Shas verified at the blob: runner
**`2ba96440ff941785`** (1,902 lines, `EXPECTED_CHECKS = 71`), protocol check
**`f20323d02303baf1`** (333, 26), `de_head_scoring` **`6e503d39b595e24c`** (373, 22), v2 DRAFT
**`a7f92966ed5e2523`** (165).
**Request of record:** `REQUEST_DE_ROUND_36_2026-09-02.md`. **Composed 2026-09-02T20:01:26Z.** One filing, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach 92c7da4` (`data/pm_5min`
mirrored); `~/ctaNew-wt-de` / `~/ctaNew-wt-be` never read; the main tree's `be_forward_day.py`
never read, run or counted (standing_rule 9). `__pycache__` cleared before every execution; both
launchers; streams separate; **no file mutated this round** and `git status --short` **0 lines**.
The declared OUTDIR was **never** passed to `--run` and remains absent; `derived/` **173 before and
after**; **no plan file edited — the v2 DRAFT included**; no unit, timer, scope or anchor;
`DA_MIDNIGHT_MODE` never set; `da_midnight_verify.sh` never run; `git worktree list` unchanged.

## 1. Counts, pin statuses and the two drives — CONFIRMED (item 1)

Eight DE modules, both launchers, PASS = summary = rc 0, zero stderr:
**22 / 71 / 26 / 21 / 24 / 21 / 184 / 92** — R-475 §1 reproduces.

`pin_statuses()` driven in-process: **12 entries — 11 NOT_CALLED** (`phase2_arms.py` among them,
closing **DE35-R1**) and **1 ADDITIVE_DECLARED**: `harmful_exposure_rows.py`,
`sha_at_fit c2e40100ddf3f7a1 → sha_at_run 1bbd8e7525fc27ac`, `functions_changed`
`[_era_or_refuse, _refuse_empty_selection, select_v2_era]`, commit `e12e2c70…`.

`--run --outdir <scratch>` → **rc 2** by name, no traceback, nothing created.
**`preflight()` now proceeds past the pin and refuses at the scorer** (0.10 s): *"no feature
assembly is wired, so q1_arrival_composed_lgbm/btc cannot be scored: the incumbent needs 60 PM+fine
values and the head under test 106…"*. So the answer to "what refuses first now" is: **the wiring,
and only the wiring** — the pin is no longer a blocker, which is what round 36 set out to do.

## 2. DE36-C1 — **CONFIRMED in every sub-claim, measured** (item 2)

Fixture: one slug, one side, three generations — gen 1 above (acts), gen 2 above but **non-acting**
(the side is HELD by gen 1's cancel), gen 3 below; `theta = 0.5`; the draw names gen 3 (size 1 =
the treated action count).

| | treated | control as built (`:947-971`) |
|---|---|---|
| events per generation | `{1: 1, 2: 1, 3: 1}` | **`{3: 2}`** — the drawn generation carries **two** events at one `t0`; gens 1 and 2 carry **none** |
| score multiset | `[0.1, 0.8, 0.9]` | **`[0.1, 0.9]`** — one above value **dropped by the `zip`** |
| cost-adjusted value | 12.000 | 4.000 |

The cycling branch (`:956-957`) is **unreachable**: `|_vals| = |_above| ≥ |actions| = |_want|`
always, since every action is an above event. So all four of R-475 §2(a)'s claims hold, and **the
DRAFT's §5 — my own wording, "no event exists in one arm that does not exist in the other" — is
false of the stream the runner builds** in both directions (missing events, and a missing value).

**And a TRUE swap does not fix it — measured.** Swapping the acting-above key (gen 1) with the drawn
key (gen 3), everything else in place:

| stream | scores | realised cancels | value |
|---|---|---|---|
| treated | `[0.9, 0.8, 0.1]` | **{1}** | 12.000 |
| true swap | `[0.1, 0.8, 0.9]` | **{2}** | 8.000 |

The non-acting above event at gen 2 **acts** once gen 1 no longer cancels, and gen 3's above value
never acts because the side is HELD by then. So the realised cancel set is neither the treated set
nor the drawn set, and `control#2` (`:981-987`) would refuse on real data — exactly as R-475
predicts.

### THE RULING — neither (α) nor (β) alone: **(γ) = (α)'s stream with (β)'s decision variable**

The measurement above shows why the choice is forced. With a state-dependent policy (HELD suppresses
later crossings) **no assignment of scores can guarantee that the realised cancel set equals the
drawn set** — and the frozen text never asks for that. `DRAFT:147-156` asks for matching on
**action count**, side and hour, and comparison on the decision metric. My own EST-R5 phrasing ("the
cancel set must BE the drawn generations") assumed a stream in which every drawn generation acts;
that assumption is false, and I correct it here rather than let the runner keep enforcing it.

**(γ), stated so it is TRUE of a stream that can be built:**

1. **The stream is a total permutation.** The control's stream is the treated arm's stream with the
   **score values permuted within `(side, hour)` strata** over **all** above-threshold events
   (acting and non-acting alike) — every generation keeps **exactly one** event at its own `t0`,
   the per-stratum multiset of scores is unchanged, nothing is invented, nothing is dropped.
2. **The draw names which generations receive above-threshold values** — a property of the STREAM,
   assertable before any replay, and the only thing the draw controls.
3. **The matching is on the decision variable, after the replay:** per stratum, the control's
   **realised action count** must equal the treated arm's. Where the permutation makes them differ,
   the draw is **rejected and redrawn** (bounded attempts), and `n_draws_attempted` /
   `n_draws_accepted` / `n_rejected_by_stratum` go in the receipt.
4. **`control#2` is withdrawn.** Set identity is a property the estimand does not require and
   state-dependence makes unattainable; keeping it guarantees a refusal on real data.

**The predicate that proves it** (all four computable, none a substring test):

| # | predicate |
|---|---|
| P1 | the two streams' `(slug, side, gen)` **key multisets are equal**, and every key appears exactly once in each |
| P2 | per `(side, hour)` stratum, the **multiset of scores is equal** in both streams (a permutation, not a substitution) |
| P3 | every **drawn** generation carries a score **≥ theta_cancel** in the control's stream, and no undrawn generation does |
| P4 | after the replay, **per-stratum realised action counts are equal**; a draw that fails P4 is rejected, counted, and redrawn |

**What §5 must say** (replacing "so that the control cancels exactly the drawn generations… the
runner already implements it"):

> The acting control's stream is the treated arm's stream with the above-threshold score values
> **permuted within `(side, hour)` strata**: one event per generation in both arms, the same score
> multiset per stratum, and the drawn generations carrying the above-threshold values. Because the
> policy is stateful, a permutation does not fix WHICH generations act: the control is therefore
> matched on the frozen decision variable — the **per-stratum realised action count** — with draws
> that fail the match **rejected and redrawn**, and the attempts, acceptances and rejections
> reported. No score value the head did not produce ever enters either stream.

## 3. DE36-C2 — **CONFIRMED**, and it is one of three (item 3)

`:1733-1734` asserts `"_above = [e for e in treated_scores" in _ctrl_src and "_below" in _ctrl_src`
— DE34-R6's class at a new line, and it is the check standing where item 2's P1–P4 belong. A parse
scan finds **three** such checks at this tip: `:1372` (the null's replay shape), `:1673` (the
preflight-order check), `:1733`. The predicate I want is item 2's, computed on the two streams a
draw actually produces — it cannot be satisfied by a comment and it fails when the shape is wrong,
which is precisely what `:1733` does not do today (it passes on the stream measured above).

## 4. The pin — DE36-C3 / C4 / C5 all **CONFIRMED**, with three rulings (item 4)

- **C3 (one-level closure).** `import_closure` (`:357-369`) parses the runner alone.
  `harmful_exposure_rows.py:262` imports `ERA_BOUNDARY_NS` from `harmful_candidate_manifest` at
  module level, so that manifest-pinned file is outside the closure and `pin_statuses` reads it
  **NOT_CALLED** — and because the closure test precedes the identity test (`:451-453`), a change
  there would never be compared. Identical today; silent tomorrow.
- **C4 (functions only).** `_fn_asts` (`:418-426`) dumps **top-level functions**; module-level
  constants and imports are outside the comparison — including `ERA`, `MARKOUT_S` and
  `FILL_HORIZON_S`, any of which moves the numbers without touching a function body.
- **C5 (unpinned declaration; BLOCKING without a falsifier).** `DECLARED_ADDITIVE` (`:155-173`)
  carries reasons and **no AST sha**, so a later edit to `select_v2_era` still reads
  ADDITIVE_DECLARED; and `called#1` (`:490-498`) is raised only on the verify path — nothing in the
  suite drives it.

**Are the three declared reasons TRUE?** **Yes.** `_era_or_refuse` returns `fi.ERA` when `era is
None` — the value this population is selected under — so it changes no non-empty selection;
`_refuse_empty_selection` only raises on an empty one; and `select_v2_era`'s remaining change is
threading those two. I verified last round that **nine of the ten feed functions the diagnostic
calls are AST-identical** to the fit commit and that `MARKOUT_S`/`FILL_HORIZON_S` are unchanged;
that measurement stands at this tip.

**RULINGS.** (i) **The closure must be transitive, bounded by the manifest** — walk first-party
imports transitively and intersect with `fit_code_files`; the intersection terminates at twelve
files, so no depth number has to be invented. (ii) **Module-level statements must enter the
comparison** — compare the module's top-level body (functions **and** `Assign`/`Import`/
`ImportFrom` nodes, docstrings excluded); a constant is exactly the kind of change a function-level
diff cannot see. (iii) **The declaration must be pinned to what it declared** — each
`DECLARED_ADDITIVE` entry carries the AST sha of that function at the fit commit **and** at the
declaring tip, so a later edit re-opens the question instead of inheriting the pass (rule 12's shape
applied to a declaration). And **`called#1` needs its falsifier**: a fixture manifest plus a
synthetic undeclared change, driven through `verify_called_code`, asserting the refusal by name
(rule 15) — the coordinator's in-process demonstration is the right test, shipped.

## 5. The compute the USER schedules on — DE36-C6 **CONFIRMED** (item 5)

**The fixture is not 471 windows.** The timed smoke's reference (`:1209-1216`) is **20 slugs**,
each with **one generation, on one side, carrying one tranche** (`SELL_UP: []`), and the projection
at `:1337-1341` scales `smoke_s / 20 × 471`. The DRAFT's §1a table (`:35-57`) reports that as
"one `arm_result` (4 legs, **471 windows**) — 0.03 s".

**Does it transfer? No — it is a floor.** Replay cost scales with generations per window, events per
generation and tranches per generation; the fixture carries one of each on one side, while the real
reference is whatever `build_reference` produces from 471 windows of a quoting policy on both
sides. Nothing has measured the population's per-window shape, because the feed has not run at this
tip.

**RULING — what should reach the USER.** Two numbers, labelled differently:
- **the feed, ~28.6 min once — measured on the real population** (round 33), the only figure with a
  real denominator;
- **the replay — UNMEASURED**, with the synthetic projection given explicitly as a **lower bound**
  and the scaling law named. The honest way to close it costs one feed run and no cell: the feed
  already emits `n_generations` and `rows_per_generation`; publish those, then price the replay as
  (measured per-generation cost) × (measured generation count) × 20 replays per cell × 56 cells +
  800 replays × 2 null cells.

**And on my DE35-R2:** its **4× half stands** — a draw is four replays, arithmetic, and §1a adopts
it. The **"~1000× overstated in total" half is DE's, not mine, and it is not established**: it rests
on the 20-window one-generation fixture. It must not travel to the USER as "measured".

## 6. Closures at the line (item 6)

| finding | status at `92c7da4` |
|---|---|
| **DE35-C2** / my DE34-R1 | **CLOSED** — `:1536`, `:1755` assert `max_cancels_per_minute` over the arms' params, and `:805-812` carries `cancels_requested / rate_passed / suppressed_rate_limited` with the identity evaluated in code. The self-grep is gone (parse scan: zero `open(__file__)` predicates) |
| **DE35-C3** | **CLOSED** — protocol check `:278-291`: value equality **plus** an `ImportFrom` of `FILL_HORIZON_S` **and** no module-level `Assign`, exactly the predicate I specified |
| **DE35-C4** | **CLOSED** — `gidx` and `treated` hoisted above the seed loop (`:910-913`) |
| **DE35-C5** | **CLOSED** — `:974` `theta=th[head]`, no `else 0.5` |
| **DE31-R2** | **CLOSED** — `null_population` (`:993-1003`): `n_strata`, `strata_with_room`, `strata_forced`, `n_distinct_draws`, `point_mass`, with the reason. This is the receipt requirement I ruled, in the shape I ruled it |
| **DE35-R3** | **CLOSED** — `de_head_scoring` gained a thresholds range guard and converts `LightGBMError` at the module boundary (22 checks, up from 21) |
| **DE34-R2, R3** | **CLOSED** — no docstring predicate and no `ok(True, …)` survives the parse scan |
| **DE34-R6** | **OPEN, three sites** — `:1372`, `:1673`, `:1733` |
| **DE35-R4** | **CLOSED in the document** — §3 now states the rule (`2 × the largest rung`) or labels 0.5 s CHOSEN, which is what I asked |
| **DE35-R5** | **CONDITIONALLY closed** — §2's pair becomes readable only when the stream is fixed per item 2; §5's text acknowledges it |
| **DE35-R1** | **CLOSED** — `phase2_arms.py` reads NOT_CALLED |

## 7. The v2 DRAFT as a document (item 7)

Still a **PROPOSAL throughout** — the header, every section and the closing ruling-request say so;
it edits no frozen document; the protocol check still verifies that it says so and that **no code
cites it**. Nothing in it is decided beyond §1, whose substance the code already carries as a
correction (ruled acceptable last round).

**But one sentence must change before it goes to the USER**, and it is mine: §5 says *"the runner
already implements it"*, and the measurement in item 2 shows it does not — events exist in the
treated stream that do not exist in the control's, and a score value is dropped. **A proposal that
misdescribes the code it asks the USER to bless is the one thing that must not reach them.** Filed
as **DE36-R1**.

**With item 2's ruling, before the package travels whole:** §5 takes the (γ) wording above (stream,
draw, decision variable, rejection accounting); §2's pair is re-read against that stream (its
tight rung is meaningful only when the control's scores are the head's own — DE35-R5); §3 is ready
as written; §1 and §4 remain ready and were already ruled ready to travel ahead. §1a's compute must
be re-labelled per item 5 before it is quoted to a scheduler.

## 8. What the coordinator missed — the class (item 8)

The parse scan is otherwise **clean**: no predicate that cannot go red, no docstring standing in for
code, no self-grep, and no constant that is a policy input in disguise (the last one, the control's
`theta`, closed at `:974`). `main()`'s except tuple is unchanged and still adequate at this tip
(the wiring refuses first). Receipt numbers now carry their populations — `null_population` beside
`null_quantiles` is the case in point.

The residue is the three substring checks of item 3 (**DE36-R4**) and one number without its
population: §1a's replay figure (item 5, **DE36-R2**).

## Corrections of my own (rule 13)

My round-35 filing repeated the module's phrase that the preflight order is *"asserted from the
parse"* (`27c1ccd:1456-1458`). Checked at the blob this round: the **scope** is AST (the source
segment of `run`), the **assertion** is a substring-position comparison
(`_runsrc.index("preflight()") < _runsrc.index("build_reference(")`). It goes red on a reformat and
green on a `preflight()` call that never executes. I should have read the predicate rather than the
label; it is item 3's class and it is now at `:1673`.

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DE36-R1 | **HIGH** | v2 DRAFT §5 | the proposal states "the runner already implements it" — measured, it does not; the sentence must not reach the USER |
| DE36-R3 | **MEDIUM** | `:981-987` | `control#2` demands set identity, which no permutation can deliver under a stateful policy (measured: a true swap realises {2} for a drawn {3}) — it will refuse on real data |
| DE36-R2 | **MEDIUM** | v2 DRAFT §1a, runner `:1209-1216`, `:1337-1341` | the compute figure calls a 20-window, one-generation, one-tranche, one-side fixture "471 synthetic windows"; the real per-window shape is unmeasured, so the number is a floor |
| DE36-R4 | LOW-MEDIUM | `:1372`, `:1673`, `:1733` | three source-substring checks, one of which is the "asserted from the parse" claim |

**DE36-C1 CONFIRMED** (all four sub-claims, measured). **DE36-C2 CONFIRMED.** **DE36-C3, C4, C5
CONFIRMED** — with the three declared reasons verified TRUE. **DE36-C6 CONFIRMED**, and the DRAFT's
own §1a inherits it.

## Disposition and round 37's order

**RELEASE `92c7da4` as round 37's base.** The round closed what it set out to: the pin is computed
and no longer blocks (11 NOT_CALLED, one ADDITIVE_DECLARED with true reasons), the rate limit is a
predicate over params, `FILL_HORIZON_S` is bound by parse, the null reports its population, the
seed loop is hoisted, and the head-scoring boundary converts its third-party error. Nothing can be
produced — the wiring refuses in 0.10 s — so no finding here can reach an artifact.

**Round 37, in this order:**
1. **§5 as ruled in item 2** — the (γ) stream, the P1–P4 predicates replacing `:1733`, and
   `control#2` withdrawn in favour of the per-stratum action-count match with rejection accounting
   (**DE36-R1, R3**). Nothing else in the package can be read until this is true of the code.
2. **The DRAFT's two sentences** — §5's "already implements it" and §1a's "471 synthetic windows"
   (**DE36-R1, R2**) — so what goes to the USER describes the code that exists.
3. **The pin's three rulings** (item 4): transitive-bounded closure, module-level statements in the
   comparison, the declaration pinned to its AST sha, and `called#1`'s falsifier.
4. **The wiring** (the feature assembly), with the feed's own counters published so item 5's replay
   figure stops being a floor.
5. **DE36-R4's three substring checks**, last — they are the cheapest and the least consequential.
