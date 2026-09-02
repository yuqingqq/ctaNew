# Review — DE round 35 at `27c1ccd` (the runner after DE34-C1..C4 and EST-R1..R5; the addendum v2 DRAFT)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `27c1ccd`** (row Q-DE-53 `19ddb43`). Shas verified at the blob: runner
**`81e0bacc2e355859`** (1,645 lines, `EXPECTED_CHECKS = 67`), protocol check
**`5add6cdfafd1ff15`** (318, 26), `de_head_scoring` unchanged **`a074c150a1f2155d`**, addendum v2
DRAFT **`6edefdfda909a897`** (106 lines).
**Request of record:** `REQUEST_DE_ROUND_35_2026-09-02.md`. **Composed 2026-09-02T19:31:21Z.** One filing, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach 27c1ccd` (`data/pm_5min`
mirrored); `~/ctaNew-wt-de` and `~/ctaNew-wt-be` never read; the main tree's `be_forward_day.py`
never read, run or counted (standing_rule 9, BE round 10's row in flight). `__pycache__` cleared
before every execution; both launchers; streams separate; the one mutant restored byte-identical
and `git status --short` **0 lines**. **The declared OUTDIR was never passed to `--run`** and
remains **absent**; `derived/` **173 before and after**; **no plan file edited** — every finding
below is in this filing, none in the DRAFT; no unit, timer, scope or anchor; `DA_MIDNIGHT_MODE`
never set; `da_midnight_verify.sh` never run; `git worktree list` unchanged.

## 1. Counts and the two drives — CONFIRMED (item 1)

All eight DE modules, both launchers, PASS lines = summary = rc 0, **zero stderr**:
**21 / 67 / 26 / 21 / 24 / 21 / 184 / 92** — R-471 §1 reproduces exactly.

`--run --outdir <scratch>` → **rc 2**, *"is not the declared output directory"*, **no traceback,
nothing created**, declared OUTDIR still absent.

**`preflight()` and `_head_scorer()`, driven in-process** — and this is the round's headline:

```
preflight()                       -> DiagRefused in 0.0 s: "the code this run CALLS has moved
                                     since the fit: harmful_exposure_rows.py {now 1bbd8e75…,
                                     fit c2e40100…}, phase2_arms.py …"
_head_scorer("q1_arrival…","btc") -> DiagRefused: "no feature assembly is wired … the incumbent
                                     needs 60 PM+fine values and the head under test 106"
_head_scorer("incumbent_linear_d","btc") -> the same refusal
```

So **DE34-C1 is closed**: the stub that scored width 1 and the branch that returned a constant
`0.5` are both gone, replaced by a named refusal, and the refusal happens **before the feed** — the
order asserted from the parse at `:1456-1458` and measured here at 0.0 s. Preflight's order is
thresholds (both coins × both heads) → `HS.verify_fit_code()` → `verify_called_code()` →
`_head_scorer(...)`; the first failure today is the moved-code pin, so item 6 is what stands
between this tip and a run.

## 2. DE35-C1 — **CONFIRMED in substance, with the mechanism corrected** (item 2)

**The artifact.** The control emits **two** events per drawn generation (`:796-804`): `score 1.0`
at the generation's own `t0`, and `score 0.0` at `t0 + REPOST_DWELL_S`, under a comment claiming
it is "the same below-threshold event the treated arm's stream would produce". The treated stream
is **one** event per generation at its `t0` (`score_events_for` `:1570-1573` →
`de_score_stream.score_events`, one event per row), and `_decision_times`' own docstring
(`:631-633`) says so.

**Where I correct R-471 §2(a)'s "no and yes".** The treated arm is not *incapable* of reposting.
Driven on a two-generation fixture through `replay_policy` (one `(slug, side)`, gen 1 scored 0.9,
gen 2 scored 0.1, `theta_cancel = 0.5` so `theta_repost = 0.25`):

| stream shape | cancels | reposts | cost-adjusted value |
|---|---|---|---|
| TREATED (one event per generation) | 1 | **1** | 4.000 |
| CONTROL as built (+ a 0.0 event one dwell later) | 2 | **2** | 0.000 |

The treated arm reposts through a **later generation's own low score** (`below_since` is set at the
score event that goes below, `:903-906`; `_repost_check` is evaluated **at event times only**,
`:719-731`). So the true statement is sharper than "the treated arm never reposts":

- the control's below-threshold event is **invented** — `0.0` is a literal, not a value any head
  produced — and it exists for **every** drawn generation;
- the treated arm reposts only where **the head's own later scores** fall below `theta_repost`;
- and because eligibility is evaluated at event times only, the control's repost is
  **draw-dependent** — it fires when another event on that `(slug, side)` exists at or after
  `t0 + 2·dwell`, so the null's economics carry a repost/no-repost mixture the treated arm has no
  counterpart for at all.

The two shapes are not economically equivalent (4.000 vs 0.000 on the fixture above), so this is not
a cosmetic asymmetry.

**THE RULING.** Between the two shapes the request offers, **(i) is the estimand `DRAFT:147-156`
describes** — "matched random cancellation on **identical opportunities**" compares which
generations were cancelled, not which policy was run; shape (ii) (both streams re-score after a
cancel) changes what the treated arm measures and needs a second decision-time feature vector the
wiring does not have, so it is a new declaration, not a control fix.

**And (i) has a better realisation than "drop the repost event": permute, don't synthesise.** The
control's stream should be **the treated arm's own stream with the cancel decisions permuted within
each `(side, hour)` stratum** — same events, same times, same scores, reassigned so that exactly
the drawn generations carry the above-threshold values. Then every event in the control exists in
the treated stream, the action count survives by construction (`DRAFT:152-154` needs no separate
matching step), and whatever repost dynamics the head's scores produce are present in both arms in
the same proportion. Synthesising `1.0`/`0.0` events invents a policy; permuting the head's own
outputs is what a matched random control is.

**What §5 should say instead of "exactly as the treated arm's stream would":**

> The acting control's score stream **is** the treated arm's stream with the above-threshold
> assignments permuted within `(side, hour)` strata, so that the control cancels exactly the drawn
> generations and no event exists in one arm that does not exist in the other. Repost behaviour is
> therefore identical in kind to the treated arm's and is never manufactured; the control introduces
> no score value the head did not produce.

## 3. DE35-C2 — **CONFIRMED**, and the file now contradicts itself (item 3)

`:1339-1341` still greps `open(__file__).read()` for *"the frozen protocol names no rate limit"* —
and it now passes because the **correction at `:229-233` quotes the false sentence in order to
retract it**. The check therefore survives on its own retraction. Meanwhile the message at
`:1343-1345` still asserts the retracted claim ("the frozen protocol's axes carry no rate limit"),
and the comment at `:135-136` still cites `:88-99` for the axes while `:229-233` corrects that to
`:99-108`. One file, both readings.

**What the check should assert** — a predicate over behaviour, which the round has already made
available: that every arm's params carry `max_cancels_per_minute` (the per-cell declaration
`DRAFT:71` asks for) **and** that the emitted per-arm counters satisfy
`cancels_requested == cancels_rate_passed + cancels_suppressed_rate_limited`. That is checkable on
`arm_result`'s output, cannot be satisfied by prose, and is exactly what v2 §4 proposes to declare.

## 4. DE35-C3 — **CONFIRMED by execution** (item 4)

I restated the constant in the runner (`from phase4_generation_tables import tranche_table` plus a
module-level `FILL_HORIZON_S = 1.0`) and ran the protocol check: **green at 26**. The predicate
compares a value while the message claims an import.

**The predicate I want** (both halves, neither a new number): `_RUN.FILL_HORIZON_S ==
phase4_generation_tables.FILL_HORIZON_S` — the binding, not the literal — **and** an AST assertion
over the runner's parse that `FILL_HORIZON_S` appears in an `ImportFrom` of
`phase4_generation_tables` and in **no** module-level `Assign`. The module already owns that
idiom (`:1456-1458` asserts call order from the parse).

## 5. DE35-C4 / C5 (item 5)

**C4 — CONFIRMED as a pattern, NOT a run-time blocker.** `_gen_index(reference)` is rebuilt inside
the per-key loop (`:790`) and `_treated_actions` per seed (`:773-774`). Measured on synthetic
references: one `_gen_index` build is **0.31 ms** at ~1,400 generations and **0.41 ms** at ~2,800,
so a 200-seed null cell at a 10 % action count costs ~**0.1–0.4 minutes** of index rebuilding.
Hoisting it is one line and worth doing; it is not what makes the null expensive — see **DE35-R2**.

**C5 — CONFIRMED dead.** `:806`'s `else 0.5` is unreachable: `:725-731` refuses any arm without a
bound threshold and `head ∈ scores_by_arm`, so `head in th` always holds.

**The parse-level ban you ask about covers it, and I ran it.** An AST scan over non-suite functions
for a `Call` whose `theta`/`theta_cancel`/`theta_repost` keyword is a literal (or an `IfExp` with a
literal branch) finds **exactly one hit at this tip — `:806`** and nothing else. Fifteen lines in
the suite, and it cannot be satisfied by a comment.

## 6. DE34-R7 ruled — run against the TIP, with the difference computed (item 6)

**What `851edaf` changed in the feed, measured rather than read.** I compared every function the
diagnostic's feed calls, at the fit commit `e12e2c7` against the tip, by AST:

| function | fit vs tip |
|---|---|
| `replay_with_recorder`, `join_fills`, `generation_table`, `label_rows`, `verify_boundary_times`, `verify_consume_clock`, `trade_receipt_times`, `v2_era_bounds`, `binance_continuity_ok` | **IDENTICAL** (nine of ten) |
| `select_v2_era` | **DIFFERS** — a keyword-only `era` parameter defaulting to `fi.ERA` (unchanged), and `_refuse_empty_selection` |

`MARKOUT_S` and `FILL_HORIZON_S` are unchanged. The commit's own words are borne out: *"every
current number is reproduced exactly"* — the era default is the fit-time one, and the only
behavioural difference is that an **empty** selection now refuses where it used to return silently.
**So the reference the diagnostic builds does not differ from the one the fits saw**, on any
non-empty population — and the §3 population is 471 windows.

**RULING: run against the tip, with the two files' status computed and declared — not against the
fit-commit bytes, and not a hold.** Running the fit-commit bytes would reinstate the silent-empty
defect that commit exists to remove, and it needs a materialised pinned import path the runner does
not have (BE's `materialise_frozen` is that machinery; DE has no counterpart). Holding the run over
a difference that is provably additive for this population would be scrupulosity, not rigour.

**How the pin should say it.** Three changes, all cheap, and they are DE34-R7 made precise by the
measurement above:

1. **Compute the called set, do not list it.** `CALLED_FIT_CODE` (`:134`) names
   `phase2_arms.py`, which the runner never imports (measured: not in the import closure;
   `harmful_exposure_rows` is imported lazily inside `build_reference`, `:253`). The pin therefore
   blocks the run on a file the diagnostic does not execute → **DE35-R1**.
2. **Refuse on a called FUNCTION's AST, not on the file's sha.** Nine of ten are identical; a
   file-level sha cannot say that, and an AST comparison can — it is the same instrument DE already
   uses for call-order.
3. **Carry the residue as a status, not by omission:** per moved file
   `{path, sha_at_fit, sha_at_run, commit, functions_changed, verdict}` with the verdict computed —
   `IDENTICAL` / `ADDITIVE_DECLARED` (with the changed functions named and why the run's path is
   unaffected) / `BLOCKING`. Today that yields `harmful_exposure_rows.py: ADDITIVE_DECLARED
   (select_v2_era: era keyword defaulted to the fit value + an empty-selection refusal)` and
   `phase2_arms.py: NOT_CALLED`.

## 7. My open findings at this tip (item 7)

| finding | status |
|---|---|
| **EST-R1** (rho's denominator a declared constant) | **CLOSED** — the tranche carries the measured mid (`:1015-1023` fixture comment; the constant is gone) |
| **EST-R2** (the receipt's 1 s mis-declaration) | **CLOSED** — `:459-468`: `value_horizon = "[t + L, end of hold]"`, `per_row_table_horizon_s` kept beside the per-row table, and the note says the horizon "is declared in addendum v2, which is a PROPOSAL until the USER rules it" |
| **EST-R5** (the control's cancel set) | **CLOSED for the cancel set** (`:788-804` acts on the named generation, asserted at `:812-818`); **OPEN for parity** — DE35-C1 above |
| **DE34-R1** (self-grep pinning a false claim) | **OPEN** — `:1339-1341`, now self-contradicting (§3) |
| **DE34-R2** (docstring stands in for behaviour) | **OPEN** — `:1297-1298`, unchanged |
| **DE34-R3** (`ok(True, …)`) | **OPEN** — `de_head_scoring:271` |
| **DE34-R4** (the control's literal theta) | **CLOSED in effect** — `:806` binds `th[head]`; the dead `else 0.5` is C5 |
| **DE34-R5** (`main()`'s except tuple) | **PARTLY** — `HSP.ReferenceIntegrityError` added (`:1633-1635`); `LightGBMError` still absent but **unreachable at this tip** because `_head_scorer` refuses first (measured). **`HSP.InvalidParameter` is NOT reachable today**: all twelve bound thresholds are strictly positive (min 0.0445, eth LGBM 15 %), so `0 < theta_repost < theta_cancel` always holds. When the wiring lands, the right closure is **not** a third-party class in the CLI tuple but converting `LightGBMError` to `HeadRefused` at `score_lgbm`'s boundary, plus a positivity guard where `theta_for` returns |
| **DE34-R6** (source-substring check) | **OPEN** — `:1194-1196` |
| **DE34-R7** (the pin) | **RULED above**; the residue is DE35-R1 |
| **DE31-R2** (freedom unreported) | **OPEN at the run** — `null_quantiles` (`:826-835`) reports `n`, the metric and the quantiles, and no freedom. **Yes, it is a receipt requirement**: report `n_strata`, `strata_with_room` and `n_distinct_draws` (a set of the drawn key-sets, already in hand); when `n_distinct_draws == 1` the quantiles are a point mass and the receipt must say so rather than let a reader read an interval |
| **DE31-R1** (rho's reachability reference) | **INERT and now documented** — `_decision_times` (`:628-637`) carries the decision time per generation and its docstring says the map and the generation starts agree *today*, which is the honest form |

## 8. The addendum v2 DRAFT as a document (item 8)

**Is it a proposal throughout? Yes**, and it is enforced rather than asserted: the header, every
section and the closing paragraph say so; it is a **new dated file** that edits neither v1 nor the
frozen protocol (rule 13); and the protocol check verifies both that the draft says "PROPOSAL"/"not
frozen" in its first lines and that **no code cites it** (`de_phase4_protocol_check:287-296`).

**Is anything in it already taken?** §1 is, in a narrow and defensible sense: the runner already
carries `value_horizon = "[t + L, end of hold]"`. Removing a false declaration needed no ruling —
a receipt may not carry a statement its number contradicts while a ruling is pending — and the field
does not claim authority (its note calls v2 a proposal). So: adopted in code, correctly, and the
ruling §1 asks for is the *declaration*, not the removal. Nothing else in the document is taken.

**§2's tight rung `theta_cancel − ε`: under the present control shape it measures nothing on one
side.** The control's repost event is a literal `0.0`, which is below **every** candidate
`theta_repost`, so the control's behaviour is invariant to the rung while the treated arm's is
highly sensitive to it. The pair would therefore report the treated arm's hysteresis sensitivity
measured against a control that has none — the DE35-C1 asymmetry contaminating the very sensitivity
the pair exists to expose. Under the §5 shape I rule in item 2 (permuted head scores, no invented
events) both arms move together and the pair becomes readable. **DE35-R5.**

**§3's 0.5 s is chosen, not computed.** "The smallest dwell that is still longer than the largest
latency rung (250 ms)" names no number: every value above 0.25 s satisfies it and the smallest does
not exist. 0.5 s happens to be **twice the largest rung** — if that is the rule, say it, and the
number is then computed; otherwise it is a chosen rung and should be labelled one. **DE35-R4.**

**RULING on the package: §1 and §4 may go to the USER ahead of §5; §2 and §3 may not.** §1
introduces no number and corrects a mis-declaration already removed from the code; §4 adopts `inf`
with a reporting identity that stands whatever the control does. §2 and §3, by contrast, declare
sensitivities **of the comparison**, and the comparison's shape is what §5 settles: the same pair of
rungs means different things under the manufactured-event control and under a permuted-stream
control (above). Sending them ahead would ask the USER to rule on numbers whose meaning changes with
a decision they have not been given yet.

## 9. What the coordinator missed — the class again (item 9)

An AST scan over the three modules for predicates that cannot go red now returns **three**, all
already mine and all still open: `:1297` (docstring), `:1339` (self-grep), `de_head_scoring:271`
(`ok(True, …)`). **Round 34's `or True` tautology is gone.** The protocol check's own
`"PROPOSAL" in _v2.read_text()` (`:288-292`) is *not* in this class: its claim is about a
document's prose, so prose is the right subject.

Two more, both filed below rather than in the list above: a **compute declaration that does not
match the cell the runner builds** (DE35-R2 — the null's per-seed cost is the four-leg conjunction,
so v1 §d's six-hour figure understates the PRIMARY null by ~4×), and a **number reported without its
population** (DE31-R2's freedom, ruled a receipt requirement in item 7).

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DE35-R1 | **MEDIUM** | `:134`, `:331-358` | the pin blocks the run on `phase2_arms.py`, which the runner never imports — the called set is asserted, not computed |
| DE35-R2 | **MEDIUM** | `:641-700`, `:770-812`; addendum v1 §d | each null draw costs **four** replays (the PRIMARY conjunction), so 200 draws is ~800 replays per cell — the declared "of order 6 hours" understates the cell by ~4× |
| DE35-R5 | **MEDIUM** | v2 DRAFT §2 | the tight `theta_repost` rung cannot move the control as built, so the §2 pair is unreadable until §5 is settled |
| DE35-R3 | LOW-MEDIUM | `:1633-1635`, `de_head_scoring:147-160`, runner `:191` | when the wiring lands `LightGBMError` should be converted at `score_lgbm`'s boundary rather than named in the CLI tuple; `theta_for` should guard positivity |
| DE35-R4 | LOW | v2 DRAFT §3 | "the smallest dwell longer than 250 ms" names no number; 0.5 s is chosen unless the rule is "twice the largest rung" |

**DE35-C1 CONFIRMED** (mechanism corrected: the treated arm *can* repost, measured; the defect is
the invented `0.0` event and its draw-dependence). **DE35-C2 CONFIRMED.** **DE35-C3 CONFIRMED** by
execution. **DE35-C4 CONFIRMED as a pattern, contested as a blocker** — measured at 0.1–0.4 min per
null cell. **DE35-C5 CONFIRMED** dead.

## Disposition

**RELEASE `27c1ccd` as round 36's base.** The round closed what it claimed at the artifact: the
stub is a named refusal that fires **before** the feed (0.0 s, measured), rho's denominator is the
measured mid (EST-R1), the receipt declares the horizon its number has (EST-R2), the control acts on
the generation it drew (EST-R5's cancel set), and the tautology of round 34 is gone. Nothing here
can produce a cell — preflight refuses twice over, by name — so no finding below can reach an
artifact.

Round 36's order, as I read it: **§5's shape first** (item 2's ruling — it is the input to the
USER's §2/§3 decision and to the control's implementation), then the pin as computed statuses
(item 6, unblocking the run), then the wiring's exception boundary (DE35-R3), then the three
long-open prose-predicates (DE34-R1/R2/R3/R6). DE35-R2 should reach the USER with the v2 package:
it changes the compute the null costs, which is a fact about the run they are being asked to
schedule.
