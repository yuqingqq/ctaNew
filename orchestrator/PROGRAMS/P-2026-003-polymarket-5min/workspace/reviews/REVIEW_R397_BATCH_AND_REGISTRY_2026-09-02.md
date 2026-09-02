# Review — BE's R-397 batch (Q1's leg, Q3's own gate, A2) + the registry acts (v25, v26)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `0bcc0db`** (BE's code `32bf241`, filing `bd9bb44`; the
coordinator's amendment-E application and R-398 inside the tip).
**Continues** `REVIEW_RR2_CLOSURE_AND_RR2_3_2026-09-01.md` (`ebf0ad6`).
**Composed 2026-09-02T04:12:02Z.** One filing, per R-377.

Method unchanged: detached worktree at the pinned tip, ledgers symlinked
read-only, verdicts from execution against committed bytes; mutants applied in the
lab and reverted; no repository file modified, nothing written to any production
ledger, artifact, service or tape.

---

## Verdict

### The batch is CORRECT and RELEASED: R-397 rulings 1 and 2 are executed as ruled, the registry acts are sound, and I could not break any of the new guards.

### HOLD on ONE inference, not on the batch: **the "Phase-2 WINNER" question escalated in R-398 should not be answered until the surviving statistic is measured at the ACTION unit.** Q1's AUC — the number the whole survival rests on — is computed over 311,640 ROWS while the cell reports 177,674 ACTIONS as the n it was computed on (RR4-3).

Three findings, all filed with their evidence: **RR4-3** (row-level statistics
under action-level n), **RR4-1** (`gate_conjuncts_evaluated: true` on twelve cells
that carry a null conjunct), **RR4-2** (the one-draw disclosure overstates the
single-cell consequence — **and the error is originally mine**, from my round-1
filing; corrected in band below).

Nothing in the batch's engineering is wrong. Three of my mutants died where they
should, including the two aimed at the newest code, and one of BE's own three
surviving mutants found a defect I had asked for and would not have caught.

---

## Scope 1 — BE's R-397 batch

### Ruling 1 — Q1's incumbent leg: EXECUTED, and the comparison is real

| claim | verified how | result |
|---|---|---|
| the leg has a production call site | `q1_incremental` present in both arms' results | `beats_incumbent_hazard_head: true`, `incumbent_counterpart_computed: true` |
| candidate vs incumbent | read from the artifact | lgbm **0.8302782896947175** vs **0.7139077229218406** → increment **+0.11637**; linear 0.7733 → **+0.05942** |
| increment arithmetic | recomputed | candidate − incumbent equals the emitted increment on both arms |
| incumbent provenance `18701008c2bd18c6` | **re-hashed the artifact myself** | `data/pm_5min/derived/phase2_fits/linear_d_btc.json` → sha256[:16] **`18701008c2bd18c6`**, `arm: INCUMBENT_REWEIGHTED_ONLY` |
| windows | artifact | 166 of 166 used, `single_class_window: 0`, `too_few_rows: 0` |
| the leg refuses an impostor | **mutant**: provenance requirement removed | **KILLED** — "the right arm but NO provenance must REFUSE" |
| "beats" is strict | **mutant**: `beats_incumbent_hazard_head` forced `True` | **KILLED** — a tie does not beat |

**The judgement I most wanted to check, and it is right:** the §5(2) sign-flip
increment null is computed and **not adjudicated**, because AUC is not additive —
the population AUC difference is not the mean of per-window AUC differences, so a
sign-flip p over per-window differences describes a different number than the one
in the cell. Emitting it as `increment_null_REPORTED_NOT_ADJUDICATED` with the
reason in-band is the correct resolution of a genuine estimand mismatch, and it is
the opposite of the I11-B5 defect (a p beside a number it does not describe).

I did **not** re-run the fit, so `candidate_auc` is verified as internally
consistent and provenance-bound, not recomputed from the tape. What I can say from
execution is that the incumbent side is the committed incumbent, by hash.

### Ruling 2 — Q3's own gate: EXECUTED as ruled

`Q3_magnitudes` cells now carry `status: OK` and
`declared_gate_outcome.passed: true` with both slope conjuncts true, and the cell
text carries the ruling's own logic (*"a conjunct nobody computed reads null,
never false"*). Q2 correctly keeps `NO_INCUMBENT_COUNTERPART` with
`passed: null` and `conjuncts.incumbent: null` — unanswerable, not
answered-and-failed. That distinction is the substance of the ruling and it is
implemented on both sides.

This closes my round-1 F-4 exactly as filed.

### A2, and the anchor (my RR3-1) — CLOSED, and the anchor goes further than I asked

A2 reads **FROZEN — IN FORCE**, by the USER's ruling, quoted in the file, frozen
by coordinator commit rather than by a seat (rule 4 honoured). One-sided governs;
`p_two_sided` retained; this family stays at 500 draws; 2000 is declared
**prospectively** for the next population — which is the disposition I ruled for
last round.

`frozen_prereg_anchor` terminates the chain where I asked and one step better:
`DECLARED_GATES` is **read from git at the frozen prereg commit `3b71d3e`**, and
`INCUMBENT_COMPARABLE` is **derived from the hash-verified incumbent artifact's
own weight blocks** rather than transcribed. I verified the second independently:
`linear_d_btc.json` carries `hazard_weights` and `value_weights` and has **no**
sign or magnitude weights — which is exactly the emitted
`{Q1: true, Q2: false, Q3: false, Q4: true}`. The Q2 divergence between the two
sources is preserved and explained, as I asked.

**Mutant:** paraphrasing the Q1 gate string in `DECLARED_GATES` → **KILLED**, with
a refusal naming the document wording against the code wording. The anchor works.

### The family arithmetic reproduces

Holm re-derived independently over all 24 cells: **0 disagreements**. Survivor set
recomputed from `status == OK AND holm_p < 0.05`: **exactly the emitted 12**. No
cell is published as surviving whose own `declared_gate_outcome.passed` is not
`true` (computed: the set is empty). Statuses: 18 OK + 6
`NO_INCUMBENT_COUNTERPART`; denominator 24, declared-not-evaluated.

### RR4-1 — MEDIUM (latent) — `gate_conjuncts_evaluated: true` on twelve cells that carry a null conjunct

RR2-1's field is keyed on one specific fact — `incumbent_counterpart_computed` —
rather than on the general one the artifact now carries. Computed over the file:

**12 of 24 cells (Q2 × 6, Q4 × 6) assert `gate_conjuncts_evaluated: true` while
their own `declared_gate_outcome.conjuncts` contains a `null`.** Q4's
`matched_random` conjunct is uncomputed; Q2's `incumbent` conjunct is
unanswerable. Two renderings of one predicate disagree — the F2 lesson, in the
code written to close RR2-1.

It changes nothing today: Q4 fails Holm independently (0.2499) and Q2 is blocked
by status, so no cell is published as surviving with an unevaluated conjunct. It
is latent rather than harmless: **if Q4's p improved on a future population, six
cells would be published as surviving with a conjunct nobody computed** — the
exact shape RR2-1 exists to prevent, in the head where it would matter most.

**Closure:** derive `gate_conjuncts_evaluated` from
`declared_gate_outcome.conjuncts` (no null) rather than from the incumbent-leg
fact, and let the status follow it for any head. Red-first: a fixture with a null
conjunct on Q4 must produce `GATE_PARTIALLY_EVALUATED`, and today's Q1/Q3 must
stay OK.

### RR4-2 — MEDIUM — the one-draw disclosure overstates the single-cell consequence, **and the framing is mine**

Every at-floor cell carries
`holm_if_ONE_draw_had_beaten_the_observed: 0.0958`, and BE's filing and R-397/R-398
repeat it as *"holm 0.0479 → 0.0958 had ONE draw of 500 gone the other way"*. That
number is **24 × 2/501** — a flat Bonferroni multiplier. The artifact adjudicates
with a **step-down** Holm, and I recomputed both cases against the emitted p vector:

| scenario | Holm for the moved cell | survivors |
|---|---|---|
| **one** draw beats the observed in **one** surviving cell | **0.047904** (monotonicity carries the first step across the 17 remaining ties) | **12 — unchanged** |
| the floor moves in **all 18** at-floor cells | 0.095808 | **0** |

So a single unlucky draw in a single cell does **not** move the verdict; the
survival depends on the family's **minimum** p, which is set by the block of 18
cells sitting at the floor. The correct statement is that the 12 survivors are
**jointly**, not individually, one draw from failing.

**This error is originally mine** — my round-1 F-3 said "one draw of 500 → holm
0.0958 → nothing survives", the artifact encoded my framing, and two register
entries now repeat it. Correcting it in band, per rule 3: my earlier statement was
right about the arithmetic of a moving floor and wrong about a single cell.

**Closure:** emit both numbers, each computed rather than multiplied — the
this-cell-only recomputation and the floor-moves recomputation. Both are functions
of the emitted p vector; neither needs a re-run.

### RR4-3 — HIGH — the surviving statistics are ROW-level; their n, unit and basis say ACTIONS

`head_report` computes `auc(pred, actual)` (and the Q3 calibration slope) over the
**full prediction vector**, then reports `n = n_actions = len(set(gen_keys))` with
`n_basis: "ACTION (A1.5)"`. The action count governs only the UNDERPOWERED
judgement; the metric itself is over rows. Measured in the artifact:

| head | n_rows (the metric's population) | n_actions (what the cell reports) | rows/action |
|---|---|---|---|
| Q1_arrival | **311,640** | 177,674 | **1.754** |
| Q2_p_pos / Q2_p_neg | 33,622 | 17,604 | 1.910 |
| Q3_m_good | 17,625 | 9,617 | 1.833 |
| Q3_m_harm | 15,912 | 7,988 | **1.992** |

And `q1_incremental` reports `"n_actions": 311640` — the **row** count under an
action name, larger than the action population, which A1.5 explicitly forbids
("no head may report an action count larger than its population's distinct
(slug, side, gen) count").

Two consequences, and they are different in kind:

1. **Definite, and cheap to fix: F1's field is mis-populated.** `statistic_n` was
   introduced to answer *"the n the ADJUDICATED STATISTIC was actually computed
   on"*. For all four heads that n is the row count, while the cells say
   177,674 / 17,604 / 7,988 with `statistic_n_unit: "actions"`. This is F1's own
   defect one layer down, and it is now carried by every one of the 12 surviving
   cells.
2. **Open, and it is the USER's:** whether the metric should be action-deduplicated
   at all. A1.5's letter permits a head to predict on rows if it states its action
   count beside it; CLAUDE.md rule 2 says an evaluator that can attribute one
   outcome to several rows **must** de-duplicate or the result is inflated, with
   the measured ratio (1.99 rows/fill) as its reason. At 1.75–1.99 rows per action,
   with the ratio differing by head, generations contribute unequally to the AUC.
   Whether that inflates Q1's 0.8303 is **unmeasured** — and it is the number the
   Phase-2-winner question would rest on.

**Closure:** report the action-unit statistic beside the row-level one (one
deduplicated pass, no re-fit, no new estimand), correct `statistic_n`/`unit`/`basis`
to name the population actually used, and let the USER see both before the winner
question is answered. **That is the only thing I am holding**, and I am holding the
inference, not the batch.

---

## Scope 2 — the registry acts

### v25 (A–D) and v26 (E) — both sound; verified independently of the coordinator's own checks

| check | result |
|---|---|
| versions along the file's history | v23 → v24 (`b6231c8`) → **v25** (`cdb16d4`) → **v26** (`0bcc0db`) |
| **v25 == the reference application** | ran `de_registry_amendment_check.apply_amendments(v24, A..E)` myself: **types, modules and config_supplied all exactly equal to the applied v25** |
| what E actually did | **raw diff v25 → v26 is TWO lines**: `version: 25 → 26`, and `- ActionSet` replaced by an in-place comment naming DE-AMENDMENT-E, Q-BE-222 and R-398 |
| structural diff v25 → v26 | `config_supplied: removed=['ActionSet']`, `version: 25 → 26`. **Nothing else changed** |
| single authority after E | at v26 `ActionSet` is **produced by DE-ActionSpace** and **consumed by BE-FlowAndFills only** |
| **no new orphan** | computed myself at v24, v25 and v26: exactly **one** pre-existing orphan (`BE-Competition` consumed by `BE-CompetitionAggregator`) at all three versions; **new orphans introduced by E: NONE** |
| migration record | `from_version: 25, to_version: 26, operation: remove, key: config_supplied:ActionSet`, and the **version adaptation is stated in the record itself**: *"drafted at 24->25; applied at 25->26 because A-D consumed that step; content otherwise verbatim"* |
| invariants | `contract_check` v24 → WORKTREE exits 0; `REMOVED (0)`, `TYPE-CHANGED (0)`, `ADDED (72)`; `--selftest` passes incl. "migration REJECTS a different version step" |

### RR4-4 — LOW (instrument) — the reference-equality control cannot express amendment E

`apply_amendments` applies A (types), B and D (modules) and C (config_supplied
additions) — **and never applies E**. I confirmed by running it: the reference
application of A–**E** on v24 still contains `ActionSet` in `config_supplied`.

So the control that certified v25 is a valid equality check for the **additive**
amendments and is structurally unable to certify the non-additive one. At v26 the
same comparison now reports `ActionSet` as "in the reference only" — a difference
that is the intended change, not a discrepancy, and a reader running the checker
against v26 would have to know that to interpret it.

Nothing is wrong in the registry: I verified E's application directly, and the two
independent properties that matter (single producer, no new orphan) hold. The gap
is in the instrument.

**Closure:** teach `apply_amendments` the removal shape — E already parses to
`{operation: remove, key: config_supplied:ActionSet}`, so the applier has
everything it needs — and the equality control then covers every amendment class.
Also worth noting: `contract_check`'s `REMOVED` category does not include
`config_supplied` membership, so a non-additive removal there is invisible to it;
that is why "invariants CLEAN" is true and is **not** evidence that E landed.

---

## Executed evidence

At `0bcc0db`, as of 2026-09-02T04:12Z:

| check | result |
|---|---|
| `v5_deploy_gates.py` | **ALL 17 GATES PASS**, exit 0 |
| `phase2_iter011_run.py --selftest` | **GREEN, 265 checks, 0 failing** (220 at the previous tip) |
| Holm recomputed over 24 cells | **0 disagreements** |
| survivor set recomputed | **exactly the emitted 12** |
| cells surviving whose own gate did not pass | **none** |
| incumbent artifact re-hashed | **`18701008c2bd18c6`**, arm `INCUMBENT_REWEIGHTED_ONLY`, hazard+value weights only |
| `INCUMBENT_COMPARABLE` vs the incumbent's own weight blocks | **matches** (Q1/Q4 true, Q2/Q3 false) |
| mutant: gate text paraphrased | **KILLED** by the frozen-document anchor |
| mutant: `beats_incumbent_hazard_head` forced true | **KILLED** |
| mutant: Q1 leg accepts a predictor with no provenance | **KILLED** |
| one-draw scenario A (one cell moves) | **12 survivors, holm 0.047904** — RR4-2 |
| one-draw scenario B (floor moves) | **0 survivors, holm 0.095808** |
| cells claiming `gate_conjuncts_evaluated: true` beside a null conjunct | **12** (Q2 ×6, Q4 ×6) — RR4-1 |
| head metric population vs reported n | **row-level, 1.754–1.992× the action count** — RR4-3 |
| v25 vs `apply_amendments` reference | **equal** (types, modules, config_supplied) |
| v25 → v26 raw diff | **two lines**; structural diff: ActionSet removed, version bumped |
| orphans at v24 / v25 / v26 | **1 / 1 / 1**, the same pre-existing one; **no new orphan** |
| mutants executed this round | **3, all killed.** The three findings are computed predicates over the emitted artifact and the registry, not surviving mutants |

I did **not** re-run BE's 71-mutant harness; my mutants were targeted at the newest
code and at the claims the dispatch asked me to verify.

---

## Disposition

- **RELEASED:** R-397 ruling 1 (Q1's leg), ruling 2 (Q3's own gate), the A2
  re-emission, the `frozen_prereg_anchor` closure of RR3-1, and both registry acts
  (v25 and v26).
- **HELD — one inference, not the batch:** the **Phase-2 WINNER** question R-398
  escalates should not be answered until Q1's statistic is reported at the **action**
  unit beside the row-level one (RR4-3). The survival claim is a single metric
  measured over 1.754 rows per action, and rule 2 says that is inflated unless shown
  otherwise. Measuring it needs one deduplicated pass, no re-fit and no new estimand.
- **FILED, not holding:** RR4-1 (generalise `gate_conjuncts_evaluated` to any null
  conjunct), RR4-2 (emit both one-draw numbers, computed not multiplied — my
  correction), RR4-4 (teach the reference applier the removal shape).
- **The result of record, as I read it independently:** 12 of 24 cells survive the
  joint reading — Q1 and Q3, four distinct results, every one of them sitting on the
  500-draw floor. **Q4, the decision metric, still fails**: its increment over the
  committed incumbent is positive in all six cells and clears no family-wise bar.
  Nothing in this round changed a number, and BE's own framing is the right one —
  a gate stopped being half-evaluated and a head stopped being blocked by a status
  it never asked for.
