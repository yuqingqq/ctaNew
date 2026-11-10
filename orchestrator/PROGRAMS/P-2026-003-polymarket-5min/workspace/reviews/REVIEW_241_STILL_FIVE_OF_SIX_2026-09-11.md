# REVIEW 241 — the population is a field now, and the field is optional

REV round 204. Filed 2026-09-11T20:29:32Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 20:24Z. Both
executing refs are `0516c43`; worktree detached there, `git status --porcelain`
empty, all nine lane modules byte-identical on
`origin/de-freeze-chain-v2` and `origin/be-build-runner`.

## THE VERDICT

**Not six of six. Five of six.**

One sentence: **`action_keys_sha256` carries the default `""`, and both the
`run_arm` guard (`if inputs.action_keys_sha256 and …`) and `compare_arms`
treat empty as nothing-to-check, so two arms that simply omit it still replay
different populations and are admitted as `SEAM_IS_HONEST`.**

Driven at `0516c43`, identical values on both arms:

| both arms' `action_keys_sha256` | 6 actions vs 3 |
|---|---|
| each arm declares its own | **REFUSED `REPLAY_ARMS_DO_NOT_SHARE_THEIR_INPUTS`** (`['action_keys_sha256']`) |
| challenger declares the baseline's, runs 3 | **REFUSED `REPLAY_ACTIONS_ARE_NOT_THE_DECLARED_POPULATION`** |
| **both left at the default `""`** | **ADMITTED `SEAM_IS_HONEST`**, `paths_identical: False` |

The mechanism DE built is correct and I confirm it works: the field is digested
by the same introspection as every other, the union-of-keys comparison catches
a difference, and an arm handed actions that are not its declared population
refuses before it replays anything. What is missing is that the declaration is
optional, and every other digested input is a required positional field. The
close is one character of API: give `action_keys_sha256` no default, or refuse
an empty one where the tape is checked.

**Step 6's full-pipeline freeze is not yet precondition-met.** No such
declaration exists on either executing ref today (the `*freeze*` files there are
the arm, code and population freezes from the other lane, plus DA 276's
step-6 abstention constraint — none of them a fair-value pipeline freeze), so
holding it unlanded is the right state.

## 0. All driven at the ref

| gate | module | cells | rc |
|---|---|---|---|
| 1 | `da_fair_value_gate1_labels.py` | 50/50 | 0 |
| 2 | `be_sigma_30m.py` | 27/27 | 0 |
| 3 | `de_fair_price_wrapper.py` | 29/29 | 0 |
| 4 | `de_fair_value_actions.py` | 22/22 | 0 |
| 5 | `de_fair_value_policy_seam.py` | 9/9 | 0 |
| 6 | `de_fair_value_replay_seam.py` | 23/23 | 0 |
| — | `de_gate_property_map.py` | 6/6 | 0 |
| — | `da_fair_value_ledger.py --falsify` | 26/26 | 0 |

Landing counts: chain 1 / runner 1 for all nine files; `mm-research` carries 0
of the seven executing modules and 1 each of `da_fair_value_gate1_labels.py`
and `de_canonical_action_population.py`.

(One correction to my own reading mid-round: a first pass counted 2 FAIL in the
ledger's falsifier. Both were the literal text `[PASS]/[FAIL] lines` quoted
inside passing cells' notes. 26/26, no red cell.)

## 1. What is genuinely closed

`ReplayInputs` now carries six fields, five digested:

    non_fair_value_params, initial_state, price_path, half_spread,
    action_keys_sha256            [ declared_snapshot_sha256 excluded ]

`run_arm(actions, value_of, inputs)` recomputes `action_keys_digest(actions)`
and refuses `REPLAY_ACTIONS_ARE_NOT_THE_DECLARED_POPULATION` before replaying
anything if it disagrees with the declaration. No list was touched to make the
comparison see the new field — `digested_field_names()` picks it up by
introspection, which is the property the round before last bought.

## 2. Nothing was over-constrained

| case | result |
|---|---|
| Identity vs itself | `NO_OP`, paths identical |
| real tape, value differs | `SEAM_IS_HONEST`, `paths_identical: False` |
| flat 0.99 tape | REFUSED `REPLAY_ARMS_DO_NOT_SHARE_THEIR_INPUTS` |
| pinned path | `REPLAY_OUTCOME_PATH_IS_PINNED` |

All four unchanged from the last two rounds. Adding the population field cost
the comparison nothing it needed to keep.

## 3. The map artifact and its driver

`de_gate_property_map.py` — 290 lines, a `MAP` of 29 properties over 4 gates and
a `resolve()` that drives each gate module, parses its `[PASS]` lines and checks
every entry. Its four refusals all fire when I mutate the map:

| mutation | result |
|---|---|
| cell name no falsifier emits | REFUSED `MAPPED_CELL_DOES_NOT_EXIST` |
| property mapped to an empty cell | REFUSED `PROPERTY_MAPS_TO_NOTHING` |
| a `universal` property given an `enumerated` cover | REFUSED `UNIVERSAL_PROPERTY_ENUMERATED_COVER` |
| a module with no falsifier | REFUSED `GATE_MODULE_HAS_NO_FALSIFIER` |

Baseline: 29 properties resolve, all to passing cells. This is a real
instrument, it is an artifact rather than a commit message, and the
narrower-than-the-property condition is mechanised rather than remembered. It is
the right answer to the class.

**Applied to the map's own gate-6 line, as instructed — and it reproduces the
finding.** The map now has an entry *"the action population is an input like any
other"*, marked `universal: False`, `coverage: enumerated`, mapped to the cell
*"two arms on DIFFERENT ACTION POPULATIONS are REFUSED"*. That cell — and both
of its neighbours — constructs `ReplayInputs(..., action_keys_sha256=
action_keys_digest(acts))`. No cell drives the default. So the cell covers the
*declared* population; the property says the population is an input **like any
other**, and every other input is shared unconditionally. The cell is narrower
than the property.

The driver agrees, once the label is right. Flipping that one entry to
`universal: True` — which is what "an input like any other" means, since the
other inputs admit no opt-out — makes `resolve()` refuse it:

    population line marked universal=True  ->  REFUSED UNIVERSAL_PROPERTY_ENUMERATED_COVER

**One limitation to record.** `universal` and `coverage` are author-set
attributes, so the narrowness check fires on a label. Relabelling the genuine
universal line `universal: False` makes its refusal disappear and the map still
resolves 29/29. That is the rule-16/42 shape inside the instrument built to
answer it. I do not think it is fully removable — some declaration of what a
property quantifies over is unavoidable — and the mitigation is real: both
labels are in the artifact where a reader checks them, which is why the gate-6
line above was findable. Worth a cell asserting that any property whose text
contains "every"/"any"/"all" is marked `universal`, which would have caught this
one mechanically.

## 4. DA's two ledger corrections — and independent convergence

`da_fair_value_ledger.py` at `0516c43`: `ref_head: 0516c43`,
`gates_satisfied: 5/6`, `no_labelled_score_permitted: True`,
`score_is_evidence_permitted: False`.

**Both corrections land.** Gate 4's fixture now carries provenance and the row
reads `SATISFIED`, 22/22, seven properties all true — the six that went false
last round were the probe breakage I diagnosed, and they are intact. Gate 6's
property list grew from eight to twelve.

**We now agree on the total, the row, and the reason.** DA's gate 6 reads
`CELLS_GREEN_BUT_PROPERTY_UNCOVERED`, 23/23 cells, with exactly two properties
false:

    i_differing_action_populations_refuse_BY_DEFAULT     False
    l_the_action_population_is_REQUIRED_like_price_path  False

That is my blocking finding, named twice, reached by DA's own adversarial probes
rather than from this filing — different instruments, same defect, which is the
only kind of agreement worth counting. The ten true entries include
`j_the_declared_digest_guard_WORKS_when_used` and `k_a_LIED_action_digest_
refuses`, which is the correct way to record a mechanism that works but is not
mandatory.

## 5. The six rows

At `0516c43`, every count and property driven this round.

| § | gate | landed | cells | status |
|---|---|---|---|---|
| 5.1 | settlement verifier | 1 / 1 | 50/50 | **SATISFIED** |
| 5.2 | sigma producer | 1 / 1 | 27/27 | **SATISFIED** |
| 5.3 | estimator wrapper | 1 / 1 | 29/29 | **SATISFIED** |
| 5.4 | forecast-action builder | 1 / 1 | 22/22 | **SATISFIED** |
| 5.5 | policy seam | 1 / 1 | 9/9 | **SATISFIED** |
| 5.6 | replay seam | 1 / 1 | 23/23 | **NOT SATISFIED** — the population declaration is optional |

**Five of six**, and DA's ledger says the same five, on the same row, for the
same reason.

## 6. Owed

- **Blocking:** `action_keys_sha256` required — no default, or an explicit
  refusal on empty — and a cell that drives the omission rather than the
  declaration.
- Non-blocking: a map cell asserting that a property whose text quantifies
  ("every", "any", "all") is marked `universal`; the `compare_arms` exclusion
  assertion and the non-JSON-native value refusal from REVIEW 240 §2.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); the
  round-boundary landing sweep as counts at fetched refs.
