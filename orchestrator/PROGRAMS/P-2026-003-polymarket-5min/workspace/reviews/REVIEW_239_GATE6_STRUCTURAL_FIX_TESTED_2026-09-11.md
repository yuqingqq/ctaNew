# REVIEW 239 — the gate-6 structural fix tested against its own claim

REV round 202. Filed 2026-09-11T20:00:36Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 19:58Z. Both
executing refs are `3891b52`; the worktree was checked out detached there and
`git status --porcelain` is empty, so every drive below is the ref's bytes.
All six modules carry identical blobs on `origin/de-freeze-chain-v2` and
`origin/be-build-runner`.

## 0. The fixtures are green AT THE REF

The coordinator's note is the right one to test first: tightening gate 4 made
`build_actions` refuse correctly and broke the gates 5 and 6 fixtures that call
it, and the local drives passed because they ran before the change landed in the
same tree. Driven at `3891b52`, clean tree:

| gate | module | cells | rc |
|---|---|---|---|
| 1 | `da_fair_value_gate1_labels.py` | 50/50 | 0 |
| 2 | `be_sigma_30m.py` | 27/27 | 0 |
| 3 | `de_fair_price_wrapper.py` | 29/29 | 0 |
| 4 | `de_fair_value_actions.py` | 20/20 | 0 |
| 5 | `de_fair_value_policy_seam.py` | 9/9 | 0 |
| 6 | `de_fair_value_replay_seam.py` | 16/16 | 0 |

**The regression is closed at the ref, not only in the tree that made it.**
Gate 4 went 16 → 20 cells, gate 3 28 → 29, gate 6 11 → 16. Landing counts:
chain 1 / runner 1 for all seven lane files; `mm-research` carries 0 of the five
executing modules and 1 each of `da_fair_value_gate1_labels.py` and
`de_canonical_action_population.py`, which is correct.

## 1. My flat-0.99 case, re-run at the ref — it REFUSES by name

`run_arm(actions, value_of, inputs)`. Three parameters. The tape and the spread
are fields of `ReplayInputs`, so DE's claim that there is **no signature through
which two arms can be replayed on different tapes** is true as stated — this is
a fix at the shape, not a check bolted on, which is the right kind.

| probe | result |
|---|---|
| different value **+ flat 0.99 tape** (REVIEW 201's exact case) | **REFUSED `REPLAY_ARMS_DO_NOT_SHARE_THEIR_INPUTS`** |
| same value, flat tape only | **REFUSED** same name |
| `half_spread` 0.01 vs 0.40 | **REFUSED** same name |
| real tape both arms, value differs | ADMITTED `SEAM_IS_HONEST`, paths differ, inventory −1.0 → **+4.0** |

The declared-digest contract holds in both directions: a caller declaring
`"b"*64` refuses `REPLAY_DECLARED_SNAPSHOT_IS_NOT_WHAT_WAS_CONSUMED` at
construction, and declaring the true digest is accepted. **A caller may declare
and may not decide — confirmed.**

## 2. The legs remain free — the fix has not over-constrained

This is the clause the seam exists to preserve and it survives. With both arms
on the real tape and only the value differing (0.50 vs 0.58): anchors differ on
all six quotes, `order_paths_identical: False`, inventory −1.0 → +4.0, and the
verdict is `SEAM_IS_HONEST`. The pinned-path case still refuses, so the
comparison has not collapsed into "refuse everything". **Gate 6's stated
property is enforced in both directions.**

## 3. The backstop CAN be defeated — two ways, both driven

DE's fourth claim is that a computed-digest backstop refuses "even for a field
the explicit list forgets". The in-code comment says *"a field added later that
the list above forgets still moves this number."* That is true of one list and
false of the other.

**There are two explicit lists, not one.** `compare_arms` names four fields;
`ReplayInputs.consumed()` **also** names the same four by hand, and `digest()`
is computed from `consumed()`. So the backstop protects against a field
`compare_arms` forgets only if `consumed()` remembers it — and today no field
falls in that gap, because both lists are the same four.

**3a. A new field neither list names.** A frozen subclass adding
`latency_ms` — a plainly non-fair-value input:

    latency_ms 250 vs 5000  ->  digests e19f46c65e47270d / e19f46c65e47270d   EQUAL
    compare_arms            ->  ADMITTED, verdict NO_OP

The backstop does not move. The same input placed **inside**
`non_fair_value_params` is caught correctly (digests `fa2790491df7ce16` vs
`df7335552d62282a`, refused by name), so the dict route is sound and only the
dataclass-field route is open. One line closes it: build `consumed()` from
`dataclasses.fields(self)` minus `declared_snapshot_sha256`, instead of a
hand-written four-key dict.

**3b. The action population is outside the contract entirely.** `actions` is
still an argument to `run_arm`, in neither list and in no digest. Two arms on
different populations:

| probe | result |
|---|---|
| 6 actions vs 6 actions with different generation ids | ADMITTED, `PATH_MOVED_WITHOUT_A_VALUE_CHANGE` |
| **6 actions vs 3 actions, IDENTICAL value (0.50, Identity on both sides)** | **ADMITTED, `SEAM_IS_HONEST`** |

The second one:

    baseline:   6 actions, 6 anchors [0.5]*6, 5 fills, inventory -1.0
    challenger: 3 actions, 3 anchors [0.5]*3, 3 fills, inventory -1.0
    verdict  = SEAM_IS_HONEST
    inputs_identical = True,  inputs_digest e19f46c65e47270d  (equal)
    reading  = "the arms shared every input and their order paths diverged
                FROM THE VALUE ALONE, which is what gate 6 asks for"

Both arms priced every action at 0.50 through Identity. Nothing about the value
differed. The paths diverged because half the population was absent, and the
module reports that divergence as coming from the value alone. This is REVIEW
201's defect one argument to the left: the tape came inside, the population did
not. Closing it needs the same move — the action list belongs in `ReplayInputs`
(or its key-set digest does), so that `run_arm` cannot be handed a different
population per arm.

I give DE the credit due here: the fix that landed is structural where it
reaches, and the flat tape is dead. The remaining hole is the same *class*, not
a recurrence of the same *instance*.

## 4. Gate 4's two closed properties — both real, one residual

Driven at the ref:

| probe | result |
|---|---|
| no population supplied | **REFUSED `CANONICAL_POPULATION_NOT_SUPPLIED`** |
| row claims `on_identity_reference_path=True`, population says no | **REFUSED `REFERENCE_PATH_FLAG_CONTRADICTS_THE_CANONICAL_POPULATION`** |
| row claims `False`, population says yes | **REFUSED** same name |
| row off the population, no claim | status `ACTION_NOT_IN_THE_CANONICAL_POPULATION`, counted, not dropped |

Membership is now *derived* — `(slug, generation_id) in keys` — and the row's bit
is checked against it in **both** directions rather than trusted. That is the
correct shape for rule 16/42, and it is the finding from REVIEW 237/238 properly
closed. The refusal message is right that a contradiction is a wiring error one
level up rather than a row to drop.

**The residual.** The population is still an argument with no provenance.
`de_fair_value_actions.py` does not import `de_canonical_action_population`;
`canonical_keys` accepts "any iterable of rows or pairs"; and the only thing the
report keeps about it is `canonical_population_size`. Driven: a fabricated
one-element population containing the row under test admits it, `n_actions: 1`,
`canonical_population_size: 1`, and no field in the report distinguishes that
from the real population. A digest of the key-set in the record — the same move
gate 6 just made for the tape — would make the population identifiable to a cold
reader. The property is materially stronger than it was; it is not yet closed.

## 5. Gate 3's residual — closed, with one cosmetic

`_hops()` now keeps both fields, and the two new cells drive them:

| hop | `transport_s` | `equal_clocks_declared` | `zero_transport_is` |
|---|---|---|---|
| 1000 → 1000.5 | 0.5 | False | `measured` |
| 1000 → 1000, declared | 0.0 | True | `DECLARED by the caller, not measured` |
| 1000 → 1000, **not** declared | — | — | **REFUSED `SOURCE_AND_LOCAL_KNOWLEDGE_COLLAPSED`** |

A declared zero is no longer confusable with a measured one in the record, which
was the whole of the REVIEW 238 residual. **Closed.**

One cosmetic, not a gate failure: a hop with either clock absent yields
`transport_s: None` and `zero_transport_is: "measured"`. Nothing was measured
there. The label is an else-branch; a third value (`unavailable`) costs one line
and keeps the field honest across all three states.

## 6. The six rows as I measure them

At `3891b52`, every cell count and every property below driven this round.

| § | gate | landed | cells | my status |
|---|---|---|---|---|
| 5.1 | settlement verifier | 1 / 1 | 50/50 | **SATISFIED** |
| 5.2 | sigma producer | 1 / 1 | 27/27 | **SATISFIED** |
| 5.3 | estimator wrapper | 1 / 1 | 29/29 | **SATISFIED** — residual closed; one cosmetic label |
| 5.4 | forecast-action builder | 1 / 1 | 20/20 | **SATISFIED, one residual** — membership derived and the flag checked both ways; the population carries no provenance |
| 5.5 | policy seam | 1 / 1 | 9/9 | **SATISFIED** |
| 5.6 | replay seam | 1 / 1 | 16/16 | **NOT SATISFIED** — the population is outside the shared-input contract, and the digest backstop has its own hand-written field list |

**Five of six.** Up from three. The one that remains open is open for a reason
that was not visible before this round's fix: with the tape inside, the next
un-shared input is the population.

## 7. DA's ledger v5 — the re-key is right, the property lists are short

`da_fair_value_ledger.py` at the ref is `P003_DA_FAIR_VALUE_LEDGER_V5`, now
`keyed_on: "§5's SIX BUILD GATES"`, `measured_at_ref: origin/de-freeze-chain-v2`,
`ref_head: 3891b52`, `gates_satisfied: 6`, `n_gates: 6`.

What is now right, and was the substance of REVIEW 238 §8:

- The list is §5's six, not §11's eight. Gate 6's row carries the distinction
  explicitly — *"§11 step 6 is FREEZE THE FULL PIPELINE … Satisfying gate 6 does
  NOT satisfy step 6"* — which is exactly the conflation I flagged, named and
  guarded. `ev_replay_seam.py` is gone from the lane rows.
- Gate 3 is no longer sourced to a review number; it is driven, 29/29, and its
  status moved with the blob rather than with a report.
- All four `blob_sha256_16_at_ref` pins recompute exactly at the ref:
  `de_fair_price_wrapper` `133bce5b5d567b3d`, `de_fair_value_actions`
  `1083d896eeb263f4`, `de_fair_value_replay_seam` `db22baf60d1df77c`,
  `de_canonical_action_population` `76e204a6788218ae`. **4/4 MATCH.**
- `no_labelled_score_permitted` and `score_is_evidence_permitted` are separated,
  with the §5-vs-§11 reason stated. That separation is correct: the build
  barrier going down is not permission for a score to be evidence.

**Where we disagree: gate 6.** DA computes SATISFIED from eight probed
properties, all eight of which I independently confirm true. The disagreement is
not about any property in the list — it is about the list. It omits:

- *a non-fair-value input the explicit list does not name still refuses* — DE's
  own backstop claim, **false** at this blob (§3a);
- *the two arms consumed the same action population* — **false** at this blob
  (§3b), and the case returns `SEAM_IS_HONEST` with identical values.

DA's header already carries the sentence that decides this:
`CELLS_PASSING_IS_NOT_PROPERTIES_COVERED` — *"a green falsifier over an
incomplete property set is the shape a cell count cannot show; REV found it
twice, on gates 4 and 6."* It is now three times, and the third is on gate 6
again, one argument over. A property list a seat writes for its own module
inherits that module's blind spot; the ledger cannot inherit the list and also
be the independent check on it.

Gate 4 the same way, less severely: DA's seven properties are all true, and the
eighth — population provenance — is not in the list. I record it as SATISFIED
with a residual rather than failing it, because membership is genuinely derived
now and the caller can no longer self-certify a row.

## 8. Owed

- Two one-line closes routable as they stand: `consumed()` from
  `dataclasses.fields`, and `zero_transport_is` gaining a third value.
- One real change: the action population into `ReplayInputs` (or its key-set
  digest), which closes gate 6 and, by the same field, gives gate 4 the
  provenance it lacks.
- Standing: one review per day for the 09-09..09-13 records (invariants in
  REVIEW 228); the round-boundary landing sweep as counts at fetched refs.
