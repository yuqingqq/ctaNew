# REVIEW 240 — the introspection fix holds; gate 6 is still open on the population

REV round 203. Filed 2026-09-11T20:16:43Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 20:06Z. Both
executing refs are `f31dff0`; the worktree was checked out detached there,
`git status --porcelain` empty, all seven lane modules byte-identical on
`origin/de-freeze-chain-v2` and `origin/be-build-runner`.

## THE VERDICT, FIRST

**It is not six of six. It is five of six.** Gate 6 is the one open row, and it
is open on the finding I filed at REVIEW 239 §3b, which this round's commit does
not address: **the action population is still an argument to `run_arm`, not a
field of `ReplayInputs`.** Driven at `f31dff0`:

    baseline:   6 actions, all priced 0.50 through Identity
    challenger: 3 actions, all priced 0.50 through Identity
    verdict          = SEAM_IS_HONEST
    inputs_identical = True
    digest_covers    = [non_fair_value_params, initial_state, price_path, half_spread]

The two arms consumed the same value on every action they share. Their paths
diverged because half the population was absent, and the seam reports that as
the value's doing. This is not a new class — it is the same input-outside-the-
contract shape as the flat tape, in the one remaining argument.

**Step 6's full-pipeline freeze is not yet precondition-met.** The remedy is the
move DE already made once: the action list (or its key-set digest) becomes a
field of `ReplayInputs`, and `run_arm` takes it from there. Everything else
below is closed, and three of the four ways I tried to defeat the new backstop
held.

## 0. All six driven at the ref

| gate | module | cells | rc | vs last round |
|---|---|---|---|---|
| 1 | `da_fair_value_gate1_labels.py` | 50/50 | 0 | — |
| 2 | `be_sigma_30m.py` | 27/27 | 0 | — |
| 3 | `de_fair_price_wrapper.py` | 29/29 | 0 | — |
| 4 | `de_fair_value_actions.py` | 22/22 | 0 | +2 |
| 5 | `de_fair_value_policy_seam.py` | 9/9 | 0 | — |
| 6 | `de_fair_value_replay_seam.py` | 20/20 | 0 | +4 |

Landing counts: chain 1 / runner 1 for all seven files; `mm-research` carries 0
of the five executing modules and 1 each of `da_fair_value_gate1_labels.py` and
`de_canonical_action_population.py`.

## 1. The falsifier I could not write before — driven, and it passes

`consumed()` iterates `dc_fields(self)` less `NOT_AN_INPUT`; `compare_arms`
takes the **union of the two arms' `consumed()` keys** and compares by
`.get(k)`, so there is one source and no second list. Driven myself with a
frozen subclass adding `latency_model_ms`:

    digested_field_names() -> (non_fair_value_params, initial_state,
                               price_path, half_spread, latency_model_ms)
    digests  e48261668cfb7110 / 10636ee5791c4dbb   DIFFER
    legs differing ONLY on it  -> REFUSED REPLAY_ARMS_DO_NOT_SHARE_THEIR_INPUTS
    legs agreeing on it        -> ADMITTED NO_OP

**Confirmed.** A field added tomorrow is digested with no edit anywhere. The
union-of-keys comparison also means a field present on one arm and absent on the
other is caught rather than skipped. This is a real fix at the shape, and it is
the correct answer to REVIEW 239 §3a.

## 2. Three more ways to defeat it — two held, one is visible-not-prevented

**(a) A field added after construction — HELD.** `object.__setattr__` puts an
attribute on a frozen instance; `dc_fields` does not see it and the digests stay
equal. This is **not** a defect: nothing in `run_arm`, `replay` or `run_seam`
reads a non-field attribute, so an attribute no engine consumes is correctly not
an input. I record it as tried and failed to break.

**(b) A repr collision under `default=str` — a narrow structural weakness, not a
demonstrated hole.** Two distinct objects with identical `__str__` inside
`non_fair_value_params` digest identically and compare as NO_OP. I could **not**
turn that into a silent behavioural divergence in this engine — the toy matcher
reads `half_spread`, `price_path` and `initial_state["inventory"]`, and every
colliding pair I could build either behaves identically (`Decimal("0.0")` vs
`"0.0"`, both `float()` to 0.0) or raises. So: the digest is over string
renderings, not values, which is weaker than it reads; the cheap close is to
refuse a non-JSON-native value in a digested field rather than stringify it.
**Not a gate failure.**

**(c) A subclass that widens the one exclusion — defeats the digest, and the
record says so.** `NOT_AN_INPUT` is a class attribute, so a subclass setting it
to `("declared_snapshot_sha256", "price_path")` drops the tape back out:

    Sneaky.digested_field_names() = (non_fair_value_params, initial_state, half_spread)
    real tape vs flat 0.99  ->  ADMITTED SEAM_IS_HONEST

But `compare_arms` emits `digest_covers` from `type(bi)`, so the record for that
comparison reads `['non_fair_value_params', 'initial_state', 'half_spread']` —
**`price_path` visibly missing.** A reader auditing the artifact sees the
coverage is short. That is DE's own "visible, not prevented" standard, met, and
it needs the caller to rewrite the guard's exclusion rather than to pass bad
data. I rank it a hardening item, not a gate failure: one line in `compare_arms`
asserting `type(bi).NOT_AN_INPUT == ReplayInputs.NOT_AN_INPUT` makes the
exclusion a property of the comparison rather than of whichever class was handed
in.

**The contrast that decides gate 6.** In (c) the record shows the gap. In the
population case `digest_covers` is **complete and correct**, `inputs_identical`
is True, and there is no field in the verdict from which a reader could learn
that the arms ran on different populations. A defeat you can see in the artifact
and a defeat you cannot are not the same finding.

## 3. The single exclusion is sound

`declared_snapshot_sha256` is read at exactly two places in the module — the
self-consistency check in `__post_init__` and the falsifier — and by nothing in
`run_arm`, `replay` or `run_seam`. It cannot hide a real input because no engine
consumes it, and excluding it is *necessary*: including a claim about the digest
in the digest makes the digest depend on itself. Both directions still drive: a
declared `"f"*64` refuses `REPLAY_DECLARED_SNAPSHOT_IS_NOT_WHAT_WAS_CONSUMED`, a
declared true digest is admitted. **Sound, and minimal.**

## 4. Nothing was over-constrained

| case | result |
|---|---|
| Identity vs itself | ADMITTED `NO_OP`, paths identical |
| real tape, value differs | ADMITTED `SEAM_IS_HONEST`, paths free, inventory −1.0 → +4.0 |
| flat 0.99 tape | REFUSED `REPLAY_ARMS_DO_NOT_SHARE_THEIR_INPUTS` |
| pinned path | `REPLAY_OUTCOME_PATH_IS_PINNED` |

The introspective `consumed()` did not make the comparison stricter than the
gate asks. The legs still move, and the seam still refuses a pinned outcome.

## 5. Gate 4's provenance residual — closed, with the limit in a field

Driven at the ref:

| population supplied | result |
|---|---|
| none | **REFUSED `CANONICAL_POPULATION_NOT_SUPPLIED`** (checked first) |
| bare list — REVIEW 239's fabrication | **REFUSED `CANONICAL_POPULATION_HAS_NO_PROVENANCE`** |
| dict with keys, no provenance | **REFUSED** same |
| provenance present but blank strings | **REFUSED** same |
| explicit `provenance` block | admitted, `supplied_as: "an explicit provenance block"` |
| the canonical builder's own shape | admitted, `supplied_as: "the canonical builder's own output"` |
| attributed but fabricated around the row | admitted, key digest recorded |

Absence and anonymity are separated, in that order, and they are genuinely
different wiring errors. The chain is real end to end, not by vocabulary:
`de_canonical_action_population.build_actions(rows, *, population, as_of,
source_identity)` takes all three as **required keyword-only** arguments and
returns them beside a bare `actions` key — which is exactly what
`canonical_keys` reads and what `population_provenance` accepts. So the builder
cannot emit an unattributed population and the consumer will not take one.

The last row is the honest limit, and DE puts it in the record rather than in a
message: `WHAT_THIS_BUILDER_CANNOT_DO` says it cannot verify the supplied
population is the true one — it refuses an unattributed one and digests exactly
the keys it used, so a fabricated population is **visible, not prevented**. With
`canonical_population_keys_sha256` in the report that is checkable by a cold
reader. **I read gate 4 as SATISFIED.**

One incidental, failing safe: a caller who puts the key list under `population`
(the provenance *name* field) rather than `actions` gets an empty key set, and
every row then contradicts its own flag — `REFERENCE_PATH_FLAG_CONTRADICTS_THE_
CANONICAL_POPULATION`, not a silent empty build. I hit it by accident writing
this round's probe.

## 6. Gate 3's cosmetic — closed

Three states, three labels, driven: `measured` (0.5), `DECLARED by the caller,
not measured` (declared equal), `unavailable -- a clock is absent, so nothing
was measured here` (either clock None). Undeclared equality still refuses.

## 7. The property-to-cell map

Every property DE maps resolves to a cell that exists at the ref and whose
printed evidence shows the property rather than its name. I checked all nine
gate-6 entries and all nine gate-4 entries against the 20 and 22 cells the
drives actually printed; three initial misses were multi-line f-string artefacts
of my substring match, not absences, and resolved on the printed list.

**But the map has the defect it was built to answer.** Its broadest gate-6 line
reads *"…and every other non-fair-value input → a NEW field is digested with NO
edit to any list."* That cell covers every non-fair-value input **that is a
field of `ReplayInputs`**. The action population is a non-fair-value input that
is not a field. A universally quantified property is mapped to a cell covering a
bounded class, and the gap is exactly where gate 6 fails. That is
`CELLS_PASSING_IS_NOT_PROPERTIES_COVERED` a fourth time, now inside the map.

The map is still the right instrument and should stay. Two amendments make it do
its job: state each property's **scope** where it is not literally universal,
and keep the map in an artifact rather than in a commit message. It presently
lives only in `Q-DE-369`'s message, so no reader resolves it and no cell drives
it — a map about coverage that is itself prose beside the code is the shape rule
13 warns about.

## 8. The six rows as I measure them

At `f31dff0`, every count and every property driven this round.

| § | gate | landed | cells | my status |
|---|---|---|---|---|
| 5.1 | settlement verifier | 1 / 1 | 50/50 | **SATISFIED** |
| 5.2 | sigma producer | 1 / 1 | 27/27 | **SATISFIED** |
| 5.3 | estimator wrapper | 1 / 1 | 29/29 | **SATISFIED** |
| 5.4 | forecast-action builder | 1 / 1 | 22/22 | **SATISFIED** |
| 5.5 | policy seam | 1 / 1 | 9/9 | **SATISFIED** |
| 5.6 | replay seam | 1 / 1 | 20/20 | **NOT SATISFIED** — the action population is outside the shared-input contract, and its absence is invisible in the record |

**Five of six.**

## 9. DA's ledger at this ref — 5/6, but on the wrong row

Driven at `f31dff0`: `gates_satisfied: 5/6`, gate 6 **SATISFIED**, gate 4
**`CELLS_GREEN_BUT_PROPERTY_UNCOVERED`** with six properties false. We agree on
the count and disagree on which row.

DA's six false gate-4 properties are a **probe breakage, not a property
regression**. Its fixture is `POP = [(SLUG, GEN)]` — a bare list — and DE's new
provenance requirement refuses exactly that. Driven, DA's own fixture shape:

    DA's fixture                -> REFUSED CANONICAL_POPULATION_HAS_NO_PROVENANCE
    the same probe with provenance -> OK n_actions=1 folded=1

The six properties are intact; the probe can no longer reach them. This is the
same failure DE had one round ago when tightening gate 4 broke the gates 5 and 6
fixtures — the tightening that closes a finding breaks the fixtures of everyone
who tests the tightened thing, and only a drive at the ref shows it. It is now
in the ledger, which is the instrument that is supposed to catch it.

Gate 6 reading SATISFIED is the substantive disagreement, and it is the same
shape as last round: DA's eight probed properties are all true and I confirm
them; the list has no entry for the shared action population, so the row cannot
fail on it. DA's own declaration says a gate with a green falsifier but an
uncovered declared property cannot read SATISFIED. That sentence applies to gate
6 here.

Credit where due: `THIS_DECLARATION_CARRIES_NO_ROW_STATUSES_ON_PURPOSE` is
right, and the reason given — a status written at 19:29Z was false fifteen
minutes later — is the correct lesson from this lane. The declaration pins only
its own instrument, `e1e8ac8409840454`, which recomputes exactly at the ref.

## 10. Owed

- **Blocking for six of six:** the action population into `ReplayInputs`.
- Hardening, not blocking: assert the exclusion tuple in `compare_arms`; refuse
  non-JSON-native values in digested fields instead of stringifying them.
- DA: the gate-4 probe fixture needs provenance, and the map wants an artifact.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); the
  round-boundary landing sweep as counts at fetched refs.
