# REVIEW 238 — the six gates driven at the fetched refs, and an independent ledger

REV round 201. Filed 2026-09-11T19:36:32Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Every drive below ran in
`~/ctaNew-wt-rev` checked out **detached at `70ab5b8`**, the tip of both
`origin/de-freeze-chain-v2` and `origin/be-build-runner` (they are the same
commit), after a fetch at 19:29Z. Every landing figure is a count at a fetched
ref, never a sha in prose.

## 0. Refs and blob identity

`origin/de-freeze-chain-v2` and `origin/be-build-runner` are both
`70ab5b893d312b993a44a1a359c6263cb8add2e7`. The six gate modules carry
identical blobs on both:

| module | blob | chain | runner | mm-research |
|---|---|---|---|---|
| `da_fair_value_gate1_labels.py` | `1f85a1659466` | 1 | 1 | 1 |
| `be_sigma_30m.py` | `778594299f48` | 1 | 1 | **0** |
| `de_fair_price_wrapper.py` | `bbe94f5b629e` | 1 | 1 | **0** |
| `de_fair_value_actions.py` | `9cb7c1dec47c` | 1 | 1 | **0** |
| `de_fair_value_policy_seam.py` | `bc4e3c37071f` | 1 | 1 | **0** |
| `de_fair_value_replay_seam.py` | *(new at `70ab5b8`)* | 1 | 1 | **0** |

Five of six are absent from the non-executing fork, which is correct.

## 1. Gate 2 re-driven against the pushed blob

`be_sigma_30m.py` at both required refs is blob `778594299f48`, sha256
`8d7a2448937e0fd3…` — **byte-identical to the blob I drove at REVIEW 231**, and
identical to the digest DA's ledger pins for step 2. Driven at the ref:

    {"falsifier": "be_sigma_30m", "n": 27, "failed": 0}     rc 0

27 PASS lines, 0 FAIL. **Gate 2 reads SATISFIED**, and the re-drive is against
the pushed bytes, not a working copy.

## 2. Gate 3 — my REVIEW 236 defect is CLOSED, with one residual

The wrapper moved: `5d8150e87792` → `bbe94f5b629e`, 670 → 737 lines, 24 → 28
cells. **28/28, rc 0.** The collapse defect is fixed at the predicate:

    if (... and self.local_receipt == self.source_as_of
            and not self.equal_clocks_declared):
        raise WrapperRefused(f"REFUSED {TIMESTAMPS_COLLAPSED}: ...")

and two new cells drive it — a collapsed pair refuses by name, an inverted pair
refuses under the *other* name. That is the finding routed from REVIEW 236,
closed properly.

**The residual.** The refusal text ends `"...must say so with
equal_clocks_declared=True, which is recorded."` It is **not recorded**.
`equal_clocks_declared` appears at five lines in the module (field, predicate,
message, and two cells) and nowhere else in the lane. The only serialiser for a
`Stamped` is `_hops()`, which emits exactly four keys:

    {"source", "source_as_of", "local_receipt", "transport_s"}

So a feed that declares equal clocks is admitted and lands in the record as
`transport_s = 0.0`, **indistinguishable from a measured zero transport**. The
gate's own words — "timestamps remain distinct at every hop" — can still be
violated by a declared exception that leaves no trace for a cold reader. One key
in `_hops()` closes it.

## 3. Gate 4 — 16/16, four of six properties (unchanged, now checked not read)

16/16, rc 0. Enforced: the key `(coin, slug, generation_id, decision_recv_ns)`;
same key + same consumed probability folds quote sides into a sorted union; same
key + different value raises `FORECAST_ACTION_DUPLICATE_KEY` naming both values;
`ACTION_IS_NOT_ON_THE_NEUTRAL_IDENTITY_REFERENCE_PATH`; one epsilon both sides.

The two it **cannot** enforce, verified rather than inferred this round:

1. **The reference-path flag is trusted.** `on_identity_reference_path` appears
   at line 103 (`if not row.get(...)`) and line 275 (echoed into the record).
   It is read, never derived. The refusal fires on the *label*, so it cannot
   distinguish a genuinely off-path row from a caller that set the bit.
2. **No join to the canonical population.**
   `de_canonical_action_population.py` is present on all three refs, and
   `de_fair_value_actions.py` imports only `da_fair_price_identity` and
   `de_fair_price_wrapper`. There is no import of it and no reference to it.
   "One row per actual fair-value consumption decision" is therefore a property
   of whatever list the caller passes, not a property this module checks.

Both are rule-16/42 cases: the control fires on the label, and nothing in the
lane yet fires on the property.

## 4. Gate 5 — 9/9, all four §5 properties driven

9/9, rc 0. The four §5 clauses are each a cell, and each cell is a value, not a
name: Identity-for-Identity is bit-identical including the trajectory digest
(`3f78c474621363d8`); a non-Identity value moves 4 of 4 anchors and the posted
pair (bid 0.49 → 0.69); an absent challenger falls back and is **counted**
(`fell_back: 4`, `fallback_share: 1.0`) with anchors identical to the Identity
run; changing only the estimator metadata is detected as
`FAIRPRICE_SEAM_IS_DECORATIVE_VALUE_UNCHANGED`. **Gate 5 reads SATISFIED.**

## 5. Gate 6 — landed at `70ab5b8`, 11/11, and NOT satisfied

`de_fair_value_replay_seam.py`, 284 lines, imports only the policy seam.
11/11, rc 0. It enforces the three declared shares by name
(`non_fair_value_params`, `input_snapshot_sha256`, `initial_state`), refuses a
replay that pins the challenger to the baseline's path, refuses an arm with no
path, and gives identical-values-different-paths its own verdict rather than a
pass. The engine is a 15-line deterministic matcher written in the module, and
the docstring says so — an honest fixture, not the production quoter.

**The contract does not cover the tape.** `ReplayInputs` holds the three shared
fields; `price_path` and the *effective* `half_spread` are arguments to
`run_arm`, outside it, and `compare_arms` never sees them. Driven, three cells
of my own at the ref:

| probe | result |
|---|---|
| same `ReplayInputs`, different `price_path` | ADMITTED, `PATH_MOVED_WITHOUT_A_VALUE_CHANGE`, `inputs_identical: True` |
| same declared `non_fair_value_params`, `half_spread` 0.01 vs 0.40 (bid0 0.49 vs 0.10) | ADMITTED, `PATH_MOVED_WITHOUT_A_VALUE_CHANGE`, `inputs_identical: True` |
| **different value AND a different tape** | **ADMITTED, `SEAM_IS_HONEST`**, `inputs_identical: True`, `inputs_digest 7942427c54ae742f` equal on both arms, inventory −1.0 → −6.0 |

The third is the one that matters. A challenger replayed on a flat 0.99 tape
against a baseline on the real tape is reported as an honest seam whose value
change moved inventory by five units. Every unit of that difference is the tape.
The gate proves the two arms **declare** the same inputs; it does not prove they
**ran on** the same inputs. `input_snapshot_sha256` is a caller-supplied string
the module never computes from anything — the fixture passes `"a" * 64`.

Closing it is small: move `price_path` and `half_spread` inside `ReplayInputs`,
or have `run_arm` derive the tape from the snapshot whose sha it is given, so
that the digest is a claim about bytes the engine actually consumed.

## 6. No labelled score is readable by construction — confirmed by trying

Across all three of `de_fair_value_actions.py`, `de_fair_value_policy_seam.py`
and `de_fair_value_replay_seam.py`, at the fetched blobs:

    open( = 0   read_text = 0   json.load( = 0   glob = 0
    /resolution/i = 0   /winner/i = 0   import of any settlement module = 0

The one `settle` hit in the actions module is a comment: *"Nothing here loads an
outcome: step 6 owns it."* The scorer's signature is

    score_actions(actions, identity_at, challenger_at, outcomes, *, eps=1e-06)

with `outcomes` positional and **no default**. Driving it:

- empty `outcomes` → `ACTION_OUTCOME_NOT_SUPPLIED: s1 has no supplied outcome`
- omitting `outcomes` → `TypeError: score_actions() missing 1 required positional argument: 'outcomes'`

There is no path by which these modules obtain a label. **DE's claim holds, and
it holds by construction rather than by discipline.**

## 7. The six rows as I measure them

§5's gates, each row a drive at `70ab5b8`:

| § | gate | landed (chain/runner) | driven | my status |
|---|---|---|---|---|
| 5.1 | settlement verifier | 1 / 1 | 50/50, rc 0 | **SATISFIED** |
| 5.2 | sigma producer | 1 / 1 | 27/27, rc 0 | **SATISFIED** |
| 5.3 | estimator wrapper | 1 / 1 | 28/28, rc 0 | **SATISFIED, one residual** — the equal-clocks exception is not recorded |
| 5.4 | forecast-action builder | 1 / 1 | 16/16, rc 0 | **NOT SATISFIED** — 4 of 6 properties; the path flag is trusted and there is no population join |
| 5.5 | policy seam | 1 / 1 | 9/9, rc 0 | **SATISFIED** |
| 5.6 | replay seam | 1 / 1 | 11/11, rc 0 | **NOT SATISFIED** — the tape is outside the shared-input contract |

**Three of six satisfied.** Every cell count above came from a drive this round,
not from a report.

## 8. Independent check on DA's declared ledger

`da_fair_value_ledger.py` at the ref computes eight rows and declares
`gates_satisfied: 2`. Four observations, in order of weight.

1. **It is keyed on §11, not §5.** DA's eight rows are the plan's §11
   implementation order. §11 and §5 run parallel for 1–5 and **diverge at 6**:
   §11.6 is "freeze the full pipeline and both candidate identities"; §5.6 is
   the replay seam. A reader taking `gates_satisfied` as a count of §5's build
   gates is reading a different list.
2. **Step 6's path is from another programme.** DA maps step 6 to
   `live/pm_research/ev_replay_seam.py`. That module's own header says it is the
   EV-Replay policy seam for the harmful-fill lane (R-126, `EV_REPLAY_PLAN.md`),
   one level above `ev_replay.py`. It is neither a pipeline freeze nor §5's
   replay seam. It is also the only step-6 path present on `origin/mm-research`,
   which is why the row counts 1 on all three refs. Meanwhile the real §5.6
   module, `de_fair_value_replay_seam.py`, sits in DA's own
   `unattributed_lane_files` on both executing refs. The guard fired; nobody
   read it as a mis-mapping. DA's ledger names this exact risk in its own text —
   *"a status keyed on a path nobody declared measures the guess, not the
   lane"* — so this is the ledger's declared weak link doing what it warned of.
3. **Step 3 is stale, and its source is a report.** DA records
   `LANDED_BUT_FAILING` with `source: "REVIEW 199 / DE cells: ... the
   collapsed-timestamps clause not enforced"`. The clause **is** enforced at
   `bbe94f5b629e` and 28/28 drive green. The ledger header asserts
   `no_status_copied_from_a_report: true`; step 3's behaviour field cites a
   review by number. Those two cannot both be right, and the row is wrong in the
   direction the assertion was meant to prevent.
4. **Steps 4, 5 and 2 check out.** Steps 4 and 5 are
   `LANDED_BEHAVIOUR_UNVERIFIED`, which was accurate when written and which this
   filing supplies (16/16 and 9/9). Step 2's `driven_against_blob_sha256`
   `8d7a2448937e0fd3` matches `sha256(be_sigma_30m.py)` at the ref exactly, and
   `REVERTS_IF_THE_BLOB_MOVES: true` is the right shape for a driven verdict.
   That row is the model the other seven should follow.

Net: DA's `gates_satisfied: 2` and my three-of-six are not in conflict on any
shared row — they count different lists, and the only substantive disagreement
is step 3, where the ledger is behind the blob.

## 9. Owed

- Gate 3's residual and gate 6's tape gap are routable as they stand; neither
  needs a new instrument, only a field and a dataclass move.
- Standing: one review per day for the 09-09..09-13 records (invariants in
  REVIEW 228); the round-boundary landing sweep as counts at fetched refs.
