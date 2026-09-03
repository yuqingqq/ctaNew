# Re-review — BE round 17, the fix batch for the three HIGHs

reviewer: pm-codex · filed 2026-09-03T06:50Z · pinned tip **`70e58cb`** (BE17 at `5565e39`, row `e78a710`)
executed in `~/ctaNew-wt-rev` at the tip. **No seal opened.** `be_forward_day.py` read, never run. Nothing written under `data/`. No other seat's worktree read.

Suites first, both launchers, all green: `be_forward_metric` **87**, `be_operating_point` **17**, `be_forward_recon` **24**, `be_forward_futility` **26**, `harmful_forward_scorer` **75**, `be_seal_relocate` **41**, `phase2_increment_null` green — rc 0 everywhere.

## Overall: **AMEND. The standing rule stays in force — no forward day is scored.**

**BEM-R3 is closed and closed well. BEM-R4, R5, R7 and R8 are closed and I drove each repair in both directions. BEM-R1 and BEM-R2 are NOT closed** — I reached the decision metric twice more, once with a mapping no fence ever saw and once through the real committed declaration with its thetas swapped for the retrospective cutoff of the scored rows. Both times `require_fenced_op` reported `causal_verified_against_scored_population: **True**`.

The shape has changed and it is worth naming, because it is the same shape twice: **round 15's fences were real but off the path; round 17's fences are on the path but do not bind the thing they are about.** The theta now has to arrive in a particular container; nothing checks that the number inside it came from where the container says.

---

### BE17-R1 — HIGH — `require_fenced_op` gates on a token it never verifies, and an undeclared split reads as verified-causal

`require_fenced_op` refuses a float and refuses a mapping with no `_operating_point_token`. That is the whole gate. It computes `want = _op_token(...)`, **reports** `token_recomputed` and **never raises on it**; and it does not re-run `require_operating_point`. So any mapping with a truthy token string passes.

Driven at the tip, rows dated 2026-08-29, theta = the top-1 score of the scored ranking:

```
forged = {"_operating_point_token": "anything-truthy",
          "theta_frozen": {"10%": 0.99},          # read off the scored rows
          "form": "FROZEN_FROM_TRAIN_QUANTILE"}   # no provenance, no split

require_fenced_op(forged, "10%", rows=rows)
  -> theta 0.99 | token_present True | token_recomputed False
     scored_split_overlap []  |  causal_verified_against_scored_population: TRUE

increment(rows, cand, inc, op=forged, budget_key="10%")
  -> causal True | theta_declared 0.99 | increment_cents 18.0 | unit ACTION
```

Two defects compound. `declared = set((op.get("derived_from_split") or {}).get("days") or ())` — an op that declares **no** split yields an empty set, an empty intersection, and `causal_verified_against_scored_population = (overlap == [])` → **True**. Absence reads as a pass; SEAT_PROTOCOL 11 is explicit that it must not. And the token, the only thing standing between a hand-built dict and the estimand, is checked for truthiness alone.

> **AMEND clause.** `require_fenced_op` RAISES on `token_recomputed False`, and RAISES when `derived_from_split.days` is empty or absent — an operating point that cannot be checked against the scored population must refuse, not report itself verified. `causal_verified_against_scored_population` may only be True when a non-empty declared split was compared and did not intersect.

### BE17-R2 — HIGH — the coordinator's question (2), answered YES: a mislabelled split passes

Nothing binds `theta_frozen` to `provenance.rows_artifact`. The three provenance checks are individually sound and mutually unconnected: the artifacts are **rehashed** (real bytes, real shas) but never linked to the numbers; `theta_map_sha16` digests the map **against itself**; `derived_from_split.days` is **free text** nobody derives from the rows artifact.

Driven against the real committed declaration — I changed one field and recomputed one digest, exactly as `be_operating_point.build_declaration` does:

```
A) the GENUINE op (rehashes the 1.24 GB rows artifact)
   theta {'5%': 1.0845…, '10%': 0.7230…, '15%': 0.5250…}   split days ['2026-08-24','2026-08-25']

B) ATTACK: theta_frozen replaced by the retrospective top-k cutoff of the SCORED rows
   theta {'5%': 0.99, '10%': 0.99, '15%': 0.99}
   derived_from_split, rows_artifact, fit_artifact  LEFT EXACTLY AS THE REAL DECLARATION'S
   only provenance.theta_map_sha16 recomputed (as the builder does)

   require_operating_point   ACCEPTED
   provenance_verified       rows_artifact: rehash | fit_artifact: rehash | theta_map_sha16 ok
   require_fenced_op         scored_split_overlap []   causal_verified_against_scored_population: TRUE
   increment()               BY_THRESHOLD | causal True | theta 0.99
                             candidate_net_cents 18.0 | incumbent 0.0 | increment_cents 18.0
```

The overlap check is a real improvement — it does close the route where the derivation split is stated honestly. It cannot close the route where it is not, because the split is asserted rather than derived, and the fence has no way to ask the rows artifact which days it contains.

> **AMEND clause.** Bind the numbers to the bytes. Either (i) the fence RECOMPUTES the quantile map from `provenance.rows_artifact` restricted to `derived_from_split.days` and refuses unless it reproduces `theta_frozen` — the artifact is already hashed, so the recomputation is over known bytes; or (ii) `derived_from_split.days` is **derived from the rows artifact** by the builder and recorded with a digest over `(rows_sha, days, theta_map)` that the fence recomputes, so a day list that does not describe the rows cannot be written. Until one of them exists, the receipt must not print `causal_verified_against_scored_population: True`; the honest field is `declared_split_does_not_intersect_scored_population`, which is what is actually computed.

### BE17-R3 — MEDIUM — `token_recomputed` is False for the genuine op, and asserted nowhere

Case A above: the op straight out of `require_operating_point`, unmodified, gives `token_recomputed: **False**`. The cause is in the code: the token is `_op_token(decl)` over the declaration's raw `provenance`, while `want` is rebuilt from `op["provenance_verified"]`, which carries the added `verified_by: "rehash"` keys. The two can never agree. `grep token_recomputed live/pm_research/*.py` returns exactly one line — the one that computes it. It is asserted in no suite and raised on nowhere.

So the field is False in the honest case and True in no case, and it will appear in receipts. A reader who takes it at face value reads tampering into every legitimate run.

> **AMEND clause.** Compute `want` over the same object the token was taken over (or store the token's input digest alongside it), assert `token_recomputed is True` on the genuine op in the suite, and raise on False — see BE17-R1.

### BE17-R4 — MEDIUM — BEM-R6 is not closed; the fix moved the guarantee rather than adding one

BE applied my one-line fix literally, discovered that it makes the whole-block comparison False by construction (a commit cannot contain its own sha), split the guard in two, and **documented both honestly** in the receipt (`block_definition_changed_this_round`) and in the suite (`ok(tu["unchanged"] is False, …)`). That candour is why this is MEDIUM. But the guarantee is now weaker in one place and no stronger in the other:

- `tolerances_unchanged_since()` is **False at the honest tip** (verified: 26 declaration lines, pointer included, `unchanged: False`). A field that is False when nothing is wrong cannot signal that something is.
- `tolerance_values_unchanged_since()`, the substantive guard, still defaults to `TOLERANCE_DECLARING_COMMIT` — a constant in the file it guards, still absent from the value-lines it compares. Re-driven in a throwaway git tree, same attack as before:

```
TOL_CENTS_ABS running: 1000000.0   (declared 1e-06)   pointer repointed at the widening commit
tolerance_values_unchanged_since()  -> checked True, unchanged: TRUE
tolerances_unchanged_since()        -> checked True, unchanged: False   (False anyway, by construction)
values guard vs the TRUE first commit -> False
```

A twelve-order-of-magnitude widening still reports unchanged, and the guard that was meant to catch the repointing is now a constant.

Credit where it is due: the replacement known-bad **drives** the guard against a tampered copy on disk instead of inferring the outcome from two strings — that half of BEM-R6 is properly closed.

> **AMEND clause.** Stop declaring the anchor in the file it guards: **compute** it. The guard reports the EARLIEST ancestor commit whose value-lines equal today's (`git rev-list` + `git show` over the file's history), and refuses if that commit is later than the first recorded `--reconcile` run. Then repointing a constant changes nothing, because no constant is consulted.

### BE17-R5 — LOW-MEDIUM — `cutoff_depends_on_scored_data` reports `forward_eligible: True` for a theta read off the scored data

Driven with the forged op of BE17-R1: holding the declared inputs fixed and changing only the scores, the cutoff does not move — because a frozen number does not move — so the instrument returns `cutoff_moved_with_the_data: False`, `is_bridge_to_development_number: False`, **`forward_eligible: True`** for a theta that was taken off the very rows being scored.

The arithmetic is right and the docstring is right about what it measures. The field NAME is not: this instrument can *falsify* BY_COUNT, it cannot *certify* BY_THRESHOLD, and `forward_eligible` reads as a certificate.

> **AMEND clause.** Rename to `cutoff_is_a_function_of_these_scores` / `not_read_off_these_scores_at_evaluation_time`, and state in the return that a frozen theta passes this check by construction whatever its derivation — the derivation is BE17-R2's job, not this one's.

---

## Closed, and driven closed by me

| was | now | how I checked |
|---|---|---|
| **BEM-R3** | **CLOSED** | `require_arm_identity` REFUSES a nonexistent path ("Nothing was hashed") and REFUSES a real path with a wrong sha, and ADMITS the declared candidate (`cfc454d6…`, `PM_PLUS_FINE (reduced fine)`) — both directions. `load_frozen()` with no `expect` REFUSES by name (`NotFrozen`, "no expected identity"). **There are FOUR production call sites, not three** — `be_forward_day.py:993` (`score_rows`), `:1362` (`run_forward_day`), `forward_dry_run.py:59` (`main`) and a new one at `be_operating_point.py:134` (`score_training_split`) — and **all four bind `expect=`**. The replaced control at `harmful_forward_scorer.py:606-614` **FIRES**: it calls `load_frozen()`, requires `NotFrozen`, and asserts the message names the repaired property, not the absence of the old string |
| **BEM-R4** | **CLOSED** | `can_ever_say_usable: True`. Verdicts now derive from the input: pair form → unusable; the evaluator's own rows → **usable**; 7-column feed form → **usable**; an unrecognised width (6) → unusable, i.e. an unknown shape refuses rather than guesses |
| **BEM-R5** | **CLOSED, and enforced in code** | the restated reason is not just prose: a shape with `gen` added but **no `t_start`** still reads `usable_for_action_estimand: False`. That is exactly the point — repairing reason 1 alone leaves the estimand undefined — and it is now a predicate, with the three consumers (`harmful_action_eval.py:67`, `:13`, `:35`) cited in the return |
| **BEM-R7** | **CLOSED** | `NOT_RECONCILED_HERE` now carries `the_null_is_a_delegation`, `new_path_functions_NOT_exercised` = `['increment','evaluate_arm','reduce_window','feed_row_to_eval_row','exclusions','cluster_disclosure']`, and `increment_cannot_be_reconciled_here`. The summary adds `n_holm_predicates 24`, `n_predicates_covered_by_all_hold 60` and a `counts_note` |
| **BEM-R8** | **CLOSED** | `sign_flip_p` sorts at consumption; two insertion orders now agree (0.8902743142144638 both). On the **real published increments** where I measured 0.0718562874 vs 0.0938123752 last round, both orders now return **0.0718562874251497** |
| BY_COUNT | **fenced correctly** | refuses without `bridge_to_development_ack`; with it, the cell carries `causal: False`, `pairing_role: REPORTED BESIDE …`, `theta_source: READ OFF THE RANKING OF THE DATA BEING SCORED`, and the ack itself. `pooled_increment` REFUSES to pool across conventions |
| BE's own wiring-predicate note | **genuinely fixed** | `_emits_feed` is scoped to `{build_and_score, run_forward_day}`, not the file, so the selftest's own call is no longer counted; it is driven positively on the real source and negatively on a source with the emission removed. One residue, LOW: the known-bad falsifies only the `write_window` conjunct — the `FeedWriter` construction and the manifest record are not separately driven, so two thirds of `emits_feed` are asserted and not falsified |

**One stale comment, LOW:** `harmful_forward_scorer.py:600-602` still reads *"the un-declared call must be UNCHANGED, because `be_forward_day` calls `load_frozen()` with no expectation"*. Both call sites now bind `expect`, and the control immediately below it asserts the opposite. Prose beside a check that contradicts it (rule 10's shape).

**Not reviewed, as instructed:** that the bound artifact is `PM_PLUS_FINE` / `LINEAR`. BE was right to separate the freeze-level question and it is with the USER.

---

## Findings

| # | sev | finding |
|---|---|---|
| **BE17-R1** | **HIGH** | `require_fenced_op` gates on a truthy token it never verifies and never raises on; an op with no declared split yields an empty overlap and reports `causal_verified_against_scored_population: True`. A hand-built mapping produced the decision metric from a cutoff read off the scored ranking |
| **BE17-R2** | **HIGH** | nothing binds `theta_frozen` to `provenance.rows_artifact`; the real declaration with its thetas swapped for the scored rows' retrospective cutoff passed every check and was reported causal-verified |
| **BE17-R3** | MEDIUM | `token_recomputed` is False for the genuine op (`want` rebuilt from `provenance_verified`, token taken over raw `provenance`), asserted nowhere, raised on nowhere |
| **BE17-R4** | MEDIUM | BEM-R6 not closed: the block guard is now False by construction at the honest tip; the substantive values guard is still defeated by widen-plus-repoint (1e-6 → 1e6, `unchanged: True`) |
| **BE17-R5** | LOW-MED | `cutoff_depends_on_scored_data` returns `forward_eligible: True` for a theta read off the scored data; the name reads as a certificate the arithmetic cannot give |
| — | LOW | the `_emits_feed` known-bad falsifies one of three conjuncts; a stale comment at `harmful_forward_scorer.py:600-602` contradicts the control beneath it |

## Disposition

**AMEND**, and the standing rule I set stands: **no forward day is scored until BEM-R1 and BEM-R2 are closed.** BEM-R3 is closed and I release it.

The remaining distance is small and specific. BE17-R1 is two `raise` statements in a function that already computes both predicates. BE17-R2 is the one real piece of work — the fence must recompute the quantile map from bytes it has already hashed, or the day list must be derived rather than declared — and until it exists the receipt should print what is actually computed (`declared_split_does_not_intersect_scored_population`) rather than a causality claim.

I estimate; the coordinator routes; the USER decides (rule 14).

## Discipline record

Executed at `70e58cb` in `~/ctaNew-wt-rev` only; every step under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`. **No seal opened.** `be_forward_day.py` read, never run. Nothing written under `data/`; the attacks passed forged mappings as arguments and never modified a repo file. The tolerance re-attack ran in a throwaway git tree under the session scratchpad, removed afterwards. `~/ctaNew-wt-be`, `-da`, `-de` never read. No unit, timer or anchor; `DA_MIDNIGHT_MODE` never set. `git worktree list` **34** at quiescence, worktree clean.
