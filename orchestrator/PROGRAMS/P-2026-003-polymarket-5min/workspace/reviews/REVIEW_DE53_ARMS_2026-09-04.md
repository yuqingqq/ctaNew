# Review — DE round 53, the first §8.1 arms run with real predictors

reviewer: pm-codex · filed 2026-09-04T11:26Z · **tip `0d0e61e`** (DE53 at `1206143`)
executed in `~/ctaNew-wt-rev`. **No race seal opened.** Nothing written under `data/`.

**Lead, because it inverts the dispatch's item 2: the emission on disk says the control IS valid.** The round's headline — and the dispatch that quotes it — says `permutation_ok` False, P2/P3 FALSE, `VALID_AS_A_CONTROL: false`. The artifact says the opposite.

---

# DE53-R1 — **HIGH** — the artifact contradicts the round's headline, and the field that would stop a reader is a hardcoded string

`arms.RANDOM_MATCHED.matched`, read verbatim from the emission:

```json
{
 "VALID_AS_A_CONTROL": true,
 "permutation_ok": true,
 "predicates": {
   "P1_key_multisets_equal": true,
   "P2_stratum_score_multisets_equal": true,
   "P3_drawn_carry_above_and_only_drawn": true,
   "P4_realised_action_counts_equal": null
 },
 "note": "permutation_ok False means the permuted stream did not satisfy P1-P3,
          so this arm is NOT a valid matched control and no arm-vs-floor
          comparison may be drawn from it",
 "drawn": 1154, "treated_actions": 1154,
 "demand": {"above_threshold_events": 1154, "realised_actions": 333, …}
}
```

**`permutation_ok` is `true`. P2 and P3 are `true`. `VALID_AS_A_CONTROL` is `true`.** And the `note` beside them asserts *"permutation_ok False … NOT a valid matched control"*. The prose contradicts the booleans it sits next to — CLAUDE.md rule 10, *"a hardcoded verdict string beside a table has contradicted the table three times"*, now a fourth.

**The two runs are not the same run.** The commit message reports `as-of 2026-09-04T11:06:34Z`; the artifact's population carries `as_of 2026-09-04T11:11:14Z` and the file's mtime is 11:11:16Z. So the round was filed from an 11:06 run and the artifact on disk is an 11:11 run that answers differently.

**Answering the dispatch's question directly: no, a reader is not prevented from drawing an arm-vs-control comparison — the artifact tells them it is permitted.** Which of the two answers is true I cannot establish, and that is DE53-R2's fault, not this one's: there is no committed producer to re-run. Both readings are bad in different directions — if the note is stale, the round's headline is wrong and a valid control is being disclaimed; if the booleans are stale, a reader takes a comparison against a floor that does not hold.

---

# DE53-R2 — **HIGH** — the round has no committed producer; its substance is a commit message and a file in a temp directory

The publication-provenance census, run over all committed modules for every field the round's claims rest on:

| field | committed modules containing it |
|---|---|
| `arm_distinctness` | **NONE** |
| `VALID_AS_A_CONTROL` | **NONE** |
| `permutation_ok` | **NONE** |
| `all_distinct` | **NONE** |
| `arm_signature` | **NONE** |
| `MATCHED_RANDOM_PERMUTATION` | **NONE** |
| `distinct_predictors` | `da_arm_replay_verify.py` — a **consumer**, not a producer |

`1206143` changes **three leaf-name strings** in `SECTION_8_1_FIELDS` (+4/−3, one file). Nothing else. The emission is a **14,386-byte file in DE's session scratchpad**, `…/93dd9a62-…/scratchpad/arms53.json`, and it **names no producer**: no `produced_by`, no `producing_code`, no `carrying_commit`.

This is the `matched_volume` situation one level up. There the headline number came from a scratch script; here an entire round's findings do — the arms, the identities, the distinctness assertion, the control's validity, the exclusion accounting, the memory figures. The standing practice adopted after that finding returns **0 committed call sites** for every one of them.

> **Clause.** The runner that produced `arms53.json` must be committed and the emission must carry `produced_by` and `carrying_commit`, before any of this round's numbers is read as a result. Until then the round is a scratch reproduction, whatever its arithmetic.

---

# ITEM 1 — distinctness: real for three arms, a declaration for the fourth

The signatures do differ, and they differ on a field that cannot coincide by accident:

```
QR_SKEW_ONLY            [  0, -0.8137, 8598.76]
CONDVALUE_X_SKEW        [333, -1.0015, 7644.84]
HAZARD_OVER_SKEWED_REF  [ 48, -0.8281, 8586.20]
RANDOM_MATCHED          [496, -0.8203, 7676.45]
n_arms 4 · n_distinct 4 · all_distinct true
```

For three arms the identity is genuine **independent** evidence — different artifacts, different shas: `linear_d_btc.json 18701008c2bd18c6` for `incumbent_linear_d`, `lgbm_haz_btc.txt ec52055214a01ed5` + `lgbm_thresholds_btc.json 0fa2f1f7a5a4c58f` for `q1_arrival_composed_lgbm`, and `{}` with `n_cancels == 0` asserted for the no-predictor arm.

**DE's flag about RANDOM_MATCHED is correct, and its consequence is larger than the flag says.** The control carries *the same two shas* as `CONDVALUE_X_SKEW` — necessarily, since it permutes that arm's stream, and a matched control that used different artifacts would not be matched. So for that pair the shas are **not** distinguishing evidence, and what remains is the `predictor` label (`MATCHED_RANDOM_PERMUTATION`) and the signature — **both produced by the emission itself**. So: *arm 4's distinctness from arm 3 rests on a declaration, not on independent evidence.* That is precisely what DA's verifier exists to refuse to accept, and it cannot be run here: the verifier reads `declared_parameters`, `contract_leg.inert_check` and `arm_runnability`, none of which this document shape carries.

**The reasoning is sound; the limit should be stated in the emission**, as `identity_evidence: "SHARED_BY_CONSTRUCTION — distinctness for this arm is declared, not evidenced"`, so a reader is not left to infer it from a note.

# ITEM 3 — exclusions: a proper status, and its ignorability is not established

```
scored_generations 29,813 + excluded_no_assembled_score 1,309 = reference_generations 31,122   ✓
excluded_fraction 0.0421
reason: "the feature pass dropped every row of these generations; scoring them
         would be scoring from nothing (rule 4: counted, never dropped)"
```

**The denominator reconciles** — I checked the arithmetic — the exclusion is a named status with a reason and a rule cite, and the scorer refused rather than scoring from nothing. That is the shape rule 4 asks for and it is right.

**What is not established is that the 4.21 % is ignorable.** The artifact carries no breakdown of the excluded generations by any covariate — not by window, side, hour, or score. A feature pass that drops every row of a generation is unlikely to do so at random with respect to the thing being measured, and the arms report over the surviving 95.79 % as though it were the population. **The count is honest; the ignorability is unexamined**, and should be stated as such rather than left to a reader to assume.

# ITEM 4 — clean

`HAZARD_OVER_SKEWED_REF` appears in **zero** committed modules, and the canonical `ARMS` tuple contains `HAZARD_ONLY_NEUTRAL` (arm 3) and not it. Nothing downstream can read the one as the other, because the name does not exist in the code. DE's naming discipline holds.

# Credit where it is due

On real output, the three unproducible fields come through exactly as the preflight required — **`NOT_AVAILABLE` with substantive reasons, never zeros**:

- `maker_pnl_cents` — *"the replay prices the DECISION, not the book"*
- `inventory_loss_cents` — *"Inventory LOSS needs a terminal mark, which the replay never takes"*
- and a fourth honest status I had not asked for: `latency_x_cost_sensitivity: NOT_RUN_THIS_ROUND` — *"needs the 9-rung latency axis; one rung run"*

That is the item-2 half of my preflight, closed on live output.

---

## Findings

| # | sev | |
|---|---|---|
| **DE53-R1** | **HIGH** | the emission carries `permutation_ok: true`, `VALID_AS_A_CONTROL: true`, P2/P3 `true` — the opposite of the round's headline — beside a hardcoded `note` asserting the control is NOT valid. Rule 10's fourth instance. A reader is told the comparison is permitted. The two runs differ (11:06:34Z filed vs 11:11:14Z on disk) and I cannot settle which is right |
| **DE53-R2** | **HIGH** | **no committed producer**: six of the round's headline fields appear in zero committed modules; the commit is three leaf-name strings; the emission is a scratchpad file naming no producer. The provenance census returns 0, as it did for `matched_volume` |
| **DE53-R3** | MEDIUM | `permutation_ok: true` with `P4_realised_action_counts_equal: null` — an **unevaluated conjunct absorbed into a pass**. P4's `null` is correctly outside the codomain; the aggregate over it is not. And `realised_actions 333` against `treated_actions 1154` is exactly what P4 would test |
| — | — | **Item 1**: distinctness is independently evidenced for three arms and **declared** for the fourth, by construction; the limit should be in the emission. **Item 3**: exclusions reconcile (29,813 + 1,309 = 31,122) and are a proper status; ignorability unexamined. **Item 4**: clean. **Credit**: the three unproducible fields are `NOT_AVAILABLE` with reasons on real output |

## Disposition

**Do not read any number from this round as a result, and do not act on the control's status in either direction, until DE53-R1 and DE53-R2 are closed.** The arithmetic may well be fine — the distinctness signatures and the exclusion accounting both look right to me — but the round asserts one thing in a commit message and the opposite in the artifact, and there is no committed code to re-run to decide between them. Those two facts compound: an unresolvable contradiction is what an uncommitted producer costs.

The order I would take them in is R-2 first. With the runner committed and the emission naming its producer, R-1 becomes a five-minute question instead of an unanswerable one.

I estimate; the coordinator routes; the USER decides (rule 14).

## Discipline record

Executed at `0d0e61e` in `~/ctaNew-wt-rev`; every step under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`. **No race seal opened**; `be_forward_day` never run. `arms53.json` was read read-only from DE's session scratchpad — a scratch artifact, not a seat worktree and not a seal; `~/ctaNew-wt-be`, `-da`, `-de` were never read. Nothing written under `data/`. Worktree clean.
