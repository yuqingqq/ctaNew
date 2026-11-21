# PI0-COVERAGE-v1 — declaration

2026-11-21 15:09Z, session aea5bfb9. This is review 216ff603's next priority (`reviews/automatic/20261121_530eac3b/next_priority.md`), with two USER decisions:
- 13:41Z, "Approve go": authorizes the proposal.
- 14:2xZ, "Extend source, all 6": builds the missing world source for four of the dates.

What this round is:
- One data / coverage round: neutral-exposure labels under the same frozen PI0 (`pi0_50_v1`) and the same neutral-hold target, on six more consumed dates.
- Development data, never validation.
- No new predictor, target horizon, model search, live order or untouched date.
- No candidate is fitted, so this round adds nothing to the multiplicity count.

The specification is the header of `predictfun/research/pf_coverage.py`. Where the two differ, the header governs.

## Step 0 (done, receipt `source/`)

The world source was missing for Oct 31 and Nov 1–3. The frozen R2 / R3 builders were run unchanged through `pf_source_extend.py` (freeze `aad6a8b6`):
- The Oct 24 control reproduced the 11-17 outputs exactly.
- Each of the four days built about 30k orders, with v1 reconciliation matching the existing days' pattern.

## What is frozen (this commit, before the pilot and any label)

| Item | File |
|---|---|
| Builder | `pf_coverage.py` |
| Selftest | `pf_coverage_selftest.py`: synthetic world end to end |
| Mutation harness | `pf_coverage_mutants.py`: 47 / 47 caught on the final sources (earlier runs kept as provenance) |
| Checks | `coverage_checks.py`: A–H, falsifier, selftest |
| Population | `manifest.json`: worlds, strata, pilot, partitions, per-date input identities |
| Pilot rule | `tools/pilot_projection_cov.py`: write-once |
| Launcher | `launch_coverage.sh` |

The earlier mutation runs kept as provenance were each a fixture gap: block overlap, severe date share, and a check added later. None was a fix to the module's logic. Separately, a selftest end-to-end run found that `run_members` would have written tuple JSON keys, and that was fixed before this freeze.

## Population

**Dates.** Six dates, from two world sources:
- 20261024 and 20261025 use `keep_pull_20261117`.
- 20261031 and 20261101–03 use `source_extend_20261121`.

**Never read:**
- Oct 23, the teacher day;
- the sealed Oct 28–30 block;
- forward Nov 4 and later.

Every tape reader goes through the guarded loader, and the identities are `pf_adv2_sample.guarded_files`. Oct 31's 24 book hours exclude Oct 30 23Z.

**Worlds.** Per UTC hour, the 8 lowest `sha256(pi0_coverage_v1_20261121|world|key)`, giving at most 192 worlds per date and 1,152 in total, with `pi_world` recorded. Worlds that do not bind keep their status and are not backfilled.

**Pilot.** The 4 lowest-ranked selected worlds of each date: 24 worlds, inside the same population.

**Partitions.** One date × 2 UTC hours each, so every sampling cell lies inside one partition.

**Census.** Every (tick, side) decision of PI0's factual episode. Only model decisions are sampled:
- absent POST / WAIT: book_ok, room ≥ 0, not near END, supported;
- the first live review of each generation: KEEP / PULL.

Everything else is a counted status: engine fallback, cap, near END, feature unavailable, support refusals, incoherent forecast, later live decisions, pending ticks.

**Queries.** At most 4 per date × side × world-hour × category, by `sha256(…|query|key)`, so at most 4,608. Each carries `pi_cell = n / N` and `weight = 1 / (pi_world × pi_cell)`. The "decision-hour" of a cell is the world's stratum hour: decisions lie within 5 s of the world start.

## Labels

Each query's world is rerun with snapshots. The rerun must reproduce the census fills and decisions; otherwise its rows are `reroll_mismatch`.

From each snapshot, `pf_r2_shadow.hold_from` gives the neutral hold of a POST made at that state:
- absent queries: the target is the value from the hold's activation;
- first-review queries: the target is the value from the review's modeled cancel effect.

**Reconciliation** (each failure is a status, never a dropped row):
- the target recomputed from the hold's own fills;
- the hold ledger's cash / position / value identity;
- for PI0's own generations, the hold gid equals the factual gid, and the factual fills before the factual removal equal the hold's;
- for a first PULL, the factual removal lands at the modeled effect;
- for an absent WAIT, the factual decision was WAIT.

The Oct 26–27 admitted cohort, read from the movement audit's committed opportunities table, is shown as `reference.csv`. It is never pooled with this build.

## Reading map (computed by `predicates()`; descriptive; one sampling design per population; no interval)

**Covered cells.** A source × side × horizon-bin cell is covered if it has ≥ 30 labeled rows and ≥ 10 fills. Any other cell lacks support. Both lists are reported.

**Concentration** (per source, pooled):
- `value_error_concentrated`: the weighted top-5 world share of |Σ w (V − value)| is ≥ 0.5;
- `severe_concentrated`: the same share for severe losses (value < −30 c) is ≥ 0.5;
- `value_error_one_date` and `severe_one_date`: one date carries ≥ 0.5 of the absolute weighted sum.

**Selects positive local exposure.** This reading uses PI0's unchanged sign rule: POST iff V > 0, KEEP iff V ≥ 0. It is computed on two combined populations, each with its inclusion weights:
- the entry population (absent POST + absent WAIT);
- the review population (KEEP + PULL).

| Answer | Condition |
|---|---|
| POSITIVE | J_rule > 0, J_rule > J_all, and J_rule > the 500-draw matched-null p95 (strata: date × side × hour, chosen count preserved) |
| NOT_POSITIVE | otherwise |
| INCONCLUSIVE | fewer than 5 free strata or fewer than 30 free rows |

Single sources are constant under the rule by construction. Their null is computed and shown as degenerate.

**Forecast quality** is reported for every source, date and side:
- Brier, and P bins;
- movement and value MSE and bias against the zero and edge × E[q] baselines;
- the conditional movement error;
- severe totals;
- weighted and unweighted top-5 shares;
- horizon / queue / ordinal bins, as diagnostics only.

No rule is fitted and no favorable side or day is chosen afterwards. Entry and review values are never summed. Local values are not PI0 P&L.

## Budget

Limits:
- Up to 8 aggregate core-hours (including step 0, about 0.25 so far).
- The four development slots; the Codex slot stays separate.
- 1,536 MiB checkpoint and 2,048 MiB hard bound per process, enforced by the Guard and the Sampler.

The pilot runs first: six jobs, about 0.5 core-hours at most. `tools/pilot_projection_cov.py` then writes its projection once.

The projection model:
- **Loads:** a fixed load (identity, worlds, day identity) plus a per-member load (real inputs, tape).
- **Census:** per-world census time × 1,152 worlds.
- **Labels:** per world, `a + b × queries`, at the 4,608 query cap.
- **Margin:** 0.3 core-hours for measure and checks.
- **Spent so far:** the round's CPU to date.

The build (72 partitions) runs only if all of these hold:
- the pilot is complete with rc 0;
- peaks are below 1,536 MiB;
- the pilot cost at most 0.5 core-hours;
- the projection is at most 8 core-hours.

Otherwise it stops with statuses. No outcome-dependent thinning, no silent expansion.
