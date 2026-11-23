# TAKE-LOGIC v2 (ambiguity-aware) — contract (frozen before computation)

2026-11-23, session aea5bfb9.

**Mandate:** the ONE next priority of review af916aba6 (`reviews/automatic/20261123_take_logic_review/report.md`, findings 1–5). This completes the USER's inventory / take-logic question; it is not a new strategy or model.

**Budget:** ≤ 0.025 core-hours all-in, including ≤ 0.010 for its review. The ledger stood at 1.8164 of 1.90 before this round.

**Data:**
- the same 19 BOOK markets × ranks 1–3 × the 300-s windows, Oct 24–27 (consumed);
- the same inputs and builders as TAKE-LOGIC v1;
- ONE bounded reconstruction of this panel, saving its per-second state and raw event keys so that no later reconstruction is needed;
- no new days, no window search, no bin or threshold change, no replay.

## Unchanged from v1 (`take_logic_contract.md`)

- the population: maker-seconds;
- the anchor: t < x_lo_hard ≤ t + 1000;
- the state builders: q from legs with hi ≤ t, BTC aligned to q, time left, own fills 5 s, unwind edge;
- every bin, the MH strata (rank in every pooled stratum), 500 circular shifts, seed 20261123;
- the market-support rule and the verdict rule.

**Naming (finding 2):**
- the anchor is a *fixed market-second, pre-earliest-match proxy*, not an identified decision time;
- q(t) is reconstructed venue inventory, not client-known inventory.

## Outcomes

Each is a per-maker-second indicator, de-duplicated within the second. Outcome types are never summed.

**Raw takes:** all 4,124 panel takes (time, side, size), ambiguous records included; the 23 before the window keep their status.

| Outcome | Definition | Role |
|---|---|---|
| **O1 opposing take** | ≥ 1 raw take with side = −sign(q(t)). Defined only for q(t) ≠ 0; q = 0 and unknown are statuses. It is a direction relative to the reconstructed anchor, not proof of a private inventory reduction. | **primary** |
| O2 same-direction take | ≥ 1 raw take with side = +sign(q(t)) | |
| O3 any take | ≥ 1 raw take, at any q | beside |
| O4 \|q\|-reduction across the saved feasible variants | see below | |

**O4 detail:** uses chain_take's recorded / certain / simultaneous classes, with `classify` / `REDUCING` as saved.
- Fine reduction subtypes are grouped BEFORE agreement: reduce_partial / full_close / reversal_reducing → reduces; add / from_flat / overshoot / no_change → not.
- Per take: `reduces_all`, `reduces_some`, `reduces_none` or `unavailable`.
- Per second: ≥ 1 `reduces_all` take, and separately ≥ 1 take with variant disagreement.
- Coverage and disagreement are published by |q| bin.

## Ratios

**Set:** the v1 set (RR_size, RR_with_5s, RR_against_5s, RR_end, RR_burst, RR_edge) for O1, O2, O3 and O4-reduces_all, for rank 1 and pooled.

**Two panels:**
- (a) the full retained-ledger panel;
- (b) the clean-ledger subset: maker-seconds with no earlier excluded leg.

**Clocks:** D0 is primary; D1000 and pending legs are sensitivities.

**Market consistency:** stated as the declared **crude** within-market comparison.

## Nulls

**O1 / O2:** circularly shift each maker-market's **raw side-specific event schedule** (raw takes with their sides) by a uniform offset mod 300 s.
- Then re-derive the direction against the destination second's q(t), de-duplicate per second, and recompute the MH ratio.
- 500 shifts, seed 20261123.
- Raw counts per pair are preserved. Collisions (raw takes sharing a destination second) are counted and reported.

**O3 / O4:** the v1-style shift of the outcome vector, kept as the benchmark only.

## Decision (declared before results)

**Primary:** O1 RR_size (|q| > 200 vs (0, 50]), pooled, D0, full panel. Readings:

| Condition | Reading |
|---|---|
| verdict is not `supported` (v1 rule) | **inventory size as a trimming trigger: WITHDRAWN as a maker rule.** No further bins or variants are tried. |
| `supported` on the full panel AND the clean-ledger subset AND at D1000 AND with pending legs | **qualified candidate association** (inventory size ↔ opposing takes); not causal, not a profitable rule |
| supported in some views only | `unidentified` |

Rank 1 gets the same reading, reported as secondary. Every other ratio is listed.

Multiplicity: 6 ratios × 4 outcomes × 2 scopes × 2 panels.

## Corrections bundled (findings 2–5)

These go into the superseding receipt `v2/report_logic_chains_v3.md`.

**Census:** the base populations are counted separately:
- maker-fill rows: 9,026;
- take records: 4,147;
- overlapping attribute incidences, kept as attributes: mint 3,709, multi-record 2,722, merge 16, self-trade 38.

These are recomputed by the tool from `correction_v1/census_graded.csv`.

**Wording:**

| First wording | Corrected |
|---|---|
| "fast reaction within ~1 s" | **withdrawn**: alignment sensitivity only |
| "BTC direction is not the trigger" | "no BTC-direction association passed this declared screen" |
| the 16/18 count | crude within-market consistency |
| "adds stop" | add takes fall to 0 in 10–30 s and 5 per 1,000 s in the last 10 s |
| "markets with ≥ 5" | counts event-seconds |
| "mostly carried into resolution" | some markets end with sizeable recorded positions |

**Stated as data:**
- 46 reduce-classified takes (13 event-seconds) whose side equals the anchor inventory sign;
- 3 overshoot records.

## Fixtures (selftest, synthetic, through chain_take's actual `classify` / `combine`)

1. **Bias control.** A random-side, random-time maker on an inventory path where small |q| makes the recorded / certain / simultaneous variants disagree more:
   - the v1 outcome (classified reduce, ambiguous dropped) must show the spurious RR_size > 1;
   - O1 must show no supported size effect.
2. **Planted size policy.** A maker whose opposing-take probability rises with |q|: O1 RR_size must be supported.
3. **Shift checks.** The raw-schedule shift preserves raw counts per pair and re-derives direction at the destination.
4. **Mutants, each caught by a check:** `shift_labels` (shifting the labelled vector), `drop_ambiguous`, `sum_types`.
5. **Clean-ledger filter:** a planted excluded leg removes exactly its later seconds.

## Outputs and cost

**Outputs:** `v2/chains/take_logic_v2/` (write-once):
- `panel.parquet` (per maker-second state);
- `events.parquet` (raw take keys, side, size, anchor second, variant classes);
- `ratios.csv`, `coverage.csv`, `consistency.csv`, `census_fix.csv`;
- `claims.json`, `receipt.json` (sha256 of source, inputs and outputs; CPU; RSS).

**Cost:**
- builder ≤ 15 CPU s (≤ 2 selftests + one Oct 24 pilot into `v2/chains/take_logic_v2_pilot_20261024_r1`);
- full run projected ≤ 25 s;
- refuse to launch above 40 s.
