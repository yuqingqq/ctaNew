# LOGIC-CHAIN CORRECTION v1 — contract (frozen before computation)

2026-11-23, session aea5bfb9.

**Mandate:** the ONE next priority of review 6bb458b30 (`reviews/automatic/20261123_logic_chains_review/report.md`, findings 1–7). The USER: "Check the new commit next priority, can you auto check".

**Scope:** saved evidence only.
- no new data, and no rerun of the lifecycle, context or chain tools;
- the original outputs and `v2/report_logic_chains.md` are retained unedited.

This packet supersedes the receipt's interpretation in-band, as `v2/report_logic_chains_v2.md`.

**Budget:** ≤ 0.025 core-hours all-in. That is ≤ 0.015 for development, tests and runs, plus the 0.010 review reserve. The packet total stays ≤ 0.11 h, the USER-approved figure.

**Outputs:** `v2/chains/correction_v1/` (write-once). Two tools, each with a selftest:
- `tools/chain_correct_post.py` — POST regrade, census grades, episode paths;
- `tools/chain_correct_pull_take.py` — PULL publication, TAKE relabel.

## §1 POST regrade (finding 1)

**Unit:** the 4,149 filled top-3 orders (`chains/post_fill/orders.csv`).

**Inputs joined per order:**
- the saved lifecycle `post_class` (`life/<day>_<mid>_orders.parquet`; observed / constrained / assumption_sensitive / unresolved);
- the packet status;
- the estimator's conflict / clock flags (`order_estimator/census_orders.csv`).

**Grades, first match:**

| Grade | Rule | Meaning |
|---|---|---|
| G0 `unbounded_or_invalid` | union window unbounded or order invalid | — |
| G1 `conflict` | capacity / clock / post-after-first-fill conflict flag | — |
| G2 `bound_constrained` | lifecycle class observed or constrained, and status ok | The window holds under the lifecycle's scenario-free bound. Still conditional on the H1 / H2 / capacity premises and the declared maximum match lag. |
| G3 `constrained_status_flagged` | constrained, but status not ok | — |
| G4 `scenario_only` | every other row (assumption_sensitive / unresolved) | The window width is a scenario estimate only. |

**Reporting:** grade × scenario width bin (≤ 1 s / ≤ 10 s / > 10 s / unbounded), per rank and day, with denominators.

**Claims:**
- Allowed: "n orders have a bound-constrained posting window ≤ 10 s."
- Never: a share of ALL orders that "rested ≤ 10 s".
- Never: "not long-resting queue builders".
- A G4 row says nothing about its true rest.

**Logic-table POST rows** (the saved episodes):
- `B_constrained` only for G2; G4 becomes `C_scenario`; G0 / G1 become `E_unresolved`.
- The queue-priority-before-demand status may be `contradicted` only for G2 with width ≤ 10 s; G4 is `timing_sensitive`.

**Cross-check against the reviewer's recount:**

| Count | Reviewer |
|---|---|
| scenario-short | 2,917 |
| constrained | 957 |
| constrained and ok | 940 |
| assumption-sensitive | 1,960 |
| conflict-flagged | 413 |

These are reproduced or the difference is reported. It is a recorded predicate, not a refusal.

## §2 PULL publication (findings 3, 4)

**Control families:** both declared families (`presence`, `fill_anchored`) are published side by side for every published group:
- cancel-compatible;
- exhausted-proven;
- the successor ≤ 5 s split.

Each row carries its support (n_events, n_with_controls, n_obs, n_ctl_valid) and a row per day. S_FIFO is primary; S_NOPRIO stands beside it.

**Reference membership per group:**
- presence-matched;
- active-span fallback;
- unavailable (reviewer: 908 / 1,826 / 7 of 2,741).

**Wording:**
- "cancel-compatible terminal window", never "pull";
- p = conditional reference-draw tail fraction (≥ 200 draws), not independent-day confidence;
- "successor ≤ 5 s" = a later first-filled same-side hash, not a replacement.

**Computed predicates:**
- adverse BTC within 1 s enriched under both families, pooled and on every day;
- the inventory difference > 0 on every day under fill-anchored (a stable trigger);
- cancel-compatible enrichment > exhausted enrichment on every day.

## §3 TAKE relabel (findings 2, 4, 6)

**Renames:**
- `r_5s_against` / `_with` become BTC against / with the TAKE side;
- the inventory-aligned reading is withdrawn;
- no new null is built.

**Case-only per-class rates:** reduce / add / timing-sensitive, in take-side alignment and in row-level inventory alignment. A reducing take's inventory side is opposite its take side; an adding take's is the same. They are labelled "cases only, no matched reference".

**Reference-match levels:** full stratum / sign dropped / inventory dropped, overall and per rank (reviewer: 3,977 / 125 / 45).

**Earlier excluded legs:** claim-eligible takes with earlier excluded legs are counted by class (reviewer: 248). Inventory-class statements are conditional on the retained ledger, and the class table is also shown without these takes.

## §4 Census grades and episode paths (finding 5)

**Every action count carries an evidence grade:**
- observed execution: maker fills, takes;
- continued execution: "keep after fill", not a private KEEP;
- scenario-dependent link: reprice / refill, tied vs sequential;
- cancel-compatible ending: pull_identified is False;
- feasible / timing-sensitive overlap: two-sided, ladder.

**Paths:** per rank, the first two episodes in the deterministic case order (`order_estimator/cases.csv` row order), up to 6. This is predetermined and not outcome-selected. Each is shown as:

post (size alternatives, price, window grade) → fills → terminal (tie) → successor (link grade) → takes (class) → the saved episode economics.

Each path also shows the alternative that would change its logic.

## §5 Binding and falsifiers (findings 5, 7)

**Binding:** each receipt binds the sha256 of every input and output.

**Synthetic falsifiers** (each with an env-gated mutant the selftest must catch):
- **grade:** a union-short but assumption_sensitive row must not grade G2;
- **sign:** a reducing sell with BTC up is against the take side and with the inventory;
- **omitted control:** publication refuses a row missing either family;
- **observed vs inferred:** a tied link is never counted as an identified repost, and the census refuses an ungraded action.

## §6 Deferred (nonblocking; recorded, not built here)

- the masked-fill / replenishment falsifier for the scenario-union classifier, before that classifier is reused;
- `chain_take` selftest: an explicit bad-state fixture and `relative_to` portability, when that suite is next touched;
- a new inventory-aligned take null (optional future work).
