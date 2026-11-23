# MAKER-TRACE-RECONSTRUCTION-v1 — declaration (frozen before the full run)

2026-11-22, session aea5bfb9 (development).

**Task:** the USER's "we figure out how top makers make money first": the [plan](../automatic/20261122_maker_trace_reconstruction/plan.md) and its [coverage amendment](../automatic/20261122_maker_trace_reconstruction/coverage_amendment.md) (8b39f9e8, 9ff1e197).

**Build rules:** [contract.md](contract.md) with **amendment 1** (real-data probe facts) and **amendment 2** (pre-run code reviews and end-to-end rehearsal), and the code anchor `pf_mt_common.py`.

**Approval:** the USER approved **0.90 core-hours** for this task, including a 0.05 h review reserve, with a **line ceiling of 4.40** (AskUserQuestion, 2026-11-22).

**Top-up:** the builders' tests and probes used 0.33 h, against an estimated 0.03 h. The USER then raised the cap to **1.35 core-hours** and the **line ceiling to 4.85** (AskUserQuestion, 2026-11-22, the recommended option), so the declared plan runs unchanged.

## Question

From public trades (every maker and taker order hash) and full aggregated book snapshots, the receipt reconstructs:
- how the most active maker addresses build, keep and lose queue priority;
- how they refill and reprice;
- how they complete two-sided turnover;
- where their fill P&L comes from.

**What it is not:** descriptive only. No strategy, fit, imitation, tradeable claim or profitability verdict for us. **Multiplicity: 0 candidates.**

## Populations (fixed here; contract §2 + amendment 1)

| Name | Definition | Role |
|---|---|---|
| ID | Oct 23, every market (REST) | ex-ante identities: top 10 by maker shares (`exante_sh`); top 10 by **maker-leg** net P&L to resolution among maker-majority addresses (`sh_mk > sh_tk`) with resolved maker shares (`exante_pnl`; amendment 2) |
| FILL | Oct 24–27, every market | per-address ledgers, P&L decomposition, markouts |
| BOOK | the 19 complete markets | full-depth lifecycle and queue reconstruction |
| GT | Nov 8 scope windows where our signer `0xba14…ced8` has a maker leg in the public records | blind reconstruction scored against our own engine's truth |

- **Boundaries:** clipped hours give `boundary_censored`, with the upper and lower margins measured from receipt lags. Such markets are excluded from the economics as a counted status.
- **Exclusions:** every exclusion is a status, and counts come with every table.
- **Days:** only Oct 23–27 and Nov 8 hours 00–22. The guard refuses any other hour before access.

## Inference rules

- **Lifecycle:** contract §5 with amendment 1. `makers[]` order is never evidence. S_FIFO exhaustion comes from price priority across a record's prices. At a record's worst price, at most one maker keeps a remainder.
- **Scenarios:** S_FIFO / S_FIFO_LO / S_NOPRIO.
- **Classes:** observed / constrained (identical in all three scenarios and implied by H1–H3) / assumption_sensitive / unresolved, per field.

## Readings (computed in code; declared before any result)

### R1 — priority assumption (pooled over BOOK and GT)

| Test | Definition |
|---|---|
| T1 | of the S_FIFO price-priority exhaustion claims, the share whose hash fills again later |
| T2 | of the records with ≥ 2 worst-price makers, the share in which ≥ 2 of them fill again later |

**Reading, per test:**
- `INSUFFICIENT` if the denominator is < 200;
- `supported` if the rate is ≤ 1%;
- `contradicted` otherwise.

`contradicted` means the S_FIFO-dependent claims are reported only as illustrations, and the coverage tables use S_NOPRIO.

**Which version is primary:** the lifecycle reports each test in two versions, and the **settle-order** version is primary. The S_FIFO claims are made in settle order, so it measures their error rate as implemented. The **window-certain** version counts only violations whose later fill is certainly later. It is reported beside the primary as a lower bound on the true priority violations. The primary reading therefore concerns the **reliability of the S_FIFO claims as implemented**, not the venue's priority rule. The lifecycle simulator, which follows price-time priority exactly, shows that out-of-order settlement ids alone make the settle-order tests fire while the window-certain versions stay at 0. A `contradicted` primary with a window-certain rate of 0 reads as "settle order is too noisy for these claims", not as "the venue is not FIFO".

### R2 — ground truth (GT; our filled orders)

A **defect** is a failed observed or constrained claim. Each counts separately:
- the true activation lies outside [post_lo, post_hi];
- the true size is < size_lb;
- `size_exact` ≠ the true size;
- an `exhausted`-only removal set where the order truly kept a remainder;
- proven presence outside the true life.

**Reading:**
- `no_defect` if 0 defects, with n;
- `defective` otherwise. Each defective field class is demoted to `assumption_sensitive` in every table, as a superseding status, never edited away.

Assumption-sensitive misses are reported with rates.

**Caveats:**
- The clock premise (the book, UUID and collector clocks equal the venue status clock) is stated with every score.
- The scored population is outcome-selected: only filled orders have public hashes.

### R3 — coverage (no threshold; "most" is an objective, not a result)

**Scope:** per maker × day × side, for the ex-ante identity sets and the field.

**Fractions:** the share of filled orders / executions / executed shares whose posting, persistence, queue and removal fields are observed / constrained / assumption_sensitive / unresolved.

**Distributions:** posting- and removal-window widths, and queue ranges.

**Ranked list:** the economically important unresolved cases, ranked by |P&L|.

### R4 — economics (no verdict)

**Breakdown:** per identity set × day (Oct 24–27):
- P&L to resolution, fees, the spread at fill, and markouts at 1 s / 10 s / 60 s / END;
- matched round-trip vs residual inventory, with the residual split into drift and resolution;
- segments by market time, price region, leg role and lifecycle class.

**Computed predicates:**
- `sign_consistent`: the same sign on all 4 days;
- `share_matched`, `share_spread` and the segment shares.

**Days and sample size:**
- 4 days is fewer than 5, so there are no intervals (rule 8);
- `n_days` is printed with every row.

**Comparisons:** ex-ante sets vs the field, and the same-day top 10, labelled descriptive.

### Timelines (deterministic selection, before any outcome)

1. For each BOOK day, take the BOOK market with the most legs by the two highest-ranked ex-ante (Oct 23 maker-shares) addresses; ties go to the smallest mid.
2. Print its whole-market timeline: those two makers' orders (fills, windows, removal sets), with the book's best levels and depth at their prices.
3. Add up to 8 example orders from that set, chosen as:
   - the 2 most-filled per maker;
   - 1 with a ≥ 2-explanation removal set;
   - 1 single-fill order;
   - ties broken by sha256(hash).

## Integration amendments (before the freeze)

These are the builders' declared conventions, adopted as they stand:
- **`lo_ms`:** the frame before the first frame in [hi − 2 s, first frame ≥ hi] that *could* reflect the record (a pending increment ≥ the smallest leg value, a large-enough depth drop, or an unknown cell). On the simulator the exact-match-first rule put `lo` after the true match on 11 of 5,346 records, and this rule on 0. The exact-match variant is kept as `x_lo_exact_ms`.
- **`persist_lo/hi`:** the certain presence span [min hi, max lo] over an order's fills, identical across scenarios. It is −1 / −1 when empty, which is always the case for a single fill.
- **Event order:** settle order in every scenario.
- **Hard lower end (lifecycle fix batch):** every observed / constrained claim uses `x_lo_hard_ms = hi − 2,000 ms` (−1 for receipt-hi fills). The evidence lower end `lo_ms` is only a refinement: the S_FIFO_LO placement, the queue frame, and the pre-trade mid in the economics. On a simulator with 0–1,000 ms out-of-order settlement lags, `lo_ms` lies before the match on 9,324 of 9,328 fills, and no observed / constrained claim fails.
- **Statuses:** '+'-joined sorted flags.
- **Economics exclusions:** any non-ok source flag excludes the leg pair and its record (counted). SHARES fees are valued at resolution (`fee_res`) and at the fill price (`fee_fill`).
- **GT population:** scope windows with an own maker leg in the public records, decided from public legs only.
- **Truth table:** built before the final code revision. The run refuses unless the frozen code reproduces it exactly (the `dry` gate).

## Run order and gates

| Step | Content |
|---|---|
| 1 | selftests, the checker rehearsal and the end-to-end rehearsal (`tools/rehearse_e2e.py`: the real builders on synthetic raw data, then mt_checks; run 7 passes 39/39 checks, 22/22 re-parse checks and 76/76 falsifier cases, with 0 observed / constrained failures against the simulator); the report module is frozen with the builders |
| 2 | **freeze commit** |
| 3 | prefix: extract Oct 24 plus the lifecycle of its first BOOK market |
| 4 | write-once projection: spent + measured projection + 0.05 reserve ≤ 1.35 and line ≤ 4.85 |
| 5 | the rest: extract Oct 23 / 25 / 26 / 27 / Nov 8 in parallel slots; lifecycle; truth (dry equivalence of the pre-built truth table, then score); econ; report; checks |

If the projection does not fit, stop for a USER resource decision.

**Amendment 1 (2026-11-23):** the first projection refused: 1.4046 h against 1.35, with the line at 4.7153 within 4.85 (`logs/projection.json`, kept). The USER approved **+0.10 h: cap 1.45**, with the line ceiling unchanged at 4.85. A write-once `logs/projection_v2.json` is computed from the same measurements and records the superseded file's sha256. The launcher's gate reads it. No builder, checker or reading rule changed.
