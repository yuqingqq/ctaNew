# COORDINATOR DISPATCH — P-2026-003, harmful-fill hazard × toxicity phase
**As-of 2026-08-25 ~06:5x UTC. Coordinator: this session. Module work runs in
separate sessions; THE INTERFACE IS FILES, not conversation. Each module session
reads this dispatch, its plan sections, and files results + one-line register
rows in COORDINATION.md (ASK: needs a coordinator ruling / FILING: self-report,
closes on ACK).**

## Governing documents, in order of authority
1. `live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md` — the user's
   plan; §10 is the implementation order, §11 the stopping rule.
2. `live/pm_research/FLOW_MODEL_STATE.md` — wins on FACTS.
3. `live/pm_research/contracts/contracts.yaml` (v24) — wins on TYPES.

## Standing rules every session inherits (hard-won this week; do not relearn)
- **Verify at the artifact the claim names**, never a proxy for it.
- **Every check ships with a false-positive analysis**; a zero from an
  instrument that never proved it can fire is not a result.
- **Nulls declared before results**: design AND minimum permutation count
  (≥200). An under-sampled correct null flatters as much as a wrong one.
- **R-105**: every population carries its n AND as-of. The tape GROWS during
  measurement; a day list is not a vintage.
- **R-109**: intervals only on the correct cluster unit; otherwise the point
  estimate with NO interval, stated.
- **R-126**: new surface carries its authorisation IN-FILE.
- **Era purity**: mm_hf events admitted only if `recv_ns >= 1787579334881534478`
  (v2 stamp boundary, from `data/mm_hf/collector_runs.jsonl`). Legacy stamps
  carry up to ~0.6 s of parse-backlog error at p99 and are inadmissible for
  fine-scale work.
- **A model estimates; it never decides.** CANCEL is DE's decision, priced
  downstream. No worker-produced boolean may encode an entitlement.

## State (verified from artifacts, not memory)
- **Exposure dataset** (plan item 2): DONE. `harmful_exposure_rows.py`
  (committed, 13 selftests) → `data/pm_5min/derived/harmful_exposure_rows_v1.json`
  — 183,043 rows, 60 windows, 3 training days, exclusions as statuses.
  v2-era btc build: 171 windows / 461,950 rows (scratch `v2_rows.json`).
- **Features + linear reference** (items 3–4): DONE. `harmful_hazard_model.py`
  (committed, 13 selftests). Ablation: fill hazard is QUEUE-STATE-driven
  (AUC 0.83; micro features add nothing to hazard).
- **THE LIVE RESULT (fragile, one fragment pair):** on v2-era data,
  out-of-time (train 08-24 → test 08-25), adding fine Binance features
  (10–250 ms, era-pure) flips net cancel value at 5/10/15% budgets from
  −10,194/−12,923/−14,024c to **+302/+1,630/+3,879c** on btc. Hazard AUC
  unchanged — the gain is in the TOXICITY head. NOT frozen, NOT confirmed.
- **In flight:** the four-check battery (forward repro / reverse split /
  per-hour / fine-family ablation) — pid 1858269, cached-feature scratch run,
  ETA ≤ ~25 min from 06:45. Results land in
  `tasks/bgwiy34y8.output`; feature cache at `scratchpad/adv/v2_features.pkl`.
- **Frozen candidates:** `be_adverse_move_candidate_v2.1.json` (settlement
  horizon, clock 2026-08-24T15:04:28Z, commit-pinned) — race of ONE. v4 VOID.

## Module assignments (a session per row; file into COORDINATION.md)
| session | scope | first action |
|---|---|---|
| **BE** | items 4–5: toxicity head + fixed-capacity nonlinear (pinned hyperparams BEFORE scoring) | absorb the battery verdict; if reverse+hourly hold, draft the freeze receipt for the PM+fine linear spec (v2.1-style: builder sha + commit + declared nulls + multiplicity) and file it as an ASK for coordinator ratification — DO NOT self-freeze |
| **DA** | dataset stewardship: eth v2-era exposure rows; day-stratified top-ups as clean tape accrues; provenance (n + as-of on every receipt) | build eth rows with `harmful_exposure_rows.py` machinery over the v2 era; report fill prevalence per fragment |
| **EV** | item 6 gates: §8 battery — rho = A/S, PR-AUC, loss capture, matched-random ≥200, per-day | re-score the battery's winning arm independently from the receipts; disagree loudly if numbers differ |
| **DE** | items 7–8: cancel×skew replay integration + 5–250 ms latency grid + queue-reset-cost sensitivity | wait for EV's gate pass; nothing integrates before it |
| **OPS** | item 10: real cancel-send/ACK/owned-fill timing — OUTSIDE this repo if a live adapter is needed | scope the measurement; file the design as an ASK |

## Blocking discipline
Exactly one thing is decision-blocking at a time. RIGHT NOW: the battery verdict
→ freeze-or-wait on the PM+fine spec. Everything else is parallel or DEBT with a
named trigger.

## Training plan — DECLARED 2026-08-25 before the data that would tune it exists
- **Min training mass (measured floor):** ≥150k exposure rows AND ≥30k fills
  before any scored day (235k/45k generalised; 76k/10k degraded — the reverse
  split is the receipt). Toxicity head binds; eth needs ~8× btc's calendar.
- **Window:** EXPANDING until 7 complete era-days exist, then ROLLING-7. The
  switch point is Class-D-style: declared now, not tunable after results.
  Pre-registered test at ~10 days: rolling-7 vs expanding, same walk-forward.
- **Cadence:** refit daily at the UTC boundary, never mid-window; every refit an
  immutable artifact (hash, span, n, fills). Walk-forward scoring only — a day
  is scored by the model frozen before it began. 60s embargo at the seam.
- **Era discipline:** stamp/collector changes (ledger) truncate the window to
  post-boundary tape. Mandatory, not judgement.
- **R-109:** day is the cluster unit; no intervals until G ≥ 5 complete days.

## BE ASSIGNMENT — Binance-mechanism study (user directive, 2026-08-25)
**Question: what ELSE in the captured Binance data carries harm-avoidance
information for PM quotes?** Today's model uses only bookTicker top-of-book
(imbalance, mid-move). Unused capture, ranked by mechanism prior:

1. **depth20@100ms — ENTIRELY UNUSED.** 20 levels/side. Candidates: depth-
   weighted pressure within X bps; deep-vs-touch imbalance (liquidity is PULLED
   deep before moves — early warning preceding any touch change); book slope;
   depletion vs replenishment rate at the touch.
2. **bookTicker qty DYNAMICS.** We read the level; the CHANGES are the signal:
   order-flow imbalance (OFI — literature-standard short-horizon predictor,
   computable from bid_qty/ask_qty deltas per event); quote flicker rate;
   microprice (size-weighted) displacement vs mid — known to lead the mid.
3. **Trade-stream shape beyond signed volume.** Large-print detector (informed
   size); multi-level sweep progression; inter-trade burst intensity.
4. **Cross-symbol lead.** 16 symbols captured; ETH/SOL co-movement as
   confirmation of a real (vs idiosyncratic) move.

**Deliverables + discipline:** (a) a mechanism memo — for each candidate, WHY it
should precede PM harm and PROOF it is computable from the actual captured
columns; (b) candidates DECLARED before any scoring (no scoreboard selection —
the rate-feature lesson); (c) tested as INCREMENTS over the reduced spec
(imb+midbps), era-pure (v2 stamps only), declared nulls ≥200 perms; (d) every
population with n + as-of. Verify depth20 semantics FIRST (snapshot cadence,
level ordering, absolute-vs-delta) against the tape before computing anything.
