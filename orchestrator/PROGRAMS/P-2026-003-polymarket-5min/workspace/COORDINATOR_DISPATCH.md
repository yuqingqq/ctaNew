# COORDINATOR DISPATCH — P-2026-003, STATEFUL harmful-cancel phase
**As-of 2026-08-26 ~03:0x UTC (supersedes the 08-25 dispatch; prior text in git
history). Coordinator: tmux `pmmm-coordinator` (session ctanew-b9). THE
INTERFACE IS FILES: each plane reads this dispatch + the governing plan, works
in its own session, and files results as register rows in `COORDINATION.md`
(ASK = needs a ruling; FILING = self-report, closes on ACK). Ruling of record
for this phase: COORDINATION.md R-145.**

## Governing documents, in order of authority
1. `live/pm_research/plans/STATEFUL_HARMFUL_CANCEL_TODO.md` — the phase plan
   (Phases 0–5, correctness tests, gates, deliverables 1–6).
2. `live/pm_research/plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN.md` — parent
   (§10 order, §11 stopping rule).
3. `live/pm_research/FLOW_MODEL_STATE.md` — wins on FACTS.
4. `live/pm_research/contracts/contracts.yaml` — wins on TYPES.

## Where the line stands (verified at artifacts, commit `f1ceec9`)
- OB dynamics loop CLOSED at I5. **Reduced fine spec (PM + L1 imbalance +
  10–250 ms mid-move) CONFIRMED** on the consumed 08-24/25 development
  fragment; extended (OFI+big-print) HELD @5%-only; depth20 REJECTED;
  PM-thinning NULL; btc→eth lead NOT ADOPTED (fails the T−5s timeliness
  control at 5%). Receipts: `data/pm_5min/derived/harmful_fine_comparison_*`.
- **Five specs consumed the fragment; no further scoreboard on it — permanent.**
  Reproduction / correctness checks / cost-latency sensitivity of an UNCHANGED
  score are allowed; selecting a new family from those outcomes is not.
- **The freeze of reduced(+extended) is the USER'S decision, still open.**
  Phase 0 must pass first (rule 12: a freeze is a commit + manifest + repro).
  Forward validation = ≥5 complete untouched UTC days AFTER the freeze instant.
- Working baseline everywhere: reduced-fine linear model (`PM_PLUS_FINE`) on
  the unchanged `QR_SKEW_ONLY` shadow trajectory, v3.4 exposure rebuild
  (471 windows, 1,125,289 OK rows, zero reconciliation failures).

## Declared development top-up (R-145(3) — recorded BEFORE any score exists)
btc+eth 5-min windows, slug start **strictly after `1787650200`** (08-25
09:30 UTC) **and strictly before 2026-08-26 00:00 UTC** (last slug
`1787702100`). Era-pure (`recv_ns >= 1787579334881534478`), complete windows
only, exclusions as counted statuses. **DEVELOPMENT only — never forward
validation; consumed at first read.** DA writes the top-up receipt (slug list,
n by coin/day/status, SHA-256 of dataset/builder/ledger) before BE reads any
Phase-2 number. **Tape from 08-26 00:00 UTC onward is untouched, reserved for
forward windows.**

## Standing rules (inherited + new; do not relearn)
- **HEAVY-RUN RULE (R-148(3), supersedes R-145(4)/R-147(5) patterns): every
  heavy run launches under
  `systemd-run --user --slice=research.slice -p MemoryMax=<job ≤14G> -p OOMScoreAdjust=1000 -- /home/yuqing/pricer-sol/venv/bin/python3 <script>`.**
  The slice caps AGGREGATE research memory at 18G (installed, smoke-tested —
  the 08-26 03:55 box death was an aggregate exhaustion no per-job cap
  bounded). Bare heavy launches from session shells are FORBIDDEN. Never
  `MemoryHigh` (swapless stall); snapshot any receipt a reproduction would
  overwrite before launching.
- Verify at the artifact a claim names; every checker ships a falsifier;
  nulls declared before results (≥200 draws); n + as-of on every population
  (R-105); intervals only on the UTC-day cluster unit, else point estimate and
  say so (R-109); new surface carries its authorisation in-file (R-126); era
  purity by `recv_ns`, never by file; **a model estimates, it never decides** —
  no worker boolean encodes an entitlement; predictor state ≠ policy state
  (TODO §2: inventory/skew/cancel-lifecycle stay OUT of the predictor).
- Corrections supersede in-band (vN+1 receipts); never edit a frozen artifact.

## Assignments (one row per plane; file into COORDINATION.md)
| plane (tmux) | scope | first action |
|---|---|---|
| **BE** (`pmmm-be`) | **Phase 0** — deliverable 1 `harmful_candidate_manifest_v1.json`: pin era boundary in-manifest; as_of/bounds/slugs/row counts/embargo; SHA-256 of dataset+ledger+schema+builders+deps; feature names/normalization/weights/thresholds; correct the hardcoded split field (superseding receipts, rule 13); atomic writes; **reproduce the reduced-fine receipt to the cent from committed code**. Then **Phase 2** on the top-up: `PM_PLUS_FINE` vs `+PRED_STATE_V1` vs ONE pinned-capacity LGBM (hyperparams pinned in-file BEFORE scoring), three heads reported separately, ≥200 matched randoms, multiplicity incremented per candidate | Phase-0 gate: a fresh process loads one manifest and reproduces the named development scores without fitting or reading growing raw data. **DO NOT self-freeze; the freeze receipt is drafted as an ASK** |
| **DA** (`pmmm-da`) | **Phase 1** — deliverable 2 `harmful_state_features.py`: the §4 feature list (time_remaining, true gen age, remaining-size/queue-ahead, exact-level depletion/replenish velocity 50/250/1000 ms, side fill-shares, PM touch-move age, feed freshness, missing/stale flags — NO inventory/skew/lifecycle) + the §4 correctness battery (feature_asof audits, post-cutoff synthetic event test, cancel-vs-execution distinction, statuses not drops). Plus: materialize the declared top-up receipt; eth exposure-rows stewardship; **verify the 08-25 daily-admission act for the v2.1 adverse-move race (R-141) ran — file status either way** | Build the top-up receipt FIRST (it gates BE's Phase 2), then the feature builder |
| **DE** (`cta`) | **Phase 3** — deliverable 4 `harmful_stateful_policy.py`: per-(slug,side,generation) state machine `LIVE → CANCEL_PENDING → HELD → REPOST_ELIGIBLE → LIVE`; first-crossing single-cancel; latency-window fills stay stale; dwell + `theta_repost < theta_cancel`; fresh generation = fresh queue + declared reset cost; reducing/increasing reported separately; rate limits count requested/effective/suppressed. **Parity battery is the gate:** disabled predictor and infinite threshold bit-identical to `QR_SKEW_ONLY`; cancel-and-hold equivalence; one cancel per generation; no policy trajectory reused as training population. Then **Phase 4** grids (5–250 ms × queue-reset-cost × budget) on the unchanged reduced-fine score. Q-DE-14 is CLOSED SUPERSEDED (R-145(5)) — do not resurrect the old optimizer protocol | Buildable NOW against `harmful_scores_*_v3` + the v3.4 exposure dataset; nothing waits on Phase 2 |
| **OPS** (`pmmm-ops`) | Enforce the heavy-run standing rule (audit `ulimit -v`, publish the launch pattern); **measure recv_ns receive-latency p99 degradation under heavy co-located load — design + metric declared in-file before reading**; collector watch (4 processes: collect_hf, collect_hl, collect_pm_prices, collect_pm); keep the timer/lane supervision | File the measurement design as a FILING before running it |

## Blocking discipline
Exactly one decision-blocking item at a time. **RIGHT NOW: BE's Phase-0
cent-exact reproduction.** It gates the user's freeze of reduced(+extended)
AND every later phase's receipts. Everything else above is parallel. The
forward clock does not start until the user freezes; if the freeze lands
during 08-26, day one is 08-27 and the earliest honest verdict is ~08-31.

## Data budget (user correction 2026-08-25 — still binding)
Sub-second Binance data exists ONLY from 2026-08-24 13:48:54 UTC. The 08-20..22
dataset is PM-only/≥1s work; its Binance side is legacy-stamped. Fine-feature
development tape = the consumed fragment + the declared top-up, nothing else.
Accrual, not cleverness, is the schedule.
