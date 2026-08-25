# CLAUDE.md

Project instructions for Claude Code working in this repo. See `AGENTS.md`
for the full agent guide; this is the Claude Code-specific summary.

## Project type

**Research code, not production.** ML pipeline for predicting alpha-residual
on Binance USDM perpetual futures using free public data. No live trading,
no exchange integrations, no execution server.

## Build & test

```bash
pip install -r requirements.txt
# No formal tests in this repo; reproduction is via the probe scripts in
# ml/research/. See README.md "Quick start".
```

## Running probes

```bash
# Cross-sectional v4 (25-symbol portfolio) — ~15 min
python3 -m ml.research.alpha_v4_xs

# v3 alpha-tailored (3-symbol pair) — needs aggTrades
python3 -m ml.research.alpha_v3

# Audits
python3 -m ml.research.alpha_feature_audit
python3 -m ml.research.alpha_v3_audit
```

Caches to `data/ml/cache/`; subsequent runs reuse them.

## Architecture

```
features_ml/  Feature pipeline (klines, regime, trade_flow, cross_asset,
              alpha_features, cross_sectional, labels)
ml/           CV, cost model, research probes (alpha_v2/v3/v4 + audits)
data_collectors/  Binance Vision daily archive loader
hf_features.py    Legacy 160+ kline indicators (used by features_ml/klines.py)
docs/         METHODOLOGY_REVIEW (audit trail), STATUS, HANDOFF
orchestrator/PROGRAMS/P-2026-001-ml-cta-engine/  Original program plan + results
```

## Key conventions

- **Point-in-time features only.** Rolling stats use trailing windows; beta
  and z-score features `.shift(1)` to avoid using current bar.
- **Pooled training across symbols** with `sym_id` indicator. The model
  trains on stacked per-symbol panels.
- **Walk-forward CV** with embargo (1 day) and label purging via `exit_time`.
  Use the `_expanding_train` helper pattern in each `alpha_v*.py`.
- **LGBM hyperparameters pinned** across v1→v4 for fair comparison. Don't
  retune per-version without flagging it.
- **Cost model is retail VIP-0** (~12 bps RT naked, ~24 bps hedged). Most
  conclusions stated relative to this.

## Look-ahead pitfalls (real bugs found during research)

1. **Target normalization** — `rolling.shift(1)` is wrong for h-bar forward
   returns; must be `.shift(horizon)`. Fixed in all `alpha_v*.py`.
2. **VPIN buckets** — was sized using full-dataset volume; now trailing 7d
   per bar. Fixed in `features_ml/trade_flow.py::_vpin`.

3. **Incremental cache updaters must MERGE, never overwrite** — a partial-range
   metrics top-up with overwrite semantics destroyed 163/176 symbol histories
   (2026-07-07; recovered from Binance Vision). Fixed in
   `data_collectors/metrics_loader.py`; a failed merge now aborts the write.
4. **Paired replays through path-coupled overlays amplify prediction noise
   ~10-20x** — the DD-stop and binary regime gate bifurcate on tiny equity-path
   differences, so variant-vs-incumbent replay deltas can be huge with zero
   ranking improvement (beta-label A/B, 2026-07-08). Measure label/feature
   variants at book level (rank-IC, top/bot-K spread); replay only with
   overlays disabled.
5. **Windowed recomputes must respect unbounded features** — `bars_since_high`
   is cumcount-based (not bounded by its 288 window; empirical max ~12d) and its
   cross-sectional rank depends on the full universe being present. Both have
   runtime parity guards (`incremental_xs_feats.py`, `incremental_panel.py`,
   2026-07-08); keep the guards when touching those paths.

When adding a new feature, sanity-check IC against forward return shifted
by +1 bar. Anything >+0.10 IC is suspicious and probably has hidden look-ahead.

## Reliability rules (hard requirements — every result-bearing session)

Distilled from failures that each cost real work (2026-08-23..25: four dissolved
positives, one voided freeze, an eight-issue dataset rebuild). These are
checkable requirements, not style.

**Labels and populations**
1. **Never train on an outcome-selected population.** Fills are endogenous:
   training on completed fills conditions on the event a policy wants to
   prevent. The unit is the decision-time exposure (a generation interval),
   built from the neutral no-cancel reference path.
2. **Rows are actions.** One row per cancellable generation. If several rows
   can share one outcome, the evaluator must de-duplicate to actions or the
   result is inflated (measured: 1.99 rows/fill, max 23).
3. **Timestamps come from the event that carries them**, never a nearby proxy
   (a resync clock cost 22–162 ms of label error). Every fill/tranche is valued
   at its own time and level. Add a reconciliation selftest against the
   generating engine; a mismatch fails the build, never absorbed.
4. **Exclusions are statuses, never silent drops** (gap, truncation, no future
   mid). Report their counts with every table.
5. **Era purity is a per-event predicate.** Collector/stamp changes (see
   `data/mm_hf/collector_runs.jsonl`) truncate admissible data at the boundary
   by `recv_ns`, not by file. Legacy-stamped data is inadmissible for any
   sub-second feature.
   **Sub-second-reliable Binance data exists ONLY from 2026-08-24 13:48:54 UTC**
   (`recv_ns >= 1787579334881534478`, the hf_ws_v2 stamp boundary in the
   ledger). Before it, rows were stamped post-parse: p50 latency is fine but
   p99 carries up to ~0.6 s of parse-backlog error, concentrated in bursts —
   exactly when fine features matter. Pre-boundary mm_hf tape is usable for
   ≥1 s bars only, and any receipt using it must say so.

**Evaluation**
6. **Declare the null before the result**: design AND minimum sample
   (≥200 permutations / draws). A null max is an extreme-order statistic —
   an under-sampled correct null flatters as much as a wrong one.
7. **Controls must be matched on the decision variable** (action count, side,
   hour) and compared on the DECISION metric (net value, rho = adverse/spread),
   not on a proxy like harm share. Latency enters the estimand: value only
   tranches after t + L.
8. **Intervals only on the correct cluster unit** (UTC day here). Below G=5
   complete days: point estimate, no interval, say so. Every quoted population
   carries its n AND as-of — the tape grows during measurement.
9. **A baseline must remove the tautology.** If the target is derived from an
   input (PM binaries settle on a Binance-derived price), report skill only
   incremental to that input; skill vs base rate is meaningless.
10. **Compute predicates, never print conclusions.** A hardcoded verdict string
    beside a table has contradicted the table three times. If a number is
    claimed ("2.4x", "excludes zero", "monotone"), the code must evaluate it.

**Selection and freezing**
11. **Choosing after seeing voids the test.** Feature subsets, thresholds,
    horizons, window lengths picked on data X may not be validated on X.
    Seen days are consumed; name them (08-20..25 are consumed for the harmful-
    fill line). Validation = later untouched days, ≥5 complete UTC days.
12. **A freeze is a commit.** Candidate = builder file committed (hash + commit
    ref in the receipt), full pipeline in the repo (data → target → fit →
    artifact; a scratch-dir builder voided one freeze), declared nulls inside
    the receipt, and the count of candidates in the forward race (multiplicity)
    recorded at freeze time.
13. **Corrections supersede in-band**: a superseding receipt (vN+1), because
    automated readers resolve receipt fields, not sidecar annotations. Never
    edit a frozen artifact; the old receipt stays as provenance.
14. **Models estimate; they never decide.** No worker-produced boolean encodes
    an entitlement (decision-eligible, admissible). Decisions live in the
    policy layer with their own priced trade-offs.

**Instruments**
15. **Every checker ships a falsifier**: a positive control it must flag and a
    known-bad input it must refuse. A zero from an instrument that never proved
    it can fire is not a result (a silent regex mismatch once reported a clean
    surface).
16. **Verify at the artifact a claim names** — not a proxy, not memory, not a
    report. Grep hits on vocabulary are not references; match identity
    (Type.field), and know what KIND of document you are reading.

## What's in/out of scope

**In scope** (free to edit):
- `features_ml/`, `ml/`, `data_collectors/`, `scripts/`, `docs/`
- `hf_features.py` (legacy but in-repo)
- `orchestrator/PROGRAMS/P-2026-001-ml-cta-engine/` (research record)
- `live/` (paper-trading harness, added 2026-05-01) — multi-symbol
  orchestrator, basis-risk diagnostics, Binance-train / Hyperliquid-execute
  pipeline. Not production-grade; this is the forward-test layer that
  validates v6_clean predictions transport from backtest data to real-time.

**Out of scope** (don't add):
- Live trading code, exchange adapters, execution servers
- Production deployment infra (Docker, CI for trading, etc.)
- Large data files in git — use `.gitignore` and external storage

## Active program tracking (code-relay protocol)

Active research programs live in `orchestrator/PROGRAMS/P-*/` (currently
P-2026-002 HF market making and P-2026-003 Polymarket 5-min). **At session
start read each active program's `workspace/HANDOFF.md`**, and **after each
completed step update that program's `STATUS.yml` (task statuses + flags) and
`workspace/HANDOFF.md` (done / in-progress / next / watch-out-for)**. State
lives there, not in conversation history — write it before context runs long,
not after.

## Where to read first

1. `README.md` — overview + quick start
2. `docs/METHODOLOGY_REVIEW.md` — full audit trail (most important)
3. `docs/STATUS.md` — current state, known issues
4. `docs/HANDOFF.md` — three ranked next-step plans

## Tone

Brief, technical, no hype. Match the existing docs — short headers, tables
where relevant, numbers with units (bps, IC, days). The methodology review
sets the tone for new docs.
