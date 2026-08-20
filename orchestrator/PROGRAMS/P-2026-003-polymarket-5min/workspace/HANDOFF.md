# HANDOFF — P-2026-003 Polymarket Crypto 5-min

Updated: 2026-08-20, session 2. All work is on branch **`mm-research`** (pushed);
nothing is on `main`. Latest commit `80f823e`.

## Read this first

The programme now has a **verified settlement target**, a **modular
architecture with machine-readable contracts**, and a **finalized plan for the
volatility estimator awaiting user review**. Two headline results from session 1
have been **withdrawn or downgraded** on discovering that the data underneath
them was wrong. Do not cite either without reading §"What was withdrawn".

Reading order:
1. `live/pm_research/PM_ARCHITECTURE.md` (v12) — the entry point; structure.
2. `live/pm_research/contracts/contracts.yaml` — machine-readable source of
   truth for types. The prose defers to this file, not the other way round.
3. `live/pm_research/EXP_RESULTS_2026-08-20.md` — first model results.
4. `live/pm_research/SIGMA_PLAN.md` — **the live document**; user is reviewing.
5. `live/pm_research/plans/` — BE_BELIEF, BE_FLOWANDFILLS, MEASUREMENT,
   PRELIMINARY.

## Done this session

**E-M6 settlement truth — PASSED, the foundation gate is cleared.**
`S60(T) vs S60(t0)` reproduces the winners Polymarket actually paid on
**1,465 windows at 99.8%** (99.9% restricted to `|margin| > 0.5 bp`). This
settles the open `w = 60 s` vs `300 s` ambiguity: the full-range reading scores
86.9% and is **refuted**. Reading the same grid at knowledge time gives 99.3%,
and that 0.5 pp gap is the size of the look-ahead a careless backtest banks.

**E0 data integrity audit** (`exp_e0_data_audit.py`, 7 checks) quantifies every
known incident rather than assuming it benign: the duplicate-collector overlap,
the ~16 min market-side outage, the 8 malformed resolution rows, restart shards,
TWAP gaps, knowledge-time lag per coin, and the up-rate drift confound.

**Architecture v12 + machine-readable contracts.** 6 planes, 85 contract types,
a structural diff checker with 13 selftests, and version-bound migration
records. Twelve external review iterations. **Two of my own artefacts were
proven unsound during that loop and replaced** — worth knowing because both
failures were of the same kind, a checker that reported success without
checking:
- the first contract checker: an invalid ref exited 0, generics were invisible,
  and narrowing a type produced an identical inventory. All three reproduced.
- the path-keyed allowlist (M11-1): entries authorised **any** change at that
  path. Reproduced `CouplingSource -> CompetitionState` passing. Replaced with
  migration records bound to (operation, key, old, new, version).

**Collector MNAR bug found and fixed.** The hot loop allocated an `asyncio`
timer per message; at BTC's rate that dominated and backed the server's send
buffer up into `1013 slow consumer` disconnects. **27 of 47 disconnects were our
own doing, and 32 of 47 were BTC** — i.e. the loss was concentrated in exactly
the busiest intervals, which is missing-not-at-random. Post-fix: **0 drops
vs 28**. Never pool an unpaired statistic across the fix boundary.

**SIGMA_PLAN finalized** (`80f823e`) — see §"Immediate next step".

## What was withdrawn — do not cite these

**1. "The book beats our model at every horizon" — WITHDRAWN, not held.**
The 2026-08-20 run showed the book winning by a stable 2.5–3.2 Brier points at
all six horizons, which read as a uniform information deficit and prompted the
conclusion *"no alpha, therefore pure market making"*. That model was
**mis-anchored**: `E_t[X_T]` used the trailing S60, which lags spot by
`w/2 ≈ 30 s`, while being paired with a *conditional* variance law. The
resulting `σ_eff` was ~2.6× too small at `r = 30`. The nowcast
`P̂ = 2·S30 − S60` fixes it and gains **−0.0101 Brier pooled, at every
horizon**. The deficit was the anchor, not sigma. The verdict must be re-read
on the corrected specification before it means anything.

**2. The FLB edge — downgraded from an edge to a rounding error.**
"+3.6 c/share at `p ∈ [0.15, 0.35)`, stable" was the measurement that
recommended the Option-B re-scope. It was computed on `book` snapshots, which
are **p90 6.2 s stale**. Rebuilt from `price_change.best_bid/ask` (the
executable quotes): `b̂ = 1.145` where the stale read inflated it to 1.182, the
walk-forward gain is **0.0004 Brier**, and the effect is **one-sided** — a drift
signature rather than a genuine bias.

**3. Everything book-derived in `PM_DEEP_REVIEW.md` inherits defect 2**,
including the "+95 bps maker gross / +136 with rebate". Treat as unverified
until re-measured on `price_change` quotes.

**4. Five premises I had briefed into the flow/fills work were wrong**, all
corrected in `plans/BE_FLOWANDFILLS_PLAN.md`: trades are NOT double-reported
(zero dupes); the modal spread is **1 tick**, not 2–4 (ATM runs 6–8 c); the
1.7 s lag is the **signal clock only** — the book itself arrives in **47 ms**,
which materially softens the session-1 FATAL-1 latency finding; `side` is the
**taker's** (90.8%); and `price_change` carries **post**-change quotes.

## ⛔ Decision point A/B is still open, but B lost its prior

Session 1 closed on Option A (fix the mechanism program) vs Option B (re-scope
to FLB harvest), with the reviewer *and the data* recommending B. **The data
that recommended B was measured on stale books** (withdrawal 2 above). Re-frame
the choice before making it; do not read `PM_MM_PLAN §17`'s recommendation as
current.

## Immediate next step — SIGMA_PLAN Phase 0

`SIGMA_PLAN.md` is finalized and with the user for review. Three decisions are
recorded at its head:

- **D1** build order inverted — measure `ω_P` and `c(r)` *before* building the
  variogram and blend, since both are cheap, run on data already on disk, and
  either can invalidate the design.
- **D2** `c(r)` is **expected to breach** the G3 `[0.8, 1.25]` band at short
  `r` — two independent lines say so (`σ_eff(30) = 1.77 bps` vs realised
  ~2.6 bps under the nowcast ⇒ `c(30) ≈ 2.1` in variance; and BTC's
  `ŵ ≈ 47 s` against the fixed 60 s ⇒ 1.63×). The plan's own rule stands: a
  breach means the parametric law is failing, **not** that the band should
  widen. Budget for redesign.
- **D3** size the build to σ's actual job — §0 scopes σ down to risk plumbing,
  so a single-scale per-symbol estimate may suffice and the multi-scale blend
  must beat that baseline before it is built.

**Phase 0, gated** (all on existing data): fix the anchor in a shared helper →
verify S30/S60 semantics against the 1 s Binance tape → measure `ω_P` (it
enters **undamped** at `r > w`, which is where we quote) → measure `c(r)` (the
go/no-go) → re-read the book-beat verdict on the corrected spec. Only if that
clears does Phase 1 build the variogram, blend, PIT and contract carrier.

The book-beat re-read is deliberately inside Phase 0: it is paired so it
survives the MNAR gap, and §0's whole argument for scoping σ down *assumes* the
book supplies the level.

## Then

- **G-FF4 queue bracket** (`plans/BE_FLOWANDFILLS_PLAN.md`) — potentially
  **programme-ending**. If the bracket on `Q_ahead` is wide enough, MM
  profitability is not knowable from data we can collect. Effort saved by D3
  goes here.
- Structure review loop is at 12 iterations and **not converged**
  (LOCAL 11 / SPREADING 1 / STRUCTURAL 1). Iteration 12 produced *zero*
  documentation change — it was a checker patch labelled as an architecture
  version. Semantic rules/producers/ports, scenario type consistency,
  incentive-to-outcome wiring, decision-snapshot equality and ModuleManifest
  coverage all remain unresolved.
- `pm_backfill.py` — historical windows. Gamma canNOT resolve these; enumerate
  via paginated CLOB `GET /markets`. Note calibration may **never** cross the
  2026-08-07 rule change (snapshot → 60 s TWAP, after the Stanford/SMU
  manipulation study).

## Standing rules (each one paid for)

- No design decision that a measurement on existing data could settle may be
  recorded as settled until that measurement is run.
- Read book state from `price_change.best_bid/ask`, **never** `book` snapshots.
- Read everything at knowledge time (`recv_ns`), never payload timestamps.
- Dedup prices by `(timestamp, symbol)`, raw by message identity — `recv_ns`
  differs per process so exact-line dedup does **not** catch a duplicate
  collector. Check with `ps -eo pid,etimes,cmd | grep live/pm_research`;
  pgrep patterns must include the `pm_research/` path segment.
- `fee_rate_bps` must be read from trades, not assumed. Currently 0 on every
  trade observed, which conflicts with both the docs (7%) and CLOB `base_fee`.

## Watch out for

- `resolutions.jsonl` holds 8 garbage rows from the first hour — filter through
  `is_final`, dedupe by slug keeping the final row.
- Thin windows exist (a BTC window with `volumeNum = 2`). Stratify by volume;
  never average across dead and live windows.
- The tick regime changes away from the money (0.01 → 0.001), which makes the
  tick a first-order economic parameter far from ATM.
- `PM_THEORY_CHECK_ORCHESTRATOR.md` §2's claim that "MM-under-obligation theory
  doesn't exist" was a **search failure** — the body is principal–agent MM
  contracts (El Euch et al.; Baldacci et al.). Marked superseded; don't cite it.
- Trading access (US status, CLOB auth) is a deployment question, out of scope
  until the gates pass. Don't let it leak into research design.
