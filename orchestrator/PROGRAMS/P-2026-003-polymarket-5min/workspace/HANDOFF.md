# HANDOFF — P-2026-003 Polymarket Crypto 5-min

Updated: 2026-08-20, session 2. All work is on branch **`mm-research`** (pushed);
nothing is on `main`. Sigma repair reviewed in
`live/pm_research/SIGMA_PLAN_REVIEW_ITER2.md`; verdict **partial, HOLD**.

## Read this first

The programme now has a **verified settlement target**, a **modular
architecture with machine-readable contracts**, and a reviewed sigma plan.
Revision 2 made useful repairs but did **not** close the anchor ledger or the
consumer contract; Phase 0A steps 1–3 are reopened and estimator implementation
remains on **HOLD**. Two headline results from session 1 have been **withdrawn or
downgraded** on discovering that the data underneath them was wrong. Do not cite
either without reading §"What was withdrawn".

Reading order:
1. `live/pm_research/PM_ARCHITECTURE.md` (v12) — the entry point; structure.
2. `live/pm_research/contracts/contracts.yaml` (**v13**) — machine-readable
   source of truth for types. The prose defers to this file, not the other way
   round.
3. `live/pm_research/EXP_RESULTS_2026-08-20.md` — first model results.
4. `live/pm_research/SIGMA_PLAN.md` — **REVISION 3, canonical.** Rewritten in
   place; one estimand, one consumer matrix, one ledger equation. v1/v2 text is
   in git history (`80f823e`, `cc1d0e7`) and nothing defers to it.
5. `live/pm_research/SIGMA_PLAN_REVIEW.md` — first implementation-readiness review.
6. `live/pm_research/SIGMA_PLAN_REVIEW_ITER2.md` — review of Revision 2; its six
   items are applied in Revision 3.
7. `live/pm_research/sigma_kernels.py` — executable model **fixture**, not a
   frozen spec. `--selftest` checks exact arithmetic under a **declared and
   still UNVERIFIED** sampling convention; it does not establish that convention
   against the Chainlink streams.
8. `live/pm_research/plans/` — BE_BELIEF, BE_FLOWANDFILLS, MEASUREMENT,
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

**SIGMA_PLAN reviewed** (`6bea435`) — direction retained, implementation on
hold; see §"Immediate next step".

## What was withdrawn — do not cite these

**1. "The book beats our model at every horizon" — WITHDRAWN, not held.**
The 2026-08-20 run showed the book winning by a stable 2.5–3.2 Brier points at
all six horizons, which read as a uniform information deficit and prompted the
conclusion *"no alpha, therefore pure market making"*. That model was
**mis-anchored**: `E_t[X_T]` used the trailing S60, which lags spot by
`w/2 ≈ 30 s`, while being paired with a *conditional* variance law. The
resulting `σ_eff` was ~2.6× too small at `r = 30`. The nowcast
`P̂ = 2·S30 − S60` fixes it and gains **−0.0101 Brier pooled, at every
horizon**. The anchor explains a large share of the deficit; it does not prove
sigma was adequate. The residual verdict must be re-read
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

## Immediate next step — sigma Phase 0A reopened

Revision 2 keeps several sound repairs: the normalised-arithmetic coordinate is
appropriate for an arithmetic settlement mark; the one-second discrete kernel
branches now use one convention; the output is called a physical forecast; the
MNAR claim was corrected; and `c(r)` is explicitly diagnostic. The code and
contract checker selftests pass.

The second review nevertheless rejects the claim that all six MUST-FIX items
are closed. The load-bearing issue is mathematical: `a(r)` is the unconditional
MSE of the chosen `P̂ = 2·S30 − S60` extrapolator, not its conditional variance.
Under the plan's own one-second Brownian model:

- `2·S30 − S60` has MSE `9.5139·σ²`;
- the translation-invariant conditional projection is approximately
  `1.499·S30 − 0.499·S60`, with MSE `8.2590·σ²`;
- therefore the chosen anchor has state-dependent conditional bias, which
  cannot be inserted into the probability law as zero-mean variance.

Five other blockers remain. Actual Chainlink S30/S60 aggregation semantics are
unverified; the claimed floor/ceiling is not an ordered bracket and a scalar
cannot represent the needed S30/S60 error covariance; `sigma_kernels.py` adds a
hidden nugget, accepts invalid/fractional horizons and does not implement the
promised refusal; contracts v13 still bypass the data-adapter/state boundary,
duplicate link ownership, omit the link derivative and target interval, and
state the fit cutoff backwards; and Revision 2 coexists with contradictory v1
claims. Also, `c(30)=1.14` combines the new model line with a 2.6 bps residual
from the old Binance-mid anchor diagnostic, so it is not evidence for the new
S30/S60 anchor.

Do not implement or fit the sigma estimator until these are complete. Full
findings and acceptance tests are in `SIGMA_PLAN_REVIEW_ITER2.md`.

### ITER2 APPLIED — all six, spec only. HOLD stands.

Every ITER2 claim was **reproduced before acting**, not taken on trust:
`α* = 1.499167` (exactly `2700/1801`), conditional variance `8.2590σ²`,
unconditional MSE `9.5139σ²`, unconstrained `1.701/−0.836 @ 8.0912`, all four
adversarial probes, the `2.6 bps` provenance at `SIGMA_DIAGNOSTICS.md:157-175`,
and every M2-6 line reference. All correct.

**M2-1, the load-bearing one: the v2 error was the v1 error one level deeper.**
v1 used a *lagging* anchor. v2 fixed the lag with `P̂ = 2·S30 − S60`, justified
by a **locally linear path** — a trend model — and then entered that estimator's
**unconditional MSE** as a zero-mean variance line. Under the driftless model
used everywhere else, trajectories have no local derivative, so with
`P̂(α) = S60 + α(S30 − S60)`:

```
P̂(α) − P_t = (α − α*)·(S30 − S60) + conditionally zero-mean residual
              └─ known at decision time ─┘
```

`α* = 2700/1801 = 1.4991671`, not 2. So v2 buried an **observable** mean error
in the variance: **sd 1.22 bps against a total `σ_eff(30)` of ~2.4 bps — half
the standard deviation at `r=30`.** Two corollaries: the conditional variance is
**α-independent** (8.2590, not 9.5139 — the gap was squared bias), and v2's
"floor" of 9.5139 is **beaten by `α*` inside v2's own model**, so it was never a
bound.

**The anchor is now a parameter to ESTIMATE, not to assume.** `α*` is what a
declared path model implies, not what the market must do.
`AnchorSpec.alpha_provenance` must read `ESTIMATED` before anything prices.
Preferred estimator, which dissolves three problems at once: **regress the
observed settlement mark `x_T` directly on the observed `(S30, S60)`**. It never
identifies latent Chainlink spot (so S3's identification problem does not
arise), assumes no state model, and is **robust to the unverified sampling
convention** — a regression on the published streams does not care how Chainlink
builds them internally. It runs on the tape, not on Bernoulli outcomes.

Applied: **M2-2** `SamplingConvention` versioned and `UNVERIFIED`, kernel step
reopened (a 1 s shift in fast-stream support moves `α*` 1.4992 → 1.4954, so the
semantics are load-bearing). **M2-3** non-ordered `AnchorErrorEvidence` plus a
2×2 `FeedErrorCov` — the same covariance raises one anchor's variance and lowers
another's, so no ordering exists to exploit. **M2-4** one ledger, no hidden
nugget, exact domain refusal (`Unavailable` is not a float), exact-rational
tests. **M2-5** contracts **v14**, all eight sub-items. **M2-6** the plan
**rewritten in place as Revision 3**, 1024 → 649 lines, not a third overlay;
contradiction scan clean.

**Worth keeping in view:** v2's Monte-Carlo independence test drew the future
innovation independently *by construction* — a test that could not fail. That is
the **fourth** artefact here to report success without checking the thing it
existed to check, after the v8 contract checker, the path-keyed allowlist, and
v13 itself. Assume the next one exists.

**Verify:** `python3 live/pm_research/sigma_kernels.py --selftest` (24 checks) ·
`python3 live/pm_research/contracts/contract_check.py --selftest` ·
`python3 live/pm_research/contracts/contract_check.py HEAD WORKTREE`.

**Next, and neither is code:**
1. **Phase 0A 5 — verify S30/S60 semantics** against the 1 s Binance tape
   (endpoints, weights, update triggers, event-time alignment). Gates the kernel
   and the anchor together.
2. **Phase 0A 6 — estimate `α`** by the direct regression above, with the
   conditional bias in the **mean** and `Ω` as a 2×2.

Still open beyond that: the fallback's own loss function; runtime enforcement of
refusal (only the fixture refuses today); and `BE-Belief`'s absence from the
contracts `modules:` block, which belongs to the structure loop.

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
