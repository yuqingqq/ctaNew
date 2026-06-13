# iter-029 — Pairs / cointegration stat-arb sleeve — NO-CANDIDATE

PHASE 2 (broadened scope). Goal: test a genuinely-different alpha *structure* the
"XS ceiling" (iter-027 feature/model, iter-028 target/horizon) doesn't cover — a
**pairs / cointegration mean-reversion sleeve** (spread reversion, not cross-sectional
rank). Hoped-for value: a fresh, diversifying alpha source for the baseline book.

## Verdict: NO-CANDIDATE — the spread-reversion signal has NEGATIVE gross edge on both universes (fails the #1 transport pre-check before cost is even considered).

## Online SOTA grounding (cited)
- **Engle-Granger + z-score** is the standard parameter-light recipe: regress logP_A
  on logP_B over a trailing window, ADF-test the residual for stationarity, trade the
  spread z-score (enter |z|>~2, exit near 0, stop on blow-out). Re-test cointegration
  on rolling windows because crypto cointegration is **unstable / regime-dependent**
  (Amberdata crypto-pairs series, 2024-25; "cointegration beats correlation" but must
  be continually re-validated or the spread "loses its equilibrium").
- Reported crypto wins exist (BTC-ETH pair ~16% ann / Sharpe ~2.45 in one 2021-24
  study; copula-cointegration variants, Financial Innovation 2024) but the literature
  is explicit that **a sudden drop in correlation / loss of cointegration signals the
  historical relationship is breaking down** — the central transport risk.
- Sources:
  - https://blog.amberdata.io/crypto-pairs-trading-why-cointegration-beats-correlation
  - https://blog.amberdata.io/crypto-pairs-trading-part-2-verifying-mean-reversion-with-adf-and-hurst-tests
  - https://link.springer.com/article/10.1186/s40854-024-00702-7 (copula-cointegration)
  - https://link.springer.com/chapter/10.1007/978-3-031-68974-1_16 (high-correlation stat-arb)

## What was built (PIT, parameter-light)
4h-grid log-price matrices reconstructed from the panels (non-overlapping 4h returns
cumulated; HL70-era 51 syms 2025-03→2026-05, EXT 23 syms 2021-01→2026-05). Engle-Granger
pairs sleeve, all stats estimated **only on a trailing 60d formation window ending t-1**,
re-formed every 30d; ADF t-stat < -2.86 (5% crit, hand-rolled — no statsmodels in env)
selects cointegrated pairs; trade spread z (entry 2.0 / exit 0.5 / stop 3.5), beta-hedged,
equal-weight top-10 pairs, decision lagged one bar, PnL realized t-1→t. Cost 4.5 bps/leg.
Script: `agents_system/research/scratch/iter029_pairs_sleeve.py`.

## Decisive numbers (transport-first)
| variant | universe | GROSS Sharpe | GROSS PnL (bps) | net PnL @4.5bps | avg pairs | cost (bps) |
|---|---|---|---|---|---|---|
| MR, broad, top-10 | HL70 | **-2.53** | **-2,670** | -8,699 | 10 | 6,029 |
| MR, broad, top-10 | EXT | **-0.99** | **-6,900** | -35,213 | 10 | 28,313 |
| MR, e2.5/x0, top-5 | HL70 | -0.49 | -502 | -5,333 | 5 | 4,831 |
| MR, e2.5/x0, top-5 | EXT | -1.02 | -8,462 | -32,461 | 5 | 23,999 |
| MR, MAJORS only, top-8 | HL70 | -1.19 | -1,101 | -6,952 | 6.7 | 5,851 |
| MR, MAJORS only, top-8 | EXT | -0.14 | -1,124 | -26,405 | 6.1 | 25,282 |

**Mean-reversion gross PnL is NEGATIVE in every cell** — broad universe, low-churn
params, and majors-only (ETH/BTC/BNB/LTC/BCH/SOL/ADA/XRP/DOGE/LINK, the most
"stably-cointegrated" set per the literature). It is **not a cost problem**: gross is
negative at ZERO cost on both universes.

### Sign-flip diagnostic (mechanism)
Flipping the trade sign (momentum-on-spread instead of reversion) makes gross POSITIVE
(HL70 +2,670 / EXT +6,900) — i.e. **once a cointegrated crypto spread diverges past 2σ,
it keeps diverging rather than reverting.** This is exactly the documented crypto failure
mode: the cointegration relationship breaks/trends under stress. But momentum-on-spread is
(a) net-negative after cost (HL70 +2,670 − 6,029 = **-3,359**; EXT +6,900 − 28,313 =
**-21,413**), and (b) just relative-momentum between alts = the iter-015 momentum /
iter-022 XS-reversal family already comprehensively rejected (universe sign-flips,
dies at the pred-pool layer). No new door.

### Cost / turnover (realistic, as warned)
Z-crossing churn is severe: ~12k legs/402d (HL70 broad) up to ~57k legs (EXT). Cost
swamps any signal even before the signal itself is shown to be edgeless.

### Diversification (computed for completeness — moot given negative edge)
corr(MR-pairs net, baseline HL70 book PnL) = **+0.006** (essentially uncorrelated), and
+0.03 in the baseline's worst-decile cycles — so it *would* diversify in principle. BUT
the sleeve **loses -4.96 bps mean in those baseline-drawdown cycles** and has a standalone
Sharpe of -2 to -8. An uncorrelated sleeve that bleeds capital does not improve combined
Calmar; it just adds a losing leg. Diversification is irrelevant when the diversifier has
negative expectancy.

## Why it failed (mechanism, ties to the codified walls)
The 4h-horizon crypto cross-section is **BTC-beta-dominated and trend-persistent**: alt-alt
log-price spreads do not revert after a 2σ stretch — they continue (the loser keeps losing
in a correlated deleverage; the winner keeps winning in a melt-up). This is the same
structural force behind the iter-006 root-cause (correlated alt selloff = no idiosyncratic
reversion) and the iter-016 finding (the productive horizon is short and the residual is
near-noise). Cointegration tests pass IN-SAMPLE on the formation window but the
relationship breaks OUT-OF-SAMPLE in the very next window (regime-dependent, the literature's
#1 caveat) — so the PIT-honest sleeve trades a broken equilibrium and loses gross.

## Conclusion
Pairs / cointegration stat-arb is **NOT a viable diversifying sleeve on free 4h crypto data**:
it fails the transport-first pre-check (negative gross Sharpe on BOTH HL70 and EXT), the
cost-realism check (huge z-crossing churn), and although its return stream is uncorrelated to
the baseline, a negative-expectancy uncorrelated sleeve cannot improve combined Calmar. The
spread-reversion premise is empirically false here (spreads trend, don't revert — sign-flip
confirms), consistent with the BTC-dominated trend-persistent structure that drives the whole
strategy family. Closes pairs/cointegration as a standalone sleeve. Honest NO-CANDIDATE.
