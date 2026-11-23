# MAKER-TRACE-RECONSTRUCTION-v1 — results, run v2 (supersedes the v1 receipt after the lifecycle fix)

2026-11-23, session aea5bfb9.

**What v2 is:**
- The run re-uses the v1 extraction (unchanged, linked read-only).
- It re-runs lifecycle → truth score → econ → report → checks under `MT_RUN=v2`, after amendment 4 (`40ee591ea`) settled the 7 v1 checker failures: 5 lifecycle bugs and 2 checker bugs, fixed.

**Where the numbers come from:** [`claims.json`](claims.json), computed by `tools/claims.py` from [`outputs/`](outputs/). The checker receipt is [`checks_20261123T023555Z.json`](checks_20261123T023555Z.json).

**Superseded:** the v1 receipt (`../report.md`) is kept as provenance. Its numbers are void where they differ.

**Scope:** consumed days only; 4 measurement days, so point estimates with no intervals (rule 8). USD; c = cents per share. Descriptive: no strategy or profitability claim for us.

## Instruments

| Instrument | Result |
|---|---|
| Independent checker (v2 outputs) | **39 / 39 checks pass**; raw re-parse of 20261024_3697902 and 20261108_4332392: all checks pass |
| End-to-end rehearsal (real builders on synthetic raw data), run 9 | 39 / 39 checks, 22 / 22 re-parse checks, 76 / 76 falsifier cases |
| Lifecycle selftest / simulator | 199 / 199; 0 of 23,668 observed / constrained claims fail with 0–1,000 ms out-of-order settle lags |

## Readings

### R1 — priority: `supported`

| Test | Violations | Rate | Certain violations |
|---|---|---|---|
| T1 (price priority) | 1 / 18,433 | 5.4e-5 | 0 |
| T2 (time priority) | 5 / 16,433 | 3.0e-4 | 0 |

The lifecycle and the checker agree on these counts.

### R2 — ground truth (Nov 8, our own orders): `no_defect`

- 86 / 86 filled own orders are covered.
- 0 defects in every scenario.
- Only filled orders can be scored, and the clock premise is stated in `outputs/readings.json`.

### R3 — coverage (BOOK; the ex-ante top 10 by maker shares)

Fractions of filled orders by class:

| Field | Observed | Constrained | Assumption-sensitive | Unresolved |
|---|---|---|---|---|
| Size | 0.46 (the lower bound) | 0.003 | 0.537 | 0 |
| Posting | 0 | 0.341 | 0.642 | 0.017 |
| Persistence | 0.032 | 0 | 0 | 0.968 (most orders fill once) |
| Queue | 0 | 0 | 0.792 | 0.208 |
| Removal | 0 | 0.169 | 0.83 | 0.001 |

"Assumption-sensitive" means S_FIFO-dependent. R1 supports price-time priority, and R2 found no defect.

### R4 — economics, Oct 24–27 (`claims.json → exante_sh.*`)

| Group | Net | Round trip | Residual | Fees | Maker legs | Taker legs | Per share | Day signs |
|---|---|---|---|---|---|---|---|---|
| Ex-ante top 10 | +35,586 | +130,789 | −72,444 | −22,759 | +7,509 | +28,077 | +0.178 c | mixed |
| Other makers (2,943) | +110,136 | +356,551 | −168,077 | −78,338 | −26,739 | +136,875 | +0.219 c | all positive |
| Pure takers (18,346) | −427,378 | | | −180,559 | | | −1.234 c | all negative |

Across all addresses, Σ net = −Σ fees.

**Maker legs per share** (pooled; the days on which each holds are counted in `claims.json`):

| Cut | Top 10 | Field | Days it holds (top 10 / field) |
|---|---|---|---|
| By distance from the best (d0 / d1 / d2 / d3+) | −0.16 / +0.25 / +0.81 / +0.87 c | −0.18 / +0.35 / +0.71 / +0.98 c | 2 / 2 |
| Sweeps vs single-price fills | +1.48 vs −0.58 c | +1.29 vs −0.50 c | 4 / 4 |
| First minute vs last minute | +0.81 vs −1.50 c | +0.37 vs −0.40 c | 4 / 3 |
| Own taker legs, 1 s markout | +1.21 c | +0.16 c | 4 / 3 |

**Incentives, beside (never inside) the trading P&L:**
- Advertised 25% taker-fee rebate (unverified): 0.68 of the top 10's net and 0.42 of the field's.
- Reward pool: 72,000 units per day (units unverified). Per-maker rewards are unidentifiable.

**Luck and consistency:**
- The top 10's end-of-window resolution part is inside the luck null every day.
- Only 3 of the top 10 are positive every day: the pure makers with flat inventory.

## Decision paths of the top 3

To follow in this receipt's decision section after the frozen decision-path run (`decision_contract.md`).
