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

## Decision paths of the top 3 (`decision/`; frozen in amendment 5 `afb589d46`; run `mt_x_decision_v2`, 46 CPU s)

**Addresses:** the ex-ante top 3 by Oct 23 maker shares.

| Rank | Address | Markets | Maker legs | Taker legs |
|---|---|---|---|---|
| 1 | 0x6Da6Cb… | 1,029 | 162,636 | 159,377 |
| 2 | 0x7d32c5… | 710 | 117,590 | 16,865 |
| 3 | 0x57FbC8… | 876 | 111,727 | 9,031 |

`decision/address_summary.csv`.

**Method:**
- All rates are per 100 ms grid step over each address's active span, at a decision lag L = 50 ms, pooled over Oct 24–27, from `decision/take_hazard_1d.csv`.
- Claims are in `decision/claims_decision.json`. A claim that flips across L ∈ {0, 50, 250} ms is flagged `lag_sensitive`.
- `p_fair` is the BTC-implied probability of Up (Binance returns since `ws`, trailing volatility). It is a descriptive reference, not their model.

### When they TAKE

**1. To unload inventory.** Take rates are around ten times higher when holding a position than when flat, and they lean toward reducing it:

| Rank | Buys when short / long / flat | Sells when long / short / flat |
|---|---|---|
| 1 | 1.45% / 1.05% / 0.12% | 1.52% / 0.99% / 0.12% |
| 2 | 0.34% / 0.18% / 0.09% | 0.23% / 0.19% / 0.05% |
| 3 | 0.18% / 0.08% / 0.04% | 0.18% / 0.09% / 0.06% |

- `takes_reduce_position` holds for all three: P(sell | long) 0.60 / 0.53 / 0.62 vs P(sell | short) 0.42 / 0.36 / 0.40.
- Takes cluster within 0.5 s of their own last fill: rank 1 2.26% vs 0.53% after more than 60 s; rank 2 0.65% vs 0.20%; rank 3 0.35% vs 0.08%.

**2. In the direction of a large 1-second BTC move.**

| Rank | Sell rate after BTC < −5 bps in 1 s vs flat | Buy rate after > +5 bps |
|---|---|---|
| 1 | 5.41% vs 0.92% | 2.78% |
| 2 | 2.86% vs 0.15% | 1.39% |
| 3 | 5.14% vs 0.06% | 2.65% |

- `take_rate_higher_when_r_1s_aligned` holds for ranks 2 and 3 at every lag.
- For rank 1 it holds at L = 0 / 50 but not at 250 ms (`lag_sensitive`): its reaction is faster than 250 ms.

**3. Not on a fixed stale-quote gap.** No address shows a sharp `g` threshold: the best split's rate ratio is < 2×, against ≥ 10× for the synthetic control. The logistic `g` coefficient is small and positive.

**Time in the window:** the take rate is roughly flat, and lower in the last 30 s; rank 3 never takes there.

### How they POST (filled orders; censored)

- **Centre at the fill:** quotes are centred on `p_fair` within 0.3 c (`quotes_centred_on_p_fair`, every lag). The half-spread at the fill is −0.02 / 0.43 / 1.19 c for ranks 1 / 2 / 3: rank 1's fills come when the fair value has crossed its quote.
- **Centre at posting:** at the latest possible posting time (BOOK only, about 1,900 orders), quotes sit 2.4–4.5 c above `p_fair` on both sides. That is a common level offset, consistent with a `p_fair` calibration gap rather than a skew, and is assumption-sensitive.
- **Inventory skew:** −0.15 to −0.39 c per 100 shares, on both sides, for all three (`quotes_skew_against_inventory`).
- **Reprice:** a new order's price follows 0.66–0.76 of the change in `p_fair` (`reprice_tracks_p_fair`).
- **Pulls:**
  - For ranks 1 and 2, removal windows of possibly-cancelled live orders hold ADVERSE BTC moves beyond a phase-matched null (p = 0.002, 500 draws, every lag).
  - For rank 3 they hold larger BTC moves (p = 0.006) but not specifically adverse ones (p = 0.82).

### Decision path, as computed (descriptive; 4 consumed days)

1. Quote both sides around a BTC-implied fair value.
2. Shade the quotes against inventory, and move them with about 70% of each fair-value change.
3. Ranks 1 and 2 pull quotes when BTC moves against them.
4. Take liquidity mainly to unload inventory, especially within 0.5 s of being filled, and in the direction of a large 1-second BTC move. Rank 1 reacts in under 250 ms.

**What it does not show:** decision times (latency is only bracketed by L), orders that never filled, and their actual model.
