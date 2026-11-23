# MAKER-TRACE-RECONSTRUCTION-v1 — results (how the top makers make money)

2026-11-23, session aea5bfb9 (development).

**Provenance:**
- Frozen `fe0a9cbce`.
- Amendment 1 `05177007a`: the gate refused at 1.4046 h > 1.35, and the USER approved cap 1.45.
- Amendment 2 `654cdb06d`: a memory-only redesign of the report and the checker, with identical outputs.

**Numbers:** every number below is a value in [`claims.json`](claims.json), computed by `tools/claims.py` from the frozen report's [`outputs/`](outputs/) and the checker receipt [`checks_20261123T011521Z.json`](checks_20261123T011521Z.json).

**Scope:**
- Consumed days only: Oct 23 for selection; Oct 24–27 for measurement; Nov 8 for the own-order truth.
- **4 days: point estimates, no intervals** (rule 8).
- USD; c = cents per share.

**What this is not:** descriptive only. No strategy, fit or profitability claim for us.

## Run

| Stage | Result |
|---|---|
| Extraction (Oct 23–27, Nov 8) | all rc 0; 326–471 CPU s per day; peak ≤ 1,039 MiB; Oct 24: 476,897 unique records, 724,873 legs (equal to step1's independent count) |
| Lifecycle | 19 BOOK + 52 GT markets, rc 0. On Nov 8 the public-leg rule selected exactly the 52 own-fill windows the private truth names |
| Econ / truth / report | rc 0. The truth dry gate reproduces table v2 exactly |
| Independent checker | **32 / 39 checks pass** (`claims.json → checker`); failures below |
| CPU | 1.341 h of the 1.45 cap; line 4.65 ≤ 4.85 |

## Readings

### R1 — priority: `supported`

| Test | Violations | Rate | Certain violations |
|---|---|---|---|
| T1 (price priority) | 1 / 18,301 | 5.5e-5 | 0 |
| T2 (time priority) | 3 / 15,900 | 1.9e-4 | 0 |

The checker recounts T2 as 5 records. Both rates stay below 1% (`both_below_1pct_even_with_checker_counts`).

### R2 — ground truth: `no_defect`

- 86 / 86 of our filled Nov 8 orders are covered.
- **0 defects** in every scenario.
- **Caveat:** only filled orders can be scored, and the clock premise is stated in `readings.json`.

### R3 — coverage (BOOK markets; the ex-ante top 10 by maker shares)

8,198 filled orders. Fractions of orders by class:

| Field | Observed | Constrained | Assumption-sensitive | Unresolved |
|---|---|---|---|---|
| Size | 0.46 (the lower bound) | 0.003 | 0.54 | 0 |
| Posting | 0 | 0.34 | 0.64 | 0.02 |
| Persistence | 0.03 | 0 | 0 | 0.97 (most orders fill once) |
| Queue | 0 | 0 | 0.79 | 0.21 |
| Removal | 0 | 0.17 | 0.83 | 0.001 |

**R3 is PROVISIONAL.** Five of the seven checker failures concern these lifecycle labels and bounds (see below).

### R4 — economics

#### Pooled Oct 24–27 (`claims.json → exante_sh.*`)

| Group | Addresses | Net P&L | Round trip | Residual | Fees | Maker legs | Taker legs | Net per share | Days positive |
|---|---|---|---|---|---|---|---|---|---|
| Ex-ante top 10 (maker share 0.807) | 10 | +35,586 | +130,789 | −72,444 | −22,759 | +7,509 | +28,077 | +0.178 c | mixed (+8.6k, −14.3k, +42.0k, −0.7k) |
| Other makers | 2,943 | +110,136 | +356,551 | −168,077 | −78,338 | −26,739 | +136,875 | +0.219 c | all 4 |
| Pure takers | 18,346 | −427,378 | | | −180,559 | | | −1.234 c | none |

- **Identities:** gross = round trip + residual, and net = gross − fees, hold in every row.
- **Zero sum:** Σ net over all addresses = −281,656 = −Σ fees (`zero_sum`).

#### Where the maker-leg money is (top 10 and field)

All values are maker legs per share, pooled.

| Cut | Top 10 | Field | Days it holds (top 10 / field) |
|---|---|---|---|
| By distance from the best (d0 / d1 / d2 / d3+) | −0.16 / +0.25 / +0.81 / +0.87 c | −0.18 / +0.35 / +0.71 / +0.98 c | 2 / 2 (monotone; only 2 of 4 days) |
| Sweep fills vs single-price fills | +1.48 vs −0.58 c | +1.29 vs −0.50 c | 4 / 4 |
| First minute vs last minute of the window | +0.81 vs −1.50 c | +0.37 vs −0.40 c | 4 / 3 |
| Own taker legs, 1 s markout | +1.21 c | +0.16 c | 4 / 3 |

Pooled, the depth gradient rises monotonically for both groups, but on only 2 of 4 days.

#### Other incentives, beside (never inside) the trading P&L

- **Advertised 25% taker-fee rebate (unverified):** +24,139 for the top 10, which is 0.68 of their net; +46,275 for the field, 0.42 of its net.
- **Liquidity-reward pool:** 72,000 units per day (units unverified). All makers' net trading P&L was 17.8k–63.3k USD per day. Per-maker rewards are unidentifiable.

#### Luck, heterogeneity and persistence

- **Luck:** the top 10's end-of-window inventory outcome lies inside the Bernoulli(end mid) null on every day (percentiles 0.20–0.52; 1,000 draws).
- **Heterogeneity:** 8 of 10 members are positive pooled, but only 3 are positive on every day. All three are pure makers (maker share 1.0) whose residual is small beside their round trips.
- **Persistence:** a member sits in the daily top 10 on 2.4 of 4 days on average.

### What the computed values say about the question (descriptive; consumed days)

1. **Top-of-book quoting does not pay.**
   - On the top makers' own maker fills, fills at the best price lose after adverse selection.
   - The maker-leg money comes from fills behind the best and from large takers that sweep several levels.
   - It is concentrated early in the window; the last minute loses.
2. **A large share of "maker" profit is aggressive.**
   - The top 10 earn more net on their taker legs than on their maker legs; the field makes all of its profit there.
   - Those taker fills move in their favour within 1 s (positive 1 s markout).
3. **Inventory risk halves the turnover profit.** The residual loses about half the round-trip gain, and the end-of-window resolution part is consistent with luck.
4. **Incentives are of the same order as the trading edge.** The advertised rebate would add about 68% to the top 10's net; the reward pool's units are unverified.
5. **Consistency comes from flat inventory.** Only three addresses earn every day, and they are pure makers that keep inventory flat.

## Checker failures (open; not result-changing for R4, provisional for R3)

| Check | What it reports | Direction / impact |
|---|---|---|
| `posting_capacity_bound` (+ re-parse `rp_capacity`) | the lifecycle posting lower bound is EARLIER than the checker's single-order capacity bound | looser bound; the checker may not apply the hard `hi − 2 s` fill end or unknown cells. Not an over-claim; coverage may be understated |
| `class_scenario_consistency` | queue classes labelled `unresolved` where scenario values differ (24,165 consistent, 71 market-scenario fails) | labelling rule mismatch; queue coverage understated |
| `priority_exhaustion`, `priority_ahead_lo` | price-priority evidence not turned into exhaustion / `ahead_lo` on some rows | fewer S_FIFO claims than the evidence allows |
| `priority_tests_T1_T2` | T2 counted 3 by the lifecycle vs 5 by the checker (2 markets) | R1 unchanged |
| `H2_presence` | 3 rows (one GT market) whose persistence span differs from [min hi, max lo] | to resolve |
| `tob_from_levels` | about 0.6% of BOOK frames whose top-of-book SIZE differs from the depth levels; prices agree | econ uses prices, not sizes |

**Passing checks:** all extraction integrity, all economics (conservation, an independent P&L recompute for every address, the ex-ante identities), ground-truth scores and coverage arithmetic.

**Next round:** resolving the lifecycle checks needs a lifecycle and checker fix plus re-runs. That is about 0.1 h, beyond the 0.05 h review reserve, so it is a USER resource decision. The checker run itself costs 230 CPU s.
