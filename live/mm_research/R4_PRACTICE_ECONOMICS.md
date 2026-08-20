# R4 — MM Practice: Economics, Evaluation, Simulation, Pitfalls

Agent R4 of 4, literature + primary-source sweep, 2026-08-19. Scope: the practical
layer — fee arithmetic, spread reality, honest measurement, simulation fidelity,
latency, and where a solo non-colocated operator has any chance.
Companion agents cover theory (inventory/AS models). All fees in bps of notional
unless stated. 1 bp = 0.01%.

---

## 1. Fee / rebate landscape (2026)

### 1.1 Binance USDⓈ-M futures VIP ladder

Anchors (VIP0 2.0/5.0, VIP3 1.2/3.2, VIP9 0.0/1.7) confirmed against 2026
third-party mirrors; the official page (binance.com/en/fee/futureFee) is
JS-rendered and could not be scraped. Middle rows are the long-stable schedule —
**re-verify in the UI before committing capital.** Requirements are 30-day
futures volume (USD) AND BNB balance (both must hold).

| VIP | Maker (bps) | Taker (bps) | 30d futures vol | BNB held |
|----|------|------|-----------|-------|
| 0  | 2.0  | 5.0  | —         | —     |
| 1  | 1.6  | 4.0  | ≥ $15M    | ≥ 25  |
| 2  | 1.4  | 3.5  | ≥ $50M    | ≥ 100 |
| 3  | 1.2  | 3.2  | ≥ $100M   | ≥ 250 |
| 4  | 1.0  | 3.0  | ≥ $600M   | ≥ 500 |
| 5  | 0.8  | 2.7  | ≥ $1B     | ≥ 1,000 |
| 6  | 0.6  | 2.5  | ≥ $2.5B   | ≥ 1,750 |
| 7  | 0.4  | 2.2  | ≥ $5B     | ≥ 3,500 |
| 8  | 0.2  | 2.0  | ≥ $12.5B  | ≥ 4,500 |
| 9  | 0.0  | 1.7  | ≥ $25B    | ≥ 5,500 |

- **BNB discount:** paying fees from BNB in the futures wallet gives an extra
  **10% off** USDT-M fees → VIP0 effective **1.8 maker / 4.5 taker**; VIP3 1.08/2.88.
- **Maker rebate programs** (application-gated, not VIP-automatic):
  - *Futures Liquidity Provider program*: rebates **0.5–0.8 bps**, plus higher
    rate limits + low-latency (colocated) connection. Qualification: institutional
    MM application; CM-futures analog requires ≥ $100M futures (or ≥ $20M spot)
    30d volume just to apply, plus weekly maker-share thresholds (~0.4% of
    platform maker volume excl. BTC/ETH).
  - *New-listing LP promotion*: 0.5 bps maker rebate on eligible new listings.
  - *Altcoin LiquidityBoost* (2025, spot): ≥ $20M 30d vol; maker share 0.5% / 1%
    → 0.5 / 1.0 bps rebate.
- MM-relevant note: MM strategies generate very high volume/capital ratios, so
  VIP1 ($15M/30d ≈ $0.5M/day) is reachable with tens of k$ of capital at high
  churn — but you pay VIP0 fees while climbing (1.8 bps × $15M ≈ $2.7k/cycle).

### 1.2 Hyperliquid (official docs, gitbook)

Tiers on **rolling 14-day weighted volume** (spot counts 2×), assessed daily.

| Tier | 14d volume | Perp taker | Perp maker |
|------|-----------|-------|-------|
| 0 | base    | 4.5 | 1.5 |
| 1 | > $5M   | 4.0 | 1.2 |
| 2 | > $25M  | 3.5 | 0.8 |
| 3 | > $100M | 3.0 | 0.4 |
| 4 | > $500M | 2.8 | 0.0 |
| 5 | > $2B   | 2.6 | 0.0 |
| 6 | > $7B   | 2.4 | 0.0 |

- **Maker rebates require platform maker-volume SHARE** (14d): >0.5% → −0.1 bps;
  >1.5% → −0.2 bps; >3.0% → −0.3 bps. Platform share, not own volume —
  effectively institutional-only.
- **HYPE staking discount** stacks: >10 HYPE 5% … >10k HYPE 20% … >500k 40%.
  E.g. Gold staker (10k HYPE ≈ mid-6-figures $) at base tier: maker 1.2 bps.
- Referral: 4% off taker (first $25M). API/MM node also in AWS Tokyo.

### 1.3 Other venues (brief)

| Venue | Base maker/taker (bps) | Best standard tier | MM program |
|-------|------------------------|--------------------|------------|
| Bybit perp | 2.0 / 5.5 | 0.0 / 3.0 (VIP) | rebate 0.25–1.25 bps; AWS Singapore |
| OKX perp   | 2.0 / 5.0 | **−1.0** / ~2.0 (VIP8) | rebate ~0.5 bps; Alibaba HK |
| dYdX v4    | 1.0 / 5.0 (governance-set) | maker **−1.1** / taker 2.5 at high 30d-vol tiers | rebate is the schedule itself |

Takeaway: **only OKX/dYdX offer negative maker fees on a public ladder**; Binance
and Hyperliquid reserve rebates for institutional-share programs. But dYdX books
are thin and the stack (Cosmos appchain) is different; OKX VIP8 is volume-gated
far beyond solo reach.

---

## 2. Spread economics

### 2.1 Empirical spreads, Binance USDM perps (computed for this report)

Binance Vision publishes daily `bookTicker` archives for UM futures only for
**2023-05-16 → 2024-03-30** (then discontinued — the forward L2/bookTicker
collection in `data/mm_hf/` is therefore the only path to current spread data).
Full-day, time-weighted stats, 2024-03-15 (high-vol day, BTC ≈ $70.5k), computed
from the raw archives (26.7M / 16.7M / 4.4M / 4.4M quote events):

| Symbol | Mid | Tick (bps of mid) | TW avg spread | % time ≤ 1 tick | Event-median (ticks) |
|--------|-----|--------------------|---------------|------------------|-----|
| BTCUSDT | $70,504 | $0.01–0.04 obs. granularity (≤0.006 bps) | **0.026 bps** ($0.18) | n/a (granular book) | ~3–4 × $0.04 |
| ETHUSDT | $3,815 | $0.01 = 0.026 bps | **0.037 bps** | 97.3% | 1 |
| INJUSDT (mid-cap) | $46.25 | $0.001 = 0.216 bps | **0.52 bps** | 63.2% | 2 |
| CHZUSDT (small) | $0.1531 | $0.00001 = 0.653 bps | **0.88 bps** | 77.8% | 1 |

Readings:
- **Majors are pinned at (sub-)tick**: BTC/ETH full spread ≈ 0.03–0.04 bps.
  There is *no spread to capture* net of any retail fee tier.
- **Liquidity-curve decay is shallow**: even a #50–80 name (CHZ) trades at
  < 1 bp full spread most of the day. Multi-bp spreads live only in the tail
  (bottom-decile ADV, new listings, stress windows).
- Cross-checks from literature: avg BTC bid-ask ~0.03% across venues in older
  spot-mixed samples (ResearchGate fig., Bitcoin exchanges study); crypto perp
  microstructure papers focus on funding/integration, not spread levels — the
  MDPI paper below is NOT a spread reference.

### 2.2 Go / no-go arithmetic

Per-fill maker edge (bps of notional), the only equation that matters:

```
E[PnL per fill] = half_spread + rebate − maker_fee − AS(τ*) − hedge/inventory cost
```

where `AS(τ*)` = adverse selection = adverse mid drift between fill and
horizon τ* at which you can realistically flatten (Section 3). Worked rows
(AS left symbolic — measure it, Section 3.2):

| Name / venue / tier | half-spread | maker fee (eff.) | requires AS < | verdict |
|---|---|---|---|---|
| BTC, Binance VIP0+BNB | 0.015 bps | 1.8 bps | −1.79 bps (impossible) | dead |
| BTC, Binance VIP9     | 0.015 bps | 0.0 | 0.015 bps | dead (queue war) |
| INJ-like mid, VIP0+BNB | 0.26 bps | 1.8 | negative | dead |
| INJ-like mid, VIP3+BNB | 0.26 bps | 1.08 | negative | dead at-touch |
| CHZ-like small, VIP1+BNB | 0.44 bps | 1.44 | negative | dead at-touch |
| Tail name @ 5 bps spread, VIP1 | 2.5 bps | 1.44 | 1.06 bps | *marginal* |
| HL mid @ 3 bps spread, base+staking | 1.5 bps | 1.2 | 0.3 bps | *marginal* |
| HL tail @ 8 bps spread, tier1 | 4.0 bps | 1.2 | 2.8 bps | possible |

Break-even full spread ≈ `2 × (maker_fee − rebate + AS)`. At Binance VIP0 with
AS ≈ 0.5–1.5 bps (typical liquid-perp markout magnitudes), that is **≥ 5–7 bps
full spread** — i.e. nothing in the liquid universe. **Fees, not latency, are
the first wall for a solo operator on Binance.**

### 2.3 Literature refs (spreads/microstructure)

- *Temporal Dynamics of Market Microstructure in Cryptocurrency Perpetual
  Futures* (IJFS 14(5):103, 2026, mdpi.com/2227-7072/14/5/103): 26 exchanges ×
  812 symbols, Nov-2025→Jan-2026, rolling GARCH / Bai-Perron / Granger. Finding:
  CEX tier tightly integrated, DEX fragmented; gradual drift, no regime breaks.
  **Funding/integration paper — cite for venue structure, not spread levels.**
- *High-frequency dynamics of Bitcoin futures* (ScienceDirect
  S2214845025001188): BTC futures microstructure, subsecond periodicities
  indicating algorithmic taker flow.
- Tiniç & Sensoy, *Adverse Selection in Cryptocurrency Markets* — AS component
  estimates for crypto (spot-heavy).
- Albers, Cucuringu, Howison, Shestopaloff, *The Market Maker's Dilemma: Fill
  Probability vs. Post-Fill Returns* (SSRN 5074873): **live experiment on
  Binance BTC perp** — fill likelihood negatively correlated with post-fill
  return; viable making must counter-trade book imbalance. The single most
  on-point empirical reference for this project.

---

## 3. Evaluation methodology (core of this brief)

### 3.1 Effective / realized spread decomposition (Huang-Stoll)

Sign convention: `q_t = +1` if trade t is taker-BUY, `−1` if taker-SELL.
`p_t` = trade price, `m_t` = prevailing mid at fill, `m_{t+τ}` = mid τ later.

```
effective half-spread:   es_t      = q_t (p_t − m_t) / m_t
realized half-spread:    rs_t(τ)   = q_t (p_t − m_{t+τ}) / m_t
adverse selection:       Λ_t(τ)    = q_t (m_{t+τ} − m_t) / m_t
identity:                es_t      = rs_t(τ) + Λ_t(τ)
```

**`rs_t(τ)` is maker gross revenue per unit notional** (the maker is the
counterparty of q_t; what the taker loses to horizon τ, the maker keeps).
Maker net edge = `rs(τ) − maker_fee + rebate`. Classical horizon τ = 5 min
(equities convention); for crypto perps use a curve (below), the 5-min point is
far past where an HFT maker flattens. Huang & Stoll (1997, RFS 10(4)) estimate
the same objects via the trade-indicator regression
`Δm_t = (α+β)(S/2) q_{t−1} + ε_t`; the direct decomposition above is the
practitioner form (cf. Bessembinder & Venkataraman, survey chapters).

### 3.2 Markout curves from Binance aggTrades alone

Binance USDM `aggTrades` fields: `price, qty, first/last_trade_id, timestamp
(ms), is_buyer_maker`. Signing: **`is_buyer_maker = true` ⇒ taker SOLD
(q = −1, print at bid); `false` ⇒ taker BOUGHT (q = +1, print at ask).**
A passive-fill markout study with no book data:

1. **Parent-event collapse.** Merge aggTrades with identical (timestamp, side)
   — Binance aggregates at ~100 ms granularity, one taker order can print many
   aggTrades. Treat the merged sweep as one fill event (else SEs are fake).
2. **Trade-based mid proxy.** `m̂_t = (P_lastbuy(t) + P_lastsell(t)) / 2` where
   P_lastbuy/lastsell are the most recent ask-side/bid-side prints ≤ t. When
   both sides printed recently this equals the touch mid; staleness during
   one-sided bursts biases AS *down* — flag windows where the other side is
   > 1 s stale.
3. **Maker-signed markout.** For each event, pretend you were the maker filled
   at `p_t` (side −q_t):
   `μ(τ) = −q_t · (m̂_{t+τ} − p_t) / p_t`, evaluated at
   τ ∈ {0.1, 0.5, 1, 5, 15, 60, 300 s}. μ(0⁺) ≈ +effective half-spread;
   μ(τ) decays by Λ(τ); the **asymptote vs. (fee − rebate) is the go/no-go**.
   Volume-weight and report per symbol × hour-of-day × vol regime.
4. **Interpretation caveat.** This measures the *average* maker's economics.
   Your fills as a marginal entrant at the back of the queue are strictly worse
   (you get filled exactly when the level is swept — "fast fills are bad
   fills": aligrithm.com writeup; SSRN Market Maker's Dilemma). Treat the
   aggTrades markout as an **upper bound**; the queue-position haircut comes
   from simulation (Section 4) and later from live fills.

### 3.3 Statistical treatment

- **Never use per-trade iid SEs** — markouts are massively cross-correlated
  within bursts and within days (understates SE by 1–2 orders of magnitude).
- **Day-clustered inference**: compute daily (or session) mean markouts, t-stat
  across days. dof = #days, honest and simple. (Repo already learned this: the
  `ci()` day-cluster bug — MEMORY 2026-07-21.)
- **Block bootstrap** for anything overlapping: block length ≫ max(markout
  horizon, autocorrelation time) — e.g. 30-min blocks for ≤ 5-min markouts.
  Overlapping-horizon markouts have MA(k) errors: either subsample to
  non-overlapping events or Newey-West with lag ≥ overlap.
- **Regime splits are mandatory**: spread/AS regimes shift (2023 ≠ 2026);
  report per-quarter, demand sign stability, not pooled significance.

---

## 4. Simulation

### 4.1 hftbacktest (github.com/nkaz001/hftbacktest) — primary choice

Rust core + Python (3.11+, Numba) bindings; backtest AND live (Rust connectors:
Binance Futures, Bybit). Docs: hftbacktest.readthedocs.io.

**Data it needs.** Tick-level L2 (MBP) + trades with *local receive
timestamps*. Its collector records three Binance futures WS streams —
`<sym>@depth@0ms` (undocumented 0ms diff-depth; documented tiers are
100/250/500 ms), `<sym>@trade`, `<sym>@bookTicker` — each line = ns local
timestamp + raw JSON. Converted to a columnar npz event stream with 8 fields:
`ev` (u64 flags: depth/trade/snapshot), `exch_ts` (ns), `local_ts` (ns), `px`,
`qty`, `order_id`, `ival`, `fval`. Converter reorders rows to repair timestamp
inversions; `create_last_snapshot()` builds EOD snapshots to seed the next
day. A **tardis-machine converter** ingests Tardis.dev historical L2+trades
(processes trades before depth so trade-triggered depth changes fill
realistically) — the only way to get *historical* Binance L2 (paid);
otherwise forward-collect (already started in `data/mm_hf/raw/`, 2026-08-19).

**Queue / fill models.** Exchange sims: `NoPartialFillExchange` (default) and
`PartialFillExchange`. Queue position: `RiskAverseQueueModel` (conservative —
you advance only via traded volume at your price, cancels assumed behind you);
`ProbQueueModel` family — ambiguous L2 reductions allocated ahead-of-you with
`p_ahead(x) = f(x) / (f(x) + f(1−x))`, f satisfying f(0)=0, f(1)=1; provided
`PowerProbQueueFunc` (3 variants, f = x^γ) and `LogProbQueueFunc` (2 variants,
f = log(1+x)). L3 (MBO) via `L3QueueModel` trait — moot on Binance (no public
L3 feed).

**Latency models.** Three latencies modeled: feed, order entry, order response.
`ConstantLatency`; `FeedLatency` (synthesizes order latency from observed feed
latency when you have nothing better); `IntpOrderLatency` — interpolates
*measured* order RTTs, collected by submitting far-from-market unexecutable
orders on a timer (recommended; most accurate). Custom via `LatencyModel` trait.

**Fees / funding.** Fee models: `trading_value_fee_model(maker_fee, taker_fee)`
(negative = rebate, e.g. −0.00005 = 0.5 bps rebate), `trading_qty_fee_model`,
`flat_per_trade_fee_model` — set on the `BacktestAsset` builder with
`tick_size` / `lot_size`. **Funding is NOT simulated** — apply the 8h Binance
(1h Hyperliquid) funding to average held inventory in post-processing.

**Alternatives (brief).** NautilusTrader: production-grade, Binance adapter,
but L2 fill model less faithful (no queue-position bracketing) — better as live
harness than MM simulator. ABIDES / mbt_gym: agent-based, for RL research, not
venue-faithful. Custom replay on own collected data: acceptable for at-depth
strategies where queue position matters less.

### 4.2 Known MM backtest pitfalls → mitigations

| # | Pitfall | Mitigation |
|---|---------|-----------|
| 1 | **Queue-position optimism** (L2 can't tell if cancels were ahead/behind) | Bracket every result: RiskAverse (pessimistic) vs ProbQueue (γ sweep). If PnL sign flips across the bracket, there is no edge. |
| 2 | **Zero-latency repricing** (backtest cancels/requotes instantly; live you're 5–200 ms stale and get picked off in the gap) | Measured `IntpOrderLatency`; stress at 2–5× measured RTT; PnL-vs-latency curve must be flat-ish, not cliff. |
| 3 | **No self-impact / self-fill** (your quotes would have absorbed flow that in data hit the book; your fills can't exceed printed volume) | Cap fill share per event at printed qty × your queue share; treat backtest capacity as an upper bound; paper-trade forward before believing it. |
| 4 | **Fills only where trades printed** (thin names: no prints ≠ no adverse moves) | Also mark to book crossings (order in cross ⇒ filled), which hftbacktest does; sanity-check fill counts vs real trade counts. |
| 5 | **L2-vs-L3 ambiguity / iceberg & hidden flow** | Accept ±; calibrate later against live fills (closed-action replay, §4.3). |
| 6 | **Feed timestamp fantasies** (using exchange ts = pretending zero feed latency) | Always trade on `local_ts`; keep collector clock NTP-disciplined; hftbacktest's reorder-repair helps but garbage-in applies. |
| 7 | **Funding ignored** | Post-process funding on inventory held across funding times; on HL (1h funding) this is first-order. |
| 8 | **Maker-only PnL without inventory close-out cost** | Mark residual inventory to mid MINUS half-spread + taker fee; report per-day flat-close PnL. |
| 9 | **Regime overfit** (2023 spread/vol regime ≠ 2026; symbol survivorship) | Rolling-quarter evaluation; PIT symbol universe (repo discipline already exists). |

### 4.3 Calibration methodology (hangukquant)

*HFT MM: Detailed Guide to Queue Modelling and Calibration*
(research.hangukquant.com/p/hft-mm-detailed-guide-to-queue-modelling): don't
calibrate queue-model parameters on backtest PnL. Run **closed-action replay** —
replay the *identical* live order actions through the sim on the same data and
score agreement with live outcomes: fill precision/recall (F1), terminal order
status accuracy, first-fill timing error, VWAP-fill similarity, cancel-race
handling, combined `S(θ) = Σ w_i · metric_i`. Requires live fills → this is the
step AFTER a small live pilot. Also: *Market making. Code* and *Tick Data
Modelling* on the same substack; Moallemi & Yuan, *A Model for Queue Position
Valuation* (moallemi.com/ciamac/papers/queue-value-2016.pdf) for why queue
position is worth real bps.

---

## 5. Latency reality

- **Matching engine:** AWS ap-northeast-1 (Tokyo) — Binance, plus Bitget,
  KuCoin, HTX, MEXC, **and Hyperliquid** concentrate there (Zenlayer blog;
  hftbacktest MM-program page lists Binance low-latency access via AWS Tokyo).
- **Feed latency in-region:** ~4 ms mean, p99 < 13 ms for Binance market data
  received in Tokyo AWS (Ember/Deltix measurements). Same-metro bare metal
  (Zenlayer): ~2 ms to ap-northeast-1.
- **Order RTT:** Tokyo VPS ≈ 5–30 ms in practice (gateway overhead on top of
  ~2–5 ms network); Singapore ≈ 100 ms; US-East ≈ 150–180 ms; EU ≈ 230–270 ms.
  Measure yourself with hftbacktest's unexecutable-order method (§4.1).
- **Hard practical finding (first-hand, 2026-08-19):** `fapi.binance.com`
  **geo-blocks this repo's US AWS box** ("restricted location", error code 0);
  today's `data/mm_hf/exchange_info_20260819.json` is empty because of it.
  Binance Vision archives and `data-api.binance.vision` still work. **A Tokyo
  (non-US) VPS is required for BOTH latency and plain API access.**
- **WS cadence:** futures diff-depth documented at 250 ms (default) / 500 ms /
  **100 ms**; `@depth@0ms` exists undocumented (hftbacktest collector uses it).
  `bookTicker` and `aggTrade` push per-event (~real-time). So the fastest
  *documented* full-book view is 100 ms; best-quote view is real-time.
- **Rate limits (USDM, defaults):** REQUEST_WEIGHT 2,400/min;
  **orders 300 / 10 s and 1,200 / min** (adjusted 10s-limit = ⅓ of the minute
  limit). Uplifts via VIP tier / fill-ratio program; LP-program members get
  higher limits + private low-latency gateways. 300/10s caps a naive
  many-symbol quoter: quoting 2-sided × 10 symbols × 1 requote/s ≈ 20 orders/s
  = 200/10s before fills — rate limits, not CPU, bound quoting breadth.
- **What latency tier survives where:** at-the-touch on BTC/ETH = queue-priority
  war on a 0.03 bps spread against colocated program members — no solo tier
  suffices (REJECT). Mid-caps at-touch: Tokyo VPS (5–30 ms) is the entry
  ticket; survivable only if quotes tolerate ~100 ms staleness (wider spread,
  smaller size). At-depth / event-driven quoting: tolerant to 100 ms+, the only
  latency class where a solo setup is not structurally last in line.

---

## 6. Competitive landscape and the honest niche

**Who you're quoting against (public info):** Wintermute (~$15B/day across 65
venues; reportedly the majority of Binance top-of-book liquidity), Jump Crypto,
GSR, Cumberland/DRW, B2C2, Amber, Flow Traders, Keyrock, QCP, plus Auros,
Kronos, Pulsar Tower-style prop firms. All sit in the exchange LP programs:
**negative-to-zero effective maker fee, private gateways, raised rate limits** —
a permanent structural advantage a solo VIP0-3 account cannot match on liquid
names. Multicoin's *Adverse Selection Rules Everything Around Me* (2026-02) is
a good strategic read: benign flow concentrates where retail actually trades;
whoever holds the fee/latency edge taxes everyone else.

**Where a solo, non-colocated operator has any chance:**
1. **Execution alpha for your own book (highest value, zero fee wall).** The
   repo's capstone conclusion (2026-08-03) is that the surviving XS edge dies
   at retail taker cost (~24 bps RT hedged) and lives at ≤ 8 bps. Passive
   entry/exit on the existing daily portfolio *is* market making with a
   guaranteed uninformed client — yourself. The whole MM stack (markouts,
   queue sim, Tokyo VPS) pays here first, converting 12–24 bps RT into
   ~2–4 bps + AS.
2. **Tail names / new listings at wide spreads** (≥ 4–10 bps), ideally under a
   new-listing LP promo (0.5 bps rebate). Costs: toxic inventory, manipulation
   ("defensive MM against manipulators" — Crypto Chassis, medium.com
   open-crypto-market-data-initiative), delist risk. Marginal at VIP0-1, real
   only with the promo rebates.
3. **At-depth / event-driven liquidity provision**: quote 2–10 bps behind the
   touch, monetize liquidation cascades and vol bursts; latency-tolerant,
   fee-tolerant (large capture per fill), low fill rate. Closest to the
   existing research skillset (conditioning models matter more than speed).
4. **Hyperliquid mid/tails**: wider spreads than Binance equivalents, same
   Tokyo latency for everyone, no colocation program moat, base maker 1.5 bps
   (1.2 with modest staking). Needs forward spread/markout data on HL before
   any commitment.

---

## 7. ADOPT / REJECT / DEFER

| Item | Verdict | Reason / condition |
|------|---------|--------------------|
| aggTrades markout + effective/realized spread pipeline (§3) | **ADOPT now** | Free data already local; produces the AS(τ) numbers every other decision needs; day-clustered + block-bootstrap stats. |
| hftbacktest as simulator (Rust/Py) | **ADOPT** | Purpose-built; queue+latency bracketing; feeds on the L2@100ms/bookTicker collection already started in `data/mm_hf/`. |
| Queue-model bracketing (RiskAverse vs ProbQueue sweep) as a reporting standard | **ADOPT** | Sign-stability across the bracket = the go/no-go gate. |
| Tokyo VPS | **ADOPT (prerequisite)** | 5–30 ms RTT vs 150 ms+; also the only fix for the observed US geo-block of fapi. $50–200/mo. |
| Forward L2 + bookTicker + aggTrade collection | **ADOPT (running)** | Binance stopped publishing bookTicker archives 2024-03-30; history only accrues from now. |
| Passive-execution layer for the existing XS strategy | **ADOPT (first deployment target)** | Attacks the repo's stated binding constraint (execution cost), no fee-tier requirement. |
| At-the-touch MM on BTC/ETH (any venue, any solo tier) | **REJECT** | 0.03–0.04 bps spread vs ≥ 0 fee + queue war with program members. |
| Any Binance making at VIP0 on liquid names | **REJECT** | Maker fee 1.8 bps ≈ 4–20× half-spread; arithmetic cannot close. |
| Trusting un-bracketed L2 backtest fills | **REJECT** | Pitfalls §4.2 #1–3 each individually flip marginal PnL. |
| Binance LP / LiquidityBoost programs | **DEFER** | Gated at ≥ $20–100M 30d volume; revisit iff a tail-name strategy scales. |
| Hyperliquid standalone MM | **DEFER** | Best solo venue on paper (no colocation moat, wider spreads) but base maker 1.5 bps; needs measured HL spreads/markouts first. |
| dYdX / OKX for the maker rebate | **DEFER** | Real negative maker fees but thin books (dYdX) / unreachable tier (OKX VIP8). |

**Venue recommendation:** research and simulate on **Binance USDM** (only venue
with free tick history + the existing repo stack); first live deployment =
**passive execution of the existing XS book on Binance from a Tokyo VPS**;
first standalone-MM candidate = **Hyperliquid mid/tail names**, decided by
forward-collected HL spread/markout data. **Minimum viable fee tier for
standalone Binance MM: VIP3 + BNB (1.08 bps maker) restricted to names with
full spread ≥ ~4 bps — below that tier/spread combination, do not quote.**

---

## References

- Binance USDM fee page: https://www.binance.com/en/fee/futureFee (JS; anchors via
  https://www.datawallet.com/crypto/binance-vip-levels-explained , https://www.bitdegree.org/crypto/tutorials/binance-fees , https://dappgrid.com/binance-futures-fees-explained/ )
- Binance LP promo (new listings, 0.5 bps): https://www.binance.com/en-TR/support/announcement/usd%E2%93%A2-margined-futures-liquidity-provider-promotion-get-0-005-maker-fee-rebates-for-trades-on-new-eligible-listings-fb1b230dde3745cd8be5ababfce89757
- Binance CM LP program (qualification thresholds): https://www.binance.com/en/support/announcement/binance-updates-coin-margined-futures-liquidity-provider-program-2023-09-26-200b596c55934d099e6cced791fb6b5c
- Binance futures rate limits: https://www.binance.com/en/support/faq/rate-limits-on-binance-futures-281596e222414cdd9051664ea621cdc3
- Hyperliquid fees (official): https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees
- hftbacktest: https://github.com/nkaz001/hftbacktest ; docs https://hftbacktest.readthedocs.io/en/latest/ (order_fill.html, latency_models.html, tutorials/Data Preparation.html, market_maker_program.html)
- hangukquant queue modelling: https://www.research.hangukquant.com/p/hft-mm-detailed-guide-to-queue-modelling ; market-making code: https://www.research.hangukquant.com/p/market-making-code
- Moallemi & Yuan, queue-position value: https://moallemi.com/ciamac/papers/queue-value-2016.pdf
- Albers et al., Market Maker's Dilemma (Binance BTC perp live experiment): https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5074873
- Fast fills are bad fills (adverse selection of fills): https://aligrithm.com/adverse-selection-is-adverse-selection-porting-fast-fills-are-bad-fills-to-fx-and-futures/
- Multicoin, Adverse Selection Rules Everything Around Me: https://multicoin.capital/2026/02/17/adverse-selection-rules-everything-around-me/
- Crypto Chassis, Defensive MM vs manipulators: https://medium.com/open-crypto-market-data-initiative/defensive-market-making-against-market-manipulators-3ceabb5d1b71
- MDPI IJFS 14(5):103 (perp microstructure/integration): https://www.mdpi.com/2227-7072/14/5/103
- BTC futures HF microstructure: https://www.sciencedirect.com/science/article/pii/S2214845025001188
- Latency/geo: https://cloud.zenlayer.com/blog/crypto-trading-latency-tokyo ; https://ember.deltixlab.com/docs/performance/ws-market-data/ ; https://aws.amazon.com/blogs/industries/ultra-low-latency-cross-region-crypto-trading-with-avelacom-and-aws/
- Spread stats in §2.1: computed 2026-08-19 from Binance Vision
  `data/futures/um/daily/bookTicker/{BTCUSDT,ETHUSDT,INJUSDT,CHZUSDT}-2024-03-15.zip`
  (time-weighted on transaction_time; archives exist 2023-05-16→2024-03-30 only).
