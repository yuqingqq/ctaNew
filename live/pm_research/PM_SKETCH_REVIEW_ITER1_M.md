# PM_SKETCH_REVIEW_ITER1_M — lens M (mechanism / competition realism)

Date: 2026-08-19/20. Object: `PM_MM_PLAN.md` §1 (economics), §3 (MM model),
§6 (hazards). Method: Polymarket primary docs + **on-chain decode of our own
recorded trades** (Polygon public RPC) + **live CLOB rewards registry** +
census over our collected `data/pm_5min/raw/20260819/` (73,559 trade events,
121 windows, 7 coins).

Every number below is marked **[M]** measured by me, **[D]** from a primary
Polymarket doc, or **[I]** inference/estimate.

---

## M1. Fees — GROUND TRUTH, conflict RESOLVED: fees are REAL and CHARGED

The three-way conflict in `PM_REVIEW_ITER1` (docs 0.07 vs `taker_base_fee=1000`
vs `fee_rate_bps=0` on all observed trades) is resolved **in favour of the
docs**. The WS field is simply not populated.

**Decisive test [M]** — decoded the Polygon receipt for a trade we recorded on
our own window (`btc-updown-5m-1787151000`, p=0.50, 94.38 shares, WS payload
said `fee_rate_bps:"0"`):

| leg | tx | on-chain fee (6-dec USDC) | predicted `C·0.07·p(1−p)` |
|---|---|---|---|
| taker, p=0.50, C=94.38 | `0xc42c0919…ce10d` | **1,651,650 = $1.65165** | 94.38·0.07·0.25 = **$1.651650** ✓ exact |
| taker, p=0.99, C=2.08 | `0x33a2a0bc…d283` | **1,440 = $0.001440** | 2.08·0.07·0.0099 = **$0.001441** ✓ exact |
| maker legs (same txs) | — | **0** | 0 ✓ |

Exchange `0xe111180000d2663c0091e4f400237545b87b996b`; `OrderFilled` topic0
`0xd543adfd9457…d8ee` (word[4] = fee); a separate `FeeCharged`-style event
`0x55bb3cade9d43b79…` carries the same amount. **Recipe for E1: read fees from
the on-chain OrderFilled/FeeCharged logs, never from `fee_rate_bps`.**

Confirmed **[D]** (docs.polymarket.com/trading/fees, help.polymarket.com
trading-fees + maker-rebates): `fee = C × feeRate × p × (1−p)`; crypto
`feeRate = 0.07`; **makers never charged**; makers receive **20%** of collected
taker fees in crypto, pro-rata by `C·feeRate·p(1−p)` on their filled maker
orders, paid daily in USDC, $1 minimum.

`maker_base_fee = taker_base_fee = 1000` is a **legacy signature cap** (10% in
bps), not the charged rate — maker legs verifiably pay 0 **[M]**.

### What this does to §1's table

Per share, at p = 0.50, with the observed 1-tick book:

| leg | $/share | % of $0.50 notional | in ticks |
|---|---|---|---|
| taker fee (their cost / our chase cost) | 0.0175 | **3.50%** | 1.75 |
| maker fee | 0 | 0 | 0 |
| **maker rebate (20% of the taker fee on our fill)** | **0.0035** | **0.70%** | 0.35 |
| half-spread at 1-tick market | 0.0050 | 1.00% | 0.50 |
| **gross maker capture per filled share** | **0.0085** | **1.70%** | 0.85 |

Fee decays away from the money: at p=0.90 the taker pays $0.0063/share (0.70%
of notional), at p=0.99 $0.0007 **[M/D]**. So the rebate is worth ~0.7% of
notional ATM but ~0.14% at p=0.90 — **the maker subsidy is concentrated exactly
where the toxicity is worst** (M3).

§1's table is directionally right (maker fee 0, huge relative spread) but three
cells are wrong or missing and must be corrected: taker fee is **3.5% of
notional ATM, not "~$0.0175/share" presented as small**; the rebate is
quantifiable at **70 bps of notional per fill**, not "~20% share"; and there is
no row for the fee's role as **our own protection** (below).

### The fee is NOT the moat it looks like

Naively, a sniper must clear half-spread + taker fee = 2.25c of probability
before lifting a resting ATM quote. But **[I]**, with BTC annualised vol ≈50%
⇒ σ over a 300 s window ≈ 15.4 bps, ATM sensitivity is
`dp/d(lnF) = φ(0)/σ_eff`:

| time in window | σ_eff | 1 bps of BTC = |
|---|---|---|
| t=0 (τ=300 s) | 15.4 bps | **2.6c** of probability |
| τ=150 s | 10.9 bps | **3.7c** |
| τ→60 s (TWAP entry) | ≪ | larger still |

So the 2.25c fee+spread moat is worth **under 1 bps of underlying movement** —
which BTC traverses many times per second. **At the money the fee buys almost
no protection.** Away from the money φ(d) collapses and the same 2.25c becomes
a genuine moat. This is the single most load-bearing quantitative fact for §3.

---

## M2. Rewards — parameters are WRONG in the plan, by a lot

Pulled the authoritative CLOB rewards registry
(`GET /rewards/markets/current`, 25 pages, 12,500 rewarded markets) and matched
against our 850 collected `conditionId`s **[M]**:

| slug | rate_per_day | rewards_max_spread | rewards_min_size |
|---|---|---|---|
| `btc-updown-5m-*` | **$10,000** | **1.5** | 50 |
| `eth-updown-5m-*`, `hype-updown-5m-*` | $1,666.67 | 1.5 | 50 |
| `bnb-updown-5m-*`, `doge-updown-5m-*` | $833.33 | 1.5 | 50 |

Two corrections, both material:

**(a) The qualifying band is 1.5c, not 4.5c.** Gamma metadata (what our
collector stores, `rewardsMaxSpread: 4.5`) **disagrees with the CLOB rewards
registry (1.5)** on the same live markets, checked simultaneously **[M]**. The
registry rows carry `start_date: 2026-08-20` — i.e. the band appears to have
been re-cut from 4.5c → 1.5c *during our collection*. The registry is what the
scorer uses. Plan §2.4 and §3 ("rest within `rewardsMaxSpread` (4.5c)") are
built on the wrong number and on the wrong source.

**(b) $550k is ONE MONTH, not one year.** Summing the registry:
`10,000 + 4×1,666.67 + 2×833.33 = $18,333/day` ⇒ **$550,000 per 30 days**,
reconciling exactly with the docs **[D]**: "$1M in liquidity rewards … through
the month of August", 5-min slice $550k (BTC $300k, SOL/ETH/HYPE/XRP $200k
split, BNB/DOGE $50k split). The plan's "$550k/yr" understates the live pool by
**12×** — and simultaneously understates the fragility: the announced program
covers **August only** (11 days left at time of writing), while the registry
rows say `end_date: 2500-12-31`. H-PM4 is not a background worry, it is the
dominant term.

Per-window pool **[I]**: a 5-min market accruing at $10,000/day for 300 s earns
**≈ $34.7 per BTC window** (288 windows/day × $34.7 = $10,000/day ✓ internally
consistent). *Caveat:* the registry lists ~2 BTC windows concurrently (live +
next), so per-window pool could be up to 2× off; E1 must pin this.

**Scoring [D]** (docs.polymarket.com/market-makers/liquidity-rewards):
`S(v,s) = ((max_spread − spread)/max_spread)² × size`; `Qmin` takes the
two-sided minimum, with single-sided scoring at `Q/c`, `c = 3`, and
**double-sided mandatory when mid ∉ [0.10, 0.90]** — note that 5-min windows
spend much of their life outside that band, so two-sided quoting is
*compulsory* for most of the payoff-relevant time. Sampling is **per-minute,
randomised**, 10,080 samples/epoch, paid daily 00:00 UTC, pro-rata on
normalised score. A 5-min market therefore contributes only **~5 samples** —
per-window reward is a high-variance lottery, not a smooth accrual **[I]**.

### The reframe is CORRECT, and it breaks gate G3a as written

Take the standard band-riding game. Maker *i* posts qualifying score `x_i`;
reward `R·x_i/X`, `X = Σx`; adverse-selection cost `c` per unit qualifying
score. Symmetric Nash:

```
max_x  R·x/(X)  − c·x     ⇒   R·(X − x)/X² = c
symmetric (n makers):          x* = R(n−1)/(c n²),   X* = R(n−1)/(c n)
n → ∞:                         X* → R/c ,   per-maker profit → R/n → 0
```

**Aggregate qualifying depth is bid up until the entire reward pool is
dissipated into adverse selection.** The observed 2–4c spread is then not
(only) Glosten–Milgrom compensation — it is the **rewards-optimal resting
configuration**, and the thin depth inside 1.5c is the market's revealed
estimate of `c`. Our own book data is consistent: BTC median depth within 1c of
mid is only **138 shares/side (~$69)**, versus 1,290 shares within 4.5c **[M]**
— depth deliberately sits *outside* the (new) scoring band.

Consequences for the plan:

1. §3's "rewards booked as a SEPARATE PnL line, never netted into the markout
   gate" is **necessary but not sufficient**. Rewards and adverse selection are
   *structurally coupled through the band constraint*: to earn, you must stand
   within 1.5c of mid, which is precisely the maximally-toxic location (M1).
   Reporting them as independent lines implies a separability that does not hold.
2. **G3a as written is not a valid counterfactual.** "Net PnL > 0 under
   pessimistic queue WITHOUT rewards" evaluated on the *same* quote policy asks
   what a rewards-optimised policy earns with the subsidy deleted — guaranteed
   negative, and uninformative. A maker who genuinely ignored rewards would
   quote *wider than 1.5c* with a completely different fill distribution. G3a
   must **re-optimise the policy under the no-rewards objective** and compare
   like with like. This is my top MUST-FIX.
3. The genuinely tradeable structure the reframe exposes: `dp/dF ∝ φ(d)`
   collapses away from the money, but the reward weight
   `((1.5 − s)/1.5)²` does not depend on `|d|` at all. **The band is cheap to
   occupy when the window is decided and ruinous when it is ATM.** Every window
   opens ATM (K = X_0) and diffuses out. So the real object is a
   `|d|`- and `τ`-conditional band-occupancy policy — which the plan's
   time-stratified machinery can measure, but §3 currently states no such
   conditioning rule. Add it.

---

## M3. Matching mechanics

**[D] + secondary, consistent:** operator-run **off-chain matching, on-chain
settlement** on Polygon; EIP-712 signed orders; **price-time priority**;
**unified book** (an Up bid at 0.60 crosses a Down bid at 0.40 via CTF mint;
paired sells merge). Our on-chain decode confirms the mint/merge behaviour:
`OrderFilled` legs pair one fee-bearing taker leg against multiple zero-fee
maker legs **[M]** (~18.7k taker legs vs ~50.7k total legs over ~21 min).

Implications, mostly *favourable* to the plan:

- §2.3 pair-harvest and the §5 E3 queue bracket (join-back vs pro-rata) are
  **sound in principle** — price-time priority is the right prior.
- **Queue position remains unobservable.** There is no MBO feed on the public
  market channel; the authenticated *user* channel gives own-fills only. The
  bracket is unavoidable, as ITER1 already concluded. Keep it.
- **Rate limits are not binding [D]:** `POST /order` 3,500/10s burst,
  36,000/10min sustained; `DELETE /order` 3,000/10s. A 5-min-window requoter
  needs orders of magnitude less.
- **Cadence mismatch worth flagging:** the plan reasons about "requote speed vs
  the 1 s settlement-stream cadence". The binding constraint is not the 1 s
  RTDS tick, it is **round-trip to the matching engine** (M4) — the stream
  tells you the truth 1×/s but your quote can only move as fast as
  London-RTT allows.

---

## M4. Operational reality (leaks into E3's assumptions)

- **PM matching infra is AWS `eu-west-2` (London)** (secondary source,
  community-benchmarked; flagged **[I]**, not an official statement).
- **Our collector box is `3.89.236.252` = AWS `us-east-1` (N. Virginia) [M].**
  us-east-1 → London ≈ **70–80 ms RTT**.
- Binance futures matching is Tokyo (`ap-northeast-1`). So a quoting stack must
  eat a **Tokyo→London** signal hop (~230 ms RTT) or a London→Tokyo data hop.
  **Realistic Binance-event → PM-order-acknowledged latency: ~120–250 ms [I]**,
  and no colocation choice removes both legs. Against `1 bps of BTC = 2.6–3.7c
  of probability` (M1), **150 ms of BTC drift is a large multiple of the tick**.
  E3 must model requote latency as ~150–250 ms, not as instantaneous, and
  δ_tox must be sized off *that* number.
- **Measured CLOB WS delivery latency (recv − payload ts), 481,482 messages
  [M]: median 48 ms, p90 1,284 ms, p99 8,971 ms.** The tail is the alarming
  part: at p99 the market-data feed is ~9 s stale. *Ambiguity:* this may be our
  single-process collector's gzip/backlog rather than PM's egress — it must be
  disambiguated (timestamp a second, idle subscriber) before E3 trusts either
  number. If the tail is real, stale-quote exposure dominates every other term
  in the model.
- RTDS (settlement stream) recv−payload ≈ **471 ms mean** (ITER1) — the truth
  arrives *later* than the Binance signal, so p̂ must be driven by Binance with
  RTDS used for basis calibration, exactly as §2 says. Good.
- **Geo/deployment:** UK/FR/DE/BE/PL are restricted; the API is reportedly open
  to US developers post-Polymarket-US **[I, secondary]**. This stays out of
  research scope per the plan — but note the irony that the lowest-latency
  colocation region (London) is a **restricted jurisdiction**, which is a real
  constraint on any eventual deployment and should be recorded now, not
  discovered at PM-E4.
- Capital rails: USDC on Polygon via a proxy wallet; per-market ERC-1155/20
  allowances; gas on Polygon is negligible (<$0.01/tx) **[I]**.

---

## M5. CTF pair-harvest — cheaper than the plan assumes

- `neg_risk = false` on these markets **[M, CLOB metadata]**. So these are plain
  binary CTF conditions; the negRisk adapter and its conversion mechanics do
  **not** apply. Good — one less complication.
- **You almost never need to call `mergePositions` yourself.** Two facts:
  (i) the unified book already mints/merges *atomically at match time* via the
  operator, so acquiring Up+Down through two maker fills costs no extra
  transaction; (ii) holding 1 Up + 1 Down **to resolution** redeems for exactly
  $1 — identical to merging. Intra-window merge is therefore purely a
  **capital-velocity** optimisation, not a requirement for pair-harvest.
- Cost of the optional cycle **[I]**: `mergePositions` on Polygon is a single
  ~150k-gas call, well under $0.01; `redeemPositions` likewise, callable from
  ~**+85 s after window end** (ITER1 measured resolution posting at +85 s).
  A full capital cycle is therefore ~7 min, and gas is not a material line item.
- So §2.3's pair-harvest is **mechanically sound and cheap**. The binding
  constraint is *leg risk* (two independent maker fills at different times),
  not merge cost — the plan should say so, because §2.3 currently implies the
  merge itself is the operation of interest.

---

## M6. Counterparty census (our own data)

From `data/pm_5min/raw/20260819/`, 73,559 trade events, 121 windows, 7 coins,
2.08 h **[M]**:

| statistic | value |
|---|---|
| total notional | $816,834 (2.08 h) ⇒ ~$5–9M/day **[I]**, peak-hours sample |
| **BTC share of notional** | **$672k / $817k = 82%** (eth $94k, sol $24k, xrp $16k, doge $4.4k, hype $3.2k, bnb $2.9k) |
| per-trade notional | mean $11.10, **median $3.40**, p90 $21, p99 $129, max $2,190 |
| per-trade size | median **6.2 shares**, p90 44, max 25,000 |
| notional/window | median $827, p90 $30,570, max $53,822 |
| timing (notional) | pre-window $83k; 0–60 s $208k; 60–120 s $163k; 120–180 s $165k; 180–240 s $135k; **240–300 s $51k** |
| price of taker flow | 24.6% at 0.4–0.6; 13.7% at 0.90–0.99; 4.6% at 0.01–0.10 |

On-chain wallet census over ~21 min of `OrderFilled` (Polymarket-wide, **not**
5-min-only — caveat) **[M]**: 3,841 distinct fee-paying taker wallets;
**2,837 of them (74%) average under $20 per trade**; top-5 wallets pay 21.5% of
fees, top-20 pay 40.5%. Maker side: 3,485 wallets, top-1 holds 11.7% of maker
fill notional, top-5 hold 19.8%.

**Read [I]:** a genuine **barbell**. The overwhelming *count* is retail
app-sized flow ($3–20 tickets, median 6 shares) — this is real, uninformed
volume and it is what makes the venue interesting. But the *fee-weighted* tail
is professional: wallets doing $141–$16,000 average tickets, with two wallets
paying ~$1,085 and ~$480 of fees on 1–2 trades each. The plan's H-PM5
("assume adversaries hold the same Binance feed") is **correct and should not
be softened**. Notably, flow does **not** concentrate at the end of the window
(240–300 s is the *smallest* notional bucket) — this **contradicts §6 H-PM2's
premise** that "flow concentrates near resolution"; the largest bucket is the
first 60 s, when the market is ATM and p̂ is most sensitive. H-PM2 should be
re-stated as *ATM-toxicity* rather than *endgame-toxicity*. τ_min pull is still
right, but it is not where the volume is.

Zero-trade windows are common (another reviewer: 8–20% by coin), and the
median window ($827) versus p90 ($30.6k) shows a ~37× spread — **never average
across windows; notional-weight everything** (already plan policy, reinforced).

---

## M7. Capacity — honest order of magnitude

Two subsidy pools, both measurable:

1. **Liquidity rewards: $18,333/day** across all 5-min crypto **[M, registry]**
   — but August-only as announced **[D]**.
2. **Maker rebates: 20% of taker fees.** Our measured flow implies
   ~$16.2k of taker fees per 2.08 h peak sample ⇒ **$95–190k/day** of taker fees
   after discounting for off-peak **[I]** ⇒ rebate pool **$19–38k/day**.
   Note this is the *larger* of the two pools and the plan barely counts it.

Total subsidy to *all* makers on 5-min crypto: **~$37–56k/day**.

One competent maker capturing 5–15% (top on-chain maker holds ~12% of
Polymarket-wide maker fill notional **[M]**):

| line | at 5% share | at 15% share |
|---|---|---|
| rewards | $27k/mo | $82k/mo |
| rebates | $28k/mo | $170k/mo |
| **gross subsidy** | **~$55k/mo** | **~$250k/mo** |

**But gross is not the answer.** The M2 equilibrium says the pool is dissipated
into adverse selection for the *marginal* maker; only a genuine p̂ edge over the
field is kept. A sanity check on the rebate line: earning $2k/day of rebate at
$0.0035/share requires ~571k shares/day ≈ **$286k/day of filled notional**;
losing merely **0.7% of notional** to markout wipes it out — and 0.7% of
notional at p=0.5 is **0.35c of probability ≈ 0.1 bps of BTC movement** (M1).
You will lose far more than that on ATM fills. The subsidy only survives on
**away-from-the-money, decided-window** fills.

**Honest capacity estimate: gross subsidy ~$50–250k/month; realistic NET for
one competent maker ~$10–30k/month central, plausibly $0, with the upper tail
requiring a real p̂ edge — and conditional on the rewards program surviving
past August.** Working capital is small (~$500/market × ~10 concurrent
markets + buffer ⇒ **$25–100k**), so return-on-capital is high if it works.

**Verdict on "worth the research cost":** yes, marginally — the return on
capital is exceptional *if* the edge is real, and the incremental research cost
is low because the collectors already run. But the program should be run with
an explicit **time-box tied to the rewards program's survival**, and the
decision-relevant question is narrow: *is there a `(|d|, τ)` region where band
occupancy is net-positive after ~150–250 ms requote latency?* If §5 cannot
answer that within ~2 weeks, the answer is no.

---

## Triage

| # | severity | finding | required plan change |
|---|---|---|---|
| M1 | **MUST-FIX** | **G3a is not a valid counterfactual.** Rewards and adverse selection are coupled through the 1.5c band constraint; deleting the rewards line from a rewards-optimised policy measures nothing. | G3a must **re-optimise the quote policy under a no-rewards objective** and compare like-for-like. State the coupling explicitly in §3; drop the "separate PnL line ⇒ separable" implication. |
| M2 | **MUST-FIX** | **`rewardsMaxSpread` is 1.5c, not 4.5c** — CLOB rewards registry vs Gamma metadata disagree on the same live markets, and the registry (authoritative) shows the band re-cut on 2026-08-20. Collector stores only the Gamma value. | Correct §2.4/§3; **switch the collector to the CLOB rewards registry** (`GET /rewards/markets/current`, incl. `rate_per_day`) as the rewards source of truth; treat band params as a *time-varying series*, not constants. |
| M3 | **MUST-FIX** | **Rewards pool is $550k/MONTH ($18,333/day), not $550k/yr** — 12× error in §1, and the announced program covers **August only** (11 days left). | Fix §1; promote H-PM4 from a standing hazard to a **time-box on the whole program**, with an explicit "what remains if rewards end" fallback. |
| M4 | **MUST-FIX** | **§1's fee row is wrong and its sign is under-stated.** Fees are real (on-chain verified, exact match to `C·0.07·p(1−p)`); taker pays **3.5% of notional ATM**; maker rebate is **70 bps of notional per fill**. `fee_rate_bps` is unpopulated and must never be used. | Rewrite the §1 table with the per-share/%-of-notional/tick columns from M1; E1 reads fees from on-chain `OrderFilled`/`FeeCharged`. Close the ITER1 "fee conflict" as RESOLVED. |
| M5 | **MUST-FIX** | **Latency is unmodelled.** PM infra is London, our box us-east-1, Binance Tokyo ⇒ ~120–250 ms Binance-event→PM-order. With `1 bps BTC = 2.6–3.7c of probability`, this dominates δ_tox. Measured CLOB WS tail p99 = 8,971 ms is unexplained. | §3 must carry an explicit requote-latency parameter (~150–250 ms) feeding δ_tox; E3 replays must apply it. Disambiguate the WS tail (collector backlog vs PM egress) before trusting fill timing. |
| M6 | SHOULD-FIX | **H-PM2 is mis-stated.** Flow does *not* concentrate at the endgame — 240–300 s is the smallest notional bucket ($51k) and 0–60 s the largest ($208k). Toxicity is ATM-driven, not clock-driven. | Re-state H-PM2 as ATM/`|d|`-toxicity; keep τ_min pull but stop justifying it with a volume-concentration claim the data contradicts. |
| M7 | SHOULD-FIX | The exploitable structure is a **`(|d|, τ)`-conditional band-occupancy policy** (reward weight is `|d|`-independent, toxicity is `φ(d)`-driven). §3 has no such conditioning rule. | Add the conditional occupancy rule to §3 and make it the primary E3 ablation, ahead of cross-venue hedge. |
| M8 | SHOULD-FIX | **Maker rebates are the larger subsidy** (~$19–38k/day pool vs $18.3k/day rewards) but are treated as a footnote in §1 and absent from §3/G3b. | Give rebates their own modelled line; G3b should read "with rewards **and rebates**". Note rebates are fill-contingent (unlike rewards), so they belong in the markout accounting, not beside it. |
| M9 | SHOULD-FIX | Per-window reward pool ≈ $34.7 (BTC) rests on the "accrues only while live" reading; registry lists ~2 concurrent windows, so this could be 2× off. Also only **~5 per-minute samples** per window ⇒ reward is a high-variance lottery. | E1 must pin the per-window pool empirically; model reward as a sampled random variable, not a deterministic accrual. |
| M10 | SHOULD-FIX | Two-sided quoting is **compulsory** whenever mid ∉ [0.10, 0.90] (`Qmin`, c=3 otherwise) — a large share of 5-min window life. | §3's four-sided quoting must treat two-sidedness as a hard constraint in the decided-window regime, not an optimisation. |
| M11 | NOTED | Matching = off-chain operator, price-time priority, unified book, mint/merge at match time — **on-chain-confirmed**. E3 bracket approach validated; queue position unobservable (no MBO). Rate limits (3,500 orders/10s) non-binding. | — |
| M12 | NOTED | `neg_risk=false`; intra-window CTF merge is **optional** (holding Up+Down to resolution ≡ merge) and gas is <$0.01. Pair-harvest's real cost is **leg risk**, not merge cost. | §2.3 wording. |
| M13 | NOTED | Counterparty is a barbell: 74% of taker wallets average <$20/trade (genuine retail), but top-20 wallets pay 40.5% of fees. H-PM5 correct as written. | — |
| M14 | NOTED | BTC = 82% of notional in our sample; median window $827 vs p90 $30.6k (37×); 8–20% of windows have zero trades. | Notional-weighting is mandatory; consider scoping E3 to BTC + ETH. |
| M15 | NOTED | Lowest-latency colocation (London/eu-west-2) is a **restricted jurisdiction** — a structural deployment constraint worth recording now. | Note in §5 PM-E4. |
