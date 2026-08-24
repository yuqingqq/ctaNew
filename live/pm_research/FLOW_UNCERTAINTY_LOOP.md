# FLOW_UNCERTAINTY_LOOP — clear the open uncertainties in the flow model

> **⚠ For current state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).** This
> document is **provenance** — correct about its own moment, not a statement of
> current belief. Where it conflicts with `FLOW_MODEL_STATE.md`, that page wins.


Charter. Opened 2026-08-21. Owner: coordinator writes this file and the decision
rules; the research agent runs the measurements and reports.

**Purpose.** `plans/BE_FLOWANDFILLS_MODEL_PLAN.md` is specified but carries a
set of named unknowns. Several are cheap and block large parts of the model.
Clear them one at a time, in ranked order, before any flow measurement is run.

## Rules for this loop

1. **One uncertainty per iteration.** Do not batch. Each gets its own decision
   record, appended to §"Ledger" below.
2. **The decision rule is written here BEFORE the measurement runs.** If a rule
   turns out to be badly posed, say so, amend it in a numbered revision, and
   record why — never silently re-cut it after seeing the answer.
3. **A cleared uncertainty must land in the plan** with its verdict and scope,
   and in this ledger with the evidence.
4. **`UNRESOLVED` is a valid outcome** and must be recorded as such, with what
   would resolve it and what it costs. Do not convert an unresolved item into a
   plausible story.
5. **Scope every finding.** One coin and one day is a scope, not a fact.
6. Standing rules of the programme apply throughout: check data before use;
   report the excluded set beside the retained one; read book state from
   `price_change.best_bid/ask`; knowledge time is `recv_ns`; never pool across
   `collector_version` eras unpaired.

## Ranked uncertainties

Rank = (how much it unblocks) x (how cheap it is to clear). U1 and U2 are first
because each is prerequisite to a whole layer of the model and both are cheap.

### U1 — `size` semantics. BLOCKS THE ENTIRE VOLUME LAYER.

`size` has p50 **0.02** against a stated `orderMinSize` of **5**. Every cash
number, `E[N]`, depth consumption and capacity estimate depends on reading it
correctly. The count layer of the flow model is available today; the volume
layer is blocked entirely on this.

> **THIS FRAMING INVERTED — see the R1 ruling below.** U1a cleared the unit, so
> the **volume layer is UNBLOCKED**; and 16.3 % of events carrying 0.0145 % of
> notional makes the **count layer the contaminated one**. Per coin the micro
> share runs **2.0 % (btc) to 90.0 % (hype)**, so above ~35 % the raw count is a
> participant measurement, not market flow. Read this section as the question as
> originally posed, not as the current state.

Known already: on 600/600 G-FF1 transactions the WS `size` equalled
`taker_amount_filled / 1e6` exactly, so the **unit** is shares at 6 decimals.
The unexplained part is the **magnitude** versus the stated minimum.

Leading hypothesis to test first, because it would dissolve the puzzle rather
than explain it away: `orderMinSize` constrains an **order**, while a
`last_trade_price` event reports one **fill**, so sub-minimum fills are just
partial fills of a conforming order. Test by aggregating fills per
`taker_order_hash` (available on-chain in `OrdersMatched`) and asking whether
order-level totals respect the minimum.

- **CLEARED if** order-level aggregates respect `orderMinSize` at >= 99%, or a
  different documented rule reproduces the observed distribution exactly.
- **UNRESOLVED if** order-level totals still fall below the minimum. Then the
  volume layer stays blocked and the plan says so.

### U2 — tick composition. PREREQUISITE TO `γ_tick`.

`exp_gff1_side.py` matched the byte pattern `"tick_size"`, which never matches
`"new_tick_size"`, so every `tick_size_change` event was ignored and the run
reported `{0.01: 600}`. That was the defect, not a market fact. `0.01 -> 0.001`
transitions are observed.

A second question rides on this one. Measured cash half-spread is **flat at
0.50 c across all moneyness** (2.29 M quotes, BTC, 2026-08-20). If the 0.001
tick is genuinely available where the spread still sits at 0.0100, then makers
are **declining to step inside when they could**, which makes the 1-cent spread
a **convention rather than a constraint** — a materially different thing to
model.

- **CLEARED when** tick composition is read from both `tick_size` and
  `new_tick_size`, reported per moneyness bucket, and the convention-versus-
  constraint question is answered with the spread-in-ticks distribution.
- Do **not** re-run the G-FF1 gate for this. The `PASS` stands on size agreement
  at 1e-6 and on direction from the amount pair.

### U3 — is gap occurrence independent of `r`? GATES `f_r`.

The plan requires this before any `f_r` is reported. If gaps concentrate near
expiry, the distortion lands exactly where the model is most load-bearing.

- **CLEARED if** gap start times are uniform in window phase under a test stated
  before running, with the excluded set reported.
- **FAILS if** they concentrate. Then `f_r` is reportable only with an explicit
  distortion bound, or not at all.
- Ledger covers only ~1/3 of windows and starts 2026-08-20 14:50:21. Before
  that, **absence of a gap record is not evidence of a clean window** — say so
  in the result rather than pooling.

### U4 — provenance of the maker markout `+0.45 c/share`. T2 RESTS ON IT.

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** `+0.45 ¢/share` and `+95 bps` are the same book-derived number (p90 6.2 s stale books) and fall together. Book-free rebuild: **+0.17 ¢/share**, and **NOT DISTINGUISHABLE FROM ZERO** — **+0.173 [-0.251, +0.596]**, all seven per-coin CIs spanning zero. The maker-edge sign is **UNDETERMINED at two days**. `side` IS the taker's (G-FF1 `PASS`), closing the `+95 → −95` flip. See `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b.


T2 predicts taker markout ~ **-2.20 c/share** from `+0.45` plus the 1.75 c fee.
But `+0.45` against a 0.50 c half-spread implies adverse selection of only
~0.05 c/share, which is implausibly small against a ~225 bps barrier. Every
book-derived number in `PM_DEEP_REVIEW.md` is already flagged
`stale_book_contamination: true`.

- **CLEARED if** the figure is traced to a source and either confirmed on
  `price_change` quotes or replaced.
- **WITHDRAWN if** it is book-derived and cannot be rebuilt. T2 then predicts a
  relationship rather than a number, and the plan must say so.

### U5 — the 10 of 754 fee-bearing maker legs.

A real 1.3% residual class under the discriminating test
(`taker == exchange address`), not a classification artefact. Mechanism
unexplained. Concentrated in a few transactions.

- **CLEARED if** the mechanism is identified from the receipts.
- **UNRESOLVED otherwise** — record it, do not narrate it.

### U6 — does the on-chain leg order carry matching-engine priority?

If `OrderFilled` legs are emitted in priority order, the tape contains an
ex-post fill sequence — which is what the queue bracket was a substitute for.
**Currently a hypothesis and possibly false.**

Honest ceiling even if true: the chain shows only orders that **filled**, so it
bounds the marginal correction; it does not identify `Q_ahead`.

- **CLEARED if** leg order correlates with an independent priority proxy
  (e.g. price then arrival) well above chance, stated before running.
- **REFUTED if** leg order is arbitrary. Record it and close the lead.

### U7 — maker rebate `ρ(p)`.

Blocks every net-of-fee estimand needing `ρ`. A zero maker fee is **not**
evidence a rebate was paid, and `OrderFilled.fee` is unsigned so a rebate could
not appear in it at all. Needs a separate on-chain transfer or a rewards-program
read; `rewards_registry.jsonl` exists and is unexamined.

- **CLEARED if** a rebate is observed and its schedule read, or its absence is
  established over the sampled transactions.
- **UNRESOLVED otherwise**; `ρ`-dependent estimands stay `Unavailable`.

### U8 — spread on thin coins.

"ATM runs 6-8 c" is refuted for BTC (median 1 tick, p90 0.020, 2.29 M
observations) but that is one coin on one day. `hype` is the thinnest and most
likely to differ.

- **CLEARED when** the spread-by-moneyness table is produced for all seven
  coins with per-coin scope stated.

## ROUND 2 — the uncertainties that remain, opened 2026-08-21 05:10

U9 is not listed here: it clears itself. `PING_TIMEOUT` stands at 7 of the
required 12 in `clob_v3_1`, accruing at ~0.63/h, so it is reachable ~8 h after
this was written. **Re-run `u9` unchanged then — do not amend the rule to fit
the n available.**

### U10 — are the per-coin maker edges real, or noise around zero?

**This is the most urgent item in the loop, because the plan currently carries a
positive claim that the data may not support.** The per-coin table shows the edge
sign flipping at every spread width (btc +0.201 / eth −0.008; doge +0.629 / sol
−0.628; xrp +0.705 / bnb −0.476). That refutes width-as-predictor either way.
But the *story* attached to it — that a wide spread prices adverse-selection
hazard, so compensation and hazard scale together — **is one explanation for a
pattern that is equally consistent with every coin having ~zero edge and the
dispersion being sampling noise.** Scattered signs is exactly what noise around
zero looks like.

Two days gives no day-clustered interval. But there are ~3,000 windows, and
windows are largely independent within a day, so **window-clustered inference is
available** and is the right instrument.

- Per-coin share-weighted maker edge with a **window-clustered bootstrap**
  (clusters = windows, >= 10,000 resamples), reported with the point estimate.
- Plus a **permutation test on coin labels**: is the observed cross-coin
  dispersion larger than a common-mean null produces?

Outcomes, enumerated exhaustively per the META-RULE:

- **REAL-DISPERSION** — permutation test rejects the common-mean null **and** at
  least two coins have intervals excluding zero **with opposite signs**. Only
  then is "coin-specific edge" a finding.
- **NOISE-CONSISTENT** — dispersion within the null, or no coin interval
  excludes zero. The per-coin table is then **not evidence of coin-specific
  edge**; report the pooled point with its interval and say so plainly.
- **PARTIAL** — some coin intervals exclude zero but dispersion is within the
  null, or the reverse. Name exactly which coins survive; authorise nothing
  beyond those.
- **UNRESOLVED** — any coin with too few windows to bootstrap; say which.

**MANDATORY caveat on every number produced here:** window clustering cannot
capture day-level common factors, and with two days the intervals **understate**
true uncertainty. A window-clustered interval that excludes zero is necessary,
not sufficient, for a day-robust claim.

### U11 — the rebate, off-trade. `rho` BLOCKS EVERY NET-OF-FEE ESTIMAND.

> **RESOLVED FROM DOCUMENTATION, NOT FROM CHAIN (2026-08-24, DA, R-112, filed
> Q-DA-50).** A **Maker Rebates Program exists** — which is why U7's in-trade
> search was always going to come back empty. Per the venue's own docs:
> `fee_equivalent = C × feeRate × p × (1−p)`, **feeRate 0.07 for crypto**,
> **rebate share 20 %** (lowest of any category), **paid daily in pUSD**, min
> **$1** accrued. It self-normalises to ≈ **20 % of the fee your own fill
> generated**. **CEILING, BY ARITHMETIC:** `p(1−p) ≤ 0.25`, so the rebate
> **cannot exceed 0.35 c/share**; measured pro-rata on our corpus it is
> **0.168 c/share**. **The threshold to flip the least-negative coin-day is
> 0.5164 c/share — so the rebate cannot flip the sign even at its maximum.**
> **INDICATIVE, NOT INFERENTIAL:** that threshold rests on context-only levels
> with no interval (Q-DA-48).
> **THE RPC LEG WAS NEVER RUN: NOT-REACHABLE, NOT NOT-AFFORDABLE** — all three
> configured public Polygon endpoints return 403/401 from this environment.
> **STILL OPEN, AND NOT BOUNDED BY THE ABOVE:** the venue runs a SECOND,
> separate **Liquidity Rewards Program** paying for orders **resting near the
> midpoint with no fill required**. It is not a share of taker fees, so the
> `p(1−p)` ceiling does not apply to it. `rho` for *that* term stays
> `Unavailable`.


U7 established only that no rebate is paid **inside the trade**. A periodic or
off-chain programme would be invisible to trade receipts, and
`rewards_registry.jsonl` turned out to be a size heartbeat, so the question is
untouched. Every `rho`-dependent estimand is `Unavailable` until this moves.

Take the maker addresses already observed (the 600 receipts, plus the tiered
addresses from U7b) and look for **USDC inflows not attributable to a trade in
our set** — testing for periodicity, and for proportionality to that address's
maker volume.

**Bound the RPC cost before running it.** `eth_getLogs` on USDC `Transfer`
filtered by recipient is block-range-limited on public endpoints; if the cost is
prohibitive, say so and stop rather than burning the budget — "not affordable on
free endpoints" is a legitimate verdict and should be recorded as one.

- **REBATE-OBSERVED** — unexplained inflows found and a schedule readable.
- **ABSENT-OVER-WINDOW** — no unexplained inflows across the observed span.
  Stronger than U7 but still bounded by that span: `rho` stays `Unavailable`
  unless the span demonstrably covers a full reward period, which must be
  argued, not assumed.
- **PARTIAL** — inflows found but not attributable to a rebate.
- **NOT-AFFORDABLE** — RPC limits make it impractical; record the cost estimate.
- **UNRESOLVED** — anything else.

## Cross-cutting defect — the quote guard, found in TWO independent codebases

**Recorded because it is a systematic risk, not bad luck.** The filter
`0.0 < bid < ask < 1.0` excludes `bid == 0.0` and `ask == 1.0` — exactly the
deep-tail quotes where the 0.001 tick lives — and both the coordinator's spread
table and the agent's `u2` probe contained it **independently**, written against
the same tape. It was caught once, in `u2`, and only because 84 observed
`0.01 -> 0.001` transitions contradicted a reported 0.00 % share at 0.001.

Quantified on re-run (coordinator, 12 BTC windows):

```
p<0.15    256,240 -> 318,626 quotes   median 0.0100 unchanged   mean 0.00968
p>=0.85   256,827 -> 319,213 quotes   median 0.0100 unchanged   mean 0.00970
middle three buckets                  unchanged
excluded by the strict guard          124,772 = 5.2%, ENTIRELY from the tails
```

Medians hold, so the flat-half-spread headline stands, and the composition
weighting (~0.470 ¢ at `p<0.15`) is consistent with the measured tail mean.

**Binding consequence:** any future quote filter **must state its exclusion
count beside its result**. This is a standing rule of the programme that was
already written into STATUS/HANDOFF earlier the same session and was then
violated twice, by two people, within hours. A rule that is written but not
enforced by the output format does not survive contact with a deadline.

## Cross-cutting defect — DENOMINATOR/POPULATION MISMATCH (FIVE instances)

**The most valuable artefact this loop has produced. It has caught more errors
than any single uncertainty has resolved.**

One error, five disguises, all in this loop:

| # | where | the mismatch | what it would have said |
|---|---|---|---|
| 1 | **U3a** | conditional-on-gap numerator vs population denominator | spurious **8× flow amplification** — which matched the coordinator's stated prediction |
| 2 | **U3a / U9** | pre-gap rate vs **window-mean** baseline while the gap sat in a busier decile | manufactured "the long gaps are the quiet ones"; the true same-decile ratios are 0.96× and 1.09× |
| 3 | **U1b** | coin×moneyness×side **stratified** sample standing in for a population | **74 %** top-1 concentration instead of the true **100 %** — likely landing in UNRESOLVED |
| 4 | **U6 — THE MOST DANGEROUS INSTANCE** | maker legs on the **complement token** scored in **taker-side** price convention | **0.231 with a 95 % CI excluding 0.50 on the WRONG SIDE** — a confident *backwards* finding that would have read as the discovery *"legs are emitted in reverse priority order"* |
| 5 | **U8** | a **pooled** statistic whose denominator is dominated by one member — btc supplies **2.28 M of 3.73 M** quotes | "modal spread is 1 tick (66.7 %)" read as a market fact when it is a **btc** fact; ATM median is 3–7 ticks on the five thin coins |

**Instance 4 deserves separate weight.** The other three produced results that
were wrong; this one produced a result that was **wrong, confident, and
self-consistent**. It had a tight CI and a plausible mechanism, and **nothing
internal contradicted it** — unlike instance 1, where the magnitude was
implausible, or instance 2, where 84 observed transitions contradicted a 0.00 %
share. **It was caught by suspicion, not by an inconsistency.** A defect class
that can produce internally coherent backwards findings cannot be relied on to
announce itself.

All five are the same error: **a statistic compared against — or pooled over — a
population that does not match the claim being made.** Instance 5 is the
pooling variant: no explicit baseline is wrong, but one member dominates the
denominator, so the pooled figure silently reports that member. Note that
**three of the five flattered a hypothesis someone already held**, which is why this class of
error survives review — it arrives looking like confirmation.

**Binding consequence:** every ratio, rate or share must **state the population
its denominator is drawn from**, and that population must **match the
numerator's conditioning**. A ratio reported without its denominator population
named is not a result.

## Cross-cutting defect — THE NAME IS NOT THE DEFINITION (three instances)

**A landmine for anyone touching collector telemetry.** The field name says
rate; `collect_pm.py:489` stores `self.msg_by_coin.get(coin, 0)` — a
**cumulative message count since process start**. Comparing it between two gaps
compares **process uptime**, not activity.

Cost of not checking: a first pass at U3a used it directly and produced a
**spurious "3.26× elevation"** of first-decile gaps over later ones. The figure
was pure uptime drift and would have supported exactly the wrong conclusion. It
was caught only because the magnitudes (25,348 to 12,590,991) were implausible
for a per-second rate.

**SECOND INSTANCE — `rewards_registry.jsonl`.** The name says a registry of
rewards; the file is **163 rows of `{recv_ns, n}`** — a heartbeat of registry
**size**, carrying no market-level or rate data at all. It had been cited for
two sessions as the place the rebate question would be settled. **Two instances
make this a pattern, not a one-off.**

**THIRD INSTANCE — self-inflicted, and therefore the most informative.** The
agent's own `KNOWN_CONTRACTS` map in `flow_uncertainty.py` labelled
`0xc011a7e1…` as "USDC.e / collateral" — **a guess, never verified against the
contract.** It was then used as a sweep target in U11, where all 9 `eth_getLogs`
calls failed and that token went uncovered, which is one of the two limits
blocking a stronger U11 verdict. A label invented for convenience became an
assumption inside a measurement two iterations later. **Guessing a name is the
same defect as trusting one.**

**Binding consequence:** any use of a collector telemetry field, data file **or
contract/address label** must confirm its *definition at the source* before it
is used as a measurement — the name is not the definition, **including a name
you wrote yourself**. Where a rate is needed and none is
recorded, derive it from the tape rather than from a counter.

**A rate does exist**: `collector.log` heartbeats carry `rate_msg_s` per coin.
U3a did not need it — a direct tape measurement of trade arrivals was both
closer to the estimand and free of this hazard.

## U7 — test specification, stated BEFORE the measurement runs

**META-RULE applied:** every branch below, including the between-cases, is fixed
before the measurement. Two questions, run together because a maker fee **tier**
and a maker **rebate** are the same schedule wearing different signs.

### U7a — is a rebate observable?

**Scope limit, stated first because it bounds every possible outcome.** Trade
receipts can only establish the absence of a **per-trade, in-transaction**
rebate. A **periodic or off-chain** rebate would not appear in them at all. So
no result here can establish "no rebate exists" — only "no rebate is paid inside
the trade".

- **Method:** scan all cached receipts for value flowing **to a maker address**
  beyond the trade settlement itself — any `Transfer`/`TransferSingle` leg whose
  recipient is a maker and which is not accounted for by the fill. Separately,
  read `rewards_registry.jsonl`, never yet examined.
- **CLEARED-OBSERVED** — a rebate-like flow is found **and** its rate is
  readable as a schedule.
- **CLEARED-ABSENT-IN-TRADE** — **zero** unexplained maker-bound value in every
  sampled receipt. This establishes only the in-trade case; `ρ` stays
  `Unavailable` for the periodic case.
- **PARTIAL** *(the between-branch)* — unexplained maker-bound flows exist but no
  schedule is readable. Record the flows; `ρ` stays `Unavailable`.
- **UNRESOLVED** — anything else.

### U7b — is the maker fee a PER-ADDRESS tier? (folded in from U5)

A flat 10 bps / 50 bps split with a third population at 0 bps has the shape of a
per-address schedule. Testable directly: group the **33** `px = 0.99` maker legs
by **maker address** and ask whether the rate is constant within address.

- **TIER-CONFIRMED** — every address with >= 2 legs shows **one** rate, and at
  least two addresses show **different** rates. Selector is the counterparty.
- **TIER-REFUTED** — any address shows two different rates. Record in the U5 row;
  U5 stays `UNRESOLVED`.
- **TIER-UNRESOLVED** *(the between-branch)* — fewer than 2 addresses have >= 2
  legs, so within-address constancy is untestable regardless of what is seen.

## U6 — test specification, stated BEFORE the measurement runs

The charter requires "an independent priority proxy … stated before running".
Fixed here, before any U6 number is read.

**Proxy: price priority.** A matching engine walking the book consumes the best
price first. So if `OrderFilled` legs are emitted in matching order, then within
one transaction, on one asset, legs should appear in **price-priority order** —
for a taker BUY, non-decreasing price (cheapest ask first); for a taker SELL,
non-increasing. Time priority operates *within* a price level and is not
observable here, so **same-price pairs are uninformative and excluded**, counted
and reported.

- **Unit:** an adjacent pair of maker legs within one transaction, same asset,
  **different prices**.
- **Null:** leg order is arbitrary ⇒ correctly-ordered fraction = 0.50.
- **CLEARED if** the correctly-ordered fraction is **>= 0.80** and its 95 %
  binomial CI **excludes 0.50**.
- **REFUTED if** the CI **includes 0.50**.
- **UNRESOLVED if** fewer than **30** informative pairs.

**Ceiling, restated so a pass cannot be over-read:** the chain shows only orders
that **filled**. Even a clean pass yields the *realised* fill sequence, which
**bounds** the marginal-maker correction. It does **not** identify `Q_ahead` for
a counterfactual quote, and no downstream estimand may treat it as if it did.

## U1b — test specification, stated BEFORE the measurement runs

R1 re-scoped U1b from a units question to a **flow-composition** question but
left the test unspecified. Fixed here, before any U1b number is read.

**What it decides.** R-DUAL already protects the *method* — both weightings are
reported regardless. U1b decides whether **excluding** the 0.02 class is
legitimate: one non-market actor ⇒ exclusion is clearly right; dispersed retail
⇒ exclusion discards genuine flow and the count layer must carry it.

- **Frame:** `last_trade_price` events with `size == 0.02` exactly, from
  immutable (`.gz`) window archives across the two complete UTC days,
  **unstratified** — every 3rd archive by sorted name, no coin, moneyness or
  side balancing. The stratified G-FF1 sample cannot answer this and its
  cached receipts are not reused for the draw.
- **Draw:** systematic, `N = 300` transactions, deterministic seed 20260821.
- **Measurement:** fetch each receipt, decode `OrdersMatched.taker_order_maker`,
  count distinct addresses and their concentration.
- **SINGLE-ACTOR if** the top address holds **>= 90 %** of sampled 0.02 trades.
- **DISPERSED if** the top address holds **<= 50 %**, or there are > 50 distinct
  addresses with no dominant one.
- **UNRESOLVED** between those bounds, or at n < 200 validated.

**Pre-committed consequence, so it cannot be chosen after the fact:**
SINGLE-ACTOR ⇒ the class may be excluded from `λ`, with the exclusion published.
DISPERSED ⇒ it may **not** be excluded; it is genuine flow with an unusual size,
and R-DUAL's notional weighting is then the only defence against it dominating
counts. UNRESOLVED ⇒ keep reporting both weightings and exclude nothing.

## U3 — test specification, stated BEFORE the measurement runs

Charter rule 2 requires the test to be fixed in advance; the charter delegates
the choice of test to the agent. Fixed here, before any U3 number is read:

- **Unit:** one `gap_closed` record from `data/pm_5min/collector_gaps.jsonl`.
- **Phase:** `elapsed = (gap_start_ns − window_start_ns) / 1e9` seconds, where
  `window_start` comes from the slug epoch. In-window means `0 <= elapsed <= 300`.
- **Null:** `elapsed` is Uniform(0, 300) for in-window gaps.
- **Statistic:** one-sample Kolmogorov–Smirnov `D` against Uniform(0, 300).
  **Reject uniformity at α = 0.05.**
- **Reported alongside, mandatory:** `n`; `D`; p-value; a decile histogram; gaps
  falling **outside** `[0, 300]` (pre-open / post-close) counted and reported
  separately, never silently dropped; and the same per `collector_version` era,
  **never pooled across eras**.
- **Coverage statement, mandatory:** the ledger begins 2026-08-20 14:50:21 and
  covers ~1/3 of windows. The test is run on the **covered set only**, and the
  uncovered majority is reported as `NO_LEDGER_COVERAGE` — absence of a gap
  record there is **not** evidence of a clean window.
- **Second statistic, because occurrence is not what biases `λ`:**
  duration-weighted **exposure loss per decile** of window phase. The charter
  asks about gap *occurrence*, but a long gap distorts `f_r` more than a short
  one, so exposure loss is the quantity that actually bounds the distortion.
  Both are reported; neither substitutes for the other.
- **Power, stated in advance:** with n on the order of 50, KS detects only gross
  departures from uniformity. A non-rejection is **not** evidence of uniformity
  and must be reported as `INSUFFICIENT_POWER` rather than `CLEARED` if the
  minimum detectable departure is large relative to the deciles.

### U9 — is `PING_TIMEOUT` actually MNAR? OPENED BY THE COORDINATOR 2026-08-21.

Opened on an incidental n=2 finding from U3a, which the agent recorded without
acting on it. Promoted because of what it would move, not because of its
current strength.

**Why this matters more than U4-U8.** `PING_TIMEOUT` is the **largest single
loss contributor — 99.2 s of 201.7 s, ~49%** — despite being only fourth by
count, because its gaps run ~8x longer than a `1013`. It is currently classified
**MNAR-suspect** in the plan, in `clob_adm_v1` and in STATUS, on the strength of
coin concentration alone (8/8 BTC, later 8 BTC + 1 ETH). Its classification
drives the cause-aware admissibility rule, which gates **every** flow, fill and
queue result in the programme.

**The evidence that opened it, and the mechanism it suggests.** In the U3a
first-decile rows the two `PING_TIMEOUT` gaps were **long AND quiet** (11.3 s at
2.70 trades/s, 22.9 s at 0.80/s) while the activity-triggered `1013`s were
**short** (~1.3 s at 4.1-23.3/s). That is a coherent mechanism rather than a
coincidence: a `1013` is an **overload** failure and needs traffic to happen; a
ping timeout is an **idle-connection** failure and is likelier when traffic is
thin. The two causes would then have **opposite** activity correlations, and the
larger of them would be benign.

If that holds, the missingness picture improves materially: ~49% of lost time
would be MAR or better, and the MNAR class would be the short one.

**Decision rule, written before the measurement.**

Measure pre-gap trade rate over a fixed 10 s pre-window for **every** gap, keyed
by cause, per `collector_version` era, never pooled for a verdict. Compare
against the unconditional trade rate over matched window phases.

- **RECLASSIFY-MAR if** the `PING_TIMEOUT` pre-gap rate is at or below the
  matched unconditional rate, with an interval that excludes elevation, at
  n >= 12 within a single era.
- **CONFIRMED-MNAR if** elevated on the same test.
- **UNRESOLVED if** underpowered — state the n required and leave the existing
  MNAR-suspect classification standing. **Underpowered defaults to keeping the
  conservative classification**, never to the convenient one.

Coin concentration alone does **not** decide this. BTC is both the busiest coin
and the one with most sockets, so concentration is expected under either
hypothesis and cannot separate them. Activity correlation is the discriminator.

**Do not amend `clob_adm_v1` on this result without a separate ruling.**

## R2 — RULING: ACCEPTED as a label, authorising nothing

**Coordinator ruling, 2026-08-21.** R2 is **accepted**, with a restriction.

R2 was raised **after** the result, which R1 was not. The test for accepting a
post-result revision is whether it moves against the raiser's interest, and this
one does: the agent's hypothesis was that leg order carries priority, a `CLEARED`
verdict would have favoured it, and `PARTIAL-SIGNAL` is the conservative reading.
Accepted on that basis.

**`PARTIAL-SIGNAL` means: the null is excluded, the point estimate is below the
usability bar, and NOTHING is authorised that `UNRESOLVED` would not authorise.**
It is a record, not a permission. It must never be read downstream as "partially
validated".

**And the substantive verdict is stronger than the label.** U6 has TWO ceilings,
and together they close the lead for the purpose it was opened for:

1. The charter's original ceiling — the chain shows only orders that **filled**,
   so it bounds the marginal correction and cannot identify `Q_ahead`.
2. The one the measurement found, which is sharper — **91 of 154 adjacent pairs
   (59%) are same-price and uninformative**, consistent with the 1-tick modal
   spread. **Time priority operates precisely there and is invisible on-chain.**

Queue position *is* a within-level concept. So even a perfect cross-price result
would say nothing about the quantity the queue bracket exists to estimate.
**U6 does not substitute for the queue bracket. Record that as settled**, while
keeping the real 0.778 price-priority signal (CI [0.6609, 0.8627], excludes 0.50)
as what it is: evidence about cross-price emission order, carrying a ~22%
violation rate, usable at most as a bound.

## META-RULE — decision rules must enumerate the outcome space exhaustively

Added 2026-08-21 after the **second** instance of the same rule-shape defect:
U1b mapped a trichotomy onto a dichotomy (single-actor vs dispersed, silently
routing "a handful of actors" to dispersed), and U6 did it again (cleared vs
refuted vs underpowered, with no branch for "signal real, below bar").

**Every decision rule in this loop must state what happens in the region between
its branches, before the measurement runs.** In particular the case *"the null is
excluded but the point estimate is below the usability threshold"* is a distinct
outcome and must be named. A rule with a gap is not a pre-registration — the gap
is filled after the answer is known, which is exactly what pre-registration
exists to prevent.

## Rule revisions

### R2 — U6's decision rule is a dichotomy over a trichotomy (raised 2026-08-21 AFTER the U6 result)

**Raised, not applied. Disclosed as post-result**, unlike R1 which was raised
before its outcome was known. The U6 verdict above stands as `UNRESOLVED`.

My U6 rule mapped: CLEARED (`f >= 0.80` **and** CI excludes 0.50), REFUTED (CI
includes 0.50), UNRESOLVED (`n < 30`). It has **no branch** for the case that
occurred — **CI excludes 0.50 but `f` is below 0.80** — which is exactly the
defect the coordinator caught in my U1b mapping one iteration earlier. I wrote
the same shape of rule again.

The substantive state is unambiguous even though the label is not: leg order is
**not arbitrary** (0.778 against a 0.50 null, CI [0.661, 0.863]), and it is
**not reliable enough** to carry queue inference at the 0.80 standard I set.

**Proposed amendment, for the coordinator to rule on:** add
**PARTIAL-SIGNAL** — CI excludes the null but the point estimate is below the
usability bar. Consequence: the lead stays **open** and may inform a bound, but
may **not** be built on as a queue proxy. That is deliberately the *conservative*
reading of a result that is directionally favourable to my own hypothesis.

### R1 — U1 is mis-posed (raised 2026-08-21 after the U1 measurement)

**Raised, not applied.** The coordinator owns the rules; this records the defect
and a proposed amendment for decision. The U1 verdict above stands as
`UNRESOLVED` under the rule **as written**.

U1 as written conflates two questions and attaches the blocking consequence to
the wrong one:

- **(a) Can we read `size`?** — *Settled before this loop opened.* WS `size`
  equalled `taker_amount_filled / 1e6` exactly on 600/600 transactions. It is
  shares at 6 decimals. This is the question the volume layer actually depends
  on, and it is answered.
- **(b) Why does the venue permit orders below its stated `orderMinSize`?** — a
  question about venue rule enforcement. The measurement shows it is **not a
  diffuse property of the tape**: it is concentrated in one address.

The charter states U1 "BLOCKS THE ENTIRE VOLUME LAYER" because "every cash
number depends on reading it correctly". That reasoning belongs to (a). Under
the decision rule as written, (b) failing forces `UNRESOLVED`, which blocks the
volume layer on a question the volume layer does not depend on.

**Proposed amendment:** split into

- **U1a — unit.** `CLEARED`: shares at 6 decimals, 600/600 exact against chain.
  Volume-layer *arithmetic* unblocks.
- **U1b — the sub-minimum participant class.** `UNRESOLVED`, and re-scoped from
  a units question to a **flow-composition** question, which is where the real
  risk now sits: if one address contributes a large share of arrival *events* at
  a fixed 0.02 shares, then λ estimated on raw event counts is partly a measure
  of one participant's behaviour rather than of market flow. That contaminates
  the **count layer** — which U1 had assumed was safe.

**Why this is not a convenient re-cut:** the amendment makes the finding *worse*
for the plan, not better. It moves a contamination risk from the volume layer,
which was already blocked, into the count layer, which the plan had declared
available today. Deciding U1b needs a population-representative sample; the
600-tx sample is stratified and cannot supply the share.

## Explicitly OUT of scope for this loop

- `own_impact` (#10) and `venue_ack_lag` (#11) are **not identifiable** and
  **not observed** without placing orders. They are closed as unreachable, not
  open as uncertain. Do not spend iterations on them.
- Anything requiring the 10-day sigma clock.
- Any change to `route_a_v1`/`v2`, the collectors, or the user's DA/EV pipeline.

## R1 — RULING: ACCEPTED, and the consequence is escalated

**Coordinator ruling, 2026-08-21.** R1 is **accepted**. U1 did conflate two
questions and attached the blocking consequence to the wrong one. The split
stands:

- **U1a — can we read `size`? `CLEARED`.** Shares at 6 decimals, exact against
  chain on 600/600. This is the only thing volume arithmetic depends on, so
  **the volume layer is UNBLOCKED.**
- **U1b — why does the venue permit sub-minimum orders? `UNRESOLVED`.**

The revision was accepted specifically **because the agent raised one that made
its own position worse** rather than more convenient, and said so. That is the
behaviour this loop exists to protect; a revision that eased a bar would have
been refused.

**Coordinator measurement, on the UNSTRATIFIED population.** The agent correctly
refused to estimate a population share from a coin x moneyness x side stratified
sample. Measured directly instead, 36,151 trade events over 81 windows, no
addresses required because the size signature is visible in the tape:

| | |
|---|---|
| `size == 0.02` exactly | **5,878 = 16.3% of EVENTS** (POOLED over 81 windows; **per coin this runs btc 2.0% to hype 90.0%** — btc is 64% of any pooled denominator) |
| side mix of that class | 5,876 SELL / 2 BUY (**99.97% one-sided**) |
| its share of notional | **0.0145%** |
| sub-minimum (`< 5`) overall | 35.1% of events |

Modal conforming sizes are round numbers at or above the minimum — 5.0 (8.3%),
10.0 (5.7%), 30.0, 20.0, 15.0. **`orderMinSize` IS respected by the conforming
population.** The distribution is bimodal, and the 0.02 class is a separate
mechanism, not a tail of normal trading.

**THE INVERSION, and it is the important part.** The plan assumed the count
layer was available today and the volume layer blocked. Both are now backwards:

- **16.3% of arrival EVENTS carry 0.0145% of NOTIONAL** — both POOLED over the 81-window sweep. **The event share is btc-dominated and ranges 2.0% (btc) to 90.0% (hype) per coin**; see `FLOW_INTENSITY_RESULTS.md`. Raw-count `λ` is
  materially contaminated by an economically empty class; notional-weighted `λ`
  is essentially untouched.
- So the **count layer is the contaminated one** and the **volume layer is both
  unblocked and cleaner**.

**Binding consequence for the flow model.** `λ` may not be specified on raw
counts alone. Every intensity estimate must be reported **both** ways — raw and
with the 0.02 class separated — with the excluded class published beside the
retained one, per the standing rule. `f_r`, `f_p` and any self-excitation term
fitted on raw counts would partly be fitting one participant's behaviour.

**Still unresolved and NOT to be narrated:** what the 0.02 class is, and whether
it is one address at population level. The one-address finding (74% of
sub-minimum legs, 228 at exactly 0.02, 97% SELL) is from the **stratified**
sample only. The 99.97% SELL signature at population level is *consistent* with
a single actor but does not establish one. Establishing it needs receipts for an
unstratified draw. Do not assert a single actor until that is run.

## Ledger

| # | uncertainty | verdict | evidence | scope |
|---|---|---|---|---|
| U11 | the rebate, off-trade | **PARTIAL — no rebate found, strongest candidate REFUTED, but coverage is incomplete and the span cannot establish absence. `ρ` stays `Unavailable`.** | **Cost bounded first, as required:** the endpoint accepts a topics OR-array of all **199** maker addresses at **10,000-block** spans (40,000 fails), so the **80,355-block (~47 h)** span costs **9 calls/token**. Affordable — measured **362 s** for the sweep, 313 s for the follow-up. **Sweep result:** 10,742 inbound USDC transfers to our 199 makers. Senders: `0x4d97dcd9…` **ConditionalTokens** 8,871 transfers / $848,800 (redemption + merge payouts, not a rebate); the **CTF Exchange** 822 / $8,954 (trade settlement); zero-address 581 to 116 recipients / $81,582 (mints/bridging); then 232 other senders, of which **16 pay >= 5 distinct makers** — the rebate-programme shape. **Strongest candidate REFUTED:** `0xc417fd8e…` turns out to make **8,227** outbound transfers to 58 recipients totalling **$3.22 M**, with a **median block gap of 0** — continuous, not periodic, and only 23 of those transfers touched our makers at all. That is a hot wallet, not a reward schedule. Remaining candidates are $10–70 in total, orders of magnitude below any plausible rebate on ~$4.7 M of sampled notional. | **TWO limits that block a stronger verdict.** (1) **Coverage is partial:** the sweep of the second token address (`0xc011a7e1…`) **failed all 9 calls** with HTTP 400, so only native USDC was swept — and that address's label in `KNOWN_CONTRACTS` was **my guess, never verified**, a third brush with *the name is not the definition*. (2) **The span is ~47 h and cannot be argued to cover a full reward period** — Polymarket schedules are plausibly daily or weekly, and the charter requires that coverage be *argued, not assumed*. So this is stronger than U7's in-trade result and still **not** `ABSENT-OVER-WINDOW`. Every `ρ`-dependent estimand remains `Unavailable`. |
| U10b | does the published `−0.211 ¢/fill` headline survive an interval? | **NO — WITHDRAWN. It spans zero. And the ONLY interval in the whole analysis that excludes zero is the un-harvestable one.** | Window-clustered bootstrap, 931 windows, 10,000 resamples. **per-fill ALL flow +0.165 [−0.377, +0.734] — spans zero. per-fill EXCLUDING the 0.02 class −0.211 [−0.849, +0.457] — SPANS ZERO. per-fill 0.02 class ALONE +1.987 [+1.529, +2.440] — EXCLUDES ZERO.** Per-coin excluding the class: **all seven span zero** (bnb −0.614, btc −0.130, doge +0.662, eth −0.526, hype −1.327, sol −0.336, xrp −0.592). So *"on real flow, makers lose per fill"* — stated as fact in `HANDOFF.md`, in commit `6a0e593`, and relayed onward — **is not supported**. The 0.02 class's interval is tight *because* it is one counterparty behaving consistently, and it carries **~$91 of capacity over two days**: the sole statistically distinguishable maker edge in the tape is un-harvestable by construction. **TWO CLAIMS KEPT APART:** the **estimator** finding — the two weightings diverge in sign, so a single-weighting spec would have reported "+0.165, makers profitable" and hidden the dependence on one counterparty — is about **estimator behaviour** and survives regardless of any interval; the **economic** claim needed the interval and does not survive. | 931 windows, 2 UTC days, every 3rd archive. **Window clustering misses day-level common factors, so these intervals UNDERSTATE uncertainty — and nothing on real flow excludes zero even so.** Repro: `flow_uncertainty.py u10b` |
| U10 | are the per-coin maker edges real, or noise around zero? | **PARTIAL — and ZERO coins survive, so it authorises NOTHING. The adverse-selection story is WITHDRAWN.** | Window-clustered bootstrap, 931 windows, 10,000 resamples. **Not one of the seven per-coin CIs excludes zero:** bnb −0.476 [−2.229, +1.359] · btc +0.201 [−0.269, +0.699] · doge +0.629 [−1.094, +2.487] · eth −0.008 [−0.890, +0.807] · **hype +1.761 [−0.304, +3.809]** · sol −0.628 [−1.624, +0.372] · xrp +0.705 [−0.226, +1.716]. Permutation test on coin labels (10,000 shuffles) gives **p = 0.0482** — dispersion is *marginally* beyond the common-mean null, but **it names no coin**, because no individual interval separates from zero. Under the charter's PARTIAL branch ("name exactly which coins survive; authorise nothing beyond those") the surviving set is **empty**. **And the POOLED edge also spans zero: +0.1727 ¢/share, 95 % CI [−0.2509, +0.5963].** So the maker edge is **not distinguishable from zero at either level**. **The mechanism I attached to the sign-flip table — that a wide spread prices adverse-selection hazard so compensation and hazard scale together — is withdrawn: the pattern is fully consistent with every coin sitting near zero and the dispersion being sampling noise.** What survives is only the *negative* result: spread width does not predict edge. | 931 windows, 2 UTC days, every 3rd archive. **MANDATORY CAVEAT: window clustering cannot capture day-level common factors, so with TWO days these intervals UNDERSTATE uncertainty — excluding zero would be NECESSARY, NOT SUFFICIENT, for a day-robust claim, and nothing here excludes zero anyway.** p=0.0482 is marginal and fragile at this n. Repro: `flow_uncertainty.py u10` |
| U8 | spread on thin coins | **CLEARED — and it REVERSES the plan's headline. "ATM runs 6–8 ¢" is refuted for BTC/ETH but CONFIRMED for the thin coins.** | ATM (`0.35–0.65`) median spread **in ticks**, 8 windows/coin, 2026-08-20, all at the 0.01 tick: **btc 1 (89.8 % at 1 tick) · eth 1 (74.4 %) · sol 3 (22.9 %) · doge 3 (13.7 %) · xrp 5 (14.8 %) · bnb 5 (4.7 %) · hype 7 (2.4 %)**. So ATM half-spread runs **0.5 ¢ on btc/eth but 1.5–3.5 ¢ on sol/doge/xrp/bnb/hype**, and share `>=5` ticks at ATM reaches **61.8 % on hype, 57.2 % on bnb, 51.4 % on xrp**. **The corpus's "modal spread is 1 tick (66.7 %)" is a POOLING ARTEFACT** — btc alone supplies 2.28 M of ~3.73 M quotes, so the pooled statistic largely reports btc. Same shape as the U4 per-coin lesson, a fifth cousin of the population-mismatch family. **Economic consequence:** against a fixed ATM fee of 1.75 ¢, a 0.5 ¢ half-spread (btc) and a 3.5 ¢ half-spread (hype) are different businesses. **But spread width does NOT confer edge — at identical widths the U4 sign FLIPS:** 3 ticks → doge **+0.629** vs sol **−0.628**; 5 ticks → xrp **+0.705** vs bnb **−0.476**. Only the extremes agree (btc 1 tick/+0.201, hype 7 ticks/+1.761). A monotone "wide spread ⇒ good business" reading is **false**; the pattern is consistent with wide spreads **pricing adverse-selection risk** rather than conferring free edge. Tick composition in the tails also varies by coin: 0.001 share runs **6.4 % (bnb) to 16.2 % (doge)**. | 8 windows per coin, **2026-08-20 only**, both pair tokens counted. Deterministic window selection. Thin coins carry far fewer quotes (hype 78 k vs btc 2.28 M), so their tails are correspondingly noisier — one `doge` cell (`0.15–0.35` @0.001, n=278) shows a 239-tick median and is flagged rather than interpreted. Repro: `flow_uncertainty.u2(coin=…)`. |
| U7 | maker rebate `ρ(p)`, + the per-address fee-tier check folded in from U5 | **U7a CLEARED-ABSENT-IN-TRADE** (`ρ` still `Unavailable`) · **U7b TIER-CONFIRMED** | **U7a — no rebate is paid inside the trade.** Across **600 receipts** only **5 distinct emitting contracts** appear and **all are accounted for** (CTF Exchange, ConditionalTokens, USDC, USDC.e, MATIC gas) — **zero unknown/third-party contracts**. Value-flow check: of 570 receipts containing a token-buying maker, only **3** showed that maker also receiving USDC, and **all 3 are two-sided in the same transaction** with inflows matching their sell legs **exactly** (51,000 / 7,800,000 / 5,070,000). So zero unexplained maker-bound value. `rewards_registry.jsonl` is **163 rows of `{recv_ns, n}`** — a registry-**size** heartbeat with no market-level data, confirming it cannot answer this. **U7b — the maker fee IS a per-address tier, which identifies U5's missing selector.** Over the 33 `px=0.99` maker legs: **5 addresses have >=2 legs and every one shows exactly ONE rate** (`0x1e746427` 0.0 ×3, `0x2277c18f` 0.001 ×3, `0xbdf22122` 0.005 ×2, `0xeebde7a0` 0.0 ×2, `0xb3b0780f` 0.001 ×2), with **three distinct tiers across addresses: 0 bps, ~10 bps, 50 bps**. The selector is the **counterparty, not the trade**. | **U7a's scope limit was stated before running and bounds the result: trade receipts can only establish absence of a PER-TRADE, IN-TRANSACTION rebate.** A periodic or off-chain rebate would not appear in them, so this is **not** evidence that no rebate exists, and every `ρ`-dependent estimand stays **`Unavailable`**. **U7b is thin: 5 addresses at n=2–3 legs each** — constancy on 2–3 observations is suggestive, not strong. 600-tx stratified sample, cached receipts, zero RPC. |
| U6 | does on-chain leg order carry matching-engine priority? | **UNRESOLVED — my own decision rule has no branch for the outcome.** The lead is **NOT refuted**; it also does not clear the bar. | **Result: 49/63 adjacent informative pairs correctly ordered = 0.7778, 95 % CI [0.6609, 0.8627].** The CI **excludes 0.50 decisively**, so leg order is **clearly not arbitrary** — but 0.778 is **below the 0.80 bar** I pre-registered, and the rule specifies only CLEARED (>=0.80 *and* CI excl. 0.50), REFUTED (CI incl. 0.50) and UNRESOLVED (n<30). n=63. **The outcome falls between the branches — see R2.** **A CORRECTION I nearly reported as a finding:** the first run returned **0.231, anti-correlated**, with the CI excluding 0.50 on the *wrong side*. Cause: in mint-match transactions the maker legs sit on the **complement** token, so their prices run inverted to the taker's book, and the taker-side convention was being applied to complement-space prices. **Same class as the DENOMINATOR/POPULATION MISMATCH entry — a statistic scored against a convention drawn from a different space.** After normalising every leg into taker-asset space (179 legs inverted) the figure is 0.778. A hypothesised multi-`OrdersMatched` contamination was checked and **does not exist**: all 600 transactions carry exactly one. | 600-tx G-FF1 sample, **stratified**, cached receipts, zero RPC. **A limitation sharper than the charter's stated ceiling: 91 of 154 adjacent pairs (59 %) are SAME-PRICE and uninformative** — consistent with the 1-tick modal spread. Time priority operates precisely there and is invisible on-chain, so even a perfect result would speak only to *cross-price* ordering while most book action is *within* a level. Practical read: ~22 % of informative pairs violate price priority, so leg order as a queue proxy carries that error rate. Chain shows only orders that **filled**: bounds the marginal correction, does **not** identify `Q_ahead`. |
| U5 | the 10 of 754 fee-bearing maker legs | **UNRESOLVED — mechanism NOT identified.** Characterisation much improved; no story offered. | **Two hypotheses REFUTED.** (a) *"an address acting as both maker and taker within one transaction"* — **false for all 10** (`maker_is_a_taker_of_this_tx = False`, 10/10). (b) *"they follow the taker fee formula"* — **no**: `0.07·p(1−p)·N` predicts 69,300 against an observed **495,000**, 7× out. **What IS established:** all 10 sit at **px = 0.9900 exactly**; the fee is a **flat rate on the USDC amount — 10 bps on 7 legs, 50 bps on 3** (`fee/mAmt` = 0.001000 or 0.005000 to six decimals); they occur in **mint-match** transactions where the taker's order is on the **complement** token at 0.01 and every fee-bearing counterparty is buying the 0.99 side; 5 distinct transactions, 6 distinct makers. The fee-leg in the same transactions matches the taker formula **exactly** (`0.07 × 0.01 × 0.99 × 101 = $0.06999 = 69,990`), which is what isolates the maker legs as a different rule. **DISCRIMINATING CHECK, and it is why this is UNRESOLVED: px = 0.99 is NECESSARY BUT NOT SUFFICIENT — 23 maker legs at exactly 0.99 carry NO fee against the 10 that do.** So price does not determine it, and what selects fee-vs-no-fee, or 10 bps vs 50 bps, is **unknown**. | 600-tx G-FF1 sample, **stratified**, `clob_v3_1`-era cached receipts, zero RPC calls. 1,354 legs: 600 fee-legs (`taker == exchange`), 744 maker zero-fee, 10 maker-with-fee. **Does not disturb the headline:** makers pay ~0 on **98.7 %** of legs; this is a precisely-described 1.3 % residual, not a competing fee schedule. Repro: inline probe over cached receipts. |
| U4 | provenance of the maker markout `+0.45 ¢/share` | **CLEARED — TRACED and REPLACED. The figure is ~2.6× too high.** **SUPERSEDED BY U10b ON THE SIGN: the `-0.211` per-fill figure in this row SPANS ZERO `[-0.849, +0.457]`, all seven per-coin CIs span zero, and the pooled `+0.173` spans zero at `[-0.251, +0.596]`. "On real flow makers lose per fill" is WITHDRAWN; the sign is UNDETERMINED at two days. The per-coin "MIXED" reading below asserts structure the intervals deny.** | **Traced:** `PM_DEEP_REVIEW.md:126` (mid-conditioned, "re-run on the mid") and `:738`, where it is **the same number as the "+95 bps maker gross / +136 with rebate" claim**. Both carry `stale_book_contamination: true`. **Replaced by a model-free rebuild that needs no mid and no book at all** — `taker BUY at L ⇒ maker sold ⇒ edge = L − outcome`; `taker SELL ⇒ edge = outcome − L`, using only trade price, taker side (G-FF1 `PASS`) and the settled `winners` field. **Mapping hard-validated before any figure was read: winning-token mean price in the last 30 s = 0.8943 (n=8,204), so Up/Down is not inverted.** **Result, in-window, n=447,380 fills: per-fill +0.165 ¢, share-weighted +0.173 ¢** — against the cited +0.45. **R-DUAL paid off here:** excluding the 0.02 single-actor class the per-fill figure goes **NEGATIVE, −0.211 ¢**, while share-weighted is **unmoved at +0.172 ¢**. The count-weighted maker edge is positive *only* because of that one actor (+1.987 ¢/fill on 17.1 % of fills). **That edge has NO CAPACITY: the class is 0.0153 % of notional — $727 against $4,740,742 **in the 980-archive U4 census** (the 0.0145 % quoted elsewhere is the same quantity over the 81-window sweep; both are correct for their own denominator, and neither was labelled) — and its entire maker edge is $30.42 in the subset, ~$91 scaled market-wide over two days.** So the sole positive per-fill maker edge in the tape is un-harvestable by construction, and **on real flow the maker loses −0.211 ¢ per fill.** **Per-coin share-weighted signs are MIXED** — bnb −0.476, sol −0.628, eth −0.008 negative; btc +0.201, doge +0.629, xrp +0.705, hype +1.761 positive — so "positive in 6 of 8 bins" does not survive per-coin. Other phases, share-weighted: pre-open **−0.573 ¢** (cited +0.70, sign flips), post-close **−1.732 ¢** (cited −1.46, confirmed directionally). | 980 archives (every 3rd), both complete UTC days, unstratified, windows with a resolved `winners` record. **Two days ⇒ NO day-clustered CI is computable. Point estimates with UNKNOWN sampling error — anything downstream needing an interval is `Unavailable`, structurally, not as a caveat.** Per-coin is primary, pooled is diagnostic. The rebuild sidesteps `stale_book_contamination` entirely rather than working around it — no book state is read. Repro: `flow_uncertainty.py u4` |
| U9 | is `PING_TIMEOUT` actually MNAR? | **UNRESOLVED — `MNAR-suspect` STANDS.** And the incidental finding that opened it **does not replicate**. | Phase-matched baseline (same window, same 30 s decile, **gap interval excluded** so it cannot depress its own comparator), `clob_v3_1`, 42 in-window gaps, 20,000-sample bootstrap CI on the mean ratio. **`PING_TIMEOUT` n=7, median 1.05, mean 1.28, CI [0.91, 1.72] — spans 1.0, and the point estimate sits mildly ABOVE 1, not below.** Rule requires n ≥ 12 in a single era; **5 more needed**. Underpowered ⇒ conservative label retained. **THE OPENING FINDING IS WITHDRAWN:** U3a's "long AND quiet" `PING_TIMEOUT` observation (elevations 0.38×, 0.40×) compared pre-gap rate against the **window mean**. The first decile is busier than the window mean, so a first-decile gap reads "quiet" against that baseline purely as a **phase confound**. Against the correct same-decile baseline the ratio is ~1.05. The idle-connection mechanism is **not supported**. Other causes, for the record: `SLOW_CONSUMER_1013` n=25, median 0.96, CI [0.83, 1.03]; `CONNECTIONCLOSEDOK` n=5, CI [0.99, 1.57]; `NO_CLOSE_FRAME` n=3, CI [0.36, 0.69]. | `clob_v3_1` only, never pooled. **Proxy caveat, load-bearing:** this measures **trade** arrival rate. A `1013` is triggered by **message** rate, which is ~97 % `price_change`, so a book-update storm with no trade burst would not show here. The measurement is therefore the right one for *"does the loss bias the flow estimand"* and the **wrong** one for *"what causes the disconnect"* — the 1013 row must not be read as evidence against its overload mechanism. `clob_adm_v1` **NOT amended** (coordinator ruling). Repro: `flow_uncertainty.py u9` |
| U1b | is the 0.02 class one actor or dispersed flow? | **CLEARED — SINGLE-ACTOR** (identical under both the original and the amended mapping) | Unstratified systematic draw, 300 transactions, **0 receipt failures**. **Concentration curve: top-1 = 100.0 %, top-5 = 100.0 %, top-10 = 100.0 %, distinct addresses = 1, HHI = 1.0000.** All 300 draws resolve to the **same single address** (`0x674887d1…`), and it is the only distinct address in **every one of the 7 coins**. Population in the scanned subset: **76,540** events at exactly 0.02 shares, side mix **76,527 SELL / 13 BUY (99.98 %)** — consistent with the coordinator's independent population figure (5,878 in 81 windows = 72.6/window; here 76,540 in 980 = 78.1/window). **Pre-committed consequence applies: the class MAY be excluded from `λ`, with the exclusion published.** R-DUAL is unaffected — both weightings still reported. **What this is NOT:** the mechanism is not established and is not narrated; "one address" is a measurement, any account of *what it is doing* would not be. | 980 archives (every 3rd, both complete UTC days), **unstratified** — no coin/moneyness/side balancing. G-FF1's cached receipts deliberately **not** reused: stratified by coin×moneyness×side, they cannot carry a population claim about concentration. Repro: `flow_uncertainty.py u1b` |
| U3a | AMENDMENT to U3: does the exposure bound bound the **flow** distortion? | **NO — corrected. The bound is EXPOSURE lost, not FLOW lost.** The coordinator's structural point was right; the coordinator's **directional prediction was WRONG** — flow loss was predicted to exceed time loss and runs at **0.43×** of it. | **The claim withdrawn:** U3 leaned on 0.155 % as if it bounded the `λ` distortion. Exposure and flow coincide only if loss is independent of `λ`, and the dominant cause is by construction the one that is not. **`coin_msg_rate_hint` cannot supply the bound** — `collect_pm.py:489` stores `msg_by_coin`, a **cumulative counter since process start**, not a rate; comparing it across gaps compares process uptime. (A first pass using it gave a spurious "3.26× elevation".) **Measured directly from the tape instead:** trades in the 10 s before each gap vs the local baseline. **First-decile cause mix, re-derived within `clob_v3_1` only: 4 `SLOW_CONSUMER_1013`, 2 `PING_TIMEOUT`, 1 `CONNECTIONCLOSEDOK` (n=7; 6/7 MNAR-class), coins btc 5 / bnb 1 / eth 1.** **Matched-denominator result, first decile:** time lost 41.0 s of 210 s = **19.53 %**; flow lost (LB) 114.4 of 1,369 first-30 s trades = **8.36 %**; **ratio flow/time = 0.43×**. **MECHANISM UNEXPLAINED (corrected after U9).** The original sentence — *"the long gaps are the quiet ones"* — is **WITHDRAWN**: it read a window-mean-relative elevation as quietness, which U9 showed to be a phase artefact. Against their own same-decile baselines the two long `PING_TIMEOUT` gaps sit at **0.96× and 1.09×** — *typical for their windows, not quiet for their moment*. Their absolute pre-rates (2.70/s, 0.80/s) were low because those windows were **absolutely** quiet. **Why long gaps landed in absolutely-quiet windows is not explained, and no replacement story is offered.** The ratio is recomputed at **0.428×** and **survives unchanged**, because every input is absolute (`pre/10 s`, first-30 s trade counts) and no baseline-relative quantity enters it. Even doubling the `1013` during-gap rate leaves the ratio at 0.65×, still < 1. **A result confirming the coordinator's prediction was available and was refused:** the *unmatched* comparison read 1.241 % flow against 0.155 % time, an apparent 8× amplification exactly as predicted, but it sets a conditional-on-gap numerator against a population denominator. The matched figures above are the reportable ones. | **n=7. Directional only — this cannot support a rate claim.** `clob_v3_1` covered set only. Pre-gap rate is a **lower** bound on during-gap rate, so the flow figure is a lower bound; the sensitivity check above is what keeps that from mattering. Incidental, n=2: the two first-decile `PING_TIMEOUT` gaps are long *and* quiet, which is **evidence against** its MNAR classification — recorded, not acted on. Repro: `flow_uncertainty.py u3b` |
| U3 | is gap occurrence independent of `r`? | **CLEARED — via the bound branch, NOT the uniformity branch** | Two pre-registered statistics **disagree, and the disagreement is the finding**. (1) **Occurrence:** KS vs Uniform(0,300) does **not** reject — pooled `D=0.132, p=0.312`; `clob_v3_1` `D=0.151, p=0.270`. But at n=51 the smallest detectable departure is `D=0.190`, i.e. ~19 % of probability mass against 10 %-wide deciles. Per the pre-stated power rule this is **INSUFFICIENT POWER, not evidence of uniformity** — the uniformity branch does **not** carry. (2) **Exposure loss, the statistic that actually biases `λ`, IS concentrated:** decile 0–30 s carries **31.7 %** of all in-window seconds lost, a **3.2×** the-mean loss rate, versus 1.4 % in the final decile. So the charter's FAIL branch applies — and its requirement is met: **explicit distortion bound = ≤ 0.155 % of exposure in the worst decile, 0.0488 % overall** (129.3 s lost across 882 covered windows × 300 s = 264,600 window-seconds). That is negligible against an `f_r` that varies by factors, so **`f_r` is reportable with the bound stated**. Outside `[0,300]`: 5 pre-open, 3 post-close, reported not dropped. | **Covered set only**, per era, **never pooled for a verdict** (the pooled row is printed as diagnostic and labelled as such). Covered windows: `clob_v3_1` 882, `clob_v3` 98, `clob_v2_1` 112, `clob_v2` 14. The bound is derived on `clob_v3_1` and **applies only there**. The 2,019 pre-ledger windows carry **no gap record, which is not evidence of cleanliness** — `f_r` estimated outside the covered single-era set inherits no bound at all. Repro: `flow_uncertainty.py u3` |
| U2 | tick composition; convention vs constraint | **CLEARED** | Tick read from **both** `tick_size` and `new_tick_size`. 6,702,978 executable quotes. **84 transitions, all `0.01 -> 0.001`, none reverting.** Composition: 0.001 exists **only in the tails** — 6.75 % of `p<0.15` quotes and 6.73 % of `p>=0.85`, **0.00 % in all three middle buckets**. **Convention-vs-constraint: CONSTRAINT, not convention.** Where the 0.001 tick is available the spread is **1 tick in 99.9 %** of quotes — makers step inside immediately when the finer grid appears. Where the tick is 0.01, spread is 1 tick in 90.8–97.2 %. Corollary: the "flat 0.50 ¢ half-spread" is a **median** artefact; conditional on the 0.001 regime the half-spread is 0.05 ¢, 10× finer, but that class is only 6.7 % of tail quotes so it does not move the median. **MY OWN DEFECT, disclosed:** the first `u2` run reported 0.00 % at 0.001 because the quote guard `0.0 < bid < ask < 1.0` excluded `bid == 0.0` / `ask == 1.0` — exactly the deep-tail quotes where the 0.001 tick lives — and reported no exclusion count. Caught **before reporting** only because 84 observed transitions contradicted a 0.00 % share. Guard fixed to `0.0 <= bid < ask <= 1.0`; 489,806 boundary quotes now retained and flagged. | **BTC only, 2026-08-20**, 24 of 288 windows (every 12th, deterministic). Both pair tokens counted, so each event contributes two quotes at complementary moneyness. Do **not** generalise to thin coins — that is U8. `exp_gff1_side.py` deliberately **left unpatched** to preserve provenance of the recorded v3 run; the corrected read lives in `flow_uncertainty.py`. Repro: `flow_uncertainty.py u2` |
| U1 | `size` semantics | **UNRESOLVED** | Partial-fill hypothesis **REFUTED**: 0 of 600 `taker_order_hash` appear in more than one match, so there is nothing to aggregate — the WS `size` already *is* the taker-order total. Order-level compliance with `orderMinSize=5` is **46.8 %** (281/600), against a 99 % bar. Alternative rules do not reproduce it either: notional ≥ $5 → 20.3 %, ≥ $1 → 50.2 %. Per-coin 38.9 %–65.6 %, `p10 = 0.020` in **every** coin. **Locus found:** 319/600 legs are sub-minimum and **74 % of them come from ONE address** (`0x674887d1…`), which is 243/600 of the whole sample, 228 of them at exactly 0.02 shares, 236 SELL vs 7 BUY, spread evenly over all 7 coins and all 5 moneyness buckets. `orderMinSize=5` uniform on 3,102 market rows. | 600-tx G-FF1 sample, **stratified** by coin×moneyness×side, 2026-08-19/20, `clob_v3_1`-era receipts. Stratification forced 300 BUY/300 SELL, so the address's **population** share is NOT 40.5 % and is not estimable from this sample. Repro: `flow_uncertainty.py u1` |
