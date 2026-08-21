# BE-FlowAndFills — design plan

> **⚠ For current state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).** This
> document is **provenance** — correct about its own moment, not a statement of
> current belief. Where it conflicts with `FLOW_MODEL_STATE.md`, that page wins.


> **SUPERSEDED 2026-08-21.** Retained as the fill-model audit trail. Its queue
> bracket and joint fill/markout insights are folded into canonical Revision 4
> in `BE_FLOWANDFILLS_MODEL_PLAN.md`; its stale sigma, markout, fee and Hawkes
> conclusions are not current. Do not implement directly from this file.

Module planner's deliverable, P-2026-003. Refines §6 of `plans/PRELIMINARY_PLANS.md`.
**Plan only — no implementation.** The canonical contract is the YAML block in §11.

This module answers *"what happens if I rest at level ℓ"*. Because the programme has
measured that we hold **no forecasting edge over the book** (EXP-BLEND: the book wins
at every horizon by 2.5–3.2 Brier points; the σ-ladder retest still loses by +0.0201),
the strategy is spread capture, and the entire P&L is this module's estimand. Today it
is a placeholder whose null pins `{ζ = 0, p_fill = 1.0, own_impact = 0}` as `ASSUMED`
— the three most optimistic values available.

---

## 0. Premises corrected before designing

Five briefing premises are wrong or mis-scoped against the data on disk. The design
below is built on the corrected column, not the briefed one.

| briefed | measured | source |
|---|---|---|
| trades **double-reported** across the pair; `transaction_hash` dedup or intensity is **2× too high** | **zero duplication.** 446,412 `last_trade_price` events on 20260819 → 446,412 distinct hashes. Each trade is reported on **one** `asset_id` only. The hazard is the **opposite**: fold the complement token or you see **half** the flow at a unified level | full parse, 20260819 |
| books typically **2–4 ticks** wide | modal spread is **1 tick (66.7%)**; 2t 14.9%, 3t 6.9%, 4t 4.3%. Per-coin P(1 tick): btc .79 · eth .57 · sol .34 · xrp .26 · bnb .13 · doge .12 · hype .06 | 938,862 book events |
| trade sizes p10=2 / p50=9 / p90=53 | that is **BTC**. Pooled: p10 **0.02** / p50 **5.81** / p90 **42.76**, mean 22.23. Alt-coin p50 is 0.02–5 shares | 446,412 trades |
| **87%** are BUY | 77.18% pooled; **btc 87%, hype 12%**. Decisively: the skew is **symmetric across both tokens** (Up-token 77% BUY, Down-token 77% BUY), which is itself the proof that `side` is a per-token taker convention and carries no unified direction | 446,412 trades |
| measured lag **1.7 s p50** | that is the **Chainlink→PM TWAP60 publication leg** (the *signal* clock). **Market-data lag is 47–58 ms p50** (hard floor 46 ms = WS edge RTT). A third clock, the spot mirror, is 448 ms p50 (≤1 s of that is payload-ts truncation) | 3 feeds |

Two more facts that reshape the design:

- **`price_change.size` is the level TOTAL** — replaying it reproduces the next book
  snapshot **exactly in 96.10%** of 937,286 checks, and its carried `best_bid`/`best_ask`
  match the **post-change** state in 99.67% of 387,022 checks. So (a) `Q_ahead` is
  reconstructible at ~96% fidelity, and (b) **the quotes carried on a `price_change` are
  post-trade** — the exact contamination that made the corpus's spread/AS split
  untrustworthy (measured capture +0.11…+0.50 c against 1.1–1.8 c spreads, below a
  half-spread, a signature of a mid that has already moved).
- **`tick_size_change` fires in 567 of 766 windows** (2,320 events, always 0.01↔0.001 as
  price approaches 0/1). A 10× level-grid change is **routine, not a regime event**.

### The three clocks

| leg | p50 | consequence |
|---|---|---|
| PM market data (book, trades) | **47–58 ms** (floor 46) | we react to *book* events fast |
| spot mirror `crypto_prices` | **448 ms** (≤1000 ms is payload truncation) | we react to *price* moves at ~0.4 s |
| Chainlink → PM `twap_sixty` (**the settlement stream**) | **1,700 ms** (1,440 ms is PM-side publication) | the venue's own fair value is 1.7 s stale |

This ladder *is* the adverse-selection mechanism and it is the reason the corpus's
"cancellation is a race lost by 2–3 orders of magnitude" needs re-scoping: it was
derived from the 1.7 s TWAP leg and applied to book-driven cancels, which run on the
47 ms leg. **Which leg binds is an open, cheap, decisive measurement (§4.5).**

---

## 1. The estimand

### 1.1 What is estimated

For an `ActionSet` **A** (all legs together) at knowledge time *t* over horizon *H*,
the module estimates the **joint law of the fill path and its consequences**:

```
L_t(A) = Law( { (τ_a, N_a, M_a(·)) }_{a∈A}  |  F_t^known , SelfState )
```

per candidate action *a* resting at unified level ℓ_a with size s_a:

| symbol | meaning | why it is in the estimand |
|---|---|---|
| `N_a ∈ [0, s_a]` | filled **quantity** by t+H | partial fills are first-class: `orderMinSize` 5 vs trade p50 5.81 shares — a 5-share order is a *whole* median trade, not a marginal slice |
| `τ_a` | **first-fill time** | markout is censored by `T − τ_a`; a fill at r=20 s has 20 s of markout, not H |
| `M_a(h)` | markout **per filled share**, ¢/share | see §2.2 for the unit argument |
| partition | `spread + transient_AS + permanent_AS + snipe + own_impact` | R-ONCE, sums exactly per fill (§7) |

**This is not "P(fill)" and not "E[markout]".** The objective form currently in
`PM_MM_PLAN` §3,

```
ℓ* = argmax_ℓ  λ_fill(ℓ, queue) · [ P_ℓ − CE_quote(q) − ζ(ℓ) ]
```

is **rejected as an estimand** (it survives only as a *reporting* decomposition). It
multiplies a marginal fill rate by an **unconditional** ζ. Fills are selected — you are
filled precisely when the flow is against you — so that product is not `E[PnL]`. The
programme has already measured how large the gap is: capture-ratio haircuts of **60%**
(t=290 s, mid ∈ [0.95,1.00): unconditional +1.8 ¢ → realised +0.72 ¢) and **97%**
(t=30 s, mid ∈ [0.15,0.30): −9.4 ¢ → +0.25 ¢). Selection is the dominant term, not a
correction. The module emits `E[N_a · M_a]` as **one** object.

### 1.2 Representation and the dependence contract

`Uncertain[ActionOutcome]` is realised as a `ScenarioDrawSet[ActionOutcome]` carrying
**one `common_random_id` shared across every action in the set and every draw**:
`Dependence = SharedDraws`.

Three things this buys, each of which is a recorded failure otherwise:

1. **Fill ⊥̸ markout.** `Independent(DECLARED)` is **refused** for this input.
2. **Join-vs-improve is compared on one flow path.** Independent draws per action make
   the comparison draw-noise-dominated; the corpus already flags the argmax over 3–4
   estimated EVs as a small-N selector with a ~2 SE best-vs-second gap, the same failure
   class as the vBTC rolling-IC selector (noise-dominated, value-negative).
3. **The second leg is priced at post-first-fill `q`.** Quoting both legs off the same
   pre-fill `q` double-counts the inventory credit. Joint evaluation on shared draws
   makes that structurally impossible rather than a convention to remember.

**Wiring constraint.** `DecisionSchemeConfig.unwrap_policy` for this input must be
`Draws(n, SharedDraws)`. `Expectation` destroys the dependence the estimand exists to
carry; the module refuses that wiring at composition time.

### 1.3 Coordinates: one book, in Up-space

`price_changes[]` has **length 2 in 100.00%** of 28.7M messages — mirrored, `side`
flipped, price complementary. The pair book is one book *as a matter of fact*, not of
modelling taste. Canonical coordinate: **Up-price space**, `Down@q ≡ Up@(1−q)`.

- A resting *buy Up @ℓ* and a resting *buy Down @(1−ℓ)* are the **same queue** at the
  same unified level; they cross via CTF mint.
- **Correction to my own first measurement:** I found the unified book strictly tighter
  than the Up book 27.2% of the time (p50 6→5 ticks) on a snapshot-based reconstruction.
  Given exact mirroring at the update level, that is **snapshot asynchrony, not free
  tightening** — retracted. The unified coordinate is a *relabelling*, and it is
  mandatory for the fill process (a Down fill is a fill at your unified level), not a
  source of a better mid.
- **Trades are single-sided.** Unlike the book, a trade appears on one `asset_id`. Flow
  at unified level ℓ = (trades on Up @ℓ) **+** (trades on Down @1−ℓ, sign-flipped).
  Skipping the fold halves every intensity estimate.

### 1.4 Window phase is a state variable, not a stratum

| phase | share of trades | measured maker markout | design consequence |
|---|---|---|---|
| pre-open (t < t₀, strike not yet formed) | **5.88%** | **+0.70 ¢/share** | benign; strike is unformed, no information to be picked off on |
| in-window | 93.89% | ≈ +0.45 ¢/share avg | the modelled regime |

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** `+0.45 ¢/share` and the `+95 bps` maker gross are the SAME book-derived number and both fall together: `book` snapshots are p90 6.2 s stale. Rebuilt with no book at all the figure is **+0.17 ¢/share**, and it is **NOT DISTINGUISHABLE FROM ZERO** — window-clustered bootstrap gives **+0.173 [-0.251, +0.596]**, with all seven per-coin CIs spanning zero. **The maker-edge sign is UNDETERMINED at two days.** Also settled: `side` IS the taker's (G-FF1 `PASS`, 600/600, Wilson [0.9936, 1.0]), so the `+95 → −95` flip scenario is closed. See `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b.

| post-close (t > T, outcome determined, book live) | **0.22%** | **−1.46 ¢/share = −806 bps** | **8% of all maker gross, lost.** Median trade at T+21 s, p95 T+75 s, resolution T+85 s |

Post-close, a resting quote is a **free option with a known answer**. The module must
price `POST_CLOSE` fills at full information (permanent_AS = |ℓ − outcome|), which will
make the decision layer stand down. There is currently **no model of this anywhere in
the corpus** and it is the cheapest large win available.

---

## 2. Measuring markout conditional on fill

### 2.1 The identification

Every aggressive trade in the tape **is** somebody's passive fill. Population markout is
therefore measurable without ever quoting. For a maker at unified level ℓ filled by an
aggressor of sign ε (ε=+1 aggressive buy):

```
markout(h)  = −ε · ( m_{t+h} − ℓ )                     realised, ¢/share
spread      = −ε · ( m_{t⁻}  − ℓ )                     realised
AS(h)       = −ε · ( m_{t+h} − m_{t⁻} )  = −R(h)       realised
markout(h)  = spread + AS(h)                           exact, per fill
```

### 2.2 Units: ¢/share, never `/m`

`PRELIMINARY_PLANS` §2 inherits `es = q(p−m)/m` from the sibling equity programme. **The
`/m` normalisation is wrong for a bounded binary**: it makes a 1 ¢ move at p=0.05 look
20× a 1 ¢ move at p=0.95, when both pay the same dollar. The payoff is $1/share, so:

- **primary unit: ¢/share (absolute).** The MarkoutPartition sums in this unit.
- return-on-capital is a *derived* report with the correct denominator — capital at risk
  is `p` for a long Up and `1−p` for a short — never `m`.
- **notional-weighted, never equal-weighted** (equal-weighting flipped +2.44 bps to
  −0.32 bps on the sibling programme and reversed a conclusion).

### 2.3 Horizons

`h ∈ {0.25, 0.5, 1, 2, 5, 15, 30, 60, 120 s} ∪ {resolution}`, each **censored at
`T − τ`** and reported with the censoring flag. Pooling censored and uncensored h is a
real bias in a 300 s instrument.

**`h = resolution` is the primitive.** For a binary, `m_{resolution} ∈ {0,1}` is
observed in `resolutions.jsonl` (1,648 of 1,676 slugs resolve). This gives a
**mid-convention-free, model-free** terminal markout — an advantage no equity markout
study has. It also yields an exact decomposition with no fitted parameter:

```
permanent_AS = −ε · ( 1{Up} − m_{t⁻} )      the move that stuck   (uses NO mid at t+h)
transient_AS = −ε · ( m_{t+h} − 1{Up} )     the part that came back
               permanent_AS + transient_AS = AS(h)   exactly, per fill
```

### 2.4 The mid convention, and the contamination fix

The whole edge is sub-tick — theoretical gross capture at p=0.5 is 0.85 ¢/share
(0.50 half-spread + 0.35 rebate) against a 1 ¢ grid — so mid choice is material, and
on a 1-tick book the mid is quantised at exactly the scale of the answer.

**Frozen-lag mid.** `m_{t⁻}` is the reconstructed book state as of `recv_ns(trade) − Δ`
with **Δ = 250 ms frozen**, plus a sensitivity ladder `Δ ∈ {0, 100, 250, 500, 1000} ms`.
This is not fastidiousness: `price_change` carries **post-change** `best_bid`/`best_ask`
(99.67%) and can be emitted *before* the `last_trade_price` for the same match, which is
precisely why the corpus's spread/AS split was withheld as untrustworthy while its `net`
column (which never touches a mid) stood.

Four conventions computed in parallel — `SimpleMid`, `MicroPrice` (size-weighted),
`PairUnifiedMid`, `DepthWeighted(k)`. **R-MIDROBUST: a conclusion that does not hold
under all four is not a conclusion.** The resolution-horizon markout, which uses no mid,
is the tiebreak.

### 2.5 The known bias: population ≠ marginal

The tape's fill is the **average** maker's, over the whole queue that cleared. A marginal
entrant at the **back** does strictly worse, twice over: they fill only when the sweep is
large enough to clear everyone ahead (conditioning on larger, more informed flow), and
they fill later within the sweep, at a worse post-trade mid.

⇒ Population markout is an **upper bound**. Declared as
`NullPin{field: "markout_population", bias_direction: OPTIMISTIC}`.

**Three estimators that escape the bias**, all observable from level totals:

| estimator | identifies | observable? |
|---|---|---|
| **E1 Improvement events** — a new best level appearing inside the previous spread | a marginal maker with **Q_ahead = 0 exactly** | yes, directly. This is *our own contemplated improve action*, run by someone else |
| **E2 Join events** — `ΔS > 0` at an existing level | a marginal maker at **Q_ahead = S_pre** | yes, but fill-vs-cancel needs the trade/cancel attribution `α_trade` (E-M1) |
| **E3 Shadow quotes** — counterfactual, **non-cancellable** orders replayed at levels/times nobody quoted | unconditional coverage of the state space we would actually choose | yes, in replay |

E3 is the deleted `PM-E2.5`, claimed absorbed into E-M7(b)(c) and **not** absorbed. It is
also the correct estimator for the snipe leg: the corpus's cancel-before-trade statistic
has a **possibly inverted sign** — cancel-before-trade → 1 is equally consistent with
"the slow maker absorbs the entire race loss", the opposite of the decision attached to
it. *The last quote standing is the slow one.* A non-cancellable shadow quote cannot
have that ambiguity.

**E1 is the highest-value of the three** because it is exactly the marginal action, it is
bias-free on queue position, and it needs no attribution model. Its residual bias is
**self-selection** (makers improve when they want to) — which E3 is designed to remove.

---

## 3. The queue bracket

No L3. Queue position is **DECLARED, never estimated**. Three constructions from
`price_change` level totals (96.10% replay fidelity):

```
PESSIMISTIC  Back:    fill iff cumulative aggressive volume at ℓ since our arrival
                      ≥ S_arrival + our_size          (the level must clear)
OPTIMISTIC   Front:   fill iff any aggressive volume at ℓ ≥ our_size
INTERMEDIATE Uniform: fill iff cumulative ≥ U·S_arrival + our_size,  U ~ Unif(0,1)
                      — a point estimate, NEVER a substitute for the bracket
```

### 3.1 The bracket is the dominant uncertainty, and it is wide

Indicative (last-full-snapshot level totals; **must be redone with `price_change`
replay**), 25 windows/coin on 20260820:

| coin | top-of-book median (sh) | trade p50 (sh) | **P(a trade clears its level)** |
|---|---|---|---|
| btc | 141 | 8.9 | **0.095** |
| eth | 43 | 5.3 | 0.129 |
| sol | 26 | 5.0 | 0.127 |
| xrp | 45 | 1.9 | 0.089 |
| doge | 39 | 0.02 | 0.043 |
| bnb | 27 | 0.02 | 0.040 |
| hype | 30 | 0.02 | 0.016 |

**~90% of trades do not clear their level.** Front-of-queue fills on nearly every trade;
back-of-queue fills on ~1 in 10. **The bracket is roughly an order of magnitude wide** —
larger than any plausible refinement of ζ or λ.

Two structural consequences:

1. **The bracket cannot be escaped by improving.** The modal spread is **1 tick (66.7%
   of book states; 79% on BTC)**, so there is usually **no room** to price-improve. The
   join/queue problem is the problem. Price improvement is available ~1/3 of the time,
   concentrated in the illiquid coins (hype 94%, doge 88%) where the flow is thinnest.
2. **This, not λ, is where the effort goes.** The corpus's own instruction — *"put the
   uncertainty bracket on `Q_ahead`, not on λ"* — is now quantified.

### 3.2 The refusal rule (R-BRACKET)

Every reported quantity carries `[pessimistic, optimistic]`.

> **If the SIGN of the net edge differs between the bracket ends, the module returns
> `Unavailable(reason="queue_bracket_sign_flip")` for that action. The midpoint is not a
> result.**

Same refusal for a sign that differs across the four mid conventions (R-MIDROBUST), or
between day-clustered and window-clustered inference. The decision layer's
`UnavailablePolicy` then decides (expected: `RefuseAction`).

This is deliberately a **refusal, not a wide interval**: a risk-neutral
`UtilityFunctional` would average a wide interval and quote anyway.

### 3.3 Prerequisites that can invalidate the bracket

Both are pre-registered in `E-M1` and neither has been read:

- **`α_trade` (trade-vs-cancel attribution, Δt = 250 ms frozen).** `α_trade ≥ 0.60` ⇒ the
  pessimistic rule is implementable. **`α_trade < 0.30` ⇒ queue-ahead is unknowable, and
  every fill-rate statement — including all of §3 — is vacuous;** fill modelling is
  replaced by a regression of fill on level and displayed depth.
- **Touch half-life.** If the touch half-life is **< 250 ms**, "queue position" is not a
  stable state variable at our latency and the default action moves from *join best* to
  *rest 1–2 back* regardless of what ζ says.

---

## 4. EXP-IMPACT — the ζ-sign experiment

The single highest-value unknown: does resting into a sweep get **paid** (transient
impact, reverting) or **run over** (permanent, informed)? Extends the unrun `E-X1`.

### 4.1 Event construction

1. **Fold to Up-space.** Trades on Down at price q become unified price 1−q with flipped
   sign. Without this the flow at any level is halved.
2. **Dedup on `transaction_hash`** — hygiene, ~0 duplicates observed; keep it as an
   assertion. **Fail loud if multiplicity ever exceeds 1**, because the corpus's design
   assumed 2× and a silent change of venue behaviour would double every intensity.
3. **Sweep, not print.** A sweep = maximal run of prints within a **250 ms frozen**
   knowledge-time burst, same instrument, same ε. Counting prints instead of sweeps
   double-counts intensity — a 10-print sweep is one event. Report both to size the
   effect.
4. **Aggressor sign ε — book-derived at the frozen lag.** `ε=+1` if the unified trade
   price ≥ frozen-lag best ask, `−1` if ≤ best bid, else `Ambiguous` (its own stratum, no
   Lee-Ready imputation; report its mass).
   - Cross-tabulate against `(side × asset_id)` and report the agreement rate. `side`'s
     77% BUY skew being **symmetric across both tokens** proves it is a per-token taker
     convention; the corpus's circumstantial check (63.7% of BTC BUY prints at the best
     ask vs 15.1% at the best bid) supports *taker*.
   - **This is load-bearing to the point of programme survival:** if `side` is the
     *maker's* side, maker gross flips from **+95 bps to −95 bps** and the programme is

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** `+0.45 ¢/share` and the `+95 bps` maker gross are the SAME book-derived number and both fall together: `book` snapshots are p90 6.2 s stale. Rebuilt with no book at all the figure is **+0.17 ¢/share**, and it is **NOT DISTINGUISHABLE FROM ZERO** — window-clustered bootstrap gives **+0.173 [-0.251, +0.596]**, with all seven per-coin CIs spanning zero. **The maker-edge sign is UNDETERMINED at two days.** Also settled: `side` IS the taker's (G-FF1 `PASS`, 600/600, Wilson [0.9936, 1.0]), so the `+95 → −95` flip scenario is closed. See `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b.

     dead on arrival. Resolve it on-chain (`OrderFilled` carries maker and taker
     addresses) on the frozen ≥500-tx sample before any ζ number is read.

### 4.2 The response function

```
R(h) = E[ ε · ( m_{t+h} − m_{t−250ms} ) ]        ¢/share, notional-weighted
ζ(h) = −R(h)
```

Fit the decay `G(h) ~ h^{−β}` after the peak. Report `peak_horizon`, β, and
`R(resolution)`.

### 4.3 The placebo (mandatory)

In a 300 s binary, `p_t` runs mechanically to 0 or 1 near expiry, and the sample carries a
**+4.5 pp up-drift on both days** (realised Up 54.1%; a 2-day rally). Either can
manufacture a response with no adverse selection in it.

> **Estimand = `R(h) − R_placebo(h)`**, where placebo events are pseudo-events matched on
> (coin, r-bucket, moneyness bucket) at random times with ε drawn from the local marginal
> sign distribution.

Second guard: report R separately for ε=+1 and ε=−1. **Asymmetry in ε is a drift
signature, not adverse selection.**

### 4.4 Frozen strata

coin · `r ∈ {pre-open, 0–60, 60–180, 180–270, 270–300, post-close}` ·
`|m−0.5| ∈ {[0,.05),[.05,.15),[.15,.25),[.25,.35),[.35,.45),[.45,.5)}` ·
sweep-size quintile · cleared-level indicator · ε sign · tick regime {0.01, 0.001}.

Frozen before reading, `strata_hash` recorded. The last one matters: **567 of 766 windows
change tick size**, and a level grid that changes by 10× fragments level totals and resets
queue positions.

### 4.5 The mechanism split (the part that decides whether a defence exists)

The market settles on the **TWAP60 stream, which PM publishes 1,440 ms late**. An informed
taker sourcing spot directly holds a ~1.25 s option against every resting quote. So
classify each sweep:

- **(a) underlying-driven** — spot moved ≥ θ over `[t−1.5 s, t]` on our own `crypto_prices`
  feed (448 ms p50, ≤1 s of which is payload truncation);
- **(b) not underlying-driven.**

If ζ concentrates in (a), a **spot-conditioned widen/pull is a real defence** and we can
run it on the 448 ms leg, not the 1,700 ms one. If ζ is uniform across (a)/(b), it is flow
toxicity and there is no defence — which is the corpus's current (TWAP-leg-derived)
conclusion, reached without this test.

**Open prerequisite:** the spot mirror's payload timestamps are second-truncated, so its
true transport leg is unresolved. Settle it (sub-second source or a paired comparison)
before claiming a defence exists.

### 4.6 Inference

- **Primary: block bootstrap clustered by DAY.** With n=2 days this is **uninformative
  and must be reported as such**, with a pre-committed re-read at 7 and 30 days.
- Secondary: window-clustered. **Never headline the window-clustered number** — 1,477
  windows ride a handful of underlying paths and effective breadth is ≈1.4 coins, not 7.
- Holm across the h-grid. Notional-weighted throughout.
- Report **quantiles of markout | fill**, not just the mean: **49% of BTC maker gross came
  from 5 of 171 windows**; per-window p10 −392 bps / p90 +597 bps; per-coin gross flips
  sign (sol −231, bnb −86, btc +98, eth +138 bps). A mean over that tail is not a
  decision statistic.

### 4.7 Decision rule (pre-registered)

| outcome | reading | action |
|---|---|---|
| R rises then reverts, β > 0, `R(resolution)` CI **includes** 0 | **transient** — resting into sweeps is **paid**; ζ gains a favourable component | MM is viable in that stratum |
| `R(resolution)` CI **excludes** 0 with ε's sign | **permanent/informed** — ζ is strictly a cost | compare \|ζ\| against gross capture (half-spread + rebate) per stratum; exclude strata where ζ exceeds it |
| sign differs across the **bracket**, the **four mids**, or **day- vs window-clustering** | **not a result** | `Unavailable` for that stratum |

### 4.8 Negative controls (mandatory, from E-M1)

Never-quote ⇒ exactly zero fills. A 0.99 bid ⇒ fill on every downward print. `R(h)` on
randomly permuted ε ⇒ ≈ 0. A harness that fails any of these is reporting its own bugs.

---

## 5. Own impact

Our size is tape-scale: `orderMinSize` 5 vs trade p50 5.81 shares pooled. "We don't move
the market" is an **assumption to test**. Scale of exposure differs sharply by coin — 5
shares is 3.5% of the BTC touch (141 sh) but 19% of the SOL touch (26 sh), and
`rewardsMinSize` 50 exceeds every non-BTC touch outright.

**It is not identifiable from a tape containing none of our orders.** Until we quote it is
`NullPin{field: "own_impact", assumption: "0", bias_direction: OPTIMISTIC}`. Do not fit it;
do not silently pin it to 0 without the pin.

**Measurable now — third-party natural experiment.** Level totals let us observe *other*
participants adding 5–50 shares. Matched on (coin, r-bucket, moneyness, spread, existing
depth), estimate the effect of `ΔS` at level ℓ on:
1. subsequent aggressive arrival intensity at ℓ (crowding-in vs flow-avoidance),
2. whether other makers step ahead (our queue position degrades),
3. the markout of the *adding* maker vs the pre-existing queue.

Channel (3) on **improvement** events is the E1 estimator of §2.5 — the same measurement
serves both.

**When live — randomised size, per window.** Randomise between `s` and `2s` (and
quote/no-quote) per window, pre-registered. Randomisation is essential: size is otherwise
endogenous (we quote bigger when we like the state). ~190 paired units per coin per day.

**Double-count guard.** The channels are disjoint by construction:
`queue position → the fill process (N_a, τ_a)`; `price reaction to our presence →
own_impact in the partition`. Never both.

---

## 6. Flow-modelling depth — recommendation

**Recommendation: no Hawkes. A two-timescale EWMA intensity conditioned on book state and
the window clock, per coin, fitted walk-forward.**

```
λ_ℓ(t) = λ₀( r_bucket, |m−0.5|, coin )            ← the deterministic window clock
         · f( distance from touch, ticks )
         · g( EWMA_fast / EWMA_slow )              ← half-lives ≈ 5 s and 60 s
         · h( level total, spread ticks, unified imbalance )
```

Poisson/GLM fit, walk-forward by day, per coin, on **dedup'd, complement-folded sweeps**.

Four reasons, in order of force:

1. **The bracket dominates.** Queue uncertainty is ~10× (§3.1). No refinement of λ can
   change a decision until the refinement exceeds the bracket width. Effort spent on λ
   before `Q_ahead` is effort that cannot pay.
2. **The clock is the main effect, and a Hawkes would eat it.** In a 300 s binary,
   intensity is a strong deterministic function of `r` and `|m−0.5|`. A Hawkes with a
   constant baseline attributes clock-driven intensity growth to self-excitation — a
   specification error that *looks like* clustering and would be adopted on its
   in-sample fit.
3. **Identifiability is coin-split, and per-coin is mandated.** BTC has 2,424 trades/window
   (p50) — a kernel is estimable in-sample. doge/bnb/hype have 108–133 — it is not. A
   model we can only fit on 1 of 7 coins is not the module's model.
4. **The decision horizon integrates bursts out.** A quote rests for seconds to minutes;
   `P(fill)` is an *integral* of intensity. Burst microstructure is exactly what
   integration removes.

**Pre-registered upgrade trigger** (BTC only, ≥7 days) — adopt Hawkes iff **all three**:
(i) branching ratio η CI excludes 0 **after** time-changing by the fitted clock baseline;
(ii) day-held-out log-likelihood beats the EWMA model; and
(iii) **the resulting `P(fill)` differs from the EWMA `P(fill)` by more than the queue
bracket width.** (iii) is the binding one, and it is the honest form of "recommend, don't
hedge": the upgrade is not rejected forever, it is rejected until it can matter.

---

## 7. No double-counted terms

This programme has made this error at least three times (`v(t)` sum-vs-min; running-vs-
terminal penalty; σ_⊥ on top of κ, which inflated σ_eff by 32–41% ⇒ 6.5–8 ¢ of p̂ error on
a 2–4 ¢ book), plus twice more in the PnL plane. Explicit rules for this module:

| # | rule | the error it blocks |
|---|---|---|
| **R-GROSS** | `MarkoutPartition` is **gross of all fees, rebates and incentives**. Every cash term crosses into `WealthLedger` exactly once via `ActionOutcome.cash_flows` | measuring ζ net of the rebate and then adding the rebate again in the ledger |
| **spread once** | the `spread` component lives only in the partition; DE may not add a half-spread term of its own | `adverse_selection` as an objective term *beside* a belief returning `E[markout\|fill]` — already live in the corpus |
| **partition exactness** | per fill: `spread + transient_AS + permanent_AS + snipe + own_impact = markout(τ)`, using the **realised** decomposition of §2.3 (no fitted split) | a "residual" term absorbing model error |
| **snipe is a partition, not an overlay** | `snipe := 1{τ < cancellable_after} · (permanent_AS + transient_AS)`, and those two are then **zero** for such fills | counting the same adverse selection as both AS and snipe |
| **own_impact disjoint** | queue-position effects → the fill process; price-reaction-to-our-presence → `own_impact` | fitting both against the same variation |
| **fees not owned here** | `f(p)` and `ρ(p)` belong to **SP-Venue**; this module consumes them. **Never read `fee_rate_bps`** — it is `"0"` on all 446,412 events | a second fee schedule drifting from the first |
| **merge books no PnL** | Up+Down = $1 either way; merging frees collateral. It is a capital decision, owned by `DE-Allocator` | a "merge PnL" event double-counting the lock |
| **ζ is conditional** | ζ enters only as `E[markout \| fill]` inside the joint law | `λ × unconditional ζ` (§1.1) |

**Unresolved and blocking a net number.** The taker fee is verified on-chain to the cent
at `94.38 × 0.07 × 0.25 = $1.651650` vs 1,651,650 µUSDC ⇒ **`f(p) = 0.07·p(1−p)` $/share
= 1.75 ¢/share at p=0.5**. The corpus's Q5 restates it as `0.07·min(p,1−p)` = 3.5 ¢/share
— a recorded **2× contradiction (§11 vs §12.1) that the on-chain arithmetic resolves
against Q5, and that no document has updated.** Since the maker rebate is 20% of the taker
fee, this 2× propagates straight into the rebate and therefore into gross capture. Under
**R-PROV**, an `ASSUMED`-provenance parameter may not gate a decision: **BE-FlowAndFills
returns `Unavailable` for any net-of-fee estimand while `FeeSchedule` is `Disputed`.**

And the rebate itself is unverified: **never observed being paid**, `rewards_authoritative`
is `null` on all 812 rows that carry the field, `rewards_registry.jsonl` is only a
heartbeat of registry size, and 5-min crypto markets are **absent from the CLOB rewards
registry** (33 pages / 16,172 rows exhausted). At p=0.5 the rebate is 0.35 of the 0.85
¢/share theoretical gross — **41% of the thesis rests on an unobserved payment** whose
verification arm (`E-M4`) reads 2026-09-03.

---

## 8. Power — what is claimable when

Sample: **2 days**; 20260819 = 115 windows/coin (btc/eth/sol/xrp) and 102 (doge/bnb/hype);
20260820 = 129/coin partial. 1,648 resolved of 1,676 slugs. **One walk-forward test day.**
446,412 trades and 938,862 book events on 20260819. Effective breadth ≈ **1.4 coins**.

**Claimable now** — deterministic/structural quantities have full power immediately
because they are censuses, not estimates:

- hash multiplicity; `side`×token symmetry; complement-fold correctness; `price_change`
  replay fidelity (96.10%); spread and depth distributions per coin; tick-regime
  frequency; window-phase trade shares; the **bracket width** and `P(level cleared)`;
  counts of improvement and join events per coin.
- The **sign** of `R(h) − R_placebo(h)` pooled on BTC, window-clustered, **explicitly
  labelled as lacking a day-level CI**.
- `α_trade`, touch half-life, and the `side`-is-taker on-chain confirmation — all
  gating, all readable now.

**Not claimable now:** any day-clustered CI (n=2); per-coin ζ on the alts; any
net-of-fee edge (fees `Disputed`, rebate unobserved); any walk-forward-validated fitted
parameter (1 test day = 1 observation); anything about the tail (49% of BTC maker gross
from 5 of 171 windows).

**At 7 days** (~1,100 windows/coin, 6 test days): day-clustered CI becomes computable but
remains weak — the corpus's own read is that a quantity with this tail needs **tens of
day-clusters**. Claimable: ζ **sign** on BTC at coarse strata; the improve-vs-join bracket
comparison; the EWMA intensity model's walk-forward fit; the Hawkes upgrade test.

**At 30 days:** per-coin ζ by r-bucket and moneyness; day-clustered block-bootstrap net
edge (conditional on the fee dispute being resolved and the rebate observed); the
FLB × fill-quality interaction; a tail-aware gross that is a mean rather than an anecdote.

**Process.** `PM_PREREG.md` does not exist while the design-freeze date is 2026-08-21.
Every experiment above must be pre-registered with frozen strata and a named read date
before any number is read.

---

## 9. Ways this design could be wrong

1. **`side` is the maker's side.** The 63.7%-at-ask evidence is circumstantial. If it is
   the maker's, maker gross flips +95 → −95 bps and everything here is inverted. *Blocking;
   settle on-chain first.*
2. **`α_trade < 0.30`** ⇒ queue-ahead unknowable ⇒ §3 is vacuous and the fill model
   degrades to a regression of fill on level and displayed depth. *Blocking.*
3. **Mid contamination outlives the frozen lag.** If `price_change`/`last_trade_price`
   emission order is worse than 250 ms, the spread/AS split stays unusable (the `net`
   column survives regardless).
4. **`size` semantics.** p50 = **0.02 shares** on bnb/doge/hype, below `orderMinSize` = 5 —
   impossible for a resting order. Either the field is not shares in all contexts, or these
   are sub-minimum residuals. If it is not shares, every `Q/S` and `P(clear)` number above
   is wrong.
5. **Matching may not be FIFO.** The pessimistic construction assumes price-time. Price-time
   is "confirmed (M lens)" but batching and latency are unverified; any batching signature
   ⇒ queue position stops mattering and the bracket is misspecified in an unknown direction.
6. **E1/E2 self-selection.** Makers improve and join when *they* want to; their markout is
   not ours. Only E3 (shadow quotes) gives the unconditional, and E3 has no realism check.
7. **Two days, one direction.** A +4.5 pp up-drift across both days; the placebo and the
   ε-symmetry split are the guards, and if R is asymmetric in ε the estimate is contaminated.
8. **Tail dominance.** 49% of BTC maker gross from 5 of 171 windows. A mean ζ is not the
   decision statistic; the quantiles may say the opposite.
9. **Post-close is 8% of maker gross and unowned.** If the module does not carry the window
   phase, the fill model is optimistic by roughly that amount.
10. **The rebate may not exist** (§7) — 41% of theoretical gross at p=0.5.
11. **Tick regime.** 0.01 → 0.001 in 567 of 766 windows, never reverting in-sample. Far from
    the money the moat is the **tick**, not the fee, so the tick regime is a first-order
    economic parameter and all level-based fits are regime-specific.
12. **Cancel feasibility has an unmeasured leg.** Venue ack is not observed; `CancelAllStatus`
    already encodes this. If the cancel round trip is dominated by the unmeasured leg rather
    than the 47 ms market-data leg, §4.5's "spot-conditioned defence" evaporates.
13. **Coverage gaps.** 43,835 gaps > 1 s and 12,546 > 5 s on 20260819, max 74.1 s. A fill
    model assuming continuous observation mis-states both intensity and queue state; use
    `StateView.coverage` and drop, never interpolate.
14. **Pooling across coins.** Effective breadth ≈ 1.4; a pooled N of 7 × windows is fiction.
15. **Selection at depth.** ζ measured at a deep level is conditional on a sweep having
    *reached* it — the rare, large, informed case. The decision-relevant quantity is
    `P(reach) × ζ|reached`; reporting `ζ|reached` alone will make deep levels look worse than
    they are for a selection reason, not an economic one.

---

## 10. Build order

1. **Complement-folded, dedup'd, sweep-aggregated unified tape** + `price_change` book replay
   with the 96.1% parity assertion. Everything below is a query on it.
2. **Gating reads, all cheap, all blocking:** `side`-is-taker (on-chain), `α_trade`, touch
   half-life, `size` semantics.
3. **Bracket census** — `P(level cleared)`, improvement/join event counts, per coin, per
   phase. Produces the bracket width that scopes everything else.
4. **EXP-IMPACT** (§4) — ζ sign, placebo-differenced, with the underlying-driven split.
5. **E1/E2/E3 marginal-maker markout** — the estimator that replaces the population upper
   bound.
6. **EWMA intensity** (§6), walk-forward, per coin.
7. Assemble the joint law; emit `ScenarioDrawSet` under `SharedDraws`.

Steps 1–5 run entirely on data already on disk. None needs the decision layer.

**Runtime path note (architectural).** Dependencies point SP ← DA ← BE ← DE, and **EV is
read by nobody** — so `BE-FlowAndFills` cannot consume `EV-Markout` at runtime. The legal
path is: the ζ / response-function / intensity fits are **offline research artifacts**
frozen with `artifact_id` + `fit_data_through`, resolved at runtime through the
`artifact_resolver` port. `EV-Markout` independently re-measures the same estimands on
realised fills as a calibration gate, and that measurement never feeds a decision.

---

## 11. Canonical contract — `BE-FlowAndFills`

Proposed addition to `contracts/contracts.yaml` (v12 → v13). **Not applied here.**
Owner-qualified fields with types, in the style the structural diff consumes. All entries
are ADDITIONS except the two flagged `MIGRATION`, which require records in
`migrations.yaml`.

Checked against v12: **0 type / rule / module name collisions, 0 dangling type references**
— and note that `BE-FlowAndFills` has **no `modules:` entry in the canonical source at
all** today, despite being specified in `PM_ARCHITECTURE.md` §5. The markdown explains; it
does not define. That gap is why the module could run on three undeclared optimistic pins.

```yaml
rules:
  R-GROSS:
    body: MarkoutPartition is GROSS of fees, rebates and incentives; every cash term crosses
      into WealthLedger exactly once via ActionOutcome.cash_flows
    checks:
    - 'no_fee_in_markout: no MarkoutPartition component carries a fee or rebate term'
    - 'snipe_disjoint: snipe != 0 => transient_AS == 0 AND permanent_AS == 0'
  R-BRACKET:
    body: queue position is DECLARED, never estimated; a net-edge sign that differs across the
      bracket ends returns Unavailable, never a midpoint
    checks:
    - 'sign_flip: QueueBracket.sign_agreement == false => Unavailable'
  R-MIDROBUST:
    body: any estimand measured against a mid must hold under every MidConvention variant; the
      resolution-horizon markout, which uses no mid, is the tiebreak
    checks:
    - 'mid_disagree: ResponseFunction.mid_agreement == false => Unavailable'
  R-FOLD:
    body: flow estimands are defined on the complement-folded unified tape; trades are reported
      on ONE asset_id, so an unfolded intensity is half the truth
    checks:
    - 'dedup_assert: transaction_hash multiplicity > 1 => FAIL LOUD (measured 1.000)'

types:
  UnifiedLevel:
    fields:
      instrument: InstrumentId
      price_up: float
      tick: float
    notes: 'canonical Up-space coordinate. price_changes[] is length 2 in 100.00% of 28.7M
      messages (mirrored, side flipped, price complementary), so the pair book is ONE book as a
      matter of fact. Down@q IS Up@(1-q): the same queue, crossed by CTF mint. A per-token fill
      model is structurally wrong.'
  LevelTotals:
    fields:
      total_size: float
      source: enum:SNAPSHOT|REPLAYED
      replay_parity: float
      tick_regime: float
    notes: 'price_change.size is the level TOTAL, not a delta - replay reproduces the next
      snapshot exactly in 96.10% of 937,286 checks. size "0" is level removal. tick_regime
      switches 0.01<->0.001 in 567 of 766 windows, fragmenting the grid 10x.'
  PairBookState:
    fields:
      levels: dict[UnifiedLevel, LevelTotals]
      best_bid: float
      best_ask: float
      spread_ticks: int
      mid_convention: MidConvention
      one_sided: bool
      coverage: Coverage
      t_known: Timestamp
    notes: 'modal spread is 1 tick (66.7%); P(1 tick) ranges btc 0.79 to hype 0.06. ~1% of book
      events are one-sided and must be excluded before any mid.'
  MidConvention:
    variants:
    - SimpleMid
    - MicroPrice
    - PairUnifiedMid
    - 'DepthWeighted(k_shares: float)'
    notes: 'gross capture at p=0.5 is 0.85 c/share on a 1 c grid, so the mid is quantised at the
      scale of the answer. All four are computed; R-MIDROBUST governs.'
  FrozenLagMid:
    fields:
      lag: Duration
      sensitivity_ladder: list[Duration]
      convention: MidConvention
      value: float
    notes: 'pre-trade mid taken at recv_ns(trade) - lag, lag = 250 ms FROZEN. price_change
      carries POST-change best_bid/best_ask (99.67% of 387,022 checks) and can precede the
      last_trade_price of the same match - the contamination that made the corpus withhold its
      spread/AS split while its net column stood.'
  AggressorSign:
    variants:
    - 'BookDerived(sign: int, frozen_lag: Duration)'
    - 'Ambiguous(reason: str)'
    notes: 'trade `side` is NOT unified direction. Its BUY skew (77.18% pooled; btc 87%, hype
      12%) is SYMMETRIC across both tokens (Up 77% / Down 77%), which proves it is a per-token
      taker convention. Direction is (side x asset_id) AND is validated against the frozen-lag
      book. If `side` is the MAKER side, maker gross flips +95 -> -95 bps: settle on-chain
      (OrderFilled carries maker and taker) before any zeta is read.'
  SweepEvent:
    fields:
      instrument: InstrumentId
      unified_price: float
      size: float
      aggressor: AggressorSign
      dedup_key: str
      n_prints: int
      burst_window: Duration
      cleared_level: bool
      phase: WindowPhase
      underlying_driven: bool
      t_known: Timestamp
    notes: 'ONE match, not one print: a burst (250 ms frozen) is one sweep, and counting prints
      double-counts intensity. dedup_key is transaction_hash - multiplicity measured at exactly
      1.000 over 446,412 events, refuting the double-report premise; the real operation is
      FOLDING the complement token (R-FOLD).'
  WindowPhase:
    variants:
    - PRE_OPEN
    - IN_WINDOW
    - POST_CLOSE
    - RESOLVED
    notes: 'measured maker markout by phase: PRE_OPEN +0.70 c/share (5.88% of trades, strike
      unformed), IN_WINDOW ~+0.45, POST_CLOSE -1.46 c/share = -806 bps on 0.22% of trades = 8%

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** `+0.45 ¢/share` and the `+95 bps` maker gross are the SAME book-derived number and both fall together: `book` snapshots are p90 6.2 s stale. Rebuilt with no book at all the figure is **+0.17 ¢/share**, and it is **NOT DISTINGUISHABLE FROM ZERO** — window-clustered bootstrap gives **+0.173 [-0.251, +0.596]**, with all seven per-coin CIs spanning zero. **The maker-edge sign is UNDETERMINED at two days.** Also settled: `side` IS the taker's (G-FF1 `PASS`, 600/600, Wilson [0.9936, 1.0]), so the `+95 → −95` flip scenario is closed. See `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b.

      of ALL maker gross. A POST_CLOSE quote is a free option with a known answer; it is priced
      at full information, not modelled as adverse selection.'
  QueuePosition:
    variants:
    - Front(DECLARED)
    - Back(DECLARED)
    - Uniform(DECLARED)
    notes: 'L3 is never observable on this venue; position is DECLARED. Uniform is a point
      estimate and NEVER a substitute for the bracket.'
  QueueBracket:
    fields:
      pessimistic: ActionOutcome
      optimistic: ActionOutcome
      sign_agreement: bool
      width_ratio: float
      alpha_trade: Known[float] | Unavailable
      touch_half_life: Known[Duration] | Unavailable
    notes: 'Back fills only when the level clears; Front fills on first arrival >= our size.
      P(a trade clears its level) is 0.016-0.129 by coin (btc 0.095), so the bracket is ~10x -
      wider than any plausible refinement of zeta or lambda, and it cannot be escaped by
      improving because the modal spread is 1 tick. alpha_trade < 0.30 => queue-ahead is
      unknowable and every fill-rate statement is vacuous. touch_half_life < 250 ms => queue
      position is not a stable state variable at our latency.'
  MarginalMakerEstimator:
    variants:
    - ImprovementEvent(DECLARED)
    - JoinEvent(DECLARED)
    - ShadowQuote(DECLARED)
    notes: 'population markout is the AVERAGE maker''s and is an UPPER BOUND: a back-of-queue
      entrant fills only when the sweep clears everyone ahead, i.e. conditional on larger and
      more informed flow. ImprovementEvent identifies a marginal maker at Q_ahead = 0 exactly
      and is our own contemplated action run by someone else. ShadowQuote is non-cancellable by
      construction, which is why it, and not cancel-before-trade, measures the snipe leg -
      cancel-before-trade -> 1 is equally consistent with the slow maker absorbing the whole
      race loss.'
  CancelFeasibility:
    fields:
      market_data_lag: Known[Duration]
      signal_lag: Known[Duration]
      venue_ack_lag: Known[Duration] | NullPin
      cancellable_after: Timestamp
      uncancellable_fill_frac: float
    notes: 'THREE clocks, not one: market data 47-58 ms p50 (floor 46), spot mirror 448 ms
      (<=1 s is payload truncation), Chainlink->PM TWAP60 1,700 ms of which 1,440 is PM-side
      publication. The 1.7 s figure is the SIGNAL leg; applying it to book-driven cancels
      mis-scopes the defence. venue_ack_lag is unobserved (see CancelAllStatus) and is a
      NullPin. Latency params are owned by SP-Params keyed by (ParamId, ScopeKey) - consumed
      here, never restated.'
  ResponseFunction:
    fields:
      horizons: list[Duration]
      R: dict[Duration, float]
      R_placebo: dict[Duration, float]
      beta: float
      peak_horizon: Duration
      permanent: float
      transient: float
      quantiles: dict[float, float]
      censoring: enum:UNCENSORED|EXPIRY_CENSORED
      cluster_unit: enum:WINDOW|DAY
      ci: dict[Duration, list[float]]
      mid_agreement: bool
      unit: str
    notes: 'R(h) = E[eps*(m_{t+h} - m_{t-250ms})] in c/share, notional-weighted, on
      complement-folded dedup-ed SWEEPS. Estimand is R - R_placebo: a 300 s binary runs
      mechanically to 0/1 and the sample carries +4.5 pp of up-drift, either of which
      manufactures a response containing no adverse selection. h = resolution is the primitive
      (m in {0,1}, no mid needed) and gives an exact model-free split: permanent = -eps*(1{Up}
      - m_t-), transient = -eps*(m_{t+h} - 1{Up}). Quantiles are required, not optional: 49% of
      BTC maker gross came from 5 of 171 windows.'
  AdverseSelectionCurve:
    fields:
      by_level: dict[UnifiedLevel, ResponseFunction]
      reach_probability: dict[UnifiedLevel, float]
      by_stratum: dict[str, ResponseFunction]
      strata_hash: Hash
      selection_warning: NullPin
      artifact_id: ImmutableId
      fit_data_through: Timestamp
      scope: ScopeKey
    notes: 'zeta measured on fills that HAPPENED is conditional on a sweep REACHING the level.
      The decision-relevant quantity is P(reach) x zeta|reached; reporting zeta|reached alone
      makes deep levels look bad for a selection reason, not an economic one.'
  ArrivalIntensity:
    fields:
      model: PluginRef
      lambda_by_level: dict[UnifiedLevel, float]
      clock_baseline: list[str]
      ewma_half_lives: list[Duration]
      artifact_id: ImmutableId
      fit_data_through: Timestamp
      scope: ScopeKey
    notes: 'two-timescale EWMA conditioned on book state AND the window clock lambda_0(r,
      |m-0.5|, coin), per coin, walk-forward. NOT Hawkes: the clock is the main effect and a
      constant-baseline Hawkes would absorb it as self-excitation; kernels are identifiable on
      btc (2,424 trades/window p50) and not on doge/bnb/hype (108-133); and P(fill) is an
      INTEGRAL of intensity over a multi-second rest, which is exactly what removes burst
      structure. Upgrade gated on branching ratio after time-change AND day-held-out
      log-likelihood AND a P(fill) shift exceeding QueueBracket.width_ratio.'
  OwnImpactModel:
    variants:
    - NotMeasured(NullPin)
    - 'ThirdPartyProxy(artifact_id: ImmutableId)'
    - 'RandomizedLive(artifact_id: ImmutableId, randomization_id: ImmutableId)'
    notes: 'our size is tape-scale - orderMinSize 5 vs trade p50 5.81 shares pooled; 5 shares is
      3.5% of the btc touch (141 sh) but 19% of the sol touch (26 sh). Not identifiable from a
      tape containing none of our orders, so until we quote it is a NullPin with OPTIMISTIC
      bias. Channels are disjoint by construction: queue position enters the fill process,
      price-reaction-to-our-presence enters own_impact - never both.'
  FillOutcomeLaw:
    fields:
      draws: ScenarioDrawSet[ActionOutcome]
      common_random_id: CommonRandomId
      dependence: Dependence
      queue: QueuePosition
      bracket: QueueBracket
      horizon_effective: Duration
      partition_exact: bool
    notes: 'the estimand is the JOINT law of (fill quantity, first-fill time, markout), not
      P(fill) times E[markout]. lambda_fill(l) * [P_l - CE - zeta(l)] is REJECTED: it multiplies
      a marginal fill rate by an UNCONDITIONAL zeta, and the measured capture-ratio haircuts
      (60% at t=290 s / mid 0.95-1.00; 97% at t=30 s / mid 0.15-0.30) show selection is the
      dominant term. Draws are SHARED across the whole ActionSet so join-vs-improve rides one
      flow path and the second leg is priced at post-first-fill q. Independent(DECLARED) is
      REFUSED; unwrap_policy must be Draws(n, SharedDraws) - Expectation destroys the
      dependence and the module refuses that wiring. horizon_effective = min(horizon, T - t).'

modules:
  BE-FlowAndFills:
    implements: OutcomeModel
    consumes:
    - PairBookState
    - SweepEvent
    - ArrivalIntensity
    - AdverseSelectionCurve
    - CancelFeasibility
    - OwnImpactModel
    - SelfState
    produces: Uncertain[ActionOutcome] | Unavailable
    protocol:
      evaluate: (ActionSet, StateView, SelfState, Duration) -> Uncertain[ActionOutcome] | Unavailable
    requires:
      dependence: SharedDraws
      queue: QueuePosition
      unwrap_policy: Draws(n, SharedDraws)
    ports:
    - state_view
    - rng
    - artifact_resolver
    - telemetry_out
    stateful: false
    null_semantics:
      zeta: 'NullPin{field: zeta, assumption: "0", bias_direction: OPTIMISTIC, declared_by: BE-FlowAndFills}'
      p_fill: 'NullPin{field: p_fill, assumption: "1.0", bias_direction: OPTIMISTIC, declared_by: BE-FlowAndFills}'
      own_impact: 'NullPin{field: own_impact, assumption: "0", bias_direction: OPTIMISTIC, declared_by: BE-FlowAndFills}'
      population_markout: 'NullPin{field: population_markout, assumption: "average maker = marginal maker", bias_direction: OPTIMISTIC, declared_by: BE-FlowAndFills}'
      venue_ack_lag: 'NullPin{field: venue_ack_lag, assumption: "0", bias_direction: OPTIMISTIC, declared_by: BE-FlowAndFills}'
    notes: 'pair-aware by construction. Returns Unavailable when: the bracket sign flips
      (R-BRACKET); the mid conventions disagree (R-MIDROBUST); FeeSchedule is Disputed and a
      net-of-fee estimand is requested (R-PROV); or alpha_trade < 0.30. Fitted objects are
      offline artifacts resolved through artifact_resolver, never a runtime read of EV-Markout
      (EV is read by nobody).'

# MIGRATION (v12 -> v13), requires records in migrations.yaml:
#   1. add field:Fill.markout_partition : MarkoutPartition?   -- per-fill attribution; the
#      ActionOutcome-level partition is share-weighted and cannot be decomposed after the fact.
#   2. change proto:OutcomeModel.evaluate return to Known[Uncertain[ActionOutcome]] | Unavailable
#      -- R-KNOW: every fitted object must be knowledge-stamped, and this module is fitted
#      (ArrivalIntensity, AdverseSelectionCurve both carry artifact_id + fit_data_through).
#      FLAGGED, NOT TAKEN: the bare signature is the agreed interface and changing it is the
#      structure loop's call, not this planner's. Interim: t_known composes through
#      ActionOutcome.provenance.

gates:
  G-FF1:
    question: is `side` the taker's direction?
    metric: on-chain OrderFilled taker direction vs WS side, agreement
    threshold: 0.99
    unit: fraction
    data_prereq: frozen >=500-tx sample with receipts
    owner: BE-FlowAndFills
    on_fail: HALT_PROGRAM
    notes: if `side` is the maker's, maker gross flips +95 -> -95 bps
  G-FF2:
    question: is queue-ahead knowable?
    metric: alpha_trade (trade-vs-cancel attribution, dt = 250 ms frozen)
    threshold: 0.30
    unit: fraction
    owner: BE-FlowAndFills
    preconditions: [G-FF1]
    on_fail: drop queue modelling; regress fill on level and displayed depth
  G-FF3:
    question: what is the sign of zeta?
    metric: R(resolution) - R_placebo(resolution), day-clustered block bootstrap
    threshold: 0.0
    unit: cents_per_share
    owner: BE-FlowAndFills
    preconditions: [G-FF1, G-FF2]
    inference_method: block bootstrap clustered by day, Holm across the h-grid, notional-weighted
    review_date: 2026-09-19
    on_fail: exclude strata where |zeta| exceeds gross capture
    notes: uninformative before ~30 day-clusters given the tail (49% of BTC gross from 5 of 171 windows)
  G-FF4:
    question: does the bracket admit a signed answer at all?
    metric: fraction of (coin, stratum) cells with QueueBracket.sign_agreement
    threshold: 0.5
    unit: fraction
    owner: BE-FlowAndFills
    preconditions: [G-FF2]
    on_fail: the module is Unavailable for most of the action space; MM is not deployable
```
