# BE-FlowAndFills — MODEL PLAN: the FLOW model

**Rebuilt from first principles, 2026-08-21; narrowed to the flow model.**
`BE_FLOWANDFILLS_PLAN.md`'s measurements are carried forward as constraints; its
design (the G-FF1..G-FF4 chain, α_trade-as-queue-knowability, the queue bracket,
ζ as a standalone experiment) is set aside and re-derived.

Status: **PLAN ONLY.** Nothing here is measured. No code accompanies it.

---

## 0. The decision, and why flow is the subject

**Can we make markets profitably on these binaries, and how would we know we
were wrong?**

Post a bid at `ℓ` when fair value is `F_t`, with `F_t − ℓ =` half-spread. It
fills at `τ`:

```
   net per filled share  =  half_spread  +  rebate  −  maker_fee  −  AS       (I)
   AS := −E[ F_τ − F_t | filled ] ≥ 0
   E[PnL]                =  E[N] × (I)                                        (II)
```

`AS ≥ 0` is not an assumption — it is what "selected" means. Two consequences
set the whole agenda:

**(a) The sign of (I) is independent of queue position.** Queue enters only
through `E[N]` in (II), and through the conditioning inside `AS`.

**(b) Substituting the realised outcome for `F_τ` gives a model-free spine:**

```
   realised edge per filled share  =  E[ outcome − ℓ | filled at ℓ ]         (III)
```

(III) needs no fair-value model, no sigma law, no route_a. **R-SPINE:** any
decomposition must reconcile to (III) on the same fills; where it does not,
(III) wins.

**Both `AS` and `E[N]` are properties of taker flow.** That is why the flow
model is the object to build: it is the only term in (I)–(II) that is both
decisive and fully observed.

### Headline finding (U4): on real flow, makers lose per fill

Measured model-free, in-window, n=447,380 fills:

```
all flow                        per-fill  +0.165 c     share-wtd  +0.173 c
EXCLUDING the 0.02 class        per-fill  -0.211 c     share-wtd  +0.172 c
the 0.02 class alone            per-fill  +1.987 c  -- CAPACITY ~$91 / 2 DAYS
```

**The only positive per-fill maker edge in this tape is against a single
counterparty with essentially no capacity.** That class is 17.1 % of fills but
**0.0153 % of notional**: $727 of notional in the scanned subset against
$4,740,742 for everything else, and its entire maker edge is **$30.42 in the
subset, ~$91 scaled across the full tape over two days** — order tens of dollars,
market-wide. It cannot be harvested at any size.

**Strip it out and the maker loses −0.211 ¢ per fill on real flow.** The pooled
positive exists only because of that counterparty. Share-weighted the edge stays
mildly positive (+0.172 ¢) because a maker earns per share, not per fill — so the
two weightings answer different questions and **both must be carried** (R-DUAL).
This is not a footnote to the flow model; it is the condition the flow model
exists to explain.

**Cost inputs, settled — folded in and moved past.** `f(p) = 0.07·p(1−p)`
$/share, matched to four decimals across the full moneyness range at **n = 600**;
corpus Q5's `0.07·min(p,1−p)` is **REFUTED at 2×**. **Incidence: the taker pays,
the maker does not** — 600/600 taker legs carry a fee, 744/754 maker legs carry
zero, under an unambiguous test (the fee-bearing leg is the one whose `taker`
field *is* the exchange address). The three-way conflict across `PM_REVIEW_ITER1`
/ `PM_REVIEW_LOOP` / `PM_SKETCH_REVIEW_ITER1_M` (docs 7 % vs CLOB `base_fee`
1000 vs `fee_rate_bps = 0`) is **dissolved, not arbitrated**: the docs were
right, CLOB is consistent, and the observed zero was the wrong source. The rule
*never read `fee_rate_bps`* stands, now explained rather than merely observed.

Two things do **not** lift. **`ρ(p)` the rebate remains UNKNOWN** — a zero maker
fee is not evidence a rebate was paid, and `OrderFilled.fee` is unsigned so a
rebate could not appear in it; anything requiring `ρ` stays `Unavailable`, and
only the **taker** side of §12.1's net-of-fee block lifts. And **10 of 754 maker
legs do carry a fee** — a real **1.3 % residual class, not a classification
artefact**, recorded as **unexplained**.

These are inputs to update, not the subject. They matter to the flow model in
exactly one way, and it is load-bearing: **crossing at ATM costs a taker
~0.50 ¢ half-spread + 1.75 ¢ fee ≈ 2.25 ¢/share, ~225 bps on a $1 binary, and
that barrier varies ~2.8× across moneyness by a known closed form.** That single
fact supplies both the falsification test of §3 and the selection premise of §4.

---

## 1. What kind of problem this is

**1.1 Fair value is computable, not latent.** Settlement is a deterministic
function of a public stream — `S60(T)` vs `S60(t0)`, verified at **99.8 %** on
1,465 windows, the full-300 s reading **refuted** at 86.9 %. The maker's edge
cannot come from knowing fair value better, only from being paid enough to bear
knowing it slightly later than someone else.

**1.2 Settlement is a trailing mean, so it is partially determined early.**
`S60(T)` averages the final 60 s; at `T − 30 s` half the settlement quantity is
already fixed. **Uncertainty collapses on a known schedule.** This is the single
most important structural fact for the flow model: intensity is dominated by
deterministic non-stationarity in `r`, not by stochastic clustering.

**1.3 The tick is enormous — but "modal spread = 1 tick" is a BTC statement,
not a market one (U8).** A 0.01 tick is **1 % of notional** on a 0–1 contract.
ATM median spread in ticks by coin: **btc 1 · eth 1 · sol 3 · doge 3 · xrp 5 ·
bnb 5 · hype 7**. The corpus's pooled "66.7 % at 1 tick" is a **pooling
artefact** — btc supplies 2.28 M of ~3.73 M quotes. So the ATM half-spread is
**0.5 ¢ on btc/eth and 1.5–3.5 ¢ on the thin coins**, against a fee that is
1.75 ¢ at ATM regardless. **These are different businesses per coin**, which is
why §6 reports per-coin as primary.

**And spread width does NOT confer edge — at identical widths the sign flips.**
Pairing U8's ATM spread against U4's share-weighted maker edge:

```
1 tick   btc  +0.201      eth  -0.008
3 ticks  doge +0.629      sol  -0.628     <- same width, opposite signs
5 ticks  xrp  +0.705      bnb  -0.476     <- same width, opposite signs
7 ticks  hype +1.761
```

The extremes agree — narrowest (btc) is thin, widest (hype) is best — but the
middle **splits almost symmetrically at every width**. A monotone
"wide spread ⇒ good business" reading is **false**, and it is the inference a
casual reader takes from the extremes alone. What the pattern is consistent with
is that a wide spread **prices adverse-selection risk** rather than conferring
free edge: the compensation and the hazard scale together, so width alone
predicts nothing. Away from the money the tick becomes 0.001
(6.4–16.2 % of tail quotes, coin-dependent) and the half-spread falls 10× while
a proportional fee does not.

---

## 2. The flow model

### 2.1 The object, and why it is privileged

A **marked point process of taker arrivals**:

```
   { (t_i , side_i , size_i , p_i , ℓ_i) }
```

Every element is directly observed. No queue, no latent state, no fair-value
model. This is the only object in the programme with that property.

**A granularity fact G-FF1 established incidentally, and it matters here.** One
WS `last_trade_price` event corresponds to one on-chain `OrdersMatched` — the
**taker order's aggregate**, not a per-maker fill. One inspected transaction
carried three `OrderFilled` legs under a single `OrdersMatched`. So counting
`last_trade_price` events counts **arrivals**; counting `OrderFilled` legs would
count *legs* and overstate intensity by a leg-multiplicity that is itself
state-dependent (bigger sweeps in busier states consume more legs). Estimating λ
on legs would build that state dependence into the intensity as a spurious
effect.

**Complement fold is mandatory.** `price_changes[]` has length 2 in 100.00 % of
28.7M messages — the pair book is one book. But **trades are single-sided**:
flow at unified level ℓ is (trades on Up @ℓ) + (trades on Down @1−ℓ,
sign-flipped). Skipping the fold halves every intensity estimate.

### 2.2 The arrival process

```
log λ(t | x)  =  f_r(r ; coin)                    ← deterministic window clock, DOMINANT
               + f_p(p)                            ← moneyness
               + β_side · 1{side = BUY}             ← MOST exposed: the class is 99.98% SELL
               + γ_tick · 1{tick = 0.001}           ← tail-only interaction, never a main effect
               + f_book( spread_ticks, touch_total, unified_imbalance )
                                                    ↑ CONTAMINATED ON COUNTS (R-DUAL)
```

Estimated as a **Poisson/GLM on binned exposure**, per coin, walk-forward by day.

**`f_r` is estimated non-parametrically first**, as an empirical count profile
per `(coin, r-bin)`, before any parametric or self-exciting term is entertained.
The reason is a specific recorded failure mode: a Hawkes process with a constant
baseline attributes clock-driven intensity growth to self-excitation, and adopts
itself on in-sample fit. §1.2 guarantees strong deterministic structure in `r`,
so that misattribution is not hypothetical here.

**The self-excitation question is asked only on the time-changed process.** With
compensator `Λ(t) = ∫λ`, the transformed times `Λ(t_i)` are unit-rate Poisson
under a correct model. Residual clustering is tested there, never on raw
inter-arrival times. A branching ratio estimated without time-changing by the
fitted clock is not evidence.

**Conditioning choices, each with its justification:**

| covariate | why |
|---|---|
| `r` | §1.2 — the mechanical uncertainty schedule. Expected dominant effect |
| `p` | uncertainty peaks at ATM; also the cost barrier varies with `p` (§3) |
| coin | ~20× intensity range: BTC 2,424 trades/window vs doge/bnb/hype 108–133 |
| side | the 77 % BUY skew is **symmetric across both tokens**, so it carries no unified direction — it is a per-token convention. Fold first, then condition |
| tick regime | `tick_size_change` fires in **567 of 766 windows** (2,320 events). A 10× grid change is routine, not a regime event, and must be a covariate rather than a stratum. **Read the tick from BOTH `book.tick_size` and `tick_size_change.new_tick_size`** — see §3.4, where matching only the former produced a false "all 0.01" reading |
| book state | spread, touch total, imbalance — read from `price_change.best_bid/ask`, **never** `book` snapshots (p90 6.2 s stale). **`imbalance` is CONTAMINATED ON COUNTS** (~−16 pp shift, state-dependent) and must be notional-weighted, with the count version reported beside it under R-DUAL |

**Contamination guard on book state.** `price_change` carries **post**-change
quotes (99.67 %) and can be emitted *before* the `last_trade_price` for the same
match. Book-state covariates must therefore be read **strictly pre-arrival at a
frozen lag** (Δ = 250 ms, with a sensitivity ladder), or the model conditions on
the consequence of the arrival it is predicting.

**Per-coin is mandatory, and it costs us the alts.** A kernel-level or
richly-conditioned fit is estimable on BTC and not on doge/bnb/hype. A model
fittable on one of seven coins is not the module's model; the alt specification
must be coarser by design, not by failure.

### 2.3 The mark distribution

Marks are `size` and the level(s) consumed.

Observed: pooled p10 **0.02** / p50 **5.81** / p90 **42.76**, mean 22.23 —
heavy-tailed and strongly coin-dependent (BTC p50 8.9; alt p50 0.02–5). Model as
a state-conditional distribution, not a mean: `size | (r, p, coin, book state)`,
with the tail modelled explicitly since the tail is where the money is (49 % of
BTC maker gross came from 5 of 171 windows).

**The unit is settled (U1a `CLEARED`):** WS `size` equalled
`taker_amount_filled / 1e6` exactly on 600/600 transactions — shares at 6
decimals. Volume arithmetic is **unblocked**.

### The inversion — the count layer is the contaminated one

An earlier draft of this plan said the count layer was available today and the
volume layer was blocked. **That is backwards, and the correction is the most
consequential thing measured so far for this model.**

Population measurement, **36,151 events over 81 windows, unstratified**:

```
size == 0.02 exactly     5,878  =  16.3% of EVENTS
  side mix               5,876 SELL / 2 BUY   (99.97% one-sided)
  share of NOTIONAL      0.0145%
sub-minimum (<5 shares)  35.1% of events
modal conforming sizes   5.0 (8.3%) · 10.0 (5.7%) · 30.0 · 20.0 · 15.0
```

**16.3 % of arrival events carry 0.0145 % of notional.** Two consequences:

| layer | status |
|---|---|
| **COUNT** — arrivals/second | **CONTAMINATED.** One economically empty class supplies ~1 in 6 events |
| **VOLUME** — shares/second, notional | **UNBLOCKED and materially cleaner.** The contaminating class is 1/7000th of notional, so notional-weighted intensity is essentially untouched |

**`orderMinSize` is respected by the conforming population.** The size
distribution is **bimodal** — round sizes at or above the minimum (5, 10, 15,
20, 30) plus a separate 0.02 SELL-only class. U1b is therefore about a distinct
mechanism, not about a soft or unenforced minimum.

### Binding specification rule (R-DUAL)

**`λ` may not be specified on raw counts alone, and neither may ANY signed flow
quantity.** Every intensity estimate *and every signed quantity* — order-flow
imbalance, side mix, signed volume — is reported **both ways**: raw, and with the
0.02 class separated, with the excluded class published beside the retained one.

**Signed quantities are the more fragile case, and the reason is arithmetic.**
Contamination in a count averages out as `n` grows; contamination in a *signed*
statistic does not, because this class is **99.98 % one-sided**. A count-based
imbalance `(B − S)/(B + S)` inherits a deterministic shift of about
**−16 percentage points** toward SELL — the class's event share — while the
notional-weighted version inherits **−0.015 pp**. That is a factor of ~1,100.

**Worse than a constant: the bias is state-dependent.** 16.3 % is the pooled
share; the class's *local* share varies by coin, window and phase. A bias that
moves with state is precisely what a model fits as structure, so a
count-imbalance covariate would not merely be noisy — it would be
*systematically misleading in a way that looks like signal*.

**Therefore notional-weighting is a much stronger defence for signed quantities
than for counts.** For `λ` it is a refinement; for imbalance it is the
difference between a usable covariate and an artefact.

This binds `f_r`, `f_p`, `γ_tick`, the book-state response and any
self-excitation term: **fitted on raw counts, each would partly be fitting one
class's behaviour rather than market flow.** The 99.97 % one-sidedness makes
`β_side` the most exposed of all, since the class is essentially a pure SELL
stream.

Notional-weighted intensity is the robust default; raw counts are the
diagnostic that must be shown next to it, never the sole basis for a claim.

**U1b `CLEARED` — SINGLE-ACTOR, and now established rather than suspected.** An
**unstratified** systematic draw of 300 transactions, 0 receipt failures,
resolves **300/300 to one address**: top-1 = top-5 = top-10 = **100.0 %**,
distinct addresses = **1**, HHI = **1.0000**, and it is the only address present
in **all 7 coins**. Population in the scanned subset: 76,540 events at exactly
0.02 shares, **99.98 % SELL**.

**Consequence, pre-committed before the measurement:** the 0.02 class **may be
excluded from `λ`, with the exclusion published**. R-DUAL is unchanged — both
weightings are still reported, and the exclusion is shown beside the retained
set rather than replacing it.

**What is established and what is not.** "One address" is a measurement.
*What that address is doing* is not established, is not narrated here, and no
downstream estimand may assume a motive for it. The economic footprint is the
part that matters and it is already measured: 16.3 % of events, **0.0145 % of
notional**.

Marks are `size` and the level(s) consumed. Model as a state-conditional
distribution, not a mean: `size | (r, p, coin, book state)`, with the tail
modelled explicitly since the tail is where the money is (49 % of BTC maker
gross came from 5 of 171 windows), and with the bimodality represented rather
than smoothed over.

---

## 3. Falsification tests

### 3.1 The naive test is NOT identified — state this before designing around it

The crossing cost `f(p) = 0.07·p(1−p)` is deterministic, exogenous and swings
**2.8×** across moneyness: **1.75 ¢/share at `p = 0.5`** versus **0.63 ¢/share
at `p = 0.1`**. It is tempting to test whether `λ(p)` responds to it.

**It cannot be identified that way.** `f(p)` is a deterministic function of `p`
alone, so its effect on `λ` is exactly collinear with any other effect of `p` on
`λ` — and there is a large one pointing the other way, since uncertainty also
peaks at ATM. The fee rate is constant across our sample (confirmed to four
decimals on every row), so there is no time variation to exploit either. A
regression of `λ` on `f(p)` alongside a flexible smooth in `p` is unidentified;
without the smooth it is confounded.

Recording this matters more than the test: the programme's failure mode is
adopting an estimator that quietly needs an unidentified quantity. **The λ(p)
version of this test is exactly that, and it must not be run as though it were
evidence.**

### 3.2 T1 — the threshold test: a sharp directional prediction, NOT identification

The fee does not bind on *how many* cross in a way we can separate; it binds on
**who** crosses. A taker pays `p + f(p)` and crosses only when expected edge
exceeds `f(p)`. The fee therefore imposes a **selection threshold on arrival
markout**, and the threshold's variation across `p` is known exactly.

**The target magnitude is now measured, not assumed.** Across **2.29 M
executable quote observations** (`price_change.best_bid/ask`, 12 BTC windows,
2026-08-20), the median spread is **0.0100 at every moneyness bucket**:

```
moneyness     n         median spread   half-spread   fee     total
p<0.15        256,240   0.0100          0.50c         0.63c   1.13c
0.15-0.35     490,730   0.0100          0.50c         1.31c   1.81c
0.35-0.65     792,554   0.0100          0.50c         1.75c   2.25c
0.65-0.85     491,721   0.0100          0.50c         1.31c   1.81c
p>=0.85       256,827   0.0100          0.50c         0.63c   1.13c
```

**Half-spread is flat in `p`; the fee is the only cost component that varies.**
So the ATM-versus-tail cost difference is exactly the fee difference —
**1.12 ¢/share** — with no confounding movement in the spread. That is a real
strengthening of the *target*, and it is worth separating from what it does not
fix (below).

**Composition-weighted target (U2).** The 0.001-tick regime is confined to the
tails at ~6.7 % of tail quotes, where the half-spread is 0.05 ¢ rather than
0.50 ¢. So the `p<0.15` half-spread is `0.9325 x 0.50 + 0.0675 x 0.05 = 0.470`
¢/share, not 0.500 — consistent with the independently measured tail mean spread
of 0.00968. The cost gap becomes `(0.50 + 1.75) − (0.470 + 0.63) = 1.15`.

```
   PREDICTION:  E[ markout of arrivals | p≈0.5 ] − E[ markout | p≈0.1 ]
                ≈  1.15 ¢/share,  in that direction
```

That is a **2.7 % shift** from the 1.12 ¢ fee-only figure — well inside the
order-of-magnitude precision this test claims, so the revision does not change
what T1 can decide.

Markout is measured against the verified settlement target, so this needs no
fair-value model.

#### Why this is NOT identified

An earlier draft of this plan called T1 "identified because it predicts a
specific magnitude in a different quantity". **That was wrong, and it is exactly
the programme's documented failure mode — a quantity assumed identified one
level of abstraction up.**

The observed quantity is `E[ X | X > f(p), p ]`, where `X` is a potential
crosser's edge and `G_p` its distribution. That conditional mean depends on
**both** the threshold `f(p)` **and** the shape of `G_p`. Even with an identical
threshold, two moneyness regions would show different conditional means whenever
`G_p` differs — and there are strong reasons it does:

- **moves that flip the outcome are rarer far from the money**, so the arrival
  rate of genuine information differs by `p`;
- **payoff geometry is asymmetric off ATM** — at `p = 0.1` upside is 0.9 and
  downside 0.1, against a symmetric 0.5/0.5 at ATM;
- **the tick regime differs**, and the flat 0.0100 median spread above does not
  settle this: at a 0.001 tick a 0.0100 spread is **10 ticks**, not 1, so the
  achievable-edge grid is finer far from the money even where the cash
  half-spread is identical. The tick composition is currently **unread** (§3.5).

So the fee-threshold channel and the `G_p`-shape channel are **not separable**
in this comparison.

**What would identify it, and why none is available:** time variation in the fee
rate (none — constant to four decimals across the sample), cross-sectional
variation in fee at fixed `p` (none — one formula everywhere), or a comparison
venue with a different schedule (out of scope). **The fee effect is not
identifiable on this data.** Full stop.

#### What a pass and a failure each license

- **A pass licenses very little.** A difference near +1.12 ¢/share is
  *consistent with* the cost-barrier mechanism. It does not exclude the
  possibility that the entire difference comes from `G_p` varying with `p`.
  Report it as consistency, never as confirmation, and never as a measurement of
  the fee's behavioural effect.
- **A failure is genuinely informative.** A difference that is zero, wrong-signed
  or an order of magnitude off means one of: takers do not respond to a cost
  barrier that varies 2.8×; the `G_p` effect exactly cancels the fee effect (a
  coincidence requiring its own explanation); or the markout machinery is broken
  (T3, §3.5, exists to rule this one out first). Each is a live problem with the model
  of who crosses and why.

That asymmetry is the test's whole value: it is a **falsification instrument,
not an estimator.** Write results in that register.

**T1b — the sharper form.** A threshold truncates a distribution from below, so
the **lower quantiles** of arrival markout should shift with `f(p)` more cleanly
than the mean, which mixes informed crossers with urgent and noise flow. Test the
quantile shift as primary, the mean as secondary. T1b inherits the same
identification limit — a `G_p` whose left tail varies with `p` produces the same
signature.

**Power caveat, stated up front:** markout is noisy and day-clustered CIs need
day clusters we do not have. T1 is a direction-and-order-of-magnitude check at
coarse strata, not a precise estimate.

### 3.3 T2 — zero-sum reconciliation, and the rebate corollary

Per filled share, gross of any rebate, the three parties must sum to zero:

```
   taker_markout  +  maker_markout  +  venue_fee  =  0
```

With `venue_fee = 1.75 ¢/share` at ATM and the **rebuilt** maker markout of
**+0.17 ¢/share** in-window (U4 — see below; the previously cited +0.45 is
withdrawn), this predicts **`taker_markout ≈ −1.92 ¢/share`**. Three quantities
measured independently, forced to reconcile.

**This is a strong test because it is an accounting identity, not a model.** If
the three do not sum to zero, one of them is mis-measured — and the recorded
maker markout is a prime suspect, since the corpus's book-derived numbers were
measured on p90 6.2 s stale snapshots and one such finding has already been
withdrawn.

**The corollary is valuable.** A rebate is a fourth party to the identity:
`taker_markout + maker_markout + venue_keeps + rebate = 0`. So a **residual in
T2 is an indirect estimate of the rebate** — estimand #3, currently UNKNOWN,
never observed being paid, and worth 41 % of theoretical gross. Treat this as
suggestive only: it is a difference of large noisy numbers, and it cannot
distinguish a rebate from a measurement error in any of the other three. It is a
lead, not a measurement.

**That tension is RESOLVED — the figure was contaminated (U4).** The +0.45 was
mid-conditioned and inherited `stale_book_contamination`; it is also the same
number as the "+95 bps maker gross" claim. A **model-free rebuild** — needing no
mid and no book, only trade price, taker side and the settled winner — gives
**+0.173 ¢/share share-weighted in-window** (n=447,380 fills), ~2.6× lower.

Against a 0.50 ¢ half-spread that implies adverse selection of **~0.33 ¢/share**,
which is entirely plausible against a 225 bps barrier — where +0.45 had implied
an implausible ~0.05 ¢. The premise of the flow model is unchanged; the number
it rests on is now measured rather than inherited.

**And R-DUAL earned its place here.** Excluding the 0.02 single-actor class the
**per-fill** figure goes *negative* (−0.211 ¢) while the **share-weighted**
figure is unmoved (+0.172 ¢). Count-weighted maker edge is positive *only*
because of that one actor. The share-weighted figure is the economically
meaningful one — a maker earns per share, not per fill — and it is the robust
one. Two weightings disagreeing in sign is exactly the finding a single
weighting would have hidden.

**PER-COIN IS PRIMARY; the pooled figure is the diagnostic.** Share-weighted
in-window:

```
NEGATIVE   bnb -0.476    sol -0.628    eth -0.008
POSITIVE   btc +0.201   doge +0.629    xrp +0.705   hype +1.761
```

**A pooled +0.17 that is negative on three of seven coins is not a market-wide
maker edge.** This programme has been burned by exactly this shape twice — a
cross-sectional result that was really one factor, and a timing edge that was
really an expanding survivor universe — so pooled-positive-with-negative-
constituents is reported constituent-first, never pooled-first.

**Other phases, share-weighted:** pre-open **−0.573 ¢** (the cited +0.70 flips
sign); post-close **−1.732 ¢** against a cited −1.46 — **that is T3's
known-answer validation stratum passing**, which is what licenses trusting the
same pipeline in-window where no ground truth exists.

### No interval is computable, and that is structural

**Two collected days ⇒ no day-clustered CI exists for any U4 figure.** These are
point estimates with **unknown sampling error**, not estimates with wide
intervals. Anything downstream requiring an interval — a gate verdict, a
significance claim, a comparison of two coins — is **`Unavailable`** until the
day count grows, and must return that rather than a number. This is a refusal,
not a caveat.

### 3.4 The tick, now READ (U2 `CLEARED`) — and it is a constraint, not a convention

The G-FF1 probe's tick lookup matched only `"tick_size"`, which never matches
`"new_tick_size"`, so every `tick_size_change` was ignored and it reported
`{0.01: 600}`. That was the defect. **U2 has now read it correctly** on
6,702,978 executable quotes (BTC, 24 windows, 2026-08-20):

```
transitions observed     84, ALL 0.01 -> 0.001, none reverting
tick composition         p<0.15    93.25% @0.01   6.75% @0.001
                         0.15-0.35  100%   @0.01   0%    @0.001
                         0.35-0.65  100%   @0.01   0%    @0.001
                         0.65-0.85  100%   @0.01   0%    @0.001
                         p>=0.85   93.27% @0.01   6.73% @0.001
spread, where tick=0.001    1 tick in 99.9% of quotes
spread, where tick=0.01     1 tick in 90.8-97.2%
```

**The 1-cent spread is a CONSTRAINT, not a convention.** When the finer grid
appears, makers use it immediately — 99.9 % of 0.001-tick quotes sit at exactly
one tick, i.e. a 0.1 ¢ spread. They are not declining to step inside; they step
inside as soon as the venue allows it. The hypothesis that the flat 0.50 ¢
half-spread reflected maker restraint is **refuted**.

**The "flat half-spread in `p`" is a median artefact, and the refinement is
small.** The 0.001 regime is confined to the tails and is only ~6.7 % of tail
quotes, so it does not move the median — but conditional on that regime the
half-spread is 0.05 ¢, ten times finer. Composition-weighted, the `p<0.15`
half-spread is ~0.470 ¢ rather than 0.500 ¢, which moves T1's predicted
magnitude from 1.12 to **~1.15 ¢/share**, a 2.7 % shift. **T1 stands** — that is
well inside the order-of-magnitude precision §3.2 claims for it.

**Consequences for `γ_tick`.** The covariate is real but **confined to the
tails**: it is exactly collinear with extreme moneyness in the middle three
buckets, where 0.001 never occurs. Specify it as an interaction within the tail
buckets, not as a main effect across the book, or it will simply re-express
`f_p`. And under **R-DUAL** its intensity must be reported both count- and
notional-weighted.

**Residual caveat on G-FF1:** the tick composition of that sample remains
unknown, since the defective read defaulted every leg to 0.01. Any genuinely
0.001-tick leg was price-validated against a tolerance 10× too loose. This
touches the **price** check only — size agreement at 1e-6 is the binding
validation and direction comes from the amount pair — so the `PASS` stands.

### 3.5 T3 — post-close as a natural validation stratum

After `T` the outcome is determined, so **any** arrival is picking off a stale
quote at full information. Recorded: 0.22 % of trades, maker markout
**−1.46 ¢/share**, i.e. takers extract +1.46 ¢/share.

This is the cleanest available check that the markout machinery has the right
sign and scale: a stratum where the theoretical answer is known
(`permanent_AS = |ℓ − outcome|`) and the measured answer already exists. **If
the pipeline cannot reproduce the post-close number, it cannot be trusted on the
in-window number**, where there is no ground truth to check against.

---

## 4. Adverse selection as a property of flow

`AS` in (I) is not a separate experiment (the old plan's ζ). It is a **property
of the arrival process**: the markout of arrivals, measured against the verified
settlement target.

**Flow here is heavily selected, and the model must expose that rather than
average over it.** All-in crossing cost at ATM is ~2.25 ¢/share (1.75 fee + 0.50
half-spread) — **~225 bps on a $1 binary**. Every arrival is from someone
expecting more than that. This is not a market where noise traders dominate by
default; the barrier is far too high.

**The incidence finding is what makes this argument valid, and it was never
established before.** A fee gates *arrivals* only because the **taker** bears
it. Had it been maker-borne it would have entered quoting, not crossing, and
§3–§4 would both be void. The measured incidence — 600/600 taker legs charged,
744/754 maker legs at zero — is therefore not a cost detail but the premise of
the flow model's selection story.

Design consequence: report markout **conditional**, never pooled, across at
least

- **`r`** — late arrivals should be better informed, since settlement is more
  determined (§1.2). A monotone relationship is a prediction, not a control;
- **size** — if large arrivals show worse maker markout, informed flow is
  arriving big, which is the classic signature and it changes the size policy;
- **window phase** — pre-open (+0.70), in-window (+0.45), post-close (−1.46
  ¢/share) are different populations and pooling them is a category error;
- **direction relative to the recent underlying move** — separates informed from
  urgent.

The dispersion of arrival markout is as much the estimand as its mean. A
distribution truncated from below at `f(p)` (§3.2) is the model's own
prediction, and its shape carries the information about who is arriving.

---

## 5. Missingness — which flow quantities survive

The `1013` loss is **MNAR by mechanism**: the socket falls behind when the
message rate spikes, so gaps land preferentially on the busiest intervals. A
naive intensity estimate is therefore **biased down exactly where flow is
heaviest** — the worst possible place for a model whose main object is intensity.

### 5.1 Exposure accounting recovers the denominator, not the selection

The gap ledger records **exact boundaries** (`gap_start_ns`, `gap_end_ns`) per
window. A point-process likelihood handles this natively: estimate λ integrating
over **observed** time only, rather than excluding whole windows.

This removes the *denominator* bias — we stop counting time we could not
observe. It does **not** remove the *selection* bias: the observed intervals are
systematically lower-λ than the unobserved ones, because high λ is what caused
the gap. The residual bias on the **level** is downward, and it is not
removable by exposure accounting.

**A partial bound is available.** The collector stamps `coin_msg_rate_hint` into
disconnect records, so the message rate at the moment of loss is known. That
bounds λ-during-gap from below and turns "biased down by an unknown amount" into
"biased down by at least a computable amount". Worth doing; it converts an
unbounded caveat into a bounded one.

### 5.2 Shape vs level

**RESOLVED by U3, and the answer is a bound rather than a clean pass.** Gap
*occurrence* is not distinguishable from uniform, but only because the test is
underpowered at n=51 (`d_crit = 0.190` against 10 %-wide deciles) — uniformity is
**not** established. Gap *exposure*, which is what actually biases `λ`, **is**
concentrated: the first 30 s of a window carry **31.7 %** of all in-window
seconds lost, a 3.2× loss rate, against 1.4 % in the final decile.

The concentration is at the **start** of the window, not near expiry. Absolute
magnitude: **≤ 0.155 % of exposure in the worst decile, 0.0488 % overall.**

**That bound is EXPOSURE lost, explicitly NOT flow lost.** The two coincide only
if loss is independent of `λ`, and the dominant cause is by construction the one
that is not — 6 of the 7 first-decile gaps in `clob_v3_1` are MNAR-class
(4 `SLOW_CONSUMER_1013`, 2 `PING_TIMEOUT`, 1 `CONNECTIONCLOSEDOK`; coins btc 5 /
bnb 1 / eth 1). Gaps cluster at window open partly *because flow does*. An
earlier draft leaned on the exposure bound as though it bounded the `λ`
distortion; that claim is **withdrawn**.

**`coin_msg_rate_hint` cannot bound it.** Despite the name, `collect_pm.py:489`
stores `msg_by_coin` — a **cumulative counter since process start**, not a rate.
Comparing it across gaps compares process uptime, and doing so produced a
spurious "3.26× elevation" before the field was checked.

**Measured directly instead** (trades in the 10 s before each gap, against the
local first-30 s baseline; matched denominators):

```
first decile, clob_v3_1, n=7
  time lost   41.0s of 210s first-30s window-time =  19.53%
  flow lost   114.4 of 1,369 first-30s trades     =   8.36%
  ratio flow/time                                 =   0.43x
```

**Flow loss runs *below* time loss. The mechanism producing that is
unexplained.** An earlier draft said "the long gaps are the quiet ones"; that is
**withdrawn** — it read a window-mean-relative elevation as quietness, which U9
showed to be a phase artefact. Against their own same-decile baselines the two
long `PING_TIMEOUT` gaps sit at **0.96×** and **1.09×**: typical for their
windows, not quiet for their moment. Their absolute pre-rates were low because
those windows were absolutely quiet. **Why long gaps landed in absolutely-quiet
windows is not explained, and no replacement story is offered here.**

The ratio itself is robust: recomputed at **0.428×**, unchanged, because every
input is absolute (`pre/10 s`, first-30 s trade counts) and no baseline-relative
quantity enters. Doubling the `1013` during-gap rate still leaves it at 0.65×.

**So the reassurance survives, but on measured grounds rather than on the
exposure bound.** `f_r` is reportable on the `clob_v3_1` covered set with the
flow ratio stated beside the exposure bound. **n=7 — directional only.** Outside
that set there is no gap record and hence no bound of either kind.

**That incidental `PING_TIMEOUT` observation is WITHDRAWN (U9).** It compared
pre-gap rate against the **window mean**; the first decile is busier than the
window mean, so a first-decile gap reads "quiet" as a pure **phase confound**.
Against a same-decile baseline the `PING_TIMEOUT` ratio is median **1.05**
(n=7, CI [0.91, 1.72]) — mildly *above* 1, not below. The idle-connection
mechanism is not supported, `PING_TIMEOUT` keeps its **MNAR-suspect**
classification, and `clob_adm_v1` is unchanged.

**A caveat that limits all of U9:** it measures **trade** arrival rate, while a
`1013` is triggered by **message** rate (~97 % `price_change`). So it is the
right test for *"does this loss bias the flow estimand"* and the wrong one for
*"what causes the disconnect"*.

| quantity | trust? | why |
|---|---|---|
| λ **level** (events/s) | **No** | MNAR downward, worst on BTC and in busy windows. Exposure accounting + the rate bound narrows it; it does not fix it |
| λ **shape in `r`** | **Yes, with two stated bounds (U3, U3a)** | Uniformity **underpowered**, not established (`D=0.151, p=0.270`; `d_crit=0.190` at n=51). Exposure loss **is** concentrated — first decile 31.7 % of seconds lost, 3.2× the mean rate — bounded at **≤ 0.155 % of exposure**. That is an **exposure** bound, not a flow bound. Flow measured separately: first-decile **flow/time ratio 0.43×** (n=7), so flow loss runs *below* exposure loss because the long gaps are the quiet ones. **`clob_v3_1` covered set only.** |
| λ **shape in `p`** | **Conditional** | same test in `p` |
| **within-window ratios** (side mix, size-distribution shape) | **Yes** | a gap removes a contiguous slice affecting all categories alike |
| **relative intensity across `p`-bins within a window** | **Yes** | window-level confounds difference out |
| **markout of observed arrivals** | **Qualified** | if bursts co-occur with information events, we lose the most informed arrivals ⇒ measured taker informedness is **understated** ⇒ MM looks **better** than it is. `NullPin{OPTIMISTIC}` |
| anything **day-clustered** | **No** | the usable set is a single era spanning under a day (§5.3) |

The markout row is the one to watch: its bias runs in the direction that
flatters the thesis, which is the direction this programme has repeatedly been
caught by.

### 5.3 The ledger coverage constraint binds hardest

The CLOB gap ledger begins **2026-08-20 14:50:21 UTC**; windows begin
2026-08-19. Only **1,057 of 3,076 windows (34.4 %)** have coverage. For the
other **65.6 %, the absence of a gap record is not evidence of a clean window** —
the collector was not yet recording gaps. Those are `NO_LEDGER_COVERAGE`, not
"probably fine": the identical error to reading `open_gaps=[]` on the prices lane
as "clean" while it logged 58 gaps in 11 hours.

Within the covered set: 47 gap-touched (4.4 %); BTC 34 of its 151 covered
windows = **22.5 %**, which reconciles with HANDOFF's "~22 %" — that claim is
correct on the ledger-covered denominator and should not be amended.

Usable set: `clob_v3_1`, ~805 covered windows less shards and gap-touched —
order of 750, ~107 per coin, **spanning under a day**. Ample for within-window
count structure; insufficient for any day-clustered statement. Never pool across
`collector_version` eras unpaired. Full specification:
`CLOB_ADMISSIBILITY_PROTOCOL.md` (`clob_adm_v1`).

---

## 6. Identified vs assumed

| # | estimand | unit | status |
|---|---|---|---|
| 1 | `half_spread(coin, moneyness, phase)` | ¢/share | **IDENTIFIED** — census |
| 2 | `maker_fee` | ¢/share | **IDENTIFIED ≈ 0** — 744/754 maker legs zero at n=600 |
| 2b | `f(p) = 0.07·p(1−p)`, taker-borne | ¢/share | **IDENTIFIED** — four decimals, full moneyness range, n=600. Q5 **REFUTED at 2×** |
| 2c | the 10/754 fee-bearing maker legs | — | **SELECTOR IDENTIFIED (U7b): a PER-ADDRESS FEE TIER** — 0 bps / ~10 bps / 50 bps, constant within address across 5 addresses with >=2 legs (thin: n=2–3 each). Prior U5 detail — all at px=0.9900, flat **10 bps (×7) / 50 bps (×3)** of USDC notional, **not** the taker formula; in mint-match transactions against a 0.01 complement. But **23 fee-free maker legs sit at the same 0.99**, so price is necessary-not-sufficient and the selector is unknown. Both the both-roles and taker-formula hypotheses are refuted. 1.3 % residual; makers pay ~0 on 98.7 % |
| 3 | **`rebate` `ρ(p)`** | ¢/share | **UNKNOWN — and now bounded (U7a).** No rebate is paid **inside the trade**: 600 receipts, zero third-party contracts, zero unexplained maker-bound value. A **periodic/off-chain** rebate remains out of reach of this data, so `ρ`-dependent estimands stay **`Unavailable`**. T2 residual is a lead only |
| 4 | `realised_edge` (III) | ¢/share | **IDENTIFIED** — census, model-free |
| 5 | λ **shape** in `r`, `p` | — | **CONDITIONALLY IDENTIFIED** — pending the gap-uniformity test (§5.2) |
| 5b | λ **level** | events/s | **BIASED DOWN, MNAR.** Bounded below via `coin_msg_rate_hint` |
| 6 | intensity, **count-weighted** | events/s | **CONTAMINATED** — 16.3 % of events carry 0.0145 % of notional. Reportable only under R-DUAL, beside the notional-weighted figure |
| 6b | intensity, **notional-weighted** | $/s | **IDENTIFIED and UNBLOCKED** — the robust default; the contaminating class is 1/7000th of notional |
| 6c | mark distribution | shares | **IDENTIFIED**, and **bimodal** — conforming round sizes (5/10/15/20/30) plus a separate 0.02 SELL-only class |
| 7 | arrival markout / `AS` | ¢/share | **IDENTIFIED** on observed arrivals; `NullPin{OPTIMISTIC}` for missingness |
| 8 | self-excitation beyond the clock | — | **TESTABLE** only on the time-changed process |
| 9 | post-close exposure | ¢/share | **IDENTIFIED** — census; T3 validation stratum |
| 10 | `own_impact` | ¢/share | **NOT IDENTIFIABLE.** `NullPin{0, OPTIMISTIC}` |
| 11 | `venue_ack_lag` | ms | **NOT OBSERVED.** `NullPin` |
| 12 | `size` **unit** (U1a) | shares | **CLEARED** — `taker_amount_filled/1e6` exact, 600/600. Volume arithmetic unblocked |
| 12b | the 0.02 SELL-only class (U1b) | — | **UNRESOLVED.** 16.3 % of events, 0.0145 % of notional, 99.97 % SELL. Mechanism not established; **do not narrate** |
| 13 | `side` convention | — | **IDENTIFIED** — G-FF1 PASS, 600/600, Wilson95 [0.9936, 1.0000]. **Tick composition of the sample is UNKNOWN** (§3.5), not "0.01 only" |
| 14 | `half_spread` vs `p` | ¢/share | **IDENTIFIED — FLAT at 0.0100 median** across all five moneyness buckets, n = 2.29 M executable quotes, 12 BTC windows. **BTC, one day** |
| 15 | tick composition | — | **CLEARED (U2)** — 0.001 confined to the tails (6.7 % of tail quotes, 0 % mid-book); spread = 1 tick in 99.9 % where it applies. **BTC, one day** |

---

## 7. What the flow model cannot reach on free data

1. **The arrivals that did not happen.** We observe crossings that occurred,
   never the taker who declined because `f(p)` was too high. The counterfactual
   arrival rate under a different fee is outside the data — which is the deeper
   reason §3.1 is unidentified.
2. **Taker type.** Informed / urgent / noise is not labelled. On-chain addresses
   permit clustering, but that is `BE-Competition` scope, not flow.
3. **Unfilled taker interest** — IOC orders that crossed nothing, and any
   order-level intent. Only executed arrivals are on the tape.
4. **`own_impact`** — not identifiable from a tape containing none of our
   orders. Only randomised live quoting identifies it.
5. **`venue_ack_lag`** — unobserved, so the cancel round trip stays open.
6. **The tick regime's effect on the side convention** — G-FF1's tick reading was
   defective (§3.4), so the sample's tick composition is unknown. This is a
   *currently unread* quantity rather than an unreachable one: it becomes
   reachable as soon as the tick is read correctly.
7. **λ level to better than the MNAR bound** (§5.1), on the current lane.
8. **Anything day-clustered** on the admissible set as it stands (§5.3).

---

## 8. Designing against the documented failure mode

The recorded pattern is *"each error was the previous error one level of
abstraction higher"*. Five guards, each aimed at a specific one:

1. **One spine, one number.** (III) is the measurement; every decomposition
   reconciles to it (R-SPINE). Prevents a second estimator of the same quantity
   appearing at a higher level of abstraction.
2. **Clock before cluster.** `f_r` non-parametric first; self-excitation only on
   the time-changed process. Prevents the Hawkes-eats-the-clock error.
3. **Unidentified means unrun.** §3.1 is written down as not identified so it
   cannot later be run as evidence.
4. **Assumed quantities cannot gate.** #3, #10, #11 carry pins with bias
   directions; under R-PROV an `ASSUMED` parameter may not gate a decision.
5. **Never a single weighting (R-DUAL), and shape before level.** Intensity is
   reported count-weighted *and* notional-weighted, with the excluded class
   beside the retained one. This guard exists because the plan previously
   asserted the opposite of the truth — that counts were clean and volume was
   blocked — and a single-weighting specification is exactly what let that pass
   unnoticed. Two weightings that disagree are a finding; one weighting is an
   assumption.

---

## 9. Carried forward, and discarded

**Carried (measured, survives redesign):** settlement `w = 60 s` at 99.8 % and
the 300 s refutation; S30/S60 as trailing simple means with the `btc/usd` basis
caveat; `side` is the taker's (G-FF1 PASS, 0.01-tick caveat); the WS trade event
as the **taker-order aggregate**; zero trade duplication; modal spread 1 tick and
the routine 0.01↔0.001 regime change; the three clocks (47 ms book / 448 ms spot
mirror / 1,700 ms settlement TWAP, of which 1,440 ms is PM-side publication);
`book` snapshots p90 6.2 s stale so read `price_change.best_bid/ask`;
`price_change.size` as level total at 96.10 % replay; the keccak-verified
on-chain ABI and maker/taker identity; `fee_rate_bps = 0` as a WS artefact;
window-phase markouts; the MNAR/MAR loss split.

**Discarded (design, not measurement):** the G-FF1..G-FF4 chain as an ordering;
α_trade as a gate — it becomes one descriptive covariate of book dynamics among
several, and the gate's undeclared 0.30–0.60 band disappears with it; the
Back/Front queue bracket as the primary construction; ζ as a standalone
experiment rather than a property of the arrival process (§4); and the
assumption that a fair-value model is needed before a maker-edge number can be
produced.

**Refuted here:** the corrected-premises list in `BE_FLOWANDFILLS_PLAN.md` says
the modal spread is 1 tick "(ATM runs 6–8 c)". The **6–8 c part is refuted for
BTC** — ATM median is 1 tick (0.0100) with p90 0.020, on 2.29 M executable quote
observations. Scope: **one coin, one day.** Do not generalise to thin coins
without re-checking; `hype`, at P(1 tick) = .06, may genuinely differ. Mark
refuted-for-BTC, unverified elsewhere.

**Do not rebuild on:** "the book beats our model at every horizon" (withdrawn,
mis-anchored); the FLB edge (downgraded to a 0.0004 Brier rounding error,
one-sided, on stale books); anything book-derived in `PM_DEEP_REVIEW.md`
including "+95 bps maker gross / +136 with rebate" (unverified). And stop
propagating **84.9 %** for the at-ask share — the measured figure is **63.7 %**
(vs 15.1 % at the bid); 84.9 % appears to be `100 − 15.1`.
