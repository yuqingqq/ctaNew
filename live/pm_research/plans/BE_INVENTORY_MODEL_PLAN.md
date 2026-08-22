# BE-Inventory — model plan

Plan only. No code, no measurement, no fitted quantity. Status: **DESIGN**, not
decision-eligible. Written 2026-08-22.

> For current programme state read [`FLOW_MODEL_STATE.md`](../FLOW_MODEL_STATE.md).
> This document is a **plan** — a design under constraints, not a statement of
> measured belief. Where it conflicts with that page, that page wins.

---

## 0. Three corrections to the brief, before anything is built

The brief asked to be treated as a hypothesis. Two of its three load-bearing
claims do not survive, and one is structurally different from how it was framed.

### 0.1 The two books are ONE book, mechanically mirrored

**Verified on the tape before writing:** across 37,394 consecutive
`price_change` messages in a BTC window, **both tokens are quoted in every
single message** (37,394 / 37,394), and

```
bid(Up) + ask(Down) = 1.0000      p10 = p50 = p90 = 1.0000
                                  within 0.005 of 1.0 : 100.0 %
```

This is an identity, not a tendency. `ask(Down) ≡ 1 − bid(Up)`, so **an order on
the Down book IS an order on the Up book**, mirrored. There are not two books to
arbitrage between; there is one book with two representations.

**What survives.** The complete-set logic is real: filled on both tokens in equal
size, you hold a set worth exactly $1 at resolution, with no direction and no
unwind. Bidding both sides pays `bid(U) + bid(D) = 1 − spread` for something
worth 1.

**What does not.** That is **ordinary two-sided market making**, not a special
mechanism. "Quote the complement to offset" is the same act as "quote the other
side of the book" — which a two-sided maker is doing anyway. And the capture is
**one spread, not two**. Any plan that treats complement-quoting as a *third*
mechanism distinct from skewing is double-counting a single action.

**Consequence for this plan.** `Skew` and `quote the complement` collapse into
one mechanism. The genuine menu is **two** mechanisms plus doing nothing, not
three.

### 0.2 "Cost per unit of standard deviation" is not the decision metric

The brief's algebra is arithmetically right. Verified:

| p | fee ¢/sh | var `p(1−p)` | fee/var | fee/sd |
|---:|---:|---:|---:|---:|
| 0.50 | 1.750 | 0.2500 | 700 | 3.500 |
| 0.20 | 1.120 | 0.1600 | 700 | 2.800 |
| 0.10 | 0.630 | 0.0900 | 700 | 2.100 |
| 0.02 | 0.137 | 0.0196 | 700 | 0.980 |

Cost per unit of **variance** removed is constant; per unit of **standard
deviation** it peaks at ATM. Both true.

**But the conclusion drawn from it — "passive unwind at ATM, taker-dump in the
tails" — is backwards.** Cost-per-sd prices risk *removal*; it ignores how much
risk is *held*. Risk penalty is quadratic in position, cost is linear, so the
decision turns on position size.

Under a mean-variance penalty `(γ/2)·N²·p(1−p)` against a dump cost
`N·[0.07·p(1−p) + h]` where `h` is the half-spread:

```
dump when   N > N*(p) = (2/γ) · [ 0.07 + h / (p(1−p)) ]
```

Half-spread `h` is **measured flat at 0.50 ¢** across all moneyness (btc/eth,
1 tick, 2.29 M quotes), while `p(1−p)` shrinks in the tails — so `h/(p(1−p))`
**grows** there and `N*` grows with it. At `h = 0.005`:

```
p = 0.50   N* = (2/γ)(0.07 + 0.020) = 0.090 · (2/γ)
p = 0.10   N* = (2/γ)(0.07 + 0.056) = 0.126 · (2/γ)      1.4x higher
```

**You tolerate a LARGER imbalance in the tails and dump MORE READILY at ATM** —
the opposite of the brief. The mechanism is that a share in the tails carries
less risk to begin with, and the quadratic penalty means more of them fit under
the same budget.

The brief's rule would have dumped precisely where dumping is least necessary.

### 0.3 Risk is NOT symmetric in `p`, and `p(1−p)` hides it

Both the brief's framings are symmetric about `p = 0.5`. Worst-case loss is not.

A long position pays for itself: **worst case is losing the price paid.** Long
100 Up at `p = 0.90` risks **$90**. Long 100 Down at `p = 0.90` — i.e. Down at
0.10 — risks **$10**. Identical `|net|`, identical `p(1−p)`, **9× different
worst case.**

The programme's §8 rule is that the cap **evaluates scenario PnL directly**
rather than summing signed loadings. Under a scenario constraint the **side** of
the imbalance is first-order and `p(1−p)` is the wrong statistic entirely.

**Consequence.** The state object must carry **signed** net and the execution
price. A limit on `|net|` shares is not a risk limit here.

---

## 1. The state object

**Unit: one `(coin, window)` market.** Not per coin, not portfolio-level, and
the reason is structural rather than convenient — **settlement is per window**.
Each 5-minute market resolves independently, so risk does not carry across
windows and there is nothing to net between them. A position in the 12:05 BTC
window and one in the 12:10 BTC window are different instruments that happen to
share an underlying.

```
InventoryState(coin, window):
    net_up_shares      signed; Down shares enter as NEGATIVE Up-equivalents
    cost_basis_usdc    signed; what was actually paid, not marked
    r_seconds          time to settlement, the hard deadline
    last_exec_price    for the side-aware risk metric (0.3)
    fills              [(t, side, size, price)]  — audit, not state
```

**Down shares are negative Up shares.** By §0.1 that is an identity, not a
modelling choice, so the state is one signed scalar and not a pair.

**Cross-window and cross-coin exposure is a PORTFOLIO question, owned by
`DE-Allocator`**, and is deliberately out of scope here. Overlapping windows of
the same coin *do* share an underlying and their residuals will correlate; that
belongs in the scenario model, which must declare which instruments lose
together rather than summing betas (§8, `R-ONCE`).

**Merging complete sets is a CAPITAL operation, not a risk one.** Up + Down = $1
either way, so merging changes collateral, not exposure, and never appears in
this model's PnL. `DE-Allocator` owns it.

---

## 2. The mechanisms, and the switching rule

Three actions, of which §0.1 collapses two into one:

| | action | fee | spread paid | fill certainty | needs fair price? |
|---|---|---|---|---|---|
| **A** | **Skew** — shade the two-sided quote toward the offsetting side | **0** | earns, does not pay | **uncertain** | **no** |
| **B** | **Dump** — cross to flatten | `0.07·p(1−p)`/sh | pays `h` | certain | no |
| **C** | **Hold** to resolution | 0 | 0 | n/a | no |

`A` and "quote the complement" are the same act (§0.1).

**The asymmetry that should drive the design.** At ATM a dump costs
`1.75 + 0.50 = 2.25 ¢/share` against a half-spread capture of `0.50 ¢`. **One
forced unwind destroys roughly 4.5 fills' worth of gross capture** — and the
maker edge sign is undetermined, so those fills may not have been profitable to
begin with. **Dumping must be rare by construction, not by tuning.**

**Proposed switching rule, stated as a hypothesis to be falsified, not a result:**

```
if |net| ≤ N_skew(p, r):          A — skew only
elif |net| ≤ N_dump(p, r):        A — skew harder, widen the far side
else:                             B — dump the excess, keep |net| at N_dump
at r → r_terminal:                see §3
```

with `N_dump(p, r) = (2/γ)·[0.07 + h/(p(1−p))]` from §0.2, **increasing in the
tails**, and the side-aware cap of §0.3 binding independently: reject any state
whose scenario loss `net × last_exec_price` (for long Up) exceeds the limit,
regardless of `p(1−p)`.

**`γ` is a free parameter and this plan does not set it.** It is a risk-appetite
choice, not a measurable, and pretending otherwise would be the third unfailable
gate this programme has written. Report results across a `γ` ladder.

---

## 3. The terminal condition, where the measured data bites

The deadline is hard and the liquidity to meet it is **measured to collapse**:

- count intensity falls to ~18 % of peak in the final bins;
- **mean size per arrival doubles-to-triples** (btc 15.5 → 24.0 USDC/arrival);
- notional **peaks in the first 5 s inside `r = 60`** then declines monotonically
  **9.5×** to settlement.

**So the terminal regime is few, large, and thinning — the worst possible
combination for an unwind, arriving exactly when the deadline is nearest.**

Two consequences the plan must encode:

1. **A design that defers unwinding to late `r` is assuming something measured to
   be false.** Passive offset (`A`) becomes progressively less available exactly
   when it is most needed.
2. **There is a measured window where unwinding is cheapest: the notional peak at
   `r ≈ 60`.** If a position is going to be reduced at all, the data says reduce
   it *before* the collapse, not during it.

**Proposed terminal policy, again a hypothesis:**

```
r > 60      normal regime; skew-first, dump only above N_dump
r ≈ 60      DECISION POINT — the last moment with peak notional available.
            Reduce |net| to the size intended to be carried to resolution.
r < 60      no new risk; skew passively; DO NOT plan to dump —
            the liquidity assumed by a late dump is not there
r → 0       carry the residual. It resolves. This is a CHOICE made at r ≈ 60,
            not a failure to act
```

**This makes "hold to resolution" the default terminal state and the `r ≈ 60`
decision the only real one.** That is a direct consequence of the measured
collapse, not a preference.

---

## 4. Identified vs assumed

| # | quantity | status | note |
|---|---|---|---|
| 1 | `net`, `cost_basis` | **IDENTIFIED** | pure accounting from own fills |
| 2 | dump fee `0.07·p(1−p)` | **IDENTIFIED** | n=600, 4 dp, taker pays |
| 3 | half-spread `h` | **IDENTIFIED** | 0.50 ¢ flat, btc/eth, 2.29 M quotes |
| 4 | terminal liquidity profile | **IDENTIFIED** | 361 windows/coin |
| 5 | resolution payoff | **IDENTIFIED** | `S60(T)` vs `S60(t0)`, 99.8 % |
| 6 | **offset fill probability** for skew `A` | **PARTIAL** | fill model gives it per placement policy; **queue position is a policy output, not an unknown** |
| 7 | **imbalance process** — do two-sided fills arrive balanced? | **NOT MEASURED** | **the central open question; see §5** |
| 8 | `γ` risk appetite | **ASSUMED** | a choice, not a measurable. Ladder it |
| 9 | reservation-price skew around fair value | **BLOCKED** | needs Route A, on **PRICING HOLD**. §2's `A` is an *inventory* skew — shading a quote toward the side that reduces `net` — and needs **no fair price**. A skew that recentres on a fair value is a different object and is unavailable |
| 10 | edge of a fill | **BROKEN ESTIMAND** | markout-vs-settlement measures hold-to-expiry drift. Layer 1 must be re-specified at a fixed horizon against book mid; this plan is Layer 2 and consumes whatever Layer 1 produces |

**Nothing in §1–§3 requires a fair price.** That is deliberate and it is what
keeps this buildable while Route A is on hold.

---

## 5. Measurable today vs waiting

**Today, on the current tape, no new collection:**

- **The imbalance process (#7) — the highest-value open item.** Simulate a
  two-sided quote at the touch, replay the tape, and record `net(t)`. Does it
  random-walk, mean-revert, or drift? Everything in §2 is parameterised on this
  and **nobody has measured it.** If two-sided fills arrive roughly balanced,
  inventory is a minor problem and this plan is mostly unnecessary. If `net`
  random-walks, the dump frequency is set by the walk's variance and the whole
  cost asymmetry of §2 becomes binding.
- The cost of a dump at each `r`, from the measured liquidity profile.
- The availability of passive offset by `r` — how often the offsetting side fills
  within a horizon, per placement policy.
- Whether the `r ≈ 60` peak is actually the cheapest reduction point, or an
  artefact of the same 60 s structure that is confounded elsewhere.

**Waiting on other work:** anything using per-fill edge (Layer 1 re-spec);
anything using a fair value (Route A); anything day-clustered (~10 days).

---

## 6. What would falsify this design

1. **The imbalance is self-correcting.** If simulated `net` mean-reverts without
   intervention, mechanisms `B` and the whole switching rule are unnecessary
   machinery. **This is the most likely falsification and it should be tested
   first.**
2. **`bid(U) + ask(D) = 1` fails off-ATM or on thin coins.** §0.1 is verified on
   one BTC window; if the identity is venue-wide it is structural, if it is
   BTC-only the one-book claim needs qualifying. Cheap to check on all seven.
3. **The `r ≈ 60` decision point is an artefact.** The terminal structure is
   confounded with the unidentified 60 s component; if the notional peak moves
   under the non-uniform grid, §3's timing is wrong.
4. **`N*` ordering reverses under a scenario cap.** §0.2 derives the threshold
   under mean-variance; §0.3 says the programme's actual constraint is
   scenario-based and side-aware. **These two may disagree, and if they do, §0.3
   wins** — the mean-variance result would then be a diagnostic, not the rule.
5. **Two-sided quoting is not available in practice.** The plan assumes a maker
   can rest on both sides simultaneously; if the venue or capital constraints
   prevent it, `A` collapses and only `B` and `C` remain.

---

## 7. What this plan deliberately does not do

- **Assume profitability.** The maker-edge sign is `+0.173 [−0.251, +0.596]` —
  undetermined. Nothing here is conditioned on it being positive.
- **Set `γ`.** A risk-appetite parameter reported as a measurement would be a
  gate that cannot fail.
- **Model competition.** Other makers affect offset fill probability; that is
  `BE-Competition`, unbuilt, and this plan consumes it as an input rather than
  inventing it.
- **Cross windows or coins.** `DE-Allocator` owns portfolio exposure, and the
  scenario model must declare what loses together rather than summing betas.
