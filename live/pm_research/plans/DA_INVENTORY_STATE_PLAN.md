# Inventory STATE — model plan

Plan only. No code, no measurement, no fitted quantity. Status: **DESIGN**, not
decision-eligible. Written 2026-08-22, split 2026-08-22.

> For current programme state read [`FLOW_MODEL_STATE.md`](../FLOW_MODEL_STATE.md).
> This document is a **plan** — a design under constraints, not a statement of
> measured belief. Where it conflicts with that page, that page wins.

---

## 0. Plane assignment, and a misnomer in this file's own name

**This file is named `DA_INVENTORY_STATE_PLAN.md` and there is no `BE-Inventory`
module.** `PM_ARCHITECTURE.md` §1 defines the belief plane as *Target ·
Uncertainty · Belief · FlowAndFills · Competition · ScenarioProvider*. Inventory
appears in none of them. The filename is retained only because three artifacts
already reference it — `inventory_walk.py`, `INVENTORY_WALK_RESULTS.md` and
`FLOW_MODEL_STATE.md` — and a dangling reference is worse than a wrong prefix.
**Renaming to `DA_INVENTORY_STATE_PLAN.md` is the correct fix and is left to the
coordinator.** Recorded as a fifth instance of *the name is not the definition*.

Inventory is **three things on three planes**, and the earlier version of this
document ran them together:

| concern | plane | owner | where specified |
|---|---|---|---|
| what the position **is** | `DA` | `DA-State` / `SelfState` | **this document** |
| what **limits** it | `SP` / `DE` | `SP-Params` · `DE-Constraints` | this document §4, enforced in DE |
| what to **do** about it | `DE` | `DE-DecisionScheme` | [`DE_PLACEMENT_POLICY_PLAN.md`](DE_PLACEMENT_POLICY_PLAN.md) |

**`BE-FlowAndFills` is inventory-agnostic and must stay that way.** It answers
`P(fill | placement, market state)` — conditioning on market observables only. It
does not know our position, and the code already respects this. The dependency
rule is **SP ← DA ← BE ← DE**: DE may read BE, BE must never read DE. An
inventory term inside the fill model would create a `DE → BE` edge and break the
plane ordering. **Do not add one.**

---

## 1. Down shares are negative Up shares — an identity

**Verified on the tape:** across 37,394 consecutive `price_change` messages in a
BTC window, both tokens are quoted in **every single message**, and

```
bid(Up) + ask(Down) = 1.0000      p10 = p50 = p90 = 1.0000
                                  within 0.005 of 1.0 : 100.0 %
```

`ask(Down) ≡ 1 − bid(Up)`. There are not two books; there is **one book with two
representations**. See `FLOW_MODEL_STATE.md` §1 for the canonical statement.

**Consequence for state, and it is the only consequence claimed here:** the
position is **one signed scalar**, not a pair. A complete set — equal size on
both tokens — is worth exactly $1 at resolution with no direction and no unwind.

The *trading* consequences of this identity (that skew and complement-quoting are
one mechanism, and that the capture is one spread not two) are **decisions** and
live in the DE plan.

---

## 2. Risk is NOT symmetric in `p`, and `p(1−p)` hides it

Worst-case loss on a long is **the price paid**. Long 100 Up at `p = 0.90` risks
**$90**; long 100 Down at the same `p` — Down at 0.10 — risks **$10**. Identical
`|net|`, identical `p(1−p)`, **9× different worst case.**

The programme's §8 rule is that the cap **evaluates scenario PnL directly**
rather than summing signed loadings. Under a scenario constraint the **side** of
the imbalance is first-order and `p(1−p)` is the wrong statistic entirely.

**Consequence for state:** it must carry **signed** net and the execution price.
**A limit on `|net|` shares is not a risk limit here.** Any downstream rule
keyed on `|net|` alone is mis-specified, including in the DE plan.

---

## 3. The state object

**Unit: one `(coin, window)` market.** Not per coin, not portfolio-level, and
the reason is structural rather than convenient — **settlement is per window**.
Each 5-minute market resolves independently, so risk does not carry across
windows and there is nothing to net between them. A position in the 12:05 BTC
window and one in the 12:10 BTC window are different instruments that happen to
share an underlying.

```
InventoryState(coin, window):
    net_up_shares      signed; Down shares enter as NEGATIVE Up-equivalents (§1)
    cost_basis_usdc    signed; what was actually paid, not marked
    r_seconds          time to settlement, the hard deadline
    last_exec_price    for the side-aware risk metric (§2)
    fills              [(t, side, size, price)]  — audit, not state
```

**Cross-window and cross-coin exposure is a PORTFOLIO question owned by
`DE-Allocator`** and is out of scope here. Overlapping windows of the same coin
*do* share an underlying and their residuals will correlate; that belongs in the
scenario model, which must declare which instruments lose together rather than
summing betas (§8, `R-ONCE`).

**Merging complete sets is a CAPITAL operation, not a risk one.** Up + Down = $1
either way, so merging changes collateral, not exposure, and never appears in
this model's PnL. `DE-Allocator` owns it.

**Scope: btc and eth only.** Three independent measurements converge on that
restriction — bracket width, micro-actor contamination, and inventory (see §5).
State may be *recorded* on all seven coins; it may not be *acted on* elsewhere.

---

## 4. Identified vs assumed — STATE quantities only

Decision-side quantities (`γ`, `N*`, offset fill probability) are in the DE plan.

| # | quantity | status | note |
|---|---|---|---|
| 1 | `net`, `cost_basis` | **IDENTIFIED** | pure accounting from own fills |
| 2 | `bid(U) + ask(D) = 1` identity | **IDENTIFIED** | 37,394/37,394, one BTC window; see §5 for the open generalisation |
| 3 | side-aware worst case | **IDENTIFIED** | `net × last_exec_price`, an accounting identity |
| 4 | resolution payoff | **IDENTIFIED** | `S60(T)` vs `S60(t0)`, 99.8 % on 1,465 windows |
| 5 | **the imbalance process** | **MEASURED 2026-08-22** | **does NOT self-balance** — see §5 |
| 6 | scenario loss limit | **ASSUMED** | a risk-appetite choice owned by `SP-Params`, not a measurable |
| 7 | edge of a fill | **BROKEN ESTIMAND** | markout-vs-settlement measures hold-to-expiry drift, not spread capture. Layer 1 must be re-specified at a fixed horizon against book mid; inventory PnL is Layer 2 and consumes whatever Layer 1 produces |

**Nothing in §1–§3 requires a fair price.** Route A is on `PRICING HOLD` and this
document is unaffected by it.

---

## 5. The imbalance process — MEASURED, and it does not self-balance

The previous version of this plan listed this as *"the central open question,
nobody has measured it"* and named it the most likely falsification. **It has
been measured** (`inventory_walk.py`, `INVENTORY_WALK_RESULTS.md`) and the plan
survives.

**No coin returns `SELF_BALANCING`.** Mean-reversion half-lives measure
**519–2726 s**, every one **longer than the 300 s window**, so even where
reversion is detected it is far too slow to matter inside a market.

*The obvious confound was tested and refuted rather than assumed:* sub-linear
variance scaling could have been manufactured by the measured terminal collapse,
but body-only `beta` at `t ≤ 240 s` is **smaller** on every coin, not larger
(btc 0.652 → 0.568). The scaling is real.

**Two-sided quoting offsets in proportion to fill frequency, and on thin coins it
is counterproductive.** Terminal `|net|` two-sided ÷ one-sided, same windows:

| coin | fills/window | ratio |
|---|---:|---:|
| btc | 317.6 | **0.101** |
| eth | 62.5 | **0.199** |
| xrp | 10.3 | 0.663 |
| doge | 2.2 | 1.173 |
| hype | 1.9 | **1.752** |

At ~2 fills/window the sides rarely pair, so the second quote adds **unpaired**
fills rather than offsetting ones. This is the **third independent** argument for
the btc/eth restriction, reached from inventory rather than from bracket width or
micro contamination.

**A defect in that test's own rule, recorded not applied:** its terminal-band
criterion cannot distinguish *balanced* from *inactive*. bnb and doge sit at
exactly the band and would have passed, but their two-sided ÷ one-sided ratio is
~1.1 — their `|net|` is small because almost nothing trades. **A rule keyed on
terminal size alone rewards illiquidity.** Any future state-side gate must pair a
size criterion with an activity criterion.

---

## 6. Measurable today vs waiting

**Today, on the current tape:**

- **Does `bid(U) + ask(D) = 1` hold venue-wide?** §1 is verified on one BTC
  window. If the identity is venue-wide it is structural; if BTC-only, the
  one-book claim needs qualifying and the single-scalar state representation
  with it. Cheap to check on all seven coins and it is the largest open state-side
  question.
- The realised distribution of terminal `|net|` and its **side-aware cash at
  risk**, per coin, under any stated placement policy.

**Waiting:** anything using per-fill edge (Layer 1 re-spec); anything using a
fair value (Route A, on hold); anything day-clustered (~10 days).

---

## 7. What would falsify this design

1. **`bid(U) + ask(D) = 1` fails off-ATM or on thin coins.** Then the state is
   not one scalar and §3 is wrong. **This is now the most likely falsification**,
   the imbalance-process one having been tested and survived.
2. **`net` proves not to be the risk-relevant state** — e.g. if correlated
   residuals across overlapping windows of the same coin dominate the
   single-market residual. Then the unit in §3 is wrong and the portfolio
   question cannot be deferred to `DE-Allocator`.
3. **A scenario cap and a `|net|` cap order states differently.** §2 says the
   side is first-order; if a downstream rule keyed on `|net|` ever disagrees with
   the scenario cap, **§2 wins** and the `|net|` rule is a diagnostic.

## 8. What this plan deliberately does not do

- **Decide anything.** Mechanisms, thresholds and the terminal decision are in
  [`DE_PLACEMENT_POLICY_PLAN.md`](DE_PLACEMENT_POLICY_PLAN.md).
- **Assume profitability.** The maker-edge sign is `+0.173 [−0.251, +0.596]` —
  undetermined.
- **Set a risk limit.** That is `SP-Params`, a choice rather than a measurable.
- **Cross windows or coins.** `DE-Allocator` owns portfolio exposure.
