# DE-Placement policy — model plan

Plan only. No code, no measurement, no fitted quantity. Status: **DESIGN**, not
decision-eligible. Written 2026-08-22 by splitting the decision content out of
[`DA_INVENTORY_STATE_PLAN.md`](DA_INVENTORY_STATE_PLAN.md).

> For current programme state read [`FLOW_MODEL_STATE.md`](../FLOW_MODEL_STATE.md).
> This document is a **plan**. Where it conflicts with that page, that page wins.

---

## 0. Plane, and the boundary this document must not cross

This is `DE-DecisionScheme` content: **what to do**, given state and beliefs.

| input | plane | owner |
|---|---|---|
| the position | `DA` | `DA-State` / `SelfState` — see [inventory state plan](DA_INVENTORY_STATE_PLAN.md) |
| `P(fill \| placement, market state)` | `BE` | `BE-FlowAndFills` |
| the risk limit | `SP` | `SP-Params`, enforced by `DE-Constraints` |
| **the placement decision** | `DE` | **this document** |

**`BE-FlowAndFills` is inventory-agnostic.** It conditions on market observables
only and does not know our position. The dependency rule is **SP ← DA ← BE ← DE**:
DE may read BE, BE must never read DE. **This document consumes the fill model;
it must never ask for an inventory term to be added to it.** Skew is implemented
here, by choosing *where* to place given `net` — not by teaching the fill model
about `net`.

**Scope: btc and eth only** (`verdict_coins`). Three independent measurements
converge on it — bracket width, micro-actor contamination, and inventory, where
two-sided quoting is actively counterproductive on thin coins (ratios 1.17 doge,
1.75 hype).

---

## 1. The menu, and the asymmetry that should drive the design

Two mechanisms plus doing nothing. **Skew and "quote the complement" are the same
act** — the two books are one book — so a menu listing them separately
double-counts a single action.

| | action | fee | spread | fill certainty | needs fair price? |
|---|---|---|---|---|---|
| **A** | **Skew** — shade the two-sided quote toward the offsetting side | **0** | earns | **uncertain** | **no** |
| **B** | **Dump** — cross to flatten | `0.07·p(1−p)`/sh | pays `h` | certain | no |
| **C** | **Hold** to resolution | 0 | 0 | n/a | no |

**The asymmetry.** At ATM a dump costs `1.75 + 0.50 = 2.25 ¢/share` against a
half-spread capture of `0.50 ¢`. **One forced unwind destroys roughly 4.5 fills'
worth of gross capture** — and the maker-edge sign is undetermined, so those
fills may not have been profitable to begin with. **Dumping must be rare by
construction, not by tuning.**

---

## 2. The dump threshold, and why the tails tolerate MORE

Cost per unit of **variance** removed is constant at 700 across all `p`; per unit
of **standard deviation** it peaks at ATM. Both are true and **neither is the
decision metric** — cost-per-sd prices risk *removal* while ignoring risk *held*.
The risk penalty is **quadratic** in position and the cost is **linear**, so the
decision turns on size.

Under a mean-variance penalty `(γ/2)·N²·p(1−p)` against a dump cost
`N·[0.07·p(1−p) + h]`:

```
dump when   N > N*(p) = (2/γ) · [ 0.07 + h / (p(1−p)) ]
```

Half-spread `h` is **measured flat at 0.50 ¢** across all moneyness (btc/eth,
1 tick, 2.29 M quotes) while `p(1−p)` shrinks in the tails, so `h/(p(1−p))`
**grows** there and `N*` grows with it:

```
p = 0.50   N* = (2/γ)(0.07 + 0.020) = 0.090 · (2/γ)
p = 0.10   N* = (2/γ)(0.07 + 0.056) = 0.126 · (2/γ)      1.4x higher
```

**Tolerate a LARGER imbalance in the tails; dump MORE READILY at ATM.** A share
in the tails carries less risk to begin with, and the quadratic penalty means
more of them fit under the same budget.

**`γ` is a free parameter and this plan does not set it.** It is a risk-appetite
choice, not a measurable, and reporting it as one would be another gate that
cannot fail. Report results across a `γ` ladder.

**The side-aware cap of the state plan §2 binds independently and wins on
conflict.** `N*` is derived under mean-variance; the programme's actual
constraint is scenario-based and side-aware. Where they disagree, the scenario
cap governs and `N*` is a diagnostic.

---

## 3. Static policies: what is already measured

Two static placements have been simulated end to end.

| | `JOIN_BBO` | `NEW_BBO` |
|---|---|---|
| queue position | behind displayed depth | front |
| fill rate (btc, 15 s) | 76.9 % | 94.6 % |
| `net` behaviour | sub-linear, `beta` 0.29–0.65 | **random walk**, `beta` 0.908–0.996 |
| btc terminal `\|net\|` p95 | 191.8 sh / $121.80 | **1805.1 sh / $541.76** |

**`NEW_BBO` carries ~9.4× the inventory risk** because it fills on **every**
reaching trade and absorbs one-sided bursts whole. Behind the queue you fill only
on sweeps, so **the queue acts as a filter that skips exactly the directional
flow that accumulates.**

**The earlier policy comparison could not see this.** It measured fill and
markout and found `NEW_BBO` ahead on fills. Front-of-queue **buys fill rate and
pays for it in inventory risk**. **Any placement comparison needs inventory risk
as a third axis**, and the earlier one was generous to `NEW_BBO`.

---

## 4. The state-dependent policy — the object this plan exists to specify

Everything measured so far compared **static** policies. The obvious next object
is one conditioned on `net`:

```
long Up    ->  reducing side (sell Up)  at NEW_BBO    front, ~94.6 % fill
               adding side  (buy  Up)   at JOIN_BBO   back,  ~76.9 % fill

short Up   ->  mirrored

near flat  ->  both at JOIN_BBO
```

This is the classic maker skew implemented **through placement rather than
price** — and placement is the only lever with resolution here, because the tick
is **1 cent on btc/eth** so there is frequently no room to skew on price at all.

**The key asymmetry, and it inverts a measured finding.** `NEW_BBO`'s 9.4× risk
comes from filling on every reaching trade and absorbing directional bursts
whole. That is a **liability when flat** and **exactly what you want when
reducing**. Same property, opposite sign, conditioned on inventory state. A
policy that switches on `net` harvests the good half.

**Free parameters, none of which this plan sets:** the flat-band width, whether
the switch is binary or graded across `|net|`, and whether the adding side is
merely demoted to `JOIN_BBO` or withdrawn entirely at large `|net|`.

**What it costs.** Quoting asymmetrically gives up spread capture on the adding
side. That is the trade being bought: less edge per round trip, less terminal
residual. **Whether it is worth it cannot be evaluated yet** — the edge estimand
is broken and the sign is undetermined.

---

## 5. The measured half-life is a BASELINE, not a bound

`net` has been measured to **not** self-balance: mean-reversion half-lives of
**519–2726 s**, all longer than the 300 s window.

**That is the UNCONTROLLED process** — symmetric quoting, no skew, reversion
arising only from incidental pairing. **It is not a ceiling on what a controlled
policy achieves**, and reading it as one would be a category error:

- a **reversion rate** is inherited from how flow happens to pair;
- a **drift** is imposed by asymmetric placement, and it is **sized by the policy**.

Asymmetric placement adds a drift toward zero. Drift is a different and stronger
mechanism than incidental pairing, and at the extreme — quoting only the reducing
side — the pull is maximal.

**Whether that drift beats the 300 s deadline is UNMEASURED.** This plan does not
assume either way, and §7 names it as the first test. A plausible negative result
is that the fill-rate ratio between placements (94.6 / 76.9 ≈ 1.23) is too small
a lever to move a 519 s half-life inside 300 s — in which case the terminal
decision of §6 stands unchanged and the dump mechanism remains load-bearing.

---

## 6. The terminal condition, where the measured data bites

The deadline is hard and the liquidity to meet it is **measured to collapse**:

- count intensity falls to ~18 % of peak in the final bins;
- **mean size per arrival doubles-to-triples** (btc 15.5 → 24.0 USDC/arrival);
- notional **peaks in the first 5 s inside `r = 60`**, then declines monotonically
  **9.5×** to settlement.

**The terminal regime is few, large and thinning — the worst combination for an
unwind, arriving exactly when the deadline is nearest.**

```
r > 60      normal regime; skew-first, dump only above N*
r ≈ 60      DECISION POINT — the last moment with peak notional available.
            Reduce |net| to the size intended to be carried to resolution.
r < 60      no new risk; skew passively; DO NOT plan to dump —
            the liquidity a late dump assumes is measured not to be there
r → 0       carry the residual. It resolves. A CHOICE made at r ≈ 60,
            not a failure to act
```

**This makes hold-to-resolution the default terminal state and `r ≈ 60` the only
real decision.** A direct consequence of the measured collapse, not a preference.

**Caveat that could move the timing:** the terminal structure is entangled with
an unidentified 60 s component, and the notional peak's location was measured on
a grid that has since been replaced. If the peak moves under the non-uniform
grid, §6's timing is wrong even though its logic is not.

---

## 7. Measurable today, in priority order

**Today, on the current tape, no new collection:**

1. **Does the state-dependent policy's drift beat 300 s?** Replay the tape with
   placement conditioned on `net(t)`; compare terminal `|net|` and side-aware
   cash at risk against both static policies. **This is the first test.** If the
   drift clears the residual inside the window, the dump mechanism may be
   unnecessary after all; if not, §6 stands unchanged.
2. **Re-run the placement comparison with inventory risk as a third axis.** Fill
   and markout alone were shown to be insufficient (§3).
3. Whether `r ≈ 60` is genuinely the cheapest reduction point under the
   non-uniform grid, or an artefact of the confounded 60 s structure.
4. The availability of passive offset by `r` — how often the reducing side fills
   within a horizon, per placement.

**Waiting:** anything using per-fill edge (Layer 1 re-spec); anything using a
fair value (Route A, on hold); anything day-clustered (~10 days).

---

## 8. What would falsify this design

1. **The state-dependent drift is too weak.** If conditioning placement on `net`
   does not materially reduce terminal `|net|` versus `JOIN_BBO`, §4 is
   machinery for nothing and only §2 and §6 survive.
2. **`N*` ordering reverses under the scenario cap.** §2 derives it under
   mean-variance; the state plan §2 says the binding constraint is side-aware.
   **If they disagree, the scenario cap wins** and `N*` becomes a diagnostic.
3. **The `r ≈ 60` decision point is an artefact** of the confounded 60 s
   structure rather than a real liquidity peak.
4. **Two-sided quoting is unavailable in practice.** The plan assumes a maker can
   rest on both sides simultaneously; if venue or capital constraints prevent it,
   `A` collapses and only `B` and `C` remain.
5. **The fill-rate lever is too small.** 94.6 / 76.9 ≈ 1.23 may not be enough
   asymmetry to impose a useful drift, whatever the theory says.

## 9. What this plan deliberately does not do

- **Define the state.** That is the [inventory state plan](DA_INVENTORY_STATE_PLAN.md).
- **Ask for an inventory-aware fill model.** That would break the plane
  ordering. Skew is implemented here, by placement.
- **Assume profitability.** The maker-edge sign is undetermined and the edge
  estimand is currently broken.
- **Set `γ`, the flat band, or the risk limit.** All choices, not measurables.
- **Use a fair price.** A reservation-price skew — recentring quotes on a fair
  value — is a different object, needs Route A, and is **blocked**. Everything
  here is an *inventory* skew and needs no fair price. That is much of why it is
  worth specifying now.
- **Model competition.** Other makers affect offset fill probability; that is
  `BE-Competition`, unbuilt, consumed as an input rather than invented here.
