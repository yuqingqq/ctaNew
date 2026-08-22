# FLOW MODEL — CURRENT STATE

**Read this first, and read nothing else to find out what we currently believe.**
Updated 2026-08-21. Supersedes every other flow document *for the question "what
is true now"*. The others remain valid as provenance — how we got here, and why
particular things were withdrawn — but they argue with each other by
construction, because they were written in sequence as claims were corrected.

If a statement anywhere else conflicts with this page, **this page wins** and
the other document is stale. Say so rather than reconciling it privately.

---

## 0. Where things live — plane ownership

Dependency rule (`PM_ARCHITECTURE` §1): **SP ← DA ← BE ← DE.** DE may read BE;
**BE must never read DE.** EV reads all planes and is read by none.

```
SP  SPECS      venue · instrument · params            risk LIMITS declared here
DA  DATA       discovery ✅ feeds ✅ normalize 🔨
               state 🔨 settlement 📋                  INVENTORY STATE here
BE  BELIEF     Target ⏳ + Uncertainty ⏳ → Belief ⏳    = FAIR PRICE
               FlowAndFills 🔬                          = FILL MODEL
               Competition ❌ ScenarioProvider ❌
DE  DECISION   ActionSpace ❌ Constraints ❌
               DecisionScheme ❌ Allocator ❌ Actuator ❌ INVENTORY POLICY here
EV  EVAL       Markout 📋 Calibration 📋 rest ❌
OP  OPS        LatencyBudget · Monitor/KillSwitch       partial (supervision)
```

`⏳` on the sigma clock · `🔬` characterised, edge estimand broken · `🔨` in
progress · `📋` queued · `❌` not built

**Inventory is THREE things in THREE planes, and conflating them creates the
forbidden `DE → BE` edge:**

| question | plane | artifact |
|---|---|---|
| what do I hold? | `DA-State` / `SelfState` | `plans/DA_INVENTORY_STATE_PLAN.md` |
| what may I hold? | `SP-Params` limit, enforced at `DE-Constraints` | scenario cap, §8 |
| what do I do about it? | **`DE-DecisionScheme`** | `plans/DE_PLACEMENT_POLICY_PLAN.md` |

**The third one IS the placement policy.** Skew, dump and hold are decisions, and
skew is expressed through placement. There is no separate "inventory module".

**`BE-FlowAndFills` is INVENTORY-AGNOSTIC and must stay so.**
`P(fill | placement, market state)` conditions on market state only — it does not
know our position. Adding an inventory term to it would make BE depend on DE and
violate §1. The code already respects this; it is written down so nobody adds one.

---

## 1. Established, with scope

| fact | evidence | scope |
|---|---|---|
| **`side` is the taker's** | G-FF1 `PASS`, 600/600, Wilson [0.9936, 1.0000] | 300 BUY / 300 SELL, 7/7 coins, 5/5 moneyness |
| **Taker pays the fee, maker does not** | `0.07·p·(1−p)` $/share to 4 dp; 600/600 taker legs charged, 744/754 maker legs zero | n=600 transactions |
| **Crossing costs ~2.25 ¢/share ATM** | 0.50 ¢ half-spread + 1.75 ¢ fee ≈ **225 bps** on a $1 binary | btc/eth |
| **Settlement is `S60(T)` vs `S60(t0)`** | 99.8 % on 1,465 windows; the 300 s reading is refuted at 86.9 % | all |
| **ATM spread is 1 tick on btc/eth** | median 0.0100, p90 0.020, 2.29 M executable quotes | btc; ticks by coin: btc 1 · eth 1 · sol 3 · doge 3 · xrp 5 · bnb 5 · hype 7 |
| **The 1-cent spread is a CONSTRAINT, not a convention** | where the 0.001 tick is available the spread is 1 tick in 99.9 % of quotes | btc |
| **0.001 tick exists only in the tails** | 6.75 % at `p<0.15`, 6.73 % at `p≥0.85`, **0 % in the middle three buckets** | btc |
| **One address is a large share of arrivals** | 0.02 shares exactly, 99.98 % SELL, all seven coins, **0.0145 % of notional** | **per coin 2.0 % (btc) → 90.0 % (hype)**; the pooled 16.3 % is btc-dominated |
| **That address is NOT independent of market flow** | A1 fails on all 7; ratio 1.75–2.79, p=0.000; direction **bidirectional/common-driver** | τ=0.25 s, circular-shift null |
| **`f_r` does not rise into settlement** | count: flat then terminal collapse. notional: rises, peaks, then falls — **peak located to the first 5 s inside `r=60`** (btc notional 97→119 through the body, **170.4** at the peak, then a monotone **9.5×** decline to 18.0) | 361 windows/coin, `clob_v3_1` |
| **The terminal minute contains essentially ALL of `f_r`'s dynamic range** | on btc/eth/sol/doge the terminal collapse ratio **equals the full-profile shape ratio exactly**; six of seven coins on notional | 361 windows/coin |
| **The terminal regime is FEW AND LARGE** | count drops to 18 % of peak while notional holds 28 %; USDC/arrival 15.5 → 24.0 (btc), 11.9 → 32.3 (eth) | btc/eth |
| **Book state carries real information** | B3 placebo does not reproduce the gain on any coin (btc −0.03 share, hype 0.02) | 24 windows/coin |
| **THE TWO BOOKS ARE ONE BOOK — and the second side is DERIVED, not quoted** | **1,081,800 checks across 560 archives, 7 coins, 4 UTC days: ZERO violations, worst deviation exactly 0.00000.** Zero in the 0.001 tick regime (8,468), the terminal minute (23,828), during gaps, and within 5 s of a tick change | **all coins, all days** — the exactness is the finding: two books agreeing would show float noise, one book computed from the other shows none |
| **A complete set is therefore worth ONE spread, not two** | paying `Up_bid + Down_bid = 1 − spread` for something worth exactly $1 | corrects an earlier coordinator claim |
| **Market self-excitation is not deletable** | bivariate 2×2 **refit 2026-08-22 with the instrument floor and continuous optimiser**: diagonal **0.236–0.477** dominates cross **0.000–0.167** on every coin; btc `market←market` **0.477 [0.282, 0.519]**, eth **0.389 [0.339, 0.431]** | **no coin censored**; both estimators refined, not grid-seeded |
| **Clustering runs at 80–218 ms, and the censoring was OUR GRID** | scalar Hawkes re-fit 2026-08-21 with the instrument floor + continuous optimiser: **no coin is censored**, branching 0.33–0.55, half-life 80.8 ms (btc) to 217.7 ms (hype) — **81×–218× the venue tick** | 24 windows/coin, `clob_v3_1` |

## 1a. The binning decision is REFUTED, and the factor would have eaten real signal

`SPEC_REV2` §2 chose **option (d)** — 15 s bins plus an explicit 4-level phase
factor — on the **untested assumption that the unidentified 60 s component is
ADDITIVE**, i.e. separable from `f_r`. I ratified that choice. The pre-registered
`phase × r` interaction test has now run and it is **`INTERACTION_MATERIAL` on
all seven coins**:

| coin | `rms(interaction)/rms(phase main effect)` | CI95 |
|---|---:|---|
| btc | **1.685** | [1.379, 2.072] |
| eth | **1.660** | [1.268, 2.230] |
| sol | 2.377 | [1.776, 3.524] |
| hype | 4.486 | [3.352, 6.055] |

The material bar is 0.50; every lower bound clears it by 2.5×+, and **every ratio
exceeds 1.0** — the non-separable structure is *larger than the effect the factor
exists to remove*.

**And the reason is worse than non-separability.** The interaction concentrates
almost entirely in the **terminal minute** — btc residual RMS by cycle
`0.049, 0.074, 0.086, 0.121 → 0.568`, so the last cycle carries **4.7×** the next
largest (eth 5.7×) — and inside it the pattern is a monotone collapse
(btc `+0.404, +0.165, −0.174, −1.034`, ≈4.2× decline). That is not the
oscillation interacting with `r`. It is **`f_r` having genuine structure finer
than 60 s in the terminal minute**, which a 5-level cycle factor cannot express,
so the additive model attributes the terminal collapse to *phase*.

**A factor labelled `unidentified_60s_component` would therefore be EATING
`f_r`'s largest real feature, not removing an artefact.** That is strictly worse
than failing to separate, and it is why this must not ship.

Excluding the terminal minute does not rescue additivity (btc 1.689 → 0.357,
eth 1.615 → 0.369, both still above the 0.25 bar with intervals reaching past
0.50). Option (b), 60 s bins, is also wrong: it puts the entire terminal collapse
inside a single bin. **Indicated replacement: a NON-UNIFORM grid, coarse in the
body and fine in the terminal** — the shape §1.3 already chose for Tier B, for
the opposite reason.

## 1b. WHAT "EDGE MEASURED AGAINST SETTLEMENT" ACTUALLY IS — read before citing any edge number

The policy comparison (`policy_v1`) returned `UNRESOLVED`, but its useful output
is a **defect in the estimand that every edge figure in this programme shares**.

Markout is measured against **settlement** (`S60(T)`), a choice made to keep the
work off the fair-price model and the sigma clock. The consequence was not
thought through: **a fill at `t=30 s` is marked at `t=300 s`.** The action
horizon controls only the FILL window, not the HOLDING period, and the simulated
order never cancels or reprices.

So `E[outcome − ℓ | fill]` at settlement is the PnL of **"fill, then hold to
expiry"** — not spread capture. On the *marginal* fills (those `NEW_BBO` catches
and `JOIN_BBO` does not) this produced an implied **+10.31 ¢/share on btc**,
against a **0.50 ¢ half-spread**. Ten cents a share on a one-cent book is not
market making; those fills are exactly the cases where price moved away and
never came back, and the measurement is dominated by **directional drift**.

**How far this reaches:** the `+0.173 ¢/share` pooled maker edge (§2) uses the
same estimand, so it too is hold-to-expiry PnL rather than spread capture. It is
**less contaminated** because it is a population census rather than a selected
subset — buys and sells, all fills, drift largely symmetric — but it is not
measuring the quantity a maker who quotes both sides and flattens would earn.
**Nobody has stated whether this programme's maker holds to expiry or flattens.**
That is an unstated strategy assumption sitting underneath every edge figure.

Isolating placement needs markout at a **fixed short horizon after fill** plus a
**stated cancellation policy** — a different estimand requiring fresh
pre-registration. It must not be swapped in after seeing the above.

## 1c. Inventory: control is LOAD-BEARING, and two-sided quoting only works where flow is thick

`net(t)` was simulated under a two-sided quote and replayed against the tape,
60 windows/coin. **No coin is self-balancing.** Every measured mean-reversion
half-life is **519–2726 s**, all of them **longer than the 300 s window** — so
even where reversion exists it is far too slow to matter inside a market. The
dump mechanism in `plans/DA_INVENTORY_STATE_PLAN.md` is **not** deleted.

*(The obvious confound was tested and refuted: sub-linear variance scaling could
have been manufactured by the measured terminal collapse, but body-only `beta`
(`t ≤ 240 s`) is SMALLER on every coin, not larger. The scaling is real.)*

**Two-sided quoting offsets in proportion to fill frequency, and on thin coins it
is actively counterproductive.** Terminal `|net|` under two-sided quoting divided
by the one-sided control, same windows:

| coin | fills/window | ratio |
|---|---:|---:|
| btc | 317.6 | **0.101** |
| eth | 62.5 | **0.199** |
| xrp | 10.3 | 0.663 |
| doge | 2.2 | 1.173 |
| hype | 1.9 | **1.752** |

At ~2 fills/window the two sides rarely pair, so the second quote adds *unpaired*
fills rather than offsetting ones. **On bnb/doge/hype `skew` does not work at
all**, leaving only `dump` or `do not quote` — which the inventory plan's §2
does not allow for, since it treats skew as a control that always helps. This is
an independent reason for the btc/eth restriction, arrived at from inventory
rather than from bracket width.

**`NEW_BBO` is a pure random walk carrying ~9.4× the inventory risk.** `beta` is
0.908–0.996 with every OU slope positive; on btc terminal `|net|` p95 goes
**191.8 → 1805.1 shares** and cash at risk **$121.80 → $541.76**. Mechanism: at
the front you fill on every reaching trade and absorb one-sided bursts in full,
while behind the queue you fill only on sweeps — **the queue acts as a filter
that skips exactly the directional flow that accumulates**.

**The policy comparison could not see this.** It measured fill and markout and
found `NEW_BBO` ahead on fills (94.6 % vs 76.9 %). Front-of-queue buys fill rate
and **pays for it in inventory risk at ~9×**, so that comparison was generous to
`NEW_BBO`. **Any placement comparison needs inventory risk as a third axis.**

## 1d. Placement skew works — as an UPPER BOUND with an untested generous assumption

The state-dependent policy (reducing side at `NEW_BBO`, adding side at
`JOIN_BBO`) was simulated against the tape, paired on the same windows:

| coin | JOIN p95 \|net\| | SKEW p95 | cut | JOIN half-life | SKEW half-life | JOIN $ | SKEW $ |
|---|---:|---:|---:|---:|---:|---:|---:|
| btc | 194.6 | **21.4** | 89.0 % | none | **6 s** | 121.80 | **8.11** |
| eth | 92.0 | **20.0** | 78.3 % | 1021 s | **8 s** | 44.66 | **2.87** |

`SKEW_SUFFICIENT` on both verdict coins — cash at risk falls **15×**, and the
implied half-life drops from *longer than the window* to **6–8 s**.

**The doubt I raised is refuted in the measured arm.** I argued the 1.23×
fill-rate lever was too small to move a 519 s half-life inside 300 s. It is not
applied once — it is a **persistent feedback bias compounding across hundreds of
fills per window**. A drift is not a scaled-up reversion rate.

**But every number above is an UPPER BOUND, and the generous assumption is not
the one that was controlled for.** When a fronted side is fully lifted it
**re-posts at `queue_ahead = 0`** — first in the queue again, immediately.
`SKEW` exercises that ~476 times per window on btc, where `NEW_BBO` symmetric
exercises it once; `JOIN` pays the displayed queue on every re-post and the
fronted side never does. `SKEW_IDEAL` barely beating `SKEW` shows the *flip*
idealisation is not driving the result — **but the flip was never the generous
part. The re-post is, and both arms share it, so their agreement cannot test
it.**

**The untested lower bound must be measured before anyone acts on this:** front
only on genuine level re-formation, re-join the back after every lift.

Also unexplained and outside the pre-registered rule: **skew increases fills by
~40 %** (btc 4,249 → 5,934), so it does not merely redirect flow. More spread
capture and more gross exposure — unresolvable while the edge estimand is broken.

## 2. Measured and UNDETERMINED — not negative, not positive

- **The maker-edge sign.** `+0.173 ¢/share [−0.251, +0.596]` pooled; **all seven
  per-coin CIs span zero on both weightings**; permutation p=0.0482 names no
  coin. Resolving it needs roughly **25–30× current data** — over a month — and
  day-clustered intervals (the correct unit, not yet computable) would be wider.
- **`PING_TIMEOUT` missingness class.** `MNAR-suspect` stands at n=7 of a
  required 12. It is **49 % of all lost time**.
- **The maker rebate `ρ`.** No per-trade in-transaction rebate found; that is
  **not** absence of a rebate. Every `ρ`-dependent estimand stays `Unavailable`.
- **`f_r`'s terminal mechanism — THE CONFOUND IS PARTIALLY BROKEN, 2026-08-22.**
  This entry previously said the mechanism was unidentifiable because window
  phase and wall-clock minute phase are perfectly collinear. The collinearity is
  real, but the two hypotheses were **never observationally equivalent** — they
  differ in **amplitude across minutes**, and nobody had tested that.

  **A uniform minute-boundary artefact is REFUTED.** It predicts comparable
  amplitude at every boundary. Measured terminal-vs-body log-range ratio, against
  a 3.0 bar: **btc 7.32 [6.06, 8.51]**, **eth 6.08 [4.62, 7.49]** — the terminal
  minute is 6–7× the body's. Thin coins fall below the 100-event floor and are
  excluded from the verdict.

  **A periodic component still exists and is still unidentifiable** — body
  minutes also decline monotonically (btc ρ = −0.96, −0.97, −0.95, −0.50). It is
  simply far too small to account for the collapse.

  **TWAP lock-in is FAVOURED but NOT ESTABLISHED.** Any effect confined to the
  final minute — including a non-stationary artefact — predicts the same shape.
  Recorded separately and **excluded from the verdict as circular** (it was
  visible before the rule was written): crossing `r=60` the log-rate **steps UP**
  by +0.422 on btc (1.53×) and +0.260 on eth, then decays smoothly with no
  sub-boundary step.

## 3. Not knowable on this data — do not schedule work against these

- **Queue position CANNOT BE INFERRED FROM THE TAPE** — displayed L2 shows how
  much size sits at a level, never whose or in what order, and cancellation
  cannot recover it (we see 40 shares leave, not *which* 40). Cancellation
  **volume** is abundant, 86–99 % of actions saturated; cancellation
  **position** is the missing quantity.

  **CORRECTED 2026-08-21 — but this does NOT mean fill is undeterminable, and an
  earlier version of this page said it did.** Queue position is an **OUTPUT OF
  THE PLACEMENT POLICY**, not an unknown of nature. Quote a level as it forms
  (new-BBO) and you are at the front; join an existing level and you are behind
  its displayed depth. So `FRONT`/`BACK_DISPLAYED` is **the span across placement
  policies, not an epistemic bracket**, and it collapses to a definite number the
  moment a policy is named. **The strategy defines the measurement; it is not
  downstream of it.**

  What genuinely remains unobserved is narrower: **whether a new-BBO quote
  actually wins the race** against other participants doing the same. That
  depends on our latency and their behaviour, neither of which is in this tape.
  So `q ≈ 0` is an **upper bound on the new-BBO policy**, not a guarantee, and
  the interior of the bracket is reachable by losing races rather than by
  ignorance.
- **Sub-millisecond structure.** The venue timestamps in **milliseconds**. Our
  `recv_ns` is stamped at parse time and manufactures a 0–50 µs pile-up
  (26 µs median, 16.2× Poisson on btc). No collector change fixes this — the
  data is not there. The Hawkes grid is floored at **10 venue ticks**.
- **Own impact, acknowledgement delay, hidden liquidity.** All require placing
  orders. The 2.3–12.1 % of trade volume with no matching displayed decrease
  (SOL one share in eight) is consistent with hidden liquidity and is not
  separable passively.
- **Branching *values* — SUBSTANTIALLY IMPROVED 2026-08-21, and one earlier claim
  is now dead.** The scalar fit was re-run with the instrument floor and a
  continuous optimiser. **Every coin now selects an INTERIOR half-life; none is
  censored**, btc included:

  | coin | branching | half-life (op) | half-life (wall) |
  |---|---:|---:|---:|
  | btc | 0.554 | 0.6935 | **80.8 ms** |
  | xrp | 0.343 | 0.1141 | 97.6 ms |
  | eth | 0.509 | 0.2209 | 115.3 ms |
  | doge | 0.358 | 0.0746 | 130.1 ms |
  | bnb | 0.434 | 0.0858 | 168.6 ms |
  | sol | 0.403 | 0.1485 | 183.4 ms |
  | hype | 0.325 | 0.1143 | 217.7 ms |

  **The earlier "btc is censored at ~36 ms, which is order-splitting" reading is
  WITHDRAWN. That was our grid reaching below venue resolution, not a market
  fact.** Excluding the two sub-resolution grid points on btc, the fit selects a
  sensible interior value. Real clustering sits at **80–218 ms** on every coin —
  reaction-time scale, not slicing and not our processing cadence. So the floor
  did not merely flag the problem; it resolved it.

  **RESOLVED 2026-08-22 — the bivariate fit has been re-run the same way and the
  provisional caveat is lifted.** No coin is censored. **btc moved 0.180 → 0.477
  (2.65×)** and its interval from the degenerate `[0.180, 0.180]` to
  `[0.282, 0.519]`; the floor excluded exactly the same two grid points the
  scalar fit excluded.

  **The two estimators agree on the timescale, which is the strongest evidence
  either produced.** btc half-life is **75.3 ms** bivariate against **80.8 ms**
  scalar — different likelihoods, independently fitted, same answer. All coins
  land at **75–352 ms**, 75×–352× the venue tick. `RETAIN` is unchanged.

## 4. Withdrawn — if you find these anywhere, that document is stale

| claim | replaced by |
|---|---|
| `+0.45 ¢/share` maker markout | `+0.17`, and that spans zero |
| `+95 bps maker gross / +136 with rebate` | the same number as above; falls with it |
| "on real flow, makers lose per fill" (`−0.211`) | spans zero `[−0.849, +0.457]` |
| wide spreads price adverse-selection hazard | unsupported; only "width does not predict edge" survives |
| "ATM runs 6–8 c" | refuted for btc/eth; true only for thin coins |
| "16.3 % of events" unqualified | pooled and btc-dominated; 2.0 %–90.0 % per coin |
| "count layer available / volume layer blocked" | **inverted** — volume is unblocked, count is contaminated |
| the FLB edge | 0.0004 Brier, one-sided; measured on stale books |
| "B2 is actively worse on bnb" | **sample-unstable, not settled** — `+0.0141` at 24 windows, full stack *beats* B1 at 12. A layer whose contribution flips sign between samples of the same era is fitting noise |
| SPEC_REV2 option (d) binning (15 s + additive phase factor) | **refuted** — see §1a. **REPLACED 2026-08-22**: body `[0,240)` as 4×60 s, terminal `[240,300)` as 12×5 s. Body at exactly 60 s spans exactly one period of the unidentified component and absorbs it **by construction** rather than by assumption — which is the only honest treatment of a term whose source is unidentifiable. The body pays almost nothing (btc within-minute log range 0.333–0.454 against 2.855 terminal) |
| "terminal mechanism is unidentifiable" | **partially broken** — uniform minute-boundary artefact refuted at 6–7× amplitude; see §2 |

## 5. Binding rules

- **R-DUAL, per coin.** Above ~35 % micro share the **raw** count is a
  participant measurement, not market flow. `verdict_coins` for fills are
  **btc and eth**; the rest are descriptive only.
- **Delete nothing; label and condition.** A1's failure kills ex-micro
  *deletion* quantities and *vindicates* the multi-type model. Notional
  weighting reweights rather than deletes and needs no independence assumption.
- **State the population of every denominator.** Six instances of that defect so
  far; three read as findings before they were caught.
- **The name is not the definition.** Four instances, one self-inflicted.
  Confirm any field, file or contract label against the code that writes it.
- Read book state from `price_change.best_bid/ask`, never `book` snapshots
  (p90 6.2 s stale). Knowledge time is `recv_ns`. Never pool across
  `collector_version` eras.

## 6. Live artifacts

| | |
|---|---|
| governing protocol | `FLOW_MODEL_PROTOCOL_V4.yaml` (V3 is `governs: false`) |
| specification | `FLOW_MODEL_SPEC_REV2.md`, `plans/BE_FLOWANDFILLS_MODEL_PLAN.md` |
| probes | `flow_intensity.py` · `flow_fill_development.py` · `flow_uncertainty.py` · `queue_and_type.py` |
| results | `FLOW_INTENSITY_RESULTS.md` · `FLOW_FILL_DEVELOPMENT_RESULTS.md` · `QUEUE_AND_TYPE_RESULTS.md` |
| ledger | `FLOW_UNCERTAINTY_LOOP.md` |

Everything else under `live/pm_research/*.md` is **provenance**: correct about
its own moment, not a statement of current belief.

## 7. What is next, and what blocks it

1. **The fill bracket is a POLICY COMPARISON, not a pending measurement.**
   Specify concrete placement policies — new-BBO (front) and join-BBO (back) at
   minimum — and measure fill **and fill-conditional markout for each**. That
   yields a comparison of real strategies instead of a bracket over an unknown.
   Expect the two to trade off rather than rank: new-BBO wins on fills (94.6 %
   vs 76.9 % on btc) and plausibly **loses on markout**, because it quotes when
   the level is forming, thin, and information is freshest. Measuring only the
   fill side would flatter it.
2. **Layer retention** needs ~10 forward days in one collector era. Days only.
3. **The maker-edge sign** needs ~25–30× current data. Days only, and many.
4. **`U9`** re-runs itself unchanged when `PING_TIMEOUT` reaches n=12.

**Collecting more days does not touch item 1**, which is the item that gates the
programme's central question.
