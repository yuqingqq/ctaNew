# FLOW MODEL — HISTORICAL FACT STATE

> **Currency notice — 2026-09-04T15:27:56Z:** this file remains evidence for
> the flow facts it establishes, but it is no longer the programme-wide current
> status or governing plan. Read P003 `workspace/RESULTS.md` §0 and
> `workspace/HANDOFF.md`, then
> `plans/HARMFUL_FILL_HAZARD_TOXICITY_PLAN_V2.md`. New offline work is
> control-first; no heavy run is authorised by this notice.
> Current checkpoint (verified 2026-09-05T09:54:58Z): Gate 1f confirms the fixed
> external owned-execution export is absent. Public raw/Tier-1 tape cannot bind
> client order, venue ack, maker fill and exact fee. Its offline contract passes
> 11 checks, but the decision metric remains null; v2 is stopped at 1/7 and
> Gates 2–6 did not start. All 17 current v2 batteries (182 checks) and the
> 223-check parent suite pass under the one-CPU/1-GiB/no-swap cap. This is an
> input-identification refusal, not P&L.

**Historical instruction retained:** “Read this first, and read nothing else
to find out what we currently believe” applied to the 2026-08-23 flow-model
state and is superseded by the currency notice above.

Updated 2026-08-23 (§1f: the published sample populations are corrected). At
that date it superseded every earlier flow document for the flow facts below.
Those older documents remain provenance — how we got here, and why particular
things were withdrawn — but they argue with each other by construction because
they were written in sequence as claims were corrected.

For the historical flow facts and scopes below, this page remains the compact
source. For current programme state, plan order or result reliability, the
current RESULTS/HANDOFF/v2-plan surfaces named above win.

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
| **Crossing costs ~2.25 ¢/share ATM — TAKER LEG ONLY** | 0.50 ¢ half-spread + 1.75 ¢ fee ≈ **225 bps** on a $1 binary. **BOTH TERMS ARE THE SAME SIDE. DO NOT SUBTRACT THIS FROM A MAKER NET.** | btc/eth; n=600 transactions, as-of 2026-08-23 |
| **Settlement is `S60(T)` vs `S60(t0)`** | 99.8 % on 1,465 windows; the 300 s reading is refuted at 86.9 % | all |
| **ATM spread is 1 tick on btc/eth** | median 0.0100, p90 0.020, 2.29 M executable quotes | btc; ticks by coin: btc 1 · eth 1 · sol 3 · doge 3 · xrp 5 · bnb 5 · hype 7 |
| **The 1-cent spread is a CONSTRAINT, not a convention** | where the 0.001 tick is available the spread is 1 tick in 99.9 % of quotes | btc |
| **0.001 tick exists only in the tails** | 6.75 % at `p<0.15`, 6.73 % at `p≥0.85`, **0 % in the middle three buckets** | btc |
| **One address is a large share of arrivals** | 0.02 shares exactly, 99.98 % SELL, all seven coins, **0.0145 % of notional** | **per coin 2.0 % (btc) → 90.0 % (hype)**; the pooled 16.3 % is btc-dominated |
| **That address is NOT independent of market flow** | A1 fails on all 7; ratio 1.75–2.79, p=0.000; direction **bidirectional/common-driver** | τ=0.25 s, circular-shift null |
| **`f_r` does not rise into settlement** | count: flat then terminal collapse. notional: rises, peaks, then falls — **peak located to the first 5 s inside `r=60`** (btc notional 97→119 through the body, **170.4** at the peak, then a monotone **9.5×** decline to 18.0) | 361 windows/coin, `clob_v3_1` |
| **The terminal minute contains essentially ALL of `f_r`'s dynamic range** | on btc/eth/sol/doge the terminal collapse ratio **equals the full-profile shape ratio exactly**; six of seven coins on notional | 361 windows/coin |
| **The terminal regime is FEW AND LARGE** | count drops to 18 % of peak while notional holds 28 %; USDC/arrival 15.5 → 24.0 (btc), 11.9 → 32.3 (eth) | btc/eth |
| **Book state carries real information** | B3 placebo does not reproduce the gain on any coin (btc −0.03 share, hype 0.02) | 24 windows/coin, **1 UTC day** (§1f) |
| **THE TWO BOOKS ARE ONE BOOK — and the second side is DERIVED, not quoted** | **1,081,800 checks across 560 archives, 7 coins, 4 UTC days: ZERO violations, worst deviation exactly 0.00000.** Zero in the 0.001 tick regime (8,468), the terminal minute (23,828), during gaps, and within 5 s of a tick change | **all coins, all days** — the exactness is the finding: two books agreeing would show float noise, one book computed from the other shows none |
| **A complete set is therefore worth ONE spread, not two** | paying `Up_bid + Down_bid = 1 − spread` for something worth exactly $1 | corrects an earlier coordinator claim |
| **Market self-excitation is not deletable** | bivariate 2×2 **refit 2026-08-22 with the instrument floor and continuous optimiser**: diagonal **0.236–0.477** dominates cross **0.000–0.167** on every coin; btc `market←market` **0.477 [0.282, 0.519]**, eth **0.389 [0.339, 0.431]** | **no coin censored**; both estimators refined, not grid-seeded |
| **Clustering runs at 80–218 ms, and the censoring was OUR GRID** | scalar Hawkes re-fit 2026-08-21 with the instrument floor + continuous optimiser: **no coin is censored**, branching 0.33–0.55, half-life 80.8 ms (btc) to 217.7 ms (hype) — **81×–218× the venue tick** | 24 windows/coin, **1 UTC day** (§1f), `clob_v3_1` |

> **NAME THE LEG BEFORE COMPUTING A NET (added 2026-08-24, DA, under R-105).**
> The `~2.25 ¢/share` row pairs a half-spread with a fee **as one side**: the
> crossing party pays both. The row above it says the other side pays neither —
> *"Taker pays the fee, maker does not"*, **744 of 754 maker legs at zero, so TEN
> legs (1.3 %) WERE charged** (n=600 transactions, as-of 2026-08-23).
> **A maker net that subtracts 2.25 ¢ understates maker economics by roughly the
> whole crossing cost**, which is the largest single term in the model. Any net
> figure must state which leg it is on before it states a number.
> This page wins over conflicting statements by its own rule at the top, so the
> caution belongs HERE and not only in `MEASUREMENT_PLAN.md:951`.
> **Under R-105 every cited population now carries its `n` and its as-of** — a
> figure without an as-of cannot be checked for staleness by anyone, its author
> included.

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
60 windows/coin — **all of them 2026-08-20; one UTC day, see §1f**. **No coin is self-balancing.** Every measured mean-reversion
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
`JOIN_BBO`) was simulated against the tape, paired on the same windows
(25/coin, **all 2026-08-20 — one UTC day, see §1f**):

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

**MEASURED 2026-08-22 — `SKEW_ROBUST`. The concern was legitimate; the magnitude
was small.** Lower bound: front only on genuine level re-formation, re-join the
back after every lift.

| coin | JOIN p95 | SKEW upper | SKEW lower | reduction UB | reduction LB | half-life LB |
|---|---:|---:|---:|---:|---:|---:|
| btc | 194.6 | 21.4 | **45.7** | 89.0 % | **76.5 %** | 12.5 s |
| eth | 92.0 | 20.0 | **17.1** | 78.3 % | **81.4 %** | 10.0 s |

Cash p95: btc `$121.80 → $8.11 (UB) → $9.14 (LB)`. **The 15× survives as ~13×.**
The switch demonstrably bites — btc grants **124 free re-fronts per window** under
UB and **zero** under LB — yet terminal `|net|` moves only 21.4 → 45.7. **The
honest btc figure is a 76–89 % reduction: the published number is the optimistic
end of a narrow band, not a different answer.** LB is if anything pessimistic,
since it re-joins behind *pre-trade* displayed depth that the lifting trade
partly consumed. `hype` alone is bound-dependent, and not for the expected
reason — the cut is real but its LB half-life is 328.5 s, outside the window.
Too slow, not too weak.

**The ~40 % fill increase is NOT the same artefact.** 63 % of btc's increase and
84 % of eth's survive the pessimistic queue (btc `MIXED`, eth and the rest
`GENUINE`). Fronting genuinely wins fills the queue loses; btc is the only coin
with a material re-post component, consistent with being the only one where
re-posting fires often enough to matter.

Also unexplained and outside the pre-registered rule: **skew increases fills by
~40 %** (btc 4,249 → 5,934), so it does not merely redirect flow. More spread
capture and more gross exposure — unresolvable while the edge estimand is broken.

## 1e. LAYER 1 MEASURED — adverse selection is ~2x spread capture, and the sign is negative

> ### ⚠ SCOPE, R-110 — THIS TABLE IS A LEVEL ON ITS OWN POPULATION AND IS NO
> ### LONGER THE POLICY ANSWER.
>
> **Population, carried with the numbers (R-105): ONE UTC day, JOIN_BBO only,
> n = 10,294 btc fills / 1,999 eth, markout against BOOK MID. As-of the Layer-1
> run; superseded as a policy statement 2026-08-24.**
>
> **What it still is:** the measured LEVEL for a book-equal maker on that day —
> spread capture is real and positive, adverse selection is about twice it, and
> the net sign is negative. Nothing here is retracted.
>
> **What it is NOT, and was being used as:** the answer to *which placement
> policy is better*. **It measures ONE arm** — JOIN_BBO — **on ONE day**, so it
> cannot rank two policies, and a five-day paired comparison of both arms now
> exists (§7 item 1). **Cite this table for the LEVEL and the decomposition.
> Cite §7 item 1 for any comparison between placements.**
>
> **Two things that do NOT conflict, though they read as if they might.** This
> table is markout vs **book mid**; the policy comparison is markout vs
> **settlement** — different estimands, deliberately, so the comparison stays off
> the sigma clock. And both say the same thing about profitability: **the level
> is negative here, and negative on all ten coin-days in both arms there.**
> **The programme has never measured a placement that pays at these horizons.**

The estimand §1b called broken has been replaced and run. Markout against
**book mid** at fixed horizons, decomposed, per `EDGE_LAYER1_PROTOCOL.md`.
Verdict **`HORIZON_DEPENDENT`**, and the decomposition is the result:

| coin | h | n | markout | CI95 | spread captured | post-fill drift |
|---|---:|---:|---:|---|---:|---:|
| btc | 5 s | 10,294 | **−0.532** | [−0.797, −0.287] | **+0.642** | **−1.175** |
| btc | 30 s | 9,714 | −0.637 | [−1.047, −0.216] | +0.636 | −1.273 |
| eth | 5 s | 1,999 | **−1.243** | [−1.726, −0.759] | **+0.778** | **−2.021** |
| eth | 60 s | 1,752 | −1.609 | [−2.479, −0.807] | +0.695 | −2.305 |

**Spread capture is real, positive and stable** (+0.61–0.64 ¢ btc, +0.70–0.78 ¢
eth — consistent with a 1-tick book plus occasional wider spreads, which is the
harness sanity check). **Post-fill drift is 1.8× larger on btc and 2.6× on eth,
and negative.** **Eight of eight cells are negative in point estimate; FIVE of
eight have the interval excluding zero** (btc h=5/15/30, eth h=5/60). The three
that span zero are btc h=60 `[-0.834, +0.633]`, eth h=15 `[-1.284, +0.089]` and
eth h=30 `[-1.393, +0.059]`.

*Corrected 2026-08-23 (DE flagged; BE owns the page).* This line read "six of
eight ... with the interval excluding zero", which is one count doing the work of
two: the sign holds on 8/8, the significance on 5/8. Counted against
`edge_layer1_v1.json` directly. **The direction of the finding is unchanged and
the headline cells are the significant ones**, but a reader was being handed a
stronger significance claim than the receipt supports.

**So a passive two-sided `JOIN_BBO` maker loses at short horizons on both verdict
coins.** This is the first *determinate* answer the programme has produced to
whether the fill itself is any good, and it closes a loop: the fee structure
predicted it. Takers pay ~225 bps to cross, so anyone crossing is heavily
informed — "the fee does not kill MM on cost, it loads the question onto adverse
selection" was the prediction, and adverse selection is roughly double the
capture.

**btc's apparent improvement at `h=60` is a POPULATION ARTEFACT, not attenuation.**
Drift reads −1.175 → −0.697 across the ladder, but `h=60` discards **1,611 btc
fills, all inside the terminal minute**, shifting the surviving `r`-population
from p50 166 s to 190 s. `h=60` cannot see the final minute at all, by
construction. The protocol pre-specified this exclusion as non-benign and it is
material here.

**Not a micro-actor artefact:** excluding the 0.02 class moves btc −0.532 →
−0.521.

**Descriptive coins suggest a wide spread buys TIME, not edge.** hype is
**+3.802** at `h=5` on a 5.86 ¢ spread, then **−3.707** by `h=30`. Wide spreads
delay adverse selection rather than defeating it. n=58, directional only.

**Methodological point worth keeping:** the settlement estimand was
`UNDETERMINED` with every interval spanning zero; the narrower Layer-1 question
has a **determinate** answer. Asking a smaller question got a sharper one.

**UNRECONCILED and deliberately not narrated:** the settlement census reported
**+0.173 ¢** pooled while Layer 1 reports **−0.53 ¢** on btc at `h=5` with the
interval excluding zero. Different estimands over different populations — the
census was all observed fills marked at settlement, this is a simulated
two-sided `JOIN_BBO` maker marked against mid — so **not** a direct
contradiction. The directions differ and the reconciliation is **unmeasured**.

**Scope:** Layer 1 only. Inventory carry, the terminal residual and the `r≈60`
decision are Layer 2 and can move the total either way. **ONE UTC day
(2026-08-20), window-clustered — see §1f.** Adopted under R-29.

The receipt's `source_days: 4` stands **beside** this as provenance, and the
coordinator's framing is sharper than BE's own: **the receipt says what the run
COULD have drawn from; the measurement says what it DID.** BE had recorded the
receipt's 4 as simply *wrong* — it is not wrong, it answers a different question.
Only the earlier "Two days" on this line was wrong. Under R-6 a sample population
is Class C, so the coordinator adopted the measurement rather than choosing
between the two numbers; which one is true was never a decision.

## 1f. EVERY headline simulation result is ONE UTC DAY — the selection rule, not the calendar

Measured 2026-08-23. `select(per_coin)` — the window chooser every replay and
flow probe shares — walks `sorted(covered_slugs(ERA))` and truncates at
`per_coin` per coin. A slug ends in its window's epoch start, so sorted order is
**chronological**: "the first N" is **the EARLIEST N**. The `clob_v3_1` era opens
2026-08-20 14:50:21, so any sample at N <= 60 never leaves 2026-08-20.

| receipt | windows/coin | UTC days actually sampled |
|---|---:|---|
| `policy_comparison_v1` | 10 | **1** — 08-20 |
| `queue_c1` · `queue_c2` · `flow_fill_development_v1` | 24 | **1** — 08-20 |
| `placement_skew_t1` · `skew_bound_v1` | 25 | **1** — 08-20 |
| `edge_layer1_v1` | 30 | **1** — 08-20 |
| `inventory_walk_v1` | 60 | **1** — 08-20 |
| `flow_phase_interaction_v1` | 273 | 2 — 08-20, 08-21 |
| `flow_grid_nonuniform_v1` · `flow_terminal_mechanism_v1` (`f_r`) | 361 | 2 — 08-20, 08-21 |

**680 windows/coin spanning four UTC days (08-20 … 08-23) are on disk now.** The
Layer-1 sample is 30 of them — **1/22.7 of the available tape**. So §1a and §2's
`f_r` results genuinely rest on two days; §1c, §1d, §1e, the A1 dependence test,
the B3 placebo, the Hawkes fits and the queue results all rest on **one**.

Three consequences, and none of them is cosmetic:

- **Day-clustered intervals on these results are not "not yet computed" — they
  are NOT COMPUTABLE.** One day is one cluster. Every published interval on a
  single-day result is a WITHIN-day interval and cannot see day-to-day variation
  at all. Where this corpus says the day unit is "the correct unit, not yet
  computable", the reason is this sample, not the calendar.
- **§2's "needs roughly 25–30× current data — over a month" conflates two
  limits.** For the settlement *census* that is a calendar statement and stands.
  For every `select()`-based result a **22.7×** larger sample is already on disk
  and needs no calendar at all.
- **Nothing here says any result CHANGES when re-sampled.** It says what the
  stated population is. Re-sampling is a SELECTION decision under R-ADMISS and is
  coordinator-gated: BE proposes it and does not self-decide it.

**Why the existing guard did not catch it.** `fi.provenance()` was written for
exactly this failure and its docstring names it — *"a run after `DAYS` grew to
four days still reproduced a three-day figure to the digit"*. But only two
receipts on disk carry a provenance block at all, `edge_layer1_v1` and
`skew_bound_v1`, and both were written by the **pre-fix** version that reports
days **read** rather than days **sampled**. Both therefore stamp `n_days: 4` on a
one-day sample — a **4× over-report**, and the reason DE read the receipt as
contradicting this page when in fact both were wrong in opposite directions. The
other ten receipts carry no provenance block whatsoever.

**The guard that would have caught it:** every published probe calls
`fi.provenance(sampled=<the slugs it actually simulated>)`, and no result is
compared against a published one except on `days_sampled`.

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
- **The name is not the definition.** **Five** instances, **two** self-inflicted.
  Confirm any field, file or contract label against the code that writes it.
- **A shrinking or growing sample must be VISIBLE in the receipt.**
  `flow_intensity.DAYS` went stale silently on 2026-08-22, omitting a whole day
  and 1,141 archives from every probe that imports `_archive_paths()`. Both
  directions are now loud in the code: `assert_days_current()` raises if a
  collected day is unlisted, and `fi.provenance()` can stamp the sample into a
  receipt.

  **CORRECTED 2026-08-23, and this bullet was wrong twice — see §1f.** It said
  every published number "was computed on three" days: for N <= 60 windows/coin
  they were computed on **one**, and only the `f_r` probes reach two. It also
  said "re-running any probe will now produce a four-day population": it will
  **not**. `select()` is EARLIEST-first, so a grown `DAYS` changes the population
  READ and leaves the population SAMPLED pinned to the era's opening hours. A
  fixed day-list was one bug; an earliest-first sampler with a growing disk is a
  second one, and fixing the first hid the second. **Compare runs on
  `days_sampled` — which requires passing `sampled=` — never on `source_days`,
  `days_read` or the constant.**
- Read book state from `price_change.best_bid/ask`, never `book` snapshots
  (p90 6.2 s stale). Knowledge time is `recv_ns`. Never pool across
  `collector_version` eras.

## 5a. Sampling binds any probe that feeds a frozen bar — R-35

Day-stratified sampling is **not** confined to V5 fits. It binds **any probe
whose output is an input to a frozen bar**, because R-20 anchors a frozen bar to
its inputs **by value** — so if those inputs came from a sampler that cannot
leave one day, the anchor is anchored to a **biased number**.

**`edge_l1_v1` is therefore in scope**: its Layer-1 markout feeds R-1's `f*`,
even though it is not a V5 fit. Two constraints, both mechanised in
`flow_intensity` rather than left to memory:

| constraint | mechanism |
|---|---|
| a re-sampled Layer-1 markout **does not move the bar** — it creates a candidate needing a Class-D amendment | `resampled_markout_is_a_candidate()` returns `status: CANDIDATE_NOT_A_BAR`, `bar_moved: False`, `amendment_is_free: False` |
| it must **never be pooled** with earlier earliest-first receipts | `assert_poolable()` raises on mixed or undeclared `sampling_rule` |

**The amendment is no longer free.** `ww_v1`'s measurement has already run, so
R-6's clause (c) — invalidate every verdict computed under the old bar — has
something to invalidate. That is the difference between the `STOP` amendment,
which was made before any evaluation existed, and this one.

## 5b. THE SIGMA LANE AND THE BOOK LANE HAVE NO OVERLAPPING DATA — R-56

Surfaced by E-X1's `VOID(NO_PAIRED_POPULATION)` and larger than E-X1. Recorded
here because it constrains **any** question needing both arms and had not been
stated anywhere.

```
sigma  (route_a_v1 OOS windows)  2026-08-20 00:00:00 -> 13:50:00
book   (clob_v3_1 era opens)     2026-08-20 14:50:21
overlap                          0 of 5,796 OOS rows
```

**What this does and does not mean — BE verified the distinction, because
"calendar-blocked" is right for two of the three blockers and wrong for the
binding one.**

| blocker | kind | does waiting fix it? |
|---|---|---|
| zero paired population | **ARTIFACT**, not data — the sigma inputs (`crypto_prices_twap_*`) exist on **all five days**, and 2,016 / 2,016 / 1,429 windows settled on 08-21/22/23 | **no — a re-run fixes it**, and D-3 authorises rerunning `route_a_v1` unchanged as days accrue |
| one OOS day vs a threshold of three | calendar | yes |
| **`route_a_v1` carries NO PROBABILITY** | **METHOD** | **no** |

**The third is binding and structural.** `route_a_v1` emits a conditional mean
and a residual variance **in bps** — it is not a probability model. Getting `p̂`
requires the link and σ, i.e. `pricing_distribution`: **the probability path
`PRICING HOLD` exists to block.**

So E-X1's framing does not survive contact. It argued `PRICING HOLD` did not
block it *because scoring is not pricing* — but **there is no artifact-level
probability to score**, so scoring requires invoking the pricing path. The
distinction was not merely fine; it was empty.

**Consequence: E-X1 does not become runnable by waiting.** It becomes runnable
only if `PRICING HOLD` lifts or a link is pinned — a Route-A gate question, not a
calendar one. Recorded so the VOID is not re-opened on the belief that time alone
clears it.

## 6. Live artifacts

| | |
|---|---|
| governing protocol | **`FLOW_MODEL_PROTOCOL_V5.yaml`** — frozen 2026-08-23 under R-19. V4 and V3 are `governs: false`. V5 changes the `f_r` grid (body 4×60 s, terminal 12×5 s), decouples `f_p`'s `r_band` from it, adds the `DAY_BLOCK_UNAVAILABLE` refusal, and retires earliest-first sampling for **day-stratified** (D-V5-3). Promotion clock RESET at the freeze; the day count is **derived, never written down** (D-V5-2) |
| specification | `FLOW_MODEL_SPEC_REV2.md`, `plans/BE_FLOWANDFILLS_MODEL_PLAN.md` |
| probes | `flow_intensity.py` · `flow_fill_development.py` · `flow_uncertainty.py` · `queue_and_type.py` |
| results | `FLOW_INTENSITY_RESULTS.md` · `FLOW_FILL_DEVELOPMENT_RESULTS.md` · `QUEUE_AND_TYPE_RESULTS.md` |
| ledger | `FLOW_UNCERTAINTY_LOOP.md` |

Everything else under `live/pm_research/*.md` is **provenance**: correct about
its own moment, not a statement of current belief.

## 7. What is next, and what blocks it

1. ~~**The fill bracket is a POLICY COMPARISON, not a pending measurement.**~~
   **DONE, AND THE PREDICTION STATED HERE IS REFUTED.** *(R-109/R-110,
   `policy_comparison_v2.json`, protocol frozen 2026-08-22 before the run.)*

   > **Population, and it must travel with every number below (R-105): 5 days
   > 2026-08-20 … 2026-08-24, verdict coins btc/eth, h = 5 s, 30
   > windows/coin/day — EXCEPT 2026-08-24, WHICH IS A PARTIAL DAY AT 21 WINDOWS.
   > "Five days" must never be written without the partial being visible; eth's
   > day-robustness turns on it. As-of 2026-08-24.**

   Revision 0 of this item predicted: *"Expect the two to trade off rather than
   rank: new-BBO wins on fills … and plausibly **loses on markout** … Measuring
   only the fill side would flatter it."*

   **THE MARKOUT PENALTY DOES NOT APPEAR ANYWHERE.** FRONT is *better* on
   markout than JOIN in the point estimate on **9 of 10 coin-days**, not worse.
   The trade-off this page told the programme to expect is not in the data.

   **What SURVIVES the measurement, stated as surviving:**

   - **The LEVELS. Both policies LOSE at h = 5, on all ten coin-days, both
     arms — 20 of 20 arm-cells negative** (btc −0.516 … −1.514 ¢/share; eth
     −0.716 … −2.862). This is a count, no interval touches it, and it is the
     single most robust thing the run produced. **Any sentence of the form
     "FRONT beats JOIN" is incomplete without "and both lose at h = 5" on the
     same line.**
   - **The FILL advantage is real and day-robust with room to spare** — FRONT
     takes ~5–6× the shares per window (btc +6,054/window, CI [5,698, 6,392];
     eth +969, CI [903, 1,033]).
   - **A pre-registered prior was tested and found wrong**, which is worth more
     than a confirmation would have been.

   **What RETRACTS — BOTH markout intervals:**

   - **btc is NOT day-robust.** Window-clustered [+0.026, +0.251] excludes zero;
     day-clustered on G=5 it is **[−0.098, +0.389] and spans it**, with the sign
     **negative on 2 of 5 days**.
   - **eth is SUGGESTIVE, NOT DAY-ROBUST.** All 5 days positive and the 5-day
     interval is [+0.093, +0.879] — but **drop the partial 08-24 and it becomes
     [−0.067, +1.082] and spans zero.** *The precise mechanism, because the loose
     phrasing misleads:* the mean **RISES** without the partial day (+0.486 →
     +0.507). **eth loses significance to a lost degree of freedom, not to a lost
     effect** — 4 clusters cannot carry the interval. That is a POWER statement
     about eth and a SIGN statement about btc, and they are not the same finding.
   - **So the markout comparison is SUGGESTIVE AND UNSETTLED.** It may not be
     cited as a result.
   - **Unresolved and blocking a verdict (`Q-BE-60`):** on **eth** the arms'
     mean fill sizes differ by up to **20 %** (FRONT:JOIN 0.796–0.918), so the
     published **share-weighted** markout and a per-fill markout are **not
     interchangeable** — on btc they are (0.992–1.029). **The receipt publishes
     only the share-weighted figure**, so the one coin whose verdict survives is
     the one where the weighting is load-bearing, and the check cannot be run
     from the artifact.

2. **Layer retention** needs ~10 forward days in one collector era. Days only.
3. **The maker-edge sign** needs ~25–30× current data. Days only, and many.
4. **`U9`** re-runs itself unchanged when `PING_TIMEOUT` reaches n=12.

~~**Collecting more days does not touch item 1**~~ — **superseded.** Item 1 has
run. **Days are now exactly what item 1 needs**: btc's markout sign is unstable
across days and eth's interval opens at G=4, so both retractions above are day-count
problems. The fill advantage needs nothing further.
