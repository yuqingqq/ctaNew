# DE-DecisionScheme — the decision module plan

Plan only. No code, no measurement, no fitted quantity. Status: **DESIGN**, not
decision-eligible. **Revision 7, 2026-08-23** — applies `DE_PLAN_REVIEW_LOOP.md`
iteration 5: the Layer-1 sign count corrected to FIVE of eight (a receipt-level
fact error that survived four iterations); the ≤|net| cap's rationale rebuilt a
second time — after two successive false derivations the cap now stands on its
DEFINITIONAL ground (REDUCING-ONLY means never past flat) with the predicate
shown unable to substitute in EITHER direction; §8.1 envelope naming aligned to
the frozen E-FLOW form. **§8.1 has been RUN and ANSWERED under Ruling R-1:
DEAD on both verdict coins — see §8/§9.** **Now Revision 9** — R-9 executed:
margin-stated phrasing (the flat "doubly dead" corrected; this bump also
covers the unbumped R-9 phrasing edits, restoring the revision-carrier
convention) and the day-series outcome recorded in §8.1 — DEAD at the
decision rung on all four era days, the btc lag-floor subclaim dropped.
Prior — **Revision 8** — iteration 7
header correction (also absorbing iteration 6's unbumped §8.1 edit):
**Revisions 7–1 predate version control of this file and are NOT preserved
beyond the pre-revision original** (the "in git history" claim was false —
only the 2026-08-22 original was ever committed); committed per-iteration
from this revision onward. Cross-session ledger:
`orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/COORDINATION.md`.

> For current programme state read [`FLOW_MODEL_STATE.md`](../FLOW_MODEL_STATE.md).
> This document is a **plan**. Where it conflicts with that page, that page wins.
>
> This is the **policy content** of `DE-DecisionScheme` only. The DE plane has
> five modules; their structure, contracts and build order are in
> [`DE_MODULE_PLAN.md`](DE_MODULE_PLAN.md), which also owns the single
> definition of "new risk", the lifecycle states, and all constants' SP
> ownership. Numbers quoted here are exposition; the owner of each constant is
> named there.

---

## 0. Plane, and the boundary this document must not cross

This is `DE-DecisionScheme` content: **what to do**, given state and beliefs.

| input | plane | owner |
|---|---|---|
| the position | `DA` | `DA-State` / `SelfState` — [inventory state plan](DA_INVENTORY_STATE_PLAN.md) |
| `P(fill \| placement, market state)` | `BE` | `BE-FlowAndFills` |
| the risk limit | `SP` | `SP-Params`, enforced by `DE-Constraints` |
| **the placement/cancel/dump/hold decision** | `DE` | **this document** |

**`BE-FlowAndFills` is inventory-agnostic.** Dependency rule **SP ← DA ← BE ←
DE**: DE may read BE, BE must never read DE. This document consumes the fill
model; it must never ask for an inventory term to be added to it. Skew is
implemented here, by choosing *where* to place given `net` — and cancellation is
implemented here, by choosing *when to leave* given the tape.

**Scope: btc and eth only** (`verdict_coins`, an SP-owned eligibility fact).
Four independent measurements converge on it — bracket width, micro-actor
contamination, inventory (two-sided quoting is counterproductive at ~2
fills/window: doge 1.17, hype 1.75), and skew speed (hype's lower-bound
half-life is 328.5 s, outside the window — too slow, not too weak).

---

## 1. What Revision 1 asked that is now answered

| Revision 1 question | answer | evidence |
|---|---|---|
| §7.1 — does the state-dependent drift beat 300 s? | **YES — `SKEW_ROBUST` on both verdict coins.** Terminal `\|net\|` cut **76.5–89.0 %** (btc), **78.3–81.4 %** (eth); cash at risk ~**13×**; half-life 519–2726 s → **10.0–12.5 s** at the pessimistic bound | `SKEW_BOUND_RESULTS.md` |
| §5's plausible negative — is the 1.23× fill lever too small? | **Refuted.** The lever is not applied once; it is a persistent feedback bias compounding across hundreds of fills/window | `PLACEMENT_SKEW_RESULTS.md` |
| §7.2 — placement comparison with inventory as a third axis | Run for the inventory axis. The markout axis now exists separately as Layer 1 — **the two have not been run together under one policy** | this revision, §8.3 |
| state plan §6 — does the one-book identity generalise? | **EXACT, venue-wide.** 1,081,800 checks, 7 coins, 4 days, zero violations, worst deviation 0.00000 | `PLACEMENT_SKEW_RESULTS.md` T2 |

**And the fact Revision 1 did not have:** a passive two-sided `JOIN_BBO` maker
**loses at short horizons on both verdict coins**. Markout against book mid,
decomposed (`edge_l1_v1`, 30 windows/coin — **all from ONE UTC day,
2026-08-20**, per `FLOW_MODEL_STATE.md` §1f: the shared sampler is
earliest-first, and the receipt's `n_days: 4` counted days *read*, not
*sampled*):

| coin | h | n | markout | CI95 | spread captured | post-fill drift |
|---|---:|---:|---:|---|---:|---:|
| btc | 5 s | 10,294 | **−0.532** | [−0.797, −0.287] | +0.642 | **−1.175** |
| eth | 5 s | 1,999 | **−1.243** | [−1.726, −0.759] | +0.778 | **−2.021** |

Spread capture is real, positive and stable; adverse selection is **1.8–2.6×
larger**. **Five of eight** cells negative with the interval excluding zero
(spanners: btc h=60, eth h=15, eth h=30 — iteration 5 corrected a
"six of eight" that had survived four iterations and originated in
`EDGE_LAYER1_RESULTS.md`; the receipt's `signs` block is the authority, and
the f*/DEAD verdicts rest on h=5 only, so nothing downstream moves).
*(Provenance history, iteration 3: iteration 2 surfaced a "Two days"-vs-"n=4"
conflict and bet on the receipt. BOTH were wrong — BE measured the true sample
at ONE day (§1f) and the receipt's n=4 as days read. Every interval on this
table is therefore WITHIN-DAY; day-clustered inference does not exist for it.
Re-sampling at 680 windows/coin over 4 days is an R-ADMISS selection decision
pending with the coordinator — BE proposal, `COORDINATION.md` D-3.)*

**READ IT NARROWLY, because the narrowness is the whole design opening:** every
simulation in this corpus rests the order until filled or window end — **nothing
ever cancels**. The measured result is *"a maker who never cancels loses here"*,
not *"market making loses here"*. The gap between those two statements is this
module's central question.

---

## 2. The menu, revised — four levers, one of them unexamined

| | lever | acts on | fee | spread | status |
|---|---|---|---|---|---|
| **A** | **Skew** — shade placement toward the offsetting side | inventory drift | 0 | earns | **MEASURED, WORKS** — 76–89 % / 78–81 % `\|net\|` cut, ~13× cash |
| **B** | **Cancel** — pull a resting quote before it fills | **fill quality (Layer 1)** | 0 | forgoes | **UNMEASURED — the lever this plan exists to specify** |
| **C** | **Dump** — cross to flatten | terminal residual | `0.07·p(1−p)`/sh | pays `h` | derived threshold `N*` (§6); backstop |
| **D** | **Hold** to resolution | terminal residual | 0 | 0 | default at `r < 60` (measured liquidity collapse, §7) |

Skew and "quote the complement" remain one act (the two books are one book —
exact at 1.08 M checks). **The asymmetry driving C and D:** one ATM dump costs
`1.75 + 0.50 = 2.25 ¢/share` against **measured** capture of +0.61–0.64 ¢ (btc)
/ +0.70–0.78 ¢ (eth) per fill — one forced unwind destroys roughly **2.9–3.7
fills'** worth of gross capture (4.5 only on the naked 0.50 ¢ half-spread
basis; iterations 2–3 — the number must come from the measurement, not the
tick, and 2.25/0.778 = 2.89 is the honest low end). **Dumping must be rare by
construction, not by tuning.**

**The division of labour:** skew is measured to solve the *inventory* problem
and does nothing for *fill quality* — the ~40 % fill increase it brings is
unresolvable while each marginal fill marks negative. Cancellation is the only
lever that acts on fill quality. **A is necessary and measured; without a
working B, A controls the inventory of a losing book.**

---

## 3. The composed policy — the object `DE-DecisionScheme` emits

```
WHERE to rest      skew on net (MEASURED as SKEW_ROBUST's LB arm):
                     long Up  -> reducing side FRONT_ON_FORMATION, adding side JOIN
                     short Up -> mirrored;  near flat -> both JOIN
WHEN to leave      cancellation rule (§4, UNMEASURED)
WHEN to cross      dump only above N* (§6) — ALWAYS preceded by cancelling any
                   own resting quote on the side being crossed into (§4.7)
WHEN to stop       terminal schedule (§7): AT r≈60 CANCEL the adding side and
                   reduce |net| to carry size; r<60 the book is REDUCING-ONLY;
                   carry the residual to resolution
```

**Placement semantics, corrected (review iterations 1–2).** At a 1-tick book —
the **modal** btc/eth state (median spread exactly 1 tick, p90 2 ticks; the
99.9 % figure is conditional on the tails-only 0.001-tick regime and is NOT
this population — iteration 2 caught the swapped denominator) — price
improvement would cross or lock, and a same-price queue jump is impossible. So `NEW_BBO` was never an
unconditioned action: **the executable policy is `FRONT_ON_FORMATION` — front a
level only at genuine level re-formation, rest at `JOIN` otherwise.** That is
exactly the measured `SKEW_LB` arm (76.5–81.4 % reduction, half-life 10.0–
12.5 s), so the honest composed policy quotes the *lower-bound* arm and the
symmetric-front figures are bounds, not the policy. Recorded as a
name-vs-definition instance in the loop charter.

Under `PM_ARCHITECTURE.md` §6 this is one `DecisionScheme`; the module-level
wiring (contract names, `Action.order_ref`, placement field, registries) is
`DE_MODULE_PLAN.md` §6.2's list, not this document's.

---

## 4. The cancellation design space — mechanism first

### 4.1 Why it could work — and the lag that disciplines the claim

1. **The drift arrives fast.** On btc, −1.175 of the −1.273 ¢ 30 s drift is
   already present at `h = 5 s` (~92 %; horizon populations differ, per the
   protocol's caveat).
2. **Aggressive flow is clustered at 75–352 ms** (Hawkes branching 0.33–0.55,
   two estimators agreeing). A burst has a first trade — **but at our frozen
   250 ms knowledge lag, the first trade is knowable ~3 btc half-lives after it
   prints, when most of that cluster's excess intensity has already decayed.**
   The single-burst reading of "the first trade is a warning" is therefore
   mostly false at knowledge time; what survives is the longer-range structure:
   directional runs span multiple clusters, and the queue mechanism below does
   not depend on sub-lag reaction.
3. **The queue is a warning buffer.** A joined quote fills after the depth
   ahead is consumed or churned away — consumption prints as trades,
   **churn prints in displayed L2 totals**, and both precede our own fill by
   however long the queue takes to drain, which can be far longer than one
   burst. Queue depth converts into reaction time; the 9.4× front-of-queue
   finding is this same property read as risk.

### 4.2 Why it could fail — decidable *before* building any policy

- **Unwarned fills.** A sweep that reaches our level in one event, or a drain
  faster than `lag + τ`, arrives with zero usable warning. If most negative
  drift sits on such fills, **no cancellation policy on this venue rescues
  Layer 1** and the "never cancels" qualifier collapses.
- **The warning window inherits the queue-position bracket.** Under `FRONT` the
  consumption channel is zero (depth-churn and off-level channels remain);
  under `BACK_DISPLAYED` it is maximal. Both bounds are reported; the truth is
  between and unidentifiable — the fill bracket restated in time.
- **Non-selective excision.** A rule that cancels benign and adverse fills in
  proportion cuts capture as fast as drift.
- **The race.** The tape contains neither our latency nor anyone else's.
  Simulated cancels are upper bounds; §4.4 keeps them honest.

**Consequence: measure the warning-window distribution first, policy-free
(§8.1).** It bounds the value of the whole family before any rule is built.

### 4.3 Trigger inventory — readable at knowledge time, no fair price, no wall clock

| id | trigger | reads | mechanism |
|---|---|---|---|
| `T-FLOW` | aggressive trade at or within k ticks of our level, hitting our side | trades | burst/run onset |
| `T-DEPLETE` | displayed depth ahead/at our level falls below a floor | L2 totals | sees churn as well as trades |
| `T-MID` | mid moves ≥ m ticks against our resting side since post | quotes | protective/momentum |

All inputs arrive through `StateView` at the frozen 250 ms lag; every time
quantity (including `r`) derives from the view's knowledge time, never a wall
clock (R-ENV).

**`T-AGE` is REMOVED (review iteration 1), and the reason is mechanical:**
waiting in queue never worsens queue position — `q_ahead` only decreases — so
cancel-and-re-post at an *unmoved* level pays queue position for nothing. The
only valid age-adjacent rationale is repricing at a *moved* level, which is
`T-MID`'s job. The earlier "re-anchors queue position" rationale was backwards.

Route A stays untouched: **no trigger uses a fair value.** A reservation-price
cancel is a different object, blocked on the sigma clock, deliberately absent.

### 4.4 The latency ladder — with the lag on the right side of the inequality

Trigger event prints at `t`; it is *knowable* at `t + lag` (frozen 250 ms); the
cancel is effective at `t + lag + τ`, `τ ∈ {0, 50, 100, 250, 500, 1000} ms`.
**A fill arriving before `t + lag + τ` still happens.** The achievable floor is
therefore the *lag*, not zero: the `τ = 0` rung means "cancel at knowability",
and no rung is an upper bound on anything faster than the lag. Every result
table carries this sentence and the race caveat (§4.2).

### 4.5 Re-arm and re-post semantics — pinned per trigger type, or arms are not comparable

- **State-predicate triggers** (`T-DEPLETE`, `T-MID`): re-post when the
  predicate clears at knowledge time.
- **Event triggers** (`T-FLOW`): a point event has no "clearing" — re-post
  after a **cooldown** parameter, a grid dimension frozen in the protocol.
- **Re-entry is a fresh placement decision from current state.** If the BBO
  moved during `[t, t+lag+τ)`, there is no memory of the old level: the new
  quote joins the back of the *current* same-side touch (the `SKEW_LB`
  convention). Cancel-and-stay-out runs only as a labelled diagnostic bound.
- **Partial fill then cancel** is a named row type: the filled part is a
  retained fill, the unfilled remainder an excised quantity, one order — the
  counterfactual accounting of §5.3 carries all three columns.

### 4.6 Side asymmetry, and one recorded tension with skew

Cancellation is primarily an **adding-side / flat-state** instrument. A
reducing-side fill is one we *want* — the alternative exit costs ~2.25 ¢/share —
so the natural composition arms cancel rules on the adding side and when flat;
the reducing side quotes through.

**The tension, recorded not resolved:** the measured skew fronts the reducing
side on formation, and at the front the queue-consumption warning channel is
zero. Skew buys inventory control partly by giving up warning time where it
quotes most aggressively. Whether that costs anything real is for the composed
replay (§8.3).

### 4.7 Self-trade sequencing (review iteration 1)

A dump (`CROSS`) into the side where our own quote rests can lift our own
order — paying the taker fee to trade with ourselves and silently shrinking the
intended reduction. **Rule: any `CROSS` is preceded by `CANCEL` of our resting
quote on the crossed-into side, and the Decision's action order is preserved by
the Actuator.** Whether the venue itself rejects self-matches is an unverified
venue fact (module plan §1.3); this sequencing rule does not depend on it.

---

## 5. Estimand and evaluation discipline

**The harness exists.** `edge_layer1.py` replays a resting two-sided quote with
lagged state, queue accounting via displayed depth, and per-fill markout
decomposition. A cancellation policy is **one more rule** in that replay.

1. **Fresh pre-registration.** A `CANCEL_POLICY_PROTOCOL.md` frozen before
   measurement: the trigger grid (§4.3 parameters and cooldowns), the `lag+τ`
   ladder, re-arm/re-post semantics (§4.5), verdict bars. **Every cell is
   reported**; a surviving cell is a *candidate* for forward days, never a
   result.
2. **Three axes, mandatory**: retained-fill count and spread capture;
   fill-conditional markout at the h-ladder; terminal `|net|` and side-aware
   cash at risk. **Plus, for any state-dependent arm: flip/cancel/re-post
   counts as a reported cost axis** — the harness prices placement flips at
   zero, so the count is the only honest proxy until actuation is real
   (§8.3, review iteration 1).
3. **Excised fills get a counterfactual row** (the avoided fill's would-have-
   been markout is known in replay), partial-fill rows per §4.5, and the
   retained/excised populations are compared — that difference is the policy's
   value and the direct test of non-selective excision.
4. **No look-ahead.** Triggers are functions of the tape strictly before the
   fill, at knowledge time, through the frozen 250 ms lag.
5. **Both queue bounds** (`FRONT` / `BACK_DISPLAYED`).
6. Standing rules unchanged: per-coin (verdict = btc/eth), R-DUAL,
   `provenance(days_sampled)`, window-clustered CIs only, gap/tick-touched
   fills as `UNAVAILABLE` rows, a selftest control that must fail if the rule
   is vacuous.
7. **Everything is `DEVELOPMENT`.** Promotion of any candidate follows the
   flow-model shape: complete forward days it was not tuned on, executed as
   the registered plugin under EV-Replay (module plan §6.1), not as a harness
   transcription. **No PnL or capacity claim** — mechanism-first stands; Layer
   1 and Layer 2 stay separately accounted.

---

## 6. The dump threshold — backstop where skew is measured, side-aware in its recorded form

Under a mean-variance penalty `(γ/2)·N²·p(1−p)` against dump cost
`N·[0.07·p(1−p) + h]`:

```
dump when   N > N*(p) = (2/γ) · [ 0.07 + h / (p(1−p)) ]
p = 0.50   N* = 0.090 · (2/γ)        p = 0.10   N* = 0.126 · (2/γ)   (1.4×)
```

Tolerate a larger imbalance in the tails; dump more readily at ATM. `γ` is a
risk-appetite choice; report across a ladder. Fee and half-spread constants are
SP-Venue facts read through handles; the numbers here are exposition.

**Side-awareness (review iteration 1).** `p(1−p)` is the *variance* of the
binary payoff and is side-symmetric; the *worst case* is 9× side-asymmetric
(state plan §2). Both are real, so the recorded diagnostic is the pair: the
variance form above, and a side-aware form keyed on unpaired cost basis at risk
(`L_adv`), which fires earlier on the expensive side. **The side-aware scenario
cap binds independently and wins every conflict** — with contingent exposure of
resting quotes included in what it prices (module plan §2). `N*` in either form
is a diagnostic.

**Status:** with `SKEW_ROBUST` measured, the dump on btc/eth is a **backstop
against burst accumulation**, not the primary control. It remains load-bearing
at the terminal decision and on any coin where skew is too slow — every
non-verdict coin, out of scope anyway.

---

## 7. The terminal condition — retraction made explicit, halt interaction stated

The deadline is hard and the liquidity to meet it is measured to collapse:
count intensity → ~18 % of peak; USDC/arrival 15.5 → 24.0 (btc); notional peaks
in the first 5 s inside `r = 60` then declines ~9.5× to settlement.

```
r > 60      normal regime; skew-first, cancel per §4, dump only above N*
r ≈ 60      DECISION POINT — last moment with peak notional available.
            CANCEL the adding side. Reduce |net| to the size intended to be
            carried to resolution (CROSS is reducing and permitted).
r < 60      REDUCING-ONLY BOOK: the reducing side may rest, and a reducing
            CROSS may fire, BOTH SIZED ≤ |net| — an EXPLICIT HARD rule of the
            REDUCING-ONLY state (module plan §2), and DEFINITIONAL: the state
            means never past flat, full stop. Iterations 3–5: the new-risk
            predicate cannot substitute for this cap IN EITHER DIRECTION —
            whether a flip past flat moves contingent L_adv up or down depends
            on size and the price pair ((s−|net|)·basis_new vs
            |net|·basis_old; L_adv is dollar cost basis, not share
            magnitude), so the predicate sometimes ADMITS a flip (e.g. net
            +10 Up at ~0.50, reduce sized 18: flip to 8 ≈ $4.00 < $5.00) and
            sometimes over-refuses one. Two successive derivations of the cap
            from L_adv arithmetic were false; none is needed. The adding side
            must not rest — the oracle cannot veto a fill on a quote already
            resting (iteration 1). DO NOT plan to dump below r≈60.
r → 0       carry the residual. A CHOICE made at r ≈ 60, not a failure to act.
```

**One definition of "new risk", owned by the module plan §2:** an action is new
risk iff it increases contingent `L_adv` (position plus worst-case fill of all
resting quotes). The `r<60` rule and the halt edge use the *same* predicate;
the retraction of already-resting adding quotes at `r≈60` is the **scheme's**
action, not the constraint's — a feasibility oracle only refuses new actions.

**If HALTED when `r≈60` arrives** (halt is latched; `cancel_all` has fired; all
venue actions including reducing `CROSS` are blocked because halted state
cannot be trusted to trade on): the residual is carried to resolution. **That
is the designed degradation, bounded by `L_adv`, not an unhandled path**
(module plan §2/§5b).

**Caveat that could move the timing (still unmeasured):** the notional peak was
located on the withdrawn uniform grid. If it moves under the non-uniform grid,
§7's timing is wrong even though its logic is not. `r = 60` itself is an
SP-owned parameter with exactly this provisional provenance.

---

## 8. Measurable today, in priority order

1. **The warning-window distribution, policy-free — on the permissive
   envelope.** For every Layer-1 fill: `W` = time from the first
   **envelope event** to the fill, under both queue bounds, jointly with
   post-fill drift. The envelope is deliberately parameter-free — `E-FLOW`:
   any aggressive same-side trade at our level or better; `E-DEPLETE`: any
   decrease in depth ahead; `E-MID`: any adverse mid tick (iteration 5 aligned
   names and the "or better" to the frozen protocol §1.2) — so no
   trigger-parameter choice can shape it (review iteration 1: a parameterised
   "policy-free" measurement leaks the joint distribution into the later grid).
   **The branch statistic is the share of negative drift on fills with
   `W > lag + τ` per rung** — the lag is in the threshold, and the `τ=0` figure
   is the lag-floor share, not "zero latency". If the envelope share is small,
   the family is dead on this venue and §9.1 fires cheaply. The envelope is an
   upper bound on every parameterised trigger, so the residual leak from seeing
   it biases toward *running* the grid, never toward passing it — recorded.
   **Population caveat (iteration 3):** the `edge_l1_v1` fill set is ONE UTC
   day (§1), so `W`'s distribution and every CI are within-day; whether that
   population suffices for the branch, or the 680-windows/coin 4-day re-sample
   runs first, is part of the coordinator's freeze (R-ADMISS). The envelope
   definitions are stated ONCE, in the list above, matching the frozen
   protocol §1.2 (iteration 6 removed a duplicate statement of the same rule —
   the drift class this loop has caught twice).
   **ANSWERED 2026-08-23 under Ruling R-1 (`ww_v1`, receipt
   `derived/warning_window_v1.json`): dead at the decision rung on both coins
   (`R(τ=250)` 15.3 % / 14.3 % vs `f*_low` 30.9 % / 49.4 % — margins 12.5 pp
   and 32 pp), and additionally at the lag floor with a THIN margin on btc
   (`R(τ=0)` CI upper 29.7 % vs 30.9 % = 1.2 pp; eth comfortable at 23 pp).
   R-9 corrected the earlier flat "doubly dead" phrasing — the operative
   verdict is not close; the τ=0 claim on btc is. Descriptive, one UTC day,
   structural reading of an upper bound. The §8.2 grid is NOT built; §9.1
   fired. R-9 further ordered the day-series re-run under the frozen bar —
   **RUN AND ANSWERED (receipt `warning_window_v1_dayseries.json`): DEAD at
   the decision rung on ALL FOUR era days, both coins** — btc `R(250)`
   12.0–17.1 % (worst within-day CI upper 21.9 % vs the 30.9 % bar), eth
   8.3–14.3 % (worst 17.5 % vs 49.4 %), ex-micro agreeing, shuffle control
   non-vacuous every day, conformance 1,680/1,680, `days_sampled` n=4.
   2026-08-20 was NOT special (it was btc's second-highest-R day; eth's R
   declines across days). **The lag-floor subclaim on btc is DROPPED
   entirely**: 08-23's within-day CI upper (32.1 %) crosses the bar —
   exactly the fragility R-9's margin caution named — and the decision-rung
   verdict never needed it. The day-robust negative stands; still
   within-day inference only (4 day observations; the 8/8 consistency IS
   the robustness statement).**
2. **The cancellation replay grid** (contingent on 1): freeze
   `CANCEL_POLICY_PROTOCOL.md` — trigger parameters, cooldowns, `lag+τ`
   ladder, re-arm semantics, partial-fill rows, verdict bars — then run on the
   static `JOIN` book under §5.
3. **The composed policy** (contingent on 2): skew × cancel in one replay,
   three axes **plus flip/cancel counts**: does cancelling the adding side
   strengthen the skew's drift or destroy its capture; does the
   fronted-on-formation reducing side's zero-consumption-warning exposure show
   up per side; does band-edge chatter generate a cancel/re-post storm (the
   flat band's entry/exit hysteresis is a **policy** grid dimension here —
   actuation-level debounce must be identity in replay, module plan §5).
4. **`r ≈ 60` under the non-uniform grid** — is the terminal decision point
   real or an artefact of the confounded 60 s structure.
5. **Offset availability by `r`** — how often the reducing side fills within a
   horizon, per placement.

**Waiting on the calendar:** anything day-clustered (~10 days); promotion of
any §8.2 candidate (forward days, as the registered plugin); anything using a
fair value (Route A, PRICING HOLD); the settlement-estimand maker-edge sign
(~25–30× data).

---

## 9. What would falsify this design

1. **The unwarned share dominates at the lag floor.** Most negative drift
   arrives on fills with `W ≤ lag` (before any τ) → no cancellation policy on
   this tape rescues Layer 1 → the "never cancels" qualifier collapses and the
   honest Layer-1 verdict hardens. §8.1 is deliberately first so this dies
   cheaply if it dies.
   **FIRED 2026-08-23 (`ww_v1` under Ruling R-1): dead at the decision rung
   on both verdict coins, and additionally at the lag floor with a thin
   margin on btc (R-9's precise phrasing; the flat "doubly dead" is
   corrected).
   The mechanism is sharper than the falsifier's own wording: only ~12 % of
   fills are UNWARNED, but the median warning (~0.16 s) is shorter than the
   250 ms lag itself, and measured `R` sits BELOW the within-window shuffle at
   every rung — the largest adverse drift concentrates on the least-warned
   fills. It died cheaply, as designed: one probe, zero grid cells. On this
   tape: a passive maker loses at Layer 1 and no cancellation policy rescues
   it. Scope: descriptive, one UTC day; the re-sample robustness question is
   the coordinator's (R-ADMISS).**
2. **Excision is non-selective** (§5.3): cancelled fills' would-have-been
   markout ≈ retained fills' markout.
3. **The composition breaks skew** — including by flip-storm: if honest
   band-hysteresis and cancel interaction degrade the measured 76–89 %
   inventory reduction materially, the levers compete and the scheme must
   choose per state rather than compose.
4. **`r ≈ 60` is an artefact** of the confounded 60 s structure.
5. **Two-sided resting or fast cancellation is unavailable in practice** —
   venue rate limits, capital lockup, self-match handling, or cancel acks
   slower than the ladder's realistic rungs (an `OP`/`DE-Actuator` fact this
   plan consumes; module plan §5 names the observation mechanism).

## 10. What this plan deliberately does not do

- **Define the state** — [inventory state plan](DA_INVENTORY_STATE_PLAN.md).
- **Ask for an inventory-aware fill model** — plane ordering.
- **Assume profitability, or estimate PnL/capacity** — mechanism-first stands.
- **Set `γ`, the flat band and its hysteresis, trigger thresholds, cooldowns,
  or τ.** Choices and grids; report across ladders, promote nothing in-sample.
- **Use a fair price.** Everything here is inventory skew plus tape-reactive
  cancellation; independence from the sigma clock is much of why it can
  proceed now.
- **Model competition or the cancel race.** `FRONT`-bound and `lag+τ` results
  are labelled bounds; `BE-Competition` remains unbuilt and is consumed, not
  invented, here.
- **Own module wiring.** Contract names, ports, lifecycle states, capital,
  telemetry: [`DE_MODULE_PLAN.md`](DE_MODULE_PLAN.md).
