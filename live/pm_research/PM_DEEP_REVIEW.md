# PM_DEEP_REVIEW — adversarial deep review of P-2026-003

Reviewer brief: find what is wrong, circular, self-defeating or fatal in
`PM_MM_PLAN.md` (§1–§12), `PM_MECHANISM_EXPERIMENTS.md`, `PM_MECHANISM_THEORY.md`.
Prior reviews (`PM_SKETCH_REVIEW_ITER1_{T,S,M}.md`, `PM_REVIEW_ITER1.md`,
`PM_THEORY_CHECK_ORCHESTRATOR.md`) were read and their findings enumerated; nothing
below is a re-report of theirs. Everything marked **[measured]** was computed
during this review from `data/pm_5min/` and `data/mm_hf/`; scripts under
`/tmp/.../scratchpad/{lag,basis,trades,side,an2,an3,mid,as}.py`.

Sample used for all empirics: 2026-08-19 15:45 → 2026-08-20 05:10 UTC,
624,445 joined trade prints, 1,193 resolved windows, 7 coins, $6.66M notional
(BTC 82%). **~1.4 days — every number below is indicative, not a gate.** That is
itself part of the finding: the numbers that decide this program are computable
now and nobody has computed them.

---

## Summary of verdicts on the five posed hypotheses

| H | verdict |
|---|---|
| **H-1** variance is the wrong risk measure at the extremes | **CONFIRMED, and the defect is one exponent worse than stated** — `Q_max = κ/(γv)` is not a variance budget at all (it is a *marginal-reservation* budget, ∝v⁻¹ where a true constant-risk budget is ∝v⁻½), it is symmetric on an asymmetric payoff, and it authorises **25× the inventory and 25× the terminal variance** at `p̂ = 0.99` while the measured per-share edge rises only ~0.75 c and the loss-given-adverse doubles to 97 c (135:1). |
| **H-2** adverse selection worst where variance lowest | **NOT SUPPORTED on the correct (model-free) conditioning — but the plan's opposite claim is equally unsupported.** Conditioning on trade *price* appears to confirm H-2; conditioning on the *book mid at fill* (the decision-relevant, model-free state the ladder itself mandates) does not: net maker markout is **−0.03 c/share ATM and +0.72 c/share at `|mid−0.5| ∈ [0.46,0.49)`**. I retract the strong form. The measurement that would settle it is specified in FATAL-2(b) and **no experiment in the ladder performs it**. |
| **H-3** σ-fit and the calibration gate are entangled | **CONFIRMED, and sharper than stated** — since `E_t[X_T]` and `K` are both public, **σ is the model's *only* private input**, so "beat the book" ≡ "forecast σ better than the book", and σ is fitted by outcome-MLE while the gate is outcome loss. Needs a book-implied-σ arm. Specified below. |
| **H-4** latency should be a differential | **CONFIRMED, and the absolute number is also wrong by ~9×** — measured stream latency is 1,700 ms, of which only 257 ms is transport. §12.1 uses 120 ms. See FATAL-1. E-M7 does not measure the marginal sniper's latency. |
| **H-5** PM's relay may not be the fastest path to the TWAP | **REFUTED as an alpha threat, CONFIRMED as a model defect** — the un-synthesisable stream component has autocorr **0.984 at 2 s [measured]**, so a direct-Chainlink subscriber's 1.7 s head start is worth ≈0.08 bps of X. But the same measurement shows our own locked-TWAP integral is *permanently* 1.7 s incomplete, which §2's construction assumes away. |

---

# FATAL

## FATAL-1 — §12.1's latency budget is wrong by ~9×, and the EU siting decision rests on a premise §2 contradicts

**Measured [lag.py], 139,661 TWAP60 messages, 6 hours, 8 coins:**

| leg | p50 | p90 |
|---|---|---|
| Chainlink payload ts → PM publishes (`envelope.timestamp − payload.timestamp`) | **1,440 ms** | 1,920 ms |
| PM publish → our us-east-1 box (`recv_ns − envelope.timestamp`) | **257 ms** | 350–395 ms |
| **total observation lag** | **1,700 ms** | 2,330 ms (p95) |

§12.1 budgets `US-east ≈ 195 ms (120 see + 75 send)` and `EU ≈ 11 ms (8 + 3)`.
The measured "see" leg is **1,700 ms**, and **85% of it is PM-side publication
delay (1,440 / 1,700), not transport** — a leg no siting decision can touch. Full
budget: US ≈ 1,440 + 257 + 75 = **1,772 ms**; EU ≈ 1,440 + ~10 + 3 = **1,453 ms**.
That is an **18% cut, not the claimed 94%.**

Consequence for the participation frontier `r* = 3k²L/(m/φ)²`, which is **linear in
L** (×9.1 US, ×7.5 EU vs the assumed 195 ms):

| \|d\| | §12.1 claim (L=195 ms) | corrected (L=1.77 s, US) | corrected (L=1.45 s, EU) |
|---|---|---|---|
| 0.0 ATM | stop quoting at r = 123 s | **r\* = 1,117 s > 300 s ⇒ never quotable** | 916 s ⇒ **never quotable** |
| 1.0 | 92 s | 835 s ⇒ never quotable | 685 s ⇒ never |
| 2.0 | 13 s | 118 s | 97 s |

The entire benefit of migrating to London is moving the `|d| = 2` stop-quoting
line from 118 s to 97 s. The decision was taken on a table that is wrong.

**And the premise is self-contradicting.** §12.1's justification is: *"Decision
signal = the Chainlink TWAP relayed by PM's OWN `ws-live-data` endpoint, so BOTH
legs (see tick / send order) terminate at PM infra in London… had we stayed
Binance-anchored, EU would NOT help (Tokyo→EU is the long leg)."* But §2's own
fair-value construction is a **hybrid**:

```
level     ← live crypto_prices_twap_sixty
increments← Binance mid changes since the last stream tick   ← the fast leg
```

The high-frequency component of `p̂` — the thing that moves between 1 s stream
ticks, and the only thing a sniper can race — comes from **Binance**, not the
relay. Binance is therefore on the critical path, and moving to London
*lengthens* it (Tokyo→London > Tokyo→Virginia). So:

> Either the decision signal is the relay, in which case `L ≈ 1.7 s`, the frontier
> is 9× off and ATM is never quotable; or the decision signal is Binance, in which
> case §12.1's stated reason for migrating is false and EU is the wrong direction.
> §12.1 cannot have it both ways, and it currently asserts both.

Note also that the corpus carries **three live, mutually inconsistent values for
the same quantity**: 120 ms (§12.1), 471 ms (M-lens M4c, propagated into T-F12b
and T-F4b's "~15 c of unobserved p̂ motion"), and 1,700 ms
(`PM_MECHANISM_EXPERIMENTS.md` design table, measured). The 471 ms figure was
measured on `crypto_prices` (the Binance **spot mirror**, per `PM_REVIEW_ITER1` A)
and then applied to the TWAP stream, which is 3.6× slower.

**Fix.** (a) Publish a per-leg latency budget with the four distinct latencies
that actually exist — Binance→us, relay-publish, relay→us, us→PM-ack — and state
which one enters `L` in the frontier and why. (b) Re-derive the §12.1 table. (c)
Defer the siting decision until E-M7 measures which leg the PM book itself reacts
to. (d) Measure `us→PM order-ack` (P-M2a) before any frontier number is quoted;
it is currently an assumed 75 ms/3 ms and is the one leg we have never observed.

## FATAL-2 — the participation frontier has no revenue term, and `Q_max` multiplies size 25× where the edge-to-tail ratio is worst

This is H-1, plus a correction to my own H-2 prior.

**(a) The frontier is a pure safety condition — it contains no edge.**
`quote at level ℓ iff m(ℓ)/φ(d) ≥ k·√(3L/r)` is a statement about pickoff exposure
only. Nothing in it asks whether the states it admits contain any revenue. That is
a category error in an optimisation: a constraint has been promoted to a policy.

Measured, on the **model-free conditioning the ladder itself mandates**
(`|book_mid − 0.5|` at fill, 627,610 prints with a pre-trade mid, in-window,
share-weighted) [as2.py]:

| `\|mid−0.5\|` | mid ≈ | shares | net maker c/share | worst-case loss c/share | wins per loss |
|---|---|---|---|---|---|
| [0.00,0.05) | 0.50 (ATM) | 1,112k | **−0.03** | 50 | — |
| [0.05,0.15) | 0.35–0.45 / 0.55–0.65 | 2,203k | +0.78 | ~62 | 79 |
| [0.15,0.25) | 0.25–0.35 / 0.65–0.75 | 1,946k | +0.25 | ~72 | 288 |
| [0.25,0.35) | 0.15–0.25 / 0.75–0.85 | 1,860k | +0.75 | ~82 | 109 |
| [0.35,0.42) | 0.08–0.15 / 0.85–0.92 | 1,398k | −0.05 | ~89 | — |
| [0.42,0.46) | 0.04–0.08 / 0.92–0.96 | 1,050k | +0.73 | ~94 | 129 |
| [0.46,0.49) | 0.01–0.04 / 0.96–0.99 | 1,802k | +0.72 | ~97 | **135** |
| [0.49,0.51) | ≤0.01 / ≥0.99 | 50k | +0.39 | ~99 | 254 |

**I must retract a claim I had drafted.** Conditioning on the *trade price* (my
first cut) makes the far-from-money region look edge-free (−0.19 to +0.30 c/share
at `p ≥ 0.95`) and the `p ∈ [0.15,0.35)` band look enormous (+3.64 c/share). That
is the wrong conditioning: trade price is an outcome of the fill, not the state a
maker chooses to rest in, and a sweep prints far from the mid it started at.
Re-run on the mid, the picture is different: the net edge is **noisy, non-monotone,
~+0.45 c/share on average, ≈0 at the money, and NOT worse far from the money.**

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** Every figure in this block is
> **book-derived** and inherits the stale-book defect (`book` snapshots are p90
> **6.2 s** stale; read `price_change.best_bid/ask` instead). Rebuilt with **no
> book at all** — from trade price, taker side (G-FF1 `PASS`) and the settled
> winners field — the maker markout is **+0.17 c/share**, so `+0.45` is ~2.6x
> too high and `+95 bps / +136 bps` falls with it, being literally the same
> number. And the corrected figure is **NOT DISTINGUISHABLE FROM ZERO**:
> window-clustered bootstrap over 931 windows gives **+0.173 [-0.251, +0.596]**,
> with all seven per-coin CIs spanning zero on both weightings. **The maker-edge
> sign is UNDETERMINED at two days**, not positive. See
> `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b and STATUS `stale_book_contamination`.

So the plan's directional instinct — don't quote ATM — is *consistent* with the
data, and my adversarial prior (H-2) is not supported. **The 2–4× swings between
adjacent bins on 1–2 M shares each are also a warning that 1.4 days is nowhere near
enough to read this quantity, which is MAJOR-9.**

**What survives, and it is decisive.** With the corrected latency (FATAL-1), the
frontier admits `|d| ≥ 1.7` only while `r ≥ 118 s`, i.e. `t ≤ 182 s` — and
`|d| < 1.7` never. From the mid distribution [mid.py], the fraction of windows with
`|mid − 0.5| ≥ 0.455` is **0% at t = 30/60 s and ~10% at t = 120 s**. So the
corrected frontier leaves us quoting in roughly **5–10% of window-time-states**,
selected without reference to revenue, at a per-share edge of ~+0.7 c against a
~97 c tail. The plan has never computed the size of its own addressable state
space.

**(b) The H-2 measurement the ladder is missing, and why it cannot be skipped.**
The quantity that decides *where to quote* is the **capture ratio**: the markout a
maker actually realises on fills at a given book state, divided by the mispricing
that was unconditionally available at that state. The denominator is measured below
and is large. The numerator is the table in (a). No experiment in the ladder
computes either, let alone the ratio.

My attempt at the underlying decomposition (`net = spread capture + adverse
selection`, where `capture = ±(P − mid)` and `AS = ±(mid − Y)`) is **not
trustworthy and I am not reporting it as a finding**: measured capture is
+0.11…+0.50 c against observed spreads of 1.1–1.8 c, which is far below a
half-spread — a signature that the `price_change` carrying `best_bid/ask` for a
match can be emitted *before* the `last_trade_price` for the same match, so my
"pre-trade" mid has already moved toward the print. The **net** column is immune
(it never uses the mid); the split is not. **Fix:** the decomposition needs a mid
taken at a frozen lag (e.g. mid as of `t_trade − 250 ms`) and a check on WS
emission order — one line in E-M1, currently absent.

Book mid (Up token) vs realised outcome, all coins, 1,206 windows, at fixed
decision times. The book is **over-dispersed at every point in the window** — the
truth is always more extreme than the quote:

| t | mid bin | mean mid | realised | gap | se | n |
|---|---|---|---|---|---|---|
| 30 s | [0.15,0.30) | 0.236 | 0.142 | **−9.4 c** | 0.040 | 113 |
| 60 s | [0.15,0.30) | 0.231 | 0.168 | −6.3 c | 0.033 | 161 |
| 120 s | [0.70,0.85) | 0.786 | 0.862 | +7.6 c | 0.029 | 203 |
| 180 s | [0.85,0.95) | 0.908 | 0.953 | +4.5 c | 0.026 | 127 |
| 270 s | [0.95,1.00) | 0.982 | **1.000** | +1.8 c | 0.006 | 538 |
| 290 s | [0.95,1.00) | 0.982 | **1.000** | **+1.8 c** | 0.006 | 576 |
| 290 s | [0.00,0.05) | 0.018 | **0.000** | **−1.8 c** | 0.006 | 520 |

(Sample-level `mean Y = 0.531` — a mild Up tilt over 1.4 days — but the pattern
above is *dispersion*, not level: low bins under-realise while high bins
over-realise, so it is not an artefact of the drift.)

**The denominators are large and the numerators are small, everywhere.** At
`t = 290 s`, `mid ∈ [0.95,1.00)` the unconditional mispricing available to a maker
is **+1.8 c/share** (n = 576, zero flips); the realised markout on fills in the
matching state bin is **+0.72 c/share** — a **~60% haircut**. At `t = 30 s`,
`mid ∈ [0.15,0.30)` the unconditional gap is **−9.4 c** while the realised markout
in the matching bin is **+0.25 c** — a **~97% haircut**. Selection destroys most of
the available edge at *every* moneyness, and on this (crude, time-mismatched) cut it
is destroyed *more* at moderate moneyness — the opposite of H-2.

I therefore record H-2 as **untested, not confirmed and not refuted**, and note
that the plan's own opposing claim (M-lens M1e/M6: *"toxicity is ATM-driven, not
clock-driven"*, *"away from the money … the same 2.25 c becomes a genuine moat"*;
M7: *"the subsidy only survives on away-from-the-money, decided-window fills"*) is
**equally untested**. It rests entirely on a `φ(d)`-collapse argument — the
*underlying move required* to pick us off is larger far from the money — which is
correct and says nothing about the *realised* markout. Both claims are load-bearing
for where to quote, both are one afternoon's work to measure on data already on
disk, and neither is in the ladder.

**(c) `Q_max` is mislabelled and multiplies size in the worst cell.**
`PM_MECHANISM_THEORY.md` M-8(iii)1 and §11 Change 3 state
`γ·|q|·p̂(1−p̂) ≤ κ ⇒ Q_max = κ/(γ·p̂(1−p̂))` and call it *"a constant risk budget"*.
It is not. From `CE(q) = −(1/γ)ln(p e^{−γq}+1−p)`, expanding,
`CE = qp − (γ/2)q²·p(1−p)`: the **risk term is `(γ/2)q²v`**, so a constant risk
budget gives `Q_max ∝ v^{−1/2}`. `γ|q|v` is the *derivative* — the marginal
reservation shift, a price-units quantity. Consequences:

- at `p̂ = 0.99` the plan's cap permits **5× more inventory** than a genuine
  constant-risk cap, and the resulting terminal variance `q²v` is **25× the ATM
  value**, not constant;
- the cap is **symmetric in ±q** while the payoff is not: at `p̂ = 0.99`, long the
  favourite is a −99 c/+1 c position and short it is a +1 c/−99 c… the two sides of
  our own two-sided quote have opposite tails, and one cap governs both. The exact
  CARA `CE` the same document adopts *does* encode this asymmetry (it saturates in
  one direction and is linear in the other); the linearised cap that was actually
  written into §11/§12 discards it. §12.2 defect 2 noticed the divergence and
  proposed "a hard capital cap on top" — that is a patch on the wrong functional
  form.
- **the economics move the wrong way against the size rule.** From (a), going from
  ATM to `|mid−0.5| ∈ [0.46,0.49)` the per-share edge rises from −0.03 c to
  +0.72 c (~+0.75 c) and the worst-case loss rises from 50 c to 97 c (~2×), so the
  wins-needed-per-loss ratio *worsens* from undefined/negative to **135:1**. `Q_max`
  meanwhile authorises **25× the inventory** and **25× the terminal variance**. The
  sizing rule's steepness in `v` is not matched by anything in the measured payoff.
- **[measured, weaker conditioning]** splitting by taker direction on trade price,
  the pennies leg at the extreme is the loss-making one: in `p ∈ [0.98,1.00)`,
  maker-bought (taker SELL) = **−0.15 c/share** vs maker-sold (taker BUY) =
  **+0.45 c/share**. Directionally consistent with the asymmetry above; conditioned
  on price rather than mid, so treat as suggestive.

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** Every figure in this block is
> **book-derived** and inherits the stale-book defect (`book` snapshots are p90
> **6.2 s** stale; read `price_change.best_bid/ask` instead). Rebuilt with **no
> book at all** — from trade price, taker side (G-FF1 `PASS`) and the settled
> winners field — the maker markout is **+0.17 c/share**, so `+0.45` is ~2.6x
> too high and `+95 bps / +136 bps` falls with it, being literally the same
> number. And the corrected figure is **NOT DISTINGUISHABLE FROM ZERO**:
> window-clustered bootstrap over 931 windows gives **+0.173 [-0.251, +0.596]**,
> with all seven per-coin CIs spanning zero on both weightings. **The maker-edge
> sign is UNDETERMINED at two days**, not positive. See
> `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b and STATUS `stale_book_contamination`.

- `κ` has no calibration recipe (T-F10, still open), so the budget is unanchored
  in the one place where the functional form matters most.

**Fix (all three parts are needed).**
1. Replace the size constraint with a **loss-given-adverse-resolution cap**:
   `max(q_up·(1−p̂), q_down·p̂)` measured as the USDC lost *if the window resolves
   against us* ≤ `L_max`. That is a known, bounded, computable number and it is the
   quantity a Bernoulli position actually risks. Optionally add a fractional-Kelly
   overlay per fill: at the extreme cell, `edge/odds ≈ 0.72 c / 97 c ≈ 7e-3` of
   bankroll — versus a `Q_max` that permits 25× the ATM position.
2. Make the participation frontier an **edge-vs-exposure** frontier, not a
   pickoff-only frontier: `quote iff  ê(state) − ζ_snipe(ℓ) > 0`, with `ê` the
   *measured* per-state markout from (a). The current frontier has no revenue term
   at all, which is why it can select a state region without anyone ever asking
   what is earned there — and why nobody noticed that the corrected version leaves
   only 5–10% of window-time addressable.
3. Add the capture-ratio measurement (b) to the ladder — realised-fill markout ÷
   unconditional mid mispricing, by `|mid−0.5|` × time, with a frozen-lag mid.
   It is computable from data already on disk and it is what decides where to quote.

## FATAL-3 — the program has no stop condition, and the scope change removed the only rung that could produce one

Every branch of every experiment in `PM_MECHANISM_EXPERIMENTS.md` terminates in
*"the model is changed"*:

| experiment | worst-case outcome | what happens |
|---|---|---|
| E0 | FAIL | redesign collector, re-collect |
| E-M6 | REFUTE | re-derive the endgame model |
| E-M1(a) | R_touch<0.90 | delete replay arms, keep the rest |
| E-M2 | REFUTE | redefine the action set in cents |
| E-M3 | violation | relabel pair-harvest |
| E-M7 | "pure speed game" | *"colocate … or restrict quoting to the `|d|` region where φ(d) collapses"* |
| E-M8 | fail | add a settlement-risk term |
| E-M4 | rebate unpaid | strike a line from §1 |
| E-M5 | NOT-ELIGIBLE | strike M-5 from the inventory |

Not one branch says *stop*. The cut list is explicit about why: PM-E3/G3a/G3b —
whose FAIL branch read *"both negative, or sign flips across the queue bracket ⇒
**kill MM track**, fall back to calibration/taker studies or stop"* — were
**deleted** as "profit gates" under the §9 scope change. That deletion removed the
program's only kill switch. The S-lens had already flagged that "or stop" had no
owner; the response was to delete the sentence rather than give it one.

Three compounding problems:

1. **E-M7's escape hatch is empirically closed.** Its FAIL branch offers
   *"restrict quoting to the `|d|` region where `φ(d)` collapses"* as the
   alternative to colocating in a restricted jurisdiction. FATAL-2(a) measures that
   region's edge at ≈0. So E-M7's FAIL branch, read honestly against data already
   collected, *is* a stop condition — and it is written as a fork.
2. **The one cheap disqualifier was cut as out-of-scope.** PM-E4 was dropped
   ("Deployment, not research"), taking with it the only rung that would have asked
   *can we legally quote on this venue at all* (M-lens M4d: London is a restricted
   jurisdiction; US access unresolved). This is answerable in an afternoon and
   dominates all eight mechanism experiments. A program that cannot be stopped by
   "we are not allowed to trade here" is not decision-shaped.
3. **The rewards time-box and the rewards experiment are scheduled inconsistently.**
   §8/M3 makes H-PM4 *"a time-box on the whole program"* because the $550k/month
   pool is announced **for August only**. E-M5 — the experiment that asks whether
   these markets are rewarded at all — has earliest read **2026-09-03**, i.e. after
   the program it is testing has expired. Its natural-experiment arm (d) would then
   identify the program's *termination*.

**Fix.** Write a one-paragraph STOP clause into `PM_PREREG.md` with an owner and
pre-committed triggers, e.g.: *stop if (i) eligibility/jurisdiction is
unresolvable; or (ii) E-M7 returns cancel-before-trade ≤ 0.5 **and** the measured
markout in the frontier-selected `|d|` region has a bootstrap CI upper bound below
the tick; or (iii) E-M5 returns NOT-ELIGIBLE **and** E-M4 returns rebate-unpaid
(both subsidies absent) **and** the unsubsidised measured maker markout CI-lo ≤ 0.*
Note (ii) and (iii) require a PnL-shaped quantity, which the §9 scope change
forbids. **That is the real problem: a program that has forbidden itself from
measuring the only quantity that could end it.** Either re-admit one narrow
markout measurement as a stop-gate, or state explicitly and in writing that
P-2026-003 is a venue study with no trading decision attached and budget it
accordingly.

---

# MAJOR

## MAJOR-1 — σ_⊥ is double-counted, and it is not noise: it is a persistent level

§12.3 specifies two things that cannot both be right:

- `σ̂_eff(r) = κ̂(r) × shape(r)` where `κ(r) = Var_chainlink / Var_binance-synthetic`
  — an estimator of the **Chainlink TWAP's own variance**, which by construction
  already contains the basis innovation; and
- *"σ_⊥ (basis noise, ≈0.7 bps) added as **separate additive variance**, never
  fitted into the blend weights"*.

Adding `σ_⊥²` on top of a `κ(r)`-corrected Chainlink variance **double-counts the
basis**. This is structurally the same error T-F9 caught in `v(t)` (a term that is
already inside the total, added again), surviving one document later.

**Second, and worse, the basis is not noise [measured, basis.py]** — BTC, 16,644
aligned 1 s observations, 5 h, stream vs 60 s TWAP of Binance perp mid:

```
residual mean −3.83 bps, sd 2.29 bps      (slow level; cancels in settlement, not in our forecast)
300 s-demeaned "fast" residual sd  0.449 bps
autocorr(fast residual): 1 s +0.991 | 2 s +0.984 | 3 s +0.975 | 5 s +0.954 | 10 s +0.887 | 30 s +0.509
1 s innovation sd  0.060 bps
```

An AR(1)-like process with a ~28 s mean-reversion timescale. The variance it
contributes over a forecast horizon `r` is `≈ σ_⊥²·2(1−ρ(r))`, which is **~0.08 bps
at r = 2 s and ~0.44 bps at r = 30 s** — not a constant 0.7 bps. Using the constant
inflates `σ_eff` catastrophically late in the window: at 15%/yr and `r = 2 s` the
true settlement vol is `0.267 × √(r³/3w²) ≈ 0.007 bps`, so a flat 0.7 bps additive
term over-states `σ_eff` by **~100×**, collapsing `p̂` toward 0.5 in windows that
are already decided. §12.3's own sensitivity note says a 2× σ error is worth up to
23 c of `p̂`; this is a ~100× error.

The plan's headline claim *"at 15%/yr vol and r = 30 s the basis noise DOMINATES
settlement vol (0.42 vs 0.70 bps) — a real quiet-regime regime"* is right only at
`r ≈ 30 s` and by coincidence: the correct comparison there is 0.42 vs 0.44.

**Fix.** Drop the additive `σ_⊥` term entirely if `κ(r)` is estimated on the
Chainlink stream (it is already in there). If a synthetic-only fallback is kept for
stream gaps, model the basis as an OU process and add `σ_⊥²·2(1−ρ̂(r))`, with `ρ̂`
estimated per coin — an E-M6c deliverable that costs nothing since the variogram is
already computed. Also: `PM_MM_PLAN` §2's `σ_eff ← settlement vol ⊕ σ_⊥` (a plain
sum in quadrature) carries the same defect and must be amended with §12.3.

## MAJOR-2 — the model's only private input is σ, and σ is fitted on the gate's own objective (H-3)

`d_t = (E_t[X_T] − K)/σ_eff(t)`. After T-F12's stream anchoring, **`E_t[X_T]` and
`K` are both functions of public feeds** (the relayed TWAP plus Binance
increments), and `μ̂ = 0` by T-F13. Every competitor with the same two public feeds
computes the same numerator. **Therefore `p̂` can differ from the book on exactly one
axis: `σ_eff`.** The program's entire informational claim reduces to
*"we forecast 5-minute BTC volatility better than the Polymarket book does."*
Nothing in the corpus states this, and it is a much narrower and much less
plausible claim than the framing.

Given that, the circularity is real:

- §12.3: *"Fit by **MLE on realized winners** (= log-loss), not by regression on
  realized vol: the loss function is probability accuracy."*
- The (deferred) G2 gate: Brier/log-loss **on realized winners**.

Walk-forward (`fit ≤ d−1`) controls temporal leakage. It does **not** control
*model-class* selection: the blend structure, `κ(r)`, the decision to estimate `α`
in `Var ∝ r^α`, the seasonal, the coin effect and the choice of MLE-on-winners were
all chosen with the outcome series in view. And the deeper problem H-3 names is
not leakage at all: **if the book already embeds a good σ, an outcome-MLE fit will
simply recover the book's implied σ, and "agreement" will be scored as skill.**

There is also a self-inflicted wound: the S-lens's guard against exactly this —
*"add a walk-forward isotonic-recalibrated book mid as a secondary baseline; if the
model beats raw mid but not recalibrated mid, the 'edge' is public recalibration,
not information"* — survives only inside PM-E2, which the cut list **demoted to a
deferred diagnostic**. The guard was retained "verbatim … for the day it is run",
and that day has been removed from the schedule.

**Fix — add a book-implied-σ arm, specified.** For each window and decision time,
invert the book: `σ_book(t) = (E_t[X_T] − K)/Φ⁻¹(mid_t)` (`E_t[X_T]`, `K` from the
stream; mid model-free). Then, walk-forward:

1. **Agreement test.** Regress `ln σ̂_eff(t)` on `ln σ_book(t)`. Slope ≈1,
   intercept ≈0, R² high ⇒ we have recovered the book's own vol estimate; G2 cannot
   pass on information and must not be run as if it could.
2. **Direction test (the only real edge test).** Does the *residual*
   `ln σ̂ − ln σ_book` predict realised `|X_T − E_t[X_T]|` out of sample, day-clustered?
   If not, there is no σ edge and therefore no `p̂` edge, full stop.
3. **Recalibration control.** Third arm = isotonic-recalibrated book mid, restored
   from the S-lens spec. `p̂` must beat *that*, not raw mid.
4. **Note the measured headroom and its nature.** `mid.py` shows the book is
   over-dispersed at **every** decision time (mid 0.236 → realised 0.142 at
   `t = 30`, 2.4σ; mid 0.982 → realised 1.000 at `t = 290`, n = 576, 0 flips) —
   i.e. `σ_book` is biased **high throughout the window**, by 2–9 probability-cents.
   That is real headroom and it is the single strongest positive finding for the
   program's alpha claim. But it is a **public, stable, sign-stable, monotone**
   bias that arm 3 (isotonic recalibration of the mid) will capture with no model,
   no stream, no Chainlink and no latency budget. If our σ model's only win is this
   bias, the honest conclusion is "the book needs recalibration", not "we have an
   informational edge" — and the recalibration is available to every competitor on
   the same terms.

## MAJOR-3 — E-M7 is the track-deciding experiment and its key estimator answers the wrong question, with a possibly inverted sign

E-M7 P2: *"A slow maker can survive: resting quotes are more often **withdrawn**
than **run over** after a Binance burst"*, estimated as
`P(pre-burst touch level is cancelled before it is traded)`, with decision
**SLOW-MAKER-SURVIVES if ≥ 0.8**.

That statistic describes **the incumbent population's** behaviour, and the
incumbent population is faster than us (M-lens: top-5 wallets pay 21.5% of fees;
BCS/Aquilina say races are won in microseconds). It is not an estimate of what
happens to *our* quote, which by construction **cannot be cancelled within `L`**.

Worse, the sign may be inverted. A high cancel-before-trade rate means the fast
field gets out of the way before the sniper arrives. The sniper still needs a
counterparty. **The last quote standing is the slow one.** So
`cancel-before-trade → 1` is consistent with *"the slow maker absorbs the entire
race loss"*, which is the opposite of the decision the rule attaches to it.

The correct estimator for the stated proposition is a **shadow-quote** measurement:
place a synthetic non-cancellable quote at the pre-burst touch, and measure
`P(traded within Δ | burst)` and the realised markout — which is what the deleted
PM-E2.5 did, and what the cut list claims was "absorbed into E-M7(b)(c)". It was
not: E-M7(b) measures `λ_snipe` over the *whole book*, E-M7(c) measures the
*field's* cancels. The non-cancellable-quote arm is missing from both.

**Compounding: the sample excludes the states the experiment is about.** E-M7
explicitly excludes E0 slow-consumer windows and says so — *"they are the burst
windows — the exclusion is load-correlated by construction"* — but the decision
thresholds (`c_snipe ≥ 1.0 c`, `cancel ≥ 0.8`) are applied to the surviving calm
sample unchanged. A PASS on this sample is not evidence about bursts. E0's
`ρ(exclusions, RV)` diagnostic detects the problem and no decision rule uses it.

**Fix.** (a) Add the non-cancellable shadow-quote arm and make *it* the P2
estimator; the field-cancel rate becomes a descriptive covariate. (b) Bound the
bias from excluded windows: report the decision statistic on the excluded set using
whatever partial capture exists, or state the result as a one-sided bound
("survival ≤ X"). (c) Add the H-4 differential: estimate the marginal sniper's
latency from the *first* PM reaction to a Binance burst (E-M7(a)'s `L_50` is the
book's **mean** reaction, not the marginal racer's — the object needed is the
**left tail** of the PM-reaction distribution, e.g. p5 of first-reaction lag). Our
exposure is `(L_us − L_marginal-sniper)`, not `L_us`; nothing in the corpus
measures the second term.

## MAJOR-4 — the ladder gates a superseded §3, and twelve rounds of amendment have left live contradictions

`PM_MM_PLAN` §10 states plainly: **"§3 is superseded by a published model"** and
§9 notes *"§3 has NOT yet been rewritten"*. `PM_MECHANISM_EXPERIMENTS.md` then
gates on §3 repeatedly — E-M2's CONFIRM criterion is literally *"CONFIRM §3's
action set"*, E-M1's model consequence is *"whether queue position is a state
variable in §3"*, E-M8 validates *"§3's terminal-payoff formulation"*. **The
mechanism-truth ladder is verifying a model the plan has declared dead.** At
minimum this means every "model consequence" line in the ladder points at text that
will be rewritten before it is read.

Live contradictions found by direct comparison (line refs to `PM_MM_PLAN.md`):

| # | §3 / §2 says (still live) | superseded by | status |
|---|---|---|---|
| 1 | L168: pull-on-burst is *"the single most important control"*; L152 burst-flag pull rule | §11 Change 2 / M-7: *"cancellation is a race we lose by 2–3 orders of magnitude"* | contradiction live |
| 2 | L174: *"Rewards are booked as a **SEPARATE PnL line**"* | §10 (two-policy HJB) then §11 Change 1 (KKT shadow price) — superseded **twice** | contradiction live |
| 3 | L65: *"K = X_0, **KNOWN at window open**"* | §10 discovery 2: *"likely FALSE — TWAP recv−payload p50 ≈ 1.7 s"* | contradiction live |
| 4 | L117: *"books are 2–4 **TICKS** wide"* | §10 discovery 3 claims **"§2.4 corrected accordingly"** — **it was not**; §2 item 4 contains no 0.001 caveat | **a claimed-applied amendment that was never applied** |
| 5 | §3's `ζ(ℓ)` = markout + `Ê[|Δp̂|·1{burst}]` | M-7's `ζ_snipe(ℓ) = (λ_J/λ)·E[(|J|−m(ℓ))⁺]` with the fee **inside** the moat | contradiction live |
| 6 | §1 table row *"maker rebate ≈70 bps of notional per fill"*, presented alongside on-chain-verified rows | E-M4: *"**is the maker rebate actually paid?** … zero payouts in 14 days ⇒ the line is struck"* | unverified claim presented as ground truth |
| 7 | §1 row *"resting subsidy … $18.3k/day"* | §8 M2 CORRECTION: our markets are **absent from the rewards registry**; eligibility unconfirmed | unverified claim presented as a cost-table fact |

Item 4 is the serious one: an amendment log that records a fix which was not made
is worse than no log, because subsequent reviewers (including the ladder authors,
who then wrote E-M2 around the 0.01 tick) trust it.

**Fix.** Before the design freeze, do one mechanical pass: rewrite §2/§3 against
§10–§12, delete superseded text rather than layering, and re-point every "model
consequence" line in the ladder at the surviving text. Add a rule that an
amendment entry must cite the line it changed.

## MAJOR-5 — notional weighting is imported from PnL discipline into mechanism-identity tests, making every mechanism claim a BTC claim

The ladder inherits *"**notional-weighted** wherever a weighting exists"* verbatim
from the S-lens. That discipline is correct for PnL gates (dollars are the unit).
It is **wrong for identity tests**, which are the entire content of E-M6, E-M1(a),
E-M3(a). Concretely:

- E-M6 scores convention agreement **notional-weighted** with BTC at 82–85%. A
  convention that reproduces BTC at 99.9% and XRP at 95% passes at 99.1% pooled.
  But the settlement convention is a per-feed property (different Chainlink feeds,
  heartbeats, deviation thresholds) — there is no reason it is common across coins,
  and money-weighting is precisely the wrong way to find out.
- Same for `R_touch`/`R_exact` (E-M1a) and the mirror-identity rate (E-M3a).

**Fix.** For identity/mechanism propositions, report **per coin, equal-weighted
within coin**, with a minimum bar each coin must clear; use notional weighting only
where the quantity is economic (`c_snipe`, depth, markout). One sentence in
`PM_PREREG.md`.

## MAJOR-6 — two regimes carrying ~8% of the traded notional have no model at all, and one of them is measurably toxic

**[measured, an2.py]:**

| regime | notional share | maker gross | note |
|---|---|---|---|
| `t < 0` (pre-window, strike not yet formed) | **7.4%** | +0.70 c/share | T-F6a ("pre-open quoting has no formula") is still a *queued* SHOULD-FIX |
| `t > T` (post-window, outcome determined, book still live) | **0.9%** | **−1.46 c/share = −806 bps** | **no model anywhere in the corpus** |

The post-window leak is small in notional but large in kind: 350,929 shares traded
after `T` at prices whose outcome is already determined, costing the maker side
**$5,114 in ~1.4 days ≈ 8% of total maker gross**. Median trade lands at `T + 21 s`;
p95 at `T + 75 s`; resolution lands at `T + 85 s`. This is a pure stale-quote
harvest in a window the model does not know exists — §2's state space stops at `T`,
§3's pull rules stop at `T`, and E-M8 measures the redemption lag but explicitly
frames it as *"the capital-cycle statement — **not** a PnL quantity"*.

**Fix.** One line in §3: *cancel all quotes at `T − ε` and do not re-quote before
resolution*, and one estimator in E-M8: the post-`T` residual-quote markout, which
is the direct measure of whether the venue lets you get run over after expiry.

## MAJOR-7 — E-M6's headline deliverable `δ_jit` is not estimable under its own decision rule, and part of `b(·)` is not identifiable from our data

E-M6 defines the do-not-quote near-tie width as `δ_jit = p99 of |m_w| among
disagreements`, where disagreements are the residual after a convention that must
reproduce ≥99.0% of windows and ≥99.5% on `|m| > 0.5 bps`. At the stated yield
(~1,590 admissible windows/day) and a ≥99% agreement bar, the discordant set is
O(10) windows per day and the *within-convention* disagreements (the ones that
define `δ_jit`) are a subset of that. **A p99 estimated from tens of observations is
not a number**, and it is load-bearing: it is the width of the unquotable band and
feeds H-PM1/H-PM1c.

Separately, `PM_MECHANISM_THEORY.md` M-6(iii)1 defines the boundary reader as
`b = (last ≤ | first ≥) × (observationsTimestamp | validFrom)` — a 2×2 object. The
relay exposes **one** timestamp per tick (`payload.timestamp`). The
observations-vs-validFrom dimension is therefore **not identifiable from our data
at all**, and E-M6's grid silently drops it. If PM re-stamps on relay (a live
possibility given the 1,440 ms publish lag, FATAL-1), the reconstruction inherits an
unmodelled boundary offset.

**Fix.** Estimate `δ_jit` as a *model-based* quantity instead: the distribution of
`|X_T − X_0|` in the neighbourhood where our reconstruction and the official winner
could disagree given the measured stream jitter — i.e. propagate the tick-timing
uncertainty forward rather than counting rare disagreements. And state explicitly
that the timestamp-semantics dimension of `b` is unresolved, with a test:
cross-check `payload.timestamp` against the on-chain report round for a sample of
windows (E-M1e already has the Polygon join).

## MAJOR-8 — the design freeze precedes the data-integrity gate, and the "prereg-safe" measurement table is not

`PM_MECHANISM_EXPERIMENTS.md` sets **design freeze 2026-08-21** while **E0 —
the precondition that defines the admissible universe — reads 2026-08-22.** The
strata, thresholds and blacklist are frozen the day before we learn whether the
sample they apply to exists (current measurement: 79% admissible, "CONDITIONAL is
the expected state", with the `1013 slow consumer` defect on the coin carrying 85%
of notional).

And the table headed *"prereg-safe: coverage/variance/schema only, no
outcome-dependent quantity read"* is used, in the body of the document, to
pre-judge outcomes:

- E-M2: *"**Already contradicted in-sample.** `tick_size_change` fires 328 times…"*
- E-M6b: *"**this is very likely to fire**"*
- E-M5(a): *"**Direction matters and is already suggestive:** … depth sits outside
  1.5 c, which is evidence *against* a 1.5 c band"* — and the candidate band grid
  `{1.0, 1.5, 2.0, 3.0, 4.5}` was chosen after seeing it.
- E-M3(a)'s answer is already visible in the raw feed: `price_change` emits the
  *same size* at `0.01 BUY` on Up and `0.99 SELL` on Down in a single message, so
  the mirror identity is confirmed before the experiment runs.

None of these is fraud — they are honest disclosures. But an experiment whose
result is already known is not an experiment, and a freeze document that contains
outcome reads is not a freeze. **Fix:** re-label E-M2's tick finding, E-M3(a) and
E-M6b as *established facts, already adopted into the model*, and reserve the
experiment label for the parts that are genuinely open (tick-regime *occupancy
fraction*, quote half-life, the E-M6 convention winner).

## MAJOR-9 — theory investment is inverted relative to evidence, and the measured pie is lottery-shaped

Two observations that bear on how much further effort is rational.

**(a) The heaviest theoretical apparatus is attached to the least-evidenced
mechanism.** M-5 (rewards) carries principal–agent MM contracts (El Euch et al.;
Baldacci et al.; Aïd et al.), a Tullock contest, a KKT shadow price and a derived
occupancy frontier — for a program whose own status line reads *"the mechanism is
**unverified from every registry we have**"*, whose markets are absent from 16,172
registry rows, and which is announced for one month. Meanwhile the mechanism with
the strongest measured signal — the favourite–longshot bias in the book — has no
theory section, no mechanism number and no experiment.

**(b) The measured maker economics are positive but lottery-shaped [measured].**
Aggregate over all makers, ~1.4 days: taker gross **−95 bps** of notional ⇒ **maker
gross +95 bps**, +135.6 bps if the (unverified) 20% rebate is paid. That is a real
pie and it deserves saying — the maker side of this venue is *not* obviously
negative-sum, which is a point in the program's favour that no prior review
established. But:

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** Every figure in this block is
> **book-derived** and inherits the stale-book defect (`book` snapshots are p90
> **6.2 s** stale; read `price_change.best_bid/ask` instead). Rebuilt with **no
> book at all** — from trade price, taker side (G-FF1 `PASS`) and the settled
> winners field — the maker markout is **+0.17 c/share**, so `+0.45` is ~2.6x
> too high and `+95 bps / +136 bps` falls with it, being literally the same
> number. And the corrected figure is **NOT DISTINGUISHABLE FROM ZERO**:
> window-clustered bootstrap over 931 windows gives **+0.173 [-0.251, +0.596]**,
> with all seven per-coin CIs spanning zero on both weightings. **The maker-edge
> sign is UNDETERMINED at two days**, not positive. See
> `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b and STATUS `stale_book_contamination`.


- **49% of BTC maker gross comes from 5 of 171 windows.** Excluding the best 10,
  gross falls from $48.4k to $10.8k. Per-window `frac > 0` = 0.53, p10 = −392 bps,
  p90 = +597 bps.
- Per-hour, `mean +56 bps, sd 180, se 45, frac > 0 = 0.62` over 16 hours.
- The sign is not uniform across coins: **SOL −231 bps and BNB −86 bps to the
  maker**, BTC +98, ETH +138.
- This is the *aggregate* maker, which includes the fast makers who cancel in time.
  Our realised edge is the adversely-selected residual of it.

Consequence for the ladder: any gate on a quantity with this tail needs **tens of
day-clusters**, not the 5 (E-M6), 10 (E-M7) or 14 (E-M5) the read dates allow. The
ladder claims to inherit the S-lens discipline *"verbatim"* but did not inherit its
power analysis (S-lens: ≥28 days for G2, 4–6 weeks for heavy-tailed markout).

## MAJOR-10 — H-5 resolved, with a different defect exposed: the locked TWAP integral is permanently 1.7 s incomplete

The H-5 threat model (someone subscribes to Chainlink Data Streams directly and
sees `X_T` before PM's relay publishes it) is **refuted by measurement**: the
component of `X` that cannot be synthesised from Binance has autocorrelation
**0.984 at 2 s** and a 1 s innovation sd of **0.060 bps**, so a 1.7 s head start on
the stream is worth ≈0.08 bps of `X` — roughly one fifth of the settlement vol at
`r = 30 s` and negligible relative to the 2–4 c book. A direct subscriber's
advantage is real but small; it is not a structural disqualifier.

But the same measurement breaks a §2 assumption. §2 constructs

```
locked ← ∫ of the stream over the elapsed part of [T−w, T]
```

At time `t` we possess stream values only through `t − 1.7 s`. The locked integral
is therefore **always 1.7 s short**, and for `r < 1.7 s` we have observed *none* of
the final segment. T-F2's caveat anticipated this for *gaps*; it is not a gap
condition, it is the steady state. The plan's `Var_t[X_T] = σ²r³/(3w²)` presumes
the locked part is known; with a 1.7 s deficit the correct in-window variance has an
extra term for the unobserved-but-already-determined segment — which is
*epistemic*, not settlement, variance, and is exactly the quantity a direct-stream
competitor does not have.

**Fix.** Add the deficit explicitly: `Var_t[X_T] = σ²r³/(3w²) + (deficit term over
[t−Δ_relay, t])`, gap-filled from Binance with the basis-innovation variance from
MAJOR-1, and make `Δ_relay` a measured, monitored parameter (it is currently
assumed to be 471 ms in three documents and is 1,700 ms).

---

# MINOR

- **E-M8's "zero winner flips" is brittle.** Over ≥2,000 windows a single operator
  hiccup fails a CONFIRM whose alternative ("a residual settlement-risk term …
  materially damages the whole thesis") is drastic. Use a rate with a CI.
- **E-M2's REFUTE branch is near-certain to fire** (328 tick transitions / 130
  windows ⇒ the 0.001 regime is very likely ≥20% of window-time). Pre-commit the
  two-mode action set now rather than "discovering" it.
- **HYPE is in the frozen 7-coin universe but has no Binance leg**
  (`HYPEUSDT absent from data/mm_hf/`), so its E-M6 convention cells that need a
  synthetic comparator are undefined; state the per-coin cell coverage at freeze.
- **E-M6c's manipulation z-screen standardises a non-stationary series.** The
  stream-vs-synthetic residual has a drifting level (mean −3.83 bps, sd 2.29 bps
  over 5 h [measured]); a `z_max > 4` screen on the *level*, standardised on a
  trailing 24 h, will fire on ordinary basis/funding regime shifts and then impose a
  do-not-quote screen. Standardise the **innovation**.
- **§12.3's shape-estimation source is not in hand.** It specifies *"a synthetic
  60 s TWAP from Binance mid over **years** of Vision data"*;
  `data/mm_hf/vision/` holds **31 days of aggTrades only** (2026-07-18 → 08-17) and
  no mid series. Either re-scope to aggTrade-price TWAP (arguably closer to what an
  aggregate oracle reads) or budget the download.
- **`fee` in E-M4 is measured against `p` = trade price with 0.999–1.001 slope
  tolerance on ≥500 receipts**, but the ladder never states what happens if the
  *maker* leg is nonzero in the 0.001 tick regime — the fee formula's `p(1−p)` and
  the tick regime interact and no cell tests the interaction.
- **`PM_PREREG.md` still does not exist** while the freeze date is 2026-08-21 and
  the ladder says *"Harnesses read constants from this file, never inline"*.
- **The sign of the venue's central economic quantity rests on an unverified field
  semantic.** Every maker/taker number in this review (and any future markout)
  depends on `last_trade_price.side` being the **taker's** side. I verified it
  circumstantially [side.py]: of 54,569 BTC `BUY` prints, **63.7% print at the
  best ask vs 15.1% at the best bid** (the residual is stale-book, since the
  `price_change` carrying `best_bid/ask` can post-date the print). If `side` were
  the *maker's* side, the maker gross flips from **+95 bps to −95 bps** and the
  program is dead on arrival. `PM_REVIEW_ITER1` D4 noticed the skew
  (BUY 16,147 / SELL 2,994) and filed it as "an E1 item, not a bug"; E-M1 does not
  test it. **Fix:** E-M3(b)/E-M4 already decode `OrderFilled` receipts, which carry
  the maker and taker addresses — add one line to confirm that the WS `side` matches
  the on-chain taker's direction on the frozen ≥500-tx sample.

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** Every figure in this block is
> **book-derived** and inherits the stale-book defect (`book` snapshots are p90
> **6.2 s** stale; read `price_change.best_bid/ask` instead). Rebuilt with **no
> book at all** — from trade price, taker side (G-FF1 `PASS`) and the settled
> winners field — the maker markout is **+0.17 c/share**, so `+0.45` is ~2.6x
> too high and `+95 bps / +136 bps` falls with it, being literally the same
> number. And the corrected figure is **NOT DISTINGUISHABLE FROM ZERO**:
> window-clustered bootstrap over 931 windows gives **+0.173 [-0.251, +0.596]**,
> with all seven per-coin CIs spanning zero on both weightings. **The maker-edge
> sign is UNDETERMINED at two days**, not positive. See
> `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b and STATUS `stale_book_contamination`.


---

# The strongest argument that this program should stop

Stated as strongly as I can make it, and I believe most of it.

**1. The only thing we can privately know is σ, and we have no reason to think we
know it better.** After stream anchoring, `d = (E_t[X_T] − K)/σ_eff` has a numerator
built entirely from two public feeds. `μ̂ = 0` by decision. So the whole
informational thesis is a 5-minute BTC volatility forecast that beats the
Polymarket book's. This repo's own history is a catalogue of price/vol transforms
that do not generalise (`MEMORY`: "new alpha needs an ORTHOGONAL factor, not more
price/vol transforms"). The one measurable book bias we found —
`σ_book` too high late in the window — is public, monotone and capturable by an
isotonic map that any competitor can fit in an hour. If that is the edge, we do not
need §2, §3, or eight mechanisms.

**2. The design has been converging on numbers nobody checked, and the checks were
cheap.** Twelve sections of amendment drove the plan toward *"quote far from the
money, size large there, migrate to London to shrink `L`"*. Checked against data
already on disk: the latency budget is **9× too small** and the migration buys
**21 seconds** of quotable horizon at `|d| = 2` rather than the 12× the table
claims; the corrected frontier leaves **5–10% of window-time addressable** and was
never asked what is earned there; the sizing rule authorises **25× the terminal
variance** at `p̂ = 0.99` for **~0.75 c/share** more edge against a **97 c** tail;
and `σ_⊥` is double-counted and mis-shaped by **~100×** in the endgame. Four of the
plan's most consequential recent decisions are wrong or unsupported, and each check
took under an hour. That is a process failure, not a bad-luck failure: the program
built theory faster than it built arithmetic.

**3. Both subsidies are unverified, and the one that is time-boxed will expire
before its experiment reads.** The rewards program: our markets are absent from
16,172 registry rows; the pool is announced for August; E-M5 reads 2026-09-03. The
maker rebate — which the M-lens established is the *larger* of the two subsidies —
has never been observed being paid; E-M4's rebate arm also reads 2026-09-03. So the
entire subsidy case rests on two claims that will be tested after the window in
which they could have mattered.

**4. The measured pie is small, lottery-shaped and shared.** ~$45k/day of gross
maker markout across *all* makers on ~$6.6M/day of notional, with 49% of BTC gross
in 5 of 171 windows, and sign-flipping across coins. The M-lens's own honest
ceiling was $10–30k/month, "plausibly $0", on $25–100k of capital — against a
program that has now generated ~3,800 lines of design documentation, adopted ~30
academic references and taken a datacentre-siting decision, without a single
markout number being computed.

**5. It cannot end.** Every branch of every experiment says "change the model".
The gates that could have said "stop" were deleted as out of scope. The one cheap
disqualifier — *are we even allowed to quote here* — was cut as "deployment, not
research". A research program whose scope excludes both the quantity that would
kill it and the constraint that would forbid it is not a research program; it is a
commitment.

**The honest counter-argument**, which I also believe: the maker side of this venue
is measurably **+95 bps of notional gross** (+136 bps if the rebate is real), i.e.
~+0.45 c/share, positive in 6 of 8 moneyness bins and in 10 of 16 hours. That is
not what a picked-over market looks like, and **no prior review established it** —
twelve sections of design were written without anyone computing the sign of the
maker's markout. The mechanism behind it is visible and mechanically sensible: the
book is **over-dispersed at every decision time** (mid 0.236 → realised 0.142 at
`t = 30 s`; mid 0.982 → realised 1.000 at `t = 290 s`, 576 windows, zero flips),
which is the classic favourite–longshot bias — retail buys lottery tickets.

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** Every figure in this block is
> **book-derived** and inherits the stale-book defect (`book` snapshots are p90
> **6.2 s** stale; read `price_change.best_bid/ask` instead). Rebuilt with **no
> book at all** — from trade price, taker side (G-FF1 `PASS`) and the settled
> winners field — the maker markout is **+0.17 c/share**, so `+0.45` is ~2.6x
> too high and `+95 bps / +136 bps` falls with it, being literally the same
> number. And the corrected figure is **NOT DISTINGUISHABLE FROM ZERO**:
> window-clustered bootstrap over 931 windows gives **+0.173 [-0.251, +0.596]**,
> with all seven per-coin CIs spanning zero on both weightings. **The maker-edge
> sign is UNDETERMINED at two days**, not positive. See
> `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b and STATUS `stale_book_contamination`.


If the program were re-scoped to *"be a passive two-sided maker on the
over-dispersion, cap loss-given-adverse-resolution, don't quote ATM, don't quote
after `T`, and measure the markout for four weeks"*, it would be small, cheap,
falsifiable, and grounded in a measured prior — and it would need almost none of
the current apparatus: no Chainlink stream, no `r³` law, no principal–agent
contract theory, no siting decision. **That, not the mechanism ladder, is what the
data currently supports.**

---

# Recommended minimum before any experiment is read

1. Compute the H-2 capture-ratio table (realised-fill markout ÷ unconditional mid
   mispricing, by `|mid−0.5|` × time) on data already collected. **This decides
   where to quote and it currently has no owner.** ~1 day of work.
2. Re-derive the §12.1 frontier with a per-leg latency budget; suspend the siting
   decision.
3. Replace `Q_max = κ/(γ·v)` with a loss-given-adverse-resolution cap.
4. Fix the σ_⊥ double-count; add the book-implied-σ arm.
5. Write the STOP clause, with an owner, into `PM_PREREG.md` — which must exist
   before the freeze date it already claims.
6. Do the mechanical §2/§3 rewrite so the ladder stops gating a superseded model.
