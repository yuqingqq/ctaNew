# PM_STRUCT_ITER1_C — lens C (portability & anti-overfit), iteration 1

Object: `PM_ARCHITECTURE.md` (2026-08-20). Question: is this structure
over-fitted to *Polymarket 5-minute BTC binaries as they exist in August 2026*,
and what would it cost to move?

## Headline

**The module DECOMPOSITION is not over-fitted. The INTERFACES are.**

Every portability test below lands on the same five objects, never on the plane
boundaries: `WindowCtx`, `(q_up, q_down)`, `σ_eff(r)`, C2's action list
`{join, improve, rest back, out}`, and the fee schedule that lives inside C2's
EV rather than in a venue record. Planes, dependency direction and the
Belief/Decision split all survive four venue changes and an instrument-class
change unmodified. That is a good result and it should be said plainly: the
expensive thing (where the seams are) is right; the cheap thing (what flows
across them) is written in Polymarket-August-2026 vocabulary.

Corollary that sets the triage: all five are near-zero cost to fix **today**
because no code exists behind them, and monotonically more expensive after
D3/K1 are built — which is build-order step 1–2. This is the last cheap moment.

---

## Test 1 — different horizon, same venue (15-min, 4-hour)

Claimed near-free (§15 X8: "one-line slug pattern extension"). It is cheap, but
not free, and the gap is instructive.

| Module | Impact |
|---|---|
| **D0 (missing)** | market discovery, slug pattern, window open/close, token→side map — **is not in the register at all**; lives in `collect_pm.py`'s discover loop. First thing every portability test touches. |
| D1/D2 | unchanged |
| D3 | unchanged if `T` is genuinely a `WindowCtx` field (it is) |
| **B2** | **new impl.** `σ²(τ − w + w/3)` is a BM approximation calibrated at τ ≤ 300 s. At 4 h the averaging correction is negligible, vol-of-vol and a term structure are not. Correct outcome — one module, no interface change. |
| **B1** | **silent-reuse hazard.** `μ̂ = 0 by default` is justified by T-F13 ("needs ~0.4 %/day-equivalent to matter") — a bound computed at 5 min. Over 4 h the same drift is 48× more material. The default is horizon-scoped but not *marked* as horizon-scoped. |
| **C1** | rule SET changes, not rule parameters. At 4 h, `L = 1.7 s` vs `r = 14,400 s` makes the participation frontier non-binding for ~99 % of the window; the binding constraint becomes inventory and drift. C1 currently lists a fixed conjunction of four rules. |
| **C4** | **interface pressure.** Aggregate is `Σ_c |q_c|(1−p̂_c)`, indexed by COIN. A 4-h BTC window contains 48 5-min BTC windows on the same underlying: the exposures are strongly dependent and the aggregate under-counts. Needs a factor key, not a coin key. |
| E3/E4 | horizon is the clustering unit: 288 windows/day → 6/day. Block-bootstrap block size and every gate's effective N are horizon-scoped parameters. |

**Verdict: cheap, 2 new impls + 1 missing module + 1 interface change.** The
real finding is not the cost, it is that *nothing in the structure makes the
horizon-scoped assumptions fail loud*. A 4-h port would inherit μ̂ = 0, the
5-min σ fit, the latency budget's relevance judgement and the FLB curve
silently. K1 has provenance (`measured/fitted/assumed`) but **no validity
domain** — see S-C6.

## Test 2 — different settlement mechanic

How much is hardcoded to "60 s TWAP, Up iff X_T ≥ X_0, ties Up"?

Structurally, the good news: the settlement *physics* is correctly isolated —
target in B1, variance law in B2. A 30 s TWAP is a parameter. A different
oracle is a D1 impl + a σ_⊥ re-measure. Those port at one module each.

Two real problems:

**(a) The snapshot rule (Polymarket's own, pre-2026-08-07) is singular, not
parametric.** `Var = σ²r³/(3w²)` divides by `w²`; the branch structure
`t ≤ T−w` vs `t > T−w` collapses at `w = 0`. So `w` is not a parameter that
spans the venue's own history — it is a discrete impl choice. Acceptable
(one module), but it must be *declared* as such, because §2 already tells us
`w` changed under this program 12 days before the architecture was written.

**(b) The settlement rule is entered FIVE times — an R-ONCE violation in
exactly the form R-ONCE exists to prevent.** The single fact "Up iff
b(X_T) ≥ b(X_0), ties Up, 60 s TWAP" is independently encoded in:

| where | as |
|---|---|
| B1 | `K`, `E_t[X_T]` |
| B2 | the branch structure and `r³` law |
| C1 | the near-tie band (`δ_tie`) |
| E2 | reconstruction of the winner as ground truth |
| X2 | redemption timing (~+85 s) |

Change the comparator, the tie rule, or `w` and five modules must agree. R-ONCE
is stated for *risk and noise sources*; it is not stated for *venue and
instrument facts*, which is why this slipped through. Note also that the
boundary-reader `b(·)` and the tie width `δ_tie` — flagged in
`PM_MECHANISM_THEORY.md` M-6 as **estimated parameters** — appear nowhere in
the module register at all.

**Verdict: 1 new impl (fine), 1 MUST-FIX (the five-way duplication).**

## Test 3 — different venue, same instrument class (Kalshi)

| Module | Impact |
|---|---|
| D1 | new impl (expected, cheap) |
| D2/D3 | unchanged |
| B1/B2/B3 | unchanged — crypto binaries settle on a comparable statistic |
| B4/B5/B6 | unchanged (price-time CLOB, same theory) |
| **C1** | rewards-band occupancy rule and the whole M-5 principal–agent layer → **null**. No rewards program. |
| **C2** | **fee model is venue-specific and is baked into a strategy module.** PM: maker $0 + ~70 bps rebate, taker ∝ min(p,1−p). Kalshi: fee on both sides, different curve, different rebate policy. There is **no fee module in the register** — fees exist only inside C2's per-level EV. |
| **C5** | **null.** Kalshi collapses the complement into one book (a "No" is the other side of the same order, netted by the exchange) — there is no second book, no CTF, no mint/merge. |
| **X2** | half survives. X2 currently glues *resolution accounting* (universal) to *CTF on-chain ops* (venue). |
| E1/E3/E4 | unchanged. **E2 → null** (still has ground truth actually; survives). |
| K1/K2/K3 | unchanged |

The important question the prompt asks: **is an absent module OK, or does the
pipeline break?** The architecture does not say. There is no policy for null
implementations. This matters far beyond Kalshi, because **the architecture is
already running with three null Belief modules today** — B4/B5/B6 are
`placeholder`, and B6's null is `ζ = 0`, i.e. *no adverse selection*, which is
the single most optimistic assumption available in this program and is
silently the default. That is structurally the same shape as FATAL-2 (a
constraint used with the revenue term absent and unremarked).

**Verdict: survivable, no plane restructuring — but exposes three missing
seams** (fee schedule as venue data; venue capability flags; a null policy).

## Test 4 — the Binance sibling (P-2026-002) — most informative

Can the perp program run on this architecture? Module by module:

| Class | Modules |
|---|---|
| **unchanged** (13) | D1 (`BinanceWS` is *already a listed impl*), D2, D3 (minus WindowCtx), B4, B5, B6, C2, C6, E1, E3, E4, K1, K2, K3 |
| **new impl, same interface** (4) | B2 (σ law), **B3 → `BookOnly` + propagator, already a listed impl**, C4 (jump-VaR), X1 |
| **null** (4) | B1, C5, X2, E2 |
| **INTERFACE CHANGE** (4) | D3's `WindowCtx`; C3's `reservation(q_up,q_down,p̂)`; B2's `σ_eff(r)`; C4's per-coin aggregation key |

Against `PM_VS_MM_THEORY_DIFF.md`'s seven structural differences, six are
absorbed *inside* modules with no interface consequence:

| DIFF item | Where it lands |
|---|---|
| 1 finite horizon / binary payoff | C3 penalty term + B1 null — **not** C2. The architecture's split (EV argmax in C2, reservation in C3) is exactly what makes terminal-vs-ergodic an impl detail. |
| 2 inverted tick regime | *same conclusion both ways* — discrete per-level EV. C2 unchanged. |
| 3 flipped fee sign | C2 parameter — **once fees are a venue record**; today it is an edit. |
| 4 different AS generator (GM vs BCS) | B6 impl. Defence differs (region+size vs width) → C1 rule set. |
| 5 complement structure | C5 null |
| 6 σ's role | *consumer* difference, not producer. B2's contract survives. |
| 7 subsidy-as-obligation | C1 rule null |

**Answer: yes, it could host the sibling, at the cost of four interface changes
— and all four are changes I want for reasons independent of the sibling
(tests 1, 2, 3 and 5 each demand at least one of them).** That is strong
evidence against over-fit: the 80 % of shared theory maps onto shared modules,
and the 20 % that differs is absorbed by impl swaps and nulls, not by edits to
neighbours.

The one genuinely hostile object is `WindowCtx{t0, T, w, K, coin, tick,
rewards params}`. A perp has no window. `WindowCtx` is (i) *mandatory* output
of D3, (ii) threaded through B1, B2, C1, C2, E3, and (iii) mixes three layers
in one struct — `t0/T/w/K` are instrument, `tick/rewards params` are venue,
`coin` is universe. It is the single most over-fitted object in the document.

**SHOULD both programs run on it? Share Data + B4/B5/B6 + Evaluation +
Cross-cutting. Do NOT build a shared Decision plane.** The DIFF's item 3 says
Binance perp MM is *arithmetically dead* at reachable tiers (fee = 4–70× the
half-spread). Building generality to host a known-dead program is generality
with no user. The sibling test's value here is **diagnostic** — it proves the
seams are in the right places — and should be spent on that and nothing more.

## Test 5 — no continuous queue (batch auction) / no book at all (AMM, LMSR)

**Batch auction.** B5's `λ_fill(ℓ, Q_ahead)` is intrinsically a continuous-time
queue object; in a batch there is no time priority and fill is set by the
clearing price with pro-rata at the margin. `Q_ahead` also leaks into C6's
requote rule (`ΔV > D_ℓ(Q_ahead)`) and into B5's bracket. So a queue-free venue
ripples B5 → C2 → C6. Fix is to make the fill primitive
`fill(price, size, state) → (P_fill, E[markout | fill])` and keep `Q_ahead` as
*internal state of the CLOB implementation* rather than public signature. Note
the pleasing side effect: batch auctions eliminate sniping by construction
(that is Budish's entire point), so `ζ_snipe → 0` is an impl swap, and the C1
participation frontier goes null. The Belief plane survives.

**AMM / LMSR (Polymarket's own history).** `MarketState{book, queue_est}` is
empty. B3's `BookOnly` and `BookPlusFLB` impls die — but **`StreamModel`
survives completely untouched.** This is worth stating because it inverts a
criticism: the DIFF's §4 treats "we ignore the book we trade on" as the
program's central gap, and it is — *for edge*. For **portability**,
stream-anchored fair value is the most transportable module in the stack, and
the Belief plane survives a bookless venue precisely because B3's interface
never mentions the book.

What does break: **C2's action set.** `{join, improve, rest back, out}` is CLOB
vocabulary written into the module *definition*. On an AMM the actions are
`{add liquidity, remove, set range}`; in a batch, `{submit limit at p, size s}`.
C2 should be an optimizer over a **venue-supplied ActionSpace**, not an
enumerator of four CLOB verbs. One fix, and it is the same fix that test 3
(fees) and test 1 (rule set) want.

**Verdict: Belief plane survives both. Decision plane needs the action space
injected rather than enumerated.**

---

## Venue / instrument / strategy layering

The prompt's diagnosis is correct: B1 is instrument, C5 is venue, C1 is
strategy, and they sit in planes organised by **dataflow** with no second axis.
Dataflow planes are the right *runtime dependency* rule and should be kept. What
is missing is an orthogonal **rate-of-change** axis. Venue facts change on the
venue's schedule (the settlement rule changed 2026-08-07; the rewards band was
re-cut 2026-08-20 — `collect_pm.py:67` already carries a comment that Gamma's
`rewardsMaxSpread`/`MinSize` are stale). Instrument facts change when we trade a
different contract. Strategy changes weekly. Mixing all three inside module
definitions means a venue's unilateral change edits strategy code.

### Proposed layering

Two new first-class records, owned by K1, consumed everywhere, restated nowhere:

```
VenueSpec@version
  matching        {price-time | pro-rata | batch(Δ) | AMM(invariant)}
  tick lattice, min size, max size
  fee(side, price, size), rebate(side, ...)          ← removes fees from C2
  rewards program | null  (band params, scoring rule, epoch)
  rate limits, cancel cost, min resting time
  ActionSpace                                        ← removes the 4 verbs from C2
  capabilities {mint/merge, self-match, post-only, hidden}  ← makes C5 conditional
  latency legs                                       → K2
  settlement/redemption delay

InstrumentSpec@version
  payoff(outcome)            binary(1,0) | linear | ...
  settlement statistic S     TWAP(w) | snapshot | VWAP(w) | oracle-report
  strike rule + boundary reader b(·)                 ← currently homeless
  comparator + tie rule                              ← currently in 5 modules
  horizon T | perpetual
  parity relations           Up+Down=1 | none        ← what C5 keys off
  unit / notional convention
```

Then `WindowCtx` = `InstrumentSpec ⊗ (t0, realised K, staleness)`, renamed
`InstrumentCtx` with an **optional** horizon; `tick` and `rewards params` move
out of it into `VenueSpec`; `coin` moves into the universe/factor record.

**New rule R-LAYER:** *no module may contain a literal fact from the venue or
instrument layer. Porting to a new venue changes VenueSpec + one X1 impl.
Porting to a new instrument changes InstrumentSpec + B1/B2 impls. A strategy
change touches neither.*

**New rule R-VERSION:** *VenueSpec and InstrumentSpec entries carry provenance
(same taxonomy as K1) **plus** a validity window; every stored dataset and every
E3 replay is stamped with the spec version in force, and E3 refuses to replay
across a spec boundary without an explicit flag.* This is not hypothetical:
pre-2026-08-07 data is both a different settlement rule and a contaminated game
(§2), and today nothing in the structure prevents E3 from replaying straight
across that boundary. The repo's standing era-mixing discipline (CLAUDE.md
pitfalls) applies here and has no home in the register.

Single fix, four problems dissolved: the five-way settlement duplication
(test 2), fees-inside-C2 (test 3), the C2 action list (test 5), and half of
`WindowCtx → InstrumentCtx` (test 4).

---

## Over-generalisation warnings — what must stay deliberately concrete

A program of this size dies of premature generality as easily as of over-fit,
and this one has one venue, no API access, an unresolved eligibility question,
and X1 does not exist. Explicit stop-signs:

1. **Do NOT build a multi-venue abstraction layer.** Write `VenueSpec` /
   `InstrumentSpec` as **versioned data records with provenance** — a config
   file and a loader, ~1 day. Do not write a `VenueAdapter` interface with two
   implementations against a venue we cannot yet reach.
2. **Do NOT generalise B1 to arbitrary payoffs or multi-outcome markets.**
   Scalar-statistic-vs-threshold covers 5-min, 15-min, 4-h, snapshot, 30 s TWAP
   and Kalshi crypto binaries. Categorical and range markets (which Polymarket
   also has) buy nothing until one is traded. Keep `K, E_t[X_T]`.
3. **Do NOT actually build the Binance host.** Test 4 is a *diagnostic*. The
   fee-sign inversion means that program is dead at our tier; hosting it is
   generality with no user. Take the four interface fixes, leave the rest.
4. **Do NOT abstract the liquidity model** (CLOB/AMM/batch variant types).
   Record the seam — `MarketState.book` optional, ActionSpace injected — and
   implement CLOB only.
5. **Do NOT build a comparator/tie DSL.** One enum and one boolean field.
6. **Keep D1 concrete and ugly.** Wire code should not be abstracted; each
   venue's WS is its own mess and D2 is the boundary that absorbs it. Resist
   any "unified feed interface".
7. **Do NOT make E2 venue-neutral.** It exists because prediction markets have
   ground truth — that is a *structural advantage*, not an inconvenience to
   abstract away. Optional, yes; generic, no.
8. **Do NOT split C1's rules into a plugin framework.** A list of small
   predicate objects composed by AND is sufficient; anything with a registry,
   priorities or dynamic loading is over-built for four rules.

The honest tension with this loop's own scorecard: three of my MUST-FIXes are
*interface* changes touching 4–6 modules, which reads as a large blast radius.
The justification is timing, not tolerance — they cost ~zero while no code
exists behind them, and they are on the critical path of build-order steps 1–2
(K1, then D3). After D3 is written, `WindowCtx` is load-bearing in five modules
and the same change is a refactor. There is no third option where they get
cheaper.

---

## Triage

### MUST-FIX

| id | fix | why now |
|---|---|---|
| **M-C1** | Introduce `VenueSpec` + `InstrumentSpec` as versioned, provenance-carrying records owned by K1; adopt **R-LAYER**. Strip venue/instrument literals from B1, B2, C1, C2, C4, C5, E2, X2. | Dissolves the 5-way settlement duplication (an R-ONCE violation), fees-in-C2, and the C2 action list. Prerequisite for K1, which is build-order step 1. |
| **M-C2** | Adopt **R-VERSION**: spec entries are time-versioned with provenance + validity window; datasets and E3 replays are stamped; E3 refuses to cross a spec boundary unflagged. | The venue already changed its settlement rule (2026-08-07) and its rewards band (2026-08-20) *during this program*. Nothing currently stops a replay across either. Data-integrity bug, not just portability. |
| **M-C3** | Adopt **R-NULL**: every module declares a null implementation with defined semantics **and a bias direction** (conservative / optimistic); an `assumed` null may not gate a decision. | Not hypothetical — B4/B5/B6 are null *today*, and null-B6 means `ζ = 0`, the most optimistic assumption in the program, undeclared. Same shape as FATAL-2. Also the mechanism by which C5/B1/E2/X2 absence is legal on other venues. |
| **M-C4** | Replace `(q_up, q_down)` / per-coin inventory with **exposure to named risk factors**; C3's reservation and C4's aggregate key on the factor vector. | One change covers Up/Down rank-1 (already the correct model per M-3's Bergault–Guéant reduction — the architecture states rank-1 in theory and rank-2 in the interface), cross-coin (§14 R1), cross-horizon (test 1), cross-instrument (test 4). |
| **M-C5** | Add the missing **D0 MarketDiscovery/Lifecycle** module: universe enumeration, slug/ticker patterns, window open/close, token→side mapping, spec resolution at discovery. | Exists in `collect_pm.py` but not in the register; it is the first module every one of the five tests touches, and it is where the horizon extension actually lives. |

### SHOULD-FIX

| id | fix |
|---|---|
| S-C1 | `σ_eff(r)` → `σ_eff(h)` over a forecast horizon; `h = τ` is the binary case. |
| S-C2 | C1 as a composable set of individually-togglable predicates with reasons, not a fixed four-way conjunction. Which rules are *active* is horizon- and venue-dependent. |
| S-C3 | C2 optimises over a venue-supplied `ActionSpace`; `{join, improve, rest back, out}` is one instance. |
| S-C4 | B5's public signature drops `Q_ahead`: `fill(price, size, state) → (P_fill, E[markout|fill])`. Queue is CLOB-impl internal state; the bracket stays on it. |
| S-C5 | Split X2 into universal `ResolutionAccounting` + venue-capability position transforms (mint/merge) exposed by X1. C5 becomes a `PositionTransform` planner reading `VenueSpec.capabilities` and `InstrumentSpec.parity`. |
| S-C6 | Add `valid_for` (horizon / spec version / regime domain) to every K1 parameter; reuse outside domain fails loud. Directly targets the μ̂ = 0 and σ-fit silent-reuse hazards in test 1. |
| S-C7 | Rename `WindowCtx` → `InstrumentCtx`, horizon optional; do it simultaneously with M-C1. |
| S-C8 | Give `b(·)` (boundary reader) and `δ_tie` (tie width) an owning module — currently they are named as estimated parameters in M-6 and appear in no module in the register. |

### NOTED (deliberately not fixed)

- Multi-outcome / range markets; AMM and batch liquidity models; a second X1;
  actually building the Binance host. Seams recorded, implementations not.
- The DIFF's §4 "we ignore the book we trade on" is an **edge** problem, not a
  structure problem. Structurally, B3's book-free interface is the asset that
  makes tests 3–5 pass.
- Plane boundaries and the downward dependency rule: **no change proposed.**
  They survived all five tests.
