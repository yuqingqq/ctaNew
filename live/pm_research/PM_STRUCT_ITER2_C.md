# PM_STRUCT_ITER2_C — lens C (portability & anti-overfit), iteration 2

Object: `PM_ARCHITECTURE.md` **v2** (rewritten 2026-08-20). Prior: `PM_STRUCT_ITER1_C.md`.
Question this lens owns: is the structure over-fitted to Polymarket-5-min-Aug-2026 —
**and, now that v2 has answered by adding a spec layer and three rules, is it
over-*generalised*?**

## Headline

**Portability: v2 passes. 1 hard interface change across five tests (v1 had 4),
0 plane/boundary changes, 0 modules with a venue literal that the tests trip
over.** The five objects I named in iter-1 (`WindowCtx`, `σ_eff(r)`, the four
CLOB verbs, fees-inside-C2, `(q_up,q_down)` keying) are gone, parameterised, or
demoted behind an interface that no longer mentions them. The decomposition was
never the problem; the vocabulary was, and the vocabulary was rewritten.

**Over-generalisation: v2 is NOT overbuilt in its rules or its records. It is
overbuilt in FRAMING and mis-ordered in BUILD SEQUENCE.** Five of the seven
rules cost ≈0 for the next real step; one (R-KNOW) *is* the next step's whole
point and its input (`recv_ns`) is already on disk; exactly one
(R-ONCE/`VarianceBudget`) is off-path and should not be built yet. The ceremony
risk is not the rules — it is §8 putting three infrastructure steps in front of
the first measurement on 3.9 GB of data we already have.

**The sharpest finding.** v2's best idea — knowledge time ≠ event time — was not
applied to the layer v2 added. `valid_from/valid_to` is an **event-time** record
of when a spec changed. On disk right now, `markets.jsonl` carries
`rewardsMinSize`/`rewardsMaxSpread` from Gamma *and* `rewards_authoritative`
from the CLOB registry, because Gamma served a stale band after the 2026-08-20
re-cut. Two values for one field at one instant, with our knowledge of the
change lagging the change. Dates cannot express that. **A content hash +
`observed_at` can, and is cheaper than a date algebra.**

---

## 1. Five portability tests, re-scored against v2

| # | test | new SP records | new impls | modules unchanged | **INTERFACE changes** | v1 |
|---|---|---|---|---|---|---|
| 1 | 15-min / 4-hour, same venue | 1 SP-Instrument version | 2 | all | **0** | 1 |
| 2 | snapshot / 30 s TWAP / other oracle | 1 SP-Instrument version | 1–2 | all | **0** | 0 (+1 MUST) |
| 3 | Kalshi | 1 SP-Venue + 1 SP-Instrument | 2 | all | **1** (`fee_schedule`) | 1 |
| 4 | Binance perp sibling | 1 + 1 | 3 | all | **0** | 4 |
| 5 | batch auction / AMM-LMSR | 1 SP-Venue | 2–3 | all | **0 hard, 1 soft** | 1 |

**Total: 1 hard + 1 soft interface change, versus 4 in v1.** No test moves a
plane boundary, and no test requires a new module.

### Test 1 — 15-min & 4-hour, same venue

Now a **data** port, not a code port. `SP-Instrument{horizon: 900|14400}` is a
new versioned record; `constraint_set[]` and `objective_terms[]` are
`SP-Strategy` lists, so v1's "the *rule set* changes, not the parameters"
(the C1 interface-change row) is a spec edit. `BE-Uncertainty` gets a new impl
for term structure and vol-of-vol; `VarianceBudget.register()` is horizon-blind
so no signature moves. The per-coin risk key I attacked in iter-1 (M-C4) is
still per-coin, but it now sits *behind* `DE-Constraints.feasible() -> {max_size,
shadow_price}` — the interface no longer names the representation, so a
factor-keyed exposure is an impl swap. **M-C4 downgraded to SHOULD-FIX: v2
fixed it structurally without fixing it literally.**

**What did NOT get fixed, and it is the one that costs money.** iter-1 S-C6
asked for a validity domain on every parameter. `SP-Params` has
`{value, provenance, owner_module, measured_at, source}` — `measured_at` is a
*timestamp*, not a *domain*. So a 4-hour port silently inherits `μ̂ = 0` (whose
justifying bound, "needs ~0.4 %/day-equivalent to matter", was computed at
τ = 300 s and is 48× more material at 4 h), the 5-min σ fit, and every gate's
block size. Nothing fails loud. This is the highest-probability port in the set
and the fix is one field.

Minor: the slug/ticker pattern and token→side map have no home. `DA-Discovery`
exists (iter-1 M-C5 landed) but the pattern is a venue/instrument *fact*, so it
will be written as a literal inside the impl unless `SP-Instrument` gains a
`discovery{}` block.

### Test 2 — different settlement mechanic

Clean pass, and the right-sizing is exactly right. `statistic: TWAP(w)|SNAPSHOT`
is **an enum, not a DSL** — my iter-1 stop-sign #5 honoured. 30 s TWAP is a
field edit. SNAPSHOT (the venue's own pre-2026-08-07 rule, where `Var = σ²r³/3w²`
is singular at `w=0`) is a `BE-Uncertainty` impl, correctly a discrete choice
rather than a parameter. A different oracle is a `DA-Feeds` impl plus a σ_⊥
re-measure into `SP-Params`.

The five-way settlement duplication is **dissolved by placement** —
`{source, statistic, w_declared, strike_rule, tie_rule}` now exists once. The
`w_declared` vs `w_hat` split landed and is the model for what should have
happened to `b(·)`: `strike_rule` is in the spec as a *declared convention*, but
`PM_MECHANISM_THEORY.md` M-6 names the boundary reader `b(·)` and tie width
`δ_tie` as **estimated** parameters. Same one-name-two-quantities bug, one
instance fixed. (SHOULD-FIX.)

**But the dissolution is by convention, not by enforcement** — see §2(b). R-SSOT's
enforcement clause names `SP-Params` only.

### Test 3 — Kalshi

Everything I asked for in iter-1 landed and works: `capabilities{}` lacking
`CTF_PAIR` removes mint/merge from `DE-ActionSpace` *derivationally* rather than
by nulling a module; no `MAKER_REWARDS` nulls the rewards objective term and its
band constraint together; `EV-Calibration` is untouched because ground truth
still exists.

**The one hard interface change in the whole review: `fee_schedule(p, side)` has
no `size` argument.** Polymarket's taker fee is `∝ min(p, 1−p)` **per share** —
size-linear, so `(p, side)` suffices *there and only there*. Kalshi's is
`0.07 · N · p(1−p)` rounded **up per order**: non-linear in size, with a rounding
floor that dominates at small size. Two consequences:

1. It is the **same bug v2 explicitly fixed elsewhere**. §4: "`size` is an
   argument — v1 omitted it, structurally forbidding own-impact." The same
   omission was reintroduced in `SP-Venue`.
2. It bites **today**, not on Kalshi. A per-level EV at min-size needs the fee
   *at that size*; and PM's rebate is band- and size-dependent by construction.

Fix is one signature: `fee(instrument, side, price, size)` — which also resolves
the layering question in §4 below.

### Test 4 — Binance perp sibling (the informative one)

v1: four interface changes. **v2: zero.** The three objects that broke are gone:
`WindowCtx` deleted, `σ_eff(r)` no longer a public signature, `C3.reservation()`
replaced by `DE-Solver.maximize(...)`. The decisive improvement is
`DE-Objective` as pluggable Σ terms: **funding carry is a new term and touches
one module.** In v1 this class of change (rewards) was scored STRUCTURAL by lens
A precisely because no module owned the objective. Funding and rewards are now
structurally the same kind of thing, which is the correct generalisation and it
was earned by a real change, not anticipated.

Field-set gap: `SP-Instrument.settlement{}` and `horizon T` read as mandatory. A
perp has neither. `settlement: Option`, `horizon: T | PERPETUAL`. Two words.

**Unchanged from iter-1: do NOT build it.** Fee sign inversion means Binance
perp MM is arithmetically dead at reachable tiers (`PM_VS_MM_THEORY_DIFF` §1.3:
fee = 4–70× the half-spread). This test is a **diagnostic** and its entire value
is the zero above.

### Test 5 — batch auction / AMM-LMSR

`matching: PRICE_TIME|BATCH` covers the batch case; sniping → 0 by construction
becomes a `BE-FlowAndFills` impl and the participation constraint becomes a
declared null. AMM/LMSR needs a third enum member; `StreamModel` survives
untouched (still the most transportable module in the stack), `BookOnly`/
`BookPlusFLB` become nulls.

**The soft interface change: v2 now has two action vocabularies.**
`DE-ActionSpace` produces `{quote(level,size)|cancel|mint|merge|cross|wait}`
*derived from venue capabilities*, but `BE-FlowAndFills.rest(level, size,
horizon, state)` re-enumerates `(level, size)` independently. On an AMM the
action is `set_range`, which `rest(level, size, …)` cannot express. The clean
form is `rest(action: Action, state)`, which (i) removes the parallel vocabulary,
(ii) makes the fill model automatically follow a venue-derived action space, and
(iii) is *simpler*, not more general. This is a consequence of v2 correctly
making ActionSpace venue-derived and forgetting its consumer.

Also required for the enum to stay honest: **the action type must be an open sum
extended per venue, not a closed enum compiled into `DE-Solver`.** One sentence.

---

## 2. Fix verification

| iter-1 | demand | v2 | verdict |
|---|---|---|---|
| **M-C1** | VenueSpec + InstrumentSpec as versioned records; strip literals | SP-Venue / SP-Instrument | **landed**, right-sized (data, not adapter) |
| **M-C2** | R-VERSION, provenance + validity window, replays stamped | R-VERSION; `valid_from/valid_to`; EV-Replay refuses to span | **landed, insufficient** — see (d) |
| **M-C3** | R-NULL with a **bias direction**; assumed null may not gate | R-NULL: null names its assumption | **half-landed** — see (c) |
| **M-C4** | factor exposure, not `(q_up,q_down)`/coin | still coin-keyed, but behind `feasible()` | **structurally sufficient** → SHOULD |
| **M-C5** | D0 MarketDiscovery | DA-Discovery | **landed** |
| S-C2/3/4/7 | composable constraints · injected ActionSpace · drop `Q_ahead` · kill `WindowCtx` | all present | **landed** |
| S-C1 | `σ_eff(r)` → `σ_eff(h)` | signature removed entirely | moot |
| S-C5 | split X2 into universal accounting + venue transforms | mint/merge became actions; **accounting deleted** | **regression** — §5.1 |
| S-C6 | validity domain per parameter | absent | **not landed** — MUST |
| S-C8 | owner for `b(·)`, `δ_tie` | `strike_rule`, `tie_rule` (declared only) | **half-landed** |

### (a) Is the field set correct and minimal?

Minimal: yes — ~30 fields across four records, no field without a named prior
failure. Correct: four gaps, all one-liners.

| record | missing | why it matters |
|---|---|---|
| SP-Venue | `size` in `fee_schedule` | test 3; and per-level EV at min size *today* |
| SP-Venue | redemption/payout delay + capital lockup | no owner anywhere in v2 (§5.1) |
| SP-Venue | cancel cost, min resting time | the rewards band has a resting requirement; `DE-Actuator` hysteresis needs both |
| SP-Instrument | `discovery{slug_pattern, token_side_map}` | test 1; otherwise a literal in `DA-Discovery` |
| SP-Instrument | `settlement` optional, `horizon: T\|PERPETUAL` | test 4 |
| SP-Params | `valid_for` (validity domain) | test 1; the μ̂=0 / 5-min-σ silent-reuse hazard |
| SP-Params | **`scope`** (global \| venue \| instrument \| **market**) | §4 rewards; per-market fee; per-coin σ. Keyed by `name` alone, these collide |
| SP-Params | `ci`/`se` alongside `value` | `w_hat = 56.31 ± 5.17 s`; a fitted param's uncertainty currently has no home |

Not missing, correctly excluded: latency legs (moved to `OP-LatencyBudget` —
better, they are *our* path not the venue's); a `VenueAdapter`; a comparator DSL.

### (b) Does `capabilities` + R-REQUIRES actually fail loud?

**The mechanism is right; the enforcement has two holes.**

1. `capabilities:{CTF_PAIR, NEG_RISK, MAKER_REWARDS, ...}` — the `...` is
   load-bearing. For R-REQUIRES to fail *loud* rather than *silently pass*, the
   capability namespace must be **closed**: absent ⇒ `false` (fails closed,
   good), and a module requiring an **unknown name** must be a wiring **error**,
   not a miss. As written, a typo'd or newly-invented capability name is
   indistinguishable from an unsupported one. One sentence fixes it.
2. **No module in the register declares a `requires:` list.** The rule has an
   enforcement point and no data to enforce against. Four declarations would
   cover it: mint/merge → `CTF_PAIR`; the rewards objective term and its band
   constraint → `MAKER_REWARDS`; `DE-Actuator` → `rate_limits`, `POST_ONLY`.

### (c) Is `SP-Strategy.nulls{module: declared_assumption}` enforceable?

**Mechanically yes, substantively no** — and the gap is FATAL-2 shaped.

"Wiring fails unless each null names its assumption" is checkable (non-empty
string per null). But a string is not a guard. The live case: null
`BE-FlowAndFills` ⇒ ζ = 0 ⇒ the `adverse_selection` objective term evaluates to
0 ⇒ the solver quotes *more* aggressively. That is the most optimistic
assumption available in the programme, and R-PROV does **not** catch it, because
R-PROV gates on `SP-Params.provenance == assumed` and ζ = 0 is not a param — it
is a module's *absence*. **v2 has two rules that each half-cover the shape, with
a seam between them.**

Right-sized fix — one field and **reuse of an existing check**, no new machinery:

```
nulls { module: { assumption, bias: CONSERVATIVE | OPTIMISTIC | UNKNOWN } }
DE-Solver refuses an objective term or gate whose value derives from an
OPTIMISTIC or UNKNOWN null   # same refusal R-PROV already implements
```

Note v2's own §7 already applies this judgement informally — Option B carries
`nulls = {}   # ζ must be measured, not nulled`. Make it structural.

### (d) Is `valid_from/valid_to` sufficient, or is a HASH needed?

**A hash is needed, and the evidence is on disk.**

`valid_from/valid_to` is an **event-time** claim: *the spec changed at time t*.
Three ways that fails here, all already observed:

1. **The venue changes specs without announcing dates.** The rewards band was
   re-cut 2026-08-20 mid-programme; the settlement rule changed 2026-08-07.
   Neither arrived with a boundary timestamp — we inferred them.
2. **Our knowledge of the change lags the change.** `collect_pm.py:65` exists
   solely because of this: "Gamma's `rewardsMaxSpread`/`MinSize` are stale — the
   band was re-cut 2026-08-20 and Gamma still served the old number." So
   `valid_from` is itself a *measured, revisable* quantity with its own
   `t_known`. **This is exactly R-KNOW, and v2 did not apply it to SP.**
3. **Two sources disagree about one field at one instant.** `markets.jsonl`
   records `rewardsMinSize`/`rewardsMaxSpread` (Gamma) *and*
   `rewards_authoritative` (CLOB registry) side by side. No date range can
   arbitrate that; a content hash of the record we actually used can.

Right-sized fix (~10 lines, cheaper than a date algebra):

```
spec_id = hash(content)            # unambiguous; the thing datasets stamp
observed_at, observed_source       # knowledge time of the spec itself
valid_from / valid_to              # keep — event time, best current belief
```

Datasets and `EV-Replay` stamp `spec_id`, not a date. Do **not** build a
bitemporal store; two extra fields and a hash.

---

## 3. THE OVER-GENERALISATION CHECK

### (a) Is the SP layer justified TODAY?

**Yes — on single-venue grounds alone, and the justification is not portability.**

| justification | is it hypothetical? |
|---|---|
| settlement rule entered 5× (R-ONCE violated in the form it forbids) | no — v1, measured |
| venue changed settlement 2026-08-07 | no — mid-programme |
| venue re-cut the rewards band 2026-08-20 | no — mid-programme |
| two live sources disagree on the band | no — both values on disk |
| `SP-Params` retires FATAL-1's class | no — build-order step 1 |

Zero of these require a second venue. Had `SP-Venue`/`SP-Instrument` existed on
2026-08-07, one record gains a version; without them, five modules must agree.
And v2 honoured the constraint that matters: §9 explicitly refuses a venue
abstraction layer, and specs are **data**. That was my stop-sign #1 and it held.

The part **not** justified today is the *unreachable enum branches* —
`BATCH`, `LINEAR`, `complement: None`. As spec **vocabulary** these cost one word
each and I will not demand their removal. As **code branches** they cost real
money. Binding rule: *record the alternative, implement one arm.* An enum with a
single member today is honest.

### (b) Do the seven rules slow the actual next step?

The next step is EV-Markout on what is on disk: 3.9 GB, 1,384 raw window files
across 20260819/20260820, 1,312 markets, 1,292 resolutions. Rule by rule:

| rule | cost to the markout measurement | verdict |
|---|---|---|
| R-SSOT | ≈0 — you need one place for tick/fee/`w` regardless | keep |
| **R-KNOW** | ≈0 *and it is the point*. `t_known` is already on disk: every raw line is `<recv_ns>\t<json>`. Without it, the measured 1.7 s relay gap prints free money (peek ≈0.08 bps vs settlement vol 0.007 bps at r=2 s) | **keep — this IS step 4** |
| **R-ONCE** (`VarianceBudget.register()`) | **off-path.** Markout composes no variance. Building the budget before `BE` exists is pure ceremony | **keep the rule, DEFER the machinery** |
| R-PROV | ≈0 — markout is measured from the tape | keep |
| **R-VERSION** | needed, and *small*: the 08-20 rewards re-cut lies **inside the 1.5 days on disk**, so era-partitioning is the first thing the measurement must do anyway (repo standing discipline). Cost: stamp two hashes | keep |
| R-NULL | ≈0 — no EV module is nulled | keep |
| R-REQUIRES | ≈0 — nothing is wired yet | keep |

**Five rules ≈0, one is the step itself, one is off-path.** The rules are not
the ceremony.

**The ceremony is §8's build order.** It places `SP-Params` + `OP-LatencyBudget`
(1), `SP-Venue/Instrument` (2), and `DA-Normalize` + `DA-State` (3) *before* the
first measurement (4). Steps 1–2 are days of typing that produce no number.
What EV-Markout actually needs from SP is: tick, fee, `w_declared`, and the
08-20 boundary — **four facts**. "Specs are data" implies you may write 20 % of
the data. Populate `SP-*` **demand-driven**: a field is added when a consumer
asks for it. Protection is unaffected — R-SSOT and R-VERSION govern *where* a
fact lives, not *how many* facts exist.

Step 3 stays where it is. `Known[V]` is not deferrable: a markout number
measured without it is a fake number, and its input is free.

One live self-inflicted block: **`EV-Replay` "refuses to span a spec version"
with no escape hatch, and a spec boundary sits inside the only dataset we have.**
Refusing is correct science (partition at 08-20 → ~788 / ~596 markets, both
usable), but a pooled-power run must be possible with the boundary and both
`spec_id`s recorded in the result. iter-1 M-C2 said "without an explicit flag";
v2 dropped the flag.

### (c) Is there a cheaper structure with the same protection?

**Yes, and it is mostly a framing change.** Everything R-SSOT/R-VERSION/R-REQUIRES
buy is delivered by:

```
pm_specs.yaml        # 4 records, ~30 fields, versioned entries with a content hash
specs.py  (~150 LOC) # load, hash, freeze, resolve: venue(id) instrument(id) params(name, scope)
```

Not a package. No base classes, no resolver hierarchy, no plug-in registry, no
second `VenueSpec` implementation. §1 lists SP as a **layer** in the layer map
and §2 calls it a **layer** ("Specs are data, not an abstraction layer" — good,
but the map disagrees). The dependency statement is right; the noun invites a
package. Say it in one line: *SP is one data file and one loader; if it acquires
more than the four record types it has failed.*

### Verdict and CUT list

**v2 is right-sized on rules and records; overbuilt in framing and mis-ordered
in sequencing. Cut four things, none of them a rule.**

| # | CUT | why |
|---|---|---|
| 1 | **§8 steps 1–2 as prerequisites.** Populate SP demand-driven; move EV-Markout ahead of full spec population | 4 facts needed vs ~30 fields typed; the data is already on disk |
| 2 | **`VarianceBudget` machinery from the near-term path** (keep R-ONCE as a rule) | nothing composes variance until `BE` exists; the only off-path rule |
| 3 | **The word "layer" for SP** → "one file + one loader", with an explicit size bound | prevents the package that would justify the "premature generality" charge |
| 4 | **Code branches for unreachable enum arms** (BATCH, AMM, LINEAR). Keep the spec vocabulary; implement PRICE_TIME/BINARY only | v2 §9 already says this in prose; make it binding on the enums |
| 5 | **Mark `DE-Solver` and `BE-Belief` impls BUILT vs NAMED SEAM** (use the ✅/⚠️ convention `DA-Feeds` already uses) | four listed solver impls read as four deliverables; only PerLevelEV is needed for Option B |

**Do not cut any of the seven rules.** Each has a named prior failure, five cost
≈0, and the two that cost anything (R-KNOW, R-VERSION) are load-bearing for the
very next measurement.

---

## 4. Venue / Instrument / Strategy layering — verification

**Cut correctly in all three contested places; two keying defects.**

**`settlement` on Instrument — correct.** It defines what the contract *pays*;
two instruments on one venue settle differently (5-min binaries vs event
markets), and the same definition could exist on two venues. `source` (the
oracle named in `resolutionSource`) is likewise part of what the contract pays →
Instrument. Correctly *not* duplicated: the oracle's **operational** properties
(1,440 ms publish, 257 ms transport) are feed facts and live in
`DA-Feeds`/`OP-LatencyBudget`. `w_declared` on Instrument, `w_hat` in
`BE-Uncertainty` — correct, and the model for the `b(·)` fix.

**The venue half of settlement is missing.** *How the venue pays out* —
redemption delay (~T+85 s measured), capital lockup until redemption, the
post-`T`-pre-resolution phase — is a **Venue** fact and has no home in v2 at all.
See §5.1.

**`fee_schedule`/`rebate_schedule` on Venue — correct, with a caveat.** The curve
is the venue's and applies across its markets. But PM's `∝ min(p, 1−p)` is a
**binary-payoff-specific functional form** — meaningless on a linear instrument;
Kalshi's is likewise binary-specific. Resolution: the schedule stays on Venue,
its **argument list** comes from the instrument:

```
SP-Venue.fee(instrument, side, price, size) -> Cost
```

One signature; fixes the layering *and* the missing-`size` interface change.

Second caveat: the **effective** fee is observed per market —
`last_trade_price` carries `fee_rate_bps` on the wire. So `SP-Venue.fee_schedule`
is a venue **default** and the observed per-market rate is an `SP-Params` entry
with `provenance = measured`. Without a `scope` key on `SP-Params` these two
collide — an R-SSOT hazard created by getting the layering right.

**Rewards — v2 splits them three ways and the split is CORRECT.** Worth stating
because it was the homeless object in v1:

| aspect | layer | v2 | verdict |
|---|---|---|---|
| programme exists at all | **Venue** | `capabilities.MAKER_REWARDS` | ✅ |
| band params (max_spread, min_size, rate/day, epoch, scoring) | **Venue programme, instrument-scoped, time-varying** | *nowhere* | ✗ |
| value in the objective | **Strategy** | `DE-Objective.rewards` term | ✅ |
| band as a constraint **with a shadow price** | **Strategy** | `DE-Constraints` | ✅ (the FATAL-2 fix) |

The gap is **keying, not placement**. The band params are per-`condition_id`
(the CLOB registry is paginated to ~16 k rows, refreshed every 10 min) and
re-cut mid-programme. They belong in `SP-Params` with
`provenance = measured, source = CLOB registry` — but `SP-Params` is keyed by
`name` alone. Lens A's "there is no SCOPE axis" finding, resurfacing inside the
layer that was added to fix scope problems. One field: `scope`.

---

## 5. NEW portability problems the rewrite introduced

**5.1 — Resolution / redemption accounting was DELETED (regression).** v1 had
X2. v2 has `mint | merge` as *actions* (correct, capability-gated) but **no
owner** for: redemption delay and capital lockup, realised-PnL accounting at
resolution, the post-`T`-pre-resolution window (8.3 % of notional, one window
measured at −806 bps), and `EV-Calibration`'s ground-truth ingestion path. This
is portability-relevant in both directions — Kalshi settles through a different
mechanic, a perp never settles — and it is a *today* gap: `resolutions.jsonl`
has 1,292 rows and no module consumes them.

**5.2 — `phase` disappeared with `WindowCtx`.** Killing `WindowCtx` was right,
but nothing replaced its `phase ∈ {pre, in, post_T, resolved}`. `DE-Constraints`
cannot express "no quoting post-`T`" against a state view that has no phase.
Put `phase` on the `StateView` (it is derivable from `SP-Instrument.horizon` +
`now`), not back into a context struct.

**5.3 — two action vocabularies** (`DE-ActionSpace` vs
`BE-FlowAndFills.rest(level, size, …)`). §1 test 5. Introduced by correctly
making ActionSpace venue-derived and not following through to its consumer.

**5.4 — R-SSOT's enforcement point covers only `SP-Params`.** "the only legal
source; readers take a handle, cannot restate" names one of four records.
`SP-Venue`/`SP-Instrument`/`SP-Strategy` have **no restatement guard** — which
is precisely the five-way settlement duplication v2 was rewritten to eliminate.
The guard exists; it points at the wrong scope. One-line fix, highest
value-per-character in this review.

**5.5 — R-VERSION is event-time only** (§2(d)), in a programme that has already
observed a stale spec source. R-KNOW not applied to the layer v2 added.

**5.6 — `EV-Replay` blocks the only dataset we have** (§3(b)). No `--span-spec`
escape.

---

## Triage

### MUST-FIX (5)

| id | fix | why now |
|---|---|---|
| **M2-C1** | Extend **R-SSOT's enforcement clause to all four SP records**, not just `SP-Params`: no module may restate a venue/instrument fact; readers take a handle. | The one guarantee the entire rewrite rests on. Without it the 5-way settlement duplication is prevented by convention only — the same convention v1 had. One line. |
| **M2-C2** | **Spec identity by content hash + `observed_at`/`observed_source`**, alongside (not replacing) `valid_from/valid_to`. Datasets and `EV-Replay` stamp `spec_id`. | Answers the loop's question directly: dates are insufficient. Gamma served a stale rewards band after the 08-20 re-cut and both values are on disk *right now*. R-KNOW applied to SP. ~10 lines. |
| **M2-C3** | **`nulls{module: {assumption, bias}}`**; `DE-Solver` refuses an objective term or gate deriving from an OPTIMISTIC/UNKNOWN null — reusing R-PROV's existing refusal. | null `BE-FlowAndFills` ⇒ ζ=0 ⇒ *more* aggressive quoting, and R-PROV cannot see it (ζ=0 is not a param). Two rules, one seam, FATAL-2 shape. One field, no new machinery. |
| **M2-C4** | **`SP-Params` gains `scope` (global\|venue\|instrument\|market) and `valid_for` (validity domain).** Reuse outside domain fails loud. | `scope`: rewards band and observed fee are per-market and currently collide on `name`. `valid_for`: test 1 silently inherits μ̂=0 (a bound computed at τ=300 s, 48× more material at 4 h) and the 5-min σ fit. Two fields. |
| **M2-C5** | **`SP-Venue.fee(instrument, side, price, size)`** — add `size`, take the instrument. | The only hard interface change in five tests; the *same* omission v2 called out and fixed in `BE-FlowAndFills.rest()`; bites at min-size per-level EV today, not just on Kalshi. |

### SHOULD-FIX (8)

| id | fix |
|---|---|
| S2-C1 | Restore **resolution/redemption accounting** as an owned module (universal accounting; venue transforms already live in `DE-ActionSpace`). 1,292 resolutions on disk with no consumer. |
| S2-C2 | **Close the capability namespace**: absent ⇒ false; unknown name at wiring ⇒ error. Add `requires:` to the ~4 modules that need one, or R-REQUIRES has no data to enforce. |
| S2-C3 | `BE-FlowAndFills.rest(action: Action, state)`; declare the action type an **open** sum extended per venue. Removes the second vocabulary; simpler, not more general. |
| S2-C4 | `EV-Replay` `--span-spec` escape recording the boundary and both `spec_id`s. A spec boundary sits inside the only dataset we have. |
| S2-C5 | `phase` on `StateView` (derived), not a new context struct. |
| S2-C6 | Field set: `settlement` optional + `horizon: T\|PERPETUAL`; `SP-Instrument.discovery{slug_pattern, token_side_map}`; `SP-Venue` cancel cost, min resting time, redemption delay; `SP-Params.ci`. |
| S2-C7 | Split `strike_rule` (declared) from `b_hat`/`δ_tie` (estimated), exactly as `w_declared`/`w_hat` was split. Same bug class, second instance. |
| S2-C8 | Factor-keyed exposure inside `DE-Constraints` (was M-C4). Now an impl change behind `feasible()`, so it can wait for a measurement that demands it. |

### NOTED

- **Plane boundaries, dependency direction, the Belief/Decision split, `SelfState`
  in `DA-State`: no change proposed.** Survived five tests for a second
  iteration. Stop reviewing them.
- Do **not** build: a `VenueAdapter`; multi-outcome/range payoffs; a second
  `DA-Feeds` venue impl; the Binance host; a comparator DSL; an AMM liquidity
  model. Seams recorded, implementations not. (iter-1 stop-signs 1–8 all still
  hold, and v2 §9 already carries them.)
- `OP-LatencyBudget` owning the four legs rather than `SP-Venue` is a better cut
  than my iter-1 proposal — latency is our path, not the venue's.
- The one place I was wrong in iter-1: I scored the per-coin risk key as a MUST
  interface change. `DE-Constraints.feasible() -> {max_size, shadow_price}`
  makes the representation private, so v2 solved it without adopting my fix.
