# FLOW_MODEL_SPEC_REV2 — the respecification the measurement forces

> **⚠ For current state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).** This
> document is **provenance** — correct about its own moment, not a statement of
> current belief. Where it conflicts with `FLOW_MODEL_STATE.md`, that page wins.


> **SUPERSEDED 2026-08-21.** This file is retained only as the audit trail for
> the first descriptive fit. The authoritative specification is Revision 4 in
> `plans/BE_FLOWANDFILLS_MODEL_PLAN.md`. In particular, Revision 4 retains the
> Revision 3 state corrections and adds the development/validation lifecycle,
> execution-reach marks and observable front/back queue-fill bounds. Revision 3 corrected four
> errors here: ex-micro estimation does not require independence; side is a
> conditional mark rather than a covariate of total intensity; notional
> throughput is derived from count intensity and the native-price monetary mark;
> and the original `f_p` estimate is withdrawn because numerator and denominator
> used different price states. Do not implement from this document.

Revision spec for `plans/BE_FLOWANDFILLS_MODEL_PLAN.md` §2. **Not the plan
itself** — folding this in is a separate step, and the plan is being written by
another agent.

Evidence base: `FLOW_INTENSITY_RESULTS.md`, `flow_intensity.py`,
`data/pm_5min/derived/flow_{fr,fp}.json`. Scope throughout: `clob_v3_1` covered
set, **945 windows, 135/coin, under two days, one era.**

Scope of this revision: **specification only.** Maker edge, adverse selection and
profitability are out of scope here.

---

## 0. Why §2 needs revising rather than annotating

Three structural assumptions failed, not three numbers:

| §2 assumed | measured |
|---|---|
| `f_r` a dominant clock, rising into settlement | count **flat then collapses**; notional **builds, peaks `r≈52–112`, then falls** |
| R-DUAL a uniform reporting convention | micro share **2.0 % (btc) → 90.0 % (hype)**; four coins have **no admissible RAW count layer** |
| one specification across seven coins | the full covariate set is fittable on **three**; the other four run the same structure at coarser resolution on the ex-micro process |

Each changes what the estimand *is*, so annotation is not enough.

---

## 0.1 THE ASSUMPTION EVERY EX-MICRO QUANTITY RESTS ON

**Stated first because it gates everything below, and because nothing in the
previous draft stated it at all.**

Every ex-micro quantity in this specification — the Tier B grid, both `f_r`
shapes, `f_p`, `β_side`, any future branching ratio — is obtained by **deleting**
the single-actor 0.02 class from the arrival stream. That deletion is legitimate
**only under an assumption**:

> **A1 (SUPERPOSITION).** The micro process and the market process are
> **independent** point processes whose superposition is the observed stream.

Under A1 the decomposition is exact: the sum of independent point processes
decomposes, so removing one component leaves the other's intensity **and its
branching structure** intact. Without A1 the arithmetic still runs and produces
numbers that mean something else.

**If micro arrivals are triggered by market events — or trigger them — then
ex-micro is a CONDITIONED sub-process, not a component.** Its `f_r` is then the
intensity of market flow *given that a correlated stream was removed*, which is
not market flow, and every downstream quantity inherits the error silently.

**The exposure is not uniform and is largest exactly where the spec leans on it:**

| coin | share of events deleted under A1 |
|---|---:|
| btc | 2.0 % |
| eth | 22.4 % |
| sol | 29.7 % |
| xrp | 59.9 % |
| doge | 68.9 % |
| bnb | 78.2 % |
| **hype** | **90.0 %** |

**On hype, A1 carries 90 % of the events. It is the single largest unstated
assumption in the specification, and it must be tested before any ex-micro
quantity is fitted, not after.**

### A1's test — pre-registered here, run before any ex-micro fit

1. **Cross-correlation at short lags, both directions.** Micro arrivals against
   ex-micro arrivals. Independence predicts no structure at any lag; a lead in
   either direction refutes A1. Run per coin — the exposure differs 45× across
   the universe, so a pooled test would again largely report btc.
2. **Conditional inter-arrival law.** Compare the micro inter-arrival
   distribution inside **busy** versus **quiet** ex-micro intervals against its
   own unconditional law. Under A1 they coincide; dependence on the market
   process's state refutes it.
3. **Verdict discipline.** If independence fails, **ex-micro is a conditioned
   sub-process and must be labelled one.** Do not report its `f_r` as market
   flow. Stop and respecify, rather than carrying a relabelled quantity forward.

**Underpowered defaults to failing A1**, not to assuming it — on hype the cost
of wrongly assuming independence is a 90 %-deletion artefact presented as a
market measurement.

---

## 1. `f_r` — respecified

### 1.1 The estimand: NOT one shape with two scales

The count and notional profiles are **not** rescalings of one another. If they
were, `λ_notional / λ_count` would be flat in `r`; measured, mean USDC per
arrival runs **11.4 → 24.0** (btc) and **6.4 → 32.3** (eth) from open to close.
The ratio *is* a function of `r`, and a strongly increasing one.

So the coherent object is the one §2.1 already declares — a **marked** point
process — with the mark law made explicitly `r`-dependent, which the plan treated
as static:

```
λ_count(r; coin)          ground intensity          ESTIMATED
S(r; coin) = E[size·p | r]  mark value law          ESTIMATED
λ_notional(r) = λ_count(r) · S(r)                   DERIVED, never fitted separately
```

**Recommendation: fit two objects, derive the third.** Fitting `λ_notional`
independently would double-count and permit `λ_count · S ≠ λ_notional`, which is
unfalsifiable-by-construction. Deriving it makes the identity an **internal
consistency check** the fit must pass.

**The decomposition runs on the EX-MICRO count wherever that count is
admissible — which is all seven coins, subject to A1 (§0.1) and to power.** What
Tier B bars is the **raw** count (§3.1), not the ex-micro one. Where ex-micro
counts are too thin to support the decomposition at usable resolution, report
`λ_notional` directly and say that the decomposition was **not powered**, rather
than that it does not exist.

### 1.2 Two regimes, with a structural boundary at `r = 60 s`

Do **not** fit one functional form across the whole window. Fit:

- **BODY**, `r ∈ [60, 300]` — the estimation region. Both weightings are
  well-behaved here; btc CI half-widths are 8–17 %.
- **TERMINAL**, `r ∈ [0, 60)` — a **separate descriptive profile**, not a
  continuation of the body's form.

Three reasons, all measured:

1. The mark law changes fastest there (btc 15.5 → 24.0 USDC/arrival across the
   last four bins alone).
2. Both weightings fall, but by different factors — btc count to **0.184×** of
   peak against notional **0.284×**; eth 0.177× against **0.482×**. A single
   form cannot express a fall whose magnitude depends on the weighting.
3. The confound is concentrated there (§6.2).

**Cost of the split:** no smooth extrapolation across `r = 60`, and the boundary
is chosen on the TWAP length, which is itself one of the confounded candidates.
State it as a modelling choice, not a discovered breakpoint.

**Cost of not splitting:** the terminal collapse propagates into the body's shape
estimate, and the body is the part that is actually identified.

### 1.3 Resolution, and what 135 windows/coin supports

Ex-micro arrivals, and per-bin counts on a uniform 20-bin (15 s) grid:

| coin | ex-micro arrivals | per bin @20 | verdict |
|---|---:|---:|---|
| btc | 264,987 | 13,249 | 15 s comfortable |
| eth | 43,666 | 2,183 | 15 s workable |
| sol | 17,551 | 878 | 30 s |
| xrp | 9,663 | 483 | 60 s body (§3: raw count barred) |
| doge | 4,439 | 222 | 60 s body |
| bnb | 3,684 | 184 | 60 s body |
| hype | 1,622 | 81 | 60 s body; thinnest, ≈12/window |

**Recommended grids:**

- **btc** — 15 s uniform, 20 bins.
- **eth, sol** — 30 s uniform in the body, 15 s in the terminal region.
- **xrp, doge, bnb, hype** — **non-uniform**: 60 s bins across the body
  (`r = 300→60`, 4 bins), 15 s bins in the terminal (`r = 60→0`, 4 bins).

The non-uniform grid is not a compromise; it is doing two jobs at once. Body
bins of exactly 60 s **absorb the oscillation by construction** (§2), and the
terminal bins retain the resolution where the structure sits. It also puts
~1,000+ ex-micro arrivals in each body bin for the thin coins instead of ~200.

---

## 2. Binning under the 60 s confound — a specification decision

**This is the choice most likely to be made by accident.** A 15 s grid inherits
an oscillation of unidentifiable origin directly into `f_r`. Bins 0, 4, 8, 12, 16
are local maxima on **every** coin, and on btc the CIs separate peaks from
adjacent troughs (`bin 4: 8.20 [7.65, 8.78]` vs `bin 3: 6.41 [5.90, 6.92]`).

| option | effect | cost |
|---|---|---|
| **(a)** 15 s bins, oscillation left in | `f_r` carries 60 s wiggles | any consumer reading `f_r` inherits a 60 s cycle **nobody can defend the source of** |
| **(b)** 60 s bins on the grid | absorbs it exactly — one full cycle per bin | destroys terminal resolution: the fall happens inside a single bin |
| **(c)** 60 s window on a 15 s slide | filters it, keeps resolution | overlapping bins ⇒ correlated CIs, reduced effective df, and it **filters rather than removes** |
| **(d)** 15 s bins **+ explicit 4-level phase factor** | separates the oscillation into a **named nuisance component** | 4 extra parameters/coin; assumes phase enters additively in log-intensity and is constant in `r` |

**Recommendation: (d) where `n` supports it — btc, and eth in the body — and (b)
via the non-uniform body grid of §1.3 for the thin coins.**

(d) is the honest treatment: the *source* is unidentifiable, but the *component*
is estimable, and reporting `f_r` net of a phase factor labelled
`unidentified_60s_component` is interpretable in a way that (a) is not. The
factor must never be named for a mechanism.

Its additivity assumption is **testable** as a `phase × r` interaction, and that
test should be pre-registered before the fit. On btc there is `n` to run it; on
the alts there is not, which is the second reason they take (b).

**What (d) does not buy:** it does not identify the source, and it does not make
the terminal region interpretable — the phase factor and the terminal fall are
still confounded there (§6.2).

---

## 3. R-DUAL moves from reporting into the model

Above **~35 % micro share** the count layer is inadmissible as market evidence.
That threshold splits the universe:

```
TIER A  btc (2.0 %) · eth (22.4 %) · sol (29.7 %)
        raw and ex-micro count both usable; ex-micro is primary
        full marked point process at 15-30 s resolution

TIER B  xrp (59.9 %) · doge (68.9 %) · bnb (78.2 %) · hype (90.0 %)
        RAW count INADMISSIBLE -- it is a participant measurement
        ex-micro count EXISTS but is THIN; notional is the robust default
        same model structure, coarser resolution, power-limited
```

### 3.1 What Tier B loses — and what it does not

**Correction to an earlier draft of this spec, which said Tier B has no count
layer at all and therefore no Hawkes layer at all. That was wrong, and §1.3
refutes it** — the non-uniform Tier B grid lifts body bins to ~1,000+ **ex-micro**
arrivals, so an ex-micro count layer plainly exists.

The precise statement:

- **Inadmissible on Tier B: the RAW count.** At 60–90 % micro share it is a
  measurement of one participant with a market residual, not market flow.
- **Available on Tier B: the EX-MICRO count — thin, but real.** Per window:
  xrp ≈ 72, doge ≈ 33, bnb ≈ 27, **hype ≈ 12** ex-micro arrivals.
- Therefore the decomposition, the compensator, the time-change diagnostic and in
  principle a branching ratio **do exist** on Tier B — computed on the ex-micro
  process, subject to A1 (§0.1) and to power.

**Tier B is not a different model. It is the same model at lower power with the
raw count barred.** Whether any given layer is fittable there is a **power
question**, resolved by accumulating days — which is exactly what a thin-but-valid
process needs. It is not a structural impossibility, and it must not be written
as one: a false impossibility closes a door that is only stiff.

Two genuine structural limits remain on Tier B, and they are narrower:

- `β_side` on the **raw** count is meaningless there (the deleted class is
  99.98 % one-sided). On the ex-micro count it is estimable in principle,
  gated on `n` per side per band (§5.1).
- Notional stays the **robust default and the primary reported series**, because
  it is the weighting that survives contamination without invoking A1 at all.

### 3.2 Is a single pooled specification still coherent? No.

Pooling requires a common ground intensity. Four coins have none. Any pooled
statement about arrival intensity is a **btc + eth + sol** statement and must be
labelled as one — with btc at 64 % of the pooled arrival denominator, an unlabelled
pool would again largely report btc.

**Recommendation: the plan's §2.2 single equation is retained for Tier A only**,
and a second, coarser Tier B equation is stated explicitly:

```
TIER A   log λ_count(t|x) = f_r(r;coin) + f_p(p) + β_side + γ_tick + f_book(...)
                            + φ(phase)          [§2 option (d)]
         log S(r;coin)     = mark value law, r-dependent
         λ_notional        = λ_count · S        [consistency check, not a fit]

TIER B   log λ_notional(t|x) = f_r^N(r;coin) + f_p^N(p)
         no side term, no tick term, no book term, no phase term, no Hawkes
```

Tier B is deliberately impoverished. Writing it down as impoverished is better
than fitting Tier A's equation on Tier B's data and discovering the poverty in
the residuals.

---

## 4. `f_p` — respecified

### 4.1 The estimand carries its own non-identification

Written into the estimand definition, not a footnote:

> `f_p` is the arrival-intensity profile across unified price. **It cannot speak
> to the fee.** The crossing cost `0.07·p(1−p)` is a deterministic function of
> `p`, so any fee effect on intensity is collinear with `f_p` **by
> construction**, and no fit of `f_p` — of any shape, at any `n` — constitutes
> evidence about the fee in either direction.

This matters because the measured shape is *hump-shaped, peaking mid-book*
(btc 2.24 at `[0,.05)`, 12.72 at `[.65,.85)`, 3.27 at `[.95,1]`), which is where
the fee is maximal. A reader will reach for "cost does not suppress arrivals".
The estimand text must block that inference at the point of definition.

### 4.2 Minimum dwell is part of the specification

**Rule: a `p`-bin is reported only if its dwell exceeds 60 s in the estimation
sample. Fenced bins are reported as `FENCED` with their dwell, never dropped
silently.**

This is a spec rule because the failure it prevents is not visible in the output:
`hype [.85,.95)` had **9 s** of dwell and produced a rate that alone drove a
`shape_ratio` of 295. `xrp [.95,1]` had **0 s**. A ratio off a 9-second
denominator is noise wearing a number, and nothing in the result announces it.

### 4.3 `f_p` must be conditioned on `r`, not marginalised over it

**Not in the current plan, and it should be.** Price disperses as the window
progresses, so `p` and `r` are dependent by construction: the `p`-distribution
near `r = 0` is far more concentrated at the extremes than near `r = 300`.
Fitting `f_p` marginally over `r` therefore conflates the moneyness profile with
the clock.

**Recommendation: fit `f_p` separately in the body and terminal regimes** at
minimum, matching §1.2's boundary. A full joint `f(r, p)` surface is preferable
but is btc-only at this `n`.

### 4.4 Sampling and bins

`f_p` needs the quote stream for dwell — ~97 % of message volume — so it cannot
run on the full covered set at the cost the count profile runs at. The spec must
state a **deterministic** window subsample per coin and report it as scope.
7 bins for btc; **5 bins for Tier B**, collapsing the extreme tails that the
dwell fence removes anyway.

---

## 5. The remaining layers, in order

Each with its precondition and its honest sample requirement.

### 5.1 `β_side`

- **Precondition:** complement fold verified; micro class excluded. The class is
  **99.98 % one-sided**, making `β_side` the single most exposed term in the
  model.
- **Requires:** ≥200 ex-micro arrivals per side per `(coin, r-band)`.
- **Available:** btc, eth. **sol marginal. Tier B: not at all** (§3.1).

### 5.2 `γ_tick`

- **Precondition:** tick read from **both** `book.tick_size` and
  `tick_size_change.new_tick_size` — matching only the former produced a false
  "all 0.01" reading.
- **Specification:** **tail interaction only, never a main effect.** The 0.001
  regime occurs in `p<0.15` (6.75 %) and `p≥0.85` (6.73 %) and is **absent from
  the middle three buckets entirely**, so a main effect across the book would
  simply re-express `f_p`.
- **Requires:** enough arrivals inside the 0.001 regime *within tail buckets* —
  a small fraction of an already-thin region.
- **Available: btc only**, and to be confirmed before fitting rather than
  assumed.

### 5.3 `f_book`

- **Precondition 1:** notional-weighted. Count-based imbalance inherits a
  **state-dependent** ~−16 pp shift — a bias that moves with state is fitted as
  structure.
- **Precondition 2:** read **strictly pre-arrival at a frozen lag** (Δ = 250 ms,
  with a sensitivity ladder). `price_change` carries **post**-change quotes and
  can be emitted *before* the `last_trade_price` for the same match, so a naive
  read conditions on the consequence of the arrival being predicted.
- **Precondition 3:** that non-leakage must be **demonstrated**, not asserted —
  a placebo at negative lag should show no predictive content.
- **Requires:** the quote stream, so the same cost class as `f_p`.
- **Available:** btc; eth/sol coarsely. **Tier B: no.**

### 5.4 Time-change diagnostic

- **Precondition:** a fitted `λ_count` compensator including **all** of the above.
  It is the last diagnostic, not an early one.
- **Test:** `Λ(t_i)` unit-rate Poisson under a correct model; residual clustering
  tested **there**, never on raw inter-arrival times.
- **Available: Tier A only** — Tier B has no compensator.

### 5.5 Hawkes / self-excitation

- **Precondition 1:** A1 (§0.1) must hold. Branching structure is the property
  superposition preserves **only** under independence; if the micro process is
  coupled to the market process, a branching ratio measured on the ex-micro
  stream is an artefact of the deletion.
- **Precondition 2:** the time-change diagnostic must **first show residual
  clustering**. If it does not, this layer is not fitted at all.
- **The honest answer on sample size: more days than we have — and that is a
  schedule, not a refusal.** The usable set is **one era spanning under a day**,
  inside two collected days. A branching ratio estimated there cannot be
  separated from one day's regime, and the recorded failure mode is precisely
  that a self-exciting term with a constant baseline adopts on in-sample fit.
- **Requires:** day-level replication — order **10+ independent days within a
  single collector era**.
- **Applies to Tier B as a power question, not an impossibility.** On the
  ex-micro process hype runs ≈12 arrivals/window (≈1,600 in the current era); at
  10 days that is order 16,000, which is a workable if modest sample. More days
  is exactly the remedy a thin-but-valid process needs.
- **Recommendation: do not fit this layer yet.** Specify it, state both
  preconditions, and revisit when the day count exists — Tier A first, Tier B
  when its ex-micro counts support it.

---

## 6. Promoted from caveat into the specification

These are not limitations of the current fit. They are properties of the venue
and the data, and the spec must carry them so they are not re-derived.

### 6.1 The 60 s oscillation source is unidentifiable

All 952 window starts are `≡ 0 (mod 60)` **and** `(mod 300)`, so window phase and
wall-clock minute phase are **perfectly collinear**. A minute-boundary effect in
the underlying reproduces the pattern exactly.

**Separated only by:** windows on a non-60 s-aligned grid; the same profile
against an underlying with no minute artefact; or a same-length venue on a
different phase grid. None exists here.

### 6.2 The terminal mechanism is confounded with the artefact

Settlement is `S60(T)` vs `S60(t0)` — a **60 s trailing mean** — so through the
final minute the settlement value progressively locks in. That is a clean
mechanism for the terminal fall, it begins exactly where the fall begins, and it
is **collinear with the 60 s artefact**: TWAP length 60 s, oscillation 60 s,
window starts on the minute.

**The most attractive explanation for the headline shape is not identifiable.**
Separated only by a venue with a **different settlement-window length on the same
5-minute grid** — which moves the TWAP timescale while leaving the minute grid
fixed. The §6.1 separators do **not** cover this one.

### 6.3 Nothing day-clustered is estimable

Two collected days, one usable era spanning under a day. Window-clustered
intervals are available and **understate** uncertainty, because they cannot
capture day-level common factors. Any estimand requiring a day-clustered interval
returns `Unavailable`, structurally — not pending more analysis.

### 6.4 Four of seven coins admit no RAW count-layer model

§3.1. A venue-composition fact. **It is not the same as "no count layer"** — the
ex-micro count exists on Tier B and is thin (hype ≈ 12 arrivals/window), so what
it bars is a raw-count claim, and what it leaves is a power question that
accumulating days addresses.

### 6.5 A1 is untested, and it is the largest open assumption

§0.1. Every ex-micro quantity in this spec is conditional on the micro process
being independent of market flow. On hype that assumption carries **90 % of the
events**. Untested as of this revision, with its test pre-registered in §0.1.

---

## 7. Summary of decisions taken

| # | decision | recommended |
|---|---|---|
| 1 | `f_r` estimand | fit `λ_count` + mark law; **derive** `λ_notional` as a consistency check |
| 2 | regime split | body `r ∈ [60,300]`, terminal `r ∈ [0,60)`, separate forms |
| 3 | binning | btc 15 s + **phase factor**; Tier B 60 s body / 15 s terminal |
| 4 | oscillation | absorb as a **named nuisance component**, never attributed |
| 5 | tiering | same structure both tiers on the **ex-micro** process; Tier B bars the **raw** count, runs coarser, notional primary. Power-limited, not impossible |
| 6 | pooling | **incoherent**; pooled statements are btc+eth+sol and labelled so |
| 7 | `f_p` | non-identification in the estimand; 60 s dwell fence in-spec; conditioned on `r` |
| 8 | Hawkes | **specified but not fitted** — needs A1 to hold, residual clustering, and ~10+ days in one era. A schedule, not a refusal |
| 9 | **A1 superposition** | **untested precondition on every ex-micro quantity**; test pre-registered in §0.1, underpowered defaults to FAILING it |

Open where genuinely open: **A1 (§0.1) is untested and gates everything
ex-micro**; the `phase × r` interaction test (§2) is pre-registerable but unrun;
and the `r = 60` boundary is a modelling choice resting on the TWAP length, which
is itself confounded. All three are stated as choices or open assumptions rather
than findings.

**Correction recorded in this revision:** an earlier draft claimed Tier B admits
no count layer and therefore no Hawkes layer, and that more days would not help.
That was a **false impossibility** — the ex-micro count exists on Tier B, it is
thin rather than absent, and more days is precisely its remedy. Corrected in
§3.1, §5.5 and §6.4.
