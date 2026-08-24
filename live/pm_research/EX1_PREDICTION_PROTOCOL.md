# E-X1 — does our forecast beat the book? · protocol, frozen BEFORE the run

**Status: FROZEN per Ruling R-51, 2026-08-23. APPEND-ONLY from this point
(R-28); R-38 clause (d) applies — an amendment buys an obligation to re-measure,
never a verdict.** No measurement has been run against it. Written by BE under
**R-46**, re-framed under **R-47** as the programme-identity question. Correctly
carries **no R-49 re-pointing**: a prediction-quality score never has to beat a
latency.

*Freeze applied late by BE — see the note in `BINANCE_LEAD_PROTOCOL.md`. A
protocol's status is what its file says.*

**§1's precondition is CLEARED (2026-08-23):** `route_a_v1` carries the corrected
anchor. `SIGMA_ROUTE_A_PROTOCOL.md:74` enforces `E[x_T] = S60 + α·(S30−S60)` with
**α FITTED** per symbol/horizon, no intercept, no pooling — a form containing
both prior errors as special cases (α=0 is v1's trailing TWAP; α=2 is v2's
assumed local trend). Measured across all 42 fits: **median α = 1.49**, range
−0.16 to 4.16, **1 of 42** near α=0 and 8 near α=2. The data chose neither failed
assumption. Nothing needs building under `PRICING HOLD`.

**Revised under R-47.** One protocol, not two: the hedgeability decomposition is
**CANCELLED** — refuted on mechanism, recorded in §8 rather than deleted, because
the refutation is worth more than the experiment would have been.

**E-X1's standing has changed and the protocol must not quietly carry the old
framing.** It is no longer the gate for cancellation work. It is the
**PROGRAMME-IDENTITY** question — alpha versus pure market-making — deferred
since session 1 and still unanswered on the corrected anchor.

---

## 0. Why this is live rather than a formality — verified, not recalled

The programme has assumed pure market-making since session 1 because an earlier
comparison said the book wins at every horizon. **That comparison is flagged in
the programme's own risk register**, `SIGMA_PLAN` §10 risk 10, verbatim:

> *"Demoting σ-for-level could be wrong. It rests on a book-beat comparison run
> against a **mis-anchored model** on **one test day** *and* on an **MNAR
> sample**. H-3 is the named re-entry point."*

And `SIGMA_PLAN` §1 item 3 states the magnitude:

> *"A large share of the model's deficit to the book **was never a σ problem. It
> was a forecast-anchor error.**"*

The arithmetic makes it decisive rather than hopeful. The anchor error was
`E_t[x_T]` set to a trailing 60 s TWAP, which **lags spot by `w/2 ≈ 30 s`**,
while the variance factor was spot-anchored (§2.1). Correcting it gained
**−0.0101 Brier pooled, at every horizon**. The book's final published margin was
**+0.0201 Brier**. **One correction closes half the gap**, and it was never
re-scored after the correction.

**Three defects are named in that risk line and this protocol must clear all
three, not one.** The task names the anchor; the other two are equally
disqualifying and are addressed in §4 (days) and §5 (MNAR).

---

## 1. The question, stated so it cannot drift

**Does our forecast `p̂` beat the venue book `m` as a predictor of the binary's
resolution?**

This is a **PREDICTION** question, not a deployment question, so `PRICING HOLD`
does not block it — nothing is priced, sized or quoted. The boundary is explicit:

- **Permitted:** scoring the output of the already-frozen fit against the book.
- **FORBIDDEN:** refitting, retuning, cell/tolerance changes, or any inspection
  of `route_a_v2` on primary days. `route_a_v1` is rerun **UNCHANGED**.

**BE must verify before running, not assume: which artifact carries the
CORRECTED anchor.** §2.1 records the correction; it does not follow that the
scored artifact contains it. If no artifact carries a corrected-anchor forecast,
this protocol **cannot run** and that is the finding — it does not license
building one under `PRICING HOLD`.

---

## 2. What the answer decides — programme identity, not cancellation

**R-47 moved this.** E-X1 was briefly the gate for predictive cancellation. It is
not, because the user's inversion — use **Binance flow to predict INCOMING PM
FLOW** and cancel before it arrives — needs **no price advantage at all**, only
to be **earlier on flow**. That route sidesteps the fair-price question entirely
and **DE owns it**.

So E-X1 reverts to what its own `STATUS.yml` note always said it was:

> *"Decides the programme's identity (alpha vs pure market-making)."*

- **If `p̂` beats `m`:** this programme is **not pure market-making**. Everything
  built since session 1 has assumed it is. A second, independent cancellation
  channel also opens beside DE's — but that is a consequence, not the reason to
  run.
- **If `p̂` does not beat `m`:** pure market-making is the correct identity, held
  on evidence rather than on an assumption that outlived its support.

**It is worth knowing regardless of the cancellation route.** That is the whole
justification now, and it is sufficient.

### 2.1 DO NOT LET THE RETRACTED ANSWER SET THE PRIOR

The earlier "the book wins at every horizon" verdict was **WITHDRAWN as
mis-anchored**. So this is **not a re-test of a finding**; it is an open question
whose previous answer was retracted. That distinction has three operational
consequences, and expectation-management is the weakest of them:

1. **The bar is symmetric and frozen** (§6): `BEATS_BOOK` and `BOOK_WINS` are
   named with identical structure, and `ev_gates.assert_directional` confirms the
   rule answers differently on an input and its mirror. Not sign-blind, and
   checkable rather than asserted.
2. **`INSUFFICIENT_POWER` may not be reported as `BOOK_WINS`** (§6.1). A retracted
   answer biases most easily by making a weak null feel like confirmation.
3. **The retracted answer's SAMPLE CONSTRUCTION is not inherited either.** This
   is the one that matters and it is easy to miss: the prior comparison's
   admissibility rule produced the 19 % MNAR exclusion that is *part of why it
   was retracted*. Re-using it would import the defect while believing the anchor
   fix had cleared it. §5 audits the exclusion instead of inheriting it.

**The honest scope limit still stands** — beating the book on pooled Brier does
not by itself establish a fill-conditional, seconds-scale, cost-clearing signal.
It establishes identity, which is what is being asked.

## 3. Estimand and scoring — frozen

```
unit          one (window, decision time) row; ONE Bernoulli outcome per window
outcome       y = 1 iff the Up token resolved to 1   (source: resolutions.jsonl
              `winners`, NOT `outcomePrices` -- the latter is the Gamma polling
              field and yields zero outcomes)
competitor    m  = knowledge-time Up-book mid from price_change.best_bid/ask
                   NEVER `book` snapshots (p90 6.2 s stale)
challenger    p_hat = the corrected-anchor forecast, read at the same knowledge
                   time as m, on the SAME row
primary       paired Brier delta   d = Brier(p_hat) - Brier(m)     NEGATIVE = ours wins
secondary     paired log-loss delta, same pairing
inference     day-clustered block bootstrap, 2000 draws, seed 20260823
              WHERE CLUSTERS PERMIT -- see section 4
stratify      time-in-window r, on the frozen grid {270, 240, 180, 120, 60}
              reported per stratum AND pooled; pooled is primary
```

**Pairing is mandatory and is the point.** Both forecasts are scored on the
identical row; a row admissible for one and not the other is **excluded from
both** and counted. Unpaired comparison is what makes two different populations
look like a skill difference.

**Pseudo-replication.** The 5 decision times in a window share one Bernoulli
draw, so every naive `n` is inflated ~5× and every naive `t` is ~45 % too large
(measured). Cluster on **window**, then on **day**; day is primary.

---

## 4. The day problem — the second defect in risk 10

The prior comparison ran on **ONE test day**. A day-clustered interval needs more
than one day block.

```
minimum to report a day-clustered CI      >= 3 scored days
below that                                 report WITHIN-day CI, labelled, and the
                                           verdict is INSUFFICIENT_EVIDENCE by
                                           construction -- never a point estimate
                                           dressed as a comparison
```

Sampling is **DAY-STRATIFIED** (R-19 D-V5-3, R-35): equal allocation per era day,
deterministic under the declared seed, and `sampling_rule` +
`provenance(sampled=...)` + `n_days_sampled` stamped in the receipt. **Never
pooled** with any earliest-first receipt — `flow_intensity.assert_poolable()`
enforces this and raises on a mixture.

---

## 5. The MNAR problem — the third defect, and the one most likely to be skipped

`route_a_v1` excluded **374 windows on `s30_window_coverage`, 19 % of the
sample**, and that selection has never been audited. The prices lane logs a
**11–13 s gap roughly every 20 minutes**, and the gaps are **load-correlated**,
so exclusion is plausibly *not* random and plausibly removes the busy windows —
exactly where a forecast edge would or would not live.

**Mandatory, and the run is VOID without it:**

1. Report the excluded fraction **and its composition** by `r`-stratum and coin.
2. Score **both arms on the retained set** — the pairing already forces this.
3. **Both-arms reporting**: if the excluded set can be scored at all (a coarser
   admissibility), report the delta on it separately. If it cannot, say so and
   state the direction the exclusion would bias the delta.
4. A comparison on a 19 %-excluded MNAR-suspect sample **is not upgraded** by a
   larger `n`. More days do not fix selection.

---

## 6. The bar — FROZEN before the receipt exists

Let `d` be the pooled paired ΔBrier (ours − book) and `[lo, hi]` its
day-clustered 95 % interval.

```
BEATS_BOOK             hi < 0            our forecast wins, interval excluding zero
BOOK_WINS              lo > 0            the book wins, interval excluding zero
UNDETERMINED           lo <= 0 <= hi
VOID                   section 5 not satisfied, or < 3 scored days, or the
                       corrected-anchor artifact does not exist
```

**Both directions are named and symmetric**, so the bar has no thumb on it. Under
`ev_gates.assert_directional` this rule answers **differently** on an input and
its mirror — it is not sign-blind, and that is checkable rather than asserted.

### 6.1 What it takes to FIRE — the a-priori power declaration

R-14 Amendment 2: a gate must declare not only a failing witness but **what it
takes to fire**. Declared **before** the run:

- **MDE at the available day count** is computed and published *before* scoring,
  from the retained row count and the observed per-day ΔBrier dispersion.
- If the MDE exceeds the effect the anchor correction implies — **0.0101 Brier**,
  the measured gain from fixing the anchor — the honest output is
  **`INSUFFICIENT_POWER`**, reported as such and **not** as `BOOK_WINS`.

That distinction is the whole point: the prior answer's problem was not that the
book won, it was that a mis-anchored model on one MNAR day was allowed to settle
the programme's identity.

### 6.2 Vacuity control — the run is VOID if this fails

A probe that cannot detect a known effect proves nothing.

- **Injected-edge control.** Score a synthetic forecast constructed to beat the
  book by a known margin on the same rows. The protocol must return
  `BEATS_BOOK`. If it does not, the harness cannot detect an edge and the real
  result is uninterpretable.
- **Null control.** Score the book against **itself** with the same pairing and
  bootstrap. Must return `UNDETERMINED` with `d ≈ 0`. A non-zero delta means the
  pairing or the bootstrap is broken.

---

## 7. What this protocol will NOT do

- Not refit, retune or inspect `route_a_v2`; `route_a_v1` runs **unchanged**.
- Not produce a probability-level output; `PRICING HOLD` is untouched.
- Not select `r`, coin or horizon after seeing results. The grid is frozen above.
- Not pool across collector eras or sampling rules.
- Not read a day count off `tier1/` — days come from the raw tape, derived.

---

## 8. The hedgeability decomposition — CANCELLED, and why it is recorded

**Refuted on mechanism by the user, not by a measurement.** Kept here rather than
deleted, because the refutation generalises and the deleted version would leave
the next person to re-propose it.

**The refutation: if the Binance perp LEADS the PM binary, a hedge is LATE BY
CONSTRUCTION.** You learn you needed to hedge only *after* the move you needed to
hedge against, so the perp leg transacts post-move and **locks the loss in**
rather than offsetting it.

**This is sharper than the counter BE had drafted, and BE should say so.** BE's
§8.1 asked *what FRACTION of the drift is hedgeable* — decomposing into
contemporaneous spot movement, TWAP-revision, and pure order flow. That framing
concedes the wrong thing: **even a 100 % hedgeable fraction does not help if the
hedge can only transact after the move.** Fraction is a magnitude question;
lead-lag is a feasibility question, and feasibility comes first. BE was
measuring how much of a door was open without checking that it opens outward.

**The general form, worth carrying:** a decomposition answers *how much*, and is
only worth running once *whether* is settled. Asking the magnitude question first
is a way of appearing rigorous while skipping the one that can refute the idea
outright.

**Consequence for the record:** the Layer-1 negative stands undiminished. Post-fill
drift of **−1.175 ¢** (btc) and **−2.021 ¢** (eth) at `h=5 s` is not offset by a
hedge that arrives after it.

## 8a. ANNOTATION — E-X1 CANNOT RUN ON CURRENT ARTIFACTS (append-only, R-28)

**Added 2026-08-23 beside the frozen text, never as an edit to it.** Found by the
`BE_BELIEF_REVIEW_LOOP` iteration-1 adversarial lens; **verified by BE
independently** before recording. Escalated as `Q-BE-13`.

**Three independent blockers, each sufficient on its own:**

**1. THE PAIRED POPULATION IS ZERO ROWS.** §3 requires both forecasts *"read at
the same knowledge time … on the SAME row"*. Measured:

```
route_a_v1 OOS window_start span : 2026-08-20 00:00:00 -> 2026-08-20 13:50:00
clob_v3_1 era opens              : 2026-08-20 14:50:21
OOS rows at or after era open    : 0 of 5,796
```

Every scoreable `route_a_v1` row ends **~1 hour before** the book series the
protocol mandates begins. The two arms live in **disjoint eras**, and §7 forbids
pooling across collector eras. Neither escape is open under the freeze: re-running
route_a on era days is §1's *"FORBIDDEN: refitting"*, and a pre-era book series
has no gap ledger (`flow_intensity.py:22`).

**2. ONE OOS DAY, AGAINST A VOID THRESHOLD OF THREE.** `oos_rows` carry
`distinct days: ['2026-08-20']`; all 42 fits record `n_oos_days: 1`. §6:
`VOID … or < 3 scored days`.

**3. `route_a_v1` CONTAINS NO PROBABILITY.** Its row keys are
`alpha_fold, coin, day, decision_ms, horizon, m_bps, pred_y_bps, residual_bps,
s30, s60, winner_up, x_bps, y_bps, …` — **no `p̂`**. The challenger E-X1 scores
does not exist as an artifact. Constructing one needs σ and a link, which §11 of
`BE_BELIEF_PLAN` declares and never pins, and §1 forbids building under
`PRICING HOLD`.

### 8a.1 BE's own error, recorded plainly

BE's header states *"§1's precondition is CLEARED"* and BE reported to the
coordinator that *"E-X1 does not need anything built under `PRICING HOLD`"*.

**That was verified on ONE dimension and reported as a cleared precondition.**
BE checked that the corrected anchor **exists** — true, and it holds:
`E[x_T] = S60 + α(S30−S60)` with α fitted, median 1.49 across 42 fits. BE did
**not** check that a scoreable paired population exists, that the day count
clears §4, or that a probability exists to score.

**This is the third instance of the same defect class BE has now logged**:
reporting `applied` / `cleared` on partial verification (R-24's prose-only
application; the `FIRE_SIDE` echo; this). The pattern is that BE verifies the
dimension it was thinking about and reports the whole gate. §1's instruction was
*"verify before running, not assume"* — BE verified, and still assumed the rest.

**The honest pre-run finding is the one §1 already anticipates**, and it is not
the one BE expected: the corrected-anchor artifact exists, and *the protocol still
cannot run*, because the anchor was never the only precondition.

## 9. Deliverables

1. `EX1_RESULTS.md` + a receipt carrying `sampling_rule`, `days_sampled`,
   `n_days_sampled`, the exclusion composition, the MDE, and both control
   outcomes.
2. The scoring code with a selftest, including the two §6.2 controls **and** an
   `assert_directional` check on the §6 bar.
3. A statement of which artifact carried the corrected anchor, or the finding
   that none does.
4. **No hedgeability deliverable** — cancelled under R-47, §8.

### 8a.2 A PREMISE THIS PROTOCOL INHERITS HAS BEEN REFUTED (BE, 2026-08-23, R-28 annotation — not an edit)

`BE_BELIEF_PLAN` §1.2 supplied this protocol's framing: *"a belief that tracks
the book cannot profit from disagreeing with it, **so the recalibration IS the
edge or there is none**."* That is what made E-X1's question strict — beat the
book as a predictor, or there is no alpha.

**The second clause is withdrawn.** `FLOW_MODEL_STATE.md` §1e measures a
two-sided `JOIN_BBO` maker — whose belief *is* the book — capturing **+0.642
c/share on btc (n=10,294)** and **+0.778 c on eth**. Profit with zero
disagreement. The tautology holds for *takers*; a **maker** is paid the spread
for supplying immediacy and needs no disagreement to earn it. The same §1e then
measures that maker's markout at **−0.532 c btc / −1.243 c eth** — so the real
question is **spread capture versus adverse selection**, which belongs to
BE-FlowAndFills, not to the belief module.

**What this changes for a successor, and what it does not.**

- **It does NOT change this protocol's verdict.** E-X1 is `VOID
  (NO_PAIRED_POPULATION)` under R-56 on three independent grounds — 0 of 5,796
  overlapping rows, `n_oos_days: 1` against a threshold of 3, and `route_a_v1`
  carrying no probability at all. Under R-38(d) an amendment buys an obligation
  to re-measure, never a verdict, and this annotation asserts none.
- **It DOES change what a successor should ask.** A protocol built on §1.2 asks
  *"does our forecast beat the book?"* and reads a NO as *"there is no edge."*
  That inference is invalid: a maker can be paid without beating the book, and
  can lose while beating it. **A successor to E-X1 should be scoped to spread
  capture net of adverse selection, not to predictive superiority alone** — and
  if it keeps the predictive question, it must not carry §1.2's "or there is
  none" with it.

**Recorded here because this is the propagation R-58 warned about**: nothing live
depends on E-X1, but a successor inherits its framing silently, and a refuted
premise travels further than a refuted result.
