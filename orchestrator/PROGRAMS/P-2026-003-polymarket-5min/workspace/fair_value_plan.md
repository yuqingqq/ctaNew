# Fair value plan — v1.1

**Status: REVISED DRAFT for the user's ruling. Nothing in it has been run.**
Written 2026-09-11 by the coordinator and revised after review on the same day,
while the descriptive valuations of 09-09..09-13 run. It does not alter the
forward test, whose verdict is fixed (R-910/R-911/R-914: both arms futile at
day two).

---

## 1. Why this exists: the estimand was blind to fair value

Every candidate this programme has raced — the 69 screened, the two that
survived to the forward test — lives in the **cancellation layer**. The quoting
layer was held fixed: fair value = the book's own best bid/ask, plus a skew.

That is not an accident of the runs, it is built into the measurement. The
statistic is

    D = (arm's settled total) − (vanilla no-cancel book's settled total)

and **both legs quote off the same fair value**. Any error in that fair value
is common to the two legs and cancels out of D. The design could therefore
never see fair-value quality, only whether cancelling beat not cancelling. It
answered that question at the declared power: neither cancellation arm was
established, and the result does not answer whether a better quoting anchor
exists.

The code carries an identity field, not a working integration. The trajectory
schema has `fairprice_estimator`, but today's exportable compositions still pass
`None` and explicitly refuse `CONDVALUE_X_SKEW_X_FAIRPRICE`. Building the
producer, policy consumption and replay seam is part of this plan; the unused
field is not evidence that the feature is already wired.

**One correction to the framing that prompted this plan.** "Vanilla performs
best" is true only *among what was tested*. The no-cancel book lost 4,438c on
09-07 and made 86,227c on 09-08. It beat both arms on both days; nothing
establishes that quoting mid-plus-skew is profitable.

## 2. What is already known about the settled event

This matters more than anything else here: a binary cannot be priced until its
settlement rule is pinned down.

- **Venue: Chainlink, not Binance.** R-253, verified at the artifact: of 17,727
  market records, 17,727 mention Chainlink and 0 mention Binance.
- **Statistic: `S60(T) vs S60(t0)`, ties UP** — the 60-second TWAP at expiry
  against the 60-second TWAP at the window's open (amendment A2, Q-DA-142).
  The full-window mean reading (`meanS60[t0,T]`) is **refuted** at 86.9%.
- **Reproduction rate**: 99.8% on the original population against a
  pre-registered gate of ≥99.0%, and **99.85% pooled / 99.9% on |margin|>0.5bp**
  on a pre-registered fresh split (2026-08-24..27, n=8,022 windows, full
  coverage, disjoint from the original) — Q-DA-146. The equivalence control
  that licensed this was found decorative and was made executable (Q-DA-148);
  the numbers survived and strengthened.

**Consequence that shapes the whole design: `S60(t0)` is fully realized before
the window opens.** The strike is known at decision time. Before `T−60`, none
of `S60(T)` is realized. Inside the final 60 seconds, its leading integral is
realized and only the remaining integral is stochastic.

**Important boundary:** those mechanics determine the future average required
for UP; they do **not** determine its probability. A probability requires an
explicit conditional distribution for the remaining path, including a
point-in-time volatility input. The existing structural estimator correctly
requires `sigma`, `sigma_as_of` and `sigma_lookback_s`; this plan does not call
that model-free.

**Two things to settle before leaning on the rule (step 0 below):** the residual
~0.15% of windows the convention does not reproduce has never been
characterised (ties? boundary reads? oracle round timing?), and `CLAUDE.md`
still carries the older line that the settlement statistic is "contested and
no form is asserted", which is now stale against R-253/Q-DA-146 and should be
reconciled or defended.

## 3. The required data exists, but the probability producer does not

- `data/pm_5min/prices/crypto_prices`, **`crypto_prices_twap_sixty`** and
  `crypto_prices_twap_thirty` — timestamped observations relevant to the
  settlement path.
- `markets.jsonl` — strikes, windows and rules text.
- `resolutions.jsonl` — realized winners.
- The PM CLOB tape — the book's own price at every admissible decision action.
- Binance fine features already in the feature set: `bnf_midbps` at
  10/25/50/100/250 ms and `bnf_imb_now`. Sub-second-reliable only from
  **2026-08-24 13:48:54 UTC**.

The typed `FairPrice` interface and the deterministic partial-TWAP accumulator
exist. The current `bn_bookticker_s60_probability` is **build-only**, requires a
caller-supplied sigma, and requires a Binance-derived partial rather than the
settlement feed's partial. It must not be relabelled as the challenger this plan
needs. Step 0 must produce and test the missing volatility source and a
source-correct fair-price producer.

## 4. Estimand and closed candidate family

### 4.1 Deterministic settlement state — no fitted probability

Let `K = S60(t0)`. At a decision time `t > T−60`, let `A_t` be the point-in-time
integral already observed over `[T−60,t]` and `r_t = T−t`. The remaining path
must satisfy

    future_average_required = (60*K − A_t) / r_t

for UP. Before `T−60`, the required future average is simply `K`; if
`60*K − A_t <= 0`, UP is already clinched. This state transformation is the
model-free part. It is useful as an input and as a falsifier, but it is not a
fair probability by itself.

### 4.2 Candidate F1 — structural settlement probability

`F1` estimates `P(S60(T) >= K | state_t)` under one predeclared remaining-path
distribution. The first implementation is the existing driftless-GBM Asian
moment-match form, rebuilt against the correct source identities. Its sigma
producer must declare, before any labelled score is read:

- source, bar interval and trailing lookback;
- annualisation/scaling convention;
- minimum observations and gap/staleness refusal;
- `sigma_as_of <= decision_recv_time` mechanically;
- one fixed fallback status — never a caller-chosen sigma.

No alternative lookbacks are raced after seeing outcomes. If more than one
sigma specification is scored, each is a separate candidate in the
multiplicity count.

### 4.3 Candidate F2 — structural probability plus oracle-lag residual

`F2` adds one predeclared signed state measuring the point-in-time disagreement
between the settlement feed and the admissible spot feed, scaled by volatility
and feed age. It is not assumed that Binance leads Chainlink; that sign and
scale must be estimated on development data and frozen. `F2` must reduce exactly
to `F1` when the residual coefficient is zero, and must abstain to `Identity`
when either feed is stale or unavailable.

### 4.4 Deferred family

A larger ML model is not part of this race. It may be proposed only if F1/F2
leave predeclared residual structure, under a new plan, new candidate count and
new untouched validation clock. Feature discovery after reading F1/F2 scores is
a new selection event, not an amendment to this family.

The fair-value race therefore contains **at most two candidates, F1 and F2**.
The 69 historical cancellation candidates remain disclosed as programme
provenance but are not members of this new estimand's family: none produced or
was selected on a fair-value score. Any broader programme-wide multiplicity
claim must report both families rather than silently mixing or discarding them.

## 5. Step 0 — reconcile the rule and build the missing producers

Deliverables, all build-only and unscored:

1. A declaration naming the settlement rule, its reproduction rate on a
   declared audit population, and the **characterisation of the residual** —
   for every mismatch, which of {tie, boundary read, oracle round timing, data
   gap, unexplained} accounts for it, with exclusions as statuses.
2. Reconciliation or defence of the stale `CLAUDE.md` settlement line.
3. A timestamped sigma producer with the contract in §4.2, plus synthetic
   scale, stale-input and future-knowledge falsifiers.
4. A source-correct `FairPrice` producer for F1/F2. It must output source-event
   time, local-knowledge time, freshness, estimator identity and a counted
   status. It may not pass a Chainlink partial into a function that declares a
   Binance partial, or vice versa.
5. An end-to-end seam proving that the quoting engine consumes the value rather
   than merely carrying `fairprice_estimator` as metadata. `Identity` with no
   challenger must still run unchanged.

Falsifier: recompute settlement for the audit population and match
`resolutions.jsonl` winner-for-winner. Any unexplained mismatch class over a
numeric threshold declared before the audit **stops the programme here**.

## 6. Step 1 — development feasibility, not evidence

### Population

Every day available before the candidate freeze is **development/consumed**.
The 08-24..27 settlement split is already consumed, and 09-03..09-09 have been
read extensively by adjacent P-003 work. They may be used to build, debug and
cross-fit F1/F2, but no score on them is validation and no interval on them is
promoted. The receipt names the exact dates, n and as-of time.

### Canonical action row

One row is one unique quote-generation decision on the neutral `Identity`
reference path, keyed by at least
`(coin, slug, side, generation_id, decision_recv_ns)`. Duplicate keys refuse the
build. Both Identity and every challenger are evaluated on exactly these paired
actions; arbitrary tape ticks and repeated copies of one action are not rows.

The challenger falls back to Identity on a declared abstention, so missing
challenger inputs cannot improve its score by shrinking its population. An
inadmissible Identity action remains a counted status and enters neither side.
Every table reports actions, unique markets, UTC days, fallbacks and every
exclusion status.

### Point-in-time and latency

Every input is read at local knowledge time (`recv_ns`). No prediction may use
an input whose local-knowledge timestamp exceeds `decision_recv_ns`. Economic
availability is evaluated at the frozen placement latency: a value that changes
after the decision can affect only a later generation, never the order already
sent.

### Development score

Use leave-one-UTC-day-out cross-fitting. Fit only on the other development days
and score the held-out day. Identity and candidate probabilities are clipped by
the same predeclared epsilon before scoring.

Primary loss is paired log loss. Brier score, calibration by time-to-expiry and
per-market equal-weight summaries are diagnostics and cannot promote a
candidate. The primary daily quantity is

    delta_g = mean_actions(logloss_Identity − logloss_candidate)

so positive is better. Action rows define the policy's decision distribution,
but uncertainty and all predicates are computed at the UTC-day unit; no row or
market is treated as an independent day.

Before any labelled score is read, the declaration binds:

- the candidate identities and total count;
- the action cadence/key and probability-clipping epsilon;
- sigma contract, estimator equations and fitting procedure;
- a numeric minimum `delta_logloss` for proceeding;
- minimum candidate coverage/fallback limits;
- the exact quote mapping and a numeric minimum economic improvement for the
  later replay.

If neither F1 nor F2 clears the development gate, stop. A development pass is
permission to freeze a candidate, not evidence that it works.

## 7. Step 2 — freeze the complete predictive and quoting pipeline

The first candidate is deliberately small and anchored to Identity:

    structural_residual = logit(clip(p_structural))
                          − logit(clip(p_Identity))
    logit(p_candidate) = logit(p_Identity)
                         + beta_1 * structural_residual
                         + beta_2 * oracle_lag_state

F1 fixes `beta_2 = 0`; F2 permits both coefficients. Equivalent bounded forms
are allowed only if chosen before development scores are read. Parameters are
fit on the declared development population, then frozen once.

The freeze is a commit and contains the full path:

    raw records -> PIT state -> sigma -> F1/F2 -> FairPrice
                -> quote mapping -> replay events -> settlement ledger

The receipt records code/file hashes, commit ref, fit-population identities,
parameter values, candidate count, null, latency, quote tick rounding,
admissibility/fallback behaviour and initial inventory. Models emit estimates;
the policy layer alone decides quotes.

The quote mapping is frozen now, before predictive validation: same skew,
sizing, inventory limits and cancellation lifecycle as Identity; only the
`FairPrice.value` anchor may differ. This prevents a predictive result from
being used to tune the later economic policy.

Required falsifiers include:

- Identity substituted for the challenger produces bit-identical quotes;
- a future-knowledge input is refused;
- a stale/missing challenger falls back exactly to Identity and is counted;
- F2 with `beta_2=0` is bit-identical to F1;
- a deliberately inverted settlement convention is detected;
- a synthetic informative predictor improves the scorer, while a known-bad
  constant does not receive a positive verdict.

## 8. Step 3 — prospective predictive validation

Validation starts on the first complete UTC day strictly after the full-pipeline
freeze. It uses at least **eight complete, untouched UTC days**. Eight days give
all `2^8 = 256` exact day-level sign assignments, satisfying the ≥200-null
requirement without Monte Carlo duplication.

The primary test uses the daily `delta_g` above and an exact two-sided paired
day-sign test. With `G=8`, the minimum attainable two-sided p-value is
`2 / 2^8 = 0.0078125`; with at most two candidates, Holm/Bonferroni remains
attainable (`0.015625 < 0.05`). If the candidate count changes, the required
number of days is recomputed before validation as

    (2 / 2^G) * candidate_count <= alpha

and the clock is extended, never the threshold relaxed. Intervals, if reported,
resample UTC days only. Every result carries G, actions, markets, status counts
and as-of time.

The success gate is the predeclared positive log-loss improvement, multiplicity
correction and minimum effect size. Brier and calibration plots cannot rescue a
failed primary. Predictive validation days become consumed whether the result
passes or fails.

## 9. Step 4 — economic validation on a new clock

Only a Step-3 winner proceeds, without refitting or remapping. Economic
validation begins on the next complete untouched UTC day and uses a new
predeclared day count and multiplicity calculation, again never fewer than eight
complete UTC days.

The comparator remains the `Identity` fair-value quoter, rerun on the same
external tape with the same starting inventory, skew, sizing, position limits,
cancel/repost lifecycle, quote rounding and latency. The candidate runs on its
own resulting order path because changing fair value changes placements and
fills. Old vanilla artifacts remain provenance; they are not reused as the
contemporaneous comparator.

The primary economic quantity is paired daily terminal settlement P&L:

    economic_delta_g = settled_PnL_candidate − settled_PnL_Identity

Its primary null is the same exhaustive day-level sign assignment used in
Step 3, now applied to `economic_delta_g`; no row-level or fill-level p-value is
reported. Candidate multiplicity and a numeric minimum P&L effect are bound in
the freeze before this clock starts.

Quotes and fills become effective only after the frozen measured latency `L`.
The ledger includes cash from buys/sells, remaining-inventory settlement,
queue/repost effects and the actual declared maker fee schedule. Zero fee may be
used only if verified for the relevant market/account and named in the receipt.

**The spread is not treated as a taker-style hurdle.** For a passive maker, the
relevant value is side-specific expected settlement value at the executable
quote, conditional on the fills the policy receives, net of fees, latency,
queue reset and inventory consequences. A fair-minus-mid difference smaller
than the full spread is not automatically unharvestable.

The economic test and minimum effect are frozen before Step 3. No predictive
validation number may be used to change the policy or this endpoint. Any change
starts a new candidate and a new clock.

## 10. Kill criteria

- Step 0's settlement residual is unexplained above its declared threshold →
  **stop**.
- No point-in-time sigma producer or source-correct partial can satisfy the
  interface → **stop**; mechanics alone are not a probability.
- Neither F1 nor F2 clears the development minimum → **stop**, recorded as a
  development failure rather than a validation result.
- A frozen candidate fails prospective incremental log loss versus Identity →
  **stop**. Calibration or Brier diagnostics cannot rescue it.
- A predictive winner fails terminal settlement P&L versus the contemporaneous
  Identity quoter → **stop**; the probability improvement is real but not
  harvestable by this maker policy.
- Any edge that requires reacting faster than measured latency `L` is valued
  only after `t + L`, or it is not valued at all.

## 11. What this plan does not do

It does not revisit the cancellation arms, alter their forward verdict, tune a
large feature model, or touch a frozen module inside the current population.
Development builders are read-only against existing data and may run alongside
descriptive valuations only when their I/O does not starve the active replay.
No fair-value score is evidence until the source-correct producer, full quote
seam, freeze and prospective day clocks above exist.
