# Phase 2B — fair-price challenger protocol

**STATUS: DRAFT-FOR-USER-FREEZE. NOTHING IS SCORED UNDER THIS DOCUMENT.**
TODO §10 gates *all* 2B scoring on this protocol being recorded and frozen, so
until the user freezes it no challenger may be evaluated — not exploratorily,
not "just to look".
**Owner:** DA. **Authorised as a draft by** the user's plan `d506a06`
(§10 item 4 / TODO §5.2), whose operative constraint is *nothing frozen or
scored*. **Baseline artifact:** `da_fair_price_identity.py` (this batch).

## 1. What is being asked

Whether any predeclared challenger beats **`Identity`** — the executable-book
price — as an estimate of `E[Y | state]`. Not whether it beats chance.

## 2. The closed set: at most TWO challengers

1. **PM microprice** (size-weighted book price).
2. **At most one cross-venue forecast.** **NAMED CANDIDATE (added before the
   review so the reviewer sees a real candidate, not a placeholder):
   the Binance USDM `bookTicker` mid for the matching symbol** — `BTCUSDT`,
   `ETHUSDT`. Verified buildable against the tape rather than assumed:
   `data/mm_hf/raw/bookTicker/<SYM>/YYYYMMDD_HH.csv.gz`, 213 hourly files per
   symbol from 2026-08-19, columns `recv_ns, E, T, updateId, bid, bidQty, ask,
   askQty`. **Both timestamps are producible**: `source_timestamp = E`
   (exchange event time) and `local_knowledge_timestamp = recv_ns`, which
   `collector_runs.jsonl` records as stamped
   `IMMEDIATELY_AFTER_WS_RECV_BEFORE_JSON_PARSE`. Measured freshness on a
   sample row: **0.0717 s**.

   **Two constraints that must travel with it, both binding:**

   - **ERA FLOOR.** The local-knowledge stamp is only trustworthy from the
     `hf_ws_v2` boundary, `recv_ns >= 1787579334881534478`
     (**2026-08-24T13:48:54Z**). Before it, rows were stamped post-parse and
     carry up to ~0.6 s of backlog error concentrated in bursts. Under §3 that
     makes pre-boundary instants **INADMISSIBLE for this challenger** — not
     merely noisier. Its admissible population is therefore strictly smaller
     than `Identity`'s, which §7.4 already tests.
   - **IT IS CLOSER TO THE SETTLEMENT SOURCE THAN `Identity` IS, AND THAT
     CHANGES WHAT A WIN MEANS.** PM binaries settle on a Binance-derived
     price. This challenger reads that same venue. That is **not** look-ahead —
     the current Binance mid is not the future settlement price, and the
     as-of rule still binds — but it does mean a positive increment is **not
     evidence of forecasting skill**. It would say *the PM book lags the
     settlement venue*, which is a latency/microstructure fact about the two
     venues, not alpha. **Declared now, before any number exists, so the
     interpretation cannot be chosen after seeing the sign.**

Declared **before** any comparison. **The set is closed at freeze**: adding a
third later is a new family and restarts multiplicity, because choosing what to
compare after seeing how the first comparison went is selection (rule 11).

## 3. Admissibility — inherited, not restated

A challenger produces the same typed record as `Identity` and is bound by the
same rules: **both timestamps or INADMISSIBLE** (never degraded, never a
default), strictly-as-of consumption, statuses instead of silent zeros.
**A challenger that cannot state when it could first have been known is not a
slower challenger — it is an unusable one.**

## 4. The score: PROPER, and INCREMENTAL TO IDENTITY

- **Proper score:** Brier on the settled binary outcome, `(p - y)²`. Proper, so
  the honest forecast maximises the expected score; a non-proper score rewards
  hedging and would make a shaded challenger look skilful.
- **Reported as skill vs `Identity`**, per day:
  `skill_d = 1 - BS(challenger)_d / BS(Identity)_d`.
- **THE INCREMENT IS THE ESTIMAND, AND THIS IS RULE 9.** PM binaries settle on
  a Binance-derived price, so a challenger reading that same price scores well
  against *chance* while adding nothing to what the book already says. **Skill
  against a base rate is meaningless here; only skill incremental to `Identity`
  is a result.**
- **Paired on identical decision instants.** Both estimators are scored on the
  same admissible records, matched by `(coin, window, outcome, decision time)`.
  A challenger scored on a different or larger population is not a comparison.

## 5. PIT parity — the challenger must not read further ahead

For every scored instant both estimators satisfy
`local_knowledge_timestamp <= decision_time`, and **the paired instants are
identical**. A challenger with a systematically later local-knowledge time is
not better, it is later — and would win by reading more of the future.
**Report the per-estimator freshness distribution beside the score**, so a skill
difference explained by a latency difference is visible rather than inferred.

## 6. Cluster unit and reporting

- **Unit: the UTC day** (rule 8). Below **G = 5 complete days**: point estimate,
  no interval, and say so.
- Per-day skill, `n` admissible paired instants, and the **status tally on both
  sides** (rule 4) — a challenger that wins by being admissible less often has
  not won.
- **Multiplicity recorded at freeze**: candidates in the race, budgets, and any
  earlier look. Two challengers × the declared budgets is the family; the joint
  reading is Holm across it, and a single uncorrected cell is not a result.

## 7. Declared before the data: what would make a challenger PASS

Stated now so it cannot be chosen afterwards. A challenger is adopted only if
**all** hold:
1. Positive `Identity`-incremental skill, **Holm-corrected across the family**.
2. **≥5 complete UTC days**, all after the protocol freeze; consumed days stay
   consumed.
3. Freshness distribution **not systematically later** than `Identity`'s (§5).
4. Admissible-instant count **not materially below** `Identity`'s (§6).

**A failed challenger does not block anything.** The full policy runs with
`Identity`. That asymmetry is deliberate — it removes any incentive to keep
looking until one passes.

## 8. Falsifiers required before any scoring run (rule 15)

1. A challenger missing `local_knowledge_timestamp` is **REFUSED**, with a
   positive control that a complete record is accepted.
2. Scoring on **unpaired** populations **REFUSES** — a synthetic case where the
   challenger has extra instants must not silently score.
3. `Identity` **versus itself** yields skill exactly **0** — the null the whole
   comparison rests on; if it is nonzero the pairing or the score is wrong.
4. A challenger that is `Identity` **plus a constant lag** shows **negative or
   zero** skill, never positive: a pure-latency "improvement" must not read as
   one.
5. An **empty** or all-inadmissible day reports **NOT EVALUABLE**, never a
   passing skill of 0.

## 9. What this document does not do

No scoring, no fitting, no promotion, no forward clock. Adoption of a passing
challenger remains a policy decision with its own priced trade-offs (rule 14):
this protocol estimates, it never decides.

---

## AMENDMENT A1 — settlement estimand and the price→probability transformation

**Status: DRAFT-FOR-USER-FREEZE, superseding the framing above in-band (rule 13;
the superseded text is left standing as provenance).** Written after the Codex
FP2 finding forced the settlement rule to be READ rather than assumed.

### A1.1 The settlement rule, read at the artifact

`data/pm_5min/markets.jsonl`, **17,727/17,727 markets**, verbatim:

> *"This market will resolve to 'Up' if the **time-weighted average price
> (TWAP)** of Bitcoin, **generated by Chainlink**, of the time range specified
> in the title is **greater than or equal to the price at the beginning of that
> range**."*

**Chainlink is the only resolution source named. Binance appears in ZERO
records.**

### A1.2 THREE CORRECTIONS THIS FORCES, all to DA's own earlier text

1. **"PM binaries settle on a Binance-derived price" is FALSE** and is withdrawn
   wherever this protocol relied on it, including as the stated rule-9
   justification.
2. **The "closest to the settlement source" reading of a bookTicker win is
   VOID.** Binance is not the settlement venue, so the challenger is a genuine
   cross-venue forecast and a positive increment would be a **cross-venue
   lead — the STRONGER claim**, not the weaker one.
3. **The estimand is path-dependent**, not a terminal-price comparison.

**Rule 9 still binds, through a different door:** `Identity` — the PM book —
already prices this event, so skill must remain incremental to `Identity`. The
conclusion is unchanged; only its reason was wrong.

**Deliberately OUT of scope:** whether Binance flows into Chainlink's aggregate.
That is a mechanism question about Chainlink's feed, not about the settlement
rule, and it does not enter the estimand.

### A1.3 The estimand, exactly

For a window `[t0, T]` with settlement reference `S_ref = P_chainlink(t0)`:

```
Y = 1  iff  TWAP_[t0,T](P_chainlink)  >=  S_ref          (ties resolve UP)
p    = P(Y = 1 | state at decision time t),   t0 <= t <= T
```

**At decision time the TWAP is PART-REALIZED**, so the transformation must
decompose it:

```
TWAP_[t0,T] = ( A_t + R_[t,T] ) / (T - t0)
  A_t   = realized integral over [t0, t]   -- KNOWN at t, but only from data
                                              with local_knowledge <= t
  R_[t,T] = residual integral over (t, T]  -- the only stochastic part
```

**A challenger reading only the CURRENT price cannot compute this.** It must
carry `A_t` as point-in-time state, itself built solely from admissible records.
That is a structural requirement on any challenger, not an implementation
detail.

### A1.4 The transformation — DECLARED, not derived

Pinned here so it is frozen before use; it is a **declared modelling choice**,
and alternatives are challengers of their own, not tweaks:

- **Model:** driftless GBM for the residual path; the average of a lognormal has
  no closed form, so a **moment-matched lognormal** on `R_[t,T]/(T-t)` is the
  declared approximation.
- **Reference `S_ref`:** the Chainlink value at `t0`, taken from the settlement
  feed. **If unavailable at decision time it is a status, not a substitution** —
  no proxy reference.
- **Tie convention:** `>=` resolves **Up**. Pinned by the venue, not chosen.
- **Horizon:** `τ = T - t`, seconds, from the window definition.
- **Volatility `σ`:** realized over a **declared 30-minute trailing lookback,
  shifted by one observation** so no input is contemporaneous with the decision
  instant, computed only from records with `local_knowledge <= t`.
- **Calibration:** none at freeze. Any calibration layer is a NEW challenger and
  restarts multiplicity.

### A1.5 Frozen terms — the loose language, pinned

| term | frozen value |
|---|---|
| "systematically later" | median freshness delta > **50 ms** vs `Identity`, over the paired set |
| "materially below" | admissible-instant count < **95%** of `Identity`'s on the common universe |
| budgets | the declared decision budgets, unchanged from Phase 2 |
| minimum paired n | **≥ 2,000** paired instants per coin-day, else the day is NOT EVALUABLE |
| statistic | **Brier skill vs `Identity`**, per UTC day |
| alpha | **0.05**, Holm-corrected across the family |
| weighting | **equal per paired instant**; no window or action reweighting |

### A1.6 The common eligible universe

Pairing alone equalises scored counts and therefore **cannot** measure
availability. Availability is measured against a **declared common eligible
universe**: every `(coin, window, outcome, decision instant)` for which the
SETTLEMENT reference exists and the window is admissible under the day-bar —
**independent of whether either estimator produced a record**. Each estimator's
admissible share is reported against that denominator.

### A1.7 Falsifiers this amendment adds

1. **TWAP vs TERMINAL must be DISTINGUISHED.** A synthetic path whose **terminal
   price is UP but whose TWAP is DOWN** must classify **Down**. A terminal-price
   transformation prices the wrong event and would pass every test built only on
   monotone paths.
2. **Tie:** a path whose TWAP equals `S_ref` exactly classifies **Up**.
3. **Part-realized state:** two decisions at different `t` within one window,
   given identical residual assumptions, must differ whenever `A_t` differs —
   proving the realized path is actually consumed.
4. **Constant-lag, on a CONTROLLED SYNTHETIC FIXTURE** (not arbitrary realised
   paths, where it is not universally true): on a fixture built so the lagged
   series carries strictly less information, `Identity`-plus-constant-lag scores
   **≤ 0** skill.
5. **`Identity` vs itself scores exactly 0** — the null the comparison rests on.
6. **Missing `S_ref`** yields a **status**, never a substituted reference.

---

## AMENDMENT A2 — the estimand is a 60-SECOND ENDPOINT comparison, not the full-window mean

**This is the second settlement correction in one day, and both were mine.** A1
corrected the *venue* (Chainlink, not Binance). A2 corrects the *statistic*. Both
pre-freeze, nothing scored on either.

### A2.1 Read at the artifact the reviewer named

`live/pm_research/EXP_RESULTS_2026-08-20.md:10-17`, EXP-M6, n = 1,465 windows:

| convention | agree | agree (\|margin\| > 0.5 bp) |
|---|---|---|
| **S60(T) vs S60(t0)** | **99.8%** | **99.9%** |
| S30(T) vs S30(t0) | 96.1% | 96.9% |
| S60(T) vs S30(t0) | 96.5% | 97.4% |
| **meanS60[t0,T] vs S60(t0)** | **86.9%** | 88.2% |

Against a **pre-registered** gate of ≥99.0% pooled and ≥99.5% on the >0.5 bp
subset. **Passed.** The artifact's own words: *"the averaging window is w = 60 s,
not the full 300 s range — the full-range reading scores 86.9% and is refuted."*

### A2.2 What that refutes is A1, not the reviewer

**My A1.3 estimand — `P(mean over [t0,T] ≥ reference)` — is the 86.9% row.** It
was reasoned from the market description's prose (*"the time-weighted average
price … of the time range specified in the title"*), which reads as the full
range. I corrected the venue by opening a market definition and then **took the
same prose as authoritative for the statistic without checking it against a
settlement reconstruction that already existed in this repo, and had already
passed a pre-registered gate.**

### A2.3 The corrected target

**Up iff `S60(T) ≥ S60(t0)`**, where `S60(x)` is the 60-second Chainlink TWAP
ending at `x`. Ties resolve **UP** — unchanged, still the venue's `>=`.

`S60(t0)` is **fully realized before the window opens.** The reference requires
no forecast at all.

### A2.4 Part-realization is TERMINAL and much lighter

Under the full-window reading, every instant contributed to the target and a
challenger had to carry the realized integral across the whole window. Under the
endpoint convention:

- **`t ≤ T − 60`** — none of `S60(T)` is realized. `A_t` contributes **nothing**;
  the target is a pure forecast of a 60-second average that has not begun.
- **`t > T − 60`** — `[T−60, t]` is fixed and only `(t, T]` is stochastic.

**So A1.3's structural claim was too strong.** "A challenger reading only the
current price cannot compute the target" is **true only inside the final 60
seconds**, not across the window. And **PM `Identity` / `pm_microprice` price the
binary event directly**, so no path integral is forced on them at all — that
requirement was an artifact of my full-window reading.

### A2.5 The reader, pinned

- **Feed**: `wss://ws-live-data.polymarket.com`, topic **`crypto_prices_twap_sixty`**
  — the Chainlink RTDS TWAP relay, 1 s cadence, values **1e18-scaled**, carrying
  `window_s` (`collect_pm_prices.py:34-50`).
- **`crypto_prices` on the same subscription is a Binance-spot mirror and is NOT
  the settlement source.** The Q-DA-117 error one level down: the wrong stream
  sits three lines from the right one, in the same subscribe block.
- **No replay exists.** Unsubscribed time is truth lost, so coverage over
  `[T−60, T]` and `[t0−60, t0]` is an **admissibility precondition**, not a
  quality note — a status, never an interpolation.

### A2.6 The tension is STATED, not resolved

**The description and the reconstruction disagree, and I cannot explain why.**
The market text says the TWAP "of the time range specified in the title"; the
settlements behave like 60-second endpoints. I adopt the reconstruction because
it is **measured against 1,465 actual settlements and passed a pre-registered
gate**, while the description is prose I have already misread once today. But
adopting is not explaining, and at least three readings survive:

1. the description is loose and always was;
2. the venue's convention changed after 2026-08-20;
3. the reconstruction's population is unrepresentative of current windows.

**(2) and (3) are testable and (1) is not.** A fresh independent audit on
**recent** windows, same grid, same pre-registered gate, would separate them —
and that audit is a legitimate build rather than a precondition for this
amendment.

**This amendment holds UNLESS a new independent same-population audit supersedes
it.** Named escape hatch, in band, so a later audit does not have to argue.

### A2.7 Knowledge time, not event time

The same artifact: reading the grid at knowledge time gives **99.3%** against
**99.8%** at event time, and *"that 0.5 pp gap is the size of the look-ahead a
careless backtest would bank."* Every S60 input is read at `local_knowledge ≤ t`,
which the accumulator already enforces and counts.

### A2.8 What survives the correction

**The machinery transfers unchanged — only the window bounds move.** Verified:
`realized_integral(ticks, t0−60, t0)` gives the reference and
`realized_integral(ticks, T−60, min(t,T))` gives the partial target, with
`covered_s` reporting exactly how much of the terminal 60 s is observed and
`mean()` still refusing on incomplete coverage. The tie rule, the era floor, the
exclusion classes and the point-in-time discipline are untouched.

**`bn_bookticker_probability` as built prices the FULL-WINDOW event and is
therefore superseded by this amendment** — marked in place, not deleted, and its
first scored use was already blocked on the user's freeze.
