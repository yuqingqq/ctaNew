# PM_SKETCH_REVIEW_ITER1_S — statistics/experiment-design lens, iteration 1

Reviewed: `PM_MM_PLAN.md` §5 ladder (PM-E0.5→E4), gates G2/G3a/G3b, §6 hazards.
Standard applied: `live/mm_research/EXPERIMENT_PLAN.md` (the parent prereg).
Data facts from `PM_REVIEW_ITER1.md` + fresh measurements on
`data/pm_5min/raw/20260819/` (136 window files, ~2 h, 7 coins — indicative
only, damaged windows included):

| coin | windows | trades/win p50 | notional/win p50 | zero-trade frac |
|---|---|---|---|---|
| btc | 25 | 2,055 | **$30,570** | 0.00 |
| eth | 25 | 466 | $3,383 | 0.16 |
| sol | 25 | 214 | $827 | 0.12 |
| xrp | 25 | 197 | $606 | 0.20 |
| doge/bnb/hype | 12 ea | 124–135 | $185–298 | 0.08 |

**BTC ≈ 85% of total per-window notional → effective coin breadth under
notional weighting ≈ 1.4 (1/Σw²).** Every "7 coins × 288/day" power intuition
must be shrunk accordingly. Zero-trade windows are 8–20% of the universe.

Verdict up front: the ladder ORDER and the gate INTENT are right, and the plan
inherits the correct standing discipline (§6 last bullet). But as written, G2
and G3a/b are not yet preregisterable: no gate statistic is fully defined
(paired vs unpaired, significance criterion, cluster unit consequences,
multiplicity), the E2 backfill era has no book data so the contest's competitor
forecast is undefined there, and the E2→E3 forking-path split is unstated. The
repo's three historical killers — weighting choice, clustering error,
difference-in-significance — are all currently possible under the plan text.

---

## S1. G2 — model-vs-book calibration contest

### S1.1 The gate is not defined as a paired test (MF-1)

Plan text: "model beats book in ≥1 time-stratum with day-clustered
significance." Nothing says the test is on the **paired per-observation loss
difference**. The failure mode this invites is exactly the repo's historical
difference-in-significance fallacy: "model's loss CI excludes the book's point
estimate" or "model significantly better than climatology, book not." Required
spec text:

> For each scored observation i (window × decision time), compute
> ΔL_i = L(book_i) − L(model_i) on identical (timestamp, window) pairs; an
> observation is scored iff BOTH forecasts are computable at that timestamp
> (paired-complete design). The gate statistic is the weighted mean of ΔL with
> day-clustered SE and stationary block bootstrap on days. No per-forecast CIs
> are ever compared; only ΔL is gated.

### S1.2 Missing spec: score, clock, grid, unit (MF-2)

1. **Primary scoring rule — pick ONE.** Both log-loss and Brier are strictly
   proper, so either is legitimate; listing both without a tie-break doubles
   the tests. Recommendation: **Brier primary** — bounded (a near-tie window
   resolving "wrong" under the still-unresolved boundary convention cannot
   produce an unbounded loss) and needs no clipping parameter. Log-loss
   secondary/diagnostic with preregistered clip p∈[1e−4, 1−1e−4] (the clip
   value must be frozen; it materially affects late-window losses).
2. **Clock and simultaneity.** Both forecasts must be as-of the same **local
   receive clock** t: book forecast = mid of the last book state with
   recv_ns ≤ t; model forecast = p̂ computed from Binance/TWAP data with
   recv_ns ≤ t. Never mix exchange timestamps for one side with recv for the
   other — Binance leads PM, so a mixed-clock contest hands the model a
   pseudo-future. Note the direction of the residual bias: the received book
   may be ~0.5 s stale vs PM's matching engine, which slightly favors the
   model; this is the deployable view (what a quoter actually sees) — state
   it, don't hide it.
3. **Decision grid.** Fix ex ante, e.g. t ∈ {30, 60, 90, …, 270} s into the
   window (9 points). All grid points within a window share the same outcome
   Y → **the inferential unit is the window, never the grid point** (9× N
   inflation otherwise). Day-clustering subsumes this, but the reported
   "n_eff" must count windows.
4. **Book forecast definition.** Mid of the Up book = (best_bid+best_ask)/2.
   Rules needed: one-sided/empty book at t → observation unscored for BOTH
   forecasts (paired-complete); tick-quantization noted (see S1.5); Up-book
   vs 1−Down-book mid discrepancy rule (use Up book; report the arb gap).
5. **Walk-forward discipline.** "OOS walk-forward by day" is stated but not
   the fit rule. Spec: all fitted components of p̂ (σ̂ estimator params, μ̂,
   Chainlink basis correction, any recalibration) use data through day d−1
   only (expanding), frozen before day d is scored. Labels resolve within
   the window, so no further purging is needed — say so.
6. **Label source.** Y = CLOB winner (`is_final` filtered). Feature gaps (our
   TWAP capture gapped, Binance feed gap) do NOT unscore a window — the model
   just does worse, which is deployable-realistic; only collector-damaged
   book files (no book forecast possible) unscore, symmetrically. This
   prevents quietly excluding exactly the hard windows for the model.

### S1.3 Stratification = garden of forking paths (MF-3)

"Beats book in ≥1 time-stratum" is explicit test-shopping: with K strata at
α=0.05 the family-wise false-positive rate is ~1−0.95^K, and it is not even
stated whether the test is per coin (×7 again). Required:

- **Primary test: pooled across coins** (weights: see S1.6), **one
  prespecified stratum structure**: time-in-window strata, count fixed at
  freeze (suggest 3: t<120 s / 120–240 s / >240 s). Gate = Holm-corrected
  across the 3 strata, OR designate pooled-all as the single primary and
  strata as confirmatory. Either is fine; pick one at freeze.
- **Per-coin results are secondary/descriptive**, never gate-bearing (BTC
  dominates the notional anyway — S1.6).
- **Strata must be model-free.** |d| is a function of the model's own
  forecast; stratifying the gate on it lets the model grade its own exam.
  Use time-in-window and |book_mid − 0.5| (market-defined moneyness). |d|
  strata allowed as diagnostics only.
- "Sign-stable across weeks" → quantify: ΔL̄ > 0 in ≥3 of 4 consecutive
  whole weeks (and the gate is unreadable before 4 weeks exist — S1.4).

### S1.4 Power (MF-8): day-clusters are the binding unit; 2 weeks is thin

Per-observation paired Brier difference: ΔB_i = (p_b−p̂)(p_b+p̂−2Y), so
σ_ΔB ≈ RMS(p_b−p̂) ≈ 0.02 (tick=1c, spreads 2–4c, typical model-book
disagreement 1–3c). Mean edge μ = RMSE²_book − RMSE²_model.

MDE at day-clustered t≥2: **MDE ≈ 2·√( σ²_ΔB/(G·n_eff) + σ²_day/G )** with G
days, n_eff windows/day, σ_day = day-level common component (regime-persistent
calibration error; unknown until data).

n_eff/day: 288 × effective-coins. Notional-weighted ⇒ effective coins ≈ 1.4
(measured) ⇒ n_eff ≈ 400/day. Scenarios (σ_ΔB = 0.02, σ_day = 0 lower bound):

| G days | window-level MDE (Brier) | means model must cut book RMSE 3c → | rel. log-loss equiv. |
|---|---|---|---|
| 14 | 5.5e−4 | ≤1.9c (−37%) | ~0.18% |
| 28 | 3.9e−4 | ≤2.3c (−25%) | ~0.13% |
| 56 | 2.8e−4 | ≤2.5c (−17%) | ~0.09% |

So at pure window-level noise, a 0.5–2% relative log-loss edge is detectable
inside 2 weeks — **but** the σ_day term dominates as soon as day-level regime
variance is comparable to the edge (needing G ≥ 4·(σ_day/μ)² days regardless
of window count: σ_day/μ = 2 → 16 days, = 3 → 36 days), and the week-sign
criterion needs 4 weeks to be non-vacuous, and 14 day-clusters give a t with
13 dof plus a fragile bootstrap. **Prereg fix: G2 is readable at ≥28 UTC days
of scored data (backfill + forward), and the freeze document publishes the
achieved MDE computed from variance components only (no means read) — a
variance-only peek is prereg-safe and should be declared as such.**

### S1.5 Beat-the-mid can be a quantization artifact (SF-8)

Late in the window the book mid is pinned to the 1c (or 0.001) grid while p̂
is continuous; near p→0/1 the model can "win" on pure tick-quantization with
zero economic content. Two cheap guards: (a) demote the late/extreme-moneyness
stratum from gate-bearing to diagnostic, or clamp both forecasts to
[tick/2, 1−tick/2]; (b) add a secondary baseline = **walk-forward
isotonic-recalibrated book mid** — if the model beats raw mid but not
recalibrated mid, the "edge" is public recalibration, not information, and the
G2 PASS text should say which one was beaten.

### S1.6 Weighting for the contest (SF-9)

"Notional-weighted, never eq-weighted" transfers, but the weight must be
**ex-ante** (e.g. trailing same-coin window-notional median), not the scored
window's own ex-post volume — same-window volume correlates with |move| and
hence with the outcome, a leakage channel through the weights. Also freeze the
7-coin universe and the weight vector at freeze time: coins must not enter
mid-sample as data becomes available (the expanding-survivor-universe artifact
is exactly what killed the OB timing signal in this repo). Consequence to
state honestly: notional weighting makes G2/G3 effectively BTC gates
(85% weight) — that is the correct deployable question, but per-coin
generalization claims are then unsupported.

### S1.7 Backfill era has no books (MF-4)

The plan runs PM-E2 on "backfill + early forward," but the backfill is Gamma
metadata + Data-API **trades** (+ prices-history availability unverified;
Gamma deletes resolved 5-min markets). There is no historical book stream, so
"book mid at decision time t" does not exist there — at best last-trade price
or a prices-history curve, which is a **different competitor forecast** (stale
where thin, no spread information). Required: define the competitor per era;
if backfill uses last-trade, the eras are separate strata and **the gate reads
forward-era (true books) only**, with the backfill contest reported as
supporting evidence. Never pool the two sources into one gate number.

---

## S2. G3a/G3b — MM replay economics

### S2.1 PnL definition and unit of account (MF-5)

"Net markout-to-resolution PnL per window, notional-weighted" is not yet
implementable. With many fills, four quoted sides, and pair-merges, the clean
formulation prices every fill at its resolution payoff:

> Per window w, in USDC:
> PnL_w = Σ_fills s_j·(1{token_j wins} − p_j)·size_j − fees_w
> where s_j = +1 for our buys (maker bid filled), −1 for our sells; token_j ∈
> {Up, Down}; a merged/redeemed pair is identical to holding both legs to
> resolution (Up+Down pays $1), so NO separate merge accounting is needed in
> the gate metric — merging affects capital usage only (report separately).
> Rewards_w is computed but NEVER added into PnL_w (separate line, G3b only).
> Day PnL = Σ_w within UTC day. **Gate statistic = mean daily USDC PnL,
> day-clustered t ≥ 2 AND stationary block-bootstrap (days, B=2000) 95% CI
> lower bound > 0.** Summing USDC within days IS the notional weighting —
> per-window bps averages (which would equal-weight dead $200 windows with
> $30k BTC windows) are forbidden as gate quantities.

Without the significance sentence, "net PnL > 0" reads as a point estimate —
that is a vibe, not a gate. Parent plan attaches t≥2 + CI to every gate; match
it. Also state fails-final/passes-provisional explicitly for G3 (the §6 bullet
implies it; the gate text should carry it).

### S2.2 Queue bracket: pro-rata is not Polymarket's mechanism (MF-6)

PM_REVIEW_ITER1 E.2: matching is **price-time priority** on a unified book.
So "moderate = pro-rata share" is not a mechanism model — it is an ad-hoc
interior point. The honest bracket, implementable from the level-total
`price_change` semantics (ITER1 #8):

- **Pessimistic (gate-bearing):** join back of queue at level L at join time
  t₀ with queue-ahead Q₀ = displayed size at L; queue-ahead is decremented by
  TRADES at L only (all cancellations assumed behind us); we fill only after
  cumulative traded size at L since t₀ exceeds Q₀. (= RiskAverse semantics.)
- **Optimistic (bracket bound):** front-of-queue — fill on first trade at L.
- **Pro-rata:** keep as an interior diagnostic, relabeled "ad-hoc
  interpolation, not a mechanism."

Critical relabeling: the bracket ends are defined by **queue mechanism, not by
presumed PnL direction**. More fills is not monotonically better for a maker —
the optimistic queue fills you on MORE toxic sweeps (fast fills are bad
fills), so PnL(optimistic) can be BELOW PnL(pessimistic). Gates read the
pessimistic-queue arm (as drafted — good); the no-sign-flip rule must compare
both ends and record a flip as ≤0 (parent E4 gate-2 wording), not assume
optimistic ≥ pessimistic. Also specify the four-sided/unified-book
de-duplication rule: an Up-bid at p and a Down-ask at 1−p are ONE economic
quote — the replay must maintain 2 queue positions per side-pair, not 4, or
one taker order will be double-filled.

### S2.3 Fees: the observed schedule is 0 — sensitivity rows required (SF-1)

19,141/19,141 observed trades have fee_rate_bps = 0, vs the plan §1 table
(taker ~$0.0175/share, 20% maker rebate) and the CLOB base_fee=1000 field.
Two consequences the plan doesn't yet draw:

1. The §1 economics table's "maker rebate ~20% of taker fees" is currently
   worth **$0** (20% of zero). The pitch's rebate line should be flagged
   accordingly until nonzero taker fees are observed.
2. G3a needs preregistered **fee-regime sensitivity rows**: PnL under
   (i) measured fees (currently 0), (ii) docs formula
   (C·0.07·p(1−p) taker, maker 0), (iii) adverse: base-fee activated on
   makers. A G3a pass that dies under (ii) is a fee-regime-fragile pass and
   must be labeled — H-PM4's fragility logic applied to fees, not just
   rewards. Extend H-PM4's text to cover fee-regime change.

### S2.4 Rewards line: estimator unspecified and competitive (SF-2)

"Rewards estimated per program rules" hides the hard part: Polymarket rewards
are **pro-rata against all makers' scores** (two-sided in-band depth,
spread-quality weighted, sampled per minute). Our share = ourScore /
(ourScore + othersScore), and othersScore is only estimable from recorded
in-band book depth (which includes competitor reaction we cannot observe).
Spec needed: per-minute scoring formula from the docs snapshot (versioned;
docs snapshot is still missing — build item 3), competitor score from recorded
band depth, our score from replay resting sizes, and a **haircut bracket**
(e.g. rewards × {1.0, 0.5}) on G3b. G3b's pass sentence must say "under the
rewards-estimate bracket," else it will be read as measured income. Per-market
daily rates come from the CLOB `rewards` object (captured post-ITER1 — verify
in iter-2).

### S2.5 G3 power asymmetry worth stating (N-3)

Rewards are a quasi-deterministic income stream (band occupancy × rate), so
G3b's PnL has small day-to-day variance and is well-powered at 14 days. G3a
(no rewards) is a classic heavy-tailed MM PnL (rare pickoff losses) — at
daily Sharpe 0.3–0.5 it needs G ≈ 16–44 days for t≥2. **Prereg consequence:
at the planned 2-week read, expect "G3b readable, G3a inconclusive." Either
extend the G3a read to ≥4–6 weeks of forward data, or predeclare the 2-week
G3a as provisional-only (cannot PASS, can only FAIL-on-sign).** Report tail
diagnostics (worst-day, worst-window contribution share) alongside.

### S2.6 Exclusions and selection (SF-10)

Replay can only run on undamaged windows. Freeze the blacklist (ITER1 D.1
damaged files + the 15:36–15:52 outage windows + TWAP-boundary-gap windows)
**before** any gate is computed, classify exclusion causes
(collector-exogenous vs load-correlated), and report the excluded fraction
per gate. Load-correlated exclusions bias PnL UP (outages cluster with
bursts = the toxic windows); state the direction in the results doc.

---

## S3. Ladder logic, forking paths, freeze process

### S3.1 E2→E3 contamination (MF-7)

PM-E2 runs on "backfill + early forward"; PM-E3 replays "≥2 weeks forward."
These windows overlap, and p̂'s configuration (basis correction, σ̂ choice,
δ_tox, τ_min, ρ_max) will have been examined against E2 results computed on
the same days E3 replays — garden of forking paths. Required spec:

> All model parameters used by the E3 replay on day d are fit on data through
> d−1 (walk-forward, same rule as E2), AND any discrete model-design choice
> made after seeing E2 results (feature sets, stratum-informed tweaks, pull
> thresholds) is frozen at a declared date; E3's gate-bearing replay period
> starts strictly after that date. E2 days may appear in E3 only under
> parameters that never saw them. Ablation arms obey the same rule.

### S3.2 Freeze process is a sentence, not a mechanism (MF-7)

"Gate numbers are FROZEN at the end of PM-E1" — by whom, where, covering
what? Parent standard: "PRE-REGISTERED …, before any E1 result is computed;
any post-hoc change must be logged as a protocol amendment with rationale."
Required: a named artifact (suggest `PM_PREREG.md` or a frozen §5-appendix in
the plan) recording, with date: gate thresholds AND significance criteria,
primary score, stratum definitions and count, cluster unit, weights + coin
universe, decision grid, exclusion blacklist, fee/rewards sensitivity grid,
minimum read dates (G2 ≥28 d, G3a per S2.5), and the amendment rule. The
orchestrator freezes it at PM-E1 end; reviewers check the E2/E3 harnesses
read constants from it, not from inline numbers.

### S3.3 Multiplicity budget across the ladder (MF-3, S2)

Beyond G2's strata: G3 has 2 gates × ≥5 ablations × bracket arms. Fine as
long as only G3a/G3b (pessimistic arm, primary PnL) are gate-bearing and
everything else is labeled diagnostic — one sentence in the plan fixes this.

### S3.4 Dangling "G1" reference (MF-9)

The G2 FAIL branch reads "still potentially viable (G1 economics only)" but
this plan defines no G1 (it lives only in PROGRAM.md's sketch). Fix: either
define G1 or rewrite the branch as "PM-E3 proceeds with p̂ := book mid
(quote-around-book + rewards harvesting); alpha-dependent ablations dropped;
only G3a/G3b remain."

---

## S4. Missing experiments — which are load-bearing

| candidate | verdict | why |
|---|---|---|
| (a) shadow-quote markout anatomy (naive join-touch quotes, pessimistic fill rule, markout-to-resolution by time-in-window × burst-flag; no inventory logic) | **ADD** (as PM-E2.5 or mandatory first arm of E3) | Isolates adverse selection before GLFT machinery obscures attribution; parent ladder has exactly this rung (E2 naive quoter) and PM plan skipped it. Cheap: same replay infra, trivial policy. If pessimistic-queue naive fills markout ≪ −spread everywhere, no quoting scheme survives → early kill. |
| (b) rewards-band occupancy feasibility (per minute: can 50-share two-sided rest within 4.5c of mid without being run over; band-compliance cost vs rewards $/min from bursts) | **ADD to PM-E1** | Direct G3b viability pre-check, measurable from book+trade data now, informs δ design. Without it G3b's rewards line assumes occupancy that bursts may make unaffordable. |
| (c) manipulation-successor screen (H-PM1b operationalized: per-window max basis z between TWAP stream and Binance-synthetic X̂ inside [T−60,T]; flagged fraction; outcome-flip rate of flagged windows) | **ADD to PM-E1 + standing monitor** | Currently a hazard sentence with no metric. Cheap once TWAP topics recorded; doubles as toxicity feature and do-not-quote screen. |
| (d) capacity methodology (PnL(α) with fills capped at α × traded volume at our level, α ∈ {10%, 25%, 50%}; rewards-share dilution ourScore/(ourScore+others)) | **ADD to PM-E3 outputs** | Parent H6 discipline; without it "small-book strategy" (H-PM6) stays a vibe and the E4 user decision package has no size number. Scale anchor: all-7-coin traded notional ≈ $10–12M/day, rewards ≈ $1.5k/day. |
| harness negative control (deliberately bad quoter: hold stale quotes through bursts; expected large negative PnL) | **ADD (1 line)** | Parent E2 keeps negative controls to catch harness bugs that make everything profitable — the single cheapest guard against a broken fill simulator. |

## S5. Kill sharpness vs parent standard

- **G2 FAIL branch**: concrete enough once the G1 dangling reference is fixed
  (S3.4) — it names the pivot (degrade to quote-around-book).
- **G3 FAIL branch**: "fall back to calibration/taker studies or stop" — "or
  stop" has no decision rule or owner. Parent names pivots and conditions.
  Fix: "kill MM track; the calibration/taker fallback vs program stop is a
  USER decision taken on the E1+E2 evidence package; no further MM work
  either way."
- **Gate criteria**: G2 lacks a significance level and week-count; G3 lacks
  t/CI machinery — covered in MF-2/MF-5.
- **Minimum-data prerequisites**: parent states them per experiment (≥14 d,
  gap fraction <5%). PM plan has only "≥2 weeks forward" for E3. Add per-
  experiment data prerequisites incl. completeness thresholds (undamaged
  window fraction, TWAP coverage) — SF-11.

## S6. Hazard-list completeness (design view): blocked vs robust

| unresolved item | blocked experiments | robust experiments |
|---|---|---|
| settlement boundary convention (ITER1 E.3: X_T≥X_0 vs literal 300s TWAP; tie/timestamp) | p̂ SPEC (model may be handicapped); near-tie exclusion definitions; τ_min placement | **G2 itself is robust** — Y = CLOB winner is observed regardless; a mis-specified p̂ loses the contest, it doesn't invalidate it. E1 must still resolve the rule before E3's quoting (unquotable-zone definition). **Add as H-PM1c** — currently absent from §6. |
| fee regime (observed 0 vs docs vs base_fee field) | none, IF S2.3 sensitivity rows adopted | all, with rows reported |
| TWAP stream no-replay + gaps | basis calibration on gap windows | G2 scoring (keep windows; model degrades honestly — S1.2.6) |
| rewards scoring formula / rates capture | G3b precision | G3a entirely |
| damaged-window blacklist not yet frozen | all gates until frozen (S2.6) | — |
| multiplicity discipline absent from §6 | — | add one bullet: "gate-bearing tests enumerated and corrected; everything else diagnostic" |

---

## Triage table

| ID | severity | finding | fix |
|---|---|---|---|
| MF-1 | MUST-FIX | G2 not specified as a paired test → difference-in-significance fallacy possible (repo's historical killer) | adopt S1.1 spec text: gate on day-clustered mean paired ΔL only |
| MF-2 | MUST-FIX | G2 spec holes: two co-primary scores, no clock/simultaneity convention, no decision grid, unit ambiguity (grid points vs windows), book-forecast edge cases, walk-forward fit rule unstated | adopt S1.2 items 1–6 (Brier primary; local-recv clock both sides; fixed 9-point grid; window = unit; paired-complete; fit ≤ d−1) |
| MF-3 | MUST-FIX | "≥1 time-stratum" = test shopping; per-coin ambiguity ×7; strata defined on model's own | d| | pooled-notional primary, 3 frozen model-free strata, Holm; per-coin descriptive (S1.3) |
| MF-4 | MUST-FIX | E2 "backfill + early forward" pools eras with different competitor forecasts — backfill has NO book data (Gamma deletes resolved markets; prices-history unverified) | define competitor per era; gate reads forward-era true-book contest only (S1.7) |
| MF-5 | MUST-FIX | G3a/b "net PnL > 0" has no significance machinery, no PnL equation, no weighting operationalization | adopt S2.1 spec: USDC per-window formula, daily sums = notional weighting, day-clustered t≥2 + bootstrap CI-lo > 0, fails-final |
| MF-6 | MUST-FIX | queue bracket: pro-rata mislabeled as a mechanism (matching is price-time); bracket labeled by presumed PnL direction (false monotonicity); 4-sided quotes double-countable on unified book | S2.2: pessimistic = trades-only decrement join-back; optimistic = front; pro-rata = diagnostic; flip ⇒ recorded ≤0; 2-queue de-dup rule |
| MF-7 | MUST-FIX | E2→E3 forking paths (params tuned on replayed windows) + freeze process has no artifact/owner/coverage | S3.1 walk-forward + design-freeze-date rule; S3.2 PM_PREREG.md contents list |
| MF-8 | MUST-FIX | power/read dates absent: 14 day-clusters is the binding constraint; notional weighting ⇒ effective breadth ≈ 1.4 coins (measured); G3a heavy-tailed | G2 readable ≥28 d + publish variance-only MDE at freeze (S1.4); G3a ≥4–6 wk or 2-wk read is provisional/FAIL-on-sign only (S2.5) |
| MF-9 | MUST-FIX | G2 FAIL branch references undefined "G1" | define or rewrite per S3.4 |
| SF-1 | SHOULD-FIX | fee-regime sensitivity rows (observed 0 vs docs vs base_fee); rebate line currently $0 | S2.3; extend H-PM4 to fees |
| SF-2 | SHOULD-FIX | rewards estimator unspecified & competitive (pro-rata vs others' scores) | S2.4: formula from docs snapshot, competitor score from band depth, ×{1.0, 0.5} haircut bracket on G3b |
| SF-3 | SHOULD-FIX | no naive-quoter baseline arm or harness negative control in E3 (parent E2 discipline dropped) | S4(a) + negative control |
| SF-4 | SHOULD-FIX | no pre-GLFT adverse-selection anatomy on OUR would-be quotes | PM-E2.5 shadow-quote markout (S4a) |
| SF-5 | SHOULD-FIX | rewards-band occupancy feasibility unmeasured | add to PM-E1 (S4b) |
| SF-6 | SHOULD-FIX | H-PM1b has no metric | basis-z screen + flip rate (S4c) |
| SF-7 | SHOULD-FIX | no capacity methodology | PnL(α) participation curve + rewards dilution (S4d) |
| SF-8 | SHOULD-FIX | beat-the-mid can be tick-quantization artifact late-window; no recalibrated-book baseline | S1.5: demote/clamp late stratum; isotonic-book secondary baseline |
| SF-9 | SHOULD-FIX | contest weights potentially ex-post (leakage via weights); universe/weights not frozen (expanding-universe = OB-timing killer) | S1.6: ex-ante weights, frozen 7-coin universe |
| SF-10 | SHOULD-FIX | exclusion blacklist not frozen; selection direction unstated | S2.6 |
| SF-11 | SHOULD-FIX | no per-experiment data prerequisites / completeness thresholds | parent-style prerequisite lines |
| SF-12 | SHOULD-FIX | "sign-stable across weeks" unquantified | ≥3 of 4 whole weeks; unreadable before 4 weeks |
| N-1 | NOTED | both scores proper; log-loss fine as secondary WITH frozen clip | — |
| N-2 | NOTED | G3b well-powered at 2 wk (rewards quasi-deterministic), G3a not — read-order asymmetry | S2.5 |
| N-3 | NOTED | heavy-tail PnL: report worst-day/worst-window shares | — |
| N-4 | NOTED | measured scale anchors: BTC ≈ 85% notional, zero-trade windows 8–20%, ~$10–12M/day traded vs ~$1.5k/day rewards | power + capacity context |
| N-5 | NOTED | G3 FAIL "or stop" needs a decision owner (user) | S5 |
