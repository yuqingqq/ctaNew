# PM_MM_PLAN — Market Making on Polymarket Crypto 5-min Binaries (P-2026-003)

> **⚠ For current state read [`FLOW_MODEL_STATE.md`](FLOW_MODEL_STATE.md).** This
> document is **provenance** — correct about its own moment, not a statement of
> current belief. Where it conflicts with `FLOW_MODEL_STATE.md`, that page wins.


Focus decision 2026-08-19: **Polymarket MM is the program's primary track**
(user call). Taker/latency-arb and pure calibration trading are secondary — the
taker niche is documented as already-occupied (live-trading writeups of 5-min
BTC latency bots exist), while the maker side has a structural subsidy.

Theory base: the P-2026-002 stack (GLFT quoting, microprice/flow fair value,
markout accounting, queue bracketing) adapted to a **bounded, expiring, digital
asset**, plus prediction-market-specific results (Dubach 2026 Polymarket
microstructure panel: longshot spread premium, 2–4 c/side spreads on 5-min
crypto; Glosten–Milgrom as the spread-persistence frame; LMSR/Othman–Sandholm
lineage noted but not used — Polymarket is a CLOB, not a scoring rule).

---

## 1. Why the economics point at maker, not taker

**Fee ground truth — RESOLVED on-chain (sketch-review M), replacing the earlier
`fee_rate_bps = 0` reading.** The WS `fee_rate_bps` field is simply
unpopulated; the real fee appears in the Polygon `OrderFilled`/`FeeCharged`
logs and matches the documented formula exactly (verified to the cent on our
own recorded trades: predicted `94.38 × 0.07 × 0.25` = $1.651650 vs on-chain
1,651,650 µUSDC; a p = 0.99 trade matched $0.001440 vs $0.001441). Fee scales
with min(p, 1−p), so it is maximal ATM and vanishes at the extremes.
`base_fee = 1000` is a legacy 10% signature cap, not a live rate.
**Never read fees from the WS field.**

| Cost leg | Binance perp (P-2026-002) | Polymarket crypto 5-min |
|---|---|---|
| maker fee | +1.8 bps (VIP0+BNB) | **$0** (maker legs pay 0 on-chain — verified) |
| maker rebate | none reachable | **≈70 bps of notional per fill** (ATM; scales with min(p,1−p)) |
| resting subsidy | none | liquidity rewards, **$550k/MONTH** ≈ $18.3k/day across 5-min markets — **announced for August only** |
| taker fee (our chase/hedge cost) | 4.5 bps | **≈3.5% of notional ATM** (350 bps — an order of magnitude above any CEX) |
| observed spread | 0.026–0.9 bps (majors–alts) | **2–4 c/share ≈ 400–800 bps of a $0.50 share** |

Reading: the maker side is subsidised twice (rebate per fill + resting
rewards) while the taker side pays punitively ATM — which is *why* spreads can
persist at 2–4 c. The open question stays **adverse selection**, but with two
corrections: (i) crossing the spread ourselves (chase/unwind) is
near-prohibitive at 350 bps, so the model must strongly prefer holding to
resolution over unwinding — this reinforces §3's Bernoulli inventory penalty
and makes the pair-merge route the only cheap exit; (ii) the rewards line is a
**time-boxed** subsidy (August announcement), so a strategy that needs it is
renting a policy decision, not owning an edge (hazard H-PM4, now a program
time-box).

## 2. Market model — what a 5-min up/down share actually is

**Settlement mechanics (VERIFIED from live market metadata 2026-08-19, updated
after the 2026-08-07 rule change):** resolution source is the **Chainlink
60-second TWAP Data Stream** (`resolutionSource:
data.chain.link/streams/{coin}-usd-twap-60s-streams`, per-market field now
captured). Let X_t = (1/w)∫_{t−w}^{t} S_u du be the stream value (w = 60 s,
S = Chainlink aggregate price). The market resolves **Up iff X_T ≥ X_0** —
settlement AND strike are both 60s-smoothed stream readings; ties go Up;
window T = 300 s. History: before 2026-08-07 settlement was a single snapshot,
which a Stanford/SMU study showed was manipulated for ~$8.2M by oracle-snapshot
sniping — the TWAP change + the $1M rewards program are the response. All
pre-Aug-07 data is BOTH a different rule and a contaminated game (§4).

Remaining settlement variance at decision time t (BM approx, τ = T−t):

```
strike:  K = X_0, KNOWN at window open if the stream is observed
         (depends on PRE-window prices [−60 s, 0] — strike forms before t=0)
t ≤ T−w: Var_t[X_T] = σ²·(τ − w + w/3) = σ²(τ − 40 s)     [drift-to-window + averaging]
t > T−w: X_T = (m/w)·TWAP_locked + unknown remainder,  m = t−(T−w), r = T−t = w−m
         Var_t[X_T] = σ²·r³/(3w²)                          [locks as r³ — fast pinning]

p̂_t = Φ(d_t),   d_t = ( E_t[X_T] − K ) / σ_eff(t)
E_t[X_T]: pre-window ≈ F_t + μ̂·(τ − w/2);  in-window = locked part + (r/w)·(F_t + μ̂r/2)
```

**Fair-value construction — STREAM-ANCHORED, not basis-corrected** (review
T-F12, the loop's most consequential model change): a Binance-synthetic X̂ with
a fitted basis carries **1.6–3.2 c of ATM p̂ error at window open and exceeds
σ_eff entirely inside r ≈ 17–26 s** — i.e. exactly where quoting decisions
matter, the proxy is noise. Correct construction, now that the Chainlink TWAP
topic is collected:

```
level     ← live crypto_prices_twap_sixty (the settlement series itself)
increments← Binance mid changes since the last stream tick (1 s cadence gap-fill)
locked    ← ∫ of the stream over the elapsed part of [T−w, T]
σ_eff     ← settlement vol ⊕ σ_⊥ (stream-vs-Binance residual, measured in PM-E1)
μ̂         ← 0 by default (T-F13: needs ~0.4%/day-equivalent to matter)
```

New pull trigger: **stream staleness** (no TWAP tick within N s → widen/pull,
since the anchor is gone). β-variogram of stream-vs-Binance residual is a
PM-E1 deliverable and sets both σ_⊥ and N.

Structural consequences vs a perp:

1. **Bounded digital with a softened endgame**: binary local vol
   λ_bin(t) ≈ φ(d_t)·∂d/∂F·σ_F explodes near the money into expiry, but the
   60 s averaging pins X_T at rate r³ — the danger zone is the ENTRY into the
   averaging window (t ≈ T−60 s, |d| small), not the final seconds. The
   underlying-delta of the settlement also decays linearly (r/w) inside the
   window: late fills carry less directional information but higher
   probability-gamma. **But that is only the UNCONDITIONAL statement** (T-F4):
   conditional on still being near the money, λ_bin = √3·φ(d)/√r *diverges*
   (1.73× a snapshot binary at matched r), and ~5% of windows are near-money
   at r = 10 s. Hence the pull rule is a (|d|, r) surface (§3), never a
   τ_min line — averaging protects the average window, not the dangerous one.
2. **Terminal time is a feature**: inventory held to resolution becomes a
   Bernoulli(p̂) payoff — "flattening" is optional. The inventory penalty is
   resolution variance q²·p̂(1−p̂), not diffusion over a holding period.
3. **Complement structure**: Up and Down books mirror (arb: p_up+p_down ≤ 1 +
   fees). A maker quotes FOUR sides; simultaneous bid fills at combined cost
   < $1 merge into a riskless redeemable pair — spread capture with zero
   directional inventory. Two-sided resting also maximizes the rewards score.
   Pair-harvest rate vs one-sided (toxic) fill rate is the core empirical
   object.
4. **Tick constraint (corrected)**: tick = $0.01 with 2–4 c spreads ⇒ books
   are 2–4 TICKS wide (not 20–40 as first drafted) — δ* granularity is coarse
   and the join-vs-improve decision matters, closer to Binance-major
   mechanics than free spread-setting. Rewards band: rest within
   `rewardsMaxSpread` (4.5 c observed) of mid at ≥ `rewardsMinSize` (50
   shares observed) to qualify; min order 5 shares. All four params are
   per-market metadata now captured at discovery.

## 3. The MM model (binary-GLFT)

State: p̂_t (model), book (b,a) both tokens, net inventory q = q_up − q_down
(shares), τ = T−t, Binance state (burst flag, ε̂ sign forecast) from
collect_hf streams.

```
reservation:  r_t = p̂_t − γ·q·v(t)
v(t) = MIN-structure, never a sum (T-F9: p̂(1−p̂) IS the total remaining
       quadratic variation of p̂ — adding a λ_bin term double-counts):
         held-to-resolution → v = p̂(1−p̂)
         unwound pre-T      → v = ∫ λ̂_bin² over the intended unwind horizon
       (exact CARA quotes a(q), b(q) = (1/γ)·ln g-ratios are closed-form and
        preferred at |q| where the linear form drifts ~2 ticks — T-F7)

QUOTE PLACEMENT — DISCRETE PER-LEVEL EV, not continuous δ* (T-F8: on a
$0.01 grid with 2-4-tick books, (1/γ)ln(1+γ/k) collapses to ≈ one tick and
is fitted on ~2 points — the closed form is not meaningful here):
   choose ℓ* = argmax_ℓ  λ_fill(ℓ, queue; bracket) · [ P_ℓ − CE_quote(q) − ζ(ℓ) ]
   over ℓ ∈ {join best, improve 1 tick, rest 1-2 ticks back, stay out},
   per side, per token. Exponential λ(δ) demoted to a smoother/prior only.
ζ(ℓ):  per-level adverse selection, markout-calibrated (PM-E2.5 shadow quotes)
       + pickoff floor Ê[|Δp̂ over requote latency| · 1{burst}]

pull rules — a (|d|, r) SURFACE, not a τ_min line (T-F4: ATM binary vol
λ_bin = √3·φ(d)/√r DIVERGES; ~5% of windows are near-money at r = 10 s):
   pull when  φ(d)·√(3/r)·√(reaction time) ≳ 1 tick
   + stream-staleness trigger (above)
   + Binance burst flag (λ_fast/λ_slow > κ) → pull the threatened side
   + |q| > Q_max → one-sided unwind mode
```

Notes against the SOTA map:
- **Finite-horizon A-S applies, not the ergodic GLFT** used on perps — T is
  real here. But because inventory expires worthless-or-full rather than
  requiring liquidation, the terminal penalty is the Bernoulli variance, and
  the quasi-static parameter feed (re-fit k_t, λ_bin per decision tick)
  replaces the closed-form time dependence. The Cartea–Wang lesson (alpha =
  reservation shift, capped) carries over verbatim: p̂ already embeds the
  Binance alpha; the cap is |r − book mid| ≤ ρ_max to survive model error.
- **Adverse selection enters at the same three hooks** as P-2026-002 (skew
  from markout surface, widen on burst, pull at extremes). The informed
  trader here is concretely identifiable: anyone faster on the same Binance
  feed. Our own 100 ms feed doubles as the defense radar — pull-on-burst is
  the single most important control (Albers Sharpe −109 logic, binary
  edition: a stale 0.50 bid during a 30 bps Binance drop is free money for
  the sniper).
- **Rewards-aware quoting**: the liquidity program pays for two-sided depth
  near mid (band rules TBD from docs/data in PM-E1). This creates an optimal
  band-riding problem — quote inside the rewarded band at minimum toxic
  exposure. Rewards are booked as a SEPARATE PnL line, never netted into the
  markout gate (a strategy that only works with subsidy must be visible as
  such).
- **Cross-venue hedge (refinement, PM-E3 ablation)**: net binary delta can be
  hedged with the Binance perp (delta = φ(d)·∂d/∂F per share), converting
  resolution risk into basis risk. Costs taker fees on Binance — only worth
  it above a size threshold; test as an ablation, not core.

## 4. Data plan — three price layers plus the market

The settlement chain is: Binance et al. → Chainlink aggregate S → 60 s TWAP
stream X_t → resolution. We trade against people watching the same chain.
Layers and their collection status:

| Layer | Source | Status | Role |
|---|---|---|---|
| L1 market | PM collector: books, price_changes, trades (fee_rate_bps), CLOB-confirmed winners, per-market rewards params + full descriptions | RUNNING (14:32; fields extended 15:15) | replay, fill curves, markout labels, fee/rewards empirics, ground truth |
| L2 signal | Binance HF feed (bookTicker/depth20/trade, 100 ms) | RUNNING (12:45) | F_t, σ̂, μ̂, burst/defense — NOT the settlement price |
| L3 PM price feed | `wss://ws-live-data.polymarket.com` topic `crypto_prices` (public, probe-verified) | RUNNING (15:15, collect_pm_prices.py) | PM's own displayed price; **identity unknown** — Chainlink relay vs spot mirror; review-loop item #1 |
| L4 settlement truth | Chainlink 60s-TWAP Data Stream (`data.chain.link/streams/*-usd-twap-60s-streams`) | **OPEN GAP** | X_0 (strike) and X_T (settlement) per window; basis calibration target |

L4 access paths, in preference order (review loop resolves): (a) L3 turns out
to BE the stream (test: tick-for-tick comparison vs Binance mid — mirror if
identical); (b) public read on data.chain.link (rate-limited 429 on first
probe — retry politely/authed); (c) on-chain: settlement posts a verified
stream report per market on Polygon — recover (K, X_T) pairs per window from
the resolution transactions (endpoint truth without the path); (d) Chainlink
Data Streams API (commercial; last resort). Even endpoint-only truth (c)
suffices to calibrate the Binance-synthetic TWAP proxy: build
X̂_t = 60s-TWAP of Binance mid + basis, validate X̂_T vs recovered X_T.

**Historical backfill (build after the loop clears data gaps):** enumerate via
paginated CLOB `GET /markets` (Gamma deletes resolved 5-min markets — verified);
Data-API trades per market. **Split hard at 2026-08-07**: pre-change windows
settled on snapshots AND carry the manipulation-era flow ($8.2M sniping game,
now extinct) — usable for fee/liquidity anatomy only, NEVER for calibration or
markout economics. Post-change backfill is only ~12 days and mostly overlaps
our forward collection.

| Aux | Status |
|---|---|
| Rewards/fee program docs snapshot (versioned; rules changed 2026-08-07) | to fetch → data/pm_5min/docs/ |

Book *depth/queue* studies need forward data only (~2 weeks, readable
~2026-09-02). Anatomy + model contest run earlier on forward + post-change
backfill.

## 5. Experiment ladder (prereg discipline: gates numeric before data is read)

**PM-E0.5 — data-completeness review loop (RUNS FIRST; charter
`PM_REVIEW_LOOP.md`).** Iterative adversarial review of this plan + the
collectors against the model's input needs, until two consecutive clean
iterations. Iteration-1 agenda: L3-identity test (PM feed vs Binance mid,
tick-level), L4 recovery path (on-chain settlement reports; data.chain.link
access), traceability matrix (every §2–§3 symbol → a collected series),
collector field audit, backfill-design constraints. No experiment runs before
the loop clears the data layer.

**PM-E1 — anatomy (forward + post-change backfill).** Descriptive, feeds every
prior: fee_rate empirics by price/side/coin; settlement reconstruction on OUR
windows (synthetic X̂_T from L2/L3 vs CLOB winners — proxy accuracy by
time-to-end AND the tie-zone width |X_T − K| where proxies disagree);
calibration curve of book price → outcome by time-in-window (longshot bias per
Dubach; post-change data only); volume/flow timing within the 300 s (endgame
concentration, thin-window fraction — volumeNum=2 windows exist); spread/tick
structure (2–4 ticks wide — join-vs-improve mechanics). No gate — but the
PM-E2/E3 gate numbers and the hazard list are FROZEN at the end of PM-E1,
before any edge computation.

**PM-E2 — model contest (backfill + early forward).** P̂ vs book mid as
predictors of resolution: log-loss + Brier, OOS walk-forward by day,
stratified by time-in-window and |d|. Gate G2: model beats book in ≥1
time-stratum with day-clustered significance, sign-stable across weeks.
FAIL → no informational edge: MM degrades to quote-around-book + rewards
harvesting — still potentially viable (G1 economics only), but the plan's
alpha claims are dropped.

**PM-E3 — MM economics replay (≥2 weeks forward book data).** Replay
binary-GLFT vs recorded books; fills under a queue bracket (price-time
priority reconstruction from book+price_change stream; pessimistic = join
back, moderate = pro-rata share — verify PM matching rules first);
fee/rebate accounting from measured rates; rewards estimated per program
rules but reported as a separate line. Primary metric: **net markout-to-
resolution PnL per window**, day-clustered, block bootstrap, notional-
weighted (E1 lesson: never eq-weighted). Gates:
- G3a: net PnL > 0 under pessimistic queue WITHOUT rewards → real edge;
- G3b: net PnL > 0 under pessimistic queue WITH rewards only → subsidy
  business (viable but fragile — flagged as such);
- both negative, or sign flips across the queue bracket → kill MM track,
  fall back to calibration/taker studies or stop.
Ablations: pull-on-burst on/off (expected dominant), toxicity skew on/off,
pair-harvest vs one-sided, cross-venue hedge, rewards-band riding.

**PM-E4 — live paper / min-size probes.** Blocked on deployment questions
(CLOB API auth, US-access/KYC status, USDC rails) — explicitly out of research
scope; goes to the user as a decision package if PM-E3 passes. Probe orders
would calibrate the queue model exactly as in P-2026-002 E5.

## 6. Hazards (standing, inherited + binary-specific)

- **H-PM1 settlement basis (now concrete)**: settlement reads the Chainlink
  60s-TWAP stream, we signal off Binance — two hops of basis (venue aggregate,
  then averaging). p̂ is systematically wrong exactly at the money if X̂ is
  mis-modeled; near-tie windows (|X_T − K| inside proxy error) are
  UNQUOTABLE until basis is calibrated on truth pairs. Also: ties resolve Up
  (≥) — a real asymmetry at d≈0.
- **H-PM1b manipulation history**: the pre-2026-08-07 game was snapshot
  sniping (freshly documented, $8.2M); the TWAP fix reduces but does not
  eliminate oracle-pushing incentives (60 s of Chainlink aggregate on thin
  coins is pushable in principle). Watch for anomalous Binance/aggregate
  divergence in the averaging window — as both a risk AND a toxicity feature.
- **H-PM2 endgame toxicity**: flow concentrates near resolution when the
  digital is pinned; fills there are maximally informed. Time-stratified
  markout mandatory; τ_min pull rule is load-bearing.
- **H-PM3 queue/matching opacity**: PM matching (price-time? operator
  batching?) must be verified empirically before any fill model is believed;
  bracket rule everywhere; sign-flip across bracket = fail (unchanged).
- **H-PM4 subsidy fragility**: rewards program rules changed within 2026 and
  will change again; G3a (no-rewards) is the only durable pass.
- **H-PM5 competition**: latency bots already farm the taker side; assume
  adversaries hold the same Binance feed with lower latency. Never model the
  counterparty as noise; δ_tox floor and pull rules exist because of this.
- **H-PM6 smallness**: $550k/yr total 5-min rewards and thin books cap
  capacity; this is a small-book strategy — size claims deferred to probes.
- Weighting/statistics discipline identical to P-2026-002 (notional-weighted
  gates, day-clustered, block bootstrap, fails-final/passes-provisional).

## 7. Immediate build queue

1. `pm_backfill.py` — enumerate historical slugs, pull Gamma metadata +
   resolutions + Data-API trades (+ prices-history where available).
2. `pm_e1_anatomy.py` — PM-E1 tables; freeze PM-E2/E3 gate numbers at its end.
3. Rewards/fee docs snapshot into `data/pm_5min/docs/` (versioned).
4. PM-E2 model contest harness (needs 1 + the running Binance feed).

---

## 8. Amendments from sketch-review iteration 1 (2026-08-19/20)

Reviews: `PM_SKETCH_REVIEW_ITER1_T.md` (theory), `_S.md` (statistics),
`_M.md` (mechanism). 13 MUST-FIX raised. Status below; §2/§3 already rewritten
inline for the T items.

**Model (T) — APPLIED inline:**
- T-F12 stream-anchored fair value replaces basis-corrected synthetic (§2).
- T-F9 v(t) is a MIN-structure; p̂(1−p̂) already is total remaining QV (§3).
- T-F8 discrete per-level EV replaces continuous-δ GLFT half-spread (§3).
- T-F4 pull rule is a (|d|, r) surface, not τ_min (§3).
- CONFIRMED CORRECT by re-derivation + Monte Carlo: all §2 variance and
  expectation laws incl. the w/2 term and the zero cross-term (T-F1–F3).
- SHOULD-FIX queued for iter 2: exact CARA quote ratios (F7), fat-tailed link
  as a PM-E2 variant (F14), μ̂=0 default + single alpha channel (F13),
  pre-open quoting formula Var = σ²(T−w/3) (F6a), boundary-jitter number
  (F6b), γ-calibration recipe + missing γ/2 (F10).

**Experiments (S) — to be adopted verbatim into `PM_PREREG.md` (not yet
written; that document IS the freeze artifact, per MF-7):**
- MF-1/2/3: G2 becomes a **paired** test on day-clustered mean ΔBrier
  (difference-in-significance fallacy is this repo's historical killer);
  Brier primary, log-loss secondary with frozen clip; local-recv clock both
  sides; fixed 9-point decision grid; window = unit; paired-complete only;
  fit ≤ d−1; pooled-notional primary with 3 frozen model-free strata + Holm;
  per-coin descriptive only (no stratum shopping).
- MF-4: backfill has NO book data → the G2 gate reads the **forward-era
  true-book contest only**; era-specific competitor definitions.
- MF-5: G3a/b get a written USDC-per-window PnL equation, daily sums as the
  notional weighting, day-clustered t ≥ 2 AND bootstrap CI-lo > 0, fails-final.
- MF-6: matching is **price-time** — pessimistic = trades-only-decrement
  join-back, optimistic = front-of-queue, pro-rata demoted to diagnostic;
  bracket sign-flip ⇒ recorded ≤ 0; explicit de-dup rule for 4-sided quotes on
  the unified book.
- MF-7: walk-forward + a design-freeze date; `PM_PREREG.md` with named owner
  and coverage list.
- MF-9: the dangling "G1" reference is removed/defined.

**READ DATES REVISED (MF-8) — the schedule was too optimistic.** Day-clusters
are the binding constraint and notional weighting makes effective breadth
≈ 1.4 coins (BTC ≈ 85% of notional; 8–20% of windows have zero trades):
- G2 (calibration contest): readable at **≥28 days** forward data (~2026-09-16),
  with a variance-only MDE published at freeze.
- G3a (MM economics, no rewards): needs **4–6 weeks**; a 2-week read is
  provisional and may only produce a FAIL-on-sign, never a pass.
- G3b (with rewards) is well-powered at 2 weeks (rewards are quasi-deterministic)
  → read-order asymmetry is expected and must not be mistaken for edge.

**New experiment arms accepted (S §4):** PM-E2.5 shadow-quote adverse-selection
anatomy before any GLFT replay; naive-quoter baseline arm + harness negative
control in E3; rewards-band occupancy feasibility measured in PM-E1;
H-PM1b basis-z manipulation screen with a flip-rate metric; capacity via a
PnL(participation α) curve with rewards dilution.

**Scale anchors measured during review** (feed capacity + power work):
BTC ≈ 85% of per-window notional (median $30.6k vs $0.2–3.4k other coins);
8–20% of windows by coin have zero trades; ~$10–12M/day traded across these
markets vs ~$1.5k/day of liquidity rewards.

**Mechanism lens (M) — 5 MUST-FIX:**
- M1 **G3a as written is not a valid counterfactual.** Rewards and adverse
  selection are COUPLED through the qualifying-band constraint: deleting the
  rewards line from a rewards-optimised policy measures nothing. G3a must
  **re-optimise the policy under a no-rewards objective** and compare
  like-for-like. (This invalidates the original "separate PnL line" framing as
  a gate mechanism — it survives only as reporting.)
- M2 **Rewards band params must come from the CLOB registry**, not Gamma:
  `GET /rewards/markets/current` (paginated) — it also carries the per-market
  `rate_per_day` Gamma never exposes. Gamma served a stale band after a
  2026-08-20 re-cut. APPLIED to the collector; Gamma copies retained only to
  measure the discrepancy.
- M3 **Pool is $550k/MONTH, not /yr** (12× my error) and is announced for
  August only → H-PM4 becomes a **time-box on the whole program**, not a
  footnote.
- M4 §1 fee table replaced with on-chain ground truth (above).
- M5 **Latency was unmodelled and dominates δ_tox**: PM infra London, our box
  us-east-1, Binance Tokyo ⇒ ~120–250 ms Binance→PM order path. At ATM, 1 bp of
  BTC ≈ 2.6–3.7 c of probability — i.e. **one bp of underlying move is ~3
  ticks of binary**. Consequences: (a) the (|d|, r) pull surface must be
  parameterised by measured one-way latency, not an assumed reaction time;
  (b) any PM-E3 replay must charge this latency; (c) co-location/venue-proximity
  is a first-order question for deployment, not an afterthought.

**Capacity (M):** gross subsidy $50–250k/mo across the program; realistic
**net $10–30k/mo (plausibly $0)** on $25–100k capital, conditional on the
rewards program surviving beyond August. This is the honest ceiling — it sizes
how much further research spend is rational.

**M2 CORRECTION (orchestrator, verified 2026-08-20 while applying the fix):**
the recommended switch to the CLOB rewards registry does NOT resolve the
rewards params for our markets — **5-min crypto markets are ABSENT from that
registry entirely** (exhausted pagination: 33 pages / 16,172 rows, our
condition_ids not present; `GET /rewards/markets/<cid>` also returns empty).
So the 1.5 c band figure cited by the review comes from markets that are not
ours. Net position: **neither Gamma nor the CLOB registry provides verified
rewards params for 5-min crypto markets**, and even their reward-eligibility
is unconfirmed. The collector now records the registry-miss (and would capture
the transition if they ever appear). Consequences:
- **G3b is gated on an unverified mechanism** — it cannot be pre-registered
  until the rewards rule for these specific markets is established.
- Iteration-2 task: establish it from primary sources (program docs/announcement
  for the August 5-min program, an actual rewards payout observed on-chain to a
  known maker, or Polymarket support/docs) — or declare G3b unrunnable and let
  **G3a (re-optimised, no-rewards) carry the program**, which by M1 is the
  scientifically cleaner gate anyway.
- This strengthens the H-PM4 time-box: a subsidy we cannot even parameterise
  from public data is not a foundation to build a strategy on.

---

## 9. Scope change (user, 2026-08-20): mechanism-first, no PnL estimation

**Dropped from scope:** PnL/capacity estimation and sizing (S-lens SF-7,
M-lens capacity bound). No forecast of earnings is a deliverable; the $10–30k/mo
figure stands only as a recorded aside, not a research target.

**Now the program's core:** the MECHANISMS of this venue, the SOTA MM theory
that each mechanism selects, and experiments that test mechanism behaviour
directly. Gates become mechanism-truth gates (does the mechanism behave as
modelled?) rather than profit gates.

### Mechanism inventory (the organizing frame)

| # | Mechanism | Status | Theory it selects |
|---|---|---|---|
| M-1 | Matching: price-time priority, off-chain operator matching, on-chain settlement | price-time confirmed (M lens); batching/latency unverified | queue-position value, fill-probability models |
| M-2 | Order lifecycle: tick $0.01, min size 5, cancel/replace latency, rate limits | partially known | discrete per-level quoting (§3), requote policy |
| M-3 | CTF split/merge/redeem: Up+Down = $1 | intra-window merge cost/latency UNVERIFIED | conversion/reversal arbitrage; creation-redemption (ETF analogue) |
| M-4 | Fee: taker ∝ min(p,1−p); maker rebate ≈70 bps/fill | resolved on-chain | fee-aware quoting; make-take economics |
| M-5 | Rewards band = a quoting obligation with params we cannot verify | UNVERIFIED for our markets | designated-MM obligation literature (constrained quoting) |
| M-6 | Settlement: Chainlink 60 s TWAP, Up iff X_T ≥ X_0, r³ pinning, tie→Up | verified; stream collected | Asian/TWAP digital pricing; pinning |
| M-7 | Latency topology (PM London / Binance Tokyo / box us-east-1); 1 bp BTC ≈ 3 ticks ATM | measured indirectly | stale-quote sniping, speed-bump/last-look, Budish–Cramton–Shim |
| M-8 | Expiry: inventory self-liquidates into Bernoulli payoff | modelled | finite-horizon MM with terminal payoff, no liquidation leg |

Iteration 2 of the sketch loop works this table: for each mechanism, (a) the
SOTA theory to adopt and why it beats the alternatives, (b) the mechanism-truth
experiment that verifies the venue behaves as the theory assumes, (c) what
observable in our collected data settles it.

**Delivered (lens T, iter 2): `PM_MECHANISM_THEORY.md`** — per-mechanism theory
selection with citations, the model object to implement, and the falsifiable
venue proposition (`P-M1a`…`P-M8d`) each theory needs. It also lists ten places
where §3 above is theoretically wrong or under-specified; §3 has NOT yet been
rewritten against it.

---

## 10. Iteration-2 outcome (2026-08-20): model re-based, ladder replaced

**§3 is superseded by a published model.** `arXiv 2607.17991 "Optimal Market
Making in Prediction Markets"` solves our exact setting (binary CLOB, terminal
resolution, quotes in [0,1]). Adopt it as the base; see
`PM_THEORY_CHECK_ORCHESTRATOR.md`. It independently reproduces two findings our
own review derived (terminal penalty q²p(1−p); quotes as argmax of intensity ×
(edge − marginal inventory cost)). Our additions on top, which it lacks:
stream-anchored p̂ (§2), the $0.01-tick restriction of its Hamiltonian, ζ(ℓ)
adverse selection from markouts, the (|d|,r) sniping pull surface, CTF
pair-merge as the cheap exit. Calibrate γ (running) and γ_T (terminal)
JOINTLY — for a martingale ending in {0,1} they price the same uncertainty.

**M-5 reformulated** (my "DMM obligation literature" claim did not survive
checking — that theory body does not exist): the rewards band is an OPTIONAL
action-space constraint, so solve the HJB twice —
`A: unconstrained, no rewards` vs `B: band-constrained, with rewards` — and
adopt B iff V_B > V_A. `V_B − V_A` is the implicit price of the subsidy. This
is also the formal statement of the M-lens's G3a re-optimisation requirement.

**§5 ladder replaced** by `PM_MECHANISM_EXPERIMENTS.md` (E0 → E-M6 → M1/M2 →
M3 → M7 → M8 → M4 → M5). **E-M6 (settlement truth) is the foundation gate**;
until it reads, all conditioning uses model-free |book_mid − 0.5|, never |d|.

**Three model-relevant discoveries:**
1. **w is ambiguous and testable**: descriptions say TWAP "of the time range"
   (⇒ possibly a 300 s average), not necessarily the 60 s stream reading §2
   assumes. We also record `twap_thirty`, so the convention grid can settle it.
   §2's w = 60 s is now a HYPOTHESIS, not a fact.
2. **"K known at t = 0" is likely FALSE** — TWAP recv−payload lag p50 ≈ 1.7 s.
   Sub-experiment E-M6b; affects every early-window quote.
3. **"2–4 ticks wide" is ATM-only** — `tick_size_change` (0.01 → 0.001) fires
   328× across 130 windows, so the tick regime shifts toward the boundaries.
   §2.4 corrected accordingly.

**Collector defects found by the design agent and FIXED:** recurring WS
`1013 slow consumer` drops on BTC (default max_queue=32 + blocking inline
writes) → 64 k queue + batched I/O; this was **load-correlated loss on ~85% of
notional**, i.e. data disappearing exactly when the market was busiest.
Remaining for E0: the 2026-08-20 00:46–01:03 UTC market-side gap.

---

## 11. Mechanism-theory map adopted (2026-08-20) — three model changes

Source: `PM_MECHANISM_THEORY.md`. Adopted theory per mechanism: M-1
queue-position value (Moallemi–Yuan) on queue-reactive dynamics
(Huang–Lehalle–Rosenbaum) in the large-tick regime (Dayri–Rosenbaum); M-2
Guilbaud–Pham regular/impulse control on a tick-valued spread; M-3 multi-asset
MM with exactly rank-1 covariance (Guéant; Bergault–Guéant) + Kozhan–Tham
legging risk; M-4 make/take non-neutrality under a binding tick
(Colliard–Foucault; Foucault–Kadan–Kandel); M-5 **principal–agent MM contracts**
(El Euch–Mastrolia–Rosenbaum–Touzi; Baldacci–Possamaï–Rosenbaum) + KKT shadow
price + Tullock contest; M-6 Asian digital from the §2 moments + Duffie–Dworczak
benchmark design + Kumar–Seppi/TWAP-oracle manipulation; M-7 Budish–Cramton–Shim
+ Menkveld–Zoican + Foucault–Kozhan–Tham; M-8 CARA utility-indifference pricing
of an unhedgeable binary claim (Hodges–Neuberger; Henderson–Hobson).

**Explicitly REJECTED: Avellaneda–Lipkin pinning for M-6.** Superficially it
fits ("pinning at expiry"), but our r³ effect is *variance lock-in*, not
delta-hedging feedback. Importing it would license exactly the error T-F4
caught (assuming the endgame is safe). Discipline worth keeping: a matching
name is not a matching mechanism.

### Change 1 — rewards are a CONSTRAINT WITH A SHADOW PRICE, not a PnL line
Both my two-policy framing (§10) and the original "separate PnL line" are
superseded. Correct rule: **occupy the band iff `R/X ≥ c(|d|, r)`**, where R is
the reward rate, X the band-occupancy requirement (computable from the book),
and c the pickoff cost of standing there — yielding a derived band-occupancy
*frontier* over (|d|, r) rather than a binary policy choice. Tullock contest
equilibrium drives `R/X → c`, so **the band only pays a maker whose c is
differentially lower** — i.e. rewards are not free money, they are a prize for
being harder to pick off. Two-sidedness outside p ∈ [0.10, 0.90] is a hard
constraint carrying negative far-side value.

### Change 2 — "pull-on-burst" is NOT a defence at our latency
Cancellation is a race we lose by 2–3 orders of magnitude; §3's pull rule was
wishful. Replace with an ex-ante **participation region** plus **size as the
risk knob**:
```
participate iff   m/φ(d) ≥ k·√(3L/r)        m = moat (ticks), L = latency
pickoff exposure  φ(d)·√(3L/r)   is VOLATILITY-FREE
```
Keep "pull" only for stale-feed (the one case where we are not racing).
Corollaries: the taker fee must be credited *inside* the sniper's moat; and the
moat far from the money is **the tick, not the fee** (fee ≈78% of the moat ATM
→ ≈2% at |d| = 3). This makes the tick-regime change the design agent
found (0.01 → 0.001, 328 events/130 windows) a **first-order economic
parameter**, not a microstructure footnote — the two findings meet here.

### Change 3 — λ_fill has no queue state, so our only decision is unmodelled
Add `Q_ahead` (and put the uncertainty **bracket on Q_ahead, not on λ**), the
dynamic queue term `D_ℓ`, and requote hysteresis (an impulse destroys queue
value — M-2). Runners-up now in scope: the ≈0.35-tick maker rebate is missing
from the objective; net `q` cannot express pair-harvest (carry `q_up`, `q_down`
separately); inventory cap should be `Q_max = κ/(γ·p̂(1−p̂))`.

---

## 12. Amendments from the walkthrough session (2026-08-20)

Derived by running the model end-to-end on worked examples; several were found
only because the numbers were actually computed.

### 12.1 Siting: migrate the solver to EU/London
Decision signal = the Chainlink TWAP relayed by PM's OWN `ws-live-data`
endpoint, so BOTH legs (see tick / send order) terminate at PM infra in London.
Measured-ish budget: US-east ≈195 ms (120 see + 75 send) vs EU ≈11 ms (8 + 3).
Effect on the quotable horizon r* (last second we may quote), moat = 1 tick +
fee credit:

| \|d\| | L=195 ms (US) | L=11 ms (EU) |
|---|---|---|
| 0.0 ATM | r* = 123 s | r* = 7 s |
| 1.0 | 92 s | 5 s |
| 2.0 | 13 s | 1 s |

> ⚠ **THIS TABLE IS WRONG — superseded by §16.3** (quant review): it multiplied
> the taker fee by price and applied the in-window λ_bin branch at r > w.
> Corrected ATM US r* = **46 s, not 123 s**, and r*(|d|) **peaks at |d| ≈ 0.82,
> not at the money**. Direction of the EU conclusion is unchanged (in fact
> strengthened), but the frontier SHAPE differs — we should quote slightly
> off-ATM, not at it.

**Coupling to remember:** had we stayed Binance-anchored, EU would NOT help
(Tokyo→EU is the long leg). The stream-anchoring change (§2) is what makes the
migration pay. **Refinement (see §12.5 issue I-3):** exposure depends on the
latency DIFFERENTIAL to the marginal sniper, not our absolute L.

### 12.2 Three model defects found by running the demo
1. **No boundary clamp** — the engine posted a bid at −0.02 when p̂→0. Quotes
   must clamp to the tradable grid [0.01, 0.99]; EV/fill logic must respect
   that the book ends.
2. **Q_max diverges** — `Q_max = κ/(γ·p̂(1−p̂))` returned 9.9e15 shares at
   p̂→0. Needs a hard capital/notional cap layered on top. See I-1 below: the
   defect is deeper than a cap.
3. **Pair-harvest was stated wrongly.** Short Up + long Down is NOT a pair —
   it is a doubled directional bet (payoff −$50/+$50 on 50 shares). The two
   configurations that actually lock:
   ```
   LONG  Up @a + LONG  Down @b, a+b<1 → MERGE → lock (1−a−b)
   SHORT Up @a + SHORT Down @b, a+b>1 → MINT $1, deliver → lock (a+b−1)
   ```
   Both legs must be on the SAME side. Confirms the theory-agent finding that
   net `q` cannot express pair-harvest: carry `q_up`, `q_down` separately and
   evaluate the JOINT condition. The demo also posts one side for legibility;
   production posts both, with skew setting the asymmetry.

### 12.3 σ is not a constant — and the target is the TWAP's own variance
**Target (fixed):** `Var[X_{t+r} − X_t]` where X is the 60 s TWAP — i.e.
σ_eff itself. **Source (empirical choice):** Chainlink stream and/or Binance.

Earlier claim "you cannot estimate σ from the TWAP stream" was TOO STRONG and
is corrected: it applies only to naive high-frequency RV scaled by √n (X is an
MA(60), so 1 s increments are MA-correlated and understate ~60×).
**Horizon-matched, non-overlapping estimation of the stream is unbiased and
model-free** — and is now the primary estimator.

Design:
- **Shape from Binance history** — construct a *synthetic* 60 s TWAP from
  Binance mid over years of Vision data: fit the r-dependence (**estimate α in
  `Var ∝ r^α`; do NOT assume α = 3**), time-of-day seasonal, and the
  trailing-vol-regime mapping.
- **Level from the Chainlink stream** — the correction ratio
  `κ(r) = Var_chainlink / Var_binance-synthetic`. A ratio needs far fewer
  observations; at r = 10–30 s we already have thousands of samples.
- `σ̂_eff(r) = κ̂(r) × shape(r; trailing-vol features)`, shifting weight to
  direct Chainlink estimation as history accrues.
- **σ_⊥ (basis noise, ≈0.7 bps) added as separate additive variance**, never
  fitted into the blend weights. NB at 15%/yr vol and r=30 s the basis noise
  DOMINATES settlement vol (0.42 vs 0.70 bps) — a real quiet-regime regime.
- Fit few parameters (~4 blend weights + coarse seasonal), non-negative,
  walk-forward (fit ≤ d−1, apply d), pooled across coins with a coin effect.
- **Fit by MLE on realized winners** (= log-loss), not by regression on
  realized vol: the loss function is probability accuracy, not vol accuracy.

**Why this matters:** measured sensitivity of p̂ to a 2× σ error is up to
**23 cents** mid-window on a 2–4 c book — σ is the dominant error term in p̂,
and its failure mode is systematic MISPRICING (pickoff exposure is
volatility-free, so bad σ never gets us sniped, it just makes us wrong).

Data status (13.5 h collected): r = 10–30 s estimable now (~1,500
non-overlapping samples/coin); r = 300 s too thin (114/coin, and the 8 coins
share a dominant crypto factor ⇒ effective N is a few, not 912). Skip any
interval containing one of the 56 recorded gaps >5 s.

**New experiment E-M6c (variance-law estimation)** — headline output is α with
a CI, plus κ(r). Turns the r³ pinning law from an assumption into a
measurement. If α ≉ 3 in-window (e.g. Chainlink heartbeat/deviation updates
make the aggregate stepwise), the entire endgame model — participation
frontier, do-not-quote zone, Q_max behaviour — is re-derived. Cluster errors
by window AND day; innovations within a window share one X_T.

**Known hole:** trailing estimators are blind to SCHEDULED events (CPI/FOMC).
Candidate complements to test, not needed for v1: Deribit IV as a
level/regime input; an event-calendar gate that widens or stands down.

---

## 13. Missing-component audit (2026-08-20) — flow, and what else fell out

Triggered by "do we have order-flow distribution?" — we do not, and auditing
for it surfaced several other adopted-in-theory / absent-in-plan components.

### 13.1 NEW component B8 — order-flow distribution (the engine of B5/B6)

We have rich DATA and no model. Measured on 40 BTC windows (70,052 trade
messages, ~1,750/window ≈ 6/s): fields `price,size,side,timestamp,asset_id,
transaction_hash,fee_rate_bps`; sizes p10=2 / p50=9 / p90=53 / max=11,439
shares (median ≈ $4.50); prices 0.001–0.999.

Three data facts that must be resolved BEFORE any distribution is fitted:
1. **`side` is not aggressor direction.** BUY 87% / SELL 13% is structural —
   in a two-token market you express "down" by BUYING Down, not selling Up.
   Direction is only meaningful as (side × asset_id → Up/Down). A naive
   signed-flow computation is wrong by construction.
2. **Trades are probably double-reported** (we subscribe to both tokens; one
   on-chain fill touches both). Dedup on `transaction_hash` — otherwise every
   intensity estimate is 2× too high.
3. **Our size is NOT negligible vs the tape.** `orderMinSize`=5 vs median trade
   9 shares; `rewardsMinSize`=50 ≈ p90. "We don't move the market" is an
   assumption to TEST (own-impact / Kyle-λ), not to assume.

**Experiment E-F (order-flow anatomy)** — objects to estimate, in order:
(a) arrival intensity λ(t) with self-excitation (EWMA-of-Hawkes form from
P-2026-002 R2); (b) size distribution (heavy tail + round-number atoms,
independent of intensity per the compound-Hawkes result — TEST here, don't
inherit); (c) properly signed direction; (d) **the conditional structure —
flow | book state, | time-in-window, | |d|** — this is the prize, because
`λ_fill(ℓ)` and `ζ(ℓ)` ARE conditional flow. Order-flow modelling is not a
sibling of B5/B6; it is their engine, so E-F precedes them in the queue.
Decision-relevant outputs: is flow concentrated near resolution (endgame
toxicity)? does size scale with |d|? does our resting size perturb arrivals?

### 13.2 Adopted-in-theory but ABSENT from the plan

| # | Component | Where it came from | Why it matters here | Status |
|---|---|---|---|---|
| X1 | **Propagator / transient impact** (Bouchaud) | R1's headline adoption — the formal version of "mid + simulated flow impact" | Does a taker sweep on the PM book move it transiently and revert? Decides fade-vs-follow and feeds ζ(ℓ) directly | **MISSING** |
| X2 | **PM book's own information** (microprice/OFI of the binary book) | R1 (microprice), C7 (OFI vs TFI) | p̂ currently uses ONLY the settlement stream — we ignore the book entirely. We want to BEAT the book, but the optimal estimate BLENDS our model with its information | **MISSING — significant** |
| X3 | **Short-horizon Binance alpha into E[X_T]** | Cartea–Wang alpha-as-reservation-shift; the whole P-2026-002 fair-value stack | We set μ̂=0 by default (T-F13: needs ~0.4%/day to matter) — but that was judged at typical σ_eff. **Late-window σ_eff collapses**, so a 0.5 bps signal at r=30 s is d≈0.44 ⇒ ~17 cents of p̂. μ̂=0 may be wrong exactly where EU latency now lets us quote | **MISSING — reopen T-F13** |
| X4 | **Queue-reactive intensities** (Huang–Lehalle–Rosenbaum) | adopted for M-1 in the theory map | No experiment estimates them; C2's join-vs-improve needs them | named, not specified |
| X5 | **Impulse control / requote hysteresis** (Guilbaud–Pham) | adopted for M-2 | C7 is still empty | named, not built |
| X6 | **Own-impact / Kyle-λ** | implied by 13.1(3) | our orders are tape-scale | **MISSING** |
| X7 | **Cross-coin structure** | Bergault–Guéant factor reduction (used only for Up/Down) | 7 coins settle on correlated underlyings; a BTC move informs the ETH window. Windows treated independently | **MISSING** |
| X8 | **Other horizons (15-min, 4-hour markets)** | rewards program covers them; we record twap_thirty | Possibly less competitive; cross-horizon structure informative. We collect 5-min only | **NOT COLLECTED** |

X2 and X3 are the two that could change the strategy's edge story, not just
its risk controls. X3 in particular contradicts a default we already set.

### 13.3 Revised review/build queue
E-F (flow) → B5+B6 (fill economics, conditional flow) → X2/X3 (does the book
or a Binance signal improve p̂?) → C4 (tail-aware sizing) → X1 (impact) →
B1/B3 (level + noise floor) → B4 (link) → X4/X5/C7 (queue + requote) →
X6/X7/X8 (impact, cross-coin, other horizons).

---

## 14. Risk layer completion (2026-08-20) — portfolio level

Audit result: per-window inventory management IS specified (§3 reservation
skew; terminal penalty `q²p̂(1−p̂)`; the never-unwind doctrine from the 350 bps
taker fee; E-M8 tests self-liquidation; E-M3 covers merge-as-capital-velocity;
E-M6c screens manipulation). What is absent is everything ABOVE a single
window. We quote up to 7 coins with overlapping 5-min cycles; nothing in the
plan aggregates them.

| # | Component | Why it matters | Status |
|---|---|---|---|
| R1 | **Correlated tail across coins** | THE compounding of I-1. Holding the 0.98 side on 7 coins is NOT 7 independent bets — one market-wide move resolves them ALL against us. Variance-based sizing treats them as diversified; they are one factor | **MISSING — most serious** |
| R2 | Cross-coin inventory skew (Bergault–Guéant multi-asset Γ) | adopted in theory for the Up/Down pair only; the same rank-reduction applies across coins | named, not applied |
| R3 | Capital allocation across concurrent windows | 288 windows/day/coin × 7, each tying capital to resolution. Capital velocity, not risk, may be the binding constraint | partial (E-M3 merge) |
| R4 | Kill switch / stand-down triggers | P-2026-002 had explicit triggers; PM has NONE defined. What makes us stop entirely — stream staleness, resolution failure, a winner flip, basis blowout? | **MISSING** |
| R5 | Rate-limit budget across coins × sides | 7 coins × 2 tokens × 2 sides = 28 quote streams against an unmeasured order/cancel limit | **MISSING** |
| R6 | γ and γ_T joint calibration recipe | flagged in §10 (running and terminal penalties price the same uncertainty) — still no procedure | **MISSING** |

### R1 in detail — the tail compounds, and our sizing hides it
At p̂ = 0.98 a single position needs ~49 wins per loss (I-1). Across 7
correlated coins the *losses arrive together*: the adverse resolution is one
underlying crypto move, not seven independent draws. Effective breadth is
≈1–2, not 7 (the same crypto-beta factor already measured in the σ work:
8 coins, effective N of a few). So the correct aggregate constraint is on
**joint loss-given-adverse-resolution**, e.g.

```
Σ_coins  |q_c| · (1 − p̂_c)   ≤  L_max        (worst-case simultaneous adverse)
```
not a per-market variance cap. This is the portfolio version of C4's fix and
should be specified together with it.

### Additions to the queue
R1+C4 together (tail-aware sizing, per-market AND aggregate) move UP the
review order — ahead of X2/X3 — because they are the difference between a
strategy that survives a bad day and one that does not. R4 (kill switch) is
cheap to specify and should be written before any live step. R3/R5 are
mechanical but block deployment. R2/R6 are modelling refinements.

---

## 15. Restoring the signal half (2026-08-20) — specifications

Per the theory diff (`PM_VS_MM_THEORY_DIFF.md` §4): re-basing fair value on the
oracle stream silently dropped the entire signal half of the P-2026-002 stack.
These specs restore it. Each is a component + an experiment with a decision
rule; all inherit the standing statistics discipline (paired where applicable,
day- AND window-clustered, block bootstrap, Holm across the family,
notional-weighted, frozen strata, walk-forward fit ≤ d−1).

### E-X2 — does the PM book know something we don't? *(the pivotal experiment)*

**Why it is pivotal.** It decides what KIND of strategy this is. We currently
ignore the book we quote on. Treat the book price `p_book` (or its
imbalance-adjusted microprice) as a rival estimator of P(Up) and fit the blend

```
p_final = w·p̂_model + (1−w)·p_book        w ∈ [0,1], fitted on resolutions
```

**Decision rule (three outcomes, all actionable):**
- `ŵ → 1` (model dominates, paired ΔBrier significant): we hold genuine
  informational edge ⇒ quote around p̂ and take the other side of the book.
  This is an ALPHA strategy.
- `ŵ → 0` (book dominates): no informational edge ⇒ the strategy is pure
  spread capture + rewards; quote around book mid with inventory skew and
  **delete every p̂-conditioned claim** in §3. This is a PURE MM strategy.
- intermediate: blend, and size conviction by |p̂ − p_book| with the frontier.

Prerequisite: E-M6 (p̂'s target must be right first). Read with G2.
⚠ Note the self-defeat check: a blend that tracks the book cannot profit from
disagreeing with it — so `ŵ` is not a free parameter to maximise fit; it is the
answer to "do we have alpha at all".

### E-X1 — transient impact on the PM book (propagator)

**Proposition.** A taker sweep moves the binary price and then partially
reverts (transient), vs moves it permanently (informed).
**Estimator.** Bouchaud-style response `R(ℓ) = E[(p_{t+ℓ} − p_t)·ε_t]` on
sweep-signed events from our `last_trade_price` + `price_change` capture
(dedup by `transaction_hash`, sign via side×asset_id per §13.1); fit
`G(ℓ) ~ ℓ^{−β}`; also the signature plot.
**Decision.** Reverting ⇒ resting into sweeps is PAID (fade) and `ζ(ℓ)` gains a
negative component; permanent ⇒ sweeps are informed, do not fade, and `ζ(ℓ)`
is strictly a cost. This directly sets the SIGN of our core P&L term.

### E-X3 — short-horizon Binance alpha into `E[X_T]` *(reopens T-F13)*

**Why reopen.** μ̂ = 0 was justified as "needs ~0.4 %/day-equivalent to matter"
— judged at TYPICAL σ_eff. Late-window σ_eff collapses (1.13 bps at r=30 s), so
a 0.5 bps signal is d ≈ 0.44 ⇒ **~17 cents of p̂**. EU latency (§12.1) has just
made late-window quoting feasible, which is precisely where this bites.
**Estimator.** Reuse the P-2026-002 fair-value stack (Stoikov microprice, OFI,
propagator, TFI) on the running Binance feed → μ̂ over horizons {1, 5, 30 s};
inject as `E[X_T] += μ̂·(effective remaining exposure)` with the Bergault cap.
**Decision.** Paired ΔBrier vs μ̂=0, **stratified by r** (expect null early,
material late). Adopt only in the strata where it wins; do NOT pool.

### E-X4 — queue-reactive intensities / E-X6 — own impact

X4: estimate arrival intensity conditional on book state (depth, imbalance,
spread in ticks) — Huang–Lehalle–Rosenbaum form; feeds `λ_fill(ℓ, Q_ahead)`.
X6: our order size is tape-scale (min 5 vs median trade 9 shares), so estimate
own-impact / Kyle-λ from any observable resting-size perturbation before
assuming price-taking. Both are inputs to B5/B6, run with E-F.

### E-X7 — cross-coin structure
7 coins settle on correlated underlyings; windows are currently independent.
(a) does a BTC move predict the ETH window's outcome beyond ETH's own stream?
(b) the inventory/risk consequence is already §14 R1 (correlated tail).
Bergault–Guéant rank-reduction applies across coins, not just Up/Down.

### X8 — other horizons (15-min, 4-hour): COLLECTION DECISION, clock running
The rewards programme covers 15-min and 4-hour markets; we collect 5-min only.
Marginal data cost is LOW (15-min ≈ 96 windows/day/coin, 4-hour ≈ 6, vs 288 for
5-min) and they are plausibly less contested. Longer horizons also have larger
σ_eff ⇒ a wider quotable frontier, and the 4-hour markets settle on the SAME
60 s TWAP we already record. **History cannot be backfilled** — like the TWAP
stream, every day unsubscribed is lost. Collector change is a one-line slug
pattern extension; awaiting go-ahead as it widens data scope.

### Queue position (revised)
E-M6 → E-F → **E-X1 + E-X2** (sign of ζ, and alpha-vs-pure-MM) → B5/B6 →
C4+R1 (tail sizing) → E-X3 → B1/B3 → B4 → X4/X5/X6 → X7 → R2–R6.
E-X2 moves early: it determines whether the program is an alpha strategy or a
spread-capture strategy, and that changes which later experiments matter.

---

## 16. Quant-review corrections (2026-08-20) — 7 MUST-FIX

Source: `PM_QUANT_REVIEW.md` (15 findings, MC-verified). §2's variance laws
(a)/(b), F6a and the pair algebra were confirmed CORRECT. The rest below are
errors in MY specifications, several of which would have produced confident
false conclusions.

### 16.1 THE BIG ONE — §12.3's estimator targets the wrong quantity
§12.3 declared the target to be `Var[X_{t+r} − X_t]`, "i.e. σ_eff itself".
**It is not.** X_t is a trailing 60 s average, so as t advances the window
ROLLS OFF — and the rolled-off part is already F_t-measurable. The rolling
increment therefore counts known information as risk:

```
rolling increment (what I specified):  Var = σ²(r²/w − r³/3w²)   for r ≤ w
conditional settlement (what we need): Var_t[X_T] = σ²·r³/(3w²)
overstatement:  4.12× at r=10 s,  2.24× at r=30 s     (MC-confirmed)
```

Worse than a scale error: fitting `Var ∝ r^α` on rolling increments yields
**α ∈ (1, 2] analytically — never 3**. Estimated α̂ ≈ 1.89 with a 95 % CI
half-width of 0.009. E-M6c would have "refuted" the r³ pinning law at ~100σ
and triggered the plan's own "re-derive the entire endgame model" clause —
**a false refutation manufactured by my estimator, not by the venue.**

**Fix (no new data needed):** fit the two-parameter variogram
`V(r; σ², w)` to the observed increment curve. On 13.5 h of ONE coin it
recovers σ² to ±7 % *and* w to ±5.2 s — so it simultaneously settles the
§10/E-M6 ambiguity over whether the averaging window is 60 s or 300 s.
E-M6c is rewritten around the variogram; the α-exponent framing is retired.

### 16.2 σ_⊥ double-counted (third instance of this pattern)
`κ(r) = Var_CL/Var_bin` ALREADY contains the stream-vs-Binance residual.
Adding σ_⊥ again on top inflates σ_eff by **32–41 %** ⇒ 6.5–8 c of p̂ error.
Use κ OR the explicit σ_⊥ floor, never both. (Compare: the running/terminal
penalty double-count in §10 and the v(t) double-count in T-F9 — three
independent instances of the same failure mode. Treat any "add a second
variance term" instinct as suspect.)

### 16.3 The participation frontier numbers were wrong
§12.1's r* table multiplied the taker fee by price and used the in-window
λ_bin branch at r > w. Corrected: **ATM US r* = 46 s (not 123 s)**; the
frontier **peaks at |d| ≈ 0.82, not ATM**. Consequences: the EU migration case
is STRONGER than stated (US is worse than I claimed), and the optimal quoting
region is slightly OFF the money rather than at it — a real change to C1/C2.

### 16.4 Q_max is not a risk budget
`Q_max = κ/(γ·p̂(1−p̂))` is dimensionally a **price-skew cap**; its own stated
derivation gives 1/√v, not 1/v. At p̂ = 0.01 it permits **50× the ATM dollar
loss** on the short side. Confirms I-1 and supplies the fix:

```
replace with:  L_adv = |q|·(1 − p̂)  ≤  κ_$        (loss-given-adverse-resolution)
```
CVaR is numerically identical for p̂ ∈ [0.05, 0.95], so the simple form is
sufficient. Combine with §14 R1's aggregate `Σ_c |q_c|(1 − p̂_c) ≤ L_max`.

### 16.5 The MLE-on-outcomes fit is not compatible with the κ/shape chain
MLE on realized winners fits a **predictive** σ (inflated 1.37× at ω = 0.5σ,
where ω is edge-measurement noise), whereas κ/shape estimates a **realized**
σ. They are different objects and cannot be composed. Additionally the Φ-link
misspecification biases the fit −3.5 … −15 %. Decide ONE route:
(a) realized-σ chain (variogram + κ), with the link tested separately; or
(b) predictive-σ MLE end-to-end, with κ/shape used only as a prior.
Do not mix. Recommend (a) for the estimator and (b) as the G2 contest arm.

### 16.6 Consequences for the ladder
E-M6c rewritten (variogram, not α). Its by-product now ALSO answers E-M6's
w-convention question, so the two should be run together. §12.3's "estimable
now at r = 10–30 s" survives — but only with the corrected estimator.

---

## 17. Deep review (2026-08-20) — 3 FATAL, 10 MAJOR. DECISION POINT.

Source: `PM_DEEP_REVIEW.md` (767 lines). Everything tagged [measured] was
computed from data already on disk (624k trade prints, 1,206 resolved windows,
140k TWAP messages, ~1.4 days). **Verdict: not sound enough to execute as
written.**

### 17.1 §12.1 SITING DECISION IS SUSPENDED (FATAL-1)
Measured TWAP-stream observation lag = **1,700 ms p50**, of which **1,440 ms is
PM-side publication delay** — a leg no siting can touch. §12.1 assumed 120 ms.
Since `r* ∝ L`: **ATM is never quotable from anywhere**, and London buys ~21 s
at |d|=2 (an 18% cut, not the 94% claimed).
Compounding: §12.1's premise ("both legs terminate at PM London") CONTRADICTS
§2, whose fair value takes *increments from Binance* — so either L ≈ 1.7 s, or
EU is the wrong direction. Cannot be both. Three values for this quantity are
live in the corpus (120 / 471 / 1700 ms); the 471 ms was measured on the SPOT
MIRROR and misapplied to the TWAP stream (3.6× slower).
**Action: no siting decision until a per-leg budget exists and E-M7 measures
which leg the PM book reacts to. Supersedes §16.3.**

### 17.2 The frontier is a constraint promoted to a policy (FATAL-2)
It contains no revenue term. Corrected for FATAL-1 it addresses only **5–10% of
window-time**, and nobody asked what is earned there. `Q_max = κ/(γv)` is one
exponent wrong (a marginal-reservation budget, not a constant risk budget ∝
v^−½): 25× inventory and 25× terminal variance at p̂ = 0.99 for +0.75 c/share
against a 97 c tail. Fix = the §16.4 loss-given-adverse cap.
**Retraction (reviewer's own, and mine):** conditioned on TRADE PRICE the far
region looks edge-free; conditioned on BOOK MID at fill (the correct,
model-free state) it does not. H-2 is UNTESTED — as is the plan's opposite
claim. Net maker edge mid-conditioned ≈ **+0.45 c/share average, ≈0 ATM**,

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** `+0.45 ¢/share` and `+95 bps` are the same book-derived number (p90 6.2 s stale books) and fall together. Book-free rebuild: **+0.17 ¢/share**, and **NOT DISTINGUISHABLE FROM ZERO** — **+0.173 [-0.251, +0.596]**, all seven per-coin CIs spanning zero. The maker-edge sign is **UNDETERMINED at two days**. `side` IS the taker's (G-FF1 `PASS`), closing the `+95 → −95` flip. See `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b.

noisy and non-monotone on 1.4 days.

### 17.3 The program cannot end (FATAL-3)
All 8 experiments terminate in "change the model". The §9 scope change deleted
G3a/G3b — the only kill gates — and cut the eligibility question as
"deployment, not research". E-M5 reads 2026-09-03, AFTER the August subsidy
window it is testing. **A STOP clause with a named owner must exist before any
experiment is read.**

### 17.4 The positive finding nobody had computed
The maker side of this venue is measurably **+95 bps of notional gross (+136

> **WITHDRAWN 2026-08-21 — DO NOT CITE.** `+0.45 ¢/share` and `+95 bps` are the same book-derived number (p90 6.2 s stale books) and fall together. Book-free rebuild: **+0.17 ¢/share**, and **NOT DISTINGUISHABLE FROM ZERO** — **+0.173 [-0.251, +0.596]**, all seven per-coin CIs spanning zero. The maker-edge sign is **UNDETERMINED at two days**. `side` IS the taker's (G-FF1 `PASS`), closing the `+95 → −95` flip. See `FLOW_UNCERTAINTY_LOOP.md` U4/U10/U10b.

with the rebate)**, with a visible mechanism: **the book is over-dispersed at
every decision time**. The favourite–longshot bias is large (**+3.6 c/share at
p ∈ [0.15, 0.35)**), stable across the sample, and mechanically explicable
(retail buys lottery tickets). That is not what a picked-over market looks
like — and it was established only by a review computing markouts, after ~3,800
lines of design.

### 17.5 THE DECISION (user's call)
**Option A — fix and continue the mechanism program.** Do the 6-item minimum
(§17.6), then resume the ladder. Cost: substantial; the apparatus is large and
four recent decisions were wrong.
**Option B — re-scope to the FLB harvest.** *"Harvest the favourite–longshot
bias as a passive two-sided maker at moderate moneyness, with a
loss-given-adverse cap, and measure the markout."* Small, cheap, falsifiable,
real measured prior, needs almost none of the current apparatus.
The reviewer recommends B. The data supports B. A is defensible only if the
mechanism detail is believed to be load-bearing for something B cannot capture.

### 17.6 Minimum before ANY experiment is read (either option)
1. H-2 capture-ratio table (realised-fill markout ÷ unconditional mid
   mispricing, by |mid−0.5| × time) — decides WHERE to quote; currently
   ownerless; ~1 day.
2. Per-leg latency budget; siting suspended until then.
3. `Q_max` → loss-given-adverse cap (§16.4) + aggregate (§14 R1).
4. Fix σ_⊥ double-count (§16.2); add a book-implied-σ arm (H-3).
5. Write the STOP clause, with an owner, into `PM_PREREG.md` — which must
   exist before the freeze date it already claims.
6. Mechanical §2/§3 rewrite so the ladder stops gating a superseded model.

### 17.7 Process diagnosis (accepted)
*"The program built theory faster than it built arithmetic."* Four consequential
decisions wrong or unsupported; each check took under an hour against data
already on disk. Standing rule going forward: **no design decision that a
measurement on existing data could settle may be recorded as settled until that
measurement is run.**
