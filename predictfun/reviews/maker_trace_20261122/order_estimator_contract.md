# MAKER-ORDER-ESTIMATOR / DECISION-LOGIC / TIMING packet — contract (frozen before any measurement)

2026-11-23, development (P-2026-005 MAKER-TRACE). This is the single contract of the whole packet: the order-history estimator with its
detailed connected episodes (`tools/order_estimator.py`) and the four census-chain tools (`tools/chain_context.py`, `tools/chain_post_fill.py`,
`tools/chain_pull.py`, `tools/chain_take.py`). Every rule below is declared before any result of this packet exists.

**Data read before this freeze:** one schema / layout probe (`logs/mt_oe_inspect.log`, 0.37 CPU s): column names, the `fills_valued`
row-group layout (90 groups, 19 holding BOOK markets), and per-market counts of top-3 orders (4,149 orders, 9,080 maker fills on the 19
BOOK markets; every top-3 S_NOPRIO removal set contains `cancelled_or_expired`). No timing, queue, economic or market-state value was read.

**Binding plans, in order:**
- timing [`../automatic/20261123_maker_decision_timing/plan.md`](../automatic/20261123_maker_decision_timing/plan.md) (sample, clocks, queue / inventory, bins, fixtures);
- logic [`../automatic/20261123_maker_decision_logic/plan.md`](../automatic/20261123_maker_decision_logic/plan.md) (connected episodes, logic table, statuses; supersedes the timing plan's primary deliverable);
- estimator [`../automatic/20261123_maker_order_estimator/plan.md`](../automatic/20261123_maker_order_estimator/plan.md) (order ledger, finite hypotheses, priors, adversarial controls);
- review [`../automatic/20261123_matched_gap_review/report.md`](../automatic/20261123_matched_gap_review/report.md), "ONE next priority" (and its carried items: boundary-sensitive records, the latent receipt-earlier-than-UUID case).

**USER directions (2026-11-23), all binding:**
1. "we need to estimate the time when they quote, do decisions etc."
2. "we need the chain of logics, not just what they do at what time"
3. "we need to estimate when they pull, and repost, post size etc."
4. "We focus on getting the logic chains from order history, like when they post, what's the market condition like, and when they pull,
   anything happens, like reverse price trend from binance, and when they get filled, is it some early posts resting there to get the front in
   queue, need to figure out the logics to get to know how they perform." → the POST / PULL / FILL chains (section 7).
5. "And they also actively take to reduce inventory, also need to check the market condition. I have not listed all actions here. You will
   analyze the actions and corresponding market context or logics." → the TAKE chain and the action-type enumeration (section 7).

**What this is:** a descriptive, evidence-graded reconstruction of public order histories and their market context on four consumed days
(Oct 24–27; point estimates, no intervals, G = 4 < 5). **What it is not:** a fitted decision model, an imitation strategy, a policy replay,
a validation, a calibrated probability, or a profitability claim. It measures no model skill. Multiplicity: 0 candidates.

## 1. Population and inputs

| Item | Definition |
|---|---|
| Addresses | ranks 1–3 of `econ/identities.json` → `ex_ante.top10_shares` (Oct 23 maker shares). They must equal `pf_mt_decision.TOP3` (else refuse). Keys use the lowercase address; two spellings of one address refuse. |
| Markets | the 19 BOOK markets: `reviews/ladder_async_fast_20261122/manifest.json` → `complete` (Oct 24: 7, Oct 25: 5, Oct 26: 5, Oct 27: 2). Refuse unless exactly 19 and every day is a measurement day. |
| Lifecycle | `life/<day>_<mid>_{orders,fills,x_links}.parquet` + `.json` (`MT_RUN=v2`; `pf_mt_lifecycle`, accepted). Reused as is; never re-run. |
| Book | `depth/<day>_<mid>_{frames,levels}.parquet`, always through `pf_mt_lifecycle._book` (usable frames, unknown cells = NaN, gaps > 1,000 ms): the lifecycle's own semantics. |
| Fills / inventory / economics | `econ/fills_valued.parquet` rows of the 3 addresses in the BOOK markets, read through the row-group `mid` statistics; every row's `day` must be its market's manifest day. Bound by sha256 (equal to the accepted `matched_gap_r1` receipt's `2a874707…cde6`, else refuse). |
| BTC | `btc/btc_<day>.parquet` through `pf_mt_decision.load_btc` / `btc_feats` unchanged (Binance USD-M bookTicker, E_ms clock; grid row g carries the last update with E_ms ≤ g). |
| Market times | `markets_<day>.parquet` (`ws`, `we`, `status`). |
| Matched-gap context | `v2/matched_gap_r1/per_market.csv` (accepted): each address's market-level completed-close net and residual, beside the episodes. |
| Own ground truth | `truth/v2/own_orders.parquet` and `life/20261108_<mid>_orders.parquet`: the instrument check only (section 9). |

**Days:** only Oct 23–27 and Nov 8 hours 00–22, through `pf_mt_common` guards or an allow-list checked on the string before any filesystem
access. No raw file, no new data, no lifecycle re-run, no replay, no fit.

## 2. Clocks, evidence grades, statuses

**Clocks (never mixed):** book `upd_ms` (venue); `settle_ms` (UUIDv7 creation: an upper bound of the match); `first_recv_ns` (collector);
Binance `E_ms`. The lifecycle's match window per record is hard `[x_lo_hard_ms, hi_ms]` (`hi − 2,000 ms`, its declared lag premise;
`−1` = unbounded) and evidence-refined `[lo_ms, hi_ms]` (a scenario refinement). Records of one settle ms are unordered.

| Event | Reported |
|---|---|
| Acceptance (posting) | per scenario `[post_lo, post_hi]` (`post_hi` = the first fill's `hi`) and their union (section 4.3) |
| Submission | acceptance − D. No delay bound is supported for other makers: **unidentified**. Declared profiles D ∈ {0, 250, 1000} ms are shown separately and never as identified. |
| Match | hard and evidence windows, UUID bound, receipt; same-ms records unordered |
| Cancel effect | the removal window, only inside a cancel-possible history (PL); exhaustion / expiry kept beside |
| Cancel submission | the effect window − D (profiles above) |
| Fill notice | private arrival unknown: recorded (venue) inventory is not client-known inventory; pending = own legs whose match window overlaps the decision window |
| Take decision | grouped by record (`sid`) and taker order (`tk_hash`); match window only; submission − D profiles |

**Evidence grades** (no probability or confidence score anywhere):

| Grade | Meaning |
|---|---|
| `A_observed` | read from identity-bearing records: a hash's fills, shares and prices, record co-membership, a later fill of the same hash |
| `B_constrained` | implied by H1–H3 window logic in every declared scenario (the same value in all three scenario rows) |
| `C_priority` | implied only under the price / time-priority scenarios S_FIFO / S_FIFO_LO (T1 / T2 `supported`); S_NOPRIO keeps alternatives |
| `D_prior` | chosen by a declared prior among evidence-tied histories (section 4.2); never evidence |
| `E_unresolved` | no non-trivial bound, or a tie with no prior applicable |
| `X_conflict` | a contradiction status (capacity, queue, FIFO, identity, clock): carried, never averaged away |

**Statuses** are `+`-joined sorted flags (`ok` = none). The lifecycle's statuses are carried. New flags: `receipt_before_uuid`,
`hi_from_receipt`, `window_unbounded`, `gap_in_window`, `unknown_cell_in_window`, `no_frames`, `simultaneous_action`, `fewer_candidates`,
`self_trade`, `excluded_row`, `no_context`. An exclusion is a status with a count, never a dropped row.

## 3. The detailed sample (≤ 96 cases; selection before any ranking)

| Rule | Definition |
|---|---|
| Maker-order unit | key `(day, market, address, order_hash)`: every hash of the address in the market's S_NOPRIO lifecycle rows, whatever its status |
| Take unit | key `(day, market, address, settlement_id)`: every record whose taker is the address (`fills_valued` role `taker`), whatever its status (self-trades included, flagged) |
| Rank within a market | sha256 of the UTF-8 text `20261123\|<day>\|<market>\|<address>\|<order_hash or settlement_id>`, hex ascending |
| Market order | per (address, day, unit kind): sha256 of `20261123\|<day>\|<address>\|<market>\|market`, ascending |
| Round-robin | round r = 0, 1, …: in market order each market adds its rank-r candidate if it has one; stop at **4** per (address, day, unit kind) |
| Fewer candidates | fewer cases, status `fewer_candidates` in the manifest |

Selection reads only key columns (`mk_hash`, `mk_signer`, `scenario`; `sid`, `role`, `addr`, `day`, `mid`) and runs before any other column of
those tables is used. Its manifest (the selection columns of `cases.csv`) and the manifest's sha256 enter the receipt. Every selected case
stays with its status; none is replaced. Maximum 3 addresses × 4 days × (4 + 4) = 96. Timelines are written for every detailed episode.

## 4. The order-history estimator (`tools/order_estimator.py`)

Inputs per selected order: its three scenario rows, its fills, the book at its cell, the address's other orders, fills and takes in the market.
**Premises stated with every result:** one hash = one signed order (H1); a removed order never re-enters and a resting order never grows
(lifecycle H2); an order is cancelled whole (no partial size reduction), which the capacity size bound `x_size_ub` relies on.

### 4.1 Size and end-of-life hypotheses (finite set)

| Code | History | Feasible under scenario s iff |
|---|---|---|
| `EX` | original size = cum (cumulative executed shares); removed by exhaustion at its last fill | `exhausted` ∈ removal_set_s (always, by the lifecycle) |
| `PL` | original size ∈ (cum + 0.01, UB_s]; the remainder r = size − cum is withdrawn (cancel or a time-in-force expiry) inside [removal_lo, removal_hi_s] | `cancelled_or_expired` ∈ removal_set_s and UB_s − cum > 0.01 (UB NaN = unbounded) |
| `EC` | original size ∈ (cum + 0.01, UB_s]; the remainder still rests at the last supported frame (market-end expiry or censored) | `end_censored` ∈ removal_set_s and UB_s − cum > 0.01 |

UB_s = the scenario row's `x_size_ub`. `missing_evidence` ∈ removal_set_s is a flag (the book cannot support the removal timing), not a
history. The feasible set is the union over the three declared scenarios; an exclusion holding only under S_FIFO / S_FIFO_LO is grade C.
"Executed total is not automatically the original size": cum is always only the lower bound.

| Size / end status | Rule |
|---|---|
| `distinguished` (B) | exactly one history feasible in every scenario (e.g. UB ≤ cum + 0.01 everywhere: capacity-tight ⇒ EX) |
| `distinguished_under_priority` (C) | one history under both FIFO scenarios, more under S_NOPRIO |
| `tied` | ≥ 2 histories feasible in every scenario: all are emitted; the representative is prior-driven (D) |
| `conflict` (X) | a contradiction status on any scenario row (feasibility still from the rows, never fitted) |
| `clock_refused` (X) | a fill of the order is `receipt_before_uuid` / `hi_from_receipt` (section 4.8): the feasible set is shown, no claim is made |

Remaining quantity after the order's j-th fill (settle order): [Σ_{i>j} q_i, UB_max − Σ_{i≤j} q_i] (lower: its later fills, observed).

### 4.2 Evidence ranking, ties and priors

A history survives unless an A / B / C observation excludes it. Survivors are **tied**: all are emitted, and none is preferred by
evidence. A representative history (for timelines and illustration only) is chosen by a declared prior and labelled `prior_driven`:

| Prior | Rule |
|---|---|
| `P_cmd` (primary representative) | fewest hidden commands attributed to the order: EX (no cancel) ≺ EC ≺ PL; ties → the smallest original size |
| `P_drop` | if, between the first frame certainly after every fill and removal_hi, the cell shows a frame interval whose drop exceeds every print that can lie in it by > 0.01 sh, PL with r = the first such excess (clipped to (0.01, UB − cum]); else as P_cmd (fewest anonymous orders: it can choose the larger size) |
| `P_lot` | round lot of 5 sh: the smallest multiple of 5 ≥ cum − 0.01 that lies in [cum, UB]: equal to cum → EX, larger → PL (EC if PL is infeasible); none fits → EX |
| `P_clip` | repeated clip: the modal S_FIFO `size_exact` of the signer's **other** orders in the same market (rounded to 0.01; the mode must occur ≥ 2 times; ties → smaller) when it lies in [cum, UB] (equal → EX, larger → PL / EC); else EX. Itself assumption-sensitive (S_FIFO). |

`prior_sensitive` = the four priors' (history, size) choices differ; counted with denominators. A prior never changes a feasible set, a
grade or a status. In the identical-public-tape pair (5-share exhausted vs 50-share partial then an unseen cancel) both histories stay
feasible and `P_cmd` chooses the 5-share history, labelled prior-driven, with the alternative shown.

### 4.3 Posting, activation and submission

- **Window:** per scenario `[post_lo_s, post_hi]`; the union `[min_s post_lo_s, post_hi]` (any −1 → unbounded; evaluation clipped to
  max(first usable frame, ws − 300 s) with status `window_unbounded`). The S_FIFO window is reported beside, never instead. Every declared
  scenario places fills on its own clock (hi or lo), so the union can be tighter than the scenario-free window logic; the lifecycle class
  (`constrained` = equal to it, `assumption_sensitive` = scenario-clock dependent) is carried beside every window.
- **Resting (queue) age at the first fill** ∈ [0, post_hi − post_lo_union]. The lower bound is 0 by construction: H1–H3 give no posting
  upper bound before the first fill, so a first fill never proves an early post.
- **Late post certain:** post_hi − post_lo_union ≤ `LATE_MS` = 1,000 ms. **Early post possible:** > `EARLY_MS` = 10,000 ms. "Early certain"
  is not available from public data.
- **Prior-narrowed posting (`P_nomask`, prior-labelled):** candidate frame intervals (f, f+1] inside the union window whose visible net
  addition at the cell, shown_{f+1} − shown_f + prints in (f, f+1] (every maker's fill at the cell placed at its evidence frame), is
  ≥ cum − 0.01 × (1 + orders accounted). An interval across a frame gap (> 1,000 ms) or with an unknown cell is never a candidate and sets
  `incomplete_gap`. Joint allocation over every order at the cell (any maker): an order with exactly one candidate claims its cum there,
  and an interval stays a candidate for an order only if its addition covers the other orders' forced claims plus that order's cum; the
  representative is assigned jointly: in ascending (first fill hi, hash), each order takes its latest candidate whose remaining visible
  addition still covers its cum, and that cum is then used up (one addition is never assigned in full to two orders). The candidate span is
  reported beside. Statuses: `single_candidate`, `multiple`, `no_visible_addition` (masking needed), `incomplete_gap`, `joint_unassignable`,
  `window_unbounded`, `conflict`.
- **Submission:** unidentified; the D profiles are shown separately.

### 4.4 Queue progression and front classes (FIFO scenarios only; S_NOPRIO has no queue)

Ahead volume at acceptance (FIFO; the history "posted just after frame f"): shown_f at the cell over the pre-post frames, from the frame at
or before post_lo_union to the last frame before post_hi: the range [min, max] (unknown cells excluded and flagged).

| Class | Rule |
|---|---|
| `front_at_acceptance` | max ahead-at-acceptance ≤ 0.01 sh: the level held nothing at every feasible posting instant |
| `front_reached_after_clearing` | min > 0.01 (volume certainly ahead at posting) and S_FIFO `ahead_hi` = 0 at the first fill's queue frame |
| `front_at_acceptance_or_reached` | min ≤ 0.01 < max and S_FIFO `ahead_hi` = 0 (timing-sensitive) |
| `behind_at_first_fill` | S_FIFO `ahead_lo` > 0.01 (rule 2: others ahead were consumed in the same record) |
| `unidentified` | S_FIFO `ahead_hi` NaN, or > 0.01 with `ahead_lo` ≤ 0.01; no frames |
| `conflicting` | `queue_contradiction`, `ahead_hi_negative`, `fifo_front_conflict`, or a capacity status |

Each later fill with S_FIFO `x_ahead_hi` = 0 after an earlier own fill counts as `front_after_partial`. A gap or resync never resets a
queue, never creates a replenishment and never proves a posting or a removal. Time to the touch: the first pre-post frame at which the
order's tick is the own-side touch, reported inside the window only.

### 4.5 Pull and remaining quantity

Inside PL: effect window [removal_lo, removal_hi_union] (open → the last usable frame, status `end_open`), withdrawn r ∈ (0.01, UB − cum].
`P_drop` locates a candidate drop interval (prior). Cancel submission = effect − D. EX and EC stay beside. A pull is `identified` only when EX
and EC are infeasible in every scenario; a disappearance alone never names a cancel.

### 4.6 Repost / replace / replenish / overlapping ladder (`link_hypotheses`, the single definition)

Candidate parents of a selected order Q (signer g, YES side s): the latest same-tick predecessor (refill) and the latest other-tick
predecessor (reprice) of `x_links`; children: the orders whose `x_links` predecessor is Q. For a pair (P earlier, Q later):

| Code | History | Feasible iff | Certain iff |
|---|---|---|---|
| `overlap` | Q rested while P was alive (second / ladder order) | Q.post_lo_union < P.end_hi (P's union removal_hi; open = +∞) | Q and P share a record, or Q.first_hi < P's last `x_lo_hard_ms` |
| `replace` | P's remainder pulled; Q posted after it | PL feasible for P, Q.post_hi > P.removal_lo, overlap not certain | never (a cancel is never identified) |
| `replenish` | P exhausted; Q posted after it | EX feasible for P, Q.post_hi > P's last `x_lo_hard_ms`, overlap not certain | never |
| `unrelated` | no relation | always | — |

Gap range: [Q.post_lo_union − P.end_hi, Q.post_hi − P's last `x_lo_hard_ms`]. Pair status: `overlap_identified`, `sequential_identified`
(overlap infeasible: Q certainly posted after P was gone) or `tied`; `scenario_dependent` = Q's posting or P's removal class is not
`constrained` (the verdict holds under the declared scenario clocks, not the scenario-free window logic). Both field spellings in use
(`post_lo_u` / `post_lo_union`, `last_lo_hard` / `last_lh`, `union` / `ex_any`·`pl_any`·`ec_any`, `records` for the shared-record test) are
accepted; the result carries `feasible` and `status`. Linked versus unrelated is never identified; new fill chronology alone
never establishes a repost. One hash with many fills (one order, A) stays distinct from many hashes at one tick (many orders, A).
`chain_take.py` imports this function; no tool redefines it.

### 4.7 What evidence is new relative to the lifecycle (declared design finding)

The lifecycle already uses identity (H1), proven presence between fills (H2), joint capacity at every usable frame (H3: posting lower bounds,
size upper bounds, gone bounds), price / time priority (scenarios) and FIFO front bounds. Beyond these, the book shows additions and
unexplained drops. Their only hard content is that **some** order was added or removed at the cell in that frame interval; a concurrent add
or cancel of an anonymous order can always mask or mimic them. They therefore exclude no history H1–H3 allow, and rank histories only under
a no-masking / fewest-hidden-commands prior (`P_nomask`, `P_drop`), reported as such and never as grade A–C. **No new solver is added.**

### 4.8 The latent receipt-earlier-than-UUID case (contract amendment: carried)

A record with first_recv_ms < settle_ms (receipt before the UUID time) mixes the collector and UUID clocks. Its windows are emitted with
status `receipt_before_uuid`, enter no precision bin (`refused_clock`) and give no bound to any claim (counted): the order's size / end
status is `clock_refused`, a link pair touching it is `clock_refused`, its logic rows are `unidentified` (grade X) with the computed
evaluation kept in `conditional_status`, and such a leg never enters a certain inventory variant. A record without a UUID
(`hi_from_receipt`) is treated the same way (its hard lower end is −1). Neither row is dropped.

## 5. Takes (detailed cases)

- **Identity:** the record (`sid`) and its taker order (`tk_hash`): every record of that taker hash in the market is listed with it.
- **Window:** the record's hard / evidence match window (all legs of a record share one); clock statuses as in 4.8.
- **Inventory** of the address in the market, from its ok `fills_valued` legs in econ's FIFO order (settle_ms, sid, px_rank, mk_idx,
  maker before taker): `pos_before` = after every action at a strictly earlier settle ms; `pos_after` = pos_before + this record's legs;
  certain variant = legs whose hi < the take's `x_lo_hard`; actions sharing the take's settle ms are `simultaneous_action` and evaluated
  both included and excluded; pending = own legs whose window overlaps the take's window.
- **Class** (actual |position|, never the opposing-side test, `v2/report_interpretation_v3.md`): `from_flat` (|before| ≤ 1e-6),
  `add` (same direction), `reduce_partial` (0 < |after| < |before|, same sign), `full_close` (|after| ≤ 1e-6), `reversal_reducing`
  (sign flip, |after| < |before|), `overshoot` (opposing, |after| ≥ |before|). Actual reduction = reduce_partial ∪ full_close ∪
  reversal_reducing. Variants that disagree → `timing_sensitive`.
- **Preceding own maker fill:** delay range [take.x_lo_hard − prev.hi, take.hi − prev.x_lo_hard] (clipped at 0 below).
- **Context:** through `chain_context` (section 7.0) over the match window: BTC r_1s / r_5s / r_30s aligned with the take, book touch /
  spread / depth, price paid vs the touch and vs p_fair, time left, fill burst.
- **Fill burst** at the take window's start lo (one expression in `order_estimator.burst_counts` and `chain_take.burst_counts`): certain =
  own maker legs whose window lies inside [lo − 5 s, lo) (x_lo_hard ≥ lo − 5 s and hi < lo); possible = own maker legs whose window meets
  [lo − 5 s, lo) (bounded: hi ≥ lo − 5 s and x_lo_hard < lo; an unbounded leg: hi ≥ lo − 5 s and settled at or before the take).

## 6. Connected episodes and the logic table

**Episode window per case:** order case [max(post_lo_union, ws − 300 s), max(last fill hi, bounded removal_hi) + 30 s]; take case
[x_lo_hard − 30 s, hi + 30 s]; both clipped to the market's [ws − 300 s, we]. Cases of one (day, market, address) whose windows overlap form
one **connected episode** (union-find), counted once; cases are never presented as independent evidence. Content: every leg of the
address with settle_ms in the window, its orders whose feasible life meets it, its takes in it.

**Economics of an episode** (the observed later outcome, never the counterfactual value of an action): Σ over its legs of `v_end` (value at
the END mark), `fee_fill`, `net_res` (to resolution), econ's FIFO matched (`mq`) and residual (`rq`) shares, the position entering and
leaving the episode (unclosed inventory), and the address's market-level completed-close net and residual from `matched_gap_r1` beside.
`profitable_continuation` / `losing_continuation` = the sign of Σ v_end − Σ fee_fill. Queue opportunity cost and the value of an
alternative are `unmeasured`.

**Logic table** (one row per decision link; the same columns for every tool that writes one):
`decision_window | state/change | alternatives | proposed_trade-off | observed_action | resulting_state | supporting_evidence |
competing_explanation | what_would_falsify | identification_status`, plus `episode_id, case_key, link, mechanism, evidence_grade,
timing_sensitive, status`. Statuses: `compatible`, `contradicted`, `timing_sensitive`, `insufficient_comparison`, `unidentified`. Private
intent is never labelled observed. A condition is evaluated at every candidate instant of the decision window (100 ms grid, features with
data ≤ the instant only); a category that changes inside the window is `timing_sensitive`.

| Link | Mechanism | Rule |
|---|---|---|
| POST | `queue_priority_before_demand` | contradicted if resting age ≤ 10 s at every feasible instant (max ≤ EARLY_MS); timing_sensitive if > 10 s is feasible (a late post always is); unidentified if the window is unbounded |
| POST | `touch_join_or_improve` | compatible if at / inside the own-side touch at every pre-post frame; contradicted if behind at all; timing_sensitive if mixed; unidentified without frames |
| POST | `reaction_to_price_change` | compatible if a BTC move (\|r_5s\| ≥ 1 bp) lies in the 5 s before every candidate instant; contradicted if before none; timing_sensitive otherwise |
| POST | `inventory_offset` | compatible if s × position < 0 at every candidate instant (recorded and certain); contradicted if > 0 at all; timing_sensitive if mixed; unidentified if flat throughout |
| KEEP (fill j → j+1) | `retain_after_fill` | observed (A): compatible; timing_sensitive when the longest feasible gap (fill_{j+1}.hi − fill_j.x_lo_hard) ≤ 1,000 ms (no time to react is equally compatible); the adverse move between the fills is reported |
| END | `pull_after_adverse_move` | only when PL is feasible; over the PL decision window: compatible if s × r_5s ≤ −1 bp at every instant, contradicted if at none, timing_sensitive otherwise; the row's status is `unidentified` whenever EX or EC is also feasible (the pull itself is not established) |
| END | `inventory_driven_pull` | as above with s × Δposition over the 5 s before each instant > 0 (inventory built on the order's side) |
| LINK | `replace_or_replenish` | compatible if `sequential_identified`, contradicted if `overlap_identified`, timing_sensitive if both feasible; `unrelated` is always the competing explanation |
| TAKE | `inventory_unwind` | compatible if an actual reduction in every inventory variant; contradicted if add / overshoot in every variant; timing_sensitive if they differ; unidentified if from flat |
| TAKE | `directional` | compatible if s × r_5s ≥ 1 bp at every instant of the match window; contradicted if ≤ −1 bp at every instant; timing_sensitive if mixed; insufficient_comparison if flat throughout |
| TAKE | `after_fill_burst` | compatible if ≥ 3 own maker fills certainly in the 5 s before the window; contradicted if none possibly; timing_sensitive otherwise |

**Contrasts in the declared sample** (descriptive): per address, post cases cross-tabulated by touch category × inventory relation and take
cases by |position| class × BTC alignment, with counts; `overlap_ok` says whether contrasting cells are populated (if not, said so).

## 7. Census chains (all top-3 actions on the 19 BOOK markets; vectorised per market)

Each tool writes write-once under `v2/<tool>/` (a `receipt.json` with inputs' sha256, source sha256, counts, statuses, CPU and VmHWM, plus
CSV tables), imports `pf_mt_common`, reads the other tools only, and takes every market-context feature from `chain_context.py`.
**Outcome selection:** only filled orders are visible; every posting / pull statement is conditional on a later fill and never an
unconditional "they post (pull) when X". Any action-versus-no-action comparison uses a matched control with ≥ 200 draws (seed 20261123,
`numpy.random.Generator(PCG64)`), reporting the observed value, the null mean, p05 / p50 / p95 and one-sided p = (1 + #null ≥ obs) / (1 + n)
(and the lower tail). Denominators (orders, executions, shares, takes, address × day) and statuses accompany every table.

### 7.0 `chain_context.py` — the one definition of market context

| Capability | Rule |
|---|---|
| BTC at x | `pf_mt_decision.btc_feats` unchanged: r_1s, r_5s, r_30s (bps), sig, p_fair, code; row g = ⌊x / 100⌋ × 100 ≤ x (no look-ahead) |
| Book at x | the last usable frame (`pf_mt_lifecycle._book`) with upd_ms ≤ x: best bid / ask tick (unknown when an unknown cell lies above the best known level), spread (ticks), mid (cents), depth at (side, tick), distance from the own-side touch (ticks; + behind, 0 at, − inside), category at / inside / behind / unknown, age, stale (> 1,000 ms) |
| Inventory at x | recorded position (ok legs with settle_ms < x), certain position (legs with hi < x), pending (legs with x_lo_hard < x ≤ hi), all signed in YES shares |
| Time left | tau = (we − x) / 1000 s; phase `pre` (x < ws) / `live` |
| Window evaluation | every 100 ms instant of [lo, hi] (and the D profiles {0, 250, 1000} ms shifting lo down): min, max, category set, `timing_sensitive` when the category is not constant, missing-point count |
| Moves (declared bins) | BTC: \|r_1s\| ≥ 0.5 bp, \|r_5s\| ≥ 1 bp, \|r_30s\| ≥ 2 bp is a move (sign = direction); against = s × r ≤ −threshold, with = ≥ +threshold (s = +1 YES bid / buy, −1 ask / sell); book: a mid or opposite-touch change ≥ 1 tick over 5 s |
| Bins | \|position\|: [0, 50), [50, 200), [200, 1000), ≥ 1000 sh; resting age ≤ 1 s / ≤ 10 s / > 10 s / unbounded; width ≤ 50, ≤ 250, ≤ 1,000, > 1,000 ms |
| Control sampler | seed 20261123, ≥ 200 draws, strata exactly as each chain declares; an empty stratum → the declared fallback, counted |

### 7.1 `chain_post_fill.py` — POST and FILL chains (every top-3 filled order)

- **POST context** over the union posting window (section 4.3), evaluated at every pre-post instant: the order's touch category and
  distance (at / inside / behind by k ticks), spread, depth at its price (= FIFO ahead volume), BTC r_1s / r_5s / r_30s aligned with its side,
  time left, inventory relation (s × position < 0 offsetting / > 0 adding / flat; recorded and certain). Constant → that category;
  otherwise `timing_sensitive` with the category set.
- **POST classes:** `late_post` (window ≤ 1,000 ms: certain); `early_queue_builder_possible` (some feasible instant ≥ 10 s before the first
  print by anyone at its price after that instant, with the order behind the touch or at a level not yet traded since that instant);
  `touch_joiner_possible` / `touch_improver_possible` (at / inside at some instant; `_only` when at every instant). The `P_nomask`
  representative class is reported beside and labelled prior-driven.
- **FILL:** resting-age bounds [0, first hi − post_lo_union]; the queue class at the first fill (section 4.4) and `front_after_partial`
  per later fill; early-resting versus fresh: `fresh_certain` (= late_post) or `early_possible`; what followed: repeated fills (n ≥ 2),
  the next same-side order and its `link_hypotheses` status, opposite-side maker shares of the address within 30 s after each fill (and
  econ `mq`), a take by the address within 30 s after the last fill with its |position| class.
- **Matched comparison** (for any "posted when X"): the posting-window context against width-matched random windows of the same address ×
  market × phase (inside the address's active span: its first to last leg in the market), ≥ 200 draws.

### 7.2 `chain_pull.py` — PULL chain

- **End classes** (never pooled): `exhausted_proven` (S_FIFO set = {exhausted}; grade C), `capacity_tight` (UB ≤ cum + 0.01 in every
  scenario; B), `cancel_possible` (PL feasible, removal window bounded before the last usable frame), `end_censored_possible`,
  `unidentified` (`missing_evidence`). Reported under S_NOPRIO and under S_FIFO separately.
- **Decision window** = [removal_lo − D, removal_hi] for D ∈ {0, 250, 1000} ms. At every instant c: BTC against the order (bid: BTC down; ask:
  BTC up) for r_1s / r_5s / r_30s; the opposite touch moving toward / away over 5 s; the order's distance from its own-side touch; depth at
  its price (queue behind it under FIFO); own fills and the address's Δposition over the 5 s before c (s × Δposition > 0 = built on the
  order's side); time left. Categories all / some / none, timing_sensitive when mixed.
- **Matched control:** primary = not-removed instants: inside the certain-presence span [first hi, last x_lo_hard] of the same address's
  same-side orders, same market and phase, the width-matched window lying inside the span; fallback = random instants in the address's
  active span, same market and phase (counted). ≥ 200 draws. Exhausted / expired / unidentified orders get the same contexts beside,
  never in the cancel-possible group.

### 7.3 `chain_take.py` — TAKE chain and the action-type enumeration

- **Takes:** every top-3 taker record, grouped by `tk_hash`; the |position| classes of section 5 (recorded, certain, simultaneous
  variants); context over the match window and the D profiles: BTC r_1s / r_5s / r_30s aligned with the take, touch, spread, depth at the
  touch on the taken side, price paid versus the touch at the last frame ≤ the window start (ticks through) and versus p_fair
  (s × (p_fair − p), cents), time left, |position| bin, time since the own last maker fill (bounds), fill burst (≥ 3 own maker fills
  certainly in the 5 s before), pending exposure.
- **Controls:** (i) matched no-take instants: same address, market, phase, position sign and |position| bin, no own take within ± 2 s,
  width-matched windows, ≥ 200 draws; (ii) reducing versus adding takes: their class labels permuted within address × market, ≥ 200
  permutations.
- **Action types** (each with denominators by address × day, context summaries, logic-table rows, rival explanations and
  timing_sensitive counts): `post_new_level` (the address's first filled order at that side and tick in the market), `post_same_level`,
  `reprice` and `refill` (`x_links` pairs with their `link_hypotheses` sets and identified subsets), `ladder_overlap` (overlap-certain
  pairs at different ticks), `keep_after_fill` (every non-last fill: observed persistence), `pull_possible` (section 7.2 classes), the take
  classes, `two_sided` (an opposite-side order of the address whose feasible life [post_lo_union, removal_hi_union or market end] overlaps;
  certain when an opposite-side fill window lies inside the order's certain-presence span; the quoted width in ticks).
  **Apparent churn:** cancel-repost pairs are bounded as [0, the number of pairs where `replace` is feasible]; the share that could instead
  be exhaustion / replenishment, an overlapping ladder or timing ambiguity is reported pair by pair; no single cancel / repost count is
  given where histories disagree.

## 8. Coverage and precision bins

`coverage.csv` (order estimator) reports, by address × day × event type (posting, removal, size / end, queue, link, fill match, take
match): total / bounded / unbounded / conflicting, the width bins ≤ 50, > 50–250, > 250–1,000, > 1,000 ms (and unbounded), and the
denominators orders / executions / shares (takes: records / shares). Two scopes: `census` (every top-3 lifecycle row on the 19 BOOK markets;
no book) and `sample` (the detailed cases). Submission coverage (D profiles) is reported separately from venue-effect coverage. The
hard match window is 2,000 ms by construction (always > 1,000 ms); the evidence window is a scenario refinement and labelled so. These are
descriptive precision bins, not tuned thresholds or evidence of skill. **Stop rule:** if most histories cannot distinguish the proposed
mechanisms, the packet says so, names the observable that would resolve them, and does not start a timestamp-refinement round.

## 9. Controls and the own-order instrument check

**Selftest (`tools/order_estimator_selftest.py`, synthetic, exact expected values, each refusal's reason asserted):**
1. the identical-public-tape pair (5-share exhausted vs 50-share partial then an unseen cancel): identical public tapes and identical
   estimator outputs; EX and PL both feasible (`tied`); `P_cmd` = the 5-share history, labelled prior-driven, the alternative shown;
2. positive controls where an extra observation distinguishes them: a sweep through the price (S_FIFO rule 1 ⇒ EX, grade C; S_NOPRIO still
   tied), a capacity-tight two-fill order (UB = cum ⇒ EX in every scenario, grade B), a later fill of the same hash (not exhausted at the
   first fill, grade A);
3. the repost control: a new same-price order after a pull vs a second order resting since before and first filled later, identical tapes →
   `tied`; positive: a shared record (`overlap_identified`) and a posting bound after the parent's removal bound (`sequential_identified`);
   many partial fills of one hash stay one order;
4. overlapping orders cannot reuse capacity: one visible addition is never assigned in full to two orders (`P_nomask` joint rule);
5. gaps / resyncs prove no posting or removal and reset no queue (`incomplete_gap`, `missing_evidence`);
6. same-ms ties (`simultaneous_action`, both evaluations, `timing_sensitive`);
7. the receipt-earlier-than-UUID record is flagged and excluded from every bound (status counted, row kept);
8. inventory overshoot, reversal and full close classified by actual |position|;
9. perturbation: changing BTC or book data after a candidate decision instant leaves that instant's features unchanged; changing them
   before it changes them (falsifier);
10. the sample: deterministic, independent of input row order, market-first round-robin, ≤ 4 per address × day × kind, key columns only;
11. refusals: a forbidden day, write-once, a wrong population, a fills_valued hash mismatch, a two-spelling address;
12. the chain_context bridge on a stub with the documented names (`load_market`, `btc_ret`, `book`, `touch_distance`, `level_size`,
    `position`): key mapping, the POST / END rules end to end, a missing function refused.

**Own-order instrument check (consumed artifacts only):** our linked Nov 8 orders (`own_orders` `link_status` unique / unique_by_time)
through the same hypothesis code on their own lifecycle rows (no depth read: `P_drop` and `P_nomask` not evaluated there). Scored: the true
history (filled → EX, cancelled → PL, expired → PL or EC) is in the feasible set (a miss on a grade-B exclusion is a defect; on a grade-C one
a priority miss), the true size lies in the true history's size range, the union posting window overlaps the true activation
[act_lo_venue_ms, act_hi_venue_ms], and how often each prior's representative equals the truth (own client only: descriptive, never a
calibration or a latency estimate for other makers).

## 10. Outputs (order estimator; write-once, built in a partial directory renamed at the end)

`v2/order_estimator/` (full run) or `v2/order_estimator_pilot_<day>_r<rank>/` (`--pilot DAY RANK`): `cases.csv` (the selection manifest
and every case's keyed windows, statuses and provenance), `manifest.csv` (one row per kind × address × day with its candidate count and
status, and the manifest sha256), `order_ledger.csv` (one row per selected order × hypothesis, and per link pair × relation),
`episodes.csv` (the logic table with episode columns), `episode_summary.csv` (one row per connected episode with its economics),
`coverage.csv`, `census_orders.csv` / `census_links.csv` (the estimator fields of every top-3 lifecycle row and x_links pair),
`contrasts.csv`, `gt_check.csv`, `timelines/` (one CSV per connected episode), `receipt.json` (inputs' sha256, chain_context's per-market
input records and checks, the manifest's sha256, source sha256, git state, counts, statuses, checks, CPU per stage, VmHWM; the pilot adds
the measured projection).

## 11. Reading rules

- Claims are computed predicates with their denominators; no verdict string is printed beside a table.
- A reconstructed feasible set is not a confidence interval; the midpoint, the latest posting bound or the first fill is never the
  command time.
- Venue-state conditions (book at upd_ms, BTC at E_ms) are not what the maker received; they are labelled venue state.
- The later economic outcome of an episode is reported separately from the action's unmeasured counterfactual value.
- No entry / exit training labels are derived from these filled-order examples.

## 12. Resources

USER ceiling for the whole packet: **≤ 0.08 core-hours all-in**, including the **0.010 h** review reserve, inside the 1.85 h task cap
(5.25 h line). Every Python run goes through `predictfun/research/run_light.sh` (tags `mt_oe_*` for the estimator; logs in `logs/`),
VmHWM self-guard 1,450 MiB, a CPU self-guard per run. The estimator's development share is ≤ 2 selftest runs and one pilot on one address-day
(`--pilot`); the pilot writes its measured stage CPU and the projection fixed + per-market × 19 + per-case × 96 into its receipt. The full
run requires the freeze (this contract, the tool, its selftest and `chain_context.py` tracked and unmodified) and is launched by the
integrator only. A projection that does not fit stops for a USER resource decision.

**Pre-freeze record (development, 2026-11-23; no rule changed by it).** Selftest `mt_oe_selftest2`: 74 / 74 (1.47 CPU s; run 1 stopped on a
pandas `DataFrame.flags` name collision, fixed by renaming the column `order_flags`). Pilot `mt_oe_pilot_20261027_r1` (Oct 27, rank 1; 7.49
CPU s, 525 MiB): every stage up to and including the input hashing ran on the real inputs (selection, the census of every top-3 row, the
cases through chain_context, the GT check), then `coverage_rows` refused a table without `clock_bad` / `conflict` columns (the x_links
census) **before any output was written** (no directory created). Fixed in code (absent columns default to false); the remaining stages
(checks, writers, receipt) have not yet executed on real inputs.

## Freeze note (integrator, 2026-11-23, before the full runs)

**USER budget:** the packet runs as 5 parallel tools, with a ceiling raised to 0.08 h and then **0.11 core-hours** all-in (AskUserQuestion). The task cap of 1.85 h is unchanged.

**Tool status at the freeze:**

| Tool | Selftest | Pilot |
|---|---|---|
| `chain_context` | 173 / 173 | — |
| `order_estimator` | 75 / 75 | r2 OK: 14 receipt checks; own-order check 85 covered, 0 defects |
| `chain_take` | 59 / 59 | OK |
| `chain_pull` v2 | 42 / 42 | OK |
| `chain_post_fill` | 251 / 251 | pilot2 reproduces pilot1 on every retained column after the cost cut; takes are join keys to `chain_take` |

**Cross-tool definitions:**
- The burst rule is unified (§5): certain only if the whole window lies in [lo − 5 s, lo).
- `link_hypotheses` is the single repost / ladder definition.

**Output paths:**
- the census tools write `v2/chains/<tool>/`, which differs from §7's `v2/<tool>/`; the layout is accepted;
- `order_estimator` writes `v2/order_estimator/`.

**Pilots:** their outputs are kept on disk; only the small receipts and summaries are committed.
