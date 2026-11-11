# P-2026-005 HANDOFF

## 2026-11-11 08:33Z (session ce68ebe0) — Fix plan, checked against the code: size by add / remove is already built (IncrementalV1); the pull hysteresis is studied (B3) but not in the engine

- Checked (USER "I remember we have already implemented the size adjustment logic?"): pmmmrust main has
  SizeLayering::IncrementalV1 (70b965b, f20fdb1, 2026-11-09; v3 examples' setting; never run live, the 11-08 run predates
  it). The B3 soft pull (handoffs/20261109_soft_pull_results.md; research/pf_soft_pull.py on research/predict-soft-pull-20261109)
  is not in pmmmrust; its next step is B4, the integration with IncrementalV1 and parity.
- USER choices (AskUserQuestion, this session): v4 = pull hysteresis + size by add / remove; code in research + a pmmmrust
  branch (not main, no tag, no deploy), reviewed and deployed by the Tokyo session.
- Next: implement B3 in pmmmrust on a branch (default off, token), with unit tests mirroring the B3 study's controls and
  a parity check against pf_soft_pull.py; then the hand-off.

## 2026-11-11 08:28Z (session ce68ebe0) — B13d diagnosis: the churn is the guard value flickering around the pull line (83.5% of touch withholds) plus same-price size flips; B13c's "not guard decisions" corrected

- Done: read pmmmrust at pfmm-ladder-v3-tokyo-6 (4d65884; read-only, the checkout was clean and unchanged) and attributed
  the touch withholds from the Rust engine's own parity decisions of tokyo-5 (105 windows of 11-07 / 11-08). Report
  handoffs/20261111_b13d_churn_diagnosis.md; scripts research/pf_b13d_withhold_attrib.py, research/pf_b13d_g4_check.py.
- Code path: pf-strategy ladder.rs:413 (withhold when v <= the pull line, every decision), :278 (size 25 / 5 by the PM95
  line), :495 (G4); pf-engine ladder.rs:439 and :479-528 (any change of the wanted price or size cancels and re-places).
- Attribution (6,866 first-minute transitions): pull line 83.5% (53% back within 100 ms), G4 14.7%, other < 2%;
  same-price size flips 2,622. G4 check on 11-08 live cancels: our side lighter in only 4.1%.
- Correction: B13c's lag-50 read of the research guard was one grid step before the triggering crossing; the pulls ARE
  the guard's line crossings (flicker). B13c report and flag annotated in place.
- Next (USER to confirm the spec): a new candidate with (1) a narrow pull hysteresis for resting touch orders (B3: 0.01 c,
  <= 100 ms), (2) size changes by adding / removing orders, (3) optionally a G4 hysteresis; in a new research engine file
  and in pmmmrust behind default-off tokens with parity; then a live run with the A4b capture.
- Watch out: do not edit predictfun/research/pf_step4_ladder.py (the scheduled v3 forward run depends on it).

## 2026-11-11 07:14Z (session ce68ebe0) — B13b / B13c DONE: the pressure model predicts when the top makers take (AUC 0.77) but they do not pull on it; we never hold the queue because our engine cancels at-best orders a median 23-38 ms after they go live

- Done: B13b and B13c (USER: do the features explain the top makers' behaviour; why do they keep the front of the queue).
  research/pf_b13b_top_behaviour.py and research/pf_b13c_queue_pulls.py committed and pushed before their runs (54dc5df9;
  outputs 07:10Z, 07:12Z). Report handoffs/20261111_b13bc_behaviour_queue.md; output live_vs_replay/b13_20261111/, copies
  reviews/b13bc_20261111/. Controls pass; K-checks against B12d / B12e / B1a pass.
- B13b: R102 FEATURES-EXPLAIN-TAKES (EMPH AUC 0.771; BASE 0.725; R104 EMPH-EXPLAINS-BETTER); R101 NO-PULL-SIGNAL (their
  share of fills 29.4% -> 27.8% from low to high adverse pressure); R100 / R103 no pressure-driven leaving (re-hit 57.9% ->
  64.9%). Under adverse pressure top 5 -0.50, ranks 21+ -1.24 c/sh.
- B13c: 12,240 first-minute pulls at the best; order life p50 23 ms (raw log, all minutes: 26 / 38 ms; 52% < 50 ms; > 10%
  cancelled before their ack); 82.5% re-placed at the same price within 1 s (R110 CHURN-HEAVY); queue ahead at activation
  p50 381 sh; 87% of pulls with the research guard ON, after which the flow at our price earned +0.45 c/sh (R112
  PULLS-COST); would-be fills had we stayed 1,631 sh at +0.04 (R113 STAYING-PAYS); R114 GUARD-PULLS-GOOD-ORDERS.
- Next: stop the churn (keep an order that is still at the best with the guard ON); measure with the A4b capture (USER
  D2); use the pressure model to step back from informed takers; churn and slow cancels to the live session.
- Watch out: would-be fills use the shown queue with FIFO and no cancels ahead; re-hits mix staying with the next order's
  arrival; 11-08 is descriptive for B13b.

## 2026-11-11 07:10Z (session ce68ebe0) — B13a DONE: a guard model with 13 engineered pressure features beats the current one on all 9 out-of-sample days (R2 0.18 -> 0.21)

- Done: B13a (USER: train on some days, validate on others, emphasise imbalance and flow). research/pf_b13a_pressure_model.py
  committed and pushed before its run (4b98b043; output 07:08Z). Report handoffs/20261111_b13a_pressure_model.md; output and
  six fold models in data/predict_fun/research/b13_20261111/; copies reviews/b13a_20261111/.
- Design: LightGBM y1000 with the pinned step-5 settings; BASE = the current 32 features, EMPH = + 13 engineered pressure
  features (imbalance changes, thin flag, best-level depletion, Binance activity, interactions); blocked 3-fold CV by day
  over 10-23..10-27, 10-31, 11-01..11-03 (no protected or later day; 11-08 not used).
- Readings: R90 IMPROVES (dR2 +0.027 [+0.023, +0.032], 9/9 days); R91 IMPROVES-UNDER-PRESSURE (+0.025, 9/9); R92
  NOT-ABSORBED (the error's link to imbalance falls by about a quarter). Top new feature: the 250 ms change of Binance's
  best bid / ask imbalance (16.7% of gain).
- Next: B13b (do these features explain the top makers' behaviour, 11-08) and B13c (do our pulls give up queue place and
  good orders); then a guard mapping for EMPH and a test on untouched days (>= 5) before any live use.

## 2026-11-11 06:48Z (session ce68ebe0) — B12f DONE: at the top makers' takes Binance is twice as busy and its top of book leans their way (price still); Predict imbalance adds 0.58 c/sh beyond our guard value; we sit at the very back of heavy levels

- Done: B12f (USER "Yea check Binance activity during those times, and imbalance etc"). research/pf_b12f_activity_imbalance.py
  committed and pushed before its run (6a51ba39; output 06:46Z). Report handoffs/20261111_b12f_activity_imbalance.md;
  output live_vs_replay/b12f_20261111/, copies reviews/b12f_20261111/. Controls pass; event counts equal B12e's.
- Readings: R80 NO-FLOW-LEAD; R81 BN-BOOK-LEAD (Binance top-of-book imbalance their way +0.17 against +0.04 at random);
  R82 BN-ACTIVE (45.7 against 21.6 trades in the second before; updates 426 against 199); R85 IMBALANCE-INCREMENTAL
  (thin-side fills -0.58 c/sh at the same guard value; null p01 -0.18); R86 THIN-OK-WHEN-ON (+0.13 with the guard ON,
  6% of ON volume); R87 NO-QUEUE-SPLIT (heavy levels: back +0.30, front +0.68).
- Ours: median reach on heavy levels 0.93 (field 0.20); heavy-side fills ON -0.14 (26 legs), OFF -2.00 (11 legs).
- Says: the top makers act on book pressure on Binance and Predict before a price tick; our models' feature table has
  these inputs (bn_imb, bn_flow*, bn_ntr1000, pf_imb1/3, pf_dm*), and the guard value under-uses Predict imbalance.
- Next: (1) a declared guard-input test (Binance / Predict book pressure) that needs >= 5 untouched days to validate;
  (2) queue: avoid joining the very back of heavy levels (descriptive here); (3) slow cancels to the live session;
  (4) USER D2 / rebate.
- Watch out: pro rata values rank moments only; one Sunday; three addresses; BBO-only Binance imbalance.

## 2026-11-11 06:01Z (session ce68ebe0) — B12e v2 DONE: Binance's best bid / ask was genuinely quiet (raw stream = research grid); the 05:49Z "coarse grid" correction is reversed; our fills sit mostly on the heavy side and lose there

- Done: B12e v2 (USER "Can you still check Binance data, whether we have that data"). The HF collectors recorded
  Binance BTCUSDT bookTicker for 11-08 (data/mm_hf/raw/bookTicker/BTCUSDT/20261108_HH.csv.gz). research/pf_b12e2_binance_raw.py
  committed and pushed before its run (f0b19fc9; output 05:58Z); report section in handoffs/20261111_b12e_smart_takes.md;
  copies reviews/b12e_20261111/.
- Result: 90.3% of 1 s intervals at random times have an unchanged best-price mid on the raw stream (grid 90.5%): the
  grid is faithful, Binance was quiet. R76 BN-RESOLUTION-WEAK by its rule (it assumed the price moves most seconds), so
  R70v2 / R71v2 / R74v2 are n/a by rule. Descriptive: Binance had moved >= 1 bp their way in the second before 4% of the
  top makers' takes (random 1%), median 0. Their trigger is on Predict for most takes.
- Reversal: B12c v2's NOT-BINANCE and B12e's R70 / R74 stand as statements about Binance's best bid / ask; the 05:49Z
  notes in the B12c report, the B12d summary and the STATUS flag are superseded in place.
- Ours by the book on our side: 72% of our at-best shares are on the heavy side at -0.59 c/sh (37 legs), where every
  other group earns +0.52 to +0.71; thin side 17% at -2.16 (14 legs). Consistent with our place at the back of long
  queues (B12b OURS-BEHIND).
- Next: (1) Binance trade flow and top-of-book sizes at the takes (raw trade / bookTicker, unchecked); (2) the
  imbalance guard-input test (B12f); (3) slow cancels to the live session; (4) USER D2 / rebate.
- Watch out: the HF collectors' 11-08 Binance files are now read by P-2026-005 (recorded under days_read); bookTicker
  only covers the best price.

## 2026-11-11 05:49Z (session ce68ebe0) — B12e DONE: the top makers take the thin side of a lopsided Predict book that has started to move, at moments our own model flags; the Binance grid used is too coarse to judge Binance

- Done: B12e (USER "can you check their take, what happens on Binance or what events happen during then?").
  research/pf_b12e_smart_takes.py committed and pushed before its run (04422a15; output 05:45Z). Report
  handoffs/20261111_b12e_smart_takes.md; output live_vs_replay/b12e_20261111/, copies reviews/b12e_20261111/. Controls pass.
- Results: 2,491 taker orders of makers 48 / 51 / 661 against 12,455 random times. Side hit median 84 sh (random 228),
  other side 312 (~235); Predict mid +0.61 c their way in the second before (random +0.08), +1.14 c 250 ms after, +1.74 c
  1 s after; our model's 1 s prediction points their way at 81% (mean +0.60 c; R72 OUR-MODEL-SEES); our guard value of
  the side hit < 0 at 90%; 55% within 250 ms of another of their orders. R70 NO-BINANCE-LEAD, R74 NO-BINANCE-FOLLOW,
  R73 THIN-SIDE.
- Correction: the Binance grid (step5 rows_rl_d2.bn_mid) is coarse (90% of 1 s intervals unchanged; median 5 distinct
  values per first minute). R70 / R74 here and B12c v2's NOT-BINANCE read as not shown; correction notes added to the
  B12c report and the B12d summary table; flag model_or_speed_b12c updated.
- Says: avoid their flow by not resting on the thin side of a lopsided book when our model points against it, and by
  pulling that side right after the first smart hit; our model already flags most of these moments.
- Next: (1) a guard-input test with book imbalance + Predict's recent move (public tape); (2) a sub-second Binance source
  for the cohort (or the live feed via the A4b capture, USER D2); (3) slow-cancel issue to the live session; (4) USER
  rebate / rewards.
- Watch out: three addresses, one Sunday; part of the move after is their own take clearing the level; the one-sided
  null covers only the upward differences.

## 2026-11-11 04:59Z (session ce68ebe0) — B12d DONE: the top makers stay after being hit; a quarter of their edge is their own taking; the rest is not visible on the public tape

- Done: B12d (USER "Run it, and give me a clear explanation"). research/pf_b12d_bursts.py committed and pushed before
  its run (7b70bfa1; output 04:54Z); post hoc research/pf_b12d_takers.py (04:57Z). Report handoffs/20261111_b12d_bursts.md
  (with a summary table of B12-B12d); output live_vs_replay/b12d_20261111/, copies reviews/b12d_20261111/. Controls pass.
- Readings: R60 SAME-POSITION (first in a run 46.9% against 43.8%; X_RUN 0.08); R61 BOOK-NOT-ENOUGH (X_THIN 0.12);
  R62 STAY-AFTER-HIT (re-hit after an absorbed fill 59.5% against 38.4%, null p99 +4.4 pp); R63 SAME-HOLD (+5.0 pp);
  R64 HOLD-THE-PRICE (from R62; the holding itself is small).
- Post hoc: makers 48, 51, 661 of the top 5 also take (1.1-1.9% of taker volume each); their taker orders cost the
  makers hit -1.23 c/sh at 1 s (others' +0.08), mostly ranks 21+; without them the gap is +0.16 instead of +0.21.
- Says: they do not pull after hits (they stay, at no cost; staying costs the others); the losses sit in stress
  moments (after the level thinned: ranks 6+ -1.04, top 5 -0.31; ours -5.83 on 21% of our at-best shares); at the best
  ranks 6-20 match the top 5, the losers are ranks 21+ and us. Most of the gap is not visible on the public tape
  (cancels are anonymous).
- Next: (1) B1 / USER D2: our reaction to Predict book changes needs the A4b capture. (2) Guard inputs from Predict's
  own book and takers (level thinning, informed takers hitting), not Binance. (3) Slow-cancel issue to the live
  session. (4) USER rebate / rewards.
- Watch out: re-hit cannot tell a cancel from an order used up exactly (the top makers' larger orders raise their
  rate a little by themselves); the taker check uses the 69% of legs with an unambiguous join; our rows are 28 legs.

## 2026-11-11 04:28Z (session ce68ebe0) — B12c DONE: not a better trend model and not faster reaction to Binance; the top makers' edge is Predict-local and earned within 250 ms

- Done: B12c (USER "Can you check if they have a better model, or they just react faster"). v1
  research/pf_b12c_model_or_speed.py (8bc18b8e, committed before its run; output 04:19Z): instrument = our guard
  value; v2 research/pf_b12c2_binance.py (dc8dde4e, committed before its run after v1's result; output 04:25Z):
  instrument = the Binance path, plus a model-free split of when the gap builds. Report
  handoffs/20261111_b12c_model_or_speed.md; output live_vs_replay/b12c_20261111/, copies reviews/b12c_20261111/.
  Controls pass (planted speed / model / visible / later / other worlds read correctly); K0 / K1 reproduce B12 / B12b.
- Readings: v1 R53 INSTRUMENT-WEAK (our value 2 s later accounts for none of the 1 s gap; R50 / R51 n/a). v2 R55
  NOT-BINANCE (the Binance path up to 2 s after the trade accounts for 22% of the 1 s gap; residual +0.165 c/sh, null
  p99 +0.040) -> R57 OTHER-INFORMATION.
- Model-free: the gap is earned in the first 250 ms after the report (+0.196 of +0.211 c/sh at 1 s); 0.25-5 s adds
  nothing (-0.03). Their fills come after Binance moves no better than everyone's.
- Says: not a trend model, not faster reaction to Binance; Predict-local and immediate. Polymarket unlikely (research
  session lead-lag 2026-11-10: Predict leads by 100-200 ms; PM cuts 1 s error 0.4%). Untested candidates: they react to
  Predict's own trades / book (pull after the first taker order of a burst); their own size / refills hold the mid.
- Next: (1) B12d (proposed): taker-order bursts and refills on the public tape. (2) Slow-cancel issue to the Tokyo /
  Dublin session (B12b). (3) USER D2 (B1). (4) USER rebate / rewards. Guard work (B7) should use Predict's own recent
  trades and book, not Binance.
- Watch out: at 5 s the Binance path up to +2 s accounts for 59% of the smaller 5 s gap (descriptive only, beside a 1-5
  s mid split near zero); report clock ~0.2 s after the match; coarse cells bias X down.

## 2026-11-11 03:21Z (session ce68ebe0) — B12b DONE: not queue place; the top makers handle moves in progress better; our first-minute loss is the venue's slow cancels

- Done: B12b (USER "Yes find out the reason"). research/pf_b12b_reason.py committed and pushed before its run
  (c882da77; output 03:10Z); post hoc research/pf_b12b_posthoc.py (03:14Z-03:17Z). Report handoffs/20261111_b12b_reason.md;
  output live_vs_replay/b12b_20261111/, copies in reviews/b12b_20261111/ (SHA256SUMS). Controls pass; K0 reproduces
  the committed b12.json exactly. One consumed Sunday, point estimates.
- Readings: R42 FIFO-LIKE (94.6% of <= 5 sh taker levels fill one maker); R40 NOT-QUEUE (within guard x reach the top
  5 still +0.23 c/sh, null p99 +0.10; reach mix -0.02); R41 SAME-PLACE (median reach 0.213 against 0.181),
  OURS-BEHIND (0.300, 28 legs); R43 n/a (65 clean flips; the guard flips ON -> OFF ~every 2 s per side); R44 MIXED;
  R45 OWN-SIGNAL (exits n/a).
- Post hoc: our guard's value and the time in the window explain none of the top-5 gap (X -0.01); 73% of it sits
  while our guard's value is falling (a move under way): -0.14 against -0.56 c/sh. At the best, ranks 6-20 do as well
  as the top 5; ranks 21+ lose at the same queue places.
- Ours: 11 of our 25 guard-OFF fills came on orders whose cancel we had sent 0.35-4.1 s before, while the venue took
  0.6-5 s to confirm it: -4.88 c/sh, -$4.16 a day = our whole first-minute loss, all 49-60 s into the window (8
  windows; 85% in three legs). The other 14 net 0.00. Live-wide: 86 of 30,219 cancels took > 500 ms, 1.29% of those
  sent 50-60 s in against 0.06-0.20% elsewhere; no throttle or reject recorded.
- Changes: the B12 addendum's "pull timing first" is superseded (pointer added there): the pulls were sent; the loss is
  the venue's slow cancels. The top makers' edge is speed / signal for moves in progress, not queue or our guard's
  states.
- Next: (1) hand the slow-cancel issue to the Tokyo / Dublin session (report section "Issue for the live session":
  kernel cancel stages; the same count on 11-07 tokyo-5). (2) USER D2: live run with the A4b capture (B1). (3) USER:
  rebate and rewards. (4) Signal / latency work aimed at moves under way (B7 reframed).
- Watch out: post hoc splits are descriptive; the slow-cancel loss is 11 legs in 8 windows; 6 of the 25 OFF legs have
  research trade times outside the audit's feasible interval; B12's guard lookup reads a NaN prediction as OFF (23 of
  72,913 at-best legs, none of the top 5's or ours).

## 2026-11-11 02:38Z (session ce68ebe0) — B12 addendum: our first-minute loss is the fills taken after the guard turns OFF (no new run)

- Done: answered the USER's question (what separates our fills from the top makers' at the same price; are their
  profits from layered quotes) from the committed B12 output. research/pf_b12_qa_derive.py (arithmetic on b12.json /
  b12_posthoc.json; output reviews/b12_20261110/b12_qa_derive.txt); addendum in handoffs/20261110_b12_profitable_makers.md.
  Post hoc, descriptive; 86 own legs, one day, no null.
- Findings: (1) at the best with the guard OFF, 25 own legs lose -1.98 c/sh at 1 s, -$4.16 a day = 99% of our
  first-minute -$4.18; the rest nets to about zero. (2) With the guard ON we make -0.25 against the top 5's +0.94 c/sh
  ($4.72 a day at our ON volume); 39.2% of our at-best shares come from trades through the level, against 14.5% for the
  top 5 and 17.0% for ranks 6 and below (consistent with a back-of-queue place; not proven). (3) 66% of the top 5's
  dollars are behind the best, but depth alone loses (ranks 21 and below), and our one-tick-back fills (v3's rung, a
  third of our shares) are near the top 5 (-0.01 against +0.24 c/sh, 27 legs).
- Next: pull timing before queue place or a deeper layer. (1) USER D2: live run with the A4b capture (B1). (2) Proposed,
  not run: join the 25 OFF legs to live's order paths (B1a) and count those after our cancel request. (3) B12b.
  (4) USER: rebate and rewards on the account.
- Watch out: research trade times for our legs are off in both directions (B1a); the research guard is not the live
  engine's guard; one consumed Sunday.

## 2026-11-10 16:50Z (session ce68ebe0) — B12 DONE: the top makers' edge is not in states our guard marks; within a state it is the move after their fills, two thirds behind the best

- Done: B12 (strategy review item 9; USER "Then we check the next"). research/pf_b12_makers.py committed and pushed
  before its first run (d5f68cf2); output live_vs_replay/b12_20261110/ (16:43Z), copies in reviews/b12_20261110/; post
  hoc spread / move split research/pf_b12_posthoc.py, written after reading the output. Report
  handoffs/20261110_b12_profitable_makers.md. Public tape of the 220 consumed 11-08 windows, first minute; point
  estimates. Controls passed (7 synthetic; K1 tape statuses equal the replay's; K2 own id; K3 line).
- Input checks: `signer` is the maker's address (on chain maker == signer on all 19,671 orders of the 10-23 14:00-14:15
  slice; 11,492 of 11,492 REST legs matched); our account is id 200 (99 legs, 849.867 sh). Maker fee in the public
  match records is 0 on all 87,155 legs read (10-23 14h; 11-08 03h, 15h): no rebate inside the match record.
- Results: top 5 +0.36 c/sh at 1 s on 462,118 sh a day (+$1,676 a day), ranks 6-20 +0.19, rest -0.33, ours -0.44 (86
  legs). At the best the top 5's gap over ranks 6+ (+0.20 c/sh) is within the same guard state and trade type (+0.19;
  null p99 +0.09), not mix (+0.01); post hoc it is the move after the fill (+0.18), not the spread (+0.01). They trade
  86% of their at-best volume while our guard reads OFF, like everyone. Two thirds of their dollars are behind the best
  (+1.00 c/sh two or more ticks back; makers below rank 20 lose there). Not faster to new levels (ratio 0.97) or to
  levels born as best (0.90). Our fills have the worst move in both guard states (ON -0.76 c/sh against the field's
  +0.07 to +0.42).
- Readings: R30 TOP-EARNS; R31 WITHIN-STATE at lags 0 / 50 / 250; R32 NO-AVOIDANCE, top-5 OFF value about 0 (its sign
  flips between lags); R33 DEPTH-FED; R34 / R35 PROPORTIONAL; R36 GUARD-CLOCK-SENSITIVE (only R32's sign); N1
  WITHIN-REAL.
- Changes: guard-state selection and speed to new levels are excluded as the top makers' edge. What remains is queue
  place, cancel timing or information, which the public tape cannot separate. A guard refit (B7) is not the first
  lever; B1 (live with A4b) is. Depth pays only with its cancel behaviour (B5, after B1). D3: a rebate can only be seen
  on the account. Strategy review revision 5 points 3.1, item 9 and section 8 to the report.
- Next: (1) USER D2: a live run with the A4b capture for B1. (2) USER: rebate and rewards on the account. (3) B12b
  (proposed): within the same cell, split the top 5's and the field's fills by taker-order size and queue-ahead
  estimate (queue place against cancel timing). (4) B11 after B1. (5) C1 / C2 as declared.
- Watch out: one consumed Sunday; trade times are the replay's; queue position inside a level is not observed.

## 2026-11-10 13:24Z (session ce68ebe0) — B1a DONE: the replay's fill model does not reproduce live's fills order by order; no constant trade-time shift fits

- Done: B1a (USER "Yes"; specification 04:56Z). research/pf_b1a_live_orders.py: v1 committed before its first run
  (7ef3de6e), v2 before its run (1a253a45; identity at the taker order instead of the maker leg row, after v1's post
  hoc check found 33 of 59 uncredited own legs with a replay fill from the same taker order). Report
  handoffs/20261110_b1a_live_orders.md; output live_vs_replay/b1a_20261110/ (v1 kept as b1a_v1.*), copies in
  reviews/b1a_20261110/. 30,221 accepted live orders of tokyo-6 + basis-tokyo-1 in the 220 consumed 11-08 windows
  (849.867 sh filled); tokyo-5 not covered (no research tape). Point estimates only. Controls: 2,000 fuzz cases equal
  the committed replay (0 differences), 4 planted errors caught, tape statuses equal the review's, the replay's trade
  time equals the audit's engine clock on all 92 eligible own legs.
- Live's own fills (92 legs, 796.9 sh): -0.25 c/sh at 1 s, -0.01 at 5 s; orders under 100 ms old +0.05, 100-500 ms
  +0.30, 500 ms or older -0.94.
- Readings: R24 OVER-CREDITS (replay / live 1.47-1.96 under M and L, 13.46 under U); R25 FAILS (precision 0.03-0.25);
  R26 YOUNG-PHANTOM (orders under 100 ms: 8-12% of the replay's shares are live's trades, 47% at 500 ms or more; null
  p99 +0.273 against +0.353); R27 CONSTANT-250-CONTRADICTED (41.2% of live's fill shares have a trade time within
  250 ms of our decision); R28 per-share value MATCHES at M 0 / M 50 / L 50 (-0.37 / -0.66 / -0.70 against live
  -0.25), UNDERSTATES at M 250 (-2.36), OVERSTATES at U 50; R29 PEND-OVERCREDITED (546.5 against 218.0 sh).
- Also: 19 own legs (162.5 sh) were filled live with less volume at our price than the replay's queue ahead (the
  shown-size queue is contradicted for a fifth of live's fills). The tape's trade time is before our decision for 11%
  of our fill shares and after the order left for 20%. The same orders' replay value changes sign with the order path
  (ack to cancel ack -0.66, decision to cancel request +0.66 c/sh at 1 s).
- Changes: no replay number in the review or round i is validated at the order level; the birth-step candidate (6.5)
  is made of young orders and cannot be ranked by replay. Strategy review revision 4 (13:22Z) points items 2, 3, 9 to
  this report and to B11 / B12. Backlog B1a: DONE.
- Next: (1) USER D2: a live run with the A4b wallet capture (new Tokyo tag from pmmmrust main 79a05d3); B1 needs its
  exact match clocks, and no replay can rank candidates before that. (2) USER: rebate and rewards on the account.
  (3) B12, what the profitable makers do (public tape; does not use our fill model). (4) B11 after B1. (5) C1 / C2 as
  declared.
- Watch out: one consumed Sunday, 86 filled orders; the research book contains live's own orders; the venue's matching
  rule is unknown; 2,722 untimed legs excluded.

## 2026-11-10 10:05Z (session ce68ebe0) — strategy review, revision 3: queue priority by early entry NOT found; the fill clock decides the sign

- Done: diagnostic research/pf_review_level_birth.py (committed before its first run, 8f448870; post hoc section added
  09:46Z, marked). One 5-sh order per side, 220 windows of 11-08 (440 window-sides, 0.764 window-days), first minute;
  consumed day; point estimates only (G < 5). Output as of 09:47Z: level_birth.{json,txt} in the 11-08 cohort directory
  under live_vs_replay/review_20261110/, copies in reviews/strategy_review_20261110/. Report:
  handoffs/20261110_strategy_review.md revision 3, section 6.5. No strategy code, frozen artifact or receipt changed.
- Controls: the join lane with v3's lines equals the queue replay on 2,200 window-side x config cells (0 differ) and
  reproduces its committed rows to 1e-9; with a 30 ms cancel latency 40 of 40 window-sides differ; 11 planted errors
  caught.
- New levels: 26.09 per side-minute are born as the best. Shown size p50 52 sh at birth, 155 sh at +100 ms, 228 sh at
  +200 ms; life p50 293 ms. Our order goes live behind p50 229 sh (no guard) to 423 sh (PM90); the v3 join 363 sh.
- Declared readings: R15 B-none BIRTH-COSTS, B-g90 MIXED; R16 GUARD-ADDS (+1.84 / +1.60 / +0.14 c/sh at 1 s);
  R17 RESIDENCE-COSTS (6 of 6 cells); R18 R-none RUNG-COSTS, R-g90 MIXED; R19 I-none IMPROVE-COSTS, I-g90 n/a (46
  fills); R20 B-none QUEUE-DECISIVE, R-none QUEUE-ROBUST; R21 PRIORITY-NOT-SHOWN; R22 MORE-TWO-SIDED only for the
  always-on arms, which lose 0.93 to 1.74 c/sh at 1 s; R23 no arm above the v3 join in all 6 $/day cells (B-g90 4).
- Without a guard every position loses: 40 of 40 cells negative, -0.84 to -2.61 c/sh; with no queue ahead -0.96 to
  +0.30 at 1 s.
- Fill clock: at PM90 15 of 16 cells are positive at trade-time shifts of 0 and 50 ms; at 250 ms all 32 cells of the 16
  arms are negative (best -0.54 c/sh), also with a rebate of 0.25 c/sh.
- Candidate (declared arm B-g90, reading MIXED; the split of the join is post hoc): the touch order enters only at the
  grid step where its price becomes the best. 89 fills, 301.7 sh: +0.99 / +0.58 / -1.10 c/sh at 1 s for shifts 0 /
  50 / 250 ms, +1.10 at 5 s (M 50). The join's other entries, 185 fills, 659.1 sh: -0.09 / -0.33 / -1.94. $/day at 1 s
  +4.73 / +2.30 / -3.92 against +4.01 / -0.58 / -27.75 for the join.
- Next (report section 7): (1) B1a, then B1: the clock decides every sign; (2) USER: rebate and rewards on the account
  (with 0.19-0.25 c/sh the PM50 line fills 6.1 times the shares at about zero trading edge at 1 s); (3) sections 3.2,
  6.2, 6.3 and 6.5 on 11-03 and 11-07; (4) birth-step entry in the full engine, then at least 5 untouched UTC days;
  (5) guard refit with the level's state as an input (B7); (6) what the profitable makers do (the timed legs carry
  `signer`; to be confirmed as the maker's address); (7) reducing side (B9 follow-up); (8) C1 / C2 as declared.
- Watch out: single 5-sh orders; Binance guards, CAP10 / RO60, size-up, throttle and inventory are not modelled; lanes
  R and I are one order alone, not v3's ladder; the public book does not contain our order; the post hoc rows have
  19-45 fills; nothing here is validated. Do not build an early-entry, improve or always-on rung rule on this evidence.
- Addendum 12:04Z: backlog brought in line with the review. handoffs/20261109_TODO.md: B5, B6, B7 and D3 updated;
  B11 (sections 3.2, 6.2, 6.3, 6.5 on 11-03 and 11-07) and B12 (what the profitable makers do) added. STATUS.yml:
  T5 note (rewards and rebate on the account) and days_read 20261108.

## 2026-11-10 08:51Z (session ce68ebe0) — strategy review, revision 2: two-line touch gate NOT supported at the order level

- Done: read-only review of pm_ladder_v3 for the user (logic, live vs replay, pull buffer, optimization). Report:
  handoffs/20261110_strategy_review.md (revision 2). Six read-only diagnostics research/pf_review_*.py; outputs in the
  11-08 cohort directory under live_vs_replay/review_20261110/, copies in reviews/strategy_review_20261110/. No strategy
  code, frozen artifact or receipt changed. Population: 220 windows of 11-08, first minute; consumed day; point
  estimates only (G < 5).
- Guard (as of 06:26Z): it separates flow before the trade, all makers pro rata: gate on +0.42 c/sh at 1 s against
  -0.03 off, read 250 ms before the trade report (41,471 touch trades, 640,523 sh).
- What runs has ONE pull line per side (frozen v3, pmmmrust main 79a05d3). The pull buffer (B3) is a two-line rule in
  form: band 0.01 c, at most 100 ms, touch order of up to 5 sh; mechanics study only, not in the live code.
- Wide band, pro rata (gate_bands.txt, 06:44Z): exit at PM50 or -0.10 c raises spell p50 from 100 ms to 800-1,100 ms;
  band flow +0.24 to +0.29 c/sh at 1 s. Tail (v at least 0.15 c below the line): -0.44 / -0.79 / -1.75 at 1 / 5 / 60 s.
- Order level (queue_reach.txt, 08:24Z): one 5-sh touch order per side joining the back of the queue (p50 362-391 sh
  ahead), replay clock and fill rule, trade time shifted 50 ms. $/day at 1 s for v3 / buffer / exit PM50 / exit -0.10:
  -0.58 / -0.86 / -6.81 / -11.09; with v3's own price rule +0.84 / +0.81 / -1.77 / -3.59. Fills taken in the band:
  -0.37 to -0.51 c/sh at 1 s. Both wide-band arms are below v3 at trade-time shifts 0 / 50 / 250 ms, at 1 s and 5 s.
- Queue position decides the sign: exit -0.10 arm at 1 s -0.43 c/sh with no credit for cancels ahead, -0.24 under the
  replay's rule, +0.73 at the front (upper bound, not a runnable arm). The buffer places 21% fewer orders at about the
  same value: churn control only.
- The 06:27Z recommendation (two-line gate for the touch order) is withdrawn in the report.
- Next (report section 7): (1) USER: rebate and liquidity rewards on the account (T5, D3); (2) B1a, then B1, and report
  replay results split by order age; (3) the same tables on 11-03 and 11-07; (4) queue-priority candidates at the order
  level first (new-level entry / improve; rungs behind the best), starting with who creates new best levels and how
  much size is ahead 100 ms after a level is born; (5) guard refit on the value of an order we can place (B7);
  (6) reducing side: line, legal price and minimum size together (B9 follow-up); (7) C1 / C2 as declared.
- Watch out: every level in the order-level test moves with the assumed trade time (v3 arm +0.31 / -0.05 / -1.75 c/sh
  at 1 s for shifts 0 / 50 / 250 ms); the fill rule is not validated (B1a); Binance guards, CAP10 / RO60, size-up,
  rung and throttle are not modelled; 60 s and settlement values do not separate the arms. Do not build a band token
  into the engine on this evidence.

## 2026-11-10 04:56Z (session 62f3ce33) — repo/disk cleanup done; B1a approved, not started; live not running

**State**
- pmmmrust `main` = GitHub at 79a05d3: A1 (366d419), A2 (70b965b + f20fdb1 incremental), A6 (c64fc4e), A3 (c6c0917),
  A4b (7fd6889) + four review rounds. No tag, not deployed. Local fix/pfmm-*-20261109 branches deleted (on GitHub, merged).
- ctaNew `mm-research` had been split from GitHub since 2026-09-11 (P-2026-003 seats). Joined by merge 335d268a (6
  conflicts kept the local side; GitHub's guards in live/pm_research/de_forward_value_day.py, c7ce57ea / dd1c3d85 /
  3c2bacc7, are in history only), pushed. Leftover files and the other research session's notes committed as found
  (latest c8f7757a). Local `main` moved to origin/main (its June commits are in mm-research). fv-pnl-integration's
  working tree committed as found (382a7cd0, local; it has no remote).
- **USER 2026-11-10: live is not running now.** A restart should be a new Tokyo tag from 79a05d3 + re-rendered config.
  First run: wallet_events.jsonl.gz session marks (`live` early, never `not_live`). B1 needs that capture; runs on
  11-09..11-13 stay unread until run_forward_v3.sh (C2, after 11-14 02:00Z).

**Cleanup** (USER: make the repo clean / remove all stale / clear useless only). Free disk 437 -> 547 GiB. Every
removal (292) with where its commits or bytes live: ~/cleanup-archives/git-cleanup-20261110/ (removed.jsonl,
branches.jsonl, summary.json incl. restore steps and the derived_rule; tar archives of uncommitted files and evidence;
two bundles for nested-clone commits; systemd-units/).
- Removed: all stale worktrees (ctaNew September seats / r942-944 / of-model / workflow / gutted /tmp/codex-*;
  fv-pnl-integration's 73, 68 locked: D1 done 10-23, P-2026-003 stopped 10-26; 12 HEADs kept under
  refs/archive/worktree-cleanup-20261110/*; pmmmrust review, Tokyo 11-07/08 and /tmp/pf-*-parity trees; the research
  session's 15 finished /tmp/cta-* trees, branches kept). review-exports 10-25/26 folders (evidence archived); the three
  pmmm-* timers stopped + disabled (the efficiency monitor's build FAILED on every revision since 10-28, 4a7023ae); 45
  dead or stopped user unit files; the locked validation study's raw/ copy (all 1,143 files byte-identical to
  data/pm_5min/raw); 11 unreferenced stale data/pm_5min/derived entries (13 MB).
- Kept: data/pm_5min/derived (43 GiB used within 14 d, ~210 GiB named by records), data/ml, the live tapes,
  pmmmrust/target/debug, review-exports 11-08/09 folders (research reads live-samewindow*), tokyo-live-data-20261109,
  the R-944-stopped pipeline units (pm-lane-health watches two of them), the research session's two current worktrees.

**Answers given (no code change)**
- D1: v3's RO60 caps each new reducing order at |known net| at decision time but does not count other live or
  cancel-pending reducing orders (known net also lags fills ~100 ms). Replay flips 4 / 3 of 2,592 windows (J / X),
  <= 5.04 sh, already inside v3's numbers. Recommended: a description-only correcting receipt; a hard guard = v4.
- B3 buffer: not in live code. Replay only (220 consumed 11-08 windows, no fills): post-side re-entry delays lose
  coverage (50 ms: requests -13.8%, wanted-price time -11.2%); pull-side 0.01 c band, 100 ms hold: requests -7.0%,
  wanted-price time +2.7%, off-target time +34%; native 50 ms: requests -3.7..-5.4%, exposure +2.6..+4.1%. No P&L claim.
- B1: not calibrated. The 11-08 audit (fill-clock, trade-timing repair) found no constant offset fits and no exact
  match clocks for our fills (99 legs, 1 s intervals), so the corrected replay refuses the historical tape.

**B1a (USER "Yes", 2026-11-10) — NOT STARTED; the next session runs it.** A historical reference, not a calibration.
- Population: live order actions of tokyo-5 (run 1794071211506, pmmm-data/review-exports/tokyo5-import-20261108),
  tokyo-6 (1794108465172) and basis-tokyo-1 (1794147344737) in pmmm-data/tokyo-live-data-20261109/tokyo-live-20261109.
  Kernel exports for tokyo-6 / basis: review-exports/live-samewindow-20261108-kernel (tokyo-5: check for an export).
  Cut at 2026-11-08 23:00Z (pf_live_vs_replay.CUTOFF_MS) so hour padding never reaches 11-09; boundary windows are an
  excluded status. 11-07 / 11-08 are consumed (recorded); never read 11-04..11-06 (C1) or 11-09..11-13 (C2).
- Parser: pf_live_vs_replay.kernel() (per cid: place / signed / accepted / cancel stages / fills, second-truncated
  match_time). Fill model under test: pf_step4.py fill model v2 (queue ahead = displayed size at join, capped by every
  frame strictly before a trade; trades at our price consume the queue then us; a sweep clears it; fills known 20 ms
  later).
- Measures: (1) per order, replay-filled vs live-filled by exact identity, split by order age, queue ahead at join and
  cancel-pending; (2) each fill's bounds [place, notice receipt]; (3) the notice lag after the public pending increase
  (A4: p50 435 ms) vs the replay's 100 ms; (4) late fills on cancel-pending orders, live vs replay; (5) a queue fill
  model fitted on outcomes (displayed size ahead at join vs volume matched at our price while resting).
- Rules: declare comparisons, null / controls and minimum sample before results; rows = order actions; exclusions are
  statuses; G < 5 days -> point estimates only; a positive control (planted fill recovered) and a known-bad input
  refused. Do NOT fit a delay or minimum resting age (fill-clock audit). Jobs via run_light.sh, one at a time.
- Prior work to reuse: branches research/predict-fill-clock-20261109 (reviews/fill_clock_20261109) and
  fix/predict-trade-timing-20261109 (pf_trade_tape full identities); research/predict-fill-calibration-20261110
  (pf_wallet_calibration.py, the A4b ingestion for later).

**Next:** (1) B1a. (2) C1 when the slot can be held ~2-3 h (free since 04:29Z). (3) USER: deploy / restart live.
(4) USER: D1, D3.

## 2026-11-10 04:29Z — inventory-cost experiments COMPLETE; no strategy promoted

- Nine consumed days, 2592 markets/arm/J or X, all18 main replays + Nov3 X stress verified. No new policy preserves
  both +1s value and common-stop mark while reducing terminal inventory versus baseline. Marginal entry cuts terminal
  inventory5.4/8.6% and square-time14.0/20.3% (J/X), but loses15.2/16.8% of +1s value. J maximum rises75->83.74 shares.
- Settlement gains for marginal entry (+$79.85/98.36) are later drift/carry, with day intervals including zero.
  Soft/budget exits recover a little inventory from late_reserved but remain worse than original baseline on the objective.
  Budget exit fills only$0.798/0.668 of $91.82/80.23 committed lifetime allowance; committed allowance is not cash spent.
- Neutral-state intersections (post hoc) show off-touch alone blocks1.38% of inventory time, score alone28.3/28.4%,
  score+off-touch27.96/27.91%; remaining41.1/41.0% have other blockers. Removing one gate alone rarely solves joint constraints.
- Report-only corrections preserve replay outputs: v3 read-only checks78d7a610, v4 explicit pandas flags column dbb96f30,
  declaration daca5b2b. Final receipt_v4.json + verification_v4.json; independent90 arm-day inventory/accounting checks,
  33 policy controls, reporting falsifiers and joint audit controls pass. Original failures/declarations retained.
- Findings: handoffs/20261110_inventory_cost_results.md. Branch research/predict-inventory-design-20261110.
  All jobs finished04:24:32Z, v2 data139MiB; slot free (recheck). Protected Nov9–13 unread; live/native/frozen code unchanged.
- Next research focus: preserve entries; joint exit feasibility and outstanding-cost reservation with defensible release,
  separately declared/tested. Current settings not adopted. B1 actual own-fill/queue calibration remains open.

## 2026-11-10 04:16Z — all inventory-cost main replays verified; stress/report pending

- All18 main day/timing replays pass (9 consumed dates, J/X, 5 arms, 2592 markets/arm/timing).
  Nov3 X slower-feedback stress running. No economic aggregate inspected; protected data remain unread.
- Source review found the original aggregator calls a verifier that exclusively creates checks.json a second time.
  Reporting-only adapter78d7a610 compares recomputed payloads/hashes read-only, ignoring only as_of; includes four
  known-bad refusal controls. Original source/artifacts stay unchanged. After the runner's expected report refusal:
  declare pf_inventory_cost_reporting --declare, commit its declaration, run that module under run_light, then
  pf_inventory_cost_verify. Compact reporting/verification receipts will be v3. Slot remains occupied.

## 2026-11-10 03:36Z — inventory-cost J replay complete; X running

- All nine consumed J dates verified, 2,592 markets per arm. Baseline and late-reservation control exactly
  reproduce prior artifacts; exit variants preserve pre60 requests/fills. No late reverse-opening or concession
  budget violation. X dates, Nov3 stress, report and independent saved-artifact verification remain.
- Frozen sources/correction and supplemental verifier pushed through deb4ad6e on
  research/predict-inventory-design-20261110. Source parameters unchanged. No completed cross-timing economic result.
- v2 data/logs retained; research slot occupied. Protected Nov9–13 remain unread; live/native code unchanged.

## 2026-11-10 02:57Z — inventory-cost audit correction; v2 RUNNING

- First full-day audit refused a quote/submission feasibility mismatch. Raw quote notional and lot-rounded order
  notional are now separate predicates; regression control included. Policies and numeric settings unchanged.
- Source542bb9e4; declaration42dfcadf, superseding d5f43299. Original partial run/logs retained. New destination
  data/predict_fun/research/inventory_cost_20261110_v2.33 controls and exact8-market baseline/control smoke pass.
- Full9 consumed days x J/X then Nov3 stress running serially; no completed economic comparison yet. Slot occupied.

## 2026-11-10 02:49Z — authorized inventory-cost experiment RUNNING

- User approved trying the reviewed methods. Source cc109212; pre-outcome declaration d5f43299.
  Three fixed candidates: narrow reducing-side score relaxation, separate one-tick concession with $0.25/window
  lifetime submission budget, and separate marginal inventory-radius score premium on entries. No parameter search.
- 32 synthetic controls and eight-market smoke pass: baseline/control exact, exit pre60 fills/requests exact,
  cash/carry/provenance and committed-concession checks pass. Full9 consumed days x J/X running serially,
  then Nov3 slower-create/fill-notice stress. No protected Nov9–13 read or live/frozen change.
- Sources/runner: pf_inventory_cost_*.py and run_inventory_cost.sh, branch research/predict-inventory-design-20261110,
  checkout /tmp/cta-inventory-design-20261110. Data/logs inventory_cost_20261110. Research slot occupied.
- Neutral baseline exit-feasibility episodes cover60–300s; intervals are decision-sampled states, not verified
  continuous queue observations. Historical single-order simulator remains uncalibrated and differs from native A2.
  No economic conclusion yet. B1 capture/calibration still pending; budget bounds concessions only, not total losses.

## 2026-11-10 02:35Z — inventory/capture design review complete; next audit specified

- User asked what can improve spread capture and inventory while new live calibration data is pending. Reviewed native
  main79a05d3, frozen v3 and prior receipts. Same PM90 gate applies to entry/exit; CAP10 is a known-net adding trigger,
  not an aggregate bound; RO quantity plus XMIN1 can make small residuals impossible to hedge through complement buys.
- Prior unconditional RX5 relaxation, small entry/exit caps and late concessions do not establish the objective.
  Proposed next work: neutral-action exit-feasibility audit, then a separately priced and bounded exit-cost experiment;
  marginal inventory-aware entry admission remains a separate arm. Guard score is not a settlement probability or cash edge.
- Full rationale/scope: handoffs/20261110_inventory_capture_review.md. Source/evidence hashes:
  reviews/inventory_capture_20261110/source_snapshot.json. Branch research/predict-inventory-design-20261110.
- This turn was a source/evidence review, not a new simulation. No new PnL or lower-variance claim, live/frozen changes,
  or protected Nov9–13 data reads. B1 calibration still pending. Exit-feasibility audit not yet implemented/run.

## 2026-11-10 02:25Z — fill timing/queue calibration prioritized; awaiting updated capture

- User asked to finish calibration before more optimization. Prepared full-action ingestion for the A4b wallet capture
  and strict kernel journal export; source freeze f22cd1db. All16 synthetic controls pass, including deduplication,
  separate status clocks, incomplete capture statuses, bad-input refusal and protected-scope refusal.
- This is ingestion preparation only: no new live capture read, calibrated matching clock, queue fit or economic result.
  No wallet_events.jsonl.gz found in the searched local export roots; remote Tokyo state has not been inspected.
- Capture requirements: handoffs/20261110_fill_calibration_request.md. Evidence:
  reviews/fill_calibration_20261110/preparation_receipt.json. Branch research/predict-fill-calibration-20261110;
  checkout /tmp/cta-fill-calibration-20261110. Numeric jobs finished; recheck the shared slot before further work.
- Next: obtain the completed logging-build run ID/path, exact UTC interval and matching public tape. Nov9–13 remain
  protected by the frozen forward procedure; any use for calibration needs an explicit scope decision first.
  Millisecond status timestamps alone do not establish matching time. B1 is not complete; B2/B5/B6 still wait for it.

## 2026-11-09 18:09Z — optimization loop bounded cycle COMPLETE; no economic promotion

- Original two-round scope plus five declared follow-ups completed. Final audit checks8 receipt sets/91 unique pinned
  artifacts, source freezes, predeclaration timing/ancestry, output/verification links and arithmetic. All jobs finished.
- Native50ms updates-first: requests-3.72..-5.35%, original-price quote time+2.94..+4.37%, active share-time+2.58..+4.10%
  across3 assumed latencies on the same220 consumed Nov8 windows. Inventory-restricted grace keeps only4.86%/0.68% of
  unrestricted gain. Trade-off remains; higher net PnL and lower realized variance are NOT established.
- LZ/LC50 generated research code delivered with synthetic controls and explicit B4 fixed-size rejection exception.
  No claim of historical file parity, independent native guards, or deployed timer behavior. No strategy promoted/deployed.
- Current findings_v8.json; handoffs/20261109_optimization_loop_final.md; completion_audit/receipt.json. Protected Nov9–13,
  frozen engine and loss budget unchanged. Slot free now (recheck before C1); no background optimization job.
- Next economically useful evidence: own-fill ms source/receipt capture and filled/unfilled path calibration, then separately
  declared comparisons and untouched-day validation. Broader B1–B8/forward backlog remains open under existing procedures.

## 2026-11-09 18:04Z — optimization loop: paired-fill bounded continuity verified

- Seventh engineering pass: opt-in LC50 on LZ;228 synthetic traces (28 explicit/200 seeded),33 Python checks +1 Rust control.
  Native f20fdb1 matches3701 requests/16628 effective pair decisions/801 fills/5 retries/73 cancellation-only probes.
- All172 holds close<=50ms; no off-guard replenishment. Checks cover partial/full fills, late-cancelled fill activating CAP10,
  RO/stop/price/lower-line preemption, exact-deadline recovery and stale timers. Native artifact replay byte-identical;
  saved accepted identities, quantities, lifecycles, counts and known-fill ledgers reconcile; corrupt controls refused.
- Oracle consumes effective Python wants/pulls, not an independent native LC50 policy. No actual scheduler or economic claim.
  Zero create rejections in this population; B4's separate rejection suite/explicit frozen parity exception still apply.
- Sourcea565e3f6/declarationb3a52e2c; findings_v7 supersedesv6. Report handoffs/20261109_optimization_loop_v7.md.
  No market/protected data/live/frozen/loss changes. Numeric jobs finished18:04:26Z;4.5MiB data/228MiB peak buildRSS.
  Slot free now (recheck before C1). Next audit remaining optimization/promotion gates; prioritize economically useful evidence.

## 2026-11-09 17:52Z — optimization loop: paired research engine verified with explicit exception

- Opt-in LZ generated from frozen v3 matches pinned native f20fdb1 incremental pairs on238 synthetic traces:
  8,998 requests,14,961 pair decisions,2,177 fill notices,219 rejections,683 retries.51 Python checks +1 Rust control test.
  Saved counts/ledgers, trace/output regeneration and byte-identical native replays verify; corrupt artifacts/requests refused.
- Found/fixed LZ-only rejected-create accounting: release unanswered slot at rejection40ms, not nominal answer44ms;
  avoids an unnecessary250ms retry. Frozen/no-token behavior stays unchanged. Ordinary fixed-size controls exact,
  but two fixed-size rejection-stress paths intentionally differ. B4 does NOT satisfy blanket fixed-size parity.
- Exact full-v3 MICRO/G1–G6/PM90/GS/LD2/CAP10/XNA/XMIN1/XV/RO60/S240 AM sizing tested on6 synthetic traces;
  v6 supersedes v5 wording that called JOIN/S1 component resize tests full v3. Existing results unchanged.
- Full historical output-file parity, native guard/model equivalence, actual timer scheduling and economic/fill calibration
  remain unverified. No PnL/variance/promotion claim. Source805d467a/declarationd0cd457b; exact-arm f4194393/a638b203.
- Full report handoffs/20261109_optimization_loop_v6.md; current register reviews/optimization_loop_20261109/plan.json.
  Numeric jobs complete17:52:26Z; slot free now (recheck before C1). No protected data read or live/frozen/loss changes.
- Loop active; next source-only bounded50ms continuity + paired fill/inventory lifecycle controls. No new strategy settings.

## 2026-11-09 17:39Z (session 62f3ce33) - A6 / A3 / A4b on pmmmrust main (79a05d3); nothing deployed

- pmmmrust main 79a05d3 (pushed): A2 incremental (f20fdb1), A6 wallet liveness, A3 decision_reasons.jsonl.gz, A4b
  wallet_events.jsonl.gz; on A1 (366d419). Plan doc entries at the top of pmmmrust docs/PREDICT_FUN_PLAN.md; the Tokyo runbook
  lists the two new run files for exports. A Tokyo release needs a new tag and a re-rendered config (user's call).
- First live run with this build: check wallet_events.jsonl.gz session marks (`live` early in each session and which signal,
  never `not_live`, why each session ended) - a wallet connection's heartbeats were never measured directly.
- B1 waits for a live run with A4b (own-fill ms match clocks; the fill-clock audit found no fixed delay fits). B4: the
  optimization-loop session is building it (pinned f20fdb1). C1: run run_forward.sh only while no other run_light job runs.

## 2026-11-09 16:37Z (session 62f3ce33) - A2 incremental on pmmmrust main (f20fdb1); A4 checked; A6 / A3 / A4b in code

- pmmmrust f20fdb1 (main, pushed): `size_layering = "incremental_v1"` in both v3 examples (USER "Go"); base_topup_v1 (70b965b)
  stays selectable. Spec: handoffs/20261109_layered_size_keep_queue.md, section "Update 2026-11-09". Not deployed: a Tokyo release
  needs a new tag and a re-rendered config (user's call).
- A4: the public book's pending increase at our price comes p50 435 ms before our fill notice (98% of 98 live fills, 11-08), but
  it is per price; the wallet event's ms stamp is not kept (cut to the second for the kernel). A4b records it.
- In code on pmmmrust branch fix/pfmm-wallet-read-timeout-20261109 (local until reviewed): A6 (30 s read limit), A3
  (decisions.jsonl.gz), A4b (wallet_events.jsonl.gz). Next: rebase on f20fdb1, build, tests, independent review, check.sh, merge.
  Exports from the deployment host should then include both new files.
- C1 waits for a quiet research slot: do not start run_forward.sh while another session's run_light jobs run (it stops at the
  first refusal and has no resume; it is frozen, so it cannot be changed).
- B4 must also copy the incremental rule.

## 2026-11-09 17:15Z — optimization loop: native latency pass VERIFIED

- Fifth pass, unchanged reference/all100/update_first50, same220 consumed Nov8 windows/279059 decisions per profile/
  13077 neutral pulls per arm/profile. Zero exclusions, one partial day. Base output byte-identical and all base metrics exact.
- Updates-first50 versus paired reference, faster/base/slower feedback: venue requests -5.35/-5.33/-3.72%, aligned time
  +2.94/+4.17/+4.37%, off-target +71.76/+28.36/+5.61%, active share-time +3.97/+4.10/+2.58%.
  Keeps97.48/94.43/90.34% of100ms coverage gain with22.86/22.96/21.90% less added off-target time.
  Both captured modes share the request/coverage direction in all profiles. Added active exposure remains; no PnL/variance claim.
- Assumed create/cancel/pending delays20/10/50,40/22/95,80/50/160ms; not estimated venue quantiles or separate days.
  Native A2 f20fdb1, retry/throttle methods,150-share cap and1s reservation grace unchanged. No fill-clock bypass.
- Source75bf977e/declarationb1441503 precede outcomes.9 Rust adapter controls +4 Python checks pass;1980 saved
  window-arm-profile rows/117693 action-arm-profile rows verify identity/lifecycle/capacity and rebuilt coverage.
  Findings_v4.json supersedes v3 by adding this pass. Full table: handoffs/20261109_optimization_loop_v4.md.
- All numeric jobs finished17:15Z,375MiB peak for this pass,19MiB new data. Research slot free now; recheck before C1.
  Protected Nov9–13, live checkout, frozen engine and dollar loss budget unchanged. No new policy parameters (21 cumulative).
- Loop stays active: next full B4 paired-order generated-engine integration, no-token frozen parity and synthetic native
  differential traces. Actual timer scheduling and calibrated fills/forward evidence remain before economic promotion.

## 2026-11-09 17:08Z — optimization loop: native A2 pass VERIFIED

- Fourth pass uses pinned Rust f20fdb1 incremental pair/request/throttle/retry methods directly, with factual quote
  targets and assumed venue feedback. Same220 consumed Nov8 windows/279059 decisions/13077 neutral pulls per arm;
  zero exclusions, one partial day. No hypothetical fills, PnL, realized inventory variance or full B4 claim.
- Updates-first50ms versus native reference: venue requests62788->59444 (-5.33%), aligned time+4.17%, coverage
  73.85->76.93%. Off-target time+28.36%, active share-time+4.10%, active range expansion+4.16%; reservations-1.22%.
  Keeps94.43% of100ms coverage gain with22.96% less added off-target time. Both captured modes share the direction.
- Native throttle attempts2237->1262; local capacity refusals2795->2851. Capacity is150 shares, not the dollar loss budget.
  Counts include native outputs and actual modeled venue sends separately. Timing40/22/95ms +1s reservation grace.
- 51 native tests,7 adapter controls,4 Python checks and positive/refusal report binding pass. Saved660 window-arm/
  39231 action-arm rows reconcile action identity, order lifecycles, capacity and independently rebuilt coverage.
- Input preparation first refused a -1/no-quote sentinel translation bug before any native outcome run. Corrected
  source4776d822 and declaration8274fc63 precede results; failed artifacts retained and superseded by native_a2_v2.
- Combined findings_v3.json supersedes v2 by adding the native model; earlier model results retain their assumptions.
  Full comparison: handoffs/20261109_optimization_loop_v3.md. Sources/checks on research/predict-opt-loop-20261109
  in /tmp/cta-opt-loop-20261109. Largest successful new job758MiB RSS. All numeric jobs finished17:08Z; slot free now,
  recheck before C1 or another job. No protected Nov9–13, live/frozen engine, loss-budget or fill-clock changes.
- Loop stays active. Next: native faster/slower latency sensitivity with exact base replay parity. Full B4 generated
  engine integration, actual timer scheduling and calibrated fills/forward evidence remain before economic promotion.

## 2026-11-09 — optimization loop: first three rounds VERIFIED; next integration pass


- User authorized a continuing optimization loop with tracked findings. Branch `research/predict-opt-loop-20261109`,
  checkout `/tmp/cta-opt-loop-20261109`; register `reviews/optimization_loop_20261109/plan.json`, current combined
  findings `findings_v2.json` (supersedes two-round `findings.json`). As-of **16:40:02Z**: same220 consumed Nov8 windows,
  279059 decisions /13077 neutral pull actions per arm/scenario, zero excluded windows. One partial day, no CI/PnL.
- Round1: reducing-only grace retains4.86% of unrestricted coverage gain (208 admissions); aggregate reservation-room
  grace0.68% (29). Both add no non-reducing share-time at base latency; benefits are small. Exact native supplement
  captures93 fill-notice states at0–65s, including partial fills without a decision; original outputs remain exact.
- Round2: timer-first50ms has no same-price bridges;25ms adds exposure with negligible base coverage gain. Round3
  tests updates stamped exactly at the50ms deadline BEFORE expiry, then cancels still-off orders at the same deadline.
  No future timestamp or longer grace. Source cb1af0a7, declaration d6f4ce11; exploratory follow-up, cumulative21 policies.
- Updates-first50ms: requests **-6.98%**, original-price aligned time **+2.37%**,
  off-target time **+25.66%**, active share-time **+2.74%** versus baseline.
  Keeps **89.44%** of100ms coverage gain, removes **24.23%** of its added off-target time;
  all2376 same-price bridges survive. Same direction in faster/base/slower scenarios with paired references.
- 37 controls pass; saved metrics/actions, old baselines, inventory identities and coverage accounting verify. Across
  rounds6160 window-arm/366156 action-arm rows; source/receipt hashes and falsifiers retained. Largest job1434MiB RSS;
  all numeric jobs complete. No protected Nov9–13, frozen/live changes, loss-budget change or fill-clock bypass.
- Next: A2/B4 paired-order integration/parity and actual timer scheduling; calibrated fills/forward evidence before
  economic selection. Current shared Rust HEAD was observed at f20fdb1 (incremental sizing default), not used as the
  counterfactual OMS here. No candidate promoted/deployed. Inventory-range follow-up remains untested.
- Detailed comparisons, limits and reproduction: `handoffs/20261109_optimization_loop_v2.md`. Original two-round
  report values were valid; its final mutable-register write used an exclusive-create helper and failed. Corrected
  publication is recorded in-band by findings_v2; original log/outputs retained. Further rounds need new declarations.

## 2026-11-09 15:45Z (session 62f3ce33) - A2 size layering implemented (pmmmrust main 70b965b)

- A1 (366d419) and A2 (70b965b) are on pmmmrust main; neither is deployed. A Tokyo release needs a new tag and a re-rendered config:
  `cancel_late_fill_window_ms = 1000` ([live]) and `size_layering = "base_topup_v1"` ([ladder]) come from the v3 examples.
- Next on the to-do list: A3 (decision reasons in captures), A4 (fill-notice latency), A6 (wallet socket inactivity timeout),
  B1 (fill-model calibration) before B3 (pull / size line buffer), B4 (research LZ token).

## 2026-11-09 — native guard / bounded soft-pull study COMPLETE

- As-of **16:01:32Z**, verified **16:02:15Z**: same220 consumed Nov8 windows /279059 first-minute decisions /
  13077 neutral touch-pull actions, zero excluded windows. Exact native replay reproduces1262612 outputs across
  both modes and all prior decisions; native wanted-guard contradiction0/137325. Old reconstructed8.77% mismatch
  is resolved for attribution by exact inputs, not by assuming reconstructed scores correct. A read-only observer
  adds1249610 due-event probes (0–65s) with original decisions/books byte-identical; no live source changed.
- Fixed incremental research planner; retain only accepted small touch<=5 when v is narrowly below the pull line,
  native hard/price/size predicates still permit it, no rung duplicate; no new guard-off order. Maximum100ms,
  early cancel at lower-line/hard changes including intervening native probes; no rung retention.
- Predeclared primary0.01c: requests65211->60651 (**-6.99%**), original-price aligned time**+2.65%**,
  coverage76.17->78.18%. Both v6/basis same direction. Off-target time**+33.87%**, active share-time**+3.40%**,
  reserved share-time**-2.42%**; no inventory-risk reduction claim.4783 retained pulls,2376 same-price reopens,
  790 cancellations at extra probes. Wider bands give smaller additional coverage gains with more exposure.
- Primary future signed midpoint drift+0.361c at1s(n4766,17 stale),+0.638c at5s(n4759,24 stale), below matched
  controls+0.387/+0.671c from500 declared draws. Action diagnostic only, no fill markout/PnL/predictive-skill claim.
- 26 controls pass;1100 window-arm/65385 action-arm rows reconciled, prior baseline parity, corrupt-count refusal.
  Four new policies, cumulative16; source2d235fd4/declaration4c69c4d1. All jobs done; study1134MiB/67.93s,
  verifier401MiB/8.78s. Native artifacts693MiB plus study1.2MiB. No protected Nov9–13, fill-clock bypass or deployment.
- Branch research/predict-soft-pull-20261109; detailed handoff `handoffs/20261109_soft_pull_results.md`, receipts
  `reviews/native_guard_20261109/` and `reviews/soft_pull_20261109/`. Mechanical candidate only: one consumed partial
  day, factual inventory/order state, modeled latency, no changed fills/throttle/rejections. New Rust A2 70b965b
  was not the counterfactual OMS; B4 parity and calibrated fills/forward evidence still needed. Symmetric/EMA,
  size-line and inventory-aware exit variants remain open. Keep primary band for research, no live promotion.

## 2026-11-09 — pull/re-entry study COMPLETE; no coverage-preserving improvement

- User requested moving beyond the existing A2 sizing task to prediction-driven pull/repost behavior. Branch
  `research/predict-reentry-churn-20261109`, checkout `/tmp/cta-reentry-churn-20261109`. Producer as-of
  **2026-11-09 14:59:08Z**, verification**14:59:57Z**. **220 consumed Nov8 windows**, first60s/four lanes,
  **279,059 recorded decisions**, **30,796 reference quote opportunities**, zero excluded windows. No fills/PnL.
- Hold incremental sizing fixed. Every original pull/reprice cancels immediately. Rapid re-entry means same price
  within100ms of a recorded pull. Compare cancel-ack wait, continuous eligibility25/50/100ms, and all-entry50 control.
- Relative to the fixed-sizing reference65,211 requests: cancel-ack **requests-0.22%, aligned quote time-0.14%**;
  re-entry25 **-8.46%/-8.19%**; re-entry50 **-13.80%/-11.19%**; re-entry100 **-19.49%/-15.84%**;
  all-entry50 **-43.15%/-27.06%**. All lose original desired-price coverage; same tradeoff direction in both runs.
  Off-target time also falls for confirmation variants; this tradeoff has no economic valuation here.
- Minimum delays are checked at original decision events, not a new timer. Diagnostics as-of15:04:07Z:
  actual delay median46.93ms for25ms (n7792 delayed admissions),75.84ms for50ms(n5976),125.15ms for100ms(n4101).
  Completed raw opportunities have median100ms (n30699);97 window-censored. Older average surviving quotes do not
  establish better capture: all-entry50 raises oldest age515->643ms while losing27.06% aligned time.
- **30 controls pass**; reference exactly matches prior incremental outputs on18 metrics for220 windows.
  Artifact verifier reconciles1320 window-arm rows and184776 action-arm rows, unchanged reference identities,
  admitted/withheld/censored statuses, lost/gained coverage and all ratios. Corrupt counts/negative waits refused.
- Six arms/five new policies; cumulative multiplicity12 including prior capture study. Producer freeze dfbe5ddf,
  declaration252bd26f. Receipts `reviews/reentry_churn_20261109/`; data/logs under matching research directory.
  All jobs finished, producer peakRSS294MiB/48.66s, output1.5MiB. No live code, protected Nov9–13, fill-clock bypass
  or frozen-engine change. One consumed partial day, no confidence interval, PnL, queue-rank or inventory-risk claim.
- **Do not promote these re-entry delays on current evidence.** Existing B3 symmetric/exit-band and model-stability
  questions remain open. Investigate unnecessary pulls themselves using exact guard values/reasons (existing A3),
  or an exact native-input replay with decision parity. Reconstructed scores cannot identify soft versus hard pulls
  reliably. No claim that all observed pulls are prediction-driven. Detailed results and limitations:
  `handoffs/20261109_reentry_results.md`. Venue throttling and independent timer-driven admission remain unmodeled.

## 2026-11-09 14:24Z (session 62f3ce33) - A1 NetCap late-fill window implemented (pmmmrust main 366d419)

- Code, two reviews and check.sh done; main fast-forwarded and pushed. Not deployed: Tokyo needs a new release tag and a
  re-rendered config (`cancel_late_fill_window_ms = 1000`). Next: A2 layered size (same workflow).

## 2026-11-09 — quote capture mechanics COMPLETE; incremental sizing candidate

- User authorized improving capture after the timing repair. Branch `research/predict-quote-capture-20261109`,
  checkout `/tmp/cta-quote-capture-20261109`. As-of **2026-11-09 14:42:01Z**, **220 consumed Nov8 windows**, first60s,
  four lanes, **279,059 recorded decisions**, zero excluded windows. Fixed factual desired path, no simulated fills.
- Incremental sizing starts with one target-sized order; on same-price small-to-large growth, keep the small base
  and add20. A later shrink cancels only the top-up. Original whole25 orders still resize when needed. Partial
  components are not replenished; minimum-notional fallback, hard pulls, side/inflight caps and reservations remain.
- Versus single-order reference: **requests68,299 ->65,211 (-4.52%)**, time active at desired prices**+3.90%**,
  oldest active quote**399 ->515ms (+28.97%)**, reserved share-time**-3.57%**. Active share-time**+2.94%** and
  off-target/cancel-pending quote time**+0.82%**: do not infer reduced inventory risk. Same desired target path100%.
  Requests lower in188 windows/equal25/higher7; both recorded runs show fewer requests and more aligned quote time.
- Always splitting gives**75,660 requests (+10.78%)**, despite preserving more quote age. Two tickets cost more on
  pulls/reprices. Entry buffers0.005/0.01/0.02/0.04c cut requests but retain only79.82/67.24/48.33/21.90% of desired
  quote time; keep off in the candidate. Reconstructed scores disagree with live quote-on at the pull line on
  **12,044/137,325 wanted lane updates (8.77%)**; no missing/stale joins. Do not use this as live guard calibration.
- **20 controls pass**. Follow-up reproduces440 reference/always-layered window-arm rows across23 columns exactly.
  Independent artifact checks verify hashes, populations, desired-price identity and summaries; corrupt counts
  refused. Reduce-only retention also subtracts prior cancellation reservations, preventing double-use of room.
- Six-arm v1 receipt invalid due to a reference-size-increase retention bug; v2 supersedes it in-band and adds
  a regression control. Only v2 is used. Incremental is one post-screen follow-up (multiplicity7), frozen source
  `5e651fae`/declaration `f5857810`; price-alignment audit declared separately after initial mechanical results.
- No PnL/fill-rate/queue-rank/variance claim or fresh-day validation. Same40/22/95ms acknowledgements,1s reserve
  grace,150-share cap,3 in-flight creates/side; no throttle/backoff/rejection model. No live code, frozen-engine or
  fill-clock-gate change; protected Nov9–13 remain unread. All jobs done, follow-up peakRSS<200MiB, outputs~8.5MiB.
- Port/research handoff: `handoffs/20261109_incremental_capture_results.md`; compare incremental with the earlier
  always-layered A2 spec before implementation selection. Full receipts in `reviews/quote_capture_20261109/`
  (use receipt_v2.json) and `reviews/quote_incremental_20261109/`; data under corresponding research directories.
  Next economic comparison requires validated matching-event evidence and filled/unfilled action reconciliation.

## 2026-11-09 — trade timing repair COMPLETE; precise replay remains data-limited

- User requested fixing the replay timing defects. Producer receipt as-of **2026-11-09 13:53:36Z**, final runtime
  verification **13:56:39Z**. Same220 consumed Nov8 windows; branch `fix/predict-trade-timing-20261109`, checkout
  `/tmp/cta-trade-timing-20261109`. No protected Nov9–13 read, strategy ranking, or live code/settings change.
- `pf_trade_tape.py` preserves full transaction/taker/maker hashes, settlement ID, exact wei prices/amounts, and
  integer local receipts. Repeated pages deduplicate by exact identity; conflicting economics/identities are refused.
  `pf_step1.py` now retains those fields; its old `exec_s`/phase summaries are explicitly confirmation-time descriptions.
  `pf_txmap.py` groups full identities directly, with a separately named identity artifact; no price/time/rank join.
- Matched and Confirmed status clocks remain separate. A first Matched floor-second stays an interval; confirmation
  never overwrites it. Missing/conflicting/failed events remain statuses. No E2, anonymous-increment association,
  earliest-leg substitution, fixed-delay fitting, or clipping to our observed lifetime supplies a fill time.
- Rebuilt **375,569 REST maker legs**, retaining pre/post confirmation records and all220 markets. **375,470** have
  no exact matched-event attachment; **99** own legs have matched-status intervals, **zero** exact subsecond match
  clocks. All99 map unambiguously to the prior exact own identities; **99/99 intervals overlap independently checked
  lifetimes**, without clipping. All**42,064 order actions** retained; zero quantity differences above1 microshare.
  Seven failed trades remain unmatched status histories. This is an engineering reconciliation, not fill-model validation.
- Preserve **80 off-cent-grid prices** as raw wei + explicit `off_cent_grid` status / nullable tick. Do not round
  them or drop their trades. Remaining375,489 legs are on the cent grid. Original build stopped at the first off-grid
  record; v2 declaration records this correction and preserves the failed log. No result-bearing v1 receipt existed.
- `pf_step4_ladder_identity.py` is the replacement replay entrypoint, using `PF_TRADE_TAPE=<manifest directory>`.
  It requires matching-event clocks with provenance and suitable precision for the full declared population;
  rejects legacy jitter/shift options; removes confirmation fallback and earliest-leg retiming from the pinned engine.
  Transaction clock disagreements are refused. Actual match time, not confirmation time, partitions market windows;
  before/in/after counts remain in the timing summary. Registry/tape market populations must match.
- Current historical tape is correctly **refused before strategy data or output**. The99 consistent one-second
  intervals are not exact matching times, and other public trades have unknown clocks. No corrected PnL is claimed.
  Full matching-event evidence (or a separately specified and validated uncertainty-aware simulator) and reconciliation
  of filled AND unfilled actions are still needed before inventory/quote-retention rankings can resume.
- Old `pf_step1_timing`/`pf_step1_booktime` and forward/C2 drivers refuse the invalid clock pipeline on this fix branch
  before further data consumption. Frozen engine files and earlier artifacts remain byte-identical. Historical reproduction
  uses the earlier commit; do not bypass the new guard for a new validation claim. These are isolated branch changes,
  not changes to the live bot or the other session's working engine.
- **14 regression controls pass**, including hash-prefix collision, exact grouping, duplicate/conflicting observations,
  confirmation drift, missing/foreign clock refusal, price preservation, transaction clock disagreement, market boundaries,
  and actual script refusal. Independent verifier checks5 artifact hashes,375,569 exact rows,42,064 actions,99 clocks,
  absence of invented timestamps on375,470 rows, and the actual replay CLI refusal. Corrupted receipts are refused.
- Source producer freeze `6d6ca8b2`, final replay-window source `301bb625`; receipts in
  `reviews/trade_timing_fix_20261109/`. `verification_v3.json` supersedes the initial verifier in-band for the later
  replay-window adapter check; producer receipt/data unchanged. Output directory
  `data/predict_fun/research/trade_timing_fix_20261109/` is109MiB; rebuild peakRSS1374MiB, one capped job at a time.
  All our jobs finished. README describes the replacement path and remaining data limitation.

## 2026-11-09 12:55Z (session 62f3ce33) - to-do list after the live vs replay review

- predictfun/handoffs/20261109_TODO.md: live fixes for the port (A1 NetCap 1 s window and A2 layered size, both user-approved; A3
  decision reasons in the capture; A4 fill-notice latency; A5 watch items), research (B1 fill-model calibration first, B2 re-run v1/v3,
  B3 pull / size line buffer approved for testing, B4-B8), scheduled forward runs (C1 due now, C2 after 11-14 02Z), decisions (D1-D3).

## 2026-11-09 — exact-order fill-clock audit COMPLETE; replay timing fails lifecycle checks

- User requested continued testing toward preserving trading value with lower inventory. Clock audit as-of
  **2026-11-09 12:24:00Z**, timestamp follow-up **12:36:36Z**; **220 consumed Nov8 windows**, 42,064 order actions.
  No protected Nov9–13 data read. Branch `research/predict-fill-clock-20261109`, worktree `/tmp/cta-fill-clock-20261109`.
- Exact Signed order hash + settlement/trade identity reconciles all99 confirmed REST maker legs (86 orders) to
  kernel notices. Seven failed trades remain statuses. All30,413 signed-order quantities agree within one microshare;
  no missing confirmed trade. This is a timing defect, not evidence of missing live quantity.
- Necessary match interval is `[Control Place, min(confirmed cancel, first positive fill notice)]`.
  Acceptance is NOT a lower bound. Original pinned engine clocks reproduce the original diagnostics exactly.
  Full9-field leg joins identify93 clocks across83 orders;6 ambiguous clocks are retained, not guessed.

| Clock, tolerance10ms | Eligible legs / orders | Before placement | After terminal / notice | Violating orders |
|---|---:|---:|---:|---:|
| J | 93 /83 | 11 | 7 | 16 |
| X | 93 /83 | 12 | 3 | 12 |

- No fixed earlier shift0/50/100/200/400/800ms passes. In fact **no constant offset of either sign** fits either run:
  basis35 legs/32 orders needs an earlier shift at least10,138.41ms AND at most-923.62ms;
  v6 58 legs/51 orders needs at least556.22ms AND at most-975.23ms. Still impossible allowing10ms tolerance.
- Follow-up on all99 exact legs: REST `executedAt` equals the **Confirmed** reported second in99/99,
  whereas it equals the first **Matched** second in49/99. Differences:49 zero,49 +1s,1 +10s.
  All99 raw REST stamps have zero fractional milliseconds. `pf-exec/user.rs` derives each status's `matchtime`
  independently from that status event's timestamp, truncated to seconds; this is not an immutable matching clock.
  The Matched floor-second interval overlaps lifecycle bounds99/99; REST floor interval misses39/99 and even a
  rounded-second assumption misses5/99. These are necessary consistency checks, not precise match-time validation.
- Exact witnesses (full identities in `semantics_receipt.json`): basis CID1794147344754480 was placed21:55:57.104,
  canceled21:55:57.320, first Matched receipt21:55:57.482. REST reports21:56:07; E2 selects a book drop21:56:07.458.
  Thus a book change10s after the possible fill is assigned as its trade time. V6 CID1794108465178750 was placed
  07:15:35.551 and canceled35.859. Its own pending-cell estimate35.710 fits, but transaction grouping chooses another
  leg's earliest pending-increment estimate34.576 across37 legs. That anchor predates submission by975ms.
  Public pending increments lack signed-order identity; amount/price proximity does not establish exact trade linkage.
- 500 predeclared within-run/hour/side permutations on83 order actions (77 permutable) produce compatibility
  mean0.4157/max0.5422 vs observed0.7831. Instrument diagnostic ONLY; one partial day, no CI/profit/strategy inference.
- **No new strategy sweep or calibrated-PnL claim.** The earlier inventory comparisons remain conditional on a
  replay clock that now fails an exact-identity test; this audit does not measure the dollar correction to those results.
  Do not fix this by fitting a delay/minimum resting age or clipping estimated times to our observed lifecycle.
  Next implementation must retain full order/transaction/settlement identities and separate match-status, confirmation,
  exchange-source, and local-receipt timestamps; preserve uncertainty when public data cannot identify actual match time.
  Reconcile both filled AND unfilled order actions before reusing the simulator to rank quote retention/inventory rules.
- Sources `pf_fill_clock.py`, `pf_fill_clock_semantics.py`, `pf_fill_clock_verify.py`; immutable receipts under
  `reviews/fill_clock_20261109/`; local data `data/predict_fun/research/fill_clock_20261109/`.
  Original source freeze80387e6e (v3), follow-up d0b4ca1f, independent verifier861935a5.
  13 positive/refusal controls pass. Artifact verifier independently checks16 clock/tolerance results,42,064 actions,
  all99 status histories and7 artifact hashes; a deliberately corrupted report is refused.
- Instrument corrections preserved in superseding declarations: v2 canonicalizes integer JSON dictionary keys;
  v3 corrects THIS auditor's Down-price conversion (Control price is Up-space on both sides). No result-bearing clock
  receipt existed before v3. Frozen extraction's derived `yes_tick` is deprecated; v3 recomputes coordinates from raw
  Control price. Earlier economic studies and live code were not changed. Failed logs/declarations remain provenance.
  All our jobs complete. No live changes, deployment, strategy tuning, or forward-data consumption.

## 2026-11-09 12:11Z (session 62f3ce33) - cancel churn; layered size spec

- 89% of live cancels are pulls / resizes at unchanged best prices that mostly reverse within 1 s (guard value jitter at the lines).
- USER asked for size changes that keep queue position: spec predictfun/handoffs/20261109_layered_size_keep_queue.md (base + top-up
  order; port session live, research engine token LZ with parity). Pull hysteresis = strategy change, not done.

## 2026-11-09 — closing-liquidity experiment COMPLETE

- As-of **2026-11-09 11:43:41Z**, nine consumed dates Oct23–27 and Oct31–Nov3, **2592 markets per arm/timing**,
  J/X, zero exclusions/missing valuations. Guards trained on eight dates: conditional historical engineering study,
  not independent validation. Declared source `62b2c98e`, declaration-copy commit `d6c44362`, report/verifier head
  `61d727da`; branch `research/predict-exit-liquidity-20261109`, checkout `/tmp/cta-exit-liquidity-20261109`.
- `exit_keep` retains a same-price guard-eligible closing order whose known remainder fits aggregate closing room,
  even above the AM target. `exit_close25` also sizes new closing orders up to25 available shares instead of AM low5.
  Targets apply before minimum-notional checks; existing partial orders retain original size identity without top-ups.
  Original prices/guards/RO60/S240, pending create/resting/cancel reservations, acknowledgement+100ms grace+.001ms,
  and submission checks retained. All pre60 fills AND requests exactly match baseline. No new adding orders from60
  or orders from240. Loss budget uncapped,150-share backstop unchanged, rebates excluded. No price concessions.

| Timing / arm | Settlement $ | +1s markout $ | FIFO matched $ | Mark at240 $ | Carry $ | Mean residual shares | Daily PnL SD $ |
|---|---:|---:|---:|---:|---:|---:|---:|
| J baseline | 514.06 | 602.83 | 134.02 | 677.12 | -163.06 | 6.56 | 134.51 |
| J late_reserved | 502.04 | 557.97 | 99.78 | 658.36 | -156.32 | 7.12 | 137.60 |
| J exit_keep | 491.08 | 564.19 | 156.66 | 668.44 | -177.36 | 6.97 | 135.65 |
| J exit_close25 | 508.39 | 553.33 | 168.95 | 660.14 | -151.75 | 6.73 | 131.55 |
| X baseline | 317.63 | 463.56 | 173.54 | 431.16 | -113.54 | 5.72 | 103.09 |
| X late_reserved | 309.94 | 440.54 | 133.41 | 385.42 | -75.48 | 6.15 | 103.12 |
| X exit_keep | 303.22 | 440.88 | 159.20 | 380.12 | -76.90 | 6.02 | 103.30 |
| X exit_close25 | 317.20 | 435.68 | 110.81 | 365.51 | -48.31 | 5.77 | 100.57 |

- **Objective not achieved versus baseline.** Keep changes residual inventory **+6.35/+5.27%**
  and +1s value **-6.41/-4.89%** J/X. Close25 changes inventory
  **+2.58/+0.80%**, +1s **-8.21/-6.01%**,
  and mark-at240 **-2.51/-15.23%**. Both leave more terminal inventory at both timings.
- Relative to `late_reserved`, keep raises actual reducing flow **+12.01/+13.94%**,
  reduces residual inventory **-2.07/-2.15%**, and changes +1s value by
  **$+6.22/+0.35**. Close25 raises reducing flow
  **+32.16/+40.84%**, reduces residual inventory
  **-5.55/-6.31%**, but changes +1s value
  **$-4.63/-4.86**. The mechanism repairs part of the reservation control's lost exit liquidity.
- Close25 still executes **-12.89/-5.00%** less reducing quantity than baseline.
  Its settlement delta is **$-5.67/-0.42**. X near-equality masks **$65.65 worse mark-at240
  offset by $65.23 better carry**; it does not demonstrate preserved trading value. FIFO changes
  **$+34.93/-62.72**; FIFO includes drift, not just spread capture.
  Baseline late fills include198.11/63.97 inventory-increasing shares, which are not equivalent to safe exits.
- Close25 daily PnL SD changes **-2.20/-2.44%**, but integrated inventory-square
  changes **+0.87/-0.30%**. This is not a proven variance benefit;
  inventory-square is an exposure proxy, not settlement variance adjusted for probability.
- Thirty predeclared comparisons,2000 shared UTC-day sign flips and2000 paired day bootstrap draws. Every settlement
  difference interval includes zero. Close25-baseline mean daily settlement intervals: **[-$14.42,+$16.60] J**,
  **[-$21.27,+$23.18] X**, family p=1. Versus late_reserved, inventory-square reductions have family p=.03348 for
  keep and .00600 for close25 at both timings. Neither establishes the objective versus baseline; all remain exploratory.
- Engagement: keep retains above-AM-target orders on13,865/12,261 J/X quote decisions. Close25 raises its size target
  on62,547/51,569 decisions. These are decisions, not distinct orders or extra fills. Use pre-retarget counters;
  the later planner sees the already-adjusted target, so its generic above-target counters are zero by construction.
- Provenance: **19,435 candidate fill rows**; no new post60 order adds economic inventory within1e-7. The13.67/0 late
  opening shares J/X come from inherited pre60 orders. Last fill240.029s precedes the240.1s carry valuation.
  Inherited-opening positive control and new-opening/missing-provenance/post-mark-fill refusals pass.
- **47 controls, eight-market smoke,18 full-day reconciliations PASS**. Fresh first-day baseline AND late_reserved J/X
  reproduce exact win/fill/request artifacts. Verified21 sources,216 result artifacts,18 new and18 reference metadata
  sets, declaration/receipt copies. Peak RSS1.6GiB;482GiB disk free and24GiB RAM available at completion. All our jobs finished.
- **Fill-model warning measured on THIS baseline:** age<100ms OR cancel-pending fills account for84.49%/81.70% of
  shares and **84.44%/78.46% of +1s value** J/X. New Nov8 live/replay evidence questions many such fills. Its frozen-v3
  baseline differs from this continuous/execution-adapted baseline; do not substitute its percentages. Live total order
  lifetime differs from replay fill age; nearest pending-message matching is not exact trade identity. Removing those
  fills from reported PnL is not a corrected counterfactual: inventory and future orders would change too.
- Conclusion: safe exit retention helps relative to strict reservations; neither candidate solves the baseline objective.
  Calibrate trade/order timing and cancellation lifecycle against exact live identities before another PnL search.
  Future inventory controls should price closing cost separately from entry edge, preserve early quoting, and use
  later independent validation. A better settlement forecast is needed to justify directional holdings, not to reduce risk.
- Reproduce from this branch: `pf_exit_study`, `bash predictfun/research/run_exit_liquidity.sh`, then
  `pf_verify_exit_liquidity`. Receipts: `reviews/exit_liquidity_20261109/`; data, evidence snapshots and complete
  `comparison.csv`: `data/predict_fun/research/exit_liquidity_20261109/`. Receipt SHA256
  `fbd470a772be94076adb0bad12114105b54d5484da8f2cdb3f3167d3d1f534a7`. No live/frozen/forward changes or protected Nov9–13 reads.

## 2026-11-09 10:32Z (session 62f3ce33) - short-lived fills are replay timing; NetCap window approved

- Live places / cancels as fast as the replay assumes; the fills it misses were matched before its order reached the book (feed
  pending messages for them arrive around or before live's acceptance). Replay fill model needs an earlier trade anchor / minimum
  resting time calibrated on live before any edge claim (user's call).
- NetCap 1 s late-fill window approved by the user: spec in predictfun/handoffs/20261109_cancel_late_fill_window.md, for the port session.

## 2026-11-09 10:19Z (session 62f3ce33) - short-lived fills are the replay's edge; NetCap fix options

- Replay +1 s profit is ~85% from fills on orders < 100 ms old or cancel-pending; live gets none of those; its fills are -0.84 c/sh at
  1 s. The v1 / v3 backtest edge is likely overstated until the fill model is calibrated on live outcomes (user's call).
- NetCap phantom: proposed live fix = count a cancelled order for a 1 s late-fill window after the cancel is confirmed, not until the
  5 s REST audit (audit stays as backstop): 11-08 refusals -95%, 0 of 71 late fills missed. Alternative: net_shares 300. For the port
  session (pmmmrust OMS unbooked_exposure / exec CANCEL_AUDIT_GRACE_MS); not implemented here.

## 2026-11-09 09:50Z (session 62f3ce33) - Tokyo live vs replay on the same 11-08 windows

- Live fills 36-46% of the replay's shares on 220 windows (STATUS same time). The same decisions are made at the same times.
- Two causes: (1) the replay fills very short-lived orders (live lifetime < 100 ms) that live almost never gets (2-6%);
  (2) live kernel NetCap refuses places because cancelled orders count at full size for ~5 s until the REST audit.
- Fix options, all the user's call: live: clear cancelled orders' exposure fast (fast reconcile for every cancel, or a shorter late-fill
  window, or a NetCap above the phantom level); replay: calibrate the fill model on live outcomes (minimum resting time / earlier
  trade anchor) and re-run v1 / v3 in-sample; strategy: hysteresis / minimum order life to cut the ~50 ms place-cancel churn (new
  version). Not replayed: the basis run's 11-09 windows (v3 block). Outputs outside git: pmmm-data/review-exports/live-samewindow-20261108.

## 2026-11-09 — late reduction experiment COMPLETE

- As-of **2026-11-09 10:23:48Z**, nine consumed dates Oct23–27 and Oct31–Nov3, **2592 markets per arm/timing**,
  J/X, zero exclusions/missing valuations. Fixed guards trained on eight dates: exploratory historical comparison.
  Branch `research/predict-late-reduction-20261109`, checkout `/tmp/cta-late-reduction-20261109`.
  Declared policy/source `8c8ee4d4`, report source/declaration-copy commit `b3cbbbd7`; data under
  `/home/yuqing/ctaNew/data/predict_fun/research/late_reduction_20261109/`.
- Two fixed candidates, no post-result tuning: `late_reserved` applies aggregate reducing reservations and eligible
  remaining-quantity retention only from60; it refuses retries that now increase inventory or fall beyond240.
  `late_taper1` adds exactly one tick of passive improvement when inventory exceeds a target declining linearly
  from10 at60 to0 at240, only strictly inside the known spread. This is an urgency trigger, NOT a hard bound or
  guaranteed liquidation. Original guards/AM5/25/RO60/S240/minimums and execution timings retained.
- Exact original pre60 fills AND requests match for every market in both variants at J/X. Cancellation history is
  retained through acknowledgement +100ms fill-report grace +.001ms, while exposing original lifecycle before60.
  This conservative simulator treatment combines reservation, retention, grace and late retry checks; their individual
  contributions are not isolated. Uncapped loss budget,150-share backstop unchanged, rebates excluded.

| Timing / arm | Settlement $ | +1s markout $ | FIFO matched $ | Mark at240 $ | Carry $ | Mean residual shares | Daily PnL SD $ |
|---|---:|---:|---:|---:|---:|---:|---:|
| J baseline | 514.06 | 602.83 | 134.02 | 677.12 | -163.06 | 6.557 | 134.51 |
| J late_reserved | 502.04 | 557.97 | 99.78 | 658.36 | -156.32 | 7.121 | 137.60 |
| J late_taper1 | 508.54 | 573.72 | 57.30 | 637.89 | -129.35 | 6.752 | 130.63 |
| X baseline | 317.63 | 463.56 | 173.54 | 431.16 | -113.54 | 5.720 | 103.09 |
| X late_reserved | 309.94 | 440.54 | 133.41 | 385.42 | -75.48 | 6.154 | 103.12 |
| X late_taper1 | 353.45 | 441.25 | 131.08 | 404.01 | -50.57 | 6.020 | 109.06 |

- **Objective not achieved versus baseline.** Late_reserved raises mean terminal inventory8.60%/7.58% J/X and
  lowers actual reducing flow34.09%/32.55%. Taper recovers30.02%/14.09% reducing flow versus that control and lowers
  its inventory5.18%/2.18%, but still leaves **2.98%/5.24% MORE inventory than baseline**. Its late time-weighted
  absolute inventory rises1.99%/3.51%, total q² rises0.16%/2.57%. It reduces less economic inventory than baseline
  by14.30%/23.05%. Tightening the urgency trigger alone does not force an executable exit.
- Taper's settlement delta versus baseline is **-$5.52/+$35.82**; +1s value falls **$29.11/$22.31** (~4.8% each),
  matched FIFO profit falls **$76.72/$42.46**, and mark-at240 falls **$39.22/$27.15**. Carry improves$33.71/$62.97.
  Thus the positive X final PnL is outcome carry, not improved trading value or less exposure. FIFO includes price drift;
  mark-at240 is a valuation, not an executable midpoint exit. Alternative FIFO residual PnL is$451.24/$222.37 for taper.
- Exact originating intent-decision, submission and fill records preserve attribution across throttled retries.
  Taper executes **1769.16/981.16 shares at a one-cent concession**, all economically reducing: **$17.69/$9.81**
  quoted concession on that filled quantity. Already included in cash PnL; never subtract twice. This same-fill price
  difference does not assume the unshifted order would have filled, and is not a causal explanation of the whole PnL delta.
- All newly submitted post60 orders have zero economic reverse-opening quantity within1e-7. The remaining13.67/0
  opening shares J/X are inherited orders submitted before60, for both candidates. Independent provenance audit:
  19,549 candidate fill rows; max fill second239.787126, all before the240.1 midpoint used for carry attribution.
- Taper beats baseline on3/9 J and5/9 X days; leave-one-day-out total delta spans **[-$59.03,+$29.97] J** and
  **[-$24.47,+$95.44] X**. Nov2 contributes+$60.29 X; excluding it descriptively turns pooled X uplift into-$24.47.
  This is a post-hoc concentration check, not a separate validation set.
- Predeclared2000 shared UTC-day sign flips and2000 paired day bootstrap draws,18 comparisons. All settlement
  difference intervals cross zero. Taper-baseline mean daily intervals **[-$14.48,+$15.03] / [-$21.28,+$29.58]**, family
  p=1 both. Reservation-only q² increase and taper-versus-reservation q² decrease have family p=.0105 both timings;
  taper has no family-significant improvement versus baseline on primary metrics. SD/tails/carry remain descriptive.
- **39 controls, eight-market smoke and all18 full-day reconciliations PASS**. Fresh first-day baseline J/X is exact;
  other references hash-verified. All17 source files,162 result artifacts,18 per-run metadata sets, declaration and
  receipt copies verified. Supplemental overfill positive/missing-identity refusal controls pass. Peak RSS1.6GiB;
  ~483GiB disk free,24GiB RAM available; no experiment job remains.
- Reproduction: `pf_late_study` declaration then `bash predictfun/research/run_late_reduction.sh`. Receipt and declaration
  under `reviews/late_reduction_20261109/`, plus `verification.json`. Data directory also has `comparison.csv`.
  Receipt SHA256 `b723eb053e79399cdb615bff975f62267ca0dba9c963a6c1fa4bf7a192447202`.
- Conclusion: do not adopt these variants as a solution to preserving trading gains while reducing inventory. The
  constrained exit treatment removes too much effective reducing liquidity; one-tick urgency only partly repairs it.
  Further work should isolate exit availability/retention/grace before adding stronger price concessions; any change to
  lifecycle assumptions needs supporting execution evidence. No live code/settings, frozen candidate/forward race,
  or protected Nov9–13 data changed/read. All work is historical research.

## 2026-11-09 — allowed inventory-range experiment COMPLETE

- As-of **2026-11-09 08:19:39Z**. All nine consumed dates (Oct23–27, Oct31–Nov3), **2,592 markets per arm/timing**,
  J/X. Fixed guard models trained on eight dates: exploratory historical policy comparison, not independent validation.
  Branch `research/predict-inventory-range-20261109`, checkout `/tmp/cta-inventory-range-20261109`; declared source
  `6d9369a5`, tested source/state `2b33793b`. Data/declaration/logs:
  `/home/yuqing/ctaNew/data/predict_fun/research/inventory_range_20261109/`.
- One new candidate `range10`: shared [-10,+10] inventory range for ALL orders, remaining room =
  max(0,10 - signed known net - all same-side known remaining reservations). At +10 with no pending sells, up to20
  total sell shares are allowed. Existing sell orders consume that allowance. Resting/pending-create/pending-cancel
  quantities count, with no opposite-side credit. Same retention, lot/notional checks, submission recheck and conservative
  cancel-confirmation +100ms fill-report grace +.001ms as strict both_reserved. No price/threshold/delay/model sweep.
- Only the aggregate reducing allowance differs from strict both_reserved. Original price guards, AM5/25 desired
  sizing, per-order RO60 clipping/rung withdrawal and S240 remain. No restart of MM cycles after60. Outstanding
  orders can collectively reverse within the range, including late; this is intentional bounded exposure.
  Total-loss budget uncapped,150-share backstop unchanged,rebates excluded.

| Timing / arm | Settlement $ | +1s markout $ | Daily PnL SD $ | Mean residual shares | Peak absolute shares |
|---|---:|---:|---:|---:|---:|
| J baseline | 514.06 | 602.83 | 134.51 | 6.557 | 75 |
| J adding-only reserved10 | 309.74 | 237.96 | 41.74 | 2.515 | 80 |
| J strict both_reserved | 149.35 | 101.40 | 33.50 | 1.803 | 10 |
| J allowed range10 | 147.09 | 121.83 | 36.98 | 1.667 | 10 |
| X baseline | 317.63 | 463.56 | 103.09 | 5.720 | 65 |
| X adding-only reserved10 | 253.05 | 176.21 | 43.28 | 2.252 | 48.5 |
| X strict both_reserved | 119.34 | 73.63 | 35.30 | 1.657 | 10 |
| X allowed range10 | 111.24 | 93.28 | 40.52 | 1.578 | 10 |

- **Mechanism works:** 56/46 direct sign-crossing fills J/X (2/1 after60), versus zero in the strict control. Every
  physical fill and independently reconstructed path remains inside +/-10 (tolerance1e-7); zero time above10.
  Thus this is an engaged reversal policy with a real aggregate bound, not a no-op or the previous soft CAP10.
- Versus strict both_reserved J/X: filled volume **+16.8%/+13.7%**, FIFO paired shares **+50.3%/+42.0%**,
  total +1s value **+20.1%/+26.7%**. +1s per100 filled shares rises **1.2571->1.2934 / 1.0382->1.1569**
  (+2.9%/+11.4%). Average residual inventory falls **7.5%/4.8%**, time-weighted mean absolute inventory
  **4.6%/2.1%**, inventory-square **4.3%/1.6%**. Volume differs; no matched-turnover execution-quality claim.
- **No observed settlement or PnL-variance improvement versus strict control:** total delta **-$2.2565/-$8.1008**
  (-1.5%/-6.8%), daily PnL SD **+10.4%/+14.8%**, worst day **-$42.09/-$62.11** versus **-$33.79/-$46.44**.
  Max drawdown **-$45.92/-$68.69** versus **-$45.93/-$53.70**. Lower average inventory alone does not establish
  lower realized PnL variance. All SD/tail/drawdown figures are descriptive on these nine dates.
- FIFO matched-trade PnL improves **-$78.66->-$66.07 J**, **-$62.24->-$47.17 X**, but remains negative.
  Residual FIFO PnL is **$213.17/$158.41**, summing with matched PnL to settlement totals. FIFO includes price drift
  between fills and is not pure spread. Mark-at240/carry is a different attribution: range **$118.02/+$29.07 J**,
  **$94.42/+$16.82 X**, versus strict **$154.72/-$5.37 J**, **$105.81/+$13.53 X**.
- Day concentration check versus strict control: settlement improves on **3/9 J**, **5/9 X**. Leaving one day out
  changes the total delta across **[-$38.25,+$11.13] J**, **[-$16.32,+$7.57] X**. Oct24 contributes +$36.00 J;
  Nov2 contributes -$15.68 X. No stable profit gain, and no parameter changes after these outcomes.
- Declared inference: 2,000 shared UTC-day sign flips +2,000 paired day bootstrap draws;18 comparisons (range vs3
  fixed references x J/X x settlement/+1s/inventory-square). Settlement mean-difference intervals versus strict:
  **[-$8.39,+$10.06] / [-$5.97,+$3.37] per day**, family p=1.0 both. All settlement intervals versus other controls
  also cross zero. +1s improvement versus strict has family p=.022 J/.064 X; q² reduction p=.059 J/.993 X.
  These are exploratory conditional comparisons after prior candidate selection, not untouched validation.
- Compared with prior adding-only reserved10, range keeps the hard bound and lowers mean residual inventory
  33.7%/29.9%, but observed settlement is 52.5%/56.0% lower and +1s value 48.8%/47.1% lower. Adding-only's larger
  profits came with peak positions80/48.5; allowed bounded reversals do not recover that whole profit difference.
- **36 preflight controls PASS**, eight-market smoke and all18 full day/timing reconciliations PASS. Fresh first-day
  baseline, adding-only and strict-control fills/requests/windows match prior artifacts exactly at J/X; remaining
  references checked by hash. Zero excluded markets, missing valuations or missing bounded Binance hours. All source
  and result artifact hashes, every per-run adapter hash, and identical data/review receipt copies verified.
  Peak replay RSS ~1.6GiB, one core;~490GiB disk free. All jobs complete.
- Receipt `reviews/inventory_range_20261109/receipt.json`; reproduce via `run_inventory_range.sh` after declaration.
  Conclusion: the allowed range restores some turnover and short-term value while enforcing the ceiling; it has
  not improved settlement PnL or daily PnL variance over strict closure in this sample. No live code/settings,
  new freeze, forward-race change, or protected Nov9–13 data read. Keep the result as research, not a live PnL claim.

## 2026-11-09 — inflight inventory-reservation experiment COMPLETE

- Main as-of **2026-11-09 07:39:37Z**, exact intent diagnostic **07:43:24Z**. All nine consumed dates
  (Oct23–27, Oct31–Nov3), **2,592 markets per arm/timing**, J/X. Fixed guards trained on eight dates:
  exploratory historical comparison, not independent validation. Uncapped loss budget, existing 150-share backstop,
  rebates excluded. Branch `research/predict-inflight-budget-20261109`, checkout `/tmp/cta-inflight-budget-20261109`;
  declared source `f0e15415`, main report source/state `26908059`, exact tie diagnostic `91f769dc`.
  Data/declaration/logs: `/home/yuqing/ctaNew/data/predict_fun/research/inflight_budget_20261109/`.
- Two new arms: `reduce_reserved` preserves original adding rules and caps combined reducing orders to
  max(0, |known inventory| minus all same-side known remaining reservations). `both_reserved` also applies the
  previous aggregate 10-share adding budget. Resting orders, pending creates and pending cancels count across lanes;
  no credit for opposite-side orders or unreported physical fills. Desired and submission quantities both checked.
- Both new arms retain eligible same-price reducing orders based on remaining quantity; both_reserved also retains
  adding orders this way. Existing lots/notional/price guards/RO60/S240 retained. Reservations persist through cancel
  confirmation plus the fixed 100ms fill-report delay +.001ms, because this simulator does not provide cumulative fills
  in the cancel acknowledgement. This is a conservative treatment component, not a verified live-OMS delay rule.

| Timing / arm | Settlement $ | +1s markout $ | Daily PnL SD $ | Mean residual shares | Peak absolute shares |
|---|---:|---:|---:|---:|---:|
| J baseline | 514.06 | 602.83 | 134.51 | 6.557 | 75 |
| J prior adding-only reserved10 | 309.74 | 237.96 | 41.74 | 2.515 | 80 |
| J reduce_reserved | 453.23 | 492.12 | 149.62 | 6.835 | 75 |
| J both_reserved | 149.35 | 101.40 | 33.50 | 1.803 | 10 |
| X baseline | 317.63 | 463.56 | 103.09 | 5.720 | 65 |
| X prior adding-only reserved10 | 253.05 | 176.21 | 43.28 | 2.252 | 48.5 |
| X reduce_reserved | 150.44 | 377.89 | 98.93 | 5.958 | 65 |
| X both_reserved | 119.34 | 73.63 | 35.30 | 1.657 | 10 |

- **Mechanism confirmed:** reducing-at-send orders that subsequently open reverse exposure fall to **zero** for both
  new arms at J/X, including the case where one fill reaches flat and a later fill opens the reverse position.
  Original baseline has 2,978.12 / 2,274.68 such opening shares in 171 / 129 markets J/X; prior adding-only reserved10
  has 3,410.20 / 2,548.06 in 179 / 132 markets. These are cumulative opening quantities, not peak positions.
- The initial diagnostic bounded same-time notification ambiguity. All nine uncertain fill tranches are in two Oct26
  markets. Instrumented replays at both timings match original windows/fills/requests exactly; exact known inventory
  at each of their order submissions is zero. They are intentional new entries, not excess reducing orders.
  `overshoot_diagnostic_v2.json` supersedes v1 in-band; both retained. Zero direct sign-crossing fills also observed
  in new arms, but that narrower metric alone would not establish the full mechanism.
- **Exit reservations alone do not reduce overall inventory risk here.** Versus baseline J/X: settlement -11.8%/-52.6%,
  filled shares -18.8%/-17.2%, residual inventory +4.2%/+4.2%, inventory-square +2.6%/+1.1%.
  Daily SD +11.2%/-4.0%. Peak exposure unchanged at 75/65 because original adding rules still permit those positions.
  Worst day -$117.26/-$90.17 versus baseline -$53.12/-$128.60. Thus some downside improves, other downside worsens.
- **Combined entry/exit reservations enforce the simulated 10-share bound**, checked on every fill and independently
  from saved paths (floating tolerance 1e-7). Zero time above10. Versus prior adding-only budget J/X:
  inventory-square -52.7%/-49.8%, residual -28.3%/-26.4%, daily SD -19.8%/-18.4%, but settlement
  **-51.8%/-52.8%**, filled volume -51.4%/-47.0%, +1s value -57.4%/-58.2%.
  Maximum drawdown improves from -$79.60/-$77.39 to -$45.93/-$53.70; worst day is slightly worse
  (-$33.79/-$46.44 versus -$30.04/-$45.96). No single uniformly dominant risk/profit result.
- Spread/trading value not improved: FIFO matched-trade profit J/X baseline $134.02/$173.54, prior reserved10
  -$33.20/$24.31, reduce_reserved $56.43/$40.38, both_reserved **-$78.66/-$62.24**.
  +1s per 100 filled shares baseline 1.7246/1.6162, reduce_reserved 1.7331/1.5905;
  prior reserved10 1.4324/1.3178, both_reserved 1.2571/1.0382. FIFO includes intervening price drift, not pure spread;
  total volume differs, so this is not an execution-quality comparison at matched turnover.
- Declared 2,000 shared UTC-day sign flips + 2,000 day bootstrap draws; 18 comparisons (three pairs x J/X x
  settlement/+1s/inventory-square). All settlement mean-difference intervals cross zero. Exit-only +1s losses have
  family p=.022/.045; combined versus prior adding-only q² decreases p=.016/.020 and +1s losses p=.006/.024.
  These are conditional exploratory comparisons on consumed dates; SD/tails/drawdowns descriptive, no live-return claim.
- **31 preflight controls PASS**, eight-market smoke and all 18 full checks PASS. First-date baseline and reserved10
  replays exactly match prior windows/fills/requests; remaining references verified by hash. Zero excluded markets,
  missing valuations, or missing bounded Binance hours. All result artifact/source hashes and receipt copies verified.
  Supplemental flat-then-reverse / event-tie positive controls and missing/shifted-event refusals pass; exact tie replay
  parity passes at J/X. Peak RSS ~1.6GiB, one core, ~490GiB disk free. No job remains running.
- Receipts under `reviews/inflight_budget_20261109/`: main `receipt.json`, initial `overshoot_diagnostic.json`, final
  `overshoot_diagnostic_v2.json`. Reproduction: `run_inflight_budget.sh`, then `pf_inflight_overshoot`, `pf_inflight_ties`.
  Conclusion: count outstanding quantities to control unintended reversals, but do not adopt strict zero-crossing
  reduction caps as a PnL improvement. A separate next experiment could allow an explicit bounded reverse inventory
  budget while still reserving all orders; not run here. No live settings/code, freezes, forward-race changes, or
  protected Nov9–13 data reads.

## 2026-11-09 — entry inventory-budget experiment COMPLETE

- As-of **2026-11-09 06:12:40Z**; supplemental peak trace **06:18:57Z**. Nine consumed dates (Oct23–27,
  Oct31–Nov3), **2,592 markets per arm/timing**, all J/X runs complete. Guards trained on eight dates:
  exploratory in-sample, not independent validation. Uncapped total-loss budget, existing 150-share exposure backstop,
  rebates excluded. Branch `research/predict-entry-budget-20261109`, checkout `/tmp/cta-entry-budget-20261109`;
  declared source `f7fc7162`, tested source/state `f2867829`. Data/logs:
  `/home/yuqing/ctaNew/data/predict_fun/research/entry_budget_20261109/`.
- `uniform10` caps new inventory-adding orders at 10 shares per order (existing 5-share targets remain 5).
  `reserved10` caps adding orders to the remaining 10-share same-side budget after known inventory and all known
  outstanding reservations, across all lanes. Adding means side-sign times known net >=0. Reducing sizes/guards
  unchanged, even if an order could cross zero. Both new arms keep eligible same-price adding orders when known
  remainder fits desired size/room, without upsizing; pending creates/cancels reserve capacity until terminal knowledge.
  Lots/minimum notional and a submission-time budget invariant apply. Baseline unchanged. No urgency/top-up/restart.

| Timing / arm | Settlement $ | +1s markout $ | Daily PnL SD $ | Mean residual shares | Mean time-weighted inventory squared |
|---|---:|---:|---:|---:|---:|
| J baseline | 514.06 | 602.83 | 134.51 | 6.557 | 168.162 |
| J uniform10 | 425.59 | 395.52 | 74.58 | 4.025 | 58.855 |
| J reserved10 | 309.74 | 237.96 | 41.74 | 2.515 | 30.388 |
| X baseline | 317.63 | 463.56 | 103.09 | 5.720 | 144.533 |
| X uniform10 | 281.63 | 292.05 | 64.93 | 3.475 | 46.731 |
| X reserved10 | 253.05 | 176.21 | 43.28 | 2.252 | 25.295 |

- Uniform cap retains **82.8% / 88.7% J/X** of observed settlement profit, with daily SD **44.6% / 37.0% lower**,
  mean residual inventory **38.6% / 39.2% lower**, and filled volume **29.7% / 32.0% lower**.
  Reserved budget retains **60.3% / 79.7%**, with SD **69.0% / 58.0% lower**, residual **61.6% / 60.6% lower**,
  volume **52.5% / 53.4% lower**. Inventory-square falls 65.0%/67.7% uniform, 81.9%/82.5% reserved.
  These describe the risk/volume tradeoff, not improved execution at matched activity or calibrated outcome variance.
- Both new arms reduce inventory-square AND total +1s value on every day at both timings. Paired day inference:
  2,000 shared sign flips + 2,000 day bootstrap draws; 18 comparisons (three arm pairs x J/X x settlement/+1s/q²).
  Settlement mean-difference intervals all cross zero; no reliable expected-profit ordering. All q²/+1s comparisons
  have max-T p=0.005997, conditional on these consumed days. Daily SD/tail/drawdown figures are descriptive.
- J/X maximum drawdown: baseline **-$293.73 / -$249.33**, uniform **-$148.76 / -$119.37**, reserved
  **-$79.60 / -$77.39**. Worst day: **-$53.12 / -$128.60**, **-$27.31 / -$59.83**, **-$30.04 / -$45.96**.
  Reserved is not uniformly better on every tail metric (J worst day worse than uniform).
- Spread capture has NOT improved: +1s value per 100 filled shares J/X falls from **1.7246 / 1.6162** to
  **1.6088 / 1.4964** uniform and **1.4324 / 1.3178** reserved. FIFO matched-trade profit J/X is
  **$134.02 / $173.54** baseline, **-$8.48 / $24.64** uniform, **-$33.20 / $24.31** reserved.
  FIFO matching includes intervening price drift, not pure spread; residual FIFO attribution is not carry-at-240.
  Both new arms share queue retention, so their comparison with baseline combines retention and size effects;
  reserved-versus-uniform isolates reservation sizing with common mechanics, but is not turnover matched.
- **Not a hard 10-share limit:** instantaneous peak absolute inventory J/X is **75 / 65** baseline,
  **80 / 48.5** for BOTH new arms. Post-hoc trace covers both tied global peaks (2 paths / 15,552 total paths,
  no exclusions): Oct26 J market 3818191, initial +10 inventory, then 90 sell shares filled through 10.546209s,
  reaching -80. Every sell order was submitted while known inventory was +10 (therefore classified reducing);
  50 sell shares filled while cancellation was pending. Send times match exact placement events; known inventory
  reconstructed with the recorded 100ms fill delay, no event-priority ties. Terminal inventory is flat.
  This confirms the remaining exposure route: unchanged reducing orders can reverse inventory, with several
  outstanding/cancel-pending orders across lanes. A hard limit must budget those zero-crossing portions too.
- **25 preflight controls PASS**; eight-market baseline smoke and all 18 full day/timing reconciliation checks PASS.
  Fresh Oct23 baseline at both timings exactly matches reference fills/requests/windows; other baselines reused
  with hashes. Zero excluded markets, missing valuations, or missing bounded Binance hour files. All input/source
  and per-run adapter identities verified. Peak diagnostic also passes positive cross-zero and known-bad missing-event /
  wrong-peak refusal controls. Peak replay RSS ~1.6 GiB, one core; no jobs remain running.
- Main receipt `reviews/entry_budget_20261109/receipt.json`; supplemental explanatory receipt
  `reviews/entry_budget_20261109/peak_diagnostic.json` (main receipt unchanged), reproduced by `pf_entry_peak.py`.
  Simplest per-order cap offers better observed profit retention; reservation budget offers greater average risk
  reduction. Neither establishes improved spread capture or a safe hard position ceiling. Next narrow experiment:
  reserve full potential exposure including reducing-order zero crossings while retaining useful reduction liquidity.
  No live code/settings, frozen artifacts, forward-race changes, or protected Nov9–13 data reads.

## 2026-11-09 — spread capture / inventory-risk review

- User asks how to improve spread capture while lowering inventory variance. Post-hoc descriptive risk calculation from
  the completed receipts (data as-of 04:15:18Z), nine consumed days / 2,592 markets per arm/timing, no exclusions/missing values:
  daily settlement PnL sample SD baseline -> urgency **134.5108 -> 127.9100 J**, **103.0877 -> 97.8480 X** (~5% lower).
  Worst day **-53.1211 -> -53.2061 J**, **-128.5982 -> -99.4829 X**. J also cuts a large winning day; X improvement remains
  concentrated on Nov2. No interval or significance claim. Receipt: `reviews/urgency_history_20261109/risk_diagnostic.json`.
- Code review: CAP10 is a threshold on current known net, not a reservation-aware 10-share bound on the next order.
  If |known net| <10, a 25-share adding order can still pass, and each ladder lane checks the same pre-fill position.
  Previously tested HCAP clips against the separate 150-share backstop; it did not test a strategic 10-share risk budget.
- Priorities to investigate: size/withhold inventory-increasing quotes against potential same-side fills across all lanes
  (including pending creates/cancels); retain effective, executable reduction sizes; isolate queue retention at matched
  target exposure; then test selective new MM cycles after returning near flat (RO60 currently prevents them after 60s).
  Budget any price concession for reduction against the risk removed; existing guards remain, no blanket late aggression.
- Evaluation should separate matched-pair trading value, adverse-selection markout, unpaired carry, and downside risk.
  Compare at matched trading activity where possible so merely trading less is not called better execution. No new variant,
  threshold sweep, live change or forward data read launched in this review.

## 2026-11-09 — earlier urgency check COMPLETE

- As-of **2026-11-09 04:15:18Z**. All four earlier complete compatible dates (Oct 23–26), **1,152 markets per arm/timing**,
  replayed at J/X; baseline versus the exact previous RPRICE rule. Branch `research/predict-urgency-history-20261109`,
  checkout `/tmp/cta-urgency-history-20261109`, declared source `10893e49`, tested source/state `a94a0dd7`.
  Receipt: `predictfun/reviews/urgency_history_20261109/receipt.json`; data/logs under
  `/home/yuqing/ctaNew/data/predict_fun/research/urgency_history_20261109/`. No study job remains running.
- These are previously consumed guard-training days. Urgency was selected after the prior four-candidate comparison;
  this is an earlier historical consistency check, not independent validation. Exact policy source remains `f014efbe...`:
  one-cent reducing-side improvement into the spread, 180<=t<240, |known inventory| >=10, original guards retained.
  Uncapped total-loss budget; existing 150-share exposure cap; rebates excluded. No parameter/model changes.

| Settlement PnL change versus baseline ($) | J | X |
|---|---:|---:|
| Oct 23 | +1.3408 | +3.4780 |
| Oct 24 | +6.0191 | -0.4361 |
| Oct 25 | -27.0320 | +0.3366 |
| Oct 26 | -1.0178 | -0.4778 |
| Earlier four days | -20.6899 | +2.9007 |
| Previous five days | +10.9085 | +25.7631 |
| All nine days, descriptive | -9.7814 | +28.6638 |

- Earlier baseline -> urgency settlement: **$691.6287 -> $670.9389 J**, **$433.5722 -> $436.4729 X**. Both timings
  improve on only 2/4 earlier days. Earlier +1s changes **-$0.0742 / -$0.6977**. Mark-at-240 changes
  **-$6.6740 / -$2.4265**; carry changes **-$14.0159 / +$5.3272**. Thus the small positive X result comes from carry,
  while both measured short-term and stop-mark value decline on these earlier days.
- Mean residual inventory falls **6.9581 -> 6.8818 shares J**, **6.2460 -> 6.2148 X**. Economically reducing volume rises
  **87.8751 / 35.9633 shares** over all 1,152 earlier markets. Time-weighted absolute inventory over 60–300s falls
  **7.5130 -> 7.4737 J**, **6.7139 -> 6.6960 X**. The reduction is real but small and not a reliable PnL gain.
- Combined nine days / **2,592 markets per arm/timing**: baseline -> urgency **$514.0559 -> $504.2745 J**,
  **$317.6259 -> $346.2897 X**. +1s delta **-$2.0719 / +$0.1535**; carry delta **-$12.7697 / +$25.0251**.
  J improves 6/9 days but Oct25 dominates the losses. X improves only 3/9 (one unchanged); remove Nov2's +$29.1153
  and its pooled gain becomes **-$0.4515**. The sign and concentration depend on timing and individual settlement days.
- Earlier inference: 2,000 predeclared shared UTC-day sign flips, four comparisons (J/X x settlement/+1s); all max-T
  family p values 1.0. **No confidence intervals on four days**; nine-day pooled points are descriptive after candidate
  selection, with no pooled significance claim. Prior four candidates / 16 comparisons retained in the declaration.
- All **22 controls pass**, including positive/refusal controls for scope, source identity, event identity and day nulls.
  Fresh Oct23 baseline parity exact at J/X; other baseline artifacts copied from the prior verified study with hashes.
  All eight day/timing runs have exact pre-180 fills/requests and per-market settlement/carry/inventory reconciliation.
  Zero excluded markets or missing valuations, zero missing bounded Binance hour files, zero ambiguous filled-order keys.
  Final receipt copy and source hashes verified. Peak replay RSS ~1.6 GiB on one core; ~493 GiB disk free.
- Clarification of previous stress interpretation, retained in this declaration: Nov3 urgency's settlement delta is
  identical with/without p99 cancel reservations at each timing. That negative day does not show incremental sensitivity
  to this delay stress. No additional stress sweep was used here.
- Conclusion: earlier data weakens the case for a general urgency PnL improvement. Keep this rule as research; do not
  enable it globally from these results. It slightly reduces inventory but can surrender favorable subsequent price moves
  and settlement carry. No live change, new freeze, forward-race change, or protected Nov9–13 data read.

## 2026-11-09 — passive inventory experiment COMPLETE

- As-of **2026-11-09 03:53:11Z**. All declared replays, reconciliation and 18 source controls passed. Isolated branch
  `research/predict-passive-inventory-20261109`, checkout `/tmp/cta-passive-inventory-20261109`, tested source `983a3b4f`.
  Receipt: `predictfun/reviews/passive_inventory_20261109/receipt.json`; retained data/logs under
  `/home/yuqing/ctaNew/data/predict_fun/research/passive_inventory_20261109/`. No study job remains running.
- Five consumed dates: Oct 27, Oct 31, Nov 1–3, **1,440 markets per arm/timing**; same fixed guards, trained on four of
  these five dates. This is in-sample policy feasibility, not independent validation or a live-return estimate. J uses
  transaction-frame jitter; X uses transactions shifted -50 ms and max activation queue. Main total-loss budget uncapped;
  separate 150-share exposure cap unchanged, rebates excluded. Baseline settlement **-$177.5728 / -$115.9463 J/X**;
  baseline +1s markout **$261.9645 / $183.1157** (a diagnostic, not realized spread PnL).
- Same-price queue retention after 60s keeps an eligible order only when its known remainder plus other reducing-side
  reservations fits known inventory. It can also retain a smaller order instead of upsizing. Small reduction caps desired
  size at 5 shares; combined may preserve an older larger original order whose remaining quantity fits 5. Urgency improves
  the reducing quote one cent into the spread during 180–240s for |known net| >=10. Existing guards must admit the original
  quote; their score is not claimed calibrated to the improved price. Price concession enters actual fill PnL.

| Change | Settlement delta J / X ($) | +1s delta J / X ($) | Mean residual shares J / X |
|---|---:|---:|---:|
| Baseline | 0 / 0 | 0 / 0 | 6.2363 / 5.2992 |
| Keep queue | -10.7803 / +11.1000 | -0.3580 / +0.1500 | 6.1357 / 5.2506 |
| Five-share reduction | -22.1177 / +17.7284 | -25.9673 / -21.8333 | 6.8862 / 5.8172 |
| Late urgency | +10.9085 / +25.7631 | -1.9977 / +0.8512 | 6.2272 / 5.2945 |
| Combined | -21.0283 / +19.2284 | -26.1020 / -22.3833 | 6.8674 / 5.8172 |

- Queue retention lowers place requests **2.5687% / 1.9782%**, and residual inventory **1.61% / 0.92%**, but settlement
  improves only 1/5 J and 2/5 X days. It remains an operational-efficiency candidate, not demonstrated PnL improvement.
- Five-share reduction cuts economically reducing volume **45.94% / 50.72%** and raises mean residual inventory
  **10.42% / 9.77%**. MIN_USD withholding decisions rise **81,403 -> 91,065 J**, **76,790 -> 86,246 X**; these are repeated
  decisions, not distinct orders. Both smaller fills and the existing $1 minimum limit reduction; no causal attribution
  of the full loss to either mechanism. Among the same pre-60 inventory population (674 J / 579 X markets), the fraction
  ending below one share falls **16.32% -> 9.35% J**, **12.95% -> 6.56% X**. Do not adopt a blanket five-share cap.
- Urgency improves 4/5 J days but only 1/5 X days (one unchanged). Its X gain disappears without Nov 2: **-$3.3522**.
  J gain decomposes to mark-at-240 **+$9.6623** / carry **+$1.2462**; X **+$6.0652 / +$19.6979**. Mean residual inventory
  falls only **0.0091 / 0.0048 shares**. It does not establish stronger spread capture or materially better inventory control.
- Declared 2,000 shared UTC-day sign flips + 2,000 day bootstraps, 16 comparisons. Urgency settlement family-adjusted p
  **0.5727 J / 1.0 X**. J pointwise bootstrap mean/day interval is positive, but coarse five-day randomization and family
  inference do not support a robust positive. No new parameter sweep or candidate adoption after outcomes.
- One-day Nov 3 stress: **288 markets per arm/timing**, cancel reservations delayed to at least measured p99 **143.3536 ms**,
  physical removal unchanged. Settlement deltas J/X: queue **+6.2466 / +0.3500**, small **+0.6981 / -8.5854**, urgency
  **-0.0850 / -2.9689**, combined **+1.8131 / -10.3354**. No interval; reservation sensitivity, not calibrated full OMS.
- Checks: exact baseline J/X smoke and fresh first-day baseline parity (also fresh Oct31 J); remaining main baseline
  artifacts reused with hashes/origin recorded; stress baseline always fresh. All pre-60 fill/request traces equal across
  arms; fill/settlement/carry and inventory paths reconcile per market. Zero excluded markets/missing valuations; no missing
  bounded Binance hour files. Guard withdrawals, pending reservations, lot/notional checks and stop remain active.
- Declaration v3 supersedes v2's filled-order diagnostic assumption: legitimate duplicate creation timestamps caused a
  checker stop. PnL/policies unchanged; counts now use immutable price/size lower bounds and sent-request/tranche upper
  bounds. Final observed bounds collapse: one duplicate create key per main arm/timing, zero ambiguous filled keys. Original
  failure log/declarations retained. Synthetic ambiguous-order positive and corrupted-identity refusal controls pass.
- All source/input/artifact identities recorded; receipt copy and source hashes verified. Peak replay RSS about 1.3 GiB,
  one core, ~493 GiB disk free. Frozen artifacts, live code/settings and protected Nov 9–13 forward dates unchanged/unread.
- Next: avoid the blanket five-share cap. Queue retention merits efficiency work; urgency needs independent evidence.
  Any further size policy should explicitly account for executable minimum notional and reduction throughput before a new
  predeclared test. Current results do not justify promoting any candidate to live.

## 2026-11-08 — sizing and settlement probability study COMPLETE

- As-of **2026-11-08 17:11:07Z**. All declared jobs and controls finished; no research job left running. Isolated branch
  `research/predict-sizing-settlement-20261108`, checkout `/tmp/cta-sizing-settlement-20261108`, tested source `beccfc63`.
  Full receipt: `predictfun/reviews/sizing_settlement_20261108/receipt.json`; retained inputs/outputs and logs under
  `/home/yuqing/ctaNew/data/predict_fun/research/sizing_settlement_20261108/`. This entry supersedes earlier running statuses.
- Main experiment: nine consumed days (Oct 23–27, Oct 31–Nov 3), 2,592 markets per arm/timing; baseline, capacity clipping,
  size hysteresis, combined; J/X; $150/uncapped total-loss budget. Fixed deployed guards were trained on eight of these days:
  **in-sample strategy comparison**, not a live-profit estimate. Continuous historical-clock basis, lot/notional checks,
  known-state reservations and refusal backoff are shared. No missing +1s or stop valuations in these results.
- Main settlement $ J/X: baseline **514.0559 / 317.6259**, capacity identical; hysteresis/combined **550.5360 / 339.7488**.
  Hysteresis place requests fall **7.7464% / 7.3397%**. +1s value changes **-10.9858 / -8.8662**; mark-at-240s changes
  **-68.4191 / -94.4639**; carry improves **104.8992 / 116.5868**. Thus settlement gains are from leftover inventory,
  not improved measured short-term value. Only 4/9 days improve at each timing; excluding Nov 2 changes the gain to
  **-92.5270 / -73.4372**. Observed drawdown falls, but no robust PnL benefit established and no candidate adopted.
- $150 is explicitly a **test scenario**, not a permanent user constraint. All 216 main capped/uncapped win/fill/request
  comparisons are exact. The separate 150-share exposure cap stays in both. Budget credits simulated window-end settlement
  and skips unstudied gaps; no live path/settlement-confirmation parity claim. Rebates excluded and never replenish budget.
- Cancellation diagnostic: same 105 Tokyo-5 windows, 13,775 cancels, 13,766 matched acknowledgements / 9 missing; completed
  delay p50 20.67 ms, p99 143.35 ms, max 14.97 s. Uniform-p99 reservation stress on Nov 3, 288 markets per arm, keeps original
  physical removal timing; fresh zero-PnL budget. Capacity lowers refusals but adds no PnL: uncapped baseline/capacity J
  -45.1963/-45.2518, X both -11.7216. With hysteresis, capacity J -45.2269 -> -49.6016, X unchanged -68.7496.
  The $150 budget changes some stress paths (both beneficial and detrimental). One day, no interval; sensitivity only,
  not calibrated OMS simulation. Rare longer/missing acknowledgements and full watchdog/OMS behavior remain unmodelled.
- Settlement forecast feasibility: initial train Oct 23–25; six chronological consumed test days, **1,728 markets /
  13,352 eligible neutral decisions**. Excluded: 453 missing market prices, 19 start-price-not-known. Each fit uses earlier
  venue outcomes known before the test day; no fill-selected population. Day-balanced Brier (lower better): market
  **0.189047**, Brownian/basis **0.192590**, calibrated market **0.189106**, market+physical logistic **0.190309**,
  fixed market-residual LightGBM **0.195683**. No alternative beats the market; nonlinear is worse on every test day.
  Added nonlinear family is explicitly post-initial-result exploration, not an untouched test or tuning search.
- Cost-aware shadow: 26,704 hypothetical 25-share Up/Down opportunities per model (separate alternatives, not strategy PnL).
  24,398 eligible; 1,854 insufficient arrival depth; 442 below notional; 6 stale; 4 invalid books. Calibrated-market mean by day
  +$0.01275 per opportunity, day bootstrap interval [-$0.11371,+$0.13730]; other models negative. No reliable value gain.
  500 matched action randomizations and 2,000 day null/bootstrap draws. Expanded reports supersede initial inference:
  max-T now uses sign-invariant RMS scaling; original point forecasts/valuations retained unchanged. See declaration v3
  and nonlinear_declaration_v3.json; five probability models / ten forecast comparisons, four shadow comparisons.
- Controls passed: 10 source tests (positive/refusal controls); eight-window exact frozen-engine parity; full 288-window,
  eight-arm cached/direct comparison exact across 24 artifacts; per-market fill-to-settlement reconciliation throughout;
  source/input hashes, label causality, probability oracle/anti-oracle, prior-offset and inventory-value controls.
  Peak observed replay/control RSS about 1.6 GiB, one core; ~499 GiB free disk. No large data added to git.
- Next: treat hysteresis as an efficiency candidate only; probability forecasts remain research outputs. Keep short-term
  adverse-selection guards separate from settlement valuation. Compare expected payout against executable exit value and
  inventory risk, not merely P(settle=1)>0.5. More independent future data and full execution calibration are needed before
  adopting a new controller. Live code/settings, frozen artifacts and the Nov 9–13 forward race are unchanged/unread.

## 2026-11-08 — sizing proposals and settlement probability study running

- Update as-of 17:02Z: all nine historical days complete at J and X. Baseline settlement $514.06 / $317.63;
  size hysteresis $550.54 / $339.75. The increase is carry-to-settlement: +1s value falls $10.99 / $8.87,
  mark-at-stop falls $68.42 / $94.46. Requests fall about 7–8%; capacity clipping leaves fills/PnL unchanged
  at ordinary timing. All are fixed-model in-sample results, not live return estimates. Full cache check,
  nonlinear forecast/inventory extension and cancel-tail stress are running sequentially; do not adopt yet.
- Update as-of 16:21Z: initial probability and inventory-shadow stages complete. Six chronological consumed test days,
  1,728 markets / 13,352 eligible neutral decisions: day-balanced Brier market 0.189047, Brownian 0.192590,
  calibrated market 0.189106, market+physical logistic 0.190309. No fitted model improves the market baseline.
  Cost-aware 25-share hypothetical inventory opportunities: physical/model extensions negative mean exit value; calibrated
  market +$0.0128 per opportunity by day, day interval spans zero. Not actual strategy PnL; see settlement_report.json and
  inventory_shadow_report.json for statuses, matched controls and source hashes. Eight-window frozen-engine exact parity passes.
  Full sizing replays running sequentially. One additional fixed-parameter market-residual LightGBM is declared separately
  AFTER these results (nonlinear_declaration_v2.json); not tuned/validated on unseen data, and not run yet.
- User clarified $150 is the current test setting, not a permanent strategy constraint; authorized testing sizing proposals
  and settlement probabilities. Live settings remain unchanged; research compares $150 and uncapped total-loss scenarios.
- Isolated checkout `/tmp/cta-sizing-settlement-20261108`, branch `research/predict-sizing-settlement-20261108`.
  Declaration and artifacts: `data/predict_fun/research/sizing_settlement_20261108/` (`declaration_v2.json` supersedes v1).
  Source commits start `fe41ccb4`; current pipeline through `48bda856`. No frozen files or Nov 9–13 inputs changed/read.
- Nine consumed days (Oct 23–27, Oct 31–Nov 3): baseline / capacity clipping / size hysteresis / combined, J and X, both loss
  budgets. Common 150-share backstop, known-state reservations, 0.01-share floor, minimum notional and risk retry backoff.
  Fixed measured timing approximation; not a full independent OMS/live parity claim. Fixed deployed guard models mean these
  strategy replays are in-sample. Continuous historical-clock feature rebuild finished; prediction preparation running.
- Settlement study: initial train Oct 23–25, expanding earlier-only training and six later consumed test days. Neutral
  market/time actions; market price, existing Brownian/basis probability, calibrated market logistic, full logistic.
  Labels must be venue outcomes and known before the next test day. Day-unit uncertainty, 2,000 null/bootstrap draws,
  matched market/calibrated-market baselines. Separate cost-aware hypothetical inventory test; not strategy PnL.
- Seven source/control tests pass; full engine parity and replay reconciliation still pending. Run only one research job
  through run_light at a time. Do not launch a duplicate study or treat pending experiments as approved strategy changes.

## 2026-11-08 — strategy PnL optimization review and cancellation audit

- Reviewed latest local `pm_ladder_v3_basis_v1` strategy/OMS/risk code (basis-tokyo-1 runtime; review checkout
  `109fa4a`) and existing spread, inventory, hedge, incremental-size and same-window reports. Tokyo's installed version
  was not observed. Continuous feature/quote parity is established; changed-state continuous PnL and full OMS parity are not.
- New descriptive audit as-of **2026-11-08 15:51:46Z**, same 105 Tokyo-5 markets (Nov 7 17:40Z–Nov 8 02:25Z):
  17,913 place requests, 13,775 cancels, 537,056 decisions, reconciled with the prior ledger/decision counts. Cancel reasons:
  11,043 withdraw quote; 1,019 same-price size increase; 584 same-price size decrease; 931 price only; 198 price and size.
  Size-only cancels are 11.637%; this is request attribution, not avoidable cancellations or a PnL estimate. Withdrawals pool
  model/book/inventory/stop/watchdog reasons. No missing same-time decisions. Excluded explicitly: 65,180 noncohort decisions,
  2,232 places and 1,598 cancels. Two partial days only, no confidence interval or inference.
- Evidence/code/declaration/controls: `/home/yuqing/pmmm-data/review-exports/strategy-opportunities-20261108/`.
  `quote_churn.py` detects synthetic size changes/withdrawals and refuses wrong identities/unchanged intents; all four controls
  passed. Run via run_light, one process, 30 MiB RSS, 2.13 s. Full input/code SHA256s in result.json.
- Recommended sequence (hypotheses, no adoption): execution-realistic baseline with continuous features + actual latency/caps;
  capacity-aware sizing and queue preservation; calibrated quote value by lane/price/time; separate settlement-value/exit-cost
  inventory controller. Current model predicts sweep-weighted 1s value, not settlement probability. CAP10 is a soft threshold,
  not a 10-share position guarantee; 5/25 sizing can overshoot. Keep guarded passive reduction and the $150 budget.
- Existing nine-day results remain selected in-sample / legacy basis. V3 retains roughly v1's +1s PnL and lowers measured risk;
  settlement uplift is unresolved. Prior INCSIZE two-day test did not establish PnL improvement; blanket taker flattening is costly.
  Preserve separate rebate reporting. No runtime/frozen artifact/forward race changed; Nov 9–13 remain untouched.

## 2026-11-08 — continuous research/Rust boundary parity completed

- Explicit continuous research variant and receipt pushed on `review/predict-continuous-basis-20261108` at `d65686b5`
  (source `4f7cf752`; isolated checkout `/tmp/cta-continuous-basis-20261108`). Current frozen/default modules are unchanged.
- On the same 105 complete Tokyo windows (Nov 7 17:40Z–Nov 8 02:25Z), all 32 features match on 494,637 rows;
  all 498,422 guard values and 537,056 conditional quote decisions match. Hour-boundary subset: 12,189 feature rows /
  12,558 decisions, zero mismatches. Frozen-research control detects 2,198 feature-row / 72 quote differences at boundaries.
- pmmmrust tools/report pushed at `109fa4a`, branch `review/pfmm-continuous-research-parity-20261108`;
  receipt `docs/receipts/20261108_pfmm_continuous_research_parity_v3.json`. Runtime remains the already tagged basis correction.
- As-of 2026-11-08 13:04:44Z; evidence `/home/yuqing/pmmm-data/review-exports/continuous-parity-20261108/`.
  Captured known-time inputs/legacy state only: no changed-state fill/PnL evaluation or deployment. Nov 7–8 consumed;
  Nov 9–13 untouched, no freeze/forward-race change. Research must explicitly select the separate `_basis_v1` rows.

## 2026-11-08 — Tokyo-5 same-window diagnostic (user-authorized)

- **Data consumption:** 2026-11-07 and 2026-11-08 have been read for the live comparison, including required warm-up and
  finalization hours. They must not be called untouched validation in the older 11-04..11-08 v1/v2 race. The v3 11-09..11-13
  block remains untouched. No frozen engine/model/receipt was edited and no fitting or strategy selection was performed.
- Compared all 105 complete Tokyo-5 windows, 11-07 17:40Z through 11-08 02:25Z. Live PnL excluding rebates -7.0128;
  frozen v3 J -57.7727, X -19.8598; shares 293.635 / 556.753 / 459.065. Two partial days, descriptive only.
- Conditional original-Python quote logic matches 537,056 live decisions; native LightGBM matches 494,637 Rust feature rows.
  Independent feature reconstruction finds basis/gap differences at hourly boundaries, changing 42 conditional quote decisions.
  Full independent request/OMS scheduling remains unrun; this is not an overall parity or live-readiness pass.
- Code and superseding receipt: pmmmrust branch `review/pfmm-same-window-local-20261108`,
  `docs/PREDICT_SAME_WINDOW_LOCAL_COMPARISON_20261108.md` and
  `docs/receipts/20261108_pfmm_same_window_local_comparison_v1.json`.
  Full local evidence: `/home/yuqing/pmmm-data/review-exports/tokyo5-local-comparison-20261108/`.

## 2026-11-07 08:20Z (session 62f3ce33) - Codex v3 port review complete (verified)

- pmmmrust 3366e61 (receipt docs/receipts/20261107_pfmm_ladder_v3_review.json; tested code 173f53c). It names our receipt by its
  current hash; its flip audit equals ours (4 / 3 of 2592 windows, <= 5.04 sh). No launch, deployment, tag or forward read.
- Not done yet: Tokyo shadow / real-venue test of v3. Live = user's call. Forward block 11-09..11-13 still pending (after 11-14 02Z).

## 2026-11-07 08:02Z (session 62f3ce33) - Codex v3 port verified; v3 receipt's no-flip sentence is wrong

- Codex ported v3 (pmmmrust b85a401, ab40361); 11-03 J / X trace hashes = i2J / i2X (checked here). Port fixes (min 1 sh with the
  1 USDT check; expiry +120 s grace) match dev.predict.fun's place-orders rules (read here). No live launch.
- v3 receipt's 'the position never flips' is wrong (STATUS same time): flips occur in the replay in 4 / 3 of 2592 windows (J / X),
  <= 5 sh, mostly via cancel-pending orders; already inside the in-sample and forward numbers. Receipt not edited.
- Open (user): a correcting receipt for the description only, and/or a v4 with a hard no-flip guard (new replay + freeze + days).

## 2026-11-07 06:18Z (session 62f3ce33) - pm_ladder_v3 FROZEN (v1 + CAP10 + RO60 + S240)

- Receipt predictfun/freeze/pm_ladder_v3/RECEIPT.md, code commit 4ecaca69; engine parity PASS (STATUS 06:18Z).
- Forward check: own block 11-09..11-13; run `predictfun/research/run_forward_v3.sh` by hand after 2026-11-14 02:00Z (no timer).
  The 11-04..11-08 race (run_forward.sh after 11-09 02:00Z) is unchanged; v3 is not in it.
- Live port would need: 10-share add limit on known net (touch and rung), 60-240 s reduce-only phase (reducing side only, size <=
  |known net|, no rung, minute 1-3 lines, all guards), stop at 240 s, guard values to 240 s (port not checked here). Live = user's call.

## 2026-11-07 04:34Z (session 62f3ce33) - CAP10 + RO60-240 tested (in-sample)

- pf_inv2.py / run_inv2.sh -> step4/report_inv2.json, checks PASS (STATUS 04:34Z). Same +1 s trading profit as v1; day sd -37 / -48%,
  worst day and max drawdown about a third of v1's, $150-cap halts 2 / 0%. Settle +$25-30/day vs v1, not resolved.
- RO60+S290 with the line is not runnable (no guard values after 240 s); would need a model refit to 300 s.
- Open (user): freeze CAP10+RO60-240 for untouched days or declare it into the 11-04..11-08 race before the 11-09 02:00Z run; live
  port would need CAP10 and the 60-240 s reduce-only phase with the minute 1-3 lines (port not checked here).

## 2026-11-07 03:40Z (session 62f3ce33) - taker hedge of the leftover

- pf_hedge_check.py -> step4/report_hedge_v1.json (STATUS 03:40Z). Buying Down = selling Up on Predict (one book); passive sell-down
  already does it. Taking at the stop costs ~1.2 c/sh at 60 s, ~0.95 c/sh at 240 s; a full hedge removes most of the risk and about
  two thirds of v1's 60 s trading profit. Partial (cut to 25 sh) costs little; CAP10 + H25 and RO60-240 + H25 are the in-sample sweet
  spots. Not tested: a hedge during the window (changes later decisions: needs an engine token and a replay).
- Open (user): candidate(s) to freeze / declare before the 11-09 02:00Z forward run.

## 2026-11-07 03:04Z (session 62f3ce33) - spread capture by arm

- pf_spread_capture.py -> step4/report_spread_capture_v1.json (STATUS 03:04Z). All arms capture the same ~1.2 c/sh spread; the pull
  line is the edge (1 s move +0.28 / +0.09 c/sh vs real top-5 makers -0.85). Arms that let the reducing side ignore the line lose
  that edge on those fills. RO60-240 (sell down to 240 s, line kept) is the only arm adding +1 s $ at both timings after the 20-way
  adjustment (+10.1 / +6.0 $/day). CAP10 cuts risk but costs +1 s volume.
- Open (user): which arm, if any, to freeze for untouched days or declare into the 11-04..11-08 race before 11-09 02:00Z
  (candidates by this view: RO60-240; by risk: CAP10).

## 2026-11-07 02:56Z (session 62f3ce33) - CAP10 attribution (Codex), verified

- Commits 96d97a31 / 13e36621; report step4/cap10_attribution_20261107/report.json. Totals, robustness stats and p-values recomputed
  here and equal. CAP10's mark@stop equals v1's because less spread is offset by less adverse drift; its settle gain is carry, last 4
  days only, p 0.42 / 0.65 raw, ~1.0 after selection. Risk lower 9/9 days. STATUS 02:56Z.
- Open (user): freeze CAP10 for untouched days, or declare it into the 11-04..11-08 race before the 11-09 02:00Z run, or neither.

## 2026-11-07 02:37Z (session 62f3ce33) - INCSIZE 2-day check DONE (Codex), verified

- Committed 73ba8594 / 2dcb5985 / 4c84e17c; report step4/incremental_20261107/report.json. v1 parity 4/4 byte-identical, totals
  recomputed here. Requests -4.4 / -4.7%, mark@60 s +6.0 / -0.5 $ (J / X, 2 days), settle gain is carry. Nothing to adopt; the slot is
  free (no Codex job running). STATUS 02:37Z.
- Supersedes the 02:27Z "in progress" note: the Codex files may now be edited normally (commit the change; the result cites 2dcb5985).

## 2026-11-07 02:27Z (session 62f3ce33) - Codex handoff: INCSIZE variant (running elsewhere)

- Codex built an opt-in INCSIZE order-management variant (pf_step4_ladder_incremental.py, builder pf_ladder_incremental_build.py,
  planner pf_incremental_size.py; receipt predictfun/reviews/incremental_size_20261106/receipt.json, 192 source-only tests). Hashes
  verified 02:27Z; frozen engine and race files unchanged. Details: STATUS 02:27Z.
- IN PROGRESS (Codex, not this seat): v1 vs v1+INCSIZE on 10-24 and 11-01, J and X, one run_light job at a time. Do not launch
  duplicate jobs or edit the 6 Codex files until Codex sends the result artifact.
- Watch out: the Codex files are untracked; commit them (with the result's run tags) before citing any number. Two consumed days =
  point estimates, no interval. Entering the forward race would need a superseding race receipt before the 11-09 02:00Z run.

## 2026-11-06 17:15Z (session 62f3ce33) - D1 inventory mechanisms on v1 (in-sample)

- Engine pf_step4_ladder_inv.py (generated, pf_ladder_inv_build.py; frozen engine untouched), runner run_inv.sh, checks
  pf_inv_check.py (PASS), report step4/report_inv_v1.json; STATUS 17:15Z has the table. No arm beats v1 with an interval excluding 0.
- CAP10 (stop adding at 10 sh) halves max drawdown and cuts $150-cap halts 27 -> 11% (J), 10 -> 3% (X); mark@stop about equal.
  Selling down after 60 s with the line exempted (RO60-240+RX5, D1ALL) halves day sd but loses 35-50% of settle; skew and the $ limit do
  ~nothing.
- Open (user): add CAP10 (or another arm) to the forward race as a declared candidate (superseding race receipt) before the
  11-04..11-08 read; capped replay with the port's halt rule; run_forward.sh after 11-09 02:00Z.

## 2026-11-06 15:23Z (session 62f3ce33) - $150 loss cap vs round-i simulation (read-only)

- Frozen v1 replay has no loss cap; the live $150 cap binds (refuses places) within 1 day from ~54% of simulated starts and the
  uncapped path reaches the port halt within 1 day from 27% (J) / 10% (X). Report step4/report_cap150_round_i.json; STATUS 15:23Z.
- Open (user decision): capped replay of v1 on the 9 round-i days with the port's halt rule, to price the cap. Needs a generated
  engine variant (frozen engine untouched).

## 2026-11-06 09:23Z (session 62f3ce33) - FROZEN pm_ladder_v2; forward race v1 + v2 declared

- Receipts: predictfun/freeze/pm_ladder_v2/RECEIPT.md, predictfun/freeze/forward_20261104_08/RECEIPT.md (supersedes v1's forward
  procedure; v1 rule unchanged). Never edit either; changes = new version.
- NEXT: after 2026-11-09 02:00Z run predictfun/research/run_forward.sh (sequential run_light jobs; ~2-3 h), then read
  step4/report_forward_20261104_08.json. Nothing may read 11-04+ before that. The +1 s vs top-5 comparison is computed after the rule.
- Live port: v2 would need a daily refit on this box + daily delivery of 2 models + 1 lines file; not in scope until the user decides.

## 2026-11-06 08:14Z (session 62f3ce33) - line stability + week split (read-only)

- Fixed pull line quotes 6-8% of minute-0 time on weekdays, 9-18% on weekends. Both arms earned far less in week 2 (10-31..11-03)
  than week 1 (10-23..27), same weekdays: candidate J +92.6 -> +8.4, X +76.9 -> -0.4 $/day. v1 may fail its forward check if this
  persists. Open user decision: a rolling-refit v2 raced beside v1 (must be frozen before any 11-04+ read).

## 2026-11-06 07:04Z (session 62f3ce33) - port: lockstep equal; guard values 1,727,227 / 1,728,000 bit-equal

- v1 known issue: the guard models split on bn_flow noise (|thr| < 1e-9); the 773 rows differ by cumsum residuals only. v2 item: round
  the flow features before training. Nothing to change in v1. Port next: live wiring (pfmm feedback, holds, shadow).

## 2026-11-06 06:26Z (session 62f3ce33) - lockstep trace for the exec port DONE

- data/predict_fun/research/step4/trace_pm_ladder_v1/README.md: 9 traced windows (8 x 11-03 J, 1 x 10-25 X own-cross), byte-identity with
  round i PASS. Regenerate: pf_ladder_trace_build.py, then pf_step4_ladder_trace.py with PF_TRACE_DIR / PF_TRACE_MIDS; check with
  pf_ladder_trace_check.py. The port session (ctanew-5c) builds model / feature parity itself; it may ask for more windows or fixtures.
- The 377-share queue figure was not live (replay definition, void L2c arm). No live queue measurement exists.

## 2026-11-06 06:09Z (session 62f3ce33) - FROZEN pm_ladder_v1; handed to the exec-port coding session

- Receipt predictfun/freeze/pm_ladder_v1/RECEIPT.md (never edit; changes = v2 receipt). Code commit 8d5b02a0; artifacts + receipt in the
  next commit. Builder pf_freeze_pm_ladder_v1.py (guard models saved; parity with round i exact).
- Forward check declared in the receipt: 11-04..11-08, PASS iff candidate sum > 0 at J and X and candidate - control > 0 at J and X.
  Needs: step1 / features / guard values for those days rebuilt with the frozen commit (frozen models, no refit), then ladder replays.
  11-08 completes 11-09 00Z. Not started.
- Hand-off message to the coding session ctanew-5c (tmux rust): implement pm_ladder_v1; earlier research numbers VOID (sent after
  the freeze commit; the send is recorded in STATUS).

## 2026-11-06 05:09Z (session 62f3ce33) - refresh decomposition (section 59) + ROUND-i SUMMARY (section 60)

- pf_mm_decomp_fw.py: section 57's decomposition on the refreshed (fwJ / fwX) and grid runs; all checks PASS. Refresh trades a little
  spread for less 1 s adverse selection at 5/25, net within noise; vs top 5 unchanged; last minute still loses after 1 s.
- Section 60 = the round-i summary (levels, lever table, 7 findings, standing candidate). Read it first for the current picture.
- Standing candidate (NOT frozen): 5/25 S60 per-minute lines + ladder, V0 + GS on the 50 ms grid; control today S60.
- Open for the user: forward replay (11-04+) of a frozen candidate; telling the exec-port session its research numbers are VOID.

## 2026-11-06 04:17Z (session 62f3ce33) - whole-window feed refresh DONE (docs section 58): adds nothing past 65 s

- Every-frame guard refresh over 0-240 s vs the 50 ms grid, fixed live engine round i, in-sample, 5/25 and 20/100, J and X, S60..S240.
  Inputs parity and gate PASS. No MM interval above 0 except s5/S60 at X (the known first-65 s result); 5/25 +1 s c/sh slightly higher
  in every minute (direction only), fewer shares filled; 11-03 worse in 14/16. Not adopted.
- Not measured: post-1 s loss by minute on the refreshed runs (pf_mm_decomp could be pointed at the fwJ / fwX dirs).
- Open for the user: forward replay (11-04+) of a frozen fixed-engine candidate; telling the exec-port session its research numbers are VOID.

## 2026-11-06 02:41Z (session 62f3ce33) - whole-window feed refresh RUNNING (declared STATUS 02:38Z) [done 04:15Z, see above]

- Correction filed: the round-i fresh-refresh result is 2 of 18 comparisons excluding 0 (today arm, bound timing, +12 $ / day), not 1.
- Running: research/run_fresh240.sh inputs -> ctrl -> fullJ -> fullX (log data/predict_fun/research/logs/run_fresh240.out, ~2.5 h).
  Every-frame guard rows over 0-240 s (rows_rl_d2_pk_ev240; first 65 s = the 65 s build, checked on 10-24), preds_tf60 -> preds_t60kF
  (merge refuses unless its first 65 s = preds_t60kE). Arms S60..S240 x 5/25 AV, 20/100 A1, J and X, tags fwJ / fwX; gate fw_ctrl_cmp
  (S60 refreshed-240 = refreshed-65, S240 refreshed != grid). Then read step4/report_fresh240.json, write docs section 58.

## 2026-11-06 02:11Z (session 62f3ce33) - pre-fix logs deleted; MM vs settlement + top makers + window diagnosis DONE (docs section 57)

- USER "2 first": 914 pre-fix replay logs deleted (kept 289: round-i / invwin / repair_smoke_v5 + data / model / market-only logs).
- pf_mm_decomp.py splits every fill into spread / informed (1 s) / drift (to stop) / carry (stop -> outcome) and compares +1 s per minute
  with the top makers' real fills; reconciles to the engine on every arm-day. Result: MM edge is spread minus 1 s adverse selection; the
  settlement part is outcome noise; longer windows add spread but lose it to adverse selection and drift; the last minute loses after 1 s
  for every maker band and every arm of ours.
- Open for the user: forward replay (11-04+) of a frozen fixed-engine candidate; telling the exec-port session its research numbers are VOID.

## 2026-11-05 21:52Z (session 62f3ce33) - inventory x window DONE (docs section 56): neither lever adds money

- Fixed live engine round i, in-sample, 5/25 and 20/100, J and X timing. Gate PASS (never-trigger controls equal, CAP5 falsifier differs);
  s5 S60 = ladder today (V0 + GS) 9 / 9 days. 36 comparisons per timing: none excludes 0 on mark@stop or settle. CAP / reduce-only cut
  carry sd on every read day (risk only); longer windows raise settle point estimates (carry), wide intervals; bound 20/100 +1 s
  markout < 0 for W >= 120 s. Neither lever adopted.
- Open for the user: forward replay (11-04+) of a frozen fixed-engine candidate (per-minute lines + ladder family is the only
  in-sample positive at both timings, section 55); whether to delete the logs that still print pre-fix numbers; telling the exec-port
  session its research numbers are VOID.

## 2026-11-05 17:35Z (session 62f3ce33) - round-i ladder DONE (docs section 55); inventory x window RUNNING

- Fixed engines, in-sample: per-minute lines + ladder positive at both timings (pm+ladder +63.6 J / +52.0 X $ / day; ladder adds
  above 0 at both); 11-03 negative almost everywhere. No forward day on the fixed engines yet.
- Running: run_invwin.sh ctrl -> fullJ -> fullX (log data/predict_fun/research/logs/run_invwin.out), ~2.5 h; then read
  step4/report_invwin.json and write docs section 56.
- Open for the user: forward replay (11-04+) of a frozen fixed-engine candidate - the only out-of-sample check available.

## 2026-11-05 17:20Z (session 62f3ce33) - PRE-FIX RESULTS CLEARED; C2 stopped; inventory x window test declared

- USER: clear the results from before the fix and use only the fixed engine. Deleted 163 pre-fix replay outputs (4.8 GB; list in STATUS
  17:17Z), stopped C2 (pf_step4.py never got the fixes) -> C1 / C2 VOID; docs/EXPLORATION banner + 33 VOID / 12 PARTLY VOID sections.
  EVERY replay number in the HANDOFF entries below (before 15:30Z) is VOID - do not quote or compare against it. kJ / kX are gone:
  the round-i baseline is kJi / kXi (C1-C4 S60, 8 read days $ / day: J 49.6 / 140.2 / 96.3 / 45.7, X -0.4 / 20.8 / 29.3 / 17.9).
- Declared (STATUS this time): run_invwin.sh ctrl -> fullJ -> fullX + pf_invwin_report.py (inventory CAP / reduce-only x window
  60-240 s, sizes 5/25 AV and 20/100 A1, both timings). Starts after the round-i ladder chain ends (fullX + report).
- Logs in data/predict_fun/research/logs still print pre-fix numbers; they are not results.

## 2026-11-05 15:30Z (session 62f3ce33) - R7 / R8 fixes verified; round-i chain RUNNING

- User fixed R7 / R8 and asked for a final review, then the tests. Review (diff vs /tmp/predictfun-repair-loop-20261105/*.before) + synthetic
  check `reviews/simulation_20261105/check_fixes_v4.py` PASS, falsifiers fire on the pre-fix files (STATUS 15:30Z). Minor non-split path note
  in STATUS (no planned run uses it).
- Running: run_ladder2.sh stages fix1 -> view -> fix2 -> par -> smoke -> fullJ -> fullX with round suffix i (kJoi, viewi_*, kJi / kXi, pari,
  ownoffi, smki, lJi / lXi, report_ladder2_i.json). Log data/predict_fun/research/logs/run_ladder2_i.out.
- Next: read fix1 PARITY (must PASS), view PASS, ownoffi own_cross > 0, kJi / kXi (corrected C1-C4), report_ladder2_i.json,
  docs section 55. (Done 17:0xZ: all gates PASS.) Watch out: C2's driver launches 11-05 at ~11-06 02:10Z via run_light, which refuses while a pf_ job runs - the chain
  (~2.5 h) must be done by then.

## 2026-11-05 15:15Z - repair/check loop COMPLETE: reported R7/R8 cleared; 145 checks + integration smoke pass

- User requested an autonomous repair loop. Superseding receipt: `reviews/simulation_20261105/receipt_v5.json`. Re-run the guarded
  regression gate with `research/run_simulation_checks.sh`; tests live in `research/tests/test_simulation_causality.py`.
- BOTH engines now keep client-visible cancel reservations independent of physical fills, including canceled pending creates.
  Reservations release on observable terminal acknowledgements. The known-view loss cap also uses notified cash/inventory and
  client-visible open quantities. Physical matching/accounting and explicit physical regression controls are preserved.
- `pf_view_check.py` refuses missing/empty/mismatched run manifests, incomplete arms/files, wrong cohorts, invalid event data and
  counts, and invalid notification delays; complete zero-fill arms remain valid. Its limited early-replace diagnostic is documented;
  full causal request-stream checks (including cancel-pending fills and throttling) are the source-only regression gate.
- Repair passes: 143 pass / 2 fixture failures (own-cross fixture corrected), then 145/145 pass twice. Earlier fixed/off controls
  pass. Coverage includes both sides, partial/full fills, notification delays, request pressure, loss caps and sequential variants.
- Integration only: first 8 already-read 10-24 windows, tag `repairloop_v5`: live and ladder base fills + requests identical; engaged
  ladder accounting / own-cross checks pass; the checker reads both real smoke runs. Comparator corruption and missing-run controls
  pass. Source in `reviews/simulation_20261105/repair_smoke_v5.py`; results in its JSON. Peak RSS 1,432 MiB, no forward days read.
- No confirmed reported code findings remain under these gates. Full economics / ladder grids still need fresh output tags on the
  repaired bytes; earlier kJh / kXh / lJ / lX are not revalidated. Fill calibration / own impact remain model limits.

## 2026-11-05 14:59Z (session 66f8e5c3) - round-3 chain done, SUPERSEDED by review 4 (R7 / R8); HOLD until the user hands back

- Done: chain br79kqi74 on the round-3 engines (kJh / kXh, parh PASS, ownoffh own_cross 1, smoke, lJ / lX + report_ladder2.json:
  recon 0 / 324, cross-check vs kJh C4 equal on 9 days). Every number in it is provenance only: R7 (throttle open_n reads the
  matcher's remaining size of cancel-pending orders) affects every arm, base included (all use XV). Do not quote kJh / kXh / lJ / lX.
- Hold: user said not to run tests now; they clear R7 / R8 themselves, then hand back. Pointers for R7 in STATUS this time.
- Next when handed back, in order: (1) fix1 regression with all old switches incl. PF_POLICYVIEW=physical (must reproduce the old
  throttle exactly); (2) view check with the fixed checker (must refuse missing / empty / mismatched runs); (3) a throttle
  look-ahead check that fires on the physical view and is 0 on the known view; (4) run_ladder2.sh fix2 -> par -> smoke -> fullJ ->
  fullX, then read kJ* / kX* vs kJ / kX (corrected C1-C4 baseline) and report_ladder2.json, write docs section 55.
- Watch out: run_ladder2.sh hardcodes tags kJh / kXh / parh / ownoffh and lJ / lX, and pf_ladder2_report.py:24 cross-checks against
  kJh - bump the tags (e.g. kJi ...) and LIVEKJ together, or the rerun overwrites lJ / lX and checks against the stale kJh.
  Results stay in-sample (lines and guard fit on the 8 read days; 11-03 beside); compare with the negative ladder test docs 36-38.
  Freeze / forward replay of C4 on 11-04..08 still waits for the user.

## 2026-11-05 14:45Z - review 4: direct rung leak fixed; throttle leak in BOTH engines + checker refusal OPEN

- Supersedes v3: `reviews/simulation_20261105/receipt_v4.json` + `check_v4.py`, via `research/run_light.sh`. Source-only synthetic
  review plus temporary synthetic parquet fixtures; no market data read, no simulator changes by reviewer; history impact unmeasured.
- All earlier fixed/off controls pass. Original R6 rung path is fixed, including full guard stack and L_FILL=1000; physical-view
  control still fires. R6 remains incomplete through R7 below.
- P1 R7 BOTH engines: `open_n` still counts cancel-pending orders via physical `rem`. With the known view + actual XV/44 ms answer
  cap, hidden fill at 343 ms (notification 443 ms; cancel effect 380 ms) enables a placement at 353 ms. Identical public-input no-fill
  branch does not place then; all newly received public frames are stamped before 343 ms. Without XV the pre-notification requests
  match. Use policy-visible lifecycle state for request reservation/authorization, including cancel-pending orders.
- P2 R8: actual `pf_view_check.py` main reports PASS / exit 0 for a nonexistent known run (physical positive fixture flags one early
  replacement). Existing known-bad fixture correctly fails. Refuse missing/empty/mismatched artifacts; compare full causal request
  streams, including cancels and cancel-pending fills, rather than relying only on its current early-replace count.

## 2026-11-05 12:40Z - review 3: prior fixes pass; ladder reacts to unreported fills (R6 OPEN)

- Supersedes review v2: `reviews/simulation_20261105/receipt_v3.json` + `check_v3.py` (run through `research/run_light.sh`). Source-only
  synthetic review, no market data read by checker, no simulator changes by reviewer; historical impact unmeasured.
- Previous R1 / R2 / R4 / R5 counterexamples now pass their fixed/off controls; missing/stale guard withholding passes with GS.
- P1 R6, ladder: a rung fills at 90 ms, notification due 190 ms. At 100 ms, on a frame stamped 85 ms, the engine places a new touch
  order + rung. Identical public-input no-fill control sends neither. L_FILL=1000 also sends at 100 ms. Reproduced under the full
  active MICRO / AV / six guards / V0+GS+LD2+XNA+XMIN1+XV+S60 stack with positive depth and flat Binance inputs.
- Cause: matcher removes `orders[lane]` immediately; `cur_order()` reads that physical state inside `desired()`, before K arrives.
  Separate policy-visible own-order state from physical matching state and update it on received acknowledgements by order id.
  Delaying knet alone does not enforce the information boundary. Add a causal equality test before re-evaluating ladder comparisons.

## 2026-11-05 11:38Z - follow-up simulation review: prior fixes pass; atomic sweep bug + rejection recovery OPEN

- Superseding receipt: `reviews/simulation_20261105/receipt_v2.json`; reproducer `check_remaining.py` in the same folder, via
  `research/run_light.sh`. Source-only synthetic inputs, no market data read, no engine changes by reviewer; history impact unmeasured.
- Verified prior price-priority and own-cross fixes with their regression/off controls; GS withholds missing/stale guard coverage.
- P1 R4, BOTH engines: trade tuples sort ascending by price at a shared timestamp, replaying a sell sweep worst price first.
  Synthetic 100 sh at 50 + 5 at 49, our bid 10 at 50 behind 100: engine fills 10, maximum valid fill 5. Better-price-earlier,
  single aggregated leg, and mirrored ask controls each fill 5. Retain taker identity; replay atomic sweeps in matching order.
- P2 R5, BOTH engines: venue-book post-only refusal leaves the rejected intent installed (own-cross refusal resets it). When
  recovered known state wants the same quote, no retry is sent. Reproduced with coalesced knowledge times; no-refusal control fills.
- Current kJf / kXf / ladder reruns still contain R4. Fix + rerun before relying on their precise economics; parity alone misses it.

## 2026-11-05 11:22Z (session 66f8e5c3) - review bugs fixed in both engines; chain fix2 -> par -> smoke -> fullJ -> fullX RUNNING

- Fill priority among our own orders, post-only vs our own resting orders, GS strict guard (STATUS this time). Regression gate
  PASS. Earlier numbers carry the fill-priority bug until kJf / kXf report the impact; quote no earlier $ figure without it.
- Next: read fix2 impact (kJf / kXf vs kJ / kX), parity parf, own-cross falsifier (tag ownoff must count own_cross > 0),
  report_ladder2.json, docs section 55.

## 2026-11-05 11:14Z (session 66f8e5c3) - ladder 2 / improve / per-minute lines / fresh refresh RUNNING (STATUS declaration 11:00Z)

- Done: fresh-refresh predictions preds_t60kE (every-frame rows 0-65 s + 50 ms grid; same models, thresholds equal); ladder
  engine pf_step4_ladder.py (parity with pf_step4_live PASS on 10-24 / 11-01, controls PMzero / AMc006 / LDinf / IMinf equal,
  falsifier ok); LH (rung hold) added after the smoke showed guard-driven rung churn; parity re-run in the chain.
- Running: run_ladder2.sh par -> smoke -> fullJ (8 arms x 2 prediction sets x 9 days, then pf_ladder2_report). Next: fullX
  (bound timing) and docs section 55.
- Watch out: in-sample (lines and guard from the same 8 days); prior negative ladder test docs 36-38. run_ladder.sh is the
  11-03 driver (restored verbatim after an accidental overwrite at 10:58Z); the new driver is run_ladder2.sh.

## 2026-11-05 11:12Z - simulation code review: two defects + guard-policy risk OPEN

- Review receipt + synthetic reproducer: `reviews/simulation_20261105/receipt.json` and `check_engine.py`. No simulator code changed;
  the reproducer reads source only, no market data. Historical frequency / P&L impact NOT measured. Run via `research/run_light.sh`.
- P1: both live and new ladder engines, with ladder tokens off, fill cancel-pending orders before better-priced replacements
  (old bid 49, replacement 50, both live: a sweep fills 49). Sort all fillable orders by price then activation time.
- P1: new ladder post-only / own-cross checks omit cancel-pending orders. Synthetic live bid 51 / pending-cancel ask 51 coexists
  while `own_cross` reports 0. Include every fillable own order in activation checks and diagnostics.
- P2 policy risk: missing / stale per-side guard predictions permit quoting; fresh adverse-value control withholds correctly.
  Verify whether this fallback is intended; report unavailable inputs, and withhold if guard coverage is required. Synthetic
  positive controls and unsupported split/amend refusal passed.
- Next: fix these paths and rerun affected comparisons before relying on precise replay economics. Passing parity does not test
  these properties. A successful reproducer run confirms these behaviors are present; it is not a correctness certificate.

## 2026-11-05 08:33Z (session 66f8e5c3) - per-minute guard percentiles + their fill P&L (descriptive; STATUS 07:27Z, 08:23Z, 08:33Z)

- Per-minute percentile values of the guard: pf_minute_pct.py. Fill P&L by percentile bucket: pf_minute_pnl.py (kJ fills, no replay).
- Minute 1: the money is in the top ~10% of the guard (8/8 or 7/8 days); p70-p90 about 0; fills below the line (pull in flight) lose.
  A minute-1 pull at ~p90 is a hypothesis read off in-sample fills (path effects ignored), not a result.
- Exec-port session's queue-ahead measurement copied to data/predict_fun/research/ext/queue_ahead_execport_20261105/.
- Freeze / forward replay still waits for the user.

## 2026-11-05 06:22Z (session 66f8e5c3) - rebuild 3 DONE (docs §54): edge small, recent days <= 0; longer / larger does not add

- Central (real key + within-frame time): C1 50, C2 155, C3 106, C4 46 $/day (8 read days, no rebate); none passes all declared
  predicates (all fail the E2-only -300 ms stress, a far-tail bound). Latest four days negative on average for every candidate.
- Window x size: nothing adds; longer stops add carry variance, not fill quality.
- C3 freeze recommendation withdrawn. Open for the user: stop here, forward-replay C4 on 11-04..08 (same replay limits), or a
  capped live pilot via the implementation session to measure own fills (the only way to settle book-timed trade timing + impact).

## 2026-11-05 03:05Z (session 66f8e5c3) - second review found timing defects; rebuild 3 (STATUS 03:01Z)

- Verified: the transaction key merged different taker orders (42% of keys on 10-24 12-18Z); trades sit at frame stamps up to one
  ~50 ms frame after the match; the replay padded Binance / books into 11-04 (read record in STATUS). The old money numbers (§53) are
  not reliable; C3 / C4 / C1 / C2 must be re-measured. The window / size grid (rbw / rbwX) was stopped, superseded.
- New: pf_txmap.py (taker order per leg from REST), engine PF_TXKEY=tk / PF_TXJIT=frame + input bounds + code hash in summary,
  pf_badfill_grid_tx PF_TXKEY=tk. Chain run_review3.sh: txmap 9 days -> grid_tk -> guard refit preds_t60 -> preds_t60k -> replays kJ
  (central) / kX (bound) on 14 arms, kS / kE / kM on C1-C4 -> pf_review3_report.py. Predicates in STATUS 03:01Z.
- No freeze, commit or implementation message before the user chooses; validation = 11-04..11-08 forward days of a frozen version.

## 2026-11-05 01:50Z (session 66f8e5c3) - second review + window / size grid RUNNING (STATUS 01:49Z)

- Second independent code review (agent, read-only) running. Fix checks done here (start withhold 0 early fills vs control 39-106;
  pending fix touches only pf_pend_*; tx-time instrument 39,351 -> 0 multi-time transactions, engine REFUSES otherwise).
- run_review2.sh waits for C2's 11-04 run (02:10Z), then: stress rbtxE (E2-only -300 ms), rbtxS50 (-50 ms); window x size grid rbw
  (S60..S240 x AV 5/25, 20/100, 40/200) + stress rbwX; then pf_window_size_report.py. ~1.7 h.
- 11-04+ are forward days: not read (no freeze yet). Freeze still waits for the user's version choice.

## 2026-11-04 15:40Z (session 66f8e5c3) - reliability review DONE (docs §53); freeze waits for the user's choice

- After every correction (one time per taker transaction, start withhold, pending features per frame, tx-timed labels + refit, XV,
  queue bound, 25 ms shift) C1 / C3 / C4 pass all declared predicates; C2 fails top-5 per share. Money about half of first reported
  (C3 138 $/day mark, no rebate). 11-03 (only unchosen day) ~0 for all. Forward days 11-04+ decide.
- Freeze path when the user picks: commit predictfun/ (untracked today), run pf_guard_freeze.py <name> 60 at that commit (selftest on
  11-03 preds_h60), commit predictfun/frozen/<name>/ + a receipt (arms, live rules, forward pass rule, multiplicity), then message
  the implementation session (exec port: needs V0 per-side guard with 2 text models + thresholds, AV size, start withhold, concurrent
  replace, XV throttle; today it runs sequential replace at 1 place/s).
- Engine options added today (default off): PF_TXTIME, PF_TXSHIFT, PF_QACT; size kind AV. Feature option PF_PEND_KNOWN=frame (_pk).

## 2026-11-04 13:45Z (session 66f8e5c3) - reliability review RUNNING before a freeze (STATUS 13:41Z)

- User: review the result, then freeze a version and tell the implementation session. run_review.sh (log run_review.out):
  pf_step5_guard_preds_wf (walk-forward guard, preds_w60), pf_preds_startknown (withhold until the start price is known: first
  predictMarketChanged patch, Tokyo clock -> preds_g60k / preds_w60k), replays tag rv (C1 5/25, C2 20/100, C3 AV 20/100 v>=0.06;
  each +/- XV). Then pf_review_report.py (predicates R_START / R_WF / R_DAYS / R_TOP5). An independent code-review agent runs too.
- Freeze builder ready, not run: pf_guard_freeze.py <name> <k> (refuses unless the pipeline is committed; selftest on 11-03).
  predictfun/ is entirely untracked in git today: the freeze commit adds it (check: no secrets found by a pattern scan, 2.9 MB).
- Implementation side: pmmmrust docs/PREDICT_FUN_PLAN.md (exec port: sequential replace + 1 place/s today; the candidate assumes
  concurrent replace + venue throttle XV - a change to flag). Sessions: "Predict fun - Tokyo" (Remote Control) / pmmmrust.
- The 50 / 70 size chain (run_size_dyn.sh) was stopped at 13:40Z after gzd50_20261102; skip logic resumes it.

## 2026-11-04 12:07Z (session 66f8e5c3) - size / stop / dynamic size at pull 60% DONE (docs §52); 50 / 70 RUNNING

- Size scales $ almost linearly at unchanged quality (S60 20/100: mark@stop 355 $/day vs 125, 8/8) - but no own impact in the
  replay and 100 sh = 37% of the touch. Longer stops: not reliable (CI spans 0; 11-03 worse). AV guard-value sizing: best per share,
  beats fixed at matched shares at v >= 0.06, but less total $ than fixed 100. A3 rejected; A1 = fixed hi in practice.
- run_size_dyn.sh continues (gz / gzd at k = 50, 70), final report size_stop_report_all. Open ideas (not declared): skip 60-120 s
  only; reduce-only after the stop to cut carry; own-impact model before trusting 50-100 sh; 11-04 unseen check after C2.
- Watch out: pausing the chain = kill -STOP its bash between runs (the report can't win the slot otherwise); never pkill -f a
  pattern that is in your own command line.

## 2026-11-04 10:38Z (session 66f8e5c3) - size x stop grid + dynamic sizes RUNNING (STATUS 10:32Z, 10:37Z)

- run_size_dyn.sh (setsid; log data/predict_fun/research/logs/run_size_dyn.out) took over from run_size_stop.sh after its first day-run;
  finished day-runs are skipped. Dynamic arms tag gzd (fixed 25/50/100, AV guard-value size, A3 depth), new engine size kind AV (additive).
- grid: run_size_stop.sh (stopped): trained guard V0 at k = 60, then 50 / 70; stop 60 / 120
  / 180 / 240 s x size A1:5:25:0 / A1:10:50:0 / A1:20:100:0; 8 read days + 11-03, tag gz. pf_size_stop_report.py runs after each k
  (non-fatal). Parity: gz S60 A1:5:25:0 must equal the gd run. Caveat to carry: the fill model has no own impact (our order is never
  shown to others), so large sizes read optimistic; compare hi size to the displayed touch.

## 2026-11-04 09:13Z (session 66f8e5c3) - step-back variants DONE: keep withholding (docs §50)

- One tick back (SBV) and prediction-priced (SBP) both fill far more and badly (+1 s -0.35 / -0.20 c/sh at 50% vs withhold +0.34).
  Best rule so far: trained per-side guard (V0), withhold, pull 50% (most dollars) to 60-70% (top-maker quality). To do: 11-04 as
  the second unseen day (after C2 at 11-05 02:10Z: pf_step5_features_rl, pf_step5_c3_preds apply, pf_step5_guard_preds with
  PF_GUARD_KS=50,60,70 and the day, replay arms).

## 2026-11-04 08:22Z (session 66f8e5c3) - pull sweep 10-70% DONE (docs §50)

- Trained per-side guard (preds_g<k>, engine token V0) reaches the top-maker quality at a 60-70% pull (bad 9-10%, ratio 33-42%,
  +0.53..+0.79 c/sh); total mark-at-stop peaks at 50% (+920 / 8 days vs +355 now). 11-03 (unseen) agrees in direction. Running:
  one-tick step-back (SBV, run_guard_stepback.sh) then prediction-priced step-back (SBP, run_guard_predstep.sh). Next: 11-04 as a
  second unseen day once C2 has processed it (after 11-05 02:30Z; build its features_rl_d2 + preds_c3 + preds_g<k>).

## 2026-11-04 05:59Z (session 66f8e5c3) - guard: trained M1 is the one to keep; latency inputs add nothing (docs §50)

- Matched-pull control: M1 guard beats the current guard's input at the same pull (+0.15 c/sh at 30%, 8 days; +0.30 on 11-03).
  Latency-aware inputs (frame age, Binance move since the frame) add nothing to it. Best so far: bad 13.5%, ratio 64%, +0.14 c/sh
  (target <= ~10%, <= ~45-50%, ~+0.3). Next candidates: pull 40 / 50%, anchor as quote centre, 11-04 as a second unseen day.

## 2026-11-04 05:25Z (session 66f8e5c3) - anchors tested; the lead-adjusted one passes (docs §50)

- Binance-only fair values (p_bn, recalibrated, fitted A5) fail; Predict's late price + the Binance move since that frame (A6..A8)
  passes: Predict makes ~1.1x the adjustment within 1 s and keeps it. Stage 3 guard: +0.14 c/sh at a 30% pull, matched-pull control
  (pf_step5_baseguard_preds, run_guard_matched.sh) running. Next: dp / A8 into the guard and as the quote centre.

## 2026-11-04 04:22Z (session 66f8e5c3) - stage 1 (anchor) and stage 2 screen DONE (docs §50)

- No Binance fair value beats Predict's own (late) price on settlement or leads it; Predict book signals add ~0.003 AUC to the
  guard model; everything we record tops out at AUC ~0.83 vs 0.78 now. Open (user decision): a lead-adjusted anchor (Predict's
  late price plus the Binance move since that frame), stage 3 replay of the best guard, or conclude.

## 2026-11-04 03:59Z (session 66f8e5c3) - research flow set (docs §50)

- Stage 1 anchor test next (pf_anchor_eval.py), then stage 2 predictor (pf_badfill_info2 running is its first screen), stage 3
  combined replay vs the top-maker target (bad <= ~10%, bad$/good$ <= ~45-50%), then fresh days 11-03+.

## 2026-11-04 03:03Z (session 66f8e5c3) - 10-venue big-order check DONE (docs §49)

- pf_big_prints.py / pf_big_prints_split.py: 1 bad fill in 11 has a big order elsewhere in time for our cancel (half of those off
  Binance); most big orders come with or after our bad fills. Not checked: Polymarket's parallel BTC 5-min markets (P-2026-003
  data, ask first), Chainlink's own fast stream.
- Watch out: data_collectors/fetch_public_trades.py writes rows sorted by (time, price); order by id before any tick test.

## 2026-11-04 02:59Z (session 66f8e5c3) - anchor sweep cut + information test DONE (docs §48)

- Sweep: R^2 does not rank anchors (synthetic 0.15 >> real model at 0.15); cut at the calibration check.
- pf_badfill_info.py: current features rank bad exposures at AUC 0.78 (guard input) / 0.83 (trained model); outside feeds add
  ~0.5%; ideal 0.999. New public trade data for 7 more venues in data/exchange_public/ (README); pf_big_prints.py (10 venues) next.

## 2026-11-04 01:29Z (session 66f8e5c3) - ideal-anchor test DONE (docs §46)

- preds_oracle (realised 1 s / 250 ms moves) through the unchanged replay: strongly positive every day; the real model
  (R^2 0.11-0.20) is break-even in the first minute. The anchor's quality is the lever.
- Next candidate (user decision): a degraded-anchor sweep (realised move blended with noise to R^2 0.2..0.9) to find the
  quality at which quoting pays, then ask whether any feed can reach it.

## 2026-11-04 01:00Z (session 66f8e5c3) - request limit verified + replace compare DONE (docs §44-45)

- Predict's headers: create/remove 40/s + 500/min, reads 500/min, one budget per key (pmmmrust plan still says 240/min).
- pf_replace_compare.py: parallel replace helps a little (+1 s -0.17 -> -0.10 c/sh) but the swept share stays ~65%;
  queue position and the late feed are the cause.

## 2026-11-04 00:28Z (session 66f8e5c3) - re-post anatomy DONE (docs §43)

- pf_repost_anatomy.py: fast re-posts are makers sending cancels and new orders in parallel within one ~50 ms publication,
  several orders at once; not aggressive-order remainders. Our live plan is sequential + 1 place/s and sees the feed ~53 ms
  late.

## 2026-11-03 23:58Z (session 66f8e5c3) - cross-venue check DONE (docs §42)

- pf_cross_venue.py, pf_sweep_reversion.py; Binance spot aggTrades for the 8 days in data/binance_spot_vision/ (public,
  checksums OK). Bad sweeps are informed (they last) but no recorded feed leads them by much; makers pull on Binance perp.
- READ RECORD (STATUS 23:51Z): io.hours_of(day, 1, 1) padding loads the Binance tape of 10-28 00Z / 10-30 23Z (for 10-27 /
  10-31) and 11-03 00Z (for 11-02), since the project's first scripts. Recorded, no harm under the 10-22 rule; fix pf_io
  (refuse sealed hours) only after C2 finishes, since pf_io is a C2 dependency.
- Open (user decision): Polymarket parallel BTC 5-min markets as the sweepers' source (P-2026-003 data: ask first); other
  exchanges (OKX / Bybit public trade archives); a Binance-perp-flow guard in the replay.

## 2026-11-03 23:31Z (session 66f8e5c3) - speed race DONE (docs §41)

- pf_speed_race.py, pf_sweep_precursor.py, pf_vacate_gap.py. Binance-led risk: we are fast enough. Most bad fills: no
  Binance move and no warning visible within our 75 ms blind spot. Front: freed prices are re-posted in the same book
  update in 62% of cases; we could be first in ~14%.
- Next candidates (user decision): what the sweepers know (do bad sweeps follow the settlement reference feed we record,
  data/predict_reference?); replay a wider Binance pull (all ladder levels on 0.5 bp); a live latency probe (our real
  Predict feed / order path vs the 13 ms feed minimum).
- Watch out: pf_speed_race peaked at 2006 MiB RSS (guard 2048); split by day before adding anything to it.

## 2026-11-03 23:05Z (session 66f8e5c3) - fill mix vs the other makers DONE (docs §40; corrects §38)

- pf_fill_mix.py (transaction sweep rule, key validated 98.4% by pf_match_type.py), pf_skip_match.py. Our quoting is not
  the gap: at our own price we match the top makers. We lose on mix (64-73% swept vs 18-21%) and on which sweeps reach us
  (ours far more informed). Both are speed / queue position. Watch out: the book "cleared" rule under-counts sweeps; use
  the transaction rule.

## 2026-11-03 22:26Z (session 66f8e5c3) - maker behaviour DONE (docs §39)

- pf_maker_behavior.py: no queue held from listing; fills come at levels formed in the last 0.1-10 s, mostly at levels born
  as a new best; the front of a new level is traded within 1 s in 55-72%. Top 5 are not over-represented at the front; they
  fill deeper, lose less when picked off, and end windows nearly flat, with small trading P&L per share.
- Natural next checks (user decision): who creates new best levels and how fast after a Binance move (could we be the
  creator at our latency?); liquidity rewards (T5) as the top makers' income.

## 2026-11-03 22:12Z (session 66f8e5c3) - hold check DONE (docs §38; corrects §37)

- pf_hold_check.py: most of the size resting behind the best at the open is cancelled within 10 s (lower bound >= 53-71%);
  top makers are swept far less than our ladder (22% vs 74%). §37's "built at listing" was read from an upper bound: void.
- Open: how the top makers reach the front (speed to new price levels vs pulling before sweeps) - a possible next check
  from the maker legs; true queue position of a resting order needs a live test.

## 2026-11-03 21:48Z (session 66f8e5c3) - queue-priority diagnosis DONE (docs §37)

- pf_queue_priority.py (market side) and pf_ladder.py PF_ORDLOG=1 (order log; summary parity with lad1 checked by
  pf_ladder_orders_report.py). We are at the back because the incumbents post at listing (~4.7 min before the open) and
  stay; our ladder re-centres and pulls (median order life 0.76 s). Next is a user decision (pre-window persistent ladder
  test, small live test, or stop).

## 2026-11-03 16:36Z (session 66f8e5c3) - ladder test DONE, negative (docs §35-36)

- pf_ladder.py (multi-order replay, selftest 6/6). All ladder arms lose on mark-at-stop; our deep orders sit behind the
  incumbents' ladders and fill mostly in sweeps. Untested structural idea: a persistent ladder placed at market
  registration (pre-window), never re-centred. Otherwise the replay finds no robust maker edge at our latency and queue
  position; a small live test is the only way to learn real queue dynamics.

## 2026-11-03 11:20Z (session 66f8e5c3) - O4 DONE, C3 2 days (docs §25-26)

- C3 (11-01/02): O3's edge did not appear (P3 -156, +1 s -0.52 c/sh). O4 (sweep protection, 8 days, no hold-out): nothing
  adopted; P3 +1,462 / 8 d with +1 s mark ~0. Tuning inside this strategy family is at the noise floor.
- Open options: real-fill calibration (small live test, user decision); liquidity rewards (T5, unverified); a sweep /
  regime model (new research); keep C2 / C3 running on new days (11-03 from 11-04 ~02:30Z).

## 2026-11-03 09:59Z (session 66f8e5c3) - round O3 DONE (docs §24)

- At the measured Tokyo clock + decision 2 ms: quote only the first 60 s of each window (S60) + stand back one tick (BKA)
  (+ Q025 250 ms guard, B2W100) -> combined +1,617 (seq) / +1,806 (conc) over six read days, 5/6 days, +1 s edge positive.
- Next: fresh-day confirmation C3 (declare + freeze before reading). Pipeline for new days needs features_rl d2, model A
  pred_1000 (model_c2_y1000 = the all-six-days fit of the same spec) and a y250 model fit on all six days' rows_rl_d2.

## 2026-11-03 08:05Z (session 66f8e5c3) - latency test L2c DONE (docs §23)

- Measured split latencies (order live 40 / resting cancel 22 / pending cancel 95 after create / fills known 100 / decision
  10 ms): live-like C2-P loses in every unthrottled and venue-throttled arm; v6 (sequential, 1/s) positive but adverse fills
  and model-sensitive. Levers: order-live time, then our decision time. Replay support for a live go: none at this latency.
- pf_step4_live.py sha 1edab4e7 (split handler + pending-cancel + fill-known settings); tokyo20 / L1 paths unchanged.

## 2026-11-03 05:51Z (session 66f8e5c3) - latency test L1 DONE (docs §22)

- Tokyo probe (pmmmrust fe073d1): no removal lock; limits 40/s + 500/min; Predict feed 53 ms median + 10 ms decision + 22 ms
  order. Replay at that clock (scenario tokyoreal; pf_clock_rl, pf_step5_features_rl, pf_step5_lodo_rl): edge mostly gone
  (+240..600$ / 6 d, +1 s mark <= 0). Live go is not supported by the replay at real latency.
- Next if asked: latency decomposition (which component costs most: Predict feed, our 10 ms, order 22 ms); strategy
  changes aimed at stale quotes (wider / one tick back, stronger guard) tuned at the Tokyo clock, confirmed on fresh days.
- pf_step4_live.py changed (sha 224fde6d): new scenario + XV only; tokyo20 arms reproduce round 2 to the cent.

## 2026-11-02 12:35Z (session 66f8e5c3) - user decision: live = sequential replace + 1 place/s

- Relayed to the exec-port session (it owns pmmmrust and re-freezes). Reference rule = pf_step4_live.py XP<lambda> + XT1
  (sha256 c0b2ba80e9e29621); expected outputs in step4/live_..._livedev4 (and livedev2 for lambda 0). FROZEN as pmmmrust
  pfmm-c2p-tokyo-v2 (code e77a376), now tag v3 (same code + runbook + scripts/pf_tokyo.sh, which reads the §19 grid). Waiting for: the probe's lambda (read off docs §19/§20) and rate headers. Do not edit pf_step4_live.py without
  telling the exec port (its parity fixture pins this hash).

## 2026-11-02 09:37Z (session 66f8e5c3) - live round 2 DONE (docs §17)

- Sequential lock (XS) and the exec-port throttle (XT) added to pf_step4_live.py; `run_live_dev2.sh`, `pf_live_report2.py`.
  Lock >= 200 ms loses in either design; throttle at L = 0 keeps +958 (1/s) / +2,056 (2/s) of +3,067. Sent to the exec port.
- Waiting on the exec port's Tokyo probe (user's step): lambda = removalLockedUntil - acceptance time. Read it off the
  placement-relative grid (docs §19): break-even 163 ms unthrottled / 204 ms at 1 place/s / 180 ms at 2/s; 100 ms keeps
  +522..+1,471$/6 d; 500 ms loses everywhere. (§18 = lock on every cancel, the worst case, break-even ~75 ms.)
  Concurrent design (frozen Tokyo version) in §20: at 1 place/s it is worse than sequential at every lambda <= 200 ms
  (break-even 138 vs 204 ms); unthrottled it is better.

## 2026-11-02 09:02Z (session 66f8e5c3) - live-deviation tests DONE (docs §16)

- For the pmmmrust exec port (C2-P frozen there): `research/pf_step4_live.py` (copy; live rules as X tokens, parity to the
  cent), `run_live_dev.sh`, `pf_live_report.py`. Removal lock and request rate dominate; no amend / min 1 USD minor; the
  150$ cap can stick at -149.96 without halting. Results sent to the exec-port session.
- Next if asked: sequential-lock design (replace only after the cancel lands); a request-budget (throttle) replay.
- pf_step4_live with 14 arms peaked 1,994 MiB (guard 2,048): run fewer arms per process.

## 2026-11-02 02:01Z (session 66f8e5c3) - C2 RUNNING

**In progress:** confirmation C2 (STATUS.yml `confirmation_C2`, declared 02:01Z before any 11-01+ content was read).
- Candidates: P = D1 stack + M025 + A1 5..25 (primary), S = combo G1+G5+M025 + A1 5..25; references R0-R5, C1c.
- Model: `research/pf_step5_c2.py` fit on the six read days (sha be6e916e, 4.02M rows, peak 1.1 GiB).
- Driver: `research/run_c2_auto.sh` (setsid nohup, pid in `data/predict_fun/research/c2/auto.pid`) runs
  `run_c2_day.sh <day>` after the next day's 01Z hour closes (+10 min), appends `c2/admission.jsonl`, and after 5 ADMITTED
  days runs `pf_c2_report` -> `c2/report_c2.json`. Heartbeat `c2/auto.heartbeat`, log `c2/auto.log`.
- Earliest verdict ~11-06 02:30Z (days 11-01..11-05). A FAILED step stops the driver: look at `logs/c2*_<day>.log`;
  any code fix = superseding declaration C2 v2 in STATUS.yml (keep v1), then remove that day's FAILED record and restart.

**Watch out for:** do not change arms, rules, code, model or the pass rule while C2 runs (hashes in the declaration).
The driver only uses run_light.sh (core 8, one process); P-2026-003 must not pin its jobs to core 8.

## 2026-11-01 05:41Z (session 66f8e5c3)

**Done** (write-up: `docs/EXPLORATION_2026-11-01.md` §1-9; exploratory; dev days 10-23..25 fully read)
- T1-T3: sources reconciled (REST volumes; WS pending = USD at YES price; executedAt rounded); plain baseline loses.
- T4: Tokyo latency (user: ~20 ms) -> scenario tokyo20; every D1 rule helps alone (3/3 days); full MICRO stack break-even.
  Untimed-trade bracket narrowed by the book-drop estimator E2 (median |err| 0-49 ms); three timing models agree.
- T6: LightGBM (pinned, `research/pf_model_spec.py`), leave-one-day-out: 1 s mid-move R2 0.14-0.18, 250 ms 0.38-0.48.
  D1 stack + model guard th 0.5c = +$237 / 3 days (+0.24 c/sh, 3/3 days, all timing modes) vs D1 stack -$5.

- C1 confirmation (06:26Z) PASSES its declared rule on 10-26/27/31: D1 stack + model guard +$190.67 (mid; early +148,
  late +226), beats D1 alone on 3/3 days (+102/+86/+87). Small: +0.27 c/share, ~$0.22/window at 5 shares.

- 09:39Z WS-vs-REST check (10-23, docs §11): WS pending is an aggregate, not a trade list; 99.7% of REST volume shows in
  WS, but 8.2% of WS increases have no REST trade at that price within +-1 s -> REST completeness OPEN (<= 9.4%).

- 10:51Z on-chain reconciliation (docs §12): REST complete and exact on 6 slices of 10-23; executedAt = floor(block time);
  orphans = taker orders listed at their signed limit price. Scripts `research/pf_chain_*.py` (public RPC, read-only).

- 13:26Z round O1 (docs §13): size scales P&L, per-share +1 s edge flat (~0.25 c); 6-day model no P&L gain; declared combo
  MICRO+G1+G5+M025 +497 dev / +680 2nd look (inventory uncapped). Scripts pf_o1_*.py, run_opt.sh, run_o1_combo.sh.

- 14:45Z round O2 (docs §14): replay loop fix (in-flight replacement after a full fill now cancelled; restated numbers within
  ~$12, C1 still passes). Adaptive size: model-based (A1) best per-share edge; adopted B1+A6 and B2+A1. Script pf_o2_report.py.

- 15:58Z code review + fill model v2 (docs §15): queue double-count fixed (material), S270/S300 invalid (no predictions
  after 240 s), smaller fixes; PF_FILLMODEL=v1 reproduces old numbers. v2: C1 still passes (+158, margin halved); B2+A1 adopted.
  Use v2 (default) for everything from here; pf_check_fills.py on every new output.

**Next (proposed)**
- Confirmation C2 on fresh days (11-01 onward, complete UTC days; check P-2026-003 has not reserved them first): model fit
  on all 6 read days; candidates B2+A1 (D1+M025, model-sized 5..25) and B1+A1 (combo, model-sized), fixed-size references, fill model v2; declared before reading.
- Optional: on-chain block-ms timing for untimed fills (10-31's 29%); extend the REST-vs-chain check to other days.
- Latency sensitivity of the candidate (Predict 20 -> 58 ms 'tape' scenario; order latency 20 -> 50/100 ms).
- Size sensitivity (5 -> 10/25 shares: queue position and the inventory caps bind differently).
- Liquidity-reward terms (T5): hourlyRate 3000 / 6c / 100 sh could dwarf the +$64/day gross.
- 10-31 data note: 29% of volume untimed (refills within a frame?); check before using later days.

**Watch out for**
- One process at a time via `research/run_light.sh` (sets ARROW_DEFAULT_MEMORY_POOL=system; the 4 GiB AS cap killed one
  model run before). Peaks: step 1 ~1.0 GiB, step 4 ~1.5 GiB, features ~1.7 GiB, model ~1.4 GiB.
- 10-28..10-30: do not read (Binance tape sealed for P-2026-003's order-flow test). Read days now: 10-22 (partial), 10-23..27, 10-31.
- Results are gross at 5 shares with a modelled fill (FIFO behind displayed size, no own impact); no real-fill calibration.
