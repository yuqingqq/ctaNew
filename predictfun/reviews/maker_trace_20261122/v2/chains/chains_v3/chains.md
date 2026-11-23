# Six complete chains v3 (posting_chain_contract.md section 4) - RUN

As of 2026-11-23T18:37:52Z. Mode **run**. Selection (predetermined, not outcome-selected): per rank, the first two episodes in `order_estimator/cases.csv` row order of first appearance = `correction_v1/paths.md`. Saved outputs only; descriptive; one consumed day (Oct 24) of four; filled orders are outcome-selected; no identified intent or private action. Times are venue ms relative to the market window start ws.

Readings: **ONE-HISTORY** = every retained history agrees; **TIED** = at least two retained histories change the explanation (the hidden action is named); **NO-EVENT** = no take record; **PENDING** = a sibling output is absent (draft). Grades: `observed_state`, `post_window:G0_unbounded_or_invalid`, `post_window:G1_conflict`, `post_window:G2_bound_constrained`, `post_window:G3_constrained_status_flagged`, `post_window:G4_scenario_only`, `scenario_queue_history`, `cancel_compatible_ending`, `scenario_dependent_ending`, `unresolved_alternatives`, `scenario_dependent_link`, `observed_execution`, `recorded_fill_accounting`, `pending`.

Siblings: posting_opportunity **ok**; queue_front_v2 **ok**. Without queue_front_v2 the queue links carry v1 labels (see section 5); without posting_opportunity the pre-state links are PENDING.

| chain | rank | episode | PRE-STATE | POST | QUEUE | ENDING | REPOST | TAKES | VALUE | first branch |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 20261024_3716727_r1_e018 | TIED | TIED | TIED | TIED | TIED | ONE-HISTORY | ONE-HISTORY | pre_state |
| 2 | 1 | 20261024_3702960_r1_e006 | TIED | TIED | TIED | TIED | TIED | ONE-HISTORY | ONE-HISTORY | pre_state |
| 3 | 2 | 20261024_3702960_r2_e007 | ONE-HISTORY | TIED | TIED | TIED | TIED | ONE-HISTORY | ONE-HISTORY | post |
| 4 | 2 | 20261024_3712725_r2_e015 | TIED | TIED | TIED | TIED | TIED | NO-EVENT | ONE-HISTORY | pre_state |
| 5 | 3 | 20261024_3712725_r3_e014 | TIED | TIED | TIED | TIED | TIED | ONE-HISTORY | ONE-HISTORY | pre_state |
| 6 | 3 | 20261024_3716727_r3_e017 | TIED | TIED | TIED | TIED | TIED | ONE-HISTORY | ONE-HISTORY | pre_state |

## Chain 1 - rank 1: episode 20261024_3716727_r1_e018

day 20261024, market 3716727, address 0x6da6cb464f92ae7ad4ec3d239c81719cb1d0ae03; case order 0x88c0095c1e; cases.csv data row 1; episode window [ws+2.325s, ws+110.524s]

- **1 PRE-STATE** — **TIED** (25 retained histories) · grade `observed_state` · filled
  - retained histories: spread=1; depth_touch_own=(0,50]; depth_touch_opp=(0,50]; depth_2_own=0; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=(0,50]; depth_touch_opp=(0,50]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(0,50]; depth_touch_opp=(50,200]; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=(0,50]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=<=0; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=against; fair_gap=<=0; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=>200; depth_touch_opp=(0,50]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=>200; depth_touch_opp=(0,50]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=2; depth_touch_own=(0,50]; depth_touch_opp=(0,50]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=<=0; flow_5s=>0; time_left=(240,300] | spread=2; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(120,240] | spread=2; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]
  - hidden action that changes the explanation: the posting second inside k = 2..80 (ws+2..ws+81 s): features spread, depth_touch_own, depth_touch_opp, depth_2_own, btc_5s_aligned, fair_gap, time_left change across it
  - window seconds k = 2..80 (79 in the market window, 0 outside; 79 saved rows at side ask; time key t_ms); post window [ws+2.325s, ws+80.506s]
  - features (D0 = the state at t; columns spread_bin_D0, ...): spread {'1': 76, '2': 3}; depth_touch_own {'(0,50]': 8, '(50,200]': 23, '>200': 48}; depth_touch_opp {'(0,50]': 13, '(50,200]': 24, '>200': 42}; depth_2_own {'(0,50]': 4, '(50,200]': 10, '0': 1, '>200': 64}; btc_5s_aligned {'against': 1, 'flat': 78}; fair_gap {'(0,1]': 12, '(1,3]': 62, '<=0': 3, '>3': 2}; flow_5s {'>0': 79}; time_left {'(120,240]': 21, '(240,300]': 58}
  - row statuses (status_D0): none
  - D1000 (the whole state lagged one second): 25 distinct state vector(s) vs 25 at D0 - a different set (timing-sensitive)
  - visible fresh creations on this side in the window: 31; at the order's cell (tick 69): 3716727|ask|69|475 (deep) (an observed market creation; the order is at most a candidate creator, section 3)
- **2 POST** — **TIED** (4 retained histories) · grade `post_window:G4_scenario_only` · filled · census: rank1 20261024 G4_scenario_only: 348 of 746 first-filled orders
  - retained histories: late_post_in_scenario_window + EX | late_post_in_scenario_window + PL | earlier_post_masked_by_replenishment + EX | earlier_post_masked_by_replenishment + PL
  - hidden action that changes the explanation: the posting instant inside [ws+2.325s, ws+80.506s] (78181 ms; scenario union estimate only); the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - order 0x88c0095c1e: ask at 69c (side and tick observed in its fills); first fill <= ws+80.506s
  - window [ws+2.325s, ws+80.506s], width 78181 ms (gt_10s); per-scenario lower ends S_FIFO ws+80.279s, S_FIFO_LO ws+2.325s, S_NOPRIO ws+80.279s; lifecycle post_class assumption_sensitive; packet status ok -> grade G4_scenario_only (scenario union estimate only)
  - size: filled 30.00 sh = a LOWER bound (observed executions); submitted size unknown; feasible EX (exhausted: = 30.00 sh, equal to the filled size only under EX); PL (pulled: [30.01, inf] sh); bounds [30.00, inf] sh
  - size status tied; prior picks (prior-driven estimates, not evidence): P_cmd ('EX', 30.0); P_drop ('EX', 30.0); P_lot ('EX', 30.0); P_clip ('EX', 30.0)
  - same-hash taker legs (a crossing at arrival): 0 leg(s), 0.00 sh
  - saved POST logic row: queue_priority_before_demand timing_sensitive -> timing_sensitive (grade B_constrained -> C_scenario)
- **3 QUEUE** — **TIED** (2 retained histories) · grade `scenario_queue_history` · filled
  - retained histories: front_at_first_fill @ level 51 (unknown) | behind_at_first_fill @ level 51 (unknown)
  - hidden action that changes the explanation: whether orders were ahead at the first fill (S_FIFO ahead [0.00, 320.00] sh at the first fill; [24.00, 442.97] sh at acceptance)
  - order 0x88c0095c1e: 1 fill(s), 30.00 sh
    - fill 1: 30.00 sh, hard window [ws+78.506s, ws+80.506s], sid 01a16f57-b19a mk_idx 0, status ok - first fill
  - S_FIFO queue class unidentified -> positions front_at_first_fill|behind_at_first_fill; ahead at acceptance [24.00, 442.97] sh, at the first fill [0.00, 320.00] sh; tied first fills 0, fronts after a partial 0; lo-clock differs False; S_NOPRIO has no queue (price-time premise stated, not tested)
  - fresh-level lives compatible with the first fill: level 51 created ws+2.077s, status unknown/untimed_record_only, creation context deep, alt_netting_possible False, alt_first_other_address False, timing_basis , members 5
  - candidate / fill rows (queue_front_v2) of this hash: none
  - endpoint occupancy (queue_front_v2 occupancy.csv): endpoint occupancy cell_occupied_at_window_lo at ws+2.325s; size_at_window_lo 25; post_grade G4_scenario_only; lookup_status ok (an endpoint observation, never an actual-post queue label)
- **4a ENDING** — **TIED** (2 retained histories) · grade `cancel_compatible_ending` · filled · census: rank1 20261024 pull_possible:S_FIFO:cancel_possible 447; pull_possible:S_NOPRIO:cancel_possible 696
  - retained histories: EX | PL
  - hidden action that changes the explanation: an unobserved cancel of the remainder (PL: withdrawn [0.01, inf] sh in [ws+78.506s, ws+80.524s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - feasible end histories (union) EX|PL; per scenario S_FIFO {EX,PL}, S_FIFO_LO {EX,PL}, S_NOPRIO {EX,PL}
  - cancel-compatible views (pulls.csv): S_FIFO, S_NOPRIO; S_FIFO: removal_class constrained, removal set cancelled_or_expired|exhausted, window [ws+78.506s, ws+80.524s] (ok), identification tied, pull_identified False, most supported book_toward_5s+mid_against_5s (compatible); S_NOPRIO: removal_class constrained, removal set cancelled_or_expired|exhausted, window [ws+78.506s, ws+80.524s] (ok), identification tied, pull_identified False, most supported book_toward_5s+mid_against_5s (compatible)
  - saved END logic rows: inventory_driven_pull unidentified (window condition compatible); pull_after_adverse_move unidentified (window condition timing_sensitive)
- **4b REPOST** — **TIED** (3 retained histories) · grade `scenario_dependent_link` · filled · census: rank1 20261024 successor_reprice sequential_under_scenarios: 150 of 505
  - retained histories: replace | replenish | unrelated
  - hidden action that changes the explanation: whether the successor 0x89e05c2583 was a new submission replacing this order (replace), a top-up (replenish), an order already resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent
  - successor: reprice to 0x89e05c2583 (first of 2 filled successor(s)); link status sequential_identified -> sequential_under_scenarios; gap [33002, 37154] ms; d_ticks -1; identified repost: no
  - saved child links: reprice 0x89e05c2583 (sequential_identified; gap [33002, 37154] ms; replace|replenish|unrelated); refill 0xaa7034170d (sequential_identified; gap [34054, 117901] ms; replace|replenish|unrelated)
  - saved parent links (was this post itself a repost?): reprice 0xfebc9ec82a (tied; gap [-78199, 2318] ms; overlap|replace|replenish|unrelated) - never identified (unrelated stays feasible)
- **5 TAKES** — **ONE-HISTORY** (1 retained history) · grade `observed_execution` · filled
  - every retained history agrees: D0/D1000/pend: q0,q0,same,same,opposing,opposing,same,same,opposing,same,same,same
  - take 01a16f56-8fa6: YES sell 5.85 sh @ 0.520, settle ws+6.278s, anchor ws+4.000s (assigned); q_D0 +0.00 -> D0 q0, D1000 q0, pend q0; class from_flat (rec from_flat / cert from_flat / sim ); O4 reduces_none; status ok
  - take 01a16f56-9270: YES sell 8.37 sh @ 0.510, settle ws+6.992s, anchor ws+4.000s (assigned); q_D0 +0.00 -> D0 q0, D1000 q0, pend q0; class timing_sensitive (rec add / cert from_flat / sim ); O4 reduces_none; status ok
  - take 01a16f57-42af: YES sell 6.26 sh @ 0.600, settle ws+52.111s, anchor ws+50.000s (assigned); q_D0 -134.23 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim ); O4 reduces_none; status ok
  - take 01a16f57-652f: YES sell 5.00 sh @ 0.600, settle ws+60.943s, anchor ws+58.000s (assigned); q_D0 -170.48 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim ); O4 reduces_none; status ok
  - take 01a16f57-76e9: YES buy 5.67 sh @ 0.510, settle ws+65.481s, anchor ws+63.000s (assigned); q_D0 -175.48 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim ); O4 reduces_all; status ok
  - take 01a16f57-a94b: YES buy 0.27 sh @ 0.560, settle ws+78.379s, anchor ws+76.000s (assigned); q_D0 -109.81 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim ); O4 reduces_all; status ok
  - take 01a16f57-c9d0: YES sell 15.00 sh @ 0.690, settle ws+86.704s, anchor ws+84.000s (assigned); q_D0 -218.75 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim add); O4 reduces_none; status simultaneous_action
  - take 01a16f57-c9d0: YES sell 2.89 sh @ 0.690, settle ws+86.704s, anchor ws+84.000s (assigned); q_D0 -218.75 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim add); O4 reduces_none; status simultaneous_action
  - take 01a16f57-d5d5: YES buy 26.19 sh @ 0.670, settle ws+89.781s, anchor ws+87.000s (assigned); q_D0 -236.64 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim ); O4 reduces_all; status ok
  - take 01a16f58-1b63: YES sell 1.51 sh @ 0.670, settle ws+107.587s, anchor ws+105.000s (assigned); q_D0 -121.03 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim add); O4 reduces_none; status simultaneous_action
  - take 01a16f58-1b63: YES sell 2.98 sh @ 0.670, settle ws+107.587s, anchor ws+105.000s (assigned); q_D0 -121.03 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim add); O4 reduces_none; status simultaneous_action
  - take 01a16f58-1b6a: YES sell 15.00 sh @ 0.660, settle ws+107.594s, anchor ws+105.000s (assigned); q_D0 -121.03 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim ); O4 reduces_none; status ok
- **6 VALUE** — **ONE-HISTORY** (1 retained history) · grade `recorded_fill_accounting` · filled
  - every retained history agrees: recorded_fills_under_every_retained_history
  - episode lot legs (status ok, not self-trades, settle_ms in [ws+2.325s, ws+110.524s]): 47 (maker 35, taker 12), 548.79 sh; excluded rows in the window 0; matched (mq) 548.79 sh, residual (rq) 0.00 sh
  - episode, all legs: 10s valid 47/47 rows, 548.79/548.79 sh, gross +4.43 $, net +3.73 $, +0.81 c/sh; 60s valid 47/47 rows, 548.79/548.79 sh, gross -4.82 $, net -5.52 $, -0.88 c/sh; END valid 47/47 rows, 548.79/548.79 sh, gross -42.23 $, net -42.93 $, -7.70 c/sh; common cohort 47 rows: 10s +0.81, 60s -0.88, end -7.70 c/sh; resolution (net_res) valid 47/47: -45.46 $ (-8.28 c/sh)
  - episode, maker legs: 10s valid 35/35 rows, 453.79/453.79 sh, gross +2.88 $, net +2.88 $, +0.63 c/sh; 60s valid 35/35 rows, 453.79/453.79 sh, gross -7.77 $, net -7.77 $, -1.71 c/sh; END valid 35/35 rows, 453.79/453.79 sh, gross -31.15 $, net -31.15 $, -6.86 c/sh; common cohort 35 rows: 10s +0.63, 60s -1.71, end -6.86 c/sh; resolution (net_res) valid 35/35: -33.09 $ (-7.29 c/sh)
  - episode, taker legs: 10s valid 12/12 rows, 95.00/95.00 sh, gross +1.55 $, net +0.85 $, +1.63 c/sh; 60s valid 12/12 rows, 95.00/95.00 sh, gross +2.95 $, net +2.25 $, +3.11 c/sh; END valid 12/12 rows, 95.00/95.00 sh, gross -11.08 $, net -11.78 $, -11.66 c/sh; common cohort 12 rows: 10s +1.63, 60s +3.11, end -11.66 c/sh; resolution (net_res) valid 12/12: -12.37 $ (-13.02 c/sh)
  - completed turnover (closes.parquet, closes.py FIFO): entries closed inside the episode 38 close(s) (+0 dust), 226.72 sh, pnl -13.80 $, net -14.26 $, hold p50 56.4 s; entries closed after it 15 close(s) (+0 dust), 95.35 sh, pnl +5.51 $, net +4.87 $, hold p50 32.6 s; closes of earlier inventory 0 close(s) (+0 dust), 0.00 sh, pnl +0.00 $, net +0.00 $, hold p50 na s
  - residual of the episode legs: 0.00 sh unclosed; value to END +0.00 $ (m_end finite for 47/47 legs), to resolution +0.00 $ (47/47 resolved)
  - case order 0x88c0095c1e own maker rows (1 rows, 1 ok): 10s valid 1/1 rows, 30.00/30.00 sh, gross -0.75 $, net -0.75 $, -2.50 c/sh; 60s valid 1/1 rows, 30.00/30.00 sh, gross +8.85 $, net +8.85 $, +29.50 c/sh; END valid 1/1 rows, 30.00/30.00 sh, gross -8.40 $, net -8.40 $, -28.00 c/sh; common cohort 1 rows: 10s -2.50, 60s +29.50, end -28.00 c/sh; resolution (net_res) valid 1/1: -9.30 $ (-31.00 c/sh); mq 30.00 sh, rq 0.00 sh; its closes 4 close(s) (+0 dust), 30.00 sh, pnl +2.38 $, net +2.27 $, hold p50 32.9 s
  - address in this market (closes_markets): 360 closes, 2045.72 sh, close net +168.32 $; residual 0.00 sh, to END +0.00 $, to resolution +0.00 $; status ok
  - matched-gap interval accounting (matched_gap_r1): carry (primary) status ok: close net +168.32 $ on 2045.72 sh (+8.23 c/sh), carry +0.00, resid +0.00, total +168.10 $; reading our_turnover_loses; zero (beside) status ok: close net +168.32 $ on 2045.72 sh (+8.23 c/sh), carry +0.00, resid +0.00, total +168.10 $; reading our_turnover_loses
- **AGREEMENT** histories agree on: takes, value; TIED: pre_state, post, queue, ending, repost; pending: none; first branching link: pre_state
  - pre_state: the posting second inside k = 2..80 (ws+2..ws+81 s): features spread, depth_touch_own, depth_touch_opp, depth_2_own, btc_5s_aligned, fair_gap, time_left change across it
  - post: the posting instant inside [ws+2.325s, ws+80.506s] (78181 ms; scenario union estimate only); the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - queue: whether orders were ahead at the first fill (S_FIFO ahead [0.00, 320.00] sh at the first fill; [24.00, 442.97] sh at acceptance)
  - ending: an unobserved cancel of the remainder (PL: withdrawn [0.01, inf] sh in [ws+78.506s, ws+80.524s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - repost: whether the successor 0x89e05c2583 was a new submission replacing this order (replace), a top-up (replenish), an order already resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent

## Chain 2 - rank 1: episode 20261024_3702960_r1_e006

day 20261024, market 3702960, address 0x6da6cb464f92ae7ad4ec3d239c81719cb1d0ae03; case order 0x02fb943dae; cases.csv data row 2; episode window [ws+211.237s, ws+300.000s]

- **1 PRE-STATE** — **TIED** (3 retained histories) · grade `observed_state` · filled
  - retained histories: spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(60,120] | spread=2; depth_touch_own=(50,200]; depth_touch_opp=(50,200]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(60,120] | spread=2; depth_touch_own=>200; depth_touch_opp=(0,50]; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(60,120]
  - hidden action that changes the explanation: the posting second inside k = 211..213 (ws+211..ws+214 s): features spread, depth_touch_own, depth_touch_opp, depth_2_own change across it
  - window seconds k = 211..213 (3 in the market window, 0 outside; 3 saved rows at side ask; time key t_ms); post window [ws+211.237s, ws+213.303s]
  - features (D0 = the state at t; columns spread_bin_D0, ...): spread {'1': 1, '2': 2}; depth_touch_own {'(50,200]': 2, '>200': 1}; depth_touch_opp {'(0,50]': 1, '(50,200]': 1, '>200': 1}; depth_2_own {'(0,50]': 1, '(50,200]': 2}; btc_5s_aligned {'flat': 3}; fair_gap {'>3': 3}; flow_5s {'>0': 3}; time_left {'(60,120]': 3}
  - row statuses (status_D0): none
  - D1000 (the whole state lagged one second): 2 distinct state vector(s) vs 3 at D0 - a different set (timing-sensitive)
  - visible fresh creations on this side in the window: 7; at the order's cell (tick 13): 3702960|ask|13|4234 (improve) (an observed market creation; the order is at most a candidate creator, section 3)
- **2 POST** — **TIED** (4 retained histories) · grade `post_window:G4_scenario_only` · filled · census: rank1 20261024 G4_scenario_only: 348 of 746 first-filled orders
  - retained histories: late_post_in_scenario_window + EX | late_post_in_scenario_window + PL | earlier_post_masked_by_replenishment + EX | earlier_post_masked_by_replenishment + PL
  - hidden action that changes the explanation: the posting instant inside [ws+211.237s, ws+213.303s] (2066 ms; scenario union estimate only); the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - order 0x02fb943dae: ask at 13c (side and tick observed in its fills); first fill <= ws+213.303s
  - window [ws+211.237s, ws+213.303s], width 2066 ms (le_10s); per-scenario lower ends S_FIFO ws+213.225s, S_FIFO_LO ws+211.237s, S_NOPRIO ws+213.225s; lifecycle post_class assumption_sensitive; packet status ok -> grade G4_scenario_only (scenario union estimate only)
  - size: filled 6.67 sh = a LOWER bound (observed executions); submitted size unknown; feasible EX (exhausted: = 6.67 sh, equal to the filled size only under EX); PL (pulled: [6.68, inf] sh); bounds [6.67, inf] sh
  - size status tied; prior picks (prior-driven estimates, not evidence): P_cmd ('EX', 6.6666); P_drop ('PL', 66.6666); P_lot ('PL', 10.0); P_clip ('PL', 30.0)
  - same-hash taker legs (a crossing at arrival): 0 leg(s), 0.00 sh
  - saved POST logic row: queue_priority_before_demand contradicted -> timing_sensitive (grade B_constrained -> C_scenario)
- **3 QUEUE** — **TIED** (4 retained histories) · grade `scenario_queue_history` · filled
  - retained histories: front_at_first_fill @ level 236 (unknown) | front_at_first_fill @ level 273 (unknown) | behind_at_first_fill @ level 236 (unknown) | behind_at_first_fill @ level 273 (unknown)
  - hidden action that changes the explanation: whether orders were ahead at the first fill (S_FIFO ahead [0.00, 177.15] sh at the first fill; [0.00, 183.82] sh at acceptance); which fresh-level life the first fill matched in (level 236 (unknown), level 273 (unknown))
  - order 0x02fb943dae: 1 fill(s), 6.67 sh
    - fill 1: 6.67 sh, hard window [ws+211.303s, ws+213.303s], sid 01a16d82-3937 mk_idx 0, status ok - first fill
  - S_FIFO queue class unidentified -> positions front_at_first_fill|behind_at_first_fill; ahead at acceptance [0.00, 183.82] sh, at the first fill [0.00, 177.15] sh; tied first fills 0, fronts after a partial 0; lo-clock differs False; S_NOPRIO has no queue (price-time premise stated, not tested)
  - fresh-level lives compatible with the first fill: level 236 created ws+178.629s, status unknown/untimed_record_only, creation context improve, alt_netting_possible False, alt_first_other_address False, timing_basis , members 1; level 273 created ws+213.275s, status unknown/untimed_record_only, creation context improve, alt_netting_possible False, alt_first_other_address False, timing_basis , members 6
  - candidate / fill rows (queue_front_v2) of this hash: none
  - endpoint occupancy (queue_front_v2 occupancy.csv): endpoint occupancy cell_occupied_at_window_lo at ws+211.237s; size_at_window_lo 183.82; post_grade G4_scenario_only; lookup_status ok (an endpoint observation, never an actual-post queue label)
- **4a ENDING** — **TIED** (2 retained histories) · grade `cancel_compatible_ending` · filled · census: rank1 20261024 pull_possible:S_FIFO:cancel_possible 447; pull_possible:S_NOPRIO:cancel_possible 696
  - retained histories: EX | PL
  - hidden action that changes the explanation: an unobserved cancel of the remainder (PL: withdrawn [0.01, inf] sh in [ws+211.303s, ws+271.016s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - feasible end histories (union) EX|PL; per scenario S_FIFO {EX,PL}, S_FIFO_LO {EX,PL}, S_NOPRIO {EX,PL}
  - cancel-compatible views (pulls.csv): S_FIFO, S_NOPRIO; S_FIFO: removal_class assumption_sensitive, removal set cancelled_or_expired|exhausted, window [ws+211.303s, ws+271.016s] (ok), identification tied, pull_identified False, most supported btc_against_1s+btc_against_5s+book_toward_5s+book_away_5s+mid_against_5s+behind_touch+own_fill_5s+inv_built_5s (timing_sensitive); S_NOPRIO: removal_class assumption_sensitive, removal set cancelled_or_expired|exhausted, window [ws+211.303s, ws+271.016s] (ok), identification tied, pull_identified False, most supported btc_against_1s+btc_against_5s+book_toward_5s+book_away_5s+mid_against_5s+behind_touch+own_fill_5s+inv_built_5s (timing_sensitive)
  - saved END logic rows: inventory_driven_pull unidentified (window condition timing_sensitive); pull_after_adverse_move unidentified (window condition timing_sensitive)
- **4b REPOST** — **TIED** (4 retained histories) · grade `scenario_dependent_link` · filled · census: rank1 20261024 successor_reprice tied: 355 of 505
  - retained histories: overlap | replace | replenish | unrelated
  - hidden action that changes the explanation: whether the successor 0xdef88dc345 was a new submission replacing this order (replace), a top-up (replenish), an order already resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent
  - successor: reprice to 0xdef88dc345 (first of 1 filled successor(s)); link status tied -> tied; gap [-57944, 2176] ms; d_ticks -1; identified repost: no
  - saved child links: reprice 0xdef88dc345 (tied; gap [-57944, 2176] ms; overlap|replace|replenish|unrelated)
  - saved parent links (was this post itself a repost?): reprice 0x38bd2b23d2 (tied; gap [-82791, 3458] ms; overlap|replace|replenish|unrelated) - never identified (unrelated stays feasible)
- **5 TAKES** — **ONE-HISTORY** (1 retained history) · grade `observed_execution` · filled
  - every retained history agrees: D0/D1000/pend: same,same,same,same,same,opposing,opposing,opposing
  - take 01a16d82-3397: YES sell 45.01 sh @ 0.140, settle ws+211.863s, anchor ws+209.000s (assigned); q_D0 -17.90 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim ); O4 reduces_none; status ok
  - take 01a16d82-3400: YES sell 3.92 sh @ 0.140, settle ws+211.968s, anchor ws+209.000s (assigned); q_D0 -17.90 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim add); O4 reduces_none; status simultaneous_action
  - take 01a16d82-3400: YES sell 6.95 sh @ 0.130, settle ws+211.968s, anchor ws+209.000s (assigned); q_D0 -17.90 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim add); O4 reduces_none; status simultaneous_action
  - take 01a16d82-378b: YES sell 5.17 sh @ 0.130, settle ws+212.875s, anchor ws+210.000s (assigned); q_D0 -17.90 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim ); O4 reduces_none; status ok
  - take 01a16d82-37a8: YES sell 16.37 sh @ 0.130, settle ws+212.904s, anchor ws+210.000s (assigned); q_D0 -17.90 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim ); O4 reduces_none; status ok
  - take 01a16d83-1b6e: YES buy 38.00 sh @ 0.100, settle ws+271.214s, anchor ws+269.000s (assigned); q_D0 -85.26 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim reduce_partial); O4 reduces_all; status simultaneous_action
  - take 01a16d83-1b6e: YES buy 5.00 sh @ 0.110, settle ws+271.214s, anchor ws+269.000s (assigned); q_D0 -85.26 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim reduce_partial); O4 reduces_all; status simultaneous_action
  - take 01a16d83-1b6e: YES buy 8.47 sh @ 0.130, settle ws+271.214s, anchor ws+269.000s (assigned); q_D0 -85.26 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim reduce_partial); O4 reduces_all; status simultaneous_action
- **6 VALUE** — **ONE-HISTORY** (1 retained history) · grade `recorded_fill_accounting` · filled
  - every retained history agrees: recorded_fills_under_every_retained_history
  - episode lot legs (status ok, not self-trades, settle_ms in [ws+211.237s, ws+300.000s]): 21 (maker 13, taker 8), 250.37 sh; excluded rows in the window 0; matched (mq) 221.68 sh, residual (rq) 28.68 sh
  - episode, all legs: 10s valid 21/21 rows, 250.37/250.37 sh, gross +2.20 $, net +1.88 $, +0.88 c/sh; 60s valid 11/21 rows, 171.98/250.37 sh, gross +1.84 $, net +1.63 $, +1.07 c/sh (invalid {'beyond_end': 10}); END valid 21/21 rows, 250.37/250.37 sh, gross +6.19 $, net +5.87 $, +2.47 c/sh; common cohort 11 rows: 10s +2.88, 60s +1.07, end +4.96 c/sh; resolution (net_res) valid 21/21: +6.59 $ (+2.63 c/sh)
  - episode, maker legs: 10s valid 13/13 rows, 121.46/121.46 sh, gross +1.63 $, net +1.63 $, +1.34 c/sh; 60s valid 6/13 rows, 94.55/121.46 sh, gross -0.30 $, net -0.30 $, -0.32 c/sh (invalid {'beyond_end': 7}); END valid 13/13 rows, 121.46/121.46 sh, gross +2.64 $, net +2.64 $, +2.17 c/sh; common cohort 6 rows: 10s +1.86, 60s -0.32, end +2.77 c/sh; resolution (net_res) valid 13/13: +1.73 $ (+1.43 c/sh)
  - episode, taker legs: 10s valid 8/8 rows, 128.90/128.90 sh, gross +0.58 $, net +0.26 $, +0.45 c/sh; 60s valid 5/8 rows, 77.43/128.90 sh, gross +2.14 $, net +1.92 $, +2.76 c/sh (invalid {'beyond_end': 3}); END valid 8/8 rows, 128.90/128.90 sh, gross +3.55 $, net +3.23 $, +2.75 c/sh; common cohort 5 rows: 10s +4.13, 60s +2.76, end +7.63 c/sh; resolution (net_res) valid 8/8: +4.86 $ (+3.77 c/sh)
  - completed turnover (closes.parquet, closes.py FIFO): entries closed inside the episode 15 close(s) (+0 dust), 101.89 sh, pnl +5.41 $, net +5.10 $, hold p50 58.3 s; entries closed after it 0 close(s) (+0 dust), 0.00 sh, pnl +0.00 $, net +0.00 $, hold p50 na s; closes of earlier inventory 3 close(s) (+0 dust), 17.90 sh, pnl -0.36 $, net -0.36 $, hold p50 10.5 s
  - residual of the episode legs: 28.68 sh unclosed; value to END +1.32 $ (m_end finite for 21/21 legs), to resolution +3.04 $ (21/21 resolved)
  - case order 0x02fb943dae own maker rows (1 rows, 1 ok): 10s valid 1/1 rows, 6.67/6.67 sh, gross +0.30 $, net +0.30 $, +4.50 c/sh; 60s valid 1/1 rows, 6.67/6.67 sh, gross +0.23 $, net +0.23 $, +3.50 c/sh; END valid 1/1 rows, 6.67/6.67 sh, gross +0.47 $, net +0.47 $, +7.00 c/sh; common cohort 1 rows: 10s +4.50, 60s +3.50, end +7.00 c/sh; resolution (net_res) valid 1/1: +0.87 $ (+13.00 c/sh); mq 0.00 sh, rq 6.67 sh; its closes 0 close(s) (+0 dust), 0.00 sh, pnl +0.00 $, net +0.00 $, hold p50 na s
  - address in this market (closes_markets): 227 closes, 1332.93 sh, close net -6.43 $; residual 28.68 sh, to END +1.32 $, to resolution +3.04 $; status ok
  - matched-gap interval accounting (matched_gap_r1): carry (primary) status ok: close net -6.43 $ on 1332.93 sh (-0.48 c/sh), carry +0.00, resid +1.32, total -5.13 $; reading exposure_gap; zero (beside) status ok: close net -6.43 $ on 1332.93 sh (-0.48 c/sh), carry +0.00, resid +1.32, total -5.13 $; reading exposure_gap
- **AGREEMENT** histories agree on: takes, value; TIED: pre_state, post, queue, ending, repost; pending: none; first branching link: pre_state
  - pre_state: the posting second inside k = 211..213 (ws+211..ws+214 s): features spread, depth_touch_own, depth_touch_opp, depth_2_own change across it
  - post: the posting instant inside [ws+211.237s, ws+213.303s] (2066 ms; scenario union estimate only); the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - queue: whether orders were ahead at the first fill (S_FIFO ahead [0.00, 177.15] sh at the first fill; [0.00, 183.82] sh at acceptance); which fresh-level life the first fill matched in (level 236 (unknown), level 273 (unknown))
  - ending: an unobserved cancel of the remainder (PL: withdrawn [0.01, inf] sh in [ws+211.303s, ws+271.016s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - repost: whether the successor 0xdef88dc345 was a new submission replacing this order (replace), a top-up (replenish), an order already resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent

## Chain 3 - rank 2: episode 20261024_3702960_r2_e007

day 20261024, market 3702960, address 0x7d32c568a9bea1714036972601c51f4902c67091; case order 0x084be2d8a4; cases.csv data row 17; episode window [ws+4.027s, ws+37.626s]

- **1 PRE-STATE** — **ONE-HISTORY** (1 retained history) · grade `observed_state` · filled
  - every retained history agrees: spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300]
  - window seconds k = 4..4 (1 in the market window, 0 outside; 1 saved rows at side bid; time key t_ms); post window [ws+4.027s, ws+4.785s]
  - features (D0 = the state at t; columns spread_bin_D0, ...): spread {'1': 1}; depth_touch_own {'(50,200]': 1}; depth_touch_opp {'>200': 1}; depth_2_own {'(50,200]': 1}; btc_5s_aligned {'flat': 1}; fair_gap {'(0,1]': 1}; flow_5s {'>0': 1}; time_left {'(240,300]': 1}
  - row statuses (status_D0): none
  - D1000 (the whole state lagged one second): 1 distinct state vector(s) vs 1 at D0 - a different set (timing-sensitive)
  - visible fresh creations on this side in the window: 0; at the order's cell (tick 49): none (an observed market creation; the order is at most a candidate creator, section 3)
- **2 POST** — **TIED** (4 retained histories) · grade `post_window:G1_conflict` · filled · census: rank2 20261024 G1_conflict: 41 of 368 first-filled orders
  - retained histories: posting_time_unresolved_early + EX | posting_time_unresolved_early + PL | posting_time_unresolved_late + EX | posting_time_unresolved_late + PL
  - hidden action that changes the explanation: the posting instant inside [ws+4.027s, ws+4.785s] (758 ms; void: conflict flags); the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - order 0x084be2d8a4: bid at 49c (side and tick observed in its fills); first fill <= ws+4.785s
  - window [ws+4.027s, ws+4.785s], width 758 ms (le_1s); per-scenario lower ends S_FIFO ws+4.027s, S_FIFO_LO ws+4.027s, S_NOPRIO ws+4.027s; lifecycle post_class assumption_sensitive; packet status capacity_bounds_void+clock_capacity_conflict+post_after_first_fill -> grade G1_conflict (void: conflict flags)
  - size: filled 50.00 sh = a LOWER bound (observed executions); submitted size unknown; feasible EX (exhausted: = 50.00 sh, equal to the filled size only under EX); PL (pulled: [50.01, 51.96] sh); bounds [50.00, 51.96] sh
  - size status conflict; prior picks (prior-driven estimates, not evidence): P_cmd ('EX', 50.0); P_drop ('PL', 51.960784); P_lot ('EX', 50.0); P_clip ('EX', 50.0)
  - same-hash taker legs (a crossing at arrival): 0 leg(s), 0.00 sh
  - saved POST logic row: queue_priority_before_demand contradicted -> unidentified (grade B_constrained -> E_unresolved)
- **3 QUEUE** — **TIED** (4 retained histories) · grade `scenario_queue_history` · filled
  - retained histories: front_at_first_fill @ level 71 (ambiguous_order) + masked_netting_possible | behind_at_first_fill @ level 71 (ambiguous_order) + masked_netting_possible | front_at_first_fill @ level 71 (ambiguous_order) + as_saved | behind_at_first_fill @ level 71 (ambiguous_order) + as_saved
  - hidden action that changes the explanation: whether orders were ahead at the first fill (S_FIFO ahead [6.86, 26.63] sh at the first fill; [21.86, 151.86] sh at acceptance); a cancellation / replenishment masked inside a frame interval (net book change; section 5 finding 1)
  - order 0x084be2d8a4: 3 fill(s), 50.00 sh (one order: its first fill and later fills of the same hash)
    - fill 1: 13.14 sh, hard window [ws+2.785s, ws+4.785s], sid 01a16d7f-0ab1 mk_idx 1, status ok - first fill
    - fill 2: 1.96 sh, hard window [ws+3.054s, ws+5.054s], sid 01a16d7f-0bbe mk_idx 0, status ok - later fill of the same hash (the same order)
    - fill 3: 34.90 sh, hard window [ws+5.030s, ws+7.030s], sid 01a16d7f-1376 mk_idx 0, status ok - later fill of the same hash (the same order)
  - S_FIFO queue class conflicting -> positions front_at_first_fill|behind_at_first_fill; ahead at acceptance [21.86, 151.86] sh, at the first fill [6.86, 26.63] sh; tied first fills 0, fronts after a partial 1; lo-clock differs False; S_NOPRIO has no queue (price-time premise stated, not tested)
  - fresh-level lives compatible with the first fill: level 71 created ws+1.321s, status ambiguous_order/untimed_record_could_be_first, creation context re_establish, alt_netting_possible True, alt_first_other_address True, timing_basis book_candidate, members 13
  - candidate / fill rows (queue_front_v2) of this hash: none
  - endpoint occupancy (queue_front_v2 occupancy.csv): endpoint occupancy cell_occupied_at_window_lo at ws+4.027s; size_at_window_lo 21.861526076923077; post_grade G1_conflict; lookup_status ok (an endpoint observation, never an actual-post queue label)
- **4a ENDING** — **TIED** (2 retained histories) · grade `scenario_dependent_ending` · filled · census: rank2 20261024 not cancel-compatible in S_FIFO / S_NOPRIO
  - retained histories: EX | PL
  - hidden action that changes the explanation: an unobserved cancel of the remainder (PL: withdrawn [0.01, 1.96] sh in [ws+5.030s, ws+7.626s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - feasible end histories (union) EX|PL; per scenario S_FIFO {EX}, S_FIFO_LO {EX,PL}, S_NOPRIO {EX}
  - cancel-compatible views (pulls.csv): none
  - saved END logic rows: inventory_driven_pull unidentified (window condition compatible); pull_after_adverse_move unidentified (window condition contradicted)
- **4b REPOST** — **TIED** (3 retained histories) · grade `scenario_dependent_link` · filled · census: rank2 20261024 successor_reprice sequential_under_scenarios: 46 of 245
  - retained histories: replace | replenish | unrelated
  - hidden action that changes the explanation: whether the successor 0xd0f156e83a was a new submission replacing this order (replace), a top-up (replenish), an order already resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent
  - successor: reprice to 0xd0f156e83a (first of 2 filled successor(s)); link status sequential_identified -> sequential_under_scenarios; gap [53, 4290] ms; d_ticks -1; identified repost: no
  - saved child links: reprice 0xd0f156e83a (sequential_identified; gap [53, 4290] ms; replace|replenish|unrelated); reprice 0x4f4c1ec8c6 (tied; gap [-5211, 5329] ms; overlap|replace|replenish|unrelated)
  - saved parent links (was this post itself a repost?): none - never identified (unrelated stays feasible)
- **5 TAKES** — **ONE-HISTORY** (1 retained history) · grade `observed_execution` · filled
  - every retained history agrees: D0/D1000/pend: q0,same,same,same,opposing
  - take 01a16d7f-08ca: YES buy 4.00 sh @ 0.500, settle ws+4.298s, anchor ws+2.000s (assigned); q_D0 +0.00 -> D0 q0, D1000 q0, pend q0; class from_flat (rec from_flat / cert from_flat / sim ); O4 reduces_none; status ok
  - take 01a16d7f-19dc: YES buy 15.00 sh @ 0.490, settle ws+8.668s, anchor ws+6.000s (assigned); q_D0 +19.10 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim ); O4 reduces_none; status ok
  - take 01a16d7f-1dbe: YES buy 13.38 sh @ 0.490, settle ws+9.662s, anchor ws+7.000s (assigned); q_D0 +19.10 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim add); O4 reduces_none; status simultaneous_action
  - take 01a16d7f-1dbe: YES buy 6.62 sh @ 0.490, settle ws+9.662s, anchor ws+7.000s (assigned); q_D0 +19.10 -> D0 same, D1000 same, pend same; class add (rec add / cert add / sim add); O4 reduces_none; status simultaneous_action
  - take 01a16d7f-6d67: YES sell 20.00 sh @ 0.430, settle ws+30.055s, anchor ws+28.000s (assigned); q_D0 +239.00 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim ); O4 reduces_all; status ok
- **6 VALUE** — **ONE-HISTORY** (1 retained history) · grade `recorded_fill_accounting` · filled
  - every retained history agrees: recorded_fills_under_every_retained_history
  - episode lot legs (status ok, not self-trades, settle_ms in [ws+4.027s, ws+37.626s]): 33 (maker 28, taker 5), 468.91 sh; excluded rows in the window 0; matched (mq) 468.91 sh, residual (rq) 0.00 sh
  - episode, all legs: 10s valid 33/33 rows, 468.91/468.91 sh, gross +23.39 $, net +22.89 $, +4.99 c/sh; 60s valid 33/33 rows, 468.91/468.91 sh, gross -37.42 $, net -37.92 $, -7.98 c/sh; END valid 33/33 rows, 468.91/468.91 sh, gross -45.83 $, net -46.33 $, -9.77 c/sh; common cohort 33 rows: 10s +4.99, 60s -7.98, end -9.77 c/sh; resolution (net_res) valid 33/33: -52.05 $ (-11.10 c/sh)
  - episode, maker legs: 10s valid 28/28 rows, 409.91/409.91 sh, gross +21.12 $, net +21.12 $, +5.15 c/sh; 60s valid 28/28 rows, 409.91/409.91 sh, gross -30.11 $, net -30.11 $, -7.35 c/sh; END valid 28/28 rows, 409.91/409.91 sh, gross -36.42 $, net -36.42 $, -8.89 c/sh; common cohort 28 rows: 10s +5.15, 60s -7.35, end -8.89 c/sh; resolution (net_res) valid 28/28: -41.34 $ (-10.09 c/sh)
  - episode, taker legs: 10s valid 5/5 rows, 59.00/59.00 sh, gross +2.27 $, net +1.78 $, +3.86 c/sh; 60s valid 5/5 rows, 59.00/59.00 sh, gross -7.30 $, net -7.80 $, -12.38 c/sh; END valid 5/5 rows, 59.00/59.00 sh, gross -9.41 $, net -9.91 $, -15.95 c/sh; common cohort 5 rows: 10s +3.86, 60s -12.38, end -15.95 c/sh; resolution (net_res) valid 5/5: -10.70 $ (-18.14 c/sh)
  - completed turnover (closes.parquet, closes.py FIFO): entries closed inside the episode 26 close(s) (+0 dust), 183.95 sh, pnl -11.81 $, net -12.31 $, hold p50 20.6 s; entries closed after it 11 close(s) (+0 dust), 101.01 sh, pnl -16.33 $, net -16.33 $, hold p50 7.0 s; closes of earlier inventory 0 close(s) (+0 dust), 0.00 sh, pnl +0.00 $, net +0.00 $, hold p50 na s
  - residual of the episode legs: 0.00 sh unclosed; value to END -0.00 $ (m_end finite for 33/33 legs), to resolution -0.00 $ (33/33 resolved)
  - case order 0x084be2d8a4 own maker rows (3 rows, 3 ok): 10s valid 3/3 rows, 50.00/50.00 sh, gross -0.75 $, net -0.75 $, -1.50 c/sh; 60s valid 3/3 rows, 50.00/50.00 sh, gross -14.25 $, net -14.25 $, -28.50 c/sh; END valid 3/3 rows, 50.00/50.00 sh, gross -21.50 $, net -21.50 $, -43.00 c/sh; common cohort 3 rows: 10s -1.50, 60s -28.50, end -43.00 c/sh; resolution (net_res) valid 3/3: -24.50 $ (-49.00 c/sh); mq 50.00 sh, rq 0.00 sh; its closes 4 close(s) (+0 dust), 50.00 sh, pnl -3.46 $, net -3.46 $, hold p50 23.7 s
  - address in this market (closes_markets): 204 closes, 1546.53 sh, close net +1.58 $; residual 2.04 sh, to END -0.10 $, to resolution -0.22 $; status ok
  - matched-gap interval accounting (matched_gap_r1): carry (primary) status ok: close net +1.58 $ on 1546.53 sh (+0.10 c/sh), carry +0.00, resid -0.10, total +1.48 $; reading our_turnover_loses; zero (beside) status ok: close net +1.58 $ on 1546.53 sh (+0.10 c/sh), carry +0.00, resid -0.10, total +1.48 $; reading our_turnover_loses
- **AGREEMENT** histories agree on: pre_state, takes, value; TIED: post, queue, ending, repost; pending: none; first branching link: post
  - post: the posting instant inside [ws+4.027s, ws+4.785s] (758 ms; void: conflict flags); the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - queue: whether orders were ahead at the first fill (S_FIFO ahead [6.86, 26.63] sh at the first fill; [21.86, 151.86] sh at acceptance); a cancellation / replenishment masked inside a frame interval (net book change; section 5 finding 1)
  - ending: an unobserved cancel of the remainder (PL: withdrawn [0.01, 1.96] sh in [ws+5.030s, ws+7.626s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - repost: whether the successor 0xd0f156e83a was a new submission replacing this order (replace), a top-up (replenish), an order already resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent

## Chain 4 - rank 2: episode 20261024_3712725_r2_e015

day 20261024, market 3712725, address 0x7d32c568a9bea1714036972601c51f4902c67091; case order 0xa46229318a; cases.csv data row 18; episode window [ws+62.538s, ws+95.085s]

- **1 PRE-STATE** — **TIED** (4 retained histories) · grade `observed_state` · filled
  - retained histories: spread=1; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240] | spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(120,240]
  - hidden action that changes the explanation: the posting second inside k = 62..65 (ws+62..ws+66 s): features depth_touch_own, depth_2_own, fair_gap change across it
  - window seconds k = 62..65 (4 in the market window, 0 outside; 4 saved rows at side bid; time key t_ms); post window [ws+62.538s, ws+65.084s]
  - features (D0 = the state at t; columns spread_bin_D0, ...): spread {'1': 4}; depth_touch_own {'(0,50]': 1, '>200': 3}; depth_touch_opp {'>200': 4}; depth_2_own {'(50,200]': 1, '>200': 3}; btc_5s_aligned {'flat': 4}; fair_gap {'(0,1]': 2, '(1,3]': 1, '>3': 1}; flow_5s {'>0': 4}; time_left {'(120,240]': 4}
  - row statuses (status_D0): none
  - D1000 (the whole state lagged one second): 4 distinct state vector(s) vs 4 at D0 - a different set (timing-sensitive)
  - visible fresh creations on this side in the window: 5; at the order's cell (tick 42): 3712725|bid|42|1670 (improve+re_establish); 3712725|bid|42|1732 (improve+re_establish) (an observed market creation; the order is at most a candidate creator, section 3)
- **2 POST** — **TIED** (2 retained histories) · grade `post_window:G2_bound_constrained` · filled · census: rank2 20261024 G2_bound_constrained: 167 of 368 first-filled orders
  - retained histories: posted_le_10s_before_first_fill + EX | posted_le_10s_before_first_fill + PL
  - hidden action that changes the explanation: the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - order 0xa46229318a: bid at 42c (side and tick observed in its fills); first fill <= ws+65.084s
  - window [ws+62.538s, ws+65.084s], width 2546 ms (le_10s); per-scenario lower ends S_FIFO ws+62.538s, S_FIFO_LO ws+62.538s, S_NOPRIO ws+62.538s; lifecycle post_class constrained; packet status ok -> grade G2_bound_constrained (lifecycle scenario-free bound (conditional on H1 / H2 / capacity and the maximum match lag))
  - size: filled 50.00 sh = a LOWER bound (observed executions); submitted size unknown; feasible EX (exhausted: = 50.00 sh, equal to the filled size only under EX); PL (pulled: [50.01, inf] sh); bounds [50.00, inf] sh
  - size status distinguished_under_priority; prior picks (prior-driven estimates, not evidence): P_cmd ('EX', 50.0); P_drop ('EX', 50.0); P_lot ('EX', 50.0); P_clip ('EX', 50.0)
  - same-hash taker legs (a crossing at arrival): 0 leg(s), 0.00 sh
  - saved POST logic row: queue_priority_before_demand contradicted -> contradicted (grade B_constrained -> B_constrained)
- **3 QUEUE** — **TIED** (4 retained histories) · grade `scenario_queue_history` · filled
  - retained histories: front_at_first_fill @ level 221 (ambiguous_order) + masked_netting_possible | behind_at_first_fill @ level 221 (ambiguous_order) + masked_netting_possible | front_at_first_fill @ level 221 (ambiguous_order) + as_saved | behind_at_first_fill @ level 221 (ambiguous_order) + as_saved
  - hidden action that changes the explanation: whether orders were ahead at the first fill (S_FIFO ahead [0.00, 509.52] sh at the first fill; [0.00, 793.04] sh at acceptance); a cancellation / replenishment masked inside a frame interval (net book change; section 5 finding 1)
  - order 0xa46229318a: 1 fill(s), 50.00 sh
    - fill 1: 50.00 sh, hard window [ws+63.084s, ws+65.084s], sid 01a16f20-86dc mk_idx 6, status ok - first fill
  - S_FIFO queue class unidentified -> positions front_at_first_fill|behind_at_first_fill; ahead at acceptance [0.00, 793.04] sh, at the first fill [0.00, 509.52] sh; tied first fills 0, fronts after a partial 0; lo-clock differs False; S_NOPRIO has no queue (price-time premise stated, not tested)
  - fresh-level lives compatible with the first fill: level 221 created ws+62.583s, status ambiguous_order/sweep_several_makers, creation context re_establish, alt_netting_possible True, alt_first_other_address True, timing_basis book_candidate, members 9
  - candidate / fill rows (queue_front_v2) of this hash: none
  - endpoint occupancy (queue_front_v2 occupancy.csv): endpoint occupancy cell_empty_at_window_lo at ws+62.538s; size_at_window_lo 0; post_grade G2_bound_constrained; lookup_status ok (an endpoint observation, never an actual-post queue label)
- **4a ENDING** — **TIED** (2 retained histories) · grade `cancel_compatible_ending` · filled · census: rank2 20261024 pull_possible:S_NOPRIO:cancel_possible 344
  - retained histories: EX | PL
  - hidden action that changes the explanation: an unobserved cancel of the remainder (PL: withdrawn [0.01, inf] sh in [ws+63.084s, ws+65.085s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - feasible end histories (union) EX|PL; per scenario S_FIFO {EX}, S_FIFO_LO {EX}, S_NOPRIO {EX,PL}
  - cancel-compatible views (pulls.csv): S_NOPRIO; S_NOPRIO: removal_class assumption_sensitive, removal set cancelled_or_expired|exhausted, window [ws+63.084s, ws+65.085s] (ok), identification tied, pull_identified False, most supported btc_against_30s+book_toward_5s+mid_against_5s+own_fill_5s+inv_built_5s (compatible)
  - saved END logic rows: inventory_driven_pull unidentified (window condition compatible); pull_after_adverse_move unidentified (window condition contradicted)
- **4b REPOST** — **TIED** (4 retained histories) · grade `scenario_dependent_link` · filled · census: rank2 20261024 successor_reprice tied: 199 of 245
  - retained histories: overlap | replace | replenish | unrelated
  - hidden action that changes the explanation: whether the successor 0x6f5fc55834 was a new submission replacing this order (replace), a top-up (replenish), an order already resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent
  - successor: reprice to 0x6f5fc55834 (first of 3 filled successor(s)); link status tied -> tied; gap [-2864, 6461] ms; d_ticks -1; identified repost: no
  - saved child links: reprice 0x6f5fc55834 (tied; gap [-2864, 6461] ms; overlap|replace|replenish|unrelated); reprice 0x92cfd50a62 (tied; gap [-63805, 6461] ms; overlap|replace|replenish|unrelated); reprice 0x9ba0185db2 (tied; gap [-64757, 6461] ms; overlap|replace|replenish|unrelated)
  - saved parent links (was this post itself a repost?): reprice 0x642a4ec8a6 (tied; gap [-11132, 5412] ms; overlap|replace|replenish|unrelated) - never identified (unrelated stays feasible)
- **5 TAKES** — **NO-EVENT** (1 retained history) · grade `observed_execution` · filled
  - no take record of this address in the episode window: every retained history agrees
- **6 VALUE** — **ONE-HISTORY** (1 retained history) · grade `recorded_fill_accounting` · filled
  - every retained history agrees: recorded_fills_under_every_retained_history
  - episode lot legs (status ok, not self-trades, settle_ms in [ws+62.538s, ws+95.085s]): 22 (maker 22, taker 0), 555.32 sh; excluded rows in the window 0; matched (mq) 555.32 sh, residual (rq) 0.00 sh
  - episode, all legs: 10s valid 22/22 rows, 555.32/555.32 sh, gross +2.30 $, net +2.30 $, +0.41 c/sh; 60s valid 22/22 rows, 555.32/555.32 sh, gross -7.16 $, net -7.16 $, -1.29 c/sh; END valid 22/22 rows, 555.32/555.32 sh, gross +77.05 $, net +77.05 $, +13.87 c/sh; common cohort 22 rows: 10s +0.41, 60s -1.29, end +13.87 c/sh; resolution (net_res) valid 22/22: +82.17 $ (+14.80 c/sh)
  - episode, maker legs: 10s valid 22/22 rows, 555.32/555.32 sh, gross +2.30 $, net +2.30 $, +0.41 c/sh; 60s valid 22/22 rows, 555.32/555.32 sh, gross -7.16 $, net -7.16 $, -1.29 c/sh; END valid 22/22 rows, 555.32/555.32 sh, gross +77.05 $, net +77.05 $, +13.87 c/sh; common cohort 22 rows: 10s +0.41, 60s -1.29, end +13.87 c/sh; resolution (net_res) valid 22/22: +82.17 $ (+14.80 c/sh)
  - episode, taker legs: 10s valid 0/0 rows, 0.00/0.00 sh, gross +0.00 $, net +0.00 $, na c/sh; 60s valid 0/0 rows, 0.00/0.00 sh, gross +0.00 $, net +0.00 $, na c/sh; END valid 0/0 rows, 0.00/0.00 sh, gross +0.00 $, net +0.00 $, na c/sh; common cohort 0 rows: 10s na, 60s na, end na c/sh; resolution (net_res) valid 0/0: +0.00 $ (na c/sh)
  - completed turnover (closes.parquet, closes.py FIFO): entries closed inside the episode 6 close(s) (+0 dust), 66.84 sh, pnl -6.01 $, net -6.01 $, hold p50 28.1 s; entries closed after it 24 close(s) (+0 dust), 274.92 sh, pnl -27.79 $, net -27.79 $, hold p50 55.3 s; closes of earlier inventory 16 close(s) (+0 dust), 146.73 sh, pnl -21.39 $, net -21.39 $, hold p50 21.3 s
  - residual of the episode legs: 0.00 sh unclosed; value to END -0.00 $ (m_end finite for 22/22 legs), to resolution -0.00 $ (22/22 resolved)
  - case order 0xa46229318a own maker rows (1 rows, 1 ok): 10s valid 1/1 rows, 50.00/50.00 sh, gross -1.25 $, net -1.25 $, -2.50 c/sh; 60s valid 1/1 rows, 50.00/50.00 sh, gross -1.25 $, net -1.25 $, -2.50 c/sh; END valid 1/1 rows, 50.00/50.00 sh, gross +27.00 $, net +27.00 $, +54.00 c/sh; common cohort 1 rows: 10s -2.50, 60s -2.50, end +54.00 c/sh; resolution (net_res) valid 1/1: +29.00 $ (+58.00 c/sh); mq 50.00 sh, rq 0.00 sh; its closes 4 close(s) (+0 dust), 50.00 sh, pnl -4.49 $, net -4.49 $, hold p50 24.9 s
  - address in this market (closes_markets): 162 closes, 1246.90 sh, close net -3.57 $; residual 99.20 sh, to END -29.40 $, to resolution -33.37 $; status ok
  - matched-gap interval accounting (matched_gap_r1): carry (primary) status ok: close net -3.57 $ on 1246.90 sh (-0.29 c/sh), carry +0.00, resid -29.40, total -33.16 $; reading both_adverse; zero (beside) status ok: close net -3.57 $ on 1246.90 sh (-0.29 c/sh), carry +0.00, resid -29.40, total -33.16 $; reading both_adverse
- **AGREEMENT** histories agree on: takes, value; TIED: pre_state, post, queue, ending, repost; pending: none; first branching link: pre_state
  - pre_state: the posting second inside k = 62..65 (ws+62..ws+66 s): features depth_touch_own, depth_2_own, fair_gap change across it
  - post: the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - queue: whether orders were ahead at the first fill (S_FIFO ahead [0.00, 509.52] sh at the first fill; [0.00, 793.04] sh at acceptance); a cancellation / replenishment masked inside a frame interval (net book change; section 5 finding 1)
  - ending: an unobserved cancel of the remainder (PL: withdrawn [0.01, inf] sh in [ws+63.084s, ws+65.085s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - repost: whether the successor 0x6f5fc55834 was a new submission replacing this order (replace), a top-up (replenish), an order already resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent

## Chain 5 - rank 3: episode 20261024_3712725_r3_e014

day 20261024, market 3712725, address 0x57fbc891e89184da071ae9d97e745531cd7f718a; case order 0x01ebf81129; cases.csv data row 33; episode window [ws+174.023s, ws+236.023s]

- **1 PRE-STATE** — **TIED** (2 retained histories) · grade `observed_state` · filled
  - retained histories: spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(60,120] | spread=1; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(60,120]
  - hidden action that changes the explanation: the posting second inside k = 186..187 (ws+186..ws+188 s): features depth_touch_own, depth_touch_opp, depth_2_own change across it
  - window seconds k = 186..187 (2 in the market window, 0 outside; 2 saved rows at side ask; time key t_ms); post window [ws+186.382s, ws+187.544s]
  - features (D0 = the state at t; columns spread_bin_D0, ...): spread {'1': 2}; depth_touch_own {'(50,200]': 1, '>200': 1}; depth_touch_opp {'(50,200]': 1, '>200': 1}; depth_2_own {'(50,200]': 1, '>200': 1}; btc_5s_aligned {'flat': 2}; fair_gap {'>3': 2}; flow_5s {'>0': 2}; time_left {'(60,120]': 2}
  - row statuses (status_D0): none
  - D1000 (the whole state lagged one second): 1 distinct state vector(s) vs 2 at D0 - a different set (timing-sensitive)
  - visible fresh creations on this side in the window: 10; at the order's cell (tick 32): 3712725|ask|32|4069 (improve) (an observed market creation; the order is at most a candidate creator, section 3)
- **2 POST** — **TIED** (4 retained histories) · grade `post_window:G1_conflict` · filled · census: rank3 20261024 G1_conflict: 33 of 471 first-filled orders
  - retained histories: posting_time_unresolved_early + EX | posting_time_unresolved_early + PL | posting_time_unresolved_late + EX | posting_time_unresolved_late + PL
  - hidden action that changes the explanation: the posting instant inside [ws+186.382s, ws+187.544s] (1162 ms; void: conflict flags); the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - order 0x01ebf81129: ask at 32c (side and tick observed in its fills); first fill <= ws+187.544s
  - window [ws+186.382s, ws+187.544s], width 1162 ms (le_10s); per-scenario lower ends S_FIFO ws+186.382s, S_FIFO_LO ws+186.382s, S_NOPRIO ws+186.382s; lifecycle post_class assumption_sensitive; packet status ahead_hi_negative+capacity_bounds_void+clock_capacity_conflict+post_after_first_fill -> grade G1_conflict (void: conflict flags)
  - size: filled 220.00 sh = a LOWER bound (observed executions); submitted size unknown; feasible EX (exhausted: = 220.00 sh, equal to the filled size only under EX); PL (pulled: [220.01, 260.00] sh); bounds [220.00, 260.00] sh
  - size status conflict; prior picks (prior-driven estimates, not evidence): P_cmd ('EX', 220.0); P_drop ('PL', 227.55); P_lot ('EX', 220.0); P_clip ('EX', 220.0)
  - same-hash taker legs (a crossing at arrival): 0 leg(s), 0.00 sh
  - saved POST logic row: queue_priority_before_demand contradicted -> unidentified (grade B_constrained -> E_unresolved)
- **3 QUEUE** — **TIED** (4 retained histories) · grade `scenario_queue_history` · filled
  - retained histories: front_at_first_fill @ level 496 (unexplained_decrease) + masked_netting_possible | behind_at_first_fill @ level 496 (unexplained_decrease) + masked_netting_possible | front_at_first_fill @ level 496 (unexplained_decrease) + as_saved | behind_at_first_fill @ level 496 (unexplained_decrease) + as_saved
  - hidden action that changes the explanation: whether orders were ahead at the first fill (S_FIFO ahead [0.00, na] sh at the first fill; [8.00, 736.27] sh at acceptance); a cancellation / replenishment masked inside a frame interval (net book change; section 5 finding 1)
  - order 0x01ebf81129: 7 fill(s), 220.00 sh (one order: its first fill and later fills of the same hash)
    - fill 1: 54.29 sh, hard window [ws+185.544s, ws+187.544s], sid 01a16f22-6538 mk_idx 0, status ok - first fill
    - fill 2: 4.95 sh, hard window [ws+185.683s, ws+187.683s], sid 01a16f22-65c3 mk_idx 0, status ok - later fill of the same hash (the same order)
    - fill 3: 10.21 sh, hard window [ws+186.168s, ws+188.168s], sid 01a16f22-67a8 mk_idx 0, status ok - later fill of the same hash (the same order)
    - fill 4: 14.11 sh, hard window [ws+186.686s, ws+188.686s], sid 01a16f22-69ae mk_idx 0, status ok - later fill of the same hash (the same order)
    - fill 5: 125.53 sh, hard window [ws+187.550s, ws+189.550s], sid 01a16f22-6d0e mk_idx 0, status ok - later fill of the same hash (the same order)
    - fill 6: 2.78 sh, hard window [ws+187.614s, ws+189.614s], sid 01a16f22-6d4e mk_idx 0, status ok - later fill of the same hash (the same order)
    - fill 7: 8.14 sh, hard window [ws+193.961s, ws+195.961s], sid 01a16f22-8619 mk_idx 2, status ok - later fill of the same hash (the same order)
  - S_FIFO queue class conflicting -> positions front_at_first_fill|behind_at_first_fill; ahead at acceptance [8.00, 736.27] sh, at the first fill [0.00, na] sh; tied first fills 0, fronts after a partial 3; lo-clock differs False; S_NOPRIO has no queue (price-time premise stated, not tested)
  - fresh-level lives compatible with the first fill: level 496 created ws+186.382s, status unexplained_decrease/unexplained_drop_before_fill, creation context improve, alt_netting_possible True, alt_first_other_address False, timing_basis book_candidate, members 17
  - candidate / fill rows (queue_front_v2) of this hash: front-candidate set at level 496 (an earliest-set member; status unexplained_decrease; candidate_addr ; candidate_group ; timing_basis book_candidate; alt_netting_possible True; alt_first_other_address False)
  - endpoint occupancy (queue_front_v2 occupancy.csv): endpoint occupancy cell_occupied_at_window_lo at ws+186.382s; size_at_window_lo 8; post_grade G1_conflict; lookup_status ok (an endpoint observation, never an actual-post queue label)
- **4a ENDING** — **TIED** (2 retained histories) · grade `cancel_compatible_ending` · filled · census: rank3 20261024 pull_possible:S_NOPRIO:cancel_possible 451
  - retained histories: EX | PL
  - hidden action that changes the explanation: an unobserved cancel of the remainder (PL: withdrawn [0.01, 40.00] sh in [ws+193.961s, ws+196.082s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - feasible end histories (union) EX|PL; per scenario S_FIFO {EX}, S_FIFO_LO {EX}, S_NOPRIO {EX,PL}
  - cancel-compatible views (pulls.csv): S_NOPRIO; S_NOPRIO: removal_class assumption_sensitive, removal set cancelled_or_expired|exhausted, window [ws+193.961s, ws+196.082s] (ok), identification tied, pull_identified False, most supported own_fill_5s (compatible)
  - saved END logic rows: inventory_driven_pull unidentified (window condition timing_sensitive); pull_after_adverse_move unidentified (window condition contradicted)
- **4b REPOST** — **TIED** (3 retained histories) · grade `unresolved_alternatives` · filled · census: rank3 20261024 successor_none none: 70 of 70
  - retained histories: pull_without_repost | unfilled_repost | no_pull
  - hidden action that changes the explanation: whether the order was pulled without a repost, reposted with an order that never filled, or exhausted (invisible in fills)
  - successor: none (no later first-filled same-side order: an unfilled successor or none (invisible in fills))
  - saved child links: none
  - saved parent links (was this post itself a repost?): refill 0x00d6f0baab (sequential_identified; gap [7195, 12331] ms; replace|replenish|unrelated); reprice 0x21c215c166 (sequential_identified; gap [496, 4192] ms; replace|replenish|unrelated) - never identified (unrelated stays feasible)
- **5 TAKES** — **ONE-HISTORY** (1 retained history) · grade `observed_execution` · filled
  - every retained history agrees: D0/D1000/pend: opposing,opposing,opposing,opposing,opposing
  - take 01a16f22-ad5a: YES buy 14.29 sh @ 0.340, settle ws+206.010s, anchor ws+204.000s (assigned); q_D0 -397.12 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim reduce_partial); O4 reduces_all; status simultaneous_action
  - take 01a16f22-ad5a: YES buy 6.59 sh @ 0.340, settle ws+206.010s, anchor ws+204.000s (assigned); q_D0 -397.12 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim reduce_partial); O4 reduces_all; status simultaneous_action
  - take 01a16f22-ad67: YES buy 12.21 sh @ 0.340, settle ws+206.023s, anchor ws+204.000s (assigned); q_D0 -397.12 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim ); O4 reduces_all; status ok [the selected take case]
  - take 01a16f22-ad9e: YES buy 19.00 sh @ 0.340, settle ws+206.078s, anchor ws+204.000s (assigned); q_D0 -397.12 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim reduce_partial); O4 reduces_all; status simultaneous_action
  - take 01a16f22-ad9e: YES buy 5.91 sh @ 0.340, settle ws+206.078s, anchor ws+204.000s (assigned); q_D0 -397.12 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim reduce_partial); O4 reduces_all; status simultaneous_action
  - saved TAKE logic rows (selected case): after_fill_burst compatible; directional insufficient_comparison; inventory_unwind compatible
- **6 VALUE** — **ONE-HISTORY** (1 retained history) · grade `recorded_fill_accounting` · filled
  - every retained history agrees: recorded_fills_under_every_retained_history
  - episode lot legs (status ok, not self-trades, settle_ms in [ws+174.023s, ws+236.023s]): 72 (maker 67, taker 5), 2081.50 sh; excluded rows in the window 0; matched (mq) 1974.96 sh, residual (rq) 106.54 sh
  - episode, all legs: 10s valid 72/72 rows, 2081.50/2081.50 sh, gross -9.68 $, net -10.07 $, -0.46 c/sh; 60s valid 72/72 rows, 2081.50/2081.50 sh, gross +34.90 $, net +34.51 $, +1.68 c/sh; END valid 72/72 rows, 2081.50/2081.50 sh, gross -263.56 $, net -263.95 $, -12.66 c/sh; common cohort 72 rows: 10s -0.46, 60s +1.68, end -12.66 c/sh; resolution (net_res) valid 72/72: -281.96 $ (-13.55 c/sh)
  - episode, maker legs: 10s valid 67/67 rows, 2023.50/2023.50 sh, gross -7.65 $, net -7.65 $, -0.38 c/sh; 60s valid 67/67 rows, 2023.50/2023.50 sh, gross +7.85 $, net +7.85 $, +0.39 c/sh; END valid 67/67 rows, 2023.50/2023.50 sh, gross -299.52 $, net -299.52 $, -14.80 c/sh; common cohort 67 rows: 10s -0.38, 60s +0.39, end -14.80 c/sh; resolution (net_res) valid 67/67: -319.08 $ (-15.77 c/sh)
  - episode, taker legs: 10s valid 5/5 rows, 58.00/58.00 sh, gross -2.03 $, net -2.42 $, -3.50 c/sh; 60s valid 5/5 rows, 58.00/58.00 sh, gross +27.05 $, net +26.66 $, +46.64 c/sh; END valid 5/5 rows, 58.00/58.00 sh, gross +35.96 $, net +35.57 $, +62.00 c/sh; common cohort 5 rows: 10s -3.50, 60s +46.64, end +62.00 c/sh; resolution (net_res) valid 5/5: +37.12 $ (+64.00 c/sh)
  - completed turnover (closes.parquet, closes.py FIFO): entries closed inside the episode 54 close(s) (+0 dust), 825.22 sh, pnl -0.80 $, net -1.20 $, hold p50 6.8 s; entries closed after it 9 close(s) (+0 dust), 216.94 sh, pnl -27.59 $, net -27.59 $, hold p50 48.1 s; closes of earlier inventory 6 close(s) (+0 dust), 107.59 sh, pnl +2.88 $, net +2.88 $, hold p50 19.4 s
  - residual of the episode legs: 106.54 sh unclosed; value to END -65.06 $ (m_end finite for 72/72 legs), to resolution -69.33 $ (72/72 resolved)
  - case order 0x01ebf81129 own maker rows (7 rows, 7 ok): 10s valid 7/7 rows, 220.00/220.00 sh, gross -17.48 $, net -17.48 $, -7.95 c/sh; 60s valid 7/7 rows, 220.00/220.00 sh, gross -40.32 $, net -40.32 $, -18.33 c/sh; END valid 7/7 rows, 220.00/220.00 sh, gross -140.80 $, net -140.80 $, -64.00 c/sh; common cohort 7 rows: 10s -7.95, 60s -18.33, end -64.00 c/sh; resolution (net_res) valid 7/7: -149.60 $ (-68.00 c/sh); mq 220.00 sh, rq 0.00 sh; its closes 9 close(s) (+0 dust), 220.00 sh, pnl +6.63 $, net +6.63 $, hold p50 2.6 s
  - address in this market (closes_markets): 252 closes, 3636.14 sh, close net -73.47 $; residual 106.54 sh, to END -65.06 $, to resolution -69.33 $; status ok
  - matched-gap interval accounting (matched_gap_r1): carry (primary) status ok: close net -73.47 $ on 3636.14 sh (-2.02 c/sh), carry +0.00, resid -65.06, total -138.53 $; reading exposure_gap; zero (beside) status ok: close net -73.47 $ on 3636.14 sh (-2.02 c/sh), carry +0.00, resid -65.06, total -138.53 $; reading exposure_gap
- **AGREEMENT** histories agree on: takes, value; TIED: pre_state, post, queue, ending, repost; pending: none; first branching link: pre_state
  - pre_state: the posting second inside k = 186..187 (ws+186..ws+188 s): features depth_touch_own, depth_touch_opp, depth_2_own change across it
  - post: the posting instant inside [ws+186.382s, ws+187.544s] (1162 ms; void: conflict flags); the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - queue: whether orders were ahead at the first fill (S_FIFO ahead [0.00, na] sh at the first fill; [8.00, 736.27] sh at acceptance); a cancellation / replenishment masked inside a frame interval (net book change; section 5 finding 1)
  - ending: an unobserved cancel of the remainder (PL: withdrawn [0.01, 40.00] sh in [ws+193.961s, ws+196.082s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - repost: whether the order was pulled without a repost, reposted with an order that never filled, or exhausted (invisible in fills)

## Chain 6 - rank 3: episode 20261024_3716727_r3_e017

day 20261024, market 3716727, address 0x57fbc891e89184da071ae9d97e745531cd7f718a; case order 0x66e4f311be; cases.csv data row 34; episode window [ws+0.896s, ws+161.901s]

- **1 PRE-STATE** — **TIED** (17 retained histories) · grade `observed_state` · filled
  - retained histories: spread=1; depth_touch_own=(0,50]; depth_touch_opp=(0,50]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(0,50]; depth_touch_opp=(50,200]; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=(0,50]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=(0,50]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=<=0; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=>200; depth_touch_opp=(0,50]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300] | spread=2; depth_touch_own=(0,50]; depth_touch_opp=(0,50]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=<=0; flow_5s=>0; time_left=(240,300] | spread=2; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=<=0; flow_5s=>0; time_left=(240,300] | spread=2; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]
  - hidden action that changes the explanation: the posting second inside k = 0..56 (ws+0..ws+57 s): features spread, depth_touch_own, depth_touch_opp, depth_2_own, fair_gap change across it
  - window seconds k = 0..56 (57 in the market window, 0 outside; 57 saved rows at side ask; time key t_ms); post window [ws+0.896s, ws+56.719s]
  - features (D0 = the state at t; columns spread_bin_D0, ...): spread {'1': 54, '2': 3}; depth_touch_own {'(0,50]': 6, '(50,200]': 20, '>200': 31}; depth_touch_opp {'(0,50]': 9, '(50,200]': 18, '>200': 30}; depth_2_own {'(0,50]': 5, '(50,200]': 8, '>200': 44}; btc_5s_aligned {'flat': 57}; fair_gap {'(0,1]': 6, '(1,3]': 48, '<=0': 3}; flow_5s {'>0': 57}; time_left {'(240,300]': 57}
  - row statuses (status_D0): none
  - D1000 (the whole state lagged one second): 18 distinct state vector(s) vs 17 at D0 - a different set (timing-sensitive)
  - visible fresh creations on this side in the window: 31; at the order's cell (tick 61): 3716727|ask|61|455 (deep) (an observed market creation; the order is at most a candidate creator, section 3)
- **2 POST** — **TIED** (4 retained histories) · grade `post_window:G2_bound_constrained` · filled · census: rank3 20261024 G2_bound_constrained: 209 of 471 first-filled orders
  - retained histories: early_post_rest_gt_10s + EX | early_post_rest_gt_10s + PL | late_post_rest_le_10s + EX | late_post_rest_le_10s + PL
  - hidden action that changes the explanation: the posting instant inside [ws+0.896s, ws+56.719s] (55823 ms; lifecycle scenario-free bound (conditional on H1 / H2 / capacity and the maximum match lag)); the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - order 0x66e4f311be: ask at 61c (side and tick observed in its fills); first fill <= ws+56.719s
  - window [ws+0.896s, ws+56.719s], width 55823 ms (gt_10s); per-scenario lower ends S_FIFO ws+0.896s, S_FIFO_LO ws+0.896s, S_NOPRIO ws+0.896s; lifecycle post_class constrained; packet status ok -> grade G2_bound_constrained (lifecycle scenario-free bound (conditional on H1 / H2 / capacity and the maximum match lag))
  - size: filled 10.00 sh = a LOWER bound (observed executions); submitted size unknown; feasible EX (exhausted: = 10.00 sh, equal to the filled size only under EX); PL (pulled: [10.01, inf] sh); bounds [10.00, inf] sh
  - size status distinguished_under_priority; prior picks (prior-driven estimates, not evidence): P_cmd ('EX', 10.0); P_drop ('PL', 58.17); P_lot ('EX', 10.0); P_clip ('EX', 10.0)
  - same-hash taker legs (a crossing at arrival): 0 leg(s), 0.00 sh
  - saved POST logic row: queue_priority_before_demand timing_sensitive -> timing_sensitive (grade B_constrained -> B_constrained)
- **3 QUEUE** — **TIED** (4 retained histories) · grade `scenario_queue_history` · filled
  - retained histories: front_at_first_fill @ level 28 (ambiguous_order) + masked_netting_possible | behind_at_first_fill @ level 28 (ambiguous_order) + masked_netting_possible | front_at_first_fill @ level 28 (ambiguous_order) + as_saved | behind_at_first_fill @ level 28 (ambiguous_order) + as_saved
  - hidden action that changes the explanation: whether orders were ahead at the first fill (S_FIFO ahead [0.00, 272.32] sh at the first fill; [0.00, 318.65] sh at acceptance); a cancellation / replenishment masked inside a frame interval (net book change; section 5 finding 1)
  - order 0x66e4f311be: 1 fill(s), 10.00 sh
    - fill 1: 10.00 sh, hard window [ws+54.719s, ws+56.719s], sid 01a16f57-54af mk_idx 3, status ok - first fill
  - S_FIFO queue class unidentified -> positions front_at_first_fill|behind_at_first_fill; ahead at acceptance [0.00, 318.65] sh, at the first fill [0.00, 272.32] sh; tied first fills 0, fronts after a partial 0; lo-clock differs False; S_NOPRIO has no queue (price-time premise stated, not tested)
  - fresh-level lives compatible with the first fill: level 28 created ws+0.981s, status ambiguous_order/untimed_record_could_be_first, creation context deep, alt_netting_possible True, alt_first_other_address False, timing_basis book_candidate, members 23
  - candidate / fill rows (queue_front_v2) of this hash: none
  - endpoint occupancy (queue_front_v2 occupancy.csv): endpoint occupancy cell_empty_at_window_lo at ws+0.896s; size_at_window_lo 0; post_grade G2_bound_constrained; lookup_status ok (an endpoint observation, never an actual-post queue label)
- **4a ENDING** — **TIED** (2 retained histories) · grade `cancel_compatible_ending` · filled · census: rank3 20261024 pull_possible:S_NOPRIO:cancel_possible 451
  - retained histories: EX | PL
  - hidden action that changes the explanation: an unobserved cancel of the remainder (PL: withdrawn [0.01, inf] sh in [ws+54.719s, ws+78.425s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - feasible end histories (union) EX|PL; per scenario S_FIFO {EX}, S_FIFO_LO {EX}, S_NOPRIO {EX,PL}
  - cancel-compatible views (pulls.csv): S_NOPRIO; S_NOPRIO: removal_class assumption_sensitive, removal set cancelled_or_expired|exhausted, window [ws+54.719s, ws+78.425s] (ok), identification tied, pull_identified False, most supported btc_against_1s+book_toward_5s+book_away_5s+mid_against_5s+behind_touch+own_fill_5s+inv_built_5s (timing_sensitive)
  - saved END logic rows: inventory_driven_pull unidentified (window condition timing_sensitive); pull_after_adverse_move unidentified (window condition contradicted)
- **4b REPOST** — **TIED** (4 retained histories) · grade `scenario_dependent_link` · filled · census: rank3 20261024 successor_reprice tied: 236 of 324
  - retained histories: overlap | replace | replenish | unrelated
  - hidden action that changes the explanation: whether the successor 0xc5d1f7e499 was a new submission replacing this order (replace), a top-up (replenish), an order already resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent
  - successor: reprice to 0xc5d1f7e499 (first of 2 filled successor(s)); link status tied -> tied; gap [-17292, 13731] ms; d_ticks -9; identified repost: no
  - saved child links: reprice 0xc5d1f7e499 (tied; gap [-17292, 13731] ms; overlap|replace|replenish|unrelated); refill 0xca5291f95f (sequential_identified; gap [40808, 64801] ms; replace|replenish|unrelated)
  - saved parent links (was this post itself a repost?): reprice 0xfdec9cd6ef (tied; gap [-47035, 25246] ms; overlap|replace|replenish|unrelated) - never identified (unrelated stays feasible)
- **5 TAKES** — **ONE-HISTORY** (1 retained history) · grade `observed_execution` · filled
  - every retained history agrees: D0/D1000/pend: opposing,opposing,opposing
  - take 01a16f57-b05c: YES buy 9.21 sh @ 0.660, settle ws+80.188s, anchor ws+78.000s (assigned); q_D0 -41.82 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim ); O4 reduces_all; status ok
  - take 01a16f58-7a5d: YES sell 4.25 sh @ 0.540, settle ws+131.901s, anchor ws+129.000s (assigned); q_D0 +27.11 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim reduce_partial); O4 reduces_all; status simultaneous_action [the selected take case]
  - take 01a16f58-7a5d: YES sell 20.78 sh @ 0.540, settle ws+131.901s, anchor ws+129.000s (assigned); q_D0 +27.11 -> D0 opposing, D1000 opposing, pend opposing; class reduce_partial (rec reduce_partial / cert reduce_partial / sim reduce_partial); O4 reduces_all; status simultaneous_action
  - saved TAKE logic rows (selected case): after_fill_burst compatible; directional insufficient_comparison; inventory_unwind compatible
- **6 VALUE** — **ONE-HISTORY** (1 retained history) · grade `recorded_fill_accounting` · filled
  - every retained history agrees: recorded_fills_under_every_retained_history
  - episode lot legs (status ok, not self-trades, settle_ms in [ws+0.896s, ws+161.901s]): 67 (maker 64, taker 3), 470.58 sh; excluded rows in the window 0; matched (mq) 468.43 sh, residual (rq) 2.15 sh
  - episode, all legs: 10s valid 67/67 rows, 470.58/470.58 sh, gross +2.26 $, net +1.96 $, +0.48 c/sh; 60s valid 67/67 rows, 470.58/470.58 sh, gross -17.92 $, net -18.21 $, -3.81 c/sh; END valid 67/67 rows, 470.58/470.58 sh, gross -8.05 $, net -8.34 $, -1.71 c/sh; common cohort 67 rows: 10s +0.48, 60s -3.81, end -1.71 c/sh; resolution (net_res) valid 67/67: -8.08 $ (-1.72 c/sh)
  - episode, maker legs: 10s valid 64/64 rows, 436.34/436.34 sh, gross -1.88 $, net -1.88 $, -0.43 c/sh; 60s valid 64/64 rows, 436.34/436.34 sh, gross -18.86 $, net -18.86 $, -4.32 c/sh; END valid 64/64 rows, 436.34/436.34 sh, gross -0.14 $, net -0.14 $, -0.03 c/sh; common cohort 64 rows: 10s -0.43, 60s -4.32, end -0.03 c/sh; resolution (net_res) valid 64/64: +0.40 $ (+0.09 c/sh)
  - episode, taker legs: 10s valid 3/3 rows, 34.24/34.24 sh, gross +4.14 $, net +3.84 $, +12.08 c/sh; 60s valid 3/3 rows, 34.24/34.24 sh, gross +0.94 $, net +0.65 $, +2.74 c/sh; END valid 3/3 rows, 34.24/34.24 sh, gross -7.91 $, net -8.20 $, -23.10 c/sh; common cohort 3 rows: 10s +12.08, 60s +2.74, end -23.10 c/sh; resolution (net_res) valid 3/3: -8.48 $ (-24.76 c/sh)
  - completed turnover (closes.parquet, closes.py FIFO): entries closed inside the episode 63 close(s) (+0 dust), 234.22 sh, pnl -9.34 $, net -9.64 $, hold p50 8.2 s; entries closed after it 0 close(s) (+0 dust), 0.00 sh, pnl +0.00 $, net +0.00 $, hold p50 na s; closes of earlier inventory 0 close(s) (+0 dust), 0.00 sh, pnl +0.00 $, net +0.00 $, hold p50 na s
  - residual of the episode legs: 2.15 sh unclosed; value to END +1.29 $ (m_end finite for 67/67 legs), to resolution +1.36 $ (67/67 resolved)
  - case order 0x66e4f311be own maker rows (1 rows, 1 ok): 10s valid 1/1 rows, 10.00/10.00 sh, gross +1.05 $, net +1.05 $, +10.50 c/sh; 60s valid 1/1 rows, 10.00/10.00 sh, gross -0.65 $, net -0.65 $, -6.50 c/sh; END valid 1/1 rows, 10.00/10.00 sh, gross -3.60 $, net -3.60 $, -36.00 c/sh; common cohort 1 rows: 10s +10.50, 60s -6.50, end -36.00 c/sh; resolution (net_res) valid 1/1: -3.90 $ (-39.00 c/sh); mq 10.00 sh, rq 0.00 sh; its closes 2 close(s) (+0 dust), 10.00 sh, pnl -0.66 $, net -0.71 $, hold p50 24.1 s
  - address in this market (closes_markets): 63 closes, 234.22 sh, close net -9.64 $; residual 2.15 sh, to END +1.29 $, to resolution +1.36 $; status ok
  - matched-gap interval accounting (matched_gap_r1): carry (primary) status ok: close net -9.64 $ on 234.22 sh (-4.11 c/sh), carry +0.00, resid +1.29, total -8.34 $; reading exposure_gap; zero (beside) status ok: close net -9.64 $ on 234.22 sh (-4.11 c/sh), carry +0.00, resid +1.29, total -8.34 $; reading exposure_gap
- **AGREEMENT** histories agree on: takes, value; TIED: pre_state, post, queue, ending, repost; pending: none; first branching link: pre_state
  - pre_state: the posting second inside k = 0..56 (ws+0..ws+57 s): features spread, depth_touch_own, depth_touch_opp, depth_2_own, fair_gap change across it
  - post: the posting instant inside [ws+0.896s, ws+56.719s] (55823 ms; lifecycle scenario-free bound (conditional on H1 / H2 / capacity and the maximum match lag)); the submitted size: exhausted at the filled size (EX) vs a cancelled remainder (PL)
  - queue: whether orders were ahead at the first fill (S_FIFO ahead [0.00, 272.32] sh at the first fill; [0.00, 318.65] sh at acceptance); a cancellation / replenishment masked inside a frame interval (net book change; section 5 finding 1)
  - ending: an unobserved cancel of the remainder (PL: withdrawn [0.01, inf] sh in [ws+54.719s, ws+78.425s]) vs exhaustion by the last fill (EX): if exhausted there was no pull decision; if pulled, the END context is a candidate trigger only
  - repost: whether the successor 0xc5d1f7e499 was a new submission replacing this order (replace), a top-up (replenish), an order already resting (overlap) or unrelated: a sequential timing never identifies a repost or its intent

