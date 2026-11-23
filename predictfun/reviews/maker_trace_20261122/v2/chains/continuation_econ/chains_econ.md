# Candidate continuation economics - the six chains (competing_transition_contract.md section 5) - RUN

As of 2026-11-23T22:21:32Z. Saved evidence only (no tape, book, lifecycle, queue or FIFO recomputed); consumed days; descriptive; filled orders are outcome-selected; no identified intent or private action; no profitability claim. Exact legs = closes.parquet FIFO rows keyed by full leg keys, clean cohort (status ok, non-self). Entry-close windows by hold time on the settlement clock (le10s / le60s cumulative, leEND = close settle <= END, afterEND = later); closing-earlier closes and the residual (to END / resolution) kept apart; END = we (market window end). Takes counted from the latest possible post (post_hi) with the END upper bound; a take is not an exit unless its taker leg FIFO-closes the order's leg. Response context requires the creation tick == resp_tick (else contextual only). Account-level P&L is never attached.

Pooled over the 6 distinct chain orders (exact available 6): entry closes le10s 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; le60s 15 (+0 dust), 148.14 sh, pnl -6.23 $, fees 0.163 $, net -6.39 $; leEND 15 (+0 dust), 148.14 sh, pnl -6.23 $, fees 0.163 $, net -6.39 $; afterEND 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; closing earlier 8 (+0 dust), 211.86 sh, pnl +6.63 $, fees 0.000 $, net +6.63 $; residual 6.67 sh, to END +0.47 $, to resolution +0.87 $ (fee 0.000 $).

## Chain 1 - rank 1: episode 20261024_3716727_r1_e018

order 0x88c0095c1e (ask 69c), day 20261024, market 3716727, address 0x6da6cb464f92ae7ad4ec3d239c81719cb1d0ae03; post grade G4_scenario_only; post window [ws+2.325s, ws+80.506s]; END ws+300.000s; other links of this order: none

| item | link | reading | histories | grade | summary |
|---|---|---|---|---|---|
| 1 | pre_state | TIED | 26 | observed_state:post_window:G4_scenario_only | G4_scenario_only: 79 saved seconds; features constant: flow_5s; + unresolved_outside_window |
| 1b | transition | TIED | 3 | transition_context | 3 retained context(s): joined_shown_cell, masked_recreation_in_window, unresolved_outside_window |
| 2 | post | TIED | 4 | post_window:G4_scenario_only | ask 69c, G4_scenario_only gt_10s, size >= 30.00 sh (EX/PL) |
| 3 | queue | TIED | 4 | scenario_queue_history | class unidentified; 1 compatible level(s); masked netting not_evaluated |
| 4a | ending | TIED | 2 | cancel_compatible_ending | EX/PL; cancel-compatible S_FIFO, S_NOPRIO |
| 4b | repost | TIED | 3 | scenario_dependent_link | reprice sequential_under_scenarios to 0x89e05c2583 |
| 5 | takes | ONE-HISTORY | 1 | observed_execution | 12 take record(s); D0 {'opposing': 3, 'q0': 2, 'same': 7}; clock-disagreeing 0; undetermined {'q0': 2} |
| 6 | value | ONE-HISTORY | 1 | recorded_fill_accounting | END net -42.93 $ (47/47 valid); 10s/60s/END same sign False |

- **Unresolved (TIED) links** (6): pre_state: spread=1; depth_touch_own=(0,50]; depth_touch_opp=(0,50]; depth_2_own=0; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=(0,50]; depth_touch_opp=(0,50]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(0,50]; depth_touch_opp=(50,200]; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=(0,50]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=<=0; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=against; fair_gap=<=0; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=>200; depth_touch_opp=(0,50]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=>200; depth_touch_opp=(0,50]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=2; depth_touch_own=(0,50]; depth_touch_opp=(0,50]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=<=0; flow_5s=>0; time_left=(240,300]|spread=2; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(120,240]|spread=2; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|unresolved_outside_window; transition: joined_shown_cell|masked_recreation_in_window|unresolved_outside_window; post: late_post_in_scenario_window + EX|late_post_in_scenario_window + PL|earlier_post_masked_by_replenishment + EX|earlier_post_masked_by_replenishment + PL; queue: front_at_first_fill @ level 51 (unknown) + masked_netting_not_evaluated|behind_at_first_fill @ level 51 (unknown) + masked_netting_not_evaluated|front_at_first_fill @ level 51 (unknown) + as_saved|behind_at_first_fill @ level 51 (unknown) + as_saved; ending: EX|PL; repost: replace|replenish|unrelated
- **Response context**: none attached by chains_v4 (1 creation line(s), 0 response episode(s) in its census) -> no_response_context
- **Exact-leg continuation** (available; legs 1 clean of 1, excluded status 0 / self 0; 30.00 sh, mq 30.00, rq 0.00): entry closes le10s 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; le60s 4 (+0 dust), 30.00 sh, pnl +2.38 $, fees 0.113 $, net +2.27 $; leEND 4 (+0 dust), 30.00 sh, pnl +2.38 $, fees 0.113 $, net +2.27 $; afterEND 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; leEND by closing leg: taker 14.83 sh net +0.93 $, maker 15.17 sh net +1.34 $; closing earlier inventory (kept apart) 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $
- **Residual** (kept apart): 0.00 sh unclosed; to END +0.00 $ (1/1 legs with m_end), to resolution +0.00 $ (1/1 resolved), prorated fee 0.000 $; fees of the clean legs 0.000 $
- **Views** (clean cohort, gross markouts beside): 10s gross -0.75 $ (1 valid), 60s +8.85 $ (1), END -8.40 $ (1); resolution net -9.30 $ (1 valid)
- **Takes of the address after the latest possible post** (anchor ws+80.506s): statuses before_anchor:6;inside_END:135; 10s 3 take(s), |dq| 44.08 sh, dir_D0 opposing:1;same:2, O4 reduces_all:1;reduces_none:2, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh); 60s 25 take(s), |dq| 262.39 sh, dir_D0 opposing:15;same:10, O4 reduces_all:4;reduces_none:12;reduces_some:9, clock-disagreeing 9, FIFO-closing this order 2 (14.83 sh); END 135 take(s), |dq| 1432.62 sh, dir_D0 opposing:82;same:53, O4 reduces_all:58;reduces_none:47;reduces_some:30, clock-disagreeing 27, FIFO-closing this order 2 (14.83 sh) - the other takes are not exits of this order

## Chain 2 - rank 1: episode 20261024_3702960_r1_e006

order 0x02fb943dae (ask 13c), day 20261024, market 3702960, address 0x6da6cb464f92ae7ad4ec3d239c81719cb1d0ae03; post grade G4_scenario_only; post window [ws+211.237s, ws+213.303s]; END ws+300.000s; other links of this order: none

| item | link | reading | histories | grade | summary |
|---|---|---|---|---|---|
| 1 | pre_state | TIED | 4 | observed_state:post_window:G4_scenario_only | G4_scenario_only: 3 saved seconds; features constant: btc_5s_aligned, fair_gap, flow_5s, time_left; + unresolved_outside_window |
| 1b | transition | TIED | 4 | transition_context | 4 retained context(s): at_creation:level_273:two_side/both_down, joined_shown_cell, masked_recreation_in_window, unresolved_outside_window |
| 2 | post | TIED | 4 | post_window:G4_scenario_only | ask 13c, G4_scenario_only le_10s, size >= 6.67 sh (EX/PL) |
| 3 | queue | TIED | 8 | scenario_queue_history | class unidentified; 2 compatible level(s); masked netting not_evaluated |
| 4a | ending | TIED | 2 | cancel_compatible_ending | EX/PL; cancel-compatible S_FIFO, S_NOPRIO |
| 4b | repost | TIED | 4 | scenario_dependent_link | reprice tied to 0xdef88dc345 |
| 5 | takes | ONE-HISTORY | 1 | observed_execution | 8 take record(s); D0 {'opposing': 3, 'same': 5}; clock-disagreeing 0; undetermined {} |
| 6 | value | ONE-HISTORY | 1 | recorded_fill_accounting | END net +5.87 $ (21/21 valid); 10s/60s/END same sign True |

- **Unresolved (TIED) links** (6): pre_state: spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(60,120]|spread=2; depth_touch_own=(50,200]; depth_touch_opp=(50,200]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(60,120]|spread=2; depth_touch_own=>200; depth_touch_opp=(0,50]; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(60,120]|unresolved_outside_window; transition: at_creation:level_273:two_side/both_down|joined_shown_cell|masked_recreation_in_window|unresolved_outside_window; post: late_post_in_scenario_window + EX|late_post_in_scenario_window + PL|earlier_post_masked_by_replenishment + EX|earlier_post_masked_by_replenishment + PL; queue: front_at_first_fill @ level 236 (unknown) + masked_netting_not_evaluated|front_at_first_fill @ level 273 (unknown) + masked_netting_not_evaluated|behind_at_first_fill @ level 236 (unknown) + masked_netting_not_evaluated|behind_at_first_fill @ level 273 (unknown) + masked_netting_not_evaluated|front_at_first_fill @ level 236 (unknown) + as_saved|front_at_first_fill @ level 273 (unknown) + as_saved|behind_at_first_fill @ level 236 (unknown) + as_saved|behind_at_first_fill @ level 273 (unknown) + as_saved; ending: EX|PL; repost: overlap|replace|replenish|unrelated
- **Response context**: none attached by chains_v4 (2 creation line(s), 0 response episode(s) in its census) -> no_response_context
- **Exact-leg continuation** (available; legs 1 clean of 1, excluded status 0 / self 0; 6.67 sh, mq 0.00, rq 6.67): entry closes le10s 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; le60s 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; leEND 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; afterEND 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; leEND by closing leg: taker 0.00 sh net +0.00 $, maker 0.00 sh net +0.00 $; closing earlier inventory (kept apart) 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $
- **Residual** (kept apart): 6.67 sh unclosed; to END +0.47 $ (1/1 legs with m_end), to resolution +0.87 $ (1/1 resolved), prorated fee 0.000 $; fees of the clean legs 0.000 $
- **Views** (clean cohort, gross markouts beside): 10s gross +0.30 $ (1 valid), 60s +0.23 $ (1), END +0.47 $ (1); resolution net +0.87 $ (1 valid)
- **Takes of the address after the latest possible post** (anchor ws+213.303s): statuses before_anchor:71;inside_END:3; 10s 0 take(s), |dq| 0.00 sh, dir_D0 none, O4 none, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh); 60s 3 take(s), |dq| 51.47 sh, dir_D0 opposing:3, O4 reduces_all:3, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh); END 3 take(s), |dq| 51.47 sh, dir_D0 opposing:3, O4 reduces_all:3, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh) - the other takes are not exits of this order

## Chain 3 - rank 2: episode 20261024_3702960_r2_e007

order 0x084be2d8a4 (bid 49c), day 20261024, market 3702960, address 0x7d32c568a9bea1714036972601c51f4902c67091; post grade G1_conflict; post window [ws+4.027s, ws+4.785s]; END ws+300.000s; other links of this order: none

| item | link | reading | histories | grade | summary |
|---|---|---|---|---|---|
| 1 | pre_state | TIED | 2 | observed_state:post_window:G1_conflict | G1_conflict: 1 saved seconds; features constant: spread, depth_touch_own, depth_touch_opp, depth_2_own, btc_5s_aligned, fair_gap, flow_5s, t |
| 1b | transition | TIED | 3 | transition_context | 3 retained context(s): joined_shown_cell, masked_recreation_in_window, unresolved_outside_window |
| 2 | post | TIED | 4 | post_window:G1_conflict | bid 49c, G1_conflict le_1s, size >= 50.00 sh (EX/PL) |
| 3 | queue | TIED | 4 | scenario_queue_history | class conflicting; 1 compatible level(s); masked netting possible |
| 4a | ending | TIED | 2 | scenario_dependent_ending | EX/PL; cancel-compatible none |
| 4b | repost | TIED | 3 | scenario_dependent_link | reprice sequential_under_scenarios to 0xd0f156e83a |
| 5 | takes | ONE-HISTORY | 1 | observed_execution | 5 take record(s); D0 {'opposing': 1, 'q0': 1, 'same': 3}; clock-disagreeing 0; undetermined {'q0': 1} |
| 6 | value | ONE-HISTORY | 1 | recorded_fill_accounting | END net -46.33 $ (33/33 valid); 10s/60s/END same sign False |

- **Unresolved (TIED) links** (6): pre_state: spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300]|unresolved_outside_window; transition: joined_shown_cell|masked_recreation_in_window|unresolved_outside_window; post: posting_time_unresolved_early + EX|posting_time_unresolved_early + PL|posting_time_unresolved_late + EX|posting_time_unresolved_late + PL; queue: front_at_first_fill @ level 71 (ambiguous_order) + masked_netting_possible|behind_at_first_fill @ level 71 (ambiguous_order) + masked_netting_possible|front_at_first_fill @ level 71 (ambiguous_order) + as_saved|behind_at_first_fill @ level 71 (ambiguous_order) + as_saved; ending: EX|PL; repost: replace|replenish|unrelated
- **Response context** level 71 (before the post window), episode 3 (restoring): resp_tick 49 vs cell 49 -> **contextual_only** (outside_post_window)
- **Exact-leg continuation** (available; legs 3 clean of 3, excluded status 0 / self 0; 50.00 sh, mq 50.00, rq 0.00): entry closes le10s 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; le60s 4 (+0 dust), 50.00 sh, pnl -3.46 $, fees 0.000 $, net -3.46 $; leEND 4 (+0 dust), 50.00 sh, pnl -3.46 $, fees 0.000 $, net -3.46 $; afterEND 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; leEND by closing leg: taker 0.00 sh net +0.00 $, maker 50.00 sh net -3.46 $; closing earlier inventory (kept apart) 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $
- **Residual** (kept apart): 0.00 sh unclosed; to END +0.00 $ (3/3 legs with m_end), to resolution +0.00 $ (3/3 resolved), prorated fee 0.000 $; fees of the clean legs 0.000 $
- **Views** (clean cohort, gross markouts beside): 10s gross -0.75 $ (3 valid), 60s -14.25 $ (3), END -21.50 $ (3); resolution net -24.50 $ (3 valid)
- **Takes of the address after the latest possible post** (anchor ws+4.785s): statuses before_anchor:1;inside_END:48; 10s 3 take(s), |dq| 35.00 sh, dir_D0 same:3, O4 reduces_none:3, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh); 60s 9 take(s), |dq| 123.00 sh, dir_D0 opposing:1;same:8, O4 reduces_all:1;reduces_none:8, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh); END 48 take(s), |dq| 515.67 sh, dir_D0 opposing:24;same:24, O4 reduces_all:24;reduces_none:24, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh) - the other takes are not exits of this order

## Chain 4 - rank 2: episode 20261024_3712725_r2_e015

order 0xa46229318a (bid 42c), day 20261024, market 3712725, address 0x7d32c568a9bea1714036972601c51f4902c67091; post grade G2_bound_constrained; post window [ws+62.538s, ws+65.084s]; END ws+300.000s; other links of this order: none

| item | link | reading | histories | grade | summary |
|---|---|---|---|---|---|
| 1 | pre_state | TIED | 4 | observed_state:post_window:G2_bound_constrained | G2_bound_constrained: 4 saved seconds; features constant: spread, depth_touch_opp, btc_5s_aligned, flow_5s, time_left |
| 1b | transition | TIED | 3 | transition_context | 3 retained context(s): at_creation:level_221:two_side/both_up, joined_shown_cell, masked_recreation_in_window |
| 2 | post | TIED | 2 | post_window:G2_bound_constrained | bid 42c, G2_bound_constrained le_10s, size >= 50.00 sh (EX/PL) |
| 3 | queue | TIED | 4 | scenario_queue_history | class unidentified; 1 compatible level(s); masked netting possible |
| 4a | ending | TIED | 2 | cancel_compatible_ending | EX/PL; cancel-compatible S_NOPRIO |
| 4b | repost | TIED | 4 | scenario_dependent_link | reprice tied to 0x6f5fc55834 |
| 5 | takes | NO-EVENT | 1 | observed_execution | no take record in the episode window |
| 6 | value | ONE-HISTORY | 1 | recorded_fill_accounting | END net +77.05 $ (22/22 valid); 10s/60s/END same sign False |

- **Unresolved (TIED) links** (6): pre_state: spread=1; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(120,240]|spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(120,240]; transition: at_creation:level_221:two_side/both_up|joined_shown_cell|masked_recreation_in_window; post: posted_le_10s_before_first_fill + EX|posted_le_10s_before_first_fill + PL; queue: front_at_first_fill @ level 221 (ambiguous_order) + masked_netting_possible|behind_at_first_fill @ level 221 (ambiguous_order) + masked_netting_possible|front_at_first_fill @ level 221 (ambiguous_order) + as_saved|behind_at_first_fill @ level 221 (ambiguous_order) + as_saved; ending: EX|PL; repost: overlap|replace|replenish|unrelated
- **Response context**: none attached by chains_v4 (1 creation line(s), 0 response episode(s) in its census) -> no_response_context
- **Exact-leg continuation** (available; legs 1 clean of 1, excluded status 0 / self 0; 50.00 sh, mq 50.00, rq 0.00): entry closes le10s 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; le60s 4 (+0 dust), 50.00 sh, pnl -4.49 $, fees 0.000 $, net -4.49 $; leEND 4 (+0 dust), 50.00 sh, pnl -4.49 $, fees 0.000 $, net -4.49 $; afterEND 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; leEND by closing leg: taker 0.00 sh net +0.00 $, maker 50.00 sh net -4.49 $; closing earlier inventory (kept apart) 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $
- **Residual** (kept apart): 0.00 sh unclosed; to END +0.00 $ (1/1 legs with m_end), to resolution +0.00 $ (1/1 resolved), prorated fee 0.000 $; fees of the clean legs 0.000 $
- **Views** (clean cohort, gross markouts beside): 10s gross -1.25 $ (1 valid), 60s -1.25 $ (1), END +27.00 $ (1); resolution net +29.00 $ (1 valid)
- **Takes of the address after the latest possible post** (anchor ws+65.084s): statuses inside_END:10; 10s 0 take(s), |dq| 0.00 sh, dir_D0 none, O4 none, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh); 60s 0 take(s), |dq| 0.00 sh, dir_D0 none, O4 none, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh); END 10 take(s), |dq| 150.24 sh, dir_D0 opposing:9;same:1, O4 reduces_all:1;reduces_none:1;reduces_some:8, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh) - the other takes are not exits of this order

## Chain 5 - rank 3: episode 20261024_3712725_r3_e014

order 0x01ebf81129 (ask 32c), day 20261024, market 3712725, address 0x57fbc891e89184da071ae9d97e745531cd7f718a; post grade G1_conflict; post window [ws+186.382s, ws+187.544s]; END ws+300.000s; other links of this order: none

| item | link | reading | histories | grade | summary |
|---|---|---|---|---|---|
| 1 | pre_state | TIED | 3 | observed_state:post_window:G1_conflict | G1_conflict: 2 saved seconds; features constant: spread, btc_5s_aligned, fair_gap, flow_5s, time_left; + unresolved_outside_window |
| 1b | transition | TIED | 4 | transition_context | 4 retained context(s): at_creation:level_496:improve_ask:straddles_window, joined_shown_cell, masked_recreation_in_window, unresolved_outsid |
| 2 | post | TIED | 4 | post_window:G1_conflict | ask 32c, G1_conflict le_10s, size >= 220.00 sh (EX/PL) |
| 3 | queue | TIED | 4 | scenario_queue_history | class conflicting; 1 compatible level(s); masked netting possible |
| 4a | ending | TIED | 2 | cancel_compatible_ending | EX/PL; cancel-compatible S_NOPRIO |
| 4b | repost | TIED | 3 | unresolved_alternatives | none none |
| 5 | takes | ONE-HISTORY | 1 | observed_execution | 5 take record(s); D0 {'opposing': 5}; clock-disagreeing 0; undetermined {} |
| 6 | value | ONE-HISTORY | 1 | recorded_fill_accounting | END net -263.95 $ (72/72 valid); 10s/60s/END same sign False |

- **Unresolved (TIED) links** (6): pre_state: spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(60,120]|spread=1; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=>3; flow_5s=>0; time_left=(60,120]|unresolved_outside_window; transition: at_creation:level_496:improve_ask:straddles_window|joined_shown_cell|masked_recreation_in_window|unresolved_outside_window; post: posting_time_unresolved_early + EX|posting_time_unresolved_early + PL|posting_time_unresolved_late + EX|posting_time_unresolved_late + PL; queue: front_at_first_fill @ level 496 (unexplained_decrease) + masked_netting_possible|behind_at_first_fill @ level 496 (unexplained_decrease) + masked_netting_possible|front_at_first_fill @ level 496 (unexplained_decrease) + as_saved|behind_at_first_fill @ level 496 (unexplained_decrease) + as_saved; ending: EX|PL; repost: pull_without_repost|unfilled_repost|no_pull
- **Response context**: none attached by chains_v4 (1 creation line(s), 0 response episode(s) in its census) -> no_response_context
- **Exact-leg continuation** (available; legs 7 clean of 7, excluded status 0 / self 0; 220.00 sh, mq 220.00, rq 0.00): entry closes le10s 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; le60s 1 (+0 dust), 8.14 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; leEND 1 (+0 dust), 8.14 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; afterEND 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; leEND by closing leg: taker 0.00 sh net +0.00 $, maker 8.14 sh net +0.00 $; closing earlier inventory (kept apart) 8 (+0 dust), 211.86 sh, pnl +6.63 $, fees 0.000 $, net +6.63 $
- **Residual** (kept apart): 0.00 sh unclosed; to END -0.00 $ (7/7 legs with m_end), to resolution -0.00 $ (7/7 resolved), prorated fee 0.000 $; fees of the clean legs 0.000 $
- **Views** (clean cohort, gross markouts beside): 10s gross -17.48 $ (7 valid), 60s -40.32 $ (7), END -140.80 $ (7); resolution net -149.60 $ (7 valid)
- **Takes of the address after the latest possible post** (anchor ws+187.544s): statuses before_anchor:16;inside_END:5; 10s 0 take(s), |dq| 0.00 sh, dir_D0 none, O4 none, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh); 60s 5 take(s), |dq| 58.00 sh, dir_D0 opposing:5, O4 reduces_all:5, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh); END 5 take(s), |dq| 58.00 sh, dir_D0 opposing:5, O4 reduces_all:5, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh) - the other takes are not exits of this order

## Chain 6 - rank 3: episode 20261024_3716727_r3_e017

order 0x66e4f311be (ask 61c), day 20261024, market 3716727, address 0x57fbc891e89184da071ae9d97e745531cd7f718a; post grade G2_bound_constrained; post window [ws+0.896s, ws+56.719s]; END ws+300.000s; other links of this order: none

| item | link | reading | histories | grade | summary |
|---|---|---|---|---|---|
| 1 | pre_state | TIED | 17 | observed_state:post_window:G2_bound_constrained | G2_bound_constrained: 57 saved seconds; features constant: btc_5s_aligned, flow_5s, time_left |
| 1b | transition | TIED | 3 | transition_context | 3 retained context(s): at_creation:level_28:unchanged, joined_shown_cell, masked_recreation_in_window |
| 2 | post | TIED | 4 | post_window:G2_bound_constrained | ask 61c, G2_bound_constrained gt_10s, size >= 10.00 sh (EX/PL) |
| 3 | queue | TIED | 4 | scenario_queue_history | class unidentified; 1 compatible level(s); masked netting possible |
| 4a | ending | TIED | 2 | cancel_compatible_ending | EX/PL; cancel-compatible S_NOPRIO |
| 4b | repost | TIED | 4 | scenario_dependent_link | reprice tied to 0xc5d1f7e499 |
| 5 | takes | ONE-HISTORY | 1 | observed_execution | 3 take record(s); D0 {'opposing': 3}; clock-disagreeing 0; undetermined {} |
| 6 | value | ONE-HISTORY | 1 | recorded_fill_accounting | END net -8.34 $ (67/67 valid); 10s/60s/END same sign False |

- **Unresolved (TIED) links** (6): pre_state: spread=1; depth_touch_own=(0,50]; depth_touch_opp=(0,50]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(0,50]; depth_touch_opp=(50,200]; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=(0,50]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=(0,50]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=<=0; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(0,1]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=(50,200]; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=>200; depth_touch_opp=(0,50]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=(50,200]; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=1; depth_touch_own=>200; depth_touch_opp=>200; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]|spread=2; depth_touch_own=(0,50]; depth_touch_opp=(0,50]; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=<=0; flow_5s=>0; time_left=(240,300]|spread=2; depth_touch_own=(0,50]; depth_touch_opp=>200; depth_2_own=(0,50]; btc_5s_aligned=flat; fair_gap=<=0; flow_5s=>0; time_left=(240,300]|spread=2; depth_touch_own=>200; depth_touch_opp=(50,200]; depth_2_own=>200; btc_5s_aligned=flat; fair_gap=(1,3]; flow_5s=>0; time_left=(240,300]; transition: at_creation:level_28:unchanged|joined_shown_cell|masked_recreation_in_window; post: early_post_rest_gt_10s + EX|early_post_rest_gt_10s + PL|late_post_rest_le_10s + EX|late_post_rest_le_10s + PL; queue: front_at_first_fill @ level 28 (ambiguous_order) + masked_netting_possible|behind_at_first_fill @ level 28 (ambiguous_order) + masked_netting_possible|front_at_first_fill @ level 28 (ambiguous_order) + as_saved|behind_at_first_fill @ level 28 (ambiguous_order) + as_saved; ending: EX|PL; repost: overlap|replace|replenish|unrelated
- **Response context**: none attached by chains_v4 (1 creation line(s), 0 response episode(s) in its census) -> no_response_context
- **Exact-leg continuation** (available; legs 1 clean of 1, excluded status 0 / self 0; 10.00 sh, mq 10.00, rq 0.00): entry closes le10s 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; le60s 2 (+0 dust), 10.00 sh, pnl -0.66 $, fees 0.050 $, net -0.71 $; leEND 2 (+0 dust), 10.00 sh, pnl -0.66 $, fees 0.050 $, net -0.71 $; afterEND 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $; leEND by closing leg: taker 7.39 sh net -0.42 $, maker 2.61 sh net -0.29 $; closing earlier inventory (kept apart) 0 (+0 dust), 0.00 sh, pnl +0.00 $, fees 0.000 $, net +0.00 $
- **Residual** (kept apart): 0.00 sh unclosed; to END +0.00 $ (1/1 legs with m_end), to resolution +0.00 $ (1/1 resolved), prorated fee 0.000 $; fees of the clean legs 0.000 $
- **Views** (clean cohort, gross markouts beside): 10s gross +1.05 $ (1 valid), 60s -0.65 $ (1), END -3.60 $ (1); resolution net -3.90 $ (1 valid)
- **Takes of the address after the latest possible post** (anchor ws+56.719s): statuses inside_END:3; 10s 0 take(s), |dq| 0.00 sh, dir_D0 none, O4 none, clock-disagreeing 0, FIFO-closing this order 0 (0.00 sh); 60s 1 take(s), |dq| 9.21 sh, dir_D0 opposing:1, O4 reduces_all:1, clock-disagreeing 0, FIFO-closing this order 1 (7.39 sh); END 3 take(s), |dq| 34.24 sh, dir_D0 opposing:3, O4 reduces_all:3, clock-disagreeing 0, FIFO-closing this order 1 (7.39 sh) - the other takes are not exits of this order

## Predicates

acceptance all hold: True; verifier violations: 0

- continuations_order_rows_unique: 1/1 hold=True
- continuations_not_split: 89/89 hold=True
- continuations_orders_equal_identity_orders: 1/1 hold=True
- continuations_response_keys_equal_identity_rows: 89/89 hold=True
- continuations_n_linked_equal: 89/89 hold=True
- continuations_addr_group_equal: 89/89 hold=True
- inputs_bound: 9/9 hold=True
- inputs_only_declared: 9/9 hold=True
- closes_built_on_this_fills_valued: 1/1 hold=True
- take_logic_v2_on_this_fills_valued: 1/1 hold=True
- touch_transition_read_this_events: 1/1 hold=True
- chains_v4_read_this_fills_valued_and_closes: 2/2 hold=True
- take_labels_equal_take_logic_v2: 2/2 hold=True
- response_links_as_declared: 1/1 hold=True
- chains_as_declared: 1/1 hold=True
- chain_has_every_link: 6/6 hold=True
- response_link_has_candidate: 89/89 hold=True
- order_addr_consistent: 95/95 hold=True
- order_group_consistent: 95/95 hold=True
- order_has_fill_rows: 95/95 hold=True
- legs_equal_saved_continuations: 89/89 hold=True
- views_equal_saved_where_all_clean: 89/89 hold=True
- chain_legs_equal_post_fill: 6/6 hold=True
- leg_keys_unique: 1/1 hold=True
- order_tick_on_grid_unique: 95/95 hold=True
- end_recovered: 95/95 hold=True
- end_equals_post_fill_we: 4149/4149 hold=True
- attribution_equals_mq: 29/29 hold=True
- no_close_on_excluded_leg: 95/95 hold=True
- fees_decompose: 29/29 hold=True
- saved_take_counts_reproduced: 12/12 hold=True
- chain_creation_lines_parse: 6/6 hold=True
- chain_creations_equal_census: 6/6 hold=True
- chain_responses_equal_census: 6/6 hold=True
- chain_context_intervals_match_episodes: 1/1 hold=True
- chain_its_closes_reconcile: 6/6 hold=True
- verifier_valued_legs_clean: 1/1 hold=True
- verifier_closes_within_end_bound: 1/1 hold=True
- verifier_takes_within_end_bound: 1/1 hold=True
- verifier_pooled_closes_distinct: 1/1 hold=True
- verifier_pooled_takes_distinct: 1/1 hold=True
- verifier_restoring_context_tick_equal: 1/1 hold=True
- verifier_attribution_equals_mq: 1/1 hold=True
