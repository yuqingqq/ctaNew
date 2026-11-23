# Logic-chain correction v1 - episode paths (contract section 4)

Selection (predetermined, not outcome-selected): per rank, the first two episodes in `order_estimator/cases.csv` row order of first appearance. Saved evidence only; descriptive, four consumed days; no identified intent. Times are venue ms relative to the market window start ws. Evidence grades: observed_execution, continued_execution, scenario_dependent_link (tied / sequential_under_scenarios), cancel_compatible_ending, post_window:G0..G4 (section 1); size priors are prior-driven estimates.

## Path 1 - rank 1: episode 20261024_3716727_r1_e018

day 20261024, market 3716727, address 0x6da6cb464f92ae7ad4ec3d239c81719cb1d0ae03; cases order:0x88c0095c1e (first appearance: cases.csv data row 1)

- **POST** order 0x88c0095c1e: ask at 69c; size alternatives EX 30.00 sh, PL >= 30.01 sh (size status tied; prior-driven: P_cmd ('EX', 30.0), P_drop ('EX', 30.0), P_lot ('EX', 30.0), P_clip ('EX', 30.0)). Union window [ws+2.325s, ws+80.506s], width 78181 ms (gt_10s); lifecycle post_class assumption_sensitive; packet status ok -> **G4_scenario_only** (scenario union estimate only). Logic table: queue_priority_before_demand timing_sensitive -> timing_sensitive; evidence grade B_constrained -> C_scenario.
  - alternative that changes the logic: scenario window only (lifecycle assumption_sensitive): the scenario-free bound is not saved; an earlier post whose first partial fill was masked by replenishment would make this a long-resting order.
- **FILLS** 1 fill(s), 30.00 sh (observed executions); first <= ws+80.506s, last <= ws+80.506s; 0 later fill(s) of the same hash (continued execution, not a private KEEP); queue at the first fill unidentified.
- **TERMINAL** feasible end histories EX|PL; per scenario S_FIFO {EX,PL}, S_FIFO_LO {EX,PL}, S_NOPRIO {EX,PL}; cancel-compatible (EX/PL tie) under S_FIFO: yes, S_NOPRIO: yes (pull_identified False, False; removal class constrained, constrained); saved END rows: inventory_driven_pull unidentified (window condition compatible); pull_after_adverse_move unidentified (window condition timing_sensitive).
  - alternative that changes the logic: EX / PL tie: if the order was exhausted there was no pull decision; if pulled, the END context is a candidate trigger only.
- **SUCCESSOR** reprice to 0x89e05c2583 (first of 2 filled successor(s)); link status sequential_identified, gap [33002, 37154] ms; relations replace|replenish|unrelated -> sequential_under_scenarios, evidence grade scenario_dependent_link; identified repost: no.
  - alternative that changes the logic: an unrelated later order: a sequential timing does not identify a repost or its intent.
- **TAKES** no selected take case in this episode (its window holds 12 taker leg(s), not classified here).
- **ECONOMICS** (saved episode_summary.csv; descriptive: a later outcome is not the value of an action) legs 47 (maker 35, taker 12), 548.79 sh; ep_net_end -42.93, ep_net_res -45.46, fee 0.70; mq 548.79, rq 0.00; position 0.00 -> -95.35; outcome losing_continuation; matched gap ok: close_net 168.32, resid 0.00, total 168.10 (8.23 c/sh).

## Path 2 - rank 1: episode 20261024_3702960_r1_e006

day 20261024, market 3702960, address 0x6da6cb464f92ae7ad4ec3d239c81719cb1d0ae03; cases order:0x02fb943dae (first appearance: cases.csv data row 2)

- **POST** order 0x02fb943dae: ask at 13c; size alternatives EX 6.67 sh, PL >= 6.68 sh (size status tied; prior-driven: P_cmd ('EX', 6.6666), P_drop ('PL', 66.6666), P_lot ('PL', 10.0), P_clip ('PL', 30.0)). Union window [ws+211.237s, ws+213.303s], width 2066 ms (le_10s); lifecycle post_class assumption_sensitive; packet status ok -> **G4_scenario_only** (scenario union estimate only). Logic table: queue_priority_before_demand contradicted -> timing_sensitive; evidence grade B_constrained -> C_scenario.
  - alternative that changes the logic: scenario window only (lifecycle assumption_sensitive): the scenario-free bound is not saved; an earlier post whose first partial fill was masked by replenishment would make this a long-resting order.
- **FILLS** 1 fill(s), 6.67 sh (observed executions); first <= ws+213.303s, last <= ws+213.303s; 0 later fill(s) of the same hash (continued execution, not a private KEEP); queue at the first fill unidentified.
- **TERMINAL** feasible end histories EX|PL; per scenario S_FIFO {EX,PL}, S_FIFO_LO {EX,PL}, S_NOPRIO {EX,PL}; cancel-compatible (EX/PL tie) under S_FIFO: yes, S_NOPRIO: yes (pull_identified False, False; removal class assumption_sensitive, assumption_sensitive); saved END rows: inventory_driven_pull unidentified (window condition timing_sensitive); pull_after_adverse_move unidentified (window condition timing_sensitive).
  - alternative that changes the logic: EX / PL tie: if the order was exhausted there was no pull decision; if pulled, the END context is a candidate trigger only.
- **SUCCESSOR** reprice to 0xdef88dc345 (first of 1 filled successor(s)); link status tied, gap [-57944, 2176] ms; relations overlap|replace|replenish|unrelated -> tied, evidence grade scenario_dependent_link; identified repost: no.
  - alternative that changes the logic: the successor may already have been resting (a ladder / overlapping order) or be unrelated: no repost decision is identified.
- **TAKES** no selected take case in this episode (its window holds 8 taker leg(s), not classified here).
- **ECONOMICS** (saved episode_summary.csv; descriptive: a later outcome is not the value of an action) legs 21 (maker 13, taker 8), 250.37 sh; ep_net_end 5.87, ep_net_res 6.59, fee 0.32; mq 221.68, rq 28.68; position -17.90 -> -28.68; outcome profitable_continuation; matched gap ok: close_net -6.43, resid 1.32, total -5.13 (-0.48 c/sh).

## Path 3 - rank 2: episode 20261024_3702960_r2_e007

day 20261024, market 3702960, address 0x7d32c568a9bea1714036972601c51f4902c67091; cases order:0x084be2d8a4 (first appearance: cases.csv data row 17)

- **POST** order 0x084be2d8a4: bid at 49c; size alternatives EX 50.00 sh, PL [50.01, 51.96] sh (size status conflict; prior-driven: P_cmd ('EX', 50.0), P_drop ('PL', 51.960784), P_lot ('EX', 50.0), P_clip ('EX', 50.0)). Union window [ws+4.027s, ws+4.785s], width 758 ms (le_1s); lifecycle post_class assumption_sensitive; packet status capacity_bounds_void+clock_capacity_conflict+post_after_first_fill -> **G1_conflict** (void: conflict flags). Logic table: queue_priority_before_demand contradicted -> unidentified; evidence grade B_constrained -> E_unresolved.
  - alternative that changes the logic: conflict flags void the capacity claims: the posting time is unresolved.
- **FILLS** 3 fill(s), 50.00 sh (observed executions); first <= ws+4.785s, last <= ws+7.030s; 2 later fill(s) of the same hash (continued execution, not a private KEEP); queue at the first fill conflicting.
- **TERMINAL** feasible end histories EX|PL; per scenario S_FIFO {EX}, S_FIFO_LO {EX,PL}, S_NOPRIO {EX}; cancel-compatible (EX/PL tie) under S_FIFO: no, S_NOPRIO: no; saved END rows: inventory_driven_pull unidentified (window condition compatible); pull_after_adverse_move unidentified (window condition contradicted).
  - alternative that changes the logic: EX / PL tie: if the order was exhausted there was no pull decision; if pulled, the END context is a candidate trigger only.
- **SUCCESSOR** reprice to 0xd0f156e83a (first of 2 filled successor(s)); link status sequential_identified, gap [53, 4290] ms; relations replace|replenish|unrelated -> sequential_under_scenarios, evidence grade scenario_dependent_link; identified repost: no.
  - alternative that changes the logic: an unrelated later order: a sequential timing does not identify a repost or its intent.
- **TAKES** no selected take case in this episode (its window holds 5 taker leg(s), not classified here).
- **ECONOMICS** (saved episode_summary.csv; descriptive: a later outcome is not the value of an action) legs 33 (maker 28, taker 5), 468.91 sh; ep_net_end -46.33, ep_net_res -52.05, fee 0.50; mq 468.91, rq 0.00; position 0.00 -> 101.01; outcome losing_continuation; matched gap ok: close_net 1.58, resid -0.10, total 1.48 (0.10 c/sh).

## Path 4 - rank 2: episode 20261024_3712725_r2_e015

day 20261024, market 3712725, address 0x7d32c568a9bea1714036972601c51f4902c67091; cases order:0xa46229318a (first appearance: cases.csv data row 18)

- **POST** order 0xa46229318a: bid at 42c; size alternatives EX 50.00 sh, PL >= 50.01 sh (size status distinguished_under_priority; prior-driven: P_cmd ('EX', 50.0), P_drop ('EX', 50.0), P_lot ('EX', 50.0), P_clip ('EX', 50.0)). Union window [ws+62.538s, ws+65.084s], width 2546 ms (le_10s); lifecycle post_class constrained; packet status ok -> **G2_bound_constrained** (lifecycle scenario-free bound (conditional on H1 / H2 / capacity and the maximum match lag)). Logic table: queue_priority_before_demand contradicted -> contradicted; evidence grade B_constrained -> B_constrained.
  - alternative that changes the logic: bound-constrained <= 10 s: an early queue-building post is excluded only under H1 / H2 / capacity and the 2 s maximum match lag; a later match or a broken premise reopens it.
- **FILLS** 1 fill(s), 50.00 sh (observed executions); first <= ws+65.084s, last <= ws+65.084s; 0 later fill(s) of the same hash (continued execution, not a private KEEP); queue at the first fill unidentified.
- **TERMINAL** feasible end histories EX|PL; per scenario S_FIFO {EX}, S_FIFO_LO {EX}, S_NOPRIO {EX,PL}; cancel-compatible (EX/PL tie) under S_FIFO: no, S_NOPRIO: yes (pull_identified False; removal class assumption_sensitive); saved END rows: inventory_driven_pull unidentified (window condition compatible); pull_after_adverse_move unidentified (window condition contradicted).
  - alternative that changes the logic: EX / PL tie: if the order was exhausted there was no pull decision; if pulled, the END context is a candidate trigger only.
- **SUCCESSOR** reprice to 0x6f5fc55834 (first of 3 filled successor(s)); link status tied, gap [-2864, 6461] ms; relations overlap|replace|replenish|unrelated -> tied, evidence grade scenario_dependent_link; identified repost: no.
  - alternative that changes the logic: the successor may already have been resting (a ladder / overlapping order) or be unrelated: no repost decision is identified.
- **TAKES** no selected take case in this episode (its window holds 0 taker leg(s), not classified here).
- **ECONOMICS** (saved episode_summary.csv; descriptive: a later outcome is not the value of an action) legs 22 (maker 22, taker 0), 555.32 sh; ep_net_end 77.05, ep_net_res 82.17, fee 0.00; mq 555.32, rq 0.00; position 146.73 -> 274.92; outcome profitable_continuation; matched gap ok: close_net -3.57, resid -29.40, total -33.16 (-0.29 c/sh).

## Path 5 - rank 3: episode 20261024_3712725_r3_e014

day 20261024, market 3712725, address 0x57fbc891e89184da071ae9d97e745531cd7f718a; cases order:0x01ebf81129|take:01a16f22-ad6 (first appearance: cases.csv data row 33)

- **POST** order 0x01ebf81129: ask at 32c; size alternatives EX 220.00 sh, PL [220.01, 260.00] sh (size status conflict; prior-driven: P_cmd ('EX', 220.0), P_drop ('PL', 227.55), P_lot ('EX', 220.0), P_clip ('EX', 220.0)). Union window [ws+186.382s, ws+187.544s], width 1162 ms (le_10s); lifecycle post_class assumption_sensitive; packet status ahead_hi_negative+capacity_bounds_void+clock_capacity_conflict+post_after_first_fill -> **G1_conflict** (void: conflict flags). Logic table: queue_priority_before_demand contradicted -> unidentified; evidence grade B_constrained -> E_unresolved.
  - alternative that changes the logic: conflict flags void the capacity claims: the posting time is unresolved.
- **FILLS** 7 fill(s), 220.00 sh (observed executions); first <= ws+187.544s, last <= ws+195.961s; 6 later fill(s) of the same hash (continued execution, not a private KEEP); queue at the first fill conflicting.
- **TERMINAL** feasible end histories EX|PL; per scenario S_FIFO {EX}, S_FIFO_LO {EX}, S_NOPRIO {EX,PL}; cancel-compatible (EX/PL tie) under S_FIFO: no, S_NOPRIO: yes (pull_identified False; removal class assumption_sensitive); saved END rows: inventory_driven_pull unidentified (window condition timing_sensitive); pull_after_adverse_move unidentified (window condition contradicted).
  - alternative that changes the logic: EX / PL tie: if the order was exhausted there was no pull decision; if pulled, the END context is a candidate trigger only.
- **SUCCESSOR** none: no later first-filled same-side order: an unfilled successor or none (invisible in fills).
  - alternative that changes the logic: a pull without a repost and an unfilled repost are both compatible.
- **TAKE** record 01a16f22-ad67: YES buy 12.21 sh @ 0.340; position -370.66 -> -358.45; class recorded reduce_partial / certain reduce_partial / simultaneous reduce_partial (timing-sensitive False); own fills in the 5 s before: certain 13, possible 19; status ok; evidence grade observed_execution (class on the retained status-ok ledger). Saved TAKE rows: after_fill_burst compatible; directional insufficient_comparison; inventory_unwind compatible.
  - alternative that changes the logic: an excluded earlier leg or an unobserved inventory path could change reduce vs add (the class is conditional on the retained ledger).
- **ECONOMICS** (saved episode_summary.csv; descriptive: a later outcome is not the value of an action) legs 72 (maker 67, taker 5), 2081.50 sh; ep_net_end -263.95, ep_net_res -281.96, fee 0.39; mq 1974.96, rq 106.54; position 107.59 -> -323.48; outcome losing_continuation; matched gap ok: close_net -73.47, resid -65.06, total -138.53 (-2.02 c/sh).

## Path 6 - rank 3: episode 20261024_3716727_r3_e017

day 20261024, market 3716727, address 0x57fbc891e89184da071ae9d97e745531cd7f718a; cases order:0x66e4f311be|take:01a16f58-7a5 (first appearance: cases.csv data row 34)

- **POST** order 0x66e4f311be: ask at 61c; size alternatives EX 10.00 sh, PL >= 10.01 sh (size status distinguished_under_priority; prior-driven: P_cmd ('EX', 10.0), P_drop ('PL', 58.17), P_lot ('EX', 10.0), P_clip ('EX', 10.0)). Union window [ws+0.896s, ws+56.719s], width 55823 ms (gt_10s); lifecycle post_class constrained; packet status ok -> **G2_bound_constrained** (lifecycle scenario-free bound (conditional on H1 / H2 / capacity and the maximum match lag)). Logic table: queue_priority_before_demand timing_sensitive -> timing_sensitive; evidence grade B_constrained -> B_constrained.
  - alternative that changes the logic: bound-constrained but wider than 10 s: an early queue-building post and a late post are both compatible.
- **FILLS** 1 fill(s), 10.00 sh (observed executions); first <= ws+56.719s, last <= ws+56.719s; 0 later fill(s) of the same hash (continued execution, not a private KEEP); queue at the first fill unidentified.
- **TERMINAL** feasible end histories EX|PL; per scenario S_FIFO {EX}, S_FIFO_LO {EX}, S_NOPRIO {EX,PL}; cancel-compatible (EX/PL tie) under S_FIFO: no, S_NOPRIO: yes (pull_identified False; removal class assumption_sensitive); saved END rows: inventory_driven_pull unidentified (window condition timing_sensitive); pull_after_adverse_move unidentified (window condition contradicted).
  - alternative that changes the logic: EX / PL tie: if the order was exhausted there was no pull decision; if pulled, the END context is a candidate trigger only.
- **SUCCESSOR** reprice to 0xc5d1f7e499 (first of 2 filled successor(s)); link status tied, gap [-17292, 13731] ms; relations overlap|replace|replenish|unrelated -> tied, evidence grade scenario_dependent_link; identified repost: no.
  - alternative that changes the logic: the successor may already have been resting (a ladder / overlapping order) or be unrelated: no repost decision is identified.
- **TAKE** record 01a16f58-7a5d: YES sell 4.25 sh @ 0.540; position 27.11 -> 22.86; class recorded reduce_partial / certain reduce_partial / simultaneous reduce_partial (timing-sensitive False); own fills in the 5 s before: certain 7, possible 8; status simultaneous_action; evidence grade observed_execution (class on the retained status-ok ledger). Saved TAKE rows: after_fill_burst compatible; directional insufficient_comparison; inventory_unwind compatible.
  - alternative that changes the logic: an excluded earlier leg or an unobserved inventory path could change reduce vs add (the class is conditional on the retained ledger).
- **ECONOMICS** (saved episode_summary.csv; descriptive: a later outcome is not the value of an action) legs 67 (maker 64, taker 3), 470.58 sh; ep_net_end -8.34, ep_net_res -8.08, fee 0.29; mq 468.43, rq 2.15; position 0.00 -> 2.15; outcome losing_continuation; matched gap ok: close_net -9.64, resid 1.29, total -8.34 (-4.11 c/sh).

