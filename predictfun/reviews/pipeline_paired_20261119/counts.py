"""Every number the paired-candidate report quotes, computed from the saved outputs (declared before they were read; no number typed by hand):
the labels (paired vs neutral), the G3 diagnostics and prediction gate per head / side / day (pf_pipeline_paired), and per evaluation day the values at
B of every policy and side, the scenario counts, the readings and the plan's metrics (pf_pipeline_paired_eval). No sum over scenarios is printed as an
expected return. Usage (run_light): counts.py [research data dir]"""
import json
import sys
import pandas as pd
R = sys.argv[1] if len(sys.argv) > 1 else 'data/predict_fun/research'; B = 'M50|A58|L37|N100'
pj = json.loads(open(f'{R}/pipeline_paired_20261119/paired.json').read())
print(f"PAIRED: pi0 on {pj['warm']}, labels {pj['label_days']} + {pj['eval']}, multiplicity {pj['multiplicity']}; readings {pj['readings']}")
for d, x in pj['labels'].items():
    print(f'  labels {d}: ' + '; '.join(f"{k} {v['rows']:,} rows, paired mean {v['mean_adv']:+.4f} c (neutral {v['mean_neutral']:+.4f}), paired sum {v['sum_adv']:+,.1f}"
                                         + (f", pi0 pulls {v['pi0_pulls']:,}" if v['pi0_pulls'] is not None else '') for k, v in x.items()))
for d, xs in pj['diagnostics'].items():
    for s, hs in xs.items():
        for h, x in hs.items():
            print(f"  G3 {d} side {s} {h:<2}: POST rank {x['post_rank']:+.4f}, KEEP rank {x['keep_rank']:+.4f} (reached {x['keep_rank_reached']:+.4f}; {x['reached_rows']:,} of {x['rows']:,}); "
                  f"gate {x['gate']} {x['gate_groups']}; " + '; '.join(f"{k}: n {v['n']:,} predicted {v['predicted']:+,.1f} realized {v['realized']:+,.1f}" for k, v in x['groups'].items()))
pol = ('PA', 'A', 'A-PA', 'AF', 'AF-PA', 'AM', 'AM-PA', 'PI0', 'PI0-PA', 'NEU', 'NEU-PA', 'E1', 'E1-PA')
for day in ('20261026', '20261027'):
    je = json.loads(open(f'{R}/pipeline_paired_eval_20261119/eval_{day}.json').read()); cv = pd.read_csv(f'{R}/pipeline_paired_eval_20261119/eval_{day}.csv')
    g = lambda p_, sd='both': cv[(cv.policy == p_) & (cv.side == sd)].set_index('scenario').V
    print(f"EVAL {day}: admissions {je['admissions']}; orders with a pull step {je['pulled']}; null A at B {je['null_A'][B]:+,.1f}")
    for p_ in pol:
        print(f'   at B {p_:<6}: both {g(p_)[B]:>+11,.1f}, bid {g(p_, "bid")[B]:>+11,.1f}, ask {g(p_, "ask")[B]:>+11,.1f}; above 0 in {(g(p_) > 0).sum()}/{len(g(p_))} scenarios '
              f'(min {g(p_).min():+,.0f}, max {g(p_).max():+,.0f})')
    for ref in ('A-PA', 'PI0', 'NEU', 'E1', 'PA'):
        x = g('A') - g(ref); print(f'   A - {ref}: at B {x[B]:+,.1f}; above in {(x > 0).sum()}/{len(x)} scenarios (min {x.min():+,.0f}, max {x.max():+,.0f})')
    for k, v in je['readings'].items():
        print(f'   reading {k}: {v}')
    for p_ in ('A', 'A-PA', 'PI0', 'NEU', 'E1', 'PA'):
        x = je['metrics'][p_]
        for sd in ('both', 'bid', 'ask'):
            y = x[sd]
            print(f"   METRICS {p_:<5} {sd:<4}: value {y['value']:+,.1f} ({y['per_original_opportunity']:+.4f} c/opp), vs same-admission PULL-ALL {y['vs_same_admission_PULL_ALL']:+,.1f}, vs PI0 {y['vs_PI0']:+,.1f}; "
                  f"attempts {y['attempts']:,}, accepted {y['accepted']:,}, filled {y['filled']:,}; resting {y['resting']}, at best {y['at_best']}; lifetime {y['lifetime_p10_p50_p90']} {y['causes']}; "
                  f"first review {y['first_review_of_live']} of live / {y['first_review_of_pulls']} of pulls; cancels {y['cancels_per_hour']}/h, {y['cancels_per_100_filled_shares']} per 100 sh; "
                  f"loss P {y['p_loss']}, severe P {y['p_severe']}, p95/p99 {y['loss_p95']}/{y['loss_p99']}")
        print(f"   METRICS {p_:<5} pairs: {x['pairs']}")
    print(f"   REACHED-STATE SHIFT: {je['shift']}")
