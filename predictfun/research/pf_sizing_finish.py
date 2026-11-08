"""Assemble a compact, identity-linked receipt after all declared jobs finish."""
import datetime as dt
import json
import subprocess
import numpy as np
import pandas as pd
from .pf_sizing_study import DAYS, ARMS, DATA, DEST, HERE, sha, write
from .pf_sizing_run import specs, stem, name, folder, reconcile


def read(path):
    return json.loads(path.read_text())


def exact_paths(a, b):
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b), check_exact=True)
    except AssertionError:
        return False
    return True


def stress_summary(tm, label, spec):
    path = DATA/'step4'/f'ladder_tokyoreal_custom_e2_preds_basis_study_sst_ack99_{tm}'/'20261103'
    s = read(path/'summary.json')
    execution = read(path/'study_execution.json')
    expected = read(DEST/'cancel_ack_latency.json')['quantiles_ms']['p99']
    if execution['cancel_ack_ms'] != expected:
        raise ValueError('wrong acknowledgement stress')
    w = pd.read_parquet(path/'win'/stem(spec))
    f = pd.read_parquet(path/'fills'/stem(spec))
    q = pd.read_parquet(path/'requests'/stem(spec))
    if len(w) != 288:
        raise ValueError('incomplete stress population')
    reconcile(w, f)
    eq = np.r_[0., w.pnl.cumsum().to_numpy()]
    return dict(timing=tm, variant=label, windows=len(w), settle=float(w.pnl.sum()),
        plus1=float(f.mo1.sum()), mark_stop=float(w.mtm_stop.sum()), carry=float(w.carry_stop.sum()),
        shares=float(w.shares.sum()), max_drawdown=float((eq-np.maximum.accumulate(eq)).min()),
        mean_abs_net=float(w.net_end.abs().mean()), refusals=int(w.refused.sum()),
        halted_windows=int(w.halted.sum()), requests=q.kind.value_counts().to_dict(),
        counters=s['arms'][name(spec)]['live']['counters'],
        statuses=dict(missing_mark_stop=int(w.mtm_stop.isna().sum()), missing_mo1=int(f.mo1.isna().sum())),
        execution=execution)


def main():
    strategy = read(DEST/'strategy_report.json')
    probability = read(DEST/'settlement_report.json')
    nonlinear = read(DEST/'settlement_nonlinear_report.json')
    shadow = read(DEST/'nonlinear_inventory/inventory_shadow_report.json')
    checks = read(DEST/'cache_full_replay_control.json')
    if not checks['exact'] or checks['comparisons'] != 24:
        raise ValueError('missing full cache-equivalence control')
    indexed = {(r['timing'], r['variant']):r for r in strategy['results']}
    comparisons = []
    for tm in ['J','X']:
        for budget in ['off','150']:
            b = indexed[(tm, 'baseline_'+budget)]
            for variant in ARMS:
                r = indexed[(tm, variant+'_'+budget)]
                if r['windows'] != 2592:
                    raise ValueError('incomplete main replay')
                daydelta = [x['settle']-y['settle'] for x,y in zip(r['per_day'], b['per_day'])]
                comparisons.append(dict(timing=tm, variant=variant, budget=budget,
                    settlement_delta=r['settle']-b['settle'], plus1_delta=r['plus1']-b['plus1'],
                    mark_stop_delta=r['mark_stop']-b['mark_stop'], carry_delta=r['carry']-b['carry'],
                    place_request_change_pct=100*(r['requests'].get('place',0)/b['requests']['place']-1),
                    cancel_request_change_pct=100*(r['requests'].get('cancel',0)/b['requests']['cancel']-1),
                    days_settlement_better=sum(d>1e-8 for d in daydelta),
                    days_settlement_worse=sum(d< -1e-8 for d in daydelta),
                    days_settlement_equal=sum(abs(d)<=1e-8 for d in daydelta)))
    budget_checks = []
    execution = {}; unique_markets = set()
    for tm in ['J','X']:
        for day in DAYS:
            path = folder(day,tm)
            if tm == 'J':
                mids = pd.read_parquet(path/'win'/stem(specs()['baseline_off']), columns=['mid']).mid
                if unique_markets.intersection(mids):
                    raise ValueError('market repeated across replay dates')
                unique_markets.update(mids)
            # Earliest uncached runs predate execution metadata; their source is
            # separately pinned, and the full cached control compares all arms.
            meta = path/'study_execution.json'
            execution[f'{tm}/{day}'] = read(meta) if meta.exists() else dict(
                cap_cache=False, metadata_predates_instrumentation=True,
                control='cache_full_replay_control.json' if day=='20261023' else 'cap_reuse_control.json')
            if execution[f'{tm}/{day}'].get('cancel_ack_ms',0) != 0:
                raise ValueError('stress delay leaked into main replay')
            for variant in ARMS:
                for kind in ['win','fills','requests']:
                    a = path/kind/stem(specs()[variant+'_off'])
                    b = path/kind/stem(specs()[variant+'_150'])
                    budget_checks.append(dict(timing=tm, day=day, variant=variant, kind=kind,
                        exact=exact_paths(a,b), uncapped_sha=sha(a), capped_sha=sha(b)))
    stress = [stress_summary(tm,label,spec) for tm in ['J','X'] for label,spec in specs().items()]
    stress_report = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        days=['20261103'], windows_per_variant=288, results=stress,
        interpretation='One consumed day, uniform p99 reservation-release delay; no confidence interval or calibrated live-OMS claim. Each arm starts with zero realised PnL.',
        latency=read(DEST/'cancel_ack_latency.json'))
    write(DEST/'cancel_stress_report.json',stress_report)
    sources = list(HERE.glob('pf_sizing*.py')) + list(HERE.glob('pf_settlement*.py'))
    sources += [HERE/'tests/test_sizing_study.py', HERE/'run_sizing_study.sh', HERE/'run_sizing_followups.sh']
    sources += [HERE/'pf_model_spec.py', HERE/'pf_step4_ladder_v3.py', HERE/'pf_step5_features_rl.py',
                HERE/'pf_clock_rl.py', HERE/'pf_io.py']
    artifacts = ['declaration.json','declaration_v2.json','declaration_v3.json',
        'nonlinear_declaration.json','nonlinear_declaration_v2.json',
        'cancel_tail_declaration.json','cancel_tail_declaration_v2.json',
        'code_identity.json','code_identity_v2.json','code_identity_v3.json',
        'engine_parity.json','cap_reuse_control.json','capacity_reuse_control.json',
        'cache_full_replay_control.json','strategy_report.json','settlement_report.json',
        'settlement_nonlinear_report.json','inventory_shadow_report.json',
        'nonlinear_inventory/inventory_shadow_report.json','cancel_ack_latency.json','cancel_stress_report.json']
    artifacts += [f'research/step5/{d}/prepare.json' for d in DAYS]
    fixed_inputs = [HERE.parent/'freeze/pm_ladder_v1'/f for f in ['params.json','guard_bid.txt','guard_ask.txt']]
    fixed_inputs += [DATA/'step4'/f for f in ['minute_lines_t60k.json','inv_r60.json']]
    test_log=(DEST/'logs/sizing_source_tests.log').read_text()
    if '\nOK\n' not in test_log or 'rc=0 ' not in test_log:
        raise ValueError('source controls did not pass')
    receipt = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        branch='research/predict-sizing-settlement-20261108', artifact_root=str(DEST),
        source_sha256={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources},
        fixed_input_sha256={str(p):sha(p) for p in fixed_inputs},
        artifact_sha256={p:sha(DEST/p) for p in artifacts},
        declarations={p:read(DEST/p) for p in artifacts if 'declaration' in p},
        source_controls_log_sha=sha(DEST/'logs/sizing_source_tests.log'),
        source_controls_log=test_log,
        population=dict(strategy_days=DAYS, windows_per_arm_timing=2592, unique_market_outcomes=len(unique_markets),
            strategy_variants=4, loss_budget_scenarios=2, timings=2,
            forecast_test_days=DAYS[3:], forecast_markets=nonlinear['n_markets'],
            forecast_actions=nonlinear['n_actions'], forecast_statuses=probability['statuses']),
        strategy_results=strategy['results'], strategy_comparisons=comparisons,
        budget_path_checks=budget_checks, budget_paths_all_equal=all(r['exact'] for r in budget_checks),
        execution_metadata=execution, cache_control=checks,
        forecast_results=nonlinear, inventory_shadow=shadow, cancellation_stress=stress_report,
        predicates=dict(
            forecast_brier_improvement={m:s['brier'] < nonlinear['overall']['market_mid']['brier']
                for m,s in nonlinear['overall'].items() if m!='market_mid'},
            sizing_settlement_improvement_both_timings={v:all(indexed[(tm,v+'_off')]['settle'] > indexed[(tm,'baseline_off')]['settle'] for tm in ['J','X']) for v in ARMS if v!='baseline'}),
        limitations=[
            'Strategy guards trained on eight of the nine consumed replay dates; in-sample comparison, not expected live return.',
            'Chronological forecast fits use earlier, available venue labels; all dates were already consumed, no new validation claim.',
            'Neutral scheduled inventory opportunities, not selected fills; shadow dollar values are per opportunity, not summed strategy PnL.',
            'Fixed delays and execution approximation omit full live OMS/watchdog parity and own market impact.',
            'Loss budget carries realised PnL across studied dates but skips unstudied gaps; no continuous live-path claim.',
            'Cancellation stress is one day at a uniform completed-cancel p99, not an empirical latency replay; missing acknowledgements remain censored.',
            'Nonlinear and cancellation stress extensions were declared after initial results; both explicitly exploratory.',
            'Paired uncertainty clusters by day; six forecast test days give weak inference. Matched random controls are conditional nulls, not performance intervals.',
            'Rebates excluded from trading PnL and the loss budget; $150 remains a test scenario, independent of the 150-share exposure cap.'
        ], forward_data_read=False, live_strategy_changed=False, live_limit_changed=False,
        deployment=False, forward_freeze_changed=False)
    write(DEST/'receipt.json',receipt)
    target=HERE.parent/'reviews/sizing_settlement_20261108/receipt.json'
    write(target,receipt)
    print(json.dumps(dict(budget_paths_all_equal=receipt['budget_paths_all_equal'],
        strategy_comparisons=comparisons, predicates=receipt['predicates']),indent=2))


if __name__=='__main__':
    main()
