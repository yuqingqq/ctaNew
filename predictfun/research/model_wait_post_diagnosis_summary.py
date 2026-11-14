"""Descriptive JSON presentation of the frozen diagnosis; no new estimators.

This file formats already-computed measurements. It does not change diagnostic
design, fitted models, samples, thresholds, nulls or acceptance decisions.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def span(values):
    values = list(values)
    return [min(values), max(values)]


def entries(rows):
    n = sum(x['generations'] for x in rows)
    posted = sum(x['posted'] for x in rows)
    initial = sum(x['initial_posts'] for x in rows)
    after = sum(x['posted_after_wait'] for x in rows)
    return dict(generations=n, posts=posted, initial_posts=initial,
                posted_after_wait=after, eventual_post_fraction=posted/n,
                initial_post_fraction=initial/n, posts_after_wait_fraction=after/posted,
                posted_unseen_fraction=sum(x['post_unseen_fraction']*x['posted'] for x in rows)/posted,
                posted_sparse_fraction=sum(x['post_sparse_fraction']*x['posted'] for x in rows)/posted,
                first_unseen_fraction=sum(x['first_unseen_fraction']*x['generations'] for x in rows)/n)


def summarize(raw):
    fits = raw['saved_audit']['fits']
    generalization = {}
    for head in ('post', 'wait', 'delta'):
        rows = [f['heads'][head] for f in fits]
        generalization[head] = dict(
            cells=len(rows),
            worse_than_train_mean=sum(x['test']['r2_vs_train_mean'] < 0 for x in rows),
            worse_than_price_rung=sum(x['test']['rmse'] > x['test_price_rung_baseline']['rmse'] for x in rows),
            train_r2_range=span(x['train']['r2_vs_train_mean'] for x in rows),
            test_r2_range=span(x['test']['r2_vs_train_mean'] for x in rows),
            test_rank_ic_range=span(x['test']['rank_ic'] for x in rows),
            train_zero_fraction_range=span(x['train_target']['zero_fraction'] for x in rows),
            train_top1pct_squared_target_fraction_range=span(x['train_target']['top1pct_squared_target_fraction'] for x in rows))
    arms = {x['arm']: x for x in raw['totals']}
    runtime = {a: entries([x for x in raw['entry_generations'] if x['arm'] == a])
               for a in ('VALUE', 'VALUE_GUARD_EXIT')}
    runtime_by_side = {str(s): entries([x for x in raw['entry_generations'] if x['arm'] == 'VALUE' and x['side'] == s]) for s in (0, 1)}
    attribution = {}
    for arm in arms:
        rows = [x for x in raw['fill_attribution'] if x['arm'] == arm]
        attribution[arm] = {k: sum(x[k] for x in rows) for k in ('quantity', 'fill_spread', 'post_fill_move', 'pnl', 'cancel_pending_shares')}
        attribution[arm]['age_bands'] = [dict(age_band=b, **{k: sum(x[k] for x in rows if x['age_band'] == b) for k in ('quantity', 'pnl')}) for b in range(5)]
    source = [x['source_status'] for x in raw['receipts'].values()]
    value, guard = arms['VALUE'], arms['VALUE_GUARD_EXIT']
    support = raw['saved_audit']['probe_support']
    calibration = {}
    for side in (0, 1):
        rows = [f['test_selection'] for f in fits if f['side'] == side]
        n = sum(x['selected'] for x in rows)
        calibration[str(side)] = dict(
            eligible_probes=sum(x['n'] for x in rows), selected=n,
            **{k: sum(x[k]*x['selected'] for x in rows)/n for k in
               ('predicted_post', 'actual_post', 'predicted_delta', 'actual_delta')})
    rates = {a: 100*x['pnl']/x['shares'] for a, x in arms.items()}
    volume_value = {}
    for baseline in ('ALWAYS', 'TIMER'):
        b = arms[baseline]
        volume_value[baseline] = dict(
            pnl_difference=value['pnl']-b['pnl'],
            quantity_change_at_baseline_rate=(value['shares']-b['shares'])*b['pnl']/b['shares'],
            per_share_value_change_at_value_quantity=value['shares']*(rates['VALUE']-rates[baseline])/100,
            interpretation='Descriptive arithmetic identity; policy-dependent fills, not a causal participation/selection test.')
    return dict(
        asof=raw['asof'], status='complete_post_hoc_diagnosis',
        forward_validation=False, new_candidates=0,
        scope=dict(complete_days=raw['n_complete_days'], market_windows=raw['n_windows'],
                   dates=sorted(raw['receipts']), clock='M50', first_seconds=60, terminal_seconds=60.2,
                   economic_units='gross simulated dollars; local value targets cents per nominal share',
                   heldout_neutral_reference_probes=sum(x['n_test'] for x in fits),
                   exclusions={k: sum(x[k] for x in source) for k in ('untimed_legs_0_60', 'untimed_shares_0_60', 'grid_steps_without_row')},
                   original_market_arm_reconciliations=sum(x['baseline_metric_reconciliations'] for x in raw['receipts'].values())),
        generalization=generalization,
        selected_heldout_calibration=calibration,
        post_wait_test_target_correlation_range=span(x['test_post_wait_correlation'] for x in fits),
        labeled_probe_support=dict(n=sum(x['n'] for x in support),
                                  unseen_fraction=sum(x['unseen'] for x in support)/sum(x['n'] for x in support)),
        runtime_generations=runtime, value_runtime_by_side=runtime_by_side,
        portfolio_totals=raw['totals'], daily=raw['daily'], fill_attribution=attribution,
        gross_pnl_cents_per_filled_share=rates,
        value_versus_baseline_volume_value_identity=volume_value,
        guard_exit_sensitivity=dict(
            pnl_change=guard['pnl']-value['pnl'], placement_ratio=guard['places']/value['places'],
            filled_quantity_ratio=guard['shares']/value['shares'],
            days_with_smaller_loss=sum(
                next(x['pnl'] for x in raw['daily'] if x['day']==d and x['arm']=='VALUE_GUARD_EXIT') >
                next(x['pnl'] for x in raw['daily'] if x['day']==d and x['arm']=='VALUE') for d in raw['receipts']),
            correctly_relabelled_combined_candidate=False),
        interpretation=[
            'Generalization failure exists on held-out neutral-reference probes before closed-loop state mismatch. The old constant-head bug is not the explanation.',
            'The timing target is mostly zero and tail-heavy; two separately fitted correlated value heads do not reliably recover their difference.',
            'Runtime support bins describe a train/runtime gap. Post-after-WAIT counts describe delayed participation, not causal attribution of loss or a test of entry skill.',
            'Fill-time spread and subsequent mark movement reconcile to terminal PNL; this is an accounting decomposition, not a forecast-skill test.',
            'The guard exit changes the continuation and exposure as well as cancellations. Its gain is not proof of selective keeper skill; churn is material.',
            'Code inspection: whole-portfolio targets include ladder interactions, while model inputs lack a per-order snapshot of other resting prices, ages and remaining sizes. Aggregate inventory/exposure cannot distinguish all such states.'
        ],
        next_research_sequence=[
            'Keep a positive absolute POST-value gate; test a direct POST-minus-WAIT advantage estimate against earlier-data baselines and the existing action-matched null before another broad replay.',
            'Represent client-known ladder state and jointly empty/waiting configurations, while retaining neutral action populations and exclusions; never train only on fills.',
            'Build entry labels under the actual frozen order-specific keeper/continuation intended for evaluation, then evaluate the combined policy with action-level churn and fill-value accounting.',
            'Require untouched later complete days for validation; these dates are consumed development data. No model size or fixed retry timer is established as a remedy.'
        ],
        limitations=raw['limitations']+[
            'Touch means exactly received public best; near means public best or one tick behind. Both exclude price-improving quotes and do not establish front-of-queue residence.',
            'The keeper sensitivity uses historical LODO guard inputs and no matched activity null; it is not new deployable skill evidence.',
            'Largest-one-percent statistic is squared target magnitude, not a measured fraction of fitted residual loss.',
            'Runtime generations are policy-dependent: a POST advances the generation, and an unposted terminal generation is censored. The eventual-POST fraction cannot establish failure to abstain or rejection skill.',
            'Other-session maker/keeper results use different caps or horizons; this report makes no direct comparable top-maker performance claim.'
        ])


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    raw=json.loads(a.input.read_text());summary=summarize(raw)
    summary['raw_report_sha256']=digest(a.input)
    summary['presentation_source_sha256']=digest(Path(__file__))
    with a.output.open('x') as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
