"""Post-result presentation of frozen artifacts; no new model or selection."""
import datetime as dt
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .pf_pm_leadlag_data import DEST, REVIEW, sha, write


def main():
    report = DEST / 'report.json'
    r = json.loads(report.read_text())
    v = json.loads((DEST / 'verification.json').read_text())
    if sha(report) != v['report_sha']:
        raise ValueError('unverified report')
    lags = r['correlations'][0]['lag_ms']
    phases = []
    for row in r['correlations']:
        if (row['clock'], row['max_age_ms']) != ('recv', 1000):
            continue
        phase = row['phase']
        coverage = next(x for x in r['coverage'] if x['phase'] == phase
                        and (x['clock'], x['max_age_ms']) == ('recv', 1000))
        daily = [x for x in r['daily_correlations'] if x['phase'] == phase
                 and (x['clock'], x['max_age_ms']) == ('recv', 1000)]
        phases.append(dict(phase=phase, windows=row['windows'],
            peak_lag_ms=row['peak_positive_correlation_lag_ms'], peak_correlation=row['peak_correlation'],
            peak_pair_n=row['pair_counts'][lags.index(row['peak_positive_correlation_lag_ms'])],
            zero_lag_eligible_n=coverage['paired_valid_changes'], scheduled_n=coverage['expected_grid'],
            coverage_percent=100 * coverage['paired_valid_changes'] / coverage['expected_grid'],
            daily=[dict(day=x['day'], peak_lag_ms=lags[int(np.argmax(x['correlations']))],
                        asymmetry_200ms=x['correlations'][lags.index(200)]-x['correlations'][lags.index(-200)])
                   for x in daily]))
    forecasts = []
    for h in (.5, 1., 5.):
        rows = [x for x in r['prediction'] if x['horizon_s'] == h and x['status'] == 'scored']
        n = sum(x['n'] for x in rows)
        mse = {k: sum(x[k] * x['n'] for x in rows)/n
               for k in ('baseline_mse', 'plus_pm_mse', 'zero_mse')}
        statuses = [x for x in r['prediction_status'] if x['horizon_s'] == h and x['day'] in {a['day'] for a in rows}]
        forecasts.append(dict(horizon_s=h, test_dates=len(rows), windows=sum(x['windows'] for x in rows),
            eligible_n=n, scheduled_n=sum(x['scheduled'] for x in statuses),
            excluded_n=sum(x['excluded_missing_or_stale'] for x in statuses), **mse,
            improvement_percent=100*(mse['baseline_mse']-mse['plus_pm_mse'])/mse['baseline_mse'],
            improved_dates=sum(x['improvement_percent'] > 0 for x in rows)))
    # Describe the primary eligibility restriction, without removing or backfilling states.
    exclusions = []
    grids = [pd.read_parquet(x['grid']) for x in r['population_status'] if x['status'] == 'built']
    for phase in phases:
        lo, hi = phase['phase']; counts = dict(scheduled=0, invalid_mid_endpoint=0, stale_endpoint=0, eligible=0)
        for g in grids:
            mids = g[['pf_recv_mid', 'pm_recv_mid']].to_numpy()
            ages = g[['pf_recv_age', 'pm_recv_age']].to_numpy()
            valid = np.isfinite(mids).all(1)
            fresh = np.isfinite(ages).all(1) & (ages >= 0).all(1) & (ages <= 1000).all(1)
            i = np.flatnonzero((g.rel >= lo) & (g.rel < hi))
            book = valid[i] & valid[i-1]; age = fresh[i] & fresh[i-1]
            counts['scheduled'] += len(i)
            counts['invalid_mid_endpoint'] += int((~book).sum())
            counts['stale_endpoint'] += int((book & ~age).sum())
            counts['eligible'] += int((book & age).sum())
        if counts['eligible'] != phase['zero_lag_eligible_n']:
            raise ValueError('coverage presentation mismatch')
        exclusions.append(dict(phase=phase['phase'], **counts))
    summary = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), result_as_of=r['as_of'],
        report_sha=sha(report), presentation_source_sha=sha(__file__),
        role='Post-result aggregation/presentation only; frozen analysis remains unchanged.',
        phases=phases, forecast=forecasts, coverage_exclusions=exclusions)
    write(DEST / 'summary.json', summary); write(REVIEW / 'summary.json', summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']
    for phase, color in zip(phases, colors):
        row = next(x for x in r['correlations'] if x['phase'] == phase['phase']
                   and (x['clock'], x['max_age_ms']) == ('recv', 1000))
        use = [i for i, lag in enumerate(lags) if abs(lag) <= 1000]
        axes[0].plot([lags[i] for i in use], [row['correlations'][i] for i in use],
                     marker='o', color=color, label=f"{phase['phase'][0]}–{phase['phase'][1]} s elapsed")
    axes[0].axvline(0, color='gray', lw=.8); axes[0].axhline(0, color='gray', lw=.8)
    axes[0].set(xlabel='Lag (ms): negative = Predict first; positive = PM first',
                ylabel='Correlation of 100 ms midpoint changes', title='Observed receipt timing · 71 matched windows')
    axes[0].legend(fontsize=8)
    rows = [x for x in r['prediction'] if x['horizon_s'] == 1]
    axes[1].bar([x['day'][4:6]+'-'+x['day'][6:] for x in rows],
                [x['improvement_percent'] for x in rows], color='#0072B2')
    axes[1].axhline(0, color='gray', lw=.8)
    axes[1].set(xlabel='Chronological test date (2026; sampled windows)',
                ylabel='MSE reduction after adding PM history (%)', title='Predict 1 s ahead · controls: Predict + Binance')
    fig.suptitle('Exploratory sample, Oct 23–Nov 3 · no full-day intervals or trading-performance claim', fontsize=11)
    fig.savefig(REVIEW / 'comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
