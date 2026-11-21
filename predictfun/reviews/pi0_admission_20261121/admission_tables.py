"""PI0-ADMISSION-EXPOSURE-v1: the receipt's tables (formatting only; every number is read from an output). Copies the outputs into outputs/
(with SHA256SUMS) and writes tables.md. Usage (run_light): admission_tables.py"""
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_pi0_admission as M                             # noqa: E402

pd = M.pd


def esc(x):
    return str(x).replace('|', '\\|')


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v)).lower()
    if isinstance(v, (int, np.integer)):
        return f'{int(v):,}'
    if isinstance(v, float):
        return f'{v:,.{nd}f}'
    return esc(v)


def table(rows, cols, heads, nd=3):
    out = ['| ' + ' | '.join(esc(h) for h in heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for d in rows:
        out.append('| ' + ' | '.join(fmt(d.get(c), nd) for c in cols) + ' |')
    return '\n'.join(out)


def recs(df):
    return [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()} for r in df.to_dict('records')]


def main(out=M.OUT):
    out = Path(out); od = HERE / 'outputs'; od.mkdir(exist_ok=True)
    for n in ('generations.csv', 'stages.csv'):
        with open(out / n, 'rb') as f_in, gzip.GzipFile(od / (n + '.gz'), 'wb', mtime=0) as f_out:
            shutil.copyfileobj(f_in, f_out)
    for n in ('economics.csv', 'forecasts.csv', 'bins.csv', 'transitions.csv', 'crosstab.csv', 'retention_v2.csv', 'admission.json'):
        shutil.copy(out / n, od / n)
    (od / 'parts').mkdir(exist_ok=True)
    for p in sorted((out / 'parts').glob('part_*.json')):
        shutil.copy(p, od / 'parts' / p.name)
    sums = sorted(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(od)}' for p in od.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (od / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    js = json.loads((out / 'admission.json').read_text()); P = js['predicates']; man = json.loads((out / 'manifest.json').read_text()); T = {}
    rd = lambda n: pd.read_csv(out / n, float_precision='round_trip')
    T['Generation statuses (every selected generation)'] = table(js['statuses'], ['day', 'status', 'n'], ['day', 'status', 'generations'])
    ws = pd.Series([w['status'] for w in man['worlds']]).value_counts()
    T['Population (the frozen manifest)'] = table([dict(item='PI0 POST generations (census)', v=man['census']), dict(item='selected generations', v=len(man['selected'])),
                                                  dict(item='nonempty cells', v=len(man['cells'])), dict(item='sum of inclusion weights', v=float(sum(g['weight'] for g in man['selected'])))]
                                                 + [dict(item=f'worlds {k}', v=int(v)) for k, v in ws.items()], ['item', 'v'], ['item', 'value'], nd=1)
    ec = recs(rd('economics.csv'))
    T['Exposure: D_EXPOSURE = R(KEEP_SELECTED) − R(factual PI0) (c; inclusion-weighted and unweighted)'] = table(
        ec, ['day', 'side', 'n', 'worlds', 'weight', 'wmean_D', 'mean_D', 'sum_D', 'wmean_D_selected', 'wmean_D_focal_rest', 'wmean_D_opposite', 'n_pos', 'n_neg', 'n_zero', 'n_severe',
             'top5_share_unweighted', 'gain_forks', 'overrides_forks'],
        ['day', 'side', 'generations', 'worlds', 'Σ weight', 'weighted mean D', 'mean D', 'Σ D', 'weighted D_selected', 'weighted D_focal_rest', 'weighted D_opposite', 'D > 0', 'D < 0',
         'D = 0', 'severe (< −30)', 'top-5 |D| share', 'gain forks', 'with an override'], nd=2)
    T['The selected generation in each arm (inclusion-weighted)'] = table(
        ec, ['day', 'side', 'factual_wmean_lifetime', 'keep_wmean_lifetime', 'factual_fill_prob', 'keep_fill_prob', 'factual_wmean_filled', 'keep_wmean_filled', 'factual_wmean_end_value',
             'keep_wmean_end_value'],
        ['day', 'side', 'lifetime PI0 (ms)', 'lifetime KEEP (ms)', 'P(fill) PI0', 'P(fill) KEEP', 'filled PI0 (sh)', 'filled KEEP (sh)', 'END value PI0 (c)', 'END value KEEP (c)'], nd=3)
    fc = rd('forecasts.csv'); cols = ['stage', 'day', 'side', 'n', 'fill_rate', 'mean_P', 'brier', 'brier_pooled_rate', 'mse_V', 'mse_zero', 'mse_mech', 'mean_error', 'mae_V', 'mae_zero',
                                      'mean_V', 'mean_value', 'q_fill_error', 'qm_fill_error']
    heads = ['stage', 'day', 'side', 'n', 'fill rate', 'mean P', 'Brier(P)', 'Brier(pooled)', 'MSE V', 'MSE 0', 'MSE edge×E[q]', 'mean error', 'MAE V', 'MAE 0', 'mean V', 'mean value',
             'q|fill error', 'qm|fill error']
    T['Forecasts against matched neutral-hold targets, inclusion-weighted (c, c²; MAE secondary)'] = table(recs(fc[fc.weighted == True]), cols, heads, nd=3)      # noqa: E712
    T['Forecasts, unweighted'] = table(recs(fc[fc.weighted == False]), cols, heads, nd=3)                                                                       # noqa: E712
    T['Probability bins (inclusion-weighted)'] = table(recs(rd('bins.csv')), ['stage', 'p_lo', 'p_hi', 'n', 'wmean_P', 'wfill_rate'], ['stage', 'P from', 'P to', 'rows', 'mean P', 'fill rate'])
    s_ = P['review_selection']
    T['First-review selection: PULL against 500 matched draws (weights carried)'] = table(
        [s_], ['n', 'n_sel', 'mean_sel', 'mean_all', 'null_mean', 'null_p05', 'null_p95', 'percentile', 'strata', 'free_strata', 'free_rows', 'thin'],
        ['eligible', 'PULL', 'PULL mean value', 'all mean', 'null mean', 'null 5%', 'null 95%', 'percentile', 'strata', 'strata that can change', 'rows in them', 'thin'])
    tr = recs(rd('transitions.csv'))
    T['Entry → first review: input changes and new observations (both stages modeled)'] = table(
        tr, ['subset', 'n', 'share_market_flow_changed', 'share_fair_changed', 'share_order_state_changed', 'share_time_changed', 'share_new_pf_age', 'share_new_bn_age', 'share_new_bnd_age',
             'share_new_pf_receipt', 'wmean_dV', 'wmean_V_entry', 'wmean_V_review', 'flip_share'],
        ['subset', 'n', 'market_flow changed', 'fair changed', 'order_state changed', 'time changed', 'new PF frame (age)', 'new BN trade', 'new BN depth', 'new PF receipt', 'weighted dV',
         'weighted V entry', 'weighted V review', 'flip share'])
    T['Cross-tab: the 768 ENTRY-EXPOSURE forks by ENTRY-RETENTION shadow admission (c; unweighted; adaptive accounting)'] = table(
        recs(rd('crosstab.csv')), ['day', 'side', 'cls', 'n', 'worlds', 'mean_D', 'sum_D', 'mean_D_initial', 'mean_D_rest', 'top5_share_unweighted'],
        ['day', 'side', 'admission', 'forks', 'worlds', 'mean D_EXPOSURE', 'Σ D_EXPOSURE', 'mean D_initial', 'mean D_rest', 'top-5 |D| share'], nd=2)
    T['RETENTION v2: the forced forks\' forecasts by mean scores (c²; supersedes the MAE gate)'] = table(
        recs(rd('retention_v2.csv')), ['stage', 'day', 'side', 'n', 'mse_V', 'mse_zero', 'mse_mech', 'mean_error', 'brier', 'brier_pooled_rate', 'mae_V', 'mae_zero'],
        ['stage', 'day', 'side', 'n', 'MSE V', 'MSE 0', 'MSE edge×E[q]', 'mean error', 'Brier(P)', 'Brier(pooled)', 'MAE V', 'MAE 0'], nd=2)
    pr = [dict(item=f'day {d} weighted mean D', value=v['wmean_D']) for d, v in sorted(P['days'].items())] + [
        dict(item='gain forks', value=P['gain_forks']), dict(item='thin', value=P['thin'])] + [dict(item=f'reading {k}', value=v) for k, v in P['readings'].items()] + [
        dict(item=f'mse_beats_zero {k}', value=v) for k, v in P['mse_beats_zero'].items()] + [dict(item='review_discriminates', value=P['review_discriminates'])] + [
        dict(item=f'retention_v2 {k}', value=v) for k, v in P['retention_v2'].items()] + [dict(item='retention_v2 both-modeled flips', value=f"{js['retention_v2_flips']['flips']} of "
                                                                                                                                        f"{js['retention_v2_flips']['both_modeled']}")]
    T['Declared predicates and readings'] = table(pr, ['item', 'value'], ['item', 'value'], nd=4)
    md = [f'# PI0-ADMISSION-EXPOSURE-v1 tables (admission_tables.py; outputs as of {js["asof"]})', '']
    for k, t in T.items():
        md += [f'### {k}', '', t, '']
    (HERE / 'tables.md').write_text('\n'.join(md))
    print(f'tables.md: {len(T)} tables; outputs/: {len(sums)} files')


if __name__ == '__main__':
    main()
