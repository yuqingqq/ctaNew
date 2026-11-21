"""PI0-MOVEMENT-CALIBRATION-AUDIT-v1: the receipt's tables (formatting only; every number is read from an output). Copies the outputs into
outputs/ (large tables gzipped exactly, with SHA256SUMS) and writes tables.md. Usage (run_light): movement_tables.py"""
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_movement_audit as M                            # noqa: E402

pd = M.pd
STR = {c: str for c in ('day', 'key', 'cell', 'side', 'admission', 'view', 'stage', 'cohort', 'subset', 'dim', 'kind', 'quantity', 'weighting', 'acc_status',
                        'hbin', 'qbin', 'obin', 'venue_qbin', 'pbin', 'mbin', 'eqbin', 'vbin', 'delay_class', 'split_status', 'ref_status')}
BIG = ('opportunities', 'fills', 'transitions')
SMALL = ('moments', 'strata', 'prepost', 'concentration', 'contributions', 'sources', 'evidence')


def esc(x):
    return str(x).replace('|', '\\|')


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v)).lower()
    if isinstance(v, (int, np.integer)):
        return f'{int(v):,}'
    if isinstance(v, (float, np.floating)):
        return f'{float(v):,.{nd}f}'
    return esc(v)


def table(df, cols, heads=None, nd=3):
    heads = heads or cols
    out = ['| ' + ' | '.join(esc(h) for h in heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for d in df.to_dict('records'):
        out.append('| ' + ' | '.join(fmt(d.get(c), nd) for c in cols) + ' |')
    return '\n'.join(out)


def main():
    src = M.OUT; dst = HERE / 'outputs'; dst.mkdir(exist_ok=True); sums = []
    for n in SMALL:
        shutil.copy(src / f'{n}.csv', dst / f'{n}.csv')
    for n in BIG:
        b = (src / f'{n}.csv').read_bytes(); g = gzip.compress(b, mtime=0); (dst / f'{n}.csv.gz').write_bytes(g)
        if gzip.decompress((dst / f'{n}.csv.gz').read_bytes()) != b:
            raise SystemExit(f'{n}: the gzip copy does not decompress to the output')
    shutil.copy(src / 'audit.json', dst / 'audit.json'); (dst / 'parts').mkdir(exist_ok=True)
    for p in sorted((src / 'parts').glob('mids_*.json')):
        if not p.name.endswith('_status.json'):
            shutil.copy(p, dst / 'parts' / p.name)
    for p in sorted(dst.rglob('*')):
        if p.is_file() and p.name != 'SHA256SUMS':
            sums.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dst)}')
    (dst / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    rd = lambda n: pd.read_csv(src / f'{n}.csv', float_precision='round_trip', dtype=STR)
    MO, ST, PP, CO, CT, SO, EV = (rd(n) for n in SMALL); au = json.loads((src / 'audit.json').read_text()); O = rd('opportunities'); F = rd('fills')
    L = ['# PI0-MOVEMENT-CALIBRATION-AUDIT-v1 — tables', '', f"Generated from `{src}` (audit.json asof {au['asof']}). Units: c per opportunity (unconditional), "
         'c per filled opportunity (c_m, c_q in shares), c per share (pre / post), c² (MSE). Admitted rows inclusion-weighted; forced rows unweighted. '
         'No interval; related generations on consumed days.', '']
    L += ['## Statuses', '', table(pd.DataFrame(au['statuses']), ['cohort', 'stage', 'acc_status', 'n']), '',
          table(pd.DataFrame(au['fill_statuses']), ['cohort', 'ref_status', 'n']), '']
    mem = au['membership']; ctl = au['control']
    L += ['## Membership and the retained first-review control', '', table(pd.DataFrame([mem]), list(mem)), '',
          table(pd.DataFrame([ctl]), ['n', 'n_sel', 'mean_sel', 'mean_all', 'null_mean', 'null_p05', 'null_p95', 'percentile', 'strata', 'free_strata', 'free_rows',
                                      'reviews_with_time', 'cell_equal_admission']), '']
    A = MO[(MO.cohort == 'admitted') & (MO.view == 'all')]
    L += ['## Admitted: moments and the exact decomposition (all rows of each stage)', '',
          table(A, ['stage', 'n', 'weight', 'ess', 'worlds', 'fills', 'same_as_review', 'fill_rate', 'mean_P', 'brier', 'brier_pooled']), '',
          table(A, ['stage', 'mean_V', 'mean_value', 'bias_V', 'qty_term', 'mov_term', 'mov_prob', 'mov_cond', 'mse_V', 'mse_zero', 'mse_mech', 'mae_V', 'mae_zero'], nd=3), '',
          table(A, ['stage', 'mean_Eq', 'mean_q', 'bias_q', 'q_prob', 'q_cond', 'c_q', 'mean_Eqm', 'mean_M', 'bias_m', 'mse_m', 'mse_m_zero', 'c_m', 'mean_mhat_fill', 'mean_M_fill']), '',
          table(A, ['stage', 'severe_n', 'severe_sum_value', 'sum_value', 'severe_sum_err_V', 'sum_err_V', 'severe_sum_err_m', 'sum_err_m'], nd=1), '']
    for nm in M.STAGES:
        g = MO[(MO.cohort == 'admitted') & (MO.stage == nm) & MO.view.isin(['day', 'side', 'day_side'])]
        L += [f'### Admitted {nm}: days and sides', '', table(g, ['view', 'day', 'side', 'n', 'weight', 'fills', 'fill_rate', 'mean_P', 'bias_V', 'mov_prob', 'mov_cond',
                                                                  'bias_m', 'sum_err_m', 'c_m', 'mse_m', 'mse_m_zero', 'severe_n']), '']
    for nm in M.STAGES:
        g = MO[(MO.cohort == 'forced') & (MO.stage == nm)]
        L += [f'## Forced {nm} (unweighted; Oct 24-27, days, the Oct 26-27 overlap, shadow admission)', '',
              table(g, ['view', 'day', 'side', 'admission', 'n', 'fills', 'fill_rate', 'mean_P', 'brier', 'brier_pooled', 'bias_V', 'qty_term', 'mov_prob', 'mov_cond',
                        'c_m', 'mse_V', 'mse_zero', 'mse_m', 'mse_m_zero']), '']
    L += ['## Calibration and strata', '', 'kind: decision_state (client inputs at the decision), forecast (the frozen moments), outcome (the realized first-fill delay; '
          'never a feature), explanatory_venue (venue knowledge; never read by a predicate). thin: < 30 opportunities; thin_conditional: < 10 fills.', '']
    for co, sub in (('admitted', 'all'), ('forced', 'overlap'), ('forced', 'all')):
        for nm in ('entry', 'first_review'):
            g = ST[(ST.cohort == co) & (ST.subset == sub) & (ST.stage == nm)]
            L += [f'### {co} ({sub}) {nm}', '', table(g, ['dim', 'kind', 'cell', 'n', 'fills', 'mean_P', 'fill_rate', 'mean_Eq', 'mean_q', 'mean_Eqm', 'mean_M', 'mean_V',
                                                           'mean_value', 'mean_u', 'c_m', 'split_n', 'mean_pre', 'mean_post', 'thin', 'thin_conditional']), '']
    L += ['## Pre / post-fill split per counted fill (c per share; ages ms)', '',
          table(PP, ['cohort', 'stage', 'view', 'cell', 'side', 'fills', 'with_mid', 'missing', 'q', 'q_with_mid', 'total_per_share', 'pre_per_share', 'post_per_share',
                     'sum_total', 'sum_pre', 'sum_post', 'median_age', 'max_age']), '']
    L += ['## Concentration (top-5 shares of |w x|; world sums signed first)', '', table(CO, ['quantity', 'weighting', 'n', 'worlds', 'top5_generation_share',
                                                                                             'top5_world_share', 'sum']), '']
    L += ['## Signed day x side contributions (admitted)', '', table(CT, ['day', 'side', 'n', 'weighted_sum_D', 'share_D', 'entry_n', 'weighted_sum_entry_u', 'share_entry_u',
                                                                         'weighted_sum_entry_err_V'], nd=2), '']
    L += ['## Observation sources, entry -> first review (ms ages; bookTicker, not trades)', '', table(SO, ['cohort', 'subset', 'n', 'weight', 'new_pf_frame', 'new_bookticker',
                                                                                                         'new_depth', 'new_pf_receipt', 'any_new', 'mean_mid_change',
                                                                                                         'mean_abs_mid_change']), '']
    L += ['## Evidence (computed by predicates(); the declared reading is the admitted entry)', '', table(EV, ['stage', 'question', 'answer', 'detail']), '']
    (HERE / 'tables.md').write_text('\n'.join(L) + '\n')
    print(f'tables.md: {len(L)} lines; outputs: {len(sums)} files; opportunities {len(O)}, fills {len(F)}')


if __name__ == '__main__':
    main()
