"""ENTRY-RETENTION-DISCRIMINATION-v1: the receipt's tables (formatting only; every number is read from an output or recomputed from the
partitions). Copies the outputs into outputs/ (with SHA256SUMS) and writes tables.md. Usage (run_light): retention_tables.py"""
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_entry_retention as M                           # noqa: E402

pd, LS = M.pd, M.LS


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
    with open(out / 'stages.csv', 'rb') as f_in, gzip.GzipFile(od / 'stages.csv.gz', 'wb', mtime=0) as f_out:
        shutil.copyfileobj(f_in, f_out)
    for n in ('calibration.csv', 'bins.csv', 'decisions.csv', 'changes.csv', 'retention.json'):
        shutil.copy(out / n, od / n)
    (od / 'parts').mkdir(exist_ok=True)
    for p in sorted((out / 'parts').glob('part_*.json')):
        shutil.copy(p, od / 'parts' / p.name)
    sums = sorted(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(od)}' for p in od.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (od / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    js = json.loads((out / 'retention.json').read_text()); P = js['predicates']; S = pd.read_csv(out / 'stages.csv', float_precision='round_trip'); S['day'] = S.day.astype(str)
    cal = pd.read_csv(out / 'calibration.csv', float_precision='round_trip'); bn = pd.read_csv(out / 'bins.csv', float_precision='round_trip')
    dc = pd.read_csv(out / 'decisions.csv', float_precision='round_trip'); ch = pd.read_csv(out / 'changes.csv', float_precision='round_trip'); T = {}
    rows = [r for k in range(12) for r in LS.gz_rows((out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes())]
    T['Fork statuses (every manifest fork)'] = table(js['fork_statuses'], ['day', 'status', 'n'], ['day', 'status', 'forks'])
    T['Stage statuses'] = table(js['stage_statuses'], ['stage', 'fork_status', 'stage_status', 'n'], ['stage', 'fork status', 'stage status', 'rows'])
    E = S[(S.stage == 'entry') & (S.fork_status == 'ok')]; ad = []
    for (day, side), g in E.groupby(['day', 'side']):
        ad.append(dict(day=day, side=int(side), forks=len(g), would_POST=int((g.admission == 'would_POST').sum()), would_WAIT=int((g.admission == 'would_WAIT').sum()),
                       model=int((g.dec_status == 'model').sum()), other=int((g.dec_status != 'model').sum())))
    T['Shadow entry admission (PI0 unforced, on the exact absent fork state)'] = table(ad, ['day', 'side', 'forks', 'would_POST', 'would_WAIT', 'model', 'other'],
                                                                                       ['day', 'side', 'evaluable forks', 'would_POST', 'would_WAIT', 'model decisions', 'other statuses'])
    cc = ['stage', 'day', 'side', 'n', 'filled', 'fill_rate', 'mean_P', 'brier', 'brier_pooled_rate', 'mean_Eq', 'mean_q', 'mae_q', 'mean_V', 'mean_value', 'mae_V', 'mae_zero', 'mae_mech',
          'mae_q_fill', 'mae_qm_fill']
    T['Forecast calibration against the matched neutral-hold targets (c, shares; eligible model rows)'] = table(
        recs(cal), cc, ['stage', 'day', 'side', 'n', 'filled', 'fill rate', 'mean P', 'Brier(P)', 'Brier(pooled rate)', 'mean E[q]', 'mean q', 'MAE q', 'mean V', 'mean value',
                        'MAE V', 'MAE 0', 'MAE edge×E[q]', 'MAE q|fill', 'MAE qm|fill'])
    T['Probability bins'] = table(recs(bn), ['stage', 'p_lo', 'p_hi', 'n', 'mean_P', 'fill_rate'], ['stage', 'P from', 'P to', 'rows', 'mean P', 'fill rate'])
    T['Realized value by forecast sign and shadow entry admission (c; unweighted rows)'] = table(
        recs(dc), ['stage', 'vsign', 'admission', 'n', 'worlds', 'n_benefit', 'sum_benefit', 'n_loss', 'sum_loss', 'n_zero', 'n_severe', 'top5_share_unweighted'],
        ['stage', 'V sign', 'admission', 'rows', 'worlds', 'benefit', 'Σ benefit', 'loss', 'Σ loss', 'zero', 'severe (< −30)', 'top-5 |value| share'], nd=2)
    sl = [dict(stage=k, **v) for k, v in P['selection'].items()]
    T['Selection against 500 matched random draws (within day × side × UTC hour)'] = table(
        sl, ['stage', 'selected_action', 'n', 'n_sel', 'mean_sel', 'mean_all', 'null_mean', 'null_p05', 'null_p95', 'percentile', 'strata', 'degenerate_strata', 'thin'],
        ['stage', 'selected', 'eligible', 'selected rows', 'selected mean value', 'all mean', 'null mean', 'null 5%', 'null 95%', 'percentile', 'strata', 'degenerate', 'thin'])
    T['Entry → first review: changed inputs and moment terms'] = table(
        recs(ch), ['day', 'side', 'forks', 'share_market_flow_changed', 'share_fair_changed', 'share_order_state_changed', 'share_time_changed', 'mean_dV', 'mean_V_entry', 'mean_V_review',
                   'flip_share', 'stable_market_forks', 'stable_market_flip_share'],
        ['day', 'side', 'forks', 'market_flow changed', 'fair changed', 'order_state changed', 'time changed', 'mean dV', 'mean V entry', 'mean V review', 'flip share',
         'unchanged-market forks', 'their flip share'])
    tm = []
    for nm in ('first_review', 'first_pull'):
        g = S[(S.stage == nm) & (S.fork_status == 'ok')]; ok = g[g.stage_status == 'ok']
        tm.append(dict(stage=nm, forks=len(g), with_stage=len(ok), mean_t=float(ok.t.mean()) if len(ok) else None, median_t=float(ok.t.median()) if len(ok) else None,
                       max_t=float(ok.t.max()) if len(ok) else None, same_as_review=int(ok.same_as_review.fillna(False).astype(bool).sum()) if nm == 'first_pull' else None,
                       pulls=int((ok.action == 'PULL').sum())))
    T['Stage timing (ms from the fork; the POST at 0, activation at 58)'] = table(tm, ['stage', 'forks', 'with_stage', 'mean_t', 'median_t', 'max_t', 'pulls', 'same_as_review'],
                                                                                  ['stage', 'evaluable forks', 'with the stage', 'mean t', 'median t', 'max t', 'PULL', 'same as review'], nd=1)
    sw = pd.Series([r.get('stop_why') for r in rows if r['status'] == 'ok']).value_counts()
    T['Prefix stop reasons'] = table([dict(why=k, n=int(v)) for k, v in sw.items()], ['why', 'n'], ['stop reason', 'forks'])
    fe = []
    for (day, side), g in pd.DataFrame([dict(day=str(r['day']), side=int(r['side']), **f) for r in rows if r['status'] == 'ok' for f in r['fills']]).groupby(['day', 'side']):
        av = g[g.fill_time_edge.notna()]
        fe.append(dict(day=day, side=int(side), fills=len(g), q=float(g.q.sum()), end_value=float(g.end_value.sum()), edge_available=len(av), edge_q=float(av.q.sum()),
                       fill_time_edge=float(av.fill_time_edge.sum()), end_value_available=float(av.end_value.sum())))
    T['Held initial fills: fill-time execution edge against the END value (c)'] = table(
        fe, ['day', 'side', 'fills', 'q', 'end_value', 'edge_available', 'edge_q', 'fill_time_edge', 'end_value_available'],
        ['day', 'side', 'fills', 'shares', 'Σ END value', 'fills with a mid', 'their shares', 'Σ fill-time edge', 'Σ END value (same fills)'], nd=2)
    pr = [dict(item=k, value=v) for k, v in (('discriminates entry', P['discriminates']['entry']), ('discriminates first_review', P['discriminates']['first_review']),
                                            ('value_fails entry', P['value_fails']['entry']), ('value_fails first_review', P['value_fails']['first_review']),
                                            ('flip_share', P['flip_share']))] + [dict(item=f'reading {k}', value=v) for k, v in P['readings'].items()]
    T['Declared predicates and readings'] = table(pr, ['item', 'value'], ['item', 'value'], nd=4)
    md = [f'# ENTRY-RETENTION-DISCRIMINATION-v1 tables (retention_tables.py; outputs as of {js["asof"]})', '']
    for k, t in T.items():
        md += [f'### {k}', '', t, '']
    (HERE / 'tables.md').write_text('\n'.join(md))
    print(f'tables.md: {len(T)} tables; outputs/: {len(sums)} files')


if __name__ == '__main__':
    main()
