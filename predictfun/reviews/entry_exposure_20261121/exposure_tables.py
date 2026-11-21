"""ENTRY-EXPOSURE-v1: the receipt's tables (formatting only; every number is read from an output or recomputed from forks.csv / the audit's
saved rows). Copies the outputs into outputs/ (with SHA256SUMS) and writes tables.md. Usage (run_light): exposure_tables.py"""
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent; RES = Path('/home/yuqing/ctaNew/predictfun/research'); sys.path.insert(0, str(RES))
import pf_entry_exposure as M                            # noqa: E402

pd = M.pd
AUDIT_ROWS = M.A.DATA / 'entry_target_audit_v1_20261121' / 'rows.csv.gz'


def esc(x):
    return str(x).replace('|', '\\|')


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v)).lower()
    if isinstance(v, (int, np.integer)):
        return f'{int(v):,}'
    if isinstance(v, float):
        return f'{v:,.{nd}f}'
    return esc(v)


def table(df, cols, heads, nd=2):
    out = ['| ' + ' | '.join(esc(h) for h in heads) + ' |', '|' + '|'.join('---' for _ in heads) + '|']
    for r in df.itertuples(index=False):
        d = r._asdict(); out.append('| ' + ' | '.join(fmt(d[c], nd) for c in cols) + ' |')
    return '\n'.join(out)


def offset_share(v, d, w=None):
    """The rowwise offset measure (review 3d8bf5ad item 2): sum w min(|v|, |d|) over rows where v and d have opposite signs / sum w |v|."""
    v, d = np.asarray(v, float), np.asarray(d, float); w = np.ones_like(v) if w is None else np.asarray(w, float)
    ov = np.where(v * d < 0, np.minimum(np.abs(v), np.abs(d)), 0.); den = float(w @ np.abs(v))
    return (float(w @ ov) / den) if den > 0 else None


def main(out=M.OUT):
    out = Path(out); od = HERE / 'outputs'; od.mkdir(exist_ok=True)
    with open(out / 'forks.csv', 'rb') as f_in, gzip.GzipFile(od / 'forks.csv.gz', 'wb', mtime=0) as f_out:
        shutil.copyfileobj(f_in, f_out)
    for n in ('summary.csv', 'reasons.csv', 'exposure.json'):
        shutil.copy(out / n, od / n)
    (od / 'parts').mkdir(exist_ok=True)
    for p in sorted((out / 'parts').glob('part_*.json')):
        shutil.copy(p, od / 'parts' / p.name)
    sums = sorted(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(od)}' for p in od.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (od / 'SHA256SUMS').write_text('\n'.join(sums) + '\n')
    F = pd.read_csv(out / 'forks.csv', float_precision='round_trip'); F['day'] = F.day.astype(str); S = pd.read_csv(out / 'summary.csv', float_precision='round_trip')
    W = pd.read_csv(out / 'reasons.csv'); js = json.loads((out / 'exposure.json').read_text()); P = js['predicates']; man = json.loads((out / 'manifest.json').read_text())
    S['day'] = S.day.astype(str); W['day'] = W.day.astype(str)
    ok = F[F.status == 'ok']; T = {}
    st = F.groupby(['day', 'status']).size().rename('n').reset_index()
    T['Statuses (every fork of the manifest)'] = table(st, ['day', 'status', 'n'], ['day', 'status', 'forks'])
    pop = pd.DataFrame([dict(set=k, worlds=len({(f['day'], f['m'], f['t_ms']) for f in v['forks']}), forks=len(v['forks']), evaluable=sum(f['status'] == 'ok' for f in v['forks']),
                             hours=len(v['population']), per_hour_population=int(np.median([p_['population'] for p_ in v['population']])),
                             per_hour_selected=int(np.median([p_['selected'] for p_ in v['population']]))) for k, v in man['sets'].items()])
    T['Population (the frozen manifest)'] = table(pop, list(pop.columns), ['set', 'worlds', 'forks', 'evaluable in the manifest', 'day-hours', 'plan worlds per hour (median)',
                                                                           'selected per hour (median)'])
    T['Contrasts per day and side (c; equal-world means over the evaluable forks)'] = table(
        S, ['day', 'side', 'forks', 'overrides_forks', 'mean_Y_PI0', 'mean_Y_KEEP', 'mean_D', 'sum_D', 'n_D_pos', 'n_D_neg', 'n_D_zero', 'worst_D', 'best_D', 'n_abs_D_gt30',
            'n_meaningful_Y_PI0', 'n_meaningful_Y_KEEP'],
        ['day', 'side', 'forks', 'forks with an override', 'mean Y_PI0', 'mean Y_KEEP_INITIAL', 'mean D_EXPOSURE', 'Σ D', 'D > 0', 'D < 0', 'D = 0', 'worst D', 'best D', '|D| > 30',
         'meaningful Y_PI0', 'meaningful Y_KEEP'])
    rows = []
    for (day, s), g in ok.groupby(['day', 'side']):
        rows.append(dict(day=day, side=int(s), mean_D_initial=float(g.D_initial.mean()), mean_D_focal_rest=float(g.D_focal_rest.mean()), mean_D_opposite=float(g.D_opposite.mean()),
                         mean_D_rest=float(g.D_rest.mean()), sum_D_initial=float(g.D_initial.sum()), sum_D_rest=float(g.D_rest.sum()),
                         offset=offset_share(g.D_initial, g.D_rest)))
    T['Components of D_EXPOSURE (c): the initial generation, the focal side\'s other generations, the opposite side'] = table(
        pd.DataFrame(rows), ['day', 'side', 'mean_D_initial', 'mean_D_focal_rest', 'mean_D_opposite', 'mean_D_rest', 'sum_D_initial', 'sum_D_rest', 'offset'],
        ['day', 'side', 'mean D_initial', 'mean D_focal_rest', 'mean D_opposite', 'mean D_rest', 'Σ D_initial', 'Σ D_rest', 'within-fork offset of |D_initial| by D_rest'], nd=3)
    ex = []
    for r in S.to_dict('records'):
        for b in ('PI0', 'KEEP'):
            ex.append(dict(day=r['day'], side=r['side'], branch=b, fill_prob=r[f'fill_prob_{b}'], mean_filled=r[f'mean_filled_{b}'], q_filled=r[f'q_filled_{b}'],
                           mean_active=r[f'mean_active_{b}'], mean_first_fill=r[f'mean_first_fill_{b}'], median_first_fill=r[f'median_first_fill_{b}'],
                           mean_vinit=r[f'mean_vinit_{b}'], unfilled_end=r[f'unfilled_end_{b}'], open_inv=r[f'open_inv_{b}'], fills=r[f'fills_{b}']))
    T['Execution of the initial generation per branch (shares, shares filled, ms, c)'] = table(
        pd.DataFrame(ex), ['day', 'side', 'branch', 'fill_prob', 'mean_filled', 'q_filled', 'mean_active', 'mean_first_fill', 'median_first_fill', 'mean_vinit', 'unfilled_end', 'open_inv', 'fills'],
        ['day', 'side', 'branch', 'P(initial fill)', 'mean filled (sh)', 'Σ filled (sh)', 'mean active (ms)', 'mean first fill (ms)', 'median first fill (ms)', 'mean V_initial (c)',
         'mean unfilled at END (sh)', 'mean |open inventory| at END (sh)', 'initial fills'], nd=3)
    T['Additional-fill forks (KEEP\'s initial generation filled more than PI0\'s)'] = table(
        S, ['day', 'side', 'forks', 'addl_fill_forks', 'mean_D_addl', 'mean_D_initial_addl', 'n_D_initial_neg_addl'],
        ['day', 'side', 'forks', 'additional-fill forks', 'mean D_EXPOSURE (c)', 'mean D_initial (c)', 'D_initial < 0'])
    mk = []
    for r in S.to_dict('records'):
        for b in ('PI0', 'KEEP'):
            mk.append(dict(day=r['day'], side=r['side'], branch=b, fills=r[f'fills_{b}'], q=r[f'q_filled_{b}'],
                           **{f'{h}': r[f'sum_{h}_{b}'] for h in ('m50', 'm100', 'm1000')}, **{f'{h}_n': r[f'{h}_fills_{b}'] for h in ('m50', 'm100', 'm1000')},
                           **{f'{h}_q': r[f'{h}_q_{b}'] for h in ('m50', 'm100', 'm1000')}))
    T['Initial-fill markouts against the venue mid (c; Σ over fills with the reference inside the episode; fills and shares as denominators)'] = table(
        pd.DataFrame(mk), ['day', 'side', 'branch', 'fills', 'q', 'm50', 'm50_n', 'm50_q', 'm100', 'm100_n', 'm100_q', 'm1000', 'm1000_n', 'm1000_q'],
        ['day', 'side', 'branch', 'initial fills', 'shares', 'Σ +50 ms', 'fills', 'shares', 'Σ +100 ms', 'fills', 'shares', 'Σ +1,000 ms', 'fills', 'shares'])
    T['Termination of the initial generation'] = table(W, ['day', 'side', 'branch', 'why', 'n', 'share'], ['day', 'side', 'branch', 'reason', 'forks', 'share'], nd=3)
    tm = []
    for (day, s), g in ok.groupby(['day', 'side']):
        for b in ('PI0', 'KEEP'):
            rq = g[g[f'{b}_req'].notna()]
            tm.append(dict(day=day, side=int(s), branch=b, requested=int(len(rq)), mean_req=float((rq[f'{b}_req'] - rq[f'{b}_act']).mean()) if len(rq) else None,
                           mean_effect=float((rq[f'{b}_remove'] - rq[f'{b}_act']).mean()) if len(rq) else None, ahead_submit=float(g.ahead_at_submit.mean()),
                           ahead_final=float(g[f'{b}_ahead_final'].mean())))
    T['Removal timing and queue-ahead of the initial generation (ms after activation; shares; venue state, a diagnostic)'] = table(
        pd.DataFrame(tm), ['day', 'side', 'branch', 'requested', 'mean_req', 'mean_effect', 'ahead_submit', 'ahead_final'],
        ['day', 'side', 'branch', 'removal requested', 'mean request − activation', 'mean effect − activation', 'mean queue ahead at activation', 'mean queue ahead at the end'], nd=1)
    pr = pd.DataFrame([dict(day=d, **v) for d, v in P['days'].items()])
    T['Declared predicates per day (both sides pooled)'] = table(pr, ['day', 'forks', 'mean_D', 'sign', 'mean_D_initial', 'mean_D_rest', 'fill_prob_PI0', 'fill_prob_KEEP'],
                                                              ['day', 'forks', 'mean D_EXPOSURE (c)', 'sign', 'mean D_initial', 'mean D_rest', 'P(initial fill) PI0', 'P(initial fill) KEEP'], nd=4)
    pooled = pd.DataFrame([{k: P[k] for k in ('diagnosis', 'forks', 'additional_fill_forks', 'fill_prob_PI0', 'fill_prob_KEEP', 'more_execution', 'sum_D_initial', 'sum_D_rest',
                                              'initial_carries', 'mean_D_initial_additional', 'share_D_initial_neg_additional', 'extra_fills_harmful')}])
    T['Declared predicates, pooled'] = table(pooled, list(pooled.columns), ['diagnosis', 'forks', 'additional-fill forks', 'P(fill) PI0', 'P(fill) KEEP', 'more_execution',
                                                                            'Σ D_initial', 'Σ D_rest', 'initial_carries', 'mean D_initial (additional)', 'share D_initial < 0 (additional)',
                                                                            'extra_fills_harmful'], nd=4)
    A = pd.read_csv(AUDIT_ROWS, float_precision='round_trip'); A = A[A.status == 'ok']; ar = []
    for k_, g in A.groupby('set'):
        ar.append(dict(set=k_, rows=int(len(g)), weighted=offset_share(g.v_initial, g.d_following, g.w), unweighted=offset_share(g.v_initial, g.d_following)))
    T['Review item 2: the audit\'s rowwise offset of V_initial by D_following (recomputed from the audit\'s saved ok rows)'] = table(
        pd.DataFrame(ar), ['set', 'rows', 'weighted', 'unweighted'], ['set', 'ok rows', 'offset share (sampling-weighted)', 'offset share (unweighted rows)'], nd=4)
    md = [f'# ENTRY-EXPOSURE-v1 tables (exposure_tables.py; outputs as of {js["asof"]})', '']
    for k, t in T.items():
        md += [f'### {k}', '', t, '']
    (HERE / 'tables.md').write_text('\n'.join(md))
    print(f'tables.md: {len(T)} tables; outputs/: {len(sums)} files')


if __name__ == '__main__':
    main()
