"""Bounded review of pinned v2: no fitting or raw tape, saved values and falsifiers."""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import contextlib
import gc
import hashlib
import io
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/yuqing/ctaNew')
SCOPE = json.loads((HERE/'scope.json').read_text())
TREE = Path(SCOPE['isolated_checkout'])
sys.path.insert(0, str(TREE/'predictfun/research'))
import numpy as np
import pandas as pd
import pf_pipeline_export as EX
import pf_pipeline_bench as B
import pf_pipeline_policy as P

KP, SC, A = P.KP, P.SC, P.A
DATA = ROOT/'data/predict_fun/research'
ARCHIVE = TREE/'predictfun/reviews/pipeline_one_day_v2_20261119'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def close(a, b, tol=1e-6):
    assert abs(float(a)-float(b)) <= tol, (a, b)


def book(nxt):
    return dict(kf=np.array([-10., nxt]), tg=np.arange(0., 201., 50.),
                P5=np.array([[50., 49., 48., 47., 46.]]*2), S5=np.array([[12., 10., 10., 10., 10.]]*2))


def block(w):
    empty = (np.array([]), np.array([]), np.array([], int))
    fills = {clock: {} for clock, _ in EX.CLOCKS}
    o = SimpleNamespace(side=0, t=0., p=50., oid=0, m=1, phase=1, mark=51., status='eligible', g_struct=np.nan,
                        q=0., v=0., t_full=np.nan, q_m0=0., v_m0=0., t_full_m0=np.nan, q_m250=0., v_m250=0., t_full_m250=np.nan, f00=51., f02=51.)
    x = np.zeros(len(EX.FC.MARKET), np.float32); x[EX.FC.MARKET.index('pf_mid')] = 51.
    feats = {0: {0: (x, 30.), 2: (x, 30.)}}; errors = []
    out = EX.order_block('20261023', o, np.array([2]), np.array([0., 100.]), np.array([0, 2]), np.array([True, True]), True,
                         w, feats, fills, empty, {p: {} for p in KP.POLICIES}, {p: [0., 0] for p in KP.POLICIES}, errors)
    assert not errors
    return out.iloc[0][EX.INPUTS].to_numpy(float)


def synthetic():
    same = np.array_equal(block(book(30.)), block(book(80.)), equal_nan=True)
    changed = book(30.); changed['S5'][0, 0] = 18.
    assert same and not np.array_equal(block(book(30.)), block(changed), equal_nan=True)
    EX.V1_POST_STATE = True
    try: assert not np.array_equal(block(book(30.)), block(book(80.)), equal_nan=True)
    finally: EX.V1_POST_STATE = False
    common = dict(day='20261027', kind='retain', j=2, side=0, t=900., decision_ms=1000., pf_mid=50., p=50., fp_remaining=30., J_P=.5)
    good = dict(common, oid=0, J_Eq=5., J_Eqm=-5., J_V=-5., J_ok=True)
    bad = dict(common, oid=1, J_Eq=1., J_Eqm=-200., J_V=-200., J_ok=False)
    mix = pd.DataFrame([good, bad]); assert P.contract_parity(mix, 0., n=10) == (2, 1, 2)
    assert len(P.triggers_of(mix)) == 1
    wrong = mix.copy(); wrong.loc[1, 'J_ok'] = True
    assert P.contract_parity(wrong, 0., n=10)[2] < 2
    val = np.linspace(-10., -1., 20); df = pd.DataFrame(dict(q_m50=np.ones(20), qm_m50=val, value_m50=val, severe_m50=np.zeros(20)))
    pr = dict(P=np.ones(20), Eq=np.ones(20), Eqm=val+20., V=val+20., ok=np.ones(20, bool))
    e = B.econ_one('retain', df, pr)
    assert e['E1 VALUE LEVEL'] == 'FAIL' and e['E2 BOUNDARY SIGN'] != 'PASS'
    df2 = pd.DataFrame(dict(value_m50=[-4., -3., 1., 2.]))
    good_e = B.econ_one('retain', df2, dict(V=df2.value_m50.values, ok=np.ones(4, bool)))
    assert all(good_e[k] == 'PASS' for k in ('E1 VALUE LEVEL', 'E2 BOUNDARY SIGN', 'E3 LOSS AMOUNT'))
    # A residual edge case: clipping quantity to zero must also make q*move zero.
    class Const:
        def __init__(self, v): self.v = v
        def predict(self, x, **kw): return np.full(len(x), self.v)
    f = pd.DataFrame(0., index=[0], columns=B.SETS['J']).assign(side=0, pf_mid=50., p=50., fp_remaining=30.)
    z = B.predict({'p': Const(.5), 'q': Const(-1.), 'qm': Const(-200.)}, f, 'J')
    assert z['Eq'][0] == 0 and z['Eqm'][0] == -100 and bool(z['ok'][0])
    zero = pd.DataFrame([dict(common, oid=2, J_Eq=0., J_Eqm=-100., J_V=-100., J_ok=True)])
    zp = P.contract_parity(zero, 0., n=10)
    assert zp == (1, 0, 0) and len(P.triggers_of(zero)) == 1
    return dict(prior_POST_case_fixed=True, prior_mixed_refusal_case_fixed=True, prior_shifted_value_case_fixed=True,
                positive_and_known_bad_controls=True, zero_quantity_nonzero_moment=dict(accepted_by_benchmark=True, Eq=0., Eqm=-100.,
                parity=list(zp), rule='pull', contract='keep'))


def saved_day(day, bench):
    bdir = DATA/'pipeline_bench_v2_20261119'; pdir = DATA/'pipeline_policy_v2_20261119'; edir = DATA/'pipeline_export_v2_20261119'/day
    pj = json.loads((pdir/f'policy_{day}.json').read_text()); exp = json.loads((edir/'receipt.json').read_text())
    assert sha(pdir/f'policy_{day}.json') == sha(ARCHIVE/f'policy_{day}.json')
    assert pj['source_sha256'] == sha(TREE/'predictfun/research/pf_pipeline_policy.py')
    for name, h in pj['outputs'].items(): assert sha(pdir/name) == h and sha(ARCHIVE/name) == h
    assert pj['inputs']['bench'] == sha(bdir/'bench.json')
    predfile = bdir/f'pred_{day}.parquet'
    assert sha(predfile) == bench['outputs'][predfile.name] == pj['inputs']['predictions']
    assert sha(edir/'receipt.json') == pj['inputs']['export_receipt'] == bench['exports'][day]
    assert exp['source_sha256'] == sha(TREE/'predictfun/research/pf_pipeline_export.py')
    assert all(sha(edir/n) == h for n, h in exp['outputs'].items())
    pred = pd.read_parquet(predfile)
    assert not pred.duplicated(['day', 'oid', 'kind', 'j']).any()
    assert set(pred.day) == {day}
    r = pred[pred.kind == 'retain'].copy(); post = pred[pred.kind == 'post'].copy()
    zero = {k: int(((g.J_Eq == 0) & (g.J_Eqm != 0) & g.J_ok).sum()) for k,g in pred.groupby('kind')}
    diagnostics = {}
    for kind, g in pred.groupby('kind'):
        diagnostics[kind] = {}
        for side, h in g.groupby('side'):
            pr = {name[2:]:h[name].values for name in h if name.startswith('J_')}
            ec = B.econ_one(kind, h, pr)
            for key, val in ec.items():
                ref = bench['economic'][kind][str(side)][day][key]
                if isinstance(val, (int, float)): close(val, ref)
                else: assert val == ref
            h2 = h.assign(severe_m50=np.maximum(-h.value_m50-30., 0.))
            read = B.read_one(h2, pr, h.F_V.values)
            ref = bench['readings'][kind][str(side)][day]['J']
            for key in ('amount_ratio', 'move_ratio', 'move_rank', 'fill_auc', 'value_rank', 'fair_edge_rank', 'severe_ratio', 'n'): close(read[key], ref[key])
            diagnostics[kind][str(side)] = dict(ec, value_rank=read['value_rank'], fair_rank=read['fair_edge_rank'], fill_auc=read['fill_auc'])
    hit = r[r.J_ok & (r.J_V < 0)].groupby('oid').j.min()
    assert hit.equals(P.triggers_of(pred).droplevel('day'))
    assert len(hit) == pj['pulled_orders']
    # Diagnostic at states the first-trigger policy actually reaches, distinct
    # from the neutral baseline's repeated hypothetical decisions.
    r['first_pull'] = r.oid.map(hit).fillna(99)
    reachable = r[r.j <= r.first_pull]
    reached = {}
    for side, g in reachable.groupby('side'):
        kept = g[g.J_ok & (g.J_V >= 0)]
        reached[str(side)] = dict(rows=len(g), kept_rows=len(kept), kept_orders=int(kept.oid.nunique()), kept_value_mean=float(kept.value_m50.mean()))
    pcsv = pd.read_csv(pdir/f'policy_{day}.csv')
    for row in pcsv.itertuples():
        v = pj['values'][row.rule][row.scenario]
        for a,b in ((row.V,v['KEEP']+v['score']), (row.PULLALL_value,v['KEEP']+v['PULLALL']), (row.PULLALL_improvement,v['PULLALL']), (row.E,v['E']), (row.pulls,v['pulls'])): close(a,b)
    # Hash-verified saved engine inputs only. No scenario/tape resimulation.
    kp = DATA/'keep_pull_20261117'/day; kr = json.loads((kp/'receipt.json').read_text())
    assert all(sha(kp/n) == h for n,h in kr['outputs'].items())
    assert pj['inputs']['days'][day]['keep_pull_receipt'] == sha(kp/'receipt.json')
    O = pd.read_parquet(kp/'orders.parquet'); E = O[O.status == 'eligible'].copy().set_index('oid')
    fl = pd.read_parquet(kp/'fills.parquet'); fl = fl[fl.clock == 50.].copy()
    fl['val'] = fl.q * np.where(fl.oid.map(E.side) == 0, 1., -1.) * (fl.oid.map(E.mark)-fl.oid.map(E.p))
    assert fl.val.notna().all()
    ej = pd.Series(A.endj_of(E.t_full.values, E.g_struct.values), index=E.index)
    T = pd.Series(E.index.map(hit).fillna(99).values, index=E.index)
    act = (T >= KP.J0) & (T <= KP.NSTEP) & (T < ej)
    removed = fl.oid.map(act).values & (fl.t.values >= KP.landing(fl.oid.map(T).values))
    pa_removed = (KP.J0 < fl.oid.map(ej).values) & (fl.t.values >= KP.landing(KP.J0))
    fl['model_value'] = np.where(removed, 0., fl.val); fl['pull_all_value'] = np.where(pa_removed, 0., fl.val)
    base = pj['values']['MODEL-J'][P.B_NAME]
    close(fl.val.sum(),base['KEEP']); close(fl.model_value.sum(),base['KEEP']+base['score'])
    close(fl.pull_all_value.sum(),base['KEEP']+base['PULLALL'])
    close(act.sum(),base['pulls'])
    per = fl.groupby('oid')[['val','model_value','pull_all_value']].sum().reindex(E.index,fill_value=0.)
    per['delta'] = per.model_value-per.pull_all_value
    per['category'] = np.where(act & (T == KP.J0), 'first_step', np.where(act, 'later_pull', 'never_model_pull'))
    split = per.groupby('category').agg(orders=('delta','size'), model_c=('model_value','sum'), pull_all_c=('pull_all_value','sum'), delta_c=('delta','sum')).to_dict('index')
    close(sum(x['delta_c'] for x in split.values()),base['score']-base['PULLALL'])
    assert abs(split.get('first_step',{}).get('delta_c',0.)) < 1e-6
    pre = fl.t < KP.STEP*KP.J0; pending = (fl.t >= KP.STEP*KP.J0) & (fl.t < KP.landing(KP.J0))
    spans = dict(before_first_decision_c=float(fl.loc[pre,'model_value'].sum()), first_cancel_latency_c=float(fl.loc[pending,'model_value'].sum()),
                 after_first_possible_cancel_c=float(fl.loc[~(pre|pending),'model_value'].sum()))
    close(sum(spans.values()),base['KEEP']+base['score'])
    positive = post[post.J_ok & (post.J_V > 0)]
    # Accounting-only conditional read; not a new trained strategy or selected threshold.
    observed = positive.set_index('oid').join(per[['model_value','pull_all_value']],how='left').fillna({'model_value':0.,'pull_all_value':0.})
    selection = dict(rows=len(post), coherent_positive=len(positive), positive_share=len(positive)/len(post),
                     realized_no_model_pull_c=float(positive.value_m50.sum()), realized_no_model_pull_mean_c=float(positive.value_m50.mean()),
                     same_orders_with_MODEL_J_c=float(observed.model_value.sum()), same_orders_with_PULL_ALL_c=float(observed.pull_all_value.sum()))
    m = pcsv[pcsv.rule == 'MODEL-J']; aa = pcsv[pcsv.rule == 'A+0'].set_index('scenario')
    summary = dict(KEEP_c=base['KEEP'], MODEL_J_c=base['KEEP']+base['score'], PULL_ALL_c=base['KEEP']+base['PULLALL'],
                   A0_c=float(aa.loc[P.B_NAME,'V']), delta_pull_all_c=base['score']-base['PULLALL'],
                   scenarios_above_pull_all=int((m.V > m.PULLALL_value).sum()), scenarios_above_A0=int((m.V.values > aa.loc[m.scenario,'V'].values).sum()),
                   total_eligible=len(E), predicted_orders=int(r.oid.nunique()), pulled_orders=len(hit), first_step_pulls=int((hit==KP.J0).sum()))
    print(json.dumps(dict(stage=day, baseline=summary, attribution=split, reachable=reached, zero_moment_bad=zero)), flush=True)
    return dict(asof=pj['asof'], benchmark_asof=bench['asof'], export_asof=exp['asof'], original_orders=len(O), statuses=O.status.value_counts().to_dict(),
                baseline=summary, economic=diagnostics, reachable_decisions=reached, attribution=split, timing_value=spans, post_selection=selection,
                accepted_zero_quantity_nonzero_moment=zero, prediction_hash=sha(predfile), prediction_rows=len(pred), verified_policy_rows=len(pcsv),
                source_engine_asof=kr['asof'])


def main():
    out = dict(target_commit=SCOPE['target_commit'], asof=datetime.now(timezone.utc).isoformat())
    for line in (ARCHIVE/'sources.sha256').read_text().splitlines():
        h,n = line.split(); assert sha(TREE/'predictfun/research'/n) == h
    benchpath = DATA/'pipeline_bench_v2_20261119/bench.json'; bench=json.loads(benchpath.read_text())
    assert sha(benchpath) == sha(ARCHIVE/'bench.json')
    assert bench['source_sha256'] == sha(TREE/'predictfun/research/pf_pipeline_bench.py')
    for n,h in bench['models'].items(): assert sha(benchpath.parent/'models'/n) == h
    assert sha(benchpath.parent/'bench.txt') == bench['outputs']['bench.txt'] == sha(ARCHIVE/'bench.txt')
    out['synthetic'] = synthetic(); print('synthetic previous-finding closure and residual falsifier complete',flush=True)
    logs=io.StringIO()
    with contextlib.redirect_stdout(logs): EX.selftest(); P.selftest()
    (HERE/'selftests.txt').write_text(logs.getvalue()); out['selftests']={'export':'pass','policy':'pass','benchmark':'not rerun; would fit synthetic models'}
    out['days'] = {}
    for day in ('20261026','20261027'):
        out['days'][day] = saved_day(day,bench); gc.collect()
    out['models_hash_verified'] = len(bench['models']); out['new_tape_training_untouched_days'] = False
    out['completed_asof'] = datetime.now(timezone.utc).isoformat()
    with (HERE/'receipt.json').open('x') as f: json.dump(out,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({'receipt':str(HERE/'receipt.json')}),flush=True)


if __name__ == '__main__': main()
