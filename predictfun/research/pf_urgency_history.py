"""Fixed urgency rule on all four available earlier, consumed research days."""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .pf_sizing_study import ROOT, HERE, BASE, DATA as INPUT, sha, write
from .pf_sizing_run import COMMON, stem, name, reconcile, folder as baseline_folder
from . import pf_passive_engine as engine
from . import pf_passive_report as metrics

DAYS = ['20261023', '20261024', '20261025', '20261026']
ARMS = {'baseline': BASE, 'urgent_quote': BASE.replace('+CAP10', '+CAP10+RPRICE')}
DEST = ROOT / 'data/predict_fun/research/urgency_history_20261109'
DATA = DEST / 'research'
PRIOR = HERE.parent / 'reviews/passive_inventory_20261109/receipt.json'
ENGINE_SHA = 'f014efbe46653d5d238a3deba81b1cd82810350b12c8426075d371bdd6a02875'
METRICS_SHA = 'fd08d96d2271a4e2bf33e432e4c6ad95e3faea9c20157b990f71412cae3b16b1'


def identity(day, timing):
    if day not in DAYS or timing not in ['J', 'X']:
        raise ValueError('undeclared earlier day or timing')
    if sha(HERE / 'pf_passive_engine.py') != ENGINE_SHA:
        raise ValueError('urgency policy source changed')
    if sha(HERE / 'pf_passive_report.py') != METRICS_SHA:
        raise ValueError('shared metric source changed')


def folder(day, timing):
    return DATA / 'step4' / f'ladder_tokyoreal_custom_e2_preds_basis_study_urgency_history_{timing}' / day


def pre180_equal(b, a, kind, ws):
    if kind == 'fills':
        select = lambda x: x[x.t < 180].reset_index(drop=True)
    elif kind == 'requests':
        select = lambda x: x[x.t_ms < x.mid.map(ws) + 180000].reset_index(drop=True)
    else:
        raise ValueError('unknown entry artifact')
    pd.testing.assert_frame_equal(select(b), select(a), check_exact=True)


def day_null(deltas):
    """Shared day sign flips; four historical days get no confidence interval."""
    a = np.asarray(deltas, float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] == 0 or not np.isfinite(a).all():
        raise ValueError('invalid day differences')
    rng = np.random.default_rng(20261109)
    null = rng.choice([-1, 1], size=(2000, len(a))) @ a / len(a)
    obs = a.mean(axis=0)
    scale = np.maximum(np.sqrt(np.mean(a*a, axis=0)/len(a)), 1e-12)
    max_t = np.max(np.abs(null/scale), axis=1)
    return [dict(mean_per_day=float(v), interval=None,
                 p_two_sided=float((1 + (np.abs(null[:, i]) >= abs(v)).sum()) / 2001),
                 p_max_t_family=float((1 + (max_t >= abs(v/scale[i])).sum()) / 2001))
            for i, v in enumerate(obs)]


def setup():
    identity(DAYS[0], 'J')
    if DEST.exists():
        raise ValueError('refuse existing experiment')
    prior = json.loads(PRIOR.read_text())
    if prior['declaration']['arms']['urgent_quote'] != ARMS['urgent_quote']:
        raise ValueError('candidate differs from previous study')
    (DEST / 'logs').mkdir(parents=True)
    (DATA / 'step4').mkdir(parents=True)
    files = []
    for day in DAYS:
        for stage in ['step1', 'step5']:
            dst = DATA / stage / day
            dst.parent.mkdir(exist_ok=True)
            dst.symlink_to(INPUT / stage / day, target_is_directory=True)
        files += [DATA/'step5'/day/'preds_basis_study.parquet']
        files += [DATA/'step1'/day/f for f in ['registry.parquet', 'clock_rl_d2.json', 'legs_timed2.parquet', 'legs_tk.parquet']]
        files += sorted((DATA/'step1'/day/'books').glob('*.parquet'))
    for f in ['minute_lines_t60k.json', 'inv_r60.json']:
        (DATA/'step4'/f).symlink_to(INPUT/'step4'/f)
        files.append(DATA/'step4'/f)
    write(DEST/'declaration.json', dict(
        as_of=dt.datetime.now(dt.timezone.utc).isoformat(), days=DAYS,
        population='all available earlier complete prepared dates; 288 venue-resolved markets per day',
        selection='urgency chosen after the previous four-candidate comparison; no threshold tuning',
        arms=ARMS, prior_receipt_sha=sha(PRIOR), policy_source_sha=ENGINE_SHA,
        policy='same one-cent reducing-side improvement into spread; 180<=t<240, abs(known inventory)>=10; original guards',
        unchanged=['fixed guard models', 'continuous basis', 'known-state reservations', 'lot/minimum notional', 'stop at 240s'],
        loss_budget=None, exposure_cap_shares=150, rebates='excluded', timings=['J', 'X'],
        null=dict(design='2000 shared UTC-day sign flips, sign-invariant RMS max-T', seed=20261109,
                  draws=2000, family='J/X x settlement/+1s = 4', interval='none: only four complete earlier days'),
        multiplicity=dict(prior_candidates=4, prior_comparisons=16, selected_candidates=1,
                          earlier_comparisons=4, pooled='descriptive points only; no pooled significance claim'),
        metrics=['settlement', '+1s markout', 'mark at 240s', 'carry', 'residual inventory',
                 'inventory area', 'reducing volume', 'requests', 'fill-generation bounds', 'day concentration'],
        controls=['unchanged policy/metric source hashes', 'fresh first-day baseline parity at J/X',
                  'exact pre-180s fills and requests', 'per-market settlement, carry and inventory reconciliation',
                  'positive controls and known-bad refusals'],
        baseline_reuse='fresh Oct23 J/X baseline; Oct24-26 prior exact baseline artifacts copied with hashes',
        earlier_days_in_guard_training=True, inference='historical in-sample extension, not untouched validation',
        limitations=['only four day clusters', 'candidate selected from prior results',
                     'fixed delays, no own market impact or full OMS', 'no new cancellation stress sweep'],
        prior_stress_clarification='Nov3 urgency delta was identical in ordinary and cancel-p99 runs; the negative day is not evidence of incremental delay sensitivity',
        prepared_input_sha={str(p):sha(p) for p in files},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=HERE, text=True).strip(),
        forward_data_read=False, live_settings_changed=False))
    print('DECLARED', DAYS, flush=True)


def engine_main(day, timing):
    identity(day, timing)
    def context():
        from . import pf_io
        pf_io.OUT = str(DATA)
    engine.context = context  # Route output only; the pinned policy implementation is untouched.
    sys.argv = [sys.argv[0], day, 'tokyoreal', 'custom']
    engine.main()


def summarize(day, timing, label, spec):
    identity(day, timing)
    metrics.folder = lambda d, t, mode: folder(d, t)
    return metrics.summarize(day, timing, 'earlier', label, spec)


def check(day, timing):
    out = folder(day, timing)
    b = {k:pd.read_parquet(out/k/stem(BASE)) for k in ['win','fills','requests']}
    ws = b['win'].set_index('mid').ws
    checked = []
    for label, spec in ARMS.items():
        w, f, q = (pd.read_parquet(out/k/stem(spec)) for k in ['win','fills','requests'])
        if len(w) != 288 or not np.array_equal(w.mid, b['win'].mid):
            raise ValueError('market coverage or identity')
        if not f.mid.isin(w.mid).all() or not q.mid.isin(w.mid).all():
            raise ValueError('orphan fill/request')
        reconcile(w, f)
        if not np.allclose(w.pnl, w.mtm_stop+w.carry_stop, atol=1e-7, rtol=0):
            raise ValueError('carry reconciliation')
        for kind, a in [('fills', f), ('requests', q)]:
            pre180_equal(b[kind], a, kind, ws)
            checked.append(dict(variant=label, kind=kind, pre180_exact=True))
    for kind in ['win','fills','requests']:
        ref = pd.read_parquet(baseline_folder(day,timing)/kind/stem(BASE))
        pd.testing.assert_frame_equal(b[kind], ref, check_exact=True)
    write(out/'passive_checks.json', dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        markets=288, day=day, timing=timing, checks=checked, baseline_exact=['win','fills','requests'],
        baseline_origin='retained reference' if (out/'baseline_reference.json').exists() else 'fresh replay',
        policy_source_sha=ENGINE_SHA))


def run(day, timing):
    identity(day, timing)
    out = folder(day, timing)
    if out.exists():
        raise ValueError('refuse overwrite')
    reuse = day != DAYS[0]
    specs = [ARMS['urgent_quote']] if reuse else list(ARMS.values())
    inherited = {k:v for k,v in os.environ.items() if not k.startswith('PF_')}
    env = dict(inherited, **COMMON, PF_FORWARD='20261104', PF_PREDS='preds_basis_study',
        PF_TAG=f'urgency_history_{timing}', PF_MAX_WINDOWS='0', PF_ARMS=';'.join(specs),
        PF_STUDY_EXEC='1', PF_CAP_CACHE='0', PF_STUDY_CANCEL_ACK_MS='0',
        PF_TXJIT='frame' if timing=='J' else '', PF_TXSHIFT='0' if timing=='J' else '-50',
        PF_QACT='' if timing=='J' else 'max')
    subprocess.run([sys.executable,'-m',__package__+'.pf_urgency_history','engine',day,timing],
                   env=env, cwd=HERE.parents[1], check=True)
    if reuse:
        ref = baseline_folder(day,timing)
        paths = {}
        for kind in ['win','fills','requests']:
            src = ref/kind/stem(BASE)
            shutil.copy2(src,out/kind/src.name)
            paths[kind] = dict(path=str(src),sha256=sha(src))
        write(out/'baseline_reference.json',dict(summary_path=str(ref/'summary.json'),
            summary_sha=sha(ref/'summary.json'),artifacts=paths))
    check(day,timing)
    audit(day,timing)


def audit(day, timing):
    for label,spec in ARMS.items():
        d,_,_ = summarize(day,timing,label,spec)
        print('RECONCILED',day,timing,label,d['settle'],flush=True)


def comparison(b, a):
    if [d['day'] for d in b] != [d['day'] for d in a] or len(set(d['day'] for d in b)) != len(b):
        raise ValueError('unmatched or duplicated days')
    keys = ['settle','plus1','mark_stop','carry','reducing_shares','reducing_mo1','shares']
    totals = lambda rows: {k:float(sum(r[k] for r in rows)) for k in keys}
    bt,at = totals(b),totals(a)
    delta = [x['settle']-y['settle'] for x,y in zip(a,b)]
    return dict(days=len(b),markets=sum(r['markets'] for r in b),baseline=bt,urgency=at,
        delta={k:at[k]-bt[k] for k in keys},
        mean_abs_net_end={k:float(np.mean([r['abs_net_end_mean'] for r in rs])) for k,rs in [('baseline',b),('urgency',a)]},
        requests={k:{kind:sum(r['requests'].get(kind,0) for r in rs) for kind in ['place','cancel']} for k,rs in [('baseline',b),('urgency',a)]},
        days_better=sum(x>1e-8 for x in delta),days_worse=sum(x< -1e-8 for x in delta),
        leave_one_out_delta=[float(sum(delta)-max(delta)),float(sum(delta)-min(delta))],
        per_day=[dict(day=x['day'],settle_delta=v,plus1_delta=x['plus1']-y['plus1']) for x,y,v in zip(a,b,delta)])


def report():
    daily=[];traces=[];checks={};execution={}
    for timing in ['J','X']:
        for day in DAYS:
            out=folder(day,timing)
            for label,spec in ARMS.items():
                d,_,tr=summarize(day,timing,label,spec);daily.append(d)
                traces.extend(dict(r,timing=timing,variant=label) for r in tr.to_dict('records'))
            checks[f'{timing}/{day}']=json.loads((out/'passive_checks.json').read_text())
            execution[f'{timing}/{day}']=dict(summary_sha=sha(out/'summary.json'),
                execution=json.loads((out/'study_execution.json').read_text()),
                baseline_reference=json.loads((out/'baseline_reference.json').read_text()) if (out/'baseline_reference.json').exists() else None,
                passive_audit_sha=sha(out/'passive_audit.json'))
    prior=json.loads(PRIOR.read_text());groups={};delta=[]
    for timing in ['J','X']:
        b=[d for d in daily if d['timing']==timing and d['variant']=='baseline']
        a=[d for d in daily if d['timing']==timing and d['variant']=='urgent_quote']
        pb=next(r['per_day'] for r in prior['results'] if r['mode']=='main' and r['timing']==timing and r['variant']=='baseline')
        pa=next(r['per_day'] for r in prior['results'] if r['mode']=='main' and r['timing']==timing and r['variant']=='urgent_quote')
        groups[timing]=dict(earlier=comparison(b,a),previous=comparison(pb,pa),pooled=comparison(b+pb,a+pa))
        for metric in ['settle','plus1']:delta.append([x[metric]-y[metric] for x,y in zip(a,b)])
    inference=dict(zip(['J/settle','J/plus1','X/settle','X/plus1'],day_null(np.array(delta).T)))
    declaration=json.loads((DEST/'declaration.json').read_text())
    for path,expected in declaration['prepared_input_sha'].items():
        if sha(path)!=expected:raise ValueError('prepared input changed')
    from . import pf_io as io
    raw=sorted({Path(io.BN_DIR)/f'{h}.csv.gz' for day in DAYS for h in io.hours_of(day,extra_before=1,extra_after=1)})
    controls=(DEST/'logs/urgency_history_controls.log').read_text()
    if '\nOK\n' not in controls or 'rc=0 ' not in controls:raise ValueError('missing controls')
    sources=[HERE/'pf_urgency_history.py',HERE/'run_urgency_history.sh',HERE/'tests/test_urgency_history.py']
    sources += [HERE/f for f in ['pf_passive_engine.py','pf_passive_report.py','pf_sizing_engine.py','pf_step4_ladder_v3.py','pf_clock_rl.py','pf_io.py']]
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),declaration=declaration,
        declaration_sha=sha(DEST/'declaration.json'),prior_receipt_sha=sha(PRIOR),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_sha={str(p.relative_to(HERE.parents[1])):sha(p) for p in sources},
        binance_input_sha={str(p):sha(p) for p in raw if p.exists()},missing_binance_hour_files=[str(p) for p in raw if not p.exists()],
        daily=daily,comparisons=groups,earlier_day_null=inference,checks=checks,execution_inputs=execution,
        source_controls=controls,intervals='none on four earlier days; pooled totals descriptive after selection',
        predicates={tm:{scope:dict(settle_better=g['delta']['settle']>0,plus1_better=g['delta']['plus1']>0,
                                   lower_inventory=g['mean_abs_net_end']['urgency']<g['mean_abs_net_end']['baseline'])
                        for scope,g in scopes.items()} for tm,scopes in groups.items()},
        interpretation='selected fixed rule on earlier consumed training days; not independent validation',
        forward_data_read=False,live_settings_changed=False,deployment=False)
    pd.DataFrame(traces).to_parquet(DEST/'inventory_paths.parquet',index=False)
    write(DEST/'report.json',result)
    write(HERE.parent/'reviews/urgency_history_20261109/receipt.json',result)
    print(json.dumps(dict(comparisons=groups,inference=inference),indent=2))


if __name__ == '__main__':
    command=sys.argv[1]
    if command=='declare' and len(sys.argv)==2:setup()
    elif command=='report' and len(sys.argv)==2:report()
    elif command in ['run','engine','audit'] and len(sys.argv)==4:
        {'run':run,'engine':engine_main,'audit':audit}[command](*sys.argv[2:])
    else:raise ValueError('unknown command')
