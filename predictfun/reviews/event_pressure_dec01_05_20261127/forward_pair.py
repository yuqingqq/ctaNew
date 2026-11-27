"""Run the frozen pressure comparison and score the five December UTC dates."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REV = HERE.parent
ROOT = REV.parents[1]
BASE = REV / 'near_holdout_nov19_23_20261127'
EVENT = REV / 'event_pressure_cancel_20261127'
HIST = REV / 'event_pressure_all_dates_20261127'
OUT = ROOT / 'data/predict_fun/research/event_pressure_dec01_05'
DAYS = tuple(f'202612{d:02}' for d in range(1, 6))
STAGES = ('pf_step1', 'pf_step1_timing', 'pf_step1_booktime', 'pf_txmap')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def preflight():
    check = module(REV/'urgency_near_selected_20261127/forward_preflight.py', 'availability')
    return check.inspect(datetime(2026, 12, 1, tzinfo=timezone.utc), datetime.now(timezone.utc))


def verify_control():
    old = ROOT/'data/predict_fun/research/near_holdout_nov19_23/control_pass.json'
    receipt = json.loads(old.read_text())
    assert receipt['day'] == '20261026' and receipt['markets'] == 288
    assert receipt['sha'] == sha(BASE/'census_20261026.csv') == 'c00077b36e7564f6251497e6b0ccce9d1275ea5e2996b0bc30c34aa04b9ab45d'
    save(OUT/'control_pass.json', receipt)


def score(panels):
    old = module(EVENT/'score.py', 'event_score')
    g = pd.concat(panels, ignore_index=True)
    assert not g.duplicated(['day', 'mid']).any() and g.tape_id.notna().all()
    old.validate(g)
    damaged = g.iloc[:1].copy()
    damaged['added_adverse'] += 1
    try:
        old.validate(damaged)
    except AssertionError:
        pass
    else:
        raise ValueError('checker accepted a corrupted changed-fill value')
    r = old.score(g)
    daily = {str(d):old.score(rows) for d,rows in g.groupby('day')}
    ds = np.array([x['difference'] for x in daily.values()])
    ev = np.array([x['event_settle'] for x in daily.values()])
    signs = np.random.default_rng(20261127).choice([-1,1], size=(4096,len(ds)))
    r.update(improved_days=int((ds>0).sum()), profitable_days=int((ev>0).sum()),
             delta_without_best_day=float(ds.sum()-ds.max()),
             directional_pass=bool(ds.sum()>0 and ds.sum()-ds.max()>0 and (ds>0).sum()>len(ds)/2),
             absolute_pass=bool(ev.sum()>0 and ev.sum()-ev.max()>0 and ev.sum()-g.event_settle.max()>0 and (ev>0).sum()>len(ds)/2),
             descriptive_day_flip_ge=float(np.mean(signs@ds >= ds.sum()-1e-8)))
    for arm in ('grid','event'):
        for field in ('shares','favorable','adverse','win_sh','loss_sh','flat_sh','censored'):
            r[arm+'_marks_'+field] = float(g[arm+'_marks_'+field].sum())
    return dict(pooled=r, daily=daily)


def historical():
    ref = json.loads((HIST/'result.json').read_text())
    panels = []
    for d, digest in sorted(ref['hashes'].items()):
        p = (HIST if (HIST/f'paired_{d}.csv').exists() else EVENT)/f'paired_{d}.csv'
        assert sha(p) == digest
        frame = pd.read_csv(p, float_precision='round_trip')
        assert frame.day.eq(int(d)).all() and len(frame) == ref['coverage'][d]['complete']
        panels.append(frame)
    r = score(panels)['pooled']
    prior = ref['blocks']['all_fourteen']
    for k in prior.keys() & r.keys():
        assert np.isclose(r[k], prior[k], atol=1e-6, rtol=0), k
    assert r['markets']==1041 and len(panels)==14
    print(json.dumps(r, indent=2))


def stage(action, day):
    sys.path.insert(0, str(BASE))
    sys.path.insert(0, str(ROOT))
    import signals
    # Load previous builders with their original module paths first.
    if action in STAGES:
        from predictfun.research import pf_io
        pf_io.OUT = str(OUT)
        os.environ['PF_FORWARD'] = '20261207'  # allow Dec 6 00/01 UTC for the final market
        sys.argv = [action, day]
        runpy.run_module('predictfun.research.'+action, run_name='__main__')
    elif action == 'signals':
        signals.OUT, signals.BTC, signals.DAYS = OUT, OUT/'btc', DAYS
        signals.build(day)
    elif action == 'baseline':
        replay = module(BASE/'replay.py', 'baseline_replay')
        replay.OUT, replay.BTC, replay.DAYS, replay.HERE = OUT, OUT/'btc', DAYS, OUT
        replay.run(day)
    elif action == 'event':
        ev = module(EVENT/'run.py', 'pressure_replay')
        ev.H.OUT, ev.H.BTC = OUT, OUT/'btc'
        # The frozen pair builder reads hashes and census from its baseline receipt.
        part = OUT/day
        part.mkdir(exist_ok=True)
        files = {}
        for prefix in ('markets','census'):
            src = OUT/f'{prefix}_{day}.csv'
            dst = part/src.name
            dst.write_bytes(src.read_bytes())
            files[src.name] = sha(src)
        c = pd.read_csv(part/f'census_{day}.csv')
        if not eligible(c, day):
            print(day, 'insufficient coverage')
            return
        save(part/'result.json', {'hashes':files})
        step = OUT/'step1'/day
        bound = [step/name for name in ('registry.parquet','legs_timed2.parquet','legs_tk.parquet')]
        bound += sorted((step/'books').glob('part_*.parquet'))
        bound += [OUT/f'pressure_{day}.parquet',OUT/f'grid_{day}.json',OUT/'btc'/f'btc_{day}.parquet',
                  OUT/'btc'/f'btc_{ev.H.prev_day(day)}.parquet']
        save(part/'inputs.json', {str(p.relative_to(OUT)):sha(p) for p in bound})
        ev.PRIOR, ev.HERE, ev.DAYS = part, part, DAYS
        ev.one_day(day)
        # Bind the row file to the exact baseline and inputs used for this run.
        save(part/'pair_hashes.json', {name:sha(part/name) for name in
             (f'paired_{day}.csv', 'inputs.json', 'result.json')})


def eligible(c, day):
    start = pd.Timestamp(day,tz='UTC').value//1_000_000
    assert c.day.eq(int(day)).all() and len(c)==288 and set(c.opening_ms)==set(range(start,start+86400000,300000))
    return (c.status=='complete').sum()>=30 and not c.status.eq('registry_missing').any()


def finish():
    panels, coverage = [], {}
    for day in DAYS:
        part = OUT/day
        c = pd.read_csv(OUT/f'census_{day}.csv')
        coverage[day] = dict(Counter(c.status))
        if not eligible(c, day):
            continue
        seal = json.loads((part/'pair_hashes.json').read_text())
        assert set(seal) == {f'paired_{day}.csv','inputs.json','result.json'}
        assert all(sha(part/name)==digest for name,digest in seal.items())
        inputs = json.loads((part/'inputs.json').read_text())
        assert all(sha(OUT/name)==digest for name,digest in inputs.items())
        hashes = json.loads((part/'result.json').read_text())['hashes']
        assert all(sha(OUT/name)==digest and sha(part/name)==digest for name,digest in hashes.items())
        p = pd.read_csv(part/f'paired_{day}.csv', float_precision='round_trip')
        assert p.day.eq(int(day)).all() and p.mid.is_unique
        assert dict(zip(p.mid,p.tape_id)) == dict(zip(c.loc[c.status=='complete','mid'],c.loc[c.status=='complete','tape_id']))
        panels.append(p)
    result = dict(coverage=coverage, sufficient=len(panels)==len(DAYS))
    if result['sufficient']:
        result.update(score(panels))
    save(OUT/'comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('preflight','historical','finish',*STAGES,'signals','baseline','event'))
    parser.add_argument('day', nargs='?', choices=DAYS)
    args = parser.parse_args()
    if args.action == 'historical':
        historical()
    elif args.action == 'preflight':
        print(json.dumps(preflight(), indent=2))
    else:
        state = preflight()
        if any(s!='hour_files_present' for _,s,_ in state):
            sys.exit('Forward replay unavailable: '+str([(d,s) for d,s,_ in state]))
        OUT.mkdir(parents=True, exist_ok=True)
        verify_control()
        if args.action == 'finish':
            finish()
        elif args.day is not None:
            stage(args.action, args.day)
        else:
            parser.error('a day is required for this stage')
