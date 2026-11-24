"""Frozen full-population evidence map and descriptive sequential-action census."""
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ['MT_RUN'] = 'v2'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'predictfun/reviews/maker_trace_20261122'
CH = OLD / 'v2/chains'
sys.path[:0] = [str(OLD / 'tools'), str(ROOT / 'predictfun/research')]
import numpy as np
import pandas as pd
import take_logic as TL

INPUTS = {
    'take_panel': CH / 'take_logic_v2/panel.parquet',
    'take_events': CH / 'take_logic_v2/events.parquet',
    'take_ratios': CH / 'take_logic_v2/ratios.csv',
    'take_consistency': CH / 'take_logic_v2/consistency.csv',
    'take_receipt': CH / 'take_logic_v2/receipt.json',
    'post_panel': CH / 'posting_opportunity/rows.parquet',
    'post_ratios': CH / 'posting_opportunity/ratios.csv',
    'post_receipt': CH / 'posting_opportunity/receipt.json',
    'ct_outcomes': CH / 'competing_transition/outcomes.csv',
    'ct_rd': CH / 'competing_transition/rd.csv',
    'ct_claims': CH / 'competing_transition/claims.json',
    'ct_receipt': CH / 'competing_transition/receipt.json',
    'terminal': ROOT / 'predictfun/reviews/maker_terminal_chain_20261124/summary.json',
}
STATES = ['inventory', 'btc', 'time_left', 'prior_fills']
OUTCOMES = ['O1', 'O2', 'O3', 'O4']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def need(ok, message):
    if not ok:
        raise ValueError(message)

def csv(path):
    return pd.read_csv(path, float_precision='round_trip')

def plan():
    files = list(INPUTS.values()) + list(HERE.glob('*.py')) + [HERE / 'contract.md']
    files += list((OLD / 'tools').glob('*.py')) + list((ROOT / 'predictfun/research').glob('pf_mt_*.py'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))}
    p = HERE / 'manifest.json'
    need(not p.exists(), 'manifest already exists')
    p.write_text(json.dumps({'version': 'condition_action_chain_v1', 'files': hashes, 'inputs': {k: str(p.relative_to(ROOT)) for k, p in INPUTS.items()}}, indent=2) + '\n')
    print('Manifest written; no outcome inspected.')

def bound():
    m = json.loads((HERE / 'manifest.json').read_text())
    for name, sha in m['files'].items():
        need(digest(ROOT / name) == sha, 'input/source hash: ' + name)
    return m

def validate_population(p, n, group):
    need(len(p) == n, 'population size')
    need(not p.duplicated(group + ['k']).any(), 'duplicate opportunity')
    for _, g in p.groupby(group):
        s = g.sort_values('k')
        need(np.array_equal(s.k.to_numpy(), np.arange(300)), 'missing second')
        need(np.array_equal(np.diff(s.t_ms), np.full(299, 1000)), 'time grid')
    need(set(p.day.astype(str)) == {'20261024','20261025','20261026','20261027'}, 'days')

def state_frame(p, variant, panel):
    raw = {c: p[c].to_numpy() for c in p.columns}
    cv = TL.variant_codes(raw, variant)
    q = cv['q']
    need(np.isfinite(q).all(), 'unknown reconstructed inventory')
    d = '1' if variant == 'D1000' else '0'
    t = p['tl' + d].to_numpy()
    x = p[['day', 'mid', 'rank', 'pair', 'k', 't_ms']].copy()
    x['inventory'] = np.array(TL.Q_LAB)[cv['qb']]
    x['btc'] = np.array(TL.CAT_LAB)[cv['cat_5s']]
    x['time_left'] = np.select([t <= 30, t <= 60, t <= 240], ['<=30', '(30,60]', '(60,240]'], '>240')
    x['prior_fills'] = np.array(TL.F5_LAB)[cv['f5b']]
    x['O1'] = p['O1_' + variant].astype(int)
    x['O2'] = p['O2_' + variant].astype(int)
    x['O3'] = p.O3.astype(int)
    x['O4'] = p.O4_reduces_all.astype(int)
    x['q_nonzero'] = np.abs(q) > TL.POS_EPS
    if panel == 'clean':
        x = x.loc[p.clean.astype(bool)].copy()
    return x

def joint(x):
    pieces = []
    for scope in ('day_rank', 'rank', 'pooled'):
        z = x.copy()
        if scope != 'day_rank': z['day'] = 'all'
        if scope == 'pooled': z['rank'] = 0
        g = z.groupby(['day', 'rank'] + STATES, dropna=False)
        r = g.agg(seconds=('k','size'), markets=('mid','nunique'), **{o:(o,'sum') for o in OUTCOMES}).reset_index()
        r['scope'] = scope
        r['thin'] = (r.seconds < 30) | (r.O3 < 5)
        for o in OUTCOMES: r[o + '_rate'] = r[o] / r.seconds
        pieces.append(r)
    return pd.concat(pieces, ignore_index=True)

def matched(x):
    z = x[x.q_nonzero & (x.prior_fills != '1-2') & (x.btc != 'missing')].copy()
    z['exposed'] = (z.prior_fills == '>=3').astype(int)
    keys = ['day', 'mid', 'rank', 'inventory', 'btc', 'time_left']
    a = z.groupby(keys + ['exposed']).agg(n=('k','size'), **{o:(o,'sum') for o in OUTCOMES[:3]}).reset_index()
    left = a[a.exposed == 1].drop(columns='exposed')
    right = a[a.exposed == 0].drop(columns='exposed')
    b = left.merge(right, on=keys, suffixes=('_1','_0'), validate='one_to_one')
    b['w'] = b.n_1 * b.n_0 / (b.n_1 + b.n_0)
    rows = []
    for day in ['all'] + sorted(x.day.astype(str).unique()):
        bb = b if day == 'all' else b[b.day.astype(str) == day]
        zz = z if day == 'all' else z[z.day.astype(str) == day]
        for o in OUTCOMES[:3]:
            w = float(bb.w.sum())
            numerator = bb.w * (bb[o + '_1'] / bb.n_1 - bb[o + '_0'] / bb.n_0)
            per_market = numerator.groupby(bb.mid).sum()
            rows.append(dict(day=day, outcome=o, strata=len(bb), markets=bb.mid.nunique(),
                exposed_all=int((zz.exposed == 1).sum()), reference_all=int((zz.exposed == 0).sum()),
                exposed_common=int(bb.n_1.sum()), reference_common=int(bb.n_0.sum()),
                exposed_events=int(bb[o+'_1'].sum()), reference_events=int(bb[o+'_0'].sum()),
                weight=w, rd=float(numerator.sum()/w) if w else None,
                top_market_abs_share=float(per_market.abs().max()/per_market.abs().sum()) if per_market.abs().sum() else None,
                descriptive_only=True))
    return pd.DataFrame(rows)

def make_map():
    rows = []
    tr = csv(INPUTS['take_ratios'])
    for r in tr[tr.scope == 'pooled'].to_dict('records'):
        rows.append(dict(layer='identified_maker_take', action=r['outcome'], condition=r['ratio'], variant=r['variant'], panel=r['panel'],
            exposed=r['exposed'], reference=r['reference'], n1=r['T_exposed'], e1=r['a_exposed'], n0=r['T_reference'], e0=r['b_reference'],
            effect=r['RR_MH'], effect_unit='matched_rate_ratio', verdict=r['verdict_rule'], p_hi=r['p_hi'], p_lo=r['p_lo'],
            support_markets=r['n_markets_supported'], creator_identified=True))
    for r in csv(INPUTS['post_ratios']).to_dict('records'):
        rows.append(dict(layer='anonymous_market_creation', action=r['type'], condition=r['contrast'], variant=r['variant'], panel='market',
            exposed=r['exposed'], reference=r['reference'], n1=r['T_exposed'], e1=r['a_exposed'], n0=r['T_reference'], e0=r['b_reference'],
            effect=r['RR_MH'], effect_unit='matched_rate_ratio', verdict=r['verdict'], p_hi=r['p_hi'], p_lo=r['p_lo'],
            support_markets=r['n_markets_supported'], creator_identified=False))
    ct = json.loads(INPUTS['ct_claims'].read_text())['contrasts']
    for r in csv(INPUTS['ct_rd']).query("scope == 'pooled'").to_dict('records'):
        c = ct[r['contrast']]
        rows.append(dict(layer='anonymous_retreat_response', action='next_same_side_response', condition=r['contrast'], variant=r['variant'], panel='episodes',
            exposed=c['exposed'], reference=c.get('reference','frozen_reference'), n1=r['n1'], e1=r['r1'], n0=r['n0'], e0=r['r0'],
            effect=r['rd'], effect_unit='matched_risk_difference', verdict=c.get('reading', c.get('verdict', 'see_original_claims')),
            p_hi=None, p_lo=None, support_markets=None, creator_identified=False))
    out = pd.DataFrame(rows)
    out['exposed_rate'] = out.e1 / out.n1.replace(0, np.nan)
    out['reference_rate'] = out.e0 / out.n0.replace(0, np.nan)
    need(not out[out.layer != 'identified_maker_take'].creator_identified.any(), 'false maker attribution')
    return out

def overview(m, seq):
    out = []
    for (layer, action, condition), g in m.groupby(['layer','action','condition']):
        key = (g.variant == 'D0') & g.panel.isin(['full','market','episodes'])
        b = g[key]
        need(len(b) == 1, 'map primary key')
        z = b.iloc[0].to_dict()
        baseline = 0 if z['effect_unit'] == 'matched_risk_difference' else 1
        s = np.sign(g.effect.to_numpy() - baseline)
        z.update(views=len(g), effect_min=float(g.effect.min()), effect_max=float(g.effect.max()),
                 same_direction_all_views=bool((s == s[0]).all()), original_positive_support_all_views=bool((g.verdict == 'supported').all()))
        out.append(z)
    return pd.DataFrame(out)

def run():
    start = time.process_time(); bound()
    out = HERE / 'outputs'; need(not out.exists(), 'write once')
    p = pd.read_parquet(INPUTS['take_panel']); q = pd.read_parquet(INPUTS['post_panel'])
    validate_population(p, 17100, ['day','mid','rank'])
    validate_population(q, 11400, ['day','mid','side'])
    cells=[]; seq=[]; exclusions=[]
    for v in TL.VARIANTS:
        for panel in ('full','clean'):
            x=state_frame(p,v,panel)
            j=joint(x); j['variant']=v; j['panel']=panel; cells.append(j)
            s=matched(x); s['variant']=v; s['panel']=panel; seq.append(s)
            exclusions.append(dict(variant=v,panel=panel,seconds=len(x),zero_inventory=int((~x.q_nonzero).sum()),
                middle_burst=int((x.prior_fills=='1-2').sum()),missing_btc=int((x.btc=='missing').sum()),
                contrast_eligible=int((x.q_nonzero & (x.prior_fills!='1-2') & (x.btc!='missing')).sum())))
    m=make_map(); ss=pd.concat(seq,ignore_index=True); ov=overview(m,ss)
    need(time.process_time()-start < 30, 'CPU cap')
    need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 < 1450, 'RSS cap')
    out.mkdir()
    pd.concat(cells,ignore_index=True).to_csv(out/'joint_states.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    ss.to_csv(out/'sequence_contrast.csv',index=False)
    pd.DataFrame(exclusions).to_csv(out/'exclusions.csv',index=False)
    m.to_csv(out/'evidence_map.csv',index=False); ov.to_csv(out/'overview.csv',index=False)
    terminal=json.loads(INPUTS['terminal'].read_text())
    report=dict(as_of_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        maker_seconds=len(p),market_side_seconds=len(q),markets=p.mid.nunique(),ranks=sorted(p['rank'].unique().tolist()),
        days=sorted(p.day.astype(str).unique()),action_comparisons=len(ov),map_rows=len(m),
        terminal_statuses=terminal['statuses'],post_pull_queue_private_logic_identified=False,
        new_joint_cells_are_descriptive=True,cpu_seconds=time.process_time()-start,
        hashes={f.name:digest(f) for f in out.iterdir() if f.is_file()})
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    {'plan':plan,'run':run}[sys.argv[1]]()
