"""Canonical causal wrapper and in-band superseding receipt for the completed study."""
import datetime as dt
import json
import subprocess
import numpy as np
import pandas as pd
from . import pf_order_policy_spec as S
from .pf_own_order_warmup_v3 import CausalWindow,controls


def check_decision_clocks(g):
    a=g[g.order_age_ms.notna()]
    lag=a.decision-a.ws-1000*a.t_rel
    if not np.isfinite(lag).all() or (lag<-.001).any() or (lag>100.001).any():raise ValueError('feature decision clock identity')
    if ((a.decision-a.ws)<250).any() or (a.t_rel<.25).any():raise ValueError('affected decision requires recomputation')


def declare():
    S.check();controls()
    good=pd.DataFrame([dict(order_age_ms=50.,decision=6500.,ws=0.,t_rel=6.5)])
    check_decision_clocks(good);bad=good.copy();bad.loc[0,'t_rel']=7.
    try:check_decision_clocks(bad)
    except ValueError:pass
    else:raise ValueError('known-bad clock accepted')
    audit=json.loads((S.OUT/'warmup_v3_audit.json').read_text());n=0
    for r in audit['rows']:
        if r['early_decisions'] or r['affected']:raise ValueError('cannot reuse affected primary results')
        p=S.OUT/'runs'/r['group']/r['day']/'generations'/f"{r['mode']}.parquet"
        if S.sha(p)!=r['input_sha']:raise ValueError('audited generation changed')
        g=pd.read_parquet(p);check_decision_clocks(g);n+=int(g.order_age_ms.notna().sum())
    files=[S.HERE/'pf_own_order_runner_v3.py',S.HERE/'pf_own_order_warmup_v3.py',S.HERE/'run_own_order_v3.sh',S.HERE/'pf_own_order_mechanism.py']
    d=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),version=3,status='CAUSAL_CORRECTION_DECLARED',
        supersedes='warm-up derivative implementation in original experiment; preserves its frozen source, artifacts and unchanged primary actions',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S.HERE,text=True).strip(),
        source_sha={str(p.relative_to(S.HERE.parents[1])):S.sha(p) for p in files},
        original_declaration_sha=S.sha(S.OUT/'declaration.json'),audit_sha=S.sha(S.OUT/'warmup_v3_audit.json'),
        audit_populations=len(audit['rows']),feature_decisions_checked=n,affected_primary_decisions=0,
        correction='Unknown imbalance derivatives until250ms of causal history. Original calm states already masked this boundary. All saved feature-decision clocks reconciled; zero first250ms feature decisions in the42 primary/neutral populations. Models and all primary actions/fills unchanged and reused. Stress replay uses CausalWindow. No policy/threshold/label/null change.',
        controls='Warm-up future-mutation invariance, original known-bad future read, unchanged calm/mature features, actual feature-clock identity and known-bad future-clock refusal pass.')
    S.write(S.OUT/'causal_v3_declaration.json',d);S.write(S.REVIEW/'causal_v3_declaration.json',d)
    print('CAUSAL_DECLARED',n,'feature decisions verified',flush=True)


def check():
    S.check();d=json.loads((S.OUT/'causal_v3_declaration.json').read_text())
    for p,h in d['source_sha'].items():
        if S.sha(S.HERE.parents[1]/p)!=h:raise ValueError('causal wrapper changed')
    if S.sha(S.OUT/'warmup_v3_audit.json')!=d['audit_sha']:raise ValueError('audit changed')
    return d


def replay(day,group):
    d=check()
    from . import pf_order_policy_hooks as H
    from .pf_order_policy_engine import run,folder
    H.Window=CausalWindow
    run(day,group)
    S.write(folder(day,group)/'causal_v3.json',dict(version=3,day=day,group=group,source_sha=S.sha(__file__),
        declaration_sha=S.sha(S.OUT/'causal_v3_declaration.json'),source=d['source_sha'],status='PASS'))


def report():
    d=check()
    for day in S.TEST:
        c=json.loads((S.OUT/'runs/stress'/day/'causal_v3.json').read_text())
        if c['declaration_sha']!=S.sha(S.OUT/'causal_v3_declaration.json') or c['status']!='PASS':raise ValueError('uncorrected stress source')
    from .pf_order_policy_report import run
    from .pf_own_order_mechanism import run as mechanism
    run();mechanism()
    base=S.OUT/'comparison.json';result=json.loads(base.read_text())
    result.update(version=3,status='CANONICAL',supersedes=dict(path=str(base),sha=S.sha(base)),causal_correction=d,
        mechanism_sha=S.sha(S.OUT/'mechanism.json'))
    S.write(S.OUT/'comparison_v3.json',result);S.write(S.REVIEW/'comparison_v3.json',result)
    old=S.HERE.parent/'handoffs/20261111_own_orders_calm_results.md';new=old.with_name('20261111_own_orders_calm_results_v3.md')
    prefix='# Canonical own-order continuation and calm-entry results (v3)\n\n'+(
        'This receipt supersedes the base comparison and records the warm-up correction. The two imbalance-change features are unknown until250ms of past history exists. '
        f"All {d['feature_decisions_checked']:,} saved feature decisions in42 neutral/primary populations were later than that boundary, so no original model, primary decision, fill or result was affected. "
        'Stress runs use the corrected wrapper. Original frozen sources/artifacts are preserved. Training v2 removes duplicate neutral actions; saved-model predictions were independently reconstructed with known-bad score refusals. '
        'Canonical machine receipt: reviews/own_orders_calm_20261111/comparison_v3.json.\n\n')
    with new.open('x') as f:f.write(prefix+'\n'.join(old.read_text().splitlines()[1:])+'\n')
    print('CANONICAL_REPORT',new,flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='declare':declare()
    elif sys.argv[1]=='report':report()
    else:replay(*sys.argv[1:])
