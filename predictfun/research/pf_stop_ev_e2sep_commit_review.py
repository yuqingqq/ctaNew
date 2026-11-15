"""Synthetic review: a WS-REACHED action may depend on an earlier E2 leg.

No historical reads, fits or changes to the running E2SEP experiment.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pf_stop_ev_accounting as A
import pf_stop_ev_accounting_check as T
import pf_stop_ev_timing_certificate as C


def main(out):
    source=Path(__file__).with_name('pf_stop_ev_e2sep.py')
    tree=ast.parse(source.read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='groups')
    scope=dict(np=np);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    classify=scope['groups'];settings=A.Settings(timing='all_anchors_sensitivity')
    infos={(0,0):A.LegInfo('earlier','e2',250.),(0,1):A.LegInfo('later','ws')}
    nominal=T.ledger(T.world(((100.,50,10.),(200.,50,5.)),queue=10.),info=infos)
    moved=T.ledger(T.world(((-150.,50,10.),(200.,50,5.)),queue=10.),info=infos)
    a=T.action(nominal,150.,settings=settings);b=T.action(moved,150.,settings=settings)
    group=classify(pd.DataFrame(dict(reach_ws=[5.],reach_e2=[0.])))[0]
    cert=C.Certificate(nominal,[]).assess(0,150.,1150.)
    assert group=='WS-REACHED'
    assert (a['label']['value_cents'],b['label']['value_cents'])==(5.,0.)
    assert not cert['certified'] and cert['prior_state_ids']==['earlier']
    # Positive controls: group function recognizes reaching E2; moving a leg
    # by zero preserves economics exactly.
    assert classify(pd.DataFrame(dict(reach_ws=[0.],reach_e2=[5.])))[0]=='E2-REACHED'
    assert T.action(T.ledger(nominal.W,info=infos),150.,settings=settings)['label']==a['label']
    est=ast.parse(source.with_name('pf_stop_ev.py').read_text())
    deps=next(ast.literal_eval(x.value) for x in est.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in x.targets))
    result=dict(asof=datetime.now(timezone.utc).isoformat(),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reviewed_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),group=str(group),decision_ms=150.,
        nominal_e2_ms=100.,shifted_e2_ms=-150.,ws_ms=200.,nominal_value_cents=a['label']['value_cents'],
        shifted_value_cents=b['label']['value_cents'],certificate=cert,positive_controls_pass=True,
        source_manifest_includes_e2sep=source.name in deps,
        interpretation='Synthetic counterexample to equating future WS reach with E2-independent queue/value; not an observed historical error rate.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
