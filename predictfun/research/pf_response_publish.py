"""Publish immutable completed evidence and preserve superseded partial stages."""
import json
from pathlib import Path
import subprocess
import pf_response_bridge as OLD
import pf_response_bridge_v2 as B
import pf_response_coarse_preserved_v2 as DIAG
C,F=B.C,B.F


def copy_new(src,dest):
    C.require(src.exists() and not dest.exists(),'archive source/destination '+str(dest))
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read_bytes())
    C.require(C.sha(src)==C.sha(dest),'archive identity')


def run():
    OLD.verify();B.verify();DIAG.verify()
    index=B.REVIEW/'index_v2.json';r=json.loads(index.read_text())
    for name,h in r['outputs'].items():C.require(C.sha(B.REVIEW/name)==h,'completed artifact changed')
    oldfiles=['fit/receipt.json','prediction_report.json']
    for stage in ('actions_20261026','actions_20261027'):
        receipt=json.loads((OLD.OUT/stage/'receipt.json').read_text())
        for name,h in receipt['outputs'].items():C.require(C.sha(OLD.OUT/stage/name)==h,'partial stage changed')
        oldfiles += [stage+'/receipt.json',stage+'/report.json']
    C.require(not (OLD.OUT/'pilot/receipt.json').exists(),'V1 pilot unexpectedly complete')
    for name in oldfiles:copy_new(OLD.OUT/name,OLD.REVIEW/'partial'/name)
    prior=OLD.REVIEW/'superseded_v2.json'
    C.dump(prior,dict(asof=C.stamp(),status='superseded; fit/action stages complete, pilot interrupted, no complete V1 strategy result',
        reason='Absolute-share response forecasts omitted the causal liquidity scale from normalized inputs; fixed in V2.',
        freeze_sha256=C.sha(OLD.FREEZE),superseding_freeze=str(B.FREEZE),superseding_freeze_sha256=C.sha(B.FREEZE),
        outputs={name:C.sha(OLD.REVIEW/'partial'/name) for name in oldfiles}))
    src=B.OUT/'oracle_coarse_preserved';receipt=json.loads((src/'receipt.json').read_text())
    for name,h in receipt['outputs'].items():C.require(C.sha(src/name)==h,'coarse-preserved diagnostic changed')
    C.require(receipt['source_manifest']==C.sha(DIAG.DECL),'coarse-preserved diagnostic source')
    for name in ('receipt.json','report.json'):copy_new(src/name,B.REVIEW/'oracle_coarse_preserved'/name)
    report=json.loads((src/'report.json').read_text());n=report['common_support_actions']
    C.require(n==sum(report['metrics']['COARSE_PRESERVED_TRUE_RICH:'+kind]['n'] for kind in ('cancel','post')),'diagnostic population arithmetic')
    latest=B.REVIEW/'latest.json';C.require(not latest.exists(),'latest evidence index exists')
    paths=[index,prior,DIAG.DECL,B.REVIEW/'oracle_coarse_preserved/receipt.json',B.REVIEW/'oracle_coarse_preserved/report.json']
    C.dump(latest,dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        publisher_sha256=C.sha(Path(__file__)),active_candidate_source=B.verify()['source_commit'],freeze_sha256=C.sha(B.FREEZE),
        evidence={str(p.relative_to(F.ROOT)):C.sha(p) for p in paths},
        status='Experiment complete and audited; V2 is not promoted over SOFT_JOINT',
        scope='Consumed Oct26/27 action tests and12-market Oct27 replay; V1 retained as superseded partial evidence; future-information checks are retrospective only.',
        next='Repair joint conditional queue/timing/price response and projection consistency before model-family or hyperparameter changes. Do not treat finer marginal forecasts as a completed order-value model.'))
    print(json.dumps(dict(asof=C.stamp(),archived_v1_stages=len(oldfiles),diagnostic_common=n,index=str(latest))),flush=True)


if __name__=='__main__':run()
