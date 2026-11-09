"""Bounded-loop artifact and scope audit; no new policy or market replay."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REVIEW=HERE.parent/'reviews/optimization_loop_20261109'
DATA=Path('/home/yuqing/ctaNew/data/predict_fun/research/optimization_loop_20261109')
ROUNDS=('inventory','duration_latency','deadline_order','native_a2_v2','native_latency','paired_engine_v2','paired_v3_arm','paired_continuity')
FROZEN='d678e7c3c4e47c6d60ebc031edf8d7fc7317210d5e58ff301d92d7ae0ece5f17'


def command(args):return subprocess.check_output(args,cwd=ROOT,text=True).strip()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def verify_hash(actual,expected):
    if actual!=expected:raise ValueError('artifact hash mismatch')
def require(value,message):
    if not value:raise ValueError(message)
def instant(value):return dt.datetime.fromisoformat(value.replace('Z','+00:00'))


def verify_round(name,cache):
    review=REVIEW/name;data=DATA/name
    d=json.loads((review/'declaration.json').read_text())
    r=json.loads((review/'receipt.json').read_text())
    v=json.loads((review/'verification.json').read_text())
    def checked(path,expected):
        path=Path(path)
        # Only exact paths already pinned by a completed declaration/receipt.
        # No directory discovery or newly available tape is read.
        if str(path) not in cache:cache[str(path)]=sha(path)
        verify_hash(cache[str(path)],expected)
    for kind in ('declaration','receipt','verification'):
        checked(data/f'{kind}.json',sha(review/f'{kind}.json'))
    verify_hash(sha(review/'receipt.json'),v['receipt_sha256'])
    if 'declaration_sha256' in r:verify_hash(sha(review/'declaration.json'),r['declaration_sha256'])
    else:require(r['declaration']==d,'embedded declaration differs')
    source=d['source_commit']
    for file,expected in d['sources'].items():
        content=subprocess.check_output(['git','show',f'{source}:predictfun/research/{file}'],cwd=ROOT)
        verify_hash(hashlib.sha256(content).hexdigest(),expected)
    for file,expected in d.get('inputs',{}).items():checked(file,expected)
    for file,expected in r['artifacts'].items():checked(Path(file) if Path(file).is_absolute() else data/file,expected)
    path=str((review/'declaration.json').relative_to(ROOT))
    addition=command(['git','log','--diff-filter=A','--format=%H','--',path]).splitlines()
    require(len(addition)==1,'ambiguous declaration commit')
    declared_commit=addition[0]
    subprocess.run(['git','merge-base','--is-ancestor',source,declared_commit],cwd=ROOT,check=True)
    committed=instant(command(['git','show','-s','--format=%cI',declared_commit]))
    require(committed<=instant(r['as_of']),'declaration committed after outcome')
    require(instant(d['as_of'])<=instant(r['as_of'])<=instant(v['as_of']),'artifact chronology regressed')
    for file in ('receipt.json','verification.json'):
        subprocess.run(['git','ls-files','--error-unmatch',str((review/file).relative_to(ROOT))],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    return dict(round=name,declaration_commit=declared_commit,source_commit=source,
                producer_as_of=r['as_of'],verifier_as_of=v['as_of'],population=r.get('population',d['population']),
                sources_verified=len(d['sources']),inputs_verified=len(d.get('inputs',{})),outputs_verified=len(r['artifacts']),
                declaration_before_outcomes=True,receipt_verification_link_exact=True,tracked=True,
                receipt_sha256=sha(review/'receipt.json'),verification_sha256=sha(review/'verification.json'))


def conclusions():
    inv=json.loads((REVIEW/'inventory/receipt.json').read_text())['summary']['base']
    latency=json.loads((REVIEW/'native_latency/receipt.json').read_text())['summary']
    b=inv['reference'];u=inv['all100']
    inventory={}
    for name in ('individual100','reserved100'):
        row=inv[name]
        inventory[name]=dict(aligned_change_pct=100*(row['aligned_lane_ms']/b['aligned_lane_ms']-1),
            unrestricted_aligned_gain_retained_pct=100*(row['aligned_lane_ms']-b['aligned_lane_ms'])/(u['aligned_lane_ms']-b['aligned_lane_ms']),
            active_share_change_pct=100*(row['share_ms']/b['share_ms']-1),
            added_active_range_share_ms=row['active_range_extra_share_ms']-b['active_range_extra_share_ms'],
            added_off_nonreducing_share_ms=row['off_other_share_ms']-b['off_other_share_ms'])
    profiles={}
    for name,summary in latency.items():
        totals=summary['totals'];b=totals['reference'];q=totals['update_first50'];long=totals['all100']
        profiles[name]={key:100*(q[key]/b[key]-1) for key in ('venue_requests','aligned_lane_ms','active_share_ms','active_range_extra_share_ms','elsewhere_lane_ms')}
        profiles[name]['aligned_gain_vs100_retained_pct']=100*(q['aligned_lane_ms']-b['aligned_lane_ms'])/(long['aligned_lane_ms']-b['aligned_lane_ms'])
    return dict(inventory=inventory,native_latency_profiles=profiles,
                every_profile_fewer_requests_more_aligned=all(q['venue_requests']<0 and q['aligned_lane_ms']>0 for q in profiles.values()),
                every_profile_more_active_exposure=all(q['active_share_ms']>0 and q['active_range_extra_share_ms']>0 for q in profiles.values()))


def main():
    as_of=dt.datetime.now(dt.timezone.utc).isoformat()
    require(sha(HERE/'pf_step4_ladder_v3.py')==FROZEN,'frozen engine changed')
    verify_hash(FROZEN,FROZEN)
    try:verify_hash(FROZEN,'0'*64)
    except ValueError:bad_hash_refused=True
    else:raise ValueError('bad hash accepted')
    cache={};rounds=[verify_round(name,cache) for name in ROUNDS]
    baseline='691f80c1'
    changes=command(['git','diff','--name-only',baseline,'HEAD']).splitlines()
    require(all(p.startswith('predictfun/') for p in changes),'out-of-scope tracked change')
    require('predictfun/research/pf_step4_ladder_v3.py' not in changes,'frozen file diff')
    original=json.loads(command(['git','show','4666fc52:predictfun/reviews/optimization_loop_20261109/plan.json']))
    require({r['round'] for r in original['loop']}<={r['round'] for r in rounds},'original round missing')
    result=dict(as_of=as_of,audit_source_commit=command(['git','rev-parse','HEAD']),audit_source_sha256=sha(HERE/'pf_optimization_loop_audit.py'),
                original_plan_commit='4666fc52',original_end_of_pass=original['end_of_pass'],
                round_receipt_sets=rounds,unique_pinned_artifacts_hashed=len(cache),pinned_artifact_hashes=cache,
                controls=dict(valid_hash_accepted=True,changed_hash_refused=bad_hash_refused),
                frozen_source_sha256=FROZEN,tracked_changes_within_predictfun=True,conclusions=conclusions(),
                evidence_scope='Source pins, predeclaration ancestry/timing, all pinned inputs and saved output hashes, receipt links and arithmetic recomputation. Existing control/lifecycle verifications are referenced, not rerun. No new raw market population, selection, or economics.',
                completed_research_requirements=['Original inventory restrictions and duration/latency comparison','Declared equal-clock/native execution/latency follow-ups',
                    'Paired engine and bounded-retention synthetic integration, with explicit parity scope/exception','Tracked immutable failed attempts and superseding findings',
                    'Source/receipt controls, saved-artifact verification and commit/push workflow'],
                unmet_promotion_requirements=[
                    dict(requirement='Higher net trading value / lower realized inventory variance',status='not established',reason='Quote coverage and exposure proxies are not calibrated fill/PnL outcomes.'),
                    dict(requirement='Calibrated own-order filled and unfilled paths',status='not established',reason='Existing fill-clock audit rejects exact matching-clock use; new own-fill millisecond source/receipt capture is needed.'),
                    dict(requirement='Independent validation on at least5 complete untouched days',status='not established',reason='Only consumed Nov8 partial-day diagnostics used; Nov9–13 remain protected.'),
                    dict(requirement='Blanket B4 fixed-size parity',status='contradicted by explicit correction',reason='Native rejected-create slot release differs from the frozen bug; scoped normal controls pass.'),
                    dict(requirement='Historical file-output parity',status='not established',reason='Only complete function-output synthetic parity was run.'),
                    dict(requirement='Deployed equal-clock LC50 ordering, native policy/guard equivalence, wallet/kernel integration',status='not established',reason='Synthetic scheduler and effective-want oracle do not independently verify these.'),
                ],
                no_new_market_data_read=True,no_protected_data_read=True,no_deployment_performed=True,
                promotion_decision='No strategy promoted. Completed research investigations can have a negative or conditional finding; production and economic gates remain open.')
    for destination in (DATA/'completion_audit',REVIEW/'completion_audit'):
        destination.mkdir(exist_ok=True)
        with (destination/'receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(rounds=len(rounds),unique_artifacts=len(cache),conclusions=result['conclusions']),indent=2))


if __name__=='__main__':main()
