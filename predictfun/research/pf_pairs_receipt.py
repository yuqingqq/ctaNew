"""Verify saved B4 artifacts and publish explicit parity scope and exceptions."""
import copy
import hashlib
import json
import re
import subprocess
from . import pf_pairs_validation as study

s=study.s

def check_rows(cases,rows):
    if len(cases)!=232 or len(rows)!=232 or [r['name'] for r in rows]!=[c['name'] for c in cases]:raise ValueError('synthetic population/identity differs')
    if len({r['name'] for r in rows})!=232:raise ValueError('duplicate scenario')
    sums={k:0 for k in ['events','requests','decisions','fills','accepts','rejects','retry_events','native_throttled','native_topup_throttled','native_topup_withheld']}
    for case,r in zip(cases,rows):
        events=case['events'];kinds=[e['kind'] for e in events]
        expect=dict(events=len(events),requests=sum(k in ['place','cancel'] for k in kinds),decisions=kinds.count('D'),fills=kinds.count('fill'),accepts=kinds.count('accepted'),rejects=kinds.count('rejected'),retry_events=kinds.count('retry'))
        if any(r[k]!=v for k,v in expect.items()):raise ValueError('native artifact counts differ')
        if r['requests_and_pair_state_match'] is not True:raise ValueError('native differential check failed')
        orders={e['id']:e for e in events if e['kind']=='place'};net=cash=0.
        for e in events:
            if e['kind']=='fill':
                o=orders[e['id']];q=e['qty']*(1 if o['lane']%2==0 else -1);net+=q;cash-=q*o['tick']/100.
        if any(abs(a-b)>1e-8 for a,b in zip(r['known_position'],[net,cash])):raise ValueError('native known-fill ledger differs')
        for k in sums:sums[k]+=r[k]
    return sums

def saved_parity_controls():
    """Record output hashes from the actual declared tests, without changing engines."""
    from .tests import test_ladder_pairs as tests
    original=tests.scenario;calls=[];records=[]
    def capture(*args,**kwargs):
        result=original(*args,**kwargs)
        body=json.dumps(result,sort_keys=True).encode()
        calls.append(dict(sha256=hashlib.sha256(body).hexdigest(),requests=len(result['requests']),fills=len(result['fills']),
            resize_count=result['live_ctr'].get('lz_resize',0),rejection_answer_releases=result['live_ctr'].get('lz_rejection_answer_release',0)))
        return result
    def invoke(name,kind,function,*args):
        calls.clear();function(*args)
        if len(calls)!=2:raise ValueError('parity test did not produce two outputs')
        equal=calls[0]['sha256']==calls[1]['sha256']
        if equal!=(kind in ['no_token','fixed_size']):raise ValueError('parity direction differs')
        records.append(dict(name=name,kind=kind,outputs=copy.deepcopy(calls),byte_identical=equal))
    tests.scenario=capture
    try:
        for side in [0,1]:
            for i,extra in enumerate([set(),{'LD2','G6','CAP10'},{'LD2','RO1','S2','XV','XMIN1'}]):
                invoke(f'no_token/{side}/{i}','no_token',tests.test_no_token_exact_frozen_outputs,side,extra)
            for size in [5.,25.]:
                for i,extra in enumerate([set(),{'LD2','XV','XMIN1'}]):
                    invoke(f'fixed/{side}/{size}/{i}','fixed_size',tests.test_fixed_size_exact_control,side,size,extra)
            invoke(f'variable/{side}','variable_positive',tests.test_size_growth_keeps_front_and_variable_arm_falsifier,side)
            invoke(f'rejection/{side}','known_frozen_bug',tests.test_rejection_releases_unanswered_slot_before_answer_timer,side)
    finally:tests.scenario=original
    return records

def main():
    d=study.check();dest=study.DEST;review=study.REVIEW
    if '35 passed' not in (dest/'logs/matrix.log').read_text() or 'rc=0 ' not in (dest/'logs/oracle.log').read_text():raise ValueError('matrix/oracle did not pass')
    if not re.search(r'test result: ok\. 1 passed; 0 failed', (dest/'logs/build.log').read_text()):raise ValueError('native falsifier controls absent')
    cases=[json.loads(line) for line in (dest/'synthetic.jsonl').open()]
    rows=[json.loads(line) for line in (dest/'native.jsonl').open()]
    totals=check_rows(cases,rows)
    bad=copy.deepcopy(rows);bad[0]['requests']+=1
    try:check_rows(cases,bad)
    except ValueError:counts_refused=True
    else:raise ValueError('corrupt native artifact accepted')
    bad=copy.deepcopy(rows);bad[0]['name']='wrong'
    try:check_rows(cases,bad)
    except ValueError:identity_refused=True
    else:raise ValueError('wrong native identity accepted')
    # Re-run the immutable native oracle against the saved synthetic artifact.
    binary=s.TARGET/f'release/examples/{study.NAME}'
    subprocess.run([str(binary),str(dest/'synthetic.jsonl'),str(dest/'native_recheck.jsonl')],check=True)
    if s.sha(dest/'native_recheck.jsonl')!=s.sha(dest/'native.jsonl'):raise ValueError('native artifact failed exact reproduction')
    corrupt=copy.deepcopy(cases[0]);corrupt['name']='known_bad_size'
    next(e for e in corrupt['events'] if e['kind']=='place')['size']+=1
    with (dest/'known_bad.jsonl').open('x') as f:
        f.write(json.dumps(corrupt,separators=(',',':'))+'\n')
    with (dest/'logs/known_bad.log').open('x') as log:
        bad_run=subprocess.run([str(binary),str(dest/'known_bad.jsonl'),str(dest/'known_bad_native.jsonl')],stdout=log,stderr=subprocess.STDOUT)
    bad_log=(dest/'logs/known_bad.log').read_text()
    if bad_run.returncode==0 or 'request mismatch' not in bad_log or 'known_bad_size' not in bad_log:raise ValueError('native wrong-request falsifier failed for the wrong reason')
    parity=saved_parity_controls();s.write(dest/'parity_outputs.json',parity)
    coverage={k:totals[k]>0 for k in ['fills','rejects','native_throttled','native_topup_throttled','native_topup_withheld']}
    if not all(coverage.values()):raise ValueError('declared lifecycle branch not exercised')
    paths=[dest/k for k in ['synthetic.json','synthetic.jsonl','native.jsonl','native_recheck.jsonl','parity_outputs.json','known_bad.jsonl','known_bad_native.jsonl','logs/known_bad.log','logs/build.log','logs/matrix.log','logs/oracle.log']]
    result=dict(as_of=s.now(),declaration_sha256=s.sha(dest/'declaration.json'),sources=d['sources'],source_commit=d['source_commit'],build=d['build'],
        checker_source_commit=s.command(['git','rev-parse','HEAD'],cwd=s.a.HERE),checker_sha256=s.sha(s.a.HERE/'pf_pairs_receipt.py'),
        scope=d['scope'],population=dict(synthetic_traces=232,explicit_traces=32,seeded_traces=200,excluded_traces=0,unit_checks=35),
        totals=totals,coverage=coverage,parity=parity,native_replay_byte_identical=True,
        controls=dict(saved_counts_refused=counts_refused,saved_identity_refused=identity_refused,native_wrong_request_refused=bad_run.returncode!=0),
        limitations=d['limitations'],supersedes=d['supersedes'],no_market_data_read=True,no_protected_dates_read=True,
        artifacts={str(p):s.sha(p) for p in paths})
    s.write(dest/'receipt.json',result);s.write(review/'receipt.json',result)
    v=dict(as_of=s.now(),receipt_sha256=s.sha(dest/'receipt.json'),synthetic_traces=232,counts_and_fill_ledger_reconciled=True,
        native_replay_byte_identical=True,parity_output_hashes_saved=True,controls=result['controls'])
    s.write(dest/'verification.json',v);s.write(review/'verification.json',v)
    previous=review.parent/'findings_v4.json'
    combined=dict(as_of=result['as_of'],supersedes='findings_v4.json',supersedes_sha256=s.sha(previous),
        reason='Add paired-engine implementation validation; no new economic study, previous quote-mechanics results unchanged.',
        paired_engine=result,verification_sha256=s.sha(dest/'verification.json'),
        next=['Verify integration of bounded quote continuity with paired fills and inventory changes using synthetic cases before economic use.',
              'Full historical output-file parity and deployed timer behavior remain unverified; fill calibration still gates PnL/variance claims.'])
    s.write(review.parent/'findings_v5.json',combined);s.write(dest.parent/'findings_v5.json',combined)
    lines=['# Optimization loop: paired research engine — 2026-11-09','',
        f"As-of **{result['as_of']}**. **232 synthetic traces** (32 explicit,200 seeded), zero exclusions;35 Python checks. No historical/forward market data was read and no performance result is asserted.",'',
        'The generated `pf_step4_ladder_pairs.py` adds opt-in `LZ` incremental paired orders to frozen v3. A new quote is one whole order; an increase preserves the existing member and adds the remainder. Notified member fills are deducted from the unchanged logical target, and a waiting partner keeps the logical order active. Physical order slots are fixed to the four policy lanes and their partners at +4.','',
        '## Native comparison','',
        f"The pinned Rust f20fdb1 request mapper and feedback methods match all **{totals['requests']:,} requests**, **{totals['decisions']:,} decision pair states**, **{totals['fills']:,} fill notices**, and **{totals['retry_events']:,} retry events** in the declared synthetic traces. Checks also cover {totals['rejects']:,} rejections, {totals['native_topup_throttled']:,} throttled second-member attempts and {totals['native_topup_withheld']:,} minimum/size withholds.",'',
        'The native oracle consumes the research decisions and feedback. It verifies order mapping, notified-fill accounting and full-fill repost obligations; it does not independently re-estimate the model, calibrate market fills, or validate kernel/wallet scheduling. The saved native report is reproduced byte for byte. Corrupt request size, artifact counts and scenario identity are refused.','',
        '## A real frozen-handler mismatch found and corrected','',
        'The first oracle attempt matched78 traces, then refused random/46. Python kept a rejected create in the unanswered-request count until its nominal44ms answer deadline. Rust removes it immediately on rejection at40ms. With the three-unanswered limit already reached, Python unnecessarily withheld a replacement and scheduled a250ms retry. A later size-up therefore started from a different order.','',
        '`LZ` now releases that unanswered slot when the create is rejected. The frozen file is untouched. Two mirrored fixed-size rejection-stress controls demonstrate the intentional difference; a counter `lz_rejection_answer_release` makes it visible. The failed declaration, traces, partial native output and log remain preserved; `paired_engine_v2` supersedes them in-band.','',
        '**This prevents a blanket fixed-size parity claim.** The original B4 spec asks for unchanged fixed-size output and native parity; those requirements conflict on this pre-existing rejection bug. The corrected implementation prioritizes native lifecycle semantics only under `LZ`, records the exception, and retains exact frozen behavior without `LZ`. It is not a claim that every original B4 gate is satisfied.','',
        '## Parity evidence and limits','',
        '- Six no-token cases have byte-identical complete returned outputs against frozen v3. Eight ordinary fixed-size cases match paired/nonpaired outputs exactly. Two variable-size v3 cases produce resize events and differ, as required. Full output hashes are saved; two additional fixed-size rejection cases intentionally differ for the correction above.',
        '- Tests include partial and whole pair fills, late cancelled-order fills, front-member rotation, pending creates, concurrent replacement, minimum notional after lot rounding, short residuals, reduce-only shrinkage, chronological paired fills and unsupported-mode refusals. Tracing does not change the returned results.',
        '- The no-token and fixed-size comparisons are source-only synthetic function-output tests. Historical parquet/file-output parity was not run. The existing fill/queue model remains uncalibrated, so simulated profits are not presented as optimization evidence.',
        '- Kernel local-refusal/capacity behavior, the deployment late-fill reservation window and actual50ms timer scheduling are outside this B4 check. Native and research guard/prediction integration is not independently validated by feeding already-computed desired quotes.',
        '- Frozen v3 remains SHA256 `'+builder_hash()+'`. No live deployment, dollar loss-budget change or protected Nov9–13 read.','',
        'Sources805d467a and declarationd0cd457b precede corrected outcomes. `LZ` is implementation work, not another strategy parameter sweep; the prior cumulative count stays21.','',
        'Build: `python3 -m predictfun.research.pf_ladder_pairs_build`. Validate with `pf_pairs_validation` stages and `pf_pairs_receipt`, under `run_light.sh`; declarations and destinations refuse overwrite. The research loop checkout is `/tmp/cta-opt-loop-20261109` on `research/predict-opt-loop-20261109`. Receipts: `reviews/optimization_loop_20261109/paired_engine_v2/`; combined index: `findings_v5.json`.','',
        'Next: combine the bounded quote-continuity policy with this paired engine in synthetic lifecycle/inventory tests. Economic claims remain gated on calibrated fills or authorized forward evidence.']
    (s.a.HERE.parent/'handoffs/20261109_optimization_loop_v5.md').write_text('\n'.join(lines)+'\n')
    p=review.parent/'plan.json';plan=json.loads(p.read_text());plan.update(status='active_between_passes',as_of=result['as_of'],findings='findings_v5.json')
    plan['loop'].append(dict(round='paired_engine',status='implemented_with_explicit_parity_exception',receipt='paired_engine_v2/receipt.json',verification='paired_engine_v2/verification.json',
        finding='Native synthetic mapping/lifecycle parity passes after correcting rejected-create slot release underLZ. No-token outputs exact; blanket fixed-size parity contradicted by that correction; historical output-file and economic validation remain.'))
    plan['next_pass']=dict(status='planned',focus=combined['next'][0],data_constraint='synthetic controls only unless separately declared consumed-data work; protected Nov9–13 unread',economic_gate='no PnL/variance claim without calibrated fills or authorized forward evidence')
    plan['next_evidence']=combined['next'];p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(totals=totals,coverage=coverage,parity_cases=len(parity),native_replay_exact=True),indent=2))

def builder_hash():return study.builder.FROZEN

if __name__=='__main__':main()
