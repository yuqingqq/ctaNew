"""Publish verified native A2 mechanics alongside the earlier research model."""
import json
import re
import subprocess
from . import pf_native_a2_continuity as s

def linked(receipt_hash,v):
    if v['receipt_sha256']!=receipt_hash or v['window_arm_rows']!=660 or v['neutral_action_rows']!=39231:
        raise ValueError('wrong receipt/population binding')
    if not all(v[k] for k in ['action_identity_exact','lifecycle_and_capacity_exact','coverage_rebuilt_from_saved_lifecycles']):
        raise ValueError('unverified native artifact')

def comparisons(totals):
    base=totals['reference'];long=totals['all100'];short=totals['update_first50']
    gain=long['aligned_lane_ms']-base['aligned_lane_ms'];extra=long['elsewhere_lane_ms']-base['elsewhere_lane_ms']
    return dict(coverage_gain_retained_vs100_pct=100*(short['aligned_lane_ms']-base['aligned_lane_ms'])/gain if gain>0 else None,
        extra_off_target_removed_vs100_pct=100*(long['elsewhere_lane_ms']-short['elsewhere_lane_ms'])/extra if extra>0 else None,
        short_fewer_venue_requests_more_aligned=short['venue_requests']<base['venue_requests'] and short['aligned_lane_ms']>base['aligned_lane_ms'],
        short_less_off_target_than100=short['elsewhere_lane_ms']<long['elsewhere_lane_ms'],
        short_lower_active_exposure_than_reference=short['active_share_ms']<base['active_share_ms'])

def main():
    s.check();r=json.loads((s.DEST/'receipt.json').read_text());v=json.loads((s.DEST/'verification.json').read_text())
    linked(s.sha(s.DEST/'receipt.json'),v)
    try:linked('wrong',v)
    except ValueError:refused=True
    else:raise ValueError('wrong receipt control accepted')
    for p,h in r['artifacts'].items():
        if s.sha(p)!=h:raise ValueError(f'changed artifact {p}')
    previous=s.REVIEW.parent/'findings_v2.json';old=json.loads(previous.read_text())
    metrics=s.pd.read_parquet(s.DEST/'window_metrics.parquet')
    by_run={run:g.groupby('arm').sum(numeric_only=True).drop(columns=['mid','peak_reserved_shares']).to_dict('index') for run,g in metrics.groupby('run')}
    logs={name:s.DEST/f'logs/{name}.log' for name in ['build','controls','python-controls','prepare','run','verify']}
    if not re.search(r'test result: ok\. 7 passed; 0 failed',logs['build'].read_text()):raise ValueError('adapter tests missing')
    if not re.search(r'test result: ok\. 51 passed; 0 failed',logs['controls'].read_text()):raise ValueError('native tests missing')
    if 'Ran 4 tests' not in logs['python-controls'].read_text() or '\nOK\n' not in logs['python-controls'].read_text():raise ValueError('Python controls missing')
    for name in ['prepare','run','verify']:
        if 'rc=0 ' not in logs[name].read_text():raise ValueError('failed numeric job')
    report=dict(as_of=s.now(),source_commit=s.command(['git','rev-parse','HEAD'],cwd=s.a.HERE),source_sha256=s.sha(s.a.HERE/'pf_native_a2_report.py'),
        supersedes='findings_v2.json',supersedes_sha256=s.sha(previous),reason='Add pinned native A2 execution comparison. Earlier research-model results remain valid within their stated simpler execution assumptions.',
        previous_three_rounds=old,receipt_sha256=s.sha(s.DEST/'receipt.json'),verification_sha256=s.sha(s.DEST/'verification.json'),
        native_a2=r,comparisons=comparisons(r['totals']),by_run={run:comparisons(q) for run,q in by_run.items()},
        controls=dict(native_tests=51,adapter_tests=7,python_tests=4,correct_receipt_accepted=True,wrong_receipt_refused=refused),
        log_sha256={str(p):s.sha(p) for p in logs.values()},
        correction=json.loads((s.DEST/'declaration.json').read_text())['supersedes'],
        next=['Repeat the native comparison under the already specified faster/slower latency assumptions, with exact base replay parity.',
              'Full B4 generated research-engine integration, including fills and paired-order traces, remains separate.',
              'Actual timer scheduling and calibrated fills/authorized forward evidence are still required for economic promotion.'])
    s.write(s.REVIEW.parent/'findings_v3.json',report);s.write(s.DEST.parent/'findings_v3.json',report)
    c=report['comparisons'];base=r['totals']['reference'];q=r['totals']['update_first50']
    lines=['# Optimization loop: native A2 execution comparison — 2026-11-09','',
        f"As-of **{report['as_of']}**. Same **220 consumed Nov8 windows**, **279,059 decisions**, **13,077 neutral touch-pull actions per arm**; zero excluded windows. One partial UTC day, no confidence intervals or independent validation.",'',
        'The quote-continuity benefit survives the current Rust A2 request mapper, venue throttle and retry backoff in this conditional no-fill test. The 50 ms policy still increases active exposure; it does not demonstrate lower inventory risk or improved P&L.','',
        '| Native A2 policy | Venue requests | Request change | Original aligned time change | Off-target time change | Active share-time change |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,name in [('reference','Reference'),('all100','100 ms timer first'),('update_first50','50 ms updates first')]:
        v=r['totals'][arm];d=r['change_pct'][arm]
        lines.append(f"| {name} | {v['venue_requests']:,.0f} | {d['venue_requests']:+.2f}% | {d['aligned_lane_ms']:+.2f}% | {d['elsewhere_lane_ms']:+.2f}% | {d['active_share_ms']:+.2f}% |")
    lines+=['',f"Original-price coverage rises from **{base['coverage_pct']:.2f}% to {q['coverage_pct']:.2f}%**. Updates-first50 retains **{c['coverage_gain_retained_vs100_pct']:.2f}%** of the 100 ms aligned-time gain and removes **{c['extra_off_target_removed_vs100_pct']:.2f}%** of its additional off-target time. It retains {q['retained_actions']:,.0f} pull actions and bridges {q['same_price_reopens']:,.0f} same-price reopens.",'',
        '## What changed in the execution model','',
        '- The isolated native checkout is pinned to `f20fdb1`. The adapter calls the original A2 pair mapper and request methods, using an exact copy of the native intent-change loop. Removing the diagnostic accessors restores the original source. No live checkout or deployment was changed.',
        '- A new quote is one whole order; size growth preserves the existing member and adds the remainder. Native request tokens, unanswered-order limits, and 250–4000 ms retry backoff are active.',
        '- Original quote targets and guards remain factual and exogenous. Probes may cancel a held pair, but cannot create a fresh quote. No captured fill is applied to a hypothetical order. Native decision counters are zero because this harness invokes the request mapper directly; the source population still contains all 279,059 decisions.',
        '- Venue feedback is assumed: 40 ms create/answer, 22 ms accepted cancel, at least 95 ms for pending-create cancellation, and 1 s external reservation grace. Capacity is 150 shares per side at hypothetical net zero, separate from the dollar loss budget. Recorded net only conditions exposure metrics.',
        '- Orders are physically active until cancel completion and reserve external capacity through the additional grace. Locally refused creates count as native requests but never reach the modeled venue. No post-only rejection, shared kernel throttle or wallet I/O model is added.','',
        '| Native counters | Reference | 50 ms updates first |','|---|---:|---:|']
    for key,name in [('native_requests','Native request outputs'),('venue_requests','Requests sent to venue'),('throttled','Native throttled attempts'),('throttled_locally','Local capacity refusals'),('retry_times','Retry timestamps executed')]:
        lines.append(f"| {name} | {base[key]:,.0f} | {q[key]:,.0f} |")
    lines+=['',
        f"The 50 ms arm changes reserved share-time by {r['change_pct']['update_first50']['reserved_share_ms']:+.2f}% but active inventory-range expansion by {r['change_pct']['update_first50']['active_range_extra_share_ms']:+.2f}%. Fewer reservations do not imply lower active risk. Both captured modes have the same request/coverage direction: **{all(z['short_fewer_venue_requests_more_aligned'] for z in report['by_run'].values())}**; they are not independent days.",'',
        '## Verification and provenance','',
        '- 51 existing native tests, seven adapter controls and four Python checks pass, covering paired partial fills/front rotation, request backoff and rejection reset, equal-clock versus late reopen, hard-probe cancellation, reservation release boundaries, malformed inputs and corrupt artifacts.',
        '- Saved output verifies across 660 window-arm rows and 39,231 action-arm rows. Action identities match the original input events. Coverage and exposure are independently rebuilt from saved order intervals; command counts, capacity and lifecycle deadlines reconcile.',
        '- The first preparation attempt refused because the adapter treated native `-1` off-quote ticks as prices. It produced no native experiment output. The corrected adapter and declaration were committed before outcomes, with a sentinel regression check. The failed declaration/input/log remain preserved and are superseded in-band by `native_a2_v2`.',
        '- Source commit `4776d822`, declaration `8274fc63`; cumulative distinct policy settings remain 21. This retests three existing policies in a new execution model. It does not validate the full B4 engine or a deployed timer scheduler.',
        '- Largest successful job: 758 MiB RSS; all numeric jobs complete. Protected Nov9–13, the frozen engine and loss-budget settings remain unchanged.','',
        'Earlier rounds and latency tests used the simpler research planner; retain their stated assumptions. The native comparison here uses only the base latency profile. Next: native faster/slower sensitivity with exact base parity, then full B4 integration and calibrated economic evidence.','',
        'Register: `reviews/optimization_loop_20261109/plan.json`. Combined superseding receipt: `findings_v3.json`. Native receipts: `native_a2_v2/`. Reproduction on the staged checkout: `pf_native_a2_continuity build`, `declare`, `prepare`, `run`, `verify`, then `pf_native_a2_report`, with numeric jobs under `run_light.sh`. Destinations are exclusive-create; preserved artifacts must not be overwritten.']
    (s.a.HERE.parent/'handoffs/20261109_optimization_loop_v3.md').write_text('\n'.join(lines)+'\n')
    p=s.REVIEW.parent/'plan.json';plan=json.loads(p.read_text());plan.update(status='active_between_passes',as_of=report['as_of'],findings='findings_v3.json')
    plan['loop'].append(dict(round='native_a2',status='complete',receipt='native_a2_v2/receipt.json',verification='native_a2_v2/verification.json',
        finding='Current native A2 preserves the quote-continuity benefit under base latency, with higher active exposure. No economic promotion. Input normalization refusal superseded before outcomes.'))
    plan['next_pass']=dict(status='planned',focus=report['next'][0],data_constraint='same consumed bounded Nov8 only; protected Nov9–13 remain unread',economic_gate='no profit/variance promotion without calibrated fills or authorized forward evidence')
    plan['next_evidence']=report['next'];p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(comparisons=c,by_run=report['by_run'],controls=report['controls']),indent=2))

if __name__=='__main__':main()
