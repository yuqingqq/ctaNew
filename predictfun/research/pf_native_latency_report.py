"""Link verified native latency results into the optimization-loop register."""
import json
from . import pf_native_a2_latency as study
from .pf_native_a2_report import comparisons

s=study.s

def linked(expected,v):
    if v['receipt_sha256']!=expected or v['window_arm_profile_rows']!=1980 or v['action_arm_profile_rows']!=117693:
        raise ValueError('latency verification identity/population mismatch')
    for k in ['base_output_byte_identical','base_metrics_exact','action_lifecycle_capacity_verified','coverage_rebuilt_from_saved_lifecycles']:
        if v[k] is not True:raise ValueError('latency verification incomplete')

def main():
    study.check();dest=study.DEST;review=study.REVIEW
    r=json.loads((dest/'receipt.json').read_text());v=json.loads((dest/'verification.json').read_text())
    linked(s.sha(dest/'receipt.json'),v)
    try:linked('wrong',v)
    except ValueError:wrong_receipt_refused=True
    else:raise ValueError('wrong receipt accepted')
    bad=dict(v,window_arm_profile_rows=1979)
    try:linked(s.sha(dest/'receipt.json'),bad)
    except ValueError:wrong_population_refused=True
    else:raise ValueError('wrong population accepted')
    for p,h in r['artifacts'].items():
        if s.sha(p)!=h:raise ValueError('latency artifact changed')
    for name in ['build','run','verify']:
        if 'rc=0 ' not in (dest/f'logs/{name}.log').read_text():raise ValueError('numeric job failed')
    previous=review.parent/'findings_v3.json';old=json.loads(previous.read_text())
    comp={p:comparisons(q['totals']) for p,q in r['summary'].items()}
    report=dict(as_of=s.now(),source_commit=s.command(['git','rev-parse','HEAD'],cwd=s.a.HERE),source_sha256=s.sha(s.a.HERE/'pf_native_latency_report.py'),
        supersedes='findings_v3.json',supersedes_sha256=s.sha(previous),reason='Add native faster/slower venue-feedback sensitivity; base output and metrics exactly reproduce the previous pass.',
        previous_four_passes=old,native_latency=r,comparisons=comp,
        receipt_sha256=s.sha(dest/'receipt.json'),verification_sha256=s.sha(dest/'verification.json'),
        controls=dict(correct_link_accepted=True,wrong_receipt_refused=wrong_receipt_refused,wrong_population_refused=wrong_population_refused),
        next=['B4 generated research-engine integration of native paired-order semantics, with frozen no-token parity and synthetic native differential traces.',
              'Actual timer scheduling and calibrated fills/authorized forward evidence before valuing profits or inventory variance.'])
    s.write(review.parent/'findings_v4.json',report);s.write(dest.parent/'findings_v4.json',report)
    lines=['# Optimization loop: native latency sensitivity — 2026-11-09','',
        f"As-of **{report['as_of']}**. **220 consumed Nov8 windows**, **279,059 decisions per profile**, **13,077 neutral pull actions per arm/profile**; zero exclusions. One partial day, no confidence intervals or independent validation. Each latency profile has its own paired reference.",'',
        'This fifth pass changes only assumed venue feedback delays. The policies, native A2 execution code, factual intent path, capacity and reservation grace are unchanged.','',
        '| Latency (create/cancel/pending ms) | Venue request change | Original aligned time change | Off-target time change | Active share-time change | Active range expansion change |',
        '|---|---:|---:|---:|---:|---:|']
    for p in ['faster','base','slower']:
        d=r['summary'][p]['change_pct']['update_first50'];l=study.LATENCIES[p]
        lines.append(f"| {p} ({l[0]:g}/{l[1]:g}/{l[2]:g}) | {d['venue_requests']:+.2f}% | {d['aligned_lane_ms']:+.2f}% | {d['elsewhere_lane_ms']:+.2f}% | {d['active_share_ms']:+.2f}% | {d['active_range_extra_share_ms']:+.2f}% |")
    lines+=['',
        f"Computed predicate—fewer venue requests and more original-price quote time than reference, with less off-target time than100ms—in all three profiles: **{r['all_profiles_primary_predicate']}**. The same predicate separately in both captured modes and each profile: **{r['all_run_profiles_primary_predicate']}**.",
        'These are assumed stress cases, not measured latency quantiles or additional days. The changes do not establish higher spread capture, settlement P&L, favorable inventory or lower variance.','',
        '| Profile | Reference coverage | 50 ms coverage | 100ms aligned gain retained | Additional off-target time removed vs100ms |',
        '|---|---:|---:|---:|---:|']
    for p in ['faster','base','slower']:
        t=r['summary'][p]['totals'];c=comp[p]
        lines.append(f"| {p} | {t['reference']['coverage_pct']:.2f}% | {t['update_first50']['coverage_pct']:.2f}% | {c['coverage_gain_retained_vs100_pct']:.2f}% | {c['extra_off_target_removed_vs100_pct']:.2f}% |")
    lines+=['','## All three policies','',
        '| Profile | Policy | Venue requests | Aligned lane-seconds | Off-target lane-seconds | Active share-seconds | Reserved share-seconds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for p in ['faster','base','slower']:
        for arm in s.ARMS:
            t=r['summary'][p]['totals'][arm]
            lines.append(f"| {p} | {arm} | {t['venue_requests']:,.0f} | {t['aligned_lane_ms']/1000:,.2f} | {t['elsewhere_lane_ms']/1000:,.2f} | {t['active_share_ms']/1000:,.2f} | {t['reserved_share_ms']/1000:,.2f} |")
    lines+=['','## Verification and limits','',
        '- The base native output is byte-identical to the previous saved experiment before either stress case is run. Every base metric is also exactly equal. This isolates the specified latency change.',
        '- Nine Rust adapter controls and four Python checks pass. Controls exercise all three create/cancel clocks, pending cancellation, rejection/backoff, paired partial fills, same-clock reopening, later-reopen refusal, corrupt lifecycle/action identity, unknown profiles and changed base artifacts. Prior native library test evidence remains linked in findings_v3.',
        '- Saved artifacts verify across1,980 window-arm-profile rows and117,693 action-arm-profile rows. Coverage and exposure are independently rebuilt from saved order lifecycles; neutral actions, commands, capacity and deadlines reconcile. Wrong receipt and population links are refused.',
        '- All native venue tokens, 250–4000ms retries, three-unanswered-per-side limit,150-share external cap and1s external reservation grace are preserved. The share cap is distinct from the dollar loss budget.',
        '- Quote targets, guards and recorded inventory remain factual/exogenous. No hypothetical fills, post-only rejection model, wallet/kernel I/O or shared cross-market throttle are introduced. Full B4 integration and real timer scheduling remain separate.',
        '- No new policy parameters:21 cumulative distinct policy settings. This retests three existing policies under three assumed latency profiles. No live change, frozen-engine change, protected Nov9–13 read or fill-clock bypass.',
        '- Sources75bf977e, declarationb1441503 were committed before outcomes. Earlier input-preparation correction and all previous receipts remain preserved.','',
        'The loop continues with B4 paired-order integration and synthetic differential checks. Calibrated fills or authorized forward evidence remain necessary before claiming preserved trading gains or lower inventory variance.','',
        'Reproduction: `python3 -m predictfun.research.pf_native_a2_latency {build,declare,run,verify}`, then `pf_native_latency_report`; run numeric stages serially under `run_light.sh`. Destinations refuse overwrites. Register and full receipts: `reviews/optimization_loop_20261109/plan.json`, `native_latency/`, and superseding `findings_v4.json`.']
    (s.a.HERE.parent/'handoffs/20261109_optimization_loop_v4.md').write_text('\n'.join(lines)+'\n')
    path=review.parent/'plan.json';plan=json.loads(path.read_text());plan.update(status='active_between_passes',as_of=report['as_of'],findings='findings_v4.json')
    plan['loop'].append(dict(round='native_latency',status='complete',receipt='native_latency/receipt.json',verification='native_latency/verification.json',finding='Native base exact parity; faster/slower execution sensitivity completed. Full table and computed predicates in findings_v4; exposure costs remain explicit.'))
    plan['next_pass']=dict(status='planned',focus=report['next'][0],data_constraint='synthetic controls and already consumed bounded inputs only; protected Nov9–13 remain unread',economic_gate='no profit/variance promotion without calibrated fills or authorized forward evidence')
    plan['next_evidence']=report['next'];path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(comparisons=comp,all_profiles=r['all_profiles_primary_predicate'],all_run_profiles=r['all_run_profiles_primary_predicate']),indent=2))

if __name__=='__main__':main()
