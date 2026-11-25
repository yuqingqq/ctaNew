"""Render the frozen metrics and descriptive attribution after independent verification."""
import csv
from collections import Counter,defaultdict
from pathlib import Path

import trial as T

H=T.HERE
d=T.S.read(H/'results.json')
verified=T.S.read(H/'verification.json')
assert verified['checks_pass'] and verified['markets']==d['market_count']
a,b=d['arms']['reference'],d['arms']['guard']
g=d['flagged_reference_exposure']
guard=d['counts']['guard']
reference=d['counts']['reference']
def dollar(x):
    return f"${abs(x):,.2f}" if x>=0 else f"−${abs(x):,.2f}"
def signed(x):
    return ('+' if x>0 else '')+dollar(x)
def table(r):
    return f"| {dollar(r['settle'])} | {dollar(r['turnover'])} | {r['cents_per_matched']:.3f} | {dollar(r['residual'])} | {r['closed']:,.0f} | {r['mean_end_abs']:.1f} | {100*r['both']:.1f}% |"
# A breakdown of the declared fixed-path diagnostic, retaining unfunded, refused and unfilled slots.
by=defaultdict(Counter)
for mid in verified['part_sha256']:
    raw=T.S.read(T.OUT/f'{mid}.json')
    for _,key,val in raw['diag']['reference']['flagged']:
        x=by[key[0]]
        x.update(val)
        x['n']+=1
        x['with_fill']+=int(val.get('post_effect_shares',0)>0)
lines=[
'# Passive Binance depth depletion guard',
'',
f"The frozen economic decision is **{d['decision']}**. All {d['market_count']} complete October 26 markets reproduced the reference before the candidate comparison. The 288-market registry also includes 124 missing the endpoint and 94 with stale books. October 26 was already consumed. Values are simulator point estimates as of November 25, with zero maker fees and rebates.",
'',
'| Policy | Settlement | FIFO completed | ¢/matched share | Residual at settlement | Matched shares | Mean absolute ending shares | Both sides active |',
'|---|---:|---:|---:|---:|---:|---:|---:|',
'| Straight-line reference '+table(a),'| With the depletion guard '+table(b),'',
f"Delta in settlement {signed(d['delta'])}, completed turnover {signed(b['turnover']-a['turnover'])}, residual {signed(b['residual']-a['residual'])}. Settlement improved in {d['better']} markets, worsened in {d['worse']} and tied in {d['equal']}. Dropping the largest improvement (market {d['largest_gain_mid']}) leaves {signed(d['without_largest_gain'])}. Dropping the largest absolute change leaves {signed(d['without_largest_absolute'])}.",
'',
'## Orders and queue exposure','',
f"The candidate issued {guard.get('live_flagged_ticks',0):,} incremental live pulls. These orders had accumulated {d['age_at_pressure_depletion_pull_ms_sum']/1000:,.1f} order-seconds since activation (mean {d['age_at_pressure_depletion_pull_ms_sum']/max(1,guard.get('live_flagged_ticks',0))/1000:.2f} s per pull). It withheld {guard.get('withheld',0):,} proposed slots and clipped {guard.get('clipped',0):,} proposals to their closing quantity. Withheld slots count decisions, so one vacancy can appear repeatedly. Total cancels from all reasons were {a['all_pulls']:,.0f} and {b['all_pulls']:,.0f}; submissions were {a['posts']:,.0f} and {b['posts']:,.0f}. The signal was unknown on {d['signal_unknown_side_ticks']:,} of {d['side_ticks']:,} side-ticks.",
'',
f"On the reference path, {reference['live_eligible']:,} purely adding live decisions and {reference['slot_eligible']:,} otherwise eligible adding or mixed slot decisions were examined. On {reference.get('old_pull_and_on',0):,} live callbacks and {reference.get('old_slot_pressure_and_on',0):,} slot callbacks the existing pressure rule already acted while depletion was on. The candidate protected mixed resting orders on {guard.get('protected_live_mixed',0):,} flagged callbacks. Age and remaining rest are durations; neither measures an executable price for queue priority.",
'',
'## First incremental flag per purely adding reference generation','',
'This attributes the baseline fills after a feasible cancellation or withheld entry. The same order can remain profitable later. Zero-fill generations stay in the denominator. The full-policy table above also includes mixed-size changes, losing queue position, re-entry, subsequent fills and other-side feedback. Dollar amounts below are signed fill contributions.',
'',
'| Action at first flag | Generations | With a later eligible fill | Shares at/after effect | Eligible +1 s | Eligible +5 s | Eligible +30 s | At settlement |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for k,x in sorted(by.items()):
    lines.append(f"| {k} | {x['n']:,} | {x['with_fill']:,} | {x['post_effect_shares']:,.1f} | {dollar(x['value_1'])} | {dollar(x['value_5'])} | {dollar(x['value_30'])} | {dollar(x['settle'])} |")
lines+=['',f"On the flagged reference generations, {g.get('already_matched_shares',0):,.1f} shares had already matched and another {g.get('inflight_shares',0):,.1f} matched in flight. Flagged live orders had {g.get('remaining_rest_ms',0)/1000:,.1f} cumulative order-seconds of feasible remaining exposure. After the effect, positive and negative settlement contributions were {dollar(g.get('positive_settle',0))} and {dollar(g.get('negative_settle',0))}.",
'','| Horizon | Valued shares | Censored shares | Favorable fill value | Adverse fill value |',
'|---|---:|---:|---:|---:|']
for h in (1,5,30):
    lines.append(f"| +{h} s | {g.get(f'valid_sh_{h}',0):,.1f} | {g.get(f'censored_sh_{h}',0):,.1f} | {dollar(g.get(f'positive_{h}',0))} | {dollar(g.get(f'negative_{h}',0))} |")
lines+=['',f"The matched settlement comparison covered {d['matched_n']:,} of {d['unique_first_flags']:,} first flags. Selected contributions were {dollar(d['matched_exposure'].get('settle',0))}, against {dollar(d['null_settlement_draw_mean'])} on average in 500 draws (5th percentile {dollar(d['null_p05'])}; observed percentile {100*d['percentile']:.1f}%). This compares reference exposure, conditional on support. A single day's simulated queue, receipt clock and fixed external trades limit interpretation.",
'','## Reproduction','',
'Candidate and score frozen in f3b9a44e4, with pre-score reservation correction 89cbd0ff8. The first four partial records were excluded. The independent checker was committed before scoring. Its first invocation stopped on an incomplete synthetic branch stub before checking any outputs. Providing the empty order list on that stub let the original assertions run. The replay and scorer were unchanged. Re-run the test and replay using the Codex run_light slot with trial.py, then score, verify.py and report.py. Raw partitions and hourly Binance input hashes are bound by the receipts.',
'','[Protocol](protocol.md) · [Per-market outcomes](markets.csv) · [Frozen readings](results.json) · [Verification and partition hashes](verification.json)']
(H/'report.md').write_text('\n'.join(lines)+'\n')
