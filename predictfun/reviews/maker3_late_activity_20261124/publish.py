"""Publish verified descriptive tables and native-action annotations only."""
import json,resource,time
from pathlib import Path
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parents[2];U=R/'predictfun/reviews/maker_dashboard_20261124'
resource.setrlimit(resource.RLIMIT_CPU,(8,9))
S=json.loads((H/'summary.json').read_text());checks=json.loads((H/'checks.json').read_text());assert checks['passed']
t=pd.read_csv(H/'market3789484_takes.csv',float_precision='round_trip').sort_values('t')
c=pd.read_csv(H/'minute_activity.csv',float_precision='round_trip');c=c[c.scope.eq('market3789484')]
lines=['# Maker 3: late-market fills and active takes — 2026-11-24','',
'Verified source freeze: e0ca38715 (declaration 11b88a99e). Fixed Oct 23 rank-3 address `0x57FbC891E89184DA071AE9D97E745531cd7F718a`. Saved Oct 24–27 data only; descriptive consumed-day evidence.', '',
'In market **3789484**, the last two minutes account for **52.8% of traded shares** (4,300 / 8,143), **63.3% of chart fill legs** (202 / 319), and **89.3% of taker shares** (184 / 206). The concentration is mainly in the final minute. Seven active take records occur; 312 passive counterparty legs. Passive fills still supply **97.47% of total shares**. This is not a taker-dominated strategy in this market.','',
'| Market minute | Passive shares | Active taker shares | Passive legs / take records |','|---|---:|---:|---:|']
for minute in range(1,6):
    a=c[(c.minute==minute)&(c.role=='maker')].iloc[0];b=c[(c.minute==minute)&(c.role=='taker')].iloc[0]
    lines.append(f'| {minute-1}:00–{minute}:00 | {a.shares:,.1f} | {b.shares:,.1f} | {int(a.legs)} / {int(b.events)} |')
lines += ['', '**What a taker marker means.** The chart side is the actor’s YES-equivalent direction, not the counterparty book side. A taker bid increases YES-equivalent exposure; a taker ask decreases it. Here all seven native orders are **buys**: five buy NO, two buy YES. NO purchases at 90¢ plot as YES-equivalent asks at 10¢. They take liquidity; they are not passive sell orders. Binary complementary matching means the normalized display does not establish the native counterparty order type.', '',
'| Record upper time after open | Native action | Native price | Shares | YES-equivalent direction / price | Recorded net flow before → after |','|---|---|---:|---:|---|---:|']
for r in t.itertuples():
    lines.append(f'| {r.t:.3f}s | {r.native_action} | {r.price_native*100:.0f}¢ | {r.q:.3f} | {r.side} {r.price_yes*100:.0f}¢ | {r.recorded_net_before:+.1f} → {r.recorded_net_after:+.1f} |')
lines += ['',
'All seven takes oppose and reduce the absolute **zero-start captured trade-flow** balance at their record time. The three NO buys at 202.668–202.705s total 109.387 shares and move it from +553.4 to +444.0. At 248.630s, another 74.535-share NO purchase moves +621.5 to +547.0. They trim exposure; they do not flatten it. At 114.530–114.553s, two YES purchases instead trim negative exposure. Thus both directions occur, consistent with active rebalancing alongside passive fills.', '',
'This is a **compatibility statement, not an identified motive or wallet position**. The proxy excludes unknown initial balances, transfers, token fees and merges; timestamp order is settlement-upper order, not a reconstructed private decision clock. Same-time records are batched without invented within-batch order. All seven example takes are isolated at their exact record timestamp, but that does not eliminate execution-window uncertainty.', '',
'**Broader pattern:** across 934 markets with accepted live actor fills in Oct 24–27, only **30.2%** of shares and **37.0%** of taker shares occur in the last two minutes. Seventy of 934 markets have more than half their shares there. Per-day total-share fractions: Oct24 25.4%, Oct25 21.1%, Oct26 35.3%, Oct27 30.9%. Thus late concentration is real in 3789484 but is not its general majority-volume pattern. Maker3 stops recorded activity early on Oct27 as previously documented; absence of fills does not establish absence of quotes.', '',
'The live census has 128,803 allocated counterparty legs, 122,275 actor record/role events, and 2,747,290.891 shares. 9,562 taker events: 4,809 buy NO, 4,752 buy YES, one sells YES. The general active-flow proxy is not a clean private-inventory classifier: 5,589 / 9,562 events share their record timestamp with other actor activity. No global inventory-motive rate is inferred.', '',
'**Validity and exclusions:** all 1,152 registry rows retained in scope; 134 canonical source-status legs and 36 self legs excluded (counts separately reported), ten accepted legs fall after 300s, none before open. One final registry market is boundary-censored; poll-incomplete and other coverage statuses are preserved in summary.json. Minute bins are half-open on integer settlement milliseconds. This can differ from true execution-minute membership near boundaries; no decision-time hazard inference. The named market’s entire 319-leg [0,300) population also reconciles to the saved chart’s [0,298] population (no extra end legs). Marker density is not traded volume.', '',
'**Verification:** independent raw maker-leg reconciliation for all319 example legs, native records bound by signer/hash and exact record, all9,562 take aggregates reconciled, 60 minute cells independently rebuilt, recorded-flow batches independently summed, all319 chart legs matched. Four planted side/duplication/quantity/time corruptions refused. Initial membership guard found the undeclared Oct23 day in the shared file, then the epoch-float clock assertion refused; both were corrected before output. The first checker hit its20CPU-second guard while constructing tiny pandas groups; replacing that loop with sorted tuple batches preserved the checks and passed in ~2s. All attempts retained and charged.', '',
'**Next priority:** a narrowly declared comparison of the observed maker3 active-rebalancing states with our simultaneous inventory and executable quote prices in this existing market, before designing a taker rule. Verify whether our problem is wrong-side exposure or expensive completion, using longer-horizon whole-book accounting; do not imitate every NO buy or add a fixed last-two-minute rule. No replay or strategy change in this packet.','']
report='\n'.join(lines);(H/'report.md').write_text(report);(U/'maker3_late_report.md').write_text(report)
# This annotation is separate from both immutable chart payloads.
te=pd.read_parquet(R/'data/predict_fun/research/maker3_late_activity_20261124/take_events.parquet')
note={}
for r in te[te.mid.isin([3702960,3789484,3877472])].itertuples():
    note[f'{r.mid}|{r.sid}']=dict(action=r.native_action,price=100*r.price_native,before=r.recorded_net_before,after=r.recorded_net_after,mixed=not r.isolated_record_time)
script='<script>window.MAKER3_TAKES='+json.dumps(note,separators=(',',':'),allow_nan=False)+';</script>'
for filename in ['index.html','opening.template.html']:
    p=U/filename;s=p.read_text();assert 'window.MAKER3_TAKES' not in s
    s=s.replace('</head>',script+'</head>')
    s=s.replace('</main>','<p><a href="maker3_late_report.md">Maker #3: all seven takes, native buy/sell direction and activity by minute</a></p></main>')
    p.write_text(s)
p=U/'app.js';s=p.read_text();old="+kv('This fill',fmt(f.q,3)+' shares')";assert s.count(old)==1
new="+(f.rank===3&&f.role==='taker'&&window.MAKER3_TAKES?.[m.id+'|'+f.sid]?(()=>{const n=window.MAKER3_TAKES[m.id+'|'+f.sid];return kv('Native active action',n.action+' at '+fmt(n.price,0)+'¢')+kv('Recorded net flow',fmt(n.before,1)+' → '+fmt(n.after,1)+' sh')+'<p>YES-equivalent side; this is not wallet inventory.'+(n.mixed?' Other actor activity shares this record timestamp.':'')+'</p>';})():'')"+old
s=s.replace(old,new);p.write_text(s)
p=U/'opening.js';s=p.read_text();old="if(e.kind==='maker'){h+=";assert s.count(old)==1
new="if(e.kind==='maker'){const n=e.role==='taker'?window.MAKER3_TAKES?.[m.id+'|'+e.sid]:null;if(n)h+=`<p><b>Native active action: ${n.action} at ${num(n.price,0)}¢.</b> Recorded net flow ${num(n.before,1)} → ${num(n.after,1)} shares. This is not wallet inventory.${n.mixed?' Other actor activity shares this record timestamp.':''}</p>`;h+="
s=s.replace(old,new);p.write_text(s)
for filename in ['index.html','opening.template.html']:
    p=U/filename;s=p.read_text();s=s.replace('YES price in cents','YES-equivalent direction and cents')
    s=s.replace('◆ maker · × taker','◆ passive · × active taker (YES-equivalent side)')
    s=s.replace('◆ Maker #3 passive · × taker','◆ Maker #3 passive · × active taker (YES-equivalent side)');p.write_text(s)
print('Report and annotations published:',len(note),'native take records; underlying chart payloads unchanged')
