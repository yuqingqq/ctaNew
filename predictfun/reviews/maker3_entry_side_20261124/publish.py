import csv,gzip,hashlib,json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
x=json.loads((H/'result.json').read_text());s=x['summary'];c=json.loads((H/'checks.json').read_text());pc=json.loads((H/'post_checks.json').read_text())
assert c['raw_first_records_and_public_books_verified']==934 and c['corruptions_refused']==6 and pc['markets']==17
raw=(H/'result.json').read_bytes();(H/'result.json.gz').write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
assert gzip.decompress((H/'result.json.gz').read_bytes())==raw
timings=[]
for name in ['schema','selftest','run','check','post_check']:
    log=(H/(name+'.log')).read_text();wall=float(re.findall(r'wall_s=([0-9.]+)',log)[-1]);timings.append(dict(stage=name,wall_upper_s=wall))
cpu=sum(z['wall_upper_s'] for z in timings)+2
assert cpu<90
budget=dict(packet_cap_cpu_s=90,measured_stage_wall_upper_s=timings,publication_reserve_s=2,packet_upper_cpu_s=round(cpu,2),prior_dashboard_packet_upper_cpu_s=179.5,task_cap_coreh=2.075,line_cap_coreh=5.375,standing_reserve_coreh=.05,authorization='maker_trace_20261122/PENDING.md USER standing short-CPU authorization; declaration.md')
(H/'resources.json').write_text(json.dumps(budget,indent=2)+'\n')
p=s['pooled'];n=p['statuses']['observed_first_maker'];hi=p['fill_context']['higher_probability'];lo=p['fill_context']['lower_probability'];med=p['first_record_s_quantiles']['median']
table='\n'.join(f"|{d}|{s[d]['statuses'].get('observed_first_maker',0)}|{s[d]['fill_context'].get('higher_probability',0)}|{s[d]['fill_context'].get('lower_probability',0)}|{s[d]['prior_taker']}|" for d in ['20261024','20261025','20261026','20261027'])
report=f'''# Maker #3 opening side — 2026-11-24

The records do not establish a general higher-probability-outcome-first quoting rule. Among {n} markets with maker fills, the first recorded passive fill buys the higher-probability outcome in {hi} ({hi/n:.1%}) and the lower-probability outcome in {lo} ({lo/n:.1%}), judged against the nearby public book. This is a fill-associated observation, not a recovered submission rule. Both-side quoting with one side filling first remains compatible with it.

Frozen audit: 7a8b690c1. Population: all1,152 BTC5m registry markets on consumed Oct24–27; fixed Oct23 rank3 address `{x['address']}`. No replay, fit, new days or validation. All218 no-maker-fill markets retained, including the final Oct27 boundary-censored market. Maker activity in934 matches the preceding coverage census. Incomplete-poll flags remain attached to each market and do not become proof of no orders.

|UTC day|First maker record available|Higher-probability purchase|Lower-probability purchase|Earlier taker record|
|---|---:|---:|---:|---:|
{table}

Every first maker record in this population is clean in the existing source and a native `Bid`: buying YES or NO. A normalized YES-space ask can be a native NO buy, not an actual sale of an existing YES holding. The normalized sides are376 bids and558 asks; all same-millisecond first-record ties are retained and none crosses normalized side in this population. There can be multiple order hashes at the first record, so the posting-context counts below are order views, not934 independent decisions.

Higher probability means YES midpoint>0.5 or NO midpoint=1-YES midpoint>0.5. It does not mean positive recent BTC return, a favorable edge relative to a model, or the eventual winning outcome. Book context is the last frame available by settlement upper bound minus1ms, using both update and public receipt timestamps, with a1s freshness bound. All934 fill-associated contexts are valid. The maker's private feed and actual execution/posting times are not observed; these book states may already reflect the execution. The count is not a causal entry test.

At the opening itself,275 first-filled directions agree with the higher-probability outcome and560 with the lower-probability outcome;95 books are neutral and4 lack a usable prior frame. This deliberately earlier reference is also not the unknown posting state. In3789484, for example, the opening midpoint is49.5%, but the first buy's later inferred window is above50%. The label can change before the order is placed or filled.

First passive fill settlement upper bounds have median{med:.3f}s after opening (middle50%:1.966–4.378s). This is NOT a delay-to-post estimate: an order could rest before it trades, and settlement is an upper bound on execution. No first maker record is pre-open. In233/934 markets a taker record precedes the first maker record, so the first passive fill is not always the maker's first economic activity. A supplemental, post-run first-any-role recount gives378 higher-probability directions and556 lower-probability directions, but inherits the same fill-clock and selection limitations; it is not the primary frozen measurement.

## What the posting bounds add

Only17 of the934 markets have a posting estimate for their first-filled order in the existing depth-supported packet. Comparing the earliest-filled bid and ask orders in those markets gives2 bid-before-ask and3 ask-before-bid conditional orderings,4 overlapping windows and8 refused/missing comparisons. These are two selected filled orders, not all posted orders; other later-filled or never-filled quotes can have been posted first. This is insufficient to identify a general first-quote side.

Among19 first-record order views with existing estimates,4 are refused; the15 enabled views have4 stable higher-probability windows,6 stable lower-probability windows and5 changing/neutral windows. Inherited grades are preserved, including weaker G4 scenarios. Ranges describe sampled public book states throughout each window and retain missing/stale gaps; they are not the maker's known decision features.

In the previously selected market3789484:

- First observed passive order buys YES at56c, record upper bound1.755s; posting window1.038–1.755s is **G4 scenario-only**. The sampled available book midpoint ranges55.5–60%, so this example is consistent with buying the favored outcome at that time.
- The earliest-filled normalized ask is a NO purchase equivalent to selling YES at58c, posting window5.239–7.428s (G2 conditional). It is a later quote under those particular bounds.
- Other ask hashes have overlapping early posting windows. This does not establish that all asks were withheld initially, a single-sided opening, a reaction to the first fill, or a general favorite-first policy.

## Checks, resources and next action

`check.py` independently reloads the raw leg and book artifacts and verifies934 first-record memberships, native-side normalization implications, first-side labels, all as-of book joins and every primary count; six corruptions are refused. `check_posts.py` independently checks17 market joins,39 enabled interval views (including repeated views of the same order), original side-first record times, context ranges and paired interval orderings. Source hashes are in `result.json.gz`; synthetic future-frame, stale-gap, neutral and normalization controls passed. This does not reconstruct private maker orders.

Packet upper{cpu:.2f}/90CPU seconds including publication; standing short-CPU authorization, task2.075/line5.375 caps with.05reserve. Previous dashboard179.50CPU seconds stays charged. All scripts through PF_SLOT=codex/run_light. Existing strategy and other sessions' changes preserved; no push.

ONE next priority: retain the current opening-only experiment scope and compare its opening prices/first completed cycles against maker3's observed orders on the three existing markets, especially the unchanged losing market3877472. Separate relative price and favorable movement from absolute probability>50%; do not add a favorite-first rule on the strength of these fill counts. This audit alone neither validates nor rejects that proposed strategy rule.
'''
(H/'report.md').write_text(report)
with (H/'market_summary.csv').open('w') as f:
    cols=['day','mid','status','first_record_s','first_side','native_first','fill_context_direction','opening_context_direction','prior_taker','bid_ask_post_relation']
    w=csv.DictWriter(f,fieldnames=cols,lineterminator='\n');w.writeheader();w.writerows({k:r.get(k,'') for k in cols} for r in x['rows'])
subprocess.run(['git','diff','--cached','--quiet'],cwd=R,check=True)
block=f'''<!-- MAKER3_ENTRY_SIDE_20261124_CODEX -->
## 2026-11-24 — Maker #3 first-fill side does not identify a favorite-first entry rule

[Receipt](reviews/maker3_entry_side_20261124/report.md), freeze7a8b690c1. All1152 consumed Oct24–27 BTC5m markets;934 have maker3 first passive records. Higher-probability outcome purchases408/934 (43.7%), lower526/934 (56.3%) at nearby public book; all native Bid YES/NO, not necessarily sells of holdings. First passive record median2.858s after opening is not posting latency.233 markets have earlier takes. Opening-reference275higher/560lower/95neutral/4missing. First-fill selection and settlement-upper/public-clock caveats prevent recovery of a first-quote rule.

Only17 first-filled markets have graded post intervals:2bid-before-ask/3ask-before-bid/4overlap/8refused for the first-filled side orders. Never-filled and later-filled earlier quotes remain unknown.3789484's first56c YES buy is consistent with favored-side entry (book55.5–60%) but its1.038–1.755s posting window is G4 scenario-only. No general rule/profit claim.934 raw first-record/book checks and17 posting comparisons pass;6 corruptions refused. New bounded packet upper{cpu:.2f}/90CPU seconds under documented standing short-CPU approval; task2.075/line5.375/.05reserve, prior dashboard179.50s unchanged.

ONE next priority: existing three-market opening-only cycle/relative-price comparison, especially3877472. Keep probability>50 separate from favorable movement/quote edge; no favorite-first rule, expanded entry filter or new replay authorized by this census alone.
<!-- /MAKER3_ENTRY_SIDE_20261124_CODEX -->

'''
state=f'''maker3_entry_side_20261124_codex:
  report: predictfun/reviews/maker3_entry_side_20261124/report.md
  freeze: 7a8b690c1
  markets_with_first_maker_record: 934
  higher_probability_first_fill: 408
  lower_probability_first_fill: 526
  private_first_quote_identified: false
  earlier_taker_markets: 233
  post_interval_markets: 17
  packet_upper_cpu_s: {cpu:.2f}
  packet_cap_cpu_s: 90
  task_cap_coreh: 2.075
  line_cap_coreh: 5.375
  standing_reserve_coreh: 0.05
'''
for rel,text in [('predictfun/HANDOFF.md',block),('predictfun/FINDINGS.md',block),('predictfun/STATUS.yml',state)]:
    p=R/rel;working=p.read_text();base=subprocess.check_output(['git','show','HEAD:'+rel],cwd=R).decode()
    assert 'MAKER3_ENTRY_SIDE_20261124_CODEX' not in working and 'maker3_entry_side_20261124_codex:' not in working
    p.write_text(text+working)
    oid=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=R,input=(text+base).encode()).decode().strip()
    subprocess.run(['git','update-index','--cacheinfo','100644',oid,rel],cwd=R,check=True)
print('Published checked descriptive receipt and staged only owned state; packet upper',cpu)
