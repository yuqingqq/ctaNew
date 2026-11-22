"""Summarize completed review evidence and verify the receipt's post-run counts.
No new economic subgroup, candidate or replay is evaluated.
"""
from pathlib import Path
import json,re
import pandas as pd
O=Path(__file__).resolve().parent;R=Path('/tmp/pf-aq-review-75e9531d/predictfun/reviews/ladder_arrival_queue_20261122')
T=pd.read_csv(R/'outputs/cells.csv.gz',dtype={'day':str,'side':str,'band':str},float_precision='round_trip')
q=T[(T.view=='queue')&(T.cohort=='common')&(T.interval=='union')]
byday={}
for h in (1000,30000):
 byday[str(h)]={d:'<'.join(g.sort_values(f'h{h}_per_share').band) for d,g in q[(q.day!='all')&(q.side=='all')].groupby('day')}
byside={s:'<'.join(g.sort_values('h1000_per_share').band) for s,g in q[(q.day=='all')&(q.side!='all')].groupby('side')}
want='beyond_R<within_R<front'
assert all(v==want for v in byday['1000'].values()) and sum(v==want for v in byday['30000'].values())==3 and all(v==want for v in byside.values())
E=json.loads((O/'evidence.json').read_text());rx=re.compile(r'cpu_user_s=([\d.]+)')
cpu=sum(float(m.group(1)) for p in O.glob('*.log') for m in rx.finditer(p.read_text()))
base=E['checks']['details']['I']['line_core_hours'];resource=dict(base_at_receipt_check=base,review_cpu_seconds_before_summary=cpu,cumulative_before_summary=base+cpu/3600,remaining_before_summary=2-base-cpu/3600)
assert resource['cumulative_before_summary']<2
support=pd.read_csv(O/'support.csv');s=support[(support.view=='queue')&(support.interval=='union')]
out=dict(day_order=byday,side_order_1s=byside,union_counts=s.to_dict('records'),resource=resource,
         corrected_front_empty=E['front_occupancy']['currently_empty'],corrected_front_shared=E['front_occupancy']['currently_displayed_positive'])
(O/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
