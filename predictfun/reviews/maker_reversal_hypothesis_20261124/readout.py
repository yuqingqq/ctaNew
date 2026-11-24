"""Post-run chronology-only join; all original outcomes retained, no new fitted rule."""
from collections import Counter
import build as B

def main():
 B.bound();out=B.HERE/'outputs';B.need(not (out/'chronology.csv').exists(),'write once')
 p={B.key(r):r for r in B.rows(B.FILES['post'])};cases=B.rows(out/'case_alignment.csv');signed={B.key(r):r for r in B.rows(B.FILES['signed'])};records=[]
 for r in cases:
  if r['group']!='conditional_withdrawal' or r['mode']!='timing_only' or r['control']!='correct_side' or not B.truth(r['matched']):continue
  q=p[B.key(r)];s=signed[B.key(r)];ws=int(q['ws_ms']);lo=B.number(q['effect_lo_ms']);hi=B.number(q['effect_hi_ms'])
  for t in map(int,r['reversal_times_ms'].split('|')):
   state='post_unknown' if not B.truth(q['estimate_enabled']) else ('before_earliest_post' if t<lo else ('after_latest_post' if t>=hi else 'inside_post_window'))
   records.append(dict(day=r['day'],mid=r['mid'],rank=r['rank'],maker=r['maker'],order_hash=r['order_hash'],side=r['side'],delay_ms=r['delay_ms'],grade=q['grade'],
    reversal_s=(t-ws)/1000,post_lo_s=(lo-ws)/1000,post_hi_s=(hi-ws)/1000,end_lo_s=(int(r['lo_ms'])-ws)/1000,end_hi_s=(int(r['hi_ms'])-ws)/1000,
    chronology=state,ending_alignment=r['status'],signed_shares=s['signed_shares'],observed_shares=s['observed_shares']))
 B.write(out/'chronology.csv',records,['day','mid','rank','maker','order_hash','side','delay_ms','grade','reversal_s','post_lo_s','post_hi_s','end_lo_s','end_hi_s','chronology','ending_alignment','signed_shares','observed_shares'])
 # Independent order of intervals and identity verification for every exported event.
 for r in records:
  q=p[(r['day'],int(r['mid']),r['maker'],r['order_hash'])];t=r['reversal_s']*1000+int(q['ws_ms'])
  if r['chronology']=='inside_post_window':B.need(float(q['effect_lo_ms'])<=t<float(q['effect_hi_ms']),'inside')
  elif r['chronology']=='before_earliest_post':B.need(t<float(q['effect_lo_ms']),'before')
  elif r['chronology']=='after_latest_post':B.need(t>=float(q['effect_hi_ms']),'after')
 signal=B.rows(out/'signal_census.csv');summary=B.rows(out/'summary.csv')
 facts=dict(as_of_utc=B.time.strftime('%Y-%m-%dT%H:%M:%SZ',B.time.gmtime()),post_run_chronology_only=True,
  reversal_count_by_day={d:sum(int(x['reversals']) for x in signal if x['day']==d) for d in sorted({x['day'] for x in signal})},
  all_scope_summaries=[r for r in summary if r['day']=='all'],
  chronology_by_delay={str(d):dict(Counter(x['chronology'] for x in records if int(x['delay_ms'])==d)) for d in (0,1000)},
  chronology_orders_by_delay={str(d):len({(r['day'],r['mid'],r['maker'],r['order_hash']) for r in records if int(r['delay_ms'])==d}) for d in (0,1000)},
  chronology_sha256=B.digest(out/'chronology.csv'))
 B.dump(B.HERE/'readout.json',facts);print(B.json.dumps(facts,indent=2))
if __name__=='__main__':main()
