"""Presentation only: join all fixed signed orders to frozen timing outputs."""
import collections
import json
import statistics
import build as B
import check as C

def main():
 out=B.HERE/'outputs'
 ps=B.rows(out/'post_estimates.csv.gz');es=B.rows(out/'ending_estimates.csv.gz');us=B.rows(out/'union_windows.csv.gz')
 pd={(r['day'],int(r['mid']),r['maker'],r['order_hash']):r for r in ps}
 ed={(r['day'],int(r['mid']),r['maker'],r['order_hash'],r['view']):r for r in es}
 ud={(r['day'],int(r['mid']),r['maker'],r['order_hash']):r for r in us}
 rows=[]
 for s in B.rows(B.INPUTS['signed']):
  k=B.key(s);p=pd[k];e=ed[(*k,'S_FIFO')];n=ed.get((*k,'S_NOPRIO'));u=ud[k];ws=int(p['ws_ms'])
  enabled=C.truth(e['pull_estimate_enabled']) and n is not None and C.truth(n['pull_estimate_enabled']) and u['union_status']=='three_view_union'
  lo=B.num(u['union_lo_ms']);hi=B.num(u['union_hi_ms'])
  r=dict(day=k[0],mid=k[1],rank=s['rank'],maker=k[2],order_hash=k[3],signed_shares=s['signed_quote_shares'],observed_shares=s['observed_shares'],signed_ending=s['status'],
   post_grade=p['grade'],post_estimate_class=p['estimate_class'],post_lo_s=(int(p['effect_lo_ms'])-ws)/1000 if C.truth(p['estimate_enabled']) else None,
   post_mid_s=(float(p['midpoint_ms'])-ws)/1000 if C.truth(p['estimate_enabled']) else None,post_hi_s=(int(p['effect_hi_ms'])-ws)/1000 if C.truth(p['estimate_enabled']) else None,
   ending_class_fifo=e['ending_class'],ending_class_noprio=n['ending_class'] if n else 'missing',conditional_withdrawal_all_views=enabled,
   pull_effect_lo_s=(lo-ws)/1000 if enabled else None,pull_effect_mid_s=((lo+hi)/2-ws)/1000 if enabled else None,pull_effect_hi_s=(hi-ws)/1000 if enabled else None,
   pull_width_s=(hi-lo)/1000 if enabled else None,expiry_relation=e['expiry_relation'],successor_hash=e['successor_hash'],successor_kind=e['successor_kind'],
   successor_order_fifo=e['successor_link'],successor_gap_lo_s=e['gap_lo_s'],successor_gap_hi_s=e['gap_hi_s'],
   post_btc_5s_set=p['btc_5s_set'],post_inventory_set=p['inv_set'],**{f:e[f] for f in B.CONTEXT})
  rows.append(r)
 path=out/'signed_chain_estimates.csv';B.need(not path.exists(),'write once rendering');B.write_csv(path,rows)
 good=[r for r in rows if r['conditional_withdrawal_all_views']]
 info=dict(signed_orders=len(rows),all_view_withdrawal_estimates=len(good),width_median_s=statistics.median([r['pull_width_s'] for r in good]) if good else None,
   width_le10s=sum(r['pull_width_s']<=10 for r in good),posting_grades_on_withdrawals=dict(collections.Counter(r['post_grade'] for r in good)),
   successor_on_withdrawals=dict(collections.Counter(r['successor_order_fifo'] for r in good)),
   posting_context_G2={field:dict(collections.Counter(r[field] for r in ps if r['grade']=='G2_bound_constrained')) for field in B.POST_CONTEXT},
   rendered_sha256=B.digest(path))
 B.dump(B.HERE/'readout.json',info);print(json.dumps(info,indent=2))

if __name__=='__main__':main()
