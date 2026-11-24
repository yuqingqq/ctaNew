import build as B

def main():
 g=dict(grade='G2_bound_constrained',status_packet='ok',census_order_flags='ok')
 r=dict(removal_lo_ms='10000',removal_hi_ms='12000',x_size_ub='100',status='ok',window_status='ok',view='S_FIFO',x_rhi_S_FIFO_LO='13000')
 s=dict(status='unresolved_remainder',signed_quote_shares='100',expiration_s='20')
 assert B.end_class(r,g,s)['ending_class']=='conditional_withdrawal_before_expiry'
 assert B.end_class(r,g,{**s,'expiration_s':'11'})['ending_class']=='conditional_withdrawal_or_expiry'
 assert B.end_class(r,g,{**s,'expiration_s':'0'})['expiry_relation']=='unknown'
 assert B.end_class(r,g,{**s,'expiration_s':'9'})['expiry_relation']=='before_window'
 assert not B.end_class(r,g,{**s,'status':'exhausted_by_observed_fills'})['pull_estimate_enabled']
 assert B.end_class(r,g,{**s,'signed_quote_shares':'101'})['ending_class']=='signed_size_capacity_conflict'
 assert not B.end_class(r,{**g,'grade':'G1_conflict'},s)['pull_estimate_enabled']
 assert B.end_class(r,g,None)['ending_class']=='unsigned_exhaustion_or_withdrawal'
 p=B.point_fields(10000,12000,True)
 assert p['midpoint_ms']==11000 and p['submit_lo_D1000_ms']==9000 and p['submit_hi_D1000_ms']==12000
 # Synthetic known effects/commands fit the full intervals; the midpoint need not equal them.
 assert 10000<=11800<=12000 and p['submit_lo_D1000_ms']<=10800<=p['submit_hi_D1000_ms']
 assert B.point_fields(10000,12000,False)['midpoint_ms'] is None
 assert B.union_windows([r])[0]=='missing_view'
 assert B.union_windows([r,{**r,'view':'S_NOPRIO','removal_hi_ms':'14000'}])==('three_view_union',10000,14000)
 assert B.union_windows([r,{**r,'view':'S_NOPRIO','removal_hi_ms':'-1'}])[0]=='unbounded_or_inconsistent_view'
 assert B.successor_gap(10000,12000,dict(estimate_enabled=True,effect_lo_ms=11000,effect_hi_ms=15000))==('overlap_or_order_tied',-1.,5.)
 assert B.successor_gap(10000,12000,dict(estimate_enabled=True,effect_lo_ms=13000,effect_hi_ms=15000))==('ordered_after_under_windows',1.,5.)
 assert B.successor_gap(10000,12000,dict(estimate_enabled=False))[0]=='next_post_or_end_refused'
 print('PASS: partial/exhausted/expiry/capacity/conflict gates, bounded estimates, delay interval, missing views, successor ambiguity')

if __name__=='__main__':main()
