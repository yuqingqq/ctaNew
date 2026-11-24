"""Checker-only supersession: missing/nonmissing equality must refuse with ValueError."""
import json
import math
import subprocess
import build as B
import check as V

def eq(a,b,msg):
 am=a in ('',None);bm=b in ('',None)
 B.need(am==bm,msg)
 if not am:B.need(math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-8),msg)

def render_check():
 out=B.HERE/'outputs'
 p={(r['day'],int(r['mid']),r['maker'],r['order_hash']):r for r in B.rows(out/'post_estimates.csv.gz')}
 e={(r['day'],int(r['mid']),r['maker'],r['order_hash'],r['view']):r for r in B.rows(out/'ending_estimates.csv.gz')}
 u={(r['day'],int(r['mid']),r['maker'],r['order_hash']):r for r in B.rows(out/'union_windows.csv.gz')}
 r=B.rows(out/'signed_chain_estimates.csv');s=B.lookup(B.rows(B.INPUTS['signed']))
 B.need(len(r)==len(s)==88,'render membership')
 for a in r:
  k=(a['day'],int(a['mid']),a['maker'],a['order_hash']);post=p[k];ws=float(post['ws_ms'])
  active=V.truth(e[(*k,'S_FIFO')]['pull_estimate_enabled']) and V.truth(e[(*k,'S_NOPRIO')]['pull_estimate_enabled']) and u[k]['union_status']=='three_view_union'
  B.need(V.truth(a['conditional_withdrawal_all_views'])==active,'render gate')
  eq(a['signed_shares'],s[k]['signed_quote_shares'],'signed shares')
  for suffix,source in [('lo','effect_lo_ms'),('mid','midpoint_ms'),('hi','effect_hi_ms')]:
   expected=(float(post[source])-ws)/1000 if V.truth(post['estimate_enabled']) else None
   eq(a['post_'+suffix+'_s'],expected,'relative post time')
  lo=float(u[k]['union_lo_ms']);hi=float(u[k]['union_hi_ms'])
  for suffix,value in [('lo',lo),('mid',(lo+hi)/2),('hi',hi)]:eq(a['pull_effect_'+suffix+'_s'],(value-ws)/1000 if active else None,'relative pull time')
 B.need(B.digest(out/'signed_chain_estimates.csv')==json.loads((B.HERE/'readout.json').read_text())['rendered_sha256'],'render hash')
 return len(r)

if __name__=='__main__':
 name=str(B.HERE.relative_to(B.ROOT)/'check_v2.py')
 B.need(subprocess.check_output(['git','show','HEAD:'+name],cwd=B.ROOT)==(B.HERE/'check_v2.py').read_bytes(),'checker v2 freeze')
 eq(None,'','missing positive')
 for a,b in [(None,1),(1,None)]:
  try:eq(a,b,'known-bad missing mismatch')
  except ValueError:pass
  else:raise AssertionError('missing mismatch not refused')
 render_n=render_check();V.eq=eq;V.run()
 print('Renderer exact identity/window check PASS:',render_n)
