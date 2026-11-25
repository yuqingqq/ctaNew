"""Read already-declared turnover metrics; no new experiment or policy.
USER: earlier exits may justify worse prices through additional profitable turnover.
Reaggregate the accepted per-market metrics, preserving residual inventory.
"""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent/'inventory_exit_exception_20261125'
expected=dict(line.split('  ')[::-1] for line in (P/'SHA256SUMS').read_text().splitlines())
assert hashlib.sha256((P/'checks.json').read_bytes()).hexdigest()==expected['checks.json']
c=json.loads((P/'checks.json').read_text());assert c['pass_all'] and len(c['rows'])==76
out={}
for population,mids in [('all19',None),('3789484',{3789484})]:
 totals={}
 for arm in ['WIDE','WIDE_COST']:
  rows=[r for r in c['rows'] if r['arm']==arm and (mids is None or r['mid'] in mids)]
  assert len(rows)==(19 if mids is None else 1)
  for r in rows:
   assert abs(r['maker_shares']-2*r['matched_shares']-abs(r['net_shares']))<1e-5
   assert abs(r['completed_c']+r['residual_c']-r['net_c'])<1e-5
  z={k:sum(r[k] for r in rows) for k in ['matched_shares','maker_shares','completed_c','residual_c','net_c']}
  z['completed_c_per_matched_share']=z['completed_c']/z['matched_shares']
  totals[arm]=z
 a,b=totals['WIDE'],totals['WIDE_COST']
 out[population]=dict(totals=totals,matched_share_change_pct=100*(b['matched_shares']/a['matched_shares']-1),
  filled_share_change_pct=100*(b['maker_shares']/a['maker_shares']-1),
  completed_delta_usd=(b['completed_c']-a['completed_c'])/100,
  residual_delta_usd=(b['residual_c']-a['residual_c'])/100,net_delta_usd=(b['net_c']-a['net_c'])/100)
target=H/'turnover_readout.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
