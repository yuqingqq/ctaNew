"""Independent record arithmetic and UI-clock check, with wrong-price refusal."""
import copy,gzip,json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
rows={}
for line in (H/'fill_record_check.log').read_text().splitlines():
 for k in ['CANONICAL_LEGS','CANONICAL_RECORDS']:
  if line.startswith(k+' '):rows[k]=json.loads(line[len(k)+1:])
e=json.loads((H/'fill_clock_evidence.json').read_text())
def check(rows,e):
 legs=rows['CANONICAL_LEGS']; records=rows['CANONICAL_RECORDS']
 for rec in records:
  ls=[x for x in legs if x['sid']==rec['sid']]
  assert abs(sum(x['shares'] for x in ls)-rec['amount_filled'])<1e-8
  assert abs(sum(x['shares']*x['y_price'] for x in ls)/rec['amount_filled']-rec['price_executed'])<1e-8
 for f in e['fills']:
  l=next(x for x in legs if x['sid']==f['sid'] and x['mk_hash']==f['hash'])
  assert abs(100*l['y_price']-f['price'])<1e-8 and l['shares']==f['q']
  assert l['mk_quote']=='Bid' and l['mk_index_set']==2
  assert abs(1-l['mk_price']-l['y_price'])<1e-8 and l['y_side']=='ask'
  assert (l['settle_ms']-1792977000000)/1000==f['t']
 frames={r['fid']:r for r in e['frames']}
 assert frames[745]['bid']==59 and frames[745]['ask']==61
 assert frames[753]['ask_shares']['62']==45 and frames[754]['ask']==63
 assert frames[753]['ask_shares']['63']-frames[754]['ask_shares']['63']==155
 assert frames[754]['ask_shares']['63']-frames[755]['ask_shares']['63']==5
check(rows,e)
for name in ['price','size','clock','older_book']:
 bad=copy.deepcopy(e)
 if name=='price':bad['fills'][0]['price']=61
 if name=='size':bad['fills'][0]['q']=156
 if name=='clock':bad['fills'][0]['t']=8.5
 if name=='older_book':next(x for x in bad['frames'] if x['fid']==745)['ask']=63
 try:check(rows,bad)
 except AssertionError:pass
 else:raise AssertionError('corruption escaped: '+name)
d=json.loads(gzip.decompress((H/'data.json.gz').read_bytes()))
m=next(x for x in d['cases'] if x['id']==3789484)
f=e['fills'][0];depth=next(r for r in m['depth'] if r[0]==8.5);b=next(r for r in m['book'] if r[0]==8.5)
assert depth[1][0][0]==59 and depth[2][0][0]==61 and b[6]==8.494
assert abs((f['t']-b[6])*1000-449)<1e-6
out={'checks_pass':True,'corruptions_refused':4,'first_record_shares':200,'at_62_shares':45,'at_63_shares':155,'second_record_63_shares':5,'display_book_update_s':b[6],'display_age_ms':round((f['t']-b[6])*1000),'payload_sha256':hashlib.sha256(gzip.decompress((H/'data.json.gz').read_bytes())).hexdigest()}
(H/'fill_clock_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
