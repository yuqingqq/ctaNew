"""Copy already-bound descriptive fields; no additional action inference."""
import gzip,json,collections
import extract_v2 as E
D=json.loads(gzip.decompress((E.H/'data.json.gz').read_bytes()));R=E.load_records();orders={r['order_hash']:r for r in E.read(E.R/'predictfun/reviews/maker_action_population_20261124/outputs/orders.csv')};extra={}
for m in D['cases']:
 extra[str(m['id'])]={}
 for arm in E.ARMS:
  raw=R[m['id']][arm];extra[str(m['id'])][arm]={'structural':raw['struct_log']};print(m['id'],arm,'structural schema',raw['struct_log'][:1])
extra['signed_orders']={h:r for h,r in orders.items() if any(f['hash']==h for m in D['cases'] for f in m['makers'])}
extra['inventory_diagnosis']=json.loads((E.H/'inventory_diagnosis.json').read_text())
(E.H/'extra.js').write_text('window.MM_EXTRA='+json.dumps(extra,separators=(',',':'))+';\n');print('signed orders',len(extra['signed_orders']))
