"""Canonical-address adapter. The first build refused zero rows vs accepted
accounting because Parquet stored checksummed addresses. Compare lowercase
identities after day/market filtering. No population, rule or economics change.
"""
import subprocess,sys
import pyarrow as pa
import pyarrow.compute as pc
import core as C
def load_fills(mids):
 cols=['day','mid','sid','mk_idx','px_rank','role','addr','hash','own_index_set','y_side','s','q','p','self_trade','settle_ms','fee_fill','status']
 t=C.pq.read_table(C.V/'econ/fills_valued.parquet',columns=cols,filters=[('day','in',C.DAYS),('mid','in',list(map(int,mids)))],use_threads=False)
 return t.filter(pc.equal(pc.utf8_lower(t['addr']),C.ADDR)).to_pandas()
assert pc.equal(pc.utf8_lower(pa.array(['0x57FbC891E89184DA071AE9D97E745531cd7F718a',C.ADDR,'other'])),C.ADDR).to_pylist()==[True,True,False]
C.load_fills=load_fills
for fn in ['run_v2.py','amendment.md']:
 p=C.H/fn;assert subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(C.R))],cwd=C.R)==p.read_bytes()
if __name__=='__main__':
 if sys.argv[1]=='build':
  import build
  build.build()
 elif sys.argv[1]=='risk':
  import risk
  risk.run()
 elif sys.argv[1]=='check':
  import check
  check.run()
