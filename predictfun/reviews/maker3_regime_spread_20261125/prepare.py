import json,sys
import common as C
def freeze():
 inputs=[C.PAY,C.P/'build.json',C.P/'checks.json',C.PANEL,C.REF,C.P/'core.py']
 assert C.sha(C.PAY)==json.loads((C.P/'build.json').read_text())['hashes']['data.json.gz']
 files=['contract.md','common.py','prepare.py','build.py','check.py','selftest.py','plot.py']
 C.dump(C.H/'manifest.json',dict(asof='2026-11-25',inputs={str(p.relative_to(C.R)):C.sha(p) for p in inputs},sources={str((C.H/f).relative_to(C.R)):C.sha(C.H/f) for f in files},new_candidates=0,validation=False))
def bounds():
 C.limits();C.bind();assert not (C.H/'bounds.json').exists();f=C.features(C.neutral());C.OUT.mkdir(exist_ok=True);f.to_parquet(C.OUT/'features.parquet',index=False);b=C.boundaries(f);C.dump(C.H/'bounds.json',b)
 C.dump(C.H/'bounds_manifest.json',dict(sha256=C.sha(C.H/'bounds.json'),features_sha256=C.sha(C.OUT/'features.parquet'),selection='neutral Oct24-25 q25/q75; no maker fills or P&L read'))
 print(json.dumps(b))
if __name__=='__main__':{'freeze':freeze,'bounds':bounds}[sys.argv[1]]()
