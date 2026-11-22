"""Close source/copy provenance and missing-value instrument scope, no replay."""
from pathlib import Path
import gzip, hashlib, importlib.util, json, subprocess, sys
ROOT=Path('/home/yuqing/ctaNew');PIN=Path('/tmp/pf-lc-review-6b806c19');OUT=Path(__file__).resolve().parent
TARGET='6b806c192980de106da80fa97ce4993d6553ec66';REL='predictfun/reviews/ladder_long_continuation_20261122'
sys.path.insert(0,str(PIN/'predictfun/research'))
import pf_ladder_long_continuation as X
def blob(p):return subprocess.check_output(['git','-C',str(PIN),'show',f'{TARGET}:{p}'])
modules={}
for name,m in list(sys.modules.items()):
    p=getattr(m,'__file__',None)
    if not p or '/predictfun/research/' not in str(p):continue
    rel='predictfun/research/'+str(p).split('/predictfun/research/',1)[1]
    assert Path(p).read_bytes()==blob(rel),rel
    modules[rel]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
gm=json.loads((PIN/REL/'manifest.json').read_text());D=ROOT/'data/predict_fun/research/ladder_long_continuation_v1_20261122'
count=0
for part,days in gm['parts'].items():
    for day in days:
        for suffix in ('.json','_parity.json'):
            p=PIN/REL/'outputs'/f'{part}_{day}{suffix}'
            assert p.read_bytes()==(D/part/f'{day}{suffix}').read_bytes(),p
            count+=1
sp=importlib.util.spec_from_file_location('lc_ch_supplement',PIN/REL/'lc_checks.py');C=importlib.util.module_from_spec(sp);sp.loader.exec_module(C)
C.REPO=PIN;C.R=PIN/REL
E=C.load();E['my']=C.my_pairs(E)
# Both variants intentionally lack a fork mark. This is not an actual saved
# case; every real mark was finite in verify.py.
stub=dict(applied=[['live',50.,1,'KEEP','same']],pre_fills=[],reconciled=True,queue_reset_ok=True,lineage_pressure_pulls=0,
          M_fork=None,post_c=2.,Q=float('nan'),focal_c=2.,other_c=0.,seq='no_focal_fill')
st=dict(day='synthetic',m=0,kind='REVIEW',side=0,gid=1,rung=0.,t=50.,key='synthetic')
missing=X.pair_result(st,stub,dict(stub,post_c=1.,focal_c=1.))
assert X.reconciles(100.,0.,0.,None,float('nan')) is True
assert missing['status']=='ok'
out=dict(target=TARGET,loaded_source_hashes=modules,receipt_day_and_parity_copies=count,
         missing_fork_mark_synthetic_status=missing['status'],missing_fork_mark_is_actual_result=False,
         caveat='Current saved Y is verified; missing marks must be explicit before expanding the builder population.')
(OUT/'supplement.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='loaded_source_hashes'},indent=2));print('Pinned imported research sources:',len(modules))
