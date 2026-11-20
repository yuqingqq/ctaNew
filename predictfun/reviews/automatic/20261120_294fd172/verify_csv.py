"""Reconcile committed support and partition CSV views to frozen JSON records."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew')
RC=Path('/tmp/pf-review-294fd172/predictfun/reviews/r2b_full_day_20261120')
OUT=ROOT/'predictfun/reviews/automatic/20261120_294fd172'
DATA=ROOT/'data/predict_fun/research/r2_shadow_20261119'
sup=json.loads((RC/'support_20261023_full.json').read_text())
checks={}
def check(name, frame):
    expected=frame.to_csv(index=False)
    actual=(RC/name).read_text()
    assert expected==actual, name
    checks[name]=hashlib.sha256(actual.encode()).hexdigest()

check('support_cells_full.csv',pd.DataFrame([dict(cell=k,**v) for k,v in sup['cells'].items()]))
for nm in ('categories','cross_cells','coverage'): check(f'support_{nm}_full.csv',pd.DataFrame(sup[nm]))
check('support_ranges_full.csv',pd.DataFrame([dict(side_head=h,field=c,lo=v[0],hi=v[1]) for h,r in sup['ranges'].items() for c,v in r.items()]))
check('support_presence_full.csv',pd.DataFrame([dict(side_head=h,dim=dm,value=x,originals=n) for h,r in sup['presence'].items() for dm,xs in r.items() for x,n in xs.items()]))
parts=[]
for i in range(477):
    r=json.loads((DATA/'census_20261023_full/parts'/f'part_{i:04d}.json').read_text())
    parts.append(dict(part=r['part'],rows=r['rows'],sampled_ok=r['sampled_ok'],elapsed_s=r['elapsed_s'],hwm_mib=r['hwm_mib'],part_peak_rss_mib=r['part_peak_rss_mib'],
        audited=len(r['causal']['audit']),retried=r['causal'].get('retried',0),status=r['status']))
check('parts_full.csv',pd.DataFrame(parts))
cross=pd.DataFrame(sup['cross_cells'])
out=dict(target_commit='294fd172d7712bdfeb4c776a33ba63eaf1b7fc0a',asof=datetime.now(timezone.utc).isoformat(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,all_checks_pass=True,
    fine_cells=len(cross),fine_cells_below_200=int((cross.originals<200).sum()),fine_cells_below_20_nonzero=int((cross.nonzero_originals<20).sum()))
(OUT/'csv.json').write_text(json.dumps(out,indent=2)+'\n')
print('All seven committed support/partition CSV views exactly match the frozen JSON data.')
