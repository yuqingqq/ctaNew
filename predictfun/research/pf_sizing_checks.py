"""Direct replay check of the optional exact-path caching optimization."""
import json
import os
import subprocess
import sys
import pandas as pd
from .pf_sizing_study import DATA,DEST,HERE,write,sha
from .pf_sizing_run import COMMON,specs,stem,folder

def main():
    day='20261023';tag='sst_cache_check'
    out=DATA/'step4'/f'ladder_tokyoreal_custom_e2_preds_basis_study_{tag}'/day
    if out.exists():raise ValueError('refuse overwrite')
    env=dict(os.environ,**COMMON,PF_PREDS='preds_basis_study',PF_TAG=tag,PF_MAX_WINDOWS='0',
        PF_ARMS=';'.join(specs().values()),PF_STUDY_EXEC='1',PF_CAP_CACHE='1',
        PF_TXJIT='frame',PF_TXSHIFT='0',PF_QACT='')
    env.pop('PF_LIVE_CAPSTATE_IN',None)
    subprocess.run([sys.executable,'-m','predictfun.research.pf_sizing_engine',day,'tokyoreal','custom'],
        env=env,cwd=HERE.parents[1],check=True)
    old=folder(day,'J');checks=[]
    for label,spec in specs().items():
        for kind in ['win','fills','requests']:
            a=old/kind/stem(spec);b=out/kind/stem(spec)
            pd.testing.assert_frame_equal(pd.read_parquet(a),pd.read_parquet(b),check_exact=True)
            checks.append(dict(variant=label,kind=kind,old_sha=sha(a),new_sha=sha(b)))
    write(DEST/'cache_full_replay_control.json',dict(day=day,windows_per_variant=288,variants=8,
        comparisons=len(checks),exact=True,checks=checks,source_sha=sha(HERE/'pf_sizing_engine.py')))
    print('Cached vs direct full replay exact:',len(checks),'artifacts, 8 variants, 288 windows')

if __name__=='__main__':main()
