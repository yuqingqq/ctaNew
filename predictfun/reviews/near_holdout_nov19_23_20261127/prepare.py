"""Run the existing market and trade timing builders under the test's own data root."""
import os
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from predictfun.research import pf_io

day,stage=sys.argv[1:3]
assert day in tuple(f'202611{d}' for d in range(19,24))
assert stage in ('pf_step1','pf_step1_timing','pf_step1_booktime','pf_txmap')
pf_io.OUT=str(ROOT/'data/predict_fun/research/near_holdout_nov19_23')
assert (Path(pf_io.OUT)/'control_pass.json').exists()
os.environ['PF_FORWARD']='20261125'  # include the final permitted two-hour tail in txmap
sys.argv=[stage,day]
runpy.run_module('predictfun.research.'+stage,run_name='__main__')
