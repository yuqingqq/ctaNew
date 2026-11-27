"""Apply the frozen December paired replay to the November 14–18 archive."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import runpy
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'event_pressure_dec01_05_20261127'))
import forward_pair as pair

DAYS = tuple(f'202611{day}' for day in range(14, 19))
pair.DAYS = DAYS
pair.OUT = pair.ROOT/'data/predict_fun/research/event_pressure_nov14_18'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', choices=(*pair.STAGES, 'signals', 'baseline', 'event', 'finish'))
parser.add_argument('day', choices=DAYS, nargs='?')
args = parser.parse_args()
pre = pair.module(pair.REV/'urgency_near_selected_20261127/forward_preflight.py', 'availability')
files = pre.inspect(datetime(2026,11,14,tzinfo=timezone.utc), datetime.now(timezone.utc))
if any(status != 'hour_files_present' for _, status, _ in files):
    sys.exit(str(files))
pair.OUT.mkdir(parents=True, exist_ok=True)
pair.verify_control()
if args.action == 'finish':
    pair.finish()
elif args.day is None:
    parser.error('a date is required')
elif args.action in pair.STAGES:
    sys.path.insert(0, str(pair.ROOT))
    from predictfun.research import pf_io
    pf_io.OUT = str(pair.OUT)
    os.environ['PF_FORWARD'] = '20261120'  # final two hourly tail files on Nov 19
    sys.argv = [args.action, args.day]
    runpy.run_module('predictfun.research.'+args.action, run_name='__main__')
else:
    pair.stage(args.action, args.day)
