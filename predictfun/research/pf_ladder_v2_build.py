"""Generate pf_step4_ladder_v2.py = the FROZEN ladder engine of pm_ladder_v1 (pf_step4_ladder.py sha256 c682229e...) with two changes:
the forward-enabled FORWARD of pf_ladder_fwd_build.py (PF_FORWARD, default the frozen '20261104'), and the per-minute guard lines read from PF_LINES_FILE (required; '{day}' is replaced by the replay day) instead of the fixed
step4/minute_lines_t60k.json. Candidate pm_ladder_v2 refits the lines every day (pf_v2_refit.py writes step5/<day>/lines_v2w7.json).
The frozen file is never edited. Refuses unless the source hash matches and each anchor occurs exactly once; the generated file records
both hashes in its header. Parity: with PF_LINES_FILE = step4/minute_lines_t60k.json and PF_PREDS=preds_t60k the generated engine must
give byte-identical candidate outputs to round i (pf_v2_engine_check.py).
Usage: python3 -m predictfun.research.pf_ladder_v2_build
"""
import hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = f'{HERE}/pf_step4_ladder.py'; DST = f'{HERE}/pf_step4_ladder_v2.py'
FROZEN = 'c682229e4fbe606b217cfa633816bcba67bce701c7e690b4cb8be9aa7ff6e37f'
PATCHES = [
    ("LINES = json.load(open(f'{io.OUT}/step4/minute_lines_t60k.json'))['lines']\n",
     "LINES_FILE = os.environ['PF_LINES_FILE'].format(day=day)      # v2: per-day lines (pf_ladder_v2_build), required\n"
     "LINES = json.load(open(LINES_FILE))['lines']\n"),
]


def main():
    from .pf_ladder_fwd_build import FWD_PATCH, generate
    generate(PATCHES + [FWD_PATCH], DST, 'lines from PF_LINES_FILE; FORWARD from PF_FORWARD (pf_ladder_v2_build)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
