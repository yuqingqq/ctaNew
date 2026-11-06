"""Parity of a generated engine (pf_step4_ladder_v2.py; with arg 4 = fwd: pf_step4_ladder_fwd.py, PF_FORWARD default or later): run with PF_LINES_FILE = step4/minute_lines_t60k.json and PF_PREDS=preds_t60k,
the candidate's fills / win / requests parquet must be byte-identical to the round-i run and its summary arm entry equal. Falsifier: the
round-i pm arm (no LD2) compared with the same files must differ. Run from anywhere.
Usage: python3 -m predictfun.research.pf_v2_engine_check <round-i tag lJi|lXi> <parity run tag> <day> [fwd]
"""
import filecmp, json, sys
from . import pf_io as io

ref, run, day = sys.argv[1:4]
A = f'{io.OUT}/step4/ladder_tokyoreal_custom_e2_preds_t60k_{ref}/{day}'; B = f'{io.OUT}/step4/ladder_tokyoreal_custom_e2_preds_t60k_{run}/{day}'
stem = 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_LD2_XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet'
other = 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet'
name = 'MICRO+G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60/mid@AM:5:25:95'
bad = 0
for d in ('fills', 'win', 'requests'):
    s = filecmp.cmp(f'{A}/{d}/{stem}', f'{B}/{d}/{stem}', shallow=False); fz = filecmp.cmp(f'{A}/{d}/{other}', f'{B}/{d}/{stem}', shallow=False)
    print(d, 'identical', s, 'falsifier identical', fz); bad += int(not s) + int(fz)
sb = json.load(open(f'{B}/summary.json'))
eq = json.load(open(f'{A}/summary.json'))['arms'][name] == sb['arms'][name]; print('summary arm equal', eq); bad += int(not eq)
if len(sys.argv) > 4 and sys.argv[4] == 'fwd':   # forward-enabled engine (pf_step4_ladder_fwd): no lines file; PF_FORWARD as run
    print('PF_FORWARD', sb.get('env', {}).get('PF_FORWARD'), 'PF_LINES_FILE', sb.get('env', {}).get('PF_LINES_FILE')); bad += int('PF_LINES_FILE' in sb.get('env', {}))
else:
    lf = sb.get('env', {}).get('PF_LINES_FILE'); print('PF_LINES_FILE', lf); bad += int(not lf or not lf.endswith('step4/minute_lines_t60k.json'))
print('CHECKS', 'PASS' if bad == 0 else f'FAIL {bad}')
sys.exit(0 if bad == 0 else 1)
