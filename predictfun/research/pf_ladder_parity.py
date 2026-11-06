"""Parity of the ladder engine (pf_step4_ladder.py) with the live engine (pf_step4_live.py), 2026-11-05.
  off   : the same arms, no new tokens -> fills (every column the live engine writes) and money fields equal to the cent
  ctrl  : engaged-but-never-triggered / identical-line controls of the new tokens vs the base arm of the same ladder run
          (PMzero = V0's line, AM:..:c006 = AV's 0.06, LD2+LLinf = rung never placed, IMinf = never improve); money and fills equal
  falsifier: the comparator must flag a known-different pair (C4 vs C3) and must refuse a missing arm.
Usage: python3 -m predictfun.research.pf_ladder_parity <day> ...   (exit 1 on any mismatch)
"""
import sys, json, os
import numpy as np, pandas as pd
from . import pf_io as io

S4 = f'{io.OUT}/step4'
# PF_PAR_A / PF_PAR_B: the two run directories (default: live kJ vs ladder par); PF_PAR_CTRL=0 skips the ladder controls
LIVE = f"{S4}/{os.environ.get('PF_PAR_A', 'live_tokyoreal_custom_e2_preds_t60k_kJ')}"
LAD = f"{S4}/{os.environ.get('PF_PAR_B', 'ladder_tokyoreal_custom_e2_preds_t60k_par')}"
CTRL = os.environ.get('PF_PAR_CTRL', '1') == '1'
B = 'MICRO+G4+G2+G6+G5+G3+G1+'; T = '+XNA+XMIN1+XV+S60/mid@'
C = {'C1': B + 'V0' + T + 'A1:5:25:0', 'C2': B + 'V0' + T + 'A1:20:100:0', 'C3': B + 'V0' + T + 'AV:20:100:0.06', 'C4': B + 'V0' + T + 'AV:5:25:0.06'}
K = {'PMzero': B + 'PMzero' + T + 'AV:5:25:0.06', 'AMc006': B + 'V0' + T + 'AM:5:25:c006',
     'LDinf': B + 'V0+LD2+LLinf' + T + 'AV:5:25:0.06', 'IMinf': B + 'V0+IMinf' + T + 'AV:5:25:0.06'}
MONEY = ['windows', 'gross_usd', 'fills', 'shares', 'mtm_stop_usd', 'carry_stop_usd', 'mo1_usd', 'mo60_usd']
FC = ['mid', 't', 'side', 'pt', 'q', 'through', 'timed', 'settle', 'mo1', 'mo60', 'rebate', 'osz', 'zombie']
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def compare(da, na, db, nb, d):
    """list of differences between arm na of run da and arm nb of run db on day d (empty = equal)."""
    sa = json.load(open(f'{da}/{d}/summary.json'))['arms']; sb = json.load(open(f'{db}/{d}/summary.json'))['arms']
    if na not in sa or nb not in sb:
        raise KeyError(f'missing arm: {na if na not in sa else nb}')
    diff = [f'{k}: {sa[na][k]} vs {sb[nb][k]}' for k in MONEY if sa[na][k] != sb[nb][k]]
    if sa[na]['live']['requests'] != sb[nb]['live']['requests']:
        diff.append(f"requests {sa[na]['live']['requests']} vs {sb[nb]['live']['requests']}")
    fa = pd.read_parquet(f'{da}/{d}/fills/{stem(na)}.parquet', columns=FC); fb = pd.read_parquet(f'{db}/{d}/fills/{stem(nb)}.parquet', columns=FC)
    if len(fa) != len(fb) or not fa.reset_index(drop=True).equals(fb.reset_index(drop=True)):
        diff.append(f'fills differ ({len(fa)} vs {len(fb)} rows)')
    return diff


def main(days):
    bad = 0
    # falsifier first: a known-different pair must be flagged, a missing arm refused
    assert compare(LAD, C['C4'], LAD, C['C3'], days[0]), 'comparator did not flag C4 vs C3'
    try:
        compare(LAD, C['C4'] + 'x', LAD, C['C4'], days[0]); raise SystemExit('comparator accepted a missing arm')
    except KeyError:
        pass
    print('falsifier ok')
    for d in days:
        for c, n in C.items():
            df = compare(LIVE, n, LAD, n, d); bad += bool(df)
            print(f'{d} off  {c}: {"EQUAL" if not df else df}')
        for k, n in (K.items() if CTRL else ()):
            df = compare(LAD, C['C4'], LAD, n, d); bad += bool(df)
            print(f'{d} ctrl {k} vs C4: {"EQUAL" if not df else df}')
        s = json.load(open(f'{LAD}/{d}/summary.json'))['arms']
        inv = {n[-40:]: (s[n]['live']['counters'].get('own_dup_price', 0), s[n]['live']['counters'].get('own_cross', 0)) for n in s}
        print(f'{d} invariants (own_dup_price, own_cross):', inv)
        if os.environ.get('PF_PAR_INV', '1') == '1':   # PF_PAR_INV=0: report only (regression runs of the old behaviour)
            bad += sum(a + b for a, b in inv.values()) > 0
    print('PARITY', 'PASS' if bad == 0 else f'FAIL ({bad})')
    sys.exit(0 if bad == 0 else 1)


if __name__ == '__main__':
    main(sys.argv[1:])
