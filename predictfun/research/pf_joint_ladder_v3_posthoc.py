"""JOINT-LADDER v3 POST HOC readings (2026-11-14, session ce68ebe0), labelled post hoc: chosen after seeing v3's printed table.

v3 declared R613 on WS-timed trades only; its table shows that over ALL trades (WS + E2) every LDG and FB25 cell at M 50 loses money. Whether
keeping (FB25) is better or worse than pulling (LDG) there, day by day, is not printed. This reads v3's own fills files (no new replay):
  PH1 ALL-TRADES KEEP: FB25 - LDG, $/day w/o young at 1 s, WS + E2, per client cap and clock; days FB25 higher / lower of 9.
      Label per cap at M 50: BELOW if lower on >= 7 of 9 days, ABOVE if higher on >= 7, else MIXED (printed, not declared).
  PH2 E2 ONLY: the same difference on E2-timed trades.
  PH3 DAY BLOCKS: WS $/day w/o young at 1 s, M 50, per client cap, 10-23..10-27 vs 10-31..11-03, and every day (v3 printed hard2 only).
Reconciliation control: the WS / E2 / all-trades cells recomputed here must equal v3's printed PART 1 (to its 0.1 rounding) for every client
cell, else no output.
Usage (run_light, PF_SLOT=claude): pf_joint_ladder_v3_posthoc.py
"""
import json
import os
import re
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_joint_ladder_v3 as J3                           # noqa: E402

OUTD = J3.OUTD
CN = [J3.cap_name(mo, k) for mo, k in J3.CAPS]
B1 = ('20261023', '20261024', '20261025', '20261026', '20261027'); B2 = ('20261031', '20261101', '20261102', '20261103')


def main():
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in K.DAYS9}
    dd = {d: stats[d]['days'] for d in K.DAYS9}; ndays = sum(dd.values())
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet', columns=['day', 'cfg', 'arm', 'capk', 'src', 'young', 'q', 'v1'])
                   for d in K.DAYS9], ignore_index=True)
    A = A[~A.young & A.capk.isin(CN)]
    A['val'] = A.q * A.v1 / 100.0
    G = A.groupby(['cfg', 'arm', 'capk', 'src', 'day']).val.sum()

    def cell(cfg, arm, cn, srcs, d=None):
        tot = 0.0
        for s_ in srcs:
            for dy in ([d] if d else K.DAYS9):
                tot += float(G.get((cfg, arm, cn, s_, int(dy)), 0.0))
        return tot / (dd[d] if d else ndays)
    # reconciliation with v3's printed PART 1
    txt = open(f'{OUTD}/joint_ladder_v3.txt').read(); bad = []; nrec = 0
    for line in txt.splitlines():
        m = re.match(r'\s+(M0|M50|M250)\s+(LDG|FB25|OFB2)\s+(\S+)\s+([+-][\d.]+) \|\s+([+-][\d.]+) \|\s+([+-][\d.]+) \|', line)
        if not m or m.group(3) not in CN:
            continue
        cfg, arm, cn = m.group(1), m.group(2), m.group(3)
        for v_, srcs in zip(m.group(4, 5, 6), (('WS',), ('E2',), ('WS', 'E2'))):
            nrec += 1
            if abs(cell(cfg, arm, cn, srcs) - float(v_)) > 0.051:
                bad.append((cfg, arm, cn, srcs, cell(cfg, arm, cn, srcs), v_))
    if bad or nrec != 3 * 3 * len(CN) * 3:
        sys.exit(f'POSTHOC RECONCILIATION FAILED -> no output: {nrec} cells read, mismatches {bad}')
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); R = {}
    L = [f'JOINT-LADDER v3 POST HOC readings (chosen after seeing v3\'s table); v3 fills files, 9 days; as of {asof}; reconciliation: {nrec} '
         f'printed cells recomputed, 0 differ', '',
         'PH1 / PH2: FB25 - LDG, $/day w/o young at 1 s (days FB25 higher / lower of 9): all trades || E2 only']
    for cn in CN:
        parts = []; R[cn] = {}
        for cfg in ('M0', 'M50', 'M250'):
            for key, srcs in (('all', ('WS', 'E2')), ('E2', ('E2',))):
                diff = cell(cfg, 'FB25', cn, srcs) - cell(cfg, 'LDG', cn, srcs)
                up = sum(cell(cfg, 'FB25', cn, srcs, d) > cell(cfg, 'LDG', cn, srcs, d) for d in K.DAYS9)
                dn = sum(cell(cfg, 'FB25', cn, srcs, d) < cell(cfg, 'LDG', cn, srcs, d) for d in K.DAYS9)
                R[cn][f'{cfg} {key}'] = dict(diff=diff, up=up, down=dn)
        for cfg in ('M0', 'M50', 'M250'):
            a_, e_ = R[cn][f'{cfg} all'], R[cn][f'{cfg} E2']
            parts.append(f'{cfg} {a_["diff"]:+7.1f} ({a_["up"]}/{a_["down"]}) || {e_["diff"]:+7.1f} ({e_["up"]}/{e_["down"]})')
        x = R[cn]['M50 all']
        R[cn]['label_M50_all'] = 'BELOW' if x['down'] >= 7 else ('ABOVE' if x['up'] >= 7 else 'MIXED')
        L.append(f'  {cn:<6}: ' + ' ;  '.join(parts) + f'   -> M 50 all trades: {R[cn]["label_M50_all"]}')
    L += ['', 'PH3: WS $/day w/o young at 1 s, M 50: 10-23..10-27 | 10-31..11-03 (block means), then each day']
    for cn in CN:
        for arm in J3.ARMS:
            b1 = sum(cell('M50', arm, cn, ('WS',), d) * dd[d] for d in B1) / sum(dd[d] for d in B1)
            b2 = sum(cell('M50', arm, cn, ('WS',), d) * dd[d] for d in B2) / sum(dd[d] for d in B2)
            R.setdefault('blocks', {})[f'{arm} {cn}'] = dict(b1=b1, b2=b2, days={d: cell('M50', arm, cn, ('WS',), d) for d in K.DAYS9})
            L.append(f'  {arm:<4} {cn:<6}: {b1:+7.1f} | {b2:+7.1f} ||' + ' '.join(f'{cell("M50", arm, cn, ("WS",), d):+7.1f}' for d in K.DAYS9))
    open(f'{OUTD}/joint_ladder_v3_posthoc.txt', 'w').write('\n'.join(L) + '\n')
    json.dump(R, open(f'{OUTD}/joint_ladder_v3_posthoc.json', 'w'), indent=1, default=float)
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    main()
