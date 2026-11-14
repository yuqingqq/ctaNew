"""ORDER-HARM POLICY v2 POST HOC reading (2026-11-14, session ce68ebe0), labelled post hoc: computed after the report was printed.

FINDINGS checklist C3 asks every replay result for the cancel-pending share of fills and its $; POLICY v2's report does not print it. This reads
the run's own fills files (no new replay):
  PH-C3 CANCEL-PENDING: per cap, fill clock and arm - filled shares a day (all trades), the share of them filled while our cancel was in flight,
        and those fills' $ a day at 1 s (all ages; w/o young), beside all fills' $ a day at 1 s.
Reconciliation control: 'all trades 1 s w/o young' recomputed here must equal PART 1 of the printed report (to its 0.1 rounding) for every
arm / clock / cap cell, else no output. Falsifier: the same comparison with young fills kept must fail.
Usage (run_light, PF_SLOT=claude): pf_order_harm_policy_v2_posthoc.py
"""
import json
import os
import re
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_order_harm_client_lane as CL                    # noqa: E402
import pf_order_harm_policy_v2 as P2                      # noqa: E402

OUTD = P2.OUTD
TESTS = P2.TESTS
ARMS = ('LDG', 'FB25', 'FB25C', 'OFB2C', 'OHQ0', 'OHQM')
CFGS = ('M0', 'M50', 'M250')
CAPN = ('hard10', 'hard20', 'hard50', 'none')


def printed_part1():
    """{(cfg, arm, cap): 'all trades 1 s w/o young'} from the printed report's PART 1 rows."""
    out = {}
    pat = re.compile(r'^\s+(M0|M50|M250)\s+(\w+)\s+(hard10|hard20|hard50|none)\s+([-+0-9.]+) \|\s+([-+0-9.]+) \|\s+([-+0-9.]+) \|')
    for line in open(f'{OUTD}/order_harm_policy_v2.txt'):
        m = pat.match(line)
        if m:
            out[(m.group(1), m.group(2), m.group(3))] = float(m.group(6))
    return out


def all_trades_aged(A, ndays, keep_young=False):
    x = A if keep_young else A[~A.young]
    g = (x.q * x.v1).groupby([x.cfg, x.arm, x.capk]).sum() / 100.0 / ndays
    return {k: float(v) for k, v in g.items()}


def mismatches(calc, printed):
    bad = [k for k in printed if k not in calc or abs(round(calc[k], 1) - printed[k]) > 0.051]
    return bad


def main():
    T0 = time.time()
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in TESTS}
    for r in stats.values():
        CL.require_contract(r)
    ndays = sum(stats[d]['days'] for d in TESTS)
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet', columns=['day', 'cfg', 'arm', 'capk', 'q', 'v1', 'pend', 'young', 'src'])
                   for d in TESTS], ignore_index=True)
    pr = printed_part1()
    if len(pr) != len(ARMS) * len(CFGS) * len(CAPN):
        sys.exit(f'REFUSED: parsed {len(pr)} PART 1 cells, expected {len(ARMS) * len(CFGS) * len(CAPN)}')
    bad = mismatches(all_trades_aged(A, ndays), pr)
    planted = mismatches(all_trades_aged(A, ndays, keep_young=True), pr)
    if bad:
        sys.exit(f'RECONCILIATION FAILED -> no output: {len(bad)} cells differ from the printed PART 1, e.g. {bad[:3]}')
    if not planted:
        sys.exit('FALSIFIER FAILED -> no output: keeping young fills still matched the printed PART 1')
    A['usd'] = A.q * A.v1 / 100.0
    L = [f'ORDER-HARM POLICY v2 POST HOC (labelled post hoc; computed after the printed report), test days {", ".join(TESTS)}; as of '
         f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; reconciliation: {len(pr)} PART 1 cells (all trades 1 s w/o young) equal, '
         f'falsifier (young fills kept) differs on {len(planted)} cells',
         '', 'PH-C3 CANCEL-PENDING (all trades, 1 s): cap clock arm | filled sh/day | cancel-pending share of shares | cancel-pending $/day all ages '
         '| w/o young | all fills $/day all ages']
    for cn in CAPN:
        for cfg in CFGS:
            for arm in ARMS:
                x = A[(A.capk == cn) & (A.cfg == cfg) & (A.arm == arm)]; p = x[x.pend]
                L.append(f'  {cn:<6} {cfg:<4} {arm:<5} | {x.q.sum() / ndays:8.0f} | {p.q.sum() / max(x.q.sum(), 1e-9):.3f} | '
                         f'{p.usd.sum() / ndays:+8.1f} | {p[~p.young].usd.sum() / ndays:+8.1f} | {x.usd.sum() / ndays:+8.1f}')
        L.append('')
    txt = '\n'.join(L)
    print(txt)
    open(f'{OUTD}/order_harm_policy_v2_posthoc.txt', 'w').write(txt + '\n')
    print(f'done in {time.time() - T0:.0f} s')


if __name__ == '__main__':
    main()
