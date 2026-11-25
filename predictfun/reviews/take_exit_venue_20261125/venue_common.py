"""Shared pieces of the venue-matched comparison (declaration.md): canonical settlement payouts (ties 50 / 50), the settlement
accounting of floor_replay.account generalised to a tie payout, and the frozen run with a declared take delay."""
import collections
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / 'take_exit_floor_20261125'))
import floor_replay as F  # noqa: E402  (frozen at da209477c: FloorBranch, worlds_yup, fee_map)

OD, MM, ER, TR, X, LB, BE, C = F.OD, F.MM, F.ER, F.TR, F.X, F.LB, F.BE, F.C
EPS = F.EPS
BASE_SLACK, FLOOR_SLACK = F.PARITY_SLACK, 0.     # BASE = the floor out of reach (exact v3 parity, floor v1 K2); FLOOR = slack 0
ARMS = {'BASE': BASE_SLACK, 'FLOOR': FLOOR_SLACK}


def payout_of(r):
    """Canonical YES payout from a registry row: 1 Up, 0 Down, 0.5 when the end price equals the start price (venue rule)."""
    C.require(int(r['up_index']) == 1 and r['outcome'] in ('Up', 'Down'), 'registry outcome')
    if float(r['start_price']) == float(r['end_price']):
        return 0.5
    return 1. if r['outcome'] == 'Up' else 0.


def settle_account(rec, y):
    """floor_replay.account with a YES payout y in {0, 0.5, 1}: round trips and held lots with fees, at the mark and at settlement."""
    C.require(y in (0., .5, 1.), f'settlement payout {y}')
    fee = F.fee_map(rec)
    mark, S = float(rec['mark']), 100. * y
    lots = collections.deque()
    rt = collections.Counter()
    cash = pos = fees = taker = maker = 0.
    for t, gid, s, q, p, it in rec['fills']:
        sg = 1. if s == 0 else -1.
        f = fee[int(gid)] if it < 0 else 0.
        fps = f / q
        cash -= sg * q * p; pos += sg * q; fees += f
        if it < 0:
            taker += q
        else:
            maker += q
        kind = 'take' if it < 0 else 'exit'
        n = q
        while n > EPS and lots and lots[0][0] != sg:
            ls, lq, lp, lf = lots[0]
            x = min(n, lq)
            gr, fe = ls * x * (p - lp), x * (lf + fps)
            for k in ('', kind + '_'):
                rt[k + 'gross_c'] += gr; rt[k + 'fees_c'] += fe; rt[k + 'shares'] += x
            n -= x; lq -= x
            if lq < EPS:
                lots.popleft()
            else:
                lots[0] = (ls, lq, lp, lf)
        if n > EPS:
            lots.append((sg, n, p, fps))
    held_mark = sum(s * q * (mark - p) for s, q, p, _ in lots)
    held_settle = sum(s * q * (S - p) for s, q, p, _ in lots)
    held_fees = sum(q * lf for _, q, _, lf in lots)
    m = dict(yup=y, mark=mark, end_position=pos, taker_shares=taker, maker_shares=maker, fees_c=fees,
             rt_shares=rt['shares'], rt_net_c=rt['gross_c'] - rt['fees_c'], rt_exit_shares=rt['exit_shares'],
             rt_exit_net_c=rt['exit_gross_c'] - rt['exit_fees_c'], rt_take_shares=rt['take_shares'], rt_take_net_c=rt['take_gross_c'] - rt['take_fees_c'],
             held_shares=sum(q for _, q, _, _ in lots), held_fees_c=held_fees, held_mark_net_c=held_mark - held_fees, held_settle_net_c=held_settle - held_fees,
             held_max_loss_c=sum(q * (p + lf) if s > 0 else q * (100. - p + lf) for s, q, p, lf in lots),
             held_won_shares=0. if y == .5 else sum(q for s, q, _, _ in lots if (s > 0) == (y == 1.)), carry_c=pos * (S - mark))
    m['mark_net_c'] = m['rt_net_c'] + m['held_mark_net_c']
    m['settle_net_c'] = m['rt_net_c'] + m['held_settle_net_c']
    tol = 1e-6 * max(1., abs(cash) + abs(pos) * 100.)
    C.require(abs(cash - rec['cash']) < tol and abs(pos - rec['position']) < 1e-6, 'cash / position differ from the ledger')
    C.require(abs(fees - rec['fees_c']) < tol and abs(rt['fees_c'] + held_fees - fees) < tol, 'fees do not add up')
    C.require(abs(m['mark_net_c'] - rec['net_value_c']) < tol, 'round trips + held at the mark != ledger net')
    C.require(abs(m['settle_net_c'] - (cash + pos * S - fees)) < tol, 'round trips + held at settlement != cash + position x settlement - fees')
    ff = X.fifo([tuple(f) for f in rec['fills']], mark)
    C.require(abs(ff['completed_c'] - rt['gross_c']) < tol and abs(ff['matched_shares'] - rt['shares']) < 1e-6, 'FIFO differs from the frozen fifo')
    return m


def one_delay(p, cache, old, edges, bps, slack, delay):
    """floor_replay.one with a declared take delay (ms from decision to venue arrival)."""
    orig, d0 = TR.SignalBranch, TR.TAKE_DELAY
    TR.SignalBranch = lambda *a, **k: F.FloorBranch(*a, slack=slack, **k)
    TR.TAKE_DELAY = float(delay)
    del F.LAST[:]
    try:
        rec = ER.run_one(p, cache, old, edges, bps, 'PASSIVE_EXIT')
    finally:
        TR.SignalBranch, TR.TAKE_DELAY = orig, d0
    C.require(len(F.LAST) == 1, 'one floor branch per run')
    C.require(all(abs(v['act'] - v['submit'] - delay) < 1e-9 for v in rec['iocs'].values()), f'an IOC delay other than {delay} ms')
    rec['floor_log'] = TR.clean(F.LAST[0].floor_log)
    rec['floor_slack'] = float(slack)
    rec['take_delay_ms'] = float(delay)
    del F.LAST[:]
    return rec
