"""Which kind of match fills makers behind the best while the better prices stand? (2026-11-03; found while answering
"what is the main difference of our strategy against theirs": 18-29% of market maker volume fills behind the best without
the better levels clearing, and the replay counts such trades as sweeps through our orders.)

From the raw REST matches (one transaction = one taker + its makers), each maker leg is typed by token and quote:
  direct  maker on the taker's token, opposite quote (YES ask filled by a YES buyer, ...)
  mint    maker on the other token, both bids (YES buyer + NO buyer -> new shares)
  merge   maker on the other token, both asks
Legs are joined to legs_timed2 on (mid, first receipt, YES side, YES price, shares); unjoined legs are a status. For timed
live legs (0-240 s): position vs the YES-space book in the frame strictly before (at best / behind) and whether the level
cleared (frame at or after the fill shows the best beyond the fill price); net 1 s = spread + informed, c/sh.
With PF_MT_TXCHECK=1 (one day): validates the transaction key used by pf_fill_mix (legs_timed2 has no transaction id):
(mid, exec_s, first receipt, taker token, taker quote, price executed) against the true transaction hash.
Usage: python3 -m predictfun.research.pf_match_type [day ...]      (default 20261023 20261101; both already read)
"""
import os, sys, collections
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = sys.argv[1:] or ['20261023', '20261101']
S4 = f'{io.OUT}/step4'; OUTD = f'{S4}/match_type'; os.makedirs(OUTD, exist_ok=True)
acc = []; st = collections.Counter()
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws']).set_index('mid')
    legs = {}
    for h in io.hours_of(d, extra_after=1, extra_before=1):
        if not os.path.exists(f'{io.PF_RUN}/rest/{h}.jsonl.gz'):
            continue
        for recv, _, _, pl in io.rest_rows(h, '/v1/orders/matches'):
            for x in pl.get('data') or []:
                mid = (x.get('market') or {}).get('id')
                if mid not in reg.index:
                    continue
                t = x['taker']; tt = t['outcome']['indexSet']; tq = t['quoteType']; tx = x['transactionHash'][2:18]; nm = len(x['makers'])
                for m in x['makers']:
                    k = tx + m['hash'][2:18]
                    if k in legs:
                        if recv < legs[k][1]:
                            legs[k] = (legs[k][0], recv) + legs[k][2:]
                        continue
                    side, py = io.yes_side(m)
                    mt, mq = m['outcome']['indexSet'], m['quoteType']
                    kind = 'direct' if (mt == tt and mq != tq) else ('mint' if (mt != tt and mq == 'Bid' and tq == 'Bid') else ('merge' if (mt != tt and mq == 'Ask' and tq == 'Ask') else 'other'))
                    legs[k] = (mid, recv, 0 if side == 'bid' else 1, int(round(py * 100)), int(m['amount']) / io.E18, kind, nm, 'YES' if mt == 1 else 'NO', tx)
        io.mem_check(f'rest {h}')
    R = pd.DataFrame(list(legs.values()), columns=['mid', 'first_recv_ns', 'side', 'ptick', 'shares', 'kind', 'n_makers', 'maker_token', 'tx']); del legs
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'first_recv_ns', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 't_rel',
                                                              'exec_s', 'taker_index', 'taker_quote', 'price_executed'])
    J = lt.merge(R.drop_duplicates(['mid', 'first_recv_ns', 'side', 'ptick', 'shares']), on=['mid', 'first_recv_ns', 'side', 'ptick', 'shares'], how='left')
    if os.environ.get('PF_MT_TXCHECK') == '1':
        key = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
        Jt = J[J.tx.notna()].copy(); Jt['pk'] = Jt.groupby(key, sort=False).ngroup()
        ntx = Jt.groupby('pk').tx.nunique(); npk = Jt.groupby('tx').pk.nunique()
        def through(df, g):
            w = df.groupby(g).ptick.transform('min'); b = df.groupby(g).ptick.transform('max')
            return np.where(df.side.values == 0, df.ptick.values > w.values, df.ptick.values < b.values)
        a = through(Jt, 'tx'); b_ = through(Jt, 'pk'); q = Jt.shares.values
        print(f"[txcheck] {d}: legs {len(Jt)}; proxy groups holding one transaction {(ntx == 1).mean():.2%} (legs in them "
              f"{q[Jt.pk.map(ntx).values == 1].sum() / q.sum():.2%}); transactions split across proxy groups {(npk > 1).mean():.2%}; "
              f"through flag agrees on {q[a == b_].sum() / q.sum():.2%} of shares (through share true {q[a].sum() / q.sum():.1%}, proxy {q[b_].sum() / q.sum():.1%})", flush=True)
        sys.exit(0)
    st[f'{d}_legs'] += len(lt); st[f'{d}_unjoined'] += int(J.kind.isna().sum())
    J = J[J.kind.notna() & (J.t_rel >= 0) & (J.t_rel < 240)]
    timed = (J.how.values > 0) | np.isfinite(J.t_e2.values); st[f'{d}_1s_only'] += int((~timed).sum()); J = J[timed].copy()
    J['ta'] = np.where(J.how.values > 0, J.t_ms.values, J.t_e2.values)
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))]).sort_values(['mid', 't_ms'], kind='stable')
    B = {m: g for m, g in bk.groupby('mid', sort=False)}; del bk
    for mid, g in J.groupby('mid', sort=False):
        b = B.get(mid)
        if b is None:
            st['no_book'] += len(g); continue
        tf = b.t_ms.values.astype(float); b1 = b.b1_p.values; a1 = b.a1_p.values; vm = (b1 >= 0) & (a1 > b1); tv = tf[vm]; mv = ((b1 + a1) / 2.0)[vm]
        ta = g.ta.values.astype(float); p = g.ptick.values; side = g.side.values; sg = np.where(side == 0, 1.0, -1.0)
        j0 = np.searchsorted(tf, ta, 'left') - 1; jn = np.searchsorted(tf, ta, 'left'); i0 = np.searchsorted(tv, ta, 'left') - 1; i1 = np.searchsorted(tv, ta + 1000.0, 'right') - 1
        ok = (j0 >= 0) & (jn < len(tf)) & (i0 >= 0) & (i1 >= 0); j0 = np.clip(j0, 0, None); jn = np.clip(jn, 0, len(tf) - 1)
        best0 = np.where(side == 0, b1[j0], a1[j0]); bestn = np.where(side == 0, b1[jn], a1[jn]); dep = np.where(side == 0, best0 - p, p - best0)
        ok &= (best0 >= 0) & (dep >= 0); st['unusable'] += int((~ok).sum())
        clr = np.where(side == 0, bestn < p, (bestn < 0) | (bestn > p))
        net = sg * (mv[np.clip(i0, 0, None)] - p) + sg * (mv[np.clip(i1, 0, None)] - mv[np.clip(i0, 0, None)])
        acc.append(pd.DataFrame(dict(day=d, kind=g.kind.values, token=g.maker_token.values, multi=g.n_makers.values > 1, q=g.shares.values,
                                     pos=np.where(dep == 0, 'at best', 'behind'), cleared=clr, qn=g.shares.values * net))[ok])
    del B, J, lt, R
    io.mem_check(f'day {d}')
A = pd.concat(acc, ignore_index=True); A.to_parquet(f'{OUTD}/legs.parquet', index=False)
print(f'statuses {dict(st)}')
v = A.q.sum()
print('timed live maker volume (0-240 s) by match kind: share, and within each kind the mix by position / cleared with net 1 s (c/sh)')
for kind, g in A.groupby('kind'):
    cells = []
    for pos in ('at best', 'behind'):
        for c in (False, True):
            m = (g.pos == pos) & (g.cleared == c)
            cells.append(f"{pos}{', cleared' if c else ', not cleared'} {g.q[m].sum() / g.q.sum():5.1%} ({g.qn[m].sum() / max(g.q[m].sum(), 1e-9):+.2f})")
    print(f"  {kind:7s} {g.q.sum() / v:5.1%} of volume | maker token YES {g.q[g.token == 'YES'].sum() / g.q.sum():4.0%} | " + ' | '.join(cells))
bh = A[(A.pos == 'behind') & ~A.cleared]
print('legs behind the best that did not clear the level: by kind ' + ', '.join(f"{k} {x:.1%}" for k, x in (bh.groupby('kind').q.sum() / bh.q.sum()).items())
      + f"; in multi-maker transactions {bh.q[bh.multi].sum() / bh.q.sum():.0%}")
io.write_json(f'{OUTD}/report.json', dict(days=DAYS, statuses=dict(st), share_by_kind=(A.groupby('kind').q.sum() / v).round(4).to_dict(),
              behind_not_cleared_by_kind=(bh.groupby('kind').q.sum() / bh.q.sum()).round(4).to_dict()))
