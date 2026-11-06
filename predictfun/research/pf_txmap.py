"""Real taker-order identity for every trade leg (second review, 2026-11-05: the replay's transaction key mid / exec_s /
first_recv_ns / taker_index / taker_quote / price_executed merged different taker orders - 42% of keys on 10-24 12-18Z).
Re-reads the REST match pages exactly as pf_step1 does (leg = transactionHash[2:18] + maker hash[2:18], first-seen receipt) and
keeps, per leg, the taker order = full transactionHash + taker hash. Joined to legs_timed2 on its content (mid, exec_s,
first_recv_ns, side, ptick, shares, price_executed, taker_index, taker_quote) + rank within equal content.
Statuses (row-aligned with legs_timed2, tk = -1 unless mapped): 0 mapped; 1 absent from the map (first seen in an hour not read
here); 2 equal-content legs counted differently in map and legs (an hour not read here); 3 equal-content legs of more than one
taker order (which row is which order is not identifiable); 4 no taker hash (tx only); 5 mapped after refining a status-3 group by
the maker signer (pf_step1's per-day signer index, reproduced by the same pass; used only if it agrees on >= 99.9% of status-0 legs).
Hours: pf_step1's (day -1 h .. +2 h), minus sealed days (10-28..30) and forward days (>= 11-04) - listed in the report.
Selftests (refuse on failure): positive control - the old key must merge > 0 taker orders on the day; invariant - one taker
order of >= 2 legs has one market and one executedAt second (violations <= 0.1% of orders) and the same check on orders shuffled across legs
must fire (>= 50%).
Writes step1/<day>/legs_tk.parquet (tk, tk_status) + txmap.json.
Usage: python3 -m predictfun.research.pf_txmap 20261024
"""
import sys, collections
import numpy as np, pandas as pd
from . import pf_io as io

day = sys.argv[1]
SEALED = ('20261028', '20261029', '20261030'); FORWARD = '20261104'
if day in SEALED or day >= FORWARD:
    sys.exit(f'REFUSED: {day} is sealed or a forward day')
S1 = f'{io.OUT}/step1/{day}'
HALL = io.hours_of(day, extra_after=2, extra_before=1)
HOURS = [h for h in HALL if h[:8] not in SEALED and h[:8] < FORWARD]
st = collections.Counter()
legs = {}; signers = {}
for h in HOURS:
    for recv, params, path, pl in io.rest_rows(h, ('/v1/orders/matches',)):
        for x in pl.get('data') or []:
            try:
                ts = int(io.iso_s(x['executedAt']))
            except (KeyError, TypeError, ValueError):
                continue
            mid = (x.get('market') or {}).get('id'); t = x['taker']; txf = x['transactionHash']; th = t.get('hash') or ''
            for m in x['makers']:
                k = txf[2:18] + m['hash'][2:18]
                old = legs.get(k)
                if old is not None:
                    if recv < old[2]:
                        legs[k] = old[:2] + (recv,) + old[3:]
                    continue
                side, py = io.yes_side(m)
                sg = signers.setdefault(m['signer'], len(signers))
                legs[k] = (mid, ts, recv, 0 if side == 'bid' else 1, int(round(py * 100)), int(m['amount']) / io.E18,
                           int(x['priceExecuted']) / io.E18, t['outcome']['indexSet'], 0 if t['quoteType'] == 'Bid' else 1,
                           txf + '|' + th, int(bool(th)), sg)
    io.mem_check(f'rest {h}')
C = ['mid', 'exec_s', 'first_recv_ns', 'side', 'ptick', 'shares', 'price_executed', 'taker_index', 'taker_quote']
M = pd.DataFrame(list(legs.values()), columns=C + ['tko', 'has_th', 'signer']); del legs
M['tk'] = pd.factorize(M.tko)[0].astype(np.int64); M.drop(columns='tko', inplace=True)
L = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=C + ['signer'])
M = M[M.mid.isin(set(L.mid))].reset_index(drop=True)
L['_row'] = np.arange(len(L))
gl = L.groupby(C, sort=False, dropna=False); L['_n'] = gl.cumcount(); L['_cnt'] = gl[C[0]].transform('size')
gm = M.groupby(C, sort=False, dropna=False); M['_n'] = gm.cumcount(); M['_cnt'] = gm[C[0]].transform('size'); M['_ntk'] = gm.tk.transform('nunique')
J = L.merge(M[C + ['_n', '_cnt', '_ntk', 'tk', 'has_th', 'signer']], on=C + ['_n'], how='left', suffixes=('', '_m'))
assert len(J) == len(L) and (J._row.values == np.arange(len(L))).all(), 'join changed rows'
s = np.zeros(len(J), np.int8)
s[J.tk.isna().values] = 1
s[(s == 0) & (J._cnt.values != J._cnt_m.values)] = 2
s[(s == 0) & (J._ntk.values > 1)] = 3
s[(s == 0) & (J.has_th.values == 0)] = 4
tk = np.where(s == 0, J.tk.values, -1).astype(np.int64)
sig_agree = float((J.signer.values[s == 0] == J.signer_m.values[s == 0]).mean()) if (s == 0).any() else 0.0
if sig_agree >= 0.999 and (s == 3).any():                                  # refine ambiguous groups by the maker signer
    C2 = C + ['signer']; L3 = L.loc[s == 3, C2 + ['_row']].copy()
    g3 = L3.groupby(C2, sort=False, dropna=False); L3['_n'] = g3.cumcount(); L3['_cnt'] = g3[C[0]].transform('size')
    gm2 = M.groupby(C2, sort=False, dropna=False); M2 = M[C2 + ['tk', 'has_th']].assign(_n=gm2.cumcount(), _cnt=gm2[C[0]].transform('size'), _ntk=gm2.tk.transform('nunique'))
    J3 = L3.merge(M2, on=C2 + ['_n'], how='left', suffixes=('', '_m')); assert len(J3) == len(L3)
    ok3 = J3.tk.notna().values & (J3._cnt.values == J3._cnt_m.values) & (J3._ntk.values == 1) & (J3.has_th.values == 1)
    rows3 = J3._row.values[ok3]; s[rows3] = 5; tk[rows3] = J3.tk.values[ok3].astype(np.int64)
out = pd.DataFrame(dict(tk=tk, tk_status=s))
sh = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['shares']).shares.values
rep = dict(day=day, signer_agreement_status0=sig_agree, hours_read=HOURS, hours_excluded=[h for h in HALL if h not in HOURS], legs=int(len(L)),
           status_legs={int(k): int(v) for k, v in zip(*np.unique(s, return_counts=True))},
           mapped_share_legs=float(((s == 0) | (s == 5)).mean()), mapped_share_shares=float(sh[(s == 0) | (s == 5)].sum() / sh.sum()))
# old key vs real taker order (on mapped legs)
OK = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
mp = (s == 0) | (s == 5); Lm = L.loc[mp, OK].assign(tk=tk[mp])
per_old = Lm.groupby(OK, sort=False).tk.nunique()
rep.update(old_keys=int(len(per_old)), old_keys_with_gt1_taker_order=int((per_old > 1).sum()), taker_orders=int(Lm.tk.nunique()),
           taker_orders_over_gt1_old_key=int((Lm.drop_duplicates(OK + ['tk']).groupby('tk').size() > 1).sum()))
def inv(frame):                                                          # testable only on orders with >= 2 legs
    f = frame[frame.tk.map(frame.tk.value_counts()).values >= 2]
    return float((f.groupby('tk')[['mid', 'exec_s']].nunique().max(axis=1) > 1).mean())
rng = np.random.default_rng(int(day))
rep['invariant_violation_share'] = inv(Lm[['mid', 'exec_s', 'tk']])
rep['invariant_violation_share_shuffled'] = inv(Lm[['mid', 'exec_s']].assign(tk=rng.permutation(Lm.tk.values)))
print(rep, flush=True)
if rep['old_keys_with_gt1_taker_order'] == 0:
    sys.exit('REFUSED: positive control - the old key merges no taker orders (instrument cannot see the defect)')
if rep['invariant_violation_share'] > 0.001 or rep['invariant_violation_share_shuffled'] < 0.5:
    sys.exit(f"REFUSED: invariant check {rep['invariant_violation_share']} / shuffled {rep['invariant_violation_share_shuffled']}")
out.to_parquet(f'{S1}/legs_tk.parquet', index=False)
io.write_json(f'{S1}/txmap.json', rep)
