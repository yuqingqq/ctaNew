"""Selftest of pf_mt_econ (MAKER-TRACE-RECONSTRUCTION-v1 economics; session aea5bfb9, 2026-11-22). SYNTHETIC data only, written to a
temporary directory; every expected number below is computed by hand in its comment (never by the module). Runs the whole CLI path
(run -> one worker per day -> every writer) on an ID day + two measurement days (chunk = 1 market, so day tables cross chunks), then direct
cases and refusals.
World (YES space, ws-relative ms; V = 1 Up / 0 Down / .5 tie):
 ID 20261023, market 9001 (Up): twelve makers at 0.50 against taker 0xt0 (ties at both top-10 cuts) + 0xc1 (wins on total P&L only through
    its taker leg; not maker-majority) + 0x01's small taker leg; market 9002 is a tie (start = end price; its registry label 'Up' is wrong).
 D1 20261024, market 101 (BOOK, Up; frames f0..f9, f9 a 'gap' frame = usable; END = +298000 -> f9 .72; depth files), 102 (not BOOK, NO
    resolution, boosted), 103 / 105 (censored), 104 (no records), 106 (registry up_index 2).
 D2 20261025, market 201 (Down), no markets table."""
import datetime as dt
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_mt_common as C  # noqa: E402
import pf_mt_econ as E  # noqa: E402

need = C.need
ID, D1, D2 = C.ID_DAY, '20261024', '20261025'
WS = {9001: 1792717200, 9002: 1792720800, 101: 1792803600, 102: 1792807200, 103: 1792810800, 104: 1792814400, 105: 1792818000, 106: 1792821600,
      201: 1792890000}
DAY = {9001: ID, 9002: ID, 101: D1, 102: D1, 103: D1, 104: D1, 105: D1, 106: D1, 201: D2}
OUTCOME = {9001: 'Up', 9002: 'Up', 101: 'Up', 102: None, 103: 'Up', 104: 'Up', 105: 'Down', 106: 'Up', 201: 'Down'}   # 103 / 105 never surface
TIE, UPIDX, BOOSTED = {9002}, {106: 2}, {102}
MKT_STATUS = {101: 'ok', 102: 'ok', 103: 'ok', 104: 'no_records', 105: 'boundary_censored'}
NAN = float('nan')
# sid -> (mid, settle offset ms, taker (addr, hash, quote, indexSet), taker fee, fee type, legs [(maker, hash, quote, indexSet, price, shares)], opts)
RECS = {
    'a1': (101, 10500, ('0xx1', 'tX1', 'Bid', 1), 0.1, 'SHARES', [('0x01', 'hM1a', 'Ask', 1, 0.53, 10.)], {}),
    'a2': (101, 20600, ('0xy1', 'tY1', 'Ask', 1), 0.0888, 'COLLATERAL', [('0x01', 'hM1b', 'Bid', 1, 0.56, 6.), ('0x0b', 'hM2a', 'Ask', 2, 0.45, 4.)], {}),
    'a3': (101, 70500, ('0xz1', 'tZ1', 'Bid', 1), 0.04 / 0.6, 'SHARES', [('0x0b', 'hM2b', 'Bid', 2, 0.40, 5.)], {}),
    'a4': (101, 299000, ('0xx1', 'tX2', 'Bid', 1), 0.0, 'SHARES', [('0x01', 'hM1a', 'Ask', 1, 0.53, 2.)], {}),
    'a5': (101, 20800, ('0xy2', 'tY2a', 'Ask', 1), 0.0, 'COLLATERAL', [('0x33', 'hM3', 'Bid', 1, 0.55, 3.)], {}),
    'a6': (101, 70600, ('0xy2', 'tY2b', 'Ask', 1), 0.0, 'COLLATERAL', [('0x33', 'hM3', 'Bid', 1, 0.55, 8.)], {}),
    'a7': (101, 150000, ('0xw2', 'tW2', 'Bid', 1), 0.0, 'SHARES', [('0xw1', 'hW1', 'Ask', 1, 0.60, 1.)], dict(rstatus='dup')),   # no fills row
    'a8': (101, 160000, ('0xw4', 'tW4', 'Bid', 1), 0.0, 'SHARES', [('0xw3', 'hW3', 'Ask', 1, 0.60, 1.)], dict(rstatus='dup')),   # lo > settle
    'b1': (102, 50000, ('0xx1', 'tX3', 'Bid', 1), 0.05, 'SHARES', [('0x01', 'hM1c', 'Ask', 1, 0.40, 5.)], dict(tk_price=0.41)),   # planted taker-price gap
    'b2': (102, 55000, ('0xq2', 'tQ2', 'Bid', 1), 0.0, 'SHARES', [('0xq1', 'hQ1', 'Bid', 1, 0.42, 3.)], {}),                       # maker side == taker side
    'b3': (102, 56000, ('0xq4', 'tQ4', 'Bid', 1), 0.0, 'SHARES', [('0xq3', 'hQ3', 'Ask', 1, 0.43, 2.)], dict(norec=True)),        # leg without record
    'b4': (102, 57000, ('0xq6', 'tQ6', 'Bid', 1), 0.0, 'SHARES', [('0xq5', 'hQ5', 'Ask', 1, 0.44, 2.), ('0xq7', 'hQ7', 'Ask', 1, 0.45, 0.)], {}),  # 0-share leg
    'b5': (103, 58000, ('0xq9', 'tQ9', 'Bid', 1), 0.0, 'SHARES', [('0xq8', 'hQ8', 'Ask', 1, 0.46, 1.)], dict(rstatus='conflict+boundary_censored')),
    'e1': (105, 1000, ('0xu2', 'tU2', 'Bid', 1), 0.0, 'SHARES', [('0xu1', 'hU1', 'Ask', 1, 0.50, 1.)], {}),              # censored by the markets table
    'u1': (106, 1000, ('0xv2', 'tV2', 'Bid', 1), 0.0, 'SHARES', [('0xv1', 'hV1', 'Ask', 1, 0.50, 1.)], {}),              # registry up_index 2
    'b10': (102, 63000, ('0xr8', 'tR8', 'Bid', 1), 0.0, 'SHARES', [('0xr7', 'hR7', 'Ask', 1, 0.48, 1.)], dict(rstatus='conflict')),
    # price priority: 0x33's two asks listed worse-price first (mk_idx 0 = .47); the taker 0xr6 fills .46 first. After END (no marks), unresolved
    'b8': (102, 299500, ('0xr6', 'tR6', 'Bid', 1), 0.0, 'SHARES', [('0x33', 'hX7', 'Ask', 1, 0.47, 2.), ('0x33', 'hX6', 'Ask', 1, 0.46, 3.)], {}),
    'b9': (102, 299600, ('0xr6', 'tR6b', 'Ask', 1), 0.0, 'COLLATERAL', [('0x33', 'hX9', 'Bid', 1, 0.50, 3.)], {}),
    'b6': (102, 59000, ('0xr2', 'tR2', 'Bid', 1), 0.01, 'WEIRD', [('0xr1', 'hR1', 'Ask', 1, 0.47, 1.)], {}),                      # unknown fee type
    'b7': (102, 60000, ('0xr4', 'tR4', 'Ask', 1), 0.0, 'SHARES', [('0xr3', 'hR3', 'Ask', 2, 0.45, 1.)], dict(ybad=True)),         # legs y_price != yes_space
    'c1': (201, 5000, ('0xy1', 'tY3', 'Bid', 2), 0.06, 'SHARES', [('0x01', 'hM1d', 'Ask', 2, 0.70, 10.)], {}),
    'c2': (201, 6000, ('0xx1', 'tX4', 'Ask', 1), 0.02, 'COLLATERAL', [('0x0b', 'hM2c', 'Bid', 1, 0.08, 4.)], dict(recv_early=True)),  # receipt 2 s before settle_ms
    's1': (201, 7000, ('0xs1', 'tS1', 'Bid', 1), 0.01, 'COLLATERAL', [('0xs1', 'hS1', 'Ask', 1, 0.25, 4.)], {}),                      # self-trade
}
# ID day: maker, side, shares -> P&L to resolution (Up, p 0.50): bid +q/2, ask -q/2
IDM = [('0x01', 'Bid', 100.), ('0x02', 'Ask', 90.), ('0x03', 'Bid', 80.), ('0xa4', 'Ask', 50.), ('0x05', 'Bid', 60.), ('0x06', 'Ask', 50.),
       ('0x07', 'Bid', 40.), ('0x08', 'Ask', 30.), ('0x09', 'Bid', 20.), ('0x10', 'Bid', 10.), ('0x0b', 'Bid', 10.), ('0x12', 'Ask', 5.)]
for i, (a, q_, sh) in enumerate(IDM):
    RECS[f'i{i:02d}'] = (9001, 1000 * (i + 1), ('0xt0', f'tT{i}', 'Ask' if q_ == 'Bid' else 'Bid', 1), 0.0, 'COLLATERAL', [(a, f'hI{i}', q_, 1, 0.50, sh)], {})
RECS['i12'] = (9001, 13000, ('0xc1', 'tC12', 'Bid', 1), 0.0, 'COLLATERAL', [('0xt0', 'hT12', 'Ask', 1, 0.01, 9.)], {})   # 0xc1 takes 9 @.01: +8.91
RECS['i13'] = (9001, 14000, ('0xt0', 'tT13', 'Ask', 1), 0.0, 'COLLATERAL', [('0xc1', 'hC13', 'Bid', 1, 0.50, 5.)], {})   # 0xc1 maker +2.50
RECS['i14'] = (9001, 15000, ('0x01', 'tA14', 'Bid', 1), 0.0, 'COLLATERAL', [('0xc1', 'hC14', 'Ask', 1, 0.50, 4.)], {})   # 0xc1 maker -2.00; 0x01 taker +2
RECS['i15'] = (9002, 1000, ('0xt0', 'tT15', 'Ask', 1), 0.0, 'COLLATERAL', [('0x07', 'hI15', 'Bid', 1, 0.40, 2.)], {})    # tie: 2 (.5 - .4) = +.20
# frames: (offset ms, best bid, best ask[, status]) - None = that side empty; f9 is a 'gap' frame (a full snapshot: usable)
FRAMES = {101: [(-1000, .48, .52), (8700, .49, .53), (10000, .50, .54), (11000, .52, .56), (20000, .55, .57), (20500, .56, .58), (30000, .58, None),
                (70000, .58, .62), (70800, .59, .63), (297500, .70, .74, 'gap')],
          102: [(49800, .38, .42), (50500, None, .52), (60000, .44, .46), (297000, .30, .34)],
          201: [(4900, .28, .32), (297900, .10, .14)]}
LO = {('a1', 0): 8700, ('a2', 0): 20000, ('a2', 1): 20500, ('a3', 0): -1, ('a4', 0): -1, ('a5', 0): 20500, ('a6', 0): 70000, ('a8', 0): 160500}
FA = {('a1', 0): 2, ('a2', 0): 5, ('a2', 1): 6, ('a5', 0): 5, ('a6', 0): 8, ('a8', 0): 9}          # lifecycle evidence frame ids (depth fids 0..9)
NO_LF, LF_ST = {'a7'}, {'a5': 'time_inversion', 'a3': 'fill_lo_unbounded'}
ORD = {'hM1a': ('0x01', 'ask', 53, 2, 12., 10500, 297500, 'cancelled_or_expired|end_censored', 0.),
       'hM1b': ('0x01', 'bid', 56, 1, 6., 20600, 20600, 'exhausted', 2.5),
       'hM2a': ('0x0b', 'bid', 55, 1, 4., 20600, 20600, 'exhausted|cancelled_or_expired', NAN),
       'hM2b': ('0x0b', 'ask', 60, 1, 5., 70500, 70500, 'end_censored', 0.),
       'hM3': ('0x33', 'bid', 55, 2, 11., 20800, 70000, 'end_censored', 0.)}


def iso(s):
    return dt.datetime.fromtimestamp(s, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')


def world():
    """The synthetic input tables: {name: DataFrame}."""
    legs, recs = [], []
    for sid, (mid, off, tk, fee, ft, lg, opt) in RECS.items():
        ws = WS[mid] * 1000
        settle = ws + off
        ys = [C.yes_space(q_, i_, p_) for (_, _, q_, i_, p_, _) in lg]
        Q = sum(x[5] for x in lg)
        tkp = opt.get('tk_price', round(sum(x[5] * (y[1] if tk[3] == 1 else 1 - y[1]) for x, y in zip(lg, ys)) / Q, 6))
        tys = C.yes_space(tk[2], tk[3], tkp)[0]
        com = dict(day=DAY[mid], mid=mid, sid=sid, tx='0xtx' + sid, settle_ms=settle, exec_ms=(settle // 1000 + 1) * 1000,
                   first_recv_ns=(settle + (-2000 if opt.get('recv_early') else 300)) * 10 ** 6)
        if not opt.get('norec'):
            recs.append(dict(com, n_recv=1, amount_filled=Q, price_executed=tkp, n_makers=len(lg), tk_hash=tk[1], tk_signer=tk[0], tk_quote=tk[2],
                             tk_index_set=tk[3], tk_price=tkp, tk_amount=Q, tk_fee=fee, tk_fee_type=ft, tk_y_side=tys, status=opt.get('rstatus', 'ok')))
        for k, (x, y) in enumerate(zip(lg, ys)):
            y = (y[0], x[4]) if opt.get('ybad') else y
            legs.append(dict(com, mk_idx=k, n_makers=len(lg), mk_hash=x[1], mk_signer=x[0], mk_quote=x[2], mk_index_set=x[3], mk_price=x[4], shares=x[5],
                             mk_fee=0., mk_fee_type='SHARES', y_side=y[0], y_price=y[1], y_ptick=int(round(y[1] * 100)), tk_hash=tk[1], tk_signer=tk[0],
                             tk_y_side=tys, phase='pre' if off < 0 else ('live' if off < 300000 else 'post'), t_rel_s=off / 1000., status='ok'))
    tob, dfr, dlv = [], [], []
    for mid, fr in FRAMES.items():
        for fid, f_ in enumerate(fr):
            off, b, a = f_[:3]
            u, fst = WS[mid] * 1000 + off, (f_[3] if len(f_) > 3 else 'ok')
            tob.append(dict(day=DAY[mid], mid=mid, upd_ms=u, recv_ns=(u + 53) * 10 ** 6, conn='c1', order_count=10, n_bid=int(b is not None),
                            n_ask=int(a is not None), best_bid=NAN if b is None else b, best_bid_sh=10., best_ask=NAN if a is None else a, best_ask_sh=10.,
                            pend_bid_usd=0., pend_ask_usd=0., status=fst))
            if mid == 101:   # depth: best levels 10 sh, a deeper level 4 ticks further out with 4 sh, on each present side
                dfr.append(dict(mid=mid, fid=fid, upd_ms=u, recv_ns=(u + 53) * 10 ** 6, conn='c1', order_count=10, pend_bids='', pend_asks='', los_id='',
                                los_side='', los_price=NAN, los_outcome='', status=fst))
                for side, px, sg in (('bid', b, -1), ('ask', a, 1)):
                    if px is not None:
                        dlv += [dict(mid=mid, fid=fid, side=side, ptick=int(round(px * 100)), shares=10., status='ok'),
                                dict(mid=mid, fid=fid, side=side, ptick=int(round((px + sg * .04) * 100)), shares=4., status='ok')]
                if fid == 2:     # a flagged level (off_grid, 50 sh) inside f2's spread: would become the best bid and add 50 sh-s if counted
                    dlv.append(dict(mid=mid, fid=fid, side='bid', ptick=51, shares=50., status='off_grid'))
    ws = WS[101] * 1000
    lfs = [dict(mid=101, sid=l_['sid'], mk_hash=l_['mk_hash'], mk_idx=l_['mk_idx'], y_side=l_['y_side'], y_ptick=l_['y_ptick'], shares=l_['shares'],
                hi_ms=l_['settle_ms'], lo_ms=(-1 if LO[(l_['sid'], l_['mk_idx'])] < 0 else ws + LO[(l_['sid'], l_['mk_idx'])]), t_method='synthetic',
                f_before=FA.get((l_['sid'], l_['mk_idx']), 0) - 1, f_after=FA.get((l_['sid'], l_['mk_idx']), -1), status=LF_ST.get(l_['sid'], 'ok'))
           for l_ in legs if l_['mid'] == 101 and l_['sid'] not in NO_LF]
    ords = []
    for h, (a, sd, tick, nf, lb, plo, phi, rs, ah) in ORD.items():
        for sc in C.SCENARIOS:
            ords.append(dict(scenario=sc, mid=101, mk_hash=h, mk_signer=a, y_side=sd, y_ptick=tick, n_fills=nf, cum_shares=lb, first_hi_ms=ws + plo,
                             last_hi_ms=ws + phi, size_lb=lb, size_exact=NAN, size_class='lower_bound', post_lo_ms=-1, post_hi_ms=ws + plo, post_class='unresolved',
                             post_method='none', persist_lo_ms=ws + plo, persist_hi_ms=ws + phi, persist_class='observed', ahead_lo=0.,
                             ahead_hi=(ah if sc == 'S_FIFO' else NAN), queue_class='assumption_sensitive',
                             removal_set=(rs if sc == 'S_FIFO' else 'exhausted|cancelled_or_expired|end_censored'), removal_lo_ms=-1, removal_hi_ms=-1,
                             removal_class='assumption_sensitive', removal_method='synthetic', status='ok'))
    reg = {d: pd.DataFrame([dict(mid=m, slug=f'btc-updown-5m-{WS[m]}', ws=float(WS[m]), we=float(WS[m] + 300), outcome=OUTCOME[m], up_index=UPIDX.get(m, 1),
                                 start_price=100., end_price=100. if m in TIE else 101., fee_bps=200, spread_thr=0.06, share_thr=100, decimal_precision=2)
                            for m in WS if DAY[m] == d]).astype({'outcome': 'str'}) for d in (ID, D1, D2)}
    sch = [dict(hourlyRate=3000, startsAt=iso(WS[101]), endsAt=iso(WS[101] + 300))]
    rw = json.dumps(dict(current=None, schedule=sch))
    rv = json.dumps([dict(first_recv_ns=1, rewards=dict(current=None, schedule=sch)), dict(first_recv_ns=2, rewards=dict(current=dict(hourlyRate=3000), schedule=sch))])
    mkts = pd.DataFrame([dict(mid=m, feeRateBps=200, makerFeeBps=0, spreadThreshold=0.06, shareThreshold=5 if m == 101 else 100, isBoosted=m in BOOSTED,
                              boostStartsAt=iso(WS[m]), boostEndsAt=iso(WS[m] + 300), rewards=rw if m == 101 else None,
                              rewards_variants=rv if m == 101 else None, status=MKT_STATUS[m]) for m in (101, 102, 103, 104, 105)])
    L, R, T = pd.DataFrame(legs), pd.DataFrame(recs), pd.DataFrame(tob)
    return dict(legs=L, records=R, tob=T, fills=pd.DataFrame(lfs), orders=pd.DataFrame(ords), reg=reg, markets=mkts, dframes=pd.DataFrame(dfr),
                dlevels=pd.DataFrame(dlv))


def write(root, W):
    """The world as the pipeline's files under `root` (OUT layout + step1 registries)."""
    root = Path(root)
    (root / 'life').mkdir(parents=True, exist_ok=True)
    (root / 'depth').mkdir(parents=True, exist_ok=True)
    for d in (ID, D1, D2):
        for k in ('legs', 'records'):
            x = W[k]
            x[x.day == d][[c for c in C.SCHEMAS[k] if c in x.columns]].to_parquet(root / f'{k}_{d}.parquet', index=False)
        if d != ID:
            W['tob'][W['tob'].day == d][list(C.SCHEMAS['tob'])].to_parquet(root / f'tob_{d}.parquet', index=False)
        (root / 'step1' / d).mkdir(parents=True, exist_ok=True)
        W['reg'][d].to_parquet(root / 'step1' / d / 'registry.parquet', index=False)
    W['markets'].to_parquet(root / f'markets_{D1}.parquet', index=False)
    W['fills'][list(C.SCHEMAS['fills'])].to_parquet(root / 'life' / f'{D1}_101_fills.parquet', index=False)
    W['orders'][list(C.SCHEMAS['orders'])].to_parquet(root / 'life' / f'{D1}_101_orders.parquet', index=False)
    W['dframes'][list(C.SCHEMAS['frames'])].to_parquet(root / 'depth' / f'{D1}_101_frames.parquet', index=False)
    W['dlevels'][list(C.SCHEMAS['levels'])].to_parquet(root / 'depth' / f'{D1}_101_levels.parquet', index=False)


def go(root, **kw):
    a = dict(inp=root, out=Path(root) / 'econ', step1=Path(root) / 'step1', book={D1: [101]}, days=(ID, D1, D2), chunk=1, log=lambda *x, **k: None)
    a.update(kw)
    return E.run(**a)


N_OK = [0]


def eq(a, b, what):
    if b is None:
        need(a is None or (isinstance(a, float) and math.isnan(a)), f'{what}: expected NaN/None, got {a!r}')
    elif isinstance(b, str) or isinstance(b, (bool, np.bool_)):
        need(a == b, f'{what}: expected {b!r}, got {a!r}')
    else:
        need(a is not None and not (isinstance(a, float) and math.isnan(a)) and abs(float(a) - float(b)) <= 1e-9, f'{what}: expected {b!r}, got {a!r}')
    N_OK[0] += 1


def refused(fn, reason, what):
    try:
        fn()
    except ValueError as e:
        need(reason in str(e), f'{what}: refused for the wrong reason: {e}')
        N_OK[0] += 1
        return
    raise AssertionError(f'{what}: not refused (expected: {reason})')


# ---- expected fill rows, by hand. Spread anchor t: a1 lo 8700 -> f1 .51 (win 1800 ms); a2 max lo 20500 -> f5 .57; a3 settle 70500 -> f7 .60
#      (age 500); a5 settle 20800 (flagged fills row) -> f5 age 300; a6 lo 70000 -> f7 .60; b1 settle 50000 -> g0 .40 (age 200); c1 settle 5000 ->
#      h0 .30. Markouts and END from settle_ms everywhere: a1 +1 s 11500 -> f3 .54, +10 s 20500 -> f5 .57, +60 s 70500 -> f7 .60 (age 500);
#      a2 +1 s 21600 -> f5 age 1100 stale; a6 +1 s 71600 -> f8 .61 (age 800). Depth at the m_minus frame (ticks from the best on the maker side):
#      a1 ask .53 vs .53 d0; a2 bid .56 vs .56 d0, .55 -> d1; a3 ask .60 vs .62 inside; a5 bid .55 vs .56 d1; a6 bid .55 vs .58 d3p; b1 ask .40
#      vs .42 inside; c1 bid .30 vs .28 inside. reb25 = .25 x the leg's taker fee (res / fill) ----
X_ = None
ROWS = {
    # a1 maker sells 10 @.53: spr -10(.51-.53) = +.20; at settle 10500 (f2 .52): -10(.52-.53) = +.10; 1s -10(.54-.53) = -.10; 10s -.40; 60s -.70;
    #    END f9 .72 (a gap frame: usable) -> -1.90; res -4.70; reb25 .25 x .1 = .025 (res), .25 x .053 = .01325 (fill)
    ('a1', 0, 'maker'): dict(addr='0x01', kind='regular', s=-1, q=10., p=.53, t_status='lo', t_ms=WS[101] * 1000 + 8700, win_ms=1800, m_minus=.51, age_ms=0,
                             spr_w=.04, st_spr='ok', v_spr=.20, st_spr_hi='ok', v_spr_hi=.10, st_1s='ok', v_1s=-.10, st_10s='ok', v_10s=-.40, st_60s='ok',
                             v_60s=-.70, st_end='ok', v_end=-1.90, st_res='ok', gross_res=-4.70, fee_fill=0., fee_res=0., net_res=-4.70, reb25_res=.025,
                             reb25_fill=.01325, status='ok', life_rep='repeated', life_exh='not_exhausted', life_front='front', life_status='ok', tseg='w000_060',
                             pseg='d_0.0_0.1', dpt='d0', sweep='single_price', cp='cp_other', boost='not_boosted', mq=6., rq=4.),
    ('a1', 0, 'taker'): dict(addr='0xx1', kind='regular', s=1, v_spr=-.20, v_spr_hi=-.10, v_1s=.10, v_10s=.40, v_60s=.70, v_end=1.90, gross_res=4.70,
                             fee_type='SHARES', fee_dy=-.1, fee_dc=0., fee_fill=.053, fee_res=.1, net_res=4.60, reb25_res=0., reb25_fill=0., cp='cp_top',
                             life_rep='na_taker', status='ok', mq=0., rq=10.),
    # a2 leg 0 maker buys 6 @.56: spr 6(.57-.56) = .06 (= at settle 20600, f5); 1s @21600 f5 age 1100 stale; 10s @30600 f6 one-sided; 60s @80600 f8
    #    age 9800 stale; END 6(.72-.56) = .96; res 2.64; reb25 .25 x .05328 = .01332
    ('a2', 0, 'maker'): dict(addr='0x01', kind='regular', s=1, q=6., p=.56, px_rank=0, px_gap=0., t_status='lo', win_ms=100, v_spr=.06, spr_w=.02, v_spr_hi=.06,
                             st_1s='stale', v_1s=X_, st_10s='one_sided', st_60s='stale', v_end=.96, gross_res=2.64, life_rep='single', life_exh='exhausted',
                             life_front='behind', dpt='d0', sweep='sweep', reb25_res=.01332, reb25_fill=.01332, mq=6., rq=0.),
    ('a2', 0, 'taker'): dict(addr='0xy1', s=-1, v_spr=-.06, v_end=-.96, gross_res=-2.64, fee_type='COLLATERAL', fee_dy=0., fee_dc=-.05328, fee_fill=.05328,
                             fee_res=.05328, net_res=-2.64 - .05328, cp='cp_top'),
    # a2 leg 1 maker NO Ask .45 = YES bid .55 (merge): spr 4(.57-.55) = .08; END .68; res 1.80; reb25 .25 x .03552 = .00888
    ('a2', 1, 'maker'): dict(addr='0x0b', kind='merge', s=1, q=4., p=.55, px_rank=1, v_spr=.08, st_1s='stale', st_10s='one_sided', st_60s='stale', v_end=.68,
                             gross_res=1.80, life_rep='single', life_exh='exhausted_or_other', life_front='unknown', dpt='d1', sweep='sweep', reb25_res=.00888,
                             mq=4., rq=0.),
    ('a2', 1, 'taker'): dict(addr='0xy1', kind='merge', s=-1, v_spr=-.08, gross_res=-1.80, fee_fill=.03552, fee_res=.03552),
    # a3 maker NO Bid .40 = YES ask .60 (mint); lo unbounded -> t = settle 70500, f7 .60 age 500: spr 0 (= spr at settle); 1s @71500 f8 .61 age 700
    #    -> -.05; END -.60; res -2.00; reb25 .25 x .04/.6 (res), .25 x .04 = .01 (fill)
    ('a3', 0, 'maker'): dict(addr='0x0b', kind='mint', s=-1, q=5., p=.60, own_index_set=2, t_status='hi_lo_unbounded', win_ms=-1, v_spr=0., age_ms=500, spr_w=.04,
                             v_spr_hi=0., v_1s=-.05, st_10s='stale', st_60s='stale', v_end=-.60, gross_res=-2.00, life_rep='single', life_exh='not_exhausted',
                             life_front='front', tseg='w060_120', pseg='d_0.1_0.2', dpt='inside', reb25_res=.25 * .04 / .6, reb25_fill=.01, mq=4., rq=1.),
    ('a3', 0, 'taker'): dict(addr='0xz1', kind='mint', s=1, v_spr=0., v_1s=.05, v_end=.60, gross_res=2.00, fee_fill=.04, fee_res=.04 / .6, net_res=2. - .04 / .6),
    # a4: settle 299000 > END 298000: spread f9 age 1500 stale, markouts beyond_end, END after_end; res -2(.47) = -.94
    ('a4', 0, 'maker'): dict(addr='0x01', s=-1, q=2., t_status='hi_lo_unbounded', st_spr='stale', v_spr=X_, age_ms=1500, st_1s='beyond_end', v_1s=X_,
                             st_10s='beyond_end', st_60s='beyond_end', st_end='after_end', v_end=X_, gross_res=-.94, life_rep='repeated', tseg='w240_300',
                             dpt='unknown', mq=0., rq=2.),
    ('a4', 0, 'taker'): dict(addr='0xx1', gross_res=.94, fee_fill=0., fee_res=0.),
    # a5's fills row is flagged (time_inversion): t = settle 20800 (hi_fill_status), f5 .57 age 300: spr .06; 1s @21800 stale; END .51; res 1.35
    ('a5', 0, 'maker'): dict(addr='0x33', s=1, t_status='hi_fill_status', age_ms=300, v_spr=.06, st_1s='stale', v_1s=X_, v_end=.51, gross_res=1.35,
                             life_rep='repeated', life_exh='not_exhausted', life_front='front', dpt='d1'),
    # a6 buys 8 @.55 at lo 70000 (f7 .60): spr .40 (settle 70600 -> f7: .40); 1s @71600 f8 .61 -> .48; END 1.36; res 3.60; win 600
    ('a6', 0, 'maker'): dict(addr='0x33', s=1, q=8., t_status='lo', win_ms=600, v_spr=.40, v_spr_hi=.40, v_1s=.48, st_10s='stale', v_end=1.36, gross_res=3.60,
                             dpt='d3p'),
    ('a6', 0, 'taker'): dict(addr='0xy2', s=-1, v_spr=-.40, gross_res=-3.60, fee_fill=0.),
    ('a7', 0, 'maker'): dict(addr='0xw1', status='src_status', src_status='dup', t_status='hi_missing_life_fill', t_ms=WS[101] * 1000 + 150000, win_ms=-1),
    ('a8', 0, 'maker'): dict(addr='0xw3', status='src_status', t_status='hi_window_inverted', t_ms=WS[101] * 1000 + 160000, win_ms=-1),
    # b1 (no resolution, not BOOK, boosted): t = settle 50000, g0 .40 age 200: spr 0; 1s g1 one-sided; 10s @60000 g2 .45 -> -.25; 60s stale;
    #    END g3 .32 age 1000 ok -> +.40; res NaN; reb25 fill .25 x .02 = .005, res NaN
    ('b1', 0, 'maker'): dict(addr='0x01', s=-1, q=5., p=.40, px_gap=.01, t_status='hi_no_lifecycle', v_spr=0., spr_w=.04, st_1s='one_sided', st_10s='ok', v_10s=-.25,
                             st_60s='stale', st_end='ok', v_end=.40, st_res='no_resolution', gross_res=X_, net_res=X_, life_rep='na_not_book',
                             life_status='na_not_book', pseg='d_0.1_0.2', dpt='inside', boost='boosted', reb25_fill=.005, reb25_res=X_, mq=0., rq=5.),
    ('b1', 0, 'taker'): dict(addr='0xx1', s=1, v_10s=.25, v_end=-.40, gross_res=X_, fee_fill=.02, fee_res=X_, cp='cp_top', boost='boosted'),
    ('b2', 0, 'maker'): dict(addr='0xq1', status='side_mismatch', mq=X_),
    ('b2', 0, 'taker'): dict(addr='0xq2', status='side_mismatch'),
    ('b3', 0, 'maker'): dict(addr='0xq3', status='no_record'),
    ('b3', 0, 'taker'): dict(addr='0xq4', status='no_record'),
    ('b4', 0, 'maker'): dict(addr='0xq5', status='record_leg_excluded'),
    ('b4', 1, 'maker'): dict(addr='0xq7', status='bad_leg'),
    ('b4', 1, 'taker'): dict(addr='0xq6', status='bad_leg'),
    ('b5', 0, 'maker'): dict(addr='0xq8', status='boundary_censored', src_status='conflict+boundary_censored', V=X_, st_res='censored', gross_res=X_),
    ('b5', 0, 'taker'): dict(addr='0xq9', status='boundary_censored', V=X_, st_res='censored'),
    ('e1', 0, 'maker'): dict(addr='0xu1', status='boundary_censored', src_status='ok', V=X_, st_res='censored', gross_res=X_, net_res=X_),
    ('u1', 0, 'maker'): dict(addr='0xv1', status='up_index', V=X_, st_res='up_index', gross_res=X_),
    ('b10', 0, 'maker'): dict(addr='0xr7', status='src_status', src_status='conflict'),
    # b8: 0x33's asks .47 (mk_idx 0) and .46 (mk_idx 1); price priority fills .46 first (px_rank 0); the later buy (b9) closes the .46 lot
    ('b8', 0, 'maker'): dict(addr='0x33', px_rank=1, p=.47, mq=0., rq=2., st_spr='stale', st_1s='beyond_end', st_end='after_end', status='ok', sweep='sweep',
                             boost='boosted', dpt='unknown'),
    ('b8', 1, 'maker'): dict(addr='0x33', px_rank=0, p=.46, mq=3., rq=0.),
    ('b8', 0, 'taker'): dict(addr='0xr6', px_rank=1, mq=0., rq=2.),
    ('b8', 1, 'taker'): dict(addr='0xr6', px_rank=0, mq=3., rq=0.),
    ('b9', 0, 'maker'): dict(addr='0x33', s=1, mq=3., rq=0., sweep='single_price'),
    ('b6', 0, 'taker'): dict(addr='0xr2', status='fee_type_unknown'),
    ('b7', 0, 'maker'): dict(addr='0xr3', status='yes_space_mismatch'),
    # c1 (Down): maker NO Ask .70 = YES bid .30: spr 0 (h0 .30, bid .28 -> inside); END h1 .12 -> 10(.12-.30) = -1.80; res -3.00; taker NO Bid
    #    SHARES .06 NO shares: fill .06 x .70 = .042, resolution .06 x (1 - V) = .06; reb25 .015 / .0105; no markets table -> boost unknown
    ('c1', 0, 'maker'): dict(addr='0x01', kind='regular', s=1, q=10., p=.30, own_index_set=2, v_spr=0., spr_w=.04, st_1s='stale', v_end=-1.80, gross_res=-3.00,
                             pseg='d_0.2_0.3', dpt='inside', boost='unknown', reb25_res=.015, reb25_fill=.0105),
    ('c1', 0, 'taker'): dict(addr='0xy1', s=-1, own_index_set=2, v_end=1.80, gross_res=3.00, fee_dy=.06, fee_dc=-.06, fee_fill=.042, fee_res=.06, net_res=2.94),
    ('c2', 0, 'maker'): dict(addr='0x0b', s=1, p=.08, t_ms=WS[201] * 1000 + 6000, st_spr='stale', v_spr=X_, v_end=.16, gross_res=-.32, pseg='p_le_0.10',
                             dpt='unknown', reb25_res=.005, reb25_fill=.005),
    ('c2', 0, 'taker'): dict(addr='0xx1', gross_res=.32, fee_fill=.02, fee_res=.02, net_res=.30),
    ('s1', 0, 'maker'): dict(addr='0xs1', self_trade=True, s=-1, gross_res=1.00, mq=0., rq=0., status='ok', reb25_res=.0025),
    ('s1', 0, 'taker'): dict(addr='0xs1', self_trade=True, s=1, gross_res=-1.00, fee_res=.01, net_res=-1.01, mq=0., rq=0.),
    # ID day: 0x01 bids 100 @.50 (Up) -> +50; tie market 9002: V = .5 although its registry says Up -> 0x07 2(.5 - .4) = +.20; 0xc1 takes 9 @.01
    ('i00', 0, 'maker'): dict(addr='0x01', s=1, q=100., gross_res=50., st_spr='id_day_no_book', st_end='id_day_no_book', t_status='hi_no_lifecycle',
                              dpt='unknown', cp='na_id_day'),
    ('i00', 0, 'taker'): dict(addr='0xt0', s=-1, gross_res=-50.),
    ('i15', 0, 'maker'): dict(addr='0x07', V=.5, st_res='tie', gross_res=.2),
    ('i12', 0, 'taker'): dict(addr='0xc1', s=1, q=9., p=.01, gross_res=8.91),
}
# ---- maker_market by hand (A: Up, END .72; FIFO as before: 0x01 rt 6(.53-.56) = -.18, short 6 @.53 -> -2.82 = -1.14 - 1.68) ----
MMX = {
    (D1, 101, '0x01'): dict(n_mk=3, sh_mk=18., n_tk=0, sh_buy=6., sh_sell=12., gross_res_mk=-3.00, sh_res_mk=18., net_res_mk=-3.00, rt_sh=6., rt_pnl=-.18,
                            net_pos=-6., resid_res=-2.82, resid_drift=-1.14, resid_resol=-1.68, v_spr_mk=.26, sh_spr_mk=16., v_1s_mk=-.10, sh_1s_mk=10.,
                            v_60s_mk=-.70, sh_60s_mk=10., v_end_mk=-.94, sh_end_mk=16., reb25_res=.025 + .01332, reb25_fill=.01325 + .01332, fifo_status='ok',
                            rew_status='ok', rew_bid_ms=0., rew_ask_ms=0., rew_both_ms=0., rew_n_orders=0, is_maker=True, book=True),
    (D1, 101, '0x0b'): dict(sh_mk=9., rt_sh=4., rt_pnl=.20, net_pos=-1., resid_res=-.40, resid_drift=-.12, resid_resol=-.28, gross_res_mk=-.20,
                            reb25_res=.00888 + .25 * .04 / .6, reb25_fill=.00888 + .01),
    (D1, 101, '0x33'): dict(sh_mk=11., rt_sh=0., rt_pnl=0., net_pos=11., resid_res=4.95, resid_drift=1.87, resid_resol=3.08, rew_bid_ms=700., rew_ask_ms=0.,
                            rew_both_ms=0., rew_n_orders=1, rew_status='ok'),
    (D1, 101, '0xx1'): dict(n_mk=0, n_tk=2, sh_tk=12., gross_res_tk=5.64, net_pos=12., resid_drift=2.28, resid_resol=3.36, fee_res=.1, fee_fill=.053, net_res=5.54,
                            net_res_mk=0., reb25_res=0., is_maker=False),
    (D1, 101, '0xy1'): dict(sh_tk=10., gross_res_tk=-4.44, resid_drift=-1.64, resid_resol=-2.80, fee_res=.0888, net_res=-4.5288),
    (D1, 101, '0xz1'): dict(gross_res_tk=2.00, fee_res=.04 / .6, fee_fill=.04, net_res=2. - .04 / .6, resid_drift=.60, resid_resol=1.40),
    (D1, 102, '0x01'): dict(sh_mk=5., n_res_na=1, net_pos=-5., rt_pnl=0., resid_res=X_, resid_drift=.40, resid_resol=X_, fifo_status='no_resolution',
                            res_status='no_resolution', end_status='ok', rew_status='not_book', book=False, reb25_fill=.005, reb25_res=0.),
    (D1, 102, '0xx1'): dict(fee_fill=.02, fee_res=0., n_res_na=1, resid_drift=-.40),
    (D1, 102, '0xq1'): dict(n_mk=0, n_excl=1, status='all_rows_excluded'),
    (D1, 102, '0xq4'): dict(n_tk=0, n_excl=1, status='all_rows_excluded'),
    (D1, 102, '0xq6'): dict(n_tk=0, n_excl=2, status='all_rows_excluded'),
    (D1, 103, '0xq8'): dict(n_mk=0, n_excl=1, status='all_rows_excluded', res_status='censored', fifo_status='censored'),
    (D1, 105, '0xu1'): dict(n_excl=1, res_status='censored'),
    (D1, 106, '0xv1'): dict(n_excl=1, res_status='up_index', fifo_status='up_index', status='all_rows_excluded'),
    (D1, 102, '0x33'): dict(sh_mk=8., sh_buy=3., sh_sell=5., rt_sh=3., rt_pnl=-.12, net_pos=-2.),
    (D1, 102, '0xr6'): dict(sh_tk=8., rt_sh=3., rt_pnl=.12, net_pos=2.),
    (D2, 201, '0x01'): dict(sh_mk=10., gross_res_mk=-3.00, net_pos=10., resid_res=-3.00, resid_drift=-1.80, resid_resol=-1.20, v_spr_mk=0., sh_spr_mk=10.,
                            v_end_mk=-1.80, rew_status='not_book', reb25_res=.015, reb25_fill=.0105),
    (D2, 201, '0x0b'): dict(gross_res_mk=-.32, resid_drift=.16, resid_resol=-.48, v_spr_mk=0., sh_spr_mk=0., v_end_mk=.16, sh_end_mk=4.),
    (D2, 201, '0xy1'): dict(gross_res_tk=3.00, fee_res=.06, fee_fill=.042, net_res=2.94, resid_drift=1.80, resid_resol=1.20),
    (D2, 201, '0xs1'): dict(n_mk=1, n_tk=1, sh_buy=4., sh_sell=4., gross_res_mk=1.00, gross_res_tk=-1.00, fee_res=.01, net_res=-.01, rt_sh=0., rt_pnl=0.,
                            net_pos=0., resid_res=0., resid_drift=0., resid_resol=0., fifo_status='ok'),
    (ID, 9002, '0x07'): dict(res_status='tie', gross_res_mk=.2, net_res_mk=.2),
}
# ---- maker_day by hand. D1 0x01 = A + B: sh 23; maker-leg net -3.00 (B unresolved); rt -.18 (all resolved: rt_pnl_res -.18); 1s -.10 on 10 sh;
#      10s -.40 + -.25 = -.65 on 15; 60s -.70 on 10; reb25 res .025 + .01332, fill .01325 + .01332 + .005. 0x33: rt -.12 in B (unresolved) ->
#      rt_pnl_res 0. P&L ranks (maker-majority, maker-leg net): D1 0x33 4.95 > 0x0b -.20 > 0x01 -3.00; D2 0x0b -.32 > 0x01 -3.00, 0xs1 (4 = 4)
#      not maker-majority. ID: 0x01 52 total (taker +2), 50 maker-leg; 0xc1 9.41 total, 0.5 maker-leg, 9 = 9 -> not a P&L candidate ----
MDX = {
    (D1, '0x01'): dict(n_markets=2, n_markets_mk=2, sh_mk=23., gross_res_mk=-3.00, sh_res_mk=18., net_res_mk=-3.00, n_res_na=1, rt_sh=6., rt_pnl=-.18,
                       rt_pnl_res=-.18, net_pos=-11., resid_res=-2.82, resid_drift=-.74, n_mkt_res_na=1, n_mkt_end_na=0, v_spr_mk=.26, sh_spr_mk=21.,
                       v_1s_mk=-.10, sh_1s_mk=10., v_10s_mk=-.65, sh_10s_mk=15., v_60s_mk=-.70, sh_60s_mk=10., v_end_mk=-.54, sh_end_mk=21.,
                       reb25_res=.025 + .01332, reb25_fill=.01325 + .01332 + .005, rank_sh=1, rank_pnl=3, in_exante_sh=True, in_exante_pnl=True,
                       in_sameday_sh=True, in_sameday_pnl=True, status='ok'),
    (D1, '0x33'): dict(sh_mk=19., rank_sh=2, rank_pnl=1, in_exante_sh=False, in_exante_pnl=False, rew_bid_ms=700., rt_pnl=-.12, rt_pnl_res=0.),
    (D1, '0x0b'): dict(rank_sh=3, rank_pnl=2, net_res=-.20),
    (D1, '0xq1'): dict(n_mk=0, rank_sh=-1, status='all_rows_excluded'),
    (D2, '0x01'): dict(sh_mk=10., gross_res_mk=-3.00, rank_sh=1, rank_pnl=2),
    (D2, '0x0b'): dict(rank_sh=2, rank_pnl=1),
    (D2, '0xs1'): dict(rank_sh=3, rank_pnl=-1, net_res=-.01, in_exante_sh=False, in_sameday_sh=True, in_sameday_pnl=False),
    (ID, '0x01'): dict(sh_mk=100., sh_tk=4., net_res=52., net_res_mk=50., rank_sh=1, rank_pnl=1, in_exante_sh=True, in_exante_pnl=True),
    (ID, '0x07'): dict(sh_mk=42., net_res_mk=20.2, rank_sh=7, rank_pnl=4),
    (ID, '0xc1'): dict(sh_mk=9., sh_tk=9., net_res=9.41, net_res_mk=.5, rank_sh=12, rank_pnl=-1, in_exante_pnl=False),
    (ID, '0xt0'): dict(n_mk=1, sh_mk=9., sh_tk=552., net_res=-59.11, rank_sh=13, rank_pnl=-1),
}
# ---- segments by hand, D1, set exante_shares (= {0x01, 0x02, 0x03, 0x05, 0x06, 0xa4, 0x07, 0x08, 0x09, 0x0b}): in = 0x01 + 0x0b (rows a1 a2.0 a4
#      b1 a2.1 a3), field = 0x33 (a5 a6 b8 b8 b9), nonmaker = 0xx1 0xy1 0xz1 0xy2 0xr6. in/maker/all: sh 32; gross -3.20 on 27; spr .34 on 30;
#      1s a1 -.10 + a3 -.05 = -.15 on 15; 10s a1 -.40 + b1 -.25 = -.65 on 15; 60s a1 -.70 on 10; END -.46 on 30; reb25 res .025 + .01332 + .00888
#      + .25 x .04/.6, fill .01325 + .01332 + .00888 + .01 + .005 = .05045. field: gross 4.95, spr .46, 1s .48 (a6), END 1.87. nonmaker/taker: gross
#      -1.75; 1s a1 +.10, a3 +.05, a6 -.48 = -.33 ----
SEGX = {
    (D1, 'exante_shares', 'in', 'maker', 'all', 'all'): dict(n_addr=2, n=6, sh=32., notional=16.92, gross_res=-3.20, sh_res=27., v_spr=.34, sh_spr=30., v_1s=-.15,
                                                              sh_1s=15., v_10s=-.65, sh_10s=15., v_60s=-.70, sh_60s=10., v_end=-.46, sh_end=30., fee_res=0.,
                                                              reb25_res=.025 + .01332 + .00888 + .25 * .04 / .6, reb25_fill=.05045, label='ex_ante',
                                                              ci='none_G2_lt_5', n_res_na=1),
    (D1, 'exante_shares', 'field', 'maker', 'all', 'all'): dict(n_addr=1, n=5, sh=19., gross_res=4.95, v_spr=.46, v_1s=.48, sh_1s=8., v_10s=0., sh_10s=0.,
                                                                 v_end=1.87),
    (D1, 'exante_shares', 'nonmaker', 'taker', 'all', 'all'): dict(n_addr=5, n=11, sh=51., gross_res=-1.75, fee_res=.1 + .0888 + .04 / .6, fee_fill=.2018,
                                                                    net_res=-1.75 - (.1 + .0888 + .04 / .6), v_spr=-.80, v_1s=-.33, v_end=-1.41, reb25_res=0.),
    (D1, 'exante_shares', 'in', 'maker', 'life_rep', 'repeated'): dict(n=2, sh=12., gross_res=-5.64),
    (D1, 'exante_shares', 'in', 'maker', 'life_rep', 'single'): dict(n=3, sh=15., gross_res=2.44),
    (D1, 'exante_shares', 'in', 'maker', 'life_rep', 'na_not_book'): dict(n=1, sh=5., gross_res=0., sh_res=0.),
    (D1, 'exante_shares', 'in', 'maker', 'life_exh', 'not_exhausted'): dict(n=3, sh=17., gross_res=-7.64),
    (D1, 'exante_shares', 'in', 'maker', 'life_exh', 'exhausted'): dict(n=1, sh=6., gross_res=2.64),
    (D1, 'exante_shares', 'in', 'maker', 'life_exh', 'exhausted_or_other'): dict(n=1, sh=4., gross_res=1.80),
    (D1, 'exante_shares', 'in', 'maker', 'life_front', 'front'): dict(n=3, sh=17., gross_res=-7.64),
    (D1, 'exante_shares', 'in', 'maker', 'life_front', 'behind'): dict(n=1, sh=6., gross_res=2.64),
    (D1, 'exante_shares', 'in', 'maker', 'life_front', 'unknown'): dict(n=1, sh=4., gross_res=1.80),
    (D1, 'exante_shares', 'field', 'maker', 'life_rep', 'repeated'): dict(n=2, sh=11., gross_res=4.95),
    (D1, 'exante_shares', 'nonmaker', 'taker', 'life_rep', 'na_taker'): dict(n=11, sh=51.),
    (D1, 'exante_shares', 'field', 'maker', 'life_rep', 'na_not_book'): dict(n=3, sh=8.),
    (D1, 'exante_shares', 'in', 'maker', 'tclock', 'lo'): dict(n=3, sh=20., gross_res=-.26),
    (D1, 'exante_shares', 'in', 'maker', 'tclock', 'hi'): dict(n=3, sh=12., gross_res=-2.94),
    (D1, 'exante_shares', 'in', 'maker', 'kind', 'regular'): dict(n=4, sh=23., gross_res=-3.00),
    (D1, 'exante_shares', 'in', 'maker', 'kind', 'merge'): dict(n=1, sh=4., gross_res=1.80),
    (D1, 'exante_shares', 'in', 'maker', 'kind', 'mint'): dict(n=1, sh=5., gross_res=-2.00),
    (D1, 'exante_shares', 'in', 'maker', 'tseg', 'w000_060'): dict(n=4, sh=25., gross_res=-.26),
    (D1, 'exante_shares', 'in', 'maker', 'tseg', 'w060_120'): dict(n=1, sh=5., gross_res=-2.00),
    (D1, 'exante_shares', 'in', 'maker', 'tseg', 'w240_300'): dict(n=1, sh=2., gross_res=-.94),
    (D1, 'exante_shares', 'in', 'maker', 'pseg', 'd_0.0_0.1'): dict(n=4, sh=22., gross_res=-1.20),
    (D1, 'exante_shares', 'in', 'maker', 'pseg', 'd_0.1_0.2'): dict(n=2, sh=10., gross_res=-2.00),
    # new dims: depth (a1 d0 10 / a2.0 d0 6 / a2.1 d1 4 / a3 inside 5 / b1 inside 5 / a4 unknown 2), sweep (a2 legs), cp, boost (b1 in 102)
    (D1, 'exante_shares', 'in', 'maker', 'depth', 'd0'): dict(n=2, sh=16., gross_res=-2.06),
    (D1, 'exante_shares', 'in', 'maker', 'depth', 'd1'): dict(n=1, sh=4., gross_res=1.80),
    (D1, 'exante_shares', 'in', 'maker', 'depth', 'inside'): dict(n=2, sh=10., gross_res=-2.00),
    (D1, 'exante_shares', 'in', 'maker', 'depth', 'unknown'): dict(n=1, sh=2., gross_res=-.94),
    (D1, 'exante_shares', 'field', 'maker', 'depth', 'd3p'): dict(n=1, sh=8., gross_res=3.60),
    (D1, 'exante_shares', 'in', 'maker', 'sweep', 'sweep'): dict(n=2, sh=10., gross_res=4.44),
    (D1, 'exante_shares', 'in', 'maker', 'sweep', 'single_price'): dict(n=4, sh=22., gross_res=-7.64),
    (D1, 'exante_shares', 'in', 'maker', 'cp', 'cp_other'): dict(n=6, sh=32.),
    (D1, 'exante_shares', 'nonmaker', 'taker', 'cp', 'cp_top'): dict(n=6, sh=32., gross_res=3.20),
    (D1, 'exante_shares', 'nonmaker', 'taker', 'cp', 'cp_other'): dict(n=5, sh=19., gross_res=-4.95),
    (D1, 'exante_shares', 'in', 'maker', 'boost', 'boosted'): dict(n=1, sh=5., gross_res=0.),
    (D1, 'exante_shares', 'in', 'maker', 'boost', 'not_boosted'): dict(n=5, sh=27., gross_res=-3.20),
    (D1, 'exante_shares', 'in', 'maker', 'self', 'not_self'): dict(n=6, sh=32.),
    (D1, 'sameday_pnl', 'in', 'maker', 'all', 'all'): dict(n_addr=3, sh=51., gross_res=1.75, label='descriptive'),
    (D2, 'exante_shares', 'in', 'maker', 'all', 'all'): dict(n_addr=2, n=2, sh=14., gross_res=-3.32, v_spr=0., sh_spr=10., v_end=-1.64, sh_end=14., reb25_res=.02,
                                                              reb25_fill=.0155),
    (D2, 'exante_shares', 'nonmaker', 'taker', 'all', 'all'): dict(n_addr=2, gross_res=3.32, fee_res=.08, fee_fill=.062),
    (D2, 'exante_shares', 'in', 'maker', 'pseg', 'p_le_0.10'): dict(n=1, sh=4., gross_res=-.32),
    (D2, 'exante_shares', 'field', 'maker', 'all', 'all'): dict(n_addr=1, n=1, sh=4., gross_res=1.00),
    (D2, 'exante_shares', 'field', 'taker', 'all', 'all'): dict(n_addr=1, n=1, sh=4., gross_res=-1.00, fee_res=.01, net_res=-1.01),
    (D2, 'exante_shares', 'field', 'maker', 'self', 'self'): dict(n=1, sh=4., gross_res=1.00),
    (D2, 'exante_shares', 'field', 'taker', 'self', 'self'): dict(n=1, sh=4., gross_res=-1.00),
}
# ---- rewards_market by hand (101: window [ws, we]; usable two-sided depth frames f1 1.0 s, f2 1.0, f3 1.0, f4 .5, f5 1.0, f7 .8, f8 1.0, f9 1.0 (gap
#      frame usable; f0 ends at ws, f6 one-sided) = 7.3 s; best levels 10 sh always within 6 ticks of the mid -> 73 sh-s per side; the deeper level
#      (4 sh, best -/+ 4 ticks) is strictly within 6 ticks only where the spread is 2 ticks (f4, f5: 5 ticks): +4 x 1.5 = 6 -> 79; levels >= 5 sh:
#      73. Pool 3000 x 300 / 3600 = 250. Lower bounds: 0x33 bid 700 ms (only in the same-day sets) ----
RMX = {
    (D1, 101): dict(book=True, pool_units=250., n_variants=2, n_frames_used=8, ub_bid_shs=79., ub_ask_shs=79., ub_bid_shs_thr=73., ub_ask_shs_thr=73.,
                    lb_bid_ms_sameday_shares=700., lb_bid_ms_sameday_pnl=700., lb_bid_ms_exante_shares=0., lb_both_ms_sameday_shares=0., depth_status='ok',
                    status='ok', boosted='not_boosted', spread_thr=.06, share_thr=5.),
    (D1, 102): dict(book=False, pool_units=X_, boosted='boosted', depth_status='not_book', ub_bid_shs=X_, status='ok', inv_status='no_rewards'),
    (D1, 103): dict(status='boundary_censored', pool_units=X_),
    (D1, 104): dict(table_status='no_records', status='ok'),
    (D1, 105): dict(status='boundary_censored', table_status='boundary_censored'),
    (D1, 106): dict(status='up_index', inv_status='markets_row_missing'),
    (D2, 201): dict(inv_status='markets_table_missing', pool_units=X_, boosted='unknown'),
}


def _cell(x):
    return None if not isinstance(x, str) and pd.isna(x) else x


def main_world(tmp):
    root = Path(tmp) / 'w'
    write(root, world())
    rec = go(root)
    out = root / 'econ'
    need(sorted(p.name for p in out.iterdir()) == sorted(['econ.json', 'fills_valued.parquet', 'identities.json', 'maker_day.parquet', 'maker_market.parquet',
                                                           'rewards_market.parquet', 'segments.parquet']), 'the output set differs')
    need(not any(p.name.startswith('.econ.partial') for p in root.iterdir()), 'a partial directory was left behind')
    FV = pd.read_parquet(out / 'fills_valued.parquet')
    need(list(FV.columns) == list(E.FV), 'fills_valued columns differ from the declared schema')
    eq(len(FV), 2 * (16 + 23 + 3), 'fills_valued rows (2 per leg: 16 ID legs + 23 D1 legs + 3 D2 legs; none dropped)')
    cz = FV[FV.mid.isin([103, 105])]
    eq(bool(cz.V.isna().all() and (cz.st_res == 'censored').all() and (cz.status == 'boundary_censored').all()), True,
       'censored markets: no outcome surfaces (V NaN), st_res censored, rows kept with status')
    for (sid, k, role), exp in ROWS.items():
        x = FV[(FV.sid == sid) & (FV.mk_idx == k) & (FV.role == role)]
        need(len(x) == 1, f'row {sid}/{k}/{role} missing or duplicated')
        x = x.iloc[0]
        for c, v in exp.items():
            eq(_cell(x[c]), v, f'fills_valued {sid}/{k}/{role}.{c}')
    ido = FV[FV.day == ID]
    eq(bool((ido.st_spr == 'id_day_no_book').all() and (ido.st_end == 'id_day_no_book').all()), True, 'ID day marks are id_day_no_book')
    MMt = pd.read_parquet(out / 'maker_market.parquet')
    need(list(MMt.columns) == list(E.MM), 'maker_market columns differ')
    for (d, m, a), exp in MMX.items():
        x = MMt[(MMt.day == d) & (MMt.mid == m) & (MMt.addr == a)]
        need(len(x) == 1, f'maker_market {d}/{m}/{a} missing or duplicated')
        for c, v in exp.items():
            eq(_cell(x.iloc[0][c]), v, f'maker_market {d}/{m}/{a}.{c}')
    MDt = pd.read_parquet(out / 'maker_day.parquet')
    for (d, a), exp in MDX.items():
        x = MDt[(MDt.day == d) & (MDt.addr == a)]
        need(len(x) == 1, f'maker_day {d}/{a} missing or duplicated')
        for c, v in exp.items():
            eq(x.iloc[0][c], v, f'maker_day {d}/{a}.{c}')
    S = pd.read_parquet(out / 'segments.parquet')
    for (d, st_, g, r, dim, b), exp in SEGX.items():
        x = S[(S.day == d) & (S.id_set == st_) & (S.grp == g) & (S.role == r) & (S.dim == dim) & (S.bucket == b)]
        need(len(x) == 1, f'segment {d}/{st_}/{g}/{r}/{dim}/{b} missing or duplicated')
        for c, v in exp.items():
            eq(x.iloc[0][c], v, f'segment {d}/{st_}/{g}/{r}/{dim}/{b}.{c}')
    eq(int(len(S[(S.day == D1) & (S.id_set == 'sameday_shares') & (S.grp == 'field')])), 0, 'D1 same-day shares set holds every maker: no field rows invented')
    RMt = pd.read_parquet(out / 'rewards_market.parquet')
    need(list(RMt.columns) == list(E.RM), 'rewards_market columns differ')
    eq(len(RMt), 6 + 1, 'rewards_market: one row per market (D1 registry + markets table 101..106; D2 201)')
    for (d, m), exp in RMX.items():
        x = RMt[(RMt.day == d) & (RMt.mid == m)]
        need(len(x) == 1, f'rewards_market {d}/{m} missing or duplicated')
        for c, v in exp.items():
            eq(_cell(x.iloc[0][c]), v, f'rewards_market {d}/{m}.{c}')
    I = json.loads((out / 'identities.json').read_text())
    # shares 100 90 80 60 50 50 42 30 20 10 10 9 9 5 -> the 50s '0x06' < '0xa4'; the cut tie at 10 goes to '0x0b' < '0x10'
    eq(','.join(x['addr'] for x in I['ex_ante']['top10_shares']), '0x01,0x02,0x03,0x05,0x06,0xa4,0x07,0x08,0x09,0x0b', 'ex-ante top 10 by maker shares')
    eq(I['ex_ante']['top10_shares_info']['tie_at_cut'], True, 'shares cut tie detected')
    eq(I['ex_ante']['top10_shares'][0]['mk_share'], 100 / 104., "0x01's maker share of volume (100 maker / 4 taker)")
    # maker-leg P&L (maker-majority): +50 +40 +30 +20.2 (tie .5) +10 +5 +5 -2.5 -15 -25 -25 -45; 0xc1 (9.41 total, taker-made) and 0xt0 excluded
    eq(','.join(x['addr'] for x in I['ex_ante']['top10_pnl']), '0x01,0x03,0x05,0x07,0x09,0x0b,0x10,0x12,0x08,0x06', 'ex-ante top 10 by maker-leg P&L')
    eq(I['ex_ante']['top10_pnl'][3]['value'], 20.2, '0x07 ranks on V = .5 in the tie market')
    eq(I['ex_ante']['top10_pnl'][9]['value'], -25., 'rank-10 P&L')
    eq(I['ex_ante']['top10_pnl_info']['n_candidates'], 12, 'P&L candidates: the 12 maker-majority makers (0xc1 and 0xt0 are not)')
    eq(I['ex_ante']['top10_pnl_info']['tie_at_cut'], True, 'P&L cut tie detected')
    eq(','.join(x['addr'] for x in I['same_day'][D1]['top10_pnl']), '0x33,0x0b,0x01', 'D1 same-day P&L order (descriptive)')
    eq(I['same_day'][D1]['label'], 'descriptive', 'same-day label')
    eq(','.join(x['addr'] for x in I['same_day'][D2]['top10_shares']), '0x01,0x0b,0xs1', 'D2 same-day shares order (tie 4 = 4 by address)')
    eq(','.join(x['addr'] for x in I['same_day'][D2]['top10_pnl']), '0x0b,0x01', 'D2 same-day P&L: the self-trader 0xs1 (4 = 4) is not maker-majority')
    eq(I['n_days'], 2, 'measured days')
    eq(I['intervals'], False, 'no intervals below 5 days (computed)')
    eq(I['ci'], 'none_G2_lt_5', 'ci field')
    eq(I['overlap']['exante_shares&exante_pnl'], 8, 'ex-ante set overlap by hand: 0x01 03 05 06 07 08 09 0b')
    cmp_ = [r for r in I['compare'] if r['day'] == D1 and r['id_set'] == 'exante_shares' and r['grp'] == 'in' and r['role'] == 'maker'][0]
    eq(cmp_['c_per_sh_res'], 100 * -3.20 / 27., 'compare c/sh at resolution')
    eq(cmp_['c_per_sh_end'], 100 * -.46 / 30., 'compare c/sh at END')
    eq(cmp_['c_per_sh_60s'], 100 * -.70 / 10., 'compare c/sh at 60 s (a1 only)')
    st = rec['counts'][D1]['status']
    for k, v in dict(ok=22, side_mismatch=2, no_record=2, bad_leg=2, record_leg_excluded=2, src_status=6, boundary_censored=4, up_index=2, fee_type_unknown=2,
                     yes_space_mismatch=2).items():
        eq(st.get(k), v, f'D1 {k} rows')
    eq(sum(st.values()), 2 * 23, 'D1 status counts cover every row')
    for k, v in dict(lo=2 * 4, hi_fill_status=2, hi_lo_unbounded=2 * 2, hi_missing_life_fill=2, hi_window_inverted=2, hi_no_lifecycle=2 * 14).items():
        eq(rec['counts'][D1]['t_status'].get(k), v, f'D1 t_status {k} rows')
    dg = rec['diagnostics'][D1]
    eq(dg['tk_px_gap_n'], 1, 'planted taker-price gap counted (b1)')
    eq(dg['tk_px_gap_usd'], .01 * 5, 'its USD at stake: |.41 - .40| x 5 sh')
    eq(dg['lf_status_not_ok'], 2, 'flagged lifecycle fills rows counted (a5 time_inversion, a3 fill_lo_unbounded: stays hi_lo_unbounded)')
    eq(','.join(str(m) for m in dg['censored_markets']), '103,105', 'censored market set (record flag + markets status)')
    eq(dg['censored_legs'], 2, 'censored legs')
    eq(dg['registry_markets_without_legs'], 1, 'registry market 104 has no legs')
    eq(dg['lifecycle']['fills_status'].get('time_inversion'), 1, 'lifecycle fills statuses counted')
    eq(dg['lifecycle']['orders_status'].get('ok'), 15, 'lifecycle orders statuses counted (5 hashes x 3 scenarios)')
    # spread clock (BOOK records ok a1..a6): evidence a1 (f_after 2; settle -> f2), a2 (min 5; settle -> f5), a6 (8; settle -> f7): 2 of 3 show the
    #    trade; a3 a4 (unbounded) a5 (flagged) have no evidence; same maker fills: lo .20 + .06 + .08 + 0 + .06 + .40 = .80, hi .10 + ... = .70, 36 sh
    sc = dg['spread_clock']
    for k, v in dict(clock_book_records=6, clock_with_evidence=3, clock_hi_shows_trade=2, clock_no_evidence=3, clock_evidence_frame_missing=0,
                     share_hi_shows_trade=2 / 3., rows_both_ok=6, sh=36., v_spr_lo=.80, v_spr_hi=.70).items():
        eq(sc[k], v, f'spread clock {k}')
    eq(rec['diagnostics'][D2]['legs_recv_before_settle'], 1, 'c2 received before its settle_ms is counted (clock-skew evidence)')
    ff = dg['fee_formula']
    eq(ff['n_fee_records'], 4, 'D1 fee records')
    eq(ff['collateral']['n_match_usd'], 1, 'D1 collateral formula matches')
    eq(ff['shares']['n_match_usd_at_fill_price'], 1, 'D1 shares fee = formula at the fill price')
    eq(ff['shares']['n_match_amount_as_formula'], 0, 'D1 shares amount = formula')
    eq(rec['diagnostics'][D2]['fee_formula']['shares']['n_match_amount_as_formula'], 1, 'D2 c1: .06 NO shares = .02 x .30 x 10')
    eq(ff['implied_rate_top']['coll'].get('0.0200'), 1, 'implied COLLATERAL rate .0888 / 4.44 = .02')
    eq(ff['implied_rate_top']['sh_usd_at_fill'].get('0.0200'), 1, 'implied SHARES rate at the fill price (a3): .04 / 2.00 = .02')
    inv = {r['mid']: r for r in rec['rewards']['inventory'][D1]}
    eq(inv[101]['pool_units'], 3000 * 300 / 3600., 'reward pool = hourlyRate x 5/60 h (rewards_variants schedule, entries counted once)')
    eq(inv[101]['pool_source'], 'rewards_variants', 'pool from the variants')
    eq(inv[101]['status'], 'ok', 'reward inventory status')
    eq(inv[101]['registry_match']['share_thr'], False, 'registry shareThreshold 100 != markets 5 is flagged')
    eq(inv[102]['status'], 'no_rewards', 'a market without a rewards object')
    eq(rec['rewards']['inventory'][D2], 'markets_table_missing', 'missing markets table is a status')
    eq(inv[101]['reward_variants']['n'], 2, 'rewards_variants unwrapped ([{first_recv_ns, rewards: {...}}])')
    eq(inv[101]['reward_variants']['n_current_non_null'], 1, 'rewards.current non-null in one variant')
    eq(inv[101]['reward_variants']['n_distinct_schedules'], 1, 'one schedule across variants')
    eq(inv[104]['table_status'], 'no_records', 'a population market without records is inventoried')
    eq(inv[105]['table_status'], 'boundary_censored', 'markets status carried')
    need('unidentifiable' in rec['rewards']['per_maker'], 'per-maker rewards stated unidentifiable')
    for k, v in rec['checks'].items():
        need(v['ok'] and v['max_dev'] <= 1e-9, f'check {k} deviation {v["max_dev"]}')
    N_OK[0] += 1
    eq(','.join(sorted(rec['memory']['worker_hwm_mib'])), ','.join((ID, D1, D2)), 'one worker process per day')
    eq(bool((out / 'fills_valued.parquet').is_file() and not (out / 'parts').exists()), True, 'parts merged and removed')
    return root


def direct_cases():
    # mid_at statuses: crossed / frame_status (bad_frame) / usable 'gap+new_conn' frame / stale (age 1001) / ok at age exactly 1000 / unknown market
    T = pd.DataFrame(dict(mid=[7, 7, 7], upd_ms=[1000, 2000, 3000], recv_ns=[1, 2, 3], n_bid=[1, 1, 1], n_ask=[1, 1, 1], best_bid=[.5, .4, .4],
                          best_ask=[.5, .6, .6], status=['ok', 'bad_frame', 'gap+new_conn']))
    m, age, st = E.mid_at(E.frames(T), np.array([7, 7, 7, 7, 7, 7, 8]), np.array([500, 1500, 2500, 3500, 4001, 4000, 3000]))
    eq(','.join(E.MARK_ST[s] for s in st), 'no_frame,crossed,frame_status,ok,stale,ok,no_frame', 'mid_at statuses')
    eq(m[3], .5, 'mid of the usable gap frame')
    eq(int(age[5]), 1000, 'age exactly 1000 ms is not stale')
    # fee_pos: COLLATERAL 1 -> dc -1; SHARES 1 YES -> dy -1; SHARES 1 NO -> dy +1, dc -1 (fee at x = -(dc + dy x): .5 -> 1, .5, .5)
    dy, dc = E.fee_pos(np.array([1., 1., 1.]), np.array(['COLLATERAL', 'SHARES', 'SHARES'], dtype=object), np.array([1, 1, 2]))
    eq(','.join(f'{a:g}/{b:g}' for a, b in zip(dy, dc)), '0/-1,-1/0,1/-1', 'fee positions')
    eq(-(dc[2] + dy[2] * 0.), 1., 'NO-share fee at V = 0 costs 1 per share')
    # intervals
    u = E.union([(5, 9), (0, 3), (2, 4), (9, 10)])
    eq(str(u), '[[0, 4], [5, 10]]', 'interval union')
    eq(E.measure(u), 9, 'union measure')
    eq(E.overlap(u, E.union([(3, 6), (8, 20)])), 1 + 1 + 2, 'intersection measure [3,4]+[5,6]+[8,10]')
    # top_k 'pnl': a taker-majority address with the best maker P&L and one without resolved maker shares are not candidates; ties by address
    D = pd.DataFrame(dict(addr=['b', 'a', 'c', 't', 'u'], sh_mk=[5., 5., 1., 2., 3.], sh_tk=[0., 0., 0., 9., 1.], sh_res_mk=[5., 5., 1., 2., 0.],
                          net_res_mk=[1., 1., 0., 99., 50.], net_res=[1., 1., 0., 99., 50.], n_res_na=[0, 0, 0, 0, 3]))
    lst, info = E.top_k(D, 'pnl', k=1)
    eq(lst[0]['addr'], 'a', 'tie broken by address ascending')
    eq(info['tie_at_cut'], True, 'tie at the cut flagged')
    eq(info['n_candidates'], 3, 'the taker-majority and the unresolved address are not candidates')
    # life_join H1 mismatch: an orders row whose signer differs -> life_mismatch, buckets life_not_ok
    W = world()
    F = pd.DataFrame(dict(mid=[101], role=pd.Categorical(['maker']), hash=pd.Categorical(['hM3']), addr=pd.Categorical(['0x33']), y_side=pd.Categorical(['bid']),
                          _ptick=[55]))
    O = W['orders'].copy()
    O.loc[O.mk_hash == 'hM3', 'mk_signer'] = '0xzz'
    Lc = pd.DataFrame(dict(mid=[101, 101], mk_hash=['hM3', 'hM3']))
    E.life_join(F, Lc, O, [101])
    eq(str(F.life_status[0]), 'life_mismatch', 'H1 mismatch status')
    eq(str(F.life_rep[0]), 'life_not_ok', 'mismatch bucket')
    O = W['orders'].copy()
    F2 = F.copy()
    E.life_join(F2, Lc.iloc[:1], O, [101])
    eq(str(F2.life_status[0]), 'nfills_mismatch', 'n_fills 2 vs 1 leg in the chunk')
    # REWARDS lower bound by hand (thr 5 sh / 0.06; window [0, 10000)). Frames of market 5: 1000 mid .61, 1500 .57, 3000 one-sided, 3100 .55.
    #  bid 'hb' p .55, presence [1000, 2600), fills lo 2000 (6) / 900 (4): proven remaining >= 5 only before 2000 -> span [1000, 2000);
    #    1000: |.55-.61| = .06 not < .06 (strict); 1500: in band until min(2000, 1500 + 1000) -> [1500, 2000) = 500 ms
    #  ask 'ha' p .60, presence [1200, 3500), fill lo 3500 (12): 1000 frame |.01| -> [1200, 1500); 1500 |.03| -> [1500, 2500); 3000 one-sided;
    #    3100 |.05| -> [3100, 3500) => union 1300 + 400 = 1700; both = [1500, 2000) = 500
    #  'hc' size_lb 3 < 5 and 'hd' assumption_sensitive presence are not counted
    Tm = pd.DataFrame(dict(mid=[5] * 4, upd_ms=[1000, 1500, 3000, 3100], recv_ns=[1, 2, 3, 4], n_bid=[1, 1, 1, 1], n_ask=[1, 1, 0, 1],
                           best_bid=[.58, .56, .50, .54], best_ask=[.64, .58, NAN, .56], status=['ok'] * 4))
    O = pd.DataFrame(dict(scenario=['S_NOPRIO'] * 4, mk_hash=['hb', 'ha', 'hc', 'hd'], mk_signer=['mm', 'mm', 'mm', 'nn'], y_side=['bid', 'ask', 'bid', 'bid'],
                          size_lb=[10., 10., 3., 10.], persist_lo_ms=[1000, 1200, 1000, 1000], persist_hi_ms=[2600, 3500, 2600, 2000],
                          persist_class=['observed', 'constrained', 'observed', 'assumption_sensitive'], status=['ok'] * 4))
    #  (a flagged fills row of 'hb' at lo 2400 must not move its cut: it would give [1500, 2400) = 900 ms)
    lfm = pd.DataFrame(dict(mk_hash=['hb', 'hb', 'hb', 'ha', 'hc', 'hd'], lo_ms=[2000, 900, 2400, 3500, 2600, 2000], shares=[6., 4., 6., 12., 3., 10.],
                            status=['ok', 'ok', 'time_inversion', 'ok', 'ok', 'ok']))
    pr = E.presence(O, lfm, E.frames(Tm), 5, dict(hb=.55, ha=.60, hc=.55, hd=.55), 5., .06, 0, 10000)
    eq(','.join(sorted(pr)), 'mm', 'only qualifying orders count')
    ub, ua = E.union(pr['mm']['bid']), E.union(pr['mm']['ask'])
    eq(E.measure(ub), 500, 'bid eligible ms (remaining-size cut, strict spread boundary)')
    eq(E.measure(ua), 1700, 'ask eligible ms')
    eq(E.overlap(ub, ua), 500, 'both-sides eligible ms')
    eq(pr['mm']['n'], 2, 'orders with eligible time')
    return W


def refusals(tmp, root_ok):
    MMt = pd.read_parquet(Path(root_ok) / 'econ' / 'maker_market.parquet')
    A = MMt[MMt.mid == 101].copy()
    E.check_market(A)
    B = A.copy()
    B.loc[B.addr == '0x01', 'rt_pnl'] += .01
    refused(lambda: E.check_market(B), 'gross_eq_rt_plus_resid', 'planted FIFO conservation error')
    B = A.copy()
    B.loc[B.addr == '0x01', 'resid_drift'] += .01
    refused(lambda: E.check_market(B), 'resid_eq_drift_plus_resol', 'planted residual split error')
    refused(lambda: E.check_market(A[A.addr != '0xz1']), 'zero_sum_gross', 'a dropped address breaks the zero sum')
    B = A.copy()
    B.loc[B.addr == '0x01', 'v_end_mk'] += .01
    refused(lambda: E.check_market(B), 'zero_sum_v_end', 'a mark value without its counterparty')

    def variant(name, mut, post=None, **kw):
        root = Path(tmp) / name
        W = world()
        mut(W, root)
        write(root, W)
        if post is not None:
            post(root)
        return lambda: go(root, **kw)

    refused(lambda: go(Path(tmp) / 'none', days=(ID, 'notaday')), 'is not a consumed day', 'a day outside the consumed days')
    refused(lambda: go(root_ok), 'write-once', 'a second run over an existing output')
    variant('ctrl', lambda W, r: None)()          # control: the unmodified world runs through every writer in a fresh directory
    need((Path(tmp) / 'ctrl' / 'econ' / 'econ.json').is_file(), 'the control run wrote no receipt')
    N_OK[0] += 1

    def no_col(W, r):
        W['legs'] = W['legs'].drop(columns=['y_price'])
    refused(variant('nocol', no_col), 'required column y_price missing', 'legs without y_price')

    def ytype(W, r):
        W['legs'] = W['legs'].assign(y_price=W['legs'].y_price.astype(str))
    refused(variant('ytype', ytype), 'column y_price is', 'legs y_price with the wrong dtype')

    def dup(W, r):
        W['legs'] = pd.concat([W['legs'], W['legs'][W['legs'].sid == 'a1']], ignore_index=True)
    refused(variant('dup', dup), 'duplicate leg key', 'duplicate (sid, mk_idx)')

    def dupr(W, r):
        W['records'] = pd.concat([W['records'], W['records'][W['records'].sid == 'a1']], ignore_index=True)
    refused(variant('dupr', dupr), 'is not unique', 'duplicate settlementId in records')

    refused(variant('nolife', lambda W, r: None, book={D1: [101, 102]}), 'input missing', 'a BOOK market without lifecycle files')

    def case(W, r):
        W['legs'].loc[W['legs'].sid == 'a5', 'mk_signer'] = '0X33'
    refused(variant('case', case), 'address case collision', 'two spellings of one address')

    def dayc_post(root):
        p_ = root / f'legs_{D1}.parquet'
        x = pd.read_parquet(p_)
        x.loc[x.sid == 'a5', 'day'] = D2
        x.to_parquet(p_, index=False)
    refused(variant('dayc', lambda W, r: None, post=dayc_post), 'day column differs', 'a legs row filed under another day')
    refused(variant('mem', lambda W, r: None, guard_mib=1.), 'memory self-guard', 'the memory guard fires')
    for name in ('nocol', 'ytype', 'dup', 'dupr', 'nolife', 'case', 'dayc', 'mem'):
        need(not (Path(tmp) / name / 'econ').exists() and not any(p.name.startswith('.econ.partial') for p in (Path(tmp) / name).iterdir()),
             f'{name}: a refused run left an output')
    N_OK[0] += 1


def main():
    with tempfile.TemporaryDirectory(prefix='mt_econ_selftest_') as tmp:
        root = main_world(tmp)
        direct_cases()
        refusals(tmp, root)
    print(json.dumps(dict(selftest='pf_mt_econ', assertions_passed=N_OK[0], result='PASS')))


if __name__ == '__main__':
    main()
