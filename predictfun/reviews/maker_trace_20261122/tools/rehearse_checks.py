"""Synthetic rehearsal of mt_checks.py (MAKER-TRACE-RECONSTRUCTION-v1 contract + amendment 1; P-2026-005; session aea5bfb9, 2026-11-22).

Writes tiny synthetic raw hour files (REST matches + predict_ws books in the real row/payload layout), step1 registries and CORRECT
builder-format outputs (contract section 4 paths, pf_mt_common.SCHEMAS dtypes, mt_checks.IFACE econ / truth columns, '+'-joined status
flags counted by flag) into a temp dir. Everything derives from ONE generative spec (orders with true sizes / posting / removal + matches):
the raw by serialisation, the outputs directly from the spec - so the checker's raw parse is an independent path. Then run_checks (every
check must pass), reparse of the BOOK-like and GT-like markets (every check must pass) and falsify (every planted defect refused with its
reason, every check name targeted). Exit 0 only if all hold. Consumed-day stems only (Oct 23 / 24 / 27, Nov 8 hours 10 and 22).
Run: PF_SLOT=claude2 predictfun/research/run_light.sh mt_checks_rehearse -- python3 <abs path of this file> [--keep]"""
import copy, datetime as dt, gzip, hashlib, json, shutil, sys, tempfile, uuid
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mt_checks as M  # noqa: E402
C = M.C
E18, UTC = C.E18, dt.timezone.utc
INFO = 'price_mismatch'          # the extract's price-mismatch flag (amendment 1 item 6; pinned: no flag is benign in economics)
WIDE = 'cancelled_or_expired|end_censored|exhausted'       # pinned: '|'-joined SORTED removal tokens


def ms_of(*a):
    return int(dt.datetime(*a, tzinfo=UTC).timestamp() * 1000)


def hx(tag, n=64):
    return hashlib.sha256(tag.encode()).hexdigest()[:n]


def addr(tag):      # mixed-case (checksum-like) address: the checker must compare case-insensitively
    return '0x' + ''.join(c.upper() if i % 3 == 0 else c for i, c in enumerate(hx('addr' + tag, 40)))


def uuid7(ms, tag):
    r = int(hx('sid' + tag), 16)
    return str(uuid.UUID(int=(ms << 80) | (7 << 76) | ((r & 0xfff) << 64) | (2 << 62) | ((r >> 12) & ((1 << 62) - 1))))


def iso(ms):
    return dt.datetime.fromtimestamp(ms / 1000, UTC).strftime('%Y-%m-%dT%H:%M:%S.000Z')


def fcount(statuses):          # per-flag counts (amendment 1 item 2), written by the fixture's own code
    c = Counter()
    for s in statuses:
        c.update(s.split('+'))
    return dict(c)


e18p = lambda p: int(round(p * 1000)) * 10 ** 15
e18q = lambda q: int(round(q * 10 ** 6)) * 10 ** 12
A_, B_, C_, X_, Y_ = (addr(t) for t in 'ABCXY')
SC = list(C.SCENARIOS)
FIFO = [s for s in SC if C.SCENARIOS[s]['priority'] == 'price_time']


def outc(i, status='OPEN', bb=None):
    return dict(name='Up' if i == 1 else 'Down', indexSet=i, onChainId='111' if i == 1 else '222', status=status, bestBid=bb, bestAsk=None, team=None, variantData=None)


def market_obj(mid, ws_s, state='OPEN'):
    return dict(id=mid, title='synthetic', tradingStatus=state, status='REGISTERED', categorySlug=f'btc-updown-5m-{ws_s}', decimalPrecision=2,
                spreadThreshold=0.06, shareThreshold=100, rewards=dict(current=None, schedule=[]))


# ------------------------------------------------------------------------------------------------------------- generative spec
# tag -> (signer, quote, indexSet, own price, true size, posted at ws+ms, removed (cancel) at ws+ms or None)
ORD = dict(hA=(A_, 'Ask', 1, .50, 6.5, 2500, None), hB=(B_, 'Ask', 1, .51, 6., -20000, 45000), hC=(C_, 'Bid', 2, .49, 4., 35000, None),
           hD=(A_, 'Bid', 1, .48, 5., -30000, None), hE=(B_, 'Ask', 2, .56, 2., 54500, None), hF=(C_, 'Bid', 1, .47, 1., 65000, None),
           hF2=(C_, 'Bid', 1, .46, 1., 75000, None), hG=(C_, 'Ask', 1, .53, 2., 85000, None), hH=(C_, 'Ask', 1, .54, 1., 95000, None),
           hR=(B_, 'Ask', 1, .62, 1., 101000, None), hS=(C_, 'Bid', 2, .37, 1., 101500, None), hM=(B_, 'Ask', 1, .61, 2., 104000, None),
           hN=(C_, 'Ask', 1, .61, 2., 104500, None), hI=(B_, 'Ask', 1, .57, 1., 105000, None), hJ=(C_, 'Ask', 1, .57, 3., 105500, None),
           hK=(B_, 'Ask', 1, .58, 1., 113000, None), hL=(C_, 'Ask', 1, .58, 3., 113500, None), hP=(B_, 'Ask', 1, .59, 2., 119000, None),
           hQ=(C_, 'Ask', 1, .60, 1., 119500, None))
MATCH = [('R1', 10300, X_, 'Bid', 1, [('hA', 3.)], 'SHARES'), ('R2', 20300, Y_, 'Bid', 1, [('hA', 2.5)], 'SHARES'),
         ('R3', 30300, X_, 'Bid', 1, [('hA', 1.), ('hB', 5.)], 'COLLATERAL'),        # hA strictly better -> exhausted by price priority
         ('R4', 40300, Y_, 'Bid', 1, [('hC', 4.)], 'SHARES'), ('R5', 50300, X_, 'Ask', 1, [('hD', 3.)], 'COLLATERAL'),
         ('R6', 60300, Y_, 'Ask', 1, [('hE', 2.)], 'COLLATERAL'), ('R7', 70300, X_, 'Ask', 1, [('hF', 1.)], 'COLLATERAL'),
         ('R8', 80300, Y_, 'Ask', 1, [('hF2', 1.)], 'COLLATERAL'), ('R9', 90300, X_, 'Bid', 1, [('hG', 2.)], 'SHARES'),
         ('R10', 100300, Y_, 'Bid', 1, [('hH', 1.)], 'SHARES'),
         ('R18', 102300, X_, 'Bid', 1, [('hS', 1.), ('hR', 1.)], 'SHARES'),        # mint + regular, worst listed first; priceExecuted = limit
         ('R16', 107300, X_, 'Bid', 1, [('hM', 1.), ('hN', 1.)], 'SHARES'),        # two worst-price makers that BOTH fill again -> T2
         ('R11', 110300, X_, 'Bid', 1, [('hI', 1.), ('hJ', 1.)], 'SHARES'),        # same price, neither fills again: count constraint only
         ('R17', 112300, Y_, 'Bid', 1, [('hN', 1.), ('hM', 1.)], 'SHARES'),
         ('R12', 115300, Y_, 'Bid', 1, [('hK', 1.), ('hL', 1.)], 'SHARES'),        # exactly one worst-price maker (hL) fills again -> hK exhausted
         ('R13', 118300, X_, 'Bid', 1, [('hL', 1.)], 'SHARES'),
         ('R14', 121300, Y_, 'Bid', 1, [('hQ', 1.), ('hP', 1.)], 'SHARES'),        # hP strictly better ... and fills again (R15) -> T1
         ('R15', 124300, X_, 'Bid', 1, [('hP', 1.)], 'SHARES')]
BG = {('ask', 50): 4., ('ask', 51): 10., ('ask', 52): 7., ('ask', 54): 3., ('bid', 48): 5., ('bid', 47): 2., ('bid', 46): 2., ('bid', 45): 8.}
FRAME_K = range(-10, 126)          # frames at ws + k s (conn c1); k = 0..4 duplicated on conn c2
UNBOUNDED = {'R5'}                 # no book evidence declared for this fill -> lo = -1, flag fill_lo_unbounded
PX_OVERRIDE = dict(R18=0.63)       # priceExecuted != maker-weighted average (0.625)
REC_STATUS = dict(R9='conflict', R10='amount_mismatch')          # declared: R18's price gap is an econ diagnostic (px_gap), never a status
FILL_STATUS = dict(R5='fill_lo_unbounded', R9='leg_conflict', R10='leg_amount_mismatch')     # pinned: extract flags carried as leg_*
STATUS_ORDERS = dict(records=['conflict', 'amount_mismatch', 'parse_error', 'boundary_censored'],      # pinned: extract_<day>.json status_orders
                     legs=['conflict', 'amount_mismatch', 'parse_error', 'boundary_censored'],
                     tob=['new_conn', 'gap', 'dup', 'same_ms_changed', 'bad_frame', 'mid_mismatch', 'out_of_order', 'off_grid', 'boundary_censored'],
                     frames=['new_conn', 'gap', 'dup', 'same_ms_changed', 'bad_frame', 'mid_mismatch', 'out_of_order', 'off_grid'],
                     levels=['bad_frame', 'off_grid'], markets=['boundary_censored'])
ORDER_STATUS = dict(hF='identity_conflict', hG='leg_conflict', hH='leg_amount_mismatch')
RECONCILING = {t for t, *_ in MATCH} - {'R10'}       # R9's kept variant reconciles exactly


def build_raw_record(mid, ws_s, W, tagp, tag, rel, taker, tq, ti, makers, ftype, rh, amt_override=None, maker_amt=None, px=None, ords=None):
    tm = W + rel; settle = tm + 20
    legs, num, tot = [], 0, 0
    for ot, sh in makers:
        sg, q, i, p = (ords or ORD)[ot][:4]
        a = e18q(maker_amt if maker_amt is not None else sh); pe = e18p(p)
        num += a * (pe if i == 1 else E18 - pe); tot += a
        legs.append(dict(quoteType=q, amount=str(a), price=str(pe), outcome=outc(i), signer=sg, hash=rh[ot], fee=dict(amount='0', type='SHARES')))
    yt = num // tot if px is None else e18p(px)
    pt = yt if ti == 1 else E18 - yt
    amt = e18q(amt_override) if amt_override is not None else tot
    fee = 2 * min(yt, E18 - yt) * amt // (100 * (pt if ftype == 'SHARES' else E18))
    tk = dict(quoteType=tq, amount=str(amt), price=str(pt), outcome=outc(ti), signer=taker, hash='0x' + hx('tk' + tagp + tag), fee=dict(amount=str(fee), type=ftype))
    return dict(market=market_obj(mid, ws_s), taker=tk, amountFilled=str(amt), priceExecuted=str(pt), makers=legs,
                transactionHash='0x' + hx('tx' + tagp + tag), settlementId=uuid7(settle, tagp + tag), executedAt=iso((tm // 1000 + 1) * 1000)), settle


def rest_line(recv_ns, mid, data, status=200, path='/v1/orders/matches'):
    raw = json.dumps(dict(success=status == 200, cursor='c', data=data), separators=(',', ':'))
    return json.dumps(dict(run_id='synthetic', seq=0, source='rest', kind='response', recv_ns=recv_ns, recv_mono_ns=0, path=path,
                           params=dict(marketId=str(mid), first='100'), sent=dict(recv_ns=recv_ns - 10 ** 8), status=status, raw=raw), separators=(',', ':'))


def ws_line(recv_ns, conn, mid, upd, bids, asks, pb, pa, oc):
    data = dict(asks=[[p, q] for p, q in asks], bids=[[p, q] for p, q in bids], lastOrderSettled=dict(id='1', kind='LIMIT', marketId=mid, outcome='Yes', price='0.50', side='Bid'),
                marketId=mid, orderCount=oc, settlementsPending=dict(asks=pa, bids=pb), updateTimestampMs=upd, version=1)
    raw = json.dumps(dict(type='M', topic=f'predictOrderbook/{mid}', data=data), separators=(',', ':'))
    return json.dumps(dict(run_id='synthetic', seq=0, source='predict_ws', kind='frame', recv_ns=recv_ns, recv_mono_ns=0, connection_id=conn, raw=raw), separators=(',', ':'))


def market(day, mid, W, tagp):
    """One BOOK/GT-like market: raw lines + every builder table, all from the spec."""
    ws_s = W // 1000
    rh = {t: '0x' + hx('order' + tagp + t) for t in ORD}; rh['hF2'] = rh['hF']          # H1: one hash at two prices
    kept, rest, first = {}, [], {}
    for tag, rel, tk, tq, ti, mk, ft in MATCH:
        if tag == 'R9':      # first receipt: a variant whose makers[] does not reconcile (1 of 2 shares); the later variant does -> kept
            v1, settle = build_raw_record(mid, ws_s, W, tagp, tag, rel, tk, tq, ti, mk, ft, rh, amt_override=2., maker_amt=1.)
            rest.append(rest_line((settle + 400) * 10 ** 6 + 123456, mid, [v1]))
        rec, settle = build_raw_record(mid, ws_s, W, tagp, tag, rel, tk, tq, ti, mk, ft, rh, amt_override=2. if tag == 'R10' else None, px=PX_OVERRIDE.get(tag))
        kept[tag] = (rec, settle, rel)
        first[tag] = (settle + (-50 if tag == 'R7' else 400)) * 10 ** 6 + 123456     # R7: received before its id time -> a receipt-hi fill
        if tag != 'R9':
            rest.append(rest_line(first[tag], mid, [rec]))
    late = []                       # later page: re-polls with volatile fields changed + R9's reconciling variant
    for tag in ('R2', 'R1'):
        v = copy.deepcopy(kept[tag][0]); v['market'] = market_obj(mid, ws_s, 'CLOSED')
        for leg in [v['taker']] + v['makers']:
            leg['outcome'] = outc(leg['outcome']['indexSet'], 'WON' if leg['outcome']['indexSet'] == 1 else 'LOST', '0.99')
        late.append(v)
    late.append(kept['R9'][0])
    rest.append(rest_line((W + 150000) * 10 ** 6 + 5, mid, late))
    rest.append(rest_line((W + 151000) * 10 ** 6, mid, [], status=503))
    rest.append(rest_line((W + 152000) * 10 ** 6, mid, [kept['R1'][0]], path='/v1/markets'))
    # ---- the book: background + spec orders, remaining = size - fills strictly before the frame; exhausted / cancelled orders vanish
    fills_of = defaultdict(list)
    for tag, rel, *_x, mk, _ft in MATCH:
        for ot, sh in mk:
            fills_of[ot].append((rel, sh))
    ys = {t: C.yes_space(o[1], o[2], o[3]) for t, o in ORD.items()}
    ypt = {t: (s, int(round(p * 100))) for t, (s, p) in ys.items()}

    def book(rel):
        lv, n = Counter(BG), len(BG)
        for t, o in ORD.items():
            if o[5] <= rel and (o[6] is None or rel < o[6]):
                rem = o[4] - sum(sh for r, sh in fills_of[t] if r < rel)
                if rem > 1e-12:
                    lv[ypt[t]] += rem; n += 1
        return lv, n
    pend = {}
    for tag, rel, tk, tq, ti, mk, ft in MATCH:
        yp = ys[mk[0][0]][1]
        pend[rel // 1000 + 1] = (C.yes_space(tq, ti, 0.5)[0], [[yp, round(sum(sh for _, sh in mk) * yp, 6)]])
    frames = []
    for k in FRAME_K:
        for conn, dl in (('c1', 0),) + ((('c2', 9),) if 0 <= k <= 4 else ()):
            lv, n = book(k * 1000)
            ps = pend.get(k)
            frames.append(dict(k=k, upd=W + k * 1000, recv=(W + k * 1000 + 53 + dl) * 10 ** 6 + 7, conn=conn, lv=lv, n=n,
                               pb=ps[1] if ps and ps[0] == 'bid' else [], pa=ps[1] if ps and ps[0] == 'ask' else []))
    frames.sort(key=lambda f: (f['upd'], f['recv']))
    for i, f in enumerate(frames):
        f['fid'] = i
    wsl = [ws_line(f['recv'], f['conn'], mid, f['upd'], sorted([(p / 100, q) for (s, p), q in f['lv'].items() if s == 'bid'], reverse=True),
                   sorted([(p / 100, q) for (s, p), q in f['lv'].items() if s == 'ask']), f['pb'], f['pa'], f['n']) for f in frames]
    # ---- records / legs (kept variant; flags joined with '+')
    R, L = [], []
    for tag, (rec, settle, rel) in kept.items():
        tk, st = rec['taker'], REC_STATUS.get(tag, 'ok')
        tys = C.yes_space(tk['quoteType'], tk['outcome']['indexSet'], int(tk['price']) / E18)[0]
        R.append(dict(day=day, mid=mid, sid=rec['settlementId'], tx=rec['transactionHash'], settle_ms=settle, exec_ms=M.iso_ms(rec['executedAt']),
                      first_recv_ns=first[tag], n_recv=2 if tag in ('R1', 'R2', 'R9') else 1, amount_filled=int(rec['amountFilled']) / E18,
                      price_executed=int(rec['priceExecuted']) / E18, n_makers=len(rec['makers']), tk_hash=tk['hash'], tk_signer=tk['signer'],
                      tk_quote=tk['quoteType'], tk_index_set=tk['outcome']['indexSet'], tk_price=int(tk['price']) / E18, tk_amount=int(tk['amount']) / E18,
                      tk_fee=int(tk['fee']['amount']) / E18, tk_fee_type=tk['fee']['type'], tk_y_side=tys, status=st, x_n_variants=2 if tag == 'R9' else 1))
        for i, m in enumerate(rec['makers']):
            ysd, yp = C.yes_space(m['quoteType'], m['outcome']['indexSet'], int(m['price']) / E18)
            trel = (settle - W) / 1000.
            L.append(dict(day=day, mid=mid, sid=rec['settlementId'], tx=rec['transactionHash'], settle_ms=settle, exec_ms=M.iso_ms(rec['executedAt']),
                          first_recv_ns=first[tag], mk_idx=i, n_makers=len(rec['makers']), mk_hash=m['hash'], mk_signer=m['signer'], mk_quote=m['quoteType'],
                          mk_index_set=m['outcome']['indexSet'], mk_price=int(m['price']) / E18, shares=int(m['amount']) / E18, mk_fee=0., mk_fee_type='SHARES',
                          y_side=ysd, y_price=yp, y_ptick=int(round(yp * 100)), tk_hash=tk['hash'], tk_signer=tk['signer'], tk_y_side=tys,
                          phase='pre' if trel < 0 else ('live' if trel < 300 else 'post'), t_rel_s=trel, status=st, x_tag=tag,
                          x_otag=MATCH[[t for t, *_ in MATCH].index(tag)][5][i][0]))
    FRt = pd.DataFrame([dict(mid=mid, fid=f['fid'], upd_ms=f['upd'], recv_ns=f['recv'], conn=f['conn'], order_count=f['n'], pend_bids=json.dumps(f['pb']),
                             pend_asks=json.dumps(f['pa']), los_id='1', los_side='Bid', los_price=0.5, los_outcome='Yes', status='ok') for f in frames])
    LVt = pd.DataFrame([dict(mid=mid, fid=f['fid'], side=s, ptick=p, shares=q, status='ok') for f in frames for (s, p), q in sorted(f['lv'].items())])
    TBt = pd.DataFrame([dict(day=day, mid=mid, upd_ms=f['upd'], recv_ns=f['recv'], conn=f['conn'], order_count=f['n'],
                             n_bid=sum(1 for s, _ in f['lv'] if s == 'bid'), n_ask=sum(1 for s, _ in f['lv'] if s == 'ask'),
                             best_bid=max(p for s, p in f['lv'] if s == 'bid') / 100, best_bid_sh=f['lv'][('bid', max(p for s, p in f['lv'] if s == 'bid'))],
                             best_ask=min(p for s, p in f['lv'] if s == 'ask') / 100, best_ask_sh=f['lv'][('ask', min(p for s, p in f['lv'] if s == 'ask'))],
                             pend_bid_usd=sum(v for _, v in f['pb']), pend_ask_usd=sum(v for _, v in f['pa']), status='ok') for f in frames])
    c1 = {f['k']: f['fid'] for f in frames if f['conn'] == 'c1'}
    Fr = []
    for l in L:       # lo = venue time of the frame before the first frame after the match (the evidence frame); hi = min(settle, first receipt)
        kb, unb = kept[l['x_tag']][2] // 1000, l['x_tag'] in UNBOUNDED
        hi = min(l['settle_ms'], l['first_recv_ns'] // 10 ** 6)
        Fr.append(dict(mid=mid, sid=l['sid'], mk_hash=l['mk_hash'], mk_idx=l['mk_idx'], y_side=l['y_side'], y_ptick=l['y_ptick'], shares=l['shares'],
                       hi_ms=hi, lo_ms=-1 if unb else W + kb * 1000, t_method='none' if unb else 'drop',
                       f_before=-1 if unb else c1[kb], f_after=-1 if unb else c1[kb + 1], status=FILL_STATUS.get(l['x_tag'], 'ok'), x_otag=l['x_otag'],
                       x_tag=l['x_tag'], x_lo_hard_ms=-1 if l['first_recv_ns'] // 10 ** 6 < l['settle_ms'] else hi - C.MAX_LAG_MS,   # pinned hard lower end
                       x_evidence_fid=-1 if unb else c1[kb + 1]))                                                                  # pinned evidence frame
    Ft = pd.DataFrame(Fr)
    Ot, tests = lifecycle(mid, W, Ft, L, frames)
    truth = dict(hA=(W + 2500, 6.5, 'FILLED', 6.5), hD=(W - 30000, 5., 'EXPIRED', 3.))
    return dict(rest=rest, ws=wsl, R=R, L=L, FR=FRt, LV=LVt, TB=TBt, F=Ft, O=Ot, rh=rh, W=W, truth=truth, tests=tests)


def prio(L, use):
    """The fixture's own price-priority rules (amendment 1), over the legs of records in `use`."""
    last = defaultdict(int)
    for l in L:
        last[l['mk_hash']] = max(last[l['mk_hash']], l['settle_ms'])
    ev, ahead, t1, t1h, t2 = set(), {}, 0, set(), 0
    by = defaultdict(list)
    for l in L:
        if l['x_tag'] in use:
            by[l['sid']].append(l)
    for sid, g in by.items():
        buy = g[0]['tk_y_side'] == 'bid'
        worst = (max if buy else min)(x['y_ptick'] for x in g)
        for x in g:
            if (x['y_ptick'] < worst) if buy else (x['y_ptick'] > worst):
                ev.add((sid, x['mk_idx']))
                if last[x['mk_hash']] > x['settle_ms']:
                    t1 += 1; t1h.add(x['mk_hash'])
        W_ = [x for x in g if x['y_ptick'] == worst]
        if len(W_) >= 2:
            again = [x for x in W_ if last[x['mk_hash']] > x['settle_ms']]
            if len(again) == 1:
                ev.update((sid, x['mk_idx']) for x in W_ if x is not again[0])
                ahead[(sid, again[0]['mk_idx'])] = float(sum(x['shares'] for x in W_ if x is not again[0]))
            elif len(again) >= 2:
                t2 += 1
    return ev, ahead, dict(t1_legs=t1, t1_hashes=len(t1h), t2=t2)


def lifecycle(mid, W, F, L, frames):
    """Reference lifecycle of the spec under the contract + amendment 1 (written for this fixture only)."""
    fr = sorted((f['upd'], f['lv']) for f in frames)
    T = [u for u, _ in fr]
    signer_of = {l['mk_hash']: l['mk_signer'] for l in L}
    ev, ahead, _ = prio(L, RECONCILING)                                   # claims may rest on any record whose kept variant reconciles
    _e, _a, tests = prio(L, {t for t in RECONCILING if t not in REC_STATUS})   # the reported tests: clean records
    rows = []
    for otag, g in F.groupby('x_otag', sort=False):
        if otag == 'hF2':
            continue
        h = g.mk_hash.iloc[0]
        g = F[F.mk_hash == h]
        sd, pt = g.y_side.iloc[0], int(g.y_ptick.iloc[0])
        lo, hi, sh = g.x_lo_hard_ms.to_numpy(), g.hi_ms.to_numpy(), g.shares.to_numpy()      # every claim is bounded by the HARD lower end
        cum, n = float(sh.sum()), len(g)
        lmax = int(lo[lo >= 0].max()) if (lo >= 0).any() else -1
        cap = None
        if (lo >= 0).all():
            c = [u for u, lv in fr if u <= lo.min() and lv.get((sd, pt), 0) < cum - 1e-9]
            cap = max(c) if c else None
        last = g.loc[g.hi_ms.idxmax()]
        first_ = g.loc[g.hi_ms.idxmin()]
        ev_last = (last.sid, int(last.mk_idx)) in ev
        j = next((i for i, u in enumerate(T) if u >= hi.max()), None)
        zero = j is not None and fr[j][1].get((sd, pt), 0) <= 1e-12
        fb = next((lv for u, lv in fr if u == first_.lo_ms), None)
        for s in SC:
            fifo_s = s in FIFO
            if fifo_s and ev_last:
                rs, rlo, rhi, rm = 'exhausted', lmax, int(last.hi_ms), 'price_priority'
            elif zero:            # declared: book-based removal evidence keeps BOTH exhausted and cancelled_or_expired (no exact size)
                rs, rlo, rhi, rm = 'cancelled_or_expired|exhausted', lmax, int(T[j]), 'shown_zero'
            else:
                rs, rlo, rhi, rm = WIDE, lmax, -1, 'none'
            rows.append(dict(scenario=s, mid=mid, mk_hash=h, mk_signer=signer_of[h], y_side=sd, y_ptick=pt, n_fills=n, cum_shares=cum, first_hi_ms=int(hi.min()),
                             last_hi_ms=int(hi.max()), size_lb=cum, size_exact=cum if (fifo_s and ev_last) else np.nan, size_class='',
                             post_lo_ms=cap if cap is not None else -1, post_hi_ms=int(hi.min()), post_class=C.CONSTRAINED if cap is not None else C.UNRESOLVED,
                             post_method='capacity' if cap is not None else 'none',
                             persist_lo_ms=int(hi.min()) if n >= 2 and hi.min() <= lmax else -1, persist_hi_ms=lmax if n >= 2 and hi.min() <= lmax else -1,
                             persist_class=C.OBSERVED if n >= 2 and hi.min() <= lmax else C.UNRESOLVED,
                             ahead_lo=ahead.get((first_.sid, int(first_.mk_idx)), 0.0) if fifo_s else np.nan,
                             ahead_hi=float(fb.get((sd, pt), 0) - cum) if fifo_s and fb is not None else np.nan, queue_class='',
                             removal_set=rs, removal_lo_ms=rlo, removal_hi_ms=rhi, removal_class='', removal_method=rm, status=ORDER_STATUS.get(otag, 'ok'), x_otag=otag,
                             x_size_ub=cum if (fifo_s and ev_last) else np.nan))       # an upper bound stated only where the exact size is (exercises size.ub)
    O = pd.DataFrame(rows)
    for grp, (cols, ccol) in M.GROUPS.items():         # class = function of the values across the three scenarios (contract section 5)
        for h, g in O.groupby('mk_hash'):
            vals = {tuple('nan' if (isinstance(r[c], float) and np.isnan(r[c])) else r[c] for c in cols) for _, r in g.iterrows()}
            if len(vals) > 1:
                cl = C.SENSITIVE
            elif grp == 'size':
                cl = C.CONSTRAINED if np.isfinite(g.size_exact.iloc[0]) else C.OBSERVED
            elif grp in ('post', 'persist'):
                cl = g[ccol].iloc[0]
            elif grp == 'queue':
                cl = C.UNRESOLVED
            else:
                cl = C.CONSTRAINED if g.removal_method.iloc[0] == 'shown_zero' else C.UNRESOLVED
            O.loc[g.index, ccol] = cl
    for c in ('post_lo_ms', 'post_hi_ms', 'persist_lo_ms', 'persist_hi_ms', 'removal_lo_ms', 'removal_hi_ms', 'n_fills', 'first_hi_ms', 'last_hi_ms', 'y_ptick', 'mid'):
        O[c] = O[c].astype(np.int64)
    return O, tests


def small_market(day, mid, W, rel, tagp, signer, taker, side_ask=True, status='ok', recv_at=None):
    """A FILL-only market record (window start W, match at W + rel): records / legs rows and three tob frames (no depth / lifecycle)."""
    ords = dict(hM=(signer, 'Ask' if side_ask else 'Bid', 1, .60, 3., -5000, None))
    rh = {'hM': '0x' + hx('order' + tagp)}
    rec, settle = build_raw_record(mid, W // 1000, W, tagp, 'M1', rel, taker, 'Bid' if side_ask else 'Ask', 1, [('hM', 2.)], 'SHARES', rh, ords=ords)
    fr = (recv_at or settle + 400) * 10 ** 6
    tk, m = rec['taker'], rec['makers'][0]
    tys = C.yes_space(tk['quoteType'], 1, int(tk['price']) / E18)[0]
    ysd, yp = C.yes_space(m['quoteType'], 1, int(m['price']) / E18)
    R = [dict(day=day, mid=mid, sid=rec['settlementId'], tx=rec['transactionHash'], settle_ms=settle, exec_ms=M.iso_ms(rec['executedAt']), first_recv_ns=fr, n_recv=1,
              amount_filled=2., price_executed=int(tk['price']) / E18, n_makers=1, tk_hash=tk['hash'], tk_signer=tk['signer'], tk_quote=tk['quoteType'], tk_index_set=1,
              tk_price=int(tk['price']) / E18, tk_amount=2., tk_fee=int(tk['fee']['amount']) / E18, tk_fee_type='SHARES', tk_y_side=tys, status=status, x_n_variants=1)]
    L = [dict(day=day, mid=mid, sid=rec['settlementId'], tx=rec['transactionHash'], settle_ms=settle, exec_ms=M.iso_ms(rec['executedAt']), first_recv_ns=fr, mk_idx=0,
              n_makers=1, mk_hash=m['hash'], mk_signer=m['signer'], mk_quote=m['quoteType'], mk_index_set=1, mk_price=.6, shares=2., mk_fee=0., mk_fee_type='SHARES',
              y_side=ysd, y_price=yp, y_ptick=int(round(yp * 100)), tk_hash=tk['hash'], tk_signer=tk['signer'], tk_y_side=tys, phase='live', t_rel_s=(settle - W) / 1000.,
              status=status, x_tag='M1', x_otag='hM')]
    TB = pd.DataFrame([dict(day=day, mid=mid, upd_ms=W + k * 1000, recv_ns=(W + k * 1000 + 50) * 10 ** 6, conn='c1', order_count=2, n_bid=1, n_ask=1, best_bid=.58,
                            best_bid_sh=5., best_ask=.60, best_ask_sh=3., pend_bid_usd=0., pend_ask_usd=0., status='ok') for k in range(3)])
    return dict(R=R, L=L, TB=TB)


# ------------------------------------------------------------------------------------------------------------------- econ / truth
def econ(day, mkts, tobs, regs, ex_sh, ex_pnl):
    """Economics of one FILL day in the PINNED columns: one taker row per maker leg valued at that leg's YES price (taker fee split by
    shares), any non-ok record flag excludes the whole record, boundary_censored markets kept only as rows WITH that status."""
    FV = []
    for mid, (R, L) in mkts.items():
        ws, we, V = regs[mid]
        for r in R:
            fl = set(r['status'].split('+'))
            st = 'ok' if fl == {'ok'} else ('boundary_censored' if 'boundary_censored' in fl else None)
            if st is None:
                continue
            ml = [l for l in L if l['sid'] == r['sid']]
            Q = sum(l['shares'] for l in ml)
            fee_res = r['tk_fee'] * (V if r['tk_index_set'] == 1 else 1 - V) if r['tk_fee_type'] == 'SHARES' else r['tk_fee']
            st_t = 1. if r['tk_y_side'] == 'bid' else -1.
            for l in ml:
                w = l['shares'] / Q      # declared (amendments 1-2): a SHARES taker fee converts at EACH LEG's own price, never priceExecuted
                own_px = l['y_price'] if r['tk_index_set'] == 1 else 1 - l['y_price']
                fee_fill = r['tk_fee'] * own_px if r['tk_fee_type'] == 'SHARES' else r['tk_fee']
                for role, s, ad, fr, ff in (('taker', st_t, r['tk_signer'], fee_res * w, fee_fill * w),
                                            ('maker', 1. if l['y_side'] == 'bid' else -1., l['mk_signer'], 0., 0.)):
                    g = s * l['shares'] * (V - l['y_price'])
                    FV.append(dict(day=day, mid=mid, sid=r['sid'], mk_idx=l['mk_idx'], role=role, addr=ad.lower(), s=s, q=l['shares'], p=l['y_price'], V=V,
                                   gross_res=g, fee_res=fr, fee_fill=ff, net_res=g - fr, status=st, x_t=l['settle_ms'], x_trel=(l['settle_ms'] - ws) / 1000.))
    FV = pd.DataFrame(FV)
    MMr = []
    for (a, mid, st), g in FV.sort_values(['x_t', 'sid', 'role', 'mk_idx'], kind='stable').groupby(['addr', 'mid', 'status'], sort=False):
        ws, we, V = regs[mid]
        tb = tobs[(tobs.mid == mid) & (tobs.upd_ms <= we - 2000)].sort_values('upd_ms')
        mend = float((tb.best_bid.iloc[-1] + tb.best_ask.iloc[-1]) / 2) if len(tb) else np.nan     # no book (ID day): END undefined
        lots, rt = [], 0.
        for s, q, p in zip(g.s, g.q, g.p):
            while q > 1e-12 and lots and lots[0][0] == -s:
                ls, lq, lp = lots[0]; m_ = min(q, lq)
                rt += m_ * ((p - lp) if s < 0 else (lp - p)); q -= m_
                lots[0:1] = [(ls, lq - m_, lp)] if lq - m_ > 1e-12 else []
            if q > 1e-12:
                lots.append((s, q, p))
        drift = sum(s * q * (mend - p) for s, q, p in lots)
        gmk, gtk = float(g[g.role == 'maker'].gross_res.sum()), float(g[g.role == 'taker'].gross_res.sum())
        fr, ff, gross = float(g.fee_res.sum()), float(g.fee_fill.sum()), gmk + gtk
        MMr.append(dict(day=day, mid=mid, addr=a, gross_res_mk=gmk, gross_res_tk=gtk, fee_res=fr, fee_fill=ff, net_res=gross - fr, rt_pnl=rt, resid_res=gross - rt,
                        resid_drift=drift, resid_resol=gross - rt - drift, net_pos=float((g.s * g.q).sum()), status=st,
                        fifo_status='ok' if np.isfinite(mend) else 'end_unavailable'))
    MM = pd.DataFrame(MMr)
    sums = ['gross_res_mk', 'gross_res_tk', 'fee_res', 'fee_fill', 'net_res', 'rt_pnl', 'resid_res', 'resid_drift', 'resid_resol', 'net_pos']
    MD = MM[MM.status == 'ok'].groupby(['day', 'addr'], as_index=False)[sums].sum()
    MD = MD.assign(in_exante_sh=MD.addr.isin(set(ex_sh)), in_exante_pnl=MD.addr.isin(set(ex_pnl)), status='ok')
    tb_ = lambda t: 'pre' if t < 0 else ('post' if t >= 300 else f'{int(t // 60) * 60}-{int(t // 60) * 60 + 60}')
    rg_ = lambda p: '>=0.4' if abs(p - .5) >= .4 else ('0.2-0.4' if abs(p - .5) >= .2 else ('0.1-0.2' if abs(p - .5) >= .1 else '<0.1'))
    ok = FV[FV.status == 'ok']
    base = dict(day=day, id_set='all', label='all', grp='field', role='all', status='ok')
    SG = []
    if len(ok):
        SG.append(dict(base, dim='all', bucket='all', net_res=float(ok.net_res.sum()), gross_res=float(ok.gross_res.sum()), n=len(ok)))
        for dim, key in (('tseg', ok.x_trel.map(tb_)), ('pseg', ok.p.map(rg_)), ('kind', ok.role), ('tclock', pd.Series('settle', index=ok.index))):
            for b, g in ok.groupby(key):
                SG.append(dict(base, dim=dim, bucket=b, net_res=float(g.net_res.sum()), gross_res=float(g.gross_res.sum()), n=len(g)))
        SG.append(dict(base, dim='life_rep', bucket='repeated', net_res=1.0, gross_res=1.0, n=1))       # BOOK-only dim: not a partition of all fills
    return (FV.drop(columns=['x_t', 'x_trel']), MM, MD,
            pd.DataFrame(SG, columns=['day', 'id_set', 'label', 'grp', 'role', 'dim', 'bucket', 'net_res', 'gross_res', 'n', 'status']))


def coverage_nested(O, F):
    """Pinned lifecycle form: {field: {class: {orders, executions, shares}}} (counts; classes are scenario-invariant -> one scenario's rows)."""
    agg = F.assign(h=F.mk_hash.str.lower()).groupby('h').agg(n=('shares', 'size'), cum=('shares', 'sum'))
    j = O[O.scenario == SC[0]].assign(h=lambda d: d.mk_hash.str.lower()).join(agg, on='h')
    return {grp: {cl: dict(orders=int(len(g)), executions=int(g.n.sum()), shares=float(g.cum.sum())) for cl, g in j.groupby(col)}
            for grp, (_, col) in M.GROUPS.items()}


def coverage(O, F):
    agg = F.assign(h=F.mk_hash.str.lower()).groupby('h').agg(n=('shares', 'size'), cum=('shares', 'sum'))
    j = O.assign(h=O.mk_hash.str.lower()).join(agg, on='h')
    out = {}
    for s, g in j.groupby('scenario'):
        out[s] = {}
        for grp, (_, col) in M.GROUPS.items():
            out[s][grp] = {}
            for den, w in (('orders', None), ('executions', 'n'), ('shares', 'cum')):
                c = g.groupby(col).size() if w is None else g.groupby(col)[w].sum()
                out[s][grp][den] = {k: round(float(v) / float(c.sum()), 4) for k, v in c.items()}
    return out


def write_gz(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as f:
        for ln in lines:
            f.write(ln + '\n')


def to_pq(df, path, schema=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(df)
    if schema:
        for c, t in schema.items():
            df[c] = df[c].astype({'int64': np.int64, 'float64': np.float64, 'str': object}[t])
        df = df[list(schema) + [c for c in df.columns if c.startswith('x_') and c not in ('x_tag', 'x_otag')]]
    df.to_parquet(path, index=False)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='mt_checks_rehearse_'))
    raw, step1, out = tmp / 'raw', tmp / 'step1', tmp / 'out'
    D1, D7, DG, DI = '20261024', '20261027', C.GT_DAY, C.ID_DAY
    W1, W2, W0, W7 = ms_of(2026, 10, 24, 5, 10), ms_of(2026, 11, 8, 10, 20), ms_of(2026, 10, 24, 5, 15), ms_of(2026, 10, 27, 23, 55)
    WI, WI0, W3 = ms_of(2026, 10, 23, 12, 0), ms_of(2026, 10, 23, 0, 0), ms_of(2026, 11, 8, 22, 55)
    M1, M0, M7, M2, M3, MI, MI0 = 9100001, 9100002, 9100007, 9200001, 9200002, 9300001, 9300002
    m1, m2 = market(D1, M1, W1, 'b'), market(DG, M2, W2, 'g')
    m0 = small_market(D1, M0, W0, 5300, 'f0', A_, X_)
    m7 = small_market(D7, M7, W7, 5300, 'f7', A_, X_, status='boundary_censored', recv_at=W7 + 120000)          # upper boundary (FILL day)
    W7b, M7b = ms_of(2026, 10, 27, 12, 0), 9100008
    m7b = small_market(D7, M7b, W7b, 5300, 'f7b', B_, Y_)                                                         # a clean Oct 27 market
    m3 = small_market(DG, M3, W3, 5300, 'f3', B_, Y_, status='conflict+boundary_censored', recv_at=W3 + 120000)   # GT day, DECLARED flag order
    m3d = dict(FR=pd.DataFrame([dict(mid=M3, fid=k, upd_ms=W3 + k * 1000, recv_ns=(W3 + k * 1000 + 50) * 10 ** 6, conn='c1', order_count=2, pend_bids='[]',
                                     pend_asks='[]', los_id='1', los_side='Bid', los_price=0.5, los_outcome='Yes', status=st) for k, st in enumerate(['ok', 'gap', 'ok'])]),
               LV=pd.DataFrame([dict(mid=M3, fid=k, side=sd, ptick=p, shares=q, status='ok') for k in range(3) for sd, p, q in (('bid', 58, 5.), ('ask', 60, 3.))]))
    # ^ depth the extractor wrote for the censored GT window M3 (no lifecycle): counted in the Nov 8 depth day totals only
    mi_a = small_market(DI, MI, WI, 5300, 'i1', A_, X_)
    mi_b = small_market(DI, MI, WI, 5400, 'i2', C_, Y_, side_ask=False)
    mi_0 = small_market(DI, MI0, WI0, 5300, 'i0', B_, X_, status='boundary_censored')                             # lower boundary (Oct 23 00:00)
    # ---- raw (consumed-day stems only)
    write_gz(raw / 'rest' / f'{D1}_05.jsonl.gz', m1['rest']); write_gz(raw / 'predict_ws' / f'{D1}_05.jsonl.gz', m1['ws'])
    write_gz(raw / 'rest' / f'{DG}_10.jsonl.gz', m2['rest']); write_gz(raw / 'predict_ws' / f'{DG}_10.jsonl.gz', m2['ws'])
    regs = {M1: (W1, W1 + 300000, 1.0), M0: (W0, W0 + 300000, 0.0), M7: (W7, W7 + 300000, 1.0), M7b: (W7b, W7b + 300000, 0.5),     # M7b: a tie -> V 0.5
            MI: (WI, WI + 300000, 1.0), MI0: (WI0, WI0 + 300000, 0.0)}
    # (mid, ws, outcome, start, end, outcome seen at ms): M7's outcome is first seen after Oct 27 ends -> sealed, blanked by the checker
    reg_rows = {D1: [(M1, W1, 'Up', 100., 101., W1 + 330000), (M0, W0, 'Down', 100., 99., W0 + 330000)],
                D7: [(M7, W7, 'Up', 100., 101., W7 + 330000), (M7b, W7b, 'Down', 100., 100., W7b + 330000)],
                DI: [(MI, WI, 'Up', 100., 101., WI + 330000), (MI0, WI0, 'Down', 100., 99., WI0 + 330000)]}
    for d, rows in reg_rows.items():
        to_pq(dict(mid=[r[0] for r in rows], slug=[f's{r[0]}' for r in rows], ws=[r[1] / 1000 for r in rows], we=[r[1] / 1000 + 300 for r in rows],
                   outcome=[r[2] for r in rows], start_price=[r[3] for r in rows], end_price=[r[4] for r in rows],
                   outcome_seen_ns=[r[5] * 10 ** 6 for r in rows]), step1 / d / 'registry.parquet')
    # ---- builder outputs
    S = C.SCHEMAS
    for day, parts in ((D1, [m1, m0]), (D7, [m7, m7b]), (DG, [m2, m3]), (DI, [mi_a, mi_b, mi_0])):
        R = sum((p['R'] for p in parts), []); L = sum((p['L'] for p in parts), [])
        to_pq(R, out / f'records_{day}.parquet', S['records']); to_pq(L, out / f'legs_{day}.parquet', S['legs'])
        counts = dict(records=fcount(r['status'] for r in R), legs=fcount(l['status'] for l in L))
        if day != DI:
            TB = pd.concat([p['TB'] for p in parts], ignore_index=True)
            to_pq(TB, out / f'tob_{day}.parquet', S['tob']); counts['tob'] = fcount(TB.status)
        depth = {D1: [m1], DG: [m2, m3d]}.get(day)          # pinned: depth status counts as DAY totals over EVERY depth window written
        if depth is not None:                               # (e2e 4: Nov 8 includes the boundary_censored M3 window, never scored)
            counts['depth_day_totals'] = dict(frames=fcount(pd.concat([x['FR'].status for x in depth])),
                                              levels=fcount(pd.concat([x['LV'].status for x in depth])))
        extra = dict(ws_population=dict(selected=[M2, M3])) if day == DG else {}     # GT windows with an own maker leg (M3 later censored)
        (out / f'extract_{day}.json').write_text(json.dumps(dict(status_orders=STATUS_ORDERS,        # pinned: per-table declared flag orders
                                                                 status_flag_counts=counts, lag_p999_all_ms=103000, note='synthetic', **extra)))
        if day == DG:
            to_pq(pd.DataFrame(dict(mid=[M2, M3], status=['ok', 'boundary_censored'])), out / f'markets_{day}.parquet')
        (out / f'inventory_{day}.json').write_text(json.dumps(dict(fields=['taker', 'makers', 'settlementId', 'executedAt'], rewards='inventoried')))
    to_pq(m3d['FR'], out / 'depth' / f'{DG}_{M3}_frames.parquet', S['frames']); to_pq(m3d['LV'], out / 'depth' / f'{DG}_{M3}_levels.parquet', S['levels'])
    for day, mid, m in ((D1, M1, m1), (DG, M2, m2)):
        to_pq(m['FR'], out / 'depth' / f'{day}_{mid}_frames.parquet', S['frames']); to_pq(m['LV'], out / 'depth' / f'{day}_{mid}_levels.parquet', S['levels'])
        to_pq(m['F'], out / 'life' / f'{day}_{mid}_fills.parquet', S['fills'])
        to_pq(m['O'], out / 'life' / f'{day}_{mid}_orders.parquet', S['orders'])
        t = m['tests']      # pinned: T1 / T2 counts under 'violations' ('makers' / 'groups' denominators), settle-order + window-certain
        unus = m['FR'].status.str.split('+').map(lambda f: bool(set(f) & {'bad_frame', 'mid_mismatch', 'out_of_order', 'off_grid'}))
        (out / 'life' / f'{day}_{mid}.json').write_text(json.dumps(dict(
            status_flag_counts=dict(fills=fcount(m['F'].status), orders=fcount(m['O'].status)),
            frames_excluded=({} if mid == M1 else int(unus.sum())),          # M1: an empty map as the e2e lifecycle writes it (no unusable frames)
            priority_tests=dict(T1=dict(settle_order=dict(violations=t['t1_legs'], makers=3), window_certain=dict(violations=1, makers=3)),
                                T2=dict(settle_order=dict(violations=t['t2'], groups=4), window_certain=dict(violations=1, groups=4))),
            coverage=coverage_nested(m['O'], m['F']))))
        to_pq(pd.DataFrame(dict(mid=[mid], kind=['refill'], h_from=['x'], h_to=['y'], status=['ok'])), out / 'life' / f'{day}_{mid}_x_links.parquet')
    (out / 'coverage.json').write_text(json.dumps(dict(all_markets=coverage(pd.concat([m1['O'], m2['O']]), pd.concat([m1['F'], m2['F']])))))
    mi_L = pd.DataFrame(mi_a['L'] + mi_b['L'])            # ex-ante identities (Oct 23, clean records only)
    sh = mi_L.groupby(mi_L.mk_signer.str.lower()).shares.sum()
    net = pd.Series({a: float(sum((1 if r.y_side == 'bid' else -1) * r.shares * (1.0 - r.y_price) for r in g.itertuples())) for a, g in mi_L.groupby(mi_L.mk_signer.str.lower())})
    ex_sh, ex_pnl = sorted(sh.index, key=lambda x: (-sh[x], x))[:10], sorted(net.index, key=lambda x: (-net[x], x))[:10]
    tob = pd.concat([pd.read_parquet(out / f'tob_{d}.parquet') for d in (D1, D7)], ignore_index=True)
    E1 = econ(D1, {M1: (m1['R'], m1['L']), M0: (m0['R'], m0['L'])}, tob, regs, ex_sh, ex_pnl)
    E7 = econ(D7, {M7: (m7['R'], m7['L']), M7b: (m7b['R'], m7b['L'])}, tob, regs, ex_sh, ex_pnl)
    EI = econ(DI, {MI: (mi_a['R'] + mi_b['R'], mi_a['L'] + mi_b['L']), MI0: (mi_0['R'], mi_0['L'])}, tob, regs, ex_sh, ex_pnl)    # pinned: the ID day is valued (no book)
    for k, a, b, c in zip(('fills_valued', 'maker_market', 'maker_day', 'segments'), E1, E7, EI):
        to_pq(pd.concat([a, b, c], ignore_index=True), out / 'econ' / f'{k}.parquet')
    (out / 'econ' / 'identities.json').write_text(json.dumps(dict(exante_top_shares=ex_sh, exante_top_pnl=ex_pnl)))
    (out / 'econ' / 'econ.json').write_text(json.dumps(dict(note='synthetic receipt')))
    st_ = dict(FILLED='filled', EXPIRED='expired')       # v2 truth (real names): activation hull, true / filled size, final_state
    own = [dict(mid=M2, mk_hash=m2['rh'][t], act_lo_venue_ms=a - 100, act_hi_venue_ms=a + 100, size_sh=s, filled_sh=f, final_state=st_[st], status='ok')
           for t, (a, s, st, f) in m2['truth'].items()]
    own.append(dict(mid=M2, mk_hash='0x' + hx('own-unfilled'), act_lo_venue_ms=W2 + 900, act_hi_venue_ms=W2 + 1100, size_sh=2., filled_sh=0.,
                    final_state='cancelled', status='ok'))
    own.append(dict(mid=M3, mk_hash=m3['L'][0]['mk_hash'], act_lo_venue_ms=W3 - 5100, act_hi_venue_ms=W3 - 4900, size_sh=3., filled_sh=2.,
                    final_state='filled', status='ok'))       # e2e 4: a FILLED own order in the boundary_censored window M3 -> outside the GT score
    tbl = pa.Table.from_pandas(pd.DataFrame(own), preserve_index=False)
    tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}), b'mt_truth': json.dumps(dict(
        act_lo_venue_ms='activation hull lower (venue ms)', act_hi_venue_ms='activation hull upper (venue ms)', size_sh='true size', filled_sh='filled shares',
        final_state='final state')).encode()})
    (out / 'truth' / 'v2').mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out / 'truth' / 'v2' / 'own_orders.parquet')
    blk = lambda n, **k: dict(n=n, failed=0, defects=0, sensitive_misses=0, **k)       # hand-derived for our own orders hA / hD (both hulls inside)
    po = lambda n: blk(n, inside=n, overlap=0, disjoint=0)
    # e2e 4: posting / size blocks NESTED by class {class: {n, <outcomes>, failed, defects, sensitive_misses}} (hA: post constrained, size
    # assumption_sensitive; hD: post unresolved, size observed; a zero 'observed' posting class is listed as the builder may)
    sc = {s: dict(coverage=dict(orders=dict(n=2, covered=2), executions=dict(n=4, covered=4), shares=dict(n=9.5, covered=9.5)),
                  posting={C.CONSTRAINED: po(1), C.UNRESOLVED: po(1), C.OBSERVED: po(0)},
                  size=dict(lb={C.OBSERVED: blk(2, held=2)},        # declared (contract §7): lb always observed; exact / ub over EVERY order
                            exact={C.SENSITIVE: blk(1, not_stated=0 if s in FIFO else 1), C.OBSERVED: blk(1, not_stated=1)},
                            ub={C.SENSITIVE: blk(1, not_stated=0 if s in FIFO else 1), C.OBSERVED: blk(1, not_stated=1)}),
                  # declared (e2e 6): every block counts EVERY scored order; orders without the claim are an explicit 'not_*' outcome
                  removal=dict(exhausted_only=blk(2, not_exhausted_only=1 if s in FIFO else 2), membership=blk(2), window=blk(2)),
                  persistence=blk(2, not_stated=2),
                  queue=dict(own_front_claims=blk(2, not_stated=2), other_makers_front_claims=blk(2, not_stated=2)), defects_total=0) for s in SC}
    (out / 'truth' / 'v2' / 'scores.json').write_text(json.dumps(dict(population=dict(own_orders=4, filled=2), scenarios=sc, defect_rows=[],
                                                               defects_total=dict(fills=0, **{s: 0 for s in SC}), inputs=dict(note='synthetic'))))
    # ---- hand-derived expectations of the spec (exact values; guards the fixture itself)
    O1 = m1['O'].set_index(['x_otag', 'scenario'])
    R1 = pd.DataFrame(m1['R'])
    exp = [(O1.loc[('hA', 'S_FIFO'), 'post_lo_ms'] == W1 + 2000, 'hA capacity bound = frame +2 s'), (O1.loc[('hE', 'S_NOPRIO'), 'post_lo_ms'] == W1 + 54000, 'hE capacity +54 s'),
           (O1.loc[('hA', 'S_FIFO'), 'size_exact'] == 6.5, 'hA exhausted by price priority at 6.5'), (np.isnan(O1.loc[('hA', 'S_NOPRIO'), 'size_exact']), 'no exact size without priority'),
           (O1.loc[('hE', 'S_NOPRIO'), 'removal_class'] == C.CONSTRAINED, 'hE shown-zero exhaustion constrained'),
           (O1.loc[('hA', 'S_NOPRIO'), 'removal_class'] == C.SENSITIVE, 'hA removal assumption-sensitive'),
           (O1.loc[('hK', 'S_FIFO'), 'removal_set'] == 'exhausted' and O1.loc[('hK', 'S_FIFO'), 'size_exact'] == 1, 'hK exhausted by the k-1 worst-price rule'),
           (O1.loc[('hI', 'S_FIFO'), 'removal_set'] != 'exhausted', 'hI NOT exhausted: makers[] order is never evidence'),
           (O1.loc[('hL', 'S_FIFO'), 'ahead_lo'] == 1.0, 'hL had hK ahead (ahead_lo = 1)'), (O1.loc[('hA', 'S_FIFO'), 'persist_class'] == C.OBSERVED, 'hA persistence observed'),
           (m1['tests'] == dict(t1_legs=1, t1_hashes=1, t2=1), 'T1 = 1 (hP), T2 = 1 (R16)'), (len(m1['O']) == 18 * 3, '18 hashes x 3 scenarios'),
           ((m1['F'].lo_ms == -1).sum() == 1, 'one unbounded fill'), ((R1.status == 'ok').sum() == 16, '16 clean records (a price gap is no status)'),
           (float(pd.DataFrame(m1['L']).query("x_tag == 'R9'").shares.sum()) == 2.0, 'R9 keeps the reconciling variant (2 shares)')]
    O2 = m2['O'].set_index(['x_otag', 'scenario'])        # guards the hand-derived nested GT blocks
    exp += [(all(O2.loc[(t, s), 'post_class'] == c for s in SC for t, c in (('hA', C.CONSTRAINED), ('hD', C.UNRESOLVED))), 'GT posting classes hA / hD'),
            (all(O2.loc[(t, s), 'size_class'] == c for s in SC for t, c in (('hA', C.SENSITIVE), ('hD', C.OBSERVED))), 'GT size classes hA / hD'),
            (all(np.isfinite(O2.loc[('hA', s), 'size_exact']) == (s in FIFO) and np.isnan(O2.loc[('hD', s), 'size_exact']) for s in SC), 'GT exact sizes')]
    L18 = [l for l in m1['L'] if l['x_tag'] == 'R18']
    R18 = next(r for r in m1['R'] if r['sid'] == L18[0]['sid'])
    exp += [(abs(sum(l['shares'] * l['y_price'] for l in L18) / sum(l['shares'] for l in L18) - 0.625) < 1e-12 and R18['price_executed'] == 0.63
             and R18['tk_index_set'] == 1 and R18['tk_fee_type'] == 'SHARES' and R18['tk_fee'] > 0 and R18['status'] == 'ok',
             'R18: own-price mean 0.625 vs priceExecuted 0.63 under a SHARES taker fee (the two conversions differ)')]
    bad_exp = [w for ok, w in exp if not ok]
    # ---- run
    E = M.load(out, days=(DI, D1, D7, DG), book={D1: [M1]}, gt_ws={M2: W2 // 1000, M3: W3 // 1000}, raw_root=raw, step1=step1)   # GT from the receipt
    exp += [(E['gt'] == {DG: [M2]} and E['gt_reparse'] == (DG, M2), 'GT population = selected minus censored; reparse = first selected'),
            (M.reg(E, D7)[M7b][2] == 0.5 and np.isnan(M.reg(E, D7)[M7][2]), 'tie -> V 0.5; an outcome first seen after Oct 27 -> blanked'),
            (E['gt_censored'] == {M3} and E['depth_only'] == [(DG, M3)], 'censored GT window M3: out of the score, its depth in the day totals')]
    bad_exp = [w for ok, w in exp if not ok]
    res = M.run_checks(E)
    g_ = res['details'].get('gt_scores', {})
    exp += [(g_.get('gt_filled_own_orders') == 2 and g_.get('gt_own_orders_in_censored_windows') == 1, 'GT filled own orders = 2 (M3 order excluded)'),
            (g_.get('gt_blocks_nested', 0) == 4 * len(SC) and g_.get('gt_blocks_flat', 0) == len(SC) and not g_.get('gt_blocks_not_recomputable'),
             'GT blocks read nested (posting, size.lb/.exact/.ub) + flat removal; none unverified')]
    bad_exp = [w for ok, w in exp if not ok]
    fails = {n: res['details'][n]['reason'] for n, v in res['checks'].items() if not v}
    rp = {k: M.reparse(d, m, E) for k, (d, m) in (('book', (D1, M1)), ('gt', E['gt_reparse']))}
    rp_fail = {f'{k}:{n}': r['details'][n]['reason'] for k, r in rp.items() for n, v in r['checks'].items() if not v}
    fz = M.falsify(E, raw, (D1, M1), base_tmp=tmp)
    fz_fail = {k: v for k, v in fz.items() if v['refused'] is not True}           # not plantable counts as a failure in the fixture
    untargeted = sorted((set(res['checks']) | set(rp['book']['checks'])) - {v['check'] for v in fz.values() if v['refused'] is True})
    summary = dict(fixture_expectations_failed=bad_exp, checks_total=len(res['checks']), checks_failed=fails, info=res['info'],
                   reparse={k: dict(n_checks=len(r['checks']), info={i: r['info'][i] for i in ('unique_records', 'legs', 'conflicts', 'amount_mismatch', 'price_mismatch',
                                                                                            'repolled', 'frames', 'hours_clipped', 'rest_hours_read', 'h3_violating_prices',
                                                                                            'capacity_bounds', 'builder_variant_column') if i in r['info']}) for k, r in rp.items()},
                   reparse_failed=rp_fail, falsifier_cases=len(fz), falsifier_not_refused=fz_fail, checks_without_a_refused_falsifier=untargeted,
                   falsifier={k: dict(check=v['check'], refused=v['refused'], got=v['got'][0][:160] if v['got'] and v['got'][0] else None) for k, v in fz.items()},
                   n_eval={n: res['details'][n]['n_eval'] for n in res['checks']},
                   priority_info={k: res['details']['priority_exhaustion'].get(k) for k in res['details']['priority_exhaustion'] if k.startswith('multi_price') or k.startswith('price')},
                   vmhwm_mib=round(M.hwm_mib(), 1))
    print(json.dumps(summary, indent=1, default=str))
    ok = not bad_exp and not fails and not rp_fail and not fz_fail and not untargeted and len(fz) >= 60
    print('REHEARSAL', 'PASS' if ok else 'FAIL')
    if '--keep' not in sys.argv:
        shutil.rmtree(tmp)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
