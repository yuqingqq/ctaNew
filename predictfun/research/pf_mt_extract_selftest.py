"""Selftest of pf_mt_extract (MAKER-TRACE-RECONSTRUCTION-v1; contract predictfun/reviews/maker_trace_20261122/contract.md sections 2-4).
Synthetic fixtures only, in a temp dir: pf_mt_common.PF_RUN / OUT / STEP1 and the module's BOOK manifest / GT scope / GT registry paths are
monkeypatched, so no real file is read. Exact expected values; known-bad inputs must be refused with the reason asserted; filesystem calls are
recorded through monkeypatched gzip.open / open / os.stat / os.path.exists / os.listdir / os.scandir.
Cases: canonical content (re-poll invariance, content change, new key); exact-int parsing, executedAt ms, UUIDv7 decode, status order;
Forbidden for a 20261028 hour and day with ZERO filesystem calls; a FILL day (20261027): de-dup across hours (n_recv 3), conflict (first kept,
both counted), mint (NO bid 0.01 -> YES ask 0.99), integer reconciliation failures hidden by floats -> statuses, non-v7 / malformed /
page-mismatch statuses, pre / post-window phases, measured margins, boundary_censored for the 23:55 market, a missing REST and WS hour,
ws_clipped, frame statuses (dup, same_ms_changed, gap, out_of_order, new_conn; off_grid, dup_level in depth), exact tob / frames / levels rows,
schemas, the receipt's hour sha256 / bytes and the clipped Oct 28 hour never touched; write-once refusal with zero raw calls; an ID day (REST
only, the lower boundary); a GT day (scope population, hour 23 never touched, depth for every window); probe refusals; an incomplete registry
refused; the VmHWM guard stops before any output. Usage (run_light.sh): pf_mt_extract_selftest.py
"""
import builtins
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_mt_extract as E  # noqa: E402

C, E18, CENT = E.C, E.C.E18, 10 ** 16
FAILS, SEQ, NCASE = [], [0], [0]
SEALED_NS = (E.day_ms('20261027') // 1000 + 86400 + 1234) * 10 ** 9 + 567891                     # first seen Oct 28 00:20Z: a sealed hour


def check(name, ok, detail=''):
    NCASE[0] += 1
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  :: {detail}'), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def refused(name, fn, exc, text):
    try:
        fn()
    except exc as e:
        return check(name, text in str(e), f'reason {e}')
    except Exception as e:  # noqa: BLE001
        return check(name, False, f'wrong exception {type(e).__name__}: {e}')
    return check(name, False, 'not refused')


class Opens:
    """Records every gzip.open / open / io.open / os.stat / os.path.exists / os.listdir / os.scandir call while active."""
    NAMES = (('gzip', gzip, 'open'), ('open', builtins, 'open'), ('open', io, 'open'), ('stat', os, 'stat'), ('exists', os.path, 'exists'),
             ('listdir', os, 'listdir'), ('scandir', os, 'scandir'))

    def __enter__(self):
        self.calls, self.saved = [], [(mod, attr, getattr(mod, attr)) for _, mod, attr in self.NAMES]
        for (tag, mod, attr), (_, _, fn) in zip(self.NAMES, self.saved):
            setattr(mod, attr, self._wrap(tag, fn))
        return self

    def _wrap(self, tag, fn):
        calls = self.calls

        def w(p='.', *a, **k):
            calls.append((tag, str(os.fspath(p)) if isinstance(p, (str, bytes, os.PathLike)) else str(p)))
            return fn(p, *a, **k)
        return w

    def __exit__(self, *a):
        for mod, attr, fn in self.saved:
            setattr(mod, attr, fn)

    def raw(self):
        return [c for c in self.calls if c[1].startswith(str(C.PF_RUN))]


# ---------------------------------------------------------------- synthetic builders
def u7(ms, n):
    h = f'{ms:012x}'
    return f'{h[:8]}-{h[8:12]}-7{n:03x}-8{n:03x}-{n:012x}'


def iso(ms):
    return dt.datetime.fromtimestamp(ms // 1000, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.') + f'{ms % 1000:03d}Z'


def stem_of(ns):
    return dt.datetime.fromtimestamp(ns // 10 ** 9, dt.timezone.utc).strftime('%Y%m%d_%H')


def ns(ms):
    return ms * 10 ** 6


def leg(q, ix, price, amount, who, fee=0, ft='SHARES', ost=None, bb=None, oc='1', sg=None):
    return dict(quoteType=q, amount=str(amount), price=str(price),
                outcome=dict(name='Up' if ix == 1 else 'Down', indexSet=ix, onChainId=oc, status=ost, bestBid=bb, bestAsk=None, team=None, variantData=None),
                signer=sg or f'0xSIGNER_{who}', hash=f'0x{who:0>64}', fee=dict(amount=str(fee), type=ft))


TRICKY = 'He said "hi" \\ back\\slash \\" \r\nline 2 \u2603 caf\u00e9 {"market":{"id":1},"taker": ,\\"taker\\": \u0000 end'


def market(mid, ws, st='OPEN', boosted=False, ost=None, res=None, up_ix=1, desc=None):
    outs = [dict(name='Up', indexSet=up_ix, onChainId='1', status=ost, bestBid=None, bestAsk=None, team=None, variantData=None),
            dict(name='Down', indexSet=3 - up_ix, onChainId='2', status=None, bestBid=None, bestAsk=None, team=None, variantData=None)]
    return dict(id=mid, title='synthetic', categorySlug=f'btc-updown-5m-{ws}', feeRateBps=200, makerFeeBps=0, spreadThreshold=0.06, shareThreshold=100,
                rewards=dict(current=None, schedule=[dict(hourlyRate=3000, startsAt='s', endsAt='e')]), isBoosted=boosted,
                boostStartsAt='2026-10-27T12:00:00.000Z', boostEndsAt='2026-10-27T12:05:00.000Z', status=st, tradingStatus='OPEN', resolution=res,
                outcomes=outs, decimalPrecision=2, **({} if desc is None else dict(description=desc)))


def rec(mkt, sid, txn, taker, makers, ems, af=None, pe=None):
    return dict(market=mkt, taker=taker, amountFilled=str(sum(int(m['amount']) for m in makers)) if af is None else af,
                priceExecuted=taker['price'] if pe is None else pe, makers=makers, transactionHash=f'0x{txn:0>64}', settlementId=sid, executedAt=iso(ems))


def rest_row(recv, pmid, recs, status=200, raw=None, order=None, spaced=False):
    """(file-order key, REST row); the raw payload is compact JSON like the venue's (spaced=True: Python's default separators, which the
    market cut must refuse); `order` places a row later in its hour file than its recv_ns (a receipt-clock step back)."""
    SEQ[0] += 1
    body = json.dumps(dict(success=True, data=recs, cursor=None), separators=None if spaced else (',', ':'))
    return (recv if order is None else order), dict(run_id='synthetic', seq=SEQ[0], source='rest', kind='response', recv_ns=recv, recv_mono_ns=0, path='/v1/orders/matches',
                      params=dict(marketId=str(pmid), first='100'), sent=dict(recv_ns=recv - 1000, recv_mono_ns=0), status=status,
                      raw=body if raw is None else raw)


def ws_row(recv, conn, mid, data, topic=None, msg=None):
    SEQ[0] += 1
    return recv, dict(run_id='synthetic', seq=SEQ[0], source='predict_ws', kind='frame', recv_ns=recv, recv_mono_ns=0, connection_id=conn,
                      raw=json.dumps(msg if msg is not None else dict(type='M', topic=topic or f'predictOrderbook/{mid}', data=data)))


def write_family(family, rows, stems, missing=(), no_final_newline=False):
    """rows (recv_ns, row) -> hour files by receipt hour; every planned stem gets a (possibly empty) file except `missing`
    (no_final_newline: the last row of each file lacks its newline, as a truncated write would leave it)."""
    by = {}
    for recv, r in sorted(rows, key=lambda z: z[0]):
        by.setdefault(stem_of(recv), []).append(r)
    assert set(by) <= set(stems) - set(missing), f'synthetic rows outside the written hours: {sorted(set(by) - set(stems))}'
    for s in stems:
        if s in missing:
            continue
        p = C.hour_path(family, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, 'wt') as f:
            rs = by.get(s, [])
            for k, r in enumerate(rs):
                f.write(json.dumps(r, separators=(',', ':')) + ('' if no_final_newline and k == len(rs) - 1 else '\n'))


def write_registry(path, day, base, n=288, rows=None, override=None):
    d0 = E.day_ms(day) // 1000
    rows = rows or [dict(dict(mid=base + i, slug=f'btc-updown-5m-{d0 + 300 * i}', ws=float(d0 + 300 * i), we=float(d0 + 300 * i + 300),
                              outcome='Up' if i % 2 == 0 else 'Down', outcome_seen_ns=(d0 + 300 * i + 320) * 10 ** 9), **(override or {}).get(i, {})) for i in range(n)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def ev_row(recv, kind, **fields):
    SEQ[0] += 1
    return recv, dict(run_id='synthetic', seq=SEQ[0], source='events', kind=kind, recv_ns=recv, recv_mono_ns=0, **fields)


def sentinel_free(out, day, needles):
    """No output file of the day carries any needle (string columns / JSON text / integer columns)."""
    hits = []
    for f in sorted(Path(out).rglob(f'*{day}*')):
        if f.suffix == '.json':
            txt = f.read_text()
            hits += [(f.name, n) for n in needles if str(n) in txt]
        elif f.suffix == '.parquet':
            t = pq.read_table(f)
            for c in t.column_names:
                vals = set(t.column(c).to_pylist())
                hits += [(f.name, c, n) for n in needles if n in vals or (isinstance(n, str) and any(isinstance(v, str) and n in v for v in vals))]
    return hits


def orders_ok(R, out, day):
    """N5: every status of every output decomposes into the declared order list of its table."""
    so, bad = R['status_orders'], []
    def ok_(s_, order):
        if s_ == so['ok']:
            return True
        f = s_.split(so['joiner'])
        return all(x in order for x in f) and [order.index(x) for x in f] == sorted(order.index(x) for x in f) and len(set(f)) == len(f)
    files = [(f'records_{day}.parquet', 'records'), (f'legs_{day}.parquet', 'legs'), (f'tob_{day}.parquet', 'tob'), (f'markets_{day}.parquet', 'markets')]
    files += [(f'depth/{f.name}', 'frames' if f.name.endswith('_frames.parquet') else 'levels') for f in (Path(out) / 'depth').glob(f'{day}_*.parquet')]
    for name, kind in files:
        if (Path(out) / name).exists():
            bad += [(name, s_) for s_ in set(pq.read_table(Path(out) / name, columns=['status']).column('status').to_pylist()) if not ok_(s_, so[kind])]
    return not bad, bad


def table(path):
    return pq.read_table(path)


def rows(path, key=None):
    t = pq.read_table(path).to_pylist()
    return {r[key]: r for r in t} if key else t


def same(a, b):
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


def sub(row, **want):
    bad = {k: (row.get(k), v) for k, v in want.items() if not same(row.get(k), v)}
    return not bad, bad


def schema_ok(path, kind):
    s = pq.read_schema(path)
    return s.names == E.schema(kind).names and all(s.field(n).type == E.schema(kind).field(n).type for n in s.names)


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- cases
def t_units():
    base = rec(market(1, 0), u7(1000, 1), '1', leg('Bid', 1, 50 * CENT, E18, 't'), [leg('Ask', 1, 50 * CENT, E18, 'm')], 1000)
    vol = rec(market(1, 0, st='CLOSED', boosted=True, ost='WON'), u7(1000, 1), '1', leg('Bid', 1, 50 * CENT, E18, 't', ost='WON', bb=0.5, oc='7'),
              [leg('Ask', 1, 50 * CENT, E18, 'm', ost='LOST', bb=0.4, oc='8')], 1000)
    chg = rec(market(1, 0), u7(1000, 1), '1', leg('Bid', 1, 50 * CENT, E18, 't'), [leg('Ask', 1, 50 * CENT, 2 * E18, 'm')], 1000)
    late = rec(market(1, 0), u7(1000, 1), '1', leg('Bid', 1, 50 * CENT, E18, 't'), [leg('Ask', 1, 50 * CENT, E18, 'm')], 2000)
    newkey = json.loads(json.dumps(base))
    newkey['makers'][0]['remaining'] = '0'
    check('canon: a re-poll differing only in market / outcome.status / bestBid / bestAsk / onChainId has equal content', E.digest(base) == E.digest(vol))
    check('canon: a maker amount change is a content change', E.digest(base) != E.digest(chg))
    check('canon: an executedAt change is a content change (not volatile)', E.digest(base) != E.digest(late))
    check('canon: a new leg key takes the generic form and is a content change', E.canon(newkey)[0] == 'G' and E.digest(base) != E.digest(newkey))
    check('u256: exact int', E.u256('1350000000000000001') == 1350000000000000001)
    for bad in ('1_0', ' 1', '+1', '1.5', 12):
        refused(f'u256 refuses {bad!r}', lambda b=bad: E.u256(b), ValueError, 'not a decimal integer string')
    check('iso_ms: executedAt -> integer ms', E.iso_ms('2026-10-27T12:00:11.000Z') == E.day_ms('20261027') + 43_211_000)
    check('iso_ms: a naive or missing time -> -1', E.iso_ms('2026-10-27T12:00:11') == -1 and E.iso_ms(None) == -1)
    check('UUIDv7 decode', C.uuid7_ms(u7(1793102411123, 5)) == 1793102411123)
    refused('UUIDv4 refused as a settle time', lambda: C.uuid7_ms('01a1ba03-3879-4fe1-88c0-f6f4c395258e'), ValueError, 'is not a UUIDv7')
    for asc in (True, False):                                                            # N4: escapes, backslashes, CRLF, unicode, marker-like text
        recs = [rec(market(7, 0, desc=TRICKY), u7(1000, 9), '9', leg('Bid', 1, 50 * CENT, E18, 't'), [leg('Ask', 1, 50 * CENT, E18, 'm')], 1000),
                rec(market(7, 0, desc=TRICKY + 'x'), u7(2000, 10), '10', leg('Bid', 1, 50 * CENT, E18, 't'), [leg('Ask', 1, 50 * CENT, E18, 'm')], 2000)]
        line = json.dumps(dict(path='/v1/orders/matches', status=200, raw=json.dumps(dict(success=True, data=recs), separators=(',', ':'), ensure_ascii=asc)),
                          separators=(',', ':'), ensure_ascii=asc).encode()
        kr = line.find(E.RAW_KEY)
        cut = E.cut_markets(line, kr)
        got = E.attach_markets(json.loads(json.loads(cut[0])['raw'])['data'], cut[1], {}) if cut else None
        check(f'N4: market text with quotes / backslashes / CRLF / unicode through the cut == a full parse (ensure_ascii={asc})',
              cut is not None and len(cut[1]) == 2 and got == json.loads(json.loads(line)['raw'])['data'] and len(cut[0]) < len(line))
    check('status: fixed order, ok when empty', E.status({'boundary_censored', 'conflict'}) == 'conflict+boundary_censored' and E.status(set()) == 'ok')
    refused('status refuses an unknown flag', lambda: E.status({'conflcit'}), ValueError, 'unknown status flags')


def t_forbidden():
    with Opens() as O:
        refused('Forbidden: a 20261028 REST hour', lambda: E.hour_file('rest', '20261028_00'), C.Forbidden, 'is not on a consumed day')
        n1 = len(O.calls)
        refused('Forbidden: Nov 8 hour 23 (WS)', lambda: E.hour_file('predict_ws', '20261108_23'), C.Forbidden, 'is not on a consumed day')
        n2 = len(O.calls)
        refused('Forbidden: a 20261028 day run', lambda: E.Day('20261028', C.OUT), C.Forbidden, 'is not a consumed day')
        n3 = len(O.calls)
    check('Forbidden: ZERO filesystem calls before each refusal', n1 == n2 == n3 == 0, f'{O.calls[:5]}')


def fill_fixture():
    day = '20261027'
    d0 = E.day_ms(day) // 1000
    write_registry(C.STEP1 / day / 'registry.parquet', day, 5000, override={287: dict(outcome='SEALED_SENTINEL_OUTCOME', outcome_seen_ns=SEALED_NS)})
    mid, ws = (lambda i: 5000 + i), (lambda i: d0 + 300 * i)
    A, B, Y, Z = 144, 72, 286, 287
    E.BOOK_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    E.BOOK_MANIFEST.write_text(json.dumps(dict(complete={day: [mid(A)]})))
    mA, msA, weB = (lambda **k: market(mid(A), ws(A), **k)), ws(A) * 1000, (ws(B) + 300) * 1000
    msY, msZ = ws(Y) * 1000, ws(Z) * 1000

    def r1(v, **mk):
        ost, bb, oc = (None, None, '1') if v == 0 else ('WON', 0.5, '999')
        return rec(mA(**mk), u7(msA + 10_000, 1), '1', leg('Bid', 1, 544 * 10 ** 15, 5 * E18, 't1', fee=45_600_000_000_000_000, ft='COLLATERAL', ost=ost, bb=bb, oc=oc),
                   [leg('Ask', 1, 54 * CENT, 3 * E18, 'm1a', ost=ost, bb=bb, oc=oc), leg('Bid', 2, 45 * CENT, 2 * E18, 'm1b', ost=ost, bb=bb, oc=oc)], msA + 11_000)
    one = lambda n, t_ms, amt=E18, mk=None, th=None, mh=None: rec(mk or mA(), u7(t_ms, n), str(n), leg('Bid', 1, 50 * CENT, amt, th or f't{n}'),
                                                                 [leg('Ask', 1, 50 * CENT, amt, mh or f'm{n}')], t_ms + 1000)
    S = dict(r1=u7(msA + 10_000, 1), r2=u7(msA + 20_000, 2), r3=u7(msA + 30_000, 3), r4=u7(msA + 40_000, 4), r5=u7(msA + 50_000, 5),
             r6='01a1ba03-3879-4fe1-88c0-f6f4c395258e', r7=u7(msA - 100_000, 7), r8=u7(weB + 4000, 8), r9=u7(msZ + 120_000, 9),
             r10=u7(msY + 60_000, 10), r11=u7(msA + 60_000, 11), r12=u7(msA + 70_000, 12), r14=u7(msA - 10_800_000, 14),
             r15=u7(ws(0) * 1000 - 30_000, 15), r16=u7(ws(240) * 1000 + 60_000, 16), r17=u7(msA + 80_000, 17), r18=u7(msA + 90_000, 18))
    r2v1 = rec(mA(), S['r2'], '2', leg('Bid', 1, 50 * CENT, 2 * E18, 't2'), [leg('Ask', 1, 50 * CENT, E18, 'm2a')], msA + 21_000, af=str(2 * E18))   # a maker missing
    r2v2 = rec(mA(), S['r2'], '2', leg('Bid', 1, 50 * CENT, 2 * E18, 't2'), [leg('Ask', 1, 50 * CENT, E18, 'm2a'), leg('Ask', 1, 50 * CENT, E18, 'm2b')], msA + 21_000)
    r3 = rec(mA(), S['r3'], '3', leg('Ask', 1, 60 * CENT, 5 * E18, 't3'), [leg('Bid', 1, 60 * CENT, 5 * E18 - 1, 'm3')], msA + 31_000, af=str(5 * E18))
    r4 = rec(mA(), S['r4'], '4', leg('Bid', 1, 52 * CENT + 1, E18, 't4'), [leg('Ask', 1, 52 * CENT, E18, 'm4')], msA + 41_000, pe=str(52 * CENT))
    r5 = rec(mA(st='RESOLVED', ost='LOST', res=dict(name='Down', indexSet=2)), S['r5'], '5', leg('Bid', 1, 99 * CENT, E18, 't5'), [leg('Bid', 2, 1 * CENT, E18, 'm5')],
             msA + 51_000)                                                                   # its market object differs from the first and the r1 receipts
    r6 = rec(mA(), S['r6'], '6', leg('Ask', 2, 40 * CENT, E18, 't6'), [leg('Bid', 2, 40 * CENT, E18, 'm6')], msA + 61_000)
    r11 = rec(mA(), S['r11'], '11', leg('Bid', 1, 50 * CENT, E18, 't11'), [leg('Ask', 1, 50 * CENT, E18, 'm11')], msA + 61_000, af='1.5')
    other = one(13, msA + 71_000, mk=market(999, ws(A)))
    r17 = rec(mA(), S['r17'], '17', leg('Bid', 1, 555 * 10 ** 15, 2 * E18, 't17'), [leg('Ask', 1, 56 * CENT, E18, 'm17a'), leg('Ask', 1, 55 * CENT, E18, 'm17b')],
              msA + 81_000)                                                                   # makers[] worst price first (YES buyer)
    r18 = rec(mA(), S['r18'], '18', leg('Ask', 1, 495 * 10 ** 15, 2 * E18, 't18'), [leg('Bid', 1, 50 * CENT, E18, 'm18a'), leg('Bid', 1, 49 * CENT, E18, 'm18b')],
              msA + 91_000)                                                                   # makers[] best price first (YES seller)
    P = [rest_row(ns(weB + 134_000), mid(B), [one(8, weB + 4000, mk=market(mid(B), ws(B)))]),                 # lag 130 s: the p99.9 tail
         rest_row(ns(msA - 90_000), mid(A), [one(7, msA - 100_000)]),
         rest_row(ns(msA + 15_000), mid(A), [r1(0)]),
         rest_row(ns(msA + 25_000), mid(A), [r2v1]),                                         # the probe's case: first receipt lacks a maker
         rest_row(ns(msA + 26_000), mid(A), [r2v2]), rest_row(ns(msA + 26_500), mid(A), [r2v2]),    # later receipts reconcile (the 2nd: fast path)
         rest_row(ns(msA + 27_000), mid(A), [r1(1, st='CLOSED')]),
         rest_row(ns(msA + 35_000), mid(A), [r3]), rest_row(ns(msA + 45_000), mid(A), [r4]), rest_row(ns(msA + 55_000), mid(A), [r5]),
         rest_row(ns(msA + 65_000), mid(A), [r6]), rest_row(ns(msA + 65_001), mid(A), [r11]),
         rest_row(ns(msA + 75_000), mid(B), [one(12, msA + 70_000, mh='m7')]),              # order m7 fills again (pre-window r7, live r12)
         rest_row(ns(msA + 76_000), mid(A), [other]),
         rest_row(ns(msA + 77_000), 999, [other]),
         rest_row(ns(msA + 78_000), mid(A), [], status=500, raw='upstream error'),
         rest_row(ns(msA + 3_660_000), mid(A), [r5]),                                        # hour 13: r5 again (parsed, cached for the hour)
         rest_row(ns(msA + 3_700_000), mid(A), [r1(1, st='CLOSED', boosted=True, ost='WON', res=dict(name='Up', indexSet=1))]),
         rest_row(ns(msA + 3_780_000), mid(A), [r5]),                                        # byte-identical payload: fast path, the latest A receipt
         rest_row(ns(msZ + 140_000), mid(Z), [one(9, msZ + 120_000, mk=market(mid(Z), ws(Z)))]),
         rest_row(ns(msY + 65_000), mid(Y), [one(10, msY + 60_000, mk=market(mid(Y), ws(Y)))]),
         rest_row(ns(msY + 62_000), mid(Y), [one(10, msY + 60_000, mk=market(mid(Y), ws(Y), st='CLOSED'))], order=ns(msY + 66_000)),   # parsed: later line, earlier receipt
         rest_row(ns(msA - 60_000), mid(A), [one(14, msA - 10_800_000, th='m4')]),          # 3 h before ws, received at discovery: lag gated; taker hash = a maker hash
         rest_row(ns(ws(0) * 1000 - 25_000), mid(0), [one(15, ws(0) * 1000 - 30_000, mk=market(mid(0), ws(0)))]),   # settles Oct 26 23Z: no WS hour read
         rest_row(ns(ws(240) * 1000 + 65_000), mid(240), [one(16, ws(240) * 1000 + 60_000, mk=market(mid(240), ws(240)))]),   # settles in the missing WS hour
         rest_row(ns(msA + 85_000), mid(A), [r17]), rest_row(ns(msA + 95_000), mid(A), [r18]),
         rest_row(ns(msA + 84_000), mid(A), [r17], order=ns(msA + 86_000)),                  # byte-identical payload (fast path): later line, earlier receipt
         rest_row(ns(msA + 96_000), mid(A), [r18], status=500)]                             # a cached payload under HTTP 500: never a re-poll
    rest_plan = C.day_hours(day, 1, 1)[0]
    write_family('rest', P, rest_plan, missing=(f'{day}_03',))
    T0 = msA + 1000
    los = dict(id='11', kind='LIMIT', marketId=mid(A), outcome='Yes', price='0.54', side='Ask')

    def fr(upd, bids, asks):
        return dict(asks=asks, bids=bids, lastOrderSettled=los, marketId=mid(A), orderCount=4, settlementsPending=dict(asks=[], bids=[[0.54, 1.62]]),
                    updateTimestampMs=upd, version=1)
    b0, a0, b2, b3 = [[0.54, 10.0], [0.53, 5.0]], [[0.55, 5.0], [0.56, 3.0]], [[0.54, 7.0], [0.53, 5.0]], [[0.54, 6.0], [0.53, 5.0]]
    F = [('c1', T0, b0, a0), ('c1', T0, b0, a0), ('c1', T0 + 50, b2, a0), ('c1', T0 + 50, b3, a0), ('c1', T0 + 2050, b2, a0), ('c1', T0 + 1000, b2, a0),
         ('c2', T0 + 2100, b2, a0), ('c2', T0 + 2150, b2, [[0.55, 5.0], [0.557, 1.0]]), ('c2', T0 + 2200, [[0.54, 3.0], [0.54, 2.0]], [[0.55, 5.0]])]
    W = [ws_row(ns(T0 + 50 + 10 * k), c, mid(A), fr(u, b, a)) for k, (c, u, b, a) in enumerate(F)]
    W += [ws_row(ns(T0 + 30), 'c1', 0, dict(x=1), topic='heartbeat'), ws_row(ns(T0 + 31), 'c1', 0, None, msg=dict(type='R', requestId=1, success=True)),
          ws_row(ns(T0 + 40), 'c1', 999, dict(fr(T0, b0, a0), marketId=999)),
          ws_row(ns(ws(B) * 1000 + 600), 'c1', mid(B), dict(asks=[[0.6, 1.0]], bids=[], lastOrderSettled=None, marketId=mid(B), orderCount=1,
                                                             settlementsPending=dict(asks=[], bids=[]), updateTimestampMs=ws(B) * 1000 + 500, version=1))]
    write_family('predict_ws', W, C.day_hours(day)[0], missing=(f'{day}_20',))
    return dict(day=day, mid=mid, A=A, B=B, Y=Y, Z=Z, S=S, msA=msA, msY=msY, weB=weB, T0=T0, ws=ws)


def t_fill():
    X = fill_fixture()
    day, mid, S, msA, T0 = X['day'], X['mid'], X['S'], X['msA'], X['T0']
    with Opens() as O:
        R = E.Day(day, C.OUT).run()
    out = Path(C.OUT)
    clip = C.day_hours(day, 1, 1)[1]
    check('FILL: the clipped REST hour is exactly the Oct 28 00Z hour of day_hours', R['rest']['clipped'] == clip and len(clip) == 1 and clip[0][:8] != day)
    check('FILL: no filesystem call ever names a clipped hour', not any(s in c[1] for c in O.calls for s in clip + R['ws']['clipped']))
    g = sorted(Path(c[1]).name for c in O.calls if c[0] == 'gzip')
    want = sorted([f'{s}.jsonl.gz' for s in C.day_hours(day, 1, 1)[0] if s != f'{day}_03'] + [f'{s}.jsonl.gz' for s in C.day_hours(day)[0] if s != f'{day}_20'])
    check('FILL: gzip opens = the planned hours minus the missing ones', g == want, f'{len(g)} vs {len(want)}')
    check('FILL: missing hours are recorded', R['rest']['missing'] == [f'{day}_03'] and R['ws']['missing'] == [f'{day}_20'])
    hs_ok = all(H['sha256'] == fsha(C.hour_path(H['family'], H['stem'])) and H['bytes'] == C.hour_path(H['family'], H['stem']).stat().st_size
                for H in R['rest']['hours'] + R['ws']['hours'] if not H.get('missing'))
    check('FILL: receipt sha256 + bytes of every hour file read', hs_ok and len(R['rest']['hours']) == 25 and len(R['ws']['hours']) == 24)
    env = R['envelope']
    check('FILL: measured margins (ungated p99.9 lag 128.46 s -> 3 min, p50 would give 1; post 4 s -> 1 min; pre recv 90 s -> 2 min)',
          (env['margin_ms'], env['margin_min'], env['post_settle_ms'], env['pre_recv_ms']) == (180000, 3, 60000, 120000)
          and abs(env['lag_p999_ms'] - 128_460) < 1e-6 and env['pre_settle_max_ms'] == 10_800_000., f'{env}')
    check('FILL: a discovery-gated lag (3 h trade received at discovery) is excluded from the margin and reported',
          env['n_discovery_gated'] == 1 and env['lag_p999_all_ms'] > 10_000_000 and env['receipt_lag_ms']['n'] == 16 and env['receipt_lag_ungated_ms']['n'] == 15, f'{env}')
    tot = {k: sum(H.get(k, 0) for H in R['rest']['hours']) for k in ('pages_other_markets', 'rows_status_500', 'records_other_markets', 'conflict_receipts', 'repolls',
                                                                      'recv_inversions', 'pages_payload_repeat')}
    check('FILL: every page took the market cut (compact payloads)', sum(H.get('market_cut_fallback', 0) for H in R['rest']['hours']) == 0)
    check('FILL: exclusions counted, not dropped (other-market page, HTTP 500, foreign record, conflict, re-poll, inverted and repeated-payload receipts)',
          tot == dict(pages_other_markets=1, rows_status_500=2, records_other_markets=1, conflict_receipts=2, repolls=6, recv_inversions=2, pages_payload_repeat=3), f'{tot}')
    Rr = rows(out / f'records_{day}.parquet', 'sid')
    check('FILL: one row per unique key (17)', len(Rr) == 17 and R['counts']['records'] == 17)
    check('FILL: repeated payloads (fast path) raise n_recv and keep the earliest receipt',
          (Rr[S['r5']]['n_recv'], Rr[S['r17']]['n_recv'], Rr[S['r17']]['first_recv_ns'], Rr[S['r18']]['n_recv']) == (3, 2, ns(msA + 84_000), 1), f"{Rr[S['r5']]['n_recv']} {Rr[S['r17']]}")
    ok, bad = sub(Rr[S['r1']], mid=mid(X['A']), settle_ms=msA + 10_000, exec_ms=msA + 11_000, first_recv_ns=ns(msA + 15_000), n_recv=3, amount_filled=5.0,
                  price_executed=0.544, n_makers=2, tk_quote='Bid', tk_index_set=1, tk_price=0.544, tk_amount=5.0, tk_fee=0.0456, tk_fee_type='COLLATERAL',
                  tk_y_side='bid', status='ok', x_n_variants=1, x_n_recv_other=0, tx='0x' + '1'.rjust(64, '0'))
    check('FILL: de-dup across hours: re-polls with volatile changes -> n_recv 3, first receipt kept', ok, bad)
    ok, bad = sub(Rr[S['r2']], n_recv=2, x_n_variants=2, x_n_recv_other=1, x_kept='reconciles', status='conflict', amount_filled=2.0, n_makers=2,
                  first_recv_ns=ns(msA + 25_000))
    check('FILL: conflict (the probe case): the later reconciling variant is kept, status conflict, every receipt counted, earliest receipt', ok, bad)
    check('FILL: one content per key -> x_kept only', Rr[S['r1']]['x_kept'] == 'only' and Rr[S['r1']]['x_n_variants'] == 1)
    check('FILL: the variant not kept is counted, not written', R['counts']['variants_not_kept'] == dict(records=1, legs=1), f"{R['counts'].get('variants_not_kept')}")
    ok, bad = sub(Rr[S['r3']], status='recon_mk_sum', amount_filled=5.0)
    check('FILL: sum(makers) short by 1 wei (equal as floats) -> recon_mk_sum', ok, bad)
    ok, bad = sub(Rr[S['r4']], status='recon_price', tk_price=0.52, price_executed=0.52)
    check('FILL: taker.price != priceExecuted by 1 wei (equal as floats) -> recon_price', ok, bad)
    RAW = {r_['x_sid_raw']: r_ for r_ in Rr.values()}
    ok, bad = sub(RAW[S['r6']], status='sid_not_v7', settle_ms=-1)
    s6 = RAW[S['r6']]['sid']
    check('FILL: a non-v7 settlementId -> sid_not_v7, settle_ms -1, a synthetic unique sid nosid:<tx>:<digest>, the raw id in x_sid_raw',
          ok and s6.startswith('nosid:0x' + '6'.rjust(64, '0') + ':') and len(s6.rsplit(':', 1)[1]) == 16 and s6 not in (S['r6'],), f'{bad} {s6}')
    check('FILL: the sid column is unique', len({r_['sid'] for r_ in Rr.values()}) == len(Rr) == R['counts']['records'])
    ok, bad = sub(Rr[S['r11']], status='malformed', n_makers=1, amount_filled=math.nan, tk_hash='')
    check('FILL: a non-integer amount -> malformed row (no legs), not a drop', ok, bad)
    check('FILL: page / record market mismatch -> page_mid_mismatch', Rr[S['r12']]['status'] == 'page_mid_mismatch' and Rr[S['r12']]['mid'] == mid(X['A']))
    check('FILL: the 23:55 market -> boundary_censored; 23:50 -> ok', Rr[S['r9']]['status'] == 'boundary_censored' and Rr[S['r10']]['status'] == 'ok')
    ok, bad = sub(Rr[S['r10']], n_recv=2, first_recv_ns=ns(X['msY'] + 62_000))
    check('FILL: first_recv_ns = the earliest receipt of the kept content (a later line with an earlier recv_ns)', ok, bad)
    Lg = rows(out / f'legs_{day}.parquet')
    by = {}
    for r in Lg:
        by.setdefault(r['sid'], []).append(r)
    check('FILL: legs rows = legs of the kept contents (20; none for the malformed record)', len(Lg) == 20 and S['r11'] not in by)
    check('FILL: the kept conflict record carries the missing maker\'s leg', sorted(r['mk_hash'][-3:] for r in by[S['r2']]) == ['m2a', 'm2b']
          and all(r['status'] == 'conflict' for r in by[S['r2']]))
    l0, l1 = sorted(by[S['r1']], key=lambda r: r['mk_idx'])
    ok0, b0 = sub(l0, mk_idx=0, n_makers=2, mk_quote='Ask', mk_index_set=1, mk_price=0.54, shares=3.0, y_side='ask', y_price=0.54, y_ptick=54, x_match='regular',
                  tk_y_side='bid', phase='live', t_rel_s=10.0, first_recv_ns=ns(msA + 15_000), status='ok', mk_fee=0.0, mk_fee_type='SHARES')
    ok1, b1 = sub(l1, mk_idx=1, mk_quote='Bid', mk_index_set=2, mk_price=0.45, shares=2.0, y_side='ask', y_price=0.55, y_ptick=55, x_match='mint')
    check('FILL: legs of a regular + mint record in YES space', ok0 and ok1, f'{b0} {b1}')
    ok, bad = sub(by[S['r5']][0], mk_quote='Bid', mk_index_set=2, mk_price=0.01, y_side='ask', y_price=0.99, y_ptick=99, x_match='mint', tk_y_side='bid', status='ok')
    check('FILL: mint: NO bid 0.01 -> YES ask 0.99', ok, bad)
    ok, bad = sub(by[s6][0], phase='unknown', t_rel_s=math.nan, y_side='ask', y_price=0.6, tk_y_side='bid', status='sid_not_v7', x_match='regular')
    check('FILL: NO-outcome regular match in YES space; no settle time -> phase unknown', ok, bad)
    check('FILL: legs carry the earliest receipt too', by[S['r10']][0]['first_recv_ns'] == ns(X['msY'] + 62_000))
    check('FILL: legs inherit record statuses', by[S['r3']][0]['status'] == 'recon_mk_sum' and by[S['r4']][0]['status'] == 'recon_price'
          and by[S['r2']][0]['status'] == 'conflict' and by[S['r9']][0]['status'] == 'boundary_censored')
    check('FILL: shares are floats only after the integer check (5e18-1 -> 5.0, flagged)', by[S['r3']][0]['shares'] == 5.0)
    check('FILL: pre / post-window phases on the settle clock', (by[S['r7']][0]['phase'], by[S['r7']][0]['t_rel_s'], by[S['r8']][0]['phase'], by[S['r8']][0]['t_rel_s'])
          == ('pre', -100.0, 'post', 304.0))
    Mk = rows(out / f'markets_{day}.parquet', 'mid')
    want_st = {X['A']: 'ok', X['B']: 'ok', X['Y']: 'no_frames', X['Z']: 'boundary_censored+no_frames+outcome_sealed', 42: 'no_records+hour_missing+no_frames',
               35: 'no_records+hour_missing+no_frames', 48: 'no_records+hour_missing+no_frames', 49: 'no_records+no_frames', 0: 'no_frames+fills_outside_ws',
               240: 'no_frames+fills_in_missing_ws_hour', 237: 'no_records+no_frames'}
    got_st = {i: Mk[mid(i)]['status'] for i in want_st}
    check('FILL: market statuses (boundary, missing REST hour, fills outside / in a missing WS hour, no records / frames)', got_st == want_st and len(Mk) == 288, f'{got_st}')
    check('FILL: a registry outcome first seen in a sealed hour is blanked (outcome_sealed) and reaches no output',
          (Mk[mid(X['Z'])]['outcome'], Mk[mid(X['Z'])]['outcome_seen_ns']) == ('', -1) and Mk[mid(X['Y'])]['outcome'] == 'Up'
          and not sentinel_free(out, day, ['SEALED_SENTINEL_OUTCOME', SEALED_NS]), f"{sentinel_free(out, day, ['SEALED_SENTINEL_OUTCOME', SEALED_NS])}")
    ok, bad = orders_ok(R, out, day)
    check('FILL: the receipt declares the status order lists and every output status follows them (N5)', ok and R['status_orders']['markets'] == list(E.MK_FLAGS), bad)
    check('FILL: WS coverage counts per market', (Mk[mid(0)]['n_legs_outside_ws'], Mk[mid(240)]['n_legs_in_missing_ws_hour'], Mk[mid(X['A'])]['n_legs_outside_ws']) == (1, 1, 0))
    ok, bad = sub(Mk[mid(X['A'])], feeRateBps=200, makerFeeBps=0, spreadThreshold=0.06, shareThreshold=100.0, isBoosted=0, changed_fields='isBoosted',
                  rewards='{"current":null,"schedule":[{"endsAt":"e","hourlyRate":3000,"startsAt":"s"}]}', market_status='RESOLVED', resolution='Down',
                  outcome_statuses='[{"indexSet":1,"name":"Up","status":"LOST"},{"indexSet":2,"name":"Down","status":null}]', n_pages=19, n_records=12, n_legs=15,
                  latest_recv_ns=ns(msA + 3_780_000), depth=1, n_frames=9, boostStartsAt='2026-10-27T12:00:00.000Z', category_slug=f'btc-updown-5m-{X["ws"](X["A"])}')
    check('FILL: markets row = the latest receipt, here a byte-identical replay of an older payload (its values re-applied); the changed field named', ok, bad)
    Tb = rows(out / f'tob_{day}.parquet')
    tA = [r for r in Tb if r['mid'] == mid(X['A'])]
    want_tob = ['ok', 'dup', 'ok', 'same_ms_changed', 'gap', 'out_of_order', 'new_conn', 'ok', 'ok']
    check('FILL: tob statuses (dup, same_ms_changed, gap > 1 s, out_of_order, new_conn)', [r['status'] for r in tA] == want_tob, f'{[r["status"] for r in tA]}')
    ok, bad = sub(tA[0], day=day, upd_ms=T0, recv_ns=ns(T0 + 50), conn='c1', order_count=4, n_bid=2, n_ask=2, best_bid=0.54, best_bid_sh=10.0, best_ask=0.55,
                  best_ask_sh=5.0, pend_bid_usd=1.62, pend_ask_usd=0.0)
    tB = [r for r in Tb if r['mid'] == mid(X['B'])]
    okB, badB = sub(tB[0], n_bid=0, n_ask=1, best_bid=math.nan, best_bid_sh=0.0, best_ask=0.6, best_ask_sh=1.0, status='ok') if len(tB) == 1 else (False, tB)
    check('FILL: exact tob rows (two-sided; one-sided with NaN best bid); frames of other markets / non-book lines not in tob',
          ok and okB and len(Tb) == 10 and sum(H.get('frames_other_markets', 0) for H in R['ws']['hours']) == 1
          and sum(H.get('lines_without_book_topic', 0) for H in R['ws']['hours']) == 2, f'{bad} {badB}')
    dA = out / 'depth' / f'{day}_{mid(X["A"])}_frames.parquet'
    Fr = rows(dA)
    check('FILL: depth frames statuses (+ off_grid, dup_level)', [r['status'] for r in Fr] == want_tob[:7] + ['off_grid', 'dup_level'], f'{[r["status"] for r in Fr]}')
    ok, bad = sub(Fr[0], mid=mid(X['A']), fid=0, upd_ms=T0, recv_ns=ns(T0 + 50), conn='c1', order_count=4, pend_bids='[[0.54,1.62]]', pend_asks='[]', los_id='11',
                  los_side='Ask', los_price=0.54, los_outcome='Yes', x_los_kind='LIMIT')
    check('FILL: exact depth frame row (lastOrderSettled kept raw)', ok and [r['fid'] for r in Fr] == list(range(9)), bad)
    Lv = rows(out / 'depth' / f'{day}_{mid(X["A"])}_levels.parquet')
    f0 = [(r['side'], r['ptick'], r['shares'], r['status']) for r in Lv if r['fid'] == 0]
    f7 = [(r['side'], r['ptick'], r['shares'], r['status'], r['x_price']) for r in Lv if r['fid'] == 7 and r['side'] == 'ask']
    check('FILL: levels = full snapshot in integer cents; an off-grid price -> status off_grid with its exact price',
          len(Lv) == 35 and f0 == [('bid', 54, 10.0, 'ok'), ('bid', 53, 5.0, 'ok'), ('ask', 55, 5.0, 'ok'), ('ask', 56, 3.0, 'ok')]
          and f7[0][:4] == ('ask', 55, 5.0, 'ok') and math.isnan(f7[0][4]) and f7[1] == ('ask', 56, 1.0, 'off_grid', 0.557), f'{len(Lv)} {f0} {f7}')
    check('FILL: depth only for BOOK markets', not (out / 'depth' / f'{day}_{mid(X["B"])}_frames.parquet').exists())
    sch = all(schema_ok(out / f'{k}_{day}.parquet', k) for k in ('records', 'legs', 'tob', 'markets')) and schema_ok(dA, 'frames') \
        and schema_ok(out / 'depth' / f'{day}_{mid(X["A"])}_levels.parquet', 'levels')
    base_ok = all(list(C.SCHEMAS[k]) == E.schema(k).names[:len(C.SCHEMAS[k])] and all(n.startswith('x_') for n in E.schema(k).names[len(C.SCHEMAS[k]):])
                  for k in ('records', 'legs', 'tob', 'frames', 'levels'))
    check('FILL: schemas = pf_mt_common.SCHEMAS (+ x_ extras only)', sch and base_ok)
    INV = json.loads((out / f'inventory_{day}.json').read_text())
    cn = INV['counters']
    oh = INV['rest']['ids']['order_hashes']
    ok = (cn['leg_match'] == dict(regular=17, mint=2) and cn['maker_fee'] == dict(legs=19, nonzero=0)
          and cn['makers_yes_price_order (multi-maker records)'] == dict(best_first=2, worst_first=1)
          and cn['priceExecuted_vs_maker_avg'] == dict(exact=16) and cn['executedAt_after_seconds'] == {'.000Z': 17}
          and INV['rest']['ids']['keys_with_conflict'] == 1 and INV['rest']['ids']['unique_keys'] == 17
          and oh == dict(taker_hashes=16, maker_hashes=19, taker_hashes_also_maker=1, maker_hashes_with_fills_in_several_records=1)
          and ['amountFilled', 'executedAt', 'makers', 'market', 'priceExecuted', 'settlementId', 'taker', 'transactionHash'] in [o['keys'] for o in INV['objects']['record']])
    check('FILL: computed inventory (match kinds, maker fees, makers[] order, avg price, sub-second, ids, order hashes, exact keys)', ok,
          f'{cn.get("leg_match")} {cn.get("priceExecuted_vs_maker_avg")} {cn.get("makers_yes_price_order (multi-maker records)")} {oh}')
    rv = json.loads(Mk[mid(X['A'])]['rewards_variants'])
    check('FILL: rewards variants kept with their first receipt', len(rv) == 1 and rv[0]['first_recv_ns'] == ns(msA - 90_000) and rv[0]['rewards']['schedule'][0]['hourlyRate'] == 3000)
    outs = R['outputs']
    check('FILL: receipt hashes the published outputs', all(fsha(out / n) == v['sha256'] for n, v in outs.items()) and (out / f'extract_{day}.json').exists())
    check('FILL: the temp dir is gone', not any(p.name.startswith('.tmp_extract_') for p in out.iterdir()))
    with Opens() as O2:
        refused('write-once: a second run of the day is refused', lambda: E.Day(day, C.OUT).run(), ValueError, 'write-once')
    check('write-once: refused with ZERO raw-file calls', not O2.raw(), f'{O2.raw()[:3]}')
    return X


def t_guard():
    out2, keep = Path(C.OUT).parent / 'out_guard', C.GUARD_MIB
    try:
        C.GUARD_MIB = 1.0
        refused('guard: VmHWM >= GUARD_MIB stops the run', lambda: E.Day('20261027', out2).run(), E.GuardStop, 'VmHWM')
    finally:
        C.GUARD_MIB = keep
    check('guard: nothing published, temp dir removed', out2.exists() and not any(out2.iterdir()))


def t_id_day():
    day = '20261023'
    d0 = E.day_ms(day) // 1000
    write_registry(C.STEP1 / day / 'registry.parquet', day, 7000)
    mk = lambda i: market(7000 + i, d0 + 300 * i)
    one = lambda n, i, t_ms: rec(mk(i), u7(t_ms, n), str(n), leg('Bid', 1, 50 * CENT, E18, f't{n}'), [leg('Ask', 1, 50 * CENT, E18, f'm{n}')], t_ms + 1000)
    P = [rest_row(ns(d0 * 1000 + 15_000), 7000, [one(1, 0, d0 * 1000 + 10_000)]),
         rest_row(ns((d0 + 3600) * 1000 - 30_000), 7012, [one(2, 12, (d0 + 3600) * 1000 - 40_000)]),
         rest_row(ns((d0 + 86100) * 1000 + 35_000), 7287, [one(3, 287, (d0 + 86100) * 1000 + 30_000)])]
    write_family('rest', P, C.day_hours(day, 1, 1)[0])
    with Opens() as O:
        R = E.Day(day, C.OUT).run()
    out = Path(C.OUT)
    Mk = rows(out / f'markets_{day}.parquet', 'mid')
    st = {i: Mk[7000 + i]['status'] for i in (0, 1, 12, 287)}
    check('ID: the 00:00 market is boundary_censored (its receipts could precede the first allowed hour); 23:55 is covered by Oct 24 00Z',
          st == {0: 'boundary_censored', 1: 'no_records', 12: 'ok', 287: 'ok'}, f'{st}')
    check('ID: REST only: no WS file touched, no tob / depth output', not any('predict_ws' in c[1] for c in O.calls) and not (out / f'tob_{day}.parquet').exists()
          and R['ws']['read'] == [] and 'tob' not in R['counts'])
    clip = C.day_hours(day, 1, 1)[1]
    check('ID: the clipped hour is the one before the day and is never touched', R['rest']['clipped'] == clip and len(clip) == 1
          and not any(s in c[1] for c in O.calls for s in clip))
    check('ID: records carry the market boundary status', rows(out / f'records_{day}.parquet', 'sid')[u7(d0 * 1000 + 10_000, 1)]['status'] == 'boundary_censored')


def t_gt_day():
    day = C.GT_DAY
    d0 = E.day_ms(day) // 1000
    W = [(9001, d0 + 43200), (9002, d0 + 82200), (9003, d0 + 82500)]
    E.GT_SCOPE.parent.mkdir(parents=True, exist_ok=True)
    E.GT_SCOPE.write_text(json.dumps(dict(windows=[dict(market=m, ws=w, run='v6') for m, w in W])))
    write_registry(E.GT_REGISTRY, day, 0, rows=[dict(mid=m, slug=f'btc-updown-5m-{w}', ws=float(w), we=float(w + 300), outcome='GT_SEALED_SENTINEL' if m == 9003 else 'Up',
                                                     outcome_seen_ns=(w + 320) * 10 ** 9) for m, w in W]          # 9003: first seen 23:00:20Z (hour 23 not consumed)
                   + [dict(mid=9999, slug='btc-updown-5m-0', ws=float(d0), we=float(d0 + 300), outcome='NONSCOPE_SENTINEL', outcome_seen_ns=(d0 + 320) * 10 ** 9)])
    own_mixed = '0x' + ''.join(c.upper() if i % 3 == 0 else c for i, c in enumerate(E.OWN_SIGNER[2:]))      # mixed case: compared case-insensitively
    sg = {9001: (None, own_mixed), 9002: (None, None), 9003: (E.OWN_SIGNER, None)}                           # (taker signer, maker signer)
    one = lambda n, m, w: rec(market(m, w), u7(w * 1000 + 30_000, n), str(n), leg('Bid', 1, 50 * CENT, E18, f't{n}', sg=sg[m][0]),
                              [leg('Ask', 1, 50 * CENT, E18, f'm{n}', sg=sg[m][1])], w * 1000 + 31_000)
    write_family('rest', [rest_row(ns(w * 1000 + 35_000), m, [one(k, m, w)]) for k, (m, w) in enumerate(W)], C.day_hours(day, 1, 1)[0])
    fr = lambda m, u: dict(asks=[[0.51, 2.0]], bids=[[0.49, 3.0]], lastOrderSettled=None, marketId=m, orderCount=2, settlementsPending=dict(asks=[], bids=[]),
                           updateTimestampMs=u, version=1)
    write_family('predict_ws', [ws_row(ns(W[0][1] * 1000 + 100 + k), 'g1', 9001, fr(9001, W[0][1] * 1000 + k * 50)) for k in range(2)]
                 + [ws_row(ns(W[1][1] * 1000 + 100 + k), 'g1', 9002, fr(9002, W[1][1] * 1000 + k * 50)) for k in range(3)], C.day_hours(day)[0])
    with Opens() as O:
        R = E.Day(day, C.OUT).run()
    out = Path(C.OUT)
    Mk = rows(out / f'markets_{day}.parquet', 'mid')
    st = {m: Mk[m]['status'] for m, _ in W}
    check('GT: every scope window in markets; the WS rule skips windows without an own maker leg (an own taker leg does not count); 22:55 boundary_censored',
          st == {9001: 'ok', 9002: 'no_own_fill_ws_skipped', 9003: 'boundary_censored+no_own_fill_ws_skipped+outcome_sealed'} and len(Mk) == 3, f'{st}')
    check('GT: the registry is read for the scope rows only; an outcome seen in hour 23 is blanked; neither value reaches an output',
          (Mk[9003]['outcome'], Mk[9003]['outcome_seen_ns'], Mk[9001]['outcome']) == ('', -1, 'Up') and not sentinel_free(out, day, ['GT_SEALED_SENTINEL', 'NONSCOPE_SENTINEL']))
    sel = R['ws_population']
    check('GT: the rule, our signer and the selected / excluded counts are in the receipt',
          (sel['signer'], sel['n_selected'], sel['n_excluded'], sel['selected'], sel['excluded']) == (E.OWN_SIGNER, 1, 2, [9001], [9002, 9003])
          and 'truth table is never read' in sel['rule'], f'{sel}')
    Tb = rows(out / f'tob_{day}.parquet')
    check('GT: tob only for the selected window; the skipped window\'s frames counted, not parsed',
          sorted({r_['mid'] for r_ in Tb}) == [9001] and len(Tb) == 2 and sum(H.get('frames_skipped_no_own_fill', 0) for H in R['ws']['hours']) == 3
          and (Mk[9002]['n_frames'], Mk[9002]['depth'], Mk[9002]['n_legs_outside_ws'], Mk[9001]['depth']) == (0, 0, -1, 1))
    check('GT: records and legs for every scope window', len(rows(out / f'records_{day}.parquet')) == 3 and len(rows(out / f'legs_{day}.parquet')) == 3)
    clip = R['rest']['clipped'] + R['ws']['clipped']
    check('GT: no clipped hour (07Z23, 08Z23, 09Z00) is ever touched', len(R['rest']['clipped']) == 3 and len(R['ws']['clipped']) == 1
          and not any(s in c[1] for c in O.calls for s in clip))
    check('GT: depth only for the selected window', len(rows(out / 'depth' / f'{day}_9001_frames.parquet')) == 2
          and all(schema_ok(out / 'depth' / f'{day}_9001_{k}.parquet', k) for k in ('frames', 'levels'))
          and not any((out / 'depth' / f'{day}_{m}_{k}.parquet').exists() for m in (9002, 9003) for k in ('frames', 'levels')))


def t_refusals():
    pr = str(Path(C.OUT).parent / 'probe')
    refused('probe: output under OUT refused', lambda: E.main(['x', 'probe', str(Path(C.OUT) / 'p'), 'rest:20261024_12']), ValueError, 'may not lie under OUT')
    refused('probe: more than 2 hour files refused', lambda: E.main(['x', 'probe', pr, 'rest:20261024_12', 'ws:20261024_12', 'rest:20261024_13']),
            ValueError, 'at most 2 hour files')
    refused('probe: another day refused', lambda: E.main(['x', 'probe', pr, 'rest:20261025_12']), ValueError, 'must be of 20261024')
    refused('probe: unknown hour family refused', lambda: E.main(['x', 'probe', pr, 'book:20261024_12']), ValueError, 'rest:<stem> or ws:<stem>')
    refused('day: a padded stem list is refused for a day run', lambda: E.Day('20261024', C.OUT, rest_stems=['20261024_12']), ValueError, 'whole plan')
    write_registry(C.STEP1 / '20261025' / 'registry.parquet', '20261025', 8000, n=287)
    with Opens() as O:
        refused('registry: 287 windows refused (a missing window would drop its records)', lambda: E.Day('20261025', C.OUT), ValueError, 'not exactly the 288')
    check('registry refusal: ZERO raw-file calls', not O.raw())


def t_conflict():
    """Amendment 1 conflict rule on a second FILL day (20261026): most receipts, tie -> first seen, none reconciles -> first, a reconciling
    first beats a non-reconciling variant with more receipts; market-cut fallbacks (spaced JSON, a record whose market is not followed by taker)."""
    day = '20261026'
    d0 = E.day_ms(day) // 1000
    write_registry(C.STEP1 / day / 'registry.parquet', day, 6000)
    E.BOOK_MANIFEST.write_text(json.dumps(dict(complete={'20261027': [5144], day: []})))
    m0, msC = 6144, (d0 + 43200) * 1000
    mC = lambda: market(m0, d0 + 43200)
    def r(n, ems, amts=(E18,), tk=E18, mh=None):
        return rec(mC(), u7(msC + 10_000 * n, 30 + n), str(30 + n), leg('Bid', 1, 50 * CENT, tk, f'tc{n}'),
                   [leg('Ask', 1, 50 * CENT, a, mh or f'mc{n}{i}') for i, a in enumerate(amts)], ems, af=str(tk))
    c1a, c1b = r(1, msC + 11_000), r(1, msC + 12_000)                                    # both reconcile; c1b has more receipts
    c2a, c2b = r(2, msC + 21_000), r(2, msC + 22_000)                                    # both reconcile; one receipt each
    c3a, c3b = r(3, msC + 31_000, (E18 - 1,), mh='mc3a'), r(3, msC + 31_000, (E18 - 2,), mh='mc3b')   # none reconciles
    c4a, c4b = r(4, msC + 41_000), r(4, msC + 41_000, (E18 - 1,), mh='mc4b')            # the first reconciles, the other has 3 receipts
    c5 = r(5, msC + 51_000)                                                              # a spaced payload: no marker, parsed whole
    c6 = dict(market=mC(), amountFilled=str(E18), taker=leg('Bid', 1, 50 * CENT, E18, 'tc6'), priceExecuted=str(50 * CENT),
              makers=[leg('Ask', 1, 50 * CENT, E18, 'mc6')], transactionHash='0x' + '6' * 64, settlementId=u7(msC + 60_000, 36), executedAt=iso(msC + 61_000))
    c7 = dict(taker=leg('Bid', 1, 50 * CENT, E18, 'tc7'), market=mC(), amountFilled=str(E18), priceExecuted=str(50 * CENT),
              makers=[leg('Ask', 1, 50 * CENT, E18, 'mc7')], transactionHash='0x' + '7' * 64, settlementId=u7(msC + 70_000, 37), executedAt=iso(msC + 71_000))
    c8 = rec(market(6143, d0 + 42900, up_ix=2, desc=TRICKY), u7(msC - 60_000, 38), '38', leg('Bid', 1, 50 * CENT, E18, 'tc8'),
             [leg('Ask', 1, 50 * CENT, E18, 'mc8')], msC - 59_000)                         # N1: Up is indexSet 2 here; N4: tricky market text (in-window trade)
    c9, c10 = r(9, msC + 91_000), r(10, msC + 101_000)                                    # N2: no settlementId at all
    del c9['settlementId'], c10['settlementId']
    P = [rest_row(ns(msC + 15_000), m0, [c1a]), rest_row(ns(msC + 16_000), m0, [c1b]), rest_row(ns(msC + 17_000), m0, [c1b]),
         rest_row(ns(msC + 25_000), m0, [c2a]), rest_row(ns(msC + 26_000), m0, [c2b]),
         rest_row(ns(msC + 35_000), m0, [c3a]), rest_row(ns(msC + 36_000), m0, [c3b]),
         rest_row(ns(msC + 45_000), m0, [c4a])] + [rest_row(ns(msC + 46_000 + k), m0, [c4b]) for k in range(3)] + [
         rest_row(ns(msC + 55_000), m0, [c5], spaced=True), rest_row(ns(msC + 65_000), m0, [c6]), rest_row(ns(msC + 66_000), 6143, [c8]),
         rest_row(ns(msC + 67_000), m0, [c9]), rest_row(ns(msC + 68_000), m0, [c9]), rest_row(ns(msC + 69_000), m0, [c10]),
         rest_row(ns(msC + 75_000), m0, [c7])]                                              # c7: last line, no newline
    write_family('rest', P, [f'{day}_12'], no_final_newline=True)
    write_family('events', [ev_row(ns(msC + 20_000), 'MATCH_POLL_INCOMPLETE', market_id=m0, reason='HTTP_UNAVAILABLE'),
                            ev_row(ns(msC + 30_000), 'MATCH_POLL_INCOMPLETE', market_id=m0, reason='BATCH_TIME_LIMIT', resume_queued=True, pages_completed=3),
                            ev_row(ns(msC + 31_000), 'MATCH_POLL_INCOMPLETE', market_id=6100, reason='HTTP_UNAVAILABLE'),        # outside 6100's envelope
                            ev_row(ns(msC + 32_000), 'MATCH_RANGE_TRAVERSED', market_id=m0, pages=2, cutoff=0, newest=0, completeness_certified=False)],
                 [f'{day}_12'])
    R = E.Day(day, C.OUT).run()
    Rr = rows(Path(C.OUT) / f'records_{day}.parquet', 'sid')
    Lg = rows(Path(C.OUT) / f'legs_{day}.parquet')
    got = {n: (Rr[u7(msC + 10_000 * n, 30 + n)]['x_kept'], Rr[u7(msC + 10_000 * n, 30 + n)]['n_recv'], Rr[u7(msC + 10_000 * n, 30 + n)]['x_n_recv_other'],
               Rr[u7(msC + 10_000 * n, 30 + n)]['exec_ms'], Rr[u7(msC + 10_000 * n, 30 + n)]['status']) for n in (1, 2, 3, 4)}
    want = {1: ('most_receipts', 2, 1, msC + 12_000, 'conflict'), 2: ('reconciles_tie_first_seen', 1, 1, msC + 21_000, 'conflict'),
            3: ('first_none_reconciles', 1, 1, msC + 31_000, 'conflict+recon_mk_sum'), 4: ('reconciles', 1, 3, msC + 41_000, 'conflict')}
    check('CONFLICT: most receipts / tie -> first seen / none -> first / a reconciling first beats more receipts', got == want, f'{got}')
    kept_h = {r_['sid']: r_['mk_hash'][-4:] for r_ in Lg}
    check('CONFLICT: the kept variant\'s legs only', kept_h[u7(msC + 30_000, 33)] == 'mc3a' and kept_h[u7(msC + 40_000, 34)] == 'mc40' and len(Lg) == 10,
          f'{kept_h}')
    fb = sum(H.get('market_cut_fallback', 0) for H in R['rest']['hours'])
    ok5, b5 = sub(Rr[u7(msC + 50_000, 35)], status='ok', amount_filled=1.0, n_recv=1)
    ok6, b6 = sub(Rr[u7(msC + 60_000, 36)], status='ok', amount_filled=1.0, n_recv=1, tx='0x' + '6' * 64)
    ok7, b7 = sub(Rr.get(u7(msC + 70_000, 37), {}), status='ok', amount_filled=1.0, n_recv=1, mid=m0, tx='0x' + '7' * 64)
    check('CONFLICT: payloads the market cut cannot validate are parsed whole (spaced JSON; market not followed by taker; taker first)',
          fb == 3 and ok5 and ok6 and ok7, f'{fb} {b5} {b6} {b7}')
    check('CONFLICT: the last line of a file without a final newline is read', u7(msC + 70_000, 37) in Rr)
    check('CONFLICT: receipt counts', sum(H.get('new_variants', 0) for H in R['rest']['hours']) == 4 and R['counts']['variants_not_kept'] == dict(records=4, legs=4)
          and INV_x_kept(day) == {'only': 6, 'most_receipts': 1, 'reconciles_tie_first_seen': 1, 'first_none_reconciles': 1, 'reconciles': 1},
          f"{R['counts'].get('variants_not_kept')} {INV_x_kept(day)}")
    ns_ = sorted((r_ for r_ in Rr.values() if r_['x_sid_raw'] == ''), key=lambda r_: r_['tx'])
    check('N2: records without a settlementId get unique synthetic sids nosid:<tx>:<digest> (re-polls still de-duplicated); legs carry them',
          [(r_['sid'].startswith(f"nosid:{r_['tx']}:"), r_['status'], r_['n_recv']) for r_ in ns_] == [(True, 'no_sid', 2), (True, 'no_sid', 1)]
          and len({r_['sid'] for r_ in Rr.values()}) == len(Rr) == 10 and {r_['sid'] for r_ in ns_} <= {l_['sid'] for l_ in Lg}, f'{ns_}')
    Mk = rows(Path(C.OUT) / f'markets_{day}.parquet', 'mid')
    check('N1: a market object whose Up outcome is not indexSet 1 -> up_index_mismatch (counted)', 'up_index_mismatch' in Mk[6143]['status'].split('+')
          and 'up_index_mismatch' not in Mk[m0]['status'].split('+') and R['markets_flagged']['up_index_mismatch'] == [6143] and u7(msC - 60_000, 38) in Rr)
    check('N3: MATCH_POLL_INCOMPLETE events inside a market envelope -> poll_incomplete with per-reason counts; outside -> none',
          'poll_incomplete' in Mk[m0]['status'].split('+') and (Mk[m0]['n_poll_incomplete'], Mk[m0]['poll_incomplete_reasons']) == (2, '{"BATCH_TIME_LIMIT":1,"HTTP_UNAVAILABLE":1}')
          and (Mk[6100]['n_poll_incomplete'], 'poll_incomplete' in Mk[6100]['status']) == (0, False) and R['events']['population_markets_flagged'] == 1
          and [H['kinds'] for H in R['events']['hours'] if not H.get('missing')] == [{'MATCH_POLL_INCOMPLETE': 3, 'MATCH_RANGE_TRAVERSED': 1}]
          and len(R['events']['missing']) == 25, f"{Mk[m0]['status']} {Mk[m0]['n_poll_incomplete']} {Mk[m0]['poll_incomplete_reasons']} {R['events']['read']}")


def INV_x_kept(day):
    return json.loads((Path(C.OUT) / f'inventory_{day}.json').read_text())['rest']['ids']['x_kept']


def main():
    tmp = Path(tempfile.mkdtemp(prefix='pf_mt_extract_selftest_'))
    saved = (C.PF_RUN, C.OUT, C.STEP1, E.BOOK_MANIFEST, E.GT_SCOPE, E.GT_REGISTRY, E.log)
    try:
        C.PF_RUN, C.OUT, C.STEP1 = tmp / 'run', tmp / 'out', tmp / 'step1'
        E.BOOK_MANIFEST, E.GT_SCOPE = tmp / 'manifest.json', tmp / 'gt' / 'scope.json'
        E.GT_REGISTRY = tmp / 'gt' / 'research' / 'step1' / C.GT_DAY / 'registry.parquet'
        E.log = lambda m: None
        for t in (t_units, t_forbidden, t_fill, t_guard, t_id_day, t_gt_day, t_refusals, t_conflict):
            try:
                t()
            except Exception as e:  # noqa: BLE001 - a crashing case is a failure, never a skip
                check(f'{t.__name__} ran to completion', False, f'{type(e).__name__}: {e}')
    finally:
        C.PF_RUN, C.OUT, C.STEP1, E.BOOK_MANIFEST, E.GT_SCOPE, E.GT_REGISTRY, E.log = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'SELFTEST {"PASS" if not FAILS else "FAIL"}: {NCASE[0] - len(FAILS)}/{NCASE[0]} checks passed' + (f'; failed: {FAILS}' if FAILS else ''), flush=True)
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
