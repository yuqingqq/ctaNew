"""Selftest of pf_mt_btc (TOP-3 DECISION PATHS inputs; contract predictfun/reviews/maker_trace_20261122/decision_contract.md, section
'Information state at a decision'). Synthetic fixtures only, in a temp dir (bookTicker hour files, a Predict reference relay run, collector
ledgers, run roots); no real file is read. Filesystem calls are recorded through monkeypatched open / io.open / gzip.open / os.open / os.stat /
os.lstat / os.path.exists / isdir / isfile / os.listdir / os.scandir / os.mkdir plus a sys audit hook (open, listing, link, rename, remove).
Cases: the allow-list (each measure day's 25-stem plan; a next-day hour and day, other dates, malformed stems, an unknown family and a probe
hour of another day refused with ZERO filesystem calls); day A (20261024): the full 864,000-point grid read back from the parquet equals an
independent two-pointer reference (g_ms / bid / ask / mid / n_upd / age_ms / status), hand-checked points (pre-day state at D0, an equal-E_ms
tie, n_upd 4, the 1,000 ms stale edge, out-of-order rows inside a file and across files, a missing hour, the day tail), Arrow skip / Python
fallback / unterminated tail / invalid price / crossed counts, the exact files opened (no listing of a data dir), receipt sha256 + bytes,
exact lag quantiles; the relay grid (duplicates, a regressed arrival, a missing second, a conflict, two payloads of one price, a non-PRICE row,
a json fallback hour, a missing relay hour, the tail) against an independent reference; day B (20261025): a no_data head with the previous
day's hour missing, unreadable and empty hours, a ledger restart hole, a row after the last grid point, relay skipped and said so; day C
(20261027): next-day trap files in both families are never touched; write-once refusals (rerun, partial outputs, missing root) with zero data
reads; falsifiers: parser parity (bookTicker and relay), the VmHWM guard, era (a pre-boundary stamp, a schema change in the span, a torn ledger
entry that could lie in the span, a bad schema in force, unsorted starts), relay identity; probe mode and its refusals.
Usage (run_light.sh): pf_mt_btc_selftest.py
"""
import builtins
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_mt_btc as M  # noqa: E402

C = M.C
FAILS, NCASE = [], [0]
HOUR_MS, DAY_MS, DAY_S, N = M.HOUR_MS, M.DAY_MS, M.DAY_S, M.N_GRID
NEXT_DAY = (dt.date(2026, 10, 27) + dt.timedelta(days=1)).strftime('%Y%m%d')    # the forbidden next day: built, never typed, never printed
NAN = float('nan')
FX = {}


def check(name, ok, detail=''):
    NCASE[0] += 1
    detail = str(detail).replace(NEXT_DAY, '<next-day>')
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  :: {detail[:400]}'), flush=True)
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


# ---------------------------------------------------------------- filesystem call recording
AUD = dict(on=False, ev=[])
FS_EVENTS = frozenset(('open', 'os.listdir', 'os.scandir', 'os.mkdir', 'os.link', 'os.rename', 'os.remove', 'os.rmdir', 'shutil.rmtree',
                       'glob.glob', 'os.symlink', 'os.truncate', 'shutil.copyfile', 'os.chdir'))


def _audit(ev, args):
    if AUD['on'] and ev in FS_EVENTS:
        AUD['ev'].append((ev, str(args[0]) if args else ''))


sys.addaudithook(_audit)


class Opens:
    NAMES = (('gzip.open', gzip, 'open'), ('open', builtins, 'open'), ('io.open', io, 'open'), ('os.open', os, 'open'), ('stat', os, 'stat'),
             ('lstat', os, 'lstat'), ('exists', os.path, 'exists'), ('isdir', os.path, 'isdir'), ('isfile', os.path, 'isfile'),
             ('listdir', os, 'listdir'), ('scandir', os, 'scandir'), ('mkdir', os, 'mkdir'))

    def __enter__(self):
        self.calls, self.saved = [], [(mod, attr, getattr(mod, attr)) for _, mod, attr in self.NAMES]
        for (tag, mod, attr), (_, _, fn) in zip(self.NAMES, self.saved):
            setattr(mod, attr, self._wrap(tag, fn))
        AUD['ev'], AUD['on'] = [], True
        return self

    def _wrap(self, tag, fn):
        calls = self.calls

        def w(p='.', *a, **k):
            calls.append((tag, str(os.fspath(p)) if isinstance(p, (str, bytes, os.PathLike)) else str(p)))
            return fn(p, *a, **k)
        return w

    def __exit__(self, *a):
        AUD['on'] = False
        for mod, attr, fn in self.saved:
            setattr(mod, attr, fn)

    def all(self):
        return self.calls + AUD['ev']

    def under(self, *dirs):
        ds = [str(d) for d in dirs]
        return [c for c in self.all() if any(c[1].startswith(d) for d in ds)]


# ---------------------------------------------------------------- synthetic builders
UID, SEQ = [11_000_000_000_000], [0]
FEED, RUN_ID, V2 = '0xfeedface', 'synthetic_relay', M.HF_WS_V2_SCHEMA


def px(c):
    return f'{c // 100}.{c % 100:02d}'


def bn(E, c, lag=70, ask_c=None, recv=None):
    """A bookTicker line at E_ms (bid c cents, ask c + 10 cents) -> (line, (recv_ns, E_ms, update_id, bid, ask))."""
    UID[0] += 37
    r = (E + lag) * 10 ** 6 + 123_457 if recv is None else recv
    b, a = px(c), px(c + 10 if ask_c is None else ask_c)
    return f'{r},{E},{E - 1},{UID[0]},{b},1.234,{a},5.678', (r, E, UID[0], float(b), float(a))


def write_gz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = gzip.compress(data)
    path.write_bytes(raw)
    return raw


def write_bn(d, stem, items, tail=b''):
    lines, rows = [], []
    for it in items:
        if isinstance(it, tuple):
            lines.append(it[0])
            rows.append(it[1])
        else:
            lines.append(it)
    return write_gz(Path(d) / f'{stem}.csv.gz', ('\n'.join(lines) + '\n').encode() + tail), rows


def rrow(P, price, recv_ms, rts=None, dup=False, reg=False, kind='PRICE', feed=FEED):
    SEQ[0] += 1
    d = dict(run_id=RUN_ID, seq=SEQ[0], source='reference_prices', kind=kind, recv_ns=recv_ms * 10 ** 6 + 4321, recv_mono_ns=SEQ[0])
    if kind == 'PRICE':
        d.update(provider_feed_id=feed, duplicate_previous_payload=dup, publish_time_regressed=reg, price=price, publish_time_s=P,
                 relay_timestamp_s=P if rts is None else rts, relay_feed_id='1', reference_source='UNSIGNED_PREDICT_CHAINLINK_RELAY',
                 source_time_precision='SECONDS')
    return json.dumps(d, separators=(',', ':')), ((P, d['recv_ns'], price, d['relay_timestamp_s']) if kind == 'PRICE' else None)


def write_ref(run_dir, stem, items, tail=b''):
    lines, pubs = [], []
    for it in items:
        if isinstance(it, tuple):
            lines.append(it[0])
            if it[1] is not None:
                pubs.append(it[1])
        else:
            lines.append(it)
    return write_gz(Path(run_dir) / 'reference_prices' / f'{stem}.jsonl.gz', ('\n'.join(lines) + '\n').encode() + tail), pubs


def write_manifest(run_dir):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / 'manifest.json').write_text(json.dumps(dict(schema='predict_reference_relay_v1', run_id=RUN_ID, topic='chainlinkAssetPriceUpdate/1',
                                                                 provider_feed_id=FEED, symbol='BTCUSDT', source='UNSIGNED_PREDICT_CHAINLINK_RELAY')))


def lentry(start, schema=V2):
    return json.dumps({'collector_schema_version': schema, 'pid': 1, 'stamp_point': 'IMMEDIATELY_AFTER_WS_RECV_BEFORE_JSON_PARSE',
                       'started_at_ns': start, 'symbols': ['BTCUSDT']})


def base_ledger():
    e1 = M.day_ms('20261024') * 10 ** 6 - 7 * 86_400 * 10 ** 9
    return [lentry(M.HF_WS_V2_NS), lentry(e1)[:60] + lentry(e1)]   # a torn fragment then a whole entry on one line, as in the real ledger


def write_ledger(path, lines):
    path.write_text('\n'.join(lines) + '\n')
    return path


def hour_ms(stem):
    return M.day_ms(stem[:8]) + int(stem[9:]) * HOUR_MS


# ---------------------------------------------------------------- independent references
def brute_btc(D0, rows, gaps):
    """Two-pointer sweep over rows sorted by (E_ms, file, line); gaps: half-open [lo, hi) from the fixture's own design (None = unbounded)."""
    rs = sorted(rows)
    n, i, last = len(rs), 0, None
    bid, ask, nup, age = [], [], [], []
    for k in range(N):
        g = D0 + 100 * k
        c = 0
        while i < n and rs[i][0] <= g:
            if rs[i][0] > g - 100:
                c += 1
            last = rs[i]
            i += 1
        nup.append(c)
        if last is None:
            bid.append(NAN)
            ask.append(NAN)
            age.append(-1)
        else:
            bid.append(last[3])
            ask.append(last[4])
            age.append(g - last[0])
    bid, ask, nup, age = np.array(bid), np.array(ask), np.array(nup, np.int64), np.array(age, np.int64)
    gg = D0 + 100 * np.arange(N, dtype=np.int64)
    gm = np.zeros(N, bool)
    for lo, hi in gaps:
        gm |= (gg >= (gg[0] if lo is None else lo)) & (gg < (gg[-1] + 1 if hi is None else hi))
    code = np.where(age < 0, 3, np.where(gm, 2, np.where(age > 1000, 1, 0)))
    return gg, bid, ask, (bid + ask) / 2, nup, age, code


def brute_ref(D0s, pubs, gaps):
    """Per second: the publishes of that second (by receipt), else the latest earlier one; gaps: exclusive (lo, hi)."""
    by = {}
    for P, r, s, rts in pubs:
        by.setdefault(P, []).append((r, s, rts))
    secs, j = sorted(by), -1
    cols = [[] for _ in range(7)]
    for i in range(DAY_S):
        t = D0s + i
        while j + 1 < len(secs) and secs[j + 1] <= t:
            j += 1
        if j < 0:
            v = (NAN, None, -1, 0, 0, -1, 4)
        else:
            P = secs[j]
            rows = sorted(by[P])
            if t == P:
                v = (float(rows[-1][1]), rows[-1][1], 0, len({(rts, float(s)) for _, s, rts in rows}), len(rows), rows[0][0],
                     1 if len({float(s) for _, s, _ in rows}) > 1 else 0)
            else:
                v = (float(rows[-1][1]), rows[-1][1], t - P, 0, 0, -1, 2)
            if any((lo is None or t > lo) and (hi is None or t < hi) for lo, hi in gaps):
                v = v[:6] + (3,)
        for c, x in zip(cols, v):
            c.append(x)
    return cols


def cmp_btc(name, path, exp):
    t = pq.read_table(path)
    gg, bid, ask, mid, nup, age, code = exp
    okc = t.schema.names == M.BTC_SCHEMA.names and all(t.schema.field(f).type == M.BTC_SCHEMA.field(f).type for f in M.BTC_SCHEMA.names)
    check(f'{name}: btc parquet has the declared columns / types and 864,000 rows', okc and t.num_rows == N, f'{t.schema} {t.num_rows}')
    st = pc.index_in(t.column('status'), value_set=pa.array(M.STATUS))
    got = {k: t.column(k).to_numpy() for k in ('g_ms', 'bid', 'ask', 'mid', 'n_upd', 'age_ms')}
    check(f'{name}: g_ms = D0 + 100 k', np.array_equal(got['g_ms'], gg))
    for k, e in (('bid', bid), ('ask', ask), ('mid', mid)):
        bad = np.flatnonzero(~((got[k] == e) | (np.isnan(got[k]) & np.isnan(e))))
        check(f'{name}: {k} equals the reference at every grid point', not len(bad), f'{len(bad)} differ, first k={bad[:3]}')
    for k, e in (('n_upd', nup), ('age_ms', age)):
        bad = np.flatnonzero(got[k] != e)
        check(f'{name}: {k} equals the reference at every grid point', not len(bad), f'{len(bad)} differ, first k={bad[:3]} got {got[k][bad[:3]]} exp {e[bad[:3]]}')
    sc = st.to_numpy(zero_copy_only=False) if st.null_count == 0 else None
    bad = np.flatnonzero(sc != code) if sc is not None else []
    check(f'{name}: status equals the reference at every grid point', sc is not None and not len(bad),
          'unknown status' if sc is None else f'{len(bad)} differ, first k={bad[:3]} got {sc[bad[:3]]} exp {code[bad[:3]]}')
    return t


def at(t, D0, g):
    k = (g - D0) // 100
    return tuple(t.column(c)[k].as_py() for c in ('bid', 'ask', 'mid', 'n_upd', 'age_ms', 'status'))


def want(row, n_upd, age, status):
    b, a = row[3], row[4]
    return (b, a, (b + a) / 2, n_upd, age, status)


def cmp_ref(name, path, D0s, pubs, gaps):
    t = pq.read_table(path)
    okc = t.schema.names == M.REF_OUT_SCHEMA.names and all(t.schema.field(f).type == M.REF_OUT_SCHEMA.field(f).type for f in M.REF_OUT_SCHEMA.names)
    check(f'{name}: ref parquet has the declared columns / types and 86,400 rows', okc and t.num_rows == DAY_S, f'{t.schema} {t.num_rows}')
    price, pstr, age, nobs, nrecv, first, code = brute_ref(D0s, pubs, gaps)
    check(f'{name}: ref t_s = every second of the day', np.array_equal(t.column('t_s').to_numpy(), D0s + np.arange(DAY_S)))
    gp = t.column('price').to_numpy()
    check(f'{name}: ref price equals the reference', np.array_equal(gp, np.array(price), equal_nan=True))
    check(f'{name}: ref price_str equals the reference (null when no_data)', t.column('price_str').to_pylist() == pstr)
    for k, e in (('age_s', age), ('n_obs', nobs), ('n_recv', nrecv), ('first_recv_ns', first)):
        g = t.column(k).to_numpy()
        bad = np.flatnonzero(g != np.array(e, np.int64))
        check(f'{name}: ref {k} equals the reference', not len(bad), f'{len(bad)} differ, first i={bad[:3]}')
    st = pc.index_in(t.column('status'), value_set=pa.array(M.REF_STATUS))
    check(f'{name}: ref status equals the reference', st.null_count == 0 and np.array_equal(st.to_numpy(zero_copy_only=False), np.array(code)))
    return t


def rat(t, D0s, s):
    i = s - D0s
    return tuple(t.column(c)[i].as_py() for c in ('price_str', 'age_s', 'n_obs', 'n_recv', 'status'))


# ---------------------------------------------------------------- fixtures
def fixture_a(tmp):
    day = '20261024'
    D0 = M.day_ms(day)
    D1, D0s = D0 + DAY_MS, D0 // 1000
    D1s = D1 // 1000
    H, Hs = (lambda h: D0 + h * HOUR_MS), (lambda h: D0s + h * 3600)
    plan = M.day_plan(day)
    bnd, refd, out_root = tmp / 'A_bn', tmp / 'A_ref', tmp / 'A_out'
    out_root.mkdir()
    cc = iter(range(8_100_000, 8_900_000, 100))
    nx = lambda: next(cc)  # noqa: E731
    F, tails = {}, {'20261024_07': b'17928'}
    F['20261023_23'] = [bn(D0 - 3_590_000, nx()), bn(D0 - 4_000, nx()), bn(D0 - 120, nx())]
    F['20261024_00'] = [bn(D0 - 60, nx()), bn(D0 + 100, nx()), bn(D0 + 150, nx()), bn(D0 + 199, nx()), bn(D0 + 200, nx()), bn(D0 + 200, nx()),
                        f'{(D0 + 300) * 10 ** 6},{D0 + 300}',                      # 2 fields: Arrow skips it
                        bn(D0 + 1500, nx()), bn(D0 + 2050, nx()), bn(D0 + 2010, nx()), bn(D0 + 3_599_000, nx())]
    F['20261024_01'] = [bn(D0 + 3_598_950, nx()), bn(D0 + 3_600_050, nx()), bn(D0 + 5_000_000, nx()), bn(H(2) - 1_000, nx())]
    for h in range(2, 23):
        s = f'{day}_{h:02d}'
        if h == 5:
            continue                                                               # a missing hour
        if h == 6:                                                                 # a header line: Arrow refuses -> Python fallback
            F[s] = ['recv_ns,E,T,u,b,B,a,A', bn(H(6) - 3_000, nx()), bn(H(6) + 2_000_000, nx()), bn(H(6) + 3_599_500, nx())]
            continue
        rows = [bn(H(h) + 1_000_000, nx())]
        if h == 8:
            rows.append(bn(H(8) + 2_000_000, 0)[0])                                # bid 0.00: invalid price (a raw line, not a valid row)
        if h == 9:
            c = nx()
            rows += [bn(H(9) + 2_000_000, c, ask_c=c - 10), bn(H(9) + 2_500_000, nx(), lag=1_500)]   # crossed; lag 1.5 s
        if h == 10:
            rows.append(bn(H(10) + 2_000_000, nx(), lag=-5))                       # a negative receipt lag
        rows.append(bn(H(h) + 3_599_500, nx()))
        F[s] = rows
    F['20261024_23'] = [bn(H(23) + 1_000_000, nx()), bn(D1 - 2_345, nx())]
    raws, valid, rows = {}, {}, []
    for k, s in enumerate(plan):
        if s in F:
            raws[s], valid[s] = write_bn(bnd, s, F[s], tails.get(s, b''))
            rows += [(r[1], k, i, r[3], r[4]) for i, r in enumerate(valid[s])]
    # ---- relay ----
    write_manifest(refd)
    R = {'20261023_23': [rrow(D0s - 3000, '81000.105', (D0s - 3000) * 1000 + 1200), rrow(D0s - 2, '81000.205', D0s * 1000 - 500)],
         '20261024_00': [rrow(D0s - 1, '81000.305', D0s * 1000 + 800), rrow(D0s - 1, '81000.305', D0s * 1000 + 801, dup=True),
                         rrow(D0s, '81000.405', D0s * 1000 + 1500), rrow(D0s, '81000.405', D0s * 1000 + 1501, dup=True),
                         rrow(D0s + 1, '81000.505', (D0s + 1) * 1000 + 1200), rrow(D0s + 2, '81000.605', (D0s + 2) * 1000 + 1200),
                         rrow(D0s + 4, '81000.805', (D0s + 4) * 1000 + 1200), rrow(D0s + 3, '81000.705', (D0s + 4) * 1000 + 1300, reg=True),
                         rrow(D0s + 6, '81000.905', (D0s + 6) * 1000 + 1200),
                         rrow(D0s + 7, '81001.005', (D0s + 7) * 1000 + 1200), rrow(D0s + 7, '81001.015', (D0s + 7) * 1000 + 1300),
                         rrow(D0s + 7, '81001.015', (D0s + 7) * 1000 + 1301, dup=True),
                         rrow(D0s + 8, '81001.105', (D0s + 8) * 1000 + 1200), rrow(D0s + 8, '81001.105', (D0s + 8) * 1000 + 1300, rts=D0s + 9),
                         rrow(0, None, (D0s + 9) * 1000, kind='NOTE'),
                         rrow(D0s + 3590, '81001.205', (D0s + 3590) * 1000 + 1200)]}
    for h in range(1, 23):
        if h != 10:                                                                # relay hour 10 missing
            R[f'{day}_{h:02d}'] = [rrow(Hs(h) + 100, f'8{h:04d}.125', (Hs(h) + 100) * 1000 + 1200),
                                   rrow(Hs(h) + 3500, f'8{h:04d}.375', (Hs(h) + 3500) * 1000 + 1200)]
    R[f'{day}_13'].insert(1, 'this line is not json')                              # Arrow refuses -> json.loads fallback
    R[f'{day}_23'] = [rrow(Hs(23) + 100, '81023.125', (Hs(23) + 100) * 1000 + 1200), rrow(D1s - 3, '81023.875', (D1s - 3) * 1000 + 1200)]
    rraws, pubs = {}, []
    for s in plan:
        if s in R:
            rraws[s], p = write_ref(refd, s, R[s])
            pubs += p
    ledger = write_ledger(tmp / 'A_ledger.jsonl', base_ledger())
    gaps = [(None, D0 - 3_590_000 + 100), (H(4) + 3_599_500, H(6) - 3_000 + 100), (D1 - 2_345, None)]
    rgaps = [(None, D0s - 3000), (Hs(9) + 3500, Hs(11) + 100), (D1s - 3, None)]
    return dict(day=day, D0=D0, D1=D1, D0s=D0s, D1s=D1s, H=H, Hs=Hs, plan=plan, roots=M.Roots(bnd, refd, ledger, out_root), bnd=bnd, refd=refd,
                ledger=ledger, raws=raws, rraws=rraws, valid=valid, rows=rows, pubs=pubs, gaps=gaps, rgaps=rgaps)


# ---------------------------------------------------------------- cases
def t_allow_list(tmp):
    for d in M.DAYS:
        p = M.day_plan(d)
        prev = (dt.date(int(d[:4]), int(d[4:6]), int(d[6:])) - dt.timedelta(days=1)).strftime('%Y%m%d')
        check(f'plan {d}: the previous day 23Z + the day 00Z..23Z, all allowed, no next-day stem',
              p == [f'{prev}_23'] + [f'{d}_{h:02d}' for h in range(24)] and all(s in M.ALLOWED for s in p) and not any(NEXT_DAY in s for s in p))
    check('allow-list: exactly 20261023_23 + 96 hours of 20261024..27', len(M.ALLOWED) == 97 and '20261023_23' in M.ALLOWED)
    M.day_ms('20261024')                                                            # warm imports before the zero-call cases
    r0 = M.Roots(tmp / 'z_bn', tmp / 'z_ref', tmp / 'z_ledger', tmp / 'z_out')
    bad_stems = [f'{NEXT_DAY}_00', f'{NEXT_DAY}_23', '20261023_22', '20261022_23', '20261024_24', '2026102400', f'20261024_00/../{NEXT_DAY}_00', '', None, 123]
    with Opens() as O:
        for i, s in enumerate(bad_stems):
            for fam in ('bookTicker', 'reference_prices'):
                refused(f'forbidden stem #{i} ({fam}) refused', lambda: M.hour_path(r0, fam, s), C.Forbidden, 'not on the allow-list')
        refused('unknown family refused', lambda: M.hour_path(r0, 'trade', '20261024_00'), C.Forbidden, 'unknown raw family')
        for i, d in enumerate([NEXT_DAY, '20261023', '20261108', 'x', None]):
            refused(f'forbidden day #{i} refused (run_day)', lambda: M.run_day(d, r0), C.Forbidden, 'not a measure day')
        refused('probe: a next-day hour refused', lambda: M.run_probe(tmp / 'z_probe', f'{NEXT_DAY}_00', r0), C.Forbidden, 'not on the allow-list')
        refused('probe: an hour of another measure day refused', lambda: M.run_probe(tmp / 'z_probe', '20261025_00', r0), ValueError, 'probe hour must be')
    check('every refusal above happened with ZERO filesystem calls', not O.all(), f'{len(O.all())} calls')


def t_day_a(tmp):
    A = FX['A'] = fixture_a(tmp)
    D0, D1, H = A['D0'], A['D1'], A['H']
    with Opens() as O:
        R = M.run_day(A['day'], A['roots'])
    out = A['roots'].out_root / 'btc'
    want_names = {f'{s}.csv.gz' for s in A['plan']} | {f'{s}.jsonl.gz' for s in A['plan']}
    touched = {Path(c[1]).name for c in O.under(A['bnd'], A['refd'] / 'reference_prices')}
    check('A: exactly the 25 planned hour files of each family were opened (missing ones attempted once), nothing else in a data dir',
          touched == want_names and not [c for c in O.under(A['bnd'], A['refd']) if c[0] in ('listdir', 'scandir', 'os.listdir', 'os.scandir', 'glob.glob')],
          f'extra {sorted(touched - want_names)[:5]} absent {sorted(want_names - touched)[:5]}')
    check('A: outputs published (btc, ref, receipt), no staging dir left',
          sorted(p.name for p in out.iterdir()) == sorted(['btc_20261024.parquet', 'ref_20261024.parquet', 'btc_20261024.json']))
    exp = brute_btc(D0, A['rows'], A['gaps'])
    t = cmp_btc('A', out / 'btc_20261024.parquet', exp)
    v0, v1 = A['valid']['20261024_00'], A['valid']['20261024_01']
    v4, v6, v23 = A['valid']['20261024_04'], A['valid']['20261024_06'], A['valid']['20261024_23']
    gold = [(D0, want(v0[0], 1, 60, 'ok'), 'D0: the row at D0 - 60 (sitting in the 00Z file), n_upd 1'),
            (D0 + 100, want(v0[1], 1, 0, 'ok'), 'a row exactly on a grid point'),
            (D0 + 200, want(v0[5], 4, 0, 'ok'), 'equal E_ms at D0 + 200: the later line wins; n_upd 4'),
            (D0 + 1200, want(v0[5], 0, 1000, 'ok'), 'age 1000 ms is still ok'),
            (D0 + 1300, want(v0[5], 0, 1100, 'stale'), 'age 1100 ms is stale'),
            (D0 + 1500, want(v0[6], 1, 0, 'ok'), 'after the quiet second'),
            (D0 + 2100, want(v0[7], 2, 50, 'ok'), 'out of order in a file (2050 then 2010): the E_ms max wins, n_upd 2'),
            (D0 + 3_599_000, want(v0[9], 2, 0, 'ok'), 'a late row in the next file (E 3,598,950) counted; the E_ms max wins'),
            (D0 + 3_600_100, want(v1[1], 1, 50, 'ok'), 'next hour'),
            (H(4) + 3_599_500, want(v4[1], 1, 0, 'gap'), 'the missing 05Z file: the hole starts at the last E_ms read'),
            (H(6) - 3_000, want(v6[0], 1, 0, 'gap'), 'first row after the hole (n_upd may miss rows): still gap'),
            (H(6) - 2_900, want(v6[0], 0, 100, 'ok'), 'the hole ends at e_min + 100'),
            (D1 - 2_400, want(v23[0], 0, 2_597_600, 'stale'), 'before the day tail'),
            (D1 - 2_300, want(v23[1], 1, 45, 'gap'), 'the day tail after the last E_ms read is gap'),
            (D1 - 100, want(v23[1], 0, 2_245, 'gap'), 'the last grid point')]
    for g, w, what in gold:
        check(f'A gold: {what}', at(t, D0, g) == w, f'got {at(t, D0, g)} want {w}')
    J = json.loads((out / 'btc_20261024.json').read_text())
    b = J['bn']
    nvalid = len(A['rows'])
    cn = b['counts']
    check('A receipt: missing / fallback / unreadable / empty lists', b['missing'] == ['20261024_05'] and b['python_fallback'] == ['20261024_06']
          and b['unreadable'] == [] and b['empty'] == [], f"{b['missing']} {b['python_fallback']}")
    check('A receipt: counts (malformed 3 = short line + header + unterminated tail; invalid price 1; e_decreases 1; late 1; crossed 1; pre-day 3)',
          (cn['rows_valid'], cn['malformed'], cn['invalid_price'], cn['e_decreases'], cn['late_vs_earlier_files'], cn['crossed'], cn['rows_pre_day'],
           cn['rows_in_grid'], cn['rows_after_last_g'], cn['update_id_nonincreasing']) == (nvalid, 3, 1, 1, 1, 1, 3, nvalid - 3, 0, 0), f'{cn}')
    fr = {f['stem']: f for f in b['files']}
    check('A receipt: sha256 + bytes of every hour file read equal the fixture bytes',
          all(fr[s]['sha256'] == hashlib.sha256(r).hexdigest() and fr[s]['bytes'] == len(r) for s, r in A['raws'].items())
          and fr['20261024_05']['state'] == 'missing' and 'sha256' not in fr['20261024_05'])
    check('A receipt: parser per file (Arrow with parity rows; fallback for the header file; unterminated tail counted)',
          fr['20261024_00']['parser'] == 'arrow' and fr['20261024_00']['malformed'] == 1 and fr['20261024_00']['parity_rows'] == 20
          and fr['20261024_06']['parser'] == 'python_fallback' and fr['20261024_07']['unterminated_tail'] == 1, f"{fr['20261024_00']}")
    sc = b['status_counts']
    check('A receipt: status counts = the reference counts and sum to 864,000',
          [sc[s] for s in M.STATUS] == [int((exp[6] == i).sum()) for i in range(4)] and sum(sc.values()) == N, f'{sc}')
    check('A receipt: gap points by reason (missing hour 35,976; day tail 23)',
          b['gap_points_by_reason'] == dict(before_first_file_read=0, missing_hour_file=35_976, after_last_row_read=23), f"{b['gap_points_by_reason']}")
    L = b['lag_ms_day']
    nday = sum(1 for r in A['rows'] if D0 <= r[0] < D1)
    check('A receipt: day lag p0 -4.877 / p50 70.1 (bin edge) / p100 1500.123 exact; n, negative 1, over 1 s 1',
          (L['p0'], L['p50'], L['p100'], L['n'], L['negative'], L['over_1s'], L['under_hist'], L['over_hist']) == (-4.877, 70.1, 1500.123, nday, 1, 1, 0, 0), f'{L}')
    check('A receipt: per-file exact lag quantiles (hour 10: p0 -4.877, p50 70.123)',
          fr['20261024_10']['lag_ms']['p0'] == -4.877 and fr['20261024_10']['lag_ms']['p50'] == 70.123, f"{fr['20261024_10']['lag_ms']}")
    e = J['era']
    check('A receipt: era (min recv >= boundary; ledger: 1 torn fragment bounded before the span, no start inside)',
          e['min_recv_ns'] == min(r[0] for f in A['valid'].values() for r in f) and e['ledger']['n_torn'] == 1 and e['ledger']['starts_inside_span'] == []
          and e['ledger']['in_force_at_span_start']['schema'] == V2, f"{e}")
    outs = J['outputs']
    check('A receipt: outputs sha256 / bytes equal the published files',
          all(hashlib.sha256((out / k).read_bytes()).hexdigest() == v['sha256'] and (out / k).stat().st_size == v['bytes'] for k, v in outs.items())
          and set(outs) == {'btc_20261024.parquet', 'ref_20261024.parquet'})
    check('A receipt: CPU / VmHWM / guard recorded', J['cpu_s'] > 0 and J['vmhwm_mib'] > 0 and J['guard_mib'] == C.GUARD_MIB and 'write' in J['cpu_stages_s'])
    # ---- relay ----
    rt = cmp_ref('A', out / 'ref_20261024.parquet', A['D0s'], A['pubs'], A['rgaps'])
    D0s, D1s, Hs = A['D0s'], A['D1s'], A['Hs']
    rgold = [(D0s, ('81000.405', 0, 1, 2, 'ok'), 'second 0: one payload received twice'),
             (D0s + 3, ('81000.705', 0, 1, 1, 'ok'), 'a regressed arrival still lands on its own second'),
             (D0s + 5, ('81000.805', 1, 0, 0, 'stale'), 'a missing second: stale, carried from the second before'),
             (D0s + 7, ('81001.015', 0, 2, 3, 'conflict'), 'two prices in one second: conflict, the latest received kept'),
             (D0s + 8, ('81001.105', 0, 2, 2, 'ok'), 'two payloads of one price: ok, n_obs 2'),
             (Hs(9) + 3501, ('80009.375', 1, 0, 0, 'gap'), 'inside the missing relay hour: gap'),
             (D1s - 3, ('81023.875', 0, 1, 1, 'ok'), 'the last publish read'),
             (D1s - 2, ('81023.875', 1, 0, 0, 'gap'), 'after the last publish read: gap')]
    for s, w, what in rgold:
        check(f'A ref gold: {what}', rat(rt, D0s, s) == w, f'got {rat(rt, D0s, s)} want {w}')
    r = J['ref']
    rf = {f['stem']: f for f in r['files']}
    check('A ref receipt: written; missing relay hour; json fallback hour; counts (non-PRICE 1, malformed 1, duplicates 3, regressed 1)',
          r['status'] == 'written' and r['missing'] == ['20261024_10'] and rf['20261024_13']['parser'] == 'python_fallback'
          and (r['counts']['nonprice'], r['counts']['malformed'], r['counts']['duplicates'], r['counts']['regressed']) == (1, 1, 3, 1)
          and r['counts']['rows_price'] == len(A['pubs']) and rf['20261024_00']['parser'] == 'arrow', f"{r['counts']}")
    check('A ref receipt: sha256 + bytes of every relay hour read', all(rf[s]['sha256'] == hashlib.sha256(x).hexdigest() and rf[s]['bytes'] == len(x)
                                                                      for s, x in A['rraws'].items()))
    check('A ref receipt: identity from the manifest', r['manifest']['topic'] == 'chainlinkAssetPriceUpdate/1' and r['expected_identity']['provider_feed_id'] == FEED)


def t_day_b(tmp):
    day = '20261025'
    D0 = M.day_ms(day)
    D1, H = D0 + DAY_MS, (lambda h: D0 + h * HOUR_MS)
    plan, bnd, out_root = M.day_plan(day), tmp / 'B_bn', tmp / 'B_out'
    out_root.mkdir()
    cc = iter(range(7_000_000, 7_900_000, 100))
    nx = lambda: next(cc)  # noqa: E731
    F = {f'{day}_00': [bn(D0 + 250, nx()), bn(D0 + 3_599_000, nx())]}
    for h in range(1, 23):
        F[f'{day}_{h:02d}'] = [bn(H(h) + 1_000_000, nx()), bn(H(h) + 3_599_500, nx())]
    F[f'{day}_08'] = ['garbage']                                                   # read, but no valid row
    F[f'{day}_12'] = [bn(H(12) + 1_000_000, nx()), bn(H(12) + 1_005_000, nx()), bn(H(12) + 3_599_500, nx())]
    F[f'{day}_23'] = [bn(H(23) + 1_000_000, nx()), bn(D1 - 500, nx()), bn(D1 - 30, nx())]
    del F[f'{day}_03']
    rows = []
    for k, s in enumerate(plan):
        if s in F:
            _, v = write_bn(bnd, s, F[s])
            rows += [(r[1], k, i, r[3], r[4]) for i, r in enumerate(v)]
    (bnd / f'{day}_03.csv.gz').write_bytes(b'not a gzip stream')                  # unreadable
    S = (H(12) + 1_002_000) * 10 ** 6
    ledger = write_ledger(tmp / 'B_ledger.jsonl', base_ledger() + [lentry(S)])
    roots = M.Roots(bnd, tmp / 'B_no_relay', ledger, out_root)
    R = M.run_day(day, roots)
    print(f"INFO day B (near-zero input bytes; no call recorder): cpu_s={R['cpu_s']} cpu_stages_s={R['cpu_stages_s']} vmhwm_mib={R['vmhwm_mib']}", flush=True)
    out = out_root / 'btc'
    check('B: relay skipped and said so; no ref parquet', R['ref']['status'] == 'skipped' and 'no relay manifest' in R['ref']['reason']
          and sorted(p.name for p in out.iterdir()) == ['btc_20261025.json', 'btc_20261025.parquet'], f"{R['ref']}")
    b = R['bn']
    check('B receipt: missing previous-day hour, unreadable 03Z, empty 08Z', b['missing'] == ['20261024_23'] and b['unreadable'] == [f'{day}_03']
          and b['empty'] == [f'{day}_08'] and b['counts']['rows_after_last_g'] == 1, f"{b['missing']} {b['unreadable']} {b['empty']} {b['counts']}")
    hs = [(h['reason'], h['lo'], h['hi']) for h in b['holes']]
    want_h = [('before_first_file_read', None, D0 + 350), ('missing_hour_file', H(2) + 3_599_500, H(4) + 1_000_100),
              ('missing_hour_file', H(7) + 3_599_500, H(9) + 1_000_100), ('after_last_row_read', D1 - 30, None),
              ('collector_start', H(12) + 1_000_000, H(12) + 1_005_100)]
    check('B receipt: holes (head, unreadable, empty, tail, the ledger restart) exactly', hs == want_h, f'{hs}')
    check('B receipt: the ledger start inside the span is recorded', R['era']['ledger']['starts_inside_span'] == [dict(started_at_ns=S, schema=V2)])
    gaps = [(lo, hi) for _, lo, hi in want_h]
    t = cmp_btc('B', out / 'btc_20261025.parquet', brute_btc(D0, rows, gaps))
    v0, v12, v23 = [r for r in rows if r[1] == 1], [r for r in rows if r[1] == 13], [r for r in rows if r[1] == 24]
    w0 = lambda r, n, a, s: (r[3], r[4], (r[3] + r[4]) / 2, n, a, s)  # noqa: E731
    for g, w, what in [(D0, (None, None, None, 0, -1, 'no_data'), 'no row read before D0 (previous-day hour missing): no_data, NaN prices'),
                       (D0 + 200, (None, None, None, 0, -1, 'no_data'), 'still no_data at D0 + 200'),
                       (D0 + 300, w0(v0[0], 1, 50, 'gap'), 'the first row: gap (the head hole ends at e_min + 100)'),
                       (D0 + 400, w0(v0[0], 0, 150, 'ok'), 'ok after the head hole'),
                       (H(12) + 1_000_000, w0(v12[0], 1, 0, 'gap'), 'a ledger restart: gap from the last E_ms before it'),
                       (H(12) + 1_005_100, w0(v12[1], 0, 100, 'ok'), 'ok again at e_b + 100'),
                       (D1 - 100, w0(v23[1], 0, 400, 'ok'), 'a row after the last grid point moves the tail hole past the day')]:
        got = at(t, D0, g)
        got = tuple(None if isinstance(x, float) and x != x else x for x in got)
        check(f'B gold: {what}', got == w, f'got {got} want {w}')


def t_day_c(tmp):
    day = '20261027'
    D0 = M.day_ms(day)
    D1, plan = D0 + DAY_MS, M.day_plan(day)
    bnd, refd, out_root = tmp / 'C_bn', tmp / 'C_ref', tmp / 'C_out'
    out_root.mkdir()
    write_manifest(refd)
    cc = iter(range(6_000_000, 6_900_000, 100))
    for s in plan + [f'{NEXT_DAY}_00']:                                             # the next-day files are TRAPS
        h0 = hour_ms(s)
        write_bn(bnd, s, [bn(h0 + 1_000_000, next(cc)), bn(h0 + 3_599_500, next(cc))])
        write_ref(refd, s, [rrow(h0 // 1000 + 100, '80000.125', h0 + 101_200), rrow(h0 // 1000 + 3500, '80000.375', h0 + 3_501_200)])
    roots = M.Roots(bnd, refd, write_ledger(tmp / 'C_ledger.jsonl', base_ledger()), out_root)
    with Opens() as O:
        R = M.run_day(day, roots)
    hits = [c for c in O.all() if NEXT_DAY in c[1]]
    check('C: the next-day trap files (both families) were never opened, listed or stat-ed', not hits, f'{len(hits)} calls')
    check('C: planned stems in the receipt = the day plan', R['allow_list']['planned'] == plan)
    t = pq.read_table(out_root / 'btc' / 'btc_20261027.parquet')
    s = lambda g: at(t, D0, g)[3:]  # noqa: E731
    check('C gold: D0 carries the previous day 23Z row (age 500, ok); D1 - 600 stale; D1 - 100 in the tail gap',
          s(D0) == (0, 500, 'ok') and s(D1 - 600) == (0, 2_599_400, 'stale') and s(D1 - 100) == (0, 400, 'gap'), f'{s(D0)} {s(D1 - 600)} {s(D1 - 100)}')
    rt = pq.read_table(out_root / 'btc' / 'ref_20261027.parquet')
    check('C ref gold: second 0 carried from the previous day 23Z (stale, age 100); the day tail after the last publish is gap',
          rat(rt, D0 // 1000, D0 // 1000)[1:] == (100, 0, 0, 'stale') and rat(rt, D0 // 1000, D1 // 1000 - 1)[4] == 'gap')


def t_write_once(tmp):
    A = FX['A']
    out = A['roots'].out_root / 'btc'
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    with Opens() as O:
        refused('write-once: a rerun of day A is refused', lambda: M.run_day(A['day'], A['roots']), ValueError, 'write-once')
    check('write-once: the refusal read no data (no hour file, relay or ledger touched)', not O.under(A['bnd'], A['refd'], A['ledger']), f'{O.under(A["bnd"], A["refd"], A["ledger"])[:3]}')
    check('write-once: the published files are unchanged', before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    rp = tmp / 'P_out'
    (rp / 'btc').mkdir(parents=True)
    (rp / 'btc' / 'btc_20261024.json').write_text('{}')
    refused('write-once: a partial earlier publish (receipt only) is refused', lambda: M.run_day(A['day'], M.Roots(A['bnd'], A['refd'], A['ledger'], rp)),
            ValueError, 'write-once')
    nr = tmp / 'no_root'
    refused('a missing run root is refused', lambda: M.run_day(A['day'], M.Roots(A['bnd'], A['refd'], A['ledger'], nr)), ValueError, 'setup_run.py')
    check('a missing run root: nothing created', not nr.exists())


def t_falsifiers(tmp):
    A = FX['A']

    def fresh(name, bnd=None, refd=None, ledger=None):
        r = tmp / f'F_{name}'
        r.mkdir()
        return M.Roots(bnd or A['bnd'], refd or A['refd'], ledger or A['ledger'], r)

    def nothing_published(name, roots):
        b = roots.out_root / 'btc'
        left = sorted(p.name for p in b.iterdir()) if b.exists() else []
        check(f'{name}: nothing published, no staging dir left', left == [], f'{left}')
    orig = M.py_bn_fields
    M.py_bn_fields = lambda line: (lambda f: None if f is None else (f[0], f[1], f[2], f[3] + 1e-9, f[4]))(orig(line))
    try:
        r = fresh('parity_bn')
        refused('falsifier: a bookTicker parser mismatch raises (parity)', lambda: M.run_day('20261024', r), ValueError, 'parser parity failed on bid')
        nothing_published('parity_bn', r)
    finally:
        M.py_bn_fields = orig
    orig_r = M.py_ref_row
    M.py_ref_row = lambda line: (lambda t: None if t is None else t[:8] + ('9' + str(t[8]),) + t[9:])(orig_r(line))
    try:
        r = fresh('parity_ref')
        refused('falsifier: a relay parser mismatch raises (parity)', lambda: M.run_day('20261024', r), ValueError, 'ref parser parity failed')
    finally:
        M.py_ref_row = orig_r
    keep = C.GUARD_MIB
    try:
        C.GUARD_MIB = 1.0
        r = fresh('guard')
        refused('falsifier: VmHWM >= GUARD_MIB stops the run', lambda: M.run_day('20261024', r), M.GuardStop, 'VmHWM')
        nothing_published('guard', r)
    finally:
        C.GUARD_MIB = keep
    day = '20261026'
    D0 = M.day_ms(day)
    bne = tmp / 'E_bn'
    write_bn(bne, '20261025_23', [bn(D0 - 1_000_000, 7_000_000, recv=M.HF_WS_V2_NS - 1)])
    r = fresh('era_row', bnd=bne, refd=tmp / 'no_relay')
    refused('falsifier: a row stamped before the hf_ws_v2 boundary is refused (era)', lambda: M.run_day(day, r), ValueError, 'era: a row is stamped')
    nothing_published('era_row', r)
    e0, e_before, e_in, e_after = M.HF_WS_V2_NS, (D0 - 3 * DAY_MS) * 10 ** 6, (D0 + HOUR_MS) * 10 ** 6, (D0 + 3 * DAY_MS) * 10 ** 6
    for name, lines, text in [('schema change inside the span', [lentry(e0), lentry(e_in, 'hf_ws_v3_other')], 'era: the collector schema'),
                              ('torn entry that could lie inside the span', [lentry(e_before), lentry(e_before)[:50], lentry(e_after)], 'torn entry'),
                              ('a non-v2 schema in force at the span start', [lentry(e0, 'legacy_post_parse')], 'era: the collector schema'),
                              ('unsorted starts', [lentry(e_before), lentry(e0)], 'append order')]:
        led = write_ledger(tmp / f'E_ledger_{len(name)}.jsonl', lines)
        r = fresh(f'ledger_{len(name)}', bnd=bne, refd=tmp / 'no_relay', ledger=led)
        with Opens() as O:
            refused(f'falsifier (ledger): {name} is refused', lambda: M.run_day(day, r), ValueError, text)
        check(f'falsifier (ledger): {name} - refused before any hour file', not O.under(bne), f'{O.under(bne)[:2]}')
    refb = tmp / 'E_ref_bad'
    write_manifest(refb)
    write_ref(refb, '20261025_23', [rrow(D0 // 1000 - 100, '80000.125', D0 - 98_800, feed='0xother')])
    r = fresh('identity', bnd=bne, refd=refb)
    with Opens() as O:
        refused('falsifier: a relay row of another feed is refused (identity)', lambda: M.run_day(day, r), ValueError, 'ref identity: provider_feed_id')
    check('falsifier: the identity refusal came before the bookTicker pass', not O.under(bne), f'{O.under(bne)[:2]}')


def t_probe(tmp):
    A = FX['A']
    D0, D0s = A['D0'], A['D0s']
    pd = tmp / 'probe'
    R = M.run_probe(pd, '20261024_00', A['roots'])
    check('probe: outputs + receipt (mode probe, one planned hour)', R['mode'] == 'probe' and R['allow_list']['planned'] == ['20261024_00']
          and sorted(p.name for p in pd.iterdir()) == ['probe_20261024_00.json', 'probe_20261024_00_btc.parquet', 'probe_20261024_00_ref.parquet'])
    t = pq.read_table(pd / 'probe_20261024_00_btc.parquet')
    v0 = A['valid']['20261024_00']
    check('probe gold: inside the hour as the day run (D0 + 200: n_upd 4 ok); D0 is gap (head hole); D0 + 3,599,000 gap with n_upd 1 (no next file)',
          at(t, D0, D0 + 200) == want(v0[5], 4, 0, 'ok') and at(t, D0, D0)[5] == 'gap' and at(t, D0, D0 + 3_599_000) == want(v0[9], 1, 0, 'gap')
          and t.num_rows == N, f'{at(t, D0, D0)} {at(t, D0, D0 + 3_599_000)}')
    rt = pq.read_table(pd / 'probe_20261024_00_ref.parquet')
    check('probe ref gold: second 0 as the day run', rat(rt, D0s, D0s) == ('81000.405', 0, 1, 2, 'ok'), f'{rat(rt, D0s, D0s)}')
    refused('probe: a rerun is refused (write-once)', lambda: M.run_probe(pd, '20261024_00', A['roots']), ValueError, 'write-once')
    refused('probe: an output dir under the run root is refused', lambda: M.run_probe(A['roots'].out_root / 'x', '20261024_01', A['roots']),
            ValueError, 'run root')


def main():
    tmp = Path(tempfile.mkdtemp(prefix='pf_mt_btc_selftest_'))
    try:
        for t in (t_allow_list, t_day_a, t_day_b, t_day_c, t_write_once, t_falsifiers, t_probe):
            try:
                t(tmp)
            except Exception as e:  # noqa: BLE001 - a crashing case is a failure, never a skip
                check(f'{t.__name__} ran to completion', False, f'{type(e).__name__}: {e}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'SELFTEST {"PASS" if not FAILS else "FAIL"}: {NCASE[0] - len(FAILS)}/{NCASE[0]} checks passed' + (f'; failed: {FAILS}' if FAILS else ''),
          flush=True)
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
