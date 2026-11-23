#!/usr/bin/env python3
"""MAKER-TRACE-RECONSTRUCTION-v1 END-TO-END INTEGRATION REHEARSAL (P-2026-005; session aea5bfb9 integration, 2026-11-22). Synthetic only.

ONE process, ONE temp dir. (1) A small price-time-priority venue simulator writes synthetic raw hour files in the real raw layouts (REST
/v1/orders/matches pages, WS predictOrderbook frames, collector events) for the declared days - ID 20261023 (REST only), measurement
20261024..27 (REST + WS) and GT 20261108 (hours 00-22) - plus step1 registries (exactly 288 windows per day: outcome / outcome_seen_ns /
up_index / start_price / end_price; one tie, the sealed Oct 27 23:55 outcome), the BOOK manifest ('complete'), the GT export (scope.json,
who.json, matches_20261108 and kernel / core orders + kernel fills of run v6 and basis for our signer 0xba14...ced8, a GT registry).
Every raw line is filed through pf_mt_common.hour_path: a line whose receipt hour is not consumed is REFUSED there (Forbidden before any
filesystem access) and dropped - counted, never printed. (2) pf_mt_common's paths and every module-level path constant / function
default of the builders and the checker are remapped into the temp dir; a leak scan refuses to start if any still names real data, and a
firewall (audit hook + pyarrow / pandas / os.stat wrappers) refuses any read of a real data path and any write outside the temp dir.
(3) The REAL builders run in production order through their own `if __name__ == '__main__'` blocks (exec'd from the same source snapshot
that was loaded): extract every day -> lifecycle of every depth market the extract receipts name -> truth build / dry / score -> econ run.
(4) mt_checks run (its CLI: run_checks + the two re-parses) and mt_checks.falsify. (5) An oracle compares every lifecycle claim with the
simulator's truth (observed / constrained claims must hold; assumption-sensitive misses are listed), feature predicates prove the world
exercised what it must. A JSON summary is printed; exit 0 only when nothing failed (else 1; 2 = harness error), naming the first failure.
Exercised: multi-price records (price-priority exhaustion), worst-price survivors that fill again, mint and merge legs, a taker address that
is also a maker (one order hash both as taker and maker, with price improvement), a self-trade, conflict re-polls (non-reconciling variant
first), byte-identical re-poll pages, volatile re-polls after resolution, a priceExecuted != maker-weighted price record, forced T1 / T2
priority violations, an unbounded fill window (masked drop, no pending), a frame gap with a fill inside it, a second connection, a dup frame,
lower / upper boundary_censored markets (ID, FILL, GT), a sealed registry outcome, a tie, a MATCH_POLL_INCOMPLETE event, own fills in GT.
Usage: PF_SLOT=claude3 predictfun/research/run_light.sh mt_e2e_<n> -- python3 <abs path of this file> [--keep] [--tmp DIR] [--no-falsify]"""
import argparse
import ast
import contextlib
import datetime as dt
import functools
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import resource
import shutil
import sys
import tempfile
import time
import traceback
import types
import uuid
from collections import Counter, defaultdict
from pathlib import Path, PurePath

sys.dont_write_bytecode = True
UTC = dt.timezone.utc
REAL_REPO = Path('/home/yuqing/ctaNew')
REAL_PMMM = Path('/home/yuqing/pmmm-data')
RESEARCH = REAL_REPO / 'predictfun' / 'research'
CHECKER_DIR = REAL_REPO / 'predictfun' / 'reviews' / 'maker_trace_20261122'
MODULES = (('pf_mt_extract', RESEARCH / 'pf_mt_extract.py'), ('pf_mt_lifecycle', RESEARCH / 'pf_mt_lifecycle.py'),
           ('pf_mt_truth', RESEARCH / 'pf_mt_truth.py'), ('pf_mt_econ', RESEARCH / 'pf_mt_econ.py'), ('pf_mt_report', RESEARCH / 'pf_mt_report.py'),
           ('mt_checks', CHECKER_DIR / 'mt_checks.py'))
DENY = (REAL_REPO / 'data', REAL_REPO / 'predictfun' / 'reviews', REAL_PMMM)      # any access refused (reading *.py / *.pyc code excepted)
SKIP_NAMES = {'__file__', '__cached__', '__builtins__', '__spec__', '__loader__', '__name__', '__doc__', '__package__', 'HERE'}
# mt_checks checks that cannot pass on synthetic inputs (name -> the real-only input it needs). Declared; empty = every check must pass.
EXPECTED_CHECK_FAILURES = {}
OWN = '0xba14f115ccf30e976f7639594fdee7b36c85ced8'
D_TRUE = 40                 # synthetic premise: bot clock = venue clock - 40 ms (the truth's clock band must contain it)
NS = 10 ** 6
MID0 = 8_000_000            # synthetic market ids: MID0 + 1000 * day index + window index
OTHER_MID = 7_999_999       # a market of no population (noise pages / frames)
ALLOWED = ('20261023', '20261024', '20261025', '20261026', '20261027', '20261108')
DAYIX = {'20261023': 0, '20261024': 1, '20261025': 2, '20261026': 3, '20261027': 4, '20261108': 8}
RUNS = ('v6', 'basis')
CUTOFF_MS = 1794178800000.0
SEQ = itertools.count(1)
C = None                    # pf_mt_common (patched), set in rehearse()
FW = None                   # the firewall


class HarnessError(RuntimeError):
    pass


def need(ok, msg):
    if not ok:
        raise HarnessError(msg)


_DATE = re.compile(r'(?<![0-9])(20[0-9]{6})(?![0-9])')


def sanitize(s):
    """Any 8-digit date token that is not a consumed day is masked: no path / stem of another date is ever printed."""
    return _DATE.sub(lambda m: m.group(1) if m.group(1) in ALLOWED else '<non-consumed-day>', str(s))


# ------------------------------------------------------------------------------------------------------------------------- firewall
class Firewall:
    """Refuses (PermissionError, recorded) every write outside the temp dir and every access to a real data path (DENY roots; reading
    *.py / *.pyc code is allowed). Fed by a sys audit hook (open, listdir / scandir, mkdir, rename, link, remove, rmtree, glob, ...) and by
    wrappers of the C++ I/O entry points the hook cannot see (pyarrow.parquet, pandas parquet) and of os.stat (exists / is_file probes)."""

    def __init__(s, tmp):
        s.tmp, s.deny = os.path.abspath(str(tmp)), [str(p) for p in DENY]
        s.violations, s.rec, s.on, s.popen, s.stage = [], None, False, [], 'harness'

    @staticmethod
    def under(p, root):
        return p == root or p.startswith(root + '/')

    def check(s, op, path, write, dfd=None):
        if not s.on or path is None or isinstance(path, int):
            return
        try:
            p = os.fsdecode(path)
        except TypeError:
            return
        if not p.startswith('/'):
            if isinstance(dfd, int) and dfd >= 0:                       # relative to an open directory (shutil.rmtree / os.fwalk)
                try:
                    p = os.path.join(os.readlink(f'/proc/self/fd/{dfd}'), p)
                except OSError:
                    return
            elif op == 'open' and not write:                            # a dir_fd-relative read-only open: the 'open' event carries no dir_fd
                return
        p = os.path.abspath(p)
        if s.rec is not None:
            s.rec.append((op, p))
        if write:
            if not (s.under(p, s.tmp) or p == '/dev/null'):
                s.refuse(op, p, 'write outside the temp dir')
        elif any(s.under(p, d) for d in s.deny) and not p.endswith(('.py', '.pyc')):
            s.refuse(op, p, 'access to a real data path')

    def refuse(s, op, p, why):
        s.violations.append(dict(stage=s.stage, op=op, path=sanitize(p), why=why))
        raise PermissionError(f'[e2e firewall] {why}: {op} {sanitize(p)}')


_WRITE_EV = {'os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime', 'os.truncate', 'shutil.rmtree', 'os.chown', 'os.chflags',
             'os.lchmod', 'os.setxattr', 'os.removexattr'}
_READ_EV = {'os.listdir', 'os.scandir', 'glob.glob', 'glob.glob/2', 'pathlib.Path.glob', 'pathlib.Path.rglob', 'os.listxattr', 'os.getxattr'}
_PAIR_EV = {'os.rename', 'os.link', 'os.symlink', 'shutil.copyfile', 'shutil.copytree', 'shutil.move', 'shutil.copymode', 'shutil.copystat'}
_DFD = {'os.remove': 1, 'os.rmdir': 1, 'os.mkdir': 2, 'os.chmod': 2, 'os.chown': 3, 'os.utime': 3, 'shutil.rmtree': 1}     # dir_fd argument position


def _is_write(mode, flags):
    if isinstance(mode, str):
        return any(c in mode for c in 'wax+')
    if isinstance(flags, int):
        return bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
    return False


def _audit(ev, args):
    if FW is None or not FW.on:
        return
    if ev == 'open':
        a = tuple(args) + (None, None, None)
        FW.check('open', a[0], _is_write(a[1], a[2]))
    elif ev in _WRITE_EV:
        i = _DFD.get(ev)
        FW.check(ev, args[0] if args else None, True, args[i] if i is not None and len(args) > i else None)
    elif ev in _READ_EV:
        FW.check(ev, (args[0] if args else '.') or '.', False)
    elif ev in _PAIR_EV:
        two = ev in ('os.rename', 'os.link')
        sd = args[2] if two and len(args) > 2 else None
        dd = args[3] if two and len(args) > 3 else (args[2] if ev == 'os.symlink' and len(args) > 2 else None)
        if ev != 'os.symlink':
            FW.check(ev, args[0], ev in ('os.rename', 'shutil.move'), sd)
        FW.check(ev, args[1], True, dd)
    elif ev == 'subprocess.Popen':
        FW.popen.append(dict(stage=FW.stage, argv=[sanitize(x) for x in (args[1] if isinstance(args[1], (list, tuple)) else [args[1]])][:6]))


def _wrap_io():
    import pandas as pd
    import pyarrow.parquet as pq

    def chk(v, write, op):
        if isinstance(v, (str, os.PathLike)):
            FW.check(op, v, write)
        elif isinstance(v, (list, tuple)):
            for x in v:
                if isinstance(x, (str, os.PathLike)):
                    FW.check(op, x, write)

    def rd(fn, op):
        @functools.wraps(fn)
        def g(*a, **k):
            chk(a[0] if a else k.get('source', k.get('path', k.get('where'))), False, op)
            return fn(*a, **k)
        return g
    for n in ('read_table', 'read_schema', 'read_metadata'):
        setattr(pq, n, rd(getattr(pq, n), f'pq.{n}'))
    wt = pq.write_table

    @functools.wraps(wt)
    def write_table(table, where, *a, **k):
        chk(where, True, 'pq.write_table')
        return wt(table, where, *a, **k)
    pq.write_table = write_table
    PW0, PF0 = pq.ParquetWriter, pq.ParquetFile

    class ParquetWriter(PW0):
        def __init__(self, where, *a, **k):
            chk(where, True, 'pq.ParquetWriter')
            super().__init__(where, *a, **k)

    class ParquetFile(PF0):
        def __init__(self, source, *a, **k):
            chk(source, False, 'pq.ParquetFile')
            super().__init__(source, *a, **k)
    pq.ParquetWriter, pq.ParquetFile = ParquetWriter, ParquetFile
    rp = pd.read_parquet

    @functools.wraps(rp)
    def read_parquet(path, *a, **k):
        chk(path, False, 'pd.read_parquet')
        return rp(path, *a, **k)
    pd.read_parquet = read_parquet
    tp = pd.DataFrame.to_parquet

    @functools.wraps(tp)
    def to_parquet(self, path=None, *a, **k):
        chk(path, True, 'DataFrame.to_parquet')
        return tp(self, path, *a, **k)
    pd.DataFrame.to_parquet = to_parquet
    st0 = os.stat

    @functools.wraps(st0)
    def stat(path, *a, **k):
        FW.check('os.stat', path, False)
        return st0(path, *a, **k)
    os.stat = stat


# ------------------------------------------------------------------------------------------------------------- modules and patching
def remap_value(v, M):
    if isinstance(v, PurePath) or (isinstance(v, str) and v.startswith('/')):
        s = str(v)
        if s.endswith(('.py', '.pyc')):
            return v, False
        for a, b, exact in M:
            if s == a or (not exact and s.startswith(a + '/')):
                n = b + s[len(a):]
                return (type(v)(n) if isinstance(v, PurePath) else n), True
    elif isinstance(v, (tuple, list)) and v and all(isinstance(x, (str, PurePath)) for x in v):
        out = [remap_value(x, M) for x in v]
        if any(c for _, c in out):
            return type(v)(x for x, _ in out), True
    return v, False


def _functions(obj, modname):
    if isinstance(obj, types.FunctionType) and obj.__module__ == modname:
        yield obj
    elif isinstance(obj, type) and obj.__module__ == modname:
        for v in vars(obj).values():
            v = v.__func__ if isinstance(v, (staticmethod, classmethod)) else v
            if isinstance(v, types.FunctionType):
                yield v


def remap_module(mod, M):
    """Every module-level path constant (str / Path / tuple of them) and every function / method default naming a real data root."""
    changed = []
    for k, v in list(vars(mod).items()):
        if k in SKIP_NAMES:
            continue
        nv, c = remap_value(v, M)
        if c:
            setattr(mod, k, nv)
            changed.append(k)
    for k, v in list(vars(mod).items()):
        for fn in _functions(v, mod.__name__):
            if fn.__defaults__:
                nd = [remap_value(x, M) for x in fn.__defaults__]
                if any(c for _, c in nd):
                    fn.__defaults__ = tuple(x for x, _ in nd)
                    changed.append(f'{fn.__qualname__}() default')
            if fn.__kwdefaults__:
                nk = {a: remap_value(x, M) for a, x in fn.__kwdefaults__.items()}
                if any(c for _, c in nk.values()):
                    fn.__kwdefaults__ = {a: x for a, (x, _) in nk.items()}
                    changed.append(f'{fn.__qualname__}() kwdefault')
    return changed


def _real_data(v):
    if isinstance(v, PurePath) or (isinstance(v, str) and v.startswith('/')):
        s = os.path.abspath(str(v))
        return (any(s == str(d) or s.startswith(str(d) + '/') for d in DENY) or s == str(REAL_REPO)) and not s.endswith(('.py', '.pyc'))
    if isinstance(v, (tuple, list)):
        return any(_real_data(x) for x in v if isinstance(x, (str, PurePath)))
    return False


def leak_scan(mod):
    out = [f'{mod.__name__}.{k}' for k, v in vars(mod).items() if k not in SKIP_NAMES and _real_data(v)]
    for k, v in vars(mod).items():
        for fn in _functions(v, mod.__name__):
            if any(_real_data(x) for x in (fn.__defaults__ or ())) or any(_real_data(x) for x in (fn.__kwdefaults__ or {}).values()):
                out.append(f'{mod.__name__}.{fn.__qualname__}() default')
    return out


def main_block(src, path):
    """The module's `if __name__ == '__main__':` body, compiled from the same source snapshot the module was executed from."""
    tree = ast.parse(src, str(path))

    def is_main(t):
        return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == '__name__' and len(t.comparators) == 1
                and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == '__main__')
    blk = [n for n in tree.body if isinstance(n, ast.If) and is_main(n.test)]
    need(len(blk) == 1, f'{path.name}: {len(blk)} __main__ blocks (expected 1)')
    return compile(ast.Module(body=blk[0].body, type_ignores=[]), str(path), 'exec')


def load_all(tmp_repo, tmp_pmmm, S, FAIL):
    sys.path.insert(0, str(RESEARCH))
    import pf_mt_common as C0
    need(Path(C0.__file__).resolve() == RESEARCH / 'pf_mt_common.py', f'pf_mt_common resolved to {C0.__file__}')
    base = Path(C0.REPO)
    for k in ('PF_RUN', 'OUT', 'RECEIPT', 'STEP1'):
        setattr(C0, k, tmp_repo / Path(getattr(C0, k)).relative_to(base))
    C0.REPO = tmp_repo
    M = [(str(REAL_REPO / 'data'), str(tmp_repo / 'data'), False), (str(REAL_REPO / 'predictfun' / 'reviews'), str(tmp_repo / 'predictfun' / 'reviews'), False),
         (str(REAL_PMMM), str(tmp_pmmm), False), (str(REAL_REPO), str(tmp_repo), True)]
    patched = {'pf_mt_common': ['REPO', 'PF_RUN', 'OUT', 'RECEIPT', 'STEP1'] + remap_module(C0, M)}
    mods, srcs, mains = {'pf_mt_common': C0}, {'pf_mt_common': Path(C0.__file__).read_bytes()}, {}
    for name, path in MODULES:
        src = path.read_bytes()
        mod = types.ModuleType(name)
        mod.__file__ = str(path)
        sys.modules[name] = mod
        exec(compile(src, str(path), 'exec'), mod.__dict__)                 # module-level code sees the patched pf_mt_common
        mods[name], srcs[name], mains[name] = mod, src, main_block(src, path)
        patched[name] = remap_module(mod, M)
    S['sources_sha256_16'] = {n: hashlib.sha256(b).hexdigest()[:16] for n, b in srcs.items()}
    S['patched'] = patched
    leaks = [x for m in mods.values() for x in leak_scan(m)]
    if leaks:
        FAIL.append(dict(module='harness', check='path_patch', evidence=f'real data paths left after patching (refused to start): {leaks[:8]}'))
        return None
    return types.SimpleNamespace(C=C0, mods=mods, srcs=srcs, mains=mains, stages=[])


# ------------------------------------------------------------------------------------------------------------------ raw formats
def e18p(p):
    return int(round(p * 1000)) * 10 ** 15


def e18q(q):
    return int(round(q * 10 ** 6)) * 10 ** 12


def hx(tag, n=64):
    return hashlib.sha256(tag.encode()).hexdigest()[:n]


def ohash(tag):
    return '0x' + hx('order:' + tag)


def cased(h40):
    """A checksum-like mixed-case spelling (one spelling per address everywhere in the raw data)."""
    k = hashlib.sha256(h40.encode()).hexdigest()
    return '0x' + ''.join(c.upper() if c.isalpha() and int(k[i], 16) >= 8 else c for i, c in enumerate(h40))


def addr(tag):
    return cased(hx('addr:' + tag, 40))


def uuid7(ms, tag):
    r = int(hx('sid' + tag), 16)
    return str(uuid.UUID(int=(ms << 80) | (7 << 76) | ((r & 0xfff) << 64) | (2 << 62) | ((r >> 12) & ((1 << 62) - 1))))


def iso(ms):
    return dt.datetime.fromtimestamp(ms / 1000, UTC).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def day_ms(d):
    return int(dt.datetime.strptime(d, '%Y%m%d').replace(tzinfo=UTC).timestamp()) * 1000


def stem_of_ms(ms):
    return dt.datetime.fromtimestamp(ms // 1000, UTC).strftime('%Y%m%d_%H')


def outc(i, status='OPEN', bb=None):
    return dict(name='Up' if i == 1 else 'Down', indexSet=i, onChainId=str(10 ** 20 + i), status=status, bestBid=bb, bestAsk=None, team=None,
                variantData=None)


def market_obj(mid, ws_s, final=None):
    live, W = final is None, ws_s * 1000
    st = (lambda i: 'OPEN') if live else (lambda i: 'WON' if (final == 'Up') == (i == 1) else 'LOST')
    return dict(id=mid, title=f'BTC Up or Down 5m {ws_s}', categorySlug=f'btc-updown-5m-{ws_s}', feeRateBps=200, makerFeeBps=0, spreadThreshold=0.06,
                shareThreshold=10, rewards=dict(current=None, schedule=[dict(hourlyRate=12, startsAt=iso(W), endsAt=iso(W + 300000))]), isBoosted=False,
                boostStartsAt=iso(W), boostEndsAt=iso(W + 300000), status='REGISTERED' if live else 'RESOLVED', tradingStatus='OPEN' if live else 'CLOSED',
                resolution=None if live else dict(name=final, indexSet=1 if final == 'Up' else 2), outcomes=[outc(1, st(1)), outc(2, st(2))],
                decimalPrecision=2)


def rest_line(recv_ns, mid, data, status=200, path='/v1/orders/matches'):
    raw = json.dumps(dict(success=status == 200, cursor='c', data=data), separators=(',', ':'))
    return json.dumps(dict(run_id='synthetic', seq=next(SEQ), source='rest', kind='response', recv_ns=recv_ns, recv_mono_ns=0, path=path,
                           params=dict(marketId=str(mid), first='100'), sent=dict(recv_ns=recv_ns - 10 ** 8, recv_mono_ns=0), status=status, raw=raw),
                      separators=(',', ':'))


def ws_raw(mid, upd, bids, asks, pb, pa, n):
    data = dict(asks=[[p, q] for p, q in asks], bids=[[p, q] for p, q in bids],
                lastOrderSettled=dict(id='1', kind='LIMIT', marketId=mid, outcome='Yes', price='0.50', side='Bid'), marketId=mid, orderCount=n,
                settlementsPending=dict(asks=pa, bids=pb), updateTimestampMs=upd, version=1)
    return json.dumps(dict(type='M', topic=f'predictOrderbook/{mid}', data=data), separators=(',', ':'))


def ws_line(recv_ns, conn, raw):
    return json.dumps(dict(run_id='synthetic', seq=next(SEQ), source='predict_ws', kind='frame', recv_ns=recv_ns, recv_mono_ns=0, connection_id=conn,
                           raw=raw), separators=(',', ':'))


def ev_line(recv_ns, kind, **fields):
    return json.dumps(dict(run_id='synthetic', seq=next(SEQ), source='events', kind=kind, recv_ns=recv_ns, recv_mono_ns=0, **fields), separators=(',', ':'))


def match_kind(tq, ti, mq, mi):
    return 'regular' if mi == ti and mq != tq else 'mint' if mi != ti and mq == tq == 'Bid' else 'merge' if mi != ti and mq == tq == 'Ask' else 'invalid'


def record_json(r, mk, final, bad):
    st = (lambda i: 'OPEN') if final is None else (lambda i: 'WON' if (final == 'Up') == (i == 1) else 'LOST')
    bb = None if final is None else '0.99'
    makers = [dict(quoteType=o['quote'], amount=str(a), price=str(e18p(o['px'])), outcome=outc(o['iset'], st(o['iset']), bb), signer=o['sg'], hash=o['h'],
                   fee=dict(amount='0', type='SHARES')) for o, a in (r['bad_legs'] if bad else r['legs'])]
    tk = dict(quoteType=r['tq'], amount=str(r['tot']), price=str(r['pt']), outcome=outc(r['ti'], st(r['ti']), bb), signer=r['sg'], hash=r['tk_hash'],
              fee=dict(amount=str(r['fee']), type=r['ft']))
    return dict(market=mk, taker=tk, amountFilled=str(r['tot']), priceExecuted=str(r['pt']), makers=makers, transactionHash=r['tx'],
                settlementId=r['sid'], executedAt=iso(r['conf']))


# --------------------------------------------------------------------------------------------------------------- venue simulator
class Venue:
    """One market. Resting orders (YES space via pf_mt_common.yes_space) and scripted takers matched in price-time priority (or a forced
    allocation, to plant priority violations); one match = one settlement record (UUIDv7 sid shortly after the match, executedAt = the
    Confirmed second, first receipt = settle + lag). The book at u shows every order posted at or before u, not removed before u, with its
    remaining after every match at or before u (premise: a frame reflects every match at or before its upd_ms); settlements pending at the
    taker's YES limit for PEND_MS after the match."""
    FRAME_MS, FRAME_OFF, PEND_MS = 500, 130, 900

    def __init__(s, day, idx, label, depth=False, frames=True, f_lo=-60000, f_hi=330000, gaps=(), dual=(), dups=(), run=None):
        s.day, s.idx, s.label = day, idx, label
        s.ws_s = day_ms(day) // 1000 + 300 * idx
        s.W, s.mid = s.ws_s * 1000, MID0 + 1000 * DAYIX[day] + idx
        s.we = s.W + 300000
        s.depth, s.frames_on, s.f_lo, s.f_hi, s.gaps, s.dual, s.dups, s.run = depth, frames, f_lo, f_hi, gaps, dual, set(dups), run
        s.orders, s.ev, s.recs, s.pend, s.n, s.forced, s.final, s.dup_hashes = {}, [], [], [], 0, set(), None, set()

    def _push(s, t, kind, a):
        s.ev.append((s.W + t, s.n, kind, a))
        s.n += 1

    def post(s, otag, sg, side, p, size, t, cancel=None, no=False, own=None, h_of=None):
        if side == 'bid':
            quote, iset, px = ('Ask', 2, round(1 - p, 2)) if no else ('Bid', 1, p)
        else:
            quote, iset, px = ('Bid', 2, round(1 - p, 2)) if no else ('Ask', 1, p)
        ys, yp = C.yes_space(quote, iset, px)
        need(ys == side and abs(yp - p) < 1e-9 and otag not in s.orders, f'{s.label}:{otag}: bad order spec')
        h = s.orders[h_of]['h'] if h_of else ohash(f'{s.label}:{otag}')     # h_of: the SAME hash at another price (an H1 identity conflict)
        if h_of:
            s.dup_hashes.add(h)
        s.orders[otag] = dict(tag=otag, h=h, sg=sg, quote=quote, iset=iset, px=px, ys=ys, yp=yp, pt=int(round(yp * 100)),
                              size=float(size), t_post=None, t_out=None, why=None, fills=[], own=own, seq=s.n, ahead_first=None, born_taker=False)
        s._push(t, 'post', otag)
        if cancel is not None:
            s._push(cancel, 'cancel', otag)

    def take(s, rtag, t, sg, side, p, size, no=False, fee='SHARES', rest=None, force=None, listing='worst_first', px_exec=None, lag=700,
             bad_first=False, pend=True):
        s._push(t, 'take', dict(rtag=rtag, sg=sg, side=side, p=p, size=float(size), no=no, fee=fee, rest=rest, force=force, listing=listing,
                                px_exec=px_exec, lag=lag, bad_first=bad_first, pend=pend))

    @staticmethod
    def rem(o, upto=None):
        return o['size'] - sum(q for t, q, _ in o['fills'] if upto is None or t <= upto)

    def simulate(s):
        for T, _n, kind, a in sorted(s.ev, key=lambda e: (e[0], e[1])):
            if kind == 'post':
                s.orders[a]['t_post'] = T
            elif kind == 'cancel':
                o = s.orders[a]
                if o['t_out'] is None and s.rem(o) > 1e-9:
                    o['t_out'], o['why'] = T, 'cancelled'
            else:
                s._match(T, a)
        s.by_hash = {o['h']: o for o in s.orders.values()}
        s.rec_by_tag = {r['rtag']: r for r in s.recs}
        s.rec_by_sid = {r['sid']: r for r in s.recs}

    def _match(s, T, d):
        buy = d['side'] == 'buy'
        if buy:
            tq, ti, tpx = ('Ask', 2, round(1 - d['p'], 2)) if d['no'] else ('Bid', 1, d['p'])
        else:
            tq, ti, tpx = ('Bid', 2, round(1 - d['p'], 2)) if d['no'] else ('Ask', 1, d['p'])
        tys, tyl = C.yes_space(tq, ti, tpx)
        tag = f'{s.label}:{d["rtag"]}'
        need(tys == ('bid' if buy else 'ask'), f'{tag}: taker side')
        live = [o for o in s.orders.values() if o['t_post'] is not None and o['t_post'] <= T and o['t_out'] is None and o['ys'] != tys and s.rem(o) > 1e-9]
        if d['force']:
            alloc = []
            for k, q in d['force']:
                o = s.orders[k]
                need(o in live and s.rem(o) >= q - 1e-9, f'{tag}: forced allocation infeasible for {k}')
                alloc.append((o, float(q)))
                s.forced.add(o['h'])
        else:
            cross = sorted([o for o in live if (o['yp'] <= tyl + 1e-9 if buy else o['yp'] >= tyl - 1e-9)],
                           key=lambda o: ((o['yp'] if buy else -o['yp']), o['t_post'], o['seq']))
            left, alloc = d['size'], []
            for o in cross:
                if left <= 1e-9:
                    break
                q = min(s.rem(o), left)
                alloc.append((o, q))
                left -= q
            need(left <= 1e-9 or d['rest'], f'{tag}: {left} shares unfilled (script inconsistent)')
        tot_sh = sum(q for _, q in alloc)
        need(tot_sh > 1e-9, f'{tag}: no liquidity')
        for i, (o, q) in enumerate(alloc):
            if not o['fills']:
                o['ahead_first'] = sum(q2 for o2, q2 in alloc[:i] if (o2['ys'], o2['pt']) == (o['ys'], o['pt']))
            o['fills'].append((T, q, d['rtag']))
            if s.rem(o) <= 1e-9:
                o['t_out'], o['why'] = T, 'exhausted'
        tk_hash = '0x' + hx('tk:' + tag)
        if d['rest']:
            left = d['size'] - tot_sh
            need(left > 1e-9 and d['rest'] not in s.orders, f'{tag}: no remainder to rest')
            o = dict(tag=d['rest'], h=ohash(f'{s.label}:{d["rest"]}'), sg=d['sg'], quote=tq, iset=ti, px=tpx, ys=tys, yp=tyl, pt=int(round(tyl * 100)),
                     size=float(left), t_post=T, t_out=None, why=None, fills=[], own=None, seq=s.n, ahead_first=None, born_taker=True)
            s.n += 1
            s.orders[d['rest']] = o
            tk_hash = o['h']
        order = list(alloc) if d['listing'] == 'best_first' else list(reversed(alloc))
        legs = [(o, e18q(q)) for o, q in order]
        tot = sum(a for _, a in legs)
        num = sum(a * (e18p(o['px']) if o['iset'] == 1 else C.E18 - e18p(o['px'])) for o, a in legs)
        yt = num // tot if d['px_exec'] is None else e18p(d['px_exec'])
        pt = yt if ti == 1 else C.E18 - yt
        fee = 2 * min(yt, C.E18 - yt) * tot // (100 * (pt if d['fee'] == 'SHARES' else C.E18))
        settle = T + 11 + len(s.recs) % 7
        r = dict(rtag=d['rtag'], T=T, settle=settle, sid=uuid7(settle, tag), tx='0x' + hx('tx:' + tag), conf=(T + 1200) // 1000 * 1000,
                 first_recv=settle + d['lag'], sg=d['sg'], tq=tq, ti=ti, tys=tys, tot=tot, pt=pt, fee=fee, ft=d['fee'], tk_hash=tk_hash, legs=legs,
                 bad_legs=([(legs[0][0], legs[0][1] // 2)] + legs[1:]) if d['bad_first'] else None,
                 kinds=[match_kind(tq, ti, o['quote'], o['iset']) for o, _ in legs], px_mismatch=d['px_exec'] is not None)
        r['good_from'] = r['first_recv'] + 1 if d['bad_first'] else None
        s.recs.append(r)
        if d['pend']:
            s.pend.append((T, T + s.PEND_MS, 'bids' if tys == 'bid' else 'asks', round(tyl, 2), round(tot / C.E18 * tyl, 6)))

    def book_at(s, u):
        lv, n = Counter(), 0
        for o in s.orders.values():
            if o['t_post'] is None or o['t_post'] > u or (o['t_out'] is not None and u >= o['t_out']):
                continue
            r_ = s.rem(o, u)
            if r_ > 1e-9:
                lv[(o['ys'], o['pt'])] += r_
                n += 1
        return lv, n

    def frame_lines(s):
        out, s.frame_times = [], []
        if not s.frames_on:
            return out
        for k in range(s.f_lo // s.FRAME_MS, s.f_hi // s.FRAME_MS):
            rel = k * s.FRAME_MS + s.FRAME_OFF
            if any(a <= rel < b for a, b in s.gaps):
                continue
            u = s.W + rel
            lv, n = s.book_at(u)
            P = {'bids': Counter(), 'asks': Counter()}
            for t0, t1, side, p, usd in s.pend:
                if t0 <= u < t1:
                    P[side][p] += usd
            bids = sorted(((p / 100, round(q, 6)) for (sd, p), q in lv.items() if sd == 'bid'), reverse=True)
            asks = sorted((p / 100, round(q, 6)) for (sd, p), q in lv.items() if sd == 'ask')
            raw = ws_raw(s.mid, u, bids, asks, [[p, round(v, 6)] for p, v in sorted(P['bids'].items())],
                         [[p, round(v, 6)] for p, v in sorted(P['asks'].items())], n)
            s.frame_times.append(u)
            for ci, conn in enumerate(['c1'] + (['c2'] if any(a <= rel < b for a, b in s.dual) else [])):
                recv = (u + 126 + k % 5 + 9 * ci) * NS + 271
                out.append((recv, ws_line(recv, conn, raw)))
                if ci == 0 and rel in s.dups:                                   # same connection, same upd, identical raw: 'dup'
                    out.append((recv + 4 * NS, ws_line(recv + 4 * NS, conn, raw)))
        return out

    def rest_lines(s):
        R = sorted(s.recs, key=lambda r: (r['first_recv'], r['settle']))
        if not R:
            return []
        times, pages = sorted({r['first_recv'] for r in R}), []
        for i, T in enumerate(times):
            data = sorted([r for r in R if r['first_recv'] <= T], key=lambda r: (-r['settle'], r['sid']))[:6]
            pages.append((T, data, None))
            if i + 1 == len(times) or times[i + 1] > T + 2600:                 # the next poll returns the same page (byte-identical payload)
                pages.append((T + 2000, data, None))
        pages.append((s.we + 90000, sorted(R, key=lambda r: (-r['settle'], r['sid']))[:100], s.final))   # after resolution: volatile fields changed
        out = []
        for T, data, final in pages:
            mk = market_obj(s.mid, s.ws_s, final)
            recs = [record_json(r, mk, final, r['bad_legs'] is not None and T < r['good_from']) for r in data]
            recv = T * NS + 123457
            out.append((recv, rest_line(recv, s.mid, recs)))
        return out


class RawWriter:
    """Hour files by receipt hour, every path from pf_mt_common.hour_path: a non-consumed hour is refused there (Forbidden, before any
    filesystem access) and its lines are dropped - the guard is exercised; the stems are never printed."""

    def __init__(s):
        s.lines, s.dropped, s.files, s.n_lines = defaultdict(list), Counter(), 0, Counter()

    def add(s, family, recv_ns, line):
        try:
            p = C.hour_path(family, stem_of_ms(recv_ns // NS))
        except C.Forbidden:
            s.dropped[family] += 1
            return
        s.lines[p].append((recv_ns, line))
        s.n_lines[family] += 1

    def ensure(s, family, stem, line):
        p = C.hour_path(family, stem)
        if not s.lines[p]:
            s.lines[p].append((day_ms(stem[:8]) * NS + int(stem[9:]) * 3600 * 10 ** 9 + 10 ** 9, line))

    def flush(s):
        for p, L in s.lines.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, 'wb') as f:
                for _, ln in sorted(L, key=lambda x: x[0]):
                    f.write(ln.encode() + b'\n')
            s.files += 1


# ------------------------------------------------------------------------------------------------------------------ the world
def build_world(tmp_pmmm):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    A = {k: addr(k) for k in ('MA', 'MB', 'MC', 'MD', 'TX', 'TY', 'TZ', 'Q')}
    MA, MB, MC, MD, TX, TY, TZ, Q = (A[k] for k in ('MA', 'MB', 'MC', 'MD', 'TX', 'TY', 'TZ', 'Q'))
    OWN_R = cased(OWN[2:])
    need(OWN_R != OWN and OWN_R.lower() == OWN, 'own signer spelling')
    D23, D24, D25, D26, D27, GT = '20261023', '20261024', '20261025', '20261026', '20261027', '20261108'
    V = []

    def venue(*a, **k):
        v = Venue(*a, **k)
        V.append(v)
        return v

    def bg(v, sb, pb, sa, pa_, size=20):
        v.post('bg_bid', sb, 'bid', pb, size, -90000)
        v.post('bg_ask', sa, 'ask', pa_, size, -90000)

    # ---- Oct 24 BOOK: price-priority exhaustion, worst-price survivors, mint / merge, refill / reprice, taker-and-maker hash, masked drop
    b = venue(D24, 62, 'B24', depth=True, dual=((0, 3000),))
    bg(b, MA, .40, MB, .60, 40)
    b.post('bg3', MD, 'bid', .41, 15, -80000)
    b.post('bg4', MC, 'ask', .59, 15, -80000)
    b.post('b0', MD, 'bid', .49, 2, -50000)
    b.post('a1', MA, 'ask', .52, 10, -60000)
    b.post('a2', MB, 'ask', .52, 6, -40000)
    b.post('a3', MC, 'ask', .53, 8, -30000, no=True)              # NO bid .47 = YES ask .53: a YES buyer mints against it
    b.post('a4', MD, 'ask', .53, 5, -20000)
    b.post('b1', MA, 'bid', .48, 10, -55000)
    b.post('b2', MB, 'bid', .48, 6, -35000, no=True)              # NO ask .52 = YES bid .48: a YES seller merges against it
    b.post('b3', MC, 'bid', .47, 8, -25000)
    b.take('P0', -30000, TX, 'sell', .49, 2, bad_first=True)        # conflict: the first receipt is a non-reconciling variant
    b.take('P1', -15000, TY, 'sell', .48, 1)
    b.take('T1', 5000, TX, 'buy', .52, 4)
    b.take('T2', 15000, TY, 'buy', .53, 14)                         # a1 / a2 strictly better than .53: exhausted by price priority
    b.post('a5', MA, 'ask', .52, 5, 16000)                          # refill
    b.post('a6', MB, 'ask', .51, 4, 20000)                          # reprice
    b.take('T3', 25000, TZ, 'buy', .53, 16, listing='best_first')   # worst .53: a3 + a4; a4 fills again (T4) -> a3 exhausted, a4 had 6 ahead
    b.take('T4', 35000, TX, 'buy', .53, 4)
    b.take('S1', 45000, TY, 'sell', .48, 12)                        # worst .48: b1 + b2 (merge); b2 fills again -> b1 exhausted
    b.take('S2', 55000, TX, 'sell', .47, 6)
    b.post('a8', MA, 'ask', .52, 4, 60000)
    b.post('a7', MD, 'ask', .50, 3, 80000)
    b.take('Q1', 85000, Q, 'buy', .51, 8, rest='q1')                 # taker hash q1 fills 3 @ .50 (price improvement), rests 5 @ .51
    b.take('S3', 95000, TX, 'sell', .51, 2)                         # ... then fills as a maker
    b.post('b4', MD, 'bid', .47, 3, 100000)
    b.take('T5', 105000, TZ, 'buy', .52, 2, pend=False)             # drop masked by the refill a9 and no pending: no evidence (lo unbounded)
    b.post('a9', MA, 'ask', .52, 2, 105040)
    b.post('b5', MA, 'bid', .47, 4, 110000)
    b.take('T6', 125000, TY, 'buy', .52, 4)
    b.post('a10', MB, 'ask', .57, 2, 130000)
    b.post('a11', MC, 'ask', .58, 2, 131000, no=True)
    b.take('T7', 140000, TY, 'buy', .58, 4, px_exec=.58)            # mint + regular; priceExecuted = the limit != maker-weighted .575
    b._push(150000, 'cancel', 'q1')
    b.post('z1', MD, 'ask', .55, 1, 150000)                         # one hash at .55 and .56 (H1 identity conflict), no trade-through
    b.post('z2', MD, 'ask', .56, 1, 158000, h_of='z1')
    b.take('T8', 155000, TZ, 'buy', .55, 1)
    b.take('T9', 165000, TX, 'buy', .56, 1)
    b.take('S4', 170000, TY, 'sell', .47, 7)                        # makers[] lists b4 first: never evidence (b4 keeps 1)
    # ---- Oct 25 BOOK: forced T1 / T2 violations, conflict re-poll, frame gap with a fill inside, dup frame, price mismatch
    c = venue(D25, 110, 'B25', depth=True, gaps=((70000, 73500),), dups=(100130,))
    bg(c, MC, .40, MD, .62, 30)
    c.post('c1', MB, 'ask', .55, 6, -50000)
    c.post('c2', MC, 'ask', .56, 6, -45000)
    c.post('c3', MA, 'bid', .45, 8, -40000)
    c.post('c4', MD, 'bid', .44, 6, -35000, cancel=150000)
    c.take('U1', 10000, TX, 'buy', .56, 4, force=[('c1', 1), ('c2', 3)])     # c1 strictly better, keeps 5 and fills again: T1 violation
    c.take('U2', 20000, TY, 'buy', .55, 3)
    c.take('U2b', 22000, TZ, 'buy', .56, 5)
    c.post('c5', MA, 'ask', .57, 3, 25000)
    c.post('c6', MB, 'ask', .57, 3, 26000)
    c.take('U3', 30000, TX, 'buy', .57, 2, force=[('c5', 1), ('c6', 1)])      # both worst-price makers fill again: T2 violation
    c.take('U4', 40000, TY, 'buy', .57, 1, force=[('c5', 1)])
    c.take('U5', 50000, TZ, 'buy', .57, 1, force=[('c6', 1)])
    c.take('U6', 60000, TX, 'sell', .45, 4, bad_first=True)
    c.take('U7', 71500, TY, 'sell', .45, 2)                          # inside the frame gap
    c.take('U8', 90000, TZ, 'sell', .44, 3)
    c.post('c7', MC, 'ask', .58, 2, 95000)
    c.post('c8', MD, 'ask', .59, 2, 96000, no=True)
    c.take('U9', 110000, TX, 'buy', .59, 6, px_exec=.59)
    # ---- Oct 26 / 27 BOOK: simple
    e = venue(D26, 150, 'B26', depth=True)
    bg(e, MB, .52, MC, .70)
    e.post('d1', MA, 'ask', .62, 5, -30000, cancel=200000)
    e.post('d2', MB, 'bid', .58, 5, -30000)
    e.post('d3', MD, 'ask', .62, 3, -10000)
    e.take('V1', 20000, TX, 'buy', .62, 2)
    e.take('V2', 50000, TY, 'sell', .58, 5)
    e.take('V3', 80000, TZ, 'buy', .62, 4)
    e.take('V4', 120000, TX, 'buy', .62, 1)
    f = venue(D27, 140, 'B27', depth=True)
    f.post('bg_ask', MA, 'ask', .45, 20, -90000)
    f.post('bg_bid', MD, 'bid', .25, 20, -30000)                    # frames before -30 s show an EMPTY bid side (no bid level at all)
    f.post('e1', MC, 'ask', .35, 4, -20000)
    f.post('e2', MD, 'bid', .30, 4, -20000)
    f.take('W1', 30000, TX, 'buy', .35, 4)
    f.take('W2', 60000, TY, 'sell', .30, 2)
    # ---- FILL-only markets (tob, no depth): Q as taker, a self-trade, the tie, the poll event, the censored + sealed Oct 27 23:55 window
    g = venue(D24, 96, 'F24')
    bg(g, MA, .64, MB, .72, 10)
    g.post('f1', MA, 'ask', .70, 5, -20000)
    g.post('f2', MB, 'bid', .66, 5, -20000)
    g.take('X1', 40000, TX, 'buy', .70, 3)
    g.take('X2', 100000, Q, 'sell', .66, 2)
    g.take('X3', 150000, TY, 'buy', .70, 2)
    h = venue(D25, 120, 'F25')
    bg(h, MA, .20, MB, .40, 10)
    h.post('g1', MC, 'ask', .30, 6, -20000)
    h.post('g2', MD, 'bid', .26, 6, -20000)
    h.take('Y1', 30000, MC, 'buy', .30, 2)                          # self-trade (MC takes its own ask)
    h.take('Y2', 90000, TX, 'sell', .26, 3)
    t = venue(D26, 168, 'F26')
    bg(t, MC, .40, MD, .55, 10)
    t.post('h1', MA, 'ask', .48, 5, -20000)
    t.post('h2', MB, 'bid', .44, 5, -20000)
    t.take('Z1', 60000, TX, 'buy', .48, 2, fee='COLLATERAL')
    t.take('Z2', 120000, TY, 'sell', .44, 2)
    k = venue(D27, 60, 'F27a')
    bg(k, MA, .70, MB, .85, 10)
    k.post('i1', MA, 'ask', .80, 5, -20000)
    k.post('i2', MB, 'bid', .76, 5, -20000)
    k.take('Z1', 40000, TX, 'buy', .80, 2)
    k.take('Z2', 80000, TY, 'sell', .76, 3)
    z = venue(D27, 287, 'F27z')
    bg(z, MC, .45, MD, .60, 10)
    z.post('j1', MC, 'ask', .55, 5, -20000)
    z.post('j2', MD, 'bid', .50, 5, -20000)
    z.take('Z1', 60000, TX, 'buy', .55, 2)
    z.take('Z2', 250000, TY, 'sell', .50, 2)
    z.take('Z3', 295000, TZ, 'buy', .55, 1)
    z.take('Z4', 320000, TX, 'buy', .55, 1)                          # received after Oct 27: unread (dropped by the guard)
    # ---- ID day (REST only): lower boundary at 00:00, ranked makers at 12:00, the 23:55 window read through Oct 24 00Z
    i0 = venue(D23, 0, 'I0', frames=False)
    i0.post('k1', MA, 'ask', .55, 5, -120000)
    i0.post('k2', MB, 'bid', .50, 5, -120000)
    i0.take('K0', -60000, TX, 'buy', .55, 1)                         # received on Oct 22: unread
    i0.take('K1', 30000, TY, 'buy', .55, 2)
    i0.take('K2', 100000, TX, 'sell', .50, 3)
    i1 = venue(D23, 144, 'I1', frames=False)
    i1.post('l1', MA, 'ask', .52, 10, -60000)
    i1.post('l2', Q, 'ask', .52, 4, -50000)
    i1.post('l3', MB, 'ask', .53, 8, -60000)
    i1.post('l4', MC, 'bid', .48, 10, -60000)
    i1.post('l5', MD, 'bid', .47, 6, -60000)
    i1.take('L0', -40000, TZ, 'buy', .52, 1)
    i1.take('L1', 10000, TX, 'buy', .52, 6)
    i1.take('L2', 40000, TY, 'buy', .53, 10)
    i1.take('L3', 80000, TZ, 'sell', .48, 7)
    i1.take('L4', 120000, Q, 'sell', .47, 5)
    i1.take('L5', 200000, TX, 'buy', .53, 3)
    i2 = venue(D23, 287, 'I2', frames=False)
    i2.post('m1', MD, 'ask', .40, 5, -30000)
    i2.post('m2', MA, 'bid', .36, 5, -30000)
    i2.take('N1', 100000, TY, 'buy', .40, 2)
    i2.take('N2', 280000, TX, 'sell', .36, 2, lag=21000)             # first received in Oct 24 00Z (read by the ID day)
    # ---- GT day: our own orders (truth export below); G2 / G5 have no own maker fill (WS skipped); G4 is upper-censored
    g1 = venue(GT, 124, 'G1', depth=True, run='v6')
    bg(g1, MB, .40, MC, .60)
    g1.post('o1', OWN_R, 'bid', .47, 5, -20000, own=dict(cid=101, run='v6'))
    g1.post('m1', MA, 'bid', .47, 4, -10000)
    g1.post('o2', OWN_R, 'ask', .53, 6, -15000, cancel=90000, no=True, own=dict(cid=102, run='v6'))
    g1.post('o3', OWN_R, 'bid', .46, 5, 10000, cancel=60000, own=dict(cid=103, run='v6'))
    g1.post('m2', MB, 'ask', .54, 5, -30000)
    g1.take('X1', 30000, TX, 'sell', .47, 5)
    g1.take('X2', 40000, TY, 'buy', .53, 2)
    g1.take('X3', 70000, TZ, 'sell', .47, 3)
    g1.take('X4', 100000, TX, 'buy', .54, 2)
    g2 = venue(GT, 125, 'G2', run='v6')
    bg(g2, MA, .40, MB, .60)
    g2.post('o4', OWN_R, 'bid', .45, 5, -10000, cancel=50000, own=dict(cid=104, run='v6'))
    g2.post('m3', MC, 'ask', .55, 4, -20000)
    g2.take('Y1', 20000, TY, 'buy', .55, 2)
    g5 = venue(GT, 126, 'G5', run='v6')
    bg(g5, MA, .40, MB, .60)
    g5.post('m6', MD, 'ask', .58, 3, -20000)
    g5.take('Y2', 50000, TX, 'buy', .58, 1)
    g3 = venue(GT, 182, 'G3', depth=True, run='basis')
    bg(g3, MA, .50, MB, .75)
    g3.post('o5', OWN_R, 'bid', .60, 4, -5000, own=dict(cid=105, run='basis'))
    g3.post('m4', MD, 'bid', .60, 3, 10000)
    g3.post('o6', OWN_R, 'ask', .66, 5, -5000, cancel=120000, no=True, own=dict(cid=106, run='basis'))
    g3.take('Z1', 20000, TX, 'sell', .60, 2)
    g3.take('Z2', 40000, TY, 'sell', .60, 2)
    g3.take('Z3', 60000, TZ, 'buy', .66, 1)
    g3.take('Z4', 80000, TX, 'sell', .60, 3)
    g4 = venue(GT, 275, 'G4', depth=True, run='basis')
    bg(g4, MC, .40, MD, .62, 10)
    g4.post('o7', OWN_R, 'bid', .50, 3, -10000, own=dict(cid=107, run='basis'))
    g4.post('m5', MA, 'ask', .56, 4, -20000)
    g4.take('Z5', 120000, TX, 'sell', .50, 3)
    g4.take('Z6', 280000, TY, 'buy', .56, 2)
    g4.take('Z7', 330000, TZ, 'buy', .56, 1)                         # received in hour 23: unread
    for v in V:
        v.simulate()
    # ---- registries: exactly 288 windows per day; the tie; the sealed outcome (Oct 27 23:55, first seen after the day) arises by rule
    tie_mid = t.mid
    for d in (D23, D24, D25, D26, D27):
        d0, rows = day_ms(d) // 1000, []
        for i in range(288):
            ws, mid = d0 + 300 * i, MID0 + 1000 * DAYIX[d] + i
            sp = 70000.0 + 0.25 * i + DAYIX[d]
            ep = sp if mid == tie_mid else sp + (12.5 if (i + DAYIX[d]) % 2 == 0 else -12.5)
            rows.append(dict(mid=mid, slug=f'btc-updown-5m-{ws}', ws=float(ws), we=float(ws + 300), outcome='Up' if ep >= sp else 'Down',
                             outcome_seen_ns=(ws + 320) * 10 ** 9, up_index=1, start_price=sp, end_price=ep, fee_bps=200, spread_thr=0.06,
                             share_thr=10.0, decimal_precision=2))
        p = C.STEP1 / d / 'registry.parquet'
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), p)
        out_of = {r['mid']: r['outcome'] for r in rows}
        for v in V:
            if v.day == d:
                v.final = out_of[v.mid]
    gt_v = [v for v in V if v.day == GT]
    gt_rows = []
    for v in gt_v:
        sp = 70000.0 + v.idx
        gt_rows.append(dict(mid=v.mid, slug=f'btc-updown-5m-{v.ws_s}', ws=float(v.ws_s), we=float(v.ws_s + 300), outcome='Up' if v.idx % 2 == 0 else 'Down',
                            outcome_seen_ns=(v.ws_s + 320) * 10 ** 9, up_index=1, start_price=sp, end_price=sp + (5. if v.idx % 2 == 0 else -5.)))
        v.final = gt_rows[-1]['outcome']
    # ---- BOOK manifest
    book = {d: [v.mid for v in V if v.day == d and v.depth] for d in (D24, D25, D26, D27)}
    man = C.REPO / 'predictfun/reviews/ladder_async_fast_20261122/manifest.json'
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(dict(complete={d: m for d, m in book.items()}, note='synthetic (mt_e2e)'), indent=1))
    # ---- GT export: scope, who, public maker legs, kernel / core orders, kernel fills (bot clock = venue - D_TRUE)
    ex = tmp_pmmm / 'review-exports' / 'live-samewindow-20261108'
    lv = ex / 'live_vs_replay'
    lv.mkdir(parents=True, exist_ok=True)
    (ex / 'research' / 'step1' / GT).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(gt_rows), ex / 'research' / 'step1' / GT / 'registry.parquet')
    scope = dict(windows=[dict(market=v.mid, ws=v.ws_s, run=v.run) for v in gt_v], cutoff_ms=CUTOFF_MS, capture=str(tmp_pmmm / 'capture_never_opened'),
                 note='synthetic (mt_e2e)')
    (ex / 'scope.json').write_text(json.dumps(scope, indent=1))
    (lv / 'who.json').write_text(json.dumps(dict(our_address=OWN)))
    nan, B_ = float('nan'), (lambda x: float(x - D_TRUE))
    K, P, F, M_ = {r: [] for r in RUNS}, {r: [] for r in RUNS}, {r: [] for r in RUNS}, []
    for v in gt_v:
        for r in v.recs:
            if r['first_recv'] < CUTOFF_MS:                     # the export's public legs: records received in the consumed hours
                for o, a in r['legs']:
                    M_.append(dict(mid=v.mid, exec_s=r['conf'] // 1000, tx=r['tx'], maker=o['sg'], maker_hash=o['h'], yes_side=o['ys'], ptick=o['pt'],
                                   shares=a / C.E18))
        for o in v.orders.values():
            if not o['own']:
                continue
            need(o['quote'] == 'Bid' and v.run == o['own']['run'], f'{v.label}:{o["tag"]}: own orders buy a token in their window run')
            r, cid, up = v.run, o['own']['cid'], o['iset'] == 1
            cx = o['why'] == 'cancelled'
            tc = o['t_out'] if cx else None
            K[r].append(dict(cid=cid, market=v.mid, up=up, price=float(o['pt']), size=o['size'], t_place=B_(o['t_post'] - 28), refused=None,
                             t_signed=B_(o['t_post'] - 22), t_accepted=B_(o['t_post'] + 55), t_CancelPendingAudit=B_(tc + 45) if cx else nan,
                             t_UserCancel=B_(tc + 80) if cx else nan))
            P[r].append(dict(market=v.mid, id=cid + 5000, lane=0 if up else 1, tick=int(o['pt']), size=o['size'], t_req=B_(o['t_post'] - 32),
                             t_Accepted=B_(o['t_post'] + 55), t_CancelDone=B_(tc + 45) if cx else nan, t_Rejected=nan, t_cancel=B_(tc - 30) if cx else nan,
                             filled=float(sum(q for _, q, _ in o['fills'])), cid=cid, refused=None, up=up))
            for T, q, rtag in o['fills']:
                rc_ = v.rec_by_tag[rtag]
                F[r].append(dict(cid=cid, at=B_(T + 160), match_ms=float(T // 1000 * 1000), size=q, price=o['px'], status='Matched'))
                F[r].append(dict(cid=cid, at=B_(rc_['conf'] + 240), match_ms=float(rc_['conf']), size=q, price=o['px'], status='Confirmed'))
    for r in RUNS:
        pd.DataFrame(K[r]).to_parquet(lv / f'kernel_orders_{r}.parquet', index=False)
        pd.DataFrame(P[r]).to_parquet(lv / f'core_orders_{r}.parquet', index=False)
        pd.DataFrame(F[r]).to_parquet(lv / f'kernel_fills_{r}.parquet', index=False)
    pd.DataFrame(M_).to_parquet(lv / 'matches_20261108.parquet', index=False)
    # ---- raw hour files (every planned hour present; unread hours refused by the guard)
    RW = RawWriter()
    for v in V:
        for recv, ln in v.rest_lines():
            RW.add('rest', recv, ln)
        for recv, ln in v.frame_lines():
            RW.add('predict_ws', recv, ln)
    for d in (D24, D25, D26, D27, GT):                    # noise: a 503 page, a page and frames of a market of no population
        v0 = next(v for v in V if v.day == d)
        RW.add('rest', (v0.W + 150000) * NS + 9, rest_line((v0.W + 150000) * NS + 9, v0.mid, [], status=503))
        RW.add('rest', (v0.W + 151000) * NS + 9, rest_line((v0.W + 151000) * NS + 9, OTHER_MID, [dict(market=dict(id=OTHER_MID), settlementId='x')]))
        RW.add('predict_ws', (v0.W + 152000) * NS + 9, ws_line((v0.W + 152000) * NS + 9, 'c1', ws_raw(OTHER_MID, v0.W + 151900, [(0.5, 1.0)], [(0.6, 1.0)], [], [], 2)))
    RW.add('events', (t.W + 200000) * NS + 5, ev_line((t.W + 200000) * NS + 5, 'MATCH_POLL_INCOMPLETE', market_id=t.mid, reason='HTTP_UNAVAILABLE'))
    for d in (D23, D24, D25, D26, D27, GT):
        for stem in C.day_hours(d, 1, 1)[0]:
            t0 = day_ms(stem[:8]) + int(stem[9:]) * 3600000 + 1000
            RW.ensure('rest', stem, rest_line(t0 * NS, OTHER_MID, [], path='/v1/markets'))
            RW.ensure('events', stem, ev_line(t0 * NS, 'COLLECTOR_HEARTBEAT'))
        if d != D23:
            for stem in C.day_hours(d)[0]:
                t0 = day_ms(stem[:8]) + int(stem[9:]) * 3600000 + 1000
                RW.ensure('predict_ws', stem, ws_line(t0 * NS, 'c1', json.dumps(dict(type='H', topic='heartbeat'))))
    RW.flush()
    expect_depth = {d: sorted(v.mid for v in V if v.day == d and v.depth) for d in (D24, D25, D26, D27)}
    expect_depth[GT] = sorted(v.mid for v in gt_v if any(o['own'] and o['fills'] for o in v.orders.values()))
    need(expect_depth[GT] == sorted(v.mid for v in gt_v if v.depth), 'GT depth markets = own maker fills')
    summ = dict(markets={v.label: dict(day=v.day, mid=v.mid, ws=iso(v.W), records=len(v.recs), orders=len(v.orders), depth=v.depth) for v in V},
                records=sum(len(v.recs) for v in V), raw_files=RW.files, raw_lines=dict(RW.n_lines),
                lines_refused_by_hour_guard=dict(RW.dropped), own_orders=sum(len(x) for x in K.values()),
                own_fill_notices=sum(len(x) for x in F.values()), export_public_legs=len(M_), book=book, expect_depth=expect_depth)
    return dict(V=V, summary=summ, expect_depth=expect_depth, tie_mid=tie_mid, sealed_mid=z.mid, censored=dict(lower=(D23, i0.mid), upper=(D27, z.mid), gt=(GT, g4.mid)),
                addrs=A, own_r=OWN_R)


# --------------------------------------------------------------------------------------------------------------------- stages
def locate(tb, H):
    files = {str(p): n for n, p in MODULES}
    where = ''
    for fr, ln in traceback.walk_tb(tb):
        if fr.f_code.co_filename in files:
            where = f'{files[fr.f_code.co_filename]}:{ln} {fr.f_code.co_name}'
    return where


def run_stage(H, label, mod, argv):
    t0, c0, ch0 = time.time(), time.process_time(), resource.getrusage(resource.RUSAGE_CHILDREN)
    logp = H.logs / (re.sub(r'[^A-Za-z0-9_.-]', '_', label) + '.log')
    rc, err, where, argv0 = 0, '', '', sys.argv
    FW.stage = label
    with open(logp, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        sys.argv = list(argv)
        try:
            exec(H.mains[mod], H.mods[mod].__dict__)
        except SystemExit as e:
            rc = 0 if e.code is None else (e.code if isinstance(e.code, int) else 1)
            if e.code is not None and not isinstance(e.code, int):
                err = str(e.code)
        except Exception as e:  # noqa: BLE001 - a builder failure is a result, never a harness crash
            rc, err, where = 1, f'{type(e).__name__}: {e}', locate(e.__traceback__, H)
            traceback.print_exc(file=f)
        finally:
            sys.argv = argv0
            FW.stage = 'harness'
    ch1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    st = dict(stage=label, module=mod, rc=rc, cpu_s=round(time.process_time() - c0, 2), child_cpu_s=round(ch1.ru_utime + ch1.ru_stime - ch0.ru_utime - ch0.ru_stime, 2),
              wall_s=round(time.time() - t0, 2))
    if rc:
        tail = logp.read_text(errors='replace').strip().splitlines()[-6:]
        st.update(error=sanitize(err)[:500], where=where, log_tail=[sanitize(x)[:300] for x in tail])
    H.stages.append(st)
    print(f'[e2e] {label}: rc={rc} cpu={st["cpu_s"]}s child_cpu={st["child_cpu_s"]}s' + (f' ERROR {st.get("error", "")[:240]}' if rc else ''), flush=True)
    return rc == 0


def guard_test(W):
    """pf_mt_common.hour_path refuses a non-consumed hour BEFORE any filesystem access (zero recorded events), for every family."""
    bad = (dt.date(2026, 10, 27) + dt.timedelta(days=1)).strftime('%Y%m%d') + '_00'        # computed, never printed
    out = {}
    for fam in ('rest', 'predict_ws', 'events'):
        for stem, lab in ((bad, 'first_hour_after_oct27'), ('20261108_23', 'nov8_hour_23')):
            FW.rec = []
            try:
                C.hour_path(fam, stem)
                res = 'NOT_REFUSED'
            except C.Forbidden as e:
                res = 'refused' if 'not on a consumed day' in str(e) else 'refused_other_reason'
            finally:
                n, FW.rec = len(FW.rec), None
            out[f'{fam}:{lab}'] = dict(result=res, fs_events=n)
    ok_path = C.hour_path('rest', '20261024_05')
    out['allowed_hour_in_temp'] = str(ok_path).startswith(FW.tmp)
    out['lines_refused_while_writing'] = W['summary']['lines_refused_by_hour_guard']
    out['ok'] = (all(v['result'] == 'refused' and v['fs_events'] == 0 for k, v in out.items() if isinstance(v, dict) and 'result' in v)
                 and out['allowed_hour_in_temp'] and all(out['lines_refused_while_writing'].get(f, 0) > 0 for f in ('rest', 'predict_ws')))
    return out


# ------------------------------------------------------------------------------------------------------------- checker diagnostics
CAPT = dict(on=False, lab=None, n_h3=0, n_pe=0, data=defaultdict(dict))


def instrument_checker(M):
    """Record (never change) the checker's own per-market H3 scans and price-priority evidence, to name the rows behind a failed check."""
    mc0, h30, pe0 = M._market_checks, M.h3_scan, M.priority_evidence

    def mc(A, E, day, mid, *a, **k):
        CAPT.update(lab=(day, int(mid)), n_h3=0, n_pe=0)
        try:
            return mc0(A, E, day, mid, *a, **k)
        finally:
            CAPT['lab'] = None

    def h3(*a, **k):                      # pass-through: the checker's signature may grow (e2e 8: h3_scan gained Uall)
        r = h30(*a, **k)
        if CAPT['on'] and CAPT['lab']:
            try:                                  # recording never alters or breaks the checker's call
                CAPT['data'][CAPT['lab']][('strict', 'permissive')[CAPT['n_h3']] if CAPT['n_h3'] < 2 else 'h3_x'] = dict(cap=r[2], zero=r[3], viol=r[0])
            except (IndexError, TypeError, KeyError):
                pass
            CAPT['n_h3'] += 1
        return r

    def pe(*a, **k):
        r = pe0(*a, **k)
        if CAPT['on'] and CAPT['lab']:
            CAPT['data'][CAPT['lab']][('req', 'ok')[CAPT['n_pe']] if CAPT['n_pe'] < 2 else 'pe_x'] = r
            CAPT['n_pe'] += 1
        return r
    M._market_checks, M.h3_scan, M.priority_evidence = mc, h3, pe


def checker_diag(W):
    """Per depth market: the ok lifecycle rows that the checker's posting-capacity / sole-exhaustion rules refuse, with the checker's own
    bound / evidence and the simulator's truth; the lifecycle's T1 / T2 report beside the checker's recount."""
    import pandas as pd
    out = {}
    for v in W['V']:
        d = CAPT['data'].get((v.day, v.mid))
        po = C.OUT / 'life' / f'{v.day}_{v.mid}_orders.parquet'
        if not v.depth or not d or not po.exists() or 'permissive' not in d:
            continue
        rel = (lambda x, v=v: None if x is None or x < 0 else int(x - v.W))
        capp = {k[0]: x for k, x in d['permissive']['cap'].items()}
        capd = {k[0]: x for k, x in d.get('strict', {}).get('cap', {}).items()}
        zh = {k[0]: x for k, x in d['permissive']['zero'].items()}
        evh = set()
        for pe_ in (d.get('req'), d.get('ok')):
            for sid, i in (pe_ or {}).get('ev', ()):
                r_ = v.rec_by_sid.get(sid)
                if r_ is not None and 0 <= int(i) < len(r_['legs']):
                    evh.add(r_['legs'][int(i)][0]['h'])
        rows = []
        for r in pd.read_parquet(po).itertuples(index=False):
            if r.status != 'ok':
                continue
            h = str(r.mk_hash).lower()
            o = v.by_hash.get(h)
            why, rs = [], {x for x in str(r.removal_set).split('|') if x}
            if r.post_method == 'capacity' and r.post_lo_ms != capp.get(h):
                why.append(f'post_lo {rel(r.post_lo_ms)} (capacity) vs checker capacity {rel(capp.get(h))} / strict {rel(capd.get(h))}')
            if rs == {'exhausted'} and not ((r.scenario in ('S_FIFO', 'S_FIFO_LO') and h in evh) or zh.get(h, False)):
                why.append(f'sole exhausted ({r.removal_method}, {r.removal_class}) without checker price-priority / shown-zero evidence')
            if why:
                rows.append(dict(order=o['tag'] if o else h[:12], scenario=r.scenario, why=why, size_exact=None if r.size_exact != r.size_exact else float(r.size_exact),
                                 truth=dict(post=o['t_post'] - v.W, removal=o['why'], at=rel(o['t_out']), size=o['size']) if o else None))
        lj = C.OUT / 'life' / f'{v.day}_{v.mid}.json'
        pt = json.loads(lj.read_text()).get('priority_tests', {}).get('S_FIFO', {}) if lj.exists() else {}
        out[v.label] = dict(rows=rows, T1_T2_lifecycle={k: {x: pt.get(k, {}).get(x) for x in ('makers', 'groups', 'violations', 'violations_certain')}
                                                         for k in ('T1_price', 'T2_time')},
                            T1_T2_checker={k: (d.get('req') or {}).get(k) for k in ('t1_legs', 't1_hashes', 't2')})
    return out


def extra_diag(W):
    """Evidence rows for the integration findings (names, never verdicts): fills with usable frames strictly between f_before and f_after,
    econ maker_market rows whose residual is not drift + resolution, taker rows with a priceExecuted gap, the ex-ante sets vs Oct 23
    maker / taker shares, the report's readings."""
    import numpy as np
    import pandas as pd
    out, lab = {}, {v.mid: v for v in W['V']}
    rows = []
    for v in W['V']:
        pf, pr = C.OUT / 'life' / f'{v.day}_{v.mid}_fills.parquet', C.OUT / 'depth' / f'{v.day}_{v.mid}_frames.parquet'
        if not (v.depth and pf.exists() and pr.exists()):
            continue
        F, FR = pd.read_parquet(pf), pd.read_parquet(pr)
        up, T = dict(zip(FR.fid, FR.upd_ms)), np.sort(FR[FR.status == 'ok'].upd_ms.to_numpy(np.int64))
        for f in F[F.lo_ms >= 0].itertuples(index=False):
            tb, ta = up.get(f.f_before), up.get(f.f_after)
            if tb is None or ta is None:
                continue
            n = int(np.searchsorted(T, ta, 'left') - np.searchsorted(T, tb, 'right'))
            if n > 0:
                ev, rec = getattr(f, 'x_evidence_fid', -1), v.rec_by_sid.get(f.sid)
                rows.append(dict(market=v.label, record=rec['rtag'] if rec else None, mk_idx=int(f.mk_idx), t_method=f.t_method, hi=int(f.hi_ms - v.W),
                                 lo=int(f.lo_ms - v.W), f_before=int(tb - v.W), f_after=int(ta - v.W), evidence=int(up[ev] - v.W) if ev in up else None,
                                 ok_frames_between=n, true_match=rec['T'] - v.W if rec else None))
    out['fills_with_frames_between_f_before_and_f_after'] = rows
    mm = C.OUT / 'econ' / 'maker_market.parquet'
    if mm.exists():
        M = pd.read_parquet(mm)
        M = M[M.status == 'ok']
        bad = M[~np.isclose(M.resid_res, M.resid_drift + M.resid_resol, atol=1e-6, rtol=1e-9)]
        out['maker_market_ok_rows_resid_ne_drift_plus_resol'] = {f'{d} fifo_status={fs} end_status={es} drift_nan={dn}': int(n) for (d, fs, es, dn), n in
                                                                bad.assign(dn=bad.resid_drift.isna()).groupby(['day', 'fifo_status', 'end_status', 'dn']).size().items()}
        out['maker_market_ok_rows_by_day'] = {str(k): int(n) for k, n in M.groupby('day').size().items()}
    fv = C.OUT / 'econ' / 'fills_valued.parquet'
    if fv.exists():
        FV = pd.read_parquet(fv, columns=['mid', 'sid', 'role', 'px_gap', 'fee_type', 'fee', 'fee_fill', 'p', 'status'])
        g = FV[(FV.role == 'taker') & (FV.px_gap.abs() > 1e-9)]
        out['taker_rows_with_px_gap'] = [dict(market=lab[m].label if m in lab else int(m), record=(lab[m].rec_by_sid.get(sd) or {}).get('rtag') if m in lab else None,
                                              px_gap=round(float(x), 6), p=float(pp), fee_type=ft, fee=round(float(fe), 8), fee_fill=round(float(ff), 8), status=st)
                                         for m, sd, x, pp, ft, fe, ff, st in zip(g.mid, g.sid, g.px_gap, g.p, g.fee_type, g.fee, g.fee_fill, g.status)]
    ip = C.OUT / 'econ' / 'identities.json'
    if ip.exists():
        J = json.loads(ip.read_text()).get('ex_ante', {})
        tags = {a.lower(): k for k, a in W['addrs'].items()}
        L, R = pd.read_parquet(C.OUT / f'legs_{C.ID_DAY}.parquet'), pd.read_parquet(C.OUT / f'records_{C.ID_DAY}.parquet')
        okr = R[R.status == 'ok']
        okl = L[L.sid.isin(okr.sid) & (L.status == 'ok')]
        mk, tk = okl.groupby(okl.mk_signer.str.lower()).shares.sum(), okr.groupby(okr.tk_signer.str.lower()).amount_filled.sum()
        ad = lambda xs: [str(x.get('addr') if isinstance(x, dict) else x).lower() for x in (xs or [])]  # noqa: E731
        sh, pn = ad(J.get('top10_shares')), ad(J.get('top10_pnl'))
        out['exante'] = dict(top_shares=[tags.get(a, a[:8]) for a in sh], top_pnl=[tags.get(a, a[:8]) for a in pn], pnl_info=J.get('top10_pnl_info'),
                             makers_not_in_top_pnl={tags.get(a, a[:8]): dict(maker_sh=float(mk.get(a, 0.)), taker_sh=float(tk.get(a, 0.))) for a in sh if a not in pn})
    rd = C.RECEIPT / 'outputs' / 'readings.json'
    if rd.exists():
        Jr = json.loads(rd.read_text())
        found = []

        def walk(o, path=''):
            if isinstance(o, dict):
                for k, x in o.items():
                    if k == 'reading' and isinstance(x, (str, int, float, bool)):
                        found.append(f'{path}.reading={x}')
                    elif len(found) < 12:
                        walk(x, f'{path}.{k}')
        walk(Jr)
        out['report'] = dict(outputs=sorted(x.name for x in (C.RECEIPT / 'outputs').iterdir())[:30], readings=found[:12])
    return out


# ------------------------------------------------------------------------------------------------------------- oracle / features
def oracle(W):
    """Every lifecycle claim of the depth markets vs the simulator's truth. observed / constrained claims must hold (else a defect);
    assumption_sensitive misses are listed (in a FIFO world the S_FIFO claims should hold too, except on the forced-violation orders)."""
    import numpy as np
    import pandas as pd
    res, cnt = dict(markets={}, defects=[], sensitive_misses=[], unknown_hashes=0, unknown_sids=0), defaultdict(Counter)
    for v in W['V']:
        if not v.depth:
            continue
        po, pf = C.OUT / 'life' / f'{v.day}_{v.mid}_orders.parquet', C.OUT / 'life' / f'{v.day}_{v.mid}_fills.parquet'
        if not (po.exists() and pf.exists()):
            res['markets'][v.label] = 'lifecycle outputs missing'
            continue
        O, F = pd.read_parquet(po), pd.read_parquet(pf)
        res['markets'][v.label] = dict(order_rows=len(O), fill_rows=len(F))
        rel = (lambda x, v=v: None if x is None or x < 0 else int(x - v.W))

        def note(field, cls, ok, scen, h, detail, v=v):
            cnt[field][f'{cls}:{"ok" if ok else "fail"}'] += 1
            if not ok:
                o = v.by_hash.get(h)
                item = dict(market=v.label, scenario=scen, field=field, cls=cls, order=o['tag'] if o else h[:14], detail=detail)
                (res['defects'] if cls in ('observed', 'constrained') else res['sensitive_misses']).append(item)
        for r in O.itertuples(index=False):
            h = str(r.mk_hash).lower()
            if h in v.dup_hashes:                                           # an identity conflict: one hash, two orders - no single truth
                res['identity_conflict_rows_skipped'] = res.get('identity_conflict_rows_skipped', 0) + 1
                continue
            o = v.by_hash.get(h)
            if o is None:
                res['unknown_hashes'] += 1
                continue
            tp, tr = o['t_post'], o['t_out']
            cause = 'exhausted' if o['why'] == 'exhausted' else 'cancelled_or_expired' if o['why'] == 'cancelled' else 'end_censored'
            a, b = int(r.post_lo_ms), int(r.post_hi_ms)
            note('posting', r.post_class, (a < 0 or a <= tp) and (b < 0 or tp <= b), r.scenario, h, f'true post {tp - v.W} not in [{rel(a)}, {rel(b)}]')
            note('size_lb', r.size_class, float(r.size_lb) <= o['size'] + 1e-9, r.scenario, h, f'size_lb {r.size_lb} > true {o["size"]}')
            if np.isfinite(r.size_exact):
                note('size_exact', r.size_class, abs(float(r.size_exact) - o['size']) <= 1e-6, r.scenario, h, f'size_exact {r.size_exact} != true {o["size"]}')
            rs = {x for x in str(r.removal_set).split('|') if x}
            note('removal_set', r.removal_class, cause in rs, r.scenario, h, f'true {cause} not in {sorted(rs)}')
            if tr is not None:
                a, b = int(r.removal_lo_ms), int(r.removal_hi_ms)
                note('removal_window', r.removal_class, (a < 0 or a <= tr) and (b < 0 or tr <= b), r.scenario, h,
                     f'true removal {tr - v.W} ({cause}) not in [{rel(a)}, {rel(b)}]')
            a, b = int(r.persist_lo_ms), int(r.persist_hi_ms)
            if a >= 0 and b >= 0:
                note('persistence', r.persist_class, tp <= a and (tr is None or b <= tr), r.scenario, h,
                     f'presence [{rel(a)}, {rel(b)}] outside the life [{tp - v.W}, {rel(tr)}]')
            if o['ahead_first'] is not None and h not in v.forced:
                lq, hq, x = float(r.ahead_lo), float(r.ahead_hi), o['ahead_first']
                note('queue', r.queue_class, (not np.isfinite(lq) or lq <= x + 1e-6) and (not np.isfinite(hq) or x <= hq + 1e-6), r.scenario, h,
                     f'true ahead {x} not in [{lq}, {hq}]')
        for fr in F.itertuples(index=False):
            if str(fr.mk_hash).lower() in v.dup_hashes:
                continue
            rec = v.rec_by_sid.get(fr.sid)
            if rec is None:
                res['unknown_sids'] += 1
                continue
            a, b = int(getattr(fr, 'x_lo_hard_ms', fr.lo_ms)), int(fr.hi_ms)      # the observed window: [x_lo_hard_ms, hi] (lo_ms refines it)
            note('fill_window', 'observed', (a < 0 or a <= rec['T']) and rec['T'] <= b, '-', str(fr.mk_hash).lower(),
                 f'{rec["rtag"]}: match {rec["T"] - v.W} not in [{rel(a)}, {b - v.W}]')
            a = int(fr.lo_ms)
            note('fill_lo_refinement', 'assumption_sensitive', a < 0 or a <= rec['T'], '-', str(fr.mk_hash).lower(),
                 f'{rec["rtag"]}: match {rec["T"] - v.W} before lo_ms {rel(a)} ({fr.t_method})')
    res['counts'] = {k: dict(sorted(c.items())) for k, c in sorted(cnt.items())}
    return res


def features(W, H):
    """Computed predicates: the synthetic world reached the builders' outputs with every feature it must exercise."""
    import pandas as pd
    out, OUT = {}, C.OUT

    def rd(p, cols=None):
        return pd.read_parquet(p, columns=cols) if p.exists() else None
    FG = ('20261024', '20261025', '20261026', '20261027', '20261108')
    L = [x for x in (rd(OUT / f'legs_{d}.parquet') for d in ('20261023',) + FG) if x is not None]
    R = [x for x in (rd(OUT / f'records_{d}.parquet') for d in ('20261023',) + FG) if x is not None]
    if not L or not R:
        return dict(error='records / legs missing')
    L, R = pd.concat(L, ignore_index=True), pd.concat(R, ignore_index=True)
    Lf = L[L.day.isin(FG)]
    out['multi_price_records'] = int((Lf.groupby('sid').y_ptick.nunique() >= 2).sum())
    out['mint_legs'] = int((L.x_match == 'mint').sum()) if 'x_match' in L else int(sum(k == 'mint' for v in W['V'] for r in v.recs for k in r['kinds']))
    out['merge_legs'] = int((L.x_match == 'merge').sum()) if 'x_match' in L else -1
    out['conflict_records'] = int(R.status.str.contains('conflict').sum())
    out['repolled_records'] = int((R.n_recv >= 2).sum())
    out['taker_addresses_also_makers'] = len(set(R.tk_signer.str.lower()) & set(L.mk_signer.str.lower()))
    out['taker_hashes_also_makers'] = len(set(R.tk_hash) & set(L.mk_hash))
    cz = {}
    for k, (d, m) in W['censored'].items():
        M = rd(OUT / f'markets_{d}.parquet', ['mid', 'status', 'outcome', 'n_records'])
        cz[k] = None if M is None else bool(M[M.mid == m].status.str.contains('boundary_censored').any())
        if k == 'upper' and M is not None:
            out['sealed_outcome_blank'] = bool((M[M.mid == m].outcome == '').all())
    out['boundary_censored'] = cz
    gaps = 0
    for p in sorted((OUT / 'depth').glob('*_frames.parquet')) if (OUT / 'depth').exists() else []:
        gaps += int(pd.read_parquet(p, columns=['status']).status.str.contains('gap').sum())
    out['frames_with_gap_flag'] = gaps
    O = [rd(p) for p in sorted((OUT / 'life').glob('*_orders.parquet'))] if (OUT / 'life').exists() else []
    if O:
        O = pd.concat(O, ignore_index=True)
        fifo = O[O.scenario == 'S_FIFO']
        out['fifo_price_exhaustions'] = int(fifo.removal_method.astype(str).str.contains('fifo_price').sum())
        out['fifo_time_exhaustions'] = int(fifo.removal_method.astype(str).str.contains('fifo_time').sum())
        out['survivors_with_ahead_lo'] = int((fifo.ahead_lo > 0).sum())
        F = pd.concat([rd(p) for p in sorted((OUT / 'life').glob('*_fills.parquet'))], ignore_index=True)
        out['fills_lo_unbounded'] = int((F.lo_ms < 0).sum())
        out['identity_conflict_orders'] = int(fifo.status.astype(str).str.contains('identity_conflict').sum())
    tb = OUT / 'tob_20261027.parquet'
    if tb.exists():
        TB = pd.read_parquet(tb, columns=['mid', 'n_bid', 'best_bid_sh'])
        e_ = TB[(TB.mid == next(v.mid for v in W['V'] if v.label == 'B27')) & (TB.n_bid == 0)]
        out['empty_bid_side_frames'] = int(len(e_))
        out['empty_bid_side_size_not_zero'] = int((e_.best_bid_sh != 0.0).sum())
    T = None
    for p in (OUT / 'truth' / 'v2' / 'own_orders.parquet', OUT / 'truth' / 'own_orders.parquet'):
        if p.exists():
            T = pd.read_parquet(p, columns=['mk_hash', 'status', 'mid'])
            break
    if T is not None:
        ok = T[(T.status == 'ok') & (T.mk_hash != '')]
        out['truth_scored_own_orders'] = int(len(ok))
        out['own_hashes_in_gt_lifecycle'] = int(ok.mk_hash.isin(set(O.mk_hash)).sum()) if len(O) else 0
    fv = OUT / 'econ' / 'fills_valued.parquet'
    if fv.exists():
        FV = pd.read_parquet(fv, columns=['mid', 'V', 'status'])
        out['econ_tie_rows'] = int(((FV.mid == W['tie_mid']) & (FV.V == 0.5)).sum())
        out['econ_ok_rows_on_censored'] = int(((FV.mid == W['sealed_mid']) & (FV.status == 'ok')).sum())
    need_pos = ('multi_price_records', 'mint_legs', 'merge_legs', 'conflict_records', 'repolled_records', 'taker_addresses_also_makers',
                'taker_hashes_also_makers', 'frames_with_gap_flag', 'fifo_price_exhaustions', 'survivors_with_ahead_lo', 'fills_lo_unbounded',
                'identity_conflict_orders', 'empty_bid_side_frames',
                'truth_scored_own_orders', 'own_hashes_in_gt_lifecycle', 'econ_tie_rows')
    bad = [k for k in need_pos if not (isinstance(out.get(k), int) and out[k] > 0)]
    bad += [f'boundary_censored.{k}' for k, x in cz.items() if x is not True]
    bad += [k for k in ('sealed_outcome_blank',) if out.get(k) is not True]
    bad += [k for k in ('econ_ok_rows_on_censored', 'empty_bid_side_size_not_zero') if out.get(k) not in (0,)]
    out['not_exercised'] = bad
    return out


# ------------------------------------------------------------------------------------------------------------------------- main
def rehearse(tmp, args, S, FAIL):
    global C, FW
    FW = Firewall(tmp)
    sys.addaudithook(_audit)
    _wrap_io()
    FW.on = True
    tmp_repo, tmp_pmmm = tmp / 'repo', tmp / 'pmmm-data'
    H = load_all(tmp_repo, tmp_pmmm, S, FAIL)
    if H is None:
        return
    C = H.C
    (tmp / 'systmp').mkdir()
    tempfile.tempdir = str(tmp / 'systmp')
    H.logs = tmp / 'logs'
    H.logs.mkdir()
    W = build_world(tmp_pmmm)
    S['world'] = W['summary']
    S['hour_guard'] = g = guard_test(W)
    if not g['ok']:
        FAIL.append(dict(module='pf_mt_common', check='hour_path guard', evidence=json.dumps(g)[:600]))
    # ---- (3) the builders, production order
    for d in (C.ID_DAY,) + tuple(C.MEASURE_DAYS) + (C.GT_DAY,):
        run_stage(H, f'extract {d}', 'pf_mt_extract', ['pf_mt_extract.py', 'day', d])
    depth = {}
    for d in tuple(C.MEASURE_DAYS) + (C.GT_DAY,):
        p = C.OUT / f'extract_{d}.json'
        if p.exists():
            rec = json.loads(p.read_text())
            depth[d] = sorted(int(m) for m in rec.get('population', {}).get('depth_mids', []))
            files = sorted(int(x.name.split('_')[1]) for x in (C.OUT / 'depth').glob(f'{d}_*_frames.parquet'))
            if depth[d] != W['expect_depth'][d] or files != W['expect_depth'][d]:
                FAIL.append(dict(module='pf_mt_extract', check=f'depth population {d}',
                                 evidence=f'receipt depth_mids {depth[d]} / depth files {files} != expected {W["expect_depth"][d]} (BOOK manifest / GT own-fill rule)'))
    for d, mids in depth.items():
        for m in mids:
            run_stage(H, f'lifecycle {d} {m}', 'pf_mt_lifecycle', ['pf_mt_lifecycle.py', 'market', d, str(m)])
    for sub in ('build', 'dry', 'score'):
        run_stage(H, f'truth {sub}', 'pf_mt_truth', ['pf_mt_truth.py', sub])
    econ_sha0 = hashlib.sha256((RESEARCH / 'pf_mt_econ.py').read_bytes()).hexdigest()[:16]
    run_stage(H, 'econ run', 'pf_mt_econ', ['pf_mt_econ.py', 'run'])
    run_stage(H, 'report', 'pf_mt_report', ['pf_mt_report.py'])           # writes RECEIPT/outputs (the temp receipt dir)
    if econ_sha0 != S['sources_sha256_16']['pf_mt_econ']:
        S['warning_econ_worker_source_differs'] = 'pf_mt_econ.py on disk (run by the day workers) differs from the snapshot the parent executed'
    for st in H.stages:
        if st['rc']:
            FAIL.append(dict(module=st['module'], check=st['stage'], evidence=st.get('error', ''), where=st.get('where', ''), log_tail=st.get('log_tail', [])))
    # ---- (4) the checker: its CLI 'run', then the falsifier
    try:
        instrument_checker(H.mods['mt_checks'])
        CAPT['on'] = True
    except AttributeError as e:                                               # diagnostics only: never blocks the checker run
        S['checker_diag_unavailable'] = sanitize(str(e))
    run_stage(H, 'checks run', 'mt_checks', ['mt_checks.py', 'run'])
    CAPT['on'] = False
    try:
        S['checker_diag'] = checker_diag(W)
    except Exception as e:  # noqa: BLE001 - diagnostics only
        S['checker_diag'] = dict(error=sanitize(f'{type(e).__name__}: {e}'))
    sc = C.OUT / 'truth' / 'v2' / 'scores.json'
    if sc.exists():
        J = json.loads(sc.read_text())
        S['truth_scores'] = dict(population=J.get('population'), defects_total=J.get('defects_total'), unique_defects=J.get('unique_defects'),
                                 defect_rows=[{k: sanitize(x)[:160] for k, x in r.items()} for r in (J.get('defect_rows') or [])[:6]])
    cj = sorted(C.RECEIPT.glob('checks_*.json')) if C.RECEIPT.exists() else []
    if not cj:
        FAIL.append(dict(module='mt_checks', check='run', evidence=f'no checks_*.json written: {H.stages[-1].get("error", "")}'))
    else:
        res = json.loads(cj[-1].read_text())
        ck, det = res.get('checks', {}), res.get('details', {})
        failed = {n: det.get(n, {}) for n, v in ck.items() if not v}
        rp = {f'reparse {k}:{n}': r.get('details', {}).get(n, {}) for k, r in (res.get('reparse') or {}).items() for n, v in r.get('checks', {}).items() if not v}
        S['checks'] = dict(n_checks=len(ck), n_passed=sum(1 for v in ck.values() if v), reparse_keys=sorted((res.get('reparse') or {}).keys()),
                           n_reparse_checks=sum(len(r.get('checks', {})) for r in (res.get('reparse') or {}).values()),
                           expected_failures={n: dict(reason=w, failed=n in failed) for n, w in EXPECTED_CHECK_FAILURES.items()},
                           n_eval={n: d.get('n_eval') for n, d in det.items()})
        for n, d in list(failed.items()) + list(rp.items()):
            if n in EXPECTED_CHECK_FAILURES:
                continue
            FAIL.append(dict(module='mt_checks', check=n, evidence=sanitize(d.get('reason') or '')[:500],
                             examples=[sanitize(x)[:300] for x in (d.get('examples') or [])[:3]]))
    if not args.no_falsify:
        FW.stage = 'falsify'
        t0 = time.process_time()
        M = H.mods['mt_checks']
        try:
            with open(H.logs / 'falsify.log', 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                (tmp / 'falsify').mkdir()
                fz = M.falsify(M.load(), base_tmp=str(tmp / 'falsify'))
            nr = {k: v for k, v in fz.items() if v.get('refused') is False}
            npl = {k: v for k, v in fz.items() if v.get('refused') is None}
            S['falsifier'] = dict(cases=len(fz), refused=sum(1 for v in fz.values() if v.get('refused') is True), cpu_s=round(time.process_time() - t0, 1),
                                  not_refused={k: dict(check=v.get('check'), expected=v.get('expected_reason'), got=[sanitize(x)[:240] for x in (v.get('got') or [])[:2]],
                                                       baseline_clean=v.get('baseline_clean')) for k, v in nr.items()},
                                  not_plantable={k: sanitize((v.get('got') or [''])[0])[:200] for k, v in npl.items()})
            for k, v in nr.items():
                FAIL.append(dict(module='mt_checks', check=f'falsifier {k}', evidence=f'not refused by {v.get("check")} (expected {v.get("expected_reason")!r}); '
                                 f'got {[sanitize(x)[:200] for x in (v.get("got") or [])[:2]]}; baseline_clean={v.get("baseline_clean")}'))
            for k, v in npl.items():
                FAIL.append(dict(module='harness fixture', check=f'falsifier {k}', evidence=f'not plantable: {sanitize((v.get("got") or [""])[0])[:200]}'))
        except Exception as e:  # noqa: BLE001
            FAIL.append(dict(module='mt_checks', check='falsify', evidence=sanitize(f'{type(e).__name__}: {e}')[:500], where=locate(e.__traceback__, H)))
        finally:
            FW.stage = 'harness'
    try:
        S['evidence'] = extra_diag(W)
    except Exception as e:  # noqa: BLE001 - diagnostics only
        S['evidence'] = dict(error=sanitize(f'{type(e).__name__}: {e}'))
    # ---- (5) oracle and feature predicates
    S['oracle'] = orc = oracle(W)
    for dfx in orc['defects']:
        FAIL.append(dict(module='pf_mt_lifecycle', check=f'oracle {dfx["field"]} ({dfx["cls"]})', evidence=json.dumps(dfx)))
    if orc['unknown_hashes'] or orc['unknown_sids']:
        FAIL.append(dict(module='pf_mt_lifecycle', check='oracle identity', evidence=f'{orc["unknown_hashes"]} order hashes / {orc["unknown_sids"]} sids unknown to the venue'))
    S['features'] = ft = features(W, H)
    for k in ft.get('not_exercised', []) + (['records / legs missing'] if 'error' in ft else []):
        FAIL.append(dict(module='harness fixture', check=f'feature {k}', evidence=f'not exercised in the builder outputs (value {ft.get(k.split(".")[0])!r})'))
    if FW.violations:
        for v_ in FW.violations[:10]:
            FAIL.append(dict(module=v_['stage'], check='firewall', evidence=json.dumps(v_)))
    S['firewall'] = dict(violations=len(FW.violations), subprocesses=FW.popen[:8])
    S['stages'] = H.stages


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--keep', action='store_true', help='keep the temp dir (outputs and stage logs)')
    ap.add_argument('--tmp', help='a temp dir to create (must not exist); default: a new mkdtemp')
    ap.add_argument('--no-falsify', action='store_true')
    a = ap.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'                                  # the econ day workers inherit it: no __pycache__ writes
    sys.path[:] = [x for x in sys.path if not any(os.path.abspath(x or '.') == str(d) or os.path.abspath(x or '.').startswith(str(d) + '/') for d in DENY)]
    if a.tmp:
        tmp = Path(a.tmp).resolve()
        if tmp.exists():
            sys.exit(f'refused: {tmp} exists')
        tmp.mkdir(parents=True)
    else:
        tmp = Path(tempfile.mkdtemp(prefix='mt_e2e_')).resolve()
    for d in DENY + (REAL_REPO,):
        if str(tmp) == str(d) or str(tmp).startswith(str(d) + '/'):
            sys.exit('refused: the temp dir lies under a real data root')
    S, FAIL, rc = dict(harness='mt_e2e_v1', started_utc=dt.datetime.now(UTC).isoformat(timespec='seconds'), tmp=str(tmp), kept=a.keep), [], 2
    c0 = time.process_time()
    try:
        rehearse(tmp, a, S, FAIL)
        rc = 1 if FAIL else 0
    except Exception as e:  # noqa: BLE001
        FAIL.insert(0, dict(module='harness', check='crash', evidence=sanitize(f'{type(e).__name__}: {e}')[:600]))
        S['harness_traceback'] = sanitize(traceback.format_exc())[-4000:]
        rc = 2
    finally:
        if FW is not None:
            FW.on = False
        ch = resource.getrusage(resource.RUSAGE_CHILDREN)
        S['cpu_s'], S['child_cpu_s'] = round(time.process_time() - c0, 1), round(ch.ru_utime + ch.ru_stime, 1)
        S['n_failures'], S['integration_failures'] = len(FAIL), FAIL
        S['first_failure'] = f'{FAIL[0]["module"]} / {FAIL[0]["check"]}' if FAIL else None
        print('E2E_SUMMARY ' + json.dumps(S, indent=1, default=str))
        print(f'E2E RESULT {"PASS" if rc == 0 else "FAIL"} rc={rc} failures={len(FAIL)} first={S["first_failure"]}', flush=True)
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()
