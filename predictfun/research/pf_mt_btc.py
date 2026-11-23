"""TOP-3 DECISION PATHS (MAKER-TRACE amendment 3) - the BTC information-state inputs (P-2026-005; session aea5bfb9, 2026-11-23).
Contract: predictfun/reviews/maker_trace_20261122/decision_contract.md, section 'Information state at a decision' (binding): BTC is the Binance
USD-M futures bookTicker at its exchange event time E_ms; clocks are never mixed. One measure day (20261024..20261027) -> write-once files in
ROOT/btc/ (ROOT = data/predict_fun/research/maker_trace_v2_20261123, made by tools/setup_run.py; this module creates only btc/ and refuses
when ROOT is missing):
  btc_<day>.parquet  one row per 100 ms grid point of the day on the E_ms clock, g_ms = D0 + 100 k (k = 0..863,999; D0 = the day 00:00:00Z):
                     bid / ask / mid of the LAST bookTicker update with E_ms <= g (equal E_ms -> the later row in receipt order; NaN when none),
                     n_upd = updates with E_ms in (g - 100, g], age_ms = g - E_ms of that update (-1 when none), status (precedence
                     no_data > gap > stale > ok):
                       no_data  no update with E_ms <= g was read;
                       gap      g in [lo, hi) of a coverage hole, where the state or n_upd may miss rows that were not read: a planned hour file
                                missing / unreadable / without a valid row ([min, max + 100) of the E_ms read on either side), a collector start
                                inside the read span (the ledger; same rule), the head before the first file read and the day's tail from the
                                last E_ms read (the next day's 00Z file is never read; the collector flushes every 5 s into the file of the hour
                                current at the flush, so the last 0-5 s of the day's rows sit in that file - measured, counted in the receipt);
                       stale    age_ms > 1000 inside covered data;  ok  otherwise.
  ref_<day>.parquet  the Predict reference relay's Chainlink BTC/USD stream (manifest topic chainlinkAssetPriceUpdate/<id>, unsigned relay), one
                     row per second t_s of the day on the stream's own clock (publishTime, whole seconds): price / price_str (exact decimal) of the
                     last publish with publish_time_s <= t_s (distinct payloads in one second -> the latest received), age_s, n_obs (distinct
                     payloads for t_s), n_recv (PRICE rows incl. duplicates), first_recv_ns (collector clock, -1 unless age_s == 0), status
                     (no_data > gap > stale > conflict > ok; gap = strictly inside a hole between the publishes read). Skipped, and said so in
                     the receipt, when the relay does not carry such a stream.
  btc_<day>.json     the receipt (written last): sha256 + bytes of every hour file read, per-file parser / counts / extents / exact receipt-lag
                     quantiles, the day's receipt-clock lag (recv_ns / 1e6 - E_ms) quantiles, holes, status counts, stale episodes, the era and
                     ledger checks (CLAUDE.md rule 5), the outputs' sha256, CPU per stage, VmHWM.
Hours: the day's 24 hour files plus the previous day's 23Z (for 20261024 that is 20261023_23, consumed and allowed as padding). An allow-list
(20261023_23 + every hour of 20261024..27) is checked on the stem STRING before any filesystem access; nothing else is ever opened, listed
or globbed. One hour file at a time; only day-sized arrays cross hours; VmHWM self-guard pf_mt_common.GUARD_MIB (1,450 MiB; never raised).
Reading: the gzip bytes are hashed, decompressed (one hour) and parsed by pyarrow (needed columns only; csv rows with a wrong field count are
skipped and counted); the first and last 1,000 rows must equal a Python parse of the same lines (else the run raises). Text Arrow refuses (a
header, a non-numeric field, torn json) falls back to a per-line parse (bookTicker: the pf_io.bn_rows rule made strict - 8 fields, digits in
the integer fields), counted. Grid: every row belongs to the one window (g - 100, g] with g = D0 + 100 ceil((E - D0) / 100); per window the
max-(E_ms, receipt order) row and the count go into day arrays, combined over files by max / sum, and the state at g is the running max over
windows (np.maximum.accumulate): exact, independent of file order and of out-of-order rows (both still counted in the receipt).
Usage (run_light.sh): pf_mt_btc.py day <YYYYMMDD> | probe <out_dir outside the run roots> <20261024_HH>
"""
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import zlib
from array import array
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.json as pajson
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import pf_io  # noqa: E402
import pf_mt_common as C  # noqa: E402

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

need, Forbidden = C.need, C.Forbidden
VERSION = 'maker_trace_v2_btc_1'
ROOT_V2 = C.REPO / 'data/predict_fun/research/maker_trace_v2_20261123'
LEDGER = C.REPO / 'data/mm_hf/collector_runs.jsonl'
DAYS = tuple(C.MEASURE_DAYS)                                    # 20261024..20261027
PAD_STEM, PROBE_DAY = '20261023_23', '20261024'
ALLOWED = frozenset((PAD_STEM,) + tuple(f'{d}_{h:02d}' for d in DAYS for h in range(24)))
EXT = {'bookTicker': '.csv.gz', 'reference_prices': '.jsonl.gz'}
GRID_MS, STALE_MS, HOUR_MS, DAY_MS, DAY_S = 100, 1000, 3_600_000, 86_400_000, 86_400
N_GRID = DAY_MS // GRID_MS
STATUS = ('ok', 'stale', 'gap', 'no_data')
REF_STATUS = ('ok', 'conflict', 'stale', 'gap', 'no_data')
HF_WS_V2_NS, HF_WS_V2_SCHEMA = 1787579334881534478, 'hf_ws_v2_recv_boundary'   # CLAUDE.md rule 5: the hf_ws_v2 stamp boundary
BN_COLS = ('recv_ns', 'E_ms', 'T_ms', 'update_id', 'bid', 'bid_qty', 'ask', 'ask_qty')
BN_USE = ('recv_ns', 'E_ms', 'update_id', 'bid', 'ask')
BN_INT = ('recv_ns', 'E_ms', 'update_id')
PARITY_ROWS, RANK_BITS = 1000, 27
LAG_LO_US, LAG_BIN_US, LAG_NB = -10_000_000, 100, 700_000      # day receipt-lag histogram: [-10 s, +60 s) in 0.1 ms bins
QS = (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 0.9999, 1.0)
REF_SCHEMA = pa.schema([('run_id', pa.string()), ('seq', pa.int64()), ('source', pa.string()), ('kind', pa.string()), ('recv_ns', pa.int64()),
                        ('provider_feed_id', pa.string()), ('duplicate_previous_payload', pa.bool_()), ('publish_time_regressed', pa.bool_()),
                        ('price', pa.string()), ('publish_time_s', pa.int64()), ('relay_timestamp_s', pa.int64()), ('relay_feed_id', pa.string()),
                        ('source_time_precision', pa.string())])
REF_F = tuple(REF_SCHEMA.names)
REF_ID = ('run_id', 'source', 'provider_feed_id', 'relay_feed_id', 'source_time_precision')
REF_REQ = ('recv_ns', 'publish_time_s', 'relay_timestamp_s', 'price', 'duplicate_previous_payload', 'publish_time_regressed') + REF_ID
BTC_SCHEMA = pa.schema([('g_ms', pa.int64()), ('bid', pa.float64()), ('ask', pa.float64()), ('mid', pa.float64()), ('n_upd', pa.int64()),
                        ('age_ms', pa.int64()), ('status', pa.string())])
REF_OUT_SCHEMA = pa.schema([('t_s', pa.int64()), ('price', pa.float64()), ('price_str', pa.string()), ('age_s', pa.int64()), ('n_obs', pa.int64()),
                            ('n_recv', pa.int64()), ('first_recv_ns', pa.int64()), ('status', pa.string())])
DEFINITIONS = dict(
    g_ms='grid point on the Binance E_ms clock: D0 + 100 k, k = 0..863,999 (D0 = the day 00:00:00.000Z)',
    bid_ask_mid='of the LAST bookTicker update with E_ms <= g; equal E_ms -> the later row in receipt order (file plan order, then line order); '
                'mid = (bid + ask) / 2; NaN when no update was read',
    n_upd='valid bookTicker rows with E_ms in (g - 100, g]', age_ms='g - E_ms of that update; -1 when none',
    status='precedence no_data > gap > stale > ok. no_data: no update with E_ms <= g was read; gap: g in [lo, hi) of a hole (bn.holes), where '
           'the state or n_upd at g may miss rows that were not read; stale: age_ms > 1000; ok: otherwise',
    holes='before_first_file_read: (-inf, e_min of the first file with a valid row + 100); missing_hour_file: [min, max + 100) of the e_max read '
          'before and the e_min read after planned files that are missing / unreadable / without a valid row; collector_start: the same around a '
          'ledger start inside the read span (max E_ms of rows received before it, min E_ms of rows received at or after it); after_last_row_read: '
          '[e_max of the last file read, +inf) - the next day\'s 00Z file is never read (hour rule) and the collector flushes every 5 s into the '
          'file of the hour current at the flush, so the day\'s last 0-5 s of rows sit in that file',
    rows='valid: 8 fields, integer recv_ns / E_ms / update_id, finite positive bid and ask. rows_pre_day: E_ms <= D0 - 100 (only the latest '
         'matters: the state at D0); rows_in_grid: D0 - 100 < E_ms <= D1 - 100; rows_after_last_g: E_ms > D1 - 100. late_vs_earlier_files: '
         'E_ms below the max E_ms of the earlier files; e_decreases: E_ms steps back in line order (both handled exactly, counted)',
    lag='receipt-clock lag recv_ns / 1e6 - E_ms (recv floored to the microsecond). Per file: exact order statistics (numpy method lower) over all '
        'valid rows. Day: rows with E_ms in [D0, D1); p0 / p100 exact, other quantiles = the lower edge of their 0.1 ms bin over [-10 s, +60 s)',
    ref="Predict's unsigned Chainlink BTC/USD relay: one row per second t_s of the day on the stream's own clock (publish_time_s). price / "
        'price_str: the last publish with publish_time_s <= t_s (several distinct payloads in one second -> the latest received; price_str is '
        'the exact published decimal, price = Python float of it); age_s = t_s - its publish second (-1 none); n_obs: distinct (relay_timestamp_s, '
        'price) payloads for t_s; n_recv: valid PRICE rows (duplicates included) for t_s; first_recv_ns: earliest receipt for t_s (collector '
        'clock; -1 unless age_s == 0)',
    ref_status='precedence no_data > gap > stale > conflict > ok. no_data: no publish <= t_s read; gap: t_s strictly inside a hole (before the '
               'first publish read; between the last publish of a file and the first of the next file read around missing / unreadable / empty '
               'ones; after the last publish read - its receipts would lie in the next day\'s 00Z file); stale: no publish for t_s (age_s >= 1); '
               'conflict: distinct prices published for t_s; ok: one price for t_s',
)


class GuardStop(RuntimeError):
    pass


class Roots:
    """Input / output roots (the selftest points them at synthetic fixtures)."""

    def __init__(self, bn=None, ref_run=None, ledger=None, out_root=None):
        self.bn = Path(pf_io.BN_DIR if bn is None else bn)
        self.ref_run = Path(pf_io.REF_RUN if ref_run is None else ref_run)
        self.ledger = Path(LEDGER if ledger is None else ledger)
        self.out_root = Path(ROOT_V2 if out_root is None else out_root)


# ---------------------------------------------------------------- the allow-list guard (string checks only: no filesystem access)
def check_stem(stem):
    if not (isinstance(stem, str) and stem in ALLOWED):
        raise Forbidden('a requested hour is not on the allow-list (20261023_23 and every hour of 20261024..20261027): nothing opened')
    return stem


def hour_path(roots, family, stem):
    """The hour file of a family ('bookTicker' | 'reference_prices'); raises Forbidden BEFORE any filesystem access."""
    if family not in EXT:
        raise Forbidden('unknown raw family: nothing opened')
    check_stem(stem)
    return (roots.bn if family == 'bookTicker' else roots.ref_run / 'reference_prices') / f'{stem}{EXT[family]}'


def day_plan(day):
    """The planned stems of a measure day: the previous day's 23Z, then the day's 00Z..23Z; Forbidden before any filesystem access."""
    if not (isinstance(day, str) and day in DAYS):
        raise Forbidden(f'not a measure day ({", ".join(DAYS)}): nothing opened')
    prev = (dt.date(int(day[:4]), int(day[4:6]), int(day[6:])) - dt.timedelta(days=1)).strftime('%Y%m%d')
    return [check_stem(f'{prev}_23')] + [check_stem(f'{day}_{h:02d}') for h in range(24)]


def day_ms(day):
    return int(dt.datetime(int(day[:4]), int(day[4:6]), int(day[6:]), tzinfo=dt.timezone.utc).timestamp()) * 1000


# ---------------------------------------------------------------- small helpers
def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(C.REPO))
    except ValueError:
        return str(p)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def vmhwm_mib():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) / 1024.
    return 0.


def guard(stage):
    m = vmhwm_mib()
    if m >= C.GUARD_MIB:
        raise GuardStop(f'VmHWM {m:.0f} MiB >= guard {C.GUARD_MIB:.0f} MiB at {stage}: stopped before any output (the guard is never raised)')
    return m


def qlabel(q):
    return 'p' + f'{q * 100:g}'


def quantiles(x, scale=1.0, nd=3):
    """Exact order statistics (numpy method 'lower') of x / scale."""
    if not len(x):
        return {}
    return {qlabel(q): round(float(v) / scale, nd) for q, v in zip(QS, np.quantile(x, QS, method='lower'))}


def episodes(mask, g, age, top=10):
    """Runs of consecutive True grid points -> count, points, the `top` longest (first / last g, points, age at the run's end)."""
    d = np.diff(np.r_[np.int8(0), mask.astype(np.int8), np.int8(0)])
    s0, s1 = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    L = s1 - s0 + 1
    o = np.argsort(-L, kind='stable')[:top]
    return dict(n=int(len(L)), points=int(L.sum()), longest=[dict(first=int(g[s0[i]]), last=int(g[s1[i]]), points=int(L[i]), age_at_last=int(age[s1[i]]))
                                                            for i in o])


def load(path):
    """(record, decompressed bytes | None). The hashed bytes are exactly the parsed bytes; missing / unreadable are statuses, never errors."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return dict(path=rel(path), state='missing'), None
    rec = dict(path=rel(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    try:
        data = gzip.decompress(raw)
    except (OSError, EOFError, zlib.error) as e:
        rec.update(state='unreadable', error=f'{type(e).__name__}: {e}'[:200])
        return rec, None
    rec.update(state='read', bytes_unzipped=len(data))
    return rec, data


def head_tail_rows(data, end, k, parse):
    """The first and last k rows `parse` accepts in data[:end] (blank lines and refused lines skipped, as the readers skip them)."""
    head, pos = [], 0
    while len(head) < k and pos < end:
        nl = data.find(b'\n', pos, end)
        line, pos = data[pos:nl], nl + 1
        if line.strip():
            f = parse(line)
            if f is not None:
                head.append(f)
    tail, pos = [], end - 1                                       # data[end - 1] is the last newline
    while len(tail) < k and pos > 0:
        st = data.rfind(b'\n', 0, pos) + 1
        line, pos = data[st:pos], st - 1
        if line.strip():
            f = parse(line)
            if f is not None:
                tail.append(f)
    return head, tail[::-1]


# ---------------------------------------------------------------- bookTicker parsing
def py_bn_fields(line):
    """One bookTicker line -> (recv_ns, E_ms, update_id, bid, ask), or None (the pf_io.bn_rows rule, strict: 8 fields, digit-only integers)."""
    a = line.split(b',')
    if len(a) != 8 or not (a[0].isdigit() and a[1].isdigit() and a[3].isdigit()):
        return None
    try:
        r = int(a[0]), int(a[1]), int(a[3]), float(a[4]), float(a[6])
    except ValueError:
        return None
    return r if max(r[:3]) < 1 << 63 else None


def parse_bn_python(data):
    """Fallback: one Python pass over the terminated lines (py_bn_fields), column arrays only; returns (arrays, refused lines)."""
    acc = {k: array('q') if k in BN_INT else array('d') for k in BN_USE}
    bad = 0
    for line in io.BytesIO(data):
        if not line.endswith(b'\n'):
            break                                                 # the unterminated tail: counted by the caller
        if not line.strip():
            continue
        f = py_bn_fields(line[:-1])
        if f is None:
            bad += 1
            continue
        for k, v in zip(BN_USE, f):
            acc[k].append(v)
    return {k: np.frombuffer(acc[k], dtype=np.int64 if k in BN_INT else np.float64).copy() for k in BN_USE}, bad


def bn_parity(A, head, tail):
    n = len(A['E_ms'])
    need(len(head) == min(PARITY_ROWS, n) and len(tail) == min(PARITY_ROWS, n), 'parser parity: Arrow and the Python parse disagree on the row count')
    for rows, lo in ((head, 0), (tail, n - len(tail))):
        if not rows:
            continue
        for k, col in zip(BN_USE, zip(*rows)):
            got = A[k][lo:lo + len(rows)]
            if not np.array_equal(got, np.array(col, dtype=got.dtype), equal_nan=k not in BN_INT):
                raise ValueError(f'parser parity failed on {k}: Arrow differs from the Python parse of the same lines')


def parse_bn(data):
    """One decompressed bookTicker hour -> ({column: numpy}, info)."""
    end = data.rfind(b'\n') + 1
    info = dict(unterminated_tail=int(bool(data[end:].strip())))
    bad = [0]

    def skip(_row):
        bad[0] += 1
        return 'skip'
    try:
        t = pacsv.read_csv(pa.BufferReader(pa.py_buffer(data).slice(0, end)),
                           read_options=pacsv.ReadOptions(column_names=list(BN_COLS), use_threads=False, block_size=1 << 24),
                           parse_options=pacsv.ParseOptions(invalid_row_handler=skip),
                           convert_options=pacsv.ConvertOptions(include_columns=list(BN_USE), null_values=[], strings_can_be_null=False,
                                                                column_types={k: pa.int64() if k in BN_INT else pa.float64() for k in BN_USE}))
    except pa.ArrowInvalid as e:
        A, nbad = parse_bn_python(data)
        info.update(parser='python_fallback', fallback_reason=(str(e).splitlines() or [''])[0][:160], malformed=nbad + info['unterminated_tail'])
        return A, info
    A = {k: t.column(k).to_numpy() for k in BN_USE}
    del t
    head, tail = head_tail_rows(data, end, PARITY_ROWS, py_bn_fields)
    bn_parity(A, head, tail)
    info.update(parser='arrow', malformed=bad[0] + info['unterminated_tail'], parity_rows=len(head) + len(tail))
    return A, info


# ---------------------------------------------------------------- the day accumulator and the grid
class BnDay:
    """Per window (g - 100, g]: the max-(E_ms, receipt order) row and the row count, combined over files by max / sum; the latest pre-day row;
    the day's receipt-lag histogram; the E_ms around each ledger start inside the span."""

    def __init__(self, D0, starts):
        self.D0 = D0
        self.key = np.full(N_GRID, -1, np.int64)
        self.e = np.full(N_GRID, -1, np.int64)
        self.bid = np.full(N_GRID, np.nan)
        self.ask = np.full(N_GRID, np.nan)
        self.n = np.zeros(N_GRID, np.int64)
        self.pre = None                                           # (E_ms, rank, bid, ask)
        self.hist = np.zeros(LAG_NB, np.int64)
        self.lag = dict(n=0, under=0, over=0, min_us=None, max_us=None, sum_us=0, negative=0, over_1s=0)
        self.emax = self.recv_min = None
        self.starts = {int(s): [None, None] for s in starts}

    def add(self, k, A):
        E, R, U, B, Q = A['E_ms'], A['recv_ns'], A['update_id'], A['bid'], A['ask']
        n, D0 = len(E), self.D0
        out = dict(rows_valid=n, rows_pre_day=0, rows_in_grid=0, rows_after_last_g=0, e_decreases=0, update_id_nonincreasing=0,
                   late_vs_earlier_files=0, crossed=0, lag_abs_gt_10s=0)
        if not n:
            return out
        need(n < (1 << RANK_BITS), 'an hour file has too many rows for the rank field')
        rank = (np.int64(k) << RANK_BITS) + np.arange(n, dtype=np.int64)
        out.update(e_decreases=int((np.diff(E) < 0).sum()), update_id_nonincreasing=int((np.diff(U) <= 0).sum()),
                   late_vs_earlier_files=0 if self.emax is None else int((E < self.emax).sum()), crossed=int((Q < B).sum()),
                   e_min=int(E.min()), e_max=int(E.max()), recv_min=int(R.min()), recv_max=int(R.max()))
        need(out['recv_min'] >= HF_WS_V2_NS, 'era: a row is stamped before the hf_ws_v2 boundary (CLAUDE.md rule 5)')
        self.emax = out['e_max'] if self.emax is None else max(self.emax, out['e_max'])
        self.recv_min = out['recv_min'] if self.recv_min is None else min(self.recv_min, out['recv_min'])
        lag = R // 1000 - E * 1000                                # microseconds, exact
        out['lag_ms'] = quantiles(lag, 1000.0)
        out['lag_abs_gt_10s'] = int((np.abs(lag) > 10_000_000).sum())
        d = lag[(E >= D0) & (E < D0 + DAY_MS)]
        if len(d):
            L = self.lag
            L['n'] += len(d)
            L['sum_us'] += int(d.sum())
            L['negative'] += int((d < 0).sum())
            L['over_1s'] += int((d > 1_000_000).sum())
            L['min_us'] = int(d.min()) if L['min_us'] is None else min(L['min_us'], int(d.min()))
            L['max_us'] = int(d.max()) if L['max_us'] is None else max(L['max_us'], int(d.max()))
            b = (d - LAG_LO_US) // LAG_BIN_US
            L['under'] += int((b < 0).sum())
            L['over'] += int((b >= LAG_NB).sum())
            self.hist += np.bincount(b[(b >= 0) & (b < LAG_NB)], minlength=LAG_NB)
        for S, ab in self.starts.items():                         # collector starts inside the span: E_ms on either side (receipt order)
            lo = R < S
            if lo.any():
                v = int(E[lo].max())
                ab[0] = v if ab[0] is None else max(ab[0], v)
            if not lo.all():
                v = int(E[~lo].min())
                ab[1] = v if ab[1] is None else min(ab[1], v)
        if out['e_decreases']:
            o = np.argsort(E, kind='stable')                      # E_ms order; equal E_ms keep receipt order
            E, B, Q, rank = E[o], B[o], Q[o], rank[o]
        j = (E - D0 + (GRID_MS - 1)) // GRID_MS                   # the window (g - 100, g] of each row: g = D0 + 100 j
        a, z = int(np.searchsorted(j, 0, 'left')), int(np.searchsorted(j, N_GRID, 'left'))
        out.update(rows_pre_day=a, rows_in_grid=z - a, rows_after_last_g=n - z)
        if a:
            cand = (int(E[a - 1]), int(rank[a - 1]), float(B[a - 1]), float(Q[a - 1]))
            if self.pre is None or cand[:2] > self.pre[:2]:
                self.pre = cand
        if z > a:
            jj, ee, rr, bb, qq = j[a:z], E[a:z], rank[a:z], B[a:z], Q[a:z]
            last = np.flatnonzero(np.r_[jj[1:] != jj[:-1], True])
            gi = jj[last]
            key = ((ee[last] - D0 + GRID_MS) << 32) | rr[last]  # > 0, ordered by (E_ms, file, line)
            better = key > self.key[gi]
            s = gi[better]
            self.key[s] = key[better]
            self.e[s] = ee[last][better]
            self.bid[s] = bb[last][better]
            self.ask[s] = qq[last][better]
            self.n[gi] += np.diff(np.r_[-1, last])
        return out

    def finalize(self, holes):
        K = np.empty(N_GRID + 1, np.int64)                        # position 0 = the latest pre-day row (key 0 < any window key)
        K[0] = 0 if self.pre else -1
        K[1:] = self.key
        run = np.maximum.accumulate(K)
        pos = np.maximum.accumulate(np.where((K >= 0) & (K == run), np.arange(N_GRID + 1), -1))[1:]
        has = pos >= 0
        p = np.where(has, pos, 0)
        pre = self.pre or (-1, -1, np.nan, np.nan)
        bid = np.where(has, np.r_[pre[2], self.bid][p], np.nan)
        ask = np.where(has, np.r_[pre[3], self.ask][p], np.nan)
        g = self.D0 + GRID_MS * np.arange(N_GRID, dtype=np.int64)
        age = np.where(has, g - np.r_[np.int64(pre[0]), self.e][p], -1)
        code = np.zeros(N_GRID, np.int8)
        code[age > STALE_MS] = 1
        for h in holes:
            i0 = 0 if h['lo'] is None else int(np.searchsorted(g, h['lo'], 'left'))
            i1 = N_GRID if h['hi'] is None else int(np.searchsorted(g, h['hi'], 'left'))
            h['grid_points'] = max(0, i1 - i0)
            if i1 > i0:
                code[i0:i1] = 2
        code[~has] = 3
        return g, bid, ask, (bid + ask) / 2, self.n, age, code


def bn_holes(files, starts):
    """Coverage holes on the E_ms clock, as gap intervals [lo, hi) (None = unbounded)."""
    pres = [i for i, f in enumerate(files) if f.get('rows_valid', 0) > 0]
    if not pres:
        return []
    H, a, z = [], pres[0], pres[-1]
    H.append(dict(reason='before_first_file_read', lo=None, hi=files[a]['e_min'] + GRID_MS, missing=[f"{f['stem']}:{f['state']}" for f in files[:a]]))
    for a, b in zip(pres, pres[1:]):
        if b > a + 1:
            lo, hi = sorted((files[a]['e_max'], files[b]['e_min']))
            H.append(dict(reason='missing_hour_file', lo=lo, hi=hi + GRID_MS, missing=[f"{f['stem']}:{f['state']}" for f in files[a + 1:b]]))
    H.append(dict(reason='after_last_row_read', lo=files[z]['e_max'], hi=None, missing=[f"{f['stem']}:{f['state']}" for f in files[z + 1:]]))
    for S, (ea, eb) in sorted(starts.items()):
        if ea is not None and eb is not None:
            lo, hi = sorted((ea, eb))
            H.append(dict(reason='collector_start', started_at_ns=S, lo=lo, hi=hi + GRID_MS, missing=[]))
    return H


# ---------------------------------------------------------------- the collector ledger (CLAUDE.md rule 5)
OBJ = re.compile(rb'\{[^{}]*\}')


def ledger_info(path, lo_ns, hi_ns):
    """The append-only collector ledger: the schema in force over the read span [lo_ns, hi_ns] must be hf_ws_v2_recv_boundary throughout, and
    no torn entry may fall inside it (a torn line's fragments are bounded by their neighbours). -> (receipt section, starts inside the span)."""
    raw = path.read_bytes()
    ents, torn = [], []
    for line in raw.split(b'\n'):
        if not line.strip():
            continue
        try:
            objs = [json.loads(line)]
        except ValueError:
            objs, pos = None, 0
            for m in OBJ.finditer(line):
                if line[pos:m.start()].strip():
                    torn.append(len(ents))
                try:
                    o = json.loads(m.group())
                except ValueError:
                    torn.append(len(ents))
                    o = None
                if isinstance(o, dict) and type(o.get('started_at_ns')) is int:
                    ents.append((o['started_at_ns'], str(o.get('collector_schema_version'))))
                pos = m.end()
            if line[pos:].strip():
                torn.append(len(ents))
        for o in objs or ():
            if isinstance(o, dict) and type(o.get('started_at_ns')) is int:
                ents.append((o['started_at_ns'], str(o.get('collector_schema_version'))))
    starts = [s for s, _ in ents]
    need(starts == sorted(starts), 'ledger: run starts are not in append order')
    before = [e for e in ents if e[0] <= lo_ns]
    inside = [e for e in ents if lo_ns < e[0] <= hi_ns]
    need(bool(before), 'ledger: no collector start before the read span')
    need(before[-1][1] == HF_WS_V2_SCHEMA and all(v == HF_WS_V2_SCHEMA for _, v in inside),
         'era: the collector schema over the read span is not hf_ws_v2_recv_boundary throughout (CLAUDE.md rule 5)')
    tb = []
    for i in torn:
        a, b = (ents[i - 1][0] if i > 0 else None), (ents[i][0] if i < len(ents) else None)
        tb.append([a, b])
        need(not ((a is None or a <= hi_ns) and (b is None or b >= lo_ns)), 'ledger: a torn entry could lie inside the read span (era unprovable)')
    return dict(path=rel(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), n_entries=len(ents), n_torn=len(torn), torn_bounds_ns=tb,
                span_ns=[lo_ns, hi_ns], in_force_at_span_start=dict(started_at_ns=before[-1][0], schema=before[-1][1]),
                starts_inside_span=[dict(started_at_ns=s, schema=v) for s, v in inside]), [s for s, _ in inside]


# ---------------------------------------------------------------- the reference relay
def ref_manifest(roots):
    p = roots.ref_run / 'manifest.json'
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return None, 'no relay manifest: the relay carries no reference stream here (ref skipped)'
    M = json.loads(raw)
    topic = str(M.get('topic', ''))
    if not (topic.startswith('chainlinkAssetPriceUpdate/') and M.get('symbol') == 'BTCUSDT' and M.get('provider_feed_id') and M.get('run_id')):
        return None, f'relay topic {topic!r} / symbol {M.get("symbol")!r}: not a Chainlink BTC stream (ref skipped)'
    keep = ('schema', 'run_id', 'topic', 'provider_feed_id', 'symbol', 'source', 'stamp_point', 'signature_verification', 'content_liveness', 'durability')
    return dict(path=rel(p), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), relay_feed_id=topic.split('/', 1)[1],
                **{k: M.get(k) for k in keep}), None


def py_ref_row(line):
    try:
        r = json.loads(line)
    except ValueError:
        return None
    return tuple(r.get(f) for f in REF_F) if isinstance(r, dict) else None


def ref_from_table(t):
    isp = pc.fill_null(pc.equal(t.column('kind'), 'PRICE'), False)
    ok = isp
    for f in REF_REQ:
        ok = pc.and_(ok, pc.is_valid(t.column(f)))
    tp = t.filter(ok)
    n_price = int(pc.sum(isp).as_py() or 0)
    col = lambda f: tp.column(f).to_numpy()  # noqa: E731
    return dict(P=col('publish_time_s'), recv=col('recv_ns'), rts=col('relay_timestamp_s'), price_str=tp.column('price').combine_chunks(),
                dup=col('duplicate_previous_payload'), reg=col('publish_time_regressed'),
                ident={f: sorted(pc.unique(tp.column(f)).to_pylist()) for f in REF_ID}, n_rows=t.num_rows, nonprice=t.num_rows - n_price,
                malformed=n_price - tp.num_rows)


def parse_ref_python(data):
    """Fallback: json.loads per terminated line; PRICE rows with every required field of the right type, the rest counted."""
    cols = {f: [] for f in REF_REQ}
    n = bad = nonprice = 0
    for line in io.BytesIO(data):
        if not line.endswith(b'\n'):
            break
        if not line.strip():
            continue
        r = py_ref_row(line)
        if r is None:
            bad += 1
            continue
        n += 1
        d = dict(zip(REF_F, r))
        if d['kind'] != 'PRICE':
            nonprice += 1
            continue
        if not (all(type(d[f]) is int for f in ('recv_ns', 'publish_time_s', 'relay_timestamp_s')) and type(d['price']) is str
                and all(type(d[f]) is bool for f in ('duplicate_previous_payload', 'publish_time_regressed')) and all(type(d[f]) is str for f in REF_ID)):
            bad += 1
            continue
        for f in REF_REQ:
            cols[f].append(d[f])
    return dict(P=np.array(cols['publish_time_s'], np.int64), recv=np.array(cols['recv_ns'], np.int64), rts=np.array(cols['relay_timestamp_s'], np.int64),
                price_str=pa.array(cols['price'], pa.string()), dup=np.array(cols['duplicate_previous_payload'], bool),
                reg=np.array(cols['publish_time_regressed'], bool), ident={f: sorted(set(cols[f])) for f in REF_ID}, n_rows=n, nonprice=nonprice,
                malformed=bad)


def parse_ref(data):
    end = data.rfind(b'\n') + 1
    info = dict(unterminated_tail=int(bool(data[end:].strip())))
    try:
        t = pajson.read_json(pa.BufferReader(pa.py_buffer(data).slice(0, end)), read_options=pajson.ReadOptions(use_threads=False, block_size=1 << 23),
                             parse_options=pajson.ParseOptions(explicit_schema=REF_SCHEMA, unexpected_field_behavior='ignore'))
    except pa.ArrowInvalid as e:
        X = parse_ref_python(data)
        info.update(parser='python_fallback', fallback_reason=(str(e).splitlines() or [''])[0][:160], malformed=X.pop('malformed') + info['unterminated_tail'])
        return X, info
    head, tail = head_tail_rows(data, end, PARITY_ROWS, py_ref_row)
    n = t.num_rows
    need(len(head) == min(PARITY_ROWS, n) and len(tail) == min(PARITY_ROWS, n), 'ref parser parity: Arrow and json.loads disagree on the row count')
    for rows, lo in ((head, 0), (tail, n - len(tail))):
        got = [tuple(r[f] for f in REF_F) for r in t.slice(lo, len(rows)).to_pylist()]
        if got != rows:
            raise ValueError('ref parser parity failed: Arrow differs from json.loads of the same lines')
    X = ref_from_table(t)
    info.update(parser='arrow', malformed=X.pop('malformed') + info['unterminated_tail'], parity_rows=len(head) + len(tail))
    return X, info


def py_prices(sarr):
    """Exact decimal strings -> float64 by Python float() of each DISTINCT string (no dependence on a vectorised float parser)."""
    d = pc.dictionary_encode(sarr)
    vals = []
    for s in d.dictionary.to_pylist():
        try:
            v = float(s)
        except (TypeError, ValueError):
            raise ValueError('ref: an unparseable price string') from None
        need(math.isfinite(v) and v > 0, 'ref: a non-finite or non-positive price')
        vals.append(v)
    return np.array(vals, np.float64)[d.indices.to_numpy(zero_copy_only=False)]


def ref_holes(files):
    """Holes on the publish clock, as EXCLUSIVE intervals (lo, hi) (None = unbounded)."""
    pres = [i for i, f in enumerate(files) if f.get('rows_price', 0) > 0]
    if not pres:
        return []
    H, a, z = [], pres[0], pres[-1]
    H.append(dict(reason='before_first_file_read', lo=None, hi=files[a]['p_min'], missing=[f"{f['stem']}:{f['state']}" for f in files[:a]]))
    for a, b in zip(pres, pres[1:]):
        if b > a + 1:
            lo, hi = sorted((files[a]['p_max'], files[b]['p_min']))
            H.append(dict(reason='missing_hour_file', lo=lo, hi=hi, missing=[f"{f['stem']}:{f['state']}" for f in files[a + 1:b]]))
    H.append(dict(reason='after_last_row_read', lo=files[z]['p_max'], hi=None, missing=[f"{f['stem']}:{f['state']}" for f in files[z + 1:]]))
    return H


def ref_finalize(D0s, parts, holes):
    t = D0s + np.arange(DAY_S, dtype=np.int64)
    n = sum(len(p[0]) for p in parts)
    code = np.full(DAY_S, 2, np.int8)                             # stale unless a publish for t_s is read
    if n:
        P, R, V, T = (np.concatenate([p[i] for p in parts]) for i in (0, 1, 2, 4))
        S = pa.concat_arrays([p[3] for p in parts])
        o = np.lexsort((R, P))
        P, R, V, T, S = P[o], R[o], V[o], T[o], S.take(pa.array(o))
        uP, st = np.unique(P, return_index=True)
        en = np.r_[st[1:], n]
        lastp, n_recv = en - 1, en - st
        o2 = np.lexsort((V, P))
        n_price = np.add.reduceat(np.r_[True, (P[o2][1:] != P[o2][:-1]) | (V[o2][1:] != V[o2][:-1])].astype(np.int64), st)
        o3 = np.lexsort((V, T, P))
        P3, T3, V3 = P[o3], T[o3], V[o3]
        n_obs = np.add.reduceat(np.r_[True, (P3[1:] != P3[:-1]) | (T3[1:] != T3[:-1]) | (V3[1:] != V3[:-1])].astype(np.int64), st)
        idx = np.searchsorted(uP, t, 'right') - 1
        has = idx >= 0
        ii = np.where(has, idx, 0)
        age = np.where(has, t - uP[ii], -1)
        exact = has & (age == 0)
        price = np.where(has, V[lastp][ii], np.nan)
        pstr = pc.if_else(pa.array(has), S.take(pa.array(lastp)).take(pa.array(ii)), pa.scalar(None, pa.string()))
        nobs_t, nrecv_t = np.where(exact, n_obs[ii], 0), np.where(exact, n_recv[ii], 0)
        frecv = np.where(exact, R[st][ii], -1)
        code[exact] = np.where(n_price[ii][exact] > 1, 1, 0)
        inday = (P >= D0s) & (P < D0s + DAY_S)
        lag = quantiles(R[inday] // 10 ** 6 - P[inday] * 1000, 1.0, 0)
    else:
        has, age, price = np.zeros(DAY_S, bool), np.full(DAY_S, -1, np.int64), np.full(DAY_S, np.nan)
        pstr = pa.nulls(DAY_S, pa.string())
        nobs_t, nrecv_t, frecv = np.zeros(DAY_S, np.int64), np.zeros(DAY_S, np.int64), np.full(DAY_S, -1, np.int64)
        lag = {}
    for h in holes:
        i0 = 0 if h['lo'] is None else int(np.searchsorted(t, h['lo'], 'right'))
        i1 = DAY_S if h['hi'] is None else int(np.searchsorted(t, h['hi'], 'left'))
        h['seconds'] = max(0, i1 - i0)
        if i1 > i0:
            code[i0:i1] = 3
    code[~has] = 4
    tbl = pa.Table.from_arrays([pa.array(t), pa.array(price), pstr, pa.array(age), pa.array(nobs_t), pa.array(nrecv_t), pa.array(frecv),
                                pc.take(pa.array(REF_STATUS), pa.array(code))], schema=REF_OUT_SCHEMA)
    return tbl, dict(status_counts={s: int((code == i).sum()) for i, s in enumerate(REF_STATUS)}, lag_recv_minus_publish_ms_day=lag,
                     stale_runs=episodes(code == 2, t, age))


def ref_pass(D0s, stems, roots):
    M, why = ref_manifest(roots)
    if M is None:
        return dict(status='skipped', reason=why), None
    expect = dict(run_id=M['run_id'], source='reference_prices', provider_feed_id=M['provider_feed_id'], relay_feed_id=M['relay_feed_id'],
                  source_time_precision='SECONDS')
    files, parts, pmax = [], [], None
    for s in stems:
        c = time.process_time()
        rec, data = load(hour_path(roots, 'reference_prices', s))
        rec = dict(stem=s, **rec)
        if data is not None:
            X, info = parse_ref(data)
            del data
            rec.update(info)
            for f in REF_ID:
                need(X['ident'][f] in ([], [expect[f]]), f'ref identity: {f} is not {expect[f]!r} in every PRICE row (feed / run mismatch)')
            P = X['P']
            rec.update(rows=X['n_rows'], rows_price=len(P), nonprice=X['nonprice'], duplicates=int(X['dup'].sum()), regressed=int(X['reg'].sum()))
            if len(P):
                need(bool((X['rts'] >= P).all()), 'ref: a relay timestamp precedes its publish time')
                rec.update(p_min=int(P.min()), p_max=int(P.max()), recv_min=int(X['recv'].min()), recv_max=int(X['recv'].max()),
                           late_vs_earlier_files=0 if pmax is None else int((P < pmax).sum()),
                           lag_recv_minus_publish_ms=quantiles(X['recv'] // 10 ** 6 - P * 1000, 1.0, 0))
                pmax = rec['p_max'] if pmax is None else max(pmax, rec['p_max'])
                parts.append((P, X['recv'], py_prices(X['price_str']), X['price_str'], X['rts']))
            else:
                rec['state'] = 'empty'
        rec['cpu_s'] = round(time.process_time() - c, 3)
        rec['vmhwm_mib'] = round(guard(f'reference hour {s}'), 1)
        files.append(rec)
    holes = ref_holes(files)
    tbl, summ = ref_finalize(D0s, parts, holes)
    by = lambda st: [f['stem'] for f in files if f['state'] == st]  # noqa: E731
    sums = {k: sum(f.get(k, 0) for f in files) for k in ('rows', 'rows_price', 'nonprice', 'malformed', 'duplicates', 'regressed', 'late_vs_earlier_files')}
    return dict(status='written', manifest=M, expected_identity=expect, dir=rel(roots.ref_run / 'reference_prices'), files=files, read=by('read'),
                missing=by('missing'), unreadable=by('unreadable'), empty=by('empty'), counts=sums, holes=holes, **summ), tbl


# ---------------------------------------------------------------- one day (or one probe hour)
def btc_summary(g, age, n_upd, code, holes):
    cov = code <= 1
    gp = {}
    for h in holes:
        gp[h['reason']] = gp.get(h['reason'], 0) + h['grid_points']
    nu = n_upd[cov]
    return dict(status_counts={s: int((code == i).sum()) for i, s in enumerate(STATUS)}, gap_points_by_reason=gp,
                age_ms_ok_or_stale=quantiles(age[cov], 1.0, 0),
                n_upd_ok_or_stale=dict(mean=round(float(nu.mean()), 4) if len(nu) else None, max=int(nu.max()) if len(nu) else None,
                                       zero_share=round(float((nu == 0).mean()), 6) if len(nu) else None),
                stale_episodes=episodes(code == 1, g, age))


def lag_day(acc):
    L, h = acc.lag, acc.hist
    out = dict(n=L['n'], negative=L['negative'], over_1s=L['over_1s'], under_hist=L['under'], over_hist=L['over'],
               mean_ms=round(L['sum_us'] / L['n'] / 1000, 3) if L['n'] else None)
    if not L['n']:
        return out
    cum = np.cumsum(h)
    for q in QS:
        r = int(math.floor(q * (L['n'] - 1))) + 1                # 1-based rank of the 'lower' order statistic
        if q == 0.0 or q == 1.0:
            out[qlabel(q)] = round((L['min_us'] if q == 0.0 else L['max_us']) / 1000, 3)
        elif r <= L['under']:
            out[qlabel(q)] = f'< {LAG_LO_US / 1000:g}'
        elif r - L['under'] > cum[-1]:
            out[qlabel(q)] = f'>= {(LAG_LO_US + LAG_NB * LAG_BIN_US) / 1000:g}'
        else:
            out[qlabel(q)] = round((LAG_LO_US + int(np.searchsorted(cum, r - L['under'], 'left')) * LAG_BIN_US) / 1000, 3)
    return out


def build(day, stems, out, names, roots, mode, note=None):
    t0, cpu0 = time.time(), time.process_time()
    D0 = day_ms(day)
    cpu = {}
    c = time.process_time()
    led, starts = ledger_info(roots.ledger, (D0 - HOUR_MS - 60_000) * 10 ** 6, (D0 + DAY_MS + 60_000) * 10 ** 6)
    cpu['ledger'] = round(time.process_time() - c, 3)
    c = time.process_time()
    ref_sec, ref_tbl = ref_pass(D0 // 1000, stems, roots)        # first: cheap, and an identity mismatch stops before the bookTicker pass
    cpu['ref'] = round(time.process_time() - c, 3)
    c = time.process_time()
    acc, files = BnDay(D0, starts), []
    for k, s in enumerate(stems):
        c1 = time.process_time()
        rec, data = load(hour_path(roots, 'bookTicker', s))
        rec = dict(stem=s, **rec)
        if data is not None:
            A, info = parse_bn(data)
            del data
            ok = np.isfinite(A['bid']) & np.isfinite(A['ask']) & (A['bid'] > 0) & (A['ask'] > 0)
            info['invalid_price'] = int((~ok).sum())
            if info['invalid_price']:
                A = {key: v[ok] for key, v in A.items()}
            rec.update(info)
            rec.update(acc.add(k, A))
            del A
            if rec['rows_valid'] == 0:
                rec['state'] = 'empty'
        rec['cpu_s'] = round(time.process_time() - c1, 3)
        rec['vmhwm_mib'] = round(guard(f'bookTicker hour {s}'), 1)
        files.append(rec)
    cpu['bn_files'] = round(time.process_time() - c, 3)
    c = time.process_time()
    holes = bn_holes(files, acc.starts)
    g, bid, ask, mid, n_upd, age, code = acc.finalize(holes)
    summ = btc_summary(g, age, n_upd, code, holes)
    tbl = pa.Table.from_arrays([pa.array(g), pa.array(bid), pa.array(ask), pa.array(mid), pa.array(n_upd), pa.array(age),
                                pc.take(pa.array(STATUS), pa.array(code))], schema=BTC_SCHEMA)
    del g, bid, ask, mid, age, code
    cpu['finalize'] = round(time.process_time() - c, 3)
    guard('finalize')
    by = lambda st: [f['stem'] for f in files if f['state'] == st]  # noqa: E731
    keys = ('rows_valid', 'malformed', 'invalid_price', 'rows_pre_day', 'rows_in_grid', 'rows_after_last_g', 'e_decreases', 'late_vs_earlier_files',
            'update_id_nonincreasing', 'crossed', 'lag_abs_gt_10s', 'bytes', 'bytes_unzipped')
    fallback = [f['stem'] for f in files if f.get('parser') == 'python_fallback']
    srcs = [Path(__file__).resolve(), Path(C.__file__).resolve(), Path(pf_io.__file__).resolve()]
    R = dict(version=VERSION, module='pf_mt_btc', mode=mode, day=day, asof=dt.datetime.now(dt.timezone.utc).isoformat(), argv=sys.argv, note=note,
             contract='predictfun/reviews/maker_trace_20261122/decision_contract.md (section: Information state at a decision)',
             sources={rel(p): sha_file(p) for p in srcs},
             env=dict(MT_RUN=os.environ.get('MT_RUN'), PF_SLOT=os.environ.get('PF_SLOT'), python=sys.version.split()[0], numpy=np.__version__,
                      pyarrow=pa.__version__),
             allow_list=dict(rule='20261023_23 and every hour of 20261024..20261027, checked on the stem string before any filesystem access',
                             planned=stems),
             grid=dict(clock='E_ms', step_ms=GRID_MS, n=N_GRID, first_g_ms=D0, last_g_ms=D0 + DAY_MS - GRID_MS, stale_ms=STALE_MS,
                       columns=BTC_SCHEMA.names),
             era=dict(rule='CLAUDE.md rule 5: every row recv_ns >= the hf_ws_v2 boundary and the ledger schema hf_ws_v2_recv_boundary over the '
                           'read span', boundary_ns=HF_WS_V2_NS, min_recv_ns=acc.recv_min, ledger=led),
             bn=dict(dir=rel(roots.bn), files=files, read=by('read'), missing=by('missing'), unreadable=by('unreadable'), empty=by('empty'),
                     python_fallback=fallback, counts={k: sum(f.get(k, 0) for f in files) for k in keys}, holes=holes, lag_ms_day=lag_day(acc), **summ),
             ref=ref_sec, definitions=DEFINITIONS, status_order=dict(btc=list(STATUS), ref=list(REF_STATUS)), cpu_stages_s=cpu, guard_mib=C.GUARD_MIB)
    tables = [('btc', tbl)] + ([('ref', ref_tbl)] if ref_tbl is not None else [])
    return publish(out, names, tables, R, t0, cpu0)


def publish(out, names, tables, R, t0, cpu0):
    """Stage in a private dir, publish by hard link (refuses an existing path); the receipt goes last."""
    tmp = out / f'.tmp_{Path(names["receipt"]).stem}_{os.getpid()}'
    need(not tmp.exists(), f'staging dir {tmp.name} exists')
    tmp.mkdir()
    try:
        c = time.process_time()
        outs = {}
        for key, tbl in tables:
            nm = names[key]
            pq.write_table(tbl, tmp / nm, compression='zstd')
            outs[nm] = dict(bytes=(tmp / nm).stat().st_size, sha256=sha_file(tmp / nm), rows=tbl.num_rows, columns=tbl.schema.names)
        R['cpu_stages_s']['write'] = round(time.process_time() - c, 3)
        R['outputs'] = outs
        R['vmhwm_mib'] = round(guard('publish'), 1)
        R['cpu_s'] = round(time.process_time() - cpu0, 3)
        R['wall_s'] = round(time.time() - t0, 3)
        (tmp / names['receipt']).write_text(json.dumps(R, indent=1, default=str))
        for nm in list(outs) + [names['receipt']]:
            os.link(tmp / nm, out / nm)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return R


def run_day(day, roots=None):
    stems = day_plan(day)                                         # Forbidden BEFORE any filesystem access
    roots = roots or Roots()
    need(roots.out_root.is_dir(), f'{rel(roots.out_root)} is missing: run tools/setup_run.py (MT_RUN=v2) first; nothing created')
    out = roots.out_root / 'btc'
    names = dict(btc=f'btc_{day}.parquet', ref=f'ref_{day}.parquet', receipt=f'btc_{day}.json')
    for nm in names.values():
        need(not (out / nm).exists(), f'btc/{nm} exists (write-once): refused before reading any data')
    out.mkdir(exist_ok=True)
    return build(day, stems, out, names, roots, 'day')


def run_probe(out_dir, stem, roots=None):
    """One hour (bookTicker + relay) of PROBE_DAY through the day code (the full-day grid is built, so finalize / write CPU is real; points
    outside the hour are gap / no_data). Outputs go to out_dir, which may not lie under a run root."""
    check_stem(stem)                                              # Forbidden BEFORE any filesystem access
    need(stem[:8] == PROBE_DAY, f'a probe hour must be of {PROBE_DAY}')
    roots = roots or Roots()
    out = Path(out_dir).resolve()
    for r in (roots.out_root, C.OUT_V1):
        rr = Path(r).resolve()
        need(out != rr and rr not in out.parents, 'probe output may not lie under a run root (write-once day outputs)')
    names = dict(btc=f'probe_{stem}_btc.parquet', ref=f'probe_{stem}_ref.parquet', receipt=f'probe_{stem}.json')
    for nm in names.values():
        need(not (out / nm).exists(), f'{nm} exists (write-once)')
    out.mkdir(parents=True, exist_ok=True)
    return build(PROBE_DAY, [stem], out, names, roots, 'probe', note='a single hour; the full-day grid is built so finalize / write CPU is '
                 'measured; grid points outside the hour are gap / no_data by construction')


def brief(R):
    b, r = R['bn'], R['ref']
    return dict(day=R['day'], mode=R['mode'], bn_read=len(b['read']), bn_missing=b['missing'], bn_unreadable=b['unreadable'], bn_empty=b['empty'],
                bn_fallback=b['python_fallback'], bn_rows_valid=b['counts']['rows_valid'], bn_bytes=b['counts']['bytes'],
                btc_status=b['status_counts'], gap_points_by_reason=b['gap_points_by_reason'],
                lag_ms_day={k: b['lag_ms_day'].get(k) for k in ('p0', 'p50', 'p99', 'p99.9', 'p100', 'n')},
                ref=r['status'], ref_status=r.get('status_counts'), ref_rows_price=r.get('counts', {}).get('rows_price'),
                cpu_s=R['cpu_s'], cpu_stages_s=R['cpu_stages_s'], bn_cpu_per_file_s=[f['cpu_s'] for f in b['files']], wall_s=R['wall_s'],
                vmhwm_mib=R['vmhwm_mib'], outputs={k: v['bytes'] for k, v in R['outputs'].items()})


USAGE = 'usage: pf_mt_btc.py day <YYYYMMDD> | probe <out_dir> <20261024_HH>'


def main(argv):
    need(len(argv) >= 2 and argv[1] in ('day', 'probe'), USAGE)
    if argv[1] == 'day':
        need(len(argv) == 3, USAGE)
        R = run_day(argv[2])
    else:
        need(len(argv) == 4, USAGE)
        R = run_probe(argv[2], argv[3])
    print(json.dumps(brief(R), default=str), flush=True)
    return R


if __name__ == '__main__':
    try:
        main(sys.argv)
    except GuardStop as e:
        sys.exit(f'[guard] {e}')
