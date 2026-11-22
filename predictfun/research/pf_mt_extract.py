"""MAKER-TRACE-RECONSTRUCTION-v1 extraction (P-2026-005, 2026-11-22). Contract: predictfun/reviews/maker_trace_20261122/contract.md sections
2-4 (binding); guard / paths / schemas / statuses come from pf_mt_common (imported, never redefined). One consumed UTC day of raw REST match
pages and WS book frames -> write-once tables under pf_mt_common.OUT:
  records_<day>.parquet   one row per UNIQUE match record. Key = settlementId; the canonical content excludes the market object, outcome.status /
                          bestBid / bestAsk / onChainId and the receipt fields (on the REST row, not in the record). A re-poll with equal content
                          raises n_recv; other content under one key -> 'conflict' (amendment 1): every variant is parsed once, and the kept
                          one is the variant that reconciles exactly (taker.amount == amountFilled == sum(makers.amount)); several -> the most
                          receipts (tie -> first seen); none -> the first (x_kept names the rule; every receipt counted). taker.amount ==
                          amountFilled == sum(makers.amount) and taker.price == priceExecuted are checked in Python ints BEFORE any float.
  legs_<day>.parquet      one row per maker leg of a unique record, in YES space (pf_mt_common.yes_space; Up = indexSet 1 = YES).
  markets_<day>.parquet   one row per population market: registry window, the latest receipt's market-object fields, coverage flags.
  tob_<day>.parquet       one row per predictOrderbook frame of the day's markets (FILL and GT days).
  depth/<day>_<mid>_{frames,levels}.parquet   full depth, integer-cent ptick, for the BOOK (manifest `complete`) and GT (scope) markets.
  inventory_<day>.json    the COMPUTED raw-field inventory (exact source keys, value types, formats, clocks, fees, ordering, ids, grid).
  extract_<day>.json      the receipt: sha256 + bytes of every hour file read, clipped / missing hours, envelope margins, statuses, cpu, VmHWM.
Rules. Every raw hour file comes from pf_mt_common.hour_path (Forbidden BEFORE any filesystem access); REST hours = day_hours(day, 1, 1)
clipped, WS hours = the day's own hours; markets belong to the UTC day of their window start ws (the step1 registry; Nov 8: the GT scope
windows). An exclusion is a status flag ('+'-joined in a fixed order, 'ok' = none), never a drop: rows that cannot exist (other markets,
unparseable pages) are counted per hour. A market whose records could be received in an unread hour is 'boundary_censored': REST envelope
[ws - pre_recv, we + post_settle + margin), margin = p99.9 receipt lag (first_recv_ns - settle_ms) rounded up to whole minutes, taken over
the records settled at or after ws - pre_recv (an earlier record is received when the collector first polls its market, ~5 min before ws:
its lag is discovery-gated and cannot reach the upper boundary; the all-record p99.9 is reported too); pre_recv / post_settle = the day's
extreme receipt-before-ws / settle-after-we, rounded up to whole minutes. WS coverage is evidence-based: a market whose legs settle in an hour
whose WS file was not read is 'fills_outside_ws' (or 'fills_in_missing_ws_hour'). Outputs are write-once: staged in a private
temp dir, published by hard link (which refuses an existing path), the receipt last. One hour at a time; only the de-dup index, per-market
states and the inventory cross hours; VmHWM self-guard pf_mt_common.GUARD_MIB (never raised). A REST page whose raw payload is byte-identical
(BLAKE2b-128) to one parsed earlier in the same hour is replayed without parsing: the same index update per record and the same market
receipt (counted as pages_payload_repeat; 40% of the pages of the 2026-10-24 12Z probe hour, outputs identical to a full parse). Every
other page is parsed with each record's market object cut out of the row text first ('{\"market\":' .. ',\"taker\":', the venue's key
order) and each distinct market text parsed once per hour; a page whose cut does not validate (record count, no market left, every
market text a JSON object) is parsed whole instead (counted as market_cut_fallback).
Clocks (contract section 3; never mixed): upd_ms venue book; recv_ns / first_recv_ns collector receipt; settle_ms = UUIDv7 ms of settlementId
(id creation: an UPPER bound of the match); exec_ms = executedAt (the Confirmed second, not a match time); t_rel_s / phase are on settle_ms.
Day roles: 20261023 ID (REST only); 20261024-27 FILL (REST, WS top of book, BOOK depth); 20261108 GT (REST of the 220 scope windows, hours 00-22;
the WS pass, tob and depth, only for the scope windows whose kept legs include a MAKER leg of our own public signer OWN_SIGNER - a declared
population rule decided from the public legs after the REST pass, blind to the private truth table, which is never read; the other windows
get 'no_own_fill_ws_skipped').
Pre-run review fix batch: a registry outcome first seen in a sealed / forward hour is blanked before use ('outcome_sealed'; the GT registry is
read for the scope rows only); 'up_index_mismatch' when a market object's Up outcome is not indexSet 1; a record without a v7 settlementId
gets the synthetic unique sid 'nosid:<tx>:<content digest>' (raw value in x_sid_raw); MATCH_POLL_INCOMPLETE collector events (events hour
files, allowed hours of the REST plan) inside a market's REST envelope -> 'poll_incomplete'; the receipt declares the status order lists.
Usage (run_light.sh): pf_mt_extract.py day <YYYYMMDD> | probe <out_dir outside OUT> rest:<stem> ws:<stem>  (2026-10-24 only, <= 2 hour files)
"""
import collections
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from array import array
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import pf_mt_common as C  # noqa: E402

need, Forbidden, E18 = C.need, C.Forbidden, C.E18
BOOK_MANIFEST = C.REPO / 'predictfun/reviews/ladder_async_fast_20261122/manifest.json'
GT_EXPORT = Path('/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108')
GT_SCOPE, GT_REGISTRY = GT_EXPORT / 'scope.json', GT_EXPORT / 'research/step1/20261108/registry.parquet'
PROBE_DAY, PROBE_MAX_FILES = '20261024', 2
OWN_SIGNER = '0xba14f115ccf30e976f7639594fdee7b36c85ced8'           # our own maker address (public on-chain); compared case-insensitively
GT_WS_RULE = ('GT day: the WS pass (tob and depth) covers only the scope windows whose kept legs include a MAKER leg signed by OWN_SIGNER '
              '(case-insensitive), decided from the public legs after the REST pass and before the WS pass; the truth table is never read; '
              'records, legs and markets cover every scope window')
WINDOW_S, GAP_MS, HOUR_MS, MIN_MS, CENT = 300, 1000, 3_600_000, 60_000, 10 ** 16
GUARD_EVERY, FLUSH_ROWS, LEVEL_FLUSH, FMT_SAMPLE, FRAME_SAMPLE = 400, 100_000, 250_000, 2000, 200
MATCH, MID_KEY, RAW_KEY = b'/v1/orders/matches"', b'"marketId":"', b',"raw":'
RX_TOPIC = re.compile(rb'predictOrderbook/(\d+)')
EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
FLAGS = ('malformed', 'conflict', 'recon_tk_amount', 'recon_mk_sum', 'recon_price', 'no_sid', 'sid_not_v7', 'bad_exec', 'bad_leg', 'side_mismatch',
         'off_grid', 'page_mid_mismatch', 'boundary_censored', 'hour_missing')
WS_FLAGS = ('bad_frame', 'mid_mismatch', 'out_of_order', 'dup', 'same_ms_changed', 'new_conn', 'gap', 'off_grid', 'dup_level')
MK_FLAGS = ('no_records', 'boundary_censored', 'hour_missing', 'no_frames', 'fills_outside_ws', 'fills_in_missing_ws_hour', 'no_own_fill_ws_skipped',
            'outcome_sealed', 'poll_incomplete', 'up_index_mismatch', 'slug_mismatch')
LEVEL_FLAGS = ('off_grid',)
POLICY = ('feeRateBps', 'makerFeeBps', 'spreadThreshold', 'shareThreshold', 'rewards', 'isBoosted', 'boostStartsAt', 'boostEndsAt')
DEFINITIONS = dict(
    status="'+'-joined flags in the module's fixed order (FLAGS / WS_FLAGS / MK_FLAGS); 'ok' = no flag",
    n_recv='receipts of the kept content; x_n_recv_other = receipts of the other contents; x_n_variants = distinct contents under the key',
    x_kept="only (one content) | reconciles (the one exactly reconciling variant) | most_receipts (several reconcile) | reconciles_tie_first_seen | "
           "first_none_reconciles; reconciles = taker.amount == amountFilled == sum(makers.amount) in ints",
    first_recv_ns='earliest receipt ns of any content under the key', settle_ms='UUIDv7 ms of settlementId (-1 when absent / not v7)',
    exec_ms='executedAt as integer ms (-1 when unparseable)', t_rel_s='settle_ms / 1000 - ws (NaN without settle_ms)',
    phase="'pre' t_rel_s < 0, 'live' < we - ws, 'post' otherwise, 'unknown' without settle_ms",
    x_match="'regular' (same outcome, opposite quotes), 'mint' (Bid vs Bid on the other outcome), 'merge' (Ask vs Ask), 'invalid'",
    tob_status='sequence flags per (connection, market) + bad / mid mismatch + off-grid BEST price', frames_status='tob flags + off_grid (any level) + dup_level',
    gap=f'updateTimestampMs more than {GAP_MS} ms after the latest earlier frame of the market (any connection): the span before is unknown',
    dup='same connection, same updateTimestampMs as the previous frame and byte-identical raw message', new_conn='first frame of the market on a connection after frames on another',
    levels='full aggregated YES-space snapshot per frame; ptick = round(price * 100); status off_grid (x_price exact) when not a whole cent',
    market_status='no_records / boundary_censored / hour_missing (REST envelope); no_frames / fills_outside_ws / fills_in_missing_ws_hour (legs whose settle_ms '
                  'hour is not a WS hour read: no book frames for them); no_own_fill_ws_skipped (GT window outside the WS rule); outcome_sealed (the registry '
                  'outcome was first seen in a sealed / forward hour: blanked); poll_incomplete (MATCH_POLL_INCOMPLETE events of the market inside its REST '
                  'envelope, every reason incl. resume-queued ones; per-reason counts in poll_incomplete_reasons); up_index_mismatch; slug_mismatch',
    sid="the settlementId when it is a UUIDv7, else the synthetic unique 'nosid:<transactionHash>:<blake2b-64 of the canonical content>' (raw value in x_sid_raw)",
    rewards_variants='every distinct market.rewards object seen, in first-receipt order, with its first receipt ns; rewards = the latest receipt')

PA = {'str': pa.string(), 'int64': pa.int64(), 'float64': pa.float64()}
EXTRA = {'records': (('x_n_variants', 'int64'), ('x_n_recv_other', 'int64'), ('x_kept', 'str'), ('x_sid_raw', 'str')), 'legs': (('x_match', 'str'),), 'tob': (),
         'frames': (('x_los_kind', 'str'),), 'levels': (('x_price', 'float64'),)}
INTERNAL = {'records': (('_key', 'str'), ('_dg', 'int64'), ('_flags', 'str')), 'legs': (('_key', 'str'), ('_dg', 'int64'), ('_rflags', 'str'), ('_lflags', 'str')),
            'tob': ()}
need('status' in C.SCHEMAS['levels'], 'pf_mt_common.SCHEMAS[levels] lacks status (amendment 1)')
MARKETS = dict(day='str', mid='int64', slug='str', ws='int64', we='int64', outcome='str', outcome_seen_ns='int64', feeRateBps='int64',
               makerFeeBps='int64', spreadThreshold='float64', shareThreshold='float64', rewards='str', isBoosted='int64', boostStartsAt='str',
               boostEndsAt='str', outcome_statuses='str', market_status='str', trading_status='str', resolution='str', category_slug='str',
               latest_recv_ns='int64', n_pages='int64', n_records='int64', n_legs='int64', changed_fields='str', rewards_variants='str',
               rest_lo_ms='int64', rest_hi_ms='int64', n_frames='int64', first_upd_ms='int64', last_upd_ms='int64', n_legs_outside_ws='int64',
               n_legs_in_missing_ws_hour='int64', n_poll_incomplete='int64', poll_incomplete_reasons='str', depth='int64', status='str')


def schema(name, internal=False):
    cols = list(MARKETS.items()) if name == 'markets' else list(C.SCHEMAS[name].items()) + list(EXTRA[name]) + (list(INTERNAL[name]) if internal else [])
    return pa.schema([(c, PA[t]) for c, t in cols])


_ORDER_SETS = {}


def status(flags, order=FLAGS):
    if not flags:
        return C.STATUS_OK
    os_ = _ORDER_SETS.get(order)
    if os_ is None:
        os_ = _ORDER_SETS[order] = frozenset(order)
    bad = set(flags) - os_
    need(not bad, f'unknown status flags {sorted(bad)}')
    return '+'.join(f for f in order if f in flags)


def role_of(day):
    if day == C.ID_DAY:
        return 'ID'
    if day in C.MEASURE_DAYS:
        return 'FILL'
    if day == C.GT_DAY:
        return 'GT'
    raise Forbidden(f'day {day!r} is not a consumed day (never opened)')


def day_ms(day):
    return int(dt.datetime.strptime(day, '%Y%m%d').replace(tzinfo=dt.timezone.utc).timestamp()) * 1000


def stem_ms(stem):
    return day_ms(stem[:8]) + int(stem[9:]) * HOUR_MS


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def hour_file(family, stem):
    """(path, bytes, sha256) of one raw hour file. hour_path refuses a non-consumed hour BEFORE any filesystem access; missing -> bytes -1."""
    p = C.hour_path(family, stem)
    if not p.exists():
        return p, -1, ''
    h, n = hashlib.sha256(), 0
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
            n += len(b)
    return p, n, h.hexdigest()


def lines(path):
    """The lines of a gzip hour file (opened through gzip.open)."""
    with gzip.open(path, 'rb') as f:
        yield from f


def vmhwm_mib():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) / 1024.
    return 0.


def log(msg):
    print(msg, flush=True)


class GuardStop(RuntimeError):
    pass


def guard(stage):
    m = vmhwm_mib()
    if m >= C.GUARD_MIB:
        raise GuardStop(f'VmHWM {m:.0f} MiB >= guard {C.GUARD_MIB:.0f} MiB at {stage}: stopped before any output (the guard is never raised)')
    return m


def u256(v):
    """A 1e18-scaled amount / price string -> exact int. Only ASCII digit strings (int() would also take '1_0', ' 1', '+1')."""
    if isinstance(v, str) and v.isascii() and v.isdigit():
        return int(v)
    raise ValueError(f'not a decimal integer string: {v!r}')


def iso_ms(s):
    """executedAt (ISO 8601 with an offset) -> integer ms (floor); -1 when unparseable or naive."""
    try:
        d = dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
    except (AttributeError, TypeError, ValueError):
        return -1
    return -1 if d.utcoffset() is None else (d - EPOCH) // dt.timedelta(milliseconds=1)


def prefix_int(line, key, lo, hi):
    """The integer value of `key` inside line[lo:hi] (the REST row's prefix before its raw payload), or None."""
    i = line.find(key, lo, hi)
    if i < 0:
        return None
    i += len(key)
    e = i
    while e < hi and 48 <= line[e] <= 57:
        e += 1
    return int(line[i:e]) if e > i else None


MK_A, MK_B = b'{\\"market\\":', b',\\"taker\\":'      # a record's first two keys inside the row's escaped raw string


def cut_markets(line, kr):
    """A REST row line (bytes) -> (the line with every record's market object removed, [each record's escaped market text]); the raw payload
    starts at kr. Structural quotes inside the raw string literal are escaped once (\\\"), quotes inside its strings three times, so the two
    markers can only be structure."""
    out, mks, i = [line[:kr]], [], kr
    while True:
        a = line.find(MK_A, i)
        if a < 0:
            break
        b = line.find(MK_B, a + len(MK_A))
        if b < 0:
            return None
        out.append(line[i:a + 1])
        mks.append(line[a + len(MK_A):b])
        i = b + 1
    out.append(line[i:])
    return b''.join(out), mks


def attach_markets(data, mks, cache):
    """Put each record's market object back (parsed once per distinct text); None when the cut does not validate."""
    if not (isinstance(data, list) and len(data) == len(mks)):
        return None
    for x, t in zip(data, mks):
        if not isinstance(x, dict) or 'market' in x:
            return None
        m = cache.get(t)
        if m is None:
            try:
                m = json.loads(json.loads(b'"' + t + b'"'))
            except (ValueError, TypeError):
                return None
            if not isinstance(m, dict):
                return None
            cache[t] = m
        x['market'] = m
    return data


def frz(v):
    return v if v is None or isinstance(v, (str, int, float, bool)) else json.dumps(v, sort_keys=True, separators=(',', ':'))


def ceil_min(ms):
    return int(max(0, math.ceil(ms / MIN_MS))) * MIN_MS


def qs(a, ps=(0, 0.1, 1, 5, 50, 95, 99, 99.9, 100)):
    a = np.asarray(a, dtype=float)
    return dict({f'p{p:g}': float(np.percentile(a, p)) for p in ps}, n=int(a.size)) if a.size else dict(n=0)


# ---- canonical content (contract section 4) ----
REC_KEEP = frozenset(('amountFilled', 'executedAt', 'makers', 'priceExecuted', 'settlementId', 'taker', 'transactionHash'))
LEG_KEEP = frozenset(('amount', 'fee', 'hash', 'outcome', 'price', 'quoteType', 'signer'))
OUT_KEEP, FEE_KEEP = frozenset(('indexSet', 'name', 'team', 'variantData')), frozenset(('amount', 'type'))
VOL_TOP, VOL_OUT = frozenset(('market',)), frozenset(('status', 'bestBid', 'bestAsk', 'onChainId'))


def _leg_c(l):
    o, f = l['outcome'], l['fee']
    return (l['quoteType'], l['amount'], l['price'], l['signer'], l['hash'], f['amount'], f['type'], o['indexSet'], o['name'], frz(o['team']), frz(o['variantData']))


def _gen(v, drop=()):
    if isinstance(v, dict):
        return ('{', tuple(sorted((k, _gen(w, VOL_OUT if k == 'outcome' else ())) for k, w in v.items() if k not in drop)))
    if isinstance(v, list):
        return ('[', tuple(_gen(w) for w in v))
    return v


def canon(x):
    """Canonical content of one match record: everything but the market object and the outcome status / bestBid / bestAsk / onChainId. The
    flat form is used when every non-volatile key set is the inventoried one; any other shape takes the tagged generic form, so a new or a
    missing field is a content difference, never ignored."""
    if type(x) is dict and x.keys() - VOL_TOP == REC_KEEP and type(x['makers']) is list:
        mk = x['makers']
        out = [x['settlementId'], x['transactionHash'], x['executedAt'], x['amountFilled'], x['priceExecuted'], len(mk)]
        for l in (x['taker'], *mk):
            if type(l) is not dict or l.keys() != LEG_KEEP:
                break
            o, f = l['outcome'], l['fee']
            if type(o) is not dict or type(f) is not dict or o.keys() - VOL_OUT != OUT_KEEP or f.keys() != FEE_KEEP:
                break
            out.append(_leg_c(l))
        else:
            return tuple(out)
    return ('G', _gen(x, VOL_TOP))


def stable_digest(x):
    """A run-independent 64-bit digest of a record's canonical content (hash() is salted per process)."""
    return hashlib.blake2b(repr(canon(x)).encode(), digest_size=8).hexdigest()


def digest(x):
    try:
        return hash(canon(x))
    except TypeError:                                         # an unhashable leaf where a scalar was inventoried -> generic form
        return hash(('G', _gen(x, VOL_TOP)))


# ---- inventory ----
RX_FMT = ((re.compile(r'0x[0-9a-f]+'), 'hex_lower'), (re.compile(r'0x[0-9a-fA-F]+'), 'hex_mixed'),
          (re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'), 'uuid'),
          (re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|[+-]\d\d:\d\d)'), 'iso'), (re.compile(r'\d+'), 'digits'),
          (re.compile(r'-?\d+\.\d+'), 'decimal'), (re.compile(r'[A-Za-z][A-Za-z0-9_]{0,31}'), 'word'))


def sfmt(s):
    if not s:
        return 'empty'
    for rx, name in RX_FMT:
        m = rx.fullmatch(s)
        if m:
            if name.startswith('hex'):
                return f'{name}{len(s) - 2}'
            if name == 'uuid':
                return f'uuid_v{s[14]}'
            if name == 'iso':
                return 'iso' + (f'_frac{len(m.group(1)) - 1}' if m.group(1) else '_nofrac') + ('_Z' if s.endswith('Z') else '_offset')
            return f'word:{s}' if name == 'word' else name
    return f'text_len{min(len(s) // 100 * 100, 5000)}+'


class Inv:
    """Computed field inventory: key sets per object kind, leaf value types (all) and string formats (declared sample), counters, samples."""

    def __init__(self):
        self.keys = collections.defaultdict(collections.Counter)
        self.tk = collections.defaultdict(collections.Counter)                  # kind -> (key, type) -> count
        self.fmts = collections.defaultdict(collections.Counter)
        self.c = collections.defaultdict(collections.Counter)
        self.a = {}

    def arr(self, name, code='d'):
        a = self.a.get(name)
        if a is None:
            a = self.a[name] = array(code)
        return a

    def obj(self, kind, o, fmt):
        if not isinstance(o, dict):
            self.keys[kind][f'<{type(o).__name__}>'] += 1
            return
        self.keys[kind][tuple(o)] += 1
        T = self.tk[kind]
        for k, v in o.items():
            T[(k, type(v))] += 1
            if fmt and isinstance(v, str):
                F = self.fmts[f'{kind}.{k}']
                f = sfmt(v)
                F[f if f in F or len(F) < 40 else 'other'] += 1

    def record(self, x, fmt):
        self.obj('record', x, fmt)
        t, mk = x.get('taker'), x.get('makers')
        for kind, legs in (('taker', [t]), ('maker', mk if isinstance(mk, list) else [mk])):
            for l in legs:
                self.obj(kind, l, fmt)
                if isinstance(l, dict):
                    self.obj(f'{kind}.outcome', l.get('outcome'), fmt)
                    self.obj(f'{kind}.fee', l.get('fee'), fmt)
        ea = x.get('executedAt')
        self.c['executedAt_after_seconds'][ea[19:] if isinstance(ea, str) and len(ea) >= 19 else f'<{type(ea).__name__}>'] += 1

    def market(self, m):
        self.obj('market', m, True)
        for o in m.get('outcomes') or []:
            self.obj('market.outcomes[]', o, True)
        for k in ('resolution', 'rewards', 'variantData'):
            if isinstance(m.get(k), dict):
                self.obj(f'market.{k}', m[k], True)
        for s in (m.get('rewards') or {}).get('schedule') or [] if isinstance(m.get('rewards'), dict) else []:
            self.obj('market.rewards.schedule[]', s, True)

    def dump(self):
        ks = {}
        for k, v in sorted(self.keys.items()):
            agg = collections.Counter()
            for s_, n in v.items():
                agg[tuple(sorted(s_)) if isinstance(s_, tuple) else s_] += n
            ks[k] = [dict(keys=list(s_) if isinstance(s_, tuple) else s_, n=n) for s_, n in agg.most_common()]
        types = collections.defaultdict(dict)
        for kind, T in self.tk.items():
            for (k, t), n in T.items():
                types[f'{kind}.{k}'][t.__name__] = n
        lv = {p: dict(types=types[p], formats=dict(self.fmts.get(p, {}))) for p in sorted(types)}
        return dict(objects=ks, leaves=lv, counters={k: {str(a): n for a, n in v.most_common()} for k, v in sorted(self.c.items())})


class Day:
    """One day's extraction. Rows go to per-hour part files in the private temp dir; across hours only the de-dup index (key -> [first digest,
    its earliest receipt, its receipts, receipts of other contents, {digest: [receipts, earliest, reconciles, order, legs]} | None, mid, settle_ms,
    first reconciles, first legs]), the per-market / per-connection states and the inventory are kept."""

    def __init__(self, day, out_root, rest_stems=None, ws_stems=None, mode='day'):
        self.day, self.mode, self.role = day, mode, role_of(day)
        self.out = Path(out_root)
        self.rest_plan, self.rest_clip = C.day_hours(day, extra_before=1, extra_after=1)
        self.ws_plan, self.ws_clip = C.day_hours(day) if self.role != 'ID' else ([], [])
        if mode == 'probe':
            need(day == PROBE_DAY, f'probe: only {PROBE_DAY}')
            rest_stems, ws_stems = list(rest_stems or ()), list(ws_stems or ())
            need(set(rest_stems) <= set(self.rest_plan) and set(ws_stems) <= set(self.ws_plan), 'probe: hours outside the day plan')
            need(len(rest_stems) + len(ws_stems) <= PROBE_MAX_FILES, f'probe: at most {PROBE_MAX_FILES} hour files')
            self.rest_stems, self.ws_stems = rest_stems, ws_stems
        else:
            need(mode == 'day' and rest_stems is None and ws_stems is None, 'a day run reads its whole plan')
            self.rest_stems, self.ws_stems = list(self.rest_plan), list(self.ws_plan)
        self.pop, self.depth, self.inputs = population(day)
        self.ws_pop, self.ws_sel = self.pop, None                        # GT: narrowed after the REST pass (select_gt_ws)
        existing = [n for n in self.outputs() if (self.out / n).exists()]
        need(not existing, f'write-once: {self.out / existing[0]} exists (refusing to overwrite)' if existing else '')
        self.tmp = self.out / f'.tmp_extract_{day}_{os.getpid()}'
        need(not self.tmp.exists(), f'temp dir {self.tmp} exists')
        self.index, self.inv, self.mk = {}, Inv(), {}
        self.n_rec, self.n_leg = collections.Counter(), collections.Counter()
        self.leg_hours = collections.defaultdict(collections.Counter)     # mid -> settle hour start ms -> legs (WS coverage evidence)
        self.rest_hours, self.ws_hours, self.ev_hours = [], [], []
        self.poll = collections.defaultdict(list)                         # mid -> [(event recv_ns, reason)] of MATCH_POLL_INCOMPLETE
        self.parts = dict(records=[], legs=[], tob=[])
        self.cstate, self.mstate, self.fid, self.dbuf, self.dw = {}, {}, {}, {}, {}
        self.n_frames = collections.Counter()
        self.S = {k: schema(k, internal=True) for k in ('records', 'legs', 'tob')}
        self.SF, self.SL = schema('frames'), schema('levels')
        self.fmt_left = FMT_SAMPLE

    def outputs(self):
        d = self.day
        names = [f'records_{d}.parquet', f'legs_{d}.parquet', f'markets_{d}.parquet']
        names += [f'tob_{d}.parquet'] if self.role != 'ID' else []
        names += [f'depth/{d}_{m}_{k}.parquet' for m in sorted(self.depth) for k in ('frames', 'levels')]
        return names + [f'inventory_{d}.json', f'extract_{d}.json']

    def buf(self, kind):
        return {c: [] for c in self.S[kind].names}

    def part(self, kind, B, stem):
        if B[self.S[kind].names[0]]:
            p = self.tmp / f'part_{kind}_{stem}_{len(self.parts[kind]):04d}.parquet'
            pq.write_table(pa.Table.from_pydict(B, schema=self.S[kind]), p)
            self.parts[kind].append(p)
        return self.buf(kind)

    # ------------------------------------------------------------------ REST
    def rest_hour(self, stem):
        p, nbytes, digest_ = hour_file('rest', stem)
        H = dict(family='rest', stem=stem, bytes=nbytes, sha256=digest_)
        self.rest_hours.append(H)
        if nbytes < 0:
            H['missing'] = True
            return
        cpu0, cnt, seen = time.process_time(), collections.Counter(), set()
        R, L, cache, pop, pages, mcache = self.buf('records'), self.buf('legs'), {}, self.pop, {}, {}
        self.fmt_left = FMT_SAMPLE
        for line in lines(p):
            cnt['lines'] += 1
            j = line.find(b'"path":"', 0, 400)
            if j < 0 or not line.startswith(MATCH, j + 8):
                continue
            cnt['match_rows'] += 1
            k = line.find(MID_KEY, j, j + 400)
            if k >= 0:
                e = line.find(b'"', k + 12, k + 40)
                pm = line[k + 12:e]
                if pm.isdigit() and int(pm) not in pop:
                    cnt['pages_other_markets'] += 1
                    continue
            kr = line.find(RAW_KEY, j)
            pdg = hashlib.blake2b(memoryview(line)[kr:], digest_size=16).digest() if kr > 0 else None
            hit = pages.get(pdg) if pdg is not None else None
            if hit is not None:
                st, rv = prefix_int(line, b'"status":', j, kr), prefix_int(line, b'"recv_ns":', 0, j)
                if st == 200 and rv is not None:
                    cnt['pages'] += 1
                    cnt['pages_payload_repeat'] += 1
                    self.repeat_page(hit, rv, cnt)
                    if cnt['pages'] % GUARD_EVERY == 0:
                        guard(f'rest {stem}')
                    continue
            cut = cut_markets(line, kr) if kr > 0 else None
            if cut is None:
                cnt['market_cut_fallback'] += 1                           # unpaired markers (or no raw field): parsed whole
            try:
                r = json.loads(cut[0] if cut else line)
                if r.get('status') != 200:
                    cnt[f"rows_status_{r.get('status')}"] += 1
                    continue
                pl = json.loads(r['raw'])
            except (ValueError, KeyError, TypeError):
                r = pl = None
            data = pl.get('data') if isinstance(pl, dict) else None
            if cut and (r is None or attach_markets(data, cut[1], mcache) is None):     # the cut did not validate: parse the page whole
                cnt['market_cut_fallback'] += 1
                try:
                    r = json.loads(line)
                    if r.get('status') != 200:
                        cnt[f"rows_status_{r.get('status')}"] += 1
                        continue
                    pl = json.loads(r['raw'])
                except (ValueError, KeyError, TypeError):
                    cnt['pages_unparseable'] += 1
                    continue
                data = pl.get('data') if isinstance(pl, dict) else None
            elif r is None:
                cnt['pages_unparseable'] += 1
                continue
            if not (isinstance(pl, dict) and pl.get('success') is True and isinstance(data, list)):
                cnt['pages_bad_payload'] += 1
                continue
            cnt['pages'] += 1
            if cnt['pages'] <= 50:
                self.inv.keys['rest_row'][tuple(r)] += 1
                self.inv.keys['rest_params'][tuple(r.get('params') or {})] += 1
                self.inv.keys['rest_payload'][tuple(pl)] += 1
            recv, prm = r['recv_ns'], r.get('params') or {}
            pm = int(prm['marketId']) if str(prm.get('marketId', '')).isdigit() else None
            mkv = self.market_page(data[0]['market'], recv, seen) if data and isinstance(data[0], dict) and isinstance(data[0].get('market'), dict) else None
            ents = [self.record(x, recv, pm, R, L, cnt, cache) for x in data]
            if pdg is not None:
                pages[pdg] = (mkv, [e for e in ents if type(e) is tuple], ents.count('other'), ents.count('bad'))
            if len(L['mid']) >= FLUSH_ROWS:
                R, L = self.part('records', R, stem), self.part('legs', L, stem)
            if cnt['pages'] % GUARD_EVERY == 0:
                guard(f'rest {stem}')
        self.part('records', R, stem)
        self.part('legs', L, stem)
        H.update(dict(sorted(cnt.items())), cpu_s=round(time.process_time() - cpu0, 3), vmhwm_mib=round(guard(f'rest {stem}'), 1))
        log(f'[rest] {stem} ' + json.dumps({k: H[k] for k in ('pages', 'record_receipts', 'new_records', 'cpu_s', 'vmhwm_mib') if k in H}))

    def market_page(self, m, recv, seen):
        """The market object of a page (all records of a page carry the same one) -> its values; applied as one receipt. None if not ours."""
        mid = m.get('id')
        if mid not in self.pop:
            return None
        if mid not in seen:                                   # the first page of a market in each hour feeds the market inventory
            seen.add(mid)
            self.inv.market(m)
        res = m.get('resolution')
        V = dict(snap=tuple(frz(m.get(k)) for k in POLICY), rewards=json.dumps(m.get('rewards'), sort_keys=True, separators=(',', ':')),
                 status=str(m.get('status') or ''), trading=str(m.get('tradingStatus') or ''), resolution=str(res.get('name') or '') if isinstance(res, dict) else '',
                 slug=str(m.get('categorySlug') or ''),
                 outcomes=json.dumps([{k: o.get(k) for k in ('indexSet', 'name', 'status')} for o in m.get('outcomes') or [] if isinstance(o, dict)], separators=(',', ':')),
                 up_ok=[o.get('indexSet') for o in m.get('outcomes') or [] if isinstance(o, dict) and o.get('name') == 'Up'] == [1])
        self.market_apply(mid, V, recv)
        return mid, V

    def market_apply(self, mid, V, recv):
        S = self.mk.get(mid)
        if S is None:
            S = self.mk[mid] = dict(first=V['snap'], changed=set(), recv=-1, n_pages=0, rv={}, up_bad=False)
        S['n_pages'] += 1
        S['up_bad'] = S['up_bad'] or not V['up_ok']
        if V['rewards'] not in S['rv']:
            S['rv'][V['rewards']] = recv
        if V['snap'] != S['first']:
            S['changed'].update(k for k, a, b in zip(POLICY, S['first'], V['snap']) if a != b)
        if recv >= S['recv']:
            S.update(V, recv=recv)

    def repeat_page(self, hit, recv, cnt):
        """A page whose raw payload is byte-identical to one parsed earlier in this hour: replay exactly what parsing it would do (each record's
        index update by its stored key + digest, the market object's values as one receipt) without parsing it."""
        mkv, entries, n_other, n_bad = hit
        cnt['record_receipts'] += len(entries) + n_other + n_bad
        cnt['records_other_markets'] += n_other
        cnt['records_not_object'] += n_bad
        if mkv is not None:
            self.market_apply(mkv[0], mkv[1], recv)
        for key, dg in entries:
            e = self.index[key]
            if e[0] == dg:
                e[2] += 1
                cnt['repolls'] += 1
                if recv < e[1]:
                    e[1] = recv
                    cnt['recv_inversions'] += 1
            else:
                self.variant(e, key, dg, recv, cnt)

    def record(self, x, recv, pm, R, L, cnt, cache):
        cnt['record_receipts'] += 1
        if not isinstance(x, dict):
            cnt['records_not_object'] += 1
            return 'bad'
        mkt = x.get('market')
        mid = mkt.get('id') if isinstance(mkt, dict) else None
        if mid not in self.pop:
            cnt['records_other_markets'] += 1
            return 'other'
        dg = digest(x)
        sid = x.get('settlementId')
        key = sid if isinstance(sid, str) and sid else f'nosid:{dg & 0xFFFFFFFFFFFFFFFF:016x}'
        e = self.index.get(key)
        if e is None:
            cnt['new_records'] += 1
            self.n_rec[mid] += 1
            settle, ok, nl = self.new_record(key, dg, x, mid, recv, pm, R, L, cache, True)
            self.index[key] = [dg, recv, 1, 0, None, mid, settle, ok, nl]
        elif e[0] == dg:
            e[2] += 1
            cnt['repolls'] += 1
            if recv < e[1]:
                e[1] = recv
                cnt['recv_inversions'] += 1
        else:
            self.variant(e, key, dg, recv, cnt, (x, mid, pm, R, L, cache))
        return key, dg

    def variant(self, e, key, dg, recv, cnt, parse=None):
        """A receipt of another content under a known key. A new content is parsed once (its rows stay in the parts, tagged by digest, until
        finalize keeps one variant per key); a known one counts a receipt."""
        e[3] += 1
        cnt['conflict_receipts'] += 1
        if e[4] is None:
            e[4] = {}
        v = e[4].get(dg)
        if v is None:
            need(parse is not None, f'a repeated page brought an unseen variant of {key}')
            cnt['new_variants'] += 1
            x, mid, pm, R, L, cache = parse
            _, ok, nl = self.new_record(key, dg, x, mid, recv, pm, R, L, cache, False)
            e[4][dg] = [1, recv, ok, len(e[4]) + 1, nl]                       # receipts, earliest receipt, reconciles, order seen, legs
        else:
            v[0] += 1
            if recv < v[1]:
                v[1] = recv
                cnt['recv_inversions'] += 1

    def new_record(self, key, dg, x, mid, recv, pm, R, L, cache, first):
        """Parse, reconcile (ints) and convert one content of a key -> one records row + its legs rows, tagged with the content digest dg;
        the inventory reads the key's first content only. Returns (settle_ms or -1, reconciles exactly, legs)."""
        inv, fl, W = self.inv, set(), self.pop[mid]
        fmt = first and self.fmt_left > 0
        self.fmt_left -= first
        if first:
            inv.record(x, fmt)
        sid_raw, settle = x.get('settlementId'), -1
        if not (isinstance(sid_raw, str) and sid_raw):
            fl.add('no_sid')
        else:
            try:
                settle = C.uuid7_ms(sid_raw)
            except ValueError:
                fl.add('sid_not_v7')
        tx = x.get('transactionHash') if isinstance(x.get('transactionHash'), str) else ''
        sid = sid_raw if settle >= 0 else f'nosid:{tx}:{stable_digest(x)}'      # N2: unique and reproducible across runs
        sid_raw = sid_raw if isinstance(sid_raw, str) else ''
        if pm is not None and pm != mid:
            fl.add('page_mid_mismatch')
        ea = x.get('executedAt')
        ex = cache.get(ea) if isinstance(ea, str) else None
        if ex is None:
            ex = iso_ms(ea)
            if isinstance(ea, str):
                cache[ea] = ex
        if ex < 0:
            fl.add('bad_exec')
        t, mk = x.get('taker'), x.get('makers')
        t_rel = settle / 1000. - W['ws'] if settle >= 0 else math.nan
        phase = 'unknown' if settle < 0 else ('pre' if t_rel < 0 else ('live' if t_rel < W['we'] - W['ws'] else 'post'))
        try:
            af, pe = u256(x['amountFilled']), u256(x['priceExecuted'])
            ta, tp, tq, ti = u256(t['amount']), u256(t['price']), t['quoteType'], t['outcome']['indexSet']
            tfa, tft = u256(t['fee']['amount']), t['fee']['type']
            M = [(u256(m['amount']), u256(m['price']), m['quoteType'], m['outcome']['indexSet'], u256(m['fee']['amount']), m['fee']['type'],
                  m.get('hash'), m.get('signer')) for m in mk]
        except (KeyError, TypeError, ValueError, AttributeError):
            fl.add('malformed')
            inv.c['malformed_records']['n'] += first
            row = (self.day, mid, sid, tx, settle, ex, recv, 1, math.nan, math.nan, len(mk) if isinstance(mk, list) else -1, '', '', '', -1,
                   math.nan, math.nan, math.nan, '', '', '', 1, 0, '', sid_raw, key, dg, '+'.join(sorted(fl)))
            for c, v in zip(self.S['records'].names, row):
                R[c].append(v)
            return settle, False, 0
        if ta != af:
            fl.add('recon_tk_amount')
        if sum(m[0] for m in M) != af:
            fl.add('recon_mk_sum')
        if tp != pe:
            fl.add('recon_price')
        try:
            tky = C.yes_space(tq, ti, tp / E18)[0]
        except ValueError:
            tky = ''
            fl.add('bad_leg')
        th, tsg = (t.get('hash') or ''), (t.get('signer') or '')
        if not (isinstance(th, str) and th and isinstance(tsg, str) and tsg):
            fl.add('malformed')
            th, tsg = str(th), str(tsg)
        tq, ti = (tq if isinstance(tq, str) else str(tq)), (ti if type(ti) is int else -1)
        rf, allf, yps, sides = '+'.join(sorted(fl)), set(fl), [], []
        for i, (a, p, q, ix, fa, ft, h, sg) in enumerate(M):
            lf = set()
            try:
                ys, yp = C.yes_space(q, ix, p / E18)
            except ValueError:
                ys, yp = '', math.nan
                lf.add('bad_leg')
            if p % CENT:
                lf.add('off_grid')
            if ys and tky and ys == tky:
                lf.add('side_mismatch')
            if not (isinstance(h, str) and h and isinstance(sg, str) and sg):
                lf.add('malformed')
                h, sg = str(h), str(sg)
            q, ix = (q if isinstance(q, str) else str(q)), (ix if type(ix) is int else -1)
            match = ('regular' if ix == ti and q != tq else 'mint' if ix != ti and q == tq == 'Bid' else 'merge' if ix != ti and q == tq == 'Ask' else 'invalid')
            row = (self.day, mid, sid, tx, settle, ex, recv, i, len(M), h, sg, q, ix, p / E18, a / E18, fa / E18, str(ft), ys, yp,
                   int(round(yp * 100)) if ys else -1, th, tsg, tky, phase, t_rel, '', match, key, dg, rf, '+'.join(sorted(lf)))
            for c, v in zip(self.S['legs'].names, row):
                L[c].append(v)
            allf |= lf
            yps.append(yp)
            sides.append(ys)
            if not first:
                continue
            inv.c['leg_match'][match] += 1
            inv.c['maker_fee_type'][str(ft)] += 1
            inv.c['maker_fee']['legs'] += 1
            inv.c['maker_fee']['nonzero'] += fa != 0
            inv.c['maker_vs_taker_yes_side']['same' if ys == tky else 'opposite' if ys and tky else 'unknown'] += 1
        row = (self.day, mid, sid, tx, settle, ex, recv, 1, af / E18, pe / E18, len(M), th, tsg, tq, ti,
               tp / E18, ta / E18, tfa / E18, str(tft), tky, '', 1, 0, '', sid_raw, key, dg, '+'.join(sorted(allf)))
        for c, v in zip(self.S['records'].names, row):
            R[c].append(v)
        ok = 'recon_tk_amount' not in fl and 'recon_mk_sum' not in fl
        if not first:
            return settle, ok, len(M)
        # ---- inventory of the key's first content ----
        if settle >= 0 and ex >= 0:
            inv.arr('settle_minus_exec_ms').append(settle - ex)
        inv.c[f'taker_fee_type | taker {tq}/indexSet {ti}/YES {tky or "?"}'][str(tft)] += 1
        den = ta * min(tp, E18 - tp)
        if den > 0:
            inv.arr(f'taker_fee_over_q_min_p_1mp | {tft}').append(tfa * E18 / den)
        inv.c['n_makers'][str(min(len(M), 20))] += 1
        if len(M) >= 2 and tky in ('bid', 'ask') and all(sides):          # makers[] order in YES space, from the taker's best price
            up, dn = all(b >= a for a, b in zip(yps, yps[1:])), all(b <= a for a, b in zip(yps, yps[1:]))
            best, worst = (up, dn) if tky == 'bid' else (dn, up)
            inv.c['makers_yes_price_order (multi-maker records)']['single_price' if len(set(yps)) == 1 else 'best_first' if best else
                                                                  'worst_first' if worst else 'neither'] += 1
        S_ = sum(m[0] for m in M)
        if S_ > 0:
            d = pe * S_ - sum(a * (p if ix == ti else E18 - p) for a, p, q, ix, *_ in M)
            inv.c['priceExecuted_vs_maker_avg']['exact' if d == 0 else 'within_rounding' if abs(d) < S_ else 'differs'] += 1
        return settle, ok, len(M)

    # ------------------------------------------------------------------ WS
    def ws_hour(self, stem):
        p, nbytes, digest_ = hour_file('predict_ws', stem)
        H = dict(family='predict_ws', stem=stem, bytes=nbytes, sha256=digest_)
        self.ws_hours.append(H)
        if nbytes < 0:
            H['missing'] = True
            return
        cpu0, cnt, T, pop = time.process_time(), collections.Counter(), self.buf('tob'), self.ws_pop
        topic = {m: f'predictOrderbook/{m}' for m in pop}
        tapp = [T[c].append for c in self.S['tob'].names]
        self.wsa = (self.inv.arr('frame_interval_ms', 'q').append, self.inv.arr('recv_minus_upd_ms', 'f').append, self.inv.arr('levels_per_frame', 'H').append)
        for line in lines(p):
            cnt['lines'] += 1
            mm = RX_TOPIC.search(line)
            if mm is None:
                cnt['lines_without_book_topic'] += 1
                continue
            mid = int(mm.group(1))
            if mid not in pop:
                cnt['frames_skipped_no_own_fill' if mid in self.pop else 'frames_other_markets'] += 1
                continue
            try:
                r = json.loads(line)
                m = json.loads(r['raw'])
            except (ValueError, KeyError, TypeError):
                cnt['population_lines_unparseable'] += 1
                continue
            if not (isinstance(m, dict) and m.get('type') == 'M' and m.get('topic') == topic[mid]):
                cnt['population_msgs_not_book'] += 1
                continue
            cnt['frames'] += 1
            if cnt['frames'] <= 50:
                self.inv.keys['ws_row'][tuple(r)] += 1
                self.inv.keys['ws_msg'][tuple(m)] += 1
            self.frame(mid, r, m.get('data'), tapp, cnt['frames'] % FRAME_SAMPLE == 1)
            if len(T['mid']) >= FLUSH_ROWS:
                T = self.part('tob', T, stem)
                tapp = [T[c].append for c in self.S['tob'].names]
            if cnt['frames'] % (50 * GUARD_EVERY) == 0:
                guard(f'ws {stem}')
        self.part('tob', T, stem)
        for mid in list(self.dbuf):
            self.flush_depth(mid)
        H.update(dict(sorted(cnt.items())), cpu_s=round(time.process_time() - cpu0, 3), vmhwm_mib=round(guard(f'ws {stem}'), 1))
        log(f'[ws] {stem} ' + json.dumps({k: H[k] for k in ('frames', 'frames_other_markets', 'cpu_s', 'vmhwm_mib') if k in H}))

    def frame(self, mid, r, d, tapp, sample):
        inv, fl = self.inv, set()
        recv, conn, hraw = r.get('recv_ns'), str(r.get('connection_id') or ''), hash(r.get('raw'))
        if type(recv) is not int:
            recv = -1
            fl.add('bad_frame')
        if not isinstance(d, dict):
            d = {}
            fl.add('bad_frame')
        inv.keys['frame'][tuple(d)] += 1
        if sample:
            inv.obj('frame', d, True)
        upd = d.get('updateTimestampMs')
        if type(upd) is not int:
            upd = -1
            fl.add('bad_frame')
        if d.get('marketId') != mid:
            fl.add('mid_mismatch')
        bids, asks, sp = d.get('bids') or [], d.get('asks') or [], d.get('settlementsPending')
        try:
            bb, ba = (max(bids) if bids else None), (min(asks) if asks else None)
            bbp, bbs = (float(bb[0]), float(bb[1])) if bb else (math.nan, 0.)
            bap, bas = (float(ba[0]), float(ba[1])) if ba else (math.nan, 0.)
            spb, spa = ((sp.get('bids') or []), (sp.get('asks') or [])) if isinstance(sp, dict) else ([], [])
            pb, pa_ = float(sum(v for _, v in spb)), float(sum(v for _, v in spa))
        except (TypeError, ValueError, IndexError):
            fl.add('bad_frame')
            bbp = bap = pb = pa_ = math.nan
            bbs = bas = 0.
            spb, spa = [], []
        for px in (bbp, bap):
            if px == px and abs(px * 100 - round(px * 100)) > 1e-6:
                fl.add('off_grid')
        if upd >= 0:                                          # sequence flags per (connection, market) and per market
            cs, ms = self.cstate.get((conn, mid)), self.mstate.get(mid)
            if cs is None:
                if ms is not None:
                    fl.add('new_conn')
            elif upd < cs[0]:
                fl.add('out_of_order')
            elif upd == cs[1]:
                fl.add('dup' if hraw == cs[2] else 'same_ms_changed')
            if ms is not None:
                if upd - ms[0] > GAP_MS:
                    fl.add('gap')
                self.wsa[0](upd - ms[0])
                ms[0], ms[1], ms[2] = max(ms[0], upd), min(ms[1], upd), ms[2] + 1
            else:
                self.mstate[mid] = [upd, upd, 1]
            self.cstate[(conn, mid)] = [upd if cs is None else max(cs[0], upd), upd, hraw]
            if recv >= 0:
                self.wsa[1](recv / 1e6 - upd)
        self.n_frames[mid] += 1
        self.wsa[2](min(len(bids) + len(asks), 65535))
        if bbp == bbp and bap == bap:
            inv.c['tob_state']['crossed' if bbp > bap else 'locked' if bbp == bap else 'two_sided'] += 1
        else:
            inv.c['tob_state']['empty' if bbp != bbp and bap != bap else 'one_sided'] += 1
        los = d.get('lastOrderSettled')
        inv.c['lastOrderSettled'][(los.get('kind'), los.get('outcome'), los.get('side')) if isinstance(los, dict) else f'<{type(los).__name__}>'] += 1
        inv.c['version'][str(d.get('version'))] += 1
        tob_st = status(fl, WS_FLAGS)
        row = (self.day, mid, upd, recv, conn, d.get('orderCount') if type(d.get('orderCount')) is int else -1, len(bids), len(asks), bbp, bbs, bap, bas, pb, pa_, tob_st)
        for a_, v in zip(tapp, row):
            a_(v)
        if mid not in self.depth:
            return
        D = self.dbuf.get(mid)
        if D is None:
            D = self.dbuf[mid] = dict(F={c: [] for c in self.SF.names}, fid=array('q'), side=array('b'), pt=array('q'), sh=array('d'), xs=array('b'), xp=array('d'))
            D['Fa'] = [D['F'][c].append for c in self.SF.names]
        fid = self.fid.get(mid, 0)
        self.fid[mid] = fid + 1
        for code, lv in ((0, bids), (1, asks)):
            prices = set()
            for e in lv:
                try:
                    px, s = float(e[0]), float(e[1])
                except (TypeError, ValueError, IndexError):
                    fl.add('bad_frame')
                    inv.c['depth_levels']['malformed'] += 1
                    continue
                pt = int(round(px * 100))
                off = abs(px * 100 - pt) > 1e-6
                if off:
                    fl.add('off_grid')
                if px in prices:
                    fl.add('dup_level')
                prices.add(px)
                D['fid'].append(fid)
                D['side'].append(code)
                D['pt'].append(pt)
                D['sh'].append(s)
                D['xs'].append(1 if off else 0)
                D['xp'].append(px if off else math.nan)
                inv.c['depth_levels']['off_grid' if off else 'whole_cent'] += 1
        if isinstance(los, dict):
            try:
                lp = float(los.get('price'))
            except (TypeError, ValueError):
                lp = math.nan
            lrow = (str(los.get('id') or ''), str(los.get('side') or ''), lp, str(los.get('outcome') or ''), str(los.get('kind') or ''))
        else:
            lrow = ('', '', math.nan, '', '')
        row = (mid, fid, upd, recv, conn, row[5], json.dumps(spb, separators=(',', ':')), json.dumps(spa, separators=(',', ':'))) + lrow[:4] + (status(fl, WS_FLAGS), lrow[4])
        for a_, v in zip(D['Fa'], row):
            a_(v)
        if len(D['fid']) >= LEVEL_FLUSH:
            self.flush_depth(mid)

    def flush_depth(self, mid):
        D = self.dbuf.pop(mid, None)
        if D is None or not D['F']['mid']:
            return
        W = self.dw.get(mid)
        if W is None:
            W = self.dw[mid] = (pq.ParquetWriter(self.tmp / f'depth/{self.day}_{mid}_frames.parquet', self.SF),
                                pq.ParquetWriter(self.tmp / f'depth/{self.day}_{mid}_levels.parquet', self.SL))
        W[0].write_table(pa.Table.from_pydict(D['F'], schema=self.SF))
        n = len(D['fid'])
        if n:
            xs = np.frombuffer(D['xs'], np.int8)
            W[1].write_table(pa.Table.from_pydict(dict(
                mid=np.full(n, mid, np.int64), fid=np.frombuffer(D['fid'], np.int64), side=np.where(np.frombuffer(D['side'], np.int8) == 0, 'bid', 'ask'),
                ptick=np.frombuffer(D['pt'], np.int64), shares=np.frombuffer(D['sh'], np.float64), status=np.where(xs == 1, 'off_grid', 'ok'),
                x_price=np.frombuffer(D['xp'], np.float64)), schema=self.SL))

    # ------------------------------------------------------------------ run / finalize
    def run(self):
        t0, cpu0 = time.time(), time.process_time()
        (self.tmp / 'depth').mkdir(parents=True)
        try:
            for s in self.rest_stems:
                self.rest_hour(s)
            if self.mode == 'day':
                for s in self.rest_plan:
                    self.events_hour(s)
            if self.role == 'GT':
                self.select_gt_ws()
            for s in self.ws_stems:
                self.ws_hour(s)
            return self.finalize(t0, cpu0)
        finally:
            for W in self.dw.values():
                for w in W:
                    try:
                        w.close()
                    except Exception:  # noqa: BLE001 - a writer already closed by finalize
                        pass
            shutil.rmtree(self.tmp, ignore_errors=True)

    def events_hour(self, stem):
        """Collector events of one hour (N3): kinds counted from the row prefix; MATCH_POLL_INCOMPLETE rows parsed for market_id / reason."""
        p, nbytes, digest_ = hour_file('events', stem)
        H = dict(family='events', stem=stem, bytes=nbytes, sha256=digest_)
        self.ev_hours.append(H)
        if nbytes < 0:
            H['missing'] = True
            return
        kinds, reasons = collections.Counter(), collections.Counter()
        for line in lines(p):
            j = line.find(b'"kind":"', 0, 300)
            if j < 0:
                kinds['<no kind>'] += 1
                continue
            e = line.find(b'"', j + 8, j + 80)
            k = line[j + 8:e].decode('ascii', 'replace')
            kinds[k] += 1
            if k != 'MATCH_POLL_INCOMPLETE':
                continue
            try:
                r = json.loads(line)
                mid, t = int(r['market_id']), int(r['recv_ns'])
            except (ValueError, KeyError, TypeError):
                reasons['<unparseable>'] += 1
                continue
            reasons[str(r.get('reason'))] += 1
            if mid in self.pop:
                self.poll[mid].append((t, str(r.get('reason'))))
        H.update(kinds=dict(kinds.most_common()), match_poll_incomplete_reasons=dict(reasons.most_common()))

    def select_gt_ws(self):
        """GT day: restrict the WS pass (tob and depth) to the scope windows whose KEPT legs include a maker leg of OWN_SIGNER (GT_WS_RULE).
        The write-once check at init covered the depth files of every window; only the selected windows' depth files are written."""
        own = set()
        for p in self.parts['legs']:
            t = pq.read_table(p, columns=['mid', 'mk_signer', '_key', '_dg']).to_pydict()
            for m, sg, k, g in zip(t['mid'], t['mk_signer'], t['_key'], t['_dg']):
                if sg.lower() == OWN_SIGNER and choose(self.index[k])[0] == g:
                    own.add(m)
        self.ws_pop = {m: W for m, W in self.pop.items() if m in own}
        self.depth = set(self.ws_pop)
        self.ws_sel = dict(rule=GT_WS_RULE, signer=OWN_SIGNER, n_selected=len(self.ws_pop), n_excluded=len(self.pop) - len(self.ws_pop),
                           selected=sorted(self.ws_pop), excluded=sorted(set(self.pop) - set(self.ws_pop)))
        log(f'[gt] WS pass for {len(self.ws_pop)} of {len(self.pop)} scope windows (own maker fill)')

    def envelope(self):
        """Measured envelope parameters over the day's unique records (ms; each rounded up to whole minutes, floored at 0)."""
        E = list(self.index.values())
        if self.mode == 'day':
            need(E, f'no match record of the {self.day} markets was read')
        if not E:
            return dict(n_records=0, margin_ms=0, post_settle_ms=0, pre_recv_ms=0, margin_min=0, note='no records: margins 0 (probe only)')
        ws = np.array([self.pop[e[5]]['ws'] for e in E], np.int64) * 1000
        we = np.array([self.pop[e[5]]['we'] for e in E], np.int64) * 1000
        rcv = np.array([recv_all(e) for e in E], np.int64)
        rms = rcv // 1_000_000 + (rcv % 1_000_000) / 1e6
        st = np.array([e[6] for e in E], np.int64)
        ok = st >= 0
        R = dict(n_records=len(E), n_with_settle=int(ok.sum()), pre_recv_max_ms=float((ws - rms).max()))
        R['pre_recv_ms'] = pre = ceil_min(R['pre_recv_max_ms'])
        if ok.any():
            lag, gated = rms[ok] - st[ok], st[ok] < ws[ok] - pre                  # gated: settled before the market's earliest receipt lead
            R.update(receipt_lag_ms=qs(lag), receipt_lag_ungated_ms=qs(lag[~gated]), n_discovery_gated=int(gated.sum()),
                     lag_p999_all_ms=float(np.percentile(lag, 99.9)), lag_p999_ms=float(np.percentile(lag[~gated], 99.9)) if (~gated).any() else 0.,
                     post_settle_max_ms=float((st[ok] - we[ok]).max()), pre_settle_max_ms=float((ws[ok] - st[ok]).max()))
        else:
            R.update(lag_p999_ms=0., post_settle_max_ms=0., pre_settle_max_ms=0., note='no settle_ms: margins 0')
        R.update(margin_ms=ceil_min(R['lag_p999_ms']), post_settle_ms=ceil_min(R['post_settle_max_ms']), margin_min=ceil_min(R['lag_p999_ms']) // MIN_MS,
                 rule='REST envelope [ws - pre_recv_ms, we + post_settle_ms + margin_ms); margin = p99.9 of first_recv_ns - settle_ms over the records '
                      'settled at or after ws - pre_recv_ms (earlier ones are discovery-gated; lag_p999_all_ms is the all-record value); pre_recv / '
                      'post_settle = the day extremes; all rounded up to whole minutes. WS coverage: legs whose settle_ms hour is not a WS hour read')
        return R

    @staticmethod
    def cover_flags(lo, hi, covered, missing, f_out, f_missing):
        fl, h = set(), (lo // HOUR_MS) * HOUR_MS
        while h < hi:
            if h not in covered:
                fl.add(f_missing if h in missing else f_out)
            h += HOUR_MS
        return fl

    def finalize(self, t0, cpu0):
        day, tmp, inv = self.day, self.tmp, self.inv
        kept = {}
        for key, e in self.index.items():                                     # one content per key (amendment 1) -> per-market legs / hours
            k = kept[key] = choose(e)
            self.n_leg[e[5]] += k[3]
            if e[6] >= 0 and k[3]:
                self.leg_hours[e[5]][e[6] // HOUR_MS * HOUR_MS] += k[3]
        env = self.envelope()
        r_cov = {stem_ms(H['stem']) for H in self.rest_hours if not H.get('missing')}
        r_mis = {stem_ms(H['stem']) for H in self.rest_hours if H.get('missing')}
        w_cov = {stem_ms(H['stem']) for H in self.ws_hours if not H.get('missing')}
        w_mis = {stem_ms(H['stem']) for H in self.ws_hours if H.get('missing')}
        mrest, mflags, spans, polls = {}, {}, {}, {}
        for mid, W in self.pop.items():
            lo, hi = W['ws'] * 1000 - env['pre_recv_ms'], W['we'] * 1000 + env['post_settle_ms'] + env['margin_ms']
            fr = self.cover_flags(lo, hi, r_cov, r_mis, 'boundary_censored', 'hour_missing')
            fl = set(fr) | ({'no_records'} if self.n_rec[mid] == 0 else set())
            n_out = n_mis = 0
            if self.role != 'ID' and mid not in self.ws_pop:                 # GT window outside the declared WS rule: WS not read for it
                fl.add('no_own_fill_ws_skipped')
                n_out = n_mis = -1
            elif self.role != 'ID':
                n_out = sum(n for h, n in self.leg_hours[mid].items() if h not in w_cov and h not in w_mis)
                n_mis = sum(n for h, n in self.leg_hours[mid].items() if h in w_mis)
                fl |= ({'no_frames'} if self.n_frames[mid] == 0 else set()) | ({'fills_outside_ws'} if n_out else set()) | ({'fills_in_missing_ws_hour'} if n_mis else set())
            S = self.mk.get(mid)
            if S and S['slug'] and S['slug'] != W['slug']:
                fl.add('slug_mismatch')
            if S and S['up_bad']:
                fl.add('up_index_mismatch')
            if W.get('outcome_sealed'):
                fl.add('outcome_sealed')
            pr = collections.Counter(why for t_, why in self.poll.get(mid, ()) if lo <= t_ // 1_000_000 < hi)
            if pr:
                fl.add('poll_incomplete')
            polls[mid] = pr
            mrest[mid], mflags[mid], spans[mid] = fr, fl, (lo, hi, n_out, n_mis)
        counts, sc = {}, {}
        # ---- records / legs: patch receipts, n_recv, variants and statuses from the index ----
        not_kept = collections.Counter()
        for kind in ('records', 'legs'):
            sch = schema(kind)
            w = pq.ParquetWriter(tmp / f'{kind}_{day}.parquet', sch)
            n, st_c = 0, collections.Counter()
            for p in self.parts[kind]:
                d = pq.read_table(p).to_pydict()
                keys, dgs = d.pop('_key'), d.pop('_dg')
                keep = [kept[k][0] == g for k, g in zip(keys, dgs)]          # rows of a content that was not kept are counted, not written
                if not all(keep):
                    not_kept[kind] += keep.count(False)
                    d = {c: [v for v, k_ in zip(col, keep) if k_] for c, col in d.items()}
                    keys = [k for k, k_ in zip(keys, keep) if k_]
                E, K = [self.index[k] for k in keys], [kept[k] for k in keys]
                d['first_recv_ns'] = [recv_all(e) for e in E]
                if kind == 'records':
                    d['n_recv'] = [k[2] for k in K]
                    d['x_n_variants'] = [1 + len(e[4] or ()) for e in E]
                    d['x_n_recv_other'] = [e[2] + e[3] - k[2] for e, k in zip(E, K)]
                    d['x_kept'] = [k[1] for k in K]
                    fls = [set(filter(None, f.split('+'))) for f in d.pop('_flags')]
                else:
                    fls = [set(filter(None, a.split('+'))) | set(filter(None, b.split('+'))) for a, b in zip(d.pop('_rflags'), d.pop('_lflags'))]
                d['status'] = [status(f | ({'conflict'} if e[4] else set()) | mrest[m]) for f, e, m in zip(fls, E, d['mid'])]
                st_c.update(d['status'])
                n += len(d['status'])
                w.write_table(pa.Table.from_pydict(d, schema=sch))
            w.close()
            counts[kind], sc[kind] = n, dict(st_c.most_common())
        need(counts['records'] == len(self.index), f"records rows {counts['records']} != unique keys {len(self.index)} (a dropped row)")
        need(counts['legs'] == sum(k[3] for k in kept.values()), f"legs rows {counts['legs']} != legs of the kept contents (a dropped row)")
        n_var = sum(len(e[4] or ()) for e in self.index.values())
        need(not_kept['records'] == n_var and not_kept['legs'] == sum(v[4] for e in self.index.values() for v in (e[4] or {}).values()) + sum(
            e[8] for e in self.index.values() if e[4]) - sum(kept[k][3] for k, e in self.index.items() if e[4]),
             f'conflict variants: {dict(not_kept)} rows not kept do not match the index (a dropped or duplicated variant)')
        counts['variants_not_kept'] = dict(records=not_kept['records'], legs=not_kept['legs'])
        # ---- tob ----
        if self.role != 'ID':
            sch, n, st_c = schema('tob'), 0, collections.Counter()
            w = pq.ParquetWriter(tmp / f'tob_{day}.parquet', sch)
            for p in self.parts['tob']:
                t = pq.read_table(p, schema=sch)
                st_c.update(t.column('status').to_pylist())
                n += t.num_rows
                w.write_table(t)
            w.close()
            counts['tob'], sc['tob'] = n, dict(st_c.most_common())
            need(n == sum(self.n_frames.values()), f'tob rows {n} != frames parsed {sum(self.n_frames.values())} (a dropped row)')
        # ---- depth: close writers, empty files for depth markets without frames, count statuses ----
        sc['frames'], sc['levels'], counts['frames'], counts['levels'] = collections.Counter(), collections.Counter(), 0, 0
        for mid in sorted(self.depth):
            W = self.dw.pop(mid, None)
            if W is None:
                pq.write_table(self.SF.empty_table(), tmp / f'depth/{day}_{mid}_frames.parquet')
                pq.write_table(self.SL.empty_table(), tmp / f'depth/{day}_{mid}_levels.parquet')
            else:
                W[0].close()
                W[1].close()
            fr = pq.read_table(tmp / f'depth/{day}_{mid}_frames.parquet', columns=['status'])
            lv = pq.read_table(tmp / f'depth/{day}_{mid}_levels.parquet', columns=['status'])
            sc['frames'].update(fr.column('status').to_pylist())
            sc['levels'].update(lv.column('status').to_pylist())
            counts['frames'] += fr.num_rows
            counts['levels'] += lv.num_rows
            need(fr.num_rows == self.fid.get(mid, 0), f'depth frames of {mid}: {fr.num_rows} rows != {self.fid.get(mid, 0)} parsed (a dropped row)')
        sc['frames'], sc['levels'] = dict(sc['frames'].most_common()), dict(sc['levels'].most_common())
        # ---- markets ----
        M = {c: [] for c in MARKETS}
        for mid in sorted(self.pop):
            W, S = self.pop[mid], self.mk.get(mid)
            snap = dict(zip(POLICY, S['snap'])) if S else {}
            ms = self.mstate.get(mid)
            row = dict(day=day, mid=mid, slug=W['slug'], ws=W['ws'], we=W['we'], outcome=W['outcome'], outcome_seen_ns=W['outcome_seen_ns'],
                       feeRateBps=snap.get('feeRateBps') if type(snap.get('feeRateBps')) is int else -1,
                       makerFeeBps=snap.get('makerFeeBps') if type(snap.get('makerFeeBps')) is int else -1,
                       spreadThreshold=float(snap['spreadThreshold']) if isinstance(snap.get('spreadThreshold'), (int, float)) else math.nan,
                       shareThreshold=float(snap['shareThreshold']) if isinstance(snap.get('shareThreshold'), (int, float)) else math.nan,
                       rewards=S['rewards'] if S else '',
                       isBoosted=int(snap['isBoosted']) if isinstance(snap.get('isBoosted'), bool) else -1,
                       boostStartsAt=str(snap.get('boostStartsAt') or ''), boostEndsAt=str(snap.get('boostEndsAt') or ''),
                       outcome_statuses=S['outcomes'] if S else '', market_status=S['status'] if S else '', trading_status=S['trading'] if S else '',
                       resolution=S['resolution'] if S else '', category_slug=S['slug'] if S else '', latest_recv_ns=S['recv'] if S else -1,
                       n_pages=S['n_pages'] if S else 0, n_records=self.n_rec[mid], n_legs=self.n_leg[mid],
                       changed_fields=','.join(k for k in POLICY if S and k in S['changed']),
                       rewards_variants=json.dumps([dict(first_recv_ns=r_, rewards=json.loads(j_)) for j_, r_ in S['rv'].items()], separators=(',', ':')) if S else '',
                       rest_lo_ms=spans[mid][0], rest_hi_ms=spans[mid][1], n_frames=self.n_frames[mid], first_upd_ms=ms[1] if ms else -1,
                       last_upd_ms=ms[0] if ms else -1, n_legs_outside_ws=spans[mid][2] if self.role != 'ID' else -1,
                       n_legs_in_missing_ws_hour=spans[mid][3] if self.role != 'ID' else -1,
                       n_poll_incomplete=sum(polls[mid].values()) if self.mode == 'day' else -1,
                       poll_incomplete_reasons=json.dumps(dict(sorted(polls[mid].items())), separators=(',', ':')) if self.mode == 'day' else '',
                       depth=int(mid in self.depth), status=status(mflags[mid], MK_FLAGS))
            for c in MARKETS:
                M[c].append(row[c])
        pq.write_table(pa.Table.from_pydict(M, schema=schema('markets')), tmp / f'markets_{day}.parquet')
        counts['markets'], sc['markets'] = len(M['mid']), dict(collections.Counter(M['status']).most_common())
        # ---- inventory (relations from the written records) ----
        rec = pq.read_table(tmp / f'records_{day}.parquet', columns=['sid', 'tx', 'tk_hash', 'settle_ms', 'exec_ms', 'first_recv_ns'])

        def per(col):
            if rec.num_rows == 0:
                return {}
            c = collections.Counter(pc.value_counts(rec.column(col)).field('counts').to_pylist())
            return {str(k): int(v) for k, v in sorted(c.items())}
        def nonempty(col):
            return col.filter(pc.not_equal(col, '')) if len(col) else col
        mk_all = nonempty(pq.read_table(tmp / f'legs_{day}.parquet', columns=['mk_hash']).column('mk_hash').combine_chunks())
        mh, th = pc.unique(mk_all), pc.unique(nonempty(rec.column('tk_hash').combine_chunks()))
        hashes = dict(taker_hashes=len(th), maker_hashes=len(mh), taker_hashes_also_maker=int(pc.sum(pc.is_in(th, value_set=mh)).as_py() or 0) if len(th) else 0,
                      maker_hashes_with_fills_in_several_records=sum(1 for c in pc.value_counts(mk_all).field('counts').to_pylist() if c > 1) if len(mk_all) else 0)
        all_sids = rec.column('sid').to_pylist()
        sids = [s for s in all_sids if s and not s.startswith('nosid:')]
        fr_ns, ex = np.asarray(rec.column('first_recv_ns'), np.int64), np.asarray(rec.column('exec_ms'), np.int64)
        okx = ex >= 0
        ids = dict(unique_keys=len(self.index), rows_with_sid=len(sids), distinct_sid=len(set(sids)), sid_column_unique=len(set(all_sids)) == len(all_sids),
                   synthetic_sids=len(all_sids) - len(sids),
                   key_kinds=dict(collections.Counter('nosid' if k.startswith('nosid:') else 'sid' for k in self.index)),
                   records_per_transactionHash=per('tx'), records_per_taker_hash=per('tk_hash'),
                   n_recv=dict(collections.Counter(min(kept[k][2], 50) for k in self.index).most_common()),
                   keys_with_conflict=sum(1 for e in self.index.values() if e[4]), conflict_receipts=sum(e[3] for e in self.index.values()),
                   x_kept=dict(collections.Counter(k[1] for k in kept.values()).most_common()), order_hashes=hashes)
        arrs = {k: qs(np.frombuffer(a, {'d': np.float64, 'q': np.int64, 'f': np.float32, 'H': np.uint16}[a.typecode])) for k, a in sorted(inv.a.items())}
        arrs['first_recv_minus_exec_ms'] = qs((fr_ns[okx] // 1_000_000 - ex[okx]).astype(float)) if okx.any() else dict(n=0)
        INV = dict(day=day, version=C.VERSION, role=self.role, mode=self.mode,
                   note='computed from the hour files of extract_<day>.json; record-level items over UNIQUE records (first canonical content), '
                        f'market objects over the first page per market per hour, frame leaves over every {FRAME_SAMPLE}th frame, string formats over '
                        f'the first {FMT_SAMPLE} new records per REST hour; key sets, types and counters otherwise over all',
                   rest=dict(ids=ids, clocks_and_lags_ms={k: v for k, v in arrs.items() if not k.startswith(('frame', 'recv_minus_upd', 'levels'))},
                             receipt_lag_ms=env.get('receipt_lag_ms')),
                   ws={k: v for k, v in arrs.items() if k.startswith(('frame', 'recv_minus_upd', 'levels'))}, **inv.dump())
        (tmp / f'inventory_{day}.json').write_text(json.dumps(INV, indent=1, default=str))
        # ---- receipt ----
        outs = {}
        for name in self.outputs()[:-1]:
            p = tmp / name
            outs[name] = dict(bytes=p.stat().st_size, sha256=sha(p))
        flags = {}
        for k, v in sc.items():
            fc = flags[k] = collections.Counter()
            for s_, n_ in v.items():
                for f in s_.split('+'):
                    fc[f] += n_
            flags[k] = dict(fc.most_common())
        R = dict(version=C.VERSION, module='pf_mt_extract', day=day, role=self.role, mode=self.mode, asof=dt.datetime.now(dt.timezone.utc).isoformat(),
                 argv=sys.argv, sources={str(Path(__file__).resolve()): sha(Path(__file__).resolve()), str(Path(C.__file__).resolve()): sha(Path(C.__file__).resolve())},
                 inputs=self.inputs, population=dict(n_markets=len(self.pop), depth_mids=sorted(self.depth)),
                 rest=dict(planned=self.rest_plan, read=self.rest_stems, clipped=self.rest_clip, missing=sorted(H['stem'] for H in self.rest_hours if H.get('missing')),
                           hours=self.rest_hours),
                 ws=dict(planned=self.ws_plan, read=self.ws_stems, clipped=self.ws_clip, missing=sorted(H['stem'] for H in self.ws_hours if H.get('missing')),
                         hours=self.ws_hours),
                 envelope=env, markets_flagged={f: sorted(m for m in self.pop if f in mflags[m]) for f in MK_FLAGS},
                 ws_population=self.ws_sel or dict(rule='every population market' if self.role == 'FILL' else 'no WS pass (ID day)'),
                 events=dict(read=[H['stem'] for H in self.ev_hours if not H.get('missing')], missing=sorted(H['stem'] for H in self.ev_hours if H.get('missing')),
                             hours=self.ev_hours, rule='MATCH_POLL_INCOMPLETE events with recv_ns inside a market REST envelope [rest_lo_ms, rest_hi_ms) -> poll_incomplete',
                             note='day runs only (a probe reads its two hour files only)', population_markets_flagged=sum(1 for m in self.pop if polls[m]),
                             population_events_in_envelope=sum(sum(v.values()) for v in polls.values())),
                 status_orders=dict(records=list(FLAGS), legs=list(FLAGS), tob=list(WS_FLAGS), frames=list(WS_FLAGS), levels=list(LEVEL_FLAGS),
                                    markets=list(MK_FLAGS), ok=C.STATUS_OK, joiner='+', rule="a status is 'ok' or flags of its list joined by '+' in list order"),
                 counts=counts, status_counts=sc, flag_counts=flags, outputs=outs, definitions=DEFINITIONS,
                 cpu_s=round(time.process_time() - cpu0, 3), wall_s=round(time.time() - t0, 3), vmhwm_mib=round(guard('finalize'), 1), guard_mib=C.GUARD_MIB)
        (tmp / f'extract_{day}.json').write_text(json.dumps(R, indent=1, default=str))
        # ---- publish: hard links refuse an existing path; the receipt goes last ----
        (self.out / 'depth').mkdir(parents=True, exist_ok=True)
        for name in self.outputs():
            os.link(tmp / name, self.out / name)
        return R


def recv_all(e):
    """The earliest receipt of any content under an index entry."""
    return e[1] if not e[4] else min(e[1], min(v[1] for v in e[4].values()))


def choose(e):
    """Amendment 1: the content kept for a key -> (digest, rule, its receipts, its legs). The variant that reconciles exactly; several ->
    the most receipts (tie -> first seen); none -> the first. Entry: [dg, recv, n, n_other, {dg: [n, recv, ok, order, legs]}, mid, settle, ok, legs]."""
    if not e[4]:
        return e[0], 'only', e[2], e[8]
    vs = [(e[0], e[2], e[7], 0, e[8])] + [(dg, v[0], v[2], v[3], v[4]) for dg, v in e[4].items()]
    ok = [v for v in vs if v[2]]
    if not ok:
        return e[0], 'first_none_reconciles', e[2], e[8]
    if len(ok) == 1:
        return ok[0][0], 'reconciles', ok[0][1], ok[0][4]
    top = max(v[1] for v in ok)
    best = [v for v in ok if v[1] == top]
    k = min(best, key=lambda v: v[3])
    return k[0], ('most_receipts' if len(best) == 1 else 'reconciles_tie_first_seen'), k[1], k[4]


def population(day):
    """Population markets of a day -> ({mid: dict(slug, ws, we, outcome, outcome_seen_ns)}, depth mids, input receipts). FILL / ID: the step1
    registry, which must hold exactly the day's 288 window starts; GT: the scope windows (ws / slug / outcome from the export's registry when
    present, else the scope's ws and the slug 'btc-updown-5m-<ws>'). Depth: FILL -> the manifest's `complete` of the day; GT -> every window."""
    r, d0, src = role_of(day), day_ms(day) // 1000, {}

    def sealed(ns_):
        """B1: True when a registry time lies in an hour that is not consumed (the old padded step1 read sealed hours)."""
        return ns_ >= 0 and not C.allowed_hour(dt.datetime.fromtimestamp(ns_ // 10 ** 9, dt.timezone.utc).strftime('%Y%m%d_%H'))

    def reg(path, only=None):
        src[str(path)] = sha(path)
        rows = pq.read_table(path, columns=['mid', 'slug', 'ws', 'we', 'outcome', 'outcome_seen_ns'],
                             filters=None if only is None else [('mid', 'in', sorted(only))]).to_pylist()
        out = {}
        for x in rows:
            seen = int(x['outcome_seen_ns']) if x['outcome_seen_ns'] is not None else -1
            sl = sealed(seen)                                                # a sealed value is blanked here and never stored
            out[int(x['mid'])] = dict(slug=str(x['slug']), ws=int(x['ws']), we=int(x['we']), outcome='' if sl else str(x['outcome'] or ''),
                                      outcome_seen_ns=-1 if sl else seen, outcome_sealed=sl)
        return out, len(rows)
    if r == 'GT':
        src[str(GT_SCOPE)] = sha(GT_SCOPE)
        wins = json.loads(Path(GT_SCOPE).read_text())['windows']
        ws_of = {int(w['market']): int(w['ws']) for w in wins}
        need(len(ws_of) == len(wins), 'the GT scope repeats a market')
        rg = reg(GT_REGISTRY, only=set(ws_of))[0] if Path(GT_REGISTRY).exists() else {}
        pop = {}
        for m, ws in ws_of.items():
            if m in rg:
                need(rg[m]['ws'] == ws, f'GT market {m}: registry ws {rg[m]["ws"]} != scope ws {ws}')
                pop[m] = rg[m]
            else:
                pop[m] = dict(slug=f'btc-updown-5m-{ws}', ws=ws, we=ws + WINDOW_S, outcome='', outcome_seen_ns=-1, outcome_sealed=False)
        depth = set(pop)
    else:
        pop, n = reg(C.STEP1 / day / 'registry.parquet')
        need(n == len(pop) and sorted(W['ws'] for W in pop.values()) == [d0 + WINDOW_S * i for i in range(288)],
             f'the {day} registry is not exactly the 288 window starts of the day (a missing window would drop its records)')
        depth = set()
        if r == 'FILL':
            src[str(BOOK_MANIFEST)] = sha(BOOK_MANIFEST)
            depth = {int(m) for m in json.loads(Path(BOOK_MANIFEST).read_text())['complete'][day]}
    need(depth <= set(pop), 'a depth market is not in the population')
    for m, W in pop.items():
        need(W['we'] - W['ws'] == WINDOW_S and d0 <= W['ws'] < d0 + 86400, f'market {m}: window [{W["ws"]}, {W["we"]}) is not a 5-minute window of {day}')
    return pop, depth, src


def main(argv):
    need(len(argv) >= 2 and argv[1] in ('day', 'probe'), 'usage: pf_mt_extract.py day <YYYYMMDD> | probe <out_dir> rest:<stem> ws:<stem>')
    if argv[1] == 'day':
        need(len(argv) == 3, 'usage: pf_mt_extract.py day <YYYYMMDD>')
        R = Day(argv[2], C.OUT).run()
    else:
        need(len(argv) >= 4, 'usage: pf_mt_extract.py probe <out_dir> rest:<stem> ws:<stem>')
        out = Path(argv[2]).resolve()
        need(out != Path(C.OUT).resolve() and Path(C.OUT).resolve() not in out.parents, f'probe output {out} may not lie under OUT (write-once day outputs)')
        rs, wss = [a[5:] for a in argv[3:] if a.startswith('rest:')], [a[3:] for a in argv[3:] if a.startswith('ws:')]
        need(len(rs) + len(wss) == len(argv) - 3, 'probe hours are rest:<stem> or ws:<stem>')
        need({s[:8] for s in rs + wss} == {PROBE_DAY}, f'probe hours must be of {PROBE_DAY}')
        R = Day(PROBE_DAY, out, rs, wss, mode='probe').run()
    print(json.dumps(dict(day=R['day'], mode=R['mode'], counts=R['counts'], status_counts=R['status_counts'], envelope={k: v for k, v in R['envelope'].items() if k != 'rule'},
                          cpu_s=R['cpu_s'], vmhwm_mib=R['vmhwm_mib']), default=str), flush=True)
    return R


if __name__ == '__main__':
    try:
        main(sys.argv)
    except GuardStop as e:
        sys.exit(f'[guard] {e}')
