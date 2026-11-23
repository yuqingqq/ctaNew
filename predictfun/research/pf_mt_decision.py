"""TOP-3 DECISION PATHS (P-2026-005, MAKER-TRACE-RECONSTRUCTION-v1 amendment 3; session aea5bfb9 development, 2026-11-23). USER: "check like
top 3 makers, totally understand what they are doing, like under what condition they post and what conditions they take, we need their
decision path." Binding spec: predictfun/reviews/maker_trace_20261122/decision_contract.md (frozen before its data run). DESCRIPTIVE ONLY, on
consumed days: no strategy, imitation, tradeable or profitability claim. p_fair is a descriptive reference, not a claim about the makers' model.
Every rule below is declared before any data run; synthetic proof (positive controls, refusals): pf_mt_decision_selftest.py.
POPULATION the contract's ex-ante top 3 (TOP3; econ/identities.json ex_ante.top10_shares ranks 1-3 must equal it, else refuse); their
       econ/fills_valued rows (maker AND taker legs) on the measurement days; on BOOK markets (pf_mt_econ.load_book: the frozen 'complete'
       list, every one required) their lifecycle rows life/<day>_<mid>_{orders,x_links}.parquet. A PAIR (address x market) is excluded whole,
       with a status, when its market is missing or flagged MKT_EXCLUDE (no_market / market_<flag>), when any own row is not status ok
       (fill_status_<s>), when a row has no settle_ms (no_settle) or when its span is empty (span_empty); its would-be grid rows are counted.
CLOCKS never mixed. BTC: btc/btc_<day>.parquet (pf_mt_btc 40ee591ea: one file per day holding its own 100 ms grid on Binance E_ms; row g =
       the last bookTicker with E_ms <= g, age_ms = g - E_ms (-1 when none), status ok | stale | gap | no_data; usable = ok, 0 <= age_ms <=
       BTC_MAX_AGE_MS, finite mid > 0; age_ms -1 or a non-ok status is missing (counted), never fresh; an ok row with age_ms < -1 refuses; a
       g_ms in two files: the usable row wins, else the own-day file, two usable rows must agree). Predict book: tob upd_ms. A fill: its
       match bound hi = fills_valued.settle_ms (UUIDv7 creation: an upper bound of the match), on every market.
LAG    every feature of a decision with bound b is evaluated at x = b - L, L in LAGS ms. A conclusion that differs across L is lag_sensitive.
BTC    at x: grid row i = floor(x / 100); r_ws = 1e4 ln(mid_i / mid_ws) (0 before ws: the start is not set yet, so p_fair = 0.5);
       r_1s / r_5s / r_30s = 1e4 ln(mid_i / mid_{i-10 / 50 / 300}); sig_60 = sqrt(sum of the 600 squared 100 ms returns (bps) in (i-600, i]
       / 60 s) in bps/sqrt(s), floored at SIG_FLOOR; tau = (we - x) / 1000 s; z = r_ws / (max(sig_60, floor) sqrt(tau)); p_fair = Phi(z).
       Every grid point in [i - 600, i] (and ws when x >= ws) must be usable (status 'ok', finite mid > 0, 0 <= age_ms <= BTC_MAX_AGE_MS),
       else btc_invalid; outside the loaded grid btc_missing; tau <= 0 after_we. Only returns, never the level against the Chainlink start.
BOOK   the last tob frame (upd_ms, recv_ns order) with upd_ms <= x in the market, with pf_mt_econ's MID rules: book_no_frame / book_frame_status
       (a BAD_FRAME flag) / book_one_sided / book_crossed / book_stale (x - upd_ms > STALE_MS); best bid / ask, mid, spread, size at best,
       imbalance (bid_sh - ask_sh) / (bid_sh + ask_sh); g_buy = 100 (p_fair - best_ask), g_sell = 100 (best_bid - p_fair), in cents.
       Frames before ws - TOB_KEEP_BEFORE_MS are not loaded (a query whose last frame is older reads book_no_frame instead of book_stale).
OWN    net YES position = sum of s q over the pair's own fills (both roles; fees excluded) with settle_ms < x; time since the last of them;
       on BOOK the number of its proven-present orders per side (PRIMARY-scenario persistence window [persist_lo, persist_hi] covers x).
TAKE   grid per pair: t = t0 + 100 k over [max(first - 30 s, ws - 300 s), min(last + 30 s, we)) (first / last own fill, any role; t0 = the
       100 ms floor). Event: a take RECORD (unique sid with role taker - rows are actions) with settle_ms in [t, t + 100), buy (s = +1) and
       sell (s = -1) separately; features at t - L; rows with a feature status are excluded and counted. Tables (rows, events, rate) per
       address, pooled and per day (4 days: point estimates, no intervals): 1-D g / r_1s / r_5s / tau / own position sign / time since own fill /
       BOOK resting state; 2-D g x sign(r_1s), g x sign(r_5s), g x position sign, g x tau (60 s). THRESHOLD: the maximum-likelihood two-rate
       split of the g buckets (edges G_EDGES), both sides >= N_MIN_SPLIT rows, ties to the smallest edge, in the slices all / r1_aligned (sign
       of r_1s = the side) / r1_against / r1_zero / r5_aligned / r5_against. LOGISTIC (one declared fit, no penalty, no tuning): y ~ 1 + g
       (clipped +-GCLIP_LOGIT c) + r_1s + r_5s + tau (min) + pos_long + pos_short per L x side x (address, pooled); IRLS from the base-rate
       intercept (warm start from the previous L when it converged), constant columns dropped and named, at most LOGIT_MAX_PASS data passes
       and FIT_CPU_CAP_S CPU per L (then status cpu_cap; only converged fits enter a claim); point estimates only (no intervals: rows are
       serially dependent and G = 4 days < 5).
POST   AT FILL (each maker leg = one order fill): offset = 100 (p - p_fair), edge = 100 s (p_fair - p), ticks behind the shown best on its side
       (bid: best_bid - p, ask: p - best_ask; < 0 = inside), size, tau, position before. AT POSTING (BOOK lifecycle orders): the same at the
       posting window hi (post_hi_ms = the first fill; scenario-free), with lo (post_lo_ms >= 0) and mid ((lo + hi) // 2) per scenario as
       sensitivity, plus size_lb and post_class. SKEW: OLS offset = a + b (position / 100 sh) per side and clock. REFILL / REPRICE (x_links,
       x_latest rows): gap quantiles; the tick change against dp = 100 (p_fair(next_first_hi - L) - p_fair(prev_last_hi - L)) (OLS slope,
       correlation, sign agreement) and the BTC return. REMOVAL: orders whose removal_set holds cancelled_or_expired (cancel_possible) vs
       those whose set is exactly {exhausted} (exhausted_only); window [removal_lo, removal_hi] (both bounded); adverse = -s dp, abs = |dp|,
       adverse_r = -s r (BTC log return, bps); PHASE = pre-ws or live [ws, we) by the window start at L = 0 (windows straddling ws or ending
       at / after we are statuses). NULL declared here: per order N_DRAWS windows of the same (100 ms-rounded) width drawn uniformly, with
       replacement, from the usable placements entirely inside the SAME PHASE of the same market (pre: [ws - 300 s, ws)), or of the same
       market-day pool when the market has < MIN_PLACE (no placement: a status); seeded per scenario x day; group-mean replicates, one-sided
       p = (1 + #(null >= observed)) / (1 + draws), z, per-order mid-rank percentile; per phase and both; cancel minus exhausted paired.
INVENTORY rows are actions (pair, record, role; self-trades excluded): does the action reduce |position before| (s x position < 0), by
       |position| bucket; P(sell | long) vs P(sell | short).
TIMELINES per address, chosen in code before any feature is computed: the BOOK pair with the most own rows; the non-BOOK pair with the most own
       take records; the non-BOOK pair with the most own maker rows (not already chosen); ties to the smallest (day, mid). 100 ms rows at L = 0.
GUARDS days: measurement days only (refused before any path is built); BTC files only for BTC_DAYS (pf_mt_common.Forbidden before access),
       rows before Oct 23 23Z clipped (counted), any row at or after Oct 28 00:00Z refuses; a missing BTC grid of a measurement day refuses;
       an address outside the contract's top 3 refuses; fills_valued rows on a non-consumed day refuse; outputs write-once (a partial
       directory renamed at the end); VmHWM self-guard pf_mt_common.GUARD_MIB (never raised).
OUTPUTS <out> = pf_mt_common.RECEIPT / 'decision': statuses.csv, take_hazard_1d.csv, take_hazard_2d.csv, take_threshold.csv, take_logit.csv,
       take_events.csv, post_fill.csv, post_fill_tau.csv, post_posting.csv, post_skew.csv, links.csv, removal.csv, inventory.csv,
       address_summary.csv, timeline_<scope>_<day>_<mid>.csv, claims_decision.json (computed predicates with n), decision.json (receipt).
Usage (run_light): MT_RUN=v2 pf_mt_decision.py run      schemas: pf_mt_decision.py schemas [root]      selftest: pf_mt_decision_selftest.py"""
import datetime as dt
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.special import expit, ndtr, xlogy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_mt_common as C  # noqa: E402
import pf_mt_econ as E  # noqa: E402

need, OK = C.need, C.STATUS_OK
VERSION = 'decision_v1'
CONTRACT = 'predictfun/reviews/maker_trace_20261122/decision_contract.md'

# ------------------------------------------------------------------ declared constants (none fitted) ------------------------------------------------------------------
TOP3 = ('0x6Da6Cb464F92AE7aD4Ec3d239c81719Cb1D0Ae03', '0x7d32c568a9bEA1714036972601c51F4902C67091',
        '0x57FbC891E89184DA071AE9D97E745531cd7F718a')            # decision_contract.md population table, ranks 1-3
LOW3 = tuple(a.lower() for a in TOP3)
SCOPES = ('rank1', 'rank2', 'rank3', 'pooled')
LAGS = (0, 50, 250)
NL = len(LAGS)
STEP = 100                       # ms: decision grid and BTC grid
PAD_MS, PRE_MS = 30_000, 300_000
R_STEPS = (('r_1s', 10), ('r_5s', 50), ('r_30s', 300))
SIG_STEPS, SIG_FLOOR = 600, 0.2  # sig_60 window (100 ms returns) and its floor (bps / sqrt(s))
BTC_MAX_AGE_MS = 1_000
STALE_MS = E.STALE_MS            # book staleness (pf_mt_econ MID rule)
TOB_KEEP_BEFORE_MS = PRE_MS + 5_000


def day_ms(d):
    return int(dt.datetime.strptime(d, '%Y%m%d').replace(tzinfo=dt.timezone.utc).timestamp()) * 1000


def prev_day(d):
    return (dt.datetime.strptime(d, '%Y%m%d') - dt.timedelta(days=1)).strftime('%Y%m%d')


BTC_DAYS = (C.ID_DAY,) + C.MEASURE_DAYS                   # bookTicker files that may be opened (Oct 23: its 23Z padding only)
PAD_LO_MS = day_ms(C.MEASURE_DAYS[0]) - 3_600_000          # 2026-10-23 23:00Z
BTC_END_MS = day_ms(C.MEASURE_DAYS[-1]) + 86_400_000       # 2026-10-28 00:00Z: never
MKT_EXCLUDE = ('boundary_censored', 'hour_missing', 'up_index_mismatch')
POS_EPS = 1e-6
G_EDGES = np.round(np.arange(-10.0, 10.0 + 1e-9, 0.5), 6)  # cents; bucket b = [E[b-1], E[b])
NG = len(G_EDGES) + 1
M1 = (np.array([-5., -2., -1., -.5]), np.array([.5, 1., 2., 5.]))      # r_1s buckets (bps): negative edges, positive edges, 0 its own
M5 = (np.array([-10., -5., -2., -1.]), np.array([1., 2., 5., 10.]))    # r_5s
NM = 11
NTAU, NTAU60 = 20, 10
S_EDGES = np.array([500., 1_000., 2_000., 5_000., 10_000., 30_000., 60_000.])   # time since own fill (ms)
NS = len(S_EDGES) + 2
POS_B = np.array([50., 200., 1_000.])                    # |position| buckets (shares) for the inventory tests
N_MIN_SPLIT = 200
GCLIP_LOGIT = 20.
LOGIT_COLS = ('const', 'g_c', 'r_1s_bps', 'r_5s_bps', 'tau_min', 'pos_long', 'pos_short')
LOGIT_MAX_IT, LOGIT_MAX_PASS, LOGIT_TOL = 30, 30, 1e-10   # Newton steps, data passes per L (incl. step halvings), deviance tolerance
FIT_CPU_CAP_S = 45.              # CPU per L for the fits (resource rule, declared; a model still running there ends 'cpu_cap', never claimed)
N_DRAWS, SEED = 500, 20261123
MIN_PLACE = 100                  # usable same-phase placements a market needs for its own removal null (else the market-day pool)
PHASES = ('pre', 'live', 'all')  # removal phases: pre-ws, live [ws, we), both (each order against its own phase-matched null)
REM_METRICS = ('adverse', 'abs', 'adverse_r')   # -s dp (c), |dp| (c), -s r (bps; BTC log return, informative before ws too)
CENTRE_TOL_C, TEN_X, OLS_MIN = 0.5, 10., 30
CHUNK_ROWS, FIT_CH = 300_000, 1_000_000
PRIMARY = 'S_FIFO'
SCEN = tuple(C.SCENARIOS)
# pf_mt_lifecycle.INVALID (an order whose identity / legs are invalid; checked equal in the selftest)
ORDER_INVALID = frozenset(('identity_conflict', 'dup_hash_in_record', 'unplaceable_leg', 'subcent_price', 'malformed_identity', 'amount_mismatch',
                           'leg_recon_mk_sum', 'leg_recon_tk_amount'))
ROW_ST = ('ok', 'btc_missing', 'btc_invalid', 'after_we', 'book_no_frame', 'book_frame_status', 'book_one_sided', 'book_crossed', 'book_stale')
NRS = len(ROW_ST)
BOOK_ST = ('ok', 'book_no_frame', 'book_frame_status', 'book_one_sided', 'book_crossed', 'book_stale')

# ---- inputs: columns read and their footer dtypes ----
FV_COLS = ('day', 'mid', 'sid', 'role', 'addr', 's', 'q', 'p', 'settle_ms', 'self_trade', 'status')
TOB_COLS = ('mid', 'upd_ms', 'recv_ns', 'n_bid', 'n_ask', 'best_bid', 'best_bid_sh', 'best_ask', 'best_ask_sh', 'status')
MKT_SCHEMA = dict(mid='int64', ws='num', we='num', status='str')
BTC_SCHEMA = dict(g_ms='int64', mid='float64', age_ms='num', status='str')
ORD_SCHEMA = {c: C.SCHEMAS['orders'][c] for c in ('scenario', 'mid', 'mk_hash', 'mk_signer', 'y_side', 'y_ptick', 'n_fills', 'size_lb', 'first_hi_ms',
                                                  'last_hi_ms', 'post_lo_ms', 'post_hi_ms', 'post_class', 'persist_lo_ms', 'persist_hi_ms',
                                                  'removal_set', 'removal_lo_ms', 'removal_hi_ms', 'removal_class', 'status')}
LNK_SCHEMA = dict(mid='int64', mk_signer='str', y_side='str', kind='str', prev_ptick='int64', next_ptick='int64', prev_last_hi_ms='int64',
                  next_first_hi_ms='int64', gap_ms='int64', x_latest='bool', status='str')
FIT_DT = np.dtype([('d', 'i1'), ('gb', 'f4'), ('gs', 'f4'), ('r1', 'f4'), ('r5', 'f4'), ('tau', 'f4'), ('ps', 'i1'), ('yb', 'i1'), ('ys', 'i1')])


def _g_labels():
    e = [f'{x:g}' for x in G_EDGES]
    return ['<' + e[0]] + [f'[{e[i]},{e[i + 1]})' for i in range(len(e) - 1)] + ['>=' + e[-1]]


def _m_labels(edges):
    n, p = [f'{x:g}' for x in edges[0]], [f'{x:g}' for x in edges[1]]
    return (['<' + n[0]] + [f'[{n[i]},{n[i + 1]})' for i in range(len(n) - 1)] + [f'[{n[-1]},0)', '=0', f'(0,{p[0]}]'] +
            [f'({p[i]},{p[i + 1]}]' for i in range(len(p) - 1)] + ['>' + p[-1]])


G_LAB = _g_labels()
SIGN_LAB = ('neg', 'zero', 'pos')
POS_LAB = ('short', 'flat', 'long')
TAU60_LAB = tuple(f'[{60 * i},{60 * (i + 1)})' for i in range(NTAU60))
S_LAB = tuple(['[0,500)'] + [f'[{int(S_EDGES[i])},{int(S_EDGES[i + 1])})' for i in range(len(S_EDGES) - 1)] + [f'>={int(S_EDGES[-1])}', 'none'])
TABLES = (('g', NG, True, G_LAB), ('r_1s', NM, False, _m_labels(M1)), ('r_5s', NM, False, _m_labels(M5)),
          ('tau', NTAU, False, tuple(f'[{30 * i},{30 * (i + 1)})' for i in range(NTAU))), ('pos', 3, False, POS_LAB), ('tsince', NS, False, S_LAB),
          ('rest', 5, False, ('none', 'bid', 'ask', 'both', 'na')),
          ('g_x_r1sign', NG * 3, True, tuple(f'{g}|{s}' for g in G_LAB for s in SIGN_LAB)),
          ('g_x_r5sign', NG * 3, True, tuple(f'{g}|{s}' for g in G_LAB for s in SIGN_LAB)),
          ('g_x_pos', NG * 3, True, tuple(f'{g}|{s}' for g in G_LAB for s in POS_LAB)),
          ('g_x_tau60', NG * NTAU60, True, tuple(f'{g}|{s}' for g in G_LAB for s in TAU60_LAB)))
TABLE_1D = ('g', 'r_1s', 'r_5s', 'tau', 'pos', 'tsince', 'rest')
TABLE_2D_DAILY = ('g_x_r1sign',)
SLICES = ('all', 'r1_aligned', 'r1_against', 'r1_zero', 'r5_aligned', 'r5_against')


def g_bucket(g):
    return np.searchsorted(G_EDGES, g, side='right')


def m_bucket(r, edges):
    b = np.full(len(r), 5, np.int64)
    n_, p_ = r < 0, r > 0
    b[n_] = np.searchsorted(edges[0], r[n_], side='right')
    b[p_] = 6 + np.searchsorted(edges[1], r[p_], side='left')
    return b


def sign3(r):
    return (np.sign(r) + 1).astype(np.int64)


def pos3(pos):
    return np.where(pos < -POS_EPS, 0, np.where(pos > POS_EPS, 2, 1)).astype(np.int64)


def ts_bucket(ts):
    b = np.searchsorted(S_EDGES, np.nan_to_num(ts, nan=0.), side='right')
    return np.where(np.isnan(ts), NS - 1, b).astype(np.int64)


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ------------------------------------------------------------------ guards, paths, schemas ------------------------------------------------------------------
def btc_path(btc_dir, day):
    """The BTC grid file of a day; refuses BEFORE any filesystem access unless the day is a declared bookTicker day."""
    if day not in BTC_DAYS:
        raise C.Forbidden(f'BTC grid day {day!r} is not an allowed bookTicker day (never opened)')
    return Path(btc_dir) / f'btc_{day}.parquet'


def _check_schema(path, want, kind):
    need(Path(path).is_file(), f'{kind}: input missing: {path}')
    sch = pq.read_schema(path)
    for c, t in want.items():
        need(c in sch.names, f'{kind}: required column {c} missing ({Path(path).name})')
        need(E._type_ok(sch.field(c).type, t), f'{kind}: column {c} is {sch.field(c).type}, expected {t} ({Path(path).name})')


def preflight(inp, btc_dir, days, book):
    """Every input path and footer schema, before any data is read -> {name: path}."""
    P = dict(fv=inp / 'econ' / 'fills_valued.parquet', identities=inp / 'econ' / 'identities.json')
    _check_schema(P['fv'], {c: E.FV[c] for c in FV_COLS}, 'fills_valued')
    need(P['identities'].is_file(), f'identities: input missing: {P["identities"]}')
    for d in days:
        bp = btc_path(btc_dir, d)
        need(bp.is_file(), f'missing BTC grid: btc/{bp.name} (measurement day {d}; refused before any data is read)')
        _check_schema(bp, BTC_SCHEMA, 'btc')
        P[('markets', d)], P[('tob', d)] = inp / f'markets_{d}.parquet', inp / f'tob_{d}.parquet'
        _check_schema(P[('markets', d)], MKT_SCHEMA, 'markets')
        _check_schema(P[('tob', d)], {c: C.SCHEMAS['tob'][c] for c in TOB_COLS}, 'tob')
        for m in book.get(d, []):
            for k, sch in (('orders', ORD_SCHEMA), ('x_links', LNK_SCHEMA)):
                p = inp / 'life' / f'{d}_{m}_{k}.parquet'
                need(p.is_file(), f'missing lifecycle output: life/{p.name} (BOOK market {m})')
                _check_schema(p, sch, k)
                P[(k, d, m)] = p
    return P


def _s(col):
    """An arrow string column, dictionary-decoded."""
    return pc.cast(col, pa.string()) if pa.types.is_dictionary(col.type) else col


def _np(col):
    return col.to_numpy() if isinstance(col, pa.ChunkedArray) else col.to_numpy(zero_copy_only=False)


def _flags(st):
    return {x for x in str(st).split('+') if x and x != OK}


# ------------------------------------------------------------------ loaders ------------------------------------------------------------------
def load_fv(path, days, cnt):
    """The top 3's fills_valued rows on `days` (streamed in batches; other addresses never leave the batch) -> (F arrays, status names)."""
    vs = pa.array(list(LOW3))
    parts, n_scan = [], 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=list(FV_COLS)):
        n_scan += b.num_rows
        m = pc.fill_null(pc.is_in(pc.utf8_lower(_s(b.column('addr'))), value_set=vs), False)
        if pc.any(m).as_py():
            parts.append(b.filter(m))
    cnt['fv_rows_scanned'] = n_scan
    need(parts, 'fills_valued: no row of the top 3 addresses')
    T = pa.Table.from_batches(parts)
    del parts
    day = _s(T.column('day'))
    ud = set(pc.unique(day).to_pylist())
    need(ud <= set(C.ALLOWED_DAYS), f'fills_valued: rows on a non-consumed day {sorted(ud - set(C.ALLOWED_DAYS))} (refused)')
    di = pc.index_in(day, value_set=pa.array(list(days)))
    keep = pc.is_valid(di)
    cnt['fv_rows_top3'] = T.num_rows
    T, di = T.filter(keep), di.filter(keep)
    cnt['fv_rows_top3_other_days'] = cnt['fv_rows_top3'] - T.num_rows
    need(T.num_rows > 0, 'fills_valued: no row of the top 3 on the measurement days')
    addr = _s(T.column('addr'))
    spell = {}
    for x in pc.unique(addr).to_pylist():
        spell.setdefault(x.lower(), set()).add(x)
    need(all(len(v) == 1 for v in spell.values()), f'fills_valued: an address appears under two spellings {spell}')
    role = pc.index_in(_s(T.column('role')), value_set=pa.array(['maker', 'taker']))
    need(role.null_count == 0, 'fills_valued: a role other than maker / taker')
    st = _s(T.column('status'))
    names = sorted(pc.unique(st).to_pylist())
    sid = _s(T.column('sid'))
    F = dict(d=_np(di).astype(np.int8), a=_np(pc.index_in(pc.utf8_lower(addr), value_set=vs)).astype(np.int8),
             mid=_np(T.column('mid')).astype(np.int64), role=_np(role).astype(np.int8), s=_np(T.column('s')).astype(np.int64),
             q=_np(T.column('q')).astype(np.float64), p=_np(T.column('p')).astype(np.float64), settle=_np(T.column('settle_ms')).astype(np.int64),
             self=_np(T.column('self_trade')).astype(bool), stc=_np(pc.index_in(st, value_set=pa.array(names))).astype(np.int64),
             sid=_np(pc.index_in(sid, value_set=pc.unique(sid))).astype(np.int64))
    del T
    F['ok'] = F['stc'] == (names.index(OK) if OK in names else -1)
    need(bool(np.isin(F['s'][F['ok']], (-1, 1)).all()), 'fills_valued: s outside {-1, +1} on an ok row')
    return F, names


def load_markets(path):
    T = pq.read_table(path, columns=list(MKT_SCHEMA)).to_pandas()
    need(T['mid'].is_unique, f'markets: market id not unique ({Path(path).name})')
    return {int(m): (int(round(float(a) * 1000)), int(round(float(b) * 1000)), str(s)) for m, a, b, s in
            zip(T['mid'], T['ws'], T['we'], T['status'])}


def load_btc(btc_dir, d, cnt):
    """Dense 100 ms BTC arrays covering [day - 1 h, day + 24 h) within the allowed hours; the previous day's file supplies the padding.
    pf_mt_btc (40ee591ea): row g = the last bookTicker with E_ms <= g; age_ms = g - E_ms (-1 when none); status ok | stale | gap | no_data.
    Usable = status ok, 0 <= age_ms <= BTC_MAX_AGE_MS and a finite mid > 0; age_ms -1 or a non-ok status is missing (counted), never fresh.
    A g_ms in two files: the usable row wins, else the row of the file whose own day holds g; two usable rows must agree (else refuse)."""
    d0 = day_ms(d)
    lo, hi = max(d0 - 3_600_000, PAD_LO_MS), min(d0 + 86_400_000, BTC_END_MS)
    prev = prev_day(d)
    parts = []
    for dd in ((prev, d) if prev in BTC_DAYS else (d,)):
        p = btc_path(btc_dir, dd)
        if not p.is_file():
            need(dd != d, f'missing BTC grid: btc/{p.name}')
            cnt[f'btc_padding_file_missing:{dd}'] += 1
            continue
        T = pq.read_table(p, columns=list(BTC_SCHEMA))
        g = _np(T.column('g_ms')).astype(np.int64)
        need(not bool((g >= BTC_END_MS).any()), f'BTC grid row at or after 2026-10-28 00:00Z (a forbidden day) in btc/{p.name} (refused)')
        cnt['btc_rows_before_padding'] += int((g < PAD_LO_MS).sum())
        k = (g >= lo) & (g < hi)
        d0f = day_ms(dd)
        parts.append((g[k], _np(T.column('mid')).astype(np.float64)[k], _np(T.column('age_ms')).astype(np.float64)[k],
                      np.asarray(_np(_s(T.column('status'))), dtype=object)[k], ((g >= d0f) & (g < d0f + 86_400_000))[k]))
        cnt[f'btc_rows_read:{dd}'] += int(len(g))
        del T
    g = np.concatenate([x[0] for x in parts])
    need(len(g) > 0, f'BTC grid: no row inside [{lo}, {hi}) for day {d}')
    mid, age, st, own = (np.concatenate([x[i] for x in parts]) for i in (1, 2, 3, 4))
    need(bool((g % STEP == 0).all()), 'BTC grid: g_ms off the 100 ms grid')
    need(not bool(((st == OK) & (age < -1)).any()),
         'BTC grid age_ms < 0 (below the -1 none-marker) on an ok row: a grid row carries a later update (look-ahead; refused)')
    stale = (st == OK) & (age > BTC_MAX_AGE_MS)
    good = (st == OK) & (age >= 0) & (age <= BTC_MAX_AGE_MS) & np.isfinite(mid) & (mid > 0)   # age -1 / non-ok status: missing, never fresh
    o = np.lexsort((~own, ~good, g))          # per g_ms: a usable row first, then the file whose own day holds g
    g, mid, age, st, own, good, stale = (x[o] for x in (g, mid, age, st, own, good, stale))
    first = np.r_[True, g[1:] != g[:-1]]
    if not first.all():                       # overlap rule (pf_mt_btc files hold their own day only; kept for any later layout)
        kf = np.maximum.accumulate(np.where(first, np.arange(len(g)), 0))
        need(not bool((~first & good & ((mid != mid[kf]) | (age != age[kf]))).any()),
             'BTC grid rows disagree across files (two usable rows of one g_ms; refused)')
        had = np.zeros(len(g), bool)
        had[kf[~first]] = True
        cnt['btc_overlap_rows'] += int((~first).sum())
        cnt['btc_overlap_filled_from_other_file'] += int((had & good & ~own).sum())
        g, mid, age, st, good, stale = (x[first] for x in (g, mid, age, st, good, stale))
    for s_, c_ in Counter(st.tolist()).items():
        cnt[f'btc_status:{s_}'] += c_
    cnt['btc_rows_stale'] += int(stale.sum())
    cnt['btc_rows_age_missing'] += int((age == -1).sum())
    G0 = int(g[0] // STEP)
    n = int(g[-1] // STEP) - G0 + 1
    idx = g // STEP - G0
    ok = np.zeros(n, bool)
    ok[idx] = good
    cnt['btc_grid_points_missing'] += int(n - len(g))
    lm = np.full(n, np.nan)
    lm[idx[good]] = np.log(mid[good])
    ret = np.zeros(n)
    ret[1:] = 1e4 * (lm[1:] - lm[:-1])
    ret[~np.isfinite(ret)] = 0.
    csq = np.zeros(n + 1)
    csq[1:] = np.cumsum(ret * ret)
    okc = np.zeros(n + 1, np.int64)
    okc[1:] = np.cumsum(ok)
    return dict(G0=G0, n=n, ok=ok, lm=lm, csq=csq, okc=okc)


def load_tob(path, um, lo, hi, cnt):
    """Frames of the markets `um` (sorted) with upd_ms in [lo, hi] per market -> arrays sorted by (market code, upd_ms, recv_ns) with each
    frame's own status (0 ok / 2 frame_status / 3 one_sided / 4 crossed: pf_mt_econ.frames' rule)."""
    parts, n_scan = [], 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=1_000_000, columns=list(TOB_COLS)):
        n_scan += b.num_rows
        if len(um) == 0:
            continue
        mid = _np(b.column('mid')).astype(np.int64)
        c = np.minimum(np.searchsorted(um, mid), len(um) - 1)
        t = _np(b.column('upd_ms')).astype(np.int64)
        k = np.flatnonzero((um[c] == mid) & (t >= lo[c]) & (t <= hi[c]))
        if not len(k):
            continue
        bb, ba = _np(b.column('best_bid')).astype(np.float64)[k], _np(b.column('best_ask')).astype(np.float64)[k]
        st = np.zeros(len(k), np.int8)
        st[E.flagged_any(np.asarray(_np(_s(b.column('status'))), dtype=object)[k], E.BAD_FRAME)] = 2
        two = (_np(b.column('n_bid'))[k] > 0) & (_np(b.column('n_ask'))[k] > 0) & np.isfinite(bb) & np.isfinite(ba)
        st[(st == 0) & ~two] = 3
        st[(st == 0) & (bb >= ba)] = 4
        parts.append((c[k].astype(np.int64), t[k], _np(b.column('recv_ns')).astype(np.int64)[k], bb, ba,
                      _np(b.column('best_bid_sh')).astype(np.float32)[k], _np(b.column('best_ask_sh')).astype(np.float32)[k], st))
    cnt['tob_frames_scanned'] += n_scan
    if not parts:
        return None
    code, t, recv, bb, ba, bs, as_, st = (np.concatenate([x[i] for x in parts]) for i in range(8))
    del parts
    need(t.min() >= 0 and t.max() < 1 << 42, 'tob: upd_ms outside the epoch-ms range')
    o = np.lexsort((recv, t, code))
    del recv
    FR = dict(code=code[o], t=t[o], bid=bb[o], ask=ba[o], bsz=bs[o], asz=as_[o], st=st[o])
    FR['key'] = (FR['code'] << 42) | FR['t']
    cnt['tob_frames_kept'] += int(len(o))
    for k_, v_ in zip(*np.unique(FR['st'], return_counts=True)):
        cnt[f'tob_frame_status:{BOOK_ST[int(k_)]}'] += int(v_)
    return FR


# ------------------------------------------------------------------ features ------------------------------------------------------------------
def btc_feats(BT, x, ws, we):
    """Per query (venue ms x; the market's ws / we in ms) -> dict(code (0 ok / 1 btc_missing / 2 btc_invalid / 3 after_we), r_ws, r_1s, r_5s,
    r_30s (bps), sig (bps / sqrt s, unfloored), tau (s), z, pf, mid). Values are NaN unless code is 0."""
    n_q = len(x)
    tau = (we - x) / 1000.
    out = dict(tau=tau)
    if BT is None or BT['n'] <= SIG_STEPS + 1:
        out['code'] = np.where(tau <= 0, 3, 1).astype(np.int8)
        for k in ('r_ws', 'r_1s', 'r_5s', 'r_30s', 'sig', 'z', 'pf', 'mid'):
            out[k] = np.full(n_q, np.nan)
        return out
    n, G0 = BT['n'], BT['G0']
    i = x // STEP - G0
    pre = x < ws
    iw = ws // STEP - G0
    inr = (i >= SIG_STEPS) & (i < n) & (pre | ((iw >= 0) & (iw < n)))
    ic = np.clip(i, SIG_STEPS, n - 1)
    iwc = np.clip(iw, 0, n - 1)
    okc = BT['okc']
    win = (okc[ic + 1] - okc[ic - SIG_STEPS]) == SIG_STEPS + 1
    wsok = pre | BT['ok'][iwc]
    code = np.where(~inr, 1, np.where(~(win & wsok), 2, np.where(tau <= 0, 3, 0))).astype(np.int8)
    lm = BT['lm']
    lmi = lm[ic]
    for nm, k in R_STEPS:
        out[nm] = 1e4 * (lmi - lm[ic - k])
    out['r_ws'] = np.where(pre, 0., 1e4 * (lmi - lm[iwc]))
    csq = BT['csq']
    out['sig'] = np.sqrt(np.maximum(csq[ic + 1] - csq[ic + 1 - SIG_STEPS], 0.) / 60.)
    z = out['r_ws'] / (np.maximum(out['sig'], SIG_FLOOR) * np.sqrt(np.where(tau > 0, tau, 1.)))
    out['z'], out['pf'], out['mid'] = z, ndtr(z), np.exp(lmi)
    bad = code != 0
    for k in ('r_ws', 'r_1s', 'r_5s', 'r_30s', 'sig', 'z', 'pf', 'mid'):
        out[k] = np.where(bad, np.nan, out[k])
    out['code'] = code
    return out


def book_feats(FR, code, x):
    """Per query (market code in FR, -1 = no frames; venue ms x) -> the last frame with upd_ms <= x: dict(code (BOOK_ST index), bid, ask,
    bsz, asz, age); prices / sizes NaN unless ok."""
    n_q = len(x)
    out = dict(code=np.ones(n_q, np.int8), age=np.full(n_q, -1, np.int64))
    for k in ('bid', 'ask', 'bsz', 'asz'):
        out[k] = np.full(n_q, np.nan)
    if FR is None or len(FR['key']) == 0 or n_q == 0:
        return out
    j = np.searchsorted(FR['key'], (np.maximum(code, 0).astype(np.int64) << 42) | x, side='right') - 1
    jc = np.maximum(j, 0)
    f = (code >= 0) & (j >= 0) & (FR['code'][jc] == code)
    c = np.where(f, FR['st'][jc], 1).astype(np.int8)
    age = np.where(f, x - FR['t'][jc], -1)
    c = np.where((c == 0) & (age > STALE_MS), 5, c).astype(np.int8)
    ok = c == 0
    out['code'], out['age'] = c, age
    for k in ('bid', 'ask', 'bsz', 'asz'):
        out[k] = np.where(ok, FR[k][jc], np.nan)
    return out


def own_state(OW, gid, x):
    """Position (sum s q of the pair's own fills with settle_ms < x) and the time since the last of them (NaN if none)."""
    if OW is None or len(OW['key']) == 0:
        return np.zeros(len(x)), np.full(len(x), np.nan)
    j = np.searchsorted(OW['key'], (gid << 42) | x, side='left') - 1
    jc = np.maximum(j, 0)
    f = (j >= 0) & (OW['gid'][jc] == gid)
    return np.where(f, OW['cpos'][jc], 0.), np.where(f, (x - OW['settle'][jc]).astype(np.float64), np.nan)


def rest_state(RS, gid, x, isbook):
    """BOOK pairs: 0 none / 1 bid / 2 ask / 3 both of its proven-present orders at x; 4 = na (not a BOOK market)."""
    out = np.full(len(x), 4, np.int64)
    ib = np.flatnonzero(isbook)
    if not len(ib):
        return out
    g_, x_ = gid[ib], x[ib]
    cnt = []
    for side in (0, 1):
        if RS is None or len(RS['klo'][side]) == 0:
            cnt.append(np.zeros(len(ib), np.int64))
            continue
        base = g_ << 42
        q = base | x_
        klo, khi = RS['klo'][side], RS['khi'][side]
        n_lo = np.searchsorted(klo, q, 'right') - np.searchsorted(klo, base, 'left')
        n_hi = np.searchsorted(khi, q, 'left') - np.searchsorted(khi, base, 'left')
        cnt.append(n_lo - n_hi)
    out[ib] = (cnt[0] > 0).astype(np.int64) + 2 * (cnt[1] > 0).astype(np.int64)
    return out


def evaluate(ctx, x, ws, we, code, gid=None, isbook=None):
    """All decision features at x: BTC, book, gaps, own state; a row status (ROW_ST index)."""
    B = btc_feats(ctx['BT'], x, ws, we)
    K = book_feats(ctx['FR'], code, x)
    rs = np.where(B['code'] > 0, B['code'], np.where(K['code'] > 0, 3 + K['code'], 0)).astype(np.int64)
    out = dict(rs=rs, bcode=B['code'], kcode=K['code'], **{k: B[k] for k in ('r_ws', 'r_1s', 'r_5s', 'r_30s', 'sig', 'tau', 'z', 'pf', 'mid')},
               bid=K['bid'], ask=K['ask'], bsz=K['bsz'], asz=K['asz'], book_age=K['age'])
    out['g_buy'] = 100. * (out['pf'] - out['ask'])
    out['g_sell'] = 100. * (out['bid'] - out['pf'])
    if gid is not None:
        out['pos'], out['tsince'] = own_state(ctx['OW'], gid, x)
        out['rest'] = rest_state(ctx['RS'], gid, x, isbook) if isbook is not None else np.full(len(x), 4, np.int64)
    return out


# ------------------------------------------------------------------ pairs ------------------------------------------------------------------
def build_pairs(F, names, mk, book, days, cnt):
    """One row per (day, address, market) with own rows: span, status, counts (gid = row index)."""
    df = pd.DataFrame(dict(d=F['d'], a=F['a'], mid=F['mid'], bad=(~F['ok']).astype(np.int64), nos=(F['settle'] < 0).astype(np.int64),
                           slo=np.where(F['settle'] >= 0, F['settle'], np.iinfo(np.int64).max),
                           shi=np.where(F['settle'] >= 0, F['settle'], -1), bst=np.where(F['ok'], 1 << 30, F['stc']),
                           tk=(F['role'] == 1).astype(np.int64), mkr=(F['role'] == 0).astype(np.int64)))
    P = df.groupby(['d', 'a', 'mid'], sort=True).agg(n_rows=('bad', 'size'), n_bad=('bad', 'sum'), n_nos=('nos', 'sum'), f_lo=('slo', 'min'),
                                                     f_hi=('shi', 'max'), worst=('bst', 'min'), n_tk_rows=('tk', 'sum'),
                                                     n_mk_rows=('mkr', 'sum')).reset_index()
    tk = pd.DataFrame(dict(d=F['d'], a=F['a'], mid=F['mid'], sid=F['sid']))[F['role'] == 1]
    ntk = tk.groupby(['d', 'a', 'mid'])['sid'].nunique().rename('n_tk_rec').reset_index()
    P = P.merge(ntk, on=['d', 'a', 'mid'], how='left')
    P['n_tk_rec'] = P['n_tk_rec'].fillna(0).astype(np.int64)
    st, ws, we, t0, nn, isb, pin = [], [], [], [], [], [], []
    for d_, m_, nb, nn_, flo, fhi, w_ in zip(P['d'], P['mid'], P['n_bad'], P['n_nos'], P['f_lo'], P['f_hi'], P['worst']):
        day = days[int(d_)]
        mrow = mk[day].get(int(m_))
        isb.append(int(m_) in set(book.get(day, ())))
        s = None
        if mrow is None:
            s = 'no_market'
            ws.append(-1), we.append(-1), t0.append(-1), nn.append(0), pin.append(False)
            st.append(s)
            continue
        ws.append(mrow[0]), we.append(mrow[1])
        fl = _flags(mrow[2])
        pin.append('poll_incomplete' in fl)                       # kept (accepted rule), counted
        bad_m = [f for f in MKT_EXCLUDE if f in fl]
        if bad_m:
            s = f'market_{bad_m[0]}'
        elif nb:
            s = f'fill_status_{names[int(w_)]}'
        elif nn_:
            s = 'no_settle'
        if fhi >= 0:
            lo, hi = max(int(flo) - PAD_MS, mrow[0] - PRE_MS), min(int(fhi) + PAD_MS, mrow[1])
            t0_ = lo // STEP * STEP
            n_ = (hi - t0_ + STEP - 1) // STEP if hi > t0_ else 0
        else:
            t0_, n_ = -1, 0
        t0.append(t0_), nn.append(int(n_))
        st.append(s if s is not None else (OK if n_ > 0 else 'span_empty'))
    P['ws'], P['we'], P['t0'], P['n'], P['book'], P['status'], P['poll_incomplete'] = ws, we, t0, nn, isb, st, pin
    P['gid'] = np.arange(len(P), dtype=np.int64)
    for (d_, s_), g_ in P.groupby(['d', 'status']):
        cnt[f'pairs:{days[int(d_)]}:{s_}'] += int(len(g_))
    return P


def select_timelines(P):
    """Deterministic example pairs per address (declared in the module docstring) -> {gid: scope}."""
    out = {}
    for a in range(3):
        V = P[(P['status'] == OK) & (P['a'] == a)]
        Bk = V[V['book']].sort_values(['n_rows', 'd', 'mid'], ascending=[False, True, True], kind='mergesort')
        if len(Bk):
            out[int(Bk['gid'].iloc[0])] = SCOPES[a]
        Nb = V[~V['book']]
        T1 = Nb[Nb['n_tk_rec'] > 0].sort_values(['n_tk_rec', 'd', 'mid'], ascending=[False, True, True], kind='mergesort')
        g1 = int(T1['gid'].iloc[0]) if len(T1) else -1
        if g1 >= 0:
            out[g1] = SCOPES[a]
        T2 = Nb[(Nb['n_mk_rows'] > 0) & (Nb['gid'] != g1)].sort_values(['n_mk_rows', 'd', 'mid'], ascending=[False, True, True], kind='mergesort')
        if len(T2):
            out[int(T2['gid'].iloc[0])] = SCOPES[a]
    return out


# ------------------------------------------------------------------ hazard accumulators ------------------------------------------------------------------
def acc_new(D):
    return {t: (np.zeros((3, D, NL, 2, B)), np.zeros((3, D, NL, 2, B))) for t, B, _, _ in TABLES}


def acc_add(ACC, d, l, a, bk, yb, ys):
    """bk: table -> bucket array (side-free) or (buy, sell) pair; yb / ys: events of the valid rows."""
    for t, B, sided, _ in TABLES:
        N, Y = ACC[t]
        b2 = bk[t] if sided else (bk[t], bk[t])
        nb = None
        for side, y in ((0, yb), (1, ys)):
            idx = a * B + b2[side]
            if sided or nb is None:
                nb = np.bincount(idx, minlength=3 * B).reshape(3, B)
            N[:, d, l, side, :] += nb
            Y[:, d, l, side, :] += np.bincount(idx, weights=y, minlength=3 * B).reshape(3, B)


def acc_get(ACC, t, sc, dd):
    """-> (N, Y) arrays (NL, 2, B) for scope sc (3 = pooled) and day index dd (None = all days)."""
    N, Y = ACC[t]
    if sc < 3:
        N, Y = N[sc], Y[sc]
    else:
        N, Y = N.sum(0), Y.sum(0)
    return (N.sum(0), Y.sum(0)) if dd is None else (N[dd], Y[dd])


def step_split(nb, eb):
    """Max-likelihood two-rate split of the g buckets: below = g < E[j], above = g >= E[j]; both sides >= N_MIN_SPLIT rows; ties to the
    smallest edge."""
    nb, eb = np.asarray(nb, float), np.asarray(eb, float)
    N, Ev = nb.sum(), eb.sum()
    r = dict(n=int(N), events=int(Ev), threshold_c=np.nan, n_above=0, e_above=0, rate_above=np.nan, n_below=0, e_below=0, rate_below=np.nan,
             ratio=np.nan, ll_gain=np.nan)
    if N == 0:
        return dict(r, status='no_rows')
    if Ev == 0:
        return dict(r, status='no_events')
    cn, ce = np.cumsum(nb)[:-1], np.cumsum(eb)[:-1]
    na, ea = N - cn, Ev - ce
    okc = (cn >= N_MIN_SPLIT) & (na >= N_MIN_SPLIT)
    if not okc.any():
        return dict(r, status='insufficient_rows')
    with np.errstate(divide='ignore', invalid='ignore'):
        pb, pa_ = np.where(cn > 0, ce / np.maximum(cn, 1), 0.), np.where(na > 0, ea / np.maximum(na, 1), 0.)
        LL = xlogy(ce, pb) + xlogy(cn - ce, 1 - pb) + xlogy(ea, pa_) + xlogy(na - ea, 1 - pa_)
    p0 = Ev / N
    LL0 = xlogy(Ev, p0) + xlogy(N - Ev, 1 - p0)
    j = int(np.argmax(np.where(okc, LL, -np.inf)))
    rb, ra = ce[j] / cn[j], ea[j] / na[j]
    return dict(r, threshold_c=float(G_EDGES[j]), n_above=int(na[j]), e_above=int(ea[j]), rate_above=float(ra), n_below=int(cn[j]),
                e_below=int(ce[j]), rate_below=float(rb), ratio=float(ra / rb) if rb > 0 else np.nan, ll_gain=float(LL[j] - LL0),
                status=OK if rb > 0 else 'below_zero_events')


def slice_counts(ACC, sc, dd, l, side, sl):
    """g-bucket (n, events) of a threshold slice."""
    if sl == 'all':
        N, Y = acc_get(ACC, 'g', sc, dd)
        return N[l, side], Y[l, side]
    t = 'g_x_r1sign' if sl.startswith('r1') else 'g_x_r5sign'
    N, Y = acc_get(ACC, t, sc, dd)
    N, Y = N[l, side].reshape(NG, 3), Y[l, side].reshape(NG, 3)
    al = 2 if side == 0 else 0
    col = dict(aligned=al, against=2 - al, zero=1)[sl.split('_')[1]]
    return N[:, col], Y[:, col]


# ------------------------------------------------------------------ logistic (IRLS, streamed from disk) ------------------------------------------------------------------
def _fit_chunks(path):
    with open(path, 'rb') as f:
        while True:
            a = np.fromfile(f, dtype=FIT_DT, count=FIT_CH)
            if len(a) == 0:
                break
            yield a


def _design(ch):
    X = np.empty((len(ch), 7))
    X[:, 0] = 1.
    X[:, 2], X[:, 3], X[:, 4] = ch['r1'], ch['r5'], ch['tau'] / 60.
    X[:, 5], X[:, 6] = ch['ps'] == 2, ch['ps'] == 0
    return X


def _upd(A, X, y, beta, first):
    eta = X @ beta
    p = expit(eta)
    w = p * (1. - p)
    A['H'] += (X * w[:, None]).T @ X
    A['g'] += X.T @ (y - p)
    A['dev'] += 2. * (np.logaddexp(0., eta).sum() - y @ eta)        # -2 sum[y log p + (1 - y) log(1 - p)]
    A['n'] += len(y)
    if first:
        A['sx'] += X.sum(0)
        A['sxx'] += np.einsum('ij,ij->j', X, X)


def _newton(m, A):
    m['passes'] += 1
    dev = float(A['dev'])
    if not np.isfinite(dev):
        m['status'] = 'numerical'
        return
    if m['act'] is None:                     # first pass: drop constant columns (named), re-evaluate if a start value moved
        var = A['sxx'] / A['n'] - (A['sx'] / A['n']) ** 2
        act = var > 1e-12
        act[0] = True
        m['act'], m['dropped'] = act, [LOGIT_COLS[i] for i in range(7) if not act[i]]
        if (m['beta'][~act] != 0).any():
            m['beta'][~act] = 0.
            return
    if m['prev_dev'] is not None and dev > m['prev_dev'] + 1e-9 * (1. + abs(m['prev_dev'])) and m['halv'] < 10:
        m['step'] = m['step'] / 2.
        m['beta'] = m['prev_beta'] + m['step']
        m['halv'] += 1
        return
    m['halv'], m['dev'] = 0, dev
    if m['prev_dev'] is not None and abs(m['prev_dev'] - dev) <= LOGIT_TOL * (1. + abs(dev)):
        m['status'] = 'converged'
        return
    if m['it'] >= LOGIT_MAX_IT:
        m['status'] = 'max_iter'
        return
    act = m['act']
    Hs, gs = A['H'][np.ix_(act, act)], A['g'][act]
    try:
        s = np.linalg.solve(Hs, gs)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(Hs, gs, rcond=None)[0]
        m['singular'] = True
    step = np.zeros(7)
    step[act] = s
    m['prev_beta'], m['prev_dev'], m['step'] = m['beta'].copy(), dev, step
    m['beta'] = m['beta'] + step
    m['it'] += 1


def fit_logits(files, base, log):
    """files: {(l, a): path of FIT_DT rows}; base: {(l, sc, side): (n, events)} -> {(l, sc, side): model dict}."""
    res, warm = {}, {}
    for l in range(NL):
        M = {}
        for sc in range(4):
            for side in (0, 1):
                n, ev = base[(l, sc, side)]
                m = dict(n=int(n), ev=int(ev), beta=np.zeros(7), act=None, dropped=[], it=0, halv=0, passes=0, prev_beta=None, prev_dev=None,
                         step=None, singular=False, dev=np.nan, status='active')
                if n == 0:
                    m['status'] = 'no_rows'
                elif ev == 0:
                    m['status'] = 'no_events'
                elif ev == n:
                    m['status'] = 'all_events'
                elif (sc, side) in warm:
                    m['beta'] = warm[(sc, side)].copy()
                else:
                    m['beta'][0] = np.log(ev / (n - ev))
                M[(sc, side)] = m
        c0 = time.process_time()
        for _ in range(LOGIT_MAX_PASS):
            act_m = [k for k, m in M.items() if m['status'] == 'active']
            if not act_m:
                break
            if time.process_time() - c0 > FIT_CPU_CAP_S:
                for k in act_m:
                    M[k]['status'] = 'cpu_cap'
                break
            A = {k: dict(H=np.zeros((7, 7)), g=np.zeros(7), dev=0., sx=np.zeros(7), sxx=np.zeros(7), n=0) for k in act_m}
            for a in range(3):
                if not any(k in A for k in ((a, 0), (a, 1), (3, 0), (3, 1))):
                    continue
                for ch in _fit_chunks(files[(l, a)]):
                    X = _design(ch)
                    for side in (0, 1):
                        ks = [k for k in ((a, side), (3, side)) if k in A]
                        if not ks:
                            continue
                        X[:, 1] = np.clip(ch['gb' if side == 0 else 'gs'], -GCLIP_LOGIT, GCLIP_LOGIT)
                        y = ch['yb' if side == 0 else 'ys'].astype(np.float64)
                        for k in ks:
                            _upd(A[k], X, y, M[k]['beta'], M[k]['act'] is None)
            for k in act_m:
                _newton(M[k], A[k])
        for k, m in M.items():
            if m['status'] == 'active':
                m['status'] = 'pass_cap'
            res[(l,) + k] = m
            if m['status'] == 'converged':
                warm[k] = m['beta'].copy()
        log(f'logit L={LAGS[l]}: ' + ', '.join(f'{SCOPES[k[0]]}/{"buy sell".split()[k[1]]}={m["status"]}:{m["it"]}' for k, m in M.items()), flush=True)
    return res


# ------------------------------------------------------------------ small helpers for the summaries ------------------------------------------------------------------
def grouped(df, keys, aggs, days):
    """Group rows per (scope, day) combination: address and pooled, each day and all days (only for the keys present)."""
    out = []
    if df is None or len(df) == 0:                  # an empty subset keeps its columns, so every later column step is a no-op
        return pd.DataFrame(columns=['scope', 'day'] + [k for k in keys if k not in ('a', 'd')] + list(aggs))
    for pooled in ((False, True) if 'a' in keys else (True,)):
        for alld in ((False, True) if 'd' in keys else (True,)):
            k = [c for c in keys if not ((c == 'a' and pooled) or (c == 'd' and alld))]
            need(len(k) > 0, 'internal: grouped() needs a key besides address and day')
            G = df.groupby(k, sort=True).agg(**aggs).reset_index()
            G.insert(0, 'scope', 'pooled' if pooled else [SCOPES[int(i)] for i in G['a']])
            G.insert(1, 'day', 'all' if alld else [days[int(i)] for i in G['d']])
            out.append(G.drop(columns=[c for c in ('a', 'd') if c in G.columns]))
    return pd.concat(out, ignore_index=True)


def ols_from_sums(G):
    """OLS y = a + b x from grouped sums (n, sx, sy, sxx, sxy, syy)."""
    n = G['n'].to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mx, my = G['sx'] / n, G['sy'] / n
        vxx, vxy, vyy = G['sxx'] / n - mx * mx, G['sxy'] / n - mx * my, G['syy'] / n - my * my
        b = np.where((n >= OLS_MIN) & (vxx > 1e-12), vxy / vxx, np.nan)
        a = np.where(np.isfinite(b), my - b * mx, np.where(n >= OLS_MIN, my, np.nan))
        res = np.sqrt(np.maximum(vyy - np.where(np.isfinite(b), b * vxy, 0.), 0.))
        corr = np.where((vxx > 1e-12) & (vyy > 1e-12), vxy / np.sqrt(vxx * vyy), np.nan)
    G = G.copy()
    G['a'], G['b'], G['resid_sd'], G['corr'] = a, b, res, corr
    return G


def _book_flags(df):
    """int8 indicators of the position against the shown best (0 when the book frame is unusable) and bk (usable)."""
    bk = df['kc'].to_numpy() == 0
    b = df['behind'].to_numpy()
    df['bk'] = bk.astype(np.int8)
    with np.errstate(invalid='ignore'):
        for nm_, cond in (('at', b == 0), ('ins', b < 0), ('b1', b == 1), ('b2', b == 2), ('b3', b >= 3)):
            df[nm_] = (bk & cond).astype(np.int8)


def _book_shares(G):
    """Shares at best / inside / 1 / 2 / >= 3 ticks behind = counts over the usable-book rows (NaN without one)."""
    nb = G['n_book'].to_numpy(float)
    for c_, o_ in (('n_at', 'at_best'), ('n_ins', 'inside'), ('n_b1', 'behind1'), ('n_b2', 'behind2'), ('n_b3', 'behind3p')):
        G[o_] = np.where(nb > 0, G[c_].to_numpy(float) / np.where(nb > 0, nb, 1.), np.nan)
    G.drop(columns=['n_at', 'n_ins', 'n_b1', 'n_b2', 'n_b3'], inplace=True)


def skew_sums(a, l, side, x, y, clock, scen):
    """OLS sums of y on x per (scope, L, side) by bincount (address scopes and pooled) -> rows for ols_from_sums."""
    idx = (a.astype(np.int64) * NL + l.astype(np.int64)) * 2 + side.astype(np.int64)
    K = 3 * NL * 2
    S = dict(n=np.bincount(idx, minlength=K).astype(float), sx=np.bincount(idx, weights=x, minlength=K), sy=np.bincount(idx, weights=y, minlength=K),
             sxx=np.bincount(idx, weights=x * x, minlength=K), sxy=np.bincount(idx, weights=x * y, minlength=K),
             syy=np.bincount(idx, weights=y * y, minlength=K))
    rows = []
    for sc in range(4):
        for l_ in range(NL):
            for s_ in (0, 1):
                ks = [(a_ * NL + l_) * 2 + s_ for a_ in (range(3) if sc == 3 else (sc,))]
                r = {k: float(v[ks].sum()) for k, v in S.items()}
                if r['n'] > 0:
                    rows.append(dict(scope=SCOPES[sc], day='all', L=LAGS[l_], clock=clock, scenario=scen, side=('bid', 'ask')[s_], **r))
    return pd.DataFrame(rows, columns=['scope', 'day', 'L', 'clock', 'scenario', 'side', 'n', 'sx', 'sy', 'sxx', 'sxy', 'syy'])


def _sums(df, x, y):
    df = df.copy()
    df['sx'], df['sy'], df['sxx'], df['sxy'], df['syy'] = df[x], df[y], df[x] ** 2, df[x] * df[y], df[y] ** 2
    return df


def _cat_arrays(rows):
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]} if rows else {}


# ------------------------------------------------------------------ the run ------------------------------------------------------------------
def run(inp=None, out=None, days=C.MEASURE_DAYS, addrs=TOP3, book=None, btc_dir=None, n_draws=N_DRAWS, guard_mib=C.GUARD_MIB,
        chunk_rows=CHUNK_ROWS, log=print):
    cpu0, wall0, asof0 = time.process_time(), time.time(), _now()
    days = tuple(days)
    for d in days:
        need(d in C.MEASURE_DAYS, f'day {d!r} is not a measurement day (refused before any path is built)')
    need(len(days) > 0 and len(set(days)) == len(days) and list(days) == sorted(days), 'days must be distinct and sorted')
    addrs = tuple(addrs)
    for a in addrs:
        need(isinstance(a, str) and a.lower() in LOW3, f"address {a!r} is not one of the contract's ex-ante top 3 (refused)")
    need(tuple(a.lower() for a in addrs) == LOW3, "addresses must be the contract's top 3 in rank order")
    inp = Path(inp) if inp is not None else C.OUT
    out = Path(out) if out is not None else C.RECEIPT / 'decision'
    btc_dir = Path(btc_dir) if btc_dir is not None else inp / 'btc'
    need(not out.exists(), f'write-once: {out} exists (refused)')
    book_src = 'argument'
    if book is None:
        book, book_src = E.load_book(), str(E.BOOK_MANIFEST.relative_to(C.REPO))
    book = {d: sorted(int(m) for m in book.get(d, [])) for d in days}
    paths = preflight(inp, btc_dir, days, book)
    ident = json.loads(paths['identities'].read_text())
    top = sorted((x for x in ident['ex_ante']['top10_shares'] if int(x['rank']) in (1, 2, 3)), key=lambda x: int(x['rank']))
    need([str(x['addr']).lower() for x in top] == list(LOW3), "identities.json ex-ante ranks 1-3 differ from the contract's top 3 (refused)")
    D = len(days)
    cnt, mem, cpu = Counter(), {}, {}
    STROWS = []        # status rows: (stage, scope, day, L, status, n)

    def stage(name):
        cpu[name] = round(time.process_time() - cpu0, 2)
        mem[name] = round(E.guard(guard_mib, name), 1)
        log(f'[{_now()}] {name}: cpu {cpu[name]} s, VmHWM {mem[name]} MiB', flush=True)

    tmp = out.parent / f'.{out.name}.partial_{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}_{os.getpid()}'
    fitdir = inp / f'.decision_fit_{os.getpid()}'
    need(not tmp.exists() and not fitdir.exists(), 'a partial directory exists')
    try:
        tmp.mkdir(parents=True)
        fitdir.mkdir()
        F, names = load_fv(paths['fv'], days, cnt)
        stage('fills_valued')
        mk = {d: load_markets(paths[('markets', d)]) for d in days}
        P = build_pairs(F, names, mk, book, days, cnt)
        TL = select_timelines(P)
        gid_row = np.full(len(F['d']), -1, np.int64)        # each own row -> its pair gid (all pairs)
        key = pd.MultiIndex.from_arrays([P['d'].to_numpy(np.int64), P['a'].to_numpy(np.int64), P['mid'].to_numpy(np.int64)])
        gid_row[:] = key.get_indexer(pd.MultiIndex.from_arrays([F['d'].astype(np.int64), F['a'].astype(np.int64), F['mid'].astype(np.int64)]))
        need(bool((gid_row >= 0).all()), 'internal: a fills_valued row without a pair')
        pst = P['status'].to_numpy(object)
        valid_row = pst[gid_row] == OK
        for (d_, a_, s_), g_ in P.groupby(['d', 'a', 'status']):
            STROWS.append(('pairs', SCOPES[a_], days[d_], '-', s_, int(len(g_))))
            STROWS.append(('pair_fv_rows', SCOPES[a_], days[d_], '-', s_, int(g_['n_rows'].sum())))
            STROWS.append(('pair_grid_rows', SCOPES[a_], days[d_], '-', s_, int(g_['n'].sum())))
        for (d_, a_, s_), g_ in P[P['poll_incomplete'].to_numpy(bool)].groupby(['d', 'a', 'status']):
            STROWS.append(('pairs_poll_incomplete_kept', SCOPES[a_], days[d_], '-', s_, int(len(g_))))
        cnt['pairs_poll_incomplete_kept'] = int((P['poll_incomplete'].to_numpy(bool) & (pst == OK)).sum())
        stage('pairs')
        ACC = acc_new(D)
        files = {(l, a): fitdir / f'fit_L{LAGS[l]}_a{a}.bin' for l in range(NL) for a in range(3)}
        fh = {k: open(p, 'ab') for k, p in files.items()}
        FILL, TAKE, POST, LINK, TLR = [], [], [], [], {}
        REM = {}                         # (l, scope, scenario, group, metric, phase) -> sums
        rows_st = np.zeros((3, D, NL, NRS), np.int64)
        ev_st = np.zeros((3, D, NL, 2, NRS), np.int64)
        take_span = Counter()
        try:
            for di, d in enumerate(days):
                ctx = day_pass(di, d, F, P, gid_row, valid_row, mk[d], book[d], paths, btc_dir, cnt, chunk_rows, ACC, fh, rows_st, ev_st,
                               take_span, TL, TLR, FILL, TAKE, POST, LINK, REM, n_draws, STROWS, guard_mib)
                del ctx
                E.trim()
                stage(f'day {d}')
        finally:
            for f_ in fh.values():
                f_.close()
        # ---- logistic fits ----
        base = {}
        for l in range(NL):
            for sc in range(4):
                for side in (0, 1):
                    N, Y = acc_get(ACC, 'pos', sc, None)
                    base[(l, sc, side)] = (int(N[l, side].sum()), int(Y[l, side].sum()))
        FITS = fit_logits(files, base, log)
        stage('logit')
        shutil.rmtree(fitdir, ignore_errors=True)
        # ---- tables ----
        tabs = write_tables(tmp, days, P, F, gid_row, valid_row, ACC, FITS, FILL, TAKE, POST, LINK, REM, rows_st, ev_st, take_span, TLR, STROWS,
                            cnt, n_draws, addrs)
        stage('tables')
        rec = dict(version=VERSION, module='pf_mt_decision', contract=CONTRACT, run=C.RUN, asof_start=asof0, asof_end=_now(),
                   wall_s=round(time.time() - wall0, 1), days=list(days), lags_ms=list(LAGS), addresses={SCOPES[i]: TOP3[i] for i in range(3)},
                   book=book, book_source=book_src,
                   inputs={str(k if isinstance(k, str) else '/'.join(map(str, k))): dict(path=str(p), sha256=E.sha256(p), bytes=p.stat().st_size)
                           for k, p in paths.items()},
                   btc_files={d_: dict(path=str(btc_path(btc_dir, d_)), sha256=E.sha256(btc_path(btc_dir, d_)))
                              for d_ in sorted(set(days) | {prev_day(x) for x in days} & set(BTC_DAYS)) if btc_path(btc_dir, d_).is_file()},
                   sources={n_: E.sha256(HERE / n_) for n_ in ('pf_mt_decision.py', 'pf_mt_common.py', 'pf_mt_econ.py', 'pf_mt_decision_selftest.py')
                            if (HERE / n_).is_file()},
                   declared=dict(STEP=STEP, PAD_MS=PAD_MS, PRE_MS=PRE_MS, R_STEPS=dict(R_STEPS), SIG_STEPS=SIG_STEPS, SIG_FLOOR=SIG_FLOOR,
                                 BTC_MAX_AGE_MS=BTC_MAX_AGE_MS, STALE_MS=STALE_MS, TOB_KEEP_BEFORE_MS=TOB_KEEP_BEFORE_MS, MKT_EXCLUDE=MKT_EXCLUDE,
                                 G_EDGES=G_EDGES.tolist(), M1=[x.tolist() for x in M1], M5=[x.tolist() for x in M5], S_EDGES=S_EDGES.tolist(),
                                 POS_B=POS_B.tolist(), N_MIN_SPLIT=N_MIN_SPLIT, GCLIP_LOGIT=GCLIP_LOGIT, LOGIT_COLS=LOGIT_COLS,
                                 LOGIT_MAX_IT=LOGIT_MAX_IT, LOGIT_MAX_PASS=LOGIT_MAX_PASS, LOGIT_TOL=LOGIT_TOL, FIT_CPU_CAP_S=FIT_CPU_CAP_S, N_DRAWS=n_draws, SEED=SEED, CENTRE_TOL_C=CENTRE_TOL_C,
                                 TEN_X=TEN_X, OLS_MIN=OLS_MIN, PRIMARY=PRIMARY, BTC_DAYS=BTC_DAYS, PAD_LO_MS=PAD_LO_MS, BTC_END_MS=BTC_END_MS,
                                 ORDER_INVALID=sorted(ORDER_INVALID), MIN_PLACE=MIN_PLACE, REM_METRICS=REM_METRICS),
                   constants_accepted=dict(by='integrator (coordinator) message, 2026-11-23: accepted as stated',
                                           sig_60_floor_bps_per_sqrt_s=SIG_FLOOR, btc_stale_after_ms=BTC_MAX_AGE_MS,
                                           threshold_split_min_rows_per_side=N_MIN_SPLIT, logistic_gap_clip_c=GCLIP_LOGIT,
                                           poll_incomplete='markets kept and counted (counts.pairs_poll_incomplete_kept, statuses '
                                                           'pairs_poll_incomplete_kept)'),
                   rules=dict(btc_semantics='pf_mt_btc 40ee591ea: row g = the last bookTicker with E_ms <= g (one file per day, its own grid only); '
                                            'age_ms = g - E_ms (-1 when none); status ok | stale | gap | no_data; usable = ok and 0 <= age_ms <= '
                                            f'{BTC_MAX_AGE_MS} and a finite mid > 0; age_ms -1 or a non-ok status = missing (counted), never fresh; '
                                            'an ok row with age_ms < -1 refuses; a g_ms in two files: the usable row wins, else the own-day file; two '
                                            'usable rows must agree', ref_files='ref_<day>.parquet (Chainlink) is never read: returns only',
                              removal_null=f'per order: {n_draws} windows of the same 100 ms-rounded width drawn uniformly (with replacement) from '
                                           'the usable placements of the SAME PHASE (pre-ws [ws - 300 s, ws) or live [ws, we), by the window start '
                                           f'at L = 0) in the same market, or in the same market-day pool when the market has < {MIN_PLACE}; '
                                           'windows straddling ws, ending at or after we, unbounded or without a placement are statuses',
                              doc=__doc__),
                   counts=dict(sorted(cnt.items())), rows=tabs['rows'], outputs=tabs['outputs'],
                   cpu_s=cpu, memory=dict(guard_mib=guard_mib, vmhwm_mib_by_stage=mem, vmhwm_mib=round(E.hwm(), 1)),
                   slot=os.environ.get('PF_SLOT'), argv=sys.argv)
        rec['cpu_s']['total'] = round(time.process_time() - cpu0, 2)
        (tmp / 'decision.json').write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + '\n')
        need(not out.exists(), f'write-once: {out} appeared during the run (refused)')
        os.rename(tmp, out)
        return rec
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(fitdir, ignore_errors=True)
        raise


def day_pass(di, d, F, P, gid_row, valid_row, mkd, bookd, paths, btc_dir, cnt, chunk_rows, ACC, fh, rows_st, ev_st, take_span, TL, TLR,
             FILL, TAKE, POST, LINK, REM, n_draws, STROWS, guard_mib=C.GUARD_MIB):
    """One measurement day: load its BTC / book, the grid pass (hazard + fit rows + timelines), fill / take / posting / link / removal rows."""
    PD = P[(P['d'] == di) & (P['status'] == OK)].sort_values('gid')
    rmask = (F['d'] == di) & valid_row
    if len(PD) == 0:
        cnt[f'day_no_valid_pair:{d}'] += 1
        return None
    um = np.array(sorted(set(int(m) for m in PD['mid'])), np.int64)
    wsm = np.array([mkd[int(m)][0] for m in um], np.int64)
    wem = np.array([mkd[int(m)][1] for m in um], np.int64)
    FR = load_tob(paths[('tob', d)], um, wsm - TOB_KEEP_BEFORE_MS, wem, cnt)
    E.guard(guard_mib, f'{d} tob')
    BT = load_btc(btc_dir, d, cnt)
    # ---- own fills of the day's valid pairs: position keys ----
    ix = np.flatnonzero(rmask)
    og = gid_row[ix]
    o = np.lexsort((F['settle'][ix], og))
    ix, og = ix[o], og[o]
    sq = F['s'][ix] * F['q'][ix]
    OW = dict(gid=og, settle=F['settle'][ix], key=(og << 42) | F['settle'][ix],
              cpos=pd.Series(sq).groupby(og).cumsum().to_numpy(np.float64))
    # ---- BOOK lifecycle rows of the top 3 ----
    pair_of = {(int(a), int(m)): int(g) for a, m, g in zip(PD['a'], PD['mid'], PD['gid'])}
    ORD, LNK = load_life(paths, d, bookd, cnt, STROWS)
    RS = None
    if len(ORD):
        O1 = ORD[(ORD['scenario'] == PRIMARY) & ORD['valid']]
        gid_o = np.array([pair_of.get((int(a), int(m)), -1) for a, m in zip(O1['a'], O1['mid'])], np.int64)
        ok_p = (gid_o >= 0) & (O1['persist_lo_ms'].to_numpy() >= 0) & (O1['persist_hi_ms'].to_numpy() >= O1['persist_lo_ms'].to_numpy())
        cnt['rest_orders_used'] += int(ok_p.sum())
        side = (O1['y_side'].to_numpy(object) == 'ask').astype(np.int64)
        RS = dict(klo=[], khi=[])
        for s_ in (0, 1):
            k_ = ok_p & (side == s_)
            RS['klo'].append(np.sort((gid_o[k_] << 42) | O1['persist_lo_ms'].to_numpy(np.int64)[k_]))
            RS['khi'].append(np.sort((gid_o[k_] << 42) | O1['persist_hi_ms'].to_numpy(np.int64)[k_]))
    ctx = dict(BT=BT, FR=FR, OW=OW, RS=RS)
    code_of = lambda mids: np.searchsorted(um, mids)    # noqa: E731  (every queried market is in um)
    # ---- take records (rows are actions) ----
    tk = rmask & (F['role'] == 1)
    T = pd.DataFrame(dict(gid=gid_row[tk], sid=F['sid'][tk], settle=F['settle'][tk], s=F['s'][tk], q=F['q'][tk], qp=F['q'][tk] * F['p'][tk],
                          selft=F['self'][tk]))
    TK = T.groupby(['gid', 'sid'], sort=True).agg(settle=('settle', 'min'), settle_hi=('settle', 'max'), s=('s', 'first'), s_lo=('s', 'min'),
                                                   s_hi=('s', 'max'), q=('q', 'sum'), qp=('qp', 'sum'), selft=('selft', 'max'), legs=('s', 'size')).reset_index()
    need(bool((TK['settle'] == TK['settle_hi']).all() and (TK['s_lo'] == TK['s_hi']).all()), f'{d}: a take record with mixed settle_ms / side')
    TK['pavg'] = TK['qp'] / TK['q']
    cnt['take_records_self_trade'] += int(TK['selft'].sum())
    tk_g, tk_t, tk_s = TK['gid'].to_numpy(np.int64), TK['settle'].to_numpy(np.int64), TK['s'].to_numpy(np.int64)
    # ---- the grid pass ----
    gids, nrow, t0s = PD['gid'].to_numpy(np.int64), PD['n'].to_numpy(np.int64), PD['t0'].to_numpy(np.int64)
    a_p, mid_p = PD['a'].to_numpy(np.int64), PD['mid'].to_numpy(np.int64)
    ws_p, we_p, bk_p = PD['ws'].to_numpy(np.int64), PD['we'].to_numpy(np.int64), PD['book'].to_numpy(bool)
    code_p = code_of(mid_p)
    s0, cuts = 0, []
    acc_rows = 0
    for j in range(len(PD)):
        if acc_rows and acc_rows + nrow[j] > chunk_rows:
            cuts.append((s0, j))
            s0, acc_rows = j, 0
        acc_rows += nrow[j]
    cuts.append((s0, len(PD)))
    # take records -> pair position in PD
    lp_all = np.searchsorted(gids, tk_g)
    kk_all = (tk_t - t0s[lp_all]) // STEP
    insp_all = (kk_all >= 0) & (kk_all < nrow[lp_all])
    for a_ in range(3):
        for side, sv in ((0, 1), (1, -1)):
            m_ = (a_p[lp_all] == a_) & (tk_s == sv)
            take_span[(a_, di, side, 'in_span')] += int((m_ & insp_all).sum())
            take_span[(a_, di, side, 'outside_span')] += int((m_ & ~insp_all).sum())
    for (c0, c1) in cuts:
        n = nrow[c0:c1]
        R = int(n.sum())
        if R == 0:
            continue
        start = np.cumsum(n) - n
        pl = np.repeat(np.arange(c1 - c0), n)
        t = np.repeat(t0s[c0:c1], n) + STEP * (np.arange(R) - np.repeat(start, n))
        ar, gr, wsr, wer = a_p[c0:c1][pl], gids[c0:c1][pl], ws_p[c0:c1][pl], we_p[c0:c1][pl]
        cr, br = code_p[c0:c1][pl], bk_p[c0:c1][pl]
        sel = (lp_all >= c0) & (lp_all < c1) & insp_all
        rr = start[lp_all[sel] - c0] + kk_all[sel]
        yb, ys = np.zeros(R), np.zeros(R)
        np.add.at(yb, rr[tk_s[sel] == 1], 1.)
        np.add.at(ys, rr[tk_s[sel] == -1], 1.)
        cnt['grid_buckets_multi_take'] += int((yb > 1).sum() + (ys > 1).sum())
        yb, ys = np.minimum(yb, 1.), np.minimum(ys, 1.)
        tl_here = [(g_, TL[g_]) for g_ in gids[c0:c1] if g_ in TL]
        on_l0 = None
        if tl_here:
            def on_l0(Fx, tl_here=tl_here, gr=gr, t=t, yb=yb, ys=ys):
                for g_, scope in tl_here:
                    timeline_rows(TLR, g_, scope, d, int(mid_p[np.searchsorted(gids, g_)]), gr == g_, t, Fx, yb, ys, F, gid_row, rmask)
        grid_chunk(ctx, t, ar, gr, wsr, wer, cr, br, yb, ys, di, ACC, fh, rows_st, ev_st, on_l0)
        E.guard(guard_mib, f'{d} grid chunk')
    # ---- maker fills (each leg) ----
    mkm = rmask & (F['role'] == 0)
    g_m = gid_row[mkm]
    lpm = np.searchsorted(gids, g_m)
    for l, L in enumerate(LAGS):
        x = F['settle'][mkm] - L
        Fx = evaluate(ctx, x, ws_p[lpm], we_p[lpm], code_p[lpm], g_m, bk_p[lpm])
        s, p, q = F['s'][mkm], F['p'][mkm], F['q'][mkm]
        beh = np.where(s == 1, np.round(100. * (Fx['bid'] - p)), np.round(100. * (p - Fx['ask'])))
        FILL.append(dict(a=a_p[lpm].astype(np.int8), d=np.full(len(x), di, np.int8), L=np.full(len(x), l, np.int8),
                         side=(s == -1).astype(np.int8), q=q.astype(np.float32), offset_c=(100. * (p - Fx['pf'])).astype(np.float32),
                         edge_c=(100. * s * (Fx['pf'] - p)).astype(np.float32), behind=beh.astype(np.float32), tau=Fx['tau'].astype(np.float32),
                         pos=Fx['pos'].astype(np.float32), bc=Fx['bcode'], kc=Fx['kcode'], book=bk_p[lpm]))
    # ---- take records (features at settle - L) ----
    for l, L in enumerate(LAGS):
        x = tk_t - L
        Fx = evaluate(ctx, x, ws_p[lp_all], we_p[lp_all], code_p[lp_all], tk_g, bk_p[lp_all])
        s, pv = tk_s, TK['pavg'].to_numpy(float)
        g_own = np.where(s == 1, Fx['g_buy'], Fx['g_sell'])
        thr = np.where(s == 1, 100. * (pv - Fx['ask']), 100. * (Fx['bid'] - pv))
        TAKE.append(dict(a=a_p[lp_all].astype(np.int8), d=np.full(len(x), di, np.int8), L=np.full(len(x), l, np.int8),
                         side=(s == -1).astype(np.int8), q=TK['q'].to_numpy(np.float32), g_own=g_own.astype(np.float32),
                         r1a=(s * Fx['r_1s']).astype(np.float32), r5a=(s * Fx['r_5s']).astype(np.float32), tau=Fx['tau'].astype(np.float32),
                         edge_c=(100. * s * (Fx['pf'] - pv)).astype(np.float32), through_c=thr.astype(np.float32), pos=Fx['pos'].astype(np.float32),
                         rs=Fx['rs'].astype(np.int8), selft=TK['selft'].to_numpy(bool), insp=insp_all))
    # ---- BOOK: posting, links, removal ----
    if len(ORD):
        life_rows(di, d, ORD, LNK, pair_of, PD, ctx, code_of, POST, LINK, REM, n_draws, cnt, STROWS)
    return ctx


def grid_chunk(ctx, t, ar, gr, wsr, wer, cr, br, yb, ys, di, ACC, fh, rows_st, ev_st, on_l0=None):
    """The per-L work of one chunk of grid rows (decision times t; address ar, pair gr, market ws / we / tob code, BOOK flag, events
    yb / ys): features at t - L, row statuses, hazard accumulators, logistic rows (FIT_DT, one file per L x address)."""
    for l, L in enumerate(LAGS):
        x = t - L
        Fx = evaluate(ctx, x, wsr, wer, cr, gr, br)
        rs = Fx['rs']
        rows_st[:, di, l, :] += np.bincount(ar * NRS + rs, minlength=3 * NRS).reshape(3, NRS)
        for side, y in ((0, yb), (1, ys)):
            ev_st[:, di, l, side, :] += np.bincount(ar * NRS + rs, weights=y, minlength=3 * NRS).reshape(3, NRS).astype(np.int64)
        v = rs == 0
        av = ar[v]
        gb, gs = Fx['g_buy'][v], Fx['g_sell'][v]
        r1, r5, tau, pos, ts_ = Fx['r_1s'][v], Fx['r_5s'][v], Fx['tau'][v], Fx['pos'][v], Fx['tsince'][v]
        bgb, bgs = g_bucket(gb), g_bucket(gs)
        s1, s5, ps = sign3(r1), sign3(r5), pos3(pos)
        t60 = np.clip((tau // 60).astype(np.int64), 0, NTAU60 - 1)
        bk = dict(g=(bgb, bgs), r_1s=m_bucket(r1, M1), r_5s=m_bucket(r5, M5), tau=np.clip((tau // 30).astype(np.int64), 0, NTAU - 1),
                  pos=ps, tsince=ts_bucket(ts_), rest=Fx['rest'][v],
                  g_x_r1sign=(bgb * 3 + s1, bgs * 3 + s1), g_x_r5sign=(bgb * 3 + s5, bgs * 3 + s5), g_x_pos=(bgb * 3 + ps, bgs * 3 + ps),
                  g_x_tau60=(bgb * NTAU60 + t60, bgs * NTAU60 + t60))
        acc_add(ACC, di, l, av, bk, yb[v], ys[v])
        rec = np.empty(int(v.sum()), FIT_DT)
        rec['d'], rec['gb'], rec['gs'], rec['r1'], rec['r5'], rec['tau'] = di, gb, gs, r1, r5, tau
        rec['ps'], rec['yb'], rec['ys'] = ps, yb[v], ys[v]
        for a_ in range(3):
            fh[(l, a_)].write(rec[av == a_].tobytes())
        if l == 0 and on_l0 is not None:
            on_l0(Fx)
        del Fx, rec


def load_life(paths, d, bookd, cnt, STROWS):
    """The top 3's lifecycle orders (every scenario) and links of a day's BOOK markets -> (orders DataFrame, links DataFrame)."""
    O, Lk = [], []
    for m in bookd:
        o = pq.read_table(paths[('orders', d, m)], columns=list(ORD_SCHEMA)).to_pandas()
        lk = pq.read_table(paths[('x_links', d, m)], columns=list(LNK_SCHEMA)).to_pandas()
        for df, lst, nm in ((o, O, 'orders'), (lk, Lk, 'links')):
            sg = np.char.lower(np.asarray(df['mk_signer'], dtype=str)) if len(df) else np.array([], dtype=str)
            ai = np.full(len(df), -1, np.int64)
            for i, a in enumerate(LOW3):
                ai[sg == a] = i
            cnt[f'life_{nm}_rows_read'] += int(len(df))
            df = df[ai >= 0].copy()
            df['a'] = ai[ai >= 0]
            lst.append(df)
    ORD = pd.concat(O, ignore_index=True) if O else pd.DataFrame(columns=list(ORD_SCHEMA) + ['a'])
    LNK = pd.concat(Lk, ignore_index=True) if Lk else pd.DataFrame(columns=list(LNK_SCHEMA) + ['a'])
    if len(ORD):
        need(set(ORD['scenario']) <= set(SCEN), f'{d}: an unknown lifecycle scenario')
        ORD['valid'] = [not (_flags(s) & ORDER_INVALID) for s in ORD['status']]
        for (a_, v_), g_ in ORD[ORD['scenario'] == PRIMARY].groupby(['a', 'valid']):
            STROWS.append(('orders', SCOPES[int(a_)], d, '-', OK if v_ else 'order_invalid', int(len(g_))))
    return ORD, LNK


def life_rows(di, d, ORD, LNK, pair_of, PD, ctx, code_of, POST, LINK, REM, n_draws, cnt, STROWS):
    mk_ws = {int(m): int(w) for m, w in zip(PD['mid'], PD['ws'])}
    mk_we = {int(m): int(w) for m, w in zip(PD['mid'], PD['we'])}
    gid_o = np.array([pair_of.get((int(a), int(m)), -1) for a, m in zip(ORD['a'], ORD['mid'])], np.int64)
    use = ORD['valid'].to_numpy(bool) & (gid_o >= 0)
    for (a_, sc_), g_ in ORD[ORD['valid'].to_numpy(bool) & (gid_o < 0)].groupby(['a', 'scenario']):
        STROWS.append(('orders_pair_excluded', SCOPES[int(a_)], d, sc_, 'pair_excluded', int(len(g_))))
    O = ORD[use].reset_index(drop=True)
    go = gid_o[use]
    if len(O) == 0:
        return
    mids = O['mid'].to_numpy(np.int64)
    ws, we, code = np.array([mk_ws[int(m)] for m in mids], np.int64), np.array([mk_we[int(m)] for m in mids], np.int64), code_of(mids)
    s = np.where(O['y_side'].to_numpy(object) == 'bid', 1, -1)
    price = O['y_ptick'].to_numpy(np.float64) / 100.
    a = O['a'].to_numpy(np.int64)
    hi, lo = O['post_hi_ms'].to_numpy(np.int64), O['post_lo_ms'].to_numpy(np.int64)
    scen = O['scenario'].to_numpy(object)
    # ---- posting: hi (scenario-free, PRIMARY rows), lo / mid per scenario ----
    for clock in ('hi', 'lo', 'mid'):
        if clock == 'hi':
            m_ = scen == PRIMARY
            xq = hi
        else:
            m_ = np.ones(len(O), bool)
            xq = lo if clock == 'lo' else np.where(lo >= 0, (lo + hi) // 2, -1)
        unb = m_ & (xq < 0)
        for (a_, sc_), n_ in Counter(zip(a[unb].tolist(), scen[unb].tolist())).items():
            STROWS.append(('posting', SCOPES[a_], d, f'{clock}/{sc_}', 'post_lo_unbounded', int(n_)))
        m_ = m_ & (xq >= 0)
        if not m_.any():
            continue
        for l, L in enumerate(LAGS):
            x = xq[m_] - L
            Fx = evaluate(ctx, x, ws[m_], we[m_], code[m_], go[m_], np.ones(int(m_.sum()), bool))
            pr, sv = price[m_], s[m_]
            POST.append(dict(a=a[m_].astype(np.int8), d=np.full(len(x), di, np.int8), L=np.full(len(x), l, np.int8),
                             side=(sv == -1).astype(np.int8), clock=np.full(len(x), clock, object),
                             scenario=np.where(clock == 'hi', '-', scen[m_]).astype(object), offset_c=(100. * (pr - Fx['pf'])).astype(np.float32),
                             edge_c=(100. * sv * (Fx['pf'] - pr)).astype(np.float32),
                             behind=np.where(sv == 1, np.round(100. * (Fx['bid'] - pr)), np.round(100. * (pr - Fx['ask']))).astype(np.float32),
                             size_lb=O['size_lb'].to_numpy(np.float32)[m_], tau=Fx['tau'].astype(np.float32), pos=Fx['pos'].astype(np.float32),
                             bc=Fx['bcode'], kc=Fx['kcode'], post_class=O['post_class'].to_numpy(object)[m_]))
    # ---- links (x_latest rows of valid pairs) ----
    if len(LNK):
        gl = np.array([pair_of.get((int(a_), int(m_)), -1) for a_, m_ in zip(LNK['a'], LNK['mid'])], np.int64)
        lst = LNK['status'].to_numpy(object)
        lat = LNK['x_latest'].to_numpy(bool)
        for (a_, st_, la_), n_ in Counter(zip(LNK['a'].tolist(), lst.tolist(), lat.tolist())).items():
            STROWS.append(('links', SCOPES[int(a_)], d, '-', f'{st_}' + ('' if la_ else '/not_latest'), int(n_)))
        uk = (gl >= 0) & (lst == OK) & lat
        cnt['links_pair_excluded'] += int(((gl < 0) & (lst == OK) & lat).sum())
        if uk.any():
            Lk = LNK[uk]
            lm_ = Lk['mid'].to_numpy(np.int64)
            lws, lwe = np.array([mk_ws[int(m)] for m in lm_], np.int64), np.array([mk_we[int(m)] for m in lm_], np.int64)
            for l, L in enumerate(LAGS):
                x0, x1 = Lk['prev_last_hi_ms'].to_numpy(np.int64) - L, Lk['next_first_hi_ms'].to_numpy(np.int64) - L
                B0, B1 = btc_feats(ctx['BT'], x0, lws, lwe), btc_feats(ctx['BT'], x1, lws, lwe)
                LINK.append(dict(a=Lk['a'].to_numpy(np.int8), d=np.full(len(x0), di, np.int8), L=np.full(len(x0), l, np.int8),
                                 side=(Lk['y_side'].to_numpy(object) == 'ask').astype(np.int8), kind=Lk['kind'].to_numpy(object),
                                 gap_ms=Lk['gap_ms'].to_numpy(np.float64), dtick=(Lk['next_ptick'] - Lk['prev_ptick']).to_numpy(np.float64),
                                 dpf_c=100. * (B1['pf'] - B0['pf']), r_bps=1e4 * (np.log(B1['mid']) - np.log(B0['mid'])),
                                 bc=np.maximum(B0['code'], B1['code'])))
    # ---- removal: cancel_possible vs exhausted_only vs a matched random-time null (same market, same PHASE, same width) ----
    rset = O['removal_set'].to_numpy(object)
    grp = np.array([0 if 'cancelled_or_expired' in str(x).split('|') else (1 if str(x) == 'exhausted' else 2) for x in rset], np.int64)
    rlo, rhi = O['removal_lo_ms'].to_numpy(np.int64), O['removal_hi_ms'].to_numpy(np.int64)
    ls = np.maximum(1, np.rint((rhi - rlo) / STEP).astype(np.int64))
    bnd = (rlo >= 0) & (rhi >= rlo)
    ph = (rlo >= ws).astype(np.int64)                                   # 0 pre-ws / 1 live [ws, we): the window start at L = 0
    wst = np.where(grp == 2, 'group_other', np.where(~bnd, 'window_unbounded', np.where((rlo < ws) & (rhi >= ws), 'straddles_ws',
                   np.where(rhi >= we, 'beyond_we', OK)))).astype(object)
    for (a_, sc_, g_, p_, w_), n_ in Counter(zip(a.tolist(), scen.tolist(), grp.tolist(), ph.tolist(), wst.tolist())).items():
        STROWS.append(('removal', SCOPES[a_], d, sc_, f"{('cancel_possible', 'exhausted_only', 'other')[g_]}/{PHASES[p_]}/{w_}", int(n_)))
    if not (wst == OK).any():
        return
    grid = {}                                                           # the market-day pool: every market of the day with usable orders
    for m in sorted(set(mids.tolist())):
        K = (mk_we[m] - (mk_ws[m] - PRE_MS)) // STEP
        xg = mk_ws[m] - PRE_MS + STEP * np.arange(K, dtype=np.int64)
        B = btc_feats(ctx['BT'], xg, np.full(K, mk_ws[m], np.int64), np.full(K, mk_we[m], np.int64))
        grid[m] = (B['pf'], np.log(B['mid']), PRE_MS // STEP, K)          # NaN where the BTC state is not usable
    cache = {}

    def placements(m, p_, w):
        """dp (c) and r (bps) of every usable placement of width w steps entirely inside phase p_ of market m."""
        k_ = (m, p_, w)
        if k_ not in cache:
            pf_, lm_, kp, K = grid[m]
            a0, a1 = (0, kp - 1 - w) if p_ == 0 else (kp, K - 1 - w)
            if a1 < a0:
                cache[k_] = (np.empty(0), np.empty(0))
            else:
                u = np.arange(a0, a1 + 1)
                dp_, r_ = 100. * (pf_[u + w] - pf_[u]), 1e4 * (lm_[u + w] - lm_[u])
                f = np.isfinite(dp_) & np.isfinite(r_)
                cache[k_] = (dp_[f], r_[f])
        return cache[k_]

    src = Counter()
    for si, sc_ in enumerate(SCEN):
        k_ = np.flatnonzero((scen == sc_) & (wst == OK))
        if not len(k_):
            continue
        rng = np.random.default_rng([SEED, si, int(d)])
        NUL = {mt: np.zeros((len(k_), n_draws), np.float32) for mt in REM_METRICS}
        okn = np.zeros(len(k_), bool)
        for j, i in enumerate(k_):
            dp_v, r_v = placements(int(mids[i]), int(ph[i]), int(ls[i]))
            where = 'market'
            if len(dp_v) < MIN_PLACE:
                pool = [placements(mm, int(ph[i]), int(ls[i])) for mm in grid]
                dp_v, r_v, where = np.concatenate([x_[0] for x_ in pool]), np.concatenate([x_[1] for x_ in pool]), 'market_day'
            if not len(dp_v):
                src[(int(a[i]), sc_, 'no_phase_placement')] += 1
                continue
            src[(int(a[i]), sc_, where)] += 1
            ix = rng.integers(0, len(dp_v), n_draws)
            NUL['adverse'][j], NUL['abs'][j], NUL['adverse_r'][j] = -s[i] * dp_v[ix], np.abs(dp_v[ix]), -s[i] * r_v[ix]
            okn[j] = True
        for l, L in enumerate(LAGS):
            B0, B1 = btc_feats(ctx['BT'], rlo[k_] - L, ws[k_], we[k_]), btc_feats(ctx['BT'], rhi[k_] - L, ws[k_], we[k_])
            dp = 100. * (B1['pf'] - B0['pf'])
            obs = dict(adverse=-s[k_] * dp, abs=np.abs(dp), adverse_r=-s[k_] * 1e4 * (np.log(B1['mid']) - np.log(B0['mid'])))
            if l == 0:
                cnt['removal_obs_invalid'] += int((okn & ~np.isfinite(dp)).sum())
            for mt in REM_METRICS:
                ov, nv = obs[mt], NUL[mt]
                okv = np.isfinite(ov) & okn
                for pi in (0, 1, 2):
                    for gi in (0, 1):
                        for a_ in (0, 1, 2, 3):
                            sel = okv & (grp[k_] == gi) & ((ph[k_] == pi) if pi < 2 else True) & ((a[k_] == a_) if a_ < 3 else True)
                            if not sel.any():
                                continue
                            R_ = REM.setdefault((l, a_, sc_, gi, mt, pi), dict(obs_sum=0., obs_n=0, null_sum=np.zeros(n_draws), pct_sum=0.))
                            o_, nn_ = ov[sel], nv[sel].astype(np.float64)
                            R_['obs_sum'] += float(o_.sum())
                            R_['obs_n'] += int(sel.sum())
                            R_['null_sum'] += nn_.sum(0)
                            R_['pct_sum'] += float((((nn_ < o_[:, None]).sum(1) + 0.5 * (nn_ == o_[:, None]).sum(1)) / n_draws).sum())
    for (a_, sc_, w_), n_ in sorted(src.items()):
        STROWS.append(('removal_null', SCOPES[a_], d, sc_, w_, int(n_)))


def timeline_rows(TLR, g_, scope, d, mid, msk, t, Fx, yb, ys, F, gid_row, rmask):
    """The L = 0 rows of one example pair (every status), with its own maker fills and takes per 100 ms bucket."""
    tt = t[msk]
    df = pd.DataFrame(dict(t_ms=tt, status=np.array(ROW_ST, dtype=object)[Fx['rs'][msk]], btc_mid=Fx['mid'][msk], r_ws_bps=Fx['r_ws'][msk],
                           r_1s_bps=Fx['r_1s'][msk], r_5s_bps=Fx['r_5s'][msk], sig60=Fx['sig'][msk], tau_s=Fx['tau'][msk], p_fair=Fx['pf'][msk],
                           best_bid=Fx['bid'][msk], best_ask=Fx['ask'][msk], bid_sh=Fx['bsz'][msk], ask_sh=Fx['asz'][msk],
                           g_buy_c=Fx['g_buy'][msk], g_sell_c=Fx['g_sell'][msk], pos_before=Fx['pos'][msk],
                           rest=np.array(('none', 'bid', 'ask', 'both', 'na'), dtype=object)[Fx['rest'][msk]],
                           take_buy=yb[msk], take_sell=ys[msk]))
    m = rmask & (gid_row == g_) & (F['role'] == 0)
    k = (F['settle'][m] - tt[0]) // STEP
    ins = (k >= 0) & (k < len(tt))
    for side, sv in (('buy', 1), ('sell', -1)):
        w = ins & (F['s'][m] == sv)
        n_ = np.bincount(k[w], minlength=len(tt))
        sh = np.bincount(k[w], weights=F['q'][m][w], minlength=len(tt))
        qp = np.bincount(k[w], weights=F['q'][m][w] * F['p'][m][w], minlength=len(tt))
        df[f'mk_{side}_n'], df[f'mk_{side}_sh'] = n_, sh
        with np.errstate(invalid='ignore', divide='ignore'):
            df[f'mk_{side}_px'] = np.where(sh > 0, qp / np.where(sh > 0, sh, 1.), np.nan)
    TLR.setdefault((scope, d, mid), []).append(df)


# ------------------------------------------------------------------ tables, claims and outputs ------------------------------------------------------------------
def write_tables(tmp, days, P, F, gid_row, valid_row, ACC, FITS, FILL, TAKE, POST, LINK, REM, rows_st, ev_st, take_span, TLR, STROWS, cnt, n_draws,
                 addrs):
    D = len(days)
    outputs, rows = {}, {}

    def put(name, df):
        p = tmp / name
        df.to_csv(p, index=False, float_format='%.10g')
        outputs[name] = dict(sha256=E.sha256(p), rows=int(len(df)))
        rows[name] = int(len(df))

    dayl = lambda dd: 'all' if dd is None else days[dd]   # noqa: E731
    # ---- statuses ----
    for a in range(3):
        for di in range(D):
            for l in range(NL):
                for k in range(NRS):
                    if rows_st[a, di, l, k]:
                        STROWS.append(('grid_rows', SCOPES[a], days[di], str(LAGS[l]), ROW_ST[k], int(rows_st[a, di, l, k])))
                    for side in (0, 1):
                        if ev_st[a, di, l, side, k]:
                            STROWS.append((f'grid_events_{("buy", "sell")[side]}', SCOPES[a], days[di], str(LAGS[l]), ROW_ST[k], int(ev_st[a, di, l, side, k])))
    for (a, di, side, k), n in sorted(take_span.items()):
        STROWS.append((f'take_records_{("buy", "sell")[side]}', SCOPES[a], days[di], '-', k, int(n)))
    for nm, R in (('fill_legs', FILL), ('take_records_features', TAKE), ('posting_rows', POST)):
        tot = np.zeros(3 * D * NL * NRS, np.int64)
        for X in R:                                   # per-day arrays, counted by an integer key (no concatenated copy)
            code = X['rs'].astype(np.int64) if 'rs' in X else np.where(X['bc'] > 0, X['bc'], np.where(X['kc'] > 0, 3 + X['kc'], 0)).astype(np.int64)
            key = ((X['a'].astype(np.int64) * D + X['d'].astype(np.int64)) * NL + X['L'].astype(np.int64)) * NRS + code
            tot += np.bincount(key, minlength=len(tot))
        for k_ in np.flatnonzero(tot):
            a, r_ = divmod(int(k_), D * NL * NRS)
            di, r_ = divmod(r_, NL * NRS)
            l, k = divmod(r_, NRS)
            STROWS.append((nm, SCOPES[a], days[di], str(LAGS[l]), ROW_ST[k], int(tot[k_])))
    # ---- hazard tables ----
    H1, H2 = [], []
    for t, B, sided, lab in TABLES:
        for sc in range(4):
            for dd in list(range(D)) + [None]:
                if t not in TABLE_1D and t not in TABLE_2D_DAILY and dd is not None:
                    continue
                N, Y = acc_get(ACC, t, sc, dd)
                for l in range(NL):
                    for side in (0, 1):
                        nz = np.flatnonzero(N[l, side] > 0)
                        for b in nz:
                            r_ = (SCOPES[sc], dayl(dd), LAGS[l], ('buy', 'sell')[side], t, lab[b], int(N[l, side, b]), int(Y[l, side, b]),
                                  float(Y[l, side, b] / N[l, side, b]))
                            (H1 if t in TABLE_1D else H2).append(r_)
    hc = ['scope', 'day', 'L', 'side', 'table', 'bucket', 'n', 'events', 'rate']
    put('take_hazard_1d.csv', pd.DataFrame(H1, columns=hc))
    put('take_hazard_2d.csv', pd.DataFrame(H2, columns=hc))
    TH = []
    for sc in range(4):
        for dd in list(range(D)) + [None]:
            for l in range(NL):
                for side in (0, 1):
                    for sl in SLICES:
                        nb, eb = slice_counts(ACC, sc, dd, l, side, sl)
                        TH.append(dict(scope=SCOPES[sc], day=dayl(dd), L=LAGS[l], side=('buy', 'sell')[side], slice=sl, **step_split(nb, eb)))
    TH = pd.DataFrame(TH)
    put('take_threshold.csv', TH)
    LG = []
    for (l, sc, side), m in sorted(FITS.items()):
        dev0 = np.nan
        if 0 < m['ev'] < m['n']:
            p0 = m['ev'] / m['n']
            dev0 = -2. * (m['ev'] * np.log(p0) + (m['n'] - m['ev']) * np.log(1 - p0))
        r_ = dict(L=LAGS[l], scope=SCOPES[sc], side=('buy', 'sell')[side], n=m['n'], events=m['ev'], status=m['status'], iterations=m['it'],
                  passes=m['passes'], dev=m['dev'], dev0=dev0, pseudo_r2=(1. - m['dev'] / dev0) if np.isfinite(dev0) and dev0 > 0 else np.nan,
                  dropped='|'.join(m['dropped']), singular=m['singular'])
        act = m['act'] if m['act'] is not None else np.zeros(7, bool)
        for i, c in enumerate(LOGIT_COLS):
            r_[f'b_{c}'] = float(m['beta'][i]) if act[i] and m['status'] in ('converged', 'max_iter', 'pass_cap', 'cpu_cap') else np.nan
        LG.append(r_)
    LG = pd.DataFrame(LG)
    put('take_logit.csv', LG)
    # ---- take records ----
    TKX = _cat_arrays(TAKE)
    TAKE.clear()
    if TKX:
        tdf = pd.DataFrame({k: v for k, v in TKX.items()})
        tv = tdf[tdf['rs'] == 0].copy()
        for c_ in (0., 1., 2.):
            tv[f'g_ge{int(c_)}'] = (tv['g_own'] >= c_).astype(float)
        tv['r1a_pos'] = (tv['r1a'] > 0).astype(float)
        TE = grouped(tv, ['a', 'd', 'L', 'side'], dict(n=('g_own', 'size'), shares=('q', 'sum'), g_med=('g_own', 'median'), share_g_ge0=('g_ge0', 'mean'),
                                                       share_g_ge1=('g_ge1', 'mean'), share_g_ge2=('g_ge2', 'mean'), r1a_med=('r1a', 'median'),
                                                       share_r1a_pos=('r1a_pos', 'mean'), r5a_med=('r5a', 'median'), tau_med=('tau', 'median'),
                                                       edge_mean=('edge_c', 'mean'), edge_med=('edge_c', 'median'), through_med=('through_c', 'median'),
                                                       pos_mean=('pos', 'mean')), days)
        TE['L'] = [LAGS[int(x)] for x in TE['L']]
        TE['side'] = [('buy', 'sell')[int(x)] for x in TE['side']]
    else:
        TE = pd.DataFrame()
    put('take_events.csv', TE)
    # ---- post path at fill (memory: the per-day lists are freed, rows filtered before the frame is built, int8 indicators) ----
    FX = _cat_arrays(FILL)
    FILL.clear()
    PF_, PFT, SKP = pd.DataFrame(), pd.DataFrame(), []
    if FX:
        keep = FX['bc'] == 0
        fv = pd.DataFrame({k: v[keep] for k, v in FX.items() if k not in ('bc', 'book')}, copy=False)
        del FX, keep
        _book_flags(fv)
        fv['qe'] = (fv['q'] * fv['edge_c']).astype(np.float32)
        PF_ = grouped(fv, ['a', 'd', 'L', 'side'], dict(n=('offset_c', 'size'), shares=('q', 'sum'), qe=('qe', 'sum'), offset_mean=('offset_c', 'mean'),
                                                        offset_med=('offset_c', 'median'), edge_mean=('edge_c', 'mean'), edge_med=('edge_c', 'median'),
                                                        n_book=('bk', 'sum'), n_at=('at', 'sum'), n_ins=('ins', 'sum'), n_b1=('b1', 'sum'), n_b2=('b2', 'sum'),
                                                        n_b3=('b3', 'sum'), q_med=('q', 'median'), tau_med=('tau', 'median'), pos_mean=('pos', 'mean')), days)
        _book_shares(PF_)
        PF_['edge_share_weighted'] = PF_['qe'] / PF_['shares']
        PF_ = PF_.drop(columns=['qe'])
        fv['tau60'] = np.clip((fv['tau'].to_numpy() // 60).astype(np.int64), -1, NTAU60 - 1)
        PFT = grouped(fv, ['a', 'L', 'side', 'tau60'], dict(n=('offset_c', 'size'), offset_mean=('offset_c', 'mean'), edge_mean=('edge_c', 'mean'),
                                                            n_book=('bk', 'sum'), n_at=('at', 'sum')), days)
        with np.errstate(invalid='ignore', divide='ignore'):
            PFT['at_best'] = np.where(PFT['n_book'] > 0, PFT['n_at'] / PFT['n_book'].where(PFT['n_book'] > 0, 1), np.nan)
        PFT = PFT.drop(columns=['n_at'])
        PFT['tau60'] = ['<0' if x < 0 else TAU60_LAB[int(x)] for x in PFT['tau60']]
        for df_ in (PF_, PFT):
            df_['L'] = [LAGS[int(x)] for x in df_['L']]
            df_['side'] = [('bid', 'ask')[int(x)] for x in df_['side']]
        SKP.append(skew_sums(fv['a'].to_numpy(), fv['L'].to_numpy(), fv['side'].to_numpy(), fv['pos'].to_numpy(np.float64) / 100.,
                             fv['offset_c'].to_numpy(np.float64), 'fill', '-'))
        del fv
        E.trim()
    put('post_fill.csv', PF_)
    put('post_fill_tau.csv', PFT)
    # ---- post path at posting ----
    PX = _cat_arrays(POST)
    POST.clear()
    PP = pd.DataFrame()
    if PX:
        for (a_, cl_, pc_), n_ in Counter(zip(PX['a'].tolist(), PX['clock'].tolist(), PX['post_class'].tolist())).items():
            STROWS.append(('posting_class', SCOPES[a_], 'all', cl_, str(pc_), int(n_)))
        keep = PX['bc'] == 0
        pv = pd.DataFrame({k: v[keep] for k, v in PX.items() if k != 'bc'}, copy=False)
        del PX, keep
        _book_flags(pv)
        PP = grouped(pv, ['a', 'd', 'L', 'clock', 'scenario', 'side'],
                     dict(n=('offset_c', 'size'), offset_mean=('offset_c', 'mean'), offset_med=('offset_c', 'median'), edge_mean=('edge_c', 'mean'),
                          edge_med=('edge_c', 'median'), n_book=('bk', 'sum'), n_at=('at', 'sum'), n_ins=('ins', 'sum'), n_b1=('b1', 'sum'),
                          n_b2=('b2', 'sum'), n_b3=('b3', 'sum'), size_lb_med=('size_lb', 'median'), tau_med=('tau', 'median'),
                          pos_mean=('pos', 'mean')), days)
        _book_shares(PP)
        PP['L'] = [LAGS[int(x)] for x in PP['L']]
        PP['side'] = [('bid', 'ask')[int(x)] for x in PP['side']]
        for (cl_, scn_), g_ in pv.groupby(['clock', 'scenario'], sort=True):
            SKP.append(skew_sums(g_['a'].to_numpy(), g_['L'].to_numpy(), g_['side'].to_numpy(), g_['pos'].to_numpy(np.float64) / 100.,
                                 g_['offset_c'].to_numpy(np.float64), cl_, scn_))
        del pv
    put('post_posting.csv', PP)
    SKD = pd.DataFrame()
    if SKP:
        SKD = ols_from_sums(pd.concat(SKP, ignore_index=True))
        SKD = SKD.drop(columns=['sx', 'sy', 'sxx', 'sxy', 'syy']).rename(columns=dict(a='a_c', b='b_c_per_100sh'))
    put('post_skew.csv', SKD)
    # ---- links ----
    LX = _cat_arrays(LINK)
    LKD = pd.DataFrame()
    if LX:
        ldf = pd.DataFrame({k: v for k, v in LX.items()})
        lv = ldf[ldf['bc'] == 0].copy()
        lv['agree'] = np.where((lv['dtick'] != 0) & (lv['dpf_c'] != 0), (np.sign(lv['dtick']) == np.sign(lv['dpf_c'])).astype(float), np.nan)
        lv['absdp'] = lv['dpf_c'].abs()
        lv['small'] = (lv['absdp'] < 1.).astype(float)
        lv['n'] = 1
        lv = _sums(lv, 'dpf_c', 'dtick')
        LKD = grouped(lv, ['a', 'L', 'kind', 'side'], dict(n=('n', 'sum'), gap_p10=('gap_ms', lambda x: x.quantile(.1)),
                                                           gap_p25=('gap_ms', lambda x: x.quantile(.25)), gap_p50=('gap_ms', 'median'),
                                                           gap_p75=('gap_ms', lambda x: x.quantile(.75)), gap_p90=('gap_ms', lambda x: x.quantile(.9)),
                                                           dtick_mean=('dtick', 'mean'), dpf_mean=('dpf_c', 'mean'), absdp_med=('absdp', 'median'),
                                                           share_absdp_lt1c=('small', 'mean'), sign_agree=('agree', 'mean'),
                                                           r_bps_med=('r_bps', 'median'), sx=('sx', 'sum'), sy=('sy', 'sum'), sxx=('sxx', 'sum'),
                                                           sxy=('sxy', 'sum'), syy=('syy', 'sum')), days)
        LKD = ols_from_sums(LKD).drop(columns=['sx', 'sy', 'sxx', 'sxy', 'syy']).rename(columns=dict(a='ols_intercept', b='ols_slope_dtick_on_dpf'))
        LKD['L'] = [LAGS[int(x)] for x in LKD['L']]
        LKD['side'] = [('bid', 'ask')[int(x)] for x in LKD['side']]
        for (a_, l_, b_), n_ in Counter(zip(ldf['a'].tolist(), ldf['L'].tolist(), ldf['bc'].tolist())).items():
            STROWS.append(('links_btc', SCOPES[a_], 'all', str(LAGS[l_]), ROW_ST[b_], int(n_)))
    put('links.csv', LKD)
    # ---- removal ----
    RM = []
    unit = dict(adverse='c', abs='c', adverse_r='bps')

    def rem_row(l, a_, sc_, gname, metric, pi, obs, reps, n_o, pct, note):
        fin = np.isfinite(reps)
        mu = float(reps[fin].mean()) if fin.any() else np.nan
        sd = float(reps[fin].std(ddof=1)) if fin.sum() > 1 else np.nan
        return dict(L=LAGS[l], scope=SCOPES[a_], scenario=sc_, phase=PHASES[pi], group=gname, metric=metric, unit=unit[metric], n_orders=int(n_o),
                    draws=n_draws, obs_mean=obs, null_mean=mu, null_sd=sd, null_p05=float(np.percentile(reps[fin], 5)) if fin.any() else np.nan,
                    null_p95=float(np.percentile(reps[fin], 95)) if fin.any() else np.nan,
                    z=float((obs - mu) / sd) if np.isfinite(obs) and np.isfinite(sd) and sd > 0 else np.nan,
                    p_one_sided=float((1 + (reps[fin] >= obs).sum()) / (1 + fin.sum())) if fin.any() and np.isfinite(obs) else np.nan,
                    mean_percentile=pct, null_declared=note)
    note = (f'per order {n_draws} windows of the same width, uniform over the usable placements of the SAME PHASE in the same market '
            f'(market-day pool when < {MIN_PLACE}), seeded; group-mean replicates')
    for (l, a_, sc_, gi, metric, pi), R_ in sorted(REM.items(), key=lambda kv: tuple(map(str, kv[0]))):
        n_o = R_['obs_n']
        RM.append(rem_row(l, a_, sc_, ('cancel_possible', 'exhausted_only')[gi], metric, pi, R_['obs_sum'] / n_o, R_['null_sum'] / n_o, n_o,
                          R_['pct_sum'] / n_o, note))
    for (l, a_, sc_, gi, metric, pi), R_ in sorted(REM.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if gi != 0 or (l, a_, sc_, 1, metric, pi) not in REM:
            continue
        Re = REM[(l, a_, sc_, 1, metric, pi)]
        RM.append(rem_row(l, a_, sc_, 'cancel_minus_exhausted', metric, pi, R_['obs_sum'] / R_['obs_n'] - Re['obs_sum'] / Re['obs_n'],
                          R_['null_sum'] / R_['obs_n'] - Re['null_sum'] / Re['obs_n'], R_['obs_n'] + Re['obs_n'], np.nan,
                          'paired replicate means: cancel_possible minus exhausted_only (each against its phase-matched null)'))
    RMD = pd.DataFrame(RM)
    put('removal.csv', RMD)
    # ---- inventory (actions, no BTC) ----
    INV = inventory(F, gid_row, valid_row, days)
    put('inventory.csv', INV)
    # ---- timelines ----
    for (scope, d, mid), parts in sorted(TLR.items()):
        df = pd.concat(parts, ignore_index=True).sort_values('t_ms', kind='mergesort')
        put(f'timeline_{scope}_{d}_{mid}.csv', df)
    # ---- claims ----
    claims = make_claims(TH, LG, SKD, INV, RMD, LKD, ACC)
    p = tmp / 'claims_decision.json'
    p.write_text(json.dumps(dict(version=VERSION, contract=CONTRACT, rule='computed predicates only (CLAUDE.md rule 10); per_L holds is None when '
                                 'the predicate cannot be evaluated; lag_sensitive = the evaluated values differ across L', claims=claims),
                            indent=1, sort_keys=True, default=_json) + '\n')
    outputs['claims_decision.json'] = dict(sha256=E.sha256(p), rows=len(claims))
    # ---- per-address summary ----
    put('address_summary.csv', address_summary(P, F, valid_row, gid_row, ACC, TH, LG, SKD, INV, days))
    stc = pd.DataFrame(STROWS, columns=['stage', 'scope', 'day', 'L', 'status', 'n'])      # written last: every stage's status rows
    put('statuses.csv', stc.sort_values(['stage', 'scope', 'day', 'L', 'status'], kind='mergesort'))
    return dict(outputs=outputs, rows=rows)


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def inventory(F, gid_row, valid_row, days):
    """Actions (pair, record, role): does each reduce |position before| (strictly earlier settle_ms)? Self-trades excluded."""
    m = valid_row & ~F['self']
    A = pd.DataFrame(dict(gid=gid_row[m], d=F['d'][m], a=F['a'][m], role=F['role'][m], sid=F['sid'][m], settle=F['settle'][m], sq=F['s'][m] * F['q'][m],
                          s=F['s'][m]))
    if len(A) == 0:
        return pd.DataFrame()
    G = A.groupby(['gid', 'sid', 'role'], sort=False).agg(d=('d', 'first'), a=('a', 'first'), settle=('settle', 'first'), sq=('sq', 'sum'),
                                                          s=('s', 'first')).reset_index()
    G = G.sort_values(['gid', 'settle'], kind='mergesort').reset_index(drop=True)
    G['cum'] = G.groupby('gid')['sq'].cumsum()
    key = (G['gid'].to_numpy(np.int64) << 42) | G['settle'].to_numpy(np.int64)
    j = np.searchsorted(key, key, side='left') - 1          # the last action strictly before this settle_ms in the same pair
    jc = np.maximum(j, 0)
    same = (j >= 0) & (G['gid'].to_numpy()[jc] == G['gid'].to_numpy())
    pos = np.where(same, G['cum'].to_numpy()[jc], 0.)
    G['pos'] = pos
    G['absb'] = np.searchsorted(POS_B, np.abs(pos), side='left')
    G['red'] = (G['s'] * pos < 0).astype(float)
    G['sell'] = (G['s'] < 0).astype(float)
    G['long'], G['short'] = pos > POS_EPS, pos < -POS_EPS
    rows = []
    lab = [f'(0,{int(POS_B[0])}]'] + [f'({int(POS_B[i])},{int(POS_B[i + 1])}]' for i in range(len(POS_B) - 1)] + [f'>{int(POS_B[-1])}']
    for pooled in (False, True):
        for alld in (False, True):
            for role in (0, 1):
                X = G[(G['role'] == role) & (np.abs(G['pos']) > POS_EPS)]
                keys = ([] if pooled else ['a']) + ([] if alld else ['d'])
                for kv, Xg in (X.groupby(keys, sort=True) if keys else [((), X)]):
                    kv = kv if isinstance(kv, tuple) else (kv,)
                    kd = dict(zip(keys, kv))
                    base = dict(scope='pooled' if pooled else SCOPES[int(kd['a'])], day='all' if alld else days[int(kd['d'])],
                                role=('maker', 'taker')[role])
                    lg, sh = Xg[Xg['long']], Xg[Xg['short']]
                    rows.append(dict(base, pos_bucket='all', n=int(len(Xg)), n_reduce=int(Xg['red'].sum()),
                                     rate_reduce=float(Xg['red'].mean()) if len(Xg) else np.nan, n_long=int(len(lg)), n_short=int(len(sh)),
                                     p_sell_long=float(lg['sell'].mean()) if len(lg) else np.nan,
                                     p_sell_short=float(sh['sell'].mean()) if len(sh) else np.nan))
                    for b in range(len(lab)):
                        Xb = Xg[Xg['absb'] == b]
                        if len(Xb):
                            rows.append(dict(base, pos_bucket=lab[b], n=int(len(Xb)), n_reduce=int(Xb['red'].sum()), rate_reduce=float(Xb['red'].mean())))
    return pd.DataFrame(rows)


def make_claims(TH, LG, SKD, INV, RMD, LKD, ACC):
    claims = []

    def add(cid, scope, side, statement, per_L):
        hv = [v['holds'] for v in per_L.values() if v.get('holds') is not None]
        claims.append(dict(claim=cid, scope=scope, side=side, statement=statement, per_L=per_L, n_L_evaluated=len(hv),
                           lag_sensitive=bool(len(set(hv)) > 1)))

    def row(df, **kw):
        if df is None or len(df) == 0:
            return None
        m = np.ones(len(df), bool)
        for k, v in kw.items():
            if k not in df.columns:
                return None
            m &= (df[k].astype(str) == str(v)).to_numpy()
        return df[m].iloc[0] if m.any() else None

    for sc in SCOPES:
        for side in ('buy', 'sell'):
            for sl in ('all', 'r1_aligned'):
                per = {}
                for L in LAGS:
                    r = row(TH, scope=sc, day='all', L=L, side=side, slice=sl)
                    if r is None or r['status'] not in (OK, 'below_zero_events'):
                        per[str(L)] = dict(holds=None, status=None if r is None else r['status'])
                        continue
                    hold = bool(r['rate_above'] >= TEN_X * r['rate_below']) if r['rate_below'] > 0 else bool(r['rate_above'] > 0)
                    per[str(L)] = dict(holds=hold, threshold_c=r['threshold_c'], rate_above=r['rate_above'], rate_below=r['rate_below'],
                                       n_above=int(r['n_above']), n_below=int(r['n_below']), e_above=int(r['e_above']), e_below=int(r['e_below']))
                add('take_rate_above_gap_threshold_ge_10x_below', sc, f'{side}/{sl}',
                    f'max-likelihood g split (both sides >= {N_MIN_SPLIT} rows): rate above >= {TEN_X:g} x rate below', per)
            per = {}
            sci = SCOPES.index(sc)
            for l, L in enumerate(LAGS):
                N, Y = acc_get(ACC, 'g_x_r1sign', sci, None)
                si = 0 if side == 'buy' else 1
                n3, y3 = N[l, si].reshape(NG, 3).sum(0), Y[l, si].reshape(NG, 3).sum(0)
                al, ag = (2, 0) if side == 'buy' else (0, 2)
                if n3[al] < N_MIN_SPLIT or n3[ag] < N_MIN_SPLIT:
                    per[str(L)] = dict(holds=None, n_aligned=int(n3[al]), n_against=int(n3[ag]))
                    continue
                ra, rg = y3[al] / n3[al], y3[ag] / n3[ag]
                per[str(L)] = dict(holds=bool(ra > rg), rate_aligned=float(ra), rate_against=float(rg), n_aligned=int(n3[al]), n_against=int(n3[ag]),
                                   e_aligned=int(y3[al]), e_against=int(y3[ag]))
            add('take_rate_higher_when_r_1s_aligned', sc, side, 'take rate with sign(r_1s) = the side > take rate with the opposite sign', per)
            per = {}
            for L in LAGS:
                r = row(LG, scope=sc, side=side, L=L)
                if r is None or r['status'] != 'converged' or not np.isfinite(r['b_g_c']):
                    per[str(L)] = dict(holds=None, status=None if r is None else r['status'])
                    continue
                per[str(L)] = dict(holds=bool(r['b_g_c'] > 0), b_g_c=float(r['b_g_c']), n=int(r['n']), events=int(r['events']))
            add('logit_gap_coefficient_positive', sc, side, 'declared logistic fit converged and its g coefficient > 0', per)
        for clock, scen in (('fill', '-'), ('hi', '-')):
            per, per2 = {}, {}
            for L in LAGS:
                rb, ra = row(SKD, scope=sc, day='all', L=L, clock=clock, scenario=scen, side='bid'), row(SKD, scope=sc, day='all', L=L, clock=clock,
                                                                                                          scenario=scen, side='ask')
                if rb is None or ra is None or not (np.isfinite(rb['a_c']) and np.isfinite(ra['a_c'])):
                    per[str(L)] = per2[str(L)] = dict(holds=None)
                    continue
                cen = (rb['a_c'] + ra['a_c']) / 2.
                per[str(L)] = dict(holds=bool(abs(cen) <= CENTRE_TOL_C), centre_c=float(cen), half_spread_c=float((ra['a_c'] - rb['a_c']) / 2.),
                                   a_bid_c=float(rb['a_c']), a_ask_c=float(ra['a_c']), n_bid=int(rb['n']), n_ask=int(ra['n']))
                ok2 = np.isfinite(rb['b_c_per_100sh']) and np.isfinite(ra['b_c_per_100sh'])
                per2[str(L)] = dict(holds=bool(rb['b_c_per_100sh'] < 0 and ra['b_c_per_100sh'] < 0) if ok2 else None,
                                    b_bid=float(rb['b_c_per_100sh']), b_ask=float(ra['b_c_per_100sh']), n_bid=int(rb['n']), n_ask=int(ra['n']))
            add('quotes_centred_on_p_fair', sc, f'{clock}', f'OLS intercepts (position 0): |(a_bid + a_ask) / 2| <= {CENTRE_TOL_C:g} c', per)
            add('quotes_skew_against_inventory', sc, f'{clock}', 'OLS slopes on position / 100 sh: b_bid < 0 and b_ask < 0', per2)
        for role, cid in (('maker', 'next_maker_fill_reduces_position'), ('taker', 'takes_reduce_position')):
            r = row(INV, scope=sc, day='all', role=role, pos_bucket='all')
            per = {'-': dict(holds=None) if r is None else dict(holds=bool(r['rate_reduce'] > 0.5), rate_reduce=float(r['rate_reduce']), n=int(r['n']),
                                                                 p_sell_long=float(r['p_sell_long']), p_sell_short=float(r['p_sell_short']))}
            add(cid, sc, role, 'share of actions (position before != 0) that reduce |position| > 0.5', per)
        for mt, cid, what in (('adverse', 'removal_windows_adverse_beyond_null', 'mean adverse p_fair move'),
                              ('abs', 'removal_windows_move_beyond_null', 'mean |p_fair move|')):
            per = {}
            for L in LAGS:
                r = row(RMD, scope=sc, L=L, scenario=PRIMARY, phase='live', group='cancel_possible', metric=mt)
                if r is None or not np.isfinite(r['p_one_sided']):
                    per[str(L)] = dict(holds=None)
                    continue
                per[str(L)] = dict(holds=bool(r['p_one_sided'] <= 0.05), p=float(r['p_one_sided']), z=float(r['z']), obs_mean_c=float(r['obs_mean']),
                                   null_mean_c=float(r['null_mean']), n_orders=int(r['n_orders']), draws=int(r['draws']))
            add(cid, sc, f'{PRIMARY}/cancel_possible/live',
                f'live phase: {what} inside removal windows exceeds the phase-matched random-time null at one-sided p <= 0.05', per)
        per = {}
        for L in LAGS:
            rr = LKD[(LKD['scope'] == sc) & (LKD['day'] == 'all') & (LKD['L'] == L) & (LKD['kind'] == 'reprice')] if len(LKD) else LKD
            if len(rr) == 0:
                per[str(L)] = dict(holds=None)
                continue
            n_ = rr['n'].sum()
            sl_ = float((rr['ols_slope_dtick_on_dpf'] * rr['n']).sum() / n_) if rr['ols_slope_dtick_on_dpf'].notna().all() else np.nan
            ag_ = float((rr['sign_agree'] * rr['n']).sum() / n_) if rr['sign_agree'].notna().all() else np.nan
            per[str(L)] = dict(holds=bool(sl_ > 0 and ag_ > 0.5) if np.isfinite(sl_) and np.isfinite(ag_) else None, slope=sl_, sign_agree=ag_, n=int(n_))
        add('reprice_tracks_p_fair', sc, 'both', 'n-weighted OLS slope of the tick change on dp > 0 and sign agreement > 0.5 (reprices)', per)
    return claims


def address_summary(P, F, valid_row, gid_row, ACC, TH, LG, SKD, INV, days):
    rows = []
    for sc in range(4):
        Pa = P if sc == 3 else P[P['a'] == sc]
        fm = valid_row & ((F['a'] == sc) if sc < 3 else True)
        r = dict(scope=SCOPES[sc], address=TOP3[sc] if sc < 3 else 'pooled', pairs_ok=int((Pa['status'] == OK).sum()),
                 pairs_excluded=int((Pa['status'] != OK).sum()), markets_ok=int(Pa[Pa['status'] == OK]['mid'].nunique()),
                 maker_legs=int((fm & (F['role'] == 0)).sum()), maker_shares=float(F['q'][fm & (F['role'] == 0)].sum()),
                 taker_legs=int((fm & (F['role'] == 1)).sum()),
                 take_records=int(len(set(zip(gid_row[fm & (F['role'] == 1)].tolist(), F['sid'][fm & (F['role'] == 1)].tolist())))))
        for l, L in enumerate(LAGS):
            N, Y = acc_get(ACC, 'pos', sc, None)
            for si, side in enumerate(('buy', 'sell')):
                n_, e_ = N[l, si].sum(), Y[l, si].sum()
                r[f'L{L}_{side}_rows'], r[f'L{L}_{side}_events'] = int(n_), int(e_)
                r[f'L{L}_{side}_rate'] = float(e_ / n_) if n_ else np.nan
                t = TH[(TH['scope'] == SCOPES[sc]) & (TH['day'] == 'all') & (TH['L'] == L) & (TH['side'] == side) & (TH['slice'] == 'r1_aligned')]
                r[f'L{L}_{side}_thr_r1_aligned_c'] = float(t['threshold_c'].iloc[0]) if len(t) else np.nan
                lg = LG[(LG['scope'] == SCOPES[sc]) & (LG['L'] == L) & (LG['side'] == side)]
                r[f'L{L}_{side}_logit_b_g'] = float(lg['b_g_c'].iloc[0]) if len(lg) else np.nan
            for side in ('bid', 'ask'):
                s_ = SKD[(SKD['scope'] == SCOPES[sc]) & (SKD['day'] == 'all') & (SKD['L'] == L) & (SKD['clock'] == 'fill') & (SKD['side'] == side)] \
                    if len(SKD) else SKD
                r[f'L{L}_fill_{side}_a_c'] = float(s_['a_c'].iloc[0]) if len(s_) else np.nan
                r[f'L{L}_fill_{side}_b_c_per_100sh'] = float(s_['b_c_per_100sh'].iloc[0]) if len(s_) else np.nan
        for role in ('maker', 'taker'):
            iv = INV[(INV['scope'] == SCOPES[sc]) & (INV['day'] == 'all') & (INV['role'] == role) & (INV['pos_bucket'] == 'all')] if len(INV) else INV
            r[f'{role}_rate_reduce'] = float(iv['rate_reduce'].iloc[0]) if len(iv) else np.nan
            r[f'{role}_n_reduce_test'] = int(iv['n'].iloc[0]) if len(iv) else 0
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ schema probe (footers only) ------------------------------------------------------------------
def schemas(root=None, btc_dir=None):
    """Footer schemas of every input family (pyarrow.read_schema only; no data page is read) + a compile check of the selftest."""
    root = Path(root) if root is not None else C.OUT
    btc_dir = Path(btc_dir) if btc_dir is not None else root / 'btc'
    res = {}

    def chk(p, want):
        if not p.is_file():
            res[str(p)] = 'missing'
            return
        s = pq.read_schema(p)
        res[str(p)] = dict(missing=[c for c in want if c not in s.names],
                           bad_type=[f'{c}:{s.field(c).type}' for c, t in want.items() if c in s.names and not E._type_ok(s.field(c).type, t)],
                           extra=[f'{n}:{s.field(n).type}' for n in s.names if n not in want])
    chk(root / 'econ' / 'fills_valued.parquet', {c: E.FV[c] for c in FV_COLS})
    book = E.load_book()
    res['btc_dir_files'] = sorted(p.name for p in Path(btc_dir).glob('*.parquet')) if Path(btc_dir).is_dir() else 'missing'
    for d in C.MEASURE_DAYS:
        chk(btc_path(btc_dir, d), BTC_SCHEMA)
        rp = Path(btc_dir) / f'ref_{d}.parquet'                     # never read by the study (returns only): footer recorded
        res[str(rp)] = [f'{f.name}:{f.type}' for f in pq.read_schema(rp)] if rp.is_file() else 'missing'
    for d in C.MEASURE_DAYS[:1]:
        chk(root / f'markets_{d}.parquet', MKT_SCHEMA)
        chk(root / f'tob_{d}.parquet', {c: C.SCHEMAS['tob'][c] for c in TOB_COLS})
        m = book[d][0]
        chk(root / 'life' / f'{d}_{m}_orders.parquet', ORD_SCHEMA)
        chk(root / 'life' / f'{d}_{m}_x_links.parquet', LNK_SCHEMA)
    st = HERE / 'pf_mt_decision_selftest.py'
    compile(st.read_text(), str(st), 'exec')
    res['selftest_compiles'] = True
    return res


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[:1] == ['schemas']:
        print(json.dumps(schemas(a[1] if len(a) > 1 else None), indent=1))
    else:
        need(a == ['run'], 'usage: pf_mt_decision.py run | schemas [root]')
        need(C.RUN != 'v1', 'MT_RUN=v1 is refused: the decision study reads the re-run roots (MT_RUN=v2)')
        r = run()
        print(json.dumps(dict(rows=r['rows'], cpu_s=r['cpu_s'], memory=r['memory']), indent=1, default=str))
