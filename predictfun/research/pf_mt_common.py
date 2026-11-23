"""MAKER-TRACE-RECONSTRUCTION-v1 shared contract (P-2026-005; session aea5bfb9, 2026-11-22). Plan:
predictfun/reviews/automatic/20261122_maker_trace_reconstruction/{plan.md, coverage_amendment.md}; semantics in
predictfun/reviews/maker_trace_20261122/contract.md. This module is the single source of the day/hour guard, paths, table schemas,
status codes, inference classes and scenarios; every pf_mt_* module and mt_checks.py import it. It never opens a data file itself."""
import datetime as dt
import uuid
from pathlib import Path

REPO = Path('/home/yuqing/ctaNew')
PF_RUN = REPO / 'data/predict_fun/runs/20261021T162823Z_0dd45f2d5a'
OUT = REPO / 'data/predict_fun/research/maker_trace_v1_20261122'
RECEIPT = REPO / 'predictfun/reviews/maker_trace_20261122'
# amendment 3 (2026-11-23): a downstream re-run (MT_RUN=v<n>) switches the ROOTS, so no module path changes: OUT -> maker_trace_<run>_20261123
# (tools/setup_run.py fills it with read-only links to the shared v1 extract outputs and the v2 truth table), RECEIPT -> the receipt's <run>/
# subdirectory. Every v1 output stays untouched (write-once); after a fix only the re-run's numbers are reported.
import os as _os
RUN = _os.environ.get('MT_RUN', 'v1')
if not (RUN == 'v1' or (RUN[:1] == 'v' and RUN[1:].isdigit())):
    raise ValueError(f'bad MT_RUN {RUN!r}')
OUT_V1, RECEIPT_V1 = OUT, RECEIPT
if RUN != 'v1':
    OUT = REPO / f'data/predict_fun/research/maker_trace_{RUN}_20261123'
    RECEIPT = RECEIPT_V1 / RUN
STEP1 = REPO / 'data/predict_fun/research/step1'
E18 = 10 ** 18
GUARD_MIB = 1450.                      # self-guard below the 1,536 MiB checkpoint (never raised)
VERSION = 'maker_trace_v1'

# ---- the date rule: consumed days only. Oct 23-27 whole days; Nov 8 hours 00..22 (consumed to 23:02Z, hour 23 is not whole) ----
ALLOWED_DAYS = ('20261023', '20261024', '20261025', '20261026', '20261027')
NOV8, NOV8_LAST_HOUR = '20261108', 22
ID_DAY = '20261023'                                                  # ex-ante identity selection only
MEASURE_DAYS = ('20261024', '20261025', '20261026', '20261027')      # reconstruction + economics
GT_DAY = NOV8                                                        # our own live orders (ground-truth control)


class Forbidden(ValueError):
    pass


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def allowed_hour(stem):
    """stem 'YYYYMMDD_HH' -> True iff the whole hour is on a consumed day."""
    need(isinstance(stem, str) and len(stem) == 11 and stem[8] == '_' and stem[:8].isdigit() and stem[9:].isdigit(), f'bad hour stem {stem!r}')
    day, h = stem[:8], int(stem[9:])
    return (day in ALLOWED_DAYS and 0 <= h <= 23) or (day == NOV8 and 0 <= h <= NOV8_LAST_HOUR)


def hour_path(family, stem):
    """The raw hour file of a family ('rest' | 'predict_ws' | 'events'); refuses BEFORE any filesystem access."""
    if family not in ('rest', 'predict_ws', 'events'):
        raise Forbidden(f'unknown raw family {family!r}')
    if not allowed_hour(stem):
        raise Forbidden(f'hour {stem} is not on a consumed day (never opened)')
    return PF_RUN / family / f'{stem}.jsonl.gz'


def day_hours(day, extra_before=0, extra_after=0):
    """Hour stems for a UTC day with padding, CLIPPED to allowed hours -> (stems, clipped). The day itself must be allowed."""
    need(day in ALLOWED_DAYS or day == NOV8, f'day {day} is not a consumed day')
    d0 = dt.datetime.strptime(day, '%Y%m%d').replace(tzinfo=dt.timezone.utc)
    stems = [(d0 + dt.timedelta(hours=h)).strftime('%Y%m%d_%H') for h in range(-extra_before, 24 + extra_after)]
    return [s for s in stems if allowed_hour(s)], [s for s in stems if not allowed_hour(s)]


def uuid7_ms(sid):
    """ms of a UUIDv7 settlementId (FINDINGS C23: id-CREATION time; bounds the match from ABOVE only)."""
    u = uuid.UUID(sid)
    need(u.version == 7, f'settlementId {sid} is not a UUIDv7')
    return u.int >> 80


def yes_space(quote, index_set, price):
    """(YES-space side of the resting/aggressing order, YES-space price): YES bid p -> bid p; YES ask p -> ask p; NO bid q -> ask 1-q;
    NO ask q -> bid 1-q (pf_io.yes_side). indexSet 1 = Up = YES."""
    need(quote in ('Bid', 'Ask') and index_set in (1, 2), f'bad leg quote/indexSet {quote!r}/{index_set!r}')
    bid = quote == 'Bid'
    if index_set == 1:
        return ('bid' if bid else 'ask'), round(price, 6)
    return ('ask' if bid else 'bid'), round(1.0 - price, 6)


# ---- inference classes (per field / event; coverage_amendment.md step 4) ----
OBSERVED, CONSTRAINED, SENSITIVE, UNRESOLVED = 'observed', 'constrained', 'assumption_sensitive', 'unresolved'
CLASSES = (OBSERVED, CONSTRAINED, SENSITIVE, UNRESOLVED)

# ---- removal explanations (plan.md step 3; kept as a SET of possible explanations per order) ----
REMOVAL = ('exhausted', 'cancelled_or_expired', 'end_censored', 'missing_evidence')

# ---- declared scenarios (contract.md section 5); never chosen after seeing outcomes ----
SCENARIOS = {
    'S_FIFO':  dict(priority='price_time', fill_clock='hi', note='price-time priority (amendment 1: makers[] order is NOT evidence; exhaustion by price priority across a record\'s prices); fill at the upper end of its window'),
    'S_FIFO_LO': dict(priority='price_time', fill_clock='lo', note='as S_FIFO with each fill at the lower end of its window (lo unbounded -> hi minus the declared max lag)'),
    'S_NOPRIO': dict(priority='none', fill_clock='hi', note='no priority rule: only quantity / visibility constraints'),
}
MAX_LAG_MS = 2000   # S_FIFO_LO fallback when a fill window has no lower end (declared, not fitted)

# ---- table schemas: column -> dtype (parquet / pandas). Builders must write exactly these columns (extra columns allowed only
#      with the prefix 'x_'); mt_checks.py refuses a table whose required columns or dtypes differ. ----
SCHEMAS = {
    # one row per UNIQUE match record (canonical content; re-polls de-duplicated)
    'records': dict(day='str', mid='int64', sid='str', tx='str', settle_ms='int64', exec_ms='int64', first_recv_ns='int64', n_recv='int64',
                    amount_filled='float64', price_executed='float64', n_makers='int64',
                    tk_hash='str', tk_signer='str', tk_quote='str', tk_index_set='int64', tk_price='float64', tk_amount='float64',
                    tk_fee='float64', tk_fee_type='str', tk_y_side='str', status='str'),
    # one row per maker leg of a unique record
    'legs': dict(day='str', mid='int64', sid='str', tx='str', settle_ms='int64', exec_ms='int64', first_recv_ns='int64',
                 mk_idx='int64', n_makers='int64', mk_hash='str', mk_signer='str', mk_quote='str', mk_index_set='int64',
                 mk_price='float64', shares='float64', mk_fee='float64', mk_fee_type='str', y_side='str', y_price='float64', y_ptick='int64',
                 tk_hash='str', tk_signer='str', tk_y_side='str', phase='str', t_rel_s='float64', status='str'),
    # one row per book frame, every market (top of book only)
    'tob': dict(day='str', mid='int64', upd_ms='int64', recv_ns='int64', conn='str', order_count='int64', n_bid='int64', n_ask='int64',
                best_bid='float64', best_bid_sh='float64', best_ask='float64', best_ask_sh='float64', pend_bid_usd='float64',
                pend_ask_usd='float64', status='str'),
    # full depth for the book-supported and ground-truth markets: frames + long levels
    'frames': dict(mid='int64', fid='int64', upd_ms='int64', recv_ns='int64', conn='str', order_count='int64', pend_bids='str', pend_asks='str',
                   los_id='str', los_side='str', los_price='float64', los_outcome='str', status='str'),
    'levels': dict(mid='int64', fid='int64', side='str', ptick='int64', shares='float64', status='str'),   # status added, amendment 1
    # per-fill timing (lifecycle step 1); window [lo_ms, hi_ms]; lo_ms = -1 when unbounded
    'fills': dict(mid='int64', sid='str', mk_hash='str', mk_idx='int64', y_side='str', y_ptick='int64', shares='float64',
                  hi_ms='int64', lo_ms='int64', t_method='str', f_before='int64', f_after='int64', status='str'),
    # one row per maker order hash per market per scenario (lifecycle)
    'orders': dict(scenario='str', mid='int64', mk_hash='str', mk_signer='str', y_side='str', y_ptick='int64', n_fills='int64',
                   cum_shares='float64', first_hi_ms='int64', last_hi_ms='int64',
                   size_lb='float64', size_exact='float64', size_class='str',
                   post_lo_ms='int64', post_hi_ms='int64', post_class='str', post_method='str',
                   persist_lo_ms='int64', persist_hi_ms='int64', persist_class='str',
                   ahead_lo='float64', ahead_hi='float64', queue_class='str',
                   removal_set='str', removal_lo_ms='int64', removal_hi_ms='int64', removal_class='str', removal_method='str',
                   status='str'),
}
STATUS_OK = 'ok'
