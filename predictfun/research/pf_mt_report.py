"""MAKER-TRACE-RECONSTRUCTION-v1 report (P-2026-005; session aea5bfb9, 2026-11-22; contract.md section 4 'integration'). USER: "we figure
out how top makers make money first". Reads ONLY the builders' outputs under pf_mt_common.OUT (extract / lifecycle / econ / truth; never a
raw file, never the truth table itself) and writes the receipt tables RECEIPT/outputs/*.csv (*.csv.gz above LARGE_ROWS rows) and
readings.json, write-once: an existing outputs/ refuses before any input is read; everything is built in a private temp dir and published
at the end, readings.json last. Every reading and predicate is computed here from the tables (rule 10); the readings are the declaration's
R1-R4 rules (declaration.md), declared before any real output was read; synthetic proof in pf_mt_report_selftest.py.
RULES
 DAYS    only pf_mt_common's consumed days; every path is built from them (no directory is listed). Lifecycle markets of a day = its
         extract_<day>.json population.depth_mids (BOOK on Oct 24-27, own-fill GT windows on Nov 8), equal to econ.json book on BOOK days.
 INVENTORY  per day every source key set / leaf type / string format / counter / quantile / id fact of inventory_<day>.json and the
         envelope of extract_<day>.json as rows, plus computed predicates: amount reconciliation from the records_<day> status flags
         (recon_tk_amount / recon_mk_sum / malformed; recon_price = taker.price != priceExecuted), executedAt precision (fraction digits all
         zero), makers[] YES-price order rates over multi-price records, maker fee all zero, priceExecuted vs the maker-weighted average.
 STATUSES   per day x table x column the count of every status string and of every '+'-joined flag beside the table's rows. records / legs
         / markets / tob are recounted and must equal extract_<day>.json status_counts; fills_valued must equal econ.json counts (else
         refuse: a dropped or unreported row); depth frames / levels are carried from the extract receipt (source column).
 R1      T1 / T2 of scenario S_FIFO summed over every lifecycle market (BOOK + GT); rate = violations / denominator. Reading on the primary
         settle-order version: INSUFFICIENT if the denominator < 200, supported if rate <= 0.01, else contradicted; the window-certain
         version is a lower bound beside it (no reading). Coverage scenario = S_NOPRIO if any test is contradicted, else S_FIFO (literal
         declaration; INSUFFICIENT keeps S_FIFO and is reported as such). Windows / removal sets are written for every scenario with that
         one flagged primary; timelines and examples use it.
 R2      truth/v2/scores.json per scenario x field x test x class: n, failed, defects, sensitive misses, failed rate; totals recomputed and
         equal to the scores' defects_total (else refuse). A (field, class) with defects in any scenario is defective; reading no_defect iff
         no defect anywhere (the fills block included). Clock premises (scores clock_band.premises) and the outcome-selection caveat are
         fields. Defective pairs of an R3 class field are DEMOTED: wherever an order class is used (coverage, unresolved_top, timelines,
         examples) its effective class is assumption_sensitive and a computed demoted flag is set; the input class stays in raw_* columns.
 R3      BOOK markets. Sets: exante_sh / exante_pnl = identities.json ex-ante top 10 (Oct 23 maker shares / P&L); field = makers of BOOK
         orders in neither ex-ante set; all = every maker; sameday_sh / sameday_pnl = each day's own top 10 (descriptive). An order's
         classes are its S_FIFO row's (identical across its scenario rows, else refuse); executions = its fills rows (== n_fills, checked),
         shares = their nansum; recomputed per market equal to the lifecycle summary coverage (else refuse). Fractions per set x day (+
         pooled) x YES side (+ both) x field x class over orders / executions / shares, the denominator n beside every fraction.
 R4      econ maker_market (FILL rows): per set x day groups in / field (the day's other makers: maker_day sh_mk > 0) / nonmaker (econ's
         rule); totals and maker-role sums reconcile with econ segments dim 'all' (else refuse); round trip / residual over res_status ok
         markets (gross = rt_pnl + resid_res checked), drift / resolution over fifo_status ok markets (checked); cents per share over the
         shares each value covers; share_matched = rt_pnl / gross, share_spread = v_spr / gross; sign_consistent = net_res of the same strict
         sign on every measurement day; n_days = days with valid fills; ci 'none (n < 5 days)' (rule 8). share_matched uses econ's
         rt_pnl_res (round trip on resolved markets); gross = rt_pnl_res + resid_res is checked on the same population.
 TIMELINES  per BOOK day the BOOK market with the most MAKER legs (lifecycle fills rows) of the two top-ranked exante_sh addresses (ties:
         smallest mid): their fills_valued rows there (maker and taker roles) with their leg's lifecycle window, their order's lifecycle row
         (coverage scenario), the tob at lo (last frame upd_ms <= lo_ms, recv_ns order) and at hi (first frame upd_ms >= hi_ms) and the depth
         shown at the traded price on the resting side at f_before / f_after (absent level in a frame = 0). examples: per top-2 maker of that
         market, from its orders there: the 2 most-filled (n_fills desc), then 1 with >= 2 removal explanations, then 1 single-fill; ties and
         order by sha256(hash) ascending; distinct orders.
 UNRESOLVED the N_TOP BOOK orders with the largest |sum net_res| of their fills_valued maker rows (status ok, st_res in RES_VALID) whose effective
         post or removal class is unresolved / assumption_sensitive (ties: sha256(hash)); the open bounds named from the three scenario rows.
 ADDITIONS (integrator, 2026-11-22, before any real output): econ_address_day = every ex-ante member per day (+ pooled) from the same
         maker_market rows (maker / taker shares, maker_frac = sh_mk / (sh_mk + sh_tk), share of the day's maker volume, P&L, fees, reb25_res /
         reb25_fill, marks, rt_pnl_res vs residual drift / resolution, self-trade shares, ranks); counterparty = members x (member / field / self)
         from fills_valued addr x cp_addr, one_sided when >= ONE_SIDED_FRAC of the ok shares lie in one role or one YES direction with >=
         ONE_SIDED_MIN_N ok fills, member pairs must mirror; resolution_null = N_NULL Bernoulli(m_END) draws per set x group x day (seed
         declared), the actual resid_resol recomputed from fills_valued V / m_end; rank_persistence = ID-day vs measurement-day ranks;
         rebate_scenario = econ reb25 sums beside net_res (never added); rewards_market = econ's table with the unidentifiability statement;
         R1 readings carried into the S_FIFO-dependent segment dims; GT unique (order, field) defects = truth's unique_defects, recomputed
         from defect_rows. Resolved markets = res_status / st_res in econ's RES_VALID (V = 0.5 ties included). Declared names not present
         in an artifact (maker_market reb25_res / reb25_fill, maker_day rt_pnl_res, segment dims depth / sweep / cp / boost / self, GT
         size.ub / unique_defects / defect_rows.field, a defect category without a field) REFUSE with the reason: never guessed.
Not a strategy: descriptive tables; no fit, imitation, tradeable claim or verdict. Usage (run_light): pf_mt_report.py"""
import collections
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_mt_common as C  # noqa: E402
import pf_mt_econ as E  # noqa: E402   (output schemas FV / MM / MD / SEG, MET, VALS, HORIZONS, G_MIN: imported, never redefined)

need, OK, Counter = C.need, C.STATUS_OK, collections.Counter
DAYS = C.ALLOWED_DAYS + (C.NOV8,)
LIFE_DAYS = C.MEASURE_DAYS + (C.GT_DAY,)
SCEN = tuple(C.SCENARIOS)
PRIO = tuple(s for s in SCEN if C.SCENARIOS[s]['priority'] == 'price_time')
FIFO, NOPRIO = 'S_FIFO', 'S_NOPRIO'
need(FIFO in PRIO and C.SCENARIOS.get(NOPRIO, {}).get('priority') == 'none', 'pf_mt_common.SCENARIOS lacks S_FIFO / S_NOPRIO')
MIN_DEN, MAX_RATE = 200, 0.01                       # declaration R1
FIELDS = dict(post='post_class', persist='persist_class', queue='queue_class', removal='removal_class', size='size_class')
COV_SETS = ('exante_sh', 'exante_pnl', 'field', 'all', 'sameday_sh', 'sameday_pnl')
ECON_SETS = dict(exante_shares='exante_sh', exante_pnl='exante_pnl', sameday_shares='sameday_sh', sameday_pnl='sameday_pnl')
LABEL = dict(exante_sh='ex_ante', exante_pnl='ex_ante', field='field', all='all', sameday_sh='descriptive', sameday_pnl='descriptive')
GRPS = ('in', 'field', 'nonmaker')
QS = (0, 10, 25, 50, 75, 90, 100)
LARGE_ROWS, N_TOP, N_MOST = 20_000, 20, 2
OPEN = (C.UNRESOLVED, C.SENSITIVE)
TOL_ABS, TOL_REL = 1e-6, 1e-9                       # sums of the econ's per-row checked identities (row tolerance 1e-9 + 1e-12 x notional)
CAVEAT = ('outcome-selected: only filled orders print a public hash, so the scored population is our FILLED orders only; unfilled orders '
          'and their lifecycles are not scored (declaration R2)')
TRUTH_SCORES = 'truth/v2/scores.json'              # integrator 2026-11-22: the truth outputs moved to OUT/truth/v2 (per-run clock bands)
ECON_NEW_MM = ('reb25_res', 'reb25_fill')               # declared econ additions in maker_market: required; absent -> refuse, never guessed
ECON_NEW_MD = ('rt_pnl_res',)                           # ... and in maker_day (round trip on resolved markets, res_status in E.RES_VALID)
RES_VALID = tuple(E.RES_VALID)                          # resolved markets: V = 1 / 0, or 0.5 on a tie (econ's declared set)
SEG_NEW_DIMS = ('depth', 'sweep', 'cp', 'boost', 'self')  # declared econ segment dims: required in segments.parquet
S_FIFO_DIMS = ('life_exh', 'life_front')                  # segment dims that depend on S_FIFO claims (R1 flag carried into them)
N_NULL, SEED = 1000, 20261122                             # resolution-luck null: draws (>= 200, rule 6) and the declared seed
ONE_SIDED_FRAC, ONE_SIDED_MIN_N = 0.95, 10                # counterparty pair 'one_sided': >= 95 % of its ok shares in one direction, >= 10 ok fills
REWARDS_STATEMENT = ('per-maker rewards are unidentifiable from public data: the pool is per market (units unverified); BOOK eligible '
                     'share-seconds are an upper bound and the demonstrated set presence a lower bound; no reward is attributed to an address')
CAT_FIELD = dict(fills='fill', identity='identity', posting='post', size_lb='size', size_exact='size', size_ub='size', removal_exhausted='removal',
                 removal_membership='removal', removal_window='removal', persistence='persist', queue_own='queue', queue_other='queue')
GT_TESTS = (('post', 'posting', ('posting',)), ('size', 'size_lb', ('size', 'lb')), ('size', 'size_exact', ('size', 'exact')),
            ('size', 'size_ub', ('size', 'ub')),
            ('removal', 'exhausted_only', ('removal', 'exhausted_only')), ('removal', 'membership', ('removal', 'membership')),
            ('removal', 'window', ('removal', 'window')), ('persist', 'persistence', ('persistence',)),
            ('queue', 'own_front_claims', ('queue', 'own_front_claims')),
            ('queue', 'other_makers_front_claims', ('queue', 'other_makers_front_claims')))
BLOCK_STD = ('n', 'failed', 'defects', 'sensitive_misses', 'gap_ms_failed')
LINK_COLS = ['mid', 'mk_signer', 'y_side', 'kind', 'prev_hash', 'next_hash', 'prev_ptick', 'next_ptick', 'prev_last_hi_ms', 'next_first_hi_ms',
             'gap_ms', 'x_latest', 'label', 'status']
MK_REQ = ['day', 'mid', 'status', 'depth']
FV_ST = ['status', 't_status', 'st_res', 'st_spr'] + [f'st_{h}' for h, _ in E.HORIZONS] + ['st_end', 'life_status', 'kind', 'src_status']
FV_VAL = ['day', 'mid', 'sid', 'mk_idx', 'role', 'addr', 'hash', 'cp_addr', 'self_trade', 'y_side', 's', 'q', 'p', 'V', 'm_end', 't_ms', 't_status',
          'gross_res', 'fee_res', 'fee_fill', 'net_res', 'st_res', 'status'] + [f'v_{h}' for h in E.VALS]
MM_ST = ['status', 'res_status', 'end_status', 'fifo_status', 'rew_status']
MM_COLS = list(dict.fromkeys(['day', 'mid', 'addr', 'population', 'book'] + MM_ST + E.MM_SUM + E.FIFO + ['n_excl'] + list(ECON_NEW_MM)))
WINDOWS = (('post_window_ms', 'post_lo_ms', 'post_hi_ms'), ('removal_window_ms', 'removal_lo_ms', 'removal_hi_ms'),
           ('ahead_range_sh', 'ahead_lo', 'ahead_hi'), ('persist_span_ms', 'persist_lo_ms', 'persist_hi_ms'))
TOB_COLS = ['mid', 'upd_ms', 'recv_ns', 'best_bid', 'best_bid_sh', 'best_ask', 'best_ask_sh', 'status']
ORD_SHOW = ['n_fills', 'cum_shares', 'size_lb', 'size_exact', 'post_lo_ms', 'post_hi_ms', 'post_method', 'persist_lo_ms', 'persist_hi_ms',
            'ahead_lo', 'ahead_hi', 'removal_set', 'removal_lo_ms', 'removal_hi_ms', 'removal_method']


# ------------------------------------------------------------------ utilities ------------------------------------------------------------------
def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def hkey(h):
    """The declared tie-break key: sha256 of the hash string."""
    return hashlib.sha256(str(h).encode()).hexdigest()


def hwm():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) / 1024.
    return 0.


def guard(where):
    h = hwm()
    need(h < C.GUARD_MIB, f'memory self-guard: VmHWM {h:.0f} MiB >= {C.GUARD_MIB:.0f} MiB at {where}')
    return h


def rate(a, b):
    return float(a) / float(b) if b else None


def fnone(x):
    """A JSON-safe number: NaN / None -> None."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def qrow(x, prefix=''):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    d = {f'{prefix}p{q}': (float(np.percentile(x, q)) if x.size else None) for q in QS}
    d[f'{prefix}avg'] = float(x.mean()) if x.size else None          # never a column named like a pandas method (mean / rank / size ...)
    return d


def day_ok(d):
    need(d in DAYS, f'day {d!r} is not a consumed day (refused before any path is built)')
    return d


def flags_of(v):
    return str(v).split('+')


class Inputs:
    """Every file read, by its path relative to OUT: sha256 + bytes at first use; a missing file or required column refuses."""

    def __init__(self, root):
        self.root, self.files = Path(root), {}

    def path(self, rel, what):
        p = self.root / rel
        need(p.is_file(), f'input missing: {what} ({rel})')
        if rel not in self.files:
            self.files[rel] = dict(sha256=sha256(p), bytes=p.stat().st_size)
        return p

    def json(self, rel, what):
        return json.loads(self.path(rel, what).read_text())

    def schema(self, rel, what, required):
        p = self.path(rel, what)
        names = pq.read_schema(p).names
        miss = [c for c in required if c not in names]
        need(not miss, f'{what}: required columns {miss} missing ({rel})')
        return p

    def table(self, rel, what, cols, required=None, filters=None):
        p = self.schema(rel, what, list(required) if required is not None else list(cols))
        return pq.read_table(p, columns=list(cols), filters=filters).to_pandas()


def count_values(path, cols):
    """One streaming pass: rows and, per column, a Counter of str(value) ('nan' for null)."""
    cnt, n = {c: Counter() for c in cols}, 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=list(cols)):
        n += b.num_rows
        for c in cols:
            vc = pc.value_counts(b.column(c))
            for v, k in zip(vc.field('values').to_pylist(), vc.field('counts').to_pylist()):
                cnt[c]['nan' if v is None else str(v)] += int(k)
    return n, cnt


def status_rows(day, table, column, cnt, n_rows, part='', source='table'):
    rows, fl = [], Counter()
    for v, k in sorted(cnt.items()):
        rows.append(dict(day=day, table=table, part=part, column=column, kind='status', value=v, n=int(k), den=int(n_rows), rate=rate(k, n_rows),
                         source=source))
        for f in flags_of(v):
            fl[f] += int(k)
    rows += [dict(day=day, table=table, part=part, column=column, kind='flag', value=f, n=k, den=int(n_rows), rate=rate(k, n_rows), source=source)
             for f, k in sorted(fl.items())]
    return rows


# ------------------------------------------------------------------ inventory ------------------------------------------------------------------
def _flat(prefix, x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            _flat(f'{prefix}.{k}' if prefix else str(k), v, out)
    else:
        out.append((prefix, x))


def inventory_rows(day, V, X, rec):
    """inventory_<day>.json (+ the extract envelope and the records status flags) -> long rows (day, section, item, key, value, n, den, rate)."""
    R = []

    def add(section, item, key='', value=None, n=None, den=None):
        R.append(dict(day=day, section=section, item=item, key=key, value=value, n=n, den=den,
                      rate=(n / den if isinstance(n, (int, float)) and den else None)))
    for kind, lst in sorted((V.get('objects') or {}).items()):
        tot = sum(int(e['n']) for e in lst)
        for e in lst:
            add('source_keys', kind, '|'.join(e['keys']) if isinstance(e['keys'], list) else str(e['keys']), n=int(e['n']), den=tot)
    for path, lv in sorted((V.get('leaves') or {}).items()):
        for sec, key in (('leaf_type', 'types'), ('leaf_format', 'formats')):
            d = lv.get(key) or {}
            tot = sum(int(v) for v in d.values())
            for k, v in sorted(d.items()):
                add(sec, path, k, n=int(v), den=tot)
    cnt = V.get('counters') or {}
    for name, d in sorted(cnt.items()):
        tot = sum(int(v) for v in d.values())
        for k, v in sorted(d.items()):
            add('counter', name, k, n=int(v), den=tot)
    qsrc = [('rest.clocks_and_lags_ms', (V.get('rest') or {}).get('clocks_and_lags_ms') or {}), ('ws', V.get('ws') or {})]
    rl = (V.get('rest') or {}).get('receipt_lag_ms')
    if isinstance(rl, dict):
        qsrc.append(('rest', {'receipt_lag_ms': rl}))
    for sec, D in qsrc:
        for name, q in sorted(D.items()):
            if isinstance(q, dict):
                for k, v in q.items():
                    if k != 'n':
                        add('quantile', f'{sec}.{name}', k, value=v, n=int(q.get('n', 0)))
                if not q.get('n'):
                    add('quantile', f'{sec}.{name}', 'n', value=0, n=0)
    flat = []
    _flat('', (V.get('rest') or {}).get('ids') or {}, flat)
    for k, v in flat:
        add('ids', k, value=v)
    flat = []
    _flat('', {k: v for k, v in (X.get('envelope') or {}).items() if k != 'rule'}, flat)
    for k, v in flat:
        add('envelope', k, value=v)
    # ---- computed predicates ----
    n_rec, st = rec
    bad_amt = sum(k for v, k in st.items() if {'recon_tk_amount', 'recon_mk_sum', 'malformed'} & set(flags_of(v)))
    bad_px = sum(k for v, k in st.items() if 'recon_price' in flags_of(v))
    for f in ('recon_tk_amount', 'recon_mk_sum', 'recon_price', 'malformed'):
        add('records_flag', f, n=sum(k for v, k in st.items() if f in flags_of(v)), den=n_rec)
    add('predicate', 'amount_reconciles (taker.amount == amountFilled == sum makers.amount, ints)', value=bool(n_rec and bad_amt == 0) if n_rec else 'not_evaluable',
        n=n_rec - bad_amt, den=n_rec)
    add('predicate', 'taker_price_equals_priceExecuted', value=bool(bad_px == 0) if n_rec else 'not_evaluable', n=n_rec - bad_px, den=n_rec)
    ea = cnt.get('executedAt_after_seconds') or {}
    whole = sum(int(v) for k, v in ea.items() if re.fullmatch(r'(\.0+)?(Z|[+-]\d\d:\d\d)', k))
    frac = sum(int(v) for k, v in ea.items() if re.fullmatch(r'\.\d+(Z|[+-]\d\d:\d\d)', k) and not re.fullmatch(r'\.0+(Z|[+-]\d\d:\d\d)', k))
    tot = sum(int(v) for v in ea.values())
    add('predicate', 'executedAt_whole_second (fraction digits all zero)', value=bool(frac == 0 and whole == tot) if tot else 'not_evaluable',
        n=whole, den=tot)
    add('predicate', 'executedAt_nonzero_fraction', n=frac, den=tot)
    add('predicate', 'executedAt_unparsed', n=tot - whole - frac, den=tot)
    mo = cnt.get('makers_yes_price_order (multi-maker records)') or {}
    multi = sum(int(mo.get(k, 0)) for k in ('best_first', 'worst_first', 'neither'))
    for k in ('best_first', 'worst_first', 'neither'):
        add('predicate', f'makers_listing_{k} (multi-price records)', n=int(mo.get(k, 0)), den=multi)
    add('predicate', 'makers_listing_single_price (multi-maker records)', n=int(mo.get('single_price', 0)), den=multi + int(mo.get('single_price', 0)))
    add('predicate', 'makers_listing_monotone_all (no neither)', value=bool(int(mo.get('neither', 0)) == 0) if multi else 'not_evaluable',
        n=multi - int(mo.get('neither', 0)), den=multi)
    mf = cnt.get('maker_fee') or {}
    legs, nz = int(mf.get('legs', 0)), int(mf.get('nonzero', 0))
    add('predicate', 'maker_fee_all_zero', value=bool(nz == 0) if legs else 'not_evaluable', n=legs - nz, den=legs)
    add('predicate', 'maker_fee_types', value='|'.join(sorted(cnt.get('maker_fee_type') or {})))
    tk = sorted({t for k, d in cnt.items() if k.startswith('taker_fee_type | ') for t in d})
    add('predicate', 'taker_fee_types', value='|'.join(tk))
    pe = cnt.get('priceExecuted_vs_maker_avg') or {}
    pt_ = sum(int(v) for v in pe.values())
    for k in ('exact', 'within_rounding', 'differs'):
        add('predicate', f'priceExecuted_vs_maker_avg_{k}', n=int(pe.get(k, 0)), den=pt_)
    return R


# ------------------------------------------------------------------ R1 / R2 ------------------------------------------------------------------
def r1_reading(den, viol):
    """declaration R1: INSUFFICIENT if the denominator < 200; supported if rate <= 1 %; contradicted otherwise."""
    if den < MIN_DEN:
        return 'INSUFFICIENT'
    return 'supported' if viol / den <= MAX_RATE else 'contradicted'


def pt_rows(J, d, m, pop):
    rows = []
    P = J.get('priority_tests') or {}
    for s in PRIO:
        need(s in P, f'lifecycle summary {d} {m}: priority tests of {s} missing')
        for test, key, dk in (('T1', 'T1_price', 'makers'), ('T2', 'T2_time', 'groups')):
            T = P[s][key]
            den = int(T[dk])
            for ver, vk, rk in (('settle_order', 'violations', 'rate'), ('window_certain', 'violations_certain', 'rate_certain')):
                v, r = int(T[vk]), T.get(rk)
                need((den == 0 and r is None) or (den > 0 and r is not None and abs(float(r) - round(v / den, 6)) <= 1e-9),
                     f'lifecycle summary {d} {m} {s} {test}: reported {rk} {r} != {v} / {den}')
                rows.append(dict(scope='market', population=pop, day=d, mid=m, scenario=s, test=test, version=ver, den=den, violations=v,
                                 rate=rate(v, den), t1_ties=T.get('ties') if test == 'T1' else None,
                                 t2_groups_tie=T.get('groups_tie') if test == 'T2' else None,
                                 t2_groups_invalid=T.get('groups_invalid') if test == 'T2' else None,
                                 t2_survivor_groups=T.get('survivor_groups') if test == 'T2' else None,
                                 t2_no_survivor_groups=T.get('no_survivor_groups') if test == 'T2' else None))
    return rows


def r1(PT):
    """Pooled priority tests (BOOK / GT / ALL) + the reading on the primary (ALL, S_FIFO, settle order)."""
    need(PT, 'R1: no lifecycle market (no priority test to pool)')
    D = pd.DataFrame(PT)
    nmk = lambda x: int(len(x[['day', 'mid']].drop_duplicates()))  # noqa: E731
    rows = []
    sums = ['den', 'violations', 't1_ties', 't2_groups_tie', 't2_groups_invalid', 't2_survivor_groups', 't2_no_survivor_groups']
    for scope, sel in (('pooled_BOOK', D.population == 'BOOK'), ('pooled_GT', D.population == 'GT'), ('pooled_ALL', D.population.notna())):
        for (s, t, v), g in D[sel].groupby(['scenario', 'test', 'version'], sort=True):
            r = {k: (float(g[k].sum()) if g[k].notna().any() else None) for k in sums}
            den, viol = int(r['den']), int(r['violations'])
            rows.append(dict(scope=scope, population=scope.split('_')[1], day='pooled', mid=-1, scenario=s, test=t, version=v, den=den, violations=viol,
                             rate=rate(viol, den), n_markets=nmk(g),
                             **{k: (int(r[k]) if r[k] is not None else None) for k in sums[2:]},
                             reading=r1_reading(den, viol) if v == 'settle_order' else '',
                             role='primary' if (scope == 'pooled_ALL' and s == FIFO and v == 'settle_order') else
                             ('lower_bound_beside_primary' if v == 'window_certain' else 'beside')))
    P = pd.DataFrame(rows)
    out = {}
    for t in ('T1', 'T2'):
        a = P[(P.scope == 'pooled_ALL') & (P.scenario == FIFO) & (P.test == t)].set_index('version')
        need(len(a) == 2, f'R1: pooled {t} rows missing (no lifecycle market?)')
        so, wc = a.loc['settle_order'], a.loc['window_certain']
        out[t] = dict(den=int(so.den), violations=int(so.violations), rate=fnone(so.rate), violations_certain=int(wc.violations), rate_certain=fnone(wc.rate),
                      reading=so.reading, n_markets=int(so.n_markets), unit='T1: S_FIFO price-priority exhaustion claims (strictly-better makers); '
                      'violation = the hash fills again later' if t == 'T1' else 'T2: records with >= 2 worst-price makers; violation = >= 2 of them '
                      'fill again later')
    lo = P[P.scenario.isin(PRIO)].pivot_table(index=['scope', 'test', 'version'], columns='scenario', values=['den', 'violations'], aggfunc='sum')
    same = bool(all((lo[(k, s)] == lo[(k, FIFO)]).all() for k in ('den', 'violations') for s in PRIO))
    contra = any(out[t]['reading'] == 'contradicted' for t in out)
    out.update(primary=dict(scope='pooled_ALL (BOOK + GT)', scenario=FIFO, version='settle_order (window_certain = lower bound beside it)'),
               rule=f'INSUFFICIENT if den < {MIN_DEN}; supported if rate <= {MAX_RATE}; contradicted otherwise',
               any_contradicted=contra, coverage_scenario=NOPRIO if contra else FIFO,
               coverage_scenario_rule='S_NOPRIO if any test is contradicted, else S_FIFO (declaration R1)',
               prio_scenarios_equal=same, n_markets=dict(BOOK=nmk(D[D.population == 'BOOK']), GT=nmk(D[D.population == 'GT'])))
    return P, out


def premises_of(x, out):
    """Every 'premises' list under the scores' clock band(s) (one band or per-run bands, truth v2), unique, in first-seen order."""
    if isinstance(x, dict):
        for k, v in x.items():
            if k == 'premises' and isinstance(v, list):
                out += [str(p) for p in v if str(p) not in out]
            else:
                premises_of(v, out)
    elif isinstance(x, list):
        for v in x:
            premises_of(v, out)
    return out


def gt(S):
    """truth/v2/scores.json -> (gt_scores rows, R2 reading, demote {field: {class}})."""
    need(isinstance(S.get('scenarios'), dict) and S['scenarios'], 'GT scores: no scenario block')
    prem = premises_of({k: v for k, v in S.items() if k.startswith('clock_band')}, [])
    need(prem, 'GT scores: no clock premise under clock_band(s) (the clock premise must be stated with every score)')
    premise = ' | '.join(prem)
    pop = S.get('population') or {}
    rows = []

    def add(scen, field, test, cls, blk):
        n, fl, de, sm = (int(blk.get(k, 0) or 0) for k in ('n', 'failed', 'defects', 'sensitive_misses'))
        rows.append(dict(scenario=scen, field=field, test=test, cls=cls, n=n, failed=fl, defects=de, sensitive_misses=sm, rate_failed=rate(fl, n),
                         sensitive_miss_rate=rate(sm, n) if cls == C.SENSITIVE else None,
                         outcomes=json.dumps({k: v for k, v in blk.items() if k not in BLOCK_STD}, sort_keys=True),
                         gap_ms_failed=json.dumps(blk.get('gap_ms_failed')) if blk.get('gap_ms_failed') is not None else '',
                         reading='no_defect' if de == 0 else 'defective', clock_premise=premise, caveat=CAVEAT,
                         scored_own_orders=pop.get('scored_own_orders'), scored_legs=pop.get('scored_legs'), scored_shares=pop.get('scored_shares')))

    def classes(blk, where):
        bad = set(blk) - set(C.CLASSES)
        need(not bad, f'GT scores {where}: unknown class {sorted(bad)} (outside pf_mt_common.CLASSES)')
        return blk
    fb = S.get('fills') or {}
    for c, b in classes(fb.get('window') or {}, 'fills.window').items():
        add('fills', 'fill', 'fill_window', c, b)
    add('fills', 'fill', 'fills_total (phantom + identity + window)', C.OBSERVED,
        dict(n=(fb.get('executions') or {}).get('n', 0), failed=fb.get('defects', 0), defects=fb.get('defects', 0)))
    totals = dict(fills=int(fb.get('defects', 0)))
    for scen, R in S['scenarios'].items():
        need(scen in C.SCENARIOS, f'GT scores: unknown scenario {scen!r}')
        idn = R['identity']
        add(scen, 'identity', 'identity', C.OBSERVED, dict(n=idn.get('n', 0), failed=idn.get('defects', 0), defects=idn.get('defects', 0)))
        for field, test, path in GT_TESTS:
            blk = R
            for k in path:
                need(isinstance(blk, dict) and k in blk, f'GT scores {scen}: declared block {".".join(path)} absent (refusing rather than guessing)')
                blk = blk[k]
            for c in C.CLASSES:
                add(scen, field, test, c, classes(blk, f'{scen}.{".".join(path)}').get(c, dict(n=0)))
        mine = sum(r['defects'] for r in rows if r['scenario'] == scen)
        need(mine == int(R['defects_total']), f'GT scores {scen}: per-class defects sum {mine} != defects_total {R["defects_total"]}')
        totals[scen] = mine
    need({k: int(v) for k, v in (S.get('defects_total') or {}).items()} == totals,
         f'GT scores: defects_total {S.get("defects_total")} != the recomputed {totals}')
    DR = S.get('defect_rows')
    need(isinstance(DR, list) and len(DR) == sum(totals.values()), f'GT scores: defect_rows missing or not one per counted defect ({sum(totals.values())})')
    badc = sorted({str(r.get('category')) for r in DR} - set(CAT_FIELD))
    need(not badc, f'GT scores: defect categories {badc} have no declared field mapping (refusing rather than guessing)')
    need(all('field' in r for r in DR), 'GT scores: a defect row without its field (truth v2 declares one per row)')
    UT = S.get('unique_defects')
    need(isinstance(UT, dict) and 'n' in UT and isinstance(UT.get('by_field'), dict), 'GT scores: unique_defects (n, by_field) absent (truth v2 declares it)')
    UQ = {}
    for r in DR:                                                   # the same (order, field) defect counted once across scenarios
        UQ.setdefault((int(r['mid']), str(r['mk_hash']), str(r['field'])), [CAT_FIELD[r['category']], set()])[1].add(str(r['cls']))
    byf = dict(sorted(Counter(f for _, _, f in UQ).items()))
    need(len(UQ) == int(UT['n']) and byf == {str(k): int(v) for k, v in UT['by_field'].items()},
         f'GT scores: unique_defects {UT} do not recompute from defect_rows (n {len(UQ)}, by_field {byf})')
    ucnt = Counter((v[0], f, '|'.join(sorted(v[1]))) for (_, _, f), v in UQ.items())
    for (f, tf, c), k in sorted(ucnt.items()):
        rows.append(dict(scenario='unique_across_scenarios', field=f, test=f'unique (order, field) defects: {tf}', cls=c, n=None, failed=k, defects=k,
                         sensitive_misses=0, rate_failed=None, sensitive_miss_rate=None, outcomes='', gap_ms_failed='', reading='defective',
                         clock_premise=premise, caveat=CAVEAT, scored_own_orders=pop.get('scored_own_orders'), scored_legs=pop.get('scored_legs'),
                         scored_shares=pop.get('scored_shares')))
    G = pd.DataFrame(rows)
    bad = G[(G.defects > 0) & (G.scenario != 'unique_across_scenarios')]
    pairs = sorted({(f, c) for f, c in zip(bad.field, bad.cls)})
    demote = {}
    for f, c in pairs:
        if f in FIELDS and c != C.SENSITIVE:
            demote.setdefault(f, set()).add(c)
    tot = sum(totals.values())
    R2 = dict(reading='no_defect' if tot == 0 else 'defective', defects_total=tot, defects_by_scope=totals,
              unique_defects=dict(total=len(UQ), by_field=byf, by_r3_field=dict(sorted(Counter(v[0] for v in UQ.values()).items())),
                                  rule='truth v2 unique_defects (distinct (mid, order hash, defect field) over every scenario), recomputed from defect_rows'),
              defective_pairs=[dict(field=f, cls=c, r3_column=FIELDS.get(f), demoted=bool(f in demote and c in demote[f])) for f, c in pairs],
              sensitive_misses={s: int(G[(G.scenario == s) & (G.cls == C.SENSITIVE)].sensitive_misses.sum()) for s in S['scenarios']},
              sensitive_n={s: int(G[(G.scenario == s) & (G.cls == C.SENSITIVE)].n.sum()) for s in S['scenarios']},
              n=dict(scored_own_orders=pop.get('scored_own_orders'), scored_legs=pop.get('scored_legs'), scored_shares=pop.get('scored_shares'),
                     truth_rows=pop.get('truth_rows'), lifecycle_markets=pop.get('lifecycle_markets')),
              coverage_of_filled_own_orders={s: R.get('coverage') for s, R in S['scenarios'].items()},
              clock_premise=premise, caveat=CAVEAT, asof=S.get('scored_utc'),
              rule='a defect = a failed observed / constrained claim; no_defect iff 0 defects (fills block included); each defective (field, class) of '
                   'an R3 class column is demoted to assumption_sensitive (computed flag; inputs never edited)')
    return G, R2, demote


def effective(raw, field, demote):
    raw = np.asarray(raw, dtype=object)
    dem = np.isin(raw, sorted(demote.get(field, ())))
    return np.where(dem, C.SENSITIVE, raw), dem


# ------------------------------------------------------------------ identities ------------------------------------------------------------------
def members(ID):
    """identities.json -> {set: frozenset} (ex ante) / {set: {day: frozenset}} (same day), ranks kept for the top-2 pick."""
    ex = ID.get('ex_ante') or {}
    need('top10_shares' in ex and 'top10_pnl' in ex, 'identities.json: ex-ante top10_shares / top10_pnl missing')
    rk = lambda L: [x['addr'] for x in sorted(L, key=lambda x: int(x['rank']))]  # noqa: E731
    M = dict(exante_sh=frozenset(rk(ex['top10_shares'])), exante_pnl=frozenset(rk(ex['top10_pnl'])), sameday_sh={}, sameday_pnl={})
    for d, v in (ID.get('same_day') or {}).items():
        need(d in C.MEASURE_DAYS, f'identities.json: same-day set of {d!r} (not a measurement day)')
        M['sameday_sh'][d], M['sameday_pnl'][d] = frozenset(rk(v['top10_shares'])), frozenset(rk(v['top10_pnl']))
    return M, rk(ex['top10_shares'])


def set_mask(name, addr, day, M):
    addr, day = pd.Series(np.asarray(addr, dtype=object)), np.asarray(day, dtype=object)
    if name == 'all':
        return np.ones(len(addr), bool)
    if name == 'field':
        return ~(addr.isin(M['exante_sh']) | addr.isin(M['exante_pnl'])).to_numpy(bool)
    if name in ('exante_sh', 'exante_pnl'):
        return addr.isin(M[name]).to_numpy(bool)
    m = np.zeros(len(addr), bool)
    for d, mem in M[name].items():
        m |= (day == d) & addr.isin(mem).to_numpy(bool)
    return m


def case_check(addrs, M, what):
    mem = {a for k in ('exante_sh', 'exante_pnl') for a in M[k]} | {a for k in ('sameday_sh', 'sameday_pnl') for v in M[k].values() for a in v}
    low = {a.lower(): a for a in mem}
    clash = sorted({a for a in addrs if isinstance(a, str) and a.lower() in low and a != low[a.lower()]})
    need(not clash, f'{what}: {len(clash)} addresses differ from an identity-set member only by case')


# ------------------------------------------------------------------ R3 tables ------------------------------------------------------------------
def coverage(O1, M, demote, sides):
    """O1: the S_FIFO order rows after with_classes (col = effective class, raw_<col> = the lifecycle's class)."""
    rows = []
    days = C.MEASURE_DAYS + ('pooled',)
    for name in COV_SETS:
        S = O1[set_mask(name, O1.mk_signer, O1.day, M)]
        for f, col in FIELDS.items():
            B = pd.DataFrame(dict(day=S.day.to_numpy(object), side=S.y_side.to_numpy(object), eff=S[col].to_numpy(object),
                                  raw=S[f'raw_{col}'].to_numpy(object), o=1, x=S.n_exec.to_numpy(np.int64), sh=S.sh_exec.to_numpy(float)))
            B4 = pd.concat([B, B.assign(day='pooled'), B.assign(side='both'), B.assign(day='pooled', side='both')], ignore_index=True)
            tod = lambda g: {k: tuple(v) for k, v in zip(g.index, g.to_numpy())}  # noqa: E731
            ge, gr = tod(B4.groupby(['day', 'side', 'eff'])[['o', 'x', 'sh']].sum()), tod(B4.groupby(['day', 'side', 'raw'])[['o', 'x', 'sh']].sum())
            gd = tod(B4.groupby(['day', 'side'])[['o', 'x', 'sh']].sum())
            gm = tod(B4[B4.eff != B4.raw].groupby(['day', 'side', 'eff'])[['o', 'x', 'sh']].sum())
            for d_ in days:
                for s_ in sides + ('both',):
                    den = gd.get((d_, s_), (0, 0, 0.))
                    for c in C.CLASSES:
                        e, r, mv = ge.get((d_, s_, c), (0, 0, 0.)), gr.get((d_, s_, c), (0, 0, 0.)), gm.get((d_, s_, c), (0, 0, 0.))
                        rows.append(dict(id_set=name, label=LABEL[name], day=d_, side=s_, field=f, cls=c, n_orders=int(e[0]), n_exec=int(e[1]),
                                         shares=float(e[2]), den_orders=int(den[0]), den_exec=int(den[1]), den_shares=float(den[2]),
                                         frac_orders=rate(e[0], den[0]), frac_exec=rate(e[1], den[1]), frac_shares=rate(e[2], den[2]),
                                         raw_n_orders=int(r[0]), raw_n_exec=int(r[1]), raw_shares=float(r[2]),
                                         demoted=bool(c in demote.get(f, ())), demoted_in_orders=int(mv[0]), demoted_in_exec=int(mv[1]),
                                         demoted_in_shares=float(mv[2])))
    return pd.DataFrame(rows)


def windows(OA, M, cov_scen):
    rows = []
    for name in COV_SETS:
        S0 = OA[set_mask(name, OA.mk_signer, OA.day, M)]
        for s in SCEN:
            S1 = S0[S0.scenario == s]
            for d_ in C.MEASURE_DAYS + ('pooled',):
                S = S1 if d_ == 'pooled' else S1[S1.day == d_]
                for met, lc, hc in WINDOWS:
                    lo, hi = S[lc].to_numpy(float), S[hc].to_numpy(float)
                    lo_o, hi_o = (np.isnan(lo), np.isnan(hi)) if met == 'ahead_range_sh' else (lo < 0, hi < 0)
                    b = ~lo_o & ~hi_o
                    w = hi[b] - lo[b]
                    rows.append(dict(id_set=name, label=LABEL[name], scenario=s, primary=bool(s == cov_scen), day=d_, metric=met,
                                     unit='shares' if met == 'ahead_range_sh' else 'ms', n_orders=int(len(S)), n_bounded=int(b.sum()),
                                     n_lo_open=int((lo_o & ~hi_o).sum()), n_hi_open=int((~lo_o & hi_o).sum()), n_both_open=int((lo_o & hi_o).sum()),
                                     n_negative=int((w < 0).sum()), **qrow(w)))
    return pd.DataFrame(rows)


def removal_sets(OA, M, cov_scen):
    rows = []
    for name in COV_SETS:
        S0 = OA[set_mask(name, OA.mk_signer, OA.day, M)]
        for s in SCEN:
            S1 = S0[S0.scenario == s]
            for d_ in C.MEASURE_DAYS + ('pooled',):
                S = S1 if d_ == 'pooled' else S1[S1.day == d_]
                g = S.groupby('removal_set').agg(n_orders=('mk_hash', 'size'), n_exec=('n_exec', 'sum'), shares=('sh_exec', 'sum'))
                for rs, r in g.iterrows():
                    rows.append(dict(id_set=name, label=LABEL[name], scenario=s, primary=bool(s == cov_scen), day=d_, removal_set=rs,
                                     n_members=len([x for x in str(rs).split('|') if x]), n_orders=int(r.n_orders), n_exec=int(r.n_exec),
                                     shares=float(r.shares), den_orders=int(len(S)), frac_orders=rate(r.n_orders, len(S))))
    return pd.DataFrame(rows, columns=['id_set', 'label', 'scenario', 'primary', 'day', 'removal_set', 'n_members', 'n_orders', 'n_exec', 'shares',
                                       'den_orders', 'frac_orders'])


def links(LK, O1, M):
    rows = []
    for name in COV_SETS:
        Ls, Os = LK[set_mask(name, LK.mk_signer, LK.day, M)], O1[set_mask(name, O1.mk_signer, O1.day, M)]
        for d_ in C.MEASURE_DAYS + ('pooled',):
            Ld = Ls if d_ == 'pooled' else Ls[Ls.day == d_]
            den = int(len(Os if d_ == 'pooled' else Os[Os.day == d_]))
            for kind in ('refill', 'reprice'):
                X = Ld[Ld.kind == kind]
                ok = (X.status == OK).to_numpy(bool)
                rows.append(dict(id_set=name, label=LABEL[name], day=d_, kind=kind, link_label='descriptive (unfilled orders are invisible)',
                                 n_links=int(len(X)), n_ok=int(ok.sum()), n_x_latest=int(X.x_latest.astype(bool).sum()), den_orders=den,
                                 links_per_order=rate(len(X), den), **qrow(X.gap_ms.to_numpy(float)[ok], 'gap_ms_')))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ R4 ------------------------------------------------------------------
def econ_frame(MM):
    """maker_market FILL rows -> (X: additive per-row values, K: row keys day / mid / addr / fifo_ok, rows without an address). Resolved =
    res_status in RES_VALID (ties included): round trip rt_pnl_res (== econ maker_day rt_pnl_res, check_rt) and residual resid_res there;
    drift / resolution over fifo_status ok markets."""
    mm = MM[MM.population == 'FILL']
    need(set(mm.day) <= set(C.MEASURE_DAYS), f'maker_market: FILL rows of days {sorted(set(mm.day) - set(C.MEASURE_DAYS))} (not measurement days)')
    n_noaddr = int(mm.addr.isna().sum())
    mm = mm[mm.addr.notna()].reset_index(drop=True)
    resok, fok = mm.res_status.isin(RES_VALID).to_numpy(bool), (mm.fifo_status == OK).to_numpy(bool)
    nz = lambda c: np.nan_to_num(mm[c].to_numpy(float))  # noqa: E731
    X = pd.DataFrame({c: nz(c) for c in dict.fromkeys(E.MM_SUM + ['n_excl'] + list(ECON_NEW_MM))})
    X['rt_sh'], X['rt_pnl_res'] = np.where(resok, nz('rt_sh'), 0.), np.where(resok, nz('rt_pnl'), 0.)   # == maker_day rt_pnl_res (checked)
    X['rt_pnl_unresolved_mkts'] = np.where(resok, 0., nz('rt_pnl'))
    X['resid_res'] = np.where(resok, nz('resid_res'), 0.)
    X['resid_res_fifo_ok'], X['resid_drift'], X['resid_resol'] = (np.where(fok, nz(c), 0.) for c in ('resid_res', 'resid_drift', 'resid_resol'))
    X['net_pos'] = nz('net_pos')
    X['n_rows'], X['n_rows_res_ok'], X['n_rows_fifo_ok'] = 1, resok.astype(np.int64), fok.astype(np.int64)
    X['n_rows_status_ok'] = (mm.status == OK).to_numpy(np.int64)
    K = pd.DataFrame(dict(day=mm.day.to_numpy(object), mid=mm.mid.to_numpy(np.int64), addr=mm.addr.to_numpy(object), fifo_ok=fok))
    return X, K, n_noaddr


def check_rt(X, K, MD):
    """econ's maker_day rt_pnl_res (per address x day) must equal the same restriction summed here from maker_market (else refuse)."""
    mine = pd.DataFrame(dict(day=K.day.to_numpy(object), addr=K.addr.to_numpy(object), rt=X.rt_pnl_res.to_numpy(float))).groupby(['day', 'addr']).rt.sum()
    md = MD[MD.population == 'FILL'].groupby(['day', 'addr']).rt_pnl_res.sum()
    keys = mine.index.union(md.index)
    a, b = mine.reindex(keys).fillna(0.).to_numpy(float), md.reindex(keys).fillna(0.).to_numpy(float)
    dev = np.abs(a - b)
    bad = dev > TOL_ABS + TOL_REL * np.maximum(1., np.abs(b))
    need(not bad.any(), f'maker_day rt_pnl_res differs from the maker_market round trip on resolved markets for {int(bad.sum())} address-days '
                        f'(max {float(dev.max()):.3g}): mismatched econ artifacts')
    return float(dev.max()) if len(dev) else 0.


def groups(K, nm, M, makers):
    """econ's comparison groups per row: in (the set's members) / field (the day's other makers) / nonmaker."""
    grp = np.full(len(K), 'nonmaker', dtype=object)
    a = K.addr
    for d in C.MEASURE_DAYS:
        sel = K.day.to_numpy(object) == d
        mem = M[nm] if nm.startswith('exante') else M[nm].get(d, frozenset())
        grp[sel & a.isin(makers[d]).to_numpy(bool)] = 'field'
        grp[sel & a.isin(mem).to_numpy(bool)] = 'in'
    return grp


def econ_tables(X, K, MD, SEG, M):
    """-> (econ_identity_day, R4 summary, checks)."""
    makers = {d: frozenset(MD.addr[(MD.day == d) & (MD.sh_mk > 0)]) for d in C.MEASURE_DAYS}
    num = list(X.columns)
    SA = SEG[(SEG.dim == 'all') & (SEG.bucket == 'all')]
    need(set(SEG.id_set) <= set(ECON_SETS), f'segments: unknown id_set {sorted(set(SEG.id_set) - set(ECON_SETS))}')
    seg_tot = SA.groupby(['id_set', 'grp', 'day'])[E.MET].sum()
    seg_mk = SA[SA.role == 'maker'].groupby(['id_set', 'grp', 'day'])[E.MET].sum()
    rows, maxdev, unrec = [], 0., set()
    for en, nm in ECON_SETS.items():
        G = X.assign(grp=groups(K, nm, M, makers), addr=K.addr.to_numpy(object), day=K.day.to_numpy(object))
        for d_ in C.MEASURE_DAYS + ('pooled',):
            Gd = G if d_ == 'pooled' else G[G.day == d_]
            S = Gd.groupby('grp')[num].sum()
            na = Gd.groupby('grp').addr.nunique()
            for g in GRPS:
                r = {c: float(S.at[g, c]) if g in S.index else 0. for c in num}
                o = derive(r)
                o.update(id_set=nm, label=LABEL[nm], grp=g, day=d_, n_addr=int(na.get(g, 0)))
                tol = TOL_ABS + TOL_REL * max(1., abs(o['notional']))
                dev = abs(o['gross_res'] - (o['rt_pnl_res'] + o['resid_res']))
                need(dev <= tol, f'econ {nm} {g} {d_}: gross {o["gross_res"]} != rt_pnl_res + resid_res ({dev:.3g}) on resolved markets')
                dev2 = abs(o['resid_res_fifo_ok'] - (o['resid_drift'] + o['resid_resol']))
                need(dev2 <= tol, f'econ {nm} {g} {d_}: resid_res {o["resid_res_fifo_ok"]} != drift + resol ({dev2:.3g})')
                if d_ != 'pooled':                                                 # reconciliation with the econ's own segments
                    k = (en, g, d_)
                    st = seg_tot.loc[k] if k in seg_tot.index else pd.Series(0., index=E.MET)
                    sm = seg_mk.loc[k] if k in seg_mk.index else pd.Series(0., index=E.MET)
                    unrec |= {c for c in E.MET if c not in o}
                    pairs = [(c, o[c], st[c]) for c in E.MET if c in o] + [(f'{c} (maker role)', r[mkc], sm[c]) for c, mkc in
                                                                           [('n', 'n_mk'), ('sh', 'sh_mk'), ('gross_res', 'gross_res_mk')] +
                                                                           [(f'{k_}_{h}', f'{k_}_{h}_mk') for h in E.VALS for k_ in ('v', 'sh')]]
                    for c, a, b in pairs:
                        dv = abs(float(a) - float(b))
                        maxdev = max(maxdev, dv)
                        need(dv <= tol, f'econ segments {en} {g} {d_}: {c} {b} != maker_market {a} (mismatched econ artifacts)')
                rows.append(o)
    D = pd.DataFrame(rows)
    R4 = {}
    for (nm, g), x in D[D.day != 'pooled'].groupby(['id_set', 'grp'], sort=True):
        x = x.set_index('day').reindex(list(C.MEASURE_DAYS))
        act = x.n.fillna(0) > 0
        nd = int(act.sum())
        net = x.net_res.fillna(0.)
        pos, neg = bool((net > 0).all()), bool((net < 0).all())
        R4.setdefault(nm, {})[g] = dict(n_days=nd, ci=ci_text(nd), sign_consistent=bool(nd == len(C.MEASURE_DAYS) and (pos or neg)),
                                        sign='+' if pos else '-' if neg else 'mixed_or_zero', net_res_by_day={d: float(v) for d, v in net.items()},
                                        share_matched_by_day={d: fnone(v) for d, v in x.share_matched.items()})
    for c in ('sign_consistent', 'sign', 'n_days', 'ci'):
        D[c] = [R4[a][g][c] for a, g in zip(D.id_set, D.grp)]
    for nm, gg in R4.items():
        for g, v in gg.items():
            p = D[(D.id_set == nm) & (D.grp == g) & (D.day == 'pooled')].iloc[0]
            v.update(share_matched_pooled=fnone(p.share_matched), share_spread_pooled=fnone(p.share_spread), net_res_pooled=float(p.net_res),
                     reb25_res_pooled=float(p.reb25_res), reb25_fill_pooled=float(p.reb25_fill), n_addr_pooled=int(p.n_addr), sh_pooled=float(p.sh),
                     label=LABEL[nm])
    return D, R4, dict(segments_max_abs_dev=maxdev, segments_metrics_not_in_maker_market=sorted(unrec), resolved=list(RES_VALID),
                       share_matched='rt_pnl_res / gross_res on resolved markets (res_status in RES_VALID; equal to econ maker_day rt_pnl_res, checked)')


def ci_text(nd):
    return f'none ({nd} < {E.G_MIN} days)' if nd < E.G_MIN else f'eligible ({nd} >= {E.G_MIN} days)'


def derive(r):
    o = dict(r)
    o['n'] = r['n_mk'] + r['n_tk']
    o['sh'] = r['sh_mk'] + r['sh_tk']
    o['gross_res'] = r['gross_res_mk'] + r['gross_res_tk']
    for h in E.VALS:
        o[f'v_{h}'], o[f'sh_{h}'] = r[f'v_{h}_mk'] + r[f'v_{h}_tk'], r[f'sh_{h}_mk'] + r[f'sh_{h}_tk']
    c = lambda a, b: 100. * a / b if b else None  # noqa: E731
    o.update(c_sh_gross=c(o['gross_res'], r['sh_res']), c_sh_fee=c(r['fee_res'], r['sh_res']), c_sh_net=c(r['net_res'], r['sh_res']),
             c_sh_fee_fill=c(r['fee_fill'], o['sh']), c_sh_rt=c(r['rt_pnl_res'], r['rt_sh']))
    for h in E.VALS:
        o[f'c_sh_{h}'], o[f'c_sh_{h}_mk'] = c(o[f'v_{h}'], o[f'sh_{h}']), c(r[f'v_{h}_mk'], r[f'sh_{h}_mk'])
    g = o['gross_res']
    o.update(share_matched=r['rt_pnl_res'] / g if g else None, share_resid=r['resid_res'] / g if g else None, share_spread=o['v_spr'] / g if g else None,
             maker_frac=rate(r['sh_mk'], r['sh_mk'] + r['sh_tk']))
    return o


def econ_segments(SEG, R1):
    miss = [d for d in SEG_NEW_DIMS if d not in set(SEG.dim)]
    need(not miss, f'econ segments: declared dims {miss} absent (econ additions not in this artifact): refusing rather than guessing')
    S = SEG.copy()
    S['id_set'] = S.id_set.map(ECON_SETS)
    dep = S.dim.isin(S_FIFO_DIMS).to_numpy(bool)
    S['s_fifo_dependent'] = dep
    S['r1_contradicted'] = dep & bool(R1['any_contradicted'])
    S['r1_readings'] = np.where(dep, f"T1={R1['T1']['reading']}; T2={R1['T2']['reading']}", '')
    nd = int(S.day.nunique())
    P = S.groupby(['id_set', 'label', 'grp', 'role', 'dim', 'bucket', 's_fifo_dependent', 'r1_contradicted', 'r1_readings'], sort=True)[E.MET].sum().reset_index()
    P = P.assign(day='pooled', n_addr=-1, ci=f'none_G{nd}_lt_{E.G_MIN}' if nd < E.G_MIN else f'eligible_G{nd}', status=OK)
    S = pd.concat([S, P[list(S.columns)]], ignore_index=True)
    tot = {k: float(v) for k, v in S[(S.dim == 'all') & (S.bucket == 'all')].groupby(['day', 'id_set', 'grp']).net_res.sum().items()}
    key = list(zip(S.day, S.id_set, S.grp))
    t = np.array([tot.get(k, np.nan) for k in key], dtype=float)
    S['set_net_res'] = t
    S['share_net'] = np.where(np.isfinite(t) & (t != 0), S.net_res.to_numpy(float) / np.where(t != 0, t, 1.), np.nan)
    S['dim_share_sum'] = S.groupby(['day', 'id_set', 'grp', 'dim']).share_net.transform('sum')
    c = lambda a, b: np.where(S[b].to_numpy(float) > 0, 100. * S[a].to_numpy(float) / np.where(S[b].to_numpy(float) > 0, S[b].to_numpy(float), 1.), np.nan)  # noqa: E731
    S['c_sh_gross'], S['c_sh_fee'], S['c_sh_net'] = c('gross_res', 'sh_res'), c('fee_res', 'sh_res'), c('net_res', 'sh_res')
    for h in E.VALS:
        S[f'c_sh_{h}'] = c(f'v_{h}', f'sh_{h}')
    S['n_days'], S['ci_days'] = nd, ci_text(nd)
    S['day_order'] = S.day.map({d: i for i, d in enumerate(C.MEASURE_DAYS + ('pooled',))})
    return S.sort_values(['id_set', 'grp', 'day_order', 'role', 'dim', 'bucket'], kind='mergesort').drop(columns='day_order').reset_index(drop=True)


def address_rows(X, K, MD, M, rk_ex, CP):
    """econ_address_day: every ex-ante member per measurement day (+ pooled), from the same maker_market rows as the sets."""
    memb = sorted(M['exante_sh'] | M['exante_pnl'])
    num = list(X.columns)
    G = X.assign(addr=K.addr.to_numpy(object), day=K.day.to_numpy(object))
    dtot = G.groupby('day').sh_mk.sum()
    ranks = {(d, a): (int(s_), int(p_)) for d, a, s_, p_ in zip(MD.day, MD.addr, MD.rank_sh, MD.rank_pnl)}
    selfq = CP[CP.cp == 'self'].groupby(['day', 'addr']).q_ok.sum() if len(CP) else pd.Series(dtype=float)
    rows = []
    for a in memb:
        Ga = G[G.addr == a]
        for d_ in C.MEASURE_DAYS + ('pooled',):
            x = Ga if d_ == 'pooled' else Ga[Ga.day == d_]
            o = derive({c: float(x[c].sum()) for c in num})
            tot = float(dtot.sum()) if d_ == 'pooled' else float(dtot.get(d_, 0.))
            sq = float(sum(selfq.get((d, a), 0.) for d in C.MEASURE_DAYS)) if d_ == 'pooled' else float(selfq.get((d_, a), 0.))
            rs = ranks.get((d_, a), (-1, -1))
            o.update(addr=a, day=d_, in_exante_sh=a in M['exante_sh'], rank_exante_sh=rk_ex['sh'].get(a, -1), in_exante_pnl=a in M['exante_pnl'],
                     rank_exante_pnl=rk_ex['pnl'].get(a, -1), n_markets=int(len(x)), share_of_day_maker_volume=rate(o['sh_mk'], tot),
                     self_trade_sh=sq, rank_sh_day=rs[0] if d_ != 'pooled' else None, rank_pnl_day=rs[1] if d_ != 'pooled' else None)
            rows.append(o)
    D = pd.DataFrame(rows)
    nd_, sc_ = {}, {}
    for a in memb:
        x = D[(D.addr == a) & (D.day != 'pooled')].set_index('day').reindex(list(C.MEASURE_DAYS))
        nd_[a] = int((x.n.fillna(0) > 0).sum())
        net = x.net_res.fillna(0.)
        sc_[a] = bool(nd_[a] == len(C.MEASURE_DAYS) and ((net > 0).all() or (net < 0).all()))
    if len(D):
        D['n_days'] = D.addr.map(nd_).astype(np.int64)
        D['ci'] = D.n_days.map(ci_text)
        D['sign_consistent'] = D.addr.map(sc_).astype(bool)
    first = ['addr', 'day', 'in_exante_sh', 'rank_exante_sh', 'in_exante_pnl', 'rank_exante_pnl', 'rank_sh_day', 'rank_pnl_day', 'n_markets', 'n_mk', 'sh_mk',
             'n_tk', 'sh_tk', 'maker_frac', 'share_of_day_maker_volume', 'notional', 'gross_res', 'fee_res', 'fee_fill', 'net_res', 'reb25_res', 'reb25_fill',
             'rt_pnl_res', 'resid_res', 'resid_drift', 'resid_resol', 'share_matched', 'self_trade_sh', 'sign_consistent', 'n_days', 'ci']
    return D[first + [c for c in D.columns if c not in first]] if len(D) else D


def counterparty(CP, memb):
    """Members x counterparty (another member, 'field' = every non-member, 'self') per day (+ pooled): ok shares / fills, net P&L to resolution
    (status and st_res ok), shares by role and by YES direction, the pair's share of the member's ok shares, one-sided flags."""
    if not len(CP):
        return pd.DataFrame(columns=['day', 'addr', 'cp', 'n_fills', 'n_ok', 'shares', 'net_res', 'role_direction', 'side_direction']), dict(pairs_checked=0)
    P = pd.concat([CP, CP.assign(day='pooled')], ignore_index=True)
    P['sh_mk'], P['sh_tk'] = np.where(P.role == 'maker', P.q_ok, 0.), np.where(P.role == 'maker', 0., P.q_ok)
    P['sh_buy'], P['sh_sell'] = np.where(P.s == 1, P.q_ok, 0.), np.where(P.s == 1, 0., P.q_ok)
    A = P.groupby(['day', 'addr', 'cp'], as_index=False)[['n', 'n_ok', 'n_res_ok', 'q_ok', 'net_ok', 'sh_mk', 'sh_tk', 'sh_buy', 'sh_sell']].sum()
    A = A.rename(columns={'q_ok': 'shares', 'net_ok': 'net_res', 'n': 'n_fills'})
    A['share_of_addr_shares'] = A.shares / A.groupby(['day', 'addr']).shares.transform('sum').replace(0., np.nan)

    def flag(a, b, tot, n):
        return np.where(n < ONE_SIDED_MIN_N, f'insufficient (n_ok < {ONE_SIDED_MIN_N})',
                        np.where(np.maximum(a, b) >= ONE_SIDED_FRAC * tot, 'one_sided', 'two_sided'))
    A['role_direction'] = flag(A.sh_mk, A.sh_tk, A.shares, A.n_ok)
    A['side_direction'] = flag(A.sh_buy, A.sh_sell, A.shares, A.n_ok)
    A['cp_kind'] = np.where(A.cp == 'field', 'field', np.where(A.cp == 'self', 'self', 'member'))
    pr = A[(A.day == 'pooled') & (A.cp_kind == 'member')].set_index(['addr', 'cp'])
    for (a, b), r in pr.iterrows():                                  # each leg is a row of both addresses: the pair's ok shares must mirror
        need((b, a) in pr.index and abs(pr.loc[(b, a)].shares - r.shares) <= 1e-9 * max(1., r.shares) and pr.loc[(b, a)].n_ok == r.n_ok,
             f'fills_valued: counterparty shares {a}->{b} do not mirror {b}->{a} (a dropped or unpaired leg)')
    A['day_order'] = A.day.map({d: i for i, d in enumerate(C.MEASURE_DAYS + ('pooled',))})
    A = A.sort_values(['day_order', 'addr', 'cp_kind', 'cp'], kind='mergesort').drop(columns='day_order').reset_index(drop=True)
    one = A[(A.day == 'pooled') & (A.cp_kind == 'member') & ((A.role_direction == 'one_sided') | (A.side_direction == 'one_sided'))]
    return A, dict(pairs_checked=int(len(pr)), one_sided_member_pairs_pooled=[dict(addr=a, cp=b, shares=float(s), role=r_, side=s_) for a, b, s, r_, s_ in
                                                                              zip(one.addr, one.cp, one.shares, one.role_direction, one.side_direction)],
                   rule=f'one_sided = >= {ONE_SIDED_FRAC} of the ok shares in one role (maker / taker) or one YES direction, with >= {ONE_SIDED_MIN_N} ok fills')


def resolution_null(X, K, marks, M, makers):
    """Resolution-luck null per set x group x day (+ pooled): S = sum over fifo-ok markets of NP_m (V_m - m_END_m), NP_m = the group's net
    position; null S* with V*_m ~ Bernoulli(m_END_m) independent per market, N_NULL draws, numpy default_rng([SEED, set, group, day]). The
    actual S (econ resid_resol) must equal the recomputation from fills_valued V / m_end (else refuse)."""
    rows = []
    days = C.MEASURE_DAYS + ('pooled',)
    for si, (en, nm) in enumerate(ECON_SETS.items()):
        grp = groups(K, nm, M, makers)
        for gi, g in enumerate(GRPS):
            for di, d_ in enumerate(days):
                sel = (grp == g) & K.fifo_ok.to_numpy(bool) & ((K.day.to_numpy(object) == d_) if d_ != 'pooled' else True)
                Z = pd.DataFrame(dict(day=K.day.to_numpy(object)[sel], mid=K.mid.to_numpy(np.int64)[sel], np_=X.net_pos.to_numpy(float)[sel],
                                      rr=X.resid_resol.to_numpy(float)[sel])).groupby(['day', 'mid'], as_index=False).sum()
                need(all(k in marks for k in zip(Z.day, Z.mid)), f'resolution null {nm} {g} {d_}: a fifo-ok market has no fills_valued END mark')
                me = np.array([marks[k][0] for k in zip(Z.day, Z.mid)], dtype=float)
                V = np.array([marks[k][1] for k in zip(Z.day, Z.mid)], dtype=float)
                npm, act = Z.np_.to_numpy(float), float(Z.rr.sum())
                need(np.isfinite(me).all() and np.isfinite(V).all(), f'resolution null {nm} {g} {d_}: a fifo-ok market without END mid / V')
                rec = float(np.sum(npm * (V - me)))
                tol = TOL_ABS + TOL_REL * max(1., float(np.abs(npm).sum()))
                need(abs(act - rec) <= tol, f'resolution null {nm} {g} {d_}: econ resid_resol {act} != recomputed {rec} (mismatched artifacts)')
                rng = np.random.default_rng([SEED, si, gi, di])
                Sn = ((rng.random((N_NULL, len(me))) < me[None, :]).astype(float) - me[None, :]) @ npm if len(me) else np.zeros(N_NULL)
                eps = 1e-9 * (1. + float(np.abs(npm).sum()))
                pct = (float(np.sum(Sn < act - eps)) + 0.5 * float(np.sum(np.abs(Sn - act) <= eps))) / N_NULL
                rows.append(dict(id_set=nm, label=LABEL[nm], grp=g, day=d_, n_markets=int(len(me)), n_markets_np_nonzero=int((npm != 0).sum()),
                                 n_tie_markets=int((V == 0.5).sum()), actual_resid_resol=act, recomputed=rec, null_mean=float(Sn.mean()), null_sd=float(Sn.std()),
                                 null_p2_5=float(np.percentile(Sn, 2.5)), null_p50=float(np.percentile(Sn, 50)), null_p97_5=float(np.percentile(Sn, 97.5)),
                                 percentile=pct if len(me) else None, p_two_sided=min(1., 2 * min(pct, 1 - pct)) if len(me) else None, n_draws=N_NULL,
                                 seed=f'default_rng([{SEED}, {si}, {gi}, {di}])',
                                 rule='S = sum NP_m (V_m - m_END_m) over fifo-ok markets; null V*_m ~ Bernoulli(m_END_m); percentile = P(S* < S) + P(S* = S) / 2'))
    return pd.DataFrame(rows)


def rank_persistence(MD, ID, M):
    """Oct 23 (ID day) rank vs each measurement day's rank, by maker shares and by net P&L to resolution, for every ex-ante member; the ID-day
    rank of a list member must equal its identities.json rank (else refuse). rank -1 = not ranked that day."""
    ex = ID['ex_ante']
    lr = {'sh': {x['addr']: int(x['rank']) for x in ex['top10_shares']}, 'pnl': {x['addr']: int(x['rank']) for x in ex['top10_pnl']}}
    memb = sorted(M['exante_sh'] | M['exante_pnl'])
    R_ = {(d, a): r for d, a, r in zip(MD.day, MD.addr, MD.itertuples(index=False))}
    rows = []
    for a in memb:
        for met, col, vcol in (('sh', 'rank_sh', 'sh_mk'), ('pnl', 'rank_pnl', 'net_res')):
            r0 = R_.get((C.ID_DAY, a))
            k0 = int(getattr(r0, col)) if r0 is not None else -1
            if a in lr[met]:
                need(k0 == lr[met][a], f'maker_day {C.ID_DAY} {col} of {a} = {k0} != identities.json rank {lr[met][a]}')
            ks = []
            for d in (C.ID_DAY,) + C.MEASURE_DAYS:
                r = R_.get((d, a))
                k = int(getattr(r, col)) if r is not None else -1
                ks.append(k)
                rows.append(dict(kind='rank', addr=a, metric=met, day=d, day_rank=k, in_top10=bool(1 <= k <= 10),
                                 value=float(getattr(r, vcol)) if r is not None else None, in_exante_sh=a in M['exante_sh'], in_exante_pnl=a in M['exante_pnl'],
                                 id_rank=k0))
            km = [k for k in ks[1:] if k >= 1]
            rows.append(dict(kind='member_summary', addr=a, metric=met, day='measure_days', day_rank=None, id_rank=k0,
                             n_days_top10=int(sum(1 <= k <= 10 for k in ks[1:])), n_days_ranked=len(km), median_rank=float(np.median(km)) if km else None,
                             in_exante_sh=a in M['exante_sh'], in_exante_pnl=a in M['exante_pnl']))
    D = pd.DataFrame(rows)
    for nm, met in (('exante_sh', 'sh'), ('exante_pnl', 'pnl')):
        for d in C.MEASURE_DAYS:
            x = D[(D.kind == 'rank') & (D.metric == met) & (D.day == d) & D.addr.isin(M[nm])]
            rows.append(dict(kind='set_summary', addr=nm, metric=met, day=d, n_members=int(len(x)), n_in_top10=int(x.in_top10.sum()),
                             frac_in_top10=rate(int(x.in_top10.sum()), len(x))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ the fills_valued pass ------------------------------------------------------------------
def fv_pass(p, book_day, tl_mids, top2, memb):
    """One streaming pass over econ/fills_valued: status counts per day, per-order maker sums on BOOK markets, the timeline rows, per-market
    END mid / V (min and max over the market's rows: they must agree), and the ex-ante members' counterparty sums (cp = a member, 'field' or
    'self'; a self-trade leg counted once, by its maker row)."""
    cnt, parts, tl, mk, cp = Counter(), [], [], [], []
    bm = pa.array(sorted(book_day), pa.int64())
    tm, ta = pa.array(sorted(tl_mids), pa.int64()), pa.array(sorted(top2), pa.string())
    ma, md = pa.array(sorted(memb), pa.string()), pa.array(list(C.MEASURE_DAYS), pa.string())
    for b in pq.ParquetFile(p).iter_batches(batch_size=250_000, columns=list(dict.fromkeys(FV_VAL + FV_ST))):
        t = pa.Table.from_batches([b])
        ds = set(pc.unique(t['day']).to_pylist())
        need(ds <= set(DAYS), 'fills_valued: rows of a day outside the consumed days')
        for c in FV_ST:
            g = t.group_by(['day', c]).aggregate([([], 'count_all')])
            for d, v, k in zip(g['day'].to_pylist(), g[c].to_pylist(), g['count_all'].to_pylist()):
                cnt[(d, c, 'nan' if v is None else str(v))] += int(k)
        sel = pc.and_(pc.equal(t['role'], 'maker'), pc.is_in(t['mid'], value_set=bm))
        X = t.filter(sel).select(['day', 'mid', 'hash', 'net_res', 'gross_res', 'fee_res', 'status', 'st_res']).to_pandas()
        if len(X):
            need(bool((X.mid.map(book_day) == X.day).all()), 'fills_valued: a BOOK market row under another day')
            ok = ((X.status == OK) & X.st_res.isin(RES_VALID)).to_numpy(bool)
            parts.append(pd.DataFrame(dict(day=X.day.to_numpy(object), mid=X.mid.to_numpy(np.int64), hash=X.hash.to_numpy(object), n_fv=1,
                                           n_fv_ok=ok.astype(np.int64), net_res=np.where(ok, X.net_res.to_numpy(float), 0.),
                                           gross_res=np.where(ok, X.gross_res.to_numpy(float), 0.), fee_res=np.where(ok, X.fee_res.to_numpy(float), 0.)))
                         .groupby(['day', 'mid', 'hash'], as_index=False).sum())
        if len(tm):
            Y = t.filter(pc.and_(pc.is_in(t['mid'], value_set=tm), pc.is_in(t['addr'], value_set=ta))).to_pandas()
            if len(Y):
                tl.append(Y)
        Z = t.select(['day', 'mid', 'm_end', 'V']).to_pandas()
        mk.append(Z.groupby(['day', 'mid']).agg(me_lo=('m_end', 'min'), me_hi=('m_end', 'max'), v_lo=('V', 'min'), v_hi=('V', 'max')).reset_index())
        Q = t.filter(pc.and_(pc.is_in(t['addr'], value_set=ma), pc.is_in(t['day'], value_set=md))).select(
            ['day', 'addr', 'cp_addr', 'role', 's', 'q', 'net_res', 'status', 'st_res', 'self_trade']).to_pandas()
        if len(Q):
            st_ = Q.self_trade.to_numpy(bool)
            Q = Q[~st_ | (Q.role == 'maker').to_numpy(bool)]
            st_ = Q.self_trade.to_numpy(bool)
            ok = (Q.status == OK).to_numpy(bool)
            okr = ok & Q.st_res.isin(RES_VALID).to_numpy(bool)
            cp.append(pd.DataFrame(dict(day=Q.day.to_numpy(object), addr=Q.addr.to_numpy(object),
                                        cp=np.where(st_, 'self', np.where(Q.cp_addr.isin(memb).to_numpy(bool), Q.cp_addr.to_numpy(object), 'field')),
                                        role=Q.role.to_numpy(object), s=Q.s.to_numpy(np.int64), n=1, n_ok=ok.astype(np.int64), n_res_ok=okr.astype(np.int64),
                                        q_ok=np.where(ok, Q.q.to_numpy(float), 0.), net_ok=np.where(okr, Q.net_res.to_numpy(float), 0.)))
                      .groupby(['day', 'addr', 'cp', 'role', 's'], as_index=False).sum())
        guard('fills_valued pass')
    N = (pd.concat(parts, ignore_index=True).groupby(['day', 'mid', 'hash'], as_index=False).sum() if parts else
         pd.DataFrame(columns=['day', 'mid', 'hash', 'n_fv', 'n_fv_ok', 'net_res', 'gross_res', 'fee_res']))
    T = pd.concat(tl, ignore_index=True) if tl else pd.DataFrame(columns=list(dict.fromkeys(FV_VAL + FV_ST)))
    MK = pd.concat(mk, ignore_index=True).groupby(['day', 'mid']).agg(me_lo=('me_lo', 'min'), me_hi=('me_hi', 'max'), v_lo=('v_lo', 'min'),
                                                                      v_hi=('v_hi', 'max')) if mk else None
    marks = {}
    if MK is not None:
        for (d, m), r in MK.iterrows():
            need((pd.isna(r.me_lo) and pd.isna(r.me_hi)) or r.me_lo == r.me_hi, f'fills_valued {d} {m}: m_end differs within the market')
            need((pd.isna(r.v_lo) and pd.isna(r.v_hi)) or r.v_lo == r.v_hi, f'fills_valued {d} {m}: V differs within the market')
            marks[(d, int(m))] = (float(r.me_lo), float(r.v_lo))
    CP = (pd.concat(cp, ignore_index=True).groupby(['day', 'addr', 'cp', 'role', 's'], as_index=False).sum() if cp else
          pd.DataFrame(columns=['day', 'addr', 'cp', 'role', 's', 'n', 'n_ok', 'n_res_ok', 'q_ok', 'net_ok']))
    return cnt, N, T, marks, CP


# ------------------------------------------------------------------ timelines / examples / unresolved ------------------------------------------------------------------
def with_classes(O, demote):
    O = O.copy()
    for f, col in FIELDS.items():
        eff, dem = effective(O[col], f, demote)
        O[f'raw_{col}'], O[col], O[f'{f}_demoted'] = O[col].to_numpy(object), eff, dem
    return O


def timeline(day, mid, top2, T, LF, OC, tob, frames, levels):
    T = T[(T.day == day) & (T.mid == mid)].copy()
    rank = {a: i + 1 for i, a in enumerate(top2)}
    lf = LF[(LF.day == day) & (LF.mid == mid)][['sid', 'mk_idx', 'mk_hash', 'lo_ms', 'hi_ms', 't_method', 'f_before', 'f_after', 'status']]
    need(not lf.duplicated(['sid', 'mk_idx']).any(), f'lifecycle fills {day} {mid}: duplicate (sid, mk_idx)')
    T = T.merge(lf.rename(columns={'mk_hash': 'leg_mk_hash', 'status': 'fill_status'}), on=['sid', 'mk_idx'], how='left', validate='many_to_one')
    T['fill_status'] = T.fill_status.fillna('no_lifecycle_fill')
    oc = OC[(OC.day == day) & (OC.mid == mid)]
    ocols = ['mk_hash', 'scenario'] + ORD_SHOW + [c for f, c in FIELDS.items()] + [f'raw_{c}' for c in FIELDS.values()] + [f'{f}_demoted' for f in FIELDS] + ['status']
    om = oc[ocols].rename(columns={'mk_hash': 'hash', 'status': 'order_status'})
    T = T.merge(om, on='hash', how='left', validate='many_to_one')
    mk = (T.role == 'maker').to_numpy(bool)
    for c in om.columns:
        if c != 'hash':
            T[c] = T[c].where(mk)                                                       # a taker row's own order is not reconstructed
    T['rank_exante_sh'] = T.addr.map(rank)
    px = T.p.to_numpy(float)
    T['ptick'] = np.where(np.isfinite(px), np.rint(np.nan_to_num(px) * 100), -1).astype(np.int64)
    T['rest_side'] = np.where(mk, T.y_side.to_numpy(object), np.where(T.y_side.to_numpy(object) == 'bid', 'ask', 'bid'))
    tb = tob.sort_values(['upd_ms', 'recv_ns'], kind='mergesort').reset_index(drop=True)
    u = tb.upd_ms.to_numpy(np.int64)
    lo, hi = T.lo_ms.fillna(-1).to_numpy(np.int64), T.hi_ms.fillna(-1).to_numpy(np.int64)
    ipre = np.where(lo >= 0, np.searchsorted(u, lo, 'right') - 1, -1)
    ipost = np.where(hi >= 0, np.searchsorted(u, hi, 'left'), len(u))
    for tag, ix, okm in (('pre', ipre, ipre >= 0), ('post', ipost, ipost < len(u))):
        j = np.where(okm, ix, 0)
        for c in ('upd_ms', 'best_bid', 'best_bid_sh', 'best_ask', 'best_ask_sh', 'status'):
            v = tb[c].to_numpy()[j] if len(tb) else np.full(len(T), None, dtype=object)
            T[f'tob_{tag}_{c}'] = np.where(okm, v, None) if c == 'status' else np.where(okm, v, np.nan)
        T[f'tob_{tag}_ref'] = np.where(okm, 'lo_ms' if tag == 'pre' else 'hi_ms', 'lo_unbounded' if tag == 'pre' else 'none_after')
    fst = {int(a): b for a, b in zip(frames.fid.to_numpy(np.int64), frames.status.to_numpy(object))}
    want = set(zip(T.rest_side, T.ptick))
    lv = levels[np.array([(s, p) in want for s, p in zip(levels.side, levels.ptick)], dtype=bool)]
    cell = {}
    for fid, s, p, sh, st in zip(lv.fid, lv.side, lv.ptick, lv.shares, lv.status):
        a = cell.setdefault((int(fid), s, int(p)), [0., True])
        a[0] += float(sh)
        a[1] &= st == OK
    for tag, col in (('before', 'f_before'), ('after', 'f_after')):
        vals, sts = [], []
        for f, s, p in zip(T[col].fillna(-1).to_numpy(np.int64), T.rest_side, T.ptick):
            if f < 0 or int(f) not in fst:
                vals.append(np.nan); sts.append('no_frame')
            elif (int(f), s, int(p)) in cell:
                a = cell[(int(f), s, int(p))]
                vals.append(a[0]); sts.append(OK if a[1] else 'cell_status')
            else:
                vals.append(0.); sts.append('absent_0')
        T[f'shown_{tag}'], T[f'shown_{tag}_status'], T[f'shown_{tag}_frame_status'] = vals, sts, [fst.get(int(f)) for f in T[col].fillna(-1)]
    first = ['day', 'mid', 'rank_exante_sh', 'addr', 'role', 'sid', 'mk_idx', 'hash', 'leg_mk_hash', 'y_side', 'rest_side', 'ptick', 'p', 'q', 's',
             'lo_ms', 'hi_ms', 't_method', 'f_before', 'f_after', 'fill_status', 't_ms', 't_status', 'gross_res', 'fee_res', 'fee_fill', 'net_res'] + \
            [f'v_{h}' for h in E.VALS] + ['st_res', 'status']
    T = T.rename(columns={'status': 'fv_status'})
    first = [('fv_status' if c == 'status' else c) for c in first]
    T = T.sort_values(['hi_ms', 'sid', 'mk_idx', 'role'], kind='mergesort', na_position='last')
    return T[first + [c for c in T.columns if c not in first]].reset_index(drop=True)


def examples(day, mid, top2, OC, NET):
    rows = []
    oc = OC[(OC.day == day) & (OC.mid == mid)].copy()
    oc['key'] = oc.mk_hash.map(hkey)
    for rk, a in enumerate(top2, 1):
        X = oc[oc.mk_signer == a].sort_values('key', kind='mergesort')
        pick = []
        for i, h in enumerate(X.sort_values(['n_fills', 'key'], ascending=[False, True], kind='mergesort').mk_hash.head(N_MOST)):
            pick.append((f'most_filled_{i + 1}', h))
        rest = X[~X.mk_hash.isin([h for _, h in pick])]
        mul = rest[rest.removal_set.map(lambda s: len([x for x in str(s).split('|') if x]) >= 2)]
        if len(mul):
            pick.append(('removal_set_ge_2_explanations', mul.mk_hash.iloc[0]))
        rest = rest[~rest.mk_hash.isin([h for _, h in pick])]
        one = rest[rest.n_fills == 1]
        if len(one):
            pick.append(('single_fill', one.mk_hash.iloc[0]))
        for rule, h in pick:
            r = X[X.mk_hash == h].iloc[0].to_dict()
            n = NET.get((day, mid, h), NET0)
            d = dict(day=day, mid=mid, rank_exante_sh=rk, addr=a, rule=rule, sha256_hash=r.pop('key'), n_fv=n[0], n_fv_ok=n[1], net_res=n[2],
                     gross_res=n[3], fee_res=n[4])
            d.update({k: v for k, v in r.items() if k not in d})
            rows.append(d)
    return rows


UNR_COLS = ['rank_abs_net', 'day', 'mid', 'mk_hash', 'sha256_hash', 'mk_signer', 'in_exante_sh', 'in_exante_pnl', 'y_side', 'y_ptick', 'n_fills', 'cum_shares',
            'net_res', 'gross_res', 'fee_res', 'n_fv', 'n_fv_ok', 'post_class', 'raw_post_class', 'post_demoted', 'removal_class', 'raw_removal_class',
            'removal_demoted', 'missing_post', 'missing_removal'] + [f'{c}_{s}' for s in SCEN for c in ('post_lo_ms', 'post_hi_ms', 'removal_set',
                                                                                                     'removal_lo_ms', 'removal_hi_ms')]
NET0 = (0, 0, 0., 0., 0.)


def unresolved(OA, O1, NET, M):
    X = O1[np.isin(O1.post_class.to_numpy(object), OPEN) | np.isin(O1.removal_class.to_numpy(object), OPEN)].copy()
    if not len(X):
        return pd.DataFrame(columns=UNR_COLS), 0
    nv = [NET.get(k, NET0) for k in zip(X.day, X.mid, X.mk_hash)]
    for i, c in enumerate(('n_fv', 'n_fv_ok', 'net_res', 'gross_res', 'fee_res')):
        X[c] = [v[i] for v in nv]
    X['abs_net'], X['key'] = np.abs(X.net_res.to_numpy(float)), X.mk_hash.map(hkey)
    n_cand = len(X)
    X = X.sort_values(['abs_net', 'key'], ascending=[False, True], kind='mergesort').head(N_TOP)
    A = OA.set_index(['day', 'mid', 'mk_hash', 'scenario']).sort_index()
    out = []
    for i, r in enumerate(X.itertuples(index=False), 1):
        sc = {s: A.loc[(r.day, r.mid, r.mk_hash, s)] for s in SCEN}
        d = dict(rank_abs_net=i, day=r.day, mid=r.mid, mk_hash=r.mk_hash, sha256_hash=r.key, mk_signer=r.mk_signer,
                 in_exante_sh=r.mk_signer in M['exante_sh'], in_exante_pnl=r.mk_signer in M['exante_pnl'], y_side=r.y_side, y_ptick=r.y_ptick,
                 n_fills=r.n_fills, cum_shares=r.cum_shares, net_res=r.net_res, gross_res=r.gross_res, fee_res=r.fee_res, n_fv=r.n_fv,
                 n_fv_ok=r.n_fv_ok, post_class=r.post_class, raw_post_class=r.raw_post_class, post_demoted=r.post_demoted,
                 removal_class=r.removal_class, raw_removal_class=r.raw_removal_class, removal_demoted=r.removal_demoted,
                 missing_post=open_bounds('post', r.post_class, r.post_demoted, r.raw_post_class, sc),
                 missing_removal=open_bounds('removal', r.removal_class, r.removal_demoted, r.raw_removal_class, sc))
        for s in SCEN:
            d.update({f'{c}_{s}': sc[s][c] for c in ('post_lo_ms', 'post_hi_ms', 'removal_set', 'removal_lo_ms', 'removal_hi_ms')})
        out.append(d)
    return pd.DataFrame(out, columns=UNR_COLS), n_cand


def open_bounds(field, eff, dem, raw, sc):
    """Names the missing evidence of an unresolved / assumption_sensitive post or removal claim from its three scenario rows."""
    if eff not in OPEN:
        return '-'
    parts = []
    if field == 'post':
        lo = {s: int(sc[s]['post_lo_ms']) for s in SCEN}
        op = [s for s in SCEN if lo[s] < 0]
        if op:
            parts.append('post_lo open (no capacity / fifo_through bound) in ' + '/'.join(op))
        if len(set(lo.values())) > 1:
            parts.append('post_lo differs: ' + ', '.join(f'{s}={lo[s]}' for s in SCEN))
    else:
        hi = {s: int(sc[s]['removal_hi_ms']) for s in SCEN}
        lo = {s: int(sc[s]['removal_lo_ms']) for s in SCEN}
        st = {s: str(sc[s]['removal_set']) for s in SCEN}
        op = [s for s in SCEN if hi[s] < 0]
        if op:
            parts.append('removal_hi open (end_open: no bound to the last usable frame) in ' + '/'.join(op))
        if any(v < 0 for v in lo.values()):
            parts.append('removal_lo open (no bounded fill lower end)')
        if len(set(hi.values())) > 1:
            parts.append('removal_hi differs: ' + ', '.join(f'{s}={hi[s]}' for s in SCEN))
        if len(set(st.values())) > 1:
            parts.append('removal_set differs: ' + ', '.join(f'{s}={st[s]}' for s in SCEN))
        me = [s for s in SCEN if 'missing_evidence' in st[s].split('|')]
        if me:
            parts.append('missing_evidence (gap / unknown cell / book end) in ' + '/'.join(me))
    if dem:
        parts.append(f'class demoted by a GT defect ({field} / {raw})')
    return '; '.join(parts) or 'no open bound: scenario disagreement only'


# ------------------------------------------------------------------ the build ------------------------------------------------------------------
def run(out=None, dest=None, log=print):
    out = Path(out) if out is not None else C.OUT
    dest = Path(dest) if dest is not None else C.RECEIPT / 'outputs'
    need(not dest.exists(), f'write-once: {dest.name}/ exists (refused before any input is read)')
    asof0, t0 = now(), time.time()
    tmp = dest.parent / f'.{dest.name}.partial_{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}_{os.getpid()}'
    need(not tmp.exists(), 'the partial output directory exists')
    tmp.mkdir(parents=True)
    try:
        R = build(Inputs(out), tmp, asof0, t0, log)
        try:
            os.mkdir(dest)
        except FileExistsError:
            need(False, f'write-once: {dest.name}/ appeared during the run')
        names = sorted(R['outputs']) + ['readings.json']
        for n in names:
            os.rename(tmp / n, dest / n)
        os.rmdir(tmp)
        return R
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def build(I, tmp, asof0, t0, log):
    OUTS, mem = {}, {}

    def put(name, df):
        gz = len(df) > LARGE_ROWS
        fn = f'{name}.csv.gz' if gz else f'{name}.csv'
        need(fn not in OUTS, f'output {fn} written twice')
        df.to_csv(tmp / fn, index=False, compression=({'method': 'gzip', 'mtime': 0} if gz else None))
        OUTS[fn] = dict(rows=int(len(df)), columns=int(len(df.columns)))

    # ---- 1. days: extract receipts, inventories, table statuses ----
    EXT, ST, INVR = {}, [], []
    for d in DAYS:
        X = EXT[d] = I.json(f'extract_{day_ok(d)}.json', f'extract receipt {d}')
        V = I.json(f'inventory_{d}.json', f'inventory {d}')
        need(X.get('day') == d and V.get('day') == d, f'{d}: the extract receipt / inventory names another day')
        sc, rec = X.get('status_counts') or {}, None
        for k in ['records', 'legs', 'markets'] + ([] if X.get('role') == 'ID' else ['tob']):
            p = I.schema(f'{k}_{d}.parquet', f'{k} {d}', MK_REQ if k == 'markets' else list(C.SCHEMAS[k]))
            n, cnt = count_values(p, ['status'])
            mine, theirs = dict(cnt['status']), {str(a): int(b) for a, b in (sc.get(k) or {}).items()}
            need(mine == theirs, f'{d} {k}: status counts {mine} != extract_{d}.json status_counts {theirs} (a dropped or unreported row)')
            need(n == int((X.get('counts') or {}).get(k, -1)), f'{d} {k}: {n} rows != extract_{d}.json counts')
            ST += status_rows(d, k, 'status', cnt['status'], n)
            if k == 'records':
                rec = (n, cnt['status'])
        for k in ('frames', 'levels'):
            c_ = Counter({str(a): int(b) for a, b in (sc.get(k) or {}).items()})
            if c_:
                ST += status_rows(d, k, 'status', c_, sum(c_.values()), source=f'extract_{d}.json')
        INVR += inventory_rows(d, V, X, rec)
        mem[f'days {d}'] = guard(f'days {d}')
    log(f'[report] days read: {len(DAYS)}', flush=True)
    # ---- 2. econ receipts + identities (the BOOK list and the sets are needed by the lifecycle stage) ----
    EJ, ID = I.json('econ/econ.json', 'econ receipt'), I.json('econ/identities.json', 'econ identities')
    book = {day_ok(d): sorted(int(m) for m in v) for d, v in (EJ.get('book') or {}).items()}
    M, top_sh = members(ID)
    top2 = top_sh[:2]
    # ---- 3. lifecycle ----
    PT, BO, BF, BK, life_n = [], [], [], [], {}
    for d in LIFE_DAYS:
        mids = sorted(int(m) for m in (EXT[d].get('population') or {}).get('depth_mids', []))
        pop = 'GT' if d == C.GT_DAY else 'BOOK'
        if pop == 'BOOK':
            need(mids == book.get(d, []), f'{d}: BOOK markets of extract_{d}.json {mids} != econ.json book {book.get(d, [])}')
        fc, kc, li, oc = Counter(), Counter(), Counter(), {s: Counter() for s in SCEN}
        nf = nk = 0
        no = {s: 0 for s in SCEN}
        for m in mids:
            tag = f'{d}_{m}'
            J = I.json(f'life/{tag}.json', f'lifecycle summary {d} {m}')
            O = I.table(f'life/{tag}_orders.parquet', f'lifecycle orders {d} {m}', list(C.SCHEMAS['orders']))
            F = I.table(f'life/{tag}_fills.parquet', f'lifecycle fills {d} {m}', list(C.SCHEMAS['fills']))
            K = I.table(f'life/{tag}_x_links.parquet', f'lifecycle links {d} {m}', LINK_COLS)
            check_life(J, O, F, K, d, m)
            PT += pt_rows(J, d, m, pop)
            li.update({k: int(v) for k, v in (J.get('inputs') or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
            fc.update(F.status.astype(str))
            kc.update(K.status.astype(str))
            nf, nk = nf + len(F), nk + len(K)
            for s in SCEN:
                x = O[O.scenario == s]
                oc[s].update(x.status.astype(str))
                no[s] += len(x)
            if pop == 'BOOK':
                BO.append(O.assign(day=d))
                BF.append(F.assign(day=d))
                BK.append(K.assign(day=d))
            mem[f'life {tag}'] = guard(f'lifecycle {tag}')
        life_n[d] = len(mids)
        ST += [dict(day=d, table='life_inputs', part='', column=k, kind='count', value='', n=v, den=None, rate=None, source='life/<day>_<mid>.json inputs')
               for k, v in sorted(li.items())]
        ST += status_rows(d, 'life_fills', 'status', fc, nf)
        ST += status_rows(d, 'life_x_links', 'status', kc, nk)
        for s in SCEN:
            ST += status_rows(d, 'life_orders', 'status', oc[s], no[s], part=s)
    PTm, R1 = r1(PT)
    cov_scen = R1['coverage_scenario']
    log(f'[report] lifecycle markets {life_n}; R1 {R1["T1"]["reading"]} / {R1["T2"]["reading"]} -> coverage scenario {cov_scen}', flush=True)
    # ---- 4. GT scores ----
    S = I.json(TRUTH_SCORES, 'GT scores')
    GTS, R2, demote = gt(S)
    # ---- 5. BOOK order tables (classes from S_FIFO rows, effective after demotion) ----
    OA = pd.concat(BO, ignore_index=True) if BO else pd.DataFrame(columns=list(C.SCHEMAS['orders']) + ['day'])
    FA = pd.concat(BF, ignore_index=True) if BF else pd.DataFrame(columns=list(C.SCHEMAS['fills']) + ['day'])
    LK = pd.concat(BK, ignore_index=True) if BK else pd.DataFrame(columns=LINK_COLS + ['day'])
    ex = FA.groupby(['day', 'mid', 'mk_hash']).agg(n_exec=('shares', 'size'), sh_exec=('shares', 'sum')).reset_index()
    OA = OA.merge(ex, on=['day', 'mid', 'mk_hash'], how='left', validate='many_to_one')
    need(bool((OA.n_exec.fillna(0).astype(np.int64) == OA.n_fills).all()), 'lifecycle: n_fills != the hash\'s fills rows (a dropped or extra fill)')
    OA = with_classes(OA, demote)
    O1 = OA[OA.scenario == FIFO].reset_index(drop=True)
    OC = OA[OA.scenario == cov_scen].reset_index(drop=True)
    case_check(set(O1.mk_signer), M, 'lifecycle orders')
    sides = tuple(sorted(set(O1.y_side.astype(str)) | {'bid', 'ask'}))
    COV = coverage(O1, M, demote, sides)
    WIN, RMS, LNK = windows(OA, M, cov_scen), removal_sets(OA, M, cov_scen), links(LK, O1, M)
    mem['r3'] = guard('R3 tables')
    # ---- 6. econ tables ----
    pmd = I.schema('econ/maker_day.parquet', 'econ maker_day', list(E.MD))
    miss = [c for c in ECON_NEW_MD if c not in pq.read_schema(pmd).names]
    need(not miss, f'econ maker_day: declared columns {miss} absent (econ additions not in this artifact): refusing rather than guessing')
    MD = I.table('econ/maker_day.parquet', 'econ maker_day', ['day', 'addr', 'population', 'sh_mk', 'net_res', 'rank_sh', 'rank_pnl', 'in_exante_sh',
                                                              'in_exante_pnl', 'in_sameday_sh', 'in_sameday_pnl', 'status'] + list(ECON_NEW_MD),
                 required=list(E.MD))
    for d in C.MEASURE_DAYS:
        x = MD[(MD.day == d) & (MD.population == 'FILL')]
        for col, nm in (('in_exante_sh', 'exante_sh'), ('in_exante_pnl', 'exante_pnl'), ('in_sameday_sh', 'sameday_sh'), ('in_sameday_pnl', 'sameday_pnl')):
            mem_ = M[nm] if nm.startswith('exante') else M[nm].get(d, frozenset())
            need(bool((x[col].astype(bool).to_numpy() == x.addr.isin(mem_).to_numpy()).all()), f'maker_day {d}: {col} != identities.json membership')
    pmm = I.schema('econ/maker_market.parquet', 'econ maker_market', list(E.MM))
    miss = [c for c in ECON_NEW_MM if c not in pq.read_schema(pmm).names]
    need(not miss, f'econ maker_market: declared columns {miss} absent (econ additions not in this artifact): refusing rather than guessing')
    MM = I.table('econ/maker_market.parquet', 'econ maker_market', MM_COLS, required=list(E.MM))
    SEG = I.table('econ/segments.parquet', 'econ segments', list(E.SEG))
    case_check(set(MM.addr.dropna()), M, 'maker_market')
    X_, K_, n_noaddr = econ_frame(MM)
    EID, R4, econ_chk = econ_tables(X_, K_, MD, SEG, M)
    econ_chk.update(maker_market_rows_without_addr=n_noaddr, rt_pnl_res_vs_maker_day_max_abs_dev=check_rt(X_, K_, MD))
    ESEG = econ_segments(SEG, R1)
    prm = I.path('econ/rewards_market.parquet', 'econ rewards_market')
    RWM = pq.read_table(prm).to_pandas()
    if 'day' in RWM.columns:
        need(set(RWM.day.astype(str)) <= set(DAYS), 'rewards_market: a day outside the consumed days')
    makers = {d: frozenset(MD.addr[(MD.day == d) & (MD.sh_mk > 0)]) for d in C.MEASURE_DAYS}
    memb = sorted(M['exante_sh'] | M['exante_pnl'])
    rk_ex = dict(sh={x['addr']: int(x['rank']) for x in ID['ex_ante']['top10_shares']}, pnl={x['addr']: int(x['rank']) for x in ID['ex_ante']['top10_pnl']})
    for d, dg in sorted((EJ.get('diagnostics') or {}).items()):
        ST += [dict(day=day_ok(d), table='econ_diagnostics', part='', column=k, kind='count', value='', n=v, den=None, rate=None,
                    source='econ/econ.json diagnostics') for k, v in sorted(dg.items()) if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for d in sorted(set(MM.day)):
        x = MM[MM.day == day_ok(d)]
        for c in MM_ST:
            ST += status_rows(d, 'econ_maker_market', c, Counter(x[c].astype(str)), len(x))
    for d in sorted(set(MD.day)):
        y = MD[MD.day == day_ok(d)]
        ST += status_rows(d, 'econ_maker_day', 'status', Counter(y.status.astype(str)), len(y))
    for d in sorted(set(SEG.day)):
        y = SEG[SEG.day == day_ok(d)]
        ST += status_rows(d, 'econ_segments', 'status', Counter(y.status.astype(str)), len(y))
    mem['econ'] = guard('econ tables')
    # ---- 7. timeline markets (deterministic, from the lifecycle legs) ----
    O1k = O1.set_index(['day', 'mid', 'mk_hash']).mk_signer
    TL = {}
    for d in C.MEASURE_DAYS:
        if not book.get(d):
            TL[d] = dict(mid=None, reason='no BOOK market on this day')
            continue
        f = FA[FA.day == d]
        sg = O1k.reindex(list(zip(f.day, f.mid, f.mk_hash))).to_numpy(object)
        n_by = Counter(int(m) for m, a in zip(f.mid, sg) if a in set(top2))
        cand = {m: int(n_by.get(m, 0)) for m in book[d]}
        best = max(cand.values())
        TL[d] = dict(mid=min(m for m, v in cand.items() if v == best), maker_legs_top2=best, candidates=cand, top2=top2,
                     rule='most maker legs (lifecycle fills rows) of the two top-ranked exante_sh addresses; ties -> smallest mid')
    # ---- 8. fills_valued pass ----
    book_day = {m: d for d, v in book.items() for m in v}
    pfv = I.schema('econ/fills_valued.parquet', 'econ fills_valued', list(E.FV))
    cnt, NETd, TLR, marks, CP = fv_pass(pfv, book_day, [v['mid'] for v in TL.values() if v['mid'] is not None], top2, memb)
    mem['fills_valued'] = guard('fills_valued pass')
    ADR = address_rows(X_, K_, MD, M, rk_ex, CP)
    CPT, cp_info = counterparty(CP, memb)
    RNULL = resolution_null(X_, K_, marks, M, makers)
    RKP = rank_persistence(MD, ID, M)
    mem['econ_extra'] = guard('address / counterparty / null / ranks')
    ec = EJ.get('counts') or {}
    for d, cols in ec.items():
        day_ok(d)
        for c, vals in cols.items():
            mine = {v: k for (d_, c_, v), k in cnt.items() if d_ == d and c_ == c}
            need(mine == {str(a): int(b) for a, b in vals.items()}, f'fills_valued {d} {c}: counts {mine} != econ.json counts {vals} (a dropped row)')
    need({d for d, _, _ in cnt} <= set(ec), 'fills_valued: a day without econ.json counts')
    for d in sorted({d for d, _, _ in cnt}):
        n_d = sum(k for (d_, c_, _), k in cnt.items() if d_ == d and c_ == 'status')
        for c in FV_ST:
            ST += status_rows(d, 'econ_fills_valued', c, Counter({v: k for (d_, c_, v), k in cnt.items() if d_ == d and c_ == c}), n_d)
    NET = {(d, int(m), h): (int(a), int(b), float(x), float(y), float(z)) for d, m, h, a, b, x, y, z in
           zip(NETd.day, NETd.mid, NETd.hash, NETd.n_fv, NETd.n_fv_ok, NETd.net_res, NETd.gross_res, NETd.fee_res)}
    # ---- 9. timelines, examples, unresolved ----
    EXR = []
    for d in C.MEASURE_DAYS:
        m = TL[d]['mid']
        if m is None:
            continue
        tob = I.table(f'tob_{d}.parquet', f'tob {d}', TOB_COLS, required=list(C.SCHEMAS['tob']), filters=[('mid', '==', m)])
        fr = I.table(f'depth/{d}_{m}_frames.parquet', f'depth frames {d} {m}', ['fid', 'upd_ms', 'status'], required=list(C.SCHEMAS['frames']))
        lv = I.table(f'depth/{d}_{m}_levels.parquet', f'depth levels {d} {m}', ['fid', 'side', 'ptick', 'shares', 'status'],
                     required=list(C.SCHEMAS['levels']))
        T = timeline(d, m, top2, TLR, FA, OC, tob, fr, lv)
        T.insert(2, 'scenario_rule', f'order columns from {cov_scen} (R1 coverage scenario)')
        put(f'timeline_{d}_{m}', T)
        TL[d].update(rows=int(len(T)), rows_maker=int((T.role == 'maker').sum()), rows_taker=int((T.role == 'taker').sum()))
        EXR += examples(d, m, top2, OC, NET)
        mem[f'timeline {d}'] = guard(f'timeline {d}')
    UNR, n_cand = unresolved(OA, O1, NET, M)
    # ---- 10. tables + readings ----
    ST = pd.DataFrame(ST)
    ST['day_order'] = ST.day.map({d: i for i, d in enumerate(DAYS)})
    put('inventory', pd.DataFrame(INVR))
    put('statuses', ST.sort_values(['day_order', 'table', 'column', 'part', 'kind', 'value'], kind='mergesort').drop(columns='day_order'))
    put('priority_tests', pd.concat([pd.DataFrame(PT), PTm], ignore_index=True))
    put('gt_scores', GTS)
    put('coverage', COV)
    put('windows', WIN)
    put('removal_sets', RMS)
    put('links', LNK)
    put('econ_identity_day', EID)
    put('econ_segments', ESEG)
    put('econ_address_day', ADR)
    put('counterparty', CPT)
    put('resolution_null', RNULL)
    put('rank_persistence', RKP)
    put('rebate_scenario', EID[['id_set', 'label', 'grp', 'day', 'n_addr', 'sh_mk', 'sh', 'gross_res', 'fee_res', 'net_res', 'reb25_res', 'reb25_fill',
                                'n_days', 'ci']].assign(note='econ rebate scenario sums beside trading P&L (net_res); not added together here'))
    put('rewards_market', RWM.assign(statement=REWARDS_STATEMENT))
    put('examples', pd.DataFrame(EXR) if EXR else pd.DataFrame(columns=['day', 'mid', 'rank_exante_sh', 'addr', 'rule', 'sha256_hash', 'mk_hash']))
    put('unresolved_top', UNR)
    pooled = COV[(COV.day == 'pooled') & (COV.side == 'both')]
    R3 = {}
    for name in COV_SETS:
        x = pooled[pooled.id_set == name]
        R3[name] = dict(label=LABEL[name], n_orders=int(x.den_orders.iloc[0]), n_exec=int(x.den_exec.iloc[0]), shares=float(x.den_shares.iloc[0]),
                        fractions={f: {r.cls: dict(orders=fnone(r.frac_orders), executions=fnone(r.frac_exec), shares=fnone(r.frac_shares),
                                                   n_orders=int(r.n_orders), raw_n_orders=int(r.raw_n_orders), demoted=bool(r.demoted))
                                       for r in x[x.field == f].itertuples()} for f in FIELDS})
    R3 = dict(per_set=R3, demoted={f: sorted(v) for f, v in demote.items()}, scope='BOOK markets, pooled days, both sides', classes_from=FIFO,
              windows_primary_scenario=cov_scen, n_book_markets=int(sum(len(v) for v in book.values())))
    for fn in OUTS:
        OUTS[fn].update(sha256=sha256(tmp / fn), bytes=(tmp / fn).stat().st_size)
    life_asof = dict(report=asof0, extract_receipts={d: EXT[d].get('asof') for d in LIFE_DAYS})
    R1['asof'], R3['asof'] = life_asof, life_asof
    rn = RNULL[RNULL.grp == 'in']
    R4x = dict(resolution_null_in={nm: {d: dict(actual=float(a_), percentile=fnone(p_), n_markets=int(n_)) for d, a_, p_, n_ in
                                        zip(x.day, x.actual_resid_resol, x.percentile, x.n_markets)} for nm, x in rn.groupby('id_set')},
               resolution_null_rule=dict(n_draws=N_NULL, seed=SEED, statistic='sum NP_m (V_m - m_END_m) over fifo-ok markets',
                                         null='V*_m ~ Bernoulli(m_END_m) per market', declared='before any real output was read'),
               address_sign_consistent={a: bool(v) for a, v in zip(ADR.addr[ADR.day == 'pooled'], ADR.sign_consistent[ADR.day == 'pooled'])}
               if len(ADR) else {})
    RK = RKP[RKP.kind == 'set_summary']
    rank_s = {f'{a}/{m}': {d: fnone(f) for d, f in zip(x.day, x.frac_in_top10)} for (a, m), x in RK.groupby(['addr', 'metric'])}
    RD = dict(version=C.VERSION, module='pf_mt_report', asof_start=asof0, asof_end=now(), wall_s=round(time.time() - t0, 1),
              source=dict(path=str(Path(__file__).resolve()), sha256=sha256(Path(__file__).resolve()), common_sha256=sha256(Path(C.__file__).resolve())),
              input_asof=dict(extract={d: EXT[d].get('asof') for d in DAYS}, econ=EJ.get('asof_end'), scores=S.get('scored_utc')),
              inputs=I.files, outputs=OUTS, R1=R1, R2=R2, R3=R3, R4=dict(per_set=R4, checks=econ_chk, days=list(C.MEASURE_DAYS), asof=EJ.get('asof_end'), **R4x,
                                                                         n_days_measured=len(C.MEASURE_DAYS), ci=ci_text(len(C.MEASURE_DAYS)),
                                                                         note='point estimates per day; no intervals below 5 complete UTC days (rule 8)'),
              timelines=TL, unresolved_top=dict(n_candidates=int(n_cand), n_listed=int(len(UNR)), rule=f'top {N_TOP} by |net_res| (maker rows, '
                                                'status ok, st_res in RES_VALID) among BOOK orders whose effective post or removal class is unresolved / '
                                                'assumption_sensitive'),
              counterparty=cp_info, rank_persistence=dict(frac_in_top10_by_day=rank_s, rule='rank -1 = not ranked that day; ID-day ranks checked '
                                                                                          'against identities.json'),
              rewards=dict(statement=REWARDS_STATEMENT, rows=int(len(RWM)), columns=list(RWM.columns)),
              lifecycle_markets=life_n, memory=dict(guard_mib=C.GUARD_MIB, vmhwm_mib=round(hwm(), 1), by_stage={k: round(v, 1) for k, v in mem.items()}),
              slot=os.environ.get('PF_SLOT'), argv=sys.argv)
    (tmp / 'readings.json').write_text(json.dumps(RD, indent=1, sort_keys=True, default=str) + '\n')
    RD['outputs'] = OUTS
    return RD


def check_life(J, O, F, K, d, m):
    """Contract checks of one market's lifecycle outputs (refuses: a mismatched or out-of-contract artifact)."""
    need(int(J.get('mid', -1)) == m, f'lifecycle summary {d} {m}: mid {J.get("mid")}')
    need(bool((O.mid == m).all()) and bool((F.mid == m).all()) and bool((K.mid == m).all()), f'lifecycle {d} {m}: a table holds another market')
    need(set(O.scenario) <= set(SCEN), f'lifecycle orders {d} {m}: unknown scenario {sorted(set(O.scenario) - set(SCEN))}')
    for col in FIELDS.values():
        bad = set(O[col].astype(str)) - set(C.CLASSES)
        need(not bad, f'lifecycle orders {d} {m}: unknown class {sorted(bad)} in {col} (outside pf_mt_common.CLASSES)')
    tok = {x for s in O.removal_set.astype(str) for x in s.split('|')}
    need(tok <= set(C.REMOVAL), f'lifecycle orders {d} {m}: removal_set member {sorted(tok - set(C.REMOVAL))} outside pf_mt_common.REMOVAL')
    need(not O.duplicated(['scenario', 'mk_hash']).any(), f'lifecycle orders {d} {m}: duplicate (scenario, mk_hash)')
    H = O.mk_hash.nunique()
    need(all(int((O.scenario == s).sum()) == H for s in SCEN), f'lifecycle orders {d} {m}: not one row per hash per scenario')
    for col in FIELDS.values():
        need(int(O.groupby('mk_hash')[col].nunique().max() if len(O) else 0) <= 1, f'lifecycle orders {d} {m}: {col} differs across scenario rows')
    need(set(F.mk_hash) == set(O.mk_hash), f'lifecycle {d} {m}: fills and orders hold different hashes')
    O1 = O[O.scenario == FIFO].set_index('mk_hash')
    sh = F.shares.to_numpy(float)
    cov = J.get('coverage') or {}
    for f, col in FIELDS.items():
        cl = O1[col].reindex(F.mk_hash).to_numpy(object)
        for c in C.CLASSES:
            mo, mf = O1[col].to_numpy(object) == c, cl == c
            got = dict(orders=int(mo.sum()), executions=int(mf.sum()), shares=round(float(np.nansum(sh[mf])), 6))
            rep = (cov.get(f) or {}).get(c)
            need(rep is not None and int(rep['orders']) == got['orders'] and int(rep['executions']) == got['executions']
                 and abs(float(rep['shares']) - got['shares']) <= 1e-5 * max(1., abs(got['shares'])),
                 f'lifecycle {d} {m}: coverage {f}/{c} recomputed {got} != summary {rep}')


def main(argv):
    need(len(argv) == 1, 'usage: pf_mt_report.py (no arguments: reads OUT, writes RECEIPT/outputs)')
    R = run()
    print(json.dumps(dict(R1={t: {k: R['R1'][t][k] for k in ('den', 'violations', 'rate', 'reading')} for t in ('T1', 'T2')},
                          coverage_scenario=R['R1']['coverage_scenario'], R2=R['R2']['reading'], outputs=sorted(R['outputs']),
                          vmhwm_mib=R['memory']['vmhwm_mib']), indent=1, default=str), flush=True)
    return R


if __name__ == '__main__':
    main(sys.argv)
