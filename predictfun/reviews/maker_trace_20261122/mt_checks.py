"""MAKER-TRACE-RECONSTRUCTION-v1 independent checker + falsifier (P-2026-005; contract.md section 8; session aea5bfb9, 2026-11-22).

Written from contract.md and pf_mt_common.py ONLY - never from pf_mt_extract / pf_mt_lifecycle / pf_mt_econ / pf_mt_truth.
  load(out_dir, ...)          -> E: output paths + populations (tables are read lazily: one day / one market batch at a time)
  run_checks(E, only=None)    -> {checks: {name: bool}, details: {name: {...}}}; a check passes only if it EVALUATED >= 1 unit and
                                 nothing failed; a check that cannot evaluate fails with its reason (no default-pass branch)
  reparse(day, mid, E, ...)   -> same shape: an independent raw re-parse of ONE market from only the hours it needs
  falsify(E, ...)             -> planted defects; each must be refused by its named check with the expected reason
CLI (only through predictfun/research/run_light.sh):  mt_checks.py smoke | run | reparse DAY MID
Contract amendment 1 (integrator, 2026-11-22) is encoded: price-priority exhaustion (mk_idx never evidence) with T1 / T2, '+'-joined status
flags counted by flag, levels.status, the reconciling-variant conflict rule, lower + upper boundary censoring excluded from economics, and
taker legs valued at the maker legs' YES prices (a priceExecuted mismatch is a status).
The econ / truth tables have no schema in the contract: IFACE declares the semantic columns this checker needs; integration maps them
to the builders' names by editing IFACE values only (until then those checks fail closed, naming the missing columns)."""
import builtins, datetime as dt, gzip, hashlib, json, os, re, shutil, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'research'))
import pf_mt_common as C  # noqa: E402

TOL_SH, TOL_USD, TOL_FR = 1e-9, 1e-6, 5e-4
PX_TOL = 2e-6       # maker-weighted YES price vs priceExecuted: yes_space rounds YES prices to 6 decimals (a diagnostic, never a status)
UNUSABLE_FRAME_FLAGS = frozenset({'bad_frame', 'mid_mismatch', 'out_of_order', 'off_grid'})   # pinned: new_conn / gap / dup / same_ms_changed are usable


def usable(status):
    """Rows whose status carries no unusable-frame flag (frames / levels)."""
    u = {s: not (flags(s) & UNUSABLE_FRAME_FLAGS) for s in pd.Series(status).dropna().unique()}
    return pd.Series(status).map(u).fillna(False).to_numpy(bool)
UTC = dt.timezone.utc
EPOCH = dt.datetime(1970, 1, 1, tzinfo=UTC)
BOOK_MANIFEST = C.REPO / 'predictfun/reviews/ladder_async_fast_20261122/manifest.json'
GT_SCOPE = Path('/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108/scope.json')
EXTERNAL_STATUSES = frozenset({'boundary_censored'})       # record statuses set from outside the record itself (contract section 2)
FIFO_FAMILY = tuple(s for s, v in C.SCENARIOS.items() if v['priority'] == 'price_time')
ARROW = {'int64': (pa.int64(),), 'float64': (pa.float64(),), 'str': (pa.string(), pa.large_string())}
GROUPS = dict(size=(('size_lb', 'size_exact'), 'size_class'), post=(('post_lo_ms', 'post_hi_ms'), 'post_class'),
              persist=(('persist_lo_ms', 'persist_hi_ms'), 'persist_class'), queue=(('ahead_lo', 'ahead_hi'), 'queue_class'),
              removal=(('removal_set', 'removal_lo_ms', 'removal_hi_ms'), 'removal_class'))
IFACE = dict(   # semantic role -> builder column / key: PINNED by the integrator (2026-11-22, all builders done)
    fills_valued=dict(day='day', mid='mid', addr='addr', role='role', sid='sid', s='s', q='q', p='p', V='V', gross='gross_res', fee='fee_res',
                      fee_px='fee_fill', pnl='net_res', status='status'),
    maker_market=dict(day='day', mid='mid', addr='addr', gross_mk='gross_res_mk', gross_tk='gross_res_tk', fees='fee_res', fees_px='fee_fill',
                      pnl='net_res', rt='rt_pnl', resid='resid_res', drift='resid_drift', resol='resid_resol', net_pos='net_pos', status='status'),
    maker_day=dict(day='day', addr='addr', gross_mk='gross_res_mk', gross_tk='gross_res_tk', fees='fee_res', pnl='net_res', rt='rt_pnl',
                   resid='resid_res', drift='resid_drift', resol='resid_resol', in_exante_sh='in_exante_sh', in_exante_pnl='in_exante_pnl',
                   status='status'),
    segments=dict(day='day', dim='dim', bucket='bucket', pnl='net_res', gross='gross_res', status='status'),
    segments_keys=('id_set', 'label', 'grp', 'role'), segments_total_dim='all',
    segments_full_dims=('tseg', 'pseg', 'kind', 'tclock'),      # each partitions the dim 'all' total of its (day, id_set, label, grp, role)
    own_orders=dict(mid=('mid', 'market', 'market_id'), h=('mk_hash', 'hash', 'order_hash'),     # real names first; registry in metadata key mt_truth
                    act_lo=('act_lo_venue_ms', 'act_lo_ms', 'true_act_ms'), act_hi=('act_hi_venue_ms', 'act_hi_ms', 'true_act_ms'),
                    size=('size_sh', 'true_size'), filled=('filled_sh', 'true_filled'), state=('final_state', 'true_state')),
    own_required=('h', 'act_lo', 'act_hi', 'size', 'filled', 'state'),                       # mid is optional (join by hash alone)
    scores_blocks=(('posting',), ('size', 'lb'), ('size', 'exact'), ('size', 'ub'), ('removal', 'exhausted_only')),   # recomputed where possible
    coverage_file='coverage.json',
    econ_info_flags=(),                       # pinned: ANY non-ok source flag excludes the record from economics
    t1_key=r'(?i)^t1(?![0-9])', t2_key=r'(?i)^t2(?![0-9])')   # where the lifecycle summary reports the priority tests T1 / T2
ECON_T = ('maker_market', 'maker_day', 'fills_valued', 'segments')


# ------------------------------------------------------------------------------------------------------------------ utilities
def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def hwm_mib():
    with open('/proc/self/status') as f:
        for ln in f:
            if ln.startswith('VmHWM:'):
                return int(ln.split()[1]) / 1024.
    return float('inf')     # unknown -> over the guard (fail closed)


def guard(stage):
    h = hwm_mib()
    if h > C.GUARD_MIB:
        raise SystemExit(f'[mt_checks] ABORT at {stage}: VmHWM {h:.0f} MiB > guard {C.GUARD_MIB:.0f} MiB')


def raw_path(family, stem, raw_root=None):
    """pf_mt_common.hour_path FIRST (raises Forbidden before any filesystem access), then rebased onto a synthetic raw root."""
    p = C.hour_path(family, stem)
    return p if raw_root is None else Path(raw_root) / family / p.name


def stems_between(t0_ms, t1_ms):
    """UTC hour stems overlapping [t0, t1] - strings only; nothing is opened or printed here."""
    return [dt.datetime.fromtimestamp(h * 3600, UTC).strftime('%Y%m%d_%H') for h in range(int(t0_ms) // 3600000, int(t1_ms) // 3600000 + 1)]


def stem_ms(s):
    return int(dt.datetime.strptime(s, '%Y%m%d_%H').replace(tzinfo=UTC).timestamp() * 1000)


def hour_of_ns(ns):
    return stems_between(int(ns) // 10 ** 6, int(ns) // 10 ** 6)[0]


def iso_ms(s):
    return int((dt.datetime.fromisoformat(s.replace('Z', '+00:00')) - EPOCH) // dt.timedelta(milliseconds=1))


def lo(s):
    return s.lower() if isinstance(s, str) else s


def isnull(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def same(a, b, tol=1e-12):
    """NaN-aware equality: floats within a relative tolerance, strings case-insensitively (hex identities)."""
    if isnull(a) or isnull(b):
        return isnull(a) and isnull(b)
    if isinstance(a, str) or isinstance(b, str):
        return lo(str(a)) == lo(str(b))
    return abs(float(a) - float(b)) <= tol * (1 + abs(float(b)))


def neq(x, y):
    """NaN-aware elementwise inequality of two aligned Series (NaN == NaN)."""
    return ~((x == y) | (x.isna() & y.isna()))


def flags(s):
    """Amendment 1: a status is '+'-joined flags in a fixed order; 'ok' only when clean."""
    return frozenset(s.split('+')) if isinstance(s, str) and s else frozenset()


def hasf(series, flag):
    """Vectorised 'flag in flags(status)'."""
    return series.astype(object).fillna('').str.contains(r'(?:^|\+)' + re.escape(flag) + r'(?:\+|$)', regex=True).to_numpy(bool)


def only_flags(series, allowed):
    """Rows whose flags are all in `allowed` (evaluated on the distinct statuses)."""
    u = {s: bool(flags(s)) and flags(s) <= set(allowed) for s in series.dropna().unique()}
    return series.map(u).fillna(False).to_numpy(bool)


def flag_counts(series):
    c = Counter()
    for s, n in series.value_counts().items():
        for f in flags(s):
            c[f] += int(n)
    return dict(c)


def rset(s):
    """removal_set -> frozenset of tokens ('a|b', 'a,b' or a JSON list); None if unparseable."""
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip()
    if t.startswith('['):
        try:
            v = json.loads(t)
        except ValueError:
            return None
        return frozenset(map(str, v)) if isinstance(v, list) else None
    return frozenset(x.strip() for x in re.split(r'[|,]', t) if x.strip())


def rkey(s):
    v = rset(s)
    return '<unparseable>' if v is None else '|'.join(sorted(v))


def find_map(obj, target, depth=0):
    """True iff a dict nested in obj equals target exactly (status -> count; bools are not counts)."""
    if depth > 12:
        return False
    if isinstance(obj, dict):
        if obj and set(obj) == set(target) and all(isinstance(obj[k], (int, float)) and not isinstance(obj[k], bool)
                                                   and float(obj[k]) == target[k] for k in target):
            return True
        return any(find_map(v, target, depth + 1) for v in obj.values())
    if isinstance(obj, list):
        return any(find_map(v, target, depth + 1) for v in obj)
    return False


def _find_key(obj, key, depth=0):
    """Every value stored under `key` anywhere in a JSON object."""
    out = []
    if depth <= 12:
        items = obj.items() if isinstance(obj, dict) else (enumerate(obj) if isinstance(obj, list) else ())
        for k, v in items:
            if k == key:
                out.append(v)
            out += _find_key(v, key, depth + 1)
    return out


def _t_reports(obj, pattern, depth=0):
    """Counts reported under keys matching `pattern`: a number value (not a rate / share / fraction key), or the n / count / n_* entries of a
    dict value - anywhere in a JSON object."""
    out = []
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    cnt = lambda k: bool(re.match(r'(?i)^(n|count|n_\w+|violations?)$', str(k))) and not re.search(r'(?i)den|tot|all|base|elig|makers|groups', str(k))

    def sub(v, d):          # every count (never a denominator) anywhere in the subtree of a T key (settle-order / window-certain versions)
        if d > 12:
            return []
        if isinstance(v, dict):
            return [float(x) for k, x in v.items() if cnt(k) and num(x)] + [y for x in v.values() if isinstance(x, (dict, list)) for y in sub(x, d + 1)]
        return [y for x in v for y in sub(x, d + 1)] if isinstance(v, list) else []
    if depth <= 12:
        items = obj.items() if isinstance(obj, dict) else (enumerate(obj) if isinstance(obj, list) else ())
        for k, v in items:
            if isinstance(k, str) and re.match(pattern, k):
                if num(v) and not re.search(r'(?i)rate|share|frac|den', k):
                    out.append(float(v))
                else:
                    out += sub(v, depth + 1)
            out += _t_reports(v, pattern, depth + 1)
    return out


def class_maps(obj, out, depth=0):
    """Every reported class-coverage vector: a dict {class: number}, or (pinned lifecycle form) {class: {orders, executions, shares}} expanded
    into one {class: number} per denominator."""
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    if depth <= 12:
        if isinstance(obj, dict):
            if obj and set(obj) <= set(C.CLASSES) and all(num(v) for v in obj.values()):
                out.append({k: float(v) for k, v in obj.items()})
            elif obj and set(obj) <= set(C.CLASSES) and all(isinstance(v, dict) for v in obj.values()):
                for den in sorted(set().union(*[set(v) for v in obj.values()]) & {'orders', 'executions', 'shares'}):
                    m = {k: float(v[den]) for k, v in obj.items() if num(v.get(den))}
                    if m:
                        out.append(m)
            for v in obj.values():
                class_maps(v, out, depth + 1)
        elif isinstance(obj, list):
            for v in obj:
                class_maps(v, out, depth + 1)
    return out


class Acc:
    """Per-check accumulator: evaluated units, failures (first 6 reasons kept), info counters."""
    def __init__(s):
        s.n, s.bad, s.ex, s.info = 0, 0, [], Counter()

    def ok(s, k=1):
        s.n += k

    def fail(s, msg):
        s.n += 1; s.bad += 1
        if len(s.ex) < 6:
            s.ex.append(msg)

    def check(s, nbad, msg):
        (s.fail(msg.format(n=nbad)) if nbad else s.ok())

    def res(s):
        return s.n > 0 and s.bad == 0

    def det(s):
        d = dict(n_eval=s.n, n_fail=s.bad, examples=list(s.ex), **{k: (v.item() if isinstance(v, np.generic) else v) for k, v in s.info.items()})
        d['reason'] = 'never evaluated: no input reached this check' if s.n == 0 else (s.ex[0] if s.bad else '')
        return d


def _call(A, name, fn, *a):
    if A['_only'] is not None and name not in A['_only']:
        return
    try:
        fn(A[name], *a)
    except SystemExit:
        raise
    except Exception as e:     # an exception is a failure WITH its reason, never a pass
        A[name].fail(f'cannot evaluate: {type(e).__name__}: {e}')


# ------------------------------------------------------------------------------------------------------------- populations / load
def book_markets(manifest=None):
    m = json.loads(Path(manifest or BOOK_MANIFEST).read_text())
    out = {d: sorted(int(x) for x in v) for d, v in m['complete'].items()}
    need(set(out) <= set(C.MEASURE_DAYS), 'BOOK manifest names a day outside the measure days')
    return out


def gt_markets(scope=None):
    """GT windows from scope.json (there is no step1 registry for Nov 8). Its 'hours' list is NOT used: it is not clipped to 00-22."""
    s = json.loads(Path(scope or GT_SCOPE).read_text())
    return {C.GT_DAY: sorted({int(w['market']) for w in s['windows']})}, {int(w['market']): int(w['ws']) for w in s['windows']}


def load(out_dir=None, days=None, book=None, gt=None, gt_ws=None, raw_root=None, step1=None, manifest=None, scope=None, gt_censored=None):
    """Builder output paths (contract section 4) + populations. Nothing big is read here."""
    out = Path(out_dir or C.OUT)
    days = tuple(days or (C.ALLOWED_DAYS + (C.NOV8,)))
    for d in days:
        need(d in C.ALLOWED_DAYS or d == C.NOV8, f'day {d} is not a consumed day')
    if book is None:
        book = book_markets(manifest)
    gt_from_receipt, scope_err = gt is None, ''    # no explicit GT list: the extract receipt's selection decides (below)
    if gt is None or gt_ws is None:
        try:
            g, w = gt_markets(scope)
        except (OSError, ValueError, KeyError, TypeError) as e:            # a missing scope must fail its checks closed, never crash the run
            g, w, scope_err = {}, {}, f'scope unreadable: {type(e).__name__}'
        gt, gt_ws = (g if gt is None else gt), (w if gt_ws is None else gt_ws)
    E = dict(out=out, days=days, book={d: v for d, v in book.items() if d in days}, gt={d: v for d, v in gt.items() if d in days},
             gt_ws=dict(gt_ws), raw_root=raw_root, step1=Path(step1 or C.STEP1), P=defaultdict(dict), n_unexpected=0, n_forbidden_named=0,
             _reg={})
    for d in days:
        for k, ext in (('records', 'parquet'), ('legs', 'parquet'), ('tob', 'parquet'), ('inventory', 'json'), ('extract', 'json')):
            if (out / f'{k}_{d}.{ext}').exists():
                E['P'][k][d] = out / f'{k}_{d}.{ext}'
    for sub in ('depth', 'life'):
        if (out / sub).is_dir():
            for e in os.scandir(out / sub):
                m = re.fullmatch(r'(\d{8})_(\d+)(?:_(frames|levels|fills|orders|x_links))?\.(parquet|json)', e.name)
                if not m or (m.group(3) is None) != (m.group(4) == 'json'):
                    E['n_unexpected'] += 1; continue
                if not (m.group(1) in C.ALLOWED_DAYS or m.group(1) == C.NOV8):
                    E['n_forbidden_named'] += 1; continue      # counted, never printed
                if m.group(1) in days:
                    E['P'][m.group(3) or 'life_json'][(m.group(1), int(m.group(2)))] = Path(e.path)
    for k in ECON_T:
        if (out / 'econ' / f'{k}.parquet').exists():
            E['P']['econ'][k] = out / 'econ' / f'{k}.parquet'
    for k, p in (('identities', out / 'econ/identities.json'), ('own_orders', out / 'truth/v2/own_orders.parquet'), ('own_orders', out / 'truth/own_orders.parquet'),
                 ('scores', out / 'truth/v2/scores.json'), ('scores', out / 'truth/scores.json'), ('coverage', out / IFACE['coverage_file'])):
        if p.exists() and k not in E['P']['misc']:      # pinned: truth writes OUT/truth/v2/ (own_orders AND scores.json); v2 preferred
            E['P']['misc'][k] = p
    E['gt_err'], E['gt_reparse'], E['gt_censored'] = scope_err, None, {int(m) for m in (gt_censored or ())}
    if gt_from_receipt and C.GT_DAY in days:     # pinned GT population: ws_population.selected minus boundary_censored markets
        sel, cen, mk = [], set(), out / f'markets_{C.GT_DAY}.parquet'
        try:
            sel = [x for r in ([rdj(E['P']['extract'][C.GT_DAY])] if C.GT_DAY in E['P']['extract'] else []) for v in _find_key(r, 'ws_population')
                   if isinstance(v, dict) for x in (v.get('selected') or [])]
            sel = [int(x if not isinstance(x, dict) else x.get('market', x.get('mid'))) for x in sel]
            if mk.exists():
                Mk = pd.read_parquet(mk, columns=['mid', 'status'])
                cen = set(Mk.mid[hasf(Mk.status, 'boundary_censored')].astype(int))
        except Exception as e:                    # fail closed: the GT population stays the scope windows and the error is reported
            sel, E['gt_err'] = [], f'GT selection unreadable: {type(e).__name__}: {e}'
        if sel:
            keep = [m for m in dict.fromkeys(sel) if m not in cen]
            E['gt'] = {C.GT_DAY: sorted(keep)}
            E['gt_reparse'] = (C.GT_DAY, keep[0]) if keep else None
            E['gt_info'] = dict(selected=len(set(sel)), censored_removed=len(set(sel) & cen), markets_table=mk.exists())
            E['gt_censored'] = set(sel) & cen        # excluded from the GT SCORE only; their depth still counts in the day totals (e2e 4)
        elif not E['gt_err']:
            E['gt_err'] = 'extract receipt holds no ws_population.selected: GT population falls back to every scope window'
    elif C.GT_DAY in E['gt'] and E['gt'][C.GT_DAY]:
        E['gt_reparse'] = (C.GT_DAY, E['gt'][C.GT_DAY][0])
    E['life'] = sorted({(d, m) for g in (E['book'], E['gt']) for d, v in g.items() for m in v})
    # e2e 4: depth tables the extractor wrote for windows outside the scored lifecycle set (e.g. the boundary_censored GT window, or the other
    # BOOK markets of a one-market sub-run) still count in the depth DAY totals; only their status column is read
    E['depth_only'] = sorted({k for t in ('frames', 'levels') for k in E['P'][t]} - set(E['life']))
    return E


def reg(E, day):
    """{mid: (ws_ms, we_ms, V)} from step1/<day>/registry.parquet; GT markets lacking a registry use scope.json ws (V unknown)."""
    if day not in E['_reg']:
        need(day in C.ALLOWED_DAYS or day == C.NOV8, f'day {day} is not a consumed day')
        out, p = {}, Path(E['step1']) / day / 'registry.parquet'
        if p.exists():
            names = pq.read_schema(p).names
            R = pd.read_parquet(p, columns=[c for c in ('mid', 'ws', 'we', 'outcome', 'outcome_seen_ns', 'start_price', 'end_price') if c in names])
            for _, r in R.iterrows():
                seen = r.get('outcome_seen_ns')       # sealed: an outcome first seen in a non-consumed hour (or never) is blanked, never used
                sealed = isnull(seen) or not C.allowed_hour(hour_of_ns(int(seen)))
                tie = not (isnull(r.get('start_price')) or isnull(r.get('end_price'))) and float(r['start_price']) == float(r['end_price'])
                V = np.nan if sealed else (0.5 if tie else {'Up': 1.0, 'Down': 0.0}.get(r.get('outcome'), np.nan))
                out[int(r['mid'])] = (int(round(r['ws'] * 1000)), int(round(r['we'] * 1000)), V)
                E.setdefault('_sealed', set()).update([(day, int(r['mid']))] if sealed else [])
        if day == C.GT_DAY:
            for m, ws in E['gt_ws'].items():
                out.setdefault(m, (ws * 1000, ws * 1000 + 300000, np.nan))
        E['_reg'][day] = out
    return E['_reg'][day]


def rdj(p):
    return json.loads(Path(p).read_text())


def yes_arrays(quote, iset, price):
    """pf_mt_common.yes_space (the definition) applied over unique (quote, indexSet, price) combinations (a left merge keeps row order)."""
    k = pd.DataFrame({'q': np.asarray(quote, dtype=object), 'i': np.asarray(iset, dtype=np.int64), 'p': np.asarray(price, dtype=float)})
    u = k.drop_duplicates().reset_index(drop=True)
    v = [C.yes_space(q, int(i), float(p)) for q, i, p in zip(u.q, u.i, u.p)]
    u['s'], u['y'] = [x[0] for x in v], [x[1] for x in v]
    m = k.merge(u, on=['q', 'i', 'p'], how='left')
    return m.s.to_numpy(dtype=object), m.y.to_numpy(dtype=float)


def fee_value(fee, ftype, iset, V):
    """USD value of a fee to resolution: COLLATERAL as is; SHARES in the leg's own token at its payoff (V for indexSet 1, 1-V for 2)."""
    fee, V, ft = np.asarray(fee, float), np.asarray(V, float), np.asarray(ftype, dtype=object)
    pay = np.where(np.asarray(iset) == 1, V, 1 - V)
    return np.where(fee == 0, 0.0, np.where(ft == 'COLLATERAL', fee, np.where(ft == 'SHARES', fee * pay, np.nan)))    # a zero fee is 0 in any unit


def fee_at_px(fee, ftype, px):
    """USD value of a fee at the fill: COLLATERAL as is; SHARES at the leg's own price px; any other type NaN (fails closed)."""
    fee, px, ft = np.asarray(fee, float), np.asarray(px, float), np.asarray(ftype, dtype=object)
    return np.where(fee == 0, 0.0, np.where(ft == 'COLLATERAL', fee, np.where(ft == 'SHARES', fee * px, np.nan)))


def ptick_unit(price, ptick):
    for K in (100, 1000):
        if len(price) and np.all(np.abs(np.asarray(price, float) * K - np.asarray(ptick, float)) < 1e-6):
            return K
    return None


# ----------------------------------------------------------------------------------------------------------- book / lifecycle math
def book_arrays(FR, LV, keys):
    """USABLE frames sorted by venue time -> (T upd_ms, {(side, ptick): shown shares}); a price absent from a full snapshot shows 0.
    Pinned: only bad_frame / mid_mismatch / out_of_order / off_grid frames are unusable (new_conn / gap / dup / same_ms_changed are usable)."""
    fr = FR[usable(FR.status)].sort_values(['upd_ms', 'recv_ns', 'fid'], kind='stable')
    T = fr.upd_ms.to_numpy(np.int64)
    pos = pd.Series(np.arange(len(fr)), index=fr.fid.to_numpy())
    S = {k: np.zeros(len(T)) for k in keys}
    lv = LV[LV.fid.isin(pos.index)]
    for (sd, pt), g in lv.groupby(['side', 'ptick']):
        if (sd, int(pt)) in S:
            S[(sd, int(pt))][pos.loc[g.fid.to_numpy()].to_numpy()] = g.shares.to_numpy(float)
    return T, S


def units_of(F):
    """(hash, YES side, ptick) -> fill-window summary (lo = -1: unbounded)."""
    U = {}
    for (h, sd, pt), g in F.groupby(['h', 'y_side', 'y_ptick']):
        l, hi, sh = g.lo_ms.to_numpy(np.int64), g.hi_ms.to_numpy(np.int64), g.shares.to_numpy(float)
        U[(h, sd, int(pt))] = dict(hmin=int(hi.min()), hmax=int(hi.max()), lo=l, sh=sh, cum=float(sh.sum()), all_lo=bool((l >= 0).all()),
                                   minlo=int(l.min()), maxlo=int(l[l >= 0].max()) if (l >= 0).any() else -1)
    return U


def h3_scan(U, T, S):
    """H3 at every ok frame u: the units proven resting at their price (a fill surely before: hi <= u; a fill surely after: lo >= u) need
    shown(p, u) >= the sum of their fills with lo >= u. Also returns the capacity posting bound per unit (latest frame u <= every fill's lo
    with shown(p, u) < cumulative fills) and shown-zero evidence (first frame at/after the last fill's hi shows 0 at p and no other unit
    is proven resting there)."""
    viol, members, cap, zero = [], set(), {}, {}
    by_p = defaultdict(list)
    for k, u in U.items():
        by_p[k[1:]].append((k, u))
    for p, us in by_p.items():
        sh, need_ = S[p], np.zeros(len(T) + 1)
        for k, u in us:
            for l, q in zip(u['lo'], u['sh']):
                if l >= 0:
                    a, b = np.searchsorted(T, u['hmin'], 'left'), np.searchsorted(T, l, 'right')
                    if a < b:
                        need_[a] += q; need_[b] -= q
        need_ = np.cumsum(need_)[:-1]
        bad = np.where(sh + TOL_SH < need_)[0]
        if len(bad):
            bt = T[bad]
            mem = [k[0] for k, u in us if ((bt >= u['hmin']) & (bt <= u['maxlo'])).any()]
            members.update(mem)
            viol.append(dict(price=p, n_frames=int(len(bad)), need=float(need_[bad[0]]), shown=float(sh[bad[0]])))
        for k, u in us:
            if u['all_lo']:
                m = np.where((T <= u['minlo']) & (sh < u['cum'] - TOL_SH))[0]
                if len(m):
                    cap[k] = int(T[m[-1]])
            j = int(np.searchsorted(T, u['hmax'], 'left'))
            zero[k] = bool(j < len(T) and sh[j] <= TOL_SH and not any(k2 != k and u2['hmin'] <= T[j] <= u2['maxlo'] for k2, u2 in us))
    return viol, members, cap, zero


def priority_evidence(Ld, Lall=None):
    """Amendment 1: makers[] is NOT fill order and mk_idx is never evidence. Price priority, per record, in YES space from the taker's best:
    makers at a price strictly better for the taker than the record's worst maker price are exhausted there; of the k worst-price makers at
    most one keeps remaining > 0 - if exactly one of them fills again later (same hash, later settle_ms) the other k-1 are exhausted there
    and that one had them ahead (ahead_lo = their shares); otherwise ahead_lo = 0. `Lall` (default Ld) supplies the later fills.
    Returns ev {(sid, mk_idx)}, ahead {(sid, mk_idx): ahead_lo}, T1 (strictly-better exhausted legs / hashes that fill again),
    T2 (records with >= 2 worst-price makers that both fill again), makers[] order counts over multi-price records."""
    Lall = Ld if Lall is None else Lall
    last = Lall.groupby('h').settle_ms.max()
    ev, ahead, t1_legs, t1_h, t2, order = set(), {}, 0, set(), 0, Counter()
    for sid, g in Ld.groupby('sid', sort=False):
        buy = g.tk_y_side.iloc[0] == 'bid'
        p = g.y_ptick.to_numpy(np.int64); idx = g.mk_idx.to_numpy(np.int64); h = g.h.to_numpy(); sh = g.shares.to_numpy(float)
        t = int(g.settle_ms.iloc[0])
        again = np.array([last.get(x, t) > t for x in h], dtype=bool)
        worst = p.max() if buy else p.min()
        better = (p < worst) if buy else (p > worst)
        for i in np.flatnonzero(better):
            ev.add((sid, int(idx[i])))
            if again[i]:
                t1_legs += 1; t1_h.add(h[i])
        w = np.flatnonzero(~better)
        if len(w) >= 2:
            re_ = [i for i in w if again[i]]
            if len(re_) == 1:
                ev.update((sid, int(idx[i])) for i in w if i != re_[0])
                ahead[(sid, int(idx[re_[0]]))] = float(sh[[i for i in w if i != re_[0]]].sum())
            elif len(re_) >= 2:
                t2 += 1
        if len(set(p)) >= 2:
            o = p[np.argsort(idx)]
            d = np.diff(o)
            best_first = np.all(d >= 0) if buy else np.all(d <= 0)
            worst_first = np.all(d <= 0) if buy else np.all(d >= 0)
            order['best_first' if best_first else ('worst_first' if worst_first else 'neither')] += 1
    return dict(ev=ev, ahead=ahead, t1_legs=t1_legs, t1_hashes=len(t1_h), t2=t2, order=dict(order))


def conflict_hashes(Ld):
    """H1: hashes mapping to more than one (signer, YES side, YES price)."""
    x = Ld.assign(sg=Ld.mk_signer.str.lower())[['h', 'sg', 'y_side', 'y_price']].drop_duplicates()
    n = x.groupby('h').size()
    return set(n[n > 1].index)


# --------------------------------------------------------------------------------------------------------------- per-day checks
DAY_CHECKS = ['outputs_present', 'schema', 'status_present', 'status_counts_reported', 'records_unique_key', 'legs_complete',
              'amounts_reconcile', 'yes_space_mapping', 'clock_fields', 'day_assignment', 'legs_phase_clock', 'boundary_censored']
MARKET_CHECKS = ['fills_complete', 'fill_hi_rule', 'fill_lo_status', 'fill_time_order', 'fill_lo_frame', 'orders_complete',
                 'orders_cumulative', 'H1_identity', 'H2_presence', 'H3_overfill_flagged', 'posting_capacity_bound',
                 'posting_before_first_fill', 'removal_after_last_fill', 'removal_lo_bound', 'removal_class_evidence',
                 'class_scenario_consistency', 'size_bounds', 'priority_exhaustion', 'priority_tests_T1_T2', 'priority_ahead_lo',
                 'levels_frames_consistent', 'tob_from_levels']
GLOBAL_CHECKS = ['coverage_arithmetic', 'econ_conservation', 'econ_pnl_recompute', 'identities_exante', 'gt_scores']
ALL_CHECKS = DAY_CHECKS + MARKET_CHECKS + GLOBAL_CHECKS


def schema_ok(a, kind, path, label, want=None):
    sch = pq.read_schema(path)
    want = want or C.SCHEMAS[kind]
    miss = [c for c in want if c not in sch.names]
    bad = [f'{c}:{sch.field(c).type}' for c, t in want.items() if c in sch.names and sch.field(c).type not in ARROW[t]]
    extra = [c for c in sch.names if c not in want and not c.startswith('x_')]
    (a.fail(f'schema {kind} {label}: missing {miss} wrong dtype {bad} extra non-x_ {extra}') if miss or bad or extra else a.ok())


def declared_order(receipts, table):
    """The extractor's FIXED flag order for one table, declared in extract_<day>.json under status_orders[table] (pinned); legacy fallback:
    FLAGS + WS_FLAGS + MK_FLAGS. None when undeclared (status checks then fail closed)."""
    so = next((v for r in receipts for v in _find_key(r, 'status_orders') if isinstance(v, dict)), None)
    if so is not None:
        v = so.get(table)
        return [str(f) for f in v] if isinstance(v, list) else None
    lists = [next((v for r in receipts for v in _find_key(r, k) if isinstance(v, list)), None) for k in ('FLAGS', 'WS_FLAGS', 'MK_FLAGS')]
    return None if all(x is None for x in lists) else list(dict.fromkeys(str(f) for x in lists for f in (x or [])))


def status_ok(a, kind, df, label, order='sorted'):
    """Non-empty statuses; '+'-joined flag syntax (no empty flag, no repeat, 'ok' never joined). Flag order: lifecycle tables join SORTED
    flags (order='sorted'); extract tables follow the extractor's declared order (order = that list; None = undeclared -> refused)."""
    s = df['status']
    a.check(int(s.isna().sum() + (~s.map(lambda v: isinstance(v, str) and v != '')).sum()), f'status {kind} {label}: {{n}} null or empty status values')
    if order is None:
        a.fail(f'status {kind} {label}: the extract receipt declares no flag order (FLAGS / WS_FLAGS / MK_FLAGS)')
    pos = {f: i for i, f in enumerate(order)} if isinstance(order, list) else None
    bad, wrong = [], []
    for v in s.dropna().unique():
        fl = v.split('+')
        if '' in fl or len(set(fl)) != len(fl) or ('ok' in fl and len(fl) > 1):
            bad.append(v); continue
        if fl == [C.STATUS_OK]:
            continue
        if order == 'sorted' and fl != sorted(fl):
            wrong.append(v)
        elif pos is not None and (any(f not in pos for f in fl) or [pos[f] for f in fl] != sorted(pos[f] for f in fl)):
            wrong.append(v)
    a.check(len(bad), f'status {kind} {label}: {{n}} statuses with bad flag syntax (empty / repeated flag, or ok joined with other flags)')
    if order == 'sorted':
        a.check(len(wrong), f'status {kind} {label}: {{n}} statuses whose flags are not sorted (lifecycle tables join SORTED flags)')
    elif pos is not None:
        a.check(len(wrong), f'status {kind} {label}: {{n}} statuses with an undeclared flag or not in the declared flag order of the extract receipt')


def counts_ok(a, kind, df, receipts, label):
    """Status counts must be reported: a receipt map equal to the per-flag counts (amendment 1) or to the per-status counts."""
    tgts = [flag_counts(df['status']), {str(k): int(v) for k, v in df['status'].value_counts().items()}]
    if not receipts:
        a.fail(f'status counts {kind} {label}: no receipt JSON holds them')
    elif any(find_map(r, t) for r in receipts for t in tgts) or (kind == 'orders' and all(
            any(find_map(r, t) for r in receipts for t in (flag_counts(g['status']), {str(k): int(v) for k, v in g['status'].value_counts().items()}))
            for _, g in df.groupby('scenario'))):
        a.ok()
    else:
        a.fail(f'status counts {kind} {label}: no receipt map equals the per-flag counts {tgts[0]} (dropped or unreported rows)')


def day_checks(A, E, day, D, G):
    R, L, rg = D['records'], D['legs'], reg(E, day)
    rec_j = D['rec_j'] = [rdj(E['P'][k][day]) for k in ('extract', 'inventory') if day in E['P'][k]]
    D['decl'] = {k: declared_order(rec_j, k) for k in ('records', 'legs', 'tob', 'frames', 'levels')}
    a = A['outputs_present']
    for k in ('records', 'legs', 'extract', 'inventory') + (('tob',) if day != C.ID_DAY else ()):
        (a.ok() if day in E['P'][k] else a.fail(f'{k} output missing for {day}'))
    if R is None or L is None:
        for n in DAY_CHECKS[1:]:
            A[n].fail(f'cannot evaluate {day}: records/legs output missing')
        return
    TS = pd.read_parquet(E['P']['tob'][day], columns=['status']) if day in E['P']['tob'] else None
    _call(A, 'schema', lambda a: [schema_ok(a, k, E['P'][k][day], day) for k in ('records', 'legs', 'tob') if day in E['P'][k]])
    _call(A, 'status_present', lambda a: [status_ok(a, k, df, day, D['decl'].get(k)) for k, df in (('records', R), ('legs', L), ('tob', TS)) if df is not None])
    _call(A, 'status_counts_reported', lambda a: [counts_ok(a, k, df, rec_j, day) for k, df in (('records', R), ('legs', L), ('tob', TS))
                                                 if df is not None])

    def _uniq(a):
        d = R[R.duplicated('sid', keep=False)]
        a.check(int(d[~hasf(d.status, 'conflict')].sid.nunique()), f'records {day}: {{n}} settlementIds duplicated without status conflict')
        a.check(int((L.duplicated(['sid', 'mk_idx'], keep=False) & ~L.sid.isin(d.sid)).sum()),
                f'legs {day}: {{n}} duplicate (sid, mk_idx) rows outside conflict records')
    _call(A, 'records_unique_key', _uniq)

    def _legs(a):        # column-wise maps, no full-table merge (a real day is ~0.75M legs)
        j = pd.concat([R.groupby('sid').n_makers.sum().rename('need'), L.groupby('sid').size().rename('have')], axis=1).fillna(0)
        a.check(int((j.need != j.have).sum()), f'legs {day}: {{n}} records whose leg count != n_makers (dropped/extra/orphan legs)')
        cols = ['day', 'mid', 'tx', 'settle_ms', 'exec_ms', 'first_recv_ns', 'n_makers', 'tk_hash', 'tk_signer', 'tk_y_side']
        one = R.loc[~R.sid.duplicated(keep=False), ['sid'] + cols].set_index('sid')
        inn = L.sid.isin(one.index).to_numpy()
        sid = L.sid[inn]
        for c in cols:
            x, y = L.loc[inn, c], sid.map(one[c])
            if c in ('tx', 'tk_hash', 'tk_signer'):
                x, y = x.str.lower(), y.str.lower()
            a.check(int(neq(x, y).sum()), f'legs {day}: {{n}} legs whose {c} differs from its record')
        g = L.loc[inn, ['sid', 'mk_idx']].groupby('sid').mk_idx.agg(['min', 'max', 'count', 'nunique']).join(one.n_makers)
        a.check(int(((g['min'] != 0) | (g['max'] != g.n_makers - 1) | (g['count'] != g.n_makers) | (g['nunique'] != g.n_makers)).sum()),
                f'legs {day}: {{n}} records whose mk_idx are not 0..n_makers-1')
    _call(A, 'legs_complete', _legs)

    def _amounts(a):
        okm = (R.status == C.STATUS_OK).to_numpy()
        s = L.groupby('sid').shares.sum().reindex(R.sid[okm]).to_numpy()
        amt = R.amount_filled.to_numpy()[okm]; tol = TOL_SH * np.maximum(1, np.abs(amt))
        a.check(int((np.abs(R.tk_amount.to_numpy()[okm] - amt) > tol).sum()), f'records {day}: {{n}} status-ok records with taker.amount != amountFilled')
        a.check(int((~(np.abs(s - amt) <= tol)).sum()), f'records {day}: {{n}} status-ok records with sum(makers.amount) != amountFilled')
        a.check(int((np.abs(R.tk_price.to_numpy()[okm] - R.price_executed.to_numpy()[okm]) > 1e-12).sum()),
                f'records {day}: {{n}} status-ok records with taker.price != priceExecuted')
        a.check(int((R.n_makers.to_numpy()[okm] < 1).sum()), f'records {day}: {{n}} status-ok records with no maker')
        # declared rule: priceExecuted vs the maker-weighted YES price is an econ DIAGNOSTIC (px_gap), never a record status - counted only
        pbar = (L.y_price * L.shares).groupby(L.sid).sum() / L.groupby('sid').shares.sum()
        tp = np.where(R.tk_index_set.to_numpy() == 1, R.price_executed.to_numpy(), 1 - R.price_executed.to_numpy())
        a.info['records_price_gap_diagnostic'] += int((~(np.abs(R.sid.map(pbar).to_numpy(float) - tp) <= PX_TOL)).sum()); a.info['records'] += len(R)
    _call(A, 'amounts_reconcile', _amounts)

    def _yes(a):
        ys, yp = yes_arrays(L.mk_quote, L.mk_index_set, L.mk_price)
        a.check(int((ys != L.y_side.to_numpy()).sum() + (np.abs(yp - L.y_price.to_numpy()) > 1e-9).sum()),
                f'legs {day}: {{n}} legs whose (y_side, y_price) != yes_space(quote, indexSet, price)')
        K = ptick_unit(L.y_price.to_numpy(), L.y_ptick.to_numpy())
        if K is None:
            a.fail(f'legs {day}: y_ptick is not round(y_price*K) for a single K in (100, 1000)')
        else:
            a.ok(); G['K'].add(K)
        ts, _ = yes_arrays(R.tk_quote, R.tk_index_set, R.tk_price)
        a.check(int((ts != R.tk_y_side.to_numpy()).sum()), f'records {day}: {{n}} records whose tk_y_side != yes_space(taker)')
        a.check(int((L.y_side == L.tk_y_side).sum()), f'legs {day}: {{n}} maker legs on the same YES side as their taker')
    _call(A, 'yes_space_mapping', _yes)

    def _clock(a):
        for name, df in (('records', R), ('legs', L)):
            ms = {s: C.uuid7_ms(s) for s in df.sid.unique()}
            a.check(int((df.sid.map(ms) != df.settle_ms).sum()), f'{name} {day}: {{n}} rows whose settle_ms != UUIDv7 ms of settlementId')
            a.check(int(((df.first_recv_ns <= 0) | (df.exec_ms <= 0)).sum()), f'{name} {day}: {{n}} rows with non-positive first_recv_ns / exec_ms')
        a.check(int((R.n_recv < 1).sum()), f'records {day}: {{n}} rows with n_recv < 1')
    _call(A, 'clock_fields', _clock)

    def _dayas(a):
        a.check(int((R.day != day).sum() + (L.day != day).sum()), f'{day}: {{n}} rows whose day column != the file day')
        for mid in R.mid.unique():
            r = rg.get(int(mid))
            if r is None:
                a.fail(f'{day}: market {mid} is not in the day registry (window start unknown)')
            else:
                a.check(int(dt.datetime.fromtimestamp(r[0] / 1000, UTC).strftime('%Y%m%d') != day), f'{day}: market {mid} window starts on another UTC day')
    _call(A, 'day_assignment', _dayas)

    def _phase(a):
        k = L.loc[(L.status == C.STATUS_OK) & L.mid.isin(list(rg)), ['mid', 'settle_ms', 'exec_ms', 'first_recv_ns', 't_rel_s', 'phase']]
        a.info['legs_without_registry'] += int((~L.mid.isin(list(rg))).sum())
        if not len(k):
            a.fail(f'legs {day}: no status-ok legs with a registry window'); return
        ws = k.mid.map({m: v[0] for m, v in rg.items()}).to_numpy(np.int64)
        anchors = dict(settle_ms=k.settle_ms.to_numpy(np.int64), exec_ms=k.exec_ms.to_numpy(np.int64),
                       hi_ms=np.minimum(k.settle_ms.to_numpy(np.int64), k.first_recv_ns.to_numpy(np.int64) // 10 ** 6))
        hit = [n for n, x in anchors.items() if np.all(np.abs((x - ws) / 1000. - k.t_rel_s.to_numpy()) <= 1e-6)]
        if not hit:
            a.fail(f'legs {day}: t_rel_s equals none of (settle_ms | exec_ms | hi_ms) - ws')
        else:
            a.ok(); G['anchor'].add(hit[0])
        iv = sorted((g.t_rel_s.min(), g.t_rel_s.max(), ph) for ph, g in k.groupby('phase'))
        ov = [(iv[i][2], iv[i + 1][2]) for i in range(len(iv) - 1) if iv[i][1] > iv[i + 1][0]]
        (a.fail(f'legs {day}: phases overlap in t_rel_s {ov}') if ov else a.ok())
    _call(A, 'legs_phase_clock', _phase)

    def _bound(a):
        """Certain implications only (the builder's margin / pre_recv are its own declared parameters): a window ending in a non-consumed
        hour (upper) or starting right after one (lower: pre-window receipts) must be censored; a censored market with no non-consumed hour
        within 2 h of its window is a false flag."""
        for mid, g in R.groupby('mid'):
            r = rg.get(int(mid))
            if r is None:
                a.fail(f'{day}: market {mid} has no window: boundary status unverifiable'); continue
            up, low = (not C.allowed_hour(stems_between(t, t)[0]) for t in (r[1], r[0] - 1))
            near = any(not C.allowed_hour(s) for s in stems_between(r[0] - 7200000, r[1] + 7200000))
            fl = hasf(g.status, 'boundary_censored')
            a.info['boundary_flagged_markets'] += int(fl.any())
            if up or low:
                a.info['boundary_certain_markets'] += 1
                a.check(int((~fl).sum()), f'{day}: market {mid} {"ends in" if up else "starts right after"} a non-consumed hour but {{n}} records are not boundary_censored')
            elif fl.any() and not near:
                a.fail(f'{day}: market {mid} is boundary_censored with no non-consumed hour within 2 h of its window (false flag)')
            else:
                a.ok()
        for k in ('lag_p999_all_ms',):
            v = [x for r_ in rec_j for x in _find_key(r_, k)]
            a.info[f'reported_{k}'] = v[0] if v else None
    _call(A, 'boundary_censored', _bound)


# ------------------------------------------------------------------------------------------------------------ per-market checks
def market_checks(A, E, day, mid, D, G, TB):
    """Fail closed: an exception in the shared per-market computation fails every market check with its reason (never a crash/pass)."""
    try:
        _market_checks(A, E, day, mid, D, G, TB)
    except SystemExit:
        raise
    except Exception as e:
        for n in MARKET_CHECKS:
            A[n].fail(f'cannot evaluate {day}_{mid}: {type(e).__name__}: {e}')


def _market_checks(A, E, day, mid, D, G, TB):
    P, lab = E['P'], f'{day}_{mid}'
    a = A['outputs_present']
    for k in ('fills', 'orders', 'life_json', 'frames', 'levels'):
        (a.ok() if (day, mid) in P[k] else a.fail(f'{k} output missing for BOOK/GT market {lab}'))
    if any((day, mid) not in P[k] for k in ('fills', 'orders', 'frames', 'levels')) or D['legs'] is None:
        for n in MARKET_CHECKS:
            A[n].fail(f'cannot evaluate {lab}: lifecycle / depth / legs outputs missing')
        return
    F, O = pd.read_parquet(P['fills'][(day, mid)]), pd.read_parquet(P['orders'][(day, mid)])
    FR, LV = pd.read_parquet(P['frames'][(day, mid)]), pd.read_parquet(P['levels'][(day, mid)])
    life_j = [rdj(P['life_json'][(day, mid)])] if (day, mid) in P['life_json'] else []
    Ld = D['legs'][D['legs'].mid == mid].copy()
    if 'status' not in LV and 'x_status' in LV:          # pinned: lifecycle reads 'status' or legacy 'x_status' on levels (schema still flags it)
        LV = LV.rename(columns={'x_status': 'status'})
    for k, df in (('fills', F), ('orders', O), ('frames', FR), ('levels', LV)):
        _call(A, 'schema', lambda a, k=k: schema_ok(a, k, P[k][(day, mid)], lab))
        if 'status' in C.SCHEMAS[k]:      # lifecycle tables: SORTED flags; depth tables are extract outputs: the declared order
            _call(A, 'status_present', lambda a, k=k, df=df: status_ok(a, k, df, lab, 'sorted' if k in ('fills', 'orders') else (D['decl'] or {}).get(k)))
    _call(A, 'status_counts_reported', lambda a: [counts_ok(a, k, df, life_j + D['rec_j'], lab) for k, df in (('fills', F), ('orders', O))])

    def _depth_counts(a):      # pinned: depth counts may be DAY totals in the extract receipt (checked at day end) + frames_excluded per market
        for k, df in (('frames', FR), ('levels', LV)):
            tg = [flag_counts(df['status']), {str(x): int(v) for x, v in df['status'].value_counts().items()}]
            found = any(find_map(r, t) for r in life_j + D['rec_j'] for t in tg)
            (a.ok() if found else None)
            D['depth_tot'][k][0].update(tg[0]); D['depth_tot'][k][1].update(tg[1]); D['depth_pm'][k].append(found)    # day totals: EVERY depth market
        fe = [v for r in life_j for v in _find_key(r, 'frames_excluded')]
        if fe:
            un = ~usable(FR.status)                    # pinned: frames_excluded counts only UNUSABLE frames
            nx = int(un.sum()); fx = {x: v for x, v in flag_counts(FR.status[un]).items() if x in UNUSABLE_FRAME_FLAGS}
            v = fe[0]
            ok_ = (isinstance(v, (int, float)) and not isinstance(v, bool) and v == nx) or (isinstance(v, dict) and (v == fx or find_map(v, fx) if fx else not any(
                isinstance(x, (int, float)) and x for x in v.values())))
            (a.ok() if ok_ else a.fail(f'lifecycle summary {lab}: frames_excluded {v} != the {nx} unusable frames'))
    _call(A, 'status_counts_reported', _depth_counts)
    if 'status' in LV:
        LV = LV[usable(LV.status)]                 # unusable levels never enter shown-size arithmetic
    F['h'], O['h'], Ld['h'] = F.mk_hash.str.lower(), O.mk_hash.str.lower(), Ld.mk_hash.str.lower()
    Ld['hi'] = np.minimum(Ld.settle_ms.to_numpy(np.int64), Ld.first_recv_ns.to_numpy(np.int64) // 10 ** 6)
    hard = 'x_lo_hard_ms' in F       # pinned (lifecycle fix): every observed / constrained claim is bounded by x_lo_hard_ms; lo_ms only refines
    Fb = F.assign(lo_ms=F['x_lo_hard_ms'].astype(np.int64)) if hard else F.iloc[0:0]      # hard-bounded fills (empty -> hard checks fail closed)
    nohard = lambda a: a.fail(f'fills {lab}: x_lo_hard_ms absent: hard claim bounds unverifiable')
    fm = F.merge(Ld[['sid', 'mk_idx', 'h', 'y_side', 'y_ptick', 'shares', 'hi', 'settle_ms', 'first_recv_ns']], on=['sid', 'mk_idx'], how='outer', suffixes=('', '_l'),
                 indicator=True)
    both = fm[fm._merge == 'both']
    clean = only_flags(Ld.status, {C.STATUS_OK} | EXTERNAL_STATUSES)        # record clean (or only externally censored)
    Rd = D['records'][D['records'].mid == mid].drop_duplicates('sid')
    rec_ok = (np.abs(Rd.tk_amount - Rd.amount_filled) <= TOL_SH * np.maximum(1, Rd.amount_filled)) & \
             (np.abs(Rd.sid.map(Ld.groupby('sid').shares.sum()) - Rd.amount_filled) <= TOL_SH * np.maximum(1, Rd.amount_filled))
    recon = Ld.sid.isin(Rd.sid[rec_ok.to_numpy()]).to_numpy()                # its kept variant reconciles exactly (any flags)
    cl_keys = set(zip(Ld.sid[clean], Ld.mk_idx[clean]))
    Fv = F[np.array([k in cl_keys for k in zip(F.sid, F.mk_idx)], dtype=bool)]       # strict: quantities that may REQUIRE a flag (H3)

    def _complete(a):
        nl, nf = int((fm._merge == 'right_only').sum()), int((fm._merge == 'left_only').sum())
        (a.fail(f'fills {lab}: {nl} maker legs have no fill row (dropped), {nf} fill rows have no leg') if nl or nf else a.ok())
        a.check(int(((both.h != both.h_l) | (both.y_side != both.y_side_l) | (both.y_ptick != both.y_ptick_l) |
                     ~np.isclose(both.shares, both.shares_l, rtol=1e-12, atol=TOL_SH)).sum()),
                f'fills {lab}: {{n}} fill rows whose hash/side/ptick/shares differ from the leg')
        a.check(int(F.duplicated(['sid', 'mk_idx']).sum()), f'fills {lab}: {{n}} duplicate (sid, mk_idx) rows')
    _call(A, 'fills_complete', _complete)
    def _hi(a):
        a.check(int((both.hi_ms != both.hi).sum()), f'fills {lab}: {{n}} rows with hi_ms != min(settle_ms, first_recv_ns//1e6)')
        if not hard:
            nohard(a); return
        rh = (both.first_recv_ns.to_numpy(np.int64) // 10 ** 6) < both.settle_ms.to_numpy(np.int64)      # receipt-hi fill
        exp_ = np.where(rh, -1, both.hi.to_numpy(np.int64) - C.MAX_LAG_MS)
        a.check(int((both.x_lo_hard_ms.to_numpy(np.int64) != exp_).sum()), f'fills {lab}: {{n}} rows with x_lo_hard_ms != hi_ms - MAX_LAG_MS (-1 for receipt-hi fills)')
    _call(A, 'fill_hi_rule', _hi)
    _call(A, 'fill_lo_status', lambda a: a.check(int(((F.lo_ms < 0).to_numpy() != hasf(F.status, 'fill_lo_unbounded')).sum()),
                                                 f'fills {lab}: {{n}} rows where a negative (unbounded) lo_ms and the flag fill_lo_unbounded disagree'))

    def _order(a):
        inv = F[(F.lo_ms >= 0) & (F.lo_ms > F.hi_ms)]
        a.info['flagged_inversions'] += int((inv.status != C.STATUS_OK).sum())
        a.check(int((inv.status == C.STATUS_OK).sum()), f'fills {lab}: {{n}} status-ok fills with lo_ms > hi_ms (time inversion)')
    _call(A, 'fill_time_order', _order)

    def _loframe(a):
        k = F[F.lo_ms >= 0]
        a.info['bounded_lo'] += len(k)
        if not len(k):
            a.ok(); return
        T = np.sort(FR[usable(FR.status)].upd_ms.to_numpy(np.int64)); ft = dict(zip(FR.fid, FR.upd_ms))
        evc = 'x_evidence_fid' if 'x_evidence_fid' in k else 'f_after'        # pinned: the evidence frame is x_evidence_fid
        a.info[f'evidence_column_{evc}'] += 1
        tb, ta = k.f_before.map(ft), k[evc].map(ft)
        btw = [int(np.searchsorted(T, y, 'left') - np.searchsorted(T, x, 'right')) if not (isnull(x) or isnull(y)) else 1 for x, y in zip(tb, ta)]
        a.check(int(neq(tb, k.lo_ms).sum()), f'fills {lab}: {{n}} bounded fills whose lo_ms != upd_ms of frame f_before')
        a.check(int((ta.isna() | (ta < tb)).sum()), f'fills {lab}: {{n}} bounded fills whose evidence frame is missing or before f_before')
        a.check(sum(n > 0 for n in btw), f'fills {lab}: {{n}} bounded fills with a usable frame strictly between f_before and the evidence frame')
        a.check(int((~k.t_method.map(lambda v: isinstance(v, str) and v != '')).sum()), f'fills {lab}: {{n}} bounded fills with an empty t_method')
        hi_ = k.hi_ms.to_numpy(np.int64); jx = np.searchsorted(T, hi_, 'left')       # pinned window: [hi - 2000 ms, first frame >= hi]
        endw = np.full(len(k), np.iinfo(np.int64).max, dtype=np.int64)
        if len(T):
            m_ = jx < len(T); endw[m_] = T[jx[m_]]
        taf = ta.fillna(-1).to_numpy(float)
        a.check(int(((taf < hi_ - 2000) | (taf > endw)).sum()), f'fills {lab}: {{n}} bounded fills whose evidence frame lies outside [hi - 2000 ms, first frame >= hi]')
    _call(A, 'fill_lo_frame', _loframe)

    def _ocomp(a):
        (a.ok() if set(O.scenario) == set(C.SCENARIOS) else a.fail(f'orders {lab}: scenarios {sorted(set(O.scenario))} != {sorted(C.SCENARIOS)}'))
        a.check(int(O.duplicated(['scenario', 'h']).sum()), f'orders {lab}: {{n}} duplicate (scenario, hash) rows')
        hs = set(F.h)
        for s, g in O.groupby('scenario'):
            d1, d2 = hs - set(g.h), set(g.h) - hs
            (a.fail(f'orders {lab} {s}: {len(d1)} filled hashes missing (dropped), {len(d2)} hashes without fills') if d1 or d2 else a.ok())
        a.check(int((O.mid != mid).sum()), f'orders {lab}: {{n}} rows with another mid')
    _call(A, 'orders_complete', _ocomp)

    agg_all = F.groupby('h').agg(n=('shares', 'size'), cum=('shares', 'sum'), hmin=('hi_ms', 'min'), hmax=('hi_ms', 'max'))
    agg_ok = Fv.groupby('h').agg(n=('shares', 'size'), cum=('shares', 'sum'), hmin=('hi_ms', 'min'), hmax=('hi_ms', 'max'))

    def _cum(a):
        def nbad(ag):
            j = O.join(ag, on='h')
            return int(((j.n_fills != j.n) | ~np.isclose(j.cum_shares, j.cum, atol=TOL_SH, rtol=1e-12) | (j.first_hi_ms != j.hmin) |
                        (j.last_hi_ms != j.hmax)).sum())
        n1, n2 = nbad(agg_all), nbad(agg_ok)
        if n1 == 0 or n2 == 0:
            a.ok(); a.info['mode_all_fills' if n1 == 0 else 'mode_ok_fills'] += 1
        else:
            a.fail(f'orders {lab}: {n1} rows whose n_fills/cum_shares/first_hi_ms/last_hi_ms differ from their fills')
    _call(A, 'orders_cumulative', _cum)

    conflict_h = conflict_hashes(Ld) | (set(O.h) & D['xm'])

    def _h1(a):
        ic = hasf(O.status, 'identity_conflict')
        a.check(int(O[O.h.isin(conflict_h).to_numpy() & ~ic].h.nunique()),
                f'orders {lab}: {{n}} hashes with >1 (signer, YES side, price) or market not labelled identity_conflict (merged silently)')
        a.check(int(O[ic & ~O.h.isin(conflict_h).to_numpy()].h.nunique()),
                f'orders {lab}: {{n}} identity_conflict labels without a conflicting identity (false flag)')
    _call(A, 'H1_identity', _h1)

    oks = O[O.status == C.STATUS_OK].join(agg_all, on='h')
    oks = oks.join(Fb[Fb.lo_ms >= 0].groupby('h').lo_ms.max().rename('Lmax'), on='h').join(Fb.groupby('h').lo_ms.min().rename('Lmin'), on='h')  # HARD bounds
    oks['Lmax'] = oks.Lmax.fillna(-1).astype(np.int64)

    def _h2(a):          # pinned: the CERTAIN presence span [min hi, max x_lo_hard] over the order's fills, every row; -1/-1 when empty (always one fill)
        if not hard:
            nohard(a); return
        OO = O.join(agg_all, on='h').join(Fb[Fb.lo_ms >= 0].groupby('h').lo_ms.max().rename('Lmx'), on='h')
        Lmx = OO.Lmx.fillna(-1).to_numpy(float); hm = OO.hmin.to_numpy(float)
        empty = (OO.n.fillna(0).to_numpy() <= 1) | (Lmx < 0) | (hm > Lmx)
        b = (OO.persist_lo_ms.to_numpy() != np.where(empty, -1, hm)) | (OO.persist_hi_ms.to_numpy() != np.where(empty, -1, Lmx))
        b &= ~hasf(OO.status, 'identity_conflict')          # pinned: identity_conflict rows are skipped (their fills span several identities)
        a.check(int(b.sum()), f'orders {lab}: {{n}} rows whose persistence window contradicts the certain span [min hi, max lo] (-1/-1 when empty)')
        a.check(int(((O.persist_class == C.OBSERVED) & (O.persist_lo_ms < 0)).sum()),
                f'orders {lab}: {{n}} rows labelled observed persistence with an empty certain span (-1/-1)')
    _call(A, 'H2_presence', _h2)

    rc_keys = set(zip(Ld.sid[recon], Ld.mk_idx[recon]))
    U = units_of(Fb[np.array([k in cl_keys for k in zip(Fb.sid, Fb.mk_idx)], dtype=bool)])           # strict, HARD-bounded
    Up = units_of(Fb[np.array([k in rc_keys for k in zip(Fb.sid, Fb.mk_idx)], dtype=bool)])          # permissive, HARD-bounded
    T, S = book_arrays(FR, LV, {k[1:] for k in U} | {k[1:] for k in Up})
    viol, members, cap, _z = h3_scan(U, T, S)              # strict: violations / capacity bounds that REQUIRE a flag or a bound
    _v, _m, cap_p, zero = h3_scan(Up, T, S)                # permissive: shown-zero evidence and capacity bounds that ALLOW a claim
    capd, capp, zh = {k[0]: v for k, v in cap.items()}, {k[0]: v for k, v in cap_p.items()}, {k[0]: v for k, v in zero.items()}

    def _h3(a):
        if not hard:
            nohard(a); return
        a.info['h3_violating_prices'] += len(viol)
        n = int(O[O.h.isin(members) & (O.status == C.STATUS_OK)].h.nunique())
        (a.fail(f'orders {lab}: H3 overfill - shown size < later fills of proven-resting orders at {viol[0]["price"]} (need {viol[0]["need"]:.4g} > '
                f'shown {viol[0]["shown"]:.4g}); {n} member hashes still status ok') if n else a.ok())
    _call(A, 'H3_overfill_flagged', _h3)

    def _cap(a):
        if not hard:
            nohard(a); return
        a.info['capacity_bounds'] += len(cap)
        k = oks[oks.h.isin(capd)]
        a.check(int((k.post_lo_ms < k.h.map(capd)).sum()), f'orders {lab}: {{n}} ok rows with post_lo_ms below the H3 capacity bound '
                                                           f'(latest frame before the first fill with shown < cumulative fills)')
        # declared rule: the lifecycle's JOINT capacity bound is valid - soundness only: at least as tight as the single-order bound (above)
        # and never after the first fill's hi
        a.check(int(((oks.post_method == 'capacity') & ((oks.post_lo_ms < 0) | (oks.post_lo_ms > oks.hmin))).sum()),
                f'orders {lab}: {{n}} rows claim post_method capacity with an unbounded bound or one after the first fill hi')
    _call(A, 'posting_capacity_bound', _cap)
    _call(A, 'posting_before_first_fill', lambda a: a.check(int(((oks.post_hi_ms < 0) | (oks.post_hi_ms > oks.hmin) |
                                                                  ((oks.post_lo_ms >= 0) & (oks.post_lo_ms > oks.post_hi_ms))).sum()),
                                                             f'orders {lab}: {{n}} ok rows with post_hi_ms unbounded or after the first fill hi, or post_lo > post_hi'))
    _call(A, 'removal_after_last_fill', lambda a: nohard(a) if not hard else a.check(int(((oks.removal_hi_ms >= 0) & (oks.removal_hi_ms < oks.Lmax)).sum()),
                                                           f'orders {lab}: {{n}} ok rows with a fill after the removal window (removal_hi_ms < the last fill x_lo_hard_ms)'))
    _call(A, 'removal_lo_bound', lambda a: nohard(a) if not hard else a.check(int(((oks.removal_lo_ms < oks.Lmax) | ((oks.removal_lo_ms >= 0) & (oks.removal_hi_ms >= 0) &
                                                                                           (oks.removal_lo_ms > oks.removal_hi_ms))).sum()),
                                                    f'orders {lab}: {{n}} ok rows with removal_lo_ms < the last fill lo_ms (hard: x_lo_hard_ms), or removal_lo > removal_hi'))

    pe_req = priority_evidence(Ld[clean], Ld)        # REQUIRES claims: clean records only; 'fills again later' over every leg
    pe_ok = priority_evidence(Ld[recon], Ld)         # ALLOWS claims: any record whose kept variant reconciles exactly
    hk = Ld.drop_duplicates(['sid', 'mk_idx']).set_index(['sid', 'mk_idx'])
    last_settle = Ld.groupby('h').settle_ms.max()
    at_last = lambda ev: {hk.h[k] for k in ev if hk.settle_ms[k] >= last_settle[hk.h[k]]}     # exhausted at the hash's LAST fill
    ev_req, ev_allow = at_last(pe_req['ev']), at_last(pe_req['ev'] | pe_ok['ev'])

    def _remclass(a):
        for c in ('size_class', 'post_class', 'persist_class', 'queue_class', 'removal_class'):
            a.check(int((~O[c].isin(C.CLASSES)).sum()), f'orders {lab}: {{n}} rows with {c} not in CLASSES')
        a.check(int(O.removal_set.map(lambda s: not (rset(s) and rset(s) <= set(C.REMOVAL))).sum()),
                f'orders {lab}: {{n}} rows whose removal_set is empty, unparseable or outside REMOVAL')
        a.check(int(O.removal_set.map(lambda s: not (isinstance(s, str) and rset(s) is not None and s == '|'.join(sorted(rset(s))))).sum()),
                f'orders {lab}: {{n}} rows whose removal_set is not the |-joined sorted token list (pinned)')
        a.check(int((O.removal_class == C.OBSERVED).sum()), f'orders {lab}: {{n}} rows with an observed removal (no public field observes a removal)')
        a.check(int((O.post_class == C.OBSERVED).sum()), f'orders {lab}: {{n}} rows with an observed posting (submission is not in public data)')
        for _, r in O[(O.removal_class == C.CONSTRAINED) & (O.status == C.STATUS_OK)].join(agg_all, on='h').iterrows():
            s = rset(r.removal_set) or frozenset()
            if s == {'exhausted'} and not zh.get(r.h, False):
                a.fail(f'orders {lab}: constrained sole exhausted without shown(p)=0 evidence after the last fill ({r.scenario})')
            elif s == {'cancelled_or_expired'} and not (np.isfinite(r.size_exact) and r.size_exact > r.cum + TOL_SH):
                a.fail(f'orders {lab}: constrained cancellation without a proven remaining size ({r.scenario})')
            else:
                a.ok()
    _call(A, 'removal_class_evidence', _remclass)

    def _cls(a):
        OO = O.assign(_rs=O.removal_set.map(rkey))
        g = OO.groupby('h')
        full = g.scenario.nunique()
        full = full[full == len(C.SCENARIOS)].index
        for grp, (cols, ccol) in GROUPS.items():
            diff = pd.concat([g['_rs' if c == 'removal_set' else c].nunique(dropna=False) > 1 for c in cols], axis=1).any(axis=1).reindex(full)
            nlab, lab1 = g[ccol].nunique(dropna=False).reindex(full), g[ccol].first().reindex(full)
            a.check(int((nlab > 1).sum()), f'orders {lab}: {{n}} hashes whose {grp} class label varies across scenario rows')
            bad = diff & (nlab == 1) & (lab1 != C.SENSITIVE)
            if bad.any():
                a.fail(f'orders {lab}: {int(bad.sum())} hashes whose {grp} differs across scenarios but is labelled {lab1[bad].iloc[0]}')
            else:
                a.ok()
            a.info['sensitive_but_identical'] += int((~diff & (lab1 == C.SENSITIVE)).sum())
    _call(A, 'class_scenario_consistency', _cls)

    def _size(a):
        a.check(int((oks.size_lb < oks.cum - TOL_SH * np.maximum(1, oks.cum)).sum()), f'orders {lab}: {{n}} ok rows with size_lb < cumulative fills')
        a.check(int((np.isfinite(oks.size_exact) & (oks.size_exact < oks.size_lb - TOL_SH * np.maximum(1, oks.size_lb))).sum()),
                f'orders {lab}: {{n}} ok rows with size_exact < size_lb')
    _call(A, 'size_bounds', _size)

    def _prio(a):
        a.info['price_priority_evidence_legs'] += len(pe_req['ev'])
        for k_, v in pe_req['order'].items():
            a.info[f'multi_price_records_{k_}'] += v
        rows = oks[oks.h.isin(ev_req) & oks.scenario.isin(FIFO_FAMILY)]
        a.check(int(sum('exhausted' not in (rset(s) or ()) or not same(x, c, TOL_SH) for s, x, c in zip(rows.removal_set, rows.size_exact, rows.cum))),
                f'orders {lab}: {{n}} S_FIFO-family rows ignore price-priority exhaustion evidence (exhausted not in set, or size_exact != cum_shares)')
        just = lambda sc, h: (sc in FIFO_FAMILY and h in ev_allow) or zh.get(h, False)
        sole = oks[oks.removal_set.map(lambda s: rset(s) == {'exhausted'}).to_numpy(bool)]
        a.check(int(sum(not just(sc, h) for sc, h in zip(sole.scenario, sole.h))),
                f'orders {lab}: {{n}} rows claim sole exhausted without price-priority or shown-zero evidence (mk_idx is never evidence)')
        se = oks[np.isfinite(oks.size_exact.to_numpy(float))]
        a.check(int(sum(not (just(sc, h) and same(x, c, TOL_SH)) for sc, h, x, c in zip(se.scenario, se.h, se.size_exact, se.cum))),
                f'orders {lab}: {{n}} rows state size_exact without price-priority or shown-zero exhaustion at the last fill (or != cum_shares)')
    _call(A, 'priority_exhaustion', _prio)

    def _t12(a):
        a.info['T1_legs'] += pe_req['t1_legs']; a.info['T1_hashes'] += pe_req['t1_hashes']; a.info['T2_records'] += pe_req['t2']
        r1, r2 = ({x for r in life_j for x in _t_reports(r, IFACE[k])} for k in ('t1_key', 't2_key'))
        if not r1 or not r2:
            a.fail(f'lifecycle summary {lab}: priority tests T1 / T2 not reported'); return
        c1, c2 = {pe[k] for pe in (pe_req, pe_ok) for k in ('t1_legs', 't1_hashes')}, {pe['t2'] for pe in (pe_req, pe_ok)}
        (a.ok() if c1 & r1 else a.fail(f'lifecycle summary {lab}: T1 reported {sorted(r1)[:4]} != recomputed {sorted(c1)} (strictly-better exhausted makers that fill again)'))
        (a.ok() if c2 & r2 else a.fail(f'lifecycle summary {lab}: T2 reported {sorted(r2)[:4]} != recomputed {sorted(c2)} (records whose >= 2 worst-price makers fill again)'))
    _call(A, 'priority_tests_T1_T2', _t12)

    def _ahead(a):          # pinned: ahead_lo refers to the order's FIRST fill (min hi; a tie: any fill tied for first)
        first = Ld[Ld.hi.to_numpy() == Ld.groupby('h').hi.transform('min').to_numpy()]
        vals = defaultdict(set)
        for s_, i_, h in zip(first.sid, first.mk_idx, first.h):
            vals[h].update({pe_req['ahead'].get((s_, int(i_)), 0.0), pe_ok['ahead'].get((s_, int(i_)), 0.0)})
        rows = oks[oks.scenario.isin(FIFO_FAMILY).to_numpy() & np.isfinite(oks.ahead_lo.to_numpy(float))]
        a.check(int(sum(not any(same(x, v, TOL_SH) for v in vals.get(h, {0.0})) for h, x in zip(rows.h, rows.ahead_lo))),
                f'orders {lab}: {{n}} S_FIFO-family rows whose ahead_lo is not the price-priority value at the first fill of the order '
                f'(the other worst-price makers when it alone refilled, else 0)')
    _call(A, 'priority_ahead_lo', _ahead)

    def _lv(a):
        K = max(G['K'] or {100})
        for n, what in ((int((~LV.fid.isin(FR.fid)).sum()), 'levels whose fid is not a frame'), (int((~LV.side.isin(['bid', 'ask'])).sum()), 'side not in bid/ask'),
                        (int(LV.duplicated(['fid', 'side', 'ptick']).sum()), 'duplicate (fid, side, ptick)'), (int((LV.shares < 0).sum()), 'negative shares'),
                        (int(((LV.ptick < 0) | (LV.ptick > K)).sum()), f'ptick outside 0..{K}'), (int(FR.fid.duplicated().sum()), 'duplicate frame fid'),
                        (int((LV.mid != mid).sum() + (FR.mid != mid).sum()), 'another mid')):
            a.check(n, f'levels/frames {lab}: {{n}} rows with {what}')
    _call(A, 'levels_frames_consistent', _lv)

    def _tob(a):
        if TB is None:
            a.fail(f'tob {day} missing'); return
        tb = TB[(TB.mid == mid) & (TB.upd_ms >= FR.upd_ms.min()) & (TB.upd_ms <= FR.upd_ms.max())]      # depth may cover a narrower span
        k1, k2 = Counter(zip(FR.upd_ms, FR.recv_ns, FR.conn)), Counter(zip(tb.upd_ms, tb.recv_ns, tb.conn))
        (a.ok() if k1 == k2 else a.fail(f'tob {lab}: frame receipts differ from depth frames ({sum((k1 - k2).values())} only in depth, '
                                         f'{sum((k2 - k1).values())} only in tob)'))
        K = max(G['K'] or {100})
        bid, ask = LV[LV.side == 'bid'], LV[LV.side == 'ask']
        bb, ba = bid.groupby('fid').ptick.max(), ask.groupby('fid').ptick.min()
        sh = LV.set_index(['fid', 'side', 'ptick']).shares
        x = FR.merge(tb, on=['upd_ms', 'recv_ns', 'conn'], suffixes=('', '_t'))
        exp = dict(best_bid=x.fid.map(bb) / K, best_ask=x.fid.map(ba) / K,
                   best_bid_sh=[sh.get((f, 'bid', int(p)), np.nan) if not isnull(p) else np.nan for f, p in zip(x.fid, x.fid.map(bb))],
                   best_ask_sh=[sh.get((f, 'ask', int(p)), np.nan) if not isnull(p) else np.nan for f, p in zip(x.fid, x.fid.map(ba))],
                   n_bid=x.fid.map(bid.groupby('fid').size()).fillna(0), n_ask=x.fid.map(ask.groupby('fid').size()).fillna(0),
                   order_count=x.order_count, pend_bid_usd=x.pend_bids.map(lambda s: sum(float(v) for _, v in json.loads(s))),
                   pend_ask_usd=x.pend_asks.map(lambda s: sum(float(v) for _, v in json.loads(s))))
        for c, v in exp.items():
            got = x[c + '_t'] if c + '_t' in x else x[c]
            a.check(int(sum(not same(p, q, 1e-9) for p, q in zip(got, v))), f'tob {lab}: {{n}} frames whose {c} differs from the depth levels/frame')
    _call(A, 'tob_from_levels', _tob)

    # coverage vectors for this market: per scenario x group x denominator (orders / executions / shares) x population (all / ok)
    cov = []
    for pop, OO in (('all', O), ('ok', O[O.status == C.STATUS_OK])):
        j = OO.join(agg_all, on='h')
        for s, g in j.groupby('scenario'):
            for grp in GROUPS:
                col = GROUPS[grp][1]
                for den, w in (('orders', None), ('executions', 'n'), ('shares', 'cum')):
                    c = g.groupby(col).size() if w is None else g.groupby(col)[w].sum()
                    cov.append((s, grp, den, pop, {k: float(v) for k, v in c.items()}))
    G['cov'][(day, mid)] = cov
    G['life_json'][(day, mid)] = life_j
    if G['own_h'] and day in E['gt'] and mid in E['gt'][day]:
        G['gt_orders'].append(O[O.h.isin(G['own_h'])].assign(day=day))
    guard(f'market {lab}')


# ---------------------------------------------------------------------------------------------------------------- global checks
def _vec_match(found, cands):
    """A reported class map matches a candidate as fractions (sum ~ 1, rounding allowed) or exactly as counts."""
    tot = sum(found.values())
    for c in cands:
        ct, keys = sum(c.values()), set(found) | set(c)
        if ct > 0 and abs(tot - 1) <= 2 * TOL_FR * len(keys) and all(abs(found.get(k, 0) - c.get(k, 0) / ct) <= TOL_FR for k in keys):
            return True
        if all(abs(found.get(k, 0) - c.get(k, 0)) <= 1e-6 * max(1, abs(c.get(k, 0))) for k in keys):
            return True
    return False


def _econ_frame(E, k, filters=None, extra=()):
    names = pq.read_schema(E['P']['econ'][k]).names
    miss = [v for v in IFACE[k].values() if v not in names]
    need(not miss, f'econ {k}: pinned columns {miss} absent')
    cols = list(IFACE[k].values()) + [c for c in extra if c in names and c not in IFACE[k].values()]
    df = pd.read_parquet(E['P']['econ'][k], columns=cols, filters=filters).rename(columns={v: r for r, v in IFACE[k].items()})
    if 'addr' in df:
        df['addr'] = df.addr.str.lower()
    return df


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


SCORE_KS = ('n', 'failed', 'defects', 'sensitive_misses', 'inside', 'overlap', 'disjoint', 'not_stated', 'not_exhausted_only')


def _is_sk(k):       # declared (e2e 6): every GT block counts EVERY scored order; orders without the claim are an explicit 'not_*' outcome
    return k in SCORE_KS or str(k).startswith('not_')


def _is_by_class(x):
    return isinstance(x, dict) and any(k in C.CLASSES and isinstance(v, dict) for k, v in x.items())


def _score_block(blk, path):
    """A scores.json block at path -> (block, 'flat') for {n, failed, ...} or (block, 'nested') for {class: {n, <outcomes>, failed, ...}}, the
    class level either under the path (size.lb.<class>) or above its tail (size.<class>.lb); (None, None) when absent. {} reads as nested."""
    rep = blk
    for p_ in path:
        rep = rep.get(p_) if isinstance(rep, dict) else None
    if isinstance(rep, dict):
        return rep, ('nested' if (_is_by_class(rep) or not rep) else 'flat')
    top = blk.get(path[0]) if isinstance(blk, dict) else None
    if len(path) > 1 and _is_by_class(top):
        out = {}
        for c, v in top.items():
            x = v if c in C.CLASSES else None
            for p_ in path[1:]:
                x = x.get(p_) if isinstance(x, dict) else None
            if isinstance(x, dict):
                out[c] = x
        if out:
            return out, 'nested'
    return None, None


def _score_cmp(rep, form, mb):
    """'' iff the reported block equals the recomputed one on every key both carry, else the differences. Nested: every reported class (any
    n) against the recomputed class (zero when absent), every recomputed class with n > 0 must be reported, totals beside the classes too."""
    eq = lambda v, w: _num(v) and float(v) == float(w)
    flat = lambda d: {**d, **d['outcomes']} if isinstance(d.get('outcomes'), dict) else d
    if form == 'flat':
        rep = flat(rep)
        ks = [k for k in rep if _is_sk(k) and k in mb]
        if not ks:
            return 'holds none of n / failed / defects / sensitive_misses'
        return '; '.join(f'{k} {rep[k]} != {mb[k]}' for k in ks if not eq(rep[k], mb[k]))
    by, zero = mb['_by'], {k: 0 for k in mb if k != '_by'}
    rc = {c: flat(v) for c, v in rep.items() if c in C.CLASSES and isinstance(v, dict)}
    bad = [f'class {c} recomputed n {v["n"]} not reported' for c, v in by.items() if v['n'] > 0 and c not in rc]
    for c, v in rc.items():
        mc = by.get(c, zero)
        if 'n' not in v:
            bad.append(f'class {c} lacks n'); continue
        bad += [f'{c}.{k} {v[k]} != {mc[k]}' for k in v if _is_sk(k) and k in mc and not eq(v[k], mc[k])]
    bad += [f'total {k} {rep[k]} != {mb[k]}' for k in rep if _is_sk(k) and k in mb and not eq(rep[k], mb[k])]
    return '; '.join(bad)


def global_checks(A, E, G):
    def _cov(a):
        allc = []
        for key, cov in G['cov'].items():
            cands = [v for *_, v in cov]
            allc += cov
            for fm in [m for r in G['life_json'].get(key, []) for m in class_maps(r, [])]:
                (a.ok() if _vec_match(fm, cands) else a.fail(f'coverage {key[0]}_{key[1]}: reported class map {fm} matches no recomputed vector'))
        if 'coverage' in E['P']['misc']:
            agg = defaultdict(Counter)
            for s, grp, den, pop, v in allc:
                agg[(s, grp, den, pop)].update(v)
            cands = [dict(v) for v in agg.values()]
            for fm in class_maps(rdj(E['P']['misc']['coverage']), []):
                (a.ok() if _vec_match(fm, cands) else a.fail(f'coverage file: class map {fm} matches no recomputed aggregate vector'))
        if a.n == 0:
            a.fail('no class coverage reported anywhere (life receipts / coverage file): coverage arithmetic unverifiable')
    _call(A, 'coverage_arithmetic', _cov)

    def _econ(a):
        miss = [k for k in ECON_T if k not in E['P']['econ']]
        if miss:
            a.fail(f'econ tables missing: {miss}'); return
        MM, MD = _econ_frame(E, 'maker_market', extra=('fifo_status',)), _econ_frame(E, 'maker_day')
        for X in (MM, MD):
            X['gross'] = X.gross_mk + X.gross_tk          # pinned: gross = gross_res_mk + gross_res_tk
        mo = MM[MM.status == C.STATUS_OK]
        a.check(int(sum((d, int(m)) in G['censored'] for d, m in zip(mo.day, mo.mid))),
                'econ maker_market: {n} status-ok rows on boundary_censored markets (must be excluded as a status)')
        noend = hasf(mo.fifo_status, 'end_unavailable') if 'fifo_status' in mo else np.zeros(len(mo), bool)   # no book: END-based split undefined
        fin = lambda *xs: np.all([np.isfinite(x.to_numpy(float)) for x in xs], axis=0)
        for lhs, rhs, what, defined in ((mo.pnl, mo.gross - mo.fees, 'pnl != gross - fees', fin(mo.pnl, mo.gross, mo.fees)),
                                        (mo.gross, mo.rt + mo.resid, 'gross != rt + resid', fin(mo.gross, mo.rt, mo.resid)),
                                        (mo.resid, mo.drift + mo.resol, 'resid != drift + resol', fin(mo.resid, mo.drift, mo.resol) & ~noend)):
            a.info[f'rows_identity_undefined[{what}]'] = int((~defined).sum())    # pinned: check only the identities that are defined
            a.check(int((~np.isclose(lhs, rhs, atol=TOL_USD, rtol=1e-9) & defined).sum()), f'econ maker_market: {{n}} rows with {what}')
        zs = mo.assign(ag=mo.gross.abs(), ap=mo.net_pos.abs()).groupby(['day', 'mid'])[['gross', 'net_pos', 'ag', 'ap']].sum()
        a.check(int(((zs.gross.abs() > TOL_USD + 1e-9 * zs.ag) | (zs.net_pos.abs() > 1e-6 + 1e-9 * zs.ap)).sum()),
                'econ maker_market: {n} markets whose sum over addresses of gross or net_pos is not 0 (zero-sum)')
        cols = ['gross_mk', 'gross_tk', 'fees', 'pnl']           # always defined (the END-based / FIFO splits may be undefined without a book)
        j = mo.groupby(['addr', 'day'])[cols].sum(min_count=1).join(MD[MD.status == C.STATUS_OK].groupby(['addr', 'day'])[cols].sum(min_count=1),
                                                                    rsuffix='_d', how='outer')
        a.check(int(sum((~np.isclose(j[c].fillna(np.inf), j[c + '_d'].fillna(-np.inf), atol=TOL_USD, rtol=1e-9)).sum() for c in cols)),
                'econ maker_day: {n} (addr, day) values != the sum of maker_market')
        for d in sorted(set(mo.day)):
            FV = _econ_frame(E, 'fills_valued', [(IFACE['fills_valued']['day'], '=', d)])
            FV = FV[FV.status == C.STATUS_OK]
            a.check(int(sum((d, int(m)) in G['censored'] for m in FV.mid)), f'econ fills_valued {d}: {{n}} status-ok rows on boundary_censored markets')
            a.check(int((~np.isclose(FV.gross, FV.s * FV.q * (FV.V - FV.p), atol=TOL_USD, rtol=1e-9)).sum()),
                    f'econ fills_valued {d}: {{n}} rows with gross_res != s*q*(V-p)')
            a.check(int((~np.isclose(FV.pnl, FV.gross - FV.fee, atol=TOL_USD, rtol=1e-9)).sum()),
                    f'econ fills_valued {d}: {{n}} rows with net_res != gross_res - fee_res')
            m = mo[mo.day == d].groupby(['addr', 'mid'])[['pnl', 'fees']].sum().join(
                FV.groupby(['addr', 'mid']).agg(pnl_v=('pnl', 'sum'), fee_v=('fee', 'sum')), how='outer')
            a.check(int((~np.isclose(m.pnl.fillna(np.inf), m.pnl_v.fillna(-np.inf), atol=TOL_USD, rtol=1e-9)).sum() +
                        (~np.isclose(m.fees.fillna(np.inf), m.fee_v.fillna(-np.inf), atol=TOL_USD, rtol=1e-9)).sum()),
                    f'econ {d}: {{n}} (addr, mid) sums of fills_valued net_res / fee_res != maker_market')
        keys = [c for c in IFACE['segments_keys'] if c in pq.read_schema(E['P']['econ']['segments']).names]
        SG = _econ_frame(E, 'segments', extra=keys)
        SG = SG[SG.status == C.STATUS_OK]
        tdim, ngrp = IFACE['segments_total_dim'], 0
        for key, g in SG.groupby(['day'] + keys, dropna=False):
            tot = g[g.dim == tdim]
            if not len(tot):
                a.fail(f'econ segments {key}: no dim {tdim} total row'); continue
            ngrp += 1
            for dim, gd in g.groupby('dim'):
                if dim in IFACE['segments_full_dims']:
                    for col in ('pnl', 'gross'):
                        t, s_ = float(tot[col].sum()), float(gd[col].sum())
                        (a.ok() if abs(s_ - t) <= TOL_USD * max(1, len(gd)) + 1e-9 * abs(t) else
                         a.fail(f'econ segments {key}: dim {dim} buckets sum {col} {s_:.6f} != the {tdim} total {t:.6f}'))
                elif dim != tdim:
                    a.info['segment_dims_not_a_partition'] += 1
        a.info['segment_groups'] = ngrp
        if not ngrp:
            a.fail('econ segments: no status-ok segment group to verify')
    _call(A, 'econ_conservation', _econ)

    def _pnl(a):
        if not G['pnl']:
            a.fail('no FILL-day legs to recompute P&L from'); return
        if 'maker_market' not in E['P']['econ']:
            a.fail('econ maker_market missing'); return
        mine = pd.concat(G['pnl'], ignore_index=True)
        a.check(int(mine.nfee_bad.sum()), 'econ: {n} valued (addr, mid, role) legs carry an unvaluable fee (missing, or type neither COLLATERAL nor SHARES)')
        MM = _econ_frame(E, 'maker_market')
        MM['gross'] = MM.gross_mk + MM.gross_tk
        g = mine.groupby(['addr', 'mid'])[['gross', 'fee', 'fee_px']].sum()
        j = g.join(MM[MM.status == C.STATUS_OK].groupby(['addr', 'mid'])[['gross', 'fees', 'fees_px']].sum(), rsuffix='_b', how='outer')
        a.info['addr_mid_recomputed'] = int(len(g)); a.info['addresses_recomputed'] = int(mine.addr.nunique())
        miss, extra = int(j.gross_b.isna().sum()), int(j.gross.isna().sum())
        (a.fail(f'econ: {miss} recomputed (addr, mid) pairs absent from maker_market, {extra} maker_market pairs with no ok legs')
         if miss or extra else a.ok())
        k = j.dropna(subset=['gross', 'gross_b'])
        a.check(int((~np.isclose(k.gross, k.gross_b, atol=TOL_USD, rtol=1e-9)).sum()),
                'econ: {n} (addr, mid) gross P&L to resolution != the independent recompute sum s*q*(V-p)')
        a.check(int((~np.isclose(k.fee, k.fees, atol=TOL_USD, rtol=1e-9)).sum()), 'econ: {n} (addr, mid) fee_res != recompute (SHARES fee at the token resolution value)')
        a.check(int((~np.isclose(k.fee_px, k.fees_px, atol=TOL_USD, rtol=1e-9)).sum()), 'econ: {n} (addr, mid) fee_fill != recompute (SHARES fee at each leg\'s own price)')
    _call(A, 'econ_pnl_recompute', _pnl)

    def _ids(a):          # pinned: maker_day carries in_exante_sh / in_exante_pnl membership flags
        if C.ID_DAY not in E['days'] or G['id_tops'] is None:
            a.fail('ID-day legs missing: ex-ante identities unverifiable'); return
        if 'maker_day' not in E['P']['econ']:
            a.fail('econ maker_day missing'); return
        MD = _econ_frame(E, 'maker_day')
        shares, pnl_variants = G['id_tops']
        addrs = set(MD.addr)
        for col, cands in (('in_exante_sh', [shares]), ('in_exante_pnl', pnl_variants)):
            got = set(MD.addr[MD[col].map(lambda v: (not isnull(v)) and bool(v)).to_numpy(bool)])
            (a.ok() if any(got == ({lo(x) for x in c} & addrs) for c in cands) else
             a.fail(f'identities: maker_day {col} marks {len(got)} addresses vs {[len(set(c) & addrs) for c in cands]} expected (differs from the Oct 23 recompute)'))
    _call(A, 'identities_exante', _ids)

    def _gt(a):           # pinned (fix batch): scenarios.<S>.{coverage{orders,executions,shares}{n,covered}, posting, size{lb,exact,ub}, removal{...}, ...}
        if G['own'] is None or 'scores' not in E['P']['misc']:
            a.fail(f'truth own_orders (role columns) or truth/scores.json missing: {G["own_err"]}'); return
        if not G['gt_orders']:
            a.fail('no GT lifecycle orders of our own hashes loaded'); return
        LO = pd.concat(G['gt_orders'], ignore_index=True)
        own = G['own']
        cen = E.get('gt_censored') or set()      # e2e 4: boundary_censored windows' own orders are outside the GT SCORE population
        lg0 = G['gt_own_legs'] if G['gt_own_legs'] is not None else pd.DataFrame(columns=['mid', 'sid', 'mk_idx', 'shares', 'h'])
        if 'mid' in own:
            in_cen = pd.to_numeric(own.mid, errors='coerce').isin(cen).to_numpy(bool)
        else:                                     # no market column in the truth table: a hash whose every leg sits in a censored window
            hm = lg0.groupby('h').mid.agg(lambda s: set(int(x) for x in s)).to_dict() if len(lg0) else {}
            in_cen = own.h.map(lambda h: bool(hm.get(h)) and hm[h] <= cen).to_numpy(bool)
        filled = own[(own.filled.astype(float) > TOL_SH).to_numpy() & ~in_cen]
        keys = ['mid', 'h'] if 'mid' in own else ['h']
        a.info['gt_filled_own_orders'] = int(len(filled))
        a.info['gt_own_orders_in_censored_windows'] = int(in_cen.sum()); a.info['gt_censored_windows'] = len(cen)
        scen = rdj(E['P']['misc']['scores']).get('scenarios')
        if not isinstance(scen, dict):
            a.fail('gt scores: no scenarios block'); return
        lg = lg0[~pd.to_numeric(lg0.mid, errors='coerce').isin(cen).to_numpy(bool)] if len(lg0) else lg0
        life_m = {m for v in E['gt'].values() for m in v}
        kf = lambda df: list(zip(*[df[k].tolist() for k in keys])) if len(df) else []
        for s in C.SCENARIOS:
            blk = scen.get(s)
            if not isinstance(blk, dict):
                a.fail(f'gt scores: scenario {s} absent'); continue
            LS = LO[LO.scenario == s]
            cands = defaultdict(list)            # coverage candidates: lifecycle rows (all / ok) x own legs (outside censored / GT life markets)
            for LSx in (LS, LS[LS.status == C.STATUS_OK]):
                cs = set(kf(LSx))
                fo = np.array([k in cs for k in kf(filled)], bool)
                fil = filled.filled.astype(float).to_numpy()
                cands['orders'].append((len(filled), int(fo.sum())))
                cands['shares'].append((float(fil.sum()), float(fil[fo].sum()) if len(fo) else 0.0))
                for lgx in (lg, lg[lg.mid.isin(life_m)]):
                    lk = np.array([k in cs for k in kf(lgx)], bool)
                    sh = lgx.shares.astype(float).to_numpy()
                    cands['executions'].append((len(lgx), int(lk.sum())))
                    cands['shares'].append((float(sh.sum()), float(sh[lk].sum()) if len(lk) else 0.0))
            cov = blk.get('coverage')
            if not isinstance(cov, dict):
                a.fail(f'gt scores: {s}.coverage block absent')
            else:
                for den in ('orders', 'executions', 'shares'):
                    rep = cov.get(den)
                    if not (isinstance(rep, dict) and _num(rep.get('n')) and _num(rep.get('covered'))):
                        a.fail(f'gt scores: {s}.coverage.{den} lacks n / covered'); continue
                    hit = any(abs(rep['n'] - n) <= 1e-6 * max(1, abs(n)) and abs(rep['covered'] - c) <= 1e-6 * max(1, abs(c)) for n, c in cands[den])
                    (a.ok() if hit else a.fail(f'gt scores: {s}.coverage.{den} reported (n {rep["n"]}, covered {rep["covered"]}) != recomputed {sorted(set(cands[den]))[:4]}'))
            jall = LS.merge(filled, on=keys, suffixes=('', '_t'))
            mine = []
            for j in (jall, jall[jall.status == C.STATUS_OK]):        # population: every matched row, or ok rows only
                wl = np.where(j.post_lo_ms.to_numpy() < 0, -np.inf, j.post_lo_ms.to_numpy(float))
                wh = np.where(j.post_hi_ms.to_numpy() < 0, np.inf, j.post_hi_ms.to_numpy(float))
                al_, ah_ = j.act_lo.astype(float).to_numpy(), j.act_hi.astype(float).to_numpy()
                hl, hh = np.where(al_ < 0, -np.inf, al_), np.where(ah_ < 0, np.inf, ah_)
                inside, disj = (wl <= hl) & (hh <= wh), (hh < wl) | (hl > wh)
                sz = j['size'].astype(float).to_numpy()
                ge = sz >= j.size_lb.to_numpy(float) - TOL_SH
                se = j.size_exact.to_numpy(float); st = np.isfinite(se); eq = np.abs(sz - se) <= 1e-6
                exh = j.removal_set.map(lambda x: rset(x) == {'exhausted'}).to_numpy(bool)
                fil_ = j.state.astype(str).str.lower().to_numpy() == 'filled'
                al = np.ones(len(j), bool)
                ubc = next((c for c in ('size_ub', 'x_size_ub') if c in j), None)

                def blk_(n_mask, fail, ccol, extra=None, j=j):     # totals + per class (e2e 4: blocks nested {class: {n, <outcomes>, failed, ...}})
                    fail = fail & n_mask                          # ccol: a class column, or a CONSTANT class for every row ('=observed')
                    cl = np.full(len(j), ccol[1:], object) if ccol.startswith('=') else j[ccol].astype(object).to_numpy()

                    def one(sel):
                        nm_, fl = n_mask & sel, fail & sel
                        d = dict(n=int(nm_.sum()), failed=int(fl.sum()), defects=int((fl & np.isin(cl, [C.OBSERVED, C.CONSTRAINED])).sum()),
                                 sensitive_misses=int((fl & (cl == C.SENSITIVE)).sum()))
                        d.update({k: int((v & nm_).sum()) for k, v in (extra or {}).items()})
                        return d
                    d = one(np.ones(len(j), bool))
                    d['_by'] = {str(c): one(cl == c) for c in dict.fromkeys(cl[n_mask].tolist())}
                    return d
                outc_ = dict(inside=inside, overlap=~inside & ~disj, disjoint=disj)
                # contract §7 (declared, e2e 5): the size LOWER bound is always class observed (cumulative fills); the exact-size and
                # upper-bound blocks take EVERY scored order as the denominator, with not_stated as an outcome (failed = stated and wrong)
                for pf in (disj, ~inside):                              # posting failed = disjoint, or anything not inside
                    m_ = {('posting',): blk_(al, pf, 'post_class', outc_), ('size', 'lb'): blk_(al, ~ge, '=' + C.OBSERVED),
                          ('size', 'exact'): blk_(al, st & ~eq, 'size_class', dict(not_stated=~st)),
                          ('removal', 'exhausted_only'): blk_(al, exh & ~fil_, 'removal_class', dict(not_exhausted_only=~exh))}   # e2e 6: n = every order
                    if ubc:
                        ub = j[ubc].to_numpy(float); us = np.isfinite(ub)
                        m_[('size', 'ub')] = blk_(al, us & ~(sz <= ub + TOL_SH), 'size_class', dict(not_stated=~us))
                    mine.append(m_)
            for path in IFACE['scores_blocks']:
                nm = s + '.' + '.'.join(path)
                if not any(path in m for m in mine):
                    a.info['gt_blocks_not_recomputable'] += 1      # e.g. size.ub without an upper-bound column: reported, never verified
                    continue
                rep, form = _score_block(blk, path)
                if rep is None:
                    a.fail(f'gt scores: {nm} block absent'); continue
                a.info[f'gt_blocks_{form}'] += 1
                errs = [_score_cmp(rep, form, m[path]) for m in mine]
                (a.ok() if '' in errs else a.fail(f'gt scores: {nm} ({form}) reported != recomputed: {errs[0][:300]}'))
        a.info['gt_blocks_never_recomputed'] = 'removal.membership, removal.window, persistence, queue.*, fill windows, defects_total'
    _call(A, 'gt_scores', _gt)


# ------------------------------------------------------------------------------------------------------------------- run_checks
def pnl_legs(E, day, R, L):
    """Independent P&L to resolution per (address, market, role): s*q*(V-p) and fees. Population (amendment 1): records whose flags are 'ok'
    or informational only (IFACE econ_info_flags) - boundary_censored and every other flag are excluded. A taker leg is valued at its maker
    legs' YES prices (priceExecuted can differ from the maker-weighted average): taker gross = s_taker * sum_i q_i (V - p_i)."""
    rg = reg(E, day)
    keep = {C.STATUS_OK, *IFACE['econ_info_flags']}
    ok = R.loc[only_flags(R.status, keep), ['sid', 'mid', 'tk_signer', 'tk_index_set', 'tk_price', 'tk_y_side', 'tk_fee', 'tk_fee_type']]
    Lk = L.loc[L.sid.isin(ok.sid).to_numpy() & only_flags(L.status, keep),
               ['sid', 'mk_signer', 'mid', 'y_side', 'shares', 'y_price', 'mk_fee', 'mk_fee_type', 'mk_index_set', 'mk_price']]
    V = {m: rg.get(int(m), (0, 0, np.nan))[2] for m in ok.mid.unique()}
    mv, tv = Lk.mid.map(V).to_numpy(float), ok.mid.map(V).to_numpy(float)
    leg_val = Lk.shares.to_numpy() * (mv - Lk.y_price.to_numpy())                      # q_i (V - p_i) at the maker's YES price
    mk = pd.DataFrame(dict(addr=Lk.mk_signer.str.lower().to_numpy(), mid=Lk.mid.to_numpy(), role='maker',
                           gross=np.where(Lk.y_side == 'bid', 1.0, -1.0) * leg_val,
                           fee=fee_value(Lk.mk_fee, Lk.mk_fee_type, Lk.mk_index_set, mv),
                           fee_px=fee_at_px(Lk.mk_fee, Lk.mk_fee_type, Lk.mk_price), sh=Lk.shares.to_numpy(float)))
    per_rec = pd.Series(leg_val, index=Lk.sid.to_numpy()).groupby(level=0).sum()
    per_sh = Lk.groupby('sid').shares.sum()
    # e2e 4 / amendments 1-2: a SHARES taker fee converts at EACH LEG's own price (the taker token's price at that maker leg: the leg's YES
    # price for indexSet 1, its complement for 2), share-weighted over the record's legs - never at priceExecuted (records can carry a gap)
    tis = Lk.sid.map(ok.drop_duplicates('sid').set_index('sid').tk_index_set).to_numpy()
    own_px = np.where(tis == 1, Lk.y_price.to_numpy(float), 1 - Lk.y_price.to_numpy(float))
    px_w = pd.Series(Lk.shares.to_numpy(float) * own_px, index=Lk.sid.to_numpy()).groupby(level=0).sum() / per_sh
    tk = pd.DataFrame(dict(addr=ok.tk_signer.str.lower().to_numpy(), mid=ok.mid.to_numpy(), role='taker',
                           gross=np.where(ok.tk_y_side == 'bid', 1.0, -1.0) * ok.sid.map(per_rec).to_numpy(float),
                           fee=fee_value(ok.tk_fee, ok.tk_fee_type, ok.tk_index_set, tv),
                           fee_px=fee_at_px(ok.tk_fee, ok.tk_fee_type, ok.sid.map(px_w).to_numpy(float)), sh=ok.sid.map(per_sh).to_numpy(float)))
    out = pd.concat([mk, tk], ignore_index=True)
    out = out[np.isfinite(out.gross.to_numpy(float))]          # unresolved / sealed outcomes (V NaN) never valued
    out = out.assign(nfee_bad=(~np.isfinite(out.fee.to_numpy(float)) | ~np.isfinite(out.fee_px.to_numpy(float))).astype(int))   # sum() skips NaN
    return out.groupby(['addr', 'mid', 'role'], as_index=False)[['gross', 'fee', 'fee_px', 'sh', 'nfee_bad']].sum()     # aggregated: bounded memory


def id_tops(E, day, R, L):
    """Ex-ante identities from the ID day: top 10 by maker shares; declared exante_pnl = top 10 by maker-leg NET P&L to resolution among
    maker-majority addresses (resolved maker shares > resolved taker shares)."""
    keep = {C.STATUS_OK, *IFACE['econ_info_flags']}
    Lk = L.loc[L.sid.isin(R.sid[only_flags(R.status, keep)]).to_numpy() & only_flags(L.status, keep), ['mk_signer', 'shares']]
    sh = Lk.groupby(Lk.mk_signer.str.lower()).shares.sum()
    top = lambda s: sorted(s.index, key=lambda x: (-s[x], x))[:10]
    P = pnl_legs(E, day, R, L).assign(net=lambda d: d.gross - d.fee)
    shr = P.pivot_table(index='addr', columns='role', values='sh', aggfunc='sum', fill_value=0.0).reindex(columns=['maker', 'taker'], fill_value=0.0)
    majority = set(shr.index[shr.maker > shr.taker])
    mnet = P[(P.role == 'maker') & P.addr.isin(majority)].groupby('addr').net.sum()
    return top(sh), [top(mnet)]


def run_checks(E, only=None, batch=40):
    A = defaultdict(Acc); A['_only'] = set(only) if only else None
    G = dict(K=set(), anchor=set(), cov={}, life_json={}, gt_orders=[], pnl=[], id_tops=None, own=None, own_h=set(), own_err='', censored=set(),
             gt_own_legs=None)
    a = A['outputs_present']
    a.check(E['n_forbidden_named'], '{n} output files named for a non-consumed day')
    a.check(E['n_unexpected'], '{n} unexpected file names under depth/ or life/')
    for k in ECON_T:
        (a.ok() if k in E['P']['econ'] else a.fail(f'econ/{k}.parquet missing'))
    for k in ('identities', 'own_orders', 'scores'):
        (a.ok() if k in E['P']['misc'] else a.fail(f'{k} output missing'))
    if 'own_orders' in E['P']['misc']:          # pinned: a column registry in parquet metadata key mt_truth; roles mapped by synonyms (fail closed)
        try:
            sch = pq.read_schema(E['P']['misc']['own_orders'])
            meta = (sch.metadata or {}).get(b'mt_truth')
            roles = {r: next((c for c in syn if c in sch.names), None) for r, syn in IFACE['own_orders'].items()}
            miss = [r for r in IFACE['own_required'] if roles.get(r) is None]
            if miss:
                G['own_err'] = f'own_orders: no column for roles {miss} (mt_truth registry {"present" if meta else "absent"}; columns {sch.names[:14]})'
            else:
                raw_ = pd.read_parquet(E['P']['misc']['own_orders'], columns=list(dict.fromkeys([c for c in roles.values() if c] + (['status'] if 'status' in sch.names else []))))
                OW = pd.DataFrame({r: raw_[c].to_numpy() for r, c in roles.items() if c})
                if 'status' in raw_:
                    OW = OW[raw_['status'].to_numpy() == C.STATUS_OK]
                OW = OW.assign(h=OW.h.astype(str).str.lower())
                G['own'] = OW.drop_duplicates(['mid', 'h'] if 'mid' in OW else ['h'])
                G['own_h'] = set(G['own'].h)
                G['own_info'] = dict(rows=len(OW), orders=len(G['own']), registry=bool(meta), roles=roles)
        except Exception as e:                    # a broken table fails its own checks closed, never the run
            G['own'], G['own_h'], G['own_err'] = None, set(), f'own_orders unreadable: {type(e).__name__}: {e}'
    if E.get('gt_err'):
        A['outputs_present'].fail(f'GT population: {E["gt_err"]}')
    for day in E['days']:
        D = {}
        for k in ('records', 'legs'):             # a missing / unreadable day table fails that day's checks closed (day_checks), never the run
            try:
                D[k] = pd.read_parquet(E['P'][k][day]) if day in E['P'][k] else None
            except Exception as e:
                D[k] = None; A['outputs_present'].fail(f'{k} {day} unreadable: {type(e).__name__}')
        D['rec_j'], D['decl'] = [], {}
        D['depth_tot'] = {k: (Counter(), Counter()) for k in ('frames', 'levels')}      # per-flag / per-status day totals of depth tables
        D['depth_pm'] = {k: [] for k in ('frames', 'levels')}                            # per depth market: its own map was found
        guard(f'load {day}')
        if D['records'] is not None:            # boundary-censored markets: economics must exclude them as a status (amendment 1 item 5)
            R0 = D['records']
            G['censored'].update((day, int(m)) for m in R0.mid[hasf(R0.status, 'boundary_censored')].unique())
        if day == C.GT_DAY and D['legs'] is not None and G['own_h']:        # our own maker legs: GT execution / share coverage
            Lg = D['legs']; m_ = Lg.mk_hash.str.lower().isin(G['own_h']).to_numpy()
            G['gt_own_legs'] = Lg.loc[m_, ['mid', 'sid', 'mk_idx', 'shares']].assign(h=Lg.mk_hash[m_].str.lower().to_numpy())
        try:
            day_checks(A, E, day, D, G)
        except SystemExit:
            raise
        except Exception as e:
            for n in DAY_CHECKS:
                A[n].fail(f'cannot evaluate {day}: {type(e).__name__}: {e}')
        D['xm'] = set()
        for name, fn in (('H1_identity', lambda: D.__setitem__('xm', (lambda n: set(n[n > 1].index))(D['legs'].groupby(D['legs'].mk_hash.str.lower()).mid.nunique()))),
                         ('econ_pnl_recompute', lambda: G['pnl'].append(pnl_legs(E, day, D['records'], D['legs'])) if (day in C.MEASURE_DAYS or day == C.ID_DAY) else None),
                         ('identities_exante', lambda: G.__setitem__('id_tops', id_tops(E, day, D['records'], D['legs'])) if day == C.ID_DAY else None)):
            if D['legs'] is None or D['records'] is None:
                break
            try:       # xm: a hash in more than one market of the day is an identity conflict; pnl / id_tops feed the global checks
                fn()
            except SystemExit:
                raise
            except Exception as e:
                A[name].fail(f'cannot evaluate {day}: {type(e).__name__}: {e}')
        mids = [m for d2, m in E['life'] if d2 == day]
        for i in range(0, len(mids), batch):
            part = mids[i:i + batch]
            try:
                TB = pd.read_parquet(E['P']['tob'][day], filters=[('mid', 'in', [int(m) for m in part])]) if day in E['P']['tob'] else None
            except Exception as e:                # tob_from_levels then fails closed ('tob missing')
                TB = None; A['outputs_present'].fail(f'tob {day} unreadable: {type(e).__name__}')
            for mid in part:
                market_checks(A, E, day, mid, D, G, TB)
            del TB

        want = A['_only'] is None or 'status_counts_reported' in A['_only']
        for d2, m2 in E['depth_only']:         # e2e 4: EVERY depth window the extractor wrote counts in the day totals (status column only)
            if d2 != day or not want:
                continue
            for k in ('frames', 'levels'):
                try:
                    s_ = pd.read_parquet(E['P'][k][(d2, m2)], columns=['status']).status if (d2, m2) in E['P'][k] else None
                except Exception:
                    s_ = None
                if s_ is None:
                    A['status_counts_reported'].fail(f'status counts {k} {d2}_{m2}: depth-only window table missing / unreadable'); continue
                tg = [flag_counts(s_), {str(x): int(v) for x, v in s_.value_counts().items()}]
                D['depth_tot'][k][0].update(tg[0]); D['depth_tot'][k][1].update(tg[1])
                D['depth_pm'][k].append(any(find_map(r, t) for r in D['rec_j'] for t in tg))

        def _day_depth(a):     # depth status counts: a DAY-total map in the extract receipt over every depth window, or every window per market
            for k, (tf, ts) in D['depth_tot'].items():
                if tf:
                    (a.ok() if any(find_map(r, t) for r in D['rec_j'] for t in (dict(tf), dict(ts))) or all(D['depth_pm'][k]) else
                     a.fail(f'status counts {k} {day}: no extract receipt map equals the day totals {dict(tf)} over every depth window'))
        _call(A, 'status_counts_reported', _day_depth)
        del D
        guard(f'day {day}')
    global_checks(A, E, G)
    names = [n for n in ALL_CHECKS if A['_only'] is None or n in A['_only']]
    return dict(checks={n: A[n].res() for n in names}, details={n: A[n].det() for n in names},
                info=dict(ptick_units=sorted(G['K']), t_rel_anchor=sorted(G['anchor']), vmhwm_mib=round(hwm_mib(), 1)))


# ---------------------------------------------------------------------------------------------------------------- raw re-parse
VOLATILE = frozenset({'bestBid', 'bestAsk', 'onChainId'})


def canon(x):
    """Canonical record content (contract section 4): without the market object, outcome.status, bestBid / bestAsk and onChainId."""
    def strip(o, parent=None):
        if isinstance(o, dict):
            return {k: strip(v, k) for k, v in o.items() if k not in VOLATILE and not (parent == 'outcome' and k == 'status')}
        if isinstance(o, list):
            return [strip(v, parent) for v in o]
        return o
    return json.dumps(strip({k: v for k, v in x.items() if k != 'market'}), sort_keys=True, separators=(',', ':'))


def rest_receipts(mid, stems, raw_root, st, hrs, full=None):
    """Every receipt of every match record of market mid in the given REST hours: {sid: [(recv_ns, canon, record)]}. Line pre-filter =
    the endpoint path + the escaped market-id text; `full` (smoke test) also parses EVERY matches line as the pre-filter's control."""
    pat = re.compile(r'\\"id\\":\s*(?:\\")?%d(?![0-9])' % mid)      # tolerant of a space or a quoted id
    rec = defaultdict(list)
    for s in stems:
        p = raw_path('rest', s, raw_root)            # Forbidden is raised here, before any open
        if not p.exists():
            st['rest_hours_missing'] += 1; continue
        n0 = sum(map(len, rec.values()))
        with gzip.open(p, 'rt') as f:
            for ln in f:
                if '/v1/orders/matches' not in ln[:1000]:
                    continue
                hit = pat.search(ln) is not None
                if not hit and full is None:
                    continue
                r = json.loads(ln)
                if not str(r.get('path', '')).startswith('/v1/orders/matches'):
                    continue
                if r.get('status') != 200:
                    st['rest_non200_rows'] += 1; continue
                try:
                    pl = json.loads(r['raw'])
                except (TypeError, ValueError):
                    st['rest_bad_payload'] += 1; continue
                for x in pl.get('data') or []:
                    m = int((x.get('market') or {}).get('id', -1))
                    if full is not None:
                        full(r, x, m, hit)
                    if m == mid:
                        st['needle_miss'] += int(not hit)
                        rec[x.get('settlementId')].append((int(r['recv_ns']), canon(x), x))
        hrs.append((s, sum(map(len, rec.values())) - n0))
        guard('rest hour')
    return rec


def ws_frames(mid, stems, raw_root, st):
    needle = 'predictOrderbook/%d\\"' % mid
    out = []
    for s in stems:
        p = raw_path('predict_ws', s, raw_root)
        if not p.exists():
            st['ws_hours_missing'] += 1; continue
        st['ws_hours_read'] += 1
        with gzip.open(p, 'rt') as f:
            for ln in f:
                if needle not in ln:
                    continue
                r = json.loads(ln); m = json.loads(r['raw'])
                if m.get('type') != 'M' or m.get('topic') != f'predictOrderbook/{mid}':
                    continue
                d = m.get('data') or {}
                if 'updateTimestampMs' not in d or int(d.get('marketId', -1)) != mid:
                    st['ws_bad_frames'] += 1; continue
                los, sp = d.get('lastOrderSettled') or {}, d.get('settlementsPending') or {}
                out.append(dict(upd_ms=int(d['updateTimestampMs']), recv_ns=int(r['recv_ns']), conn=r.get('connection_id'),
                                order_count=int(d.get('orderCount', -1)), bids=[(float(a), float(b)) for a, b in d.get('bids') or []],
                                asks=[(float(a), float(b)) for a, b in d.get('asks') or []],
                                pb=sorted((float(a), float(b)) for a, b in sp.get('bids') or []), pa=sorted((float(a), float(b)) for a, b in sp.get('asks') or []),
                                los=(None if los.get('id') is None else str(los['id']), los.get('side'),
                                     np.nan if los.get('price') is None else float(los['price']), los.get('outcome'))))
        guard('ws hour')
    return out


R_COLS = ['day', 'mid', 'sid', 'tx', 'settle_ms', 'exec_ms', 'first_recv_ns', 'kept_recv_ns', 'n_recv', 'n_contents', 'amount_filled', 'price_executed',
          'n_makers', 'tk_hash', 'tk_signer', 'tk_quote', 'tk_index_set', 'tk_price', 'tk_amount', 'tk_fee', 'tk_fee_type', 'tk_y_side', 'status', 'px_mismatch',
          'kept_content']
L_COLS = ['day', 'mid', 'sid', 'tx', 'settle_ms', 'exec_ms', 'first_recv_ns', 'mk_idx', 'n_makers', 'mk_hash', 'mk_signer', 'mk_quote', 'mk_index_set',
          'mk_price', 'shares', 'mk_fee', 'mk_fee_type', 'y_side', 'y_price', 'tk_hash', 'tk_signer', 'tk_y_side', 'status']


def _reconciles(x):
    """Amendment 1 item 4: a variant reconciles exactly iff taker.amount == amountFilled == sum(makers.amount) (integers)."""
    try:
        return len(x['makers']) >= 1 and int(x['taker']['amount']) == int(x['amountFilled']) == sum(int(m['amount']) for m in x['makers'])
    except (KeyError, TypeError, ValueError):
        return False


def unique_records(day, mid, rec):
    """One row per settlementId. Distinct canonical contents are variants: the KEPT one is the first variant (by receipt) that reconciles
    exactly, else the first (amendment 1 item 4); > 1 variant -> 'conflict'. px_mismatch: maker-weighted YES price != priceExecuted (YES)."""
    R, L = [], []
    for sid, rr in rec.items():
        rr.sort(key=lambda t: t[0])
        var = {}
        for t, c, x_ in rr:
            var.setdefault(c, (t, x_))
        kc, (kept_recv, x) = next(((c, v) for c, v in var.items() if _reconciles(v[1])), next(iter(var.items())))
        recv = rr[0][0]
        try:
            tk, mk = x['taker'], x['makers']
            ia, ip = int(x['amountFilled']), int(x['priceExecuted'])
            rec_ok = _reconciles(x) and int(tk['price']) == ip
            settle, ex = C.uuid7_ms(sid), iso_ms(x['executedAt'])
            tys = C.yes_space(tk['quoteType'], int(tk['outcome']['indexSet']), int(tk['price']) / C.E18)[0]
            legs = [(i, m, C.yes_space(m['quoteType'], int(m['outcome']['indexSet']), int(m['price']) / C.E18)) for i, m in enumerate(mk)]
            q = np.array([int(m['amount']) / C.E18 for m in mk]); yp_ = np.array([yp for _, _, (_, yp) in legs])
            tyes = ip / C.E18 if int(tk['outcome']['indexSet']) == 1 else 1 - ip / C.E18
            pxm = bool(q.sum() > 0 and abs(float((q * yp_).sum() / q.sum()) - tyes) > PX_TOL)
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            R.append(dict(sid=sid, status='parse_error', first_recv_ns=recv, n_recv=len(rr), n_contents=len(var), kept_content=kc, err=str(e)[:60]))
            continue
        status = '+'.join(f for f, on in (('conflict', len(var) > 1), ('amount_mismatch', not rec_ok)) if on) or 'ok'
        R.append(dict(day=day, mid=mid, sid=sid, tx=x['transactionHash'], settle_ms=settle, exec_ms=ex, first_recv_ns=recv, kept_recv_ns=kept_recv, n_recv=len(rr),
                      n_contents=len(var), amount_filled=ia / C.E18, price_executed=ip / C.E18, n_makers=len(mk), tk_hash=tk.get('hash'),
                      tk_signer=tk['signer'], tk_quote=tk['quoteType'], tk_index_set=int(tk['outcome']['indexSet']), tk_price=int(tk['price']) / C.E18,
                      tk_amount=int(tk['amount']) / C.E18, tk_fee=int(tk['fee']['amount']) / C.E18, tk_fee_type=tk['fee']['type'], tk_y_side=tys,
                      status=status, px_mismatch=pxm, kept_content=kc))
        for i, m, (ys, yp) in legs:
            L.append(dict(day=day, mid=mid, sid=sid, tx=x['transactionHash'], settle_ms=settle, exec_ms=ex, first_recv_ns=recv, mk_idx=i, n_makers=len(mk),
                          mk_hash=m['hash'], mk_signer=m['signer'], mk_quote=m['quoteType'], mk_index_set=int(m['outcome']['indexSet']),
                          mk_price=int(m['price']) / C.E18, shares=int(m['amount']) / C.E18, mk_fee=int(m['fee']['amount']) / C.E18,
                          mk_fee_type=m['fee']['type'], y_side=ys, y_price=yp, tk_hash=tk.get('hash'), tk_signer=tk['signer'], tk_y_side=tys, status=status))
    return pd.DataFrame(R, columns=R_COLS + (['err'] if any('err' in r for r in R) else [])), pd.DataFrame(L, columns=L_COLS)


def reparse(day, mid, E=None, raw_root=None, step1=None, window=None, pre_s=900, post_s=3600, tail_max_h=6, rest_stems=None, ws_stems=None):
    """Independent re-parse of ONE market from raw - only the hours around its window (padding CLIPPED by allowed_hour, never opened) plus a
    quiet-tail extension while the last hour still carries its receipts - compared with the builder outputs in E (if given).
    window=(ws_ms, we_ms) overrides the registry (tests)."""
    need(day in C.ALLOWED_DAYS or day == C.NOV8, f'day {day} is not a consumed day')
    A, st, hrs = defaultdict(Acc), Counter(), []
    raw_root = raw_root if raw_root is not None else (E or {}).get('raw_root')
    if window is None:
        EE = E if E is not None else dict(step1=Path(step1 or C.STEP1), _reg={}, gt_ws=gt_markets()[1] if day == C.GT_DAY else {})
        r = reg(EE, day).get(mid)
        need(r is not None, f'market {mid} has no window on {day}')
        ws, we = r[0], r[1]
    else:
        ws, we = window
    B = {}
    if E is not None:
        for k in ('records', 'legs', 'tob'):
            if day in E['P'][k]:
                B[k] = pd.read_parquet(E['P'][k][day], filters=[('mid', '=', int(mid))])
        for k in ('frames', 'levels', 'fills', 'orders'):
            if (day, mid) in E['P'][k]:
                B[k] = pd.read_parquet(E['P'][k][(day, mid)])
    a = A['rp_hours']
    if rest_stems is None:
        cand = set(stems_between(ws - pre_s * 1000, we + post_s * 1000))
        if 'records' in B:
            extra = {hour_of_ns(t) for t in B['records'].first_recv_ns}
            a.check(sum(not C.allowed_hour(s) for s in extra), 'builder holds {n} first receipts in non-consumed hours')
            cand |= extra
        rest_stems = sorted(s for s in cand if C.allowed_hour(s))
        st['hours_clipped'] += len(cand) - len(rest_stems)
    rec = rest_receipts(mid, rest_stems, raw_root, st, hrs)
    while hrs and hrs[-1][0] == rest_stems[-1] and hrs[-1][1] > 0 and st['tail_hours_added'] < tail_max_h:    # quiet-tail extension
        nxt = stems_between(stem_ms(hrs[-1][0]) + 3600000, stem_ms(hrs[-1][0]) + 3600000)[0]
        if not C.allowed_hour(nxt):
            st['tail_stopped_at_nonconsumed_hour'] += 1; st['hours_clipped'] += 1; break
        n0 = len(hrs)
        for k, v in rest_receipts(mid, [nxt], raw_root, st, hrs).items():
            rec[k].extend(v)
        st['tail_hours_added'] += 1; rest_stems = rest_stems + [nxt]
        if len(hrs) == n0:
            break
    a.check(st['needle_miss'], 'raw pre-filter missed {n} records of the market')
    R, L = unique_records(day, mid, rec)
    st.update(unique_records=len(R), legs=len(L), conflicts=int(hasf(R.status, 'conflict').sum()), amount_mismatch=int(hasf(R.status, 'amount_mismatch').sum()),
              parse_errors=int((R.status == 'parse_error').sum()), repolled=int((R.n_recv > 1).sum()), rest_hours_read=len(hrs),
              price_mismatch=int(R.px_mismatch.fillna(False).astype(bool).sum()))
    read = {s for s, _ in hrs}
    K = (ptick_unit(B['legs'].y_price.to_numpy(), B['legs'].y_ptick.to_numpy()) if 'legs' in B and len(B['legs']) else None) or 100
    if E is not None:
        a = A['rp_records_set']
        if 'records' not in B:
            a.fail('builder records missing')
        else:
            Bm = B['records']
            mine, theirs = set(R.sid), set(Bm.sid)
            scoped = {s for s, t in zip(Bm.sid, Bm.first_recv_ns) if hour_of_ns(t) in read}
            st['builder_records_outside_scan'] = len(theirs - scoped)
            d1, d2 = mine - theirs, scoped - mine
            (a.fail(f'records {day}_{mid}: raw has {len(mine)} unique records, builder {len(theirs)}: {len(d1)} missing from the builder (dropped), '
                    f'{len(d2)} builder records absent from raw') if d1 or d2 else a.ok())
            com = R[R.sid.isin(theirs)].merge(Bm, on='sid', suffixes=('', '_b'))
            b = A['rp_records_fields']
            ok_ = com[com.status != 'parse_error']
            for c in ('tx', 'settle_ms', 'exec_ms', 'first_recv_ns', 'amount_filled', 'price_executed', 'n_makers', 'tk_hash', 'tk_signer', 'tk_quote',
                      'tk_index_set', 'tk_price', 'tk_amount', 'tk_fee', 'tk_fee_type', 'tk_y_side'):
                alt = ok_.kept_recv_ns if c == 'first_recv_ns' else ok_[c]      # conflicts: first receipt of the id or of the kept variant
                b.check(sum(not (same(p, q) or same(p2, q)) for p, p2, q in zip(ok_[c], alt, ok_[c + '_b'])), f'records {day}_{mid}: {{n}} records whose {c} differs from raw')
            cl = com[com.status == 'ok']
            b.check(int((cl.n_recv != cl.n_recv_b).sum()), f'records {day}_{mid}: {{n}} clean records whose n_recv differs from the raw receipts')
            vc = [x for x in Bm.columns if 'variant' in x]
            if vc:      # amendment 1 item 4: the variant count is recorded
                cf = com[com.n_contents > 1]
                b.check(sum(not same(p, q) for p, q in zip(cf.n_contents, cf[vc[0]])), f'records {day}_{mid}: {{n}} conflicts whose variant count differs from raw')
            st['builder_variant_column'] = vc[0] if vc else None
            c = A['rp_records_status']
            for _, r in com.iterrows():
                fb, conflict = flags(r.status_b), r.n_contents > 1
                if conflict and 'conflict' not in fb:
                    c.fail(f'records {day}_{mid}: a settlementId with {r.n_contents} distinct canonical contents is status {r.status_b} '
                           f'(duplicate-key conflict merged silently)')
                elif 'amount_mismatch' in flags(r.status) or r.status == 'parse_error':
                    (c.fail(f'records {day}_{mid}: a record failing integer reconciliation / parsing is status ok') if r.status_b == C.STATUS_OK else c.ok())
                elif conflict:
                    keep = B['legs'][B['legs'].sid == r.sid].sort_values('mk_idx') if 'legs' in B else pd.DataFrame(columns=['shares'])
                    kept = json.loads(r.kept_content)['makers']
                    c.check(int(not (len(keep) == len(kept) and all(abs(int(m['amount']) / C.E18 - s) <= TOL_SH for m, s in zip(kept, keep.shares)))),
                            f'records {day}_{mid}: a conflict record does not keep the variant that reconciles exactly')
                elif not fb <= {C.STATUS_OK} | EXTERNAL_STATUSES:     # declared: a price gap is a diagnostic (px_gap), never a status
                    c.fail(f'records {day}_{mid}: a clean record is status {r.status_b}')
                else:
                    c.ok()
        a = A['rp_legs']
        if 'legs' not in B or not len(L):
            a.fail('builder legs missing or no raw legs')
        else:
            m = L.merge(B['legs'], on=['sid', 'mk_idx'], how='left', suffixes=('', '_b'), indicator=True)
            a.check(int((m._merge == 'left_only').sum()), f'legs {day}_{mid}: {{n}} raw legs missing from the builder (dropped)')
            m = m[m._merge == 'both']
            for c in ('mk_hash', 'mk_signer', 'mk_quote', 'mk_index_set', 'mk_price', 'shares', 'mk_fee', 'mk_fee_type', 'y_side', 'y_price', 'tk_hash',
                      'tk_signer', 'tk_y_side', 'n_makers'):
                a.check(sum(not same(p, q) for p, q in zip(m[c], m[c + '_b'])), f'legs {day}_{mid}: {{n}} legs whose {c} differs from raw')
            a.check(int((np.round(m.y_price * K).astype(np.int64) != m.y_ptick).sum()), f'legs {day}_{mid}: {{n}} legs whose y_ptick != round(raw y_price*{K})')
    # ---- WS frames / levels / tob
    if ws_stems is None:
        cand = set(stems_between(ws - pre_s * 1000, we + post_s * 1000)) | ({hour_of_ns(t) for t in B['frames'].recv_ns} if 'frames' in B else set())
        ws_stems = sorted(s for s in cand if C.allowed_hour(s))
        st['hours_clipped'] += len(cand) - len(ws_stems)
    FRm = ws_frames(mid, ws_stems, raw_root, st)
    st['frames'] = len(FRm)
    wread = set(ws_stems)
    if E is not None:
        a = A['rp_frames']
        if 'frames' not in B or 'levels' not in B:
            a.fail('builder depth frames/levels missing')
        else:
            BF = B['frames']; u0, u1 = BF.upd_ms.min(), BF.upd_ms.max()
            mine = Counter((f['upd_ms'], f['recv_ns'], f['conn']) for f in FRm if u0 <= f['upd_ms'] <= u1)
            theirs = Counter((u, r_, c_) for u, r_, c_ in zip(BF.upd_ms, BF.recv_ns, BF.conn) if hour_of_ns(r_) in wread)
            st['raw_frames_outside_builder_range'] = sum(1 for f in FRm if not u0 <= f['upd_ms'] <= u1)
            (a.ok() if mine == theirs else a.fail(f'frames {day}_{mid}: raw {sum(mine.values())} frame receipts vs builder {sum(theirs.values())} '
                                                 f'({sum((mine - theirs).values())} dropped, {sum((theirs - mine).values())} extra)'))
            fid = {(u, r_, c_): f for u, r_, c_, f in zip(BF.upd_ms, BF.recv_ns, BF.conn, BF.fid)}
            lvd = {f: sorted(zip(g.side, g.ptick, g.shares)) for f, g in B['levels'].groupby('fid')}
            bfr = BF.set_index('fid')
            nb = nc = 0
            for f in FRm:
                k = fid.get((f['upd_ms'], f['recv_ns'], f['conn']))
                if k is None:
                    continue
                ml = sorted([('bid', int(round(p * K)), q) for p, q in f['bids']] + [('ask', int(round(p * K)), q) for p, q in f['asks']])
                th = lvd.get(k, [])
                nb += int(len(ml) != len(th) or any(x[0] != y[0] or x[1] != y[1] or abs(x[2] - y[2]) > 1e-9 for x, y in zip(ml, th)))
                r_ = bfr.loc[k]
                pend = all((sorted(map(tuple, json.loads(r_[c]))) == f[k2]) if isinstance(r_[c], str) else not f[k2] for c, k2 in (('pend_bids', 'pb'), ('pend_asks', 'pa')))
                los = same(r_.los_id, f['los'][0]) and same(r_.los_side, f['los'][1]) and same(r_.los_price, f['los'][2], 1e-9) and same(r_.los_outcome, f['los'][3])
                nc += int(not (pend and los and int(r_.order_count) == f['order_count']))
            a.check(nb, f'levels {day}_{mid}: {{n}} frames whose levels differ from the raw bids/asks')
            a.check(nc, f'frames {day}_{mid}: {{n}} frames whose settlementsPending / lastOrderSettled / orderCount differ from raw')
            if 'tob' in B:
                TB = B['tob']
                t2 = Counter((u, r_, c_) for u, r_, c_ in zip(TB.upd_ms, TB.recv_ns, TB.conn) if hour_of_ns(r_) in wread and u0 <= u <= u1)
                (a.ok() if mine == t2 else a.fail(f'tob {day}_{mid}: frame receipts differ from raw ({sum((mine - t2).values())} dropped)'))
    # ---- lifecycle against raw-derived legs and frames
    if E is not None:
        if not ('fills' in B and 'orders' in B and len(L)):
            for n in ('rp_fill_hi', 'rp_cumulative', 'rp_H1', 'rp_H3', 'rp_capacity'):
                A[n].fail(f'builder lifecycle outputs missing for {day}_{mid} (or no raw legs)')
        else:
            F, O = B['fills'].copy(), B['orders'].copy()
            Lm = L[~hasf(L.status, 'conflict')].copy()
            Lm['h'], F['h'], O['h'] = Lm.mk_hash.str.lower(), F.mk_hash.str.lower(), O.mk_hash.str.lower()
            Lm['hi_ms'] = np.minimum(Lm.settle_ms.to_numpy(np.int64), Lm.first_recv_ns.to_numpy(np.int64) // 10 ** 6)
            j = F.merge(Lm[['sid', 'mk_idx', 'hi_ms']], on=['sid', 'mk_idx'], suffixes=('', '_raw'))
            A['rp_fill_hi'].check(int((j.hi_ms != j.hi_ms_raw).sum()), f'fills {day}_{mid}: {{n}} fills whose hi_ms != raw min(settle_ms, first receipt)')
            ch = conflict_hashes(Lm)
            La = L.assign(h=L.mk_hash.str.lower())

            def nbad(Lx):     # all legs (conflicts keep their first content) or reconciling legs only: either population, applied consistently
                jj = O[~O.h.isin(ch)].join(Lx.groupby('h').agg(n=('shares', 'size'), cum=('shares', 'sum')), on='h')
                return int(((jj.n_fills != jj.n) | ~np.isclose(jj.cum_shares, jj.cum.fillna(-1), atol=TOL_SH)).sum())
            nb = min(nbad(La), nbad(La[La.status == C.STATUS_OK]))
            A['rp_cumulative'].check(nb, f'orders {day}_{mid}: {{n}} rows whose n_fills/cum_shares differ from the raw per-hash fills')
            Lm = Lm[Lm.status == C.STATUS_OK]       # quantities for H3 / capacity only from records that reconcile
            A['rp_H1'].check(int(O[O.h.isin(ch).to_numpy() & ~hasf(O.status, 'identity_conflict')].h.nunique()),
                             f'orders {day}_{mid}: {{n}} raw identity conflicts not labelled identity_conflict')
            Lm['y_ptick'] = np.round(Lm.y_price * K).astype(np.int64)
            lo_map = dict(zip(zip(F.sid, F.mk_idx), F.lo_ms))
            Lm['lo_ms'] = [int(lo_map.get(k, -1)) for k in zip(Lm.sid, Lm.mk_idx)]
            Tm = np.array(sorted({f['upd_ms'] for f in FRm}), dtype=np.int64)
            a = A['rp_H3']
            a.check(sum(1 for x in Lm.lo_ms if x >= 0 and not (len(Tm) and Tm[min(int(np.searchsorted(Tm, x)), len(Tm) - 1)] == x)),
                    f'fills {day}_{mid}: {{n}} bounded lo_ms are not the venue time of any raw frame')
            frm = pd.DataFrame([dict(fid=i, upd_ms=f['upd_ms'], recv_ns=f['recv_ns'], status='ok') for i, f in enumerate(FRm)])
            lvm = pd.DataFrame([dict(fid=i, side=sd, ptick=int(round(p * K)), shares=q) for i, f in enumerate(FRm)
                                for sd, lv in (('bid', f['bids']), ('ask', f['asks'])) for p, q in lv], columns=['fid', 'side', 'ptick', 'shares'])
            hardr = 'x_lo_hard_ms' in F               # pinned: hard claims use x_lo_hard_ms; lo_ms (checked above) only refines
            if hardr:
                hmap = dict(zip(zip(F.sid, F.mk_idx), F.x_lo_hard_ms))
                Lm['lo_ms'] = [int(hmap.get(k, -1)) for k in zip(Lm.sid, Lm.mk_idx)]
            if len(frm) and hardr:
                U = units_of(Lm)
                viol, members, cap, _ = h3_scan(U, *book_arrays(frm, lvm, {k[1:] for k in U}))
                n = int(O[O.h.isin(members) & (O.status == C.STATUS_OK)].h.nunique())
                (a.fail(f'orders {day}_{mid}: raw H3 overfill at {viol[0]["price"]} (need {viol[0]["need"]:.4g} > shown {viol[0]["shown"]:.4g}); '
                        f'{n} member hashes status ok') if n else a.ok())
                capd = {k[0]: v for k, v in cap.items()}
                k = O[(O.status == C.STATUS_OK) & O.h.isin(capd)]
                A['rp_capacity'].check(int((k.post_lo_ms < k.h.map(capd)).sum()), f'orders {day}_{mid}: {{n}} ok rows below the raw H3 capacity posting bound')
                st.update(h3_violating_prices=len(viol), capacity_bounds=len(cap))
            else:
                a.fail(f'no raw frames or no x_lo_hard_ms for {day}_{mid}: H3 unverifiable')
                A['rp_capacity'].fail(f'no raw frames or no x_lo_hard_ms for {day}_{mid}')
    return dict(checks={n: A[n].res() for n in sorted(A)}, details={n: A[n].det() for n in sorted(A)},
                info={k: (v.item() if isinstance(v, np.generic) else v) for k, v in st.items()}, records=R, legs=L, frames=FRm)


# -------------------------------------------------------------------------------------------------------------------- falsifier
class FsTrace:
    """Counts every filesystem call whose path names a given marker (forbidden-hour falsifier: must stay at zero)."""
    def __init__(s, marker):
        s.marker, s.hits = marker, 0

    def __enter__(s):
        s.saved = (builtins.open, gzip.open, os.stat, os.scandir, os.listdir)

        def wrap(f):
            def g(p, *a, **k):
                s.hits += int(s.marker in str(p))
                return f(p, *a, **k)
            return g
        builtins.open, gzip.open, os.stat, os.scandir, os.listdir = map(wrap, s.saved)
        return s

    def __exit__(s, *a):
        builtins.open, gzip.open, os.stat, os.scandir, os.listdir = s.saved


class NotPlantable(Exception):
    pass


def _ok_date_name(name):
    return all(d in C.ALLOWED_DAYS or d == C.NOV8 for d in re.findall(r'(?<![0-9])(20\d{6})(?![0-9])', name))


def _mirror(src, dst):
    """Read-only mirror of the builder outputs: every (consumed-day) file symlinked; a mutation detaches only the file it edits."""
    for root, _d, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if _ok_date_name(f):
                os.symlink(Path(root) / f, dst / rel / f)


def _detach(p):
    """Never write through a link into the real outputs: drop the link first, then write a fresh file."""
    p = Path(p)
    if p.is_symlink():
        p.unlink()
    return p


def _edit(path, fn):
    df = fn(pd.read_parquet(path))
    df.to_parquet(_detach(path), index=False)


def _edit_json(path, fn):
    J = fn(json.loads(Path(path).read_text()))
    _detach(path).write_text(json.dumps(J))


def _first(mask, what):
    idx = np.flatnonzero(np.asarray(mask))
    if not len(idx):
        raise NotPlantable(what)
    return idx[0]


def falsify(E, raw_root=None, book_key=None, base_tmp=None):
    """Planted defects on a symlinked mirror of the builder outputs (the originals are never written). Each case names its check and
    expected reason; refused = the mutated run fails that check WITH the reason AND the unmutated run on the same scope does not carry
    that reason (paired baseline). Cases whose structure is absent report refused=None ('not plantable'), never a pass.
    book_key = (day, mid) of a lifecycle market whose raw hours exist (synthetic or real)."""
    raw_root = raw_root if raw_root is not None else E['raw_root']
    day, mid = book_key or E['life'][0]
    res, base = {}, {}
    bad_day = (dt.date(2026, 10, 27) + dt.timedelta(days=1)).strftime('%Y%m%d')     # first non-consumed day after Oct 27 (never printed)
    fdays = tuple(d for d in E['days'] if d in C.MEASURE_DAYS)

    def sub(out, days, life):
        bk, gt = ({d: v for d, v in g.items() if d in days} for g in (E['book'], E['gt']))
        if life == 'one':
            bk, gt = ({d: [m for m in v if (d, m) == (day, mid)] for d, v in g.items()} for g in (bk, gt))
        elif life == 'none':
            bk, gt = {}, {}
        return load(out, days=days, book=bk, gt=gt, gt_ws=E['gt_ws'], raw_root=raw_root, step1=E['step1'], gt_censored=E['gt_censored'])

    def run(Ex, check, via):
        r = run_checks(Ex, only=[check]) if via == 'run' else reparse(day, mid, Ex)
        d = r['details'].get(check, {})
        return r['checks'].get(check), (d.get('examples') or [d.get('reason', '')])

    def case(name, check, why, mutate, via='run', days=None, life='one', trace=None):
        days = tuple(days or (day,))
        with tempfile.TemporaryDirectory(dir=base_tmp) as tmp:
            dst = Path(tmp) / 'out'
            _mirror(E['out'], dst)
            try:
                mutate(dst)
            except NotPlantable as e:
                res[name] = dict(check=check, refused=None, expected_reason=why, got=[f'not plantable here: {e}']); return
            except Exception as e:            # a missing table / column fails the case closed, never the run
                res[name] = dict(check=check, refused=None, expected_reason=why, got=[f'not plantable here: {type(e).__name__}: {e}']); return
            try:
                if trace:
                    with FsTrace(trace) as tr:
                        ok, ex = run(sub(dst, days, life), check, via)
                    ex = ex + [f'fs calls naming the non-consumed day = {tr.hits}']
                else:
                    ok, ex = run(sub(dst, days, life), check, via)
            except Exception as e:
                res[name] = dict(check=check, refused=False, expected_reason=why, got=[f'check run crashed: {type(e).__name__}: {e}']); return
        key = (check, days, life, via)
        if key not in base:
            try:
                base[key] = run(sub(E['out'], days, life), check, via)
            except Exception as e:
                base[key] = (None, [f'baseline crashed: {type(e).__name__}: {e} ({why})'])     # carries the reason -> the case is never 'clean'
        clean = not any(why in x for x in base[key][1])
        hit = ok is False and any(why in x for x in ex) and (trace is None or ex[-1].endswith('= 0'))
        res[name] = dict(check=check, refused=bool(hit and clean), expected_reason=why, baseline_clean=clean, got=ex[:2])

    P = lambda dst, k: {'records': dst / f'records_{day}.parquet', 'legs': dst / f'legs_{day}.parquet', 'tob': dst / f'tob_{day}.parquet',
                        'fills': dst / 'life' / f'{day}_{mid}_fills.parquet', 'orders': dst / 'life' / f'{day}_{mid}_orders.parquet',
                        'json': dst / 'life' / f'{day}_{mid}.json', 'frames': dst / 'depth' / f'{day}_{mid}_frames.parquet',
                        'levels': dst / 'depth' / f'{day}_{mid}_levels.parquet'}[k]

    def setcol(k, col, rowsel, val):
        """Set col on the FIRST row selected by rowsel(df) (val: value or function of the old value)."""
        def f(dst):
            def g(df):
                i = df.index[_first(rowsel(df), f'no row for {k}.{col}')]
                df.loc[i, col] = val(df.loc[i, col]) if callable(val) else val
                return df
            _edit(P(dst, k), g)
        return f

    def allrows(k, col, rowsel, val):
        def f(dst):
            def g(df):
                m = np.asarray(rowsel(df)); _first(m, f'no row for {k}.{col}')
                df.loc[m, col] = val(df.loc[m, col]) if callable(val) else val
                return df
            _edit(P(dst, k), g)
        return f

    def lifecycle_view(dst):
        F, O, FR, LV = (pd.read_parquet(P(dst, k)) for k in ('fills', 'orders', 'frames', 'levels'))
        L = pd.read_parquet(P(dst, 'legs'), filters=[('mid', '=', int(mid))])
        F['h'], O['h'], L['h'] = F.mk_hash.str.lower(), O.mk_hash.str.lower(), L.mk_hash.str.lower()
        if 'status' in LV:
            LV = LV[usable(LV.status)]
        cl = set(zip(L.sid[only_flags(L.status, {C.STATUS_OK} | EXTERNAL_STATUSES)], L.mk_idx[only_flags(L.status, {C.STATUS_OK} | EXTERNAL_STATUSES)]))
        Fv = F[np.array([k in cl for k in zip(F.sid, F.mk_idx)], dtype=bool)]
        U = units_of(Fv)
        T, S = book_arrays(FR, LV, {k[1:] for k in units_of(F)})
        return F, O, FR, LV, L, U, h3_scan(U, T, S), h3_scan(units_of(F), T, S)[3]     # (strict scan, permissive shown-zero)

    ok_ = lambda df: df.status == C.STATUS_OK

    # ---- the contract's falsifier list (section 8) + the brief's additions
    def overfill(dst):          # H3: shown depth under a proven-resting order shrunk below its later fills; the order stays ok
        F, O, FR, LV, L, U, _s, _z = lifecycle_view(dst)
        for (h, sd, pt), u in U.items():
            if (O[O.h == h].status == C.STATUS_OK).all() and u['maxlo'] >= u['hmin'] and (u['lo'] >= u['hmin']).any():
                fr = FR[(FR.upd_ms >= u['hmin']).to_numpy() & (FR.upd_ms <= u['maxlo']).to_numpy() & usable(FR.status)].fid
                LV.loc[LV.fid.isin(fr) & (LV.side == sd) & (LV.ptick == pt), 'shares'] = 0.0
                LV.to_parquet(_detach(P(dst, 'levels')), index=False)
                return
        raise NotPlantable('no ok order with a proven-resting span')
    case('overfill_H3', 'H3_overfill_flagged', 'H3 overfill', overfill)
    case('fill_after_removal', 'removal_after_last_fill', 'fill after the removal window',
         allrows('orders', 'removal_hi_ms', lambda O: ok_(O) & (O.n_fills >= 2), lambda v: v * 0 + 1))
    case('identity_conflict_merged', 'H1_identity', 'not labelled identity_conflict', lambda dst: setcol(
        'legs', 'mk_signer', lambda L: (L.mid == mid) & L.mk_hash.str.lower().isin(
            set(pd.read_parquet(P(dst, 'orders')).query("status == 'ok' and n_fills >= 2").mk_hash.str.lower())), '0x' + '9' * 40)(dst))
    with tempfile.TemporaryDirectory(dir=base_tmp) as tmp:          # instrument positive controls, then the forbidden-hour path
        with FsTrace('fstrace_probe') as probe:
            (Path(tmp) / 'fstrace_probe').exists()
            try:
                gzip.open(Path(tmp) / 'fstrace_probe.gz', 'rt').close()
            except OSError:
                pass
        p = Path(tmp) / 'rest' / f'{day}_00.jsonl.gz'
        p.parent.mkdir(parents=True)
        payload = json.dumps(dict(data=[dict(market=dict(id=mid), settlementId='x')]), separators=(',', ' : '))   # '"id" : mid' defeats the needle
        with gzip.open(p, 'wt') as f:
            f.write(json.dumps(dict(path='/v1/orders/matches', status=200, recv_ns=1, raw=payload)) + '\n')
        stn = Counter()
        rest_receipts(mid, [p.name[:11]], Path(tmp), stn, [], full=lambda *a: None)
    res['instrument_controls'] = dict(check='FsTrace + needle-miss counter', refused=bool(probe.hits >= 2 and stn['needle_miss'] == 1),
                                      expected_reason='each instrument fires on a planted positive',
                                      got=[f'FsTrace probe hits = {probe.hits} (>= 2 expected); needle misses counted = {stn["needle_miss"]} (1 expected)'])
    stem = bad_day + '_00'
    with FsTrace(bad_day) as tr:
        raised = []
        for fn in (lambda: rest_receipts(mid, [stem], raw_root, Counter(), []), lambda: ws_frames(mid, [stem], raw_root, Counter())):
            try:
                fn(); raised.append(False)
            except C.Forbidden as e:
                raised.append('not on a consumed day' in str(e))
        ws_ = int(dt.datetime(2026, 10, 27, 23, 55, tzinfo=UTC).timestamp() * 1000)
        r = reparse('20261027', mid, None, raw_root=raw_root, window=(ws_, ws_ + 300000))
    res['forbidden_hour_path'] = dict(check='pf_mt_common.hour_path via raw_path', refused=bool(all(raised) and tr.hits == 0 and r['info'].get('hours_clipped', 0) >= 2),
                                      expected_reason='Forbidden before any open; padding clipped unopened',
                                      got=[f'Forbidden raised (rest, ws) = {raised}; filesystem calls naming the day = {tr.hits}; padding hours clipped = '
                                           f'{r["info"].get("hours_clipped", 0)}'])
    case('dropped_leg_row', 'legs_complete', 'dropped', lambda dst: _edit(P(dst, 'legs'), lambda L: L.drop(L.index[_first(L.mid == mid, 'no leg')])))
    case('dropped_fill_row', 'fills_complete', 'dropped', lambda dst: _edit(P(dst, 'fills'), lambda F: F.iloc[1:]))
    case('dropped_record_vs_raw', 'rp_records_set', 'missing from the builder', lambda dst: _edit(
        P(dst, 'records'), lambda R: R.drop(R.index[_first((R.mid == mid) & ok_(R), 'no ok record')])), via='reparse')
    case('observed_cancel', 'removal_class_evidence', 'observed removal', lambda dst: [setcol('orders', c, lambda O: O.index == O.index[0], v)(dst)
                                                                                      for c, v in (('removal_set', 'cancelled_or_expired'), ('removal_class', 'observed'))])
    case('time_inversion', 'fill_time_order', 'time inversion', lambda dst: _edit(P(dst, 'fills'), lambda F: F.assign(
        lo_ms=np.where(np.arange(len(F)) == _first((F.lo_ms >= 0) & ok_(F), 'no bounded ok fill'), F.hi_ms + 5, F.lo_ms))))
    case('dup_key_conflict_merged', 'rp_records_status', 'conflict merged silently', lambda dst: [
        allrows(k, 'status', lambda X: (X.mid == mid).to_numpy() & hasf(X.status, 'conflict'), 'ok')(dst) for k in ('records', 'legs')], via='reparse')
    case('conflict_wrong_variant', 'rp_records_status', 'does not keep the variant that reconciles exactly', allrows(
        'legs', 'shares', lambda L: (L.mid == mid).to_numpy() & hasf(L.status, 'conflict'), lambda v: v * 0.5), via='reparse')
    case('dup_key_rows_ok', 'records_unique_key', 'without status conflict', lambda dst: _edit(
        P(dst, 'records'), lambda R: pd.concat([R, R.iloc[[_first((R.mid == mid) & ok_(R), 'no ok record')]]], ignore_index=True)))

    def sens(O):
        m = (O.mk_hash == O.mk_hash.iloc[_first(ok_(O), 'no ok order')])
        O.loc[m, 'removal_class'] = 'constrained'
        O.loc[m & (O.scenario == FIFO_FAMILY[0]), 'removal_lo_ms'] += 7
        return O
    case('sensitive_labelled_constrained', 'class_scenario_consistency', 'differs across scenarios but is labelled constrained',
         lambda dst: _edit(P(dst, 'orders'), sens))
    f, g = IFACE['maker_market'], IFACE['maker_day']
    case('econ_total_ne_components', 'econ_conservation', 'pnl != gross - fees', lambda dst: _edit(dst / 'econ/maker_market.parquet', lambda M: M.assign(
        **{f['pnl']: M[f['pnl']] + np.where(np.arange(len(M)) == _first(M[f['status']] == 'ok', 'no ok maker_market row'), 0.5, 0)})), days=fdays, life='none')
    # ---- one planted defect for every other check (each check proves it can fire)
    case('status_count_unreported', 'status_counts_reported', 'no receipt map equals', setcol('legs', 'status', lambda L: L.index == L.index[-1], 'xx_unreported'))
    case('schema_dtype', 'schema', 'wrong dtype', lambda dst: _edit(P(dst, 'fills'), lambda F: F.assign(hi_ms=F.hi_ms.astype(float))))
    case('schema_extra_col', 'schema', 'extra non-x_', lambda dst: _edit(P(dst, 'orders'), lambda O: O.assign(note='x')))
    case('schema_missing_col', 'schema', "missing ['lo_ms']", lambda dst: _edit(P(dst, 'fills'), lambda F: F.drop(columns=['lo_ms'])))   # run must not crash
    case('hi_off_by_one', 'fill_hi_rule', 'hi_ms != min', lambda dst: _edit(P(dst, 'fills'), lambda F: F.assign(hi_ms=F.hi_ms + 1)))

    def hard_bump(F):
        if 'x_lo_hard_ms' not in F:
            raise NotPlantable('fills carry no x_lo_hard_ms')
        i = F.index[_first(F.x_lo_hard_ms >= 0, 'no bounded x_lo_hard_ms')]
        F.loc[i, 'x_lo_hard_ms'] = F.loc[i, 'x_lo_hard_ms'] + 1
        return F
    case('lo_hard_wrong', 'fill_hi_rule', 'x_lo_hard_ms != hi_ms - MAX_LAG_MS', lambda dst: _edit(P(dst, 'fills'), hard_bump))
    case('size_exact_below_lb', 'size_bounds', 'size_exact < size_lb', setcol('orders', 'size_exact', ok_, lambda v: -1.0))
    case('post_after_first_fill', 'posting_before_first_fill', 'after the first fill hi', allrows('orders', 'post_hi_ms', ok_, lambda v: v + 10 ** 7))
    case('unbounded_lo_status_ok', 'fill_lo_status', 'disagree', setcol('fills', 'status', lambda F: F.lo_ms < 0, 'ok'))

    def cov_bump(J):
        maps = []
        class_maps(J, maps)
        if not maps:
            raise NotPlantable('no class map in the lifecycle receipt')
        need(_bump_first_map(J), 'class map not reachable')
        return J
    case('coverage_wrong', 'coverage_arithmetic', f'coverage {day}_{mid}: reported class map', lambda dst: _edit_json(P(dst, 'json'), cov_bump))
    def gt_bump(J):          # e2e 4: posting flat or nested by class -> bump the first numeric 'failed' (the total, else a class block)
        b = ((J.get('scenarios') or {}).get(FIFO_FAMILY[0]) or {}).get('posting')
        tg = ([b] if isinstance(b, dict) and _num(b.get('failed')) else
              [v for c, v in (b.items() if isinstance(b, dict) else ()) if c in C.CLASSES and isinstance(v, dict) and _num(v.get('failed'))])
        if not tg:
            raise NotPlantable('no numeric failed in scenarios.<S>.posting (flat or nested by class)')
        tg[0]['failed'] += 1
        return J
    def size_blk(J, sub):    # the FIFO scenario's size.<sub> class blocks (nested form only)
        b = (((J.get('scenarios') or {}).get(FIFO_FAMILY[0]) or {}).get('size') or {}).get(sub)
        if not _is_by_class(b):
            raise NotPlantable(f'scores.json size.{sub} is not nested by class')
        return b

    def lb_by_size_class(J):     # contract §7 (e2e 5): the lower bound is always 'observed' -> the same counts under another class must fail
        b = size_blk(J, 'lb')
        if not (isinstance(b.get(C.OBSERVED), dict) and _num(b[C.OBSERVED].get('n')) and b[C.OBSERVED]['n'] > 0):
            raise NotPlantable('size.lb holds no observed block with n > 0')
        b[C.SENSITIVE] = b.pop(C.OBSERVED)
        return J

    def exact_stated_only(J):    # contract §7 (e2e 5): exact-size n is EVERY scored order -> n counting stated sizes only must fail
        b = size_blk(J, 'exact')
        v = next((v for c, v in b.items() if c in C.CLASSES and isinstance(v, dict) and _num(v.get('n')) and _num(v.get('not_stated'))
                  and v['not_stated'] > 0), None)
        if v is None:
            raise NotPlantable('size.exact holds no class block with not_stated > 0')
        v['n'] -= v['not_stated']
        return J

    def removal_stated_only(J):  # declared (e2e 6): removal.exhausted_only n is EVERY scored order -> n over exhausted-only rows must fail
        b = (((J.get('scenarios') or {}).get(FIFO_FAMILY[0]) or {}).get('removal') or {}).get('exhausted_only')
        cand = ([] if not isinstance(b, dict) else [v for c, v in b.items() if c in C.CLASSES and isinstance(v, dict)] if _is_by_class(b) else [b])
        v = next((v for v in cand if _num(v.get('n')) and _num(v.get('not_exhausted_only')) and v['not_exhausted_only'] > 0), None)
        if v is None:
            raise NotPlantable('removal.exhausted_only holds no block with not_exhausted_only > 0')
        v['n'] -= v['not_exhausted_only']
        return J
    gt_cases = (('gt_score_wrong', gt_bump), ('gt_size_lb_not_observed', lb_by_size_class), ('gt_size_exact_stated_only', exact_stated_only),
                ('gt_removal_stated_only', removal_stated_only))
    if C.GT_DAY in E['days'] and 'scores' in E['P']['misc']:       # the scores path as loaded (pinned: truth/v2/scores.json)
        rel_sc = E['P']['misc']['scores'].relative_to(E['out'])
        for nm_, fn_ in gt_cases:
            case(nm_, 'gt_scores', 'gt scores', lambda dst, fn_=fn_: _edit_json(dst / rel_sc, fn_), days=(C.GT_DAY,), life='all')
    else:                                                             # fail closed: never silently skipped
        for nm_, _fn in gt_cases:
            res[nm_] = dict(check='gt_scores', refused=None, expected_reason='gt scores', got=['not plantable here: no GT day or no truth scores.json loaded'])
    pdays = ((C.ID_DAY,) if C.ID_DAY in E['days'] else ()) + fdays      # e2e 4: the recompute covers the ID day too, as in the full run
    case('pnl_sign_flip', 'econ_pnl_recompute', 'independent recompute', lambda dst: _flip_pnl(dst, pdays), days=pdays, life='none')

    def fee_at_executed(dst):     # e2e 4 positive control: one SHARES taker fee converted at priceExecuted, not at each leg's own price
        f = IFACE['maker_market']
        M = pd.read_parquet(dst / 'econ/maker_market.parquet')
        dd = M[f['day']].map(lambda v: str(v).replace('-', '')[:8])
        for d in pdays:
            cols = ['sid', 'mid', 'status', 'tk_signer', 'tk_index_set', 'tk_price', 'price_executed', 'tk_fee', 'tk_fee_type']
            R = pd.read_parquet(dst / f'records_{d}.parquet', columns=[c for c in cols if c in pq.read_schema(dst / f'records_{d}.parquet').names])
            R = R[(R.status == C.STATUS_OK) & (R.tk_fee_type == 'SHARES') & (R.tk_fee.astype(float) > 0)].head(200)
            if not len(R):
                continue
            L = pd.read_parquet(dst / f'legs_{d}.parquet', columns=['sid', 'shares', 'y_price', 'status'], filters=[('sid', 'in', list(R.sid))])
            L = L[L.status == C.STATUS_OK]
            tis = L.sid.map(R.drop_duplicates('sid').set_index('sid').tk_index_set).to_numpy()
            L = L.assign(w=L.shares * np.where(tis == 1, L.y_price, 1 - L.y_price))
            pxw = L.groupby('sid').w.sum() / L.groupby('sid').shares.sum()
            pe = R.set_index('sid')['price_executed' if 'price_executed' in R else 'tk_price']
            gap = (pe.reindex(pxw.index) - pxw).dropna()
            for sid, g_ in gap.items():
                r = R[R.sid == sid].iloc[0]
                dlt = float(r.tk_fee) * float(g_)
                sel = ((M[f['addr']].str.lower() == str(r.tk_signer).lower()) & (M[f['mid']] == r.mid) & (M[f['status']] == 'ok') & (dd == d)).to_numpy()
                if abs(dlt) > 10 * TOL_USD and sel.any():
                    M.loc[M.index[np.flatnonzero(sel)[0]], f['fees_px']] += dlt
                    M.to_parquet(_detach(dst / 'econ/maker_market.parquet'), index=False)
                    return
        raise NotPlantable('no ok SHARES-fee taker record whose priceExecuted differs from its legs\' own prices')
    case('fee_fill_at_executed_price', 'econ_pnl_recompute', 'fee_fill != recompute', fee_at_executed, days=pdays, life='none')
    case('missing_life_json', 'outputs_present', 'life_json output missing', lambda dst: P(dst, 'json').unlink())
    case('empty_status', 'status_present', 'null or empty status', setcol('records', 'status', lambda R: R.mid == mid, ''))
    case('taker_amount_mismatch', 'amounts_reconcile', 'taker.amount != amountFilled', setcol('records', 'tk_amount', lambda R: (R.mid == mid) & ok_(R), lambda v: v + 1))
    case('yes_side_flip', 'yes_space_mapping', '!= yes_space', setcol('legs', 'y_side', lambda L: L.mid == mid, lambda v: 'bid' if v == 'ask' else 'ask'))
    case('settle_not_uuid7', 'clock_fields', 'settle_ms != UUIDv7', setcol('records', 'settle_ms', lambda R: R.mid == mid, lambda v: v + 1))
    other = next(d for d in C.ALLOWED_DAYS if d != day)
    case('day_column_wrong', 'day_assignment', 'day column != the file day', setcol('records', 'day', lambda R: R.mid == mid, other))
    case('t_rel_off', 'legs_phase_clock', 't_rel_s equals none', setcol('legs', 't_rel_s', lambda L: (L.mid == mid) & ok_(L), lambda v: v + 0.5))
    for nm, which, why in (('upper_boundary_not_censored', 1, 'ends in a non-consumed hour'), ('lower_boundary_not_censored', 0, 'starts right after')):
        try:
            bnd = [(d, m) for d in E['days'] if d in E['P']['records'] for m, r in reg(E, d).items()
                   if not C.allowed_hour(stems_between(r[which] - (1 - which), r[which] - (1 - which))[0])]
            bnd = [(d, m) for d, m in bnd if (pd.read_parquet(E['P']['records'][d], columns=['mid']).mid == m).any()]
        except Exception as e:                # unreadable registry / records: fail closed
            bnd = []; res[nm] = dict(check='boundary_censored', refused=None, expected_reason=why, got=[f'not plantable here: {type(e).__name__}: {e}'])
            continue
        if bnd:
            bd, bm = bnd[0]
            case(nm, 'boundary_censored', why, lambda dst, bd=bd, bm=bm: _edit(
                dst / f'records_{bd}.parquet', lambda R: R.assign(status=np.where(R.mid == bm, 'ok', R.status))), days=(bd,), life='none')
        else:
            res[nm] = dict(check='boundary_censored', refused=None, expected_reason=why, got=['not plantable here: no such boundary market'])
    case('boundary_false_flag', 'boundary_censored', 'false flag', allrows('records', 'status', lambda R: R.mid == mid, 'boundary_censored'), life='none')
    try:
        cens = [(d, m) for d in fdays if d in E['P']['records'] for m in pd.read_parquet(E['P']['records'][d], columns=['mid', 'status']).query(
            "status.str.contains('boundary_censored')", engine='python').mid.unique()]
    except Exception:
        cens = []
    if cens and 'maker_market' in E['P']['econ']:
        case('econ_includes_censored', 'econ_conservation', 'boundary_censored markets', lambda dst: _edit(dst / 'econ/maker_market.parquet', lambda M: M.assign(
            **{f['status']: np.where([(d, int(m)) in set(cens) for d, m in zip(M[f['day']], M[f['mid']])], 'ok', M[f['status']])})), days=fdays, life='none')
    else:
        res['econ_includes_censored'] = dict(check='econ_conservation', refused=None, expected_reason='boundary_censored markets',
                                             got=['not plantable here: no boundary_censored FILL-day market'])
    def decl_reversed(dst):   # two declared extract flags joined AGAINST the extractor's declared order
        order = declared_order([json.loads((dst / f'{k}_{day}.json').read_text()) for k in ('extract', 'inventory') if (dst / f'{k}_{day}.json').exists()], 'records')
        if not order or len(order) < 2:
            raise NotPlantable('the extract receipt declares fewer than two flags')
        setcol('records', 'status', lambda R: R.mid == mid, order[1] + '+' + order[0])(dst)
    case('extract_flag_order', 'status_present', 'declared flag order', decl_reversed)
    case('extract_flag_undeclared', 'status_present', 'undeclared flag', setcol('records', 'status', lambda R: R.mid == mid, 'zz_undeclared'))
    case('ok_joined', 'status_present', 'bad flag syntax', setcol('records', 'status', lambda R: R.mid == mid, 'ok+conflict'))

    case('f_before_wrong', 'fill_lo_frame', 'lo_ms != upd_ms of frame f_before', setcol('fills', 'f_before', lambda F: F.lo_ms >= 0, lambda v: v + 1))

    def ev_shift(dst):        # the EVIDENCE frame (pinned: x_evidence_fid; legacy f_after) moved far outside [hi - 2000 ms, first frame >= hi]
        evc = 'x_evidence_fid' if 'x_evidence_fid' in pq.read_schema(P(dst, 'fills')).names else 'f_after'
        setcol('fills', evc, lambda F: F.lo_ms >= 0, lambda v: v + 10)(dst)
    case('evidence_outside_window', 'fill_lo_frame', 'outside [hi - 2000 ms', ev_shift)
    case('removal_set_unsorted', 'removal_class_evidence', 'not the |-joined sorted', setcol(
        'orders', 'removal_set', lambda O: O.removal_set.str.contains('|', regex=False), lambda v: '|'.join(sorted(v.split('|'), reverse=True))))
    case('lifecycle_status_unsorted', 'status_present', 'not sorted', setcol('fills', 'status', lambda F: F.index == F.index[0], 'leg_conflict+fill_lo_unbounded'))

    def pers_obs(O):          # an observed persistence label on an order whose certain span is empty (-1/-1)
        h = O.mk_hash.iloc[_first(ok_(O) & (O.persist_lo_ms < 0), 'no ok order with an empty certain span')]
        O.loc[O.mk_hash == h, 'persist_class'] = C.OBSERVED
        return O
    case('persist_observed_empty', 'H2_presence', 'observed persistence with an empty', lambda dst: _edit(P(dst, 'orders'), pers_obs))

    def fx_bump(J):          # e2e 4: always a REAL change (number +5; first numeric map entry +5; an empty map -> {'bad_frame': 5}), verified
        if not _find_key(J, 'frames_excluded'):
            raise NotPlantable('the lifecycle summary holds no frames_excluded')
        before = json.dumps(J, sort_keys=True, default=str)

        def walk(o):
            if isinstance(o, dict):
                for k in list(o):
                    v = o[k]
                    if k == 'frames_excluded' and _num(v):
                        o[k] = v + 5
                    elif k == 'frames_excluded' and isinstance(v, dict):
                        nk = next((x for x, y in v.items() if _num(y)), None)
                        if nk is not None:
                            v[nk] += 5
                        elif not v:
                            o[k] = {'bad_frame': 5}
                    else:
                        walk(v)
            elif isinstance(o, list):
                for x in o:
                    walk(x)
        walk(J)
        if json.dumps(J, sort_keys=True, default=str) == before:
            raise NotPlantable('frames_excluded holds neither a number nor a numeric / empty map: nothing to plant')
        return J
    case('frames_excluded_wrong', 'status_counts_reported', 'frames_excluded', lambda dst: _edit_json(P(dst, 'json'), fx_bump))
    case('scenario_row_dropped', 'orders_complete', 'filled hashes missing', lambda dst: _edit(P(dst, 'orders'), lambda O: O.drop(O.index[0])))
    case('n_fills_wrong', 'orders_cumulative', 'differ from their fills', setcol('orders', 'n_fills', ok_, lambda v: v + 1))
    case('persist_beyond_last_fill', 'H2_presence', 'persistence window contradicts',
         lambda dst: _edit(P(dst, 'orders'), lambda O: O.assign(persist_hi_ms=np.where(np.arange(len(O)) == _first(ok_(O) & (O.persist_hi_ms >= 0), 'no persistence window'),
                                                                                      O.last_hi_ms + 1000, O.persist_hi_ms))))

    def no_cap(dst):
        if not lifecycle_view(dst)[6][2]:
            raise NotPlantable('no H3 capacity bound in this market')
        allrows('orders', 'post_lo_ms', ok_, -1)(dst); allrows('orders', 'post_method', ok_, 'none')(dst)
    case('capacity_ignored', 'posting_capacity_bound', 'below the H3 capacity bound', no_cap)
    case('removal_lo_early', 'removal_lo_bound', 'removal_lo_ms < the last fill lo_ms', allrows('orders', 'removal_lo_ms', ok_, -1))

    def prio_sets(dst):
        """(hashes exhausted by price priority at their last fill [clean records], hashes with shown-zero evidence, old mk_idx-rule hashes)."""
        F, O, FR, LV, L, U, _s, zero = lifecycle_view(dst)
        cl = L[only_flags(L.status, {C.STATUS_OK} | EXTERNAL_STATUSES)]
        pe, last = priority_evidence(cl, L), L.groupby('h').settle_ms.max()
        hk = L.drop_duplicates(['sid', 'mk_idx']).set_index(['sid', 'mk_idx'])
        ev_last = {hk.h[k] for k in pe['ev'] if hk.settle_ms[k] >= last[hk.h[k]]}
        old = set()               # the retired rule: index i < last, YES price no worse than a later maker's
        for sid, g in cl.sort_values(['sid', 'mk_idx']).groupby('sid'):
            p, buy = g.y_ptick.to_numpy(), g.tk_y_side.iloc[0] == 'bid'
            old.update(g.h.iloc[i] for i in range(len(p) - 1) if (p[i] <= p[i + 1:].max() if buy else p[i] >= p[i + 1:].min()))
        return O, ev_last, {k[0] for k, v in zero.items() if v}, old

    def prio_drop(dst):
        O, ev_last, _z, _o = prio_sets(dst)
        if not ev_last:
            raise NotPlantable('no price-priority exhaustion evidence at a last fill in this market')
        allrows('orders', 'removal_set', lambda O: ok_(O) & O.scenario.isin(FIFO_FAMILY) & O.mk_hash.str.lower().isin(ev_last),
                'cancelled_or_expired|end_censored')(dst)
    case('priority_evidence_ignored', 'priority_exhaustion', 'ignore price-priority exhaustion evidence', prio_drop)

    def mk_idx_claim(dst):    # an exhaustion claim justified only by makers[] order (retired rule) must be refused
        O, ev_last, zero, old = prio_sets(dst)
        okh = set(O[ok_(O)].mk_hash.str.lower())
        cand = sorted((old - ev_last - zero) & okh)
        if not cand:
            raise NotPlantable('no hash exhausted by the mk_idx rule alone')
        def g(O):
            m = O.mk_hash.str.lower().isin(cand[:1]) & O.scenario.isin(FIFO_FAMILY)
            O.loc[m, 'removal_set'] = 'exhausted'; O.loc[m, 'size_exact'] = O.loc[m, 'cum_shares']
            return O
        _edit(P(dst, 'orders'), g)
    case('mk_idx_only_exhaustion', 'priority_exhaustion', 'without price-priority or shown-zero evidence', mk_idx_claim)

    def nopriority_size(dst):
        O, _e, zero, _o = prio_sets(dst)
        def g(O):
            m = np.flatnonzero((ok_(O) & (O.scenario == 'S_NOPRIO') & ~O.mk_hash.str.lower().isin(zero) & O.size_exact.isna()).to_numpy())
            if not len(m):
                raise NotPlantable('no S_NOPRIO row without an exact size')
            O.loc[O.index[m[0]], 'size_exact'] = O.loc[O.index[m[0]], 'cum_shares']
            return O
        _edit(P(dst, 'orders'), g)
    case('size_exact_unjustified', 'priority_exhaustion', 'state size_exact without', nopriority_size)

    def t1_bump(J):          # every T1 count (any version: settle-order / window-certain) shifted by +7
        hit = []
        num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)

        def bump(v, d=0):
            if isinstance(v, dict):
                for kk in list(v):
                    if re.match(r'(?i)^(n|count|n_\w+|violations?)$', str(kk)) and num(v[kk]):
                        v[kk] += 7; hit.append(kk)
                    elif isinstance(v[kk], (dict, list)) and d < 12:
                        bump(v[kk], d + 1)
            elif isinstance(v, list):
                for x in v:
                    bump(x, d + 1)

        def walk(o, d=0):
            items = o.items() if isinstance(o, dict) else (enumerate(o) if isinstance(o, list) else ())
            for k, v in list(items):
                if isinstance(k, str) and re.match(IFACE['t1_key'], k):
                    if num(v):
                        o[k] = v + 7; hit.append(k)
                    else:
                        bump(v)
                elif d < 12:
                    walk(v, d + 1)
        walk(J)
        if not hit:
            raise NotPlantable('no T1 report in the lifecycle summary')
        return J
    case('t1_reported_wrong', 'priority_tests_T1_T2', 'T1 reported', lambda dst: _edit_json(P(dst, 'json'), t1_bump))
    case('ahead_lo_wrong', 'priority_ahead_lo', 'ahead_lo is not the price-priority value at the first fill',
         setcol('orders', 'ahead_lo', lambda O: ok_(O) & O.scenario.isin(FIFO_FAMILY), 123.25))

    def ahead_first(dst):    # a non-first fill's value stated as the order's ahead_lo (pinned: the FIRST fill's)
        O = pd.read_parquet(P(dst, 'orders')); L = pd.read_parquet(P(dst, 'legs'), filters=[('mid', '=', int(mid))])
        L['h'] = L.mk_hash.str.lower()
        L['hi'] = np.minimum(L.settle_ms.to_numpy(np.int64), L.first_recv_ns.to_numpy(np.int64) // 10 ** 6)
        pe = priority_evidence(L[only_flags(L.status, {C.STATUS_OK} | EXTERNAL_STATUSES)], L)
        first = L[L.hi.to_numpy() == L.groupby('h').hi.transform('min').to_numpy()]
        nz = {h for s_, i_, h in zip(first.sid, first.mk_idx, first.h) if pe['ahead'].get((s_, int(i_)), 0.0) > 0}
        m = (ok_(O) & O.scenario.isin(FIFO_FAMILY) & O.mk_hash.str.lower().isin(nz)).to_numpy()
        _first(m, 'no order whose first fill has a non-zero price-priority ahead_lo')
        O.loc[m, 'ahead_lo'] = 0.0
        O.to_parquet(_detach(P(dst, 'orders')), index=False)
    case('ahead_lo_not_first_fill', 'priority_ahead_lo', 'ahead_lo is not the price-priority value at the first fill', ahead_first)
    case('level_side_bad', 'levels_frames_consistent', "side not in", setcol('levels', 'side', lambda LV: LV.index == LV.index[0], 'b'))
    case('tob_best_sh_wrong', 'tob_from_levels', 'best_bid_sh differs', setcol('tob', 'best_bid_sh', lambda T: T.mid == mid, lambda v: v + 1))
    if C.ID_DAY in E['days'] and 'maker_day' in E['P']['econ']:
        def ids_flip(M):     # every row of one address gets the opposite ex-ante membership flag
            c, a_ = IFACE['maker_day']['in_exante_sh'], IFACE['maker_day']['addr']
            v = bool(M[c].iloc[0]) if not isnull(M[c].iloc[0]) else False
            M.loc[M[a_] == M[a_].iloc[0], c] = not v
            return M
        case('identities_wrong', 'identities_exante', 'differs from the Oct 23 recompute', lambda dst: _edit(dst / 'econ/maker_day.parquet', ids_flip),
             days=(C.ID_DAY,) + fdays, life='none')
    else:                                                             # fail closed: never silently skipped
        res['identities_wrong'] = dict(check='identities_exante', refused=None, expected_reason='differs from the Oct 23 recompute',
                                       got=['not plantable here: no ID day or no econ maker_day'])
    sgc = IFACE['segments']

    def zs_break(M):         # one ok row shifted consistently (pnl = gross - fees and gross = rt + resid still hold): the market no longer sums to 0
        i = M.index[_first(M[f['status']] == 'ok', 'no ok maker_market row')]
        for k in ('gross_mk', 'rt', 'pnl'):
            M.loc[i, f[k]] = M.loc[i, f[k]] + 1.0
        return M
    case('zero_sum_broken', 'econ_conservation', 'zero-sum', lambda dst: _edit(dst / 'econ/maker_market.parquet', zs_break), days=fdays, life='none')

    def seg_break(S):
        i = S.index[_first((S[sgc['dim']] == IFACE['segments_full_dims'][0]) & (S[sgc['status']] == 'ok'), 'no ok first-full-dim segment row')]
        S.loc[i, sgc['pnl']] = S.loc[i, sgc['pnl']] + 1.0
        return S
    case('segments_partition_broken', 'econ_conservation', 'buckets sum', lambda dst: _edit(dst / 'econ/segments.parquet', seg_break), days=fdays, life='none')
    # ---- the raw re-parse checks
    case('rp_exec_ms', 'rp_records_fields', 'exec_ms differs from raw', setcol('records', 'exec_ms', lambda R: (R.mid == mid) & ok_(R), lambda v: v + 1000), via='reparse')
    case('rp_leg_shares', 'rp_legs', 'shares differs from raw', setcol('legs', 'shares', lambda L: (L.mid == mid) & ok_(L), lambda v: v + 0.5), via='reparse')
    case('rp_frame_dropped', 'rp_frames', 'dropped', lambda dst: _edit(P(dst, 'frames'), lambda FR: FR.drop(FR.index[len(FR) // 2])), via='reparse')
    case('rp_hi', 'rp_fill_hi', 'hi_ms != raw', lambda dst: _edit(P(dst, 'fills'), lambda F: F.assign(hi_ms=F.hi_ms - 1)), via='reparse')
    case('rp_cum', 'rp_cumulative', 'differ from the raw per-hash fills', allrows('orders', 'n_fills', ok_, lambda v: v + 1), via='reparse')
    case('rp_ident', 'rp_H1', 'raw identity conflicts not labelled', allrows('orders', 'status', lambda O: O.status == 'identity_conflict', 'ok'), via='reparse')
    case('rp_lo_not_frame', 'rp_H3', 'not the venue time of any raw frame', allrows('fills', 'lo_ms', lambda F: F.lo_ms >= 0, lambda v: v - 1), via='reparse')
    case('rp_capacity_ignored', 'rp_capacity', 'below the raw H3 capacity', no_cap, via='reparse')
    ns = (int(dt.datetime(2026, 10, 27, tzinfo=UTC).timestamp()) + 86400 + 5) * 10 ** 9      # 5 s into the first non-consumed day (a number, never a path)
    case('rp_forbidden_receipt', 'rp_hours', 'non-consumed hours', setcol('records', 'first_recv_ns', lambda R: R.mid == mid, ns), via='reparse', trace=bad_day)
    return res


def _bump_first_map(J):
    """Perturb the first class-coverage vector: flat {class: number} or the pinned {class: {orders, executions, shares}} form."""
    def walk(o):
        if isinstance(o, dict):
            if o and set(o) <= set(C.CLASSES) and all(isinstance(v, (int, float)) for v in o.values()):
                k = sorted(o)[0]; o[k] = o[k] + 0.25
                return True
            if o and set(o) <= set(C.CLASSES) and all(isinstance(v, dict) for v in o.values()):
                for k in sorted(o):
                    for den in ('orders', 'executions', 'shares'):
                        if isinstance(o[k].get(den), (int, float)) and not isinstance(o[k].get(den), bool):
                            o[k][den] = o[k][den] + 0.25
                            return True
            return any(walk(v) for v in o.values())
        return any(walk(v) for v in o) if isinstance(o, list) else False
    return walk(J)


def _flip_pnl(dst, days):
    """A sign error in one (addr, mid) maker_market row, internally consistent (pnl = gross - fees, gross = rt + resid, resid = drift + resol):
    only the independent recompute from the legs can refuse it (the case runs econ_pnl_recompute alone). e2e 4: the row is on a day the
    case's recompute covers (never an unchecked day)."""
    f = IFACE['maker_market']
    M = pd.read_parquet(dst / 'econ/maker_market.parquet')
    gsum = M[f['gross_mk']] + M[f['gross_tk']]
    dd = M[f['day']].map(lambda v: str(v).replace('-', '')[:8])
    i = M.index[_first((M[f['status']] == 'ok') & (gsum.abs() > 0.01) & dd.isin(days), f'no ok maker_market row with |gross| > 0.01 on {len(days)} recomputed days')]
    for k in ('gross_mk', 'gross_tk', 'rt', 'resid', 'drift', 'resol'):
        M.loc[i, f[k]] = -M.loc[i, f[k]]
    M.loc[i, f['pnl']] = M.loc[i, f['gross_mk']] + M.loc[i, f['gross_tk']] - M.loc[i, f['fees']]
    M.to_parquet(_detach(dst / 'econ/maker_market.parquet'), index=False)


# -------------------------------------------------------------------------------------------------------------------- smoke / CLI
def smoke(stem='20261024_04', mid=3697902):
    """Bounded real-data smoke test on ONE REST hour of 2026-10-24: independent parse of one BOOK market + a full parse of every matches line
    of the same hour as the pre-filter's positive control, measured receipt lags / tails, makers[] order, price mismatch and conflict
    variants over the hour's unique records, and the price-priority tests for the market. Needs no builder output."""
    need(stem[:8] == '20261024', 'the smoke test is bounded to one hour of 2026-10-24')
    st, hrs, tail, first, cont, uniq, seen = Counter(), [], {}, {}, defaultdict(dict), Counter(), set()
    h0 = stem_ms(stem)

    def full(r, x, m, hit):
        st['all_records'] += 1
        slug = str((x.get('market') or {}).get('categorySlug') or '')
        rc = int(r['recv_ns']) // 10 ** 6
        if slug.rsplit('-', 1)[-1].isdigit():
            tail[m] = max(tail.get(m, -1e18), (rc - int(slug.rsplit('-', 1)[-1]) * 1000 - 300000) / 1000.)
        sid = x.get('settlementId')
        try:
            sm = C.uuid7_ms(sid)
            if h0 + 600000 <= sm <= h0 + 3000000:
                first[sid] = min(first.get(sid, 1 << 62), rc - sm)
        except (ValueError, TypeError, AttributeError):
            st['bad_settlement_id'] += 1
        cont[sid][hashlib.sha1(canon(x).encode()).digest()] = _reconciles(x)
        st['needle_miss_full_parse'] += int(m == mid and not hit)
        tk, mk = x['taker'], x['makers']
        st['int_mismatch_receipts'] += int(not _reconciles(x) or int(tk['price']) != int(x['priceExecuted']))
        st['maker_fee_nonzero_legs'] += sum(int(q['fee']['amount']) != 0 for q in mk)
        st['taker_fee_' + str(tk['fee']['type'])] += 1
        ts = C.yes_space(tk['quoteType'], int(tk['outcome']['indexSet']), int(tk['price']) / C.E18)[0]
        yp = [C.yes_space(q['quoteType'], int(q['outcome']['indexSet']), int(q['price']) / C.E18) for q in mk]
        st['maker_same_side_as_taker'] += sum(s == ts for s, _ in yp)
        if sid in seen:
            return
        seen.add(sid)
        comp = [int(q['outcome']['indexSet']) != int(tk['outcome']['indexSet']) for q in mk]
        kind = 'mixed' if any(comp) and not all(comp) else ('complementary' if all(comp) else 'regular')
        qs, p = np.array([int(q['amount']) for q in mk], float), np.array([b for _, b in yp])
        pe = int(x['priceExecuted']) / C.E18
        tyes = pe if int(tk['outcome']['indexSet']) == 1 else 1 - pe
        uniq[f'records_{kind}'] += 1
        uniq[f'px_mismatch_{kind}'] += int(qs.sum() > 0 and abs(float((qs * p).sum() / qs.sum()) - tyes) > PX_TOL)
        if len(set(p)) >= 2:
            d = np.diff(p)
            bf, wf = (np.all(d >= 0), np.all(d <= 0)) if ts == 'bid' else (np.all(d <= 0), np.all(d >= 0))
            uniq['multi_price_' + ('best_first' if bf else ('worst_first' if wf else 'neither'))] += 1
    rec = rest_receipts(mid, [stem], None, st, hrs, full=full)
    R, L = unique_records('20261024', mid, rec)
    q = lambda a, p: (round(float(np.percentile(a, p)), 1) if len(a) else None)
    lg, tl = np.array(sorted(first.values()), float), np.array(list(tail.values()), float)
    conf = [v for v in cont.values() if len(v) > 1]
    out = dict(stem_day='20261024', market=mid, counters=dict(st), unique_record_stats=dict(uniq), market_receipts=sum(map(len, rec.values())),
               market_unique_records=len(R), market_legs=len(L), market_status=R.status.value_counts().to_dict(), market_repolled=int((R.n_recv > 1).sum()),
               market_max_n_recv=int(R.n_recv.max()) if len(R) else 0, market_price_mismatch=int(R.px_mismatch.fillna(False).astype(bool).sum()),
               sids_seen=len(cont), sids_with_distinct_canonical_contents=len(conf), conflict_variants=[dict(n=len(v), reconciling=sum(v.values())) for v in conf][:5],
               first_receipt_lag_ms=dict(n=len(lg), p50=q(lg, 50), p99=q(lg, 99), p999=q(lg, 99.9), max=q(lg, 100)),
               markets_in_hour=len(tail), receipt_after_window_end_s=dict(p50=q(tl, 50), p99=q(tl, 99), max=q(tl, 100)), vmhwm_mib=round(hwm_mib(), 1))
    if len(L):
        Lx = L.assign(y_ptick=np.round(L.y_price * 100).astype(np.int64), h=L.mk_hash.str.lower())
        pe = priority_evidence(Lx[Lx.status == C.STATUS_OK], Lx)
        out.update(market_price_priority_evidence_legs=len(pe['ev']), market_T1_legs=pe['t1_legs'], market_T1_hashes=pe['t1_hashes'],
                   market_T2_records=pe['t2'], market_makers_order=pe['order'], market_ahead_lo_nonzero=sum(v > 0 for v in pe['ahead'].values()),
                   market_identity_conflict_hashes=len(conflict_hashes(Lx)), market_hashes=int(Lx.h.nunique()),
                   market_repeat_fill_hashes=int((Lx.groupby('h').size() > 1).sum()))
    return out


def _write_once(path, obj):
    path = Path(path)
    need(not path.exists(), f'refusing to overwrite {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'smoke':
        print(json.dumps(smoke(), indent=1, default=str))
    elif cmd == 'run':
        E = load()
        res = run_checks(E)
        keys = [k for k in E['life'] if k[0] != C.GT_DAY][:1] + ([E['gt_reparse']] if E['gt_reparse'] else [])   # BOOK market + first selected GT window
        res['reparse'] = {f'{d}_{m}': {k: v for k, v in reparse(d, m, E).items() if k in ('checks', 'details', 'info')} for d, m in keys}
        _write_once(C.RECEIPT / f'checks_{dt.datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")}.json', res)
        bad = [n for n, v in res['checks'].items() if not v] + [f'{k}:{n}' for k, r in res['reparse'].items() for n, v in r['checks'].items() if not v]
        print(json.dumps(dict(n_checks=len(res['checks']), failed=bad), indent=1))
        sys.exit(1 if bad else 0)
    elif cmd == 'reparse':
        r = reparse(sys.argv[2], int(sys.argv[3]), load())
        print(json.dumps({k: v for k, v in r.items() if k in ('checks', 'details', 'info')}, indent=1, default=str))
    else:
        sys.exit('usage: mt_checks.py smoke | run | reparse DAY MID')
