"""Stage 2b of the chase live pilot (pmmmrust docs/receipts/20261128_tokyo_chase_live_pilot_request; USER 2026-11-28: "Check Tokyo session
question commit, do the response"). Research's decision points, 66 inputs, scores and actions on Tokyo's stage-1 recording (pmmmrust
data/tokyo-chase-stage1-20261128 0fec30b: 2026-11-28 09:50-11:01Z), computed with the frozen code, compared input by input with Tokyo's
port (replay/decisions.csv.gz). P-2026-005, no orders. No data of this research host's forward days is read: the recording is Tokyo's.
The validation stage runs the same mode on a consumed day (11-02, read by this line) against CHASE_TAKES' stored rows first.
Tokyo-data mode (declared patches, in this process only):
  - Binance: Tokyo's frames rewritten as the research collector's CSV rows (bookTicker recv_ns,E,T,u,bid,bid_qty,ask,ask_qty; depth20
    recv_ns,E,T,u,bids,asks as p@q|...; trade recv_ns,E,T,trade_id,price,qty,is_buyer_maker), one file per UTC hour of receipt, in a root
    that replaces depletion_cause.RAW; read through chase_forward's grid-filtered readers, as the forward run did;
  - Predict: Tokyo's predictOrderbook frames rewritten as pf_step1's Pass C does (top 5 levels a side, prices in cents, a missing level
    -1 / 0, rows within [ws - 300 s, we + 5 s]); a step-1 registry from Tokyo's category answers (startsAt, endsAt, the market id, the Up
    index, the decimal precision, the fee). A world is built by chase_forward.light_worlds_all's code with Tokyo's receipt as the receipt
    and shift_ms = 0 (Tokyo's receipt is the known clock itself); PF_MIN and d_dec stay research's;
  - the forward run's other patches: no other venues, no Binance-spot fair value (those inputs are not in the model), lean day data, no
    Tape offset; the day guards admit the recording's day; empty winners-legs and fire tables (inputs and diagnostics not in the model);
    payout NaN (a label, not an input).
Comparison: Tokyo's row at each (t, d)'s last frame against research's row of that tick, at or after the recording start + 60 s (Tokyo's
warm-up). The 52 float32 inputs are compared as float32, exactly. The 14 Predict inputs are float64, within 1e-9 relative. Points only one
side has are listed with Tokyo's decision and research's status. Then the prediction, MOVE and the action (the first finite MOVE >= the
threshold in a move) and the forward test's scope (expected match >= 15.022 s before the window's last 30 s).
Stages: selftest | convert <tokyo root> | validate | run | compare <tokyo root>.  Outputs (write-once): OUT/chase_tokyo_2b/...,
chase_tokyo_2b.json, chase_tokyo_2b_validate.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_tokyo_2b.py <stage> ...
"""
import collections
import datetime as dt
import glob
import gzip
import json
import os
import pickle
import resource
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_forward as CF  # noqa: E402

C, TR, CT, WT, MM, VR, I = CF.C, CF.TR, CF.CT, CF.WT, CF.MM, CF.VR, CF.I
PS, P3, FE, DCm = CF.PS, CF.P3, CF.FE, CF.DCm
DAY, VDAY, VHOURS = '20261128', '20261102', (13, 14)
OUT = WT.OUT / 'chase_tokyo_2b'
RESULT, VRESULT = H / 'chase_tokyo_2b.json', H / 'chase_tokyo_2b_validate.json'
WARM_MS, REL_TOL = 60000., 1e-9
LEGS_COLS = dict(mid='int64', side='int64', shares='float64', t_ms2='float64', how2='int64', taker='object', ambiguous='bool', ptick='float64')
FIRE_COLS = dict(day='object', rank='int64', t_set='float64', fire='float64', d='float64')


def tick(p):
    return int(round(p * 100))                                               # pf_step1.tick


def hour_key(recv_ns):
    return dt.datetime.fromtimestamp(recv_ns / 1e9, dt.timezone.utc).strftime('%Y%m%d_%H')


# ------------------------------------------------------------------------------------------------------------------ convert
def bn_lines(frames):
    """Tokyo Binance frames [recv_ns, mono_ns, text] -> {kind: {YYYYMMDD_HH: [csv line]}} in the research collector's row format."""
    out = {k: collections.defaultdict(list) for k in ('bookTicker', 'depth20', 'trade')}
    for recv, _mono, txt in frames:
        x = json.loads(txt)
        s, d = x.get('stream', ''), x.get('data') or {}
        hk = hour_key(int(recv))
        if s.endswith('@bookTicker'):
            out['bookTicker'][hk].append(f"{recv},{d['E']},{d['T']},{d['u']},{d['b']},{d['B']},{d['a']},{d['A']}")
        elif '@depth20' in s:
            b = '|'.join(f'{p}@{q}' for p, q in d['b']); a = '|'.join(f'{p}@{q}' for p, q in d['a'])
            out['depth20'][hk].append(f"{recv},{d['E']},{d['T']},{d['u']},{b},{a}")
        elif s.endswith('@trade'):
            out['trade'][hk].append(f"{recv},{d['E']},{d['T']},{d['t']},{d['p']},{d['q']},{1 if d['m'] else 0}")
    return out


def registry_of(answers):
    """Tokyo's category answers ([recv_ns, mono, 'slug\\nbody']) -> pf_step1-shaped registry rows (one per market window)."""
    reg = {}
    for recv, _mono, txt in answers:
        slug, body = txt.split('\n', 1)
        d = (json.loads(body) or {}).get('data') or {}
        for mk in d.get('markets') or []:
            mid = int(mk['id'])
            if mid in reg:
                continue
            outs = mk.get('outcomes') or []
            up = [o.get('indexSet') for o in outs if o.get('name') == 'Up']
            won = [o.get('name') for o in outs if (o.get('status') or '').upper() == 'WON']
            ws = dt.datetime.fromisoformat(d['startsAt'].replace('Z', '+00:00')).timestamp()
            we = dt.datetime.fromisoformat(d['endsAt'].replace('Z', '+00:00')).timestamp()
            reg[mid] = dict(mid=mid, slug=slug, ws=float(ws), we=float(we), first_seen_ns=int(recv), up_index=int(up[0]) if len(up) == 1 else -1,
                            fee_bps=int(mk.get('feeRateBps') or 0), decimal_precision=int(mk.get('decimalPrecision') or 2),
                            start_price=np.nan, end_price=np.nan, outcome=won[0] if won else 'Up', outcome_known=bool(won))
    R = pd.DataFrame(sorted(reg.values(), key=lambda r: r['ws']))
    C.require(len(R) and R.mid.is_unique, 'no markets in the category answers')
    return R


def book_rows(frames, R):
    """Tokyo Predict frames -> pf_step1 Pass C rows (mid, t_ms, recv_ns, order_count, b1..b5 / a1..a5 price ticks and sizes, pend)."""
    win = {int(r.mid): (r.ws * 1000 - 300_000, r.we * 1000 + 5_000) for r in R.itertuples()}
    rows = []
    for recv, _mono, txt in frames:
        x = json.loads(txt)
        if not str(x.get('topic', '')).startswith('predictOrderbook/'):
            continue
        d = x.get('data') or {}
        mid, t = d.get('marketId'), d.get('updateTimestampMs')
        w = win.get(mid)
        if w is None or t is None or not (w[0] <= t <= w[1]):
            continue
        bids = sorted(d.get('bids') or [], key=lambda z: -z[0])[:5]
        asks = sorted(d.get('asks') or [], key=lambda z: z[0])[:5]
        r = [int(mid), int(t), int(recv), int(d.get('orderCount') or 0)]
        for lv in (bids, asks):
            for i in range(5):
                r += [tick(lv[i][0]), float(lv[i][1])] if i < len(lv) else [-1, 0.0]
        sp = d.get('settlementsPending') or {}
        r += [float(sum(v for _p, v in sp.get('bids') or [])), float(sum(v for _p, v in sp.get('asks') or []))]
        rows.append(r)
    cols = ['mid', 't_ms', 'recv_ns', 'order_count'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps'] + ['pend_bid', 'pend_ask']
    return pd.DataFrame(rows, columns=cols)


def gz_frames(paths):
    for p in sorted(paths):
        with gzip.open(p, 'rt') as fh:
            for ln in fh:
                yield json.loads(ln)


def stage_convert(src):
    """Tokyo's hour1/ -> OUT/data/{raw, step1, wt_out} in research's layouts (write-once)."""
    hd = Path(src) / 'tokyo-chase-stage1-20261128' / 'hour1'
    C.require(hd.is_dir(), f'{hd} missing')
    root = OUT / 'data'
    C.require(not root.exists(), f'{root} exists (write-once)')
    lines = bn_lines(gz_frames(hd.glob('binance-*.jsonl.gz')))
    counts = {}
    for kind, by in lines.items():
        for hk, ls in by.items():
            p = root / 'raw' / kind / 'BTCUSDT' / f'{hk}.csv.gz'
            p.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(p, 'wt') as fh:
                fh.write('\n'.join(ls) + '\n')
            counts[f'{kind}/{hk}'] = len(ls)
    R = registry_of(gz_frames(hd.glob('markets-*.jsonl.gz')))
    B = book_rows(gz_frames(hd.glob('predict-*.jsonl.gz')), R)
    s1 = root / 'step1' / DAY
    (s1 / 'books').mkdir(parents=True)
    R.to_parquet(s1 / 'registry.parquet', index=False)
    for hk, g in B.groupby(B.recv_ns.map(hour_key)):
        g.to_parquet(s1 / 'books' / f'part_{hk}.parquet', index=False)
    TR.dump(s1 / 'clock_rl_d2.json', dict(day=DAY, shift_ms=0.0, d_dec=2.0, note="Tokyo's receipt is the known clock (stage 2b)"))
    wo = root / 'wt_out'
    wo.mkdir(parents=True)
    pd.DataFrame({k: pd.Series(dtype=v) for k, v in LEGS_COLS.items()}).to_parquet(wo / f'winners_legs_{DAY}.parquet', index=False)
    pd.DataFrame({k: pd.Series(dtype=v) for k, v in FIRE_COLS.items()}).to_parquet(wo / 'fire_times_orders.parquet', index=False)
    TR.dump(root / 'convert.json', dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), lines=counts, markets=int(len(R)), book_rows=int(len(B)),
                                        markets_outcome_known=int(R.outcome_known.sum())))
    print(json.dumps(dict(lines=sum(counts.values()), markets=len(R), book_rows=len(B))), flush=True)


# ------------------------------------------------------------------------------------------------------------------ the mode
def worlds_from(s1_root, ck):
    """chase_forward.light_worlds_all's code on a step-1 root, with the given clock (shift_ms, d_dec); the sealed block and day bounds as
    research's (nothing of either is in these days)."""
    def worlds(day, mids, src, off, reg):
        S1d = f'{s1_root}/{day}'
        batch = set(int(x) for x in mids)
        rg = pd.read_parquet(f'{S1d}/registry.parquet', columns=['mid', 'ws', 'outcome'])
        d0 = I._ms(f'{day[:4]}-{day[4:6]}-{day[6:]}'); d1 = d0 + 86400000
        W0 = {}
        for m, ws, o in zip(rg.mid, rg.ws, rg.outcome):
            w = float(ws) * 1000.0
            if d0 <= w < d1 and not (I.SEAL_MS[0] <= w < I.SEAL_MS[1]) and o in ('Up', 'Down') and int(m) in batch:
                W0[int(m)] = w
        cols = ['mid', 't_ms', 'recv_ns'] + [f'{s}{i}_{x}' for s in 'ba' for i in range(1, 6) for x in 'ps']
        parts = []
        for fn in sorted(glob.glob(f'{S1d}/books/part_*.parquet')):
            b = os.path.basename(fn); dd, hh = b[5:13], b[14:16]
            h0 = I._ms(f'{dd[:4]}-{dd[4:6]}-{dd[6:]}') + int(hh) * 3600000
            if not (d0 - 3600000 <= h0 < d0 + 86400000 and not (I.SEAL_MS[0] <= h0 < I.SEAL_MS[1])):
                continue
            if not W0 or not I.part_overlaps(fn, W0):
                continue
            x = pd.read_parquet(fn, columns=cols); x = x[x.mid.isin(W0)]
            rel = x.t_ms.values - x.mid.map(W0).values
            parts.append(x[(rel >= -I.PREROLL_MS) & (rel <= 300000.0)])
        B = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
        BI = B.groupby('mid', sort=False).indices
        seen = set()
        for m in sorted(W0, key=W0.get):
            if m not in BI:
                continue
            ws = W0[m]
            bb_ = B.iloc[BI[m]].sort_values('t_ms', kind='stable')
            tf = bb_.t_ms.values.astype(float)
            kf = np.maximum.accumulate(np.maximum(tf + I.PF_MIN, bb_.recv_ns.values / 1e6 - ck['shift_ms']) + ck['d_dec'])
            bb = bb_.b1_p.values.astype('int64'); ba = bb_.a1_p.values.astype('int64')
            lev = {0: (bb_[[f'b{i}_p' for i in range(1, 6)]].values.astype('int64'), bb_[[f'b{i}_s' for i in range(1, 6)]].values.astype(float)),
                   1: (bb_[[f'a{i}_p' for i in range(1, 6)]].values.astype('int64'), bb_[[f'a{i}_s' for i in range(1, 6)]].values.astype(float))}
            tg = ws + CF.GRID_MS * np.arange(CF.GRID_N)
            jk = np.searchsorted(kf, tg, side='right') - 1
            W = tuple(dict(m=m, side=s, ws=ws, tf=tf, kf=kf, jk=jk, tg=tg, opp=(ba, bb)[s], P5=lev[s][0], S5=lev[s][1], anc=np.zeros(0), ptl=[], shl=[])
                      for s in (0, 1))
            seen.add(m)
            yield m, W, ('precision_not_2' if int(reg.loc[m, 'decimal_precision']) != 2 else 'loaded')
        for m in mids:
            if int(m) not in seen:
                yield int(m), None, 'loader_excluded'
    return worlds


def apply_mode(day, raw, s1_root, wt_out, ck):
    """The declared patches for this process (see the docstring)."""
    for mod in (CT, WT, CF.TK, CF.TM):
        if day not in mod.DAYS:
            mod.DAYS = tuple(mod.DAYS) + (day,)
    DCm.RAW = Path(raw)
    DCm.book_hours = lambda d_, raw_=Path(raw): CF.book_hours_grid(d_, raw_)
    DCm.read_trades = lambda d_, raw_=Path(raw): CF.read_trades_grid(d_, raw_)
    PS.VEN, P3.VEN3 = (), ()
    CF.CV.load = CF.spot_off
    FE.day_data = CF.lean_day_data
    PS.day_features, P3.day_v3 = CF.ORIG_PS_FEATS, CF.ORIG_P3_V3
    WT.OUT = Path(wt_out)
    MM.S1 = Path(s1_root)
    VR.worlds_all = worlds_from(s1_root, ck)
    MM.KP.Tape.offset = lambda self, d_: 0.0
    CT.DV.payout = lambda r: np.nan


def score(day, od):
    """The frozen MOVE model on the mode's rows -> the frame with pred, MOVE, above, act, early."""
    R = json.loads((H / 'chase_forward_refit.json').read_text())
    kf, thr = R['inputs'], float(R['thresholds']['MOVE_top10'])
    with open(CF.FOUT / 'refit' / 'models.pkl', 'rb') as fh:
        Mo = pickle.load(fh)
    X = CF.fwd_frame((day,), kf, od)
    pred = Mo['move'].predict(X[kf])
    mv = pred - CF.cost_of(X)
    act = CT.first_per_move(np.isfinite(mv) & (mv >= thr), X.key.to_numpy())
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet', columns=['mid', 'we']).set_index('mid')
    X['pred'], X['MOVE'], X['above'] = pred, mv, (np.isfinite(mv) & (mv >= thr))
    X['act'] = False
    X.loc[act, 'act'] = True
    lead = CF.TM.FEED + CF.TM.DECIDE + CF.E.NET + CF.E.HOLD
    X['early'] = (rg.loc[X.mid, 'we'].to_numpy(float) * 1000. - CF.E.LAST_MS - X.t.to_numpy(float) - lead) >= CF.EF.EARLY_MS
    return X, kf, thr


# ------------------------------------------------------------------------------------------------------------------ compare
def same_values(a, b, f32):
    """Per row: both NaN, or equal (float32 exactly, or float64 within REL_TOL relative)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if f32:
        a, b = a.astype(np.float32).astype(float), b.astype(np.float32).astype(float)
    nan = np.isnan(a) & np.isnan(b)
    ok = np.abs(a - b) <= REL_TOL * np.maximum(1., np.abs(b)) if not f32 else (a == b)
    return nan | ok


def compare(Rx, Tk, kf, f32, start_ms):
    """Research rows Rx and Tokyo rows Tk (one per frame) -> the per-input comparison at each (t, d)'s last frame."""
    Tk = Tk.assign(t=Tk.t.astype(float), d=Tk.d.astype(float))
    Rx = Rx.assign(t=Rx.t.astype(float), d=Rx.d.astype(float))
    T = Tk.sort_values(['t', 'd', 'recv_ns'], kind='stable').groupby(['t', 'd'], as_index=False).tail(1)
    T = T[T.t >= start_ms + WARM_MS]
    Rr = Rx[Rx.t >= start_ms + WARM_MS]
    m = Rr.merge(T, on=['t', 'd'], how='outer', suffixes=('_r', '_t'), indicator=True)
    both = m[m._merge == 'both']
    per = {}
    for c in kf:
        ok = same_values(both[f'{c}_r'], both[f'{c}_t'], c in f32)
        bad = both[~ok]
        per[c] = dict(n=int(len(both)), differ=int((~ok).sum()), nan_mismatch=int((np.isnan(bad[f'{c}_r'].to_numpy(float)) != np.isnan(bad[f'{c}_t'].to_numpy(float))).sum()),
                      max_abs=float(np.nanmax(np.abs(bad[f'{c}_r'].to_numpy(float) - bad[f'{c}_t'].to_numpy(float)))) if len(bad) and np.isfinite(bad[f'{c}_r'].to_numpy(float) - bad[f'{c}_t'].to_numpy(float)).any() else None,
                      example=None if not len(bad) else dict(t=float(bad.t.iloc[0]), d=float(bad.d.iloc[0]), research=float(bad[f'{c}_r'].iloc[0]), tokyo=float(bad[f'{c}_t'].iloc[0])))
    rows_all = np.ones(len(both), bool)
    for c in kf:
        rows_all &= same_values(both[f'{c}_r'], both[f'{c}_t'], c in f32)
    pm = same_values(both.pred_r, both.pred_t, False) if 'pred_r' in both else np.zeros(len(both), bool)
    only_r, only_t = m[m._merge == 'left_only'], m[m._merge == 'right_only']
    return dict(matched=int(len(both)), research_only=int(len(only_r)), tokyo_only=int(len(only_t)),
                tokyo_only_decisions=collections.Counter(only_t['decision'].astype(str)).most_common(12) if 'decision' in only_t else [],
                rows_all_inputs_equal=int(rows_all.sum()), pred_equal=int(pm.sum()), per_input=per), both


# ------------------------------------------------------------------------------------------------------------------ stages
def stage_validate():
    """The mode on a consumed day's research data (two hours of 11-02 through a raw root of links), against CHASE_TAKES' stored rows."""
    C.require(not VRESULT.exists(), f'{VRESULT.name} exists (write-once)')
    CF.frozen()                                                              # the frozen points / labels set their own CPU limits
    root = OUT / 'validate'
    C.require(not root.exists(), f'{root} exists (write-once)')
    raw = root / 'raw'
    for kind in ('bookTicker', 'depth20', 'trade'):
        (raw / kind / 'BTCUSDT').mkdir(parents=True)
        for h in VHOURS:
            src = DCm.RAW / kind / 'BTCUSDT' / f'{VDAY}_{h:02d}.csv.gz'
            C.require(src.exists(), f'{src.name} missing')
            os.symlink(src, raw / kind / 'BTCUSDT' / src.name)
    ck = json.load(open(f'{I.IO}/step1/{VDAY}/clock_rl_d2.json'))
    wo = root / 'wt_out'; wo.mkdir()
    pd.read_parquet(WT.OUT / f'winners_legs_{VDAY}.parquet').head(0).to_parquet(wo / f'winners_legs_{VDAY}.parquet', index=False)
    pd.DataFrame({k: pd.Series(dtype=v) for k, v in FIRE_COLS.items()}).to_parquet(wo / 'fire_times_orders.parquet', index=False)
    real_out = WT.OUT
    apply_mode(VDAY, raw, Path(I.IO) / 'step1', wo, ck)
    od = root / 'rows'
    CT.points(VDAY, od)
    CT.labels(VDAY, od)
    X, kf, thr = score(VDAY, od)
    # CHASE_TAKES' stored rows of the same ticks (its inputs, before forward mode: the shared 66 compare)
    Pp = pd.read_parquet(real_out / 'chase_takes' / f'{VDAY}_points.parquet')
    Lb = pd.read_parquet(real_out / 'chase_takes' / f'{VDAY}_labels.parquet')
    S = Lb.merge(Pp, on=['day', 't', 'd'], how='inner')
    h14 = WT.OD_day_ms(VDAY) + VHOURS[1] * 3600000.
    f32 = [c for c in kf if str(X[c].dtype) == 'float32']
    Sx = S[(S.t >= h14) & (S.t < h14 + 3600000.)]
    Xr = X[(X.t >= h14) & (X.t < h14 + 3600000.)]
    m = Xr.merge(Sx[['t', 'd'] + kf], on=['t', 'd'], how='outer', suffixes=('_r', '_t'), indicator=True)
    both = m[m._merge == 'both']
    per = {c: int((~same_values(both[f'{c}_r'], both[f'{c}_t'], c in f32)).sum()) for c in kf}
    out = dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), day=VDAY, hours=list(VHOURS), compared_hour=VHOURS[1], matched=int(len(both)),
               mode_only=int((m._merge == 'left_only').sum()), stored_only=int((m._merge == 'right_only').sum()), inputs_differ=per,
               inputs_all_equal=bool(not any(per.values())), clock=dict(shift_ms=ck['shift_ms'], d_dec=ck['d_dec']))
    TR.dump(VRESULT, out)
    print(json.dumps({k: v for k, v in out.items() if k != 'inputs_differ'}), {k: v for k, v in per.items() if v}, flush=True)


def stage_run():
    C.require(not (OUT / 'rows').exists(), 'rows exist (write-once)')
    CF.frozen()                                                              # the frozen points / labels set their own CPU limits
    root = OUT / 'data'
    C.require((root / 'convert.json').exists(), 'run convert first')
    ck = json.load(open(root / 'step1' / DAY / 'clock_rl_d2.json'))
    C.require(ck['shift_ms'] == 0.0, "the Tokyo-data clock must be Tokyo's receipt (shift 0)")
    apply_mode(DAY, root / 'raw', root / 'step1', root / 'wt_out', ck)
    od = OUT / 'rows'
    CT.points(DAY, od)
    CT.labels(DAY, od)
    X, kf, thr = score(DAY, od)
    X.to_parquet(OUT / 'research_rows.parquet', index=False)
    print(DAY, len(X), 'rows', int(X.act.sum()), 'actions', int((X.act & X.early).sum()), 'in scope', flush=True)


def stage_compare(src):
    C.require(not RESULT.exists(), f'{RESULT.name} exists (write-once)')
    hd = Path(src) / 'tokyo-chase-stage1-20261128'
    X = pd.read_parquet(OUT / 'research_rows.parquet')
    R = json.loads((H / 'chase_forward_refit.json').read_text())
    kf, thr = R['inputs'], float(R['thresholds']['MOVE_top10'])
    f32 = [c for c in kf if str(X[c].dtype) == 'float32']
    Tk = pd.read_csv(hd / 'replay' / 'decisions.csv.gz', float_precision='round_trip')
    summ = json.loads((hd / 'hour1' / 'summary.json').read_text())
    ev = [json.loads(ln) for ln in open(hd / 'hour1' / 'events.jsonl') if ln.strip()]
    st = [e for e in ev if e.get('kind') == 'start']
    C.require(len(st) == 1, "expected one start event in Tokyo's events.jsonl")
    start_ms = st[0]['recv_ns'] / 1e6
    res, both = compare(X, Tk, kf, f32, start_ms)
    acts_r = X[(X.t >= start_ms + WARM_MS) & X.act]
    acts_t = Tk[Tk.decision.astype(str) == 'act']
    acts_r = acts_r.assign(t=acts_r.t.astype(float), d=acts_r.d.astype(float)); acts_t = acts_t.assign(t=acts_t.t.astype(float), d=acts_t.d.astype(float))
    res.update(asof=dt.datetime.now(dt.timezone.utc).isoformat(), day=DAY, threshold=thr, start_ms=start_ms, warm_ms=WARM_MS, float32_inputs=f32,
               research_rows=int(len(X)), research_actions=int(len(acts_r)), research_actions_in_scope=int((acts_r.early).sum()),
               tokyo_actions=int(len(acts_t)), actions_both=int(len(acts_r.merge(acts_t, on=['t', 'd']))),
               tokyo_decisions=collections.Counter(Tk.decision.astype(str)).most_common(20) if 'decision' in Tk else [],
               tokyo_summary_frames=summ.get('frames'), convert=json.loads((OUT / 'data' / 'convert.json').read_text()))
    TR.dump(RESULT, res)
    bad = {c: v['differ'] for c, v in res['per_input'].items() if v['differ']}
    print(json.dumps({k: res[k] for k in ('matched', 'research_only', 'tokyo_only', 'rows_all_inputs_equal', 'pred_equal', 'research_actions', 'tokyo_actions', 'actions_both')}),
          'inputs that differ:', bad, flush=True)


def selftest():
    fr = [[1795860000032282948, 0, json.dumps(dict(stream='btcusdt@bookTicker', data=dict(e='bookTicker', u=7, b='59562.80', B='0.118', a='59562.90', A='9.497', T=11, E=12)))],
          [1795860000042312343, 0, json.dumps(dict(stream='btcusdt@trade', data=dict(e='trade', E=41, T=40, t=99, p='59562.80', q='0.001', m=True)))],
          [1795860000052537719, 0, json.dumps(dict(stream='btcusdt@depth20@100ms', data=dict(e='depthUpdate', E=19, T=18, u=5, b=[['1.0', '2']], a=[['3.0', '4'], ['3.1', '5']])))]]
    L = bn_lines(fr)
    hk = hour_key(1795860000032282948)
    C.require(L['bookTicker'][hk] == ['1795860000032282948,12,11,7,59562.80,0.118,59562.90,9.497'], f"bookTicker {L['bookTicker'][hk]}")
    C.require(L['trade'][hk] == ['1795860000042312343,41,40,99,59562.80,0.001,1'], f"trade {L['trade'][hk]}")
    C.require(L['depth20'][hk] == ['1795860000052537719,19,18,5,1.0@2,3.0@4|3.1@5'], f"depth20 {L['depth20'][hk]}")
    C.require(hk == '20261128_10', f'hour key {hk}')
    ans = [[1, 0, 'btc-updown-5m-1795860000\n' + json.dumps(dict(data=dict(startsAt='2026-11-28T10:00:00.000Z', endsAt='2026-11-28T10:05:00.000Z',
                                                                            markets=[dict(id=4850525, feeRateBps=200, decimalPrecision=2, outcomes=[dict(name='Up', indexSet=1), dict(name='Down', indexSet=2)])])))]]
    R = registry_of(ans)
    C.require(R.mid.tolist() == [4850525] and R.ws.iloc[0] == 1795860000. and R.we.iloc[0] == 1795860300. and R.up_index.iloc[0] == 1, f'registry {R.to_dict("records")}')
    pf = [[1795860000005384482, 0, json.dumps(dict(topic='predictOrderbook/4850525', data=dict(marketId=4850525, updateTimestampMs=1795860000000, orderCount=3,
                                                                                               bids=[[0.48, 10.], [0.5, 5.]], asks=[], settlementsPending=dict(bids=[[0.5, 1.5]]))))],
          [1, 0, json.dumps(dict(topic='predictOrderbook/4850525', data=dict(marketId=4850525, updateTimestampMs=1795850000000, bids=[], asks=[])))]]
    B = book_rows(pf, R)
    C.require(len(B) == 1 and B.b1_p.iloc[0] == 50 and B.b2_p.iloc[0] == 48 and B.b3_p.iloc[0] == -1 and B.a1_p.iloc[0] == -1 and B.a1_s.iloc[0] == 0.
              and B.pend_bid.iloc[0] == 1.5, f'books {B.iloc[0].to_dict() if len(B) else None}')
    # the comparison: an identical pair agrees, a planted float32-visible change and a NaN mismatch are counted, one-sided points listed
    kf, f32 = ['a', 'b'], ['a']
    Rx = pd.DataFrame(dict(t=[1e6, 1e6 + 1, 1e6 + 2], d=[1., 1., -1.], a=np.array([0.1, 0.2, 0.3], np.float32), b=[1.5, np.nan, 2.], pred=[1., 2., 3.]))
    Tk = pd.DataFrame(dict(recv_ns=[1, 2, 3, 4], t=[1e6, 1e6, 1e6 + 1, 1e6 + 3], d=[1., 1., 1., 1.], a=[0.9, 0.1, 0.2000001, 0.4], b=[9., 1.5, 7., 1.],
                           pred=[0., 1., 2., 4.], decision=['x', 'act', 'below', 'below']))
    res, _ = compare(Rx, Tk, kf, f32, 1e6 - WARM_MS)
    C.require(res['matched'] == 2 and res['research_only'] == 1 and res['tokyo_only'] == 1, f'compare counts {res}')
    C.require(res['per_input']['a']['differ'] == 1 and res['per_input']['b']['differ'] == 1 and res['per_input']['b']['nan_mismatch'] == 1, f"per input {res['per_input']}")
    C.require(res['rows_all_inputs_equal'] == 1 and res['pred_equal'] == 2, f'rows {res}')
    C.require(same_values([0.1], [0.1 + 1e-12], False)[0] and not same_values([0.1], [0.1 + 1e-6], False)[0], 'float64 tolerance')
    global RESULT
    keep = RESULT
    RESULT = Path(__file__)
    try:
        stage_compare('/nonexistent')
        raise AssertionError('compare overwrote its result')
    except ValueError as e_:
        assert 'write-once' in str(e_), e_
    finally:
        RESULT = keep
    print('chase_tokyo_2b selftest PASS')


if __name__ == '__main__':
    cmd, a_ = sys.argv[1], sys.argv[2:]
    dict(selftest=selftest, convert=lambda: stage_convert(a_[0]), validate=stage_validate, run=stage_run, compare=lambda: stage_compare(a_[0]))[cmd]()
