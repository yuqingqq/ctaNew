"""chains_v3_selftest.py - synthetic proof of chains_v3.py (P-2026-005 MAKER-TRACE; POSTING-OPPORTUNITY-AND-CHAIN v1 section 4; 2026-11-23).
In-process, synthetic inputs only (a temporary directory mirroring the real file names, columns, receipts and parquet row groups): no real
data path is opened. Sections:
  units       second mapping, first-appearance selection, paths.md parse, reading tokens, size bounds, order grouping, receipt bindings;
              each planted mutant is shown LIVE at unit level (it really changes the output it targets).
  draft       the whole draft pipeline on a six-chain world without siblings: four outputs, receipt hashes = bytes, nothing else touched;
              pre-state links PENDING with the reason, queue links on v1 labels with the section-5 marker; TIED iff >= 2 histories (positive
              control: single-history links render ONE-HISTORY); 3 same-hash fills = ONE order; the submitted size never set; reconciliations,
              bindings and the selection hold; row groups pruned; hand-computed END value.
  refusals    overwrite (write-once), run without siblings (message), a mutant outside the selftest, an unknown mutant.
  mutants     tied_as_identified / split_same_hash / submitted_from_filled: each refused by the rendering verifier before any write.
  siblings    run mode with both sibling outputs bound: no pending link, v2 queue labels (alt_netting False -> ONE-HISTORY), pre-state TIED
              when a feature changes inside the post window and ONE-HISTORY when constant.
  falsifiers  a perturbed episode_summary, a byte-changed bound input and a swapped paths.md: recorded in a draft, refused by a run.
  schema      a sibling missing a feature column and a sibling file not matching its receipt: PENDING / v1 in a draft, refused by a run.
Usage (run_light, MT_RUN=v2): chains_v3_selftest.py"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import chains_v3 as CH  # noqa: E402

RES = []
NAN = float('nan')
G0, G1, G2, G3, G4 = CH.G0, CH.G1, CH.G2, CH.G3, CH.G4
DAY = '20261024'
WS = {11: 1_792_000_000_000, 12: 1_792_000_400_000, 13: 1_792_000_800_000}
WE = {m: w + 300_000 for m, w in WS.items()}
M_END = {11: 0.65, 12: 0.20, 13: 0.40}
V_RES = {11: 1.0, 12: 0.0, 13: 1.0}
ADDR = {1: '0x' + 'a1' * 20, 2: '0x' + 'b2' * 20, 3: '0x' + 'c3' * 20}
D_H = {'1s': 0.005, '10s': -0.01, '60s': 0.02}
TK_FEE = 0.002
SPEC = [
    dict(i=0, r=1, e='e018', mid=11, side='ask', tick=69, sh=[30.0], grade=G4, wb='gt_10s', w=78_000, hi1=90_000, pl=(True, True, True),
         nk='reprice', ns='sequential_identified', qc='unidentified', takes=[]),
    dict(i=1, r=1, e='e006', mid=12, side='ask', tick=13, sh=[6.6666], grade=G2, wb='le_1s', w=500, hi1=150_000, pl=(False, False, False),
         nk='none', ns='none', qc='front_at_acceptance', takes=[]),
    dict(i=2, r=2, e='e007', mid=12, side='bid', tick=49, sh=[13.14, 1.96, 34.9], grade=G1, wb='le_1s', w=758, hi1=40_000, pl=(False, True, False),
         nk='reprice', ns='tied', qc='conflicting', takes=[]),
    dict(i=3, r=2, e='e015', mid=13, side='bid', tick=42, sh=[50.0], grade=G2, wb='le_10s', w=2_546, hi1=70_000, pl=(False, False, True),
         nk='refill', ns='tied', qc='behind_at_first_fill', takes=[('0e015aaa-0001-7000-8000-000000000001', False, ('same', 'same', 'same'), 'reduces_all')]),
    dict(i=4, r=3, e='e014', mid=13, side='ask', tick=32, sh=[100.0, 120.0], grade=G4, wb='le_10s', w=1_162, hi1=200_000, pl=(False, False, True),
         nk='none', ns='none', qc='unidentified',
         takes=[('0e014aaa-0001-7000-8000-000000000001', True, ('opposing', 'same', 'opposing'), 'reduces_all'),
                ('0e014aaa-0002-7000-8000-000000000002', False, ('opposing', 'opposing', 'opposing'), 'reduces_some')]),
    dict(i=5, r=3, e='e017', mid=11, side='ask', tick=61, sh=[10.0], grade=G2, wb='gt_10s', w=55_823, hi1=60_000, pl=(False, False, True),
         nk='reprice', ns='tied', qc='unidentified', takes=[('0e017aaa-0001-7000-8000-000000000001', True, ('opposing', 'opposing', 'opposing'), 'reduces_all')]),
]
EXTRA_EPISODES = [(1, 'e001', 13), (2, 'e003', 11), (3, 'e002', 12)]     # later in cases.csv: never selected


def check(name, ok, info=''):
    RES.append((name, bool(ok), str(info)[:500]))


def refuses(fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return needle in str(e), f'{type(e).__name__}: {e}'
    return False, 'did not raise'


def expect(name, fn, needle, extra=True):
    ok, info = refuses(fn, needle)
    check(name, ok and extra, info)


def section(name, fn, *a):
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f'{name}: section crashed', False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-1200:]}')


class Mut:
    def __init__(self, m):
        self.m = m

    def __enter__(self):
        self.old = os.environ.get('MT_CH3_MUT')
        os.environ['MT_CH3_MUT'] = self.m

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop('MT_CH3_MUT', None)
        else:
            os.environ['MT_CH3_MUT'] = self.old


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ck(a):
    return '0x' + a[2:].upper()                                   # a mixed-case spelling (the tool lowercases)


def hx(i, pat):
    return f'0x{i + 1:02d}' + pat * 31


def sid(i, tag, k=0):
    return f'{i}{tag}{k:06d}-0000-7000-8000-{i:012d}'


# ------------------------------------------------------------------ the synthetic world ------------------------------------------------------------------
def chain_frame(s):
    """one chain's derived quantities: hashes, fills, window, legs (fills_valued rows), closes."""
    i, mid, ws = s['i'], s['mid'], WS[s['mid']]
    sgn = 1 if s['side'] == 'bid' else -1
    p = s['tick'] / 100.
    h = hx(i, 'ab')
    hi = [ws + s['hi1'] + 1500 * k for k in range(len(s['sh']))]
    fills = [dict(sid=sid(i, 'f', k), mk_idx=0, mk_hash=h, y_side=s['side'], y_ptick=s['tick'], shares=float(q), hi_ms=int(t), lo_ms=int(t - 2000),
                  x_lo_hard_ms=int(t - 2000), status='ok') for k, (q, t) in enumerate(zip(s['sh'], hi))]
    post_lo, post_hi = hi[0] - s['w'], hi[0]
    t0, t1 = post_lo, min(hi[-1] + 30_000, WE[mid])
    tot = float(sum(s['sh']))
    addr = ADDR[s['r']]

    def leg(sid_, mk_idx, role, hsh, s_, q, p_, settle, status='ok', self_trade=False, st60='ok'):
        d = dict(day=DAY, mid=mid, sid=sid_, mk_idx=mk_idx, px_rank=0, role=role, addr=ck(addr), hash=hsh, s=int(s_), q=float(q), p=float(p_),
                 self_trade=bool(self_trade), settle_ms=int(settle), V=V_RES[mid], st_res='ok')
        d['gross_res'] = s_ * q * (V_RES[mid] - p_)
        for hz, dd in D_H.items():
            d[f'st_{hz}'] = 'ok' if not (hz == '60s' and st60 != 'ok') else st60
            d[f'v_{hz}'] = (s_ * q * dd) if d[f'st_{hz}'] == 'ok' else NAN
        d['m_end'], d['st_end'] = M_END[mid], 'ok'
        d['v_end'] = s_ * q * (M_END[mid] - p_)
        d['fee_fill'] = TK_FEE * q if role == 'taker' else 0.
        d['net_res'] = d['gross_res'] - d['fee_fill']
        d['status'] = status
        return d
    legs = [leg(f['sid'], 0, 'maker', h, sgn, f['shares'], p, f['hi_ms'], st60=('beyond_end' if (i == 0 and k == 0) else 'ok'))
            for k, f in enumerate(fills)]
    pt = p - 0.01 * sgn
    T = leg(sid(i, 't'), 1, 'taker', hx(i, '12'), -sgn, 0.5 * tot, pt, hi[-1] + 5_000)
    Z = leg(sid(i, 'z'), 0, 'maker', hx(i, '34'), -sgn, 0.25 * tot, p, t1 + 10_000)
    legs += [T, Z]
    if i == 2:
        legs.append(leg(sid(i, 'x'), 0, 'maker', hx(i, '56'), 1, 5.0, p, hi[0] + 100, status='src_status'))
        legs.append(leg(sid(i, 's'), 2, 'taker', hx(i, '78'), 1, 3.0, p, hi[0] + 200, self_trade=True))
    # FIFO closes: the order's fills (entries) closed by T, then by Z
    closes, att = [], {}
    rem = [f['shares'] for f in fills]
    for C_, cap in ((T, T['q']), (Z, Z['q'])):
        for k, f in enumerate(fills):
            if cap <= 0:
                break
            x = min(rem[k], cap)
            if x <= 0:
                continue
            rem[k] -= x
            cap -= x
            sell, buy = (p, C_['p']) if sgn < 0 else (C_['p'], p)
            fee_c = C_['fee_fill'] * x / C_['q']
            closes.append(dict(day=DAY, mid=mid, addr=ck(addr), rank=s['r'], entry_sid=f['sid'], entry_mk_idx=0, entry_hash=h, entry_role='maker',
                               entry_side=s['side'], entry_settle_ms=f['hi_ms'], entry_p=p, close_sid=C_['sid'], close_mk_idx=C_['mk_idx'],
                               close_hash=C_['hash'], close_role=C_['role'], close_settle_ms=C_['settle_ms'], close_p=C_['p'],
                               role_pair=f"maker->{C_['role']}", sh=float(x), pnl=float(x * (sell - buy)), fee_fill_entry=0., fee_fill_close=fee_c,
                               fee_res_entry=0., fee_res_close=fee_c, net=float(x * (sell - buy) - fee_c), net_feeres=float(x * (sell - buy) - fee_c),
                               hold_ms=int(C_['settle_ms'] - f['hi_ms']), dust=False, res_ok=True))
            att[(f['sid'], 0, 'maker')] = att.get((f['sid'], 0, 'maker'), 0.) + x
            att[(C_['sid'], C_['mk_idx'], C_['role'])] = att.get((C_['sid'], C_['mk_idx'], C_['role']), 0.) + x
    for d in legs:
        lot = d['status'] == 'ok' and not d['self_trade']
        d['mq'] = att.get((d['sid'], d['mk_idx'], d['role']), 0.) if lot else 0.
        d['rq'] = (d['q'] - d['mq']) if lot else 0.
    return dict(h=h, fills=fills, post_lo=post_lo, post_hi=post_hi, t0=t0, t1=t1, legs=legs, closes=closes, tot=tot, addr=addr, sgn=sgn, p=p,
                eid=f"{DAY}_{mid}_r{s['r']}_{s['e']}", ck_order=f"order|{DAY}|{mid}|{addr}|{h}")


def write_pq(df, p, row_group_size=None, dict_cols=()):
    t = pa.Table.from_pandas(df, preserve_index=False)
    if dict_cols:
        t = pa.table({c: (t.column(c).dictionary_encode() if c in dict_cols else t.column(c)) for c in t.column_names})
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, p, row_group_size=row_group_size)


def write_csv(df, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def build_world(root, siblings=False, es_perturb=False, paths_swap=False, po_drop=None, qf2_bad_sha=False):
    rr, dr = Path(root) / 'v2', Path(root) / 'data'
    F = [(s, chain_frame(s)) for s in SPEC]
    # ---- cases.csv: rank 1 e018 then e006 (first appearance, not sorted), interleaved ranks, extra episodes later, take rows at the end ----
    case_cols = CH.NEED['cases']
    rows = []

    def order_case(s, f):
        return dict(kind='order', day=DAY, mid=s['mid'], addr=f['addr'], key=f['h'], rank=s['r'], case_key=f['ck_order'], episode_id=f['eid'],
                    y_side=s['side'], tick=float(s['tick']), n_fills=float(len(f['fills'])), cum=f['tot'], size_status='tied',
                    union='EX|PL' if any(s['pl']) else 'EX', ub=NAN, P_cmd="('EX', %s)" % f['tot'], P_drop="('PL', %s)" % (f['tot'] * 2),
                    P_lot="('EX', %s)" % f['tot'], P_clip="('EX', %s)" % f['tot'], prior_sensitive=True, front_class='multiple',
                    ahead_lo_fifo=0.0, ahead_hi_fifo=20.0, ahead_at_acceptance_lo=0.0, ahead_at_acceptance_hi=40.0, status='ok')
    order_idx = [0, 1, 2, 3, 4, 5]
    for j in order_idx[:3]:
        rows.append(order_case(*F[j]))
    for (r, e, m) in EXTRA_EPISODES[:1]:
        rows.append(dict(kind='order', day=DAY, mid=m, addr=ADDR[r], key=hx(90 + r, 'aa'), rank=r, case_key=f'order|{DAY}|{m}|{ADDR[r]}|x',
                         episode_id=f'{DAY}_{m}_r{r}_{e}', status='ok'))
    for j in order_idx[3:]:
        rows.append(order_case(*F[j]))
    for (r, e, m) in EXTRA_EPISODES[1:]:
        rows.append(dict(kind='order', day=DAY, mid=m, addr=ADDR[r], key=hx(90 + r, 'aa'), rank=r, case_key=f'order|{DAY}|{m}|{ADDR[r]}|y',
                         episode_id=f'{DAY}_{m}_r{r}_{e}', status='ok'))
    for s, f in F:
        for (tsid, is_case, dirs, o4) in s['takes']:
            if is_case:
                rows.append(dict(kind='take', day=DAY, mid=s['mid'], addr=f['addr'], key=tsid, rank=s['r'], case_key=f"take|{DAY}|{s['mid']}|{f['addr']}|{tsid}",
                                 episode_id=f['eid'], status='ok', s=-f['sgn'], q=7.0, p_avg=f['p'], settle_ms=f['fills'][-1]['hi_ms'] + 2000,
                                 pos_before=-50., pos_after=-43., cls='reduce_partial', cls_simultaneous='reduce_partial', cls_certain='reduce_partial',
                                 timing_sensitive=False, burst_cert=3, burst_pos=4))
    CS = pd.DataFrame(rows, columns=list(case_cols))
    write_csv(CS, rr / 'order_estimator/cases.csv')
    # ---- episode_summary.csv (computed from the legs: order_estimator.episode_econ's rule) ----
    es_rows = []
    for s, f in F:
        L = pd.DataFrame(f['legs'])
        lot = (L.status == 'ok') & ~L.self_trade
        win = (L.settle_ms >= f['t0']) & (L.settle_ms <= f['t1'])
        E = L[lot & win]
        ve = E.v_end.to_numpy(float)
        es_rows.append(dict(episode_id=f['eid'], day=DAY, mid=s['mid'], addr=f['addr'], rank=s['r'], t0=f['t0'], t1=f['t1'], n_cases=1 + len(s['takes']),
                            case_keys=f"order:{f['h'][:12]}", ep_n_legs=len(E), ep_n_maker=int((E.role == 'maker').sum()),
                            ep_n_taker=int((E.role == 'taker').sum()), ep_sh=float(E.q.sum()) + (1.0 if (es_perturb and s['i'] == 0) else 0.),
                            ep_v_end=float(np.nansum(ve)), ep_n_v_end_na=int(np.isnan(ve).sum()), ep_fee=float(E.fee_fill.sum()),
                            ep_net_end=float(np.nansum(ve) - E.fee_fill.sum()), ep_net_res=float(np.nansum(E.net_res)), ep_mq=float(E.mq.sum()),
                            ep_rq=float(E.rq.sum()), ep_pos_in=0., ep_pos_out=1., ep_outcome='flat',
                            ep_n_excluded=int((~lot & win & (L.status != 'ok')).sum()), mg_status='ok', mg_close_net=1.5 + s['i'], mg_resid=-0.5,
                            mg_total=1.0 + s['i'], mg_close_net_c_sh=0.1))
    write_csv(pd.DataFrame(es_rows, columns=list(CH.NEED['episode_summary'])), rr / 'order_estimator/episode_summary.csv')
    # ---- order_ledger.csv / episodes.csv / episodes_post_regraded.csv ----
    led, eps, epr = [], [], []
    for s, f in F:
        pl_union = any(s['pl'])
        base = {c: NAN for c in CH.NEED['order_ledger']}
        led.append(dict(base, case_key=f['ck_order'], row_kind='size_end', hypothesis='EX', status='tied', feasible_union=True, feasible_S_FIFO=True,
                        feasible_S_FIFO_LO=True, feasible_S_NOPRIO=True, size_lo=f['tot'], size_hi=f['tot'], window_lo_ms=f['fills'][-1]['hi_ms'],
                        window_hi_ms=f['fills'][-1]['hi_ms'], remaining_lo=0., remaining_hi=0.))
        led.append(dict(base, case_key=f['ck_order'], row_kind='size_end', hypothesis='PL', status='tied', feasible_union=pl_union,
                        feasible_S_FIFO=s['pl'][0], feasible_S_FIFO_LO=s['pl'][1], feasible_S_NOPRIO=s['pl'][2], size_lo=f['tot'] + 0.01,
                        size_hi=NAN, window_lo_ms=f['fills'][-1]['hi_ms'] - 2000, window_hi_ms=f['fills'][-1]['hi_ms'] + 50, remaining_lo=0.01,
                        remaining_hi=NAN))
        led.append(dict(base, case_key=f['ck_order'], row_kind='size_end', hypothesis='EC', status='tied', feasible_union=False, feasible_S_FIFO=False,
                        feasible_S_FIFO_LO=False, feasible_S_NOPRIO=False, size_lo=f['tot'] + 0.01, size_hi=NAN, window_lo_ms=-1, window_hi_ms=-1,
                        remaining_lo=0.01, remaining_hi=NAN))
        if s['nk'] != 'none':
            hyp = 'replace|replenish|unrelated' if s['ns'] == 'sequential_identified' else 'overlap|replace|replenish|unrelated'
            led.append(dict(base, case_key=f['ck_order'], row_kind='link', hypothesis=hyp, role='child', other_hash=hx(s['i'], 'cd'), kind=s['nk'],
                            d_ticks=-1, status=s['ns'], gap_lo_ms=100., gap_hi_ms=4000., other_first_hi_ms=f['fills'][-1]['hi_ms'] + 4000, other_cum=10.))
        if s['i'] == 0:
            led.append(dict(base, case_key=f['ck_order'], row_kind='link', hypothesis='overlap|replace|replenish|unrelated', role='parent',
                            other_hash=hx(s['i'], 'ef'), kind='reprice', d_ticks=3, status='tied', gap_lo_ms=-7000., gap_hi_ms=2000.,
                            other_first_hi_ms=f['fills'][0]['hi_ms'] - 300, other_cum=19.2))
        for mech, cond in (('inventory_driven_pull', 'compatible'), ('pull_after_adverse_move', 'contradicted')):
            eps.append(dict(episode_id=f['eid'], case_key=f['ck_order'], link='END', mechanism=mech, identification_status='unidentified',
                            conditional_status=cond, evidence_grade='E_unresolved', timing_sensitive=False))
        for (tsid, is_case, dirs, o4) in s['takes']:
            if is_case:
                eps.append(dict(episode_id=f['eid'], case_key=f"take|{DAY}|{s['mid']}|{f['addr']}|{tsid}", link='TAKE', mechanism='after_fill_burst',
                                identification_status='compatible', conditional_status='', evidence_grade='A_observed', timing_sensitive=False))
        epr.append(dict(episode_id=f['eid'], case_key=f['ck_order'], link='POST', mechanism='queue_priority_before_demand',
                        identification_status='contradicted', identification_status_corrected='timing_sensitive', evidence_grade='B_constrained',
                        evidence_grade_corrected='C_scenario'))
    write_csv(pd.DataFrame(led, columns=list(CH.NEED['order_ledger'])), rr / 'order_estimator/order_ledger.csv')
    write_csv(pd.DataFrame(eps, columns=list(CH.NEED['episodes'])), rr / 'order_estimator/episodes.csv')
    cv = rr / 'chains/correction_v1'
    write_csv(pd.DataFrame(epr, columns=list(CH.NEED['episodes_post_regraded'])), cv / 'episodes_post_regraded.csv')
    # ---- post_regrade_orders.csv / post_fill orders.csv ----
    T_rows, P_rows = [], []
    for s, f in F:
        ws = WS[s['mid']]
        T_rows.append(dict(day=DAY, mid=s['mid'], rank=s['r'], addr=ck(f['addr']), mk_hash=f['h'], y_side=s['side'], tick=s['tick'],
                           n_fills=len(f['fills']), cum_shares=f['tot'], first_hi_ms=f['fills'][0]['hi_ms'], post_lo_union_ms=f['post_lo'],
                           post_lo_S_FIFO_ms=f['post_lo'], post_lo_S_FIFO_LO_ms=f['post_lo'], post_lo_S_NOPRIO_ms=f['post_lo'], post_hi_ms=f['post_hi'],
                           window_width_ms=s['w'], width_bin=s['wb'], life_post_class='assumption_sensitive', status_packet='ok', g1_flags_matched='',
                           grade=s['grade'], window_basis=CH.WINDOW_BASIS[s['grade']]))
        L = pd.DataFrame(f['legs'])
        own = L[(L.role == 'maker') & (L.hash == f['h'])]
        ok = own.status == 'ok'
        coh = ok & ~own.self_trade & own.st_res.isin(['ok', 'tie']) & (own.st_1s == 'ok') & (own.st_10s == 'ok') & (own.st_60s == 'ok')
        nk = s['nk']
        P_rows.append(dict(day=DAY, mid=s['mid'], rank=s['r'], addr=ck(f['addr']), mk_hash=f['h'], y_side=s['side'], tick=s['tick'],
                           n_fills=len(f['fills']), cum_shares=f['tot'], first_hi_ms=f['fills'][0]['hi_ms'], last_hi_ms=f['fills'][-1]['hi_ms'],
                           ws_ms=ws, we_ms=WE[s['mid']], post_lo_union_ms=f['post_lo'], post_hi_ms=f['post_hi'], queue_class=s['qc'],
                           ahead_acc_min=0.0, ahead_acc_max=40.0, ahead_lo_fifo=0.0, ahead_hi_fifo=20.0, lo_clock_queue_differs=False,
                           touch_first_ms=-1, touch_at_start='mixed', nomask_class='unidentified', nomask_status='multiple',
                           n_later_fills=len(f['fills']) - 1, n_tied_first=0, n_front_after_partial=0, size_status='tied',
                           union_histories='EX|PL' if any(s['pl']) else 'EX', next_kind=nk, next_link_status=s['ns'], n_children=0 if nk == 'none' else 1,
                           next_hash='' if nk == 'none' else hx(s['i'], 'cd'), next_gap_lo_ms=NAN if nk == 'none' else 100.,
                           next_gap_hi_ms=NAN if nk == 'none' else 4000., next_d_ticks=0 if nk == 'none' else -1,
                           mq_sh=float(own.mq[ok].sum()), rq_sh=float(own.rq[ok].sum()), n_own_rows=len(own), pnl_rows=int(coh.sum()),
                           pnl_shares=float(own.q[coh].sum()), net_res=float(own.net_res[coh].sum()), fee_fill=float(own.fee_fill[coh].sum()),
                           pnl_excluded_rows=int((~coh).sum()), v_1s=float(own.v_1s[coh].sum()), v_10s=float(own.v_10s[coh].sum()),
                           v_60s=float(own.v_60s[coh].sum()), status='ok'))
    T = pd.DataFrame(T_rows)
    write_csv(T, cv / 'post_regrade_orders.csv')
    P = pd.DataFrame(P_rows, columns=list(CH.NEED['post_orders']))
    write_csv(P, rr / 'chains/post_fill/orders.csv')
    # ---- pulls.csv ----
    pu = []
    for s, f in F:
        for v, ok_ in (('S_FIFO', s['pl'][0]), ('S_NOPRIO', s['pl'][2])):
            if ok_:
                pu.append(dict(day=DAY, mid=s['mid'], scope=f"rank{s['r']}", addr=ck(f['addr']), mk_hash=f['h'], view=v,
                               removal_set='cancelled_or_expired|exhausted', removal_lo_ms=f['fills'][-1]['hi_ms'] - 2000,
                               removal_hi_ms=f['fills'][-1]['hi_ms'] + 50, removal_method='book_gone', removal_class='assumption_sensitive',
                               x_size_ub=NAN, end_histories='EX|PL', identification='tied', pull_identified=False, window_status='ok', width_ms=2050,
                               d_sensitive=False, most_supported='ok', support_grade='anchors', status='ok'))
    PU = pd.DataFrame(pu, columns=list(CH.NEED['pulls']))
    write_csv(PU, rr / 'chains/pull/pulls.csv')
    # ---- census_graded.csv (counts from the keyed rows above) ----
    cen = []
    for r in (1, 2, 3):
        Tm = T[T['rank'] == r]
        for g in (G0, G1, G2, G3, G4):
            cen.append(dict(source='chains/take/actions.csv', action_type='post_new_level', scope=f'rank{r}', day=DAY, subset=g,
                            n=int((Tm.grade == g).sum()), n_action_total=len(Tm), evidence_grade=f'post_window:{g}'))
            cen.append(dict(source='chains/take/actions.csv', action_type='post_same_level', scope=f'rank{r}', day=DAY, subset=g, n=0,
                            n_action_total=0, evidence_grade=f'post_window:{g}'))
        for v in ('S_FIFO', 'S_NOPRIO'):
            cen.append(dict(source='chains/take/actions.csv', action_type=f'pull_possible:{v}:cancel_possible', scope=f'rank{r}', day=DAY, subset='all',
                            n=int(((PU.scope == f'rank{r}') & (PU.view == v)).sum()), n_action_total=int(((PU.scope == f'rank{r}') & (PU.view == v)).sum()),
                            evidence_grade='cancel_compatible_ending'))
        Pm = P[P['rank'] == r]
        for kind in ('reprice', 'refill', 'none'):
            Pk = Pm[Pm.next_kind == kind]
            for st_, n_ in Pk.next_link_status.value_counts().items():
                cen.append(dict(source='chains/post_fill/orders.csv', action_type=f'successor_{kind}', scope=f'rank{r}', day=DAY,
                                subset='none' if kind == 'none' else CH.LINK_SUBSET[st_], n=int(n_), n_action_total=len(Pk),
                                evidence_grade='unresolved_alternatives' if kind == 'none' else 'scenario_dependent_link'))
    write_csv(pd.DataFrame(cen, columns=list(CH.NEED['census_graded'])), cv / 'census_graded.csv')
    # ---- paths.md ----
    sel = [(s['r'], f['eid']) for s, f in F]
    if paths_swap:
        sel[0], sel[1] = sel[1], sel[0]
    md = ['# Logic-chain correction v1 - episode paths (contract section 4)', '']
    for k, (r, e) in enumerate(sel, 1):
        md += [f'## Path {k} - rank {r}: episode {e}', '', '- **POST** ...', '']
    (cv / 'paths.md').write_text('\n'.join(md) + '\n')
    # ---- take_logic_v2 events.parquet (one record outside every window) ----
    ev = []
    for s, f in F:
        for j, (tsid, is_case, dirs, o4) in enumerate(s['takes']):
            st_ = f['fills'][-1]['hi_ms'] + 2000 + 500 * j
            ev.append(dict(day=DAY, mid=s['mid'], rank=s['r'], addr=ck(f['addr']), sid=tsid, s=-f['sgn'], dq=-f['sgn'] * 7.0, p_avg=f['p'],
                           x_lo_hard_ms=st_ - 2000, hi_ms=st_, settle_ms=st_, status='ok', take_class='reduce_partial', cls_rec='reduce_partial',
                           cls_sim='reduce_partial', cls_cert_0='reduce_partial', pop_status='assigned', t_anchor_ms=st_ - 1500, o4_status=o4,
                           q_D0=-50.0 * f['sgn'], dir_D0=dirs[0], q_D1000=-48.0, dir_D1000=dirs[1], q_pend=-49.0, dir_pend=dirs[2], tk_hash=hx(s['i'], '9a')))
    s3, f3 = F[3]
    ev.append(dict(ev[0], sid='0e015aaa-0009-7000-8000-000000000009', settle_ms=f3['t1'] + 50_000, hi_ms=f3['t1'] + 50_000, t_anchor_ms=f3['t1'] + 49_000))
    write_pq(pd.DataFrame(ev), rr / 'chains/take_logic_v2/events.parquet')
    # ---- queue_front v1 levels.csv / fronts.csv ----
    lv, fr = [], []
    for s, f in F:
        f1, fl = f['fills'][0], f['fills'][-1]
        lbase = dict(day=DAY, mid=s['mid'], side=s['side'], tick=s['tick'], unit='E1', size_created=50.0, holder='', holder_group='', front_hash='',
                     n_behind=0, n_unattributed=len(f['fills']), front_timing='', mechanism='deep')
        lv.append(dict(lbase, level_id=100 + s['i'], t_open_ms=f1['hi_ms'] - 5_000, t_create_ms=f1['hi_ms'] - 4_950, status='unknown',
                       reason='untimed_record_only', end_type='zero', end_ms=fl['hi_ms'] + 100, n_members=len(f['fills'])))
        lv.append(dict(lbase, level_id=200 + s['i'], t_open_ms=fl['hi_ms'] + 5_000, t_create_ms=fl['hi_ms'] + 5_050, status='no_fill',
                       reason='returned_to_zero', end_type='zero', end_ms=fl['hi_ms'] + 9_000, n_members=0))
        if s['i'] == 3:                                              # a second compatible life (a gap boundary: open to we)
            lv.append(dict(lbase, level_id=300 + s['i'], t_open_ms=f1['hi_ms'] - 9_000, t_create_ms=f1['hi_ms'] - 8_950, status='gap',
                           reason='gap_before_fill', end_type='gap', end_ms=-1, n_members=-1))
        if s['i'] == 2:
            for k, fl_ in enumerate(f['fills'], 1):
                fr.append(dict(day=DAY, mid=s['mid'], level_id=100 + s['i'], role='front' if k == 1 else 'behind', k=k, sid=fl_['sid'], mk_idx=0,
                               mk_hash=f['h'], addr=ck(f['addr']), q=fl_['shares'], hi_ms=fl_['hi_ms'], lo_ms=fl_['lo_ms']))
            fr.append(dict(day=DAY, mid=s['mid'], level_id=100 + s['i'], role='behind', k=4, sid=sid(9, 'o'), mk_idx=0, mk_hash=hx(9, 'ee'),
                           addr=ck(ADDR[1]), q=4.0, hi_ms=fl['hi_ms'] + 10, lo_ms=fl['hi_ms'] - 1990))
    LV = pd.DataFrame(lv, columns=list(CH.NEED['qf1_levels']))
    write_csv(LV, rr / 'chains/queue_front/levels.csv')
    write_csv(pd.DataFrame(fr, columns=list(CH.NEED['qf1_fronts'])), rr / 'chains/queue_front/fronts.csv')
    # ---- closes.parquet (two row groups: the chain day and another day) / closes_markets.parquet / matched_gap_r1 ----
    CLa = pd.DataFrame([c for _, f in F for c in f['closes']])
    other = CLa.iloc[[0]].copy()
    other['day'], other['mid'] = '20261025', 98
    cdir = rr / 'closes'
    cdir.mkdir(parents=True, exist_ok=True)
    t1_, t2_ = pa.Table.from_pandas(CLa, preserve_index=False), pa.Table.from_pandas(other, preserve_index=False)
    with pq.ParquetWriter(cdir / 'closes.parquet', t1_.schema) as W:
        W.write_table(t1_)
        W.write_table(t2_.cast(t1_.schema))
    cm = []
    for s, f in F:
        X = CLa[(CLa['mid'] == s['mid']) & (CLa['addr'] == ck(f['addr']))]
        cm.append(dict(day=DAY, mid=s['mid'], addr=ck(f['addr']), n_rows_ok=len(f['legs']), n_rows_excl=0, n_closes=len(X), n_dust=0,
                       close_sh=float(X.sh.sum()), close_pnl=float(X.pnl.sum()), close_fee_fill=float(X.fee_fill_close.sum()), resid_sh=1.0,
                       resid_end=0.1, resid_res=0.2, net_pos=1.0, max_abs_pos=10.0, res_ok=True, end_ok=True, status='ok'))
    write_pq(pd.DataFrame(cm), cdir / 'closes_markets.parquet')
    mg = []
    for s, f in F:
        for v in ('carry', 'zero'):
            mg.append(dict(day=DAY, m=s['mid'], party=f"rank{s['r']}", view=v, primary=v == 'carry', status='ok', n_rows_in=5, n_excl_in=0,
                           n_closes=2, close_sh=10.0, close_net=1.5 + s['i'], carry=0.0, resid=-0.5, fee_unclosed=0.0, self_gross=0.0,
                           total=1.0 + s['i'], close_net_c_sh=0.1, exposure=-0.5, reading='mixed_unsupported'))
    write_csv(pd.DataFrame(mg, columns=list(CH.NEED['matched_gap'])), rr / 'matched_gap_r1/per_market.csv')
    # ---- data root: fills_valued (row groups by market incl. a non-chain market; dictionary columns) and lifecycle fills ----
    FVr = pd.DataFrame([d for _, f in F for d in f['legs']]).sort_values(['mid', 'settle_ms'], kind='stable').reset_index(drop=True)
    oth = FVr.iloc[[0]].copy()
    oth['mid'], oth['sid'] = 99, 'zz'

    def _t(df):
        t = pa.Table.from_pandas(df, preserve_index=False)
        return pa.table({c: (t.column(c).dictionary_encode() if c in ('day', 'role', 'status', 'st_10s') else t.column(c)) for c in t.column_names})
    ta, tb = _t(FVr), _t(oth)
    (dr / 'econ').mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(dr / 'econ/fills_valued.parquet', ta.schema) as W:      # row group 2 = a non-chain market only (pruned)
        W.write_table(ta)
        W.write_table(tb.cast(ta.schema))
    lp = {}
    for s, f in F:
        lp.setdefault(s['mid'], []).extend(f['fills'])
    for m, fl in lp.items():
        fl = fl + [dict(fl[0], sid='other-fill', mk_hash=hx(8, 'dd'))]
        write_pq(pd.DataFrame(fl).assign(mid=m)[list(CH.LIFE_READ)], dr / f'life/{DAY}_{m}_fills.parquet')
    # ---- receipts binding every input ----
    j = lambda p, o: (p.parent.mkdir(parents=True, exist_ok=True), p.write_text(json.dumps(o, indent=1)))   # noqa: E731
    oe = rr / 'order_estimator'
    j(cv / 'receipt_post.json', dict(inputs={k: dict(path=f'x/{k}', sha256=sha(oe / f'{k}.csv')) for k in ('cases', 'episode_summary', 'order_ledger', 'episodes')},
                                     outputs={nm: dict(sha256=sha(cv / nm)) for nm in ('paths.md', 'post_regrade_orders.csv', 'episodes_post_regraded.csv',
                                                                                       'census_graded.csv')}))
    j(rr / 'chains/post_fill/receipt.json', dict(outputs={'orders.csv': dict(sha256=sha(rr / 'chains/post_fill/orders.csv'))}))
    j(rr / 'chains/pull/receipt.json', dict(outputs={'pulls.csv': dict(sha256=sha(rr / 'chains/pull/pulls.csv'))}))
    j(rr / 'chains/take_logic_v2/receipt.json', dict(
        outputs={'events.parquet': dict(sha256=sha(rr / 'chains/take_logic_v2/events.parquet'))},
        inputs={'data/x/econ/fills_valued.parquet': sha(dr / 'econ/fills_valued.parquet'),
                **{f'data/x/life/{DAY}_{m}_fills.parquet': sha(dr / f'life/{DAY}_{m}_fills.parquet') for m in lp}}))
    j(rr / 'chains/queue_front/receipt.json', dict(outputs={nm: dict(sha256=sha(rr / 'chains/queue_front' / nm)) for nm in ('levels.csv', 'fronts.csv')}))
    j(cdir / 'closes_receipt.json', dict(outputs={nm: dict(sha256=sha(cdir / nm)) for nm in ('closes.parquet', 'closes_markets.parquet')}))
    j(rr / 'matched_gap_r1/receipt.json', dict(outputs={'per_market.csv': dict(sha256=sha(rr / 'matched_gap_r1/per_market.csv'))}))
    if siblings:                                                  # the siblings' declared schemas (posting_opportunity.rows_frame; queue_front_v2 OUTPUTS)
        po = rr / 'chains/posting_opportunity'
        prow = []
        f0 = F[0][1]
        for m in WS:
            for side in ('bid', 'ask'):
                for k in range(300):
                    d = dict(day=DAY, mid=m, side=side, k=k, t_ms=WS[m] + 1000 * k)
                    for v in ('D0', 'D1000'):
                        d.update({f'status_{v}': 'ok', f'spread_ticks_{v}': 1, f'spread_bin_{v}': '2' if (m == 11 and side == 'ask' and k >= 50) else '1',
                                  f'depth_touch_own_bin_{v}': '(0,50]', f'depth_touch_opp_bin_{v}': '(50,200]', f'depth_2_own_bin_{v}': '0',
                                  f'btc_5s_aligned_{v}': 'flat', f'fair_gap_bin_{v}': '(0,1]', f'flow_5s_bin_{v}': '0', f'time_left_bin_{v}': '(60,240]'})
                    key = f"{m}|ask|69|600" if (m == 11 and side == 'ask' and k == 60) else ''
                    d.update(E_any=int(bool(key)), N_any=int(bool(key)), keys_any=key, keys_improve='', keys_re_establish='', keys_deep=key,
                             keys_at_empty_touch='', n_straddle=0, n_gap=0, n_unknown_cell=0)
                    if po_drop:
                        d.pop(po_drop)
                    prow.append(d)
        assert sum(1 for r in prow if r['keys_any']) == 1 and CH.sec_of(f0['post_lo'], WS[11]) <= 60 <= CH.sec_of(f0['post_hi'], WS[11])
        write_pq(pd.DataFrame(prow), po / 'rows.parquet')
        j(po / 'receipt.json', dict(outputs={'rows.parquet': dict(sha256=sha(po / 'rows.parquet'))}))
        q2 = rr / 'chains/queue_front_v2'
        L2 = LV.copy()
        L2['alt_netting_possible'] = [('False' if int(x) == 101 else 'True') for x in L2['level_id']]
        L2['creation_context'] = 'deep'
        write_csv(L2.drop(columns=['mechanism']), q2 / 'levels.csv')
        s2, f2 = F[2]
        write_csv(pd.DataFrame([dict(day=DAY, mid=s2['mid'], level_id=102, status='front_candidate', candidate_hash=f2['h'],
                                     earliest_set_hashes=f"{f2['h']}|{hx(7, 'ee')}", timing_basis='book_candidate', alt_netting_possible=True,
                                     alt_first_other_address=False)]), q2 / 'candidates.csv')
        write_csv(pd.DataFrame([dict(day=DAY, mid=s2['mid'], level_id=102, role='earliest_set' if k == 0 else 'later_same_hash', k=k, sid=fl_['sid'],
                                     mk_idx=0, mk_hash=f2['h']) for k, fl_ in enumerate(f2['fills'])]), q2 / 'fills.csv')
        write_csv(pd.DataFrame([dict(day=DAY, mid=s['mid'], mk_hash=f['h'], occupancy='cell_empty_at_window_lo', window_lo_ms=f['post_lo'],
                                     post_grade=s['grade'], size_at_window_lo=0.0, lookup_status='ok') for s, f in F]), q2 / 'occupancy.csv')
        j(q2 / 'receipt.json', dict(outputs={'levels.csv': dict(sha256=sha(q2 / 'levels.csv')),
                                             'candidates.csv': dict(sha256=('0' * 64) if qf2_bad_sha else sha(q2 / 'candidates.csv')),
                                             'fills.csv': dict(sha256=sha(q2 / 'fills.csv')), 'occupancy.csv': dict(sha256=sha(q2 / 'occupancy.csv'))}))
    return rr, dr, F


def run_q(rr, dr, out, mode='draft'):
    return CH.run(mode=mode, rroot=rr, droot=dr, out=out, require_freeze=False, selftest=True, log=lambda *a, **k: None)


def snapshot(root):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in Path(root).rglob('*') if p.is_file()}


def links_of(out):
    return pd.read_csv(Path(out) / 'chain_links.csv', dtype={'day': str})


# ------------------------------------------------------------------ sections ------------------------------------------------------------------
def s_units():
    ws = 1_000_000
    check('second map: ws+1 and ws+1000 lie in k=0, ws+1001 in k=1, ws in k=-1',
          (CH.sec_of(ws + 1, ws), CH.sec_of(ws + 1000, ws), CH.sec_of(ws + 1001, ws), CH.sec_of(ws, ws)) == (0, 0, 1, -1))
    CS = pd.DataFrame(dict(episode_id=['b', 'b', 'a', 'x', 'c', 'y', 'z', 'w'], rank=[1, 1, 1, 2, 1, 2, 3, 3]))
    check('selection: first two per rank in row order of first appearance (b before a; c never)', CH.select_chains(CS) == [(1, 'b'), (1, 'a'), (2, 'x'), (2, 'y'), (3, 'z'), (3, 'w')],
          CH.select_chains(CS))
    check('paths.md parse', CH.paths_md_selection('# t\n\n## Path 1 - rank 2: episode E1\nx\n## Path 2 - rank 3: episode E2 \n') == [(2, 'E1'), (3, 'E2')])
    L2 = dict(status='filled', link='post', n_histories=2, n_records=0)
    L1 = dict(status='filled', link='post', n_histories=1, n_records=0)
    check('reading: >= 2 histories -> TIED, 1 -> ONE-HISTORY (positive control), pending -> PENDING, no take -> NO-EVENT',
          (CH.reading_token(L2), CH.reading_token(L1), CH.reading_token(dict(L2, status='pending')), CH.reading_token(dict(L1, link='takes')))
          == ('TIED', 'ONE-HISTORY', 'PENDING', 'NO-EVENT'))
    with Mut('tied_as_identified'):
        check('mutant tied_as_identified is live (a tied link would render IDENTIFIED)', CH.reading_token(L2) == 'IDENTIFIED' and CH.reading_token(L1) == 'ONE-HISTORY')
    hyps = dict(EX=dict(union=True, size_lo=30., size_hi=30.), PL=dict(union=True, size_lo=30.01, size_hi=float('inf')))
    sub, lo, up, txt = CH.size_fields(30., hyps)
    check('size: the filled size is a lower bound, the submitted size unknown (never the filled bound)',
          np.isnan(sub) and lo == 30. and np.isinf(up) and 'submitted size unknown' in txt and 'LOWER bound' in txt, txt)
    with Mut('submitted_from_filled'):
        sub_m, _, _, txt_m = CH.size_fields(30., hyps)
        check('mutant submitted_from_filled is live', sub_m == 30. and 'submitted size 30.00' in txt_m, txt_m)
    fills = [dict(hash='0xaa' + '0' * 62, sid=f's{k}' * 4, mk_idx=0, shares=10. + k, hi=ws + 10 * k, xlo=ws - 2000, status='ok') for k in range(3)]
    lines, n = CH.render_orders(fills, ws)
    od = [m for m in (CH.ORDER_RE.match(x) for x in lines) if m]
    check('three same-hash fills render as ONE order with 3 fills', n == 1 and len(od) == 1 and od[0]['k'] == '3', lines)
    lines2, n2 = CH.render_orders(fills[:2] + [dict(fills[2], hash='0xbb' + '0' * 62)], ws)
    check('two hashes render as two orders', n2 == 2, lines2)
    with Mut('split_same_hash'):
        _, n3 = CH.render_orders(fills, ws)
        check('mutant split_same_hash is live', n3 == 3)
    rec = dict(outputs={'a.csv': dict(sha256='A' * 64), 'b.csv': 'b' * 64}, inputs={'data/x/econ/f.parquet': 'c' * 64, 'q/f.parquet': 'd' * 64})
    check('binding lookup: dict / string / unique suffix / ambiguous suffix / missing',
          (CH.bound_sha(rec, 'outputs', 'a.csv'), CH.bound_sha(rec, 'outputs', 'b.csv'), CH.bound_sha(rec, 'inputs', 'econ/f.parquet'),
           CH.bound_sha(rec, 'inputs', 'f.parquet'), CH.bound_sha(rec, 'outputs', 'z.csv')) == ('a' * 64, 'b' * 64, 'c' * 64, None, None))


def s_draft(world):
    rr, dr, F = world
    out = rr / 'chains/chains_v3_draft'
    s0 = snapshot(rr.parent)
    res = run_q(rr, dr, out)
    s1 = {k: v for k, v in snapshot(rr.parent).items() if not k.startswith(str(out))}
    check('draft writes exactly the four outputs into chains_v3_draft', sorted(p.name for p in out.iterdir()) == sorted(CH.OUT_FILES))
    check('draft touches no other file (and leaves no partial directory)', s0 == s1, set(s0) ^ set(s1))
    rec = json.loads((out / 'receipt.json').read_text())
    check('receipt: sha256 of every output = its bytes; CPU and VmHWM recorded',
          all(sha(out / nm) == v['sha256'] for nm, v in rec['outputs'].items()) and set(rec['outputs']) == {'chains.md', 'chain_links.csv', 'claims.json'}
          and rec['cpu_s']['process_total'] > 0 and rec['vmhwm_mib'] > 0)
    check('receipt: every input recorded with its sha256 and an ok binding', all(v['binding']['status'] == 'ok' for v in rec['inputs'].values())
          and len(rec['inputs']) == len(CH.INPUTS) + 3, {k: v['binding']['status'] for k, v in rec['inputs'].items()})
    cl = json.loads((out / 'claims.json').read_text())
    LK = links_of(out)
    md = (out / 'chains.md').read_text()
    check('six chains x seven links, every link graded and with a declared reading', len(LK) == 42 and cl['predicates']['every_link_graded']
          and cl['predicates']['every_link_has_a_declared_reading'] and cl['predicates']['every_chain_has_every_link'])
    check('selection = paths.md (first appearance: e018 before e006)', cl['predicates']['selection_equals_correction_v1_paths_md']
          and [e['episode_id'] for e in cl['population']['episodes']][:2] == [f"{DAY}_11_r1_e018", f"{DAY}_12_r1_e006"])
    pre = LK[LK.link == 'pre_state']
    check('missing posting_opportunity: every pre-state link PENDING with the reason (draft)',
          (pre.status == 'pending').all() and (pre.reading == 'PENDING').all() and pre.pending_reason.str.contains('posting_opportunity receipt absent').all())
    qu = LK[LK.link == 'queue']
    check("missing queue_front_v2: queue links on v1 labels with the 'v1 labels, see section 5' marker",
          (qu.status == 'filled_v1_labels').all() and md.count('[v1 labels, see section 5]') == 12)
    fl = LK[LK.status != 'pending']
    tie_ok = ((fl.n_histories >= 2) == (fl.reading == 'TIED')) | (fl.reading == 'NO-EVENT')
    check('TIED iff >= 2 retained histories; no IDENTIFIED anywhere', bool(tie_ok.all()) and '**IDENTIFIED**' not in md
          and not (LK.reading == 'IDENTIFIED').any() and cl['predicates']['tied_links_never_rendered_identified'])
    c2 = LK[LK.chain == 2].set_index('link')
    check('positive control: chain 2 (G2 <= 1 s, EX only) renders post and ending ONE-HISTORY', c2.loc['post', 'reading'] == 'ONE-HISTORY'
          and c2.loc['ending', 'reading'] == 'ONE-HISTORY' and c2.loc['value', 'reading'] == 'ONE-HISTORY', c2[['reading', 'n_histories']].to_dict())
    check('v1 labels never let a queue link agree (masked netting kept)', (qu.reading == 'TIED').all())
    B = CH.parse_md(md)
    b3 = B[(3, 'QUEUE')]
    od = [m for m in (CH.ORDER_RE.match(x) for x in b3['lines']) if m]
    check('chain 3: three same-hash fills render as ONE order (3 fills); v1 behind rows of the front hash -> later_same_hash',
          len(od) == 1 and od[0]['k'] == '3' and int(qu[qu.chain == 3].n_orders_rendered.iloc[0]) == 1
          and sum('later_same_hash' in x for x in b3['lines']) == 1 and 'v1 behind k=2' in '\n'.join(b3['lines']), b3['lines'])
    po = LK[LK.link == 'post']
    exp = [f['tot'] for _, f in F]
    check('post: submitted size never set; the filled size kept as the lower bound', po.submitted_size.isna().all()
          and np.allclose(po.size_lower_bound.to_numpy(float), exp) and all('submitted size unknown' in ' '.join(B[(k, 'POST')]['lines']) for k in range(1, 7)))
    tk = LK[LK.link == 'takes'].set_index('chain')
    check('takes: chain 1 NO-EVENT; chain 4 one record (the outside-window record dropped) ONE-HISTORY; chain 5 TIED (clock disagreement x variant '
          'disagreement = 4 histories); chain 6 ONE-HISTORY with the case marked',
          tk.loc[1, 'reading'] == 'NO-EVENT' and tk.loc[4, 'n_records'] == 1 and tk.loc[4, 'reading'] == 'ONE-HISTORY' and tk.loc[5, 'n_histories'] == 4
          and tk.loc[6, 'reading'] == 'ONE-HISTORY' and '[the selected take case]' in '\n'.join(B[(6, 'TAKES')]['lines']), tk[['reading', 'n_histories', 'n_records']].to_dict())
    en = LK[LK.link == 'ending'].set_index('chain')
    check('ending grades: S_FIFO/S_NOPRIO cancel-compatible -> cancel_compatible_ending; PL only under S_FIFO_LO -> scenario_dependent_ending',
          en.loc[1, 'evidence_grade'] == 'cancel_compatible_ending' and en.loc[3, 'evidence_grade'] == 'scenario_dependent_ending'
          and en.loc[2, 'evidence_grade'] == 'scenario_dependent_ending' and en.loc[3, 'reading'] == 'TIED')
    rp = LK[LK.link == 'repost'].set_index('chain')
    check('repost: sequential and tied successors stay scenario_dependent_link (unrelated feasible), none -> unresolved; never identified',
          rp.loc[1, 'evidence_grade'] == 'scenario_dependent_link' and rp.loc[2, 'evidence_grade'] == 'unresolved_alternatives'
          and (rp.reading == 'TIED').all() and 'identified repost: no' in md and not LK.identified_private_action.any())
    check('reconciliations hold on the consistent world (episode legs, own rows, closes per leg, carry row, census, pulls views, union, life fills)',
          cl['reconciliation_all_hold'] and len(cl['reconciliation']) >= 30, [k for k, v in cl['reconciliation'].items() if not v['holds']])
    v2 = [ln for ln in B[(2, 'VALUE')]['lines'] if ln.startswith('  - episode, all legs')][0]
    end_net = -6.6666 * (0.20 - 0.13) + 3.3333 * (0.20 - 0.14) - TK_FEE * 3.3333
    seg = v2.split('END valid')[1].split(';')[0] if 'END valid' in v2 else ''
    check('value: hand-computed chain 2 END net (-0.27 $ on 2/2 valid legs) and the 60 s beyond_end row counted invalid in chain 1',
          f'net {end_net:+.2f} $' in seg and seg.strip().startswith('2/2') and 'beyond_end' in ' '.join(B[(1, 'VALUE')]['lines']), v2)
    check('row groups pruned: fills_valued (a non-chain market) and closes (another day)', rec['counts'].get('fills_valued_groups_pruned', 0) >= 1
          and rec['counts'].get('closes_groups_pruned', 0) >= 1, rec['counts'])
    ag = cl['agreement']['2']
    check('agreement: per chain the agreeing links, the tied links with their hidden actions and the first branching link',
          'post' in ag['agree'] and ag['first_branch'] == 'queue' and all(ag['hidden'][k] for k in ag['tied']) and 'pre_state' in ag['pending'], ag)
    check('draft records the pending links in claims (no_pending_link False)', not cl['predicates']['no_pending_link'] and len(cl['pending']) == 6)
    check('run returns the same predicates it writes', res['claims']['predicates'] == cl['predicates'])


def s_refusals(world):
    rr, dr, _ = world
    out = rr / 'chains/chains_v3_draft'
    expect('overwrite: a second draft into the same directory is refused (write-once)', lambda: run_q(rr, dr, out), 'write-once')
    o2 = rr / 'chains/run_nosib'
    ok0, info0 = refuses(lambda: run_q(rr, dr, o2, mode='run'), 'run requires the sibling outputs')
    check('run without the sibling receipts refuses before any write', ok0 and not o2.exists(), info0)
    ok, info = refuses(lambda: run_q(rr, dr, rr / 'chains/run_nosib2', mode='run'), 'queue_front_v2 receipt absent')
    ok2, _ = refuses(lambda: run_q(rr, dr, rr / 'chains/run_nosib3', mode='run'), 'posting_opportunity receipt absent')
    check('the run refusal message names each missing sibling receipt', ok and ok2, info)
    with Mut('tied_as_identified'):
        expect('a mutant outside the selftest is refused', lambda: CH.run(mode='draft', rroot=rr, droot=dr, out=rr / 'chains/m0', require_freeze=False,
                                                                          log=lambda *a: None), 'planted mutant')
    with Mut('bogus'):
        expect('an unknown mutant is refused', lambda: run_q(rr, dr, rr / 'chains/m1'), 'unknown MT_CH3_MUT')


def s_mutants(world):
    rr, dr, _ = world
    for m, needle in (('tied_as_identified', 'a TIED link is never rendered as identified'),
                      ('split_same_hash', 'repeated same-hash fills must render as one order'),
                      ('submitted_from_filled', 'a submitted size is set')):
        o = rr / f'chains/mut_{m}'
        with Mut(m):
            ok, info = refuses(lambda: run_q(rr, dr, o), needle)
        check(f'mutant {m} caught by the rendering verifier before any write',
              ok and not o.exists() and not any(p.name.startswith(f'.mut_{m}') for p in (rr / 'chains').iterdir()), info)


def s_run_siblings(world):
    rr, dr, F = world
    out = rr / 'chains/chains_v3'
    run_q(rr, dr, out, mode='run')
    cl = json.loads((out / 'claims.json').read_text())
    LK = links_of(out)
    md = (out / 'chains.md').read_text()
    check('run with both siblings bound: every predicate holds, no pending link, reconciliations and bindings ok',
          all(cl['predicates'].values()) and cl['reconciliation_all_hold'] and cl['bindings_all_ok'] and not cl['pending'],
          [k for k, v in cl['predicates'].items() if not v])
    qu = LK[LK.link == 'queue'].set_index('chain')
    B = CH.parse_md(md)
    check('queue links use queue_front_v2 labels (no v1 marker in any queue block)', (qu.status == 'filled').all()
          and not any('v1 labels' in ' '.join(B[(k, 'QUEUE')]['lines']) for k in range(1, 7)))
    check('v2 positive control: chain 2 level with alt_netting_possible False and one S_FIFO position -> ONE-HISTORY; others TIED',
          qu.loc[2, 'reading'] == 'ONE-HISTORY' and (qu.drop(index=2).reading == 'TIED').all(), qu[['reading', 'n_histories']].to_dict())
    q3 = ' '.join(B[(3, 'QUEUE')]['lines'])
    check('v2 rows of the 3-fill hash under ONE order: the front-candidate set (candidates.csv) and earliest_set + later_same_hash x2 (fills.csv)',
          'front-candidate set at level 102 (the candidate hash' in q3 and q3.count('later_same_hash at level 102') == 2
          and 'earliest_set at level 102 k=0' in q3 and qu.loc[3, 'n_orders_rendered'] == 1, q3)
    check('v2 endpoint occupancy shown as an endpoint observation (never an actual-post queue label)',
          all('endpoint occupancy cell_empty_at_window_lo' in ' '.join(B[(k, 'QUEUE')]['lines']) for k in range(1, 7))
          and 'never an actual-post queue label' in q3)
    rec = json.loads((out / 'receipt.json').read_text())
    check('receipt binds every sibling file read (levels, candidates, fills, occupancy, rows)',
          sorted(rec['siblings']['queue_front_v2']['files']) == ['candidates', 'fills', 'levels', 'occupancy']
          and list(rec['siblings']['posting_opportunity']['files']) == ['rows'], rec['siblings'])
    pre = LK[LK.link == 'pre_state'].set_index('chain')
    check('pre-state: chain 1 TIED (spread changes inside its 78 s window, the hidden action names it); chain 2 ONE-HISTORY (constant)',
          pre.loc[1, 'reading'] == 'TIED' and 'spread' in str(pre.loc[1, 'hidden_action']) and pre.loc[2, 'reading'] == 'ONE-HISTORY'
          and pre.loc[1, 'evidence_grade'] == 'observed_state', pre[['reading', 'n_histories', 'hidden_action']].to_dict())
    p1 = ' '.join(B[(1, 'PRE-STATE')]['lines'])
    check("pre-state: the D0 columns are read (spread_bin_D0), D1000 beside, and the visible creation at the order's cell is listed with its type",
          'spread_bin_D0' in p1 and 'D1000 (the whole state lagged one second): 2 distinct' in p1 and "(tick 69): 11|ask|69|600 (deep)" in p1, p1)


def s_falsifiers(tmp):
    rr, dr, _ = build_world(tmp / 'fal', siblings=True, es_perturb=True, paths_swap=True)
    with open(rr / 'chains/pull/pulls.csv', 'a') as f:                # a byte change after the receipt was written
        f.write('\n')
    out = rr / 'chains/chains_v3_draft'
    run_q(rr, dr, out)
    cl = json.loads((out / 'claims.json').read_text())
    check('falsifier recorded in a draft: perturbed ep_sh -> episode_ep_sh_reconciles False', not cl['reconciliation']['episode_ep_sh_reconciles']['holds'])
    check('falsifier recorded in a draft: a byte-changed bound input -> binding mismatch', cl['bindings']['pulls'] == 'mismatch' and not cl['bindings_all_ok'])
    check('falsifier recorded in a draft: swapped paths.md -> selection predicate False', not cl['predicates']['selection_equals_correction_v1_paths_md'])
    o = rr / 'chains/chains_v3'
    ok, info = refuses(lambda: run_q(rr, dr, o, mode='run'), 'a run refuses on failed predicates')
    check('a run refuses on the falsifiers, naming selection, reconciliation and bindings, before any write',
          ok and all(x in info for x in ('selection_equals_correction_v1_paths_md', 'reconciliation', 'bindings')) and not o.exists(), info)


def s_sibling_schema(tmp):
    rr, dr, _ = build_world(tmp / 'sch', siblings=True, po_drop='fair_gap_bin_D0', qf2_bad_sha=True)
    out = rr / 'chains/chains_v3_draft'
    run_q(rr, dr, out)
    LK = links_of(out)
    pre, qu = LK[LK.link == 'pre_state'], LK[LK.link == 'queue']
    check("a sibling missing a feature column: pre-state links PENDING naming the column (draft)",
          (pre.status == 'pending').all() and pre.pending_reason.str.contains("required columns missing \\['fair_gap'\\]").all(), pre.pending_reason.iloc[0])
    check('a sibling file not matching its receipt: queue links fall back to v1 labels with the reason (draft)',
          (qu.status == 'filled_v1_labels').all() and qu.pending_reason.str.contains('differs from its receipt').all(), qu.pending_reason.iloc[0])
    o = rr / 'chains/chains_v3'
    ok, info = refuses(lambda: run_q(rr, dr, o, mode='run'), 'differs from its receipt')
    check('a run refuses a sibling whose file does not match its receipt, before any write', ok and not o.exists(), info)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='chains_v3_selftest_'))
    os.environ.pop('MT_CH3_MUT', None)
    try:
        section('units', s_units)
        W = {}
        section('world', lambda: W.update(base=build_world(tmp / 'base'), sib=build_world(tmp / 'sib', siblings=True)))
        if 'base' in W:
            section('draft', s_draft, W['base'])
            section('refusals', s_refusals, W['base'])
            section('mutants', s_mutants, W['base'])
        if 'sib' in W:
            section('siblings', s_run_siblings, W['sib'])
        section('falsifiers', s_falsifiers, tmp)
        section('schema', s_sibling_schema, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_bad = sum(not ok for _, ok, _ in RES)
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  :: {info}'))
    print(json.dumps(dict(checks=len(RES), failed=n_bad, cpu_s=round(__import__('time').process_time(), 3), vmhwm_mib=round(CH.hwm(), 1))))
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
