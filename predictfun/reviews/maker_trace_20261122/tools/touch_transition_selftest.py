"""touch_transition_selftest.py - synthetic proof of touch_transition.py (P-2026-005 MAKER-TRACE, TOUCH-TRANSITION-AND-RESPONSE v1;
touch_response_contract.md sections 1-4, commit 668e560db). The REAL pipeline runs in-process on synthetic BOOK markets
(posting_opportunity_selftest.SM -> chain_context.build_market -> pf_mt_lifecycle._book; queue_front.market_book / market_fills / market_core;
posting_opportunity.market_pass / identity / rows_frame / verdict; take_logic.mh_rr / tail_counts / tl_codes / align_codes; chains_v3.views /
read_groups / bound_sha), with every saved upstream artifact written to a temporary tree, bound by its receipt and read back through the tool's
readers. Frames every 200 ms at ws + 100 + 200 j; base book bid 49 / ask 50 (spread 1); every expectation is planted or hand-computed.
GATE (contract section 4): each fixture has a named check that passes on the base AND fails under its planted error MT_TT_MUT:
 (1) G1 anchor_mask        a same-second spread-one -> retreat -> restoring sequence (the anchor spread is 1: the v1 anchor mask excluded it) is
                           a retreat episode with a restoring response (no frame between, exposure 200 ms, bounds [0, 200]); its creation is an
                           anchor-excluded improve event reconciled to class improve_ask, role response_new_best.
 (2) G2 split_two_side     simultaneous best changes (widen / narrow / both_up / both_down) are two_side, start no episode, fill two_side.csv.
 (3) G3 drop_alternatives  masked cancellation / replenishment: the response's candidate keeps queue_front_v2's alternatives (netting possible,
                           two first-filler addresses), graded G4 -> candidate_scenario, pre-state unresolved.
 (4) G4 conflict_identified a G1 (conflicted) posting window inside the episode -> candidate_scenario, unresolved_outside_window; the G2
                           control -> candidate_constrained, episode_state_G2_conditional.
 (5) G5 split_same_hash    three later fills of one hash = ONE order (three legs / records, 30 sh) with its 10 s / 60 s / END views and the
                           shared valid cohort, turnover / residual, takes after the creation and the successor alternatives.
 (6) G6 missing_as_unchanged unknown-touch pairs are class unknown; an episode followed by an unknown touch / a frame gap is censored_gap with
                           its reason and exposure; the census counts gap and unknown.
 (7) G7 no_horizon         a later-frame perturbation (every frame after T arrives 37 ms later, a new later cell, BTC +5 bp after T): every
                           episode starting <= T - 2 s is unchanged in every column; a later one changes (the perturbation bites).
 (8) G8 state_leak         the D0 state is frame f - 1 (spread 1, depth 100, its time), LAG frame f - 2; changing frame f's sizes and the BTC
                           grid in (upd(f-1), upd(f)] leaves every D0 / LAG state column unchanged.
Also: classes / census / reconciliation / termination predicates on the base (the termination census by the class of the ending pair); the
pooled continuation counts a take after two linked creations of one address ONCE (distinct, order-sum and overlap); the owner grading reproduces the saved posting_opportunity
identity of every response cell (po_parity equal); the MH toy (hand-computed 5/3, crude 1.5) and take_logic.mh_rr parity; the permutation
(per-stratum exposed counts kept in every draw, labels move, deterministic per seed); a planted hazard world (btc_with RR 3 -> supported) and a
twin world (RR exactly 1 for three contrasts, testable, no verdict fires); the readers' sha256 bindings and interface refusals; the pilot path
end to end through execute() (nine outputs, receipt hashes, write-once); the production refusals (mutants, days, freeze / projection).
Usage (run_light, MT_RUN=v2): touch_transition_selftest.py"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import touch_transition as TT  # noqa: E402
import posting_opportunity_selftest as POS  # noqa: E402  (its synthetic-market builder SM and regrade helpers; reused, never edited)

PO, QF, QF2, CX, C, TL, CH3 = TT.PO, TT.QF, TT.QF2, TT.CX, TT.C, TT.TL, TT.CH3
NAN = float('nan')
RES = []
A1, A2, A3 = TT.TOP3
A1L, A2L, A3L = TT.TOP3_L
OTHER = POS.OTHER
OTHER_L = OTHER.lower()
DAY = '20261024'
CTX = PO.CT.make_ctx(CX)
T7 = 50_000                                             # fixture 7: the perturbation instant (ws-relative ms)
G_REL = (10100, 10300, 10500, 20300, 21300, 22300, 23300, 30500, 30700, 30900, 31300, 31500, 35300, 35500, 70500, 75500, 78500, 80500)
GRADES = {'0xhR3': 'G4_scenario_only', '0xhR4': 'G1_conflict', '0xhR4b': PO.G2, '0xhR5': 'G4_scenario_only'}
PLANT = {('bid', 49, 70500): dict(addr=A1L, h='0xhR3', alt=f'{A1L}|{OTHER_L}' if A1L < OTHER_L else f'{OTHER_L}|{A1L}', n_alt=2, other=True),
         ('ask', 50, 75500): dict(addr=A2L, h='0xhR4', alt=A2L, n_alt=1, other=False),
         ('bid', 49, 78500): dict(addr=A3L, h='0xhR4b', alt=A3L, n_alt=1, other=False),
         ('bid', 49, 80500): dict(addr=A1L, h='0xhR5', alt=A1L, n_alt=1, other=False)}
EXPECT_CATCH = dict(anchor_mask='G1', split_two_side='G2', drop_alternatives='G3', conflict_identified='G4', split_same_hash='G5',
                    missing_as_unchanged='G6', no_horizon='G7', state_leak='G8')


def check(name, ok, info=''):
    RES.append((name, bool(ok), str(info)[:900]))


def expect(name, fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, needle in str(e), f'{type(e).__name__}: {e}')
        return
    check(name, False, 'did not raise')


def section(name, fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f'{name}: section crashed', False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-1800:]}')
        return None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return NAN


def _eqv(a, b):
    a, b = _f(a), _f(b)
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-9 * max(1., abs(a), abs(b))


# ------------------------------------------------------------------ synthetic markets ------------------------------------------------------------------
def gate_market():
    """Fixtures 1-6 and 8 on one synthetic BOOK market (base book bid 49 / ask 50: spread 1 everywhere else)."""
    m = POS.SM(DAY, 980001, 1)
    m.setc(-10 ** 6, 10 ** 7, 'bid', 49, 100.).setc(-10 ** 6, 10 ** 7, 'ask', 50, 100.)
    m.setc(10300, 10500, 'ask', 50, 0.).setc(10300, 10500, 'ask', 51, 60.)                      # G1 / G8: retreat 10.3 s, restoring 10.5 s
    m.setc(20300, 21300, 'bid', 49, 0.).setc(20300, 21300, 'bid', 48, 100.)                     # G2: widen 20.3 s, narrow 21.3 s
    m.setc(20300, 21300, 'ask', 50, 0.).setc(20300, 21300, 'ask', 51, 100.)
    m.setc(22300, 23300, 'bid', 50, 100.).setc(22300, 23300, 'ask', 50, 0.).setc(22300, 23300, 'ask', 51, 100.)   # both_up / both_down
    m.unknown(30500, 'bid', 60).unknown(30700, 'bid', 60)                                       # G6a: an unknown cell above the best bid
    m.setc(31300, 33100, 'bid', 49, 0.).setc(31300, 33100, 'bid', 48, 100.)                     # G6b: retreat 31.3 s, unknown touch 31.7 s
    m.unknown(31700, 'bid', 70)
    m.setc(35300, 37500, 'ask', 50, 0.).setc(35300, 37500, 'ask', 51, 100.)                     # G6c: retreat 35.3 s, frame gap 35.5 -> 36.9 s
    m.drop(35600, 36800)
    m.setc(70300, 70500, 'bid', 49, 0.).setc(70300, 70500, 'bid', 48, 100.)                     # G3: masked-netting alternatives
    m.fill(A1, 'bid', 49, 20., 71000, hsh='0xhR3', post_lo=(70200,) * 3, post_hi=71000)
    m.setc(75300, 75500, 'ask', 50, 0.).setc(75300, 75500, 'ask', 51, 100.)                     # G4: G1 window inside the episode
    m.fill(A2, 'ask', 50, 15., 76000, hsh='0xhR4', post_lo=(75200,) * 3, post_hi=75500)
    m.setc(78300, 78500, 'bid', 49, 0.).setc(78300, 78500, 'bid', 48, 100.)                     # G4 control: G2 window inside the episode
    m.fill(A3, 'bid', 49, 10., 79000, hsh='0xhR4b', post_lo=(78200,) * 3, post_hi=78500)
    m.setc(80300, 80500, 'bid', 49, 0.).setc(80300, 80500, 'bid', 48, 100.)                     # G5: three later same-hash fills
    for h_ in (81000, 82000, 83000):
        m.fill(A1, 'bid', 49, 10., h_, hsh='0xhR5', post_lo=(80200,) * 3, post_hi=81000)
    return m


def g8_market(g):
    """G8: the gate market with frame f (10.3 s) resized (ask 51 60 -> 80, a new deep bid 47) and the BTC grid +20 bp after 10.1 s."""
    m = g.copy(980003)
    j = m.fi(10300)
    m.S[j, 1, 51] = 80.
    m.S[j, 0, 47] = 33.
    gt = (m.G0 + np.arange(len(m.lm), dtype=np.int64)) * 100
    m.lm = np.where(gt > m.ws + 10100, m.lm + 2e-3, m.lm)
    return m


def perturb_pair():
    """G7: a market and its record with every frame after T arriving 37 ms later, a new later cell and BTC +5 bp after T."""
    m = POS.SM(DAY, 980002, 2, seed=7, btc_sd=0.3)
    m.setc(-10 ** 6, 10 ** 7, 'bid', 49, 100.).setc(-10 ** 6, 10 ** 7, 'ask', 50, 100.)
    m.setc(40300, 40500, 'ask', 50, 0.).setc(40300, 40500, 'ask', 51, 100.)                     # an early response (40.5 s)
    m.setc(46300, 50500, 'bid', 49, 0.).setc(46300, 50500, 'bid', 48, 100.)                     # an early no-response episode; refill after T
    m.setc(52300, 52700, 'ask', 50, 0.).setc(52300, 52700, 'ask', 51, 100.)                     # a later episode
    p = m.copy(980004)
    late = p.t > p.ws + T7
    p.t = np.where(late, p.t + 37, p.t)
    p.S[p.t > p.ws + T7 + 3000, 0, 20] = 15.
    gt = (p.G0 + np.arange(len(p.lm), dtype=np.int64)) * 100
    p.lm = np.where(gt > p.ws + T7, p.lm + 5e-4, p.lm)
    p.mid = m.mid
    return m, p


def frame_index(M, rels):
    t = np.asarray(CX.frame_times(M), np.int64)
    out = {}
    for r in rels:
        j = int(np.searchsorted(t, M.ws + r))
        TT.need(j < len(t) and int(t[j]) == M.ws + r, f'selftest: no usable frame at ws + {r}')
        out[r] = j
    return out


def candidates_for(m, M):
    """queue_front_v2-shaped candidates (QF2.CAND_COLS) of every E1 level of the synthetic market (status no_fill, no earliest record) with the
    PLANT fixture rows set to front_candidate (book_candidate timing, the planted alternatives)."""
    bk = QF.market_book(M)
    fl, _ = QF.market_fills(M, None)
    LV = QF.market_core(m.day, m.mid, M.ws, M.we, bk, fl)['levels']
    E1 = LV[(LV['unit'] == 'E1').to_numpy()].reset_index(drop=True)
    neg1 = ('candidate_interval_lo_ms', 'candidate_interval_hi_ms', 'candidate_lo_ms', 'candidate_hi_ms', 'n_earliest_set', 'n_members')
    zero = ('alt_sites_life', 'alt_sites_unexplained', 'alt_first_n_addresses', 'n_old_life')
    bools = ('alt_netting_possible', 'alt_first_other_address', 'earliest_one_address', 'straddle_record')
    t = np.asarray(bk['t'], np.int64)
    rows = []
    for r in E1.to_dict('records'):
        x = {c: '' for c in QF2.CAND_COLS}
        x.update({c: -1 for c in neg1})
        x.update({c: 0 for c in zero})
        x.update({c: False for c in bools})
        x.update(day=m.day, mid=m.mid, level_id=int(r['level_id']), side=str(r['side']), tick=int(r['tick']), frame=int(r['frame']), fid=int(r['fid']),
                 t_open_ms=int(r['t_open_ms']), t_create_ms=int(r['t_create_ms']), phase=str(r['phase']), status='no_fill',
                 reason='window_end_without_fill', creation_context=str(r['mechanism']))
        key = (str(r['side']), int(r['tick']), int(t[int(r['frame'])]) - M.ws)
        pl = PLANT.get(key)
        if pl is not None:
            o = m.orders[pl['h']]
            x.update(status=QF2.FRONT, reason='ok', candidate_addr=pl['addr'], candidate_group=QF.GROUPS[TT.TOP3_L.index(pl['addr'])],
                     candidate_hash=pl['h'], earliest_set_hashes=pl['h'], timing_basis='book_candidate', candidate_interval_lo_ms=int(r['t_open_ms']),
                     candidate_interval_hi_ms=int(r['t_create_ms']), candidate_lo_ms=int(o['post_lo'][0]), candidate_hi_ms=int(o['post_hi']),
                     alt_netting_possible=True, alt_sites_life=1, alt_first_addresses=pl['alt'], alt_first_n_addresses=pl['n_alt'],
                     alt_first_other_address=pl['other'], earliest_set_addresses=pl['addr'], n_earliest_set=1, n_members=1)
        rows.append(x)
    D = pd.DataFrame(rows, columns=list(QF2.CAND_COLS))
    TT.need(int((D['status'] == QF2.FRONT).sum()) == len(PLANT), 'selftest: a planted candidate cell is not an E1 level')
    return D


def fv_rows(m):
    """Synthetic fills_valued rows (TT.FVV_READ) of the market's maker legs (0xhR5 planted: 10 s 0.1, 60 s 0.2, END 0.3 / 0.3 / beyond_end,
    mq 10 / 10 / 5, rq 0 / 0 / 5) plus one taker row that must stay out."""
    rows = []
    k5 = 0
    for L in m.legs:
        r = dict(day=m.day, mid=m.mid, sid=L['sid'], mk_idx=int(L['mk_idx']), role='maker', addr=L['addr'], hash=L['mk_hash'], q=float(L['q']),
                 self_trade=False, settle_ms=int(L['hi']), st_res='ok', st_10s='ok', v_10s=0., st_60s='ok', v_60s=0., st_end='ok', v_end=0.,
                 fee_fill=0., net_res=0., mq=float(L['q']), rq=0., status='ok')
        if L['mk_hash'] == '0xhR5':
            r.update(v_10s=0.1, v_60s=0.2, v_end=0.3 if k5 < 2 else NAN, st_end='ok' if k5 < 2 else 'beyond_end', fee_fill=0.01, net_res=0.05,
                     mq=(10., 10., 5.)[k5], rq=(0., 0., 5.)[k5])
            k5 += 1
        rows.append(r)
    t5 = next(L for L in m.legs if L['mk_hash'] == '0xhR5')
    rows.append(dict(rows[-1], sid=t5['sid'], mk_idx=int(t5['mk_idx']), role='taker', addr=OTHER, hash='0xhtaker', v_10s=9., v_60s=9., v_end=9.))
    return pd.DataFrame(rows, columns=list(TT.FVV_READ))


def pf_rows(m):
    """Synthetic chain_post_fill orders.csv rows (TT.PF_READ) of the top-3 orders."""
    rows = []
    for h, o in m.orders.items():
        legs = [L for L in m.legs if L['mk_hash'] == h]
        rk = TT.TOP3_L.index(o['addr'].lower()) + 1
        nk = 'reprice' if h == '0xhR5' else 'none'
        rows.append(dict(day=m.day, mid=m.mid, rank=rk, addr=o['addr'], mk_hash=h, y_side=o['side'], tick=o['tick'], n_fills=len(legs),
                         cum_shares=float(sum(L['q'] for L in legs)), first_hi_ms=min(L['hi'] for L in legs), last_hi_ms=max(L['hi'] for L in legs),
                         post_lo_union_ms=min(o['post_lo']), post_hi_ms=o['post_hi'], size_status='tied', union_histories='EX|PL', next_kind=nk,
                         next_link_status='tied' if h == '0xhR5' else 'none', n_children=2 if h == '0xhR5' else 0,
                         next_hash='0xhnext' if h == '0xhR5' else '', mq_sh=25. if h == '0xhR5' else 0., rq_sh=5. if h == '0xhR5' else 0., status='ok'))
    return pd.DataFrame(rows, columns=list(TT.PF_READ))


def tk_rows(m):
    """Synthetic take_logic_v2 events (TT.TK_READ) of rank 1 in the gate market: one before every creation, one 1-3 s after 80.5 s, one ~20 s after."""
    w = m.ws
    rows = [dict(day=m.day, mid=m.mid, addr=A1, sid='tk0', s=1, dq=4., x_lo_hard_ms=w + 60000, hi_ms=w + 62000, settle_ms=w + 62100, status='ok', dir_D0='same'),
            dict(day=m.day, mid=m.mid, addr=A1, sid='tk1', s=-1, dq=-8., x_lo_hard_ms=w + 81500, hi_ms=w + 83500, settle_ms=w + 83600, status='ok',
                 dir_D0='opposing'),
            dict(day=m.day, mid=m.mid, addr=A1, sid='tk2', s=1, dq=5., x_lo_hard_ms=w + 100000, hi_ms=w + 102000, settle_ms=w + 102100, status='ok',
                 dir_D0='same')]
    return pd.DataFrame(rows, columns=list(TT.TK_READ))


def write_inputs(tmp, g, MG):
    """Every saved upstream artifact of the gate market in a temporary tree, each bound by its receipt: posting_opportunity (events.csv +
    rows.parquet from the REAL posting_opportunity.market_pass graded with the planted candidates), queue_front_v2 (candidates.csv, levels.csv),
    correction_v1 (regrade), chain_post_fill (orders.csv), take_logic_v2 (events.parquet), the data root (identities.json, fills_valued.parquet)."""
    root = tmp / 'root'
    (root / 'econ').mkdir(parents=True)
    (root / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(day=C.ID_DAY, top10_shares=[dict(addr=a, rank=i + 1)
                                                                                                            for i, a in enumerate(TT.TOP3)]))))
    fvp = root / 'econ' / 'fills_valued.parquet'
    fv_rows(g).to_parquet(fvp, index=False)
    fsha = TT._sha(fvp)
    D = {k: tmp / k for k in ('po', 'qf2', 'regrade', 'post_fill', 'tl2')}
    for p in D.values():
        p.mkdir()
    CD = candidates_for(g, MG)
    CD.to_csv(D['qf2'] / 'candidates.csv', index=False)
    CD.to_csv(D['qf2'] / 'levels.csv', index=False)
    lsha = TT._sha(D['qf2'] / 'levels.csv')
    (D['qf2'] / 'receipt.json').write_text(json.dumps(dict(version=QF2.VERSION, kind='pilot', inputs={'econ/fills_valued.parquet': fsha},
                                                           outputs={'candidates.csv': dict(sha256=TT._sha(D['qf2'] / 'candidates.csv')),
                                                                    'levels.csv': dict(sha256=lsha)})))
    rg = POS.rg_rows(g, GRADES)
    POS.write_rg(D['regrade'], rg)
    Q2 = CD.rename(columns=PO.QF2_RENAME)
    pp = PO.market_pass(g.day, g.mid, MG, CTX, POS.rg_map(rg), Q2)
    pp['events'].to_csv(D['po'] / 'events.csv', index=False)
    PO.rows_frame(PO.assemble([pp])).to_parquet(D['po'] / 'rows.parquet', index=False)
    (D['po'] / 'receipt.json').write_text(json.dumps(dict(version=PO.VERSION, kind='pilot', inputs={'econ/fills_valued.parquet': fsha},
                                                          outputs={'events.csv': dict(sha256=TT._sha(D['po'] / 'events.csv')),
                                                                   'rows.parquet': dict(sha256=TT._sha(D['po'] / 'rows.parquet'))},
                                                          dependency=dict(queue_front_v2=dict(levels_sha256=lsha)))))
    PFd = pf_rows(g)
    PFd.to_csv(D['post_fill'] / 'orders.csv', index=False)
    (D['post_fill'] / 'receipt.json').write_text(json.dumps(dict(version='chain_post_fill_v1', inputs={'econ/fills_valued.parquet': dict(sha256=fsha)},
                                                                 outputs={'orders.csv': dict(sha256=TT._sha(D['post_fill'] / 'orders.csv'))},
                                                                 rows=dict(orders=len(PFd)))))
    tk_rows(g).to_parquet(D['tl2'] / 'events.parquet', index=False)
    (D['tl2'] / 'receipt.json').write_text(json.dumps(dict(version=TL2_VERSION(), kind='pilot',
                                                           inputs={'data/x/econ/fills_valued.parquet': fsha},
                                                           outputs={'events.parquet': dict(sha256=TT._sha(D['tl2'] / 'events.parquet'))})))
    return root, fvp, fsha, D, pp


def TL2_VERSION():
    return TT.TL2.VERSION


def fixtures(tmp):
    FX = {}
    g = gate_market()
    FX['G'], FX['MG'] = g, g.build()
    g8 = g8_market(g)
    FX['G8'], FX['MG8'] = g8, g8.build()
    p, p2 = perturb_pair()
    FX['P'], FX['MP'], FX['P2'], FX['MP2'] = p, p.build(), p2, p2.build()
    FX['ix'] = frame_index(FX['MG'], G_REL)
    root, fvp, fsha, D, pp = write_inputs(tmp, g, FX['MG'])
    FX.update(root=root, fvp=fvp, fsha=fsha, dirs=D, po_part=pp)
    mk = [(DAY, g.mid)]
    h = {}
    EV, RW, poi = TT.read_po(D['po'], mk, h, fsha)
    Q2, q2i = TT.read_qf2(D['qf2'], mk, h, fsha, poi['levels_sha256'])
    RG, _ = PO.read_regrade(D['regrade'], mk, h)
    PF, _ = TT.read_post_fill(D['post_fill'], mk, h, fsha)
    TK, _ = TT.read_takes(D['tl2'], mk, h, fsha)
    FX.update(EV=EV[mk[0]], RW=RW[mk[0]], Q2=Q2[mk[0]], RGm=RG.get(mk[0], {}), PF=PF, TK=TK, hashes=h, poi=poi, q2i=q2i)
    check('readers: posting_opportunity / queue_front_v2 / regrade / post_fill / take_logic_v2 read back through their receipts (every file hashed)',
          len(FX['EV']) == len(pp['events']) and len(FX['RW']) == 2 * g.K and len(FX['Q2']) > 0 and len(PF) == len(g.orders) and len(TK) == 3
          and len(h) == 11, dict(ev=len(FX['EV']), rw=len(FX['RW']), q2=len(FX['Q2']), pf=len(PF), tk=len(TK), hashes=len(h)))
    return FX


def compute(FX):
    """The tool's market passes (gate, G8, the perturbation pair) and the gate analysis, under whatever MT_TT_MUT is set."""
    W = {}
    pG = TT.market_pass(DAY, FX['G'].mid, FX['MG'], CTX, FX['EV'], FX['RW'], FX['Q2'], FX['RGm'])
    FV = TT.read_fv(FX['fvp'], pG['legs'], Counter())
    frames, tables, claims, dg = TT.analyse('selftest', [DAY], [pG], Counter(), dict(PF=FX['PF'], TK=FX['TK'], FV=FV), FX['RW'])
    W['G'] = dict(part=pG, EPD=frames['episodes.parquet'], T=tables, claims=claims, FV=FV)
    W['G8'] = TT.market_pass(DAY, FX['G8'].mid, FX['MG8'], CTX)
    W['P'] = TT.market_pass(DAY, FX['P'].mid, FX['MP'], CTX)
    W['P2'] = TT.market_pass(DAY, FX['P2'].mid, FX['MP2'], CTX)
    return W


# ------------------------------------------------------------------ the gate checks (each re-run under its mutant) ------------------------------------------------------------------
def _ep(E, rel_frame_ix, side):
    X = E[(E['frame'] == rel_frame_ix).to_numpy() & (E['side'] == side).to_numpy()]
    return X.iloc[0] if len(X) == 1 else None


def _resp(E, g_ix, side):
    X = E[(E['end_type'] == 'response').to_numpy() & (E['end_frame'] == g_ix).to_numpy() & (E['side'] == side).to_numpy()]
    return X.iloc[0] if len(X) == 1 else None


def _pair(W, f):
    PR = W['G']['part']['pairs']
    r = PR[(PR['f'] == f).to_numpy()]
    return (TT.CLASSES[int(r['cls'].iloc[0])], str(r['sub'].iloc[0])) if len(r) == 1 else (None, None)


def chk_G1(FX, W):
    ix, ws = FX['ix'], FX['G'].ws
    e = _ep(W['G']['EPD'], ix[10300], 'ask')
    ok_e = e is not None and e['end_type'] == 'response' and e['resp_type'] == 'restoring' and int(e['exposure_ms']) == 200 and \
        int(e['exposure_lo_ms']) == 0 and bool(e['no_frame_between']) and _eqv(e['delta_ticks'], 1.) and int(e['t_start_ms']) == ws + 10300 and \
        int(e['end_frame']) == ix[10500]
    RC = W['G']['T']['reconcile.csv']
    r = RC[(RC['frame'] == ix[10500]).to_numpy() & (RC['side'] == 'ask').to_numpy() & (RC['tick'] == 50).to_numpy()]
    ok_r = len(r) == 1 and r['po_status'].iloc[0] == 'event' and bool(r['is_improve'].iloc[0]) and bool(r['anchor_excluded'].iloc[0]) and \
        r['cls'].iloc[0] == 'improve_ask' and r['recon_status'].iloc[0] == 'class:improve_ask' and r['episode_role'].iloc[0] == 'response_new_best'
    return bool(ok_e and ok_r), dict(episode=None if e is None else {k: e[k] for k in ('end_type', 'resp_type', 'exposure_ms', 'exposure_lo_ms')},
                                     reconcile=r[['po_status', 'anchor_excluded', 'cls', 'episode_role']].to_dict('records') if len(r) else None)


def chk_G2(FX, W):
    ix = FX['ix']
    got = [_pair(W, ix[x]) for x in (20300, 21300, 22300, 23300)]
    ok = got == [('two_side', 'widen'), ('two_side', 'narrow'), ('two_side', 'both_up'), ('two_side', 'both_down')]
    E = W['G']['EPD']
    no_ep = not bool(E['frame'].isin([ix[x] for x in (20300, 21300, 22300, 23300)]).any())
    TS = W['G']['T']['two_side.csv']
    tot = TS[(TS['scope'] == 'pooled').to_numpy() & (TS['feature'] == 'total').to_numpy()]
    cnt = {str(r['sublabel']): int(r['n_pairs']) for r in tot.to_dict('records')}
    ok_t = cnt == dict(both_up=1, both_down=1, widen=1, narrow=1, all=4)
    return bool(ok and no_ep and ok_t), dict(classes=got, no_episode=no_ep, two_side_table=cnt)


def chk_G3(FX, W):
    r = _resp(W['G']['EPD'], FX['ix'][70500], 'bid')
    ok = r is not None and r['identity'] == 'candidate_scenario' and r['cand_addr'] == A1L and r['grade'] == 'G4_scenario_only' and \
        r['alt_netting'] == 'possible' and r['alt_first_other'] == 'possible' and set(str(r['alt_first_addresses']).split('|')) == {A1L, OTHER_L} and \
        int(r['alt_first_n_addresses']) == 2 and r['pre_state_link'] == 'unresolved_outside_window' and r['po_parity'] == 'equal'
    SU = W['G']['T']['continuations.csv']
    su = SU[(SU['table'] == 'summary').to_numpy() & (SU['scope'] == 'all').to_numpy()]
    met = {str(x['metric']): x['value'] for x in su.to_dict('records')}
    ok_s = int(met.get('alt_first_other:possible', -1)) == 1 and int(met.get('alt_netting:possible', -1)) == 4
    return bool(ok and ok_s), dict(row=None if r is None else {k: r[k] for k in ('identity', 'alt_netting', 'alt_first_other', 'alt_first_addresses')},
                                   summary={k: met.get(k) for k in ('alt_first_other:possible', 'alt_netting:possible')})


def chk_G4(FX, W):
    E = W['G']['EPD']
    r4, r4b = _resp(E, FX['ix'][75500], 'ask'), _resp(E, FX['ix'][78500], 'bid')
    ok = r4 is not None and r4b is not None and r4['identity'] == 'candidate_scenario' and r4['grade'] == 'G1_conflict' and bool(r4['win_contains']) and \
        r4['pre_state_link'] == 'unresolved_outside_window' and r4b['identity'] == 'candidate_constrained' and r4b['grade'] == PO.G2 and \
        r4b['pre_state_link'] == 'episode_state_G2_conditional'
    return bool(ok), dict(g1=None if r4 is None else {k: r4[k] for k in ('identity', 'grade', 'pre_state_link')},
                          g2=None if r4b is None else {k: r4b[k] for k in ('identity', 'grade', 'pre_state_link')})


def chk_G5(FX, W):
    CO = W['G']['T']['continuations.csv']
    O = CO[(CO['table'] == 'order').to_numpy() & (CO['cand_hash'] == '0xhR5').to_numpy()]
    if len(O) != 1:
        return False, dict(order_rows=int(len(O)))
    r = O.iloc[0]
    exp = dict(n_legs=3, n_records=3, filled_shares=30., v10s_n_valid=3, v10s_gross=0.3, v60s_c_sh=2., vend_n_valid=2, vend_sh_valid=20., common_n=2,
               common_sh=20., common_c_sh_10s=1., common_c_sh_60s=2., common_c_sh_end=3., turnover_mq_sh=25., residual_rq_sh=5., takes_10s=1,
               takes_60s=2, takes_end=2, n_responses_linked=1, fv_rows=3)
    bad = {k: (r[k], v) for k, v in exp.items() if not _eqv(r[k], v)}
    ok = not bad and r['next_kind'] == 'reprice' and r['successor_status'] == 'evaluated' and r['takes_status'] == 'evaluated' and \
        r['takes_end_dir_D0'] == 'opposing:1;same:1' and r['union_histories'] == 'EX|PL'
    return bool(ok), dict(bad=bad, next_kind=r['next_kind'], dirs=r['takes_end_dir_D0'])


def chk_G6(FX, W):
    ix, ws = FX['ix'], FX['G'].ws
    got = [_pair(W, ix[x]) for x in (30500, 30700, 30900)]
    ok_a = got == [('unknown', 'unknown_touch')] * 3
    E = W['G']['EPD']
    eb, ec = _ep(E, ix[31300], 'bid'), _ep(E, ix[35300], 'ask')
    ok_b = eb is not None and eb['end_type'] == 'censored_gap' and eb['end_reason'] == 'unknown_touch' and int(eb['exposure_ms']) == 200 and \
        int(eb['t_end_ms']) == ws + 31500
    ok_c = ec is not None and ec['end_type'] == 'censored_gap' and ec['end_reason'] == 'gap' and int(ec['exposure_ms']) == 200 and \
        int(ec['t_end_ms']) == ws + 35500
    CEN = W['G']['T']['census.csv']
    pc = CEN[(CEN['scope'] == 'pooled').to_numpy()]
    n_gap = int(pc[(pc['cls'] == 'gap').to_numpy() & (pc['sublabel'] == 'all').to_numpy()]['n_pairs'].iloc[0])
    n_unk = int(pc[(pc['cls'] == 'unknown').to_numpy() & (pc['sublabel'] == 'unknown_touch').to_numpy()]['n_pairs'].iloc[0])
    return bool(ok_a and ok_b and ok_c and n_gap >= 1 and n_unk >= 4), dict(classes=got, b=None if eb is None else (eb['end_type'], eb['end_reason']),
                                                                             c=None if ec is None else (ec['end_type'], ec['end_reason']), gap=n_gap, unk=n_unk)


def chk_G7(FX, W):
    A, B = TT.episodes_frame([W['P']]), TT.episodes_frame([W['P2']])
    cut = FX['P'].ws + T7 - 2000
    ea = A[(A['t_start_ms'] <= cut).to_numpy()].reset_index(drop=True)
    eb = B[(B['t_start_ms'] <= cut).to_numpy()].reset_index(drop=True)
    same = len(ea) >= 2 and len(ea) == len(eb) and all(all(_eqv(x, y) if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
                                                            else x == y for x, y in zip(ea[c].tolist(), eb[c].tolist())) for c in ea.columns)
    la = A[(A['t_start_ms'] > FX['P'].ws + T7).to_numpy()].reset_index(drop=True)
    lb = B[(B['t_start_ms'] > FX['P'].ws + T7).to_numpy()].reset_index(drop=True)
    bites = len(la) >= 1 and len(lb) >= 1 and not la[['t_start_ms', 't_end_ms']].equals(lb[['t_start_ms', 't_end_ms']])
    nr = ea[(ea['end_type'] == 'no_response_1s').to_numpy()]
    return bool(same and bites and len(nr) == 1 and int(nr['exposure_ms'].iloc[0]) == 1000), dict(early=len(ea), early_record=len(eb), later=len(la),
                                                                                                  bites=bites, early_end=ea['end_type'].tolist())


def chk_G8(FX, W):
    ix, ws = FX['ix'], FX['G'].ws
    E, E8 = W['G']['EPD'], TT.episodes_frame([W['G8']])
    e, e8 = _ep(E, ix[10300], 'ask'), _ep(E8, ix[10300], 'ask')
    if e is None or e8 is None:
        return False, 'episode missing'
    ok_v = int(e['state_frame_D0']) == ix[10100] and int(e['state_ms_D0']) == ws + 10100 and _eqv(e['spread_D0'], 1.) and _eqv(e['depth_own_D0'], 100.) \
        and int(e['state_frame_LAG']) == ix[10100] - 1 and e['state_status_LAG'] == 'ok' and e['btc_5s_aligned_D0'] == 'flat'
    cols = [c for c in E.columns if c.endswith('_D0') or c.endswith('_LAG')]
    diff = [c for c in cols if not (_eqv(e[c], e8[c]) if isinstance(e[c], (int, float, np.integer, np.floating)) and not isinstance(e[c], bool) else e[c] == e8[c])]
    return bool(ok_v and not diff), dict(frame=int(e['state_frame_D0']), spread=e['spread_D0'], depth=e['depth_own_D0'], diff=diff[:6])


CHECKS = dict(G1=chk_G1, G2=chk_G2, G3=chk_G3, G4=chk_G4, G5=chk_G5, G6=chk_G6, G7=chk_G7, G8=chk_G8)


def run_gate(FX, W):
    for k, fn in CHECKS.items():
        try:
            ok, info = fn(FX, W)
        except Exception as e:  # noqa: BLE001
            ok, info = False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-900:]}'
        check(f'gate {k} (base): passes', ok, info)


def test_mutants(FX):
    for m_, k in EXPECT_CATCH.items():
        os.environ[TT.MUT_ENV] = m_
        try:
            Wm = compute(FX)
            ok, info = CHECKS[k](FX, Wm)
            check(f'gate {k}: MT_TT_MUT={m_} is caught by check {k}', not ok, info)
        except Exception as e:  # noqa: BLE001  (a crash is not a catch by a check)
            check(f'gate {k}: MT_TT_MUT={m_} is caught by check {k}', False, f'crashed instead: {type(e).__name__}: {e} | {traceback.format_exc()[-600:]}')
        finally:
            os.environ.pop(TT.MUT_ENV, None)


# ------------------------------------------------------------------ base-run predicates ------------------------------------------------------------------
def test_base(FX, W):
    G = W['G']
    cl, PR, E = G['claims'], G['part']['pairs'], G['EPD']
    CEN = G['T']['census.csv']
    pc = CEN[(CEN['scope'] == 'pooled').to_numpy() & (CEN['sublabel'] == 'all').to_numpy()]
    check('census: every frame pair in exactly one class (pooled class rows sum to the pairs; the market rows equal the pooled ones)',
          int(pc['n_pairs'].sum()) == len(PR) == cl['census']['pairs'] and
          int(CEN[(CEN['scope'] == 'market').to_numpy() & (CEN['sublabel'] == 'all').to_numpy()]['n_pairs'].sum()) == len(PR),
          dict(pairs=len(PR), census=int(pc['n_pairs'].sum())))
    nE = int(len(E))
    check('episodes: 7 retreat episodes (5 responses, 2 censored_gap), one per retreat pair in the window',
          nE == 7 and cl['episodes']['termination']['response']['n'] == 5 and cl['episodes']['termination']['censored_gap']['n'] == 2
          and nE == int(pc['n_episode_starts'].sum()) and nE == int(pc.loc[pc['cls'].isin(['retreat_bid', 'retreat_ask']).to_numpy(), 'n_pairs'].sum()),
          dict(n=nE, term={k: v['n'] for k, v in cl['episodes']['termination'].items()}))
    check('episodes: termination / bound predicates all true; every response typed restoring here',
          cl['episodes']['predicates']['all'] is True and cl['episodes']['response_mix']['all'] == dict(restoring=5, partial=0, overshoot=0),
          cl['episodes']['predicates'])
    rec = cl['reconciliation']
    RC = G['T']['reconcile.csv']
    imp = RC[(RC['po_status'] == 'event').to_numpy() & RC['is_improve'].to_numpy(bool)]
    check('reconciliation: predicates all true; every improve event row has a frame pair with a class; the anchor-excluded ones counted',
          rec['predicates']['all'] is True and rec['improve_events'] == len(imp) and len(imp) >= 7 and rec['anchor_excluded'] >= 5
          and all(str(x).startswith('class:') for x in imp['recon_status']), dict(rec=rec['predicates'], n=len(imp), by=rec['improve_by_status']))
    roles = Counter(imp['episode_role'].tolist())
    check('reconciliation: five improve creations are the responses\' new best cells; the two_side and post-censoring refills are not responses',
          roles.get('response_new_best', 0) == 5 and roles.get('not_a_response', 0) >= 2, dict(roles))
    R = E[(E['end_type'] == 'response').to_numpy()]
    check('owners: every response graded; the planted four linked (G4 x2 scenario + G1 scenario + G2 constrained), one owner_unknown in the '
          'denominator; the grading reproduces the saved posting_opportunity identity of every response cell',
          Counter(R['identity'].tolist()) == Counter(dict(candidate_scenario=3, candidate_constrained=1, owner_unknown=1))
          and set(R['po_parity']) == {'equal'}, dict(ids=Counter(R['identity'].tolist()), parity=Counter(R['po_parity'].tolist())))
    RT = G['T']['ratios.csv']
    r0 = RT[(RT['variant'] == 'D0').to_numpy() & (RT['contrast'] == 'time_left').to_numpy()].iloc[0]
    tl = E['time_left_bin_D0'].astype(str)
    ex = E['state_status_D0'].eq('ok') & tl.isin(['(10,30]', '[0,10]'])
    rf = E['state_status_D0'].eq('ok') & tl.isin(['(120,240]', '(60,120]'])
    check('ratios: 2 variants x 5 contrasts with declared verdicts; D0 time_left person-time and responses recounted from episodes.parquet',
          len(RT) == 10 and set(RT['verdict']) <= set(TT.VERDICTS) and _eqv(r0['T_exposed_s'], E.loc[ex.to_numpy(), 'exposure_ms'].sum() / 1000.)
          and _eqv(r0['T_reference_s'], E.loc[rf.to_numpy(), 'exposure_ms'].sum() / 1000.)
          and _eqv(r0['a_exposed'], int((ex & E['end_type'].eq('response')).sum())), dict(T1=r0['T_exposed_s'], T0=r0['T_reference_s']))
    TS = G['T']['two_side.csv']
    check('two_side.csv: separate descriptive table (no response column); its improve creations are the four two_side refills',
          int(TS[(TS['scope'] == 'pooled').to_numpy() & (TS['sublabel'] == 'all').to_numpy() & (TS['feature'] == 'total').to_numpy()]['n_cre_improve'].iloc[0]) == 4
          and 'end_type' not in TS.columns, TS.head(3).to_dict('records'))
    lk = R[R['identity'].isin(TT.LINKED).to_numpy()]
    CO = G['T']['continuations.csv']
    O = CO[(CO['table'] == 'order').to_numpy()]
    check('continuations: one row per distinct candidate order (4 linked responses -> 4 orders), keyed by hash; non-candidate legs and the taker '
          'row stay out of the views', len(O) == len(lk) == 4 and set(O['cand_hash']) == set(PLANT_H()) and int(O['fv_rows'].sum()) == 6,
          dict(orders=len(O), fv=int(O['fv_rows'].sum()) if len(O) else None))
    su = CO[(CO['table'] == 'summary').to_numpy() & (CO['scope'] == 'all').to_numpy()]
    met = {str(x['metric']): x['value'] for x in su.to_dict('records')}
    want = {'takes_end_distinct': 2, 'takes_end_order_sum': 4, 'takes_end_overlap': 2, 'takes_60s_distinct': 2, 'takes_60s_order_sum': 4,
            'takes_60s_overlap': 2, 'takes_10s_distinct': 1, 'takes_10s_order_sum': 1, 'takes_10s_overlap': 0}
    got = {k: met.get(k) for k in want}
    check('continuations: rank 1\'s two linked orders share the same later takes - the pooled tables count each take record once (distinct 2 of '
          'order-sum 4: overlap 2), never once per candidate creation', all(_eqv(got[k], v) for k, v in want.items()), got)
    term = cl['episodes']['termination']
    check('termination census by the class of the ending pair: responses improve_ask 2 / improve_bid 3; censored_gap by an unknown pair 1 and a '
          'straddling gap (no ending pair) 1', term['response']['by_end_class'] == dict(improve_ask=2, improve_bid=3)
          and term['censored_gap']['by_end_class'] == dict(unknown=1, none=1), {k: v['by_end_class'] for k, v in term.items()})


def PLANT_H():
    return [v['h'] for v in PLANT.values()]


# ------------------------------------------------------------------ the ratio machinery ------------------------------------------------------------------
def _E(n, **kw):
    E = dict(mi=np.zeros(n, np.int64), side=np.zeros(n, np.int64), dbin=np.zeros(n, np.int64), resp=np.zeros(n), expo_ms=np.full(n, 1000.))
    for v in TT.VARIANTS:
        E.update({f'st_{v}': np.zeros(n, np.int64), f'btc_c_{v}': np.full(n, 3, np.int64), f'fg_c_{v}': np.full(n, 4, np.int64),
                  f'd_c_{v}': np.full(n, 4, np.int64), f'tl_c_{v}': np.zeros(n, np.int64)})
    for k, v in kw.items():
        E[k] = np.asarray(v)
        if k.endswith('_D0'):
            E[k[:-3] + '_LAG'] = np.asarray(v).copy()
    return E


def test_mh():
    E = _E(7, side=[0, 0, 0, 0, 1, 1, 1], resp=[1., 1., 1., 0., 1., 0., 1.], expo_ms=[500., 500., 1000., 1000., 1000., 1000., 1000.],
           btc_c_D0=[0, 0, 2, 2, 0, 0, 2])
    r, _ = TT.ratio_block(E, 0, 'D0', 0, 'btc_with', 'x', 'y', [(DAY, 1)], [DAY])
    check('MH toy: hand-computed person-time MH 5/3 and crude 1.5 (equal to take_logic.mh_rr inside the block)',
          abs(r['RR_MH'] - 5. / 3.) < 1e-12 and abs(r['RR_crude'] - 1.5) < 1e-12 and r['a_exposed'] == 3 and _eqv(r['T_exposed_s'], 3.)
          and r['verdict_variant'] == 'insufficient_support', dict(rr=r['RR_MH'], crude=r['RR_crude']))


def test_perm():
    rng0 = np.random.default_rng(5)
    st = rng0.integers(0, 6, 400).astype(np.int64)
    g1 = rng0.random(400) < 0.4
    L1 = TT.perm_labels(st, g1, np.random.Generator(np.random.PCG64(np.random.SeedSequence([TT.SEED, 0, 0]))), 50)
    L2 = TT.perm_labels(st, g1, np.random.Generator(np.random.PCG64(np.random.SeedSequence([TT.SEED, 0, 0]))), 50)
    n1 = np.bincount(st[g1], minlength=6)
    kept = all(np.array_equal(np.bincount(st[L1[r]], minlength=6), n1) for r in range(50))
    moved = float((L1 != g1[None, :]).mean())
    check('permutation: per-stratum exposed counts kept in every draw, labels move, the same seed gives the same draws',
          kept and moved > 0.2 and np.array_equal(L1, L2), dict(kept=kept, moved=moved))
    L3 = TT.perm_labels(st, g1, np.random.Generator(np.random.PCG64(np.random.SeedSequence([TT.SEED, 0, 1]))), 50)
    check('permutation: another contrast index gives another stream', not np.array_equal(L1, L3))


def planted_world(seed=41, n_mk=6, n_side=200):
    rng = np.random.default_rng(seed)
    n = n_mk * 2 * n_side
    mi = np.repeat(np.arange(n_mk), 2 * n_side)
    side = np.tile(np.repeat([0, 1], n_side), n_mk)
    btc = rng.choice([0, 1, 2], n)
    lam = np.where(btc == 0, 3., 1.)
    tau = rng.exponential(1. / lam)
    resp = (tau < 1.).astype(float)
    expo = np.rint(np.minimum(tau, 1.) * 1000.)
    return _E(n, mi=mi, side=side, dbin=rng.integers(0, 3, n), resp=resp, expo_ms=expo, btc_c_D0=btc, fg_c_D0=rng.choice([0, 1, 2, 3], n),
              d_c_D0=rng.choice([1, 2, 3], n), tl_c_D0=rng.choice([2, 3, 4, 5, 6], n))


def twin_world(seed=43, n_mk=6, n_side=100):
    rng = np.random.default_rng(seed)
    n = n_mk * 2 * n_side
    base = dict(mi=np.repeat(np.arange(n_mk), 2 * n_side), side=np.tile(np.repeat([0, 1], n_side), n_mk), dbin=rng.integers(0, 3, n),
                resp=(rng.random(n) < 0.5).astype(float), expo_ms=rng.integers(100, 1001, n).astype(float), btc=rng.choice([0, 2], n),
                fg=rng.choice([0, 2], n), d=rng.choice([1, 3], n), tl=rng.choice([2, 3, 4, 5, 6], n))
    twin = dict(base, btc=2 - base['btc'], fg=2 - base['fg'], d=4 - base['d'])
    cat = {k: np.concatenate([base[k], twin[k]]) for k in base}
    return _E(2 * n, mi=cat['mi'], side=cat['side'], dbin=cat['dbin'], resp=cat['resp'], expo_ms=cat['expo_ms'], btc_c_D0=cat['btc'], fg_c_D0=cat['fg'],
              d_c_D0=cat['d'], tl_c_D0=cat['tl'])


def test_worlds():
    mk = [(DAY, 990000 + i) for i in range(6)]
    E = planted_world()
    r, _ = TT.ratio_block(E, 0, 'D0', 0, 'btc_with', 'x', 'y', mk, [DAY])
    check('planted hazard world: btc with (hazard x3) -> supported (RR > 2, p_hi <= 0.05, >= 5 supported markets, >= 2/3 above 1)',
          r['verdict_variant'] == 'supported' and r['RR_MH'] > 2. and r['p_hi'] <= 0.05 and r['n_markets_supported'] >= 5,
          {k: r[k] for k in ('RR_MH', 'p_hi', 'n_markets_supported', 'n_markets_rr_gt1', 'n_events_informative', 'verdict_variant')})
    T2 = twin_world()
    got = {}
    for ci, cn in ((0, 'btc_with'), (2, 'fair_gap'), (3, 'depth_own')):
        r, _ = TT.ratio_block(T2, 0, 'D0', ci, cn, 'x', 'y', mk, [DAY])
        got[cn] = (r['RR_MH'], r['n_markets_supported'], r['n_events_informative'], r['verdict_variant'], r['n_markets_rr_gt1'], r['n_markets_rr_lt1'])
    check('twin world: RR exactly 1 for btc_with / fair_gap / depth_own while testable (>= 5 supported markets, >= 30 responses); no verdict fires',
          all(abs(v[0] - 1.) < 1e-12 and v[1] >= 5 and v[2] >= 30 and v[3] not in ('supported', 'supported_lower') and v[4] == 0 and v[5] == 0
              for v in got.values()), got)
    RK, cons, FIN = TT.ratio_tables(E, mk, [DAY])
    check('ratio tables: final verdict = D0 unless LAG differs (identical states here -> no timing_sensitive); 10 blocks, consistency rows per market',
          len(RK) == 10 and all(FIN[cn] == RK[('D0', cn)]['verdict_variant'] for cn in TT.CN_NAMES)
          and sum(1 for c in cons if c['row'] == 'market') == 10 * len(mk), FIN)


# ------------------------------------------------------------------ readers, e2e, refusals ------------------------------------------------------------------
def test_readers(tmp, FX):
    D, fsha, mk = FX['dirs'], FX['fsha'], [(DAY, FX['G'].mid)]
    bad = tmp / 'po_bad'
    shutil.copytree(D['po'], bad)
    (bad / 'events.csv').write_text((bad / 'events.csv').read_text() + '\n')
    expect('reader: a posting_opportunity events.csv unlike its receipt -> refused', lambda: TT.read_po(bad, mk, {}, fsha), "differs from its receipt")
    expect('reader: posting_opportunity bound to another fills_valued -> refused', lambda: TT.read_po(D['po'], mk, {}, '0' * 64), 'another fills_valued')
    expect('reader: a queue_front_v2 levels.csv other than the one posting_opportunity read -> refused',
           lambda: TT.read_qf2(D['qf2'], mk, {}, fsha, '1' * 64), 'not the one posting_opportunity')
    q2 = tmp / 'qf2_bad'
    shutil.copytree(D['qf2'], q2)
    pd.read_csv(q2 / 'candidates.csv').drop(columns=['alt_first_addresses']).to_csv(q2 / 'candidates.csv', index=False)
    J = json.loads((q2 / 'receipt.json').read_text())
    J['outputs']['candidates.csv']['sha256'] = TT._sha(q2 / 'candidates.csv')
    (q2 / 'receipt.json').write_text(json.dumps(J))
    expect('reader: a candidates.csv without the alternatives column -> refused (interface)',
           lambda: TT.read_qf2(q2, mk, {}, fsha, FX['poi']['levels_sha256']), 'lacks')
    pf2 = tmp / 'pf_bad'
    shutil.copytree(D['post_fill'], pf2)
    J = json.loads((pf2 / 'receipt.json').read_text())
    J['rows']['orders'] = 99
    (pf2 / 'receipt.json').write_text(json.dumps(J))
    expect('reader: a post_fill orders.csv row count unlike its receipt -> refused', lambda: TT.read_post_fill(pf2, mk, {}, fsha), 'rows differ')


def test_e2e(tmp, FX, W):
    out = tmp / 'e2e_pilot'
    built = {(DAY, FX['G'].mid): FX['MG']}
    kw = dict(inp=FX['root'], loader=lambda d, mid, r, hc: built[(d, mid)], book={DAY: [FX['G'].mid]}, dirs=FX['dirs'], fv_sha=FX['fsha'],
              log=lambda s: None)
    rec = TT.execute('pilot', (DAY,), out, **kw)
    names = sorted(p.name for p in out.iterdir())
    check('e2e: execute (pilot path, injected loader / inputs) writes exactly the nine outputs', names == sorted(TT.OUT_FILES), names)
    Rj = json.loads((out / 'receipt.json').read_text())
    pj = Rj.get('projection') or {}
    check('e2e: receipt hashes every other output, the sources and the inputs; no mutant; a finite projection; pilot kind',
          all(Rj['outputs'][f]['sha256'] == TT._sha(out / f) for f in TT.OUT_FILES if f != 'receipt.json') and Rj['mutant'] == ''
          and len(Rj['sources']) >= 15 and Rj['inputs'].get('econ/fills_valued.parquet') == FX['fsha'] and isinstance(pj.get('full_run_cpu_s'), float)
          and rec["kind"] == "pilot" and len(Rj["inputs"]) >= 13, dict(proj=pj, n_src=len(Rj['sources']), n_in=len(Rj['inputs'])))
    EP = pd.read_parquet(out / 'episodes.parquet')
    Cl = json.loads((out / 'claims.json').read_text())
    RC = pd.read_csv(out / 'reconcile.csv', dtype={'day': str}, keep_default_na=False)
    check('e2e: the written episodes / reconcile rows equal the in-process gate analysis; claims carry the computed predicates',
          len(EP) == len(W['G']['EPD']) and EP['end_type'].tolist() == W['G']['EPD']['end_type'].tolist() and len(RC) == len(W['G']['T']['reconcile.csv'])
          and Cl['reconciliation']['predicates']['all'] is True and Cl['episodes']['predicates']['all'] is True and Cl['descriptive_only'] is True
          and Cl['intervals'] is None and set(Cl['owners']['po_parity']) == {'equal'}, dict(ep=len(EP), rc=len(RC)))
    expect('e2e: a second write into the written directory is refused (before any input is read)', lambda: TT.execute('pilot', (DAY,), out, **kw),
           'write-once')
    expect('e2e: a fills_valued unlike the binding -> refused', lambda: TT.execute('pilot', (DAY,), tmp / 'e2e_badfv', **dict(kw, fv_sha='0' * 64)),
           'another fills_valued')


def test_refusals(tmp):
    d = tmp / 'exists'
    d.mkdir()
    expect('refuse: write-once on an existing output directory', lambda: TT.TL2.write_out(d, {}, {}, {}, lambda o: {}), 'write-once')
    for ev_, m_, needle in (('MT_TT_MUT', 'state_leak', 'MT_TT_MUT'), ('MT_TT_MUT', 'bogus', 'not a declared mutant'),
                            ('MT_PO_MUT', 'leak_1s', 'MT_PO_MUT'), ('MT_QF2_MUT', 'drop_no_fill', 'MT_QF2_MUT')):
        os.environ[ev_] = m_
        try:
            expect(f'refuse: production with {ev_}={m_} before any input is read', lambda: TT.execute('pilot', (TT.PILOT_DAY,), tmp / 'never'), needle)
        finally:
            os.environ.pop(ev_, None)
    expect('refuse: a pilot day outside the measurement days', lambda: TT.execute('pilot', (C.ID_DAY,), tmp / 'never'), 'measurement days')
    check('refuse: nothing written by a refused production call', not (tmp / 'never').exists())
    g = TT._git([f for f in TT.TOOL_FILES if Path(f).exists()])
    needle = 'freeze' if not g.get('unmodified') else 'no recorded pilot projection'
    expect(f'refuse: the full run before any input ({needle})', lambda: TT.execute('run', C.MEASURE_DAYS, tmp / 'run_out', pilot_receipt=tmp / 'absent.json'),
           needle)
    for nm, J, needle in (('p121.json', dict(kind='pilot', version=TT.VERSION, projection=dict(full_run_cpu_s=121.0)), '> 120 s'),
                          ('pother.json', dict(kind='pilot', version='posting_opportunity_v1', projection=dict(full_run_cpu_s=10.0)), 'not a touch_transition_v1')):
        (tmp / nm).write_text(json.dumps(J))
        expect(f'refuse: a recorded pilot receipt {nm}', lambda p=tmp / nm: TT.preflight_projection(p, {}), needle)
    (tmp / 'p40.json').write_text(json.dumps(dict(kind='pilot', version=TT.VERSION, projection=dict(full_run_cpu_s=40.0))))
    h = {}
    r = TT.preflight_projection(tmp / 'p40.json', h)
    check('preflight: a 40 s projection passes and the pilot receipt is hashed', r['full_run_cpu_s'] == 40.0 and len(h) == 1)


def main():
    t0 = time.process_time()
    check('selftest environment: MT_RUN=v2, no planted error set', C.RUN == 'v2' and all(e not in os.environ for e in (TT.MUT_ENV,) + TT.FOREIGN_MUT), C.RUN)
    tmp = Path(tempfile.mkdtemp(prefix='touch_transition_selftest_'))
    try:
        section('mh', test_mh)
        section('perm', test_perm)
        section('worlds', test_worlds)
        FX = section('fixtures', fixtures, tmp)
        if FX:
            W = section('compute', compute, FX)
            if W:
                run_gate(FX, W)
                section('base', test_base, FX, W)
                section('mutants', test_mutants, FX)
                section('readers', test_readers, tmp, FX)
                section('e2e', test_e2e, tmp, FX, W)
        section('refusals', test_refusals, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop(TT.MUT_ENV, None)
    fails = [r for r in RES if not r[1]]
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  | {info}'))
    print(json.dumps(dict(checks=len(RES), failed=len(fails), cpu_s=round(time.process_time() - t0, 3), process_cpu_s=round(time.process_time(), 3),
                          vmhwm_mib=round(QF.hwm(), 1))))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
