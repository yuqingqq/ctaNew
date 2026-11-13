"""LEARN-PULL (2026-11-13, session ce68ebe0). USER: "We learn from them first, check how do they avoid being hit, and why they keep quoting in
some big moves, do they always quote in big moves? Can you tell why they pull, we pull in sometimes they pull as well, and we pull much more
often than theirs, can you check the difference so that we can learn from them".
After MOVE-CASES (b9c12978): the top makers take adverse selection on the side the price moves into and are paid by the other side; EXIT
(11fb2adc) showed their front order at the best is gone before a level-taking trade 28% -> 80% from low to high values of our risk model.

Public tape only for them (real maker legs, real book frames), 9 consumed replay days, first 60 s, WS-timed taker orders.
PART A, their pulls against ours, at the best:
  presence    maker x is the only maker filled at p by a taker order that STOPPED at p and the next frame still shows p (EXIT's sole-maker
              stop): x's order is at the front of p with size left at t1
  next trade  the next WS-timed taker order with a leg at p on that side, at t2 in (t1, t1 + 3 s]; STAYED if x has a leg at p in it (at the
              front it is hit first under price-time priority), GONE if not (x's order left p in (t1, t2): cancelled or moved). Ambiguous
              (status): an E2-timed leg of x at p in [t1 - 250, t2 + 250] ms
  next trade's character: STOP (p was its last price) or THROUGH (it went past p); its volume at p; the value of a fill at p at t2 for a
              resting order, c/sh at 1 s and 5 s
  ours        our guard (v <= -0.10) at any 50 ms step in (t1, t2 - 22 ms] (a pull decided then lands before the trade): WE PULL
  features    at the last step <= t2 - 100 ms: our guard value v, FRONT-PULL's P, the 37 S4 inputs; side-adjusted 'against the order'
              versions of the Binance return / flow / imbalance and the Predict mid change / imbalance; the shown size at p; the time
              since t1; x's YES position in the window from all its fills (maker and taker) before t1, signed so that > 0 means this order
              would ADD to the position (inventory)
PART B, big moves (MOVE-CASES' 'big' events, M >= 3 ticks within 2 s, on the side the price moves into): per event whether any top-5 maker
  was filled at the prices run over; the top-5 share of the run-over volume by level index (1, 2, 3+) and by the move's speed (time to the
  peak <= 250 / 250-1000 / > 1000 ms); each run-over fill's inventory direction for its maker (reducing / flat / increasing).
Groups: top 5 by daily maker volume vs the field.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule >= 7 of 9; < 30 observations in a compared cell: the day does not count):
 R550 THEY-AVOID: top-5 GONE share before THROUGH next trades / before STOP next trades >= 1.2 pooled and higher on >= 7 days ->
      AVOID-ADVERSE; else NOT-SHOWN. Our pull share in the same split and the mean next-trade value for STAYED vs GONE printed.
 R551 PULL RATE (descriptive): our pull share against the top-5 GONE share on the same intervals, by interval (0-250 / 250-1000 /
      1000-3000 ms).
 R552 OUR EXTRA PULLS: intervals where we pull and the top 5 stay: the next trade's value at 5 s (volume-weighted) > 0 on >= 7 days ->
      OUR-EXTRA-PULLS-COST (we give up good fills); < 0 on >= 7 -> OUR-EXTRA-PULLS-SAVE; else NOT-SHOWN.
 R553 THEIR EXTRA PULLS: intervals where the top 5 pull and we do not: the next trade's value at 5 s < 0 on >= 7 days ->
      THEY-SEE-WHAT-WE-MISS; > 0 on >= 7 -> THEY-PULL-NEEDLESSLY; else NOT-SHOWN.
 R554 WHY (descriptive + model): GONE share by quintile of each named feature (top 5, field) beside our pull share; LightGBM, leave one day
      out, AUC of top-5 GONE from all features, from v alone, from inventory alone (parameters pinned below).
 R555 INVENTORY: top-5 GONE share for orders that would add >= 10 sh to their position / that would reduce it by >= 10 sh: >= 1.2 pooled
      and higher on >= 7 days -> INVENTORY-DRIVEN; else NOT-SHOWN. Field printed.
 R556 BIG MOVES (descriptive): the PART B tables.
What the outcomes mean, fixed in advance: AVOID-ADVERSE = they leave the front before trades that go through the price more than before
trades that stop there; OUR-EXTRA-PULLS-COST = where we pull and they do not, the fill we gave up was worth having; THEY-SEE-WHAT-WE-MISS =
where they pull and we do not, the fill we kept was bad; INVENTORY-DRIVEN = they pull orders that would grow their position more.
Controls (any failure -> no output): (i) synthetic cases for the presence / next-trade outcome, the 3 s limit, the E2 ambiguity, our pull
window, the position signs (maker bid +, maker ask -, taker buy +, taker sell -) and the inventory direction, with three planted errors (the
outcome inverted, the taker position sign flipped, our pull window running to the trade itself) that must each fail a case; (ii) no
observation uses a record at or after t2 (asserted); (iii) PART B's events equal PULL-DEPTH's before the exclusion; (iv) memory self-guard.
Limits fixed in advance: fills only (a pull is inferred from a front order not hit; a maker with several orders at p needs all gone;
matching assumed price-time; B1a found a fifth of live fills contradicting the shown-size queue); the presence proof selects front orders
after small stopping trades; positions are within the window only (start of the window = flat); our guard and P are our signals; consumed
days (C8); P carries the nested-fold leak noted by the other session's review.
Usage (run_light, PF_SLOT=claude): pf_learn_pull.py selftest | day <day> | report; driver run_learn_pull.sh
"""
import bisect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_qr_queue_race as QR                             # noqa: E402
import pf_pull_depth as PD                                # noqa: E402
import pf_robust_check as RC                              # noqa: E402
import pf_move_cases as MC                                # noqa: E402
import pf_exit_check as EX                                # noqa: E402  (classify)

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/learn_pull_20261113'
FPDIR = f'{K.IO}/front_pull_20261113'
W_MAX = 3000.0; E2_TOL = 250.0; L_CANCEL = 22.0; DEC_LAG = 100.0; INV_MIN = 10.0
GAPB = (('0-250 ms', 0.0, 250.0), ('250-1000 ms', 250.0, 1000.0), ('1000-3000 ms', 1000.0, 3000.0))
LGB_PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, seed=1, num_threads=1, verbose=-1)
N_ROUNDS = 200; SUB_N = 300000; MIN_N = 30
XN = list(K.G32) + ['v', 'own_d250', 'opp_d250', 'act_rel', 'd_bnimb250']
NAMED = ['v', 'P', 'adv_bn_r250', 'adv_bn_r1000', 'adv_bn_flow250', 'adv_bn_imb', 'adv_pf_dm250', 'adv_pf_imb1', 'own_d250', 'opp_d250',
         'pf_spread', 'size_p', 'gap', 'inv', 't_rel', 'sigma', 'act_rel']
INF = float('inf'); NAN = float('nan')
MUT_OUTCOME_INV = MUT_TAKER_SIGN = MUT_PULL_LATE = 0      # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


def next_trade_obs(side, orders, present_after, e2_legs):
    """orders: [(t, legs [(signer, p, q)])] time-sorted WS taker orders of one window-side. Returns observations: presence of x at p at t1
    (sole-maker stop with p still shown after) and the next taker order with a leg at p at t2 <= t1 + 3 s: STAYED / GONE."""
    obs = []; amb = 0; open_ = {}                         # p -> (signer, t1) of the latest presence at p not yet resolved
    for t, legs in orders:
        ps = {p for _, p, _ in legs}
        for p in list(ps):
            if p in open_:
                s, t1 = open_.pop(p)
                if t1 >= t:
                    raise ValueError('LEARN-PULL: an observation uses a record at or after t2')
                if t - t1 > W_MAX:
                    continue
                if any(t1 - E2_TOL <= x <= t + E2_TOL for x in e2_legs.get((s, p), ())):
                    amb += 1; continue
                inE = {s_ for s_, q, _ in legs if q == p}
                stayed = int(s in inE)
                if MUT_OUTCOME_INV:
                    stayed = 1 - stayed
                taken, stop, _ = EX.classify(side, legs)
                obs.append(dict(signer=s, p=p, t1=t1, t2=t, gap=t - t1, stayed=stayed, through=int(p in taken),
                                vol_p=float(sum(q_ for _, q, q_ in legs if q == p))))
        taken, stop, at_stop = EX.classify(side, legs)
        if len(at_stop) == 1 and present_after(stop, t):
            open_[stop] = (next(iter(at_stop)), t)
    return obs, amb


def positions(events):
    """events: [(t, signer, dq)] -> {signer: (times sorted, cumulative position after each)}."""
    by = {}
    for t, s, dq in sorted(events, key=lambda e: e[0]):
        by.setdefault(s, ([], []))
        ts, cs = by[s]; ts.append(t); cs.append((cs[-1] if cs else 0.0) + dq)
    return by


def pos_before(by, s, t):
    if s not in by:
        return 0.0
    ts, cs = by[s]; i = bisect.bisect_left(ts, t) - 1
    return cs[i] if i >= 0 else 0.0


def leg_events(side, signer, taker_sig, q, t):
    """Position changes of one leg: maker bid +q, maker ask -q; taker the opposite (taker sold to a bid -q, bought from an ask +q)."""
    mk = q if side == 0 else -q
    tk = -mk if not MUT_TAKER_SIGN else mk
    out = [(t, signer, mk)]
    if taker_sig is not None:
        out.append((t, taker_sig, tk))
    return out


def inv_dir(pos, side):
    """> 0: an order on this side would add to the position (a bid when long, an ask when short)."""
    return pos if side == 0 else -pos


def we_pull(tg, vg, t1, t2):
    lo = bisect.bisect_right(tg, t1); hi = bisect.bisect_right(tg, t2 - (0.0 if MUT_PULL_LATE else L_CANCEL))
    return any(vg[lo:hi])


def selftest():
    bad = []
    pa = lambda p, t: True
    # maker 7 sole stop at 50 at t = 100; next trade at 50 at t = 300 fills 9 only -> GONE (7 left); then 9's stop at 50 (t = 300) and the
    # next trade at 50 at t = 3500 is beyond 3 s -> no observation; maker 4 sole stop at 51 at 400, next trade at 51 at 600 fills 4 -> STAYED
    orders = [(100.0, [(7, 50, 5.0)]), (300.0, [(9, 50, 3.0)]), (400.0, [(4, 51, 2.0)]), (600.0, [(4, 51, 6.0), (8, 52, 1.0)]),
              (3500.0, [(9, 50, 2.0)])]
    obs, amb = next_trade_obs(1, orders, pa, {})
    got = [(o['signer'], o['p'], o['stayed'], o['through']) for o in obs]
    if got != [(7, 50, 0, 0), (4, 51, 1, 1)]:
        bad.append(f'next-trade outcomes {got}')
    obs2, amb2 = next_trade_obs(1, orders, pa, {(7, 50): [250.0]})
    if amb2 != 1 or [(o['signer'], o['stayed']) for o in obs2] != [(4, 1)]:
        bad.append('E2 ambiguity')
    ev = leg_events(0, 7, 3, 10.0, 1.0) + leg_events(1, 7, 3, 4.0, 2.0)          # 7 bought 10 on its bid, sold 4 on its ask; 3 the taker
    by = positions(ev)
    if pos_before(by, 7, 5.0) != 6.0 or pos_before(by, 3, 5.0) != -6.0 or pos_before(by, 7, 1.5) != 10.0:
        bad.append(f'positions {pos_before(by, 7, 5.0)}, {pos_before(by, 3, 5.0)}')
    if inv_dir(6.0, 0) != 6.0 or inv_dir(6.0, 1) != -6.0:
        bad.append('inventory direction')
    tg = [0.0, 50.0, 100.0, 150.0]; vg = [False, False, False, True]
    if we_pull(tg, vg, 10.0, 160.0) is not False or we_pull(tg, vg, 10.0, 180.0) is not True:
        bad.append('our pull window')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_OUTCOME_INV', 'MUT_TAKER_SIGN', 'MUT_PULL_LATE'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    MC.all_selftests()
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'LEARN-PULL SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    return m


def day_run(day):
    import lightgbm as lgb
    import pf_gap_top_makers as GP
    mut = all_selftests()
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'])
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    LT = D['LT']; ix = LT.index.values
    LT = LT.assign(signer=full.signer.values[ix].astype('int64'), tk=np.where(tkf[ix] >= 0, tkf[ix], -1 - ix))
    LT = LT.assign(wsg=pd.Series(LT.how.values > 0).groupby(LT.tk.values).transform('any').values)
    MP = pd.read_parquet(f'{PD.ORD}/map_{day}.parquet', columns=['row', 'maker', 'taker', 'maker_ok']).set_index('row')
    mk_ok = MP.maker_ok.eq(True).values & MP.maker.notna().values
    addr2sig = dict(zip(MP.maker.values[mk_ok], full.signer.values[MP.index.values[mk_ok]].astype('int64')))
    LT = LT.assign(tsig=[addr2sig.get(a_, -1) if isinstance(a_, str) else -1 for a_ in MP.taker.reindex(LT.index).values])
    del full
    gname = lambda s_: 'top 5' if G.get(int(s_), 'rest') == 'top 5' else 'field'
    mdl = lgb.Booster(model_file=f'{FPDIR}/models/av_{day}.txt')
    LTI = LT.groupby(['mid', 'side'], sort=False).indices; LTM = LT.groupby('mid', sort=False).indices
    PE = pd.read_parquet(f'{PD.OUTD}/events_{day}.parquet', columns=['eid', 'kind', 'side', 'p', 'tb'])
    st.update(window_sides=0, obs=0, e2_ambiguous=0, big_events=0)
    rows = []; brows = []; bev = []; evs = []; pend = {}; nb = len(K.G32)
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg = w['m'], w['side'], w['sg']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, S5, opp = w['P5'], w['S5'], w['opp']; tg = w['tg']; n = w['n']
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        runs = QR.runs(tf, P5, opp, side)
        val = lambda t, h, pt: sg * (mff[max(int(np.searchsorted(tf, t + h, side='right')) - 1, 0)] - pt)
        obs = []
        if (m, side) in LTI:
            x = LT.iloc[LTI[(m, side)]]
            e2 = {}
            for s_, p_, a_, g_ in zip(x.signer.values, x.ptick.values, x.anc.values, x.wsg.values):
                if not g_:
                    e2.setdefault((int(s_), int(p_)), []).append(float(a_))
            y = x[x.wsg.values & (x.anc.values >= w['ws']) & (x.anc.values < w['te'])]
            orders = sorted(((float(g.anc.values.min()), [(int(s_), int(p_), float(q_)) for s_, p_, q_ in zip(g.signer.values, g.ptick.values, g.shares.values)])
                             for _, g in y.groupby('tk', sort=False)), key=lambda o: o[0])

            def present_after(p, t):
                j = bisect.bisect_right(tfl, t)
                return j < len(tfl) and bool(np.any((P5[j] == p) & (S5[j] > 0)))
            obs, amb = next_trade_obs(side, orders, present_after, e2); st['e2_ambiguous'] += amb
        Pw = PD.av_P(w, mdl) if n else np.zeros(0)
        tgl = list(tg); r0 = len(rows)
        for o in obs:
            if not (w['ws'] <= o['t2'] < w['ws'] + 60000.0):
                continue
            k = int(np.searchsorted(tg, o['t2'] - DEC_LAG, side='right')) - 1
            if k < 0:
                continue
            xr = np.asarray(w['X'][k], float)
            d = {nm: xr[i] for i, nm in enumerate(XN)}
            adv = 1.0 if side == 1 else -1.0
            jk = int(w['jk'][k]); size_p = float((S5[jk] * (P5[jk] == o['p'])).sum()) if jk >= 0 else NAN
            feat = dict(v=d['v'], P=float(Pw[k]), adv_bn_r250=adv * d['bn_r250'], adv_bn_r1000=adv * d['bn_r1000'], adv_bn_flow250=adv * d['bn_flow250'],
                        adv_bn_imb=adv * d['bn_imb'], adv_pf_dm250=adv * d['pf_dm250'], adv_pf_imb1=adv * d['pf_imb1'], own_d250=d['own_d250'],
                        opp_d250=d['opp_d250'], pf_spread=d['pf_spread'], size_p=size_p, gap=o['gap'], t_rel=d['t_rel'], sigma=d['sigma'], act_rel=d['act_rel'])
            rows.append(dict(day=int(day), m=m, side=side, signer=o['signer'], grp=gname(o['signer']), p=o['p'], t1=o['t1'], t2=o['t2'], gap=o['gap'],
                             stayed=o['stayed'], through=o['through'], vol_p=o['vol_p'], v1=val(o['t2'], 1000.0, o['p']), v5=val(o['t2'], 5000.0, o['p']),
                             we_pull=int(we_pull(tgl, vg, o['t1'], o['t2'])), **feat, **{f'x_{nm}': d[nm] for nm in XN}))
        pend.setdefault(m, {})[side] = dict(w=w, vg=vg, runs=runs, tfl=tfl, rix=range(r0, len(rows)))
        w.pop('cache', None)
        if len(pend[m]) == 2:
            pair = pend.pop(m)
            # positions in this window from all legs of both sides (maker and taker roles)
            ev = []
            if m in LTM:
                xm = LT.iloc[LTM[m]]
                for s_, sd_, q_, a_, ts_ in zip(xm.signer.values, xm.side.values, xm.shares.values, xm.anc.values, xm.tsig.values):
                    ev += leg_events(int(sd_), int(s_), int(ts_) if ts_ >= 0 else None, float(q_), float(a_))
            by = positions(ev)
            for s_ in (0, 1):
                for i_ in pair[s_]['rix']:
                    r = rows[i_]; r['inv'] = inv_dir(pos_before(by, r['signer'], r['t1']), r['side'])
            # PART B: big moves on the side the price moves into
            ons = {s_: [float(pair[s_]['w']['tg'][k]) for k in PD.onsets(pair[s_]['vg'])] for s_ in (0, 1)}
            used = set()

            def starts(s_, kd):
                b0_, valid_, rs_, kind_ = pair[s_]['runs']; tl = pair[s_]['tfl']; out = []
                for j in range(1, len(tl)):
                    if rs_[j] == j and kind_[j] == kd:
                        jn = j + 1
                        while jn < len(tl) and rs_[jn] == j:
                            jn += 1
                        out.append((j, int(b0_[j]), tl[j], tl[jn] if jn < len(tl) else INF, int(b0_[j - 1])))
                return out
            ST = {(s_, kd): starts(s_, kd) for s_ in (0, 1) for kd in (1, 2)}
            wsm = pair[0]['w']['ws']; allon = sorted(ons[0] + ons[1]); E = []
            for s_ in (0, 1):
                for t0 in ons[s_]:
                    if not (wsm <= t0 < wsm + 60000.0):
                        continue
                    for x_, kd, kev in ((s_, 2, 'adverse'), (1 - s_, 1, 'other')):
                        c = [e for e in ST[(x_, kd)] if t0 - RC.WIN_PRE < e[2] <= t0 + RC.WIN_POST]
                        if not c or (x_, c[0][0]) in used:
                            continue
                        j, p, tb, te_run, po = c[0]; used.add((x_, j)); E.append((f'{day}|{m}|{x_}|{j}', kev, x_, j, p, tb, po))
            for x_ in (0, 1):
                for kd, kev in ((2, 'calm fallback'), (1, 'calm improve')):
                    for j, p, tb, te_run, po in ST[(x_, kd)]:
                        if (x_, j) in used or not (wsm <= tb < wsm + 60000.0):
                            continue
                        lo = bisect.bisect_left(allon, tb - RC.WIN_POST)
                        if lo < len(allon) and allon[lo] < tb + RC.WIN_PRE:
                            continue
                        E.append((f'{day}|{m}|{x_}|{j}', kev, x_, j, p, tb, po))
            for eid, kev, x_, j, p, tb, po in E:
                evs.append((eid, kev, x_, p, tb))
                if kev != 'adverse':
                    continue
                b0x, validx, _, _ = pair[x_]['runs']
                if not validx[j] or p < 0 or p == po or tb > wsm + MC.LAST_TB:
                    continue
                M, t_pk, Fk = MC.move_path(pair[x_]['tfl'], b0x, validx, x_, po, tb)
                if M < 3 or not (Fk == Fk):
                    continue
                st['big_events'] += 1; spd = t_pk - tb
                xm = LT.iloc[LTI[(m, x_)]] if (m, x_) in LTI else LT.iloc[:0]
                xm = xm[xm.wsg.values & (xm.anc.values > tb - MC.PRE) & (xm.anc.values <= t_pk)]
                anyt5 = False
                for s_, p_, q_, a_ in zip(xm.signer.values, xm.ptick.values, xm.shares.values, xm.anc.values):
                    lev = (int(p_) - po) if x_ == 1 else (po - int(p_))
                    if not (0 <= lev <= M):
                        continue
                    lv = 'old' if lev == 0 else ('peak' if lev == M else ('1' if lev == 1 else ('2' if lev == 2 else '3+')))
                    g_ = gname(s_); inv = inv_dir(pos_before(by, int(s_), float(a_)), x_)
                    ivb = 'reducing' if inv <= -INV_MIN else ('increasing' if inv >= INV_MIN else 'flat')
                    brows.append((int(day), 'UP' if x_ == 1 else 'DOWN', M, spd, lv, g_, float(q_), ivb))
                    anyt5 = anyt5 or (g_ == 'top 5' and lv not in ('old', 'peak'))
                bev.append((int(day), 'UP' if x_ == 1 else 'DOWN', M, spd, int(anyt5)))
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    EV = pd.DataFrame(evs, columns=['eid', 'kind', 'side', 'p', 'tb'])
    st['events_differ'] = int(len(EV) != len(PE) or not (EV.eid.values.astype(str) == PE.eid.values.astype(str)).all()
                              or not (EV.p.values == PE.p.values).all() or not (EV.tb.values == PE.tb.values).all())
    st['unpaired_windows'] = len(pend); st['obs'] = len(rows); st['inv_missing'] = int(sum('inv' not in r for r in rows))
    if st['events_differ'] or st['unpaired_windows'] or st['inv_missing']:
        sys.exit(f'LEARN-PULL CONTROL FAILED on {day} -> no output: { {k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)} }')
    st['planted_caught'] = mut
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(rows).to_parquet(f'{OUTD}/obs_{day}.parquet', index=False)
    pd.DataFrame(brows, columns=['day', 'dir', 'M', 'spd', 'lv', 'grp', 'q', 'invb']).to_parquet(f'{OUTD}/bigfills_{day}.parquet', index=False)
    pd.DataFrame(bev, columns=['day', 'dir', 'M', 'spd', 'anyt5']).to_parquet(f'{OUTD}/bigevents_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {json.dumps({k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)}, default=float)} in {time.time() - T0:.0f} s', flush=True)


def auc(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return NAN
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def report():
    import lightgbm as lgb
    stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}
    A = pd.concat([pd.read_parquet(f'{OUTD}/obs_{d}.parquet') for d in DAYS9], ignore_index=True)
    BF = pd.concat([pd.read_parquet(f'{OUTD}/bigfills_{d}.parquet') for d in DAYS9], ignore_index=True)
    BE = pd.concat([pd.read_parquet(f'{OUTD}/bigevents_{d}.parquet') for d in DAYS9], ignore_index=True)
    A['gone'] = 1 - A.stayed
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    T5 = A[A.grp == 'top 5']; FD = A[A.grp == 'field']
    vw = lambda x, c='v5': float((x[c] * x.vol_p).sum() / x.vol_p.sum()) if len(x) and x.vol_p.sum() else NAN
    R = {}
    L = [f'LEARN-PULL, 9 consumed replay days, first 60 s, WS-timed tape; as of {asof}; controls passed; observations {len(A)} (top 5 {len(T5)},'
         f' field {len(FD)}); E2-ambiguous {sum(stats[d]["stat"]["e2_ambiguous"] for d in DAYS9)}; big events {sum(stats[d]["stat"]["big_events"] for d in DAYS9)}', '']
    # R551 rates
    L.append('R551 PULL RATE on the same intervals (front order at the best, next trade at that price): their GONE share | our pull share, by interval')
    for g_, X in (('top 5', T5), ('field', FD)):
        parts = []
        for nm, lo, hi in GAPB:
            y = X[(X.gap >= lo) & (X.gap < hi)]
            parts.append(f'{nm}: {y.gone.mean():5.1%} | {y.we_pull.mean():5.1%} ({len(y)})')
        L.append(f'  {g_:<6} all: {X.gone.mean():5.1%} | {X.we_pull.mean():5.1%};  ' + ';  '.join(parts))
    # R550 avoid
    L += ['', 'R550 / values: GONE share and our pull share before THROUGH vs STOP next trades; next-trade value at 5 s (volume-weighted c/sh)']
    for g_, X in (('top 5', T5), ('field', FD)):
        th = X[X.through == 1]; sp = X[X.through == 0]
        L.append(f'  {g_:<6} THROUGH: GONE {th.gone.mean():5.1%}, we pull {th.we_pull.mean():5.1%}, value {vw(th):+.2f} ({len(th)}) | STOP: GONE {sp.gone.mean():5.1%},'
                 f' we pull {sp.we_pull.mean():5.1%}, value {vw(sp):+.2f} ({len(sp)})')
        L.append(f'         next-trade value when they STAYED {vw(X[X.stayed == 1]):+.2f} / when they were GONE {vw(X[X.stayed == 0]):+.2f};'
                 f' when we pull {vw(X[X.we_pull == 1]):+.2f} / when we do not {vw(X[X.we_pull == 0]):+.2f}')
    hi = 0
    for d in DAYS9:
        y = T5[T5.day == int(d)]; th = y[y.through == 1]; sp = y[y.through == 0]
        if len(th) >= MIN_N and len(sp) >= MIN_N:
            hi += int(th.gone.mean() > sp.gone.mean())
    ratio = T5[T5.through == 1].gone.mean() / max(T5[T5.through == 0].gone.mean(), 1e-9)
    R['R550'] = dict(ratio=float(ratio), higher_days=hi, label='AVOID-ADVERSE' if ratio >= 1.2 and hi >= 7 else 'NOT-SHOWN')
    L.append(f'  R550: top-5 GONE before THROUGH / before STOP = {ratio:.2f}, higher on {hi} of 9 -> {R["R550"]["label"]}')
    # R552 / R553: the 2x2
    L += ['', 'R552 / R553 THE 2 x 2 (top 5): share of intervals | next-trade value at 5 s (c/sh, volume-weighted) | share THROUGH']
    cells = {('pull', 'gone'): (1, 1), ('pull', 'stay'): (1, 0), ('keep', 'gone'): (0, 1), ('keep', 'stay'): (0, 0)}
    for (ours, theirs), (wp, gn) in cells.items():
        y = T5[(T5.we_pull == wp) & (T5.gone == gn)]
        L.append(f'  we {ours:<4}, they {"pull" if gn else "stay":<4}: {len(y) / len(T5):5.1%} | {vw(y):+.2f} | {y.through.mean():5.1%}')
    for key, (wp, gn), good, bad in (('R552', (1, 0), 'OUR-EXTRA-PULLS-COST', 'OUR-EXTRA-PULLS-SAVE'), ('R553', (0, 1), 'THEY-PULL-NEEDLESSLY', 'THEY-SEE-WHAT-WE-MISS')):
        pos = neg = 0
        for d in DAYS9:
            y = T5[(T5.day == int(d)) & (T5.we_pull == wp) & (T5.gone == gn)]
            if len(y) >= MIN_N:
                pos += int(vw(y) > 0); neg += int(vw(y) < 0)
        R[key] = dict(value=vw(T5[(T5.we_pull == wp) & (T5.gone == gn)]), pos=pos, neg=neg, label=good if pos >= 7 else (bad if neg >= 7 else 'NOT-SHOWN'))
        L.append(f'  {key}: value {R[key]["value"]:+.2f}, > 0 on {pos}, < 0 on {neg} of 9 -> {R[key]["label"]}')
    # R555 inventory
    L += ['', 'R555 INVENTORY: GONE share by what the order does to the maker\'s position (reduce <= -10 sh | flat | add >= 10 sh)']
    for g_, X in (('top 5', T5), ('field', FD)):
        L.append(f'  {g_:<6} ' + ' | '.join(f'{nm} {y.gone.mean():5.1%} ({len(y)})' for nm, y in
                                           (('reduce', X[X.inv <= -INV_MIN]), ('flat', X[(X.inv > -INV_MIN) & (X.inv < INV_MIN)]), ('add', X[X.inv >= INV_MIN]))))
    hi = 0
    for d in DAYS9:
        y = T5[T5.day == int(d)]; a_ = y[y.inv >= INV_MIN]; r_ = y[y.inv <= -INV_MIN]
        if len(a_) >= MIN_N and len(r_) >= MIN_N:
            hi += int(a_.gone.mean() > r_.gone.mean())
    rr = T5[T5.inv >= INV_MIN].gone.mean() / max(T5[T5.inv <= -INV_MIN].gone.mean(), 1e-9)
    R['R555'] = dict(ratio=float(rr), higher_days=hi, label='INVENTORY-DRIVEN' if rr >= 1.2 and hi >= 7 else 'NOT-SHOWN')
    L.append(f'  R555: add / reduce = {rr:.2f}, higher on {hi} of 9 -> {R["R555"]["label"]}')
    # R554 why: quintiles and model
    L += ['', 'R554 WHY: GONE share by feature quintile (Q1 .. Q5, low to high) - top 5 | field | our pull share']
    for f_ in NAMED:
        x = A[np.isfinite(A[f_])]
        if x[f_].nunique() < 5:
            continue
        q = pd.qcut(x[f_].rank(method='first'), 5, labels=False)
        t5 = [x[(q == i) & (x.grp == 'top 5')].gone.mean() for i in range(5)]
        fd = [x[(q == i) & (x.grp == 'field')].gone.mean() for i in range(5)]
        wp = [x[q == i].we_pull.mean() for i in range(5)]
        L.append(f'  {f_:<15} top 5 ' + ' '.join(f'{v_:4.0%}' for v_ in t5) + ' | field ' + ' '.join(f'{v_:4.0%}' for v_ in fd) + ' | ours ' + ' '.join(f'{v_:4.0%}' for v_ in wp))
    feats = NAMED + [f'x_{nm}' for nm in XN if f'x_{nm}' not in ('x_v',)]
    out = {}
    for nm, cols in (('all', feats), ('v only', ['v']), ('inventory only', ['inv'])):
        pr = np.full(len(T5), np.nan); rng = np.random.default_rng(11)
        for d in DAYS9:
            tr = T5[T5.day != int(d)]; te = T5.day.values == int(d)
            if len(tr) > SUB_N:
                tr = tr.iloc[np.sort(rng.choice(len(tr), SUB_N, replace=False))]
            b = lgb.train(LGB_PARAMS, lgb.Dataset(tr[cols].values.astype(np.float32), tr.gone.values), N_ROUNDS)
            pr[te] = b.predict(T5[cols].values[te].astype(np.float32))
            if nm == 'all' and d == DAYS9[-1]:
                imp = sorted(zip(cols, b.feature_importance('gain')), key=lambda z: -z[1])[:10]
        out[nm] = auc(T5.gone.values, pr)
    R['R554'] = dict(auc=out, top_gain=[(c, float(g)) for c, g in imp])
    L.append(f'  model (top-5 GONE, leave one day out): AUC all features {out["all"]:.3f}, v only {out["v only"]:.3f}, inventory only {out["inventory only"]:.3f};'
             f' top gain: ' + ', '.join(f'{c}' for c, _ in imp))
    # R556 big moves
    L += ['', f'R556 BIG MOVES (>= 3 ticks in 2 s after a signal; {len(BE)} events): share of events with a top-5 fill at a price run over, by speed'
          ' (time to the peak)']
    spb = (('<= 250 ms', -1, 250), ('250-1000 ms', 250, 1000), ('> 1000 ms', 1000, 1e18))
    for nm, lo, hi_ in spb:
        e = BE[(BE.spd > lo) & (BE.spd <= hi_)]
        L.append(f'  {nm:<12} {e.anyt5.mean():5.1%} of {len(e)} events')
    L.append('  top-5 share of the volume filled at each price (old best | run-over 1 | 2 | 3+ | peak), by speed')
    for nm, lo, hi_ in spb:
        b = BF[(BF.spd > lo) & (BF.spd <= hi_)]
        L.append(f'  {nm:<12} ' + ' | '.join(f'{lv} {b.q[(b.lv == lv) & (b.grp == "top 5")].sum() / max(b.q[b.lv == lv].sum(), 1e-9):5.1%}' for lv in ('old', '1', '2', '3+', 'peak')))
    L.append('  run-over volume (levels 1, 2, 3+) by the maker\'s inventory direction: reducing | flat | increasing')
    ro = BF[BF.lv.isin(['1', '2', '3+'])]
    for g_ in ('top 5', 'field'):
        b = ro[ro.grp == g_]; tot = max(b.q.sum(), 1e-9)
        L.append(f'  {g_:<6} ' + ' | '.join(f'{ib} {b.q[b.invb == ib].sum() / tot:5.1%}' for ib in ('reducing', 'flat', 'increasing')))
    L += ['', 'READINGS: ' + json.dumps({k_: v_.get('label', '') for k_, v_ in R.items() if isinstance(v_, dict)})]
    json.dump(R, open(f'{OUTD}/learn_pull.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/learn_pull.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        print(f'selftest passed; planted errors caught: {all_selftests()}')
    elif cmd == 'day' and len(sys.argv) > 2:
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
