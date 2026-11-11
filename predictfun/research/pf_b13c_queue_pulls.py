"""B13c (2026-11-11, USER "check why top makers still have front position in the queue, it looks our model guards on
some orders that should be there?"). Does our engine pull at-best orders that should have stayed, and what does each pull
cost in queue place? Read-only, descriptive. Live order paths of tokyo-6 + basis-tokyo-1 (core_orders, as B1a) in the 220
consumed 11-08 cohort windows, first minute; the research tape and book frames; the research guard (lag 50). G = 1 day:
point estimates only.

Orders: accepted live orders that were cancelled (t_cancel present), whose price was the best of their side in the
research frame strictly before the cancel request ('pulled at the best'); first minute by cancel time. Classes by the
research guard of the order's side at t_cancel - 50 ms: ON (the research guard still said quote) or OFF.
Measures:
 life      t_cancel - t_Accepted.
 churn     a new order of ours on the same side and price accepted within 1 s after the cancel request (a re-place).
 queue     queue ahead when an order went live = shown size at its price in the frame strictly before t_Accepted (raised
           to the next frame's if larger and <= 100 ms later, as B1a's rule M); at the cancel it is reduced by the volume
           traded at that price on that side in between (FIFO; cancels ahead are not visible, so this overstates it). For a
           re-place: its own queue ahead at activation minus the old order's queue ahead carried to that moment = queue lost.
 flow      the maker legs traded at our price on our side in (t_cancel, t_cancel + 2 s]: their share-weighted net value at
           1 s (c/sh), pro rata.
 would-be  had the order stayed: trades at our price after the cancel first use up the carried queue ahead, then fill the
           order up to its size (rule M as B1a; no cancels ahead; B1a found this rule over-credits live by 1.5-2x); shares
           and net value at 1 s within 2 s.

Readings DECLARED BEFORE THE FIRST RUN:
 R110 churn       CHURN-HEAVY if >= 30% of pulls at the best are re-placed at the same price within 1 s; else CHURN-LIGHT.
 R111 queue       QUEUE-LOST if the median queue lost per re-place is >= 50 sh; else QUEUE-KEPT.
 R112 flow        PULLS-COST if the flow after ON pulls nets > +0.3 c/sh; PULLS-PROTECT if the flow after OFF pulls nets
                  < -0.3 c/sh; both may apply; NEITHER otherwise. n/a under 200 legs of flow in a class.
 R113 would-be    for ON pulls: STAYING-PAYS if the would-be fills net > 0 c/sh on >= 100 sh; STAYING-DOES-NOT-PAY if < 0 on
                  >= 100 sh; else n/a.
 R114 summary     GUARD-PULLS-GOOD-ORDERS if R110 CHURN-HEAVY and (R112 includes PULLS-COST or R113 STAYING-PAYS); else
                  NOT-SHOWN.
What the outcomes mean (fixed in advance): GUARD-PULLS-GOOD-ORDERS = we pull at-best orders often, re-place them at the
back of the queue, and the flow we step away from (where the research guard still said quote) is good: we give up queue
place and good fills. PULLS-PROTECT = pulls on an OFF guard step away from bad flow (the guard works there).
Controls (any failure -> no table): B12b's controls; queue carry and the would-be fill rule on planted orders (a trade
smaller than the queue ahead fills nothing; the next fills the rest up to the size); re-place matching on planted
orders (same side and price within 1 s -> re-place; other price or later -> not); K-check: the live filled shares equal
849.867 sh (B1a); label rules.
Limits fixed in advance: one consumed Sunday; the shown queue and FIFO carry ignore cancels ahead (the queue ahead is
overstated, would-be fills understated); pro rata flow is not our fill (B1a); the live guard is not the research guard;
trade times are reports.
Usage: predictfun/research/run_light.sh b13c_queue_pulls -- python3 -m predictfun.research.pf_b13c_queue_pulls <out_dir>
       python3 -m predictfun.research.pf_b13c_queue_pulls --selftest
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_b12_makers as B
from . import pf_b12b_reason as BB
from . import pf_b12c_model_or_speed as C1
from . import pf_b12d_bursts as DB

LIVE_SH = 849.867319
REPLACE_MS = 1000.0; FLOW_MS = 2000.0; RAISE_MS = 100.0
NAN = float('nan')


def queue_at(tf, Pk, Sk, t, p):
    """Shown size at price p in the frame strictly before t, raised to the next frame's if larger and <= 100 ms later."""
    j = np.searchsorted(tf, t, side='left') - 1
    if j < 0:
        return NAN
    q = DB.size_at(Pk, Sk, j, p)
    if j + 1 < len(tf) and tf[j + 1] - t <= RAISE_MS:
        q = max(q, DB.size_at(Pk, Sk, j + 1, p))
    return q


def would_fill(q_ahead, size, trade_sh):
    """FIFO: each trade first uses the queue ahead, then fills us up to size. Returns our fill per trade."""
    out = np.zeros(len(trade_sh)); qa = q_ahead; left = size
    for i, s in enumerate(trade_sh):
        if left <= 0:
            break
        use = min(qa, s); qa -= use; s -= use
        f = min(left, s); out[i] = f; left -= f
    return out


def label_r110(x):
    return 'n/a' if not BB.fin(x) else ('CHURN-HEAVY' if x >= 0.30 else 'CHURN-LIGHT')


def label_r111(x):
    return 'n/a' if not BB.fin(x) else ('QUEUE-LOST' if x >= 50.0 else 'QUEUE-KEPT')


def label_r112(v_on, n_on, v_off, n_off):
    lab = []
    if n_on >= 200 and BB.fin(v_on) and v_on > 0.3:
        lab.append('PULLS-COST')
    if n_off >= 200 and BB.fin(v_off) and v_off < -0.3:
        lab.append('PULLS-PROTECT')
    if n_on < 200 and n_off < 200:
        return ['n/a']
    return lab or ['NEITHER']


def label_r113(v, sh):
    if not BB.fin(v) or sh < 100:
        return 'n/a'
    return 'STAYING-PAYS' if v > 0 else 'STAYING-DOES-NOT-PAY'


def label_r114(r110, r112, r113):
    return 'GUARD-PULLS-GOOD-ORDERS' if (r110 == 'CHURN-HEAVY' and ('PULLS-COST' in r112 or r113 == 'STAYING-PAYS')) else 'NOT-SHOWN'


def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    okb, repb = BB.controls(); chk('B12b_controls', okb)
    chk('Q1_would_fill', list(would_fill(10.0, 5.0, np.array([4.0, 8.0, 5.0]))) == [0.0, 2.0, 3.0] and list(would_fill(0.0, 5.0, np.array([3.0, 3.0]))) == [3.0, 2.0])
    tf = np.array([0.0, 50.0, 120.0]); Pk = np.array([[50, 49, 48, 47, 46]] * 3); Sk = np.array([[100.0, 5, 5, 5, 5], [140.0, 5, 5, 5, 5], [90.0, 5, 5, 5, 5]])
    chk('Q2_queue_at', queue_at(tf, Pk, Sk, 40.0, 50) == 140.0 and queue_at(tf, Pk, Sk, 100.0, 50) == 140.0 and queue_at(tf, Pk, Sk, 130.0, 50) == 90.0
        and np.isnan(queue_at(tf, Pk, Sk, 0.0, 50)))
    o = pd.DataFrame(dict(market=[1, 1, 1, 1], side=[0, 0, 0, 1], tick=[50, 50, 49, 50], t_cancel=[1000.0, np.nan, np.nan, np.nan], t_Accepted=[0.0, 1500.0, 1200.0, 1100.0]))
    rp = replace_match(o)
    chk('Q3_replace', list(rp) == [1, -1, -1, -1])
    chk('Q4_rules', label_r110(0.4) == 'CHURN-HEAVY' and label_r111(60) == 'QUEUE-LOST' and label_r112(0.5, 300, -0.5, 300) == ['PULLS-COST', 'PULLS-PROTECT']
        and label_r112(0.1, 300, 0.1, 300) == ['NEITHER'] and label_r112(0.5, 10, -0.5, 10) == ['n/a'] and label_r113(0.2, 150) == 'STAYING-PAYS'
        and label_r113(0.2, 50) == 'n/a' and label_r114('CHURN-HEAVY', ['PULLS-COST'], 'n/a') == 'GUARD-PULLS-GOOD-ORDERS'
        and label_r114('CHURN-LIGHT', ['PULLS-COST'], 'STAYING-PAYS') == 'NOT-SHOWN' and label_r110(NAN) == 'n/a')
    rep['B12b_controls_detail'] = repb
    return ok, rep


def replace_match(o):
    """For each cancelled order: index of our next order on the same market, side and tick accepted in (t_cancel, t_cancel
    + 1 s], else -1."""
    if not isinstance(o.index, pd.RangeIndex) or o.index.start != 0:
        raise ValueError('replace_match needs a 0-based RangeIndex')
    out = np.full(len(o), -1, np.int64)
    acc = o[o.t_Accepted.notna()]
    grp = {k: (g.t_Accepted.values, g.index.values) for k, g in acc.sort_values('t_Accepted').groupby(['market', 'side', 'tick'])}
    for i, r in enumerate(o.itertuples()):
        if not np.isfinite(r.t_cancel):
            continue
        g = grp.get((r.market, r.side, r.tick))
        if g is None:
            continue
        ts, ix = g; j = np.searchsorted(ts, r.t_cancel, side='right')
        if j < len(ts) and ts[j] <= r.t_cancel + REPLACE_MS:
            out[i] = int(ix[j])
    return out


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps({k: v for k, v in rep.items() if not k.endswith('detail')}, indent=1)); print('selftest', 'PASS' if ok else 'FAIL')
        sys.exit(0 if ok else 1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    X = BB.build(); R = X['R']; W0 = X['W0']; G = X['G']; LINE = X['LINE']
    k0 = BB.reconcile(R, X['days'])
    if not k0['ok']:
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    core = []
    for r in ('v6', 'basis'):
        K = pd.read_parquet(f'{BB.LV}/core_orders_{r}.parquet')
        core.append(K[K.t_Accepted.notna() & K.refused.isna() & K.market.isin(list(W0))].assign(run=r))
    core = pd.concat(core, ignore_index=True); core['side'] = (core.lane % 2).astype(int); core['ws'] = core.market.map(W0).astype(float)
    kc = dict(live_filled=float(core.filled.sum())); kc['ok'] = abs(kc['live_filled'] - LIVE_SH) < 1e-3
    if not kc['ok']:
        sys.exit(f'CONTROL FAILED (K-check live shares) -> no table: {kc}')
    core = core.sort_values(['market', 't_Accepted'], kind='stable').reset_index(drop=True)
    core['rp'] = replace_match(core)
    FR = DB.frames(W0)
    # trades per (market, side, price): research time and shares, value at 1 s
    Tl = R[R.val_ok & np.isfinite(R.anc.values)][['mid', 'side', 'ptick', 'anc', 'shares', 'v1', 'g']]
    TG = {k: (g.anc.values, g.shares.values, g.v1.values, g.g.values) for k, g in Tl.sort_values('anc').groupby(['mid', 'side', 'ptick'])}
    rows = []
    for o in core.itertuples():
        if not np.isfinite(o.t_cancel) or not (o.ws <= o.t_cancel < o.ws + 60000.0) or o.market not in FR:
            continue
        tf, S = FR[o.market]; Pk, Sk = S[o.side]; j = np.searchsorted(tf, o.t_cancel, side='left') - 1
        if j < 0 or Pk[j, 0] != o.tick:
            continue
        tau, vb, va = G[o.market]; v = C1.vlook(tau, (vb, va)[o.side], np.array([o.t_cancel - 50.0]), LINE)[0]
        gs = 'ON' if (np.isfinite(v) and v > LINE) else ('OFF' if np.isfinite(v) else 'n/a')
        tg = TG.get((o.market, o.side, int(o.tick)))
        qa0 = queue_at(tf, Pk, Sk, o.t_Accepted, int(o.tick))
        if tg is not None:
            ta, ts, tv, tgr = tg
            between = (ta > o.t_Accepted) & (ta <= o.t_cancel); after = (ta > o.t_cancel) & (ta <= o.t_cancel + FLOW_MS)
            vol_between = float(ts[between].sum()); qa_c = max(0.0, qa0 - vol_between) if np.isfinite(qa0) else NAN
            fsh = ts[after]; fv = tv[after]; fg = tgr[after]
            wf = would_fill(qa_c, float(o.size) - float(o.filled), fsh) if np.isfinite(qa_c) else np.zeros(len(fsh))
        else:
            qa_c = qa0; fsh = np.array([]); fv = np.array([]); fg = np.array([]); wf = np.array([])
        own = fg == 'ours'
        q_lost = NAN
        if o.rp >= 0:
            n_ = core.iloc[o.rp]; qn = queue_at(tf, Pk, Sk, n_.t_Accepted, int(o.tick))
            vol2 = float(tg[1][(tg[0] > o.t_cancel) & (tg[0] <= n_.t_Accepted)].sum()) if tg is not None else 0.0
            if np.isfinite(qn) and np.isfinite(qa_c):
                q_lost = qn - max(0.0, qa_c - vol2)
        rows.append(dict(market=o.market, side=o.side, tick=int(o.tick), size=float(o.size), life=float(o.t_cancel - o.t_Accepted), guard=gs,
                         replaced=bool(o.rp >= 0), q_ahead_act=qa0, q_ahead_cancel=qa_c, q_lost=q_lost,
                         flow_sh=float(fsh[~own].sum()), flow_val=float((fsh[~own] * fv[~own]).sum()), flow_legs=int((~own).sum()),
                         wf_sh=float(wf.sum()), wf_val=float((wf * fv).sum()) if len(wf) else 0.0))
    P = pd.DataFrame(rows)
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08)', controls=ctl, K0=k0['ok'], Kcheck=kc, pulls_at_best=int(len(P)))
    life = P.life.values
    res['life_ms_p10_50_90'] = [float(v) for v in np.percentile(life, [10, 50, 90])]
    res['life_share_le_1s'] = float(np.mean(life <= 1000.0)); res['life_share_le_2s'] = float(np.mean(life <= 2000.0))
    res['by_guard'] = {}
    for gs in ('ON', 'OFF', 'n/a'):
        x = P[P.guard == gs]
        res['by_guard'][gs] = dict(pulls=int(len(x)), replaced_share=float(x.replaced.mean()) if len(x) else NAN,
                                   flow_legs=int(x.flow_legs.sum()), flow_sh=float(x.flow_sh.sum()), flow_net1=float(x.flow_val.sum() / x.flow_sh.sum()) if x.flow_sh.sum() > 0 else NAN,
                                   would_be_sh=float(x.wf_sh.sum()), would_be_net1=float(x.wf_val.sum() / x.wf_sh.sum()) if x.wf_sh.sum() > 0 else NAN,
                                   q_ahead_act_p50=float(np.nanmedian(x.q_ahead_act)) if len(x) else NAN)
    rp_ = P[P.replaced & np.isfinite(P.q_lost.values)]
    res['replace'] = dict(share=float(P.replaced.mean()), n_with_queue=int(len(rp_)), q_lost_p25_50_75=[float(v) for v in np.percentile(rp_.q_lost, [25, 50, 75])] if len(rp_) else [],
                          q_ahead_act_p50_all=float(np.nanmedian(P.q_ahead_act)))
    on, off = res['by_guard']['ON'], res['by_guard']['OFF']
    r110 = label_r110(res['replace']['share']); r111 = label_r111(res['replace']['q_lost_p25_50_75'][1] if res['replace']['q_lost_p25_50_75'] else NAN)
    r112 = label_r112(on['flow_net1'], on['flow_legs'], off['flow_net1'], off['flow_legs']); r113 = label_r113(on['would_be_net1'], on['would_be_sh'])
    res['readings'] = dict(R110=r110, R111=r111, R112=r112, R113=r113, R114=label_r114(r110, r112, r113))
    json.dump(res, open(f'{out}/b13c.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not BB.fin(v) else f'{v:+.2f}'
    Lt = ['B13c does our engine pull at-best orders that should stay, and what does a pull cost in queue place? (11-08, first minute)',
          f"Controls: {json.dumps({k: v for k, v in ctl.items() if not k.endswith('detail')})}; live filled {kc['live_filled']:.3f} sh",
          f"Pulls at the best: {len(P)}; order life p10/50/90 {[round(v) for v in res['life_ms_p10_50_90']]} ms; <= 1 s {res['life_share_le_1s'] * 100:.0f}%, <= 2 s {res['life_share_le_2s'] * 100:.0f}%",
          f"Re-placed at the same price within 1 s: {res['replace']['share'] * 100:.1f}%; queue lost per re-place p25/50/75 {[round(v) for v in res['replace']['q_lost_p25_50_75']]} sh "
          f"(n {res['replace']['n_with_queue']}); queue ahead at activation p50 {res['replace']['q_ahead_act_p50_all']:.0f} sh", '',
          'by the research guard at the pull: pulls | re-placed | flow at our price in the next 2 s: legs, sh, net 1 s | would-be fills had we stayed: sh, net 1 s | queue ahead at activation p50']
    for gs, x in res['by_guard'].items():
        Lt.append(f"{gs} | {x['pulls']} | {x['replaced_share'] * 100:.0f}% | {x['flow_legs']}, {x['flow_sh']:,.0f}, {f(x['flow_net1'])} | {x['would_be_sh']:,.1f}, {f(x['would_be_net1'])} | {x['q_ahead_act_p50']:.0f}")
    Lt += ['', 'Readings: ' + json.dumps(res['readings'])]
    open(f'{out}/b13c.txt', 'w').write('\n'.join(Lt) + '\n'); print('\n'.join(Lt))


if __name__ == '__main__':
    main()
