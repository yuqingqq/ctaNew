"""ORDER-HARM WHY - POST HOC follow-up (2026-11-14, session ce68ebe0), labelled post hoc: chosen after reading W2 (every arm and clock
WITHIN-STATE). Reads ORDER-HARM WHY's own outputs (no new replay).

QUESTION: inside the same state (phase x move x depth), why are our fills worth less than the top 5's? A trade that STOPS at a price fills
only the front of that price's queue; a trade that goes THROUGH it fills the whole level and moves on (pf_depth_timing.trade_type). A late
queue place leaves an order mostly the THROUGH fills.
  WT  WITHIN = sum_s w_ours(s) (v_ours(s) - v_top5(s)) splits, per state, over the trade type t (STOP / THROUGH):
      TYPE-MIX  = sum_s w_ours(s) sum_t (pi_ours(t|s) - pi_top5(t|s)) v_top5(t, s)   (we get a different share of STOP vs THROUGH fills)
      SAME-TYPE = sum_s w_ours(s) sum_t pi_ours(t|s) (v_ours(t, s) - v_top5(t, s))   (same state and type, still worse)
      (+ a residual for types the top 5 never get in that state). Label per arm x clock (WS-timed, ours w/o young, 1 s): QUEUE-PLACE if
      TYPE-MIX < 0 and |TYPE-MIX| >= |SAME-TYPE|; SAME-TYPE-WORSE if SAME-TYPE < 0 and larger; else NEITHER.
  ST  printed: STOP share of shares and c/sh by type, top 5 vs ours, by phase group (CALM / AFTER / ON) x depth (at best / behind), WS, M 50.
  E2  printed: c/sh at 1 s by trade source (WS / E2) - top 5 by phase group, ours by timing class (M 50, all ages).
  KT  printed: OHQ0's KEPT-ON fills at M 50 by trade type and depth: sh/day, c/sh at 1 s and at the 60.2 s mark.
Control: WITHIN recomputed here from the per-state sums equals ORDER-HARM WHY's printed WITHIN (to 1e-3 c/sh) for every arm x clock (WS),
else no output; falsifier: the same with the top 5's state values swapped for ours must differ.
Usage (run_light, PF_SLOT=claude): pf_order_harm_why_posthoc.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_order_harm_why as WY                            # noqa: E402

OUTD = WY.OUTD
TYPES = ('STOP', 'THROUGH')
GROUP = {'CALM': 'CALM', 'AFTER<250': 'AFTER', 'AFTER250-1000': 'AFTER', 'ON<100': 'ON', 'ON100-250': 'ON', 'ON250-500': 'ON',
         'ON500-1000': 'ON', 'ON>=1000': 'ON', 'masked': 'other', 'none': 'other'}


def within_split(ours, theirs, swap=False):
    """ours / theirs: {(state, type): (shares, value sum)}. Returns WITHIN, TYPE-MIX, SAME-TYPE, RESIDUAL (c/sh of our shares)."""
    qo = sum(v[0] for v in ours.values())
    st_o = {}; st_t = {}
    for (s, t), v in ours.items():
        st_o.setdefault(s, {})[t] = v
    for (s, t), v in theirs.items():
        st_t.setdefault(s, {})[t] = v
    within = tmix = same = resid = 0.0
    for s, to in st_o.items():
        tt = st_t.get(s)
        qs_o = sum(v[0] for v in to.values())
        if not tt or qs_o <= 0:
            continue
        qs_t = sum(v[0] for v in tt.values())
        if qs_t <= 0:
            continue
        w = qs_o / qo
        vo_s = sum(v[1] for v in to.values()) / qs_o; vt_s = sum(v[1] for v in tt.values()) / qs_t
        if swap:
            vo_s, vt_s = vt_s, vo_s
        within += w * (vo_s - vt_s)
        for t in TYPES:
            po = to.get(t, (0.0, 0.0))[0] / qs_o; pt = tt.get(t, (0.0, 0.0))[0] / qs_t
            vt = tt[t][1] / tt[t][0] if t in tt and tt[t][0] > 0 else None
            vo = to[t][1] / to[t][0] if t in to and to[t][0] > 0 else None
            if vt is None:
                if vo is not None:
                    resid += w * po * (vo - vt_s)
                continue
            tmix += w * (po - pt) * vt
            if vo is not None:
                same += w * po * (vo - vt)
    return dict(within=within, tmix=tmix, same=same, resid=resid)


def label(dd):
    if dd['tmix'] < 0 and abs(dd['tmix']) >= abs(dd['same']):
        return 'QUEUE-PLACE'
    if dd['same'] < 0 and abs(dd['same']) > abs(dd['tmix']):
        return 'SAME-TYPE-WORSE'
    return 'NEITHER'


def main():
    T0 = time.time()
    S = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in WY.TESTS}
    ndays = sum(S[d]['days'] for d in WY.TESTS)
    O = pd.concat([pd.read_parquet(f'{OUTD}/ours_{d}.parquet') for d in WY.TESTS], ignore_index=True)
    T = pd.concat([pd.read_parquet(f'{OUTD}/theirs_{d}.parquet') for d in WY.TESTS], ignore_index=True)
    R = json.load(open(f'{OUTD}/order_harm_why.json'))['readings']
    O['mv'] = [WY.move_bucket(v) for v in O.dfv_trade.values]
    t5 = T[(T.who == 'top 5')]
    t5w = t5[t5.src == 'WS']
    th = {(k[:3], k[3]): (float(v.q.sum()), float(v.v1.sum())) for k, v in t5w.groupby(['phase', 'move', 'depth', 'typ'])}
    L = []; out = {}; bad = []; planted = 0
    for cfg in ('M0', 'M50', 'M250'):
        for arm in WY.ARMS:
            x = O[(O.cfg == cfg) & (O.arm == arm) & O.in60 & ~O.young & (O.src == 'WS')]
            ou = {(k[:3], k[3]): (float(v.q.sum()), float((v.q * v.v1).sum())) for k, v in x.groupby(['phase', 'mv', 'depth', 'typ'])}
            dd = within_split(ou, th); lab = label(dd)
            ref = R[f'W2 WS-timed {cfg} {arm}']['within']
            if abs(dd['within'] - ref) > 1e-3:
                bad.append((cfg, arm, dd['within'], ref))
            if abs(within_split(ou, th, swap=True)['within'] - ref) > 1e-3:
                planted += 1
            out[f'WT {cfg} {arm}'] = dict(label=lab, **dd)
            L.append(f'  {cfg:<4} {arm:<5}: WITHIN {dd["within"]:+.3f} = TYPE-MIX {dd["tmix"]:+.3f} + SAME-TYPE {dd["same"]:+.3f} + RESIDUAL '
                     f'{dd["resid"]:+.3f} -> {lab}')
    if bad:
        sys.exit(f'RECONCILIATION FAILED -> no output: WITHIN differs from ORDER-HARM WHY on {bad[:3]}')
    if planted != len(WY.ARMS) * 3:
        sys.exit(f'FALSIFIER FAILED -> no output: the swapped values matched on {len(WY.ARMS) * 3 - planted} cells')
    H = [f'ORDER-HARM WHY POST HOC (labelled post hoc; after W2), test days {", ".join(WY.TESTS)}; as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; '
         f'control: WITHIN equals ORDER-HARM WHY\'s W2 on all {len(WY.ARMS) * 3} WS cells, falsifier (values swapped) differs on {planted}', '',
         'WT WITHIN-STATE SPLIT (WS-timed, 1 s, ours w/o young): TYPE-MIX (a different STOP / THROUGH share in the same state) + SAME-TYPE']
    L = H + L + ['']
    # ST: STOP share and per-type value by phase group x depth (WS, M 50)
    L.append('ST STOP SHARE AND VALUE (WS-timed, 1 s; ours M 50 w/o young): phase group x depth: STOP share | STOP c/sh | THROUGH c/sh - '
             'top 5 (real) || ' + ' | '.join(WY.ARMS))
    t5w = t5w.assign(pg=t5w.phase.map(GROUP))
    O50 = O[(O.cfg == 'M50') & O.in60 & ~O.young & (O.src == 'WS')].assign(pg=lambda z: z.phase.map(GROUP))
    for pg in ('CALM', 'AFTER', 'ON'):
        for dep in ('at best', 'behind'):
            a = t5w[(t5w.pg == pg) & (t5w.depth == dep)]; qa = a.q.sum()
            sa = a[a.typ == 'STOP']; ha = a[a.typ == 'THROUGH']
            parts = [f'{sa.q.sum() / max(qa, 1e-9):5.1%} {sa.v1.sum() / max(sa.q.sum(), 1e-9):+6.3f} {ha.v1.sum() / max(ha.q.sum(), 1e-9):+6.3f}']
            for arm in WY.ARMS:
                x = O50[(O50.arm == arm) & (O50.pg == pg) & (O50.depth == dep)]; qx = x.q.sum()
                sx = x[x.typ == 'STOP']; hx = x[x.typ == 'THROUGH']
                parts.append(f'{sx.q.sum() / max(qx, 1e-9):5.1%} {(sx.q * sx.v1).sum() / max(sx.q.sum(), 1e-9):+6.3f} '
                             f'{(hx.q * hx.v1).sum() / max(hx.q.sum(), 1e-9):+6.3f}')
            L.append(f'  {pg:<5} {dep:<7}: ' + ' || '.join(parts))
    L.append('')
    # E2: per-share value by trade source
    L.append('E2 BY TRADE SOURCE (1 s, c/sh; sh/day): top 5 by phase group || ours at M 50, all ages, by timing class')
    t5g = t5.assign(pg=t5.phase.map(GROUP))
    for pg in ('CALM', 'AFTER', 'ON'):
        a = t5g[t5g.pg == pg]
        parts = [f'{s_} {a[a.src == s_].v1.sum() / max(a[a.src == s_].q.sum(), 1e-9):+6.3f} ({a[a.src == s_].q.sum() / ndays:7.0f})' for s_ in ('WS', 'E2')]
        L.append(f'  top 5 {pg:<5}: ' + ' | '.join(parts))
    for arm in WY.ARMS:
        x = O[(O.cfg == 'M50') & (O.arm == arm)]
        parts = []
        for c in WY.CLASSES:
            y = x[x.cls == c]
            if not len(y):
                continue
            parts.append(c + ' ' + ' '.join(f'{s_} {(y[y.src == s_].q * y[y.src == s_].v1).sum() / max(y[y.src == s_].q.sum(), 1e-9):+6.3f} '
                                             f'({y[y.src == s_].q.sum() / ndays:6.0f})' for s_ in ('WS', 'E2')))
        L.append(f'  {arm:<5}: ' + ' | '.join(parts))
    L.append('')
    # KT: OHQ0's kept-on fills by type and depth
    L.append('KT OHQ0 KEPT-ON FILLS AT M 50 (all trades, all ages): type x depth: sh/day | c/sh 1 s | c/sh 60.2 s mark')
    k = O[(O.cfg == 'M50') & (O.arm == 'OHQ0') & (O.cls == 'KEPT-ON')]
    for typ in TYPES:
        for dep in ('at best', 'behind', 'inside'):
            y = k[(k.typ == typ) & (k.depth == dep)]; qy = y.q.sum()
            L.append(f'  {typ:<7} {dep:<7}: {qy / ndays:7.0f} | {(y.q * y.v1).sum() / max(qy, 1e-9):+6.3f} | {(y.q * y.vend).sum() / max(qy, 1e-9):+6.3f}')
    txt = '\n'.join(L)
    open(f'{OUTD}/order_harm_why_posthoc.txt', 'w').write(txt + '\n')
    json.dump(dict(readings=out), open(f'{OUTD}/order_harm_why_posthoc.json', 'w'), indent=1, default=float)
    print(txt)
    print(f'done in {time.time() - T0:.0f} s')


if __name__ == '__main__':
    main()
