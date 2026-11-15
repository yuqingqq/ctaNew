"""STOP-PLAY (2026-11-15, session aea5bfb9): the STOP-MODEL's OWN decisions in the oracle's lane - how much of the ceiling the model captures.

USER (2026-11-15): "You mean we have the ceiling now, how about the model performance" / "Before playing model performance, can you check if
our model/data has any issues? Or preprocess data and refine the model if it has" (-> the dev-day diagnostics and the v4 fit come first;
this replay uses v4's predictions).

THE USER'S DESIGN (HANDOFF START HERE 2026-11-14 16:38Z, C18): every ladder level rests by default (rung 0 included); at every 50 ms step,
per side, the model predicts P(the best moves >= d + 1 ticks against the side) for each level d; pull ONLY the levels inside the predicted
stop; keep the rest; no guard trigger; hysteresis against flicker; joint ladder, client-known lane. Posting / re-entry belongs to the other
session's post-vs-wait model; until it lands a pulled level re-enters when its predicted risk is gone (here: below tau_off).
LANE (C11, C13, C14): STOP-MODEL's oracle lane unchanged - pf_stop_model.sim_joint_g (the client lane pf_order_harm_client_lane.sim_joint
plus per-rung entry gates): 6 rungs x 30 sh each side, no cap, REPOST_MS 0, decisions every 50 ms on client-known state; clocks (C6): Predict
book known at feed >= 13.2 ms (median ~53) + 2 ms decision, placement 40 ms, cancel 22 ms; Binance at event + 3.5 ms (P15).
ARMS (the same windows, M 0 / 50 / 250):
  KA        keep-all (rest every rung, never pull)
  OR{H}     the ceiling: act on the TRUE label (STOP-MODEL R701's oracle)
  MD{H}     the MODEL: a resting order d ticks behind the known best (depth < 0 -> 0; > 5 kept) is cancelled iff P^H_(d+1)(k) > tau_on;
            rung d is not (re-)posted while P^H_(d+1)(k) >= tau_off (hysteresis: tau_off = r x tau_on); a step without a model row keeps /
            allows (rest by default). P = v4's final predictions of the fold (calibration-day predictions for the choice, test-day for the test)
  SM{H}     the SHIFTED control: MD{H}'s rule and parameters on the same side's predictions of the day's previous window (same pull rate and
            clustering, no information about this window)
  LDG       the incumbent comparator: every rung pulled while the frozen guard is on (v <= -0.10), entries blocked (C18: a comparator only)
CHOICE (C16, chronological): per fold and H, (tau_on, r) over TAUS x HYST maximising the calibration day's whole-portfolio $ without fills on
  orders < 100 ms old at M 50 (ties -> the larger tau_on, then the larger r); applied unchanged to the test day at every clock.
READINGS DECLARED BEFORE THE RUN (5 fold test days; day rule >= 4 of 5; whole portfolio, C15):
 R800 MODEL-VS-KEEPALL (per H): MD - KA > 0 pooled AND on >= 4 days at M 50 AND at M 250, with AND without young fills -> BEATS-KEEP-ALL.
 R801 INFORMATION (per H): MD - SM > 0 pooled AND on >= 4 days at M 50 without young fills -> INFORMED; else NOT-SHOWN.
 R802 MODEL-VS-LDG (per H): MD - LDG > 0 pooled AND on >= 4 days at M 50 AND M 250 without young fills -> BEATS-LDG; < 0 the same way ->
      LOSES-TO-LDG; else NOT-SHOWN.
 R803 CAPTURE (printed): (MD - KA) / (OR - KA) per clock, pooled, without young fills.
 Printed beside (not readings): the WS / E2 split (C12), fills by order age < 100 / 100-500 / >= 500 ms (C1), the cancel-pending share of
 fills and its $ (C3), pulls and gated entries a day, quote availability per rung (C17), the chosen parameters per fold, and the test day's
 whole tau curve at M 50 (POST HOC: never a choice).
LIMITS: consumed days (development only; C8); replay rule M (queue place estimated, B1a, C5 / C10); replay volumes inflated (C4); E2 trade
anchors +-250 ms (C12); the model's short-horizon skill rests on Binance at event + 3.5 ms (P15).
Controls (any failure -> no output): pf_stop_model's selftests with planted errors; lane conservation violations 0 on every run; terminal
reconciliation of every arm; prediction keys aligned with their rows; a planted model gate that blocks rung 0 everywhere removes its fills
(selftest); MD with tau_on above every P equals KA fill for fill on every window (off-parity through the override, M 50).
Usage (run_light, PF_SLOT=claude): pf_stop_play.py selftest | cal <test_day> | test <test_day> | report
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_stop_model as M                                 # noqa: E402

TAUS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60)
HYST = (1.0, 0.6)
COMBOS = [(t, r) for t in TAUS for r in HYST]
AGE_B = ((0.0, 100.0), (100.0, 500.0), (500.0, M.INF))
TAU_OFFPAR = 2.0                                           # above every probability: MD must equal KA
MUT_GATE_IGNORE = MUT_DEC_ALWAYS = 0
OUT = M.OUTD
PV = os.environ.get('PF_PLAY_V', '4')                     # the model version played (v4, or v5 = v4 + adopted extra inputs)
PSZ = float(os.environ.get('PF_PLAY_SIZE', '30'))           # order size per rung (USER 2026-11-15: "we can use a larger size for testing or make it
VS = f'_v{PV}' + ('' if PSZ == 30.0 else f'_s{int(PSZ)}')    # configurable"); 30 sh = the oracle's size; other sizes write their own files


def pred_arrays(path, h):
    """Per (m, side): six arrays over the grid step k of v4's final P^H_j (NaN without a row) and the day's window order."""
    Q = pd.read_parquet(path, columns=['m', 'side', 'k'] + [M.pcol(h, j) for j in M.JS])
    out = {}; order = list(dict.fromkeys(Q.m.values.tolist()))
    for (m, s), g in Q.groupby(['m', 'side'], sort=False):
        kk = g.k.values; n = int(kk.max()) + 1; arr = []
        for j in M.JS:
            a = np.full(n, np.nan, np.float32); a[kk] = g[M.pcol(h, j)].values; arr.append(a)
        out[(int(m), int(s))] = arr
    return out, [int(x) for x in order]


def model_dec(P, best, s, tau):
    """Cancel a resting order d ticks behind the known best iff P_(d+1)(k) > tau (no row: kept; deeper than 5: kept)."""
    def dec(k, t, cur):
        if MUT_DEC_ALWAYS:
            return True
        b = best[k]
        if b < 0 or P is None:
            return False
        dpt = max(0, (b - cur['pt']) if s == 0 else (cur['pt'] - b))
        if dpt > 5:
            return False
        x = M.look(P[dpt], k)
        return bool(x == x and x > tau)
    return dec


def model_gate(P, d, tau_off):
    """Entry gate of rung d: closed while P_(d+1)(k) >= tau_off (no row: open)."""
    def gate(k):
        if MUT_GATE_IGNORE or P is None:
            return False
        x = M.look(P[d], k)
        return bool(x == x and x >= tau_off)
    return gate


def arm_value2(r, tf, mff, src):
    """M.arm_value plus splits: by order age at the fill (C1), by trade-anchor source WS / E2 (C12), cancel-pending fills (C3), and the share
    of the window's steps each rung has a live order (C17)."""
    v = M.arm_value(r, tf, mff); mark = M.CL.portfolio_terminal(r, tf, mff)['mark']
    age = [0.0] * len(AGE_B); ws = [0.0, 0.0]; ws_old = [0.0, 0.0]; pend = pend_sh = 0.0
    for (s, d), Fl in r['fills'].items():
        sg = 1.0 if s == 0 else -1.0
        for f in Fl:
            x = f[2] * sg * (mark - f[1]) / 100.0; a = float(f[4]); w_ = 0 if bool(src[s][int(f[0])]) else 1
            for i, (lo, hi) in enumerate(AGE_B):
                if lo <= a < hi:
                    age[i] += x
            ws[w_] += x
            if a >= M.YOUNG_MS:
                ws_old[w_] += x
            if f[5] == 'pend':
                pend += x; pend_sh += f[2]
    v.update(age_lt100=age[0], age_100_500=age[1], age_ge500=age[2], ws=ws[0], e2=ws[1], ws_old=ws_old[0], e2_old=ws_old[1], pend=pend,
             pend_shares=pend_sh, gated=float(sum(r['stat']['gated'])))
    return v


def live_share(r, ws, we):
    """Per rung: share of [ws, we] with a live (accepted, not refused, not done) order, both sides averaged."""
    out = []
    for d in range(M.NR):
        tot = 0.0
        for s in (0, 1):
            for (ta, te, _, ref) in r['orders'][(s, d)]:
                if not ref:
                    tot += max(0.0, min(te, we) - max(ta, ws))
        out.append(tot / (2.0 * (we - ws)))
    return out


def selftest():
    bad = []

    def chk(nm, c):
        if not c:
            bad.append(nm)
    W = M.toy_lane_window([200.0, 700.0], [50, 50], [130.0, 130.0]); n = W[0]['n']; aux = {0: [False] * n, 1: [False] * n}
    r0 = M.sim_joint_g(W, 'M', 0.0, None, aux, decide=None, gate=None, idle_only=0, mode='none', cooldown=M.CL.REPOST_MS)
    P1 = [np.full(n, 0.9, np.float32)] + [np.full(n, 0.0, np.float32)] * 5          # level 1 risky everywhere, deeper levels safe
    dec = {(0, d): model_dec(P1, W[0]['best'], 0, 0.5) for d in range(M.NR)}; gate = {(0, d): model_gate(P1, d, 0.5) for d in range(M.NR)}
    rm = M.sim_joint_g(W, 'M', 0.0, None, aux, decide=dec, gate=gate, idle_only=0, mode='none', cooldown=M.CL.REPOST_MS)
    chk(f'model gate blocks rung 0 {rm["stat"]["gated"]}', not rm['fills'][(0, 0)] and rm['stat']['gated'][0] > 0 and r0['fills'][(0, 0)])
    dec2 = {(0, d): model_dec(P1, W[0]['best'], 0, TAU_OFFPAR) for d in range(M.NR)}; gate2 = {(0, d): model_gate(P1, d, TAU_OFFPAR) for d in range(M.NR)}
    rp = M.sim_joint_g(W, 'M', 0.0, None, aux, decide=dec2, gate=gate2, idle_only=0, mode='none', cooldown=M.CL.REPOST_MS)
    chk('off-parity (tau above every P) = keep-all', rp['fills'] == r0['fills'] and rp['endpos'] == r0['endpos'])
    Pn = [np.full(n, np.nan, np.float32)] * 6
    rn = M.sim_joint_g(W, 'M', 0.0, None, aux, decide={(0, d): model_dec(Pn, W[0]['best'], 0, 0.0) for d in range(M.NR)},
                       gate={(0, d): model_gate(Pn, d, 0.0) for d in range(M.NR)}, idle_only=0, mode='none', cooldown=M.CL.REPOST_MS)
    chk('no model row = rest by default', rn['fills'] == r0['fills'])
    src = {0: np.array([True, False]), 1: np.zeros(0, bool)}
    v = arm_value2(r0, W[0]['tf_mff'][0], W[0]['tf_mff'][1], src)
    chk(f'value splits {v}', abs(v['ws'] + v['e2'] - v['pnl']) < 1e-9 and abs(v['age_lt100'] + v['age_100_500'] + v['age_ge500'] - v['pnl']) < 1e-9
        and abs(v['ws_old'] + v['e2_old'] - v['pnl_old']) < 1e-9)
    ls = live_share(r0, float(W[0]['tg'][0]), float(W[0]['te']))
    chk(f'live share {ls}', all(0.0 <= x <= 1.0 for x in ls) and ls[0] > 0)
    return bad


def all_selftests():
    M.all_selftests()
    bad = selftest()
    if bad:
        sys.exit(f'STOP-PLAY SELFTEST FAILED: {bad}')
    out = {}
    for nm in ('MUT_GATE_IGNORE', 'MUT_DEC_ALWAYS'):
        globals()[nm] = 1
        try:
            out[nm] = len(selftest()) > 0
        finally:
            globals()[nm] = 0
    if not all(out.values()):
        sys.exit(f'STOP-PLAY PLANTED ERROR NOT CAUGHT: {out}')
    print(f'stop-play selftests passed; planted errors caught: {out}', flush=True)


def window_src(D, W, WSg, TKS):
    """Per side: the WS-timed flag of every leg row of the window (C12), in the lane's leg order."""
    m = int(W[0]['m']); src = {}
    for s in (0, 1):
        if (m, s) in D['LI']:
            l = D['LT'].iloc[D['LI'][(m, s)]]
            o_ = np.lexsort(((-l.ptick.values if s == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
            src[s] = WSg.reindex(lrow).values.astype(bool)
        else:
            src[s] = np.zeros(0, bool)
    return src


def play_day(test, phase):
    """phase 'cal': the fold's calibration day, every (tau_on, r) per H at M 50 (the choice); 'test': the test day, every arm at every clock,
    plus the tau curve at M 50 with the chosen r (post hoc, printed)."""
    all_selftests()
    if test not in M.TESTS:
        sys.exit(f'REFUSED: {test} is not a fold test day')
    day = M.FOLDS[test][1] if phase == 'cal' else test
    fj = json.load(open(f'{OUT}/fit_{test}{VS}.json'))
    if fj['version'] != PV:
        sys.exit(f'REFUSED: not a v{PV} fit')
    T0 = time.time(); D = M.K.load_day(day); cut = M.boundary_cut(day)
    tkf = pd.read_parquet(f'{M.K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values
    LT = D['LT']; ix = LT.index.values; grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WSg = pd.Series(M.TS.ws_groups(LT.how.values, grp), index=LT.index); TKS = pd.Series(grp, index=LT.index)
    path = f'{OUT}/pred_fold{test}_{day}{VS}.parquet'
    PA = {h: pred_arrays(path, h) for h in M.HS}; order = PA[M.HS[0]][1]; pos_ = {m: i for i, m in enumerate(order)}
    LA = M.label_arrays(day)[0] if phase == 'test' else None
    choice = json.load(open(f'{OUT}/play_cal_{test}{VS}.json'))['choice'] if phase == 'test' else None
    st = dict(windows=0, boundary_windows=0, runs=0, viol={v_: 0 for v_ in M.J3.VIOL}, offpar_windows=0, offpar_differ=0, no_pred=0)
    val = {}; live = {}

    def add(cfg, arm, v_, ls=None):
        a = val.setdefault(cfg, {}).setdefault(arm, {})
        for k_, x in v_.items():
            a[k_] = a.get(k_, 0.0) + x
        if ls is not None:
            b = live.setdefault(cfg, {}).setdefault(arm, [0.0] * M.NR)
            for i, x in enumerate(ls):
                b[i] += x

    def run(W, q, shf, dec=None, gate=None, aux=None):
        n = W[0]['n']; aux = aux or {0: [False] * n, 1: [False] * n}
        if dec is None and gate is None:
            r = M.CL.sim_joint(W, q, shf, None, aux, decide=None, idle_only=0, size=PSZ, mode='none', cooldown=M.CL.REPOST_MS)
        else:
            r = M.sim_joint_g(W, q, shf, None, aux, decide=dec, gate=gate, idle_only=0, size=PSZ, mode='none', cooldown=M.CL.REPOST_MS)
        st['runs'] += 1
        for v_ in M.J3.VIOL:
            st['viol'][v_] += r['stat']['viol'][v_]
        return r

    def md(W, P, tau, rr):
        dec = {(s, d): model_dec(P[s], W[s]['best'], s, tau) for s in (0, 1) for d in range(M.NR)}
        gate = {(s, d): model_gate(P[s], d, tau * rr) for s in (0, 1) for d in range(M.NR)}
        return dec, gate

    for W in M.P1.window_pairs(D):
        m = int(W[0]['m'])
        if W[0]['ws'] < cut:
            st['boundary_windows'] += 1; continue
        st['windows'] += 1; tf, mff = W[0]['tf_mff']; src = window_src(D, W, WSg, TKS); ws, we = float(W[0]['tg'][0]), float(W[0]['te'])
        prv = order[pos_[m] - 1] if m in pos_ else None
        if m not in pos_:
            st['no_pred'] += 1
        if phase == 'cal':
            q, shf = 'M', 50.0
            for h in M.HS:
                P = {s: PA[h][0].get((m, s)) for s in (0, 1)}
                for (tau, rr) in COMBOS:
                    r = run(W, q, shf, *md(W, P, tau, rr)); add('M50', f'MD{h}|{tau}|{rr}', arm_value2(r, tf, mff, src))
            if st['windows'] % 40 == 0:
                M.mem_guard(f'play cal {day}')
            continue
        for q, shf in M.CFGS:
            cfg = f'{q}{int(shf)}'; n = W[0]['n']
            rk = run(W, q, shf); add(cfg, 'KA', arm_value2(rk, tf, mff, src), live_share(rk, ws, we))
            vg = {s: (np.asarray(W[s]['vkl'], float) <= M.GUARD).tolist() for s in (0, 1)}
            rl = run(W, q, shf, aux=vg); add(cfg, 'LDG', arm_value2(rl, tf, mff, src), live_share(rl, ws, we))
            for h in M.HS:
                arr = {s: LA.get((m, s), {}).get(h) for s in (0, 1)}
                dec = {(s, d): M.oracle_dec(arr[s], W[s]['best'], s) for s in (0, 1) for d in range(M.NR)}
                gate = {(s, d): M.oracle_gate(arr[s], d) for s in (0, 1) for d in range(M.NR)}
                ro = run(W, q, shf, dec, gate); add(cfg, f'OR{h}', arm_value2(ro, tf, mff, src), live_share(ro, ws, we))
                tau, rr = choice[str(h)]
                P = {s: PA[h][0].get((m, s)) for s in (0, 1)}; Ps = {s: (PA[h][0].get((prv, s)) if prv is not None else None) for s in (0, 1)}
                r_ = run(W, q, shf, *md(W, P, tau, rr)); add(cfg, f'MD{h}', arm_value2(r_, tf, mff, src), live_share(r_, ws, we))
                rs = run(W, q, shf, *md(W, Ps, tau, rr)); add(cfg, f'SM{h}', arm_value2(rs, tf, mff, src), live_share(rs, ws, we))
                if cfg == 'M50':
                    if h == M.HP:
                        rz = run(W, q, shf, *md(W, P, TAU_OFFPAR, 1.0)); st['offpar_windows'] += 1
                        st['offpar_differ'] += int(rz['fills'] != rk['fills'] or rz['endpos'] != rk['endpos'])
                    for t2 in TAUS:
                        if t2 != tau:
                            r2 = run(W, q, shf, *md(W, P, t2, rr)); add('M50', f'CURVE{h}|{t2}', arm_value2(r2, tf, mff, src))
                    add('M50', f'CURVE{h}|{tau}', arm_value2(r_, tf, mff, src))
        if st['windows'] % 40 == 0:
            M.mem_guard(f'play test {day}')
    if any(st['viol'].values()) or st['offpar_differ']:
        sys.exit(f'REFUSED (no output for {day}): lane violations / off-parity differences {st}')
    st['seconds'] = round(time.time() - T0, 1); res = dict(test=test, day=day, phase=phase, stat=st, value=val, live=live)
    if phase == 'cal':
        ch = {}
        for h in M.HS:
            best = max(COMBOS, key=lambda c: (val['M50'][f'MD{h}|{c[0]}|{c[1]}']['pnl_old'], c[0], c[1]))
            ch[str(h)] = list(best)
        res['choice'] = ch
        json.dump(res, open(f'{OUT}/play_cal_{test}{VS}.json', 'w'), indent=1, default=float)
        print(f'play cal {test} (day {day}): windows {st["windows"]}, runs {st["runs"]}, {st["seconds"]} s; choice {ch}; M50 w/o young: '
              + ', '.join(f'H{h} {val["M50"][f"MD{h}|{ch[str(h)][0]}|{ch[str(h)][1]}"]["pnl_old"]:+.0f}' for h in M.HS), flush=True)
    else:
        res['choice'] = choice
        json.dump(res, open(f'{OUT}/play_{test}{VS}.json', 'w'), indent=1, default=float)
        print(f'play test {test}: windows {st["windows"]}, runs {st["runs"]}, off-parity {st["offpar_windows"]} windows / {st["offpar_differ"]} differ, '
              f'{st["seconds"]} s; M50 w/o young: ' + ', '.join(f'{a} {x["pnl_old"]:+.0f}' for a, x in val['M50'].items() if not a.startswith('CURVE')), flush=True)


def report():
    all_selftests()
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    PJ = {t: json.load(open(f'{OUT}/play_{t}{VS}.json')) for t in M.TESTS}; CJ = {t: json.load(open(f'{OUT}/play_cal_{t}{VS}.json')) for t in M.TESTS}
    T = M.TESTS; DR = M.DAY_RULE; R_ = {}; L = []
    L.append(f'STOP-PLAY (the v{PV} model\'s own decisions in the oracle\'s lane; 6 rungs x {PSZ:.0f} sh both sides, no guard), 5 chronological test days {", ".join(T)}; '
             f'as of {asof}')
    L.append('chosen (tau_on, r) per fold on its calibration day (M 50, w/o young): ' + '; '.join(f'{t}: ' + ' '.join(f'H{h} {CJ[t]["choice"][str(h)]}' for h in M.HS) for t in T))
    per = lambda cfg, a, key: np.array([PJ[t]['value'][cfg][a][key] for t in T])          # noqa: E731
    L.append('WHOLE PORTFOLIO $ a day (mean of 5 days; C15): arm | M0 all / w/o young | M50 all / w/o young | M250 all / w/o young')
    arms = ['KA', 'LDG'] + [f'{a}{h}' for h in M.HS for a in ('OR', 'MD', 'SM')]
    for a in arms:
        L.append(f'  {a:7s} ' + ' | '.join(f'{per(c, a, "pnl").mean():+9.1f} / {per(c, a, "pnl_old").mean():+9.1f}' for c in ('M0', 'M50', 'M250')))
    for h in M.HS:
        md, ka, sm, ld, orc = f'MD{h}', 'KA', f'SM{h}', 'LDG', f'OR{h}'
        f800 = []
        for c in ('M50', 'M250'):
            for key in ('pnl', 'pnl_old'):
                x, y = per(c, md, key), per(c, ka, key)
                if not (x.mean() > y.mean() and M.day_rule(x, y)[0] >= DR):
                    f800.append(f'{c} {key}')
        R_[f'R800_H{h}'] = 'BEATS-KEEP-ALL' if not f800 else 'NOT-SHOWN: ' + ', '.join(f800)
        x, y = per('M50', md, 'pnl_old'), per('M50', sm, 'pnl_old')
        R_[f'R801_H{h}'] = 'INFORMED' if (x.mean() > y.mean() and M.day_rule(x, y)[0] >= DR) else 'NOT-SHOWN'
        bt = all(per(c, md, 'pnl_old').mean() > per(c, ld, 'pnl_old').mean() and M.day_rule(per(c, md, 'pnl_old'), per(c, ld, 'pnl_old'))[0] >= DR
                 for c in ('M50', 'M250'))
        ls = all(per(c, md, 'pnl_old').mean() < per(c, ld, 'pnl_old').mean() and M.day_rule(per(c, md, 'pnl_old'), per(c, ld, 'pnl_old'))[1] >= DR
                 for c in ('M50', 'M250'))
        R_[f'R802_H{h}'] = 'BEATS-LDG' if bt else ('LOSES-TO-LDG' if ls else 'NOT-SHOWN')
        cap = {c: float((per(c, md, 'pnl_old').mean() - per(c, ka, 'pnl_old').mean()) / (per(c, orc, 'pnl_old').mean() - per(c, ka, 'pnl_old').mean()))
               for c in ('M0', 'M50', 'M250')}
        R_[f'R803_H{h}'] = cap
        L.append(f'H {h}: R800 {R_[f"R800_H{h}"]}; R801 {R_[f"R801_H{h}"]}; R802 {R_[f"R802_H{h}"]}; R803 capture of the ceiling (w/o young) '
                 + ', '.join(f'{c} {v:+.1%}' for c, v in cap.items()))
        L.append(f'   per day M50 w/o young MD | KA | SM | LDG | OR: ' + '; '.join(f'{t[4:]} {PJ[t]["value"]["M50"][md]["pnl_old"]:+.0f} | '
                 f'{PJ[t]["value"]["M50"][ka]["pnl_old"]:+.0f} | {PJ[t]["value"]["M50"][sm]["pnl_old"]:+.0f} | {PJ[t]["value"]["M50"][ld]["pnl_old"]:+.0f} | '
                 f'{PJ[t]["value"]["M50"][orc]["pnl_old"]:+.0f}' for t in T))
    L.append('SPLITS at M 50 (mean $ a day): arm | WS / E2 (all) | WS / E2 (w/o young) | by order age < 100 / 100-500 / >= 500 ms | cancel-pending $ (shares) '
             '| pulls | gated entries | live share per rung 0..5')
    for a in arms:
        g = lambda k_: np.mean([PJ[t]['value']['M50'][a][k_] for t in T])           # noqa: E731
        lv = np.mean([np.array(PJ[t]['live']['M50'][a]) / PJ[t]['stat']['windows'] for t in T], axis=0)
        L.append(f'  {a:7s} {g("ws"):+8.1f} / {g("e2"):+8.1f} | {g("ws_old"):+8.1f} / {g("e2_old"):+8.1f} | {g("age_lt100"):+8.1f} / {g("age_100_500"):+8.1f} / '
                 f'{g("age_ge500"):+8.1f} | {g("pend"):+7.1f} ({g("pend_shares"):.0f}) | {g("pulls"):8.0f} | {g("gated"):8.0f} | ' + ' '.join(f'{x:.2f}' for x in lv))
    L.append('TAU CURVE at M 50 w/o young, test days (POST HOC, printed - never a choice): H | tau_on: $ a day')
    for h in M.HS:
        L.append(f'  H {h}: ' + ' '.join(f'{t2}: {np.mean([PJ[t]["value"]["M50"][f"CURVE{h}|{t2}"]["pnl_old"] for t in T]):+.0f}' for t2 in TAUS))
    L.append('(development only: consumed days, C8 / C16; replay rule M, volumes inflated C4 / C5 / C10; E2 anchors +-250 ms C12; P15)')
    L.append(f'READINGS {json.dumps(R_)}')
    json.dump(dict(asof=asof, readings=R_, value={t: PJ[t]['value'] for t in T}, live={t: PJ[t]['live'] for t in T},
                   choice={t: CJ[t]['choice'] for t in T}, stat={t: PJ[t]['stat'] for t in T}), open(f'{OUT}/stop_play{VS}.json', 'w'), indent=1, default=float)
    open(f'{OUT}/stop_play{VS}.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest':
        all_selftests()
    elif cmd in ('cal', 'test'):
        play_day(sys.argv[2], cmd)
    elif cmd == 'report':
        report()
    else:
        sys.exit(f'unknown command {cmd}')
