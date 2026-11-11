"""B13b (2026-11-11, USER "Then check if these top makers pull their quotes because of these moves, do these features
somehow explain their behaviors"). Read-only, descriptive. 220 consumed 11-08 cohort windows, first minute; G = 1 day:
point estimates only.

Pressure signal: B13a's fold models (pf_b13a_pressure_model.py, trained on 6 of 10-23..11-03 each, none on 11-08) applied
to the cohort's step5 rows of 11-08 (same 40 columns): pred_base / pred_emph = the mean of the three folds' BASE / EMPH
predictions of Predict's mid change over the next 1 s (c), on the 50 ms grid, read at t - 50 ms (latest grid point, <= 100
ms old). Adverse pressure for a maker on side s: C_adv = -sg x pred (sg = +1 bids, -1 asks; > 0 = predicted to move
against that maker). Favourable for a taker: C_fav = tsign x pred (B12e's sign). Raw features (signed the same way, no
fitting): pf_imb1, bn_imb, bn_flow250, pf_dm250; and bn_ntr1000 (unsigned).
Behaviours (as B12d / B12e, same builds):
 staying   B12d's re-hit opportunities (absorbed fill at the best, the next same-price taker order within 500 ms): stay = the
           maker is filled again; leave = it is not.
 presence  the top 5's share of at-best fill shares (all makers except ours) by tercile of C_adv at the fill.
 taking    B12e's SMART taker orders against RANDOM times (same seed): how well C_fav separates them (AUC).

Readings DECLARED BEFORE THE FIRST RUN (EMPH composite unless stated):
 R100 staying   the top 5's stay rate in the most adverse tercile of C_adv minus in the least adverse (terciles over all
                opportunities): PRESSURE-DRIVES-STAYING if <= -15 pp; else NO-PRESSURE-STAYING. O's change reported beside it.
 R101 presence  the top 5's share in the most adverse tercile / in the least adverse: THEY-PULL-UNDER-PRESSURE if <= 0.8;
                else NO-PULL-SIGNAL.
 R102 taking    AUC of C_fav, SMART against RANDOM: FEATURES-EXPLAIN-TAKES if >= 0.75; PARTLY if >= 0.60; else
                NOT-EXPLAINED.
 R103 leaving   AUC of C_adv for leave against stay, top 5: TOP-LEAVE-ON-PRESSURE if >= 0.60 and >= O's + 0.05; else
                NOT-PRESSURE-LEAVING.
 R104 model     AUC(EMPH) - AUC(BASE) for the takes: EMPH-EXPLAINS-BETTER if >= +0.03; else SIMILAR.
Descriptive: the same rates and AUCs per raw feature and for BASE; stay rate by tercile for top 5 and O; presence by
tercile; AUCs of each raw feature for the takes.
Controls (any failure -> no table): B12e's controls; the AUC on planted scores (perfect -> 1, reversed -> 0, random ->
~0.5); terciles and sign conventions on planted rows; the label rules (NaN reads n/a); K-checks: event counts equal
b12e.json's and re-hit opportunities equal b12d.json's; the B13a model files exist and their feature names match.
Limits fixed in advance: one consumed Sunday (the day the pressure idea came from, so descriptive only); cancels are not
visible (stay / leave is read from re-hits, presence from fill shares); the composites are predictions from models
trained on other days.
Usage: predictfun/research/run_light.sh b13b_top_behaviour -- python3 -m predictfun.research.pf_b13b_top_behaviour <b13a_dir> <out_dir>
       python3 -m predictfun.research.pf_b13b_top_behaviour --selftest
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import pf_b12_makers as B
from . import pf_b12b_reason as BB
from . import pf_b12d_bursts as DB
from . import pf_b12e_smart_takes as E1
from . import pf_b12e2_binance_raw as E2
from . import pf_b13a_pressure_model as M

ROWS08 = f'{BB.RP}/research/step5/20261108/rows_rl_d2.parquet'
B12D = f'{BB.LV}/b12d_20261111/b12d.json'
NAN = float('nan')
RAW = ['pf_imb1', 'bn_imb', 'bn_flow250', 'pf_dm250']


def auc(score, lab):
    ok = np.isfinite(score); s = score[ok]; y = lab[ok].astype(bool)
    n1 = int(y.sum()); n0 = int((~y).sum())
    if n1 == 0 or n0 == 0:
        return NAN
    rk = pd.Series(s).rank(method='average').values
    return float((rk[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def terciles(x):
    e = np.nanquantile(x, [1 / 3, 2 / 3]); return np.where(~np.isfinite(x), -1, np.searchsorted(e, x, side='right')), e


def label_r100(d):
    return 'n/a' if not BB.fin(d) else ('PRESSURE-DRIVES-STAYING' if d <= -0.15 else 'NO-PRESSURE-STAYING')


def label_r101(r):
    return 'n/a' if not BB.fin(r) else ('THEY-PULL-UNDER-PRESSURE' if r <= 0.8 else 'NO-PULL-SIGNAL')


def label_r102(a):
    return 'n/a' if not BB.fin(a) else ('FEATURES-EXPLAIN-TAKES' if a >= 0.75 else ('PARTLY' if a >= 0.60 else 'NOT-EXPLAINED'))


def label_r103(aT, aO):
    return 'n/a' if not BB.fin(aT, aO) else ('TOP-LEAVE-ON-PRESSURE' if (aT >= 0.60 and aT >= aO + 0.05) else 'NOT-PRESSURE-LEAVING')


def label_r104(d):
    return 'n/a' if not BB.fin(d) else ('EMPH-EXPLAINS-BETTER' if d >= 0.03 else 'SIMILAR')


def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    oke, repe = E1.controls(); chk('B12e_controls', oke)
    rng = np.random.default_rng(5); lab = rng.random(4000) < 0.3; s = rng.random(4000)
    chk('H1_auc', abs(auc(lab.astype(float), lab) - 1.0) < 1e-12 and abs(auc(-lab.astype(float), lab)) < 1e-12 and abs(auc(s, lab) - 0.5) < 0.05)
    t, e = terciles(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan]))
    sg = np.where(np.array([0, 1]) == 0, 1.0, -1.0)
    chk('H2_terciles_sign', list(t) == [0, 0, 1, 1, 2, 2, -1] and list(-sg * np.array([0.5, 0.5])) == [-0.5, 0.5])
    chk('H3_rules', label_r100(-0.2) == 'PRESSURE-DRIVES-STAYING' and label_r100(-0.1) == 'NO-PRESSURE-STAYING' and label_r101(0.7) == 'THEY-PULL-UNDER-PRESSURE'
        and label_r101(0.9) == 'NO-PULL-SIGNAL' and label_r102(0.8) == 'FEATURES-EXPLAIN-TAKES' and label_r102(0.65) == 'PARTLY' and label_r102(0.55) == 'NOT-EXPLAINED'
        and label_r103(0.65, 0.55) == 'TOP-LEAVE-ON-PRESSURE' and label_r103(0.65, 0.62) == 'NOT-PRESSURE-LEAVING' and label_r104(0.05) == 'EMPH-EXPLAINS-BETTER'
        and label_r104(0.01) == 'SIMILAR' and label_r100(NAN) == 'n/a' and label_r102(NAN) == 'n/a')
    rep['B12e_controls_detail'] = repe
    return ok, rep


def grid_preds(b13a):
    """Mean BASE / EMPH predictions of B13a's three fold models on 11-08's 50 ms grid."""
    models = {}
    for kind in ('base', 'emph'):
        models[kind] = [lgb.Booster(model_file=f'{b13a}/model_{kind}_fold{k}.txt') for k in M.FOLDS]
        names = models[kind][0].feature_name()
        if names != (M.BASE if kind == 'base' else M.EMPH):
            raise ValueError(f'B13a {kind} model features differ from the spec')
    cols = list(dict.fromkeys(['mid', 'tau'] + M.BASE + ['bn_ntr1000']))
    r = pd.read_parquet(ROWS08, columns=cols).sort_values(['mid', 'tau'], kind='stable').reset_index(drop=True)
    ex = M.engineer(r); Xb = r[M.BASE].values.astype(np.float32); Xe = np.concatenate([Xb, ex.values], axis=1)
    pb = np.mean([m.predict(Xb, num_threads=1) for m in models['base']], axis=0); pe = np.mean([m.predict(Xe, num_threads=1) for m in models['emph']], axis=0)
    G = {}
    for m_, ii in r.groupby('mid', sort=False).indices.items():
        G[int(m_)] = dict(tau=r.tau.values[ii].astype(float), base=pb[ii], emph=pe[ii], **{c: r[c].values[ii].astype(float) for c in RAW + ['bn_ntr1000']})
    return G


def look(G, mids, t, key):
    out = np.full(len(t), np.nan)
    for m_ in np.unique(mids):
        ii = np.nonzero(mids == m_)[0]; g = G.get(int(m_))
        if g is None:
            continue
        j = np.searchsorted(g['tau'], t[ii], side='right') - 1; jj = np.clip(j, 0, None)
        out[ii] = np.where((j >= 0) & (t[ii] - g['tau'][jj] <= B.FRESH), g[key][jj], np.nan)
    return out


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps({k: v for k, v in rep.items() if not k.endswith('detail')}, indent=1)); print('selftest', 'PASS' if ok else 'FAIL')
        sys.exit(0 if ok else 1)
    b13a, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    ok, ctl = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    b12e = json.load(open(E2.B12E)); b12d = json.load(open(B12D))
    X = BB.build(); R = X['R']; W0 = X['W0']; OWN = X['OWN']
    k0 = BB.reconcile(R, X['days'])
    if not k0['ok']:
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    G = grid_preds(b13a)
    # ---- events exactly as B12e ----
    Mt = pd.read_parquet(E1.MATCH, columns=['mid', 'exec_s', 'taker', 'maker', 'ptick', 'shares'])
    Mt = Mt[Mt.mid.isin(W0)].copy(); Mt['exec_s'] = Mt.exec_s.astype('int64'); Mt['sh6'] = Mt.shares.round(6)
    key = ['mid', 'exec_s', 'ptick', 'sh6']
    Mu = Mt.merge(Mt.groupby(key).size().rename('n').reset_index().query('n == 1')[key], on=key)
    L = R[['mid', 'exec_s', 'ptick', 'shares', 'signer', 'grp']].copy(); L['exec_s'] = L.exec_s.astype('int64'); L['sh6'] = L.shares.round(6)
    Lu = L.merge(L.groupby(key).size().rename('nl').reset_index().query('nl == 1')[key], on=key)
    J = Lu.merge(Mu[key + ['taker', 'maker']], on=key)
    addr = J.groupby('signer').maker.agg(lambda s: s.value_counts().index[0]).str.lower()
    tk = J.groupby('grp').taker.agg(lambda s: s.value_counts().index[0]).str.lower()
    if not (OWN in addr.index and addr[OWN] == str(json.load(open(E1.WHO))['our_address']).lower()):
        sys.exit('CONTROL FAILED (KJ): own address')
    top5 = sorted(int(s) for s in R.signer[R.g == 'top 5'].unique()); smart_addr = {addr[s]: s for s in top5 if s in addr.index}
    R['taker'] = tk.reindex(R.grp.values).values
    A = R[R.val_ok & (R.status == 'ok') & (R.depth == 0) & R.taker.notna()]
    TO = A.groupby('grp').agg(mid=('mid', 'first'), side=('side', 'first'), anc=('anc', 'first'), taker=('taker', 'first')).reset_index()
    TO['smart'] = TO.taker.isin(set(smart_addr))
    rng = np.random.default_rng(E1.SEED); Sm = TO[TO.smart]
    rr = pd.DataFrame(dict(mid=np.repeat(Sm.mid.values, E1.K_RAND), side=np.repeat(Sm.side.values, E1.K_RAND)))
    rr['anc'] = rr.mid.map(W0).values + rng.uniform(1050.0, 59000.0, len(rr))
    EV = pd.concat([Sm[['mid', 'side', 'anc']].assign(kind='SMART'), rr.assign(kind='RANDOM')], ignore_index=True)
    counts = dict(SMART=int((EV.kind == 'SMART').sum()), RANDOM=int((EV.kind == 'RANDOM').sum()))
    if counts['SMART'] != b12e['counts']['SMART'] or counts['RANDOM'] != b12e['counts']['RANDOM']:
        sys.exit(f'CONTROL FAILED (K-check events) -> no table: {counts}')
    tsg = E1.tsign(EV.side.values); te = EV.anc.values - 50.0; lab = (EV.kind == 'SMART').values
    take = {}
    for k in ('base', 'emph'):
        take[k] = auc(tsg * look(G, EV.mid.values, te, k), lab)
    for c in RAW:
        take[c] = auc(tsg * look(G, EV.mid.values, te, c), lab)
    take['bn_ntr1000 (unsigned)'] = auc(look(G, EV.mid.values, te, 'bn_ntr1000'), lab)
    # ---- staying: B12d's re-hit opportunities ----
    V = R[R.val_ok]
    AB = V[(V.status == 'ok') & (V.depth == 0) & (V.sweep == 0) & (V.n_lvl == 1)].copy()
    gb = R[R.anc.notna()].groupby(['mid', 'side', 'ptick', 'grp'], sort=False)
    lv = gb.anc.first().to_frame('anc').join(gb.signer.apply(lambda s: frozenset(int(v) for v in s)).rename('signers')).reset_index()
    lv = lv.sort_values(['mid', 'side', 'ptick', 'anc', 'grp'], kind='stable'); nxt = {}
    for kk, ii in lv.groupby(['mid', 'side', 'ptick'], sort=False).indices.items():
        nxt[(int(kk[0]), int(kk[1]), int(kk[2]))] = (lv.anc.values[ii], lv.grp.values[ii], list(lv.signers.values[ii]))
    keys = [(int(a), int(b), int(c)) for a, b, c in zip(AB.mid.values, AB.side.values, AB.ptick.values)]
    has, gap, hit = DB.rehit(keys, AB.anc.values, AB.grp.values, [int(s) for s in AB.signer.values], nxt)
    AB['has'] = has; AB['hit'] = hit
    if int(AB.has.sum()) != b12d['rehit']['opportunities']:
        sys.exit(f"CONTROL FAILED (K-check re-hit) -> no table: {int(AB.has.sum())} against {b12d['rehit']['opportunities']}")
    OP = AB[AB.has & AB.g.isin(['top 5', '6-20', 'rest'])].copy(); sgo = np.where(OP.side.values == 0, 1.0, -1.0); to = OP.anc.values - 50.0
    stay = {}
    for k in ['emph', 'base'] + RAW:
        cadv = -sgo * look(G, OP.mid.values, to, k); tcl, edges = terciles(cadv); row = dict(edges=[float(x) for x in edges])
        for gname, gm in (('top 5', OP.g.values == 'top 5'), ('O (ranks >= 6)', OP.g.values != 'top 5')):
            rates = [float(OP.hit.values[gm & (tcl == i)].mean()) if (gm & (tcl == i)).any() else NAN for i in range(3)]
            row[gname] = dict(rate_by_tercile=rates, n_by_tercile=[int((gm & (tcl == i)).sum()) for i in range(3)], delta_high_minus_low=rates[2] - rates[0],
                              auc_leave=auc(cadv[gm], ~OP.hit.values[gm]))
        stay[k] = row
    # ---- presence: top-5 share of at-best fills by adverse tercile ----
    P = V[(V.status == 'ok') & (V.depth == 0) & (V.g != 'ours')].copy(); sgp = np.where(P.side.values == 0, 1.0, -1.0)
    pres = {}
    for k in ['emph', 'base'] + RAW:
        cadv = -sgp * look(G, P.mid.values, P.anc.values - 50.0, k); tcl, edges = terciles(cadv); sh = P.shares.values; isT = P.g.values == 'top 5'
        shares = [float(sh[isT & (tcl == i)].sum() / sh[tcl == i].sum()) if sh[tcl == i].sum() > 0 else NAN for i in range(3)]
        nets = {g_: [B.wmean(sh[(P.g.values == g_) & (tcl == i)], P.v1.values[(P.g.values == g_) & (tcl == i)]) for i in range(3)] for g_ in ('top 5', '6-20', 'rest')}
        pres[k] = dict(top5_share_by_tercile=shares, ratio_high_low=shares[2] / shares[0] if BB.fin(shares[0], shares[2]) and shares[0] > 0 else NAN, net1_by_tercile=nets)
    readings = dict(R100=label_r100(stay['emph']['top 5']['delta_high_minus_low']), R101=label_r101(pres['emph']['ratio_high_low']),
                    R102=label_r102(take['emph']), R103=label_r103(stay['emph']['top 5']['auc_leave'], stay['emph']['O (ranks >= 6)']['auc_leave']),
                    R104=label_r104(take['emph'] - take['base']))
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08); the day the pressure idea came from', controls=ctl, K0=k0['ok'],
               counts=counts, opportunities=int(len(OP)), take_auc=take, staying=stay, presence=pres, readings=readings)
    json.dump(res, open(f'{out}/b13b.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not BB.fin(v) else f'{v:+.3f}'
    pc = lambda v: 'n/a' if not BB.fin(v) else f'{v * 100:.1f}%'
    Lt = ['B13b do the pressure features explain the top makers\' behaviour? (11-08; composites from B13a models trained on other days)',
          f"Controls: {json.dumps({k: v for k, v in ctl.items() if not k.endswith('detail')})}; events {counts}; re-hit opportunities (top 5 + O) {len(OP)}", '',
          'Taking: AUC, SMART against RANDOM (0.5 = no separation): ' + '; '.join(f"{k} {v:.3f}" for k, v in take.items()), '',
          'Staying after an absorbed fill, by tercile of adverse pressure (low / mid / high): stay rate (n) | high - low | AUC for leaving']
    for k, row in stay.items():
        for gname in ('top 5', 'O (ranks >= 6)'):
            x = row[gname]
            Lt.append(f"{k} | {gname}: " + ' / '.join(f"{pc(r_)} ({n_})" for r_, n_ in zip(x['rate_by_tercile'], x['n_by_tercile'])) + f" | {pc(x['delta_high_minus_low'])} | {x['auc_leave']:.3f}")
    Lt += ['', 'Presence: the top 5\'s share of at-best fill shares by tercile of adverse pressure (low / mid / high) | high / low | net 1 s top 5 ; 6-20 ; rest by tercile']
    for k, x in pres.items():
        Lt.append(f"{k}: " + ' / '.join(pc(v) for v in x['top5_share_by_tercile']) + f" | {x['ratio_high_low']:.2f} | "
                  + ' ; '.join(f"{g_} " + '/'.join(f(v) for v in vs) for g_, vs in x['net1_by_tercile'].items()))
    Lt += ['', 'Readings: ' + json.dumps(readings)]
    open(f'{out}/b13b.txt', 'w').write('\n'.join(Lt) + '\n'); print('\n'.join(Lt))


if __name__ == '__main__':
    main()
