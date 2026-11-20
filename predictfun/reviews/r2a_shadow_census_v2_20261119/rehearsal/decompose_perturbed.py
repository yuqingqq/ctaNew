"""Diagnose the rehearsal's 51 stored-vs-perturbed differences (8 originals, the same consumed Oct 23 subset): decompose each failing row's
perturbed rebuild into (a) perturbed windows for the market / order inputs only (original path, fills, fair); (b) the perturbed path only
(original windows); (c) the unnotified-fill change only; (d) the future fair change only. Also the known-frame indices: jk (the builder's
perturbation boundary at the step) vs jt (the last frame known_steps reads: kf <= d), on both sides, and whether the two sides share a grid."""
import inspect
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2  # noqa: E402
X, QP, KP, BE, FC = R2.X, R2.QP, R2.KP, R2.BE, R2.FC
print(inspect.getsource(FC.V.order_inputs)[:1500])
day = '20261023'; src = KP.Tape(); O = pd.read_parquet(KP.OUT / day / 'orders.parquet')
sel = pd.concat([O[O.side == s_].sort_values('oid').head(4) for s_ in (0, 1)])
need = sorted(set(sel.m.astype(int)) & set(src.markets(day))); fair, depth, _, _ = X.real_inputs(day, need)
data = src.load(day, need, src.offset(day)); cols = R2.INPUTS + R2.CTX; END = R2.END; out = []
for W in src.pairs(data):
    m = int(W[0]['m'])
    if m not in need:
        continue
    print('market', m, 'grids equal across sides:', np.array_equal(np.asarray(W[0]['tg']), np.asarray(W[1]['tg'])), 'jk equal:',
          np.array_equal(np.asarray(W[0]['jk']), np.asarray(W[1]['jk'])), flush=True)
    for o in sel[sel.m == m].itertuples():
        if str(o.status) not in ('eligible', 'refused_post_only'):
            continue
        s, t0 = int(o.side), float(o.t); path, _ = QP.build_path(W, t0, t0 + END)
        if path is None:
            continue
        er = R2.probes(path, o, day, []); fv = np.array([float(getattr(o, f'f{j:02d}')) for j in range(KP.NSTEP + 1)])
        R2.add_inputs(er, W, o, fair.get(m), depth, X.real_features, lambda k, fv=fv: float(fv[min(k, len(fv) - 1)]))
        for r in [x for x in er if x.get('sampled') and x.get('status') == 'ok']:
            g = r['decision_ms']; d = t0 + g; tg = np.asarray(W[s]['tg'], float); kk = min(int(np.searchsorted(tg, d)), len(tg) - 1); k_ep = int(round(g / R2.STEP))
            Wp, fair_p, dep_p = X.real_perturb(W, fair.get(m), depth, kk, d, np.random.default_rng(int(d) % 100003)); pth_p, _ = QP.build_path(Wp, t0, t0 + END)
            fvp = fv.copy(); fvp[k_ep + 1:] += 7.
            late = lambda ft_, fq_, d_: (np.where(ft_ + KP.NOTICE > d_, ft_ + 777., ft_), np.where(ft_ + KP.NOTICE > d_, fq_ * .5, fq_))
            full = R2.rebuild_row(pth_p, Wp, fvp, o, day, r, fair_p, dep_p, X.real_features, late) if pth_p is not None else None
            bad = R2.same_f32(r, full, cols) if full is not None else ['NO PATH']
            if not bad:
                continue
            va = R2.rebuild_row(path, Wp, fv, o, day, r, fair_p, dep_p, X.real_features)
            vb = R2.rebuild_row(pth_p, W, fv, o, day, r, fair.get(m), depth, X.real_features)
            vc = R2.rebuild_row(path, W, fv, o, day, r, fair.get(m), depth, X.real_features, late)
            vd = R2.rebuild_row(path, W, fvp, o, day, r, fair.get(m), depth, X.real_features)
            kf = {a: np.asarray(W[a]['kf'], float) for a in (0, 1)}
            jk = {a: int(W[a]['jk'][kk]) for a in (0, 1)}; jt = {a: int(np.searchsorted(kf[a], d, side='right') - 1) for a in (0, 1)}
            Bp = R2.resim(pth_p, o, day, r); B0 = R2.resim(path, o, day, r)
            g0 = B0.open[s]; gp = Bp.open[s]
            out.append(dict(oid=int(o.oid), side=s, source=r['source'], head=r['head'], decision_ms=g, full=bad[:6],
                            a_windows=R2.same_f32(r, va, cols)[:6], b_path=R2.same_f32(r, vb, cols)[:6], c_fills=R2.same_f32(r, vc, cols)[:6],
                            d_fair=R2.same_f32(r, vd, cols)[:6], jk=jk, jt=jt, kf_at_jt=float(kf[s][jt[s]]) if jt[s] >= 0 else None, d=d,
                            gen_orig=(B0.E.o[g0]['price'], B0.E.o[g0]['act']) if g0 is not None else None,
                            gen_pert=(Bp.E.o[gp]['price'], Bp.E.o[gp]['act']) if gp is not None else None,
                            stored=dict(o_age=r.get('o_age'), fp_queue=r.get('fp_queue'), activation_ms=r.get('activation_ms'), price=r.get('price'))))
print('failures', len(out))
for x in out[:12]:
    print(json.dumps(x, default=float))
print('summary a/b/c/d nonempty:', {k: sum(bool(x[k]) for x in out) for k in ('a_windows', 'b_path', 'c_fills', 'd_fair')},
      'jk<jt on the focal side:', sum(x['jk'][x['side']] < x['jt'][x['side']] for x in out), 'jk==jt:', sum(x['jk'][x['side']] == x['jt'][x['side']] for x in out),
      'kf==d at jt:', sum(x['kf_at_jt'] == x['d'] for x in out))
