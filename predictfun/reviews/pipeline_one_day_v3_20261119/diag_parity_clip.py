"""The 451 rows of the first v3 keep / pull run (Oct 26) on which the contract refused a forecast the benchmark v3 accepted: the v3 parity's reconstruction
(E[q | fill] = E[q] / P without the v3.1 snap at the remaining-size cap) collected with each row's numbers and the contract's refusal message. Usage (run_light): diag_parity_clip.py <day>"""
import sys
from collections import Counter
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import numpy as np
import pandas as pd
import pf_keep_pull as KP
import pf_pipeline_contract as PC
import pf_pipeline_policy as P
day = sys.argv[1]; I = KP.I
pred = pd.read_parquet(f'{P.BENCH}/pred_{day}.parquet', columns=['day', 'oid', 'kind', 'j', 'side', 't', 'decision_ms', 'p', 'pf_mid', 'fp_remaining', 'J_P', 'J_Eq', 'J_Eqm', 'J_V', 'J_ok'])
p = pred[pred.kind == 'retain']; trained = float(I._ms('2026-10-25') + 86400000); tol = lambda x: 1e-9 * max(1., abs(x)); rows = []; why = Counter()
for r in p.itertuples():
    side = 'bid' if r.side == 0 else 'ask'; rem = float(r.fp_remaining); t = float(r.t); d = float(r.decision_ms); val = None; msg = ''
    tg = PC.Target(f'{r.day}:{int(r.oid)}', side, 'retain', d, t + float(KP.landing(int(r.j))), t + KP.H, float(r.pf_mid), float(r.p), rem, 'B', 'c', 'e')
    qf = float(r.J_Eq / r.J_P) if r.J_P > 0 else 0.; mv = float(r.J_Eqm / r.J_Eq) if r.J_Eq > 0 else 0.
    try:
        atoms = (PC.Atom(float(r.J_P), qf, mv), PC.Atom(1. - float(r.J_P), 0., 0.)) if 0. < r.J_P < 1. else (PC.Atom(1., qf if r.J_P >= 1 else 0., mv),)
        fc = PC.Forecast(tg, atoms, d, trained, 'learned', 'm'); val = PC.map_value(tg, fc)
        st = PC.ClientState(tg.order_id, side, d, 'live', rem, 0., rem if side == 'bid' else 0., rem if side == 'ask' else 0., 1e9)
        action = PC.decide_one(tg, fc, st, PC.Economics(tg, 0., 0., 0., 0.)).action; accepted = True
    except ValueError as e:
        accepted, action, msg = False, 'keep', str(e)
    rt = val is not None and abs(val.quantity - r.J_Eq) <= tol(r.J_Eq) and abs(val.movement - r.J_Eqm) <= tol(r.J_Eqm) and abs(val.retained - r.J_V) <= tol(r.J_V)
    coh = accepted and rt; action = action if coh else 'keep'; rule = 'pull' if (bool(r.J_ok) and r.J_V < 0) else 'keep'
    if not (coh == bool(r.J_ok) and (abs(r.J_V) < 1e-6 or action == rule)):
        k = (f'verdict {coh} vs J_ok {bool(r.J_ok)}', msg or ('round trip' if accepted and not rt else 'action'))
        why[k] += 1
        rows.append(dict(oid=int(r.oid), j=int(r.j), side=int(r.side), P=float(r.J_P), Eq=float(r.J_Eq), qf=qf, rem=rem, qf_minus_rem=qf - rem, mv=mv,
                         end=float(r.pf_mid) + (1. if r.side == 0 else -1.) * mv, pf_mid=float(r.pf_mid), V=float(r.J_V), ok=bool(r.J_ok), coh=coh, msg=msg))
print(f'{day}: {len(p):,} RETAIN rows; {len(rows):,} disagreements')
for k, v in why.most_common():
    print(f'  {v:>5,}  {k}')
D = pd.DataFrame(rows)
if len(D):
    print(D.describe().T[['min', '50%', 'max']].to_string())
    print(D.head(8).to_string())
    print('qf > rem:', int((D.qf > D.rem).sum()), ' qf - rem max:', float(D.qf_minus_rem.max()), ' end outside [0,100]:', int(((D.end < -1e-9) | (D.end > 100 + 1e-9)).sum()))
