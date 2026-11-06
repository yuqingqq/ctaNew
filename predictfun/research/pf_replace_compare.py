"""Parallel vs sequential replace at the measured Tokyo clock, fill by fill (2026-11-04; user: "cancel and post at the same
time is the most efficient one already, why it does not help"). Reads test L2c's outputs (order live 40 ms, cancel 22 ms,
fills known 100 ms, decision 10 ms, model A, six read days; docs §23): C2-P with parallel replace (no XS) or sequential
(XS0), with and without the venue limit (XV), and the frozen v6 (sequential, 1 place/s). Per arm: gross settlement P&L,
shares, mark-out at +1 s / +60 s (c/sh), swept-through share and its +1 s, fills at our own price and theirs, and fills on
an order whose cancel was already sent (the replay's 'zombie' flag) and theirs.
Usage: python3 -m predictfun.research.pf_replace_compare
"""
import json
import pandas as pd
from . import pf_io as io

D = f'{io.OUT}/step4/live_tokyoreal_custom_e2_preds_rl_a_l2c_40_22_f100_d10'
DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
B = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1'
ARMS = [('parallel, no limit', f'{B}/mid@A1:5:25:0'), ('parallel, venue limit', f'{B}+XV/mid@A1:5:25:0'),
        ('sequential, no limit', f'{B}+XS0/mid@A1:5:25:0'), ('sequential, venue limit', f'{B}+XS0+XV/mid@A1:5:25:0'),
        ('sequential, 1 place/s (v6)', f'{B}+XS0+XT1/mid@A1:5:25:0')]
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
out = {}
print(f"{'arm':28s} {'gross $':>8s} {'shares':>7s} {'+1s c/sh':>8s} {'+60s':>7s} | {'swept':>6s} {'+1s swept':>9s} {'+1s at px':>9s} | {'zombie':>6s} {'+1s zomb':>8s}")
for lab, n in ARMS:
    f = pd.concat([pd.read_parquet(f'{D}/{d}/fills/{stem(n)}.parquet') for d in DAYS]); S = [json.load(open(f'{D}/{d}/summary.json'))['arms'][n] for d in DAYS]
    q = f.q.sum(); c = lambda m, col='mo1': 100 * f[col][m].sum() / max(f.q[m].sum(), 1e-9); al = f.q > -1
    r = dict(gross=sum(s['gross_usd'] for s in S), shares=float(q), mo1=c(al), mo60=c(al, 'mo60'), swept=float(f.q[f.through].sum() / q), mo1_swept=c(f.through),
             mo1_at_price=c(~f.through), zombie=float(f.q[f.zombie].sum() / q), mo1_zombie=c(f.zombie) if f.zombie.any() else None)
    out[lab] = r
    print(f"{lab:28s} {r['gross']:8.0f} {q:7.0f} {r['mo1']:8.3f} {r['mo60']:7.3f} | {r['swept']:6.1%} {r['mo1_swept']:9.3f} {r['mo1_at_price']:9.3f} | {r['zombie']:6.1%} {(r['mo1_zombie'] or 0):8.3f}")
io.write_json(f'{io.OUT}/step4/report_replace_compare.json', out)
