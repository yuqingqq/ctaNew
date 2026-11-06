"""Anchor quality sweep report (declared in STATUS.yml 2026-11-04 01:57Z, part A). P60 and P240 with degraded anchors
(preds_degXX: R^2 = XX/100), the real model and the ideal anchor (runs of the anchor test, tag anc). Per arm and anchor:
mark-at-stop (6 read days, 11-01/02), days > 0, +1 s c/sh, bad-fill share of shares (<= -2 c/sh at 1 s), bad $ / good $,
good $ per share filled. Top makers for reference (pf_maker_bad_fills): first 60 s 9.0% bad, 40%.
Usage: python3 -m predictfun.research.pf_anchor_sweep_report
"""
import json
import pandas as pd
from . import pf_io as io

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']; FRESH = ['20261101', '20261102']; ALL = READ + FRESH
S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1'
ARMS = [('P60', f'{B}+S60/mid@A1:5:25:0'), ('P240', f'{B}/mid@A1:5:25:0')]
KINDS = [('real model', None), ('R2 0.15', 'preds_deg15'), ('R2 0.30', 'preds_deg30'), ('R2 0.50', 'preds_deg50'), ('R2 0.70', 'preds_deg70'),
         ('R2 0.90', 'preds_deg90'), ('ideal', 'preds_oracle')]
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def ddir(d, pre):
    return f"{S4}/live_tokyoreal_custom_e2_{pre or ('preds_c3' if d in FRESH else 'preds_o3')}_anc/{d}"


out = {}
print(f"{'arm':5s} {'anchor':11s} {'mark@stop read':>14s} {'fresh':>7s} {'d+':>4s} {'shares':>7s} {'+1s c/sh':>8s} | {'bad %':>6s} {'bad$/good$':>10s} {'good c/sh':>9s}")
for lab, n in ARMS:
    for kl, pre in KINDS:
        S = {d: json.load(open(f'{ddir(d, pre)}/summary.json'))['arms'][n] for d in ALL}
        f = pd.concat([pd.read_parquet(f'{ddir(d, pre)}/fills/{stem(n)}.parquet', columns=['q', 'mo1']) for d in ALL])
        c = 100 * f.mo1 / f.q; q = f.q.sum(); bad = c <= -2; good = c >= 0
        m = [S[d]['mtm_stop_usd'] for d in ALL]
        r = dict(mtm_read=sum(m[:6]), mtm_fresh=sum(m[6:]), days_pos=sum(x > 0 for x in m), shares=float(q), mo1=float(100 * f.mo1.sum() / q),
                 bad_share=float(f.q[bad].sum() / q), bad_over_good=float(-f.mo1[bad].sum() / f.mo1[good].sum()), good_c=float(100 * f.mo1[good].sum() / q))
        out[f'{lab}|{kl}'] = r
        print(f"{lab:5s} {kl:11s} {r['mtm_read']:14.0f} {r['mtm_fresh']:7.0f} {r['days_pos']:2d}/8 {q:7.0f} {r['mo1']:8.3f} | {r['bad_share']:6.1%} {r['bad_over_good']:10.0%} {r['good_c']:9.2f}")
io.write_json(f'{S4}/report_anchor_sweep.json', out)
