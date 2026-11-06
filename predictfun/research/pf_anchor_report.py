"""Ideal-anchor test report (declared in STATUS.yml 2026-11-04 01:10Z). Per arm, with the oracle and with the real predictions:
mark-at-stop (P&L marked at the venue mid at stop + 100 ms: spread minus adverse selection, no outcome), gross settlement P&L,
shares, +1 s mark-out (c/sh), swept-through share and +1 s of swept / at-our-price fills, fills on an order whose cancel was
already sent; read days (6) and 11-01/02 separately for the mark-at-stop.
Usage: python3 -m predictfun.research.pf_anchor_report
"""
import json
import pandas as pd
from . import pf_io as io

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']; FRESH = ['20261101', '20261102']; ALL = READ + FRESH
S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1'
ARMS = [('P60 (first 60 s)', f'{B}+S60/mid@A1:5:25:0'), ('P60Q (+250 ms guard)', f'{B}+Q025+S60/mid@A1:5:25:0'), ('P240 (full window)', f'{B}/mid@A1:5:25:0')]
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def ddir(d, kind):
    return f"{S4}/live_tokyoreal_custom_e2_{'preds_oracle' if kind == 'oracle' else ('preds_c3' if d in FRESH else 'preds_o3')}_anc/{d}"


out = {}
print(f"{'arm':22s} {'anchor':6s} {'mark@stop read':>14s} {'fresh':>7s} {'d+':>4s} {'gross':>7s} {'shares':>7s} {'+1s c/sh':>8s} | {'swept':>6s} {'+1s swept':>9s} {'+1s at px':>9s} | {'zombie':>6s}")
for lab, n in ARMS:
    for kind in ('real', 'oracle'):
        S = {d: json.load(open(f'{ddir(d, kind)}/summary.json'))['arms'][n] for d in ALL}
        f = pd.concat([pd.read_parquet(f'{ddir(d, kind)}/fills/{stem(n)}.parquet', columns=['q', 'through', 'mo1', 'zombie']) for d in ALL])
        q = f.q.sum(); c = lambda m: 100 * f.mo1[m].sum() / max(f.q[m].sum(), 1e-9)
        m = [S[d]['mtm_stop_usd'] for d in ALL]
        r = dict(mtm_read=sum(m[:6]), mtm_fresh=sum(m[6:]), mtm_pos=sum(x > 0 for x in m), gross=sum(S[d]['gross_usd'] for d in ALL), shares=float(q),
                 mo1=c(f.q > -1), swept=float(f.q[f.through].sum() / q), mo1_swept=c(f.through), mo1_at_price=c(~f.through), zombie=float(f.q[f.zombie].sum() / q), mtm_by_day=m)
        out[f'{lab}|{kind}'] = r
        print(f"{lab:22s} {kind:6s} {r['mtm_read']:14.0f} {r['mtm_fresh']:7.0f} {r['mtm_pos']:2d}/8 {r['gross']:7.0f} {q:7.0f} {r['mo1']:8.3f} | {r['swept']:6.1%} {r['mo1_swept']:9.3f} {r['mo1_at_price']:9.3f} | {r['zombie']:6.1%}")
print('\nbad fills: share of shares and $ at +1 s by fill quality (c/sh at +1 s: bad <= -2, small loss -2..0, good >= 0)')
print(f"{'arm':22s} {'anchor':6s} {'shares':>7s} | {'bad sh':>7s} {'bad %':>6s} {'bad $':>7s} | {'loss sh':>7s} {'loss $':>7s} | {'good sh':>7s} {'good $':>7s}")
for lab, n in ARMS:
    for kind in ('real', 'oracle'):
        f = pd.concat([pd.read_parquet(f'{ddir(d, kind)}/fills/{stem(n)}.parquet', columns=['q', 'mo1']) for d in ALL])
        c = 100 * f.mo1 / f.q; bad = c <= -2; small = (c > -2) & (c < 0); good = c >= 0; q = f.q.sum()
        out[f'{lab}|{kind}'].update(bad_shares=float(f.q[bad].sum()), bad_usd=float(f.mo1[bad].sum()), small_loss_shares=float(f.q[small].sum()),
                                    small_loss_usd=float(f.mo1[small].sum()), good_shares=float(f.q[good].sum()), good_usd=float(f.mo1[good].sum()))
        print(f"{lab:22s} {kind:6s} {q:7.0f} | {f.q[bad].sum():7.0f} {f.q[bad].sum() / q:6.1%} {f.mo1[bad].sum():7.0f} | {f.q[small].sum():7.0f} {f.mo1[small].sum():7.0f} | "
              f"{f.q[good].sum():7.0f} {f.mo1[good].sum():7.0f}")
io.write_json(f'{S4}/report_anchor.json', out)
