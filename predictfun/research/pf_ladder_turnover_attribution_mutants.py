"""MUTATION CHECK of pf_ladder_turnover_attribution (PAIRED-TURNOVER-ATTRIBUTION-v1; 2026-11-22, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_turnover_attribution_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once.
Not planted (exercised on the saved pairs only and recomputed by ta_checks.py): run()'s Y-unchanged, event-identity and 64-pair guards and
the delay comparison with the pilot's stored field."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_turnover_attribution.py'
SELFTEST = 'pf_ladder_turnover_attribution_selftest'
FAIL = 'TURNOVER ATTRIBUTION SELFTEST FAILED'
MUTANTS = [
    ('pre-fork lots matched on the same side', "        while q > 1e-12 and lots and lots[0][0] == -s:\n            x = min(q, lots[0][1]); lots[0][1] -= x; q -= x",
     "        while q > 1e-12 and lots and lots[0][0] == s:\n            x = min(q, lots[0][1]); lots[0][1] -= x; q -= x"),
    ('carried basis at the acquisition price', "        lot[5], lot[6] = True, float(M0)", "        lot[5], lot[6] = True, lot[2]"),
    ('post-fork lots matched on the same side', "        while q > 1e-12 and lots and lots[0][0] == -s:\n            lot = lots[0]; x = min(q, lot[1])",
     "        while q > 1e-12 and lots and lots[0][0] == s:\n            lot = lots[0]; x = min(q, lot[1])"),
    ('profit without the lot sign', "close_gid=int(f[1]), profit=x * (p - lot[6]) * lot[0]))", "close_gid=int(f[1]), profit=x * (p - lot[6])))"),
    ('LIFO instead of FIFO', "            lot = lots[0]; x = min(q, lot[1])", "            lot = lots[-1] if lots[-1][0] == -s else lots[0]; x = min(q, lot[1])"),
    ('residual without the lot sign', "resid = float(sum(l[1] * (M1 - l[6]) * l[0] for l in lots))", "resid = float(sum(l[1] * (M1 - l[6]) for l in lots))"),
    ('the reconciliation dropped', "    need(abs(matched + resid - float(a['Q'])) <= TOL * max(1., abs(float(a['Q']))), f'the FIFO decomposition does not reconcile Q",
     "    need(True, f'the FIFO decomposition does not reconcile Q"),
    ('holding from zero', "hold = float(sum(c['q'] * (c['close_t'] - c['open_t']) for c in closes) / mq)", "hold = float(sum(c['q'] * c['close_t'] for c in closes) / mq)"),
    ('carried matched as new', "matched_carried=float(sum(c['profit'] for c in closes if c['carried']))", "matched_carried=float(sum(c['profit'] for c in closes if not c['carried']))"),
    ('the remaining quantity unsigned', "remaining_q=float(sum(l[1] * l[0] for l in lots))", "remaining_q=float(sum(l[1] for l in lots))"),
    ('own END sign inverted', "k = ('own_end_negative' if own < -TOL else ('own_end_positive' if own > TOL else 'own_end_zero'))", "k = ('own_end_negative' if own > TOL else ('own_end_positive' if own < -TOL else 'own_end_zero'))"),
    ('closes by a focal fill ignored', "rt = sum(c['profit'] for c in closes if (c['close_t'] == t and c['close_gid'] == g) or", "rt = sum(c['profit'] for c in closes if (False and c['close_gid'] == g) or"),
    ('the event key by gid', "return (round(float(f[0]), 6), int(f[5]), int(f[2]), round(float(f[4]), 9), round(float(f[3]), 9))",
     "return (round(float(f[0]), 6), int(f[1]), int(f[2]), round(float(f[4]), 9), round(float(f[3]), 9))"),
    ('no multiset difference', "only_r, only_a = cr - ca, ca - cr", "only_r, only_a = cr, ca"),
    ('the WAIT rule on the POST arm', "    if kind == 'ADMISSION' and arm == 'retain':", "    if kind == 'ADMISSION' and arm == 'alternative':"),
    ('accepted as refused', "acc = next((float(r['submit']) - t for r in rows if not bool(r['refused'])), np.nan)", "acc = next((float(r['submit']) - t for r in rows if bool(r['refused'])), np.nan)"),
    ('day signs swapped', "sign='POSITIVE_EVERY_DAY' if (s > TOL).all() else ('NEGATIVE_EVERY_DAY' if (s < -TOL).all() else 'MIXED')",
     "sign='NEGATIVE_EVERY_DAY' if (s > TOL).all() else ('POSITIVE_EVERY_DAY' if (s < -TOL).all() else 'MIXED')"),
    ('the turnover share of the residual', "d['turnover_share_abs'] = float(g.dMatched.abs().sum() / (g.dMatched.abs().sum() + g.dResidual.abs().sum()))",
     "d['turnover_share_abs'] = float(g.dResidual.abs().sum() / (g.dMatched.abs().sum() + g.dResidual.abs().sum()))"),
    ('the top pair only', "d['top5_share_abs_Y'] = float(a.head(5).sum() / a.sum())", "d['top5_share_abs_Y'] = float(a.head(1).sum() / a.sum())"),
    ('a tie as retain_better', "d.update(pairs=int(len(g)), retain_better=int((g.Y > TOL).sum())", "d.update(pairs=int(len(g)), retain_better=int((g.Y >= -TOL).sum())"),
    ('run without write-once', "need(not (od / 'audit.json').exists(), 'the audit exists (write-once)')", "need(True, 'the audit exists (write-once)')"),
    ('build_manifest without write-once', "path = Path(path or REPO / RECEIPT / 'manifest.json'); need(not path.exists(), 'the manifest exists (write-once)')",
     "path = Path(path or REPO / RECEIPT / 'manifest.json'); need(True, 'the manifest exists (write-once)')"),
]
