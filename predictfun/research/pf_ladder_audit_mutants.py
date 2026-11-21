"""MUTATION CHECK of pf_ladder_audit (RECEIPT-LEAD-TIME-AUDIT-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_audit_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_audit.py'
SELFTEST = 'pf_ladder_audit_selftest'
FAIL = 'AUDIT SELFTEST FAILED'
MUTANTS = [
    ('the effect without the pending clock', "    return max(d + CANCEL, float(o['submit']) + PENDING) if d < float(o['act']) else d + CANCEL", "    return d + CANCEL"),
    ('pending_first from the first live tick', "pending_first=eff(float(o['submit']) + STEP, o)", "pending_first=math.ceil(float(o['act']) / STEP) * STEP + CANCEL"),
    ('the window strictly after the effect', "win = fl if kind == 'entry' else [y for y in fl if y[0] >= e_]", "win = fl if kind == 'entry' else [y for y in fl if y[0] > e_]"),
    ('entries at the request effect', "e_ = float(o['act']) if kind == 'entry' else eff(d, o)", "e_ = eff(d, o)"),
    ('a censored markout as zero', "    mo = None if any(x[5][idx] is None for x in fl) else", "    mo = 0. if any(x[5][idx] is None for x in fl) else"),
    ('sweeps by timestamp', "tr = len(ev[(s, i)]) >= 2; tg = len(ts[(s, t)]) >= 2", "tr = len(ts[(s, t)]) >= 2; tg = len(ts[(s, t)]) >= 2"),
    ('reversals as recoveries', "outcome='recovered' if new == 'UNRESTRICTED' else 'reversal'", "outcome='recovered'"),
    ('early requests far past the first live tick', "        if float(o['req']) <= math.ceil(float(o['act']) / STEP) * STEP:",
     "        if float(o['req']) <= math.ceil(float(o['act']) / STEP) * STEP + 10 * STEP:"),
    ('durations without the full fill', "iv[int(o['side'])].append((float(o['act']), min(float(o['remove']), full if full is not None else np.inf, end)))",
     "iv[int(o['side'])].append((float(o['act']), min(float(o['remove']), end)))"),
    ('leads without the sign consistency', "lead = bool(ok and len(signs) == 1 and 0 not in signs and pooled is not None)", "lead = bool(ok and pooled is not None)"),
    ('leads without the cell minimum', "int(q1.uncensored.iloc[0]) >= MIN_CELL and int(q5.uncensored.iloc[0]) >= MIN_CELL", "True"),
    ('censored rows in the bin means', "u = h[~h.censored_1s & h.win_mo1s.notna() | (h.win_n == 0)];", "u = h;"),
    ('notified fills perturbed in the order check', "fq2 = np.where(np.asarray(ft, float) + NOTICE > d, np.asarray(fq, float) * .5 + 1., fq)", "fq2 = np.asarray(fq, float) * .5 + 1."),
    ('pre-effect fills before the decision', "pre = [] if kind == 'entry' else [y for y in fl if d <= y[0] < e_]", "pre = [] if kind == 'entry' else [y for y in fl if y[0] < e_]"),
    ('the inventory move from the wrong tick', "k2 = k + int(1000. / STEP)", "k2 = k + int(1000. / STEP) + 1"),
    ('a stage before the freeze', "C.require(frozen(), 'the audit module / selftest / declaration / launcher are not the committed ones');", ""),
]
