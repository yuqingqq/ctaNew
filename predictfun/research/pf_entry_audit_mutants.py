"""MUTATION CHECK of pf_entry_audit (ENTRY-TARGET-AUDIT-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with pf_entry_audit_selftest.
Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_entry_audit.py'
SELFTEST = 'pf_entry_audit_selftest'
FAIL = 'ENTRY AUDIT SELFTEST FAILED'
MUTANTS = [
    ('the fork-tick fill counted post-fork', "fa = [f for f in a['fill_list'] if float(f[0]) > g]; fb = [f for f in b['fill_list'] if float(f[0]) > g]",
     "fa = [f for f in a['fill_list'] if float(f[0]) >= g]; fb = [f for f in b['fill_list'] if float(f[0]) >= g]"),
    ('the forced generation of the other side', "int(x[1]) == side and x[2] == 'absent' and x[3] == 'POST' and x[4] is not None]", "int(x[1]) != side and x[2] == 'absent' and x[3] == 'POST' and x[4] is not None]"),
    ('V_initial of every POST-arm generation', "    v0 = sum(fv(f, mark) for f in fa if int(float(f[1])) == gid0)", "    v0 = sum(fv(f, mark) for f in fa)"),
    ('D_opposite: the WAIT arm added', "- sum(fv(f, mark) for f in fb if int(float(f[2])) != side)", "+ sum(fv(f, mark) for f in fb if int(float(f[2])) != side)"),
    ('a loose identity', "ok = bool(pre_equal and consistent and abs(gap) <= TOL and np.isfinite(size0))", "ok = bool(pre_equal and consistent and abs(gap) <= 1e3 and np.isfinite(size0))"),
    ('the pre-fork equality unchecked', "ok = bool(pre_equal and consistent", "ok = bool(consistent"),
    ('offset before same cash / inventory', "    elif abs(dcash) <= TOL and abs(dpos) <= TOLP:\n        cls = 'same_cash_inventory'\n    elif abs(Y) <= MEANINGFUL:\n        cls = 'offset_by_mark'\n",
     "    elif abs(Y) <= MEANINGFUL:\n        cls = 'offset_by_mark'\n    elif abs(dcash) <= TOL and abs(dpos) <= TOLP:\n        cls = 'same_cash_inventory'\n"),
    ('the economic comparison with times', "    econ = lambda fl: sorted((int(float(f[2])), round(float(f[3]), 9), round(float(f[4]), 9)) for f in fl)",
     "    econ = lambda fl: sorted((float(f[0]), int(float(f[2])), round(float(f[3]), 9), round(float(f[4]), 9)) for f in fl)"),
    ('a partial fill counted full', "init_fill = 'no_fill' if q0 <= TOLP else ('full' if q0 >= size0 - TOLP else 'partial')", "init_fill = 'no_fill' if q0 <= TOLP else 'full'"),
    ('the residue counted an exact zero', "exact_zero=bool(Y == 0.)", "exact_zero=bool(abs(Y) <= MEANINGFUL)"),
    ('a wider cancellation threshold', "RARE_MEANINGFUL, RARE_NOFILL, CANCEL = .10, .50, .5", "RARE_MEANINGFUL, RARE_NOFILL, CANCEL = .10, .50, .9"),
    ('rare execution without the no-fill condition', "s['share_meaningful'] < RARE_MEANINGFUL and s['init_no_fill'] >= RARE_NOFILL)", "s['share_meaningful'] < RARE_MEANINGFUL)"),
    ('unweighted summaries', "w = g.w.astype(float).values if weighted else np.ones(len(g))", "w = np.ones(len(g))"),
    ('the join by row position', "            i = idx.get((r['day'], r['m'], r['t_ms'], r['side'], r['pick']))", "            i = rows.index(r) if rows.index(r) < len(P) else None"),
    ('a missing join dropped silently', "                    r['pred_status'] = 'prediction_missing'", "                    pass"),
]
