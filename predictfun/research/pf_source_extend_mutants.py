"""MUTATION CHECK of pf_source_extend (SOURCE-EXTEND-v1; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with pf_source_extend_selftest.
Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_source_extend.py'
SELFTEST = 'pf_source_extend_selftest'
FAIL = 'SOURCE EXTEND SELFTEST FAILED'
MUTANTS = [
    ('the sealed block allowed', " and not (IN.SEAL_MS[0] <= d0 < IN.SEAL_MS[1]) and d0 + 86400000 <= IN.FWD_MS)", " and d0 + 86400000 <= IN.FWD_MS)"),
    ('a forward day allowed', "and not (IN.SEAL_MS[0] <= d0 < IN.SEAL_MS[1]) and d0 + 86400000 <= IN.FWD_MS)", "and not (IN.SEAL_MS[0] <= d0 < IN.SEAL_MS[1]))"),
    ('an undeclared day allowed', "    return bool(day in NEW_DAYS + (CONTROL_DAY,) and not", "    return bool(True and not"),
    ('the day lists not restored', "        QP.DAYS, QL.DAYS, KPm.DAYS, QP.OUT = keep\n", "        pass\n"),
    ('keep / pull keeps its own day list', "    QP.DAYS = QL.DAYS = KPm.DAYS = (day,); QP.OUT = Path(qp_out)", "    QP.DAYS = QL.DAYS = (day,); QP.OUT = Path(qp_out)"),
    ('keep / pull reconciles with the 11-17 v1', "        d = KPm.day_run(day, v1=root / 'queue_place', out=root / 'keep_pull')", "        d = KPm.day_run(day, v1=OLD_QP, out=root / 'keep_pull')"),
    ('keep / pull before v1', "        QP.day_run(day)\n        d = KPm.day_run(day, v1=root / 'queue_place', out=root / 'keep_pull')",
     "        d = KPm.day_run(day, v1=root / 'queue_place', out=root / 'keep_pull')\n        QP.day_run(day)"),
    ('frames compared by their columns only', "    return bool(A.equals(B))\n", "    return bool(list(A.columns) == list(B.columns))\n"),
    ('frames compared by shape only', "    return bool(A.equals(B))\n", "    return bool(A.shape == B.shape)\n"),
    ('the keep / pull receipt without status', "for k in ('stat', 'status', 'reconciliation', 'fair_missing'))", "for k in ('stat', 'reconciliation', 'fair_missing'))"),
    ('the keep / pull receipt without fair_missing', "for k in ('stat', 'status', 'reconciliation', 'fair_missing'))", "for k in ('stat', 'status', 'reconciliation'))"),
    ('v1 markets uncompared', "qp_receipt=bool(qn['markets'] == qo['markets'] and qn['stat'] == qo['stat'])", "qp_receipt=bool(qn['stat'] == qo['stat'])"),
    ('frozen sources uncompared', "kp_frozen_sources=bool(kn['provenance']['source'] == ko['provenance']['source'])", "kp_frozen_sources=True"),
    ('the control written over', "C.require(not cj.exists(), 'the control exists (write-once)')", "C.require(True, 'the control exists (write-once)')"),
    ('the control from an uncommitted wrapper', "C.require(not cj.exists(), 'the control exists (write-once)'); C.require(committed(), 'the wrapper is not committed')",
     "C.require(not cj.exists(), 'the control exists (write-once)')"),
    ('a new day without the control', "    C.require(cj.exists() and json.loads(cj.read_text()).get('all_ok') is True, 'the positive control has not passed');",
     "    C.require(cj.exists(), 'the positive control has not passed');"),
    ('a new day from an uncommitted wrapper', "'the positive control has not passed'); C.require(committed(), 'the wrapper is not committed')",
     "'the positive control has not passed')"),
    ('the control day as a new day', "C.require(day in NEW_DAYS, f'{day}: not a new day of this extension')", "C.require(day in NEW_DAYS + (CONTROL_DAY, '20261023'), f'{day}: not a new day of this extension')"),
]

if __name__ == '__main__':
    base = SRC.read_text(); bad = [(i, n) for i, (n, o, _) in enumerate(MUTANTS) if base.count(o) != 1]
    print(f'{len(MUTANTS)} mutants; targets not exactly once: {bad}')
