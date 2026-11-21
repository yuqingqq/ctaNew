"""MUTATION CHECK of pf_entry_exposure_summary (ENTRY-EXPOSURE-v1 summary v2; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_entry_exposure_summary_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_entry_exposure_summary.py'
SELFTEST = 'pf_entry_exposure_summary_selftest'
FAIL = 'ENTRY EXPOSURE SUMMARY V2 SELFTEST FAILED'
MUTANTS = [
    ('overrides from the frozen counter', "n = logged_overrides(r); a, b = ", "n = int(r.get('overrides') or 0); a, b = "),
    ('logged overrides from the counter', "    return len(r.get('shadow') or [])", "    return int(r.get('overrides') or 0)"),
    ('the baseline ignores the decisions hash', " and a['decisions_sha256'] == b['decisions_sha256']", ""),
    ('the baseline ignores the fills hash', "a['fills_sha256'] == b['fills_sha256'] and ", ""),
    ('the baseline ignores the value', " and a['value'] == b['value']))", "))"),
    ('a baseline mismatch kept ok', "            out['status'] = 'keep_baseline_mismatch'", "            pass"),
    ('the frozen counter not kept', "out['overrides_counter_v1'] = r.get('overrides')", "out['overrides_counter_v1'] = None"),
    ('partitions read without the hash check', "C.require(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['manifest_sha256']", "C.require(True and rec['manifest_sha256']"),
    ('an existing v2 summary overwritten', "C.require(not (out2 / 'exposure.json').exists(), 'the v2 summary exists (immutable)')", "pass"),
    ('the summaries over the frozen flat', "f = flat_v2(r); changed", "f = M.flat(r); changed"),
    ('the changed count over every row', "changed += int(f.get('overrides') != f.get('overrides_counter_v1') and r.get('arms') is not None)", "changed += 1"),
]
