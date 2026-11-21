"""MUTATION CHECK of pf_adv2_entry_replay_summary (the R4E replay summary v2; 2026-11-21, session aea5bfb9), run by pf_mutant_one.py with
pf_adv2_entry_replay_summary_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_adv2_entry_replay_summary.py'
SELFTEST = 'pf_adv2_entry_replay_summary_selftest'
FAIL = 'ADV2 ENTRY REPLAY SUMMARY SELFTEST FAILED'
MUTANTS = [
    ('the totals with the missing column again (the frozen defect)', "ledger_ok=('ledger_ok', 'all')).reset_index()", "ledger_ok=('ledger_ok', 'all'), binding_bad=('binding_bad', 'sum')).reset_index()"),
    ('the CSV identity unchecked', "        C.require(df.to_csv(index=False).encode() == (out / f'{name}.csv').read_bytes(), f\"{name}.csv differs from the frozen summarize's\")\n", "        pass\n"),
    ('a reversed difference', "diff_sum=float((g[p] - g[b]).sum())", "diff_sum=float((g[b] - g[p]).sum())"),
    ('a tie counted as above', "positive=int(((g[p] - g[b]) > 0).sum())", "positive=int(((g[p] - g[b]) >= 0).sum())"),
    ('no difference against PI0', "for b in ('ENTRY_ONLY', 'PI0') if p != b", "for b in ('ENTRY_ONLY',) if p != b"),
    ('a missing binding counter counted as none', "get('binding_bad', 1))", "get('binding_bad', 0))"),
    ('worlds counted as episodes', "worlds=int(len(piv))", "worlds=int(len(E))"),
    ('the reached count column named 0', ".size().astype(int).rename('n').reset_index()", ".size().astype(int).reset_index()"),
]
