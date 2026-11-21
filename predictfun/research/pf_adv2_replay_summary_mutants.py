"""MUTATION CHECK of pf_adv2_replay_summary (the R3 replay summary v2; 2026-11-21, session aea5bfb9). Each planted error is a textual
replacement in a copy of pf_adv2_replay_summary.py, executed under its real path in a fresh process by pf_mutant_one.py, which runs
pf_adv2_replay_summary_selftest.run on it. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_adv2_replay_summary.py'
SELFTEST = 'pf_adv2_replay_summary_selftest'
FAIL = 'ADV2 REPLAY SUMMARY SELFTEST FAILED'
MUTANTS = [
    ('the count column named 0 again (the frozen defect)', ".size().astype(int).rename('n').reset_index()", ".size().astype(int).reset_index()"),
    ('the CSV identity unchecked', "        C.require(df.to_csv(index=False).encode() == (out / f'{name}.csv').read_bytes(), f\"{name}.csv differs from the frozen summarize's\")\n", "        pass\n"),
    ('the part hash unchecked', "        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'], f'part {k}: bytes differ from its record')\n", ""),
    ('a reversed difference', "diff_sum=float((g[p] - g[b]).sum())", "diff_sum=float((g[b] - g[p]).sum())"),
    ('a tie counted as above', "positive=int(((g[p] - g[b]) > 0).sum())", "positive=int(((g[p] - g[b]) >= 0).sum())"),
    ('only ok worlds counted', "st[r['status']] = st.get(r['status'], 0) + 1", "st['ok'] = st.get('ok', 0) + int(r['status'] == 'ok')"),
    ('the bid value as the value', "value=('value', 'sum'), value_bid", "value=('value_bid', 'sum'), value_bid"),
    ('no difference against ENTRY_PA', "for b in ('PI0', 'ENTRY_PA') if p != b", "for b in ('PI0',) if p != b"),
    ('worlds counted as episodes', "worlds=int(len(piv))", "worlds=int(len(E))"),
    ('the CSV hashes of another file', "csv_sha256={n: C.sha(out / f'{n}.csv') for n in CSVS}", "csv_sha256={n: C.sha(out / 'episodes.csv') for n in CSVS}"),
]
