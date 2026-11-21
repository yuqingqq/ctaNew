"""Assemble the ENTRY-TARGET-AUDIT-v1 receipt (report.md) from checks.json / checks_falsifier.json, the receipt's output copies, tables.md
(audit_tables.py) and the run_light logs. Every verdict sentence is chosen by a predicate computed here or declared in the module; every number
is read from an output (or a count, sum, difference or ratio of read values). Writes report.md and audit_values.json (scratchpad)."""
import json
import re
from pathlib import Path

import pandas as pd

R = Path('/home/yuqing/ctaNew/predictfun/reviews/entry_target_audit_20261121'); L = R / 'logs'; S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
COMMITS = dict(review='d6d5d55e254c507db9799ef2fc1c6129d4657d06', freeze='811fd7b3501602c3427b74dbc07032b611900559', amendment='6157b1a0983f8a0cb763682d3108d76786e41eb0')
REC = dict(both='Both mechanisms hold: the continuation removes at least half of the initial generation\'s absolute value, and execution is rare.',
           cancellation='Cancellation: the continuation removes at least half of the initial generation\'s absolute value. A larger decision unit is '
                        'worth declaring next, with its alternative action and opportunity cost specified before any label is built.',
           rare_execution='Rare execution only: the forced 50 ms deviation rarely changes execution, so relabeling the initial-generation component '
                          'alone will not solve the entry problem.',
           not_identified='Not identified: the declared predicates identify neither mechanism, and no target change follows from this audit alone.')


def suite(name):
    t = (L / name).read_text(); m = re.search(r'(\d+) / (\d+) planted code errors are caught', t); return int(m.group(1)), int(m.group(2))


def sections(md):
    out, cur = {}, None
    for ln in md.splitlines():
        if ln.startswith('### '):
            cur = ln[4:].strip(); out[cur] = []
        elif cur is not None:
            out[cur].append(ln)
    return {k: '\n'.join(v).strip('\n') for k, v in out.items()}


def main():  # noqa: C901
    J = json.loads((R / 'checks.json').read_text()); F = json.loads((R / 'checks_falsifier.json').read_text()); A = json.loads((R / 'outputs' / 'audit.json').read_text())
    X = pd.read_csv(R / 'outputs' / 'crosstab.csv'); T = sections((R / 'tables.md').read_text()); v = {}; P = A['predicates']
    cpu = sum(float(m.group(6)) for p in L.glob('*.log') for ln in p.read_text().splitlines() for m in [RL.match(ln)] if m) / 3600.
    rk = {}
    for cand in ('SMOOTH', 'RICH'):
        x = X[(X.set == 'eval') & (X.candidate == cand)]; n = hi = 0
        for (day, s), g in x.groupby(['day', 'side']):
            a, b = g[g.action == 'POST'], g[g.action == 'WAIT']
            if len(a) and len(b):
                n += 1; hi += int(float(a.wmean_v_initial.iloc[0]) > float(b.wmean_v_initial.iloc[0]))
        rk[cand] = (hi, n)
    agree = P['train']['diagnosis'] == P['eval']['diagnosis']
    ck = J['checks']
    lines = ['# ENTRY-TARGET-AUDIT-v1 — the accounting of the existing POST / WAIT labels (no fit, no new label, no replay)', '',
             f"Review {COMMITS['review']} (one next priority). Freeze `{COMMITS['freeze'][:8]}`, amendment `{COMMITS['amendment'][:8]}` (outputs under distinct names, before any "
             f"output). Evidence as of {J['asof']}. Development session Claude aea5bfb9.", '',
             '## Verdict (every sentence chosen by a computed or declared predicate)', '',
             f"- **Declared checks:** {'all true' if J['all_true'] else 'NOT all true: ' + ', '.join(k for k, x in sorted(ck.items()) if not x)} ([checks.json](checks.json)); "
             f"falsifier {'passed' if F['falsifier_ok'] else 'FAILED'} ([checks_falsifier.json](checks_falsifier.json)).",
             f"- **Population:** {A['rows']:,} POST rows (every status kept; expected {sum(A['expected_post_rows'].values()):,}) and {A['reached_rows']:,} reached pairs. The "
             f"identity Y = V_initial + D_focal + D_opposite holds on every accounted row (check B {'true' if ck.get('B_accounting') else 'false'}).",
             ]
    for k, lab in (('train', 'Oct 24–25 (in-sample for the frozen models)'), ('eval', 'Oct 26–27 (adaptive development)')):
        p = P[k]; ratio = p['abs_Y'] / p['abs_v_initial'] if p['abs_v_initial'] else None
        lines.append(f"- **{lab}:** meaningful share {p['share_meaningful']:.3f}, initial generation with no fill {p['init_no_fill']:.3f}, Σ w|Y| / Σ w|V_initial| "
                     f"{('%.3f' % ratio) if ratio is not None else '—'}: rare_execution {str(p['rare_execution']).lower()}, cancellation {str(p['cancellation']).lower()}, "
                     f"**diagnosis {p['diagnosis']}**.")
    lines.append(f"- **The two sets {'agree' if agree else 'DISAGREE'}** on the diagnosis.")
    lines.append(f"- **Recommendation (from the adaptive Oct 26–27 diagnosis, by the declared rule):** {REC[P['eval']['diagnosis']]}")
    lines.append(f"- **The frozen models' POST groups carry a higher mean V_initial than their WAIT groups** in {rk['SMOOTH'][0]} of {rk['SMOOTH'][1]} day × side cells (SMOOTH) and "
                 f"{rk['RICH'][0]} of {rk['RICH'][1]} (RICH). This is a cross-tab, not a skill claim.")
    lines += ['- **Scope:** an accounting of the existing labels. It fits nothing, builds no label and proposes no candidate here; any new target is a new '
              'declared candidate. V_initial is not a deployable entry advantage.', '', '## Tables (audit_tables.py, copied from the outputs)', '']
    for k, t in T.items():
        lines += [f'### {k}', '', t, '']
    lines += ['## Review d6d5d55e', '',
              '- **Item 1:** the smoothing experiment is closed as failed; candidates and multiplicity are kept; this audit is the next step. No threshold '
              'mining or regularization sweep.',
              '- **Item 2:** R4E\'s first-review table is renamed per leg (A, V_post, V_retain), and its predicates table separates the mandatory-anchor '
              'control value from the all-sampled selected-POST mean (tables above). No data rerun.',
              '- **Item 3:** the replay checker is not reused here, so the review\'s verified workaround stands. A future replay checker adds episode END, '
              'settings, fill-time bounds and common marks with single-field falsifiers.', '',
              '## Provenance', '',
              f"- Mutation checks: the final audit suite {suite('mutants_audit_summary.log')[0]} / {suite('mutants_audit_summary.log')[1]}. Run 1 "
              f"({suite('mutants_audit_run1_summary.log')[0]} / {suite('mutants_audit_run1_summary.log')[1]}) had a crash at a summary fixture, now guarded. "
              f"Run 2 ({suite('mutants_audit_run2_summary.log')[0]} / {suite('mutants_audit_run2_summary.log')[1]}) preceded the amendment.",
              '- The first audit run was stopped by hand before writing any file: the per-row reached data would have been overwritten by its summary. '
              'The amendment gives every output its own name.', '',
              '## Cost', '', f"- {cpu:.3f} core-hours of CPU over every run_light job under logs/ (the ceiling is 1); check F "
              f"{'true' if ck.get('F_resources') else 'false'}.", '',
              '## Limits', '',
              '- In-sample Oct 24–25 and adaptive Oct 26–27 development days only: no interval, no validation.',
              '- The accounting explains the existing label under its frozen continuation. A different decision unit needs its own declared alternative '
              'and costs before any label is built.', '']
    (R / 'report.md').write_text('\n'.join(lines) + '\n')
    v.update(checks={k: bool(x) for k, x in ck.items()}, all_true=bool(J['all_true']), falsifier_ok=bool(F['falsifier_ok']), predicates=P, agree=agree, rank=rk,
             rows=A['rows'], reached_rows=A['reached_rows'], core_hours=cpu, asof=J['asof'], recommendation=REC[P['eval']['diagnosis']])
    (S / 'audit_values.json').write_text(json.dumps(v, indent=1, default=str)); print(json.dumps(v, indent=1, default=str))


if __name__ == '__main__':
    main()
