"""CPU LEDGER of MAKER-TRACE-RECONSTRUCTION-v1 (session aea5bfb9, 2026-11-22). Sums cpu_user_s of every run_light line whose tag starts
'mt_' in predictfun/research/run_light.log (the launcher's authoritative ledger; every Python run of this task goes through run_light),
from the task start (2026-11-22T08:00Z), and checks the USER cap: 1.35 core-hours including the 0.05 h review reserve, line ceiling 4.85 (first approval 0.90 / 4.40)
from a line of at most 3.3107 before the task; the USER topped it up to 1.35 h / line 4.85 (AskUserQuestion, 2026-11-22) after the
builders' tests used 0.33 h. A line without a parsable cpu_user_s is counted at its wall_s (never as zero); a line with
neither refuses. Usage: cpu_ledger.py [--need <projected_more_h>]"""
import json
import re
import sys

LOG = '/home/yuqing/ctaNew/predictfun/research/run_light.log'
START, CAP, RESERVE, LINE_BEFORE, CEILING = '2026-11-22T08:00:00Z', 1.95, 0.05, 3.3107, 5.25   # USER top-ups: 0.90 -> 1.35 h (4.40 -> 4.85); -> 1.45 (amendment 1); -> 1.85 / line 5.25 for the lifecycle fix + top-3 decision paths (amendment 3); -> 1.90 for TAKE-LOGIC v1 (USER 2026-11-23, line unchanged); -> 1.95 for POSTING-CHAIN v1 (USER standing CPU authorization 2026-11-23; review 3f640ad9b)
EXTRA_H = 0.021 + 0.025 + 0.010 + 0.010 + 0.010 + 0.010   # Codex reviews 567f741d9 (0.021), 8f2a2e753 (<= 0.025), 6ddad3533 (0.010), 6bb458b30 (0.010), af916aba6 (0.010), 3f640ad9b (0.010): their declared conservative charges (runs outside the mt_ prefix)
PAT = re.compile(r'^(\S+) tag=(mt_\S+) rc=(\S+) .*?wall_s=(\S+) cpu_user_s=(\S+)')


def ledger(log=LOG):
    rows = []
    for line in open(log):
        m = PAT.match(line)
        if not m or m.group(1) < START:
            continue
        t, tag, rc, wall, cpu = m.groups()
        try:
            s = float(cpu)
        except ValueError:
            try:
                s = float(wall)
            except ValueError:
                raise SystemExit(f'unparsable ledger line: {line.strip()}')
        rows.append(dict(t=t, tag=tag, rc=rc, cpu_s=s))
    return rows


if __name__ == '__main__':
    need = float(sys.argv[sys.argv.index('--need') + 1]) if '--need' in sys.argv else 0.
    rows = ledger(); spent = sum(r['cpu_s'] for r in rows) / 3600. + EXTRA_H
    res = dict(runs=len(rows), spent_h=round(spent, 4), cap_h=CAP, reserve_h=RESERVE, need_h=need, fits=bool(spent + need + RESERVE <= CAP and LINE_BEFORE + spent + need + RESERVE <= CEILING),
               line_now_max=round(LINE_BEFORE + spent, 4), ceiling=CEILING, by_prefix={})
    for r in rows:
        k = '_'.join(r['tag'].split('_')[:2]); res['by_prefix'][k] = round(res['by_prefix'].get(k, 0.) + r['cpu_s'], 2)
    print(json.dumps(res, indent=1))
