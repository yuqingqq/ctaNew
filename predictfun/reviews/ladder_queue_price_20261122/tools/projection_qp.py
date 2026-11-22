"""WRITE-ONCE PROJECTION of QUEUE-PRICE-STATE-AUDIT-v1 (declaration.md; session aea5bfb9, 2026-11-22). Review 72a36de9: the whole batch <= 0.020
core-hours and the ladder line <= 2.0, "if projection cannot fit, stop". Spent = every run_light line under this receipt's logs; the frozen
stages (selftest, run, readings, checks) are projected at their measured rehearsal / last-selftest costs x 1.25. Writes logs/projection.json
once; the launcher's run stage refuses unless run_full is true. Usage (run_light): projection_qp.py"""
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_queue_price_20261122'; OUTP = R / 'logs' / 'projection.json'
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')
sys.path.insert(0, str(R)); import qp_checks as Q                                     # noqa: E402  its LINE / REVIEWS lists and cpu()


def tag_cpu(tag):
    v = [float(m.group(6)) for p in glob.glob(str(R / 'logs' / '*.log')) for ln in Path(p).read_text(errors='replace').splitlines() for m in [RL.match(ln)] if m and m.group(2) == tag]
    return v[-1] if v else None


def main():
    assert not OUTP.exists(), 'the projection exists (write-once)'
    man = json.loads((R / 'manifest.json').read_text()); cap = float(man['cap_core_hours'])
    spent, _ = Q.cpu(glob.glob(str(R / 'logs' / '**' / '*.log'), recursive=True))
    line, _ = Q.cpu([p for n in Q.LINE for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
    rev, _ = Q.cpu([p for n in Q.REVIEWS for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))])
    stages = dict(selftest=tag_cpu('qp_selftest_final'), run=tag_cpu('qp_rehearsal_run'), readings=tag_cpu('qp_rehearsal_readings'), checks=tag_cpu('qp_rehearsal_checks'))
    ok_inputs = all(v is not None for v in stages.values()); proj = 1.25 * sum(v for v in stages.values() if v is not None) / 3600.
    res = dict(spent_core_hours=spent, line_core_hours=line + rev + spent, stage_cpu_s=stages, projected_core_hours=proj, cap=cap, round_total=spent + proj, line_total=line + rev + spent + proj,
               run_full=bool(ok_inputs and spent + proj <= cap and line + rev + spent + proj <= 2.0))
    OUTP.write_text(json.dumps(res, indent=1) + '\n'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
