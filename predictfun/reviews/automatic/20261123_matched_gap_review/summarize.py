"""Summarize the review's saved evidence, without another reconstruction."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = Path('/tmp/pf-gap-review-55625bfa/predictfun/reviews/maker_trace_20261122')
e = json.loads((HERE / 'evidence.json').read_text())
counts = {k: sum(r[k] for r in e['clock_boundary']) for k in e['clock_boundary'][0] if k not in ('day', 'm')}
changes = [r for r in e['old_fix_changed_rows'] if r['total_delta'] is not None and abs(r['total_delta']) > 1e-8]
roles = [{k: r[k] for k in ('party', 'scope', 'role_pair', 'sh', 'net', 'fee', 'c_net_sh', 'hold_p50_s')}
         for r in e['roles'] if (r['party'] == 'ours' and r['scope'] == 'active')
         or (r['party'] in ('rank1', 'rank2', 'rank3', 'top3', 'top10') and r['scope'] == 'both_active')]
sys.path.insert(0, str(P / 'tools'))
import cpu_ledger as C
spent = sum(r['cpu_s'] for r in C.ledger()) / 3600 + C.EXTRA_H
out = dict(clock_counts=counts, changed_totals=changes, corrected_top10_total_delta=sum(r['total_delta'] for r in changes),
           roles=roles, independent=e['independent_fifo'], ledger_before_this_review_h=spent,
           line_before_this_review_h=C.LINE_BEFORE + spent,
           this_review_charge_h=.010, queued_packet_cap_including_its_review_h=.035,
           task_after_both_h=spent+.010+.035, line_after_both_h=C.LINE_BEFORE+spent+.010+.035)
(HERE / 'summary.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
