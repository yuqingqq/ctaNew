"""The per-day table of the export receipt, computed from the published receipts (no number typed by hand)."""
import json
import sys
from pathlib import Path
R = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/predict_fun/research/pipeline_export_20261119')
print('day,states,post_rows,retain_rows,retain_eligible,retain_feature_unavailable,retain_off_grid,post_refused_post_only,post_missing_endpoint,'
      'post_stale_future_book,kernel_checked,causal_checked,known_checked,hours_first,hours_last,a0_pulls,seconds,megabytes')
for d in ('20261023', '20261024', '20261025', '20261026', '20261027'):
    r = json.loads((R / d / 'receipt.json').read_text()); st, s = r['stat'], r['statuses']; mb = sum((R / d / x).stat().st_size for x in r['outputs']) / 1e6
    print(','.join(map(str, (d, st['states'], st['post_rows'], st['retain_rows'], s.get('retain:eligible', 0), s.get('retain:feature_unavailable', 0), s.get('retain:off_grid', 0),
                              s.get('post:refused_post_only', 0), s.get('post:missing_endpoint', 0), s.get('post:stale_future_book', 0), st['kernel_checked'], st['causal_checked'],
                              st['known_checked'], r['hours'][0], r['hours'][-1], r['policies']['A+0']['pulls'], round(r['seconds']), round(mb, 1)))))
