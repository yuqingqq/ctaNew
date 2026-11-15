"""Post-run interpretation with computed artifact checks; no policy selection."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from predictfun.research.latency_attribution import require, verify_frozen


def verify(summary, attribution, freeze):
    count = 0
    for cfg, arms in summary['totals'].items():
        for arm, row in arms.items():
            other = attribution['cells'][cfg][arm]
            require(abs(row['pnl'] - other['primary_pnl']) < 1e-6, 'independent PnL mismatch')
            require(abs(row['shares'] - other['shares']) < 1e-5, 'independent shares mismatch')
            require(row['windows'] == other['windows'], 'independent window count mismatch')
            count += 1
    for day, receipt in summary['days'].items():
        verify_frozen(receipt, freeze, day)
    return dict(cells_reconciled=count, frozen_receipts_verified=len(summary['days']))


def run(folder):
    summary = json.loads((folder / 'report.json').read_text())
    attribution = json.loads((folder / 'attribution_v3.json').read_text())
    freeze = json.loads((folder / 'freeze_v3.json').read_text())
    checks = verify(summary, attribution, freeze)
    bad = copy.deepcopy(attribution)
    bad['cells']['current']['EV']['shares'] += 1.
    caught = False
    try:
        verify(summary, bad, freeze)
    except ValueError:
        caught = True
    require(caught, 'corrupted artifact was not refused')
    checks['corrupted_artifact_refused'] = caught
    old = json.loads((folder / 'interpretation_v3.json').read_text())
    old.update(version=4, asof=datetime.now(timezone.utc).isoformat(),
               supersedes='interpretation_v3.json: same economic numbers; verification flags now recomputed '
               'at the named artifacts, with a planted corruption refused, rather than summarizing earlier manual checks.',
               interpretation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    old['verification'].pop('all_twenty_cell_pnl_shares_windows_reconciled')
    old['verification'].pop('source_and_fit_freeze_verified')
    old['verification'].update(checks)
    out = folder / 'interpretation_v4.json'
    require(not out.exists(), 'interpretation already exists')
    out.write_text(json.dumps(old, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(asof=old['asof'], verification=old['verification'], output=str(out))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, required=True)
    run(p.parse_args().folder)
