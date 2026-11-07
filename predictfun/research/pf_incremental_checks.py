"""Build and validate INCSIZE without reading market tapes or forward results.

Run through run_incremental_checks.sh. Writes a code-identity/test receipt;
this is implementation validation, not evidence of trading profitability.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .pf_ladder_incremental_build import main as build


class Results:
    def __init__(self):
        self.counts = Counter()

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or (report.when in ('setup', 'teardown') and report.failed):
            self.counts[report.outcome] += 1


def main():
    build()
    import pytest
    root = Path(__file__).resolve().parents[2]
    paths = [
        'predictfun/research/pf_step4_ladder.py',
        'predictfun/research/pf_step4_ladder_incremental.py',
        'predictfun/research/pf_incremental_size.py',
        'predictfun/research/pf_ladder_incremental_build.py',
        'predictfun/research/pf_ladder_inv_build.py',
        'predictfun/research/pf_ladder_fwd_build.py',
        'predictfun/research/pf_incremental_checks.py',
        'predictfun/research/tests/test_incremental_size.py',
        'predictfun/research/tests/test_simulation_causality.py',
    ]
    def identities():
        return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
    before = identities()
    results = Results()
    rc = int(pytest.main(['-q', '--tb=short', str(root / paths[-2]), str(root / paths[-1])], plugins=[results]))
    same = before == identities()
    receipt = dict(as_of=datetime.now(timezone.utc).isoformat(), scope='source-only synthetic implementation checks',
                   market_data_read=False, forward_data_read=False, profitability_tested=False,
                   pytest_exit=rc, tests=dict(results.counts), source_unchanged_during_checks=same,
                   code_sha256=before, passed=rc == 0 and same and results.counts['passed'] > 0)
    dest = root / 'predictfun/reviews/incremental_size_20261106/receipt.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Implementation receipt: {dest}')
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
