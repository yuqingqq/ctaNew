"""Finalize preflight (review e6b0efa4 finding 2 workaround): before finalizing a census directory, the finalizer's ACTUAL source map (the
modules this process imports) must equal the plan's recorded sources; the assertion and both maps are written to preflight_<tag>.json.
Usage: preflight.py <day> <tag>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_runner as RN                                  # noqa: E402
import pf_r2_shadow as R2                                  # noqa: E402

day, tag = sys.argv[1], sys.argv[2]; d = Path(R2.OUT) / f'census_{day}{tag}'
plan = json.loads((d / 'plan.json').read_text()); actual = RN.sources(R2); ok = actual == plan['sources']
out = dict(asof=R2.C.stamp(), census=str(d), plan_sha256=R2.C.sha(d / 'plan.json'), equal=ok, plan_sources=plan['sources'], actual_sources=actual,
           differing=sorted(k for k in set(actual) | set(plan['sources']) if actual.get(k) != plan['sources'].get(k)))
(Path(__file__).resolve().parent / f'preflight{tag}.json').write_text(json.dumps(out, indent=1)); print(json.dumps(dict(equal=ok, differing=out['differing'])))
sys.exit(0 if ok else 1)
