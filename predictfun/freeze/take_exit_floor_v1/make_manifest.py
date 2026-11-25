"""TAKE_EXIT_FLOOR_v1 freeze manifest: every repo source the frozen rule imports (sha256, committed unmodified), the rule's parameters
read from the code, and the code commit. Output (write-once): manifest.json beside this file.
Usage: PF_SLOT=claude predictfun/research/run_light.sh tef1_manifest -- python3 predictfun/freeze/take_exit_floor_v1/make_manifest.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[2]
sys.path.insert(0, str(R / 'predictfun/reviews/take_exit_floor_20261125'))
import floor_replay as F  # noqa: E402  (the frozen candidate: FloorBranch with slack 0 inside exit_replay.run_one, STRICT58)


def git(*a):
    return subprocess.check_output(('git',) + a, cwd=R, text=True).strip()


def main():
    out = H / 'manifest.json'
    assert not out.exists(), 'manifest.json exists (write-once)'
    srcs = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if not f:
            continue
        p = Path(f).resolve()
        if R in p.parents and p.suffix == '.py':
            rel = str(p.relative_to(R))
            srcs[rel] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), committed_unmodified=git('status', '--porcelain', '--', rel) == '' and
                             subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=R, capture_output=True).returncode == 0)
    bad = [k for k, v in srcs.items() if not v['committed_unmodified']]
    assert not bad, f'uncommitted or modified sources: {bad}'
    TR, BE, LB = F.TR, F.BE, F.LB
    edges = TR.T.P.frozen_edges()
    prof = dict(sys.modules['pf_ladder_async_fast'].PROFILES['ASYNC_FAST10'])
    res = dict(
        asof='2026-11-25', candidate='TAKE_EXIT_FLOOR_v1', head=git('rev-parse', 'HEAD'),
        code_commit=git('log', '-1', '--format=%H', '--', 'predictfun/reviews/take_exit_floor_20261125/floor_replay.py'),
        entry_point='floor_replay.one(path, cache, old, edges, bps, slack=0.)  (exit_replay.run_one, arm PASSIVE_EXIT, FloorBranch, TAKE_DELAY 58)',
        parameters=dict(take_q=TR.TAKE_Q, take_delay_ms=58., quiet_ticks=TR.QUIET_TICKS, step_ms=BE.STEP, cap=F.ER.CAP_POS, exit_q0=100., exit_rung=0,
                        distance_hold_ticks=BE.HOLD, fill_notice_ms=BE.FILL_NOTICE, cancel_ack_ms=BE.CANCEL_ACK, profile=dict(name='ASYNC_FAST10', **prof),
                        taker_fee_bps=200., maker_fee_bps=0., end_ms=298000., bn_in_ms=F.MM.B_IN, strict_ioc=True, floor='break_even_average_cost_v1', floor_slack=0.,
                        edges={str(k): (v.tolist() if hasattr(v, 'tolist') else v) for k, v in (edges.items() if isinstance(edges, dict) else enumerate(edges))}),
        sources=srcs, n_sources=len(srcs))
    F.TR.dump(out, res)
    print(json.dumps(dict(n_sources=len(srcs), head=res['head'], code_commit=res['code_commit'], parameters={k: v for k, v in res['parameters'].items() if k != 'edges'}), indent=1))
    print('edges', json.dumps(res['parameters']['edges'])[:600])


if __name__ == '__main__':
    main()
