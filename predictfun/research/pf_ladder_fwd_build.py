"""Generate pf_step4_ladder_fwd.py = the FROZEN ladder engine of pm_ladder_v1 (pf_step4_ladder.py sha256 c682229e...) with one change:
the first inadmissible day FORWARD is read from PF_FORWARD (default '20261104', the frozen value) so the declared forward check
(11-04..11-08) can replay those days. FORWARD only bounds which days' input is admissible (the engine refuses day >= FORWARD and drops
input from day >= FORWARD as padding); no rule of the strategy reads it. Parity: with PF_FORWARD unset the generated engine gives
byte-identical candidate outputs to round i (pf_v2_engine_check.py). FWD_PATCH is shared with pf_ladder_v2_build.py.
Usage: python3 -m predictfun.research.pf_ladder_fwd_build
"""
import hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = f'{HERE}/pf_step4_ladder.py'; DST = f'{HERE}/pf_step4_ladder_fwd.py'
FROZEN = 'c682229e4fbe606b217cfa633816bcba67bce701c7e690b4cb8be9aa7ff6e37f'
FWD_PATCH = ("SEALED = ('20261028', '20261029', '20261030'); FORWARD = '20261104'   # no input from a sealed or forward day (padding included)\n",
             "SEALED = ('20261028', '20261029', '20261030'); FORWARD = os.environ.get('PF_FORWARD', '20261104')   # forward-enabled: first inadmissible day\n"
             "assert len(FORWARD) == 8 and FORWARD.isdigit() and FORWARD >= '20261104', f'PF_FORWARD {FORWARD}'\n")


def generate(patches, dst, what):
    src = open(SRC, 'rb').read()
    h = hashlib.sha256(src).hexdigest()
    if h != FROZEN:
        sys.exit(f'REFUSED: {SRC} sha256 {h} is not the frozen engine {FROZEN}')
    s = src.decode()
    for a, b in patches:
        n = s.count(a)
        if n != 1:
            sys.exit(f'REFUSED: anchor occurs {n} times: {a[:80]!r}')
        s = s.replace(a, b)
    out = (f'# GENERATED from pf_step4_ladder.py sha256 {FROZEN} (frozen engine of pm_ladder_v1)\n'
           f'# with {len(patches)} anchored change(s): {what}. Do not edit; regenerate.\n') + s
    open(dst, 'w').write(out)
    print(f'wrote {dst} sha256 {hashlib.sha256(out.encode()).hexdigest()} ({len(patches)} change(s))')


def main():
    generate([FWD_PATCH], DST, 'FORWARD from PF_FORWARD (pf_ladder_fwd_build)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
