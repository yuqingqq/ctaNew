"""SELFTEST v2 of pf_branch_ladder / pf_ladder_rest (review 0065191f finding 6; 2026-11-21, session aea5bfb9). SUPERSEDES the use of
pf_ladder_rest_selftest.py (v1, frozen and bound into the LADDER_REST run, kept unchanged): v1's group-10 guard fixtures assumed the module
uncommitted and the pilot projection absent, and fail once the run is committed and projected. v2 runs the frozen v1 suite unchanged with
those two NEGATIVE conditions provided explicitly (frozen() false, projection_allows() false) and then checks the corresponding ALLOWED
conditions: with frozen() true and projection_allows() true, run_day passes both guards for the pilot and the full run and reaches its
loader (a test binder that raises), and with frozen() true and projection_allows() false the pilot still runs while the full run refuses.
The repository's actual stage is never read by the fixtures. run(mod) is the mutation-harness entry (as v1)."""
import importlib
import json
import sys
import tempfile
from pathlib import Path

FAIL = 'LADDER SELFTEST FAILED'


def run(mod):
    main(mod if hasattr(mod, 'run_ladder') else importlib.import_module('pf_ladder_rest'))


def main(M):
    import pf_ladder_rest_selftest as V1
    bad = []; saved = (M.frozen, M.projection_allows)
    M.frozen = lambda: False; M.projection_allows = lambda: False
    try:
        V1.main(M)
    except SystemExit as e:
        bad.append(f'the frozen suite under the explicit negative conditions: {str(e)[:300]}')
    finally:
        M.frozen, M.projection_allows = saved

    class Reached(Exception):
        pass

    def raiser():
        raise Reached('the loader was reached')
    with tempfile.TemporaryDirectory(dir=str(M.KP.I.K.IO)) as tmp:
        mp = Path(tmp) / 'man.json'; mp.write_text(json.dumps(dict(selected=[dict(day='20261024', m=1, hour=0, ws=0., parent_worlds=[0])], pilot={'20261024': [1]})))
        for fz, pa, pilot, want in ((True, True, True, 'reach'), (True, True, False, 'reach'), (True, False, True, 'reach'), (True, False, False, 'projection'),
                                    (False, True, True, 'freeze')):
            M.frozen = (lambda v=fz: v); M.projection_allows = (lambda v=pa: v)
            try:
                M.run_day('20261024', pilot=pilot, binder=raiser, man_path=mp, out=Path(tmp)); got = 'ran'
            except Reached:
                got = 'reach'
            except ValueError as e:
                got = 'freeze' if 'freeze' in str(e) else 'projection' if 'projection' in str(e) else f'other: {e}'
            finally:
                M.frozen, M.projection_allows = saved
            if got != want:
                bad.append(f'run_day guards (frozen {fz}, projection {pa}, pilot {pilot}): {got} != {want}')
    if bad:
        raise SystemExit(FAIL + ': ' + ' | '.join(bad))
    print('LADDER SELFTEST v2 PASSED: the frozen suite under explicit negative guard conditions, and the allowed guard conditions')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(importlib.import_module('pf_ladder_rest'))
