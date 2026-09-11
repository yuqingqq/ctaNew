"""DRIVE THE REAL PATH: does the payload actually see the snapshot root?

The chain exported PM_DATA_ROOT and `be_heavy_run.sh` passed
`--setenv=PM_DATA_ROOT="$REPO"` to the unit, overwriting it. The 289
falsifier passed because it ran the payload DIRECTLY -- it proved the unit
and not the wiring. This probe runs THROUGH the wrapper, so it tests the
path a valuation actually takes.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main() -> int:
    import de_data_root as DR
    import de_multiday_gate1_runner as R
    r = DR.resolve()
    w = R.winner_source()
    out = {"branch": r.get("branch"),
           "PM_DATA_ROOT_env": os.environ.get("PM_DATA_ROOT"),
           "winner_source_path": w["path"],
           "winner_source_sha256": w["sha256"],
           "n_records": w["n_records"], "pid": os.getpid()}
    print(json.dumps(out), flush=True)
    want = os.environ.get("DE_EXPECT_SNAPSHOT_ROOT")
    if want and not str(w["path"]).startswith(str(want)):
        print(f"REFUSED VALUATION_DID_NOT_SEE_SNAPSHOT_ROOT: the payload "
              f"read {w['path']}, which is not under {want}. The snapshot "
              f"was built and the run did not use it.", flush=True)
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
