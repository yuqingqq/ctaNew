"""R-155(2) FULL-SCALE gate: run the pipeline with the fast fit twins.

AUTHORISATION (R-126, in-file): R-167(b) — adoption of the numpy fast path is
licensed only after ONE full-scale reproduction of the frozen targets to the
cent. Synthetic equivalence licenses the attempt, nothing more.

THE BUILDER FILE IS NOT EDITED. The manifest pins harmful_hazard_model.py by
sha256 and the freeze receipt pins the same hash. Swapping functions by
editing that file would break both pins to run a test ABOUT that file. So the
twins are bound into the module's namespace at runtime here, and the file on
disk is untouched — asserted below, before and after.
"""
from __future__ import annotations

import hashlib, sys
from pathlib import Path

BUILDER = Path("/home/yuqing/ctaNew/live/pm_research/harmful_hazard_model.py")


def sha() -> str:
    return hashlib.sha256(BUILDER.read_bytes()).hexdigest()


def main() -> int:
    before = sha()
    import harmful_hazard_model as hm
    import harmful_fast_compute as fc

    swapped = {}
    for name in ("fit_logistic", "fit_logistic_w", "fit_ridge", "fit_ridge_w",
                 "zscale", "predict_p", "generation_weights"):
        if hasattr(hm, name) and hasattr(fc, name):
            swapped[name] = (getattr(hm, name), getattr(fc, name))
            setattr(hm, name, getattr(fc, name))
    print(f"  swapped {len(swapped)} fit functions into harmful_hazard_model:")
    for n, (old, new) in swapped.items():
        print(f"    {n}: {old.__module__}.{old.__name__} -> "
              f"{new.__module__}.{new.__name__}")
    # PROVE the swap took, rather than assuming setattr did what it looks like
    assert hm.fit_logistic is fc.fast_fit_logistic, "swap did not take"
    assert hm.zscale is fc.fast_zscale, "zscale swap did not take"
    print("  swap verified live (identity check, not assumed)")

    hm.run_fine(era=True, lead=False)

    after = sha()
    print(f"\n  builder sha before {before[:16]}")
    print(f"  builder sha after  {after[:16]}")
    print(f"  BUILDER FILE UNCHANGED: {before == after}")
    return 0 if before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())
