"""Build the TOP-UP exposure rows. Separate output path, by design.

AUTHORISATION (R-126, in-file): R-170 Phase 2; DA's v2 receipt declares the
population and explicitly disclaims the dataset (`derived_dataset.built:
false`), naming the Q-DA-67 era floor as the blocker. That defect is fixed
(Q-BE-120/138), so this build is now possible.

WHY A SEPARATE RUNNER AND NOT `harmful_exposure_rows.py run --v2-era`:
that CLI has no --population flag and writes unconditionally to OUT_ERA =
harmful_exposure_rows_v3_eraB.json -- **the CONSUMED FRAGMENT dataset that the
manifest pins by sha256 and that the frozen candidate was fitted on.** Running
it for the top-up would have silently overwritten the frozen population's data
with a different population under the same filename. Nothing would have
errored; the manifest's anchor would simply have stopped matching, and the
freeze's provenance would have been destroyed by a build command.

So: distinct output, and a REFUSAL if the output path is ever the pinned one.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
OUT_TOPUP = DERIVED / "harmful_exposure_rows_v3_topup.json"
PINNED = {DERIVED / "harmful_exposure_rows_v3_eraB.json",
          DERIVED / "harmful_exposure_rows_v3.json"}
POPULATION = "da_development_topup"


class WouldOverwritePinned(RuntimeError):
    """The build tried to write a dataset the manifest pins."""


def guard_output(path: Path) -> None:
    if path in PINNED:
        raise WouldOverwritePinned(
            f"{path.name} is pinned by the manifest and underlies the frozen "
            f"candidate. A top-up build writing there would replace the frozen "
            f"population's data with a different population under the same "
            f"name -- silently, with nothing raising.")


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    guard_output(OUT_TOPUP)
    ok(True, "KNOWN-GOOD: the top-up's own output path is allowed")
    for p in PINNED:
        try:
            guard_output(p)
            ok(False, f"{p.name} must be refused")
        except WouldOverwritePinned:
            ok(True, f"POSITIVE CONTROL: writing to the pinned {p.name} is "
                     f"REFUSED -- this is the exact command that would have "
                     f"destroyed the freeze's provenance without erroring")
    ok(OUT_TOPUP not in PINNED, "the top-up output is distinct from every "
                                "pinned dataset")
    print(f"build_topup_rows selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    selftest()
    guard_output(OUT_TOPUP)
    import harmful_exposure_rows as H
    print(f"  population {POPULATION}; era end read from DA's v2 receipt")
    b = H.v2_era_bounds(POPULATION)
    print(f"  bounds: floor={b[0]:.3f} end={b[1]}")
    lo, hi = H.population_slug_interval(POPULATION)
    print(f"  population interval: t0 > {lo} and t0 < {hi} (BOTH bounds)")
    built = H.build_rows(v2_era=True, population=POPULATION,
                         coins=("btc", "eth"))
    kept = [r for r in built.get("rows", [])
            if H.slug_in_population(int(r["slug"].rsplit("-", 1)[1]), POPULATION)]
    if len(kept) != len(built.get("rows", [])):
        raise SystemExit(
            f"REFUSED: build_rows returned {len(built['rows'])} rows but only "
            f"{len(kept)} fall inside {POPULATION}'s declared interval. The "
            f"selector should already have enforced this; a mismatch means the "
            f"interval is not being applied where it is declared.")
    rows = built.get("rows", [])
    if not rows:
        raise SystemExit(
            "REFUSED: the top-up build produced ZERO rows. That is the "
            "Q-DA-67 symptom; check the era bounds before anything else. "
            "Writing an empty dataset would let Phase 2 'run' on nothing.")
    # Q-DA-77: ATOMIC + STREAMED. `write_text(json.dumps(...))` materialises
    # the ENTIRE serialized dataset as one ~2 GB string before a byte reaches
    # disk -- the R-148 allocation shape that took the box down -- and a crash
    # mid-write leaves a truncated file under the real name. Stream to a temp
    # file, fsync, then os.replace.
    import os as _os, tempfile as _tf
    fd, tmp = _tf.mkstemp(dir=str(OUT_TOPUP.parent), suffix=".tmp")
    try:
        with _os.fdopen(fd, "w") as fh:
            for chunk in json.JSONEncoder().iterencode(built):
                fh.write(chunk)
            fh.flush(); _os.fsync(fh.fileno())
        _os.replace(tmp, OUT_TOPUP)
    except BaseException:
        Path(tmp).unlink(missing_ok=True); raise
    import collections
    st = collections.Counter(r["status"] for r in rows)
    coin = collections.Counter(r["coin"] for r in rows)
    print(f"  WROTE {OUT_TOPUP.name}: {len(rows):,} rows, "
          f"{built.get('n_windows')} windows, days {built.get('days')}")
    print(f"  statuses: {dict(st)}")
    print(f"  by coin : {dict(coin)}")
    for k in ("reconciliation_failures", "boundary_time_violations",
              "consume_clock_violations", "unhooked_state_changes"):
        print(f"  {k}: {built.get(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
