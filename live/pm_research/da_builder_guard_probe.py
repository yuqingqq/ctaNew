"""(a) BOTH-DIRECTIONS probe of the builder's absorption guard (R-203/R-204).

SURFACE AUTHORISATION (R-126): R-203(a). Written BEFORE BE's candidate lands,
against the RULED semantics, so it cannot be shaped to the implementation.

WHY IT EXISTS. be-tape6c was killed because the builder's 1% absorption guard
iterated ALL statuses, so PRE_WINDOW at 3.69% would have REFUSED a VALID
population at write time. The guard was right to exist and wrong about what it
counted: a row that is EMITTED with a status is part of the population; a slug
DROPPED before emission is not. Those are different quantities and only the
second is absorption.

THE RULED SEMANTICS, as two directions that must BOTH hold:
  BUILD  -- a high share of an EMITTED ROW STATUS (e.g. PRE_WINDOW at 4%) must
            NOT refuse. Those rows are in the tape; nothing was absorbed.
  REFUSE -- a small share of PRE-EMISSION SKIPS (e.g. 2% NO_TOKEN_MAP or
            NO_ARCHIVE_PATH) MUST refuse. Those rows are gone.
A guard that only satisfies one direction is not half-right: refusing valid
populations and admitting absorbed ones are both fatal, and a guard tuned to
avoid one failure typically causes the other.

DA CHECKS, DA DOES NOT SPECIFY (R-185): this asserts the behaviour BE's guard
must exhibit. The implementation is BE's.

    python3 live/pm_research/da_builder_guard_probe.py            # self-check
    python3 live/pm_research/da_builder_guard_probe.py --against  # BE's guard
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROW_STATUSES = ("OK", "PRE_WINDOW", "POST_WINDOW", "NO_BOOK",
                "GAP_AT_CUTOFF", "NO_LEVEL_HISTORY")
BOUND = 0.01


def absorption_share(status_counts: dict, emitted_rows: int,
                     row_statuses=ROW_STATUSES) -> float:
    """Share of INPUT rows lost BEFORE emission. The quantity the bound is on.

    Emitted statuses are excluded by construction -- that exclusion is the
    whole correction, and stating it as a formula rather than a list means a
    NEW emitted status does not silently become 'absorption'.
    """
    skipped = sum(v for k, v in status_counts.items()
                  if k not in set(row_statuses))
    denom = emitted_rows + skipped
    return (skipped / denom) if denom else 0.0


def should_refuse(status_counts: dict, emitted_rows: int,
                  bound: float = BOUND) -> bool:
    return absorption_share(status_counts, emitted_rows) > bound


CASES = [
    # (label, status_counts, emitted_rows, must_refuse)
    ("PRE_WINDOW at 3.69% (the tape6c killer) -- MUST BUILD",
     {"OK": 963_100, "PRE_WINDOW": 36_900}, 1_000_000, False),
    ("PRE_WINDOW at 4% -- MUST BUILD",
     {"OK": 960_000, "PRE_WINDOW": 40_000}, 1_000_000, False),
    ("every emitted status large -- MUST BUILD",
     {"OK": 800_000, "PRE_WINDOW": 150_000, "GAP_AT_CUTOFF": 30_000,
      "NO_LEVEL_HISTORY": 20_000}, 1_000_000, False),
    ("NO_TOKEN_MAP at 2% -- MUST REFUSE",
     {"OK": 1_000_000, "NO_TOKEN_MAP": 20_408}, 1_000_000, True),
    ("NO_ARCHIVE_PATH at 2% -- MUST REFUSE",
     {"OK": 1_000_000, "NO_ARCHIVE_PATH": 20_408}, 1_000_000, True),
    ("total absorption (the tape6b failure) -- MUST REFUSE",
     {"NO_TOKEN_MAP": 1_764_206}, 0, True),
    ("a skip status invented tomorrow at 2% -- MUST REFUSE",
     {"OK": 1_000_000, "SOME_FUTURE_SKIP": 20_408}, 1_000_000, True),
    ("skips just UNDER the bound (0.5%) -- MUST BUILD",
     {"OK": 1_000_000, "NO_TOKEN_MAP": 5_025}, 1_000_000, False),
    ("clean build, no skip counters -- MUST BUILD",
     {"OK": 1_000_000, "PRE_WINDOW": 60_000}, 1_060_000, False),
]


def run(fn) -> int:
    npass = 0
    for label, counts, emitted, must in CASES:
        got = fn(counts, emitted)
        ok = (got == must)
        npass += ok
        share = absorption_share(counts, emitted)
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}\n"
              f"         absorption {100*share:5.2f}%  refuse={got} "
              f"(must be {must})")
    print(f"\nboth-directions probe: {npass}/{len(CASES)}")
    return 0 if npass == len(CASES) else 1


def main() -> int:
    if "--against" in sys.argv:
        try:
            import build_state_tape_v2 as B
        except Exception as ex:
            print(f"REFUSED: cannot import the builder ({ex}).")
            return 2
        fn = None
        for name in ("should_refuse_absorption", "absorption_refuses",
                     "check_absorption", "_absorption_guard"):
            if hasattr(B, name):
                fn = getattr(B, name)
                print(f"probing builder.{name}")
                break
        if fn is None:
            print("REFUSED: no absorption-guard entry point found in the "
                  "builder under the names this probe knows. Not a pass -- "
                  "the guard must be callable to be testable.")
            return 2
        return run(fn)
    print("self-check of the RULED semantics (no builder involved):")
    return run(should_refuse)


if __name__ == "__main__":
    raise SystemExit(main())
