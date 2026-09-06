"""THE GATE-1 FEATURE FRAGMENT, ONE FILE PER RULED DAY. Never a pinned name.

WHY THIS EXISTS. Round 48's book build refused at `fragment_slice`: *"the
slice is EMPTY: no window of harmful_exposure_rows_v3_eraB.json matched the
247 wanted slugs."* Verified at the artifact -- eraB is 1.24 GB, 471 windows,
2026-08-24 to 2026-08-25, and holds no September slug. It is the CONSUMED,
PINNED population the frozen candidate was fitted on. **September was never
materialised.** A scored day-book cannot be assembled until it is.

THE MECHANISM ALREADY EXISTED AND IT IS `build_topup_rows.py`'s: a SEPARATE
OUTPUT PATH, with a refusal if the build ever aims at a pinned name. That
module's own docstring says what is at stake -- running the ordinary CLI
"would have silently overwritten the frozen population's data with a
different population under the same filename. Nothing would have errored."
This module copies that discipline and WIDENS it:

  * the guard refuses EVERY `harmful_exposure_rows_*` dataset now on disk,
    not the two the top-up named;
  * and it refuses ANY path that already exists, because the second-worst
    outcome after overwriting a pinned name is overwriting yesterday's
    fragment with today's under a name nobody pinned.

ONE FILE PER DAY, at `harmful_exposure_rows_v3_gate1_<YYYYMMDD>.json`. Not
one file for the ruled set: a per-day file can be built, measured, refused
and re-built independently, and a day that will not fit under the cap costs
only itself.

THE POPULATION IS THE DAY'S ADMISSIBLE WINDOWS, DIGEST-PINNED. Not a declared
population interval -- those end 2026-08-26T00:00 and no September day passes
them. It is `de_admissible_windows.supply(day, present_from_ledger(day))`,
the same selection the forward scorer gated 09-03 through at 12/12 and the
same 247 btc slugs round 48 selected. The receipt carries the digest of that
slug list, so a fragment built from a different selection cannot be mistaken
for this one.

THE DIGEST IS OF THE BYTES AS WRITTEN, AND IT IS STREAMED. `write_text(
json.dumps(...))` materialises the whole ~600 MB dataset as one string before
a byte reaches disk -- the allocation shape the top-up builder was changed to
avoid. So the encoder streams into a temp file AND into a hashlib at the same
time: the digest is of exactly the bytes that landed, with nothing held.

R8: IF IT DOES NOT FIT, IT REFUSES. Never a raised cap and never a smaller
population -- a fragment over fewer windows than the day supplies is a
fragment about a different day.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import be_data_root as _BDR

ROOT = HERE.parents[1]
DERIVED = _BDR.derived()
OUT_DERIVED = ROOT / "data/pm_5min/derived"

#: EVERY exposure-row dataset on disk, not the two the top-up guard named.
#: A name that is not pinned today can be pinned tomorrow; the cheap rule is
#: to refuse them all and require a Gate-1 name.
PINNED_PREFIX = "harmful_exposure_rows"
GATE1_STEM = "harmful_exposure_rows_v3_gate1_"
COIN = "btc"
MEM_CAP_GB = 8.0


class FragmentRefused(RuntimeError):
    """A named refusal."""


def out_path(day: str, coin: str = COIN) -> Path:
    return DERIVED / f"{GATE1_STEM}{day}_{coin}.json"


def pinned_names(derived: Path | None = None) -> set:
    d = Path(derived) if derived is not None else DERIVED
    return {p.name for p in d.glob(f"{PINNED_PREFIX}*.json")
            if not p.name.startswith(GATE1_STEM)}


def guard_output(path: Path, *, derived: Path | None = None) -> None:
    """Refuse a pinned name, AND refuse anything that already exists."""
    p = Path(path)
    if p.name in pinned_names(derived):
        raise FragmentRefused(
            f"REFUSED: {p.name} is an existing exposure-row dataset. A "
            f"Gate-1 build writing there would replace one population's data "
            f"with another under the same name -- silently, with nothing "
            f"raising, which is exactly what destroyed nothing only because "
            f"`build_topup_rows` refused it first.")
    if not p.name.startswith(GATE1_STEM):
        raise FragmentRefused(
            f"REFUSED: {p.name} is not a Gate-1 fragment name. The output "
            f"must be `{GATE1_STEM}<day>_<coin>.json` so no build can aim at "
            f"a name another population owns.")
    if p.exists():
        raise FragmentRefused(
            f"REFUSED: {p} already exists. Overwriting yesterday's fragment "
            f"with today's under a name nobody pinned is the second-worst "
            f"version of the same defect. Move or delete it deliberately.")


def population(day: str, coin: str = COIN) -> dict:
    """The day's admissible windows, and the digest that pins them."""
    import be_forward_day as FD
    import de_admissible_windows as AW
    sup = AW.supply(day, FD.present_from_ledger(day))
    slugs = sorted(w["slug"] for w in (sup.get("windows") or {}).get(coin, []))
    if not slugs:
        raise FragmentRefused(
            f"REFUSED: no supplied {coin} windows for {day}.")
    blob = "\n".join(slugs).encode()
    return {"day": day, "coin": coin, "n_windows": len(slugs),
            "slugs_sha256": hashlib.sha256(blob).hexdigest(),
            "source": "de_admissible_windows.supply(day, "
                      "be_forward_day.present_from_ledger(day))",
            "why_not_a_declared_population": "the declared population "
                                             "intervals end "
                                             "2026-08-26T00:00; no September "
                                             "day passes slug_in_population",
            "slugs": slugs}


def selector_for(pop: dict):
    """`build_rows`' selector hook, over the day's slugs only."""
    import flow_intensity as fi
    import harmful_exposure_rows as HER
    want = list(pop["slugs"])
    era = HER._era_or_refuse(fi, None, "be_gate1_fragment")
    paths, toks = fi._archive_paths(), fi.token_map()
    gaps = fi.gaps_by_slug(era)
    missing = [s for s in want if s not in paths or s not in toks]
    if missing:
        raise FragmentRefused(
            f"REFUSED: {len(missing)} supplied slug(s) have no archive path "
            f"or token entry, e.g. {missing[:3]}. Dropping them would build "
            f"a fragment about a different day (R8: never a smaller "
            f"population).")

    def _sel(coins, population_name):
        return [(s, paths[s], toks[s][0], toks[s][1], gaps.get(s, []))
                for s in want], 0
    _sel.era = era
    return _sel


def _rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def stream_write(obj, dst: Path) -> dict:
    """Encode ONCE, into the file and into the hash at the same time.

    The digest is therefore of exactly the bytes that landed -- the B-1
    discipline on the write side -- without ever holding the document."""
    h = hashlib.sha256()
    n = 0
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            for chunk in json.JSONEncoder(default=str).iterencode(obj):
                b = chunk.encode()
                h.update(b)
                fh.write(b)
                n += len(b)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return {"sha256": h.hexdigest(), "bytes": n,
            "digest_is_of_the_bytes_as_written": True,
            "streamed_never_materialised": True}


#: The ruled day set. btc-only is R-560; the day set is the coordinator's.
RULED_DAYS = ("20260901", "20260902", "20260903", "20260904", "20260905")


def declaration() -> dict:
    """WHAT THIS BUILDER IS, declared, so DE's runner can bind to it."""
    return {
        "protocol": "BE_GATE1_FRAGMENT_DECLARATION_V1",
        "why": ("round 48's book refused at fragment_slice because "
                "harmful_exposure_rows_v3_eraB.json is the CONSUMED, PINNED "
                "08-24/25 population (1.24 GB, 471 windows, verified at the "
                "artifact) and September was never materialised into it"),
        "one_file_per_ruled_day": True,
        "why_per_day_and_not_one_file": ("a per-day fragment can be built, "
                                         "measured, refused and rebuilt on "
                                         "its own, and a day that will not "
                                         "fit costs only itself"),
        "path_scheme": f"data/pm_5min/derived/{GATE1_STEM}<YYYYMMDD>_<coin>.json",
        "coin": COIN, "coin_ruling": "R-560, btc-only for all ruled days",
        "ruled_days": list(RULED_DAYS),
        "population": {
            "definition": "that day's admissible windows from "
                          "de_admissible_windows.supply(day, "
                          "be_forward_day.present_from_ledger(day))",
            "digest_pinned": "sha256 over the sorted slug list, in every "
                             "receipt, so a fragment built from a different "
                             "selection cannot be mistaken for this one",
            "why_not_a_declared_population": "the declared intervals end "
                                             "2026-08-26T00:00; no September "
                                             "day passes slug_in_population",
            "never_reduced": "R8 -- a fragment over fewer windows than the "
                             "day supplies is a fragment about another day",
        },
        "guard": {
            "model": "build_topup_rows.guard_output, WIDENED",
            "refuses_every_existing_exposure_row_dataset": sorted(
                pinned_names()),
            "refuses_any_name_outside_the_gate1_stem": True,
            "refuses_any_path_that_already_exists": True,
            "why_widened": "the top-up guard named TWO pinned datasets; "
                           "there are seven on disk, and overwriting "
                           "yesterday's Gate-1 fragment with today's under a "
                           "name nobody pinned is the same defect one step "
                           "down",
        },
        "receipt_fields_per_day": ["population.slugs_sha256",
                                   "population.n_windows", "build.n_rows",
                                   "resources.wall_s",
                                   "resources.peak_rss_gb",
                                   "fragment.sha256", "fragment.bytes"],
        "digest_discipline": ("the encoder streams into the file and into "
                              "the hash at once, so the sha256 is of exactly "
                              "the bytes that landed and the document is "
                              "never materialised (the B-1 rule on the write "
                              "side; write_text(json.dumps(...)) is the "
                              "allocation shape build_topup_rows was changed "
                              "to avoid)"),
        "cap": {"gb": MEM_CAP_GB, "raised": False,
                "on_exceed": "REFUSE the day and report the measured peak"},
        "builds_no_book_and_runs_no_assembly": True,
        "decides_nothing": "REPORTED (rule 14).",
    }


def build(day: str, *, coin: str = COIN, progress: bool = True) -> dict:
    import harmful_exposure_rows as HER
    t0 = time.time()
    dst = out_path(day, coin)
    guard_output(dst)
    pop = population(day, coin)
    sel = selector_for(pop)
    if progress:
        print(json.dumps({"stage": "population", "windows": pop["n_windows"],
                          "slugs_sha256": pop["slugs_sha256"][:16],
                          "out": str(dst)}), flush=True)
    built = HER.build_rows(coins=(coin,), selector=sel)
    rows = built.get("rows", [])
    if not rows:
        raise FragmentRefused(
            f"REFUSED: the {day} fragment produced ZERO rows. Writing an "
            f"empty dataset would let the assembly 'run' on nothing -- the "
            f"empty answer that looks like a result.")
    peak = _rss_gb()
    if peak > MEM_CAP_GB:
        raise FragmentRefused(
            f"REFUSED: peak RSS {peak} GB exceeded the {MEM_CAP_GB} GB cap. "
            f"R8: the cap is NOT raised and the population is NOT reduced. "
            f"The day is reported with its measured peak and refused.")
    if progress:
        print(json.dumps({"stage": "built", "rows": len(rows),
                          "windows": built.get("n_windows"),
                          "peak_gb": peak}), flush=True)
    w = stream_write(built, dst)
    return {
        "protocol": "BE_GATE1_FRAGMENT_V1",
        "day": day, "coin": coin,
        "fragment": {"path": str(dst), **w},
        "population": {k: v for k, v in pop.items() if k != "slugs"},
        "build": {"n_rows": len(rows), "n_windows": built.get("n_windows"),
                  "days": built.get("days"),
                  "reconciliation_failures":
                      built.get("reconciliation_failures"),
                  "boundary_time_violations":
                      built.get("boundary_time_violations"),
                  "consume_clock_violations":
                      built.get("consume_clock_violations"),
                  "unhooked_state_changes":
                      built.get("unhooked_state_changes")},
        "resources": {"wall_s": round(time.time() - t0, 1),
                      "peak_rss_gb": _rss_gb(),
                      "cap_gb": MEM_CAP_GB, "cap_raised": False},
        "guard": {"refuses_existing_datasets": sorted(pinned_names()),
                  "refuses_any_existing_path": True,
                  "required_stem": GATE1_STEM},
        "data_root": _BDR.receipt_block(),
        "one_file_per_day": True,
        "no_book_built": True, "no_assembly_run": True,
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 10


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    names = pinned_names()
    ok(len(names) >= 5 and "harmful_exposure_rows_v3_eraB.json" in names,
       f"the guard's refusal set is READ FROM DISK, not typed: "
       f"{len(names)} existing exposure-row datasets including the pinned "
       f"eraB one -- `build_topup_rows` named two, and there are {len(names)}")
    for n in sorted(names)[:3]:
        try:
            guard_output(DERIVED / n)
            ok(False, f"{n} must refuse")
        except FragmentRefused as e:
            ok("existing exposure-row dataset" in str(e),
               f"KNOWN-BAD: writing to {n} REFUSES -- the exact command that "
               f"would replace one population with another under one name")
    try:
        guard_output(DERIVED / "some_other_name.json")
        ok(False, "a non-Gate-1 name must refuse")
    except FragmentRefused as e:
        ok("not a Gate-1 fragment name" in str(e),
           "KNOWN-BAD: any name outside the Gate-1 stem REFUSES, so no build "
           "can aim at a name another population owns")
    good = out_path("20260903")
    if good.exists():
        ok(True, f"POSITIVE CONTROL SKIPPED: {good.name} already exists")
    else:
        guard_output(good)
        ok(True, f"POSITIVE CONTROL: the Gate-1 name {good.name} is ALLOWED "
                 f"while it does not exist")
    ok(GATE1_STEM not in "".join(names),
       "and no existing dataset already uses the Gate-1 stem, so the two "
       "namespaces do not overlap")

    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        d = Path(td)
        f = d / "x.json"
        w = stream_write({"a": [1, 2, 3], "b": "x"}, f)
        ok(w["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
           and w["bytes"] == f.stat().st_size,
           f"THE STREAMED DIGEST IS OF THE BYTES THAT LANDED: "
           f"{w['sha256'][:16]}… recomputed from the file matches, over "
           f"{w['bytes']} bytes, with the document never materialised")
        f2 = d / "y.json"
        w2 = stream_write({"a": [1, 2, 4], "b": "x"}, f2)
        ok(w2["sha256"] != w["sha256"],
           "and a DIFFERENT document digests differently -- the check is on "
           "the bytes, not a constant")

    d = declaration()
    ok(d["path_scheme"].endswith(f"{GATE1_STEM}<YYYYMMDD>_<coin>.json")
       and set(d["guard"]["refuses_every_existing_exposure_row_dataset"])
       == pinned_names()
       and d["cap"]["gb"] == MEM_CAP_GB,
       f"THE DECLARATION IS DERIVED FROM THE CODE IT DECLARES: the stem, the "
       f"{len(names)}-dataset refusal set and the cap all come from the "
       f"module's own constants, so a declaration cannot drift from the "
       f"builder")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--declare" in argv:
        out = declaration()
        dst = (HERE / "declarations" /
               "be_gate1_fragment_declaration_v1.json")
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"written": str(dst),
                          "days": len(out["ruled_days"]),
                          "refuses": len(out["guard"][
                              "refuses_every_existing_exposure_row_dataset"])}))
        return 0
    if "--day" in argv:
        day = argv[argv.index("--day") + 1]
        out = build(day)
        dst = OUT_DERIVED / f"be_gate1_fragment_receipt_{day}_{COIN}.json"
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"receipt": str(dst),
                          "fragment": out["fragment"]["path"],
                          "sha256": out["fragment"]["sha256"],
                          "bytes": out["fragment"]["bytes"],
                          "n_rows": out["build"]["n_rows"],
                          "wall_s": out["resources"]["wall_s"],
                          "peak_rss_gb": out["resources"]["peak_rss_gb"]}))
        return 0
    print("usage: be_gate1_fragment.py --selftest | --declare | --day <YYYYMMDD>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
