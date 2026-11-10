#!/usr/bin/env python3
"""BE 139 -- THE GAP CENSUS ON WALL-CLOCK ATTRIBUTION (coordinator ruling).

WHY. `flow_intensity.gaps_by_slug` attributes a gap to a slug by the
COLLECTOR's `window_start` field, not by the gap's own wall clock, then clamps
the offsets to [0, WINDOW_S] and keeps only `g1 > g0`. When a disconnect spans
a window boundary the `gap_closed` row is stamped against the window that was
OPEN WHEN THE DISCONNECT BEGAN, so the gap instant lies in the NEXT window --
the clamped interval is empty, the row is DROPPED, and it is never
re-attributed. The gap lands in NEITHER window and is counted NOWHERE.

Measured on 09-07 BTC `clob_v4_1`: 35 rows, 33 survive -> 27 windows, and TWO
are dropped (stamped 14:40 landing in 14:45; stamped 15:50 landing in 15:55).

That is rule 4 -- an exclusion that is not a counted status. This census gives
those gaps the status `GAP_RECORDED_NOT_SEEN_BY_REPLAY` and reports BOTH
attributions side by side. IT CHANGES NOTHING THE REPLAY COMPUTES: it reads
the ledger and reports. `flow_intensity` is in the pinned build path and is
NOT touched until the last forward book exists (coordinator ruling, BE 139).

THE ANTI-DRIFT CONTROL. The census re-implements the clamp in order to
classify each row, and a re-implementation can drift from the code it
describes. So it RECONCILES its own replay-side window set against
`fi.gaps_by_slug(era)` -- the real function -- and REFUSES on any mismatch
rather than reporting a stale model (rule 15).

Exit codes (75 is the wrapper's and is not among them, rule 20):
  0  census written   1  reconciliation REFUSED   2  usage   3  input unreadable
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SEEN = "GAP_SEEN_BY_REPLAY"
NOT_SEEN = "GAP_RECORDED_NOT_SEEN_BY_REPLAY"
EXIT_CODES = {0: "CENSUS_WRITTEN", 1: "RECONCILIATION_REFUSED",
              2: "USAGE", 3: "INPUT_UNREADABLE"}
assert 75 not in EXIT_CODES, "75 is the wrapper's conflict code (rule 20)"


class CensusRefused(RuntimeError):
    """The census cannot be trusted, so none is written."""


def _utc(t) -> str:
    return datetime.datetime.fromtimestamp(int(t), datetime.UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def classify(row: dict, window_s: float) -> dict | None:
    """One ledger row under BOTH attributions. None if it carries no interval."""
    s, e = row.get("gap_start_ns"), row.get("gap_end_ns")
    ws, slug = row.get("window_start"), row.get("slug")
    if not (s and e and ws and slug):
        return None
    ws = int(ws)
    g0_raw, g1_raw = s / 1e9 - ws, e / 1e9 - ws
    g0, g1 = max(0.0, g0_raw), min(window_s, g1_raw)
    seen = g1 > g0
    wall = int((s / 1e9) // window_s * window_s)
    return {
        "slug": slug,
        "stamped_window": ws, "stamped_window_utc": _utc(ws),
        "wall_clock_window": wall, "wall_clock_window_utc": _utc(wall),
        "attribution_agrees": wall == ws,
        "raw_offsets": [round(g0_raw, 3), round(g1_raw, 3)],
        "clamped_offsets": [round(g0, 3), round(g1, 3)],
        "gap_seconds": round(e / 1e9 - s / 1e9, 6),
        "status": SEEN if seen else NOT_SEEN,
    }


def census(day: str, coin: str = "btc") -> dict:
    import flow_intensity as fi
    import be_era_for_day as EFD
    import be_gate1_fragment as FR

    pop = FR.population(day, coin)
    era = EFD.resolve(fi, pop["day"], list(pop["slugs"]))["era"]
    window_s = float(fi.WINDOW_S)
    day0 = min(int(s.rsplit("-", 1)[1]) for s in pop["slugs"])
    day1 = day0 + 86400

    if not fi.GAPS.exists():
        raise CensusRefused(f"REFUSED: the gap ledger {fi.GAPS} does not exist")
    rows = []
    with fi.GAPS.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # KEY ON THE SLUG, NOT THE `coin` FIELD. The anti-drift control
            # caught this on its first real run: `gaps_by_slug` keys purely on
            # the slug and never reads `coin`, so filtering rows by `coin`
            # made my replay-side set 160 where the real function had 165.
            # A row whose `coin` is absent is still a BTC row if its slug says
            # so, and the slug is what the replay indexes by.
            if r.get("collector_version") != era:
                continue
            if not str(r.get("slug") or "").startswith(f"{coin}-"):
                continue
            ws = r.get("window_start")
            if not ws or not (day0 <= int(ws) < day1):
                continue
            c = classify(r, window_s)
            if c is not None:
                rows.append(c)

    seen_windows = sorted({r["stamped_window"] for r in rows
                           if r["status"] == SEEN})
    wall_windows = sorted({r["wall_clock_window"] for r in rows})
    not_seen = [r for r in rows if r["status"] == NOT_SEEN]

    # ---- THE ANTI-DRIFT CONTROL: my clamp against the REAL function.
    real = fi.gaps_by_slug(era)
    real_windows = sorted({int(s.rsplit("-", 1)[1]) for s, v in real.items()
                           if v and s.startswith(f"{coin}-")
                           and day0 <= int(s.rsplit("-", 1)[1]) < day1})
    if real_windows != seen_windows:
        raise CensusRefused(
            f"REFUSED: this census's replay-side window set does not match "
            f"flow_intensity.gaps_by_slug({era!r}). Mine has "
            f"{len(seen_windows)}, the real function {len(real_windows)}; "
            f"only in mine {sorted(set(seen_windows) - set(real_windows))[:5]}, "
            f"only in the function "
            f"{sorted(set(real_windows) - set(seen_windows))[:5]}. The census "
            f"models a clamp it no longer describes -- reporting it would be "
            f"a stale model presented as a measurement.")

    return {
        "protocol": "BE_GAP_CENSUS_V1",
        "day": day, "coin": coin, "era": era,
        "as_of_utc": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "ledger": str(fi.GAPS),
        "n_supplied_windows": pop["n_windows"],
        "n_gap_rows_with_an_interval": len(rows),
        "REPLAY_ATTRIBUTION": {
            "what_it_is": ("the collector's own window_start, clamped to "
                           "[0, WINDOW_S] and kept only if g1 > g0 -- this is "
                           "what the fragment and therefore the replay sees"),
            "n_rows_seen": len(rows) - len(not_seen),
            "n_windows": len(seen_windows),
            "windows": seen_windows,
            "RECONCILED_AGAINST": "flow_intensity.gaps_by_slug(era)",
        },
        "WALL_CLOCK_ATTRIBUTION": {
            "what_it_is": "the window whose span contains the gap's own start",
            "n_windows": len(wall_windows),
            "windows": wall_windows,
        },
        NOT_SEEN: {
            "n": len(not_seen),
            "meaning": ("the collector recorded this gap and the replay does "
                        "not carry it: the row is stamped to a window that "
                        "ends before the gap begins, so its clamped interval "
                        "is empty and it is dropped -- and it is never "
                        "re-attributed to the window that contains it. "
                        "Counted here so it is an accounted exclusion and not "
                        "a quietly smaller denominator (rule 4)."),
            "rows": not_seen,
            "windows_that_gain_a_gap_under_wall_clock": sorted(
                {r["wall_clock_window"] for r in not_seen}
                - set(seen_windows)),
        },
        "WHAT_THIS_DOES_NOT_CHANGE": (
            "nothing the replay computes. This is metadata: it touches no "
            "pinned bytes and no fragment, tape or book. The per-window "
            "tripwire table is built on the REPLAY attribution, because a gap "
            "the replay never sees cannot contribute to delta-D."),
        "DEFERRED_FIX": (
            "re-attributing a boundary-spanning gap to the window that "
            "contains it is a change to flow_intensity.gaps_by_slug, which is "
            "in the PINNED build path. It lands after the last forward book "
            "(coordinator ruling, BE 139), never mid-test."),
    }


def falsify() -> int:
    """Rule 15: a positive control it must flag and a known-bad it refuses."""
    checks = []

    def note(n, ok):
        checks.append((n, bool(ok)))
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")

    W = 300.0
    base = 1788796200

    def row(start_off, end_off, ws=base):
        return {"gap_start_ns": int((ws + start_off) * 1e9),
                "gap_end_ns": int((ws + end_off) * 1e9),
                "window_start": ws, "slug": f"btc-updown-5m-{ws}"}

    # POSITIVE CONTROL: wholly inside its stamped window is SEEN.
    c = classify(row(71.759, 81.489), W)
    note("a gap wholly inside its stamped window is SEEN",
         c["status"] == SEEN and c["clamped_offsets"] == [71.759, 81.489]
         and c["attribution_agrees"] is True)

    # THE KNOWN-BAD: wholly PAST the stamped window's end.
    c = classify(row(319.259, 320.812), W)
    note("a gap wholly past its stamped window's end is NOT SEEN",
         c["status"] == NOT_SEEN)
    note("and the census names the window that actually CONTAINS it",
         c["wall_clock_window"] == base + 300
         and c["attribution_agrees"] is False)
    note("the two attributions DISAGREE on exactly that row",
         c["stamped_window"] == base and c["wall_clock_window"] == base + 300)

    # A STRADDLER is seen, clamped -- it is not silently dropped.
    c = classify(row(295.0, 305.0), W)
    note("a gap straddling the end is SEEN, clamped to the boundary",
         c["status"] == SEEN and c["clamped_offsets"] == [295.0, 300.0])

    # A gap BEFORE the window start (negative offsets) is not seen.
    c = classify(row(-20.0, -5.0), W)
    note("a gap entirely BEFORE the stamped window is NOT SEEN",
         c["status"] == NOT_SEEN)

    # A row with no interval is skipped, never counted as a zero-length gap.
    note("a row carrying no interval is skipped, not counted",
         classify({"window_start": base, "slug": "s"}, W) is None)
    note("a row with a start and no end is skipped",
         classify({"gap_start_ns": 1, "window_start": base, "slug": "s"}, W)
         is None)

    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_gap_census", "n": len(checks),
                      "n_failed": len(bad), "failed": bad}))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    days = [a for a in argv if a.isdigit()]
    if not days:
        print("usage: be_gap_census.py <YYYYMMDD> [...] | --selftest")
        return 2
    out = {}
    try:
        for d in days:
            out[d] = census(d)
    except CensusRefused as e:
        print(json.dumps({"refused": str(e)}, indent=1))
        return 1
    except Exception as e:                       # unreadable input
        print(json.dumps({"refused": f"{type(e).__name__}: {e}"}, indent=1))
        return 3
    p = Path("data/pm_5min/derived/be_gap_census_wallclock.json")
    p.write_text(json.dumps({"protocol": "BE_GAP_CENSUS_V1", "days": out},
                            indent=1) + "\n")
    for d, c in out.items():
        print(f"{d} era={c['era']} rows={c['n_gap_rows_with_an_interval']} "
              f"replay_windows={c['REPLAY_ATTRIBUTION']['n_windows']} "
              f"wallclock_windows={c['WALL_CLOCK_ATTRIBUTION']['n_windows']} "
              f"{NOT_SEEN}={c[NOT_SEEN]['n']} "
              f"gain={c[NOT_SEEN]['windows_that_gain_a_gap_under_wall_clock']}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
