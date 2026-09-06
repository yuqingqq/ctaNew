"""THE RACE READER. The code the declaration's strongest safeguard had none of.

BE48 §C.2, residual: *"`be_race_read_result_v1` appears in exactly two places
on the whole `live/` surface: the string literal in the declaration and the
emitted JSON. THERE IS NO READER."* So `REQUIRED_AFTER_THE_READ.who_runs_it`
-- *"whoever opens"* -- made the byte-identity recheck an instruction to a
person. This is the code it lives in.

WHAT IT DOES, EXACTLY AS DECLARATION v2 SPECIFIES AND NOTHING MORE:
  * opens the five `be_forward_day_SEALED_scores_<DAY>.json`;
  * computes the declared statistic -- the interim's WINDOW-LEVEL SIGN-FLIP
    OF PAIRED INCREMENTS per day, then the per-day SIGN, then the two
    permutation floors;
  * writes THE ONE declared artifact and nothing else;
  * RECOMPUTES the five sha256 after the read and VOIDS on any mismatch --
    "a read that moved the bytes it read is not a read, it is an edit";
  * states the Gate-1 separation FROM THE PATHS IT ACTUALLY OPENED.

THE SEPARATION FIELD IS NOT CONSTANT-VS-CONSTANT, WHICH WAS THE OTHER
RESIDUAL (§C.3). The declaration matched a module constant against a module
constant, so the field could only fire if its own constant were edited. Here
the haystack is `self.opened` -- the paths this run actually read -- so a run
that opened a Gate-1 object would say so.

IT DOES NOT RUN ON THE REAL FILES BY IMPORT. `--open` is required, the
opening is the coordinator's or the USER's act on GO, and the selftest drives
everything on SYNTHETIC sealed files it writes itself.

THE ROW SHAPE IS TAKEN FROM THE WRITER, NEVER FROM A SEALED FILE.
`be_forward_day.seal()` writes `per_coin_scores: {coin: [list(x) for x in v]}`
over `scores[coin].append((t0, value))` -- so a row is `[t0, value]`. The
reader REFUSES a shape it does not recognise rather than guessing, because a
statistic computed over a misread row is worse than no statistic.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import be_data_root as _BDR
import be_race_read_declaration as DECL

ROOT = HERE.parents[1]
OUT_NAME = "be_race_read_result_v1.json"

#: Substrings that would mean a Gate-1 object had entered the read's path.
GATE1_PATTERNS = DECL.GATE1_ARTIFACT_PATTERNS


class ReadVoid(RuntimeError):
    """The read is void. Never downgraded to a warning."""


class ReadRefused(RuntimeError):
    """A named refusal, before anything was consumed."""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assert_separation(opened) -> dict:
    """FROM THE PATHS ACTUALLY OPENED, not from a constant (§C.3)."""
    paths = [str(p) for p in opened]
    hits = sorted({f"{pat} in {p}" for pat in GATE1_PATTERNS
                   for p in paths if pat in p})
    if hits:
        raise ReadRefused(
            f"REFUSED: a Gate-1 object is on this read's path: {hits}. The "
            f"race read opens the forward scorer's sealed scores; the Gate-1 "
            f"objects are the day books and the arms' thetas. They do not "
            f"share a path, and a run where they do is not this read.")
    return {"no_gate1_artifact_on_the_read_path": True,
            "checked_patterns": list(GATE1_PATTERNS),
            "haystack": "the paths THIS RUN opened, not a module constant",
            "n_paths_checked": len(paths), "matches": []}


def day_statistic(per_coin_scores: dict) -> dict:
    """WINDOW-LEVEL SIGN-FLIP OF PAIRED INCREMENTS, then the day's sign.

    Rows are `[t0, value]` per coin, in the writer's own shape. Within a coin
    the rows are ordered by window start; the PAIRED INCREMENT is the change
    between consecutive windows, and the statistic is the net of those
    increments' signs -- flips up minus flips down. The DAY QUANTITY is that
    net; the DAY SIGN is its sign. A day with fewer than two windows in every
    coin has no increment and is a STATUS, never a zero (rule 4)."""
    if not isinstance(per_coin_scores, dict) or not per_coin_scores:
        raise ReadRefused("REFUSED: per_coin_scores is absent or not a "
                          "mapping; the sealed file is not the shape the "
                          "writer produces.")
    up = down = flat = 0
    n_rows = n_incr = 0
    for coin, rows in sorted(per_coin_scores.items()):
        if not isinstance(rows, list):
            raise ReadRefused(f"REFUSED: {coin} rows are "
                              f"{type(rows).__name__}, not a list.")
        vals = []
        for r in rows:
            if not (isinstance(r, (list, tuple)) and len(r) >= 2):
                raise ReadRefused(
                    f"REFUSED: a {coin} row is {r!r}, not the writer's "
                    f"[t0, value] shape. A statistic over a misread row is "
                    f"worse than no statistic.")
            vals.append((float(r[0]), float(r[1])))
        vals.sort()
        n_rows += len(vals)
        for a, b in zip(vals, vals[1:]):
            d = b[1] - a[1]
            n_incr += 1
            if d > 0:
                up += 1
            elif d < 0:
                down += 1
            else:
                flat += 1
    if n_incr == 0:
        return {"status": "NO_INCREMENT", "n_rows": n_rows,
                "why": "fewer than two windows in every coin; there is no "
                       "paired increment, and this is a STATUS not a zero"}
    net = up - down
    return {"status": "OK", "n_rows": n_rows, "n_increments": n_incr,
            "flips_up": up, "flips_down": down, "flat": flat,
            "day_quantity": net,
            "day_sign": (1 if net > 0 else (-1 if net < 0 else 0))}


def floors(g_optimistic: int, g_pessimistic: int, m: int = 2) -> dict:
    o, p = m / 2 ** g_optimistic, m / 2 ** g_pessimistic
    return {"optimistic": {"G": g_optimistic, "best_possible_adjusted_p": o},
            "pessimistic": {"G": g_pessimistic, "best_possible_adjusted_p": p},
            "resolved_best_possible_adjusted_p": max(o, p),
            "WHICH_ONE_IS_RESOLVED": "the CONSERVATIVE one, as declaration "
                                     "v2 requires",
            "neither_clears_0_05": max(o, p) > 0.05 and min(o, p) > 0.05}


def read(paths: dict, *, outdir: Path = None, write: bool = True) -> dict:
    """The read. Digests before, statistic, digests after, VOID on mismatch."""
    opened = [Path(v) for v in paths.values()]
    sep = assert_separation(opened)
    missing = [str(p) for p in opened if not p.exists()]
    if missing:
        raise ReadRefused(f"REFUSED: sealed file(s) absent: {missing}")
    before = {d: _sha(p) for d, p in paths.items()}
    per_day = {}
    for d, p in sorted(paths.items()):
        doc = json.loads(Path(p).read_text())
        per_day[d] = day_statistic(doc.get("per_coin_scores"))
    after = {d: _sha(p) for d, p in paths.items()}
    moved = sorted(d for d in before if before[d] != after[d])
    if moved:
        raise ReadVoid(
            f"REFUSED — THE READ IS VOID: the sealed bytes for {moved} "
            f"CHANGED between the digest taken before the read and the one "
            f"taken after. A read that moved the bytes it read is not a "
            f"read, it is an edit. No result is emitted.")
    signs = {d: v.get("day_sign") for d, v in per_day.items()}
    fresh = [d for d in paths if d not in
             DECL.ALREADY_OPENED_UNDER_THE_INTERIM]
    out = {
        "protocol": "BE_RACE_READ_RESULT_V1",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "R_529_A_UP_FRONT": "THIS READ ESTABLISHES DIRECTION AND CONSISTENCY "
                            "AND NEVER A HOLM-CLEARING VERDICT (USER ruling "
                            "R-529(A)).",
        "days": sorted(paths),
        "per_day": per_day,
        "day_signs": signs,
        "n_positive": sum(1 for v in signs.values() if v == 1),
        "n_negative": sum(1 for v in signs.values() if v == -1),
        "n_zero_or_status": sum(1 for v in signs.values() if not v),
        "permutation_floors": floors(len(paths), len(fresh)),
        "byte_identity": {"before": before, "after": after,
                          "all_unchanged": True,
                          "required_by": "declaration v2 "
                                         "REQUIRED_AFTER_THE_READ",
                          "on_mismatch": "the read is VOID and no result is "
                                         "emitted -- enforced here, not "
                                         "instructed"},
        "gate1_separation": sep,
        "writes": {"artifact": OUT_NAME, "and_nothing_else": True},
        "data_root": _BDR.receipt_block(),
        "decides_nothing": "REPORTED (rule 14). The USER weighs the "
                           "re-reads; this reader does not.",
    }
    if write:
        d = Path(outdir) if outdir is not None else _BDR.derived()
        (d / OUT_NAME).write_text(json.dumps(out, indent=1, sort_keys=True,
                                             default=str))
        out["_written"] = str(d / OUT_NAME)
    return out


EXPECTED_CHECKS = 9


def _synthetic(d: Path, day: str, rows) -> Path:
    p = d / f"be_forward_day_SEALED_scores_{day}.json"
    p.write_text(json.dumps({"protocol": "BE_FORWARD_DAY_SEALED_SCORES_V1",
                             "day": day, "SEALED": "synthetic fixture",
                             "per_coin_scores": {"btc": rows},
                             "report": {}}))
    return p


def selftest() -> int:
    import tempfile
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    st = day_statistic({"btc": [[1, 1.0], [2, 2.0], [3, 5.0]]})
    ok(st["flips_up"] == 2 and st["day_quantity"] == 2 and st["day_sign"] == 1,
       f"POSITIVE CONTROL: three rising windows give 2 up-flips, day "
       f"quantity {st['day_quantity']}, sign {st['day_sign']}")
    st2 = day_statistic({"btc": [[1, 5.0], [2, 2.0], [3, 1.0]]})
    ok(st2["day_sign"] == -1 and st2["flips_down"] == 2,
       "and three falling windows give the OPPOSITE sign -- the statistic "
       "tracks the data, not a constant")
    ok(day_statistic({"btc": [[1, 1.0]]})["status"] == "NO_INCREMENT",
       "a day with one window is NO_INCREMENT -- a STATUS, never a zero "
       "that would enter the sign count as agreement (rule 4)")
    try:
        day_statistic({"btc": [{"t0": 1}]})
        ok(False, "a wrong row shape must refuse")
    except ReadRefused as e:
        ok("not the writer's [t0, value] shape" in str(e),
           "KNOWN-BAD: a row that is not the writer's shape REFUSES -- a "
           "statistic over a misread row is worse than no statistic")

    f = floors(5, 3)
    ok(f["optimistic"]["best_possible_adjusted_p"] == 0.0625
       and f["pessimistic"]["best_possible_adjusted_p"] == 0.25
       and f["resolved_best_possible_adjusted_p"] == 0.25,
       "BOTH FLOORS computed and the CONSERVATIVE one resolved: G=5 -> "
       "0.0625, three fresh -> 0.25, resolved 0.25")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        paths = {"20260903": _synthetic(d, "20260903", [[1, 1.0], [2, 3.0]]),
                 "20260904": _synthetic(d, "20260904", [[1, 3.0], [2, 1.0]])}
        r = read(paths, outdir=d)
        ok(r["byte_identity"]["all_unchanged"] and r["n_positive"] == 1
           and r["n_negative"] == 1,
           "A CLEAN READ ADMITS: two synthetic days read, one sign each way, "
           "and the byte-identity check passes")
        ok((d / OUT_NAME).exists()
           and sorted(x.name for x in d.glob("be_race_read_*")) == [OUT_NAME],
           f"and it writes THE ONE declared artifact and nothing else "
           f"({OUT_NAME})")

        # TAMPER: a file that changes under the read must VOID it
        # TAMPER FOR REAL: the first version monkeypatched `_sha` by
        # call-count, which is brittle and did not fire. This MUTATES THE
        # FILE between the before- and after-digest, which is the actual
        # thing the safeguard exists to catch.
        # PATCH THIS MODULE'S OWN GLOBALS. `import be_race_reader` from
        # inside a run started as `python3 be_race_reader.py` binds a SECOND
        # module object -- the selftest patched that one and the running
        # `read` never saw it, so the falsifier silently did not fire. It
        # reported FAIL rather than passing vacuously, which is the only
        # reason this was visible.
        _g = globals()
        _orig_stat = _g["day_statistic"]

        def _mutate(pcs):
            # runs once per day, between the two digest passes
            Path(paths["20260904"]).write_text(
                Path(paths["20260904"]).read_text() + " ")
            return _orig_stat(pcs)
        _g["day_statistic"] = _mutate
        try:
            read(paths, outdir=d)
            ok(False, "a tampered sealed file must VOID the read")
        except ReadVoid as e:
            ok("THE READ IS VOID" in str(e) and "is not a read, it is an "
               "edit" in str(e),
               "KNOWN-BAD: bytes that differ between the before- and "
               "after-digest VOID the read, and NO result is emitted -- the "
               "declaration's safeguard now has code to run in")
        finally:
            _g["day_statistic"] = _orig_stat

        # a planted Gate-1 path in the opened set must REFUSE
        bad = dict(paths)
        bad["planted"] = d / "be_daybook_20260903_btc.pkl"
        try:
            assert_separation(list(bad.values()))
            ok(False, "a planted Gate-1 path must refuse")
        except ReadRefused as e:
            ok("Gate-1 object is on this read's path" in str(e)
               and "be_daybook_" in str(e),
               "KNOWN-BAD: a Gate-1 object planted into the OPENED set "
               "REFUSES -- and the haystack is the paths this run opened, "
               "not a module constant that only its own author can move")

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
    if "--open" in argv:
        out = read({d: Path(p) for d, p in DECL.SEALED_SCORES.items()})
        print(json.dumps({"written": out.get("_written"),
                          "day_signs": out["day_signs"],
                          "resolved_floor":
                              out["permutation_floors"][
                                  "resolved_best_possible_adjusted_p"]}))
        return 0
    print("usage: be_race_reader.py --selftest | --open   "
          "(--open CONSUMES the five sealed days; it is the coordinator's or "
          "the USER's act on GO, never this seat's)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
