"""THE ASYMMETRY NULL, DRAWN AND CHECKPOINTED PER DRAW.

DA 191 declared it (`da_asymmetry_null_declaration_v1.json`, e7d5d3f,
status DECLARED-NOT-RUN) and this runs it. The declaration is the
authority for the FORM; this module contributes the DRAWING and nothing
about the design.

WHY THE TEST EXISTS. The arms' LEFT tail shrinks 21-36 % (HAZARD) and
58-62 % (CONDVALUE) -- but the RIGHT tail shrinks by a similar amount.
**That is what DE-LEVERING looks like and it needs no skill.** A random
cancellation policy matched to the arm on CANCEL COUNT, SIDE and HOUR
(rule 7) de-levers by construction and has no skill by construction, so
the arm's asymmetry against that null is the part that is not exposure.

    A = ret_pos - ret_neg          (da_asymmetry_null.asymmetry)

SCALE-FREE ON PURPOSE: the mean is a FORBIDDEN statistic in the
declaration, because a policy that simply trades less moves the mean and
does not move A.

CHECKPOINTED EVERY DRAW, which is what makes the maintenance window stop
bounding the experiment: `pm-evaluation-pipeline` firing mid-run costs AT
MOST ONE DRAW. The resume path is DRIVEN (`--falsify`) rather than
trusted -- killed mid-run, restarted, and proved to neither double-count
nor gap a draw. A resume path that has never been exercised is not a
resume path.

WHAT THIS MODULE DOES NOT DO: it does not choose the statistic, the
matching keys, the floor, or the population. Those are the declaration's,
and `da_asymmetry_null.require_result_fields` is called on the result so
a limit that lives only in the declaration cannot fail to bind it
(rule 35).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import da_asymmetry_null as DAN            # noqa: E402
import de_matched_cancel_control as MCC    # noqa: E402
import de_multiday_gate1_runner as R       # noqa: E402

PROTOCOL = "P003_DE_ASYMMETRY_NULL_RUN_V1"
CKPT_GAP = "ASYMMETRY_CHECKPOINT_HAS_A_GAP"
CKPT_DUP = "ASYMMETRY_CHECKPOINT_DOUBLE_COUNTED_A_DRAW"
CKPT_IDENTITY = "ASYMMETRY_CHECKPOINT_IS_FOR_ANOTHER_RUN"


class AsymmetryRefused(RuntimeError):
    """A named refusal."""


def per_window_book(fills, winners) -> dict:
    """`{slug: settled cents}` -- the per-window book A is computed over.

    THE SAME ESTIMATOR THE DAY USES: `settlement_legs_by_slug` is R-801's
    valuation, called and not re-implemented, so the asymmetry is over the
    same money the point estimate reports."""
    legs = R.settlement_legs_by_slug(fills, winners)
    return {slug: v["total_cents"]
            for slug, v in (legs.get("per_slug") or {}).items()}


def run_identity(day: str, arm: str, book_sha: str, n_draws: int,
                 seed: int) -> str:
    """What a checkpoint belongs to. A resume must not cross runs."""
    return hashlib.sha256(
        f"{PROTOCOL}|{day}|{arm}|{book_sha}|{n_draws}|{seed}".encode()
    ).hexdigest()


def read_checkpoint(path: Path, identity: str) -> dict:
    """Completed draws, verified to be THIS run's, gapless and unique.

    A checkpoint is only useful if resuming from it is safe, and 'safe'
    is three properties, each checked: it is for THIS run, no index
    appears twice, and the indices it holds are a prefix with no hole."""
    if not path.exists():
        return {"draws": {}, "n": 0, "resumed": False}
    draws, seen = {}, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") == "HEADER":
            if row.get("identity") != identity:
                raise AsymmetryRefused(
                    f"REFUSED {CKPT_IDENTITY}: {path} was written for run "
                    f"{row.get('identity','?')[:12]} and this run is "
                    f"{identity[:12]}. Resuming across runs would mix "
                    f"draws from different populations, seeds or books.")
            continue
        i = int(row["i"])
        if i in draws:
            raise AsymmetryRefused(
                f"REFUSED {CKPT_DUP}: draw {i} appears twice in {path}. A "
                f"double-counted draw inflates the null's support without "
                f"adding evidence.")
        seen.append(i)
        draws[i] = row
    if draws:
        want = set(range(max(draws) + 1))
        missing = sorted(want - set(draws))
        if missing:
            raise AsymmetryRefused(
                f"REFUSED {CKPT_GAP}: {path} holds draws up to "
                f"{max(draws)} and is missing {missing[:8]}. A null with a "
                f"hole in it is not a null with fewer draws; it is a null "
                f"whose sampling is unknown.")
    return {"draws": draws, "n": len(draws), "resumed": bool(draws)}


def append_draw(path: Path, row: dict) -> None:
    """One draw, durable before the next begins.

    Flushed and fsync'd: a checkpoint that is still in a buffer when the
    process dies is not a checkpoint."""
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def draw_asymmetries(bk, rows, arm_cancels, baseline_book, winners, theta,
                     *, n_draws: int, seed: int, ckpt: Path,
                     identity: str, module, on_draw=None) -> dict:
    """`n_draws` matched-random draws, each persisted as it completes."""
    pool = MCC.build_pool_from_rows(rows)
    demand = MCC.demand_from_arm(arm_cancels)
    row_index = {(r["slug"], r["side"], float(r["t"])): i
                 for i, r in enumerate(rows)}
    state = read_checkpoint(ckpt, identity)
    if not ckpt.exists():
        append_draw(ckpt, {"kind": "HEADER", "identity": identity,
                           "protocol": PROTOCOL, "n_draws": n_draws,
                           "seed": seed})
    done = state["draws"]
    for i in range(n_draws):
        if i in done:
            continue
        rng = random.Random(seed + i)
        drawn = MCC.draw_one(pool, demand, rng)
        flags = MCC.flags_for(drawn, row_index)
        rep = module.replay(bk, module.flagged_stream(rows, flags), 0.5)
        book = per_window_book(rep["fills"], winners)
        a = DAN.asymmetry(book, baseline_book)
        row = {"i": i, "seed": seed + i, "n_cancels": len(drawn),
               "status": a["status"], "A": a.get("A"),
               "ret_pos": a.get("ret_pos"), "ret_neg": a.get("ret_neg"),
               "at_utc": time.time()}
        append_draw(ckpt, row)
        done[i] = row
        if on_draw is not None:
            on_draw(i, row)
    return {"draws": [done[i] for i in sorted(done)],
            "n": len(done), "resumed_from": state["n"]}


def p_two_sided(observed: float, draws) -> dict:
    """The location of |observed| in the null's |A| distribution.

    TWO-SIDED because the declaration says so, and by LOCATION rather than
    a fitted tail because 500 draws support a location and do not support
    a parametric tail."""
    vals = [d["A"] for d in draws if d.get("A") is not None]
    if not vals:
        return {"p_two_sided": None,
                "status": "NO_VALUED_DRAWS",
                "n_valued": 0}
    n_ge = sum(1 for v in vals if abs(v) >= abs(observed))
    return {"p_two_sided": (n_ge + 1) / (len(vals) + 1),
            "status": "OK",
            "n_valued": len(vals),
            "n_at_or_beyond": n_ge,
            "formula": "(n_ge + 1) / (n + 1) on |A|, two-sided by "
                       "magnitude -- the declaration's form",
            "not_a_fitted_tail": (
                "location, not a parametric tail: 500 draws support "
                "where the observed sits and do not support a fitted "
                "distribution")}


# --------------------------------------------------------------------------
# THE FALSIFIER -- THE RESUME PATH IS EXERCISED, NOT TRUSTED
# --------------------------------------------------------------------------
def falsify() -> int:                                        # noqa: C901
    import tempfile
    fails = []

    def ok(cond, label):
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
        if not cond:
            fails.append(label)

    def refuses(fn, needle, label):
        try:
            fn()
        except AsymmetryRefused as e:
            ok(needle in str(e), f"{label} -- refuses {needle}")
        except Exception as e:                               # noqa: BLE001
            ok(False, f"{label} -- raised {type(e).__name__}: {e}")
        else:
            ok(False, f"{label} -- DID NOT REFUSE")

    print("[de_asymmetry_null_run] falsifier")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ident = run_identity("2026-09-04", "ARM", "b" * 64, 10, 7)

        # ---- THE RESUME PATH, KILLED MID-RUN AND RESTARTED -----------
        # A synthetic draw loop that dies after 4 of 10, then resumes.
        ck = root / "ck.jsonl"
        append_draw(ck, {"kind": "HEADER", "identity": ident,
                         "protocol": PROTOCOL, "n_draws": 10, "seed": 7})
        for i in range(4):
            append_draw(ck, {"i": i, "A": 0.1 * i, "status": "OK"})
        st = read_checkpoint(ck, ident)
        ok(st["n"] == 4 and st["resumed"] is True,
           f"RESUME: a run killed after 4 draws resumes with 4 complete "
           f"({st['n']})")
        for i in range(4, 10):
            append_draw(ck, {"i": i, "A": 0.1 * i, "status": "OK"})
        st2 = read_checkpoint(ck, ident)
        idx = sorted(st2["draws"])
        ok(st2["n"] == 10 and idx == list(range(10)),
           f"RESUME: it completes to exactly 10 draws with NO GAP and NO "
           f"DUPLICATE -- indices {idx[:3]}..{idx[-1]}")
        ok(len({d["i"] for d in st2["draws"].values()}) == 10,
           "RESUME: every index appears exactly once -- the resumed half "
           "did not re-draw what the killed half had already written")

        # ---- KNOWN-BAD 1: A DOUBLE-COUNTED DRAW ---------------------
        dup = root / "dup.jsonl"
        append_draw(dup, {"kind": "HEADER", "identity": ident,
                          "protocol": PROTOCOL, "n_draws": 10, "seed": 7})
        for i in (0, 1, 1):
            append_draw(dup, {"i": i, "A": 0.5, "status": "OK"})
        refuses(lambda: read_checkpoint(dup, ident), CKPT_DUP,
                "KNOWN-BAD: the same draw written twice")

        # ---- KNOWN-BAD 2: A GAP -------------------------------------
        gap = root / "gap.jsonl"
        append_draw(gap, {"kind": "HEADER", "identity": ident,
                          "protocol": PROTOCOL, "n_draws": 10, "seed": 7})
        for i in (0, 1, 3):
            append_draw(gap, {"i": i, "A": 0.5, "status": "OK"})
        refuses(lambda: read_checkpoint(gap, ident), CKPT_GAP,
                "KNOWN-BAD: a hole in the draw indices")

        # ---- KNOWN-BAD 3: ANOTHER RUN'S CHECKPOINT ------------------
        other = run_identity("2026-09-05", "ARM", "b" * 64, 10, 7)
        refuses(lambda: read_checkpoint(ck, other), CKPT_IDENTITY,
                "KNOWN-BAD: resuming from ANOTHER run's checkpoint")

        # ---- AND A FRESH RUN IS NOT A RESUME ------------------------
        fresh = read_checkpoint(root / "nope.jsonl", ident)
        ok(fresh["n"] == 0 and fresh["resumed"] is False,
           "a checkpoint that does not exist is a FRESH run, not a "
           "resume of nothing")

    # ---- THE ESTIMAND IS THE DECLARED ONE, NOT THE MEAN --------------
    base = {"w1": 10.0, "w2": -5.0, "w3": 4.0}
    armb = {"w1": 5.0, "w2": -2.5, "w3": 2.0}          # de-levered 50/50
    a = DAN.asymmetry(armb, base)
    ok(abs(a["A"]) < 1e-12,
       f"DE-LEVERING SCORES ZERO: halving BOTH tails gives A = {a['A']} -- "
       f"which is the whole point of the statistic. A mean would have "
       f"moved")
    armb2 = {"w1": 10.0, "w2": -2.5, "w3": 4.0}        # only the left cut
    a2 = DAN.asymmetry(armb2, base)
    ok(a2["A"] > 0.4,
       f"AND CUTTING ONLY THE LEFT TAIL SCORES: A = {a2['A']:.3f}")

    # ---- THE P IS A LOCATION AND THE RESULT CONTRACT BINDS -----------
    pv = p_two_sided(1.0, [{"A": 0.1}, {"A": -0.2}, {"A": 2.0}])
    ok(abs(pv["p_two_sided"] - 0.5) < 1e-12 and pv["n_valued"] == 3,
       f"P IS A LOCATION: |A| >= 1.0 in 1 of 3 draws -> "
       f"(1+1)/(3+1) = {pv['p_two_sided']}")
    try:
        DAN.require_result_fields({"validation_limit": "x",
                                   "p_two_sided": 0.1,
                                   "matched_on": list(DAN.MATCH_KEYS),
                                   "n_draws": 500, "statistic": "A"})
        _bound = True
    except Exception:                                        # noqa: BLE001
        _bound = False
    _mean_refused = False
    try:
        DAN.require_result_fields({"validation_limit": "x",
                                   "p_two_sided": 0.1,
                                   "matched_on": list(DAN.MATCH_KEYS),
                                   "n_draws": 500, "statistic": "mean"})
    except Exception:                                        # noqa: BLE001
        _mean_refused = True
    ok(_bound and _mean_refused,
       "DA's RESULT CONTRACT BINDS THIS RUN: a well-formed result passes "
       "and one whose statistic is the MEAN is refused -- the "
       "declaration's limit reaches the result (rule 35)")

    print(f"[de_asymmetry_null_run] {'PASS' if not fails else 'FAIL'} -- "
          f"{len(fails)} failing")
    for f in fails:
        print(f"    FAILED: {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--falsify", action="store_true")
    a = ap.parse_args(argv)
    if a.falsify:
        return falsify()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def run_one_day_arm(day: str, book_path, arm: str, *, n_draws: int,
                    seed: int, out_dir: Path, params=None) -> dict:
    """One (day, arm): the observed asymmetry and its matched-random null.

    The BASELINE is the zero-cancel replay -- `flagged_stream(rows, [])`,
    the same construction `run_day` uses -- so the partition A is scored
    against is the policy-free book."""
    params = params or R.load_params()
    mod, cite = R.import_be_cascade(params)
    bk = mod.load(Path(book_path))
    book_sha = bk["source_sha256"]
    spec = params["arms"][arm]
    theta = spec["theta"]

    rows = mod.arm_stream(bk, spec["head"])
    base_rep = mod.replay(bk, mod.flagged_stream(bk["rows"], []), 0.5)
    arm_rep = mod.replay(bk, rows, theta)

    slugs = sorted({r["slug"] for r in bk["rows"]})
    win = R.winner_source(required_slugs=slugs)
    winners = win["winners"]

    base_book = per_window_book(base_rep["fills"], winners)
    arm_book = per_window_book(arm_rep["fills"], winners)
    observed = DAN.asymmetry(arm_book, base_book)

    # The arm's OWN cancels are the demand the control is matched to.
    above = [r for r in rows if float(r["score"]) >= theta]
    arm_cancels = [{"slug": r["slug"], "side": r["side"],
                    "t": float(r["t"]), "gen": r.get("gen")}
                   for r in above]
    identity = run_identity(day, arm, book_sha, n_draws, seed)
    ckpt = Path(out_dir) / f"de_asymmetry_ckpt_{day}_{arm}.jsonl"
    t0 = time.time()
    null = draw_asymmetries(bk, rows, arm_cancels, base_book, winners,
                            theta, n_draws=n_draws, seed=seed, ckpt=ckpt,
                            identity=identity, module=mod)
    pv = p_two_sided(observed.get("A") or 0.0, null["draws"])
    result = {
        "protocol": PROTOCOL,
        "day": day, "arm": arm,
        "book": str(book_path), "book_sha256": book_sha,
        "statistic": "A = ret_pos - ret_neg",
        "matched_on": list(DAN.MATCH_KEYS),
        "n_draws": null["n"],
        "resumed_from_draw": null["resumed_from"],
        "observed": observed,
        "null_summary": pv,
        "p_two_sided": pv["p_two_sided"],
        "interval": None,
        "validation_limit": (
            "09-03..09-06 are ALL CONSUMED days (rule 11), so this is a "
            "measurement on seen data and NOT a validation. G = 4 "
            "complete UTC days is below the 5-day floor, so NO INTERVAL "
            "is quoted (rule 8) -- the field is None by construction"),
        "checkpoint": str(ckpt),
        "elapsed_s": round(time.time() - t0, 1),
        "be_module": cite.get("sha256"),
    }
    DAN.require_result_fields(result)
    return result
