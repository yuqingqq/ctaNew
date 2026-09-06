"""THE RACE READER, REBUILT TO THE DECLARED ESTIMAND — AND IT REFUSES.

R-581 / REVIEW_BE50_RACE_READER §A.2-A.4: my first reader computed a
SIGN-FLIP COUNT OF RAW SCORE INCREMENTS. That is not the declared estimand,
and the reviewer's known-bad exposed it: a collapsing series whose
within-window values are TIED, versus the same series perturbed by 1e-9,
gave DIFFERENT answers -- ties counted as `flat`, perturbations as up/down.
A statistic whose sign turns on 1e-9 is not measuring the thing.

THE DECLARED ESTIMAND IS READ FROM THE DECLARATION'S OWN FIELDS, never
re-typed here: NET CENTS against the INCUMBENT, at the unit of the ACTION
(slug, side, gen) DE-DUPLICATED, L = 50 ms, pairing BY_THRESHOLD. The day
quantity is the per-day NET; the day sign is its sign. Net cents does not
depend on counting flips, so ties and a 1e-9 perturbation give the SAME
sign -- which is the falsifier that killed the old one.

AND THEN IT REFUSES ON THE REAL FILES, FOR A REASON I CHECKED AT THE WRITER.
`be_forward_day` seals `out[coin].append((int(r["t0"]),
FS.expected_cancel_value(fit, fp + ff)))` -- a per-action EXPECTED CANCEL
VALUE from ONE fit. Those bytes carry:

    no INCUMBENT      (one fit, not a pair -- nothing to difference against)
    no ACTION IDENTITY beyond t0 (no slug, no side, no gen -- so the ruled
                       de-duplication unit cannot be formed)
    no REALISED CENTS  (an expected value is not a net)

**So the declared estimand is NOT COMPUTABLE from
`be_forward_day_SEALED_scores_<DAY>.json`.** A reader that produced a number
from those bytes and called it "net cents against the incumbent at the action
unit" would be fabricating three of the four things the estimand names. This
one REFUSES and says which fields are missing and where it looked.

That refusal is the finding. It is routed, not worked around: either the
sealed artifact must carry the action-level records the estimand needs, or
the estimand the race read declares must change. Both are rulings, and
neither is this seat's.

IT DOES NOT RUN ON THE REAL FILES. `--open` is the coordinator's or the
USER's act; everything below is driven on synthetic sealed files this module
writes itself.
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
GATE1_PATTERNS = DECL.GATE1_ARTIFACT_PATTERNS


class ReadVoid(RuntimeError):
    """The read is void. Never downgraded to a warning."""


class ReadRefused(RuntimeError):
    """A named refusal, before anything is concluded."""


def estimand() -> dict:
    """THE ESTIMAND, READ FROM THE DECLARATION. Never re-typed here.

    If the declaration changes, this reader changes with it; a re-typed copy
    is a second declaration that can disagree with the first."""
    e = DECL.statistic()["estimand"]
    return {"quantity": e["quantity"], "unit": e["unit_of_analysis"],
            "pairing": e["pairing_convention"],
            "latency": e["latency_axis"], "comparator": e["comparator"],
            "source": "be_race_read_declaration.statistic()['estimand']",
            "re_typed_here": False}


#: What an action-level record must carry for the DECLARED estimand to be
#: computable. Derived from the estimand's own words, not invented.
REQUIRED_ACTION_FIELDS = ("slug", "side", "gen", "net_cents_vs_incumbent")


def day_net_cents(actions) -> dict:
    """NET CENTS at the ACTION unit, DE-DUPLICATED. Ties are irrelevant.

    Rule 2: several rows can share one outcome, so the unit is the action
    (slug, side, gen) and duplicates are collapsed before summing. The sign
    of a SUM does not turn on whether two within-window values are equal --
    which is exactly why this passes the falsifier the flip-count failed."""
    if not isinstance(actions, list):
        raise ReadRefused(f"REFUSED: actions is {type(actions).__name__}.")
    seen, net, dupes = {}, 0.0, 0
    for a in actions:
        missing = [f for f in REQUIRED_ACTION_FIELDS if f not in a]
        if missing:
            raise ReadRefused(
                f"REFUSED: an action record is missing {missing}. The "
                f"declared estimand is net cents against the INCUMBENT at "
                f"the unit of the ACTION; a record without those fields "
                f"cannot supply it.")
        k = (a["slug"], a["side"], a["gen"])
        if k in seen:
            dupes += 1
            continue
        seen[k] = float(a["net_cents_vs_incumbent"])
    net = sum(seen.values())
    return {"status": "OK", "n_rows": len(actions), "n_actions": len(seen),
            "n_duplicates_collapsed": dupes,
            "day_net_cents": net,
            "day_sign": (1 if net > 0 else (-1 if net < 0 else 0)),
            "unit": "the ACTION (slug, side, gen), de-duplicated (rule 2)"}


def assert_estimand_supported(doc: dict, path) -> list:
    """Can THESE BYTES supply the declared estimand? Checked, then refused.

    The sealed scores' shape is read at the WRITER (`be_forward_day.seal`
    over `out[coin].append((int(r["t0"]), expected_cancel_value(...)))`), so
    this states what is missing rather than discovering it by crashing."""
    acts = doc.get("per_action_records")
    if isinstance(acts, list) and acts:
        return acts
    pcs = doc.get("per_coin_scores")
    if pcs is None:
        raise ReadRefused(f"REFUSED: {Path(path).name} carries neither "
                          f"`per_action_records` nor `per_coin_scores`.")
    raise ReadRefused(
        f"REFUSED — THE DECLARED ESTIMAND IS NOT COMPUTABLE FROM "
        f"{Path(path).name}. It carries `per_coin_scores`: rows of "
        f"[t0, expected_cancel_value] from ONE fit, written by "
        f"`be_forward_day.seal`. The estimand needs, and these bytes do not "
        f"have: (1) the INCUMBENT -- one fit is not a pair, so there is "
        f"nothing to difference against; (2) ACTION IDENTITY beyond t0 -- no "
        f"slug, no side, no gen, so the ruled de-duplication unit cannot be "
        f"formed; (3) REALISED CENTS -- an expected value is not a net. "
        f"Producing a number from these and calling it 'net cents against "
        f"the incumbent at the action unit' would fabricate three of the "
        f"four things the estimand names. ROUTED: either the sealed artifact "
        f"carries action-level records, or the declared estimand changes. "
        f"Both are rulings and neither is this seat's.")


def assert_separation(opened) -> dict:
    """FROM THE PATHS ACTUALLY OPENED, not from a constant."""
    paths = [str(p) for p in opened]
    hits = sorted({f"{pat} in {p}" for pat in GATE1_PATTERNS
                   for p in paths if pat in p})
    if hits:
        raise ReadRefused(
            f"REFUSED: a Gate-1 object is on this read's path: {hits}.")
    return {"no_gate1_artifact_on_the_read_path": True,
            "checked_patterns": list(GATE1_PATTERNS),
            "haystack": "the paths THIS RUN opened, not a module constant",
            "n_paths_checked": len(paths), "matches": []}


def floors(g_opt: int, g_pess: int, m: int = 2) -> dict:
    o, p = m / 2 ** g_opt, m / 2 ** g_pess
    return {"optimistic": {"G": g_opt, "best_possible_adjusted_p": o},
            "pessimistic": {"G": g_pess, "best_possible_adjusted_p": p},
            "resolved_best_possible_adjusted_p": max(o, p),
            "WHICH_ONE_IS_RESOLVED": "the CONSERVATIVE one",
            "neither_clears_0_05": min(o, p) > 0.05}


def read(paths: dict, *, outdir: Path = None, write: bool = True) -> dict:
    """Digest the parsed bytes, compute, digest again, VOID on mismatch."""
    opened = [Path(v) for v in paths.values()]
    sep = assert_separation(opened)
    missing = [str(p) for p in opened if not p.exists()]
    if missing:
        raise ReadRefused(f"REFUSED: sealed file(s) absent: {missing}")
    per_day, before = {}, {}
    for d, p in sorted(paths.items()):
        raw = Path(p).read_bytes()
        # THE DIGEST IS OF THE BYTES PARSED, not of a separate read.
        before[d] = hashlib.sha256(raw).hexdigest()
        doc = json.loads(raw)
        per_day[d] = day_net_cents(assert_estimand_supported(doc, p))
    after = {d: hashlib.sha256(Path(p).read_bytes()).hexdigest()
             for d, p in paths.items()}
    moved = sorted(d for d in before if before[d] != after[d])
    if moved:
        raise ReadVoid(
            f"REFUSED — THE READ IS VOID: the sealed bytes for {moved} "
            f"CHANGED between the digest of the bytes PARSED and the one "
            f"taken after. A read that moved the bytes it read is not a "
            f"read, it is an edit. No result is emitted.")
    signs = {d: v["day_sign"] for d, v in per_day.items()}
    fresh = [d for d in paths if d not in DECL.ALREADY_OPENED_UNDER_THE_INTERIM]
    out = {
        "protocol": "BE_RACE_READ_RESULT_V1",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "R_529_A_UP_FRONT": "THIS READ ESTABLISHES DIRECTION AND CONSISTENCY "
                            "AND NEVER A HOLM-CLEARING VERDICT (R-529(A)).",
        "estimand": estimand(),
        "days": sorted(paths), "per_day": per_day, "day_signs": signs,
        "n_positive": sum(1 for v in signs.values() if v == 1),
        "n_negative": sum(1 for v in signs.values() if v == -1),
        "n_zero": sum(1 for v in signs.values() if v == 0),
        "permutation_floors": floors(len(paths), len(fresh)),
        "byte_identity": {"before": before, "after": after,
                          "all_unchanged": True,
                          "digest_is_of_the_bytes_parsed": True,
                          "on_mismatch": "the read is VOID -- enforced, not "
                                         "instructed"},
        "gate1_separation": sep,
        "writes": {"artifact": OUT_NAME, "and_nothing_else": True},
        "data_root": _BDR.receipt_block(),
        "decides_nothing": "REPORTED (rule 14).",
    }
    if write:
        d = Path(outdir) if outdir is not None else _BDR.derived()
        (d / OUT_NAME).write_text(json.dumps(out, indent=1, sort_keys=True,
                                             default=str))
        out["_written"] = str(d / OUT_NAME)
    return out


EXPECTED_CHECKS = 10


def _seal(d: Path, day: str, *, actions=None, scores=None) -> Path:
    body = {"protocol": "BE_FORWARD_DAY_SEALED_SCORES_V1", "day": day,
            "SEALED": "synthetic fixture", "report": {}}
    if actions is not None:
        body["per_action_records"] = actions
    if scores is not None:
        body["per_coin_scores"] = {"btc": scores}
    p = d / f"be_forward_day_SEALED_scores_{day}.json"
    p.write_text(json.dumps(body))
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

    e = estimand()
    ok("NET CENTS" in e["quantity"] and "ACTION" in e["unit"]
       and e["pairing"] == "BY_THRESHOLD" and not e["re_typed_here"],
       f"THE ESTIMAND IS READ FROM THE DECLARATION: {e['quantity']!r} at "
       f"{e['unit'][:34]!r}, pairing {e['pairing']} -- not re-typed here")

    # ---- THE REVIEWER'S KNOWN-BAD, THE ONE THAT KILLED THE OLD READER ----
    def collapsing(perturb):
        return [{"slug": "s", "side": "BUY_UP", "gen": i,
                 "net_cents_vs_incumbent": -1.0 + (i * perturb)}
                for i in range(6)]
    tied = day_net_cents(collapsing(0.0))
    pert = day_net_cents(collapsing(1e-9))
    ok(tied["day_sign"] == pert["day_sign"] == -1,
       f"THE FALSIFIER THAT EXPOSED THE OLD READER: a collapsing series with "
       f"within-window values TIED and the same series perturbed by 1e-9 "
       f"give the SAME sign ({tied['day_sign']}). The old flip-count did "
       f"not; a NET does not turn on 1e-9")
    ok(abs(tied["day_net_cents"] - pert["day_net_cents"]) < 1e-6,
       f"and the quantities agree to 1e-6 ({tied['day_net_cents']:.6f} vs "
       f"{pert['day_net_cents']:.6f}) -- the perturbation moves the number "
       f"by less than it could move a sign")

    # ---- a KNOWN net, reproduced by construction -------------------------
    acts = [{"slug": "s", "side": "BUY_UP", "gen": 1,
             "net_cents_vs_incumbent": 3.5},
            {"slug": "s", "side": "BUY_UP", "gen": 2,
             "net_cents_vs_incumbent": -1.25},
            {"slug": "s", "side": "BUY_UP", "gen": 2,
             "net_cents_vs_incumbent": 99.0}]        # duplicate action
    r = day_net_cents(acts)
    ok(abs(r["day_net_cents"] - 2.25) < 1e-9 and r["n_actions"] == 2
       and r["n_duplicates_collapsed"] == 1,
       f"A KNOWN NET REPRODUCES BY CONSTRUCTION: 3.5 + (-1.25) = "
       f"{r['day_net_cents']}, with the repeated (slug, side, gen) "
       f"DE-DUPLICATED (rule 2) -- the 99.0 duplicate does not enter")

    # ---- the REAL shape refuses, naming what is missing -------------------
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        real = _seal(d, "20260903", scores=[[1, 0.5], [2, 0.7]])
        try:
            read({"20260903": real}, outdir=d)
            ok(False, "the real sealed shape must refuse")
        except ReadRefused as ex:
            ok("NOT COMPUTABLE" in str(ex) and "INCUMBENT" in str(ex)
               and "ACTION IDENTITY" in str(ex) and "REALISED CENTS" in str(ex),
               "KNOWN-BAD, AND IT IS THE REAL ARTIFACT'S SHAPE: "
               "`per_coin_scores` REFUSES, naming all three missing inputs "
               "-- the incumbent, action identity, and realised cents")

        p1 = _seal(d, "20260903", actions=acts)
        p2 = _seal(d, "20260904", actions=[{"slug": "t", "side": "SELL_UP",
                                            "gen": 1,
                                            "net_cents_vs_incumbent": -4.0}])
        paths = {"20260903": p1, "20260904": p2}
        res = read(paths, outdir=d)
        ok(res["byte_identity"]["all_unchanged"]
           and res["byte_identity"]["digest_is_of_the_bytes_parsed"]
           and res["n_positive"] == 1 and res["n_negative"] == 1,
           "A CLEAN READ ADMITS on action-level records, one sign each way, "
           "with the digest taken from THE BYTES PARSED")
        ok((d / OUT_NAME).exists()
           and sorted(x.name for x in d.glob("be_race_read_*")) == [OUT_NAME],
           f"and it writes THE ONE declared artifact and nothing else")
        f = res["permutation_floors"]
        ok(f["optimistic"]["best_possible_adjusted_p"] == 0.5
           and f["pessimistic"]["best_possible_adjusted_p"] == 0.5
           and f["resolved_best_possible_adjusted_p"] == 0.5
           and floors(5, 3)["resolved_best_possible_adjusted_p"] == 0.25,
           f"both floors are COMPUTED and the CONSERVATIVE one resolved: on "
           f"these two fixture days 2/2^2 = 0.5, and on the real 5/3 split "
           f"the resolved floor is 0.25 (not the flattering 0.0625)")

        _g = globals()
        _orig = _g["day_net_cents"]

        def _mutate(a):
            Path(p2).write_text(Path(p2).read_text() + " ")
            return _orig(a)
        _g["day_net_cents"] = _mutate
        try:
            read(paths, outdir=d)
            ok(False, "a tampered sealed file must VOID the read")
        except ReadVoid as ex:
            ok("THE READ IS VOID" in str(ex),
               "KNOWN-BAD: bytes mutated between the parse-digest and the "
               "after-digest VOID the read, and no result is emitted")
        finally:
            _g["day_net_cents"] = _orig

        try:
            assert_separation([p1, d / "be_daybook_20260903_btc.pkl"])
            ok(False, "a planted Gate-1 path must refuse")
        except ReadRefused as ex:
            ok("Gate-1 object is on this read's path" in str(ex),
               "KNOWN-BAD: a Gate-1 object planted into the OPENED set "
               "REFUSES -- the haystack is what this run opened")

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
                          "day_signs": out["day_signs"]}))
        return 0
    print("usage: be_race_reader.py --selftest | --open  (--open CONSUMES the "
          "five sealed days; the coordinator's or the USER's act on GO)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
