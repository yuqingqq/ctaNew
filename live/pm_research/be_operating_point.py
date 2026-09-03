#!/usr/bin/env python3
"""BUILD THE FROZEN_FROM_TRAIN_QUANTILE OPERATING POINT — from consumed rows only.

WHAT THIS PRODUCES. A theta map {coin: {budget: threshold}}, where theta at
budget b is the score that would cancel exactly `int(n_actions * b)` of the
TRAINING split's generations, ranked by per-generation MAX score. That is the
form R-497 (F)(2) selected: causal, budget-parameterised, and declarable before
any forward day opens.

WHY IT COSTS THE RACE NOTHING. It reads `harmful_exposure_rows_v3_eraB.json`,
the population the freeze's own manifest binds by sha256, covering 2026-08-24
and 08-25 -- the freeze's training split, already consumed under CLAUDE.md
rule 11. No seal is opened, no forward day is read, no unspent day is touched.

WHY THE PROVENANCE IS BYTES AND NOT A LABEL (BEM-R2). The reviewer built a
theta map from the quantiles of the rows being SCORED, labelled it
`FROZEN_FROM_TRAIN_QUANTILE`, and `require_operating_point` accepted it as
causal -- because the form was a string and the numbers arrived as a bare
`{budget: float}` dict with no derivation a checker could see. So this module
emits, and the fence recomputes: the fit artifact's path and sha256, the rows
artifact's path and sha256, the SPLIT the quantiles were taken over as an
explicit day list, and a digest of the map itself. The overlap between that
declared split and the population actually being scored is then a COMPUTED
refusal at the point of use, which is the only place the question can be
answered at all.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harmful_forward_scorer as FS
import phase2_declaration as PD

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"

#: The population the freeze's manifest binds. Not chosen here.
TRAIN_ROWS = DERIVED / "harmful_exposure_rows_v3_eraB.json"
MANIFEST = DERIVED / "harmful_candidate_manifest_v1.json"

DECLARATION_PATH = (Path(__file__).resolve().parent / "declarations"
                    / "be_operating_point_declaration_v1.json")


class OperatingPointBuildRefused(RuntimeError):
    """A named refusal."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def theta_map_digest(theta_map: dict) -> str:
    """A digest of the MAP ITSELF, so the numbers that ran can be tied to the
    numbers that were declared."""
    return hashlib.sha256(json.dumps(theta_map, sort_keys=True,
                                     separators=(",", ":")).encode()
                          ).hexdigest()[:16]


def stream_rows(path: Path):
    """Yield row dicts one at a time from a {"rows": [...]} file.

    The artifact is 1.2 GB; parsing it whole costs several GB of Python
    objects for a pass that needs one row at a time. Scanned with a brace
    counter that respects strings and escapes."""
    with open(path, "r") as f:
        buf = f.read(1 << 20)
        i = buf.index("[") + 1
        depth = 0
        start = None
        in_str = False
        esc = False
        while True:
            while i < len(buf):
                ch = buf[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield json.loads(buf[start:i + 1])
                        buf = buf[i + 1:]
                        i = -1
                        start = None
                elif ch == "]" and depth == 0:
                    return
                i += 1
            more = f.read(1 << 20)
            if not more:
                return
            buf += more
            i = max(i, 0)


def score_training_split(days=("2026-08-24", "2026-08-25"), coins=("btc", "eth"),
                         rows_path: Path = None, progress=None) -> dict:
    """Per-generation MAX score over the consumed split, with the FROZEN fit.

    One `window_streams` load per slug, rows scored, stream dropped -- the
    same streaming discipline `be_forward_day.build_and_score` uses, and for
    the same reason."""
    import harmful_hazard_model as hm
    rows_path = Path(rows_path or TRAIN_ROWS)
    # NO BACKDOOR. The builder needs the fit before any expectation exists,
    # so it computes the identity and binds to it -- self-consistent, and it
    # says so in the declaration it emits. The place where self-consistency
    # is NOT good enough is the SCORING call site, where the expectation must
    # come from a committed declaration rather than from the artifact itself
    # (BEM-R3). A `_internal_build=True` escape hatch here would be the thing
    # a future caller reaches for.
    _ident = FS.candidate_identity()
    frozen = FS.load_frozen(expect=_ident)
    paths = hm.fi._archive_paths()
    tokens = hm.fi.token_map()
    gmax: dict = {}
    _identity = _ident
    counters = {"rows_seen": 0, "rows_in_split": 0, "rows_scored": 0,
                "rows_without_features": 0, "slugs": 0,
                "slugs_missing_archive": 0}
    cur_slug = None
    stream = None
    for r in stream_rows(rows_path):
        counters["rows_seen"] += 1
        if r.get("day") not in days or r.get("coin") not in coins:
            continue
        counters["rows_in_split"] += 1
        slug = r["slug"]
        if slug != cur_slug:
            del stream
            stream = None
            cur_slug = slug
            counters["slugs"] += 1
            if progress and counters["slugs"] % 100 == 0:
                progress(counters)
            if slug in paths and slug in tokens:
                stream = hm.window_streams(paths[slug], *tokens[slug])
            else:
                counters["slugs_missing_archive"] += 1
        if stream is None:
            continue
        coin = r["coin"]
        fit = frozen["fits"].get(coin)
        if fit is None:
            continue
        fp = hm.features(stream, r["t_start"], r["side"], r.get("level"),
                         r.get("resting"), r.get("qahead"))
        ff = hm.fine_feats(r["t0"] + r["t_start"], r["side"], coin)
        if fp is None or ff is None:
            counters["rows_without_features"] += 1
            continue
        counters["rows_scored"] += 1
        s = FS.expected_cancel_value(fit, fp + ff)
        k = (coin, r["slug"], r["side"], r["gen"])
        if s > gmax.get(k, -float("inf")):
            gmax[k] = s
    del stream
    return {"gmax": gmax, "counters": counters,
            "candidate_identity": _identity}


def theta_at_budgets(gmax: dict, coins, budgets=None) -> dict:
    """theta_b = the score that cancels exactly int(n*b) generations.

    The k-th largest per-generation maximum, matching `_select_by_count`'s
    cutoff exactly -- but taken on the TRAINING split, which is what makes it
    an input to the forward read rather than a reading of it."""
    budgets = tuple(budgets if budgets is not None else PD.BUDGETS)
    out: dict = {}
    for coin in coins:
        vals = sorted((v for (c, *_), v in gmax.items() if c == coin),
                      reverse=True)
        if not vals:
            raise OperatingPointBuildRefused(
                f"REFUSED: no scored generations for {coin!r}. An empty "
                f"quantile is not a threshold (R-141).")
        per = {}
        for b in budgets:
            kk = max(1, int(len(vals) * b))
            per[f"{int(b * 100)}%"] = float(vals[kk - 1])
        out[coin] = per
    return out


def build_declaration(days=("2026-08-24", "2026-08-25"),
                      coins=("btc", "eth"), budgets=None,
                      rows_path: Path = None, progress=None) -> dict:
    """The declaration artifact: the theta map AND everything a fence needs to
    recompute where it came from."""
    rows_path = Path(rows_path or TRAIN_ROWS)
    budgets = tuple(budgets if budgets is not None else PD.BUDGETS)
    scored = score_training_split(days=days, coins=coins,
                                  rows_path=rows_path, progress=progress)
    theta = theta_at_budgets(scored["gmax"], coins, budgets)
    man = json.loads(MANIFEST.read_text())
    bound = (man.get("hashes") or {}).get(
        "data/pm_5min/derived/harmful_exposure_rows_v3_eraB.json")
    rows_sha = sha256_file(rows_path)
    return {
        "protocol": "BE_OPERATING_POINT_DECLARATION_V1",
        "form": "FROZEN_FROM_TRAIN_QUANTILE",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "theta_frozen_by_coin": theta,
        "theta_map_sha16": theta_map_digest(theta),
        "budgets": [f"{int(b * 100)}%" for b in budgets],
        "budgets_source": "phase2_declaration.BUDGETS",
        "budgets_module_sha256": hashlib.sha256(
            Path(PD.__file__).read_bytes()).hexdigest(),
        "derived_from_split": {
            "days": list(days), "coins": list(coins),
            "population": "v3_4_consumed_fragment",
            "already_consumed": True,
            "why_free": ("the freeze's OWN training split, consumed under "
                         "CLAUDE.md rule 11; reading it opens no seal and "
                         "spends no forward day"),
        },
        "rows_artifact": {
            "path": str(rows_path), "sha256": rows_sha,
            "bytes": rows_path.stat().st_size,
            "manifest_binds_this_sha": bound == rows_sha,
            "manifest_sha256": bound,
            "manifest_path": str(MANIFEST),
        },
        "fit_artifact": scored["candidate_identity"],
        "counters": scored["counters"],
        "quantile_rule": ("theta_b is the k-th largest per-generation MAX "
                          "score with k = max(1, int(n_generations * b)) -- "
                          "the same cutoff `_select_by_count` computes, taken "
                          "on the TRAINING split so it is an INPUT to the "
                          "forward read rather than a reading of it"),
        "selected_by_this_module": False,
        "who_selects": ("nobody: R-497 (F)(2) rules declare-a-grid / "
                        "report-every-cell / select-none"),
    }


EXPECTED_CHECKS = 19


def selftest() -> int:
    """Falsifiers for the two things this module actually computes: the
    streaming reader and the quantile rule."""
    import tempfile
    checks = 0
    fails = []

    def ok(c, label):
        nonlocal checks
        checks += 1
        print(f"PASS: {label}" if c else f"FAIL: {label}")
        if not c:
            fails.append(label)

    with tempfile.TemporaryDirectory() as td:
        # -- the streaming reader, against a file with the shapes that break
        # naive brace counting: braces and brackets INSIDE strings, escapes.
        rows = [{"slug": "s{1}", "side": "BUY_UP", "gen": 1, "coin": "btc",
                 "day": "2026-08-24", "t0": 1, "t_start": 0.5,
                 "note": 'a "quoted} brace" and a \\ backslash'},
                {"slug": "s2", "side": "SELL_UP", "gen": 2, "coin": "eth",
                 "day": "2026-08-25", "t0": 2, "t_start": 1.5, "note": "]"}]
        f = Path(td) / "rows.json"
        f.write_text(json.dumps({"rows": rows}))
        got = list(stream_rows(f))
        ok(len(got) == 2, f"POSITIVE CONTROL: the streaming reader yields "
                          f"every row ({len(got)} of 2)")
        ok(got == rows,
           "POSITIVE CONTROL: and yields them byte-faithfully, including "
           "braces and brackets INSIDE strings and an escaped backslash -- "
           "the cases a naive brace counter loses")
        big = Path(td) / "big.json"
        many = [{"i": i, "pad": "x" * 300} for i in range(4000)]
        big.write_text(json.dumps({"rows": many}))
        ok(sum(1 for _ in stream_rows(big)) == 4000,
           "POSITIVE CONTROL: it spans the 1 MB read boundary (4,000 rows "
           "well past one chunk) without dropping or duplicating a row")

    # -- the quantile rule, driven against its own definition
    g = {("btc", "s", "BUY_UP", i): float(i) for i in range(100)}
    t = theta_at_budgets(g, ("btc",), (0.05, 0.10, 0.15))
    ok(t["btc"]["5%"] == 95.0 and t["btc"]["10%"] == 90.0
       and t["btc"]["15%"] == 85.0,
       f"POSITIVE CONTROL: theta_b is the k-th largest gmax "
       f"({t['btc']}) -- on 0..99, the 5% cutoff is 95.0")
    n_at_5 = sum(1 for v in g.values() if v >= t["btc"]["5%"])
    ok(n_at_5 == 5,
       f"POSITIVE CONTROL: and applying it cancels exactly int(n*b) = 5 "
       f"generations ({n_at_5}) -- the cutoff means what it says")
    ok(theta_at_budgets(g, ("btc",), (0.10,))["btc"]["10%"]
       < theta_at_budgets(g, ("btc",), (0.05,))["btc"]["5%"],
       "a LARGER budget gives a LOWER threshold -- the rule is monotone, so "
       "a constant would fail here")
    try:
        theta_at_budgets({}, ("btc",))
        ok(False, "KNOWN-BAD: an empty gmax REFUSES")
    except OperatingPointBuildRefused as e:
        ok("not a threshold" in str(e),
           "KNOWN-BAD: an empty gmax REFUSES BY NAME -- an empty quantile is "
           "not a threshold (R-141)")
    g2 = dict(g); g2[("eth", "s", "BUY_UP", 0)] = 1e9
    ok(theta_at_budgets(g2, ("btc",))["btc"] == t["btc"],
       "a generation of ANOTHER coin does not move btc's threshold -- the "
       "map is per coin because the fits are")

    # -- the digest binds the map
    d1 = theta_map_digest({"btc": {"5%": 1.0}})
    ok(len(d1) == 16, "the theta map carries a digest of its own contents")
    ok(d1 != theta_map_digest({"btc": {"5%": 1.0000001}}),
       "KNOWN-BAD: a one-part-in-ten-million change to a threshold CHANGES "
       "the digest -- so the numbers that ran can be tied to the ones declared")
    ok(d1 == theta_map_digest({"btc": {"5%": 1.0}}),
       "POSITIVE CONTROL: and the same map digests the same, so the check "
       "above is not simply always-different")

    # -- provenance the fence will recompute must be present and real
    ok(TRAIN_ROWS.exists() and MANIFEST.exists(),
       "the population and manifest this module binds to exist on disk")
    man = json.loads(MANIFEST.read_text())
    ok((man.get("hashes") or {}).get(
        "data/pm_5min/derived/harmful_exposure_rows_v3_eraB.json"),
       "the freeze's manifest binds the training rows BY SHA, so the split "
       "this module reads is the split the freeze was fitted on")
    if DECLARATION_PATH.exists():
        import be_forward_metric as _FM
        _d = json.loads(DECLARATION_PATH.read_text())
        for _c in _d["theta_frozen_by_coin"]:
            _op = op_declaration_for(_c, _d)
            _f = _FM.require_operating_point(_op)
            ok(_f["causal_declared"] and _f[_FM.OP_TOKEN_FIELD],
               f"POSITIVE CONTROL: the built declaration for {_c} PASSES the "
               f"fence -- provenance rehashed, theta digest recomputed")
            if VERIFICATION_PATH.exists():
                _fo = dict(_f, coin=_c, verification=_op["verification"])
                _fen = _FM.require_fenced_op(_fo, "10%")
                ok(_fen["token_recomputed"] is True
                   and _fen["recomputation_verified"]["bound_by"].startswith(
                       "recomputation"),
                   f"BE17-R2 POSITIVE CONTROL: the REAL declaration for {_c} "
                   f"passes the BINDING fence -- its numbers were recomputed "
                   f"from the rows artifact it names")
            else:
                ok(False, f"no verification artifact for {_c}; run --verify")
        _bad = op_declaration_for(sorted(_d["theta_frozen_by_coin"])[0], _d)
        _bad["theta_frozen"] = {k: v + 1.0 for k, v in _bad["theta_frozen"].items()}
        try:
            _FM.require_operating_point(_bad)
            ok(False, "KNOWN-BAD: a moved theta PASSED the fence")
        except Exception as e:                        # noqa: BLE001
            ok("not the numbers that were declared" in str(e),
               "KNOWN-BAD: moving a theta after the declaration REFUSES at "
               "the fence -- the digest binds the numbers to the artifact")
    else:
        ok(False, "the built operating-point declaration is absent")
        ok(False, "and so the fence cannot be driven against it")
    ok(json.loads(Path(FS.CANDIDATE).read_text())["trained_on"]["days"]
       == ["2026-08-24", "2026-08-25"],
       "and the freeze itself names those two days as its training split")

    print(f"\n{checks} checks passed" if not fails
          else f"\n{len(fails)} FAILURES of {checks} checks")
    for f in fails:
        print(f"  - {f}")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    return 1 if fails else 0


VERIFICATION_PATH = (Path(__file__).resolve().parent / "declarations"
                     / "be_operating_point_verification_v1.json")


def derive_days_from_rows(rows_path: Path = None, coins=("btc", "eth")) -> dict:
    """WHICH DAYS DOES THE ROWS ARTIFACT ACTUALLY CONTAIN? Derived, not read.

    BE17-R2 route (ii). `derived_from_split.days` was free text nobody derived
    from the rows, so a day list that does not describe the rows could be
    written. This asks the artifact. It reads only `day` and `coin`, so it
    costs a streaming pass and no feature assembly."""
    rows_path = Path(rows_path or TRAIN_ROWS)
    per: dict = {}
    n = 0
    for r in stream_rows(rows_path):
        n += 1
        c = r.get("coin")
        if c in coins:
            per.setdefault(c, {})
            d = r.get("day")
            per[c][d] = per[c].get(d, 0) + 1
    days = sorted({d for v in per.values() for d in v})
    return {"rows_path": str(rows_path), "n_rows_scanned": n,
            "days_present": days, "rows_by_coin_day":
                {c: dict(sorted(v.items())) for c, v in sorted(per.items())},
            "derived_not_declared": True}


def verify_declaration_by_recomputation(declaration: dict = None,
                                        progress=None) -> dict:
    """BE17-R2 route (i): RECOMPUTE the map from the bytes and compare.

    This is the only check that binds the NUMBERS to the ARTIFACT. It re-runs
    the derivation over the rows the declaration names, restricted to the days
    it names, and reports whether the result reproduces `theta_frozen_by_coin`
    field for field. Expensive by nature -- that is why it is a receipted act
    run once rather than something a per-cell fence does."""
    d = declaration or json.loads(DECLARATION_PATH.read_text())
    rows_path = Path(d["rows_artifact"]["path"])
    rows_sha = sha256_file(rows_path)
    days = tuple(d["derived_from_split"]["days"])
    coins = tuple(d["derived_from_split"]["coins"])
    budgets = tuple(float(b.rstrip("%")) / 100.0 for b in d["budgets"])
    derived = derive_days_from_rows(rows_path, coins)
    scored = score_training_split(days=days, coins=coins,
                                  rows_path=rows_path, progress=progress)
    got = theta_at_budgets(scored["gmax"], coins, budgets)
    want = d["theta_frozen_by_coin"]
    per_coin = {}
    for c in sorted(set(got) | set(want)):
        gw, ww = got.get(c, {}), want.get(c, {})
        per_coin[c] = {
            "recomputed": gw, "declared": ww,
            "matches": gw == ww,
            "max_abs_difference": max(
                (abs(gw[k] - ww[k]) for k in gw if k in ww), default=None),
        }
    return {
        "protocol": "BE_OPERATING_POINT_VERIFICATION_V1",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "verifies_declaration_theta_map_sha16": d["theta_map_sha16"],
        "rows_artifact_path": str(rows_path),
        "rows_artifact_sha256": rows_sha,
        "rows_sha_matches_declaration":
            rows_sha == d["rows_artifact"]["sha256"],
        "declared_days": list(days),
        "days_derived_from_the_rows": derived["days_present"],
        "declared_days_match_the_rows":
            sorted(days) == sorted(derived["days_present"]),
        "rows_by_coin_day": derived["rows_by_coin_day"],
        "per_coin": per_coin,
        "recomputed_theta_map": got,
        "all_coins_reproduce": all(v["matches"] for v in per_coin.values()),
        "counters": scored["counters"],
        "what_this_binds": ("the NUMBERS to the BYTES. Every other provenance "
                            "check in this path rehashes an artifact or "
                            "digests the map against itself; this one derives "
                            "the map from the artifact and compares."),
        "residual_limitation_stated": (
            "a forged declaration could carry a forged verification block "
            "asserting a recomputation that never happened. That is not "
            "undetectable: the assertion is FALSIFIABLE by one command "
            "(`--verify`) over known bytes, which is a different situation "
            "from a claim nobody can check."),
    }


def op_declaration_for(coin: str, declaration: dict = None,
                       declared_by: str = "USER", source: str = None) -> dict:
    """The per-coin declaration in the shape `require_operating_point` reads.

    The theta map is PER COIN because the fits are; the fence validates one
    budget map at a time. This projects the committed declaration onto one
    coin and carries the SAME provenance, so the fence rehashes the artifacts
    and recomputes the digest over exactly the numbers that will run."""
    d = declaration or json.loads(DECLARATION_PATH.read_text())
    tm = d["theta_frozen_by_coin"][coin]
    return {
        "form": d["form"],
        "theta_frozen": tm,
        "derived_from_split": d["derived_from_split"],
        "provenance": {
            "rows_artifact": {"path": d["rows_artifact"]["path"],
                              "sha256": d["rows_artifact"]["sha256"]},
            "fit_artifact": {"path": d["fit_artifact"]["path"],
                             "sha256": d["fit_artifact"]["sha256"]},
            "theta_map_sha16": theta_map_digest(tm),
        },
        "declared_by": declared_by,
        "declared_at_utc": d["as_of_utc"],
        "source": source or (
            f"be_operating_point --build over {d['derived_from_split']['days']}"
            f" ({d['counters']['rows_scored']:,} rows scored)"),
        "coin": coin,
        "verification": (json.loads(VERIFICATION_PATH.read_text())
                         if VERIFICATION_PATH.exists() else None),
    }


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--verify" in argv:
        def prog(c):
            print(f"  verify: slugs={c['slugs']} scored={c['rows_scored']}",
                  flush=True)
        v = verify_declaration_by_recomputation(progress=prog)
        VERIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        VERIFICATION_PATH.write_text(
            json.dumps(v, indent=1, sort_keys=True, default=str))
        print(json.dumps({k: val for k, val in v.items()
                          if k not in ("rows_by_coin_day",)},
                         indent=1, sort_keys=True, default=str))
        return 0 if v["all_coins_reproduce"] else 1
    if "--build" in argv:
        def prog(c):
            print(f"  slugs={c['slugs']} rows_in_split={c['rows_in_split']} "
                  f"scored={c['rows_scored']}", flush=True)
        d = build_declaration(progress=prog)
        DECLARATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        DECLARATION_PATH.write_text(
            json.dumps(d, indent=1, sort_keys=True, default=str))
        print(json.dumps({k: v for k, v in d.items()
                          if k != "counters"}, indent=1, sort_keys=True,
                         default=str))
        print("counters:", json.dumps(d["counters"]))
        return 0
    print("usage: be_operating_point.py --selftest | --build | --verify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
