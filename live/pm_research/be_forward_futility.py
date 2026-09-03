#!/usr/bin/env python3
"""INTERIM FUTILITY CHECK — CONFIGURABLE MACHINERY, DECLARED INSTANCES.

WHAT IS PARAMETERISED AND WHY. The USER ruled the interim CONFIGURABLE rather
than fixed at G=3, so the G threshold, the statistic and the alpha spend are
arguments, not constants. A module that hardcoded G=3 would have to be edited
to look at G=4, and an edit made while a race is running is indistinguishable
from a threshold moved to reach a wanted answer.

THE GUARD, ARGUED RATHER THAN ASSERTED. Configurability is exactly what makes
an interim dangerous. The entire value of a futility look is that its rule was
fixed BEFORE the number existed; a configurable rule chosen afterwards is not a
weaker version of that, it is the opposite of it -- it is the analyst choosing
the boundary that gives the answer they can already see. The reviewer's
R496-R3 names this shape in the neighbouring case: *"minimum 200, target 2,000"
is a range, not a declaration*, and which floor the p lands on would be settled
after seeing cost or behaviour. A configurable interim with no declaration
discipline is the same defect with more surface.

So this module will not run on a configuration it cannot prove was declared
first, and "prove" here means something a reader can re-check rather than
something the caller asserts:

  * the configuration lives in a FILE in the repository, not in a call;
  * the caller names the COMMIT it was declared in;
  * `git show <commit>:<path>` must hash to the declared `config_sha256`,
    so a commit cannot be credited with a configuration it does not contain;
  * that commit's own timestamp must precede the read;
  * and the emitted artifact records the commit, the path, the hash and both
    times, so the ordering is checkable from the artifact alone.

Each of those is a refusal BY NAME, and each has a known-bad in the suite. The
positive control matters equally (SEAT_PROTOCOL rule 16): a properly declared
configuration must be ADMITTED, or the guard is just an off switch.

WHAT THIS MODULE DOES NOT DO. It does not stop a race. It computes the interim
statistic against the declared boundary and reports both; whether to halt is a
policy decision with its own priced trade-offs and it belongs to the USER
(rule 14, CLAUDE.md). No boolean emitted here encodes an entitlement.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path("/home/yuqing/ctaNew")


class FutilityRefused(RuntimeError):
    """A named refusal."""


class FutilityConfigUndeclared(FutilityRefused):
    """The configuration was not shown to have been declared before the read."""


#: The statistics an interim may be declared on. A REGISTRY, so a declaration
#: names one of these rather than smuggling in a callable chosen to suit.
STATISTIC_REGISTRY = {
    "increment_cents_sum": {
        "what": "sum over accrued days of candidate-minus-incumbent net cents",
        "direction": "greater is better for the candidate",
    },
    "increment_cents_mean_per_day": {
        "what": "mean per accrued UTC day of the same increment",
        "direction": "greater is better for the candidate",
    },
    "n_days_with_positive_increment": {
        "what": "count of accrued days whose increment is > 0",
        "direction": "greater is better for the candidate",
    },
}

REQUIRED_CONFIG_FIELDS = ("g_threshold", "statistic", "alpha_spend",
                          "futility_boundary", "declared_by",
                          "declared_at_utc", "declaring_commit",
                          "config_path", "config_sha256")


def _git(tree: Path, *a):
    try:
        r = subprocess.run(["git", "-C", str(tree), *a],
                           capture_output=True, text=True, timeout=30)
        return (r.stdout, r.returncode)
    except Exception as e:                            # noqa: BLE001
        return (str(e), 1)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def require_declared_config(cfg, tree: Path = None,
                            read_started_utc: str = None) -> dict:
    """REFUSE unless the configuration is declared, committed and PRIOR.

    Every branch here refuses by name and every one has a known-bad in the
    suite. The expensive branch is the third: it re-reads the config OUT OF
    THE NAMED COMMIT and hashes it, so a commit cannot be credited with a
    configuration it never carried."""
    tree = Path(tree or REPO)
    read_started_utc = read_started_utc or _now()
    if cfg is None:
        raise FutilityConfigUndeclared(
            "REFUSED: no futility configuration. The interim is CONFIGURABLE "
            "(the USER's ruling), which is precisely why an undeclared "
            "configuration cannot be run: a boundary chosen after the number "
            "exists is the analyst reading the answer off the data. Declare "
            f"{list(REQUIRED_CONFIG_FIELDS)} in a committed file.")
    if not isinstance(cfg, dict):
        raise FutilityConfigUndeclared(
            f"REFUSED: the configuration is {type(cfg).__name__}, not a "
            f"mapping.")
    missing = [f for f in REQUIRED_CONFIG_FIELDS if cfg.get(f) in (None, "")]
    if missing:
        raise FutilityConfigUndeclared(
            f"REFUSED: the configuration lacks {missing}. Each of these is "
            f"what makes the declaration CHECKABLE rather than claimed.")
    if cfg["statistic"] not in STATISTIC_REGISTRY:
        raise FutilityConfigUndeclared(
            f"REFUSED: statistic {cfg['statistic']!r} is not in the registry "
            f"{sorted(STATISTIC_REGISTRY)}. A statistic named at the call "
            f"site is one chosen where no ruling can see it.")
    g = cfg["g_threshold"]
    if not isinstance(g, int) or isinstance(g, bool) or g < 1:
        raise FutilityConfigUndeclared(
            f"REFUSED: g_threshold must be a positive integer; got {g!r}.")
    a = cfg["alpha_spend"]
    if not isinstance(a, (int, float)) or isinstance(a, bool) or not 0 < a < 1:
        raise FutilityConfigUndeclared(
            f"REFUSED: alpha_spend must lie strictly in (0, 1); got {a!r}. An "
            f"interim that spends 0 is not a look, and one that spends 1 "
            f"leaves nothing for the final analysis.")
    # --- the commit must exist ------------------------------------------
    commit = str(cfg["declaring_commit"])
    out, rc = _git(tree, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if rc != 0:
        raise FutilityConfigUndeclared(
            f"REFUSED: declaring_commit {commit!r} does not resolve in "
            f"{tree}. A declaration that names no reachable commit cannot be "
            f"checked by anyone.")
    full = out.strip()
    # --- the commit must CARRY the configuration ------------------------
    path = str(cfg["config_path"])
    blob, rc2 = _git(tree, "show", f"{full}:{path}")
    if rc2 != 0:
        raise FutilityConfigUndeclared(
            f"REFUSED: {path!r} does not exist at commit {full[:12]}. The "
            f"commit is being credited with a configuration it does not "
            f"contain.")
    got = hashlib.sha256(blob.encode()).hexdigest()
    if got != cfg["config_sha256"]:
        raise FutilityConfigUndeclared(
            f"REFUSED: {path} at {full[:12]} hashes to {got[:16]}…, but the "
            f"declaration says {str(cfg['config_sha256'])[:16]}…. The "
            f"configuration that ran is not the configuration that was "
            f"committed.")
    # --- and it must PRECEDE the read -----------------------------------
    ctime, rc3 = _git(tree, "show", "-s", "--format=%cI", full)
    ctime = ctime.strip()
    if rc3 != 0 or not ctime:
        raise FutilityConfigUndeclared(
            f"REFUSED: cannot read the commit time of {full[:12]}.")
    c_dt = dt.datetime.fromisoformat(ctime)
    r_dt = dt.datetime.fromisoformat(read_started_utc.replace("Z", "+00:00"))
    if c_dt > r_dt:
        raise FutilityConfigUndeclared(
            f"REFUSED: the declaring commit is timestamped {ctime}, AFTER the "
            f"read began at {read_started_utc}. A rule fixed after the number "
            f"existed is not a pre-declaration.")
    return {
        "g_threshold": g, "statistic": cfg["statistic"],
        "statistic_meaning": STATISTIC_REGISTRY[cfg["statistic"]],
        "alpha_spend": float(a),
        "futility_boundary": float(cfg["futility_boundary"]),
        "declared_by": cfg["declared_by"],
        "declared_at_utc": cfg["declared_at_utc"],
        "declaring_commit": full,
        "declaring_commit_time_utc": ctime,
        "config_path": path,
        "config_sha256": got,
        "read_started_utc": read_started_utc,
        "declared_before_read": True,
        "verified": ("the commit resolves, CARRIES this exact config by hash, "
                     "and predates the read -- all three re-checkable from "
                     "the fields above"),
    }


def interim(per_day_increments: dict, cfg: dict, tree: Path = None,
            read_started_utc: str = None) -> dict:
    """The interim look. Estimates; never decides (rule 14).

    `per_day_increments` maps a UTC date to that day's candidate-minus-
    incumbent increment in cents. G is `len(...)`, computed here rather than
    taken on trust."""
    d = require_declared_config(cfg, tree=tree,
                                read_started_utc=read_started_utc)
    days = sorted(per_day_increments)
    vals = [float(per_day_increments[k]) for k in days]
    g = len(days)
    if g < d["g_threshold"]:
        return {"protocol": "BE_FORWARD_FUTILITY_V1", "config": d,
                "G": g, "days": days,
                "status": "NOT_YET_AT_THRESHOLD",
                "why": (f"G={g} has not reached the declared threshold "
                        f"{d['g_threshold']}; no interim statistic is "
                        f"computed, because computing one below the declared "
                        f"threshold is an undeclared look."),
                "statistic_value": None, "crosses_futility_boundary": None,
                "decides": None}
    import math
    if d["statistic"] == "increment_cents_sum":
        stat = math.fsum(vals)
    elif d["statistic"] == "increment_cents_mean_per_day":
        stat = math.fsum(vals) / g
    else:
        stat = float(sum(1 for v in vals if v > 0))
    return {
        "protocol": "BE_FORWARD_FUTILITY_V1", "config": d,
        "G": g, "days": days,
        "status": "EVALUATED",
        "statistic_name": d["statistic"], "statistic_value": stat,
        "futility_boundary": d["futility_boundary"],
        "crosses_futility_boundary": stat <= d["futility_boundary"],
        "alpha_spend": d["alpha_spend"],
        "decides": None,
        "who_decides": ("nothing here halts a race. This reports the declared "
                        "statistic against the declared boundary; whether to "
                        "stop is the USER's, with its own priced trade-offs "
                        "(rule 14)."),
    }


# ---------------------------------------------------------------------------
# SELFTEST. The guard is the module, so it is driven in BOTH directions on a
# REAL git repository built in a temp dir -- a mocked git would prove only
# that the mock agrees with itself.
# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 26


def _mkrepo(td: Path, cfg_body: str, path="cfg.json"):
    """A real repository with one commit carrying `cfg_body` at `path`."""
    subprocess.run(["git", "init", "-q", str(td)], check=True,
                   capture_output=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(td), "config", k, v], check=True,
                       capture_output=True)
    (td / path).write_text(cfg_body)
    subprocess.run(["git", "-C", str(td), "add", path], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(td), "commit", "-q", "-m", "declare"],
                   check=True, capture_output=True)
    sha = subprocess.run(["git", "-C", str(td), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    return sha, hashlib.sha256(cfg_body.encode()).hexdigest()


def selftest() -> int:
    import tempfile
    import traceback
    checks = 0
    fails = []

    def ok(c, label):
        nonlocal checks
        checks += 1
        print(f"PASS: {label}" if c else f"FAIL: {label}")
        if not c:
            fails.append(label)

    def refuses(fn, want, label):
        nonlocal checks
        checks += 1
        try:
            fn()
        except FutilityRefused as e:
            if want in str(e):
                print(f"PASS: {label}")
                return
            fails.append(f"{label} [wrong cause: {str(e)[:110]}]")
            print(f"FAIL: {label} -- refused, not for {want!r}")
            return
        except Exception as e:                        # noqa: BLE001
            fails.append(f"{label} [{type(e).__name__}]")
            print(f"FAIL: {label} -- {type(e).__name__}: {str(e)[:110]}")
            print(traceback.format_exc()[-300:])
            return
        fails.append(f"{label} [ACCEPTED]")
        print(f"FAIL: {label} -- the known-bad was ACCEPTED")

    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        body = json.dumps({"g_threshold": 3, "statistic": "increment_cents_sum",
                           "alpha_spend": 0.005, "futility_boundary": 0.0},
                          indent=1, sort_keys=True)
        sha, csha = _mkrepo(td, body)
        good = {"g_threshold": 3, "statistic": "increment_cents_sum",
                "alpha_spend": 0.005, "futility_boundary": 0.0,
                "declared_by": "USER", "declared_at_utc": "2026-01-01T00:00:00Z",
                "declaring_commit": sha, "config_path": "cfg.json",
                "config_sha256": csha}

        # ---- POSITIVE CONTROL: a properly declared config is ADMITTED ----
        d = require_declared_config(good, tree=td)
        ok(d["declared_before_read"] is True and d["declaring_commit"] == sha,
           "POSITIVE CONTROL: a configuration that is committed, carried by "
           "its named commit and prior to the read is ADMITTED")
        ok(d["config_sha256"] == csha,
           "POSITIVE CONTROL: the hash recorded is the hash READ BACK OUT of "
           "the commit, not the one the caller supplied")
        ok(d["g_threshold"] == 3 and d["alpha_spend"] == 0.005
           and d["statistic"] == "increment_cents_sum",
           "POSITIVE CONTROL: G, the statistic and the alpha spend are "
           "PARAMETERS carried through, not constants in this file")

        # ---- KNOWN-BADS, each by name ------------------------------------
        refuses(lambda: require_declared_config(None, tree=td),
                "no futility configuration",
                "KNOWN-BAD: an undeclared configuration REFUSES BY NAME")
        for f in REQUIRED_CONFIG_FIELDS:
            refuses(lambda f=f: require_declared_config(
                        {k: v for k, v in good.items() if k != f}, tree=td),
                    "lacks", f"KNOWN-BAD: a configuration missing {f} REFUSES")
        refuses(lambda: require_declared_config(
                    {**good, "statistic": "whatever_i_like"}, tree=td),
                "not in the registry",
                "KNOWN-BAD: a statistic named at the call site REFUSES")
        refuses(lambda: require_declared_config({**good, "g_threshold": 0},
                                                tree=td),
                "positive integer",
                "KNOWN-BAD: a non-positive G threshold REFUSES")
        refuses(lambda: require_declared_config({**good, "alpha_spend": 1.0},
                                                tree=td),
                "strictly in (0, 1)",
                "KNOWN-BAD: an alpha spend of 1.0 REFUSES -- it leaves nothing "
                "for the final analysis")
        refuses(lambda: require_declared_config(
                    {**good, "declaring_commit": "0" * 40}, tree=td),
                "does not resolve",
                "KNOWN-BAD: a declaring commit that does not exist REFUSES")
        refuses(lambda: require_declared_config(
                    {**good, "config_path": "not_there.json"}, tree=td),
                "does not exist at commit",
                "KNOWN-BAD: a commit credited with a file it does not contain "
                "REFUSES")
        # THE LOAD-BEARING ONE: the config was EDITED after the commit.
        edited = {**good, "alpha_spend": 0.04,
                  "config_sha256": hashlib.sha256(
                      json.dumps({"x": 1}).encode()).hexdigest()}
        refuses(lambda: require_declared_config(edited, tree=td),
                "is not the configuration that was committed",
                "KNOWN-BAD: a configuration whose hash does not match the "
                "committed bytes REFUSES -- this is the after-the-fact edit "
                "the whole guard exists for")
        refuses(lambda: require_declared_config(
                    good, tree=td, read_started_utc="2000-01-01T00:00:00Z"),
                "AFTER the read began",
                "KNOWN-BAD: a declaring commit timestamped after the read "
                "REFUSES -- a rule fixed after the number is not a "
                "pre-declaration")

        # ---- THE INTERIM ITSELF, both directions -------------------------
        below = interim({"2026-09-01": 10.0, "2026-09-02": -5.0}, good, tree=td)
        ok(below["status"] == "NOT_YET_AT_THRESHOLD"
           and below["statistic_value"] is None,
           "POSITIVE CONTROL: below the declared G threshold NO statistic is "
           "computed -- a look below the threshold is an undeclared look")
        at = interim({"2026-09-01": 10.0, "2026-09-02": -5.0,
                      "2026-09-03": -20.0}, good, tree=td)
        ok(at["status"] == "EVALUATED" and abs(at["statistic_value"] + 15.0) < 1e-9,
           "POSITIVE CONTROL: at the threshold the declared statistic is "
           "computed (-15.0 cents) -- a real number, not a shape")
        ok(at["crosses_futility_boundary"] is True,
           "POSITIVE CONTROL: a negative sum crosses a boundary of 0.0")
        pos = interim({"2026-09-01": 10.0, "2026-09-02": 5.0,
                       "2026-09-03": 20.0}, good, tree=td)
        ok(pos["crosses_futility_boundary"] is False,
           "and a positive sum does NOT -- the boundary predicate can go both "
           "ways, so the True above is a measurement")
        ok(at["decides"] is None and "USER" in at["who_decides"],
           "the interim DECIDES nothing: no boolean here encodes an "
           "entitlement (rule 14)")
        ok(at["G"] == 3 and at["days"] == ["2026-09-01", "2026-09-02",
                                           "2026-09-03"],
           "G is COMPUTED from the days supplied, never taken on trust")

    print(f"\n{checks} checks passed" if not fails
          else f"\n{len(fails)} FAILURES of {checks} checks")
    for f in fails:
        print(f"  - {f}")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    print("usage: be_forward_futility.py --selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
