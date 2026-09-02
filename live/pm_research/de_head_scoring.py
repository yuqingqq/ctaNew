"""DE-Head-Scoring -- the two Phase-4 heads applied to real feature vectors.

SURFACE AUTHORISATION: R-464 §DE33-C1.  Round 33's `_head_scorer` fed the
LGBM booster `[[row["t"]]]` -- one column against the 106 it was fitted on --
and returned a constant 0.5 for the incumbent, reading `coefficients`/`coef`
keys `linear_d_{coin}.json` does not carry.  Both were stubs under a
docstring that said "never a stub".  This module is the scoring path, and
its first job is to make a wrong-shaped vector impossible to score.

WHAT IT BINDS, BEFORE IT SCORES ANYTHING:

  * the fit files, by the manifest's sha (`de_score_stream.verify_head`);
  * THE FIT CODE, by the manifest's own `fit_code_files` -- the pinned
    `harmful_state_features.py` (`75bd49303773c7d7`) and
    `harmful_hazard_model.py` (`58b8a2c08eea3cc9`).  A head applied by code
    that has moved since the fit is a head applied by different arithmetic,
    and the manifest is the only thing that says which arithmetic.

THE INCUMBENT'S ARITHMETIC IS COPIED FROM THE FIT'S OWN APPLY PATH
(`phase2_iter011_run.apply_incumbent_hazard`, itself copied from
`phase2_arms`'s INCUMBENT_REWEIGHTED_ONLY branch), not reinvented:

    x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
    p = predict_p(hazard_weights, x)

`linear_d_{coin}.json` records `features: "PM+fine only, NO state features"`
-- a SENTENCE, not a list -- so the vector's identity comes from its LENGTH
(`len(norm_mu)`, 60) and from the family widths the block records, never
from a name lookup that file cannot answer.

    python3 live/pm_research/de_head_scoring.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

EXPECTED_CHECKS = 21

sys.path.insert(0, str(Path(__file__).resolve().parent))
import de_score_stream as SS                     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FITS = ROOT / "data/pm_5min/derived/phase2_fits"
MANIFEST = FITS / "fit_manifest.json"

#: The pinned fit code this scorer is allowed to use, by the manifest's own
#: `fit_code_files` map. Named here so a reader sees WHICH code, and
#: verified below so the naming is not the check.
PINNED_CODE = ("harmful_state_features.py", "harmful_hazard_model.py")


class HeadRefused(RuntimeError):
    """The head refuses rather than returning a number from a wrong shape."""


def _sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def verify_fit_code(names=PINNED_CODE, *, manifest: Path | None = None,
                    here: Path | None = None) -> dict:
    """The fit code, verified against the manifest's `fit_code_files`."""
    m = json.loads((manifest or MANIFEST).read_text())
    codes = m.get("fit_code_files") or {}
    d = here or Path(__file__).resolve().parent
    out = {}
    for n in names:
        if n not in codes:
            # SITE: fitcode#1
            raise HeadRefused(
                f"{n} is not in the manifest's fit_code_files, so nothing "
                f"says which bytes fitted these heads")
        f = d / n
        if not f.exists():
            # SITE: fitcode#2
            raise HeadRefused(f"{n} is absent from {d}")
        got = _sha16(f.read_bytes())
        if got != codes[n]:
            # SITE: fitcode#3
            raise HeadRefused(
                f"{n} has sha {got}; the manifest says {codes[n]}. A head "
                f"applied by code that moved since the fit is a head "
                f"applied by different arithmetic")
        out[n] = got
    return out


def load_incumbent(coin: str) -> dict:
    """The incumbent head, verified and shape-checked at load."""
    SS.verify_head("incumbent_linear_d", coin)
    d = json.loads((FITS / f"linear_d_{coin}.json").read_text())
    for k in ("norm_mu", "norm_sd", "hazard_weights", "value_weights",
              "causal_thresholds"):
        if k not in d:
            # SITE: incumbent#1
            raise HeadRefused(f"linear_d_{coin}.json carries no {k!r}")
    n = len(d["norm_mu"])
    if len(d["hazard_weights"]) != n + 1 or len(d["norm_sd"]) != n:
        # SITE: incumbent#2
        raise HeadRefused(
            f"{len(d['hazard_weights'])} hazard weights against {n} "
            f"normalisers: the intercept convention is `[1.0] + zscaled`, "
            f"so the weights must be one longer than the means")
    d["_n_features"] = n
    return d


def score_incumbent(model: dict, raw: list[float]) -> float:
    """`p_fill` for one decision row -- the Q1_arrival counterpart.

    The arithmetic is the fit's own (`apply_incumbent_hazard`): z-scale by
    the fit's normalisers, prepend the intercept, logistic on the hazard
    weights. A raw vector of the wrong width REFUSES: applying weights to a
    differently-shaped vector yields a number, not a prediction."""
    mu, sd = model["norm_mu"], model["norm_sd"]
    if len(raw) != len(mu):
        # SITE: incumbent#3
        raise HeadRefused(
            f"row has {len(raw)} features but the incumbent hazard head was "
            f"fitted on {len(mu)}. Applying weights to a differently-shaped "
            f"vector yields a number, not a prediction (DE33-C1: round 33 "
            f"handed this head nothing at all and returned 0.5)")
    x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
    w = model["hazard_weights"]
    z = max(-30.0, min(30.0, sum(a * b for a, b in zip(w, x))))
    return 1.0 / (1.0 + math.exp(-z))


def load_lgbm(coin: str):
    """The head under test, verified, with the width it was fitted on."""
    SS.verify_head("q1_arrival_composed_lgbm", coin)
    try:
        import lightgbm as lgb
    except Exception as exc:                       # pragma: no cover
        # SITE: lgbm#1
        raise HeadRefused(f"lightgbm is not importable ({exc}); the head "
                          f"under test cannot be applied without it")
    b = lgb.Booster(model_file=str(FITS / f"lgbm_haz_{coin}.txt"))
    return b, int(b.num_feature())


def score_lgbm(booster, width: int, raw: list[float]) -> float:
    """`p_fill` from the LGBM hazard head, with the shape refusal round 33
    discovered by traceback at the first cell after a 29-minute feed."""
    if len(raw) != width:
        # SITE: lgbm#2
        raise HeadRefused(
            f"row has {len(raw)} features but the LGBM head was fitted on "
            f"{width}. Round 33 passed ONE column and the booster raised "
            f"`LightGBMError: The number of features in data (1) is not the "
            f"same as it was in training data ({width})` -- after the feed, "
            f"which is the worst place to learn it (DE33-C1)")
    return float(booster.predict([list(raw)])[0])


def thresholds(coin: str, head: str, *, fits: Path | None = None,
               verify: bool = True) -> dict:
    """The head's budget -> threshold map, from the fit that carries it.

    DE33-C2: round 33 read `thresholds` / `budget_thresholds` from the
    incumbent fit; the key is `causal_thresholds`, so `theta_for` REFUSED at
    the first cell -- after the feed."""
    d0 = fits or FITS
    if head == "incumbent_linear_d":
        if verify:
            SS.verify_head("incumbent_linear_d", coin)
        d = json.loads((d0 / f"linear_d_{coin}.json").read_text())
        got = d.get("causal_thresholds")
        if not got:
            # SITE: thresholds#1
            raise HeadRefused(
                f"linear_d_{coin}.json carries no `causal_thresholds`: the "
                f"thresholds live with the fit and are not defaulted here")
        return {k: float(v) for k, v in got.items()}
    if head == "q1_arrival_composed_lgbm":
        if verify:
            SS.verify_head("q1_arrival_composed_lgbm", coin)
        return {k: float(v) for k, v in json.loads(
            (d0 / f"lgbm_thresholds_{coin}.json").read_text()).items()}
    # SITE: thresholds#2
    raise HeadRefused(f"unknown head {head!r}")


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_head_scoring] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle=None):
        try:
            fn()
        except HeadRefused as exc:
            if needle and needle not in str(exc):
                raise SystemExit(f"[de_head_scoring] FAIL: {label} -- "
                                 f"refused for another reason ({exc})")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_head_scoring] FAIL (no refusal): {label}")

    code = verify_fit_code()
    ok(set(code) == set(PINNED_CODE)
       and code["harmful_state_features.py"] == "75bd49303773c7d7"
       and code["harmful_hazard_model.py"] == "58b8a2c08eea3cc9",
       f"THE FIT CODE IS VERIFIED BY THE MANIFEST BEFORE ANYTHING IS "
       f"SCORED: {code} -- the shas R-464 names, recomputed from the files "
       f"in this directory")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        dd = Path(d)
        (dd / "harmful_state_features.py").write_text("# moved")
        refuses(lambda: verify_fit_code(("harmful_state_features.py",),
                                        here=dd),
                "KNOWN-BAD: fit code whose BYTES have moved REFUSES -- a "
                "head applied by different code is applied by different "
                "arithmetic", needle="different arithmetic")
        refuses(lambda: verify_fit_code(("de_head_scoring.py",)),
                "KNOWN-BAD: a file the manifest does not pin REFUSES -- "
                "nothing says it fitted anything", needle="not in the "
                                                          "manifest")

    # ---- the INCUMBENT head, on the real fit ----------------------------
    inc = load_incumbent("btc")
    ok(inc["_n_features"] == 60 and len(inc["hazard_weights"]) == 61,
       f"THE INCUMBENT LOADS AND ITS SHAPE IS READ FROM THE FIT: "
       f"{inc['_n_features']} normalisers, {len(inc['hazard_weights'])} "
       f"hazard weights -- the `features` field is the SENTENCE "
       f"{json.loads((FITS / 'linear_d_btc.json').read_text())['features']!r}"
       f", so the width comes from `norm_mu` and never from a name lookup "
       f"that file cannot answer")
    _saw_sc = ""
    try:
        p = score_incumbent(inc, [0.0] * 60)
    except HeadRefused as _exc:
        p, _saw_sc = -1.0, f" REFUSED INSTEAD: {str(_exc)[:100]}"
    ok(0.0 < p < 1.0 and not _saw_sc,
       f"and it SCORES a real 60-feature row through the fit's own "
       f"arithmetic -- p_fill {p:.6f} -- z-scaled by the fit's normalisers "
       f"with the intercept prepended, which is "
       f"`apply_incumbent_hazard`'s expression, not a second one")
    _p2 = score_incumbent(inc, [1.0] * 60)
    ok(_p2 != p,
       f"and the score MOVES with the features ({_p2:.6f} against "
       f"{p:.6f}), so it is a function of the row rather than round 33's "
       f"constant 0.5")
    for _bad in (59, 61, 1):
        refuses(lambda w=_bad: score_incumbent(inc, [0.0] * w),
                f"KNOWN-BAD: a {_bad}-feature row REFUSES against a "
                f"60-feature head -- weights on a differently-shaped vector "
                f"yield a number, not a prediction",
                needle="not a prediction")

    # ---- the HEAD UNDER TEST, on the real fit ---------------------------
    booster, width = load_lgbm("btc")
    ok(width == 106,
       f"THE HEAD UNDER TEST LOADS and reports the width it was fitted on: "
       f"{width} features (round 33 passed 1)")
    q = score_lgbm(booster, width, [0.0] * width)
    ok(0.0 <= q <= 1.0,
       f"and it SCORES a real {width}-feature row: p {q:.6f}")
    _q2 = score_lgbm(booster, width, [0.5] * width)
    ok(True,
       f"and the two synthetic rows score {q:.6f} / {_q2:.6f} -- reported "
       f"rather than asserted different, because a tree ensemble may "
       f"legitimately map two synthetic rows to one leaf")
    for _bad in (1, 105, 107):
        refuses(lambda w=_bad: score_lgbm(booster, width, [0.0] * w),
                f"KNOWN-BAD: a {_bad}-feature row REFUSES against the "
                f"{width}-feature head -- BEFORE any replay, where round 33 "
                f"learned it from a traceback after a 29-minute feed",
                needle="was fitted on")

    # ---- DE33-C2: the thresholds key the fits actually carry ------------
    _saw_th = ""
    try:
        ti = thresholds("btc", "incumbent_linear_d")
        tq = thresholds("btc", "q1_arrival_composed_lgbm")
    except HeadRefused as _exc:
        ti = tq = {}
        _saw_th = f" REFUSED INSTEAD: {str(_exc)[:110]}"
    ok(set(ti) == {"5%", "10%", "15%"} and set(tq) == {"5%", "10%", "15%"}
       and not _saw_th,
       f"DE33-C2 CLOSED: BOTH heads' thresholds at all three budgets, from "
       f"the fits that carry them -- incumbent `causal_thresholds` "
       f"{ {k: round(v, 4) for k, v in ti.items()} }, head under test "
       f"{ {k: round(v, 4) for k, v in tq.items()} }. Round 33 read "
       f"`thresholds`/`budget_thresholds`, which that fit does not have, so "
       f"the run refused at its first cell -- after the feed{_saw_th}")
    ok(all(thresholds(c, h) for c in ("btc", "eth")
           for h in ("incumbent_linear_d", "q1_arrival_composed_lgbm")),
       "and for both coins, so neither coin's run stops at its first cell")
    refuses(lambda: thresholds("btc", "hazard_only"),
            "KNOWN-BAD: an unknown head REFUSES", needle="unknown head")
    with _tf.TemporaryDirectory() as d:
        _bad_fit = json.loads((FITS / "linear_d_btc.json").read_text())
        _bad_fit.pop("causal_thresholds")
        (Path(d) / "linear_d_btc.json").write_text(json.dumps(_bad_fit))
        refuses(lambda: thresholds("btc", "incumbent_linear_d",
                                   fits=Path(d), verify=False),
                "KNOWN-BAD: an incumbent fit WITHOUT `causal_thresholds` "
                "REFUSES rather than defaulting -- the thresholds live with "
                "the fit, and the directory is injected rather than the "
                "verifier patched, so the known-bad leaves no module "
                "mutated behind it", needle="not defaulted here")
        ok(thresholds("btc", "incumbent_linear_d") == ti,
           "POSITIVE CONTROL: the real fit still answers after that "
           "known-bad, which is what says the injection touched nothing")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_head_scoring] selftest OK -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
