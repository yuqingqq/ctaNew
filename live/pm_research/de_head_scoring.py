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

The decision score is then the fit's frozen expected cancel value, not that
probability alone:

    score = p * dot(value_weights, x)

The LGBM arm applies the same contract with its manifest-bound value booster.
`phase2_arms.freeze_thresholds` fitted both arms' cutoffs on this product;
comparing those cutoffs with `p` is a unit error and is refused by the policy
contract below.

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

EXPECTED_CHECKS = 37

#: The thresholds persisted by ``phase2_arms.freeze_thresholds`` are
#: quantiles of ``p_fill * conditional_value``.  This is therefore the only
#: score contract that may be compared with them by the replay policy.
EXPECTED_VALUE_SCORE_KIND = "P_FILL_X_CONDITIONAL_VALUE_FIT_TARGET_UNITS"

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
    if (len(d["hazard_weights"]) != n + 1
            or not isinstance(d["value_weights"], list)
            or len(d["value_weights"]) != n + 1
            or len(d["norm_sd"]) != n):
        # SITE: incumbent#2
        raise HeadRefused(
            f"{len(d['hazard_weights'])} hazard and "
            f"{len(d['value_weights']) if isinstance(d['value_weights'], list) else 'no'} "
            f"value weights against {n} normalisers: the intercept "
            f"convention is `[1.0] + zscaled`, so both heads must be one "
            f"longer than the means")
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


def score_incumbent_condvalue(model: dict, raw: list[float]) -> float:
    """Frozen incumbent ``p_fill * conditional_value`` decision score."""
    p_fill = score_incumbent(model, raw)
    mu, sd = model["norm_mu"], model["norm_sd"]
    x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
    conditional_value = float(sum(
        weight * value for weight, value in zip(model["value_weights"], x)))
    return p_fill * conditional_value


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
    # DE35-R3: the library's error is converted HERE, at the boundary that
    # owns the call, rather than named in a CLI except-tuple far away --
    # so any caller of this function gets the module's own refusal.
    try:
        return float(booster.predict([list(raw)])[0])
    except Exception as exc:                       # pragma: no cover
        # SITE: lgbm#3
        raise HeadRefused(
            f"the LGBM head refused the row it was handed ({exc}); the "
            f"width matched ({width}) so this is the model's own "
            f"objection, converted at its boundary") from exc


def load_lgbm_condvalue(coin: str):
    """Verified hazard and value boosters for the frozen LGBM policy."""
    SS.verify_head("q1_arrival_composed_lgbm", coin)
    record = json.loads((FITS / "val_models.json").read_text())
    if record.get(coin) is not True:
        # SITE: lgbmvalue#1
        raise HeadRefused(
            f"the fit does not record a value model for {coin}; the frozen "
            f"threshold is an expected-value threshold and must not be "
            f"compared with hazard probability alone")
    hazard, width = load_lgbm(coin)
    try:
        import lightgbm as lgb
        value = lgb.Booster(model_file=str(FITS / f"lgbm_val_{coin}.txt"))
    except Exception as exc:                       # pragma: no cover
        # SITE: lgbmvalue#2
        raise HeadRefused(
            f"the recorded LGBM value head for {coin} cannot be loaded "
            f"({exc})") from exc
    value_width = int(value.num_feature())
    if value_width != width:
        # SITE: lgbmvalue#3
        raise HeadRefused(
            f"the LGBM hazard head has {width} features but its value head "
            f"has {value_width}; their product is not a fitted score")
    return hazard, value, width


def score_lgbm_condvalue(hazard, value, width: int,
                         raw: list[float]) -> float:
    """Frozen LGBM ``p_fill * conditional_value`` decision score."""
    p_fill = score_lgbm(hazard, width, raw)
    if len(raw) != width:
        raise HeadRefused(
            f"row has {len(raw)} features but the LGBM value head was fitted "
            f"on {width}")
    try:
        conditional_value = float(value.predict([list(raw)])[0])
    except Exception as exc:                       # pragma: no cover
        # SITE: lgbmvalue#4
        raise HeadRefused(
            f"the LGBM value head refused the row it was handed ({exc})") \
            from exc
    return p_fill * conditional_value


def score_contract(head: str) -> dict:
    """Machine-readable score/threshold identity for result receipts."""
    if head not in SS.HEADS:
        raise HeadRefused(f"unknown head {head!r}")
    return {
        "score_kind": EXPECTED_VALUE_SCORE_KIND,
        "formula": "p_fill * predicted_conditional_value",
        "threshold_kind": EXPECTED_VALUE_SCORE_KIND,
        "comparison": "cancel iff generation_max_score >= theta",
    }


def load_lgbm_normalisers(coin: str) -> dict:
    """The z-scale the LGBM head was FITTED THROUGH, read off the fit.

    `phase2_arms:1481-1482` builds `XF = PM + FN + ST` and then
    `Xf, mu, sd = fc.fast_zscale(XF, XF)`; `:1550-1552` fits the booster on
    `Xf` itself. `fast_zscale` PREPENDS a 1.0 intercept column
    (`harmful_fast_compute:285-287`), so the booster's columns are
    `[1.0] + (raw - mu) / sd`, and `mu`/`sd` are what arm B persisted to
    `linear_{coin}.json` -- 105 long against 106 columns."""
    f = FITS / f"linear_{coin}.json"
    d = json.loads(f.read_text())
    for k in ("norm_mu", "norm_sd"):
        if k not in d:
            # SITE: lgbmnorm#1
            raise HeadRefused(
                f"{f.name} carries no {k!r}: the LGBM head's input transform "
                f"is arm B's z-scale, and without it the only vectors that "
                f"could be handed to the booster are unscaled ones -- which "
                f"are the RIGHT WIDTH and so refuse nothing")
    if len(d["norm_mu"]) != len(d["norm_sd"]):
        # SITE: lgbmnorm#2
        raise HeadRefused(
            f"{f.name}: {len(d['norm_mu'])} means against "
            f"{len(d['norm_sd'])} sds")
    return {"norm_mu": d["norm_mu"], "norm_sd": d["norm_sd"],
            "n_raw": len(d["norm_mu"]), "source": f.name}


def compose_head_inputs(pm, fn, st, *, norms, incumbent_width, lgbm_width):
    """One feature triple -> the vector EACH head was fitted on.

    THE TWO HEADS DO NOT TAKE THE SAME VECTOR, and neither takes the raw
    concatenation:

    | head | fit site | what it takes |
    |---|---|---|
    | `incumbent_linear_d` | `:1512`, applied `:1928` | raw `PM + FN` (60); `score_incumbent` prepends the intercept and z-scales |
    | `q1_arrival_composed_lgbm` | `:1481`, fitted `:1552` | `[1.0] + (PM+FN+ST - mu)/sd` (106) |

    THE SILENT ERROR THIS FUNCTION EXISTS TO PREVENT: `[1.0] + raw` is 106
    columns too. It would satisfy `score_lgbm`'s width check, reach the
    booster, and return a probability that is not a prediction -- the
    round-33 failure with the traceback removed. So the scale is applied
    HERE, from the fit's own normalisers, and every width is asserted
    against an artifact rather than a literal."""
    raw_d = list(pm) + list(fn)
    raw_b = raw_d + list(st)
    mu, sd = norms["norm_mu"], norms["norm_sd"]
    if len(raw_d) != incumbent_width:
        # SITE: compose#1
        raise HeadRefused(
            f"PM+fine is {len(raw_d)} wide and the incumbent was fitted on "
            f"{incumbent_width} (PM {len(pm)} + fine {len(fn)})")
    if len(raw_b) != len(mu):
        # SITE: compose#2
        raise HeadRefused(
            f"PM+fine+state is {len(raw_b)} wide and {norms['source']} "
            f"normalises {len(mu)} (PM {len(pm)} + fine {len(fn)} + state "
            f"{len(st)})")
    if len(mu) + 1 != lgbm_width:
        # SITE: compose#3
        raise HeadRefused(
            f"{norms['source']} normalises {len(mu)} features and the "
            f"booster was fitted on {lgbm_width} columns. The convention is "
            f"`[1.0] + zscaled`, so these must differ by exactly the "
            f"intercept -- if they do not, the two artifacts are from "
            f"different fits and nothing composed here is that model's input")
    return {"incumbent_linear_d": raw_d,
            "q1_arrival_composed_lgbm":
                [1.0] + [(raw_b[i] - mu[i]) / sd[i] for i in range(len(mu))]}


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
        out = {k: float(v) for k, v in got.items()}
    elif head == "q1_arrival_composed_lgbm":
        if verify:
            SS.verify_head("q1_arrival_composed_lgbm", coin)
        out = {k: float(v) for k, v in json.loads(
            (d0 / f"lgbm_thresholds_{coin}.json").read_text()).items()}
    else:
        # SITE: thresholds#2
        raise HeadRefused(f"unknown head {head!r}")
    # These are expected-value thresholds, not probabilities.  Negative and
    # >1 values are valid in the fit target's units; only non-finite values
    # have no ordered comparison semantics.
    bad = sorted(k for k, v in out.items() if not math.isfinite(v))
    if bad:
        # SITE: thresholds#3
        raise HeadRefused(
            f"{head}/{coin} carries non-finite expected-value threshold(s) "
            f"for {bad}")
    return out


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
    _raw0 = [0.0] * 60
    _x0 = [1.0] + [(_raw0[i] - inc["norm_mu"][i])
                    / inc["norm_sd"][i] for i in range(60)]
    _v0 = float(sum(a * b for a, b in zip(inc["value_weights"], _x0)))
    _ev0 = score_incumbent_condvalue(inc, _raw0)
    ok(abs(_ev0 - p * _v0) < 1e-12 and _ev0 != p,
       f"THE INCUMBENT POLICY SCORE IS EXPECTED VALUE: p_fill {p:.6f} x "
       f"conditional value {_v0:.6f} = {_ev0:.6f}, not the probability "
       f"alone that the replay previously compared with this fit's "
       f"expected-value threshold")
    for _bad in (59, 61, 1):
        refuses(lambda w=_bad: score_incumbent(inc, [0.0] * w),
                f"KNOWN-BAD: a {_bad}-feature row REFUSES against a "
                f"60-feature head -- weights on a differently-shaped vector "
                f"yield a number, not a prediction",
                needle="not a prediction")
    with _tf.TemporaryDirectory() as d:
        _bad = dict(inc)
        _bad["value_weights"] = _bad["value_weights"][:-1]
        _bad.pop("_n_features", None)
        (Path(d) / "linear_d_btc.json").write_text(json.dumps(_bad))
        _sv = globals()["FITS"]
        globals()["FITS"] = Path(d)
        try:
            refuses(lambda: load_incumbent("btc"),
                    "KNOWN-BAD: an incumbent whose VALUE head has the wrong "
                    "width REFUSES at load -- hazard-only scoring is not a "
                    "fallback for an expected-value policy",
                    needle="value weights")
        finally:
            globals()["FITS"] = _sv

    # ---- the HEAD UNDER TEST, on the real fit ---------------------------
    booster, width = load_lgbm("btc")
    ok(width == 106,
       f"THE HEAD UNDER TEST LOADS and reports the width it was fitted on: "
       f"{width} features (round 33 passed 1)")
    q = score_lgbm(booster, width, [0.0] * width)
    ok(0.0 <= q <= 1.0,
       f"and it SCORES a real {width}-feature row: p {q:.6f}")
    _q2 = score_lgbm(booster, width, [0.5] * width)
    _q3 = score_lgbm(booster, width, [0.0] * width)
    # DE34-R3: this was `ok(True, ...)` -- a check that could not fail. The
    # property that CAN is determinism and range: the same row twice gives
    # the same score, and both scores are probabilities. Two synthetic rows
    # may legitimately land on one leaf, so difference is reported, not
    # asserted.
    ok(_q3 == q and 0.0 <= _q2 <= 1.0,
       f"and the head is DETERMINISTIC and in range: the same row scores "
       f"{q:.6f} twice and a second row {_q2:.6f} -- difference is "
       f"reported rather than asserted, because a tree ensemble may map "
       f"two synthetic rows to one leaf")
    _hb, _vb, _vw = load_lgbm_condvalue("btc")
    ok(_vw == width and int(_vb.num_feature()) == width,
       f"THE LGBM VALUE HEAD LOADS FROM THE MANIFEST-BOUND FIT and matches "
       f"the hazard width: {width} columns each")
    _lv = float(_vb.predict([[0.0] * width])[0])
    _lev = score_lgbm_condvalue(_hb, _vb, width, [0.0] * width)
    ok(abs(_lev - q * _lv) < 1e-12 and _lev != q,
       f"THE LGBM POLICY SCORE IS EXPECTED VALUE: p_fill {q:.6f} x "
       f"conditional value {_lv:.6f} = {_lev:.6f}, matching the statistic "
       f"whose training maxima produced the frozen threshold")
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
        _hi = json.loads((FITS / "linear_d_btc.json").read_text())
        _hi["causal_thresholds"] = dict(_hi["causal_thresholds"], **{"10%": 1.5})
        (Path(d) / "linear_d_btc_hi.json").write_text(json.dumps(_hi))
        (Path(d) / "linear_d_btc.json").write_text(json.dumps(_hi))
        ok(thresholds("btc", "incumbent_linear_d", fits=Path(d),
                      verify=False)["10%"] == 1.5,
           "EXPECTED-VALUE CONTRACT: a finite threshold above one is valid "
           "in the fit target's units; treating it as a probability was the "
           "same unit error as scoring on p_fill alone")
        _hi["causal_thresholds"]["10%"] = float("nan")
        (Path(d) / "linear_d_btc.json").write_text(json.dumps(_hi))
        refuses(lambda: thresholds("btc", "incumbent_linear_d",
                                   fits=Path(d), verify=False),
                "KNOWN-BAD: a non-finite expected-value threshold REFUSES",
                needle="non-finite expected-value")
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
    ok(all(score_contract(h)["score_kind"]
           == score_contract(h)["threshold_kind"]
           == EXPECTED_VALUE_SCORE_KIND for h in SS.HEADS),
       "BOTH replay arms declare the same expected-value identity for score "
       "and threshold; a probability/expected-value comparison cannot pass "
       "this contract")

    # ---- the composition: what each head is actually fed (DE36 item 4) ----
    _nb = load_lgbm_normalisers("btc")
    _inc = load_incumbent("btc")
    _b, _wl = load_lgbm("btc")
    _iw, _nr = _inc["_n_features"], _nb["n_raw"]
    ok(_iw == 60 and _nr == 105 and _wl == 106 and _nr + 1 == _wl,
       f"MEASURED off the three artifacts, never assumed: the incumbent "
       f"normalises {_iw} (PM+fine), {_nb['source']} normalises {_nr} "
       f"(PM+fine+state) and the booster carries {_wl} columns -- so the "
       f"state block is {_nr - _iw} wide and the booster's extra column is "
       f"the intercept `fast_zscale` prepends, not a feature")
    _pm = [0.5 + 0.01 * i for i in range(31)]
    _fn = [1.0 - 0.02 * i for i in range(_iw - 31)]
    _st = [0.25 * (i % 7) for i in range(_nr - _iw)]
    _cmp = compose_head_inputs(_pm, _fn, _st, norms=_nb,
                               incumbent_width=_iw, lgbm_width=_wl)
    _rawb = _pm + _fn + _st
    _mu, _sd = _nb["norm_mu"], _nb["norm_sd"]
    ok(len(_cmp["incumbent_linear_d"]) == _iw
       and _cmp["incumbent_linear_d"] == _pm + _fn
       and len(_cmp["q1_arrival_composed_lgbm"]) == _wl
       and _cmp["q1_arrival_composed_lgbm"][0] == 1.0
       and all(abs(_cmp["q1_arrival_composed_lgbm"][k + 1]
                   - (_rawb[k] - _mu[k]) / _sd[k]) < 1e-12
               for k in range(_nr)),
       f"and one feature triple composes to BOTH vectors: {_iw} raw for the "
       f"incumbent (which z-scales inside `score_incumbent`) and {_wl} for "
       f"the booster, intercept first, every remaining column equal to "
       f"`(raw - mu)/sd` from {_nb['source']} to 1e-12")
    _unscaled = [1.0] + _rawb
    _ndiff = sum(1 for k in range(_nr)
                 if abs(_unscaled[k + 1]
                        - _cmp["q1_arrival_composed_lgbm"][k + 1]) > 1e-9)
    _same = [k for k in range(_nr)
             if abs(_unscaled[k + 1]
                    - _cmp["q1_arrival_composed_lgbm"][k + 1]) <= 1e-9]
    ok(len(_unscaled) == _wl and _ndiff > 0,
       f"KNOWN-BAD, THE SILENT ONE: `[1.0] + raw` is ALSO {_wl} columns, so "
       f"it passes `score_lgbm`'s width check, reaches the booster and "
       f"returns a probability -- it differs from the composed vector in "
       f"{_ndiff}/{_nr} columns ({len(_same)} agree, which is what a "
       f"column with mu 0 and sd 1 looks like -- a guard flag the fit "
       f"z-scaled to itself). No width guard can catch this shape; only "
       f"applying the fit's own normalisers can, which is why the scale is "
       f"applied HERE and not left to the caller")
    _p_ok = score_lgbm(_b, _wl, _cmp["q1_arrival_composed_lgbm"])
    _p_bad = score_lgbm(_b, _wl, _unscaled)
    ok(0.0 <= _p_ok <= 1.0 and 0.0 <= _p_bad <= 1.0 and _p_ok != _p_bad,
       f"DRIVEN through the REAL booster: the composed vector scores "
       f"{_p_ok:.6f} and the unscaled one {_p_bad:.6f}. Both are "
       f"probabilities and only one is a prediction -- the demonstration "
       f"that this composition is not cosmetic (SHAPE probe: the input is a "
       f"synthetic triple, so neither number means anything about a market)")
    refuses(lambda: compose_head_inputs(_pm[:-1], _fn, _st, norms=_nb,
                                        incumbent_width=_iw, lgbm_width=_wl),
            "KNOWN-BAD: a PM block one short REFUSES at the incumbent width",
            needle="the incumbent was fitted on")
    refuses(lambda: compose_head_inputs(_pm, _fn, _st[:-1], norms=_nb,
                                        incumbent_width=_iw, lgbm_width=_wl),
            "KNOWN-BAD: a STATE block one short REFUSES against the fit's "
            "normaliser count -- the width the booster would still accept "
            "if the intercept were prepended blindly",
            needle="normalises")
    refuses(lambda: compose_head_inputs(_pm, _fn, _st, norms=_nb,
                                        incumbent_width=_iw,
                                        lgbm_width=_wl + 1),
            "KNOWN-BAD: normalisers and a booster from DIFFERENT fits REFUSE "
            "-- 105 normalisers against 107 columns is not the intercept "
            "convention, and composing anyway would hand the booster a "
            "vector no fit ever produced",
            needle="differ by exactly the intercept")
    with _tf.TemporaryDirectory() as d:
        _bad = json.loads((FITS / "linear_btc.json").read_text())
        _bad.pop("norm_mu")
        (Path(d) / "linear_btc.json").write_text(json.dumps(_bad))
        _sv = globals()["FITS"]
        globals()["FITS"] = Path(d)
        try:
            refuses(lambda: load_lgbm_normalisers("btc"),
                    "KNOWN-BAD: a fit carrying no `norm_mu` REFUSES rather "
                    "than falling back to unscaled inputs, which are the "
                    "right width and refuse nothing", needle="carries no")
        finally:
            globals()["FITS"] = _sv
    ok(load_lgbm_normalisers("btc")["n_raw"] == _nr,
       "POSITIVE CONTROL: the real fit still answers after that injection")

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
