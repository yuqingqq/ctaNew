"""DE-Score-Stream -- the named Phase-2 heads to `{t, slug, side, score}`.

SURFACE AUTHORISATION: R-459 §3(ii).  `harmful_stateful_policy.validate_scores`
(`:383-404`) requires one event per generation carrying `t`, `slug`, `side`
and `score`; LANE4's scorer is a sha256 STUB (`de_lane4_real_parity.py:76-84`)
that exists to prove the plumbing, not to score; and no adapter from the
named heads to that shape existed.  This is that adapter, and it is new code
on a scoring path, so it ships with its falsifiers (rule 15).

WHAT IT BINDS.  Every file it reads is verified against
`phase2_fits/fit_manifest.json` by sha256 prefix BEFORE it is opened as a
model -- the incumbent head `linear_d_{coin}.json` (btc `18701008c2bd18c6`,
R-398) and the head under test, `Q1_arrival` of `composed_lgbm`
(`lgbm_haz_{coin}.txt` with `lgbm_thresholds_{coin}.json`).  A file whose
bytes do not match the manifest is REFUSED by name; a fit belonging to the
OTHER coin is REFUSED even though it loads perfectly, because the manifest
knows which coin a file is for and the model does not.

THE DECLARED LIMIT -- IR-R4, stated rather than worked around (R-459 §3):
there is no generation-tranche artifact with a production consumer, so this
adapter does NOT read features from a file.  It takes the feature table the
caller has (the tranche table the runner builds) and turns head + table into
score events.  That is the whole of what it does: it does not invent a
population, and until the runner supplies one there is nothing here that
could silently score the wrong rows.  The limit is written into the
addendum, not closed by a fixture pretending to be data.

    python3 live/pm_research/de_score_stream.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import ast as _ast
import json
import math
import sys
from pathlib import Path

EXPECTED_CHECKS = 26

#: The event contract, in ONE place: `score_events` checks these keys and
#: builds its events from the same tuple, so the two cannot disagree
#: (DE38-R3). `gen` is here because the (gamma) control permutes the
#: above-threshold values over GENERATIONS.
REQUIRED_EVENT_KEYS = ("t", "slug", "side", "gen")

ROOT = Path(__file__).resolve().parents[2]
FITS = ROOT / "data/pm_5min/derived/phase2_fits"
MANIFEST = FITS / "fit_manifest.json"

#: The two heads the diagnostic names, and the files each is made of.
HEADS = {
    "incumbent_linear_d": ("linear_d_{coin}.json",),
    "q1_arrival_composed_lgbm": ("lgbm_haz_{coin}.txt",
                                 "lgbm_thresholds_{coin}.json"),
}
COINS = ("btc", "eth")

#: THE SIDE VOCABULARY IS THE CONSUMER'S, IMPORTED AND NOT RESTATED.
#: `validate_scores` refuses a side outside `harmful_stateful_policy.SIDES`,
#: and that tuple is `("BUY_UP", "SELL_UP")` -- not the `("BUY", "SELL")` a
#: reader of this file would assume. The first version of this adapter
#: restated it, built a stream the policy refused, and the check that
#: caught it was the one that runs the CONSUMER over the output instead of
#: asserting the shape from its docstring (rule 16).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harmful_stateful_policy import SIDES        # noqa: E402


class ScoreStreamRefused(RuntimeError):
    """The adapter refuses rather than scoring something it cannot bind."""


def _sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def manifest_hashes(path: Path | None = None) -> dict:
    p = path or MANIFEST
    if not p.exists():
        # SITE: manifest#1
        raise ScoreStreamRefused(
            f"no fit manifest at {p} -- the heads are bound by their shas "
            f"and an unbound fit is not loaded")
    return json.loads(p.read_text())["file_hashes"]


def verify_head(head: str, coin: str, *, fits: Path | None = None,
                hashes: dict | None = None) -> dict:
    """The head's files, verified against the manifest BEFORE any of them
    is read as a model.  Returns `{filename: sha16}` for the receipt."""
    if head not in HEADS:
        # SITE: verify_head#1
        raise ScoreStreamRefused(f"unknown head {head!r}; the diagnostic "
                                 f"names {sorted(HEADS)}")
    if coin not in COINS:
        # SITE: verify_head#2
        raise ScoreStreamRefused(f"unknown coin {coin!r}, not in {COINS}")
    d = fits or FITS
    h = hashes if hashes is not None else manifest_hashes()
    out = {}
    for pat in HEADS[head]:
        name = pat.format(coin=coin)
        f = d / name
        if not f.exists():
            # SITE: verify_head#3
            raise ScoreStreamRefused(f"{head}/{coin}: {name} is absent")
        if name not in h:
            # SITE: verify_head#4
            raise ScoreStreamRefused(
                f"{head}/{coin}: {name} is not in the manifest, so nothing "
                f"says which bytes it should have")
        got = _sha16(f.read_bytes())
        if got != h[name]:
            # SITE: verify_head#5
            raise ScoreStreamRefused(
                f"{head}/{coin}: {name} has sha {got}, the manifest says "
                f"{h[name]} -- the bytes moved under a name that is bound")
        out[name] = got
    return out


def coin_of(filename: str) -> str | None:
    """Which coin a fit file belongs to, from the manifest's naming."""
    for c in COINS:
        if f"_{c}." in filename:
            return c
    return None


def score_events(rows, *, head: str, coin: str, scorer,
                 verified: dict) -> list[dict]:
    """`{t, slug, side, score}` per generation row, in the shape
    `harmful_stateful_policy.validate_scores` demands.

    `scorer` is the loaded head applied to one row.  `verified` is the
    output of `verify_head` and is REQUIRED: an adapter that could score
    without it would be an adapter that could score an unbound file."""
    if not verified:
        # SITE: score_events#1
        raise ScoreStreamRefused(
            "no verified head supplied; the manifest check is not optional "
            "-- it is the only thing that says these bytes are that head")
    wrong = sorted({n for n in verified if coin_of(n) not in (None, coin)})
    if wrong:
        # SITE: score_events#2
        raise ScoreStreamRefused(
            f"{coin}: the verified files {wrong} belong to another coin. A "
            f"cross-coin fit LOADS PERFECTLY and scores nonsense, which is "
            f"why the manifest -- which knows the coin -- refuses it and "
            f"the model cannot")
    out = []
    for i, r in enumerate(rows):
        # DE37-C1: `gen` is REQUIRED. The (gamma) control permutes the
        # above-threshold VALUES over GENERATIONS within a stratum, so an
        # event that does not name its generation cannot be placed --
        # round 37's stream carried none, every key collapsed to
        # `(slug, side, None)`, and the permutation silently became the
        # identity. A missing generation is refused here, at the adapter
        # that builds the stream, rather than inferred downstream.
        for k in REQUIRED_EVENT_KEYS:
            if k not in r:
                # SITE: score_events#3
                raise ScoreStreamRefused(f"row[{i}]: missing {k!r}")
        if r["side"] not in SIDES:
            # SITE: score_events#4
            raise ScoreStreamRefused(f"row[{i}]: side {r['side']!r}")
        v = scorer(r)
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or math.isnan(v):
            # SITE: score_events#5
            raise ScoreStreamRefused(
                f"row[{i}]: score {v!r} refused -- a NaN compares False "
                f"against every threshold and becomes a silent no-op "
                f"(harmful_stateful_policy:383-404 refuses it downstream; "
                f"refusing it here names the row)")
        # DE38-R3: ONE SOURCE. The required-key tuple and the event
        # construction were two lists saying the same thing, so dropping a
        # key from the check left it in the output and the failure arrived
        # as a bare `KeyError` from this line. Built from the tuple, a key
        # removed from the contract is removed from the events, and the
        # runner's `null#3` refuses BY NAME.
        out.append({**{k: r[k] for k in REQUIRED_EVENT_KEYS},
                    "score": float(v), "head": head, "coin": coin})
    return out


def lift(events, harmful: dict) -> float:
    """A crude ranking lift: mean score on harmful slugs minus mean score
    on the rest.  It exists so the permuted-score control has something to
    destroy (rule 15), and it is not a decision metric."""
    a = [e["score"] for e in events if harmful.get(e["slug"])]
    b = [e["score"] for e in events if not harmful.get(e["slug"])]
    if not a or not b:
        return 0.0
    return sum(a) / len(a) - sum(b) / len(b)


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_score_stream] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle=None):
        try:
            fn()
        except ScoreStreamRefused as exc:
            if needle and needle not in str(exc):
                raise SystemExit(f"[de_score_stream] FAIL: {label} -- "
                                 f"refused for another reason ({exc})")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_score_stream] FAIL (no refusal): {label}")

    # ---- the REAL manifest and the REAL heads ---------------------------
    h = manifest_hashes()
    ok(h.get("linear_d_btc.json") == "18701008c2bd18c6",
       f"the incumbent head's sha is the one R-398 names: "
       f"linear_d_btc.json {h.get('linear_d_btc.json')}")
    v_inc = verify_head("incumbent_linear_d", "btc")
    v_q1 = verify_head("q1_arrival_composed_lgbm", "btc")
    ok(v_inc == {"linear_d_btc.json": "18701008c2bd18c6"},
       f"INCUMBENT HEAD VERIFIED at the bytes: {v_inc}")
    ok(set(v_q1) == {"lgbm_haz_btc.txt", "lgbm_thresholds_btc.json"}
       and all(v_q1[k] == h[k] for k in v_q1),
       f"HEAD UNDER TEST VERIFIED at the bytes -- R-424's component of "
       f"record, Q1_arrival of composed_lgbm: {v_q1}")
    ok(verify_head("q1_arrival_composed_lgbm", "eth"),
       "and both coins' files verify, so the diagnostic can run btc and "
       "eth from the same manifest")

    # ---- KNOWN-BADS on the binding --------------------------------------
    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        dd = Path(d)
        (dd / "linear_d_btc.json").write_text("{}")
        refuses(lambda: verify_head("incumbent_linear_d", "btc", fits=dd,
                                    hashes=h),
                "KNOWN-BAD: a file whose BYTES do not match the manifest "
                "REFUSES, naming both shas -- the bytes moved under a name "
                "that is bound",
                needle="the bytes moved under a name")
        refuses(lambda: verify_head("incumbent_linear_d", "btc", fits=dd,
                                    hashes={}),
                "KNOWN-BAD: a file the manifest does not mention REFUSES -- "
                "nothing says which bytes it should have",
                needle="not in the manifest")
        refuses(lambda: verify_head("incumbent_linear_d", "btc",
                                    fits=dd / "nope", hashes=h),
                "KNOWN-BAD: an absent file REFUSES by name", needle="absent")
    refuses(lambda: verify_head("hazard_only", "btc"),
            "KNOWN-BAD: an unknown head REFUSES and names the ones the "
            "diagnostic declares", needle="unknown head")
    refuses(lambda: verify_head("incumbent_linear_d", "sol"),
            "KNOWN-BAD: an unknown coin REFUSES", needle="unknown coin")
    refuses(lambda: manifest_hashes(Path("/nonexistent/fit_manifest.json")),
            "KNOWN-BAD: a missing manifest REFUSES -- an unbound fit is not "
            "loaded", needle="bound by their shas")

    # ---- the adapter's own shape ----------------------------------------
    rows = [{"t": 1000.0 + i, "slug": f"btc-updown-5m-{i}",
             "side": SIDES[i % 2], "gen": 1 + i // 5, "x": i / 10.0}
            for i in range(10)]
    ev = score_events(rows, head="q1_arrival_composed_lgbm", coin="btc",
                      scorer=lambda r: r["x"], verified=v_q1)
    ok(len(ev) == 10 and all(set(e) >= {"t", "slug", "side", "gen", "score"}
                             for e in ev)
       and [e["gen"] for e in ev] == [r["gen"] for r in rows],
       f"the adapter emits the shape `validate_scores` demands -- "
       f"{sorted(ev[0])} -- one event per generation row, and the "
       f"GENERATION travels with it (DE37-C1: without it the null's "
       f"permutation cannot name what it is permuting)")
    refuses(lambda: score_events(
        [{"t": 1.0, "slug": "btc-updown-5m-1", "side": SIDES[0]}],
        head="q1_arrival_composed_lgbm", coin="btc",
        scorer=lambda r: 0.5, verified=v_q1),
        "KNOWN-BAD: a row with no `gen` REFUSES -- an event that does not "
        "name its generation collapses every key in its stratum to one, "
        "and the (gamma) permutation becomes the identity in silence",
        needle="missing 'gen'")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import harmful_stateful_policy as _hsp
        _hsp.validate_scores(ev)
        _accepted = True
    except Exception as _e:                                # pragma: no cover
        _accepted = f"{type(_e).__name__}: {_e}"
    ok(_accepted is True,
       f"and THE CONSUMER ITSELF accepts them: "
       f"`harmful_stateful_policy.validate_scores` returns on this stream "
       f"({_accepted}) -- the shape is checked against the code that reads "
       f"it, not against my reading of its docstring")
    refuses(lambda: score_events(rows, head="q1_arrival_composed_lgbm",
                                 coin="btc", scorer=lambda r: float("nan"),
                                 verified=v_q1),
            "KNOWN-BAD: a NaN score REFUSES AT THE ROW -- downstream it "
            "would compare False against every threshold and become a "
            "silent no-op", needle="silent no-op")
    refuses(lambda: score_events(rows, head="q1_arrival_composed_lgbm",
                                 coin="btc", scorer=lambda r: r["x"],
                                 verified={}),
            "KNOWN-BAD: scoring with NO verified head REFUSES -- the "
            "manifest check is not optional", needle="not optional")
    refuses(lambda: score_events(
        rows, head="q1_arrival_composed_lgbm", coin="btc",
        scorer=lambda r: r["x"],
        verified=verify_head("q1_arrival_composed_lgbm", "eth")),
        "KNOWN-BAD: an ETH fit scoring BTC rows REFUSES -- a cross-coin fit "
        "LOADS PERFECTLY and scores nonsense, so the refusal comes from the "
        "manifest, which knows the coin, and not from the model, which "
        "cannot", needle="belong to another coin")
    refuses(lambda: score_events([{"t": 1.0, "slug": "s"}],
                                 head="incumbent_linear_d", coin="btc",
                                 scorer=lambda r: 1.0, verified=v_inc),
            "KNOWN-BAD: a row missing `side` REFUSES by name",
            needle="missing 'side'")
    refuses(lambda: score_events([{"t": 1.0, "slug": "s", "side": "LONG", "gen": 1}],
                                 head="incumbent_linear_d", coin="btc",
                                 scorer=lambda r: 1.0, verified=v_inc),
            "KNOWN-BAD: an unknown side REFUSES", needle="side 'LONG'")

    # ---- the PERMUTED-SCORE control (rule 15) ---------------------------
    harmful = {r["slug"]: (i >= 5) for i, r in enumerate(rows)}
    real = lift(ev, harmful)
    import random as _rnd
    rng = _rnd.Random(11)
    lifts = []
    for _ in range(200):
        vals = [e["score"] for e in ev]
        rng.shuffle(vals)
        lifts.append(lift([dict(e, score=v) for e, v in zip(ev, vals)],
                          harmful))
    _beaten = sum(1 for x in lifts if x >= real)
    ok(real > 0 and _beaten <= 2,
       f"POSITIVE CONTROL: the real stream ranks the harmful slugs above "
       f"the rest (lift {real:.3f}) and only {_beaten} of 200 PERMUTED "
       f"streams match it -- the permutation destroys the lift, which is "
       f"what says the lift came from the ordering and not from the scale")
    ok(abs(sum(lifts) / len(lifts)) < abs(real),
       f"and the permuted mean is near zero ({sum(lifts) / len(lifts):.3f} "
       f"against {real:.3f}), so the control is centred where a null "
       f"should be rather than merely noisy")
    ok(len(lifts) == 200,
       "with 200 draws, the protocol's declared minimum (§6), not a number "
       "chosen after looking")

    # ---- the DECLARED LIMIT is in the module, not only in the addendum --
    ok("IR-R4" in (__doc__ or "") and "does NOT read features from a file"
       in (__doc__ or ""),
       "THE IR-R4 LIMIT IS DECLARED IN THIS FILE: no generation-tranche "
       "artifact has a production consumer, so the adapter takes the "
       "caller's table and invents no population -- stated rather than "
       "worked around (R-459 §3)")
    _src = Path(__file__).read_text()
    _t = _ast.parse(_src)
    # THE EVENT PATH -- the functions that build or shape the stream. The
    # HEAD-BINDING path (`manifest_hashes`, `verify_head`) reads files by
    # design: that is how a fit's bytes are bound, and IR-R4's limit is
    # about FEATURES, not about the manifest. Naming the two rather than
    # excluding "everything that reads" is the point.
    _evpath = {"score_events", "lift", "coin_of", "validate_scores"}
    _reads = {"open", "read_text", "read_bytes", "load", "loads", "loadtxt"}
    _opens = [(fn.name, _ast.unparse(nd)[:60]) for fn in _ast.walk(_t)
              if isinstance(fn, _ast.FunctionDef) and fn.name in _evpath
              for nd in _ast.walk(fn)
              if (isinstance(nd, _ast.Call)
                  and (getattr(nd.func, "id", "") == "open"
                       or getattr(nd.func, "attr", "") in _reads))]
    _binders = sorted({fn.name for fn in _ast.walk(_t)
                       if isinstance(fn, _ast.FunctionDef)
                       and fn.name not in ("selftest", "main")
                       for nd in _ast.walk(fn)
                       if isinstance(nd, _ast.Call)
                       and getattr(nd.func, "attr", "") in _reads})
    ok(not _opens and _binders == ["manifest_hashes", "verify_head"],
       f"DE38-R2: AND THE LIMIT IS COMPUTED, not only declared -- the "
       f"EVENT PATH ({sorted(_evpath)}) opens no file ({len(_opens)} such "
       f"calls, from the parse), and the only functions in this module "
       f"that read one are {_binders} (the suite excepted -- it reads "
       f"fixtures), which bind the head's BYTES. The "
       f"docstring states the limit; this predicate goes red the day an "
       f"edit makes the adapter read a population instead of taking the "
       f"caller's table (IR-R4)")
    ok(SIDES == tuple(_hsp.SIDES) and "BUY_UP" in SIDES,
       f"AND THE SIDE VOCABULARY IS THE CONSUMER'S OWN OBJECT: {SIDES} -- "
       f"imported from `harmful_stateful_policy`, not restated. The first "
       f"version of this adapter restated `('BUY', 'SELL')` and built a "
       f"stream the policy refused; the check that caught it is the one "
       f"above, which runs the CONSUMER over the output instead of "
       f"asserting the shape from a docstring (rule 16)")
    ok(coin_of("lgbm_haz_eth.txt") == "eth"
       and coin_of("fit_manifest.json") is None,
       f"and the coin of a fit is read from the manifest's naming "
       f"({coin_of('lgbm_haz_eth.txt')!r}), with a file that names no coin "
       f"reading None rather than guessing")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_score_stream] selftest OK -- {n[0]} checks")
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
