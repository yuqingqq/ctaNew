"""THE WHOLE VALIDATION PATH, REHEARSED BEFORE THE CLOCK STARTS.

Day 1 of §8's clock is a day that cannot be re-taken: rule 11 consumes
the days whatever happens, and a crash consumes them just as thoroughly
as a result does. So this drives the entire path on synthetic-but-
realistic inputs -- day inputs -> per-day log loss for C1, C2 and
Identity -> delta_LL_g -> the exact 2^G sign test -> Holm at m=2 -> the
four conditions as a conjunction -> the futility rung -> a verdict
artifact ON DISK -- and then the §9 path the same way, including the fee
sensitivity.

WHAT THIS IS FOR IS NOT "IT PASSES". It is the inventory of every
refusal the path can emit, each one DRIVEN, with the one question that
matters beside it: would this fire on an ORDINARY day? A refusal that
can fire on ordinary data is a landmine under a run that cannot be
re-taken.

THE SIGN CONVENTION IS THE FIRST THING A REHEARSAL CATCHES.
`score_actions` reports `increment_vs_identity = policy_LL - identity_LL`
-- log loss, so BETTER is NEGATIVE -- while §8's verdict requires the
mean and median increment to be POSITIVE. Wiring one into the other
without the flip would fail every good candidate and pass every bad one.
The conversion is named here, and it is driven both ways: a genuinely
better candidate must come out positive and a worse one negative.

THREE THINGS PROVED RATHER THAN ASSUMED:

  (a) the path completes with a candidate that is NON-OK on some
      actions. Identity fallback is the normal case, not the exception.
  (b) it completes on a day whose increment is EXACTLY ZERO, which is a
      REPORTED TIE -- never a crash, never the candidate's sign.
  (c) it completes with the fee UNDECLARED, which is the state today. If
      the §8 leg could not run without a fee, the predictive clock would
      be blocked by an economic gap.

Usage:  de_fair_value_rehearsal.py --falsify
        de_fair_value_rehearsal.py --rehearse [--out DIR]
        de_fair_value_rehearsal.py --inventory
"""
from __future__ import annotations

import ast
import json
import random
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import da_fair_price_identity as FP                # noqa: E402
import de_fair_value_actions as ACT                # noqa: E402
import de_fair_value_economic as ECON              # noqa: E402
import de_fair_value_fee_sensitivity as SENS       # noqa: E402
import de_fair_value_pnl as PNL                    # noqa: E402
import de_fair_value_predictive as PRED            # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_REHEARSAL_V1"

#: THE FLIP, NAMED. Log loss: lower is better, so an improvement over
#: Identity is Identity MINUS policy. §8's conditions read positive as
#: better, and this is the only place the two conventions meet.
INCREMENT_SIGN = "delta_LL_g = Identity_mean_LL - policy_mean_LL"

#: The modules whose refusals can reach a run on this path.
PATH_MODULES = ("de_fair_value_actions", "de_fair_value_predictive",
                "de_fair_value_pnl", "de_fair_value_economic",
                "de_fair_value_fee_sensitivity")

ORDINARY = "ORDINARY"          # something a normal day can contain
TODAY = "TODAY"                # fires right now, BEFORE a leg starts
DEFECT = "DEFECT"              # a build/wiring fault, not ordinary data
POPULATION = "POPULATION"      # a legitimate stop on a short population

W0 = 1788825600
SLUGS = {"btc": "btc-updown-5m-", "eth": "eth-updown-5m-"}


class RehearsalRefused(ValueError):
    """The rehearsal itself could not be driven as declared."""


# ---------------------------------------------------------- fixtures

def _fp(coin, ws, value, *, status=FP.OK, estimator="Identity", lk=None):
    """One fair price, with value None whenever the status is not OK."""
    src = float(ws)
    fresh = 0.5
    return FP.FairPrice(
        coin=coin, window_start=ws, outcome="UP",
        value=(float(value) if status == FP.OK else None),
        source_timestamp=src, freshness_s=fresh,
        local_knowledge_timestamp=(src + fresh if lk is None else float(lk)),
        status=status, estimator=estimator)


def synthetic_day(day: str, *, seed: int, n_slugs: int = 8,
                  cand_edge: float = 0.35, fallback_frac: float = 0.25,
                  identity_bad_frac: float = 0.05,
                  tie: bool = False, coins=("btc", "eth")) -> dict:
    """ONE REALISTIC DAY: actions, both estimators, and the outcomes.

    `cand_edge` shrinks the candidate's error toward the truth, so a
    positive edge is a genuinely better candidate. `fallback_frac` is the
    share of actions where the candidate is NOT natively OK -- the normal
    case, not the exception.
    """
    rng = random.Random(seed)
    consumptions, identity, challenger, outcomes, native = [], {}, {}, {}, {}
    pop_rows = []
    for coin in coins:
        native[coin] = {"ok": 0, "eligible": 0}
        for i in range(n_slugs):
            ws = W0 + i * 300
            slug = f"{SLUGS[coin]}{ws}"
            gen = f"g{i}"
            stamp = 1_788_000_000_000_000_000 + i * 1_000_000
            pop_rows.append((slug, gen))
            p_true = rng.uniform(0.15, 0.85)
            up = rng.random() < p_true
            ident_p = min(0.98, max(0.02, p_true + rng.gauss(0, 0.10)))
            if tie:
                cand_p, cand_status = ident_p, FP.OK
            else:
                cand_p = ident_p + cand_edge * (p_true - ident_p)
                cand_status = FP.OK
            if rng.random() < fallback_frac:
                cand_status = "STALE"      # the normal non-OK case
            ident_status = (FP.OK if rng.random() >= identity_bad_frac
                            else "NO_INPUT")
            identity[(coin, slug, gen)] = _fp(coin, ws, ident_p,
                                              status=ident_status)
            challenger[(coin, slug, gen)] = _fp(
                coin, ws, cand_p, status=cand_status, estimator="C1")
            outcomes[slug] = up
            if ident_status == FP.OK:
                native[coin]["eligible"] += 1
                if cand_status == FP.OK:
                    native[coin]["ok"] += 1
            for side in ("BID", "ASK"):
                consumptions.append(
                    {"coin": coin, "slug": slug, "generation_id": gen,
                     "decision_recv_ns": stamp, "quote_side": side,
                     "up_probability_consumed": round(ident_p, 6),
                     "window_start": ws,
                     "on_identity_reference_path": True})
    population = {"actions": pop_rows,
                  "population": "P003_NEUTRAL_REFERENCE_PATH_REHEARSAL",
                  "as_of": f"{day}T00:00:00Z",
                  "source_identity": "de_fair_value_rehearsal.synthetic_day"}
    return {"day": day, "consumptions": consumptions,
            "population": population, "identity": identity,
            "challenger": challenger, "outcomes": outcomes,
            "native": native}


def _lookup(table, worsen: float = 0.0):
    def read(a):
        rec = table.get((a.coin, a.slug, a.generation_id))
        if rec is None or worsen == 0.0 or rec.status != FP.OK:
            return rec
        return _fp(rec.coin, rec.window_start,
                   min(0.98, max(0.02, rec.value + worsen)),
                   estimator=rec.estimator)
    return read


# ------------------------------------------------------------- §8 path

def score_day(fx: dict, *, worsen: float = 0.0) -> dict:
    """Build the day's canonical actions and score them, paired."""
    built = ACT.build_actions(fx["consumptions"],
                              canonical_population=fx["population"])
    scored = ACT.score_actions(built["actions"],
                               _lookup(fx["identity"]),
                               _lookup(fx["challenger"], worsen),
                               fx["outcomes"])
    return {"built": built, "scored": scored}


def delta_ll(scored: dict) -> dict:
    """THE FLIP, IN ONE PLACE. Positive = the policy beat Identity."""
    prim = scored["PRIMARY_fallback_scored"]
    i, c = prim["identity_mean_log_loss"], prim["policy_mean_log_loss"]
    if i is None or c is None:
        raise RehearsalRefused(
            "REFUSED REHEARSAL_DAY_HAS_NO_SCORED_ACTIONS: a day with no "
            "Identity-eligible action has no increment, and zero is not "
            "the same as no evidence.")
    return {"delta_LL": i - c, "identity_mean_LL": i, "policy_mean_LL": c,
            "n": prim["n"], "n_native": prim["n_native"],
            "n_fallback": prim["n_fallback"],
            "sign_convention": INCREMENT_SIGN,
            "is_a_tie": (i - c) == 0.0}


def predictive_path(day_fixtures, *, blind_rows, worsen_by_candidate=None,
                    outdir: Path = None) -> dict:
    """THE WHOLE §8 LEG, end to end, for C1 and C2 together."""
    worsen_by_candidate = worsen_by_candidate or {"C1": 0.0, "C2": 0.12}
    elig = PRED.eligible_days(blind_rows)
    acc = PRED.accrual(blind_rows)
    per_candidate, native, eligible_counts = {}, {}, {}
    for name, worsen in worsen_by_candidate.items():
        incs, days = [], []
        for fx in day_fixtures:
            if fx["day"] not in elig["evaluable_days"]:
                continue
            got = score_day(fx, worsen=worsen)
            d = delta_ll(got["scored"])
            incs.append(d["delta_LL"])
            days.append({"day": fx["day"], **d})
            for coin, c in fx["native"].items():
                native[coin] = native.get(coin, 0) + c["ok"]
                eligible_counts[coin] = eligible_counts.get(coin, 0) + c["eligible"]
        per_candidate[name] = {"increments": incs, "days": days}
    cov = PRED.coverage_gate(native, eligible_counts)
    pvals = {n: PRED.exact_sign_p(v["increments"]).get("p")
             for n, v in per_candidate.items()}
    holm = PRED.holm(pvals)
    out = {"protocol": PROTOCOL, "leg": "SECTION_8_PREDICTIVE",
           "sign_convention": INCREMENT_SIGN,
           "eligibility": elig, "accrual": acc, "coverage": cov,
           "p_values": pvals, "holm": holm, "candidates": {}}
    for name, v in per_candidate.items():
        ties = [d["day"] for d in v["days"] if d["is_a_tie"]]
        out["candidates"][name] = {
            "increments": v["increments"], "days": v["days"],
            "exact_sign_test": PRED.exact_sign_p(v["increments"]),
            "ties_reported_never_favourable": ties,
            "futility": PRED.futility(v["increments"]),
            "verdict": PRED.verdict(
                name, increments=v["increments"], holm_row=holm[name],
                coverage=cov,
                predicates={"actions_are_canonical": True,
                            "identity_pairing_is_per_action": True}),
        }
    if outdir is not None:
        write_artifact(Path(outdir) / "rehearsal_section8_verdict.json", out)
    return out


# ------------------------------------------------------------- §9 path

def economic_path(day_fixtures, *, declared_fee=None, decl_dir=None,
                  params=None, outdir: Path = None) -> dict:
    """THE WHOLE §9 LEG: day legs, both gates, and the sensitivity."""
    pairs = []
    for k, fx in enumerate(day_fixtures):
        t = 1_700_000_000_000.0 + k * 86_400_000.0
        cand = tuple(PNL.Fill(slug=s, token="UP", ts_ms=t + 1000.0,
                              q=0.50 + 0.01 * j, dq=10.0,
                              order_decision_ms=t, maker=True)
                     for j, s in enumerate(sorted(fx["outcomes"])[:3]))
        iden = tuple(PNL.Fill(slug=s, token="UP", ts_ms=t + 1000.0,
                              q=0.52 + 0.01 * j, dq=10.0,
                              order_decision_ms=t, maker=True)
                     for j, s in enumerate(sorted(fx["outcomes"])[:3]))
        pairs.append(SENS.DayPair(
            day=fx["day"], candidate_fills=cand, identity_fills=iden,
            settlement={"UP": 1.0}, candidate_active_ms=86_400_000.0,
            identity_active_ms=86_400_000.0))
    fee = declared_fee if declared_fee is not None else PNL.declared_fee(
        decl_dir)
    out = SENS.run("C1", pairs, declared_fee=fee, latency_ms=0.0,
                   params=params, decl_dir=decl_dir)
    out = {"protocol": PROTOCOL, "leg": "SECTION_9_ECONOMIC", **out}
    if outdir is not None:
        write_artifact(Path(outdir) / "rehearsal_section9_verdict.json", out)
    return out


def write_artifact(path: Path, doc: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, default=str))
    return path


# ------------------------------------------------- the refusal inventory

def scan_refusal_names(module: str) -> dict:
    """EVERY refusal name a path module can emit, from its source.

    Module-level constants whose VALUE is a refusal token and whose NAME
    is raised somewhere. Scanned, never hand-listed: a hand-list is the
    thing that goes stale silently.
    """
    src = (HERE / f"{module}.py").read_text()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        text = None
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            text = val.value
        elif (isinstance(val, ast.JoinedStr) or isinstance(val, ast.BinOp)):
            continue
        if text is None or not text.replace("_", "").isalnum():
            continue
        if not (text.isupper() and "_" in text and len(text) > 8):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and (
                    f"REFUSED {{{t.id}}}" in src or f"{{{t.id}}}:" in src):
                out[text] = {"module": module, "constant": t.id}
    return out


def scan_unnamed_refusals(module: str = None, src: str = None) -> list:
    """RAISES THAT NAME NOTHING. An anonymous refusal cannot be
    inventoried, so it cannot be cleared before a run that cannot be
    re-taken -- and it reaches a reader as prose to grep."""
    src = src if src is not None else (HERE / f"{module}.py").read_text()
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        fn = getattr(node.exc, "func", None)
        cls = getattr(fn, "id", None) or getattr(fn, "attr", None) or ""
        if not cls.endswith("Refused") and not cls.endswith("Inadmissible"):
            continue                       # SystemExit is not a refusal
        args = getattr(node.exc, "args", [])
        if not args:
            continue
        a = args[0]
        parts = a.values if isinstance(a, ast.JoinedStr) else [a]
        # A FORWARDED message is named by the refusal it forwards.
        if parts and isinstance(parts[0], ast.FormattedValue):
            continue
        if isinstance(a, ast.Call):
            continue
        named = False
        for i, piece in enumerate(parts):
            if (isinstance(piece, ast.Constant)
                    and isinstance(piece.value, str)):
                txt = piece.value
                if "REFUSED " in txt:
                    tail = txt.split("REFUSED ", 1)[1]
                    if tail[:6].isupper() and tail[:6].strip("_ "):
                        named = True
                    elif (i + 1 < len(parts)
                          and isinstance(parts[i + 1], ast.FormattedValue)
                          and txt.endswith("REFUSED ")):
                        named = True
        if not named:
            out.append({"module": module, "line": node.lineno})
    return out


def _refusal_name(exc) -> str:
    txt = str(exc)
    if "REFUSED " not in txt:
        return "UNNAMED_REFUSAL"
    head = txt.split("REFUSED ", 1)[1]
    for stop in (":", " ", "("):
        head = head.split(stop)[0]
    return head.strip()


def drive_perturbations(outdir: Path) -> list:
    """DRIVE each condition and record what it emits. Nothing is read."""
    rows = []

    def drive(label, kind, fn, expect_completion=False):
        try:
            fn()
            rows.append({"label": label, "kind": kind,
                         "emitted": None, "completed": True})
        except Exception as exc:                            # noqa: BLE001
            rows.append({"label": label, "kind": kind,
                         "emitted": _refusal_name(exc),
                         "exception": type(exc).__name__,
                         "completed": False,
                         "detail": str(exc)[:120]})
        return rows[-1]

    days = [synthetic_day(f"2026-09-{13 + i:02d}", seed=100 + i)
            for i in range(10)]
    blind = [PRED.BlindDayInputs(day=d["day"], book_gate_pass=True,
                                 official_resolutions_present=True,
                                 settlement_verification_covered=True)
             for d in days]

    # --- (a) (b) (c): the three ORDINARY cases that must COMPLETE ----
    drive("NORMAL DAY: candidate non-OK on ~25% of actions "
          "(Identity fallback is the normal case)", ORDINARY,
          lambda: predictive_path(days, blind_rows=blind, outdir=outdir))
    tie_days = [synthetic_day(f"2026-09-{13 + i:02d}", seed=200 + i,
                              tie=(i == 3)) for i in range(10)]
    drive("NORMAL DAY: one day's increment is EXACTLY ZERO", ORDINARY,
          lambda: predictive_path(tie_days, blind_rows=blind))
    drive("NORMAL DAY: the fee is UNDECLARED and the §8 leg runs",
          ORDINARY,
          lambda: predictive_path(days, blind_rows=blind))
    drive("NORMAL DAY: an action's coin is out of scope", ORDINARY,
          lambda: ACT.build_actions(
              [dict(days[0]["consumptions"][0], coin="sol")],
              canonical_population=days[0]["population"]))
    drive("NORMAL DAY: a leg with zero filled shares", ORDINARY,
          lambda: PNL.settlement_edge((), settlement={"UP": 1.0}))

    # --- §8 defects --------------------------------------------------
    drive("an outcome is missing for a slug", DEFECT,
          lambda: ACT.score_actions(
              ACT.build_actions(days[0]["consumptions"],
                                canonical_population=days[0]["population"]
                                )["actions"],
              _lookup(days[0]["identity"]), _lookup(days[0]["challenger"]),
              {}))
    drive("no canonical population supplied", DEFECT,
          lambda: ACT.build_actions(days[0]["consumptions"],
                                    canonical_population=None))
    drive("the population is anonymous (no provenance)", DEFECT,
          lambda: ACT.build_actions(days[0]["consumptions"],
                                    canonical_population=[("s", "g")]))
    drive("a row's reference-path flag contradicts the population",
          DEFECT,
          lambda: ACT.build_actions(
              [dict(days[0]["consumptions"][0],
                    on_identity_reference_path=False)],
              canonical_population=days[0]["population"]))
    drive("two rows at one key disagree about the value consumed", DEFECT,
          lambda: ACT.build_actions(
              [days[0]["consumptions"][0],
               dict(days[0]["consumptions"][0], quote_side="ASK",
                    up_probability_consumed=0.99)],
              canonical_population=days[0]["population"]))
    drive("day eligibility is handed a candidate-named field", DEFECT,
          lambda: PRED.assert_blind(
              type("X", (), {"day": "d", "candidate_score": 1.0})()))
    drive("fewer than ten evaluable days in the band", POPULATION,
          lambda: _raise_if(PRED.accrual(blind[:6])["refusal"]))
    drive("a day with no scored actions", POPULATION,
          lambda: PRED.day_log_loss([]))
    drive("more days scored than the declared G", DEFECT,
          lambda: PRED.futility([0.1] * 11))

    # --- §9 defects and states ---------------------------------------
    good_fee = dict(SENS.ZERO)
    params = SENS.sensitivity_parameters(
        SENS._decl(str(Path(outdir) / "decl")))
    drive("THE STATE TODAY: the fee is undeclared, so the §9 leg stops",
          TODAY,
          lambda: economic_path(days, params=params))
    drive("a fee value with no supporting rule", DEFECT,
          lambda: SENS.fee_provenance({"value": 0.0}))
    drive("the fee source cannot distinguish charged from uncharged",
          DEFECT,
          lambda: SENS.fee_provenance(dict(good_fee, source=SENS._ws_source)))
    drive("the fee source probe has no uncharged control", DEFECT,
          lambda: SENS.fee_provenance(
              {k: v for k, v in good_fee.items()
               if k != "uncharged_controls"}))
    drive("a status string in place of a probeable source", DEFECT,
          lambda: SENS.fee_provenance(dict(good_fee, source="OBSERVED")))
    drive("the sensitivity rate is not declared", TODAY,
          lambda: SENS.sensitivity_parameters(
              SENS._decl(str(Path(outdir) / "norate"),
                         drop=("sensitivity_rate_worst_observed",))))
    drive("the sensitivity basis is pinned to the observed price", DEFECT,
          lambda: SENS.sensitivity_parameters(
              SENS._decl(str(Path(outdir) / "basis"),
                         basis=SENS.OBSERVED_CONSTANT)))
    drive("the sensitivity rate is the modal tier, not the cap", DEFECT,
          lambda: SENS.sensitivity_parameters(
              SENS._decl(str(Path(outdir) / "modal"), rate=0.099)))
    drive("the sensitivity charges one leg only", DEFECT,
          lambda: SENS._arm([], fee_candidate={"value": 1},
                            fee_identity={"value": 2}, latency_ms=0.0))
    drive("the qualified fee is asked for a verdict with no sensitivity",
          DEFECT,
          lambda: SENS.run("C1", [], declared_fee=good_fee, latency_ms=0.0,
                           with_sensitivity=False))
    drive("the worst-case fee is used as the primary", DEFECT,
          lambda: SENS.fee_provenance(SENS.worst_case_fee(params)))
    drive("an eleventh day is offered for a not-evaluable one", DEFECT,
          lambda: ECON.evaluate("C1", [{"day": f"d{i}", "delta_pnl": 1.0,
                                        "pair_comparable_for_edge": True,
                                        "edge_increment": 1.0,
                                        "filled_shares": {"candidate": 1,
                                                          "identity": 1}}
                                       for i in range(11)],
                                holm_pnl={}, holm_edge={}))
    drive("an interval resamples something other than UTC days", DEFECT,
          lambda: ECON.day_bootstrap([1.0, 2.0], unit="fills"))
    drive("a fill lands before its order was effective", DEFECT,
          lambda: PNL.pnl([PNL.Fill(slug="s", token="UP", ts_ms=1.0,
                                    q=0.5, dq=1.0,
                                    order_decision_ms=1000.0)],
                          settlement={"UP": 1.0},
                          fee={"value": 0.0}, placement_latency_ms=250.0,
                          quote_active_ms=1.0))
    drive("a P&L is asked for with no quote-active time", DEFECT,
          lambda: PNL.pnl([], settlement={}, fee={"value": 0.0},
                          placement_latency_ms=0.0))
    # --- the remainder of the scanned set, driven ---------------------
    wc = SENS.worst_case_fee(params)
    drive("the worst case is applied to a TAKER leg", DEFECT,
          lambda: PNL.fee_for(PNL.Fill(slug="s", token="UP", ts_ms=1.0,
                                       q=0.5, dq=1.0,
                                       order_decision_ms=0.0, maker=False),
                              wc))
    drive("a fee model this code does not implement", DEFECT,
          lambda: PNL.fee_for(PNL.Fill(slug="s", token="UP", ts_ms=1.0,
                                       q=0.5, dq=1.0,
                                       order_decision_ms=0.0),
                              {"value": 0.1, "model": "invented"}))
    drive("the fee source charges the uncharged controls too", DEFECT,
          lambda: SENS.fee_provenance(
              dict(good_fee, source=lambda a: 0.099)))
    drive("the sensitivity consumed different FILLS", DEFECT,
          lambda: SENS.run("C1", SENS._pairs(), declared_fee=good_fee,
                           latency_ms=0.0, params=params,
                           primary_pairs=SENS._pairs(cand_q=0.55)))
    drive("the sensitivity consumed a different DAY POPULATION", DEFECT,
          lambda: SENS.run("C1", SENS._pairs(), declared_fee=good_fee,
                           latency_ms=0.0, params=params, g_declared=11,
                           primary_pairs=SENS._pairs(first_day=2)))
    drive("the declared price basis is not recognised", DEFECT,
          lambda: SENS.sensitivity_parameters(
              SENS._decl(str(Path(outdir) / "badbasis"), basis="notional")))
    drive("a worst case is reported with no primary beside it", DEFECT,
          lambda: SENS.report({"worst_case": {}}))
    drive("a pair spans two portfolio days", DEFECT,
          lambda: ECON.day_increment(
              ECON.DayLeg(day="d1", pnl={}, edge={}),
              ECON.DayLeg(day="d2", pnl={}, edge={})))
    drive("A TOKEN HAS NO OFFICIAL SETTLEMENT", ORDINARY,
          lambda: PNL.pnl([PNL.Fill(slug="s", token="UP", ts_ms=1000.0,
                                    q=0.5, dq=1.0,
                                    order_decision_ms=1000.0)],
                          settlement={}, fee={"value": 0.0},
                          placement_latency_ms=0.0, quote_active_ms=1.0))
    return rows


def _raise_if(msg):
    if msg:
        raise PRED.PredictiveRefused(msg)


def inventory(outdir: Path) -> dict:
    """THE TABLE: every refusal, its trigger, and the one question."""
    scanned = {}
    for m in PATH_MODULES:
        scanned.update(scan_refusal_names(m))
    driven = drive_perturbations(outdir)
    by_name = {}
    for row in driven:
        if row["emitted"]:
            by_name.setdefault(row["emitted"], []).append(row)
    table = []
    for name in sorted(set(scanned) | set(by_name)):
        hits = by_name.get(name, [])
        kinds = {h["kind"] for h in hits}
        if not hits:
            normal = "NOT DRIVEN HERE"
        elif ORDINARY in kinds:
            normal = "YES -- LANDMINE (fires mid-run)"
        elif TODAY in kinds:
            normal = "YES -- but it stops the leg BEFORE it starts"
        elif POPULATION in kinds:
            normal = "POSSIBLE -- a legitimate stop, not a fault"
        else:
            normal = "no -- only under a defect"
        table.append({
            "refusal": name,
            "module": scanned.get(name, {}).get("module", "(driven only)"),
            "trigger": hits[0]["label"] if hits else "not exercised by "
                                                    "this rehearsal",
            "fires_on_a_normal_day": normal,
            "driven_here": bool(hits)})
    completions = [r for r in driven if r["completed"]]
    doc = {"protocol": PROTOCOL,
            "n_refusals_scanned": len(scanned),
            "n_driven": len(by_name),
            "n_not_driven": sum(1 for r in table if not r["driven_here"]),
            "landmines": [r for r in table if r[
                "fires_on_a_normal_day"].startswith("YES -- LANDMINE")],
            "blocked_today": [r for r in table if r[
                "fires_on_a_normal_day"].startswith("YES -- but")],
            "legitimate_stops": [r for r in table if r[
                "fires_on_a_normal_day"].startswith("POSSIBLE")],
           "ordinary_cases_that_completed":
               [r["label"] for r in completions],
           "table": table}
    write_artifact(Path(outdir) / "rehearsal_refusal_inventory.json", doc)
    return doc


# ---------------------------------------------------------- rehearsal

def rehearse(outdir=None) -> dict:
    outdir = Path(outdir or tempfile.mkdtemp(prefix="de_rehearsal_"))
    days = [synthetic_day(f"2026-09-{13 + i:02d}", seed=100 + i)
            for i in range(10)]
    blind = [PRED.BlindDayInputs(day=d["day"], book_gate_pass=True,
                                 official_resolutions_present=True,
                                 settlement_verification_covered=True)
             for d in days]
    s8 = predictive_path(days, blind_rows=blind, outdir=outdir)
    params = SENS.sensitivity_parameters(SENS._decl(str(outdir / "decl")))
    try:
        s9 = economic_path(days, declared_fee=dict(SENS.ZERO),
                           params=params, outdir=outdir)
        s9_state = "COMPLETED on a fixture fee"
    except Exception as exc:                                # noqa: BLE001
        s9, s9_state = None, f"REFUSED {_refusal_name(exc)}"
    inv = inventory(outdir)
    write_artifact(outdir / "rehearsal_refusal_inventory.json", inv)
    return {"outdir": str(outdir), "section8": s8, "section9": s9,
            "section9_state": s9_state, "inventory": inv}


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    out = Path(tempfile.mkdtemp(prefix="de_rehearsal_cells_"))
    days = [synthetic_day(f"2026-09-{13 + i:02d}", seed=100 + i)
            for i in range(10)]
    blind = [PRED.BlindDayInputs(day=d["day"], book_gate_pass=True,
                                 official_resolutions_present=True,
                                 settlement_verification_covered=True)
             for d in days]

    print("== the sign convention, driven both ways ==")
    better = delta_ll(score_day(days[0])["scored"])["delta_LL"]
    worse = delta_ll(score_day(days[0], worsen=0.25)["scored"])["delta_LL"]
    ck("a BETTER candidate yields a POSITIVE increment",
       better > 0, f"delta_LL {better:+.5f}")
    ck("a WORSE candidate yields a NEGATIVE one -- the flip is real",
       worse < 0, f"delta_LL {worse:+.5f}")
    ck("and the convention is named in the record, not implied",
       delta_ll(score_day(days[0])["scored"])["sign_convention"]
       == INCREMENT_SIGN)

    print("== (a) the path completes with the candidate non-OK ==")
    s8 = predictive_path(days, blind_rows=blind, outdir=out)
    fb = sum(d["n_fallback"] for d in s8["candidates"]["C1"]["days"])
    nat = sum(d["n_native"] for d in s8["candidates"]["C1"]["days"])
    ck("Identity fallback happened on a large share of actions",
       fb > 0 and nat > 0, f"{fb} fallback / {nat} native actions")
    ck("  and every evaluable day still produced an increment",
       len(s8["candidates"]["C1"]["increments"]) == 10)

    print("== (b) an exactly-zero day is a REPORTED TIE ==")
    tie_days = [synthetic_day(f"2026-09-{13 + i:02d}", seed=200 + i,
                              tie=(i == 3)) for i in range(10)]
    s8t = predictive_path(tie_days, blind_rows=blind)
    ties = s8t["candidates"]["C1"]["ties_reported_never_favourable"]
    est = s8t["candidates"]["C1"]["exact_sign_test"]
    ck("the tie is REPORTED, by day", len(ties) == 1, str(ties))
    ck("  and EXCLUDED from the sign count, never given the "
       "candidate's sign",
       est["n_ties_excluded"] == 1 and est["n_nonzero"] == 9)
    ck("  and the path did not crash on it", bool(s8t["candidates"]["C1"][
        "verdict"]["conditions"]))

    print("== (c) the §8 leg runs with the fee UNDECLARED ==")
    try:
        PNL.declared_fee()
        fee_state = "declared"
    except PNL.PnLRefused as exc:
        fee_state = _refusal_name(exc)
    ck("the fee is undeclared right now", fee_state != "declared", fee_state)
    ck("  and §8 still produced a verdict for both candidates",
       set(s8["candidates"]) == {"C1", "C2"}
       and all("passes" in v["verdict"] for v in s8["candidates"].values()),
       "the predictive clock is NOT blocked by the economic gap")

    print("== the §8 machinery, end to end ==")
    ck("Holm ran at m=2 over the two candidates",
       all(r["m"] == 2 for r in s8["holm"].values()))
    ck("the exact test enumerated 2^10 = 1024 assignments",
       s8["candidates"]["C1"]["exact_sign_test"]["n_assignments"] == 1024)
    ck("the four conditions are a CONJUNCTION the code evaluates",
       set(s8["candidates"]["C1"]["verdict"]["conditions"]) == {
           "1_holm_corrected_p_below_alpha",
           "2_mean_and_median_increment_positive",
           "3_coverage_gate_passes", "4_all_predicates_pass"})
    ck("the futility rung is computed in advance, with best-attainable p",
       "best_attainable_p_if_every_remaining_day_is_positive"
       in s8["candidates"]["C1"]["futility"])
    ck("a verdict ARTIFACT is on disk",
       (out / "rehearsal_section8_verdict.json").is_file(),
       str(out / "rehearsal_section8_verdict.json"))
    art = json.loads((out / "rehearsal_section8_verdict.json").read_text())
    ck("  and it carries the sign convention, the p-values and the holm "
       "rows", art["sign_convention"] == INCREMENT_SIGN
       and set(art["p_values"]) == {"C1", "C2"})

    print("== the §9 leg, end to end ==")
    params = SENS.sensitivity_parameters(SENS._decl(str(out / "decl")))
    s9 = economic_path(days, declared_fee=dict(SENS.ZERO), params=params,
                       outdir=out)
    ck("both gates and the sensitivity ran on the same ten days",
       s9["primary"]["n_days"] == 10
       and s9["worst_case"]["fills_sha256"] == s9["primary"]["fills_sha256"])
    ck("  and the §9 artifact is on disk with its conclusion",
       (out / "rehearsal_section9_verdict.json").is_file()
       and s9["conclusion"]["status"] in (SENS.IMMATERIAL,
                                          SENS.NOT_SETTLEABLE),
       s9["conclusion"]["status"])
    try:
        economic_path(days, params=params)
        ck("the §9 leg on the REAL declarations stops on the fee", False)
    except PNL.PnLRefused as exc:
        ck("the §9 leg on the REAL declarations stops on the fee",
           PNL.FEE_NOT_DECLARED in str(exc), _refusal_name(exc))

    print("== the inventory is COMPUTED, and it can find a landmine ==")
    inv = inventory(out)
    ck("every refusal name was SCANNED from source, not hand-listed",
       inv["n_refusals_scanned"] >= 25, f"{inv['n_refusals_scanned']} names")
    ck("the ordinary cases (a) (b) (c) all COMPLETED",
       len(inv["ordinary_cases_that_completed"]) >= 3,
       f"{len(inv['ordinary_cases_that_completed'])} completed")
    ck("THE POSITIVE CONTROL: the inventory reports the landmine it "
       "found rather than an empty list",
       any(r["refusal"] == PNL.NO_SETTLEMENT for r in inv["landmines"]),
       f"landmines: {[r['refusal'] for r in inv['landmines']]}")
    ck("  and it separates a legitimate population stop from a fault",
       any(r["refusal"] == PRED.BAND_NOT_MET
           for r in inv["legitimate_stops"]),
       f"stops: {[r['refusal'] for r in inv['legitimate_stops']]}")
    ck("undriven refusals are REPORTED as undriven, never as safe",
       all(r["fires_on_a_normal_day"] == "NOT DRIVEN HERE"
           for r in inv["table"] if not r["driven_here"]),
       f"{inv['n_not_driven']} not driven")
    ck("the inventory artifact is on disk",
       (out / "rehearsal_refusal_inventory.json").is_file())
    ck("EVERY scanned refusal was DRIVEN -- none left unexercised",
       inv["n_not_driven"] == 0,
       f"{inv['n_driven']} driven of {inv['n_refusals_scanned']} scanned")

    print("== no refusal on this path is anonymous ==")
    unnamed = {m: scan_unnamed_refusals(m) for m in PATH_MODULES}
    ck("the path modules raise NO unnamed refusal",
       not any(unnamed.values()),
       str({k: len(v) for k, v in unnamed.items()}))
    ck("  and the scan can SEE one -- positive control",
       len(scan_unnamed_refusals(src="class XRefused(ValueError):\n"
                                     "    pass\n"
                                     "def f():\n"
                                     "    raise XRefused('something "
                                     "went wrong')\n")) == 1)
    ck("  and it does not flag a FORWARDED refusal, which is named by "
       "the one it forwards",
       len(scan_unnamed_refusals(src="class XRefused(ValueError):\n"
                                     "    pass\n"
                                     "def f(exc):\n"
                                     "    raise XRefused(f'{exc} more')\n"
                                     )) == 0)

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    if "--rehearse" in argv or "--inventory" in argv:
        out = None
        if "--out" in argv:
            out = argv[argv.index("--out") + 1]
        got = rehearse(out)
        inv = got["inventory"]
        print(f"artifacts: {got['outdir']}")
        print(f"§9 on a fixture fee: {got['section9_state']}")
        print(f"\n{'REFUSAL':<52} {'NORMAL DAY?':<34} TRIGGER")
        for r in inv["table"]:
            print(f"{r['refusal']:<52} {r['fires_on_a_normal_day']:<34} "
                  f"{r['trigger'][:60]}")
        print(f"\nscanned {inv['n_refusals_scanned']}, driven "
              f"{inv['n_driven']}, not driven {inv['n_not_driven']}, "
              f"landmines {len(inv['landmines'])}")
        return 0
    print(json.dumps({"protocol": PROTOCOL,
                      "sign_convention": INCREMENT_SIGN}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
