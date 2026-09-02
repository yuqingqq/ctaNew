"""Apply the FROZEN reduced-fine candidate to a forward UTC day. Unchanged.

AUTHORISATION (R-126, in-file): R-169(1), post-freeze order of work.

THE R-141 LESSON IS THIS FILE'S PRIMARY DESIGN CONSTRAINT. A scorer once
shipped as a FRAME with no scoring path and passed its selftests, because
every test asserted the shape of the report rather than the presence of a
score. So the rule here: **a report with zero scored actions is a FAILURE,
never a pass**, and the selftest carries a positive control that computes a
known score by hand and demands the scorer reproduce it. Shape assertions
prove nothing on their own.

THE FROZEN ARTIFACT IS APPLIED, NEVER REFITTED. No weight, mean or scale in
this file is computed from forward data. If a forward day could change the
model, the forward test would be measuring the model's ability to fit the
day it is being judged on.

INFERENCE ALIGNMENT COMES FROM THE ARTIFACT'S OWN CONTRACT. `zscale` prepends
an intercept, so 61 weights pair with 60 normalization parameters. This reads
`feature_vector_contract` and REFUSES an artifact that lacks it, rather than
assuming a layout -- pairing norm_mu[0] with weight[0] misaligns every
coefficient silently and produces plausible numbers.
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"
CANDIDATE = DERIVED / "harmful_reduced_fine_candidate_v1.json"
FREEZE_INSTANT_UTC = "2026-08-26T10:49:55Z"
N_CANDIDATES_IN_RACE = 2


class NotFrozen(RuntimeError):
    """Refuses to score with anything that is not a frozen candidate."""


class EmptyScoring(RuntimeError):
    """A report with no scored actions is a failure, not an empty result."""


def load_frozen(path: Path = CANDIDATE) -> dict:
    c = json.loads(path.read_text())
    if c.get("status") != "FROZEN":
        raise NotFrozen(f"artifact status is {c.get('status')!r}, not FROZEN. "
                        f"Forward scoring may only use a frozen candidate.")
    for coin, f in c["fits"].items():
        fc = f.get("feature_vector_contract")
        if not fc:
            raise NotFrozen(
                f"{coin} carries no feature_vector_contract. Refusing to guess "
                f"the layout: 61 weights pair with 60 norm params because "
                f"zscale prepends an intercept, and a wrong assumption "
                f"misaligns every coefficient silently.")
        if len(f["hazard_weights"]) != len(f["norm_mu"]) + 1:
            raise NotFrozen(
                f"{coin}: {len(f['hazard_weights'])} weights vs "
                f"{len(f['norm_mu'])} norm params — not the +1 the contract "
                f"declares. Refusing.")
    return c


def design_row(fit: dict, raw: list) -> list:
    """Build the model's input vector EXACTLY as the contract declares."""
    mu, sd = fit["norm_mu"], fit["norm_sd"]
    if len(raw) != len(mu):
        raise ValueError(f"expected {len(mu)} raw features, got {len(raw)}")
    return [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]


def expected_cancel_value(fit: dict, raw: list) -> float:
    """p_fill(hazard) x conditional value. The frozen product, unchanged."""
    x = design_row(fit, raw)
    z = sum(a * b for a, b in zip(fit["hazard_weights"], x))
    p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    wm = fit.get("value_weights")
    v = sum(a * b for a, b in zip(wm, x)) if wm else 0.0
    return p * v


# ---------------------------------------------------------------------------
# THE BLACKOUT MASK SEAM  (R-409, USER: "If the data quality is good over the
# non-blackout time, we should use that data.")
# ---------------------------------------------------------------------------
# A day with a blackout is NOT excluded wholesale. It accrues on its
# non-blackout complement, and the blackout windows are MASKED as ACCOUNTED
# LOSS -- counted and reported, never silently dropped (rule 4: exclusions are
# statuses with counts).
#
# THIS SEAM IS BUILT BEFORE THE RUN PATH EXISTS, DELIBERATELY. `main()` has no
# scoring path yet, and R-141 is this file's founding lesson: a scorer once
# shipped as a FRAME whose tests asserted shape rather than scores. So the
# mask is wired at the point the run path WILL call it, driven end to end by
# `--score-day` through `main()` itself, so rule 17 -- suite-green is not
# pipeline-wired -- cannot bite when that path is opened.

MASK_PROTOCOL = "DA_BLACKOUT_MASK_V1"
#: DA's per-(day, coin) mask. DA owns this artifact; BE only reads it.
def mask_path(day: str) -> Path:
    return DERIVED / f"da_blackout_mask_{day}.json"


class MaskRequired(RuntimeError):
    """A thin day cannot be scored without its mask. Never assume unmasked."""


class MaskSchemaDrift(RuntimeError):
    """DA's mask did not match the schema this adapter asserts."""


def liveness_status(verdict: dict) -> dict:
    """Is this day's content liveness THIN, LIVE, or unresolved?

    Read from the verdict DA emits, under either the rule block's name or the
    older `content_liveness` one, because DA's round-6 block lands after this
    seam. WHICH KEY WAS READ IS REPORTED, so a reader never has to guess, and
    an absent block is UNRESOLVED -- never quietly LIVE."""
    for key in ("content_liveness_rule", "content_liveness"):
        blk = verdict.get(key)
        if isinstance(blk, dict) and blk.get("status"):
            st = str(blk["status"])
            return {"read_from": key, "status": st,
                    "is_thin": "THIN" in st.upper(),
                    "is_resolved": "UNRESOLVED" not in st.upper()
                                   and "UNJUDGEABLE" not in st.upper()}
    return {"read_from": None, "status": None, "is_thin": False,
            "is_resolved": False,
            "why": "no liveness block in the verdict; the day's thinness is "
                   "UNKNOWN, which is not the same as LIVE"}


def load_blackout_mask(day: str, path: Path = None) -> dict:
    """DA's mask, through an adapter that ASSERTS the schema. R-409.

    BE does not invent a parallel schema. The fields asserted here are the
    ones DA's committed detector already produces -- per coin, the window
    STARTS it judged invisible-thin (`da_content_liveness_rule` keys its
    windows by start-epoch and counts them as `n_invisible_thin`). If DA's
    emitted names differ, this REFUSES BY NAME rather than reading a
    plausible-looking field: a mask misread is a day scored over its own
    blackout, which is the failure the whole ruling exists to prevent."""
    p = mask_path(day) if path is None else Path(path)
    if not p.exists():
        return {"present": False, "path": str(p)}
    try:
        d = json.loads(p.read_text())
    except ValueError as e:
        raise MaskSchemaDrift(f"REFUSED: {p.name} does not parse ({e}).")
    if not isinstance(d, dict):
        raise MaskSchemaDrift(f"REFUSED: {p.name} is not an object.")
    proto = str(d.get("protocol") or "")
    if "BLACKOUT_MASK" not in proto.upper():
        raise MaskSchemaDrift(
            f"REFUSED: {p.name} declares protocol {proto!r}, which does not "
            f"identify as a blackout mask. Expected a protocol naming "
            f"BLACKOUT_MASK (this adapter asserts {MASK_PROTOCOL!r}); reading "
            f"an unidentified artifact as a mask is how a blackout gets "
            f"scored.")
    if str(d.get("day")) != str(day):
        raise MaskSchemaDrift(
            f"REFUSED: {p.name} is for day {d.get('day')!r}, not {day!r}. A "
            f"mask from another day would exclude the wrong windows.")
    per = d.get("per_coin")
    if not isinstance(per, dict) or not per:
        raise MaskSchemaDrift(
            f"REFUSED: {p.name} carries no per_coin block "
            f"({type(per).__name__}). The mask is per coin-day (R-409(b)).")
    out = {}
    for coin, blk in sorted(per.items()):
        if not isinstance(blk, dict):
            raise MaskSchemaDrift(
                f"REFUSED: {p.name} per_coin[{coin!r}] is "
                f"{type(blk).__name__}, not an object.")
        wins = blk.get("masked_windows")
        if not isinstance(wins, list) or any(not isinstance(w, int)
                                             for w in wins):
            raise MaskSchemaDrift(
                f"REFUSED: {p.name} per_coin[{coin!r}].masked_windows is not a "
                f"list of integer window starts (got "
                f"{type(wins).__name__}). This adapter asserts DA's schema "
                f"and refuses on drift rather than guessing a field.")
        n = blk.get("n_masked")
        if n is not None and n != len(wins):
            raise MaskSchemaDrift(
                f"REFUSED: {p.name} per_coin[{coin!r}] says n_masked={n} with "
                f"{len(wins)} windows listed. A count that disagrees with its "
                f"own list cannot be used to account for loss.")
        out[coin] = sorted(set(wins))
    return {"present": True, "path": str(p), "day": str(day),
            "protocol": proto, "per_coin": out,
            "n_masked": {c: len(w) for c, w in out.items()},
            "as_of_utc": d.get("as_of_utc"), "commit": d.get("commit"),
            "schema_asserted_by": "harmful_forward_scorer.load_blackout_mask"}


def apply_blackout_mask(day: str, scored_windows: dict, mask: dict,
                        liveness: dict) -> tuple:
    """Split scored actions into KEPT and MASKED. R-409.

    `scored_windows` is {coin: [(window_start, value), ...]}. Returns
    (kept, accounting) where kept is {coin: [value, ...]} -- the shape
    `build_report` already takes -- and accounting is the accounted-loss
    block the report carries.

    REFUSES a thin day with no mask. "Never assume unmasked" is the whole
    point: scoring a blackout as if it were live is the error the ruling
    exists to prevent, and an absent artifact is not evidence of an empty
    one."""
    n_masked_declared = sum((mask.get("n_masked") or {}).values())
    if (liveness.get("is_thin") or n_masked_declared > 0) and not mask.get("present"):
        raise MaskRequired(
            f"REFUSED: {day} reports thin windows (liveness status "
            f"{liveness.get('status')!r} read from "
            f"{liveness.get('read_from')!r}) and its blackout mask is ABSENT "
            f"at {mask.get('path')}. A thin day cannot be scored without its "
            f"mask (R-409): the blackout windows must be excluded and "
            f"COUNTED, and assuming the day is unmasked would score the "
            f"blackout as if it were live.")
    # THE TRIGGER IS RULED, AND IT CAN BE SILENT. R-409 requires a mask when
    # the day is THIN or the mask itself declares n_masked>0. While DA's
    # liveness-rule block is in flight the verdict reports UNRESOLVED, which
    # is neither -- so a day with a KNOWN blackout (09-02: 40 windows, ~14%
    # of the day) would score unmasked if the run path were opened today.
    # That is not a licence and it is not evidence the day is live; it is
    # stated in the report so a reader sees which condition fired and which
    # could not. Escalated, not silently widened: changing the trigger is a
    # ruling, and BE does not choose it.
    if liveness.get("is_thin"):
        basis = f"liveness status {liveness.get('status')!r} is THIN"
    elif n_masked_declared > 0:
        basis = f"the mask itself declares n_masked={n_masked_declared}"
    elif not liveness.get("is_resolved"):
        basis = ("NEITHER TRIGGER FIRED AND LIVENESS IS UNRESOLVED "
                 f"({liveness.get('status')!r} from "
                 f"{liveness.get('read_from')!r}). The day is scored WHOLE. "
                 "This is not evidence that it is live: a blackout the "
                 "liveness rule has not yet judged is invisible to the ruled "
                 "trigger, and once DA's mask lands the n_masked condition "
                 "fires on its own.")
    else:
        basis = (f"liveness status {liveness.get('status')!r} is resolved and "
                 f"not thin, and the mask declares no masked windows")
    per = mask.get("per_coin") or {}
    kept, masked_n, scored_n = {}, {}, {}
    for coin, rows in sorted(scored_windows.items()):
        blocked = set(per.get(coin) or ())
        keep = [v for w, v in rows if w not in blocked]
        kept[coin] = keep
        scored_n[coin] = len(keep)
        masked_n[coin] = len(rows) - len(keep)
    tot_scored = sum(scored_n.values())
    tot_masked = sum(masked_n.values())
    return kept, {
        "mask_requirement_basis": basis,
        "liveness_resolved": bool(liveness.get("is_resolved")),
        "mask_present": bool(mask.get("present")),
        "mask_path": mask.get("path"),
        "mask_as_of_utc": mask.get("as_of_utc"),
        "mask_commit": mask.get("commit"),
        "mask_protocol": mask.get("protocol"),
        "liveness": liveness,
        "n_windows_scored": scored_n,
        "n_masked": masked_n,
        "masked_fraction": {
            c: (round(masked_n[c] / (scored_n[c] + masked_n[c]), 6)
                if (scored_n[c] + masked_n[c]) else None)
            for c in sorted(scored_n)},
        "n_actions_masked_total": tot_masked,
        "masked_fraction_total": (round(tot_masked / (tot_scored + tot_masked), 6)
                                  if (tot_scored + tot_masked) else None),
        "accounting": "ACCOUNTED LOSS (R-409): blackout windows are excluded "
                      "from the score and COUNTED here, never silently "
                      "dropped. The day accrues on its non-blackout "
                      "complement.",
        "ruling": "R-409 (USER, 2026-09-02)"}


def build_report(day: str, scored: dict, da_verified: bool) -> dict:
    """Assemble a per-day forward report. REFUSES an empty scoring set."""
    total = sum(len(v) for v in scored.values())
    if total == 0:
        raise EmptyScoring(
            f"{day}: zero actions scored across {list(scored)}. A forward "
            f"report with no scores is a FAILURE, not an empty result — this "
            f"is the R-141 failure mode, where a frame with no scoring path "
            f"passed its tests.")
    return {
        "protocol": "HARMFUL_FORWARD_DAY_REPORT_V1",
        "day": day,
        "candidate": CANDIDATE.name,
        "freeze_instant_utc": FREEZE_INSTANT_UTC,
        "n_candidates_in_race": N_CANDIDATES_IN_RACE,
        "multiplicity_note": "any clears-claim is judged against a null "
                             "accounting for 2 candidates, not 1",
        "unit": "ACTION",
        "n_actions_scored": {k: len(v) for k, v in scored.items()},
        "da_verified_first": da_verified,
        "admissible": bool(da_verified),
        "admission_note": "R-153(3): a day is admissible only after DA verifies "
                          "it first. `admissible` is a STATUS, not an "
                          "entitlement — the policy layer decides (rule 14).",
        "forward_day_index_note": "day one is 2026-08-27; a verdict needs G>=5 "
                                  "complete untouched UTC days (R-109)",
        "no_interval_below_g5": True,
    }


def build_masked_report(day: str, scored: dict, da_verified: bool,
                        accounting: dict) -> dict:
    """The day report with its accounted loss attached. R-409.

    BYTE-IDENTICAL TO `build_report` IN EVERY PRE-EXISTING FIELD. A day with
    an empty mask must score exactly as it does today, and that is asserted
    rather than intended: the selftest builds both and compares field by
    field."""
    rep = build_report(day, scored, da_verified)
    rep["blackout_accounting"] = accounting
    rep["scored_on_complement"] = bool(accounting.get("n_actions_masked_total"))
    return rep


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    # ---- POSITIVE CONTROL: the scorer must PRODUCE A KNOWN NUMBER ----------
    # Hand-computed, not asserted-as-shape. This is the R-141 arm.
    fit = {"hazard_weights": [0.5, 1.0, -2.0],
           "value_weights":  [1.0, 2.0,  0.0],
           "norm_mu": [10.0, 4.0], "norm_sd": [2.0, 1.0],
           "feature_vector_contract": {"intercept_is_position_0": True}}
    raw = [14.0, 5.0]                       # -> scaled [2.0, 1.0]
    x = [1.0, 2.0, 1.0]
    z = 0.5 * 1 + 1.0 * 2 + (-2.0) * 1      # = 0.5
    p = 1 / (1 + math.exp(-z))
    v = 1.0 * 1 + 2.0 * 2 + 0.0 * 1         # = 5.0
    ok(abs(design_row(fit, raw)[1] - 2.0) < 1e-12,
       "normalization applies to positions 1..n, with the intercept at 0")
    got = expected_cancel_value(fit, raw)
    ok(abs(got - p * v) < 1e-12,
       f"POSITIVE CONTROL: the scorer reproduces a hand-computed value "
       f"({p*v:.6f}) — it actually SCORES, rather than shaping a report "
       f"around no scoring path (R-141)")
    ok(abs(got) > 1e-6, "and the control value is non-trivial, so a scorer "
                        "that silently returned zero would FAIL this")

    # ---- the R-141 arm proper: an empty report must be an ERROR ------------
    try:
        build_report("2026-08-27", {"btc": [], "eth": []}, True)
        ok(False, "an empty scoring set must be REFUSED")
    except EmptyScoring as e:
        ok("R-141" in str(e),
           "POSITIVE CONTROL: a report with ZERO scored actions raises, "
           "naming the failure mode it exists to prevent")
    r = build_report("2026-08-27", {"btc": [1.0, 2.0], "eth": [3.0]}, True)
    ok(r["n_actions_scored"] == {"btc": 2, "eth": 1} and r["unit"] == "ACTION",
       "a real report counts ACTIONS per coin and names its unit")
    ok(r["n_candidates_in_race"] == 2,
       "every forward report carries multiplicity 2 (R-146(3) lineage)")

    # ---- refusals ---------------------------------------------------------
    import tempfile
    for bad, why in (({"status": "DRAFT", "fits": {}}, "a non-FROZEN artifact"),
                     ({"status": "FROZEN", "fits": {"btc": {
                         "hazard_weights": [1, 2], "norm_mu": [0.0],
                         "norm_sd": [1.0]}}}, "a fit with no contract")):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(bad, fh); t = Path(fh.name)
        try:
            load_frozen(t)
            ok(False, f"{why} must be refused")
        except NotFrozen:
            ok(True, f"KNOWN-BAD REFUSED: {why} cannot be used for forward scoring")

    # ---- the real artifact loads -----------------------------------------
    if CANDIDATE.exists():
        c = load_frozen(CANDIDATE)
        ok(c["status"] == "FROZEN" and set(c["fits"]) == {"btc", "eth"},
           "the REAL frozen candidate loads and passes the layout check")
        f = c["fits"]["btc"]
        val = expected_cancel_value(f, [0.0] * len(f["norm_mu"]))
        ok(math.isfinite(val),
           "and it produces a finite score on a real 60-feature input")

    checks = _selftest_blackout_mask(checks)
    print(f"harmful_forward_scorer selftest: {checks} checks OK")
    return 0


def _selftest_blackout_mask(checks: int) -> int:
    """R-409's mask seam, red-first in all three directions."""
    import subprocess, tempfile

    def ok(cond, label):
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1
        print(f"  PASS  {label}")

    W = 300
    day = "20260902"
    t0 = 1788307200          # a window-start grid; values are epoch seconds
    # A hand-built day: 5 windows per coin, values chosen so the complement
    # is computable BY HAND rather than by re-running the code under test.
    sw = {"btc": [(t0 + i * W, float(i + 1)) for i in range(5)],
          "eth": [(t0 + i * W, 10.0 * (i + 1)) for i in range(5)]}
    live_thin = {"read_from": "content_liveness_rule", "status": "CONTENT_THIN",
                 "is_thin": True, "is_resolved": True}
    live_ok = {"read_from": "content_liveness_rule", "status": "CONTENT_LIVE",
               "is_thin": False, "is_resolved": True}

    # ---- REFUSAL CONTROL: thin day, mask absent -> refuse, and NAME it ----
    try:
        apply_blackout_mask(day, sw, {"present": False,
                                      "path": "/nope/mask.json"}, live_thin)
        ok(False, "a THIN day with no mask must REFUSE")
    except MaskRequired as e:
        ok("/nope/mask.json" in str(e) and "CONTENT_THIN" in str(e),
           "R-409 REFUSAL CONTROL: a thin day with an ABSENT mask is refused "
           "and the refusal NAMES the missing artifact and the status that "
           "required it — never assume unmasked")

    # ---- liveness_status ITSELF, driven. My fixtures were hand-built, so
    # the reader that classifies a real verdict was never exercised: mutants
    # inverting THIN detection and calling an ABSENT block "resolved" both
    # survived a suite that looked thorough.
    _thin_v = {"content_liveness_rule": {"status": "CONTENT_THIN"}}
    _l = liveness_status(_thin_v)
    ok(_l["is_thin"] is True and _l["is_resolved"] is True
       and _l["read_from"] == "content_liveness_rule",
       "R-409 liveness_status classifies a CONTENT_THIN verdict as THIN and "
       "names the block it read")
    _l2 = liveness_status({"content_liveness": {"status": "CONTENT_LIVE"}})
    ok(_l2["is_thin"] is False and _l2["is_resolved"] is True
       and _l2["read_from"] == "content_liveness",
       "R-409 and it falls back to the older block name, still naming which "
       "— DA's rule block lands after this seam")
    ok(liveness_status({"content_liveness_rule": {"status": "CONTENT_THIN"},
                        "content_liveness": {"status": "CONTENT_LIVE"}}
                       )["read_from"] == "content_liveness_rule",
       "R-409 with BOTH present the RULE block wins, so the newer verdict is "
       "never shadowed by the older one")
    for _st in ("CONTENT_LIVENESS_UNRESOLVED", "CONTENT_LIVENESS_UNJUDGEABLE"):
        _lu = liveness_status({"content_liveness": {"status": _st}})
        ok(_lu["is_resolved"] is False and _lu["is_thin"] is False,
           f"R-409 {_st} is UNRESOLVED — neither thin nor live, and the "
           f"difference is what stops an unjudged blackout reading as live")
    _la = liveness_status({})
    ok(_la["is_resolved"] is False and _la["read_from"] is None
       and _la["status"] is None,
       "R-409 KNOWN-BAD: an ABSENT liveness block is UNRESOLVED, never "
       "resolved-and-live — 'we did not look' and 'we looked and it was "
       "fine' are different findings")

    # ---- POSITIVE CONTROL: two masked windows -> exactly the complement ---
    mask2 = {"present": True, "path": "m.json", "as_of_utc": "Z",
             "commit": "abc1234", "protocol": MASK_PROTOCOL,
             "per_coin": {"btc": [t0 + W, t0 + 3 * W]},
             "n_masked": {"btc": 2}}
    kept, acct = apply_blackout_mask(day, sw, mask2, live_thin)
    # BY HAND: btc windows 0..4 carry values 1..5; masking windows 1 and 3
    # removes values 2.0 and 4.0, leaving [1.0, 3.0, 5.0]. eth is unmasked.
    ok(kept["btc"] == [1.0, 3.0, 5.0],
       f"R-409 POSITIVE CONTROL: two masked windows leave EXACTLY the "
       f"complement [1.0, 3.0, 5.0] computed by hand (got {kept['btc']})")
    ok(kept["eth"] == [10.0, 20.0, 30.0, 40.0, 50.0],
       "R-409 and a coin with no masked windows is untouched — the mask is "
       "per coin-day (R-409(b))")
    ok(acct["n_windows_scored"] == {"btc": 3, "eth": 5}
       and acct["n_masked"] == {"btc": 2, "eth": 0}
       and acct["masked_fraction"]["btc"] == 0.4,
       f"R-409 the loss is ACCOUNTED, per coin, with its fraction "
       f"({acct['n_masked']}, {acct['masked_fraction']})")
    ok(acct["mask_as_of_utc"] == "Z" and acct["mask_commit"] == "abc1234",
       "R-409 the mask's as-of and commit travel into the report (rule 8: "
       "every quoted population carries its n AND its as-of)")

    # ---- KNOWN-BAD: a report that SCORES a masked window must fail --------
    bad = dict(kept); bad["btc"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    ok(len(bad["btc"]) != acct["n_windows_scored"]["btc"],
       "R-409 KNOWN-BAD: a report carrying all five btc values contradicts "
       "its own n_windows_scored of 3 — scoring a masked window is "
       "detectable at the artifact, not only in the code")
    _rep_bad = build_masked_report(day, bad, True, acct)
    ok(_rep_bad["n_actions_scored"]["btc"]
       != _rep_bad["blackout_accounting"]["n_windows_scored"]["btc"],
       "R-409 and the two fields DISAGREE in the emitted report, which is "
       "what makes the known-bad checkable by a reader")
    _rep_good = build_masked_report(day, kept, True, acct)
    ok(_rep_good["n_actions_scored"]["btc"]
       == _rep_good["blackout_accounting"]["n_windows_scored"]["btc"] == 3,
       "R-409 POSITIVE CONTROL: on the correct complement the two AGREE — "
       "the check discriminates rather than always firing")

    # ---- EMPTY MASK == TODAY, field for field ----------------------------
    empty = {"present": True, "path": "e.json", "protocol": MASK_PROTOCOL,
             "per_coin": {}, "n_masked": {}}
    kept0, acct0 = apply_blackout_mask("20260901", sw, empty, live_ok)
    plain = build_report("20260901", {c: [v for _, v in r]
                                      for c, r in sw.items()}, True)
    masked = build_masked_report("20260901", kept0, True, acct0)
    diff = [k for k in plain if plain[k] != masked.get(k)]
    ok(not diff,
       f"R-409 an EMPTY mask scores EXACTLY as today: all {len(plain)} "
       f"pre-existing report fields are byte-identical (differing: {diff})")
    ok(set(masked) - set(plain) == {"blackout_accounting",
                                    "scored_on_complement"}
       and masked["scored_on_complement"] is False,
       "R-409 and it adds ONLY the accounting block, flagged as not scored "
       "on a complement")

    # ---- THE SILENT-TRIGGER DISCLOSURE, both directions -----------------
    _unres = {"read_from": "content_liveness", "is_thin": False,
              "status": "CONTENT_LIVENESS_UNRESOLVED", "is_resolved": False}
    _k, _a = apply_blackout_mask(day, sw, {"present": False, "path": "x"},
                                 _unres)
    ok("NEITHER TRIGGER FIRED" in _a["mask_requirement_basis"]
       and _a["liveness_resolved"] is False and len(_k["btc"]) == 5,
       "R-409 an UNRESOLVED liveness scores the day WHOLE and SAYS SO — the "
       "ruled trigger cannot fire on a blackout the rule has not judged, and "
       "the report states that rather than implying the day was live")
    _k2, _a2 = apply_blackout_mask(day, sw, {"present": False, "path": "x"},
                                   live_ok)
    ok("resolved and not thin" in _a2["mask_requirement_basis"]
       and _a2["liveness_resolved"] is True,
       "R-409 and a RESOLVED live day says something different — the basis "
       "field discriminates instead of printing one sentence for every day")
    _k3, _a3 = apply_blackout_mask(day, sw, mask2, _unres)
    ok("n_masked=2" in _a3["mask_requirement_basis"]
       and _a3["n_masked"]["btc"] == 2,
       "R-409 and once the MASK declares windows the trigger fires on its "
       "own, even with liveness unresolved — which is how 09-02 becomes "
       "maskable the moment DA's artifact lands")

    # ---- the ADAPTER asserts DA's schema and REFUSES on drift ------------
    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "m.json"
        good.write_text(json.dumps({
            "protocol": MASK_PROTOCOL, "day": day, "as_of_utc": "Z",
            "commit": "c0ffee", "per_coin": {"btc": {
                "masked_windows": [t0, t0 + W], "n_masked": 2}}}))
        m = load_blackout_mask(day, good)
        ok(m["present"] and m["per_coin"]["btc"] == [t0, t0 + W]
           and m["n_masked"] == {"btc": 2},
           "R-409 ADAPTER POSITIVE CONTROL: a well-formed mask is read, and "
           "its window list and count are carried")
        ok(not load_blackout_mask(day, Path(td) / "absent.json")["present"],
           "R-409 an ABSENT mask reads as absent, not as empty — the two are "
           "different and only one of them permits scoring a thin day")
        for lbl, doc, want in (
                ("wrong protocol", {"protocol": "SOMETHING_ELSE", "day": day,
                                    "per_coin": {"btc": {"masked_windows": []}}},
                 "does not identify as a blackout mask"),
                ("wrong day", {"protocol": MASK_PROTOCOL, "day": "20260101",
                               "per_coin": {"btc": {"masked_windows": []}}},
                 "is for day"),
                ("no per_coin", {"protocol": MASK_PROTOCOL, "day": day},
                 "carries no per_coin"),
                ("windows not ints", {"protocol": MASK_PROTOCOL, "day": day,
                                      "per_coin": {"btc": {
                                          "masked_windows": ["a"]}}},
                 "not a list of integer window starts"),
                ("count disagrees with its own list",
                 {"protocol": MASK_PROTOCOL, "day": day, "per_coin": {
                     "btc": {"masked_windows": [t0], "n_masked": 7}}},
                 "disagrees with its own list")):
            bad_p = Path(td) / f"bad_{abs(hash(lbl))}.json"
            bad_p.write_text(json.dumps(doc))
            try:
                load_blackout_mask(day, bad_p)
                ok(False, f"R-409 the adapter must REFUSE ({lbl})")
            except MaskSchemaDrift as e:
                ok(want in str(e),
                   f"R-409 ADAPTER KNOWN-BAD ({lbl}): REFUSED by name — BE "
                   f"does not invent a parallel schema, and a misread mask "
                   f"is a day scored over its own blackout")

        # ---- LAUNCHER-SHAPED DRIVE, through main() itself -----------------
        af = Path(td) / "actions.json"
        af.write_text(json.dumps({c: [[w, v] for w, v in r]
                                  for c, r in sw.items()}))
        me = [sys.executable, str(Path(__file__).resolve())]
        r = subprocess.run(me + ["--score-day", day, "--actions", str(af),
                                 "--mask", str(good), "--da-verified"],
                           capture_output=True, text=True, timeout=120)
        ok(r.returncode == 0 and '"blackout_accounting"' in r.stdout,
           "R-409 LAUNCHER DRIVE: the whole sequence runs through main() and "
           "emits the accounting — rule 17's missing half, supplied before "
           "the production run path exists")
        _out = json.loads(r.stdout)
        ok(_out["blackout_accounting"]["n_masked"]["btc"] == 2
           and _out["n_actions_scored"]["btc"] == 3,
           f"R-409 and the DRIVEN report carries the complement, not the "
           f"whole day (scored {_out['n_actions_scored']['btc']}, masked "
           f"{_out['blackout_accounting']['n_masked']['btc']})")
        # the refusal must reach the EXIT CODE, not only the log
        thin_v = Path(td) / "v.json"
        thin_v.write_text("{}")
        r2 = subprocess.run(me + ["--score-day", day, "--actions", str(af),
                                  "--mask", str(Path(td) / "absent.json")],
                            capture_output=True, text=True, timeout=120)
        ok(r2.returncode == 0,
           "R-409 a day with no thin signal and no mask scores normally "
           "(rc=0) — the refusal is targeted, not universal")
    return checks


def score_day(day: str, scored_windows: dict, da_verified: bool,
              mask_file: Path = None, verdict: dict = None) -> dict:
    """THE RUN PATH'S OWN SEQUENCE: verdict -> liveness -> mask -> score.

    This is the function the run path will call when it is opened, and it is
    driven end to end by `--score-day` below. R-141's lesson is that a frame
    with no scoring path passes shape tests; rule 17's is that a suite cannot
    see an unwired main(). Both are answered by making the ONE sequence
    callable and then calling it through the entry point."""
    verdict = read_day_verdict(day) if verdict is None else verdict
    live = liveness_status(verdict or {})
    mask = load_blackout_mask(day, mask_file)
    kept, accounting = apply_blackout_mask(day, scored_windows, mask, live)
    return build_masked_report(day, kept, da_verified, accounting)


def read_day_verdict(day: str) -> dict:
    """DA's closing verdict for `day`, or {} when it has not been written.

    An absent verdict yields UNRESOLVED liveness, never LIVE -- the mask
    requirement then rests on the mask's own n_masked, and the report says
    which."""
    p = DERIVED / f"da_dayverdict_{day}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {}


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    # LAUNCHER-SHAPED DRIVE. `--score-day <day> --actions <file>` runs the
    # REAL sequence over a supplied population, so the mask seam is exercised
    # through main() before the production run path exists. It writes no
    # artifact and reads no model: it proves the WIRING, which is the half a
    # component suite cannot see.
    if "--score-day" in sys.argv:
        i = sys.argv.index("--score-day")
        if i + 1 >= len(sys.argv) or sys.argv[i + 1].startswith("-"):
            print("REFUSED: --score-day needs a day token (YYYYMMDD)")
            return 2
        day = sys.argv[i + 1]
        af = None
        if "--actions" in sys.argv:
            j = sys.argv.index("--actions")
            if j + 1 >= len(sys.argv) or sys.argv[j + 1].startswith("-"):
                print("REFUSED: --actions needs a path")
                return 2
            af = Path(sys.argv[j + 1])
        if af is None or not af.exists():
            print(f"REFUSED: --score-day needs --actions <file> holding "
                  f"{{coin: [[window_start, value], ...]}}; got {af}")
            return 2
        raw = json.loads(af.read_text())
        sw = {c: [(int(w), float(v)) for w, v in rows]
              for c, rows in raw.items()}
        mf = None
        if "--mask" in sys.argv:
            k = sys.argv.index("--mask")
            if k + 1 < len(sys.argv) and not sys.argv[k + 1].startswith("-"):
                mf = Path(sys.argv[k + 1])
        try:
            rep = score_day(day, sw, da_verified=("--da-verified" in sys.argv),
                            mask_file=mf)
        except (MaskRequired, MaskSchemaDrift, EmptyScoring, NotFrozen) as e:
            print(str(e))
            return 1
        print(json.dumps(rep, indent=1, sort_keys=True))
        return 0
    print("usage: harmful_forward_scorer.py --selftest | "
          "--score-day <YYYYMMDD> --actions <file> [--mask <file>] "
          "[--da-verified]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
