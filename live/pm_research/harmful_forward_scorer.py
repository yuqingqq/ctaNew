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

#: DA's committed artifact identifier. THE PRODUCER'S ARTIFACT IS THE
#: CONTRACT (R-412 ruling 1). BE previously asserted a `protocol`/`per_coin`
#: envelope of its own invention and REFUSED DA's real mask -- both suites
#: green, each testing its own side, the exact class R-402 named. The names
#: below are transcribed from
#: `data/pm_5min/derived/da_blackout_mask_20260901.json`.
MASK_ARTIFACT = "da_blackout_mask_v1"
#: The first GOVERNED day, read from the frozen rule rather than repeated
#: here: from this day a mask is REQUIRED, not merely consumed when present
#: (R-410 as amended by R-411).
try:
    from da_content_liveness_rule import EFFECTIVE_FROM_DAY
except Exception:                                    # noqa: BLE001
    EFFECTIVE_FROM_DAY = None
#: DA's per-(day, coin) mask. DA owns this artifact; BE only reads it.
def mask_path(day: str) -> Path:
    return DERIVED / f"da_blackout_mask_{day}.json"


class MaskRequired(RuntimeError):
    """A thin day cannot be scored without its mask. Never assume unmasked."""


class MaskSchemaDrift(RuntimeError):
    """DA's mask did not match the schema this adapter asserts."""


#: R-412(2): where an UNJUDGEABLE day goes. Emitted as TEXT — the scorer
#: states the routing and never decides the disposition (rule 14).
ROUTED_UNJUDGEABLE = ("frozen rule §7 — coordinator exclusion with a stated "
                      "reason")


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
    """DA's mask, through an adapter that ASSERTS DA'S schema. R-412(1).

    THE PRODUCER'S COMMITTED ARTIFACT IS THE CONTRACT. The window STARTS come
    from DA's MASK PRODUCER (`da_blackout_mask`), not from the liveness
    DETECTOR, which emits only a count (`n_invisible_thin`) -- the previous
    docstring named the wrong source, and the envelope BE asserted alongside
    it (`protocol`/`per_coin`) was BE's own invention. It refused DA's real
    mask while both suites stayed green, because each tested its own side.

    Every field below is transcribed from the committed artifact. Drift still
    REFUSES BY NAME: a mask misread is a day scored over its own blackout."""
    p = mask_path(day) if path is None else Path(path)
    if not p.exists():
        return {"present": False, "path": str(p)}
    try:
        d = json.loads(p.read_text())
    except ValueError as e:
        raise MaskSchemaDrift(f"REFUSED: {p.name} does not parse ({e}).")
    if not isinstance(d, dict):
        raise MaskSchemaDrift(f"REFUSED: {p.name} is not an object.")
    art = str(d.get("artifact") or "")
    if art != MASK_ARTIFACT:
        raise MaskSchemaDrift(
            f"REFUSED: {p.name} declares artifact {art!r}, not "
            f"{MASK_ARTIFACT!r}. Reading an unidentified artifact as a mask "
            f"is how a blackout gets scored.")
    if str(d.get("day")) != str(day):
        raise MaskSchemaDrift(
            f"REFUSED: {p.name} is for day {d.get('day')!r}, not {day!r}. A "
            f"mask from another day would exclude the wrong windows.")
    coins = d.get("coins")
    if not isinstance(coins, dict):
        raise MaskSchemaDrift(
            f"REFUSED: {p.name} carries no `coins` block "
            f"({type(coins).__name__}). The mask is per coin-day (R-409(b)).")
    closed = d.get("day_closed_calendar")
    if not isinstance(closed, bool):
        raise MaskSchemaDrift(
            f"REFUSED: {p.name} has no boolean `day_closed_calendar` (got "
            f"{closed!r}). DA's own consumer_note makes this the field that "
            f"separates a final mask from a mid-day diagnostic (RR8-3).")
    out, meta = {}, {}
    for coin, blk in sorted(coins.items()):
        if not isinstance(blk, dict):
            raise MaskSchemaDrift(
                f"REFUSED: {p.name} coins[{coin!r}] is "
                f"{type(blk).__name__}, not an object.")
        wins = blk.get("masked_windows")
        if not isinstance(wins, list) or any(not isinstance(w, int)
                                             for w in wins):
            raise MaskSchemaDrift(
                f"REFUSED: {p.name} coins[{coin!r}].masked_windows is not a "
                f"list of integer window starts (got "
                f"{type(wins).__name__}). This adapter asserts DA's schema "
                f"and refuses on drift rather than guessing a field.")
        n = blk.get("n_masked")
        if n is not None and n != len(wins):
            raise MaskSchemaDrift(
                f"REFUSED: {p.name} coins[{coin!r}] says n_masked={n} with "
                f"{len(wins)} windows listed. A count that disagrees with its "
                f"own list cannot be used to account for loss.")
        out[coin] = sorted(set(wins))
        meta[coin] = {k: blk.get(k) for k in
                      ("n_windows_total", "longest_run_windows",
                       "agrees_with_frozen_L1_numerator", "status")}
    # DA's OWN TOTAL, cross-checked against DA's own per-coin lists. A total
    # that disagrees with the windows it summarises cannot account for loss.
    tot = d.get("total_masked_windows")
    have = sum(len(w) for w in out.values())
    if tot is not None and tot != have:
        raise MaskSchemaDrift(
            f"REFUSED: {p.name} declares total_masked_windows={tot} while its "
            f"own per-coin lists hold {have}.")
    return {"present": True, "path": str(p), "day": str(day),
            "artifact": art, "per_coin": out,
            "n_masked": {c: len(w) for c, w in out.items()},
            "total_masked_windows": tot,
            "day_closed_calendar": closed,
            "as_of_utc": d.get("as_of_utc"), "detector": d.get("detector"),
            "day_status_frozen": d.get("day_status_frozen"),
            "per_coin_meta": meta,
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
    # THE TRIGGER IS THE DAY'S GOVERNED STATUS, NOT A READING OF LIVENESS
    # (R-410, amended by R-411 and R-412). A reading can be ABSENT; a status
    # cannot. PRESENCE CONSUMES -- a mask is honoured for any day, so 09-01's
    # 141 windows are masked at scoring, which is the USER's R-409 principle.
    # GOVERNANCE REQUIRES -- from EFFECTIVE_FROM_DAY a mask must EXIST.
    governed = bool(EFFECTIVE_FROM_DAY) and str(day) >= str(EFFECTIVE_FROM_DAY)
    n_masked_declared = sum((mask.get("n_masked") or {}).values())

    # RR8-3: a MID-DAY mask lists only the windows that exist so far, so
    # scoring off it scores the complement of a day that has not finished.
    # DA's own consumer_note says exactly this.
    if mask.get("present") and mask.get("day_closed_calendar") is not True:
        raise MaskRequired(
            f"REFUSED: {day}'s mask at {mask.get('path')} has "
            f"day_closed_calendar={mask.get('day_closed_calendar')!r}. A "
            f"PARTIAL mask lists only the windows that exist so far, so "
            f"scoring off it scores the complement of a day that has not "
            f"finished (RR8-3). It is a diagnostic, not a final mask.")

    # PERMANENT before TEMPORARY: an UNJUDGEABLE day cannot become judgeable
    # with later data (too few windows, or a zero median), so retrying is not
    # the remedy -- a disposition is. The scorer REFUSES and ROUTES; it never
    # decides the disposition itself (rule 14).
    if str(liveness.get("status") or "").upper().endswith("UNJUDGEABLE"):
        raise MaskRequired(
            f"REFUSED: {day} liveness is {liveness.get('status')!r}, which is "
            f"PERMANENT -- no later data makes it judgeable -- so this is not "
            f"a retry. routed_to: {ROUTED_UNJUDGEABLE!r}. The scorer states "
            f"the routing and does NOT decide the disposition (rule 14).")

    if governed and not mask.get("present"):
        raise MaskRequired(
            f"REFUSED: {day} is GOVERNED (on or after EFFECTIVE_FROM_DAY "
            f"{EFFECTIVE_FROM_DAY}) and its blackout mask is ABSENT at "
            f"{mask.get('path')}. From the governed day every scored day "
            f"REQUIRES a mask artifact, EMPTY PERMITTED (R-410): absence "
            f"means the producer did not run, never that nothing was thin.")

    if governed and not liveness.get("is_resolved"):
        raise MaskRequired(
            f"REFUSED: {day} is GOVERNED and its liveness reads "
            f"{liveness.get('status')!r} from {liveness.get('read_from')!r}, "
            f"which is UNRESOLVED -- the rule block lands with the closing "
            f"verdict. This is TEMPORARY: retry when the verdict lands. No "
            f"disposition is implied and none is decided here.")

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
    if mask.get("present"):
        basis = (f"a mask is PRESENT and is CONSUMED for any day (R-411: "
                 f"presence consumes, governance requires); this day is "
                 f"{'GOVERNED' if governed else 'PRE-GOVERNED'} against "
                 f"EFFECTIVE_FROM_DAY {EFFECTIVE_FROM_DAY!r}")
    elif liveness.get("is_thin"):
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
        "day_governed": governed,
        "effective_from_day": EFFECTIVE_FROM_DAY,
        "mask_day_closed_calendar": mask.get("day_closed_calendar"),
        "mask_total_masked_windows": mask.get("total_masked_windows"),
        "liveness_resolved": bool(liveness.get("is_resolved")),
        "mask_present": bool(mask.get("present")),
        "mask_path": mask.get("path"),
        "mask_as_of_utc": mask.get("as_of_utc"),
        "mask_detector": mask.get("detector"),
        "mask_artifact": mask.get("artifact"),
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
    # PRE-GOVERNED vs GOVERNED are now different regimes (R-410/R-411), so the
    # controls name which they are exercising instead of sharing one token.
    day = "20260901"          # pre-governed: presence consumes, absence is OK
    day_gov = "20260902"      # governed: a mask is REQUIRED, empty permitted
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
           "R-409 REFUSAL CONTROL: a thin PRE-GOVERNED day with an ABSENT "
           "mask is refused and the refusal NAMES the missing artifact and "
           "the status that required it — never assume unmasked")

    # ---- R-410/R-411: GOVERNANCE requires, PRESENCE consumes -------------
    try:
        apply_blackout_mask(day_gov, sw, {"present": False,
                                          "path": "/nope/g.json"}, live_ok)
        ok(False, "a GOVERNED day with no mask must REFUSE")
    except MaskRequired as e:
        ok("GOVERNED" in str(e) and "EMPTY PERMITTED" in str(e)
           and "/nope/g.json" in str(e),
           "R-410 a GOVERNED day REFUSES without a mask even when liveness is "
           "LIVE — absence means the producer did not run, never that nothing "
           "was thin")
    _kg, _ag = apply_blackout_mask(day, sw, {"present": False, "path": "x"},
                                   live_ok)
    ok(len(_kg["btc"]) == 5 and _ag["day_governed"] is False,
       "R-410 POSITIVE CONTROL: a PRE-GOVERNED day with no mask scores whole "
       "— the requirement is the day's governed status, and the report says "
       "which regime applied")

    # ---- R-412(2): UNRESOLVED (retry) vs UNJUDGEABLE (route) -------------
    _unres_g = {"read_from": "content_liveness", "is_thin": False,
                "status": "CONTENT_LIVENESS_UNRESOLVED", "is_resolved": False}
    try:
        apply_blackout_mask(day_gov, sw, {"present": True, "per_coin": {},
                                          "n_masked": {},
                                          "day_closed_calendar": True},
                            _unres_g)
        ok(False, "UNRESOLVED on a governed day must REFUSE")
    except MaskRequired as e:
        ok("TEMPORARY" in str(e) and "retry when the verdict lands" in str(e)
           and "no disposition" in str(e).lower(),
           "R-412(2) UNRESOLVED on a GOVERNED day REFUSES and states the "
           "remedy — it is TEMPORARY, the rule block lands with the closing "
           "verdict")
    _unj = {"read_from": "content_liveness_rule", "is_thin": False,
            "status": "CONTENT_LIVENESS_UNJUDGEABLE", "is_resolved": False}
    try:
        apply_blackout_mask(day, sw, {"present": False, "path": "x"}, _unj)
        ok(False, "UNJUDGEABLE must REFUSE")
    except MaskRequired as e:
        ok(ROUTED_UNJUDGEABLE in str(e) and "PERMANENT" in str(e)
           and "not a retry" in str(e),
           "R-412(2) UNJUDGEABLE REFUSES and emits routed_to as TEXT — it is "
           "PERMANENT, so retrying is not the remedy and the scorer states "
           "the routing without deciding the disposition (rule 14)")
    ok("retry" not in str(ROUTED_UNJUDGEABLE),
       "R-412(2) and the two statuses are DISTINGUISHED: one routes, the "
       "other retries — collapsing them would send a permanently unjudgeable "
       "day round a loop that cannot end")

    # ---- RR8-3: a PARTIAL mask is refused, by name ----------------------
    try:
        apply_blackout_mask(day, sw, {"present": True, "path": "mid.json",
                                      "per_coin": {}, "n_masked": {},
                                      "day_closed_calendar": False}, live_ok)
        ok(False, "a PARTIAL mask must REFUSE")
    except MaskRequired as e:
        ok("day_closed_calendar=False" in str(e) and "RR8-3" in str(e),
           "RR8-3 a mask with day_closed_calendar False is REFUSED by name — "
           "a mid-day mask lists only the windows that exist so far, so "
           "scoring off it scores the complement of an unfinished day")
    _kc, _ac = apply_blackout_mask(day, sw, {"present": True, "path": "f.json",
                                             "per_coin": {}, "n_masked": {},
                                             "day_closed_calendar": True},
                                   live_ok)
    ok(_ac["mask_day_closed_calendar"] is True and len(_kc["btc"]) == 5,
       "RR8-3 POSITIVE CONTROL: a CLOSED-day mask is accepted and the flag is "
       "carried into the report — the check discriminates on the field")

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
             "detector": {"authority": "fixture"}, "artifact": MASK_ARTIFACT,
             "day_closed_calendar": True, "total_masked_windows": 2,
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
    ok(acct["mask_as_of_utc"] == "Z"
       and acct["mask_detector"] == {"authority": "fixture"},
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
    empty = {"present": True, "path": "e.json", "artifact": MASK_ARTIFACT,
             "day_closed_calendar": True, "per_coin": {}, "n_masked": {}}
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
    ok("PRESENT and is CONSUMED" in _a3["mask_requirement_basis"]
       and _a3["n_masked"]["btc"] == 2 and _k3["btc"] == [1.0, 3.0, 5.0],
       "R-411 PRESENCE CONSUMES: a mask is honoured on a PRE-GOVERNED day "
       "with liveness UNRESOLVED — which is exactly how 09-01's 141 windows "
       "are masked at scoring, the USER's R-409 principle")

    # ---- the ADAPTER asserts DA's schema and REFUSES on drift ------------
    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "m.json"
        good.write_text(json.dumps({
            "artifact": MASK_ARTIFACT, "day": day, "as_of_utc": "Z",
            "day_closed_calendar": True, "total_masked_windows": 2,
            "detector": {"authority": "fixture"}, "coins": {"btc": {
                "masked_windows": [t0, t0 + W], "n_masked": 2,
                "n_windows_total": 288, "longest_run_windows": 2,
                "agrees_with_frozen_L1_numerator": True}}}))
        m = load_blackout_mask(day, good)
        ok(m["present"] and m["per_coin"]["btc"] == [t0, t0 + W]
           and m["n_masked"] == {"btc": 2},
           "R-409 ADAPTER POSITIVE CONTROL: a well-formed mask is read, and "
           "its window list and count are carried")
        ok(not load_blackout_mask(day, Path(td) / "absent.json")["present"],
           "R-409 an ABSENT mask reads as absent, not as empty — the two are "
           "different and only one of them permits scoring a thin day")
        for lbl, doc, want in (
                ("wrong artifact id", {"artifact": "SOMETHING_ELSE",
                                       "day": day, "day_closed_calendar": True,
                                       "coins": {"btc": {"masked_windows": []}}},
                 "not 'da_blackout_mask_v1'"),
                ("wrong day", {"artifact": MASK_ARTIFACT, "day": "20260101",
                               "day_closed_calendar": True,
                               "coins": {"btc": {"masked_windows": []}}},
                 "is for day"),
                ("no coins block", {"artifact": MASK_ARTIFACT, "day": day,
                                    "day_closed_calendar": True},
                 "carries no `coins` block"),
                ("no day_closed_calendar", {"artifact": MASK_ARTIFACT,
                                            "day": day,
                                            "coins": {"btc": {
                                                "masked_windows": []}}},
                 "no boolean `day_closed_calendar`"),
                ("windows not ints", {"artifact": MASK_ARTIFACT, "day": day,
                                      "day_closed_calendar": True,
                                      "coins": {"btc": {
                                          "masked_windows": ["a"]}}},
                 "not a list of integer window starts"),
                ("count disagrees with its own list",
                 {"artifact": MASK_ARTIFACT, "day": day,
                  "day_closed_calendar": True, "coins": {
                     "btc": {"masked_windows": [t0], "n_masked": 7}}},
                 "disagrees with its own list"),
                ("total disagrees with the lists it summarises",
                 {"artifact": MASK_ARTIFACT, "day": day,
                  "day_closed_calendar": True, "total_masked_windows": 99,
                  "coins": {"btc": {"masked_windows": [t0], "n_masked": 1}}},
                 "own per-coin lists hold")):
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

            # ---- RR8-1: THE REAL COMMITTED ARTIFACT. A fixture cannot close
        # this finding — the adapter passed its own fixtures while REFUSING
        # DA's real mask, because BE asserted an envelope it had invented.
        # This check reads the artifact DA committed, and it is the only kind
        # that could have failed today.
        real = mask_path("20260901")
        if real.exists():
            da = json.loads(real.read_text())
            rm = load_blackout_mask("20260901")
            ok(rm["present"] and rm["artifact"] == MASK_ARTIFACT,
               f"RR8-1 the adapter READS DA's real committed mask "
               f"({real.name}) — it refused it before, both suites green")
            ok(rm["total_masked_windows"] == da["total_masked_windows"]
               == sum(len(w) for w in rm["per_coin"].values()),
               f"RR8-1 and the window count agrees with DA's own "
               f"total_masked_windows ({da['total_masked_windows']}) — "
               f"asserted against the ARTIFACT, not a literal")
            # per-coin, against DA's numbers rather than transcribed ones
            want = {c: b["n_masked"] for c, b in da["coins"].items()}
            ok(rm["n_masked"] == want,
               f"RR8-1 per-coin masked counts equal DA's artifact exactly "
               f"({want})")
            # and SCORING it must exclude exactly those windows
            sw_real = {c: [(w, float(i)) for i, w in
                           enumerate(sorted(b["masked_windows"])[:3]
                                     + [b["masked_windows"][0] - 300])]
                       for c, b in list(da["coins"].items())[:2]}
            live_unres = liveness_status(read_day_verdict("20260901"))
            kept_r, acct_r = apply_blackout_mask("20260901", sw_real, rm,
                                                 live_unres)
            ok(all(acct_r["n_masked"][c] == 3 for c in sw_real),
               f"RR8-1 scoring 09-01 with its REAL mask excludes exactly the "
               f"windows DA listed (masked {acct_r['n_masked']}) and keeps "
               f"the one that is not in it")
            ok(acct_r["mask_day_closed_calendar"] is True
               and acct_r["mask_as_of_utc"] == da["as_of_utc"],
               "RR8-1 and the real mask's closed-day flag and as-of travel "
               "into the report")

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
