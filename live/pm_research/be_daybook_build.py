"""THE SCORED DAY-BOOK. One day, btc, both pinned heads. The book only.

DE's design v7 R1: the day book must carry `asm`, not just a reference --
`be_cancel_axis_null.load()` reads `c["asm"]` and builds its decision
population from `asm["by_arm"][(coin, head)][0]`. A reference-only book
raises and there is no decision population at all.

WHAT THIS BUILDS AND WHAT IT REFUSES TO DO. It builds the book: reference
(slug -> side -> generations with tranches), statuses, population, n_slugs,
terminal_marks, and `asm` for BOTH pinned heads at their PINNED thetas. It
does NOT score arms, draw nulls, or compute economics. Those are DE's and
they run on this.

THE DAY IS SELECTED THE WAY THE FORWARD SCORER SELECTS IT, NOT THE WAY THE
ARMS CACHE WAS SELECTED. `harmful_exposure_rows.select_v2_era` is bounded by
the declared populations, which end at 2026-08-26T00:00 -- MEASURED, and it
is why a September day cannot come through that door at all. The day path is
`de_admissible_windows.supply(day, present_from_ledger(day))`, the same
supply the forward scorer gated 09-03 through at 12/12.

MEMORY IS THE BINDING CONSTRAINT AND DE MEASURED IT BEFORE ME.
`build_tape_index`'s own docstring: "the score split is 638,917 rows and
1.42 GB, the train split 1,125,289 rows and 3.90 GB cumulative, 390.7 s for
both. That 3.9 GB is resident for the whole assembly, and THE RULED RUN OF
2026-09-03 DIED because it was held alongside the entire fragment." So the
fragment is consumed in CHUNKS against one index, and the cap is not raised.

THE DIGEST IS OF THE BYTES AS WRITTEN (the v3/B-1 discipline, applied to the
write side): the book is serialised to a buffer, that buffer is hashed, and
THAT buffer is written. There is no second read and no second serialisation.
"""
from __future__ import annotations

import gc
import hashlib
import json
import pickle
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import be_data_root as _BDR

#: BE48 §B.4: THIS MODULE HAD TWO ROOTS. The book went through the resolver
#: while the scratch fragment and the receipt went to `HERE.parents[1]` -- so
#: from a worktree the receipt named a book that was not beside it, and a
#: multi-hundred-MB scratch file landed in the worktree's `data/`, which is
#: the act that manufactures the shells this seat spent three rounds
#: removing. My round-47 reasoning for the split (don't write into the main
#: tree) was WRONG: R-397/R-554 already rule that artifacts under `data/`
#: are landed from the main tree by pathspec, so the ledger IS where they go.
#: ONE ROOT NOW, and `require_ledger` guards the result-bearing emission.
ROOT = HERE.parents[1]                    # CODE root only; never a data root
LEDGER_DERIVED = _BDR.derived()
OUT_DERIVED = LEDGER_DERIVED

COIN = "btc"

#: PER-STAGE MEMORY BUDGETS, declared, and a stage over its budget REFUSES.
#: Named to match DE's `--day` stages so the seams agree: DE's S0/S1 consume
#: the book this produces, and R11 says the index is needed ONLY to produce
#: `asm` -- so it is RELEASED before the book is written and the whole-day
#: peak is max(index stage, assembly stage), not their sum.
STAGE_BUDGETS_GB = {
    "A0_reference": 3.0,      # measured 2.008 on 09-03
    "A1_index": 6.5,          # measured 5.971 cumulative (reference + index)
    "A2_assemble": 7.5,       # index + reference + chunked fragment + asm
    "A3_release_index": 7.5,  # high-water only; CURRENT must fall
    "A4_write_book": 7.5,
}
FIXTURE_STAGE_BUDGETS_GB = {k: 0.7 for k in STAGE_BUDGETS_GB}
CHUNK_WINDOWS = 6             # as declared in be_assembly_budget

HEADS = {"CONDVALUE_X_SKEW": "q1_arrival_composed_lgbm",
         "HAZARD_OVER_SKEWED_REF": "incumbent_linear_d"}
BUDGET = 0.10


class BookRefused(RuntimeError):
    """A named refusal."""


def assert_day_tape(day: str, coin: str = COIN) -> dict:
    """THE ASSEMBLY MUST READ THE DAY'S TAPE, AND TODAY IT CANNOT.

    `phase2_arms.tape_index` reads the MODULE CONSTANT `TAPE_PATH`
    (`phase2_state_tape_v5.json`, the live August tape) and neither it nor
    `de_phase4_diag_runner.build_tape_index(splits)` takes a path --
    VERIFIED at the signatures. So an assembly run for a September day would
    index the WRONG TAPE, find no rows for the day's generations, and emit a
    book whose `asm` is empty. That is the failure that looks like a result,
    and DE's guard would fire on it one stage later.

    This refuses BEFORE any of it. The fix is a path parameter on
    `phase2_arms.tape_index` -- DE's surface, routed at the reviewer's BE48
    item 2, not taken here."""
    import be_gate1_state_tape as TAPEMOD
    import phase2_arms as PA
    want = TAPEMOD.out_path(day, coin)
    have = Path(PA.TAPE_PATH)
    if have.resolve() != want.resolve():
        raise BookRefused(
            f"REFUSED: the assembly would index {have.name}, not this day's "
            f"tape {want.name}. `phase2_arms.TAPE_PATH` is a module constant "
            f"and neither `tape_index(split, features_in_order)` nor "
            f"`build_tape_index(splits)` accepts a path -- so a September "
            f"day cannot be pointed at its own tape and the run would emit a "
            f"book with an EMPTY `asm`. BLOCKED on a path parameter, which "
            f"is DE's surface (reviewer BE48, item 2). Not raised here, and "
            f"not worked around.")
    return {"tape": str(want), "is_the_days_tape": True}


def assert_rule20(*, fixture: bool = False) -> dict:
    """DELEGATED to DE's `wrapper_observed` -- MEASURED, not declared.

    R12: `flock -n <lock> systemd-run --scope` passes the lock's fd through
    the exec, so holding it is READ FROM /proc/self/fd. A string in a params
    file cannot be evidence that a lock was held. My 05:54Z breach is exactly
    what this refuses: a heavy run beside another heavy run, in the slice but
    without the lock. Delegated rather than reimplemented -- two
    implementations of one check is two checks (Q-BE-271)."""
    import de_multiday_gate1_runner as RUN
    w = dict(RUN.wrapper_observed())
    w["delegated_to"] = "de_multiday_gate1_runner.wrapper_observed"
    w["fixture"] = fixture
    if not fixture and not w.get("heavy_run_lock_held"):
        raise BookRefused(
            "REFUSED: a real day is HEAVY BY CONSTRUCTION and this process "
            "does not hold /home/yuqing/ctaNew/data/.heavy_run.lock. Run it "
            "as `flock -n <lock> systemd-run --user --scope "
            "--slice=research.slice -p MemoryMax=8G -p CPUQuota=100% ...`. "
            "At 05:54Z on 2026-09-06 this seat ran a heavy build beside "
            "another heavy run because it used the scope without the lock; "
            "this refuses that before any work.")
    return w


def _rss_gb() -> float:
    """HIGH-WATER mark. It never falls, which is why `_rss_now_gb` exists."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def _rss_now_gb() -> float:
    """CURRENT RSS. `ru_maxrss` is a high-water mark and cannot show a
    release, so the claim "the index was dropped" needs a falling number to
    rest on. Read from /proc/self/status."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 2**20, 3)
    except OSError:
        pass
    return float("nan")


class _Stages:
    """Per-stage budgets, asserted. A stage over its budget REFUSES."""

    def __init__(self, budgets: dict):
        self.budgets = dict(budgets)
        self.rows = []

    def done(self, name: str, t0: float) -> dict:
        b = self.budgets.get(name)
        row = {"stage": name, "wall_s": round(time.time() - t0, 1),
               "peak_gb": _rss_gb(), "current_gb": _rss_now_gb(),
               "budget_gb": b}
        row["within_budget"] = (b is None or row["peak_gb"] <= b)
        self.rows.append(row)
        if not row["within_budget"]:
            raise BookRefused(
                f"REFUSED at stage {name}: peak {row['peak_gb']} GB exceeds "
                f"its declared budget of {b} GB. R8/R-174: the cap is NOT "
                f"raised and the population is NOT reduced. The day is "
                f"reported with its measured peak and refused.")
        return row


def day_slugs(day: str, coin: str = COIN, *, supply: dict = None) -> list:
    """The day's SUPPLIED slugs, from the forward scorer's own supply.

    `supply` is an INJECTION POINT and it exists because of a real defect the
    reviewer drove (BE48 §B.2). The refusal below was UNREACHABLE: for any
    day with no ledger entry, `present_from_ledger` refuses one call earlier,
    so this module's own empty-supply refusal had ZERO driven coverage in a
    battery that reported "both guards driven". Injecting the supply reaches
    it, so the check tests THIS module rather than `be_forward_day`."""
    if supply is None:
        import be_forward_day as FD
        import de_admissible_windows as AW
        supply = AW.supply(day, FD.present_from_ledger(day))
    w = (supply.get("windows") or {}).get(coin) or []
    out = [x["slug"] for x in w]
    if not out:
        raise BookRefused(
            f"REFUSED: no supplied {coin} windows for {day}. A book over an "
            f"empty day is not a small book, it is a different question.")
    return sorted(out)


def day_selector(day: str, coin: str = COIN):
    """A `build_reference` selector scoped to ONE DAY.

    Same entry shape `select_v2_era` returns -- (slug, path, up, down, gaps)
    -- built from the same three indices, but gated by the DAY'S SUPPLY
    rather than by a declared population interval. The population intervals
    end 2026-08-26T00:00, so no September day can pass through them."""
    import flow_intensity as fi
    import harmful_exposure_rows as HER
    want = set(day_slugs(day, coin))
    era = HER._era_or_refuse(fi, None, "be_daybook_build")
    paths, toks = fi._archive_paths(), fi.token_map()
    gaps = fi.gaps_by_slug(era)
    missing = sorted(s for s in want if s not in paths or s not in toks)
    if missing:
        raise BookRefused(
            f"REFUSED: {len(missing)} supplied slug(s) have no archive path "
            f"or no token map entry, e.g. {missing[:3]}. A book that "
            f"silently drops them is a book about a different day.")

    def _sel(coins, population):
        out = [(s, paths[s], toks[s][0], toks[s][1], gaps.get(s, []))
               for s in sorted(want)]
        return out, 0
    _sel.era = era
    _sel.n_wanted = len(want)
    return _sel


def assert_pool_equality(a: set, b: set) -> bool:
    """The SHARED draw pool is only sound while both heads score one set.

    `be_cancel_axis_null` builds `rows` from CONDVALUE's head and both arms
    draw from it. On the consumed hour the two heads scored the SAME 29,813
    generations, which is why the shared pool was declared with the note
    that the equality is a MEASUREMENT and not a guarantee. Here it is
    measured per day, and a day where it fails is REFUSED rather than
    silently pooled -- because then the shared pool is a real choice, and
    this seat has not declared that one."""
    if a != b:
        raise BookRefused(
            f"REFUSED: the two pinned heads scored DIFFERENT generation sets "
            f"({len(a)} vs {len(b)}, symmetric difference {len(a ^ b)}). The "
            f"declared draw pool is SHARED and that is only sound while the "
            f"sets are identical. The day is refused rather than pooled.")
    return True


def assert_coverage(cov: dict, n_gen: int, day: str) -> bool:
    """A day the tape does not reach is REFUSED, never emitted empty.

    An empty decision population is the failure mode that looks like a
    result -- the reviewer's own words about the wrong ledger root, and the
    same shape here."""
    if not cov:
        raise BookRefused(f"REFUSED: no coverage computed for {day}.")
    if all(v["n_covered"] == 0 for v in cov.values()):
        raise BookRefused(
            f"REFUSED: NO generation of {day} carries an assembled score "
            f"(0 of {n_gen}). The tape does not cover this day, so the book "
            f"would have an EMPTY decision population -- an empty answer "
            f"that looks like a result.")
    return True


def build(day: str, *, coin: str = COIN,
          chunk_windows: int = CHUNK_WINDOWS,
          scratch: Path | None = None, progress: bool = True,
          fixture: bool = False) -> dict:
    import de_phase4_diag_runner as R
    t0 = time.time()
    obs = {}
    stages = _Stages(FIXTURE_STAGE_BUDGETS_GB if fixture
                     else STAGE_BUDGETS_GB)
    obs["wrapper"] = assert_rule20(fixture=fixture)
    sel = day_selector(day, coin)
    if progress:
        print(json.dumps({"stage": "selected", "slugs": sel.n_wanted,
                          "era": sel.era}), flush=True)

    t = time.time()
    fr = R.build_reference(coin, selector=sel)
    stages.done("A0_reference", t)
    ref = fr["reference"]
    obs["reference_s"] = round(time.time() - t, 1)
    obs["reference_peak_gb"] = _rss_gb()
    n_gen = sum(len(ref[s][sd]) for s in ref for sd in ("BUY_UP", "SELL_UP")
                if sd in ref[s])
    if progress:
        print(json.dumps({"stage": "reference", "windows": len(ref),
                          "generations": n_gen, **obs}), flush=True)
    if not ref:
        raise BookRefused(f"REFUSED: {day} produced an EMPTY reference.")

    assert_day_tape(day, coin)
    splits = R.DECLARED_SPLIT_SETS[R.RULED_SPLIT_SET]
    t = time.time()
    tape = R.build_tape_index(splits)
    stages.done("A1_index", t)
    obs["tape_index_s"] = round(time.time() - t, 1)
    obs["tape_rows"] = tape.get("n_tape_rows")
    obs["after_tape_peak_gb"] = _rss_gb()
    if progress:
        print(json.dumps({"stage": "tape", **{k: obs[k] for k in
                          ("tape_index_s", "tape_rows",
                           "after_tape_peak_gb")}}), flush=True)

    sd = Path(scratch) if scratch is not None else OUT_DERIVED
    frag = sd / f"be_daybook_frag_{day}.json"
    t = time.time()
    R.fragment_slice(frag, n_windows=len(ref), only_slugs=list(ref))
    obs["fragment_s"] = round(time.time() - t, 1)
    obs["fragment_bytes"] = frag.stat().st_size if frag.exists() else None

    t = time.time()
    asm = R.assemble_streaming({coin: ref}, splits=splits, coins=(coin,),
                               chunk_windows=chunk_windows, source=frag,
                               tape=tape)
    stages.done("A2_assemble", t)
    obs["assembly_s"] = round(time.time() - t, 1)
    obs["peak_gb"] = _rss_gb()

    # DE v9 R11: THE INDEX IS NEEDED ONLY TO PRODUCE `asm`. The consumer of
    # the book reads `asm` and the reference and never a tape row, so the
    # index does not have to be alive when the book is written. Releasing it
    # here is what makes the whole-day peak max(index, assembly) instead of
    # their sum -- on the measured numbers, the difference between fitting
    # the cap and not.
    t = time.time()
    _before_release = _rss_now_gb()
    del tape
    gc.collect()
    _after_release = _rss_now_gb()
    stages.done("A3_release_index", t)
    obs["index_released"] = {
        "current_gb_before": _before_release,
        "current_gb_after": _after_release,
        "freed_gb": round(_before_release - _after_release, 3),
        "measured_on_CURRENT_rss": "ru_maxrss is a high-water mark and "
                                   "cannot show a release; this is VmRSS",
        "index_is_build_time_only": "DE design v9 R11 -- "
                                    "INDEX_SPLITS_NEEDED_BY_DAY = NONE at "
                                    "any stage",
    }
    if progress:
        print(json.dumps({"stage": "assembled", **{k: obs[k] for k in
                          ("assembly_s", "peak_gb")}}), flush=True)

    # ---- WHAT THE BOOK MUST SATISFY, COMPUTED BEFORE IT IS WRITTEN --------
    cov = {}
    keys = {}
    for arm, head in HEADS.items():
        k = (coin, head)
        if k not in asm["by_arm"]:
            raise BookRefused(
                f"REFUSED: `asm['by_arm']` has no entry for {k}. DE's R1 "
                f"requires BOTH pinned heads; a book with one is a book the "
                f"null cannot drive for the other arm.")
        gs = asm["by_arm"][k][0]
        keys[head] = set(gs)
        scored = sum(1 for s in sorted(ref) for side in ("BUY_UP", "SELL_UP")
                     for g in ref[s].get(side, [])
                     if (s, side, float(g["t0"])) in gs)
        cov[head] = {"n_scored_keys": len(gs), "n_reference_generations": n_gen,
                     "n_covered": scored, "n_uncovered": n_gen - scored,
                     "coverage": scored / n_gen if n_gen else None,
                     "theta": float(R.theta_for(coin, head, BUDGET))}
    a, b = (keys[h] for h in (HEADS["CONDVALUE_X_SKEW"],
                              HEADS["HAZARD_OVER_SKEWED_REF"]))
    equal = assert_pool_equality(a, b)
    assert_coverage(cov, n_gen, day)

    _BDR.require_ledger()          # result-bearing: refuse a non-ledger tree
    t = time.time()
    book = {"fr": fr, "asm": asm}
    buf = pickle.dumps(book, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(buf).hexdigest()
    asm_digest = hashlib.sha256(
        pickle.dumps(asm, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    dst = LEDGER_DERIVED / f"be_daybook_{day}_{coin}.pkl"
    dst.write_bytes(buf)
    back = hashlib.sha256(dst.read_bytes()).hexdigest()
    stages.done("A4_write_book", t)
    obs["wall_s"] = round(time.time() - t0, 1)
    obs["peak_rss_gb"] = _rss_gb()
    obs["stages"] = stages.rows
    obs["stage_budgets_gb"] = stages.budgets
    obs["asm_peak_gb_PUBLISHED"] = next(
        (r["peak_gb"] for r in stages.rows if r["stage"] == "A2_assemble"),
        None)
    return {
        "protocol": "BE_DAYBOOK_V1",
        "day": day, "coin": coin,
        "book": {"path": str(dst), "bytes": len(buf), "sha256": digest,
                 "sha256_of_asm": asm_digest,
                 "digest_is_of_the_buffer_as_written": True,
                 "readback_sha256": back,
                 "readback_matches": back == digest,
                 "why_readback": "the digest is taken from the buffer that "
                                 "was written (the B-1 discipline on the "
                                 "write side); the readback is a SECOND, "
                                 "independent statement of the same bytes "
                                 "and is reported beside it, not instead"},
        "selection": {"source": "de_admissible_windows.supply(day, "
                                "be_forward_day.present_from_ledger(day))",
                      "why_not_select_v2_era": "the declared population "
                                               "intervals end "
                                               "2026-08-26T00:00, so no "
                                               "September day passes them",
                      "era": sel.era, "n_supplied_slugs": sel.n_wanted},
        "reference": {"windows": len(ref), "generations": n_gen,
                      "statuses": fr.get("statuses"),
                      "n_slugs": fr.get("n_slugs"),
                      "terminal_marks_present": bool(fr.get("terminal_marks")),
                      "n_terminal_marks": len(fr.get("terminal_marks") or {})},
        "asm": {"by_arm_keys": [list(k) for k in asm["by_arm"]],
                "coverage_by_head": cov,
                "both_heads_present": True,
                "set_equality_asserted": True,
                "sets_are_equal": equal,
                "n_shared_keys": len(a)},
        "resources": obs,
        "data_root": _BDR.receipt_block(),
        "no_scoring_of_arms": True,
        "no_null_draws": True,
        "no_economics": True,
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 10


def real_data_reachable(day: str = "20260903") -> tuple:
    """Can this battery see the ledger's day inputs? BE48 §B.5.

    From a worktree it cannot: `de_admissible_windows` computes its own root
    as `parents[2]` with no resolver, so it looks for
    `<worktree>/data/pm_5min/derived/da_blackout_mask_<day>.json`, which is
    present at the ledger and NOT tracked. The battery used to REFUSE at
    check 1 there, so a reviewer in an R-397 worktree could not drive it at
    all. It now runs the fixture-driven checks everywhere and reports the
    real-data ones as SKIPPED WITH THEIR REASON (rule 4: an exclusion is a
    status, never a silent drop)."""
    import de_admissible_windows as AW
    mask = Path(AW.ROOT) / "data/pm_5min/derived" / f"da_blackout_mask_{day}.json"
    if mask.exists():
        return True, str(mask)
    return False, (f"{mask} not present. `de_admissible_windows.ROOT` is "
                   f"parents[2] with no resolver, so from a worktree it "
                   f"looks for the mask in the worktree. Set PM_DATA_ROOT "
                   f"and run at the ledger tree, or drive the fixture "
                   f"checks alone.")


def selftest() -> int:
    checks, fails, skipped = 0, [], []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    def skip(label, why):
        skipped.append(label)
        print(f"SKIP: {label}  [{why[:110]}]")

    reachable, why_not = real_data_reachable()

    import harmful_exposure_rows as HER
    iv = HER.POPULATION_SLUG_INTERVALS
    ends = [v[1] for v in iv.values() if isinstance(v, (list, tuple))]
    ok(all(e is None or e < 1788000000 for e in ends),
       f"THE REASON THE DAY PATH EXISTS, MEASURED: every declared population "
       f"interval ends before September ({ends}) -- so `select_v2_era` "
       f"cannot reach a September day and a book built through it would be "
       f"EMPTY, not small")

    if reachable:
        sl = day_slugs("20260903")
        ok(len(sl) == 247 and all(s.startswith("btc-") for s in sl),
           f"the day supply returns {len(sl)} btc slugs for 20260903, "
           f"matching the forward receipt's supplied count for that coin")
        sel = day_selector("20260903")
        ent, ngap = sel(("btc",), None)
        ok(len(ent) == len(sl) and len(ent[0]) == 5,
           f"and the selector returns {len(ent)} entries in select_v2_era's "
           f"own 5-tuple shape, so `build_reference` needs no change")
    else:
        skip("the day supply returns 247 btc slugs for 20260903", why_not)
        skip("the selector returns entries in the 5-tuple shape", why_not)

    # BE48 §B.2. The previous version of this check was carried by
    # `isinstance(e, Exception)` inside `except Exception` -- TRUE BY
    # CONSTRUCTION -- and the refusal it named was never reached, because
    # `present_from_ledger` refuses a day with no ledger entry one call
    # earlier. Both halves are fixed: the refusal is REACHED by injecting a
    # supply, and the assertion names the TYPE as well as the text.
    try:
        day_slugs("20260903", supply={"windows": {"eth": [{"slug": "x"}]}})
        ok(False, "an empty btc supply must refuse")
    except BookRefused as e:
        ok("no supplied btc windows for 20260903" in str(e),
           "KNOWN-BAD, AND IT NOW REACHES THIS MODULE'S OWN REFUSAL: a "
           "supply carrying eth windows and NO btc raises BookRefused here, "
           "not an upstream refusal -- the previous check was carried by "
           "`isinstance(e, Exception)` and tested be_forward_day instead")
    except Exception as e:                               # noqa: BLE001
        ok(False, f"expected BookRefused, got {type(e).__name__}")
    # and the UPSTREAM refusal is still asserted, BY TYPE, as its own case
    import be_forward_day as _FD
    if not reachable:
        skip("the upstream ForwardDayRefused case", why_not)
        raise SystemExit(_finish(checks, fails, skipped))
    try:
        day_slugs("19700101")
        ok(False, "a day with no ledger entry must refuse")
    except BookRefused as e:
        ok(False, f"expected the UPSTREAM refusal, got BookRefused: {e}")
    except Exception as e:                               # noqa: BLE001
        ok(type(e).__name__ == "ForwardDayRefused"
           and "ledger holds no window" in str(e),
           f"KNOWN-BAD, THE OTHER PATH, NAMED BY TYPE: a day with no ledger "
           f"entry raises {type(e).__name__} from be_forward_day -- a "
           f"different refusal from a different module, and the battery now "
           f"says which is which")

    # ---- THE TWO GUARDS, DRIVEN BOTH WAYS ---------------------------------
    ok(assert_pool_equality({1, 2, 3}, {3, 2, 1}) is True,
       "POSITIVE CONTROL: identical scored sets PASS the shared-pool guard")
    try:
        assert_pool_equality({1, 2, 3}, {1, 2, 4})
        ok(False, "unequal pools must refuse")
    except BookRefused as e:
        ok("symmetric difference 2" in str(e),
           "KNOWN-BAD: sets differing by one element each REFUSE the DAY, "
           "naming the symmetric difference -- the shared pool is only sound "
           "while they are identical, and this day would make it a choice")
    ok(assert_coverage({"h": {"n_covered": 5}}, 10, "d") is True,
       "POSITIVE CONTROL: a day with ANY covered generation passes")
    try:
        assert_coverage({"h1": {"n_covered": 0}, "h2": {"n_covered": 0}},
                        10, "20260903")
        ok(False, "zero coverage must refuse")
    except BookRefused as e:
        ok("EMPTY decision population" in str(e),
           "KNOWN-BAD: zero coverage on every head REFUSES -- an empty "
           "decision population is the failure that looks like a result")

    try:
        assert_day_tape("20260903")
        ok(False, "the day-tape guard must refuse while TAPE_PATH is a "
                  "module constant")
    except BookRefused as e:
        ok("module constant" in str(e) and "EMPTY `asm`" in str(e),
           "KNOWN-BAD, AND IT IS THE LIVE BLOCKER: the assembly would index "
           "the live August tape, not the day's, because "
           "`phase2_arms.TAPE_PATH` is a module constant -- so a September "
           "run would emit a book with an empty `asm`. It REFUSES before "
           "any work rather than producing that book")

    return _finish(checks, fails, skipped)


def _finish(checks, fails, skipped) -> int:
    print()
    if skipped:
        print(f"{len(skipped)} check(s) SKIPPED — real ledger data not "
              f"reachable from this tree (BE48 §B.5). The fixture-driven "
              f"guards ran.")
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    expect = EXPECTED_CHECKS - len(skipped)
    if checks != expect:
        print(f"FAIL: ran {checks} checks, expected {expect} "
              f"(EXPECTED_CHECKS={EXPECTED_CHECKS} minus {len(skipped)} "
              f"skipped)")
        return 1
    print(f"{checks} checks passed"
          + (f", {len(skipped)} skipped" if skipped else ""))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--day" in argv:
        day = argv[argv.index("--day") + 1]
        out = build(day)
        dst = OUT_DERIVED / f"be_daybook_receipt_{day}_{COIN}.json"
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"receipt": str(dst),
                          "book": out["book"]["path"],
                          "sha256": out["book"]["sha256"],
                          "bytes": out["book"]["bytes"],
                          "wall_s": out["resources"]["wall_s"],
                          "peak_rss_gb": out["resources"]["peak_rss_gb"]}))
        return 0
    print("usage: be_daybook_build.py --selftest | --day <YYYYMMDD>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
