"""BE 194 -- the 30-minute sigma producer, to fair_value_plan.md v1.2 §4 C2.

THE CONTRACT, as landed (every clause is a predicate below, not a sentence):

  sigma = sqrt(mean(r_1s^2)) over the trailing 30 minutes of ONE-SECOND
  Binance USDM bookTicker MIDPOINT log returns, in PER-SQUARE-ROOT-SECOND
  units.  NO ANNUALISATION -- the grid spacing is one second, so the estimator
  is the raw root-mean-square return and nothing multiplies it.

  SHIFTED BY ONE COMPLETE OBSERVATION.  Let S be the second containing
  `decision_recv_ns`.  S is INCOMPLETE at decision time, so the grid ends at
  S-1: the last second that is a complete observation.  Every grid instant is
  therefore strictly earlier than the decision, and
  `sigma_local_knowledge_ns <= decision_recv_ns` is structural, not checked
  after the fact -- it is checked anyway, because a structural claim that is
  never evaluated is a sentence.

  THE GRID RULE.  Each one-second grid value is the LATEST midpoint whose
  LOCAL-KNOWLEDGE time (recv_ns, column 1) is AT OR BEFORE that grid instant.
  No interpolation.  No later tick may fill it.  A grid instant with no such
  tick has NO VALUE, and the two returns touching it are ABSENT -- they are
  not zero.  Zero is a measurement; absence is not.

  ADMISSIBILITY, all four required:
    1. n_returns >= 90% of the 1,800 expected one-second returns (>= 1620);
    2. no source gap above five seconds;
    3. finite POSITIVE result;
    4. sigma_local_knowledge_ns <= decision_recv_ns.

  ERA FLOOR, applied PER EVENT: recv_ns >= 1787579334881534478
  (2026-08-24 13:48:54 UTC).  Earlier rows are NOT ADMITTED -- they are
  dropped at the row, not the file, because the boundary falls inside an hour.

  ON ANY FAILURE this emits a TYPED non-OK status and NO sigma.  It never
  accepts a caller-supplied fallback and never substitutes Identity: §4 gives
  the declared fallback to the policy wrapper, which must also COUNT it.
  A sigma field is either a measurement or absent; there is no third thing.

WHY STALENESS IS ITS OWN STATUS.  A dead feed does not reduce the return
count -- the grid rule forward-fills the last known midpoint, so 1,800
returns of exactly 0.0 are produced and the 90% count test PASSES on a feed
that stopped.  Coverage cannot see it and neither can the interior-gap test
if the stall is at the trailing edge.  So the gap from the last tick to the
grid end is measured separately and refuses as STALE_INPUT.
"""
from __future__ import annotations

import gzip
import math
from pathlib import Path

ERA_FLOOR_NS = 1787579334881534478          # 2026-08-24 13:48:54 UTC, hf_ws_v2
NS = 1_000_000_000
WINDOW_S = 1800                              # thirty minutes
N_EXPECTED_RETURNS = 1800                    # 1801 grid points -> 1800 returns
MIN_COVERAGE = 0.90
MIN_RETURNS = math.ceil(MIN_COVERAGE * N_EXPECTED_RETURNS)   # 1620
MAX_SOURCE_GAP_NS = 5 * NS

OK = "OK"
PRE_ERA = "PRE_ERA"
NO_SOURCE_ROWS = "NO_SOURCE_ROWS"
INSUFFICIENT_RETURNS = "INSUFFICIENT_RETURNS"
SOURCE_GAP_OVER_LIMIT = "SOURCE_GAP_OVER_LIMIT"
STALE_INPUT = "STALE_INPUT"
ZERO_VOLATILITY = "ZERO_VOLATILITY"
NON_FINITE_SIGMA = "NON_FINITE_SIGMA"
FUTURE_KNOWLEDGE_IN_SOURCE = "FUTURE_KNOWLEDGE_IN_SOURCE"
BAD_REQUEST = "BAD_REQUEST"

CONTRACT = "BE_SIGMA_30M_V1"


def _record(status, **kw):
    """Every return of this module has the same shape and names its status."""
    out = {
        "contract": CONTRACT,
        "status": status,
        "ok": status == OK,
        "sigma_per_sqrt_s": None,
        "sigma_local_knowledge_ns": None,
        "annualised": False,
        "units": "per_sqrt_second",
        "n_returns": None,
        "n_expected_returns": N_EXPECTED_RETURNS,
        "coverage": None,
        "min_returns_required": MIN_RETURNS,
        "max_source_gap_ns": None,
        "max_source_gap_limit_ns": MAX_SOURCE_GAP_NS,
        "trailing_staleness_ns": None,
        "grid_start_ns": None,
        "grid_end_ns": None,
        "decision_recv_ns": None,
        "era_floor_ns": ERA_FLOOR_NS,
        "n_rows_seen": None,
        "n_rows_pre_era": None,
        "n_rows_after_decision": None,
        "shifted_by_one_complete_observation": True,
    }
    out.update(kw)
    return out


def sigma_30m(ticks, decision_recv_ns, *, era_floor_ns=ERA_FLOOR_NS):
    """ticks: iterable of (recv_ns, mid) ASCENDING in recv_ns.

    Returns the typed record.  Never raises on data; raises only on a caller
    error it must not paper over (a non-integer decision time).
    """
    if not isinstance(decision_recv_ns, int) or decision_recv_ns <= 0:
        return _record(BAD_REQUEST, decision_recv_ns=decision_recv_ns)

    # ---- THE SHIFT.  S is the second containing the decision and is
    # INCOMPLETE at decision time; the last complete observation is S-1.
    decision_sec = decision_recv_ns // NS
    grid_end_sec = decision_sec - 1
    grid_start_sec = grid_end_sec - WINDOW_S          # 1801 points inclusive
    grid_end_ns = grid_end_sec * NS
    grid_start_ns = grid_start_sec * NS
    base = dict(decision_recv_ns=decision_recv_ns,
                grid_start_ns=grid_start_ns, grid_end_ns=grid_end_ns)

    if grid_start_ns < era_floor_ns:
        # the window reaches below the era floor: the rows that would fill its
        # left edge are INADMISSIBLE, so the window itself is.
        return _record(PRE_ERA, **base)

    # ---- ONE PASS, the grid rule as an algorithm.
    # Ticks arrive ascending in recv_ns.  Before a tick at `recv` is taken in,
    # every grid instant STRICTLY BEFORE `recv` is final: no later tick can be
    # its latest-at-or-before, so it is settled with whatever came earlier --
    # which may be None, and None is ABSENCE, not zero.  A tick therefore only
    # ever fills instants at or after its own recv_ns.  "No later tick may fill
    # it" is enforced by the flush happening BEFORE the tick is adopted.
    grid = [None] * (WINDOW_S + 1)
    idx = 0
    last_mid = None
    last_recv = None
    prev_recv_in_window = None
    max_gap = 0
    n_seen = n_pre_era = n_after_decision = 0

    for recv_ns, mid in ticks:
        n_seen += 1
        if recv_ns < era_floor_ns:
            n_pre_era += 1
            continue                          # NOT ADMITTED, per event
        if recv_ns > decision_recv_ns:
            # Nothing at or after the decision instant may inform it. Counted,
            # not silently skipped: a source handing us future rows is a fact
            # about the caller worth reporting.
            n_after_decision += 1
            continue
        if mid is None or not math.isfinite(mid) or mid <= 0.0:
            continue                          # a malformed row is not a value
        while idx <= WINDOW_S and grid_start_ns + idx * NS < recv_ns:
            grid[idx] = last_mid
            idx += 1
        if grid_start_ns <= recv_ns <= grid_end_ns:
            if prev_recv_in_window is not None:
                gap = recv_ns - prev_recv_in_window
                if gap > max_gap:
                    max_gap = gap
            prev_recv_in_window = recv_ns
        last_mid = mid
        last_recv = recv_ns

    if last_mid is None:
        return _record(NO_SOURCE_ROWS, n_rows_seen=n_seen,
                       n_rows_pre_era=n_pre_era,
                       n_rows_after_decision=n_after_decision, **base)

    # every instant still unsettled is at or after the last tick, so the last
    # tick is its latest-at-or-before.
    while idx <= WINDOW_S:
        grid[idx] = last_mid
        idx += 1

    # ---- trailing staleness: the gap from the last admitted tick to the grid
    # end.  A feed that stopped produces full coverage of ZERO returns; only
    # this sees it.
    trailing = grid_end_ns - last_recv if last_recv is not None else None

    # ---- returns.  A return exists only where BOTH endpoints have a value.
    ssq = 0.0
    n_ret = 0
    for j in range(1, WINDOW_S + 1):
        a, b = grid[j - 1], grid[j]
        if a is None or b is None:
            continue
        r = math.log(b / a)
        ssq += r * r
        n_ret += 1

    coverage = n_ret / N_EXPECTED_RETURNS
    common = dict(n_returns=n_ret, coverage=coverage, max_source_gap_ns=max_gap,
                  trailing_staleness_ns=trailing, n_rows_seen=n_seen,
                  n_rows_pre_era=n_pre_era,
                  n_rows_after_decision=n_after_decision,
                  sigma_local_knowledge_ns=last_recv, **base)

    # ---- ADMISSIBILITY.  Order is fixed so a record's status names the FIRST
    # clause it failed, and is stable across runs.
    if trailing is not None and trailing > MAX_SOURCE_GAP_NS:
        return _record(STALE_INPUT, **common)
    if max_gap > MAX_SOURCE_GAP_NS:
        return _record(SOURCE_GAP_OVER_LIMIT, **common)
    if n_ret < MIN_RETURNS:
        return _record(INSUFFICIENT_RETURNS, **common)
    sigma = math.sqrt(ssq / n_ret)
    if not math.isfinite(sigma):
        return _record(NON_FINITE_SIGMA, **common)
    if sigma <= 0.0:
        return _record(ZERO_VOLATILITY, **common)
    if last_recv > decision_recv_ns:          # structural, checked anyway
        return _record(FUTURE_KNOWLEDGE_IN_SOURCE, **common)
    return _record(OK, sigma_per_sqrt_s=sigma, **common)


# --------------------------------------------------------------------------
# the real source: Binance USDM bookTicker.  Column 1 is recv_ns (LOCAL
# KNOWLEDGE); columns 5..8 are bid px, bid qty, ask px, ask qty.
# --------------------------------------------------------------------------
def bookticker_ticks(symbol, start_ns, end_ns, root=None):
    """Yield (recv_ns, mid) for hours covering [start_ns, end_ns]."""
    import datetime
    base = Path(root or "/home/yuqing/ctaNew/data/mm_hf/raw/bookTicker") / symbol
    if not base.is_dir():
        return
    start = datetime.datetime.fromtimestamp(
        start_ns / 1e9, datetime.timezone.utc).replace(
            minute=0, second=0, microsecond=0)
    end = datetime.datetime.fromtimestamp(
        end_ns / 1e9, datetime.timezone.utc).replace(
            minute=0, second=0, microsecond=0)
    hour = start
    while hour <= end:
        for path in sorted(base.glob(hour.strftime("%Y%m%d_%H") + ".csv*")):
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rb") as fh:
                for line in fh:
                    p = line.split(b",")
                    if len(p) < 8:
                        continue
                    try:
                        recv = int(p[0])
                        bp = float(p[4]); ap = float(p[6])
                    except ValueError:
                        continue
                    if recv > end_ns:
                        return
                    if recv < start_ns:
                        continue
                    yield recv, (bp + ap) / 2.0
        hour += datetime.timedelta(hours=1)


# The grid rule asks for the latest midpoint AT OR BEFORE the first grid
# instant, which is in general a tick from BEFORE the window. Loading from the
# grid start exactly therefore discards the only row that can fill grid[0] and
# silently costs one return on every dense window (measured: 1799 of 1800).
# The estimator was right and the loader was starving it. PREROLL supplies it.
# If nothing falls in the preroll the first instant is genuinely absent, which
# is the correct outcome -- absence, not a value invented by reaching further.
PREROLL_NS = 60 * NS


def sigma_30m_for_symbol(symbol, decision_recv_ns, **kw):
    grid_start = (decision_recv_ns // NS - 1 - WINDOW_S) * NS
    return sigma_30m(
        bookticker_ticks(symbol, grid_start - PREROLL_NS, decision_recv_ns),
        decision_recv_ns, **kw)


# ==========================================================================
# FALSIFIERS -- fair_value_plan.md §5 gate 2, all seven, each TWO-WAY.
# Every cell calls the production entry point `sigma_30m` (or
# `sigma_30m_for_symbol`); none reaches inside it.
# ==========================================================================
def _synth(decision_recv_ns, s, *, first_index=0, last_index=WINDOW_S,
           mid0=100.0, skip=(), extra_offsets_ns=(), sign_flip=True):
    """One tick exactly on each one-second grid instant.

    Alternating log returns of magnitude `s` give mean(r^2) == s^2 EXACTLY,
    so the expected sigma is `s` with no sampling error -- a random path would
    make every tolerance an argument about noise instead of about the
    estimator.
    """
    decision_sec = decision_recv_ns // NS
    grid_end_sec = decision_sec - 1
    grid_start_sec = grid_end_sec - WINDOW_S
    out = []
    mid = mid0
    for k in range(0, WINDOW_S + 1):
        if k > first_index:
            step = s if (not sign_flip or k % 2 == 1) else -s
            mid = mid * math.exp(step)
        if k < first_index or k > last_index or k in skip:
            continue
        out.append(((grid_start_sec + k) * NS, mid))
    for off in extra_offsets_ns:
        out.append((decision_recv_ns + off if off > 0
                    else grid_end_sec * NS + off, mid))
    out.sort(key=lambda t: t[0])
    return out


def falsify(verbose=True):
    import json
    rc = 0
    rows = []

    def note(name, passed, detail=""):
        nonlocal rc
        if not passed:
            rc = 1
        rows.append((name, passed, detail))
        if verbose:
            print(f"  {'PASS' if passed else 'FAIL'}  {name}"
                  + (f"   [{detail}]" if detail else ""))

    # a decision instant comfortably inside the era
    D = (ERA_FLOOR_NS // NS + 100_000) * NS + 123_456_789

    # ---- 1. SCALE ---------------------------------------------------------
    a = sigma_30m(_synth(D, 2e-5), D)
    b = sigma_30m(_synth(D, 4e-5), D)
    # The fixture builds mids by exp() and the estimator recovers them by
    # log(); that round trip costs ~1 ulp of the MIDPOINT, which is ~2.2e-16
    # absolute and therefore ~1.1e-11 RELATIVE to a return of 2e-5. So the
    # nominal comparison gets a relative tolerance, and the cell below pins
    # the residual to the FIXTURE rather than leaving it as a loose bound.
    note("scale: a path with per-second sigma 2e-5 is measured as 2e-5",
         a["status"] == OK
         and abs(a["sigma_per_sqrt_s"] / 2e-5 - 1.0) < 1e-9,
         f"{a['status']} sigma={a['sigma_per_sqrt_s']!r} "
         f"rel={a['sigma_per_sqrt_s'] / 2e-5 - 1.0:.3e}")
    fixture = _synth(D, 2e-5)
    mids = [m for _, m in fixture]
    independent = math.sqrt(
        sum(math.log(mids[i] / mids[i - 1]) ** 2
            for i in range(1, len(mids))) / (len(mids) - 1))
    note("scale: the estimate equals an INDEPENDENT rms over the fixture's own "
         "midpoints to within 1 ulp -- so the residual above is the fixture's "
         "exp/log round trip, not the estimator",
         abs(a["sigma_per_sqrt_s"] - independent)
         <= 4 * abs(math.ulp(independent)),
         f"delta={a['sigma_per_sqrt_s'] - independent:.3e} "
         f"ulp={math.ulp(independent):.3e}")
    note("scale: DOUBLING the path's volatility DOUBLES the estimate",
         b["status"] == OK
         and abs(b["sigma_per_sqrt_s"] / a["sigma_per_sqrt_s"] - 2.0) < 1e-9,
         f"ratio={b['sigma_per_sqrt_s'] / a['sigma_per_sqrt_s']:.12f}")
    note("scale: the two estimates are NOT equal -- the estimator can see scale",
         a["sigma_per_sqrt_s"] != b["sigma_per_sqrt_s"])
    note("scale: NO ANNUALISATION -- the estimate is per-sqrt-second, not "
         "sigma*sqrt(31_536_000)",
         abs(a["sigma_per_sqrt_s"] - 2e-5 * math.sqrt(31_536_000)) > 1e-6
         and a["units"] == "per_sqrt_second" and a["annualised"] is False)

    # ---- 2. MINIMUM COUNT -------------------------------------------------
    ok_count = sigma_30m(_synth(D, 2e-5, first_index=WINDOW_S - MIN_RETURNS), D)
    bad_count = sigma_30m(
        _synth(D, 2e-5, first_index=WINDOW_S - MIN_RETURNS + 1), D)
    note(f"minimum-count: exactly {MIN_RETURNS} returns (90.0%) is ADMITTED",
         ok_count["status"] == OK and ok_count["n_returns"] == MIN_RETURNS,
         f"{ok_count['status']} n={ok_count['n_returns']}")
    note(f"minimum-count: {MIN_RETURNS - 1} returns REFUSES INSUFFICIENT_RETURNS",
         bad_count["status"] == INSUFFICIENT_RETURNS
         and bad_count["n_returns"] == MIN_RETURNS - 1
         and bad_count["sigma_per_sqrt_s"] is None,
         f"{bad_count['status']} n={bad_count['n_returns']}")

    # ---- 3. GAP -----------------------------------------------------------
    ok_gap = sigma_30m(_synth(D, 2e-5, skip=(900, 901, 902, 903)), D)
    bad_gap = sigma_30m(_synth(D, 2e-5, skip=(900, 901, 902, 903, 904)), D)
    note("gap: a source gap of exactly 5s is ADMITTED (the limit is ABOVE 5s)",
         ok_gap["status"] == OK and ok_gap["max_source_gap_ns"] == 5 * NS,
         f"{ok_gap['status']} max_gap={ok_gap['max_source_gap_ns']}")
    note("gap: a source gap of 6s REFUSES SOURCE_GAP_OVER_LIMIT",
         bad_gap["status"] == SOURCE_GAP_OVER_LIMIT
         and bad_gap["max_source_gap_ns"] == 6 * NS
         and bad_gap["sigma_per_sqrt_s"] is None,
         f"{bad_gap['status']} max_gap={bad_gap['max_source_gap_ns']}")

    # ---- 4. ZERO VOLATILITY ----------------------------------------------
    flat = sigma_30m(_synth(D, 0.0), D)
    tiny = sigma_30m(_synth(D, 1e-12), D)
    note("zero-volatility: a constant midpoint REFUSES ZERO_VOLATILITY, "
         "with no sigma",
         flat["status"] == ZERO_VOLATILITY and flat["sigma_per_sqrt_s"] is None,
         f"{flat['status']} n={flat['n_returns']}")
    note("zero-volatility: a 1e-12 path is ADMITTED -- the guard is zero, "
         "not 'small'",
         tiny["status"] == OK and tiny["sigma_per_sqrt_s"] > 0.0,
         f"{tiny['status']} sigma={tiny['sigma_per_sqrt_s']!r}")

    # ---- 5. STALE INPUT ---------------------------------------------------
    ok_stale = sigma_30m(_synth(D, 2e-5, last_index=WINDOW_S - 5), D)
    bad_stale = sigma_30m(_synth(D, 2e-5, last_index=WINDOW_S - 6), D)
    note("stale-input: a feed whose last tick is 5s before the grid end is "
         "ADMITTED",
         ok_stale["status"] == OK
         and ok_stale["trailing_staleness_ns"] == 5 * NS,
         f"{ok_stale['status']} trailing={ok_stale['trailing_staleness_ns']}")
    note("stale-input: 6s of trailing silence REFUSES STALE_INPUT",
         bad_stale["status"] == STALE_INPUT
         and bad_stale["sigma_per_sqrt_s"] is None,
         f"{bad_stale['status']} trailing={bad_stale['trailing_staleness_ns']}")
    note("stale-input: and the COUNT TEST ALONE WOULD HAVE PASSED IT -- "
         "the forward fill produced full coverage of dead returns",
         bad_stale["n_returns"] >= MIN_RETURNS,
         f"n_returns={bad_stale['n_returns']} >= {MIN_RETURNS}")

    # ---- 6. PRE-ERA -------------------------------------------------------
    d_pre = (ERA_FLOOR_NS // NS + 10) * NS      # window reaches below the floor
    pre = sigma_30m(_synth(d_pre, 2e-5), d_pre)
    note("pre-era: a window reaching below the hf_ws_v2 floor REFUSES PRE_ERA",
         pre["status"] == PRE_ERA and pre["sigma_per_sqrt_s"] is None,
         f"{pre['status']}")
    note("pre-era: a window entirely at or above the floor is ADMITTED",
         a["status"] == OK)
    mixed = _synth(D, 2e-5)
    mixed = [(ERA_FLOOR_NS - 10 * NS, 100.0)] + mixed   # one inadmissible row
    per_event = sigma_30m(mixed, D)
    note("pre-era: the floor is applied PER EVENT -- one sub-floor row is "
         "dropped and counted, the rest still measure",
         per_event["status"] == OK and per_event["n_rows_pre_era"] == 1
         and per_event["sigma_per_sqrt_s"] == a["sigma_per_sqrt_s"],
         f"n_pre_era={per_event['n_rows_pre_era']}")

    # ---- 7. FUTURE KNOWLEDGE ---------------------------------------------
    base_ticks = _synth(D, 2e-5)
    with_future = base_ticks + [(D + k * NS, 1e6) for k in (1, 2, 3)]
    fut = sigma_30m(with_future, D)
    note("future-knowledge: rows AFTER the decision instant change nothing -- "
         "the estimate is bit-identical",
         fut["status"] == OK
         and fut["sigma_per_sqrt_s"] == a["sigma_per_sqrt_s"]
         and fut["n_rows_after_decision"] == 3,
         f"n_after={fut['n_rows_after_decision']}")
    # a stream whose ONLY volatility lies after the decision must measure NONE
    flat_then_wild = _synth(D, 0.0) + [(D + k * NS, 100.0 * (1 + k))
                                       for k in (1, 2, 3, 4, 5)]
    fw = sigma_30m(flat_then_wild, D)
    note("future-knowledge: a path that is FLAT until the decision and wild "
         "after it measures ZERO, not the future",
         fw["status"] == ZERO_VOLATILITY and fw["sigma_per_sqrt_s"] is None,
         f"{fw['status']}")
    note("future-knowledge: every admitted record satisfies "
         "sigma_local_knowledge_ns <= decision_recv_ns",
         all(r["sigma_local_knowledge_ns"] <= r["decision_recv_ns"]
             for r in (a, b, ok_count, ok_gap, tiny, ok_stale, fut, per_event)))
    # the SHIFT itself: a tick inside the decision second is at or before the
    # decision, yet must not reach the estimate, because the grid ends at S-1.
    in_decision_second = base_ticks + [(D - 1, 1e6)]
    shifted = sigma_30m(in_decision_second, D)
    note("shift: a tick INSIDE the incomplete decision second is not used -- "
         "the grid ends one complete observation earlier",
         shifted["status"] == OK
         and shifted["sigma_per_sqrt_s"] == a["sigma_per_sqrt_s"]
         and shifted["grid_end_ns"] < (D // NS) * NS,
         f"grid_end={shifted['grid_end_ns']} decision_sec_start={(D // NS) * NS}")

    # ---- 8. NO CALLER FALLBACK, EVER -------------------------------------
    note("no fallback: every non-OK record carries sigma_per_sqrt_s = None",
         all(r["sigma_per_sqrt_s"] is None
             for r in (bad_count, bad_gap, flat, bad_stale, pre, fw)))
    note("no fallback: sigma_30m takes no fallback/default argument",
         "fallback" not in sigma_30m.__code__.co_varnames
         and "default" not in sigma_30m.__code__.co_varnames)

    # ---- 9. WIRED TO THE REAL SOURCE (rule 17: suite-green is not wired).
    # A CONSUMED day only -- fair_value_plan.md §6 names 09-03..09-09 consumed;
    # 09-10 onward is the protected population and is not read here.
    import datetime
    dec_real = int(datetime.datetime(2026, 9, 3, 12, 0, 0,
                                     tzinfo=datetime.timezone.utc
                                     ).timestamp() * NS) + 500_000_000
    real = sigma_30m_for_symbol("BTCUSDT", dec_real)
    note("real source: a consumed-day BTCUSDT window is ADMITTED through the "
         "production entry point, off the real bookTicker files",
         real["status"] == OK and real["sigma_per_sqrt_s"] > 0.0,
         f"{real['status']} sigma={real['sigma_per_sqrt_s']!r} "
         f"n={real['n_returns']} gap={real['max_source_gap_ns']}")
    note("real source: its local knowledge does not reach the decision",
         real["sigma_local_knowledge_ns"] is not None
         and real["sigma_local_knowledge_ns"] <= dec_real,
         f"lag_ns={dec_real - (real['sigma_local_knowledge_ns'] or 0)}")
    note("real source: a DENSE window yields all 1,800 returns -- the loader "
         "supplies the tick that fills the FIRST grid instant",
         real["n_returns"] == N_EXPECTED_RETURNS,
         f"n_returns={real['n_returns']} of {N_EXPECTED_RETURNS}")
    note("real source: and the same window one hour EARLIER gives a DIFFERENT "
         "number -- the reader is reading, not returning a constant",
         (lambda o: o["status"] == OK
          and o["sigma_per_sqrt_s"] != real["sigma_per_sqrt_s"])(
              sigma_30m_for_symbol("BTCUSDT", dec_real - 3600 * NS)))

    if verbose:
        print(json.dumps({"falsifier": "be_sigma_30m",
                          "n": len(rows),
                          "failed": sum(1 for _, p, _ in rows if not p)}))
    return rc


if __name__ == "__main__":
    import sys
    if "--falsify" in sys.argv:
        sys.exit(falsify())
    if "--smoke" in sys.argv:
        import json
        sym = sys.argv[sys.argv.index("--smoke") + 1]
        dec = int(sys.argv[sys.argv.index("--smoke") + 2])
        print(json.dumps(sigma_30m_for_symbol(sym, dec), indent=1))
        sys.exit(0)
    print(__doc__)
