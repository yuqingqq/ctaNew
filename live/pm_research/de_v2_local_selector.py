"""Memory-bounded one-window selector for the P003 v2 Gate-0 smoke.

The historical selector builds a Binance gap index from every bookTicker hour
through the population end before its caller applies ``limit=1``.  That is
correct for a broad run but defeated the smoke's resource bound.  This module
evaluates the identical max-gap predicate only on the selected window interval,
including a prior hour for left coverage and a following hour for right
coverage.  It is intentionally fixed to one window and is not a broad-run
replacement.
"""
from __future__ import annotations

import datetime
import gzip
from pathlib import Path

import harmful_exposure_rows as HER


PROTOCOL = "P003_V2_ONE_WINDOW_LOCAL_CONTINUITY_SELECTOR_V1"
ONE_WINDOW = 1


class LocalSelectorRefused(RuntimeError):
    """The local predicate cannot prove the declared window's continuity."""


def continuity_from_recv_ns(recv_ns, *, interval_start_ns: int,
                            interval_end_ns: int, era_floor_ns: int,
                            era_end_ns: int,
                            max_gap_ns: int = 1_000_000_000) -> dict:
    """Exact coverage/max-gap predicate over sorted receive timestamps."""
    if interval_end_ns <= interval_start_ns:
        raise LocalSelectorRefused("continuity interval must have positive span")
    if not (era_floor_ns <= interval_start_ns < interval_end_ns <= era_end_ns):
        raise LocalSelectorRefused("continuity interval lies outside era bounds")
    previous = None
    n_read = 0
    for raw in recv_ns:
        try:
            current = int(raw)
        except (TypeError, ValueError) as exc:
            raise LocalSelectorRefused(
                f"non-integer receive timestamp {raw!r}") from exc
        n_read += 1
        if current < era_floor_ns:
            continue
        if current > era_end_ns:
            break
        if previous is not None and current < previous:
            raise LocalSelectorRefused("receive timestamps are not sorted")
        if current <= interval_start_ns:
            previous = current
            continue
        if previous is None:
            return {"ok": False, "status": "NO_LEFT_COVERAGE",
                    "n_recv_rows": n_read, "max_gap_ns": None}
        gap = current - previous
        if gap > max_gap_ns:
            return {"ok": False, "status": "GAP_OVER_LIMIT",
                    "n_recv_rows": n_read, "max_gap_ns": gap,
                    "gap_start_ns": previous, "gap_end_ns": current}
        previous = current
        if current >= interval_end_ns:
            return {"ok": True, "status": "OK", "n_recv_rows": n_read,
                    "max_gap_ns": max_gap_ns}
    return {"ok": False, "status": "NO_RIGHT_COVERAGE",
            "n_recv_rows": n_read, "max_gap_ns": None}


def _hour_paths(symbol: str, *, start_ns: int, end_ns: int) -> list[Path]:
    root = HER.qr.base.fi.DATA_ROOT / "data/mm_hf/raw/bookTicker" / symbol
    if not root.is_dir():
        raise LocalSelectorRefused(f"bookTicker directory is absent: {root}")
    start = datetime.datetime.fromtimestamp(
        start_ns / 1e9, datetime.timezone.utc).replace(
            minute=0, second=0, microsecond=0) - datetime.timedelta(hours=1)
    end = datetime.datetime.fromtimestamp(
        end_ns / 1e9, datetime.timezone.utc).replace(
            minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    paths = []
    hour = start
    while hour <= end:
        stem = hour.strftime("%Y%m%d_%H")
        matches = sorted(root.glob(stem + ".csv*"))
        if len(matches) > 1:
            raise LocalSelectorRefused(
                f"hour {stem} has multiple bookTicker files: {matches}")
        paths.extend(matches)
        hour += datetime.timedelta(hours=1)
    if not paths:
        raise LocalSelectorRefused(
            f"no bookTicker files cover {start.isoformat()}..{end.isoformat()}")
    return paths


def _recv_ns(paths: list[Path]):
    previous = None
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as fh:
            for line in fh:
                comma = line.find(b",")
                if comma < 1:
                    continue
                try:
                    current = int(line[:comma])
                except ValueError:
                    continue
                if previous is not None and current < previous:
                    raise LocalSelectorRefused(
                        f"bookTicker timestamps regress at {path}")
                previous = current
                yield current


def local_continuity(t0: int, coin: str, bounds: tuple[float, float]) -> dict:
    symbol = {"btc": "BTCUSDT", "eth": "ETHUSDT"}.get(coin)
    if symbol is None:
        raise LocalSelectorRefused(f"unsupported coin {coin!r}")
    interval_start_ns = int((t0 - 10.0) * 1e9)
    interval_end_ns = int(
        (t0 + HER.qr.base.fi.WINDOW_S + HER.MARKOUT_S + 1.0) * 1e9)
    floor_ns, end_ns = int(bounds[0] * 1e9), int(bounds[1] * 1e9)
    paths = _hour_paths(
        symbol, start_ns=interval_start_ns, end_ns=interval_end_ns)
    verdict = continuity_from_recv_ns(
        _recv_ns(paths), interval_start_ns=interval_start_ns,
        interval_end_ns=interval_end_ns, era_floor_ns=floor_ns,
        era_end_ns=end_ns, max_gap_ns=int(HER.BN_MAX_GAP_S * 1e9))
    return {**verdict, "symbol": symbol,
            "interval_start_ns": interval_start_ns,
            "interval_end_ns": interval_end_ns,
            "files": [str(p) for p in paths]}


class OneWindowSelector:
    """Callable with the historical selector signature; fixed to one window."""

    def __init__(self):
        self.receipt = None

    def __call__(self, coins, population):
        if tuple(coins) != ("btc",):
            raise LocalSelectorRefused(
                f"one-window smoke is fixed to ('btc',), got {tuple(coins)!r}")
        if population != "v3_4_consumed_fragment":
            raise LocalSelectorRefused(
                f"one-window smoke population changed to {population!r}")
        fi = HER.qr.base.fi
        era = HER._era_or_refuse(fi, None, "v2_one_window_local_selector")
        bounds = HER.v2_era_bounds(population)
        paths = fi._archive_paths()
        tokens = fi.token_map()
        gaps = fi.gaps_by_slug(era)
        n_bn_gap = 0
        considered = 0
        for slug in sorted(fi.covered_slugs(era)):
            coin = slug.split("-")[0]
            if coin != "btc" or slug not in paths or slug not in tokens:
                continue
            try:
                t0 = int(slug.rsplit("-", 1)[1])
            except ValueError:
                continue
            if (t0 < bounds[0]
                    or t0 + fi.WINDOW_S + HER.MARKOUT_S + 5.0 > bounds[1]
                    or not HER.slug_in_population(t0, population)):
                continue
            considered += 1
            continuity = local_continuity(t0, coin, bounds)
            if not continuity["ok"]:
                n_bn_gap += 1
                continue
            up, down = tokens[slug]
            selected = [(slug, paths[slug], up, down, gaps.get(slug, []))]
            self.receipt = {
                "protocol": PROTOCOL,
                "era": era,
                "population": population,
                "selection_rule": (
                    "first sorted BTC slug passing the historical population/"
                    "era predicates and an interval-local exact continuity "
                    "check; fixed to one window"),
                "n_candidates_considered": considered,
                "n_binance_gap_excluded_before_selection": n_bn_gap,
                "selected_slug": slug,
                "continuity": continuity,
            }
            return selected, n_bn_gap
        raise LocalSelectorRefused(
            f"no one-window smoke candidate passed after {considered} candidates")


def selftest() -> int:
    checks = 0

    def ok(condition: bool, label: str) -> None:
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_local_selector] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(fn, label: str, needle: str) -> None:
        nonlocal checks
        try:
            fn()
        except LocalSelectorRefused as exc:
            if needle not in str(exc):
                raise SystemExit(
                    f"[de_v2_local_selector] FAIL: {label}: {exc}")
            checks += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(
            f"[de_v2_local_selector] FAIL (no refusal): {label}")

    good = continuity_from_recv_ns(
        [0, 500_000_000, 1_000_000_000, 1_500_000_000,
         2_000_000_000, 2_500_000_000],
        interval_start_ns=500_000_000, interval_end_ns=2_000_000_000,
        era_floor_ns=0, era_end_ns=3_000_000_000)
    ok(good["ok"] and good["status"] == "OK",
       "positive control proves left/right coverage with sub-limit gaps")
    gap = continuity_from_recv_ns(
        [0, 500_000_000, 1_700_000_001, 2_100_000_000],
        interval_start_ns=500_000_000, interval_end_ns=2_000_000_000,
        era_floor_ns=0, era_end_ns=3_000_000_000)
    ok(not gap["ok"] and gap["status"] == "GAP_OVER_LIMIT",
       "known-bad gap over one second fails")
    left = continuity_from_recv_ns(
        [600_000_000, 1_000_000_000, 2_000_000_000],
        interval_start_ns=500_000_000, interval_end_ns=2_000_000_000,
        era_floor_ns=0, era_end_ns=3_000_000_000)
    ok(not left["ok"] and left["status"] == "NO_LEFT_COVERAGE",
       "known-bad missing left coverage fails")
    right = continuity_from_recv_ns(
        [0, 500_000_000, 1_000_000_000],
        interval_start_ns=500_000_000, interval_end_ns=2_000_000_000,
        era_floor_ns=0, era_end_ns=3_000_000_000)
    ok(not right["ok"] and right["status"] == "NO_RIGHT_COVERAGE",
       "known-bad missing right coverage fails")
    refuses(lambda: continuity_from_recv_ns(
        [0, 1], interval_start_ns=-1, interval_end_ns=2,
        era_floor_ns=0, era_end_ns=3),
        "known-bad interval outside era refuses", "outside era bounds")
    refuses(lambda: continuity_from_recv_ns(
        [0, 2, 1, 3], interval_start_ns=1, interval_end_ns=3,
        era_floor_ns=0, era_end_ns=4),
        "known-bad timestamp regression refuses", "not sorted")
    paths = _hour_paths(
        "BTCUSDT", start_ns=int(1787579390 * 1e9),
        end_ns=int(1787579706 * 1e9))
    ok(paths and all("BTCUSDT" in str(p) for p in paths),
       "real path resolver finds only adjacent BTC hourly files")

    print(f"[de_v2_local_selector] PASS -- {checks} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())
