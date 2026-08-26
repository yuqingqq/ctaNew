"""ADDITIVE smoke test for the btc token-shard fix. Touches nothing live.

SURFACE AUTHORISATION (R-126, in-file): R-179 step (2) — verify the one
semantic the BTC_GAP_DIAGNOSIS_2026-08-26 memo lists as UNVERIFIED, namely that
a SINGLE-ASSET subscription is honoured by the venue and carries roughly half
the per-connection bytes of a full-market subscription.

WHY THIS IS A STANDALONE PROBE AND NOT A PATCH TO collect_pm.py.
`pm-collector-clob.service` runs `live/pm_research/collect_pm.py` directly with
`Restart=always`. For that unit the repo file IS the deployment artifact: an
in-place edit does not stage a patch, it ARMS one, to fire on the next process
death with nobody deciding — possibly mid-day-one. Readiness is therefore built
here, where nothing executes on a restart.

HONEST RISK NOTE. This opens a Polymarket WS connection from the SAME HOST AND
IP as the live collector. The memo is HIGH-confidence the bottleneck is remote
and per-connection but only MEDIUM on venue-infra vs network-path, and a
per-IP component is NOT ruled out by its evidence. One extra connection against
the hundreds the collector already opens is marginal, so the risk is small --
but it is not zero, which is why this runs BEFORE day-one tape starts accruing
and never opens two connections at once.

DESIGN: A -> B -> A on ONE market, sequentially.
  A = subscribe [up_token] only.   B = subscribe [up, down] (the current live
  shape). Activity rises through a 5-minute window, so a single A-then-B would
  confound the byte reduction with the trend. Running A twice and bracketing B
  lets the trend be seen and averaged out instead of being reported as the
  effect.

    python3 live/pm_research/btc_shard_probe.py --selftest
    python3 live/pm_research/btc_shard_probe.py run [--secs 20]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import websockets
import flow_intensity as fi

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def current_btc_market() -> tuple[str, str, str, int]:
    """(slug, up_token, down_token, window_start) for a LIVE btc window."""
    tm = fi.token_map()
    now = int(time.time())
    t0 = (now // 300) * 300
    for k in (t0, t0 - 300):
        slug = f"btc-updown-5m-{k}"
        if slug in tm:
            up, dn = tm[slug]
            return slug, up, dn, k
    raise SystemExit("no live btc market in the token map; refusing to guess")


async def sample(assets: list[str], secs: float, label: str) -> dict:
    """One connection, one subscription, fixed duration. Never concurrent."""
    stats = {
        "label": label, "n_assets": len(assets), "secs": secs,
        "msgs": 0, "bytes": 0, "frames": 0,
        "by_event": collections.Counter(), "by_asset": collections.Counter(),
        "error": None,
    }
    t_end = time.time() + secs
    try:
        async with websockets.connect(WS_URL, ping_interval=10, ping_timeout=10,
                                      open_timeout=15, max_queue=2 ** 16) as ws:
            await ws.send(json.dumps({"assets_ids": assets, "type": "market"}))
            while time.time() < t_end:
                try:
                    raw = await asyncio.wait_for(ws.recv(),
                                                 timeout=max(0.1, t_end - time.time()))
                except asyncio.TimeoutError:
                    break
                stats["frames"] += 1
                stats["bytes"] += len(raw if isinstance(raw, (bytes, str)) else b"")
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                for m in (payload if isinstance(payload, list) else [payload]):
                    if not isinstance(m, dict):
                        continue
                    stats["msgs"] += 1
                    stats["by_event"][str(m.get("event_type"))] += 1
                    aid = str(m.get("asset_id"))
                    if aid and aid != "None":
                        stats["by_asset"][aid] += 1
                    for pc in m.get("price_changes", []) or []:
                        a = str(pc.get("asset_id"))
                        if a:
                            stats["by_asset"][a] += 1
    except Exception as ex:            # a failed phase is a STATUS, not a crash
        stats["error"] = f"{type(ex).__name__}: {ex}"
    stats["by_event"] = dict(stats["by_event"])
    stats["by_asset"] = dict(stats["by_asset"])
    stats["bytes_per_s"] = round(stats["bytes"] / secs, 1) if secs else None
    return stats


async def _run(secs: float, n_reps: int = 3) -> dict:
    slug, up, dn, ws_start = current_btc_market()
    age = int(time.time()) - ws_start
    # PRECONDITION, learned the hard way: the first interleaved run picked a
    # market already near its end and measured it DYING -- B3 saw 14 frames and
    # the price_change ratio came out BACKWARDS (0.65), which is the tell. An
    # instrument that does not check its own preconditions reports the death of
    # a market as a property of a subscription shape. Refuse instead.
    need = n_reps * 2 * secs + 15
    remaining = 300 - age
    if remaining < need:
        raise SystemExit(
            f"REFUSING: {slug} has {remaining}s left but the plan needs "
            f"{need:.0f}s. Phases would straddle the window end and measure "
            f"the market expiring, not the subscription. Wait for a fresh "
            f"window (next boundary in {remaining}s).")
    print(f"market {slug} (age {age}s, {remaining}s left, plan needs "
          f"{need:.0f}s)  up={up[:14]}...", flush=True)
    # INTERLEAVED, not A-then-B. The first run of this probe measured 66%
    # drift between two IDENTICAL single-asset phases -- comparable to the ~50%
    # effect being tested. A naive A-then-B on that data would have reported
    # 2.13x and "confirmed" the memo's prediction almost entirely out of the
    # market quieting through the window. Alternating decorrelates the two.
    plan = []
    for i in range(n_reps):
        plan.append((f"A{i+1}_single_up", [up]))
        plan.append((f"B{i+1}_full_both", [up, dn]))
    phases = []
    for label, assets in plan:
        print(f"  phase {label} ({len(assets)} asset(s)) for {secs}s ...",
              flush=True)
        phases.append(await sample(assets, secs, label))
    a = [p for p in phases if p["label"].startswith("A")]
    bs = [p for p in phases if p["label"].startswith("B")]
    b = bs[0]
    a_bps = sum(p["bytes_per_s"] for p in a) / len(a)
    b_bps = sum(p["bytes_per_s"] for p in bs) / len(bs)
    ratio = round(b_bps / a_bps, 3) if a_bps else None
    # PAIRED ratios: each B against the mean of its neighbouring A phases, so a
    # monotone within-window trend cancels instead of being reported as effect.
    paired = []
    for i, bp in enumerate(bs):
        nb = [a[j]["bytes_per_s"] for j in (i, i + 1) if j < len(a)]
        if nb:
            paired.append(bp["bytes_per_s"] / (sum(nb) / len(nb)))
    av = [p["bytes_per_s"] for p in a]
    drift = (round((max(av) - min(av)) / max(1.0, a_bps), 3) if len(av) > 1
             else None)
    pc_a = sum(p["by_event"].get("price_change", 0) for p in a) / max(1, len(a))
    pc_b = sum(p["by_event"].get("price_change", 0) for p in bs) / max(1, len(bs))
    up_seen = sum(p["by_asset"].get(up, 0) for p in a)
    dn_seen_in_a = sum(p["by_asset"].get(dn, 0) for p in a)
    return {
        "probe": "btc_shard_probe_v1", "slug": slug, "market_age_s": age,
        "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phases": phases,
        "single_asset_bytes_per_s_mean": round(a_bps, 1),
        "full_market_bytes_per_s_mean": round(b_bps, 1),
        "paired_ratios": [round(x, 3) for x in paired],
        "paired_ratio_mean": (round(sum(paired) / len(paired), 3)
                              if paired else None),
        "price_change_msgs_A_mean": round(pc_a, 1),
        "price_change_msgs_B_mean": round(pc_b, 1),
        "price_change_ratio_B_over_A": (round(pc_b / pc_a, 3) if pc_a else None),
        "full_over_single_ratio": ratio,
        "A_phase_drift_fraction": drift,
        "predicates": {
            "single_asset_subscription_delivers": up_seen > 0,
            "single_asset_carries_up_token_events": up_seen > 0,
            "down_token_absent_from_single_phase": dn_seen_in_a == 0,
            "full_market_ratio_near_2x": (
                ratio is not None and 1.5 <= ratio <= 2.5),
            "drift_smaller_than_effect": (
                drift is not None and drift < 0.5),
            "no_phase_errored": all(p["error"] is None for p in phases),
        },
        "counts": {"up_events_in_A": up_seen, "down_events_in_A": dn_seen_in_a},
    }


def _selftests() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            raise AssertionError(f"selftest failed: {label}")

    # the probe must never open two connections at once -- that is the whole
    # basis of the "marginal added load" judgement.
    src = Path(__file__).read_text()
    ok("gather" not in src.split("def _selftests")[0],
       "no asyncio.gather in the probe path -- phases are strictly sequential")
    body = src.split("def _selftests")[0]
    ok(body.count("websockets.connect") == 1,
       "exactly one connect site in the probe path, used serially -- counted "
       "in the BODY only, because the first version counted its own search "
       "string and failed on itself (same self-matching defect as the "
       "policy-state guard earlier today)")
    ok("_single_up" in src and "_full_both" in src and "n_reps" in src,
       "the design INTERLEAVES A and B, so within-window drift cancels rather "
       "than being reported as the effect (measured 66% drift on the first run)")
    ok(WS_URL.startswith("wss://"), "the venue endpoint is TLS")
    # never writes to the tape
    ok("open(" not in src.split("def _selftests")[0].replace("open_timeout", ""),
       "the probe opens no files -- it cannot touch the live tape")
    print(f"btc_shard_probe selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    rep = asyncio.run(_run(a.secs, a.reps))
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if all(rep["predicates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
