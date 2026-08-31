"""O1 new-path BEHAVIORAL tests (R-238 condition i) — v4 code from HEAD, live process untouched.

Drives PMCollector._market against fake websockets. Four paths:
  O1a  connect kwargs carry ping_interval=3, ping_timeout=3 -- this mode
       reproduces WHAT ACTUALLY RAN from 2026-08-30T05:30:01Z and must stay
       exact. The 10/10 rollback lives in `control-v4-slow` under its own era
       identity `clob_v4_1` (Codex COL-R3), not inside this one.
  O1c  silent subscription -> SubscribeUnconfirmed within the bound, distinct
       cause in the gap ledger, reconnect attempted
  O1d  never-connected socket -> gap_start_ns == scope start (not error instant)
  O1b  consecutive failures consume the exponential ladder via reconnect_delay;
       a delivered message RESETS the ladder
Positive control: a healthy fake feed produces rows and NO gap records.
"""
import asyncio, json, sys, time, types
from pathlib import Path

# Extract the v4 collector from GIT (never the working tree, which is HELD at
# v3_1 until the boundary). Ref overridable: O1_REF env, default HEAD.
import os, subprocess, tempfile
_ref = os.environ.get("O1_REF", "HEAD")
_tmp = Path(tempfile.mkdtemp(prefix="o1beh_"))
_code = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
                        "show", f"{_ref}:live/pm_research/collect_pm.py"],
                       capture_output=True, check=True).stdout
(_tmp / "collect_pm_v4.py").write_bytes(_code)
sys.path.insert(0, str(_tmp))
import collect_pm_v4 as C
assert C.COLLECTOR_VERSION == "clob_v4", f"expected clob_v4 at {_ref}, got {C.COLLECTOR_VERSION}"

SCRATCH = Path(__file__).parent

OUT = _tmp / "o1_test_out"
RESULTS = []

def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))

class FakeWS:
    """Async context manager + minimal ws surface."""
    def __init__(self, behavior, captured):
        self.behavior = behavior; self.captured = captured
        self.sent = []
    async def __aenter__(self):
        if self.behavior == "refuse_connect":
            raise OSError("connection refused (fake)")
        return self
    async def __aexit__(self, *a):
        return False
    async def send(self, m):
        self.sent.append(m)
    async def recv(self):
        if self.behavior == "silent":
            await asyncio.sleep(3600)
        if self.behavior == "healthy":
            await asyncio.sleep(0.02)
            return json.dumps({"event_type": "book", "asset_id": "t1"})
        if self.behavior == "one_then_die":
            if not self.captured.get("died"):
                self.captured["died"] = True
                await asyncio.sleep(0.02)
                return json.dumps({"event_type": "book"})
            raise ConnectionResetError("died after first message (fake)")
        raise RuntimeError("unknown behavior")

def fake_connect_factory(script, captured):
    """script: list of behaviors, one per successive connect attempt."""
    calls = {"n": 0}
    def fake_connect(url, **kw):
        captured.setdefault("connect_kwargs", []).append(dict(kw))
        i = min(calls["n"], len(script) - 1); calls["n"] += 1
        return FakeWS(script[i], captured)
    return fake_connect

_test_n = [0]
async def run_market(script, captured, window_s=2, grace_s=0, confirm_s=0.5):
    # redirect every write surface to scratch; compress the clock
    _test_n[0] += 1
    C.RAW = OUT / f"raw{_test_n[0]}"; C.GAP_LEDGER = OUT / f"gaps{_test_n[0]}.jsonl"
    C.ROOT = OUT
    C.WINDOW_S = window_s; C.GRACE_S = grace_s
    C.SUBSCRIBE_CONFIRM_S = confirm_s
    C.websockets = types.SimpleNamespace(connect=fake_connect_factory(script, captured))
    # record reconnect sleeps without waiting them out
    real_sleep = asyncio.sleep
    async def rec_sleep(d):
        if d > 0.4:                       # reconnect_delay sleeps only
            captured.setdefault("delays", []).append(round(d, 2))
            d = 0.01
        await real_sleep(d)
    C.asyncio = types.SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("_")})
    C.asyncio.sleep = rec_sleep
    col = C.PMCollector()
    ts = int(time.time())                  # window opens now, stop_at = ts+window_s
    try:
        await asyncio.wait_for(col._market("btc-updown-5m-%d" % ts, ts, ["t1", "t2"]), timeout=25)
    finally:
        captured["msgs"] = int(col.counts.get("msgs", 0))
        col.disk_pool.shutdown(wait=False); col.http_pool.shutdown(wait=False)
    rows = []
    if C.GAP_LEDGER.exists():
        rows = [json.loads(l) for l in C.GAP_LEDGER.read_text().splitlines()]
    return rows

async def main():
    import shutil
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # ---- POSITIVE CONTROL: healthy feed -> rows recorded, NO gap events ----
    cap = {}
    gaps = await run_market(["healthy"], cap, window_s=1)
    check("POS: healthy feed RECEIVES rows (counts.msgs)", cap.get("msgs", 0) >= 3,
          f"msgs={cap.get('msgs')}")
    check("POS: healthy feed opens no gap", not any(r.get("event") in ("disconnect", "gap_open_at_exit") for r in gaps),
          str([r.get("event") for r in gaps]))
    # O1a on every connect call
    kw = cap["connect_kwargs"][0]
    check("O1a: control-v4 still carries ping_interval=3 — the ROLLBACK "
          "does not mutate this mode, because reproducing the era that ran "
          "requires the bytes that ran (COL-R3)",
          kw.get("ping_interval") == 3, str(kw))
    check("O1a: control-v4 still carries ping_timeout=3",
          kw.get("ping_timeout") == 3, str(kw))

    # ---- O1c: silent subscription -> SUBSCRIBE_UNCONFIRMED + reconnect ----
    cap = {}
    t0 = time.time()
    gaps = await run_market(["silent", "silent"], cap, window_s=2, confirm_s=0.4)
    causes = [r.get("cause") for r in gaps if r.get("event") == "disconnect"]
    check("O1c: silent socket -> SUBSCRIBE_UNCONFIRMED cause", "SUBSCRIBE_UNCONFIRMED" in causes, str(causes))
    check("O1c: reconnect attempted (>=2 connects)", len(cap["connect_kwargs"]) >= 2,
          f"connects={len(cap['connect_kwargs'])}")

    # ---- O1d: never-connected -> gap_start == scope start, not error time ----
    cap = {}
    t_before = time.time_ns()
    gaps = await run_market(["refuse_connect"] * 50, cap, window_s=2)
    t_after_first_err = t_before + int(0.5e9)
    opens = [r for r in gaps if r.get("event") in ("gap_open_at_exit",)]
    check("O1d: never-connected emits gap_open_at_exit", len(opens) == 1, str(len(opens)))
    if opens:
        gs = opens[0]["gap_start_ns"]
        # scope start is captured before the first connect attempt; with the old
        # code (err-instant stamping after ~1s+ of retries) gs would be later.
        check("O1d: gap_start at scope start (within 300ms of task start)",
              abs(gs - t_before) < 0.3e9, f"delta={(gs - t_before)/1e9:.3f}s")

    # ---- O1b: ladder consumed + reset on delivered message ----
    cap = {}
    gaps = await run_market(["refuse_connect", "refuse_connect", "refuse_connect",
                             "one_then_die", "refuse_connect"], cap, window_s=3)
    d = cap.get("delays", [])
    check("O1b: consecutive failures escalate (d2>d1 ladder, jitter-tolerant)",
          len(d) >= 3 and d[1] > d[0] * 1.15, str(d))
    # after 'one_then_die' delivered a message, consec resets -> the next delay
    # drops back toward base (< the escalated pre-reset delay)
    check("O1b: delivered message RESETS the ladder", len(d) >= 4 and d[3] < d[2],
          str(d))

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\nO1 BEHAVIORAL: {len(RESULTS) - n_fail}/{len(RESULTS)} pass")
    return n_fail

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
