"""Measured Tokyo clock ('tokyoreal'), from the exec-port probe/latency runs of 2026-11-02 (pmmmrust fe073d1,
docs/receipts/20261102_pf_tokyo_probe_and_latency_results.json, reference run 20261102T162608Z):
  Predict book: received updateTimestampMs + 53.1 ms median (min 13.2, p90 80.1, p99 114.7)
  Binance bookTicker: received E + 1.5 ms median
  decision (reorder hold + tick): ~10 ms;  create round trip p50 44 ms -> order live ~22 ms after it is sent
Per-frame Predict knowledge time = this research host's receipt shifted so the day's median delay equals Tokyo's
53.1 ms (keeps the real frame-to-frame jitter; this host's tail is slightly heavier: p99 +77 vs +62 ms around the
median, so conservative), never earlier than updateTimestampMs + 13.2 ms, plus the 10 ms decision time.
Overrides (2026-11-03, exec-port correction from the probe's wallet-event stamps): PF_RL_DDEC = decision time (default 10;
2 = the exec port's reorder_ms 0 / advance_ms 1 config); PF_RL_LPLACE / PF_RL_LCANCEL = order live / cancel effective after
sending (measured p50 58 / 37 ms, clock-offset lower bound 46 / 25); when set, the replay uses its split-latency handler
(a cancel sent before the order is live takes effect L_CANCEL after it goes live, or PF_RL_LPENDCANCEL after the create
was sent when set); PF_RL_LFILL = own fill known after (default 20 + decision). Defaults reproduce latency test L1.
The per-day shift is computed once by pf_step5_features_rl.py and stored in step1/<day>/clock_rl.json; the replay
refuses to run without it, so features and replay use the same clock.
"""
import json, os
import numpy as np

PF_P50, PF_MIN, BN_IN = 53.1, 13.2, 1.5
D_DEC = float(os.environ.get('PF_RL_DDEC', '10'))
L_ORDER = 22.0                         # L1: order/cancel live 22 ms after sending (superseded by the split values below)
L_FILL = float(os.environ.get('PF_RL_LFILL', 20.0 + D_DEC))   # own fill known (L1: 20 ms + decision; wallet events ~100 ms)
L_PLACE = float(os.environ['PF_RL_LPLACE']) if os.environ.get('PF_RL_LPLACE') else None
L_CANCEL = float(os.environ['PF_RL_LCANCEL']) if os.environ.get('PF_RL_LCANCEL') else None
L_PENDCANCEL = float(os.environ['PF_RL_LPENDCANCEL']) if os.environ.get('PF_RL_LPENDCANCEL') else None
                                       # a remove that reaches Predict before the order is live takes effect this long after the
                                       # CREATE was sent (probe: answered 95-135 ms after it, p50 99; Predict holds it)
L_ANSWER = 44.0                        # create answer round trip p50 (for the throttle's unanswered-place count)
SFX = '' if D_DEC == 10.0 else f'_d{int(D_DEC)}'   # rows_rl{SFX}.parquet, clock_rl{SFX}.json, preds_rl_a{SFX}
RELAY_NET = 72.0                       # relay (recorded on this host): receipt - 72 ms, as before


def day_shift(t_ms, recv_ns):
    """this host's median (receipt - update) minus Tokyo's 53.1 ms."""
    return float(np.median(np.asarray(recv_ns, np.float64) / 1e6 - np.asarray(t_ms, np.float64)) - PF_P50)


def pf_known(t_ms, recv_ns, shift):
    t = np.asarray(t_ms, np.float64)
    return np.maximum(t + PF_MIN, np.asarray(recv_ns, np.float64) / 1e6 - shift) + D_DEC


def load_shift(s1_dir):
    p = f'{s1_dir}/clock_rl{SFX}.json'
    if not os.path.exists(p):
        raise SystemExit(f'REFUSED: {p} missing (run pf_step5_features_rl first; features and replay share one clock)')
    c = json.load(open(p))
    if abs(c['d_dec'] - D_DEC) > 1e-9:
        raise SystemExit(f'REFUSED: {p} was built with decision time {c["d_dec"]} ms, this run uses {D_DEC}')
    return c['shift_ms']
