"""Descriptive attribution of all frozen latency cells, never a filtered policy.

Declared during the five-day replay, after inspecting the first date only;
per-rung exposure added after the third date, before the last two completed.
All fills remain primary. No selection, new fit, null or inferential claim.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

DAYS = ('20261027', '20261031', '20261101', '20261102', '20261103')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify(df, receipt):
    require(not df.duplicated(['day', 'm', 'cfg', 'arm']).any(), 'duplicate window/cell')
    for cfg, arms in receipt['totals'].items():
        for arm, total in arms.items():
            g = df[(df.cfg == cfg) & (df.arm == arm)]
            require(len(g) == total['windows'], 'window count mismatch')
            for column in ['pnl', 'shares', 'orders', 'pulls']:
                require(abs(float(g[column].sum()) - total[column]) < 1e-6, f'{column} reconciliation')


def verify_frozen(receipt, freeze, day):
    require(receipt['model_version'] == 'v3', 'model version')
    for name, digest in freeze['sources'].items():
        require(receipt['sources'][name] == digest, 'frozen source identity')
    fits = [v['sha256'] for p, v in receipt['inputs'].items() if p.endswith(f'/fit3_{day}.json')]
    require(fits == [freeze['model_fit_hashes'][day]], 'fit changed since freeze')


def selftest():
    df = pd.DataFrame([dict(day='d', m=i, cfg='c', arm='a', pnl=p, shares=10, orders=2, pulls=1)
                       for i, p in enumerate([2., -1.])])
    receipt = dict(totals=dict(c=dict(a=dict(windows=2, pnl=1., shares=20., orders=4., pulls=2.))))
    verify(df, receipt)
    caught = 0
    for bad in [pd.concat([df, df.iloc[:1]]), df.assign(pnl=df.pnl + 1)]:
        try:
            verify(bad, receipt)
        except ValueError:
            caught += 1
    require(caught == 2, 'attribution refusals did not fire')
    freeze = dict(sources={'s.py': 's'}, model_fit_hashes={'d': 'f'})
    receipt = dict(model_version='v3', sources={'s.py': 's'}, inputs={'/fit3_d.json': dict(sha256='f')})
    verify_frozen(receipt, freeze, 'd')
    receipt['inputs']['/fit3_d.json']['sha256'] = 'changed'
    caught = False
    try:
        verify_frozen(receipt, freeze, 'd')
    except ValueError:
        caught = True
    require(caught, 'changed frozen fit was not refused')
    return dict(reconciliation_positive=True, duplicate_refused=True, changed_value_refused=True, frozen_fit_change_refused=True)


def run(src, out):
    controls = selftest()
    freeze_path = Path(__file__).resolve().parents[1] / 'reviews/stop_ev_latency_20261115/freeze_v3.json'
    freeze = json.loads(freeze_path.read_text())
    frames = []
    identities = {}
    for day in DAYS:
        rp = src / day / 'receipt.json'
        wp = src / day / 'windows.parquet'
        receipt = json.loads(rp.read_text())
        require(receipt['day'] == day and not receipt['smoke_only'], 'day identity')
        verify_frozen(receipt, freeze, day)
        frame = pd.read_parquet(wp)
        verify(frame, receipt)
        require(set(frame.day) == {day}, 'window day identity')
        identities[day] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [rp, wp]}
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    cells = {}
    for (cfg, arm), g in df.groupby(['cfg', 'arm'], sort=False):
        shares = float(g.shares.sum())
        orders = float(g.orders.sum())
        pnl = float(g.pnl.sum())
        sorted_pnl = g.pnl.sort_values()
        cells.setdefault(cfg, {})[arm] = dict(
            windows=len(g), trading_windows=int((g.shares > 0).sum()),
            positive_windows=int((g.pnl > 0).sum()), negative_windows=int((g.pnl < 0).sum()),
            live_share_by_rung={str(j): float(g[f'live{j}'].mean()) for j in range(6)},
            primary_pnl=pnl, shares=shares, orders=orders,
            pulls_per_order=float(g.pulls.sum()) / orders if orders else None,
            young_share_fraction=float(g.shares_young.sum()) / shares if shares else None,
            young_pnl=float((g.pnl - g.pnl_old).sum()),
            ws_pnl=float(g.ws.sum()), estimated_pnl=float(g.e2.sum()),
            imputed_pnl=float(g.imputed_pnl.sum()),
            unknown_queue_pnl=float(g.qunk.sum()), pending_cancel_pnl=float(g.pend.sum()),
            imputed_share_fraction=float(g.imputed_shares.sum()) / shares if shares else None,
            unknown_queue_share_fraction=float(g.qunk_shares.sum()) / shares if shares else None,
            pending_cancel_share_fraction=float(g.pend_shares.sum()) / shares if shares else None,
            markout1_cents_per_share=float(g.markout1_cents.sum()) / shares if shares else None,
            markout5_cents_per_share=float(g.markout5_cents.sum()) / shares if shares else None,
            best_window_pnl=float(sorted_pnl.iloc[-1]), worst_window_pnl=float(sorted_pnl.iloc[0]),
            best_five_window_pnl=float(sorted_pnl.iloc[-5:].sum()),
            worst_five_window_pnl=float(sorted_pnl.iloc[:5].sum()),
            pnl_less_best_window=pnl - float(sorted_pnl.iloc[-1]),
            pnl_less_best_five_windows=pnl - float(sorted_pnl.iloc[-5:].sum()),
            mean_terminal_abs_position=float(g.terminal_abs_position.mean()),
            max_abs_position=float(g.max_abs_position.max()))
    result = dict(asof=datetime.now(timezone.utc).isoformat(), controls=controls,
                  freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
                  input_hashes=identities, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cells=cells, interpretation='Attribution only, all fills primary; overlapping source/age/queue splits. '
                  'Tail summaries do not define a filtered strategy. Markouts use recorded anchors, not M-shifted fill times. '
                  'No significance, matched-retention or causal skill claim.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(asof=result['asof'], controls=controls, output=str(out))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--src', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.selftest:
        print(json.dumps(selftest()))
    else:
        run(a.src, a.out)
