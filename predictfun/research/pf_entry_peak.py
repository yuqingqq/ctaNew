"""Post-hoc explanation of the largest entry-study inventory excursion."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from .pf_entry_study import DEST, HERE, ARMS, sha, write
from .pf_entry_run import folder, stem


def trace(f, requests, ws, place_ms, fill_ms, expected_peak):
    g = f.copy().reset_index(drop=True)
    if not g.t.is_monotonic_increasing or not g.side.isin([0, 1]).all():
        raise ValueError('invalid fill chronology')
    g['signed_qty'] = g.q * g.side.map({0: 1., 1: -1.})
    g['net_after'] = g.signed_qty.cumsum()
    g['net_before'] = g.net_after - g.signed_qty
    if abs(g.net_after.abs().max() - expected_peak) > 1e-7:
        raise ValueError('peak mismatch')
    # age_ms is venue fill time minus activation time; split activation = send + place_ms.
    g['sent_rel'] = g.t - (g.age_ms + place_ms) / 1000
    for r in g.itertuples():
        hit = requests[(requests.kind == 'place') & (requests.side == r.side)
                       & (requests.lane == r.lane)
                       & ((requests.t_ms - (ws + 1000*r.sent_rel)).abs() < .001)]
        if len(hit) != 1:
            raise ValueError('ambiguous or missing exact placement event')
        if ((g.t + fill_ms/1000 - r.sent_rel).abs() < 1e-7).any():
            raise ValueError('event-priority tie needs engine trace')
    g['known_net_sent'] = [float(g.loc[g.t + fill_ms/1000 < t, 'signed_qty'].sum())
                           for t in g.sent_rel]
    g['reducing_when_sent'] = g.known_net_sent * g.side.map({0: 1., 1: -1.}) < 0
    return g


def controls():
    f = pd.DataFrame(dict(t=[.05, .3], side=[0, 1], q=[10., 25.],
                          age_ms=[10., 60.], lane=[0, 1]))
    r = pd.DataFrame(dict(t_ms=[0., 200.], kind=['place', 'place'], side=[0, 1], lane=[0, 1]))
    x = trace(f, r, 0., 40., 100., 15.)
    assert x.net_after.tolist() == [10., -15.]
    assert x.known_net_sent.tolist() == [0., 10.]
    assert x.reducing_when_sent.tolist() == [False, True]
    for bad_r, bad_peak in [(r.iloc[:1], 15.), (r, 10.)]:
        try:
            trace(f, bad_r, 0., 40., 100., bad_peak)
        except ValueError:
            continue
        raise AssertionError('known-bad trace accepted')
    return dict(positive_cross_zero=True, missing_request_refused=True, wrong_peak_refused=True)


def main():
    control = controls()
    p = pd.read_parquet(DEST/'inventory_paths.parquet')
    selected = p[p.peak_abs_inventory == p.peak_abs_inventory.max()]
    traces = []
    for row in selected.itertuples():
        root = folder(row.day, row.timing)
        paths = {k: root/k/stem(ARMS[row.variant]) for k in ['fills', 'requests', 'win']}
        f, r, w = (pd.read_parquet(paths[k]) for k in ['fills', 'requests', 'win'])
        f, r, w = (v[v.mid == row.mid] for v in [f, r, w])
        if len(w) != 1:
            raise ValueError('market identity')
        summary = json.loads((root/'summary.json').read_text())
        g = trace(f, r, float(w.ws.iloc[0]), summary['order_live_ms'], summary['fill_known_ms'], row.peak_abs_inventory)
        if abs(g.net_after.iloc[-1] - row.net_end) > 1e-7:
            raise ValueError('terminal net mismatch')
        peak_i = int(np.argmax(g.net_after.abs().to_numpy()))
        before = g.iloc[:peak_i+1]
        overshoots = before[(before.net_before * before.net_after < 0) & before.reducing_when_sent]
        traces.append(dict(day=row.day, timing=row.timing, variant=row.variant, mid=int(row.mid),
            source_sha={k: sha(v) for k,v in paths.items()}, summary_sha=sha(root/'summary.json'),
            peak=float(row.peak_abs_inventory), net_end=float(row.net_end),
            cross_zero_reducing_fills=len(overshoots),
            sell_quantity_through_peak=float(before.loc[before.side == 1, 'q'].sum()),
            pending_cancel_sell_quantity_through_peak=float(before.loc[(before.side == 1) & before.zombie, 'q'].sum()),
            all_sells_reducing_when_sent=bool(before.loc[before.side == 1, 'reducing_when_sent'].all()),
            fills=json.loads(g.to_json(orient='records', double_precision=15))))
    out = dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(), kind='post_hoc_peak_explanation',
        selection='all tied global peak inventory paths; explanatory only, no policy selection or inference',
        source_sha=sha(__file__), report_sha=sha(DEST/'report.json'),
        inventory_paths_sha=sha(DEST/'inventory_paths.parquet'), population_paths=len(p),
        selected_paths=len(traces), excluded_selected_paths=0, controls=control, traces=traces)
    write(DEST/'peak_diagnostic.json', out)
    write(HERE.parent/'reviews/entry_budget_20261109/peak_diagnostic.json', out)
    print(json.dumps({k:v for k,v in out.items() if k != 'traces'}, indent=2))
    for x in traces:
        print(json.dumps({k:v for k,v in x.items() if k not in ['fills', 'source_sha']}))


if __name__ == '__main__':
    main()
