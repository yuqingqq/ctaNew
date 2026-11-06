"""Diagnostic for early replacements after live-order fills.

Missing or mismatched run artifacts are REFUSED, never counted as zero. This
diagnostic does not certify throttle causality; paired request-stream tests do.
Usage: python3 -m predictfun.research.pf_view_check <physical run> <known run> <day> [L_FILL ms]
"""
import sys, os, json
import numpy as np
import pandas as pd
from . import pf_io as io


def arm_stem(name):
    return name.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-') + '.parquet'


def read_run(run, day):
    base = f'{io.OUT}/step4/{run}/{day}'
    try:
        with open(f'{base}/summary.json') as fh:
            summary = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f'{base}: missing or invalid summary') from exc
    if not isinstance(summary, dict):
        raise ValueError(f'{base}: summary is not an object')
    arms = summary.get('arms')
    if summary.get('day') != day or not isinstance(arms, dict) or not arms:
        raise ValueError(f'{base}: wrong day or empty arm manifest')
    manifest = {arm_stem(name): value for name, value in arms.items()}
    if len(manifest) != len(arms):
        raise ValueError(f'{base}: arm filenames collide')
    for part in ('fills', 'requests', 'win'):
        folder = f'{base}/{part}'
        if not os.path.isdir(folder):
            raise ValueError(f'{folder}: missing directory')
        found = {name for name in os.listdir(folder) if name.endswith('.parquet')}
        if found != set(manifest):
            raise ValueError(f'{folder}: incomplete or extra arm artifacts')
    frames = {}
    for stem, value in manifest.items():
        if not isinstance(value, dict):
            raise ValueError(f'{base}/{stem}: invalid arm manifest')
        f = pd.read_parquet(f'{base}/fills/{stem}')
        q = pd.read_parquet(f'{base}/requests/{stem}')
        w = pd.read_parquet(f'{base}/win/{stem}', columns=['mid', 'ws'])
        if not {'mid', 't', 'zombie'} <= set(f) or not {'mid', 't_ms', 'kind'} <= set(q):
            raise ValueError(f'{base}/{stem}: missing event columns')
        for panel in (f, q):
            if 'lane' not in panel:
                if 'side' not in panel:
                    raise ValueError(f'{base}/{stem}: missing lane and side')
                panel['lane'] = panel.side
            if not panel.lane.isin([0, 1, 2, 3]).all():
                raise ValueError(f'{base}/{stem}: invalid lane')
        if len(w) == 0 or w.mid.isna().any() or w.mid.duplicated().any() or not np.isfinite(w.ws).all():
            raise ValueError(f'{base}/{stem}: invalid window identities')
        if len(f) != value.get('fills') or len(w) != value.get('windows'):
            raise ValueError(f'{base}/{stem}: manifest row counts disagree')
        counts = value.get('live', {}).get('requests')
        if not isinstance(counts, dict) or q.kind.value_counts().to_dict() != {k: v for k, v in counts.items() if v}:
            raise ValueError(f'{base}/{stem}: request counts disagree')
        if not q.kind.isin(['place', 'cancel', 'amend']).all() or not f.zombie.isin([True, False]).all():
            raise ValueError(f'{base}/{stem}: invalid event status')
        f['zombie'] = f.zombie.astype(bool)
        if not np.isfinite(f.t.to_numpy(dtype=float)).all() or not np.isfinite(q.t_ms.to_numpy(dtype=float)).all():
            raise ValueError(f'{base}/{stem}: non-finite event time')
        if not f.mid.isin(w.mid).all() or not q.mid.isin(w.mid).all():
            raise ValueError(f'{base}/{stem}: event references unknown window')
        f['tm'] = f.mid.map(w.set_index('mid').ws) + f.t * 1000.0
        frames[stem] = (f, q, w.sort_values('mid').reset_index(drop=True))
    return frames


def count_early(frames, lfill):
    out = {}
    for stem, (f, q, w) in frames.items():
        n = n_pl = 0
        for (mid, lane), requests in q.groupby(['mid', 'lane']):
            fills = np.sort(f.loc[(f.mid == mid) & (f.lane == lane) & ~f.zombie, 'tm'].values)
            if not len(fills):
                continue
            places = requests.loc[requests.kind == 'place', 't_ms'].values
            cancels = np.sort(requests.loc[requests.kind == 'cancel', 't_ms'].values)
            for tp in places:
                j = np.searchsorted(fills, tp, side='right') - 1
                if j < 0 or tp - fills[j] >= lfill:
                    continue
                n_pl += 1
                k = np.searchsorted(cancels, fills[j], side='left')
                if not (k < len(cancels) and cancels[k] <= tp):
                    n += 1
        out[stem] = dict(windows=len(w), fills=len(f), requests=len(q),
                         places_within_lfill_of_a_fill=n_pl, early_replaces=n)
    return out


def early(run, day, lfill):
    if not np.isfinite(lfill) or lfill <= 0:
        raise ValueError('fill notification delay must be finite and positive')
    return count_early(read_run(run, day), lfill)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) not in (3, 4):
            raise ValueError('expected physical run, known run, day, and optional fill delay')
        a, b, day = args[:3]
        lfill = float(args[3]) if len(args) == 4 else 100.0
        if not np.isfinite(lfill) or lfill <= 0:
            raise ValueError('fill notification delay must be finite and positive')
        fa, fb = read_run(a, day), read_run(b, day)
        if set(fa) != set(fb):
            raise ValueError('compared runs have different arm identities')
        for stem in fa:
            if not fa[stem][2].equals(fb[stem][2]):
                raise ValueError(f'{stem}: compared runs have different window identities')
        ea, eb = count_early(fa, lfill), count_early(fb, lfill)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        return 2
    na = sum(v['early_replaces'] for v in ea.values())
    nb = sum(v['early_replaces'] for v in eb.values())
    for stem in ea:
        print(stem[-48:], 'physical', ea[stem], '| known', eb[stem])
    print(f'early re-places: physical {na}, known {nb}')
    ok = na > 0 and nb == 0
    print('VIEW CHECK', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
