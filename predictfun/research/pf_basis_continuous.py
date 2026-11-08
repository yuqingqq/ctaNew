"""continuous_v1 basis samples from a known-time event stream, independent of Rust.

One applied relay event yields one ratio against the latest valid Binance mid
already applied. Equal timestamps follow stream order; equal prices remain
distinct. No Binance yet is an explicit exclusion. File/hour boundaries never
create samples. This module changes no clock or frozen research artifact.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = 'PF_CONTINUOUS_BASIS_SAMPLES_V1'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def samples(events):
    rows, counts = [], Counter()
    last, mid = -math.inf, None
    first = None
    for seq, event in enumerate(events, 1):
        if len(event) != 1:
            raise ValueError('one event variant required')
        (kind, v), = event.items()
        counts['inputs'] += 1
        t = v.get('known_ms', v.get('now_ms'))
        if t is None:
            if kind != 'OpenWindow':
                raise ValueError('missing known time')
            continue
        if not math.isfinite(t) or t < last:
            raise ValueError('non-finite or out-of-order known time')
        first = t if first is None else first
        last = t
        if kind == 'RelaySample':
            raise ValueError('raw reference events required, not precomputed ratios')
        if kind == 'BinanceBook':
            b, a = v['bid'], v['ask']
            if not (math.isfinite(b) and math.isfinite(a)):
                raise ValueError('non-finite Binance price')
            if b > 0 and a > b:
                mid = (b + a) / 2.
            else:
                counts['invalid_binance'] += 1
        elif kind == 'Relay':
            counts['relay_events'] += 1
            p = v['price']
            if not math.isfinite(p) or p <= 0:
                raise ValueError('invalid relay price')
            if mid is None:
                counts['no_binance'] += 1
            else:
                rows.append((t, p / mid, seq))
                counts['sampled'] += 1
    return rows, dict(counts), first, (last if first is not None else None)


def build(input_path, expected_sha, out):
    import numpy as np
    input_path, out = Path(input_path), Path(out)
    if sha(input_path) != expected_sha:
        raise ValueError('capture SHA-256 mismatch')
    with gzip.open(input_path, 'rt') as f:
        header = json.loads(next(f))
        if header['protocol'] != 'PFMM_LADDER_JOURNAL_V3':
            raise ValueError('unexpected journal protocol')
        rows, counts, first, last = samples(json.loads(line) for line in f)
    if not rows:
        raise ValueError('empty basis sample population')
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'samples.npz', t=np.array([r[0] for r in rows], np.float64),
                        ratio=np.array([r[1] for r in rows], np.float64),
                        input_index=np.array([r[2] for r in rows], np.int64))
    report = dict(protocol=PROTOCOL, basis_mode='continuous_v1', as_of_utc=datetime.now(timezone.utc).isoformat(),
                  input_path=str(input_path.resolve()), input_sha256=expected_sha, header=header,
                  samples_sha256=sha(out/'samples.npz'), builder_sha256=sha(__file__),
                  counts=counts, first_known_ms=first, last_known_ms=last)
    (out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def load(out):
    import numpy as np
    out = Path(out)
    report = json.loads((out/'manifest.json').read_text())
    if report['protocol'] != PROTOCOL or report['basis_mode'] != 'continuous_v1':
        raise ValueError('basis sample version mismatch')
    if sha(out/'samples.npz') != report['samples_sha256']:
        raise ValueError('basis sample SHA-256 mismatch')
    with np.load(out/'samples.npz', allow_pickle=False) as f:
        t, v, i = f['t'], f['ratio'], f['input_index']
    if (len(t) != report['counts']['sampled'] or len(t) != len(v) or len(t) != len(i)
            or not len(t) or not np.isfinite(t).all() or not np.isfinite(v).all()
            or (np.diff(t) < 0).any() or (np.diff(i) <= 0).any()):
        raise ValueError('invalid basis sample population/order')
    return t, v


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(json.dumps(build(a.inputs, a.sha256, a.out)))
