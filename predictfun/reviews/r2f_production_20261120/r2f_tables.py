"""R2-F receipt tables: markdown copies of r2f_checks.py's outputs (checks.json and the CSVs in a directory), so that every number in the
receipt is copied by code, not by hand. A formatter only: it computes no predicate. Each number is a value read from an output, a sum or
ratio of such values, or a count of rows.
Usage: r2f_tables.py [dir] (default: this directory) -> markdown on stdout"""
import csv
import json
import sys
from pathlib import Path

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent


def rows(name):
    with open(D / name, newline='') as f:
        return list(csv.DictReader(f))


def f_(x, nd=2):
    if x in (None, '', 'None'):
        return '—'
    v = float(x)
    return f'{int(v):,}' if v.is_integer() else f'{v:,.{nd}f}'


def table(head, body):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in body])


def pct(a, b):
    return f'{100. * a / b:.1f} %' if b else '—'


def verdict(J):
    """The receipt's verdict lines: the checks' own all_true and the failing names, then counts read from y_summary.csv (day both, head
    both) and population.csv. No threshold, no adjective: every word that carries a number is a value or a ratio of values."""
    ck = J['checks']; bad = [k for k, v in sorted(ck.items()) if not v]
    print('### Verdict\n')
    print(f"- **Declared checks A–H: all_true = {str(J['all_true']).lower()}**" + (f"; false: {', '.join(bad)}." if bad else f" ({len(ck)} of {len(ck)} true)."))
    y = {(r['day'], r['head']): r for r in rows('y_summary.csv')}[('both', 'both')]
    n, z, ps, ng = (int(float(y[k])) for k in ('n_ok', 'zero', 'positive', 'negative'))
    pop = rows('population.csv'); wr = sum(int(r['n']) for r in pop if r['item'] == 'worlds_realized')
    print(f"- **Labels (both days, both heads):** {n:,} ok pairs; **{z:,} exactly 0 c ({pct(z, n)})**; {ps:,} positive and {ng:,} negative "
          f"({pct(ps + ng, n)} nonzero).")
    print(f"- The nonzero labels come from {int(float(y['originals_nonzero_union'])):,} of the {int(float(y['originals'])):,} originals with an ok pair "
          f"({pct(float(y['originals_nonzero_union']), float(y['originals']))}) and {int(float(y['worlds_nonzero'])):,} of {wr:,} realized worlds "
          f"({pct(float(y['worlds_nonzero']), wr)}). The 5 largest |Y| carry {100. * float(y['top5_abs_share']):.1f} % of all |Y|; the top 1 % "
          f"carry {100. * float(y['top1pct_abs_share']):.1f} %.")
    print(f"- Y range {f_(y['min_c'])} to {f_(y['max_c'])} c; mean {f_(y['mean_c'], 3)} c; median {f_(y['median_c'])} c.\n")


def main():
    J = json.loads((D / 'checks.json').read_text()); ck, rd = J['checks'], J['readings']
    verdict(J)
    print('### Checks\n'); print(table(['check', 'value'], [[k, str(v).lower()] for k, v in sorted(ck.items())] + [['all_true', str(J['all_true']).lower()]]))
    acc = J['accepted']
    print(f"\naccepted: dataset_sha256 `{acc['dataset_sha256']}`; rows {acc['rows']:,}; pairs {acc['pairs']}; audit_every {acc['audit_every']}; plan_sha256 `{J['plan_sha256']}`\n")
    c = rd['causal']
    print('### Audit and binding totals\n'); print(table(list(c), [[json.dumps(v) if isinstance(v, dict) else f_(v) for v in c.values()]]))
    print('\n### Controls on real rows (every partition)\n')
    ctl = rd['controls']; names = sorted(ctl[0]['refused']) if ctl else []
    print(table(['control', 'refused on'], [[n, f"{sum(1 for x in ctl if x['refused'][n])} / {len(ctl)}"] for n in names]
                + [['the intact part valid', f"{sum(1 for x in ctl if x['intact'])} / {len(ctl)}"]]))
    pop = rows('population.csv'); days = sorted({r['day'] for r in pop})
    print('\n### Population\n')
    items = ['worlds_planned', 'worlds_realized', 'markets', 'utc_hours', 'originals', 'picks', 'pairs_run', 'ok']
    g = {(r['day'], r['item']): r['n'] for r in pop}
    print(table(['item'] + days + ['both'], [[i] + [f_(g[(d, i)]) for d in days] + [f_(sum(int(g[(d, i)]) for d in days))] for i in items]))
    st = {}
    for r in pop:
        if r['item'].startswith('side '):
            _, sd, hd, src, s_ = r['item'].split(' ', 4); st.setdefault((hd, s_), {}).setdefault(r['day'], 0); st[(hd, s_)][r['day']] += int(r['n'])
    print('\nStatuses by head (both sides, every source):\n')
    print(table(['head', 'status'] + days + ['both'], [[h, s_] + [f_(v.get(d, 0)) for d in days] + [f_(sum(v.values()))] for (h, s_), v in sorted(st.items())]))
    src = {}
    for r in pop:
        if r['item'].startswith('side '):
            _, sd, hd, s0, s_ = r['item'].split(' ', 4)
            if s_ == 'ok':
                src.setdefault((hd, s0), {}).setdefault(r['day'], 0); src[(hd, s0)][r['day']] += int(r['n'])
    print('\nok rows by head and source:\n')
    print(table(['head', 'source'] + days + ['both'], [[h, s0] + [f_(v.get(d, 0)) for d in days] + [f_(sum(v.values()))] for (h, s0), v in sorted(src.items())]))
    print('\n### Y (cents; ok pairs)\n')
    ys = rows('y_summary.csv')
    print(table(['day', 'head', 'n_ok', 'zero', '> 0', '< 0', 'nonzero originals (union)', 'nonzero worlds', 'originals', 'mean', 'median', 'p01', 'p05', 'p95', 'p99', 'min', 'max',
                 'top-5 share', 'top-1 % share'],
                [[r['day'], r['head'], f_(r['n_ok']), f_(r['zero']), f_(r['positive']), f_(r['negative']), f_(r['originals_nonzero_union']), f_(r['worlds_nonzero']), f_(r['originals']),
                  f_(r['mean_c']), f_(r['median_c']), f_(r['p01_c']), f_(r['p05_c']), f_(r['p95_c']), f_(r['p99_c']), f_(r['min_c']), f_(r['max_c']), f_(r['top5_abs_share'], 3),
                  f_(r['top1pct_abs_share'], 3)] for r in ys]))
    print('\n### Fill paths (ok pairs)\n')
    pt = rows('paths.csv'); cls = sorted({r['class'] for r in pt}); dh = sorted({(r['day'], r['head']) for r in pt})
    pg = {(r['day'], r['head'], r['class']): int(r['ok_pairs']) for r in pt}
    print(table(['day', 'head'] + cls, [[d, h] + [f_(pg.get((d, h, c_), 0)) for c_ in cls] for d, h in dh]))
    print('\n### Arms (sums over ok pairs)\n')
    ar = rows('arms.csv'); base = ['rows', 'value_sum', 'cash_sum', 'position_sum', 'inventory_nonzero_rows', 'fills', 'filled_qty', 'generations', 'open_at_end_rows', 'teacher_calls']
    print(table(['day', 'head', 'arm'] + base, [[r['day'], r['head'], r['arm']] + [f_(r[k]) for k in base] for r in ar]))
    extra = [k for k in ar[0] if k not in base + ['day', 'head', 'arm']] if ar else []
    print('\nTeacher actions and statuses (calls):\n')
    print(table(['day', 'head', 'arm'] + extra, [[r['day'], r['head'], r['arm']] + [f_(r[k]) for k in extra] for r in ar]))
    print('\n### Coverage of the ok rows (absent categories are explicit zeros)\n')
    cv = rows('coverage.csv'); dims = list(dict.fromkeys(r['dimension'] for r in cv))
    for dim in dims:
        vals = list(dict.fromkeys(r['value'] for r in cv if r['dimension'] == dim)); dh = list(dict.fromkeys((r['day'], r['head']) for r in cv if r['dimension'] == dim))
        cg = {(r['day'], r['head'], r['value']): r['ok_rows'] for r in cv if r['dimension'] == dim}
        print(f'\n{dim}:\n'); print(table(['day', 'head'] + vals, [[d, h] + [f_(cg.get((d, h, v), 0)) for v in vals] for d, h in dh]))
    print('\n### Weights\n')
    wt = rows('weights.csv')
    print(table(['day', 'head', 'picked', 'raw weight total', 'ok', 'ok weight total', 'weighted mean Y (c)', 'ESS', 'max weight share', 'top-10 % weight share', 'raw mean Y (c)'],
                [[r['day'], r['head'], f_(r['picked']), f_(r['raw_weight_total'], 1), f_(r['ok']), f_(r['ok_weight_total'], 1), f_(r['weighted_mean_Y_c'], 3), f_(r['ess'], 1),
                  f_(r['max_weight_share'], 4), f_(r['top10pct_weight_share'], 3), f_(r['raw_mean_Y_c'], 3)] for r in wt]))
    print('\n### Cost\n'); print(table(['item', 'value'], [[r['item'], f_(r['value'], 3)] for r in rows('cost.csv')]))
    print(f"\nstops: {len(rd['stops'])}; records as of {rd['asof_records']}; checks as of {J['asof']}")


if __name__ == '__main__':
    main()
