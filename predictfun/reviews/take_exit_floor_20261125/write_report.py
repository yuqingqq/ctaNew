"""TAKE-EXIT FLOOR v1: report.md from readings.json, checks.json and the run receipt (every number read from those files).
Usage: PF_SLOT=claude predictfun/research/run_light.sh floor_write_report -- python3 predictfun/reviews/take_exit_floor_20261125/write_report.py
"""
import json
import subprocess
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[2]


def usd(x):
    return '—' if x is None else f'{x:+,.2f}'


def num(x, d=1):
    return '—' if x is None else f'{x:,.{d}f}'


def pct(x):
    return '—' if x is None else f'{100 * x:.1f} %'


def main():
    rd = json.loads((H / 'readings.json').read_text())
    ck = json.loads((H / 'checks.json').read_text())
    pj = json.loads((H / 'projection.json').read_text())
    fz = subprocess.check_output(['git', 'log', '-1', '--format=%h', '--', str(H / 'floor_replay.py')], cwd=R, text=True).strip()
    b, f, d = rd['base'], rd['floor'], rd['floor_minus_base']
    L = [f"# TAKE-EXIT FLOOR v1 — break-even floor on the maker exit, valued at settlement ({rd['asof']})", '',
         f"[amendment.md](amendment.md) (declared before the run), code frozen at {fz}. {rd['markets']} complete markets on "
         f"{', '.join(rd['days'])} (the v3 days; consumed, development evidence). BASE = the saved v3 STRICT58 ledgers; FLOOR = STRICT58 "
         "with the exit never offered beyond break-even (average cost incl. taker fees). Shares open at END (298 s) settle on the "
         "market's resolved outcome.", '',
         '## Money split (USD, all markets)', '',
         '| | BASE | FLOOR | FLOOR − BASE |', '|---|---:|---:|---:|']
    rows = [('Round trips (spread capture), after fees', 'round_trips'), ('… closed by the maker exit', 'round_trips_closed_by_exit'),
            ('… closed by an opposite take', 'round_trips_closed_by_take'), ('Held to settlement, after fees', 'held_to_settlement'),
            ('**Total at settlement**', 'total_at_settlement'), ('Held, at the 298 s mark', 'held_at_mark'), ('Total at the 298 s mark', 'total_at_mark'),
            ('Fees (included above)', 'fees'), ('Settlement − mark', 'carry_settlement_minus_mark')]
    for lab, k in rows:
        L.append(f"| {lab} | {usd(b['money_usd'][k])} | {usd(f['money_usd'][k])} | {usd(f['money_usd'][k] - b['money_usd'][k])} |")
    L += ['', f"Per market at settlement: BASE {usd(b['per_market_usd']['total_at_settlement'])}, FLOOR {usd(f['per_market_usd']['total_at_settlement'])}.", '',
          '## By day, total at settlement (USD)', '', '| Day | BASE | FLOOR | FLOOR − BASE |', '|---|---:|---:|---:|']
    for day in rd['days']:
        L.append(f"| {day} | {usd(b['day_total_at_settlement_usd'][day])} | {usd(f['day_total_at_settlement_usd'][day])} | "
                 f"{usd(d['settle_net_c']['by_day_usd'][day])} |")
    L += [f"| Days positive | {b['days_positive_at_settlement']} / 5 | {f['days_positive_at_settlement']} / 5 | {d['settle_net_c']['days_positive']} / 5 |", '',
          '## Risk carried to settlement', '', '| | BASE | FLOOR |', '|---|---:|---:|']
    rb, rf = b['risk'], f['risk']
    rk = [('Markets with a position at END', 'markets_with_position', lambda x: f'{x:,}'), ('… of which ≥ 1 share', 'markets_with_1_share_or_more', lambda x: f'{x:,}'),
          ('Shares held to settlement', None, None), ('Share of shares taken', 'held_share_of_taken', pct),
          ('Shares per market with a position, p50', 'held_shares_per_market_p50', num), ('… p90', 'held_shares_per_market_p90', num),
          ('… max', 'held_shares_per_market_max', num), ('Loss if every held lot settles against us (USD)', 'max_loss_if_all_settle_against_usd', usd),
          ('… largest one market (USD)', 'max_loss_one_market_usd', usd), ('Held shares that won', 'held_shares_won_share', pct)]
    for lab, k, fmt in rk:
        if k is None:
            L.append(f"| {lab} | {num(b['shares']['held_to_settlement'])} | {num(f['shares']['held_to_settlement'])} |")
        else:
            L.append(f'| {lab} | {fmt(rb[k])} | {fmt(rf[k])} |')
    hb, hf = rb['held_settlement_usd_per_market_with_position'], rf['held_settlement_usd_per_market_with_position']
    for lab, k in (('Settlement P&L of held lots per market: mean', 'mean'), ('… std', 'std'), ('… p5', 'p5'), ('… worst', 'min'), ('… best', 'max')):
        L.append(f'| {lab} (USD) | {usd(hb[k])} | {usd(hf[k])} |')
    fm = rd['floor_mechanics']
    L += ['', '## What the floor changed', '',
          f"- FLOOR exit posts: {fm.get('exit_posts', 0):,}; priced up to the floor (above the touch): {fm.get('floor_bound_posts', 0):,}; "
          f"cancelled because the floor rose: {fm.get('floor_cancels', 0):,}.",
          f"- FLOOR structural cancels: distance {fm.get('cancel_distance', 0):,}, cross {fm.get('cancel_cross', 0):,}, no book {fm.get('cancel_no_book', 0):,}, "
          f"floor {fm.get('cancel_floor', 0):,}.",
          f"- Shares closed by the maker exit: BASE {num(b['shares']['closed_by_exit'])}, FLOOR {num(f['shares']['closed_by_exit'])}; "
          f"by an opposite take: BASE {num(b['shares']['closed_by_take'])}, FLOOR {num(f['shares']['closed_by_take'])}; "
          f"shares taken: BASE {num(b['shares']['taken'])}, FLOOR {num(f['shares']['taken'])}.",
          f"- Paired per market at settlement: FLOOR better in {d['settle_net_c']['markets_better']}, worse in {d['settle_net_c']['markets_worse']}, "
          f"within ½ ¢ in {d['settle_net_c']['markets_within_half_cent']}; the top 10 markets carry {pct(d['settle_net_c']['top10_abs_share'])} of |difference|.", '',
          '## FLOOR round trips by how they closed ([fill_split.json](fill_split.json))', '',
          '`touch`: the exit was posted at the best price (the floor did not bind at that post, including re-posts after the price came back); '
          '`floor`: the exit was raised above the touch to break-even and rested away from it, where the replay\'s optimistic queue rule '
          'matters most; `take`: an opposite-signal take closed the position.', '',
          '| Closed by | Round-trip shares | Round trips after fees (USD) | ¢ / share | Exit fill age p50 / p90 (ms) |', '|---|---:|---:|---:|---:|']
    sp = json.loads((H / 'fill_split.json').read_text())
    for k in ('touch', 'floor', 'take'):
        x = sp[k]
        age = '—' if x['fill_age_ms_p50'] is None else f"{x['fill_age_ms_p50']:,.0f} / {x['fill_age_ms_p90']:,.0f}"
        L.append(f"| {k} | {num(x['rt_shares'])} | {usd(x['rt_net_usd'])} | {num(x['rt_net_c_per_share'], 2)} | {age} |")
    L += ['', '## Readings (computed)', '']
    for k, v in rd['readings'].items():
        L.append(f'- `{k}`: {v}')
    L += ['', '## Checks', '']
    for k, v in ck.items():
        if isinstance(v, dict):
            L.append(f"- {'PASS' if v['ok'] else 'FAIL'} `{k}`: {v['detail']}")
    L += [f"- First-day projection {pj['projected_cpu_s']:.0f} CPU s against the {pj['cap_cpu_s']:.0f} s cap (go = {pj['go']}).", '',
          '## Limits', '',
          '- Consumed days, one floor level fixed before the run; no validation claim. Five day clusters: no interval.',
          '- The replay is path-coupled: the floor changes positions, so later takes (room, flips) differ too; per-market differences are noisy.',
          '- Exit fills keep the replay\'s optimistic queue rule (shrinking shown size assumed ahead of us) and 10 ms exit latency; a cross '
          'cancels without a fill. A floor order resting away from the touch leans on the same queue rule when the price comes back.',
          '- Settlement uses the venue\'s resolved outcome for each market; trading stops at 298 s, so what is held then is held to resolution.']
    (H / 'report.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
