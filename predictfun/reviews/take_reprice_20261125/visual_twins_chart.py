"""Draw visual_twins_examples.json with SIGNAL_VS_NOISE's chart: the typical fire next to its two nearest look-alikes on the nine key book
inputs, whatever their outcome (USER 2026-11-27: "It looks the cases they fire are very different from normal cases, why our model can't
differentiate"). Every number is read from visual_twins_examples.json / visual_twins.json. Writes visual_twins_chart.svg / .html (write-once).
Usage: python3 predictfun/reviews/take_reprice_20261125/visual_twins_chart.py [<out dir>]
"""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import signal_vs_noise_chart as SC  # noqa: E402

OUTD = Path(sys.argv[1]) if len(sys.argv) > 1 else H
EX = json.loads((H / 'visual_twins_examples.json').read_text())
R = json.loads((H / 'visual_twins.json').read_text())
ORDER = ('typical_fire', 'twin_1', 'twin_2')


def outcome(e):
    return 'moved' if e['y'] == 1 else 'no move'


TITLES = {'typical_fire': "Typical fire: the top taker's order",
          'twin_1': f"Closest look-alike ({outcome(EX['twin_1'])}, no fire)", 'twin_2': f"2nd closest look-alike ({outcome(EX['twin_2'])}, no fire)"}
HEAD = ('The same picture without the top taker: its fire next to the two ordinary moments that look most like it',
        'Binance futures best level, 1 s before to 0.3 s after (t = 0: the fire\'s order reaching Predict). Look-alikes: nearest on nine key book inputs, same day, same bet (up).')
mr, ds = R['move_rate'], R['distance']
LINES = [(f"For each of its {R['fires']:,} fires, the {R['k']} ordinary moments that look most like it moved their way within 300 ms {100 * mr['twins_any_time']:.1f}% of the time "
          f"(the single closest {100 * mr['nearest_twin']:.1f}%), against {100 * mr['fires']:.0f}% after its fires.", 't1'),
         (f"The closest fifth of look-alikes moved {100 * R['by_distance_fifth']['1']['twin_move_rate']:.1f}%. A fire's closest ordinary look-alike sits nearer to it "
          f"(median distance {ds['fire_to_nearest_twin_median']:.2f}) than its closest other fire ({ds['fire_to_nearest_fire_median']:.2f}).", 't1'),
         (f"Our model already rates these pictures high (fires {100 * R['public_score']['fires']:.1f}%, look-alikes {100 * R['public_score']['twins_any_time']:.1f}%, random "
          f"{100 * R['public_score']['random']:.1f}%); what it cannot see is which of them will move.", 't1'),
         ('Four consumed days; development evidence. The look-alikes are taken whatever their outcome.', 't3')]


if __name__ == '__main__':
    SC.draw(EX, ORDER, TITLES, HEAD, LINES, 'visual_twins_chart', OUTD)
