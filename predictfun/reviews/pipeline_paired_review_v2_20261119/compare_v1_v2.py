"""The four-part review v1 (module beada765, commit 6775eabd) against v2 (review 92d8e52f fixed: the plan v2 screen decides zero / sign / amount
on the episode-balanced label mean; V3 compares timing at an absolute 1 us). Declared before v2 ran: every reading, prediction group (with its
old gate and plan v2 screen), controller, risk, matched-null, quoting and FINDINGS value must be identical, except the module's own source hash
and output paths. Prints one predicate per check and ALL TRUE at the end."""
import json
import sys

R = '/home/yuqing/ctaNew/data/predict_fun/research'
SKIP = {'asof', 'source_sha256', 'outputs', 'seconds', 'provenance'}
ok_all = True
for d in ('20261026', '20261027'):
    a = json.load(open(f'{R}/pipeline_paired_review_20261119/review_{d}.json')); b = json.load(open(f'{R}/pipeline_paired_review_v2_20261119/review_{d}.json'))
    diff = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k in SKIP and len(path) == 0:
                    continue
                if k not in x or k not in y:
                    diff.append(f'{path + [k]}: missing'); continue
                walk(x[k], y[k], path + [k])
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, path + [i])
        elif x != y and not (isinstance(x, float) and isinstance(y, float) and x != x and y != y):
            diff.append(f'{path}: {x} -> {y}')
    walk(a, b, [])
    scr = [(k, g) for k, v in a['prediction'].items() for g, x in v.items() if x.get('plan_v2_screen') != b['prediction'][k][g].get('plan_v2_screen')]
    same_src = a['source_sha256'] == b['source_sha256']
    print(f"{d}: differences outside the module's own hash / outputs: {len(diff)} {diff[:5]}; plan v2 screen verdicts changed: {len(scr)} {scr[:5]}; "
          f"module source changed: {not same_src}")
    ok_all &= (not diff) and (not scr) and (not same_src)
print(f'REVIEW v2 = v1 ON EVERY READING, VALUE AND SCREEN VERDICT: {ok_all}')
sys.exit(0 if ok_all else 1)
