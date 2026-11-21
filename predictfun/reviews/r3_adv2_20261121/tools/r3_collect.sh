#!/bin/bash
# Copy the round's outputs into the receipt (CSV copies; any CSV over 1 MB gzipped, -n for stable bytes), the evaluation sample's
# accepted.json and a sha256 manifest of every evaluation-label and replay part left in data/.
set -eu
R=/home/yuqing/ctaNew/predictfun/reviews/r3_adv2_20261121; O=/home/yuqing/ctaNew/data/predict_fun/research/adv2_rich_v1_20261121
mkdir -p $R/eval $R/replay $R/checkpoint
put() { if [ "$(stat -c %s "$1")" -gt 1000000 ]; then gzip -n -c "$1" > "$2/$(basename "$1").gz"; else cp "$1" "$2/"; fi; }
for f in readings screens calibration control support sources horizon coverage coverage_train; do put $O/eval/$f.csv $R/eval; done
cp $O/eval/eval.json $O/eval/predictions.csv.gz $R/eval/
for f in episodes reached churn exposure; do put $O/replay_v1/$f.csv $R/replay; done
cp $O/replay_v1/replay.json $R/replay/; cp $O/replay_v1/parts/part_*[0-9].json $R/replay/
cp $O/eval_labels_v1/accepted.json $R/checkpoint/accepted.json
(cd $O && find eval_labels_v1 replay_v1/parts -type f | sort | xargs sha256sum) > $R/checkpoint/outputs.sha256
ls -la $R/eval $R/replay | awk '{print $5, $9}' | grep -v '^ *$'; wc -l < $R/checkpoint/outputs.sha256
