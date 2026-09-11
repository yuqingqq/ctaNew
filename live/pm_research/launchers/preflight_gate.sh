#!/bin/bash
# The day's gates, run WITHOUT the lock, before an offer.
# A SCRIPT UNDER A RUNNING UNIT IS FROZEN BYTES: bash reads by byte offset,
# so editing a running script makes the shell resume at its saved offset in
# a DIFFERENT file. Two chains were mid-flight with offsets landing inside
# a word. This lives in its own file so only FUTURE launches pick it up.
set -u
day="${1:?usage: preflight_gate.sh <YYYY-MM-DD> <certificate>}"
cert="${2:?}"
/home/yuqing/pricer-sol/venv/bin/python3 \
  /home/yuqing/ctaNew-wt-deval/live/pm_research/de_preflight_matrix.py \
  --tree /home/yuqing/ctaNew-wt-deval \
  --derived /home/yuqing/ctaNew/data/pm_5min/derived \
  --declarations /home/yuqing/ctaNew-wt-deval/live/pm_research/declarations \
  --certification "$cert" --days "$day" --gate
