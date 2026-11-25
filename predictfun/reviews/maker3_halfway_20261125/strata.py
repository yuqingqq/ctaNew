"""Descriptive follow-up: compare flat/other within dates and halfway book states."""
import csv
from collections import defaultdict
from pathlib import Path
root = Path(__file__).resolve().parent
cells = defaultdict(list)
for row in csv.DictReader((root/'market_rows.csv').open()):
    for axis in ('day','book'):
        cells[(axis,row[axis],abs(float(row['pos']))<=10)].append(float(row['second'])==0)
for key, stopped in sorted(cells.items()):
    print(*key, sum(stopped), len(stopped))
