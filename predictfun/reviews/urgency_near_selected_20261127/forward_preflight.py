"""Check forward feed-file availability from filenames and sizes only, without reading market outcomes."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3] / 'data'
MARKET = ROOT / 'predict_fun/runs/20261021T162823Z_0dd45f2d5a'
REFERENCE = ROOT / 'predict_reference/runs/20261022T132901Z_8c043d17b6'
FEEDS = ((MARKET/'rest', '.jsonl.gz'), (MARKET/'predict_ws', '.jsonl.gz'),
         (REFERENCE/'raw_ws', '.jsonl.gz'), (REFERENCE/'reference_prices', '.jsonl.gz'),
         (ROOT/'mm_hf/raw/bookTicker/BTCUSDT', '.csv.gz'))


def inspect(first, now):
    statuses = []
    for day_index in range(5):
        day = first + timedelta(days=day_index)
        # Step 1 reads the preceding 23Z hour and both following 00Z/01Z hours.
        hours = [day + timedelta(hours=h) for h in range(-1, 26)]
        gaps = {}
        for folder, suffix in FEEDS:
            missing = []
            for hour in hours:
                stem = hour.strftime('%Y%m%d_%H')
                path = folder / (stem + suffix)
                if not path.is_file() or path.stat().st_size == 0:
                    missing.append(stem)
            if missing:
                gaps[folder.name] = len(missing)
        status = 'awaiting_close_and_tail' if now < day + timedelta(hours=26) else (
            'source_gap' if gaps else 'hour_files_present')
        statuses.append((day.date().isoformat(), status, gaps))
    return statuses


if __name__ == '__main__':
    start = datetime.strptime(sys.argv[1], '%Y%m%d').replace(tzinfo=timezone.utc)
    for date, status, gaps in inspect(start, datetime.now(timezone.utc)):
        print(date, status, gaps)
