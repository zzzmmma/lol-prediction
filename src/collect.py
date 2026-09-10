"""Optional immutable collection of yearly CSVs from an explicit URL template."""
import csv
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from .preprocess import discover_raw_files, resolve_raw_file, fingerprint, write_json


def collect(raw_dir, url_template, years):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    results = []
    existing = discover_raw_files(raw_dir)
    for year in years:
        destination = raw_dir / f'{year}_LoL_esports_match_data_from_OraclesElixir.csv'
        if year in existing:
            results.append({'status': 'preserved_existing', **fingerprint(resolve_raw_file(existing[year]))})
            continue
        url = url_template.format(year=year)
        if not url.startswith('https://'):
            raise ValueError('Use an HTTPS source URL')
        fd, temporary = tempfile.mkstemp(dir=raw_dir, suffix='.download')
        try:
            with os.fdopen(fd, 'wb') as output, urlopen(url, timeout=120) as response:
                while block := response.read(1024 * 1024):
                    output.write(block)
            with open(temporary, encoding='utf-8-sig', newline='') as f:
                header = next(csv.reader(f))
                if not {'gameid', 'league', 'position', 'date'} <= set(header):
                    raise ValueError(f'{url}: not an Oracle\'s Elixir CSV')
            # A hard link fails if another process already created the destination.
            os.link(temporary, destination)
            results.append({'status': 'downloaded', 'url': url, **fingerprint(destination)})
        finally:
            Path(temporary).unlink(missing_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(raw_dir / f'collection_{stamp}.json', results)
    return results
