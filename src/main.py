"""Python entry point for collection, preprocessing, validation and optional splits."""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from src.preprocess import preprocess, read_games, validate
from src.splits import create_splits
from src.collect import collect

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('preprocess')
    prep.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    prep.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed')
    prep.add_argument('--start-year', type=int, default=2015)
    prep.add_argument('--end-year', type=int, help='Defaults to the latest year found in raw-dir')
    prep.add_argument('--events', type=Path)
    prep.add_argument('--stage-calendar', type=Path, help='JSON calendar for promotion and postseason boundaries')
    prep.add_argument('--strict-targets', action='store_true', help='Exit nonzero after reporting any unresolved target')
    split = commands.add_parser('split')
    split.add_argument('--dataset', type=Path, default=ROOT / 'data/processed/games.csv')
    split.add_argument('--output-dir', type=Path, default=ROOT / 'data/splits')
    split.add_argument('--config', type=Path, help='JSON with train and test filters: years, splits, stages, date_from, date_to')
    check = commands.add_parser('validate')
    check.add_argument('--dataset', type=Path, default=ROOT / 'data/processed/games.csv')
    check.add_argument('--strict-targets', action='store_true')
    download = commands.add_parser('collect')
    download.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    download.add_argument('--url-template', required=True, help='Verified HTTPS download URL containing {year}')
    download.add_argument('--start-year', type=int, default=2015)
    download.add_argument('--end-year', type=int, default=date.today().year)
    args = parser.parse_args()
    if hasattr(args, 'start_year') and args.end_year is not None and args.start_year > args.end_year:
        parser.error('start-year must be <= end-year')
    if args.command == 'preprocess':
        _, report = preprocess(args.raw_dir, args.output_dir, args.start_year, args.end_year, args.events, args.stage_calendar)
        if args.strict_targets and report['missing_target_cells']:
            raise SystemExit(2)
    elif args.command == 'split':
        config = json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
        print(json.dumps(create_splits(args.dataset, args.output_dir, config.get('train'), config.get('test')), indent=2))
    elif args.command == 'validate':
        report = validate(read_games(args.dataset))
        print(json.dumps(report, indent=2))
        if args.strict_targets and report['missing_target_cells']:
            raise SystemExit(2)
    else:
        print(json.dumps(collect(args.raw_dir, args.url_template, range(args.start_year, args.end_year + 1)), indent=2))


if __name__ == '__main__':
    main()
