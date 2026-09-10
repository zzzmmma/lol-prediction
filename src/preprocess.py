"""Auditable Oracle's Elixir -> one-row-per-game preprocessing (stdlib only)."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROLES = {'top': 'TOP', 'jng': 'JUNGLE', 'mid': 'MID', 'bot': 'ADC', 'sup': 'SUPPORT'}
SIDES = ('BLUE', 'RED')
TARGETS = ('winner_side', 'first_dragon_side', 'more_dragons_side',
           'first_four_dragon_side', 'dragon_soul_side',
           'elder_dragon_side', 'first_baron_side')
INPUTS = tuple(f'{side.lower()}_{kind}_{role.lower()}'
               for kind in ('player', 'champion') for side in SIDES for role in ROLES.values())
# Post-game analysis columns must never be added to INPUTS.
DRAGON_COUNTS = ('blue_dragon_count', 'red_dragon_count')
METADATA = ('game_id', 'date', 'year', 'split', 'stage', 'patch', 'blue_team', 'red_team')
COLUMNS = METADATA + INPUTS + TARGETS + DRAGON_COUNTS
TARGET_VALUES = {t: (*SIDES, 'NONE') for t in TARGETS}
TARGET_VALUES.update(winner_side=SIDES, more_dragons_side=(*SIDES, 'TIE'),
                     dragon_soul_side=(*SIDES, 'NONE', 'N/A'),
                     elder_dragon_side=(*SIDES, 'NONE', 'N/A'))


def column_roles():
    return {'input_columns': list(INPUTS), 'target_columns': list(TARGETS),
            'analysis_only_columns': list(DRAGON_COUNTS), 'metadata_columns': list(METADATA)}
DEFAULT_STAGE_CALENDAR = Path(__file__).resolve().parents[1] / 'config/stage_calendar.json'
STAGES = ('Regular Season', 'Playoff', 'Cup', 'Promotion', 'Regional Qualifier',
          'Road to MSI', 'Play-In')
RAW_FILENAME = re.compile(r'(\d{4})_LoL_esports_match_data_from_OraclesElixir(?: \(\d+\))?\.csv')


def discover_raw_files(raw_dir):
    """Recognize browser download suffixes without renaming any source file."""
    found = defaultdict(list)
    for path in sorted(Path(raw_dir).glob('*.csv')):
        match = RAW_FILENAME.fullmatch(path.name)
        if match:
            found[int(match[1])].append(path)
    return dict(found)


def resolve_raw_file(paths):
    if len(paths) > 1:
        raise ValueError(f'Ambiguous raw snapshots; use a directory with one file per year: {paths}')
    return paths[0]


def load_stage_calendar(path=None):
    calendar = json.loads(Path(path or DEFAULT_STAGE_CALENDAR).read_text(encoding='utf-8-sig'))
    for name in ('summer_start', 'playoff_start'):
        if not isinstance(calendar.get(name), dict):
            raise ValueError(f'Stage calendar needs a {name} object')
        for year, day in calendar[name].items():
            parsed = datetime.strptime(day, '%Y-%m-%d')
            if str(parsed.year) != year or parsed.strftime('%Y-%m-%d') != day:
                raise ValueError(f'Invalid calendar boundary: {year} {day}')
    return calendar


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_games(path):
    # csv keeps literal N/A distinct from an empty/missing value (unlike pandas defaults).
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': Path(path).stat().st_size, 'sha256': digest.hexdigest()}


def patch_tuple(value):
    match = re.fullmatch(r'(\d+)\.(\d+)(?:\.\d+)?', value)
    if not match:
        raise ValueError(f'Invalid patch: {value!r}')
    return tuple(map(int, match.groups()))


def count(value):
    if value in ('', None):
        return None
    number = float(value)
    if not number.is_integer() or number < 0:
        raise ValueError(f'Invalid objective count: {value!r}')
    return int(number)


def classify_stage(row, calendar=None):
    calendar = calendar if calendar is not None else load_stage_calendar()
    year, split, day = int(row['year']), row['split'], row['date'][:10]
    rounds = re.fullmatch(r'Rounds 3-(\d+)', split)
    later_rounds = rounds is not None and int(rounds[1]) > 3
    if split == 'Cup':
        return 'Cup'
    if not split:
        return 'Regional Qualifier'
    if year != int(day[:4]):
        return 'Promotion'
    if split == 'Summer' and str(year) in calendar['summer_start'] and day < calendar['summer_start'][str(year)]:
        return 'Promotion'
    if row['playoffs'] not in ('0', '1'):
        raise ValueError(f"Unknown playoffs flag: {row['playoffs']!r}")
    if row['playoffs'] == '1':
        if split == 'Rounds 1-2':
            return 'Road to MSI'
        if later_rounds:
            if str(year) not in calendar['playoff_start']:
                raise ValueError(f'No postseason calendar for {year}; supply --stage-calendar with a verified boundary')
            return 'Play-In' if day < calendar['playoff_start'][str(year)] else 'Playoff'
        if split in ('Spring', 'Summer'):
            return 'Playoff'
    elif split in ('Spring', 'Summer', 'Rounds 1-2') or later_rounds:
        return 'Regular Season'
    raise ValueError(f'Unknown competition: {year} {split}')


def unique_side(values, threshold=1):
    if any(x is None for x in values):
        return None, 'missing_source_count'
    winners = [side for side, n in zip(SIDES, values) if n >= threshold]
    if len(winners) > 1:
        return None, 'both_teams_qualify_without_event_order'
    return (winners[0] if winners else 'NONE'), 'team_totals'



def source_count(row, field):
    """Malformed objective observations are missing, not zero."""
    try:
        return count(row.get(field))
    except (ValueError, TypeError, OverflowError):
        return None


def dragon_counts(teams, patch):
    values, reasons = [], []
    for side in SIDES:
        row = teams[side]
        unknown = source_count(row, 'dragons (type unknown)')
        elemental = source_count(row, 'elementaldrakes')
        total, elder = source_count(row, 'dragons'), source_count(row, 'elders')
        if row.get('dragons (type unknown)') not in ('', None) and unknown is None:
            value, reason = None, 'invalid_unknown_dragon_count'
        elif unknown not in (None, 0):
            value, reason = None, 'unknown_dragon_type'
        elif row.get('elementaldrakes') not in ('', None):
            value, reason = elemental, 'elementaldrakes' if elemental is not None else 'invalid_elementaldrakes'
        elif patch < (6, 9):
            value, reason = None, 'legacy_zero_filled_dragon_totals'
        elif total is None or elder is None:
            value, reason = None, 'missing_or_invalid_dragons_or_elders'
        elif total < elder:
            value, reason = None, 'dragons_less_than_elders'
        else:
            value, reason = total - elder, 'dragons_minus_elders'
        if patch >= (9, 23) and value is not None and value > 4:
            value, reason = None, 'impossible_post_soul_drake_total'
        if patch >= (6, 9) and value is not None and total is not None and elder is not None and total - elder != value:
            value, reason = None, 'conflicting_dragon_totals'
        values.append(value)
        reasons.append(reason)
    if patch >= (9, 23) and all(n is not None and n >= 4 for n in values):
        return [None, None], ['impossible_both_teams_have_soul'] * 2
    return values, reasons


def first_dragon(teams, drakes):
    flags = [source_count(teams[s], 'firstdragon') for s in SIDES]
    if any(v not in (0, 1) for v in flags):
        return None, 'missing_or_invalid_firstdragon_flags'
    if flags == [1, 1]:
        return None, 'contradictory_firstdragon_flags'
    if flags == [0, 0]:
        if any(n is not None and n > 0 for n in drakes):
            return None, 'firstdragon_flags_conflict_with_counts'
        return 'NONE', 'firstdragon_flags_no_capture'
    index = flags.index(1)
    if drakes[index] == 0:
        return None, 'firstdragon_flags_conflict_with_counts'
    return SIDES[index], 'firstdragon_flags'


def more_dragons(drakes):
    blue, red = drakes
    if blue is None or red is None:
        return None, 'missing_dragon_count'
    return ('BLUE' if blue > red else 'RED' if red > blue else 'TIE'), 'compare_dragon_counts'


def derive_first_baron(teams):
    flags = [source_count(teams[s], 'firstbaron') for s in SIDES]
    totals = [source_count(teams[s], 'barons') for s in SIDES]
    if any(teams[s].get('firstbaron') not in ('', None) and v not in (0, 1)
           for s, v in zip(SIDES, flags)) or flags == [1, 1]:
        return None, 'invalid_or_contradictory_firstbaron_flags'
    if any(flag == 1 and total == 0 for flag, total in zip(flags, totals)):
        return None, 'firstbaron_flags_conflict_with_counts'
    if None not in flags:
        if flags == [0, 0] and any(n is not None and n > 0 for n in totals):
            return None, 'firstbaron_flags_conflict_with_counts'
        return unique_side(flags)[0], 'firstbaron_flags'
    value, reason = unique_side(totals)
    if value in SIDES and flags[SIDES.index(value)] == 0:
        return None, 'firstbaron_flags_conflict_with_counts'
    return value, reason


def derive_targets(teams, patch, timeline=None):
    """Never equate unavailable/ambiguous observations with NONE."""
    result, reasons = {}, {}
    values = [count(teams[s].get('result')) for s in SIDES]
    if any(v not in (None, 0, 1) for v in values):
        raise ValueError('Invalid binary result')
    if values == [1, 1]:
        raise ValueError('Both teams have result=1')
    result['winner_side'], reasons['winner_side'] = unique_side(values)
    if result['winner_side'] == 'NONE':
        result['winner_side'], reasons['winner_side'] = None, 'no_recorded_winner_possible_remake'
    result['first_baron_side'], reasons['first_baron_side'] = derive_first_baron(teams)
    elder_active, soul_active = patch >= (6, 9), patch >= (9, 23)
    elders = [source_count(teams[s], 'elders') for s in SIDES]
    if elder_active:
        result['elder_dragon_side'], reasons['elder_dragon_side'] = unique_side(elders)
    else:
        result['elder_dragon_side'], reasons['elder_dragon_side'] = 'N/A', 'system_not_available'
    drakes, count_reasons = dragon_counts(teams, patch)
    for column, value, reason in zip(DRAGON_COUNTS, drakes, count_reasons):
        result[column], reasons[column] = value, reason
    result['first_dragon_side'], reasons['first_dragon_side'] = first_dragon(teams, drakes)
    result['more_dragons_side'], reasons['more_dragons_side'] = more_dragons(drakes)
    result['first_four_dragon_side'], reasons['first_four_dragon_side'] = unique_side(drakes, 4)
    if soul_active:
        result['dragon_soul_side'] = result['first_four_dragon_side']
        reasons['dragon_soul_side'] = reasons['first_four_dragon_side']
    else:
        result['dragon_soul_side'], reasons['dragon_soul_side'] = 'N/A', 'system_not_available'
    if timeline is not None:
        # Require a complete, provenance-backed event stream; partial feeds cannot prove NONE.
        if timeline.get('complete') is not True or not timeline.get('source'):
            raise ValueError('Timeline needs complete=true and source')
        events = timeline['events']
        previous = -1
        seen = set()
        totals = Counter()
        first_four = first_elder = first_baron = first_drake = 'NONE'
        for event in events:
            t, monster, side = event['timestamp_ms'], event['monster'], event['side']
            if type(t) is not int or t < 0 or t < previous or side not in SIDES or monster not in ('DRAGON', 'ELDER', 'BARON'):
                raise ValueError('Invalid or unsorted objective event')
            key = (t, monster)
            if key in seen:
                raise ValueError('Duplicate/tied objective event')
            seen.add(key)
            previous = t
            if monster == 'DRAGON':
                if soul_active and first_four != 'NONE':
                    raise ValueError('Elemental dragon event after soul acquisition')
                if first_drake == 'NONE':
                    first_drake = side
                totals[side] += 1
                if soul_active and (totals[side] > 4 or min(totals[s] for s in SIDES) >= 4):
                    raise ValueError('Impossible post-soul dragon timeline')
                if totals[side] == 4 and first_four == 'NONE':
                    first_four = side
            elif monster == 'ELDER':
                if not elder_active:
                    raise ValueError('Elder event before patch 6.9')
                if soul_active and first_four == 'NONE':
                    raise ValueError('Elder event before soul acquisition')
                if first_elder == 'NONE':
                    first_elder = side
            elif first_baron == 'NONE':
                first_baron = side
        if result['first_dragon_side'] is not None and result['first_dragon_side'] != first_drake:
            raise ValueError('Timeline disagrees with firstdragon flags')
        resolved = {**dict(zip(DRAGON_COUNTS, (totals[s] for s in SIDES))),
                    'more_dragons_side': more_dragons([totals[s] for s in SIDES])[0],
                    'first_four_dragon_side': first_four, 'first_baron_side': first_baron,
                    'elder_dragon_side': first_elder if elder_active else 'N/A',
                    'dragon_soul_side': first_four if soul_active else 'N/A'}
        for target, value in resolved.items():
            if result[target] is not None and result[target] != value:
                raise ValueError(f'Timeline disagrees with team totals for {target}')
            result[target], reasons[target] = value, 'complete_timeline'
    return result, reasons


def build_vocab(rows, kind):
    tokens = sorted({r[c] for r in rows for c in INPUTS if f'_{kind}_' in c and r[c]})
    if '<UNK>' in tokens:
        raise ValueError('Reserved vocabulary token in raw data')
    return {'<UNK>': 0, **{token: i for i, token in enumerate(tokens, 1)}}


def validate(rows):
    if not rows:
        raise ValueError('Empty dataset')
    for row in rows:
        missing = set(COLUMNS) - set(row)
        if missing:
            raise ValueError(f'Missing dataset columns: {sorted(missing)}; rerun preprocess')
        for column in METADATA:
            if row[column] is None or not str(row[column]).strip():
                raise ValueError(f'Missing metadata: {column}')
        if not re.fullmatch(r'\d{4}', str(row['year'])):
            raise ValueError(f'Invalid competition year: {row["year"]}')
        if row['stage'] not in STAGES:
            raise ValueError(f'Unknown stage: {row["stage"]}')
        if row['blue_team'] == row['red_team']:
            raise ValueError('Blue and Red teams must differ')
    ids = Counter(r['game_id'] for r in rows)
    for row in rows:
        if not row['game_id']:
            raise ValueError('Empty game_id')
        datetime.fromisoformat(row['date'])
        patch = patch_tuple(row['patch'])
        for t in TARGETS:
            allowed = (*TARGET_VALUES[t], '', None)
            if row[t] not in allowed:
                raise ValueError(f'Invalid target {t}: {row[t]}')
        for target, introduced in [('dragon_soul_side', (9, 23)), ('elder_dragon_side', (6, 9))]:
            if (patch < introduced) != (row[target] == 'N/A'):
                raise ValueError(f'Patch/system applicability mismatch: {target}')
        drakes = [count(row[c]) for c in DRAGON_COUNTS]
        expected_more = more_dragons(drakes)[0]
        if (row['more_dragons_side'] or None) != expected_more:
            raise ValueError('more_dragons_side disagrees with dragon counts')
        if patch >= (9, 23) and any(n is not None and n > 4 for n in drakes):
            raise ValueError('Impossible post-soul dragon count')
        if patch >= (9, 23) and all(n is not None and n >= 4 for n in drakes):
            raise ValueError('Both teams cannot acquire a soul')
        four = row['first_four_dragon_side']
        if four in SIDES and (drakes[SIDES.index(four)] is None or drakes[SIDES.index(four)] < 4):
            raise ValueError('First-four winner has no confirmed four dragons')
        if four == 'NONE' and any(n is not None and n >= 4 for n in drakes):
            raise ValueError('First-four NONE conflicts with dragon counts')
        if four in (*SIDES, 'NONE') and any(n is None for n in drakes):
            raise ValueError('First-four cannot be confirmed with incomplete dragon counts')
        expected_four = unique_side(drakes, 4)[0]
        if four in SIDES and expected_four not in (None, four):
            raise ValueError('First-four winner disagrees with dragon counts')
        if patch >= (9, 23) and (row['dragon_soul_side'] or None) != (four or None):
            raise ValueError('Soul target disagrees with first-four target')
        first = row['first_dragon_side']
        if first in SIDES and drakes[SIDES.index(first)] == 0:
            raise ValueError('First-dragon winner has zero dragons')
        if first == 'NONE' and any(n is not None and n > 0 for n in drakes):
            raise ValueError('First-dragon NONE conflicts with dragon counts')
        for c in INPUTS:
            if not row[c]:
                raise ValueError(f'Missing roster/champion: {row["game_id"]} {c}')
    duplicates = {k: v for k, v in ids.items() if v > 1}
    report = {
        'total_games': len(rows),
        'games_by_year': dict(sorted(Counter(str(r['year']) for r in rows).items())),
        'games_by_stage': dict(sorted(Counter(r['stage'] for r in rows).items())),
        'unique_players': len(build_vocab(rows, 'player')) - 1,
        'unique_champions': len(build_vocab(rows, 'champion')) - 1,
        'target_distributions': {t: {v: sum(r[t] == v for r in rows) for v in ('BLUE', 'RED', 'NONE', 'N/A', 'TIE')}
                                 | {'MISSING': sum(r[t] in ('', None) for r in rows)} for t in TARGETS},
        'missing_by_column': {c: sum(r[c] in ('', None) for r in rows) for c in COLUMNS},
        'dragon_count_statistics': {c: {
            'known': sum(r[c] not in ('', None) for r in rows),
            'missing': sum(r[c] in ('', None) for r in rows),
            'zero': sum(count(r[c]) == 0 for r in rows)} for c in DRAGON_COUNTS},
        'column_roles': column_roles(),
        'duplicate_game_ids': duplicates,
    }
    report['missing_target_cells'] = sum(report['missing_by_column'][c] for c in TARGETS)
    report['missing_dragon_count_cells'] = sum(report['missing_by_column'][c] for c in DRAGON_COUNTS)
    report['total_missing_cells'] = sum(report['missing_by_column'].values())
    if duplicates:
        raise ValueError(f'Duplicate game IDs: {duplicates}')
    return report



def write_summary(path, report):
    lines = ['# LCK 전처리 검증 결과', '',
             f"전체 게임 레코드: {report['total_games']:,}", '',
             '| 대회 연도 | 게임 수 |', '|---|---:|']
    lines += [f'| {year} | {n:,} |' for year, n in report['games_by_year'].items()]
    lines += ['', '| Stage | 게임 수 |', '|---|---:|']
    lines += [f'| {stage} | {n:,} |' for stage, n in report['games_by_stage'].items()]
    lines += ['', f"고유 선수명: {report['unique_players']:,}, 고유 챔피언: {report['unique_champions']:,}", '',
              '| Target | BLUE | RED | NONE | N/A | TIE | 결측 |', '|---|---:|---:|---:|---:|---:|---:|']
    for target, counts in report['target_distributions'].items():
        lines.append('| ' + target + ' | ' + ' | '.join(str(counts[k]) for k in ('BLUE', 'RED', 'NONE', 'N/A', 'TIE', 'MISSING')) + ' |')
    lines += ['', '| 분석용 Dragon count | 확인됨 | 0개 | 결측 |', '|---|---:|---:|---:|']
    for column, stats in report['dragon_count_statistics'].items():
        lines.append(f"| {column} | {stats['known']} | {stats['zero']} | {stats['missing']} |")
    lines += ['', 'Dragon count 2개는 분석/타깃 생성용이며 모델 입력은 선수·챔피언 20개로 제한한다.']
    lines += ['', f"전체 결측 셀: {report['total_missing_cells']:,}",
              f"중복 game_id: {len(report['duplicate_game_ids'])}", '',
              '입력/메타데이터 열의 결측 개수 및 타깃별 결측은 validation_report.json 참조.', '',
              '원본 스냅샷의 전체 레코드를 처리한 결과이며 전체 공식 경기 목록과의 대조 인증은 아님.']
    if 'coverage' in report:
        lines += [f"원본 날짜 범위: {report['coverage']['date_min']} ~ {report['coverage']['date_max']}"]
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def preprocess(raw_dir, output_dir, start_year=2015, end_year=None, events_path=None, stage_calendar_path=None):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calendar = load_stage_calendar(stage_calendar_path)
    discovered = discover_raw_files(raw_dir)
    if end_year is None:
        end_year = max(discovered, default=start_year - 1)
    if start_year > end_year:
        raise ValueError('No raw years in range, or start_year > end_year')
    # Resolve all requested years before parsing hundreds of megabytes.
    paths = []
    for year in range(start_year, end_year + 1):
        if year not in discovered:
            raise FileNotFoundError(f'Missing raw year {year} in {raw_dir}')
        paths.append((year, resolve_raw_file(discovered[year])))
    groups, sources, manifest = defaultdict(list), defaultdict(set), []
    required = {'gameid', 'date', 'year', 'split', 'playoffs', 'patch', 'league',
                'side', 'position', 'playername', 'champion', 'teamname', 'result', 'firstbaron'}
    for year, path in paths:
        info = fingerprint(path)
        info['calendar_year'] = year
        matched = 0
        with path.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if required - set(reader.fieldnames or []):
                raise ValueError(f'{path}: missing columns {required - set(reader.fieldnames or [])}')
            for row in reader:
                if row['league'] != 'LCK' and not (year == 2015 and row['league'] == 'OGN'):
                    continue
                if not row['gameid']:
                    raise ValueError(f'{path}: empty gameid')
                groups[row['gameid']].append(row)
                sources[row['gameid']].add(path.name)
                matched += 1
        info['selected_rows'] = matched
        manifest.append(info)
    timelines = json.loads(Path(events_path).read_text(encoding='utf-8-sig')) if events_path else {}
    if set(timelines) - set(groups):
        raise ValueError('Timeline contains game IDs absent from selected source files')
    rows, audit, provenance, errors = [], [], [], []
    for game_id, participants in groups.items():
        try:
            if len(participants) != 12:
                raise ValueError(f'Expected 12 participant rows, got {len(participants)}')
            for field in ('date', 'year', 'split', 'playoffs', 'patch', 'league'):
                if len({r[field] for r in participants}) != 1:
                    raise ValueError(f'Inconsistent metadata: {field}')
            indexed = {}
            for r in participants:
                key = (r['side'].upper(), r['position'])
                if key in indexed:
                    raise ValueError(f'Duplicate participant slot: {key}')
                indexed[key] = r
            expected = {(s, role) for s in SIDES for role in (*ROLES, 'team')}
            if set(indexed) != expected:
                raise ValueError('Missing/invalid team or role slots')
            meta = participants[0]
            game = dict(game_id=game_id, date=datetime.fromisoformat(meta['date']).isoformat(sep=' '),
                        year=int(meta['year']), split=meta['split'] or 'Regional',
                        stage=classify_stage(meta, calendar), patch=meta['patch'])
            for side in SIDES:
                team_name = indexed[side, 'team']['teamname'].strip()
                if not team_name or any(indexed[side, role]['teamname'].strip() != team_name for role in ROLES):
                    raise ValueError(f'Missing/inconsistent teamname: {side}')
                game[f'{side.lower()}_team'] = team_name
                for role, normalized in ROLES.items():
                    player = indexed[side, role]
                    for kind, source_field in [('player', 'playername'), ('champion', 'champion')]:
                        game[f'{side.lower()}_{kind}_{normalized.lower()}'] = player[source_field].strip()
                    if player['result'] != indexed[side, 'team']['result']:
                        raise ValueError('Player/team result mismatch')
            targets, reasons = derive_targets({s: indexed[s, 'team'] for s in SIDES},
                                              patch_tuple(meta['patch']), timelines.get(game_id))
            game.update(targets)
            validate([game])
            rows.append(game)
            for target in (*TARGETS, *DRAGON_COUNTS):
                audit.append({'game_id': game_id, 'target': target,
                              'value': targets[target], 'reason': reasons[target]})
            provenance.append({'game_id': game_id, 'source_files': '|'.join(sorted(sources[game_id])),
                               'source_league': meta['league'], 'source_split': meta['split'],
                               'source_playoffs': meta['playoffs'], 'source_url': meta.get('url', ''),
                               'calendar_year': meta['date'][:4], 'stage': game['stage']})
        except (ValueError, KeyError) as exc:
            errors.append({'game_id': game_id, 'error': str(exc)})
    write_json(output_dir / 'structural_errors.json', errors)
    if errors:
        raise ValueError(f'{len(errors)} structural errors; see {output_dir / "structural_errors.json"}. No dataset written.')
    rows.sort(key=lambda r: (r['date'], r['game_id']))
    report = validate(rows)
    report['coverage'] = {
        'status': 'source_snapshot_not_independently_exhaustive',
        'date_min': rows[0]['date'] if rows else None, 'date_max': rows[-1]['date'] if rows else None,
        'source_files': len(manifest), 'source_game_ids': len(groups),
        'requested_years': list(range(start_year, end_year + 1)),
        'note': 'Includes OGN 2015 and all LCK-labelled events. Future games and events absent from OE are not fabricated. Year is competition year; calendar year may differ for promotion.',
    }
    report['warnings'] = [
        'Pre-6.9 OE dragon totals are unavailable/zero-filled; first-four target remains missing without events.',
        'First elder and first-four order cannot be recovered when both teams qualify; missing unless complete timeline supplied.',
        'Cup, Promotion, Regional Qualifier, Road to MSI and Play-In are retained but excluded from the default split.',
        'Player vocabulary uses source playername, not verified cross-alias person identity.',
        'A record without a winner is retained for audit and excluded from default train/test; see target_audit.csv.',
    ]
    write_csv(output_dir / 'games.csv', rows, COLUMNS)
    write_csv(output_dir / 'target_audit.csv', audit, ('game_id', 'target', 'value', 'reason'))
    write_csv(output_dir / 'game_provenance.csv', provenance, tuple(provenance[0]) if provenance else ('game_id',))
    for kind in ('player', 'champion'):
        write_json(output_dir / f'{kind}_vocab.json', build_vocab(rows, kind))
    write_json(output_dir / 'raw_manifest.json', {'files': manifest,
               'events': fingerprint(events_path) if events_path else None,
               'stage_calendar': fingerprint(stage_calendar_path or DEFAULT_STAGE_CALENDAR)})
    write_json(output_dir / 'validation_report.json', report)
    write_summary(output_dir / 'validation_summary.md', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return rows, report
