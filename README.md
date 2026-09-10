# LCK 경기 데이터 전처리

Oracle's Elixir 연간 CSV를 읽어 **게임(세트)당 1행**으로 변환한다. 현재 검증 범위는 2015~2026이며, 이후 연도도 CLI와 일정 설정으로 추가할 수 있다. Python 3.10 이상 표준 라이브러리만 사용하며 Transformer, ELO 및 경기력 추가 feature는 구현하지 않는다. 기존 `src/model.py`, `train.py`, `evaluate.py`, `dataset.py`, `utils.py`는 빈 파일이어서 변경하지 않았다.

요구사항 대조, 수정된 Baron 정답 4개, 검증 수치와 변경 파일 목록은 [데이터셋 점검 결과](docs/dataset_review.md)를 참조한다.

## 실행

프로젝트 루트에서 실행한다. Windows PowerShell:

```powershell
# .venv가 없는 새 환경에서만 생성 (Mac의 .venv를 복사하지 않는다)
py -3 -m venv .venv
.\.venv\Scripts\python.exe main.py preprocess --start-year 2015 --end-year 2026
.\.venv\Scripts\python.exe main.py validate
.\.venv\Scripts\python.exe main.py split
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS:

```bash
python3 -m venv .venv  # 새 환경에서만 생성
.venv/bin/python main.py preprocess --start-year 2015 --end-year 2026
.venv/bin/python main.py validate
.venv/bin/python main.py split
.venv/bin/python -m unittest discover -s tests -v
```

데이터 작업에는 `pip install -r requirements.txt`가 필요 없다. 기존 `requirements.txt`의 모델·분석 라이브러리 버전 목록은 이번에 변경하거나 설치하지 않았다. 이후 PyTorch 환경은 각 OS에서 별도로 준비한다. `data/raw/`는 Git에서 제외되므로 새 컴퓨터에는 원본 CSV를 별도로 복사하거나 수집해야 한다.

`preprocess`의 기본 시작 연도는 2015이고, `--end-year` 생략 시 원본 폴더의 최대 연도까지 처리한다. 범위 중간의 연간 파일 누락은 오류다. `split`을 별도로 실행하면 **데이터셋에 있는 모든 연도**의 `Regular Season`을 train, `Playoff`를 test로 저장한다. 원본에서 승자가 확정되지 않은 레코드는 split에서 제외한다. 다른 타깃의 결측은 그대로 유지하므로 향후 학습 시 타깃별 마스킹이 필요하다.

`preprocess --strict-targets`는 파일과 검증 보고서를 생성한 후 7개 타깃 중 미확정 값이 있으면 종료 코드 2를 반환한다. `validate --strict-targets`도 결측 target이 있으면 2를 반환한다. 기본 `validate`는 결측 자체를 오류로 보지 않고 집계한다. 분석 count 결측은 별도로 집계한다. 원본 파일 누락·중복 후보·필수 헤더 누락은 입력 단계에서 오류를 낸다. 중복 참가자, 잘못된 패치, 팀명/메타데이터 불일치 등 경기 구조 오류는 `structural_errors.json`에 기록하고 데이터셋 생성을 중단한다. 실패한 실행은 이전에 성공한 데이터셋을 지우지 않으므로 종료 코드를 확인한다.

## 파일 구조

```text
main.py                          # 수집 / 전처리 / 검증 / split CLI
requirements.txt                 # 기존 환경의 라이브러리 목록
config/stage_calendar.json       # 확인된 승강전/포스트시즌 날짜 경계
src/
├── collect.py                   # 원본 보존 수집
├── preprocess.py                # 타깃 생성, 변환, 감사, vocabulary, 검증
├── splits.py                    # stage 및 사용자 조건에 따른 분할
└── dataset.py / model.py / train.py / evaluate.py / utils.py  # 현재 빈 파일
tests/test_preprocessing.py      # 회귀 및 파이프라인 테스트
data/
├── raw/                         # 기존 연간 CSV 12개, 읽기 전용 취급
├── processed/
│   ├── games.csv                # 한 게임당 1행, 37개 열 (입력 20 + 메타데이터 8 + 타깃 7 + 분석 2)
│   ├── player_vocab.json        # 전체 데이터의 선수명 -> ID
│   ├── champion_vocab.json      # 전체 데이터의 챔피언명 -> ID
│   ├── validation_report.json   # 전체/연도/stage/타깃 분포, 결측, 중복
│   ├── validation_summary.md    # 이번 실행의 사람이 읽는 검증 결과
│   ├── raw_manifest.json        # 입력 파일 크기, SHA-256, 선택된 행 수
│   ├── game_provenance.csv      # 원본 대회명, split, playoffs, URL 등
│   ├── target_audit.csv         # 게임·타깃/분석 count별 값과 판정/결측 사유
│   └── structural_errors.json
└── splits/
    ├── train.csv
    ├── test.csv
    ├── player_vocab.json        # train에서만 만든 vocabulary
    ├── champion_vocab.json
    └── split_manifest.json      # 선택 조건, 제외 수, 입력 해시
```

`games.csv`의 최종 37개 열 (저장 순서):

```text
game_id
date
year
split
stage
patch
blue_team
red_team
blue_player_top
blue_player_jungle
blue_player_mid
blue_player_adc
blue_player_support
red_player_top
red_player_jungle
red_player_mid
red_player_adc
red_player_support
blue_champion_top
blue_champion_jungle
blue_champion_mid
blue_champion_adc
blue_champion_support
red_champion_top
red_champion_jungle
red_champion_mid
red_champion_adc
red_champion_support
winner_side
first_dragon_side
more_dragons_side
first_four_dragon_side
dragon_soul_side
elder_dragon_side
first_baron_side
blue_dragon_count
red_dragon_count
```

`INPUTS`는 기존 선수 10명 + 챔피언 10명만 포함한다. `DRAGON_COUNTS`는 분석 전용이고 `TARGETS`는 예측 정답이다. `validation_report.json`과 `split_manifest.json`의 `column_roles`에도 이 구분을 기록한다. CSV 전체에서 메타데이터만 제외해 모델 입력을 만드는 방식은 사용하지 않는다.

`blue_team`, `red_team`은 OE 팀 행의 `teamname`을 보존하고 해당 팀 선수 5명의 팀명과 일치하는지 검증한다. 분석용 메타데이터이므로 입력에는 포함하지 않는다. 입력 token 0~4/5~9는 Blue/Red 선수, 10~14/15~19는 Blue/Red 챔피언이다. 이후 모델은 선수·챔피언별 별도 `nn.Embedding`, embedding 차원 256, 고정 위치 및 side/role 정보를 표현하도록 구현할 예정이다. 현재 단계에서는 embedding과 학습 코드를 구현하지 않는다.

선수와 챔피언 슬롯 순서는 TOP → JUNGLE → MID → ADC → SUPPORT이며 OE `jng/bot/sup`를 정규화한다. `patch`는 문자열로 보존하고 비교할 때 정수 쌍을 사용한다. `date`는 원본 시각을 보존하며 원본에 없는 시간대는 부여하지 않는다. `year`는 **원본 대회 연도**다. 가을 승강전의 대회 연도가 실제 경기 날짜보다 다음 해일 수 있다. 달력 연도는 provenance에 보존한다.

## 대회 구분과 범위

- 2015 `OGN`을 LCK의 과거 명칭으로 포함하고, 모든 파일의 `LCK` 행을 포함한다. LCK CL, 국제 대회, KeSPA Cup 등 다른 league는 포함하지 않는다.
- Spring/Summer 정규 및 playoffs를 구분하고 승강전(`Promotion`)과 빈 split인 선발전(`Regional Qualifier`)을 분리한다. 승강전은 대회 연도/날짜 불일치 및 Summer 시작 전 날짜로 식별한다.
- 2025~2026 `Cup`은 별도 `Cup`으로 보존한다. Cup 내부 세부 단계는 원본 playoffs 플래그로 구분할 수 없어 정규시즌/시즌 플레이오프에 넣지 않는다.
- `Rounds 1-2`의 postseason은 `Road to MSI`, 마지막 라운드 postseason은 날짜 경계에 따라 `Play-In`/`Playoff`로 구분한다. 시즌 플레이오프 시작은 2025-09-10, 2026-08-29이다.
- 날짜 경계는 `config/stage_calendar.json`에 있다. 새 시즌의 `Rounds 3-*` postseason을 추가할 때 확인된 날짜를 `playoff_start`에 추가하거나 `preprocess --stage-calendar 경로.json`을 사용한다. 사용자 파일은 기본 일정 전체를 대체하므로 기존 연도 경계도 포함한다. 일정이 없는 새 시즌의 postseason은 추정하지 않고 오류를 낸다. 대회 형식 자체가 바뀌면 분류 규칙도 검토해야 한다.
- 현재 결과는 **보유 원본의 전체 LCK/OGN 레코드**이며, 독립적인 전체 공식 경기 목록과 대조한 완전성 인증은 아니다. 2026 파일은 2026-09-06까지만 포함한다. 아직 열리지 않았거나 원본에 없는 경기를 생성하지 않는다.
- 2023-08-04 `ESPORTSTMNT01_3408461`은 201초 길이에 양 팀 result=0인 레코드다. 임의로 승자를 넣거나 정식 완료 경기라고 확정하지 않는다. 원래 stage로 보존하고 split에서 제외한다.

## 타깃의 의미

- `BLUE`, `RED`: 해당 조건을 충족한 팀.
- `NONE`: 시스템은 존재하며, 확인된 기록에서 어느 팀도 조건을 충족하지 않음. Elder는 생성/등장 여부와 무관하게 해당 경기에서 획득 자체가 없으면 NONE.
- `TIE`: `more_dragons_side`에서만 사용. 일반 드래곤 수가 같음(0:0 포함).
- `N/A`: 시스템 도입 전. Soul은 패치 9.23 전, Elder는 6.9 전.
- **빈 CSV 셀**: 원본 부족·불일치 또는 획득 순서 불명으로 확정 불가. `NONE`/`N/A`와 다르다. 분포 보고서에는 `MISSING`으로 집계한다.

| Prediction target | 생성 기준 |
|---|---|
| `winner_side` | 원본 팀 result의 승리 팀. 승자 미기록은 missing. |
| `first_dragon_side` | 양 팀 `firstdragon`이 1/0이면 BLUE, 0/1이면 RED, 0/0이면 NONE. 누락·비정상 값·1/1 또는 확인된 일반 드래곤 수와 모순이면 missing. 합계나 승자로 추정하지 않음. |
| `more_dragons_side` | 장로를 제외한 Blue/Red 수를 비교해 BLUE / RED / TIE. 한쪽이라도 missing이면 missing. |
| `first_four_dragon_side` | 일반 드래곤 4개를 먼저 획득한 팀. 양 팀 모두 4 미만이면 NONE, 둘 다 4 이상이면 완전한 이벤트 순서 없이는 missing. |
| `dragon_soul_side` | 9.23 이전 N/A. 도입 이후 일반 드래곤 4개 도달 팀, 미획득 NONE, 판단 불가 missing. |
| `elder_dragon_side` | 6.9 이전 N/A. 첫 장로 획득 팀, 양 팀 장로 0개이면 NONE. 양쪽 모두 획득하고 순서가 없으면 missing. |
| `first_baron_side` | firstbaron 기준 첫 획득 팀. 누락일 때만 한쪽 단독 획득/양쪽 0 같은 확정 가능한 barons 합계로 보완. 양 팀 1, 비정상 플래그, 플래그와 합계의 모순은 missing. 미획득 NONE. |

`blue_dragon_count`, `red_dragon_count`는 **경기 후 결과**이며 타깃 생성/분석용이다. 정상적인 `elementaldrakes`를 우선 사용하고, 필드가 비어 있을 때만 `dragons - elders`로 복원한다. 원본의 OE `dragons`는 장로를 포함한다. 누락·음수·비정수·잘못된 값, `dragons < elders`, 양수인 `dragons (type unknown)`, Soul 도입 이후 4마리 초과 등 확정 불가능한 count는 missing으로 남긴다. 알려진 반대편 count는 유지한다. 비정상 elementaldrakes를 다른 값으로 덮어서 추정하지 않는다.

6.9 이후 정상적인 `elementaldrakes`와 `dragons - elders`가 서로 다르면 해당 팀 count도 missing이다. Soul 도입 이후 양 팀 모두 4개인 모순은 어느 쪽이 틀렸는지 알 수 없으므로 양 팀 count를 missing으로 처리한다. 이벤트 입력에서도 Soul 획득 뒤 일반 드래곤, Soul 이전 장로, 음수/boolean timestamp는 거부한다.

**2015~6.9 이전 원본은 firstdragon이 기록되어 있는데도 dragons 합계가 전부 0**이다. 이 구간은 기존 보호 로직을 유지해 합계에서 count를 복원하지 않는다. count가 없으면 more/first-four도 missing이며, 정상적인 `firstdragon` 플래그는 독립적으로 사용할 수 있다. 시스템 도입 전 N/A는 Soul/Elder에만 적용한다. 유효한 별도 elementaldrakes 또는 완전한 이벤트 기록이 있으면 count 복원에 사용할 수 있다.

모든 7개 타깃과 분석용 count 2개의 값·생성 이유는 `target_audit.csv`에 경기당 9행으로 남긴다. count가 0인 경우는 실제 숫자 0으로 저장하며 missing과 구분한다. `validation_report.json`에는 7개 target의 BLUE/RED/NONE/N/A/TIE/MISSING 분포, `dragon_count_statistics`의 known/missing/zero, `missing_target_cells`, `missing_dragon_count_cells`, 열별 결측 수를 저장한다. `validation_summary.md`에도 타깃 분포와 count 결측 표를 출력한다.

완전한 이벤트 기록이 확보되면 다음 형식의 JSON을 `data/raw/`에 별도 보존하고 `preprocess --events data/raw/objective_events.json`으로 보완할 수 있다.

```json
{
  "실제 game_id": {
    "source": "원본 이벤트 파일 또는 출처 URL",
    "complete": true,
    "events": [
      {"timestamp_ms": 300000, "monster": "DRAGON", "side": "BLUE"}
    ]
  }
}
```

`first_dragon_side`의 불완전/모순된 OE 플래그는 이벤트를 넣더라도 missing으로 유지한다. 완전한 이벤트로는 count 2개와 more/first-four/Soul/첫 Elder/첫 Baron을 보완하며, 이미 확정된 값과 count가 충돌하면 거부한다.

위는 형식 예시다. 실제 입력에는 **경기 전체**의 DRAGON/ELDER/BARON 처치 이벤트를 오름차순으로 담아야 한다. `complete=true`는 공급자가 보증하는 완전성 선언이며 코드가 외부 기록의 누락까지 증명하지는 않는다. 부분 타임라인, 중복 이벤트, 기존 확정 값과 충돌하는 타임라인은 거부한다. 타임라인 자동 수집 API는 현재 원본에 없으므로 이 기능은 보완 입력 인터페이스다.

CSV를 pandas로 읽을 때는 `pd.read_csv(path, keep_default_na=False, dtype=str)`를 사용해야 문자열 `N/A`가 NaN으로 바뀌지 않는다. 제공하는 `read_games()`는 이 구분을 유지한다.

## 유연한 split과 vocabulary

사용자 JSON 설정 파일 예:

```json
{
  "train": {"years": [2022, 2023], "stages": ["Regular Season"], "date_to": "2023-12-31"},
  "test": {"years": [2024], "splits": ["Spring"], "stages": ["Playoff"], "date_from": "2024-01-01"}
}
```

```bash
.venv/bin/python main.py split --config my_split.json --output-dir data/splits/experiment1
```

`years`, `splits`, `stages`, `date_from`, `date_to`를 조합한다. 날짜 경계는 양 끝 포함이다. train/test game_id 중복 및 빈 split은 오류로 처리한다. 기본 split은 시간 순 holdout이 아니므로 과거 playoffs보다 이후 regular 데이터가 train에 들어갈 수 있다.

vocabulary는 정렬된 토큰에 연속 정수 ID를 부여하며 `<UNK>`는 항상 0이다. `vocab.get(name, vocab['<UNK>'])`로 인코딩할 수 있다. 전체 데이터 vocabulary는 탐색/공통 인덱스용이고, 학습에서는 `splits/`의 train 전용 vocabulary를 사용한다. 선수 토큰은 OE `playername`이다. 별칭 변경이나 동명이인의 신원을 수동 통합하지 않았으므로 고유 선수 수는 고유 **선수명 토큰 수**다. 이름이 바뀌면 다른 토큰이 된다.

## 원본 수집·보존

이미 있는 연간 CSV 12개를 재사용하며 바이트를 변경하지 않는다. `raw_manifest.json`에 SHA-256을 남긴다. 외부 수집이 필요하면 공식 다운로드 페이지에서 유효한 HTTPS URL을 확인하여 `collect --url-template '확인한 {year} 포함 URL'`에 전달한다. 기존 파일은 덮어쓰지 않는다. 새 스냅샷은 다른 `--raw-dir`에 수집하고 그 경로로 preprocess를 실행한다. URL을 코드에서 추정하거나 미확인 최신 파일로 기존 원본을 교체하지 않는다.

`collect`의 기본 종료 연도는 실행 시점의 현재 연도이며 `--end-year`로 제한할 수 있다. 수집과 전처리 모두 `2024_LoL_esports_match_data_from_OraclesElixir (1).csv` 같은 브라우저 다운로드 이름을 인식한다. 같은 연도 파일이 여러 개라면 자동으로 선택하거나 합치지 않고 오류를 낸다. 필요한 스냅샷을 별도 폴더에 모아 `--raw-dir`로 지정한다. 현재 원본 파일을 개명할 필요는 없다.

## 참고 출처

- [Oracle's Elixir 다운로드](https://oracleselixir.com/tools/downloads)
- [Riot 패치 9.23: 드래곤 영혼 도입](https://www.leagueoflegends.com/en-us/news/game-updates/patch-9-23-notes/)
- [Riot 개발자 설명: 6.9 드래곤 개편](https://nexus.leagueoflegends.com/en-us/2016/10/dev-on-depth-vs-accessibility/)
- [LCK 2025 대회 방식](https://lolesports.com/ko-KR/news/2025-lck-new-format/)
- [2025 시즌 playoffs 일정](https://lol.fandom.com/wiki/LCK/2025_Season/Season_Playoffs)
- [2026 시즌 playoffs 일정](https://lol.fandom.com/wiki/LCK/2026_Season/Season_Playoffs)

파일별 구현: `main.py`(CLI), `src/collect.py`(선택형 다운로드), `src/preprocess.py`(변환·vocabulary·검증), `src/splits.py`(별도 split), `tests/test_preprocessing.py`(경계·오류·누수 방지 테스트).
