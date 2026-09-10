# 데이터셋 점검 결과 — 2026-09-10

기존 파이프라인을 보완했으며 새 전처리 파이프라인이나 학습 코드를 만들지 않았다. 저장소의 Python 파일, README, 의존성 목록, Git 추적 상태, 원본 12개와 기존 산출물을 확인했다. 세부 폴더 구조와 실행 명령은 [README](../README.md)에 있다.

## 요구사항 대조

| 항목 | 확인 결과와 조치 |
|---|---|
| Oracle's Elixir 연간 CSV 수집 | 기존 HTTPS 템플릿 기반 수집과 원본 보존을 유지. Windows 다운로드 이름의 `(1)` 접미사도 기존 파일로 인식 |
| 2015~2026 LCK / 2015 OGN | 기존 league 필터 유지. 실제 12개 원본 전체 재처리, OGN 및 타 리그 제외 회귀 테스트 추가 |
| 향후 연도 추가 | 전처리 종료 연도는 보유 파일의 최대 연도, 수집 종료 연도는 현재 연도. 기본 split의 2026 상한 제거 |
| Regular Season / Playoff | 기본 train/test 분류 유지. Cup, 승강전, 선발전, Road to MSI, Play-In은 별도 stage로 보존하고 기본 split에서 제외 |
| 새 시즌 포스트시즌 | 기존 날짜 경계를 `config/stage_calendar.json`으로 이동. 검증된 경계 추가로 확장하며, 없는 경계를 임의로 추정하지 않음 |
| 한 경기 1행 | 12개 원본 참가자 행을 팀 2개와 선수 슬롯 10개로 검증하는 기존 구현 유지 |
| 입력 token 20개 | `INPUTS` 순서 유지: Blue 선수 5, Red 선수 5, Blue 챔피언 5, Red 챔피언 5. 각 그룹은 TOP/JUNGLE/MID/ADC/SUPPORT |
| 별도 vocabulary | player/champion 및 `<UNK>=0` 유지. 학습용 vocabulary는 train에서만 생성 |
| 필수 메타데이터 | 누락된 `blue_team`, `red_team` 추가. 팀 행과 선수 5명의 `teamname` 일치 검증 |
| target 7개 및 분석 count 2개 | 기존 생성기 유지. 합계·플래그 모순과 불가능한 이벤트 순서 검증 보강 |
| NONE / N/A / MISSING | `NONE`과 문자열 `N/A`, 빈 CSV 셀(MISSING)을 별도로 보존. strict 검증 시 target 결측은 종료 코드 2 |
| 입력 누수 방지 | 팀명은 metadata, dragon count는 analysis only, 결과 7개는 target으로 분리. `INPUTS`에 추가하지 않음 |
| 256차원 PyTorch Transformer | 아직 미구현. 별도 `nn.Embedding`과 위치/side/role 표현은 이후 모델 단계의 요구사항으로 README에 명시 |

`src/dataset.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`, `src/utils.py`는 기존 빈 파일 상태를 유지했다. 추가 경기력 feature도 도입하지 않았다.

## 실제로 수정된 정답

다음 4경기는 양 팀 `firstbaron=0`인데 적어도 한 팀의 `barons` 합계가 양수다. 기존 코드는 이 모순을 검사하지 않아 `NONE`으로 저장했다. 어느 관측이 잘못됐는지 원본만으로 결정하지 않고 `first_baron_side`를 MISSING으로 바꿨다.

| game_id | Blue barons | Red barons | 변경 |
|---|---:|---:|---|
| ESPORTSTMNT01/1325195 | 0 | 2 | NONE → MISSING |
| ESPORTSTMNT01/1315361 | 0 | 1 | NONE → MISSING |
| ESPORTSTMNT03/1351811 | 2 | 0 | NONE → MISSING |
| ESPORTSTMNT03/1442655 | 1 | 0 | NONE → MISSING |

그 밖의 기존 데이터 열 값은 변경 전 Git 버전과 모두 동일했다. 추가한 두 팀명은 원본 그대로이며, OE에서 과거 팀명을 현재 명칭으로 정규화했을 가능성까지 교정한 역사적 팀명 데이터는 아니다.

추가한 방어 로직은 `elementaldrakes`와 `dragons - elders`가 다를 때 count를 결측 처리하고, Soul 시대의 양 팀 4드래곤을 확정된 동률로 쓰지 않는다. 이벤트에서는 Soul 이후 일반 드래곤과 Soul 이전 장로를 거부한다. 해당 규칙의 근거는 [Riot 9.23 패치 노트](https://www.leagueoflegends.com/en-us/news/game-updates/patch-9-23-notes/)의 Dragon Souls 및 Elemental Drake Mechanics다. 기존 Elder 도입 경계는 유지했다. 이번 원본에서는 이 방어 로직으로 바뀐 dragon count는 없었다.

## 재실행 결과

| 검증 항목 | 결과 |
|---|---:|
| 원본 연간 CSV | 12개, 변경 전후 SHA-256 모두 동일 |
| 처리된 게임 | 5,863 |
| games.csv 열 | 37개: metadata 8 + input 20 + target 7 + analysis 2 |
| 기본 train | 4,880 |
| 기본 test | 438 |
| split 제외 | 545: 기타 stage 544 + 승자 미확정 1 |
| train/test game_id 중복 | 0 |
| 구조 오류 | 0 |
| 감사 레코드 | 52,767 = 5,863 × 9, 중복 없음·저장된 값과 일치 |
| 고유 선수명 / 챔피언 | 365 / 171 |
| 회귀 테스트 | 39개 통과 |

| target | MISSING 셀 |
|---|---:|
| winner_side | 1 |
| first_dragon_side | 0 |
| more_dragons_side | 682 |
| first_four_dragon_side | 682 |
| dragon_soul_side | 44 |
| elder_dragon_side | 77 |
| first_baron_side | 4 |

target 결측은 총 1,490셀, 분석용 dragon count 결측은 1,356셀이다. 2015~6.9 이전 zero-filled dragon 합계, 알 수 없는 드래곤 종류, 양 팀 장로 획득 순서 등 원본 한계는 그대로 결측으로 남긴다. 모든 게임에 모든 target이 있는 데이터셋은 아니다. 학습 시 target별 결측 마스킹이 필요하며, `N/A`는 결측으로 자동 변환하면 안 된다.

실제 확인한 실행은 전체 전처리, CSV 재검증, 기본 split, strict 검증, 회귀 테스트다. 외부 다운로드는 새로 실행하지 않았고, 수집은 기존 원본 보존 경로를 검증했다. `requirements.txt` 전체 설치와 PyTorch 학습은 이번 검증 범위에 포함하지 않았다.

원본 스냅샷 날짜 범위는 2015-01-07~2026-09-06이다. 공식 경기 목록과 모든 세트를 독립 대조한 완전성 인증은 아니다. 기본 stage split은 시간 순 holdout이 아니므로 과거 playoffs보다 뒤의 regular 경기가 train에 포함될 수 있다. 선수명 vocabulary도 별칭·동명이인 신원을 통합한 player ID가 아니다.

## 변경 파일

| 파일 | 변경 내용 |
|---|---|
| `main.py` | 동적 종료 연도, `--stage-calendar`, validate strict 옵션, UTF-8 JSON |
| `src/preprocess.py` | 원본 파일 탐색, 팀 메타데이터, 일정 설정, target 및 구조 검증 |
| `src/collect.py` | 다운로드 접미사가 붙은 기존 원본도 재사용 |
| `src/splits.py` | 기본 연도 상한 제거, 기존 stage 분할과 train 전용 vocabulary 유지 |
| `config/stage_calendar.json` | 기존 승강전/포스트시즌 날짜 경계 외부화 |
| `tests/test_preprocessing.py` | 기존 테스트 유지·스키마 조정, 회귀 테스트 13개 추가 |
| `README.md` | Windows/macOS 실행, 37열 스키마, 확장 방법 및 결측 규칙 |
| `docs/dataset_review.md` | 요구사항 대조와 검증 결과 |
| `data/processed/games.csv` | 팀명 추가 및 Baron target 4개 수정 |
| `data/processed/game_provenance.csv`, `raw_manifest.json` | 실제 Windows 원본 경로/파일명, 일정 파일 해시 |
| `data/processed/target_audit.csv` | Baron 판정 사유 갱신 |
| `data/processed/validation_report.json`, `validation_summary.md` | 현재 검증 결과 갱신 |
| `data/splits/train.csv`, `test.csv`, `split_manifest.json` | 갱신된 스키마와 target 반영, 필터·입력 해시 갱신 |

vocabulary 4개와 `structural_errors.json`은 재생성했지만 내용은 기존과 같았다. 원본, 빈 모델 파일, 기존 의존성 목록은 수정하지 않았다.
