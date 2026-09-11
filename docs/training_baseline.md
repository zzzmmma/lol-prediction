# Transformer baseline

## 입력과 모델

`src/dataset.py`는 전처리의 `INPUTS` 순서를 그대로 재사용한다. 팀명, 날짜, 연도, patch, dragon count와 다른 경기 후 결과는 모델 feature에 포함하지 않는다.

| token 위치 | 종류 | side | 역할 순서 |
|---|---|---|---|
| 0~4 | Player | Blue | TOP/JUNGLE/MID/ADC/SUPPORT |
| 5~9 | Player | Red | TOP/JUNGLE/MID/ADC/SUPPORT |
| 10~14 | Champion | Blue | TOP/JUNGLE/MID/ADC/SUPPORT |
| 15~19 | Champion | Red | TOP/JUNGLE/MID/ADC/SUPPORT |

Player와 Champion은 서로 독립된 vocabulary와 `nn.Embedding`을 사용한다. 각 token은 256차원이며, learnable position(20개), side(2개), role(5개), type(Player/Champion 2개) embedding을 더한다. LayerNorm과 dropout 후 Transformer Encoder를 적용한다. 모든 게임은 길이가 20이므로 padding/CLS token을 추가하지 않는다. `<UNK>=0`도 padding이 아닌 일반 embedding이다.

기본 Encoder는 2층, attention head 8개, feedforward 512차원, GELU, dropout 0.1, pre-norm이다. `--layers`로 2/3/4층을 선택할 수 있다. PyTorch Encoder의 복제된 layer들은 서로 다른 초기 가중치로 초기화한다. 출력 20개를 mean pooling해 256차원 경기 표현을 만들고 task별 독립 Linear head 7개에 전달한다. 현재 vocabulary에서 binary first-dragon 모델의 전체 파라미터 수는 1,197,587개다. 초기 3-class 버전은 1,197,844개였다.

## Class 및 마스킹

| task | class ID 순서 |
|---|---|
| winner_side | BLUE=0, RED=1 |
| first_dragon_side | BLUE=0, RED=1 |
| more_dragons_side | BLUE=0, RED=1, TIE=2 |
| first_four_dragon_side | BLUE=0, RED=1, NONE=2 |
| dragon_soul_side | BLUE=0, RED=1, NONE=2 |
| elder_dragon_side | BLUE=0, RED=1, NONE=2 |
| first_baron_side | BLUE=0, RED=1, NONE=2 |

빈 CSV 셀, 문자열 `MISSING`, `N/A`는 학습 tensor에서 `-100`으로 인코딩한다. 해당 task에서만 loss와 평가를 제외한다. 같은 경기의 나머지 정답은 계속 사용한다. `N/A`와 MISSING의 원본 의미는 `config.json`의 별도 개수로 보존하며, 원본 CSV는 수정하지 않는다.

각 minibatch에서 task별 유효 정답에 대한 평균 cross entropy를 구한 다음, 유효 정답이 있는 task들의 loss를 동일 가중치로 평균한다. 정답이 전부 마스킹된 task는 loss 합산에서 제외한다. 배치 전체에 유효 정답이 없으면 optimizer step 자체를 건너뛴다. `first_dragon_side`의 실제 `NONE`은 binary 문제 범위 밖이므로 해당 task에서만 제외하고 `excluded_nonbinary`로 별도 집계한다. 다른 task의 `NONE`/`TIE`는 마스킹하지 않는다.

2026-09-12 수정부터 class weighting을 기본 적용한다. `--class-weight`/`--no-class-weight`로 설정한다. 유효 train label 빈도에 대해 `weight[c] = N / (K_observed × count[c])`를 계산하며, 빈도 0인 class는 weight=0이다. `CrossEntropyLoss(weight=...)`와 동일하게 `F.cross_entropy(..., weight=...)`에 적용한다. 가중 평균의 분모는 유효 표본의 class weight 합이다. 0으로 나누는 task는 건너뛴다. test 빈도는 가중치 계산에 사용하지 않는다. 설정·빈도·가중치를 config와 checkpoint에 보존한다. 기존 `loss` 및 Accuracy/F1은 unweighted로 유지하고 실제 batch 목적함수 평균을 `optimization_loss`로 별도 기록한다. 추가 feature와 Encoder/embedding 구조는 변경하지 않았다.

## 학습 및 평가 정책

- 기본 AdamW, learning rate 0.0003, weight decay 0.01, gradient clipping 1.0, batch size 64, 10 epoch, seed 42다.
- Regular Season만 학습하고 Playoff만 평가한다. train/test game_id 중복과 잘못된 stage를 거부한다.
- train vocabulary가 train에서 관측한 이름 + `<UNK>`로 구성되어 있는지 검증한다. 전체 processed vocabulary를 잘못 넘기면 오류가 난다. test의 새 이름은 `<UNK>`로 매핑하고 개수를 기록한다.
- epoch 수는 시작 전에 정한다. test metric은 모든 epoch가 끝난 후 한 번만 계산하며, test로 early stopping·학습률 조절·best checkpoint 선택을 하지 않는다. 검증 데이터 분리는 이번 baseline에 추가하지 않았다.
- 각 task의 평가 분모는 마스킹을 제외한 표본 수다. Accuracy와 class별 precision/recall/F1/support, 혼동행렬을 저장한다. 혼동행렬 행은 정답, 열은 예측이다.
- Macro F1은 설정된 모든 class F1의 동일 가중 평균이다. 표본이나 예측이 없어 분모가 0인 class metric은 0으로 계산한다. task 전체의 유효 표본이 0이면 Accuracy/Macro F1/loss 및 class metric을 JSON `null`로 저장한다.
- epoch train 지표는 dropout을 켜고 학습 중 각 batch의 업데이트 전 예측을 누적한 값이다. epoch 마지막 모델로 train 전체를 다시 평가한 값은 아니다. dataset loss는 전체 유효 표본으로 가중 집계한 task 평균 loss의 평균이다.
- 자동 device 선택은 CUDA → MPS → CPU이며 `--device`로 명시할 수 있다. 기본 DataLoader worker 수는 0으로 Windows 직접 실행을 지원한다. seed를 저장하지만 서로 다른 OS/GPU/PyTorch 버전에서 bit 단위 재현까지 보장하지 않는다.

## 산출물

기본 저장 경로는 `runs/transformer_날짜_시각/`다. `--output-dir` 지정 시 비어 있는 새 폴더를 사용한다. `data/` 내부 저장과 기존 실험 덮어쓰기는 거부한다.

| 파일 | 내용 |
|---|---|
| config.json | 모델/optimizer 인수, seed, 버전/device, 원본 SHA-256, split 및 task 정답 개수, class mapping |
| player_vocab.json / champion_vocab.json | 이번 실행에서 실제 사용한 vocabulary 사본 |
| last.pt | 매 epoch 후 마지막 checkpoint. 가중치, optimizer state, epoch, 모델 설정, vocabulary, 입력 순서, task class 순서 포함 |
| train.log | 콘솔과 같은 epoch별 진행 및 최종 task/class 지표 |
| history.jsonl | epoch별 train 지표, 마지막에 test 지표 1개 |
| history.csv | epoch/split/task별 loss, Accuracy, Macro F1, valid/ignored 수 |
| test_metrics.json | 최종 task/class 지표와 혼동행렬, checkpoint/데이터 출처 |
| test_task_metrics.csv | 최종 task별 지표 |
| test_class_metrics.csv | class별 precision/recall/F1/support/predicted |

`src/evaluate.py --checkpoint .../last.pt`는 checkpoint의 vocabulary와 모델 설정을 로드하므로 별도의 vocabulary 경로가 필요 없다. `--test`로 다른 Playoff 파일을 지정할 수 있다. 재평가 결과는 기본적으로 새 `runs/evaluation_날짜_시각/`에 저장된다. checkpoint는 `weights_only=True`, CPU map_location으로 읽은 후 선택한 device로 이동한다. 학습 재개 CLI는 아직 제공하지 않는다.

## 최초 실행 결과 — 2026-09-11

아래는 **수정 전 unweighted·first-dragon 3-class** 실행 기록이다. 현재 binary 버전으로 새로 학습한 결과가 아니며, 3-class checkpoint는 새 모델로 로드할 수 없다. 가중치 효과를 비교하려면 현재 binary 모델에서 on/off를 각각 실행해야 한다. 아래 파라미터 수와 성능 수치는 당시 기준으로 보존한다.

`runs/baseline_20260911/`에 10 epoch 결과를 저장했다. Windows, Python 3.14.0, PyTorch 2.11.0+cu128, RTX 4060 Ti에서 train 4,880경기와 test 438경기를 사용했다. train loss는 첫 epoch 0.8463에서 마지막 epoch 0.6544로 감소했다.

| task | test 유효 표본 | 제외 표본 | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| winner_side | 438 | 0 | 0.6096 | 0.6083 |
| first_dragon_side | 438 | 0 | 0.5525 | 0.3395 |
| more_dragons_side | 399 | 39 | 0.5113 | 0.3703 |
| first_four_dragon_side | 399 | 39 | 0.4937 | 0.3720 |
| dragon_soul_side | 303 | 135 | 0.4554 | 0.3717 |
| elder_dragon_side | 398 | 40 | 0.8417 | 0.3055 |
| first_baron_side | 437 | 1 | 0.5469 | 0.3615 |

이 결과는 baseline 측정값이며 튜닝한 성능이 아니다. `first_dragon_side`는 train/test에서 NONE 표본이 0개지만, 정의된 3개 class를 사용하므로 해당 class의 0점도 Macro F1에 포함된다. Elder의 BLUE/RED recall은 둘 다 0이며 NONE 비중이 크기 때문에 Accuracy가 높다. class별 지표와 함께 판단해야 한다.

기존 전처리/EDA 테스트와 신규 학습 테스트를 합해 53개가 통과했다. 학습 전후 train/test CSV와 vocabulary 2개의 SHA-256이 동일했고, 저장 checkpoint를 독립 평가 CLI에서 다시 읽은 task/class 지표도 최초 평가와 모두 일치했다. 마스킹된 label의 gradient가 0인지, 외부 작업 폴더에서 파일 직접 실행이 되는지, checkpoint 로드 전후 예측이 같은지를 포함해 검증했다.

이 stage split은 미래 시간 holdout이 아니어서 일부 test 경기보다 나중에 열린 정규시즌 경기가 train에 포함된다. 독립 head 구조는 Soul과 first-four의 예측 일치 같은 task 간 제약을 강제하지 않는다. 현재 보고된 성능을 미래 시즌 예측 성능이나 서로 일관된 오브젝트 시나리오의 보장으로 해석하지 않는다.

## 참고

- [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)
- [TransformerEncoder](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TransformerEncoder.html)
- [모델 저장 및 불러오기](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
