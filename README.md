# Harness² skeleton

과제 두 개를 가장 단순하게 돌리는 시작 코드입니다. 과제마다 폴더가 따로 있고, 폴더 하나에 `model.py`와 `train.py`만 있습니다. 두 폴더는 서로 코드를 나누어 쓰지 않습니다. `train.py`가 데이터 받기, 학습, 평가를 하고, 마지막에 점수 한 줄(JSON)을 출력합니다. 자율연구에서는 주로 `model.py`를 고칩니다. 모두 처음부터(from scratch) 학습합니다.

| 폴더 | 과제 | model | 학습 | 점수 |
|---|---|---|---|---|
| `cifar10/` | CIFAR-10 이미지 분류 | `CNN`: conv 2층 + linear 1층 | 10 epoch, AdamW 1e-3 | `test_acc` (높을수록 좋음) |
| `shakespeare/` | Tiny Shakespeare 다음 글자 예측 | `GPTLanguageModel`: 6층, 폭 384, head 6, dropout 0.2 (약 1000만 parameter) | 5분 고정, AdamW 3e-4 | `val_loss` (낮을수록 좋음) |

```bash
uv venv .venv && source .venv/bin/activate && uv pip install torch torchvision
cd cifar10 && python train.py
cd shakespeare && python train.py
```

각 과제는 자기 폴더 안에서 실행합니다. 데이터는 그 폴더의 `data/`에 받습니다.

- CIFAR-10: 공식 train 5만 장에서 train 5천 장과 val 1만 장을 뽑고(나머지 3만 5천 장은 쓰지 않음), 공식 test 1만 장을 test로 씁니다. seed 0.
- GPT: karpathy "Let's build GPT" 강의 코드([ng-video-lecture](https://github.com/karpathy/ng-video-lecture) `gpt.py`)입니다. `shakespeare/model.py`의 `Head`~`GPTLanguageModel`은 원본과 글자 하나까지 같습니다(2026-11-27 원본과 대조). hyperparameter 값도 같고, `vocab_size`만 데이터에서 세지 않고 65로 적었습니다(세어도 65). 저장소에 license 파일은 없습니다.
- `shakespeare/train.py`가 원본과 다른 점은 세 가지입니다. 나머지(글자 단위 데이터, 앞 90% train, batch 64, AdamW 3e-4, seed 1337, 학습 step 내용)는 원본과 같습니다.
  - 5000 step 대신 5분(`TIME_BUDGET`) 동안 학습합니다.
  - 500 step마다 무작위 200 batch로 loss를 추정하지 않고, 끝에 val 전체로 한 번 잽니다. 그래서 학습 batch의 무작위 순서는 원본과 조금 다릅니다.
  - 끝에 글을 생성하지 않고 점수 한 줄을 출력합니다.

## 기준 점수 (RTX 3090 한 장, 2026-11-27)

| 과제 | 점수 | 학습 | 걸린 시간 |
|---|---|---|---|
| `cifar10/` | test_acc 0.543 | 10 epoch | 23초 |
| `shakespeare/` | val_loss 1.504 | 1967 step | 306초 |

강의 원본은 5000 step에서 val loss 약 1.48입니다. 같은 코드를 다시 돌려도 GPU 연산 때문에 점수가 조금 흔들립니다(CIFAR 0.538–0.543, GPT 1.504–1.505).
