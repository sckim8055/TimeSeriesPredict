# BIVA 원본 사본 — 수정 목록

원본: `D:/test/SVU_SCT_FIN-main/Model/BIVA/model` (BIVA.py 2026-03-18, BIVA_RITS.py 2026-02-28, BIVA_BRITS.py 2025-07-05, BIVA_LSTM_VAE.py 2025-05-07).
복사 직후 해시: `ORIGIN_SHA256.txt`. 원본과의 차이: `python tools/biva_src_diff.py`. 표시 없는 수정 검사: `--check`(테스트에 포함).

| # | 파일 | 원본 | 수정 | 왜 | 결과값 영향 |
|---|---|---|---|---|---|
| 1 | BIVA_BRITS.py | 불러올 때 `os.chdir('D:\...')` | 줄 제거 | 다른 컴퓨터에서 오류, 플랫폼 전체의 작업 폴더를 바꿈 | 없음 |
| 2 | BIVA.py, BIVA_BRITS.py | `from model import ...`, `import BIVA_RITS` | `from . import ...` | 패키지 안에서 불러오기(원본은 chdir 에 의존) | 없음 |
| 3 | BIVA_RITS.py, BIVA_BRITS.py | `.cuda()` 고정, 은닉상태 h·c 는 항상 CPU | 입력과 같은 장치 | 원본은 CPU 전용 컴퓨터에서도, GPU 에서도(h·c 불일치) 오류 | 없음 |
| 4 | BIVA.py `RIN` | `off_RIN` 이 정의 안 된 `means`·`stdev` 사용 | `set_RIN` 에서 보관해 사용 | `RIN=True` 면 항상 오류 | RIN=False(원본 기본) 영향 없음 |
| 5 | BIVA.py, BIVA_LSTM_VAE.py | 중간 크기 `int(channels/4)`, `int(channels/16)`, `channels//2` | `max(1, ·)` | 변수 16개(VAE 는 4개) 미만이면 크기 0 → 생성 오류 | 변수 16개 이상이면 원본과 같음 |
| 6 | BIVA_RITS.py `delta_calc` | 배치×시점×변수 3중 파이썬 반복 | 변수 축 반복 1개(값 동일), 원본은 `delta_calc_orig` 로 보존 | 학습 시간 대부분을 차지(원본 6~7시간의 주원인) | **없음** — 무작위 결측 패턴에서 비트 단위 동일 시험 |

바꾸지 않은 것(원본 그대로): 신경망 구조, 초기화 순서, 손실식(VAE: 0.5·MSE + 0.5·KLD — KLD 가 잠재 차원 벡터로 남아 torch.sum 되므로 실제 = 0.5·잠재·MSE + 0.5·ΣKLD, 보간 손실 × imputed_weight, 일관성 손실 ×0.1),
`c_h = |0.5·z_h + 0.5·x_h|`(α 계산 후 쓰지 않음), sigmoid 보간 추정, delta 규칙의 변수 축 d−1 참조(원본 그대로의 규칙),
VAE 표본추출이 예측 때도 일어나는 것, dropout 정의 후 미사용.

검증: `tests/test_biva_src.py` — 같은 시드에서 원본 모듈과 사본의 초기 가중치·6개 출력이 빈 칸 있는 입력에서 비트 단위로 같음.
