# 점검 보고 — 영역: 모델 구현 (models/*, biva_src/, engine/experiment.py 의 호출부)

작업 파일: /tmp/rv6_models/ (t1_window_align.py · t2_lgbm_es.py · t2b_lgbm_all.py · t3_arima.py · t4_sarima.py · t5_lstm.py · t6_biva.py). 저장소는 읽기만 함.
환경: CPU 2스레드, torch 2.15, lightgbm 4.7, statsmodels 0.15, sklearn 1.9. GPU 없음(장치 관련은 코드 읽기만).

## 1. 총평

모델 파일들의 **정렬·누출 측면은 깨끗하다** — 창 모델의 윈도우·테스트 창 구성에 off-by-one 없음(가짜 모델로 직접 확인, 차이 0.0), 스케일러·타겟 표준화는 학습 구간만, 규칙 모델 지표에 미래 참조 없음, 시리즈·수준 모델은 결정일까지의 값만 쓴다. 반면 **"값이 맞다"에 해당하는 추정 품질 쪽은 방어가 어렵다**: (a) 일간 LightGBM 은 조기종료가 "안 되는" 것이 아니라 **1~2 그루에서 바로 멈추는** 폴드가 149개 중 45개(중앙값 9그루)라 사실상 '학습 구간 평균 + 미세 조정' 모델이고, (b) SARIMA 기본값(D=1)은 변화율 입력에서 계절 과차분이라 폴드 120개 중 79개에서 Θ≈−1(중앙값 −0.967)로 붙는 **모형 오설정**이며, (c) LSTM·GRU 월간 설정은 48개 창에 2,449개 가중치(학습 R² 0.90, 시드 간 예측 표준편차 ≈ 예측 크기)로 결과가 시드 잡음이다. (d) 월간 Ridge alpha=100·LightGBM 월간 기본값은 **개발 구간 결과를 본 뒤** 정해졌다(results_round1 2026-11-03 → DECISIONS 11-04). (e) 저장된 결과 49개 전부 `code_version="nogit"` 이라 절대규칙 7(커밋 기록)이 한 건도 지켜지지 않았다. (f) GRU·SARIMA 는 CLAUDE.md 가 요구하는 합성 테스트(심은 신호·잡음 무능)가 없다. BIVA 는 원본 재현이라는 목표에는 충실하나(사본 비트 동일 시험 7개 통과 확인), 검증 손실이 같은 가중치에서도 16% 흔들리는 구조(셔플·drop_last·VAE 표본)라 '최저 검증 epoch' 선택이 잡음 선택이다.

## 2. 발견 목록

### 치명
없음(결과를 무효로 만드는 누출·정렬 오류는 이 영역에서 찾지 못함).

### 중대

**M1. 일간 LightGBM: 조기종료가 1~2 그루에서 멈춤 → 사실상 상수 예측(알려진 "조기 종료 미작동"은 월간에만 맞음)**
- 근거: `models/lightgbm_model.py:26` `early_stopping(50)`, 엔진 `_split_fit`(experiment.py:375~387) 로 검증 15%·purge. 05_lgbm_A 설정 전체 149 폴드 재적합(t2b_lgbm_all.py): best_iteration 최소 1·중앙값 9·최대 280, **≤2 그루 45개, ≤5 그루 62개, 300 도달 0개**. 저장 결과 `05_lgbm_A_v1107` 예측 표준편차 1.4e-3, 양수 비율 71%(= 거의 항상 롱). 06·07·08 도 같은 패턴(양수 비율 68~71%).
- 영향: 일간 LightGBM 결과는 "입력을 쓴 모델" 이 아니라 **학습 평균의 부호(롱) + 1~2 그루의 미세 보정**이다. 논문에서 "LightGBM 도 신호 없음" 은 맞지만 "LightGBM 을 학습시켜 비교했다" 는 서술은 공격받는다(모델이 사실상 학습을 거부한 것). `degenerate_model` 검사(experiment.py:786, 중요도 합=0)는 1그루라도 있으면 통과시키므로 잡지 못한다.
- 월간(12_lgbm_ABCD_M): 검증 0(8 < 10) → 150~200 그루 전부, 학습 R² 0.83(DECISIONS 11-07) — 이쪽은 알려진 대로 조기종료 없음·과적합.
- 방향: 결과 json 에 폴드별 `best_iteration` 분포를 기록하고, 퇴화 판정을 "중요도 합 0" 에서 "사용 그루 수 ≤ k 또는 예측 분산/타겟 분산 < ε" 로 넓힐 것. 논문 표에는 '실효 트리 수' 를 같이 적을 것. 확실.

**M2. SARIMA 기본값 (1,0,1)(1,1,1,s) 는 변화율 입력에서 계절 과차분 → 계절 MA 가 경계에 붙는 오설정**
- 근거: `models/sarima.py:30~36` D 기본 1, trend 없음. f11_sarima_k1 120 폴드 재적합(t4_sarima.py): Θ(ma.S.L12) 분위 — 10% −0.992 · 25% −0.988 · **중앙값 −0.967** · 75% −0.548; |Θ|>0.9 가 87개(−쪽 79, +쪽 8 — DECISIONS 11-08 의 79/9 와 일치). 수렴 안 됨 0개(scale 100).
- 해석: Θ≈−1 은 "(1−B¹²) 차분을 MA 로 되돌리고 있다" 는 과차분의 교과서적 신호. 월간 변화율에 계절 단위근이 있다는 근거가 없고, 노트북이 수준값에 (1,1,1)(1,1,1,12) 를 쓴 것을 그대로 옮긴 결과. ARIMA 쪽도 22/120 폴드에서 MA 근이 1.01 미만(θ 경계).
- 영향: SARIMA 결과(naive 미달)는 "계절 모형이 못 맞혔다" 가 아니라 "오설정 모형" 의 결과. 심사에서 "왜 계절 차분을 했나, 검정했나" 에 답이 없다.
- 방향: D 기본을 0 으로 두고 계절 AR/MA 만, 또는 계절 단위근 검정(OCSB/CH)·정보기준으로 D 를 폴드별 선택. 노트북 재현은 '옵션' 으로 남긴다. 확실.

**M3. LSTM·GRU 월간: 표본 대비 과대 모델 → 결과가 시드 잡음(알려진 문제의 정량 확인 + 설정값이 주기 기본값보다 큼)**
- 근거: `configs/f11_lstm_macro_k1.json` hidden 16·epochs 60(월간 기본은 hidden 8·epochs 200, `models/lstm.py:24,28`). 마지막 폴드 재현(t5_lstm.py): 창 48·검증 0(7 < 10)·**가중치 2,449개**·60 epoch 마지막 가중치(best_epoch None)·학습 창 R² 0.901. 시드 1~5 예측(3 폴드) 표준편차/|예측| 중앙값 = **1.06**, 부호도 시드마다 바뀜(2023-11: +0.00098 / −0.00054 / +0.00106 …). 같은 시드 CPU 2회 차이 0.0(결정적).
- 영향: 월간 LSTM/GRU 의 RMSE·방향 수치는 한 시드의 실현값. "LSTM 이 naive 보다 나쁘다" 는 결론은 유지되나, 모델 간 순위(LSTM vs GRU vs Ridge)는 근거가 없다.
- 방향: 여러 시드 평균·분산을 결과에 기록(또는 시드 앙상블)하고, 월간은 hidden 4~8·weight_decay 상향·입력 변수 축소. 설정이 기본값을 덮어쓴 이유(16, 60)를 DECISIONS 에 남길 것. 확실.

**M4. 하이퍼파라미터 기본값 일부가 개발 구간 결과를 본 뒤 정해짐(개발 구간 = 사실상 튜닝 표본)**
- 근거: `docs/results_round1.md` "월간 재실험 — Ridge (2026-11-03)" alpha 1/100/1000 비교표(100 이 샤프 최고) → 다음 날 `models/ridge.py:12` `default_by_freq M: 100.0` 확정(DECISIONS 11-04 저녁). LightGBM 월간 `min_child_samples` 도 40→8 시도(results_round1) 후 M 기본 6·num_leaves 4·n_estimators 150. alpha=100 은 n=58 에서 OLS 대비 계수를 평균 0.43 배로 수축(합성 확인).
- 영향: 개발 구간(테스트 폴드) 성과로 기본값을 고른 것이므로 개발 구간 수치는 낙관 편향. 봉인 구간이 이를 막는 장치이나, 봉인 평가는 "후보당 1회" 라 봉인 결과가 나오기 전까지의 모든 비교표가 해당. DSR 의 시도 수(N)에 이 탐색이 들어갔는지 결과 파일로는 알 수 없음(제 영역 밖).
- 방향: DECISIONS 에 "기본값 선택에 쓴 실험 목록" 을 명시하고, 논문에는 기본값을 '사전 고정값' 으로 쓰지 말 것. 학습 창 안 시간순 CV(또는 검증 15%)로 alpha 를 고르는 옵션이 정석. 확실(문서 근거), 의도는 불확실.

**M5. 저장된 결과 49개 전부 `code_version = "nogit"` — 절대규칙 7(커밋 기록) 미충족**
- 근거: `engine/experiment.py:289~304` `_code_version` 은 git 없으면 "nogit". results/*/*.json 49개 집계: code_version {nogit: 49}, device {cpu 38, cuda 11}, seed {2025: 48, 42: 1}.
- 영향: 어떤 결과도 "이 커밋으로 냈다" 를 증명할 수 없다. `engine_rev` 날짜가 대신하지만 모델 파일 변경(예: ARIMA 제약 기본값 11-06, 배율 11-07, 창 눈금 11-07)을 추적하지 못한다 — LEGACY_DEFAULTS 라는 우회로가 생긴 원인이기도 하다.
- 방향: git 초기화(저장소에 `git_first_push.bat` 는 있음) 후 결과 재생성, 또는 최소한 모델 파일들의 sha256 을 결과에 기록. 확실.

**M6. GRU·SARIMA 에 합성 테스트(심은 신호 찾기·잡음 무능)가 없음 — CLAUDE.md 머리말·절대규칙 6 위반**
- 근거: `tests/test_engine.py:119,131,170,180` 파라미터화 대상은 ridge·lightgbm_model·lstm·biva 뿐. `pytest --collect-only` 에서 gru 는 test_explain(재현)·codewalk 에만, sarima 는 제약·배율·explain 시험에만 등장. VECM 은 단위시험(test_engine.py:412)은 있으나 엔진 경로(levels 분기 experiment.py:431~461)를 통과하는 심은-신호 시험은 없음. sox_gap_fade 는 분위수 시점 시험만.
- 영향: 규약상 '등록' 요건 미충족. GRU 는 LSTM 복제라 위험은 낮지만, 규약을 스스로 어긴 사실은 심사에서 신뢰를 깎는다.
- 방향: 네 모델에 planted/no-edge 시험 추가(SARIMA 는 계절 AR 신호 심기). 확실.

### 경미

**m1. 합성 테스트의 변별력이 낮음** — `tests/synthetic.py:30` 신호 0.02·s_t(sd≈1.4) vs 잡음 0.01 → 신호 R²≈0.89, 방향 정확도 기대 ≈0.9. 임계 0.75/0.70 은 "정렬이 맞고 누출이 없다" 만 확인하고, 신호의 10% 만 잡는 모델도 통과. 랜덤워크 시험(0.40~0.60, 1,000개)은 55% 짜리 부분 누출을 못 잡는다. 방향: 약한 신호(R² 0.05 수준)·DM 기반 시험 추가.

**m2. ARIMA 수렴 실패가 결과에 기록되지 않음** — `models/arima.py:52` `.fit(disp=False, maxiter=200)` 뒤 `mle_retvals["converged"]` 미확인(explain.py:421 에서만 봄). f11_arima 120 폴드 중 2개 미수렴(scale 100) → 그대로 예측에 쓰임. 방향: 폴드 진단(`diag`)에 converged·llf·근 크기 기록(VECM 처럼).

**m3. 시리즈 모델 `apply` 의 정보집합 = 자료 시작부터(학습 창 밖 과거 포함)** — experiment.py:477 `r_all.loc[:t]`. 누출 아님. 크기(t3/t4): ARIMA 는 전체이력 vs 학습창이력 예측 차이 중앙값 1e-15 이나 MA 근이 경계인 폴드에서 최대 5.0e-3(예측 크기 중앙값 3.5e-3), 부호 반전 0; SARIMA 는 중앙값 3.1e-4(예측의 11%), 최대 6.1e-3, **부호 반전 4 폴드**, RMSE 는 거의 같음(0.00385 vs 0.00382). 방향: 논문에 "모수는 60개월 창, 상태(칼만 필터)는 2009-02 부터 누적" 으로 명시하거나 학습 창 이력만 넘기도록 통일.

**m4. BIVA 검증 손실이 같은 가중치에서도 흔들림 → '최저 검증 epoch' 선택이 잡음 선택** — `models/biva.py:180~185` 검증 DataLoader shuffle=True·drop_last=True(원본 재현), `_vali` 에 VAE 표본추출 포함. 합성(t6_biva.py, 창 10·배치 4): 같은 가중치 검증 손실 5회 변동 16%, drop_last 로 2창 버림. 같은 가중치 예측 2회 차이가 예측 표준편차의 3.4배(3 epoch 소형 모델이라 과장됨; 실데이터 기록은 DECISIONS 11-07 표준편차 3.7e-5). patience 500 = 500 epoch 전체에서 argmin → 잡음 최소점 선택(실제로 epoch 1 선택됨, DECISIONS 11-07). 월간 F11(창 48)은 검증 0 → **학습 손실로 선택**(LSTM 은 마지막 가중치) — 두 창 모델의 선택 규칙이 다름. 원본 재현 목적이라 코드는 두되 논문에 '원본 구현의 성질' 로 적고, 비교용으로는 shuffle=False·drop_last=False·z=μ 옵션을 권함.

**m5. 플러그인 규약(절대규칙 3) 소소한 위반** — `models/biva.py:60,132` `init_checkpoint` 파일 경로를 모델이 받아 `torch.load`; `models/rsi_rule.py:4` 가 `engine.features` 를 import(규칙 5: 모델은 엔진을 import 하지 않음). `models/base.py:7` 모듈 docstring 의 INPUT_SHAPE 목록에 `levels` 없음(CLAUDE.md 요약도).

**m6. 시드 소비 구조** — `_seed` 는 실험당 1회(experiment.py:653). 폴드 N 의 초기 가중치는 앞 폴드들이 쓴 난수 뒤의 값(DECISIONS 11-07 이미 기록). 따라서 test_start·폴드 수·앞 폴드의 epoch 수가 바뀌면 **모든 뒤 폴드 결과가 바뀐다**(같은 설정의 봉인 평가도 개발 평가와 다른 난수 흐름). LightGBM 은 `random_state=0` 고정(알려짐). GPU 결정성 플래그(`cudnn.deterministic`, `use_deterministic_algorithms`) 없음 — 코드 전체 grep 결과 0건(알려진 장치 민감도의 원인). 방향: 폴드마다 `seed + fold_index` 로 재시드(DECISIONS 에 검토 흔적 있음), GPU 는 플래그 추가.

**m7. 월간 LightGBM 첫 폴드들의 실제 학습 행 35~37(설정 60)** — 변수가 늦게 시작해 dropna 후 남는 행(t2_lgbm_es.py). `len(tr) < 30` 에서만 건너뛰므로 35행·35열로 150~200 그루 학습. 결과 json `periods.first_fold.n_train` 에는 남으나 표에 "학습 창 60" 으로 적으면 틀림.

**m8. VECM 수준값의 시점 혼합** — `ds.levels` 는 발표시차를 적용한 값(experiment.py:237)이라 VECM 의 y_t 벡터는 [F11_t, B1_{t−3개월}, …]. 정보집합으로는 맞지만 "공적분 관계" 의 뜻이 '시차 섞인 수준 간 관계' 로 바뀐다. 결정항 n·r=1 고정은 알려진 문제(검정 120 폴드 모두 r≥2) — 추세 있는 지수에 상수 없음(n)은 드리프트를 ECT 가 떠안게 해 체계적 과소예측. 논문 서술에 명시 필요.

### 건의

**S1.** 모델별 '실효 복잡도' 를 결과에 자동 기록(LightGBM 사용 그루 수·LSTM 가중치 수/창 수·ARIMA 근 크기·수렴·VECM diag 는 이미 있음). 비용 작음, 심사 방어에 직접 쓰임.
**S2.** 신경망(LSTM·GRU·BIVA)은 시드 ≥5 반복을 기본으로 하고 평균·표준편차를 표에. 월간은 폴드당 수초라 비용 작음(일간 BIVA 는 GPU 필요).
**S3.** SARIMA 는 D 선택 규칙(검정 또는 정보기준)을, ARIMA 는 다중 출발점 추정(이미 DECISIONS 11-09 에서 확인된 더 높은 극대)을 옵션으로.

## 3. 확인해서 문제없던 것

- 창 구성/테스트 창 정렬: `cal.make_windows`(calendar.py:233) 창 마지막 행 = 결정일, 타겟 y[t]; 테스트는 직전 seq_len−1 행을 붙임(experiment.py:520~531). 가짜 모델(창 마지막 행 반환)로 pred[t] 역변환 = x[t] 차이 0.0, 첫 테스트 창의 앞 4행 = 테스트 직전 4행 확인(t1).
- 창 모델 검증 분할의 purge(`_split_fit` emb) — 학습 마지막 타겟 종료 < 첫 검증 결정일(계산으로 확인).
- 스케일러·타겟 표준화(LSTM/GRU/BIVA μ_y·σ_y)는 학습 구간만. 기울기 자르기 1.0, 드롭아웃은 fit 끝의 `eval()` 로 추론 때 꺼짐, 은닉 상태는 배치마다 0 초기화.
- Ridge: 닫힌 해 = sklearn coef(1.3e-18), 절편 = ȳ, 엔진 표준화 뒤 파이프라인 표준화는 항등. 분류는 C=1/alpha.
- LightGBM: 표준화는 분위 binning 에 영향 없음(알려짐), bagging/feature_fraction 시드는 random_state 에서 파생, n_jobs=1 로 결정적.
- ARIMA/SARIMA: scale 배율은 같은 모형(기존 시험), LEGACY_DEFAULTS·pinned_model_params 로 옛 결과 재현 경로 있음. 제약 기본 True 는 statsmodels·노트북과 같음.
- VECM: 재귀식 = statsmodels predict(기존 시험 6조합), level→return 변환이 예측/매매 타겟 정의와 일치(forecast: path[k−1]/v_t, trading: path[k]/path[0]), r ≤ K−1, dof 검사, 상수열 제외.
- 규칙 모델: MA·모멘텀·RSI(Wilder ewm)·SOX(shift(1) 뒤 rolling quantile) 모두 t 까지의 값만. `engine/features.py` 에 center/bfill/shift(−) 없음.
- BIVA 사본: `tests/test_biva_src.py` 7개 통과(원본 모듈과 비트 동일) — 직접 실행. PATCHES 6건은 경로·장치·채널 0 방지·속도뿐이고 손실식·구조 불변. 수정 #5 는 f11(변수 20)·repro(33)에서 원본과 같은 채널 수.
- 빠른 합성 시험 7개(ridge/lgbm planted·no-edge, series AR, VECM, SOX 분위) 통과.

## 4. 확인 못 한 것

- GPU: cuDNN 비결정성 실측, BIVA·LSTM CPU/GPU 예측 차이 크기(장치 없음). 코드상 결정성 플래그가 없다는 것만 확인.
- BIVA 500 epoch 죽은 ReLU·epoch 1 선택의 재현(10분 제한) — DECISIONS 기록에 의존.
- `legacy/biva_orig` 와 원본 D:\ 의 동일성(`tools/biva_src_diff.py --check` 는 원본 경로 필요).
- 하이퍼파라미터 탐색(alpha 1/100/1000, min_child 8, sox q75/q90/bottom50)이 DSR 의 시도 수 N 에 들어갔는지(결과·카운터 영역).
- VECM 결정항 'co' 로 바꿨을 때의 F11 수치(알려진 "다른 설정도 naive 미달" 은 재확인 안 함).
- 일간 LightGBM 이 1~2 그루에서 멈추는 것이 `learning_rate 0.03·patience 50` 조합 때문인지 신호 부재 때문인지(둘 다로 보이나 분리 실험 안 함).

## 5. 교수가 물을 법한 질문과 지금 답할 수 있는지

1. **"LightGBM 을 어떻게 학습했고, 실제로 몇 그루를 썼나?"** — 지금 결과 파일로는 답 못 함(best_iteration 미기록). 재적합하면 일간 중앙값 9그루·45폴드는 1~2그루. 답하려면 M1 기록 추가 필요.
2. **"SARIMA 에서 계절 차분을 왜 했고, 계절 MA 계수가 −1 에 붙는 것을 어떻게 해석하나?"** — 코드·문서에 근거 없음(노트북 승계). 과차분이라는 답이 정직하고, 그러면 "오설정 모형과 비교했다" 가 된다. M2.
3. **"LSTM 결과는 시드를 바꾸면 얼마나 달라지나? GRU 가 LSTM 보다 낫다는 근거는?"** — 달라진다(표준편차 ≈ 예측 크기), 우열 근거 없음. DECISIONS 11-08 에 이미 "시드 하나·폴드 하나라 우열 근거 아님" 으로 적혀 있어 답은 가능하나 수치표가 없음. M3/S2.
4. **"alpha=100, min_child_samples=6 은 어떻게 정했나? 테스트 구간을 보고 정한 것 아닌가?"** — 문서상 개발 구간 결과를 본 뒤 정했다(M4). "봉인 구간은 보지 않았다" 로 방어 가능하나 개발 구간 표는 낙관 편향을 인정해야 함.
5. **"이 결과를 낸 코드 버전을 특정할 수 있나?"** — 현재는 못 함(전부 nogit, M5). engine_rev 날짜와 설정 JSON 은 있음.
6. **"ARIMA 의 모수가 최대우도 해인가?"** — 아님이 이미 확인됨(DECISIONS 11-09, +0.722·예측 부호 반대). 영상에는 명시했으나 결과 파일·논문 표에는 반영 경로 없음. 미수렴 2폴드도 미기록(m2).
