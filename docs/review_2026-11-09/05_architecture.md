# 점검 보고 — 아키텍처·재현성·소프트웨어 품질

대상: /home/claude/tsp (읽기 전용). 작업 사본에서 테스트 전체 실행(184개 통과, 4분 30초)과 저장 결과 4건 재실행·비교를 직접 했다.
환경: Python 3.13.18 · numpy 2.5.3 · pandas 2.3.3 · scipy 1.18.1 · sklearn 1.9.1 · statsmodels 0.15.0 · lightgbm 4.7.0 · torch 2.15.0+cu132(CPU 실행).

## 1. 총평

엔진 한 경로(`prepare → make_folds → _walkforward_predict → backtest/forecast_metrics`)로 모든 모델이 지나가는 구조, 결과 json 에 설정·해시·데이터 버전·lag·변환을 남기는 습관, 봉인 평가의 사전 기록·재개봉 경고는 잘 짜여 있다. 결정적 모델(Ridge·규칙·ARIMA)은 저장된 설정만으로 **예측·지표가 비트 단위로 재현**됐다(3절). 그러나 심사 방어 관점에서 세 가지가 약하다. ① **코드 버전이 전부 `nogit`** — 50개 결과 파일 어느 것도 "어떤 코드로 냈는지"를 증명할 수 없고, 11-05/11-06/11-07 엔진 수정 전후 결과가 `engine_rev`(11-07 도입) 하나로만 구분된다. ② **`config_hash` 가 스키마 버전에 묶여 있어** 결과 50개 중 25개가 지금 코드로 다시 계산하면 다른 해시가 된다 — "설정 해시 → 결과 파일" 대응과 봉인 '같은 설정' 판정의 토대가 흔들린다. ③ **시도 횟수(N)·Holm 가족 크기(m)·V 의 정의가 "파일 상태"에 의존**한다 — 카운터는 손으로 고쳐졌고(DECISIONS 11-07), 중복 재실행·연습 실행(`log_smoke`)·규칙 기준선까지 세며, Holm 의 m 은 디스크에 남은 파일 수(같은 모델의 재실행본 3개가 가설 3개로 계산)다. 그 밖에 CLAUDE.md 가 말하는 RQ+Redis 는 없고(파일 큐·스레드 1개), 동시 실행 금지는 API 안에서만 지켜지며 CLI 경로는 막지 않는다(11-07 로그에 실제 동시 실행 흔적).

## 2. 발견 목록

### 중대

**A1. 코드 버전 기록 없음(`nogit`) — 절대 규칙 7 미충족, 결과→코드 대응 불가**
- 근거: 결과 50개 전부 `code_version: "nogit"`. `engine/experiment.py:289-304` 는 git 이 있을 때만 해시를 적고, 저장소에 `.git` 없음(`git_first_push.bat` 미실행). 엔진 계산이 바뀐 날(11-05 bfill→ffill, 11-06 ARIMA 제약·봉인 폴드, 11-07 창 눈금·중복 ID)마다 결과가 달라지는데, 11-07 이전 결과에는 `engine_rev` 도 없다(`rev=None` 25개).
- 영향: "이 표의 수치는 어떤 코드로 냈나"에 DECISIONS 산문 외 답이 없다. 유일한 코드 보관본은 `_snapshots/…20261107_01.zip`(git 제외, 노트북에만).
- 방향: 첫 커밋 후 논문 표에 들어갈 결과는 전부 커밋된 코드로 재실행. `engine_rev` 와 별도로 결과 스키마 버전을 기록. 확실.

**A2. `config_hash` 가 스키마 진화에 따라 바뀜 — 결과 파일명·봉인 '같은 설정' 판정 기준 불안정**
- 근거: `experiment.py:94-95` 는 `model_dump_json(exclude={"name"})` 전체(기본값 채운 모든 필드)를 해시. 저장된 50개 결과의 `config` 를 지금 코드로 다시 해시하면 **25개가 저장 해시와 다름**(예: `f11_ridge_macro_k1_v1105_088044cf2a3f` → 지금 `cd1cc2ab1a10`; 원인은 뒤에 추가된 `forecast_info_time`·`mode`·`variable_transforms`·`lag_override`·`validation.test_start/test_end`). 실제 재실행하니 다른 이름의 파일이 생겼다.
- 영향: 같은 실험인데 파일명이 둘(이름 같으면 덮어쓰기, 다르면 중복). `engine/sealed.py:60-63 history()` 는 해시 일치로 "이미 연 설정"을 찾으므로 스키마가 한 번 더 바뀌면 재개봉 경고가 빠질 수 있다(봉인 0회라 지금 피해 없음). 반대로 `lstm_ccfc51bdb577` 과 `lstm_v1107_ccfc51bdb577` 처럼 같은 해시가 두 파일로 남아 있다.
- 방향: 해시는 "의미 있는 필드만·기본값 정규화" 또는 해시 버전 접두어; 봉인 기록은 설정 본문을 보존해 대조. 확실.

**A3. 시도 횟수 N(실험 카운터)의 정의·관리가 느슨 — Deflated Sharpe·사전 등록 방어 약화**
- 근거: `cli.py:57-58` 시작 때 읽고 `:92-94` 끝에 `max(cur+1, n)` 로 씀(성공 실행만). 세는 것: 규칙 기준선(B&H 93, MA 94, 모멘텀 95), 같은 설정 재실행(`_v1107` 13건), 연습 실행(`logs/tsp_20261107.log`: `log_smoke_…` 카운터 14, 파일은 삭제됨), ablation 변형(`ablate.py:47-53`, 여기선 `_bump(n)` 가 덮어쓰기라 cli 의 경합 수정 미적용). 안 세는 것: 실패(11-07 00:44~45 13건), 봉인, lag_shift, explain. 카운터는 평문 JSON 이고 **손으로 20 으로 고친 기록**(DECISIONS 11-07 359행). 저장된 `n_trials_so_far` 는 시작 시점 값이라 파일 카운터와 어긋남(14,14 / 17,17 중복). 루트 `results/experiment_counter.json`(81) 이 남아 있어 `results_dir` 이주 로직(`experiment.py:940-954`)이 trading 카운터를 잃으면 81 로 되살릴 수 있다.
- 영향: DSR 의 N=98 을 "독립 전략 시도 수"로 설명하기 어렵고(B&H 도 시도?), 삭제된 실험 목록은 로그 이틀치(logs/ 는 git 제외) 외에 복원 불가.
- 방향: append-only 시도 원장(설정 해시·시각·결과 파일·종류 dev/sealed/ablation/smoke) 을 두고 N 을 정의대로 계산. 카운터 파일 폐기. 확실(정의 문제).

**A4. Holm 보정의 가족 크기 m = "디스크에 남은 결과 파일 수"**
- 근거: `api/main.py:430-447 _holm_by_family` 는 `/api/results` 가 읽은 파일을 (대상·k·종류·info_time·평가구간) 로 묶는다. 지금 F11 k=1 가족은 **m=14**: ridge_macro 3개, lstm 3, biva 3, arima 2, sarima 2, ridge_tech 1 — 같은 모델의 재실행본이 가설로 중복 계산되고, `DELETE /api/results/{stem}`(`:544-554`) 로 지우면 m 이 줄어 p_holm 이 작아진다.
- 영향: 보정 p 가 파일 정리 상태에 따라 변한다. 논문 표에 그대로 옮기면 안 된다.
- 방향: 가족을 사전 등록 모델 목록으로 고정하거나, 같은 (모델·설정 의미) 는 최신본 1개만. 확실.

**A5. 동시 실행 금지가 CLI 경로에서는 강제되지 않음 — 실제로 겹쳐 돌았음**
- 근거: `api/jobs.py:46,95-104` 는 서버 안 스레드 1개로 직렬화하지만 `cli.py`·`run_all.sh`·`tools/rerun_stale.py:161`·`tools/resumable_run.py` 는 잠금 없이 실행(잠금·pid 파일 전무). `logs/tsp_20261107.log`: `f11_biva_macro_k1_v1107` 00:10:32 시작 → 01:26:23 저장, 그 사이 매매 06~12·01~05 재실행과 `repro_biva_orig`(01:12~01:26) 가 같이 돌았다. 예측 카운터 14,14 / 17,17 중복이 그 흔적(DECISIONS 11-07 "매매 모드는 동시 실행 없음" 은 모드별로는 맞음).
- 영향: 카운터 경합 외에 CPU 경합이 torch 비결정성·`seconds` 에 영향. "절차를 지켰다"는 진술 근거가 약하다.
- 방향: `results/<mode>/.lock` 파일 잠금을 `run_experiment` 진입에서 잡고, 잡혀 있으면 거부. 확실.

**A6. 재현에 필요한 환경 정보가 결과에 없음**
- 근거: 결과 키는 `config, config_hash, data_version, code_version, engine_rev, device, seconds` 뿐(`experiment.py:789-815, 905-916`). Python·numpy·torch·statsmodels·lightgbm 버전, OS, 스레드 수, CUDA/cuDNN 설정 없음. `requirements.txt`·`pyproject.toml` 은 pandas 만 상한, 나머지 무버전, lock 없음. 노트북(Python 3.14, torch 2.11+cu128)과 클라우드(3.13, torch 2.15) 결과가 섞여 있고 `device` 로만 구분.
- 영향: 창 모델 결과의 재현 범위를 말할 수 없다. 결정적 모델도 기계마다 1e-17 차이(3절).
- 방향: `prepare()` 에서 platform·sys.version·패키지 버전·스레드 수·cuDNN 플래그를 `env` 키로 기록, `pip freeze` 보관. 확실.

**A7. 창 모델 비결정성의 구조적 원인 — 시드만으로는 장치 간 동일 불가**
- 근거: `experiment.py:278-286 _seed` 는 `random/np/torch.manual_seed` 만; `use_deterministic_algorithms`·`cudnn.deterministic` 없음. `models/lstm.py:72 torch.randperm(n, device=dev)` 는 CUDA 생성기를 써 CPU/GPU 셔플 순서가 애초에 다르다. 11-07 발견(폴드 하나만 재실행 시 불일치)과 같은 뿌리.
- 방향: 폴드마다 `seed+fold_no` 재고정(사용자 결정 대기), 셔플은 CPU `Generator`, deterministic 강제 + 다중 시드 보고. 확실(원인) / 효과 크기 미확인.

### 경미

**B1. 의존 방향(규칙 5) 위반 1건**: `models/rsi_rule.py:4 from engine.features import rsi`. api→models 직접 import 없음. `models/base.py:14` "장치는 엔진이 device 인자로 전달"은 CLAUDE.md 규칙 3 문구와 충돌 — 구현이 맞고 규칙 문구를 고칠 것.

**B2. 규칙 1("날짜·시프트·fill 은 calendar.py 한 곳") 미준수**: calendar.py 밖 shift/pct_change/diff/ffill 39곳(`experiment.py:168,464,510-522,897`, `backtest.py:111-113`, `api/main.py:482-534`, `models/momentum.py:14`, `features.py:14`). 누출은 아니나 "유일한 지점" 주장은 사실이 아님.

**B3. CLAUDE.md 기술 스택 불일치**: "RQ+Redis · PostgreSQL · Docker" 라 적혀 있으나 실제는 메모리 큐 + `results/jobs/*.json`(`api/jobs.py:1-5`), DB 없음(DECISIONS 98행에 미룸 결정 있음 → CLAUDE.md 만 낡음).

**B4. 설정 스키마 느슨 — 오타 무시**: pydantic `extra=ignore`. 직접 확인: `validation={"sealed_form":…}` → `sealed_from=None`(봉인 없이 전 기간 평가), `seq_length`·`frequncy`·`trade={"fee":…}` 무시, `train_len=-5`·`test_len=0`·`val_frac=3.0`·`fee_rate=-1`·날짜 `"2025-13-45"`·없는 모델 이름 모두 통과. 현재 설정 45개엔 유해 키 없음(`_comment` 1건). 방향: `extra="forbid"` + 범위·날짜 검증.

**B5. 결과 덮어쓰기**: `experiment.py:957-970 save_result` 는 `{name}_{hash}.json` 을 경고 없이 덮어씀(봉인만 번호). 같은 설정 재실행은 카운터 +1 이면서 이전 결과를 지움 → N 은 늘고 V 는 하나만 봄.

**B6. 조용히 넘어가는 지점**: `experiment.py:367-372 _eval_window` ValueError → 규칙 모델을 **전 기간** 평가하며 표시 없음(비교 무효 가능); `:429-430` 폴드 건너뜀을 기록 안 함(저장 결과 5건은 건너뜀 0 확인); `:446-450` VECM 결측 `continue`; `:285` torch 시드 실패 무시; `:935` V 계산 시 못 읽는 결과 무시; `models/vecm.py:115`; `api/jobs.py:135`. `engine/store.py:30 DataVersion.__init__` 의 `mkdir` — 잘못된 ID/루트면 빈 버전 폴더 생성 후 뒤에서 "변수 없음"으로 실패; 실제 흔적 `data/versions/versions/20261103T141116Z/`(빈 폴더). `tools/rerun_stale.py:145` 동일 부작용.

**B7. API 버그·운영**: `api/main.py:619 FileResponse` 미import → `/api/guide/walk/audio/...` 항상 500(TestClient·ruff F821 확인). CORS `*` + 무인증 `DELETE /api/results`. `jobs.cancel` 은 Windows 에서 returncode 1 → "failed" 기록(`jobs.py:128`). `report.py:9-13` 은 `sealed_log.json` 을 안 거름 → 봉인 1회 뒤 KeyError. `results_dir` 이주 메시지는 옮긴 게 없어도 출력.

**B8. 테스트 공백**: `tests/test_engine.py:97-102` DM 은 부호·p<0.01 만, `:288-298` DSR 은 단조성만 — **알려진 값 대조 없음**; `bootstrap_excess_pvalue` 테스트 0건; 비용 모델은 수수료만(슬리피지 0, `:74-86`); GRU 는 심은 신호·랜덤워크 테스트 없음(`:171,181` lstm·biva 만 — 규칙 6 위반); 봉인 경계 테스트 `:644` 는 "빠짐·겹침 없음"만. 느린 테스트 2개(BIVA 94초·78초)가 전체의 60%. 비결정 테스트 전부 시드 고정, flaky 징후 없음.

**B9. 결과 스키마 버전 없음**: 11-05 결과엔 `periods` 키 없음(KeyError 확인), 11-07 전엔 `engine_rev` 없음. 매매 `data_version=dv.id`(`:793`) vs 예측 `cfg.data_version`(`:908`) — CLI `--data-version` 으로 주면 예측 결과엔 빈 문자열(현재 결과는 전부 설정에 버전 있어 피해 없음).

**B10. 인수인계 문서의 낡은 수치**: `docs/STATUS.md:8` "테스트 30개"(실제 184), `:9` "모델 12개"(VECM 빠짐, 13개). `.gitignore` 가 `logs/`·`results/jobs/`·`_snapshots/` 제외 → 실험 로그·작업 기록이 저장소에 남지 않음(감사 추적은 노트북에만).

**B11. 코드 품질 핫스팟**: `experiment.py` 970줄 — `_run_experiment`/`_finish_forecast` 가 결과 dict 를 각각 조립(`:789-815`/`:905-916`, data_version 차이의 원인), 신호 문턱·퇴화 검사 2벌(`:700-703`/`:712-715`, `:786`/`:902`), `_walkforward_predict` 170줄에 4갈래. `explain.py` 1105줄은 capture 를 읽는 구조라 위험 낮음. `configs/` 45개 중 결과 남은 것 23개(h1~h5·r1·13·14 삭제됨). ruff 125건(대부분 스타일).

### 건의
**C1.** 논문 표 결과마다 (설정·manifest 해시·커밋·`pip freeze`·명령·장치·결과) 를 묶는 `tools/export_repro.py` — 반나절.
**C2.** DSR·Holm 의 N·m 을 원장 기반으로 재계산, 사전 등록 가설 목록과 대조 — 반나절 + 통계 검토.
**C3.** 창 모델 시드 3~5개 반복 평균·범위 보고(폴드별 시드 결정 후) — GPU 수 시간.

## 3. 확인해서 문제없던 것
- **결정적 모델 재현(비트 단위)**: `04_ridge_A_v1107`(일간 Ridge 149폴드), `02_ma_cross_v1107`, `f11_arima_k1_v1107`(월간 ARIMA 120폴드) — 예측·수익·RMSE·DM 전부 동일. 다른 것은 `deflated_sharpe_p`·`n_trials_so_far`·`trial_sr_var`(파일 상태 의존) 뿐이고 셋 다 결과에 저장돼 p 재계산 가능. `f11_ridge_macro_k1_v1105`(노트북 CUDA·11-07 전 엔진): 예측 최대 차 1.4e-17, 지표 상대차 1e-15.
- 데이터 버전 3개 `manifest.json` sha256 전부 일치(20261102 버전의 `prices_*_merged.parquet` 2개만 manifest 밖).
- 변수 parquet 간 (variable_id, date) 중복 0건 → `store.wide(aggfunc="last")` 잠재 비결정성 미발현.
- api 가 models 직접 import 없음; engine→models 는 `importlib`.
- 봉인 평가: `_sealed` 분리 저장, 카운터 미증가, 재개봉 경고, nogit 경고(`sealed.py:109-112`) 구현됨. 봉인 0회.
- `tools/resumable_run.py`: 폴드 단위 저장·난수 복원, 지문에 장치 포함 — 설계 타당, 테스트 2건 통과.
- 저장 결과 5건에서 생성 폴드 수 = 실행 폴드 수.

## 4. 확인 못 한 것
- 노트북(Windows·GPU) 환경 재현·`run_server.bat`·`jobs.cancel` 실동작.
- 창 모델(LSTM·GRU·BIVA) 재실행 비교(10분 초과).
- `web/` 는 훑어보기만. `explain.py` 세부, `registry.py`·`collect.py`·어댑터(다른 영역).
- `_snapshots/…zip` MANIFEST(클라우드 사본에 없음).

## 5. 교수가 물을 법한 질문과 지금 답할 수 있는지
1. **"이 수치는 어떤 코드·데이터·환경에서 나왔고 다시 돌리면 같나?"** — 데이터: 답 가능. 코드: **불가**(`nogit`). 환경: 불가(미기록). 재실행: 결정적 모델은 같음, 창 모델은 같은 장치·시드·전체 폴드 조건에서만.
2. **"DSR 의 N=98 은 무엇을 센 것인가? B&H 도? 삭제한 실험은?"** — 코드 동작은 설명 가능하나 '독립 시도'로 정당화 어렵고 삭제 목록 복원 불가. **부분 답**.
3. **"Holm 가족은 어떻게 정했나?"** — "파일이 있는 결과 묶음"이라 재실행본이 중복 계산. **불가(정의 수정 필요)**.
4. **"봉인 구간을 한 번도 안 봤다는 증거는?"** — 기록 구조 있음·봉인 0회. 단 `sealed_log.json` 평문, `--sealed` CLI 는 미리보기 차단을 안 거침(`cli.py:38-90`), 옛 매매 결과 13개의 봉인 구간 포함 이력과 봉인 경계 문제를 함께 설명해야 함. **조건부 답**.
5. **"같은 해시인데 파일이 둘, 다른 해시인데 같은 실험 — 결과는 어떻게 식별하나?"** — A2. 지금은 파일명 관례(`_v1105/_v1107`)에 의존. **불가(해시 정의 고정 필요)**.
