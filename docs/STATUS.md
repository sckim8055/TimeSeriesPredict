# 진행 상태 (STATUS) — 세션 시작 시 읽고, 끝날 때 갱신

최종 갱신: 2026-11-20 (Claude 세션 — 다음 세션은 어떤 모델이든 이 절부터 읽을 것)

## 0. 다음 세션 인수인계 (반드시 먼저)
**작업 방식**: 코드는 `F:\works\TimeSeriesPredict` 가 원본. Claude 클라우드 세션은 `/home/claude/tsp` 에 사본을 두고 작업한 뒤 바뀐 파일만 노트북으로 커밋했다(커밋 도구는 같은 스테이징 경로를 재사용하면 옛 내용을 보내므로 매번 새 경로 c22, c23 … 를 쓰고 `device_list_dir` 로 크기 확인). 새 세션은 노트북 폴더를 다시 읽어 사본을 만들어야 한다. 노트북 셸(device_bash)은 프로젝트 폴더가 안 보이고 네트워크가 막혀 있으므로 실행은 클라우드 사본에서, 결과 파일은 커밋으로.
**실행 환경(노트북)**: Windows, Python `C:\Users\test\AppData\Local\Python\pythoncore-3.14-64`, `run_server.bat` → http://127.0.0.1:8000 (uvicorn + `web/dist` 빌드본 포함, Node 불필요). 실험 로그는 `results/jobs/<id>.log` 와 서버 콘솔 양쪽. API 키는 `.env`(ECOS/FRED/DATAGO/KRX) — 채팅에 노출된 적이 있어 공개 전 재발급 권고.
**품질 기준(사용자 요구)**: 박사논문·SCI 용. 데이터 정확도 > 기능. 긍정적 답보다 문제를 그대로. 결정은 `docs/DECISIONS.md` 에 날짜별 기록, 수치는 결과 json 에서만 인용, 가이드 표는 `tools/build_guide.py` 로 생성(손으로 쓰지 않음), 테스트 `python -m pytest tests -q`(30개) 통과 후 커밋.
**지금 상태 한 줄**(11-05 갱신: lag 49/65 확인·근거 `tools/lag_evidence.py`·보고서 `docs/guide/25_lag검증.md`): 두 연구 모드(매매/예측) 작동, 모델 12개(규칙 5, Ridge, LightGBM, LSTM, GRU, BIVA, ARIMA, SARIMA), 변수 65개(변환·lag 기본값 전부 채움, lag 는 전부 미확인 추정), 결론은 "통계적으로 유의한 매매 신호 없음 / 예측 연구 F11 은 모든 모델이 naive 에 미달".
**사용자가 다음에 물을 가능성이 큰 것**: ① lag 확인·lag-shift(아래), ② VECM·앙상블 추가, ③ KRX 수급 CSV 경로(사용자 보류), ④ H1 봉인 평가 여부(사용자 결정 사항, 먼저 묻지 말고 요청 시).

### 2026-11-20 결과 화면: 모든 폴드 기간·설정 전체와 뜻·로그 내려받기 (근거 DECISIONS 같은 날)
- 새 결과부터 periods.folds(모든 폴드). 결과 '설정' 칸에 설정 전체·모델 파라미터 표(뜻 포함)와 '로그 전체 내려받기 (zip)'.
- 2026-11-21 결과 꾸러미(`/api/results/{stem}/package.zip`, engine/report_pack.py): 사전·사중·사후 폴더 + Word 보고서. 노트북에 `pip install python-docx` 필요(없으면 HTML 보고서). 같은 날 '데이터 변화 단계'(변수별 단계표·폴드별 눈금·계산 따라가기) 추가 — engine/datatrace.py. 꾸러미는 작업이 끝난 직후 결과 옆에 저장(실행 당시 그대로), 내려받기는 저장본을 그대로. 시험용 httpx2 를 requirements 에.

### 2026-11-20 워크포워드 계획 표·도식 숫자 (근거 DECISIONS '2026-11-20')
- 워크포워드 = 밀려가는 창. 화면 '실제로 쓰이는 구간' 표가 두 모드 모두 첫·마지막 폴드의 자르는 순서와 개수를 보여 줌(실행 기록과 같음). 도식 '학습 N' 바로잡음.

### 2026-11-18 1회 분할 비율·분할 계획·고르는 법 (근거 DECISIONS '2026-11-18 — 1회 분할')
- 원본 BIVA = 60:10:30 한 번 학습 + 입력 창 슬라이딩 = 플랫폼 '1회 분할'. 화면에 비율로 날짜 채우기, '실제로 쓰이는 구간' 표(실행 폴드 기록과 같음), 파라미터 '고르는 법'.
- 남은 일: 늦게 시작하는 변수 때문에 창·BRITS 폴드가 건너뛰어지는 규칙(엔진, 사용자 결정), BIVA 1회 분할 실행(사용자), 새 파이썬 codewalk 실패 5개(pip 목록 비교).
- 파이썬: 노트북 서버가 이제 `...\Programs\Python\Python314`(3.14.10) — run_server.bat·narrate.bat 바꿈.

### 2026-11-17 로그 날짜별 파일·접속 기록 (근거 DECISIONS '2026-11-17 — 로그')
- 날짜가 바뀌면 새 파일(tsp_·access_), 쓰기 실패하면 다시 열기, 접속 기록(보낸 주소·노트북 직접/외부·주소·결과·브라우저). 화면 '개발 로그'에서 종류 선택.
- 11-16 16:18 이후 로그 파일 쓰기 실패(F: 연결 의심) — 사용자가 서버 재시작 후 확인.

### 2026-11-16 미국 마감 → 한국 시가 시차 검증 (근거 DECISIONS '2026-11-16 — 미국 마감 → 한국 시가 시차 검증')
- 미국 움직임은 대부분 시가 갭에서 끝남(SOX b +0.36, t 24). 시가 뒤에는 작은 되돌림(엔진 정답 b −0.09, t −3.9, 뒤 절반에서만 뚜렷). 따라가기는 못 씀. 도움말 50장 끝, docs/analysis/.
- 다음: 서로 다르게 움직이는 국내 ETF 몇 개로 넓히기(새 자료), 되돌림 규칙을 쓸지 사용자 결정.

### 2026-11-15 켈리 투자 비율·자신감 확인표·설명 정리 (근거 DECISIONS '2026-11-15 — 켈리 투자 비율 + 자신감 확인표')
- 실험 '투자 비율 정하기'(문턱/켈리, 절반 켈리 기본, 과거만으로 보정), 결과 '자신감 확인표'(예측 5묶음의 실제 결과). 리허설: 켈리는 문턱보다 나빴음(04_ridge_A, h1).
- 오늘·어제 추가분(섀넌·HMM·켈리·확인표·다른 출처 대조·기술지표 출처)을 수식 읽는 법 29~34절·용어집·화면 안내·엔진 알고리즘·결과 읽기에 넣음. 영상·음성은 보류.
- 다음: 미국 마감 → 한국 시가 시차 검증 → ETF 몇 개로 넓히기(사용자와 합의한 순서), 로드맵 Doc 솔루션 중심으로.

### 2026-11-15 HMM 시장 상태 모델 (근거 DECISIONS '2026-11-15 — HMM 시장 상태 모델 추가')
- 모델 목록 '통계(단변량)'에 HMM 시장 상태(2·3상태, Baum–Welch, 앞방향 필터로만 예측). 매매·예측 연구 둘 다. ENGINE_REV 그대로.
- 실데이터 리허설: 단독으로는 효과 없음(091160 샤프 −0.03, F11 은 지난 값보다 유의하게 나쁨). 상태 확률을 입력으로 쓰는 방식은 안 해 봄.
- 다음: 켈리 → 지표 세트.

### 2026-11-15 섀넌 재조정 기준선 (근거 DECISIONS '2026-11-15 — 섀넌 재조정 기준선')
- 모델 목록 '규칙 (기준선)'에 섀넌 재조정(기본 ETF 50%·달마다). 모든 매매 결과에 '섀넌 50:50' 비교 기준(같은 구간·같은 비용). 백테스트에 흘러가는 비율 회계(새 모델에만). ENGINE_REV 그대로(기존 숫자 불변).
- 다음: HMM 시장 상태 → 켈리 → 지표 세트.

### 2026-11-14 다른 출처와 대조 (근거 DECISIONS '2026-11-14 — 다른 출처와 대조')
- duplicate_of 7짝을 기간별로 대조, 표시만(값 그대로). 노랑 3(L1·US10Y_YF·VIX_YF), 초록 2, 회색 2(O3·O4 값 없음). 엑셀 N12 국고채는 월평균 자료로 확인. 도움말 20장 2-7.
- 남은 일: USDKRW 2008 빈 기간은 이 짝으로 확인 불가(다른 출처 필요). 실험 결과 json 에 대조 줄 넣기는 보류.

### 2026-11-14 기술지표 출처 기록 (근거 DECISIONS '2026-11-14 — 기술지표 18개·창 길이의 출처 기록')
- 18개와 창 길이는 Claude 가 관례(RSI 14 Wilder, MACD 12·26·9 Appel, 이동평균 5·20·60·120 HTS 기본)와 흔한 길이로 정함 — 데이터로 고르지 않음. 주간·월간 창은 Claude 가 정한 값이고 '같은 달력 길이 환산'이 아님(설명 정정, 값 그대로). 변수 사전 기술지표 판·도움말 20장 5-9절.
- 저장본 `_snapshots/TSP_저장본_20261114_01.zip`(c101 과 같은 코드). 합의한 다음 순서: 자료 오류 판별 → 섀넌 재조정 비교 기준 → HMM 시장 상태 → 켈리 → 지표 세트 직접 만들기.

### 2026-11-14 자료 점검표 (근거는 DECISIONS '2026-11-14 — 자료 점검표')
- 변수마다 정상성(ADF+KPSS)·계절성(월·분기 F 검정)·계절조정 표시·빈 기간·확인해 볼 날 → 신호(초록·노랑·회색). 값·설정은 바꾸지 않음. 변수 사전 '자료 점검표' 판, [값 보기], 실험 결과 `data_checks`(봉인 시작일 앞 값만).
- 변수 사전 새 선택 칸 `seasonal_adj`(SA/NSA). 시험 295개 통과. ENGINE_REV 그대로.
- 다음(사용자와 정한 순서): 자료 오류 판별(다른 출처 대조·신호등) → 보조 실험(변동성 조정·극단값 잘라 내기).

### 2026-11-13 독립 검토 반영 (근거는 DECISIONS '2026-11-13 — 독립 검토(2026-11-13) 반영')
- 높음 2: 변수 추가의 주기를 자료로 정함(월 자료를 '일'로 등록해 발표시차 1일 → 누출 막음) · 거의 같은 종목(091230, ρ≈0.99)으로 본 봉인 구간도 경고·봉인 결과에 기록.
- 중간 4: 계산 따라가기 변동성 거절 · restat 변동성 줄 · collect 종료 코드 3(시작 전 멈춤) · 누출 교란 시험(변이 M1~M4 로 시험이 잡는 것 확인).
- 낮음·의심: 되돌림 규칙 정의대로 · pct0 거절 · realign 함수 단위 · BIVA 변경 덩어리 잠금(PATCHES_LOCK) · select_top/pool 중복 거절 · **CSRF(다른 사이트가 노트북 서버 조종) 막음 — CORS 없앰·Origin 검사** · 네이버 소급 배수 맞춤 · 선별 순위를 검증 행 앞에서만 · ETF 판별 120일.
- ENGINE_REV 그대로(예측·성과 계산은 안 바뀜. LightGBM·창 모델의 진단표 feature_relevance 숫자만 달라짐 — DECISIONS 에 기록).
- 노트북에서 사용자 확인: 외부 주소(Funnel)에서 [데이터 가져오기] → 암호 창 → 진행되는지.

### 2026-11-08 세션에서 바뀐 것 (근거는 DECISIONS 2026-11-08)
- **BIVA 코드 따라가기 음성 대본 완성**: 63단계 say·줄 설명·직접 실행해 보기 13개, 검증 48/48, `check_say` 문제 0. 독립 검토 2회(6명)로 사실 오류 11건·과장 다수를 고침(목록은 DECISIONS).
  **녹음 완료(11-08)**: 음성 KO_Sian `5n5gqmaQi9Ewevrz7bOS` — BIVA 63단계 35분 21초, Ridge 16단계 8분 8초(Daniel 녹음은 교체). 검사 통과 → biva.mp4·ridge.mp4 다시 만듦(`render_mp4 --work logs/render/<화면>`). 녹음 뒤 클라우드에서 지문·글자 시각 확인 → biva.mp4·ridge.mp4 다시 만듦.
- **읽기 규칙**: 유월·시월, 칸·줄·자리 고유어, 화면 글 조사를 소리에 맞춤(`speech.josa_display`). 실습 출력 5자리(엔진 32비트와 글자까지 같게).
- **나머지 7개 화면 대본 완성(11-08~09)**: LSTM·GRU·ARIMA·SARIMA·LightGBM·VECM·규칙 — 작성 → 독립 검토 3회(1차 전체, 2차 새 검토자 전체, 3차 바뀐 부분) → 수정. check_say 9개 모두 0, 시험 184개.
  **녹음 대기**: 노트북 `narrate.bat` → `ridge lstm gru arima sarima lgbm vecm rule` (41,258자 — Ridge 는 [4]·[15] 두 단계만). 녹음 뒤 클라우드에서 검사 → 영상 8개(ridge 포함) 렌더.
  새 규칙: 검증 줄 종류(독립 대조·식 확인·재실행·추가 탐색·참고), 봉인 경계 공개, 빌드 환경과 무관한 음성(출처 없으면 빌드 멈춤), `common.RESULT_FILES`(결과 파일 목록 고정), `series_script.PINNED`(탐색 숫자 고정).
- **할 일(11-09 결정)**: FRED 변수만 ALFRED 실제 공개일·첫 발표치로 돌린 민감도 결과(본 분석은 고정 lag 유지) — DECISIONS 11-09 '발표시차 처리 방침'.
- **엑셀 입력 단일화(11-09)**: 직접 넣는 데이터도 `data/variable_master.xlsx` 의 데이터 시트(df_M·df_Q…)에. `data/uploads/` 는 읽지 않음(원본은 `legacy/uploads_20261109/`). API 원본은 `data/raw/*.json` + `fetch_log.jsonl`. 근거 DECISIONS 11-09.
- **11-12 추가(근거 DECISIONS 2026-11-12, 노트북 반영 c77·c78)**: 변동성 타겟(`target.kind=volatility`, 통과 기준 미정 → 봉인 평가 막음), 폴드 안 입력 연관성 순위(`feature_relevance`)·선별(`select_top`), 변수 사전 화면의 '변수 추가·데이터 가져오기'(야후·FRED·ECOS, 키는 .env 만), 접속 키 가리기(`scrub_secrets`). 시험 247개. 엔진 판은 그대로(기존 숫자 불변). 저장본 `_snapshots/TSP_저장본_20261112_01.zip`. **노트북에서 확인할 것**: '받아 보기'·'데이터 가져오기'가 실제 출처에서 받아 오는지(클라우드는 연결 불가). 계획서에 정할 것: 변동성의 통과 기준·k, 선별 개수, 주기에 맞춘 입력 묶음.
- **11-13 추가**: 변수 사전 [값 보기](그래프·점검·CSV)·발표시차·변환 고치기, 종목 추가 한 번에, 화면을 열 때 암호 창 없앰, 네이버 이동 창 처리. 점검에서 **야후 원/달러 2008-08~09 자료 오류 의심**(26일 구멍·하루씩 번갈아 튐) — 확증 전에 다른 출처와 대조할 것. 시험 262개.
- **우선순위(11-12 사용자 지시)**: 정확성·재현성·타당성·정합성 검증이 먼저. 음성·동영상·ElevenLabs 는 보류(계산이 확정된 뒤 한 번에).
- **11-12 추가 3**: 외부 주소 = Tailscale Funnel 기본 주소(가비아 도메인 연결은 불가). 바꾸는 동작에 암호(`TSP_WRITE_PASSWORD`, 보기는 열림). 노트북 `.env` 는 포맷 때 없어져 사용자가 다시 넣어야 함(`.env.example` 참고). 시험 255개.
- **11-12 추가 2(근거 DECISIONS '2026-11-12 (계속)', 노트북 c79)**: 여러 종목 묶어 학습(`pool_tickers`, 학습에만 보탬·평가는 대상만, 릿지·LightGBM), 종목 목록 파일 `data/tickers.json`, 화면에서 종목 추가. 리허설: 091230(상관 0.973)을 묶어도 효과 없음 — 다른 종목을 넣을지는 계획서에서 정할 것. 시험 253개.
- **지금 상태(11-10 심야, 다음 세션은 여기부터)** — 근거 DECISIONS '2026-11-10 심야':
  · **엔진 판 2026-11-10**(봉인 경계·결정일 사이 변화·폴드 시드·`n_seeds`·LightGBM 시드·SARIMA 기본·여러 출발점·k>1 분산·학습 창 평균 기준). **그 전 결과는 전부 '옛 방식' — 논문용 결과는 전부 새로 돌려야 한다.** 시험 239개 통과. 실데이터 실행은 사본에서 리허설만(노트북 결과는 그대로).
  · 독립 검토 22건(F1~F22) 반영: 수집 실패 시 직전 값 유지·'일부 받지 못함' 버전은 자동 최신에서 제외, Clark–West 는 **학습 창 평균 기준**(`cw_vs_mean`)이 '예측 정보' 검정, Holm 가족 정의·한쪽 p(`engine/multitest.py`), **봉인 정책 `seal_policy.json`**(091160: 2025-01-01, F11: 2024-01-01 — 빈 설정에는 채움, 넘기려면 `--past-seal "이유"`), 원장 머리 고정·잠금, 엑셀 엄격 읽기, 저장 되돌림.
  · 새 결과 칸: `model_diag`(시드·쓰인 트리/반복 수·상수 예측 폴드·여러 출발점), `pred_sd_ratio`, `cw_vs_mean`·`dm_vs_mean`·`r2_oos_vs_mean`, `seal_boundary`, `seal_policy`, `transform_basis`, `run_record.ledger_before`, 일별 `bench_mean`.
  · **다음 순서**: ① 사용자: `docs/PREREG_초안_2026-11.md` 의 ☐ 항목 결정(시드 수, 주간·월간 포함 여부, LightGBM 규제, 봉인 후보, F11 포함 등) ② 사용자: git 첫 커밋·GitHub(Private) — 지금 `code_version` 이 `nogit` ③ 데이터 새로 수집(키: ECOS·FRED·DATAGO) → 그 버전을 계획서에 적음 ④ 계획서 확정·커밋 ⑤ 확증 실험을 새 결과 폴더에서 실행(GPU 는 노트북) ⑥ `tools/restat.py` 로 표.
  · **미작업·주의**: 코드 따라가기 화면의 숫자는 옛 기록 — 재실행 뒤 다시 기록(ARIMA·SARIMA 화면은 여러 출발점 서사로 손볼 것). `tools/rerun_stale.py --run` 은 옛 기본값으로 고정하는 도구라 `--old-defaults` 없이는 돌지 않게 막음(논문용 재실행에 쓰지 말 것). 화면에 `n_seeds` 칸 없음. 분배금·ALFRED·거시 갱신·검출력 확대 미착수. GPU 실행 시간(`n_seeds` 5)은 재지 않음.
- **(앞 단계) 11-10 밤**: 재학습 없는 정확성·재현성 수정 1차 끝(시험 211 통과, 독립 검토 미완). F 드라이브 포맷 → 클라우드 사본으로 복구 끝(DECISIONS 11-10 밤). **다음 순서**: ① 오늘 수정분 독립 검토 다시 ② 사용자: git 저장소·.env 저장·Python 환경 확인 ③ 사전 분석 계획서 초안 → 사용자 확정 ④ 재학습 필요한 엔진·모델 수정(한 번에 모아서) ⑤ 동결 뒤 일괄 재실행(노트북 GPU) ⑥ 검출력 절·비교표. 노트북 커밋은 device_commit_files(셸 mount 안 됨), 다음 번호 c76.
- **논문 틀 확정(11-10 사용자)**: 기여 = 누출·다중검정·검정력을 통제하는 재현 가능한 평가 절차와 구현. SaaS 는 논문 뒤로. 순서 = 정확성 → 재현성 → SCIE 방어(REVIEW_2026-11-10 7절의 1~4·6). 화면 편의·SaaS 는 보류.
- **전면 점검 2차(11-10)**: `docs/REVIEW_2026-11-10.md` — 11-09 지적 상태표, 검출력 측정(54% 미만 신호는 지금 기준으로 못 잡음), 새 코드 결함 16건(N1~N16), 서비스 결함 15건(S1~S15), SaaS 격차, 권하는 순서 7단계. 코드는 고치지 않음. **다음 세션은 7절 순서와 사용자 결정(논문 틀: SaaS 를 뺄지 만들지)을 먼저 확인.**
- **전면 점검 보고(11-09 밤)**: `docs/REVIEW_2026-11-09.md` — 치명 5·중대 30여 건·우선순위 12단계. 다음 세션은 이 보고서 4절(건의)과 아래 결정 대기를 먼저 읽을 것. 코드는 아직 고치지 않음.
- **사용자 결정 대기(11-09 검토에서 드러남, 근거 DECISIONS 11-08~09)**: ① 봉인 경계 — 엔진이 결정일만 sealed_from 전으로 잘라 개발 마지막 결정의 정답·수익이 봉인 값을 씀(v1107 매매 13개 중 10개가 성과에 포함: 월간 LightGBM +10.01%, 총수익 +1.20% → 빼면 −8.01%) — 엔진 수정·재실행 여부 ② LightGBM random_state 0 고정(엔진 시드 2025 가 안 전해짐) ③ 창 모델 시드·장치 민감도(LSTM GPU 결과는 이 달 예측 부호까지 다름) ④ ARIMA(2,2) 다중 극대·경계(마지막 폴드도 더 높은 점, 예측 부호 반대) ⑤ SARIMA 계절 MA 경계(D=0 비교 안 함) ⑥ VECM 결정항 없음·r=1 고정 ⑦ DSR 의 N·V 표본 불일치, 부트스트랩 블록 5일 vs 설명 20일 ⑧ 변수 사전: K2 영문 설명(실업자 수인데 unemployment rate), US10Y(FRED 인데 lag 0·확인됨 — note 는 FRED 면 1), VIX(FRED), USDKRW(야후, 휴장일 값, lag 미확인), registry F11 lag 30 vs 마스터 32 ⑨ 원자료 가격 빈칸 2007-02-07→2009-04-17(실제 시작 2009-04-17) ⑩ 날짜 읽기 '십이 월' → '십이월'로 붙일지(BIVA 4·Ridge 7단계 약 2,400자 재녹음) ⑪ '+' 를 '플러스'로 읽을지(BIVA·Ridge 재녹음 필요).

### 2026-11-07 세션에서 바뀐 것 (근거는 DECISIONS 2026-11-07 항목들)
- **코드 따라가기 음성·자막·코드 설명·직접 실행해 보기**(Ridge 시범, DECISIONS 11-07): `tools/codewalk/speech.py`·`tools/narrate.py`·`narrate.bat`. 녹음은 노트북에서 ElevenLabs(`.env` 에 키·음성 ID) — 클라우드는 연결 불가. **Ridge 는 Daniel 음성으로 녹음·영상 완료(11-08, 9분 25초).**
- **이어 돌리기**: `tools/resumable_run.py` — 긴 실행을 폴드마다 저장, 끊기면 같은 명령으로 이어서(결과 동일, `tests/test_resumable.py`). 10_biva_ABC 20 epoch 판 실행에 사용.
- **저장본**: `_snapshots/TSP_저장본_20261107_01.zip`(.env 제외, 690파일, sha256 은 같은 폴더 MANIFEST). `_snapshots/` 는 git 제외.
- **엔진 버그 2개 수정**: ① 창 모델(LSTM·GRU·BIVA) 눈금을 학습 폴드 전체 행으로(전에는 빈 칸 없는 행만 → 늦게 시작하는 변수 앞 행이 눈금 밖) ② 중복 변수 ID 로 적은 변환·lag 덮어쓰기를 대표 ID 로 옮김(전에는 버려짐 — repro_biva_orig 의 USDKRW·BOK_RATE·KR3Y 가 변환된 채 들어갔음). **창 모델 결과와 repro_biva_orig 결과는 재실행 대상.**
- **코드 따라가기(BIVA)**: `tools/codewalk/`(trace·biva_capture·biva_script·player.html·render_mp4). 원본 코드 실행 줄·텐서·손 계산 검증 33개로 62단계 화면 `docs/guide_build/walk/biva.html` + 영상 `biva.mp4`(약 25분, 소리 없음; HTML 은 브라우저 한국어 음성). 실험 화면 BIVA 흐름도·도움말 35장 17절에서 연결. 기록 `biva_capture_e500.pkl`(원본 설정 500 epoch, 마지막 폴드).
- **BIVA 실행에서 확인한 사실**: 고른 가중치에서 VAE μ·log σ² 가 ReLU 로 모든 창에서 0 → z = 잡음(죽은 ReLU), 검증 최저가 epoch 1. 원본 코드는 그대로, 30장 BIVA 절과 DECISIONS 에 기록.
- **코드 따라가기 전 모델 완료**(사용자 요청 "모델 하나씩"): LSTM·GRU·ARIMA·SARIMA·Ridge·LightGBM·VECM·규칙(백테스트 포함) — `tools/codewalk/{rnn,series,table,vecm,rule}_{capture,script}.py`,
  화면·영상 `docs/guide_build/walk/<lstm|gru|arima|sarima|ridge|lgbm|vecm|rule>.{html,mp4}`, 실험 화면 흐름도(`ModelFlow.tsx` WALKS)·도움말 30장 표·35장 표에서 연결.
  영상의 예측을 결과 파일과 대조(`common.match_result`) — ARIMA·SARIMA·Ridge·LightGBM·LSTM·규칙 일치. 시험 `tests/test_codewalk.py`(저장 기록 8개 + Ridge·VECM 실기록).
- **ARIMA·SARIMA 추정용 배율 `scale`(기본 100)**: 배율 1 은 L-BFGS 가 최댓값에 닿지 못함(수렴 105/120·37/120, 로그우도 120 폴드 모두 개선). LEGACY 1.0 으로 옛 결과 재현. 재실행 `_v1107`.
- **재실행 완료(_v1107, 최신 데이터 버전)**: f11 LSTM·ARIMA·SARIMA, repro_biva_orig(옛 설정), 매매 01~09·11·12, lstm. `rerun_stale` 보완(데이터 버전 ID 명시, `091160_merged`→`091160`).
  **옛 매매 결과 13개는 봉인 구간(2025~)까지 포함해 평가된 것**(DECISIONS 2026-11-07) — 논문 표에는 _v1107 만.
- **발견: 창 모델 폴드 재현** — 시드를 실험 시작에 한 번만 고정 → 폴드 하나만 다시 돌리면 결과 파일과 다름(앞 폴드를 돌린 뒤면 같음). 폴드마다 시드 재고정은 사용자 결정(아래).

### 2026-11-06 세션에서 바뀐 것 (자세한 근거는 DECISIONS 2026-11-06 항목들)
- **ARIMA·SARIMA 추정 제약**: `enforce_stationarity`·`enforce_invertibility` 파라미터(기본 예 = statsmodels·원 노트북 기본). 그 전 결과는 아니오로 낸 것 — 결과 파일에 키가 없으면 `pinned_model_params` 가 아니오로 채워 같은 실험을 재현(F11 ARIMA 비트 단위 재현 확인, 기본(예)이면 RMSE 0.003699). 계산 따라가기에 '근 크기' 단계.
- **봉인 평가 버튼**: 결과 상세 아래 '봉인 평가'(미리보기 → 사전 기록 → 확인 → 실행), 결과 목록 위 '봉인 기록'. `engine/sealed.py`, API `/api/sealed/*`. 봉인 결과는 `_sealed` 로 따로 저장(전에는 개발 결과를 덮어썼음 — 봉인 0회라 피해 없음). 1회 분할 후보의 봉인 규칙을 개발 규칙과 맞춤. 봉인 평가는 실험 카운터를 올리지 않음.
- **버그 수정**: Deflated Sharpe 의 V(시도 샤프 분산)가 2026-11-04 폴더 이주 뒤 늘 None(근사 사용)이었음 → 모드 폴더를 읽게 수정. 기존 매매 결과의 deflated p 는 근사값(trial_sr_var=null).
- **움직이는 그림 20개 추가**(`tools/formula_anims.py`): Ridge 가중합·alpha, LightGBM, 칼만, 근 크기, SARIMA, VECM, GRU, 경사하강, 실제 LSTM 학습 기록, BRITS, 이동평균, Conv1d, VAE, BIVA 전체, 규칙 4종, 결정일 흐름. 숫자는 실제 코드로 계산, `tests/test_formula_guide.py` 가 확인.
- 테스트 104개 통과(11-06).
- **사용자 결정 대기**: 봉인 시작일(사용자가 정함, 지금 기본 매매 2025-01-01·예측 2024-01-01), 기존 ARIMA/SARIMA·매매 결과 재실행 여부, git 첫 커밋(코드 버전 'nogit' — 봉인 전에 권장).

### 최근 실측(인용 가능, 결과 json 기준)
- H1(미국 반도체 익일 역추세, Ridge 1회 분할 2021~24): 방향 54.8%, DM p 0.003, 샤프 1.05, Deflated p 0.21 → 사전 기준 미달. lag-shift: Δ=+1일에 완전 소멸.
- F11(KB 주택매매가격지수 k=1, 2026-11-07 재실행 기준): naive RMSE 0.00328 < ARIMA 0.00356(_v1107, DM p 0.28) < SARIMA 0.00385(_v1107, p 0.055) < Ridge 0.00413(_v1105) < LSTM 0.00452(_v1107, p 0.001). 모두 naive 미달. F11 월 변화율의 전월 자기상관 0.84 — naive 가 강한 이유. lag-shift Δ+90 까지 불변.
- 매매(_v1107, 평가 2012-05~2024-12, 봉인 제외): B&H 샤프 0.310 · MA 교차 0.262 · 모멘텀 0.025 · LightGBM(월간 12) 0.112 · LSTM(09) 0.331 — 부트스트랩·Deflated p 모두 0.5 이상(신호 없음).


## 완료
- [x] 2026-11-04 변수 기본값(마스터 `transform` 열, lag 추정값+근거 note) · 모델 파라미터 주기별 기본값(`default_by_freq`) · 화면/가이드/결과 json 반영(`transforms_used`)
- [x] 기획서 (claude.ai 문서) https://claude.ai/artifact/Rq9FKapaoUmh5Ybac8BZeD
- [x] `docs/DECISIONS.md` 확정 사항
- [x] A. BIVA 점검 + 재현성 검증 → `docs/biva_review.md` (1~8장)
- [x] 1A 엔진 골격: engine/ models/ adapters/ tests/ collect.py cli.py (파일 목록은 DECISIONS 참조)
- [x] 테스트: `python -m pytest tests -q` — calendar·backtest 회계·DM·플러그인 누출 검사(심은 신호 찾기 / 랜덤워크 무능 / 타겟 섞기)

## 다음 할 것
- [x] (11-09) duplicate_of·active 규칙 확정(값 없음+N = 빼고 기록) · 실행 기록 `run_record`·`run_log.jsonl`·결과 덮어쓰기 대신 `history/` 보관 · 변수 사전 사본 `data/master_history/` (DECISIONS 11-09)
- [x] (11-09) 기술지표 개별·묶음 선택(`technical_features`, 처음 전부 켬, 해시 불변) — DECISIONS 11-09
- [x] (11-09) 한 변수 = 한 출처 검사(`engine/master_check.py`) — 규칙대로 자동 처리, 애매한 것만 멈춤 · 보기 전용 내보내기 `tools/export_version.py` (DECISIONS 11-09)
- [x] (11-09) lag_confirmed 빈칸 = 확인 안 됨(N) — 지금 파일 영향 없음
- [ ] git 저장소 만들기(code_version nogit 해소 — 실행 기록의 남은 빈칸)
- [x] 창 모델 재실행(11-07 엔진 수정) — f11_lstm·lstm·09_lstm·repro_biva_orig·f11_biva_macro·repro_biva_verified(_v1107) 완료, `configs/repro_biva_orig.json`(지금 설정) 첫 실행 `repro_biva_orig_9d68fbbfb7e3`.
- [x] **10_biva_ABC 20 epoch 판**(사용자 요청, 11-07): `configs/10_biva_ABC_e20.json` → `10_biva_ABC_e20_v1107_6f27528777bf`(클라우드 CPU, 폴드 학습 157분). 샤프 −0.034 vs B&H 0.310, 부트스트랩 p 0.9965 — 유의한 신호 없음. 폴드 5/50 은 epoch 20 이 검증 최저(덜 학습).
- [ ] 10_biva_ABC 를 epoch 상한을 늘려(60 또는 원본 500) 돌릴지 — 노트북 GPU 권장(`python tools/resumable_run.py <설정>`, 끊기면 같은 명령으로 이어 돌림). 옛 결과 2개(cfe97·420477)는 봉인 구간까지 포함된 옛 평가.
- [x] 코드 따라가기 확장(전 모델).
- [ ] **코드 따라가기 음성(ElevenLabs)**: Ridge 완료(Daniel). 사용자 확인 뒤 나머지 8개 화면의 say·코드 줄 설명·직접 실행해 보기를 쓴다(대본 약 4만 자 — 크레딧 확인).
- [ ] **사용자 결정**: 창 모델(LSTM·GRU·BIVA)의 시드를 폴드마다 다시 고정할지(폴드 하나만으로 재현 가능해지지만 모든 창 모델 결과가 바뀜).
- [ ] 아직 결과가 없는 설정: `f11_vecm_k1`·`f11_vecm_auto_k1`, GRU(f11), `configs/12_lgbm_ABCD_M.json`(파라미터 바뀐 뒤), `configs/repro_biva_verified.json`(지금 설정). 옛 결과 04·05 는 _v1107 로 대체.
- [x] 실험 카운터 경합 수정(`cli.py`, 동시 실행 시 횟수 감소) — 예측 카운터 20 으로 보정.
- [ ] **BIVA 재현 실행(GPU)**: `python cli.py run configs/repro_biva_orig.json`, `configs/repro_biva_verified.json` → 가이드 30_모델 「BIVA 원본 설정 재현」 표에 채우기(원본 fixed-diff RMSE 0.706).
- [ ] 매매 결과 앙상블, Model Confidence Set, BIVA lr 감쇠 옵션.
- [ ] **옛 결과 재실행** (`python tools/rerun_stale.py` 로 목록, `--run` / GPU 는 `--run --all`): lag 검증(11-05) 이전에 저장된 결과(F11 ridge/lstm/biva macro, 매매 h·01~14)는 옛 lag 로 계산됨(결과의 `lag_days_used` 로 구분). 논문 표는 검증 lag 로 다시 돌린 결과만 쓸 것. LSTM·BIVA 는 노트북 GPU 에서.
- [ ] **git 저장소**: `git_first_push.bat` 준비됨(사용자가 GH_USER 입력 후 1회 실행). 이후 세션은 GitHub 저장소를 직접 붙여 작업 가능. 그 전 결과의 `code_version` 은 "nogit".
- [x] 다중비교: Holm 보정 표시(2026-11-05). MCS 는 미구현.
- [ ] 논문용 표 내보내기(선택한 결과 → CSV/LaTeX) — 미구현.
- [ ] **실시간(vintage) 데이터**: 관측별 실제 공개일 표 + 발표 당시 값. 미국은 ALFRED, 한국은 KOSIS 시점자료/보도자료 원문. 엔진에 `release_dates` 입력 경로 추가(고정 lag 대체). 논문의 남은 가장 큰 데이터 한계.
- [ ] lag 미확인 16개: N19~N21·O4(2016년 이후 공표 방식), H3·O3(표본 2건), USDKRW(Yahoo 일봉 경계 → ECOS 731Y003 교체 권장), 수급 9개(데이터 없음).
- [x] ~~VECM~~ (2026-11-05 구현) — 이전 메모: 수준값 입력(transform none) 여러 개 + 대상 수준값 예측 → 수익률 변환. `models/base.SeriesModel` 을 다변량으로 확장하거나 새 INPUT_SHAPE "levels" 필요. 공적분 검정(Johansen) 결과를 결과 json 에 기록.
- [x] ~~앙상블~~ (2026-11-05 구현, 예측 연구만) — 이전 메모: 저장된 결과 2개 이상의 prediction 시리즈를 평균해 같은 지표로 재평가하는 후처리(`tools/ensemble.py` → results/<mode>/ensemble_*.json). 모델 플러그인이 아니라 결과 조합.
- [ ] 화면 "결과" 에 lag-shift 표 버튼(지금은 `tools/lag_shift.py` CLI 만).
- [ ] **lag_days 확인**: 65개 전부 AI 추정(2026-11-04). 변수 사전 "비고" 의 근거를 기관 공표일정과 대조 → 맞으면 엑셀 `lag_confirmed=Y`. 우선순위: 실험에 자주 쓰는 F7, K1~K4, B1, A1, F11, M3, L9.
- [ ] lag-shift 테스트(모든 lag +30/+60)로 추정 lag 가 너무 작지 않은지 확인.
- [x] 데이터 수집 완료 (`data/versions/20261102T134628Z`, 네이버+Yahoo 가격, FRED·ECOS·엑셀 변수). 결측일 보정 `engine/repair.py`
- [x] 실데이터 실험 12건 완료 → `docs/results_round1.md`. 결론(정정 후): **일간·주간·월간 모두 신호 없음**
- [x] BIVA 속도: 결측 없으면 BRITS 건너뜀(예측 동일) 6시간+ → 1분. 폴드 진행률 출력 추가. 기술지표 창을 주기별로 조정
- [x] 변수 사전 `data/variable_master.xlsx` 를 유일 기준으로: `Registry.load_master`, duplicate_of 자동 치환, `collect.py` 마스터 기반 재작성, uploads 자동 등록, DERIVED 계산, ID 통일(데이터 버전 `20261103T130711Z`)
- [ ] 사용자: 마스터 엑셀 검토(group / lag_days / duplicate_of). 특히 **BOK_RATE lag_days 가 초안에서 0**(1차 수집 때는 7) — 이 차이만으로 06번 결과가 dir_acc 0.5045→0.5000 로 바뀜. O1~O4 는 엑셀에 데이터가 없음(var_info 에만 있음)
- [x] `ablate.py` 구현. 첫 실행에서 월간 12번이 퇴화 모델(상수 예측)임을 발견 → **1차 월간 후보 철회**(`results_round1.md` 정정). lag_days 변수별 + `lag_confirmed` 열 + 주기별 기본값 구현
- [x] 공공데이터포털 어댑터 검증 완료(V2 경로, 연 단위 분할, 재시도). **정식 데이터 버전 `20261103T141116Z`**: 가격 = datago(2020~) + Yahoo(2007~2019, 공식과 100% 일치) + 네이버 결측 보정(091160 7일, 091230 13일). 변수 52개. 설정 12개를 이 버전·`ticker: 091160` 으로 갱신. 기준 실험(B&H, 06) 재실행 — 수치 변화 없음
- [ ] 주의: 2026-07-28~31 두 ETF 모두 -25%→+29% 급변. 공식·Yahoo 동일하므로 실제 시장으로 보이나, 봉인 구간 성과 해석 시 염두
- [x] 월간 Ridge 재실험(alpha 1/100/1000): 신호 없음, 0예측에 수렴. ablation 은 기준 신호가 없어 보류(`results_round1.md` 월간 재실험 절)
- [x] 2차 가설 실험 14건 → `docs/results_round2.md`. H1(미국 반도체 익일 **역추세**)만 후보: Ridge 샤프 0.68(B&H 0.565), 유의하지 않음. 거래비용이 효과 대부분을 먹음
- [x] H1 정식 검정: 규칙 모델 `sox_gap_fade`(롤링 분위수) 는 B&H 와 동률, Ridge 는 샤프 0.544 vs 0.367 이나 p 0.24·슬리피지 10bp 면 소멸 → **성공 기준 미달, 봉인 미사용**. 평가구간 버그 수정(봉인 구간이 성과에 섞여 있었음). `--sealed` 옵션 + sealed_log 추가
- [x] 1B 1단계: `api/main.py`(메타·제출·작업·결과·시리즈) + `api/jobs.py`(파일 큐) + `web/`(React). `run_server.bat` 로 실행. 클라우드에서 제출→실행→결과 표시까지 확인
- [x] 1회 분할 모드(`validation.mode=split`) + 실제/학습 적합/테스트 예측 차트. H1 Ridge 2021~24: DM p 0.0004, 부트스트랩 p 0.003, Deflated p 0.57(미달) → `results_round2.md`
- [x] 코드 점검 A1~A6 수정(Ridge 전체 학습, 시드 통일, 변동성 연율화, Deflated 정식, dead code 제거, 분류 DM) + 테스트 27개
- [x] 가이드 문서 8편 `docs/guide/` → `tools/build_guide.py` 가 파라미터·변수 표를 코드에서 채워 `docs/guide_build/`·`docs/GUIDE.md` 생성, 화면 '도움말' 메뉴
- [x] 두 연구 모드 분리(매매/예측): 엔진 forecast 모드, 결과 저장소·카운터 분리, 화면 모드 스위치·예측 전용 폼·결과·차트(naive 포함), 가이드·의사코드, 테스트 28개. F11 1차 결과 기록
- [ ] 예측 연구 다음: lag 확인, k=3/6/12, 1회 분할로 원본 BIVA 설정 재현, F12
- [ ] 1B 다음: ablation 화면, 결과 비교(두 실험 겹쳐 보기), 마스터 엑셀 편집 UI(지금은 읽기 전용)
- [x] 봉인 평가 버튼(2026-11-06: 미리보기·사전 기록·확인 2개·재개봉 확인·봉인 기록)

## 사용자 확인 필요
- 데이터 출처 결정: 공공데이터포털 키 발급 여부 / 허용 도메인 추가 여부
- BIVA 점검 결과를 교재·연구실에 어떻게 반영할지 (플랫폼 외부)

## 메모
- GPU: 노트북 torch 2.11+cu128 (드라이버 576, CUDA 12.9 → cu13x 불가). `python check_env.py` 로 확인.
- 노트북 Python 3.14 사용 중(3.13 권고였으나 동작 확인).
- Claude 셸(노트북 Linux VM: CPU 2, RAM 3GB, GPU 없음) 및 클라우드 모두 금융 데이터 사이트 차단. pip 만 허용.
- `TimeSeriesPredict` 폴더는 Claude 셸에 마운트되지 않음 → 파일 전달은 stage/commit 도구.
- `.env` 는 도구로 쓸 수 없음(보안) → 사용자가 직접 생성(완료).
- API 키가 채팅에 노출됨 → 공개 서비스 전 재발급 권장.
- Python 3.13 기준. 클라우드 컨테이너에는 pandas 2.3 설치(3.x 는 원본 BIVA 로더와 비호환).
