# TimeSeriesPredict — 프로젝트 지침 (모든 세션이 먼저 읽는 파일)

이 저장소는 **반도체 ETF 자동매매 시뮬레이션 플랫폼**의 전체 코드·문서·데이터를 담는다.
어떤 AI 모델·어떤 세션이 작업하든 아래 규칙을 따른다. 세션 시작 시 반드시 `docs/DECISIONS.md`(확정 사항)와
`docs/STATUS.md`(현재 진행 상태, **0절 인수인계부터**)를 읽고, 작업이 끝나면 `docs/STATUS.md`를 갱신한다.
모델 플러그인 규약 요약: INPUT_SHAPE 는 table(행렬) / window(과거 seq_len 행) / series(대상 변화율만, `SeriesModel`) / rule(가격→포지션). 새 모델은 `tests/test_engine.py` 의 합성 테스트(심은 신호 찾기·잡음에서 무능)를 추가해야 등록.

## 사용자와의 소통 규칙
- 한국어로 답한다. 영어 UI·문서는 만들지 않는다.
- 긍정적 답변보다 **문제·오류·한계를 그대로** 말한다. 안 되는 것은 안 된다고 명확히 말한다.
- 전문용어(누출, 워크포워드, 봉인구간 등)는 처음 쓸 때 한 줄로 풀어 설명한다.
- 사용자는 Docker에 익숙하지 않다. Docker 관련 작업은 명령 하나하나에 설명을 붙인다.

## 프로젝트 한 줄 정의
정해진 입력변수 세트 안에서 **변수 조합 × 모델 × 타겟(예측 지평) × 매매 주기(D/W/M) × 매매규칙**을 바꿔
예측력과 매매 성과를 비교하는 시뮬레이션 도구. 논문(SCI)은 별도 프로젝트이며, 이 도구는 그 실험 도구다.

## 사용자 원칙(2026-11-09) — 쉽게
- 쓰는 분들은 이 분야에 익숙하지 않은 어르신들. 사용자에게 여러 판단을 넘기거나 여러 단계 절차를 만들지 말 것. 프로그램이 한 규칙으로 처리할 수 있으면 알아서 처리하고 알림·기록만, 짐작할 수 없을 때만 멈추고 고칠 곳을 한 줄로.

## 변수 사전
- `data/variable_master.xlsx` 가 유일한 변수 사전이자 **직접 넣는 데이터의 유일한 파일**(2026-11-09): variables 시트 = 변수 정의, 데이터 시트 df_D/df_W/df_M/df_Q/df_H/df_Y = 엑셀로 넣는 값(source=XLSX, source_code=시트!열). `data/uploads/` 의 엑셀은 읽지 않는다. 한 변수 = 한 출처(`engine/master_check.py`): API 변수와 같은 이름의 데이터 시트 열은 읽지 않는다. API 로 받는 값은 엑셀에 넣지 않고 collect.py 가 `data/raw/<출처>/*.json` 원본 + `data/raw/fetch_log.jsonl` 이력으로 남긴다. 엔진·실험은 엑셀을 직접 읽지 않고 collect 로 만든 데이터 버전(`data/versions/<id>`)만 읽는다(엑셀을 고쳐도 이미 낸 결과는 안 바뀜).
- 변수 사전 설명: `registry.csv` 는 파생물 — 고치지 말 것. 열 의미는 엑셀 "설명" 시트와 `docs/DECISIONS.md`(2026-11-03). `duplicate_of` 와 `active` 는 분리된 2열, 합치지 않는다. 실험 규칙(2026-11-09 사용자 확정): duplicate_of 있음 → 대표로 치환(자기 active 무관), 없음+Y → 포함, 없음+N(또는 대표가 N) → 빼고 run_record 에 기록, 모두 빠지면 오류.
- 실행 기록(2026-11-09 사용자 요구 "모든 작업은 이력"): 결과 json `run_record`, `results/<mode>/run_log.jsonl`(추가만), 같은 이름 결과는 `history/` 로 보관(덮어쓰지 않음), 변수 사전 사본 `data/master_history/`. 이 기록을 빼거나 덮어쓰는 수정 금지. `transform` 열 = 기본 변환(설정이 우선), `lag_days` 는 2026-11-04 현재 전부 AI 추정값(근거는 `note`)이며 확인 전까지 `lag_confirmed=N` 을 지운다거나 Y 로 바꾸지 말 것.

## 문서
- 사용자 가이드 원본은 `docs/guide/*.md`(8편). 파라미터·변수 표는 자리표시자로 두고 `tools/build_guide.py` 가 코드에서 채운다 — 표를 손으로 쓰지 말 것. 코드(PARAMS·PRESETS·마스터)를 바꾸면 가이드가 자동으로 따라오지만, 수치·사례·설명 문장은 사람이 고친다.

## 절대 규칙 (어기면 비교가 무효가 된다)
1. **타겟 정렬**: 모델이 맞히는 값은 항상 입력 구간 *바깥*의 미래 값. 체결은 다음 거래일 시가 기준.
   날짜·시프트·리샘플·fill 연산은 `engine/calendar.py` 한 곳에서만 한다.
2. **스케일링·변수선택은 학습 구간 데이터만으로 fit**. 검증·테스트 구간은 transform만.
3. **모델 파일은 텐서만 받고 텐서만 돌려준다.** 날짜·변수명·가격·경로·장치·시드를 모델 파일이 알지 못하게 한다.
4. **모델 1개 = `models/` 파일 1개.** 파라미터는 파일 안 `PARAMS` 선언으로 모델별로 받는다. 하드코딩 금지. 기본값이 주기에 따라 달라야 하면 `default_by_freq: {D,W,M}` 로 선언한다(`resolve_params` 가 처리).
5. **의존 방향**: `api → engine → models / adapters`. 모델·어댑터는 엔진을 import하지 않는다.
6. 새 모델은 `tests/`의 합성 데이터 테스트 2개(누출 검사, 사인파 기준선)를 통과해야 등록된다.
7. 모든 실험 결과에 `data_version`, `code_version`(git 커밋), 시드, 설정 JSON을 기록한다.
8. 외부 데이터는 공식 API/라이브러리(pykrx, yfinance, ECOS, FRED)만. 네이버 금융 스크래핑 금지.
9. 실전(당일·미래) 매매 신호는 운영자 전용 화면에만. 공개 화면은 과거 백테스트 비교까지만.
10. 원본 연구 코드(`D:\test\SVU_SCT_FIN-main`)는 **수정하지 않는다**. 복사본만 `legacy/`에 둔다.

## 기술 스택 (확정)
Python 3.13 · FastAPI · RQ+Redis · PostgreSQL · Parquet · PyTorch(CUDA) · LightGBM · statsmodels ·
React+TypeScript · TradingView Lightweight Charts · ECharts · Docker Compose · (공개 시) Cloudflare Tunnel

## 폴더 구조
```
CLAUDE.md            이 파일
docs/                DECISIONS.md(확정 사항) STATUS.md(진행 상태) 기획서 사본, 검토 보고서
legacy/              원본 연구 코드 복사본 및 점검 (biva_orig: 원본 그대로 / biva_orig_run: 경로·장치만 패치 / biva_fixed: 수정본)
engine/ models/ adapters/ api/ web/ tests/   (1A 단계에서 생성)
```

## 작업 방식
- 엔진 먼저, 화면 나중. 실데이터 결과를 보고 기획이 바뀔 수 있다.
- 코드를 바꾸면 테스트를 돌린다. "돌아간다"와 "값이 맞다"는 다르다 — 결과가 너무 좋으면 버그를 먼저 의심한다.
- 큰 결정이 바뀌면 `docs/DECISIONS.md`에 날짜와 함께 추가한다. 지우지 않는다.
