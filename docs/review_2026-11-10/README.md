# 2026-11-10 점검의 증거 스크립트

이 폴더의 스크립트는 **그때의 사본 경로**(`/tmp/e2e`, `/home/claude/tsp`)를 적은 채로 보관한 기록이다. 다시 돌리려면 맨 위의 경로(`sys.path.insert`, `ROOT`)를 이 저장소 위치로 바꿔야 한다. 돌린 시점의 코드는 2026-11-10 낮의 것(엔진 2026-11-07 판)이라, 지금 코드로 돌리면 고쳐진 뒤의 값이 나온다.

| 파일 | 무엇을 쟀나 | 결과가 쓰인 곳 |
|---|---|---|
| `power_check.py`, `power_check_result.json` | 심은 신호의 크기별 검출 횟수(합성 자료, 20회) | REVIEW_2026-11-10 2절, 사전 분석 계획서 8절 |
| `lgbm_trees_probe.py`, `.json` | LightGBM 이 조기 종료 뒤 실제로 쓴 트리 수(일간 149·주간 166·월간 127 폴드) | 도움말 '결과 읽기 — 학습 진단', DECISIONS 2026-11-10 심야 |
| `arima_multistart_probe.py` | ARIMA·SARIMA 를 여러 출발점에서 추정했을 때 기본 출발점보다 로그우도가 높은 해가 나오는 폴드 수(F11, 30개 폴드 표본) | `models/base.py` `fit_multi_start` 의 설명 |
| `indep_c/cw_size.py` | Clark–West 의 크기(정보 없는 변수의 기각률): 타겟 평균·겹치는 타겟·분산 핵에 따라 | 독립 검토 F2·F12 → `engine/backtest.py` |
| `indep_c/cw_decomp.py` | F11 에서 CW(0 기준) 통계량 가운데 '평균을 맞힌 몫' | 독립 검토 F2 |
| `indep_c/ledger_tests.py`, `ledger_conc.py` | 시도 원장: 끝 줄 삭제·끊긴 줄·동시 쓰기 | 독립 검토 F9·F15 → `engine/ledger.py` |
| `indep_c/store_tests.py`, `sheet_tests.py` | 엑셀 날짜·값 칸 읽기, 낡은 크기 표시 | 독립 검토 F5·F6·F19 → `engine/store.py` |
| `indep_c/api_tests.py`, `seal_fail_after_metrics.py`, `collect_rehearse.sh` | 봉인 접수·개봉 횟수, 수집 실패 리허설 | 독립 검토 F1·F7·F8·F10 |
