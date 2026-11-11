"""ARIMA(p,d,q) — 단변량 통계 모델. INPUT_SHAPE='series'.

엔진이 넘기는 것은 대상 시계열의 **1기간 변화율** r_t 이므로, 기본 d=0 이 원 노트북(시계열기초강좌 노트북2)의
"수준값에 ARIMA(2,1,2)" 에 대응한다(둘 다 한 번 차분한 것을 모델링). 같은 모델은 아니다 — 수준 차분 Δv 와 변화율 Δv/v 는
다르고, 기본 trend='c'(절편)는 수준 d=1 ARIMA 의 기본(상수 없음)과 다르다. 외생변수는 쓰지 않는다 — 입력 변수를 골라도 무시된다.
예측: 결정일까지의 변화율로 k 기간 앞까지 예측(statsmodels `apply`, 재적합 없음) → 엔진이 복리로 묶어 k기간 수익률로 비교.
매매 모드의 타겟은 시가→시가 수익률이지만 이 모델은 종가 변화율로 예측한다(1기간 차이, 문서화).
추정 제약(2026-11-06): enforce_stationarity·enforce_invertibility 를 파라미터로 둔다. 기본은 예(True) — statsmodels 와 원 노트북
(`ARIMA(훈련, order=(2,1,2)).fit()`)의 기본과 같다. 그 전 이 도구는 둘 다 아니오(False)로 고정이었고, 실데이터에서 가역이 아닌 MA 가 나왔다.
옛 결과를 그대로 재현하려면 둘 다 아니오로 둔다.
수치 배율(2026-11-07): 변화율(월간 0.3% 안팎 → 분산 1e−5)을 그대로 최우추정하면 L-BFGS 가 잘 수렴하지 않는다(F11 월간 120 폴드: ARIMA 15개,
SARIMA 83개 미수렴, 수렴했다고 나온 폴드도 최대점이 아닌 경우가 있음 — 같은 자료에 ×100 을 하면 로그우도가 더 높은 점을 찾음). 그래서 변화율에
scale(기본 100)을 곱해 추정하고 예측을 다시 나눈다. ARMA 는 배율에 대해 같은 모형(φ·θ 그대로, c 는 ×scale, σ² 는 ×scale²)이라 바뀌는 것은
최적화의 수치 조건뿐이다(정상성 제약을 켜면 정상 초기화라, 환산한 같은 모수에서 로그우도 차이가 정확히 n·ln(scale) — F11 마지막 폴드 확인). 단 배율 100 도 최댓값을 보장하지는 않는다
— ARMA(2,2) 가능도에는 극대점이 여러 개 있을 수 있다(2026-11-08 독립 검토: 다른 최적화법으로 더 높은 점이 있는 폴드가 있었음, 엔진은 그대로).
scale=1 이면 2026-11-07 전과 같다. 옛 결과(2026-11-06 전)를 그대로 재현하려면 제약도 꺼야 한다 — LEGACY_DEFAULTS 가 결과 파일에 기록이 없을 때 둘 다 채운다.
"""
from __future__ import annotations

import warnings

import numpy as np

from models.base import SeriesModel

NAME = "ARIMA"
INPUT_SHAPE = "series"
TARGET_TYPES = ["regression"]
PARAMS = {
    "p": {"group": "차수·상수", "type": "int", "default": 2, "min": 0, "max": 12, "desc": "자기회귀 차수 — 과거 몇 기간의 변화율을 쓰나"},
    "d": {"group": "차수·상수", "type": "int", "default": 0, "min": 0, "max": 2, "desc": "차분 횟수 — 입력이 이미 변화율이므로 보통 0 (수준값 ARIMA(2,1,2) 에 대응. 같은 모델은 아님: Δv 와 Δv/v 는 다르다)"},
    "q": {"group": "차수·상수", "type": "int", "default": 2, "min": 0, "max": 12, "desc": "이동평균 차수 — 과거 몇 기간의 예측오차를 쓰나"},
    "trend": {"group": "차수·상수", "type": "str", "default": "c", "choices": ["n", "c"], "desc": "상수항 — c: 절편 포함(기본, 평균 변화율 = c/(1−Σφ)), n: 없음"},
    "enforce_stationarity": {"type": "bool", "default": True, "group": "추정 제약",
                             "desc": "정상성 제약 — 예(기본): AR 계수를 '충격이 시간이 지나면 사라지는' 범위(AR 다항식 근의 크기 > 1)로만 찾는다(statsmodels·원 노트북 기본). 아니오: 제약 없이 찾는다(2026-11-06 전 이 도구의 기본)"},
    "enforce_invertibility": {"type": "bool", "default": True, "group": "추정 제약",
                              "desc": "가역성 제약 — 예(기본): MA 계수를 '과거 오차가 갈수록 덜 남는' 범위(MA 다항식의 근 > 1)로만 찾는다(statsmodels·원 노트북 기본). 아니오: 제약 없이 찾는다 — 가역이 아닌 θ 가 나올 수 있고 그때 θ 는 해석할 수 없다(예측값은 계산됨)"},
}
PARAMS["scale"] = {"type": "float", "default": 100.0, "min": 1.0, "max": 1e6, "group": "추정 제약",
                   "desc": "추정용 배율 — 변화율에 이 수를 곱해 추정하고 예측을 다시 나눈다. 모형·예측식은 같고(φ·θ 는 배율과 무관) 최우추정의 수치 조건만 좋아진다(변화율 분산이 1e−5 수준이면 잘 수렴하지 않음). 1 = 배율 없음(2026-11-07 전)"}
PARAM_GROUPS = ["차수·상수", "추정 제약"]
# 결과 파일의 model.params 에 이 키가 없으면(2026-11-06 전 결과) 그 실험은 이 값으로 돌았다 — 같은 실험을 다시 돌릴 때 엔진이 채운다
LEGACY_DEFAULTS = {"enforce_stationarity": False, "enforce_invertibility": False, "scale": 1.0}


class Model(SeriesModel):
    def fit_series(self, r: np.ndarray) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        p = self.params
        self.scale = float(p.get("scale", 1.0) or 1.0)  # 수치 배율(위 설명) — 추정·예측 모두 r × scale 단위, 예측은 다시 나눔
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = SARIMAX(r * self.scale, order=(p["p"], p["d"], p["q"]), trend=p["trend"], enforce_stationarity=p["enforce_stationarity"], enforce_invertibility=p["enforce_invertibility"]).fit(disp=False, maxiter=200)

    def forecast(self, history: np.ndarray, steps: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = getattr(self, "scale", 1.0)
            return np.asarray(self.res.apply(np.asarray(history, dtype=float) * sc, refit=False).forecast(steps), dtype=float) / sc
