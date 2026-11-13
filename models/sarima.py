"""SARIMA(p,d,q)(P,D,Q,s) — 계절 ARIMA. INPUT_SHAPE='series'.

원 노트북(노트북2)의 SARIMAX(order=(1,1,1), seasonal_order=(1,1,1,12)) 를 변화율 입력에 맞춰 옮긴 것:
수준값의 d=1 은 변화율 입력에서 d=0 과 같다.
계절 차분 D(2026-11-10): 기본 0. 그 전에는 노트북의 D=1 을 그대로 옮겼는데, 입력이 이미 변화율이라 계절 차분을 또 하면 과차분이다 — F11 월간 120 폴드 중
87개에서 계절 MA 계수 |Θ|>0.9(중앙값 −0.967): 차분을 MA 가 되돌리는 교과서적 신호. 변화율에 계절 단위근이 있다는 근거(검정)도 없었다.
D=0 이면 상수항을 넣는다(trend=auto) — 변화율의 평균(물가라면 양수)을 0 으로 두지 않기 위해. 노트북 재현은 D=1, trend=n 으로.
s 는 주기별 기본값: 일간 5(한 주), 주간 52, 월간 12. 일간 5 는 요일 효과 가정이라 근거가 약하다 — 월간 데이터에 쓰는 것이 본래 용도.
추정 제약(2026-11-06): enforce_stationarity·enforce_invertibility 를 파라미터로 둔다. 기본은 예(True) — statsmodels 와 원 노트북
(`SARIMAX(훈련, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)`)의 기본과 같다. 그 전 이 도구는 둘 다 아니오(False)로 고정이었다.
옛 결과를 그대로 재현하려면 둘 다 아니오로 둔다.
수치 배율(2026-11-07): 변화율(월간 0.3% 안팎 → 분산 1e−5)을 그대로 최우추정하면 L-BFGS 가 잘 수렴하지 않는다(F11 월간 120 폴드: ARIMA 15개,
SARIMA 83개 미수렴, 수렴했다고 나온 폴드도 최대점이 아닌 경우가 있음 — 같은 자료에 ×100 을 하면 로그우도가 더 높은 점을 찾음). 그래서 변화율에
scale(기본 100)을 곱해 추정하고 예측을 다시 나눈다. ARMA 는 배율에 대해 같은 모형(φ·θ 그대로, c 는 ×scale, σ² 는 ×scale²)이라 바뀌는 것은
사실상 최적화의 수치 조건이다 — 다만 계절 차분(D=1) 상태의 근사 산만 초기값 분산 10⁶ 은 자료 단위에 고정이라, 같은 모수에서도 로그우도가
아주 조금 다르다(F11 마지막 폴드 5.5e-5, 2026-11-08 확인). scale=1 이면 2026-11-07 전과 같다. 옛 결과(2026-11-06 전)를 그대로 재현하려면
제약도 꺼야 한다 — LEGACY_DEFAULTS 가 결과 파일에 기록이 없을 때 둘 다 채운다.
"""
from __future__ import annotations

import warnings

import numpy as np

from models.base import SeriesModel, fit_multi_start

NAME = "SARIMA"
INPUT_SHAPE = "series"
TARGET_TYPES = ["regression"]
PARAMS = {
    "p": {"group": "차수", "type": "int", "default": 1, "min": 0, "max": 6, "desc": "비계절 자기회귀 차수"},
    "d": {"group": "차수", "type": "int", "default": 0, "min": 0, "max": 2, "desc": "비계절 차분 — 입력이 변화율이므로 보통 0"},
    "q": {"group": "차수", "type": "int", "default": 1, "min": 0, "max": 6, "desc": "비계절 이동평균 차수"},
    "P": {"group": "차수", "type": "int", "default": 1, "min": 0, "max": 3, "desc": "계절 자기회귀 차수"},
    "D": {"group": "차수", "type": "int", "default": 0, "min": 0, "max": 1,
          "desc": "계절 차분 — 기본 0(하지 않음). 입력이 이미 변화율이라 계절 차분을 또 하면 과차분이 된다(2026-11-10 전 기본은 1 — F11 120 폴드 중 87개에서 계절 MA 계수가 −1 근처에 붙었다). 1 이면 s기간 전 대비"},
    "Q": {"group": "차수", "type": "int", "default": 1, "min": 0, "max": 3, "desc": "계절 이동평균 차수"},
    "s": {"group": "차수", "default_by_freq": {"D": 5, "W": 52, "M": 12}, "type": "int", "default": 12, "min": 2, "max": 365, "desc": "계절 주기(기간 수) — 월간 12, 주간 52, 일간 5"},
    "enforce_stationarity": {"type": "bool", "default": True, "group": "추정 제약",
                             "desc": "정상성 제약 — 예(기본): AR 계수를 '충격이 시간이 지나면 사라지는' 범위(AR 다항식 근의 크기 > 1)로만 찾는다(statsmodels·원 노트북 기본). 아니오: 제약 없이 찾는다(2026-11-06 전 이 도구의 기본)"},
    "enforce_invertibility": {"type": "bool", "default": True, "group": "추정 제약",
                              "desc": "가역성 제약 — 예(기본): MA 계수를 '과거 오차가 갈수록 덜 남는' 범위(MA 다항식의 근 > 1)로만 찾는다(statsmodels·원 노트북 기본). 아니오: 제약 없이 찾는다 — 가역이 아닌 θ 가 나올 수 있고 그때 θ 는 해석할 수 없다(예측값은 계산됨)"},
}
PARAMS["scale"] = {"type": "float", "default": 100.0, "min": 1.0, "max": 1e6, "group": "추정 제약",
                   "desc": "추정용 배율 — 변화율에 이 수를 곱해 추정하고 예측을 다시 나눈다. 모형·예측식은 같고(φ·θ 는 배율과 무관) 최우추정의 수치 조건만 좋아진다(변화율 분산이 1e−5 수준이면 잘 수렴하지 않음). 1 = 배율 없음(2026-11-07 전)"}
PARAMS["trend"] = {"type": "str", "choices": ["auto", "c", "n"], "default": "auto", "group": "차수",
                   "desc": "상수항 — auto(기본): 차분을 하지 않으면(d=0, D=0) 상수항을 넣고(변화율의 평균이 0 이 아니므로), 차분을 하면 넣지 않는다. c: 항상 넣음. n: 넣지 않음(2026-11-10 전의 동작)"}
PARAM_GROUPS = ["차수", "추정 제약"]
PARAMS["starts"] = {"type": "str", "choices": ["multi", "default"], "default": "multi", "group": "추정 제약",
                    "desc": "최우추정의 출발점 — multi(기본): 정해진 출발점 7개에서 추정해 로그우도가 가장 높은 해를 쓴다(가능도에 극대점이 여러 개일 수 있어서. 폴드마다 어느 출발점이 뽑혔는지·기본 출발점보다 얼마나 높은지 결과의 학습 진단에 남는다). default: 기본 출발점 하나(2026-11-10 전)"}
# 결과 파일의 model.params 에 이 키가 없으면(2026-11-06 전 결과) 그 실험은 이 값으로 돌았다 — 같은 실험을 다시 돌릴 때 엔진이 채운다
# D 는 넣지 않는다: D 는 처음부터 있던 파라미터라 옛 결과에도 그때 값(1)이 기록돼 있다. trend 는 2026-11-10 에 생긴 칸이라 옛 결과에는 없다 → 그때 값(n)
LEGACY_DEFAULTS = {"enforce_stationarity": False, "enforce_invertibility": False, "scale": 1.0, "starts": "default", "trend": "n"}
# 기본값이 바뀐 파라미터: 옛 결과를 같은 실험으로 다시 돌릴 때(봉인 평가 등) 결과에 기록된 값으로 고정한다(engine.experiment.pinned_model_params)
PIN_RECORDED = ("D",)


class Model(SeriesModel):
    def fit_series(self, r: np.ndarray) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        p = self.params
        self.scale = float(p.get("scale", 1.0) or 1.0)  # 수치 배율(위 설명) — 추정·예측 모두 r × scale 단위, 예측은 다시 나눔
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr = str(p.get("trend", "n") or "n")
            if tr == "auto":
                tr = "c" if (int(p["d"]) == 0 and int(p["D"]) == 0) else "n"
            self.trend_used = tr
            mod = SARIMAX(r * self.scale, order=(p["p"], p["d"], p["q"]), seasonal_order=(p["P"], p["D"], p["Q"], p["s"]), trend=tr,
                          enforce_stationarity=p["enforce_stationarity"], enforce_invertibility=p["enforce_invertibility"])
            self.res, self.diag = fit_multi_start(mod, starts=str(p.get("starts", "default")))  # 여러 출발점 중 로그우도 최대(models/base.py)

    def forecast(self, history: np.ndarray, steps: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = getattr(self, "scale", 1.0)
            return np.asarray(self.res.apply(np.asarray(history, dtype=float) * sc, refit=False).forecast(steps), dtype=float) / sc
