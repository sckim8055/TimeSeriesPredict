"""모델 플러그인 규격 (기획서 5-1). 모델 1개 = models/ 파일 1개.

모델 파일이 선언하는 것
  NAME          화면·결과에 쓰는 이름
  INPUT_SHAPE   "series" | "table" | "window" | "rule"
                - series: fit(y[T], X[T,k]) — 통계 모델(ARIMAX 등). 예측은 마지막 시점 이후 1스텝.
                - table : fit(X[T,k], y[T]) — 행 독립 가정 ML.
                - window: fit(X[N,L,k], y[N]) — 딥러닝.
                - rule  : 학습 없음. predict(prices_df) → 목표 포지션(0/1). 기준선 전략.
  TARGET_TYPES  ["regression"] / ["classification"] / 둘 다 / ["signal"](rule)
  PARAMS        {이름: {"type", "default", "min"?, "max"?, "choices"?, "desc"}} — UI 폼 자동 생성, 모델별로 따로 저장
  class Model(BaseModel)

모델이 받지 못하는 것: 날짜, 변수 이름, 가격 수준, 경로, 시드(엔진이 torch/numpy 시드를 설정한 뒤 호출), 장치(엔진이 device 인자로 전달).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

ParamSpec = dict[str, dict[str, Any]]


def resolve_params(spec: ParamSpec, given: dict[str, Any] | None, freq: str | None = None) -> dict[str, Any]:
    """PARAMS 선언에 given을 덧씌우고 타입·범위를 검사한다. 선언에 없는 키는 오류(오타 방지).
    freq(D/W/M)가 주어지고 선언에 `default_by_freq`가 있으면 그 주기의 기본값을 쓴다(표본 수에 맞춘 기본값)."""
    given = dict(given or {})
    unknown = set(given) - set(spec)
    if unknown:
        raise KeyError(f"선언되지 않은 파라미터: {sorted(unknown)}")
    out: dict[str, Any] = {}
    for k, s in spec.items():
        default = s.get("default")
        if freq and isinstance(s.get("default_by_freq"), dict) and freq in s["default_by_freq"]:
            default = s["default_by_freq"][freq]
        v = given.get(k, default)
        t = s.get("type", "float")
        if v is not None:
            if t == "int":
                v = int(v)
            elif t == "float":
                v = float(v)
            elif t == "bool":
                v = bool(v)
            elif t == "str":
                v = str(v)
            if "min" in s and v < s["min"]:
                raise ValueError(f"{k}={v} < min {s['min']}")
            if "max" in s and v > s["max"]:
                raise ValueError(f"{k}={v} > max {s['max']}")
            if "choices" in s and v not in s["choices"]:
                raise ValueError(f"{k}={v} not in {s['choices']}")
        out[k] = v
    return out


class BaseModel(ABC):
    """모든 모델의 공통 인터페이스.

    USES_VALIDATION: True 면 엔진이 학습 창 뒤 val_frac 를 검증용으로 떼어 X_val/y_val 로 넘긴다(조기종료 등).
    False 면 학습 창 전체를 학습에 쓴다(검증을 쓰지 않는 모델이 15% 를 버리지 않도록)."""

    USES_VALIDATION = True

    def __init__(self, params: dict[str, Any], *, n_features: int, target_type: str, device: str = "cpu"):
        self.params = params
        self.n_features = n_features
        self.target_type = target_type
        self.device = device

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> None: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """regression → 예측값 [N]; classification → 양(+) 클래스 확률 [N]."""

    def feature_importance(self) -> np.ndarray | None:
        """지원 모델만. [k] 배열."""
        return None


class SeriesModel(BaseModel):
    """단변량 통계 모델(ARIMA 계열). INPUT_SHAPE='series'.
    엔진은 대상 시계열의 **1기간 변화율** r_t = close_t/close_{t-1} − 1 만 넘긴다(외생변수 없음, 스케일링 없음).
    fit(r_train) 으로 적합하고, forecast(r_history, steps) 가 steps 기간 앞까지의 1기간 변화율 예측 [steps] 를 돌려준다.
    엔진이 이를 복리로 묶어 k기간 타겟 ∏(1+r̂)−1 로 바꾼다. 회귀 타겟만 지원."""

    USES_VALIDATION = False

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> None:  # X = r_train [N]
        self.fit_series(np.asarray(X, dtype=float).ravel())

    def predict(self, X: np.ndarray) -> np.ndarray:  # 엔진은 series 모델에 predict 대신 forecast 를 쓴다
        raise NotImplementedError("series 모델은 forecast(history, steps) 를 쓴다")

    @abstractmethod
    def fit_series(self, r: np.ndarray) -> None: ...

    @abstractmethod
    def forecast(self, history: np.ndarray, steps: int) -> np.ndarray:
        """history: 결정 시점까지 알려진 1기간 변화율 전부(학습 구간 포함). 반환 [steps]."""


class LevelsModel(BaseModel):
    """다변량 수준값 모델(VECM 등). INPUT_SHAPE='levels'.
    엔진은 [대상 수준값, 선택한 변수들의 수준값(변환 없음, 발표시차 적용)] 행렬을 넘긴다(열 0 = 대상).
    fit_levels(학습창 행렬) 로 적합하고, forecast_levels(결정 시점까지의 행렬, steps) 가 대상 수준값 경로 [steps] 를 돌려준다.
    엔진이 v̂[t+k]/v[t]−1 (로그수익률이면 로그) 로 바꿔 다른 모델과 같은 타겟으로 비교한다. 회귀 타겟만."""

    USES_VALIDATION = False

    def fit(self, X, y, X_val=None, y_val=None) -> None:
        self.fit_levels(np.asarray(X, dtype=float))

    def predict(self, X):
        raise NotImplementedError("levels 모델은 forecast_levels(history, steps) 를 쓴다")

    @abstractmethod
    def fit_levels(self, Y: np.ndarray) -> None: ...

    @abstractmethod
    def forecast_levels(self, history: np.ndarray, steps: int) -> np.ndarray: ...


class RuleModel(ABC):
    """학습 없는 기준선 전략. INPUT_SHAPE='rule'. 가격 DataFrame(open/high/low/close/volume)을 받아 목표 포지션(0/1)을 낸다.
    결정일 t의 포지션은 t 마감까지의 정보만 쓴다.
    X(선택): 설정의 variable_ids 로 조립된 외생변수 DataFrame(정렬·시차·변환 적용 후). 쓰지 않는 규칙은 무시한다."""

    USES_VARIABLES = False  # True 면 positions(prices, X) 로 호출

    def __init__(self, params: dict[str, Any]):
        self.params = params

    @abstractmethod
    def positions(self, prices, X=None) -> "np.ndarray": ...
