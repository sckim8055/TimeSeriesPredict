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
    # STOCHASTIC: 같은 자료·같은 파라미터라도 난수(가중치 초기값·미니배치 순서·행/열 표본추출)에 따라 결과가 달라지는 모델이면 True.
    # 엔진은 이런 모델에 한해 설정의 n_seeds 만큼 시드를 바꿔 학습하고 예측을 평균한다(2026-11-10). 결정적 모델(릿지·ARIMA 등)은 한 번만 학습.
    STOCHASTIC = False

    def __init__(self, params: dict[str, Any], *, n_features: int, target_type: str, device: str = "cpu"):
        self.params = params
        self.n_features = n_features
        self.target_type = target_type
        self.device = device
        # 시드는 모델이 정하지 않는다(절대 규칙 3). 엔진이 학습 직전에 전역 난수(random·numpy·torch)를 이 폴드·이 반복의 시드로 맞춘다
        # (engine.experiment.fold_seed). 자체 난수기를 가진 라이브러리(LightGBM)는 그 numpy 전역 난수에서 시드를 하나 뽑아 쓴다.
        # 학습 진단(조기 종료 지점 등). 모델이 채우면 엔진이 폴드 기록·결과 요약에 남긴다. 계산에는 쓰지 않는다.
        self.diag: dict | None = None

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


def fit_multi_start(mod, *, starts: str = "multi", maxiter: int = 200):
    """statsmodels 상태공간 모형(SARIMAX)을 여러 출발점에서 최우추정해 로그우도가 가장 높은 해를 고른다. 반환 (결과, 진단 dict).

    왜(2026-11-10): ARMA 가능도에는 극대점이 여러 개 있을 수 있고, 기본 출발점 하나에서 L-BFGS 가 '수렴했다'고 해도 최댓값이 아닐 수 있다.
    F11 월간(학습 창 60) 폴드 30개 표본에서: ARIMA(2,0,2) 는 10개 폴드에서 다른 출발점의 로그우도가 0.01 이상 높았고(최대 +0.99),
    SARIMA(1,0,1)(1,0,1,12) 는 8개(최대 +1.82). 그때 1기간 예측은 평균 0.00025·0.00007 달랐다(변화율 표준편차 0.0049).
    최우추정량의 정의는 '가능도가 가장 큰 모수'이므로, 여러 출발점 중 가장 높은 것을 쓰는 것이 정의에 맞다 — 그래도 전역 최댓값이라는 보장은 아니다.

    출발점(모두 정해진 값 — 난수 없음, 같은 자료면 같은 결과)
      default  statsmodels 기본 출발점(조건부 제곱합 추정)
      zero     AR·MA 계수 전부 0 (상수·분산만)
      ar+ ar-  첫 AR 계수 ±0.5 (계절 AR ±0.3), 나머지 0
      ma+ ma-  첫 MA 계수 ±0.5 (계절 MA ±0.3), 나머지 0
      nm       기본 출발점에서 Nelder–Mead(도함수 없이 탐색) → 그 해에서 다시 L-BFGS
    starts="default" 면 기본 출발점 하나만(2026-11-10 전과 같다).
    고르는 규칙: 로그우도가 유한한 해 가운데 최대. 단 앞의 해보다 1e−4 넘게 높을 때만 바꾼다(default 우선) — 사실상 같은 최대점에서 출발점만 다른 해로
    바뀌어 예측이 소수 끝자리에서 흔들리는 것을 막는다(컴퓨터·라이브러리 판에 따라 어느 해가 '아주 조금' 높은지가 달라질 수 있다)."""
    import numpy as _np

    names = list(mod.param_names)
    base = _np.asarray(mod.start_params, dtype=float)

    def vec(ar, ma, sar, sma):
        v = base.copy()
        for i, n in enumerate(names):
            if n.startswith("ar.S."):
                v[i] = sar if n.split(".L")[-1] == str(getattr(mod, "seasonal_periods", 0)) else 0.0
            elif n.startswith("ma.S."):
                v[i] = sma if n.split(".L")[-1] == str(getattr(mod, "seasonal_periods", 0)) else 0.0
            elif n.startswith("ar.L"):
                v[i] = ar if n == "ar.L1" else 0.0
            elif n.startswith("ma.L"):
                v[i] = ma if n == "ma.L1" else 0.0
        return v

    plan = [("default", None)]
    if starts == "multi":
        plan += [("zero", vec(0, 0, 0, 0)), ("ar+", vec(0.5, 0, 0.3, 0)), ("ar-", vec(-0.5, 0, -0.3, 0)), ("ma+", vec(0, 0.5, 0, 0.3)), ("ma-", vec(0, -0.5, 0, -0.3)), ("nm", "nm")]
    tried = []
    best = None
    for nm, sp in plan:
        try:
            if isinstance(sp, str):  # Nelder–Mead 뒤 L-BFGS 로 다듬기
                r0 = mod.fit(method="nm", disp=False, maxiter=2000)
                res = mod.fit(start_params=r0.params, disp=False, maxiter=maxiter)
            else:
                res = mod.fit(start_params=sp, disp=False, maxiter=maxiter)
            llf = float(res.llf)
        except Exception:  # noqa: BLE001 — 한 출발점이 실패해도 나머지로 계속
            tried.append({"start": nm, "llf": None})
            continue
        ok = bool(_np.isfinite(llf)) and bool(_np.all(_np.isfinite(_np.asarray(res.params, dtype=float))))
        tried.append({"start": nm, "llf": llf if ok else None, "converged": bool((getattr(res, "mle_retvals", None) or {}).get("converged", False))})
        if ok and (best is None or llf > best[1] + 1e-4):
            best = (nm, llf, res)
    if best is None:
        raise RuntimeError("최우추정이 모든 출발점에서 실패했습니다")
    d = next((t["llf"] for t in tried if t["start"] == "default"), None)
    diag = {"start_used": best[0], "llf": best[1], "llf_default": d, "llf_gain": (best[1] - d) if d is not None else None,
            "n_starts": len(plan), "converged": bool((getattr(best[2], "mle_retvals", None) or {}).get("converged", False)), "tried": tried}
    return best[2], diag
