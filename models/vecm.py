"""VECM — 벡터 오차수정모형(Johansen). INPUT_SHAPE='levels'.

여러 수준값 시계열이 장기적으로 함께 움직이는(공적분) 관계를 쓴다:
    Δy_t = α(β'y_{t−1} + c_in) + Σ_{i=1..p} Γ_i Δy_{t−i} + c_out + Σ_s δ_s·계절_s,t + ε_t
β'y = 장기 균형식, α = 균형에서 벗어났을 때 되돌아오는 속도, Γ = 단기 동학. y = [대상, 변수1, 변수2 …] 수준값.

기본값 = 노트북 2(시계열기초강좌_기획) 의 `VECM(훈련, k_ar_diff=2, coint_rank=1)` — statsmodels 기본이라 상수항 없음(n), 원래 수준값.
  노트북은 학습 끝에서 12개월을 한 번에 예측했고, 플랫폼은 테스트 시점 t 마다 t 까지의 값을 넣어 k 앞을 예측한다(재적합 없음).

엔진 규약: 폴드마다 학습 창의 수준값으로 적합(fit_levels)하고, 테스트 시점 t 마다 **t 까지의 값만** 넣어
적합된 계수로 k 기간 앞까지 재귀 예측한다. 재귀식은 statsmodels VECMResults.predict 와 1e-13 수준으로 같음을 시험으로 확인
(상수 n/ci/co × 계절 0/12).
- coint_rank=0: 학습 창마다 Johansen 검정(rank_test, rank_signif)으로 고른다. 0 이 나오면 공적분이 없으므로 차분 VAR(같은 p)로 대체.
- lag_select≠fixed: 학습 창마다 정보기준으로 p 를 0..k_ar_diff 에서 고른다.
- 학습 창에서 값이 하나도 안 바뀐 변수 열은 그 폴드에서 뺀다(행렬이 특이해져 적합 불가). 대상 열이 그러면 오류.
- 계수가 표본에 비해 너무 많으면(자유도 < 10) 오류로 멈춘다 — 엉뚱한 예측을 조용히 내지 않게.
- 변수의 '변환' 설정은 무시된다(수준값이 필요). log_levels 면 학습 창에서 모두 양수인 열만 로그.
진단(폴드별 차수·시차·뺀 열·단위근 검정)은 self.diag 에 남기고 엔진이 결과에 기록한다.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from models.base import LevelsModel

log = logging.getLogger("tsp.model.vecm")

NAME = "VECM"
INPUT_SHAPE = "levels"
TARGET_TYPES = ["regression"]
PARAMS = {
    "k_ar_diff": {"default_by_freq": {"D": 5, "W": 4, "M": 2}, "type": "int", "default": 2, "min": 0, "max": 24, "group": "단기 동학",
                  "desc": "차분 시차 수 p — 지난 몇 기간의 변화(Δy)를 볼지. lag_select 가 fixed 가 아니면 최대값. 노트북 2 = 2"},
    "lag_select": {"type": "str", "default": "fixed", "choices": ["fixed", "aic", "bic", "hqic"], "group": "단기 동학",
                   "desc": "p 를 고르는 방법. fixed: k_ar_diff 그대로(노트북). aic·bic·hqic: 학습 창마다 정보기준으로 0..k_ar_diff 중 선택(bic 가 가장 보수적)"},
    "coint_rank": {"type": "int", "default": 1, "min": 0, "max": 10, "group": "장기 균형",
                   "desc": "공적분 관계(장기 균형식) 개수 r. 0 이면 학습 창마다 Johansen 검정으로 자동 선택. 노트북 2 = 1"},
    "rank_test": {"type": "str", "default": "trace", "choices": ["trace", "maxeig"], "group": "장기 균형",
                  "desc": "coint_rank=0 일 때 Johansen 검정 종류. trace(대각합, 일반적) / maxeig(최대고유값)"},
    "rank_signif": {"type": "float", "default": 0.05, "choices": [0.01, 0.05, 0.1], "group": "장기 균형",
                    "desc": "coint_rank=0 일 때 검정 유의수준(statsmodels 는 1·5·10% 만 지원)"},
    "deterministic": {"type": "str", "default": "n", "choices": ["n", "ci", "co"], "group": "상수·계절",
                      "desc": "상수항. n: 없음(노트북 2 = statsmodels 기본). ci: 장기 균형식 안(수준이 일정한 값 주위로 균형). co: 균형식 밖(수준에 꾸준한 추세 = 집값처럼 늘 오르는 시계열에 흔히 씀)"},
    "seasons": {"type": "int", "default": 0, "choices": [0, 4, 12, 52], "group": "상수·계절",
                "desc": "계절 더미 주기. 0: 없음(노트북). 12: 월간 데이터의 달마다 다른 평균 변화(이사철 등). 4: 분기, 52: 주간"},
    "log_levels": {"type": "bool", "default": False, "group": "데이터",
                   "desc": "양수 열을 로그 수준값으로 적합(비율 관계를 선형으로). 노트북 2 = 아니오(원래 수준값)"},
}
PARAM_GROUPS = ["장기 균형", "단기 동학", "상수·계절", "데이터"]
_DET_ORDER = {"n": -1, "ci": 0, "co": 0}  # Johansen 검정의 결정항: −1 없음 / 0 상수. ci(제약 상수)는 statsmodels 검정에 없어 0 으로 근사


class Model(LevelsModel):
    def fit_levels(self, Y: np.ndarray) -> None:
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, select_order

        p = self.params
        Y = np.asarray(Y, dtype=float)
        T, K0 = Y.shape
        const = [j for j in range(K0) if np.nanmax(Y[:, j]) == np.nanmin(Y[:, j])]
        if 0 in const:
            raise ValueError("VECM: 학습 창에서 대상 값이 한 번도 바뀌지 않았습니다")
        self.keep = np.array([j for j in range(K0) if j not in const])
        if len(self.keep) < 2:
            raise ValueError(f"VECM: 값이 바뀌는 변수가 없습니다(학습 창에서 상수인 열 {len(const)}개 제외 후 대상만 남음)")
        Y = Y[:, self.keep]
        K = Y.shape[1]
        self._log = np.array([bool(p["log_levels"]) and bool(np.all(Y[:, j] > 0)) for j in range(K)])
        Z = self._to_model(Y)
        self.det, self.seasons = p["deterministic"], int(p["seasons"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.p = int(p["k_ar_diff"])
            if p["lag_select"] != "fixed" and self.p > 0:
                o = select_order(Z, maxlags=self.p, deterministic=self.det, seasons=self.seasons)
                self.p = int(getattr(o, p["lag_select"]))
            n_det = (1 if self.det in ("co", "ci") else 0) + max(self.seasons - 1, 0)
            dof = (T - self.p - 1) - (K * self.p + K + n_det)  # 식 하나당 (관측 수 − 계수 수), r ≤ K 로 넉넉히
            if dof < 10:
                raise ValueError(f"VECM: 계수가 표본에 비해 너무 많습니다 — 학습 {T}행, 변수 {K}개, p={self.p}, 결정항 {n_det}개 → 자유도 {dof} (10 이상 필요). "
                                 "변수 수나 k_ar_diff·seasons 를 줄이거나 학습 창을 늘리세요")
            r = int(p["coint_rank"])
            if r == 0:
                r = int(select_coint_rank(Z, det_order=_DET_ORDER[self.det], k_ar_diff=self.p, method=p["rank_test"], signif=float(p["rank_signif"])).rank)
            r = min(r, K - 1)
            self.rank = r
            if r >= 1:
                res = VECM(Z, k_ar_diff=self.p, coint_rank=r, deterministic=self.det, seasons=self.seasons).fit()
                self.alpha, self.beta, self.gamma = res.alpha, res.beta, res.gamma
                self.c_in = res.det_coef_coint.ravel() if self.det == "ci" else None
                self.det_out = res.det_coef if res.det_coef.size else None  # 열: [상수(co)] + [계절 s−1]
            else:  # 공적분 없음 → 차분 VAR(p). ci(상수가 균형식 안)는 균형식이 없으니 상수도 없다
                from statsmodels.tsa.api import VAR

                dZ = np.diff(Z, axis=0)
                self.alpha = np.zeros((K, 1)); self.beta = np.zeros((K, 1)); self.c_in = None
                if self.p == 0:
                    mu = dZ.mean(axis=0) if self.det == "co" else np.zeros(K)
                    self.gamma = np.zeros((K, 0)); self.det_out = mu[:, None] if self.det == "co" else None
                else:
                    vr = VAR(dZ).fit(self.p, trend="c" if self.det == "co" else "n")
                    self.gamma = np.hstack(list(vr.coefs))
                    self.det_out = vr.intercept[:, None] if self.det == "co" else None
                if self.seasons:
                    log.warning("VECM: 공적분 차수 0 → 차분 VAR 대체 시 계절 더미는 쓰지 않음")
        self.T_fit = T
        unit = []
        for j in range(K):  # 단위근 진단(정보용): 수준이 이미 정상(I(0))이면 공적분 차수를 부풀린다
            try:
                unit.append(round(float(adfuller(Z[:, j], autolag="AIC")[1]), 4))
            except Exception:  # noqa: BLE001
                unit.append(None)
        self.diag = {"rank": self.rank, "p": self.p, "dropped_constant_cols": [int(j) for j in const], "adf_p_levels": unit,
                     "n_train": T, "n_vars": K}
        log.debug("VECM fit: 학습 %d행 · 변수 %d(상수 열 제외 %d) · p=%d · r=%d · 결정항 %s · 계절 %d · ADF p(수준)=%s",
                 T, K, len(const), self.p, self.rank, self.det, self.seasons, unit)

    def _to_model(self, Y):
        Z = np.array(Y, dtype=float, copy=True)
        Z[:, self._log] = np.log(Z[:, self._log])
        return Z

    def forecast_levels(self, history: np.ndarray, steps: int, pos: int | None = None) -> np.ndarray:
        """history 의 마지막 행까지의 값으로 steps 기간 앞까지의 대상 수준값 경로.
        pos: history 마지막 행이 적합 표본(학습 창 첫 행 = 0)에서 몇 번째 행인지 — 계절 더미의 위상에 필요."""
        from statsmodels.tsa.vector_ar.util import seasonal_dummies

        H = np.asarray(history, dtype=float)[:, self.keep]
        if np.any(H[:, self._log] <= 0):
            log.warning("VECM: 로그를 쓰는 열의 이력에 0 이하 값 → 이 시점 예측 불가(NaN)")
            return np.full(steps, np.nan)
        Z = self._to_model(H)
        K, p = Z.shape[1], self.p
        h = [row for row in Z[-(p + 1):]]
        if self.seasons and pos is None:
            raise ValueError("VECM: seasons>0 이면 pos(학습 시작부터의 행 번호)가 필요합니다")
        S = seasonal_dummies(self.seasons, pos + steps + 1, centered=True) if self.seasons else None
        out = []
        for st in range(1, steps + 1):
            dy = [h[-i] - h[-i - 1] for i in range(1, p + 1)]
            ect = self.beta.T @ h[-1]
            if self.c_in is not None:
                ect = ect + self.c_in
            d = self.alpha @ ect + sum((self.gamma[:, i * K:(i + 1) * K] @ dy[i] for i in range(p)), np.zeros(K))
            if self.det_out is not None:
                vec = ([1.0] if self.det == "co" else []) + (list(S[pos + st]) if (S is not None and self.rank >= 1) else [])
                d = d + self.det_out[:, :len(vec)] @ np.asarray(vec)
            nxt = h[-1] + d
            h.append(nxt)
            out.append(nxt[0])
        out = np.array(out)
        return np.exp(out) if self._log[0] else out
