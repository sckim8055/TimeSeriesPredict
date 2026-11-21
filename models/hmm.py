"""숨은 마르코프 모형(HMM, 가우스 방출) — 단변량 통계 모델. INPUT_SHAPE='series' (ARIMA 와 같은 자리). 2026-11-15.

생각: 시장에는 눈에 보이지 않는 '상태'(예: 잔잔한 상승기 / 출렁이는 하락기)가 있고, 상태마다 하루 변화율의 평균·흩어짐이 다르며,
상태는 마르코프 연쇄로 바뀐다(오늘 상태만 보고 내일 상태 확률이 정해짐). 관찰되는 것은 변화율뿐이라 상태는 확률로만 추정한다.
  상태 s_t ∈ {0..K−1},  P(s_t = j | s_{t−1} = i) = A[i, j],   r_t | s_t = j  ~  정규분포(μ_j, σ_j²)
학습: Baum–Welch(EM) — 레너드 바움 등(1970)의 방법. 학습 창의 변화율만 쓴다(엔진이 폴드마다 새로 학습).
  E 단계: 앞방향(α)·뒷방향(β) 확률로 각 날의 상태 확률 γ_t(j) 와 이웃한 두 날의 상태 쌍 확률 ξ_t(i,j)
  M 단계: π = γ_0,  A[i,j] = Σξ_t(i,j)/Σγ_t(i),  μ_j = Σγ_t(j) r_t / Σγ_t(j),  σ_j² = Σγ_t(j)(r_t − μ_j)² / Σγ_t(j)
  로그우도가 tol 보다 덜 늘거나 max_iter 에 닿으면 멈춤. EM 은 로그우도를 줄이지 않는다(시험으로 확인).
예측(누출 없음): 결정일 t 까지의 변화율로 **앞방향만**(필터링) 계산한 상태 확률 p_t(j) = P(s_t = j | r_1..r_t).
  뒷방향(β)·비터비처럼 t 이후 자료를 쓰는 계산은 예측에 쓰지 않는다.
  h 기간 앞 상태 확률 = p_t · A^h,  h 기간 앞 변화율 예측 = (p_t · A^h) · μ  →  엔진이 k 기간을 복리로 묶음.
결정적: 난수를 쓰지 않는다. 출발점 2개(① 평균을 분위수로 나눔 ② 흩어짐으로 나눔) 중 로그우도가 높은 해. 상태 번호는 평균이 작은 순.
수치: 변화율에 scale(기본 100, ARIMA 와 같은 이유)을 곱해 추정하고 예측을 다시 나눈다. 분산 바닥 = 학습 분산 × 1e−4(한 상태가 한 점에 붙는 퇴화 막음).
외생변수는 쓰지 않는다 — 입력 변수를 골라도 무시된다(ARIMA 와 같음).
"""
from __future__ import annotations

import numpy as np

from models.base import SeriesModel

NAME = "HMM 시장 상태"
INPUT_SHAPE = "series"
TARGET_TYPES = ["regression"]
PARAMS = {
    "n_states": {"type": "int", "default": 2, "min": 2, "max": 3, "desc": "숨은 상태 수 — 2: 잔잔한 때/출렁이는 때, 3: 상승·횡보·하락처럼 셋으로"},
    "max_iter": {"type": "int", "default": 200, "min": 10, "max": 2000, "desc": "Baum–Welch 반복 최대 횟수"},
    "tol": {"type": "float", "default": 1e-6, "min": 1e-10, "max": 1e-2, "desc": "멈춤 기준 — 한 번 반복에서 로그우도가 이것보다 덜 늘면 멈춤"},
    "scale": {"type": "float", "default": 100.0, "min": 1.0, "max": 1e6, "desc": "추정용 배율 — 변화율에 곱해 추정하고 예측을 다시 나눔(모형은 같고 수치 조건만 좋아짐)"},
}

_LOG2PI = np.log(2 * np.pi)


def _emis(r: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """정규분포 밀도 b_t(j) [n, K]."""
    return np.exp(-0.5 * (_LOG2PI + np.log(var)[None, :] + (r[:, None] - mu[None, :]) ** 2 / var[None, :]))


def forward(r: np.ndarray, pi: np.ndarray, A: np.ndarray, mu: np.ndarray, var: np.ndarray, alpha0: np.ndarray | None = None):
    """앞방향(척도 조정): 반환 (α̂ [n,K] — 각 행 = P(s_t | r_1..r_t), c [n] — 척도, 로그우도 = Σ log c).
    alpha0 를 주면 그 바로 다음 날부터 이어 계산(앞 계산의 마지막 필터 확률)."""
    b = _emis(r, mu, var)
    n, K = b.shape
    a = np.empty((n, K)); c = np.empty(n)
    prev = None if alpha0 is None else alpha0
    for t in range(n):
        x = (pi if prev is None else prev @ A) * b[t]
        s = x.sum()
        if not np.isfinite(s) or s <= 0:          # 모든 상태에서 밀도가 0 으로 내려간 극단값 — 예측 확률로 대신(로그우도는 아주 작게)
            x, s = (pi if prev is None else prev @ A).copy(), 1e-300
            c[t] = s; a[t] = x / x.sum(); prev = a[t]; continue
        c[t] = s; a[t] = x / s; prev = a[t]
    return a, c


def baum_welch(r: np.ndarray, pi, A, mu, var, max_iter: int, tol: float, var_floor: float):
    """EM. 반환 (pi, A, mu, var, 로그우도, 반복 수, 수렴 여부, 로그우도 기록)."""
    n, K = len(r), len(mu)
    hist = []
    ll_old = -np.inf
    it, conv = 0, False
    for it in range(1, max_iter + 1):
        b = _emis(r, mu, var)
        a, c = forward(r, pi, A, mu, var)
        c = np.maximum(c, 1e-300)
        ll = float(np.sum(np.log(c)))
        hist.append(ll)
        be = np.empty((n, K)); be[-1] = 1.0
        for t in range(n - 2, -1, -1):
            be[t] = (A @ (b[t + 1] * be[t + 1])) / c[t + 1]
        g = a * be
        g /= g.sum(axis=1, keepdims=True)
        xi = (a[:-1, :, None] * A[None, :, :] * (b[1:] * be[1:])[:, None, :]) / c[1:, None, None]
        pi = g[0] / g[0].sum()
        A = xi.sum(axis=0) / np.maximum(g[:-1].sum(axis=0)[:, None], 1e-300)
        A /= A.sum(axis=1, keepdims=True)
        w = g.sum(axis=0)
        mu = (g * r[:, None]).sum(axis=0) / np.maximum(w, 1e-300)
        var = np.maximum((g * (r[:, None] - mu[None, :]) ** 2).sum(axis=0) / np.maximum(w, 1e-300), var_floor)
        if ll - ll_old < tol:
            conv = True
            break
        ll_old = ll
    # 마지막 모수의 로그우도(위 기록은 그 반복의 '갱신 전' 모수 값)
    _, c = forward(r, pi, A, mu, var)
    ll = float(np.sum(np.log(np.maximum(c, 1e-300))))
    return pi, A, mu, var, ll, it, conv, hist


def _starts(r: np.ndarray, K: int):
    v = float(np.var(r))
    out = []
    qs = np.quantile(r, (np.arange(K) + 0.5) / K)                 # ① 평균을 분위수로
    out.append(("평균 분위수", qs, np.full(K, v)))
    out.append(("흩어짐 나눔", np.full(K, float(np.mean(r))), v * np.geomspace(0.5, 2.0, K)))   # ② 같은 평균, 흩어짐 작게·크게
    return out


class Model(SeriesModel):
    def fit_series(self, r: np.ndarray) -> None:
        p = self.params
        K = int(p["n_states"])
        self.scale = float(p.get("scale", 1.0) or 1.0)
        x = np.asarray(r, dtype=float) * self.scale
        floor = float(np.var(x)) * 1e-4
        best = None
        tried = []
        for name, mu0, var0 in _starts(x, K):
            A0 = np.full((K, K), 0.05 / (K - 1)); np.fill_diagonal(A0, 0.95)
            res = baum_welch(x, np.full(K, 1.0 / K), A0, mu0.copy(), var0.copy(), int(p["max_iter"]), float(p["tol"]), floor)
            tried.append({"start": name, "loglik": res[4], "iters": res[5], "converged": res[6]})
            if best is None or res[4] > best[1][4]:
                best = (name, res)
        name, (pi, A, mu, var, ll, it, conv, hist) = best
        o = np.argsort(mu)                                            # 상태 번호 = 평균이 작은 순(폴드 사이 이름이 흔들리지 않게)
        self.pi, self.A, self.mu, self.var = pi[o], A[np.ix_(o, o)], mu[o], var[o]
        self.loglik_history = hist
        self.diag = {"start": name, "loglik": ll, "iters": it, "converged": conv, "tried": tried,
                     "mean_per_period": [float(m / self.scale) for m in self.mu],
                     "vol_per_period": [float(np.sqrt(v) / self.scale) for v in self.var],
                     "stay_prob": [float(self.A[j, j]) for j in range(K)]}
        self._cache = None

    def filtered(self, history: np.ndarray) -> np.ndarray:
        """history 끝 날의 상태 확률 P(s_T | r_1..r_T) — 앞방향만. 같은 폴드 안에서 history 가 늘어나기만 하면 앞 계산을 이어 씀(결과는 같음)."""
        x = np.asarray(history, dtype=float) * self.scale
        x = x[np.isfinite(x)]
        if self._cache is not None:
            n0, a0, x0 = self._cache
            if len(x) >= n0 and np.array_equal(x[:n0], x0):
                if len(x) == n0:
                    return a0
                a, _ = forward(x[n0:], self.pi, self.A, self.mu, self.var, alpha0=a0)
                self._cache = (len(x), a[-1], x.copy())
                return a[-1]
        a, _ = forward(x, self.pi, self.A, self.mu, self.var)
        self._cache = (len(x), a[-1], x.copy())
        return a[-1]

    def forecast(self, history: np.ndarray, steps: int) -> np.ndarray:
        p = self.filtered(history)
        out = np.empty(steps)
        for h in range(steps):
            p = p @ self.A
            out[h] = float(p @ self.mu) / self.scale
        return out
