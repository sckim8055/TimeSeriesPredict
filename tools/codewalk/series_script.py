"""ARIMA·SARIMA 코드 따라가기 대본 — series_capture 의 실제 실행 기록으로 단계별 화면을 만든다.

  python -m tools.codewalk.series_script --model arima     (또는 --model sarima)

statsmodels SARIMAX 의 계산(상태공간·칼만 필터·최우추정)은 라이브러리 안이라 줄 단위 추적 대신, 실행 기록(반복마다의 모수·로그우도,
상태공간 행렬·초기값, 필터의 예측오차·분산)을 꺼내 **식대로 numpy 로 다시 계산**하고 실제 값과 맞는지 '검증' 줄로 보인다.
하나라도 맞지 않으면 대본을 만들지 않는다(AssertionError).
2026-11-08: 단계마다 say(음성·자막 대본 — 숫자는 화면과 같은 변수), notes(코드 줄 설명), lab(직접 실행해 보기 — 실제로 돌려 엔진 값과 맞춤) 추가.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from tools.codewalk.common import ROOT, Sources, arr, build_html, f, match_result, num, py_list, run_lab

EXP = "engine/experiment.py"
BASE = "models/base.py"


def pct(v, d=3, sign=True):
    return "NaN" if v is None or not np.isfinite(v) else (f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%")


def pp(v, d=4, say=False):
    """두 변화율의 차이(퍼센트포인트). 화면은 %p, 대본(say)은 '%포인트'(읽기: 퍼센트포인트)."""
    return "NaN" if v is None or not np.isfinite(v) else f"{v * 100:+.{d}f}" + ("%포인트" if say else "%p")


def pi_weights(phi, theta, n):
    """실효 AR(∞) 무게 — 가역 ARMA 의 충격을 모두 과거 값으로 풀었을 때의 무게.
    (1 − Σ φ_i L^i) ÷ (1 + Σ θ_j L^j) = 1 − Σ π_i L^i 의 π_1 … π_n. 한 달 앞 예측 = c ÷ (1 + Σθ) + Σ π_i r_(t+1−i)."""
    phi = np.asarray(phi, dtype=float); theta = np.asarray(theta, dtype=float); n = max(int(n), len(phi))
    a = np.zeros(n + 1); a[0] = 1.0; a[1:len(phi) + 1] = -phi
    qq = np.zeros(n + 1)
    for i in range(n + 1):
        qq[i] = a[i] - sum(theta[j - 1] * qq[i - j] for j in range(1, min(i, len(theta)) + 1))
    return -qq[1:]


def css(endog, k_ar, ar_cols, k_ma, ma_cols, k_trend):
    """statsmodels SARIMAX._conditional_sum_squares 와 같은 절차(시작값): ① MA 가 있으면 2q차 AR 최소제곱으로 잔차(충격 추정)를 만들고
    ② [상수, 과거 값, 과거 잔차] 에 한 번에 최소제곱 회귀. 분산 시작값 = 그 회귀 잔차 중 앞 (MA 모수 수)개를 뺀 나머지 제곱의 평균."""
    from statsmodels.tsa.tsatools import lagmat
    k = 2 * k_ma; r = max(k + k_ma, k_ar); resid = None; info1 = {}
    if k_ma > 0:
        Y = endog[k:]; X = lagmat(endog, k, trim="both"); resid = Y - X @ (np.linalg.pinv(X) @ Y)
        info1 = {"rows1": X.shape[0], "cols1": X.shape[1], "res1max": float(np.max(np.abs(resid)))}
    Y = endog[r:]; X = np.empty((len(Y), 0))
    if k_trend: X = np.c_[X, np.ones(len(Y))]
    if k_ar: X = np.c_[X, lagmat(endog, k_ar)[r:, ar_cols]]
    if k_ma: X = np.c_[X, lagmat(resid, k_ma)[r - k:, ma_cols]]
    prm = np.linalg.pinv(X) @ Y; res = Y - X @ prm
    drop = len(ma_cols) if k_ma else 0
    return prm, float((res[drop:] ** 2).mean()), {"k": k, "r": r, "rows": len(Y), "cols": X.shape[1], "drop": drop, "n_var": len(res) - drop, **info1}


def stable(poly):
    poly = np.asarray(poly, dtype=float)
    return bool(np.all(np.abs(np.roots(poly[::-1])) > 1)) if len(poly) > 1 and np.any(poly[1:]) else True


def csu(u):
    """statsmodels constrain_stationary_univariate 와 같은 식(Monahan 1984 — statsmodels 소스가 인용): 제약 없는 u → (−1,1) 의 ρ(= −편자기상관) → 재귀 → 정상 AR 계수."""
    u = np.asarray(u, dtype=float); n = len(u)
    r = u / np.sqrt(1 + u ** 2)
    y = np.zeros((n, n))
    for k in range(n):
        for i in range(k):
            y[k, i] = y[k - 1, i] + r[k] * y[k - 1, k - i - 1]
        y[k, k] = r[k]
    return -y[n - 1, :], r


def kfilter(kf):
    """교과서 칼만 필터(시불변 행렬): v_t = y_t − Z a − d,  F_t = Z P Z' + H,  K = T P Z'/F,  a ← T a + c + K v,  P ← T P T' + R Q R' − K F K'."""
    Z, d, H, T, c, R, Q = (np.asarray(kf[k], dtype=float) for k in ("Z", "d", "H", "T", "c", "R", "Q"))
    a = np.asarray(kf["a0"], dtype=float).copy(); P = np.asarray(kf["P0"], dtype=float).copy()
    RQR = R @ Q @ R.T
    out = {"v": [], "F": [], "yhat": [], "K": [], "a": [], "P": []}
    for yt in kf["y"]:
        yh = float((Z @ a)[0] + d[0]); Ft = float((Z @ P @ Z.T)[0, 0] + H[0, 0]); vt = float(yt - yh)
        K = (T @ P @ Z.T)[:, 0] / Ft
        out["a"].append(a.copy()); out["P"].append(P.copy()); out["yhat"].append(yh); out["v"].append(vt); out["F"].append(Ft); out["K"].append(K.copy())
        a = T @ a + c + K * vt
        P = T @ P @ T.T + RQR - np.outer(K, K) * Ft
    return {k: np.asarray(v) for k, v in out.items()}


# ---------------------------------------------------------------- 직접 실행해 보기(실험 코드) — 2026-11-08
# 실행 기록의 숫자를 그대로 넣은, 혼자 도는 파이썬 코드. 대본을 만들 때 실제로 실행해 출력이 엔진(statsmodels) 값과 같은지 확인한다(common.run_lab).
# 출력은 유효숫자 5자리 — 보이는 코드를 그대로 돌린 출력이 기대 출력과 글자까지 같아야 한다(허용오차 없음).
def _g5(v) -> str:
    return f"{float(v):.5g}"


def _pl(vals, sig=17):
    return py_list([float(v) for v in vals], sig)


def _mat(M, sig=17):
    return "[\n" + "".join(f"    {_pl(r_, sig)},\n" for r_ in np.asarray(M, dtype=float)) + "]"


def _labels(names):
    """statsmodels 이름 → 실습 출력 이름(화면 표의 '뜻' 과 같은 말)."""
    out = []
    for n_ in names:
        if n_ == "intercept": out.append("상수 c")
        elif n_ == "sigma2": out.append("σ²")
        elif n_.startswith("ar.S.L"): out.append("계절 AR Φ1")
        elif n_.startswith("ma.S.L"): out.append("계절 MA Θ1")
        elif n_.startswith("ar.L"): out.append(f"AR φ{n_[4:]}")
        elif n_.startswith("ma.L"): out.append(f"MA θ{n_[4:]}")
        else: out.append(n_)
    return out


def lab_constrain(names, u, sc, est_e):
    """u(제약 없는 값) → 계수. 기대 출력 = statsmodels 추정값(원래 단위)."""
    g = {"intercept": [], "ar": [], "ma": [], "sar": [], "sma": [], "sigma2": []}
    for n_, x in zip(names, u):
        key = "intercept" if n_ == "intercept" else "sigma2" if n_ == "sigma2" else "sar" if n_.startswith("ar.S.L") else "sma" if n_.startswith("ma.S.L") else "ar" if n_.startswith("ar.L") else "ma"
        g[key].append(float(x))
    lines = ["import math", "", f"# 마지막 폴드의 최종 추정값을 statsmodels 역변환(untransform_params)으로 되돌린 '제약 없는 값' u (배율 {sc:g} 단위) — 실행 기록 그대로"]
    if g["intercept"]: lines.append(f"u_c = {g['intercept'][0]!r}            # 상수")
    lines.append(f"u_ar = {_pl(g['ar'])}     # AR")
    lines.append(f"u_ma = {_pl(g['ma'])}     # MA")
    if g["sar"]: lines.append(f"u_sar = {_pl(g['sar'])}    # 계절 AR (12개월)")
    if g["sma"]: lines.append(f"u_sma = {_pl(g['sma'])}    # 계절 MA (12개월)")
    lines.append(f"u_s2 = {g['sigma2'][0]!r}            # 오차 분산")
    lines.append(f"scale = {float(sc)!r}")
    lines += ["", "def to_g(u):",
              "    # u → ρ ∈ (−1, 1) → 재귀식 → 마지막 줄 g (Monahan 1984, statsmodels 와 같은 식)",
              "    n = len(u)",
              "    rho = [x / math.sqrt(1 + x * x) for x in u]",
              "    g = [[0.0] * n for _ in range(n)]",
              "    for j in range(n):",
              "        for i in range(j):",
              "            g[j][i] = g[j - 1][i] + rho[j] * g[j - 1][j - i - 1]",
              "        g[j][j] = rho[j]",
              "    return g[n - 1]", "",
              "phi = [-x for x in to_g(u_ar)]       # AR: φ = −g  → 언제나 정상",
              "theta = to_g(u_ma)                   # MA: θ = g (부호 반대) → 언제나 가역"]
    if g["sar"]: lines.append("Phi = [-x for x in to_g(u_sar)]      # 계절 AR 도 같은 변환")
    if g["sma"]: lines.append("Theta = to_g(u_sma)                  # 계절 MA 도 같은 변환")
    lines += ["sigma2 = u_s2 ** 2                   # σ² = u² → 언제나 0 이상", "",
              "# 원래 단위로: " + ("상수 ÷ 배율, " if g["intercept"] else "") + "σ² ÷ 배율² (φ·θ 는 배율과 무관)"]
    if g["intercept"]: lines.append('print("상수 c", f"{u_c / scale:.5g}")')
    lines += ["for i, x in enumerate(phi, 1):", '    print("AR φ" + str(i), f"{x:.5g}")',
              "for i, x in enumerate(theta, 1):", '    print("MA θ" + str(i), f"{x:.5g}")']
    if g["sar"]: lines += ["for i, x in enumerate(Phi, 1):", '    print("계절 AR Φ" + str(i), f"{x:.5g}")']
    if g["sma"]: lines += ["for i, x in enumerate(Theta, 1):", '    print("계절 MA Θ" + str(i), f"{x:.5g}")']
    lines.append('print("σ²", f"{sigma2 / scale ** 2:.5g}")')
    code = "\n".join(lines) + "\n"
    exp = "\n".join(f"{lb} {_g5(v)}" for lb, v in zip(_labels(names), est_e))
    unit_note = ("상수는 표 값 ÷ " + f"{sc:g}" + ", " if g["intercept"] else "") + f"σ² 는 표 값 ÷ {sc:g}²"
    return {"title": "제약 없는 u 를 계수로 바꾸기 (정상·가역 보장)",
            "desc": "L-BFGS 가 움직이는 값 u 를 statsmodels 와 같은 식으로 계수로 바꿉니다. 출력이 statsmodels 의 추정값(원래 단위)과 같아야 합니다. "
                    f"화면 표의 '계수' 는 배율 단위라, 원래 단위로는 {unit_note} 입니다(φ·θ 는 같음).",
            **run_lab(code, exp, pure=True)}


def lab_css_arima(y, p, q, sc, names, start_e):
    """ARIMA 시작값(조건부 제곱합 두 단계, numpy). 기대 출력 = statsmodels start_params(원래 단위)."""
    code = ("import numpy as np\n\n"
            f"# 마지막 폴드의 학습 변화율 × {sc:g} ({len(y)}개월) — 실행 기록 그대로\n"
            f"y = np.array({_pl(y)})\n"
            f"p, q = {p}, {q}\n"
            f"scale = {float(sc)!r}\n\n"
            "def lags(x, k, start):\n"
            "    # start 번째 값부터 한 행씩: [x_(t−1), x_(t−2), …, x_(t−k)]\n"
            "    return np.column_stack([x[start - i:len(x) - i] for i in range(1, k + 1)])\n\n"
            "# 1단계: 2q 차 AR 을 최소제곱으로 맞춰 잔차 e (충격의 추정). e[j] 는 y[k + j] 의 잔차\n"
            "k = 2 * q\n"
            "X1 = lags(y, k, k)\n"
            "e = y[k:] - X1 @ (np.linalg.pinv(X1) @ y[k:])\n\n"
            "# 2단계: [상수, y_(t−1), y_(t−2), e_(t−1), e_(t−2)] 에 한 번에 최소제곱 (앞 r0 개월은 과거가 모자라 뺌)\n"
            "r0 = max(k + q, p)\n"
            "Y = y[r0:]\n"
            "X = np.column_stack([np.ones(len(Y)), lags(y, p, r0), lags(e, q, r0 - k)])\n"
            "b = np.linalg.pinv(X) @ Y\n"
            "res = Y - X @ b\n"
            "s2 = np.mean(res[q:] ** 2)           # σ² 시작값: 앞 q 개 잔차는 빼고 제곱 평균\n"
            "# (statsmodels 는 시작값이 정상·가역이 아니면 그 부분을 0 으로 바꿉니다 — 이 폴드는 해당 없음)\n\n"
            "print(\"회귀 행\", len(Y))\n"
            "print(\"상수 c\", f\"{b[0] / scale:.5g}\")\n"
            "for i in range(p):\n"
            "    print(\"AR φ\" + str(i + 1), f\"{b[1 + i]:.5g}\")\n"
            "for j in range(q):\n"
            "    print(\"MA θ\" + str(j + 1), f\"{b[1 + p + j]:.5g}\")\n"
            "print(\"σ²\", f\"{s2 / scale ** 2:.5g}\")\n")
    exp = f"회귀 행 {len(y) - max(2 * q + q, p)}\n" + "\n".join(f"{lb} {_g5(v)}" for lb, v in zip(_labels(names), start_e))
    return {"title": "시작값(두 단계 최소제곱)을 직접 구하기 (numpy)",
            "desc": "statsmodels 가 L-BFGS 의 출발점을 구하는 두 단계 최소제곱(내부 함수 이름은 _conditional_sum_squares)을 그대로 옮겼습니다. 출력이 statsmodels 의 start_params(원래 단위)와 같아야 합니다. numpy 가 필요해 Colab 등에 붙여 실행하세요.",
            **run_lab(code, exp, pure=False)}


def lab_css_sarima(y, s, sc, start_e, rows1, cols1):
    """SARIMA(1,0,1)(1,1,1,s) 시작값(numpy). 기대 출력 = statsmodels start_params(원래 단위)."""
    code = ("import numpy as np\n\n"
            f"# 마지막 폴드의 학습 변화율 × {sc:g} ({len(y)}개월) — 실행 기록 그대로\n"
            f"y = np.array({_pl(y)})\n"
            f"s = {s}\n"
            f"scale = {float(sc)!r}\n"
            "x = y[s:] - y[:-s]                   # 계절 차분: 12개월 전과의 차이\n\n"
            "def lags(z, k, start):\n"
            "    # start 번째 값부터 한 행씩: [z_(t−1), …, z_(t−k)]\n"
            "    return np.column_stack([z[start - i:len(z) - i] for i in range(1, k + 1)])\n\n"
            "def css(z, k_ar, k_ma, lag):\n"
            "    # statsmodels 의 시작값 계산(내부 이름 '조건부 제곱합' — 실제는 두 단계 최소제곱). lag: 쓰는 시차 하나 — 비계절 1, 계절 12\n"
            "    k = 2 * k_ma\n"
            "    X1 = lags(z, k, k)                # 1단계: 2·k_ma 차 AR → 잔차 e\n"
            "    e = z[k:] - X1 @ (np.linalg.pinv(X1) @ z[k:])\n"
            "    r0 = max(k + k_ma, k_ar)\n"
            "    Y = z[r0:]\n"
            "    X = np.column_stack([lags(z, k_ar, r0)[:, lag - 1], lags(e, k_ma, r0 - k)[:, lag - 1]])   # 2단계: [z_(t−lag), e_(t−lag)]\n"
            "    b = np.linalg.pinv(X) @ Y\n"
            "    return b, Y - X @ b, X1.shape\n\n"
            "b1, res1, _ = css(x, 1, 1, 1)         # 비계절: x_(t−1), e_(t−1)\n"
            "b2, _, shape2 = css(x, s, s, s)       # 계절: x_(t−12), e_(t−12) — 1단계가 24차 AR\n"
            "s2 = np.mean(res1[1:] ** 2)           # σ² 시작값: 비계절 회귀 잔차(앞 1개 제외)의 제곱 평균\n"
            "# (statsmodels 는 시작값이 정상·가역이 아니면 그 부분을 0 으로 바꿉니다 — 이 폴드는 해당 없음)\n\n"
            "print(\"계절 차분한 자료\", len(x))\n"
            "print(\"계절 1단계 AR: 행\", shape2[0], \"· 계수\", shape2[1])\n"
            "print(\"AR φ1\", f\"{b1[0]:.5g}\")\n"
            "print(\"MA θ1\", f\"{b1[1]:.5g}\")\n"
            "print(\"계절 AR Φ1\", f\"{b2[0]:.5g}\")\n"
            "print(\"계절 MA Θ1 이 0 에 가까움\", abs(b2[1]) < 1e-10)\n"
            "print(\"σ²\", f\"{s2 / scale ** 2:.5g}\")\n")
    exp = (f"계절 차분한 자료 {len(y) - s}\n계절 1단계 AR: 행 {rows1} · 계수 {cols1}\nAR φ1 {_g5(start_e[0])}\nMA θ1 {_g5(start_e[1])}\n"
           f"계절 AR Φ1 {_g5(start_e[2])}\n계절 MA Θ1 이 0 에 가까움 {abs(float(start_e[3])) < 1e-10}\nσ² {_g5(start_e[4])}")
    return {"title": "시작값(두 단계 최소제곱)을 직접 구하기 (numpy)",
            "desc": "계절 차분한 자료로 statsmodels 의 두 단계 최소제곱을 비계절·계절 부분에 따로 합니다. 계절 1단계는 계수가 행보다 많아 잔차가 0 이 되고, 그래서 계절 MA 시작값이 0 입니다. 출력이 statsmodels 의 start_params 와 같아야 합니다(numpy — Colab 등에서).",
            **run_lab(code, exp, pure=False)}


def lab_kalman(kf, sc, burn, show, v_e, F_e, ll_e):
    """칼만 필터 + 로그우도(순수 파이썬). 기대 출력 = statsmodels forecasts_error·forecasts_error_cov·llf."""
    n = int(np.asarray(kf["T"]).shape[0])
    code = ("import math\n\n"
            f"# 학습 적합의 상태공간 행렬·초기값 (배율 {sc:g} 단위, 상태 {n}개) — statsmodels 가 만든 것 그대로\n"
            f"T = {_mat(kf['T'])}\n"
            f"c = {_pl(np.asarray(kf['c']).ravel())}\n"
            f"R = {_pl(np.asarray(kf['R'])[:, 0])}     # 충격이 들어가는 몫\n"
            f"Z = {_pl(np.asarray(kf['Z'])[0])}     # 관측에 쓰는 몫\n"
            f"s2 = {float(np.asarray(kf['Q'])[0, 0])!r}     # 충격 분산 σ² (배율 단위)\n"
            f"a = {_pl(kf['a0'])}     # 초기 상태 a₀\n"
            f"P = {_mat(kf['P0'])}     # 초기 분산 P₀\n"
            f"y = {_pl(kf['y'])}     # 관측: 학습 변화율 × {sc:g}\n"
            f"burn = {burn}            # 로그우도에서 뺄 처음 달 수\n"
            f"scale = {float(sc)!r}\n"
            f"show = {dict(show)!r}     # 출력할 달 (0 부터 센 번호: 월)\n"
            "n = len(a)\n\n"
            "ll = 0.0\n"
            "for t in range(len(y)):\n"
            "    PZ = [sum(P[i][j] * Z[j] for j in range(n)) for i in range(n)]          # P Z'\n"
            "    F = sum(Z[i] * PZ[i] for i in range(n))                                 # 예측오차 분산 F = Z P Z'\n"
            "    v = y[t] - sum(Z[i] * a[i] for i in range(n))                           # 예측오차 v = y − Z a\n"
            "    K = [sum(T[i][j] * PZ[j] for j in range(n)) / F for i in range(n)]      # 칼만 이득 K = T P Z' ÷ F\n"
            "    if t >= burn:\n"
            "        ll += -0.5 * (math.log(2 * math.pi * F) + v * v / F)                # 정규분포 로그밀도\n"
            "    if t in show:   # 배율 단위의 v 는 곧 % 값\n"
            "        print(\"t=\" + str(t + 1), show[t], \"v(%)\", f\"{v:.5g}\", \"F\", f\"{F / scale ** 2:.5g}\")\n"
            "    a = [sum(T[i][j] * a[j] for j in range(n)) + c[i] + K[i] * v for i in range(n)]               # a ← T a + c + K v\n"
            "    TP = [[sum(T[i][k] * P[k][j] for k in range(n)) for j in range(n)] for i in range(n)]\n"
            "    P = [[sum(TP[i][k] * T[j][k] for k in range(n)) + R[i] * R[j] * s2 - K[i] * K[j] * F for j in range(n)] for i in range(n)]   # P ← T P T' + R σ² R' − K F K'\n\n"
            "# 원래 단위의 로그우도 = 배율 단위 + (더한 달 수) × ln(배율)\n"
            "print(\"로그우도(원래 단위)\", f\"{ll + (len(y) - burn) * math.log(scale):.7g}\")\n")
    exp = "\n".join(f"t={t + 1} {m_} v(%) {_g5(v_e[t])} F {_g5(F_e[t])}" for t, m_ in show.items()) + f"\n로그우도(원래 단위) {float(ll_e):.7g}"
    return {"title": "칼만 필터로 로그우도를 직접 계산",
            "desc": f"statsmodels 가 만든 상태공간 행렬로 {len(kf['y'])}개월을 한 달씩 거르며 예측오차 v·분산 F 를 구하고 로그우도를 더합니다"
                    + (f"(처음 {burn}개월은 빼고)" if burn else "") + ". 출력이 statsmodels 의 값과 같아야 합니다. 반복이 많아 Python Tutor 대신 복사해서 실행하세요.",
            **run_lab(code, exp, pure=True, tutor=False)}


def lab_recursion(c, w, m, r_back, v_back, months, date, pred_e):
    """ARIMA 교과서 재귀식(ε 자리에 필터 예측오차). 기대 출력 = 엔진 예측."""
    code = (f"# 결정월 {date}: 추정한 계수(원래 단위)와 최근 변화율 r, 필터의 예측오차 v — 실행 기록 그대로\n"
            f"c = {float(c)!r}            # 상수\n"
            f"w = {_pl(w)}     # i달 전 변화율의 무게 (= φ1, φ2)\n"
            f"m = {_pl(m)}     # j달 전 충격의 무게 (= θ1, θ2)\n"
            f"r = {_pl(r_back)}     # {', '.join(months)} 의 변화율 (예측할 달에서 1달 전, 2달 전)\n"
            f"v = {_pl(v_back)}     # 같은 달의 예측오차 (관측되지 않는 충격 ε 대신)\n\n"
            "pred = c\n"
            "for i in range(len(w)):\n"
            "    pred += w[i] * r[i]\n"
            "for j in range(len(m)):\n"
            "    pred += m[j] * v[j]\n\n"
            "print(\"다음 달 변화율 예측(%)\", f\"{pred * 100:.5g}\")\n")
    return {"title": "교과서 재귀식으로 예측을 직접 계산",
            "desc": "상수 + 무게 × 지난 변화율 + 무게 × 지난 충격(필터의 예측오차로 대신)입니다. 이 폴드는 필터가 수렴해 엔진의 칼만 예측과 같아야 합니다.",
            **run_lab(code, f"다음 달 변화율 예측(%) {_g5(pred_e * 100)}", pure=True)}


def lab_ssforecast(T, a_last, c, Z, sc, n_hist, date, pred_e):
    """상태공간 한 달 앞 예측(순수 파이썬). 기대 출력 = 엔진 예측."""
    code = (f"# 결정월 {date}: apply 가 변화율 {n_hist}개를 거른 뒤의 마지막 상태 a (배율 {sc:g} 단위)와 행렬 — 실행 기록 그대로\n"
            f"a = {_pl(a_last)}\n"
            f"T = {_mat(T)}\n"
            f"c = {_pl(c)}\n"
            f"Z = {_pl(Z)}\n"
            f"scale = {float(sc)!r}\n"
            "n = len(a)\n\n"
            "a_next = [c[i] + sum(T[i][j] * a[j] for j in range(n)) for i in range(n)]   # 한 달 앞 상태 = T a + c\n"
            "pred = sum(Z[i] * a_next[i] for i in range(n)) / scale                       # 관측 = Z a, 배율로 나눔\n\n"
            "print(\"다음 달 변화율 예측(%)\", f\"{pred * 100:.5g}\")\n")
    return {"title": "마지막 상태를 한 달 앞으로 밀어 예측",
            "desc": f"상태 {len(a_last)}개를 T 로 한 달 앞으로 밀고 Z 로 관측값을 읽은 뒤 배율로 나눕니다. 출력이 엔진의 예측과 같아야 합니다.",
            **run_lab(code, f"다음 달 변화율 예측(%) {_g5(pred_e * 100)}", pure=True)}


def lab_rmse(KN, yy, pr, nv, e_model, e_naive, e_zero, src_note):
    """기대 출력 = 엔진이 결과 파일(json)의 prediction_metrics 에 저장한 RMSE(e_*) — 대본의 계산이 아니다."""
    code = (f"# {len(yy)}개 폴드의 실제 변화율, {KN} 예측, '그대로' 예측(결정월의 변화율) — 실행 기록 그대로\n"
            f"actual = {_pl(yy)}\n"
            f"model = {_pl(pr)}\n"
            f"same = {_pl(nv)}\n\n"
            "def rmse(pred):\n"
            "    s = 0.0\n"
            "    for p, a in zip(pred, actual):\n"
            "        s += (p - a) ** 2\n"
            "    return (s / len(actual)) ** 0.5\n\n"
            f"print(\"{KN}\", f\"{{rmse(model) * 100:.4f}}%\")\n"
            "print(\"그대로\", f\"{rmse(same) * 100:.4f}%\")\n"
            "print(\"0 예측\", f\"{rmse([0.0] * len(actual)) * 100:.4f}%\")\n")
    exp = f"{KN} {e_model * 100:.4f}%\n그대로 {e_naive * 100:.4f}%\n0 예측 {e_zero * 100:.4f}%"
    return {"title": "RMSE 를 직접 계산", "desc": f"오차를 제곱해 평균하고 제곱근을 씌웁니다. 모델·'그대로'·'항상 0' 을 같은 방법으로 비교합니다. 기대 출력은 {src_note} 입니다.",
            **run_lab(code, exp, pure=True)}


def lab_piweights(c, phi, theta, hist, date, pred_e):
    """ARMA 의 실효 무게 π 로 예측(순수 파이썬). 기대 출력 = 엔진 예측."""
    code = (f"# 결정월 {date}: 추정한 계수(원래 단위)와 결정월까지의 변화율 {len(hist)}개 — 실행 기록 그대로\n"
            f"c = {float(c)!r}            # 상수\n"
            f"phi = {_pl(phi)}     # AR φ1, φ2\n"
            f"theta = {_pl(theta)}     # MA θ1, θ2\n"
            f"r = {_pl(hist)}     # 오래된 달 → 결정월\n\n"
            "# (1 − φ1 L − φ2 L²) ÷ (1 + θ1 L + θ2 L²) 를 L 의 거듭제곱으로 펼친 계수 q (q[0] = 1)\n"
            "n = len(r)\n"
            "a = [1.0] + [-x for x in phi] + [0.0] * (n - len(phi))\n"
            "q = []\n"
            "for i in range(n + 1):\n"
            "    q.append(a[i] - sum(theta[j - 1] * q[i - j] for j in range(1, min(i, len(theta)) + 1)))\n"
            "w = [-x for x in q[1:]]                 # 실효 무게: i달 전 변화율에 거는 무게\n\n"
            "pred = c / (1 + sum(theta))             # 상수도 같은 나눗셈으로 바뀜\n"
            "for i in range(n):\n"
            "    pred += w[i] * r[n - 1 - i]         # i+1 달 전 변화율 × 무게\n\n"
            "print(\"실효 무게 1~3달 전\", f\"{w[0]:.5g}\", f\"{w[1]:.5g}\", f\"{w[2]:.5g}\")\n"
            "print(\"다음 달 변화율 예측(%)\", f\"{pred * 100:.5g}\")\n")
    from statsmodels.tsa.arima_process import arma2ar          # 기대 무게는 statsmodels 의 AR(∞) 변환으로(대본의 계산이 아님)
    w = -arma2ar(np.r_[1.0, -np.asarray(phi, dtype=float)], np.r_[1.0, np.asarray(theta, dtype=float)], 4)[1:]
    exp = f"실효 무게 1~3달 전 {_g5(w[0])} {_g5(w[1])} {_g5(w[2])}\n다음 달 변화율 예측(%) {_g5(pred_e * 100)}"
    return {"title": "충격을 과거 변화율로 풀어 예측 (실효 무게)",
            "desc": f"가역인 ARMA 는 충격을 과거 변화율의 합으로 풀 수 있습니다. 그 무게로 변화율 {len(hist)}개를 더하면 엔진(칼만 필터)의 예측과 같아야 합니다. "
                    "반복이 많아 Python Tutor 대신 복사해서 실행하세요.",
            **run_lab(code, exp, pure=True, tutor=False)}


# 대본(say·자막)·화면에 나오는 숫자 중 빌드 때 새로 계산하는 것(최적화 탐색·statsmodels·scipy 재계산)의 기대 표시값 — 2026-11-08 고정.
# 녹음은 대본 지문으로 묶이므로, 라이브러리 버전·시드가 바뀌어 이 값이 달라지면 조용히 대본이 바뀌지 않게 빌드를 멈춘다.
PINNED = {
    "arima": {
        "3장 예측 차이 최대(capture)": "0.485%p (2020-06)",
        "5장 넓은 탐색: 로그우도 차(대본)": "0.72", "5장 넓은 탐색: 로그우도 차(화면)": "+0.722",
        "5장 넓은 탐색: MA 근": "1.0003", "5장 넓은 탐색: AR 근": "1.0469", "5장 넓은 탐색: 예측": "+0.133%",
        "5장 넓은 탐색: φ·θ": "(1.898, -0.912) (-1.020, 0.020)", "5장 넓은 탐색: 도착 셈(최고·0.01초과·같음·낮음)": "2/3/17/10",
        "5장 120폴드: 0.01 넘게": "8", "5장 120폴드: 1e-06 넘게": "55", "5장 120폴드: 최대": "+0.098 2017-09", "5장 120폴드: RMSE": "0.3561% → 0.3572%",
        "7장 approx·robust 의 2 넘는 계수": "approx θ₁ +2.2873; robust θ₁ +2.2868",
        "8장 봉인 첫 달 뺀 RMSE·DM p": "0.3576% 0.3292% 0.285",
        "3장 같은 배율 1 모형으로 잰 폴드": "120",
    },
    "sarima": {
        "3장 예측 차이 최대(capture)": "1.775%p (2020-08)",
        "3장 산만 초기값 차이(대본)": "0.001", "3장 산만 초기값 차이(화면)": "-5.5e-05 / 1.0e-03",
        "5장 프로파일 끝(대본)": "0.0001", "5장 프로파일 끝(화면)": "+0.000116",
        "7장 Θ·σ² 비율(approx·oim·robust)": "-3.2490 -0.0209 -0.0200 +2.6729 +0.0212 +0.0202",
        "8장 봉인 첫 달 뺀 RMSE·DM p": "0.3868% 0.3292% 0.055",
        "3장 같은 배율 1 모형으로 잰 폴드": "120",
    },
}


def pin(kind: str, key: str, got: str) -> None:
    want = PINNED[kind][key]
    if want != got:
        raise RuntimeError(f"버전·시드에 따라 대본 숫자가 바뀜 — 대본·녹음 갱신 필요: [{kind}] {key}: 고정값 {want!r}, 지금 {got!r}")


def build(cap: dict) -> dict:
    kind = cap["model"]; KN = kind.upper(); seasonal = kind == "sarima"
    MODEL = f"models/{kind}.py"
    src = Sources([MODEL, EXP, BASE])
    p = cap["params"]; sc = cap["scale"]; names = cap["param_names"]; kf = cap["kf"]
    unit = np.array([1 / sc if n == "intercept" else 1 / sc ** 2 if n == "sigma2" else 1.0 for n in names])
    est = cap["est"] * unit; bse = cap["bse"] * unit
    b = cap["burn"]; n_eff = cap["nobs"] - b; lsc = n_eff * np.log(sc)   # 로그우도 단위 환산: 원래 = 배율 + n_eff·ln(sc)
    kor = {"intercept": "상수 c", "sigma2": "오차 분산 σ²"}

    def nm(n):
        if n in kor: return kor[n]
        if n.startswith("ar.S.L"): return f"계절 AR Φ ({n[6:]}개월)"
        if n.startswith("ma.S.L"): return f"계절 MA Θ ({n[6:]}개월)"
        if n.startswith("ar.L"): return f"AR φ{n[4:]}"
        if n.startswith("ma.L"): return f"MA θ{n[4:]}"
        return n

    order = f"({p['p']},{p['d']},{p['q']})" + (f"({p['P']},{p['D']},{p['Q']},{p['s']})" if seasonal else "")
    kd = p["d"] + (p.get("D", 0) * p.get("s", 0) if seasonal else 0)   # 차분 때문에 상태에 들어간 과거값 개수
    steps: list[dict] = []; checks_all: list[dict] = []

    # 검증 줄의 종류(검토 2026-11-08 공통 규칙 1): 독립 = 다른 경로·다른 계산법과 대조, 식 = 같은 배열·같은 식·정의·항등식 확인,
    # 재실행 = 엔진·statsmodels 를 다시 돌려 기록(capture)·결과 파일과 같은지, 실습 = 직접 실행해 보기 출력 = 엔진 값. 모든 검증은 필수(건너뛰지 않음).
    KIND = {"독립": "독립 대조", "식": "식 확인", "재실행": "재실행 = 기록·결과 파일", "탐색": "추가 탐색(새 최적화)", "실습": "실습"}

    def chk(what, a, b_, tol=1e-6, rel=False, *, k):
        assert k in KIND, k
        a = np.asarray(a, dtype=float); b_ = np.asarray(b_, dtype=float)
        d = float(np.max(np.abs(a - b_) / (np.abs(b_) if rel else 1.0)))
        c = {"what": f"[{KIND[k]}] {what} ({'상대 ' if rel else ''}차이 {d:.1e})", "ok": bool(d <= tol), "kind": k}
        checks_all.append({**c, "step": len(steps) + 1}); return c

    def chk_lab(lab):
        c = {"what": f"[{KIND['실습']}] 직접 실행해 보기 '{lab['title']}' 출력 = 엔진 값", "ok": bool(lab["ok"]), "kind": "실습"}
        checks_all.append({**c, "step": len(steps) + 1}); return c

    def chk_ok(what, ok, *, k):
        """참·거짓 검증(문자열·개수 같은 것)."""
        assert k in KIND, k
        c = {"what": f"[{KIND[k]}] {what}", "ok": bool(ok), "kind": k}
        checks_all.append({**c, "step": len(steps) + 1}); return c

    def S(chapter, narr, *, code=None, stack=None, nodes=(), edges=(), tensors=(), formula=None, hint=None, checks=(), say=None, notes=None, lab=None):
        """say = 음성·자막 대본(화면 숫자 그대로 — narr 와 같은 변수에서), notes = {원본 줄 번호: 그 줄 설명}, lab = 직접 실행해 보기(실행해 엔진 값과 맞춘 것)."""
        checks = list(checks)
        narr = _lt(narr); tensors = [_lt(t_) for t_ in tensors]      # 화면 글·표는 innerHTML 로 들어감 → 태그가 아닌 '<' 는 &lt; (힌트·검증 줄은 textContent 라 그대로)
        if lab is not None:
            checks.append(chk_lab(lab))
        if code is not None and notes:
            code = {**code, "notes": {str(k_): v_ for k_, v_ in notes.items()}}
        steps.append({"chapter": chapter, "narr": narr, "say": say, "code": code, "stack": list(stack or []), "nodes": list(nodes), "edges": [list(e) for e in edges],
                      "tensors": list(tensors), "formula": formula, "hint": hint, "checks": checks, "lab": lab})

    import re as _re0

    def _lt(o):
        if isinstance(o, str): return _re0.sub(r"<(?![/a-zA-Z])", "&lt;", o)
        if isinstance(o, list): return [_lt(v_) for v_ in o]
        if isinstance(o, dict): return {k_: (v_ if k_ == "type" else _lt(v_)) for k_, v_ in o.items()}
        return o

    def N(*pairs):
        """코드 줄 설명 — 같은 줄(ARIMA 는 SARIMAX(...).fit 이 한 줄)에 두 설명이 오면 이어 붙인다."""
        out: dict = {}
        for ln, tx in pairs:
            out[ln] = f"{out[ln]} · {tx}" if ln in out else tx
        return out

    def C(fn, lines, focus=None, func="", before=6, after=8):
        lines = sorted(set(lines)); return {"file": fn, "lines": lines, "focus": focus or lines[0], "func": func, "before": before, "after": after}

    def Hm(title, data, rows=None, cols=None, note=None, digits=None, numbers=True, **kw):
        t = {"type": "heat", "title": title, "data": data if isinstance(data, list) else arr(data), "rows": rows, "cols": cols, "note": note, "scale": "div", "hl": {}, "numbers": numbers,
             **{k_: v_ for k_, v_ in kw.items() if v_ is not None}}
        if digits is not None: t["digits"] = digits
        return t

    stk = ["engine/experiment.py · _walkforward_predict", "models/base.py · SeriesModel.fit", f"{MODEL} · Model.fit_series"]
    # 2026-11-10: 추정이 '여러 출발점 중 로그우도 최대'(models/base.py fit_multi_start)로 바뀌었다. 이 대본은 출발점 하나의 L-BFGS 를 따라간다 —
    # 옛 기록(출발점 하나)에서는 그대로 맞고, 새 기록에서는 '뽑힌 출발점의 추정'을 보여 주도록 다시 기록·손봐야 한다(docs/STATUS.md).
    fit_line = src.line(MODEL, "self.res, self.diag = fit_multi_start(mod"); sx_line = src.line(MODEL, "mod = SARIMAX(r * self.scale")
    ms_note = " (2026-11-10 부터는 정해진 출발점 7개에서 각각 이렇게 추정해 로그우도가 가장 높은 해를 씁니다 — 이 화면의 기록은 기본 출발점 하나로 추정한 것)"
    st_line = src.line(MODEL, '"enforce_stationarity": {"type": "bool"'); inv_line = src.line(MODEL, '"enforce_invertibility": {"type": "bool"')
    ser_line = src.line(EXP, 'if plugin.INPUT_SHAPE == "series":')
    lev = cap["level"]; r_all = cap["r_all"]; r_tr = cap["r_tr"]; tr_idx = cap["tr_idx"]
    af, a1 = cap["all_folds"], cap["all_folds_scale1"]
    ap = cap["apply"]
    n_folds = cap["fold"]["n_folds"]; tr0, tr1, te0 = tr_idx[0][:7], tr_idx[-1][:7], cap["fold"]["test"][0][:7]
    emb_m = [d.strftime("%Y-%m") for d in lev.index if pd.Timestamp(tr_idx[-1]) < d < pd.Timestamp(cap["fold"]["test"][0])]   # 학습 끝과 테스트 사이(비움)
    emb_s = ", ".join(emb_m); L_tr = cap["config"]["validation"]["train_len"]
    emb_next = (pd.Timestamp(emb_m[-1] + "-01") + pd.DateOffset(months=1)).strftime("%Y-%m") if emb_m else ""   # 비운 달의 정답이 쓰는 달(k = 1)
    ALL = ["lev", "ret", "train", "scale", "ssm", "cons", "mle", "kf", "roots", "apply", "pred"]
    sealed_from = pd.Timestamp(cap["config"]["validation"]["sealed_from"])
    t_dec = pd.Timestamp(cap["fold"]["test"][0]); t_next = t_dec + pd.DateOffset(months=1); nxt = t_next.strftime("%Y-%m")   # 결정월 → 예측 대상 달
    # ---------------------------------------------------------------- 검토(2026-11-08) 반영: 기록에 없는 사실은 엔진·statsmodels 경로로 다시 계산(읽기만)
    import importlib, inspect, warnings
    import scipy.optimize as so
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    mod = importlib.import_module(f"models.{kind}"); LEG = dict(getattr(mod, "LEGACY_DEFAULTS", {}) or {})

    def mk(yy, prm):   # models/<kind>.py 의 fit_series 와 같은 SARIMAX (아래에서 기록된 로그우도를 그대로 재현해 확인)
        kw = dict(order=(prm["p"], prm["d"], prm["q"]), enforce_stationarity=prm["enforce_stationarity"], enforce_invertibility=prm["enforce_invertibility"])
        if seasonal: kw["seasonal_order"] = (prm["P"], prm["D"], prm["Q"], prm["s"])
        else: kw["trend"] = prm["trend"]
        return SARIMAX(np.asarray(yy, dtype=float), **kw)

    def refit(prm):
        """엔진의 모델 파일(models/<kind>.py)로 마지막 폴드를 다시 추정. scipy fmin_l_bfgs_b 의 결과 정보(멈춘 이유)를 옆에서 받아 둔다 — 계산은 그대로."""
        rec = {}; orig = so.fmin_l_bfgs_b
        def wrap(*a_, **kw_):
            out = orig(*a_, **kw_); rec["d"] = out[2]; rec["kw"] = {k_: v_ for k_, v_ in kw_.items() if k_ in ("factr", "pgtol", "epsilon", "approx_grad", "maxiter")}; return out
        so.fmin_l_bfgs_b = wrap
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m0 = mod.Model(prm, n_features=1, target_type="regression", device="cpu"); m0.fit_series(np.asarray(r_tr, dtype=float))
        finally:
            so.fmin_l_bfgs_b = orig
        return m0, rec
    m_ref, lb = refit(p)
    lb_sig = inspect.signature(so.fmin_l_bfgs_b).parameters
    factr = float(lb["kw"].get("factr", lb_sig["factr"].default)); pgtol = float(lb["kw"].get("pgtol", lb_sig["pgtol"].default)); epsm = float(np.finfo(float).eps)
    task = str(lb["d"]["task"]); gmax = float(np.max(np.abs(lb["d"]["grad"])))
    by_rel = "RELATIVE REDUCTION OF F" in task
    # 모든 폴드의 학습 자료를 규칙(결정월 앞 L_tr 개월 창 − 끝 비움)으로 다시 만들고, 기록된 로그우도(배율 단위 → 원래 단위)를 재현해 확인
    p1 = {**p, "scale": 1.0}
    fold_r = lambda t_: r_all.loc[t_ - pd.DateOffset(months=L_tr): t_ - pd.DateOffset(months=1 + len(emb_m))].dropna().values   # 폴드의 학습 변화율(규칙으로 다시 자름)
    rc_ok = 0; eff = []; gain_same = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, y1 in zip(af, a1):
            t_ = pd.Timestamp(x["test"])
            rr_f = fold_r(t_)
            m100_, m1_ = mk(rr_f * sc, p), mk(rr_f, p1)
            nf_ = len(rr_f) - m100_.loglikelihood_burn
            l100 = m100_.loglike(np.asarray(x["params"])) + nf_ * np.log(sc); l1 = m1_.loglike(np.asarray(x["params"]) * unit)
            rc_ok += abs(l100 - x["llf"]) < 1e-7
            eff.append(l1 - l100); gain_same.append(l1 - y1["llf"])   # 같은 배율 1 모형(초기값 규칙도 같음)으로 잰 '배율 100 의 점 − 배율 1 의 점'
    eff = np.array(eff); gain_same = np.array(gain_same)
    eff_last = float(eff[-1]); eff_max = float(np.max(np.abs(eff))); n_gs = int((gain_same > 0).sum())
    # 결과 파일(같은 실험): 엔진이 저장한 예측·지표, 옛 결과 파일의 설정
    # 결과 파일은 대본이 말하는 내용의 출처 — 없거나 목록이 다르면 멈춘다(대체 문장을 만들지 않음: 녹음은 대본 지문으로 묶임)
    mr = match_result("forecast", cap["config"]["name"], {x["test"]: x["pred"] for x in af})
    if not mr:
        raise AssertionError("결과 파일이 없습니다 — 8장 대본이 결과 파일과의 비교를 말하므로 멈춥니다")
    rj = json.loads((ROOT / "results" / "forecast" / mr["file"].replace("_daily.parquet", ".json")).read_text(encoding="utf-8"))
    olds = [a_ for a_, n_, d_ in mr["all"] if a_ != mr["file"]]
    if len(olds) != 1:
        raise AssertionError(f"옛 결과 파일이 하나가 아닙니다({olds}) — 3·8장 대본은 옛 파일 하나를 말합니다")
    of = olds[0]; oj = json.loads((ROOT / "results" / "forecast" / of.replace("_daily.parquet", ".json")).read_text(encoding="utf-8"))
    miss = {k_: v_ for k_, v_ in LEG.items() if k_ not in oj["model"]["params"]}
    leg_ok = miss.get("enforce_stationarity") is False and miss.get("enforce_invertibility") is False and float(miss.get("scale", 0)) == 1.0
    if not leg_ok:
        raise AssertionError(f"옛 결과 파일의 채워진 설정이 대본과 다릅니다: {miss}")
    leg_txt = "배율 1 에 정상성·가역성 제약도 끈"; leg_say = "배율 1에, 정상성과 가역성 제약도 끈"
    od = pd.read_parquet(ROOT / "results" / "forecast" / of, columns=["prediction"])["prediction"].dropna()
    # 옛 파일 설정(LEGACY_DEFAULTS)으로 120개 폴드를 모두 엔진 모델 파일로 다시 추정·예측 → 옛 파일의 예측과 대조
    pr_old_all = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x in af:
            t_ = pd.Timestamp(x["test"])
            m0 = mod.Model({**p, **miss}, n_features=1, target_type="regression", device="cpu"); m0.fit_series(fold_r(t_))
            pr_old_all.append(float(m0.forecast(r_all.loc[:t_].dropna().values, 1)[0]))
    tests_ = pd.DatetimeIndex([pd.Timestamp(x["test"]) for x in af])
    s1p = pd.Series([y1["pred"] for y1 in a1], index=tests_)
    old = {"file": of.replace("_daily.parquet", ""), "miss": miss, "n": len(af),
           "d_old": float(np.max(np.abs(np.array(pr_old_all) - od.reindex(tests_).values))),
           "d_s1": float(np.max(np.abs(od.reindex(tests_).values - s1p.values))), "n_s1": len(af)}
    # ================================================================ 0. 시작
    order_say = (f"차수는 자기회귀 {p['p']}차, 차분 {p['d']}, 이동평균 {p['q']}차입니다. 상수항도 있습니다. " if not seasonal else
                 f"차수는 비계절 부분이 자기회귀 {p['p']}차, 차분 {p['d']}, 이동평균 {p['q']}차이고, "
                 + (f"계절 부분은 자기회귀·차분·이동평균이 각각 {p['P']}차씩, " if p["P"] == p["D"] == p["Q"] else f"계절 부분은 자기회귀 {p['P']}차, 계절 차분 {p['D']}, 계절 이동평균 {p['Q']}차, ")
                 + f"주기는 {p['s']}개월입니다. ")
    S("0. 시작",
      f"이 영상은 <b>{KN}{order}</b> 가 실제로 계산되는 순서를 따라갑니다. 설정 파일은 <b>{Path(cap['config_file']).name}</b>. "
      "이 모델은 다른 변수를 쓰지 않고 <b>대상(F11) 자기 자신의 월 변화율 하나</b>만 씁니다. 모형은 '과거 변화율과 과거 충격(예측하지 못한 변화)의 가중합'이고, "
      "실제 예측값은 그 모형으로 칼만 필터가 계산합니다(8장). "
      f"계수는 statsmodels 의 <b>SARIMAX</b> 클래스로 추정합니다. "
      f"워크포워드 {n_folds}개 폴드 각각에서 계수를 다시 추정하며, 화면의 숫자는 마지막 폴드(학습 {tr0} ~ {tr1}, 테스트 결정월 {te0} — 예측 대상은 다음 달 {nxt} 의 변화율, "
      f"봉인 구간 {sealed_from.strftime('%Y-%m')} ~ 의 첫 달: 8장)의 실제 실행 기록입니다.",
      say=(f"이 영상은 {KN} 모델이 실제로 계산되는 순서를 따라갑니다. " + order_say
           + "다른 변수 없이, KB 주택매매가격지수 F11의 월 변화율 하나만 씁니다. "
           "모형은 과거 변화율과 과거 충격의 가중합입니다. 계수는 statsmodels의 SARIMAX 클래스로 추정하고, 실제 예측값은 칼만 필터로 계산합니다. "
           f"{n_folds}개 폴드 각각에서 계수를 다시 추정하며, 화면의 숫자는 마지막 폴드의 실제 기록입니다. "
           f"곧 {tr0}부터 {tr1}까지로 학습하고, 결정월 {te0}에 다음 달인 {nxt}의 변화율을 예측한 것입니다. "
           "노란 줄이 지금 보는 코드, 그 아래 회색 글이 줄 설명입니다."),
      checks=[chk(f"결정월 {te0} 의 정답 = 다음 달 {nxt} 의 변화율", cap["y"].loc[t_dec], r_all.loc[t_next], 0, k="식"),
              chk_ok(f"예측 대상 달 {nxt} ≥ sealed_from {sealed_from.strftime('%Y-%m')} (봉인 구간), 결정월 {te0} < sealed_from", t_next >= sealed_from > t_dec, k="식"),
              chk(f"모든 폴드의 학습 = 결정월 앞 {L_tr}개월 창에서 끝 {len(emb_m)}개월을 뺀 구간 ({rc_ok}/{len(af)}개 폴드의 로그우도를 이 자료로 재현)", rc_ok, len(af), 0, k="재실행")],
      notes=N((src.line(MODEL, "추정 제약(2026-11-06)"), "정상성·가역성 제약 — 기본 켬(5장)"),
              (src.line(MODEL, "수치 배율(2026-11-07)"), f"변화율 × {sc:g} 로 추정하는 이유(3장)")),
      code=C(MODEL, src.span(MODEL, '"""', '"""'), 1, "모듈 설명", 0, 14), stack=[MODEL], nodes=ALL,
      tensors=[{"type": "table", "title": f"실행 설정 ({cap['config_file']} + {MODEL} 기본값)", "columns": ["항목", "값", "뜻"], "rows": [
          ["대상", "F11 다음 달 변화율", f"k = {cap['config']['target']['k']}"],
          *[[k_, str(v), {"p": "AR 차수(과거 변화율 몇 달)", "d": "차분(입력이 이미 변화율 → 0)", "q": "MA 차수(과거 예측오차 몇 달)", "trend": "c = 상수 있음", "P": "계절 AR 차수", "D": "계절 차분(12개월 전과의 차이)",
                                "Q": "계절 MA 차수", "s": "계절 주기(개월)", "enforce_stationarity": "정상성 제약", "enforce_invertibility": "가역성 제약", "scale": "추정용 배율(2026-11-07 추가)"}.get(k_, "")] for k_, v in p.items()],
          ["워크포워드", f"학습 창 {L_tr}개월(끝 {len(emb_m)}개월 비움) · 테스트 1개월", f"폴드 {n_folds}개, 폴드마다 다시 추정"]]}],
      hint="숫자의 출처: tools/codewalk/series_capture.py 가 엔진의 실제 경로(_walkforward_predict → models/" + kind + ".py → statsmodels SARIMAX)로 돌리면서 "
           "최우추정 반복마다의 모수(fit 의 callback), 상태공간 행렬·초기값, 칼만 필터의 예측오차·분산을 기록했습니다. 계산 자체는 바꾸지 않습니다.")
    n_hist = ap["n_hist"]
    info_t = cap["config"].get("forecast_info_time", "stamp")
    if info_t != "stamp":
        raise AssertionError(f"forecast_info_time = {info_t} — 이 대본의 '그달까지' 설명은 stamp 규약만 다룹니다")
    S("0. 시작",
      "호출 순서: 엔진의 <b>series 분기</b>가 F11 월별 값(코드의 close 열)의 변화율을 만들고 학습 구간만 잘라 <b>m.fit</b> 에 넘깁니다. "
      "그다음 테스트 결정월마다 <b>그달까지의 변화율 전부</b>를 m.forecast 에 넘겨 한 달 앞을 예측하고, _agg 가 k개월 값으로 묶습니다(k = 1 이면 그대로). "
      f"'그달까지' 는 엔진 규약(forecast_info_time = {info_t})을 따른 것입니다: 기준월의 값을 그달에 안다고 보고, 공표까지 걸리는 시차는 여기서 반영하지 않습니다. "
      "학습 변화율이 30개보다 적으면 그 폴드는 건너뜁니다.",
      say=("호출 순서입니다. 엔진의 시계열 모델 분기는 먼저 F11의 월별 값을 전월 대비 변화율로 바꿉니다. 그리고 학습 구간의 변화율만 잘라 모델에 넘겨 계수를 추정합니다. "
           "그다음 테스트 결정월마다, 그달까지의 변화율 전부를 예측 함수에 넘겨 한 달 앞을 예측합니다. "
           "여기서 엔진은 기준월의 값을 그달에 이미 안다고 보고, 공표까지 걸리는 시차는 반영하지 않습니다. 이 설정은 한 달 앞만 맞히므로 그 예측을 그대로 씁니다. "
           "학습 변화율이 30개보다 적은 폴드는 건너뜁니다."),
      notes=N((src.line(EXP, 'r_all = ds.prices["close"].pct_change()'), "F11 월별 값 → 전월 대비 변화율 r (첫 달은 빈칸)"),
              (src.line(EXP, "m.fit(r_tr, np.array([]))"), f"학습 변화율 {len(r_tr)}개로 계수 추정 — 정답(다음 달 변화율)은 넘기지 않음"),
              (src.line(EXP, "hist = r_all.loc[:t].dropna().values"), f"결정월 t 까지의 변화율 전부(이 폴드 {n_hist}개) — 결정월 뒤의 값은 없음. 엔진 규약({info_t}): 기준월 값은 그달에 안다고 봄, 공표 시차는 반영 안 함")),
      code=C(EXP, src.span(EXP, 'if plugin.INPUT_SHAPE == "series":', "pred.loc[t] = _agg(np.asarray(m.forecast(hist, k + skip), dtype=float)[skip:])"),
             src.line(EXP, "m.fit(r_tr, np.array([]))"), "_walkforward_predict (series 분기)", 2, 2),
      stack=["engine/experiment.py · _walkforward_predict"], nodes=["ret", "train", "mle", "apply", "pred"])
    S("0. 시작",
      f"m.fit 은 공통 틀 <b>SeriesModel.fit</b> 이고, 받은 변화율을 1차원으로 펴서 {KN} 의 <b>fit_series</b> 를 부릅니다. "
      "fit_series 가 statsmodels <b>SARIMAX(...).fit()</b> 으로 계수를 추정해 self.res 에 둡니다. 검증 구간은 쓰지 않습니다(USES_VALIDATION = False).",
      say=(f"모델 쪽의 학습 함수는 시계열 모델의 공통 틀입니다. 받은 변화율을 1차원 배열로 펴서, {KN} 모델 파일의 추정 함수를 부릅니다. "
           "그 함수가 statsmodels의 SARIMAX로 계수를 추정해 결과를 저장합니다. 이 모델은 검증 구간을 쓰지 않으므로, 학습 구간 전체를 추정에 씁니다."),
      notes=N((src.line(BASE, "USES_VALIDATION = False"), "검증 구간 없음 → 학습 구간 전체를 추정에 씀"),
              (src.line(BASE, "self.fit_series(np.asarray(X, dtype=float).ravel())"), f"1차원으로 편 변화율 → {MODEL} 의 fit_series")),
      code=C(BASE, src.lines(BASE, "USES_VALIDATION = False", "def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> None:  # X = r_train [N]", "self.fit_series(np.asarray(X, dtype=float).ravel())"),
             src.line(BASE, "self.fit_series(np.asarray(X, dtype=float).ravel())"), "SeriesModel.fit", 8, 12),
      stack=stk[:2], nodes=["train", "mle"])
    # ================================================================ 1. 변화율
    rr_full = r_all.dropna()
    rr = rr_full[rr_full.index < sealed_from]          # 통계·그림은 봉인 구간(최종 평가용으로 떼어 둔 달)을 뺀 개발 구간만
    rr0, rr1 = rr.index[0].strftime("%Y-%m"), rr.index[-1].strftime("%Y-%m"); sf_m = sealed_from.strftime("%Y-%m")
    n_in1 = int((rr.abs() < 0.01).sum())
    rows = []
    for d in r_all.loc[:tr_idx[-1]].dropna().index[-3:]:
        i = list(lev.index).index(d)
        rows.append([d.strftime("%Y-%m"), f"{lev.iloc[i - 1]:.6f}", f"{lev.iloc[i]:.6f}", f"{r_all.loc[d]:.6f}", pct(r_all.loc[d])])
    shown_ok = all(f"{float(r_[2]) / float(r_[1]) - 1:.6f}" == r_[3] for r_ in rows)   # 표에 보인 F11(소수 6자리)로 나눠도 표의 r 와 같은지
    idx = list(rr.index.strftime("%Y-%m-%d"))
    # 독립 대조: 데이터 저장소의 원자료(variables_*.parquet)를 엔진을 거치지 않고 직접 읽은 대상 변수 — 없으면 멈춤
    vdir = ROOT / "data" / "versions" / cap["config"]["data_version"]
    raw = pd.concat([pd.read_parquet(fp_) for fp_ in sorted(vdir.glob("variables_*.parquet"))], ignore_index=True)
    raw = raw[raw["variable_id"] == cap["config"]["target_variable"]].set_index("date")["value"].sort_index()
    raw.index = pd.DatetimeIndex(raw.index)
    if raw.empty:
        raise AssertionError(f"원자료에 {cap['config']['target_variable']} 가 없습니다: {vdir}")
    ex_i = 1 if len(rows) > 1 else 0                   # 대본에서 읽을 예: '0.112%'(영 점 일일이 — '일일이'로 들림) 대신 가운데 줄
    S("1. 변화율",
      f"F11 지수(수준)를 <b>전월 대비 변화율</b> r_t = F11_t ÷ F11_(t−1) − 1 로 바꿉니다(pandas pct_change). 첫 달은 전월이 없어 빈칸이 되고 dropna 로 빠집니다. "
      f"변화율은 대부분 ±1% 안쪽입니다(봉인 구간 {sf_m} 부터를 뺀 {rr0} ~ {rr1} 의 {len(rr)}개월 중 {n_in1}개월, 표준편차 {pct(rr.std(), sign=False)}). "
      f"봉인 구간은 최종 평가용으로 떼어 둔 달이라 이 장의 통계·그림에서 뺐습니다(마지막 폴드의 정답만은 봉인 첫 달 {nxt} 의 값 — 8장).",
      say=(f"F11 값을 전월 대비 변화율로 바꿉니다. 이번 달 값을 지난달 값으로 나누고 1을 뺍니다. 첫 달은 지난달 값이 없어 빈칸이 되고 빠집니다. "
           f"표는 학습 구간 마지막 3개월의 예로, {rows[ex_i][0]}의 변화율은 {rows[ex_i][4]}입니다. "
           f"최종 평가용으로 떼어 둔 봉인 구간을 빼면, {len(rr)}개월 가운데 {n_in1}개월의 변화율이 플러스 마이너스 1% 안쪽이고, 표준편차는 {pct(rr.std(), sign=False)}입니다."),
      notes=N((src.line(EXP, 'r_all = ds.prices["close"].pct_change()'), f"pct_change: F11_t ÷ F11_(t−1) − 1. 예) {rows[-1][2]} ÷ {rows[-1][1]} − 1 = {rows[-1][3]}")),
      code=C(EXP, [src.line(EXP, 'r_all = ds.prices["close"].pct_change()')], src.line(EXP, 'r_all = ds.prices["close"].pct_change()'), "_walkforward_predict", 3, 4),
      stack=["engine/experiment.py · _walkforward_predict"], nodes=["lev", "ret"], edges=[("lev", "ret")],
      formula="r_t = F11_t ÷ F11_(t−1) − 1",
      tensors=[{"type": "table", "title": "예: 학습 구간 마지막 3개월", "columns": ["월", "F11 전월", "F11", "변화율 r", "%"], "rows": rows},
               {"type": "line", "title": f"F11 월 변화율 (봉인 전 {rr0} ~ {rr1}; 초록 선 = 마지막 폴드의 학습 구간)", "x": [d[:7] for d in idx], "xlabel": "월",
                "series": [{"name": "r", "color": "#3b6fb6", "y": [num(v) for v in rr.values]}], "height": 220,
                "vlines": [{"i": idx.index(tr_idx[0]), "label": "학습 시작", "color": "#2f8f5b"}, {"i": idx.index(tr_idx[-1]), "label": "학습 끝", "color": "#2f8f5b"}]}],
      checks=[chk(f"원자료(data/versions/{cap['config']['data_version']}/variables_*.parquet 의 {cap['config']['target_variable']}, 직접 읽음) = 엔진의 월별 값 ({len(raw)}개월)",
                  raw.values, lev.reindex(raw.index).values, 0, k="독립"),
              chk("원자료로 계산한 변화율 = 엔진 r", raw.pct_change().dropna().values, rr_full.reindex(raw.index[1:]).values, 1e-15, k="독립"),
              chk("pct_change 손 계산 = 엔진 r", (lev / lev.shift(1) - 1).dropna().values, rr_full.values, 1e-15, k="식"),
              chk_ok(f"표의 F11(소수 6자리)로 다시 나눈 변화율 = 표의 r ({len(rows)}줄)", shown_ok, k="식"),
              chk_ok(f"통계·그림 범위 = 봉인 전 (마지막 달 {rr1} < sealed_from {sf_m}, {len(rr)}개월)", rr.index[-1] < sealed_from and len(rr) == len(rr_full) - int((rr_full.index >= sealed_from).sum()), k="식")])
    S("1. 변화율",
      f"학습 구간의 변화율 <b>{len(r_tr)}개</b>({tr0} ~ {tr1})만 계수 추정에 씁니다. 학습 창 {L_tr}개월에서 테스트 바로 앞 {len(emb_m)}개월({emb_s})은 비움(embargo)으로 빠집니다. "
      f"엔진이 모든 모델에 쓰는 폴드 규칙으로, 학습 정답(다음 달 변화율)이 다음 구간의 값과 겹치지 않게 비우는 것입니다(예측 모드에서 비우는 달 수 = k). "
      f"k = 1 인 이 설정에서는 비운 달 {emb_s} 의 정답이 {emb_next} 의 변화율, 곧 결정월에 이미 알려진 값이라 보수적인 여유입니다(테스트 정답 {nxt} 과 겹치지 않음). "
      f"이 모델은 정답이 아니라 변화율 r 자체로 추정하므로, {emb_s} 의 변화율은 계수 추정에서만 빠지고 예측할 때는 입력으로 들어갑니다(8장).",
      say=(f"계수 추정에는 학습 구간, {tr0}부터 {tr1}까지의 변화율 {len(r_tr)}개만 씁니다. 학습 창 {L_tr}개월 가운데 테스트 바로 앞의 {emb_s}은 비움으로 빠집니다. "
           "엔진이 모든 모델에 쓰는 폴드 규칙으로, 학습 정답이 다음 구간의 값과 겹치지 않게 비우는 것입니다. "
           "한 달 앞을 맞히는 이 설정에서는, 비운 달의 정답이 결정월에 이미 알려진 값이라 보수적인 여유입니다. "
           f"이 모델은 정답이 아니라 변화율 자체로 추정하므로, {emb_s}의 변화율은 추정에서만 빠지고, 예측할 때는 입력으로 들어갑니다."),
      notes=N((src.line(EXP, 'r_tr = r_all.loc[f.train].dropna().values'), f"f.train = 학습 창 {L_tr}개월에서 끝 {len(emb_m)}개월(비움)을 뺀 {len(tr_idx)}개월 → 변화율 {len(r_tr)}개"),
              (src.line(EXP, "if len(r_tr) < 30:"), "30개 미만이면 이 폴드 건너뜀")),
      code=C(EXP, src.lines(EXP, 'r_tr = r_all.loc[f.train].dropna().values', "if len(r_tr) < 30:", "m.fit(r_tr, np.array([]))"), src.line(EXP, 'r_tr = r_all.loc[f.train].dropna().values'), "_walkforward_predict", 3, 4),
      stack=stk[:1], nodes=["train"], edges=[("ret", "train")],
      tensors=[{"type": "line", "title": f"학습 변화율 r_tr ({len(r_tr)}개, 평균 {pct(np.mean(r_tr))}, 표준편차 {pct(np.std(r_tr, ddof=1), sign=False)})", "x": [d[:7] for d in tr_idx],
                "series": [{"name": "r", "color": "#3b6fb6", "y": [num(v) for v in r_tr], "dots": True}], "height": 220}],
      checks=[chk("r_tr = 학습 구간 r (dropna)", r_tr, r_all.loc[tr_idx[0]:tr_idx[-1]].dropna().values, 0, k="식"),
              chk_ok(f"비운 달 {emb_s} 의 정답 달 {emb_next} = 결정월 {te0} (k = {cap['config']['target']['k']}) — 테스트 정답 달 {nxt} 과 다름",
                     len(emb_m) == 1 and emb_next == te0 and emb_next != nxt and cap["config"]["target"]["k"] == 1, k="식")])
    # ================================================================ 2. 모형식
    def full(base, key, sign):   # 계절 다항식까지 곱해 펼친 다항식 (statsmodels polynomial_reduced_ar/ma)
        base = np.asarray(base, dtype=float)
        if not seasonal or key not in names: return base
        sp = np.zeros(p["s"] + 1); sp[0] = 1.0; sp[p["s"]] = sign * cap["est"][names.index(key)]
        return np.convolve(base, sp)
    arP, maP = full(cap["ar_poly"], "ar.S.L12", -1), full(cap["ma_poly"], "ma.S.L12", +1)
    dif = np.array([1.0])
    for _ in range(p["d"]): dif = np.convolve(dif, [1, -1])
    if seasonal:
        for _ in range(p["D"]):
            sp = np.zeros(p["s"] + 1); sp[0] = 1; sp[-1] = -1; dif = np.convolve(dif, sp)
    aS = np.convolve(arP, dif)          # r 에 걸리는 전체 다항식 (AR × 차분)
    if seasonal:
        form = "(1 − φ₁L)(1 − Φ₁L¹²)(1 − L¹²) r_t = (1 + θ₁L)(1 + Θ₁L¹²) ε_t,   ε ~ N(0, σ²)    (L = 한 달 전으로 미는 기호)"
        narr = (f"SARIMA{order}: ① 계절 차분 D = 1 — 변화율 자체가 아니라 '12개월 전과의 차이' r_t − r_(t−12) 를 모델링, ② AR·MA 각 1차(한 달 전), ③ 계절 AR·MA 각 1차(12개월 전). "
                "상수항은 없습니다(이 모델 파일은 trend 를 넘기지 않고, statsmodels 기본은 상수 없음). 추정할 모수는 φ₁, θ₁, Φ₁, Θ₁, σ² 5개입니다.")
        say5 = ("모형식입니다. 계절 차분 D가 1이므로, 변화율에서 12개월 전 변화율을 뺀 값을 모델링합니다. "
                "여기에 한 달 전에 거는 자기회귀와 이동평균이 1차씩, 12개월 전에 거는 계절 자기회귀와 계절 이동평균이 1차씩 있습니다. "
                "충격은 평균 0, 분산 시그마 제곱인 정규분포로 봅니다. 이 모델 파일은 상수항을 넘기지 않고 statsmodels의 기본도 상수 없음이라, 상수는 없습니다. "
                "추정할 모수는 φ1, θ1, Φ1, Θ1, 시그마 제곱의 5개입니다.")
        nt5 = N((sx_line, f"SARIMAX(r × {sc:g}, order=({p['p']}, {p['d']}, {p['q']}), seasonal_order=({p['P']}, {p['D']}, {p['Q']}, {p['s']})) — trend 없음 → 상수 없음"),
                (fit_line, "정상성·가역성 제약 켬 · .fit: 기본 방법 L-BFGS, 최대 200번 반복" + ms_note))
    else:
        form = "r_t = c + φ₁ r_(t−1) + φ₂ r_(t−2) + ε_t + θ₁ ε_(t−1) + θ₂ ε_(t−2),   ε ~ N(0, σ²)"
        narr = (f"ARIMA{order} + 상수: 이번 달 변화율 = 상수 + 지난 두 달 변화율의 가중합(AR) + 이번 달 충격 ε + 지난 두 달 충격의 가중합(MA). "
                "d = 0 인 것은 입력이 이미 변화율이라서입니다(수준값을 한 번 차분한 것과 비슷하지만 같지는 않음). 추정할 모수는 c, φ₁, φ₂, θ₁, θ₂, σ² 6개입니다.")
        say5 = ("모형식입니다. 이번 달 변화율은 상수, 지난 두 달 변화율의 자기회귀 부분, 이번 달 충격, 지난 두 달 충격의 이동평균 부분을 더한 것입니다. "
                "충격은 평균 0, 분산 시그마 제곱인 정규분포로 봅니다. 차분 d가 0인 것은 입력이 이미 변화율이기 때문입니다. "
                "추정할 모수는 상수, φ1, φ2, θ1, θ2, 시그마 제곱의 6개입니다. statsmodels에서 상수는 자기회귀 식 안에 들어가므로, 상수 자체가 평균은 아닙니다.")
        nt5 = N((sx_line, f"SARIMAX(r × {sc:g}, order=({p['p']}, {p['d']}, {p['q']}), trend='{p['trend']}') — 정상성·가역성 제약 켬"),
                (fit_line, ".fit: 기본 방법 L-BFGS, 최대 200번 반복" + ms_note))
    S("2. 모형식", narr, say=say5, notes=nt5,
      code=C(MODEL, [sx_line, fit_line], sx_line, "Model.fit_series", 6, 2),
      stack=stk, nodes=["mle"], formula=form,
      tensors=[{"type": "table", "title": "추정할 모수 (statsmodels 이름 → 뜻)", "columns": ["이름", "뜻"], "rows": [[n, nm(n)] for n in names]}],
      hint=("statsmodels 의 trend='c' 는 상수가 AR 식 안에 들어가는 형태입니다. 그래서 상수 c 는 평균이 아니고, 장기 평균 변화율은 c ÷ (1 − φ₁ − φ₂) 입니다." if not seasonal else
            "ε_t 는 '그달의 예측 못한 충격'입니다. 직접 관측되지 않으므로 칼만 필터가 예측오차로 추정해 가며 계산합니다(6장)."))
    lag_rows = []; wgt = {}; mgt = {}
    for i in range(1, max(len(aS), len(maP))):
        ca = -aS[i] if i < len(aS) else 0.0; cm = maP[i] if i < len(maP) else 0.0
        if abs(ca) > 1e-12 or abs(cm) > 1e-12:
            lag_rows.append([f"t−{i}", f"{ca:+.5f}" if abs(ca) > 1e-12 else "", f"{cm:+.5f}" if abs(cm) > 1e-12 else ""])
            wgt[i] = f"{ca:+.5f}"; mgt[i] = f"{cm:+.5f}"
    lr_mean = est[0] / (1 - est[1] - est[2]) if not seasonal else None
    ar_i = [i for i, n in enumerate(names) if n.startswith("ar.L")]; ma_i = [i for i, n in enumerate(names) if n.startswith("ma.L")]
    if seasonal:
        s_ = p["s"]
        narr6 = ("식을 곱해 펼치면 이번 달 변화율이 <b>어느 과거 달에 얼마의 무게</b>를 두는지 바로 보입니다(마지막 폴드의 추정값). "
                 f"{s_}개월 전 변화율에는 계절 차분의 1 과 Φ₁ 을 더한 {wgt[s_]}, {s_ + 1}개월 전에는 {wgt[s_ + 1]} 의 무게가 걸리고, {s_}개월 전 충격에는 Θ₁ = {mgt[s_]} 의 무게가 걸립니다. "
                 "식 양쪽의 계절 차분 (1 − L¹²) 과 계절 MA (1 + Θ₁L¹²) 가 거의 같은 모양이 되어, 서로 지워지는 것에 가깝습니다(7장).")
        say6 = ("계수를 넣고 식을 곱해 펼치면, 이번 달 변화율이 과거 어느 달에 얼마의 무게를 두는지 보입니다. "
                f"마지막 폴드에서 {s_}개월 전 변화율에는 계절 차분에서 오는 1과 Φ1을 더한 {wgt[s_]}, {s_}개월 전 충격에는 계절 이동평균 계수 {mgt[s_]}의 무게가 걸립니다. "
                "그래서 식 양쪽의 계절 차분과 계절 이동평균이 거의 같은 모양이 되어, 서로 지워지는 것에 가깝습니다. 이것은 7장에서 다시 봅니다.")
    else:
        # 실효 무게 π: 가역 ARMA 의 충격을 과거 변화율로 모두 풀었을 때 예측이 i달 전 변화율에 거는 무게 (검토 2026-11-08)
        hv = ap["hist"].values
        piw = pi_weights(est[ar_i], est[ma_i], len(hv)); c_pi = est[0] / (1 + float(np.sum(est[ma_i])))
        pred_pi = c_pi + float(sum(piw[i] * hv[-1 - i] for i in range(len(hv))))
        from statsmodels.tsa.arima_process import arma2ar
        piw_sm = -arma2ar(np.r_[1.0, -est[ar_i]], np.r_[1.0, est[ma_i]], 12)[1:]
        pi_s = [f"{x:.3f}" for x in piw[:5]]
        narr6 = ("계수를 넣으면 이번 달 변화율이 <b>어느 과거 달에 얼마의 무게</b>를 두는지 보입니다(마지막 폴드의 추정값). 이 모형은 계절 항·차분이 없어 무게 w 는 φ, m 은 θ 그대로입니다. "
                 f"AR 두 계수의 합은 {est[1] + est[2]:.2f} 입니다. 장기 평균 변화율 c ÷ (1 − φ₁ − φ₂) = {pct(lr_mean, 3)}, 학습 구간 평균은 {pct(np.mean(r_tr), 3)}. "
                 "<b>실효 무게</b>: MA 가 가역이라 충격 ε 도 과거 변화율로 풀 수 있으므로, 예측을 과거 변화율만의 합으로 다시 쓰면 i달 전 변화율의 무게 π_i 는 "
                 f"1달 전 {pi_s[0]}, 2달 전 {pi_s[1]}, 3달 전 {pi_s[2]}, 4달 전 {pi_s[3]}, 5달 전 {pi_s[4]} … 입니다 — φ₁ = {est[1]:.3f} 이 아니라 거의 1 에 가까운 무게를 지난달 변화율에 두고 "
                 f"그 앞 달들로 조금 고치는 식입니다. 이 무게와 상수 c ÷ (1 + θ₁ + θ₂) 로 결정월까지의 변화율 {len(hv)}개를 더하면 엔진의 예측과 같습니다(검증, 8장). "
                 "다만 계수 하나하나는 표준오차가 커서 해석하기 어렵습니다(7장).")
        say6 = ("계수를 넣으면 이번 달 변화율이 과거 어느 달에 얼마의 무게를 두는지 보입니다. 계절 항도 차분도 없으므로, 표의 무게는 계수 그대로입니다. "
                f"마지막 폴드에서 한 달 전 변화율에 {float(wgt[1]):.3f}, 두 달 전 변화율에 {float(wgt[2]):.3f}의 무게가 걸립니다. "
                f"그런데 충격도 과거 변화율로 풀 수 있으므로, 예측이 실제로 거는 무게로 다시 쓰면 한 달 전 {pi_s[0]}, 두 달 전은 " + ("거의 0" if abs(piw[1]) < 0.02 else pi_s[1]) + f", 세 달 전 {pi_s[2]}입니다. "
                "지난달 변화율에 거의 1의 무게를 두고, 그 앞 달들로 조금 고치는 식입니다. 이 무게로 다시 더한 예측이 엔진의 예측과 같은 것을 검증 줄이 확인합니다.")
    S("2. 모형식", narr6, say=say6,
      lab=(lab_piweights(est[0], est[ar_i], est[ma_i], ap["hist"].values, ap["date"][:7], ap["pred"]) if not seasonal else None),
      checks=([chk("펼친 식의 무게 w (손 계산: 계절 다항식을 곱하고 차분을 곱함) = statsmodels polynomial_reduced_ar × 차분", aS, np.convolve(np.asarray(m_ref.res.polynomial_reduced_ar, dtype=float), dif), 1e-12, k="독립"),
               chk("펼친 식의 충격 무게 m (손 계산) = statsmodels polynomial_reduced_ma", maP, np.asarray(m_ref.res.polynomial_reduced_ma, dtype=float), 1e-12, k="독립")] if seasonal else
              [chk("w = φ, m = θ (계절·차분 없음 — 추정값 그대로) · statsmodels polynomial_reduced_ar/ma 와 같음", np.r_[-aS[1:], maP[1:], -np.asarray(m_ref.res.polynomial_reduced_ar, dtype=float)[1:], np.asarray(m_ref.res.polynomial_reduced_ma, dtype=float)[1:]],
                   np.r_[cap["est"][ar_i], cap["est"][ma_i], cap["est"][ar_i], cap["est"][ma_i]], 1e-12, k="식")])
             + ([chk("실효 무게 π (손 계산) = statsmodels arma2ar (12달)", piw[:11], piw_sm[:11], 1e-12, k="독립"),
               chk("π₁ = φ₁ + θ₁", piw[0], est[ar_i][0] + est[ma_i][0], 1e-12, k="식"),
               chk_ok(f"대본의 '두 달 전은 거의 0': π₂ = {piw[1]:+.5f}, |π₂| < 0.02", abs(piw[1]) < 0.02, k="식"),
               chk(f"실효 무게로 결정월까지 {len(hv)}개를 더한 예측 = 엔진 예측", pred_pi, ap["pred"], 1e-12, k="독립")] if not seasonal else []),
      stack=stk + ["statsmodels · polynomial_reduced_ar / _ma"], nodes=["mle"],
      formula="r_t = " + ("c + " if not seasonal else "") + "Σ_i w_i r_(t−i) + ε_t + Σ_j m_j ε_(t−j)     (w = −[AR 다항식 × 차분 다항식], m = MA 다항식)",
      tensors=[{"type": "table", "title": "펼친 식의 무게 (0 이 아닌 것만)", "columns": ["과거 달", "변화율 r 에 거는 무게 w", "충격 ε 에 거는 무게 m"], "rows": lag_rows, "split": 2 if len(lag_rows) > 8 else 1}]
      + ([{"type": "num", "title": "상수와 장기 평균", "items": [{"label": "상수 c", "value": pct(est[0], 4)}, {"label": "장기 평균 c ÷ (1 − φ₁ − φ₂)", "value": pct(lr_mean, 3)},
                                                              {"label": "학습 구간 평균", "value": pct(np.mean(r_tr), 3)}]},
           {"type": "table", "title": "실효 무게 π (충격을 과거 변화율로 푼 것; (1 − φ₁L − φ₂L²) ÷ (1 + θ₁L + θ₂L²) = 1 − Σ π_i Lⁱ)", "columns": ["과거 달", "변화율에 거는 실효 무게 π"],
            "rows": [[f"t−{i + 1}", f"{piw[i]:+.5f}"] for i in range(6)] + [["상수 c ÷ (1 + θ₁ + θ₂)", pct(c_pi, 4)]]}] if not seasonal else []))
    # ================================================================ 3. 배율
    var = float(np.var(r_tr, ddof=1))                          # 1장 학습 변화율 그림 제목의 표준편차와 같은 식(ddof=1) — 표준편차² = 분산
    gain = np.array([x["llf"] - y["llf"] for x, y in zip(af, a1)])
    c1 = np.array([y["converged"] for y in a1]); c100 = np.array([x["converged"] for x in af])
    n_c1, n_c100, n_gain = int(c1.sum()), int(c100.sum()), int((gain > 1e-6).sum())
    n_c1_up = int((c1 & (gain > 0.01)).sum())
    nc100 = [x["test"][:7] for x in af if not x["converged"]]
    u_s100 = float(kf["uncon"][-1]); u_s1 = u_s100 / sc       # σ² 의 제약 없는 값 u = √σ² (이 폴드의 추정값, 배율 100 / 배율 1 로 쓴 경우)
    eps_fd = float(lb["kw"]["epsilon"])                        # 다시 추정할 때 statsmodels 가 scipy 에 넘긴 수치 미분 간격(MLEModel.fit 기본값)
    dpr = np.array([x["pred"] - y["pred"] for x, y in zip(af, a1)]); i_dpr = int(np.argmax(np.abs(dpr)))
    pin(kind, "3장 예측 차이 최대(capture)", f"{abs(dpr[i_dpr]) * 100:.3f}%p ({af[i_dpr]['test'][:7]})")
    pin(kind, "3장 같은 배율 1 모형으로 잰 폴드", str(n_gs))
    if seasonal:
        pin(kind, "3장 산만 초기값 차이(대본)", f"{eff_max:.3f}"); pin(kind, "3장 산만 초기값 차이(화면)", f"{eff_last:+.1e} / {eff_max:.1e}")
    s_same = ("같은 모형입니다" if not seasonal else
              f"같은 모형입니다. 다만 계절 차분 상태의 근사 산만 초기값(분산 10⁶, 4장)은 추정 단위에 고정이라 원래 단위로 보면 배율 1 은 10⁶, 배율 {sc:g} 는 {1e6 / sc ** 2:g} 이고, "
              f"그래서 같은 모수에서도 로그우도가 조금 다릅니다(배율 1 모형 − 배율 {sc:g} 모형: 이 폴드 {eff_last:+.1e}, {len(af)}개 폴드 중 절댓값 최대 {eff_max:.1e})")
    var_s = f"{var:.7f}"                                       # 읽기용(지수 표기 대신 소수)
    narr7 = (f"<b>추정용 배율</b>: 학습 변화율의 분산은 약 {var:.5f}({var_s} = {f(var)}) 정도로 매우 작습니다(표준편차 {pct(np.sqrt(var), sign=False)}, 1장의 학습 변화율 그림). 그래서 r × {sc:g} 로 바꿔 추정하고 예측을 다시 {sc:g} 로 나눕니다. "
             "φ·θ 는 배율과 무관하고 " + (f"c 는 ×{sc:g}, " if not seasonal else "") + f"σ² 는 ×{sc ** 2:g} 가 될 뿐 " + s_same
             + (f" — 정상 분포 초기값이라 원래 단위의 로그우도가 같은 모수에서 같습니다({len(af)}개 폴드 최대 차이 {eff_max:.0e})" if not seasonal else "")
             + ". 그래서 바뀌는 것은 사실상 최적화 계산의 수치 조건입니다. "
             f"이 설정의 {len(af)}개 폴드를 두 방식으로 모두 추정했습니다(배율 1 은 정상성·가역성 제약을 지금처럼 켠 채 배율만 바꾼 것): 수렴 표시 {n_c1}개 → <b>{n_c100}개</b>, "
             f"로그우도(원래 단위)는 <b>{n_gain}개 폴드에서 배율 {sc:g} 쪽이 높음</b>(증가 중앙값 +{f(np.median(gain), 2)}, 최소 +{f(gain.min(), 4)}, 최대 +{f(gain.max(), 2)}). "
             f"초기값 차이를 없애고 같은 배율 1 모형으로 두 점의 로그우도를 재도 {n_gs}개 폴드에서 배율 {sc:g} 이 찾은 점이 높습니다. "
             f"배율 1 에서 수렴 표시가 난 {n_c1}개 " + ("모두에서" if n_c1_up == n_c1 else f"중 {n_c1_up}개에서") + " 로그우도가 0.01 넘게 높아졌습니다 — 수렴 표시가 최댓값을 뜻하지는 않았습니다."
             + (f" 배율 {sc:g} 에서도 {len(nc100)}개 폴드({', '.join(nc100)})는 수렴 표시가 나지 않았습니다." if nc100 else "")
             + f" 배율 {sc:g} 에서도 수렴 표시가 전역 최댓값을 보장하지는 않습니다(" + ("이 폴드는 최댓값이 가역성 경계 쪽, 5장)." if seasonal else "이 폴드에도 로그우도가 더 높은 점이 있음, 5장)."))
    say7 = (f"추정용 배율입니다. 학습 변화율의 분산은 약 {var:.5f}로 매우 작아서, 변화율에 {sc:g}을 곱해 추정하고 예측을 다시 {sc:g}으로 나눕니다. "
            "φ와 θ는 그대로, " + (f"상수는 {sc:g}배, " if not seasonal else "") + f"시그마 제곱은 {sc ** 2:g}배가 되는 같은 모형입니다. "
            + (f"다만 계절 차분 칸의 초기 분산은 추정 단위에 고정이라, 같은 모수에서도 로그우도가 조금 다릅니다. 그 차이는 가장 큰 폴드에서도 {eff_max:.3f} 정도입니다. " if seasonal else "")
            + f"제약은 켠 채 배율 1과 {sc:g}으로 각각 추정해 비교하면, "
            + (f"원래 단위의 로그우도는 {len(af)}개 폴드 모두에서 배율 {sc:g} 쪽이 높았습니다. " if n_gain == len(af) else f"원래 단위의 로그우도는 {n_gain}개 폴드에서 배율 {sc:g} 쪽이 높았습니다. ")
            + (f"배율 1에서 수렴 표시가 났던 {n_c1}개 폴드도 모두 0.01 넘게 높아졌으니, " if n_c1_up == n_c1 else f"배율 1에서 수렴 표시가 났던 {n_c1}개 가운데 {n_c1_up}개에서도 0.01 넘게 높아졌으니, ")
            + f"수렴 표시가 최댓값을 뜻하지는 않았던 것입니다. 배율 {sc:g}에서도 수렴 표시가 최댓값을 보장하지는 않습니다.")
    hint_old = (f"옛 결과 파일({old['file']})은 {leg_txt} 설정으로 돌았습니다 — 그 결과 파일에 {', '.join(old['miss'])} 기록이 없어 엔진이 모델 파일의 LEGACY_DEFAULTS "
                f"({', '.join(f'{k_}={v_}' for k_, v_ in old['miss'].items())})로 채웁니다. 이 설정으로 {old['n']}개 폴드를 모두 다시 추정하면 파일의 예측과 최대 {old['d_old']:.1e} 차이로 같습니다(8장의 검증). "
                f"그래서 위 비교의 배율 1(제약 켬)과 옛 파일은 같은 것이 아니고, 두 예측은 {old['n_s1']}개 폴드에서 최대 {old['d_s1'] * 100:.3f}%p 다릅니다. ")
    S("3. 추정용 배율", narr7, say=say7,
      notes=N((src.line(MODEL, 'self.scale = float(p.get("scale", 1.0) or 1.0)'), f"PARAMS 의 scale 기본값 → self.scale = {sc:g} (설정 파일은 scale 을 정하지 않음)"),
              (sx_line, f"추정은 r × {sc:g} 단위"),
              (src.line(MODEL, "return np.asarray(self.res.apply(np.asarray(history"), f"예측도 history × {sc:g} 로 필터 → forecast ÷ {sc:g}")),
      code=C(MODEL, src.lines(MODEL, 'self.scale = float(p.get("scale", 1.0) or 1.0)', "mod = SARIMAX(r * self.scale", 'sc = getattr(self, "scale", 1.0)', "return np.asarray(self.res.apply(np.asarray(history"),
             src.line(MODEL, 'self.scale = float(p.get("scale", 1.0) or 1.0)'), "Model.fit_series / forecast", 4, 4),
      stack=stk, nodes=["scale"], edges=[("train", "scale")],
      formula=f"추정: SARIMAX(r × {sc:g}),   예측: forecast(history × {sc:g}) ÷ {sc:g},   로그우도(원래 단위) = 로그우도(배율) + n·ln {sc:g}   (n = 로그우도에 더한 개월 수; 배율 모형의 로그우도를 원래 단위로 바꾼 것)",
      tensors=[{"type": "table", "title": f"{len(af)}개 폴드: 배율 1 (제약은 지금처럼 켬) vs 배율 {sc:g} (지금)", "columns": ["", "배율 1", f"배율 {sc:g}"], "rows": [
          ["수렴 표시가 난 폴드", n_c1, n_c100],
          ["로그우도가 더 높은 폴드", int((gain < -1e-6).sum()), n_gain],
          ["같은 배율 1 모형으로 잰 로그우도가 더 높은 폴드", int((gain_same < 0).sum()), n_gs],
          [f"배율 1 에서 수렴 표시가 났지만 배율 {sc:g} 의 로그우도가 0.01 넘게 높은 폴드", n_c1_up, ""],
          ["한 달 예측의 차이 (절댓값 최대)", "", f"{abs(dpr[i_dpr]) * 100:.3f}%p ({af[i_dpr]['test'][:7]})"]]},
               {"type": "line", "title": f"폴드별 로그우도 증가 (배율 {sc:g} − 배율 1, 원래 단위; 0 위면 더 높은 점을 찾음)", "x": [x["test"][:7] for x in af], "xlabel": "테스트 월",
                "series": [{"name": "증가", "color": "#2f8f5b", "y": [num(g) for g in gain], "dots": True}], "height": 190}],
      hint=f"2026-11-07 에 발견해 줄인 수치 문제입니다. 배율 1 은 많은 폴드에서, 로그우도가 더 높은 점이 있는데도 그 전에 멈춘 계수로 예측했습니다. 위 비교의 배율 1 은 제약을 켠 채 배율만 바꾼 것입니다. "
           + hint_old + "화면의 숫자는 대부분 원래 단위로 바꿔 보이고, 배율 단위로 보이는 표에는 '(배율 단위)' 라고 적었습니다.\n"
           f"왜 숫자의 크기가 최적화에 영향을 주나 — 가능한 설명(확인하지 않았습니다): statsmodels 는 L-BFGS 의 기울기를 수치 미분(간격 {eps_fd:g}, 제약 없는 u 에서)으로 구합니다. "
           f"이 폴드의 σ² 를 배율 1 로 쓰면 u = √σ² ≈ {u_s1:.4f} 로 간격이 u 의 {eps_fd / u_s1 * 100:.1f}% 이지만, 배율 {sc:g} 에서는 u ≈ {u_s100:.4f} 로 {eps_fd / u_s100 * 100:.3f}% 입니다. "
           "멈춤 기준(기울기 크기·목적함수의 상대 감소)도 값의 크기에 영향을 받습니다. 어느 것이 주원인인지는 따로 시험하지 않았습니다.",
      checks=[chk("분산(ddof=1) = 1장 학습 변화율 그림의 표준편차²", var, np.std(r_tr, ddof=1) ** 2, 1e-18, k="식"),
              chk(f"같은 모수의 로그우도: 배율 1 모형 − 배율 {sc:g} 모형(원래 단위), 이 폴드 (다시 계산 {eff_last:+.2e})", float(mk(r_tr, p1).loglike(cap["est"] * unit)) - (cap["llf"] + lsc), eff_last, 1e-9, k="식"),
              chk_ok(f"같은 배율 1 모형으로 재도 배율 {sc:g} 의 점이 높은 폴드 {n_gs}/{len(af)} (로그우도 재현 {rc_ok}/{len(af)})", rc_ok == len(af), k="재실행")])
    # ================================================================ 4. 상태공간
    Z, T, R, c0 = kf["Z"], kf["T"], kf["R"], kf["c"]
    nst = T.shape[0]; blk = slice(kd, nst)
    sname = [f"차분{i + 1}" for i in range(kd)] + [f"s{i + 1}" for i in range(nst - kd)]
    ar_col = np.zeros(nst - kd); ar_col[:len(arP) - 1] = -(arP[1:])
    ma_col = np.zeros(nst - kd); ma_col[:len(maP)] = maP
    if nst <= 6:
        tens = [Hm("전이 행렬 T (상태 → 다음 달 상태)", T, rows=sname, cols=sname, digits=4, cellW=86),
                {"type": "table", "title": "나머지 행렬", "columns": ["상태", "R (충격이 들어가는 몫)", "Z (관측에 쓰는 몫)", "c (상수, 원래 단위)"],
                 "rows": [[sname[i], f"{R[i, 0]:.5f}", f"{Z[0, i]:.0f}", pct(c0[i] / sc, 4)] for i in range(nst)]}]
        narr = (f"statsmodels 는 ARMA 식을 <b>상태공간</b>으로 바꿔 계산합니다. 상태 {nst}개 = 이번 달 값(s1)과, 다음 달·다다음 달로 넘겨줄 몫(s2, s3). "
                "매달 '상태 = T × 지난 상태 + c + R × 이번 충격', '관측 = Z × 상태'. T 의 첫 열이 φ1·φ2, R 이 (1, θ1, θ2) 입니다.")
        say8 = (f"statsmodels는 이 식을 상태공간 형태로 바꿔 계산합니다. 상태는 {nst}개로, 첫 번째는 이번 달 값, 두 번째와 세 번째는 다음 달과 다다음 달로 넘겨줄 몫입니다. "
                "매달 새 상태는 전이 행렬 T 곱하기 지난 상태에 상수와 이번 충격의 몫을 더한 것이고, 관측값은 첫 번째 상태 그대로입니다. "
                "전이 행렬의 첫 열에 두 자기회귀 계수가, 충격이 들어가는 몫 R에 1과 두 이동평균 계수가 놓이는 것을 검증 줄이 확인합니다.")
    else:
        tens = [Hm(f"전이 행렬 T ({nst}×{nst}) — 왼쪽 위 {kd}칸은 계절 차분(과거 {kd}개월 값 보관), 오른쪽 아래는 ARMA", T, rows=sname, cols=sname, numbers=False, cellW=24, cellH=15, colLabelH=46),
                Hm("R (충격이 들어가는 몫) · Z (관측에 쓰는 몫)", np.vstack([R[:, 0], Z[0]]), rows=["R", "Z"], cols=sname, numbers=False, cellW=24, cellH=18, colLabelH=46)]
        narr = (f"statsmodels 는 SARIMA 식을 <b>상태공간</b>으로 바꿔 계산합니다. 상태 {nst}개 = 계절 차분을 되돌리는 데 필요한 과거 {kd}개월 값 + 펼친 ARMA({len(arP) - 1}차 AR, {len(maP) - 1}차 MA)를 위한 {nst - kd}개. "
                f"관측 = Z × 상태 = (12개월 전 값) + (ARMA 부분). ARMA 블록 첫 열에 펼친 AR 계수, R 에 펼친 MA 계수가 놓입니다.")
        say8 = (f"statsmodels는 이 식을 상태공간 형태로 바꿔 계산합니다. 상태는 {nst}개입니다. 앞의 {kd}개는 계절 차분을 되돌리는 데 필요한 지난 {kd}개월 값을 보관하고, "
                f"나머지 {nst - kd}개는 펼친 자기회귀 이동평균 부분입니다. 관측값은 12개월 전 값과 자기회귀 이동평균 부분의 합으로, Z 줄에서 색이 칠해진 두 칸이 그것입니다. "
                "펼친 AR 계수와 MA 계수가 제자리에 놓인 것을 검증 줄이 확인합니다.")
    S("4. 상태공간", narr, say=say8,
      notes=N((sx_line, f"statsmodels 가 이 식을 상태공간 행렬(T · R · Z)로 바꿔 둠 — 상태 {nst}개")),
      code=C(MODEL, [sx_line, fit_line], sx_line, "SARIMAX → ssm (상태공간)", 6, 2),
      stack=stk + ["statsmodels · SARIMAX.ssm"], nodes=["ssm"], edges=[("scale", "ssm")],
      formula="α_(t+1) = T α_t + c + R ε_(t+1),     r_t × 배율 = Z α_t,     ε ~ N(0, σ²)   (관측 상수·관측 오차는 0 — 검증)",
      tensors=tens,
      checks=[chk("T 의 ARMA 첫 열 = 펼친 AR 계수", T[blk, kd], ar_col, 1e-12, k="독립"), chk("R 의 ARMA 부분 = 펼친 MA 계수 (1, m₁, …)", R[blk, 0], ma_col, 1e-12, k="독립"),
              chk("관측 상수(obs_intercept) = 0, 관측 오차 분산 H = 0, 충격 분산 Q = σ²", np.r_[np.ravel(kf["d"]), np.ravel(kf["H"]), np.ravel(kf["Q"])], np.r_[0.0, 0.0, cap["est"][-1]], 0, k="식")])
    P0 = kf["P0"]; a0 = kf["a0"]; RQR = R @ kf["Q"] @ R.T
    if np.abs(kf["P0inf"]).max() > 0:
        raise AssertionError("정확 산만(exact diffuse) 초기값 — 이 대본의 손 계산 필터는 근사 산만만 다룹니다")
    lyap = solve_discrete_lyapunov(T[blk, blk], RQR[blk, blk])
    ck = [chk("정상 부분 초기 분산 P₀ = 이산 랴푸노프 해 (P = T P T' + R σ² R')", P0[blk, blk], lyap, 1e-10, k="독립")]
    if not seasonal:
        ck.append(chk("초기 상태 a₀ = (I − T)⁻¹ c (장기 평균)", a0, np.linalg.solve(np.eye(nst) - T, c0), 1e-12, k="독립"))
    else:
        ck.append(chk(f"차분 상태 {kd}개의 초기 분산 = 10⁶", np.diag(P0)[:kd], np.full(kd, 1e6), 0, k="식"))
    S("4. 상태공간",
      ("필터를 시작하려면 첫 달 전의 상태를 정해야 합니다. 정상성 제약을 켰으므로 statsmodels 는 <b>정상 분포</b>(무한히 오래 돌았을 때의 평균·분산)에서 시작합니다: "
       f"a₀ 는 장기 평균(s1 = {pct(a0[0] / sc)}), 분산 P₀ 는 P = T P T' + R σ² R' 를 푼 값." if not seasonal else
       f"필터를 시작하려면 첫 달 전의 상태를 정해야 합니다. ARMA 부분은 <b>정상 분포</b>(P = T P T' + R σ² R' 의 해)에서, 계절 차분이 만든 과거 {kd}개월 값은 알 수 없으므로 "
       f"<b>분산 10⁶ 의 '거의 모름'</b>(근사 산만 초기값)에서 시작합니다. 그래서 처음 {kd}개월은 예측이 사실상 없고, 로그우도 합에서도 뺍니다(loglikelihood_burn = {b}). "
       "(여기서 P 는 상태의 분산입니다 — 0장 표의 계절 차수 P 와 다른 뜻.)"),
      say=("필터를 시작하려면 첫 달 이전의 상태를 정해야 합니다. 정상성 제약을 켰으므로, statsmodels는 이 과정이 아주 오래 돌았을 때의 평균과 분산, 곧 정상 분포에서 시작합니다. "
           f"초기 상태의 첫 칸은 장기 평균인 {pct(a0[0] / sc)}이고, 초기 분산은 상태 분산의 방정식을 푼 값입니다. 초기 분산이 랴푸노프 방정식의 해와 일치함을 검증 줄이 확인합니다." if not seasonal else
           "필터를 시작하려면 첫 달 이전의 상태를 정해야 합니다. 자기회귀 이동평균 부분은 정상 분포, 곧 이 과정이 아주 오래 돌았을 때의 분포에서 시작합니다. "
           f"하지만 계절 차분이 만든 지난 {kd}개월의 값은 알 수 없으므로, 분산이 백만인, 사실상 모른다는 초기값에서 시작합니다. 이것을 근사 산만 초기값이라고 합니다. "
           f"그래서 처음 {kd}개월은 예측이 사실상 없고, statsmodels는 이 {b}개월을 로그우도 합에서도 뺍니다."),
      stack=stk + ["statsmodels · ssm.initialization"], nodes=["ssm"],
      formula="P₀ = T P₀ T' + R σ² R'" + ("     a₀ = (I − T)⁻¹ c" if not seasonal else f"     (차분 상태 {kd}개: P₀ = 10⁶ · I)"),
      tensors=[Hm("초기 분산 P₀ (배율 단위)" + (f" — 빗금 = 10⁶ (차분 상태 {kd}개, 거의 모름)" if seasonal else ""), np.where(np.abs(P0) > 1e5, np.nan, P0) if seasonal else P0,
                  rows=sname, cols=sname, numbers=not seasonal, digits=4, cellW=86 if not seasonal else 24, cellH=15 if seasonal else None, colLabelH=46 if seasonal else None)]
      + ([{"type": "table", "title": "초기 상태 a₀ (원래 단위)", "columns": ["상태", "a₀"], "rows": [[sname[i], pct(a0[i] / sc, 4)] for i in range(nst)]}] if not seasonal else []),
      checks=ck)
    # ================================================================ 5. 최우추정
    u = kf["uncon"]; th_hand = u.copy()
    pos = 0; grp_rows = []; rho_of = {}
    for key, nk, sign in (("intercept", 1 if "intercept" in names else 0, 0), ("ar.L", p["p"], 1), ("ma.L", p["q"], -1),
                          ("ar.S.L", p.get("P", 0) if seasonal else 0, 1), ("ma.S.L", p.get("Q", 0) if seasonal else 0, -1)):
        if not nk: continue
        sl = slice(pos, pos + nk)
        if sign:
            phi, rr_ = csu(u[sl]); th_hand[sl] = sign * phi
            for j in range(nk):
                grp_rows.append([names[pos + j], nm(names[pos + j]), f"{u[pos + j]:+.5f}", f"{rr_[j]:+.5f}", f"{th_hand[pos + j]:+.5f}"])
                rho_of[names[pos + j]] = f"{rr_[j]:+.5f}"
        else:
            grp_rows.append([names[pos], nm(names[pos]), f"{u[pos]:+.5f}", "(변환 없음)", f"{u[pos]:+.5f}"])
        pos += nk
    th_hand[-1] = u[-1] ** 2
    grp_rows.append([names[-1], nm(names[-1]), f"{u[-1]:+.5f}", "(제곱)", f"{th_hand[-1]:.5f}"])
    ex_n = "ma.S.L12" if seasonal else "ar.L2"; ex_i = names.index(ex_n)
    from statsmodels.tsa.arima_process import ArmaProcess
    rho_ar = csu(u[ar_i])[1]; pacf_ar = ArmaProcess(np.r_[1.0, -cap["est"][ar_i]], [1.0]).pacf(len(ar_i) + 1)[1:]   # AR 부분의 편자기상관(이론값)
    say10 = ("제약을 지키는 방법입니다. L-BFGS는 아무 실수나 될 수 있는 값 u를 움직이고, statsmodels가 u를 계수로 바꿉니다. "
             "u를, 1 더하기 u 제곱의 제곱근으로 나누면 항상 −1과 1 사이에 들어가고, 이 값들로 재귀식을 돌리면 언제나 정상인 AR 계수와 가역인 MA 계수만 나옵니다. "
             "시그마 제곱은 u의 제곱이라 0 이상입니다. "
             + (f"계절 이동평균의 u는 {u[ex_i]:+.5f}로 절댓값이 큽니다. 그래서 변환한 값이 {rho_of[ex_n]}로 −1에 매우 가깝습니다. −1에 다가갈수록 u의 절댓값은 한없이 커져야 합니다." if seasonal else
                f"예를 들어 두 번째 AR 계수의 u는 약 {u[ex_i]:.2f}로, 변환한 값 {float(rho_of[ex_n]):.3f}을 거쳐, 부호가 바뀐 계수 " + ("플러스 " if th_hand[ex_i] > 0 else "") + f"{abs(th_hand[ex_i]) if th_hand[ex_i] > 0 else th_hand[ex_i]:.3f}이 됩니다."))
    S("5. 최우추정",
      "<b>제약을 어떻게 보장하나</b>: L-BFGS 는 아무 실수나 될 수 있는 u 를 움직이고, statsmodels 가 u 를 실제 계수로 바꿉니다. "
      "u → ρ = u ÷ √(1 + u²) 은 항상 −1 과 1 사이이고, 이 ρ 들로 재귀식을 돌리면 <b>언제나 정상(AR)·가역(MA)인 계수</b>만 나옵니다. σ² 는 u² 라 항상 0 이상입니다."
      + (" ρ 가 ±1 에 다가갈수록 u 의 절댓값은 한없이 커져야 합니다." if seasonal else ""),
      say=say10,
      notes=N((st_line, "예(기본) → AR 계수를 u → ρ → φ 변환으로만 만듦 → 언제나 정상"),
              (inv_line, "예(기본) → MA 계수도 같은 변환(부호 반대) → 언제나 가역")),
      lab=lab_constrain(names, u, sc, est),
      code=C(MODEL, src.lines(MODEL, '"enforce_stationarity": {"type": "bool"', '"enforce_invertibility": {"type": "bool"'), st_line, "PARAMS (추정 제약)", 2, 4),
      stack=stk + ["statsmodels · SARIMAX.transform_params"], nodes=["cons"], edges=[("ssm", "cons")],
      formula="ρ_j = u_j ÷ √(1 + u_j²),   g_(j,i) = g_(j−1,i) + ρ_j · g_(j−1,j−i−1),   g_(j,j) = ρ_j,   φ = −(마지막 줄 g),  θ = +(마지막 줄 g)",
      tensors=[{"type": "table", "title": "마지막 폴드의 추정값: 제약 없는 u → ρ → 계수 (배율 단위)", "columns": ["이름", "뜻", "u (L-BFGS 가 움직이는 값)", "ρ ∈ (−1, 1)", "계수"], "rows": grp_rows}],
      hint=("Monahan(1984)의 변환입니다(statsmodels 소스가 인용하는 문헌). 이 부호 규약에서 AR 의 ρ 는 부호를 바꾼 편자기상관입니다 — "
            f"이 폴드의 AR: ρ = ({', '.join(f'{x:+.5f}' for x in rho_ar)}), 그 AR 과정의 편자기상관 = ({', '.join(f'{x:+.5f}' for x in pacf_ar)}) (검증). "
            "편자기상관이 모두 (−1, 1) 안이면 그 AR 과정은 정상입니다. 표의 u 는 최종 추정값을 statsmodels 역변환(untransform_params)으로 되돌린 값입니다."),
      checks=[chk("손 계산 변환(u → 계수) = statsmodels 추정값", th_hand, cap["est"], 1e-12, k="독립"),
              chk("AR 의 ρ = −(편자기상관) (statsmodels ArmaProcess.pacf)", rho_ar, -pacf_ar, 1e-10, k="독립")])
    st = cap["start"] * unit
    yk = kf["y"]
    if seasonal:
        e = yk[p["s"]:] - yk[:-p["s"]]                        # 시작값은 계절 차분한 자료로 구함
        prm, var0, i1 = css(e, p["p"], np.arange(p["p"]), p["q"], np.arange(p["q"]), 0)
        ar0, ma0 = prm[:p["p"]], prm[p["p"]:]
        z1 = [not stable(np.r_[1, -ar0]), not stable(np.r_[1, ma0])]
        ar0 = ar0 * (not z1[0]); ma0 = ma0 * (not z1[1])
        prs, _, i2 = css(e, p["P"] * p["s"], np.array([p["s"] - 1]), p["Q"] * p["s"], np.array([p["s"] - 1]), 0)
        sar0, sma0 = prs[:1], prs[1:]
        sp_ = lambda c_: np.r_[1, np.zeros(p["s"] - 1), c_]
        z2 = [not stable(sp_(-sar0)), not stable(sp_(sma0))]
        sar0 = sar0 * (not z2[0]); sma0 = sma0 * (not z2[1])
        st_hand = np.r_[ar0, ma0, sar0, sma0, max(var0, 1e-10)]; zz = z1 + z2
        how = (f"① 계절 차분한 자료 x = (r_t − r_(t−{p['s']})) × 배율 {len(e)}개로 비계절 부분: 2q = {i1['k']}차 AR 최소제곱 → 잔차, 그다음 [x_(t−1), 잔차_(t−1)] 에 최소제곱 회귀({i1['rows']}행). "
               f"σ² 시작값 = 이 회귀 잔차 중 앞 {i1['drop']}개를 뺀 {i1['n_var']}개의 제곱 평균. "
               f"② 같은 자료로 계절 부분을 따로: {i2['k']}차 AR → 잔차, [x_(t−{p['s']}), 잔차_(t−{p['s']})] 에 회귀({i2['rows']}행). "
               f"그런데 {i2['k']}차 AR 을 {i2['rows1']}행으로 맞추면 계수({i2['cols1']}개)가 행보다 많아 자료에 그대로 맞고 잔차가 0 이 됩니다(최대 |잔차| {i2['res1max']:.1e}). "
               f"그래서 그 잔차에 거는 계절 MA 의 시작값이 0 입니다.")
        say11 = ("L-BFGS의 출발점인 시작값은, statsmodels 안의 이름은 조건부 제곱합이지만, 실제 계산은 두 단계 최소제곱입니다. "
                 f"먼저 {p['s']}개월 차분한 자료 {len(e)}개를 만들고, 비계절 부분과 계절 부분을 따로 회귀합니다. "
                 f"그런데 계절 부분의 첫 단계는 {i2['k']}차 자기회귀를 {i2['rows1']}행으로 맞추게 되어, 계수가 행보다 많습니다. "
                 f"그래서 잔차가 0이 되고, 그 잔차에 거는 계절 이동평균의 시작값도 0이 됩니다. 최종값 {est[3]:.6g}과는 아주 먼 곳에서 출발한 셈입니다.")
        lab11 = lab_css_sarima(yk, p["s"], sc, st, i2["rows1"], i2["cols1"])
    else:
        prm, var0, i1 = css(yk, p["p"], np.arange(p["p"]), p["q"], np.arange(p["q"]), 1)
        st_hand = np.r_[prm, max(var0, 1e-10)]
        z1 = [not stable(np.r_[1, -prm[1:1 + p["p"]]]), not stable(np.r_[1, prm[1 + p["p"]:]])]; zz = z1
        ma0_r = np.sort(np.abs(np.roots(np.r_[1.0, st[ma_i]][::-1])))          # 시작값 MA 다항식 1 + θ1 z + θ2 z² 의 근 크기
        th1_big = abs(st[ma_i[0]]) > 1
        how = (f"① 2q = {i1['k']}차 AR 을 최소제곱으로 맞춰 잔차(충격의 추정)를 만들고, ② [상수, r_(t−1), r_(t−2), 잔차_(t−1), 잔차_(t−2)] 에 한 번에 최소제곱 회귀({i1['rows']}행, 앞 {i1['r']}개월은 과거가 모자라 뺌). "
               f"σ² 시작값 = 그 회귀 잔차 중 앞 {i1['drop']}개를 뺀 {i1['n_var']}개의 제곱 평균.")
        say11 = ("L-BFGS의 출발점인 시작값은, statsmodels 안의 이름은 조건부 제곱합이지만, 실제 계산은 두 단계 최소제곱입니다. "
                 f"첫째, 이동평균 차수의 두 배인 {i1['k']}차 자기회귀를 최소제곱으로 맞춰, 그 잔차를 충격의 추정값으로 씁니다. "
                 f"둘째, 상수와 지난 두 달의 변화율, 지난 두 달의 잔차에 한 번에 회귀합니다. 과거가 모자란 앞 {i1['r']}개월을 빼고 {i1['rows']}행을 씁니다. "
                 "시작값이 제약 밖이면 0으로 바꾸는데, 이 폴드는 해당하지 않습니다."
                 + (f" θ1의 시작값 {st[ma_i[0]]:.6g}은 1보다 크지만, MA 근의 크기가 {ma0_r[0]:.4f}와 {ma0_r[1]:.4f}로 모두 1보다 커서 가역입니다." if th1_big else ""))
        lab11 = lab_css_arima(yk, p["p"], p["q"], sc, names, st)
    S("5. 최우추정",
      "L-BFGS 의 출발점(<b>시작값</b>)은 statsmodels 의 내부 함수 _conditional_sum_squares('조건부 제곱합')가 구하지만, 실제 계산은 비선형 최소화가 아니라 "
      "<b>두 단계 최소제곱</b>입니다(긴 AR 의 잔차를 충격 대신 넣는 해넌–리사넨(1982) 방식과 같은 생각). " + how
      + " 시작값이 제약 밖(정상·가역이 아님)이면 그 부분을 0 으로 바꿉니다" + (f" — 이 폴드는 해당 없음." if not any(zz) else " — 이 폴드에서 일어남.")
      + ((f" θ₁ 시작값 {st[ma_i[0]]:.6g} 은 1 보다 크지만 MA 근의 크기가 {ma0_r[0]:.4f}, {ma0_r[1]:.4f} 로 모두 1 보다 커 가역입니다(가역 여부는 계수 하나가 아니라 근으로 정해짐)."
          if not seasonal and th1_big else ""))
      + " 아래는 원래 단위의 시작값과 최종 추정값입니다.",
      say=say11,
      notes=N((fit_line, ".fit() 이 먼저 start_params(내부 이름 '조건부 제곱합' — 실제는 두 단계 최소제곱)를 구해 L-BFGS 의 출발점으로 씀")),
      lab=lab11,
      code=C(MODEL, [fit_line], fit_line, "SARIMAX.fit → start_params (_conditional_sum_squares)", 6, 2),
      stack=stk + ["statsmodels · SARIMAX.start_params"], nodes=["cons"],
      formula=(f"1단계: x_t = Σ_(i≤2q) b_i x_(t−i) + e_t   2단계: x_t = φ x_(t−1) + θ e_(t−1) + 오차   (x = 계절 차분한 r × 배율, 모두 최소제곱; 계절 부분은 {p['s']}개월 시차로 같은 방식)" if seasonal else
               "1단계: r_t = Σ_(i≤2q) b_i r_(t−i) + e_t   2단계: r_t = c + Σ φ_i r_(t−i) + Σ θ_j e_(t−j) + 오차   (모두 최소제곱)"),
      tensors=[{"type": "table", "title": "시작값 → 최종값 (원래 단위)", "columns": ["모수", "뜻", "시작값", "최종값"],
                "rows": [[n, nm(n), f"{v:.6g}", f"{e_:.6g}"] for n, v, e_ in zip(names, st, est)]}],
      checks=[chk("손 계산 시작값(두 단계 최소제곱) = statsmodels start_params", st_hand, cap["start"], 1e-10, k="독립")]
             + ([chk_ok(f"시작값 MA 근의 크기 {ma0_r[0]:.4f}, {ma0_r[1]:.4f} > 1 (가역) — statsmodels 가 시작값을 0 으로 바꾸지 않은 것과 맞음", bool(np.all(ma0_r > 1)) and not z1[1], k="식")]
                if not seasonal else []))
    it = cap["iters"]
    llf_it = [x["llf"] + lsc for x in it]
    show = sorted(set([0, 1, 2, 3, 5, len(it) // 2, len(it) - 2, len(it) - 1]) & set(range(len(it))))
    rise = np.array(llf_it) - llf_it[0]
    k_half = int(np.argmax(rise >= 0.5 * rise[-1])) + 1             # 오른 폭의 절반을 넘긴 반복(1부터)
    mr_ = cap["mle_retvals"]; nobs = cap["nobs"]; fopt = float(mr_["fopt"])
    import re as _re
    gopt_cap = np.array([float(x_) for x_ in _re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", str(mr_["gopt"]))])
    rmr = m_ref.res.mle_retvals
    mdl = m_ref.res.model; ll_hat = cap["llf"] + lsc
    stop_txt = (f"기울기 기준(가장 큰 기울기 ≤ pgtol {pgtol:g})이 아니라, 목적함수의 상대 감소가 factr × 기계 정밀도 = {factr:g} × {epsm:.3g} = {factr * epsm:.2e} 보다 작아져 멈춘 것입니다. "
                f"마지막 기울기(제약 없는 u 에 대한 목적함수의 기울기)의 절댓값 최대는 {gmax:.2e} 로 pgtol 보다 " + ("큽니다. " if gmax > pgtol else "작습니다. ")
                if by_rel else "")
    tens12 = []; ck12 = []; extra12 = ""; say12x = ""
    if seasonal:
        i_th = names.index("ma.S.L12"); th_hat = float(cap["est"][i_th])
        grid = [-0.99, -0.995, th_hat, -0.998, -0.999, -0.9995, -0.9999]          # −1 쪽으로 차례로
        prof = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for th_ in grid:
                with mdl.fix_params({"ma.S.L12": th_}):
                    r_ = mdl.fit(start_params=np.delete(cap["est"], i_th), disp=False, maxiter=200)
                prof.append(float(r_.llf) + lsc)
        prof = np.array(prof); mono = bool(np.all(np.diff(prof) > 0)); d_end = float(prof[-1] - ll_hat)
        pin(kind, "5장 프로파일 끝(대본)", f"{d_end:.4f}"); pin(kind, "5장 프로파일 끝(화면)", f"{d_end:+.6f}")
        extra12 = (f" <b>이 폴드의 최댓값은 제약의 경계 쪽</b>: Θ 를 고정하고 나머지 모수를 다시 최대화하면(프로파일 로그우도, 아래 표) Θ 를 −1 쪽으로 밀수록 로그우도가 조금씩 올라, "
                   f"Θ = {grid[-1]} 에서 멈춘 점보다 {d_end:+.6f} 높습니다. 곧 Θ̂ = {th_hat:.6g} 는 L-BFGS 가 멈춘 점이고, 로그우도는 가역성 제약의 경계(Θ = −1, 제약 밖) 쪽으로 계속 오릅니다"
                   f"(계산한 {len(grid)}개 점에서).")
        say12x = (f"계절 이동평균 계수를 고정하고 나머지를 다시 최대화하면, 이 계수를 −1 쪽으로 밀수록 로그우도가 약 {d_end:.4f} 더 오릅니다. "
                  f"최댓값은 제약의 경계 쪽에 있고, {th_hat:.6g}는 L-BFGS가 멈춘 점입니다. ")
        tens12 = [{"type": "table", "title": "Θ 를 고정하고 나머지를 다시 최대화한 로그우도 (프로파일, 원래 단위)", "columns": ["Θ (고정)", "로그우도", "L-BFGS 가 멈춘 점과의 차이"],
                   "rows": [[(f"{th_:.6g} (멈춘 점 Θ̂)" if th_ == th_hat else f"{th_}"), f"{ll_:.6f}", f"{ll_ - ll_hat:+.6f}"] for th_, ll_ in zip(grid, prof)]}]
        ck12 = [chk_ok(f"프로파일 로그우도가 Θ → −1 쪽으로 계속 증가 (계산한 {len(grid)}개 점; 마지막 {d_end:+.2e})", mono and d_end > 0, k="탐색")]
    else:
        rng = np.random.default_rng(0); u0_ = mdl.untransform_params(cap["est"])
        runs = [("최종값", "Nelder–Mead", "nm", cap["est"]), ("최종값", "Powell", "powell", cap["est"])]
        runs += [(f"무작위 {k_ + 1} (u + 정규 잡음)", "L-BFGS", "lbfgs", mdl.transform_params(u0_ + rng.normal(0, 1, len(u0_)))) for k_ in range(6)]
        ms = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for lab_, meth_, m_k, st_ in runs:
                r_ = mdl.fit(start_params=st_, method=m_k, disp=False, maxiter=200 if m_k == "lbfgs" else 2000)
                ms.append(float(r_.llf) - float(cap["llf"]))
        ms = np.array(ms); thr = 1e-6; n_rand = len(runs) - 2
        tens12 = [{"type": "table", "title": "마지막 폴드: 최종값 가까이에서 다시 찾기 (u + N(0, 1) 6개 · Nelder–Mead · Powell)", "columns": ["출발점", "방법", "로그우도 − 기록"],
                   "rows": [[lab_, meth_, f"{d_:+.1e}"] for (lab_, meth_, _, _), d_ in zip(runs, ms)]}]
        # 마지막 폴드를 더 넓게: 출발점을 u + N(0, 2²) 로 30개 흩음(numpy 시드 0 고정), 각각 엔진과 같은 L-BFGS(maxiter 200) — 검토 2026-11-08(3차)
        rng2 = np.random.default_rng(0); n_wide = 30; wide = []
        hh_last = ap["hist"].values * sc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k_ in range(n_wide):
                st_ = mdl.transform_params(u0_ + rng2.normal(0, 2, len(u0_)))
                r_ = mdl.fit(start_params=st_, disp=False, maxiter=200)
                rs_ = mdl.smooth(np.asarray(r_.params, dtype=float))
                wide.append({"d": float(r_.llf) - float(cap["llf"]), "p": np.asarray(r_.params, dtype=float), "cv": bool(r_.mle_retvals.get("converged")),
                             "ar": float(min(abs(rs_.arroots))), "ma": float(min(abs(rs_.maroots))), "pr": float(rs_.apply(hh_last, refit=False).forecast(1)[0]) / sc})
        j_w = int(np.argmax([w_["d"] for w_ in wide])); W = wide[j_w]
        n_near_w = sum(1 for w_ in wide if w_["d"] > W["d"] - 0.01); n_up_w = sum(1 for w_ in wide if w_["d"] > 0.01)
        n_rec_w = sum(1 for w_ in wide if abs(w_["d"]) <= 0.01); n_low_w = sum(1 for w_ in wide if w_["d"] < -0.01)
        rw = mdl.smooth(W["p"])
        m_w = mod.Model(p, n_features=1, target_type="regression", device="cpu"); m_w.res = rw; m_w.scale = sc        # 엔진의 Model.forecast 로 그 점의 예측
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr_w_eng = float(m_w.forecast(ap["hist"].values, 1)[0])
            ll_w_fresh = float(mk(r_tr * sc, p).loglike(W["p"])) - float(cap["llf"])
        phi_w, th_w = W["p"][ar_i], W["p"][ma_i]
        pin(kind, "5장 넓은 탐색: 로그우도 차(대본)", f"{W['d']:.2f}"); pin(kind, "5장 넓은 탐색: 로그우도 차(화면)", f"{W['d']:+.3f}")
        pin(kind, "5장 넓은 탐색: MA 근", f"{W['ma']:.4f}"); pin(kind, "5장 넓은 탐색: AR 근", f"{W['ar']:.4f}"); pin(kind, "5장 넓은 탐색: 예측", pct(W["pr"], 3))
        pin(kind, "5장 넓은 탐색: φ·θ", f"({phi_w[0]:.3f}, {phi_w[1]:.3f}) ({th_w[0]:.3f}, {th_w[1]:.3f})")
        ar_w_hand = float(min(abs(np.roots(np.r_[1.0, -phi_w][::-1])))); ma_w_hand = float(min(abs(np.roots(np.r_[1.0, th_w][::-1]))))
        grp_w = {}
        for w_ in wide:
            grp_w.setdefault(round(w_["d"], 2), []).append(w_)
        tens12.append({"type": "table", "title": f"마지막 폴드: 더 넓게 찾기 (u + N(0, 2²) {n_wide}개, numpy 시드 0, 각 L-BFGS) — 도착한 높이별", "columns":
                       ["로그우도 − 기록 (반올림)", "출발점 수", "AR 근 최소 |z|", "MA 근 최소 |z|", "그 점의 예측"],
                       "rows": [[f"{k_:+.2f}", len(g_), f"{min(x['ar'] for x in g_):.4f} ~ {max(x['ar'] for x in g_):.4f}", f"{min(x['ma'] for x in g_):.4f} ~ {max(x['ma'] for x in g_):.4f}",
                                 f"{pct(min(x['pr'] for x in g_), 3)} ~ {pct(max(x['pr'] for x in g_), 3)}"] for k_, g_ in sorted(grp_w.items(), key=lambda kv: -kv[0])]})
        # 120개 폴드 각각: 최종값에서 Nelder–Mead·Powell 을 한 번씩(무작위 없음) — 기록된 점보다 로그우도가 높은 점이 있는지 (검토 2026-11-08)
        ups = []; ups_fresh = []; pr_rep = []; pr_best = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x in af:
                t_ = pd.Timestamp(x["test"]); mf = mk(fold_r(t_) * sc, p); prm0 = np.asarray(x["params"], dtype=float)
                ll0 = float(mf.loglike(prm0)); best_ll, best_p = ll0, prm0
                for m_k in ("nm", "powell"):
                    r_ = mf.fit(start_params=prm0, method=m_k, disp=False, maxiter=2000)
                    if float(r_.llf) > best_ll: best_ll, best_p = float(r_.llf), np.asarray(r_.params, dtype=float)
                ups.append(best_ll - ll0)
                ups_fresh.append(float(mk(fold_r(t_) * sc, p).loglike(best_p)) - float(x["llf"]) + (len(fold_r(t_)) * np.log(sc)))   # 새 모형으로 다시 잼(기록 llf 는 원래 단위)
                hh = r_all.loc[:t_].dropna().values * sc
                pr_rep.append(float(mf.smooth(prm0).apply(hh, refit=False).forecast(1)[0]) / sc)
                pr_best.append(float(mf.smooth(best_p).apply(hh, refit=False).forecast(1)[0]) / sc)
        ups = np.array(ups); ups_fresh = np.array(ups_fresh); pr_rep = np.array(pr_rep); pr_best = np.array(pr_best)
        n_up01, n_up6 = int((ups > 0.01).sum()), int((ups > 1e-6).sum()); j_up = int(np.argmax(ups))
        y_f = np.array([cap["y"].loc[pd.Timestamp(x["test"])] for x in af]); pr_af = np.array([x["pred"] for x in af])
        rm_ = lambda e_: float(np.sqrt(np.mean(e_ ** 2)))
        rmse_af, rmse_best = rm_(pr_af - y_f), rm_(pr_best - y_f)
        pin(kind, "5장 넓은 탐색: 도착 셈(최고·0.01초과·같음·낮음)", f"{n_near_w}/{n_up_w}/{n_rec_w}/{n_low_w}")
        pin(kind, "5장 120폴드: 0.01 넘게", str(n_up01)); pin(kind, "5장 120폴드: 1e-06 넘게", str(n_up6))
        pin(kind, "5장 120폴드: 최대", f"+{ups[j_up]:.3f} {af[j_up]['test'][:7]}"); pin(kind, "5장 120폴드: RMSE", f"{pct(rmse_af, 4, False)} → {pct(rmse_best, 4, False)}")
        extra12 = (f" 마지막 폴드에서 최종값 가까이 다시 찾으면(출발점 u + N(0, 1) {n_rand}개와 Nelder–Mead·Powell) 로그우도가 {thr:g} 넘게 높은 점이 없었습니다(가장 큰 차이 {ms.max():+.1e}, 아래 첫 표). "
                   f"<b>그러나 더 넓게 찾으면 더 높은 점이 있습니다</b>: 출발점을 u + N(0, 2²) 로 {n_wide}개 흩어(numpy 시드 0) 각각 L-BFGS 로 찾으면, 가장 높은 점의 로그우도가 기록보다 "
                   f"<b>{W['d']:+.3f}</b> 높습니다({n_near_w}개 출발점이 이 높이에 도착; 0.01 넘게 높은 점에 간 출발점 {n_up_w}개, 기록과 같은 높이 {n_rec_w}개, 더 낮은 점 {n_low_w}개 — 아래 둘째 표). "
                   f"그 점은 φ = ({phi_w[0]:.3f}, {phi_w[1]:.3f}), θ = ({th_w[0]:.3f}, {th_w[1]:.3f}), AR 근 최소 |z| = {W['ar']:.4f}, MA 근 최소 |z| = {W['ma']:.4f} 로 가역성 경계 바로 안쪽이고(수렴 표시 {'있음' if W['cv'] else '없음'}), "
                   f"그 점으로 낸 예측은 <b>{pct(W['pr'], 3)}</b> 로 이 영상의 예측 {pct(ap['pred'], 3)} 와 부호가 다릅니다(실제 {pct(ap['actual'], 3)}). "
                   "<b>곧 이 영상이 따라가는 계수와 예측은 엔진이 실제로 낸 값이며, 가능도의 최댓값은 아닙니다.</b> 이 탐색에서 가장 높은 점이 최댓값이라는 보장도 없습니다. "
                   f"다른 폴드에서도 더 높은 점이 있었습니다: {len(af)}개 폴드 각각 최종값에서 Nelder–Mead·Powell 을 한 번씩 돌리면 "
                   f"{n_up01}개 폴드에서 로그우도가 0.01 넘게 오릅니다({thr:g} 넘게는 {n_up6}개, 가장 큰 것 +{ups[j_up]:.3f}, {af[j_up]['test'][:7]}). "
                   f"이 수들은 이 영상의 확인 방법 기준이고, 찾는 방법을 바꾸면 달라집니다. 그 120개 폴드의 찾은 점으로 예측하면 RMSE 가 {pct(rmse_af, 4, False)} → {pct(rmse_best, 4, False)} 입니다.")
        say12x = (f"마지막 폴드도 더 넓게 찾으면, 로그우도가 {W['d']:.2f} 높은 점이 있습니다. 그 점은 이동평균 근의 크기가 {W['ma']:.4f}로 가역성 경계 바로 안쪽이고, "
                  f"그 점으로 낸 예측은 " + ("플러스 " if W["pr"] > 0 else "") + f"{pct(W['pr'], 3, False) if W['pr'] > 0 else pct(W['pr'], 3)}로, 이 영상의 예측 {pct(ap['pred'], 3)}와 부호가 다릅니다. "
                  "곧 이 영상이 따라가는 계수와 예측은 엔진이 실제로 낸 값이며, 가능도의 최댓값은 아닙니다. "
                  f"다른 폴드에서도, {len(af)}개 각각을 최종값에서 두 방법으로 다시 찾으면 {n_up01}개 폴드에서 로그우도가 0.01 넘게 오릅니다. 이 수들은 찾는 방법에 따라 달라집니다. ")
        ck12 = [chk_ok(f"마지막 폴드, 최종값 가까이 다시 찾기 {len(runs)}번: 기록보다 {thr:g} 넘게 높은 점 없음 (최대 {ms.max():+.1e})", ms.max() < thr, k="탐색"),
                chk(f"넓게 찾은 가장 높은 점의 로그우도 − 기록을 새 모형으로 다시 계산 = 탐색 결과 ({W['d']:+.4f})", ll_w_fresh, W["d"], 1e-9, k="탐색"),
                chk("그 점의 AR·MA 근 최소 |z| (np.roots 손 계산) = statsmodels arroots·maroots", [ar_w_hand, ma_w_hand], [W["ar"], W["ma"]], 1e-9, k="식"),
                chk(f"그 점의 예측: 엔진 Model.forecast(그 계수) = statsmodels apply·forecast ({pct(W['pr'], 4)})", pr_w_eng, W["pr"], 1e-15, k="식"),
                chk_ok(f"대본의 사실: 그 점의 예측 부호가 영상의 예측과 반대({pct(W['pr'], 3)} · {pct(ap['pred'], 3)}), MA 근 {W['ma']:.4f} < 1.01, 로그우도 차 {W['d']:+.3f} > 0.01",
                       np.sign(W["pr"]) != np.sign(ap["pred"]) and W["ma"] < 1.01 and W["d"] > 0.01, k="탐색"),
                chk(f"{len(af)}개 폴드: 기록된 계수로 다시 낸 예측 = 기록 (다시 찾기의 출발점 확인)", pr_rep, pr_af, 1e-12, k="재실행"),
                chk(f"{len(af)}개 폴드: 기록된 예측의 RMSE = 결과 파일 rmse", rmse_af, rj["prediction_metrics"]["rmse"], 1e-12, k="재실행"),
                chk(f"{len(af)}개 폴드: 찾은 점의 로그우도 − 기록을 새 모형으로 다시 계산 = 셈에 쓴 값 (0.01 넘게 {n_up01}개, {thr:g} 넘게 {n_up6}개)", ups_fresh, ups, 1e-7, k="탐색")]
    S("5. 최우추정",
      f"실제 반복 기록입니다. statsmodels 의 fit 은 scipy 의 L-BFGS-B 로 <b>−로그우도 ÷ {nobs}</b>(배율 단위)를 최소화합니다(기울기는 수치 미분 — 그래서 함수 계산이 반복보다 많음). "
      f"<b>{len(it)}번</b> 반복하고 멈췄고 표시는 <b>{'수렴' if mr_.get('converged') else '미수렴'}</b>(warnflag {mr_.get('warnflag')}, 함수 계산 {mr_.get('fcalls')}번). "
      f"로그우도(원래 단위)는 첫 반복 뒤 {f(llf_it[0], 3)} → 마지막 {f(llf_it[-1], 3)}, 오른 폭의 절반을 넘긴 것은 {k_half}번째 반복입니다. "
      f"<b>멈춘 이유</b>(같은 설정으로 다시 추정해 scipy 의 메시지를 받음 — 계수·반복 수가 기록과 같음): '{task}' — " + stop_txt
      + "수렴 표시는 멈춤 조건을 만족했다는 뜻이지, 전역 최댓값이라는 보장은 아닙니다(3장의 배율 1)." + extra12,
      say=(f"반복 기록입니다. statsmodels는 로그우도를 관측 수 {nobs}로 나누고 부호를 바꾼 값을 L-BFGS로 최소화합니다. "
           f"{len(it)}번 반복하고 {'수렴' if mr_.get('converged') else '미수렴'} 표시로 멈췄고, 함수 계산은 {mr_.get('fcalls')}번입니다. "
           + (f"로그우도는 첫 반복 뒤 {llf_it[0]:.2f}에서 {llf_it[-1]:.2f}까지 올랐습니다. " if seasonal else "")
           + ("같은 설정으로 다시 추정해 보면, 멈춘 이유는 기울기가 작아져서가 아니라, 목적함수가 거의 줄지 않아서였습니다. " if by_rel and gmax > pgtol else "")
           + say12x
           + ("수렴 표시만으로는 전역 최댓값인지 알 수 없습니다." if seasonal else "")),
      notes=N((fit_line, f"L-BFGS {len(it)}번 반복 · 함수 계산 {mr_.get('fcalls')}번 · 수렴 표시(warnflag {mr_.get('warnflag')})"
                         + (" — 멈춘 이유: 목적함수의 상대 감소(factr), 기울기 기준(pgtol) 아님" if by_rel and gmax > pgtol else ""))),
      code=C(MODEL, [fit_line], fit_line, "SARIMAX.fit (L-BFGS, maxiter 200)", 6, 2),
      stack=stk + ["statsmodels · SARIMAX.fit (scipy L-BFGS-B)"], nodes=["mle"], edges=[("cons", "mle")],
      formula=f"L-BFGS 가 줄이는 값 = −ℓ(배율 단위) ÷ {nobs} = {fopt:.6f}   (statsmodels 의 mle_retvals 'fopt')",
      tensors=[{"type": "line", "title": "반복마다의 로그우도 (원래 단위)", "x": [str(i + 1) for i in range(len(it))], "xlabel": "반복", "series": [{"name": "로그우도", "color": "#3b6fb6", "y": [num(v, 7) for v in llf_it], "dots": True}], "height": 190},
               {"type": "table", "title": "반복마다의 모수 (원래 단위, 일부 반복)", "columns": ["반복", *[nm(n) for n in names], "로그우도"],
                "rows": [[i + 1, *[f"{v:.4g}" for v in it[i]["params"] * unit], f"{llf_it[i]:.3f}"] for i in show]}] + tens12,
      checks=[chk("마지막 반복의 모수 = 최종 추정값", it[-1]["params"], cap["est"], 1e-10, k="식"),
              chk(f"L-BFGS 의 최종 목적함수 fopt = −로그우도(배율 단위) ÷ {nobs}", fopt, -cap["llf"] / nobs, 1e-12, k="식"),
              chk("같은 설정으로 다시 추정한 계수 = 기록 (멈춘 이유를 받으려고 다시 돌림)", np.asarray(m_ref.res.params, dtype=float), cap["est"], 1e-12, k="재실행"),
              chk_ok(f"다시 추정한 반복 {rmr['iterations']}·함수 계산 {rmr['fcalls']} = 기록", rmr["iterations"] == mr_["iterations"] and rmr["fcalls"] == mr_["fcalls"], k="재실행"),
              chk("scipy 가 돌려준 마지막 기울기 = 기록 gopt", lb["d"]["grad"], gopt_cap, 1e-8, rel=True, k="재실행"),
              chk_ok(f"멈춘 이유(scipy task) = 목적함수 상대 감소, 마지막 기울기 최대 {gmax:.2e} > pgtol {pgtol:g}", by_rel and gmax > pgtol, k="재실행")] + ck12)
    # ================================================================ 6. 칼만 필터
    hk = kfilter(kf)
    v = hk["v"] / sc; F = hk["F"] / sc ** 2
    kpos = min(kd, nst - 1)
    trows = list(range(3)) + (list(range(b, b + 3)) if b else [])
    rows_k = [[t + 1, tr_idx[t][:7], pct(hk["yhat"][t] / sc), pct(r_tr[t]), pct(v[t]), f"{F[t]:.3e}", *([f"{hk['K'][t][0]:.4f}"] if seasonal else []), f"{hk['K'][t][kpos]:.4f}"] for t in trows]
    ck13 = [chk(f"손 계산 v_t = statsmodels forecasts_error (배율 단위, {len(r_tr)}개월)", hk["v"], cap["v"], 1e-8, k="독립"),
            chk("손 계산 F_t = statsmodels forecasts_error_cov", hk["F"], cap["F"], 1e-8, rel=True, k="독립")]
    if seasonal:
        ck13.append(chk(f"{b + 1}번째 달 예측 ≈ {p['s']}개월 전 값 (ARMA 칸이 아직 0)", hk["yhat"][b], kf["y"][0], 1e-5, rel=True, k="식"))
        K_s1_burn = float(np.max(np.abs(hk["K"][:kd, kpos])))
        ck13.append(chk(f"차분1 칸의 이득 K = 1 ({len(r_tr)}개월 모두)", hk["K"][:, 0], np.ones(len(r_tr)), 1e-9, k="식"))
        ck13.append(chk_ok(f"처음 {kd}개월 s1 칸의 이득 ≈ 0 (최대 |K| {K_s1_burn:.1e} < 1e-4)", K_s1_burn < 1e-4, k="식"))
    S("6. 칼만 필터",
      "로그우도는 <b>칼만 필터</b>로 계산합니다. 한 달씩 나아가며 ① 지금까지로 본 이번 달 예측 ŷ = Z a, ② 실제와의 차이(예측오차) v = r − ŷ 와 그 분산 F, "
      "③ 오차를 얼마나 믿을지 K(칼만 이득), ④ 상태를 a ← T a + c + K v 로 고칩니다. 아래는 처음 3개월" + (f"과 {b + 1}~{b + 3}번째 달" if b else "") + "을 numpy 로 직접 돌린 값입니다."
      + (f" 처음 {kd}개월은 초기값이 '거의 모름'이라 예측이 0 이고 F 가 매우 큽니다 — 이 동안 ARMA 칸(s1)의 이득은 0 에 가까워(최대 |K| = {K_s1_burn:.1e}) s1 은 거의 움직이지 않습니다. "
         f"차분1 칸(첫 번째 차분 칸)의 이득은 {len(r_tr)}개월 내내 1 이라 관측값이 그 칸에 그대로 들어갑니다(검증). "
         f"12개월이 쌓인 {b + 1}번째 달부터 정상적인 예측이 나오고 s1 이 오차를 받아들입니다. {b + 1}번째 달의 예측 {pct(hk['yhat'][b] / sc)} 는 {p['s']}개월 전({tr_idx[0][:7]}) 변화율과 거의 같습니다."
         if seasonal else " 첫 달은 과거 정보 없이 장기 분포(a₀·P₀)만으로 예측하므로 분산 F 가 그다음 달보다 큽니다."),
      say=("로그우도는 칼만 필터로 계산합니다. 한 달씩 나아가며, 지금까지의 정보로 이번 달을 예측하고, 실제와의 차이인 예측오차 v와 그 분산 F를 구합니다. "
           "그리고 예측오차에 칼만 이득 K를 곱한 만큼 상태를 고칩니다. K는 오차를 얼마나 믿을지 정하는 값입니다. 표는 넘파이로 직접 돌린 값이고, 검증 줄에서 statsmodels의 값과 일치합니다. "
           + (f"처음 {kd}개월은 초기값을 거의 모르므로 예측이 0이고 분산이 매우 큽니다. 이 동안 자기회귀 이동평균 칸의 이득은 0에 가깝습니다. "
              "첫 번째 차분 칸의 이득은 내내 1이라, 관측값이 그 칸에 그대로 들어갑니다. "
              f"{b + 1}번째 달의 예측 {pct(hk['yhat'][b] / sc)}는, {p['s']}개월 전인 {tr_idx[0][:7]}의 변화율과 거의 같습니다."
              if seasonal else "첫 달은 과거 정보 없이 장기 분포만으로 예측하므로, 분산 F가 그다음 달보다 큽니다.")),
      notes=N((fit_line, f"로그우도 한 번 = 칼만 필터로 {len(r_tr)}개월을 한 번 거름(statsmodels 내부)")),
      code=C(MODEL, [fit_line], fit_line, "SARIMAX.loglike → KalmanFilter", 6, 2),
      stack=stk + ["statsmodels · KalmanFilter (교과서 식으로 다시 계산)"], nodes=["kf"], edges=[("mle", "kf")],
      formula="ŷ_t = Z a_t,   v_t = y_t − ŷ_t (y_t = r_t × 배율),   F_t = Z P_t Z',   K_t = T P_t Z' ÷ F_t,   a_(t+1) = T a_t + c + K_t v_t,   P_(t+1) = T P_t T' + R σ² R' − K_t F_t K_t'",
      tensors=[{"type": "table", "title": (f"칼만 필터 처음 3개월 (원래 단위; K 는 상태 {sname[kpos]} 칸)" if not seasonal else f"칼만 필터 처음 3개월과 {b + 1}~{b + 3}번째 달 (원래 단위)"),
                "columns": ["t", "월", "예측 ŷ", "실제 r", "오차 v", "분산 F", *(["이득 K (차분1 = 첫 번째 차분 칸)"] if seasonal else []), f"이득 K ({sname[kpos]} 칸)"], "rows": rows_k, "hlRows": [3, 4, 5] if b else []}],
      checks=ck13)
    ll_hand = -0.5 * np.sum(np.log(2 * np.pi * F[b:]) + v[b:] ** 2 / F[b:])
    show_k = {t: tr_idx[t][:7] for t in trows}
    S("6. 칼만 필터",
      (f"처음 {b}개월을 뺀 {n_eff}개월({tr_idx[b][:7]} ~ {tr_idx[-1][:7]})" if b else f"학습 {len(r_tr)}개월 전체") + "의 예측오차 v_t 와 ±2√F_t(정규분포라면 오차의 약 95% 가 들어가는 범위)입니다. 로그우도 = Σ log 정규밀도(v_t; 0, F_t)"
      + (f"(처음 {b}개월 제외)" if b else "") + f" = <b>{f(ll_hand, 4)}</b>. 최우추정은 이 합이 커지는 쪽으로 모수를 움직이다 L-BFGS 가 멈춘 점입니다.",
      say=((f"처음 {b}개월을 뺀 {n_eff}개월" if b else f"학습 {len(r_tr)}개월") + "의 예측오차와, 위아래로 표준편차의 두 배까지의 범위입니다. 정규분포라면 오차의 약 95%가 이 안에 들어갑니다. "
           "로그우도는 달마다 예측오차를 평균 0, 분산 F인 정규분포의 밀도에 넣어 로그를 더한 값으로, "
           + (f"처음 {b}개월을 빼고 더하면 이 폴드에서 " if b else "이 폴드에서는 ") + f"{ll_hand:.3f}로 계산됩니다. 최우추정은 이 합이 커지는 쪽으로 모수를 움직이다 멈춘 점입니다. "
           "직접 실행해 보기에서 이 계산을 순수 파이썬으로 다시 할 수 있습니다."),
      notes=N((fit_line, f"최종 로그우도(원래 단위) {f(ll_hand, 4)}")),
      lab=lab_kalman(kf, sc, b, show_k, cap["v"], cap["F"] / sc ** 2, cap["llf"] + lsc),
      code=C(MODEL, [fit_line], fit_line, "SARIMAX.loglike", 6, 2),
      stack=stk + ["statsmodels · KalmanFilter"], nodes=["kf"],
      formula=f"ℓ(모수) = −½ Σ_(t>{b}) [ ln(2π F_t) + v_t² ÷ F_t ] = {f(ll_hand, 4)}   (원래 단위 = 배율 단위 + {n_eff}·ln {sc:g})" if b else
              f"ℓ(모수) = −½ Σ_t [ ln(2π F_t) + v_t² ÷ F_t ] = {f(ll_hand, 4)}   (원래 단위 = 배율 단위 + {n_eff}·ln {sc:g})",
      tensors=[{"type": "line", "title": "학습 구간의 한 달 앞 예측오차 v_t 와 ±2√F_t (원래 단위)", "x": [d[:7] for d in tr_idx[b:]], "xlabel": "월",
                "series": [{"name": "v_t", "color": "#3b6fb6", "y": [num(x) for x in v[b:]], "dots": True}, {"name": "+2√F", "color": "#c8710f", "y": [num(2 * np.sqrt(x)) for x in F[b:]]},
                           {"name": "−2√F", "color": "#c8710f", "y": [num(-2 * np.sqrt(x)) for x in F[b:]]}], "height": 240}],
      checks=[chk("손 계산 로그우도 = statsmodels llf (원래 단위)", ll_hand, cap["llf"] + lsc, 1e-8, k="독립")])
    # ================================================================ 7. 결과
    k_par = len(names)
    tz = [x / s_ for n, x, s_ in zip(names, est, bse) if n not in ("intercept", "sigma2") and s_]
    n_in2 = sum(abs(x) < 2 for x in tz)
    sym = {"ar.L1": "φ₁", "ar.L2": "φ₂", "ma.L1": "θ₁", "ma.L2": "θ₂", "ar.S.L12": "Φ₁", "ma.S.L12": "Θ₁"}
    sym_say = {k_: v_.replace("₁", "1").replace("₂", "2") for k_, v_ in sym.items()}
    coefs = [n for n in names if n not in ("intercept", "sigma2")]
    ratio = {n: est[names.index(n)] / bse[names.index(n)] for n in coefs}
    say_lead = ("최종 계수와 표준오차입니다. 표준오차는 statsmodels 기본인 기울기 외적 방식의 근사입니다. "
                "추정값의 크기가 표준오차의 두 배보다 작으면, 그 계수는 0과 구별하기 어렵습니다. ")
    rule_txt = "표준오차가 계수 크기의 절반보다 크면(추정값 ÷ 표준오차의 절댓값이 2 보다 작으면) 그 계수는 0 과 구별하기 어렵습니다."
    tens15 = []; ck15 = []
    # 같은 추정값에서 표준오차 계산법(statsmodels cov_type)만 바꾼 비율 — 두 모델 모두 (검토 2026-11-08). 'robust' 는 statsmodels 에서 robust_oim.
    cts = {"opg": "OPG (기본, 기울기 외적)", "approx": "approx (수치 헤시안)", "oim": "oim (Harvey 1989 방식 헤시안)", "robust": "robust (= robust_oim, 준최우)"}
    rows_ct = coefs + ["sigma2"]
    rat_ct = {}; bse_ct = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ct in cts:
            r_ = mdl.filter(cap["est"], cov_type=ct); bse_ct[ct] = np.asarray(r_.bse, dtype=float)
            rat_ct[ct] = {n: float(cap["est"][names.index(n)] / bse_ct[ct][names.index(n)]) for n in rows_ct}
    tens15 = [{"type": "table", "title": "같은 추정값, 표준오차 계산법만 바꾼 추정값 ÷ 표준오차 (statsmodels cov_type)", "columns": ["모수", *cts.values()],
               "rows": [[nm(n), *[f"{rat_ct[ct][n]:+.4f}" for ct in cts]] for n in rows_ct]}]
    ck15.append(chk("OPG 로 다시 계산한 표준오차 = 기록 bse", bse_ct["opg"], cap["bse"], 1e-10, rel=True, k="재실행"))
    if seasonal:
        i_th = names.index("ma.S.L12"); i_s2 = names.index("sigma2")
        bnd = [n for n in ("ar.S.L12", "ma.S.L12") if n in names and abs(est[names.index(n)]) > 0.99]     # 경계에 붙은 계절 계수 — 비율로 판단하지 않음
        pin(kind, "7장 Θ·σ² 비율(approx·oim·robust)", " ".join(f"{rat_ct[ct][n_]:+.4f}" for n_ in ("ma.S.L12", "sigma2") for ct in ("approx", "oim", "robust")))
        ins = [n for n in coefs if n not in bnd and abs(ratio[n]) < 2]; strong = [n for n in coefs if n not in bnd and abs(ratio[n]) >= 2]
        body15 = (f" 이 폴드에서는 {'·'.join(sym[n] for n in ins)} {len(ins)}개가 ±2 안쪽이고, " + ", ".join(f"{sym[n]} 의 비율은 {ratio[n]:+.2f}" for n in strong) + " 로 뚜렷합니다. "
                  + (f"{'·'.join(sym[n] for n in bnd)} 은 −1 경계에 붙어 이 비율로 판단할 수 없습니다 — OPG 표준오차가 Θ {bse[i_th]:.3g}, σ² {bse[i_s2]:.3g} 처럼 매우 크고, "
                     f"같은 추정값에서 계산법만 바꾸면 Θ 의 비율이 " + ", ".join(f"{ct} {rat_ct[ct]['ma.S.L12']:+.4f}" for ct in ("approx", "oim", "robust"))
                     + f", σ² 의 비율이 " + ", ".join(f"{ct} {rat_ct[ct]['sigma2']:+.4f}" for ct in ("approx", "oim", "robust"))
                     + " 로 크게 달라집니다(아래 둘째 표). 경계 근처에서는 어느 근사도 그대로 믿기 어렵습니다(정규 근사의 전제가 깨질 수 있음)." if bnd else ""))
        say15 = (say_lead + f"이 폴드에서는 {'과 '.join(sym_say[n] for n in ins)}이 그렇고, "
                 + (f"{sym_say[strong[0]]} 하나만 비율이 {ratio[strong[0]]:+.2f}로 뚜렷합니다. " if len(strong) == 1 else ", ".join(f"{sym_say[n]}은 {ratio[n]:+.2f}" for n in strong) + "로 뚜렷합니다. ")
                 + (f"계절 이동평균 계수는 경계인 −1에 붙어 있어, 이 비율로는 판단할 수 없습니다. 표준오차도 {bse[i_th]:.3g}으로 매우 크고, "
                    "같은 추정값에서 계산법만 바꾸면 이 비율이 크게 달라집니다. 이런 경계 근처에서는 표준오차 근사를 그대로 믿기 어렵습니다." if bnd else ""))
        ck15.append(chk_ok(f"경계에 붙어 판단에서 뺀 계수({', '.join(bnd)})의 MA 근이 1.01 안쪽", (not bnd) or min(abs(z) for z in cap["ma_roots"]) < 1.01, k="식"))
    else:
        over = {ct: [n for n in coefs if abs(rat_ct[ct][n]) >= 2] for ct in cts}
        alt = [ct for ct in ("approx", "oim", "robust") if over[ct]]
        pin(kind, "7장 approx·robust 의 2 넘는 계수", "; ".join(f"{ct} " + ", ".join(f"{sym[n]} {rat_ct[ct][n]:+.4f}" for n in over[ct]) for ct in alt))
        body15 = (f" 이 폴드에서는 OPG 기준으로 φ·θ {len(tz)}개 중 <b>{n_in2}개가 ±2 안쪽</b>입니다. "
                  + ("그러나 같은 추정값에서 표준오차 계산법만 바꾸면 결론이 달라집니다(아래 둘째 표): "
                     + "; ".join(f"{cts[ct]} 에서는 " + ", ".join(f"{sym[n]} {rat_ct[ct][n]:+.4f}" for n in over[ct]) for ct in alt)
                     + " 로 2 를 넘습니다. 표준오차 계산법에 따라 결론이 달라질 만큼 불확실하다는 뜻입니다." if alt else "다른 표준오차 계산법(아래 둘째 표)에서도 같습니다."))
        say15 = (say_lead
                 + (f"이 기준으로는 φ와 θ {len(tz)}개 모두 0과 뚜렷이 구별되지 않습니다. " if n_in2 == len(tz) else f"이 기준으로는 φ와 θ {len(tz)}개 가운데 {n_in2}개가 그렇습니다. ")
                 + ("하지만 같은 추정값에서 표준오차를 수치 헤시안으로 계산하면, " + "과 ".join(sym_say[n] for n in over["approx"]) + "의 추정값이 표준오차의 두 배를 넘습니다. "
                    "표준오차 계산법에 따라 결론이 달라질 만큼 불확실하다는 뜻입니다. " if over["approx"] else "")
                 + "AIC와 BIC는 표 제목에 있습니다.")
    S("7. 추정 결과",
      f"최종 계수입니다(원래 단위). 표준오차는 statsmodels 기본(OPG, 기울기 외적) 근사입니다. AIC = −2ℓ + 2 × (모수 수), BIC = −2ℓ + (모수 수) × ln n (모수 {k_par}개, n = {n_eff}개월). "
      + rule_txt + body15,
      say=say15,
      notes=N((fit_line, "결과 self.res: 계수 · 표준오차(OPG) · 로그우도 · AIC · BIC")),
      code=C(MODEL, [fit_line], fit_line, "SARIMAXResults", 6, 2),
      stack=stk, nodes=["mle"],
      tensors=[{"type": "table", "title": f"추정값 · 로그우도 {f(cap['llf'] + lsc, 3)} · AIC {f(cap['aic'] - 2 * lsc, 3)} · BIC {f(cap['bic'] - 2 * lsc, 3)} "
                                          f"(원래 단위로 환산; statsmodels 의 aic {f(cap['aic'], 3)} · bic {f(cap['bic'], 3)} 는 배율 단위)",
                "columns": ["모수", "뜻", "추정값", "표준오차", "추정값 ÷ 표준오차"], "rows": [[n, nm(n), f"{x:.6g}", f"{s_:.3g}", f"{x / s_:+.2f}" if s_ else ""] for n, x, s_ in zip(names, est, bse)]}] + tens15,
      checks=[chk("AIC = −2ℓ + 2 × 모수 수 (배율 단위)", cap["aic"], -2 * cap["llf"] + 2 * k_par, 1e-9, k="식"), chk("BIC = −2ℓ + 모수 수 × ln n", cap["bic"], -2 * cap["llf"] + k_par * np.log(n_eff), 1e-9, k="식"),
              chk(f"원래 단위 AIC = statsmodels aic − 2·{n_eff}·ln {sc:g}", cap["aic"] - 2 * lsc, -2 * (cap["llf"] + lsc) + 2 * k_par, 1e-9, k="식")] + ck15)

    def roots(poly):
        poly = np.asarray(poly, dtype=float)
        return np.roots(poly[::-1]) if len(poly) > 1 and np.any(poly[1:]) else np.array([])
    rows_r = []
    for k_, poly in (("AR", arP), ("MA", maP)):
        zs = roots(poly); mods = np.round(np.abs(zs), 6)
        for m_ in sorted(set(mods)):
            grp = zs[mods == m_]; ex = grp[0]
            rows_r.append([k_, len(grp), (f"{ex.real:.4f}{ex.imag:+.4f}i" if abs(ex.imag) > 1e-9 else f"{ex.real:.4f}") + (" 등" if len(grp) > 1 else ""), f"{m_:.4f}", "> 1" if m_ > 1 else "≤ 1"])
    near = [r_ for r_ in rows_r if float(r_[3]) < 1.02]
    nA = sum(1 for x in af if min(x["ar_roots"] or [9]) < 1.01); nM = sum(1 for x in af if min(x["ma_roots"] or [9]) < 1.01)
    minA = min((r_ for r_ in rows_r if r_[0] == "AR"), key=lambda r_: float(r_[3])); minM = min((r_ for r_ in rows_r if r_[0] == "MA"), key=lambda r_: float(r_[3]))
    if seasonal:
        i_th = names.index("ma.S.L12")
        th_all = np.array([x["params"][i_th] for x in af]); mr_all = np.array([min(x["ma_roots"] or [9]) for x in af]); nearM = mr_all < 1.01
        nM_neg, nM_pos = int((nearM & (th_all < 0)).sum()), int((nearM & (th_all > 0)).sum())
        th_cut = 1.01 ** (-p["s"])                             # 계절 MA (1 + Θ z¹²) 의 근 크기 = |Θ|^(−1/12) → |z| < 1.01 ⇔ |Θ| > 1.01^(−12)
        cut_ok = bool(np.all(nearM == (np.abs(th_all) > th_cut)))
        th1_all = np.array([x["params"][names.index("ma.L1")] for x in af])
        say16 = ("근 점검입니다. 다항식의 근이 모두 크기 1보다 크면 정상이고 가역이며, 제약 변환 덕분에 추정값은 항상 이 조건을 만족합니다. "
                 f"그런데 마지막 폴드에서는 MA 근 {minM[1]}개의 크기가 {minM[3]}로, 가역성 제약의 경계 바로 안쪽입니다. "
                 f"{len(af)}개 폴드 가운데 {nM}개에서도 MA 근이 1.01 안쪽인데, 그중 {nM_neg}개는 계절 이동평균 계수가 −1 쪽, {nM_pos}개는 플러스 1 쪽입니다. "
                 "−1에 가까운 것은 흔히 계절 차분이 과하다는 신호로 봅니다. 차분과 계절 이동평균이 서로 지워져, 계절 차분 없이 달마다 고정된 수준을 둔 모형에 가까워지기 때문입니다. "
                 "플러스 1 쪽은 차분을 지우지 않습니다. 이 영상에서 계절 차분이 없는 모형과 비교하지는 않았습니다.")
        tail16 = (f" 계절 MA (1 + Θz¹²) 의 근의 크기는 |Θ|^(−1/12) 이라 |z| < 1.01 ⇔ |Θ| > {th_cut:.3f} 이고, 비계절 θ₁ 의 근 1/|θ₁| 은 모든 폴드에서 약 {1 / np.max(np.abs(th1_all)):.1f} 이상입니다(최소 {1 / np.max(np.abs(th1_all)):.4f}, 검증). "
                  f"MA 근이 1.01 안쪽인 {nM}개 폴드 가운데 <b>{nM_neg}개는 Θ < 0(−1 쪽), {nM_pos}개는 Θ > 0(+1 쪽)</b>입니다."
                  + (" Θ ≈ −1 은 흔히 계절 차분(D = 1)이 과하다는 신호로 봅니다 — (1 − L¹²) 차분과 (1 − L¹²) 에 가까운 계절 MA 가 서로 지워져, "
                     "<b>계절 차분을 하지 않고 달마다 고정된 수준(결정적 계절성)을 둔 모형</b>에 가까워집니다. Θ ≈ +1 쪽의 (1 + L¹²) 는 차분을 지우지 않습니다. "
                     "이 영상에서 D = 0 모형과 비교하지는 않았습니다. "
                     f"또 로그우도에 쓰는 달이 {n_eff}개월(주기 {p['s']}개월로 {n_eff / p['s']:.1f}번)뿐이라, 작은 표본 때문에 경계 추정이 나오기도 합니다(가능한 설명 — 확인하지 않음)."
                     if nM_neg > len(af) // 3 else ""))
        ck16 = [chk_ok(f"MA 근 1.01 안쪽 폴드 {nM} = Θ<0 쪽 {nM_neg} + Θ>0 쪽 {nM_pos}", nM == nM_neg + nM_pos, k="식"),
                chk_ok(f"{len(af)}개 폴드: statsmodels MA 근 |z| < 1.01 ⇔ |Θ| > 1.01^(−12) = {th_cut:.3f} (식으로 셈과 같음), 비계절 근 최소 {1 / np.max(np.abs(th1_all)):.4f} > 1.01",
                       cut_ok and 1 / np.max(np.abs(th1_all)) > 1.01, k="독립")]
    else:
        say16 = ("근 점검입니다. 자기회귀와 이동평균 다항식의 근이 모두 크기 1보다 크면 정상이고 가역인데, 제약 변환 덕분에 추정값은 항상 이 조건을 만족합니다. "
                 + (f"마지막 폴드에서 가장 작은 근의 크기는 AR 쪽이 {minA[3]}이고, MA 쪽 근은 그보다 더 커서 경계와 멉니다. " if float(minM[3]) > float(minA[3]) else
                  f"마지막 폴드에서 가장 작은 근의 크기는 MA 쪽이 {minM[3]}이고, AR 쪽 근은 그보다 더 큽니다. ")
                 + f"{len(af)}개 폴드 전체에서는 AR 근이 1.01 안쪽인 폴드가 {nA}개, MA 근이 1.01 안쪽인 폴드가 {nM}개로, 경계에 붙은 폴드도 있습니다.")
        tail16 = ""; ck16 = []
    S("7. 추정 결과",
      "<b>근 점검</b>: AR 다항식과 MA 다항식" + ("(계절은 곱해 펼친 것, 계절 차분 1 − z¹² 은 넣지 않음)" if seasonal else "") + "의 근 z 가 모두 크기 1 보다 크면 정상·가역입니다(제약 변환으로 보장). "
      + (f"이 폴드에는 크기가 1 에 매우 가까운 근이 있습니다({', '.join(r_[0] + ' |z| = ' + r_[3] for r_ in near)}) — 추정값이 제약의 경계(|z| = 1) 바로 안쪽에 있습니다. " if near else "")
      + f"{len(af)}개 폴드 전체: AR 근이 1.01 안쪽인 폴드 <b>{nA}개</b>, MA 근이 1.01 안쪽인 폴드 <b>{nM}개</b>." + tail16,
      say=say16,
      notes=N((st_line, "정상성: AR 다항식의 근 |z| > 1"), (inv_line, "가역성: MA 다항식의 근 |z| > 1")),
      code=C(MODEL, src.lines(MODEL, '"enforce_stationarity": {"type": "bool"', '"enforce_invertibility": {"type": "bool"'), inv_line, "PARAMS (추정 제약)", 2, 4),
      stack=stk + ["statsmodels · arroots / maroots"], nodes=["roots"], edges=[("mle", "roots")],
      formula=("AR: (1 − φ₁z)(1 − Φ₁z¹²) = 0,   MA: (1 + θ₁z)(1 + Θ₁z¹²) = 0,   판정: 모든 |z| > 1" if seasonal else
               "AR: 1 − φ₁z − φ₂z² = 0,   MA: 1 + θ₁z + θ₂z² = 0,   판정: 모든 |z| > 1"),
      tensors=[{"type": "table", "title": "마지막 폴드의 근 (크기가 같은 근은 묶음)", "columns": ["다항식", "근 개수", "근 z (예)", "|z|", "판정"], "rows": rows_r},
               {"type": "line", "title": "폴드별 가장 작은 근의 크기 (1 에 붙으면 제약의 경계; 2 보다 큰 값은 위에서 잘림)", "x": [x["test"][:7] for x in af], "xlabel": "테스트 월", "ylo": 0.95, "yhi": 2.0,
                "series": [{"name": "AR 최소 |z|", "color": "#3b6fb6", "y": [num(min(x["ar_roots"] or [np.nan])) for x in af], "dots": True},
                           {"name": "MA 최소 |z|", "color": "#c8710f", "y": [num(min(x["ma_roots"] or [np.nan])) for x in af], "dots": True}], "height": 210}],
      checks=[chk("AR 근 크기 = statsmodels arroots", sorted(abs(z) for z in roots(arP)), sorted(abs(z) for z in cap["ar_roots"]), 1e-8, k="식"),
              chk("MA 근 크기 = statsmodels maroots", sorted(abs(z) for z in roots(maP)), sorted(abs(z) for z in cap["ma_roots"]), 1e-8, k="식")] + ck16)
    PP = np.array([x["params"] for x in af]) * unit
    iqr = lambda v_: float(np.percentile(v_, 75) - np.percentile(v_, 25))
    rng_rows = [[nm(n), f"{np.min(PP[:, i]):.4g}", f"{np.median(PP[:, i]):.4g}", f"{np.max(PP[:, i]):.4g}", f"{iqr(PP[:, i]):.4g}"] for i, n in enumerate(names)]
    cols_ = ["#3b6fb6", "#c8710f", "#2f8f5b", "#6a3fa0", "#b3261e"]
    i_p1 = names.index("ar.L1")
    ser17 = [{"name": nm(n), "color": cols_[j % len(cols_)], "y": [num(v_) for v_ in PP[:, i]]} for j, (i, n) in enumerate((i, n) for i, n in enumerate(names) if n not in ("intercept", "sigma2"))]
    ck17 = []
    if seasonal:
        th_ = PP[:, names.index("ma.S.L12")]; n_th95 = int((th_ <= -0.95).sum()); n_thpos = int((th_ > 0).sum())
        n_pos_near = int(((th_ > 0) & nearM).sum())
        add17 = (f" 계절 MA Θ 는 {len(af)}개 폴드 중 {n_th95}개에서 −0.95 이하(−1 경계 쪽)이고, {n_thpos}개 폴드에서는 양수이며 "
                 f"그중 {n_pos_near}개는 MA 근이 1.01 안쪽인 +1 경계 쪽입니다.")
        narr17 = (f"워크포워드는 폴드마다 계수를 다시 추정합니다. 아래는 {len(af)}개 폴드의 계수(원래 단위)입니다. 계수가 폴드마다 크게 바뀌면 '안정된 관계'라고 말하기 어렵습니다." + add17)
        say17 = (f"폴드마다 계수를 다시 추정하므로 계수도 폴드마다 다릅니다. φ1은 {len(af)}개 폴드에서 {rng_rows[i_p1][1]}부터 {rng_rows[i_p1][3]}까지입니다. "
                 f"계절 이동평균 계수는 {n_th95}개 폴드에서 −0.95 이하로 −1 경계 쪽이고, {n_thpos}개 폴드에서는 양수이며, 그중 {n_pos_near}개는 플러스 1 경계 쪽입니다. "
                 "계수가 이만큼 바뀌면 안정된 관계라고 말하기 어렵습니다.")
    else:
        pi1 = PP[:, ar_i[0]] + PP[:, ma_i[0]]                                   # 실효 무게 π₁ = φ₁ + θ₁ (2장)
        pi1_chk = np.array([pi_weights(x_[ar_i], x_[ma_i], 3)[0] for x_ in PP])
        d_phi, d_pi = float(np.median(np.abs(np.diff(PP[:, i_p1])))), float(np.median(np.abs(np.diff(pi1))))
        j_ = int(np.argmax(np.abs(np.diff(PP[:, i_p1])))); corr_ = float(np.corrcoef(PP[:, ar_i[0]], PP[:, ma_i[0]])[0, 1])
        narr17 = (f"워크포워드는 폴드마다 계수를 다시 추정합니다. 아래는 {len(af)}개 폴드의 계수(원래 단위)입니다. 계수 하나하나는 폴드마다 크게 바뀝니다"
                  f"(φ₁ 은 {rng_rows[i_p1][1]} ~ {rng_rows[i_p1][3]}, 사분위범위 {iqr(PP[:, i_p1]):.2f}). 그러나 예측이 지난달 변화율에 거는 <b>실효 무게 π₁ = φ₁ + θ₁</b>(2장)은 훨씬 덜 바뀝니다"
                  f"(사분위범위 {iqr(pi1):.2f}; 이웃 폴드 사이 변화의 중앙값 φ₁ {d_phi:.3f} · π₁ {d_pi:.3f}). "
                  f"예) φ₁ 이 가장 크게 바뀐 {af[j_]['test'][:7]} → {af[j_ + 1]['test'][:7]}: φ₁ {PP[j_, i_p1]:.3f} → {PP[j_ + 1, i_p1]:.3f} 이지만 π₁ {pi1[j_]:.3f} → {pi1[j_ + 1]:.3f}. "
                  f"φ₁ 과 θ₁ 이 폴드 간에 서로 반대로 움직여(상관 {corr_:.2f}) 비슷한 예측식을 만들 수 있어, 계수 하나하나가 잘 정해지지 않는 것으로 보입니다(가능한 설명).")
        say17 = (f"폴드마다 계수를 다시 추정하므로 계수도 폴드마다 다릅니다. φ1은 {len(af)}개 폴드에서 {rng_rows[i_p1][1]}부터 {rng_rows[i_p1][3]}까지 크게 바뀝니다. "
                 f"그러나 예측이 지난달 변화율에 거는 실효 무게는 훨씬 덜 바뀝니다. 사분위범위가 φ1은 {iqr(PP[:, i_p1]):.2f}, 실효 무게는 {iqr(pi1):.2f}입니다. "
                 "φ1과 θ1이 서로 반대로 움직여 비슷한 예측식을 만들 수 있어서, 계수 하나하나가 잘 정해지지 않는 것으로 보입니다.")
        ser17.append({"name": "실효 무게 π₁ = φ₁ + θ₁", "color": "#222222", "y": [num(v_) for v_ in pi1]})
        rng_rows.append(["실효 무게 π₁ = φ₁ + θ₁", f"{np.min(pi1):.4g}", f"{np.median(pi1):.4g}", f"{np.max(pi1):.4g}", f"{iqr(pi1):.4g}"])
        ck17 = [chk(f"폴드별 π₁ = φ₁ + θ₁ = 펼친 식의 첫 무게 ({len(af)}개 폴드)", pi1, pi1_chk, 1e-12, k="식")]
    S("7. 추정 결과", narr17,
      say=say17,
      stack=["tools/codewalk/series_capture.py · 폴드마다 같은 엔진 경로로 추정"], nodes=["mle"],
      tensors=[{"type": "line", "title": "폴드별 계수 (φ·θ" + ("·Φ·Θ" if seasonal else " · 실효 무게 π₁") + ")", "x": [x["test"][:7] for x in af], "xlabel": "테스트 월",
                "series": ser17, "height": 250},
               {"type": "table", "title": "폴드 간 범위 (원래 단위)", "columns": ["모수", "최소", "중앙", "최대", "사분위범위"], "rows": rng_rows}],
      checks=ck17)
    # ================================================================ 8. 예측
    vv = ap["v"] / sc; yh = ap["yhat"] / sc; FF = ap["F"] / sc ** 2
    hist = ap["hist"]
    m_ = cap["ssm"]; a_next = m_["T"] @ m_["a_last"] + m_["c"]; fc_ss = float((m_["Z"] @ a_next + m_["d"])[0]) / sc
    lab18 = lab_ssforecast(m_["T"], m_["a_last"], np.ravel(m_["c"]), np.ravel(m_["Z"]), sc, ap["n_hist"], ap["date"][:7], ap["pred"]) if seasonal else None
    # 표의 각 달 d 를 실제로 맞힌 워크포워드 예측 = 결정월 d − 1 달인 폴드의 예측(기록), 결과 파일의 예측과 대조
    af_by = {pd.Timestamp(x["test"]): x["pred"] for x in af}
    d6 = list(hist.index[-6:]); dprev = [d - pd.DateOffset(months=1) for d in d6]
    wf6 = np.array([af_by.get(d, np.nan) for d in dprev])
    bd = pd.read_parquet(ROOT / "results" / "forecast" / mr["file"], columns=["prediction"])["prediction"] if mr else None
    tr_set = {pd.Timestamp(d_) for d_ in tr_idx}; in_tr = [d for d in d6 if d in tr_set]
    t18 = (f"결정월까지 6개월: 실제 · 마지막 폴드 계수로 필터가 낸 한 달 앞 예측 · 오차 (원래 단위) — 필터 예측은 워크포워드 예측(오른쪽 열: 그 전달이 결정월인 폴드의 예측)과 다릅니다"
           + (f"; {in_tr[0].strftime('%Y-%m')} ~ {in_tr[-1].strftime('%Y-%m')} 은 이 계수의 학습 구간 안" if in_tr else ""))
    S("8. 예측",
      f"테스트 결정월 {ap['date'][:7]}: forecast 는 계수를 그대로 두고(refit=False) <b>{ap['hist_from'][:7]} ~ {ap['date'][:7]} 의 변화율 {ap['n_hist']}개 전부</b>를 칼만 필터에 다시 통과시킵니다(apply). "
      f"학습 구간보다 앞의 달과 비움으로 뺐던 {emb_s} 도 상태를 맞추는 데 쓰이지만 계수는 다시 정하지 않고, 결정월 다음 달 값은 쓰지 않습니다. 마지막 상태를 한 달 앞으로 밀면 예측이 됩니다.",
      say=(f"이제 예측입니다. 결정월 {ap['date'][:7]}에 예측 함수는 추정한 계수를 그대로 두고, {ap['hist_from'][:7]}부터 {ap['date'][:7]}까지의 변화율 {ap['n_hist']}개 전부를 칼만 필터에 다시 통과시킵니다. "
           f"학습 구간보다 앞의 달과, 비움으로 뺐던 {emb_s}도 상태를 맞추는 데 쓰이지만, 계수는 다시 추정하지 않습니다. 결정월 다음 달의 값은 쓰지 않습니다. "
           f"마지막 상태를 한 달 앞으로 밀고 배율 {sc:g}으로 나누면, 예측 {pct(fc_ss, 4)}가 됩니다. "
           "표의 예측은 이 폴드의 계수로 지난 달들을 다시 거른 값이라, 각 달을 실제로 맞힌 워크포워드 예측과는 다릅니다."
           + (" 직접 실행해 보기에서 이 한 걸음을 계산해 볼 수 있습니다." if seasonal else "")),
      notes=N((src.line(MODEL, 'sc = getattr(self, "scale", 1.0)'), f"학습 때 정한 배율 {sc:g}"),
              (src.line(MODEL, "return np.asarray(self.res.apply(np.asarray(history"), f"apply(변화율 {ap['n_hist']}개 × {sc:g}, refit=False) → forecast(1) ÷ {sc:g} = {pct(fc_ss, 4)}")),
      lab=lab18,
      code=C(MODEL, src.lines(MODEL, 'sc = getattr(self, "scale", 1.0)', "return np.asarray(self.res.apply(np.asarray(history"), src.line(MODEL, "return np.asarray(self.res.apply(np.asarray(history"), "Model.forecast", 4, 2),
      stack=["engine/experiment.py · _walkforward_predict", f"{MODEL} · Model.forecast", "statsmodels · apply → forecast"], nodes=["apply"], edges=[("roots", "apply")],
      formula=f"a_(t+1) = T · a_(t|t) + c,   예측 = Z · a_(t+1) ÷ {sc:g} = {pct(fc_ss, 4)}   (t = 결정월 {ap['date'][:7]}, 상태 {len(m_['a_last'])}개)",
      tensors=[{"type": "table", "title": t18, "columns": ["월", "실제 r", "이 폴드 계수의 필터 예측", "오차 v (실제 − 예측)", "워크포워드 예측"],
                "rows": [[d.strftime("%Y-%m") + (" (학습 구간)" if d in tr_set else ""), pct(x), pct(y_), pp(e, 3), pct(w_)]
                         for d, x, y_, e, w_ in zip(d6, hist.values[-6:], yh[-6:], vv[-6:], wf6)]}],
      checks=[chk("상태공간 손 계산 예측 = statsmodels forecast", fc_ss, ap["forecast"][0] / sc, 1e-12, k="독립"), chk("엔진이 낸 예측 = forecast ÷ 배율", ap["pred"], ap["forecast"][0] / sc, 1e-12, k="식"),
              chk(f"apply 입력에 비움 달({emb_s}) 포함 — 결정월까지의 변화율 전부", float(all(pd.Timestamp(d_ + "-01") in hist.index for d_ in emb_m)), 1.0, 0, k="식")]
             + ([chk(f"표의 워크포워드 예측 = 결과 파일 {mr['file'].replace('_daily.parquet', '')} 의 예측 (결정월 {dprev[0].strftime('%Y-%m')} ~ {dprev[-1].strftime('%Y-%m')})",
                     wf6, bd.reindex(dprev).values, 1e-15, k="재실행")] if mr else []))
    h = hist.values
    rec = (0.0 if seasonal else est[0]) - sum(aS[i] * h[-i] for i in range(1, len(aS))) + sum(maP[j] * vv[-j] for j in range(1, len(maP)))
    conv = FF[-1] / est[-1]
    same = abs(rec - ap["pred"]) < 1e-10
    ck = [chk("교과서 재귀식(ε 자리에 필터 오차 v) = 엔진 예측", rec, ap["pred"], 1e-10, k="독립")] if same else []
    rec2 = (0.0 if seasonal else est[0]) + float(np.dot(-aS[1:], h[::-1][:len(aS) - 1])) + float(np.dot(maP[1:], vv[::-1][:len(maP) - 1]))   # 같은 식을 내적으로
    if not same:
        ck.append(chk(f"재귀식 {pct(rec, 5)} (반복문) = 같은 식의 내적 계산 — 칼만 예측과의 차이 {pp(rec - ap['pred'], 4)}", rec, rec2, 1e-15, k="식"))
    lab19 = None
    if same and not seasonal:
        lab19 = lab_recursion(est[0], -aS[1:], maP[1:], [h[-i] for i in range(1, len(aS))], [vv[-j] for j in range(1, len(maP))],
                              [hist.index[-i].strftime("%Y-%m") for i in range(1, len(aS))], ap["date"][:7], ap["pred"])
    mz = min(abs(z) for z in roots(maP))
    S("8. 예측",
      ("교과서 식으로도 같은 값이 나오는지 봅니다: 예측 = " + ("c + " if not seasonal else "") + "Σ w_i × (i달 전 변화율) + Σ m_j × (j달 전 충격). 충격 ε 은 관측되지 않으므로 필터의 예측오차 v 를 씁니다. "
       + (f"이 폴드는 MA 근이 1 에서 충분히 멀어(가장 작은 |z| = {mz:.4f}) 필터가 수렴했고(마지막 F ÷ σ² = {conv:.4f}), 두 계산이 소수 10자리까지 같습니다." if same else
          f"이 폴드는 MA 근이 1 에 붙어 있어(가장 작은 |z| = {mz:.4f}) {ap['n_hist']}개월을 지나도 필터가 아직 수렴하지 않았습니다(마지막 F ÷ σ² = {conv:.4f}; 1 이면 수렴). "
          f"그래서 v 는 ε 과 같지 않고, 재귀식 {pct(rec, 4)} 와 칼만 예측 {pct(ap['pred'], 4)} 가 {pp(rec - ap['pred'], 4)} 다릅니다. 엔진은 칼만 예측(관측한 값이 주어졌을 때의 조건부 기댓값)을 씁니다.")),
      say=(("같은 값이 교과서 식으로도 나오는지 봅니다. 예측은 상수에 무게 곱하기 지난 변화율과 무게 곱하기 지난 충격을 더한 것이고, 충격 자리에는 필터의 예측오차를 씁니다. "
            f"이 폴드는 MA 근이 1에서 충분히 멀어 필터가 수렴했고, 마지막 예측오차 분산을 시그마 제곱으로 나눈 값이 {conv:.4f}입니다. 그래서 두 계산이 같습니다.") if same else
           ("같은 값이 교과서 식으로도 나오는지 봅니다. 예측은 무게 곱하기 지난 변화율과 무게 곱하기 지난 충격의 합이고, 충격 자리에는 필터의 예측오차를 씁니다. "
            f"그런데 이 폴드는 MA 근의 크기가 {mz:.4f}로 1에 붙어 있어, {ap['n_hist']}개월을 지나도 필터가 아직 수렴하지 않았습니다. "
            f"마지막 예측오차 분산을 시그마 제곱으로 나눈 값이 {conv:.3f}로, 아직 1까지 내려오지 않았습니다. 그래서 교과서 식은 {pct(rec, 4)}, 칼만 필터는 {pct(ap['pred'], 4)}로 서로 다릅니다. "
            "엔진은 관측값이 주어졌을 때의 조건부 기댓값인 칼만 필터의 예측을 씁니다.")),
      lab=lab19,
      stack=[f"{MODEL} · Model.forecast", "교과서 재귀식으로 다시 계산"], nodes=["apply"],
      formula="r̂_(t+1) = " + ("c + " if not seasonal else "") + "Σ_i w_i r_(t+1−i) + Σ_j m_j v_(t+1−j)   (w·m = 2장의 무게, t = 결정월)",
      tensors=[{"type": "num", "title": "두 계산", "items": [{"label": "재귀식", "value": pct(rec, 5)}, {"label": "칼만 필터 (엔진)", "value": pct(ap["pred"], 5)}, {"label": "마지막 F ÷ σ²", "value": f"{conv:.4f}"}]},
               {"type": "line", "title": f"apply 의 예측오차 분산 F_t ÷ σ² (1 로 내려오면 필터 수렴 → v ≈ ε)", "x": [d.strftime("%Y-%m") for d in hist.index[kd + 1:]], "xlabel": "월", "ylo": 0.95,
                "yhi": float(min(3.0, max(1.2, np.nanmax(FF[kd + 1:] / est[-1]) * 1.05))),
                "series": [{"name": "F ÷ σ²", "color": "#6a3fa0", "y": [num(x / est[-1], 5) for x in FF[kd + 1:]]}], "height": 190}],
      checks=ck)
    pred = ap["pred"]
    S("8. 예측",
      f"k = 1 이라 한 달 예측이 그대로 정답과 같은 단위입니다(_agg: Π(1 + r̂) − 1). 결정월 {ap['date'][:7]} 의 예측 변화율 <b>{pct(pred)}</b>, 실제 다음 달({nxt}) 변화율은 {pct(ap['actual'])} 였습니다. "
      f"참고로 지수로 바꾸면 F11 {f(ap['level'])} × (1 {'−' if pred < 0 else '+'} {abs(pred):.6f}) = {f(ap['level'] * (1 + pred))} 입니다(보기용 — 엔진은 변화율로 비교합니다). "
      f"<b>이 정답({nxt})은 봉인 구간({sealed_from.strftime('%Y-%m')} ~)의 값입니다</b> — 엔진은 결정일만 봉인 전으로 자릅니다. 다음 화면의 RMSE·방향 적중·DM 에도 이 한 달이 들어갑니다.",
      say=(f"한 달 앞만 맞히는 설정이라, 엔진의 묶기 함수는 이 예측을 그대로 정답과 같은 단위로 씁니다. 결정월 {ap['date'][:7]}의 예측 변화율은 {pct(pred)}이고, 실제 다음 달 변화율은 {pct(ap['actual'])}였습니다. "
           f"참고로 지수로 바꾸면, {f(ap['level'])}에 1 더하기 예측 변화율을 곱해 약 {ap['level'] * (1 + pred):.2f}입니다. 엔진은 이 지수가 아니라 변화율로 비교합니다. "
           "그리고 이 실제값은 봉인 구간의 첫 달 값입니다. 엔진이 결정일만 봉인 전으로 자르기 때문입니다."),
      notes=N((src.line(EXP, "return float(np.sum(np.log1p(fc))) if cfg.target.kind"), f"정답이 변화율(return)이라 Π(1 + r̂) − 1. k = {cap['config']['target']['k']} 이면 r̂ 그대로"),
              (src.line(EXP, "pred.loc[t] = _agg(np.asarray(m.forecast(hist, k + skip), dtype=float)[skip:])"), f"결정월 {ap['date'][:7]} 의 예측 {pct(pred, 4)} 기록")),
      code=C(EXP, src.lines(EXP, "def _agg(fc):", "return float(np.sum(np.log1p(fc))) if cfg.target.kind", "pred.loc[t] = _agg(np.asarray(m.forecast(hist, k + skip), dtype=float)[skip:])"),
             src.line(EXP, "return float(np.sum(np.log1p(fc))) if cfg.target.kind"), "_walkforward_predict · _agg", 2, 6),
      stack=["engine/experiment.py · _walkforward_predict · _agg"], nodes=["pred"], edges=[("apply", "pred")],
      formula="정답 예측 = Π_(h=1..k) (1 + r̂_(t+h)) − 1   (k = 1 이면 r̂ 그대로)",
      tensors=[{"type": "num", "title": f"결정월 {ap['date'][:7]}", "items": [{"label": "예측 변화율", "value": pct(pred, 4)}, {"label": "실제 변화율", "value": pct(ap["actual"], 4)},
                                                                       {"label": "예측 F11 (보기용)", "value": f(ap["level"] * (1 + pred), 4)}, {"label": "오차 (실제 − 예측)", "value": pp(ap["actual"] - pred, 4)}]}],
      checks=[chk("_agg(k=1) = 한 달 예측", float(np.prod(1.0 + np.array([ap["forecast"][0] / sc])) - 1.0), pred, 1e-15, k="식"),
              chk(f"실제 변화율 = {nxt} 의 r", ap["actual"], r_all.loc[t_next], 0, k="식")])
    yy = np.array([cap["y"].get(pd.Timestamp(x["test"]), np.nan) for x in af]); pr_all = np.array([x["pred"] for x in af])
    nv = np.array([r_all.get(pd.Timestamp(x["test"]), np.nan) for x in af])          # '이번 달과 같은 변화율' 예측 (결정월 t 의 r_t)
    ok = np.isfinite(yy) & np.isfinite(pr_all) & np.isfinite(nv)
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    hit = lambda q: float(np.mean(np.sign(q[ok]) == np.sign(yy[ok])))
    r_dev = r_all.dropna(); r_dev = r_dev[r_dev.index < sealed_from]
    ac1 = float(r_dev.autocorr(1))                                                    # 봉인 구간을 뺀 개발 구간의 1개월 자기상관
    e_m, e_n, e_0 = rmse(pr_all[ok] - yy[ok]), rmse(nv[ok] - yy[ok]), rmse(yy[ok])
    worse = e_m > e_n
    pm = rj["prediction_metrics"]                                                    # 없으면 멈춤(KeyError) — 대본이 이 값을 말함
    lab21 = lab_rmse(KN, yy[ok], pr_all[ok], nv[ok], pm["rmse"], pm["rmse_naive"], pm["rmse_zero"], f"결과 파일 {mr['file'].replace('_daily.parquet', '')}.json 의 prediction_metrics(엔진이 저장)")
    dm_p, dm_s = float(pm["dm_vs_naive_p"]), float(pm["dm_vs_naive"])
    from engine.backtest import diebold_mariano                                      # 엔진의 DM 함수(읽기만) — 결과 파일 값 재현 후 봉인 첫 달을 뺀 경우에 씀
    e_mod, e_nai = pr_all[ok] - yy[ok], nv[ok] - yy[ok]
    dm_s_re, dm_p_re = diebold_mariano(e_mod, e_nai, h=cap["config"]["target"]["k"])
    in_dev = np.array([pd.Timestamp(x["test"]) + pd.DateOffset(months=1) < sealed_from for x in af])[ok]   # 정답 달이 봉인 전인 폴드
    dm_s_dev, dm_p_dev = diebold_mariano(e_mod[in_dev], e_nai[in_dev], h=cap["config"]["target"]["k"])
    e_m_dev, e_n_dev = rmse(e_mod[in_dev]), rmse(e_nai[in_dev])
    pin(kind, "8장 봉인 첫 달 뺀 RMSE·DM p", f"{pct(e_m_dev, 4, False)} {pct(e_n_dev, 4, False)} {dm_p_dev:.3f}")
    worse_txt = "모델이 그대로보다 나쁜 쪽" if dm_s > 0 else "모델이 그대로보다 좋은 쪽"
    lvl = "5% 수준에서 유의합니다" if dm_p < 0.05 else ("5% 수준에서는 유의하지 않지만 10% 수준에서는 유의합니다" if dm_p < 0.10 else "10% 수준에서도 유의하지 않습니다")
    dm_txt = (f" 공식 판정(결과 파일의 prediction_metrics): RMSE 비율(모델 ÷ 그대로) {pm['rmse'] / pm['rmse_naive']:.3f}, naive 대비 DM 검정 통계량 {dm_s:+.2f}({worse_txt}), p = {dm_p:.3f} — {lvl}. "
              f"정답 달이 봉인 구간인 마지막 {int((~in_dev).sum())}개 폴드({nxt})를 빼고 {int(in_dev.sum())}개로 다시 재면 RMSE {KN} {pct(e_m_dev, 4, False)}, 그대로 {pct(e_n_dev, 4, False)}, "
              f"DM p = {dm_p_dev:.3f} 입니다(엔진의 DM 함수로 계산).")
    dm_say = (f"공식 판정에 쓰는 결과 파일의 DM 검정 p값은 {dm_p:.3f}로, " + {"5% 수준에서 유의합니다": "5% 수준에서 그 차이가 유의합니다.",
              "5% 수준에서는 유의하지 않지만 10% 수준에서는 유의합니다": "5% 수준에서는 유의하지 않지만 10% 수준에서는 유의합니다. 곧 그대로 기준보다 나쁜 쪽이며, 유의성은 5%와 10% 사이의 경계선입니다." if dm_s > 0 else
              "5% 수준에서는 유의하지 않지만 10% 수준에서는 유의합니다.",
              "10% 수준에서도 유의하지 않습니다": "그 차이가 유의하지는 않습니다."}[lvl])
    S("8. 예측",
      f"{len(af)}개 폴드의 한 달 예측(각 폴드의 테스트 1개월)을 실제와 나란히 놓았습니다. 비교 기준 두 가지를 함께 둡니다: '항상 0' 과 '이번 달 변화율이 다음 달에도 그대로'. "
      f"F11 변화율은 전월과의 상관이 {ac1:.2f}(봉인 전 {r_dev.index[0].strftime('%Y-%m')} ~ {r_dev.index[-1].strftime('%Y-%m')}) 로 매우 매끄러워서(관성), '그대로' 기준도 방향을 {hit(nv) * 100:.1f}% 맞힙니다 — 방향 적중이 높은 것은 이 모델만의 능력이 아니라 자료의 성질이므로 단순 기준과 비교해야 합니다. "
      f"RMSE: {KN} {pct(e_m, 3, False)}, 그대로 {pct(e_n, 3, False)}, 0 예측 {pct(e_0, 3, False)}"
      + (" — <b>이 설정에서는 '그대로' 보다 오차가 큽니다.</b>" if worse else ".")
      + " '그대로' 는 결과 화면의 naive(지난 변화 그대로) 기준과 같은 것입니다(k = 1)." + dm_txt,
      say=(f"{len(af)}개 폴드의 한 달 예측을 실제와 나란히 놓았습니다. 기준은 항상 0이라는 예측과, 이번 달 변화율이 다음 달에도 그대로라는 예측입니다. "
           f"F11 변화율은 전월과의 상관이 {ac1:.2f}로 매끄러워서, 그대로 기준도 방향을 {hit(nv) * 100:.1f}% 맞힙니다. 그래서 방향 적중만으로 평가하면 안 됩니다. "
           f"RMSE 값은 {KN} {pct(e_m, 3, False)}, 그대로 {pct(e_n, 3, False)}, 0 예측 {pct(e_0, 3, False)}로, "
           + (f"이 설정에서 {KN}는 그대로 기준보다 오차가 큽니다. " if worse else f"이 설정에서 {KN}가 그대로 기준보다 오차가 작습니다. ")
           + dm_say),
      lab=lab21,
      stack=["tools/codewalk/series_capture.py · 폴드마다 같은 엔진 경로"], nodes=["pred"],
      tensors=[{"type": "line", "title": f"{len(af)}개 폴드의 예측과 실제 (x = 결정월, 값 = 다음 달 변화율; 마지막 점 {nxt} 은 봉인 구간의 값)", "x": [x["test"][:7] for x in af], "xlabel": "결정월",
                "series": [{"name": "실제", "color": "#3b6fb6", "y": [num(v_) for v_ in yy], "dots": True}, {"name": KN, "color": "#c8710f", "y": [num(v_) for v_ in pr_all], "dots": True}], "height": 230},
               {"type": "table", "title": f"요약 ({int(ok.sum())}개 폴드, 원래 단위) — 공식 판정은 결과 화면의 naive 대비 비율·DM 검정", "columns": ["예측 방법", "RMSE", "방향 적중"],
                "rows": [[KN, pct(e_m, 4, False), f"{hit(pr_all) * 100:.1f}%"], ["이번 달 변화율 그대로", pct(e_n, 4, False), f"{hit(nv) * 100:.1f}%"],
                         ["항상 0", pct(e_0, 4, False), "—"]]}],
      checks=[chk("실제값(정답) = 다음 달 변화율 r(t+1) (정답 정의 확인)", yy[ok], r_all.shift(-1).reindex([pd.Timestamp(x["test"]) for x in af]).values[ok], 0, k="식"),
              chk("영상의 RMSE(모델·그대로·0) = 결과 파일 prediction_metrics (rmse·rmse_naive·rmse_zero)", [e_m, e_n, e_0], [pm["rmse"], pm["rmse_naive"], pm["rmse_zero"]], 1e-12, k="재실행"),
              chk(f"엔진의 DM 함수로 다시 계산한 통계량·p = 결과 파일 dm_vs_naive·dm_vs_naive_p ({int(ok.sum())}개)", [dm_s_re, dm_p_re], [dm_s, dm_p], 1e-12, k="재실행"),
              chk_ok(f"봉인 첫 달 1개만 뺐음({int(in_dev.sum())}개) — 그 달 = {nxt}", int((~in_dev).sum()) == 1 and not in_dev[-1] and np.isfinite(dm_p_dev), k="식")])
    if mr["max_diff"] != 0:
        raise AssertionError(f"영상의 예측이 결과 파일 {mr['file']} 과 다릅니다(최대 {mr['max_diff']:.1e}) — 8장 대본은 '같은 계산'을 말합니다")
    S("8. 예측",
      (f"이 영상의 {len(af)}개 예측을 결과 폴더의 같은 실험 결과 <b>{mr['file'].replace('_daily.parquet', '')}</b> 의 예측과 맞춰 봅니다: 겹치는 {mr['n']}개, 최대 차이 {mr['max_diff']:.1e}"
       " — <b>결과 화면의 숫자와 같은 계산</b>입니다."
       f" 옛 결과 파일 {old['file']} 은 {leg_txt} 설정(결과 파일에 그 기록이 없어 엔진이 LEGACY_DEFAULTS 로 채움, 3장)이라 값이 다른 것이 정상입니다 — "
       f"이 설정으로 {old['n']}개 폴드를 모두 다시 추정하면 옛 파일의 예측과 최대 {old['d_old']:.1e} 차이로 같습니다(검증). "
       "3장의 배율 1 비교는 제약을 켠 것이라, 3장의 예측 차이와 이 표의 차이는 크기가 다릅니다."),
      say=(f"마지막으로, 이 영상의 {len(af)}개 예측을 결과 폴더에 저장된 같은 실험의 결과 파일과 비교합니다. "
           f"겹치는 {mr['n']}개 예측의 최대 차이는 0으로, 결과 화면의 숫자와 같은 계산입니다. "
           f"{leg_say} 설정으로 만든 옛 결과 파일은 값이 다른 것이 정상이고, 그 설정으로 다시 추정하면 옛 파일의 예측도 그대로 나옵니다."),
      stack=["results/forecast/" + mr["file"]], nodes=["pred"],
      tensors=[{"type": "table", "title": f"결과 파일별 비교 (최대 차이는 변화율의 차이, %p; 옛 결과는 {leg_txt} 설정이라 다른 것이 정상)", "columns": ["결과 파일", "겹친 예측 수", "최대 차이"],
                "rows": [[a_.replace("_daily.parquet", ""), n_, f"{d_ * 100:.2f}%p"] for a_, n_, d_ in mr["all"]]}],
      checks=[chk(f"영상의 예측 = 결과 파일 {mr['file'].replace('_daily.parquet', '')} 의 예측 ({mr['n']}개)", mr["max_diff"], 0.0, 1e-12, k="재실행"),
              chk(f"옛 파일 설정({', '.join(f'{k_}={v_}' for k_, v_ in old['miss'].items())})으로 {old['n']}개 폴드를 엔진 모델 파일로 다시 추정·예측 = 옛 결과 파일 {old['file']} 의 예측",
                  old["d_old"], 0.0, 1e-12, k="재실행")])
    # ================================================================ 9. 정리
    n_ok = sum(c["ok"] for c in checks_all)
    if n_ok != len(checks_all):
        raise AssertionError(f"검증 실패: {[c for c in checks_all if not c['ok']]}")
    cnt = {k_: sum(1 for c in checks_all if c["kind"] == k_) for k_ in KIND}
    all_gain = n_gain == len(af)
    gmax_txt = (f"이 영상의 확인 방법으로 {len(af)}개 중 {n_up01}개 폴드에서 로그우도가 0.01 넘게 높은 점이 있었고, 마지막 폴드도 더 넓게 찾으면 {W['d']:+.3f} 높은 점이 있으며 그 점의 예측은 "
                f"{pct(W['pr'], 3)} 로 부호가 다릅니다 — 이 영상의 계수·예측은 엔진이 실제로 낸 값이지 가능도의 최댓값이 아님, 5장" if not seasonal else "마지막 폴드도 최댓값은 가역성 경계 쪽, 5장")
    S("9. 정리",
      f"정리: {KN} 는 F11 의 월 변화율 하나로, 학습 구간에서 상태공간·칼만 필터의 로그우도를 L-BFGS 로 최대화해 계수를 정하고(정상·가역 제약은 u → ρ → 계수 변환으로 보장), "
      f"결정월마다 그달까지의 변화율을 필터에 통과시켜 한 달 앞을 예측합니다. 2026-11-07 부터는 변화율 × {sc:g} 로 추정해 최적화의 수치 문제를 줄였습니다 — "
      f"제약을 켠 채 배율 1 과 비교해 {len(af)}개 폴드 " + ("모두" if all_gain else f"중 {n_gain}개") + f"에서 로그우도가 올랐습니다(같은 배율 1 모형으로 재도 {n_gs}개, 3장). "
      f"다만 배율 {sc:g} 도 전역 최댓값을 보장하지는 않습니다({gmax_txt}). "
      f"이 영상의 검증 <b>{len(checks_all)}줄</b> — {KIND['독립']} {cnt['독립']} · {KIND['식']} {cnt['식']}(같은 배열·같은 식·정의) · {KIND['재실행']} {cnt['재실행']} · "
      f"{KIND['탐색']} {cnt['탐색']}(최적화를 새로 돌려 찾아본 것) · {KIND['실습']} {cnt['실습']}, 모두 통과. "
      "각 검증 줄 앞의 [ ] 가 종류입니다.",
      say=(f"정리합니다. {KN}는 F11의 월 변화율 하나로, 상태공간과 칼만 필터로 계산한 로그우도를 L-BFGS로 최대화해 계수를 정합니다. "
           "정상성과 가역성은 u를 계수로 바꾸는 변환으로 지키고, 결정월마다 그달까지의 변화율을 필터에 통과시켜 한 달 앞을 예측합니다. "
           f"2026-11-07부터 변화율에 {sc:g}을 곱해 추정하면서, 제약을 켠 채 배율 1과 비교해 {len(af)}개 폴드 " + ("모두에서" if all_gain else f"가운데 {n_gain}개에서") + " 로그우도가 올랐습니다. "
           + ("다만 5장에서 보았듯이 마지막 폴드를 포함해 여러 폴드에서 로그우도가 더 높은 점이 있어, 이 영상의 계수와 예측은 가능도의 최댓값이 아닙니다. "
              if not seasonal else "다만 이것이 전역 최댓값을 보장하지는 않습니다. ")
           + f"이 영상의 검증은 {len(checks_all)}줄입니다. 다른 경로와 맞춘 독립 대조가 {cnt['독립']}줄, 같은 식이나 정의를 확인한 것이 {cnt['식']}줄, "
           f"다시 실행해 기록이나 결과 파일과 맞춘 것이 {cnt['재실행']}줄, 최적화를 새로 돌려 찾아본 것이 {cnt['탐색']}줄, 직접 실행해 보기가 {cnt['실습']}줄이고, 모두 통과했습니다."),
      stack=["요약"], nodes=ALL,
      tensors=[{"type": "table", "title": "검증 전체 (장 = 본문의 'n장', 화면 = 플레이어의 화면 번호; [ ] = 검증의 종류)", "columns": ["장 (화면)", "무엇", "결과"],
                "rows": [[f"{steps[c['step'] - 1]['chapter'] if c['step'] <= len(steps) else '9. 정리'} (화면 {c['step']})", c["what"], "✓ 통과" if c["ok"] else "✗ 실패"] for c in checks_all]}])
    if n_ok != len(checks_all):
        raise AssertionError(f"검증 실패: {[c for c in checks_all if not c['ok']]}")

    def clean(o):  # numpy 정수·실수·bool → 파이썬 기본형 (JSON)
        if isinstance(o, dict): return {k_: clean(v_) for k_, v_ in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v_) for v_ in o]
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        return o
    steps[:] = clean(steps)
    used = sorted({s_["code"]["file"] for s_ in steps if s_.get("code")})
    return {"title": f"{KN} 코드 따라가기 — 실제로 계산되는 순서", "model": kind, "steps": steps, "sources": {fn: src.text[fn] for fn in used}, "diagram": diagram(), "checks": clean(checks_all)}


def diagram() -> dict:
    labels = [("lev", "F11 수준", "월별 지수"), ("ret", "변화율", "pct_change"), ("train", "학습 구간", "폴드의 r"), ("scale", "× 배율", "수치 안정"), ("ssm", "상태공간", "T·R·Z·초기값"),
              ("cons", "제약 변환", "u → ρ → φ"), ("mle", "최우추정", "L-BFGS 반복"), ("kf", "칼만 필터", "로그우도"), ("roots", "근 점검", "정상·가역"),
              ("apply", "필터 다시", "그달까지 전부"), ("pred", "예측", "÷ 배율 · _agg")]
    nodes = [{"id": i, "label": l, "x": 10 + 170 * k, "y": 14, "w": 150, "h": 52, "sub": s} for k, (i, l, s) in enumerate(labels)]
    E = [{"from": a[0], "to": b_[0]} for a, b_ in zip(labels[:-1], labels[1:])]
    return {"width": 10 + 170 * len(labels), "height": 84, "nodes": nodes, "edges": E, "groups": []}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="arima", choices=["arima", "sarima"])
    ap.add_argument("--capture"); ap.add_argument("--out")
    a = ap.parse_args()
    cap = pickle.load(open(a.capture or ROOT / f"docs/guide_build/walk/{a.model}_capture.pkl", "rb"))
    sc = build(cap)
    out = build_html(sc, Path(a.out) if a.out else ROOT / f"docs/guide_build/walk/{a.model}.html")
    print("저장:", out, "· 단계", len(sc["steps"]), "· 검증", sum(c["ok"] for c in sc["checks"]), "/", len(sc["checks"]), "· 크기", round(out.stat().st_size / 1e6, 2), "MB")
