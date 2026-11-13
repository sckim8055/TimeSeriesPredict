"""VECM 코드 따라가기 대본 — vecm_capture 의 실제 실행 기록으로 단계별 화면을 만든다.

  python -m tools.codewalk.vecm_script [--capture docs/guide_build/walk/vecm_capture.pkl] [--out docs/guide_build/walk/vecm.html]

수준값 행렬 → 단위근 → Johansen(축소 순위 회귀: R₀·R₁·S 행렬·고유값) → β·α·Γ → 결정일 재귀 예측 → 모든 폴드.
β·α·Γ 와 예측을 Lütkepohl(2005) 의 식(= statsmodels VECM 이 쓰는 절차)대로 numpy 로 다시 계산해 맞춰 본다. 다르면 대본을 만들지 않는다.
2026-11-08: say(음성·자막 대본, 숫자는 화면과 같은 변수에서) · notes(코드 줄 설명) · lab(직접 실행해 보기 — 대본을 만들 때 실제로 실행해 출력 = 엔진 값 확인).
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tools.codewalk.common import ROOT, Sources, build_html, match_result, num, py_list, run_lab

EXP = "engine/experiment.py"; MODEL = "models/vecm.py"; CAL = "engine/calendar.py"
# 음성·자막에서 부르는 변수 이름(화면 표에는 변수 사전 이름이 그대로 나옴). 설정이 바뀌면 멈춘다 — 이름을 새로 정해야 함.
SPOKEN = {"F12": "전세가격지수", "M3": "통화량(M2)", "N19": "가계대출", "KR3Y": "국고채 3년 금리"}   # "통화량(M2)" → 자막엔 M2, 음성은 "통화량"(괄호 속 코드 이름은 읽지 않음)
# 검증 줄의 종류(공통 규칙 1 — 정리 화면에서 나눠 셈):
#   독립 = 다른 경로로 대조(원자료에서 따로 찾기, statsmodels 의 다른 구현·표현, 다른 계산법)
#   식 = 같은 배열에 같은 식을 다시 계산했거나, 엔진(statsmodels VECM.fit)과 같은 순서의 계산을 옮겨 재현한 것 — 직접 실행해 보기는 모두 여기
#   재실행 = 같은 함수를 다시 불러 기록과 맞춘 것
KIND = {"독립": "독립 대조(다른 경로)", "식": "식 확인·재현(같은 식·같은 순서)", "재실행": "재실행 = 기록", "출처": "출처 확인(개수에 넣지 않음)"}
# 대본(say)·자막은 빌드하는 컴퓨터에 따라 달라지면 안 된다(녹음은 읽을 문장의 지문으로 연결됨). 그래서
#  - 원 노트북 2 의 사실은 고정 문장으로 쓰고(2026-11-08 업로드된 노트북을 읽어 확인), 파일이 있을 때만 '출처 확인' 줄로 다시 대조한다 — 없으면 그 줄만 '건너뜀'.
#  - 변수 사전(data/variable_master.xlsx)은 대본이 그 값(F11 발표 시점 메모, 가계대출 시차 미확인)을 말하므로 필수 — 없으면 빌드를 멈춘다.
#  - 결과 폴더에 이 실험의 결과 파일이 생기면 대본의 '대조하지 못했다' 가 틀리게 되므로 빌드를 멈춘다(대본을 고쳐 대조 단계를 넣을 것).
NB2_FACT = {"call": "VECM(훈련다, k_ar_diff=2, coint_rank=1).fit()", "k_ar_diff": 2, "coint_rank": 1, "det": "n", "det_default": True, "cols": ["F11", "F7", "K1"], "steps": 12}
NB2 = [Path("/mnt/user-data/uploads/시계열기초강좌_기획/코랩_노트북/노트북2_통계마법_ARIMA_SARIMA_VECM.ipynb"),   # 원 노트북 2 (있으면 VECM 칸을 읽어 설정 비교)
       ROOT / "legacy" / "시계열기초강좌_기획" / "코랩_노트북" / "노트북2_통계마법_ARIMA_SARIMA_VECM.ipynb"]


def pct(v, d=3, sign=True):
    return "NaN" if v is None or not np.isfinite(v) else (f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%")


def nat_months(n: int) -> str:
    """음성용 달 수: 1 → 한 달, 2 → 두 달, 3 → 세 달 ('이 개월' 이 조사처럼 들리지 않게). 그 밖은 'n개월'."""
    return {1: "한 달", 2: "두 달", 3: "세 달", 4: "네 달"}.get(int(n), f"{int(n)}개월")


def months_before(t, d) -> int:
    t, d = pd.Timestamp(t), pd.Timestamp(d)
    return (t.year - d.year) * 12 + t.month - d.month


def master_info(vids):
    """변수 사전(data/variable_master.xlsx)의 발표시차·확인 여부·메모 — 실행 기록에 대상의 시차와 확인 여부가 없어 사전에서 읽는다.
    대본이 이 값을 말하므로 필수: 없거나 읽지 못하면 빌드를 멈춘다(다른 문장을 만들지 않음)."""
    from engine.registry import Registry
    fp = ROOT / "data" / "variable_master.xlsx"
    try:
        reg, _ = Registry.load_master(fp)
        return {v_: {"lag": int(reg.get(v_).lag_days), "ok": bool(reg.get(v_).lag_confirmed), "note": reg.get(v_).note or "", "transform": reg.get(v_).transform} for v_ in vids}
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"VECM 대본: 변수 사전 {fp} 를 읽지 못했습니다 — 대본이 사전 값(발표 시점 메모·시차 확인 여부)을 말하므로 빌드를 멈춥니다 ({e})") from e


def store_raw(version: str, ids: list[str]) -> dict:
    """데이터 저장소의 변수 parquet(data/versions/<버전>/variables_*.parquet)을 pandas 로 직접 읽어 변수별 (기준일 → 값) 시리즈를 만든다.
    엔진의 DataVersion.wide(pivot_table)를 거치지 않는 독립 경로(공통 규칙 1). 검증에 필요하므로 파일이 없으면 멈춘다. 같은 (변수, 기준일)이 두 번 있으면 멈춘다."""
    d = ROOT / "data" / "versions" / version
    fs = sorted(d.glob("variables_*.parquet"))
    if not fs:
        raise RuntimeError(f"VECM 대본: 데이터 저장소 {d} 에 variables_*.parquet 가 없습니다 — 원자료 대조를 할 수 없어 빌드를 멈춥니다")
    df = pd.concat([pd.read_parquet(f_, columns=["variable_id", "date", "value"]) for f_ in fs], ignore_index=True)
    df = df[df["variable_id"].isin(ids)]
    if df.duplicated(["variable_id", "date"]).any():
        raise AssertionError("원자료에 같은 (변수, 기준일)이 두 번 있음 — 직접 읽기 규칙을 정해야 합니다")
    out = {v_: df[df["variable_id"] == v_].set_index("date")["value"].astype(float).sort_index() for v_ in ids}
    for v_, s_ in out.items():
        if s_.empty:
            raise RuntimeError(f"VECM 대본: 원자료에 {v_} 가 없습니다")
        s_.index = pd.DatetimeIndex(s_.index).normalize()
    return {"dir": d.relative_to(ROOT).as_posix(), "series": out}


def notebook2():
    """원 노트북 2 의 VECM 칸을 (있으면) 읽어 설정을 돌려준다 — 파일은 자료로만 읽고 실행하지 않는다. 없으면 None."""
    import inspect
    import json
    import re
    from statsmodels.tsa.vector_ar.vecm import VECM
    for fp in NB2:
        if not fp.exists():
            continue
        nb = json.loads(fp.read_text(encoding="utf-8"))
        for c in nb.get("cells", []):
            s_ = "".join(c.get("source", []))
            m = re.search(r"VECM\(([^)]*)\)\.fit\(\)", s_) if c.get("cell_type") == "code" else None
            if not m:
                continue
            args = m.group(1)
            kd, cr = re.search(r"k_ar_diff=(\d+)", args), re.search(r"coint_rank=(\d+)", args)
            det = re.search(r"deterministic=['\"](\w+)['\"]", args)
            cols_ = re.findall(r"'(\w+)'", (re.search(r"\[\[([^\]]*)\]\]", s_) or re.search("()", "")).group(1))
            steps_ = re.search(r"predict\(steps=(\d+)\)", s_)
            return {"file": fp.name, "call": m.group(0), "k_ar_diff": int(kd.group(1)) if kd else None, "coint_rank": int(cr.group(1)) if cr else None,
                    "det": det.group(1) if det else inspect.signature(VECM.__init__).parameters["deterministic"].default, "det_default": det is None,
                    "cols": cols_, "steps": int(steps_.group(1)) if steps_ else None}
    return None


def g(v, d=4):
    return "NaN" if v is None or not np.isfinite(v) else f"{v:.{d}g}"


def johansen(Z, p, r):
    """Lütkepohl(2005) 7.2 의 축소 순위 최우추정 — 결정항 없음(n)·계절 없음. statsmodels VECM._estimate_vecm_ml 과 같은 순서의 계산."""
    y = Z.T; K = y.shape[0]; P = p + 1
    y_1_T = y[:, P:]; T = y_1_T.shape[1]; dy = np.diff(y); dy_1_T = dy[:, P - 1:]; y_lag1 = y[:, P - 1:-1]
    dx = np.zeros((p * K, T))
    for j in range(T):
        dx[:, j] = np.concatenate([dy[:, P + j - 2 - i] for i in range(p)])     # [Δy_(t−1), …, Δy_(t−p)]
    xx_inv = np.linalg.inv(dx @ dx.T)
    res = lambda M: M - M @ dx.T @ xx_inv @ dx
    R0, R1 = res(dy_1_T), res(y_lag1)
    S00, S01, S11 = R0 @ R0.T / T, R0 @ R1.T / T, R1 @ R1.T / T
    u_, s_, v_ = np.linalg.svd(S11, full_matrices=False); S11_ = np.linalg.inv(u_ @ (np.sqrt(s_)[:, None] * v_))
    A = S01 @ S11_
    lam, V = np.linalg.eig(A.T @ np.linalg.inv(S00) @ A)
    o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    beta = np.real_if_close((V[:, :r].T @ S11_).T); beta = beta @ np.linalg.inv(beta[:r])
    alpha = S01 @ beta @ np.linalg.inv(beta.T @ S11 @ beta)
    gamma = (dy_1_T - alpha @ beta.T @ y_lag1) @ dx.T @ xx_inv
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "lam": np.real(lam), "T": T, "S00": S00, "S01": S01, "S11": S11, "R0": R0, "R1": R1, "dx": dx}


# ---------------------------------------------------------------- 직접 실행해 보기(실험 코드) — 2026-11-08
# 실행 기록의 숫자를 그대로 넣은, 혼자 도는 파이썬 코드. 대본을 만들 때 실제로 실행해 출력이 엔진(statsmodels) 값과 글자까지 같은지 확인한다(common.run_lab).
def _pl(vals, sig=17):
    return py_list([float(v) for v in vals], sig)


def _rows(M_, ind="    "):
    return "[\n" + "".join(f"{ind}{_pl(r_)},\n" for r_ in M_) + "]"


# Johansen 실습(numpy) 두 개가 같이 쓰는 앞부분: Δy_t·y_(t−1)·[Δy_(t−1); Δy_(t−2)] 만들기 — johansen() 과 같은 순서
_LAB_DIFFS = ("y = Y.T                            # 변수 × 달\n"
              "K, P = y.shape[0], p + 1\n"
              "dy = np.diff(y)                    # Δy (이웃한 두 달의 차이)\n"
              "dY = dy[:, P - 1:]                 # Δy_t        — 식의 왼쪽\n"
              "Y1 = y[:, P - 1:-1]                # y_(t−1)     — 지난달 수준\n"
              "T = dY.shape[1]                    # 쓸 수 있는 식의 수\n"
              "dX = np.zeros((p * K, T))          # [Δy_(t−1); Δy_(t−2)] — 지난 두 달의 변화\n"
              "for j in range(T):\n"
              "    dX[:, j] = np.concatenate([dy[:, P + j - 2 - i] for i in range(p)])\n"
              "XXi = np.linalg.inv(dX @ dX.T)\n")


def lab_trace(lam, T, cv95, trace_e, sel_e):
    code = ("import math\n\n"
            "# 마지막 폴드 학습 창의 고유값 λ (큰 순서) — statsmodels coint_johansen, 실행 기록 그대로\n"
            f"lam = {_pl(lam)}\n"
            f"T = {int(T)}          # 식의 수 = 학습 행 수 − (차분 시차 + 1)\n"
            "# 귀무가설 '균형식 ≤ r' 마다의 5% 임계값 (결정항 없음, coint_johansen 의 표)\n"
            f"cv95 = {_pl(cv95)}\n\n"
            "chosen = None\n"
            "for r in range(len(lam)):\n"
            "    stat = 0.0\n"
            "    for i in range(r, len(lam)):\n"
            "        stat += -T * math.log(1 - lam[i])     # 대각합 통계량 = −T Σ_(i>r) ln(1 − λ_i)\n"
            "    reject = stat > cv95[r]\n"
            "    print(\"균형식 ≤\", r, f\"통계량 {stat:.3f}\", f\"임계값 {cv95[r]:.3f}\", \"기각\" if reject else \"기각 못 함\")\n"
            "    if not reject and chosen is None:\n"
            "        chosen = r                             # 처음으로 기각하지 못한 r\n"
            "print(\"고른 r =\", chosen if chosen is not None else len(lam))\n")
    exp = "\n".join(f"균형식 ≤ {i} 통계량 {trace_e[i]:.3f} 임계값 {cv95[i]:.3f} {'기각' if trace_e[i] > cv95[i] else '기각 못 함'}" for i in range(len(lam))) + f"\n고른 r = {sel_e}"
    return {"title": "대각합 통계량과 균형식 개수 고르기를 직접 계산",
            "desc": "고유값으로 '균형식 ≤ r' 마다 −T Σ ln(1 − λ) 를 더하고 5% 임계값과 비교해, 처음으로 기각하지 못한 r 을 고릅니다. 통계량은 statsmodels(coint_johansen) 값과 같아야 합니다. "
                    "이 설정은 r 을 1 로 고정하므로 엔진은 이 계산을 하지 않습니다(정보용).",
            "kind": "식", "ref_label": "statsmodels(coint_johansen) 값 — 같은 λ·같은 식", **run_lab(code, exp, pure=True)}


def lab_johansen(Z, names, p, r, beta_e, alpha_e, d_layout):
    code = ("import numpy as np\n\n"
            f"# 마지막 폴드의 학습 행렬 — {Z.shape[0]}행 × {Z.shape[1]}열 수준값 {names}, 실행 기록 그대로\n"
            f"names = {names!r}\n"
            f"Y = np.asfortranarray({_rows(Z)})   # 엔진(판다스 .values)과 같은 열 우선 메모리 순서 — 설명의 '주의' 참고\n"
            f"p, r = {p}, {r}        # 차분 시차 수(k_ar_diff), 균형식 개수(coint_rank). 상수항 없음(deterministic = 'n')\n\n"
            + _LAB_DIFFS +
            "R0 = dY - dY @ dX.T @ XXi @ dX      # ① 단기 움직임을 걷어 낸 잔차\n"
            "R1 = Y1 - Y1 @ dX.T @ XXi @ dX\n"
            "S00, S01, S11 = R0 @ R0.T / T, R0 @ R1.T / T, R1 @ R1.T / T   # ②\n\n"
            "u, s, v = np.linalg.svd(S11, full_matrices=False)\n"
            "S11_h = np.linalg.inv(u @ (np.sqrt(s)[:, None] * v))         # S₁₁^(−½)\n"
            "A = S01 @ S11_h\n"
            "lam, V = np.linalg.eig(A.T @ np.linalg.inv(S00) @ A)         # ③ 고유값 문제\n"
            "o = np.argsort(lam)[::-1]\n"
            "lam, V = lam[o], V[:, o]                                      # 큰 순서\n\n"
            "beta = np.real_if_close((V[:, :r].T @ S11_h).T)\n"
            "beta = beta @ np.linalg.inv(beta[:r])                         # 첫 원소(F11) = 1\n"
            "alpha = S01 @ beta @ np.linalg.inv(beta.T @ S11 @ beta)\n\n"
            "for name, b, a in zip(names, beta[:, 0], alpha[:, 0]):\n"
            "    print(name, f\"β {b:.5g}\", f\"α {a:.5g}\")\n")
    exp = "\n".join(f"{n_} β {b_:.5g} α {a_:.5g}" for n_, b_, a_ in zip(names, beta_e, alpha_e))
    return {"title": "β·α 를 요한센 절차로 직접 계산 (numpy)",
            "desc": "학습 행렬에서 잔차 R₀·R₁ → S 행렬 → 고유값 문제 → β 정규화 → α 순서로, statsmodels VECM.fit 과 같은 계산을 합니다. 출력이 엔진이 적합한 β·α 와 같아야 합니다. "
                    "numpy 가 필요해 Python Tutor 대신 Colab 등에 붙여 실행하세요. "
                    f"주의: M2·가계대출의 단위가 커서 이 계산은 반올림 오차에 민감합니다. 같은 숫자를 행 우선(np.array) 배열로 넣기만 해도 β 가 상대 {d_layout:.1e} 달라져 다섯째 자리가 바뀝니다. "
                    "그래서 엔진과 같은 열 우선 배열을 썼고, 다른 컴퓨터(다른 선형대수 라이브러리)에서는 마지막 자리가 다를 수 있습니다.",
            "kind": "식", "ref_label": "엔진 값 — VECM.fit 과 같은 순서의 재현", **run_lab(code, exp, pure=False)}


def lab_gamma(Z, names, p, alpha, beta, gamma_e):
    K = Z.shape[1]
    code = ("import numpy as np\n\n"
            f"# 마지막 폴드의 학습 행렬({Z.shape[0]}행 × {K}열)과 엔진이 적합한 α·β — 실행 기록 그대로\n"
            f"names = {names!r}\n"
            f"Y = np.asfortranarray({_rows(Z)})   # 엔진과 같은 열 우선 메모리 순서\n"
            f"alpha = np.array({_rows(alpha)})\n"
            f"beta = np.array({_rows(beta)})\n"
            f"p = {p}\n\n"
            + _LAB_DIFFS +
            "# α·β 로 설명한 부분(오차수정항)을 빼고, 남은 변화를 지난 두 달의 변화에 최소제곱 회귀\n"
            "G = (dY - alpha @ beta.T @ Y1) @ dX.T @ XXi           # K × (p·K) = [Γ₁ | Γ₂]\n\n"
            "for i in range(p):\n"
            "    for j in range(K):\n"
            "        print(f\"Γ{i + 1}\", \"F11 ←\", names[j], f\"{G[0, i * K + j]:.5g}\")   # F11 행만\n")
    exp = "\n".join(f"Γ{i + 1} F11 ← {names[j]} {gamma_e[0, i * K + j]:.5g}" for i in range(p) for j in range(K))
    return {"title": "Γ 를 최소제곱으로 직접 계산 (numpy) — F11 행",
            "desc": "α·β 를 정한 뒤, 이번 달 변화에서 오차수정항 몫을 빼고 지난 두 달의 변화에 회귀합니다. 출력(F11 이 이번 달 변할 때 각 변수의 지난 변화에 붙는 계수)이 엔진의 Γ 와 같아야 합니다. "
                    "numpy 가 필요해 Colab 등에서 실행하세요.",
            "kind": "식", "ref_label": "엔진 값 — 같은 식", **run_lab(code, exp, pure=False)}


def lab_forecast(rows3, dates3, names, alpha, beta, G1, G2, pred_e, v_t):
    code = (f"# 결정일 {dates3[-1]}: 마지막 3행({', '.join(dates3)})의 수준값 {names} — 실행 기록 그대로\n"
            f"y_2 = {_pl(rows3[0])}   # {dates3[0]}\n"
            f"y_1 = {_pl(rows3[1])}   # {dates3[1]}\n"
            f"y_0 = {_pl(rows3[2])}   # {dates3[2]} (결정일)\n"
            "# 엔진이 학습 창에서 적합한 α, β (열 하나), Γ₁, Γ₂ (행 = 이번 달 변하는 변수, 열 = 지난 변화의 변수)\n"
            f"alpha = {_pl(alpha)}\n"
            f"beta = {_pl(beta)}\n"
            f"G1 = {_rows(G1)}\n"
            f"G2 = {_rows(G2)}\n\n"
            "K = len(y_0)\n"
            "dy1 = [y_0[j] - y_1[j] for j in range(K)]     # Δy_t      (이번 달 변화)\n"
            "dy2 = [y_1[j] - y_2[j] for j in range(K)]     # Δy_(t−1)  (지난달 변화)\n\n"
            "ect = 0.0\n"
            "for j in range(K):\n"
            "    ect += beta[j] * y_0[j]                   # 오차수정항 β′y_t (상수 없음)\n\n"
            "y_next = []\n"
            "for i in range(K):\n"
            "    d = alpha[i] * ect                        # 되돌림 몫\n"
            "    for j in range(K):\n"
            "        d += G1[i][j] * dy1[j] + G2[i][j] * dy2[j]   # 단기 움직임 몫\n"
            "    y_next.append(y_0[i] + d)                 # ŷ_(t+1) = y_t + Δŷ\n\n"
            "pred = y_next[0] / y_0[0] - 1                 # 엔진: 대상 수준 → 변화율\n"
            "print(\"다음 달 F11\", f\"{y_next[0]:.7g}\")\n"
            "print(\"예측 변화율\", f\"{pred:.5g}\")\n")
    exp = f"다음 달 F11 {v_t * (1 + pred_e):.7g}\n예측 변화율 {pred_e:.5g}"
    return {"title": "결정일의 예측을 직접 계산",
            "desc": "오차수정항 β′y_t 를 만들고, 변수마다 α × 오차수정항 + Γ₁·이번 달 변화 + Γ₂·지난달 변화를 더해 다음 달 수준을 냅니다. 마지막에 엔진처럼 F11 예측 ÷ 지금 F11 − 1 로 변화율을 만듭니다. "
                    "출력이 엔진이 기록한 예측과 같아야 합니다.",
            "kind": "식", "ref_label": "엔진 값 — forecast_levels 와 같은 식", **run_lab(code, exp, pure=True)}


def lab_rmse(actual, model, naive, e_model, e_naive, e_zero):
    code = (f"# {len(actual)}개 폴드의 실제 변화율, VECM 예측, naive 예측(결정월까지의 최근 1개월 변화를 그대로) — 실행 기록 그대로\n"
            f"actual = {_pl(actual)}\n"
            f"model = {_pl(model)}\n"
            f"naive = {_pl(naive)}\n\n"
            "def rmse(pred):\n"
            "    s = 0.0\n"
            "    for p, a in zip(pred, actual):\n"
            "        s += (p - a) ** 2\n"
            "    return (s / len(actual)) ** 0.5\n\n"
            "print(\"VECM\", f\"{rmse(model) * 100:.4f}%\")\n"
            "print(\"naive\", f\"{rmse(naive) * 100:.4f}%\")\n"
            "print(\"0 예측\", f\"{rmse([0.0] * len(actual)) * 100:.4f}%\")\n")
    exp = f"VECM {e_model * 100:.4f}%\nnaive {e_naive * 100:.4f}%\n0 예측 {e_zero * 100:.4f}%"
    return {"title": "RMSE 를 직접 계산", "desc": "오차를 제곱해 평균하고 제곱근을 씌웁니다. VECM·naive·'항상 0' 을 같은 방법으로 비교합니다. "
                    "기대 출력은 이 대본이 엔진의 폴드별 예측으로 계산한 값입니다(이 설정의 엔진 결과 파일은 아직 없음).",
            "kind": "식", "ref_label": "영상 계산 값 — 같은 배열·같은 식", **run_lab(code, exp, pure=True)}


def build(cap: dict) -> dict:
    import warnings
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank
    src = Sources([MODEL, EXP, CAL, "models/base.py"])
    cfg = cap["config"]; p_ = cap["params"]; M = cap["model"]; k = cap["k"]
    Ltr = cap["Ltr"]; Ld = cap["Ld"]; Lall = cap["Lall"]; y = cap["y"]
    cols = ["F11" if c == "__target__" else c for c in Lall.columns]; K = len(cols)
    names = [cap["target_name"]] + [cap["variables"][c]["name"] for c in cols[1:]]
    if cols[1:] != list(SPOKEN):
        raise AssertionError(f"변수가 바뀜 {cols[1:]} — SPOKEN(음성에서 부르는 이름)을 새로 정하세요")
    if cfg.get("forecast_info_time", "stamp") != "stamp" or cfg["mode"] != "forecast":
        raise AssertionError("이 대본의 시점 설명은 예측 모드 · forecast_info_time = stamp(대상에 발표시차 없음)를 전제로 씁니다")
    sp = SPOKEN
    t = cap["pred"].index[-1]
    steps: list[dict] = []; checks_all: list[dict] = []

    def chk(what, a, b_, tol=1e-10, rel=False, kind="독립", vs="엔진"):
        a = np.asarray(a, dtype=float); b_ = np.asarray(b_, dtype=float)
        d = float(np.nanmax(np.abs(a - b_) / (np.maximum(np.abs(b_), 1e-300) if rel else 1.0))) if a.size else 0.0
        c = {"what": f"[{KIND[kind]}] {what} ({'상대 ' if rel else ''}차이 {d:.1e}{f', 허용 {tol:.0e}' if tol >= 1e-6 else ''})", "ok": bool(d <= tol), "kind": kind, "vs": vs}
        checks_all.append({**c, "step": len(steps) + 1}); return c

    def shown(cd) -> set:
        """플레이어(player.html showCode)가 실제로 보여 주는 줄 — 줄 설명(notes)이 화면에 나오는지 확인용."""
        ls = sorted(set(cd["lines"] or [cd["focus"]])); gs: list[list[int]] = []
        for l_ in ls:
            if gs and l_ - gs[-1][-1] <= 14: gs[-1].append(l_)
            else: gs.append([l_])
        fg = next((g_ for g_ in gs if cd["focus"] in g_), gs[0]); out = set()
        for g_ in gs:
            main = g_ is fg
            out |= set(range(max(1, g_[0] - (cd["before"] if main else 2)), g_[-1] + (cd["after"] if main else 2) + 1))
        return out

    def S(chapter, narr, *, code=None, stack=None, nodes=(), edges=(), tensors=(), formula=None, hint=None, checks=(), say=None, notes=None, lab=None):
        """say = 음성·자막 대본(화면 숫자 그대로 — 숫자는 narr 와 같은 변수에서), notes = {원본 줄 번호: 그 줄 설명}, lab = 직접 실행해 보기(실행해 기대 출력과 맞춘 것)."""
        checks = list(checks)
        if lab is not None:
            kd_ = lab["kind"]
            c_ = {"what": f"[{KIND[kd_]}] 직접 실행해 보기 '{lab['title']}' 출력 = {lab['ref_label']}", "ok": bool(lab["ok"]), "kind": kd_}
            checks_all.append({**c_, "step": len(steps) + 1}); checks.append(c_)
        if code is not None and notes:
            hidden = [l_ for l_ in notes if l_ not in shown(code)]
            if hidden:
                raise AssertionError(f"단계 {len(steps) + 1}: 줄 설명이 화면에 안 보이는 줄에 붙음 {hidden}")
            code = {**code, "notes": {str(k_): v_ for k_, v_ in notes.items()}}
        steps.append({"chapter": chapter, "narr": narr, "say": say, "code": code, "stack": list(stack or []), "nodes": list(nodes), "edges": [list(e) for e in edges],
                      "tensors": list(tensors), "formula": formula, "hint": hint, "checks": checks, "lab": lab})

    def C(fn, lines, focus=None, func="", before=6, after=8):
        lines = sorted(set(lines)); return {"file": fn, "lines": lines, "focus": focus or lines[0], "func": func, "before": before, "after": after}

    def mat_table(title, Mx, rows, colsx, d=4):
        return {"type": "table", "title": title, "columns": ["", *colsx], "rows": [[rows[i], *[g(v_, d) for v_ in Mx[i]]] for i in range(len(rows))]}

    ALL = ["lev", "unit", "joh", "beta", "gamma", "pred", "allf"]
    stk = ["engine/experiment.py · _walkforward_predict ('levels' 분기)", "models/base.py · LevelsModel.fit", f"{MODEL} · Model.fit_levels"]
    p = M["p"]; r = M["rank"]
    if p != 2:
        raise AssertionError("이 대본의 설명·식은 차분 시차 p = 2(Δy_(t−1), Δy_(t−2), Γ₂)를 전제로 씁니다")
    if M["det"] != "n" or M["seasons"] or np.any(M["log"]):
        raise AssertionError("이 대본의 손 계산은 상수 없음(n)·계절 없음·로그 없음 설정만 다룹니다")
    tl = cfg["validation"]["train_len"]; tlen = cfg["validation"]["test_len"]
    gap = list(Lall.loc[Ltr.index[-1]:t].index[1:-1])          # 학습 창 끝과 결정일 사이(비움)
    need = cap["need"]; hist = Ld.loc[:t].iloc[-max(need, 2):]
    nxt_m = (t + pd.offsets.MonthBegin(1)).strftime("%Y-%m")   # 정답이 쓰는 다음 달
    Z = np.asarray(Ltr.values, dtype=float)[:, M["keep"]]
    # ================================================================ 0
    if (NB2_FACT["k_ar_diff"], NB2_FACT["coint_rank"], NB2_FACT["det"]) != (p_["k_ar_diff"], p_["coint_rank"], p_["deterministic"]):
        raise AssertionError("설정이 원 노트북 2 의 VECM 설정과 다름 — '원 노트북 2 와 같이' 문장을 고치세요")
    nb = notebook2()                                             # 파일이 있을 때만 — 대본에는 영향 없음(출처 확인 줄만)
    if nb is None:
        nb_chk = {"what": f"[{KIND['출처']}] 원 노트북 2 의 VECM 칸 대조 — 건너뜀 — 노트북 파일 없음 (화면 문장은 2026-11-08 에 확인한 고정 문장)", "ok": True, "kind": "출처"}
    else:
        nb_diff = [k_ for k_ in NB2_FACT if nb.get(k_) != NB2_FACT[k_]]
        nb_chk = {"what": f"[{KIND['출처']}] 원 노트북 2({nb['file']}) 의 VECM 칸 = 화면 문장의 설정 (호출 · k_ar_diff · coint_rank · 결정항(statsmodels 기본값) · 변수 · 예측 길이)"
                          + (f" — 다름: {nb_diff}" if nb_diff else ""), "ok": not nb_diff, "kind": "출처"}
    checks_all.append({**nb_chk, "step": len(steps) + 1})
    mi = master_info(cols)                                       # 변수 사전(발표시차·확인 여부·메모) — 필수
    tr_src = {c: ("설정 파일" if c in (cfg.get("variable_transforms") or {}) else "변수 사전 기본") for c in cols[1:]}
    for c in cols[1:]:
        if tr_src[c] == "변수 사전 기본" and mi[c]["transform"] != cap["variables"][c]["transform_ignored"]:
            raise AssertionError(f"{c}: 기록된 변환이 변수 사전 기본값과 다름 — 변환 출처 설명을 고치세요")
    nb_hint = (f"모델 설정의 출처: 원 노트북 2(노트북2_통계마법_ARIMA_SARIMA_VECM.ipynb)의 VECM 칸 `{NB2_FACT['call']}` — deterministic 인자가 없어 statsmodels 기본값 '{NB2_FACT['det']}'(상수 없음). "
               "이 설정의 k_ar_diff · coint_rank · 상수 없음과 같습니다. "
               f"다만 노트북의 변수는 {', '.join(NB2_FACT['cols'])} 이고 학습 끝에서 {NB2_FACT['steps']}개월을 한 번에 예측했으므로, 변수·평가 방식은 이 실험과 다릅니다"
               "(업로드된 노트북 파일을 자료로 읽어 확인 — 저장소에는 없음. 파일이 있는 컴퓨터에서는 빌드 때 다시 대조: 검증 표의 '출처 확인' 줄).")
    S("0. 시작",
      f"이 영상은 <b>VECM</b>(벡터 오차수정모형) 이 실제로 계산되는 순서를 따라갑니다. 설정 <b>{Path(cap['config_file']).name}</b>: 대상 F11({names[0]})과 변수 {K - 1}개({', '.join(cols[1:])}), "
      f"모두 {K}개 시계열의 <b>수준값</b>을 함께 모델링합니다. 생각: 여러 시계열이 각자 떠돌아도 장기적으로 함께 움직이는 관계(공적분, 장기 균형식)가 있다면, 균형에서 벗어난 만큼 다음 달에 되돌아오는 힘을 예측에 쓸 수 있다. "
      f"워크포워드 폴드 {cap['n_folds']}개마다 다시 추정하며, 화면의 숫자는 마지막 폴드(결정일 {t.date()})의 실제 실행 기록입니다.",
      say=(f"이 영상은 벡터 오차수정모형, VECM이 실제로 계산되는 순서를 따라갑니다. 대상은 KB 주택매매가격지수 F11이고, {sp['F12']}, {sp['M3']}, {sp['N19']}, {sp['KR3Y']}, "
           f"이렇게 변수 {K - 1}개를 더해 모두 {K}개 시계열의 수준값을 함께 모델링합니다. 여러 시계열이 각자 떠돌아도 장기적으로 함께 움직이는 관계가 있다면, "
           "그 관계에서 벗어난 만큼 되돌아오는 힘을 예측에 쓰자는 생각입니다. "
           + f"모델 설정은 원 노트북의 VECM 설정과 같이 지난 {nat_months(p_['k_ar_diff'])}의 변화를 보고, 균형식은 {p_['coint_rank']}개, 상수항은 없습니다. 다만 변수와 평가 방식은 노트북과 다릅니다. "
           + f"폴드 {cap['n_folds']}개마다 다시 추정하며, 화면의 숫자는 마지막 폴드, 결정일 {t.date()}의 실행 기록입니다."),
      notes={src.line(MODEL, "Δy_t = α(β'y_{t−1} + c_in)"): "이 설정(상수 없음 n · 계절 0)에서는 c_in · c_out · 계절 항이 모두 없음 → Δy_t = α(β′y_(t−1)) + Γ₁Δy_(t−1) + Γ₂Δy_(t−2) + ε_t",
             src.line(MODEL, "기본값 = 노트북 2"): f"설정 파일의 model_params 가 비어 있어 모두 기본값: k_ar_diff {p_['k_ar_diff']} · coint_rank {p_['coint_rank']} · deterministic {p_['deterministic']} · log_levels 아니오. "
                                                     f"노트북의 실제 호출은 `{NB2_FACT['call']}` (이 주석은 변수 이름 '훈련다' 를 '훈련' 으로 줄여 씀)"},
      code=C(MODEL, [1, src.line(MODEL, "Δy_t = α(β'y_{t−1} + c_in)"), src.line(MODEL, "기본값 = 노트북 2")], 1, "모듈 설명", 0, 12), stack=[MODEL], nodes=ALL,
      formula="Δy_t = α(β′y_(t−1)) + Γ₁Δy_(t−1) + … + Γ_pΔy_(t−p) + ε_t     (y = [F11, 변수들] 수준값, β′y = 장기 균형식, α = 되돌아오는 속도, Γ = 단기 동학)",
      tensors=[{"type": "table", "title": f"실행 설정 ({cap['config_file']} + {MODEL} 기본값)", "columns": ["항목", "값", "뜻"], "rows": [
          ["대상", "F11 다음 달 변화율", f"k = {k} — 수준 예측 v̂_(t+1) ÷ v_t − 1 로 바꿔 다른 모델과 비교"],
          *[[c, f"{nm_} ({cap['variables'][c]['freq']}, lag {cap['variables'][c]['lag']}일)", f"{tr_src[c]} 변환({cap['variables'][c]['transform_ignored']})은 무시(수준값 필요)"] for c, nm_ in zip(cols[1:], names[1:])],
          *[[k_, str(v_), ""] for k_, v_ in p_.items()]]}],
      hint="숫자의 출처: tools/codewalk/vecm_capture.py 가 엔진의 실제 경로로 실행하며 학습 행렬과 적합된 α·β·Γ, 예측을 기록했습니다. " + nb_hint,
      checks=[nb_chk])
    S("0. 시작",
      "호출 순서: 엔진의 <b>levels 분기</b>가 [대상 수준, 변수 수준들] 행렬을 만들고 학습 창의 빈 칸 없는 행으로 <b>m.fit</b>(→ fit_levels) 합니다. "
      "정답 y 를 따로 넘기지 않는 것은 대상 F11 의 수준이 이 행렬의 첫 열에 이미 들어 있기 때문입니다(정답 y_t = F11_(t+1) ÷ F11_t − 1 도 이 열에서 나옴). "
      f"테스트 결정일마다 결정월 행까지의 마지막 {need}행(need = p + 2)만 넣어 <b>forecast_levels</b> 로 한 달 뒤 수준을 재귀 계산하고, v̂_(t+1) ÷ v_t − 1 을 예측으로 씁니다. "
      f"한 폴드 안에서는 계수를 다시 추정하지 않지만, 테스트가 {tlen}개월(test_len = {tlen})이라 결정일마다 새 폴드에서 다시 추정합니다.",
      say=("코드는 이렇게 불립니다. 엔진은 대상과 변수들의 수준값을 옆으로 붙인 행렬을 만들고, 학습 창의 빈 칸 없는 행으로 모델을 적합합니다. "
           "정답 y를 따로 넘기지 않는 것은, 대상 F11의 수준이 이 행렬의 첫 열에 이미 들어 있기 때문입니다. "
           f"결정일에는 결정월 행까지의 마지막 {need}개 행을 넘겨 한 달 뒤 F11의 수준을 계산하고, 결정월 값으로 나누고 1을 빼 변화율 예측으로 씁니다. "
           f"한 폴드 안에서는 계수를 다시 추정하지 않지만, 테스트가 {nat_months(tlen)}이라 결정일마다 새 폴드에서 다시 추정합니다."),
      notes={src.line(EXP, "Ltr = Lall.loc[f.train].dropna()"): f"학습 부분(f.train, 비움 뒤) {len(Lall.loc[cap['fold']['train']])}개 달에서 빈 칸 있는 행을 뺌(이 폴드 {len(Lall.loc[cap['fold']['train']]) - len(Ltr)}행) → {len(Ltr)}행 × {K}열",
             src.line(EXP, 'need = int(getattr(m, "p", params.get("k_ar_diff", 2))) + 2'): f"need = p + 2 = {need} → 결정월 행까지의 마지막 {need}행을 넘김(모델은 그중 마지막 {p + 1}행만 씀)",
             src.line(EXP, 'pred.loc[t] = float(np.log(ratio)) if cfg.target.kind == "logreturn" else float(ratio - 1.0)'):
                 f"정답 종류 {cfg['target']['kind']} → ratio − 1. 결정일 {t.date()}: {pct(cap['pred'].loc[t])}"},
      code=C(EXP, src.span(EXP, 'if plugin.INPUT_SHAPE == "levels":', 'pred.loc[t] = float(np.log(ratio)) if cfg.target.kind == "logreturn" else float(ratio - 1.0)'),
             src.line(EXP, "m.fit(Ltr.values, np.array([]))"), "_walkforward_predict (levels)", 1, 1),
      stack=stk[:1], nodes=["lev", "joh", "pred"])
    # ================================================================ 1. 수준값 (시점)
    from datetime import timedelta as _td
    import re as _re
    st_raw = store_raw(cfg["data_version"], cols)                 # 원자료 직접 읽기(엔진 경로와 독립)
    rows_l = []; hand = []; eng = []; when = {}; mb = {}; mbn = {}
    for c in cols[1:]:
        v = cap["variables"][c]; raw = st_raw["series"][c]
        avail = raw.index + pd.to_timedelta(v["lag"], unit="D"); ok = avail <= t
        j = int(np.flatnonzero(ok)[-1]); hand.append(float(raw.iloc[j])); eng.append(float(Lall.loc[t, c]))
        when[c] = (raw.index[j].strftime("%Y-%m-%d"), avail[j].strftime("%Y-%m-%d"), v["lag"])
        mbn[c] = months_before(t, raw.index[j])
        mb[c] = "결정일 당일" if raw.index[j] == t else f"{mbn[c]}개월 전"
        lag_txt = f"{v['lag']}" if mi[c]["ok"] else f"{v['lag']} (사전 미확인)"
        rows_l.append([c, v["name"], v["freq"], lag_txt, raw.index[j].strftime("%Y-%m-%d"), avail[j].strftime("%Y-%m-%d"), mb[c], g(raw.iloc[j], 7), v["unit"] or "% (사전에 단위 비어 있음)"])
    unconf = [c for c in cols[1:] if not mi[c]["ok"]]
    lagF = mi["F11"]["lag"]
    if unconf != ["N19"]:
        raise AssertionError(f"시차 미확인 변수가 바뀜 {unconf} — '가계대출의 시차는 확인되지 않은 값' 문장을 고치세요")
    rows_l.insert(0, ["F11", names[0], "M", f"적용 안 함 (사전 {lagF})", t.strftime("%Y-%m-%d"), "— (stamp)", "결정월", g(Lall.loc[t, "__target__"], 7), "지수"])
    prev_m = (t - pd.offsets.MonthBegin(1))
    if True:                                                     # (변수 사전은 필수 — 위 master_info)
        mo = _re.search(r"관측 (\d+)건 공개일−기준일 (\d+)~(\d+)일", mi["F11"]["note"]); late = "발표 = 같은 달 하순" in mi["F11"]["note"]
        if not (mo and late):
            raise AssertionError("F11 의 사전 메모 형식이 바뀜 — 발표 시점 설명을 고치세요")
        f11_txt = (f" 변수 사전의 F11 시차 {lagF}일은 보수적으로 잡은 값으로(사전 메모: 관측 {mo.group(1)}건 공개일 − 기준일 {mo.group(2)}~{mo.group(3)}일, '발표 = 같은 달 하순'), "
                   f"이 기준이면 {t.strftime('%Y-%m')} 값은 {t.date() + _td(days=lagF)} 부터 쓰입니다. 메모대로 그 달 하순에 발표된다면 결정일 {t.date()} 에 "
                   f"{t.strftime('%Y-%m')} 값은 공개 전이고, {prev_m.strftime('%Y-%m')} 값은 이미 공개됐을 수 있습니다(실제 공개일은 이 화면에서 확인하지 않음).")
        f11_say = (f"사전 메모로는 F11이 그 달 하순에 발표되므로, 결정일에 {t.strftime('%Y-%m')} 값은 아직 공개 전이고, {prev_m.strftime('%Y-%m')} 값은 이미 공개됐을 수 있습니다. ")
        f11_hint = (f"F11 의 시차 {lagF}일과 메모, 변수별 확인 여부는 변수 사전(data/variable_master.xlsx)에서 읽은 값입니다(실행 기록에는 없음). ")
    rev_notes = [f"{c}: “{_re.search(r'수정: ([^|]*)', mi[c]['note']).group(1).strip()}”" for c in cols[1:] if _re.search(r"수정: ([^|]*)", mi[c]["note"]) and c in ("M3", "N19")]
    S("1. 수준값",
      "VECM 은 변화율이 아니라 <b>수준값</b>을 씁니다(변환은 무시). 각 변수는 발표시차를 적용해 결정일 달력에 맞춥니다: 공개일 = 기준일 + lag 일(실제 발표일이 아니라 사전의 시차로 잡은 날), 공개일이 결정일과 같거나 이전인 마지막 값 "
      "(값은 지금 저장된 수정치를 그 공개일에 놓은 것이라, 당시 처음 발표된 값과 다를 수 있음). "
      f"예: 결정일 {t.date()} 에 쓰는 M2 는 기준일 {when['M3'][0]} 의 값(공개일 {when['M3'][1]} = 기준일 + {when['M3'][2]}일)입니다. "
      + (f"{', '.join(unconf)} 의 시차는 변수 사전에서 확인되지 않은 값입니다(lag_confirmed = N). " if unconf else "")
      + f"<b>대상 F11 에는 발표시차를 적용하지 않습니다</b>(forecast_info_time = stamp — 결정월 t 행에 그 달 F11 값).{f11_txt} "
      f"정답은 {nxt_m} 값이라 누출은 아니지만, 실제로는 그 달 F11 이 공개된 뒤에 예측하는 셈이고 변수는 결정일까지 공개된 값만 씁니다. "
      f"그래서 한 행 안에 서로 다른 달이 섞입니다: F11 결정월(t), 전세가격지수 {mb['F12']}, M2 {mb['M3']}, 가계대출 {mb['N19']}, 금리 {mb['KR3Y']}.",
      say=("VECM은 수준값을 쓰므로 변환은 무시하고, 변수는 발표 시차를 넣어 결정일까지 공개된 마지막 값을 씁니다. "
           f"예를 들어 결정일의 {sp['M3']}은 {when['M3'][0]} 기준 값이고, 공개일은 기준일에 {when['M3'][2]}일을 더한 {when['M3'][1]}로 봅니다. "
           "다만 값은 지금의 수정치라, 당시 처음 발표된 값과 다를 수 있습니다. "
           + "가계대출의 시차는 변수 사전에서 확인되지 않은 값입니다. "
           + "반면 대상 F11에는 시차를 적용하지 않고 결정월 값을 그대로 씁니다. "
           + f11_say
           + "정답은 그다음 달 값이라 누출은 아니지만, 실제로는 그 달 F11이 공개된 뒤에 예측하는 셈입니다. "
           f"그래서 한 행 안에서 F11은 결정월, 전세가격지수는 {nat_months(mbn['F12'])} 전, 통화량과 가계대출은 {nat_months(mbn['M3'])} 전 값이고, 시차가 없는 일간 금리는 결정일 당일 값입니다.")
      if mb["M3"] == mb["N19"] and mb["KR3Y"] == "결정일 당일" and when["KR3Y"][2] == 0 else None,
      notes={src.line(EXP, "level_cols[vid] = cal.align_to_calendar(wide[vid].dropna(), info_calendar, lag_days=lag, fill=cfg.fill, apply_lag=cfg.apply_lag).set_axis(calendar)"):
                 f"변환하지 않은 원래 값(wide[vid])을 공개일(기준일 + lag 일)부터 쓰게 달력에 맞춤. M2: {when['M3'][0]} + {when['M3'][2]}일 = {when['M3'][1]}",
             src.line(EXP, 'Lall = pd.concat([ds.prices["close"].rename("__target__"), ds.levels], axis=1)'): f"열 0 = 대상 F11 의 수준(가격 자리, 발표시차 없음), 그 뒤 변수 {K - 1}개의 수준 → {K}열"},
      code=C(EXP, src.lines(EXP, "level_cols[vid] = cal.align_to_calendar(wide[vid].dropna(), info_calendar, lag_days=lag, fill=cfg.fill, apply_lag=cfg.apply_lag).set_axis(calendar)",
                            "levels = pd.DataFrame(level_cols).reindex(calendar) if cfg.variable_ids else pd.DataFrame(index=calendar)",
                            'Lall = pd.concat([ds.prices["close"].rename("__target__"), ds.levels], axis=1)'),
             src.line(EXP, "level_cols[vid] = cal.align_to_calendar(wide[vid].dropna(), info_calendar, lag_days=lag, fill=cfg.fill, apply_lag=cfg.apply_lag).set_axis(calendar)"), "build_dataset · levels", 2, 2),
      stack=["engine/experiment.py · build_dataset", "engine/calendar.py · align_to_calendar"], nodes=["lev"],
      tensors=[{"type": "table", "title": f"결정일 {t.date()} 의 행 — 열마다 어느 기준일의 값인가", "columns": ["열", "이름", "주기", "lag(일)", "기준일", "공개일", "결정월과의 차이", "값", "단위"], "rows": rows_l}],
      hint=f11_hint + (f"수정 관련 사전 메모 — {' · '.join(rev_notes)}. " if rev_notes else "")
           + "forecast_info_time = release 로 두면 결정 시점을 대상 공개 시점(t + 대상 lag)으로 옮겨 변수도 그때까지 공개된 값을 씁니다 — 이 설정은 stamp(기본)입니다.",
      checks=[chk(f"변수 {K - 1}개: 데이터 저장소의 parquet({st_raw['dir']}/variables_*.parquet)을 엔진을 거치지 않고 직접 읽어 발표시차로 찾은 결정일 값 = 엔진 행렬", hand, eng, 0)])
    # 학습 행렬 전부를 직접 읽은 원자료로 다시 만들기: F11 은 결정월 값 그대로(stamp), 변수는 발표시차 적용 — 엔진의 정렬 함수와 저장소 읽기 모두와 독립
    hand_tr = np.full((len(Ltr), K - 1), np.nan)
    f11_hand = st_raw["series"]["F11"].reindex(Ltr.index).values
    for jj, c in enumerate(cols[1:]):
        v = cap["variables"][c]; raw = st_raw["series"][c]
        idx = (pd.DatetimeIndex(raw.index).normalize() + pd.to_timedelta(v["lag"], unit="D")).values
        pos = np.searchsorted(idx, Ltr.index.values, side="right") - 1
        hand_tr[:, jj] = np.where(pos >= 0, raw.values[np.maximum(pos, 0)], np.nan)
    if cap["variables"]["M3"]["unit"] != "십억원" or cap["variables"]["N19"]["unit"] != "십억원":
        raise AssertionError("M2·가계대출 단위가 십억원이 아님 — 조 원 환산을 고치세요")
    jo = {c: (Ltr[c].min() / 1000, Ltr[c].max() / 1000) for c in ("M3", "N19")}       # 십억원 ÷ 1,000 = 조 원
    jw = lambda v_: f"{v_:,.0f}"
    base0 = Ltr.iloc[0]
    S("1. 수준값",
      f"학습 행렬 {len(Ltr)}행 × {K}열({Ltr.index[0].date()} ~ {Ltr.index[-1].date()}). 학습 창 {tl}개월 중 끝의 {len(gap)}개월({', '.join(str(d.date()) for d in gap)})은 비움 — "
      f"학습 정답이 다음 구간의 값과 겹치지 않게 비우는 엔진 공통 규칙입니다. k = {k}·stamp 에서는 비운 행의 정답(F11_{t.strftime('%Y-%m')} ÷ F11_{gap[-1].strftime('%Y-%m')} − 1)도 결정월 행에 이미 있는 값이고 "
      f"테스트 정답({nxt_m})과 겹치지 않아, 누출 방지에 꼭 필요한 비움은 아니고 보수적 여유입니다. VECM 도 같은 폴드를 씁니다. "
      f"단위가 크게 달라(M2 약 {jw(jo['M3'][0])}조 ~ {jw(jo['M3'][1])}조 원, 가계대출 약 {jw(jo['N19'][0])}조 ~ {jw(jo['N19'][1])}조 원 — 원자료 단위 십억원 ÷ 1,000, 금리는 %) "
      "그림에는 첫 달을 100 으로 맞춘 값을 그렸습니다(계산은 원래 수준값 그대로).",
      say=(f"마지막 폴드의 학습 행렬은 {len(Ltr)}행, 열 {K}개로, {Ltr.index[0].date()}부터 {Ltr.index[-1].date()}까지입니다. "
           f"학습 창 {tl}개월 가운데 끝의 {nat_months(len(gap))}, {gap[-1].date()}은, 학습 정답이 다음 구간 값과 겹치지 않게 비우는 엔진 공통 규칙에 따라 비웁니다. "
           "한 달 앞 예측에서는 꼭 필요한 비움이 아니라 보수적인 여유입니다. "
           f"{sp['M3']}은 약 {jw(jo['M3'][0])}조에서 {jw(jo['M3'][1])}조 원, 가계대출은 약 {jw(jo['N19'][0])}조에서 {jw(jo['N19'][1])}조 원이고 금리는 퍼센트라, "
           "그림에는 첫 달을 100으로 맞춘 값을 그렸습니다. 계산은 원래 수준값 그대로 합니다.")
      if k == 1 and len(gap) == 1 else None,
      notes={src.line(EXP, "Ltr = Lall.loc[f.train].dropna()"): f"학습 창({Ltr.index[0].date()} ~ {Ltr.index[-1].date()})의 빈 칸 없는 행 → {len(Ltr)} × {K}",
             src.line(EXP, "m.fit(Ltr.values, np.array([]))"): "LevelsModel.fit → fit_levels(Ltr.values). 정답 자리는 빈 배열 — 대상 F11 의 수준이 첫 열에 들어 있음"},
      code=C(EXP, src.lines(EXP, "Ltr = Lall.loc[f.train].dropna()", "m.fit(Ltr.values, np.array([]))"), src.line(EXP, "Ltr = Lall.loc[f.train].dropna()"), "_walkforward_predict", 2, 6),
      stack=stk[:1], nodes=["lev"],
      tensors=[{"type": "line", "title": "학습 창의 수준값 (첫 달 = 100)", "x": [d.strftime("%Y-%m") for d in Ltr.index], "xlabel": "월",
                "series": [{"name": c, "color": col, "y": [num(v_ / base0.iloc[j] * 100) for v_ in Ltr.iloc[:, j]]} for j, (c, col) in enumerate(zip(cols, ["#3b6fb6", "#c8710f", "#2f8f5b", "#6a3fa0", "#b3261e"]))], "height": 250},
               {"type": "table", "title": "학습 행렬 끝 3행 (원래 수준값 — M2·가계대출은 십억원)", "columns": ["월", *cols], "rows": [[d.strftime("%Y-%m"), *[g(v_, 7) for v_ in r_]] for d, r_ in zip(Ltr.index[-3:], Ltr.values[-3:])]}],
      checks=[chk(f"학습 행렬 {len(Ltr)}행 × {K}열: 직접 읽은 원자료로 다시 만든 행렬(F11 은 그 달 값, 변수는 발표시차 적용) = 엔진 행렬", np.column_stack([f11_hand, hand_tr]), Ltr.values, 0)])
    # ================================================================ 2. 단위근
    adf = M["diag"]["adf_p_levels"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_ct = [float(adfuller(Z[:, j], regression="ct", autolag="AIC")[1]) for j in range(K)]
    all_big = all(v_ is not None and v_ > 0.05 for v_ in adf); jmin = int(np.argmin(adf))
    rej_ct = [j for j in range(K) if adf_ct[j] < 0.05]
    S("2. 단위근",
      "VECM 의 전제: 각 수준값은 <b>단위근</b>이 있어(I(1)) 혼자서는 평균으로 돌아오지 않지만, 몇 개의 조합은 정상이다. 진단으로 학습 창마다 수준값에 ADF 검정을 합니다 — "
      "p 값이 크면(> 0.05) '단위근이 있다' 를 기각하지 못합니다. 이 폴드(엔진 진단): " + ", ".join(f"{c} {v_:.2f}" for c, v_ in zip(cols, adf)) + ". "
      "엔진 진단의 검정식은 <b>상수만 있는 것</b>(regression = 'c', statsmodels 기본)입니다. 추세를 넣으면(ct — 이 화면에서 statsmodels 로 따로 계산) "
      + (", ".join(f"{cols[j]} 의 p 값이 {adf_ct[j]:.4f}" for j in rej_ct) + " 로 5% 에서 단위근을 기각합니다 — <b>결론이 검정식에 따라 달라집니다</b>." if rej_ct else "모두 0.05 보다 큽니다."),
      say=("VECM은 각 수준값에 단위근이 있어 혼자서는 평균으로 돌아오지 않지만, 몇 개를 묶은 조합은 정상이라고 전제합니다. "
           "그래서 학습 창마다 수준값에 ADF 검정을 진단으로 합니다. p 값이 0.05보다 크면 단위근이 있다는 가설을 기각하지 못합니다. "
           + (f"이 폴드에서는 다섯 시계열 모두 0.05보다 크고, 가장 작은 {cols[jmin]}도 {adf[jmin]:.4f}입니다. " if all_big and K == 5 else "")
           + "다만 엔진의 진단은 상수만 넣은 검정식입니다. "
           + (f"영상에서 추세까지 넣어 따로 계산하면 {cols[rej_ct[0]]}의 p 값은 {adf_ct[rej_ct[0]]:.4f}로 단위근을 기각합니다. 검정식에 따라 결론이 달라집니다. "
              if len(rej_ct) == 1 else "")
           + "이 진단은 기록만 하고 모델 계산에는 쓰지 않습니다."),
      notes={src.line(MODEL, 'unit.append(round(float(adfuller(Z[:, j], autolag="AIC")[1]), 4))'):
                 "statsmodels adfuller(자동 시차 AIC, 기본 회귀식 regression = 'c' = 상수만)의 p 값을 소수 4자리로 diag 에 기록 — 모델 계산에는 쓰지 않음"},
      code=C(MODEL, src.lines(MODEL, "for j in range(K):  # 단위근 진단(정보용)", 'unit.append(round(float(adfuller(Z[:, j], autolag="AIC")[1]), 4))'),
             src.line(MODEL, 'unit.append(round(float(adfuller(Z[:, j], autolag="AIC")[1]), 4))'), "fit_levels · 진단", 2, 6),
      stack=stk + ["statsmodels · adfuller"], nodes=["unit"], edges=[("lev", "unit")],
      tensors=[{"type": "table", "title": "ADF 검정 p 값 (수준값, 자동 시차 AIC)", "columns": ["열", "이름", "p 값: 상수만 'c' (엔진 진단)", "판단(5%)", "p 값: 상수+추세 'ct' (영상 계산)", "판단(5%)"],
                "rows": [[c, nm_, f"{v_:.4f}", "단위근 기각 못 함" if v_ > 0.05 else "단위근 기각", f"{w_:.4f}", "단위근 기각 못 함" if w_ > 0.05 else "단위근 기각"]
                         for c, nm_, v_, w_ in zip(cols, names, adf, adf_ct)]}],
      hint="ADF 는 정보용 진단입니다. '기각하지 못함' 은 단위근이 있다는 증명이 아닙니다. 'ct' 열은 엔진이 계산하지 않는 값으로, 이 대본이 같은 학습 행렬에 statsmodels adfuller(regression = 'ct') 를 돌린 결과입니다. "
           "수준이 이미 정상(I(0))인 변수가 섞이면 Johansen 검정이 공적분 차수를 부풀립니다(그 변수 하나만으로 '정상인 조합'이 됨).")
    # ================================================================ 3. Johansen
    J = johansen(Z, p, r)
    eig_cj = np.real(cap["johansen"]["eig"]); d_eig = float(np.max(np.abs(J["lam"] - eig_cj)))
    # 두 구현의 고유값 차이가 단위 차이로 생긴 반올림 오차인지: 큰 열(M2·가계대출)을 100만으로 나눠 같은 계산(고유값은 열의 단위에 따라 바뀌지 않음)
    sc = np.array([1e-6 if np.median(np.abs(Z[:, j])) > 1e5 else 1.0 for j in range(K)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        J2 = johansen(Z * sc, p, r); cj_s = coint_johansen(Z * sc, -1, p); eig_cj2 = np.real(cj_s.eig)
        cj_u = coint_johansen(Z, -1, p)                            # 원래 단위의 coint_johansen (고유벡터 — 정확한 β 비교용)
    d_eig2 = float(np.max(np.abs(J2["lam"] - eig_cj2))); cond1, cond2 = np.linalg.cond(J["S11"]), np.linalg.cond(J2["S11"])
    d_cj_scale = float(np.max(np.abs(eig_cj - eig_cj2)))             # coint_johansen: 단위를 바꿔도 고유값이 거의 같음
    if not (d_cj_scale < 1e-9 and d_eig2 < 1e-9 and d_eig > 1e3 * max(d_cj_scale, d_eig2)):
        raise AssertionError("고유값 오차가 어느 쪽에 있는지에 대한 설명이 기록과 맞지 않음 — 힌트를 고치세요")
    b_cj = np.real(cj_u.evec[:, :r]); b_cj = b_cj @ np.linalg.inv(b_cj[:r])   # coint_johansen 고유벡터를 같은 방식(첫 원소 1)으로 정규화
    b_ex = J2["beta"] * sc[:, None]; b_ex = b_ex @ np.linalg.inv(b_ex[:r])       # 단위를 맞춘 손 계산 β 를 원래 단위로
    d_beta_ex = float(np.max(np.abs(M["beta"][:, 0] - b_cj[:, 0]) / np.abs(b_cj[:, 0])))
    # 같은 숫자를 행 우선(C) 배열로 넣으면: 엔진(판다스 .values = 열 우선)과 내부 계산 순서만 다른 계산 → β·예측이 얼마나 흔들리나
    Jc = johansen(np.ascontiguousarray(Z), p, r)
    d_lay = float(np.max(np.abs(Jc["beta"][:, 0] - M["beta"][:, 0]) / np.abs(M["beta"][:, 0])))
    Hc = Ld.loc[:t].iloc[-(p + 1):].values[:, M["keep"]]
    fc_ = lambda A_, B_, G_: float((Hc[-1] + A_ @ (B_.T @ Hc[-1]) + sum(G_[:, i * K:(i + 1) * K] @ (Hc[-1 - i] - Hc[-2 - i]) for i in range(p)))[0] / Hc[-1][0] - 1)
    d_lay_pred = abs(fc_(Jc["alpha"], Jc["beta"], Jc["gamma"]) - fc_(M["alpha"], M["beta"], M["gamma"]))
    S("3. Johansen",
      f"<b>공적분 추정(Johansen, 축소 순위 회귀)</b>: ① Δy_t 와 y_(t−1) 을 각각 지난 변화 Δy_(t−1), Δy_(t−2) 에 회귀해 잔차 R₀, R₁ 을 만듭니다(단기 동학을 걷어 냄). "
      f"② S₀₀ = R₀R₀′/T, S₀₁ = R₀R₁′/T, S₁₁ = R₁R₁′/T (T = {J['T']} = {len(Ltr)} − {p + 1}). ③ 행렬 S₁₁^(−½) S₁₀ S₀₀⁻¹ S₀₁ S₁₁^(−½) 의 고유값 λ 는 R₁ 의 조합과 R₀ 의 조합 사이 상관의 제곱(정준상관의 제곱)이고, "
      f"큰 고유값의 고유벡터부터 S₁₁^(−½) 를 곱해 장기 균형식 β 가 됩니다. 이 설정은 가장 큰 r = {r} 개만 씁니다.",
      say=("이제 요한센 방법으로 장기 균형식을 추정합니다. 첫째, 이번 달 변화와 지난달 수준을 각각 지난 두 달의 변화에 회귀해, 단기 움직임을 걷어 낸 잔차를 만듭니다. "
           f"둘째, 두 잔차로 공분산 행렬 세 개를 만듭니다. 쓸 수 있는 식은 {len(Ltr)}행에서 {p + 1}을 뺀 {J['T']}개입니다. "
           "셋째, 화면에 있는 행렬의 고유값을 구합니다. 고유값은 지난 수준 쪽 잔차의 조합과 이번 달 변화 쪽 잔차의 조합 사이 상관의 제곱입니다. "
           f"이 설정은 가장 큰 {J['lam'][0]:.6f} 하나만 써서 균형식을 {r}개로 둡니다."),
      notes={src.line(MODEL, "dof = (T - self.p - 1) - (K * self.p + K + n_det)"): f"자유도 = ({len(Ltr)} − {p} − 1) − ({K} × {p} + {K} + 0) = {(len(Ltr) - p - 1) - (K * p + K)} ≥ 10 → 통과",
             src.line(MODEL, "res = VECM(Z, k_ar_diff=self.p, coint_rank=r, deterministic=self.det, seasons=self.seasons).fit()"):
                 f"statsmodels VECM(k_ar_diff = {p}, coint_rank = {r}, deterministic = '{M['det']}', seasons = {M['seasons']}).fit() — 최우추정(요한센 절차)",
             src.line(MODEL, "self.alpha, self.beta, self.gamma = res.alpha, res.beta, res.gamma"): f"적합된 α({K}×{r}) · β({K}×{r}) · Γ({K}×{p * K})를 보관 — 예측은 이 세 행렬로 직접 계산"},
      code=C(MODEL, src.lines(MODEL, "dof = (T - self.p - 1) - (K * self.p + K + n_det)",
                              "res = VECM(Z, k_ar_diff=self.p, coint_rank=r, deterministic=self.det, seasons=self.seasons).fit()",
                              "self.alpha, self.beta, self.gamma = res.alpha, res.beta, res.gamma"),
             src.line(MODEL, "res = VECM(Z, k_ar_diff=self.p, coint_rank=r, deterministic=self.det, seasons=self.seasons).fit()"), "fit_levels", 12, 3),
      stack=stk + ["statsmodels · VECM.fit (Lütkepohl 2005 7.2 — 아래는 numpy 로 다시 계산)"], nodes=["joh"], edges=[("unit", "joh")],
      formula="R₀ = Δy − 회귀(Δy | ΔX),  R₁ = y₋₁ − 회귀(y₋₁ | ΔX);   고유값 문제  S₁₁^(−½) S₁₀ S₀₀⁻¹ S₀₁ S₁₁^(−½) v = λ v,   β = S₁₁^(−½) v",
      tensors=[{"type": "table", "title": "고유값 λ (큰 순서) — 손 계산(VECM.fit 과 같은 순서) vs statsmodels(coint_johansen)", "columns": ["번호", "λ (손 계산)", "λ (statsmodels)"],
                "rows": [[i + 1, f"{a_:.6f}", f"{b_:.6f}"] for i, (a_, b_) in enumerate(zip(J["lam"], eig_cj))]}],
      hint=("엔진이 부르는 VECM.fit 은 이 고유값을 안에서 계산하지만 결과(VECMResults)로 돌려주지 않습니다. 그래서 VECM.fit 과 같은 순서로 다시 계산한 손 계산 값을 statsmodels 의 다른 함수 coint_johansen 의 값과 비교합니다. "
            f"이 자료에서는 최대 {d_eig:.1e} 차이가 납니다. M2·가계대출 열을 100만으로 나눠(고유값은 열의 단위에 따라 바뀌지 않음) 같은 두 계산을 하면 차이가 {d_eig2:.1e} 로 줄어듭니다 — S₁₁ 의 조건수 {cond1:.1e} → {cond2:.1e}. "
            f"coint_johansen 의 고유값은 단위를 바꿔도 {d_cj_scale:.1e} 안에서 같으므로, <b>오차는 VECM.fit 순서의 계산 쪽</b>(단위가 크게 다른 열 때문에 생긴 반올림 오차)에 있습니다. "
            f"엔진의 β 도 이 쪽 계산이라 정확한 값(coint_johansen 고유벡터)과 상대 {d_beta_ex:.1e} 다릅니다 — '4. 장기 균형' 표. 같은 숫자를 행 우선 배열로 넣기만 해도(같은 식, 내부 계산 순서만 달라짐) "
            f"β 가 상대 {d_lay:.1e}, 이 결정일의 예측이 {d_lay_pred:.1e} 달라집니다 — 예측에는 무시할 크기입니다. "
            "아래 β·α·Γ 는 VECM.fit 과 같은 순서·같은 배열로 계산해 엔진 값과 맞춥니다."),
      checks=[chk("손 계산 고유값(VECM.fit 순서) = statsmodels coint_johansen 고유값 (다른 구현 — 차이는 위 힌트의 반올림 오차)", J["lam"], eig_cj, 1e-6, vs="coint_johansen")])
    # ---------------- 균형식 개수: 대각합·최대고유값·소표본 보정 — 폴드별 학습 창을 실행 기록(Lall·폴드 날짜)에서 다시 만들어 statsmodels 로 계산(정보용)
    tr_, cv_ = cap["johansen"]["trace"], cap["johansen"]["trace_cv"]
    me_, cvm_ = cap["johansen"]["maxeig"], cap["johansen"]["maxeig_cv"]
    Tn = J["T"]; fac = (Tn - K * (p + 1)) / Tn                   # 소표본 보정: 통계량 × (T − K·(p+1)) / T
    first_keep = lambda stat, cv: next((i for i in range(K) if stat[i] < cv[i]), K)
    sel = first_keep(tr_, cv_[:, 1]); sel_me = first_keep(me_, cvm_[:, 1]); sel_cor = first_keep(tr_ * fac, cv_[:, 1])
    af = cap["all_folds"]; calx = Lall.index; emb = len(gap)
    if any(x["keep"] != list(range(K)) or x["p"] != p for x in af):
        raise AssertionError("폴드마다 열·차수가 다름 — 폴드별 재계산을 고쳐야 합니다")
    fold_Z, rk_tr, rk_me, rk_cor, b_rel, vr_pred = [], [], [], [], 0.0, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x in af:
            i_ = calx.get_loc(pd.Timestamp(x["test"][0]))
            Zf = np.asarray(Lall.loc[calx[i_ - tl:i_ - emb]].dropna().values, dtype=float)
            fold_Z.append(Zf)
            rk_tr.append(int(select_coint_rank(Zf, -1, p, method="trace", signif=0.05).rank))
            rk_me.append(int(select_coint_rank(Zf, -1, p, method="maxeig", signif=0.05).rank))
            cjf = coint_johansen(Zf, -1, p); Tf = Zf.shape[0] - (p + 1)
            rk_cor.append(first_keep(cjf.lr1 * (Tf - K * (p + 1)) / Tf, cjf.cvt[:, 1]))
            Jf = johansen(Zf, p, r); b_rel = max(b_rel, float(np.max(np.abs(Jf["beta"][:, 0] - x["beta"][:, 0]) / np.abs(x["beta"][:, 0]))))
            # 독립 대조용: statsmodels VECM 을 다시 맞춰 수준 VAR 표현(var_rep)으로 결정일의 다음 달 F11 → 변화율 (forecast_levels 와 다른 식)
            A_ = VECM(Zf, k_ar_diff=p, coint_rank=r, deterministic=M["det"]).fit().var_rep; tf_ = pd.Timestamp(x["test"][-1])
            Hf_ = Ld.loc[:tf_].iloc[-(p + 1):].values[:, x["keep"]]
            vr_pred.append(float(sum(A_[i] @ Hf_[-1 - i] for i in range(A_.shape[0]))[0] / Hf_[-1][0] - 1))
    autos = Counter(x["auto_rank"] for x in af)
    if None in autos:
        raise AssertionError("검정이 실패한 폴드가 있음 — 대본 문장을 고쳐야 합니다")
    c_tr, c_me, c_cor = Counter(rk_tr), Counter(rk_me), Counter(rk_cor)
    auto_items = sorted(autos.items())
    rk_rows = [[q, c_tr.get(q, 0), c_me.get(q, 0), c_cor.get(q, 0)] for q in range(K + 1) if c_tr.get(q) or c_me.get(q) or c_cor.get(q)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        llf = {q: float(np.real(VECM(Z, k_ar_diff=p, coint_rank=q, deterministic=M["det"]).fit().llf)) for q in range(1, K + 1)}
    lr_tr = [2 * (llf[K] - llf[q]) for q in range(1, K)]; lr_me = [2 * (llf[q + 1] - llf[q]) for q in range(1, K)]
    S("3. Johansen",
      f"<b>몇 개의 균형식이 있나 — 요한센 검정</b>: 대각합(trace) 검정은 '균형식이 r 개 이하' 를 r = 0, 1, … 차례로 검정해 처음 기각하지 못하는 r 을 고릅니다. 이 폴드에서는 5% 기준으로 <b>r = {sel}</b> 이 나옵니다. "
      f"그러나 설정은 원 노트북 2 와 같이 <b>r = {r} 로 고정</b>입니다(coint_rank = {p_['coint_rank']}) — 엔진은 이 검정을 하지 않고, 표는 엔진의 자동 선택과 같은 조건(결정항 없음, 차분 시차 {p})으로 따로 계산한 정보용입니다. "
      f"{len(af)}개 폴드의 대각합 검정: " + ", ".join(f"r = {k_} 이 {v_}개" for k_, v_ in auto_items)
      + f". r = {K}(변수 수)는 모든 수준값이 정상이라는 뜻이라 VECM 의 단위근(I(1)) 전제와 맞지 않습니다(자동 선택이면 엔진이 {K - 1} 로 줄임). "
      f"<b>검정에 따라 결과가 달라집니다</b>: 최대고유값(maxeig — 엔진이 rank_test 로 지원) 검정은 이 폴드에서 r = {sel_me}, {len(af)}개 폴드 중 {c_me.get(r, 0)}개에서 r = {r}. "
      f"소표본 보정(통계량 × (T − K·(p+1))/T = × {fac:.4f})을 한 대각합 검정은 이 폴드에서 r = {sel_cor}, {c_cor.get(r, 0)}개 폴드에서 r = {r}.",
      say=("균형식의 개수는 요한센 검정으로 고를 수 있습니다. 대각합 검정은 균형식 개수에 대한 가설을 0개부터 하나씩 늘려 가며 검정하고, 처음으로 기각하지 못한 개수를 고릅니다. "
           f"이 폴드에서는 5% 기준으로 {sel}개가 나옵니다. 그러나 이 설정은 원 노트북의 VECM 설정과 같이 {r}개로 고정되어 있어, 엔진은 이 검정을 하지 않습니다. 표는 따로 계산한 정보용입니다. "
           f"{len(af)}개 폴드에서 대각합 검정이 고른 개수는 " + ", ".join(f"{k_}개가 {v_}번" for k_, v_ in auto_items) + "입니다. "
           f"{K}개는 모든 수준값이 정상이라는 뜻이라, 단위근 전제와 맞지 않습니다. "
           f"검정을 바꾸면 결과가 달라집니다. 최대고유값 검정은 이 폴드에서 {sel_me}개를 고르고, {len(af)}개 폴드 중 {c_me.get(r, 0)}개에서 {r}개를 고릅니다. "
           f"표본 크기를 보정한 대각합 검정은 이 폴드에서 {sel_cor}개, {c_cor.get(r, 0)}개 폴드에서 {r}개입니다.")
      # 이 대본이 전제하는 것: 대각합 검정이 고른 개수가 설정값과 다르고("그러나"), 다른 검정은 대각합 검정과 다른 개수를 고르며("검정을 바꾸면 달라집니다"), r = K 인 폴드가 있다.
      # (2026-11-10: 봉인 경계가 바뀌어 마지막 폴드가 한 달 앞당겨졌고 보정 검정이 0개를 골랐다 — 문장은 개수를 변수로 읽으므로 그대로 맞다. 전제를 문장이 실제로 요구하는 것으로 좁힘)
      if sel != r and (sel_me != sel or sel_cor != sel) and K in autos else None,
      notes={src.line(MODEL, 'r = int(p["coint_rank"])'): f"설정값 coint_rank = {p_['coint_rank']}",
             src.line(MODEL, "r = int(select_coint_rank("): f"coint_rank = 0 일 때만 실행 — 이 설정에서는 건너뜀. 자동일 때의 조건: det_order {-1 if M['det'] == 'n' else 0}(상수 없음) · k_ar_diff {p} · {p_['rank_test']} · {p_['rank_signif']}",
             src.line(MODEL, "r = min(r, K - 1)"): f"r 상한 = 변수 수 − 1 = {K - 1} (검정이 {K}개 = 모두 정상을 고르면 {K - 1}개로 줄임)"},
      code=C(MODEL, src.lines(MODEL, 'r = int(p["coint_rank"])', "if r == 0:", "r = int(select_coint_rank(", "r = min(r, K - 1)", "self.rank = r"),
             src.line(MODEL, "if r == 0:"), "fit_levels · 차수", 3, 3),
      lab=lab_trace(eig_cj, J["T"], cv_[:, 1], tr_, sel),
      stack=["statsmodels · coint_johansen / select_coint_rank (정보용)"], nodes=["joh"],
      formula="대각합(r) = −T Σ_(i>r) ln(1 − λ_i),   최대고유값(r) = −T ln(1 − λ_(r+1));   통계량 > 임계값(5%) 이면 'r 개 이하' 기각",
      tensors=[{"type": "table", "title": "이 폴드의 검정 (결정항 없음, coint_johansen — 실행 기록)", "columns": ["귀무가설: 균형식 ≤", "대각합", "임계값 95%", "판단", "최대고유값", "임계값 95%", "판단", f"보정 대각합 (× {fac:.4f})", "판단"],
                "rows": [[i, f"{tr_[i]:.3f}", f"{cv_[i, 1]:.3f}", "기각" if tr_[i] > cv_[i, 1] else "기각 못 함", f"{me_[i]:.3f}", f"{cvm_[i, 1]:.3f}", "기각" if me_[i] > cvm_[i, 1] else "기각 못 함",
                          f"{tr_[i] * fac:.3f}", "기각" if tr_[i] * fac > cv_[i, 1] else "기각 못 함"] for i in range(K)]},
               {"type": "table", "title": f"{len(af)}개 폴드에서 검정이 고르는 r (5%, 설정은 {r} 고정 — 영상 계산)", "columns": ["r", "대각합", "최대고유값", "보정 대각합"], "rows": rk_rows}],
      checks=[chk(f"대각합 통계량 = −T Σ ln(1−λ): 같은 λ 로 식·T(= {J['T']}) 확인", [-J["T"] * np.sum(np.log(1 - eig_cj[i:])) for i in range(K)], tr_, 1e-10, rel=True, kind="식"),
              chk(f"로그우도 비: VECM.fit 을 r = 1~{K} 로 맞춘 2(llf_K − llf_r)·2(llf_(r+1) − llf_r) = coint_johansen 의 대각합·최대고유값 통계량 (r = 1~{K - 1}) — "
                  "statsmodels 의 VECMResults.llf 는 VECM.fit 과 같은 고유값(_sij)으로 만든 닫힌 식이라, 앞 화면의 고유값 대조를 식만 바꿔 다시 한 것",
                  lr_tr + lr_me, list(tr_[1:]) + list(me_[1:]), 1e-5, rel=True, kind="식"),
              chk(f"폴드별 학습 창 재구성: select_coint_rank 를 다시 불러 대각합 r = 기록({len(af)}개 폴드)", rk_tr, [x["auto_rank"] for x in af], 0, kind="재실행"),
              chk(f"폴드별 학습 창 재구성: VECM.fit 순서의 손 계산 β = 엔진 β ({len(af)}개 폴드, 상대 최대 — 같은 순서의 재현)", b_rel, 0.0, 1e-8, kind="식")],
      hint="coint_rank = 0 으로 두면 엔진이 폴드마다 rank_test(trace 또는 maxeig)로 r 을 고릅니다(configs/f11_vecm_auto_k1.json). 검정이 r = 변수 수(전부 정상)를 고르면 엔진은 r = 변수 수 − 1 로 줄입니다. "
           "폴드별 수는 실행 기록의 수준값 행렬과 폴드 날짜로 학습 창을 다시 만들어 statsmodels 로 계산했습니다(재구성한 창의 대각합 r 과 β 가 기록과 같음 — 검증 줄). "
           "보정 대각합은 엔진에 없는 계산으로, 통계량에 (T − K·(p+1))/T 를 곱하는 소표본 보정을 이 화면에서만 적용한 것입니다. "
           "작은 표본에서 요한센 검정에 유한 표본 편의가 있다는 연구가 있습니다(예: Cheung·Lai 1993, Oxford Bulletin of Economics and Statistics 55(3):313–328). 편의의 방향·크기를 이 자료에서 확인하지는 않았습니다.")
    al, be, ga = M["alpha"], M["beta"], M["gamma"]
    ect = Z @ be[:, 0]
    H = hist.values[:, M["keep"]]
    h0 = H[-1]
    ectv = float(be[:, 0] @ h0); comps = be[:, 0] * h0
    post = list(Ld.loc[Ltr.index[-1]:t].index[1:])                # 학습 창 뒤 ~ 결정일
    ect_post = [float(be[:, 0] @ Ld.loc[d].values[M["keep"]]) for d in post]
    ect_pos = bool(np.all(ect > 0))
    n_never = sum(bool(np.all(Zf @ x["beta"][:, 0] > 0) or np.all(Zf @ x["beta"][:, 0] < 0)) for Zf, x in zip(fold_Z, af))
    # 학습 창 T 개 식의 평균: 평균 Δy_t = α·평균(β′y_(t−1)) + 평균(Γ·ΔX) + 평균(잔차) — 엔진 α·β·Γ 로 계산(상수가 없어 잔차 평균은 0 이 아님)
    dY_ = np.diff(Z.T)[:, p:]; Y1_ = Z.T[:, p:-1]
    m_dy = dY_.mean(1); m_ect = al[:, 0] * (be[:, 0] @ Y1_).mean(); m_g = (ga @ J["dx"]).mean(1); m_res = m_dy - m_ect - m_g
    jM3 = cols.index("M3")
    jo_dy, jo_ect, jo_g = m_dy[jM3] / 1000, m_ect[jM3] / 1000, m_g[jM3] / 1000          # 조 원
    # 독립 대조: β 를 엔진에 주고 α·Γ 를 한꺼번에 최소제곱(ΔY 를 [β′Y₋₁; ΔX] 에 회귀, np.linalg.lstsq) — VECM.fit 의 단계별 식과 다른 계산
    Xj = np.vstack([be[:, 0] @ Y1_, J["dx"]]); coef_j = np.linalg.lstsq(Xj.T, dY_.T, rcond=None)[0].T
    S("4. 장기 균형",
      f"<b>β (장기 균형식)</b>는 첫 원소가 1 이 되게 맞춥니다: β′y = F11 + Σ β_j × 변수_j. 이 모형은 상수가 없어(deterministic = n) <b>오차수정항은 β′y_(t−1) 그대로</b>이고, 균형은 β′y = 0 입니다. "
      f"<b>α</b>는 오차수정항이 1 단위 클 때 각 변수의 다음 달 변화에 더해지는 양입니다. F11 의 α = {g(al[0, 0], 4)} — β′y 가 양수면 다음 달 F11 을 낮추는 쪽으로 작용합니다. "
      f"학습 창 {len(Ltr)}개월의 β′y 는 {ect.min():.3f} ~ {ect.max():.3f}(평균 {ect.mean():.3f})" + (" 로 늘 양수였습니다 — 이 창 안에서 β′y = 0(균형)에 닿은 달은 없습니다" if ect_pos else "")
      + f"({len(af)}개 폴드 중 {n_never}개 폴드에서 학습 창의 β′y 가 0 을 한 번도 지나지 않음). 결정일 {t.date()} 의 β′y = {ectv:.4f}. "
      f"<b>상수가 없어서 α·β′y 가 평균적으로 0 이 아닌 몫(상수 같은 역할)까지 맡습니다</b>: 학습 창 {J['T']}개 식의 평균으로 M2 의 월 변화 약 {jo_dy:.0f}조 원({jo_dy:.2f}조) 중 α·β′y 몫이 {jo_ect:.0f}조 원({jo_ect:.2f}조)으로 오히려 크고, 단기 항 몫이 {jo_g:.0f}조 원({jo_g:.2f}조)입니다. "
      f"F11 은 월 변화 {g(m_dy[0], 4)} 중 α·β′y 몫 {g(m_ect[0], 4)} · 단기 항 몫 {g(m_g[0], 4)}. 그래서 α 를 순수한 '되돌아오는 속도' 로만 읽을 수는 없습니다. "
      f"표의 '정확한 β' 는 coint_johansen 고유벡터로, 엔진 β 와 상대 {d_beta_ex:.1e} 다릅니다 — 엔진 β 의 여섯째 자리는 믿기 어렵습니다(예측에 실제로 쓰인 값은 엔진 β).",
      say=("장기 균형식 베타는 F11의 계수가 1이 되게 맞춥니다. 상수가 없는 모형이라 균형식의 값 그대로가 오차수정항이고, 이 값이 0이면 균형입니다. "
           f"알파는 오차수정항이 한 단위 클 때 각 변수의 다음 달 변화에 더해지는 양입니다. F11의 알파는 {g(al[0, 0], 4)}로 음수라서, 오차수정항이 양수면 F11을 낮추는 쪽으로 작용합니다. "
           f"그런데 이 폴드의 학습 창에서 이 값은 {ect.min():.3f}부터 {ect.max():.3f}까지 늘 양수였고, {len(af)}개 폴드 중 {n_never}개 폴드에서도 0을 한 번도 지나지 않았습니다. "
           f"상수가 없으니 이 항이 평균적으로 상수 같은 몫까지 맡습니다. 예를 들어 통화량의 평균 월 변화는 약 {jo_dy:.0f}조 원인데, 이 항의 몫이 {jo_ect:.0f}조 원으로 오히려 크고 단기 항이 {jo_g:.0f}조 원으로 그만큼을 덜어 냅니다. "
           "그래서 알파를 순수한 되돌아오는 속도로만 읽을 수는 없습니다.")
      if al[0, 0] < 0 and ect_pos and m_ect[jM3] > m_dy[jM3] > 0 > m_g[jM3] else None,
      notes={src.line(MODEL, "ect = self.beta.T @ h[-1]"): f"오차수정항 = β′y_t (상수가 없어 c_in = None → 더하는 것 없음). 결정일 {t.date()}: {ectv:.4f}",
             src.line(MODEL, "d = self.alpha @ ect + sum("): f"α × 오차수정항: F11 은 {g(al[0, 0], 4)} × {ectv:.4f} = {al[0, 0] * ectv:.4g}"},
      code=C(MODEL, src.lines(MODEL, "ect = self.beta.T @ h[-1]", "d = self.alpha @ ect + sum("), src.line(MODEL, "ect = self.beta.T @ h[-1]"), "forecast_levels · 오차수정항", 3, 4),
      lab=lab_johansen(Z, cols, p, r, be[:, 0], al[:, 0], d_lay),
      stack=stk + ["statsmodels · VECMResults.alpha / beta"], nodes=["beta"], edges=[("joh", "beta")],
      formula=f"β = β̃ (β̃의 첫 r 행)⁻¹,   α = S₀₁ β (β′ S₁₁ β)⁻¹;   결정일 β′y_t = " + " + ".join(f"({v_:.4f})" if v_ < 0 else f"{v_:.4f}" for v_ in comps) + f" = {ectv:.4f}",
      tensors=[{"type": "table", "title": f"β 와 α (원래 단위) · 결정일 {t.date()} 의 β_j × 값", "columns": ["열", "이름", "엔진 β (예측에 쓰임)", "정확한 β (coint_johansen 고유벡터)", "β 손 계산 (VECM.fit 순서)", "α", "α 손 계산", "β_j × 결정일 값"],
                "rows": [[c, nm_, g(be[i, 0], 6), g(b_cj[i, 0], 6), g(J["beta"][i, 0], 6), g(al[i, 0], 6), g(J["alpha"][i, 0], 6), f"{comps[i]:.4f}"] for i, (c, nm_) in enumerate(zip(cols, names))]},
               {"type": "table", "title": f"학습 창 {J['T']}개 식의 평균 월 변화 나누기 — 엔진 α·β·Γ 로 영상 계산 (원래 단위: M2·가계대출은 십억원 · 상수가 없어 잔차 평균이 0 이 아님)", "columns": ["열", "평균 Δy_t", "α·β′y_(t−1) 몫", "Γ 몫", "잔차 평균"],
                "rows": [[c, g(m_dy[i], 5), g(m_ect[i], 5), g(m_g[i], 5), g(m_res[i], 3)] for i, c in enumerate(cols)]},
               {"type": "line", "title": "오차수정항 β′y_t (상수 없음 — 균형 = 0)", "x": [d.strftime("%Y-%m") for d in list(Ltr.index) + post], "xlabel": "월",
                "series": [{"name": "β′y (학습 창)", "color": "#6a3fa0", "y": [num(v_) for v_ in ect] + [None] * len(post), "dots": True},
                           {"name": "β′y (비움·결정일 — 같은 β)", "color": "#c8710f", "y": [None] * (len(ect) - 1) + [num(ect[-1])] + [num(v_) for v_ in ect_post], "dots": True},
                           {"name": "0 (균형)", "color": "#8a93a3", "y": [0.0] * (len(ect) + len(post))}], "height": 200}],
      checks=[chk("VECM.fit 순서의 손 계산 β = 엔진 β (상대 — 같은 순서의 재현)", J["beta"], be, 1e-8, rel=True, kind="식"),
              chk("VECM.fit 순서의 손 계산 α = 엔진 α (상대 — 같은 순서의 재현)", J["alpha"], al, 1e-8, rel=True, kind="식"),
              chk("엔진 β = coint_johansen 고유벡터로 만든 β (상대, 다른 구현 — 허용은 위 반올림 오차 크기)", be[:, 0], b_cj[:, 0], 1e-4, rel=True),
              chk("열 단위를 맞춘 손 계산 β = coint_johansen 고유벡터로 만든 β (상대 — '정확한 β' 의 근거)", b_ex[:, 0], b_cj[:, 0], 1e-8, rel=True, vs="coint_johansen")],
      hint="M2·가계대출은 원자료 단위(십억원)로 백만 단위의 값이라 β 가 1e−5 수준으로 작게 나옵니다 — 단위 차이일 뿐, 결정일 값에 곱하면 F11 과 같은 자릿수입니다(표의 마지막 열). "
           "이 모형은 상수가 없어(deterministic = n) 균형식에도 상수가 없습니다. β′y 가 0 에서 떨어져 머무는 몫을 상수(ci·co)로 따로 두는 설정과는 비교하지 않았습니다. "
           "폴드별 β′y 는 다시 만든 폴드별 학습 창에 기록된 폴드별 β 를 곱해 셌습니다.")
    S("4. 단기 동학",
      f"<b>Γ (단기 동학)</b>: 지난 {p}개월의 변화 Δy 가 이번 달 변화에 주는 영향입니다. Γ_i 의 행 = 이번 달 변하는 변수, 열 = i달 전 변화의 변수. α·β 를 정한 뒤 남은 부분을 최소제곱으로 구합니다. "
      f"예: F11 행의 ΔF11 열 — Γ₁ {g(ga[0, 0], 5)}, Γ₂ {g(ga[0, K], 5)}. 계수의 크기는 변수 단위에 따라 달라 열끼리 바로 비교할 수 없습니다. "
      "검증: β 를 엔진 값으로 두고 ΔY 를 [β′Y₋₁; ΔX] 에 한꺼번에 최소제곱 회귀해도(np.linalg.lstsq — 다른 계산법) 같은 α·Γ 가 나옵니다.",
      say=(f"감마는 단기 움직임으로, 지난 {nat_months(p)}의 변화가 이번 달 변화에 주는 영향입니다. 행은 이번 달에 변하는 변수, 열은 몇 달 전 어느 변수의 변화인지를 뜻합니다. "
           "알파와 베타를 먼저 정한 뒤, 이번 달 변화에서 오차수정항 몫을 빼고 남은 부분을 최소제곱으로 구합니다. "
           f"예를 들어 지난달 F11 변화가 이번 달 F11 변화에 주는 계수는 {g(ga[0, 0], 5)}, 두 달 전 것은 {g(ga[0, K], 5)}입니다. "
           "계수 크기는 변수 단위에 따라 달라서, 열끼리 바로 비교할 수는 없습니다."),
      notes={src.line(MODEL, "d = self.alpha @ ect + sum("): f"gamma({K} × {p * K}) = [Γ₁ | Γ₂]. 예측할 때 gamma[:, 0:{K}] = Γ₁ 은 dy[0] = Δy_t 에, gamma[:, {K}:{2 * K}] = Γ₂ 는 dy[1] = Δy_(t−1) 에 곱함"},
      code=C(MODEL, src.lines(MODEL, "dy = [h[-i] - h[-i - 1] for i in range(1, p + 1)]", "d = self.alpha @ ect + sum("), src.line(MODEL, "d = self.alpha @ ect + sum("), "forecast_levels · 단기 동학", 4, 3),
      lab=lab_gamma(Z, cols, p, al, be, ga),
      stack=stk + ["statsmodels · VECMResults.gamma"], nodes=["gamma"], edges=[("beta", "gamma")],
      formula="Γ = (ΔY − αβ′Y₋₁) ΔX′ (ΔX ΔX′)⁻¹,   ΔX = [Δy_(t−1); …; Δy_(t−p)]",
      tensors=[mat_table(f"Γ_{i + 1} ({i + 1}달 전 변화 → 이번 달 변화)", ga[:, i * K:(i + 1) * K], cols, [f"Δ{c}" for c in cols]) for i in range(p)],
      checks=[chk("VECM.fit 순서의 손 계산 Γ = 엔진 Γ (상대 — 열마다 크기 기준, 같은 순서의 재현)", J["gamma"] / np.maximum(np.abs(ga).max(0, keepdims=True), 1e-300), ga / np.maximum(np.abs(ga).max(0, keepdims=True), 1e-300), 1e-7, kind="식"),
              chk("한꺼번에 최소제곱(lstsq)한 α = 엔진 α (상대, 다른 계산법)", coef_j[:, 0], al[:, 0], 1e-8, rel=True),
              chk("한꺼번에 최소제곱(lstsq)한 Γ = 엔진 Γ (상대 — 열마다 크기 기준, 다른 계산법)", coef_j[:, 1:] / np.maximum(np.abs(ga).max(0, keepdims=True), 1e-300), ga / np.maximum(np.abs(ga).max(0, keepdims=True), 1e-300), 1e-8)])
    # ================================================================ 5. 예측
    dys = [H[-i] - H[-i - 1] for i in range(1, p + 1)]
    parts = [al[:, 0] * ectv] + [ga[:, i * K:(i + 1) * K] @ dys[i] for i in range(p)]
    dhat = sum(parts); yhat = h0 + dhat; pred = float(yhat[0] / h0[0] - 1)
    d3 = [d.strftime("%Y-%m") for d in hist.index[-(p + 1):]]
    if abs(pred - float(cap["pred"].loc[t])) > 1e-12:                 # 표(Δŷ 구성)가 엔진 예측과 맞는지 — 검증 줄은 아래 var_rep 대조
        raise AssertionError("Δŷ 구성 표의 합이 엔진 예측과 다름")
    seal = pd.Timestamp(cfg["validation"]["sealed_from"]); nxt_ts = pd.Timestamp(nxt_m + "-01")
    y_t_hand = float(Lall.loc[nxt_ts, "__target__"] / Lall.loc[t, "__target__"] - 1)
    if abs(y_t_hand - float(y.loc[t])) > 1e-15:
        raise AssertionError("정답 = 다음 달 F11 ÷ 결정월 F11 − 1 이 아님")
    in_seal = nxt_ts >= seal
    seal_txt = (f" 이 정답은 봉인 구간({seal.strftime('%Y-%m')}~)의 값({nxt_m} F11)을 씁니다 — 엔진은 결정일만 봉인 전으로 자릅니다." if in_seal else "")
    S("5. 예측",
      f"결정일 {t.date()}: 결정월 행까지의 마지막 {len(hist)}행({hist.index[0].strftime('%Y-%m')} ~ {hist.index[-1].strftime('%Y-%m')})을 넘기고, 재귀에는 그중 마지막 {p + 1}행(수준 1개 + 변화 {p}개)을 씁니다. "
      f"비운 {gap[-1].strftime('%Y-%m')} 과 결정일 {t.strftime('%Y-%m')} 행도 들어가지만 계수는 그대로이고, 정답이 쓰는 {nxt_m} 값은 들어가지 않습니다. "
      f"Δŷ = α(β′y_t) + Γ₁Δy_t + Γ₂Δy_(t−1), ŷ_(t+1) = y_t + Δŷ. F11: {g(h0[0], 7)} + ({g(dhat[0], 5)}) = {g(yhat[0], 7)} → 엔진이 ŷ ÷ y_t − 1 = 예측 변화율 <b>{pct(pred)}</b> (실제 {pct(y.loc[t])}).{seal_txt} "
      f"독립 대조(계수는 같은 VECM.fit — 예측 식만 다른 구현): statsmodels VECM 을 다시 맞춰 수준 VAR 표현(var_rep: y_(t+1) = A₁y_t + A₂y_(t−1) + A₃y_(t−2))으로 계산한 예측과 {abs(vr_pred[-1] - pred):.1e}, {len(af)}개 폴드 전체로 최대 {float(np.max(np.abs(np.array(vr_pred) - np.array([x['pred'][-1] for x in af])))):.1e} 차이.",
      say=(f"이제 결정일 {t.date()}의 예측입니다. 엔진이 넘긴 마지막 {len(hist)}개 행 가운데, 모델은 마지막 {p + 1}개 행만 씁니다. "
           f"비운 {gap[-1].strftime('%Y-%m')} 행과 결정월 행도 입력이지만, 정답이 쓰는 {nxt_m} 값은 들어가지 않습니다. "
           "다음 달 변화는 알파 곱하기 오차수정항에, 첫째 감마 곱하기 이번 달 변화와, 둘째 감마 곱하기 지난달 변화를 더한 것입니다. "
           f"F11에서는 세 항이 {g(parts[0][0], 4)}, {g(parts[1][0], 4)}, {g(parts[2][0], 4)}로, 오차수정항만 낮추는 쪽입니다. "
           f"합 {dhat[0]:.4f}을 지금 값 {g(h0[0], 6)}에 더하면 다음 달 F11은 {yhat[0]:.4f}이고, 엔진은 이것을 지금 값으로 나누고 1을 빼 예측 변화율 {pct(pred)}를 냅니다. "
           f"실제는 {pct(y.loc[t])}였는데, "
           + (f"이 정답은 봉인 구간인 {nxt_m}의 값을 씁니다. 엔진은 결정일만 봉인 전으로 자르기 때문입니다. " if in_seal else "")
           + "이 예측은 같은 계수를 statsmodels의 다른 표현으로 계산한 값과도 같았습니다.")
      if parts[0][0] < 0 < parts[1][0] and parts[2][0] > 0 else None,
      notes={src.line(MODEL, "h = [row for row in Z[-(p + 1):]]"): f"넘겨받은 {len(hist)}행 중 마지막 {p + 1}행({', '.join(d3)})만 씀",
             src.line(MODEL, "d = self.alpha @ ect + sum("): f"F11: {g(parts[0][0], 4)} + {g(parts[1][0], 4)} + {g(parts[2][0], 4)} = {g(dhat[0], 4)} (상수·계절 없음 → 다음 if 건너뜀)",
             src.line(MODEL, "nxt = h[-1] + d"): f"ŷ_(t+1) = y_t + Δŷ → F11 {g(h0[0], 7)} + {g(dhat[0], 4)} = {g(yhat[0], 7)}"},
      code=C(MODEL, [src.line(MODEL, "h = [row for row in Z[-(p + 1):]]")] + src.span(MODEL, "for st in range(1, steps + 1):", "out.append(nxt[0])"), src.line(MODEL, "nxt = h[-1] + d"), "forecast_levels", 14, 2),
      lab=lab_forecast(H[-(p + 1):], d3, cols, al[:, 0], be[:, 0], ga[:, :K], ga[:, K:2 * K], float(cap["pred"].loc[t]), float(h0[0])),
      stack=["engine/experiment.py · _walkforward_predict", f"{MODEL} · forecast_levels"], nodes=["pred"], edges=[("gamma", "pred")],
      formula=f"F11 의 Δŷ = α₁·(β′y_t) + [Γ₁Δy_t]₁ + [Γ₂Δy_(t−1)]₁ = {g(parts[0][0], 4)} + {g(parts[1][0], 4)} + {g(parts[2][0], 4)}",
      tensors=[{"type": "table", "title": "넣은 행 (수준값 — 열마다 기준일이 다름, '1. 수준값' 화면)", "columns": ["월", *cols], "rows": [[d.strftime("%Y-%m"), *[g(v_, 7) for v_ in r_]] for d, r_ in zip(hist.index, hist.values)]},
               {"type": "table", "title": "다음 달 변화 Δŷ 의 구성 (열마다)", "columns": ["", *cols], "rows": [["α·(β′y_t)", *[g(v_, 4) for v_ in parts[0]]]]
                + [[f"Γ_{i + 1}·Δy_t" if i == 0 else f"Γ_{i + 1}·Δy_(t−{i})", *[g(v_, 4) for v_ in parts[i + 1]]] for i in range(p)] + [["합 Δŷ", *[g(v_, 4) for v_ in dhat]], ["ŷ_(t+1)", *[g(v_, 7) for v_ in yhat]]]}],
      checks=[chk("statsmodels VECM 을 다시 맞춘 수준 VAR 표현(var_rep)의 예측 = 엔진 예측 (계수는 같은 VECM.fit — 예측 식만 forecast_levels 와 다른 구현)", vr_pred[-1], cap["pred"].loc[t], 1e-12),
              chk(f"같은 대조를 {len(af)}개 폴드 전체에서 (최대 차이)", vr_pred, [x["pred"][-1] for x in af], 1e-12)])
    if any(st_["say"] is None for st_ in steps):
        raise AssertionError(f"대본 없는 단계 {[i for i, st_ in enumerate(steps) if st_['say'] is None]} — 대본 문장이 전제한 기록의 성질(부호·검정 결과·시점)이 바뀌었습니다. 대본을 고치세요")
    # ================================================================ 6. 모든 폴드
    dts = [pd.Timestamp(x["test"][-1]) for x in af]; pp = np.array([x["pred"][-1] for x in af]); yy = y.reindex(dts).values; nv = y.shift(k).reindex(dts).values
    ok = np.isfinite(pp) & np.isfinite(yy) & np.isfinite(nv)
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    B = np.array([x["beta"][:, 0] for x in af]); A0 = np.array([x["alpha"][0, 0] for x in af])
    if {x["rank"] for x in af} != {r}:
        raise AssertionError("폴드마다 차수가 다름 — β 표의 해석을 고쳐야 합니다")
    flips = [int(np.sum(np.sign(B[:, j]) != np.sign(np.median(B[:, j])))) for j in range(K)]
    j12 = cols.index("F12")
    S("6. 모든 폴드",
      f"폴드마다 다시 추정한 F11 의 α₁ 과 균형식 계수 β 입니다. α₁ 이 음수인 폴드 {int(np.sum(A0 < 0))}개 / {len(af)}개. "
      "균형식 계수가 폴드마다 크게 바뀌거나 부호가 바뀌면 '안정된 장기 관계' 로 보기 어렵습니다.",
      say=(f"폴드마다 다시 추정하면 계수도 달라집니다. F11의 알파가 음수인 폴드는 {len(af)}개 가운데 {int(np.sum(A0 < 0))}개입니다. "
           f"균형식 계수는 폴드에 따라 크게 바뀝니다. 예를 들어 {sp['F12']}의 계수는 {g(np.min(B[:, j12]))}에서 {g(np.max(B[:, j12]))}까지 움직이고, "
           f"중앙값과 부호가 다른 폴드가 {flips[j12]}개입니다. 이렇게 계수가 폴드마다 크게 바뀌거나 부호가 바뀌면, 안정된 장기 관계로 보기 어렵습니다."),
      stack=["tools/codewalk/vecm_capture.py · 폴드마다 같은 엔진 경로"], nodes=["allf"], edges=[("pred", "allf")],
      tensors=[{"type": "line", "title": "폴드별 F11 의 α₁", "x": [d.strftime("%Y-%m") for d in dts], "xlabel": "테스트 월", "series": [{"name": "α₁", "color": "#b3261e", "y": [num(v_) for v_ in A0], "dots": True}], "height": 190},
               {"type": "table", "title": "β 의 폴드 간 범위", "columns": ["열", "최소", "중앙", "최대", "중앙값과 부호가 다른 폴드"],
                "rows": [[c, g(np.min(B[:, j])), g(np.median(B[:, j])), g(np.max(B[:, j])), flips[j]] for j, c in enumerate(cols)]}],
      hint="β 는 F11 계수를 1 로 맞춘 값입니다. 정규화 전 β̃ 의 F11 원소가 0 에 가까운 폴드에서는 나머지 계수가 크게 나올 수 있습니다 — 범위가 큰 이유의 가능한 설명이며, 폴드별로 확인하지는 않았습니다. "
           "α₁ 의 부호만 셌고, 유의성(표준오차)은 보지 않았습니다. α 에는 상수 같은 몫도 섞여 있습니다('4. 장기 균형' 화면).")
    e_m, e_n, e_0 = rmse(pp[ok] - yy[ok]), rmse(nv[ok] - yy[ok]), rmse(yy[ok])
    hit = lambda a_: float(np.mean(np.sign(a_[ok]) == np.sign(yy[ok])))
    n_up = int(np.sum(yy[ok] > 0)); up = n_up / int(ok.sum())
    worse = e_m > e_n
    mr = match_result("forecast", cfg["name"], {d: v_ for d, v_ in zip(dts, pp)})
    if mr is not None:
        raise AssertionError(f"결과 폴더에 이 실험의 결과 파일이 생겼습니다({mr['file']}) — 대본의 '결과 파일이 없어 대조하지 못했다' 가 틀리게 됩니다. "
                             "대본을 고쳐 대조 단계를 넣으세요(table_script.py 의 결과 파일 대조 단계 참고)")
    S("6. 모든 폴드",
      f"{len(af)}개 폴드의 한 달 예측과 실제, naive(결정월까지의 최근 {k}개월 변화를 다음 달에 그대로 = y_(t−{k}) = F11_t ÷ F11_(t−{k}) − 1). RMSE: VECM {pct(e_m, 3, False)}, naive {pct(e_n, 3, False)}, 0 예측 {pct(e_0, 3, False)}"
      + (" — <b>naive 보다 오차가 큽니다.</b>" if worse else " — naive 보다 오차가 작습니다(유의성은 결과 화면의 DM 검정).")
      + f" 방향 적중: VECM {hit(pp) * 100:.1f}%, naive {hit(nv) * 100:.1f}%, 항상 상승 {up * 100:.1f}%({len(af)}개 중 {n_up}개 달이 상승)."
      + " 이 RMSE 는 영상이 계산한 값입니다. 이 설정은 아직 결과 폴더에 실행 결과가 없어 결과 화면과의 대조는 실행 후에 할 수 있습니다."
      + (f" 마지막 폴드({t.strftime('%Y-%m')})의 정답은 봉인 구간({seal.strftime('%Y-%m')}~)의 값을 씁니다 — 엔진은 결정일만 봉인 전으로 자름('5. 예측')." if in_seal else ""),
      say=(f"{len(af)}개 폴드의 한 달 예측을 실제와 나란히 놓았습니다. 비교 기준은 두 가지로, 항상 0이라고 예측하는 것과, 결정월까지의 최근 한 달 변화가 다음 달에도 그대로라고 보는 나이브 예측입니다. "
           f"오차의 크기인 RMSE 값은 VECM {pct(e_m, 3, False)}, 나이브 {pct(e_n, 3, False)}, 0 예측 {pct(e_0, 3, False)}로, "
           + ("VECM이 나이브보다 오차가 큽니다. " if worse else "VECM이 나이브보다 오차가 작습니다. 우연인지는 결과 화면의 DM 검정으로 봅니다. ")
           + f"방향 적중률은 VECM {hit(pp) * 100:.1f}%, 나이브 {hit(nv) * 100:.1f}%이고, 항상 오른다고만 해도 {up * 100:.1f}%입니다. "
           + "이 값들은 영상에서 계산한 것이고, 이 설정의 실행 결과가 아직 결과 폴더에 없어 결과 화면과는 대조하지 못했습니다."),
      notes={src.line(CAL, 'return y.shift(k).rename("naive")'): f"정답을 {k}칸 미룸 = 결정월까지의 최근 {k}개월 변화(F11_t ÷ F11_(t−{k}) − 1)를 다음 달 예측으로"},
      code=C(CAL, src.span(CAL, "def naive_forecast(y: pd.Series, k: int) -> pd.Series:", 'return y.shift(k).rename("naive")'), src.line(CAL, 'return y.shift(k).rename("naive")'), "naive_forecast", 0, 0),
      lab=lab_rmse(yy[ok], pp[ok], nv[ok], e_m, e_n, e_0),
      stack=["engine/calendar.py · naive_forecast"], nodes=["allf"],
      tensors=[{"type": "line", "title": "폴드별 예측과 실제", "x": [d.strftime("%Y-%m") for d in dts], "xlabel": "테스트 월",
                "series": [{"name": "실제", "color": "#3b6fb6", "y": [num(v_) for v_ in yy], "dots": True}, {"name": "VECM", "color": "#c8710f", "y": [num(v_) for v_ in pp], "dots": True},
                           {"name": "naive", "color": "#8a93a3", "y": [num(v_) for v_ in nv]}], "height": 230},
               {"type": "table", "title": f"요약 ({int(ok.sum())}개 폴드 — 영상 계산)", "columns": ["예측", "RMSE", "방향 적중"],
                "rows": [["VECM", pct(e_m, 4, False), f"{hit(pp) * 100:.1f}%"], ["naive", pct(e_n, 4, False), f"{hit(nv) * 100:.1f}%"], ["항상 0", pct(e_0, 4, False), "—"],
                         ["항상 상승", "—", f"{up * 100:.1f}% ({n_up}/{int(ok.sum())})"]]}],
      checks=[chk("모든 폴드 기록의 마지막 예측 = 마지막 폴드만 따로 붙잡은 기록의 예측 (같은 엔진 경로를 두 번 실행)", pp[-1], cap["pred"].loc[t], 1e-14, kind="재실행")])
    n_ok = sum(c["ok"] for c in checks_all)
    by_k = Counter(c["kind"] for c in checks_all if c["kind"] != "출처")       # 출처 확인 줄(노트북 대조 — 파일이 있을 때만)은 개수에 넣지 않음: 대본이 컴퓨터마다 같게
    n_cnt = sum(by_k.values())
    n_ind_eng = sum(1 for c in checks_all if c["kind"] == "독립" and c.get("vs") == "엔진")
    if by_k.get("독립", 0) - n_ind_eng != sum(1 for c in checks_all if c["kind"] == "독립" and c.get("vs") == "coint_johansen"):
        raise AssertionError("독립 대조의 비교 대상(엔진 / coint_johansen)이 표시되지 않은 줄이 있음")
    n_lab = sum(1 for st_ in steps if st_.get("lab"))
    kind_txt = (f"독립 대조 {by_k.get('독립', 0)} · 식 확인·재현 {by_k.get('식', 0)}(직접 실행해 보기 {n_lab}줄 포함) · 재실행 {by_k.get('재실행', 0)} "
                f"— 독립이 아닌 줄 합 {by_k.get('식', 0) + by_k.get('재실행', 0)}")
    S("7. 정리",
      f"정리: VECM 은 F11 과 변수 {K - 1}개의 수준값으로 Johansen 축소 순위 회귀를 해 장기 균형식 β(r = {r})·α·단기 동학 Γ 를 정하고, 결정일마다 결정월 행까지의 마지막 {p + 1}개 행(열마다 기준일이 다름)으로 한 달 뒤를 재귀 계산해 변화율로 바꿉니다. "
      f"검증 {n_cnt}줄 — {kind_txt}. 모두 일치(출처 확인 줄은 따로 — 아래 표, 개수에 넣지 않음). 독립 대조 {by_k.get('독립', 0)}줄 가운데 {n_ind_eng}줄은 엔진 값을 다른 경로"
      "(parquet 직접 읽기·coint_johansen 고유벡터·lstsq·var_rep)로 맞춘 것이고, "
      f"{by_k.get('독립', 0) - n_ind_eng}줄(고유값, 열 단위를 맞춘 β)은 이 대본의 손 계산을 statsmodels 의 다른 구현 coint_johansen 과 맞춘 것입니다. "
      "식 확인·재현은 같은 배열에 같은 식이나 같은 순서의 계산을 다시 한 것이라 엔진의 계산 순서를 보여 줄 뿐 독립 확인은 아닙니다. "
      f"{len(af)}개 폴드의 RMSE 는 naive 보다 {'큽니다' if worse else '작습니다'} · 결과 파일 대조는 아직 없음(실행 결과 없음).",
      say=(f"정리하면, VECM은 F11과 변수 {K - 1}개의 수준값으로 요한센 축소 순위 회귀를 해서 장기 균형식 베타 {r}개, 알파, 단기 움직임 감마를 정하고, "
           f"결정일마다 결정월 행까지의 마지막 {p + 1}개 행으로 한 달 뒤를 계산해 변화율로 바꿉니다. 검증 {n_cnt}줄은 모두 일치했습니다. "
           f"그중 다른 경로로 대조한 것은 {by_k.get('독립', 0)}줄이고, 나머지 {by_k.get('식', 0) + by_k.get('재실행', 0)}줄은 같은 식이나 같은 계산을 다시 한 것입니다. "
           f"다만 {len(af)}개 폴드에서 VECM의 오차는 나이브보다 {'컸고' if worse else '작았고'}, "
           + "결과 파일과의 대조는 실행 결과가 아직 없어 하지 못했습니다."),
      stack=["요약"], nodes=ALL,
      tensors=[{"type": "table", "title": f"검증 전체 ({kind_txt})", "columns": ["단계", "종류", "무엇", "결과"], "rows": [[c["step"], KIND[c["kind"]], c["what"].split("] ", 1)[1], "✓ 일치" if c["ok"] else "✗ 다름"] for c in checks_all]}])
    if n_ok != len(checks_all):
        raise AssertionError(f"검증 실패: {[c for c in checks_all if not c['ok']]}")

    def clean(o):
        if isinstance(o, dict): return {k_: clean(v_) for k_, v_ in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v_) for v_ in o]
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        return o
    steps[:] = clean(steps)
    used = sorted({s_["code"]["file"] for s_ in steps if s_.get("code")})
    return {"title": "VECM 코드 따라가기 — 실제로 계산되는 순서", "model": "vecm", "steps": steps, "sources": {fn_: src.text[fn_] for fn_ in used}, "diagram": diagram(), "checks": clean(checks_all)}


def diagram() -> dict:
    labels = [("lev", "수준값 행렬", "대상 + 변수"), ("unit", "단위근", "ADF 진단"), ("joh", "Johansen", "R₀·R₁·고유값"), ("beta", "장기 균형", "β·α"), ("gamma", "단기 동학", "Γ"),
              ("pred", "재귀 예측", "ŷ = y + Δŷ"), ("allf", "모든 폴드", "안정성·기준 비교")]
    nodes = [{"id": i, "label": l, "x": 10 + 268 * k_, "y": 14, "w": 240, "h": 52, "sub": s} for k_, (i, l, s) in enumerate(labels)]
    E = [{"from": a[0], "to": b[0]} for a, b in zip(labels[:-1], labels[1:])]
    return {"width": 10 + 268 * len(labels), "height": 84, "nodes": nodes, "edges": E, "groups": []}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture"); ap.add_argument("--out")
    a = ap.parse_args()
    cap = pickle.load(open(a.capture or ROOT / "docs/guide_build/walk/vecm_capture.pkl", "rb"))
    sc = build(cap)
    out = build_html(sc, Path(a.out) if a.out else ROOT / "docs/guide_build/walk/vecm.html")
    print("저장:", out, "· 단계", len(sc["steps"]), "· 검증", sum(c["ok"] for c in sc["checks"]), "/", len(sc["checks"]), "· 크기", round(out.stat().st_size / 1e6, 2), "MB")
