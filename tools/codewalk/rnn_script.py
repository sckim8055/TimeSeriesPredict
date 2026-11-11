"""LSTM·GRU 코드 따라가기 대본 — rnn_capture 가 붙잡은 실제 실행으로 단계별 화면을 만든다.

  python -m tools.codewalk.rnn_script --model lstm [--capture docs/guide_build/walk/lstm_capture.pkl] [--out docs/guide_build/walk/lstm.html]

원칙(BIVA 와 같음): 화면 숫자는 실행 기록과 그것으로 직접 계산한 값만, 직접 계산은 '검증' 줄로 기록과 비교(다르면 멈춤), 코드 줄은 원본에서 글자로 찾음(바뀌면 멈춤).
nn.LSTM / nn.GRU 의 셀 계산은 PyTorch 내부(C++)라 줄 추적이 안 된다 → PyTorch 문서의 식을 numpy 64비트로 그대로 계산해 실제 출력과 맞는지 보인다.
say = 음성·자막 대본(숫자는 화면과 같은 변수에서), notes = 코드 줄 설명, lab = 직접 실행해 보기(실제로 실행해 엔진 값과 맞춤) — 2026-11-08.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools.codewalk.common import ROOT, Sources, arr, build_html, f, match_result, num, py_list, run_lab
from tools.codewalk.speech import TERMS, attach

EXP, CAL = "engine/experiment.py", "engine/calendar.py"
SIG = lambda a: 1 / (1 + np.exp(-a))
F32 = 2e-5   # 엔진(신경망)은 32비트 실수 → 실험 코드(64비트)와 6번째 자리가 1 정도 다를 수 있어 상대오차 2e-5 까지 같은 값으로 본다


def cell_np(kind, x, sd, B, H):
    """nn.LSTM / nn.GRU 한 층을 PyTorch 문서 식 그대로(64비트). 시점마다 게이트·상태를 돌려준다. x: [B, L, k]"""
    W_ih, W_hh = sd["weight_ih_l0"], sd["weight_hh_l0"]
    b_ih, b_hh = sd["bias_ih_l0"], sd["bias_hh_l0"]
    h = np.zeros((B, H)); c = np.zeros((B, H)); rec = []
    for t in range(x.shape[1]):
        a_i = x[:, t, :] @ W_ih.T + b_ih  # 입력 쪽
        a_h = h @ W_hh.T + b_hh  # 지난 상태 쪽
        if kind == "lstm":
            g = a_i + a_h
            i_, f_, g_, o_ = SIG(g[:, :H]), SIG(g[:, H:2 * H]), np.tanh(g[:, 2 * H:3 * H]), SIG(g[:, 3 * H:])
            c = f_ * c + i_ * g_; h = o_ * np.tanh(c)
            rec.append({"pre": g, "a_i": a_i, "a_h": a_h, "i": i_, "f": f_, "g": g_, "o": o_, "c": c.copy(), "h": h.copy()})
        else:
            r = SIG(a_i[:, :H] + a_h[:, :H]); z = SIG(a_i[:, H:2 * H] + a_h[:, H:2 * H])
            n = np.tanh(a_i[:, 2 * H:] + r * a_h[:, 2 * H:])
            h_prev = h; h = (1 - z) * n + z * h_prev
            rec.append({"a_i": a_i, "a_h": a_h, "r": r, "z": z, "n": n, "h": h.copy(), "h_prev": h_prev})
    return rec


# ---------------------------------------------------------------- 직접 실행해 보기(실험 코드)
# 실행 기록의 숫자를 그대로 넣은, 혼자 도는 파이썬 코드(math 만). 대본을 만들 때 실제로 실행해 출력이 엔진 값과 같은지 확인한다(common.run_lab).
FREQ_KO = {"D": "일간", "W": "주간", "M": "월간", "Q": "분기"}
UNIT_KO = {"D": "하루", "W": "한 주", "M": "한 달", "Q": "한 분기"}
ORD_KO = {4: "넷째", 5: "다섯째", 6: "여섯째", 7: "일곱째", 8: "여덟째", 9: "아홉째", 10: "열째"}


def data_info(cap):
    """변수 사전·원자료(data 저장소)를 읽기만 해서(학습은 하지 않음) 원래 주기·출처·발표시차 확인 여부, 대상 F11 의 원자료,
    마지막 학습 행이 쓰는 원자료 기준일을 구한다. 엔진 함수(apply_transform·align_to_calendar)는 부르지 않는다 — 검증 줄이 독립 대조가 되도록
    변환(pct·diff)과 '공개일 ≤ 그 달 1일' 고르기를 여기서 직접 한다.
    음성 대본(say)이 이 값들(원래 주기·발표시차·기준일)을 쓰므로 자료가 없으면 빌드를 멈춘다 — 빌드하는 곳에 따라 음성이 달라지면 녹음(음성 해시)과 맞지 않는다."""
    need_files = [ROOT / "data" / "variable_master.xlsx"]
    miss = [str(p_) for p_ in need_files if not p_.exists()]
    if miss:
        raise FileNotFoundError(f"대본(음성)에 필요한 자료가 없습니다: {miss} — 이 자료가 있는 곳에서 빌드하세요(음성이 빌드 환경에 따라 달라지면 안 됨)")
    if True:
        from engine.registry import Registry
        from engine.store import DataVersion
        cfg = cap["config"]
        reg, _ = Registry.load_master(ROOT / "data" / "variable_master.xlsx")
        dv = DataVersion(ROOT / "data", cfg.get("data_version"))
        t_last = cap["raw_train"].index[-1]
        tv = cfg["target_variable"]
        wide = dv.wide([tv] + list(cap["transforms"]))
        out = {"target_lag": int((cfg.get("lag_override") or {}).get(tv, reg.get(tv).lag_days)), "target": wide[tv].dropna().astype(float), "vars": {}}
        for vid, tr in cap["transforms"].items():
            lv = wide[vid].dropna().astype(float)
            if tr == "pct":
                s_ = (lv / lv.shift(1) - 1).dropna()
            elif tr == "diff":
                s_ = (lv - lv.shift(1)).dropna()
            else:
                raise ValueError(f"직접 계산하지 않는 변환 {tr}")
            lag = int(cap["lags"][vid]) if cfg.get("apply_lag", True) else 0
            pub = pd.DatetimeIndex(s_.index).normalize() + pd.to_timedelta(lag, unit="D")     # 공개일 = 기준일 + 발표시차
            ref = s_.index[np.flatnonzero(pub <= t_last)[-1]]                                 # 마지막 학습 행(그 달 1일)까지 공개된 마지막 값의 기준일
            prev = lv.index[lv.index < ref][-1]
            r_ = reg.get(vid)
            out["vars"][vid] = {"freq": str(r_.freq), "tr": tr, "lag": lag, "ref": ref, "val": float(s_.loc[ref]), "prev": prev, "src": str(r_.source),
                                "src_code": str(r_.source_code), "lag_ok": bool(r_.lag_confirmed), "lv": lv,
                                "lv_prev": float(lv.loc[prev]), "lv_ref": float(lv.loc[ref]), "col": f"{vid}_{tr}" if tr not in ("", "none") else vid}
        return out


def fpos(v, sig=3) -> str:
    """지수 표기 없이 유효숫자 sig 자리: 1.654e-05 → 0.0000165."""
    return np.format_float_positional(float(v), precision=sig, unique=False, fractional=False, trim="-")


def g4(v) -> str:
    """유효숫자 4자리(작은 수를 소수 넷째 자리로 자르면 다시 계산한 값이 맞지 않아서)."""
    return f"{float(v):.4g}"


def _pl(vals, sig=17):
    return py_list([float(v) for v in vals], sig)


def _g6(v):
    return f"{float(v):.5g}"


def _rows(M, sig=12):
    return "[\n" + "".join(f"    {_pl(r, sig)},\n" for r in M) + "]"


def lab_scale(col, var, last_date, mean_e, sd_e, z_e, about=""):
    col = list(col)
    if any(v != v for v in col):
        raise ValueError(f"{var} 에 빈 칸이 있어 이 실험 코드(빈 칸 없는 열 전제)를 쓸 수 없음 — 다른 열을 고르세요")
    code = ((f"# {var}: {about}\n" if about else "")
            + f"# {var} 의 학습 폴드 값 {len(col)}개 (월 달력의 행, 실행 기록 그대로 — 이 열은 빈 칸이 없어 앞 값 채우기가 바꾸는 것 없음)\n"
            f"x = {_pl(col)}\n\n"
            "n = len(x)\n"
            "total = 0.0\n"
            "for v in x:\n"
            "    total += v\n"
            "mean = total / n\n\n"
            "sq = 0.0\n"
            "for v in x:\n"
            "    sq += (v - mean) ** 2\n"
            "sd = (sq / n) ** 0.5               # n 으로 나눔 (StandardScaler 와 같음)\n\n"
            f"z = (x[-1] - mean) / sd           # 마지막 행({last_date})의 값 → 표준화\n\n"
            "print(\"평균\", f\"{mean:.5g}\")\nprint(\"표준편차\", f\"{sd:.5g}\")\nprint(\"z\", f\"{z:.5g}\")\n")
    exp = f"평균 {_g6(mean_e)}\n표준편차 {_g6(sd_e)}\nz {_g6(z_e)}"
    return {"title": f"{var} 의 평균·표준편차와 z 를 직접 계산",
            "desc": "학습 폴드 값만으로 평균과 표준편차(n 으로 나눔)를 구하고, 마지막 행의 값을 표준화합니다. 출력이 엔진의 StandardScaler 값(mean_·scale_)과 실제 창의 값과 같아야 합니다.",
            **run_lab(code, exp, pure=True, rtol=F32)}


def lab_t1(kind, x1, W_ih, b_ih, b_hh, H, unit, dec, h_e, c_e=None):
    """τ1 에서 은닉 단위 하나(코드 색인 unit−1). 지난 상태가 0 이라 W_hh 항이 없다."""
    u = unit - 1
    rows = lambda gi: _pl(W_ih[gi * H + u], 12)
    head = ("import math\n\n"
            "def sigmoid(a):\n    return 1 / (1 + math.exp(-a))\n\n"
            f"# 첫 번째 창(결정월 {dec})의 τ1 행 — 변수 {len(x1)}개 (표준화 값, 실행 기록 그대로)\n"
            f"x = {_pl(x1)}\n")
    if kind == "lstm":
        code = (head + f"# 은닉 단위 {unit}(코드 색인 {u})의 W_ih 네 줄(입력 문 i · 잊기 문 f · 새 정보 g · 출력 문 o)과 편향 b_ih + b_hh — 학습이 끝난 가중치, 유효숫자 12자리\n"
                f"Wi = {rows(0)}\nWf = {rows(1)}\nWg = {rows(2)}\nWo = {rows(3)}\n"
                + "bi, bf, bg, bo = " + ", ".join(f"{float(b_ih[gi * H + u] + b_hh[gi * H + u]):.12g}" for gi in range(4)) + "\n\n"
                "# τ1: 지난 은닉 상태 h₀ = 0, 셀 상태 c₀ = 0 → W_hh·h₀ 항은 0 (편향 b_hh 만 남아 위 편향에 더해 둠)\n"
                "ai, af, ag, ao = bi, bf, bg, bo\n"
                "for k in range(len(x)):\n"
                "    ai += Wi[k] * x[k]\n    af += Wf[k] * x[k]\n    ag += Wg[k] * x[k]\n    ao += Wo[k] * x[k]\n"
                "i, f, g, o = sigmoid(ai), sigmoid(af), math.tanh(ag), sigmoid(ao)\n"
                "c = f * 0.0 + i * g                # 셀 상태: 잊기 문 × 지난 c₀(= 0) + 입력 문 × 새 정보\n"
                "h = o * math.tanh(c)               # 은닉 상태: 출력 문 × tanh(셀 상태)\n\n"
                "print(\"c\", f\"{c:.5g}\")\nprint(\"h\", f\"{h:.5g}\")\n")
        exp = f"c {_g6(c_e)}\nh {_g6(h_e)}"
        desc = (f"입력 {len(x1)}개에 네 문의 가중치를 곱해 더하고 시그모이드·tanh 를 씌운 뒤, 셀 상태 c 와 은닉 상태 h 를 계산합니다. "
                "기대 출력의 h 는 실제 실행 기록(out 의 τ1)이고, c 는 기록(out)에 없어서 같은 가중치로 PyTorch nn.LSTM 을 한 시점 돌려 얻은 값입니다.")
    else:
        code = (head + f"# 은닉 단위 {unit}(코드 색인 {u})의 W_ih 세 줄(재설정 문 r · 갱신 문 z · 새 후보 n), 입력 쪽 편향 b_ih, 지난 상태 쪽 편향 b_hh — 학습이 끝난 가중치, 유효숫자 12자리\n"
                f"Wr = {rows(0)}\nWz = {rows(1)}\nWn = {rows(2)}\n"
                + "b_ir, b_iz, b_in = " + ", ".join(f"{float(b_ih[gi * H + u]):.12g}" for gi in range(3)) + "\n"
                + "b_hr, b_hz, b_hn = " + ", ".join(f"{float(b_hh[gi * H + u]):.12g}" for gi in range(3)) + "\n\n"
                "# τ1: 지난 은닉 상태 h₀ = 0 → 지난 상태 쪽 W_hh·h₀ + b_hh 는 편향 b_hh 만\n"
                "ar, az, an = b_ir, b_iz, b_in      # 입력 쪽: W_ih·x + b_ih\n"
                "for k in range(len(x)):\n"
                "    ar += Wr[k] * x[k]\n    az += Wz[k] * x[k]\n    an += Wn[k] * x[k]\n"
                "r = sigmoid(ar + b_hr)             # 재설정 문: 두 쪽을 더해 시그모이드\n"
                "z = sigmoid(az + b_hz)             # 갱신 문\n"
                "n = math.tanh(an + r * b_hn)       # 새 후보: 지난 상태 쪽 덩어리(τ1 에서는 b_hn)에 r 을 먼저 곱함\n"
                "h = (1 - z) * n + z * 0.0          # 새 은닉 상태 = (1 − z)·n + z·지난 h₀(= 0)\n\n"
                "print(\"h\", f\"{h:.5g}\")\n")
        exp = f"h {_g6(h_e)}"
        desc = (f"입력 {len(x1)}개에 세 덩어리의 가중치를 곱해 더한 뒤 r·z 는 시그모이드, n 은 r 을 지난 상태 쪽 편향에 곱한 뒤 tanh 를 씌웁니다. "
                "기대 출력 h 는 실제 실행 기록(out 의 τ1)입니다.")
    return {"title": f"τ1 의 은닉 단위 {unit} 을 직접 계산", "desc": desc, **run_lab(code, exp, pure=True, rtol=F32)}


def lab_linear(h12, d, w, b, dec, keep, ntr_e, train_e, eval_e):
    code = (f"# 첫 번째 창(결정월 {dec}): 마지막 시점의 은닉 상태 h ({len(h12)}개)와 드롭아웃 뒤 값 d — 실행 기록 그대로\n"
            f"h = {_pl(h12)}\n"
            f"d = {_pl(d)}\n"
            f"# Linear({len(w)}→1) 의 가중치 w 와 편향 b_fc — 학습이 끝난 가중치\n"
            f"w = {_pl(w)}\nb_fc = {float(b)!r}\n\n"
            "erased = 0\n"
            "for du in d:\n"
            "    if du == 0:\n"
            "        erased += 1                # 드롭아웃이 지운 칸\n\n"
            "train = b_fc\n"
            "for wu, du in zip(w, d):\n"
            "    train += wu * du               # 학습 모드: 드롭아웃 뒤 값을 씀\n"
            "pred = b_fc\n"
            "for wu, hu in zip(w, h):\n"
            "    pred += wu * hu                # 예측 모드: 드롭아웃 없음\n\n"
            "print(\"지워진 칸\", erased)\nprint(\"학습 모드\", f\"{train:.5g}\")\nprint(\"예측 모드\", f\"{pred:.5g}\")\n")
    exp = f"지워진 칸 {ntr_e}\n학습 모드 {_g6(train_e)}\n예측 모드 {_g6(eval_e)}"
    return {"title": "Linear 출력을 직접 계산 (학습 모드 · 예측 모드)",
            "desc": f"은닉 값에 가중치를 곱해 더하고 편향 b_fc 를 더합니다. 학습 모드는 드롭아웃 뒤 값(지운 칸 0, 남은 칸 × 1 ÷ {keep})을, 예측 모드는 은닉 값을 그대로 씁니다. 두 출력이 기록한 net 의 출력과 같아야 합니다.",
            **run_lab(code, exp, pure=True, rtol=F32)}


def lab_adam(g_pre, total, before, lr, wd, eps, betas, coef_e, after_e):
    code = ("import math\n\n"
            "# 이 폴드의 맨 첫 갱신(첫 epoch 의 첫 배치, 처음 가중치): fc.bias 의 기울기 ∇θ(자르기 전), 모든 기울기를 합친 크기 ‖∇‖, 갱신 전 값 θ — 실행 기록 그대로\n"
            f"grad = {float(g_pre)!r}\ntotal_norm = {float(total)!r}\ntheta = {float(before)!r}\n"
            f"lr, wd, eps = {float(lr)!r}, {float(wd)!r}, {float(eps)!r}\nbeta1, beta2 = {float(betas[0])!r}, {float(betas[1])!r}\n\n"
            "coef = min(1.0, 1.0 / (total_norm + 1e-6))   # clip_grad_norm_(…, 1.0): 전체 크기가 1 을 넘으면 같은 비율로 줄임\n"
            "grad = grad * coef\n"
            "grad = grad + wd * theta           # weight_decay: 기울기에 wd × 가중치를 더함 (L2 방식)\n"
            "m = (1 - beta1) * grad             # 기울기 평균 (처음이라 이전 값 0)\n"
            "v = (1 - beta2) * grad ** 2        # 기울기 제곱 평균\n"
            "m_hat = m / (1 - beta1)            # 편향 보정 → grad\n"
            "v_hat = v / (1 - beta2)            # 편향 보정 → grad²\n"
            "theta = theta - lr * m_hat / (math.sqrt(v_hat) + eps)\n\n"
            "print(\"자른 배율\", f\"{coef:.5g}\")\nprint(\"갱신 뒤 fc.bias\", f\"{theta:.5g}\")\n")
    exp = f"자른 배율 {_g6(coef_e)}\n갱신 뒤 fc.bias {_g6(after_e)}"
    return {"title": "기울기 잘라내기와 Adam 의 첫 걸음을 직접 계산",
            "desc": "실제 학습의 맨 첫 갱신(처음 가중치)입니다. 기울기를 전체 크기로 자르고, wd × 가중치를 더한 뒤 Adam 의 첫 걸음을 계산합니다. 첫 걸음에서는 편향 보정 뒤 m̂ = ∇θ, v̂ = ∇θ² 라서 거의 학습률만큼 움직입니다. 기대 출력은 실제로 자른 기울기의 배율과 갱신 뒤 fc.bias 입니다.",
            **run_lab(code, exp, pure=True, rtol=F32)}


def lab_forward(kind, win, sd, w, b, mu, sdv, level, dec, z_e, pred_e, next_e):
    W_ih, W_hh, b_ih, b_hh = sd["weight_ih_l0"], sd["weight_hh_l0"], sd["bias_ih_l0"], sd["bias_hh_l0"]
    H = W_hh.shape[1]; G = W_ih.shape[0] // H; KN = kind.upper()
    order = "i·f·g·o (입력 문·잊기 문·새 정보·출력 문)" if kind == "lstm" else "r·z·n (재설정 문·갱신 문·새 후보)"
    code = ("import math\n\n"
            "def sigmoid(a):\n    return 1 / (1 + math.exp(-a))\n\n"
            f"# 결정월 {dec} 의 테스트 창: 시점 {len(win)}개 × 변수 {len(win[0])}개 (표준화 값, 실행 기록 그대로)\n"
            f"X = {_rows(win, 17)}\n"
            f"# nn.{KN} 가중치 (학습이 끝난 값, 유효숫자 12자리): W_ih [{G * H}][{len(win[0])}], W_hh [{G * H}][{H}], 편향 [{G * H}] — 줄 순서 {order}, {H}줄씩\n"
            f"W_ih = {_rows(W_ih)}\nW_hh = {_rows(W_hh)}\nb_ih = {_pl(b_ih, 12)}\nb_hh = {_pl(b_hh, 12)}\n"
            f"# Linear({H}→1) 의 w, b_fc 와 학습 창 정답의 평균 μ_y·표준편차 σ_y, 결정월의 F11\n"
            f"w = {_pl(w, 12)}\nb_fc = {float(b):.12g}\nmu_y = {float(mu)!r}\nsigma_y = {float(sdv)!r}\nf11 = {float(level)!r}\n\n"
            f"H = {H}\n"
            "h = [0.0] * H                          # 지난 은닉 상태 (0 에서 시작)\n"
            + ("c = [0.0] * H                          # 지난 셀 상태\n" if kind == "lstm" else "")
            + "for x in X:                            # τ1 → τ" + str(len(win)) + " 차례로\n"
            "    ai, ah = [], []\n"
            f"    for j in range({G} * H):\n"
            "        s = b_ih[j]\n"
            "        for k in range(len(x)):\n"
            "            s += W_ih[j][k] * x[k]       # 입력 쪽 W_ih·x + b_ih\n"
            "        ai.append(s)\n"
            "        s = b_hh[j]\n"
            "        for k in range(H):\n"
            "            s += W_hh[j][k] * h[k]       # 지난 상태 쪽 W_hh·h + b_hh\n"
            "        ah.append(s)\n"
            "    new_h = []\n"
            "    for u in range(H):\n"
            + ("        i = sigmoid(ai[u] + ah[u])\n"
               "        f = sigmoid(ai[H + u] + ah[H + u])\n"
               "        g = math.tanh(ai[2 * H + u] + ah[2 * H + u])\n"
               "        o = sigmoid(ai[3 * H + u] + ah[3 * H + u])\n"
               "        c[u] = f * c[u] + i * g           # 셀 상태\n"
               "        new_h.append(o * math.tanh(c[u])) # 은닉 상태\n"
               if kind == "lstm" else
               "        r = sigmoid(ai[u] + ah[u])                       # 재설정 문\n"
               "        z = sigmoid(ai[H + u] + ah[H + u])               # 갱신 문\n"
               "        n = math.tanh(ai[2 * H + u] + r * ah[2 * H + u]) # 새 후보: 지난 상태 쪽에 r 을 먼저 곱함\n"
               "        new_h.append((1 - z) * n + z * h[u])              # 새 은닉 상태\n")
            + "    h = new_h\n\n"
            "out = b_fc\n"
            "for wu, hu in zip(w, h):\n"
            "    out += wu * hu                     # Linear — 예측 모드라 드롭아웃 없음\n"
            "y_hat = out * sigma_y + mu_y           # 표준화 되돌리기 → 다음 달 변화율 예측\n"
            "next_f11 = f11 * (1 + y_hat)\n\n"
            "print(\"신경망 출력\", f\"{out:.5g}\")\nprint(\"예측 변화율\", f\"{y_hat:.5g}\")\nprint(\"다음 달 F11\", f\"{next_f11:.6g}\")\n")
    exp = f"신경망 출력 {_g6(z_e)}\n예측 변화율 {_g6(pred_e)}\n다음 달 F11 {float(next_e):.6g}"
    return {"title": f"테스트 창 하나를 처음부터 끝까지 직접 계산 ({KN} {len(win)}시점 → Linear → 되돌리기)",
            "desc": f"결정월 {dec} 의 테스트 창을 {KN} 식으로 {len(win)}개 시점 차례로 읽고, 마지막 은닉 상태를 Linear 에 넣은 뒤 σ_y 를 곱하고 μ_y 를 더합니다. "
                    "출력이 엔진(_walkforward_predict)이 낸 예측과 같아야 합니다. 반복이 많아 Python Tutor 대신 복사해서 실행하세요.",
            **run_lab(code, exp, pure=True, rtol=F32, tutor=False)}


def build(cap: dict) -> dict:
    kind = cap["model"]; KN = kind.upper()
    J = lambda j: attach(TERMS[KN], j)[len(TERMS[KN]):]    # KN 뒤 조사(읽는 소리 기준): LSTM 은 / GRU 는
    MODEL = f"models/{kind}.py"
    src = Sources([MODEL, EXP, CAL])
    MM = importlib.import_module(f"models.{kind}")
    p = cap["params"]; feats = cap["feats"]; names = cap["feat_names"]
    K, L, Hn, bs = len(feats), cap["seq_len"], int(p["hidden"]), cap["batch_size"]
    colnames = [f"{c} {names.get(c, '')}" for c in feats]
    sdt = cap["net_state"]
    sd = {k_.split(".", 1)[1]: v.double().numpy() for k_, v in sdt.items() if k_.startswith(f"{kind}.")}
    W_fc, b_fc = sdt["fc.weight"].double().numpy()[0], float(sdt["fc.bias"].double().numpy()[0])
    raw = cap["raw_train"]; dates = [d.strftime("%Y-%m") for d in raw.index]
    Xw, yw = cap["Xw"], cap["yw"]; win_idx = cap["win_idx"]
    n_tr, n_va = cap["n_tr"], cap["n_va"]
    xb = cap["batch"]["x"]; bidx = cap["batch_idx"]; B = len(bidx)
    gates = ["i", "f", "g", "o"] if kind == "lstm" else ["r", "z", "n"]
    G = len(gates)
    gname = {"i": "입력 문 i", "f": "잊기 문 f", "g": "새 정보 g", "o": "출력 문 o", "r": "재설정 문 r", "z": "갱신 문 z", "n": "새 후보 n"}
    steps: list[dict] = []
    checks_all: list[dict] = []
    tau = [f"τ{t + 1}" for t in range(L)]
    units = [f"u{u + 1}" for u in range(Hn)]
    train_len = cap["config"]["validation"]["train_len"]; emb = cap["embargo"]
    var_names = cap.get("var_names", {})
    dec0 = win_idx[bidx[0]][:7]          # 따라갈 배치의 첫 번째 창(b=0)의 결정월
    DI = data_info(cap)                  # 변수 사전·원자료(읽기만) — 없으면 빌드가 멈춤(음성이 환경에 따라 달라지지 않게)
    pn = lambda x: f"({x})" if str(x).startswith("-") else str(x)      # 식 속 음수는 괄호로

    def var_of(c):
        return next((vid for vid in var_names if c == vid or c.startswith(vid + "_")), None)

    def kind_word(vid, short=False):
        """변환 뜻: 일간 pct → '하루 변화율', 월간 diff → '한 달 차이'."""
        tr_ = cap["transforms"].get(vid)
        w_ = "변화율" if tr_ == "pct" else "차이" if tr_ == "diff" else tr_
        fq = DI["vars"][vid]["freq"]
        return (f"{UNIT_KO[fq]} " if fq in UNIT_KO else "") + w_

    def nm(c):
        """음성용 변수 이름: 'USDKRW_pct' → '원 달러 환율 하루 변화율'."""
        vid = var_of(c)
        return f"{var_names[vid].replace('/', ' ')} {kind_word(vid)}" if vid else c

    def H(title, data, rows=None, cols=None, shape=None, note=None, scale="div", hl=None, digits=None, numbers=True, **kw):
        t = {"type": "heat", "title": title, "data": data if isinstance(data, list) else arr(data), "rows": rows, "cols": cols, "shape": shape, "note": note,
             "scale": scale, "hl": hl or {}, "numbers": numbers, **kw}
        if digits is not None:
            t["digits"] = digits
        if cols == feats:
            t["colNames"] = colnames
        return t

    KIND = {"ind": "독립 대조", "eq": "식 확인(같은 배열·같은 식)", "rerun": "재실행"}

    def chk(what, a, b, tol=1e-5, kind="ind"):
        """화면의 '검증' 줄. kind: ind = 다른 경로·다른 계산법으로 맞춰 본 독립 대조, eq = 같은 배열에 같은 식을 다시 계산한 식 확인,
        rerun = 같은 코드를 다시 돌린 값(결과 파일 포함)과의 비교."""
        d = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
        c = {"what": ("" if kind == "ind" else f"{KIND[kind]}: ") + f"{what} (차이 {d:.1e})", "ok": bool(d <= tol), "kind": kind}
        checks_all.append({**c, "base": f"{what} (차이 {d:.1e})", "step": len(steps) + 1})
        return c

    def S(chapter, narr, *, code=None, stack=None, nodes=(), edges=(), tensors=(), formula=None, hint=None, checks=(), say=None, notes=None, lab=None):
        """say = 음성·자막 대본(화면 숫자 그대로 — 숫자는 narr 와 같은 변수에서), notes = {원본 줄 번호: 그 줄 설명}, lab = 직접 실행해 보기(실행해 엔진 값과 맞춘 것)."""
        checks = list(checks)
        if lab is not None:
            c_ = {"what": f"직접 실행해 보기 '{lab['title']}' 출력 = 엔진 값", "ok": bool(lab["ok"]), "kind": "ind"}
            checks_all.append({**c_, "base": c_["what"], "step": len(steps) + 1}); checks.append(c_)
        if code is not None and notes:
            code = {**code, "notes": {str(k_): v_ for k_, v_ in notes.items()}}
        steps.append({"chapter": chapter, "narr": narr, "say": say, "code": code, "stack": list(stack or []), "nodes": list(nodes), "edges": [list(e) for e in edges],
                      "tensors": list(tensors), "formula": formula, "hint": hint, "checks": checks, "lab": lab})

    def C(fn, lines, focus=None, func="", before=8, after=10):
        lines = sorted(set(lines))
        return {"file": fn, "lines": lines, "focus": focus or lines[0], "func": func, "before": before, "after": after}

    stk_fit = ["engine/experiment.py · _walkforward_predict", "engine/experiment.py · _split_fit", f"{MODEL} · Model.fit"]
    stk_fwd = stk_fit + [f"{MODEL} · _Net.forward"]
    cell_nodes = [f"c{t + 1}" for t in range(L)]
    L_RNN = src.line(MODEL, f"self.{kind} = nn.{KN}(k, hidden, layers, batch_first=True")
    L_OUT = src.line(MODEL, f"out, _ = self.{kind}(x)")
    L_RET = src.line(MODEL, "return self.fc(self.drop(out[:, -1, :])).squeeze(-1)")
    L_LOSS = src.line(MODEL, "loss = loss_fn(self.net(Xt[idx]), yt[idx])")
    # 셀 계산(64비트 문서 식)과 기록한 out 의 차이 — 게이트·셀 상태는 기록이 아니라 이 계산으로 보여 준다
    xbn = xb.double().numpy()
    rec = cell_np(kind, xbn, sd, B, Hn)
    out_rec = next(e for e in cap["events"] if "out" in e.produced).produced["out"].double().numpy()   # 기록한 out [B, L, H]
    d_cell = float(max(np.max(np.abs(rec[t]["h"] - out_rec[:, t, :])) for t in range(L)))
    gate_txt = "게이트(i·f·g·o)·셀 상태 c" if kind == "lstm" else "게이트(r·z)·새 후보 n"
    # 결과 파일(같은 설정 이름) — 모델이 같은 파일이 있으면 대조한다
    mr = match_result("forecast", cap["config"]["name"], {cap["predict"]["date"]: cap["predict"]["pred"]})

    if not mr:
        raise FileNotFoundError(f"results/forecast/ 에 설정 이름 {cap['config']['name']} 의 결과 파일이 없습니다 — 「11. 예측」의 음성이 결과 파일 대조에 따라 정해지므로 결과 폴더가 있는 곳에서 빌드하세요")

    def res_json(fn):
        jf = ROOT / "results" / "forecast" / fn.replace("_daily.parquet", ".json")
        if not jf.exists():
            raise FileNotFoundError(f"결과 json 이 없습니다: {jf} — 「11. 예측」의 음성(결과 파일 설명)이 이 파일을 씁니다")
        return json.loads(jf.read_text(encoding="utf-8"))
    has_result = (res_json(mr["file"]).get("model") or {}).get("file") == kind

    # ---------------------------------------------------------------- 0. 시작
    cfgf = cap["config_file"]
    same_cfg = cap["config"]["model"] == kind and Path(cfgf).stem.split("_")[1] == kind
    n_cell = lambda g_: g_ * Hn * (K + Hn) + 2 * g_ * Hn + Hn + 1      # 셀 가중치 + 편향 2벌 + Linear
    dflt = lambda k_: MM.PARAMS[k_].get("default_by_freq", {}).get("M", MM.PARAMS[k_]["default"])   # 모델 파일의 월간 기본값
    _m = re.search(r"torch\.manual_seed\((\d+)\)\s*\n\s*net\.train\(\)", (ROOT / "tools/codewalk/rnn_capture.py").read_text(encoding="utf-8"))
    if not _m:
        raise AssertionError("rnn_capture.py 에서 다시 넣기의 드롭아웃 시드(torch.manual_seed … net.train())를 찾지 못함 — 힌트를 고치세요")
    seed_fwd = _m.group(1)                                              # 다시 넣기의 드롭아웃 난수 시드(rnn_capture.py)
    S("0. 시작",
      f"이 영상은 <b>{KN} 모델 코드가 실제로 실행되는 순서</b>를 따라갑니다. 설정은 <b>{Path(cfgf).stem}</b>"
      + ("" if same_cfg else f" 에서 <b>모델만 {KN}</b> 로 바꾼 것(데이터·검증 설정은 LSTM 영상과 같음)")
      + f" 입니다. 대상은 KB 주택매매가격지수(F11)의 다음 달 변화율, 입력은 변수 {K}개 × 최근 {L}개월 창입니다. "
      f"화면의 숫자는 엔진의 실제 경로로 워크포워드 마지막 폴드를 학습·예측한 <b>기록</b>과, 그 기록(가중치·입력)으로 이 스크립트가 <b>직접 계산</b>한 값입니다(어느 것이 어느 쪽인지는 힌트)."
      + ("" if same_cfg else f" 이 영상의 숫자는 <b>시드 하나·폴드 하나</b>의 값이라 LSTM 과의 우열 근거가 아닙니다."),
      say=(f"이 영상은 {KN} 모델 코드가 실제로 실행되는 순서를 따라갑니다. "
           + ("" if same_cfg else f"설정은 LSTM 영상과 같고 모델만 {KN}로 바꿨습니다. 이 영상의 숫자는 시드 하나, 폴드 하나의 값이라 LSTM과의 우열 근거가 아닙니다. ")
           + f"대상은 KB 주택매매가격지수의 다음 달 변화율이고, 입력은 변수 {K}개, 최근 {L}개월 창입니다. "
           "화면의 숫자는 엔진의 실제 경로로 워크포워드 마지막 폴드를 학습하고 예측한 기록과, 그 가중치와 입력으로 직접 계산한 값입니다. "
           "오른쪽 코드에서 노란 줄이 지금 보는 부분이고, 줄 아래 회색 글은 그 줄의 설명입니다."),
      notes=({4: f"이 설정의 폴드는 검증 창이 없어 조기종료 없이 마지막 epoch 의 가중치를 씁니다(「4. 검증 나누기」)"} if kind == "lstm" else
             {1: f"이 실행은 GRU({K}→{Hn}) — 입력 변수 {K}개, 은닉 {Hn}. 이 설정에서 가중치 수 GRU {cap['n_params']:,}개 · 같은 크기의 LSTM {n_cell(4):,}개 → {1 - cap['n_params'] / n_cell(4):.1%} 적음. 게이트 2개 = 재설정 문 r · 갱신 문 z (새 후보 n 은 게이트가 아님)",
              4: "이 설정의 폴드는 검증 창이 없어 조기종료 없이 마지막 epoch 의 가중치를 씁니다(「4. 검증 나누기」)"}),
      code=C(MODEL, src.span(MODEL, '"""', '"""'), 1, "모듈 설명", 0, 12), stack=[MODEL],
      nodes=["data", "scale", "win", "split", "batch", "x", *cell_nodes, "last", "drop", "fc", "loss", "clip", "adam", "epoch", "pred"],
      tensors=[{"type": "table", "title": f"이 영상의 실행 설정 ({cfgf} + {MODEL} 기본값)", "columns": ["항목", "값", "뜻"], "rows": [
          ["대상", "F11 다음 달 변화율", "y_t = F11_(t+1) ÷ F11_t − 1 (k = 1)"],
          ["입력 변수", f"{K}개", f"대상의 기술지표 {sum(c.startswith('A_') for c in feats)}개(A_…) + 거시 변수 {sum(not c.startswith('A_') for c in feats)}개"],
          ["창 길이 seq_len", L, f"최근 {L}개월"], ["은닉 크기 hidden", Hn, "월간 기본" if Hn == dflt("hidden") else f"설정값(월간 기본 {dflt('hidden')})"], ["층 layers", p["layers"], ""],
          ["dropout", p["dropout"], "마지막 은닉 → Linear 앞"],
          ["epochs", p["epochs"], ("정해진 횟수 — 검증이 없어 조기종료 없음" if n_va == 0 else "상한(조기종료 있음)") + (" · 월간 기본" if p["epochs"] == dflt("epochs") else f" · 설정값(월간 기본 {dflt('epochs')})")], ["학습률 lr", p["lr"], "Adam"],
          ["batch_size", bs, "월간 기본" if bs == dflt("batch_size") else f"설정값(월간 기본 {dflt('batch_size')})"], ["patience", p["patience"], "검증이 있을 때만 쓰임"], ["weight_decay", p["weight_decay"], "Adam 의 L2"],
          ["워크포워드", f"학습 구간 {train_len}개월(끝 {emb}개월 비움 → {len(dates)}행) · 테스트 {cap['config']['validation']['test_len']}개월 · 폴드 {cap['fold']['n_folds']}개", f"이 영상은 마지막 폴드(학습 {cap['fold']['train'][0][:7]} ~ {cap['fold']['train'][1][:7]}, 테스트 {cap['fold']['test'][0][:7]})"],
          ["전체 가중치 수", f"{cap['n_params']:,}", "parameters() 합"]]}],
      checks=[chk(f"전체 가중치 수 = {G}·H·(K+H) + 2·{G}·H + H + 1 (셀 가중치 · 편향 2벌 · Linear)", cap["n_params"], n_cell(G), 0)],
      hint="숫자의 출처: tools/codewalk/rnn_capture.py 가 엔진의 실제 경로(_walkforward_predict → models/" + kind + ".py)로 마지막 폴드를 학습·예측하며 기록했습니다(계산에는 손대지 않음).\n"
           "· 학습 중 기록: 배치 순서(torch.randperm), 맨 첫 갱신(첫 epoch 의 첫 배치, 처음 가중치)의 기울기·잘라내기·Adam 걸음, 갱신마다의 기울기 크기, epoch 별 학습 손실.\n"
           f"· 학습 뒤 기록: 끝난 가중치, 마지막 epoch 의 첫 배치를 끝난 가중치로 다시 넣은 입력·마지막 은닉 상태(out)·드롭아웃 전후·출력·손실(sys.settrace·hook, 드롭아웃 난수 시드 {seed_fwd}), 테스트 창의 예측.\n"
           f"· 직접 계산(기록이 아님): {gate_txt}·그 평균, 학습 창 R² 는 이 스크립트(rnn_script.py)가 같은 가중치·입력으로 계산한 값입니다. "
           f"{KN} 셀 내부는 PyTorch(C++)라 줄 단위로 따라갈 수 없어서, PyTorch 문서의 식을 64비트로 계산했습니다 — 그 h 는 기록한 out 과 모든 시점(τ1 ~ τ{L})·배치에서 최대 {d_cell:.1e} 차이로 같습니다(「8. 시점별 계산」 τ3 ~ τ{L} 단계의 '검증' 줄).")
    S("0. 시작",
      f"호출 순서입니다. 엔진이 학습 폴드로 눈금을 맞추고 창을 만들어 <b>{MODEL} 의 Model.fit</b> 에 넘깁니다. fit 은 epoch 마다 창을 섞어 {bs}개씩 "
      f"<b>_Net.forward</b> 에 넣습니다. forward 는 ① {KN} {J('이')} {L}개 시점을 차례로 읽고 ② 마지막 시점의 은닉 상태만 꺼내 ③ 드롭아웃 ④ Linear({Hn}→1) 로 값 하나를 냅니다. "
      "fit 은 MSE 를 역전파하고, 모든 기울기를 합친 크기가 1 을 넘으면 1 로 줄인 뒤 Adam 으로 고칩니다. 끝나면 predict 가 테스트 창을 예측해 원래 단위로 되돌립니다.",
      say=("호출 순서입니다. 엔진이 학습 폴드로 눈금을 맞추고 창을 만들어 모델의 학습 함수에 넘깁니다. "
           f"학습 함수는 에포크마다 창을 섞어 {bs}개씩 순전파에 넣습니다. 순전파는 네 단계입니다. 첫째, {KN}{J('이')} {L}개 시점을 차례로 읽습니다. "
           "둘째, 마지막 시점의 은닉 상태만 꺼냅니다. 셋째, 드롭아웃을 거칩니다. 넷째, 선형 층이 값 하나를 냅니다. "
           "학습 함수는 평균제곱오차를 역전파하고, 기울기 전체 크기가 1보다 크면 크기가 1이 되게 줄인 뒤 아담으로 가중치를 고칩니다. "
           "끝나면 예측 함수가 테스트 창을 예측해 원래 단위로 되돌립니다."),
      notes={L_RNN: f"nn.{KN}({K}→{Hn}, {p['layers']}층). 층 사이 dropout 은 층이 2개 이상일 때만 — 이 설정은 {p['layers']}층이라 0",
             L_OUT: f"out = 시점마다의 은닉 상태 h [B, {L}, {Hn}] (두 번째 반환값은 쓰지 않음)",
             L_RET: "마지막 시점 → Dropout → Linear → 창마다 숫자 하나 [B]"},
      code=C(MODEL, src.span(MODEL, "class _Net(nn.Module):", "return self.fc(self.drop(out[:, -1, :])).squeeze(-1)"), src.line(MODEL, "def forward(self, x):"), "_Net", 2, 2),
      stack=stk_fwd, nodes=["batch", "x", *cell_nodes, "last", "drop", "fc", "loss", "clip", "adam", "epoch"],
      tensors=[{"type": "table", "title": "호출 순서 (파일 · 함수 · 줄)", "columns": ["순서", "파일 · 함수", "하는 일"], "rows": [
          [1, f"engine/experiment.py · _walkforward_predict (줄 {src.line(EXP, 'StandardScaler()).fit(full[')}~{src.line(EXP, 'Xw, yw = cal.make_windows(Xf')})", f"학습 폴드 → 눈금 → 창 [N, {L}, {K}]"],
          [2, f"engine/experiment.py · _split_fit (줄 {src.line(EXP, 'def _split_fit')})", "검증 나누기(이 폴드는 검증 없음)"],
          [3, f"{MODEL} · Model.fit (줄 {src.line(MODEL, 'def fit(self, X, y')})", "정답 표준화 · epoch · 배치"],
          [4, f"{MODEL} · _Net.forward (줄 {src.line(MODEL, 'def forward(self, x):')})", f"{KN} → 마지막 시점 → Dropout → Linear"],
          [5, f"{MODEL} · fit (줄 {L_LOSS}~{src.line(MODEL, 'opt.step()')})", "MSE → backward → 잘라내기 → Adam"],
          [6, f"{MODEL} · predict (줄 {src.line(MODEL, 'def predict(self, X):')})", "테스트 창 → ŷ (원래 단위)"]]}])

    # ---------------------------------------------------------------- 1. 데이터
    rv = raw.values.astype(float)
    tech = [c for c in feats if c.startswith("A_")]
    every = [d if i % 6 == 0 else "" for i, d in enumerate(dates)]
    nnan = int(np.isnan(rv).sum())
    L_TR = src.line(CAL, "tr = cal[i - train_len : i - emb] if emb else cal[i - train_len : i]")
    lev = cap["target_level"]                                   # 엔진 데이터셋의 F11(ds.prices["close"])
    f11r = DI["target"]                                         # 원자료(data 저장소)에서 따로 읽은 F11 — 대조용
    kd_raw = "ind"
    pr = cap["predict"]
    t_test = pd.Timestamp(pr["date"]); i_test = list(lev.index).index(t_test)
    emb_ts = list(lev.index[i_test - emb:i_test]); emb_ms = [d.strftime("%Y-%m") for d in emb_ts]; test_m = t_test.strftime("%Y-%m")
    # 비운 달의 정답(원자료 F11 로 계산)과 테스트 창 마지막 행의 A_ret_1(엔진 ds.X, 원래 단위) — 같은 값이면 비움은 누출 방지에 꼭 필요한 것이 아님
    emb_same = None
    if emb == 1 and cap["k"] == 1 and "A_ret_1" in feats:
        ja = feats.index("A_ret_1")
        y_emb = float(f11r.loc[t_test] / f11r.loc[emb_ts[0]] - 1)
        a1_test = float(pr["window"][-1, ja] * cap["scaler_scale"][ja] + cap["scaler_mean"][ja])
        emb_same = abs(y_emb - a1_test) <= 1e-15
    emb_txt = (f"그 달의 정답 y({emb_ms[0]}) = F11({test_m}) ÷ F11({emb_ms[0]}) − 1 = {y_emb:.8f} 은 테스트 결정월 {test_m} 행의 입력 A_ret_1(한 달 수익률)과 같은 값(아래 '검증')이라, "
               f"예측 시점(F11({test_m}) 공개 뒤)에 이미 알려진 값입니다. 그래서 이 비움은 누출을 막는 데 꼭 필요한 것이 아니라, 가장 최근 표본 하나를 버리는 <b>보수적인 규칙</b>입니다(engine/experiment.py _embargo: 예측 모드는 k 개월)."
               if emb_same else "그 달의 정답(다음 달 변화율)이 테스트 달의 값을 쓰기 때문입니다(engine/experiment.py _embargo).")
    emb_say = ("예측 모드에서 예측하는 기간만큼 비우는 엔진의 규칙입니다. 그런데 그 달의 정답은 테스트 결정월의 입력인 한 달 수익률과 똑같은 값이라, 예측할 때 이미 알려져 있습니다. "
               "그래서 여기서 비움은 누출을 막는 데 꼭 필요한 것이 아니라, 가장 최근 표본 하나를 버리는 보수적인 규칙입니다. " if emb_same else
               "그 달의 정답이 테스트 달의 값을 쓰기 때문입니다. ")
    macro = list(cap["transforms"])
    lag_txt = lambda v: f"{cap['lags'][v]}" + ("" if DI["vars"][v]["lag_ok"] else "(미확인)")
    tbl_cols = ["변수", "이름", "출처", "원래 주기", "변환", "발표시차(일)", f"{dates[-1]} 행의 원자료 기준일"]
    tbl_rows = [[v, var_names.get(v, v)[:22], f"{DI['vars'][v]['src']} {DI['vars'][v]['src_code']}", FREQ_KO.get(DI["vars"][v]["freq"], DI["vars"][v]["freq"]),
                 f"{cap['transforms'][v]} · {kind_word(v)}", lag_txt(v), DI["vars"][v]["ref"].strftime("%Y-%m-%d")] for v in macro]
    daily = [v for v in macro if DI["vars"][v]["freq"] == "D"]
    unconf = [v for v in macro if not DI["vars"][v]["lag_ok"]]
    j_unconf = res_json(mr["file"]).get("lag_unconfirmed")      # 결과 json 에 남은 '발표시차 미확인' 목록(대조용)
    j_model = (res_json(mr["file"]).get("model") or {}).get("file") or "?"; j_name = mr["file"].replace("_daily.parquet", ".json")
    if j_unconf is None:   # 정리 화면의 음성이 검증 줄 수를 읽으므로, 이 줄도 건너뛰지 않고 빌드를 멈춘다
        raise KeyError(f"{mr['file'].replace('_daily.parquet', '.json')} 에 lag_unconfirmed 가 없습니다 — 「1. 데이터 표」 검증에 필요")
    # 야후(YF) 일간 변수가 쓰는 날 가운데 국내 자료(ECOS 일간)에는 값이 없는 날 — 한국 휴장일에도 야후에는 값이 있음
    kr_d = next((v for v in daily if DI["vars"][v]["src"] == "ECOS"), None)
    hol = {}
    for v in daily:
        if DI["vars"][v]["src"] == "YF" and kr_d:
            hol[v] = [d for d in (DI["vars"][v]["prev"], DI["vars"][v]["ref"]) if d not in DI["vars"][kr_d]["lv"].index]
    hol_txt = "".join(f" {v}({DI['vars'][v]['src']}) 가 쓰는 {'·'.join(f'{d:%Y-%m-%d}' for d in ds_)} 는 국내 자료 {kr_d}({DI['vars'][kr_d]['src']})에는 값이 없는 날입니다 — 야후 자료는 한국 휴장일에도 값이 있습니다."
                      for v, ds_ in hol.items() if ds_)
    day_desc = "그 달 1일(포함)까지 자료에 값이 있는 마지막 날의, 바로 앞 관측일 대비 변화"
    S("1. 데이터 표",
      f"출발점은 엔진이 만든 <b>학습 폴드의 표</b>입니다. 행은 월({dates[0]} ~ {dates[-1]}, {len(dates)}개월)이고 열은 변수 {K}개입니다. 워크포워드 학습 구간 {train_len}개월 중 "
      f"테스트 바로 앞 {emb}개월({', '.join(emb_ms)})은 <b>비움(embargo)</b>으로 뺍니다. " + emb_txt + " "
      f"앞의 {len(tech)}개(A_…)는 대상 F11 의 그 달까지의 값으로 만든 기술지표(수익률·이동평균 비율·변동성·RSI·MACD), 뒤의 {K - len(tech)}개는 그 달 1일까지 공개된 거시 변수입니다"
      + (f"({', '.join(daily)} 는 원래 주기가 일간이라 {day_desc})" if daily else "") + f". 이 폴드에는 빈 칸이 {nnan}개입니다."
      + (f" 표의 발표시차 '(미확인)'은 변수 사전에서 확인되지 않은 값입니다({', '.join(unconf)})." if unconf else ""),
      say=(f"출발점은 엔진이 만든 학습 폴드의 표입니다. 행은 {dates[0]}부터 {dates[-1]}까지 {len(dates)}개월이고, 열은 변수 {K}개입니다. "
           f"워크포워드 학습 구간 {train_len}개월 가운데 테스트 바로 앞 " + ("한 달" if emb == 1 else f"{emb}개월") + f", {', '.join(emb_ms)}은 비워 둡니다. "
           + emb_say
           + f"앞의 {len(tech)}개는 대상 F11의 그 달까지의 값으로 만든 기술지표이고, 뒤의 {K - len(tech)}개는 그 달 첫날까지 공개된 거시 변수입니다. "
           + f"그 달 F11은 변수 사전에서 공개까지 {DI['target_lag']}일로 잡혀 있어 실제 예측은 그 뒤에 하는 셈이고, 거시 변수는 더 이른 첫날 기준이라 보수적입니다. "
           + (f"다만 {', '.join(var_names[v].replace('/', ' ') for v in unconf)}의 발표시차는 아직 확인되지 않았습니다. " if unconf else "")
           + ("이 폴드에는 빈 칸이 없습니다." if nnan == 0 else f"이 폴드에는 빈 칸이 {nnan}개 있습니다.")),
      notes={L_TR: f"학습 = 달력[i − {train_len} : i − {emb}] → {len(dates)}개월 ({dates[0]} ~ {dates[-1]}), 끝 {emb}개월({', '.join(emb_ms)}) 비움",
             src.line(CAL, "te = cal[i : i + test_len]"): f"테스트 = 달력[i] 한 달 ({cap['fold']['test'][0][:7]})"},
      code=C(CAL, [L_TR], L_TR, "walkforward_folds", 12, 4),
      stack=["engine/experiment.py · make_folds", "engine/calendar.py · walkforward_folds"], nodes=["data"],
      tensors=[H("학습 폴드 원자료 (색만 표시: 열마다 최소~최대)", (rv - np.nanmin(rv, 0)) / np.where(np.nanmax(rv, 0) > np.nanmin(rv, 0), np.nanmax(rv, 0) - np.nanmin(rv, 0), 1),
                 rows=every, cols=feats, shape=[len(dates), K], scale="01", numbers=False, cellH=7, rowLabelW=64),
               {"type": "table", "title": "거시 변수: 출처 · 원래 주기 · 변환 · 발표시차 — 엔진 데이터셋의 transforms_used · lag_days_used, 출처·확인 여부는 변수 사전, 기준일은 원자료에서 직접 계산",
                "columns": tbl_cols, "rows": tbl_rows}],
      checks=([chk(f"비운 달 정답 y({emb_ms[0]}) (원자료 F11 로 계산) = 테스트 창 마지막 행의 A_ret_1 (엔진 입력, 원래 단위)", y_emb, a1_test, 1e-15, kind=kd_raw)] if emb_same is not None else [])
             + [chk(f"거시 변수 {len(macro)}개: {dates[-1]} 행의 값 = 원자료에서 직접 변환·고른 기준일 값", [DI["vars"][v]["val"] for v in macro], [float(raw[DI["vars"][v]["col"]].iloc[-1]) for v in macro], 1e-15)]
             + [chk(f"발표시차 미확인 변수: 변수 사전 {unconf} = " + (f"같은 설정의 {j_model.upper()} 결과 json({j_name})" if j_model != kind else f"결과 json({j_name})") + f" 의 lag_unconfirmed {j_unconf}",
                    float(sorted(unconf) == sorted(j_unconf)), 1.0, 0)],
      hint=(f"정보 시점(forecast_info_time = {cap['config'].get('forecast_info_time')}): 결정월 t 행의 ① 기술지표(A_…)는 F11 의 t 달까지의 값(F11_t 포함)으로, "
            "② 거시 변수는 기준일 + 발표시차 ≤ t 달 1일인 값, 곧 그 달 1일까지 공개된 값으로 만듭니다(표의 기준일; engine/calendar.py align_to_calendar). "
            + f"F11 의 발표시차는 변수 사전에서 {DI['target_lag']}일이라 t 달 1일에는 F11_t 가 아직 공개되지 않았습니다. "
            "그래서 실제 예측 시점은 F11_t 가 공개된 뒤이고, 거시 변수는 그보다 이른 1일 기준이라 보수적으로(조금 낡은 값으로) 맞춰져 있습니다. "
            + f"원래 주기가 일간인 변수는 일간 자료에서 먼저 변환(하루 변화율·하루 차이)한 뒤 월 달력에 앞 값으로 맞추므로, {day_desc}입니다." + hol_txt))
    rows_y = []
    for d in raw.index[-3:]:
        i = list(lev.index).index(d)
        rows_y.append([d.strftime("%Y-%m"), f"{lev.iloc[i]:.6f}", lev.index[i + 1].strftime("%Y-%m"), f"{lev.iloc[i + 1]:.6f}", f"{cap['y_train'].loc[d]:.8f}"])
    y_last = float(cap["y_train"].iloc[-1])
    S("1. 데이터 표",
      f"정답 y 는 결정월 t 의 F11 에 대한 <b>다음 달 F11 의 변화율</b>입니다. 예: {rows_y[-1][0]} 의 정답 = {rows_y[-1][3]} ÷ {rows_y[-1][1]} − 1 = {rows_y[-1][4]} (약 {y_last * 100:.4f}%).",
      say=(f"정답 y는 결정월 t의 F11에 대한 다음 달 F11의 변화율입니다. 다음 달 값을 이번 달 값으로 나누고 1을 뺍니다. "
           f"예를 들어 {rows_y[-1][0]}의 정답은 {rows_y[-1][2]}의 F11을 {rows_y[-1][0]}의 F11로 나누고 1을 뺀 값으로, 약 {y_last * 100:.4f}%입니다."),
      notes={src.line(CAL, "v = level.astype(float)"): "대상 F11 의 월별 값",
             src.line(CAL, "r = v.shift(-k) / v - 1.0"): f"shift(−{cap['k']}): 다음 달 값을 지금 행으로 당겨 와 나눔 → 다음 달 변화율 (마지막 {cap['k']}행은 다음 값이 없어 NaN)"},
      code=C(CAL, src.span(CAL, "def make_forecast_target", "y.name = f\"forecast_"), src.line(CAL, "r = v.shift(-k) / v - 1.0"), "make_forecast_target", 2, 2),
      stack=["engine/experiment.py · build_dataset", "engine/calendar.py · make_forecast_target"], nodes=["data"],
      formula=f"y_t = F11_(t+1) ÷ F11_t − 1",
      checks=[chk("원자료(data 저장소)에서 따로 읽은 F11 로 계산한 F11_(t+1) ÷ F11_t − 1 = 엔진 y (학습 폴드 전체)",
                  (f11r.shift(-cap["k"]) / f11r - 1).reindex(cap["y_train"].index).values, cap["y_train"].values, 1e-15, kind=kd_raw)],
      tensors=[{"type": "table", "title": "정답 만들기 (학습 폴드 마지막 3개월)", "columns": ["결정월 t", "F11_t", "다음 달", "F11_(t+1)", "정답 y_t"], "rows": rows_y},
               {"type": "line", "title": "학습 폴드의 정답 y", "x": dates, "series": [{"name": "y", "color": "#3b6fb6", "y": [num(v) for v in cap["y_train"].values]}], "height": 170}])

    # ---------------------------------------------------------------- 2. 눈금
    mu_s, sc_s = cap["scaler_mean"], cap["scaler_scale"]
    j0 = feats.index("USDKRW_pct") if "USDKRW_pct" in feats else K - 1
    v0 = float(raw[feats[j0]].iloc[-1])
    z0 = (v0 - mu_s[j0]) / sc_s[j0]
    full_scaled = np.vstack([Xw[0]] + [Xw[i][-1:] for i in range(1, len(Xw))])
    neg = pn
    vj = var_of(feats[j0]); dj = DI["vars"][vj] if vj else None
    day_ex = dj is not None and dj["freq"] == "D"
    about_j = (f"{var_names[vj]}의 {kind_word(vj)}" + (f"({day_desc[:-3]}" + (", 출처 야후 — 한국 휴장일에도 값이 있음)" if dj["src"] == "YF" else ")") if day_ex else "")) if vj else ""
    ex_txt = (f"{feats[j0]} = {about_j}: {dates[-1]} 행 = {dj['ref']:%Y-%m-%d} 의 {dj['lv_ref']:.2f} ÷ {dj['prev']:%Y-%m-%d} 의 {dj['lv_prev']:.2f} − 1 = {g4(v0)}. "
              if day_ex and dj["tr"] == "pct" else "")
    S("2. 눈금 맞추기",
      f"변수마다 <b>평균 0, 표준편차 1</b> 로 바꿉니다(StandardScaler — 이 설정에는 input_scaler 가 없어 표준화, MinMax 는 input_scaler = minmax 일 때만). "
      "평균·표준편차는 <b>학습 폴드의 행만</b>으로 구합니다. " + ex_txt +
      f"예: {feats[j0]} 의 학습 평균 {g4(mu_s[j0])}, 표준편차 {g4(sc_s[j0])} → {dates[-1]} 값 {g4(v0)} 은 ({g4(v0)} − {neg(g4(mu_s[j0]))}) ÷ {g4(sc_s[j0])} = {f(z0)}. "
      "빈 칸은 앞 값으로 채우고(ffill), 그래도 남는 빈 칸은 학습 평균(표준화 뒤 0)으로 채웁니다.",
      say=(f"변수마다 평균 0, 표준편차 1이 되도록 바꿉니다. 평균과 표준편차는 학습 폴드의 행으로만 구합니다. "
           + (f"예를 들어 {nm(feats[j0])}은, 그 달 첫날까지 자료에 값이 있는 마지막 날의, 바로 앞 관측일 대비 변화율입니다. {dates[-1]} 행의 값은 {dj['ref']:%Y-%m-%d}의 변화율 {g4(v0)}입니다. "
              if ex_txt else f"예를 들어 {nm(feats[j0])}의 {dates[-1]} 행 값은 {g4(v0)}입니다. ")
           + f"학습 평균은 {g4(mu_s[j0])}, 표준편차는 {g4(sc_s[j0])}입니다. 평균을 빼고 표준편차로 나누면 {f(z0)}입니다. "
           "빈 칸은 앞 값으로 채우고, 그래도 남으면 학습 평균, 곧 표준화 뒤 0으로 채웁니다."),
      notes={src.line(EXP, "StandardScaler()).fit(full["): f"학습 폴드 {len(dates)}행(앞 값 채운 뒤)으로 변수별 평균·표준편차(n 으로 나눔)",
             src.line(EXP, "Xf = scaler.transform(full[ds.feature_names].ffill().values)"): f"표준화 → [{len(dates)}, {K}]",
             src.line(EXP, "fill_val = np.nanmean(Xf, axis=0)"): "빈 칸 채움값 = 표준화 뒤 열 평균(≈ 0)",
             src.line(EXP, "Xf = np.where(np.isnan(Xf), fill_val[None, :], Xf)"): f"남은 빈 칸을 채움값으로 (이 폴드는 빈 칸 {nnan}개)"},
      lab=lab_scale(raw[feats[j0]].values, feats[j0], dates[-1], mu_s[j0], sc_s[j0], Xw[-1][-1, j0], about=about_j),
      code=C(EXP, src.lines(EXP, "StandardScaler()).fit(full[", "Xf = scaler.transform(full[ds.feature_names].ffill().values)", "fill_val = np.nanmean(Xf, axis=0)",
                            "Xf = np.where(np.isnan(Xf), fill_val[None, :], Xf)"), src.line(EXP, "StandardScaler()).fit(full["), "_walkforward_predict", 6, 4),
      stack=["engine/experiment.py · _walkforward_predict"], nodes=["scale"], edges=[("data", "scale")],
      formula=f"x′ = (x − 평균) ÷ 표준편차   {feats[j0]}: ({g4(v0)} − {neg(g4(mu_s[j0]))}) ÷ {g4(sc_s[j0])} = {f(z0)}",
      tensors=[{"type": "table", "split": 2, "title": "변수별 학습 평균·표준편차 (StandardScaler.mean_ / scale_)", "columns": ["변수", "평균", "표준편차"],
                "rows": [[c, num(mu_s[j]), num(sc_s[j])] for j, c in enumerate(feats)], "hlRows": [j0]},
               H("표준화한 학습 폴드 (파랑 +, 주황 −)", full_scaled, rows=[d if i % 6 == 0 else "" for i, d in enumerate(dates[:len(full_scaled)])],
                 cols=feats, shape=list(full_scaled.shape), numbers=False, cellH=6, rowLabelW=64)],
      checks=[chk(f"{feats[j0]} 표준화 (x − mean_) ÷ scale_ = 창의 값", z0, Xw[-1][-1, j0], 1e-9, kind="eq")]
             + ([chk(f"{feats[j0]}({dates[-1]}) = 원자료 {dj['ref']:%Y-%m-%d} ÷ {dj['prev']:%Y-%m-%d} − 1", v0, dj["lv_ref"] / dj["lv_prev"] - 1, 1e-15)] if ex_txt else []),
      hint="StandardScaler 의 표준편차는 n 으로 나눈 값(모표준편차)입니다.")

    # ---------------------------------------------------------------- 3. 창
    n_win_all = len(dates) - L + 1
    S("3. 창 만들기",
      f"연속한 {L}개월을 한 창으로 자릅니다. 창 j 는 표의 j ~ j+{L - 1} 행, 마지막 행이 결정월입니다. 표가 {len(dates)}행이므로 창은 {len(dates)} − {L} + 1 = {n_win_all}개"
      + (f"이고, 정답이 없는 창을 빼면 {len(Xw)}개" if len(Xw) != n_win_all else "") + f"입니다. 모양은 [{len(Xw)}, {L}, {K}].",
      say=(f"연속한 {L}개월을 한 창으로 자릅니다. 창의 마지막 행이 결정월이고, 그 창의 정답은 결정월의 정답입니다. "
           f"창은 모두 {n_win_all}개입니다. 표의 {len(dates)}행에서 창 길이 {L}을 빼고 1을 더한 수입니다. "
           + (f"정답이 없는 창을 빼면 {len(Xw)}개입니다. " if len(Xw) != n_win_all else "")
           + f"첫 창은 {dates[0]}부터 {dates[L - 1]}까지이고, 결정월은 {win_idx[0][:7]}입니다. 창 전체의 모양은 창 {len(Xw)}개, 시점 {L}개, 변수 {K}개입니다."),
      notes={src.line(CAL, "idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]"): f"idx[j] = [j, j+1, …, j+{L - 1}] → 창 {n_win_all}개 (j = 0 … {n_win_all - 1})",
             src.line(CAL, "return X[idx], y[seq_len - 1 :]"): f"정답 = y[{L - 1}:] — 창 마지막 행(결정월)의 정답"},
      code=C(CAL, src.span(CAL, "def make_windows", "return X[idx], y[seq_len - 1 :]"), src.line(CAL, "idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]"), "make_windows", 2, 2),
      stack=["engine/experiment.py · _walkforward_predict", "engine/calendar.py · make_windows"], nodes=["win"], edges=[("scale", "win")],
      formula=f"창 j = 표[j … j+{L - 1}],  정답 = y[j+{L - 1}]   (j = 0 … {len(Xw) - 1})",
      tensors=[{"type": "table", "title": "창 번호 → 달 → 결정월 → 정답", "columns": ["창 j", f"{L}개월", "결정월", "정답 y"],
                "rows": [[j, f"{dates[j]} ~ {dates[j + L - 1]}", win_idx[j][:7], f"{yw[j]:.6f}"] for j in (0, 1, 2, len(Xw) - 2, len(Xw) - 1)]},
               H(f"창 0 (= 표의 0~{L - 1}행, 표준화 값)", Xw[0], rows=[f"τ{t + 1} {dates[t]}" for t in range(L)], cols=feats, shape=[L, K], cellH=19, digits=2)])

    # ---------------------------------------------------------------- 4. 검증
    vf = cap["fold"]["val_frac"]
    nval = int(len(win_idx) * vf)
    max_win = train_len - emb - L + 1              # 어느 폴드든 학습 행은 train_len − emb 개 → 창은 많아야 이만큼
    L_IF = src.line(EXP, "if n_val >= 10 and n - n_val - emb >= 10:")
    S("4. 검증 나누기",
      f"검증에 쓸 개수는 입력 창 {len(win_idx)}개 × val_frac {vf} = {len(win_idx) * vf:.1f} → 소수점을 버린 <b>{nval}개</b>입니다. _split_fit 은 검증이 10개 미만이면 검증 없이 <b>전체 {len(Xw)}개로 학습</b>합니다. "
      + ("그래서 이 폴드에는 검증 손실도 조기종료도 없고, patience 는 쓰이지 않으며, <b>마지막 epoch 의 가중치</b>로 예측합니다." if n_va == 0 else f"검증 {n_va}개, 학습 {n_tr}개입니다."),
      say=(f"검증에 쓸 창의 개수는 입력 창 {len(win_idx)}개에 검증 비율 {vf}를 곱하면 {len(win_idx) * vf:.1f}가 되고, 소수점을 버리면 {nval}개입니다. 검증 나누기 함수는 검증이 10개보다 적으면, 검증 없이 창 {len(Xw)}개 전부로 학습합니다. "
           + ("그래서 이 폴드에는 검증 손실도 조기 종료도 없고, 마지막 에포크의 가중치로 예측합니다. " if n_va == 0 else f"검증은 {n_va}개, 학습은 {n_tr}개입니다. ")
           + (f"워크포워드 학습 구간이 {train_len}개월이라 어느 폴드든 입력 창은 많아야 {max_win}개이고 검증은 많아야 {int(max_win * vf)}개이므로, 이 설정의 모든 폴드가 같은 조건입니다."
              if int(max_win * vf) < 10 else "")),
      notes={L_IF: f"n_val = {nval} < 10 → 거짓 → 아래 m.fit(X, y)" if nval < 10 else f"n_val = {nval}",
             src.line(EXP, "m.fit(X, y)"): f"창 {len(Xw)}개 전부로 학습 (검증 없음)",
             src.line(EXP, "return n, 0"): f"(학습 {n_tr}개, 검증 {n_va}개)"},
      code=C(EXP, src.span(EXP, "def _split_fit", "return n, 0"), L_IF, "_split_fit", 1, 2),
      stack=stk_fit[:2], nodes=["split"], edges=[("win", "split")],
      formula=f"n_val = int({len(win_idx)} × {vf}) = int({len(win_idx) * vf:.1f}) = {nval}  →  " + ("10 미만 → 검증 없음, fit(X, y)" if n_va == 0 else f"검증 {n_va}"),
      hint=(f"워크포워드 학습 구간이 {train_len}개월(비움 {emb}개월을 빼면 {train_len - emb}행)이라 어느 폴드든 입력 창은 많아야 {train_len - emb} − {L} + 1 = {max_win}개, "
            f"검증 개수는 많아야 int({max_win} × {vf}) = {int(max_win * vf)}개입니다. 그래서 이 설정의 모든 폴드가 같은 조건이고, 이 설정에서는 조기종료 없이 {p['epochs']} epoch 를 다 돕니다. "
            "결과 비교 때 '검증으로 고른 시점'이 아니라 '정해진 횟수 학습'이라는 점을 적어야 합니다.") if int(max_win * vf) < 10 else None)

    # ---------------------------------------------------------------- 5. 정답 표준화
    ytr = yw[:n_tr]
    y_mu, y_sd = cap["y_mu"], cap["y_sd"]
    dec_tr = raw.index[L - 1:L - 1 + n_tr]                    # 학습 창의 결정월 = 학습 폴드 표의 L 번째 행부터(창 마지막 행)
    if [d.strftime("%Y-%m-%d") for d in dec_tr] != list(win_idx[:n_tr]):
        raise AssertionError("학습 창 결정월이 학습 폴드 표의 L 번째 행부터와 다름 — 「5. 정답 표준화」 대조를 고치세요")
    i_f = f11r.index.get_indexer(dec_tr)
    if (i_f < 0).any() or any((f11r.index[i_ + cap["k"]].to_period("M") - f11r.index[i_].to_period("M")).n != cap["k"] for i_ in i_f):
        raise AssertionError("원자료 F11 에 학습 창 결정월이나 그 다음 달이 없음")
    y_raw = f11r.values[i_f + cap["k"]] / f11r.values[i_f] - 1          # 원자료 F11 로 결정월마다 F11(t+1) ÷ F11(t) − 1 을 직접
    g5 = lambda v: f"{float(v):.5g}"
    S("5. 정답 표준화",
      f"fit 이 학습 창 정답의 평균 μ_y = {g5(y_mu)}, 표준편차 σ_y = {g5(y_sd)} 를 구해 정답을 (y − μ_y) ÷ σ_y 로 바꿉니다. 예측은 거꾸로 × σ_y + μ_y 로 되돌립니다.",
      say=(f"학습 함수는 정답도 표준화합니다. 학습 창 정답의 평균은 {g5(y_mu)}, 표준편차는 {g5(y_sd)}입니다. "
           "정답에서 평균을 빼고 표준편차로 나눈 값으로 학습하고, 예측할 때는 거꾸로 표준편차를 곱하고 평균을 더해 되돌립니다."),
      notes={src.line(MODEL, "self.y_mu, self.y_sd = float(np.mean(y))"): f"μ_y = {g5(y_mu)}, σ_y = {g5(y_sd)} (+1e−12: 0 으로 나누기 방지)",
             src.line(MODEL, "ys = (y - self.y_mu) / self.y_sd"): "정답을 표준화 — 손실(MSE)은 이 단위로 잽니다"},
      code=C(MODEL, src.lines(MODEL, "self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y) + 1e-12)", "ys = (y - self.y_mu) / self.y_sd"), src.line(MODEL, "self.y_mu, self.y_sd = float(np.mean(y))"), "Model.fit", 4, 4),
      stack=stk_fit, nodes=["split"],
      formula=f"y_z = (y − μ_y) ÷ σ_y = (y − {g5(y_mu)}) ÷ {g5(y_sd)}",
      tensors=[{"type": "table", "title": "예: 앞 3개 창", "columns": ["창", "결정월", "y", "y_z"], "rows": [[j, win_idx[j][:7], f"{ytr[j]:.6f}", f"{(ytr[j] - y_mu) / y_sd:.4f}"] for j in range(3)]}],
      checks=[chk(f"fit 의 μ_y = 원자료(data 저장소) F11 로 직접 구한 결정월 {dec_tr[0]:%Y-%m} ~ {dec_tr[-1]:%Y-%m} ({len(y_raw)}개)의 F11(t+1) ÷ F11(t) − 1 평균", y_mu, float(np.mean(y_raw)), 1e-12),
              chk(f"fit 의 σ_y = 같은 {len(y_raw)}개의 표준편차(n 으로 나눔) + 1e−12", y_sd, float(np.std(y_raw)) + 1e-12, 1e-12)])

    # ---------------------------------------------------------------- 6. 배치
    p0 = cap["perm_first"]
    nb = int(np.ceil(n_tr / bs))
    divisible = n_tr % bs == 0
    S("6. 배치",
      f"epoch 마다 <b>torch.randperm({n_tr})</b> 으로 창 순서를 섞고 앞에서부터 {bs}개씩 자릅니다. "
      + (f"{n_tr} ÷ {bs} = {nb} 이라 epoch 당 {nb}번 가중치를 고칩니다(나누어떨어지지 않으면 마지막 배치가 작아질 뿐 버리지는 않음). " if divisible else
         f"epoch 당 ⌈{n_tr} ÷ {bs}⌉ = {nb}번 가중치를 고칩니다(마지막 배치는 {n_tr - (nb - 1) * bs}개 — 버리지 않음). ")
      + "왼쪽 표는 실제 첫 epoch 의 순서입니다.",
      say=(f"에포크마다 창 {n_tr}개의 순서를 무작위로 섞고, 앞에서부터 {bs}개씩 잘라 배치를 만듭니다. "
           + (f"창 {n_tr}개를 {bs}개씩 나누면 배치가 {nb}개이므로, 에포크마다 가중치를 {nb}번 고칩니다. " if divisible else f"그래서 에포크마다 가중치를 {nb}번 고치고, 마지막 배치는 작습니다. ")
           + "왼쪽 표는 실제 첫 에포크에서 섞인 순서입니다."),
      notes={src.line(MODEL, "perm = torch.randperm(n, device=dev)"): f"0 ~ {n_tr - 1} 을 섞은 순서 (epoch 마다 새로)",
             src.line(MODEL, "for i in range(0, n, p[\"batch_size\"]):"): f"i = {', '.join(str(i) for i in range(0, n_tr, bs))} → 배치 {nb}개",
             src.line(MODEL, "idx = perm[i : i + p[\"batch_size\"]]"): f"섞인 순서의 i ~ i+{bs - 1} 번째 → 창 번호 {bs}개"},
      code=C(MODEL, src.lines(MODEL, "perm = torch.randperm(n, device=dev)", "for i in range(0, n, p[\"batch_size\"]):", "idx = perm[i : i + p[\"batch_size\"]]"), src.line(MODEL, "perm = torch.randperm(n, device=dev)"), "Model.fit", 3, 6),
      stack=stk_fit, nodes=["batch"], edges=[("split", "batch")],
      tensors=[{"type": "table", "title": "첫 epoch 의 배치 (창 번호)", "columns": ["배치", "창 번호"], "rows": [[f"{b + 1}번째", ", ".join(map(str, p0[b * bs:(b + 1) * bs]))] for b in range(nb)]}])
    S("6. 배치",
      f"이제부터 따라갈 배치는 <b>마지막 epoch 의 첫 배치</b>(창 {B}개: {', '.join(map(str, bidx))})를 학습이 끝난 가중치로 다시 넣은 것입니다. 모양은 [{B}, {L}, {K}] (배치 · 시점 τ · 변수). "
      f"왼쪽 그림은 앞쪽 창 3개입니다(위에서부터 b=0·1·2). 배치 안 번호 b=0 은 <b>첫 번째 창</b>(창 #{bidx[0]}, 결정월 {dec0})이고, 앞으로 이 창의 계산을 따라갑니다.",
      say=(f"이제부터 따라갈 배치는 마지막 에포크의 첫 배치, 창 {B}개입니다. 학습이 끝난 가중치로 이 배치를 다시 넣어 기록했습니다. "
           "왼쪽 그림은 앞쪽 창 3개입니다. "
           f"배치의 첫 번째 창은 결정월 {dec0}의 창이고, 앞으로 이 창의 계산을 따라갑니다."),
      notes={src.line(MODEL, "loss = loss_fn(self.net(Xt[idx]), yt[idx])"): f"Xt[idx]: [{B}, {L}, {K}] → net → 창마다 숫자 하나 [{B}] → 정답 yt[idx] 와 MSE"},
      code=C(MODEL, [src.line(MODEL, "loss = loss_fn(self.net(Xt[idx]), yt[idx])")], src.line(MODEL, "loss = loss_fn(self.net(Xt[idx]), yt[idx])"), "Model.fit", 6, 6),
      stack=stk_fit, nodes=["batch", "x"],
      tensors=[{"type": "batch", "title": "bx 의 앞 3개 창 (표준화 값)", "mats": [arr(xbn[b]) for b in range(3)], "labels": [f"b={b} ({('첫', '두', '세')[b]} 번째) — 창 #{bidx[b]} (결정월 {win_idx[bidx[b]][:7]})" for b in range(3)],
                "rows": tau, "cols": feats, "colNames": colnames, "scale": "div", "hl": {}, "cellH": 13, "numbers": False}])

    # ---------------------------------------------------------------- 7. 셀 가중치
    W_ih, W_hh = sd["weight_ih_l0"], sd["weight_hh_l0"]
    S(f"7. {KN} 가중치",
      f"<b>nn.{KN}({K}, {Hn})</b> 의 <b>학습이 끝난</b> 가중치입니다(맨 첫 갱신의 기울기와 fc.bias 의 처음 값은 「10. 손실 · 역전파 · Adam」). 입력용 W_ih [{G * Hn}, {K}], 지난 은닉 상태용 W_hh [{G * Hn}, {Hn}], 편향 b_ih·b_hh [{G * Hn}]. "
      f"세로 {G * Hn}줄은 {G}개 덩어리({', '.join(gname[g] for g in gates)})로 {Hn}줄씩 나뉩니다(PyTorch 의 순서). "
      f"같은 가중치를 {L}개 시점 모두에 씁니다."
      + ("" if kind == "lstm" else f" LSTM(덩어리 4개)보다 하나 적어 전체 가중치 수가 {cap['n_params']:,}개입니다(같은 크기의 LSTM 이면 {n_cell(4):,}개)."),
      say=((f"LSTM 층의 학습이 끝난 가중치입니다. 입력에 곱하는 가중치는 세로 {G * Hn}줄, 가로 {K}칸이고, 지난 은닉 상태에 곱하는 가중치는 세로 {G * Hn}줄, 가로 {Hn}칸이며, 편향이 두 벌 있습니다. "
            f"세로 {G * Hn}줄은 {Hn}줄씩 네 덩어리로, 차례로 입력 문, 잊기 문, 새 정보, 출력 문의 몫입니다. 이 가중치 한 벌을 {L}개 시점 모두에 똑같이 씁니다.")
           if kind == "lstm" else
           (f"GRU 층의 학습이 끝난 가중치입니다. 입력에 곱하는 가중치는 세로 {G * Hn}줄, 가로 {K}칸이고, 지난 은닉 상태에 곱하는 가중치는 세로 {G * Hn}줄, 가로 {Hn}칸이며, 편향이 두 벌 있습니다. "
            f"세로 {G * Hn}줄은 {Hn}줄씩 세 덩어리로, 차례로 재설정 문, 갱신 문, 새 후보의 몫입니다. "
            f"LSTM의 네 덩어리보다 하나 적어서, 전체 가중치는 {cap['n_params']:,}개입니다. 같은 크기의 LSTM이라면 {n_cell(4):,}개입니다. 이 가중치 한 벌을 {L}개 시점 모두에 씁니다.")),
      notes={L_RNN: f"W_ih [{G * Hn}, {K}] · W_hh [{G * Hn}, {Hn}] · b_ih · b_hh [{G * Hn}] — {'·'.join(gates)} 순서로 {Hn}줄씩"},
      code=C(MODEL, [L_RNN], L_RNN, "_Net.__init__", 3, 6),
      stack=[f"{MODEL} · _Net.__init__"], nodes=cell_nodes,
      tensors=[{"type": "row", "items": [H(f"W_ih [{G * Hn}, {K}]", W_ih, rows=[f"{g}{u + 1}" if u % 4 == 0 else "" for g in gates for u in range(Hn)], cols=feats, shape=list(W_ih.shape), numbers=False, cellH=5, cellW=14, rowLabelW=40),
                                         H(f"W_hh [{G * Hn}, {Hn}]", W_hh, rows=[f"{g}{u + 1}" if u % 4 == 0 else "" for g in gates for u in range(Hn)], cols=units, shape=list(W_hh.shape), numbers=False, cellH=5, cellW=14, rowLabelW=40)]}],
      hint=("가로줄 이름 i1, i5 … 는 '입력 문의 1번·5번 은닉 단위' 처럼 읽습니다(f… 잊기 문, g… 새 정보, o… 출력 문). 색: 파랑 +, 주황 −." if kind == "lstm" else
            "가로줄 이름 r1, r5 … 는 '재설정 문의 1번·5번 은닉 단위' 처럼 읽습니다(z… 갱신 문, n… 새 후보). 색: 파랑 +, 주황 −."))

    # ---------------------------------------------------------------- 8. 시점별 계산
    out_hand = rec[-1]["h"]
    drop_in = cap["batch"]["drop_in"].double().numpy()
    f4 = lambda v: f"{float(v):.4f}"                     # 예시 값은 소수 넷째 자리까지(끝 0 도 그대로)
    tN = lambda t: f"c{t + 1}"
    x1 = xbn[0, 0]
    r0 = rec[0]
    u1 = int(np.argmax(np.abs(out_rec[0, 0])))          # τ1 에서 |h| 가 가장 큰 은닉 단위(직접 실행해 보기)
    if kind == "lstm":
        rnn1 = torch.nn.LSTM(K, Hn, batch_first=True)
        rnn1.load_state_dict({k_.split(".", 1)[1]: v for k_, v in sdt.items() if k_.startswith("lstm.")})
        with torch.no_grad():
            _, (h_t1, c_t1) = rnn1(xb[0:1, 0:1])           # PyTorch 로 τ1 한 시점 — c 는 out 에 없어서
        lab1 = lab_t1(kind, x1, W_ih, sd["bias_ih_l0"], sd["bias_hh_l0"], Hn, u1 + 1, dec0, out_rec[0, 0, u1], float(c_t1[0, 0, u1]))
        S("8. 시점별 계산",
          f"<b>τ1</b>: 첫 번째 창(b=0)의 첫 행 x₁ (변수 {K}개)을 읽습니다. 지난 은닉 상태 h₀ 와 셀 상태 c₀ 는 0 으로 시작하므로(forward 가 넘기지 않으면 PyTorch 기본값 0) W_hh·h₀ = 0, 편향만 남습니다. "
          f"W_ih·x₁ + b_ih + b_hh 를 계산하면 {G * Hn}개 숫자가 나오고, 이것을 {Hn}개씩 4덩어리로 잘라 i·f·o 에는 sig(시그모이드, 0~1), g 에는 tanh(−1~1) 를 씌웁니다.",
          say=(f"첫 시점 τ1입니다. 첫 번째 창의 첫 행, 변수 {K}개를 읽습니다. 지난 은닉 상태와 셀 상태는 0에서 시작하므로, 지난 상태 쪽 항은 편향만 남습니다. "
               f"입력에 가중치를 곱해 더하고 편향 두 벌을 더하면 {G * Hn}개 숫자가 나오고, 이것을 {Hn}개씩 네 덩어리로 자릅니다. "
               "입력 문, 잊기 문, 출력 문에는 0과 1 사이로 누르는 시그모이드를, 새 정보에는 −1과 1 사이로 누르는 하이퍼볼릭 탄젠트를 씌웁니다."),
          notes={L_OUT: "h₀·c₀ 를 넘기지 않음 → 0 에서 시작. τ1 의 지난 상태 쪽 W_hh·h₀ + b_hh = b_hh"},
          code=C(MODEL, [L_OUT], L_OUT, "_Net.forward → nn.LSTM (C++)", 3, 4),
          stack=stk_fwd + ["nn.LSTM · 시점 τ1"], nodes=["x", "c1"], edges=[("x", "c1")],
          formula=f"[a_i a_f a_g a_o] = W_ih·x₁ + b_ih + W_hh·h₀ + b_hh   (h₀ = 0)\ni = sig(a_i), f = sig(a_f), g = tanh(a_g), o = sig(a_o)   (sig(a) = 1 ÷ (1 + e^(−a)), 시그모이드)",
          tensors=[H("b=0 의 x₁ (τ1 행)", x1[None, :], rows=["x₁"], cols=feats, shape=[1, K], digits=2, cellH=26),
                   H(f"b=0 의 W_ih·x₁ + b (4덩어리 × {Hn})", r0["pre"][0].reshape(G, Hn), rows=["a_i", "a_f", "a_g", "a_o"], cols=units, shape=[G, Hn], digits=2, cellH=26)])
        S("8. 시점별 계산",
          f"<b>τ1 상태 갱신</b>: c₁ = f ⊙ c₀ + i ⊙ g = i ⊙ g (c₀ = 0), h₁ = o ⊙ tanh(c₁). 표는 b=0 의 은닉 단위 {Hn}개 전부입니다. "
          f"입력 문 i 가 클수록 새 정보 g 를 셀 상태 c 에 많이 넣고, 출력 문 o 가 클수록 셀 상태를 은닉 상태 h 로 많이 내보냅니다. "
          f"예: 은닉 단위 u{u1 + 1}: i = {f4(r0['i'][0, u1])}, g = {f4(r0['g'][0, u1])}, o = {f4(r0['o'][0, u1])} → c₁ = {f4(r0['c'][0, u1])}, h₁ = {f4(r0['h'][0, u1])}.",
          say=("첫 시점의 상태 갱신입니다. 새 셀 상태는, 지난 셀 상태에 잊기 문을 곱한 것과 새 정보에 입력 문을 곱한 것을 더한 값입니다. "
               "첫 시점은 지난 셀 상태가 0이라, 새 정보에 입력 문을 곱한 것만 남습니다. 새 은닉 상태는 셀 상태에 하이퍼볼릭 탄젠트를 씌운 값에 출력 문을 곱한 것입니다. "
               "입력 문이 클수록 새 정보를 셀 상태에 많이 넣고, 출력 문이 클수록 셀 상태를 은닉 상태로 많이 내보냅니다. "
               f"예를 들어 첫 번째 창의 은닉 단위 {u1 + 1}은 셀 상태 {f4(r0['c'][0, u1])}, 은닉 상태 {f4(r0['h'][0, u1])}입니다."),
          notes={L_OUT: f"τ1: c₁ = i ⊙ g, h₁ = o ⊙ tanh(c₁) — 예) u{u1 + 1}: c₁ = {f4(r0['c'][0, u1])}, h₁ = {f4(r0['h'][0, u1])}"},
          lab=lab1,
          code=C(MODEL, [L_OUT], L_OUT, "nn.LSTM (C++)", 3, 4),
          stack=stk_fwd + ["nn.LSTM · 시점 τ1"], nodes=["c1"],
          formula="c₁ = f ⊙ c₀ + i ⊙ g,   h₁ = o ⊙ tanh(c₁)",
          tensors=[H("b=0 τ1 의 문·상태 (sig·tanh 적용 뒤, 직접 계산)", np.stack([r0[g_][0] for g_ in ("i", "f", "g", "o", "c", "h")]), rows=["i 입력 문", "f 잊기 문", "g 새 정보", "o 출력 문", "c₁ 셀 상태", "h₁ 은닉 상태"],
                     cols=units, shape=[6, Hn], digits=2, cellH=26, rowLabelW=88)],
          checks=[chk(f"τ1 직접 계산(64비트) h₁ = 기록한 out[:, 0, :] (배치 {B}개 전부)", r0["h"], out_rec[:, 0, :], 2e-6)])
        r1 = rec[1]
        fm = f(r1['f'][0].mean(), 3)
        S("8. 시점별 계산",
          f"<b>τ2</b>: 이번에는 지난 은닉 상태 h₁ 이 W_hh 를 거쳐 들어옵니다. 같은 문 계산에 '입력 쪽' W_ih·x₂ + b_ih 와 '지난 상태 쪽' W_hh·h₁ + b_hh 가 더해집니다. "
          f"그리고 c₂ = f ⊙ c₁ + i ⊙ g — 잊기 문 f 가 지난 셀 상태 c₁ 을 얼마나 남길지 정합니다. b=0 의 f 평균({Hn}개)은 {fm} 입니다.",
          say=("두 번째 시점을 봅니다. 이번에는 지난 은닉 상태가 지난 상태 쪽 가중치를 거쳐 들어와, 문마다 입력 쪽 값에 더해집니다. "
               "새 셀 상태는 지난 셀 상태에 잊기 문을 곱한 것과 새 정보에 입력 문을 곱한 것을 더한 값이라, 잊기 문이 지난 셀 상태를 얼마나 남길지 정합니다. "
               f"두 번째 시점에서 첫 번째 창의 잊기 문 {Hn}개의 평균은 {fm}입니다."),
          notes={L_OUT: "τ2: 지난 상태 쪽 W_hh·h₁ + b_hh 가 더해짐, c₂ = f ⊙ c₁ + i ⊙ g"},
          code=C(MODEL, [L_OUT], L_OUT, "nn.LSTM (C++)", 3, 4),
          stack=stk_fwd + ["nn.LSTM · 시점 τ2"], nodes=["c1", "c2"], edges=[("c1", "c2")],
          formula="a = (W_ih·x₂ + b_ih) + (W_hh·h₁ + b_hh),   c₂ = f ⊙ c₁ + i ⊙ g,   h₂ = o ⊙ tanh(c₂)",
          tensors=[H("b=0 τ2: 입력 쪽 W_ih·x₂ + b_ih (4덩어리)", r1["a_i"][0].reshape(G, Hn), rows=["i", "f", "g", "o"], cols=units, shape=[G, Hn], digits=2, cellH=22),
                   H("b=0 τ2: 지난 상태 쪽 W_hh·h₁ + b_hh (4덩어리)", r1["a_h"][0].reshape(G, Hn), rows=["i", "f", "g", "o"], cols=units, shape=[G, Hn], digits=2, cellH=22),
                   H("b=0 τ2 의 문·상태", np.stack([r1[g_][0] for g_ in ("i", "f", "g", "o", "c", "h")]), rows=["i", "f", "g", "o", "c₂", "h₂"], cols=units, shape=[6, Hn], digits=2, cellH=22)])
        show = [("f", "잊기 문 f (1 에 가까우면 셀 상태 유지)"), ("i", "입력 문 i"), ("c", "셀 상태 c"), ("h", "은닉 상태 h")]
        show_say = "위 줄은 왼쪽부터 잊기 문과 입력 문, 아래 줄은 셀 상태와 은닉 상태입니다."
    else:
        lab1 = lab_t1(kind, x1, W_ih, sd["bias_ih_l0"], sd["bias_hh_l0"], Hn, u1 + 1, dec0, out_rec[0, 0, u1])
        S("8. 시점별 계산",
          f"<b>τ1</b>: GRU 의 문은 2개(재설정 문 r, 갱신 문 z)이고, 여기에 새 후보 n 을 더해 3덩어리를 계산합니다. 상태는 은닉 상태 h 하나뿐입니다(LSTM 의 셀 상태 c 가 없음). "
          f"첫 번째 창(b=0)의 첫 행 x₁ 에 대한 입력 쪽 W_ih·x₁ + b_ih 와 지난 상태 쪽 W_hh·h₀ + b_hh(h₀ = 0 이라 b_hh 만)를 각각 {Hn}개씩 3덩어리로 자릅니다. "
          "r·z 는 두 쪽을 더해 sig(시그모이드) 를 씌우지만, n 은 지난 상태 쪽 덩어리에 r 을 먼저 곱한 뒤 입력 쪽 덩어리와 더해 tanh 를 씌웁니다 — LSTM 은 네 덩어리 모두 두 쪽을 그냥 더합니다.",
          say=("첫 시점 τ1입니다. GRU의 문은 두 개, 재설정 문과 갱신 문이고, 여기에 새 후보를 더해 세 덩어리를 계산합니다. 상태는 은닉 상태 하나뿐이고, LSTM의 셀 상태는 없습니다. "
               f"입력 쪽과 지난 상태 쪽을 따로 계산해, 각각 {Hn}개씩 세 덩어리로 자릅니다. 첫 시점은 지난 상태가 0이라 지난 상태 쪽은 편향만 남습니다. "
               "재설정 문과 갱신 문은 두 쪽을 더해 시그모이드를 씌웁니다. 새 후보는 지난 상태 쪽 덩어리에 재설정 문을 먼저 곱한 뒤 입력 쪽 덩어리와 더하고, 하이퍼볼릭 탄젠트를 씌웁니다. "
               "네 덩어리 모두 두 쪽을 그냥 더하는 LSTM과 다른 점입니다."),
          notes={L_OUT: "h₀ 를 넘기지 않음 → 0 에서 시작. τ1 의 지난 상태 쪽 W_hh·h₀ + b_hh = b_hh"},
          code=C(MODEL, [L_OUT], L_OUT, "_Net.forward → nn.GRU (C++)", 3, 4),
          stack=stk_fwd + ["nn.GRU · 시점 τ1"], nodes=["x", "c1"], edges=[("x", "c1")],
          formula="r = sig(W_ir·x + b_ir + W_hr·h + b_hr),  z = sig(W_iz·x + b_iz + W_hz·h + b_hz)   (sig(a) = 1 ÷ (1 + e^(−a)), 시그모이드)\nn = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn)),  h′ = (1 − z) ⊙ n + z ⊙ h",
          hint="이 식은 PyTorch nn.GRU 문서의 식입니다. 재설정 문 r 을 지난 상태 쪽 덩어리 전체에 곱합니다: r ⊙ (W_hn·h + b_hn). "
               "GRU 원 논문(Cho et al., 2014)은 행렬을 곱하기 전에 곱합니다: W_hn·(r ⊙ h). 두 식은 같은 값이 아니며, 이 플랫폼(PyTorch)은 앞의 식을 씁니다.",
          tensors=[H("b=0 의 x₁ (τ1 행)", x1[None, :], rows=["x₁"], cols=feats, shape=[1, K], digits=2, cellH=26),
                   H(f"b=0 입력 쪽 W_ih·x₁ + b_ih (3덩어리 × {Hn})", r0["a_i"][0].reshape(G, Hn), rows=["r", "z", "n"], cols=units, shape=[G, Hn], digits=2, cellH=24),
                   H(f"b=0 지난 상태 쪽 W_hh·h₀ + b_hh = b_hh", r0["a_h"][0].reshape(G, Hn), rows=["r", "z", "n"], cols=units, shape=[G, Hn], digits=2, cellH=24)])
        S("8. 시점별 계산",
          "<b>τ1 상태 갱신</b>: h₁ = (1 − z) ⊙ n + z ⊙ h₀. 갱신 문 z 가 1 에 가까우면 지난 은닉 상태를 그대로 두고, 0 에 가까우면 새 후보 n 으로 바꿉니다. "
          "τ1 은 h₀ = 0 이라 h₁ = (1 − z) ⊙ n. 재설정 문 r 은 n 을 만들 때 지난 상태 쪽 덩어리(τ1 에서는 b_hn)를 얼마나 쓸지 정합니다. "
          f"예: 은닉 단위 u{u1 + 1}: r = {f4(r0['r'][0, u1])}, z = {f4(r0['z'][0, u1])}, n = {f4(r0['n'][0, u1])} → h₁ = (1 − {f4(r0['z'][0, u1])}) × {pn(f4(r0['n'][0, u1]))} = {f4(r0['h'][0, u1])}.",
          say=("첫 시점의 상태 갱신입니다. 새 은닉 상태는, 새 후보에 1에서 갱신 문을 뺀 값을 곱한 것과, 지난 은닉 상태에 갱신 문을 곱한 것을 더한 값입니다. "
               "갱신 문이 1에 가까우면 지난 상태를 그대로 두고, 0에 가까우면 새 후보로 바꿉니다. 첫 시점은 지난 상태가 0이라, 새 후보에 1에서 갱신 문을 뺀 값을 곱한 것만 남습니다. "
               f"예를 들어 첫 번째 창의 은닉 단위 {u1 + 1}은 갱신 문 {f4(r0['z'][0, u1])}, 새 후보 {f4(r0['n'][0, u1])}, 은닉 상태 {f4(r0['h'][0, u1])}입니다."),
          notes={L_OUT: f"τ1: h₁ = (1 − z) ⊙ n — 예) u{u1 + 1}: (1 − {f4(r0['z'][0, u1])}) × {pn(f4(r0['n'][0, u1]))} = {f4(r0['h'][0, u1])}"},
          lab=lab1,
          code=C(MODEL, [L_OUT], L_OUT, "nn.GRU (C++)", 3, 4),
          stack=stk_fwd + ["nn.GRU · 시점 τ1"], nodes=["c1"],
          formula="h₁ = (1 − z) ⊙ n + z ⊙ h₀ = (1 − z) ⊙ n",
          tensors=[H("b=0 τ1 의 문·상태 (직접 계산)", np.stack([r0[g_][0] for g_ in ("r", "z", "n", "h")]), rows=["r 재설정 문", "z 갱신 문", "n 새 후보", "h₁ 은닉 상태"], cols=units, shape=[4, Hn], digits=2, cellH=26, rowLabelW=88)],
          checks=[chk(f"τ1 직접 계산(64비트) h₁ = 기록한 out[:, 0, :] (배치 {B}개 전부)", r0["h"], out_rec[:, 0, :], 2e-6)])
        r1 = rec[1]
        zm, rm = f(r1['z'][0].mean(), 3), f(r1['r'][0].mean(), 3)
        S("8. 시점별 계산",
          f"<b>τ2</b>: 지난 은닉 상태 h₁ 이 W_hh 를 거쳐 들어옵니다. r 이 0 에 가까우면 n 을 만들 때 지난 상태 쪽 덩어리(W_hn·h₁ + b_hn)를 거의 쓰지 않고, 1 에 가까우면 그대로 씁니다. "
          f"h₂ = (1 − z) ⊙ n + z ⊙ h₁ — b=0 의 z 평균({Hn}개)은 {zm}, r 평균은 {rm} 입니다.",
          say=("두 번째 시점을 봅니다. 이제 지난 은닉 상태가 지난 상태 쪽 가중치를 거쳐 들어옵니다. 재설정 문이 0에 가까우면 새 후보를 만들 때 지난 상태 쪽 덩어리를 거의 쓰지 않고, 1에 가까우면 그대로 씁니다. "
               "새 은닉 상태는, 새 후보에 1에서 갱신 문을 뺀 값을 곱한 것과, 지난 은닉 상태에 갱신 문을 곱한 것을 더한 값입니다. "
               f"두 번째 시점에서 첫 번째 창의 갱신 문 {Hn}개의 평균은 {zm}, 재설정 문의 평균은 {rm}입니다."),
          notes={L_OUT: "τ2: n = tanh(입력 쪽 + r ⊙ (W_hn·h₁ + b_hn)), h₂ = (1 − z) ⊙ n + z ⊙ h₁"},
          code=C(MODEL, [L_OUT], L_OUT, "nn.GRU (C++)", 3, 4),
          stack=stk_fwd + ["nn.GRU · 시점 τ2"], nodes=["c1", "c2"], edges=[("c1", "c2")],
          formula="n = tanh(W_in·x₂ + b_in + r ⊙ (W_hn·h₁ + b_hn)),   h₂ = (1 − z) ⊙ n + z ⊙ h₁",
          tensors=[H("b=0 τ2: 지난 상태 쪽 W_hh·h₁ + b_hh", r1["a_h"][0].reshape(G, Hn), rows=["r", "z", "n"], cols=units, shape=[G, Hn], digits=2, cellH=22),
                   H("b=0 τ2 의 문·상태", np.stack([r1[g_][0] for g_ in ("r", "z", "n", "h")]), rows=["r", "z", "n", "h₂"], cols=units, shape=[4, Hn], digits=2, cellH=22)])
        show = [("z", "갱신 문 z (1 에 가까우면 은닉 상태 유지)"), ("r", "재설정 문 r"), ("n", "새 후보 n"), ("h", "은닉 상태 h")]
        show_say = "위 줄은 왼쪽부터 갱신 문과 재설정 문, 아래 줄은 새 후보와 은닉 상태입니다."
    d12 = d_cell; place12 = -int(np.floor(np.log10(d12))) if d12 > 0 else 0         # 모든 시점의 최대 차이(「0. 시작」 힌트와 같은 값)
    S("8. 시점별 계산",
      f"<b>τ3 ~ τ{L}</b>: 같은 계산을 시점마다 반복합니다(가중치는 같고, 입력 x_τ 와 지난 상태만 바뀜). 왼쪽 네 그림은 b=0 의 {L}개 시점 전부입니다(그림마다 행 = 시점, 열 = 은닉 단위; "
      f"위 줄 {show[0][0]}·{show[1][0]}, 아래 줄 {show[2][0]}·{show[3][0]} — 직접 계산). "
      f"PyTorch 의 out 은 시점마다의 은닉 상태 h 를 모은 [{B}, {L}, {Hn}] 입니다. 64비트로 직접 계산한 h 는 τ1 ~ τ{L} 모든 시점에서 기록한 out 과 최대 {d12:.1e} 차이로 같습니다(아래 '검증' 줄).",
      say=(f"τ3부터 τ{L}까지 같은 계산을 시점마다 되풀이합니다. 가중치는 같고, 입력 행과 지난 상태만 바뀝니다. "
           f"왼쪽 네 그림은 첫 번째 창의 {L}개 시점을 모은 것으로, 그림마다 행이 시점, 열이 은닉 단위입니다. {show_say} "
           f"이 식을 64비트로 직접 계산한 은닉 상태는, 모든 시점과 배치의 창 {B}개 모두에서 기록한 출력과 소수 {ORD_KO.get(place12, str(place12) + '째')} 자리 수준의 차이 안에서 같았습니다."),
      notes={L_OUT: f"out [{B}, {L}, {Hn}] = τ1 … τ{L} 의 h 를 모은 것"},
      code=C(MODEL, [L_OUT], L_OUT, f"nn.{KN} (C++)", 3, 4),
      stack=stk_fwd + [f"nn.{KN} · 시점 τ1 … τ{L}"], nodes=cell_nodes, edges=[(tN(t), tN(t + 1)) for t in range(L - 1)],
      tensors=[{"type": "row", "items": [H(f"b=0 {ttl}", np.stack([rr[g_][0] for rr in rec]), rows=tau, cols=units, shape=[L, Hn], digits=2, cellW=29, cellH=18, rowLabelW=34,
                                           scale="01" if g_ in ("f", "i", "o", "z", "r") else "div") for g_, ttl in show]}],
      checks=[chk(f"{L}개 시점 모두(τ1 ~ τ{L}) 직접 계산(64비트) h = 기록한 out (배치 {B}개 전부)", np.stack([rr["h"] for rr in rec], 1), out_rec, 2e-5)])

    # ---------------------------------------------------------------- 9. 마지막 시점 → 드롭아웃 → Linear
    S("9. 마지막 시점 · 드롭아웃 · Linear",
      f"forward 는 <b>out[:, −1, :]</b> — 마지막 시점 τ{L} 의 은닉 상태만 꺼냅니다([{B}, {L}, {Hn}] → [{B}, {Hn}]). 앞 시점들의 h 는 버리지만, "
      f"h₁₂ 는 시점마다 이어받은 상태를 거쳐 계산되므로 창의 {L}개 행 x₁ ~ x₁₂ 모두에 따라 달라집니다.",
      say=(f"순전파는 마지막인 {L}번째 시점의 은닉 상태만 꺼냅니다. 창마다 은닉 {Hn}개가 남습니다. "
           f"앞 시점들의 은닉 상태는 버리지만, 마지막 은닉 상태는 시점마다 이어받은 상태를 거쳐 계산되므로, 창의 {L}개 행 모두에 따라 달라집니다."),
      notes={L_RET: f"out[:, −1, :]: [{B}, {L}, {Hn}] → [{B}, {Hn}] (마지막 시점 τ{L})"},
      code=C(MODEL, [L_RET], L_RET, "_Net.forward", 3, 2),
      stack=stk_fwd, nodes=["last"], edges=[(f"c{L}", "last")],
      tensors=[H(f"h₁₂ = out[:, −1, :]  [{B}, {Hn}]", drop_in, rows=[f"b={b} 창#{bidx[b]}" for b in range(B)], cols=units, shape=[B, Hn], digits=2, cellH=17, rowLabelW=86)])
    dout = cap["batch"]["drop_out"].double().numpy()
    nz = int((dout == 0).sum())
    keep = 1 - p["dropout"]
    L_DROP = src.line(MODEL, "self.drop = nn.Dropout(dropout)")
    S("9. 마지막 시점 · 드롭아웃 · Linear",
      f"<b>Dropout({p['dropout']})</b>: 학습 때만 칸마다 {int(p['dropout'] * 100)}% 확률로 0 으로 지우고, 남은 칸은 1 ÷ {keep} = {f(1 / keep)} 배로 키웁니다(기댓값이 유지되게). "
      f"이 화면은 학습 모드로 다시 넣은 것(드롭아웃 난수 시드 {seed_fwd})이라 {B * Hn}칸 중 {nz}칸({nz / (B * Hn) * 100:.1f}%)이 지워졌습니다 — 실제 학습 때 이 배치에서 지운 칸은 따로 뽑은 난수라 기록에 없습니다. "
      "예측(eval)할 때는 아무것도 하지 않습니다.",
      say=(f"드롭아웃은 학습할 때만 작동합니다. 칸마다 {int(p['dropout'] * 100)}% 확률로 0으로 지우고, 남은 칸은 1 나누기 {keep}, 곧 {f(1 / keep)}배로 키워서 기댓값이 유지되게 합니다. "
           f"이 화면은 학습 모드로 다시 넣은 것이고, 지워진 칸은 {nz}칸으로 전체의 {nz / (B * Hn) * 100:.1f}%입니다. 실제 학습 때 이 배치에서 지운 칸은 기록에 없습니다. "
           "예측할 때는 드롭아웃이 아무것도 하지 않습니다."),
      notes={L_DROP: f"Dropout(p = {p['dropout']}) — 학습 모드에서만 작동",
             L_RET: f"self.drop(…): 지운 칸 0, 남은 칸 × {f(1 / keep)}"},
      code=C(MODEL, [L_DROP, L_RET], L_RET, "_Net", 3, 2),
      stack=stk_fwd + ["nn.Dropout (학습 모드)"], nodes=["drop"], edges=[("last", "drop")],
      formula=f"학습: d = h ⊙ m ÷ {keep}  (m ∈ {{0, 1}}, P(m = 0) = {p['dropout']}),   예측: d = h",
      tensors=[H("드롭아웃 뒤 (0 = 지워진 칸)", dout, rows=[f"b={b}" for b in range(B)], cols=units, shape=[B, Hn], digits=2, cellH=17,
                 hl={"cells": [[int(i), int(j)] for i, j in zip(*np.where(dout == 0))]})],
      checks=[chk(f"남은 칸 = h × {f(1 / keep)}", dout[dout != 0], drop_in[dout != 0] / keep, 1e-5)])
    yz_b = np.array(cap["batch"]["y_z"])
    out_tr = cap["batch"]["out_train"].double().numpy(); out_ev = cap["batch"]["out_eval"].double().numpy()
    L_FC = src.line(MODEL, "self.fc = nn.Linear(hidden, 1)")
    nz0 = int((dout[0] == 0).sum())                 # 첫 번째 창에서 지워진 칸 수
    S("9. 마지막 시점 · 드롭아웃 · Linear",
      f"<b>Linear({Hn}→1)</b>: 은닉 {Hn}개에 가중치를 곱해 더하고 편향을 더해 창마다 숫자 하나를 냅니다. squeeze(−1) 로 [{B}, 1] → [{B}]. "
      f"b=0(첫 번째 창): 학습 모드(드롭아웃 켬) {f(out_tr[0])}, 예측 모드 {f(out_ev[0])}. 두 값의 차이는 드롭아웃에서만 옵니다(아래 두 검증)"
      + (f" — 이 창은 지워진 칸이 {nz0}개라, 남은 칸을 {f(1 / keep)}배로 키운 것만으로 달라졌습니다. " if nz0 == 0 else f" — 이 창은 {Hn}칸 중 {nz0}칸이 지워지고 나머지는 {f(1 / keep)}배로 커졌습니다. ")
      + "이 값은 표준화한 정답 단위입니다.",
      say=(f"선형 층은 은닉 {Hn}개에 가중치를 곱해 더하고 편향을 더해, 창마다 숫자 하나를 냅니다. "
           f"첫 번째 창의 값은 드롭아웃을 켠 학습 모드에서는 {f(out_tr[0])}로, 예측 모드에서는 {f(out_ev[0])}로 나옵니다. "
           "두 값의 차이는 드롭아웃에서만 옵니다. "
           + (f"이 창은 지워진 칸이 없어서, 남은 칸을 {f(1 / keep)}배로 키운 것만으로 값이 달라졌습니다. " if nz0 == 0 else f"이 창은 {nz0}칸이 지워지고, 나머지는 {f(1 / keep)}배로 커졌습니다. ")
           + "이 값은 표준화한 정답의 단위입니다."),
      notes={L_FC: f"Linear({Hn}→1): 가중치 w [{Hn}], 편향 b_fc = {f(b_fc)} (학습이 끝난 값)",
             L_RET: f"fc(…) → [{B}, 1] → squeeze(−1) → [{B}]"},
      lab=lab_linear(drop_in[0], dout[0], W_fc, b_fc, dec0, keep, nz0, out_tr[0], out_ev[0]),
      code=C(MODEL, [L_FC, L_RET], L_RET, "_Net", 3, 2),
      stack=stk_fwd + ["nn.Linear"], nodes=["fc"], edges=[("drop", "fc")],
      formula=f"ŷ_z = Σ_u w_u · d_u + b_fc = " + " + ".join(f"{pn(f(W_fc[u], 2))} × {pn(f(dout[0, u], 2))}" for u in range(min(4, Hn))) + f" + … + {pn(f(b_fc))} = {f(out_tr[0])}   (b_fc = Linear 의 편향, 소수 둘째 자리로 줄여 보임)",
      tensors=[H(f"fc 가중치 w [{Hn}] · 편향 b_fc = {f(b_fc)}", W_fc[None, :], rows=["w"], cols=units, shape=[1, Hn], digits=2, cellH=26),
               {"type": "table", "title": f"배치 {B}개의 출력과 정답", "columns": ["b(배치 안 번호)", "창", "학습 모드 ŷ_z", "예측 모드 ŷ_z", "정답 y_z"],
                "rows": [[b, bidx[b], f"{out_tr[b]:.4f}", f"{out_ev[b]:.4f}", f"{yz_b[b]:.4f}"] for b in range(B)], "split": 2}],
      checks=[chk("Linear 직접 계산(학습 모드) = 기록한 출력", out_tr, dout @ W_fc + b_fc, 1e-5), chk("Linear 직접 계산(예측 모드, 드롭아웃 없음) = 기록한 출력", out_ev, out_hand @ W_fc + b_fc, 2e-5)])

    # ---------------------------------------------------------------- 10. 손실·역전파·잘라내기·Adam
    lt = cap["batch"]["loss_train"]
    _cl = (ROOT / "tools/codewalk/rnn_capture.py").read_text(encoding="utf-8").split("\n")
    L_CAPMSE = next((i + 1 for i, t_ in enumerate(_cl) if "mse_loss(out_tr" in t_), None)      # 화면 손실을 잰 줄(기록 도구)
    L_CAPFWD = next((i + 1 for i, t_ in enumerate(_cl) if t_.strip() == "out_tr = net(xb)"), None)          # 다시 넣기(학습 모드 순전파) 줄
    if L_CAPMSE is None or L_CAPFWD is None:
        raise AssertionError("rnn_capture.py 에서 손실을 재는 줄(mse_loss(out_tr …))을 찾지 못함 — 「10」의 출처 설명을 고치세요")
    S("10. 손실 · 역전파 · Adam",
      f"손실은 <b>MSE</b> = 평균((ŷ_z − y_z)²) 입니다. 학습 중에는 줄 {L_LOSS} 이 배치마다 이 값을 계산합니다. "
      f"화면의 {f(lt)} 은 그 줄이 낸 값이 아니라, 기록 도구 tools/codewalk/rnn_capture.py" + f"(다시 넣기 줄 {L_CAPFWD}, 손실 줄 {L_CAPMSE})"
      + " 가 학습이 끝난 가중치로 이 배치를 학습 모드(드롭아웃 켬)로 다시 넣고 같은 식(F.mse_loss)으로 잰 값입니다. loss.backward() 가 모든 가중치의 기울기를 계산합니다.",
      say=(f"손실은 평균제곱오차입니다. 창마다 출력과 표준화한 정답의 차이를 제곱해, 창 {B}개에 대해 평균냅니다. "
           f"화면의 값은 학습 중에 잰 것이 아니라, 기록 도구가 학습이 끝난 가중치로 이 배치를 학습 모드로 다시 넣고 같은 식으로 잰 값으로, {f(lt)}로 나옵니다. "
           "그다음 역전파가 모든 가중치의 기울기를 계산합니다."),
      notes={src.line(MODEL, "loss_fn = nn.MSELoss()"): "MSELoss: 평균((출력 − 정답)²)",
             L_LOSS: f"학습 중 배치 {B}개의 MSE 를 계산하는 줄 — 화면의 {f(lt)} 은 이 줄이 낸 값이 아니라 rnn_capture.py 가 같은 식으로 잰 값",
             src.line(MODEL, "loss.backward()"): "모든 가중치의 기울기를 .grad 에 계산 (바로 앞 zero_grad 로 0 에서 시작)"},
      code=C(MODEL, src.lines(MODEL, "loss_fn = nn.MSELoss()", "loss = loss_fn(self.net(Xt[idx]), yt[idx])", "loss.backward()"), src.line(MODEL, "loss.backward()"), "Model.fit", 3, 4),
      stack=stk_fit, nodes=["loss"], edges=[("fc", "loss")],
      formula=f"MSE = (1/{B}) Σ_b (ŷ_z,b − y_z,b)² = {f(lt)}",
      checks=[chk("MSE (numpy) = rnn_capture.py 가 F.mse_loss 로 잰 값", lt, float(np.mean((out_tr - yz_b) ** 2)), 1e-5, kind="eq")])
    fs = cap["first_step"]
    gn = {k_: float(v.norm()) for k_, v in fs["pre_clip"].items()}; tot0 = cap["clip"][1][0]
    clip = cap["clip"]
    clipped = sum(x > 1 for v in clip.values() for x in v); allc = sum(len(v) for v in clip.values())
    ep_hi = [e for e in clip if np.mean(clip[e]) > 1]; ep_lo = [e for e in clip if np.mean(clip[e]) <= 1]      # 갱신 평균이 1 위 / 1 이하인 epoch
    un_hi = sum(x <= 1 for e in ep_hi for x in clip[e]); cl_hi = sum(x > 1 for e in ep_hi for x in clip[e]); cl_lo = sum(x > 1 for e in ep_lo for x in clip[e])
    gbi = fs["pre_clip"][f"{kind}.bias_ih_l0"].double().numpy(); gbh = fs["pre_clip"][f"{kind}.bias_hh_l0"].double().numpy()
    if kind == "lstm":
        bias_txt = "표에서 bias_ih 와 bias_hh 의 기울기 크기가 같은 것은 두 편향이 늘 합(b_ih + b_hh)으로만 쓰이기 때문입니다(4덩어리 모두 같음 — 아래 '검증')."
        bias_say = "표에서 두 편향의 기울기 크기가 같은 것은, 두 편향이 늘 더해진 합으로만 쓰이기 때문입니다."
        bias_chk = lambda: chk("기울기(자르기 전): bias_ih = bias_hh (4덩어리 모두)", gbi, gbh, 1e-7)
    else:
        d_n = float(np.max(np.abs(gbi[2 * Hn:] - gbh[2 * Hn:])))
        a_bi = fs["grads"][f"{kind}.bias_ih_l0"].double().numpy(); a_bh = fs["grads"][f"{kind}.bias_hh_l0"].double().numpy()
        d_n_after = float(np.max(np.abs(a_bi[2 * Hn:] - a_bh[2 * Hn:])))
        bias_txt = (f"표에서 bias_ih 와 bias_hh 의 기울기 크기가 다른 것은 새 후보 n 덩어리 때문입니다: b_in 은 그대로 더해지지만 b_hn 은 r 을 곱한 뒤 더해집니다(n = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn))). "
                    f"r·z 덩어리에서는 두 편향의 기울기가 같고(아래 '검증'), n 덩어리에서는 자르기 전 최대 {f(d_n)}(자른 뒤 {f(d_n_after)}) 다릅니다.")
        bias_say = "표에서 두 편향의 기울기 크기가 다른 것은 새 후보 덩어리에서 지난 상태 쪽 편향에만 재설정 문이 곱해지기 때문입니다. 재설정 문과 갱신 문 덩어리에서는 두 편향의 기울기가 같습니다."
        bias_chk = lambda: chk("기울기(자르기 전): r·z 덩어리의 bias_ih = bias_hh", gbi[:2 * Hn], gbh[:2 * Hn], 1e-7)
    L_CLIP = src.line(MODEL, "nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)")
    tot_s = f(np.sqrt(sum(v ** 2 for v in gn.values())))
    for sg in (4, 5, 6, 7):                                     # 식의 항(제곱)으로 다시 계산해도 보인 ‖∇‖ 가 나오도록 자릿수를 고른다
        terms_s = [fpos(v ** 2, sg) for v in gn.values()]
        if f(np.sqrt(sum(float(t_) for t_ in terms_s))) == tot_s:
            break
    else:
        raise AssertionError("기울기 크기 식의 자릿수를 정하지 못함")
    S("10. 손실 · 역전파 · Adam",
      f"<b>기울기 잘라내기</b>: clip_grad_norm_(…, 1.0) 은 모든 기울기를 한 벡터로 본 크기 ‖∇‖ 가 1 보다 크면 1 이 되도록 같은 비율로 줄입니다. "
      "이 화면의 기울기는 앞에서 따라온 배치(끝난 가중치)가 아니라 <b>실제 학습의 맨 첫 갱신</b>(첫 epoch 의 첫 배치, 처음 가중치)에서 기록한 것입니다. "
      f"그때 ‖∇‖ = {f(tot0)} → {'1 ÷ ' + f(tot0) + ' = ' + f(1 / tot0) + ' 배로 줄임' if tot0 > 1 else '그대로'}. "
      f"잘림은 갱신마다 따로 정해집니다: 전체 {allc}번 갱신 중 <b>{clipped}번({clipped / allc * 100:.1f}%)</b> 잘렸고, 그래프에서 평균이 1 을 넘은 epoch {len(ep_hi)}개에도 잘리지 않은 갱신이 {un_hi}번, "
      f"평균이 1 이하인 epoch {len(ep_lo)}개에도 잘린 갱신이 {cl_lo}번 있습니다({clipped} = {cl_hi} + {cl_lo}). " + bias_txt,
      say=("기울기 잘라내기입니다. 모든 가중치의 기울기를 한 줄로 이어 붙인 벡터의 크기가 1보다 크면, 크기가 1이 되도록 전체를 같은 비율로 줄입니다. "
           "표의 기울기는 따라온 배치가 아니라, 실제 학습의 맨 첫 갱신, 곧 첫 에포크의 첫 배치를 처음 가중치로 계산할 때 기록한 것입니다. "
           + (f"그때 크기는 {f(tot0)}로 1보다 커서, {f(1 / tot0)}배로 줄였습니다. " if tot0 > 1 else f"그때 크기는 {f(tot0)}로 1 이하라 그대로 두었습니다. ")
           + f"잘림 여부는 갱신마다 정해집니다. 잘린 갱신은 {clipped}번으로, 전체 갱신의 {clipped / allc * 100:.1f}%입니다. " + bias_say),
      notes={L_CLIP: (f"맨 첫 갱신: ‖∇‖ = {f(tot0)} > 1 → × {f(1 / tot0)}" if tot0 > 1 else f"맨 첫 갱신: ‖∇‖ = {f(tot0)} ≤ 1 → 그대로")},
      code=C(MODEL, [L_CLIP], L_CLIP, "Model.fit", 4, 4),
      stack=stk_fit + ["torch.nn.utils.clip_grad_norm_"], nodes=["clip"], edges=[("loss", "clip")],
      formula=f"‖∇‖ = √(Σ 모든 기울기²) = √({' + '.join(terms_s)}) = {f(np.sqrt(sum(v ** 2 for v in gn.values())))},  ∇ ← ∇ × min(1, 1/‖∇‖)",
      tensors=[{"type": "table", "title": "맨 첫 갱신(첫 epoch 의 첫 배치, 처음 가중치)의 가중치별 기울기 크기", "columns": ["가중치", "모양", "자르기 전 ‖∇‖", "자른 뒤 ‖∇‖"],
                "rows": [[k_, str(list(fs["before"][k_].shape)), num(v), num(fs["grad_norms"][k_])] for k_, v in gn.items()]},
               {"type": "line", "title": f"epoch 별 갱신 {nb}번의 ‖∇‖ 평균 — 잘림은 갱신마다 정해짐(평균이 1 위여도 안 잘린 갱신, 1 아래여도 잘린 갱신이 있음)", "x": [str(e) for e in sorted(clip)], "xlabel": "epoch",
                "series": [{"name": "평균 ‖∇‖", "color": "#3b6fb6", "y": [num(np.mean(clip[e])) for e in sorted(clip)]},
                           {"name": "1 (자르는 기준)", "color": "#b3261e", "y": [1.0] * len(clip)}], "height": 190}],
      checks=[chk("clip_grad_norm_ 이 돌려준 ‖∇‖ = 가중치별 기울기 크기의 제곱합 제곱근", tot0, np.sqrt(sum(v ** 2 for v in gn.values())), 1e-4, kind="eq"),
              chk("자른 뒤 ‖∇‖ = min(1, ‖∇‖)", np.sqrt(sum(v ** 2 for v in fs["grad_norms"].values())), min(1.0, tot0), 1e-4), bias_chk()])
    gb, b0, b1 = float(fs["pre_clip"]["fc.bias"][0]), float(fs["before"]["fc.bias"][0]), float(fs["after"]["fc.bias"][0])
    cf = min(1.0, 1 / tot0); g_eff = gb * cf + fs["wd"] * b0
    step_hand = b0 - fs["lr"] * g_eff / (abs(g_eff) + fs["eps"])
    L_OPT = src.line(MODEL, "opt = torch.optim.Adam(self.net.parameters(), lr=p[\"lr\"], weight_decay=p[\"weight_decay\"])")
    for sg in (4, 5, 6, 7):                                     # 보인 숫자로 다시 계산해도 보인 ∇θ′ 가 나오도록 자릿수를 고른다
        gb_s, cf_s = f"{gb:.{sg}g}", f"{cf:.{sg}g}"
        if g4(float(gb_s) * float(cf_s) + fs["wd"] * float(g5(b0))) == g4(g_eff):
            break
    else:
        raise AssertionError("Adam 식의 자릿수를 정하지 못함")
    S("10. 손실 · 역전파 · Adam",
      f"<b>Adam</b>(weight_decay {fs['wd']}): 잘라낸 기울기 ∇θ 에 wd × 가중치를 더한 뒤(L2), 첫 걸음은 거의 '학습률 × 부호' 만큼 움직입니다. "
      f"예: 맨 첫 갱신(처음 가중치)에서 fc.bias(Linear 의 편향)는 {g5(b0)} → <b>{g5(b1)}</b>. 학습이 끝난 값은 「9. 마지막 시점 · 드롭아웃 · Linear」의 b_fc = {f(b_fc)} 입니다.",
      say=(f"아담이 잘라낸 기울기로 가중치를 고칩니다. 이 설정은 가중치 감쇠 {fs['wd']}를 쓰는데, 파이토치의 아담에서는 기울기에 감쇠 곱하기 가중치를 더하는 방식입니다. "
           f"첫 걸음에서는 편향 보정 뒤 기울기 평균이 기울기 그 자체, 기울기 제곱 평균이 기울기의 제곱이 되므로, 가중치가 기울기 반대 방향으로 거의 학습률, 곧 {fs['lr']}만큼 움직입니다. "
           f"예를 들어 맨 첫 갱신에서 선형 층 편향은 처음 값 {g5(b0)}에서 {g5(b1)}로 바뀌었습니다. 학습이 끝난 값은 앞에서 본 {f(b_fc)}입니다."),
      notes={L_OPT: f"Adam(lr = {fs['lr']}, weight_decay = {fs['wd']}) — 기울기에 wd·θ 를 더하는 L2 방식",
             src.line(MODEL, "opt.step()"): f"맨 첫 갱신: fc.bias {g5(b0)} → {g5(b1)}"},
      lab=lab_adam(gb, tot0, b0, fs["lr"], fs["wd"], fs["eps"], fs["betas"], float(fs["grads"]["fc.bias"][0]) / gb, b1),
      code=C(MODEL, [L_OPT, src.line(MODEL, "opt.step()")], src.line(MODEL, "opt.step()"), "Model.fit", 4, 3),
      stack=stk_fit + ["torch.optim.Adam.step"], nodes=["adam"], edges=[("clip", "adam")],
      formula=(f"∇θ′ = ∇θ × {cf_s} + {fs['wd']} × θ = {pn(gb_s)} × {cf_s} + {fs['wd']} × {pn(g5(b0))} = {g4(g_eff)}\n"
               f"m̂ = ∇θ′, v̂ = ∇θ′²  →  θ ← θ − lr · m̂ / (√v̂ + ε) ≈ θ − lr × sign(∇θ′) = {pn(g5(b0))} − {fs['lr']} × {pn(f(np.sign(g_eff), 0))} = {g5(step_hand)}   (θ = fc.bias, ε = {fs['eps']:g})"),
      checks=[chk("자른 기울기 = 자르기 전 × 배율", float(fs["grads"]["fc.bias"][0]), gb * cf, 1e-6), chk("Adam 첫 걸음 직접 계산 = 기록한 fc.bias", b1, step_hand, 1e-6)],
      hint="torch.optim.Adam 의 weight_decay 는 AdamW 와 달리 기울기에 wd·θ 를 더하는 L2 방식입니다(그 뒤 Adam 의 크기 조정을 함께 받음). 이 화면의 θ·∇θ 는 맨 첫 갱신의 값입니다.")
    hist = cap["history"]
    L_HIST = src.line(MODEL, "self.history.append({\"epoch\": ep + 1, \"train_loss\": tl / max(tn, 1), \"val_loss\": None})")
    S("10. 손실 · 역전파 · Adam",
      f"이것을 epoch 마다 {nb}번, {len(hist)} epoch 반복했습니다. 학습 손실(epoch 평균)은 {f(hist[0]['train_loss'])} → {f(hist[-1]['train_loss'])}. "
      + ("검증이 없어 조기종료 없이 <b>마지막 epoch 의 가중치</b>를 씁니다." if n_va == 0 else f"검증 손실이 가장 낮았던 epoch {cap['best_epoch']} 의 가중치를 씁니다."),
      say=(f"이것을 에포크마다 {nb}번씩, {len(hist)} 에포크 동안 반복했습니다. 에포크 평균 학습 손실은 {f(hist[0]['train_loss'])}에서 {f(hist[-1]['train_loss'])}로 줄었습니다. "
           + ("검증이 없으므로 조기 종료 없이 마지막 에포크의 가중치를 씁니다." if n_va == 0 else f"검증 손실이 가장 낮았던 에포크 {cap['best_epoch']}의 가중치를 씁니다.")),
      notes={src.line(MODEL, "tl += float(loss.detach()) * len(idx); tn += len(idx)"): "배치 손실 × 배치 크기를 더해 둠 (드롭아웃 켠 상태, 배치마다 가중치가 바뀌는 중에 잼)",
             L_HIST: "epoch 평균 학습 손실 = 합 ÷ 창 수 — 검증이 없을 때의 기록"},
      code=C(MODEL, src.span(MODEL, "            if has_val:", "self.history.append({\"epoch\": ep + 1, \"train_loss\": tl / max(tn, 1), \"val_loss\": None})", after=src.line(MODEL, "opt.step()")),
             L_HIST, "Model.fit", 2, 6),
      stack=stk_fit, nodes=["epoch"], edges=[("adam", "epoch")],
      tensors=[{"type": "line", "title": f"실제 학습 기록 — {len(hist)} epoch (표준화 단위 MSE, epoch 평균)", "x": [str(h["epoch"]) for h in hist], "xlabel": "epoch",
                "series": [{"name": "학습 손실", "color": "#3b6fb6", "y": [num(h["train_loss"]) for h in hist], "dots": True}]
                + ([{"name": "검증 손실", "color": "#c8710f", "y": [num(h["val_loss"]) for h in hist], "dots": True}] if n_va else []), "height": 230}])
    # 학습 창 적합 정도(예측 모드, 고른 가중치)
    net = MM._Net(K, Hn, p["layers"], p["dropout"]); net.load_state_dict(sdt); net.eval()
    with torch.no_grad():
        fit_z = net(torch.tensor(Xw[:n_tr], dtype=torch.float32)).double().numpy()
    fit = fit_z * y_sd + y_mu
    r2 = 1 - np.sum((ytr - fit) ** 2) / np.sum((ytr - ytr.mean()) ** 2)
    no_res = "" if has_result else f" {KN} 는 아직 이 설정의 워크포워드 결과 파일이 없습니다(「11. 예측」)."
    # 비교 기준 naive: 결정월 t 의 이번 달 변화율 y_(t−1)(= F11_t ÷ F11_(t−1) − 1)을 다음 달 예측으로 그대로 씀 — engine/calendar.py naive_forecast(y.shift(k))
    naive = cap["y_train"].values[L - 2:L - 2 + n_tr]
    r2n = 1 - np.sum((ytr - naive) ** 2) / np.sum((ytr - ytr.mean()) ** 2)
    S("10. 손실 · 역전파 · Adam",
      f"학습이 끝난 가중치로 학습 창 {n_tr}개를 다시 예측해 보면(예측 모드, 직접 계산) 결정계수 R² = <b>{f(r2, 3)}</b> 입니다. "
      + ("학습 창에는 가깝게 맞지만, " if r2 > 0.5 else "학습 창에도 잘 맞지 않습니다. 어느 쪽이든 ")
      + f"이웃한 창끼리 {L - 1}개월씩 겹치는 학습 창 {n_tr}개에 가중치 {cap['n_params']:,}개를 맞춘 것이라, 학습 창의 R² 는 예측력의 근거가 되지 않습니다. "
      f"비교로, 이번 달 변화율을 다음 달 예측으로 그대로 쓰는 naive(아무것도 학습하지 않음)도 같은 학습 창 {n_tr}개에서 R² = {f(r2n, 3)} 입니다. "
      "예측력은 워크포워드 테스트 결과로만 판단합니다." + no_res,
      say=(f"학습이 끝난 가중치로 학습 창 {n_tr}개를 예측 모드로 다시 계산하면, 결정계수 R²는 {f(r2, 3)}입니다. "
           + ("학습 창에는 가깝게 맞습니다. 하지만 " if r2 > 0.5 else "학습 창에도 잘 맞지 않습니다. 어느 쪽이든 ")
           + f"서로 {L - 1}개월씩 겹치는 학습 창 {n_tr}개에 가중치 {cap['n_params']:,}개를 맞춘 것이라, 이 값은 예측력의 근거가 되지 않습니다. "
           f"비교하면, 이번 달 변화율을 그대로 다음 달 예측으로 쓰는, 아무것도 학습하지 않는 방법도 같은 창에서 R²가 {f(r2n, 3)}입니다. "
           "예측력은 워크포워드 테스트 결과로만 판단합니다." + ("" if has_result else f" {KN}는 아직 이 설정의 워크포워드 결과 파일이 없습니다.")),
      notes={src.line(MODEL, "out = self.net(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()"): "예측 모드(fit 끝의 net.eval() — 드롭아웃 없음), 기울기 계산 없음",
             src.line(MODEL, "return out * self.y_sd + self.y_mu"): "표준화 되돌리기: × σ_y + μ_y"},
      code=C(MODEL, src.span(MODEL, "def predict(self, X):", "return 1.0 / (1.0 + np.exp(-out))"), src.line(MODEL, "out = self.net(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()"), "Model.predict", 2, 2),
      stack=["tools/codewalk/rnn_script.py · 학습 창 다시 예측(직접 계산)"], nodes=["epoch"],
      formula=f"R² = 1 − Σ(y − 모델)² ÷ Σ(y − ȳ)²   (학습 창, 원래 단위)   모델 {f(r2, 3)},  naive(ŷ_t = y_(t−1)) {f(r2n, 3)}",
      checks=[chk("naive 예측 y_(t−1) (ds.y 를 한 칸 민 것) = 결정월 행의 입력 A_ret_1 (ds.X, 원래 단위)", naive, raw["A_ret_1"].values[L - 1:L - 1 + n_tr], 1e-15)] if "A_ret_1" in feats else [],
      tensors=[{"type": "line", "title": "학습 창: 실제 정답 vs 학습된 모델의 값", "x": [w[:7] for w in win_idx[:n_tr]], "xlabel": "결정월",
                "series": [{"name": "정답 y", "color": "#3b6fb6", "y": [num(v) for v in ytr], "dots": True}, {"name": "모델", "color": "#c8710f", "y": [num(v) for v in fit], "dots": True},
                           {"name": "naive (이번 달 변화율 그대로)", "color": "#8a93a3", "y": [num(v) for v in naive]}], "height": 230}])

    # ---------------------------------------------------------------- 11. 예측
    pr = cap["predict"]
    z = (pr["pred"] - y_mu) / y_sd
    wd_ = [d[:7] for d in pr["window_dates"]]
    n_ov = L - 1 - emb                                          # 테스트 창 중 학습 폴드에 있는 달 수
    emb_months = wd_[n_ov:L - 1]                                # 비움 달(학습 폴드에는 없고 테스트 창에는 있음)
    nxt = pr["level"] * (1 + pr["pred"])
    L_CTX = src.line(EXP, "ctx = ds.X.loc[: f.test[0]].iloc[-cfg.seq_len:-1]")
    lab6 = lab_forward(kind, pr["window"], sd, W_fc, b_fc, y_mu, y_sd, pr["level"], pr["date"][:7], z, pr["pred"], nxt)
    pred_s = f(pr["pred"])
    for sg in (4, 5, 6, 7, 8):                                  # 식의 숫자로 다시 계산해도 표시한 ŷ 가 나오도록 자릿수를 고른다
        zs, ss, ms = f"{z:.{sg}g}", f"{y_sd:.{sg}g}", f"{y_mu:.{sg}g}"
        if f(float(zs) * float(ss) + float(ms)) == pred_s:
            break
    emb_note = f"비운 달 {', '.join(emb_months)} 행의 입력은 예측 시점(F11({test_m}) 공개 뒤)에 이미 알려진 값이라 누출이 아닙니다. "
    sealed = cap["config"]["validation"].get("sealed_from")
    t_next = lev.index[i_test + cap["k"]]                     # 정답이 쓰는 다음 달
    seal_hit = bool(sealed) and t_next >= pd.Timestamp(sealed)
    act_raw = float(f11r.loc[t_next] / f11r.loc[t_test] - 1)
    seal_txt = (f" 이 정답은 봉인 구간({pd.Timestamp(sealed):%Y-%m}~)의 F11({t_next:%Y-%m})을 씁니다 — 엔진은 결정일만 봉인 전으로 자릅니다." if seal_hit else "")
    S("11. 예측",
      f"테스트 결정월 {pr['date'][:7]} 의 창은 <b>테스트 달 바로 앞 {L - 1}개월</b>({wd_[0]} ~ {wd_[L - 2]})에 테스트 달을 붙여 만듭니다(ds.X 에서 직접 — 학습 폴드의 끝 {n_ov}개월과 "
      f"학습에서 비운 {', '.join(emb_months)}). " + emb_note +
      f"눈금은 학습 폴드의 평균·표준편차를 그대로 씁니다. predict 는 예측 모드(드롭아웃 없음)로 ŷ_z = {f(z)} 를 내고 ŷ = ŷ_z × σ_y + μ_y = <b>{pred_s}</b> ({pr['pred'] * 100:.4f}%) 로 되돌립니다. "
      f"결정월의 F11 {f(pr['level'])} → 다음 달 예측 {f(nxt)}. 실제 변화율은 {f(pr['actual'])} ({pr['actual'] * 100:.4f}%) 였습니다." + seal_txt,
      say=(f"테스트 결정월 {pr['date'][:7]}의 창은, 테스트 달 바로 앞 {L - 1}개월에 테스트 달을 붙여 만듭니다. "
           f"그 {L - 1}개월에는 학습에서 비워 둔 {', '.join(emb_months)}도 들어갑니다. 그 달의 입력은 예측 시점, 곧 결정월 F11이 공개된 뒤에 이미 알려진 값이라 누출이 아닙니다. "
           + f"눈금은 학습 폴드의 평균과 표준편차를 그대로 씁니다. 예측 모드로 넣은 신경망 출력은 {f(z)}로 나오고, 표준편차를 곱하고 평균을 더하면 {pr['pred'] * 100:.4f}%입니다. "
           f"결정월의 F11 값 {f(pr['level'])}에서 다음 달 예측은 {f(nxt)}로 나옵니다. 실제 변화율은 {pr['actual'] * 100:.4f}%였습니다."
           + (f" 이 실제 값은 봉인 구간인 {t_next:%Y-%m}의 F11을 씁니다." if seal_hit else "")),
      notes={L_CTX: f"테스트 달 바로 앞 {L - 1}행: {wd_[0]} ~ {wd_[L - 2]} (비움 달 {', '.join(emb_months)} 포함)",
             src.line(EXP, "Xtf = scaler.transform(te_full[ds.feature_names].ffill().values)"): "학습 폴드의 눈금(평균·표준편차) 그대로",
             src.line(EXP, "p = m.predict(Xtw)"): f"예측 → {pred_s}"},
      lab=lab6,
      code=C(EXP, src.lines(EXP, "ctx = ds.X.loc[: f.test[0]].iloc[-cfg.seq_len:-1]", "Xtf = scaler.transform(te_full[ds.feature_names].ffill().values)", "p = m.predict(Xtw)"),
             L_CTX, "_walkforward_predict", 2, 4),
      stack=["engine/experiment.py · _walkforward_predict", f"{MODEL} · Model.predict"], nodes=["pred"], edges=[("epoch", "pred")],
      formula=f"ŷ = ŷ_z × σ_y + μ_y = {pn(zs)} × {ss} + {ms} = {pred_s}",
      tensors=[H(f"결정월 {pr['date'][:7]} 의 테스트 창 (표준화 값)", pr["window"], rows=[f"τ{t + 1} {d}" for t, d in enumerate(wd_)], cols=feats, shape=[L, K], digits=2, cellH=19)],
      checks=[chk(f"테스트 창의 앞 {n_ov}행 = 학습 폴드 끝 {n_ov}행(같은 눈금)", pr["window"][:n_ov], Xw[-1][L - n_ov:], 1e-12),
              chk(f"같은 창을 {len(pr['repeats'])}번 예측해도 같은 값(예측 모드는 무작위 없음)", np.std(pr["repeats"]), 0.0, 1e-12, kind="rerun"),
              chk("기록 도구가 끝난 가중치로 다시 낸 예측 = 엔진(_walkforward_predict)이 낸 예측", pr["repeats"][0], pr["pred"], 1e-6, kind="rerun"),
              chk(f"실제 변화율 = 원자료(data 저장소) F11({t_next:%Y-%m}) ÷ F11({t_test:%Y-%m}) − 1", pr["actual"], act_raw, 1e-15, kind=kd_raw)])
    nrep = cap.get("n_replayed", 0)
    ORD_SAY = ("첫", "두", "세", "네", "다섯")

    def res_diffs(jo, jr):
        """결과 파일 jo 가 대조한 파일 jr 와 설정·실행 조건이 다른 점 — (화면 글 목록, 음성 낱말 목록). 결과 json 에서만 읽는다."""
        out, sy = [], []
        po, pr_ = (jo.get("model") or {}).get("params", {}), (jr.get("model") or {}).get("params", {})
        for k_ in sorted(set(po) | set(pr_)):
            if po.get(k_) != pr_.get(k_):
                out.append(f"{k_} {po.get(k_)}(대조 파일 {pr_.get(k_)})"); sy.append("배치 크기" if k_ == "batch_size" else "모델 설정")
        lo, lr_ = jo.get("lag_days_used") or {}, jr.get("lag_days_used") or {}
        dl = [v for v in sorted(set(lo) | set(lr_)) if lo.get(v) != lr_.get(v)]
        if dl:
            out.append("발표시차 " + ", ".join(f"{v} {lo.get(v)}일({lr_.get(v)}일)" for v in dl)); sy.append("발표시차")
        if jo.get("device") != jr.get("device"):
            out.append(f"장치 {jo.get('device')}({jr.get('device')})"); sy.append("실행 장치")
        if jo.get("engine_rev") != jr.get("engine_rev"):
            out.append(f"engine_rev {jo.get('engine_rev') or '기록 없음'}({jr.get('engine_rev')})"); sy.append("엔진 판")
        return out, list(dict.fromkeys(sy))
    if has_result:
        same = mr["max_diff"] < 1e-12
        jref = res_json(mr["file"])
        rows_r, oth = [], []
        for a_, n_, d_ in mr["all"]:
            jo = res_json(a_); dd, sy = res_diffs(jo, jref) if a_ != mr["file"] else ([], [])
            rows_r.append([a_.replace("_daily.parquet", ""), (jo.get("model") or {}).get("file") or "?", f"{d_:.1e}", "대조한 파일" if a_ == mr["file"] else ("; ".join(dd) or "차이 못 찾음")])
            if a_ != mr["file"]:
                oth.append(sy)
        # 장치가 다른 결과 파일: 결과 parquet 에서 이 달 예측과 겹친 폴드 전체의 차이를 직접 계산
        def preds(fn):
            return pd.read_parquet(ROOT / "results" / "forecast" / fn, columns=["prediction"])["prediction"].dropna()
        p_ref = preds(mr["file"]); dev_info = []
        for a_, n_, d_ in mr["all"]:
            jo = res_json(a_)
            if a_ != mr["file"] and jo.get("device") and jo.get("device") != jref.get("device"):
                p_o = preds(a_); c_ = p_o.index.intersection(p_ref.index); dd_ = (p_o.loc[c_] - p_ref.loc[c_]).abs()
                dev_info.append({"name": a_.replace("_daily.parquet", ""), "dev": jo["device"], "rev": jo.get("engine_rev"), "this": float(p_o.loc[t_test]), "ref": float(p_ref.loc[t_test]),
                                 "tab": d_, "n": len(c_), "mean": float(dd_.mean()), "max": float(dd_.max()), "sd": float(p_ref.loc[c_].std())})
        dev_txt = "".join(
            f"장치가 다른 결과 {d['name']}({d['dev']})는 이 달 예측이 {fpos(d['this'])} 로 대조 파일 {fpos(d['ref'])} 와 " + ("부호까지 " if np.sign(d["this"]) != np.sign(d["ref"]) else "")
            + f"다르고, 겹친 {d['n']}개 예측의 평균 |차이| {d['mean']:.1e}, 최대 {d['max']:.1e} 입니다(대조 파일 예측의 표준편차 {d['sd']:.1e}). "
            + ("이 파일은 엔진 판(engine_rev 기록 없음)도 달라 원인을 나눌 수 없지만, " if d["rev"] != jref.get("engine_rev") else "")
            + "GPU 에서는 배치 섞기(torch.randperm(…, device=dev))와 드롭아웃이 GPU 의 난수 생성기를 써서 학습 경로 자체가 바뀐 것으로 보입니다. "
            for d in dev_info)
        oth_say = ""
        if oth and same:
            parts = [f"{ORD_SAY[i]} 번째는 " + ", ".join(sy_[:-1] + [attach(sy_[-1], "이")]) if sy_ else f"{ORD_SAY[i]} 번째는 차이를 찾지 못했고" for i, sy_ in enumerate(oth[:len(ORD_SAY)])]
            oth_say = (f"표의 다른 결과 파일 {len(oth)}개는 대조한 파일과 설정이나 실행 조건이 다릅니다. " + ", ".join(p_ + (" 다르고" if i < len(parts) - 1 else " 다릅니다") for i, p_ in enumerate(parts))
                       + ". 그래서 값이 다른 것이 정상입니다. ")
        S("11. 예측",
          (f"이 예측을 결과 폴더의 같은 실험 결과 <b>{mr['file'].replace('_daily.parquet', '')}</b> 의 {pr['date'][:7]} 예측과 맞춰 봅니다: 차이 {mr['max_diff']:.1e}"
           + (" — <b>결과 화면의 숫자와 같은 계산</b>입니다." if same else " — 다릅니다.")
           + (" 왼쪽 표의 다른 결과 파일은 대조한 파일과 설정이나 실행 조건이 달라(표의 마지막 열, 결과 json 에서 읽음) 값이 다른 것이 정상입니다." if oth and same else "")
           + f" 엔진은 시드를 실험 시작에 한 번만 고정하므로, 이 영상은 앞의 폴드 {nrep}개를 그대로 먼저 돌려(기록 없이) 난수 상태를 실제 실험과 같게 맞춘 뒤 마지막 폴드를 기록했습니다."),
          say=(f"이 예측을 결과 폴더에 저장된 같은 실험의 결과 파일과 맞춰 봅니다. 결정월 {pr['date'][:7]}의 예측 "
               + ("차이는 0으로, 결과 화면의 숫자와 같은 계산입니다. " if same else f"차이는 {mr['max_diff']:.1e}로 다릅니다. ")
               + oth_say
               + f"엔진은 실험을 시작할 때 시드를 한 번만 고정합니다. 그래서 이 영상은 앞의 폴드 {nrep}개를 기록 없이 먼저 돌려 난수 상태를 실제 실험과 같게 맞춘 뒤, 마지막 폴드를 기록했습니다."),
          stack=["results/forecast/" + mr["file"]], nodes=["pred"],
          tensors=[{"type": "table", "title": "결과 파일별 비교 (같은 결정월의 예측, 설정·실행 조건은 결과 json 에서)", "columns": ["결과 파일", "모델", f"{test_m} 예측 차이", "대조한 파일과 다른 점(괄호 = 대조 파일)"], "rows": rows_r}],
          hint=(dev_txt + "CPU 스레드 수만 다를 때는 덧셈 순서 차이로 끝자리가 다를 수 있습니다(이 설정에서는 재 보지 않음). "
                f"이 기록은 CPU 스레드 {cap.get('torch_threads')}개로 만들었고, 대조한 결과 파일은 장치 '{jref.get('device')}' 로 만들었습니다(결과 파일에는 스레드 수가 기록되지 않음). "
                f"예측이 차이 {mr['max_diff']:.1e} 로 같아 같은 조건으로 판단합니다."),
          checks=[chk(f"영상의 예측 = 결과 파일 {mr['file'].replace('_daily.parquet', '')} 의 예측", mr["max_diff"], 0.0, 1e-12, kind="rerun")] if same else [])
    else:
        mr_rows = [(a_.replace("_daily.parquet", ""), (res_json(a_).get("model") or {}).get("file")) for a_, n_, d_ in mr["all"]]
        others = sorted({m_ or "?" for _, m_ in mr_rows})
        S("11. 예측",
          (f"결과 폴더에는 이 설정 이름({cap['config']['name']})의 결과 파일이 {len(mr_rows)}개 있지만 모델이 모두 {', '.join(o.upper() for o in others)} 입니다(왼쪽 표). " if mr_rows else
           f"결과 폴더에 이 설정 이름({cap['config']['name']})의 결과 파일이 없습니다. ")
          + f"<b>{KN} 로 돌린 결과 파일은 없어서</b>, 이 영상의 {KN} 예측을 결과 화면의 숫자와 맞춰 보지는 못했습니다. "
          f"난수 상태는 LSTM 영상과 같은 방법(앞의 폴드 {nrep}개를 기록 없이 먼저 돌림)을 썼지만, 비교할 결과 파일이 없습니다.",
          say=((f"결과 폴더에는 이 설정을 LSTM으로 돌린 결과 파일만 있고, {KN}로 돌린 결과 파일은 없습니다. " if mr_rows and others == ["lstm"] else f"결과 폴더에 {KN}로 돌린 결과 파일이 없습니다. ")
               + "그래서 이 영상의 예측을 결과 화면의 숫자와 맞춰 보지는 못했습니다. "
               f"난수 상태는 LSTM 영상과 같은 방법, 곧 앞의 폴드 {nrep}개를 기록 없이 먼저 돌리는 방법을 썼지만, 비교할 결과 파일이 없습니다."),
          stack=["results/forecast"], nodes=["pred"],
          tensors=([{"type": "table", "title": "결과 폴더의 같은 설정 이름 결과 파일 (모델은 결과 json 의 model.file)", "columns": ["결과 파일", "모델"], "rows": [[a_, m_ or "?"] for a_, m_ in mr_rows]}] if mr_rows else []))
    n_ok = sum(c["ok"] for c in checks_all)
    n_lab = sum(1 for c in checks_all if c["what"].startswith("직접 실행해 보기"))
    n_k = {k_: sum(1 for c in checks_all if c["kind"] == k_) for k_ in KIND}
    kinds_txt = f"독립 대조 {n_k['ind']}줄(직접 실행해 보기 {n_lab}줄 포함) · 식 확인 {n_k['eq']}줄 · 재실행 {n_k['rerun']}줄"
    all_ok = n_ok == len(checks_all)
    S("11. 예측",
      f"정리: 엔진이 학습 폴드로 눈금을 맞추고 {L}개월 창을 만들면, {KN} {J('이')} {L}개 시점을 차례로 읽어(같은 가중치, 상태를 이어받으며) 마지막 은닉 상태만 Linear 로 보내 값 하나를 냅니다. "
      + ("" if kind == "lstm" else f"GRU 는 LSTM 보다 덩어리가 하나 적어 가중치가 {cap['n_params']:,}개이고, 셀 상태가 없습니다. ")
      + f"학습은 MSE · 기울기 잘라내기(1) · Adam(L2 {p['weight_decay']}) 이고, 이 설정의 폴드는 입력 창이 {len(Xw)}개뿐이라 검증 창·조기종료 없이 {p['epochs']} epoch 를 돕니다. "
      f"검증 {len(checks_all)}줄 — {kinds_txt}. 독립 대조는 다른 경로(원자료·기록·다른 계산법·순수 파이썬 실습)로 맞춰 본 것, 식 확인은 같은 배열에 같은 식을 다시 계산한 것, "
      "재실행은 같은 코드를 다시 돌린 값" + ("(결과 파일 포함)" if has_result else "") + "과의 비교입니다. " + ("모두 허용 오차 안이었습니다." if all_ok else f"{n_ok}줄이 허용 오차 안이었습니다."),
      say=(f"정리합니다. 엔진이 학습 폴드로 눈금을 맞추고 {L}개월 창을 만들면, {KN}{J('이')} {L}개 시점을 차례로 읽습니다. 가중치는 시점마다 같고, 상태를 이어받습니다. "
           "마지막 은닉 상태만 선형 층으로 보내 값 하나를 냅니다. "
           + ("" if kind == "lstm" else "GRU는 LSTM보다 덩어리가 하나 적고, 셀 상태가 없습니다. ")
           + f"학습은 평균제곱오차, 기울기 잘라내기, 아담이고, 이 설정의 폴드는 입력 창이 {len(Xw)}개뿐이라 검증 창과 조기 종료 없이 {p['epochs']} 에포크를 돕니다. "
           + f"이 영상의 검증 {len(checks_all)}줄 가운데, 다른 경로로 맞춰 본 독립 대조가 {n_k['ind']}줄, 같은 식을 다시 계산한 식 확인이 {n_k['eq']}줄, 다시 돌린 값과의 비교가 {n_k['rerun']}줄이고, "
           + ("모두 허용 오차 안이었습니다." if all_ok else f"{n_ok}줄이 허용 오차 안이었습니다.")),
      stack=["요약"], nodes=["data", "scale", "win", "split", "batch", "x", *cell_nodes, "last", "drop", "fc", "loss", "clip", "adam", "epoch", "pred"],
      tensors=[{"type": "table", "title": f"검증 전체 {len(checks_all)}줄 — {kinds_txt}", "columns": ["장", "종류", "무엇", "결과"],
                "rows": [[steps[c["step"] - 1]["chapter"] if c["step"] - 1 < len(steps) else "", KIND[c["kind"]], c["base"], "✓ 허용 오차 안" if c["ok"] else "✗ 벗어남"] for c in checks_all]}])
    if n_ok != len(checks_all):
        raise AssertionError(f"검증 실패: {[c for c in checks_all if not c['ok']]}")
    used = sorted({s_["code"]["file"] for s_ in steps if s_.get("code")})
    return {"title": f"{KN} 코드 따라가기 — 코드가 실제로 실행되는 순서", "model": kind, "steps": steps, "sources": {fn: src.text[fn] for fn in used},
            "diagram": diagram(kind, K, L, Hn, p["dropout"]), "checks": checks_all}


def diagram(kind, K, L, Hn, drop) -> dict:
    KN = kind.upper()
    N = lambda id_, label, x, y, w, h, sub=None: {"id": id_, "label": label, "x": x, "y": y, "w": w, "h": h, "sub": sub}
    nodes = [N("data", "원자료 표", 10, 8, 140, 42, f"월 × 변수 {K}"), N("scale", "눈금 표준화", 172, 8, 140, 42, "학습 폴드로"),
             N("win", "창 만들기", 334, 8, 140, 42, f"[N, {L}, {K}]"), N("split", "검증 나누기", 496, 8, 150, 42, "_split_fit"),
             N("batch", "배치 randperm", 668, 8, 160, 42, "섞어서 나눔"),
             N("loss", "MSE 손실", 1130, 8, 140, 42, "loss.backward()"), N("clip", "기울기 자르기", 1292, 8, 140, 42, "‖∇‖ ≤ 1"),
             N("adam", "Adam", 1454, 8, 120, 42, "lr · L2"), N("epoch", "epoch", 1596, 8, 110, 42, "반복"), N("pred", "예측", 1728, 8, 140, 42, "ŷ = s·σ_y + μ_y"),
             N("x", "입력 창 x", 24, 104, 110, 60, f"[B, {L}, {K}]")]
    x0, w, gap = 156, 80, 8
    for t in range(L):
        nodes.append(N(f"c{t + 1}", f"τ{t + 1}", x0 + t * (w + gap), 104, w, 60, f"{KN} 셀"))
    xe = x0 + L * (w + gap)
    nodes += [N("last", "마지막 시점", xe + 10, 104, 128, 60, f"out[:, −1, :] [B, {Hn}]"), N("drop", "Dropout", xe + 160, 104, 110, 60, f"p = {drop} (학습 때)"),
              N("fc", "Linear", xe + 292, 104, 120, 60, f"{Hn} → 1")]
    E = [{"from": a, "to": b} for a, b in [("data", "scale"), ("scale", "win"), ("win", "split"), ("split", "batch"), ("loss", "clip"), ("clip", "adam"), ("adam", "epoch"), ("epoch", "pred"),
                                           ("x", "c1"), *[(f"c{t + 1}", f"c{t + 2}") for t in range(L - 1)], (f"c{L}", "last"), ("last", "drop"), ("drop", "fc")]]
    fc_cx = xe + 292 + 60
    E += [{"from": "batch", "to": "x", "path": [[748, 50], [748, 66], [79, 66], [79, 104]]},
          {"from": "fc", "to": "loss", "path": [[fc_cx, 104], [fc_cx, 66], [1200, 66], [1200, 50]]}]
    return {"width": 1880, "height": 186, "nodes": nodes, "edges": E,
            "groups": [{"x": 8, "y": 78, "w": xe + 430, "h": 100, "label": f"_Net.forward (models/{kind}.py) — 같은 셀 가중치를 시점마다 다시 씀"}]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--capture")
    ap.add_argument("--out")
    a = ap.parse_args()
    cap = pickle.load(open(a.capture or ROOT / f"docs/guide_build/walk/{a.model}_capture.pkl", "rb"))
    sc = build(cap)
    out = build_html(sc, Path(a.out) if a.out else ROOT / f"docs/guide_build/walk/{a.model}.html")
    print("저장:", out, "· 단계", len(sc["steps"]), "· 검증", sum(c["ok"] for c in sc["checks"]), "/", len(sc["checks"]), "· 크기", round(out.stat().st_size / 1e6, 2), "MB")
