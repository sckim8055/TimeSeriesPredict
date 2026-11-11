"""BIVA 코드 따라가기 대본 — biva_capture 가 붙잡은 실제 실행(원본 코드 줄·텐서·학습 기록)을 단계별 설명으로 바꾼다.

  python -m tools.codewalk.biva_script [--capture docs/guide_build/walk/biva_capture.pkl] [--out docs/guide_build/walk/biva.html]

원칙
- 화면의 숫자는 전부 실행 기록에서 꺼낸다. 손으로 다시 계산한 값은 '검증' 줄로만 쓰고(일치 여부 표시), 다르면 대본 생성이 멈춘다.
- 코드 줄 번호는 원본 파일에서 글자로 찾는다(Sources.line). 원본이 바뀌어 줄을 못 찾으면 멈춘다 → 대본이 낡지 않는다.
- 설명은 코드가 실제로 하는 일만 말한다. 원본 코드의 특징(쓰지 않는 계산, 독특한 규칙)은 '힌트'에 그대로 적는다.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch

from tools.codewalk.common import ROOT, Sources, Trace, arr, build_html, f, match_result, num, py_list, run_lab

RITS, BRITS, BIVA, VAE, TOOLS = "models/biva_src/BIVA_RITS.py", "models/biva_src/BIVA_BRITS.py", "models/biva_src/BIVA.py", "models/biva_src/BIVA_LSTM_VAE.py", "models/biva_src/tools_orig.py"
WRAP, EXP, CAL = "models/biva.py", "engine/experiment.py", "engine/calendar.py"
SIG = lambda a: 1 / (1 + np.exp(-a))


def lstm_np(x, sd, pre):
    """PyTorch nn.LSTM 한 층(배치 1개, 시점 순서대로) 64비트 손 계산 — 게이트까지 돌려준다."""
    Wih, Whh = sd[pre + "weight_ih_l0"].double().numpy(), sd[pre + "weight_hh_l0"].double().numpy()
    bih, bhh = sd[pre + "bias_ih_l0"].double().numpy(), sd[pre + "bias_hh_l0"].double().numpy()
    H = Whh.shape[1]; h = np.zeros(H); c = np.zeros(H); outs, gates = [], []
    for t in range(len(x)):
        g = Wih @ x[t] + bih + Whh @ h + bhh
        i_, f_, g_, o_ = SIG(g[:H]), SIG(g[H:2 * H]), np.tanh(g[2 * H:3 * H]), SIG(g[3 * H:])
        c = f_ * c + i_ * g_; h = o_ * np.tanh(c)
        outs.append(h.copy()); gates.append((i_, f_, g_, o_, c.copy(), h.copy()))
    return np.array(outs), gates


# ---------------------------------------------------------------- 직접 실행해 보기(실험 코드) — 2026-11-08
# 실행 기록의 숫자를 그대로 넣은, 혼자 도는 파이썬 코드. 대본을 만들 때 실제로 실행해 출력이 엔진 값과 같은지 확인한다(common.run_lab).
# 순수 파이썬(math 만) — Python Tutor 에서 한 줄씩 볼 수 있다. 큰 가중치 목록은 유효숫자 12자리로 넣는다(출력은 5자리라 결과가 같다).
F32 = 2e-5   # 엔진(신경망)은 32비트 실수로 계산 → 실험 코드(64비트)와 6번째 자리가 1 정도 다를 수 있어 상대오차 2e-5 까지 같은 값으로 본다


def g4(v) -> str:
    """유효숫자 4자리(작은 값 두 개를 견줄 때 소수 넷째 자리에서 같아 보이지 않도록)."""
    return f"{float(v):#.4g}"


def _pl(vals, sig=17):
    return py_list([float(v) for v in vals], sig)


def _g6(v):
    return f"{float(v):.5g}"


def lab_minmax(col, var, last_date, smin_e, smax_e, scaled_e):
    code = (f"# {var} 의 학습 폴드 값 {len(col)}개 (월별, 실행 기록 그대로)\n"
            f"x = {_pl(col)}\n\n"
            "lo = x[0]\nhi = x[0]\n"
            "for v in x:\n"
            "    if v < lo:\n        lo = v\n"
            "    if v > hi:\n        hi = v\n\n"
            f"last = x[-1]                        # 결정월 {last_date} 의 값\n"
            "scaled = (last - lo) / (hi - lo)    # MinMax: 최소 → 0, 최대 → 1\n\n"
            "print(\"최소\", f\"{lo:.5g}\")\nprint(\"최대\", f\"{hi:.5g}\")\nprint(\"눈금\", f\"{scaled:.5g}\")\n")
    exp = f"최소 {_g6(smin_e)}\n최대 {_g6(smax_e)}\n눈금 {_g6(scaled_e)}"
    return {"title": f"{var} 의 최소·최대와 눈금을 직접 계산", "desc": "학습 폴드 값만으로 최소·최대를 찾고, 결정월 값을 0~1 눈금으로 바꿉니다. 출력이 엔진의 MinMaxScaler 값과 같아야 합니다.",
            **run_lab(code, exp, pure=True, rtol=F32)}


def lab_delta(mask, win_no, delta_e):
    T, D = len(mask), len(mask[0])
    rows = "".join(f"    {[int(v) for v in r]},\n" for r in mask)
    code = (f"# 창 번호 {win_no} 의 관측 표시 m — 시점 {T}개 × 변수 {D}개 (1 = 관측, 0 = 빈 칸), 실행 기록 그대로\n"
            f"m = [\n{rows}]\n"
            "T, D = len(m), len(m[0])\n"
            "delta = [[0.0] * D for _ in range(T)]\n\n"
            "# 원본 delta_calc 규칙 그대로 (d = 0 이면 d − 1 = −1 → 마지막 열을 가리킴)\n"
            "for t in range(T):\n"
            "    for d in range(D):\n"
            "        if t == 0:\n"
            "            delta[t][d] = 0.0                      # 첫 시점은 0\n"
            "        elif m[t - 1][d] == 0:\n"
            "            delta[t][d] = delta[t][d - 1] + 1.0    # 같은 변수가 앞 시점에 비었으면: 왼쪽 이웃 변수의 δ + 1\n"
            "        elif m[t][d - 1] == 1:\n"
            "            delta[t][d] = 1.0                      # 왼쪽 이웃 변수가 지금 관측되었으면 1\n"
            "        # 그 밖에는 0 그대로\n\n"
            "for t in range(T):\n"
            "    print(\"τ\" + str(t + 1), delta[t])\n")
    exp = "\n".join(f"τ{t + 1} {[float(v) for v in delta_e[t]]}" for t in range(T))
    return {"title": "δ(델타)를 원본 규칙으로 직접 계산", "desc": "원본 delta_calc 의 세 갈래 규칙을 그대로 옮겼습니다. 출력이 실행 기록의 δ 와 같아야 합니다. (반복이 1,000단계 가까이 되어 Python Tutor 의 단계 한도에 걸릴 수 있으므로, 복사해서 실행하도록 했습니다.)",
            **run_lab(code, exp, pure=True, rtol=F32, tutor=False)}


def lab_gamma(d, w, b, tau, win_no, unit, gamma_e):
    code = ("import math\n\n"
            f"# 창 번호 {win_no}, τ{tau} 의 δ (변수 {len(d)}개) — 실행 기록 그대로\n"
            f"d = {_pl(d)}\n"
            f"# temp_decay_h 의 가중치 W 중 은닉 단위 {unit}(코드 색인 W[{unit - 1}]) 의 행, 편향 b (되돌린 가중치 = 검증 손실이 가장 낮았던 epoch, 유효숫자 12자리)\n"
            f"w = {_pl(w, 12)}\n"
            f"b = {float(b):.12g}\n\n"
            "s = b\n"
            "for dj, wj in zip(d, w):\n"
            "    s += wj * dj                       # W·δ + b\n"
            "gamma = math.exp(-max(0.0, s))        # 음수는 0 으로 자르고(ReLU) exp(−·)\n\n"
            "print(\"γ_h\", f\"{gamma:.5g}\")\n")
    return {"title": f"감마 γ_h 하나를 직접 계산 (은닉 단위 {unit} — 코드 색인 {unit - 1})", "desc": "δ 에 가중치를 곱해 더하고, 음수는 0 으로 자른 뒤 exp(−·) 를 합니다. 기억을 얼마나 남길지(0~1)입니다. 기억을 가장 많이 줄인 은닉 단위를 골랐습니다(단위마다 다릅니다).",
            **run_lab(code, f"γ_h {_g6(gamma_e)}", pure=True, rtol=F32)}


def lab_xhat(h_prev, gamma, w, b, var, win_no, xhat_e):
    code = ("import math\n\n"
            f"# 창 번호 {win_no}: τ1 에서 LSTMCell 이 만든 기억 h (128개) 와 τ2 의 γ_h (128개) — 실행 기록 그대로\n"
            f"h_prev = {_pl(h_prev, 12)}\n"
            f"gamma = {_pl(gamma, 12)}\n"
            f"# hist_reg(Linear 128→34) 의 '{var}' 행과 편향 (되돌린 가중치 = 검증 손실이 가장 낮았던 epoch, 유효숫자 12자리)\n"
            f"w = {_pl(w, 12)}\n"
            f"b = {float(b):.12g}\n\n"
            "s = b\n"
            "for hk, gk, wk in zip(h_prev, gamma, w):\n"
            "    s += wk * (hk * gk)                # 줄인 기억 h·γ 에 가중치를 곱해 더함\n"
            "x_hat = 1 / (1 + math.exp(-s))        # sigmoid → 0~1\n\n"
            f"print(\"τ2 x̂ {var}\", f\"{{x_hat:.5g}}\")\n")
    return {"title": f"τ2 의 x̂ 를 직접 계산 (빈 칸 {var})", "desc": "τ1 은 h = 0 이라 편향만 남아 단순하므로, 한 시점 뒤인 τ2 를 계산합니다. τ2 에서 기억 h 를 γ_h 만큼 줄이고, hist_reg 의 한 행으로 가중합한 뒤 sigmoid 를 씌웁니다. 이 값이 빈 칸을 채우는 재료가 됩니다.",
            **run_lab(code, f"τ2 x̂ {var} {_g6(xhat_e)}", pure=True, rtol=F32)}


def lab_lstmcell(inp, Wi, Wf, Wg, Wo, bi, bf, bg, bo, win_no, unit, c_e, h_e):
    code = ("import math\n\n"
            "def sigmoid(a):\n    return 1 / (1 + math.exp(-a))\n\n"
            f"# 창 번호 {win_no}, τ1 의 LSTMCell 입력 [c_c ; m] ({len(inp)}개) — 실행 기록 그대로\n"
            f"x = {_pl(inp)}\n"
            f"# 은닉 단위 {unit} 의 가중치 W_ih 네 행(i·f·g·o)과 편향 b_ih + b_hh (되돌린 가중치 = 검증 손실이 가장 낮았던 epoch, 유효숫자 12자리)\n"
            f"Wi = {_pl(Wi, 12)}\nWf = {_pl(Wf, 12)}\nWg = {_pl(Wg, 12)}\nWo = {_pl(Wo, 12)}\n"
            f"bi, bf, bg, bo = {float(bi):.12g}, {float(bf):.12g}, {float(bg):.12g}, {float(bo):.12g}\n\n"
            "# τ1 에서는 지난 기억 h = 0, c = 0 이라 W_hh·h 항은 0\n"
            "ai, af, ag, ao = bi, bf, bg, bo\n"
            "for k in range(len(x)):\n"
            "    ai += Wi[k] * x[k]\n    af += Wf[k] * x[k]\n    ag += Wg[k] * x[k]\n    ao += Wo[k] * x[k]\n"
            "i, f, g, o = sigmoid(ai), sigmoid(af), math.tanh(ag), sigmoid(ao)   # 입력 문, 잊기 문, 새 정보, 출력 문\n"
            "c = f * 0.0 + i * g                    # 새 셀 상태 (지난 c = 0)\n"
            "h = o * math.tanh(c)                   # 새 기억\n\n"
            "print(\"c\", f\"{c:.5g}\")\nprint(\"h\", f\"{h:.5g}\")\n")
    return {"title": f"LSTMCell 의 은닉 단위 {unit} 을 직접 계산 (τ1)", "desc": "입력 68개에 네 문의 가중치를 곱해 더하고 σ·tanh 를 씌운 뒤, c = f·c + i·g, h = o·tanh(c) 로 갱신합니다.",
            **run_lab(code, f"c {_g6(c_e)}\nh {_g6(h_e)}", pure=True, rtol=F32)}


def lab_movavg(col, k, n_front, n_end, win_no, var, trend_e, seas_e):
    code = (f"# 창 번호 {win_no} 의 채운 창에서 '{var}' 열 (시점 {len(col)}개) — 실행 기록 그대로\n"
            f"x = {_pl(col)}\n"
            f"k = {k}                                  # 이동평균 창\n\n"
            f"padded = [x[0]] * {n_front} + x + [x[-1]] * {n_end}   # 끝값을 앞에 {n_front}번, 뒤에 {n_end}번 복사\n"
            "trend = []\n"
            "for a in range(len(x)):\n"
            "    s = 0.0\n"
            "    for v in padded[a:a + k]:\n"
            "        s += v\n"
            "    trend.append(s / k)                 # 한 칸씩 밀며 k 개 평균\n"
            "season = [xi - ti for xi, ti in zip(x, trend)]   # 주기 = 채운 값 − 추세\n\n"
            "print(\"추세\", [round(v, 6) for v in trend])\nprint(\"주기\", [round(v, 6) for v in season])\n")
    exp = f"추세 {[round(float(v), 6) for v in trend_e]}\n주기 {[round(float(v), 6) for v in seas_e]}"
    return {"title": f"'{var}' 열의 추세·주기를 직접 계산", "desc": "양끝을 복사해 붙인 뒤 k 개씩 평균을 내 추세를 만들고, 원래 값에서 빼 주기를 만듭니다.",
            **run_lab(code, exp, pure=True, rtol=F32)}


def lab_trend_linear(a3, w, b, win_no, out_e):
    code = (f"# 창 번호 {win_no}: 합성곱 3단을 지난 추세 a³ (시점 {len(a3)}개) — 실행 기록 그대로\n"
            f"a3 = {_pl(a3)}\n"
            "# Linear_Trend 의 시간 가중치 w 와 편향 b (되돌린 가중치)\n"
            f"w = {_pl(w)}\nb = {float(b)!r}\n\n"
            "y = b\n"
            "for wt, at in zip(w, a3):\n"
            "    y += wt * at                      # 시점마다 가중치 × 값\n\n"
            "print(\"추세 경로 출력\", f\"{y:.5g}\")\n")
    return {"title": "추세 경로 출력(Linear_Trend)을 직접 계산", "desc": "시간 6칸의 값에 학습된 가중치를 곱해 더하고 편향을 더합니다(표준화한 정답 단위).",
            **run_lab(code, f"추세 경로 출력 {_g6(out_e)}", pure=True, rtol=F32)}


def lab_reparam(mu, lv, eps, win_no, z_e):
    code = ("import math\n\n"
            f"# 창 번호 {win_no}, τ1 의 요인 {len(mu)}개 — μ, log σ², 이번에 뽑힌 ε (실행 기록 그대로)\n"
            f"mu = {_pl(mu)}\nlogvar = {_pl(lv)}\neps = {_pl(eps)}\n\n"
            "z = []\n"
            "for m, lv, e in zip(mu, logvar, eps):\n"
            "    sigma = math.exp(0.5 * lv)        # σ = exp(½ · log σ²)\n"
            "    z.append(m + e * sigma)           # z = μ + ε · σ\n\n"
            "print(\"z\", [round(v, 6) for v in z])\n")
    return {"title": "z = μ + ε·σ 를 직접 계산", "desc": "로그분산으로 σ 를 만들고, 무작위 수 ε 에 곱해 μ 에 더합니다. μ = 0, log σ² = 0 이면 z 는 ε 그대로입니다.",
            **run_lab(code, f"z {[round(float(v), 6) for v in z_e]}", pure=True, rtol=F32)}


def lab_vae(mu, lv, recon_mse, kld_e, vae_e):
    B_, T_, J_ = len(mu), len(mu[0]), len(mu[0][0])
    m3 = "[\n" + "".join("    [" + ", ".join(_pl(r) for r in mb) + "],\n" for mb in mu) + "]"
    l3 = "[\n" + "".join("    [" + ", ".join(_pl(r) for r in lb) + "],\n" for lb in lv) + "]"
    code = ("import math\n\n"
            f"# 이 배치의 μ, log σ² — [배치 {B_}][시점 {T_}][요인 {J_}] (ReLU 뒤, 실행 기록 그대로)\n"
            f"mu = {m3}\nlogvar = {l3}\n"
            f"recon_mse = {float(recon_mse)!r}            # 재구성 MSE (실행 기록)\n\n"
            "B, T, J = len(mu), len(mu[0]), len(mu[0][0])\n"
            "kld = []\n"
            "for j in range(J):                       # 요인마다\n"
            "    total = 0.0\n"
            "    for b in range(B):\n"
            "        s = 0.0\n"
            "        for t in range(T):                # 시간 축으로 더함 (dim=1)\n"
            "            s += 1 + logvar[b][t][j] - mu[b][t][j] ** 2 - math.exp(logvar[b][t][j])\n"
            "        total += -0.5 * s\n"
            "    kld.append(total / B)                  # 배치로 평균 (dim=0)\n"
            "vae = sum(0.5 * recon_mse + 0.5 * k for k in kld)   # 원본 식: 요인 수만큼 MSE 가 더해짐\n\n"
            "print(\"KLD\", [round(v, 6) + 0.0 for v in kld])\nprint(\"VAE 손실\", f\"{vae:.5g}\")\n")
    exp = f"KLD {[round(float(v), 6) + 0.0 for v in kld_e]}\nVAE 손실 {_g6(vae_e)}"
    return {"title": "KLD 와 VAE 손실을 직접 계산", "desc": "KLD 를 시간 축으로 더하고 배치로 평균낸 뒤, 원본 식대로 0.5·MSE + 0.5·KLD 를 요인마다 더합니다.",
            **run_lab(code, exp, pure=True, rtol=F32)}


def lab_combine(seas, trend, alpha, st_e):
    code = ("# 이 배치 세 창의 주기 경로 출력, 추세 경로 출력, 학습된 α — 실행 기록 그대로\n"
            f"seasonal = {_pl(seas)}\ntrend = {_pl(trend)}\nalpha = {float(alpha)!r}\n\n"
            "states = []\n"
            "for s, t in zip(seasonal, trend):\n"
            "    states.append(s * alpha + t * (1 - alpha))   # 주기 × α + 추세 × (1 − α)\n\n"
            "print(\"states\", [round(v, 6) for v in states])\n")
    return {"title": "두 경로 출력 합치기를 직접 계산", "desc": "주기 경로 출력에 α, 추세 경로 출력에 1 − α 를 곱해 더합니다.",
            **run_lab(code, f"states {[round(float(v), 6) for v in st_e]}", pure=True, rtol=F32)}


def lab_loss(states, yz, vae, imp, mse_e, total_e):
    code = ("# 이 배치의 예측 states, 표준화한 정답 y_z, VAE 손실, 보간 손실 — 실행 기록 그대로\n"
            f"states = {_pl(states)}\ny_z = {_pl(yz)}\nvae_loss = {float(vae)!r}\nimputed_loss = {float(imp)!r}\n\n"
            "sq = 0.0\n"
            "for s, y in zip(states, y_z):\n"
            "    sq += (s - y) ** 2\n"
            "mse = sq / len(states)                  # 예측 MSE\n"
            "loss = vae_loss + imputed_loss + mse    # 원본: 가중치 없는 합\n\n"
            "print(\"MSE\", f\"{mse:.5g}\")\nprint(\"손실\", f\"{loss:.5g}\")\n")
    return {"title": "예측 MSE 와 손실 합을 직접 계산", "desc": "예측과 정답의 차이를 제곱해 평균내고, VAE 손실·보간 손실과 그대로 더합니다.",
            **run_lab(code, f"MSE {_g6(mse_e)}\n손실 {_g6(total_e)}", pure=True, rtol=F32)}


def lab_adam(grad, before, lr, eps, after_e):
    code = ("import math\n\n"
            "# 맨 첫 갱신에서 α 의 기울기 g, 갱신 전 α, 학습률, ε — 실행 기록 그대로\n"
            f"g = {float(grad)!r}\nalpha = {float(before)!r}\nlr = {float(lr)!r}\neps = {float(eps)!r}\n"
            "beta1, beta2 = 0.9, 0.999\n\n"
            "m = (1 - beta1) * g                      # 1차 모멘트 (처음이라 이전 값 0)\n"
            "v = (1 - beta2) * g ** 2                 # 2차 모멘트\n"
            "m_hat = m / (1 - beta1)                  # 편향 보정 → g\n"
            "v_hat = v / (1 - beta2)                  # 편향 보정 → g²\n"
            "alpha = alpha - lr * m_hat / (math.sqrt(v_hat) + eps)\n\n"
            "print(\"갱신 뒤 α\", f\"{alpha:.5g}\")\n")
    return {"title": "Adam 의 첫 걸음을 직접 계산", "desc": "첫 갱신에서는 편향 보정 뒤 m̂ = g, v̂ = g² 라서, 걸음이 기울기 반대 방향으로 거의 학습률만큼입니다.",
            **run_lab(code, f"갱신 뒤 α {_g6(after_e)}", pure=True, rtol=F32)}


def lab_pred(out, sd, mu, level, date, pred_e, next_e):
    code = (f"# 결정월 {date}: 신경망 출력(표준화 단위), 학습 정답의 σ·μ, 그 달의 F11 — 실행 기록 그대로\n"
            f"out = {float(out)!r}\nsigma_y = {float(sd)!r}\nmu_y = {float(mu)!r}\nf11 = {float(level)!r}\n\n"
            "y_hat = out * sigma_y + mu_y                  # 표준화 되돌리기 → 다음 달 변화율 예측\n"
            "next_f11 = f11 * (1 + y_hat)            # 다음 달 F11 예측\n\n"
            "print(\"예측 변화율\", f\"{y_hat:.5g}\")\nprint(\"다음 달 F11\", f\"{next_f11:.5g}\")\n")
    return {"title": "예측을 원래 단위로 되돌리기", "desc": "신경망 출력에 σ_y(학습 창 정답의 표준편차)를 곱하고 μ_y(평균)를 더해 변화율로, 다시 F11 × (1 + 변화율) 로 다음 달 값을 냅니다.",
            **run_lab(code, f"예측 변화율 {_g6(pred_e)}\n다음 달 F11 {_g6(next_e)}", pure=True, rtol=F32)}


def build(cap: dict) -> dict:
    tr = Trace(cap["events"])
    src = Sources([RITS, BRITS, BIVA, VAE, TOOLS, WRAP, EXP, CAL])
    feats, names = cap["feats"], cap["feat_names"]
    K, L, B = len(feats), cap["seq_len"], 3
    p, args = cap["params"], cap["args"]
    sd = {k: v.double() for k, v in cap["net_state"].items()}
    raw = cap["raw_train"]; dates = [d.strftime("%Y-%m") for d in raw.index]
    win = cap["batch"]["win"]; xb = cap["batch"]["x"]
    wdates = [dates[w:w + L] for w in win]  # 각 창의 6개 달
    dec = [cap["win_idx"][w][:7] for w in win]
    colnames = [f"{c} {names.get(c, '')}" for c in feats]
    lat, h1, h2 = args["vae_latent_size"], max(1, K // 2), max(1, max(1, K // 2) // 2)
    c4, c16 = max(1, K // 4), max(1, K // 16)
    steps: list[dict] = []
    checks_all: list[dict] = []
    ORIG_EPOCHS = json.loads((ROOT / "configs/repro_biva_orig.json").read_text(encoding="utf-8"))["model_params"]["train_epochs"]
    best_ep = min(cap["history"], key=lambda h: h["vali_loss"])["epoch"]   # 검증 손실이 가장 낮았던 epoch (되돌린 가중치)

    def H(title, data, rows=None, cols=None, shape=None, note=None, scale="div", hl=None, digits=None, numbers=True, **kw):
        t = {"type": "heat", "title": title, "data": arr(data) if not isinstance(data, list) else data, "rows": rows, "cols": cols,
             "shape": shape, "note": note, "scale": scale, "hl": hl or {}, "numbers": numbers, **kw}
        if digits is not None:
            t["digits"] = digits
        if cols == feats:
            t["colNames"] = colnames
        return t

    def BH(title, x3, rows_list, note=None, shape=None, scale="div", hl=None, cols=None, **kw):
        a = arr(x3)
        kw.setdefault("cellH", 19)
        return {"type": "batch", "title": title, "mats": a, "labels": [f"배치 b={i} — 창 #{win[i]} (결정월 {dec[i]})" for i in range(len(a))],
                "rows": rows_list, "cols": cols or feats, "colNames": colnames, "shape": shape or list(np.shape(x3)), "note": note, "scale": scale, "hl": hl or {}, **kw}

    def chk(what, a, b, tol=1e-5):
        d = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
        c = {"what": f"{what} (차이 {d:.1e})", "ok": bool(d <= tol)}
        checks_all.append({**c, "step": len(steps) + 1})
        return c

    def S(chapter, narr, *, code=None, stack=None, nodes=(), edges=(), tensors=(), formula=None, hint=None, checks=(), say=None, dur=None, notes=None, lab=None):
        """say = 음성·자막 대본(화면 숫자 그대로 — 숫자는 narr 와 같은 변수에서), notes = {원본 줄 번호: 그 줄 설명}, lab = 직접 실행해 보기(실행해 엔진 값과 맞춘 것)."""
        checks = list(checks)
        if lab is not None:
            c_ = {"what": f"직접 실행해 보기 '{lab['title']}' 출력 = 엔진 값", "ok": bool(lab["ok"])}
            checks_all.append({**c_, "step": len(steps) + 1}); checks.append(c_)
        if code is not None and notes:
            code = {**code, "notes": {str(k_): v_ for k_, v_ in notes.items()}}
        steps.append({"chapter": chapter, "narr": narr, "say": say, "code": code, "stack": list(stack or []), "nodes": list(nodes), "edges": [list(e) for e in edges],
                      "tensors": list(tensors), "formula": formula, "hint": hint, "checks": checks, "dur": dur, "lab": lab})

    def C(fn, lines, focus=None, func="", before=8, after=10):
        lines = sorted(set(lines))
        return {"file": fn, "lines": lines, "focus": focus or lines[0], "func": func, "before": before, "after": after}

    stk_fit = ["engine/experiment.py · _walkforward_predict", "engine/experiment.py · _split_fit", "models/biva.py · Model.fit"]
    stk_fwd = stk_fit + ["models/biva.py · _run(bx)", "BIVA.py · Model.forward"]
    stk_brits = stk_fwd + ["BIVA_BRITS.py · Model.forward"]

    def stk_rits(direct, t=None):
        s = stk_brits + [f"BIVA_RITS.py · Model.forward(direct='{direct}')"]
        return s + [f"반복 t = {t} (τ{t + 1 if direct == 'forward' else L - t})"] if t is not None else s

    # ================================================================== 0. 시작
    cfg = cap["config"]
    S("0. 시작",
      f"이 영상은 <b>BIVA 원본 코드가 실제로 실행되는 순서</b>를 한 줄씩 따라갑니다. 설정은 원본 재현 설정 <b>repro_biva_orig</b>입니다. "
      f"대상은 KB 주택매매가격지수(F11)의 다음 달 변화율이고, 입력은 변수 {K}개 × 최근 {L}개월 창, 배치 {p['batch_size']}입니다. "
      f"화면의 숫자는 모두 이 설정으로 실데이터를 <b>실제로 돌린 기록</b>에서 꺼낸 값입니다. "
      f"오른쪽은 그 순간 실행된 코드 줄입니다. 파일이 <b>models/biva_src/</b> 에 있으면 BIVA 원본 코드(고친 곳은 [TSP 수정 #n] 표시)이고, "
      "그 밖(engine/, models/biva.py)은 이 플랫폼의 코드입니다(models/biva.py 의 학습 순서는 원본 exp_main 을 그대로 옮긴 것). 위쪽 그림은 지금 위치입니다.",
      say=(f"이 영상은 BIVA 원본 코드가 실제로 실행되는 순서를 한 줄씩 따라갑니다. 설정은 원본 재현 설정입니다. 대상은 KB 주택매매가격지수의 다음 달 변화율이고, "
           f"입력은 변수 {K}개, 최근 {L}개월 창, 배치 크기 {p['batch_size']}입니다. 화면의 숫자는 모두 이 설정으로 실제 데이터를 돌린 기록에서 꺼낸 값입니다. "
           "오른쪽은 그 순간 실행된 코드 줄입니다. 파일 경로가 BIVA 원본 폴더를 가리키면 원본 코드이고, 그 밖은 이 플랫폼의 코드입니다. "
           "노란 줄 아래 회색 글이 그 줄의 설명입니다. 위쪽 그림은 지금 위치를 보여 줍니다."),
      code=C(WRAP, list(range(1, 18)), 1, "모듈 설명", 0, 20), stack=["models/biva.py"],
      nodes=[n for n in ("data", "scale", "win", "split", "batch", "fwd", "bwd", "merge", "dec", "tc", "tl", "e1", "e2", "e3", "e4", "mu", "z", "decd", "vl", "sc", "sl", "comb", "loss", "back", "epoch", "pred")],
      tensors=[{"type": "table", "title": "이 영상의 실행 설정 (configs/repro_biva_orig.json + 원본 기본값)", "columns": ["항목", "값", "뜻"], "rows": [
          ["대상", "F11 다음 달 변화율", "y_t = F11_(t+1) ÷ F11_t − 1 (예측 연구 모드, k = 1)"],
          ["입력 변수", f"{K}개", "원본 노트북의 변수 목록(중복 3개는 대표 ID 로 치환)"],
          ["창 길이 seq_len", L, "최근 6개월"], ["batch_size", p["batch_size"], "원본 3"], ["BRITS 은닉 rnn_hid_size", p["rnn_hid_size"], "원본 128"],
          ["잠재 vae_latent_size", lat, "원본 4"], ["이동평균 moving_avg", p["moving_avg"], "원본 6"], ["학습률", p["learning_rate"], "원본 0.01, lradj 3"],
          ["학습 반복(이 영상)", cap["epochs"], f"원본 {ORIG_EPOCHS}" + (" — 같음" if cap["epochs"] == ORIG_EPOCHS else " — 줄임(계산 순서·식은 같음)")],
          ["학습 구간", f"{cap['fold']['train'][0][:7]} ~ {cap['fold']['train'][1][:7]}", (f"1회 분할(validation.mode = {cfg['validation']['mode']}) — 폴드 1개" if cfg["validation"]["mode"] == "split" else "엔진의 마지막 폴드") + f" · 테스트 {cap['fold']['test'][0][:7]} ~ {cap['fold']['test'][1][:7]}"],
          ["예측에 쓴 가중치", f"epoch {best_ep}", f"{cap['epochs']} epoch 을 모두 돈 뒤, 검증 손실이 가장 낮았던 epoch 의 가중치로 되돌림(EarlyStopping)"],
          ["전체 가중치 수", f"{cap['n_params']:,}", "nn.Module 의 parameters() 합"]]}],
      hint="숫자의 출처: tools/codewalk/biva_capture.py 가 엔진의 실제 경로(_walkforward_predict → models/biva.py → models/biva_src/*)로 학습·예측을 돌리고(이때는 손실·기울기 등 통계만 기록), "
           "학습이 끝난 뒤 한 배치를 되돌린 가중치로 다시 넣으면서 sys.settrace 로 원본 파일의 실행 줄과 그 줄이 만든 텐서를 기록했습니다. 손으로 다시 계산한 값과의 일치는 각 단계 아래 '검증'에 표시됩니다.\n"
           + (f"학습 반복: 영상은 {cap['epochs']} epoch 로 돌렸습니다(원본 {ORIG_EPOCHS}). 계산 순서·식은 같고, 학습 횟수만큼 가중치 값이 다릅니다. "
              if cap["epochs"] != ORIG_EPOCHS else f"학습 반복: 원본과 같은 {ORIG_EPOCHS} epoch 입니다. ")
           + "다시 만들기: python -m tools.codewalk.biva_capture --epochs N → python -m tools.codewalk.biva_script --capture 기록파일.pkl\n"
           + (f"기록 범위: 이 설정은 1회 분할(validation.mode = {cfg['validation']['mode']})이라 폴드가 하나입니다(학습 {cap['fold']['train'][0][:7]} ~ {cap['fold']['train'][1][:7]}, "
              f"테스트 {cap['fold']['test'][0][:7]} ~ {cap['fold']['test'][1][:7]}). 그래서 이 기록이 엔진 실행 전체와 같고, 「19. 예측」에서 결과 파일과 대조합니다." if cfg["validation"]["mode"] == "split" else
              f"기록 범위: 엔진 워크포워드의 마지막 폴드(학습 {cap['fold']['train'][0][:7]} ~ {cap['fold']['train'][1][:7]}) 하나를 학습했습니다. 전체 실행은 폴드마다 따로 학습합니다."))
    S("0. 시작",
      "전체 호출 순서입니다. 엔진이 학습 창을 만들어 <b>models/biva.py 의 fit</b>에 넘깁니다. fit 은 3개씩 묶은 배치마다 원본 신경망 <b>BIVA.Model.forward</b>를 부릅니다. "
      "forward 안에서는 ① BRITS(앞→뒤·뒤→앞 RITS) ② 분해 ③ 추세 경로(Conv1d·Linear) ④ 주기 경로(LSTM-VAE → Conv1d·Linear) ⑤ 합치기가 차례로 실행됩니다. "
      f"fit 은 손실 3개를 더해 역전파하고 Adam 으로 가중치를 고칩니다. 이것을 epoch 만큼 반복하고, <b>검증 손실이 가장 낮았던 epoch({best_ep})의 가중치로 되돌린 뒤</b> predict 가 테스트 창을 예측합니다.",
      say=(f"전체 호출 순서입니다. 엔진이 학습 창을 만들어 BIVA 모델의 학습 함수에 넘깁니다. 학습 함수는 창 {p['batch_size']}개씩 묶은 배치마다 원본 신경망의 순전파를 부릅니다. "
           "순전파 안에서는 다섯 단계가 차례로 실행됩니다. 첫째, BRITS가 창을 앞에서 뒤로, 뒤에서 앞으로 두 번 읽어 빈 칸을 채웁니다. 둘째, 채운 창을 추세, 곧 이동평균과, 주기 성분, 곧 이동평균을 뺀 나머지로 나눕니다. "
           "셋째, 추세는 합성곱과 선형 층을 거쳐 값 하나를 냅니다. 넷째, 주기는 LSTM VAE를 거친 뒤 합성곱과 선형 층으로 값 하나를 냅니다. 다섯째, 두 값을 합칩니다. "
           "학습 함수는 손실 세 개를 더해 역전파하고, 아담으로 가중치를 고칩니다. 이것을 정해진 에포크만큼 반복하고, 검증 손실이 가장 낮았던 에포크의 가중치로 되돌린 뒤 테스트 창을 예측합니다."),
      notes={src.line(WRAP, "for bx, by in tr:"): f"학습 배치마다(epoch 당 {cap['n_tr'] // p['batch_size']}번): 순전파 → 손실 → 역전파 → Adam",
             src.line(WRAP, "loss = VAE_loss + imputed_loss + loss_states"): "세 손실을 계수 없이(1 : 1 : 1) 그대로 더함(원본)",
             src.line(WRAP, 'early(vali_loss, self.net, "", "")'): "epoch 끝: 검증 손실이 지금까지 최저면 가중치 보관",
             src.line(WRAP, "self.net.load_state_dict(early.best_state)"): f"학습이 끝나면 검증 손실 최저 epoch({best_ep})의 가중치로 되돌림"},
      code=C(WRAP, src.span(WRAP, "def fit(self, X, y", "self.net.load_state_dict(early.best_state)"), src.line(WRAP, "for bx, by in tr:"), "Model.fit", 2, 4),
      stack=stk_fit, nodes=["batch", "fwd", "bwd", "merge", "dec", "tc", "e1", "sc", "comb", "loss", "back", "epoch"],
      tensors=[{"type": "table", "title": "호출 순서 (파일 · 함수 · 줄)", "columns": ["순서", "파일 · 함수", "하는 일"], "rows": [
          [1, f"engine/experiment.py · _walkforward_predict (줄 {src.line(EXP, 'Xw, yw = cal.make_windows(Xf')})", "학습 폴드 → 눈금 → 창 [N, 6, 34]"],
          [2, f"engine/experiment.py · _split_fit (줄 {src.line(EXP, 'def _split_fit')})", "뒤쪽 검증 창 분리, 비움"],
          [3, f"models/biva.py · Model.fit (줄 {src.line(WRAP, 'def fit(self, X, y')})", "정답 표준화 · DataLoader · epoch 반복"],
          [4, f"BIVA.py · Model.forward (줄 {src.line(BIVA, 'def forward(self, x):', after=200)})", "아래 5~9 를 차례로 부름"],
          [5, f"BIVA_BRITS.py · Model.forward → BIVA_RITS.py · Model.forward ×2", "빈 칸 채우기(LSTMCell 을 시점마다)"],
          [6, f"BIVA.py · series_decomp → moving_avg", "추세 / 주기 나누기"],
          [7, "BIVA.py · Conv1d_Trend_1~3 → Linear_Trend", "추세 → 값 하나"],
          [8, "BIVA_LSTM_VAE.py · encode → reparameterize → decode, BIVA.py · loss_function", "주기(창 − 이동평균, 코드 이름 seasonal) → 잠재변수 z(4차원), VAE 손실"],
          [9, "BIVA.py · Conv1d_Seasonal_1~3 → Linear_Seasonal → 합치기 α", "주기 → 값 하나, 두 값을 합침"],
          [10, f"models/biva.py · fit (줄 {src.line(WRAP, 'loss = VAE_loss + imputed_loss + loss_states')})", "손실 합 → backward → Adam.step"],
          [11, f"tools_orig.py · EarlyStopping / adjust_learning_rate", "epoch 마다 검증·학습률"],
          [12, f"models/biva.py · predict (줄 {src.line(WRAP, 'def predict(self, X)')})", "테스트 창 → ŷ (원래 단위)"]]}])

    # ================================================================== 1. 데이터
    rawv = raw.values.astype(float)
    mn, mx = np.nanmin(rawv, 0), np.nanmax(rawv, 0)
    disp = (rawv - mn) / np.where(mx > mn, mx - mn, 1)
    nan_cols = [(c, raw[c].first_valid_index().strftime("%Y-%m")) for c in feats if raw[c].isna().any()]
    every = [d if i % 6 == 0 else "" for i, d in enumerate(dates)]
    S("1. 데이터 표",
      f"출발점은 엔진이 만든 <b>학습 폴드의 표</b>입니다. 행은 월({dates[0]} ~ {dates[-1]}, {len(dates)}개월)이고 열은 변수 {K}개입니다. "
      f"색은 열마다 최소~최대를 0~1 로 바꿔 칠한 것입니다(보기용 — 단위가 서로 달라서). 빗금 칸은 <b>빈 칸</b>입니다. "
      + ", ".join(f"{c}는 {d}부터" for c, d in nan_cols) + " 값이 있습니다. 이 빈 칸을 뒤에서 BRITS 가 채웁니다.",
      say=(f"출발점은 엔진이 만든 학습 폴드의 표입니다. 행은 {dates[0]}부터 {dates[-1]}까지 {len(dates)}개월이고, 열은 변수 {K}개입니다. "
           "색은 열마다 최소에서 최대를 0에서 1로 바꿔 칠한 것으로, 보기 위한 것입니다. 빗금 칸은 빈 칸입니다. "
           + ", ".join(f"{c}는 {d}부터" for c, d in nan_cols) + " 값이 있습니다. 이 빈 칸을 뒤에서 BRITS가 채웁니다."),
      notes={src.line(EXP, "full = ds.X.loc[f.train].join(ds.y.loc[f.train]"): f"학습 폴드의 입력 표({len(dates)}행 × 변수 {K}개)와 정답을 붙임"},
      code=C(EXP, [src.line(EXP, "full = ds.X.loc[f.train].join(ds.y.loc[f.train]")], src.line(EXP, "full = ds.X.loc[f.train].join(ds.y.loc[f.train]"), "_walkforward_predict", 6, 6),
      stack=["engine/experiment.py · _walkforward_predict"], nodes=["data"],
      tensors=[H("학습 폴드 원자료 (색: 열마다 최소~최대)", disp, rows=every, cols=feats, shape=[len(dates), K], scale="01", numbers=False, cellH=4, rowLabelW=64,
                 note="빗금 = 빈 칸(NaN)")],
      hint=f"이 설정은 apply_lag=false(발표시차 미적용, 원본 노트북과 같은 조건)입니다. 그래서 한 달의 행에 그 달이 끝난 뒤(다음 달 이후)에야 발표되는 값이 들어갈 수 있습니다. 실제 시점의 예측력을 보려면 apply_lag=true 로 비교해야 합니다.\n"
           "변수 중복 처리: 설정의 L1·N1·N12 는 변수 사전에서 USDKRW·BOK_RATE·KR3Y 의 중복이라 대표 ID 로 바뀝니다. "
           "2026-11-07 수정 전에는 설정에 적은 '변환 none' 이 대표 ID 로 넘어가지 않아 USDKRW 는 변화율, BOK_RATE·KR3Y 는 차분으로 들어갔습니다. 지금은 원본처럼 원래 값입니다.")
    yv = cap["y_train"].values
    lev = cap["target_level"]
    rows_y = []
    for d in raw.index[-4:]:
        i = list(lev.index).index(d)
        rows_y.append([d.strftime("%Y-%m"), num(lev.iloc[i], 6), lev.index[i + 1].strftime("%Y-%m"), num(lev.iloc[i + 1], 6), num(cap["y_train"].loc[d], 5)])
    S("1. 데이터 표",
      "정답 y 는 결정월 t 의 F11 값에 대한 <b>다음 달 F11 값의 변화율</b>입니다(엔진 make_forecast_target, k = 1). "
      f"표의 마지막 행처럼, {rows_y[-1][0]} 의 정답은 {rows_y[-1][2]} 값 {rows_y[-1][3]} ÷ {rows_y[-1][1]} − 1 = {rows_y[-1][4]} 입니다. "
      "정답에는 입력 창 바깥의 값, 곧 한 달 뒤 F11 값이 들어갑니다(원본은 입력의 마지막 행을 정답으로 썼지만, 여기서는 그렇게 하지 않음 — models/biva.py 머리말 a).",
      say=(f"정답 y는 결정월 F11 값에 대한 다음 달 F11 값의 변화율입니다. 표의 마지막 행처럼, {rows_y[-1][0]}의 정답은 {rows_y[-1][2]}의 값 {rows_y[-1][3]}을 "
           f"{rows_y[-1][1]}로 나누고 1을 뺀 {rows_y[-1][4]}입니다. 정답에는 입력 창 바깥의 값, 곧 한 달 뒤 F11 값이 들어갑니다. 원본은 입력의 마지막 행을 정답으로 썼지만, 여기서는 그렇게 하지 않습니다."),
      notes={src.line(CAL, "r = v.shift(-k) / v - 1.0"): f"한 칸 뒤(다음 달) 값 ÷ 이번 달 값 − 1 → 정답 y. 예) {rows_y[-1][3]} ÷ {rows_y[-1][1]} − 1 = {rows_y[-1][4]}"},
      code=C(CAL, src.span(CAL, "def make_forecast_target", "y.name = f\"forecast_"), src.line(CAL, "r = v.shift(-k) / v - 1.0"), "make_forecast_target", 2, 2),
      stack=["engine/experiment.py · build_dataset", "engine/calendar.py · make_forecast_target"], nodes=["data"],
      formula=f"y_t = F11_(t+1) ÷ F11_t − 1   예) {rows_y[-1][3]} ÷ {rows_y[-1][1]} − 1 = {rows_y[-1][4]}",
      tensors=[{"type": "table", "title": "정답 만들기 (학습 폴드 마지막 4개월)", "columns": ["결정월 t", "F11_t", "다음 달", "F11_(t+1)", "정답 y_t"], "rows": rows_y},
               {"type": "line", "title": "학습 폴드의 정답 y (월별 F11 변화율)", "x": dates, "series": [{"name": "y", "color": "#3b6fb6", "y": [num(v) for v in yv]}], "height": 180}])

    # ================================================================== 2. 눈금
    smin, smax = cap["scaler_min"], cap["scaler_max"]
    j11 = feats.index("F11")
    last = raw.index[-1]
    v11 = float(raw["F11"].iloc[-1]); s11 = (v11 - smin[j11]) / (smax[j11] - smin[j11])
    full_scaled_ = np.vstack([cap["Xw"][0]] + [cap["Xw"][i][-1:] for i in range(1, len(cap["Xw"]))])   # 엔진이 눈금을 맞춘 표(실험 코드의 기대 출력)
    S("2. 눈금 맞추기",
      "단위가 다른 변수(%, 원, 지수)를 같은 눈금으로 바꿉니다. 원본처럼 <b>최소 = 0, 최대 = 1</b>(MinMax)입니다. "
      f"최소·최대는 <b>학습 폴드의 행만</b>으로 정합니다(빈 칸은 무시). 테스트 기간 값은 쓰지 않습니다. "
      f"예: F11 의 학습 최소 {f(smin[j11])}, 최대 {f(smax[j11])} → {last.strftime('%Y-%m')} 값 {f(v11)} 은 {f(s11)} 이 됩니다.",
      say=(f"단위가 다른 변수를 같은 눈금으로 바꿉니다. 원본처럼 최소를 0, 최대를 1로 맞춥니다. 최소와 최대는 학습 폴드의 행만으로 정하고, 빈 칸은 무시합니다. "
           f"테스트 기간의 값은 쓰지 않습니다. 예를 들어 F11의 학습 최소는 {f(smin[j11])}, 최대는 {f(smax[j11])}입니다. 그래서 {last.strftime('%Y-%m')}의 값 {f(v11)}은 {f(s11)}이 됩니다."),
      notes={src.line(EXP, "StandardScaler()).fit(full["): f"MinMax: 학습 폴드 행으로 열마다 최소·최대(빈 칸 무시). F11 {f(smin[j11])} ~ {f(smax[j11])}",
             src.line(EXP, "Xf = scaler.transform(full[ds.feature_names].ffill().values)"): "앞 값으로 채운 뒤(ffill) 0~1 로 바꿈 — 변수가 시작되기 전 칸은 NaN 그대로"},
      lab=lab_minmax(raw["F11"].values, "F11", last.strftime("%Y-%m"), smin[j11], smax[j11], full_scaled_[-1, j11]),
      code=C(EXP, src.lines(EXP, "scaler = (MinMaxScaler() if params.get(\"input_scaler\") == \"minmax\" else StandardScaler()).fit(full[",
                            "Xf = scaler.transform(full[ds.feature_names].ffill().values)"),
             src.line(EXP, "StandardScaler()).fit(full["), "_walkforward_predict", 6, 6),
      stack=["engine/experiment.py · _walkforward_predict"], nodes=["scale"], edges=[("data", "scale")],
      formula=f"x′ = (x − 최소) ÷ (최대 − 최소)   F11: ({f(v11)} − {f(smin[j11])}) ÷ ({f(smax[j11])} − {f(smin[j11])}) = {f(s11)}",
      tensors=[{"type": "table", "split": 2, "title": "변수별 학습 구간 최소·최대 (sklearn MinMaxScaler.data_min_ / data_max_)", "columns": ["변수", "이름", "최소", "최대"],
                "rows": [[c, names.get(c, "")[:22], num(smin[j]), num(smax[j])] for j, c in enumerate(feats)], "hlRows": [j11]}],
      hint="원본 노트북은 전체 기간(테스트 포함)으로 눈금을 맞췄습니다. 이는 미래 정보가 섞이는 누출입니다. 이 플랫폼은 학습 폴드만 씁니다.\n"
           "2026-11-07 수정: 그 전에는 '빈 칸이 하나도 없는 행'(이 폴드에서 135행 중 63행, 2015년 이후)만으로 맞춰서, 2009~2014 의 일부 값이 0~1 밖(−2.2 ~ 4.8)으로 나갔습니다. "
           "BRITS 가 채우는 값은 sigmoid 출력이라 0~1 이므로 눈금이 맞지 않았습니다. 지금은 학습 폴드의 모든 행(빈 칸만 무시)으로 맞춥니다.")
    Xw = cap["Xw"]
    n_lead = n_mid = 0                                  # 학습 폴드 표의 빈 칸: 변수가 시작되기 전 / 시작된 뒤(중간)
    for c_ in raw.columns:
        fv_ = raw[c_].first_valid_index()
        if fv_ is not None:
            n_lead += int(raw[c_].loc[:fv_].isna().sum()); n_mid += int(raw[c_].loc[fv_:].isna().sum())
    full_scaled = np.vstack([Xw[0]] + [Xw[i][-1:] for i in range(1, len(Xw))])
    S("2. 눈금 맞추기",
      f"바꾼 표입니다. 빈 칸은 앞의 값으로 채우는데(ffill), 변수가 시작되기 전의 빈 칸은 앞에 값이 없어 그대로 남습니다. "
      + (f"이 폴드에는 중간 빈 칸이 없어서, 남은 빈 칸 {n_lead}개는 모두 변수가 시작되기 전의 칸입니다. " if n_mid == 0 else f"중간 빈 칸 {n_mid}개는 앞 값으로 채워졌고, 변수가 시작되기 전의 빈 칸 {n_lead}개가 남습니다. ")
      + 
      "BIVA 는 이 빈 칸을 <b>NaN 그대로</b> 받아 BRITS 가 채웁니다. 다른 창 모델은 학습 평균으로 채웁니다.",
      say=("바꾼 표입니다. 빈 칸은 앞의 값으로 채우는데, 변수가 시작되기 전의 빈 칸은 앞에 값이 없어서 그대로 남습니다. "
           + ("이 폴드에는 중간 빈 칸이 없어서, 남은 빈 칸은 모두 변수가 시작되기 전의 칸입니다. " if n_mid == 0 else "")
           + "BIVA는 이 빈 칸을 비어 있는 그대로 받아 BRITS가 채웁니다. 다른 창 모델은 학습 평균으로 채웁니다."),
      notes={src.line(EXP, "if not accepts_nan:"): "BIVA 는 빈 칸(NaN)을 받는 모델(ACCEPTS_NAN) → 평균으로 채우는 이 줄을 건너뜀"},
      code=C(EXP, src.lines(EXP, "Xf = scaler.transform(full[ds.feature_names].ffill().values)", "if not accepts_nan:", "Xf = np.where(np.isnan(Xf), fill_val[None, :], Xf)"),
             src.line(EXP, "Xf = scaler.transform(full[ds.feature_names].ffill().values)"), "_walkforward_predict", 8, 6),
      stack=["engine/experiment.py · _walkforward_predict"], nodes=["scale"],
      tensors=[H("눈금 맞춘 학습 폴드 (0~1)", full_scaled, rows=every, cols=feats, shape=[len(dates), K], scale="01", numbers=False, cellH=4, rowLabelW=64, note="빗금 = NaN(BRITS 가 채움)")])

    # ================================================================== 3. 창
    n_win = len(Xw)
    S("3. 창 만들기",
      f"연속한 {L}개월을 한 묶음(<b>창</b>)으로 자릅니다. 창 j 는 표의 j ~ j+{L - 1} 행이고, 마지막 행이 <b>결정월</b>, 정답은 그 결정월의 y 입니다. "
      f"표가 {len(dates)}행이므로 창은 {len(dates)} − {L} + 1 = <b>{n_win}개</b>, 모양은 [{n_win}, {L}, {K}] 입니다. "
      f"그림의 상자는 이 영상이 따라갈 배치의 창 #{win[0]}·#{win[1]}·#{win[2]} 입니다.",
      say=(f"연속한 {L}개월을 한 묶음, 곧 창으로 자릅니다. 창의 마지막 행이 결정월이고, 정답은 그 결정월의 y입니다. "
           f"표가 {len(dates)}행이므로 창은 {len(dates)} 빼기 {L} 더하기 1, 모두 {n_win}개입니다. 그림의 상자는 이 영상이 따라갈 배치의 창, 창 번호 {win[0]}, {win[1]}, {win[2]}입니다."),
      notes={src.line(CAL, "idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]"): f"창 j = 행 j … j+{L - 1} → 창 {n_win}개, 모양 [{n_win}, {L}, {K}]"},
      code=C(CAL, src.span(CAL, "def make_windows", "return X[idx], y[seq_len - 1 :]"), src.line(CAL, "idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]"), "make_windows", 2, 2),
      stack=["engine/experiment.py · _walkforward_predict", "engine/calendar.py · make_windows"], nodes=["win"], edges=[("scale", "win")],
      formula=f"idx[j] = [j, j+1, …, j+{L - 1}]   창 j = 표[idx[j]],  정답 = y[j+{L - 1}]   (j = 0 … {n_win - 1})",
      tensors=[H("표 위의 창 (상자 = 창 하나 = 6행)", full_scaled, rows=every, cols=feats, shape=[len(dates), K], scale="01", numbers=False, cellH=4, rowLabelW=64,
                 hl={"boxes": [[w, 0, w + L - 1, K - 1, col] for w, col in zip(win, ["#c8710f", "#2f8f5b", "#6a3fa0"])]}),
               {"type": "table", "title": "창 번호 → 행 번호 → 결정월 → 정답", "columns": ["창 j", "행(idx[j])", "달", "결정월", "정답 y"],
                "rows": [[w, f"{w}~{w + L - 1}", f"{dates[w]} ~ {dates[w + L - 1]}", cap["win_idx"][w][:7], num(cap["yw"][w], 5)] for w in [0, 1] + list(win)]}])

    # ================================================================== 4. 검증 분할
    n_tr, n_va, emb = cap["n_tr"], cap["n_va"], cap["embargo"]
    S("4. 검증 나누기·정답 표준화",
      f"뒤쪽 창 중 결정월이 검증 시작일 {cap['fold']['val_from'][:7]} 이후인 것이 <b>검증</b> 창입니다({n_va}개). "
      f"그 바로 앞 {emb}개는 <b>비움</b>(embargo)으로 버립니다. 결정월 t 의 정답은 t+1 달의 값을 쓰므로, 비우지 않으면 학습에 들어갈 결정월 {cap['win_idx'][n_tr + emb - 1][:7]} 창의 정답이 "
      f"검증 구간의 첫 달 {cap['win_idx'][n_tr + emb][:7]} 값을 쓰게 됩니다. 비움은 <b>학습 정답이 검증 구간({cap['win_idx'][n_tr + emb][:7]} 이후)의 값을 쓰지 않게</b> 합니다(검증 창의 입력은 6개월 전부터라 학습 기간과 겹칠 수 있습니다). "
      f"나머지 <b>{n_tr}개</b>가 학습 창입니다. "
      "검증 창은 가중치를 고치는 데 쓰지 않고, epoch 마다 <b>검증 손실</b>(VAE·보간·예측 손실의 합, models/biva.py · _vali)을 재서 그 값이 가장 낮았던 시점을 고르는 데만 씁니다.",
      say=(f"뒤쪽 창 가운데 결정월이 검증 시작일 {cap['fold']['val_from'][:7]} 이후인 것이 검증 창으로, {n_va}개입니다. 그 바로 앞 {emb}개는 비움으로 버립니다. "
           f"결정월의 정답은 다음 달 값을 쓰므로, 이렇게 비워야 학습 창의 정답이 검증 구간, 곧 {cap['win_idx'][n_tr + emb][:7]} 이후의 값을 쓰지 않습니다. 나머지 {n_tr}개가 학습 창입니다. "
           "검증 창은 가중치를 고치는 데 쓰지 않고, 에포크마다 검증 손실, 곧 VAE, 보간, 예측 손실의 합을 재서, 그 값이 가장 낮았던 시점을 고르는 데만 씁니다."),
      notes={src.line(EXP, "cut = n - n_val - emb"): f"cut = {n_win} − {n_va} − {emb} = {n_tr} → 앞 {n_tr}개 학습, 뒤 {n_va}개 검증"},
      code=C(EXP, src.span(EXP, "def _split_fit", "return cut, n_val"), src.line(EXP, "cut = n - n_val - emb"), "_split_fit", 1, 2),
      stack=stk_fit[:2], nodes=["split"], edges=[("win", "split")],
      formula=f"n_val = (결정월 ≥ {cap['fold']['val_from'][:7]} 인 창 수) = {n_va},  cut = {n_win} − {n_va} − {emb} = {n_tr}  →  학습 창 0~{n_tr - 1}, 검증 창 {n_win - n_va}~{n_win - 1}",
      tensors=[{"type": "segs", "title": f"창 {n_win}개 나누기", "segs": [
          {"label": "학습", "n": n_tr, "color": "#cfe0f5", "sub": f"{cap['win_idx'][0][:7]} ~ {cap['win_idx'][n_tr - 1][:7]}"},
          {"label": "비움", "n": emb, "color": "#e3e6ea"},
          {"label": "검증", "n": n_va, "color": "#fde2b8", "sub": f"{cap['win_idx'][n_win - n_va][:7]} ~ {cap['win_idx'][-1][:7]}"}]}])
    ytr = cap["yw"][:n_tr]
    fs0 = cap["first_step"]
    S("4. 검증 나누기·정답 표준화",
      f"정답은 크기가 매우 작아(절댓값 평균 {f(np.abs(ytr).mean())}) 그대로 두면 예측 손실이 다른 손실에 묻힙니다. 그래서 학습 창 정답의 평균 μ = {g4(cap['y_mu'])}, "
      f"표준편차 σ = {g4(cap['y_sd'])} 로 <b>표준화</b>해서 학습합니다(target_scale=standard). 예측할 때는 거꾸로 되돌립니다.",
      say=(f"정답은 크기가 매우 작아서, 절댓값 평균이 {f(np.abs(ytr).mean())} 정도입니다. 그대로 두면 예측 손실이 다른 손실에 묻힙니다. "
           f"그래서 학습 창 정답의 평균 {g4(cap['y_mu'])}과 표준편차 {g4(cap['y_sd'])}로 표준화해서 학습합니다. 예측할 때는 거꾸로 되돌립니다."),
      notes={src.line(WRAP, "self.y_mu, self.y_sd = float(np.mean(y))"): f"학습 창 정답의 평균 μ = {g4(cap['y_mu'])}, 표준편차 σ = {g4(cap['y_sd'])}",
             src.line(WRAP, "yt = torch.tensor(((np.asarray(y"): "정답을 (y − μ) ÷ σ 로 바꿔 텐서로"},
      code=C(WRAP, src.span(WRAP, "if p[\"target_scale\"] == \"standard\":", "self.y_mu, self.y_sd = 0.0, 1.0") + [src.line(WRAP, "yt = torch.tensor(((np.asarray(y")],
             src.line(WRAP, "self.y_mu, self.y_sd = float(np.mean(y))"), "Model.fit / _tensor", 2, 6),
      stack=stk_fit, nodes=["split"],
      formula=f"y_z = (y − μ) ÷ σ = (y − {g4(cap['y_mu'])}) ÷ {g4(cap['y_sd'])}",
      tensors=[{"type": "line", "title": f"학습 창 {n_tr}개의 표준화한 정답 y_z (평균 0, 표준편차 1)", "x": [cap["win_idx"][i][:7] for i in range(n_tr)], "xlabel": "결정월",
                 "series": [{"name": "y_z = (y − μ) ÷ σ", "color": "#3b6fb6", "y": [num((v - cap["y_mu"]) / cap["y_sd"]) for v in ytr]}], "height": 200},
               {"type": "table", "title": "예: 앞 3개 창", "columns": ["창", "결정월", "y", "y − μ", "÷ σ = y_z"],
                "rows": [[i, cap["win_idx"][i][:7], f"{ytr[i]:.5f}", f"{ytr[i] - cap['y_mu']:.5f}", f"{(ytr[i] - cap['y_mu']) / cap['y_sd']:.4f}"] for i in range(3)]}],
      checks=[chk("μ = 학습 창 정답 평균", cap["y_mu"], float(np.mean(ytr)), 1e-12), chk("σ = 학습 창 정답 표준편차", cap["y_sd"], float(np.std(ytr)), 1e-12)],
      hint=(f"원본은 정답을 표준화하지 않고 지수 수준값 그대로 맞혔습니다. 이 플랫폼의 정답(변화율)은 크기(절댓값 평균)가 {f(np.abs(ytr).mean())} 정도입니다. "
            f"그대로 두면 예측 MSE 는 정답 제곱 평균 {float(np.mean(ytr ** 2)):.2e} 수준이라, 첫 배치의 VAE·보간 손실 합 {f(fs0['vae_loss'] + fs0['imp_loss'], 3)} 보다 "
            f"약 {round((fs0['vae_loss'] + fs0['imp_loss']) / float(np.mean(ytr ** 2)) / 1e4):.0f}만 배 작습니다. "
            "그러면 세 손실이 함께 닿는 가중치(BRITS·VAE)의 학습과 검증 손실 합에서 예측 오차가 거의 반영되지 않습니다(models/biva.py 머리말 e — 표준화하지 않은 비교 학습은 하지 않았습니다). "
            "target_scale=none 으로 원본처럼 둘 수 있습니다."))

    # ================================================================== 5. 배치
    fs = cap["first_step"]
    S("5. 배치",
      f"DataLoader 가 학습 창 {n_tr}개를 epoch 마다 <b>섞은 뒤 {p['batch_size']}개씩</b> 묶습니다. 남는 창은 버립니다(drop_last, 원본 그대로). "
      f"{n_tr} ÷ {p['batch_size']} = {n_tr // p['batch_size']} 배치이고 남는 창은 {n_tr % p['batch_size']}개입니다. 검증 창 {n_va}개도 같은 방식이라 {n_va // p['batch_size']} 배치입니다. "
      f"실제 첫 epoch 의 첫 배치: 창 #{fs['win'][0]}·#{fs['win'][1]}·#{fs['win'][2]}.",
      say=(f"데이터로더가 학습 창 {n_tr}개를 에포크마다 섞은 뒤 {p['batch_size']}개씩 묶습니다. 남는 창은 버리는데, 원본 그대로입니다. "
           f"{n_tr} 나누기 {p['batch_size']}는 {n_tr // p['batch_size']}이므로 배치는 {n_tr // p['batch_size']}개이고, "
           + (f"남는 창은 {n_tr % p['batch_size']}개입니다. " if n_tr % p["batch_size"] else "남는 창은 없습니다. ")
           + f"검증 창 {n_va}개도 같은 방식이라 배치 {n_va // p['batch_size']}개입니다. 실제 첫 에포크의 첫 배치는 창 번호 {fs['win'][0]}, {fs['win'][1]}, {fs['win'][2]}입니다."),
      notes={src.line(WRAP, "tr = DataLoader(TensorDataset(*self._tensor(X, y)), batch_size="): f"섞기(shuffle) · {p['batch_size']}개씩 · 남는 창 버림(drop_last) → epoch 당 {n_tr // p['batch_size']} 배치",
             src.line(WRAP, "va = DataLoader(TensorDataset(*self._tensor(X_val, y_val))"): f"검증 창 {n_va}개 → {n_va // p['batch_size']} 배치"},
      code=C(WRAP, [src.line(WRAP, "tr = DataLoader(TensorDataset(*self._tensor(X, y)), batch_size="), src.line(WRAP, "va = DataLoader(TensorDataset(*self._tensor(X_val, y_val))")],
             src.line(WRAP, "tr = DataLoader(TensorDataset(*self._tensor(X, y)), batch_size="), "Model.fit", 4, 6),
      stack=stk_fit, nodes=["batch"], edges=[("split", "batch")],
      formula=f"bx: [{p['batch_size']}, {L}, {K}] (창 3개),  by: [{p['batch_size']}, 1, 1] (표준화한 정답 3개) — epoch 당 {n_tr // p['batch_size']}번 가중치를 고침",
      tensors=[{"type": "table", "title": "실제 첫 epoch 의 첫 배치", "columns": ["b", "창 #", "6개월", "결정월", "정답 y", "표준화 y_z"],
                "rows": [[i, w, f"{dates[w]} ~ {dates[w + L - 1]}", cap["win_idx"][w][:7], f"{cap['yw'][w]:.5f}", f"{fs['y_z'][i]:.4f}"] for i, w in enumerate(fs["win"])]}],
      hint="원본 data_factory 와 같이 학습·검증 모두 shuffle=True, drop_last=True 입니다. 섞는 순서가 epoch 마다 달라서, 남는 창이 생기는 설정이면 epoch 마다 다른 창이 빠집니다. 이 폴드는 111 이 3 으로 나누어떨어져 빠지는 창이 없습니다.")
    rows_w = [[f"τ{t + 1} {wdates[i][t]}" for t in range(L)] for i in range(B)]
    S("5. 배치",
      f"이제부터 따라갈 배치입니다. 마지막 epoch 에 실제로 쓰인 배치 중 <b>빈 칸이 가장 많은</b> 것을 골랐습니다(창 #{win[0]}·#{win[1]}·#{win[2]}, 빈 칸 {cap['batch']['n_nan']}개). "
      f"모양은 [{B}, {L}, {K}] 이고 배치 · 시점 τ · 변수 순서입니다. 아래 그림은 창 3개를 위에서부터 b = 0, 1, 2 로 보여 줍니다(음성에서는 첫 번째·두 번째·세 번째 창 — b=0 이 창 #{win[0]}). 빗금이 BRITS 가 채울 칸입니다. "
      f"이 배치를 학습이 끝난 뒤 되돌린 가중치(검증 손실이 가장 낮았던 epoch {min(cap['history'], key=lambda h: h['vali_loss'])['epoch']})로 다시 넣어, 원본 코드가 실행되는 줄과 값을 기록했습니다. 예측에 쓰이는 것도 이 가중치입니다.",
      say=(f"이제부터 따라갈 배치입니다. 마지막 에포크에 실제로 쓰인 배치 가운데 빈 칸이 가장 많은 것을 골랐습니다. 창 번호 {win[0]}, {win[1]}, {win[2]}이고, "
           f"빈 칸은 {cap['batch']['n_nan']}개입니다. 모양은 배치 {B}, 시점 {L}, 변수 {K}입니다. 아래 그림은 창 {B}개를 위에서부터 보여 주고, 빗금이 BRITS가 채울 칸입니다. 앞으로 이 순서대로 첫 번째, 두 번째, 세 번째 창이라고 부릅니다. 첫 번째 창이 창 번호 {win[0]}입니다. "
           f"이 배치를, 학습이 끝난 뒤 되돌린 가중치, 곧 검증 손실이 가장 낮았던 에포크 {best_ep}의 가중치로 다시 넣어서, 원본 코드가 실행되는 줄과 값을 기록했습니다. "
           "예측에 쓰이는 것도 이 가중치입니다."),
      notes={src.line(WRAP, "return self.net(bx.float().to(self.device))"): f"배치 [{B}, {L}, {K}] 를 원본 신경망 BIVA.Model 에 넣음(순전파)"},
      code=C(WRAP, [src.line(WRAP, "def _run(self, bx):"), src.line(WRAP, "return self.net(bx.float().to(self.device))")], src.line(WRAP, "return self.net(bx.float().to(self.device))"), "_run", 2, 4),
      stack=stk_fit + ["models/biva.py · _run(bx)"], nodes=["batch"],
      tensors=[{**BH("bx — 이번 배치 (눈금 0~1, 빗금 = NaN)", xb, rows_w[0], shape=[B, L, K], scale="01"), "rowsEach": rows_w}])
    steps[-1]["tensors"][0]["labels"] = [f"b={i} — 창 #{win[i]} ({wdates[i][0]} ~ {wdates[i][-1]}, 결정월 {dec[i]}) · 정답 y_z {num(cap['batch']['y_z'][i], 4)}" for i in range(B)]

    # ================================================================== 6. forward 진입
    e0 = tr.one(BIVA, func="forward", call=True)
    S("6. BIVA.forward",
      "배치가 원본 신경망 <b>BIVA.Model.forward</b>에 들어갑니다. RIN 은 꺼져 있어(원본 기본) 건너뛰고, 곧바로 <b>self.BRITS(x)</b>를 부릅니다. "
      "forward 는 마지막에 6개를 돌려줍니다: 예측 states, VAE 손실, 되살린 주기, 주기 성분, BRITS 손실(보간 손실 + 일관성 손실), 채운 창.",
      say=("배치가 원본 신경망의 순전파 함수에 들어갑니다. 입력 정규화인 RIN은 원본 기본값대로 꺼져 있어 건너뛰고, 곧바로 BRITS를 부릅니다. "
           "순전파는 마지막에 여섯 가지를 돌려줍니다. 예측값, VAE 손실, 되살린 주기, 주기 성분, 일관성 손실까지 더한 보간 손실, 그리고 채운 창입니다."),
      notes={src.line(BIVA, "x = self.BRITS(x)"): "빈 칸 채우기: 앞→뒤 RITS 와 뒤→앞 RITS 를 돌려 평균",
             src.line(BIVA, "imputed_x = x['imputations']", after=200): "채운 창 [3, 6, 34]",
             src.line(BIVA, "imputed_loss = x['loss']", after=200): "BRITS 손실(보간 손실 + 일관성 손실)"},
      code=C(BIVA, src.span(BIVA, "def forward(self, x):", "imputed_loss = x['loss']", after=200), src.line(BIVA, "x = self.BRITS(x)"), "Model.forward", 1, 4),
      stack=stk_fwd, nodes=["fwd", "bwd", "merge"], edges=[("batch", "fwd")],
      tensors=[{"type": "num", "title": "들어온 x", "items": [{"label": "모양", "value": str(list(e0.produced["x"].shape))}, {"label": "빈 칸(NaN) 수", "value": int(torch.isnan(e0.produced["x"]).sum())},
                                                     {"label": "RIN", "value": "꺼짐"}, {"label": "combination", "value": "켜짐(α 학습)"}]}])

    # ================================================================== 7. BRITS 앞→뒤
    S("7. BRITS 앞→뒤",
      "BRITS 는 창을 <b>두 번</b> 읽습니다. rits_f 는 앞→뒤(τ1→τ6)로, rits_b 는 뒤→앞으로 읽습니다. 둘은 구조는 같지만 <b>가중치가 따로</b>입니다. "
      "뒤→앞 결과는 시간 순서를 다시 뒤집어(reverse) 앞→뒤와 맞춘 뒤, merge_ret 에서 평균을 냅니다.",
      say=(f"BRITS는 창을 두 번 읽습니다. 하나는 앞에서 뒤로, τ1에서 τ{L} 방향으로 읽고, 다른 하나는 뒤에서 앞으로 읽습니다. 둘은 구조는 같지만 가중치가 따로입니다. "
           "뒤에서 앞으로 읽은 결과는 시간 순서를 다시 뒤집어 앞에서 뒤로 읽은 결과와 맞춘 뒤, 두 결과의 평균을 냅니다."),
      notes={src.line(BRITS, "ret_f = self.rits_f(data, 'forward')"): "앞→뒤로 읽기(rits_f, 자기 가중치)",
             src.line(BRITS, "ret_b = self.reverse(self.rits_b(data, 'backward'))"): "뒤→앞으로 읽기(rits_b, 다른 가중치) → 시간 순서를 다시 뒤집음",
             src.line(BRITS, "imputations = (ret_f['imputations'] + ret_b['imputations']) / 2"): "두 방향의 채운 값을 평균"},
      code=C(BRITS, src.span(BRITS, "def forward(self, data):", "return imputations"), src.line(BRITS, "ret_f = self.rits_f(data, 'forward')"), "BRITS.forward", 4, 18),
      stack=stk_brits, nodes=["fwd", "bwd", "merge"])
    ev_mask = tr.one(RITS, contains="masks = torch.logical_not", direct="forward")
    masks_f = ev_mask.produced["masks"]
    S("7. BRITS 앞→뒤",
      "먼저 어느 칸이 비었는지 표시합니다. <b>m = 1</b> 이면 관측된 칸, <b>m = 0</b> 이면 빈 칸(NaN)입니다. 이 m 은 뒤에서 손실을 관측 칸에서만 재고, 빈 칸만 추정값으로 바꾸는 데 쓰이며, "
      "LSTMCell 입력으로도 함께 들어갑니다([c_c ; m]).",
      say="먼저 어느 칸이 비었는지 표시합니다. m이 1이면 관측된 칸, 0이면 빈 칸입니다. 이 표시는 뒤에서 손실을 관측 칸에서만 재고, 빈 칸만 추정값으로 바꾸는 데 쓰이며, LSTM 셀의 입력으로도 함께 들어갑니다.",
      notes={ev_mask.line: f"NaN 이 아니면 1(관측), NaN 이면 0(빈 칸) — 이 배치의 빈 칸 {cap['batch']['n_nan']}개"},
      code=C(RITS, [src.line(RITS, "if direct == 'forward':"), src.line(RITS, "values = data", after=150), ev_mask.line], ev_mask.line, "RITS.forward", 4, 4),
      stack=stk_rits("forward"), nodes=["fwd"],
      formula="m = 1 − isnan(x)   (관측 1, 빈 칸 0)",
      tensors=[BH("masks m (흰색 0 = 빈 칸, 진한 색 1 = 관측)", masks_f, rows_w[0], scale="01")])
    steps[-1]["tensors"][0]["rowsEach"] = rows_w
    ev_d = tr.one(RITS, contains="deltas = self.delta_calc(masks)", direct="forward")
    dlt = ev_d.produced["deltas"]
    from models.biva_src.BIVA_RITS import Model as RitsM  # 원본 규칙 대조용(같은 모듈의 원본 함수)
    from types import SimpleNamespace
    rr = RitsM(SimpleNamespace(seq_len=L, channels=K, rnn_hid_size=4, imputed_weight=1.0))
    d_orig = rr.delta_calc_orig(masks_f)
    # 실제 예: 늘 비어 있는 변수와 그 오른쪽(관측) 변수
    dex = ""
    for j in range(1, K - 1):
        if masks_f[0, :, j].sum() == 0 and masks_f[0, :, j + 1].all():
            dex = (f"\n예) b=0 τ2: {feats[j]} 는 앞 시점 τ1 에 비어 있음 → δ = δ[{feats[j - 1]}] + 1 = {f(dlt[0, 1, j - 1])} + 1 = {f(dlt[0, 1, j])},  "
                   f"오른쪽 {feats[j + 1]} 는 관측인데 왼쪽 {feats[j]} 가 비어 ② 가 아니라 ③ → δ = {f(dlt[0, 1, j + 1])} (논문 정의면 1)")
            break
    S("7. BRITS 앞→뒤",
      "<b>δ(델타)</b>는 '마지막 관측 뒤로 얼마나 지났나'를 나타내려는 수입니다. 원본 delta_calc 의 규칙은 이렇습니다. 첫 시점은 0. 그다음부터는 "
      "① 같은 변수가 바로 앞 시점에 비었으면 <b>같은 시점 왼쪽 이웃 변수(d−1)의 δ + 1</b>, ② 아니고 왼쪽 이웃 변수가 지금 관측되었으면 1, ③ 그 밖에는 0. "
      "곧 원본 코드는 δ 값을 같은 변수의 앞 시점 δ 가 아니라 <b>같은 시점 바로 왼쪽 열의 δ</b> 에서 이어 받습니다. 그래서 같은 변수가 계속 비어 있어도 δ 가 시간에 따라 쌓이지 않고, "
      "이웃 변수의 빈 칸에 영향을 받습니다. 원본 그대로 두었습니다. 오른쪽은 이 규칙을 담은 원본 함수 delta_calc_orig 이고, 실제 실행은 값이 똑같은 빠른 버전(delta_calc, [TSP 수정 #6])으로 했습니다(아래 검증).",
      say=("델타는 마지막 관측 뒤로 얼마나 지났는지를 나타내려는 수입니다. 원본 코드의 규칙은 이렇습니다. 첫 시점은 0입니다. 그다음부터는, 첫째, 같은 변수가 바로 앞 시점에 비어 있었으면 "
           "같은 시점 왼쪽 이웃 변수의 델타에 1을 더합니다. 둘째, 그렇지 않고 왼쪽 이웃 변수가 지금 관측되었으면 1입니다. 셋째, 그 밖에는 0입니다. "
           "곧 원본 코드는 델타 값을 같은 변수의 앞 시점 델타가 아니라, 같은 시점 바로 왼쪽 열의 델타에서 이어 받습니다. 그래서 같은 변수가 계속 비어 있어도 델타가 시간에 따라 쌓이지 않고, "
           "이웃 변수의 빈 칸에 영향을 받습니다. 논문의 정의와 다르지만, 원본 그대로 두었습니다. "
           "오른쪽은 이 규칙을 담은 원본 함수이고, 실제 실행은 값이 똑같은 빠른 버전으로 했습니다."),
      notes={src.line(RITS, "delta[b,t,d] = delta[b,t,d-1]+t-(t-1)"): "같은 변수가 앞 시점에 비었으면: 왼쪽 이웃 변수(d−1)의 δ + 1  (t−(t−1) = 1)",
             src.line(RITS, "delta[b,t,d] = t - (t-1)"): "왼쪽 이웃 변수가 지금 관측되었으면 1 — 아무 조건도 아니면 0 그대로"},
      lab=lab_delta(masks_f[0].numpy().astype(int).tolist(), win[0], dlt[0].numpy().tolist()),
      code=C(RITS, src.span(RITS, "def delta_calc_orig(self, masks):", "return delta", after=130), src.line(RITS, "delta[b,t,d] = delta[b,t,d-1]+t-(t-1)"), "delta_calc_orig (원본 규칙)", 2, 2),
      stack=stk_rits("forward") + ["delta_calc"], nodes=["fwd"],
      formula="δ[t,d] = 0 (t = 0);  m[t−1,d] = 0 이면 δ[t,d−1] + 1;  아니고 m[t,d−1] = 1 이면 1;  그 밖은 0   (위에서부터 먼저 맞는 규칙)" + dex,
      tensors=[{**BH("deltas δ", dlt, rows_w[0], scale="auto01"), "rowsEach": rows_w}],
      checks=[chk("빠른 delta_calc(TSP 수정 #6) = 원본 delta_calc_orig", dlt, d_orig, 0)],
      hint="실행은 [TSP 수정 #6] delta_calc 로 했습니다. 이는 원본 delta_calc_orig 와 값이 완전히 같은 빠른 버전입니다(원본은 배치×시점×변수마다 파이썬 반복). 오른쪽 코드는 원본 규칙입니다. 같음은 위 검증 줄과 tests/test_biva_src.py 가 확인합니다.\n"
           "원본 규칙의 특징: d=0 열에서 d−1 = −1 은 파이썬 음수 색인이라 마지막 열(변수 34)을 가리킵니다. 그 시점의 마지막 열 δ 는 아직 계산 전이라 0 이고, 마스크는 마지막 변수의 것을 씁니다. "
           "BRITS 논문의 δ 정의(같은 변수의 마지막 관측 이후 지난 시간: 앞 시점이 관측이면 1, 비었으면 앞 δ + 1)와 다릅니다. 논문과 코드의 차이는 docs/biva_paper_vs_code.md.")
    ev_h0 = tr.one(RITS, contains="h = Variable(torch.zeros", direct="forward")
    body_ = [n for n in src.span(RITS, "x = values[:, t, :]", "imputations.append(c_c.unsqueeze(dim=1))") if src.text[RITS][n - 1].strip() and not src.text[RITS][n - 1].strip().startswith("#")]
    nb_ = len(body_)   # 시점마다 실행되는 줄 수(원본 BIVA_RITS.py 반복문 몸통)
    S("7. BRITS 앞→뒤",
      f"기억 <b>h</b> 와 셀 상태 <b>c</b> 를 0 으로 시작합니다. 모양은 [배치 {B}, 은닉 {p['rnn_hid_size']}] 입니다. 보간 손실 x_loss = 0, 채운 값을 모을 목록 imputations = [] 도 준비합니다. "
      f"이제 <b>for t in range({L})</b> 로 시점마다 같은 계산 {nb_}줄(빈 줄·주석 제외)을 반복합니다.",
      say=(f"기억 h와 셀 상태 c를 0으로 시작합니다. 모양은 배치 {B}, 은닉 크기 {p['rnn_hid_size']}입니다. 보간 손실도 0에서 시작하고, 채운 값을 모을 목록도 준비합니다. "
           f"이제 시점마다 같은 계산 {nb_}줄을 {L}번 반복합니다."),
      notes={ev_h0.line: f"기억 h = 0  [{B}, {p['rnn_hid_size']}]", ev_h0.line + 1: f"셀 상태 c = 0  [{B}, {p['rnn_hid_size']}]",
             src.line(RITS, "for t in range(self.seq_len):"): f"시점 t = 0 … {L - 1} (τ1 … τ{L}) 마다 아래 {nb_}줄"},
      code=C(RITS, [ev_h0.line, ev_h0.line + 1, src.line(RITS, "x_loss = 0.0"), src.line(RITS, "imputations = []", after=170), src.line(RITS, "for t in range(self.seq_len):")], ev_h0.line, "RITS.forward", 3, 4),
      stack=stk_rits("forward"), nodes=["fwd"],
      tensors=[{"type": "num", "title": "시작 상태", "items": [{"label": "h", "value": f"[{B}, {p['rnn_hid_size']}] 모두 0"}, {"label": "c", "value": f"[{B}, {p['rnn_hid_size']}] 모두 0"}, {"label": "x_loss", "value": "0"}]}])

    # ---- t = 0 자세히
    def rv(name, contains, direct, t, nth=0):
        return tr.p(name, RITS, contains=contains, direct=direct, t=t, nth=nth)
    x0, m0, d0 = rv("x", "x = values[:, t, :]", "forward", 0), rv("m", "m = masks[:, t, :]", "forward", 0), rv("d", "d = deltas[:, t, :]", "forward", 0)
    rows_b = [f"b={i}" for i in range(B)]
    S("7. BRITS 앞→뒤",
      "<b>τ1 (t = 0)</b>: 창 [3, 6, 34] 에서 시점 하나를 잘라 냅니다. <b>values[:, t, :]</b> 는 '모든 배치, 시점 t, 모든 변수'라는 뜻이고, 모양은 [3, 34] 입니다. "
      "같은 방법으로 m(관측 표시)과 d(δ)도 잘라 냅니다. 위 그림의 주황 테두리가 잘라 낸 행입니다.",
      say=(f"τ1, 첫 시점입니다. 창 묶음에서 시점 하나를 잘라 냅니다. 모든 배치의 첫 시점, 모든 변수를 꺼내므로 모양은 배치 {B}, 변수 {K}입니다. "
           "같은 방법으로 관측 표시 m과 델타 d도 잘라 냅니다. 위 그림의 주황 테두리가 잘라 낸 행입니다."),
      notes={src.line(RITS, "x = values[:, t, :]"): f"모든 배치 · 시점 t · 모든 변수 → [{B}, {K}]", src.line(RITS, "m = masks[:, t, :]"): "같은 시점의 관측 표시",
             src.line(RITS, "d = deltas[:, t, :]"): "같은 시점의 δ"},
      code=C(RITS, src.lines(RITS, "x = values[:, t, :]", "m = masks[:, t, :]", "d = deltas[:, t, :]"), src.line(RITS, "x = values[:, t, :]"), "RITS.forward", 3, 4),
      stack=stk_rits("forward", 0), nodes=["fwd"],
      formula="x = values[:, 0, :]  →  [3, 6, 34] 에서 τ1 행만 → [3, 34]",
      tensors=[{**BH("values (주황 = 지금 자르는 τ1)", xb, rows_w[0], scale="01", hl={"rows": [0]}), "rowsEach": rows_w},
               H("x = values[:, 0, :]", x0, rows=rows_b, cols=feats, shape=[B, K], scale="01"), H("m (관측 1 / 빈 칸 0)", m0, rows=rows_b, cols=feats, shape=[B, K], scale="01"),
               H("d (δ)", d0, rows=rows_b, cols=feats, shape=[B, K], scale="auto01")])
    gh0 = rv("gamma_h", "gamma_h = self.temp_decay_h(d)", "forward", 0)
    Wdh, bdh = sd["BRITS.rits_f.temp_decay_h.W"].numpy(), sd["BRITS.rits_f.temp_decay_h.b"].numpy()
    gh_hand = np.exp(-np.maximum(d0.double().numpy() @ Wdh.T + bdh, 0))
    neg_pct = round(100 * float((Wdh < 0).mean()))            # temp_decay_h.W 원소 중 음수 비율(%)
    pos_rows = int((Wdh >= 0).all(axis=1).sum())               # 원소가 모두 0 이상인 행(은닉 단위) 수
    tg = max(range(1, L), key=lambda t: float(rv("d", "d = deltas[:, t, :]", "forward", t).std()))  # δ 가 가장 다양한 시점
    dg, ghg = rv("d", "d = deltas[:, t, :]", "forward", tg), rv("gamma_h", "gamma_h = self.temp_decay_h(d)", "forward", tg)
    gb_, gu_ = [int(v) for v in np.unravel_index(int(np.argmin(ghg.numpy())), tuple(ghg.shape))]   # 실험 코드: 기억을 가장 많이 줄인 (배치, 은닉 단위)
    S("7. BRITS 앞→뒤",
      f"<b>γ_h(감마)</b>는 기억 h 를 은닉 단위마다 얼마나 줄일지 정하는 0~1 사이 값입니다(TemporalDecay). δ 에 가중치 W [{p['rnn_hid_size']}×{K}] 를 곱하고 b 를 더한 뒤, 음수는 0 으로 자르고(ReLU) exp(−·) 를 합니다. "
      f"W·δ + b 가 클수록 값이 작아져 기억을 많이 줄입니다. δ 가 커질 때 γ_h 가 작아질지 커질지는 W 의 부호에 달려 있습니다 — 되돌린 가중치(epoch {best_ep})에서 W 원소의 {neg_pct}% 가 음수이고, "
      f"원소가 모두 0 이상인 행은 {p['rnn_hid_size']}개 중 {pos_rows}개입니다. 그다음 <b>h = h × γ_h</b> 로 기억을 줄입니다. τ1 에서는 h 가 아직 0 이라 그대로 0 입니다.",
      say=("감마 에이치는 기억 h를 은닉 단위마다 얼마나 줄일지 정하는, 0에서 1 사이의 값입니다. 델타에 가중치를 곱하고 편향을 더한 뒤, 음수는 0으로 자르고, 마이너스를 붙여 지수 함수를 씌웁니다. "
           f"이 합이 클수록 값이 작아져서 기억을 많이 줄입니다. 델타가 커질 때 감마가 작아질지 커질지는 가중치의 부호에 달려 있는데, 되돌린 가중치에서는 원소의 {neg_pct}%가 음수입니다. "
           "그다음 기억 h에 감마 에이치를 곱해 기억을 줄입니다. τ1에서는 h가 아직 0이라 그대로 0입니다."),
      notes={src.line(RITS, "gamma_h = self.temp_decay_h(d)"): f"γ_h = exp(−max(0, W·δ + b))  [{B}, {p['rnn_hid_size']}]", src.line(RITS, "h = h * gamma_h"): "기억을 γ_h 만큼 줄임(τ1 은 h = 0 이라 0)",
             src.line(RITS, "gamma = F.relu(F.linear(d, self.W, self.b))"): "W·δ + b 의 음수를 0 으로(ReLU)", src.line(RITS, "gamma = torch.exp(-gamma)"): "exp(−·) → 0~1 (W·δ + b 가 클수록 작음)"},
      lab=lab_gamma(dg[gb_].numpy(), Wdh[gu_], bdh[gu_], tg + 1, win[gb_], gu_ + 1, ghg[gb_, gu_]),
      code=C(RITS, [src.line(RITS, "gamma_h = self.temp_decay_h(d)"), src.line(RITS, "h = h * gamma_h")] + src.span(RITS, "def forward(self, d):", "return gamma"),
             src.line(RITS, "gamma_h = self.temp_decay_h(d)"), "RITS.forward → TemporalDecay.forward", 2, 2),
      stack=stk_rits("forward", 0) + ["TemporalDecay(diag=False)"], nodes=["fwd"],
      formula=f"γ_h = exp(−max(0, W·d + b))  W: [{p['rnn_hid_size']}, {K}],  h ← h ⊙ γ_h",
      tensors=[H("τ1 의 γ_h (δ = 0 이라 세 배치가 같음)", gh0, rows=rows_b, shape=[B, p["rnn_hid_size"]], scale="01", numbers=False, cellH=18),
               H(f"비교: τ{tg + 1} 의 δ", dg, rows=rows_b, cols=feats, shape=[B, K], scale="auto01"),
               H(f"비교: τ{tg + 1} 의 γ_h (δ 가 다르면 줄이는 정도도 다름)", ghg, rows=rows_b, shape=[B, p["rnn_hid_size"]], scale="01", numbers=False, cellH=18),
               {"type": "num", "title": "γ_h 평균(1 에 가까울수록 기억을 덜 줄임)", "items": [{"label": f"τ1 b={i}", "value": f(gh0[i].mean())} for i in range(B)] + [{"label": f"τ{tg + 1} b={i}", "value": f(ghg[i].mean())} for i in range(B)]}],
      checks=[chk("γ_h 손 계산 = 원본 출력", gh0, gh_hand), chk(f"τ{tg + 1} γ_h 손 계산 = 원본 출력", ghg, np.exp(-np.maximum(dg.double().numpy() @ Wdh.T + bdh, 0)))])
    gx0 = rv("gamma_x", "gamma_x = self.temp_decay_x(d)", "forward", 0)
    al0 = rv("alpha", "alpha = self.weight_combine", "forward", 0)
    S("7. BRITS 앞→뒤",
      "<b>γ_x</b> 도 같은 방식이지만 가중치 W 의 대각선만 씁니다(diag=True, 변수마다 자기 δ 만 봄). "
      "원본에서 γ_x 는 뒤에 나오는 BRITS 안의 <b>alpha = weight_combine([γ_x, m])</b>(「17. 합치기」의 학습되는 α 와 다른 값)(시그모이드 없는 Linear — 0~1 비율이 아님) 계산에만 들어갑니다. 그런데 이 alpha 는 <b>쓰이지 않습니다</b>. "
      "섞는 줄이 주석 처리되어 있고, 대신 고정된 반반(0.5·ẑ + 0.5·x̂)을 쓰기 때문입니다. 계산은 하지만 결과에 영향이 없는 줄입니다.",
      say=("감마 엑스도 같은 방식이지만, 가중치의 대각선만 써서 변수마다 자기 델타만 봅니다. 원본에서 감마 엑스는 바로 뒤에 나오는 BRITS 안의 알파를 계산하는 데에만 들어갑니다. "
           "이 알파는 뒤에 나오는 합치기 알파와는 다른 값입니다. 논문에서는 이 값에 시그모이드를 씌워 두 추정을 섞는 비율로 쓰지만, 이 코드는 시그모이드도 없고 알파를 쓰지도 않습니다. "
           "섞는 줄이 주석 처리되어 있고, 대신 고정된 반반을 쓰기 때문입니다. 계산은 하지만 결과에는 영향이 없는 줄입니다."),
      notes={src.line(RITS, "gamma_x = self.temp_decay_x(d)"): "γ_x (대각선 가중치 — 변수마다 자기 δ 만)", src.line(RITS, "alpha = self.weight_combine"): "alpha = Linear([γ_x, m]) — 시그모이드 없음(0~1 아님), 계산되지만 쓰이지 않음",
             src.line(RITS, "# c_h = alpha * z_h + (1 - alpha) * x_h"): "논문의 섞기(alpha 사용) — 원본에서 주석 처리됨"},
      code=C(RITS, [src.line(RITS, "gamma_x = self.temp_decay_x(d)"), src.line(RITS, "alpha = self.weight_combine"), src.line(RITS, "# c_h = alpha * z_h + (1 - alpha) * x_h")],
             src.line(RITS, "alpha = self.weight_combine"), "RITS.forward", 14, 4),
      stack=stk_rits("forward", 0), nodes=["fwd"],
      tensors=[H("γ_x (변수별, 0~1)", gx0, rows=rows_b, cols=feats, shape=[B, K], scale="01"), H("alpha (계산되지만 쓰이지 않음)", al0, rows=rows_b, cols=feats, shape=[B, K])],
      hint="BRITS 논문은 β(여기서 alpha) = σ(W[γ_x, m] + b) 로 두 추정을 섞습니다. 이 원본 코드는 그 줄을 주석으로 두고 |0.5·ẑ + 0.5·x̂| 를 씁니다. 그래서 weight_combine·temp_decay_x 의 가중치는 출력에 이어지지 않아 역전파 때 기울기가 아예 생기지 않고(grad = None), Adam 이 건너뛰어 처음 값 그대로 남습니다. 원본 그대로 두었습니다.")
    xh0 = rv("x_h", "x_h = F.sigmoid(x_h)", "forward", 0)
    # 실험 코드: τ2 의 x̂ — 첫 창에서 τ2 에 비어 있는 첫 변수
    m1_ = rv("m", "m = masks[:, t, :]", "forward", 1).numpy()
    jx_ = int(np.where(m1_[0] == 0)[0][0]) if (m1_[0] == 0).any() else 0
    _W, _b = sd["BRITS.rits_f.hist_reg.weight"].numpy(), sd["BRITS.rits_f.hist_reg.bias"].numpy()
    lab_xhat_ = lab_xhat(tr.p("h", RITS, contains="h, c = self.rnn_cell(inputs, (h, c))", direct="forward", t=0)[0].numpy(),
                         rv("gamma_h", "gamma_h = self.temp_decay_h(d)", "forward", 1)[0].numpy(), _W[jx_], _b[jx_], feats[jx_], win[0],
                         rv("x_h", "x_h = F.sigmoid(x_h)", "forward", 1)[0, jx_])
    Whr, bhr = sd["BRITS.rits_f.hist_reg.weight"].numpy(), sd["BRITS.rits_f.hist_reg.bias"].numpy()
    S("7. BRITS 앞→뒤",
      f"<b>x̂(기억으로 추정)</b>: 줄인 기억 h [3, {p['rnn_hid_size']}] 를 Linear(hist_reg, {p['rnn_hid_size']}→{K})에 넣고 sigmoid 로 0~1 로 만듭니다. "
      "'지금까지 본 흐름으로 보면 이번 달 각 변수는 이 정도'라는 추정입니다. τ1 에서는 h = 0 이라 편향 b 만 남아, 세 배치가 <b>같은 값</b>입니다.",
      say=(f"x̂는 기억으로 만든 추정입니다. 줄인 기억 h를 선형 층에 넣어 변수 {K}개로 바꾸고, 시그모이드로 0에서 1 사이로 만듭니다. "
           "지금까지 본 흐름으로 보면 이번 달 각 변수는 이 정도라는 추정입니다. τ1에서는 h가 0이라 편향만 남아서, 세 배치가 같은 값입니다."),
      notes={src.line(RITS, "x_h = self.hist_reg(h)"): f"Linear {p['rnn_hid_size']}→{K}: W_hist · h + b_hist", src.line(RITS, "x_h = F.sigmoid(x_h)"): "sigmoid → 0~1 (τ1 은 σ(b_hist))"},
      lab=lab_xhat_,
      code=C(RITS, src.lines(RITS, "x_h = self.hist_reg(h)", "x_h = F.sigmoid(x_h)"), src.line(RITS, "x_h = self.hist_reg(h)"), "RITS.forward", 3, 4),
      stack=stk_rits("forward", 0), nodes=["fwd"],
      formula=f"x̂ = σ(W_hist · h + b_hist)   W_hist: [{K}, {p['rnn_hid_size']}]   (τ1: h = 0 → x̂ = σ(b_hist))",
      tensors=[H("x̂ = x_h (기억으로 추정)", xh0, rows=rows_b, cols=feats, shape=[B, K], scale="01"), H("x (실제, NaN 빗금)", x0, rows=rows_b, cols=feats, shape=[B, K], scale="01")],
      checks=[chk("x̂ 손 계산 σ(b) = 원본 출력(τ1, h=0)", xh0, np.tile(SIG(bhr), (B, 1)))])
    xl1 = float(tr.p("x_loss", RITS, contains="torch.abs(torch.nan_to_num(x) - x_h)", direct="forward", t=0))
    x0n, m0n = np.nan_to_num(x0.double().numpy()), m0.double().numpy()
    l1 = np.sum(np.abs(x0n - xh0.double().numpy()) * m0n) / (m0n.sum() + 1e-5)
    S("7. BRITS 앞→뒤",
      "<b>보간 손실 ①</b>: 관측된 칸에서 x̂ 가 실제와 얼마나 틀렸는지 잽니다. |x − x̂| 를 관측 칸(m = 1)에서만 더한 뒤 관측 칸 수로 나눕니다(배치 3개 전체를 한꺼번에). "
      f"이 값은 나중에 imputed_weight {p['imputed_weight']} 가 곱해져 전체 손실에 더해지고, 학습은 그 전체 손실을 줄이는 방향으로 hist_reg 등을 고칩니다. 빈 칸은 정답을 모르므로 손실에 넣지 않습니다.",
      say=(f"보간 손실의 첫 번째 부분입니다. 관측된 칸에서 x̂가 실제와 얼마나 틀렸는지 잽니다. 실제 값과 x̂의 차이의 절댓값을 관측 칸에서만 더한 뒤, 관측 칸 수 {int(m0n.sum())}로 나눕니다. "
           f"이번 시점에서는 {f(l1)}입니다. 이 값은 나중에 {p['imputed_weight']}가 곱해져 전체 손실에 더해지고, 학습은 그 전체 손실을 줄이는 방향으로 가중치를 고칩니다. 빈 칸은 정답을 모르므로 손실에 넣지 않습니다."),
      notes={src.line(RITS, "x_loss += torch.sum(torch.abs(torch.nan_to_num(x) - x_h) * m)"): f"관측 칸만: Σ|x − x̂|·m ÷ (관측 칸 {int(m0n.sum())} + 1e−5) = {f(l1)}"},
      code=C(RITS, [src.line(RITS, "x_loss += torch.sum(torch.abs(torch.nan_to_num(x) - x_h) * m)")], src.line(RITS, "x_loss += torch.sum(torch.abs(torch.nan_to_num(x) - x_h) * m)"), "RITS.forward", 3, 4),
      stack=stk_rits("forward", 0), nodes=["fwd"],
      formula=f"x_loss += Σ|x − x̂|·m ÷ (Σm + 1e−5) = {f(np.sum(np.abs(x0n - xh0.double().numpy()) * m0n))} ÷ ({int(m0n.sum())} + 1e−5) = {f(l1)}",
      tensors=[H("|x − x̂| · m (관측 칸의 틀린 정도)", np.abs(x0n - xh0.double().numpy()) * m0n, rows=rows_b, cols=feats, shape=[B, K], scale="auto01")],
      checks=[chk("보간 손실 ① 손 계산 = 원본 x_loss", xl1, l1)])
    xc0 = rv("x_c", "x_c = m * torch.nan_to_num(x) + (1 - m) * x_h", "forward", 0)
    miss0 = [[int(i), int(j)] for i, j in zip(*np.where(m0n == 0))]
    S("7. BRITS 앞→뒤",
      "<b>x_c</b>: 관측 칸은 실제 값, 빈 칸은 방금의 x̂ 로 채운 임시 행입니다. 빨간 테두리가 x̂ 로 채운 칸입니다. "
      "이 임시 행은 다음 단계에서 '다른 변수로 추정'할 재료가 됩니다.",
      say=("이번에는 임시 행을 만듭니다. 관측 칸은 실제 값, 빈 칸은 방금 만든 x̂로 채웁니다. 빨간 테두리가 x̂로 채운 칸입니다. "
           "이 임시 행은 다음 단계에서 다른 변수로 추정할 때 재료가 됩니다."),
      notes={src.line(RITS, "x_c = m * torch.nan_to_num(x) + (1 - m) * x_h"): f"관측 칸 = 실제 값, 빈 칸 = x̂ (이번 시점 빈 칸 {len(miss0)}개)"},
      code=C(RITS, [src.line(RITS, "x_c = m * torch.nan_to_num(x) + (1 - m) * x_h")], src.line(RITS, "x_c = m * torch.nan_to_num(x) + (1 - m) * x_h"), "RITS.forward", 3, 4),
      stack=stk_rits("forward", 0), nodes=["fwd"],
      formula="x_c = m ⊙ x + (1 − m) ⊙ x̂",
      tensors=[H("x_c (빨강 = x̂ 로 채운 빈 칸)", xc0, rows=rows_b, cols=feats, shape=[B, K], scale="01", hl={"cells": miss0})])
    zh0 = rv("z_h", "z_h = self.feat_reg(x_c)", "forward", 0)
    Wf, bf = sd["BRITS.rits_f.feat_reg.W"].numpy(), sd["BRITS.rits_f.feat_reg.b"].numpy()
    mfeat = 1 - np.eye(K)
    zh_hand = SIG(xc0.double().numpy() @ (Wf * mfeat).T + bf)
    S("7. BRITS 앞→뒤",
      f"<b>ẑ(다른 변수로 추정)</b>: x_c 에 가중치 W [{K}×{K}] 를 곱해 각 변수를 추정합니다. 이때 W 의 <b>대각선을 0 으로 막아</b>(막 self.m = 1 − I — 관측 표시 m 과는 다른 변수) 자기 자신은 쓰지 않습니다. "
      f"'같은 달의 다른 {K - 1}개 변수(빈 칸은 x̂ 으로 채운 값)로 보면 이 변수는 이 정도'라는 추정이고, sigmoid 로 0~1 로 만듭니다.",
      say=(f"ẑ는 같은 달의 다른 변수로 만든 추정입니다. 임시 행에 {K} 곱하기 {K} 가중치를 곱해 각 변수를 추정하는데, 가중치의 대각선을 0으로 막아 자기 자신은 쓰지 않습니다. "
           f"같은 달의 다른 {K - 1}개 변수로 보면 이 변수는 이 정도라는 추정이고, 시그모이드로 0에서 1 사이로 만듭니다. 다른 변수 가운데 빈 칸은 앞에서 x̂로 채운 값이 쓰입니다."),
      notes={src.line(RITS, "z_h = self.feat_reg(x_c)"): f"다른 변수로 추정 [{B}, {K}]", src.line(RITS, "z_h = F.linear(x, self.W * Variable(self.m), self.b)"): "W ⊙ (1 − I): 대각선(자기 자신) 0",
             src.line(RITS, "m = torch.ones(input_size, input_size)"): "1 로 채운 뒤 대각선을 0 으로 만드는 막(mask)"},
      code=C(RITS, [src.line(RITS, "z_h = self.feat_reg(x_c)")] + src.span(RITS, "def forward(self, x):", "return z_h") + [src.line(RITS, "m = torch.ones(input_size, input_size)"), src.line(RITS, "m = torch.ones(input_size, input_size)") + 1],
             src.line(RITS, "z_h = F.linear(x, self.W * Variable(self.m), self.b)"), "FeatureRegression.forward", 3, 3),
      stack=stk_rits("forward", 0) + ["FeatureRegression"], nodes=["fwd"],
      formula=f"ẑ = σ((W ⊙ (1 − I)) · x_c + b)   W: [{K}, {K}], 대각선 = 0",
      tensors=[H("ẑ = z_h (다른 변수로 추정)", zh0, rows=rows_b, cols=feats, shape=[B, K], scale="01"),
               H("실제로 쓰인 W ⊙ (1 − I) (대각선 0)", Wf * mfeat, rows=feats, cols=feats, shape=[K, K], numbers=False, cellH=7, rowLabelW=70)],
      checks=[chk("ẑ 손 계산 = 원본 출력", zh0, zh_hand)])
    xl2 = float(tr.p("x_loss", RITS, contains="torch.abs(torch.nan_to_num(x) - z_h)", direct="forward", t=0))
    ch0 = rv("c_h", "c_h = torch.abs(0.5 * z_h + 0.5 * x_h)", "forward", 0)
    cc0 = rv("c_c", "c_c = m * torch.nan_to_num(x) + (1 - m) * c_h", "forward", 0)
    if miss0:
        bi_, vj = miss0[0]
        ex = f"예: b={bi_}·{feats[vj]}: x̂ {f(xh0[bi_, vj])}, ẑ {f(zh0[bi_, vj])} → |0.5×{f(zh0[bi_, vj])} + 0.5×{f(xh0[bi_, vj])}| = {f(ch0[bi_, vj])}"
    else:
        ex = "이 시점에는 빈 칸이 없습니다"
    S("7. BRITS 앞→뒤",
      "먼저 보간 손실 ②(관측 칸에서 ẑ 가 틀린 정도)를 x_loss 에 더합니다. 그다음 두 추정을 <b>반반 섞어</b> ĉ = |0.5·ẑ + 0.5·x̂| 를 만들고, <b>빈 칸에만</b> 넣습니다(c_c). "
      "관측 칸은 실제 값 그대로입니다. 빨간 테두리 칸이 이 시점에 채운 값입니다.",
      say=("먼저 관측 칸에서 ẑ가 틀린 정도를 보간 손실에 더합니다. 그다음 두 추정을 반반 섞습니다. ẑ와 x̂를 절반씩 더하고 절댓값을 씌운 값을 빈 칸에만 넣습니다. "
           "관측 칸은 실제 값 그대로입니다. 빨간 테두리 칸이 이 시점에 채운 값입니다. "
           + (f"예를 들어 {['첫', '두', '세'][miss0[0][0]]} 번째 창의 변수 {feats[miss0[0][1]]}는 x̂가 {f(xh0[miss0[0][0], miss0[0][1]])}, ẑ가 {f(zh0[miss0[0][0], miss0[0][1]])}입니다. "
              f"그래서 채운 값은 {f(ch0[miss0[0][0], miss0[0][1]])}입니다." if miss0 else "이 시점에는 빈 칸이 없습니다.")),
      notes={src.line(RITS, "x_loss += torch.sum(torch.abs(torch.nan_to_num(x) - z_h) * m)"): "보간 손실 ② (관측 칸에서 ẑ 가 틀린 정도)",
             src.line(RITS, "c_h = torch.abs(0.5 * z_h + 0.5 * x_h)"): "반반 섞기(고정) — 논문의 β(코드에서는 alpha, 위 줄에서 계산하지만 쓰지 않음) 대신",
             src.line(RITS, "c_c = m * torch.nan_to_num(x) + (1 - m) * c_h"): "빈 칸에만 ĉ, 관측 칸은 실제 값"},
      code=C(RITS, src.lines(RITS, "x_loss += torch.sum(torch.abs(torch.nan_to_num(x) - z_h) * m)", "c_h = torch.abs(0.5 * z_h + 0.5 * x_h)", "c_c = m * torch.nan_to_num(x) + (1 - m) * c_h"),
             src.line(RITS, "c_h = torch.abs(0.5 * z_h + 0.5 * x_h)"), "RITS.forward", 3, 4),
      stack=stk_rits("forward", 0), nodes=["fwd"],
      formula=f"ĉ = |0.5·ẑ + 0.5·x̂|,  c_c = m ⊙ x + (1 − m) ⊙ ĉ\n{ex}",
      tensors=[H("c_c (빨강 = 이번에 채운 칸)", cc0, rows=rows_b, cols=feats, shape=[B, K], scale="01", hl={"cells": miss0})],
      checks=[chk("ĉ 손 계산 |0.5ẑ + 0.5x̂| = 원본", ch0, np.abs(0.5 * zh0.double().numpy() + 0.5 * xh0.double().numpy())),
              chk("x_loss 누계 = ① + ② 손 계산", xl2, l1 + np.sum(np.abs(x0n - zh0.double().numpy()) * m0n) / (m0n.sum() + 1e-5))],
      hint="원본은 절댓값 |…| 을 씌웁니다. ẑ, x̂ 가 모두 sigmoid(0~1)라서 실제로는 절댓값이 바꾸는 것이 없습니다.")
    inp0 = rv("inputs", "inputs = torch.cat([c_c, m], dim=1)", "forward", 0)
    h_after = tr.p("h", RITS, contains="h, c = self.rnn_cell(inputs, (h, c))", direct="forward", t=0)
    c_after = tr.p("c", RITS, contains="h, c = self.rnn_cell(inputs, (h, c))", direct="forward", t=0)
    Wih, Whh = sd["BRITS.rits_f.rnn_cell.weight_ih"].numpy(), sd["BRITS.rits_f.rnn_cell.weight_hh"].numpy()
    bih, bhh = sd["BRITS.rits_f.rnn_cell.bias_ih"].numpy(), sd["BRITS.rits_f.rnn_cell.bias_hh"].numpy()
    Hn = p["rnn_hid_size"]
    g = inp0.double().numpy() @ Wih.T + bih + np.zeros((B, Hn)) @ Whh.T + bhh  # τ1: h = 0, c = 0
    i_, f_, gg, o_ = SIG(g[:, :Hn]), SIG(g[:, Hn:2 * Hn]), np.tanh(g[:, 2 * Hn:3 * Hn]), SIG(g[:, 3 * Hn:])
    c_hand = i_ * gg; h_hand = o_ * np.tanh(c_hand)
    lab_lstm_ = lab_lstmcell(inp0[0].numpy(), Wih[0], Wih[Hn], Wih[2 * Hn], Wih[3 * Hn], bih[0] + bhh[0], bih[Hn] + bhh[Hn], bih[2 * Hn] + bhh[2 * Hn], bih[3 * Hn] + bhh[3 * Hn],
                             win[0], 1, c_after[0, 0], h_after[0, 0])
    S("7. BRITS 앞→뒤",
      f"채운 행 c_c 와 관측 표시 m 을 옆으로 이어 붙여(<b>torch.cat(dim=1)</b>) [3, {2 * K}] 을 만들고, <b>LSTMCell</b>({2 * K}→{Hn})에 넣어 기억 h·c 를 새로 씁니다. "
      f"LSTMCell 은 입력 문 i, 잊기 문 f, 새 정보 g, 출력 문 o 를 계산해 c ← f⊙c + i⊙g, h ← o⊙tanh(c) 로 갱신합니다(여기서 i·f·o 는 σ 를, g 는 tanh 를 씌운 뒤의 값). 이 h 가 다음 시점의 '지난 기억'이 됩니다. "
      f"채운 행 c_c 는 imputations 목록에 쌓입니다. 이것으로 τ1 의 {nb_}줄이 끝납니다.",
      say=(f"채운 행과 관측 표시를 옆으로 이어 붙여 {2 * K}개짜리 입력을 만들고, LSTM 셀에 넣어 기억 h와 c를 새로 씁니다. LSTM 셀은 입력 문, 잊기 문, 새 정보, 출력 문을 계산합니다. "
           "새 셀 상태는 잊기 문 곱하기 지난 셀 상태에, 입력 문 곱하기 새 정보를 더한 것이고, 새 기억은 출력 문 곱하기, 셀 상태에 하이퍼볼릭 탄젠트를 씌운 값입니다. "
           "이 h가 다음 시점의 지난 기억이 됩니다. 채운 행은 목록에 쌓입니다. 이것으로 τ1의 계산이 끝납니다."),
      notes={src.line(RITS, "inputs = torch.cat([c_c, m], dim=1)"): f"[채운 행 ; 관측 표시] → [{B}, {2 * K}]", src.line(RITS, "h, c = self.rnn_cell(inputs, (h, c))"): f"LSTMCell {2 * K}→{Hn}: 문 4개로 c·h 갱신",
             src.line(RITS, "imputations.append(c_c.unsqueeze(dim=1))"): "이 시점의 채운 행을 쌓음"},
      lab=lab_lstm_,
      code=C(RITS, src.lines(RITS, "inputs = torch.cat([c_c, m], dim=1)", "h, c = self.rnn_cell(inputs, (h, c))", "imputations.append(c_c.unsqueeze(dim=1))") + [src.line(RITS, "self.rnn_cell = nn.LSTMCell(self.input_size*2, self.rnn_hid_size)")],
             src.line(RITS, "h, c = self.rnn_cell(inputs, (h, c))"), "RITS.forward (LSTMCell)", 3, 4),
      stack=stk_rits("forward", 0) + ["nn.LSTMCell"], nodes=["fwd"],
      formula=f"[i f g o] = W_ih·[c_c; m] + b_ih + W_hh·h + b_hh   (W_ih: [{4 * Hn}, {2 * K}], W_hh: [{4 * Hn}, {Hn}])\nc ← σ(f)⊙c + σ(i)⊙tanh(g),  h ← σ(o)⊙tanh(c)",
      tensors=[H("inputs = [c_c ; m]", inp0, rows=rows_b, shape=[B, 2 * K], numbers=False, cellH=18, scale="01"), H("새 기억 h", h_after, rows=rows_b, shape=[B, Hn], numbers=False, cellH=18),
               {"type": "table", "digits": 4, "title": "b=0, 은닉 단위 1~6 의 문 값 (손 계산, i·f·o 는 σ 뒤, g 는 tanh 뒤)", "columns": ["단위", "입력 문 i", "잊기 문 f", "새 정보 g", "출력 문 o", "셀 상태 c", "기억 h"],
                "rows": [[f"u{u + 1}", num(i_[0, u]), num(f_[0, u]), num(gg[0, u]), num(o_[0, u]), num(c_hand[0, u]), num(h_hand[0, u])] for u in range(6)]}],
      checks=[chk("LSTMCell 손 계산 h = 원본", h_after, h_hand), chk("LSTMCell 손 계산 c = 원본", c_after, c_hand)])

    # ---- t = 1..L-1 요약 (시점마다 한 단계)
    def compact(direct, t, b=0):
        g_ = lambda n, c: tr.p(n, RITS, contains=c, direct=direct, t=t)
        xs, ms, ds_ = g_("x", "x = values[:, t, :]"), g_("m", "m = masks[:, t, :]"), g_("d", "d = deltas[:, t, :]")
        xh, zh, cc = g_("x_h", "x_h = F.sigmoid(x_h)"), g_("z_h", "z_h = self.feat_reg(x_c)"), g_("c_c", "c_c = m * torch.nan_to_num(x) + (1 - m) * c_h")
        hh = tr.p("h", RITS, contains="h, c = self.rnn_cell(inputs, (h, c))", direct=direct, t=t)
        return xs, ms, ds_, xh, zh, cc, hh
    loss_f_steps = []
    for t in range(1, L):
        xs, ms, ds_, xh, zh, cc, hh = compact("forward", t)
        xlt = float(tr.p("x_loss", RITS, contains="torch.abs(torch.nan_to_num(x) - z_h)", direct="forward", t=t))
        loss_f_steps.append(xlt)
        mn_ = ms.numpy()
        miss = [[int(i), int(j)] for i, j in zip(*np.where(mn_ == 0))]
        rows7 = ["x 실제", "m 관측", "δ", "x̂ 기억 추정", "ẑ 변수 추정", "c_c 채운 값"]
        mats = [np.stack([xs[b_].numpy(), ms[b_].numpy(), ds_[b_].numpy(), xh[b_].numpy(), zh[b_].numpy(), cc[b_].numpy()]) for b_ in range(B)]
        fills = ", ".join(f"b{i}·{feats[j]} {f(cc[i, j], 3)}" for i, j in miss[:6]) + (f" 외 {len(miss) - 6}칸" if len(miss) > 6 else "")
        S("7. BRITS 앞→뒤",
          f"<b>τ{t + 1} (t = {t})</b>: 같은 {nb_}줄을 반복합니다. 이번에는 h 가 앞 시점의 기억이라, x̂ 가 배치마다 다릅니다. "
          + (f"빈 칸 {len(miss)}개를 채웠습니다: {fills}. " if miss else "이 시점의 세 창에는 빈 칸이 없어 채운 값이 실제와 같습니다. ")
          + f"보간 손실 누계 x_loss = {f(xlt)}.",
          say=(f"τ{t + 1}입니다. 같은 계산을 반복합니다. 이번에는 h가 앞 시점의 기억이라서, x̂가 배치마다 다릅니다. "
               + (f"빈 칸 {len(miss)}개를 채웠습니다. 예를 들어 {['첫', '두', '세'][miss[0][0]]} 번째 창의 변수 {feats[miss[0][1]]}는 {f(cc[miss[0][0], miss[0][1]], 3)}로 채웠습니다. " if miss else
                  "이 시점의 세 창에는 빈 칸이 없어서, 채운 값이 실제와 같습니다. ")
               + f"보간 손실 누계는 {f(xlt)}입니다."),
          notes={src.line(RITS, "c_c = m * torch.nan_to_num(x) + (1 - m) * c_h"): f"τ{t + 1}: 빈 칸 {len(miss)}개를 채움", src.line(RITS, "h, c = self.rnn_cell(inputs, (h, c))"): "앞 시점의 기억 h 를 받아 갱신"},
          code=C(RITS, src.span(RITS, "x = values[:, t, :]", "imputations.append(c_c.unsqueeze(dim=1))"), src.line(RITS, "c_c = m * torch.nan_to_num(x) + (1 - m) * c_h"), "RITS.forward", 1, 2),
          stack=stk_rits("forward", t), nodes=["fwd"],
          tensors=[{"type": "batch", "title": f"τ{t + 1} 에서 계산된 값 (행: 계산 순서, 색은 0~1 기준 — δ 는 숫자로 보세요)", "mats": [arr(m_) for m_ in mats], "cellH": 19,
                    "labels": [f"b={i} — 창 #{win[i]} 의 τ{t + 1} = {wdates[i][t]}" for i in range(B)], "rows": rows7, "cols": feats, "colNames": colnames, "scale": "01",
                    "hl": {}, "shape": [B, K]},
                   H("새 기억 h (LSTMCell 출력)", hh, rows=rows_b, shape=[B, Hn], numbers=False, cellH=12)])
        for i_b in range(B):
            steps[-1]["tensors"][0].setdefault("hlEach", []).append({"cells": [[5, j] for i, j in miss if i == i_b]})
    imp_f = tr.p("imputations", RITS, contains="imputations = torch.cat(imputations, dim=1)", direct="forward")
    ret_f_loss = float(xlt) * p["imputed_weight"]
    miss_all = [[int(i), int(t_), int(j)] for i, t_, j in zip(*np.where(np.isnan(xb.numpy())))]
    S("7. BRITS 앞→뒤",
      f"반복이 끝나면 시점마다 쌓은 c_c 6개를 시간 축으로 이어 붙여(<b>torch.cat(dim=1)</b>) [3, 6, 34] 를 만듭니다. 이것이 앞→뒤 방향의 채운 창입니다. "
      f"손실은 x_loss × imputed_weight = {f(xlt)} × {p['imputed_weight']} = <b>{f(ret_f_loss)}</b> 입니다.",
      say=(f"반복이 끝나면 시점마다 쌓은 채운 행 {L}개를 시간 축으로 이어 붙여, 앞에서 뒤로 읽은 방향의 채운 창을 만듭니다. "
           f"이 방향의 손실은 보간 손실 누계 {f(xlt)}에 가중치 {p['imputed_weight']}를 곱한 {f(ret_f_loss)}입니다."),
      notes={src.line(RITS, "imputations = torch.cat(imputations, dim=1)"): f"채운 행 {L}개 → [{B}, {L}, {K}]",
             src.line(RITS, "return {'loss': x_loss * self.imputed_weight"): f"손실 = {f(xlt)} × {p['imputed_weight']} = {f(ret_f_loss)}"},
      code=C(RITS, src.lines(RITS, "imputations = torch.cat(imputations, dim=1)", "return {'loss': x_loss * self.imputed_weight"), src.line(RITS, "imputations = torch.cat(imputations, dim=1)"), "RITS.forward", 3, 3),
      stack=stk_rits("forward"), nodes=["fwd"],
      tensors=[{**BH("앞→뒤로 채운 창 (빨강 = 채운 칸)", imp_f, rows_w[0], scale="01"), "rowsEach": rows_w,
                "hlEach": [{"cells": [[t_, j] for i, t_, j in miss_all if i == b_]} for b_ in range(B)]}])

    # ================================================================== 8. 뒤→앞
    ev_flip = tr.one(RITS, contains="values = torch.flip(data,[1])", direct="backward")
    vflip = ev_flip.produced["values"]
    dlt_b = tr.p("deltas", RITS, contains="deltas = self.delta_calc(masks)", direct="backward")
    rows_rev = [[f"τ{L - t} {wdates[i][L - 1 - t]}" for t in range(L)] for i in range(B)]
    S("8. BRITS 뒤→앞",
      "뒤→앞 RITS(rits_b)는 먼저 창을 <b>시간 축으로 뒤집습니다</b>(torch.flip(data, [1])). 그러면 τ6 이 맨 앞, τ1 이 맨 뒤가 됩니다. "
      "그다음은 앞→뒤와 똑같은 코드가 <b>별도의 가중치</b>로 돕니다. 뒤집은 창에서 m 과 δ 를 다시 계산하므로 δ 값도 달라집니다.",
      say=(f"뒤에서 앞으로 읽는 쪽은 먼저 창을 시간 축으로 뒤집습니다. 그러면 τ{L}이 맨 앞, τ1이 맨 뒤가 됩니다. 그다음은 앞의 방향과 똑같은 코드가 별도의 가중치로 돕니다. "
           "뒤집은 창에서 관측 표시와 델타를 다시 계산하므로, 델타 값도 달라집니다."),
      notes={ev_flip.line: f"시간 축을 뒤집음: τ{L} … τ1", src.line(RITS, "deltas = self.delta_calc(masks)"): "뒤집은 창으로 δ 를 다시 계산"},
      code=C(RITS, [src.line(RITS, "else: # 'backward'"), ev_flip.line, src.line(RITS, "masks = torch.logical_not"), src.line(RITS, "deltas = self.delta_calc(masks)")], ev_flip.line, "RITS.forward(direct='backward')", 4, 6),
      stack=stk_rits("backward"), nodes=["bwd"], edges=[("batch", "bwd")],
      tensors=[{**BH("뒤집은 values (위에서부터 τ6 → τ1)", vflip, rows_rev[0], scale="01"), "rowsEach": rows_rev},
               {**BH("뒤집은 창의 δ", dlt_b, rows_rev[0], scale="auto01"), "rowsEach": rows_rev}])
    rows_bw = []
    for t in range(L):
        xs, ms, ds_, xh, zh, cc, hh = compact("backward", t)
        mn_ = ms.numpy()
        for i, j in zip(*np.where(mn_ == 0)):
            rows_bw.append([f"{t} (τ{L - t})", f"b{i}", feats[j], num(xh[i, j]), num(zh[i, j]), num(cc[i, j])])
    xl_b = float(tr.p("x_loss", RITS, contains="torch.abs(torch.nan_to_num(x) - z_h)", direct="backward", t=L - 1))
    imp_b_rev = tr.p("imputations", RITS, contains="imputations = torch.cat(imputations, dim=1)", direct="backward")
    S("8. BRITS 뒤→앞",
      f"뒤집은 순서로 t = 0 … {L - 1} 을 돕니다(τ6 → τ1). 빈 칸마다 x̂·ẑ 를 반씩 섞어 채우는 것은 같습니다. 다만 이번에는 <b>같은 입력 창 안의 나중 달 값들로 만든 기억</b>을 씁니다(창 밖의 값은 쓰지 않음). "
      f"표는 이 방향에서 채운 빈 칸 전부입니다({len(rows_bw)}칸). 손실 = {f(xl_b)} × {p['imputed_weight']} = {f(xl_b * p['imputed_weight'])}.",
      say=(f"뒤집은 순서로 {L}개 시점을 돕니다. 빈 칸마다 x̂와 ẑ를 반씩 섞어 채우는 것은 같습니다. 다만 이번에는 같은 입력 창 안의 나중 달 값들로 만든 기억을 씁니다. 창 밖의 값은 쓰지 않습니다. "
           f"표는 이 방향에서 채운 빈 칸 {len(rows_bw)}개 전부입니다. 이 방향의 손실은 {f(xl_b)}에 {p['imputed_weight']}를 곱한 {f(xl_b * p['imputed_weight'])}입니다."),
      notes={src.line(RITS, "c_c = m * torch.nan_to_num(x) + (1 - m) * c_h"): f"뒤→앞: 빈 칸 {len(rows_bw)}칸을 채움"},
      code=C(RITS, src.span(RITS, "for t in range(self.seq_len):", "return {'loss': x_loss * self.imputed_weight"), src.line(RITS, "c_c = m * torch.nan_to_num(x) + (1 - m) * c_h"), "RITS.forward(direct='backward')", 1, 1),
      stack=stk_rits("backward"), nodes=["bwd"],
      tensors=[{"type": "table", "split": 3, "digits": 3, "title": "뒤→앞 방향에서 채운 칸 (시점은 뒤집힌 순서의 t, 괄호는 원래 시점 τ)", "columns": ["t", "배치", "변수", "x̂", "ẑ", "채운 값"], "rows": rows_bw}])
    ev_rev = tr.one(BRITS, contains="return tensor_.index_select(1, indices)")
    imp_b = torch.flip(imp_b_rev, [1])
    S("8. BRITS 뒤→앞",
      "뒤→앞 결과는 시간 순서가 거꾸로이므로 <b>reverse()</b>가 다시 뒤집습니다. 시간 축의 번호를 [5, 4, 3, 2, 1, 0] 순서로 골라 담는 방식입니다(index_select). "
      "손실 같은 0차원 값은 그대로 둡니다. 이제 앞→뒤 결과와 같은 시간 순서(τ1→τ6)가 되었습니다.",
      say=("뒤에서 앞으로 읽은 결과는 시간 순서가 거꾸로이므로, 한 번 더 뒤집어 원래 순서로 되돌립니다. 시간 축의 번호를 거꾸로 골라 담는 방식입니다. "
           "손실처럼 숫자 하나인 값은 그대로 둡니다. 이제 앞에서 뒤로 읽은 결과와 같은 시간 순서가 되었습니다."),
      notes={ev_rev.line: f"번호 [{', '.join(str(i) for i in range(L - 1, -1, -1))}] 순서로 골라 담음 → τ1 … τ{L}"},
      code=C(BRITS, src.span(BRITS, "def reverse(self, ret):", "return ret"), ev_rev.line, "BRITS.reverse", 1, 2),
      stack=stk_brits + ["BRITS.reverse"], nodes=["bwd"],
      formula="indices = [5, 4, 3, 2, 1, 0],  결과 = imputations.index_select(1, indices)",
      tensors=[{**BH("되돌린 뒤→앞 결과 (τ1 → τ6)", imp_b, rows_w[0], scale="01"), "rowsEach": rows_w}])

    # ================================================================== 9. 합치기
    lc = float(tr.p("loss_c", BRITS, contains="loss_c = self.get_consistency_loss"))
    lsum = float(tr.p("loss", BRITS, contains="loss = loss_f + loss_b + loss_c"))
    imp = tr.p("imputations", BRITS, contains="imputations = (ret_f['imputations'] + ret_b['imputations']) / 2")
    rows_m = [[f"b{i}", wdates[i][t_], feats[j], num(imp_f[i, t_, j]), num(imp_b[i, t_, j]), num(imp[i, t_, j])] for i, t_, j in miss_all]
    S("9. BRITS 합치기",
      f"<b>merge_ret</b>: 두 방향의 채운 창을 <b>평균</b>냅니다(관측 칸은 둘 다 실제 값이라 그대로). 두 방향이 서로 다르게 채운 만큼 <b>일관성 손실</b> 0.1 × 평균|앞 − 뒤| = {f(lc)} 을 더합니다(평균은 창 전체 {B}×{L}×{K} = {B * L * K}칸 — 관측 칸은 두 방향이 같아 차이 0). "
      f"BRITS 손실 = 앞 {f(ret_f_loss)} + 뒤 {f(xl_b * p['imputed_weight'])} + 일관성 {f(lc)} = <b>{f(lsum)}</b>. 표는 빈 칸 {len(rows_m)}개 전부의 두 방향 값과 평균입니다.",
      say=(f"이제 두 방향의 채운 창을 평균냅니다. 관측 칸은 둘 다 실제 값이라 그대로입니다. 두 방향이 서로 다르게 채운 만큼 일관성 손실을 더하는데, "
           f"두 결과 차이의 절댓값을 창 전체 {B * L * K}칸에서 평균하고 0.1을 곱한 {f(lc)}입니다. 관측 칸은 두 방향이 같아서 차이가 0입니다. BRITS 손실은 앞 방향 {f(ret_f_loss)}, 뒤 방향 {f(xl_b * p['imputed_weight'])}, 일관성 {f(lc)}를 더한 {f(lsum)}입니다. "
           f"표는 빈 칸 {len(rows_m)}개 전부의 두 방향 값과 평균입니다."),
      notes={src.line(BRITS, "imputations = (ret_f['imputations'] + ret_b['imputations']) / 2"): "두 방향 평균(관측 칸은 같은 값)",
             src.line(BRITS, "loss_c = self.get_consistency_loss"): f"일관성 손실 = 0.1 × 평균|앞 − 뒤| (전체 {B * L * K}칸) = {f(lc)}",
             src.line(BRITS, "loss = loss_f + loss_b + loss_c"): f"{f(ret_f_loss)} + {f(xl_b * p['imputed_weight'])} + {f(lc)} = {f(lsum)}"},
      code=C(BRITS, src.span(BRITS, "def merge_ret(self, ret_f, ret_b):", "return ret_f") + src.span(BRITS, "def get_consistency_loss", "return loss"),
             src.line(BRITS, "imputations = (ret_f['imputations'] + ret_b['imputations']) / 2"), "BRITS.merge_ret", 1, 2),
      stack=stk_brits + ["BRITS.merge_ret"], nodes=["merge"], edges=[("fwd", "merge"), ("bwd", "merge")],
      formula=f"채운 창 = (앞→뒤 + 뒤→앞) ÷ 2,  손실 = {f(ret_f_loss)} + {f(xl_b * p['imputed_weight'])} + {f(lc)} = {f(lsum)}",
      tensors=[{"type": "table", "split": 3, "digits": 3, "title": "빈 칸 전부: 앞→뒤 · 뒤→앞 · 평균(최종)", "columns": ["배치", "달", "변수", "앞→뒤", "뒤→앞", "평균"], "rows": rows_m}],
      checks=[chk("평균 = (앞 + 뒤) ÷ 2", imp, (imp_f + imp_b) / 2), chk("일관성 손실 = 0.1·평균|앞−뒤|", lc, float(torch.abs(imp_f - imp_b).mean()) * 0.1)])
    S("9. BRITS 합치기",
      f"BRITS 가 돌려준 <b>채운 창 imputed_x</b> [3, 6, 34] 입니다. 빈 칸 {len(miss_all)}개(빨간 테두리)만 채운 값이고, 나머지는 원래 값과 똑같습니다. "
      f"BIVA.forward 는 이것을 imputed_x, 손실 {f(lsum)} 을 imputed_loss 로 받습니다. <b>채운 창만</b> 다음 단계(분해)로 가고, 손실은 forward 가 마지막에 그대로 돌려주어 "
      "학습 때 전체 손실에 더해집니다(models/biva.py · fit: loss = VAE_loss + imputed_loss + loss_states).",
      say=(f"BRITS가 돌려준 채운 창입니다. 빈 칸 {len(miss_all)}개만 채운 값이고, 나머지는 원래 값과 똑같습니다. "
           f"순전파 함수는 채운 창을 다음 단계인 분해로 넘깁니다. 손실 {f(lsum)}은 따로 두었다가 마지막에 그대로 돌려주고, 학습 때 전체 손실에 더해집니다."),
      notes={src.line(BIVA, "imputed_x = x['imputations']", after=200): f"채운 창 — 빈 칸 {len(miss_all)}개만 바뀜", src.line(BIVA, "imputed_loss = x['loss']", after=200): f"BRITS 손실 {f(lsum)}"},
      code=C(BIVA, src.lines(BIVA, "imputed_x = x['imputations']", "imputed_loss = x['loss']", after=200), src.line(BIVA, "imputed_x = x['imputations']", after=200), "Model.forward", 4, 6),
      stack=stk_fwd, nodes=["merge"],
      tensors=[{**BH("BRITS 가 채운 최종 창 imputed_x (빨강 = 채운 칸)", imp, rows_w[0], scale="01"), "rowsEach": rows_w,
                "hlEach": [{"cells": [[t_, j] for i, t_, j in miss_all if i == b_]} for b_ in range(B)]}],
      checks=[chk("관측 칸은 원래 값 그대로", np.nan_to_num(xb.numpy(), nan=0) * (~np.isnan(xb.numpy())), imp.numpy() * (~np.isnan(xb.numpy()))),
              chk("빈 칸이 남지 않음(NaN 0개)", float(torch.isnan(imp).sum()), 0.0, 0)])

    # ================================================================== 10. 분해
    ev_pad = tr.one(BIVA, contains="x = torch.cat([front, x, end], dim=1)")
    padded = ev_pad.produced["x"]
    kk = int(args["moving_avg"]) if not isinstance(args["moving_avg"], list) else None
    fr, en = tr.p("front", BIVA, contains="front = x[:, 0:1, :].repeat"), tr.p("end", BIVA, contains="end = x[:, -1:, :].repeat")
    rows_pad = [["앞 복사"] * fr.shape[1] + [f"τ{t + 1}" for t in range(L)] + ["뒤 복사"] * en.shape[1]][0]
    S("10. 분해",
      f"BIVA.forward 로 돌아와 채운 창을 <b>추세</b>와 <b>주기</b>로 나눕니다(series_decomp). 먼저 moving_avg 가 창 양끝을 늘립니다. "
      f"이동평균 창이 {kk}칸이라 끝에서는 이웃이 모자라기 때문입니다. 첫 행을 앞에 {fr.shape[1]}번, 마지막 행을 뒤에 {en.shape[1]}번 복사해 붙여 [3, {padded.shape[1]}, 34] 를 만듭니다. "
      f"창 길이 {kk}가 짝수라서 앞뒤 개수가 다릅니다(원본 그대로).",
      say=(f"순전파로 돌아와 채운 창을 추세와 주기로 나눕니다. 먼저 이동평균이 창의 양끝을 늘립니다. 이동평균 창이 {kk}칸이라 끝에서는 이웃이 모자라기 때문입니다. "
           f"첫 행을 앞에 {fr.shape[1]}번, 마지막 행을 뒤에 {en.shape[1]}번 복사해 붙여 시점 {padded.shape[1]}개짜리 창을 만듭니다. 이동평균 창 길이 {kk}가 짝수라서 앞뒤 개수가 다른데, 원본 그대로입니다."),
      notes={ev_pad.line: f"[앞 복사 {fr.shape[1]} ; 창 {L} ; 뒤 복사 {en.shape[1]}] → 시점 {padded.shape[1]}개",
             src.line(BIVA, "seasonal_init, trend_init = self.decomposition(imputed_x)"): "채운 창 → 주기, 추세"},
      code=C(BIVA, src.span(BIVA, "def forward(self, x):", "x = torch.cat([front, x, end], dim=1)") + [src.line(BIVA, "seasonal_init, trend_init = self.decomposition(imputed_x)")],
             ev_pad.line, "moving_avg.forward", 2, 4),
      stack=stk_fwd + ["series_decomp.forward", "moving_avg.forward"], nodes=["dec"], edges=[("merge", "dec")],
      tensors=[H(f"b=0 의 채운 창 [{L}, {K}]", imp[0], rows=rows_w[0], cols=feats, shape=[L, K], scale="01", cellH=19),
               H(f"b=0 끝값을 복사해 붙인 창 [{padded.shape[1]}, {K}] (주황 = 복사한 행, b=1·2 도 같은 방식)", padded[0], rows=rows_pad, cols=feats, shape=list(padded.shape[1:]), scale="01", cellH=19,
                 hl={"rows": list(range(fr.shape[1])) + list(range(fr.shape[1] + L, padded.shape[1]))})])
    trend = tr.p("moving_mean", BIVA, contains="moving_mean = self.moving_avg(x)")
    seas = tr.p("res", BIVA, contains="res = x - moving_mean")
    pd_ = padded.double().numpy()
    tr_hand = np.stack([pd_[:, a:a + kk, :].mean(1) for a in range(L)], 1)
    jF = j11
    lab_ma_ = lab_movavg(imp[0, :, jF].numpy(), kk, fr.shape[1], en.shape[1], win[0], "F11", trend[0, :, jF].numpy(), seas[0, :, jF].numpy())
    S("10. 분해",
      f"AvgPool1d({kk}, stride 1) 가 붙인 열을 한 칸씩 밀며 {kk}개씩 평균을 냅니다 → <b>추세</b> [3, 6, 34]. 그리고 <b>주기 = 채운 창 − 추세</b>입니다. "
      f"예: b=0 의 F11 열 τ1 추세 = 앞에 붙인 복사 {fr.shape[1]}개와 τ1 … τ{kk - fr.shape[1]} 의 평균 = ({' + '.join(f(v, 3) for v in pd_[0, 0:kk, jF])}) ÷ {kk} = {f(trend[0, 0, jF])}. "
      "추세(이동평균)는 위쪽 Conv 경로로, 주기(채운 창 − 추세, 곧 나머지 — 코드 이름 seasonal)는 아래쪽 LSTM-VAE 경로로 갑니다.",
      say=(f"이동평균이 붙인 열을 한 칸씩 밀며 {kk}개씩 평균을 내서 추세를 만듭니다. 그리고 주기는 채운 창에서 추세를 뺀 것입니다. "
           f"예를 들어 첫 번째 창의 F11 열에서 τ1의 추세는, 앞에 붙인 복사 {fr.shape[1]}개와 τ1부터 τ{kk - fr.shape[1]}까지 {kk - fr.shape[1]}개, 모두 {kk}개를 평균한 {f(trend[0, 0, jF])}입니다. "
           "추세, 곧 이동평균은 위쪽 합성곱 경로로, 주기, 곧 채운 창에서 추세를 뺀 나머지는 아래쪽 LSTM VAE 경로로 갑니다."),
      notes={src.line(BIVA, "x = self.avg(x.permute(0, 2, 1))"): f"AvgPool1d({kk}, stride 1): 시간 축으로 {kk}개씩 평균 → 추세",
             src.line(BIVA, "res = x - moving_mean"): "주기 = 채운 창 − 추세"},
      lab=lab_ma_,
      code=C(BIVA, src.span(BIVA, "x = self.avg(x.permute(0, 2, 1))", "return x") + src.span(BIVA, "moving_mean = self.moving_avg(x)", "return res, moving_mean"),
             src.line(BIVA, "x = self.avg(x.permute(0, 2, 1))"), "moving_avg / series_decomp", 2, 2),
      stack=stk_fwd + ["series_decomp.forward"], nodes=["dec"],
      formula=f"추세_τ = 평균(붙인 열[τ … τ+{kk - 1}]),  주기 = x − 추세   (permute 로 [3, 34, 11] 로 돌려 시간 축을 AvgPool1d 에 넣고 다시 [3, 6, 34])",
      tensors=[H("b=0 추세 trend_init (붙인 창을 6개씩 평균)", trend[0], rows=rows_w[0], cols=feats, shape=[L, K], scale="01", cellH=19, hl={"cols": [jF]}),
               H("b=0 주기 seasonal_init = 채운 창 − 추세 (파랑 +, 주황 −, 숫자는 마우스로)", seas[0], rows=rows_w[0], cols=feats, shape=[L, K], cellH=19, numbers=False)],
      checks=[chk("추세 손 계산(6개 평균) = 원본", trend, tr_hand), chk("주기 = 채운 창 − 추세", seas, imp.double().numpy() - tr_hand)])

    # ================================================================== 11. 추세 경로
    t_in = tr.p("trend_init", BIVA, contains="trend_init = trend_init.permute(0, 2, 1)")
    to1, to2, to3 = [tr.p(f"trend_output_{i}", BIVA, contains=f"trend_output_{i} = self.Conv1d_Trend_{i}") for i in (1, 2, 3)]
    tout = tr.p("trend_output", BIVA, contains="trend_output = self.Linear_Trend(trend_output_3)")
    W1 = sd["Conv1d_Trend_1.weight"].numpy()[:, :, 0]; b1 = sd["Conv1d_Trend_1.bias"].numpy()
    S("11. 추세 경로",
      f"추세를 <b>permute(0, 2, 1)</b> 로 [3, 34, 6] 으로 돌립니다(변수 = 채널, 시간 = 길이). Conv1d 의 커널이 1 이라 <b>시점마다 따로</b> 34개 변수를 가중합해 {c4}개로 줄입니다(Conv1d_Trend_1, 가중치 [{c4}, 34, 1]). "
      "시점끼리는 섞지 않습니다. 같은 가중치를 6개 시점에 똑같이 씁니다.",
      say=(f"추세를 돌려서 변수를 채널로, 시간을 길이로 놓습니다. 일차원 합성곱의 커널 크기가 1이라서, 시점마다 따로 {K}개 변수를 가중합해 {c4}개로 줄입니다. "
           f"시점끼리는 섞지 않고, 같은 가중치를 {L}개 시점에 똑같이 씁니다."),
      notes={src.line(BIVA, "trend_init = trend_init.permute(0, 2, 1)"): f"[{B}, {L}, {K}] → [{B}, {K}, {L}] (변수 = 채널)",
             src.line(BIVA, "trend_output_1 = self.Conv1d_Trend_1(trend_init)"): f"커널 1: 시점마다 {K}개 → {c4}개 가중합"},
      code=C(BIVA, [src.line(BIVA, "trend_init = trend_init.permute(0, 2, 1)"), src.line(BIVA, "trend_output_1 = self.Conv1d_Trend_1(trend_init)"), src.line(BIVA, "self.Conv1d_Trend_1 = nn.Conv1d(")],
             src.line(BIVA, "trend_output_1 = self.Conv1d_Trend_1(trend_init)"), "Model.forward", 3, 4),
      stack=stk_fwd, nodes=["tc"], edges=[("dec", "tc")],
      formula=f"a¹[:, τ] = W¹ · 추세[:, τ] + b¹   W¹: [{c4}, 34]   (τ = 1…6 마다)",
      tensors=[{"type": "row", "items": [H(f"b=0 추세 (permute 뒤 [{K}, {L}])", t_in[0], rows=feats, cols=[f"τ{t + 1}" for t in range(L)], shape=[K, L], scale="01", cellW=50, cellH=15, rowLabelW=70, digits=2),
                                         H(f"b=0 Conv1d_Trend_1 출력 [{c4}, {L}] = W¹ [{c4}×{K}] · 왼쪽 + b¹", to1[0], rows=[f"채널{c + 1}" for c in range(to1.shape[1])], cols=[f"τ{t + 1}" for t in range(L)], shape=list(to1.shape[1:]), cellW=60, digits=3)]}],
      checks=[chk("Conv1d(커널 1) 손 계산 = 원본", to1[0], W1 @ t_in[0].double().numpy() + b1[:, None])])
    W2 = sd["Conv1d_Trend_2.weight"].numpy()[:, :, 0]; b2 = sd["Conv1d_Trend_2.bias"].numpy()
    W3 = sd["Conv1d_Trend_3.weight"].numpy()[:, :, 0]; b3 = sd["Conv1d_Trend_3.bias"].numpy()
    wl, bl = sd["Linear_Trend.weight"].numpy()[0], float(sd["Linear_Trend.bias"].numpy()[0])
    a3 = to3[0, 0].double().numpy()
    S("11. 추세 경로",
      f"두 번 더 줄입니다: {c4} → {c16} (Conv1d_Trend_2), {c16} → 1 (Conv1d_Trend_3). 사이에 활성화 함수가 없어서 세 단은 결국 한 번의 가중합과 같은 모양입니다. "
      f"마지막으로 <b>Linear_Trend</b>({L}→1)가 시간 6칸에 가중치를 곱해 더하고 편향을 더해 값 하나를 냅니다. 원본은 이 가중치를 1/{L}(= 평균)로 시작해 학습으로 바꿉니다. "
      f"b=0 의 추세 경로 출력 = <b>{f(tout[0, 0, 0])}</b> (표준화한 정답 단위). 최종 예측은 아니고, 뒤의 「17. 합치기」에서 주기 경로 출력과 α 비율로 섞입니다.",
      say=(f"두 번 더 줄입니다. {c4}개에서 {c16}개로, 다시 한 개로 줄입니다. 사이에 활성화 함수가 없어서, 세 단은 결국 한 번의 가중합과 같은 모양입니다. "
           f"마지막으로 선형 층이 시간 {L}칸에 가중치를 곱해 더하고 편향을 더해서 값 하나를 냅니다. 원본은 이 가중치를 {L}분의 1, 곧 평균으로 시작해 학습으로 바꿉니다. "
           f"첫 번째 창의 추세 경로 출력은 표준화한 정답 단위로 {f(tout[0, 0, 0])}입니다. 아직 최종 예측은 아니고, 뒤에서 주기 경로 출력과 알파 비율로 섞입니다."),
      notes={src.line(BIVA, "trend_output_2 = self.Conv1d_Trend_2(trend_output_1)"): f"{c4} → {c16}", src.line(BIVA, "trend_output_3 = self.Conv1d_Trend_3(trend_output_2)"): f"{c16} → 1",
             src.line(BIVA, "trend_output = self.Linear_Trend(trend_output_3)"): f"시간 {L}칸 가중합 → 추세 경로 출력 {f(tout[0, 0, 0])} (b=0)"},
      lab=lab_trend_linear(a3, wl, bl, win[0], tout[0, 0, 0]),
      code=C(BIVA, src.lines(BIVA, "trend_output_2 = self.Conv1d_Trend_2(trend_output_1)", "trend_output_3 = self.Conv1d_Trend_3(trend_output_2)", "trend_output = self.Linear_Trend(trend_output_3)")
             + src.lines(BIVA, "self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)", "(1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))", after=src.line(BIVA, "self.Conv1d_Trend_3 = nn.Conv1d(")),
             src.line(BIVA, "trend_output = self.Linear_Trend(trend_output_3)"), "Model.forward", 3, 4),
      stack=stk_fwd, nodes=["tc", "tl"], edges=[("tc", "tl")],
      formula=f"추세 경로 출력 = Σ_τ w_τ · a³_τ + b = " + " + ".join(f"{f(wl[t], 3)}×{f(a3[t], 3)}" for t in range(L)) + (f" − {f(-bl, 3)}" if bl < 0 else f" + {f(bl, 3)}") + f" = {f(tout[0, 0, 0])}",
      tensors=[H("b=0 Conv1d_Trend_2 출력", to2[0], rows=[f"채널{c + 1}" for c in range(to2.shape[1])], cols=[f"τ{t + 1}" for t in range(L)], shape=list(to2.shape[1:]), cellW=72),
               H("b=0 Conv1d_Trend_3 출력 a³", to3[0], rows=["a³"], cols=[f"τ{t + 1}" for t in range(L)], shape=list(to3.shape[1:]), cellW=72),
               {"type": "table", "title": "Linear_Trend 의 시간 가중치(되돌린 가중치) · 결과", "columns": ["", *[f"τ{t + 1}" for t in range(L)], "편향", "b=0", "b=1", "b=2"],
                "rows": [["w", *[num(v) for v in wl], num(bl), num(tout[0, 0, 0]), num(tout[1, 0, 0]), num(tout[2, 0, 0])]]}],
      checks=[chk("Conv 2·3단 + Linear 손 계산 = 원본", float(tout[0, 0, 0]), float(wl @ (W3 @ (W2 @ to1[0].double().numpy() + b2[:, None]) + b3[:, None])[0] + bl))])

    # ================================================================== 12. VAE 인코더
    vin = tr.p("x", VAE, func="forward", call=True)
    enc = [tr.p("x", VAE, contains=f"x, _ = self.encoder_lstm_{i}(x)") for i in (1, 2, 3, 4)]
    dims = [K, K, h1, h2, lat]
    S("12. VAE 인코더 (LSTM 4층)",
      f"주기 성분 [3, 6, 34] 가 <b>LSTM_VAE</b> 로 들어가 encode 에서 <b>LSTM 4개를 차례로</b> 통과합니다. 크기는 {K} → {K} → {h1} → {h2} → {lat} 입니다. "
      "LSTM 층 하나는 안에서 <b>6개 시점을 차례로</b> 돌며 시점마다 기억을 갱신하고, 시점마다 출력을 냅니다(batch_first, 출력 모양 [3, 6, 은닉]). "
      "다음 층은 앞 층의 6개 시점 출력 전체를 입력으로 받아 다시 6개 시점을 돕니다.",
      say=(f"주기 성분이 LSTM VAE로 들어가, 인코더에서 LSTM 네 개를 차례로 통과합니다. 크기는 {K}, {K}, {h1}, {h2}, {lat} 순서로 바뀝니다. "
           f"LSTM 층 하나는 안에서 {L}개 시점을 차례로 돌며 시점마다 기억을 갱신하고, 시점마다 출력을 냅니다. 다음 층은 앞 층의 {L}개 시점 출력 전체를 입력으로 받아 다시 {L}개 시점을 돕니다."),
      notes={src.line(VAE, "x, _ = self.encoder_lstm_1(x)"): f"{K} → {K}", src.line(VAE, "x, _ = self.encoder_lstm_2(x)"): f"{K} → {h1}",
             src.line(VAE, "x, _ = self.encoder_lstm_3(x)"): f"{h1} → {h2}", src.line(VAE, "x, _ = self.encoder_lstm_4(x)"): f"{h2} → {lat} (잠재 요인 수)"},
      code=C(VAE, src.span(VAE, "def encode(self, x):", "return x, mu, logvar") + src.span(VAE, "self.encoder_lstm_1 = nn.LSTM(", "self.hidden_size_2, self.latent_size, num_layers, batch_first=True)"),
             src.line(VAE, "x, _ = self.encoder_lstm_1(x)"), "LSTM_VAE.encode", 2, 2),
      stack=stk_fwd + ["BIVA_LSTM_VAE.py · Model.forward", "encode"], nodes=["e1", "e2", "e3", "e4"], edges=[("dec", "e1")],
      tensors=[{"type": "table", "title": "BIVA 안에서 LSTM 이 쓰이는 곳 (이 배치 한 번의 forward)", "columns": ["위치", "모듈", "입력 → 출력", "한 번 부를 때 도는 시점", "부르는 횟수", "창 하나당 셀 계산"],
                "rows": [["BRITS 앞→뒤", "rits_f.rnn_cell (LSTMCell)", f"{2 * K} → {Hn}", "1", f"{L} (시점마다)", L],
                         ["BRITS 뒤→앞", "rits_b.rnn_cell (LSTMCell)", f"{2 * K} → {Hn}", "1", f"{L}", L],
                         ["VAE 인코더 1", "encoder_lstm_1", f"{K} → {K}", L, 1, L], ["VAE 인코더 2", "encoder_lstm_2", f"{K} → {h1}", L, 1, L],
                         ["VAE 인코더 3", "encoder_lstm_3", f"{h1} → {h2}", L, 1, L], ["VAE 인코더 4", "encoder_lstm_4", f"{h2} → {lat}", L, 1, L],
                         ["VAE 디코더 1~4", "decoder_lstm_1~4", f"{2 * lat}→{h2}→{h1}→{K}→{K}", L, "각 1", 4 * L],
                         ["합계", "LSTM 모듈 10개", "", "", "", 2 * L + 8 * L]]}],
      hint="'LSTM 을 4~5번 돌리며 들어갈 때마다 다른 잡음을 제거한다'는 설명과 코드의 대조:\n"
           f"• 예측값까지 이어지는 경로에서 주기 성분이 거치는 LSTM 은 인코더 4층({K}→{K}→{h1}→{h2}→{lat})입니다. 디코더 LSTM 4층(14. 디코더)은 재구성 손실 계산에만 쓰이고 예측에는 들어가지 않습니다. "
           "층마다 가중치가 따로이고, 앞 층의 출력이 다음 층의 입력입니다.\n"
           "• 하지만 층마다 '이 층은 이런 잡음을 뺀다'고 정한 계산이나 손실은 없습니다. 층별 정답이 없고, 학습 신호는 전체 손실(예측 MSE + VAE 재구성·KLD + BRITS) 하나뿐입니다. "
           "어떤 정보가 어느 층에서 줄어드는지는 학습이 정하며, 코드만으로는 '층마다 다른 잡음'을 확인할 수 없습니다.\n"
           f"• 코드에 '잡음'을 따로 다루는 계산은 없습니다. 나누기에 가까운 장치는 「10. 분해」의 이동평균(느린 흐름인 추세와 나머지인 주기로 나눔 — 잡음 제거는 아님)과 인코더 출력 폭 {lat}(정보량 제한)입니다. "
           "KLD 는 z 의 분포(μ·log σ²)에만 걸리고 디코더와 주기 경로는 결정적인 인코더 출력도 함께 받으므로(x_z = [e ; z]), 이 구조에서 KLD 가 정보를 제한하는 힘은 약합니다. 이 실행에서 KLD 가 실제로 얼마인지는 「13. μ · log σ² · z」「15. VAE 손실」에서 봅니다.\n"
           "• 아래 단계에서 층별 실제 출력을 볼 수 있습니다. 층별로 무엇이 남는지 분석하려면 별도 실험(층 출력으로 원래 주기 성분을 얼마나 복원하는지 등)이 필요합니다.")
    for li in range(4):
        e_in = vin if li == 0 else enc[li - 1]
        hand, gts = lstm_np(e_in[0].double().numpy(), sd, f"LSTM_VAE.encoder_lstm_{li + 1}.")
        rows_g = [[f"τ{t + 1}", num(gts[t][0][0]), num(gts[t][1][0]), num(gts[t][2][0]), num(gts[t][3][0]), num(gts[t][4][0]), num(gts[t][5][0])] for t in range(L)]
        S("12. VAE 인코더 (LSTM 4층)",
          f"<b>encoder_lstm_{li + 1}</b>: [3, 6, {dims[li]}] → [3, 6, {dims[li + 1]}]. 시점 τ1 부터 τ6 까지 차례로, 입력 x_τ 와 직전 출력 h 로 문 4개(i·f·g·o)를 계산해 셀 상태 c 와 출력 h 를 갱신합니다. "
          f"표는 b=0 의 은닉 단위 1 이 시점마다 어떻게 바뀌는지 보여 줍니다. f(잊기 문)가 클수록 지난 셀 상태 c 를 많이 남깁니다. "
          + (f"이 층은 크기를 줄이지 않고({K}→{K}) 시간 흐름을 한 번 거칩니다." if li == 0 else f"크기를 {dims[li]} → {dims[li + 1]} 로 줄입니다."),
          say=(f"인코더의 {['첫', '두', '세', '네'][li]} 번째 층입니다. τ1부터 τ{L}까지 차례로, 입력과 직전 출력 h로 문 네 개를 계산하고, 셀 상태 c와 출력 h를 갱신합니다. "
               f"표는 첫 번째 창의 은닉 단위 1이 시점마다 어떻게 바뀌는지 보여 줍니다. 잊기 문이 클수록 지난 셀 상태 c를 많이 남깁니다. "
               + (f"이 층은 크기를 바꾸지 않고, {K}개 그대로 시간 흐름을 한 번 거칩니다." if li == 0 else f"크기를 {dims[li]}에서 {dims[li + 1]}로 줄입니다.")),
          notes={src.line(VAE, f"x, _ = self.encoder_lstm_{li + 1}(x)"): f"[{B}, {L}, {dims[li]}] → [{B}, {L}, {dims[li + 1]}] — 시점 {L}개를 차례로"},
          code=C(VAE, [src.line(VAE, f"x, _ = self.encoder_lstm_{li + 1}(x)"), src.line(VAE, f"self.encoder_lstm_{li + 1} = nn.LSTM(")], src.line(VAE, f"x, _ = self.encoder_lstm_{li + 1}(x)"), "LSTM_VAE.encode", 3, 6),
          stack=stk_fwd + ["BIVA_LSTM_VAE.py · Model.forward", "encode", f"encoder_lstm_{li + 1} (nn.LSTM)"], nodes=[f"e{li + 1}"], edges=[(f"e{li}", f"e{li + 1}")] if li else [],
          formula=f"시점마다: [i f g o] = W_ih·x_τ + W_hh·h_(τ−1) + b,  c_τ = σ(f)⊙c_(τ−1) + σ(i)⊙tanh(g),  h_τ = σ(o)⊙tanh(c_τ)   W_ih: [{4 * dims[li + 1]}, {dims[li]}]",
          tensors=[H(f"b=0 입력 [{L}, {dims[li]}]", e_in[0], rows=[f"τ{t + 1}" for t in range(L)], cols=feats if dims[li] == K else [f"u{u + 1}" for u in range(dims[li])], shape=[L, dims[li]], cellH=16),
                   H(f"b=0 출력 h_τ [{L}, {dims[li + 1]}]", enc[li][0], rows=[f"τ{t + 1}" for t in range(L)], cols=[f"u{u + 1}" for u in range(dims[li + 1])], shape=[L, dims[li + 1]], cellH=16),
                   {"type": "table", "digits": 4, "title": "b=0 · 은닉 단위 1 의 시점별 계산 (손 계산, σ·tanh 적용 후)", "columns": ["시점", "입력 문 i", "잊기 문 f", "새 정보 g", "출력 문 o", "셀 상태 c", "출력 h"], "rows": rows_g}],
          checks=[chk(f"encoder_lstm_{li + 1} 손 계산 = 원본(6개 시점 전부)", enc[li][0], hand)])

    # ================================================================== 13. μ·σ·z
    mu_pre, lv_pre = tr.p("mu", VAE, contains="mu = self.encoder_fc_mu(x)"), tr.p("logvar", VAE, contains="logvar = self.encoder_fc_logvar(x)")
    mu, lv = tr.p("mu", VAE, contains="mu = F.relu(mu)"), tr.p("logvar", VAE, contains="logvar = F.relu(logvar)")
    rows_t = [f"τ{t + 1}" for t in range(L)]; cols_l = [f"요인{u + 1}" for u in range(lat)]
    S("13. μ · log σ² · z",
      f"인코더 마지막 출력(시점마다 {lat}개)에서 Linear 두 개가 시점마다 <b>평균 μ</b> 와 <b>로그분산 log σ²</b> 를 만듭니다. 원본은 둘 다 <b>ReLU</b>를 거쳐 음수를 0 으로 자릅니다. "
      f"그래서 log σ² ≥ 0, 곧 <b>σ ≥ 1</b> 입니다. b=0 에서 ReLU 로 0 이 된 칸은 μ {int((mu[0] == 0).sum())}개, log σ² {int((lv[0] == 0).sum())}개(전체 {L * lat}개 중)입니다.",
      say=(f"인코더 마지막 출력에서 선형 층 두 개가 시점마다 평균 뮤와 로그 분산을 만듭니다. 원본은 둘 다 렐루를 거쳐 음수를 0으로 자릅니다. "
           f"그래서 로그 분산은 0 이상, 곧 시그마는 1 이상입니다. 첫 번째 창에서는 전체 {L * lat}칸 가운데 뮤는 {int((mu[0] == 0).sum())}칸, "
           f"로그 분산은 {int((lv[0] == 0).sum())}칸이 렐루로 0이 되었습니다."),
      notes={src.line(VAE, "mu = self.encoder_fc_mu(x)"): f"Linear {lat}→{lat}: 평균 μ", src.line(VAE, "logvar = self.encoder_fc_logvar(x)"): f"Linear {lat}→{lat}: 로그분산 log σ²",
             src.line(VAE, "mu = F.relu(mu)"): f"음수 → 0 (b=0: {int((mu[0] == 0).sum())}/{L * lat}칸이 0)", src.line(VAE, "logvar = F.relu(logvar)"): f"음수 → 0 → σ ≥ 1 ({int((lv[0] == 0).sum())}/{L * lat}칸이 0)"},
      code=C(VAE, src.lines(VAE, "mu = self.encoder_fc_mu(x)", "logvar = self.encoder_fc_logvar(x)", "mu = F.relu(mu)", "logvar = F.relu(logvar)"), src.line(VAE, "mu = F.relu(mu)"), "LSTM_VAE.encode / forward", 3, 4),
      stack=stk_fwd + ["BIVA_LSTM_VAE.py · Model.forward"], nodes=["mu"], edges=[("e4", "mu")],
      formula=f"μ = ReLU(W_μ·e + b_μ),  log σ² = ReLU(W_v·e + b_v)   W: [{lat}, {lat}]",
      tensors=[{"type": "row", "items": [H(f"b=0 {nm}", v_[0], rows=rows_t, cols=cols_l, shape=[L, lat], cellW=50, cellH=24, rowLabelW=34, digits=3)
                                          for nm, v_ in (("μ (ReLU 전)", mu_pre), ("μ (ReLU 후)", mu), ("log σ² (ReLU 전)", lv_pre), ("log σ² (ReLU 후)", lv))]}],
      checks=[chk("ReLU = max(0, ·)", mu, np.maximum(mu_pre.numpy(), 0), 0), chk("ReLU(log σ²) = max(0, ·)", lv, np.maximum(lv_pre.numpy(), 0), 0)],
      hint="보통의 VAE 는 μ·log σ² 에 활성화 함수를 쓰지 않습니다(음수 가능). 이 원본은 ReLU 를 씌워 μ ≥ 0, σ ≥ 1 이 됩니다. 원본 그대로 두었습니다.")
    # ---- 발견: 학습된 가중치에서 μ·log σ² 가 ReLU 로 모두 0 이 되는지(모든 창·학습 과정)
    va_ = cap.get("vae_all") or {}
    dead = bool(va_) and va_["mu_pos"] == 0 and va_["lv_pos"] == 0
    vp = cap.get("vae_pos") or {}
    eps_list = [e for e in sorted(vp) if e <= len(cap["history"])]  # 학습 epoch 만(기록용 순전파 제외)
    if va_:
        first_dead = next((e for e in eps_list if vp[e]["mu"] == 0 and vp[e]["lv"] == 0), None)
        revive = [e for e in eps_list if first_dead and e > first_dead and (vp[e]["mu"] > 0 or vp[e]["lv"] > 0)]
        runs = []                                   # 다시 양수였던 epoch 의 연속 구간
        for e in revive:
            if runs and e == runs[-1][1] + 1:
                runs[-1][1] = e
            else:
                runs.append([e, e])
        if revive:
            long_ = max(runs, key=lambda r_: r_[1] - r_[0])
            pk_e = max(revive, key=lambda e: max(vp[e]["mu"], vp[e]["lv"]))
            pk_lv = vp[pk_e]["lv"] >= vp[pk_e]["mu"]
            pk_n = vp[pk_e]["lv"] if pk_lv else vp[pk_e]["mu"]
            pk_pct = f(100 * pk_n / vp[pk_e]["n"], 2)
        S("13. μ · log σ² · z",
          (f"<b>실행에서 확인한 사실</b>: 고른 가중치로 학습 폴드의 창 {va_['n_windows']}개 전부(학습 {cap['n_tr']} · 비움 {cap['embargo']} · 검증 {cap['n_va']})를 넣어 보면, μ 의 ReLU 앞 값 {va_['n']}칸 중 양수가 {va_['mu_pos']}칸, "
           f"log σ² 는 {va_['lv_pos']}칸입니다. "
           + (f"곧 <b>이 {va_['n_windows']}개 창 모두에서 μ = 0, log σ² = 0</b> → σ = 1, <b>z = ε</b>(입력과 상관없는 표준정규 무작위 수)이고 KLD 는 0 입니다. "
              f"잠재 z 가 주기 정보를 담지 못하고, 주기 경로의 입력 {2 * lat}채널 중 z {lat}채널은 잡음이 됩니다(테스트 창은 「19. 예측」에서 확인). " if dead else
              f"일부 칸만 0 보다 큽니다(μ 최대 {f(va_['mu_max'])}, log σ² 최대 {f(va_['lv_max'])}). ")
           + (f"아래 그림은 학습 중 epoch 마다 양수였던 비율입니다. epoch {first_dead} 부터 0 이 되었습니다"
              + (f"(그 뒤 epoch {revive[0]}~{revive[-1]} 사이에는 모두 {len(revive)}개 epoch, {len(runs)}개 구간에서 일부 칸이 다시 양수였다가 0 으로 돌아갔습니다 — "
                 f"가장 긴 구간은 epoch {long_[0]}~{long_[1]}, 가장 많을 때는 epoch {pk_e} 의 {'log σ²' if pk_lv else 'μ'} {pk_n}칸 / {vp[pk_e]['n']}칸({pk_pct}%))." if revive else "(이후 다시 살아나지 않음).")
              if first_dead else "아래 그림은 학습 중 epoch 마다 양수였던 비율입니다.")),
          say=(f"실행에서 확인한 사실입니다. 고른 가중치로 학습 폴드의 창 {va_['n_windows']}개 전부를 넣어 보면, 뮤의 렐루 앞 값 {va_['n']}칸 가운데 "
               + ("양수인 칸이 하나도 없고, 로그 분산도 하나도 없습니다. " if dead else f"양수가 {va_['mu_pos']}칸, 로그 분산은 {va_['lv_pos']}칸입니다. ")
               + (f"곧 이 창들 모두에서 뮤와 로그 분산이 0이고, 시그마는 1입니다. 그래서 z는 입력과 상관없는 표준정규 무작위 수 그대로이고, KLD도 0입니다. "
                  f"잠재 변수가 주기 정보를 담지 못하고, 주기 경로의 입력 {2 * lat}개 채널 가운데 z의 {lat}개 채널은 잡음이 됩니다. " if dead else "일부 칸만 0보다 큽니다. ")
               + ((f"아래 그림은 학습 중 에포크마다 양수였던 비율입니다. 에포크 {first_dead}부터 0이 되었습니다. "
                   + (f"그 뒤 에포크 {revive[0]}에서 {revive[-1]} 사이에는 모두 {len(revive)}개 에포크, {len(runs)}개 구간에서 일부 칸이 다시 양수였다가 0으로 돌아갔습니다. "
                      f"가장 긴 구간은 에포크 {long_[0]}에서 {long_[1]}까지이고, 가장 많을 때는 에포크 {pk_e}의 {'로그 분산으로' if pk_lv else '뮤로'}, {pk_n}칸, {pk_pct}%였습니다." if revive else "그 뒤로는 다시 살아나지 않았습니다."))
                  if first_dead else "아래 그림은 학습 중 에포크마다 양수였던 비율입니다.")),
          notes={src.line(VAE, "mu = F.relu(mu)"): f"학습 폴드 창 {va_['n_windows']}개에서 ReLU 앞 값 양수 {va_['mu_pos']}/{va_['n']}칸",
                 src.line(VAE, "logvar = F.relu(logvar)"): f"같은 창들에서 양수 {va_['lv_pos']}/{va_['n']}칸",
                 src.line(VAE, "z = mu + eps * std"): "μ = 0, σ = 1 이면 z = ε (입력과 무관한 잡음)" if dead else "z = μ + ε·σ"},
          code=C(VAE, src.lines(VAE, "mu = F.relu(mu)", "logvar = F.relu(logvar)") + [src.line(VAE, "z = mu + eps * std")],
                 src.line(VAE, "mu = F.relu(mu)"), "LSTM_VAE.forward · reparameterize", 4, 4),
          stack=["tools/codewalk/biva_capture.py · 고른 가중치로 학습 폴드의 모든 창 실행", "BIVA_LSTM_VAE.py · encoder_fc_mu / encoder_fc_logvar"], nodes=["mu", "z"],
          formula="∂KLD/∂μ = μ,  ∂KLD/∂(log σ²) = ½(σ² − 1)  →  ReLU 뒤라 μ ≥ 0, log σ² ≥ 0 이므로, 0 보다 큰 동안 KLD 는 둘을 0 쪽으로 민다(0 에서는 기울기 0)\n"
                  "ReLU 앞 값이 모든 입력에서 음수면 이 두 층에는 기울기가 0 → 스스로는 되돌아오지 못함(앞단 e 가 바뀌면 일부가 다시 양수가 될 수 있음 — 그래프)",
          tensors=[{"type": "num", "title": f"고른 가중치 · 학습 폴드 창 {va_['n_windows']}개 × 시점 {L} × 요인 {lat}", "items": [
                       {"label": "μ ReLU 앞 양수 칸", "value": f"{va_['mu_pos']} / {va_['n']}", "color": "#b3261e" if dead else None},
                       {"label": "log σ² ReLU 앞 양수 칸", "value": f"{va_['lv_pos']} / {va_['n']}", "color": "#b3261e" if dead else None},
                       {"label": "μ 앞 값 최대", "value": f(va_["mu_max"])}, {"label": "log σ² 앞 값 최대", "value": f(va_["lv_max"])}]},
                   {"type": "line", "title": "학습 중 μ·log σ² 의 ReLU 앞 값이 양수였던 비율 (epoch 별, 학습 배치 전체)", "x": [str(e) for e in eps_list], "xlabel": "epoch", "ylo": 0, "yhi": 1,
                    "series": [{"name": "μ > 0 비율", "color": "#3b6fb6", "y": [num(vp[e]["mu"] / max(1, vp[e]["n"])) for e in eps_list], "dots": len(eps_list) <= 40},
                               {"name": "log σ² > 0 비율", "color": "#c8710f", "y": [num(vp[e]["lv"] / max(1, vp[e]["n"])) for e in eps_list], "dots": len(eps_list) <= 40}], "height": 210}],
          hint="가능한 설명(확인한 것은 아님): VAE 손실의 KLD 는 μ = 0, log σ² = 0 일 때 가장 작고, μ·log σ² 가 0 보다 큰 동안 둘을 0 쪽으로 밉니다. 원본은 둘에 ReLU 를 씌웠으므로, "
               "ReLU 앞 값이 모든 입력에서 음수가 되면 encoder_fc_mu·encoder_fc_logvar 에는 기울기가 0 이 되어 스스로는 되돌아오지 못합니다(Adam 의 관성으로 몇 걸음 더 움직일 수는 있고, "
               "인코더 출력 e 가 바뀌어 다시 양수가 될 수도 있습니다 — 그래프의 '다시 양수' epoch). 예측 손실도 z 의 잡음을 줄이려고 log σ² 를 낮추는 쪽으로 작용할 수 있으므로, "
               "어느 손실이 주원인인지는 KLD 를 뺀 비교 학습으로 확인해야 합니다(하지 않았습니다).\n"
               "뜻: 이 가중치에서 z 는 입력과 무관한 표준정규 잡음이고, 예측값의 무작위 흔들림(「19. 예측」)의 원인입니다. 이 실행에서 주기 경로가 입력에서 얻는 정보는 e(인코더 출력)뿐입니다. "
               "이 사실은 이 설정·이 폴드·이 시드의 실행 기록에서 확인한 것이므로, 다른 폴드·시드에서도 같은지는 같은 방법(tools/codewalk/biva_capture.py 의 vae_all·vae_pos)으로 확인해야 합니다.")

    std_, eps_, z_ = tr.p("std", VAE, contains="std = torch.exp(0.5 * logvar)"), tr.p("eps", VAE, contains="eps = torch.randn_like(std)"), tr.p("z", VAE, contains="z = mu + eps * std")
    S("13. μ · log σ² · z",
      f"<b>reparameterize</b>: σ = exp(½·log σ²), ε 는 표준정규에서 새로 뽑은 무작위 수, <b>z = μ + ε·σ</b> 입니다. ε 는 부를 때마다 새로 뽑힙니다(예측할 때도). "
      f"예: b=0 τ1 요인1: {f(mu[0, 0, 0])} + ({f(eps_[0, 0, 0])}) × {f(std_[0, 0, 0])} = {f(z_[0, 0, 0])}.",
      say=(f"재매개변수화 단계입니다. 시그마는 로그 분산의 절반에 지수 함수를 씌운 것이고, 엡실론은 표준정규 분포에서 새로 뽑은 무작위 수입니다. z는 뮤 더하기 엡실론 곱하기 시그마입니다. "
           f"엡실론은 부를 때마다 새로 뽑히고, 예측할 때도 그렇습니다. 예를 들어 첫 번째 창, τ1의 첫 번째 요인은 "
           f"{f(mu[0, 0, 0])} 더하기 {f(eps_[0, 0, 0])} 곱하기 {f(std_[0, 0, 0])}로, {f(z_[0, 0, 0])}입니다."),
      notes={src.line(VAE, "std = torch.exp(0.5 * logvar)"): "σ = exp(½·log σ²) ≥ 1", src.line(VAE, "eps = torch.randn_like(std)"): "ε ~ N(0, 1) — 부를 때마다 새로 뽑음",
             src.line(VAE, "z = mu + eps * std"): f"z = μ + ε·σ  예) {f(mu[0, 0, 0])} + ({f(eps_[0, 0, 0])}) × {f(std_[0, 0, 0])} = {f(z_[0, 0, 0])}"},
      lab=lab_reparam(mu[0, 0].numpy(), lv[0, 0].numpy(), eps_[0, 0].numpy(), win[0], z_[0, 0].numpy()),
      code=C(VAE, src.span(VAE, "def reparameterize(self, mu, logvar):", "return z") + [src.line(VAE, "z = self.reparameterize(mu, logvar)")], src.line(VAE, "z = mu + eps * std"), "LSTM_VAE.reparameterize", 3, 4),
      stack=stk_fwd + ["BIVA_LSTM_VAE.py · Model.forward", "reparameterize"], nodes=["z"], edges=[("mu", "z")],
      formula="σ = exp(0.5·log σ²) ≥ 1,  ε ~ N(0, 1),  z = μ + ε ⊙ σ",
      tensors=[H("b=0 σ", std_[0], rows=rows_t, cols=cols_l, shape=[L, lat], cellW=70), H("b=0 ε (이번에 뽑힌 수)", eps_[0], rows=rows_t, cols=cols_l, shape=[L, lat], cellW=70),
               H("b=0 z", z_[0], rows=rows_t, cols=cols_l, shape=[L, lat], cellW=70)],
      checks=[chk("z = μ + ε·σ", z_, mu.numpy() + eps_.numpy() * std_.numpy())])

    # ================================================================== 14. 디코더
    xz = tr.p("x_z", VAE, contains="x_z = torch.cat([x, z], dim=2)")
    dec_out = [tr.p("out", VAE, contains=f"self.decoder_lstm_{i}(") for i in (1, 2, 3, 4)]
    recon = tr.p("output", VAE, contains="output = self.output_fc(out)")
    rl_ = float(tr.p("recons_loss", BIVA, contains="recons_loss = F.mse_loss(recons, input)")); ms0_ = float((seas.double() ** 2).mean())
    rsd_, ssd_ = float(recon.double().std()), float(seas.double().std())    # 이 배치: 되살린 값·원래 주기 성분의 표준편차
    S("14. 디코더",
      f"인코더 출력 e(시점마다 {lat}개)와 z({lat}개)를 <b>마지막 축으로 이어 붙여</b>(torch.cat(dim=2)) [3, 6, {2 * lat}] 을 만듭니다. "
      f"디코더 LSTM 4층({2 * lat}→{h2}→{h1}→{K}→{K})과 output_fc({K}→{K})가 이것으로 <b>주기 성분을 되살립니다</b>. "
      "되살린 값은 VAE 손실에만 쓰이고, 예측 경로에는 들어가지 않습니다. "
      + f"이 배치에서 되살린 값은 0 근처에 몰려 있습니다(표준편차 {f(rsd_, 3)} — 원래 주기 성분은 {f(ssd_, 3)}): 재구성 MSE {g4(rl_)} 는 '전부 0 으로 되살렸을 때'의 MSE {g4(ms0_)} "
      + ("보다 오히려 조금 큽니다" if rl_ > ms0_ else "와 비슷합니다" if rl_ > 0.8 * ms0_ else "보다 작습니다")
      + f"(설명 비율 1 − {g4(rl_)} ÷ {g4(ms0_)} = {f(1 - rl_ / ms0_, 3)}" + (f" — 전부 0 으로 되살린 것보다 오차가 {f((rl_ / ms0_ - 1) * 100, 1)}% 더 큼" if rl_ > ms0_ else "") + ").",
      say=(f"인코더 출력과 z를 이어 붙여 시점마다 {2 * lat}개 값을 만듭니다. 디코더 LSTM 네 층과 마지막 선형 층이 이것으로 주기 성분을 되살립니다. "
           "되살린 값은 VAE 손실에만 쓰이고, 예측 경로에는 들어가지 않습니다. "
           f"이 배치에서 되살린 값은 0 근처에 몰려 있습니다. 표준편차가 {f(rsd_, 3)}로, 원래 주기 성분의 {f(ssd_, 3)}보다 작습니다. "
           f"재구성 오차 {g4(rl_)}는, 전부 0으로 되살렸을 때의 오차 {g4(ms0_)}"
           + ("보다 오히려 조금 큽니다." if rl_ > ms0_ else "와 비슷합니다." if rl_ > 0.8 * ms0_ else "보다 작습니다.")
           + (f" 곧 전부 0으로 되살린 것보다 오차가 {f((rl_ / ms0_ - 1) * 100, 1)}% 더 큽니다." if rl_ > ms0_ else "")
           + f" 설명 비율로는 {f(1 - rl_ / ms0_, 3)}입니다."),
      notes={src.line(VAE, "x_z = torch.cat([x, z], dim=2)"): f"[e ; z] → 시점마다 {2 * lat}개", src.line(VAE, "output = self.decode(x_z)"): "디코더 LSTM 4층 + output_fc → 되살린 주기(손실에만 씀)"},
      code=C(VAE, [src.line(VAE, "x_z = torch.cat([x, z], dim=2)"), src.line(VAE, "output = self.decode(x_z)")] + src.span(VAE, "out, _ = self.decoder_lstm_1(x_z)", "output = self.output_fc(out)"),
             src.line(VAE, "x_z = torch.cat([x, z], dim=2)"), "LSTM_VAE.forward / decode", 2, 2),
      stack=stk_fwd + ["BIVA_LSTM_VAE.py · Model.forward", "decode"], nodes=["decd"], edges=[("z", "decd")],
      tensors=[{"type": "row", "items": [H("b=0 x_z = [e ; z] (z 는 ε 그대로)" if dead else "b=0 x_z = [e ; z]", xz[0], rows=rows_t, cols=[f"e{u + 1}" for u in range(lat)] + [f"z{u + 1}" for u in range(lat)], shape=[L, 2 * lat], cellW=52, rowLabelW=34),
                                         {"type": "table", "title": "디코더 층별 모양", "columns": ["층", "입력", "출력"], "rows": [[f"decoder_lstm_{i + 1}", str([B, L, [2 * lat, h2, h1, K][i]]), str(list(dec_out[i].shape))] for i in range(4)] + [["output_fc", str([B, L, K]), str(list(recon.shape))]]}]},
               {"type": "row", "items": [H("b=0 주기 seasonal_init (원래)", seas[0], rows=rows_w[0], cols=feats, shape=[L, K], cellH=17, cellW=12, rowLabelW=70, numbers=False, abs=float(np.abs(seas[0].numpy()).max())),
                                         H("b=0 되살린 주기 recon_output", recon[0], rows=rows_w[0], cols=feats, shape=[L, K], cellH=17, cellW=12, rowLabelW=70, numbers=False, abs=float(np.abs(seas[0].numpy()).max()))]}])
    rl = float(tr.p("recons_loss", BIVA, contains="recons_loss = F.mse_loss(recons, input)"))
    kl = tr.p("kld_loss", BIVA, contains="kld_loss = torch.mean(")
    vl = float(tr.p("VAE_loss", BIVA, contains="VAE_loss,_,_ = self.loss_function("))
    kl_hand = np.mean(-0.5 * np.sum(1 + lv.double().numpy() - mu.double().numpy() ** 2 - np.exp(lv.double().numpy()), axis=1), axis=0)
    S("15. VAE 손실",
      f"<b>loss_function</b>: 재구성 MSE = 평균(되살린 값 − 주기)² = {g4(rl)}. KLD 는 '평균 μ·표준편차 σ 인 정규분포가 표준정규 분포와 얼마나 다른가'로, <b>시간 축(dim=1)으로 더하고 배치로 평균</b>내어 요인별 {lat}개가 남습니다. "
      f"원본 식은 0.5·MSE(숫자 하나)와 0.5·KLD(4개)를 더해 4개짜리를 만든 뒤 모두 더합니다. 그래서 실제 값은 <b>0.5 × {lat} × MSE + 0.5 × ΣKLD = {g4(vl)}</b> 입니다."
      + (" KLD 가 모두 0 인 것은 앞에서 본 대로 μ = 0, log σ² = 0 이기 때문입니다(1 + 0 − 0 − e⁰ = 0)." if float(np.abs(kl.numpy()).max()) == 0 else ""),
      say=(f"VAE 손실입니다. 재구성 오차는 되살린 값과 주기 성분 차이의 제곱 평균으로, {g4(rl)}입니다. KLD는 평균이 뮤, 표준편차가 시그마인 정규분포가 표준정규 분포와 얼마나 다른지를 재는 값입니다. "
           f"시간 축으로 더하고 배치로 평균내어 요인별로 {lat}개가 남습니다. 원본 식은 재구성 오차의 절반과 KLD의 절반을 더해 {lat}개짜리를 만든 뒤 모두 더합니다. "
           f"그래서 실제 값은 0.5 곱하기 {lat} 곱하기 재구성 오차에, 0.5 곱하기 KLD 합을 더한 {g4(vl)}입니다."
           + (" KLD가 모두 0인 것은 앞에서 본 대로 뮤와 로그 분산이 0이기 때문입니다." if float(np.abs(kl.numpy()).max()) == 0 else "")),
      notes={src.line(BIVA, "recons_loss = F.mse_loss(recons, input)"): f"재구성 MSE = {g4(rl)}", src.line(BIVA, "kld_loss = torch.mean("): f"시간 축 합 → 배치 평균 → 요인별 {lat}개",
             src.line(BIVA, "return torch.sum(loss), recons_loss, -kld_loss"): f"Σ(0.5·MSE + 0.5·KLD_j) = {g4(vl)}"},
      lab=lab_vae(mu.numpy().tolist(), lv.numpy().tolist(), rl, kl.numpy(), vl),
      code=C(BIVA, src.span(BIVA, "def loss_function(self, recons, input, mu, log_var):", "return torch.sum(loss), recons_loss, -kld_loss") + [src.line(BIVA, "VAE_loss,_,_ = self.loss_function(")],
             src.line(BIVA, "kld_loss = torch.mean("), "Model.loss_function", 1, 2),
      stack=stk_fwd + ["BIVA.py · loss_function"], nodes=["vl"], edges=[("decd", "vl")],
      formula=f"KLD_j = 평균_b(−½ Σ_τ (1 + log σ² − μ² − σ²)) = [{', '.join(f(v) for v in kl)}]\nVAE = Σ_j (0.5·{g4(rl)} + 0.5·KLD_j) = {g4(vl)}",
      tensors=[{"type": "num", "title": "VAE 손실 구성", "items": [{"label": "재구성 MSE", "value": g4(rl)}, *[{"label": f"KLD 요인{j + 1}", "value": f(v)} for j, v in enumerate(kl)], {"label": "VAE 손실", "value": g4(vl), "color": "#c8710f"}]}],
      checks=[chk("KLD 손 계산 = 원본", kl, kl_hand), chk("VAE 손실 = 0.5·잠재수·MSE + 0.5·ΣKLD", vl, 0.5 * lat * rl + 0.5 * float(np.sum(kl_hand)), 1e-4)],
      hint="원본 주석에는 kld_weight 가 'minibatch 크기 보정'이라고 되어 있지만 값은 0.5 로 고정입니다. 요인 수만큼 MSE 가 반복해 더해지는 것은 원본 식의 모양(브로드캐스트) 때문입니다.")

    # ================================================================== 16. 주기 경로
    sx = tr.p("seasonal_output_x_z", BIVA, contains="seasonal_output_x_z = torch.cat(")
    so = [tr.p(f"seasonal_output_{i}", BIVA, contains=f"seasonal_output_{i} = self.Conv1d_Seasonal_{i}") for i in (1, 2, 3)]
    sout = tr.p("seasonal_output", BIVA, contains="seasonal_output = self.Linear_Seasonal(seasonal_output_3)")
    Ws = [sd[f"Conv1d_Seasonal_{i}.weight"].numpy()[:, :, 0] for i in (1, 2, 3)]; bs_ = [sd[f"Conv1d_Seasonal_{i}.bias"].numpy() for i in (1, 2, 3)]
    wsl, bsl = sd["Linear_Seasonal.weight"].numpy()[0], float(sd["Linear_Seasonal.bias"].numpy()[0])
    a_ = sx[0].double().numpy()
    for Wc, bc in zip(Ws, bs_):
        a_ = Wc @ a_ + bc[:, None]
    s_hand = float(wsl @ a_[0] + bsl)
    S("16. 주기 경로",
      f"BIVA.forward 로 돌아와 인코더 출력 e 와 z 를 각각 permute 해 [3, {lat}, 6] 으로 만들고, <b>채널 축으로 이어 붙입니다</b>(torch.cat(dim=1)) → [3, {2 * lat}, 6]. "
      f"추세 경로와 같은 방식으로 Conv1d 3단({2 * lat}→{lat}→{lat // 2}→1, 커널 1)과 Linear_Seasonal({L}→1)을 거쳐 값 하나를 냅니다. b=0 의 주기 경로 출력 = <b>{f(sout[0, 0, 0])}</b>.",
      say=(f"순전파로 돌아와, 인코더 출력과 z를 각각 축 순서를 바꿔서 채널 축으로 이어 붙입니다. 그러면 채널 {2 * lat}개, 시점 {L}개가 됩니다. "
           f"추세 경로와 같은 방식으로 합성곱 세 단과 선형 층을 거쳐 값 하나를 냅니다. 첫 번째 창의 주기 경로 출력은 {f(sout[0, 0, 0])}입니다."),
      notes={src.line(BIVA, "seasonal_output_x_z = torch.cat("): f"[e ; z] → 채널 {2 * lat}개", src.line(BIVA, "seasonal_output = self.Linear_Seasonal(seasonal_output_3)"): f"시간 {L}칸 가중합 → 주기 경로 출력 {f(sout[0, 0, 0])}"},
      code=C(BIVA, src.span(BIVA, "seasonal_enc_output_x = seasonal_enc_output_x.permute(0, 2, 1)", "seasonal_output = self.Linear_Seasonal(seasonal_output_3)"), src.line(BIVA, "seasonal_output_x_z = torch.cat("), "Model.forward", 2, 2),
      stack=stk_fwd, nodes=["sc", "sl"], edges=[("z", "sc"), ("e4", "sc"), ("sc", "sl")],
      formula=f"[e; z]: [{2 * lat}, 6] → Conv {2 * lat}→{lat}→{lat // 2}→1 → Linear(6→1) = {f(sout[0, 0, 0])}",
      tensors=[H("b=0 [e ; z] (permute·cat 뒤)", sx[0], rows=[f"e{u + 1}" for u in range(lat)] + [f"z{u + 1}" for u in range(lat)], cols=rows_t, shape=[2 * lat, L], cellW=64),
               H("b=0 Conv1d_Seasonal_3 출력", so[2][0], rows=["s³"], cols=rows_t, shape=[1, L], cellW=64),
               {"type": "table", "title": "Linear_Seasonal 시간 가중치(되돌린 가중치) · 결과", "columns": ["", *rows_t, "편향", "b=0", "b=1", "b=2"],
                "rows": [["w", *[num(v) for v in wsl], num(bsl), num(sout[0, 0, 0]), num(sout[1, 0, 0]), num(sout[2, 0, 0])]]}],
      checks=[chk("주기 경로 손 계산 = 원본", float(sout[0, 0, 0]), s_hand)])

    # ================================================================== 17. 합치기
    st = tr.p("states", BIVA, contains="states = (seasonal_output*(self.alpha))")
    alpha = float(sd["alpha"].numpy().ravel()[0])
    S("17. 합치기",
      f"<b>states = 주기 경로 출력 × α + 추세 경로 출력 × (1 − α)</b>. α 는 학습되는 숫자 하나이고, 원본은 1 로 시작합니다. 지금 가중치의 α = <b>{f(alpha)}</b>, 1 − α = {f(1 - alpha, 3)} 입니다. "
      "α 는 0~1 로 제한되지 않습니다. 마지막에 permute 로 [배치, 출력 길이, 채널] 순서로 맞춥니다(모양은 [3, 1, 1] 그대로).",
      say=(f"이제 두 경로의 출력을 합칩니다. 주기 경로 출력에 알파를 곱하고, 추세 경로 출력에 1 빼기 알파를 곱해 더합니다. 알파는 학습되는 숫자 하나이고, 원본은 1로 시작합니다. "
           f"지금 가중치의 알파는 {f(alpha)}입니다. 알파는 0에서 1 사이로 제한되지 않습니다. 첫 번째 창은 {f(sout[0, 0, 0])} 곱하기 {f(alpha)}에, "
           f"{f(tout[0, 0, 0])}에 1 빼기 {f(alpha)}, 곧 {f(1 - alpha, 3)}을 곱한 값을 더해 {f(st[0, 0, 0])}입니다."),
      notes={src.line(BIVA, "states = (seasonal_output*(self.alpha))"): f"주기 × α + 추세 × (1 − α), α = {f(alpha)}", src.line(BIVA, "self.alpha = nn.Parameter(torch.ones(1, 1, 1))"): "α 는 1 에서 시작해 학습됨(범위 제한 없음)"},
      lab=lab_combine(sout[:, 0, 0].numpy(), tout[:, 0, 0].numpy(), alpha, st[:, 0, 0].numpy()),
      code=C(BIVA, src.span(BIVA, "if self.combination:", "states = states.permute(0, 2, 1)", after=240) + [src.line(BIVA, "self.alpha = nn.Parameter(torch.ones(1, 1, 1))")], src.line(BIVA, "states = (seasonal_output*(self.alpha))"), "Model.forward", 2, 6),
      stack=stk_fwd, nodes=["comb"], edges=[("tl", "comb"), ("sl", "comb")],
      formula=f"b=0: {f(sout[0, 0, 0])} × {f(alpha)} + {f(tout[0, 0, 0])} × (1 − {f(alpha)}) = {f(st[0, 0, 0])}",
      tensors=[{"type": "table", "title": "세 창의 두 경로 출력과 합친 값 (표준화한 정답 단위)", "columns": ["배치", "주기 경로 출력", "추세 경로 출력", "α", "states", "정답 y_z"],
                "rows": [[f"b{i}", num(sout[i, 0, 0]), num(tout[i, 0, 0]), num(alpha), num(st[i, 0, 0]), num(cap["batch"]["y_z"][i])] for i in range(B)]}],
      checks=[chk("합치기 손 계산 = 원본", st[:, 0, 0], sout[:, 0, 0].numpy() * alpha + tout[:, 0, 0].numpy() * (1 - alpha))])
    S("17. 합치기",
      "BIVA.forward 가 6개를 돌려줍니다. fit 은 이 중 <b>states, VAE 손실, 보간 손실</b> 3개를 씁니다. 되살린 주기·주기 성분·채운 창은 쓰지 않습니다(밑줄 _ 로 받음).",
      say=("순전파가 여섯 가지를 돌려줍니다. 학습 함수는 이 가운데 예측값, VAE 손실, 보간 손실 세 가지를 씁니다. 되살린 주기, 주기 성분, 채운 창은 쓰지 않습니다."),
      notes={src.line(BIVA, "return states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x"): "fit 은 1·2·5번째만 씀"},
      code=C(BIVA, [src.line(BIVA, "return states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x")],
             src.line(BIVA, "return states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x"), "Model.forward", 2, 2),
      stack=stk_fwd, nodes=["comb"],
      tensors=[{"type": "table", "title": "forward 가 돌려주는 것", "columns": ["순서", "이름", "모양", "fit 에서"], "rows": [
          [1, "states", str(list(st.shape)), "예측 → MSE"], [2, "VAE_loss", "[] (숫자 하나)", "손실에 더함"], [3, "recon_output", str(list(recon.shape)), "안 씀"],
          [4, "seasonal_init", str(list(seas.shape)), "안 씀"], [5, "imputed_loss", "[]", "손실에 더함"], [6, "imputed_x", str(list(imp.shape)), "안 씀"]]}])

    # ================================================================== 18. 손실·역전파·Adam
    mse = cap["outputs"]["mse"]
    S("18. 손실 · 역전파 · Adam",
      f"fit 이 세 손실을 <b>계수 없이 1 : 1 : 1 로 그대로 더합니다</b>(보간 손실 안에는 이미 imputed_weight {p['imputed_weight']} 와 일관성 손실의 × 0.1 이 들어 있음). 고른 가중치로 이 배치를 넣으면 VAE {f(vl)} + 보간 {f(lsum)} + 예측 MSE {f(mse)} = <b>{f(vl + lsum + mse)}</b> 입니다. "
      "예측 MSE 는 states 와 표준화한 정답 y_z 의 평균 제곱 차이입니다.",
      say=(f"학습 함수가 세 손실을 계수 없이 일 대 일 대 일로 그대로 더합니다. 고른 가중치로 이 배치를 넣으면 VAE 손실 {f(vl)}, 보간 손실 {f(lsum)}, 예측 오차 {f(mse)}를 더해 {f(vl + lsum + mse)}입니다. "
           "예측 오차는 예측값과 표준화한 정답 차이의 제곱 평균입니다."),
      notes={src.line(WRAP, "loss_states = criterion(states, by.to(self.device))"): f"예측 MSE = {f(mse)}", src.line(WRAP, "loss = VAE_loss + imputed_loss + loss_states"): f"{f(vl)} + {f(lsum)} + {f(mse)} = {f(vl + lsum + mse)}",
             src.line(WRAP, "loss.backward()"): "모든 가중치의 기울기", src.line(WRAP, "opt.step()"): "Adam 으로 가중치 갱신"},
      lab=lab_loss(st[:, 0, 0].numpy(), cap["batch"]["y_z"], vl, lsum, mse, vl + lsum + mse),
      code=C(WRAP, src.span(WRAP, "for bx, by in tr:", "opt.step()"), src.line(WRAP, "loss = VAE_loss + imputed_loss + loss_states"), "Model.fit", 2, 3),
      stack=stk_fit, nodes=["loss"], edges=[("comb", "loss"), ("vl", "loss")],
      formula=f"MSE = 평균((states − y_z)²) = {f(mse)},   손실 = {f(vl)} + {f(lsum)} + {f(mse)} = {f(vl + lsum + mse)}",
      tensors=[{"type": "table", "title": "예측 MSE 를 이루는 값 (이 배치, 고른 가중치)", "columns": ["배치", "states (예측)", "정답 y_z", "차이", "차이²"],
                "rows": [[f"b{i}", f"{float(st[i, 0, 0]):.4f}", f"{cap['batch']['y_z'][i]:.4f}", f"{float(st[i, 0, 0]) - cap['batch']['y_z'][i]:.4f}",
                          f"{(float(st[i, 0, 0]) - cap['batch']['y_z'][i]) ** 2:.4f}"] for i in range(B)]},
               {"type": "num", "title": "세 손실", "items": [{"label": "VAE 손실", "value": f(vl)}, {"label": "보간 손실(BRITS)", "value": f(lsum)}, {"label": "예측 MSE", "value": f(mse)},
                                                        {"label": "합 = loss", "value": f(vl + lsum + mse), "color": "#c8710f"}]}],
      checks=[chk("MSE 손 계산", mse, float(np.mean((st[:, 0, 0].numpy() - np.array(cap["batch"]["y_z"])) ** 2)))])
    gn = fs["grad_norms"]
    S("18. 손실 · 역전파 · Adam",
      "<b>loss.backward()</b> 가 모든 가중치의 기울기를 계산합니다. 표는 <b>실제 학습의 맨 첫 갱신</b>에서 잰 모듈별 기울기 크기입니다. "
      "추세 경로(Conv1d_Trend·Linear_Trend)의 기울기가 <b>0</b> 입니다. 첫 갱신 때 α = 1 이라 추세 몫 (1 − α) 이 0 이기 때문입니다. "
      "추세 경로는 α 가 1 아래로 내려온 뒤부터 학습됩니다.",
      say=("역전파가 모든 가중치의 기울기를 계산합니다. 표는 실제 학습의 맨 첫 갱신에서 잰 모듈별 기울기 크기입니다. 추세 경로의 기울기가 0입니다. "
           "첫 갱신 때 알파가 1이라서, 추세의 몫인 1 빼기 알파가 0이기 때문입니다. 추세 경로는 알파가 1 아래로 내려온 뒤부터 학습됩니다."),
      notes={src.line(WRAP, "loss.backward()"): "기울기 계산 — 첫 갱신 때 추세 경로는 0 (α = 1)"},
      code=C(WRAP, [src.line(WRAP, "loss.backward()"), src.line(WRAP, "opt.step()")], src.line(WRAP, "loss.backward()"), "Model.fit", 4, 4),
      stack=stk_fit, nodes=["back"], edges=[("loss", "back")],
      tensors=[{"type": "table", "title": "첫 갱신의 모듈별 기울기 크기 ‖∇‖ (실제 학습 기록)", "columns": ["모듈", "기울기 크기"],
                "rows": [[k_, num(v)] for k_, v in gn.items()], "hlRows": [i for i, k_ in enumerate(gn) if gn[k_] == 0]}],
      hint="BRITS 안의 temp_decay_x·weight_combine 은 출력에 이어지지 않아 기울기가 아예 생기지 않습니다(grad = None, 「7. BRITS 앞→뒤」). 그래서 BRITS 의 기울기 크기는 나머지 부분에서만 옵니다.")
    lr, (b1_, b2_), eps = fs["lr"], fs["betas"], fs["eps"]
    S("18. 손실 · 역전파 · Adam",
      f"<b>opt.step()</b>: Adam 이 기울기로 가중치를 고칩니다. 첫 갱신에서 α 의 기울기는 {f(fs['alpha_grad'])} 이고, α 는 {f(fs['alpha_before'])} → <b>{f(fs['alpha_after'])}</b> 로 바뀌었습니다. "
      f"Adam 의 첫 걸음은 기울기가 0 이 아니면 그 크기와 상관없이, 기울기 반대 방향으로 거의 학습률({lr})만큼 움직입니다(기울기가 0 인 추세 경로는 움직이지 않음).",
      say=(f"아담이 기울기로 가중치를 고칩니다. 첫 갱신에서 알파의 기울기는 {f(fs['alpha_grad'])}입니다. 알파는 {f(fs['alpha_before'])}에서 {f(fs['alpha_after'])}로 바뀌었습니다. "
           f"아담의 첫 걸음은 기울기가 0이 아니면 그 크기와 상관없이, 기울기 반대 방향으로 거의 학습률 {lr}만큼 움직입니다."),
      notes={src.line(WRAP, "opt = optim.Adam(self.net.parameters(), lr=p[\"learning_rate\"])"): f"Adam, 학습률 {lr}",
             src.line(WRAP, "opt.step()"): f"첫 갱신: α {f(fs['alpha_before'])} → {f(fs['alpha_after'])}"},
      lab=lab_adam(fs["alpha_grad"], fs["alpha_before"], lr, eps, fs["alpha_after"]),
      code=C(WRAP, [src.line(WRAP, "opt = optim.Adam(self.net.parameters(), lr=p[\"learning_rate\"])"), src.line(WRAP, "opt.step()")], src.line(WRAP, "opt.step()"), "Model.fit", 14, 3),
      stack=stk_fit + ["torch.optim.Adam.step"], nodes=["back"],
      formula=f"m = (1−β₁)g, v = (1−β₂)g², m̂ = m/(1−β₁) = g, v̂ = v/(1−β₂) = g²\nΔ = lr · m̂ / (√v̂ + ε) = {lr} × {f(fs['alpha_grad'])}/({f(abs(fs['alpha_grad']))} + {eps}) ≈ {f(lr * fs['alpha_grad'] / (abs(fs['alpha_grad']) + eps))}  →  α = {f(fs['alpha_before'])} − Δ = {f(fs['alpha_before'] - lr * fs['alpha_grad'] / (abs(fs['alpha_grad']) + eps))}",
      checks=[chk("Adam 첫 걸음 손 계산 = 실제 α 변화", fs["alpha_after"], fs["alpha_before"] - lr * fs["alpha_grad"] / (abs(fs["alpha_grad"]) + eps), 1e-6)])
    hist = cap["history"]
    best = min(hist, key=lambda h: h["vali_loss"])
    E_ = len(hist)
    # 표에 보일 epoch: 많으면 앞 12개 · 최저 근처 · 50번째마다 · 마지막만(그래프에는 전부)
    show_ep = sorted({e for e in range(1, E_ + 1) if E_ <= 24 or e <= 12 or abs(e - best["epoch"]) <= 1 or e % 50 == 0 or e == E_})
    upd = [h["epoch"] for i, h in enumerate(hist) if h["vali_loss"] == min(x["vali_loss"] for x in hist[:i + 1])]  # 가중치를 보관한 epoch
    tl_ = [h["train_loss"] for h in hist]
    n_up = sum(1 for a_, b_ in zip(tl_, tl_[1:]) if b_ > a_)   # 학습 손실이 앞 epoch 보다 오른 횟수(2026-11-08: '계속 줄었지만' 은 사실과 달라 고침)
    ep_min = int(np.argmin(tl_)) + 1
    tr_trend = (f"줄었지만(epoch 사이 {n_up}번은 올랐고, 가장 낮은 값은 epoch {ep_min} 의 {f(tl_[ep_min - 1])})" if tl_[-1] < tl_[0] else "줄지 않았고")
    late = best["epoch"] <= max(2, E_ // 20)
    vl_ = np.array([h["vali_loss"] for h in hist])
    blk = 50 if E_ >= 200 else max(1, E_ // 4)
    vmed0, vmed1 = float(np.median(vl_[:blk])), float(np.median(vl_[-blk:]))      # 첫·마지막 구간의 검증 손실 중앙값
    tmed0, tmed1 = float(np.median(tl_[:blk])), float(np.median(tl_[-blk:]))
    rising = vmed1 > 2 * vmed0 and tmed1 < tmed0                                   # 학습 손실은 줄고 검증 손실은 커짐
    lr_ep = next((h["epoch"] for h in hist if h["lr"] < hist[0]["lr"]), None)
    S("18. 손실 · 역전파 · Adam",
      f"한 epoch 은 학습 배치 {n_tr // p['batch_size']}개를 차례로 처리한 것입니다(배치마다 순전파 → 손실 → 역전파 → Adam). 그다음 검증 배치 {n_va // p['batch_size']}개로 <b>검증 손실</b>을 잽니다(_vali, 가중치는 고치지 않음). "
      f"EarlyStopping 이 지금까지 가장 낮은 검증 손실의 가중치를 보관합니다. 그 뒤 <b>adjust_learning_rate</b> 가 학습률을 정합니다(lradj 3: 10 epoch 이 끝난 뒤부터 처음 값 × 0.1 = {f(p['learning_rate'] * 0.1)} 로 고정 — 계속 곱하지 않음).",
      say=(f"한 에포크는 학습 배치 {n_tr // p['batch_size']}개를 차례로 처리한 것입니다. 배치마다 순전파, 손실, 역전파, 아담 순서입니다. "
           f"그다음 검증 배치 {n_va // p['batch_size']}개로 검증 손실을 잽니다. 이때는 가중치를 고치지 않습니다. 얼리 스토핑이 지금까지 가장 낮은 검증 손실의 가중치를 보관합니다. "
           f"그 뒤 학습률 조정 함수가 학습률을 정하는데, 열 번째 에포크가 끝난 뒤부터는 학습률을 처음 값의 0.1배인 {f(p['learning_rate'] * 0.1)}로 고정합니다."),
      notes={src.line(WRAP, 'early(vali_loss, self.net, "", "")'): f"검증 손실이 최저면 가중치 보관 (이 실행: epoch {best_ep})",
             src.line(WRAP, "adjust_learning_rate(opt, epoch + 1, self.args)"): f"lradj 3: 10 epoch 이 끝난 뒤부터 처음 값 × 0.1 = {f(p['learning_rate'] * 0.1)} (고정)"},
      code=C(WRAP, src.span(WRAP, "train_loss = float(np.average(train_loss))", "adjust_learning_rate(opt, epoch + 1, self.args)"), src.line(WRAP, "early(vali_loss, self.net, \"\", \"\")"), "Model.fit", 2, 2),
      stack=stk_fit, nodes=["epoch"], edges=[("back", "epoch")],
      tensors=[{"type": "line", "title": f"실제 학습 기록 — {len(hist)} epoch", "x": [str(h["epoch"]) for h in hist], "xlabel": "epoch", "ylog": True, "height": 250,
                "series": [{"name": "학습 손실", "color": "#3b6fb6", "y": [num(h["train_loss"]) for h in hist], "dots": E_ <= 40}, {"name": "검증 손실", "color": "#c8710f", "y": [num(h["vali_loss"]) for h in hist], "dots": E_ <= 40}],
                "vlines": [{"i": best["epoch"] - 1, "label": f"검증 최저 epoch {best['epoch']}", "color": "#2f8f5b"}] + ([{"i": lr_ep - 1, "label": f"학습률 {f(hist[0]['lr'])}→{f(hist[lr_ep - 1]['lr'])}", "color": "#b3261e"}] if lr_ep else [])},
               {"type": "table", "split": 2 if len(show_ep) > 14 else 1, "title": "epoch 별 값" + ("" if len(show_ep) == E_ else f" (일부: 앞 12개 · 최저 근처 · 50번째마다 · 마지막 — 전체 {E_}개는 그래프)"),
                "columns": ["epoch", "학습 손실", "검증 손실", "학습률", "시작 때 α"],
                "rows": [[e, num(hist[e - 1]["train_loss"]), num(hist[e - 1]["vali_loss"]), num(hist[e - 1]["lr"]), num(cap["alpha_epoch"].get(e, float("nan")))] for e in show_ep],
                "hlRows": [show_ep.index(best["epoch"])]}],
      hint="tools_orig.adjust_learning_rate 는 epoch 이 끝난 뒤 epoch+1 로 불립니다. lradj '3' 은 넘겨받은 수(epoch + 1) < 10 이면 처음 값 그대로, 10 이상이면 처음 값 0.01 × 0.1 = 0.001 이라서, 10번째 epoch 이 끝난 뒤(11번째부터) 0.001 로 고정됩니다.")
    S("18. 손실 · 역전파 · Adam",
      "fit 이 epoch 마다 부르는 원본 도구 두 개입니다(tools_orig.py, 원본 utils/tools.py 그대로). <b>EarlyStopping</b> 은 검증 손실이 지금까지의 최저보다 낮거나 같으면 가중치를 보관하고(save_checkpoint) 세는 수를 0 으로, "
      f"최저보다 높으면 세는 수를 하나 늘립니다(delta = 0). 세는 수가 patience({p['patience']})에 닿으면 학습을 멈춥니다. 이 플랫폼은 save_checkpoint 만 바꿔 파일 대신 메모리에 보관합니다(_MemEarlyStopping). "
      f"<b>adjust_learning_rate</b> 는 lradj '3' 이면 넘겨받은 수(epoch + 1, 곧 방금 끝난 epoch 의 번호)가 10 보다 작을 때 {p['learning_rate']}, 10 이상이면 {p['learning_rate']} × 0.1 로 학습률을 바꿉니다.",
      say=(f"학습 함수가 에포크마다 부르는 원본 도구 두 개입니다. 얼리 스토핑은 검증 손실이 지금까지의 최저보다 낮거나 같으면 가중치를 보관하고, 세는 수를 0으로 되돌립니다. "
           f"최저보다 높으면 세는 수를 하나 늘립니다. 세는 수가 인내 값 {p['patience']}에 닿으면 학습을 멈춥니다. 이 플랫폼은 가중치를 파일 대신 메모리에 보관하도록 그 부분만 바꿨습니다. "
           f"학습률 조정 함수는 방금 끝난 에포크 번호가 10보다 작으면 학습률 {p['learning_rate']}, 10 이상이면 {f(p['learning_rate'] * 0.1)}로 바꿉니다. "
           + (f"실제 기록에서 가중치를 보관한 것은 에포크 {upd[0]}, 한 번뿐입니다." if len(upd) == 1 else f"실제 기록에서 가중치를 보관한 것은 {len(upd)}번입니다.")),
      notes={src.line(TOOLS, "elif score < self.best_score + self.delta:"): f"score = −검증 손실: 최저보다 높으면 세기 + 1 (patience {p['patience']}) · 낮거나 같으면 아래 else 에서 보관"},
      code=C(TOOLS, src.span(TOOLS, "def __call__(self, val_loss, model, path, model_id):", "self.counter = 0") + src.span(TOOLS, "elif args.lradj == '3':", "10 else args.learning_rate*0.1}"),
             src.line(TOOLS, "elif score < self.best_score + self.delta:"), "EarlyStopping.__call__ · adjust_learning_rate", 2, 2),
      stack=stk_fit + ["tools_orig.py · EarlyStopping.__call__ / adjust_learning_rate"], nodes=["epoch"],
      formula=f"검증 손실 ≤ 지금까지 최저 → 가중치 보관, 세기 0 · 검증 손실 > 최저 → 세기 + 1 (세기 ≥ {p['patience']} 이면 멈춤)\n학습률 = {p['learning_rate']} (epoch+1 < 10) · {f(p['learning_rate'] * 0.1)} (epoch+1 ≥ 10)",
      tensors=[{"type": "table", "title": f"실제 기록: 가중치를 보관한 epoch (검증 손실이 그때까지의 최저를 갱신) — {len(upd)}번", "columns": ["epoch", "검증 손실", "그 뒤 세기"],
                "rows": [[e, num(hist[e - 1]["vali_loss"]), (upd[k + 1] - e - 1) if k + 1 < len(upd) else E_ - e] for k, e in enumerate(upd)], "hlRows": [upd.index(best["epoch"])]},
               {"type": "num", "title": "결과", "items": [{"label": "마지막 세기(counter)", "value": E_ - best["epoch"]}, {"label": "patience", "value": p["patience"]},
                                                       {"label": "조기종료?", "value": "예" if E_ - best["epoch"] >= p["patience"] else "아니오(끝까지 돎)"}]}],
      hint=(f"원본 설정은 patience 도 {p['patience']} 입니다. "
            + (f"세기는 많아야 epoch 수 − 1 = {p['train_epochs'] - 1} 이라 이 설정에서는 '조기종료'로 멈출 수 없고, " if p["patience"] >= p["train_epochs"] else "그래서 '조기종료'로 멈추는 일은 드물고, ")
            + "정해진 epoch 을 모두 돈 뒤 검증 손실이 가장 낮았던 시점의 가중치를 고르는 방식이 됩니다."))
    S("18. 손실 · 역전파 · Adam",
      f"학습이 끝나면 <b>검증 손실이 가장 낮았던 epoch({best['epoch']})의 가중치</b>로 되돌립니다(α = {f(cap['final_alpha'])}). "
      f"이 실행에서 학습 손실은 {f(hist[0]['train_loss'])} → {f(hist[-1]['train_loss'])} 로 {tr_trend}, 검증 손실은 epoch {best['epoch']} 에서 {f(best['vali_loss'])} 로 가장 낮았고 마지막에는 {f(hist[-1]['vali_loss'])} 입니다. "
      + (f"곧 {E_} epoch 중 <b>epoch {best['epoch']}(갱신 {best['epoch'] * (n_tr // p['batch_size'])}번 뒤)의 가중치</b>가 쓰입니다 — 그 뒤의 학습은 검증 손실을 낮추지 못했습니다. " if late else "")
      + (f"{blk} epoch 구간의 중앙값으로 보면 학습 손실은 {f(tmed0, 3)} → {f(tmed1, 3)} 로 줄었지만 검증 손실은 {f(vmed0, 1)} → {f(vmed1, 1)} 로 <b>꾸준히 커졌습니다</b> — "
         "과적합으로 볼 수 있는 모습입니다. 다만 검증 손실은 세 손실(VAE·보간·예측)의 합이라, 어느 손실이 커졌는지는 따로 나누어 보지 않았습니다. " if rising else "")
      + f"검증 손실은 epoch 사이에도 흔들립니다(검증 창 {n_va}개, 배치 {n_va // p['batch_size']}개).",
      say=(f"학습이 끝나면 검증 손실이 가장 낮았던 에포크 {best['epoch']}의 가중치로 되돌립니다. 이때 알파는 {f(cap['final_alpha'])}입니다. "
           f"이 실행에서 학습 손실은 {f(hist[0]['train_loss'])}에서 {f(hist[-1]['train_loss'])}로 "
           + (f"줄었습니다. 에포크 사이에 {n_up}번은 올랐고, 가장 낮은 값은 에포크 {ep_min}의 {f(tl_[ep_min - 1])}입니다. " if tl_[-1] < tl_[0] else "줄지 않았습니다. ")
           + f"그런데 검증 손실은 에포크 {best['epoch']}에서 {f(best['vali_loss'])}로 가장 낮았고, 마지막에는 {f(hist[-1]['vali_loss'])}입니다. "
           + (f"곧 {E_}번의 에포크 가운데 에포크 {best['epoch']}, 갱신 {best['epoch'] * (n_tr // p['batch_size'])}번 뒤의 가중치가 쓰입니다. 그 뒤의 학습은 검증 손실을 낮추지 못했습니다. " if late else "")
           + (f"{blk} 에포크 구간의 중앙값으로 보면, 학습 손실은 {f(tmed0, 3)}에서 {f(tmed1, 3)}로 줄었지만, 검증 손실은 {f(vmed0, 1)}에서 {f(vmed1, 1)}로 꾸준히 커졌습니다. "
              "과적합으로 볼 수 있는 모습입니다. 다만 검증 손실은 세 손실의 합이라, 어느 손실이 커졌는지는 따로 나누어 보지 않았습니다. " if rising else "")
           + f"검증 손실은 에포크 사이에도 흔들립니다. 검증 창은 {n_va}개, 배치로는 {n_va // p['batch_size']}개입니다."),
      notes={src.line(WRAP, "self.net.load_state_dict(early.best_state)"): f"검증 최저 epoch {best['epoch']} 의 가중치로 되돌림 (α = {f(cap['final_alpha'])})"},
      code=C(WRAP, src.lines(WRAP, "if early.best_state is not None:", "self.net.load_state_dict(early.best_state)") + src.span(WRAP, "class _MemEarlyStopping(EarlyStopping):", "self.val_loss_min = val_loss"),
             src.line(WRAP, "self.net.load_state_dict(early.best_state)"), "Model.fit / EarlyStopping", 2, 2),
      stack=stk_fit + ["_MemEarlyStopping.best_state"], nodes=["epoch"],
      tensors=[{"type": "num", "title": "학습 결과", "items": [{"label": "돈 epoch", "value": len(hist)}, {"label": "선택된 epoch", "value": best["epoch"], "color": "#2f8f5b"},
                                                         {"label": "그때 검증 손실", "value": f(best["vali_loss"])}, {"label": "마지막 학습 손실", "value": f(hist[-1]["train_loss"])}, {"label": "선택된 α", "value": f(cap["final_alpha"])}]}],
      hint=(f"이 영상은 {E_} epoch 실행입니다(원본 {ORIG_EPOCHS}). " if E_ != ORIG_EPOCHS else f"원본과 같은 {E_} epoch 실행입니다. ")
           + ("이 설정은 폴드가 1개(1회 분할)라서 이 기록이 엔진 실행 전체와 같습니다(「19. 예측」에서 결과 파일과 대조). " if cfg["validation"]["mode"] == "split" else
              "이 기록은 마지막 폴드 하나만 학습한 것입니다. ")
           + "폴드가 여러 개인 설정에서는 폴드를 차례로 학습해 난수 흐름이 달라지므로, 같은 폴드라도 선택 epoch 이 다를 수 있습니다. "
           "검증 창이 적을 때 시점 선택이 우연에 좌우될 수 있다는 점은 논문의 한계로 적을 만합니다.")

    # ================================================================== 19. 예측
    pr = cap["predict"]
    wpred = pr["window"]
    zp = (pr["pred"] - cap["y_mu"]) / cap["y_sd"]
    S("19. 예측",
      f"predict 는 eval 모드에서 테스트 창(이 폴드 {pr['n_test']}개)을 {p['batch_size']}개씩 forward 에 넣습니다. <b>states[:, −1, 0]</b>(창마다 숫자 하나)을 꺼내 표준화를 되돌립니다: ŷ = states × σ_y + μ_y (σ_y·μ_y: 학습 창 정답의 표준편차·평균 — VAE 의 μ·σ 와 다름). "
      f"결정월 {pr['date'][:7]}: ŷ = {f(zp)} × {g4(cap['y_sd'])} + {g4(cap['y_mu'])} = <b>{f(pr['pred'])}</b>. "
      f"F11 {f(pr['level'])} → 다음 달 예측 {f(pr['level'] * (1 + pr['pred']))}. 실제 변화율은 {f(pr['actual'])} 였습니다.",
      say=(f"예측 단계입니다. 평가 모드에서 테스트 창, 이 폴드에서는 {pr['n_test']}개를 {p['batch_size']}개씩 순전파에 넣습니다. 창마다 나오는 숫자 하나에 학습 창 정답의 표준편차를 곱하고 평균을 더해 표준화를 되돌립니다. "
           f"결정월 {pr['date'][:7]}의 예측은 {f(zp)} 곱하기 {g4(cap['y_sd'])} 더하기 {g4(cap['y_mu'])}로, {f(pr['pred'])}입니다. "
           f"F11 {f(pr['level'])}에서 다음 달 예측은 {f(pr['level'] * (1 + pr['pred']))}입니다. 실제 변화율은 {f(pr['actual'])}입니다."),
      notes={src.line(WRAP, "pred = np.concatenate(out).astype(np.float64) * self.y_sd + self.y_mu"): f"ŷ = 출력 × σ_y + μ_y  예) {f(zp)} × {g4(cap['y_sd'])} + {g4(cap['y_mu'])} = {f(pr['pred'])}"},
      lab=lab_pred(zp, cap["y_sd"], cap["y_mu"], pr["level"], pr["date"][:7], pr["pred"], pr["level"] * (1 + pr["pred"])),
      code=C(WRAP, src.span(WRAP, "def predict(self, X):", "return pred"), src.line(WRAP, "pred = np.concatenate(out).astype(np.float64) * self.y_sd + self.y_mu"), "Model.predict", 1, 1),
      stack=["engine/experiment.py · _walkforward_predict", "models/biva.py · Model.predict"], nodes=["pred"], edges=[("epoch", "pred")],
      formula=f"ŷ = states[:, −1, 0] × σ_y + μ_y = {f(zp)} × {g4(cap['y_sd'])} + {g4(cap['y_mu'])} = {f(pr['pred'])}   (다음 달 F11 = {f(pr['level'])} × (1 + ŷ) = {f(pr['level'] * (1 + pr['pred']))})",
      tensors=[H(f"결정월 {pr['date'][:7]} 의 테스트 창 (학습 구간으로 맞춘 눈금 — 테스트 값은 0~1 밖일 수 있음)", wpred, rows=[f"τ{t + 1} {d[:7]}" for t, d in enumerate(pr["window_dates"])], cols=feats, shape=[L, K], scale="01")])
    reps = np.array(pr["repeats"])
    from models.biva import Model as _MB            # 고른 가중치로 이 테스트 창의 μ·log σ²(ReLU 앞)를 다시 계산
    _m = _MB(cap["params"], n_features=K, target_type="regression", device="cpu")
    _net = _m._build(L); _net.load_state_dict(cap["net_state"]); _net.eval()
    _vv = {}
    _hs = [_net.LSTM_VAE.encoder_fc_mu.register_forward_hook(lambda m_, i_, o_: _vv.__setitem__("mu", o_.detach())),
           _net.LSTM_VAE.encoder_fc_logvar.register_forward_hook(lambda m_, i_, o_: _vv.__setitem__("lv", o_.detach()))]
    with torch.no_grad():
        torch.manual_seed(0)        # 기록의 100번 반복 중 시드 0 과 같은 조건
        te_rep0 = float(_net(torch.tensor(np.asarray(wpred, dtype=np.float32)[None]))[0][0, -1, 0]) * cap["y_sd"] + cap["y_mu"]
    for _h in _hs:
        _h.remove()
    te_pos = int((_vv["mu"] > 0).sum()) + int((_vv["lv"] > 0).sum()); te_n = int(_vv["mu"].numel())
    te_dead = te_pos == 0
    y_all = pr["y_all"]; te0, te1 = cap["fold"]["test"]
    y_te_sd = float(y_all.loc[te0:te1].dropna().std(ddof=0))
    S("19. 예측",
      f"z 를 예측 때도 새로 뽑으므로 <b>같은 창도 계산할 때마다 값이 조금씩 다릅니다</b>. 같은 창을 무작위 시드만 바꿔 {len(reps)}번 예측하면 평균 {f(reps.mean())}, 표준편차 {f(reps.std())} 입니다. "
      f"엔진은 한 번 계산한 값({f(pr['pred'])})을 씁니다. 같은 결과를 다시 얻으려면 같은 시드, 같은 순서, 같은 CPU 스레드 수로 돌려야 합니다. "
      + (f"이 테스트 창에서도 μ·log σ² 의 ReLU 앞 값이 모두 음수입니다(양수 {te_pos}칸 / μ·log σ² 각 {te_n}칸, 고른 가중치로 다시 계산). 그래서 μ = 0, σ = 1 이라 z = ε 그대로입니다 — "
         f"BRITS 등 나머지 계산에는 무작위 요소가 없으므로, 흔들림은 전부 표준정규 잡음 z 가 주기 경로를 지나며 생긴 것입니다. 흔들림의 크기(표준편차 {f(reps.std())})는 "
         f"이 폴드 테스트 기간 실제 변화율의 표준편차 {f(y_te_sd)} 의 {f(reps.std() / y_te_sd * 100, 1)}% 입니다." if te_dead else ""),
      say=(f"z를 예측할 때도 새로 뽑으므로, 같은 창도 계산할 때마다 값이 조금씩 다릅니다. 같은 창을 무작위 시드만 바꿔 {len(reps)}번 예측하면 평균 {f(reps.mean())}, 표준편차 {f(reps.std())}입니다. "
           f"엔진은 한 번 계산한 값 {f(pr['pred'])}을 씁니다. 같은 결과를 다시 얻으려면 같은 시드, 같은 순서, 같은 CPU 스레드 수로 돌려야 합니다. "
           + (f"이 테스트 창에서도 뮤와 로그 분산의 렐루 앞 값이 모두 음수여서, 뮤가 0, 시그마가 1이고 z는 엡실론 그대로입니다. "
              "나머지 계산에는 무작위 요소가 없으므로, 흔들림은 전부 표준정규 잡음 z가 주기 경로를 지나며 생긴 것입니다. "
              f"흔들림의 크기는 이 폴드 테스트 기간 실제 변화율 표준편차 {f(y_te_sd)}의 {f(reps.std() / y_te_sd * 100, 1)}%입니다." if te_dead else "")),
      notes={src.line(VAE, "eps = torch.randn_like(std)"): f"예측 때도 새로 뽑음 → 같은 창 {len(reps)}번: 표준편차 {f(reps.std())}"},
      code=C(VAE, [src.line(VAE, "eps = torch.randn_like(std)")], src.line(VAE, "eps = torch.randn_like(std)"), "reparameterize", 4, 4),
      stack=["models/biva.py · Model.predict", "BIVA.py · forward", "BIVA_LSTM_VAE.py · reparameterize"], nodes=["pred", "z"],
      checks=[chk("다시 만든 신경망(고른 가중치)의 이 창 예측(시드 0) = 기록의 시드 0 예측 — μ·log σ² 확인에 쓴 신경망이 같음", te_rep0, float(reps[0]), 1e-9)],
      tensors=[{"type": "line", "title": "같은 창 100번 예측 (시드 0~99)", "x": [str(i) for i in range(len(reps))], "xlabel": "시드",
                "series": [{"name": "ŷ", "color": "#c8710f", "y": [num(v, 6) for v in reps], "dots": True}], "height": 200}])
    ap_ = pr.get("all_pred")
    mr = match_result("forecast", cap["config"].get("name", ""), {str(d)[:10]: v for d, v in ap_.items()}) if ap_ is not None else None
    if mr:
        same = mr["max_diff"] < 1e-12
        others = []
        jme = json.loads((ROOT / "results" / "forecast" / mr["file"].replace("_daily.parquet", ".json")).read_text(encoding="utf-8"))
        for a_, n_, d_ in mr["all"]:
            if a_ == mr["file"]:
                continue
            jf = ROOT / "results" / "forecast" / a_.replace("_daily.parquet", ".json")
            jo = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else {}
            co = jo.get("config") or {}
            dif = []
            ids_o, ids_me = co.get("variable_ids") or [], cfg.get("variable_ids") or []
            if ids_o != ids_me:
                miss_ = [v for v in ids_me if v not in ids_o]
                dif.append(f"변수 {len(ids_o)}개" + (f"({', '.join(miss_)} 없음)" if miss_ else ""))
            if not jo.get("engine_rev"):
                dif.append("이전 엔진(engine_rev 기록 없음)")
            elif jo.get("engine_rev") != jme.get("engine_rev"):
                dif.append(f"엔진 판 {jo.get('engine_rev')}")
            if jo.get("device") and jo.get("device") != jme.get("device"):
                dif.append(f"장치 {jo['device']}")
            pk_ = ("first_fold", "last_fold", "test", "embargo")            # 실제 학습·검증·테스트 구간(적는 규칙이 달라도 구간이 같으면 같음)
            same_per = bool(jo.get("periods")) and all((jo["periods"] or {}).get(k) == (jme.get("periods") or {}).get(k) for k in pk_)
            others.append((a_.replace("_daily.parquet", ""), d_, dif, same_per))
        oth_txt = (" 표의 다른 결과 파일은 이전 실행입니다: " + "; ".join(f"{a_} — {', '.join(dif) or '설정 차이 못 찾음'}" + (" (학습·검증·테스트 구간은 같음)" if sp_ else "") + f", 최대 차이 {d_:.1e}"
                                                         for a_, d_, dif, sp_ in others)
                   + ". 그래서 끝자리가 아니라 값 자체가 다릅니다.") if others and same else ""
        S("19. 예측",
          f"이 영상의 테스트 예측 {len(ap_)}개를 결과 폴더의 같은 실험 결과 <b>{mr['file'].replace('_daily.parquet', '')}</b> 와 맞춰 봅니다: 최대 차이 {mr['max_diff']:.1e}"
          + (" — <b>결과 화면의 숫자와 같은 계산</b>입니다." if same else " — 다릅니다(아래 표: 설정·엔진이 다른 때의 결과는 다른 것이 정상).")
          + " 신경망은 같은 시드라도 CPU 스레드 수·GPU 에 따라 덧셈 순서가 달라 소수 끝자리가 다를 수 있습니다(이 설정에서 스레드 2개와 1개의 epoch 1 학습 손실이 소수 9자리에서 다름). "
          "이 기록과 결과 파일은 같은 조건(클라우드 CPU 1스레드)에서 만들었습니다." + oth_txt,
say=(f"이 영상의 테스트 예측 {len(ap_)}개를 결과 폴더에 저장된 같은 실험의 결과와 맞춰 봅니다. "
     + (f"최대 차이는 {mr['max_diff']:g}로, 결과 화면의 숫자와 같은 계산입니다. " if same else f"최대 차이는 {mr['max_diff']:.1e}로 다릅니다. 설정이나 엔진이 다른 때의 결과는 다른 것이 정상입니다. ")
     + "신경망은 같은 시드라도 CPU 스레드 수나 GPU에 따라 덧셈 순서가 달라져 소수 끝자리가 다를 수 있습니다. 이 기록과 결과 파일은 같은 조건, 곧 클라우드 CPU 한 스레드에서 만들었습니다."
     + (" 표의 다른 결과 파일들은 입력 변수나 엔진이 다른 이전 실행이라서, 끝자리가 아니라 값 자체가 다릅니다." if others and same and all(dif for _, _, dif, _ in others) else "")),
          stack=["results/forecast/" + mr["file"]], nodes=["pred"],
          tensors=[{"type": "table", "title": "결과 파일별 비교", "columns": ["결과 파일", "겹친 예측 수", "최대 차이"], "rows": [[a_.replace("_daily.parquet", ""), n_, f"{d_:.1e}"] for a_, n_, d_ in mr["all"]]}],
          checks=[chk(f"영상의 예측 = 결과 파일 {mr['file'].replace('_daily.parquet', '')} 의 예측 ({mr['n']}개)", mr["max_diff"], 0.0, 1e-12)] if same else [])
    n_ok = sum(c["ok"] for c in checks_all)
    S("19. 예측",
      f"정리: 이 영상이 따라간 배치의 forward 에서 BRITS 가 빈 칸 {cap['batch']['n_nan']}개를 채웁니다(LSTMCell {2 * L}번). 분해가 추세·주기를 나누고, 추세는 Conv1d 3단 + Linear 를 거칩니다. "
      f"주기는 인코더 LSTM 4층을 거친 출력 e 와 z 를 Conv1d 3단 + Linear 에 넣고, 두 경로의 출력을 α 로 합칩니다. 디코더 LSTM 4층은 VAE 재구성 손실 계산에만 쓰입니다. "
      "학습은 세 손실의 합을 Adam 으로 줄이고, 검증 손실이 가장 낮았던 epoch 을 고릅니다. "
      f"이 영상의 검증 {len(checks_all)}개(손 계산·직접 실행해 보기·결과 파일 대조) 중 <b>{n_ok}개가 일치</b>했습니다."
      + (f" 실행에서 확인한 사실: 고른 가중치에서 μ·log σ² 가 학습 폴드의 창 {va_['n_windows']}개와 이 영상의 테스트 창 모두에서 0(ReLU)이라 z 는 입력과 무관한 잡음입니다(「13. μ · log σ² · z」)." if dead and te_dead else ""),
say=(f"정리합니다. 이 영상이 따라간 배치의 순전파에서 BRITS가 빈 칸 {cap['batch']['n_nan']}개를 채우고, 이때 LSTM 셀을 {2 * L}번 계산합니다. 분해가 추세와 주기를 나눕니다. "
     "추세는 합성곱 세 단과 선형 층을 거칩니다. 주기는 인코더 LSTM 네 층을 거친 출력과 z를 합성곱 세 단과 선형 층에 넣습니다. 두 경로의 출력을 알파로 합칩니다. "
     "디코더 LSTM 네 층은 VAE 손실 계산에만 쓰입니다. 학습은 세 손실의 합을 아담으로 줄이고, 검증 손실이 가장 낮았던 에포크를 고릅니다. "
     f"이 영상의 검증 {len(checks_all)}개 가운데 {n_ok}개가 일치했습니다. 손 계산, 직접 실행해 보기, 결과 파일 대조를 모두 포함한 수입니다."
     + (" 실행에서 확인한 사실로, 고른 가중치에서는 뮤와 로그 분산이 학습 폴드의 창 모두와 이 영상의 테스트 창에서 0이라서, z는 입력과 무관한 잡음입니다." if dead and te_dead else "")),
      stack=["요약"], nodes=["fwd", "bwd", "merge", "dec", "tc", "tl", "e1", "e2", "e3", "e4", "mu", "z", "decd", "vl", "sc", "sl", "comb", "loss", "back", "epoch", "pred"],
      tensors=[{"type": "table", "title": "검증 전체 (손 계산 · 직접 실행해 보기 · 결과 파일 대조)", "columns": ["단계", "무엇", "결과"], "rows": [[c["step"], c["what"], "✓ 일치" if c["ok"] else "✗ 다름"] for c in checks_all]}],
      hint=f"'LSTMCell {2 * L}번'은 이 배치의 forward 한 번에서 앞→뒤·뒤→앞 RITS 가 시점마다 한 번씩 부른 수입니다. 빈 칸 수는 배치마다 다릅니다(이 영상은 마지막 epoch 의 학습 배치 {n_tr // p['batch_size']}개 중 빈 칸이 가장 많은 배치를 골랐음).")
    if n_ok != len(checks_all):
        bad = [c for c in checks_all if not c["ok"]]
        raise AssertionError(f"검증 실패 {len(bad)}개: {bad}")

    sources_used = sorted({s_["code"]["file"] for s_ in steps if s_.get("code")})
    return {"title": "BIVA 코드 따라가기 — 원본 코드가 실제로 실행되는 순서", "model": "biva", "epochs": cap["epochs"], "steps": steps,
            "sources": {fn: src.text[fn] for fn in sources_used}, "diagram": diagram(K, L, h1, h2, lat, c4, c16, Hn), "checks": checks_all}


def diagram(K, L, h1, h2, lat, c4, c16, Hn) -> dict:
    """위쪽 구조도. 좌표는 화면 1920 폭 기준(viewBox 1880). 꺾인 연결선은 path 로 직접 그린다(선이 상자를 가로지르지 않게)."""
    N = lambda id_, label, x, y, w, h, sub=None: {"id": id_, "label": label, "x": x, "y": y, "w": w, "h": h, "sub": sub}
    nodes = [N("data", "원자료 표", 10, 8, 140, 42, f"월 × 변수 {K}"), N("scale", "눈금 MinMax", 172, 8, 140, 42, "학습 폴드로"),
             N("win", "창 만들기", 334, 8, 140, 42, f"[N, {L}, {K}]"), N("split", "검증 · 비움", 496, 8, 150, 42, "학습·검증 창"),
             N("batch", "배치 DataLoader", 668, 8, 160, 42, f"[3, {L}, {K}]"),
             N("loss", "손실 합", 1130, 8, 150, 42, "VAE + BRITS + MSE"), N("back", "역전파 · Adam", 1302, 8, 150, 42, "loss.backward()"),
             N("epoch", "epoch · 조기종료", 1474, 8, 170, 42, "검증·학습률"), N("pred", "예측 · 되돌림", 1666, 8, 200, 42, "ŷ = s·σ + μ"),
             N("fwd", "RITS 앞→뒤", 44, 104, 150, 44, f"LSTMCell {2 * K}→{Hn} ×{L}"), N("bwd", "RITS 뒤→앞", 44, 186, 150, 44, f"뒤집어 ×{L}"),
             N("merge", "평균 · 일관성", 214, 145, 120, 44, "채운 창"), N("dec", "분해", 360, 145, 112, 44, "이동평균"),
             N("tc", "Conv1d ×3", 496, 82, 150, 40, f"{K}→{c4}→{c16}→1"), N("tl", "Linear 시간", 664, 82, 134, 40, f"{L}→1 추세 경로 출력"),
             N("e1", "LSTM 1", 496, 150, 92, 44, f"{K}→{K}"), N("e2", "LSTM 2", 600, 150, 92, 44, f"{K}→{h1}"), N("e3", "LSTM 3", 704, 150, 92, 44, f"{h1}→{h2}"),
             N("e4", "LSTM 4", 808, 150, 92, 44, f"{h2}→{lat}"), N("mu", "μ · log σ²", 914, 150, 114, 44, "Linear + ReLU"), N("z", "z = μ+ε·σ", 1042, 150, 114, 44, "무작위"),
             N("sc", "Conv1d ×3", 1172, 150, 160, 44, f"[e;z] {2 * lat}→{lat}→{lat // 2}→1"), N("sl", "Linear 시간", 1350, 150, 120, 44, f"{L}→1 주기 경로 출력"),
             N("decd", "디코더 LSTM ×4", 1172, 208, 160, 40, f"{2 * lat}→{h2}→{h1}→{K}→{K} + FC"), N("vl", "VAE 손실", 1350, 208, 120, 40, "MSE + KLD"),
             N("comb", "합치기", 1500, 82, 150, 112, "α·주기 + (1−α)·추세")]
    straight = [("data", "scale"), ("scale", "win"), ("win", "split"), ("split", "batch"), ("loss", "back"), ("back", "epoch"), ("epoch", "pred"),
                ("fwd", "merge"), ("bwd", "merge"), ("merge", "dec"), ("dec", "tc"), ("tc", "tl"), ("tl", "comb"), ("dec", "e1"), ("e1", "e2"), ("e2", "e3"),
                ("e3", "e4"), ("e4", "mu"), ("mu", "z"), ("z", "sc"), ("sc", "sl"), ("sl", "comb"), ("decd", "vl")]
    E = [{"from": a, "to": b} for a, b in straight]
    E += [{"from": "batch", "to": "fwd", "path": [[748, 50], [748, 57], [18, 57], [18, 126], [44, 126]]},
          {"from": "batch", "to": "bwd", "path": [[748, 50], [748, 57], [18, 57], [18, 208], [44, 208]]},
          {"from": "e4", "to": "sc", "path": [[854, 150], [854, 137], [1212, 137], [1212, 150]]},
          {"from": "z", "to": "decd", "path": [[1099, 194], [1099, 228], [1172, 228]]},
          {"from": "comb", "to": "loss", "path": [[1575, 82], [1575, 58], [1195, 58], [1195, 50]]},
          {"from": "vl", "to": "loss", "path": [[1470, 228], [1670, 228], [1670, 58], [1215, 58], [1215, 50]]}]
    return {"width": 1880, "height": 262, "nodes": nodes, "edges": E,
            "groups": [{"x": 6, "y": 64, "w": 1680, "h": 194, "label": "BIVA.Model.forward (models/biva_src/BIVA.py)"},
                       {"x": 26, "y": 84, "w": 318, "h": 168, "label": "BRITS (BIVA_BRITS.py · BIVA_RITS.py)"}]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="docs/guide_build/walk/biva_capture.pkl")
    ap.add_argument("--out", default="docs/guide_build/walk/biva.html")
    a = ap.parse_args()
    cap = pickle.load(open(ROOT / a.capture, "rb"))
    sc = build(cap)
    out = build_html(sc, ROOT / a.out)
    print("저장:", out, "· 단계", len(sc["steps"]), "· 검증", sum(c["ok"] for c in sc["checks"]), "/", len(sc["checks"]), "· 크기", round(out.stat().st_size / 1e6, 2), "MB")
