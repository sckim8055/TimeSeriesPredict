"""규칙 모델 + 백테스트 코드 따라가기 대본 — rule_capture 의 실제 실행 기록으로 단계별 화면을 만든다.

  python -m tools.codewalk.rule_script

규칙 신호(MA 교차) → 평가 구간 맞춤 → 다음 날 시가 체결·비용 → 자산곡선 → 성적표 → 부트스트랩 p · Deflated Sharpe → 다른 규칙(B&H·모멘텀·RSI·SOX 급등 회피).
매매 모드의 모든 모델(LightGBM·LSTM·BIVA 포함)이 같은 백테스트·성적표·검정을 쓴다. 모든 숫자를 식대로 numpy 로 다시 계산해 맞춰 본다.
say(음성 대본)·notes(코드 줄 설명)·lab(직접 실행해 보기)은 2026-11-08 추가 — 숫자는 모두 기록에서 계산한 변수로 넣는다.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tools.codewalk.common import ROOT, Sources, build_html, num, py_list, run_lab

EXP = "engine/experiment.py"; BT = "engine/backtest.py"


def pct(v, d=2, sign=True):
    return "NaN" if v is None or not np.isfinite(v) else (f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%")


# ---------------------------------------------------------------- 직접 실행해 보기(실험 코드) — 2026-11-08
# 실행 기록의 숫자를 그대로 넣은, 혼자 도는 파이썬 코드. 대본을 만들 때 실제로 실행해 출력이 엔진 값과 같은지 확인한다(common.run_lab).
# 기대 출력은 엔진이 만든 값(run_backtest 의 daily, 결과 파일의 성적표·p값, 엔진 positions)으로 만든다.
def _g5(v) -> str:
    return f"{float(v):.5g}"


def _rows(vals, per: int = 8, fmt=lambda v: repr(float(v))) -> str:
    """긴 숫자 목록을 한 줄에 per 개씩(실수는 값 그대로 — 17자리)."""
    vals = list(vals)
    return "[\n" + "".join("    " + ", ".join(fmt(v) for v in vals[i:i + per]) + ",\n" for i in range(0, len(vals), per)) + "]"


def lab_ma(closes, date, F, L, fast_e, slow_e, sig_e):
    assert np.all(np.asarray(closes) == np.round(closes)), "종가가 정수가 아님"
    code = (f"# 091160 종가(원) — 결정일 {date} 까지 마지막 {L}개 (실행 기록 그대로)\n"
            f"c = {_rows([int(v) for v in closes], 10, fmt=str)}\n\n"
            f"fast = sum(c[-{F}:]) / {F}    # 단기: 최근 {F}일 평균 (그날 종가 포함)\n"
            f"slow = sum(c[-{L}:]) / {L}    # 장기: 최근 {L}일 평균\n"
            "signal = 1 if fast > slow else 0    # 단기 > 장기 → 1(다음 날 보유), 아니면 0(현금)\n\n"
            f"print(\"{F}일 평균\", f\"{{fast:.2f}}\")\n"
            f"print(\"{L}일 평균\", f\"{{slow:.2f}}\")\n"
            "print(\"신호\", signal)\n")
    exp = f"{F}일 평균 {fast_e:.2f}\n{L}일 평균 {slow_e:.2f}\n신호 {int(sig_e)}"
    return {"title": f"결정일 {date} 의 이동평균 교차 신호를 직접 계산",
            "desc": f"마지막 {L}개 종가로 두 평균을 구해 비교합니다. 기대 출력의 평균은 엔진 줄(rolling mean)을 기록된 가격에 돌린 값, 신호는 엔진 positions 의 값입니다.",
            **run_lab(code, exp, pure=True)}


def lab_fill(dates, sigs, opens, prev, fee, slip, pos_e, hold_e, cost_e, net_e):
    code = ("# 결정일 7개(가운데 = 포지션이 바뀐 날)의 신호와 시가(원) — 실행 기록 그대로\n"
            f"dates = {dates!r}\n"
            f"signal = {sigs}    # 결정일 장 마감 후 정한 신호\n"
            f"opens = {opens}    # 첫 결정일의 시가부터, 마지막 결정일의 그다음 날 시가까지 {len(opens)}개\n"
            f"prev = {prev}    # 첫 행 바로 전 결정일의 포지션\n"
            f"one_way = {fee!r} + {slip!r} / 10000    # 편도 비용 = 수수료 + 슬리피지(bp → 비율)\n\n"
            "for i in range(len(dates)):\n"
            "    pos = signal[i]                    # 0 = 현금, 1 = 전부 보유 (이 구간엔 빈 신호가 없어 신호 = 포지션)\n"
            "    entry = opens[i + 1]               # 다음 날(t+1) 시가에 체결\n"
            "    exit_ = opens[i + 2]               # 그다음 날(t+2) 시가까지 보유\n"
            "    hold = exit_ / entry - 1           # 보유수익\n"
            "    cost = abs(pos - prev) * one_way   # 바꾼 양 × 편도 비용\n"
            "    net = pos * hold - cost + 0        # 순수익 (+ 0: −0.000 대신 +0.000 으로 보이게)\n"
            "    print(dates[i], pos, f\"보유수익 {hold * 100:+.3f}%\", f\"비용 {cost * 100:.3f}%\", f\"순수익 {net * 100:+.3f}%\")\n"
            "    prev = pos\n")
    exp = "\n".join(f"{d_} {int(p_)} 보유수익 {h_ * 100:+.3f}% 비용 {c_ * 100:.3f}% 순수익 {(n_ + 0.0) * 100:+.3f}%" for d_, p_, h_, c_, n_ in zip(dates, pos_e, hold_e, cost_e, net_e))
    return {"title": "체결·비용·순수익을 직접 계산 (7일)",
            "desc": "신호를 다음 날 시가에 체결하고 그다음 날 시가까지 들고 간다는 규칙, 포지션이 바뀐 날에만 내는 비용을 그대로 옮겼습니다. 출력이 run_backtest 의 기록(아래 기대 출력)과 같아야 합니다.",
            **run_lab(code, exp, pure=True)}


def lab_perf(r, perf_e, ppy):
    code = ("import math\n\n"
            f"# 평가 구간 {len(r)}일의 일간 순수익 r (run_backtest 의 net_return, 실행 기록 그대로)\n"
            f"r = {_rows(r)}\n\n"
            "n = len(r)\n"
            "eq = 1.0\npeak = 0.0\nmdd = 0.0\n"
            "for x in r:\n"
            "    eq *= 1 + x                      # 자산곡선: 누적 곱\n"
            "    if eq > peak:\n"
            "        peak = eq                    # 그때까지의 최고점\n"
            "    mdd = min(mdd, eq / peak - 1)    # 최고점 대비 낙폭 가운데 가장 깊은 값\n\n"
            "mean = sum(r) / n\n"
            "sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (n - 1))    # 표본 표준편차(n − 1 로 나눔)\n\n"
            "print(\"총수익\", f\"{eq - 1:.5g}\")\n"
            f"print(\"연복리\", f\"{{eq ** ({ppy} / n) - 1:.5g}}\")              # 년수 = n ÷ {ppy}\n"
            f"print(\"연 변동성\", f\"{{sd * math.sqrt({ppy}):.5g}}\")\n"
            f"print(\"샤프\", f\"{{mean / sd * math.sqrt({ppy}):.5g}}\")          # 무위험 수익률 0\n"
            "print(\"최대 낙폭\", f\"{mdd:.5g}\")\n"
            "print(\"승률\", f\"{sum(x > 0 for x in r) / sum(x != 0 for x in r):.5g}\")    # 손익이 0 인 날(현금)은 뺌\n")
    keys = [("총수익", "total_return"), ("연복리", "cagr"), ("연 변동성", "ann_vol"), ("샤프", "sharpe"), ("최대 낙폭", "mdd"), ("승률", "win_rate")]
    exp = "\n".join(f"{lab_} {_g5(perf_e[k_])}" for lab_, k_ in keys)
    return {"title": "성적표 6개 지표를 순수익에서 직접 계산",
            "desc": f"순수익 {len(r)}개만으로 반복문을 돌려 총수익·연복리·변동성·샤프·최대 낙폭·승률을 구합니다. 기대 출력은 결과 파일의 성적표입니다. (반복이 많아 Python Tutor 대신 복사해서 실행하세요.)",
            **run_lab(code, exp, pure=True, tutor=False)}


def lab_boot(d, blk, n_boot, seed, obs_e, p_e):
    code = ("import numpy as np\n\n"
            f"# 평가 구간 {len(d)}일의 날마다 초과수익 d = 전략 순수익 − B&H 보유수익 (실행 기록 그대로)\n"
            f"d = np.array({_rows(d)})\n\n"
            f"block = {blk}            # 엔진(_run_experiment)은 block = max(5, 2 × k) 로 부른다 — 이 실행은 k = 1\n"
            f"n_boot = {n_boot}\n"
            f"rng = np.random.default_rng({seed})    # 시드 {seed} — 엔진과 같은 난수\n"
            "obs = d.mean()                        # 실제 평균 초과수익\n"
            "centered = d - obs                    # 평균을 0 으로 옮김 = '효과 없음' 세계\n"
            "nb = int(np.ceil(len(d) / block))     # 이어 붙일 블록 수\n"
            "means = np.empty(n_boot)\n"
            "for i in range(n_boot):\n"
            "    starts = rng.integers(0, len(d) - block + 1, size=nb)\n"
            "    sample = np.concatenate([centered[s:s + block] for s in starts])[:len(d)]\n"
            "    means[i] = sample.mean()\n\n"
            "print(\"실제 평균\", f\"{obs:.5g}\")\n"
            "print(\"p\", f\"{(means >= obs).mean():.4f}\")\n")
    exp = f"실제 평균 {_g5(obs_e)}\np {p_e:.4f}"
    return {"title": "블록 부트스트랩 p 를 직접 계산 (numpy)",
            "desc": "엔진 bootstrap_excess_pvalue 와 같은 순서로 같은 시드의 난수를 씁니다. 기대 출력은 결과 파일의 excess_mean_per_period 와 bootstrap_p_excess 입니다. numpy 가 필요해 Colab 등에 붙여 실행하세요.",
            **run_lab(code, exp, pure=False)}


def lab_dsr(sharpe, n, N, V, sk, ku, ppy, sr0_s, z_s, p_e):
    code = ("import math\n\n"
            "def Phi(x):                       # 표준정규 누적분포 Φ\n"
            "    return 0.5 * (1 + math.erf(x / math.sqrt(2)))\n\n"
            "def Phi_inv(p):                   # Φ⁻¹: 뉴턴법으로 Φ(x) = p 인 x 를 찾음\n"
            "    x = 0.0\n"
            "    for _ in range(50):\n"
            "        step = (Phi(x) - p) / (math.exp(-x * x / 2) / math.sqrt(2 * math.pi))\n"
            "        x -= step\n"
            "        if abs(step) < 1e-12:\n"
            "            break\n"
            "    return x\n\n"
            "# 결과 파일에 기록된 값 그대로\n"
            f"sharpe_annual = {float(sharpe)!r}    # 연 샤프\n"
            f"n = {int(n)}                          # 평가 일수\n"
            f"N = {int(N)}                            # 매매 모드 실험 수(n_trials_so_far)\n"
            f"V = {float(V)!r}       # 시도들의 기간 샤프 분산(trial_sr_var)\n"
            f"skew = {float(sk)!r}\n"
            f"kurt = {float(ku)!r}             # 첨도(정규분포 = 3)\n"
            "gamma = 0.5772156649015329            # 오일러 상수\n\n"
            f"sr = sharpe_annual / math.sqrt({ppy})    # 연 샤프 → 일간 샤프\n"
            "sr0 = math.sqrt(V) * ((1 - gamma) * Phi_inv(1 - 1 / N) + gamma * Phi_inv(1 - 1 / (N * math.e)))    # 운으로도 기대되는 최고 샤프\n"
            "z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)\n"
            "p = 1 - Phi(z)\n\n"
            "print(\"SR0\", f\"{sr0:.5g}\")\n"
            "print(\"z\", f\"{z:.5g}\")\n"
            "print(\"p\", f\"{p:.5g}\")\n")
    exp = f"SR0 {_g5(sr0_s)}\nz {_g5(z_s)}\np {_g5(p_e)}"
    return {"title": "Deflated Sharpe p 를 직접 계산",
            "desc": "scipy 없이 math.erf 와 뉴턴법으로 Φ·Φ⁻¹ 를 만들어 같은 식을 계산합니다. 기대 출력의 p 는 결과 파일의 deflated_sharpe_p, SR0·z 는 화면의 값(엔진과 같은 식을 scipy 로 계산해 p 가 결과 파일과 같음을 확인한 값)입니다.",
            **run_lab(code, exp, pure=True)}


def _trading_results() -> list[tuple[str, dict]]:
    """results/trading 의 매매 결과 json 전부(카운터·봉인 기록·ablation 제외). 폴더가 없거나 읽을 수 없는 파일이 있으면 멈춘다 —
    대본이 '결과 파일이 있다/없다'·V 의 표본 수를 말하므로, 빌드하는 컴퓨터에 따라 대본이 조용히 바뀌면 안 된다(녹음은 대본 해시에 묶임)."""
    import json as _json
    d = ROOT / "results" / "trading"
    if not d.is_dir():
        raise FileNotFoundError(f"{d} 가 없습니다 — 대본이 결과 파일 목록에 따라 달라지므로 이 폴더 없이 빌드할 수 없습니다")
    out = []
    for p_ in sorted(d.glob("*.json")):
        if p_.name in ("experiment_counter.json", "sealed_log.json") or p_.name.startswith("ablation_"):
            continue
        out.append((p_.name, _json.loads(p_.read_text(encoding="utf-8"))))     # 읽지 못하면 예외 그대로(빌드 실패)
    return out


def _v_sample(cap: dict, V: float) -> dict:
    """Deflated Sharpe 의 V(trial_sr_var)가 어떤 결과들로 계산됐는지 다시 만든다(engine/experiment.py trial_sharpe_variance 의 규칙대로).
    '그때 저장돼 있던 결과' 는 파일 시각이 아니라 내용으로 고른다: 실험 카운터 값(n_trials_so_far)이 이 실행보다 작은 결과 — 파일 시각은 복사·내려받기로 바뀌어
    컴퓨터마다 다를 수 있기 때문. 다시 계산한 분산이 기록된 V 와 다르면 멈춘다(대본이 이 표본 수를 말함)."""
    fq = cap["config"]["frequency"]; N = int(cap["result"]["performance"]["n_trials_so_far"]); sf = cap["sealed_from"]
    ppy_ = {"D": 252, "W": 52, "M": 12}
    before = [(nm, r_) for nm, r_ in _trading_results() if int(r_["performance"]["n_trials_so_far"]) < N and not r_.get("sealed_evaluation")]
    rows = [(nm, r_) for nm, r_ in before if r_["config"].get("frequency") == fq and r_["performance"].get("sharpe") is not None and np.isfinite(r_["performance"]["sharpe"])]
    sr = [r_["performance"]["sharpe"] / np.sqrt(ppy_[fq]) for _, r_ in rows]
    v_re = float(np.var(sr, ddof=1)) if len(sr) >= 2 else float("nan")
    if not (abs(v_re - V) <= 1e-12 * abs(V)):
        raise AssertionError(f"V 표본 재구성 실패: 다시 계산한 분산 {v_re} ≠ 기록 {V} (결과 파일 {len(rows)}개) — results/trading 이 그때와 달라졌으면 대본의 V 설명을 고칠 것")
    old = {nm for nm, r_ in rows if r_.get("data_version") != cap["data_version"]}
    noeval = {nm for nm, r_ in rows if not r_.get("evaluation_period")}
    old_seal = {nm for nm, r_ in rows if nm in old and r_.get("period") and pd.Timestamp(r_["period"][1]) >= pd.Timestamp(sf)}
    ev_me = cap["result"]["evaluation_period"]
    rest_other = {nm for nm, r_ in rows if nm not in old and r_.get("evaluation_period") != ev_me}
    return {"n": len(rows), "v": v_re, "n_old": len(old), "same_old_noeval": old == noeval, "n_old_seal": len(old_seal), "n_rest": len(rows) - len(old),
            "n_rest_other": len(rest_other), "n_before_all": len(before),
            "n_wm": sum(1 for _, r_ in before if r_["config"].get("frequency") in ("W", "M"))}          # 주간·월간: 설정의 frequency 로 직접 셈


def build(cap: dict) -> dict:
    src = Sources([EXP, BT, "models/ma_cross.py", "models/buy_and_hold.py", "models/momentum.py", "models/rsi_rule.py", "models/sox_gap_fade.py", "engine/features.py"])
    cfg = cap["config"]; prm = cap["params"]; P = cap["prices"]; D = cap["daily_full"]; run = cap["run"]; perf = run["performance"]; perf0 = cap["result"]["performance"]
    lo, hi = cap["eval"]; ppy = 252
    steps: list[dict] = []; checks_all: list[dict] = []

    # 검증 줄의 종류(2026-11-08 공통 규칙 1). 독립 = 엔진의 중간 배열·엔진 함수를 쓰지 않고 원자료(데이터 저장소 파일을 따로 읽음)·다른 결과 파일·다른 계산법으로
    # 다시 구해 맞춘 것(식은 엔진과 같은 정의를 따름) / 식 = 엔진이 쓴 것과 같은 배열에 같은 식을 다른 코드로 다시 계산했거나 날짜 수 합 같은 구조 조건 /
    # 재실행 = 같은 알고리즘·같은 시드로 다시 돌려 결과 파일과 맞춘 것.
    KIND = {"독립": "독립 대조", "식": "식·구조 확인", "재실행": "재실행·결과 파일 대조", "참고": "참고(결론 비교)"}   # 참고 = 값이 아니라 결론만 비교(메인 결정 2026-11-08)

    def chk(what, a, b_, tol=1e-12, rel=False, kind="식", show_diff=True):
        assert kind in KIND
        a = np.asarray(a, dtype=float); b_ = np.asarray(b_, dtype=float)
        nan_ok = bool(np.all(np.isnan(a) == np.isnan(b_)))
        d = float(np.nanmax(np.abs(a - b_) / (np.maximum(np.abs(b_), 1e-300) if rel else 1.0))) if a.size and np.any(~np.isnan(a)) else 0.0
        c = {"what": f"{what} ({'상대 ' if rel else ''}차이 {d:.1e})" if show_diff else what, "ok": bool(d <= tol and nan_ok), "kind": kind}
        checks_all.append({**c, "step": len(steps) + 1}); return c

    def S(chapter, narr, *, code=None, stack=None, nodes=(), edges=(), tensors=(), formula=None, hint=None, checks=(), say=None, notes=None, lab=None):
        """say = 음성·자막 대본(화면 숫자 그대로 — 숫자는 narr 와 같은 변수에서), notes = {원본 줄 번호: 그 줄 설명}, lab = 직접 실행해 보기(실행해 엔진 값과 맞춘 것)."""
        checks = list(checks)
        if lab is not None:
            kd = lab.get("kind", "식")
            c_ = {"what": f"직접 실행해 보기 '{lab['title']}' 출력 = {'결과 파일 값(같은 알고리즘·같은 시드)' if kd == '재실행' else '엔진 값'}", "ok": bool(lab["ok"]), "kind": kd}
            checks_all.append({**c_, "step": len(steps) + 1}); checks.append(c_)
        if notes:
            assert code is not None, "코드 없는 단계에 줄 설명"
            off = [k_ for k_ in notes if k_ not in code["lines"]]
            assert not off, f"{chapter}: 줄 설명이 강조 줄 밖 {off} (강조 줄 {code['lines']})"
            code = {**code, "notes": {str(k_): v_ for k_, v_ in notes.items()}}
        steps.append({"chapter": chapter, "narr": narr, "say": say, "code": code, "stack": list(stack or []), "nodes": list(nodes), "edges": [list(e) for e in edges],
                      "tensors": list(tensors), "formula": formula, "hint": hint, "checks": checks, "lab": lab})

    def C(fn, lines, focus=None, func="", before=6, after=8):
        lines = sorted(set(lines)); return {"file": fn, "lines": lines, "focus": focus or lines[0], "func": func, "before": before, "after": after}

    ALL = ["sig", "win", "fill", "eq", "perf", "test", "oth"]
    stk = ["engine/experiment.py · _run_experiment"]
    idx = P.index; n = len(idx)
    # 원자료: 엔진(DataVersion·prepare)을 거치지 않고 데이터 저장소의 가격 파일을 pandas 로 따로 읽는다 — 손 계산(신호·체결·성적표)은 이 값으로 한다(공통 규칙 1 의 '다른 경로').
    raw_f = ROOT / "data" / "versions" / cap["data_version"] / f"prices_{cfg['ticker']}.parquet"
    raw = pd.read_parquet(raw_f); raw = raw.loc[raw.index >= pd.Timestamp(cfg["start"])]
    if cfg.get("end"):
        raw = raw.loc[raw.index <= pd.Timestamp(cfg["end"])]
    assert len(raw) == n and (raw.index == idx).all(), "원자료 날짜 ≠ 실행 기록 날짜"
    c = raw["close"].astype(float).values; o = raw["open"].astype(float).values
    raw_rel = raw_f.relative_to(ROOT).as_posix()
    cP = P["close"].astype(float).values; oP = P["open"].astype(float).values      # 실행 기록(엔진 prepare 의 가격) — 직접 실행해 보기에 그대로 넣는 값
    fee, slip = cfg["trade"]["fee_rate"], cfg["trade"]["slippage_bp"]
    one = fee + slip / 1e4
    fee_p, slip_p, one_p = f"{fee * 100:.3f}%", f"{slip / 100:.2f}%", f"{one * 100:.3f}%"     # 0.015% · 0.05% · 0.065% (작은 비율은 % 로 읽기)

    def has_result(model_name):
        """results/trading 에 이 모델의 결과 파일이 있는지(설정의 model 로 찾음). 폴더·파일을 읽지 못하면 멈춘다(_trading_results)."""
        return [nm for nm, r_ in _trading_results() if r_.get("config", {}).get("model") == model_name]
    # ================================================================ 0
    S("0. 시작",
      f"이 영상은 <b>규칙 모델</b>(학습 없는 기준선)과, 매매 모드의 모든 모델이 함께 쓰는 <b>백테스트·성적표·검정</b>이 실제로 계산되는 순서를 따라갑니다. "
      f"예: <b>이동평균 교차</b> — {cfg['ticker']} 일간 종가의 {prm['fast']}일 평균이 {prm['slow']}일 평균보다 크면 다음 날 보유, 아니면 현금. "
      f"결과 파일 <b>{Path(cap['result_file']).stem}</b> 을 낸 설정·데이터 버전({cap['data_version']})·실험 수 그대로 다시 돌린 기록입니다. 과거 가격으로 한 모의 계산이며 투자 권유가 아닙니다.",
      say=("이 영상은 규칙 모델과 백테스트가 실제로 계산되는 순서를 따라갑니다. 규칙 모델은 학습 없이 정해 둔 규칙으로 포지션을 내는 기준선이고, "
           "백테스트와 성적표, 검정은 매매 모드의 모든 모델이 같은 함수로 계산합니다. "
           f"예로 드는 이동평균 교차는, 코덱스 반도체의 일간 종가로 {prm['fast']}거래일 평균이 {prm['slow']}거래일 평균보다 크면 다음 날 보유하고, 아니면 현금으로 둡니다. "
           "화면의 숫자는 결과 파일과 같은 설정, 데이터 버전, 실험 수로 다시 실행한 기록입니다. 과거 가격으로 한 모의 계산이며, 투자 권유가 아닙니다."),
      code=C("models/ma_cross.py", [1], 1, "모듈 전체", 0, 30), stack=["models/ma_cross.py"], nodes=ALL,
      tensors=[{"type": "table", "title": "설정", "columns": ["항목", "값"], "rows": [["대상", f"{cfg['ticker']} (KODEX 반도체) 일간"], ["모델", f"MA Cross (fast {prm['fast']}, slow {prm['slow']})"],
                                                                         ["포지션", "0 = 현금, 1 = 전부 보유 (공매도·레버리지 없음)"],
                                                                         ["비용", f"수수료 {fee_p} + 슬리피지 {slip:g}bp({slip_p}) = 편도 {one_p}"],
                                                                         ["평가 구간", f"{lo.date()} ~ {hi.date()} (학습 모델과 같은 구간)"], ["봉인 시작", cfg["validation"]["sealed_from"]],
                                                                         ["실험 수(n_trials)", perf["n_trials_so_far"]]]}],
      hint="숫자의 출처: tools/codewalk/rule_capture.py 가 엔진의 _run_experiment 를 그대로 실행하고, 같은 엔진 함수로 신호·평가 구간을 다시 불러 기록했습니다. 결과 파일의 성적표와 소수 끝자리까지 같습니다('5. 성적표' 단계의 검증).")
    S("0. 시작",
      "호출 순서: 규칙 모델은 학습이 없어 전체 기간의 포지션을 한 번에 냅니다(positions). 그다음 ① 학습 모델과 같은 평가 구간만 남기고(_eval_window) ② 봉인 구간을 지운 뒤 "
      "③ <b>run_backtest</b> 가 다음 날 시가에 체결해 자산곡선을 만들고 ④ <b>performance</b> 가 성적표를, ⑤ 부트스트랩·Deflated Sharpe 가 '우연인가' 를 계산합니다.",
      say=("호출 순서입니다. 규칙 모델이 학습 없이 전체 기간의 포지션을 한 번에 내면, 학습 모델과 같은 평가 구간만 남기고 봉인 구간의 신호를 지웁니다. "
           "그다음 백테스트, 성적표, 그리고 부트스트랩과 디플레이티드 샤프 검정 순서로 계산합니다."),
      code=C(EXP, src.span(EXP, 'if plugin.INPUT_SHAPE == "rule":', 'pred[:] = pos.values') + src.span(EXP, "daily_full, _ = run_backtest(ds.prices, signal, cost=cost, periods_per_year=ppy)", "perf.update(n_trades=int((turnover > 0).sum())"),
             src.line(EXP, "daily_full, _ = run_backtest(ds.prices, signal, cost=cost, periods_per_year=ppy)"), "_run_experiment", 1, 1),
      notes={src.line(EXP, "pos = pd.Series(model.positions(ds.prices, ds.X)"): "규칙 모델: 학습 없이 전체 기간의 포지션(0 또는 1)을 한 번에",
             src.line(EXP, "daily_full, _ = run_backtest(ds.prices, signal, cost=cost, periods_per_year=ppy)"): "다음 날 시가 체결·비용 → 날마다의 순수익",
             src.line(EXP, 'perf = performance(daily["net_return"], daily["hold_return"], periods_per_year=ppy)'): "첫 신호 ~ 마지막 신호 구간만으로 성적표"},
      stack=stk, nodes=["sig", "win", "fill", "perf"])
    # ================================================================ 1. 신호
    F, L = prm["fast"], prm["slow"]
    t = hi; ti = idx.get_loc(t)
    fa = np.full(n, np.nan); sl = np.full(n, np.nan)
    for i in range(n):
        if i >= F - 1: fa[i] = c[i - F + 1:i + 1].mean()
        if i >= L - 1: sl[i] = c[i - L + 1:i + 1].mean()
    pos_h = np.where(np.isnan(sl), np.nan, (fa > sl).astype(float))
    cs = P["close"].astype(float)
    fa_e, sl_e = float(cs.rolling(F).mean().iloc[ti]), float(cs.rolling(L).mean().iloc[ti])   # models/ma_cross.py 의 줄을 기록된 가격에 그대로
    sig_t = int(cap["pos_raw"].iloc[ti])
    S("1. 신호",
      f"결정일 t 장 마감 후, t 까지의 종가로 단기·장기 단순이동평균을 계산합니다. 예: {t.date()} — {F}일 평균 {fa[ti]:,.2f}, {L}일 평균 {sl[ti]:,.2f} → "
      f"{'단기 > 장기 → 보유(1)' if fa[ti] > sl[ti] else '단기 ≤ 장기 → 현금(0)'}. 장기 평균이 차기 전({L - 1}일)은 결정할 수 없어 빈칸입니다. 아래 검증은 엔진을 거치지 않고 데이터 저장소의 원자료({raw_rel})를 따로 읽어 전체 {n}일을 반복문으로 다시 계산한 것입니다.",
      say=(f"신호입니다. 결정일 장이 끝난 뒤, 그날 종가까지 포함한 단순이동평균을 계산합니다. {F}거래일 평균이 {L}거래일 평균보다 크면 신호는 1, 곧 다음 날 보유이고, "
           "같거나 작으면 0, 곧 현금입니다. "
           f"평가 구간의 마지막 날인 {t.date()}에는 {F}거래일 평균이 {fa[ti]:,.2f}, {L}거래일 평균이 {sl[ti]:,.2f}이라서 신호는 {sig_t}입니다. "
           f"처음 {L - 1}거래일은 {L}거래일 평균이 없어 신호가 비어 있습니다. 검증은 데이터 저장소의 원자료를 따로 읽어, 전체 {n}거래일을 반복문으로 다시 계산해 엔진과 맞춘 것입니다."),
      code=C("models/ma_cross.py", src.span("models/ma_cross.py", "def positions(self, prices, X=None):", "return pos.values"), src.line("models/ma_cross.py", "pos = (fast > slow).astype(float)"), "Model.positions", 2, 2),
      notes={src.line("models/ma_cross.py", 'fast = c.rolling(self.params["fast"]).mean()'): f"최근 {F}일 종가 평균(그날 포함). {t.date()}: {fa[ti]:,.2f}",
             src.line("models/ma_cross.py", "pos = (fast > slow).astype(float)"): "단기 > 장기 → 1(보유), 같거나 작으면 0(현금)",
             src.line("models/ma_cross.py", "pos[slow.isna()] = np.nan"): f"{L}일 평균이 아직 없는 처음 {L - 1}일은 결정 불가(빈칸)"},
      lab={**lab_ma(cP[ti - L + 1:ti + 1], str(t.date()), F, L, fa_e, sl_e, sig_t), "kind": "식"},
      stack=["models/ma_cross.py · Model.positions"], nodes=["sig"],
      formula=f"신호_t = 1  if  평균(종가_(t−{F - 1}..t)) > 평균(종가_(t−{L - 1}..t))  else 0",
      tensors=[{"type": "line", "title": f"마지막 250일: 종가와 이동평균 (아래 표: 포지션)", "x": [d.strftime("%Y-%m-%d") for d in idx[ti - 249:ti + 1]], "xlabel": "날짜",
                "series": [{"name": "종가", "color": "#8a93a3", "y": [num(v) for v in c[ti - 249:ti + 1]]}, {"name": f"{F}일", "color": "#c8710f", "y": [num(v) for v in fa[ti - 249:ti + 1]]},
                           {"name": f"{L}일", "color": "#3b6fb6", "y": [num(v) for v in sl[ti - 249:ti + 1]]}], "height": 250},
               {"type": "table", "title": "결정일까지 마지막 5일", "columns": ["날짜", "종가", f"{F}일 평균", f"{L}일 평균", "신호"],
                "rows": [[idx[i].strftime("%Y-%m-%d"), f"{c[i]:,.0f}", f"{fa[i]:,.2f}", f"{sl[i]:,.2f}", int(pos_h[i])] for i in range(ti - 4, ti + 1)]}],
      checks=[chk(f"원자료 {raw_rel} 의 시가·종가 {n}행 = 실행 기록 가격(엔진 prepare)", np.r_[o, c], np.r_[oP, cP], 0, kind="독립"),
              chk(f"전체 {n}일 신호: 원자료 종가로 반복문 손 계산 = 엔진 positions", pos_h, cap["pos_raw"].values, 0, kind="독립")])
    # ================================================================ 2. 평가 구간
    sig = D["signal"]; first, last = sig.first_valid_index(), sig.last_valid_index()
    sf = pd.Timestamp(cap["sealed_from"])
    n_warm = int(np.isnan(pos_h).sum()); n_pre = int(((idx < lo) & ~np.isnan(pos_h)).sum()); n_eval = int(((idx >= lo) & (idx <= hi)).sum()); n_seal = int((idx >= sf).sum())
    vcfg = cfg["validation"]; tl, tlen, k_ = vcfg["train_len"], vcfg["test_len"], cfg["target"]["k"]
    assert vcfg["embargo"] is None and cfg["mode"] == "trading" and vcfg["mode"] == "walkforward" and not vcfg.get("test_start") and not vcfg.get("test_end")
    emb = 1 + k_      # 비우는 행 수: 매매 모드 정답이 t+1+k 의 시가까지 쓰므로 1 + k (experiment.py _embargo 의 기본값 — 엔진 함수는 부르지 않고 정의대로)
    KN = {1: "한", 2: "두", 3: "세", 4: "네"}; assert emb in KN
    # 폴드를 엔진 함수 없이 반복문으로(calendar.walkforward_folds 의 정의대로): 테스트는 tl 행부터 tlen 행씩, 봉인 시작 전 날짜까지. 학습 창 = 그 앞 tl 행 − 끝 emb 행
    n_dev = int((idx < sf).sum()); folds_h = []; i_ = tl
    while i_ < n_dev:
        folds_h.append((list(range(i_ - tl, i_ - emb)), list(range(i_, min(i_ + tlen, n_dev))))); i_ += tlen
    tr0, te0 = folds_h[0]; trL, teL = folds_h[-1]
    n_fit0 = len(tr0)                                   # 비운 뒤 첫 폴드 학습 창 행 수(학습 표본 수가 아님)
    lo_h, hi_h = idx[te0[0]], idx[teL[-1]]              # 반복문으로 구한 평가 구간
    # 같은 검증 설정·같은 데이터의 학습 모델 결과 파일(periods.first_fold·last_fold)로 '실제 학습 행' 과 폴드를 대조
    import json as _json
    import re as _re

    def _res(pattern):
        fs = sorted((ROOT / "results" / "trading").glob(pattern)); assert len(fs) == 1, pattern
        return fs[0].stem, _json.loads(fs[0].read_text(encoding="utf-8"))
    me_res = _json.loads((ROOT / cap["result_file"]).read_text(encoding="utf-8"))
    rid_n, rid = _res("04_ridge_A_v1107_*.json"); lgb_n, lgb = _res("05_lgbm_A_v1107_*.json")
    for r_ in (rid, lgb):
        rc = r_["config"]
        assert (r_["data_version"] == cap["data_version"] and rc["validation"] == vcfg and rc["target"]["k"] == k_ and rc["ticker"] == cfg["ticker"]
                and rc["frequency"] == cfg["frequency"] and rc["start"] == cfg["start"] and rc.get("end") == cfg.get("end")), "학습 모델 결과 파일의 설정이 다름"
    assert lgb["feature_names"] == rid["feature_names"]
    rf, rl, lf, ll = rid["periods"]["first_fold"], rid["periods"]["last_fold"], lgb["periods"]["first_fold"], lgb["periods"]["last_fold"]
    T = lambda d_: pd.Timestamp(d_).value
    warm_r = int(idx.get_loc(pd.Timestamp(rf["train"][0])))                       # Ridge A 첫 폴드 학습 첫날의 행 번호 = 앞에서 빠진 행 수
    n_contig = int(((idx >= pd.Timestamp(rf["train"][0])) & (idx <= pd.Timestamp(rf["train"][1]))).sum())   # 학습 첫날~끝날의 거래일 수(중간에 빠진 행이 없는지)
    ma_wins = [int(m_.group(1)) for m_ in (_re.match(r"A_ma_ratio_(\d+)$", f_) for f_ in rid["feature_names"]) if m_]
    win_max = max(ma_wins)                                                          # 가장 긴 이동평균 창(A_ma_ratio_N)
    vf = vcfg["val_frac"]
    n_r, n_lt, n_lv = rf["n_train"], lf["n_train"], lf["n_val"]
    fit_txt = f"예: Ridge A {n_r}행, LightGBM A 학습 {n_lt} + 검증 {n_lv}행 — 결과 파일 periods.first_fold"
    # 평가 구간 밖을 지우는 줄·봉인 이후를 지우는 줄이 각각 지우는 날 수 — 엔진 positions 에서 센다
    l_win = src.line(EXP, "pos = pos.where((pos.index >= lo) & (pos.index <= hi))"); l_seal = src.line(EXP, "signal = signal.where(signal.index < sealed_from)")
    pr_ = cap["pos_raw"]; in_w = (idx >= lo) & (idx <= hi)
    n_front = int(((idx < lo) & pr_.notna().values).sum()); n_back = int(((idx > hi) & pr_.notna().values).sum())
    n_731 = int(((idx >= sf) & pr_.where(in_w).notna().values).sum())
    # 보유수익이 봉인 구간의 시가를 쓰는 평가 구간 결정일(청산 시가 t+2 가 봉인 시작 이후)
    bnd = [j for j in range(idx.get_loc(lo), idx.get_loc(hi) + 1) if idx[j + 2] >= sf]
    bnd_txt = "; ".join(f"{idx[j].date()} → 체결 {idx[j + 1].date()}·청산 {idx[j + 2].date()} 시가, 전략 포지션 {int(D['position'].iloc[j])}, B&H 보유수익 {pct(D['hold_return'].iloc[j])}" for j in bnd)
    S("2. 평가 구간",
      f"규칙 모델은 첫날부터 신호를 낼 수 있지만, 학습 모델과 <b>같은 날짜로 성적을 매겨야</b> 비교가 공정합니다. 같은 검증 설정(학습 창 {tl}행 · 테스트 {tlen}행)이면 "
      f"학습 모델이 평가받는 구간은 첫 폴드 테스트 시작 {lo.date()} ~ 마지막 폴드 테스트 끝 {hi.date()} 입니다. 첫 폴드는 학습 창 {tl}행 중 끝 {emb}행을 비우고 그 앞 {n_fit0}행 안에서 학습합니다 — "
      f"실제 학습 행은 변수 워밍업 결측·검증 분할을 빼 더 적습니다({fit_txt}; 마지막 폴드는 결측이 없어 Ridge A {rl['n_train']}행, LightGBM A 학습 {ll['n_train']} + 검증 {ll['n_val']}행 — periods.last_fold). "
      f"그 밖과 봉인 구간({sf.date()} 이후)의 신호는 지웁니다.",
      say=("규칙 모델은 첫날부터 신호를 낼 수 있지만, 학습 모델과 공정하게 비교하려면 같은 날짜로 성적을 매겨야 합니다. "
           f"학습 모델의 첫 폴드는 처음 {tl}거래일을 학습 창으로 쓰되, 끝의 {KN[emb]} 거래일을 비운 뒤, 그다음 날부터 예측합니다. "
           "검증을 쓰는 모델은 검증 몫만큼, 그리고 특히 앞쪽 폴드에서는 변수의 결측만큼, 실제로 학습에 쓰는 행이 학습 창보다 적습니다. "
           f"규칙 모델의 평가 구간은 첫 폴드의 테스트 시작일 {lo.date()}부터 마지막 폴드의 테스트 끝 {hi.date()}까지로 맞춥니다. "
           f"그 앞의 신호 {n_pre}거래일과, {sf.date()}부터 시작하는 봉인 구간 {n_seal}거래일의 신호는 지웁니다. 성적을 매기는 날은 {n_eval}거래일이 남습니다."),
      code=C(EXP, src.span(EXP, "def _eval_window(cfg, ds, k):", "return (folds[0].test[0], folds[-1].test[-1]) if folds else (None, None)") + src.lines(EXP, "lo, hi = _eval_window(cfg, ds, k)", "pos = pos.where((pos.index >= lo) & (pos.index <= hi))", "signal = signal.where(signal.index < sealed_from)"),
             src.line(EXP, "pos = pos.where((pos.index >= lo) & (pos.index <= hi))"), "_eval_window", 1, 1),
      notes={src.line(EXP, "return (folds[0].test[0], folds[-1].test[-1]) if folds else (None, None)"): f"첫 폴드 테스트 첫날 {lo.date()} ~ 마지막 폴드 테스트 마지막 날 {hi.date()}",
             src.line(EXP, "pos = pos.where((pos.index >= lo) & (pos.index <= hi))"): f"평가 구간 밖 신호는 NaN — 앞 {n_front}일 + 뒤(봉인) {n_back}일",
             src.line(EXP, "signal = signal.where(signal.index < sealed_from)"): f"봉인 이후를 한 번 더 지움(규칙 모델은 {l_win}행에서 이미 지워져 추가 {n_731}일)"},
      stack=stk + ["engine/experiment.py · _eval_window"], nodes=["win"], edges=[("sig", "win")],
      hint=(f"첫 테스트 행 = 학습 창 길이({tl}행): 처음 {tl}일(이동평균 워밍업 {n_warm} + 평가 전 {n_pre})은 첫 폴드의 학습 창이고, 그 끝 {emb}행은 정답이 테스트 구간 가격과 겹쳐 비웁니다"
            f"(비운 뒤 창 {n_fit0}행 — 학습 표본 수가 아님). 실제 학습 행: 학습 표에서 결측 행을 빼면(experiment.py 의 .dropna()) Ridge A 는 앞 {warm_r}행이 빠져 {n_r}행입니다 — "
            f"가장 긴 변수 창({win_max}일 이동평균 비율)의 워밍업. LightGBM A 는 그 {n_r}행에서 뒤 {vf:.0%}({n_lv}행)를 검증으로 떼고 그 앞 {emb}행을 비워 {n_lt}행으로 학습합니다. "
            f"폴드는 {tl}행부터 {tlen}행씩 봉인 전날까지 {len(folds_h)}개이고, 마지막 폴드는 남은 {len(teL)}일입니다. "
            f"경계 주의: 평가 구간 끝의 결정일 {len(bnd)}개는 보유수익 계산에 봉인 구간({sf.date()}~)의 시가를 씁니다(엔진은 결정일만 봉인 전으로 자릅니다) — {bnd_txt}. "
            + ("이 날들의 전략 순수익에는 영향이 없지만(포지션 0), B&H 보유수익은 비교 기준·초과수익에 들어갑니다."
               if all(int(D["position"].iloc[j]) == 0 for j in bnd) else "이 날들의 보유수익은 전략 순수익과 비교 기준·초과수익에 모두 들어갑니다.")),
      tensors=[{"type": "segs", "title": "전체 날짜의 쓰임", "segs": [{"label": "워밍업", "n": n_warm, "color": "#c3c8d0"}, {"label": "평가 전", "n": n_pre, "color": "#e3e6ea"},
                                                               {"label": "평가", "n": n_eval, "color": "#2f8f5b", "sub": f"{lo.date()} ~ {hi.date()}"}, {"label": "봉인", "n": n_seal, "color": "#b3261e"}]}],
      checks=[chk("남은 신호 = 원자료 손 계산 신호를 반복문 평가 구간으로 자른 것", sig.values, np.where((idx >= lo_h) & (idx <= hi_h), pos_h, np.nan), 0, kind="독립"),
              chk("첫·마지막 신호일 = 평가 구간 끝점", [first.value, last.value], [lo.value, hi.value], 0),
              chk(f"처음 {tl}일 = 워밍업 + 평가 전, 날짜 수 합 = 전체", [n_warm + n_pre, n_warm + n_pre + n_eval + n_seal], [tl, n], 0),
              chk(f"{l_win}행이 지우는 날 = 앞 {n_pre} + 뒤 {n_seal}, {l_seal}행이 규칙 모델에서 더 지우는 날 = 0", [n_front, n_back, n_731], [n_pre, n_seal, 0], 0),
              chk(f"폴드 반복문(엔진 함수 없이 {tl}행부터 {tlen}행씩, 봉인 전까지): 폴드 {len(folds_h)}개·첫 폴드 테스트 {len(te0)}일·마지막 폴드 테스트 {len(teL)}일과 그 날짜 = "
                  f"Ridge A 결과 파일({rid_n}) n_folds·periods, 평가 구간 = 이 결과 파일 evaluation_period·엔진 _eval_window",
                  [len(folds_h), T(idx[te0[0]]), T(idx[te0[-1]]), len(te0), T(idx[teL[0]]), T(idx[teL[-1]]), len(teL), lo_h.value, hi_h.value, lo_h.value, hi_h.value],
                  [rid["n_folds"], T(rf["test"][0]), T(rf["test"][1]), rf["n_test"], T(rl["test"][0]), T(rl["test"][1]), rl["n_test"],
                   T(me_res["evaluation_period"][0]), T(me_res["evaluation_period"][1]), lo.value, hi.value], 0, kind="독립"),
              chk(f"반복문 첫 폴드 학습 창(비운 뒤)의 끝날 = Ridge A 첫 폴드 학습 끝날, 비우는 행 {emb} = 결과 파일 periods.embargo, "
                  f"반복문 마지막 폴드 창 {len(trL)}행 = Ridge A 마지막 폴드 학습 행 = LightGBM A 마지막 폴드 학습 + 검증 + 비움",
                  [T(idx[tr0[-1]]), emb, emb, len(trL), len(trL)], [T(rf["train"][1]), me_res["periods"]["embargo"], rid["periods"]["embargo"], rl["n_train"], ll["n_train"] + ll["n_val"] + emb], 0, kind="독립"),
              chk(f"실제 학습 행(결과 파일): Ridge A {n_r} = {n_fit0} − 앞 {warm_r}행(이동평균 창 {win_max} − 1, 중간 빠짐 없음), "
                  f"LightGBM A 학습 {n_lt} + 검증 {n_lv} + 비움 {emb} = {n_r}, 검증 {n_lv} = ⌊{n_r} × {vf:g}⌋",
                  [n_fit0 - warm_r, n_contig, n_lt + n_lv + emb, int(n_r * vf), warm_r, T(lf["train"][0]), T(lf["val"][1])],
                  [n_r, n_r, n_r, n_lv, win_max - 1, T(rf["train"][0]), T(rf["train"][1])], 0, kind="독립")])
    # ================================================================ 3. 체결·비용
    # 손 계산 경로: 원자료 종가로 만든 신호(pos_h)를 반복문 평가 구간(lo_h~hi_h, 봉인 전)으로 자르고, 원자료 시가로 체결 — 엔진의 signal·daily 배열은 쓰지 않는다
    sg = np.where((idx >= lo_h) & (idx <= hi_h) & (idx < sf), pos_h, np.nan)
    posx = np.empty(n); prev_ = 0.0
    for i in range(n):                                  # 빈 신호는 직전 포지션, 첫 신호 전은 0, 0~1 로 자름
        prev_ = prev_ if np.isnan(sg[i]) else min(max(float(sg[i]), 0.0), 1.0); posx[i] = prev_
    entry = np.r_[o[1:], np.nan]; exit_ = np.r_[o[2:], np.nan, np.nan]; hr = exit_ / entry - 1
    to = np.abs(np.diff(np.r_[0.0, posx])); cost = to * one; net = posx * hr - cost
    assert not np.isnan(sg[(idx >= lo_h) & (idx <= hi_h)]).any()   # 평가 구간 안에는 빈 신호가 없음(화면 문장)
    ch = int(np.flatnonzero((to > 0) & (idx >= lo))[3]); w_ = 3
    rr_ = range(ch - w_, ch + w_ + 1)
    rows = [[idx[i].strftime("%Y-%m-%d"), f"{o[i]:,.0f}", "" if np.isnan(sg[i]) else int(sg[i]), int(posx[i]), f"{entry[i]:,.0f}", f"{exit_[i]:,.0f}", pct(hr[i], 3), int(to[i]),
             pct(cost[i] + 0.0, 3, sign=False), pct(net[i] + 0.0, 3)] for i in rr_]
    sgE = sig.values                                    # 직접 실행해 보기에는 실행 기록(엔진 signal)을 그대로 넣는다
    assert all(np.isfinite(sgE[i]) for i in rr_) and all(oP[i] == round(oP[i]) for i in range(ch - w_, ch + w_ + 3))
    lab_fill_ = {**lab_fill([idx[i].strftime("%Y-%m-%d") for i in rr_], [int(sgE[i]) for i in rr_], [int(oP[i]) for i in range(ch - w_, ch + w_ + 3)], int(D["position"].values[ch - w_ - 1]), fee, slip,
                            D["position"].values[ch - w_:ch + w_ + 1], D["hold_return"].values[ch - w_:ch + w_ + 1], D["cost"].values[ch - w_:ch + w_ + 1], D["net_return"].values[ch - w_:ch + w_ + 1]), "kind": "식"}

    def spct(v, d=3):
        """대본용 퍼센트: 양수는 '플러스'를 붙여 쓴다(읽기 규칙이 + 기호를 읽지 않고, '이 점 …' 이 지시어로 들리지 않게)."""
        return ("플러스 " if v > 0 else "") + pct(v, d, sign=v < 0)
    last_pos = int(D.loc[hi, "position"])
    S("3. 체결과 비용",
      f"이 영상의 일간 실행에서는 결정일 t 의 신호를 <b>다음 날(t+1) 시가에 체결</b>하고, 그 포지션을 그다음 날(t+2) 시가까지 들고 갑니다(오늘 종가로는 이미 살 수 없음). "
      "주간·월간 실행이면 같은 규칙이 '다음 주기의 첫 시가'가 됩니다(백테스트는 한 행을 한 주기로 읽음). "
      f"포지션은 0(현금) 또는 1(전부 보유)이고, 신호가 빈 날은 전날 포지션을 그대로 씁니다(평가 구간 안에는 빈 신호가 없음). "
      f"포지션이 바뀐 날에만 편도 비용 {one_p}(= 수수료 {fee_p} + 슬리피지 {slip:g}bp = {slip_p})를 냅니다. 아래는 포지션이 바뀐 날(가운데 행)과 그 앞뒤 {w_}일씩, 모두 {2 * w_ + 1}일입니다.",
      say=("체결입니다. 결정일 종가를 본 뒤에는 그날 종가에 살 수 없으므로, 이 영상의 일간 실행에서는 다음 날 시가에 체결하고 그다음 날 시가까지 들고 갑니다. "
           "주간이나 월간 실행이면 다음 주기의 시가입니다. 포지션은 0 아니면 1, 곧 전부 현금이거나 전부 보유입니다. "
           f"비용은 포지션이 바뀐 날에만 내고, 수수료 {fee_p}에 슬리피지 {slip_p}를 더해 편도 {one_p}입니다. "
           f"가운데 행인 {idx[ch].date()}에는 신호가 {int(sg[ch])}으로 바뀌어, 다음 거래일 시가 {entry[ch]:,.0f}원에 "
           + ("팝니다. 그래서 이날 순수익은 비용만큼 손실인 " + pct(net[ch] + 0.0, 3) + "입니다. 들고 있었다면 얻었을 보유수익 " + spct(hr[ch])
              + "는 들어가지 않습니다."
              if posx[ch] == 0 else "삽니다. 그래서 이날 순수익은 보유수익 " + spct(hr[ch]) + "에서 비용을 뺀 " + pct(net[ch] + 0.0, 3) + "입니다.")),
      code=C(BT, src.span(BT, "    pos = sig.ffill().fillna(0.0).clip(0.0, 1.0)  # 1단계: 롱/현금", "    bench = initial_capital * (1.0 + hold_ret.fillna(0.0)).cumprod()"),
             src.line(BT, "hold_ret = exit_ / entry - 1.0"), "run_backtest", 2, 2),
      notes={src.line(BT, "pos = sig.ffill().fillna(0.0).clip(0.0, 1.0)"): "빈 신호는 전날 포지션 그대로, 첫 신호 전은 0, 0~1 로 자름 → 현금(0) 또는 전부 보유(1)",
             src.line(BT, "hold_ret = exit_ / entry - 1.0"): "결정일 t 행에 기록: 시가_(t+2) ÷ 시가_(t+1) − 1 (일간 기준; 주간·월간이면 한 행 = 한 주기)",
             src.line(BT, "costs = turnover * cost.one_way"): f"바꾼 양 × 편도 비용 {one_p} (바꾸지 않은 날은 0)"},
      lab=lab_fill_,
      stack=stk + ["engine/backtest.py · run_backtest"], nodes=["fill"], edges=[("win", "fill")],
      formula=f"보유수익_t = 시가_(t+2) ÷ 시가_(t+1) − 1,   비용_t = |포지션_t − 포지션_(t−1)| × {one:.5f}({one_p}),   순수익_t = 포지션_t × 보유수익_t − 비용_t",
      hint=("코드 주석의 알려진 한계: 마지막 날 들고 있던 것을 파는 비용은 넣지 않습니다"
            + (f" — 이 실행은 마지막 결정일({hi.date()}) 포지션이 0 이라 해당 없음." if last_pos == 0 else f" — 이 실행은 마지막 결정일 포지션이 {last_pos} 이라 청산 비용 한 번이 빠져 있습니다.")
            + " 세금은 넣지 않습니다(CostModel.tax_rate 는 계산에 쓰이지 않음)."),
      tensors=[{"type": "table", "title": "체결 기록 (결정일 행에 기록)", "columns": ["결정일 t", "시가_t", "신호", "포지션", "체결 시가_(t+1)", "그다음 시가_(t+2)", "보유수익", "바꾼 양", "비용", "순수익"],
                "rows": rows, "hlRows": [w_]}],
      checks=[chk(f"포지션·보유수익·비용·순수익 {n}행: 원자료 시가·손 계산 신호로 반복문 계산 = run_backtest", np.r_[posx, hr, cost, net], np.r_[D["position"].values, D["hold_return"].values, D["cost"].values, D["net_return"].values], 1e-15, kind="독립")])
    ev = D.loc[lo:hi]; r = ev["net_return"].dropna()   # 엔진 기록(직접 실행해 보기·부트스트랩 재현에 넣는 값)
    # 손 계산 경로: 원자료·손 계산 신호로 만든 순수익(net)·보유수익(hr)을 반복문 평가 구간으로 자른 것
    m_ev = (idx >= lo_h) & (idx <= hi_h); i_lo = int(idx.get_loc(lo))
    rH = pd.Series(net[m_ev], index=idx[m_ev]); bH = pd.Series(hr[m_ev], index=idx[m_ev]); pH = posx[m_ev]
    assert len(rH) == len(r) and np.isfinite(rH.values).all() and np.isfinite(bH.values).all()
    eq = np.cumprod(1 + rH.values); beq = np.cumprod(1 + bH.values)
    n_tr = 0; prev_ = 0.0                                # 엔진처럼 평가 첫날은 0 에서 시작한 것으로 보고, 포지션이 바뀐 날을 반복문으로 센다
    for v_ in pH:
        n_tr += int(v_ != prev_); prev_ = v_
    expo = float(sum(pH) / len(pH))
    cash = float(sum(1 for v_ in pH if v_ == 0) / len(pH))                       # 현금으로 보낸 평가일 비율(포지션은 0 또는 1)
    assert set(np.unique(pH)) <= {0.0, 1.0}
    S("4. 자산곡선",
      f"평가 구간({len(r)}일)의 날마다 1 + 순수익을 차례로 곱하면 자산곡선입니다(곱하기 전 값 1 — 그림의 첫 점은 첫날 수익을 이미 한 번 곱한 값이라 B&H 는 {beq[0]:.4f}, 전략은 {eq[0]:.4f}). "
      f"비교 기준 B&H 는 같은 날들 동안 계속 들고 있었을 때(비용 없음)입니다. "
      f"끝 값: 전략 {eq[-1]:.2f}배, B&H {beq[-1]:.2f}배. 보유 비율 {expo:.1%}(현금 {cash:.1%}), 포지션을 바꾼 날 {n_tr}일, "
      f"비용 합 {perf['total_cost']:.4f}(= {perf['total_cost'] * 100:.2f}%, {n_tr} × {one_p}).",
      say=(f"평가 구간 {len(r)}거래일 동안, 1 더하기 순수익을 날마다 차례로 곱한 것이 자산곡선입니다. 그림의 첫 점은 첫날의 수익을 이미 한 번 곱한 값입니다. "
           "비교 기준인 바이 앤 홀드는 같은 날들 동안 비용 없이 계속 들고 있었을 때입니다. "
           f"끝 값은 전략이 {eq[-1]:.2f}배, 바이 앤 홀드가 {beq[-1]:.2f}배입니다. "
           f"전략은 평가일의 {expo:.1%}를 보유하고 {cash:.1%}는 현금으로 보냈으며, 포지션을 {n_tr}번 바꿨습니다. 날마다 낸 비용을 수익률로 모두 더하면 {perf['total_cost'] * 100:.2f}%입니다."),
      code=C(EXP, src.lines(EXP, "has = signal.dropna()", "daily = daily_full.loc[has.index.min(): has.index.max()].copy()", 'perf = performance(daily["net_return"], daily["hold_return"], periods_per_year=ppy)'),
             src.line(EXP, "daily = daily_full.loc[has.index.min(): has.index.max()].copy()"), "_run_experiment", 2, 2),
      notes={src.line(EXP, "daily = daily_full.loc[has.index.min(): has.index.max()].copy()"): f"첫 신호 {first.date()} ~ 마지막 신호 {last.date()} 만 성적에 씀 ({len(r)}일)",
             src.line(EXP, 'perf = performance(daily["net_return"], daily["hold_return"], periods_per_year=ppy)'): "전략 = 순수익, 비교 기준 = 보유수익(B&H, 비용 없음)"},
      stack=stk, nodes=["eq"], edges=[("fill", "eq")],
      formula="자산_t = Π_(s≤t) (1 + 순수익_s),   B&H_t = Π_(s≤t) (1 + 보유수익_s)",
      tensors=[{"type": "line", "title": "자산곡선 (평가 구간, 세로축 로그)", "x": [d.strftime("%Y-%m-%d") for d in rH.index], "xlabel": "날짜", "ylog": True,
                "series": [{"name": "MA 교차", "color": "#c8710f", "y": [num(v) for v in eq]}, {"name": "B&H", "color": "#3b6fb6", "y": [num(v) for v in beq]}], "height": 260}],
      checks=[chk(f"비용 합: 원자료 경로의 바꾼 날 {n_tr} × 편도 비용, 날마다 비용의 합 = 결과 파일 total_cost", [n_tr * one, cost[m_ev].sum()], [perf0["total_cost"], perf0["total_cost"]], 1e-12, kind="독립"),
              chk("바꾼 날 수·보유 비율: 원자료 경로의 포지션을 반복문으로 센 값 = 결과 파일 n_trades·exposure", [n_tr, expo], [perf0["n_trades"], perf0["exposure"]], 1e-12, kind="독립"),
              chk(f"그림의 첫 점({eq[0]:.4f}·{beq[0]:.4f})과 끝 값({eq[-1]:.2f}·{beq[-1]:.2f}배) = 일간 기록 equity·benchmark_equity 를 평가 시작 전날 값으로 나눈 것",
                  [eq[0], beq[0], eq[-1], beq[-1]], [D["equity"].iloc[i_lo] / D["equity"].iloc[i_lo - 1], D["benchmark_equity"].iloc[i_lo] / D["benchmark_equity"].iloc[i_lo - 1],
                                                     D["equity"].loc[hi] / D["equity"].iloc[i_lo - 1], D["benchmark_equity"].loc[hi] / D["benchmark_equity"].iloc[i_lo - 1]], 1e-12, rel=True, kind="독립")])
    # ================================================================ 5. 성적표
    years = len(rH) / ppy; mu, sd = rH.mean(), rH.std(ddof=1); dn = rH[rH < 0].std(ddof=1)     # 원자료 경로의 순수익으로
    hand = {"total_return": eq[-1] - 1, "cagr": eq[-1] ** (1 / years) - 1, "ann_vol": sd * np.sqrt(ppy), "sharpe": mu / sd * np.sqrt(ppy), "sortino": mu / dn * np.sqrt(ppy),
            "mdd": float(np.min(eq / np.maximum.accumulate(eq) - 1)), "win_rate": float((rH > 0).sum() / max((rH != 0).sum(), 1)),
            "benchmark_total_return": beq[-1] - 1, "benchmark_cagr": beq[-1] ** (1 / years) - 1, "benchmark_mdd": float(np.min(beq / np.maximum.accumulate(beq) - 1))}
    lab = {"total_return": ("총수익", "Π(1+r) − 1"), "cagr": ("연복리", "(1+총수익)^(252/n) − 1"), "ann_vol": ("연 변동성", "표준편차(n−1) × √252"), "sharpe": ("샤프", "평균 ÷ 표준편차 × √252 (무위험 0)"),
           "sortino": ("소르티노", "평균 ÷ 손실일 표준편차 × √252"), "mdd": ("최대 낙폭", "min(자산 ÷ 그때까지 최고 − 1)"), "win_rate": ("승률", "수익일 ÷ 손익이 0 이 아닌 날"),
           "benchmark_total_return": ("B&H 총수익", ""), "benchmark_cagr": ("B&H 연복리", ""), "benchmark_mdd": ("B&H 최대 낙폭", "")}
    sh_bh = bH.mean() / bH.std(ddof=1) * np.sqrt(ppy)
    l_upd = src.line(EXP, "turnover = daily[\"position\"].diff().abs().fillna(daily[\"position\"].abs())"); l_upd2 = src.line(EXP, "perf.update(n_trades=int((turnover > 0).sum())")
    l_bs = src.line(EXP, 'perf["bootstrap_p_excess"] = bootstrap_excess_pvalue('); l_ds = src.line(EXP, 'skew=sk, kurt=ku, periods_per_year=ppy, trial_sr_var=trial_sr_var)')
    yrs_cal = (hi - lo).days / 365.25; cagr_cal = eq[-1] ** (1 / yrs_cal) - 1
    S("5. 성적표",
      f"표의 10개 지표(총수익·연복리·연 변동성·샤프·소르티노·최대 낙폭·승률, B&H 의 총수익·연복리·최대 낙폭)와 초과 총수익은 engine/backtest.py 의 <b>performance</b> 에서 계산합니다. "
      f"바꾼 횟수·회전율·보유 비율·비용 합은 {EXP} 의 _run_experiment({l_upd}–{l_upd2}행), 부트스트랩·Deflated Sharpe p 는 같은 함수 {l_bs}–{l_ds}행에서 계산합니다. "
      f"원자료에서 따로 계산한 순수익(3. 체결 단계의 검증)으로 다시 계산해 결과 파일의 값과 맞춰 봅니다. 샤프 {hand['sharpe']:.3f} (B&H 를 같은 식으로 계산하면 {sh_bh:.3f}). 최대 낙폭 {pct(hand['mdd'], 1)} (B&H {pct(hand['benchmark_mdd'], 1)}).",
      say=("성적표입니다. 표의 지표는 성적표 함수가 계산한 값이고, 그 값을 손으로 다시 계산해 결과 파일과 맞췄습니다. "
           f"샤프는 일간 순수익의 평균을 표본 표준편차로 나눈 뒤, 1년을 {ppy}거래일로 보고 그 제곱근을 곱한 값이며, 무위험 수익률은 0으로 둡니다. "
           f"같은 식으로 계산하면 바이 앤 홀드의 샤프는 {sh_bh:.3f}이고, 이 전략은 그보다 낮은 {hand['sharpe']:.3f}입니다. "
           f"최대 낙폭은 자산이 그때까지의 최고점에서 가장 많이 떨어진 비율로, 전략은 {pct(hand['mdd'], 1)}, 바이 앤 홀드는 {pct(hand['benchmark_mdd'], 1)}입니다."),
      code=C(BT, src.span(BT, "    r = r.dropna()  # 계산할 수 없는 날(NaN) 제외", '        "win_rate": float((r > 0).sum() / max((r != 0).sum(), 1)),  # max(…,1): 0 으로 나누기 방지'),
             src.line(BT, "sharpe = float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float(\"nan\")"), "performance", 2, 2),
      notes={src.line(BT, "cagr = float(eq.iloc[-1] ** (1 / years) - 1)"): f"연복리: 최종 배수^(1/년수) − 1, 년수 = {len(r)} ÷ {ppy}",
             src.line(BT, "sharpe = float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float(\"nan\")"): f"평균 ÷ 표본 표준편차(n−1) × √{ppy}, 무위험 수익률 0 → {hand['sharpe']:.3f}",
             src.line(BT, "dd = eq / eq.cummax() - 1.0"): f"그때까지 최고점 대비 낙폭(0 이하) → 가장 깊은 값 = 최대 낙폭 {pct(hand['mdd'], 1)}"},
      lab={**lab_perf(r.values + 0.0, perf0, ppy), "kind": "식"},          # + 0.0: −0.0 을 0.0 으로(계산 결과는 같음)
      stack=stk + ["engine/backtest.py · performance"], nodes=["perf"], edges=[("eq", "perf")],
      hint=(f"가정 ① 현금 수익 0: 순수익 = 포지션 × 보유수익 − 비용 이라, 현금으로 있는 날(이 실행은 평가일의 {cash:.1%})의 수익은 0 입니다(이자 없음 — B&H 는 늘 보유라 영향 없음). "
            "② 가격 수익률만: 분배금은 넣지 않습니다(DECISIONS 2026-11-03, 전략·B&H 모두). "
            f"③ 연율화는 1년 = {ppy}거래일로 셉니다 — 이 평가 구간은 달력으로 {yrs_cal:.2f}년(연 {len(r) / yrs_cal:.0f}거래일꼴)이라, 달력 기준 연복리는 {pct(cagr_cal)}(표 {pct(hand['cagr'])})입니다. "
            "소르티노의 분모는 '손실 난 날만의 표본표준편차'(r[r<0].std)입니다 — 교과서의 하방편차(모든 날에서 0 아래 부분의 제곱평균 제곱근)와 다른 정의라, "
            "다른 문헌의 소르티노와 바로 비교할 수 없습니다. 최대 낙폭은 첫날 자산(1+r₁)부터의 최고점을 기준으로 합니다."),
      tensors=[{"type": "table", "title": "성적표: 손 계산 vs 결과 파일", "columns": ["지표", "식", "손 계산", "결과 파일"],
                "rows": [[lab[k][0], lab[k][1], f"{v:.6f}", f"{perf0[k]:.6f}"] for k, v in hand.items()]}],
      checks=[chk("성적표 10개 지표: 원자료 경로의 순수익으로 손 계산 = 결과 파일", list(hand.values()), [perf0[k] for k in hand], 1e-10, rel=True, kind="독립")])
    # ================================================================ 6. 검정
    valid = ev.dropna(subset=["net_return", "hold_return"]); dd = (valid["net_return"] - valid["hold_return"]).values
    blk = max(5, 2 * cfg["target"]["k"]); obs = dd.mean(); cen = dd - obs; NB = 2000

    def boot_means(seed):
        """엔진 bootstrap_excess_pvalue 와 같은 알고리즘(겹칠 수 있는 블록을 무작위 시작점에서 뽑아 이어 붙임)을 옮긴 것 — 같은 시드면 엔진과 같은 난수."""
        rng_ = np.random.default_rng(seed); nb_ = int(np.ceil(len(dd) / blk)); out = np.empty(NB)
        for i in range(NB):
            st_ = rng_.integers(0, len(dd) - blk + 1, size=nb_)
            out[i] = np.concatenate([cen[s:s + blk] for s in st_])[:len(dd)].mean()
        return out
    nb = int(np.ceil(len(dd) / blk)); means = boot_means(0)
    pb = float((means >= obs).mean()); n_ge = int((means >= obs).sum())
    p_seed = {s_: float((boot_means(s_) >= obs).mean()) for s_ in (1, 2)}          # 시드만 바꾼 것 — 몬테카를로 흔들림을 보이기 위해
    se_mc = float(np.sqrt(pb * (1 - pb) / NB))                                      # 2,000번 뽑기의 몬테카를로 표준오차 √(p(1−p)/2000)
    # 다른 계산법: Newey–West(바틀렛 가중, 시차 = 블록 길이 − 1) 장기분산으로 표본평균의 표준오차를 구한 정규 근사 — 이웃한 날의 상관을 넣은 점근 p
    import math as _math
    nw_lag = blk - 1; e_ = dd - obs; n_d = len(dd)
    gam_ = [float((e_[l_:] * e_[:n_d - l_]).sum() / n_d) for l_ in range(nw_lag + 1)]
    se_nw = _math.sqrt((gam_[0] + 2 * sum((1 - l_ / (nw_lag + 1)) * gam_[l_] for l_ in range(1, nw_lag + 1))) / n_d)
    p_nw = 0.5 * _math.erfc((obs / se_nw) / _math.sqrt(2))                       # = 1 − Φ(실제 평균 ÷ 표준오차)
    import statsmodels.api as _sm
    se_nw_sm = float(_sm.OLS(dd, np.ones(n_d)).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lag, "use_correction": False}).bse[0])
    from engine.backtest import bootstrap_excess_pvalue
    p_eng = {s_: float(bootstrap_excess_pvalue(valid["net_return"], valid["hold_return"], n_boot=NB, block=blk, seed=s_)) for s_ in (1, 2)}
    qs = np.quantile(means, [0.01, 0.05, 0.5, 0.95, 0.99])
    S("6. 우연인가 — 부트스트랩",
      f"<b>전략이 B&H 보다 낫다(평균 초과수익 > 0)고 볼 근거가 있는지</b> 봅니다(단측 검정, 귀무가설: 평균 초과수익 ≤ 0). 초과수익 = 날마다 전략 순수익 − B&H 보유수익, "
      f"실제 평균은 하루 <b>{obs * 1e4:.2f}bp</b> 입니다. 초과수익에서 평균을 빼 '효과 0' 인 세계로 옮긴 뒤, 무작위 시작점에서 {blk}거래일씩 떼어 낸 블록(겹칠 수 있음)을 이어 붙인 "
      f"가짜 역사 {NB:,}개를 만듭니다(시드 0). 가짜 역사의 평균이 실제 평균 이상인 비율 = p값 = <b>{pb:.3f}</b>({n_ge:,} ÷ {NB:,}). 0.05 보다 작아야 'B&H 보다 낫다' 고 볼 근거가 됩니다. "
      f"p 는 난수에 따라 흔들립니다 — 시드만 1·2 로 바꾸면 {p_seed[1]:.4f}·{p_seed[2]:.4f}, {NB:,}번 뽑기의 몬테카를로 표준오차 √(p(1−p)/{NB:,}) ≈ {se_mc:.3f}. "
      f"계산법이 다른 대조: 이웃한 날의 상관을 넣은 Newey–West 정규 근사(시차 {nw_lag})로 구하면 p ≈ {p_nw:.3f} — 결론(0.05 보다 큼)이 같습니다.",
      say=("부트스트랩은 평균 초과수익이 0보다 커서 전략이 바이 앤 홀드보다 낫다고 볼 근거가 있는지 봅니다. "
           f"초과수익은 날마다 전략의 순수익에서 바이 앤 홀드의 보유수익을 뺀 값이고, 실제 평균은 하루 {obs * 1e4:.2f} 베이시스 포인트입니다. "
           f"초과수익에서 평균을 빼 효과가 0인 세계로 옮기고, 무작위 시작점에서 {blk}거래일씩 떼어 낸 블록을 이어 붙여 가짜 역사 {NB:,}개를 만듭니다. "
           "블록으로 떼는 것은 이웃한 날끼리 닮은 성질을 남기려는 것입니다. "
           f"가짜 평균이 실제 평균 이상인 비율이 p값이고, 이 실행에서는 {pb:.3f}입니다. "
           f"시드만 바꿔도 {p_seed[1]:.4f}, {p_seed[2]:.4f}로, 몬테카를로 표준오차가 약 {se_mc:.3f}이라 이 정도는 흔들립니다. "
           + ("어느 쪽이든 0.05보다 훨씬 커서, 전략이 바이 앤 홀드보다 낫다고 볼 근거가 없습니다." if min([pb] + list(p_seed.values())) > 0.05 else "0.05와 비교할 때 이 흔들림을 함께 봐야 합니다.")),
      code=C(BT, src.span(BT, '    d = (strategy - bench).dropna().values  # 날마다 초과수익', "    return float((means >= obs).mean())  # 가짜 세계에서 실제 이상으로 좋게 나온 비율"),
             src.line(BT, "starts = rng.integers(0, n - block + 1, size=nb)  # 블록 시작 위치를 무작위로"), "bootstrap_excess_pvalue", 2, 2),
      notes={src.line(BT, "centered = d - obs"): f"실제 평균 {obs:.2e} 을 빼서 평균 0 = '효과 없음' 세계",
             src.line(BT, "starts = rng.integers(0, n - block + 1, size=nb)"): f"{blk}일 블록 {nb}개의 시작 위치를 무작위로 (시드 0, 블록끼리 겹칠 수 있음)",
             src.line(BT, "return float((means >= obs).mean())"): f"가짜 평균 ≥ 실제 평균인 비율 = p = {n_ge:,} ÷ {NB:,} = {pb:.3f}"},
      lab={**lab_boot(dd + 0.0, blk, NB, 0, perf0["excess_mean_per_period"], perf0["bootstrap_p_excess"]), "kind": "재실행"},
      stack=stk + ["engine/backtest.py · bootstrap_excess_pvalue"], nodes=["test"], edges=[("perf", "test")],
      formula=f"p = #{{ 가짜 평균 ≥ 실제 평균 {obs:.2e} }} ÷ {NB}",
      hint=(f"함수의 기본값은 block = 20 이지만, 엔진(_run_experiment)은 block = max(5, 2 × k) 로 부릅니다 — 이 실행(k = {cfg['target']['k']})은 {blk}일. 귀무가설은 '평균 초과수익 ≤ 0' 입니다. "
            f"Newey–West 정규 근사는 블록 부트스트랩과 계산법이 다른 대조라 값이 같을 필요는 없고, 결론(0.05 와의 대소)만 비교합니다. "
            f"시차 {nw_lag} 는 블록 길이 {blk} − 1 로 정했습니다(같은 블록 안의 이웃 상관까지)."),
      tensors=[{"type": "table", "title": f"가짜 역사 {NB:,}개의 평균 초과수익 분포 (일간, 시드 0)", "columns": ["", "값"],
                "rows": [["1% 분위", f"{qs[0]:.2e}"], ["5% 분위", f"{qs[1]:.2e}"], ["중앙", f"{qs[2]:.2e}"], ["95% 분위", f"{qs[3]:.2e}"], ["99% 분위", f"{qs[4]:.2e}"], ["실제 평균", f"{obs:.2e}"]]}],
      checks=[chk("같은 알고리즘·같은 시드 재현 p = 결과 파일", pb, perf0["bootstrap_p_excess"], 0, kind="재실행"),
              chk(f"다른 계산법(Newey–West 정규 근사, 시차 {nw_lag}) p {p_nw:.3f}: 결론(p > 0.05)이 부트스트랩 p {pb:.3f} 와 같음 — 값의 일치 판정이 아님",
                  float((p_nw > 0.05) == (pb > 0.05)), 1.0, 0, kind="참고", show_diff=False),
              chk(f"시드 1·2 의 p {p_seed[1]:.4f}·{p_seed[2]:.4f}: 옮긴 코드 = 엔진 bootstrap_excess_pvalue(seed = 1·2)", [p_seed[1], p_seed[2]], [p_eng[1], p_eng[2]], 0, kind="재실행"),
              chk(f"몬테카를로 표준오차 {se_mc:.3f} = √({n_ge} × {NB - n_ge}) ÷ {NB}^1.5, Newey–West 표준오차 손 계산 = statsmodels OLS(HAC, 시차 {nw_lag})",
                  [se_mc, se_nw], [np.sqrt(n_ge * (NB - n_ge)) / NB ** 1.5, se_nw_sm], 1e-10, rel=True, kind="식")])
    # 왜도·첨도: 원자료 경로의 순수익(rH)으로 적률 공식에서 직접(엔진 return_moments 는 scipy 의 skew·kurtosis — 여기서는 쓰지 않음)
    xh = rH.values; mx = xh.mean(); m2 = float(((xh - mx) ** 2).mean()); m3 = float(((xh - mx) ** 3).mean()); m4 = float(((xh - mx) ** 4).mean())
    sk_, ku_ = m3 / m2 ** 1.5, m4 / m2 ** 2
    n_zero = int((xh == 0).sum()); xz = xh[xh != 0]; mz = xz.mean(); ku_nz = float(((xz - mz) ** 4).mean() / ((xz - mz) ** 2).mean() ** 2)
    to_ev = to[m_ev]; hr_ev = hr[m_ev]
    n_zero_cash = int(((pH == 0) & (to_ev == 0)).sum())                 # 현금이고 바꾸지 않은 날(비용 0)
    n_zero_hold = int(((pH == 1) & (hr_ev == 0) & (to_ev == 0)).sum())  # 보유했지만 두 시가가 같아 보유수익 0 인 날
    from scipy.stats import kurtosis as _kurt
    from scipy.stats import norm
    srp = hand["sharpe"] / np.sqrt(ppy); N = int(perf0["n_trials_so_far"]); V = perf0.get("trial_sr_var"); n_per = len(rH)
    if V is None or not np.isfinite(V) or V <= 0:
        raise ValueError("결과 파일에 trial_sr_var 가 없습니다 — 대본의 V 설명(저장된 결과들의 분산)이 맞지 않으므로 멈춤")
    gam = 0.5772156649015329

    def dsr(N_, ku):
        sr0_ = float(np.sqrt(V) * ((1 - gam) * norm.ppf(1 - 1 / N_) + gam * norm.ppf(1 - 1 / (N_ * np.e)))) if N_ > 1 else 0.0
        z_ = (srp - sr0_) * np.sqrt(n_per - 1) / np.sqrt(1 - sk_ * srp + (ku - 1) / 4 * srp ** 2)
        return sr0_, float(z_), float(1 - norm.cdf(z_))
    sr0, z, pd_ = dsr(N, ku_)
    p_k3 = dsr(N, 3.0)[2]                                # 첨도를 정규분포 값 3 으로 넣으면
    p_n1 = dsr(1, ku_)[2]                                # 보정 없이(N = 1, SR₀ = 0)
    from engine.backtest import deflated_sharpe_pvalue
    p_n1_e = float(deflated_sharpe_pvalue(perf0["sharpe"], perf0["n_periods"], 1, skew=perf0["return_skew"], kurt=perf0["return_kurt"], periods_per_year=ppy, trial_sr_var=V))
    p_k3_e = float(deflated_sharpe_pvalue(perf0["sharpe"], perf0["n_periods"], N, skew=perf0["return_skew"], kurt=3.0, periods_per_year=ppy, trial_sr_var=V))
    vs = _v_sample(cap, V)
    assert vs["same_old_noeval"] and vs["n_old_seal"] == vs["n_old"], "옛 결과 묶음이 예상과 다름 — 힌트 문장을 고칠 것"
    S("6. 우연인가 — Deflated Sharpe",
      f"<b>여러 번 시험해서 운 좋게 나온 최고 성적이 아닌가</b>: 매매 모드에서 지금까지 돌린 실험 수(실험 카운터, 이 실험 포함) N = {N} 을 반영합니다. N 번 시도했을 때 운으로도 기대되는 최고 일간 샤프 SR₀ = {sr0:.4f}"
      f"(연 샤프로 {sr0 * np.sqrt(ppy):.3f}), 이 전략의 일간 샤프 {srp:.4f}. "
      f"V = {V:.2e} 는 그때 results/trading 에 저장돼 있던 일간 결과 {vs['n']}개의 기간 샤프 분산입니다 — N = {N} 은 주간·월간 실행과 결과 파일이 남지 않은 실행까지 센 수라 V 의 표본과 다릅니다. "
      f"수익률의 치우침(왜도 {sk_:.2f})·꼬리(첨도 {ku_:.1f})까지 넣은 p = <b>{pd_:.4f}</b>. "
      f"평가일 {n_per}일 중 {n_zero:,}일은 순수익이 정확히 0 입니다(현금이면서 포지션을 바꾸지 않은 {n_zero_cash:,}일 + 보유했지만 체결·청산 시가가 같았던 {n_zero_hold}일). 0 인 날을 빼고 계산하면 첨도는 {ku_nz:.2f} 입니다(이 0 들이 첨도를 키움). "
      f"첨도 항이 p 에 주는 영향은 작습니다 — 첨도를 정규분포 값 3 으로 넣어도 p = {p_k3:.4f}. "
      f"보정 없이(N = 1, SR₀ = 0) 계산해도 p = {p_n1_e:.3f} 로 0.05 보다 큽니다. N 이 1 보다 크면 SR₀ ≥ 0 이라 p 는 이보다 커지기만 하므로, 이 결론은 N·V 를 어떻게 잡아도 같습니다.",
      say=("다음은 여러 번 시험한 것에 대한 보정입니다. 전략을 많이 시험하고 가장 좋은 것만 고르면, 운만으로도 높은 샤프가 나옵니다. "
           f"이 실험은 매매 모드 실험 카운터로 {N}번째였고, {N}번 시도했을 때 운으로도 기대되는 최고 일간 샤프는 {sr0:.4f}, 연 샤프로 {sr0 * np.sqrt(ppy):.3f}입니다. "
           f"이 전략의 일간 샤프는 {srp:.4f}로 그보다 {'낮습니다' if srp < sr0 else '높습니다'}. "
           f"수익률의 왜도와 첨도까지 넣은 디플레이티드 샤프 p값은 {pd_:.4f}로, "
           + ("이 샤프가 운으로 기대되는 수준보다 높다고 볼 근거가 없습니다. " if pd_ > 0.05 else "0.05보다 작습니다. ")
           + (f"실험 수 보정을 아예 빼도 p값은 {p_n1_e:.3f}로, 0.05보다 큽니다. " if p_n1_e > 0.05 else f"실험 수 보정을 빼면 p값은 {p_n1_e:.3f}입니다. ")
           + "이동평균 교차는 여러 시도 가운데 골라낸 것이 아니라 미리 정해 둔 기준선이라, 여기서는 보수적인 보정입니다."),
      code=C(BT, src.span(BT, "    sr = sharpe_annual / sqrt(periods_per_year)  # 연 샤프 → 하루(기간) 샤프", "    return float(1 - norm.cdf(z))"),
             src.line(BT, "sr0 = sqrt(v) * ((1 - gamma) * z1 + gamma * z2)  # N번 시도했을 때 운으로도 기대되는 최고 샤프"), "deflated_sharpe_pvalue", 2, 2),
      notes={src.line(BT, "sr = sharpe_annual / sqrt(periods_per_year)"): f"연 샤프 {perf['sharpe']:.3f} ÷ √{ppy} = 일간 {srp:.4f}",
             src.line(BT, "sr0 = sqrt(v) * ((1 - gamma) * z1 + gamma * z2)"): f"N = {N}, V = {V:.2e} → SR₀ = {sr0:.4f}",
             src.line(BT, "z = (sr - sr0) * sqrt(n_periods - 1) / sqrt(denom)"): f"z = {z:.3f} → p = 1 − Φ(z) = {pd_:.4f}"},
      lab={**lab_dsr(perf0["sharpe"], perf0["n_periods"], perf0["n_trials_so_far"], perf0["trial_sr_var"], perf0["return_skew"], perf0["return_kurt"], ppy, sr0, z, perf0["deflated_sharpe_p"]), "kind": "식"},
      stack=stk + ["engine/backtest.py · deflated_sharpe_pvalue (Bailey & López de Prado 2014)"], nodes=["test"],
      formula="SR₀ = √V · [(1−γ)Φ⁻¹(1 − 1/N) + γΦ⁻¹(1 − 1/(Ne))],   z = (SR − SR₀)·√(n−1) ÷ √(1 − 왜도·SR + (첨도−1)/4·SR²),   p = 1 − Φ(z)",
      tensors=[{"type": "num", "title": "", "items": [{"label": "실험 수 N", "value": str(N)}, {"label": "SR₀ (일간)", "value": f"{sr0:.4f}"}, {"label": "SR (일간)", "value": f"{srp:.4f}"},
                                                     {"label": "z", "value": f"{z:.3f}"}, {"label": "p", "value": f"{pd_:.4f}"}, {"label": "p (N = 1)", "value": f"{p_n1_e:.3f}"}]}],
      checks=[chk("왜도·첨도: 원자료 경로의 순수익으로 적률 공식(scipy 안 씀) = 결과 파일 return_skew·return_kurt", [sk_, ku_], [perf0["return_skew"], perf0["return_kurt"]], 1e-10, rel=True, kind="독립"),
              chk("Deflated Sharpe p: 원자료 경로의 샤프·왜도·첨도와 결과 파일의 N·V 로 같은 식 = 결과 파일", pd_, perf0["deflated_sharpe_p"], 1e-10, kind="독립"),
              chk(f"V = 지금 남은 일간 결과 가운데 실험 카운터 값 < {N} 인 {vs['n']}개(봉인 평가 제외)의 기간 샤프 분산을 결과 파일들에서 다시 계산 = 결과 파일 trial_sr_var", vs["v"], V, 1e-12, rel=True, kind="독립"),
              chk("보정 없는 p(N = 1)·첨도 3 으로 넣은 p: 손 계산 식 = 엔진 deflated_sharpe_pvalue", [p_n1, p_k3], [p_n1_e, p_k3_e], 1e-10, kind="식"),
              chk(f"순수익 0 인 날 {n_zero:,} = 현금·안 바꾼 날 {n_zero_cash:,} + 보유·보유수익 0·안 바꾼 날 {n_zero_hold}, 0 을 뺀 첨도 {ku_nz:.2f} = scipy kurtosis",
                  [n_zero, ku_nz], [n_zero_cash + n_zero_hold, float(_kurt(xz, fisher=False))], 1e-10, kind="식"),
              chk(f"V 표본 {vs['n']}개의 구성(결과 파일): 옛 데이터 버전 {vs['n_old']}개 = 평가 구간 기록 없음과 같은 묶음 = period 끝이 봉인 뒤, "
                  f"일간 {vs['n']} + 주간·월간 {vs['n_wm']}(frequency 로 직접 셈) = 카운터 값 < {N} 인 결과 전체 {vs['n_before_all']}",
                  [float(vs["same_old_noeval"]), vs["n_old_seal"], vs["n"] + vs["n_wm"]], [1.0, vs["n_old"], vs["n_before_all"]], 0, kind="식")],
      hint=("엔진은 매매 모드의 모든 결과에 같은 식의 다중검정 보정을 적용합니다. N 은 개발 결과에서는 이 실행을 포함한 실험 카운터(results/trading/experiment_counter.json) 값이고"
            "(데이터 버전·설정·주기를 가리지 않고 셈), 봉인 평가에서는 카운터를 올리지 않고 그때까지의 개발 시도 수를 씁니다(cli.py). "
            "MA 교차는 여러 시도 가운데 고른 것이 아니라 미리 정한 기준선이라, 이 결과에는 보정이 보수적입니다(p 가 크게 나오는 쪽). "
            "V 는 그때까지 저장된 매매 모드 결과 가운데 같은 주기(일간)인 것들의 기간 샤프 분산입니다(봉인 평가 제외, 결과 파일 trial_sr_var). "
            f"이 실행의 V 는 그보다 먼저 저장된(실험 카운터 값이 더 작은) 일간 결과 {vs['n']}개로 다시 계산하면 그대로 나옵니다 — 그중 {vs['n_old']}개는 옛 데이터 버전이면서 "
            "평가 구간이 기록되지 않은 옛 결과로, 봉인 구간까지 포함해 평가됐습니다(결과 파일의 period 끝이 봉인 시작 뒤; docs/STATUS.md). "
            f"나머지 {vs['n_rest']}개 중 {vs['n_rest_other']}개도 평가 구간이 이 실행과 다릅니다. "
            f"지금 남아 있는 매매 결과 파일 가운데 이 실행보다 먼저 저장된(카운터 값이 작은) 것은 주간·월간 {vs['n_wm']}개를 더해 {vs['n_before_all']}개라, N 과 V 의 표본은 서로 다른 것을 셉니다. "
            "시도가 늘수록 SR₀ 가 올라가 같은 샤프라도 p 가 커집니다."))
    # ================================================================ 7. 다른 규칙
    oth = cap["others"]

    def other_step(name, title, narr, hand_fn, code_fn, func, formula, say_fn, notes_fn, hint=None):
        o_ = oth.get(name)
        if not o_ or "error" in o_:      # 대체 화면을 만들지 않는다 — 대본이 기록에 따라 조용히 바뀌지 않게
            raise RuntimeError(f"실행 기록(rule_capture.pkl)에 {name} 기록이 없거나 실패: {(o_ or {}).get('error', '없음')} — rule_capture 를 다시 돌릴 것")
        Pp = o_["prices"]; pe = o_["pos"].values
        use_raw = name in ("momentum", "rsi_rule")       # 종가만 쓰는 규칙은 원자료 종가로 손 계산(같은 데이터 버전·같은 날짜여야 함)
        if use_raw:
            assert o_["data_version"] == cap["data_version"] and len(Pp) == n and (Pp.index == idx).all(), f"{name}: 원자료와 날짜·데이터 버전이 다름"
        ph = hand_fn(o_, raw if use_raw else Pp)
        last = np.flatnonzero(~np.isnan(pe) & (Pp.index <= hi))[-260:]   # 평가 구간 끝까지의 마지막 260일
        S("7. 다른 규칙", narr(o_, ph), say=say_fn(o_),
          code=C(code_fn, src.span(code_fn, "def positions(self, prices, X=None):", "return pos.values" if name != "buy_and_hold" else "return np.ones(len(prices))"), src.line(code_fn, "def positions(self, prices, X=None):"), func, 2, 2),
          notes=notes_fn(o_), hint=hint,
          stack=[f"{code_fn} · Model.positions"], nodes=["oth"], formula=formula,
          tensors=[{"type": "line", "title": f"{title}: 평가 구간 마지막 260일의 포지션 (1 = 보유)", "x": [Pp.index[i].strftime("%Y-%m-%d") for i in last], "xlabel": "날짜", "ylo": -0.1, "yhi": 1.1,
                    "series": [{"name": "포지션", "color": "#2f8f5b", "y": [num(pe[i]) for i in last]}], "height": 160}],
          checks=[chk(f"{title} 전체 {len(pe)}일 포지션: " + ("엔진 포지션이 모든 날 1 인지 확인" if name == "buy_and_hold" else
                      f"{'원자료 종가로' if use_raw else '실행 기록의 ' + o_['params']['column'] + '(SOX 변화율)로'} 반복문 손 계산 = 엔진"), ph, pe, 0, kind="독립" if use_raw else "식")])
    other_step("buy_and_hold", "Buy & Hold", lambda o_, ph: "<b>Buy & Hold</b>: 항상 보유(1). 모든 비교 화면의 기준선이고, 백테스트의 benchmark(비용 없는 계속 보유)와 같은 생각입니다(B&H 모델은 평가 첫날 진입 비용 한 번을 냄).",
               lambda o_, Pp: np.ones(len(Pp)), "models/buy_and_hold.py", "Model.positions", "신호_t = 1",
               lambda o_: ("다른 규칙들도 똑같은 평가 구간과 백테스트를 거칩니다. 바이 앤 홀드 모델은 모든 날 보유 신호를 냅니다. "
                           "백테스트의 비교 기준과 같은 생각이지만, 모델로 돌리면 평가 첫날 진입 비용을 한 번 냅니다."),
               lambda o_: {src.line("models/buy_and_hold.py", "return np.ones(len(prices))"): "모든 날 1(보유) — 평가 첫날 0→1 로 바뀔 때만 비용"})

    def mom(o_, Pp):
        cc = Pp["close"].astype(float).values; lb = o_["params"]["lookback"]
        return np.array([np.nan if i < lb else float(cc[i] / cc[i - lb] - 1 > 0) for i in range(len(cc))])
    MOM = "models/momentum.py"
    other_step("momentum", "모멘텀", lambda o_, ph: f"<b>시계열 모멘텀</b>: 과거 {o_['params']['lookback']}일 수익률이 0 보다 크면 보유. (Moskowitz·Ooi·Pedersen 2012 의 단순형)",
               mom, MOM, "Model.positions", "신호_t = 1  if  종가_t ÷ 종가_(t−L) − 1 > 0",
               lambda o_: (f"시계열 모멘텀은 오늘 종가가 {o_['params']['lookback']}거래일 전 종가보다 높으면, 곧 과거 {o_['params']['lookback']}거래일 수익률이 0보다 크면 보유하고, 아니면 현금으로 둡니다. "
                           "모스코위츠와 동료들이 2012년에 낸 시계열 모멘텀의 단순한 형태입니다."),
               lambda o_: {src.line(MOM, 'r = c / c.shift(self.params["lookback"]) - 1'): f"{o_['params']['lookback']}일 전 종가 대비 수익률",
                           src.line(MOM, "pos = (r > 0).astype(float)"): "0 보다 크면 1(보유), 아니면 0(현금)",
                           src.line(MOM, "pos[r.isna()] = np.nan"): f"처음 {o_['params']['lookback']}일은 결정 불가(빈칸)"})

    def rsi_h(o_, Pp):
        cc = Pp["close"].astype(float).values; w = o_["params"]["window"]; lo_, up_ = o_["params"]["lower"], o_["params"]["upper"]; a = 1 / w
        up = dn = None; out = np.full(len(cc), np.nan); state = None
        for i in range(1, len(cc)):
            d_ = cc[i] - cc[i - 1]; u_, v_ = max(d_, 0.0), max(-d_, 0.0)
            up = u_ if up is None else (1 - a) * up + a * u_; dn = v_ if dn is None else (1 - a) * dn + a * v_
            rs = np.nan if dn == 0 else up / dn; rv = 100 - 100 / (1 + rs) if np.isfinite(rs) else np.nan
            if np.isnan(rv): continue
            if rv < lo_: state = 1.0
            elif rv > up_: state = 0.0
            out[i] = state if state is not None else 0.0
        return out
    RSI = "models/rsi_rule.py"
    rsi_res = has_result("rsi_rule")
    if rsi_res:   # 대본이 '결과 파일 없음' 을 말한다 — 결과가 생기면 조용히 바꾸지 말고 대본을 다시 쓸 것
        raise AssertionError(f"RSI 결과 파일이 생겼습니다({rsi_res}) — RSI 단계의 '결과 파일 없음' 문장을 고칠 것")
    other_step("rsi_rule", "RSI", lambda o_, ph: f"<b>RSI 규칙</b>: RSI({o_['params']['window']}) 가 {o_['params']['lower']:g} 아래로 내려가면 매수, {o_['params']['upper']:g} 위로 올라가면 매도, 그 사이는 직전 상태 유지. "
               "RSI = 100 − 100/(1 + RS) (0~100), RS = Wilder 평활(EMA, α = 1/창)한 상승폭 ÷ 하락폭입니다"
               + "(결과 파일 없음 — results/trading 에 이 모델의 결과가 없어 원자료 종가로 신호만 확인).",
               rsi_h, RSI, "Model.positions", "RSI < lower → 1,  RSI > upper → 0,  그 사이 → 직전 값",
               lambda o_: (f"RSI 규칙은 {o_['params']['window']}거래일 RSI가 {o_['params']['lower']:g} 아래로 내려가면 사고, {o_['params']['upper']:g} 위로 올라가면 팝니다. "
                           "그 사이에서는 직전 상태를 유지하고, 첫 매수 신호 전까지는 현금입니다. "
                           f"상승폭과 하락폭은 각각 지수이동평균으로 평활하고, 평활 계수는 {o_['params']['window']}분의 1입니다. "
                           "상승 평균을 하락 평균으로 나눈 비율을 0에서 100 사이 값으로 바꾼 것이 RSI입니다. "
                           "이 규칙은 아직 결과 파일이 없어서, 같은 가격으로 신호만 확인했습니다."),
               lambda o_: {src.line(RSI, 'r = rsi(prices["close"].astype(float), self.params["window"])'): f"engine/features.py rsi: 상승폭·하락폭을 α = 1/{o_['params']['window']} 로 평활, RS = 상승 ÷ 하락 → RSI = 100 − 100/(1+RS)",
                           src.line(RSI, 'pos[r < self.params["lower"]] = 1.0'): f"RSI < {o_['params']['lower']:g} → 매수(1)",
                           src.line(RSI, "pos.loc[first:] = pos.loc[first:].ffill().fillna(0.0)"): "그 사이는 직전 상태 유지, 첫 신호 전은 0(현금)"},
               hint="engine/features.py 의 rsi 는 ewm(alpha = 1/창, adjust=False) 로 첫 값부터 바로 평활합니다. Wilder 의 원래 방식(처음 창 길이만큼은 단순평균으로 시작)과는 초반 값이 다르고, 시간이 지날수록 차이가 줄어듭니다.")

    def sox_h(o_, Pp):
        s = o_["X"][o_["params"]["column"]].astype(float).reindex(Pp.index).values; w, q = int(o_["params"]["window"]), float(o_["params"]["q"])
        out = np.full(len(s), np.nan)
        for i in range(len(s)):
            past = s[max(0, i - w):i]; past = past[~np.isnan(past)]
            if len(past) < w // 2 or np.isnan(s[i]) or i - w < 0 and np.sum(~np.isnan(s[:i])) < w // 2:
                continue
            thr = np.quantile(past, q) if o_["params"]["mode"] == "avoid_top" else np.quantile(past, 1 - q)
            out[i] = float(not (s[i] > thr)) if o_["params"]["mode"] == "avoid_top" else float(s[i] < thr)
        return out
    SOX = "models/sox_gap_fade.py"
    sox_res = has_result("sox_gap_fade")
    if sox_res:   # 대본이 '지금 엔진으로 낸 결과 파일 없음' 을 말한다
        raise AssertionError(f"SOX 급등 회피 결과 파일이 생겼습니다({sox_res}) — SOX 단계의 '결과 파일 없음' 문장을 고칠 것")
    so_ = oth.get("sox_gap_fade")
    if not so_ or "error" in so_:
        raise RuntimeError(f"실행 기록에 sox_gap_fade 기록이 없거나 실패: {(so_ or {}).get('error', '없음')}")
    s_ev = so_["X"][so_["params"]["column"]].astype(float).reindex(so_["prices"].index).loc[lo:hi]
    n_rep = int(((s_ev.diff() == 0) & s_ev.notna()).sum())
    sox_hint = (f"미국 휴장일처럼 그날 SOX 값이 없는 한국 거래일에는 ffill 로 직전 미국 거래일의 변화율이 한 번 더 쓰입니다 — 평가 구간 {len(s_ev)}일 중 {n_rep}일이 "
                "바로 전날과 같은 값입니다(같은 급등이 이틀 연속 회피를 부를 수 있음). 미래 정보는 아니지만 '그날 변화율' 이 아닌 날이 있습니다.")
    other_step("sox_gap_fade", "SOX 급등 회피", lambda o_, ph: f"<b>SOX 급등 회피</b>(2차 실험 H1 에서 나온 후보): 기본 보유, 미국 반도체지수 SOX 의 그날 변화율이 <b>과거 {o_['params']['window']}일(그날 제외) 분포의 상위 {1 - o_['params']['q']:.0%}</b> 를 넘으면 다음 날 현금. "
               "분위수는 t 이전 값으로만 계산합니다. 미국 t일 마감은 한국 t+1일 개장 전이라 lag 0 이 허용됩니다(결정 시점 = t+1 개장 직전). "
               + f"지금 엔진(평가 구간을 학습 모델과 맞춘 뒤)으로 낸 설정 {o_['source']} 의 결과 파일은 results/trading 에 없어, 실행 기록의 SOX 변화율로 신호만 확인합니다"
               "(2차 실험 때 옛 평가 구간으로 돌린 결과는 docs/results_round2.md 의 'H1 정식 검정' 표).",
               sox_h, SOX, "Model.positions", "신호_t = 0  if  SOX변화_t > 분위수_q(SOX변화_(t−w..t−1))  else 1",
               lambda o_: ("SOX 급등 회피는 2차 실험의 가설 H1에서 나온 후보입니다. 기본은 보유이고, 미국 반도체지수 SOX의 그날 변화율이 "
                           f"그날을 뺀 과거 {o_['params']['window']}거래일 분포의 상위 {1 - o_['params']['q']:.0%}를 넘으면 다음 날은 현금으로 둡니다. "
                           "미국의 그날 장은 한국의 다음 날 개장 전에 끝납니다. 그래서 이 규칙은 실제로는 다음 날 개장 직전에 결정하는 셈이고, 체결은 똑같이 다음 날 시가입니다. "
                           + "이 규칙은 평가 구간을 학습 모델과 맞춘 지금의 엔진으로 낸 결과 파일이 없어서, 신호만 확인했습니다."),
               lambda o_: {src.line(SOX, "past = s.shift(1)"): "한 칸 밀어 당일 값을 뺌 → 분위수에는 t−1 까지의 값만",
                           src.line(SOX, "thr = past.rolling(w, min_periods=w // 2).quantile(q)"): f"과거 {o_['params']['window']}일의 {o_['params']['q']:g} 분위수 (처음엔 {o_['params']['window'] // 2}일만 차도 계산)",
                           src.line(SOX, "pos = (~(s > thr)).astype(float)"): "임계를 넘으면 0(현금), 아니면 1(보유)"},
               hint=sox_hint)
    # ================================================================ 8. 정리
    n_ok = sum(c_["ok"] for c_ in checks_all) + 0
    n_all = len(checks_all)
    pb_, pd2 = perf["bootstrap_p_excess"], perf["deflated_sharpe_p"]
    ex_pp = f"{perf['excess_total_return'] * 100:+.2f}%p"          # 총수익의 차 → 퍼센트포인트
    total_steps = len(steps) + 1
    n_kind = {k_: sum(1 for c_ in checks_all if c_["kind"] == k_) for k_ in KIND}
    assert sum(n_kind.values()) == n_all
    kind_txt = " · ".join(f"{KIND[k_]} {n_kind[k_]}" for k_ in KIND)
    S("8. 정리",
      f"정리: 규칙 모델은 학습 없이 포지션을 내고, 학습 모델과 같은 평가 구간에서 다음 날 시가 체결·비용을 넣어 자산곡선을 만듭니다. 성적표·부트스트랩·Deflated Sharpe 는 모든 매매 모드 모델이 같은 함수로 계산합니다. "
      f"이 예(MA 교차): 샤프 {perf['sharpe']:.3f}, B&H 대비 초과 총수익 {ex_pp}(총수익 {pct(perf['total_return'])} − B&H {pct(perf['benchmark_total_return'])}). "
      f"부트스트랩 p {pb_:.3f} — " + ("'B&H 보다 낫다(평균 초과수익 > 0)' 고 볼 근거 없음. " if pb_ > 0.05 else "0.05 미만. ")
      + f"Deflated Sharpe p {pd2:.3f} — " + (f"이 샤프가 {N}번 시도에서 운으로 기대되는 수준(SR₀)보다 높다고 볼 근거 없음(B&H 와 비교하는 검정이 아님). " if pd2 > 0.05 else "0.05 미만. ")
      + ("<b>규칙 모델은 DM 이 계산되지 않아(예측값이 없음) 사전 등록 성공 기준(DM·봉인 구간 방향·Deflated Sharpe 모두 통과, DECISIONS 2026-11-03)을 처음부터 다 채울 수 없고, "
         "Deflated Sharpe p 도 0.05 보다 커서 기준에 못 미칩니다</b>. " if pd2 > 0.05 else
         "<b>규칙 모델은 DM 이 계산되지 않아(예측값이 없음) 사전 등록 성공 기준(DM·봉인 구간 방향·Deflated Sharpe 모두 통과, DECISIONS 2026-11-03)을 다 채울 수 없습니다</b>. ")
      + f"가정: 현금 수익 0(이 실행은 평가일의 {cash:.1%} 가 현금), 분배금을 뺀 가격 수익률. docs/STATUS.md 의 결론도 '통계적으로 유의한 매매 신호 없음' 입니다. "
      + f"검증 {n_all}줄 — {kind_txt}, <b>{n_ok}줄 통과</b>(종류: 독립 대조 = 엔진의 중간 배열·함수 없이 원자료·다른 결과 파일·다른 계산법으로 다시 구함(식은 엔진과 같은 정의), "
        "식·구조 확인 = 엔진이 쓴 배열에 같은 식을 다른 코드로 다시 계산하거나 날짜 수 같은 구조 조건, 재실행·결과 파일 대조 = 같은 알고리즘·같은 시드로 다시 돌림, "
        "참고(결론 비교) = 계산법이 다른 값과 결론(0.05 와의 대소)만 비교 — 값의 일치 판정이 아님).",
      say=("정리합니다. "
           f"이 실행에서 이동평균 교차의 초과 총수익은 바이 앤 홀드 대비 {perf['excess_total_return'] * 100:.2f} 퍼센트포인트이고, 샤프는 {perf['sharpe']:.3f}입니다. "
           + (f"부트스트랩 p {pb_:.3f}는 전략이 바이 앤 홀드보다 낫다고 볼 근거가 없다는 뜻입니다. " if pb_ > 0.05 else f"부트스트랩 p는 {pb_:.3f}입니다. ")
           + (f"디플레이티드 샤프로 보아도, 이 샤프가 {N}번의 시도에서 운으로 기대되는 수준보다 높다고 볼 근거가 없습니다. 그 p값은 {pd2:.3f}입니다. "
              "사전 등록 성공 기준은, 규칙 모델에 DM 검정이 없어 처음부터 다 채울 수 없고, 디플레이티드 샤프 기준에도 못 미칩니다. "
              if pd2 > 0.05 else f"디플레이티드 샤프 p값은 {pd2:.3f}입니다. 규칙 모델은 DM 검정이 없어 사전 등록 성공 기준을 다 채울 수 없습니다. ")
           + "가정은 분배금을 뺀 가격 수익률, 그리고 현금으로 있는 날의 수익 0입니다. "
           + (f"이 영상의 검증 {n_all}줄은 모두 통과했고, " if n_ok == n_all else f"이 영상의 검증 {n_all}줄 가운데 {n_ok}줄을 통과했고, ")
           + f"그중 {n_kind['독립']}줄은 원자료나 다른 결과 파일, 다른 계산법으로 따로 맞춘 독립 대조"
           + (f"이고, 다른 {n_kind['참고']}줄은 값이 아니라 결론만 비교한 참고입니다." if n_kind["참고"] else "입니다.")),
      stack=["요약"], nodes=ALL,
      tensors=[{"type": "table", "title": f"검증 전체 ({kind_txt})", "columns": ["화면", "종류", "무엇", "결과"],
                "rows": [[f"{steps[c_['step'] - 1]['chapter']} ({c_['step']}/{total_steps})", KIND[c_["kind"]], c_["what"], ("✓ 결론 같음" if c_["kind"] == "참고" else "✓ 일치") if c_["ok"] else "✗ 다름"] for c_ in checks_all]}])
    if n_ok != len(checks_all):
        raise AssertionError(f"검증 실패: {[c_ for c_ in checks_all if not c_['ok']]}")
    bad_lab = [s_["lab"] for s_ in steps if s_.get("lab") and not s_["lab"]["ok"]]
    if bad_lab:
        raise AssertionError(f"직접 실행해 보기 실패: {[(l_['title'], l_['got'], l_['err']) for l_ in bad_lab]}")

    def clean(o_):
        if isinstance(o_, dict): return {k_: clean(v_) for k_, v_ in o_.items()}
        if isinstance(o_, (list, tuple)): return [clean(v_) for v_ in o_]
        if isinstance(o_, np.bool_): return bool(o_)
        if isinstance(o_, np.integer): return int(o_)
        if isinstance(o_, (np.floating, float)): return None if not np.isfinite(o_) else float(o_)
        return o_
    steps[:] = clean(steps)
    used = sorted({s_["code"]["file"] for s_ in steps if s_.get("code")})
    return {"title": "규칙 모델 · 백테스트 코드 따라가기 — 실제로 계산되는 순서", "model": "rule", "steps": steps, "sources": {fn_: src.text[fn_] for fn_ in used}, "diagram": diagram(), "checks": clean(checks_all)}


def diagram() -> dict:
    labels = [("sig", "신호", "규칙 positions"), ("win", "평가 구간", "학습 모델과 같게"), ("fill", "체결·비용", "다음 날 시가"), ("eq", "자산곡선", "누적 곱"), ("perf", "성적표", "performance"),
              ("test", "검정", "부트스트랩·DSR"), ("oth", "다른 규칙", "B&H·모멘텀·RSI·SOX")]
    nodes = [{"id": i, "label": l, "x": 10 + 268 * k_, "y": 14, "w": 240, "h": 52, "sub": s} for k_, (i, l, s) in enumerate(labels)]
    E = [{"from": a[0], "to": b_[0]} for a, b_ in zip(labels[:-1], labels[1:])]
    return {"width": 10 + 268 * len(labels), "height": 84, "nodes": nodes, "edges": E, "groups": []}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture"); ap.add_argument("--out")
    a = ap.parse_args()
    cap = pickle.load(open(a.capture or ROOT / "docs/guide_build/walk/rule_capture.pkl", "rb"))
    sc = build(cap)
    out = build_html(sc, Path(a.out) if a.out else ROOT / "docs/guide_build/walk/rule.html")
    print("저장:", out, "· 단계", len(sc["steps"]), "· 검증", sum(c["ok"] for c in sc["checks"]), "/", len(sc["checks"]), "· 크기", round(out.stat().st_size / 1e6, 2), "MB")
