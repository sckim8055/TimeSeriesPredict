"""표 입력 모델(Ridge·LightGBM) 코드 따라가기 대본 — table_capture 의 실제 실행 기록으로 단계별 화면을 만든다.

  python -m tools.codewalk.table_script --model ridge
  python -m tools.codewalk.table_script --model lgbm

데이터셋 만들기(대상·정답·기술지표·외생변수의 변환·발표시차·달력 정렬) → 폴드 → 눈금 → 모델 학습 → 예측 → 모든 폴드.
각 단계의 숫자를 식대로 numpy 로 다시 계산해 엔진·라이브러리 값과 맞는지 '검증' 줄로 보인다. 하나라도 다르면 대본을 만들지 않는다.
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tools.codewalk.common import run_lab, py_list, ROOT, Sources, arr, build_html, f, match_result, num

EXP = "engine/experiment.py"; CAL = "engine/calendar.py"; FEAT = "engine/features.py"; TC = "tools/codewalk/table_capture.py"
F32 = 2e-5   # LightGBM 은 기울기(잔차)를 32비트로 더한다 → 64비트 실습 코드와 5~6번째 자리가 다를 수 있어 상대오차 2e-5 까지 같은 값으로 본다(biva_script 와 같음)


def g5(v) -> str:
    """실습 출력용 유효숫자 5자리."""
    return f"{float(v):.5g}"


# ---------------------------------------------------------------- LightGBM 직접 실행해 보기 (2026-11-08) — 기대 출력은 모두 엔진·LightGBM·결과 파일 값
def lab_leaf_gain(m_, node, yL, FL, yR, FR, lr, lam, feat):
    """나무 m_+1 의 마디 하나: 두 자식 잎 값과 분할 이득을 잔차로 직접 계산 (두 잎 모두 학습 행이 다 뽑힌 마디)."""
    L_, R_ = node["left_child"], node["right_child"]
    code = (f"# 나무 {m_ + 1} 의 마디 '{feat} ≤ {node['threshold']:+.3f}' — 두 자식 잎에 든 학습 행 (실행 기록 그대로)\n"
            f"# y = 정답,  F = 앞 {m_}그루까지의 예측 (LightGBM predict 기록)\n"
            f"yL = {py_list(yL)}\nFL = {py_list(FL)}\n"
            f"yR = {py_list(yR)}\nFR = {py_list(FR)}\n"
            f"lr = {float(lr)!r}      # 학습률\n"
            f"lam = {float(lam)!r}     # λ (reg_lambda)\n\n"
            "GL = 0.0\n"
            "for y, F in zip(yL, FL):\n"
            "    GL += y - F                 # 왼쪽 잎의 잔차 합\n"
            "GR = 0.0\n"
            "for y, F in zip(yR, FR):\n"
            "    GR += y - F                 # 오른쪽 잎의 잔차 합\n"
            "nL, nR = len(yL), len(yR)\n\n"
            "leaf_L = lr * GL / (nL + lam)   # 잎 값 = 학습률 × 잔차 합 ÷ (행 수 + λ)\n"
            "leaf_R = lr * GR / (nR + lam)\n"
            "gain = GL ** 2 / (nL + lam) + GR ** 2 / (nR + lam) - (GL + GR) ** 2 / (nL + nR + lam)\n\n"
            "print(\"왼쪽 잎 값\", f\"{leaf_L:.5g}\")\n"
            "print(\"오른쪽 잎 값\", f\"{leaf_R:.5g}\")\n"
            "print(\"분할 이득\", f\"{gain:.5g}\")\n")
    exp = f"왼쪽 잎 값 {g5(L_['leaf_value'])}\n오른쪽 잎 값 {g5(R_['leaf_value'])}\n분할 이득 {g5(node['split_gain'])}"
    return {"title": f"나무 {m_ + 1} 의 잎 값 2개와 분할 이득을 직접 계산",
            "desc": f"두 자식 잎에 든 학습 행({len(yL)}행·{len(yR)}행)이 모두 이 나무에 뽑힌 마디라 손으로 계산할 수 있습니다. 기대 출력은 LightGBM 이 저장한 잎 값과 분할 이득입니다(32비트 계산이라 끝자리 차이는 허용).",
            **run_lab(code, exp, pure=True, rtol=F32)}


def lab_tree_sum(trees, z, date, pred_e):
    """결정일 z 로 나무를 모두 따라 내려가 잎 값을 더한다 — 엔진 예측과 같아야 함."""
    def enc(n):
        if "leaf_value" in n:
            return repr(float(n["leaf_value"]))
        return f"[{n['split_feature']}, {float(n['threshold'])!r}, {enc(n['left_child'])}, {enc(n['right_child'])}]"
    code = (f"# 결정일 {date} 의 표준화된 재료 z ({len(z)}개, 실행 기록 그대로)\n"
            f"z = {py_list(z)}\n\n"
            f"# 나무 {len(trees)}그루 (LightGBM dump 그대로). 마디 = [열 번호, 기준, 왼쪽(z ≤ 기준), 오른쪽], 잎 = 숫자\n"
            "trees = [\n" + "".join(f"    {enc(t_['tree_structure'])},\n" for t_ in trees) + "]\n\n"
            "pred = 0.0\n"
            "for node in trees:\n"
            "    while isinstance(node, list):          # 잎(숫자)에 닿을 때까지 내려감\n"
            "        j, thr, left, right = node\n"
            "        node = left if z[j] <= thr else right\n"
            "    pred += node                           # 도착한 잎 값을 더함\n\n"
            "print(\"예측(%)\", f\"{pred * 100:.6g}\")\n")
    return {"title": f"나무 {len(trees)}그루를 따라 내려가 예측을 직접 계산",
            "desc": "나무마다 z 와 기준을 비교해 잎까지 내려가고, 도착한 잎 값을 모두 더합니다(첫 나무의 잎 값에 시작값 포함). 기대 출력은 엔진 예측(model.predict)입니다. 반복이 Python Tutor 의 단계 한도(1,000)를 넘어 복사해서 실행하도록 했습니다.",
            **run_lab(code, f"예측(%) {float(pred_e) * 100:.6g}", pure=True, tutor=False)}


def lab_imp_mean(names, cols, res_vals, n_folds):
    """결과 화면의 변수 중요도 = 폴드별 gain 의 평균 (결과 파일 feature_importance 와 비교)."""
    code = (f"# 폴드 {n_folds}개의 gain 중요도 — 결과 파일 중요도 상위 {len(names)}개 열만 (실행 기록 그대로)\n"
            "gain = {\n" + "".join(f"    {n_!r}: {py_list(c_)},\n" for n_, c_ in zip(names, cols)) + "}\n\n"
            "for name, g in gain.items():\n"
            "    mean = sum(g) / len(g)          # 폴드 평균 = 결과 화면의 변수 중요도\n"
            "    print(name, f\"{mean:.6g}\")\n")
    exp = "\n".join(f"{n_} {float(v_):.6g}" for n_, v_ in zip(names, res_vals))
    return {"title": "결과 화면의 변수 중요도를 폴드 평균으로 직접 계산", "kind": "재실행",
            "desc": f"폴드마다 다시 학습한 모델의 gain 중요도를 {n_folds}개 폴드에 걸쳐 평균합니다. 기대 출력은 결과 파일(feature_importance)에 저장된 엔진 값입니다.",
            **run_lab(code, exp, pure=True)}


def lab_trade(pp, yy, thr, exposure_e, rmse_e, dir_e):
    """폴드 예측으로 보유 비율·RMSE·방향 적중 — 결과 파일의 exposure·rmse·dir_acc 와 비교."""
    code = (f"# 폴드 {len(pp)}개: 예측과 실제 수익률 (실행 기록 그대로)\n"
            f"pred = {py_list(pp)}\n"
            f"actual = {py_list(yy)}\n"
            f"threshold = {float(thr)!r}\n\n"
            "n = len(pred)\n"
            "held = 0\n"
            "sq = 0.0\n"
            "hit = 0\n"
            "for p, a in zip(pred, actual):\n"
            "    if p > threshold:              # 매매 규칙: 예측 > 문턱이면 다음 달 보유\n"
            "        held += 1\n"
            "    sq += (p - a) ** 2\n"
            "    if (p > 0) == (a > 0):         # 방향(오름/내림)이 맞았나\n"
            "        hit += 1\n\n"
            "print(\"보유 비율\", f\"{held / n:.5g}\")\n"
            "print(\"RMSE\", f\"{(sq / n) ** 0.5:.5g}\")\n"
            "print(\"방향 적중\", f\"{hit / n:.5g}\")\n")
    exp = f"보유 비율 {g5(exposure_e)}\nRMSE {g5(rmse_e)}\n방향 적중 {g5(dir_e)}"
    return {"title": "보유 비율·RMSE·방향 적중을 직접 계산", "kind": "재실행",
            "desc": "폴드마다의 예측과 실제 수익률로 세 값을 계산합니다. 기대 출력은 결과 파일의 exposure·rmse·dir_acc(엔진 값)입니다.",
            **run_lab(code, exp, pure=True)}


def pct(v, d=2, sign=True):
    return "NaN" if v is None or not np.isfinite(v) else (f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%")


def g(v, d=4):
    return "NaN" if v is None or not np.isfinite(v) else f"{v:.{d}g}"


# ---------------------------------------------------------------- 기술지표 손 계산 (pandas 없이 반복문 — engine/features.py 의 식 그대로)
def ema(x, alpha):
    out = np.full(len(x), np.nan); m = None
    for i, v in enumerate(x):
        if np.isnan(v):
            out[i] = m if m is not None else np.nan; continue
        m = v if m is None else (1 - alpha) * m + alpha * v
        out[i] = m
    return out


def tech_hand(close, volume, P, ppy, t_idx):
    """결정일 위치 t_idx 의 기술지표 값을 반복문으로 계산."""
    c = np.asarray(close, dtype=float); i = t_idx; out = {}
    for l in P["lags"]:
        out[f"A_ret_{l}"] = c[i] / c[i - l] - 1
    for w in P["ma"]:
        out[f"A_ma_ratio_{w}"] = c[i] / np.mean(c[i - w + 1:i + 1]) - 1
    lr = np.diff(np.log(c))      # lr[j-1] = ln c_j − ln c_(j−1)
    for w in P["vol"]:
        out[f"A_vol_{w}"] = np.std(lr[i - w:i], ddof=1) * np.sqrt(ppy)
    d = np.r_[np.nan, np.diff(c)]
    up = ema(np.where(np.isnan(d), np.nan, np.maximum(d, 0)), 1 / P["rsi"]); dn = ema(np.where(np.isnan(d), np.nan, np.maximum(-d, 0)), 1 / P["rsi"])
    out[f"A_rsi_{P['rsi']}"] = 100 - 100 / (1 + up[i] / dn[i])
    fa, sl, sg = P["macd"]
    ef = ema(c, 2 / (fa + 1)); es = ema(c, 2 / (sl + 1)); m = ef - es; s = ema(m, 2 / (sg + 1))
    out["A_macd"], out["A_macd_signal"], out["A_macd_hist"] = m[i] / c[i], s[i] / c[i], (m[i] - s[i]) / c[i]
    if volume is not None:
        v = np.asarray(volume, dtype=float)
        for w in P["volume"]:
            out[f"A_vol_ratio_{w}"] = v[i] / np.mean(v[i - w + 1:i + 1]) - 1
    return out


def build(cap: dict) -> dict:
    model = cap["model"]; is_ridge = model == "ridge"; KN = "Ridge" if is_ridge else "LightGBM"
    MODEL = "models/ridge.py" if is_ridge else "models/lightgbm_model.py"
    src = Sources([MODEL, EXP, CAL, FEAT, "models/base.py"] + ([] if is_ridge else [TC]))
    cfg = cap["config"]; p = cap["params"]; mode = cap["mode"]; k = cap["k"]; emb = cap["embargo"]
    feats = cap["feats"]; desc = cap["desc"]; K = len(feats)
    tech = [c for c in feats if c.startswith("A_")]; exo = [c for c in feats if not c.startswith("A_")]
    prices = cap["prices"]; X = cap["X"]; y = cap["y"]; tr = cap["tr"]; te = cap["te_valid"]
    t = te.index[-1]; ti = list(prices.index).index(t)
    freq = cfg["frequency"] if mode == "trading" else "M"
    ppy = {"D": 252, "W": 52, "M": 12}[freq]
    steps: list[dict] = []; checks_all: list[dict] = []

    KIND = {"독립": "독립 대조", "식": "식·구조 확인", "재실행": "재실행·결과 파일 대조"}   # 검증 줄의 종류(2026-11-08 공통 규칙 1)

    def chk(what, a, b_, tol=1e-10, rel=False, kind="독립"):
        """kind: 독립 = 다른 경로·다른 계산법으로 대조 / 식 = 같은 배열에 같은 식·구조 조건 확인 / 재실행 = 다시 돌린 기록·LightGBM 재구성·결과 파일과 대조."""
        a = np.asarray(a, dtype=float); b_ = np.asarray(b_, dtype=float)
        d = float(np.nanmax(np.abs(a - b_) / (np.maximum(np.abs(b_), 1e-300) if rel else 1.0))) if a.size else 0.0
        same_nan = bool(np.all(np.isnan(a) == np.isnan(b_)))
        c = {"what": f"{what} ({'상대 ' if rel else ''}차이 {d:.1e})", "ok": bool(d <= tol and same_nan), "kind": kind}
        checks_all.append({**c, "step": len(steps) + 1}); return c

    def S(chapter, narr, *, code=None, stack=None, nodes=(), edges=(), tensors=(), formula=None, hint=None, checks=(), say=None, notes=None, lab=None):
        """say = 음성·자막 대본(화면 숫자 그대로, 없으면 narr), notes = {원본 줄 번호: 그 줄 설명}, lab = 직접 실행해 보기(run_lab 결과 + 제목)."""
        checks = list(checks)
        if lab is not None:
            checks.append(chk_lab(lab))
        if code is not None and notes:
            code = {**code, "notes": {str(k_): v_ for k_, v_ in notes.items()}}
        steps.append({"chapter": chapter, "narr": narr, "code": code, "stack": list(stack or []), "nodes": list(nodes), "edges": [list(e) for e in edges],
                      "tensors": list(tensors), "formula": formula, "hint": hint, "checks": checks, "say": say, "lab": lab})

    def skip(what, reason, kind="독립"):
        """검증을 할 수 없을 때(파일·비공개 API 가 없는 환경) — 줄은 남기고(검증 줄 수가 환경에 따라 바뀌지 않게) '건너뜀' 으로 표시. 음성은 바뀌지 않는다."""
        c = {"what": f"건너뜀 — {reason}: {what}", "ok": None, "skipped": True, "kind": kind}
        checks_all.append({**c, "step": len(steps) + 1}); return c

    def chk_lab(lab):
        kd = lab.get("kind", "독립")
        c = {"what": f"직접 실행해 보기 '{lab['title']}' 출력 = {'결과 파일 값' if kd == '재실행' else '엔진 값'}", "ok": bool(lab["ok"]), "kind": kd}
        checks_all.append({**c, "step": len(steps) + 1}); return c

    def C(fn, lines, focus=None, func="", before=6, after=8):
        lines = sorted(set(lines)); return {"file": fn, "lines": lines, "focus": focus or lines[0], "func": func, "before": before, "after": after}

    def Hm(title, data, rows=None, cols=None, note=None, digits=None, numbers=True, **kw):
        t_ = {"type": "heat", "title": title, "data": data if isinstance(data, list) else arr(data), "rows": rows, "cols": cols, "note": note, "scale": "div", "hl": {}, "numbers": numbers,
              **{k_: v_ for k_, v_ in kw.items() if v_ is not None}}
        if digits is not None: t_["digits"] = digits
        return t_

    def nm(c, j=None):
        """음성·자막용 재료 이름: 'A_rsi_6' → '6개월 RSI(A_rsi_6)' — 괄호 안 코드 이름은 화면에만 보이고 읽지 않는다(speech.speakify).
        j: 붙일 조사(은/는/이/가/…) — 괄호 앞 한국어 이름의 받침에 맞춘다."""
        lab_ = _nm(c)
        if not j:
            return lab_
        from tools.codewalk.speech import attach
        base = lab_.split("(")[0]
        return lab_ + attach(base, j)[len(base):]

    def _re_name(nm_):
        import re as _re4
        return _re4.sub(r"\([^)]*\)", "", nm_).strip()

    def nms(c, j=None):
        """음성용 재료 이름(LightGBM 화면 전용 — 릿지 음성은 nm 그대로): 외부 변수는 사전의 한국어 이름(단위가 천 명이면 '… 수'),
        변환 pct → 변화율, diff → 변화. 'SK하이닉스' 의 영문 SK 는 음성 용어(speech.TERMS)에 없어 한글로 적는다."""
        from tools.codewalk.speech import attach
        for vid, v in cap["variables"].items():
            if c == (vid if v["transform"] in ("", "none") else f"{vid}_{v['transform']}"):
                import re as _re3
                base = (_re3.sub(r"\([^)]*\)", "", v["name"]).strip() + (" 수" if str(v.get("unit") or "").strip() == "천명" else "")
                        + (" 변화율" if v["transform"] == "pct" else " 변화" if v["transform"] == "diff" else "")).replace("SK하이닉스", "에스케이하이닉스")
                return f"{base}({c})" + (attach(base, j)[len(base):] if j else "")
        return nm(c, j).replace("SK하이닉스", "에스케이하이닉스")

    def _nm(c):
        import re as _re
        unit = {"M": "개월", "W": "주", "D": "일"}[freq]
        rules = [(r"A_ret_(\d+)$", "{0}%s 수익률" % unit), (r"A_ma_ratio_(\d+)$", "{0}%s 이동평균 비율" % unit), (r"A_vol_ratio_(\d+)$", "{0}%s 거래량 비율" % unit),
                 (r"A_vol_(\d+)$", "{0}%s 변동성" % unit), (r"A_rsi_(\d+)$", "{0}%s RSI" % unit), (r"A_macd$", "MACD"), (r"A_macd_signal$", "MACD 신호선"), (r"A_macd_hist$", "MACD 히스토그램")]
        for pat, fmt in rules:
            m_ = _re.match(pat, c)
            if m_:
                return f"{fmt.format(*m_.groups())}({c})"
        for vid, v in cap["variables"].items():
            if c == vid or c.startswith(vid + "_"):
                tr_ = v["transform"]
                return f"{v['name']}{' 변화율' if tr_ == 'pct' else ' 차이' if tr_ == 'diff' else ''}({c})"
        return c

    ALL = ["tgt", "tech", "exo", "ds", "fold", "scale", "fit", "pred", "allf"]
    stk_ds = ["engine/experiment.py · prepare", "engine/experiment.py · build_dataset"]
    stk_wf = ["engine/experiment.py · _walkforward_predict (표 분기)"]
    target_name = cfg["target_variable"] if mode == "forecast" else f"{cfg['ticker']} (KODEX 반도체)"
    tk_say = "KODEX 반도체" if mode == "trading" and cfg.get("ticker") == "091160" else cfg.get("ticker")   # 음성: 종목 번호 091160 을 '구만…' 으로 읽지 않게
    res = None   # 결과 파일(설정을 결과 파일에서 가져온 경우) — 결과 화면의 숫자와 맞춰 보기 위해. 대본(음성)이 이 값을 쓰므로 없으면 멈춘다(빌드 환경에 따라 음성이 달라지지 않게)
    if str(cap["config_file"]).startswith("results/"):
        if not (ROOT / cap["config_file"]).exists():
            raise FileNotFoundError(f"결과 파일 {cap['config_file']} 이 없습니다 — 대본(음성·자막)이 이 파일의 값(DM·중요도 등)을 씁니다")
        import json as _json
        res = _json.loads((ROOT / cap["config_file"]).read_text(encoding="utf-8"))
    af = cap["all_folds"]
    tl = cfg["validation"]["train_len"]
    emb_w = {1: "한 달", 2: "두 달", 3: "석 달"}.get(emb, f"{emb}개월")   # 음성: '이개월' 대신 '두 달'
    assert len(cap["fold"]["train"]) == tl - emb, "학습 창 − 비움 ≠ 마지막 폴드 학습 칸 수"   # 엔진(walkforward_folds)과 맞는지
    # 발표 뒤 값이 바뀌는 변수(엑셀 값 = 최신 vintage)와 lag 미확인 변수 — 변수 사전(variable_master.xlsx) note 로 확인 (DECISIONS 2026-11-05)
    VINT_WHY = {"B1": "순환변동치 — 새 달이 추가될 때마다 과거 전체 재계산", "B2": "순환변동치 — 새 달이 추가될 때마다 과거 전체 재계산",
                "K1": "계절조정 계열 — 매월 재추정", "K2": "계절조정 계열 — 매월 재추정", "F1": "잠정치 → 다음 달 확정치로 수정"}
    VINT_KEY = {"B1": "vintage", "B2": "vintage", "K1": "재추정", "K2": "재추정", "F1": "확정치"}
    mfile = ROOT / "data" / "variable_master.xlsx"
    if not mfile.exists():   # 대본(음성)의 '최신 수정값·공개 시각 미확인' 문장이 변수 사전을 쓰므로, 없으면 멈춘다(빌드 환경에 따라 음성이 달라지지 않게)
        raise FileNotFoundError(f"{mfile} 이 없습니다 — 대본(음성·자막)이 변수 사전의 note·lag_confirmed·duplicate_of 를 씁니다")
    mst = pd.read_excel(mfile, "variables").set_index("id")
    vint = [v_ for v_ in cap["variables"] if v_ in VINT_WHY] if mst is not None else []
    for v_ in vint:
        assert VINT_KEY[v_] in str(mst.loc[v_, "note"]), f"변수 사전 note 가 바뀜: {v_}"
    unconf = [v_ for v_ in cap["variables"] if mst is not None and str(mst.loc[v_, "lag_confirmed"]).strip().upper() != "Y"]
    # lag_confirmed = Y 라도 note 의 조건(예: '재무부 원천 기준 … FRED 를 실시간 채널로 쓰면 다음 날 게시')과 실제 출처가 다르면 공개 시각 미확인
    t_unconf = [v_ for v_ in cap["variables"] if mst is not None and v_ not in unconf and "조건:" in str(mst.loc[v_, "note"])
                and str(mst.loc[v_, "source"]).strip() in str(mst.loc[v_, "note"]).split("조건:", 1)[1] and "실시간 채널로 쓰면" in str(mst.loc[v_, "note"])]
    from tools.codewalk.speech import attach as _att
    KO = {"USDKRW": "원 달러 환율", "US10Y": "미 국채 금리"}
    tu_say = "과 ".join(KO.get(v_, v_) for v_ in unconf + t_unconf)
    vint_name = _re_name(cap["variables"][vint[0]]["name"]) if len(vint) == 1 else ""   # 음성: 변수 하나면 이름으로
    tu_verb = "체결 전에 공개되는지 확인되지 않았습니다" if mode == "trading" else "공개 시각이 확인되지 않았습니다"   # 매매: 체결 시점 기준 / 예측: 공개 시각 기준
    col_of = {vid: (vid if v["transform"] in ("", "none") else f"{vid}_{v['transform']}") for vid, v in cap["variables"].items()}

    def vlabel(c):
        """표·음성용 외부 변수 이름: 단위가 천 명이면 '… 수(천 명)', 변환은 변화율/변화."""
        for vid, v in cap["variables"].items():
            if c == col_of[vid]:
                u = str(v.get("unit") or "").strip()
                nm_ = v["name"] + (" 수(천 명)" if u == "천명" else "")
                return nm_ + (" · 변화율" if v["transform"] == "pct" else " · 변화(전 관측 대비 차이)" if v["transform"] == "diff" else "")
        return desc[c]

    def dsc(c):
        """표의 '뜻' 칸: 기술지표는 식 전체, 외부 변수는 사전의 한국어 이름·단위(영문 설명은 쓰지 않음)."""
        return desc[c] if c.startswith("A_") else vlabel(c)
    # ================================================================ 0. 시작
    S("0. 시작",
      f"이 영상은 <b>{KN}</b> {'가' if is_ridge else '이'} 실제로 계산되는 순서를 따라갑니다. 설정 <b>{Path(cap['config_file']).name}</b>: "
      + (f"<b>예측 연구 모드</b> — 대상 {cfg['target_variable']}(월간)의 다음 달 변화율을 맞히는 문제입니다. " if mode == "forecast" else
         f"<b>매매 모드</b> — {cfg['ticker']} 를 월말까지의 값(미국 마감 값 포함)으로 다음 거래일 개장 직전에 판단해 다음 달 첫 거래일 시가에 사고 그다음 달 첫 거래일 시가에 파는 수익률을 맞히는 문제입니다. ")
      + f"재료(입력 열)는 {K}개: 대상 자신의 {'가격' if mode == 'forecast' else '가격·거래량'}으로 만든 기술지표 {len(tech)}개 + 외부 변수 {len(exo)}개. 워크포워드 폴드 {cap['n_folds']}개마다 다시 학습하며, "
      f"화면의 숫자는 마지막 폴드(결정일 {t.date()})의 실제 실행 기록입니다.",
      say=(f"이 영상은 {KN} 모델이 실제로 계산되는 순서를 처음부터 끝까지 따라갑니다. "
           + (f"문제는 {cfg['target_variable']}의 다음 달 변화율을 맞히는 것입니다. " if mode == "forecast" else
              f"문제는 {tk_say}의 다음 달 수익률을 맞혀, 다음 달에 들고 있을지 정하는 것입니다. ")
           + f"입력 재료는 {K}개로, 대상 자신의 과거 {'값' if mode == 'forecast' else '가격과 거래량'}으로 만든 기술지표 {len(tech)}개와 외부 경제 변수 {len(exo)}개입니다. "
           f"모델은 워크포워드 방식으로 {cap['n_folds']}번 다시 학습하고, 화면의 숫자는 그중 마지막 폴드, 결정일 {t.date()}의 실제 실행 기록입니다. "
           "오른쪽 코드에서 노란 줄이 지금 실행되는 부분이고, 줄 아래 회색 글은 그 줄의 설명입니다."),
      code=C(MODEL, [1], 1, "모듈 전체", 0, 40),
      stack=[MODEL], nodes=ALL,
      tensors=[{"type": "table", "title": f"실행 설정 ({cap['config_file']} + {MODEL} 기본값)", "columns": ["항목", "값"], "rows": [
          ["모드", mode], ["대상", target_name], ["정답", f"{'다음 달 변화율' if mode == 'forecast' else '다음 달 시가→그다음 달 시가 수익률'} (k = {k})"], ["주기", freq],
          ["재료", f"기술지표 {len(tech)} + 외부 변수 {len(exo)} = {K}열"], ["워크포워드", f"학습 창 {tl}개월(끝 {emb}개월 비움 → 학습 {tl - emb}개월) · 테스트 1개월 · 폴드 {cap['n_folds']}개"],
          *[[f"모델: {k_}", str(v)] for k_, v in p.items()]]}],
      hint="숫자의 출처: tools/codewalk/table_capture.py 가 엔진의 실제 경로(prepare → build_dataset → make_folds → _walkforward_predict → " + MODEL + ")로 실행하며 중간 객체를 기록했습니다. 계산 자체는 바꾸지 않습니다."
           + (f" 설정은 결과 파일 {cap['config_file']} 에 기록된 것 그대로입니다(결과 화면의 숫자와 맞춰 보기 위해). configs/ 의 같은 이름 설정 파일은 그 뒤 모델 파라미터가 바뀌어 아직 결과가 없습니다." if cap["config_file"].startswith("results/") else ""))
    S("0. 시작",
      ("호출 순서: <b>prepare</b> 가 시드를 고정하고 " if is_ridge else
       f"호출 순서: <b>prepare</b> 가 엔진 시드를 고정하고(cfg.seed = {cfg['seed']}: random·numpy·torch — 폴드를 학습하기 직전에는 이 시드와 그 폴드의 첫 테스트 결정일로 폴드 시드를 다시 정해 난수를 맞추고, LightGBM 은 그 numpy 난수에서 행·열 뽑기 시드(random_state)를 뽑습니다) ")
      + "모델의 실제 파라미터(주기별 기본값 채움)를 정한 뒤 <b>build_dataset</b> 으로 결정일 × 재료 표 X 와 정답 y 를 만듭니다. "
      "<b>make_folds</b> 가 워크포워드 폴드를 만들고, <b>_walkforward_predict</b> 가 폴드마다 학습 표를 자르고 눈금을 맞춘 뒤 <b>m.fit</b> → <b>m.predict</b> 합니다.",
      code=C(EXP, src.lines(EXP, 'tr = ds.X.loc[f.train].join(ds.y.loc[f.train].rename("y")).dropna()', "if len(tr) < 30 or te_empty:",
                            'scaler = (MinMaxScaler() if params.get("input_scaler") == "minmax" else StandardScaler()).fit(fit_rows)',
                            "Xtr = scaler.transform(tr[feat].values)", "Xte = scaler.transform(te_valid[feat].values)",
                            "m, n_tr, n_va, p, fit_p, imp_f, fd = _fit_reps(f, Xfit_in, yfit_in, n_val_t, Xte, Xtr if want_fit else None, n_feat=len(feat))", "pred.loc[te_valid.index] = p")
             + ([] if is_ridge else [src.line(EXP, "sd = fold_seed(cfg.seed, f.test[0], r)")]),
             src.line(EXP, "pred.loc[te_valid.index] = p"), "_walkforward_predict", 1, 1),
      say=(("코드는 이런 순서로 불립니다. 먼저 시드를 고정하고 모델 설정을 정한 다음, 결정일마다 재료 표와 정답을 만듭니다. "
            "그다음 워크포워드 폴드를 나누고, 폴드마다 학습 표를 잘라 눈금을 맞춘 뒤, 학습하고 예측합니다. 화면의 코드는 그 마지막 부분, 폴드 하나를 처리하는 곳입니다.") if is_ridge else
           (f"코드는 이런 순서로 불립니다. 먼저 엔진 시드를 {cfg['seed']}로 고정하고 모델 설정을 정한 다음, 결정일마다 재료 표와 정답을 만듭니다. "
            "폴드를 학습하기 직전에는 이 시드와 그 폴드의 날짜로 폴드 시드를 다시 정해 난수를 맞추고, 라이트지비엠은 그 난수에서 행과 열 뽑기 시드를 뽑습니다. "
            "그다음 워크포워드 폴드를 나누고, 폴드마다 학습 표를 잘라 눈금을 맞춘 뒤, 학습하고 예측합니다. 화면의 코드는 그 마지막 부분, 폴드 하나를 처리하는 곳입니다.")),
      notes={src.line(EXP, 'tr = ds.X.loc[f.train].join(ds.y.loc[f.train].rename("y")).dropna()'): f"학습 창의 재료 X 와 정답 y 를 옆으로 붙이고, 빈 칸이 있는 행은 뺍니다 → 이번 폴드 {len(tr)}행",
             src.line(EXP, "if len(tr) < 30 or te_empty:"): "학습 행이 30개 미만이거나 테스트 입력이 없으면 이 폴드는 건너뜁니다",
             src.line(EXP, 'scaler = (MinMaxScaler() if params.get("input_scaler") == "minmax" else StandardScaler()).fit(fit_rows)'): f"눈금(평균·표준편차)을 학습 표로만 구합니다 — 재료 {K}개",
             src.line(EXP, "Xtr = scaler.transform(tr[feat].values)"): f"학습 표를 z 로 바꿈 → {len(tr)} × {K}",
             src.line(EXP, "Xte = scaler.transform(te_valid[feat].values)"): f"테스트 행도 학습 표의 평균·표준편차로 바꿈 → {len(te)} × {K}",
             src.line(EXP, "m, n_tr, n_va, p, fit_p, imp_f, fd = _fit_reps(f, Xfit_in, yfit_in, n_val_t, Xte, Xtr if want_fit else None, n_feat=len(feat))"): "모델 학습·예측 (_fit_reps: 검증을 쓰는 모델이면 학습 표 끝을 검증으로 떼어 냄. 난수를 쓰는 모델은 시드 수만큼 반복해 평균)",
             src.line(EXP, "pred.loc[te_valid.index] = p"): f"예측을 결정일 자리에 기록 → {t.date()} 에 {pct(cap['pred'].loc[t], 3)}",
             **({} if is_ridge else {src.line(EXP, "sd = fold_seed(cfg.seed, f.test[0], r)"): f"_fit_reps 안: 실험 시드 {cfg['seed']} 와 이 폴드의 첫 테스트 결정일로 폴드 시드를 정해 random·numpy·torch 를 맞춤(LightGBM 은 그 numpy 난수에서 random_state 를 뽑음)"})},
      stack=stk_wf, nodes=["ds", "fold", "scale", "fit", "pred"])
    # ================================================================ 1. 대상과 정답
    if mode == "forecast":
        lev = prices["close"]
        y_hand = lev.shift(-k) / lev - 1
        i0 = ti
        S("1. 대상과 정답",
          f"대상 {cfg['target_variable']} 의 월별 값(지수)을 '가격' 틀에 넣습니다(시가 = 종가 = 지수 값, 거래량 없음). 결정일 t 의 정답은 <b>t 의 값 대비 {k}개월 뒤 값의 변화율</b>입니다. "
          f"마지막 {k}개월은 미래 값이 없어 정답이 비어 있습니다."
          + (f" 결정일 뒤 행({prices.index[ti + 1].strftime('%Y-%m')}~)은 봉인 평가 대상이라 표에 넣지 않았습니다. 다만 결정일 {t.date()} 의 정답은 {prices.index[ti + k].strftime('%Y-%m')} 값, "
             f"곧 봉인 구간({cfg['validation']['sealed_from']}~)의 값을 쓰고 개발 평가 성과에도 들어갑니다 — 엔진은 결정일만 봉인 전으로 자릅니다."
             if cfg["validation"].get("sealed_from") and ti + k < len(prices) and prices.index[ti + k] >= pd.Timestamp(cfg["validation"]["sealed_from"]) else ""),
          code=C(CAL, src.span(CAL, "def make_forecast_target(", 'y.name = f"forecast_{kind}_k{k}"'), src.line(CAL, "r = v.shift(-k) / v - 1.0"), "make_forecast_target", 0, 2),
          say=(f"대상 {cfg['target_variable']}의 월별 지수 값을 가격 자리에 넣습니다. 결정일 t의 정답은, t의 값에 비해 "
               + ("한 달" if k == 1 else f"{k}개월") + " 뒤의 값이 몇 퍼센트 변했는지입니다. 그래서 맨 마지막 "
               + ("달" if k == 1 else f"{k}개월") + "은 아직 다음 값이 없어 정답이 비어 있습니다."),
          notes={src.line(CAL, "v = level.astype(float)"): "대상의 월별 값(지수)",
                 src.line(CAL, "r = v.shift(-k) / v - 1.0"): f"shift(−{k}): {k}칸 뒤 값을 지금 행으로 당겨 와 나눔 → 다음 달 변화율. 마지막 {k}행은 뒤 값이 없어 NaN"},
          stack=stk_ds + ["engine/calendar.py · make_forecast_target"], nodes=["tgt"],
          formula=f"y_t = v_(t+{k}) ÷ v_t − 1",
          tensors=[{"type": "table", "title": f"결정일 {t.date()} 앞뒤", "columns": ["결정일 t", "v_t", f"v_(t+{k})", "정답 y_t"],
                    "rows": [[d.strftime("%Y-%m"), f"{lev.iloc[j]:.4f}", f"{lev.iloc[j + k]:.4f}" if j + k < len(lev) else "—", pct(y.iloc[j], 3)] for j, d in ((j, prices.index[j]) for j in range(i0 - 3, i0 + 1))]}],
          checks=[chk("식 확인: 정답 = 같은 월별 값에 같은 식(shift)", y_hand.values, y.values, 1e-15, kind="식")])
    else:
        dly = cap["daily"]
        mo = dly.loc[(dly.index > prices.index[ti - 1]) & (dly.index <= t)]
        S("1. 대상과 정답",
          f"{cfg['ticker']} 의 일간 시세를 <b>월간 막대</b>로 묶습니다: 시가 = 그달 첫 거래일 시가, 종가 = 마지막 거래일 종가, 고가·저가 = 그달 최고·최저, 거래량 = 합. "
          f"막대의 날짜는 그달 <b>마지막 거래일</b>입니다(그날 장 마감 뒤의 값으로 판단 — 판단 시점은 다음 거래일 개장 직전). 예: {t.strftime('%Y-%m')} 은 일간 {len(mo)}행 → 한 줄.",
          code=C(CAL, src.span(CAL, "def resample_prices(", "return out"), src.line(CAL, '"open": g["open"].first(),'), "resample_prices", 0, 2),
          say=(f"매매 모드에서는 {tk_say}의 일간 시세를 월간 막대로 묶습니다. 시가는 그달 첫 거래일의 시가, 종가는 마지막 거래일의 종가, "
               "고가와 저가는 그달의 최고와 최저, 거래량은 합입니다. 막대의 날짜는 그달의 마지막 거래일로, 그날 장이 끝난 뒤의 값으로 판단한다는 뜻입니다. "
               f"예를 들어 {t.strftime('%Y-%m')}은 일간 {len(mo)}행이 한 줄이 되고, 엔진이 만든 막대는 손으로 묶은 값과 같습니다."),
          notes={src.line(CAL, 'rule = {"W": "W-FRI", "M": "ME", "Q": "QE", "Y": "YE"}[freq]'): "월간(M) → 달력의 달마다 묶음(ME = 달 끝)",
                 src.line(CAL, '"open": g["open"].first(),'): f"시가 = 그달 첫 거래일 시가 → {t.strftime('%Y-%m')}: {prices.loc[t, 'open']:,.0f}",
                 src.line(CAL, '"volume": g["volume"].sum(min_count=1),'): f"거래량 = 그달 합 → {prices.loc[t, 'volume']:,.0f}",
                 src.line(CAL, 'out.index = pd.DatetimeIndex(last_day.loc[out.index].values, name="date")'): f"날짜 = 달력의 말일이 아니라 그달 실제 마지막 거래일 → {t.date()}"},
          stack=stk_ds + ["engine/calendar.py · resample_prices"], nodes=["tgt"],
          tensors=[{"type": "table", "title": f"{t.strftime('%Y-%m')} 일간 → 월간 (처음 3일·마지막 3일)", "columns": ["날짜", "시가", "고가", "저가", "종가", "거래량"],
                    "rows": [[d.strftime("%Y-%m-%d"), *[f"{r[c]:,.0f}" for c in ("open", "high", "low", "close", "volume")]] for d, r in pd.concat([mo.head(3), mo.tail(3)]).iterrows()]
                    + [[f"월간 막대 {t.date()}", *[f"{prices.loc[t, c]:,.0f}" for c in ("open", "high", "low", "close", "volume")]]], "hlRows": [6]}],
          checks=[chk("손 계산 월간 막대(시가·고가·저가·종가·거래량) = 엔진", [mo["open"].iloc[0], mo["high"].max(), mo["low"].min(), mo["close"].iloc[-1], mo["volume"].sum()],
                      prices.loc[t, ["open", "high", "low", "close", "volume"]].values.astype(float), 0)])
        o = prices["open"]; y_hand = o.shift(-(1 + k)) / o.shift(-1) - 1

        def open_day(j_):   # 막대 j_ 의 시가가 정해지는 날 = 그 막대의 첫 거래일 (일간 기록)
            return dly.index[(dly.index > prices.index[j_ - 1]) & (dly.index <= prices.index[j_])][0]
        sealed = cfg["validation"].get("sealed_from")
        # 학습 창 끝에서 비우는 행(4. 폴드)마다: 정답의 청산 시가가 결정일 t 뒤인가(결정 시점에 아직 모름) / 앞인가(이미 확정)
        emb_rows = [(prices.index[j_], open_day(j_ + 1), open_day(j_ + 1 + k)) for j_ in range(ti - emb, ti)]
        emb_post = [r_ for r_ in emb_rows if r_[2] > t]; emb_pre = [r_ for r_ in emb_rows if r_[2] <= t]
        sealed_rows = [j_ for j_ in range(max(0, ti - 3), ti + 1) if sealed and open_day(j_ + 1 + k) >= pd.Timestamp(sealed)]   # 표의 결정일 중 정답이 봉인 구간 가격을 쓰는 행
        # 독립 대조용 정답: 월간 막대를 쓰지 않고, 일간 기록에서 '다음 막대 첫 거래일' 의 시가를 직접 찾아 계산
        y_daily = np.full(len(prices), np.nan)
        for j_ in range(len(prices) - 1 - k):
            y_daily[j_] = float(dly.loc[open_day(j_ + 1 + k), "open"]) / float(dly.loc[open_day(j_ + 1), "open"]) - 1
        S("1. 대상과 정답",
          f"정답(타겟): 판단 시점을 't+1 개장 직전'(미국 t일 마감 값까지 본 뒤 — DECISIONS 2026-11-03 규약)으로 정했으므로 <b>다음 막대(t+1)의 시가에 사서 {k}막대 뒤(t+1+{k})의 시가에 파는 수익률</b>입니다. "
          f"월간이면 '다음 달 첫 거래일 시가 → 그다음 달 첫 거래일 시가'. 정답이 {1 + k}막대 뒤의 가격까지 쓰므로 데이터 맨 끝 {1 + k}행은 정답이 비어 있습니다. "
          f"학습 창 끝 {emb}개월을 비우는 규칙(1 + k)은 4. 폴드에서 봅니다. "
          + (f"결정일 뒤 행({prices.index[ti + 1].date()}~)은 봉인 평가 대상이라 표에 넣지 않았습니다. 다만 "
             + ", ".join(f"{prices.index[j_].date()} 행의 정답({open_day(j_ + 1).date()}→{open_day(j_ + 1 + k).date()} 시가)" for j_ in sealed_rows)
             + f" 은 봉인 구간({sealed}~) 가격을 쓰고 개발 평가 성과에도 들어갑니다(8. 모든 폴드) — 엔진은 결정일만 봉인 전으로 자릅니다." if sealed_rows else ""),
          code=C(CAL, src.span(CAL, "def make_target(", 'y.name = f"target_{kind}_k{k}"'), src.line(CAL, "r = exit_ / entry - 1.0"), "make_target", 0, 2),
          say=("정답은 결정일 t까지의 값으로 판단해 얻을 수 있는 수익률입니다. 판단은 다음 거래일 개장 직전, 미국 마감 값까지 본 뒤에 한다고 정했으므로, "
               + ("다음 달 첫 거래일 시가에 사서 그다음 달 첫 거래일 시가에 판다고 봅니다. " if freq == "M" else "다음 막대의 시가에 사서 그다음 막대의 시가에 판다고 봅니다. ")
               + f"예를 들어 결정일 {t.date()}의 정답은 {o.iloc[ti + 1]:,.0f}원에 사서 {o.iloc[ti + 1 + k]:,.0f}원에 판 수익률, {pct(y.loc[t])}입니다. "
               f"정답이 {1 + k}칸 뒤 막대의 시가까지 쓰므로, 데이터 맨 끝 {1 + k}줄은 정답이 비어 있습니다."),
          notes={src.line(CAL, "entry = p.shift(-1)", after=src.line(CAL, "def make_target(")): "진입 = 다음 막대(t+1)의 시가 — shift(−1) 로 한 칸 뒤 값을 지금 행으로",
                 src.line(CAL, "exit_ = p.shift(-(1 + k))", after=src.line(CAL, "def make_target(")): f"청산 = {1 + k}칸 뒤 막대(t+1+{k})의 시가",
                 src.line(CAL, "r = exit_ / entry - 1.0", after=src.line(CAL, "def make_target(")): f"수익률 → 결정일 {t.date()}: {pct(y.loc[t], 3)}. 맨 끝 {1 + k}행은 NaN"},
          stack=stk_ds + ["engine/calendar.py · make_target"], nodes=["tgt"],
          formula=f"y_t = 시가_(t+1+{k}) ÷ 시가_(t+1) − 1",
          tensors=[{"type": "table", "title": f"결정일 {t.date()} 앞뒤 (월간 막대)", "columns": ["결정일 t", "시가_(t+1)", f"시가_(t+1+{k})", "정답 y_t"],
                    "rows": [[prices.index[j].strftime("%Y-%m-%d"), f"{o.iloc[j + 1]:,.0f}", f"{o.iloc[j + 1 + k]:,.0f}" if j + 1 + k < len(o) else "—", pct(y.iloc[j])] for j in range(ti - 3, ti + 1)]}],
          checks=[chk(f"정답 = 일간 기록에서 직접 찾은 '다음 달 첫 거래일 시가 → {'그다음 달' if k == 1 else f'{k}달 뒤'} 첫 거래일 시가' 수익률 (결정일 {int(np.isfinite(y_daily).sum())}개, 끝 {1 + k}개는 둘 다 빈 칸)",
                      y_daily, y.values, 1e-15)])
    # ================================================================ 2. 재료 — 기술지표
    from engine.features import PRESETS
    P = PRESETS[freq]
    vol = prices["volume"] if mode == "trading" else None
    th = tech_hand(prices["close"].values, None if vol is None else vol.values, P, ppy, ti)
    th_rows = [[c, desc[c], g(th[c], 6), g(X.loc[t, c], 6)] for c in tech]
    # 직접 실행해 보기: 기술지표 4개 (순수 파이썬 — Python Tutor 에서 한 줄씩)
    lab_tech = None
    want = [c_ for c_ in ("A_ret_1", "A_ret_12", "A_ma_ratio_3", "A_vol_6") if c_ in feats]
    if len(want) == 4 and ti >= 13:
        c_last = prices["close"].values[: ti + 1][-13:]
        code_tech = (
            "import math\n\n"
            f"# 대상의 {'월별 값' if freq == 'M' else '종가'} — 결정일 {t.date()} 까지 마지막 13개 (실행 기록 그대로)\n"
            f"c = {py_list(c_last)}\n"
            "t = len(c) - 1                      # 결정일 = 맨 끝\n\n"
            "ret_1 = c[t] / c[t - 1] - 1         # 1칸 전 대비 수익률\n"
            "ret_12 = c[t] / c[t - 12] - 1       # 12칸 전 대비 수익률\n\n"
            "ma_3 = (c[t] + c[t - 1] + c[t - 2]) / 3   # 최근 3개 평균\n"
            "ma_ratio_3 = c[t] / ma_3 - 1              # 최근 3개 평균보다 얼마나(비율, × 100 = %) 위(+)·아래(−)인가\n\n"
            f"# 변동성: 최근 6개 로그 변화의 표본표준편차(n − 1 로 나눔) × √{ppy}\n"
            "lr = []\n"
            "for i in range(t - 5, t + 1):\n"
            "    lr.append(math.log(c[i]) - math.log(c[i - 1]))\n"
            "m = sum(lr) / len(lr)\n"
            "var = sum((x - m) ** 2 for x in lr) / (len(lr) - 1)\n"
            f"vol_6 = math.sqrt(var) * math.sqrt({ppy})\n\n"
            "print(\"A_ret_1\", f\"{ret_1:.6g}\")\n"
            "print(\"A_ret_12\", f\"{ret_12:.6g}\")\n"
            "print(\"A_ma_ratio_3\", f\"{ma_ratio_3:.6g}\")\n"
            "print(\"A_vol_6\", f\"{vol_6:.6g}\")\n")
        exp_tech = "\n".join(f"{c_} {float(X.loc[t, c_]):.6g}" for c_ in want)
        lab_tech = {"title": "기술지표 4개를 직접 계산", "desc": "판다스 없이 반복문만으로 수익률·이동평균 비율·변동성을 계산합니다. 출력이 엔진 값(아래 기대 출력)과 같아야 합니다.",
                    **run_lab(code_tech, exp_tech, pure=True)}
    S("2. 재료 — 기술지표",
      f"<b>기술지표 {len(tech)}개</b>는 대상 자신의 과거 {('값' if mode == 'forecast' else '가격·거래량')}으로만 만듭니다(창 길이는 월간 기본: 수익률 {P['lags']}, 이동평균 {P['ma']}, 변동성 {P['vol']}, RSI {P['rsi']}, MACD {P['macd']}"
      + (f", 거래량 {P['volume']}" if mode == "trading" else "") + "개월). 모두 t 까지의 값만 씁니다(rolling·shift 는 과거 방향). "
      f"아래는 결정일 {t.date()} 의 값을 <b>pandas 없이 반복문으로 다시 계산</b>한 것과 엔진 값입니다.",
      code=C(FEAT, src.span(FEAT, "def returns(", 'return pd.DataFrame({"macd": m / close, "macd_signal": s / close, "macd_hist": (m - s) / close})'), src.line(FEAT, "def rsi("), "engine/features.py", 0, 2),
      say=(f"기술지표 {len(tech)}개는 대상 자신의 과거 값으로만 만듭니다. 수익률, 이동평균에서 떨어진 정도, 변동성, RSI, MACD 같은 것들인데, "
           f"모두 결정일까지의 값만 씁니다. 화면의 표는 결정일 {t.date()}의 지표를 판다스 없이 반복문으로 다시 계산한 값과 엔진 값을 나란히 놓은 것이고, 모두 같습니다."),
      notes={src.line(FEAT, 'return pd.DataFrame({f"ret_{l}": close / close.shift(l) - 1.0 for l in lags})'): f"l 칸 전 값과의 비율 − 1 = 수익률 (l = {P['lags']})",
             src.line(FEAT, 'return pd.DataFrame({f"ma_ratio_{w}": close / close.rolling(w).mean() - 1.0 for w in windows})'): f"지금 값 ÷ 최근 w개 평균 − 1 (w = {P['ma']})",
             src.line(FEAT, "lr = np.log(close).diff()"): "로그 변화 = ln(c_t) − ln(c_(t−1))",
             src.line(FEAT, 'return pd.DataFrame({f"vol_{w}": lr.rolling(w).std() * np.sqrt(periods_per_year) for w in windows})'): f"최근 w개 로그 변화의 표본표준편차 × √{ppy} (연율화)",
             src.line(FEAT, "up = d.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()"): "오른 폭만 남겨 지수평균 (Wilder: 가중치 1/window)",
             src.line(FEAT, "dn = (-d.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()"): "내린 폭만 남겨 지수평균",
             src.line(FEAT, "return (100 - 100 / (1 + rs)).rename(f\"rsi_{window}\")"): "RSI = 100 − 100 ÷ (1 + 오른 폭 ÷ 내린 폭)",
             src.line(FEAT, "m = ef - es"): "MACD = 빠른 지수평균 − 느린 지수평균 (뒤에서 종가로 나눔)"},
      lab=lab_tech,
      stack=stk_ds + ["engine/features.py · technical_block"], nodes=["tech"], edges=[("tgt", "tech")],
      formula="ret_l = c_t ÷ c_(t−l) − 1 · ma_ratio_w = c_t ÷ 평균(c_(t−w+1..t)) − 1 · vol_w = 표준편차(ln 변화, n−1) × √12 · RSI = 100 − 100 ÷ (1 + EMA(상승) ÷ EMA(하락)) · MACD = (EMA_빠름 − EMA_느림) ÷ c",
      tensors=[{"type": "table", "title": f"결정일 {t.date()} 의 기술지표: 손 계산 vs 엔진", "columns": ["열", "식", "손 계산", "엔진 X"], "rows": th_rows}],
      checks=[chk(f"기술지표 {len(tech)}개 손 계산 = 엔진 (상대)", [th[c] for c in tech], X.loc[t, tech].values.astype(float), 1e-10, rel=True)])
    # ================================================================ 3. 재료 — 외부 변수 (변환·발표시차·달력 정렬)
    info_t = t  # stamp 모드·매매 모드: 정보 시점 = 결정일
    vrows = []; hand = []; eng = []; bases = []; tf_hand = True; any_period = False
    for vid, v in cap["variables"].items():
        name = vid if v["transform"] in ("", "none") else f"{vid}_{v['transform']}"
        raw = v["raw"]; raw.index = pd.DatetimeIndex(raw.index).normalize()
        rv = raw.dropna(); av_ = rv.values.astype(float); tf_ = (v["transform"] or "none").lower()
        if v.get("basis") == "model_period" and tf_ in ("pct", "diff"):
            # 변수 주기가 모델 주기보다 촘촘한 경우(일간 변수 → 월간·주간 모델, 2026-11-10): 결정일마다 '그날까지 공개된 마지막 원값'을 먼저 놓고,
            # 이번 결정일 값과 바로 앞 결정일 값 사이의 변화를 구한다(= 모델 주기 한 칸의 변화). 엔진 함수를 부르지 않고 손으로.
            av_idx = rv.index + pd.to_timedelta(v["lag"], unit="D")
            def _known(d_):
                ok_ = av_idx <= d_
                return (int(np.flatnonzero(ok_)[-1]) if ok_.any() else None)
            i_t = X.index.get_loc(t); j = _known(t); jp = _known(X.index[i_t - 1]) if i_t > 0 else None
            if j is None or jp is None:
                val = np.nan
            else:
                val = float(av_[j] / av_[jp] - 1) if tf_ == "pct" else float(av_[j] - av_[jp])
            hand.append(val); eng.append(float(X.loc[t, name])); any_period = True
            vrows.append([name, v["name"], v["freq"], v["transform"] + " (결정일 사이)", v["lag"], rv.index[j].strftime("%Y-%m-%d") if j is not None else "—",
                          av_idx[j].strftime("%Y-%m-%d") if j is not None else "—", g(val, 5)])
            bases.append((v["freq"], v["transform"], rv.index[j] if j is not None else None, vid))
            continue
        if tf_ in ("", "none"):
            trv = rv
        elif tf_ in ("pct", "diff"):   # 전 관측 대비 변화율·차이를 배열로 직접 (엔진 apply_transform 을 부르지 않음)
            trv = pd.Series(av_[1:] / av_[:-1] - 1 if tf_ == "pct" else av_[1:] - av_[:-1], index=rv.index[1:])
        else:
            from engine.experiment import apply_transform
            trv = apply_transform(raw, v["transform"]).dropna(); tf_hand = False
        avail = trv.index + pd.to_timedelta(v["lag"], unit="D")
        ok = avail <= info_t
        j = int(np.flatnonzero(ok)[-1]) if ok.any() else None
        val = float(trv.iloc[j]) if j is not None else np.nan
        hand.append(val); eng.append(float(X.loc[t, name]))
        vrows.append([name, v["name"], v["freq"], v["transform"], v["lag"], trv.index[j].strftime("%Y-%m-%d") if j is not None else "—",
                      avail[j].strftime("%Y-%m-%d") if j is not None else "—", g(val, 5)])
        bases.append((v["freq"], v["transform"], trv.index[j] if j is not None else None, vid))
    # 매매 모드: 결정일 당일이 기준일인 변수(일간 시장 변수, 전 거래일 대비 하루 변화) vs 기준 달이 앞선 변수(월간 경제 변수)
    same = [b_ for b_ in bases if b_[2] is not None and b_[2] == t]; old = [b_ for b_ in bases if b_[2] is not None and b_[2] < t]
    gaps = sorted({(t.year * 12 + t.month) - (b_[2].year * 12 + b_[2].month) for b_ in old})
    mtxt = {(1,): "한 달", (2,): "두 달", (1, 2): "한두 달"}.get(tuple(gaps), "몇 달")
    gap_of = {b_[3]: (t.year * 12 + t.month) - (b_[2].year * 12 + b_[2].month) for b_ in old}   # 변수마다 기준 달이 결정 달보다 몇 달 앞인가

    def _mw(gs):   # 1,2 → '한두 달', 2,3 → '두세 달'
        W_ = {1: "한", 2: "두", 3: "세", 4: "네"}; gs = sorted(set(gs))
        if gs and all(g_ in W_ for g_ in gs) and (len(gs) == 1 or (len(gs) == 2 and gs[1] - gs[0] == 1)):
            return "".join(W_[g_] for g_ in gs) + " 달"
        return f"{gs[0]}~{gs[-1]}개월" if gs else ""
    gap_det = ", ".join("·".join(v_ for v_, g_ in gap_of.items() if g_ == gg) + f" {gg}개월" for gg in sorted(set(gap_of.values())))
    lab_old = "월간 변수" if all(b_[0] == "M" for b_ in old) else "변수"; lab_same = "일간 변수" if all(b_[0] == "D" for b_ in same) else "변수"
    same_daily = bool(same) and all(b_[0] == "D" and b_[1] in ("pct", "diff") for b_ in same) and all(b_[0] != "D" for b_ in old)
    g_rank, c_rank, r_rank = {}, {}, {}
    if not is_ridge:   # LightGBM: 이 폴드 gain 순위 · 이 날 기여 순위 · 결과 화면(폴드 평균) 순위
        g_rank = {feats[j_]: r_ + 1 for r_, j_ in enumerate(np.argsort(-cap["lgbm"]["importance_gain"]))}
        c_rank = {feats[j_]: r_ + 1 for r_, j_ in enumerate(np.argsort(-np.abs(cap["lgbm"]["contrib_te"][-1][:-1])))}
        if res and res.get("feature_names") == feats and res.get("feature_importance"):
            r_rank = {feats[j_]: r_ + 1 for r_, j_ in enumerate(np.argsort(-np.asarray(res["feature_importance"], dtype=float)))}
    cav = ""   # 공개 시점은 지켜도 남는 한계: 최신 수정값(vintage) · 공개 시각 미확인
    def _note_rev(v_):   # 변수 사전 note 의 '수정:' 뒤 문구 그대로
        nt = str(mst.loc[v_, "note"]); return nt.split("수정:", 1)[1].strip() if "수정:" in nt else nt
    if vint:
        groups = {}
        for v_ in vint:
            groups.setdefault(_note_rev(v_), []).append(v_)
        cav += (f" <b>최신 수정값 문제</b>: 경제 변수 {len(vint)}개({'·'.join(vint)})는 기준 달의 첫 발표일(lag) 뒤부터 쓰지만, 쓰는 값은 그 뒤 수정·재계산된 최신 값이라 "
                "결정일에 실제로 공개되어 있던 값과 다를 수 있습니다 — 결정일에는 없던 정보가 섞이는 셈입니다(변수 사전 note: "
                + "; ".join(f"{'·'.join(vs)} '{q}'" for q, vs in groups.items()) + f"). 그래서 이 변수{'들' if len(vint) > 1 else ''}의 {'계수·기여' if is_ridge else '기여·중요도'}가 실시간 자료로 했을 때보다 크게(예측 성과는 좋게) 나올 수 있습니다(이 영상에서 확인하지 않음).")
        tops = ([f"{col_of[v_]} 는 이 폴드 gain {g_rank[col_of[v_]]}위" for v_ in vint if g_rank.get(col_of[v_], 99) <= 5]
                + [f"{col_of[v_]} 는 결과 화면(폴드 평균) {r_rank[col_of[v_]]}위" for v_ in vint if r_rank.get(col_of[v_], 99) <= 5])
        if tops:
            cav += " 이 가운데 " + ", ".join(tops) + "입니다."
    for v_ in unconf:
        cav += (f" <b>{v_}</b> 는 lag 0 이 아직 확인되지 않았습니다(lag_confirmed = N" + (("; Yahoo 일봉 경계가 09:00 KST 이후면 누출 — 변수 사전 note" if mode == "trading" else "; Yahoo 일봉 경계 미확인 — 변수 사전 note") if v_ == "USDKRW" else "") + ")"
                + (f" — 이 폴드에서 gain {g_rank[col_of[v_]]}위 · 기여 {c_rank[col_of[v_]]}위입니다." if g_rank else "."))
    G_all = np.array([x_["gain"] for x_ in af]) if not is_ridge else None
    for v_ in t_unconf:
        cav += (f" <b>{v_}</b> 는 lag_confirmed = Y 이지만 변수 사전 note 의 조건이 '{str(mst.loc[v_, 'note']).split('조건:', 1)[1].strip()}' 인데 이 변수의 출처(source)는 "
                f"{mst.loc[v_, 'source']} 라 {tu_verb.replace('않았습니다', '않았습니다')}"
                + (f" — {col_of[v_]} 는 {len(af)}개 폴드 중 {int(sum(col_of[v_] in [feats[j_] for j_ in np.argsort(-gr_)[:5]] for gr_ in G_all))}개에서 이득 상위 5 입니다." if G_all is not None else "."))
    req_ids = list(cfg.get("variable_ids") or [])
    dup_ids = [v_ for v_ in req_ids if v_ not in cap["variables"] and mst is not None and v_ in mst.index and str(mst.loc[v_, "duplicate_of"]) not in ("", "nan")]
    if dup_ids:
        cav += (f" 설정의 variable_ids 는 {len(req_ids)}개지만 " + ", ".join(f"{v_}(→ {mst.loc[v_, 'duplicate_of']})" for v_ in dup_ids)
                + f" 는 변수 사전의 duplicate_of 에 따라 대표 변수로 바뀌어 이미 있는 변수와 겹치므로 한 번만 쓰여 {len(cap['variables'])}개입니다(Registry.resolve_many).")
    S("3. 재료 — 외부 변수",
      f"외부 변수 {len(exo)}개는 ① 원래 주기에서 변환(pct = 전기 대비 변화율, diff = 전기 대비 차이) ② <b>발표시차</b>: 기준일 + lag 일 = 공개일, 공개일부터만 사용 "
      f"③ 결정일 달력에 맞춰 '그날까지 공개된 마지막 값'(ffill) 으로 놓입니다. "
      + ("단, 주기가 모델보다 촘촘한 변수(표의 변환 칸에 '결정일 사이')는 순서가 다릅니다: 원값을 먼저 결정일에 맞춰 놓고(②③), 이번 결정일과 바로 앞 결정일 사이의 변화를 구합니다 — 모델 주기 한 칸의 변화입니다. " if any_period else "")
      + ((f"그래서 결정일 {t.date()} 에 {lab_old} {len(old)}개는 기준 달이 {_mw(gap_of.values())} 전({gap_det}), {lab_same} {len(same)}개는 결정일 당일 값입니다. " if same or old else "")
         if mode == "forecast" or not same_daily else
         f"그래서 결정일 {t.date()} 에 일간 시장 변수 {len(same)}개는 <b>그날의 값</b>(전 거래일 대비 하루 변화율·차이)을, 월간 경제 변수 {len(old)}개는 기준 달이 {mtxt} 전인 값을 씁니다. ")
      + "아래 표는 변수마다 그 값을 원자료에서 직접 찾아 엔진 값과 맞춰 본 것입니다."
      + (f" (예측 연구 모드의 결정일 t 는 대상 값 {cfg['target_variable']}_t 의 기준일입니다. 외부 변수는 보수적으로 <b>기준일 t 까지 공개된 값</b>만 씁니다(forecast_info_time = {cfg.get('forecast_info_time')}); "
         "대상 값이 실제로 공개되는 날까지 기다려 그때까지의 정보를 쓰려면 release 로 바꿉니다.)" if mode == "forecast" else
         " (미국 주가·VIX 의 lag 0: 미국 t일 마감은 한국 시각 다음 날 05~06시라 체결 시점(다음 거래일 09:00 시가) 전에 알려집니다 — '결정 시점 = t+1 개장 직전' 규약, DECISIONS 2026-11-03.)")
      + cav,
      code=C(CAL, src.span(CAL, "    s = s.dropna().copy()", "return filled.reindex(calendar).rename(s.name)"), src.line(CAL, "s.index = s.index + pd.to_timedelta(int(lag_days), unit='D')"), "align_to_calendar", 2, 2),
      say=(f"외부 변수 {len(exo)}개는 원래 주기에서 변화율이나 차이로 바꾸고, 기준일에 발표 시차를 더한 공개일부터만 쓰며, 결정일 달력에 맞춰 그날까지 공개된 마지막 값을 놓습니다. "
           + f"그래서 결정일 {t.date()}에, 일간 시장 변수 {len(same)}개는 그날의 하루 변화를, 월간 경제 변수 {len(old)}개는 {mtxt} 전 달의 값을 씁니다. "
           "미국 주가와 VIX는 한국 시각 다음 날 새벽에 나와, 다음 거래일 아침 시가로 체결하기 전에 알려집니다. "
           + (f"{_att(tu_say, '은')} 체결 전에 공개되는지 아직 확인되지 않았고" + (f", 원 달러 환율은 이 폴드에서 중요도 {g_rank[col_of['USDKRW']]}위입니다. " if "USDKRW" in unconf and g_rank else "습니다. ") if tu_say else "")
           + (f"또 경제 변수 {len(vint)}개는 첫 발표일 뒤부터 쓰지만, 값은 그 뒤 고쳐진 최신 값이라 결정일에는 없던 정보가 섞일 수 있습니다." if vint else "")
           if mode == "trading" and same_daily else
           f"외부 변수 {len(exo)}개는 세 단계를 거칩니다. 첫째, 원래 주기에서 변화율이나 차이로 바꿉니다. 둘째, 발표 시차를 넣습니다. "
           "기준일에 시차 일수를 더한 날이 공개일이고, 공개일부터만 씁니다. 셋째, 결정일 달력에 맞춰 그날까지 공개된 마지막 값을 놓습니다. "
           + ((f"그래서 결정일 {t.date()}에 {lab_old} {len(old)}개는 기준 달이 {_mw(gap_of.values())} 전인 값을, {lab_same} {len(same)}개는 결정일 당일의 값을 씁니다. "
               + (("다만 " + ((f"{_att(vint_name, '은')} 발표 뒤 고쳐진 최신 값" if len(vint) == 1 else f"경제 변수 {len(vint)}개는 발표 뒤 고쳐진 최신 값") if vint else "")
                   + (("이고, " if vint else "") + f"{_att(tu_say, '은')} {tu_verb}" if tu_say else "입니다") + ". ") if (vint or tu_say) else ""))
              if mode == "forecast" or not same_daily else
              f"그래서 결정일 {t.date()}에, 일간 시장 변수 {len(same)}개는 그날의 하루 변화를, 월간 경제 변수 {len(old)}개는 {mtxt} 전 달의 값을 씁니다. "
              "미국 시장 값은 한국 시각 다음 날 새벽에 나와, 다음 거래일 아침 시가로 체결하기 전에 알려집니다. "
              + (f"다만 원 달러 환율은 이것이 아직 확인되지 않았고, 이 폴드에서 중요도 {g_rank[col_of['USDKRW']]}위입니다. " if "USDKRW" in unconf and g_rank else "")
              + ("또 일부 경제 변수는 발표 뒤에 고쳐진 최신 값이라, 결정 당시에 보였던 값과 다를 수 있습니다. " if vint else ""))
           + "화면의 표는 변수마다 원자료에서 그 값을 직접 찾아 엔진 값과 맞춰 본 것입니다."
           + (" 예측 연구에서는 보수적으로, 외부 변수는 결정일까지 이미 공개된 값만 씁니다." if mode == "forecast" else "")),
      notes={src.line(CAL, "s.index = s.index + pd.to_timedelta(int(lag_days), unit='D')"): "기준일 + lag 일 = 공개일 — 값의 날짜 자체를 공개일로 옮김",
             src.line(CAL, "filled = on_union.ffill()"): "공개된 마지막 값을 다음 공개 전까지 이어 씀 (미래 값으로 채우지 않음)",
             src.line(CAL, "return filled.reindex(calendar).rename(s.name)"): "결정일 달력에 맞춰 그날의 값만 꺼냄"},
      stack=stk_ds + ["engine/calendar.py · align_to_calendar"], nodes=["exo"], edges=[("tgt", "exo")],
      formula="공개일 = 기준일 + lag 일,   x_t = 공개일 ≤ t 인 마지막 값 (변환은 원래 주기에서 먼저)",
      tensors=[{"type": "table", "title": f"결정일 {t.date()} 에 쓰인 외부 변수 값 — 어느 기준일의 값인가", "columns": ["열", "이름", "주기", "변환", "lag(일)", "기준일", "공개일", "값"], "rows": vrows}],
      hint=("lag 은 변수 사전(variable_master.xlsx)의 lag_days 입니다(2026-11-05 공개일 실측으로 검증). "
            + ("일간 시장 변수는 lag 0 — 기준일 t 당일의 값(그날 장 마감·고시 값)을 씁니다. " if mode == "forecast" else
               "일간 시장 변수는 lag 0 — 결정일 t 의 값을 씁니다: 한국 시장 값은 t 의 장 마감 값, 미국 주가·VIX 는 미국 t일 마감 값(한국 시각 다음 날 새벽)이라 체결 시점(다음 거래일 시가) 전에 알려집니다. "
               + "".join(f"{v_} 의 변수 사전 note: '{str(mst.loc[v_, 'note']).split('[lag 검증 2026-11-05]', 1)[-1].strip()}' — 이 변수의 출처(source)는 {mst.loc[v_, 'source']} 입니다. " for v_ in t_unconf))
            + "빈 칸을 미래 값으로 채우는 방식(bfill·interp)은 미래 정보를 쓰므로 쓰지 않고(ffill), 엔진은 정답과 |상관| > 0.999 인 입력 열이 있으면 멈춥니다(assert_no_target_in_inputs — 명백한 누출만 잡는 검사)."),
      checks=[chk(f"외부 변수 {len(exo)}개: 원자료에서 변화율·차이를 손으로 계산하고 공개일로 찾은 값 = 엔진 X" if tf_hand else f"외부 변수 {len(exo)}개: 원자료에서 직접 찾은 값 = 엔진 X (변환은 엔진 함수)", hand, eng, 1e-12)])
    first_ok = X.dropna().index[0]
    exo_from_first = all(X[c_].first_valid_index() == X.index[0] for c_ in exo)   # 외부 변수가 모두 첫 행부터 값이 있나
    import re as _re2
    win_max = max([int(m_.group(1)) for c_ in X.columns[X.loc[X.index[X.index.get_loc(first_ok) - 1]].isna()] for m_ in [_re2.search(r"_(\d+)$", c_)] if m_] or [0])
    S("3. 재료 — 외부 변수",
      f"이렇게 만든 결정일 × 재료 표 X 는 {X.shape[0]}행 × {K}열입니다. 지표의 창이 차기 전{'' if exo_from_first else ', 변수가 공개되기 전'}의 앞쪽 행은 빈 칸이 있고(첫 완전한 행 {first_ok.date()}), "
      "빈 칸이 있는 행은 학습 표에서 빠집니다. "
      + ("" if is_ridge else
         f"이 실행에서 외부 변수는 모두 {', '.join(sorted({str(X[c_].first_valid_index().date()) for c_ in exo}))} 부터 값이 있고, 첫 완전한 행 바로 앞 {X.index[X.index.get_loc(first_ok) - 1].date()} 에는 "
         f"{', '.join(_nm(c_) for c_ in X.columns[X.loc[X.index[X.index.get_loc(first_ok) - 1]].isna()])} 만 비어 있습니다(창이 아직 차지 않음). ")
      + f"아래 그림은 마지막 폴드 학습 표({len(tr)}행)를 열(재료)마다 그 학습 표의 평균·표준편차로 <b>표준화</b>해 색칠한 것입니다 — 그림에서는 행 = 재료, 열 = 결정일.",
      code=C(EXP, src.lines(EXP, "X = pd.concat(parts, axis=1).reindex(calendar) if parts else pd.DataFrame(index=calendar)", "cal.assert_no_target_in_inputs(X, y, cfg.target.k)"),
             src.line(EXP, "X = pd.concat(parts, axis=1).reindex(calendar) if parts else pd.DataFrame(index=calendar)"), "build_dataset", 2, 6),
      say=((f"이렇게 만든 재료 표는 {X.shape[0]}행, {K}열입니다. 앞쪽 행은 지표의 창이 아직 차지 않았거나 변수가 공개되기 전이라 빈 칸이 있고, "
            f"빈 칸이 있는 행은 학습에서 빠집니다. 빈 칸 없는 첫 행은 {first_ok.date()}입니다. 화면의 그림은 마지막 폴드 학습 표의 값을 열마다 표준화해 색으로 칠한 것입니다.") if (is_ridge or not exo_from_first) else
           (f"이렇게 만든 재료 표는 {X.shape[0]}행, {K}열입니다. 앞쪽 행은 기술지표의 창이, 가장 긴 것은 {win_max}개월이라, 아직 차지 않아 빈 칸이 있습니다. 외부 변수는 첫 행부터 값이 있습니다. "
            f"빈 칸이 있는 행은 학습에서 빠지고, 빈 칸 없는 첫 행은 {first_ok.date()}입니다. 화면의 그림은 마지막 폴드 학습 표의 값을 재료마다 표준화해 색으로 칠한 것입니다.")),
      notes={src.line(EXP, "X = pd.concat(parts, axis=1).reindex(calendar) if parts else pd.DataFrame(index=calendar)"): f"기술지표 {len(tech)}열 + 외부 변수 {len(exo)}열을 붙여 결정일 달력에 맞춤 → {X.shape[0]} × {K}",
             src.line(EXP, "cal.assert_no_target_in_inputs(X, y, cfg.target.k)"): "정답과 |상관| > 0.999 인 입력 열이 있으면 멈춤 (명백한 누출만 잡는 검사)"},
      stack=stk_ds, nodes=["ds"], edges=[("tech", "ds"), ("exo", "ds")],
      tensors=[Hm(f"학습 표의 재료 (재료(그림의 행)마다 따로 색칠: 학습 평균에서 표준편차 몇 배 떨어졌나, z)", ((tr[feats] - tr[feats].mean()) / tr[feats].std(ddof=0)).values.T, rows=feats,
                  cols=[d.strftime("%y-%m") for d in tr.index], numbers=False, cellH=12 if K > 25 else 15, cellW=13, rowLabelW=120, colLabelH=40)])
    # ================================================================ 4. 폴드
    ftr = cap["fold"]["train"]; fte = cap["fold"]["test"]
    n_val_raw = int(len(tr) * cap["val_frac"]) if cap["uses_val"] else 0
    emb_d = list(prices.index[prices.index.get_loc(ftr[-1]) + 1: prices.index.get_loc(fte[0])])   # 비우는 행
    i_te = prices.index.get_loc(fte[0]); win_te = prices.index[i_te - tl: i_te - emb]   # 학습 창 = 달력[i − train_len : i − embargo] (달력 = 가격 날짜)
    rows_te = [d for d in win_te if X.loc[d, feats].notna().all() and np.isfinite(y.loc[d])]
    if mode == "trading":
        assert [r_[0] for r_ in emb_rows] == emb_d
        f_d = lambda rs: ", ".join(str(r_[0].date()) for r_ in rs)
        emb_why = (f"규칙은 1 + k = {1 + k}개월입니다(정답이 t+1+k 막대 시가까지 씀). 이 폴드에서 결정({t.date()}) 뒤의 가격을 쓰는 정답은 "
                   + ", ".join(f"{r_[0].date()} 행({r_[1].date()}→{r_[2].date()} 시가)" for r_ in emb_post) + " 이고"
                   + (", " + ", ".join(f"{r_[0].date()} 행의 정답({r_[1].date()}→{r_[2].date()} 시가)" for r_ in emb_pre) + " 은 결정 전에 이미 확정되지만 규칙에 따라 보수적으로 함께 뺍니다. " if emb_pre else ". "))
    S("4. 폴드",
      f"워크포워드: 결정일 {tl}개(월) 창에서 끝 {emb}개월을 뺀 {tl - emb}개월로 학습하고 다음 1개월을 예측, 한 달 밀어 반복합니다(폴드 {cap['n_folds']}개). 학습 창 끝의 <b>{emb}개월은 비웁니다</b>(embargo) — "
      + (f"{'그 달' if emb == 1 else '그 달들'}의 정답이 테스트 기간의 {'값' if mode == 'forecast' else '가격'}으로 계산되기 때문입니다. " if mode == "forecast" else emb_why)
      + f"마지막 폴드: 학습 {ftr[0].date()} ~ {ftr[-1].date()} ({len(ftr)}개) → 테스트 {fte[0].date()}. "
      f"빈 칸 없는 행만 남기면 학습 표는 <b>{len(tr)}행</b>입니다. "
      + ("Ridge 는 검증 구간을 쓰지 않습니다(USES_VALIDATION = False) — 학습 표 전체로 맞춥니다." if is_ridge else
         f"LightGBM 은 검증을 쓸 수 있는 모델이지만 검증 개수 = int({len(tr)} × {cap['val_frac']}) = {n_val_raw} 가 10 보다 작아 <b>검증 없이</b> 학습 표 전체로 {p['n_estimators']}그루를 모두 만듭니다(조기 종료 없음). "
         f"기록으로 보면 폴드 {len(af)}개 모두 검증 0개입니다(폴드별 학습 행 {min(x_['n_tr'] for x_ in af)}~{max(x_['n_tr'] for x_ in af)}개 — 가장 많아도 {max(x_['n_tr'] for x_ in af)}행이라 검증 개수가 10 에 못 미침)."),
      code=C(EXP, src.lines(EXP, "def _split_fit(m, X, y, n_val: int, emb: int):", "if n_val >= 10 and n - n_val - emb >= 10:", "m.fit(X, y)", "return int(len(idx) * cfg.validation.val_frac)"),
             src.line(EXP, "if n_val >= 10 and n - n_val - emb >= 10:"), "_split_fit", 2, 10) if not is_ridge else
           C(CAL, src.span(CAL, "    emb = (1 + k) if embargo is None else embargo", "return folds"), src.line(CAL, "tr = cal[i - train_len : i - emb] if emb else cal[i - train_len : i]"), "walkforward_folds", 0, 0),
      say=((f"워크포워드는 최근 {cfg['validation']['train_len']}개월로 학습하고 다음 한 달을 예측한 뒤, 한 달씩 밀며 반복합니다. 모두 {cap['n_folds']}번입니다. "
            + f"학습 창 끝의 {'한 달' if emb == 1 else f'{emb}개월'}은 비워 둡니다. {'그 달' if emb == 1 else '그 달들'}의 정답을 계산하려면 테스트 기간의 값이 필요하기 때문입니다. "
            + f"마지막 폴드는 {ftr[0].date()}부터 {ftr[-1].date()}까지로 학습하고, {fte[0].date()}을 예측합니다. 빈 칸 없는 행만 남기면 학습 표는 {len(tr)}행입니다. "
            + ("릿지는 검증 구간을 따로 쓰지 않고 학습 표 전체로 맞춥니다." if is_ridge else
               f"라이트지비엠은 검증 구간을 쓸 수 있지만, 여기서는 검증 개수가 {n_val_raw}개로 10개보다 적어서 검증 없이 나무 {p['n_estimators']}그루를 모두 만듭니다. "
               f"학습 행이 가장 많은 폴드도 {max(x_['n_tr'] for x_ in af)}행이라, 모든 폴드에서 검증과 조기 종료가 없었습니다.")) if (is_ridge or mode != "trading") else
           (f"워크포워드는 {tl}개월 창에서 끝 {emb_w}을 뺀 {tl - emb}개월로 학습하고, 다음 한 달을 예측하며 한 달씩 밉니다. 모두 {cap['n_folds']}번입니다. "
            + f"끝 {emb_w}을 비우는 까닭은 {f_d(emb_post)}의 정답이 결정 뒤 가격인 {emb_post[0][2].date()} 시가를 쓰기 때문이고"
            + (f", {f_d(emb_pre)}의 정답은 결정 전에 정해지지만 규칙대로 함께 뺍니다. " if emb_pre else ". ")
            + f"마지막 폴드는 {ftr[-1].date()}까지 {len(tr)}행으로 학습해, 결정일 {t.date()}의 다음 달 수익률을 예측합니다. "
            + f"검증 개수가 {n_val_raw}개로 10개보다 적어, 모든 폴드에서 검증과 조기 종료 없이 {p['n_estimators']}그루를 만듭니다.")),
      notes=({src.line(CAL, "emb = (1 + k) if embargo is None else embargo"): f"embargo 를 넘기지 않으면 1 + k. 이 실행은 엔진(make_folds → _embargo)이 {emb} 을 넘김 — 예측 모드는 k, 매매 모드는 1 + k",
              src.line(CAL, "tr = cal[i - train_len : i - emb] if emb else cal[i - train_len : i]"): f"학습 = 테스트 시작 i 바로 앞 {cfg['validation']['train_len']}개 중 끝 {emb}개를 뺀 {cfg['validation']['train_len'] - emb}개",
              src.line(CAL, "te = cal[i : i + test_len]"): "테스트 = i 부터 test_len 개 (여기선 1개월)",
              src.line(CAL, "i += test_len"): "한 칸(1개월) 밀어 다음 폴드"} if is_ridge else
             {src.line(EXP, "if n_val >= 10 and n - n_val - emb >= 10:"): f"검증 개수 {n_val_raw} 가 10 보다 작아 이 조건이 거짓 → 검증 없이 학습",
              src.line(EXP, "m.fit(X, y)"): f"학습 표 전체({len(tr)}행)로 학습 — 모델 안에서 X_val = None 이라 조기 종료도 없음",
              src.line(EXP, "return int(len(idx) * cfg.validation.val_frac)"): f"검증 개수 = int({len(tr)} × {cap['val_frac']}) = {n_val_raw}"}),
      stack=["engine/experiment.py · make_folds", "engine/calendar.py · walkforward_folds"], nodes=["fold"], edges=[("ds", "fold")],
      formula=f"학습 = 달력[i − {cfg['validation']['train_len']} : i − {emb}],   테스트 = 달력[i : i + 1],   i += 1",
      tensors=[{"type": "segs", "title": "마지막 폴드 (월)", "segs": [{"label": "학습", "n": len(ftr), "color": "#2f8f5b", "sub": f"{ftr[0].strftime('%Y-%m')} ~ {ftr[-1].strftime('%Y-%m')}"},
                                                               {"label": "비움", "n": emb, "color": "#c3c8d0"}, {"label": "테스트", "n": 1, "color": "#c8710f"}]},
               {"type": "table", "title": "학습 표 끝 3행 → 비움 → 테스트", "columns": ["결정일", "역할", "정답 y"],
                "rows": [[d.strftime("%Y-%m-%d"), "학습", pct(tr.loc[d, "y"])] for d in tr.index[-3:]]
                + ([[d.strftime("%Y-%m-%d"), f"비움(정답이 테스트 기간 {'값' if mode == 'forecast' else '가격'}을 씀)", pct(y.loc[d])] for d in emb_d] if mode != "trading" else
                   [[r_[0].strftime("%Y-%m-%d"), (f"비움(정답이 결정 뒤 가격 {r_[2].date()} 시가를 씀)" if r_[2] > t else f"비움(보수적 — 정답은 결정 전 {r_[2].date()} 에 확정)"), pct(y.loc[r_[0]])] for r_ in emb_rows])
                + [[d.strftime("%Y-%m-%d"), "테스트", pct(y.loc[d])] for d in fte]}],
      checks=[chk(f"학습 창을 달력 번호로 따로 계산(달력[{i_te - tl} : {i_te - emb}] = {win_te[0].date()} ~ {win_te[-1].date()})해 빈 칸 없는 행만 남긴 표(날짜·재료·정답, {len(rows_te)}행) "
                  f"= 엔진 학습 표 · 모든 행이 첫 완전한 행({first_ok.date()}) 이후",
                  np.r_[[d.toordinal() for d in rows_te], np.c_[X.loc[rows_te, feats].values, y.loc[rows_te].values].ravel(), float(min(rows_te) >= first_ok)] if len(rows_te) == len(tr) else [np.nan],
                  np.r_[[d.toordinal() for d in tr.index], np.c_[tr[feats].values, tr["y"].values].ravel(), 1.0] if len(rows_te) == len(tr) else [0.0], 0)]
             + ([] if is_ridge else [chk(f"폴드 {len(af)}개 모두 검증 개수 = 0 (기록 = _split_fit 규칙: int(학습 행 × {cap['val_frac']}) < 10 이면 0)", [x_["n_va"] for x_ in af],
                                       [int(x_["n_tr"] * cap["val_frac"]) if (int(x_["n_tr"] * cap["val_frac"]) >= 10 and x_["n_tr"] - int(x_["n_tr"] * cap["val_frac"]) - emb >= 10) else 0 for x_ in af], 0, kind="식")]))
    # ================================================================ 5. 눈금
    mu, sd = cap["scaler_mean"], cap["scaler_scale"]
    xt = te.loc[t, feats].values.astype(float); z = (xt - mu) / sd
    jz = int(np.argmax(np.abs(z)))
    code_scale = (
        f"# 마지막 폴드 학습 표({len(tr)}행)의 '{feats[jz]}' 열 — 표준화 전 값 (실행 기록 그대로)\n"
        f"x = {py_list(X.loc[rows_te, feats[jz]].values)}\n"
        f"x_test = {float(xt[jz])!r}   # 결정일 {t.date()} 의 값\n\n"
        "n = len(x)\n"
        "total = 0.0\n"
        "for v in x:\n"
        "    total += v\n"
        "mean = total / n\n\n"
        "sq = 0.0\n"
        "for v in x:\n"
        "    sq += (v - mean) ** 2\n"
        "std = (sq / n) ** 0.5          # n 으로 나눔 (sklearn StandardScaler 와 같음)\n\n"
        "z = (x_test - mean) / std      # 평소보다 표준편차 몇 칸인가\n"
        "print(\"평균\", f\"{mean:.6g}\")\n"
        "print(\"표준편차\", f\"{std:.6g}\")\n"
        "print(\"z\", f\"{z:.6g}\")\n")
    exp_scale = f"평균 {mu[jz]:.6g}\n표준편차 {sd[jz]:.6g}\nz {cap['Xte'][-1][jz]:.6g}"
    lab_scale = {"title": f"'{feats[jz]}' 의 z 를 직접 계산", "desc": f"결정일 {t.date()} 에 평소에서 가장 멀리 떨어진 재료입니다. 평균·표준편차를 학습 표로만 구해 z 를 계산합니다.",
                 **run_lab(code_scale, exp_scale, pure=True)}
    zero_n = node_n = 0   # LightGBM: 기준이 0 경계(±1e-35)인 마디 수 — 표준화가 그 경계를 학습 평균 자리에 놓는다
    if not is_ridge:
        def _walk_nodes(n_):
            if "leaf_value" in n_: return []
            return [n_] + _walk_nodes(n_["left_child"]) + _walk_nodes(n_["right_child"])
        _nodes = [n_ for t_ in cap["lgbm"]["dump"]["tree_info"] for n_ in _walk_nodes(t_["tree_structure"])]
        node_n = len(_nodes); zero_n = sum(abs(n_["threshold"]) < 1e-30 for n_ in _nodes)
    S("5. 눈금 맞추기",
      "단위가 다른 재료를 같은 눈금으로 바꿉니다: z = (x − 평균) ÷ 표준편차. 평균·표준편차는 <b>학습 표로만</b> 계산합니다(테스트·미래 값은 쓰지 않음). "
      f"결정일 {t.date()} 의 재료를 z 로 바꾸면 '평소보다 몇 칸(표준편차) 큰가' 가 됩니다. 이 날 평소에서 가장 멀리 떨어진 재료는 {_nm(feats[jz])} 로 z = {z[jz]:+.3f} 입니다."
      + ("" if is_ridge else
         f" <b>나무 모델에서도 표준화는 중립이 아닙니다</b>: LightGBM 은 값을 구간으로 묶을 때 0 을 늘 경계로 따로 두므로(덤프의 기준 ±1e-35), 표준화하면 그 경계가 원래 단위의 0 이 아니라 "
         f"학습 평균 자리에 놓입니다 — 이 폴드의 기준 {node_n}개 중 {zero_n}개가 그 자리입니다(6. 학습 화면). 표준화하지 않았을 때의 결과는 이 영상에서 계산하지 않았습니다."),
      code=C(EXP, src.lines(EXP, "scaler = (MinMaxScaler() if params.get(\"input_scaler\") == \"minmax\" else StandardScaler()).fit(fit_rows)",
                            "Xtr = scaler.transform(tr[feat].values)", "Xte = scaler.transform(te_valid[feat].values)"),
             src.line(EXP, "scaler = (MinMaxScaler() if params.get(\"input_scaler\") == \"minmax\" else StandardScaler()).fit(fit_rows)"), "_walkforward_predict", 2, 6),
      say=("단위가 서로 다른 재료를 같은 눈금으로 바꿉니다. 각 값에서 평균을 빼고 표준편차로 나누는데, 이 평균과 표준편차는 학습 표로만 계산합니다. "
           "테스트나 미래의 값은 쓰지 않습니다. 이렇게 바꾼 값 z는, 평소보다 표준편차 몇 칸만큼 크거나 작은지를 뜻합니다. "
           f"예를 들어 결정일 {t.date()}의 {nm(feats[jz], '는')} z가 {z[jz]:+.2f}로, 이 날 재료 가운데 평소에서 가장 멀리 떨어져 있습니다."
           + ("" if is_ridge else " 라이트지비엠에서도 이 표준화는 결과를 바꿀 수 있습니다. 값 0을 늘 구간 경계로 두는데, 표준화하면 그 경계가 학습 평균 자리로 가기 때문입니다.")),
      notes={src.line(EXP, 'scaler = (MinMaxScaler() if params.get("input_scaler") == "minmax" else StandardScaler()).fit(fit_rows)'): "StandardScaler: 열마다 학습 표의 평균과 표준편차(n 으로 나눔)를 저장",
             src.line(EXP, "Xtr = scaler.transform(tr[feat].values)"): f"학습 표 → z ({len(tr)} × {K})",
             src.line(EXP, "Xte = scaler.transform(te_valid[feat].values)"): f"결정일 {t.date()} 의 재료 → 같은 평균·표준편차로 z (1 × {K})"},
      lab=lab_scale,
      stack=stk_wf, nodes=["scale"], edges=[("fold", "scale")],
      formula="z_j = (x_j − 평균_j) ÷ 표준편차_j   (표준편차는 n 으로 나눔, ddof = 0 — sklearn StandardScaler)",
      tensors=[{"type": "table", "title": f"결정일 {t.date()}: x → z", "columns": ["열", "x", "평균", "표준편차", "z"],
                "rows": [[c, g(xt[j], 5), g(mu[j], 5), g(sd[j], 5), f"{z[j]:+.3f}"] for j, c in enumerate(feats)], "split": 2 if K > 20 else 1,
                "hlRows": [] if K > 20 else [int(np.argmax(np.abs(z)))]}],
      checks=[chk(f"엔진 눈금의 평균·표준편차 = 4. 폴드에서 달력으로 따로 고른 학습 행({len(rows_te)}행)의 열 평균·표준편차(ddof 0) — 학습 표로만 맞췄다는 확인",
                  np.r_[mu, sd], np.r_[X.loc[rows_te, feats].mean().values, X.loc[rows_te, feats].std(ddof=0).values], 1e-12, rel=True),
              chk(f"결정일 {t.date()} 의 원래 값(재료 표 X)을 그 평균·표준편차로 바꾼 z = 엔진 Xte", (X.loc[t, feats].values.astype(float) - X.loc[rows_te, feats].mean().values) / X.loc[rows_te, feats].std(ddof=0).values,
                  cap["Xte"][-1], 1e-12)])
    Ztr = cap["Xtr"]; ytr = cap["ytr"]
    # ================================================================ 6~ 모델별
    if is_ridge:
        rd = cap["ridge"]; a = float(p["alpha"]); w = rd["coef"]; b = rd["intercept"]
        S("6. Ridge 학습",
          "Ridge 는 sklearn <b>make_pipeline(StandardScaler(), Ridge(alpha))</b> 입니다. 엔진이 이미 표준화했으므로 모델 안의 두 번째 표준화는 평균 0·표준편차 1 을 다시 맞추는 것이라 "
          "<b>사실상 아무것도 바꾸지 않습니다</b>(아래 검증). 그다음 Ridge 가 '예측과 정답 차이의 제곱합 + alpha × 계수 제곱합' 을 최소로 하는 계수를 공식 한 번으로 구합니다.",
          code=C(MODEL, src.span(MODEL, "def fit(self, X, y, X_val=None, y_val=None):", "self.pipe = make_pipeline(StandardScaler(), est).fit(X, y)"), src.line(MODEL, "self.pipe = make_pipeline(StandardScaler(), est).fit(X, y)"), "Model.fit", 6, 8),
          say=("릿지 모델은 사이킷런의 파이프라인으로, 표준화 다음에 릿지 회귀를 합니다. 엔진이 이미 표준화를 했기 때문에, 모델 안의 두 번째 표준화는 사실상 아무것도 바꾸지 않습니다. "
               "화면의 검증이 그것을 확인합니다. 그다음 릿지는, 예측 오차의 제곱합에 계수 제곱합의 알파 배를 더한 값이 가장 작아지는 계수를 공식 한 번으로 구합니다. "
               f"알파는 계수가 너무 커지지 않게 누르는 벌점의 세기이고, 이 설정에서는 {a:g}입니다."),
          notes={src.line(MODEL, 'a = self.params["alpha"]'): f"벌점 세기 alpha = {a:g}",
                 src.line(MODEL, "est = Ridge(alpha=a) if self.target_type == \"regression\" else LogisticRegression(C=1.0 / a, max_iter=2000)"): "정답이 숫자(회귀)라 Ridge",
                 src.line(MODEL, "self.pipe = make_pipeline(StandardScaler(), est).fit(X, y)"): "표준화 → Ridge 를 이어 붙여 학습 (표준화는 이미 돼 있어 사실상 그대로 통과)"},
          stack=stk_wf + [f"{MODEL} · Model.fit", "sklearn · Pipeline(StandardScaler → Ridge)"], nodes=["fit"], edges=[("scale", "fit")],
          formula="(w, b) = argmin Σ_i (y_i − b − Σ_j w_j z_ij)² + α Σ_j w_j²   →   w = (Z̃ᵀZ̃ + αI)⁻¹ Z̃ᵀỹ,   b = ȳ − z̄ᵀw    (Z̃, ỹ = 열 평균을 뺀 것; 절편 b 에는 벌점 없음)",
          tensors=[{"type": "num", "title": "모델 안 두 번째 표준화", "items": [{"label": "평균′ 의 최대 |값|", "value": f"{np.max(np.abs(rd['inner_mean'])):.1e}"},
                                                                     {"label": "표준편차′ − 1 의 최대 |값|", "value": f"{np.max(np.abs(rd['inner_scale'] - 1)):.1e}"}, {"label": "alpha", "value": f"{a:g}"}]}],
          checks=[chk("모델 안 표준화 = 항등 (평균′ = 0, 표준편차′ = 1)", np.r_[rd["inner_mean"], rd["inner_scale"]], np.r_[np.zeros(K), np.ones(K)], 1e-12, kind="식")])
        Zc = Ztr - Ztr.mean(0); yc = ytr - ytr.mean()
        w_cf = np.linalg.solve(Zc.T @ Zc + a * np.eye(K), Zc.T @ yc); b_cf = float(ytr.mean() - Ztr.mean(0) @ w_cf)
        fit_tr = Ztr @ w + b; r2 = 1 - np.sum((ytr - fit_tr) ** 2) / np.sum(yc ** 2)
        order = np.argsort(-np.abs(w))
        code_cf = (
            "import numpy as np\n\n"
            f"# 마지막 폴드의 학습 표 z ({len(tr)}행 × {K}열)와 정답 y — 실행 기록 그대로\n"
            f"names = {feats!r}\n"
            "Z = np.array([\n" + "".join(f"    {py_list(r_)},\n" for r_ in Ztr) + "])\n"
            f"y = np.array({py_list(ytr)})\n"
            f"alpha = {a!r}\n\n"
            "Zc = Z - Z.mean(axis=0)            # 열 평균을 뺌 (절편은 벌점 없이 따로)\n"
            "yc = y - y.mean()\n"
            "w = np.linalg.solve(Zc.T @ Zc + alpha * np.eye(Z.shape[1]), Zc.T @ yc)   # (ZᵀZ + αI) w = Zᵀy\n"
            "b = y.mean() - Z.mean(axis=0) @ w\n\n"
            "for name, v in zip(names, w):\n"
            "    print(name, f\"{v:.6g}\")\n"
            "print(\"b\", f\"{b:.6g}\")\n")
        exp_cf = "\n".join(f"{n_} {v_:.6g}" for n_, v_ in zip(feats, w)) + f"\nb {b:.6g}"
        lab_cf = {"title": "Ridge 계수를 공식으로 직접 풀기 (numpy)", "desc": "학습 표로 닫힌 해를 풀어 sklearn 이 구한 계수·절편과 비교합니다. numpy 가 필요해 Python Tutor 대신 Colab 등에 붙여 실행하세요.",
                  **run_lab(code_cf, exp_cf, pure=False)}
        S("6. Ridge 학습",
          f"학습 표 {len(tr)}행 × {K}열로 닫힌 해를 직접 풀어 sklearn 의 계수와 맞춰 봅니다. 계수 w_j = '재료 j 가 평소보다 1 표준편차 클 때 예측이 얼마나 움직이는가'. "
          f"절편 b = 정답 평균({pct(ytr.mean(), 3)}) — 재료가 모두 평소 값이면 예측은 학습 기간 평균 변화율입니다. 학습 구간 설명력 R² = {r2:.3f}.",
          say=(f"학습 표 {len(tr)}행으로 그 공식을 직접 풀어서, 사이킷런이 구한 계수와 맞춰 봅니다. 계수 하나는, 그 재료가 평소보다 표준편차 한 칸 클 때 예측이 얼마나 움직이는지를 뜻합니다. "
               f"가장 큰 계수는 {nm(feats[order[0]])}의 {w[order[0]]:+.4f}입니다. 절편은 정답의 평균, {pct(ytr.mean(), 3)}입니다. 재료가 모두 평소 값이면 예측은 학습 기간의 평균 변화율이 된다는 뜻입니다. "
               f"학습 구간의 설명력, R²는 {r2:.3f}입니다."),
          lab=lab_cf,
          stack=stk_wf + ["sklearn · Ridge.fit (닫힌 해를 numpy 로 다시 계산)"], nodes=["fit"],
          tensors=[{"type": "table", "title": "계수 (크기 순)", "columns": ["열", "뜻", "계수 w", "닫힌 해"], "rows": [[feats[j], dsc(feats[j]), f"{w[j]:+.6f}", f"{w_cf[j]:+.6f}"] for j in order], "split": 2 if K > 14 else 1},
                   {"type": "num", "title": "", "items": [{"label": "절편 b", "value": pct(b, 4)}, {"label": "정답 평균", "value": pct(ytr.mean(), 4)}, {"label": "학습 R²", "value": f"{r2:.3f}"}]}],
          checks=[chk("닫힌 해 w = sklearn coef_", w_cf, w, 1e-12), chk("닫힌 해 b = sklearn intercept_", b_cf, b, 1e-14)])
        alphas = np.logspace(-2, 4, 25)
        path = np.array([np.linalg.solve(Zc.T @ Zc + al * np.eye(K), Zc.T @ yc) for al in alphas])
        r2p = [1 - np.sum((yc - Zc @ wp) ** 2) / np.sum(yc ** 2) for wp in path]
        top = order[:6]; cols6 = ["#3b6fb6", "#c8710f", "#2f8f5b", "#6a3fa0", "#b3261e", "#8a93a3"]
        ia = int(np.argmin(np.abs(np.log10(alphas) - np.log10(a))))
        S("6. Ridge 학습",
          f"<b>alpha 의 역할</b>: 같은 학습 표로 alpha 만 바꿔 계수를 다시 풀었습니다(닫힌 해). alpha 가 작으면 계수가 커지고 학습 R² 가 올라가지만(학습 표에 과하게 맞춤), "
          f"alpha 가 크면 모든 계수가 0 으로 줄어 예측이 평균으로 모입니다. 이 설정은 alpha = {a:g}(세로선) — 여기서 학습 R² 는 {r2:.3f}.",
          say=("알파의 역할을 봅니다. 같은 학습 표로 알파만 바꿔 계수를 다시 풀었습니다. 알파가 작으면 계수가 커지고 학습 표에 과하게 맞춰지며, "
               f"알파가 크면 모든 계수가 0 쪽으로 줄어서 예측이 평균으로 모입니다. 이 설정의 알파는 {a:g}로 세로선 위치이고, 이때 학습 R²는 {r2:.3f}입니다."),
          stack=["numpy · alpha 별 닫힌 해 (설명용 계산, 엔진은 alpha 하나만 씀)"], nodes=["fit"],
          tensors=[{"type": "line", "title": "alpha 에 따른 계수 (현재 크기 상위 6개)", "x": [f"{al:.3g}" for al in alphas], "xlabel": "alpha (로그 간격)",
                    "series": [{"name": feats[j], "color": cols6[n], "y": [num(v_) for v_ in path[:, j]]} for n, j in enumerate(top)], "height": 230,
                    "vlines": [{"i": ia, "label": f"alpha = {a:g}", "color": "#1c2027"}]},
                   {"type": "line", "title": "alpha 에 따른 학습 R²", "x": [f"{al:.3g}" for al in alphas], "xlabel": "alpha", "series": [{"name": "학습 R²", "color": "#2f8f5b", "y": [num(v_) for v_ in r2p]}],
                    "height": 170, "vlines": [{"i": ia, "label": "", "color": "#1c2027"}]}])
        contrib = w * z; yhat = b + contrib.sum(); oc = np.argsort(-np.abs(contrib))
        code_pred = (
            f"# 결정일 {t.date()}: 재료 이름, 계수 w, 표준화된 값 z, 절편 b — 실행 기록 그대로\n"
            f"names = {feats!r}\n"
            f"w = {py_list(w)}\n"
            f"z = {py_list(z)}\n"
            f"b = {float(b)!r}\n\n"
            "yhat = b\n"
            "for name, wi, zi in zip(names, w, z):\n"
            "    yhat += wi * zi               # 기여 = 계수 × z\n\n"
            "print(\"예측(%)\", f\"{yhat * 100:.6g}\")\n")
        lab_pred = {"title": "예측을 가중합으로 직접 계산", "kind": "식", "desc": "절편에서 시작해 재료마다 계수 × z 를 더합니다. 엔진 예측(model.predict)과 같아야 합니다.",
                    **run_lab(code_pred, f"예측(%) {float(cap['pred'].loc[t]) * 100:.6g}", pure=True)}
        S("7. 예측",
          f"결정일 {t.date()} 의 z 에 계수를 곱해 더하고 절편을 더하면 예측입니다. 기여 = w × z — 어느 재료가 예측을 위·아래로 얼마나 밀었는지 보여 줍니다. "
          f"예측 <b>{pct(yhat, 3)}</b>, 실제 {pct(y.loc[t], 3)}.",
          code=C(MODEL, src.span(MODEL, "def predict(self, X):", "return self.pipe.predict_proba(X)[:, 1]"), src.line(MODEL, "return self.pipe.predict(X)"), "Model.predict", 2, 2),
          say=(f"이제 예측입니다. 결정일 {t.date()}의 표준화된 재료에 계수를 곱해 모두 더하고, 절편을 더합니다. 표의 기여는 계수 곱하기 z로, "
               f"어느 재료가 예측을 위나 아래로 얼마나 밀었는지를 보여 줍니다. 가장 크게 민 재료는 {nm(feats[oc[0]])}입니다. 예측은 {pct(yhat, 3)}, 실제는 {pct(y.loc[t], 3)}였습니다."),
          notes={src.line(MODEL, "return self.pipe.predict(X)"): f"표준화(사실상 그대로) → b + Σ w·z = {pct(yhat, 4)}"},
          lab=lab_pred,
          stack=stk_wf + [f"{MODEL} · Model.predict"], nodes=["pred"], edges=[("fit", "pred")],
          formula=f"ŷ = b + Σ_j w_j z_j = {pct(b, 4)} + ({pct(contrib.sum(), 4)}) = {pct(yhat, 4)}",
          tensors=[{"type": "table", "title": "기여 (크기 순, 상위 12개)", "columns": ["열", "z", "w", "기여 w·z"], "rows": [[feats[j], f"{z[j]:+.3f}", f"{w[j]:+.6f}", pct(contrib[j], 4)] for j in oc[:12]]}],
          checks=[chk("식 확인: ŷ = b + Σ w·z (sklearn 계수로) = 엔진 예측", yhat, cap["pred"].loc[t], 1e-14, kind="식")])
    else:
        L = cap["lgbm"]; dm = L["dump"]; trees = dm["tree_info"]; lr = p["learning_rate"]; lam = p["reg_lambda"]
        n_tr = cap["n_tr"]; yt = ytr[:n_tr]; init = float(np.mean(yt)); fn = feats; nT = len(trees); stg = L["staged_tr"]

        def leaves(node, out):
            if "leaf_value" in node: out[node["leaf_index"]] = node; return out
            leaves(node["left_child"], out); leaves(node["right_child"], out); return out

        def inner(node, out):
            if "leaf_value" in node: return out
            out.append(node); inner(node["left_child"], out); inner(node["right_child"], out); return out

        def depth(node):
            return 0 if "leaf_value" in node else 1 + max(depth(node["left_child"]), depth(node["right_child"]))
        nl_cnt = Counter(t_["num_leaves"] for t_ in trees)
        roots = [t_["tree_structure"].get("internal_count", t_["tree_structure"].get("leaf_count")) for t_ in trees]   # 나무마다 뽑힌 행 수
        all_leaves = [nd for t_ in trees for nd in leaves(t_["tree_structure"], {}).values()]
        d_max = max(depth(t_["tree_structure"]) for t_ in trees); l_min = min(nd["leaf_count"] for nd in all_leaves)
        same_sub = p["subsample"] == p["colsample_bytree"]
        # 이 기록을 만들 때 LightGBM 이 쓴 시드. 2026-11-10 전 기록에는 없다(그때는 모델 파일이 0 으로 고정). 그 뒤로는 엔진이 폴드마다 정한 시드.
        lgb_seed = int(cap["fold_seed"]) if cap.get("fold_seed") is not None else 0
        seed_txt = (f"시드 {lgb_seed} — 엔진이 실험 시드 {cfg['seed']} 와 이 폴드의 첫 테스트 결정일로 폴드 시드를 정해 numpy 난수를 맞추고, 모델이 그 난수에서 뽑은 값" if cap.get("fold_seed") is not None
                    else f"시드 0 — 이 기록을 만든 때에는 모델 파일에 고정, 엔진 시드 {cfg['seed']} 와 별개였음(2026-11-10 부터는 엔진이 폴드마다 정한 시드를 씀)")
        S("6. LightGBM 학습",
          f"LightGBM 은 <b>얕은 나무를 차례로 더하는</b> 그래디언트 부스팅입니다(목적함수 {dm['objective']} = 제곱오차). 시작값 F₀ = 학습 정답 {n_tr}개의 평균({pct(init, 3)}). "
          f"나무 m 은 지금까지의 예측이 못 맞힌 만큼(잔차 y − F_(m−1))을 맞히도록 만들어지고, 잎마다 '잔차 합 ÷ (행 수 + λ)' 에 학습률 {lr} 를 곱한 것이 그 잎 값이 되어 예측에 더해집니다. "
          f"이 설정의 나무: 깊이 ≤ {p['max_depth']}, 잎 ≤ {p['num_leaves']}, 잎마다 행 ≥ {p['min_child_samples']}, 나무마다 행 {p['subsample']:.0%}·열 {p['colsample_bytree']:.0%} 무작위 추출({seed_txt}), {p['n_estimators']}그루. "
          "실제로 만들어진 나무: " + " · ".join(f"잎 {k_}개 {v_}그루" for k_, v_ in sorted(nl_cnt.items(), reverse=True))
          + f", 나무마다 뽑힌 행 {min(roots)}~{max(roots)}개(전체 {n_tr}행 — 정확히 {p['subsample']:.0%} 가 아니라 나무마다 다름).",
          code=C(MODEL, src.span(MODEL, "def fit(self, X, y, X_val=None, y_val=None):", "self.est.fit(X, y)"), src.line(MODEL, "self.est.fit(X, y)"), "Model.fit", 4, 6),
          say=(f"라이트지비엠은 얕은 나무를 차례로 더해 가는 그래디언트 부스팅입니다. 시작값은 학습 정답 {n_tr}개의 평균, {pct(init, 3)}입니다. "
               "나무마다 지금까지의 예측이 못 맞힌 만큼, 곧 잔차를 맞히도록 만들고, "
               f"잎마다 잔차의 합을 행 수에 람다 {lam:g}을 더한 값으로 나눈 뒤, 학습률 {lr}을 곱해 예측에 더합니다. "
               f"질문은 많아야 {p['max_depth']}번, 잎 하나의 행은 {p['min_child_samples']}개 이상입니다. "
               + (f"나무마다 행과 열을 {p['subsample']:.0%}씩 무작위로 뽑는데, " if same_sub else f"나무마다 행은 {p['subsample']:.0%}, 열은 {p['colsample_bytree']:.0%}를 무작위로 뽑는데, ")
               + f"실제로 뽑힌 행은 나무마다 {min(roots)}개에서 {max(roots)}개로 달랐습니다."),
          notes={src.line(MODEL, "kw = dict(self.params, subsample_freq=1, verbose=-1, random_state=seed, n_jobs=1)"): f"subsample_freq=1: 나무마다 행을 새로 뽑음(이 폴드: 나무마다 {min(roots)}~{max(roots)}행) · random_state: {seed_txt}",
                 src.line(MODEL, 'cls = lgb.LGBMRegressor if self.target_type == "regression" else lgb.LGBMClassifier'): f"정답이 수익률(숫자) → LGBMRegressor, 목적함수 {dm['objective']}(제곱오차)",
                 src.line(MODEL, "if X_val is not None and len(X_val) > 10:"): "검증이 없어(X_val = None) 조기 종료(early_stopping) 갈래로 가지 않음",
                 src.line(MODEL, "self.est.fit(X, y)"): f"학습 표 {n_tr}행 × {K}열 → 나무 {nT}그루를 모두 만듦"},
          stack=stk_wf + [f"{MODEL} · Model.fit", "lightgbm · LGBMRegressor.fit"], nodes=["fit"], edges=[("scale", "fit")],
          formula=f"F_0 = 평균(y),   F_m(x) = F_(m−1)(x) + 잎 값_m(x),   잎 값 = 학습률 × Σ_(i∈잎)(y_i − F_(m−1)(x_i)) ÷ (잎의 행 수 + λ)   (제곱오차; λ = reg_lambda = {lam:g})",
          tensors=[{"type": "table", "title": "실제 파라미터 (월간 기본값 + 설정)", "columns": ["파라미터", "값", "뜻"],
                    "rows": [["n_estimators", p["n_estimators"], "나무 수"], ["learning_rate", lr, "나무 하나의 반영 비율"],
                             ["max_depth / num_leaves", f"{p['max_depth']} / {p['num_leaves']}", f"질문 최대 {p['max_depth']}번 → 잎 최대 {min(2 ** p['max_depth'], p['num_leaves'])}개"],
                             ["min_child_samples", p["min_child_samples"], "잎 하나의 최소 행"], ["subsample (매 나무)", p["subsample"], f"행 추출 비율 — 시드 {lgb_seed}"], ["colsample_bytree", p["colsample_bytree"], "열 추출 비율"],
                             ["reg_lambda", lam, "잎 값을 0 쪽으로 당기는 벌점"]]}],
          checks=[chk(f"나무 {nT}그루가 설정을 지킴: 가장 깊은 깊이 {d_max} ≤ {p['max_depth']} · 가장 많은 잎 {max(nl_cnt)} ≤ {p['num_leaves']} · 가장 작은 잎의 행 {l_min} ≥ {p['min_child_samples']}",
                      [d_max <= p["max_depth"], max(nl_cnt) <= p["num_leaves"], l_min >= p["min_child_samples"]], [1, 1, 1], 0, kind="식")])

        def pct_daily(j):   # 일간 변수의 전 관측 대비 변화율 열인가
            return any(fn[j] == col_of[v_] and cap["variables"][v_]["transform"] == "pct" and cap["variables"][v_]["freq"] == "D" for v_ in cap["variables"])

        def orig_say(j, thz):
            v_ = thz * sd[j] + mu[j]
            return f"하루 변화율 {v_ * 100:+.2f}%" if pct_daily(j) else g(v_, 4)

        def node_rows(node, depth_=0, out=None):
            out = [] if out is None else out
            if "leaf_value" in node:
                out.append(["　" * depth_ + f"잎 {node['leaf_index']}", "", "", f"{node['leaf_value']:+.6f}", node["leaf_count"]]); return out
            j = node["split_feature"]; thz = node["threshold"]
            out.append(["　" * depth_ + f"{fn[j]} ≤ {thz:+.3f} ?", f"원래 단위 {g(thz * sd[j] + mu[j], 4)}" + (f" ({orig_say(j, thz)})" if pct_daily(j) else ""), f"이득 {node['split_gain']:.2e}", "", node["internal_count"]])
            node_rows(node["left_child"], depth_ + 1, out); node_rows(node["right_child"], depth_ + 1, out); return out
        # 기준의 후보 = LightGBM 의 구간(bin) 경계. 같은 학습 표·같은 파라미터로 lightgbm.Dataset 을 다시 만들어(학습은 하지 않음)
        # 열마다 구간 수와 행마다 구간 번호를 꺼낸 뒤, 나무의 기준이 모두 구간 경계(이웃한 두 구간 사이의 한가운데 또는 0 경계 ±1e-35)인지 본다.
        Zt = Ztr[:n_tr]; Bn = None; bin_why = ""
        import os, tempfile
        import lightgbm as lgb
        # 구간(bin)은 LightGBM 버전·환경에 따라 다를 수 있어 음성에는 쓰지 않는다(화면 설명·검증 줄에만). 다시 만들지 못하면 그 부분만 건너뜀.
        nb = None
        try:
            dsb = lgb.Dataset(Zt, label=yt, params=dict(p, subsample_freq=1, verbose=-1, random_state=lgb_seed, n_jobs=1, objective=dm["objective"]), free_raw_data=False).construct()
            nb = [dsb.feature_num_bin(j_) for j_ in range(K)]
        except Exception as e_:
            bin_why = f"lightgbm.Dataset 으로 구간을 다시 만들지 못함({type(e_).__name__})"; print("  ⚠", bin_why, e_)
        try:   # 행마다 구간 번호(비공개 API _dump_text)
            if nb is None:
                raise RuntimeError(bin_why)
            with tempfile.TemporaryDirectory() as td_:
                fp_ = os.path.join(td_, "bins.txt"); dsb._dump_text(fp_)
                rows_txt = [l_ for l_ in Path(fp_).read_text(encoding="utf-8").split("\n") if l_.count(",") >= K - 1 and l_.replace(",", "").replace(" ", "").isdigit()]
            Bn = np.array([[int(v_) for v_ in r_.strip().rstrip(",").split(",")] for r_ in rows_txt])
            assert Bn.shape == (n_tr, K), f"구간 번호 표 모양 {Bn.shape} ≠ ({n_tr}, {K})"
        except Exception as e_:
            Bn = None; bin_why = bin_why or f"lightgbm.Dataset._dump_text(비공개 API) 로 행마다 구간 번호를 꺼내지 못함({type(e_).__name__})"
            print("  ⚠", bin_why, e_)
        n_nodes = n_mid = n_zero = n_zero_pos = 0
        for t_ in trees:
            for nd in inner(t_["tree_structure"], []):
                n_nodes += 1; jj = nd["split_feature"]; col = Zt[:, jj]; th_ = nd["threshold"]
                lo_, hi_ = col[col <= th_], col[col > th_]
                if not (len(lo_) and len(hi_)):
                    continue
                if abs(th_) < 1e-30:
                    n_zero += 1; n_zero_pos += th_ > 0
                elif abs(th_ - (lo_.max() + hi_.min()) / 2) < 1e-12 and (Bn is None or Bn[col == lo_.max(), jj][0] != Bn[col == hi_.min(), jj][0]):
                    n_mid += 1
        bin_small = Counter(); bin_small_ok = True; bin_sum_ok = True   # 구간마다 행 수: 가장 작은(비어 있지 않은) 구간, 그 위치
        if Bn is not None:
            for jj in range(K):
                cnt_ = np.bincount(Bn[:, jj]); nz_ = np.flatnonzero(cnt_ > 0); zb_ = [b_ for b_ in range(nz_.min(), nz_.max() + 1) if cnt_[b_] == 0]
                bin_small[int(cnt_[nz_].min())] += 1; bin_sum_ok &= int(cnt_.sum()) == n_tr
                bin_small_ok &= all(b_ in (nz_.min(), nz_.max()) or any(abs(b_ - z_) == 1 for z_ in zb_) for b_ in nz_ if cnt_[b_] < 3)
        zero_gap = (sum(1 for jj in range(K) if (lambda b_: len(set(b_)) < max(b_) - min(b_) + 1)(Bn[:, jj])
                        and Bn[Zt[:, jj] < 0, jj].max() < Bn[Zt[:, jj] > 0, jj].min() - 1) if Bn is not None else 0)
        nb_c = Counter(nb) if nb else Counter()
        r0 = trees[0]["tree_structure"]; j0r = r0["split_feature"]
        S("6. LightGBM 학습",
          "첫 나무입니다. 각 마디는 '열 j 의 z ≤ 기준?' 질문이고, 왼쪽이 '예'. "
          + "<b>기준의 후보</b>: LightGBM 은 학습 전에 열마다 값을 구간(bin)으로 묶고 구간 경계만 기준 후보로 씁니다. "
          + (f"이 학습 표 {n_tr}행은 열마다 구간 {nb_c.most_common(1)[0][0]}개({nb_c.most_common(1)[0][1]}열; 최소 {min(nb)}개)로 묶이고(feature_num_bin), 0 자리는 비어 있는 구간 하나로 따로 둡니다(덤프의 기준 ±1e-35). " if nb else
             f"(구간 수 확인은 건너뜀 — {bin_why}.) ")
          + (f"구간은 행 3개(min_data_in_bin 기본값)를 목표로 묶지만 끝 구간과 0 구간 바로 옆 구간은 더 적을 수 있습니다 — 이 표에서 가장 작은 구간이 "
             + " · ".join(f"{k_}행인 열 {v_}개" for k_, v_ in sorted(bin_small.items())) + ("(1~2행 구간은 모두 양 끝이나 0 구간 바로 옆)" if bin_small_ok else "") + f"; 빈 0 구간은 {zero_gap}열 모두. "
             f"이 폴드의 마디 {n_nodes}개 기준은 모두 구간 경계입니다: {n_mid}개는 이웃한 두 구간 사이의 한가운데, {n_zero}개는 0 경계(+1e-35 {n_zero_pos}개 · −1e-35 {n_zero - n_zero_pos}개 — 표준화했으므로 학습 평균 자리). " if Bn is not None else
             f"기준은 이웃한 두 학습 값 사이에 놓입니다: 이 폴드의 마디 {n_nodes}개 중 {n_mid}개는 두 값의 한가운데, {n_zero}개는 0 경계(= 학습 평균 자리). (행마다 구간 번호 확인은 건너뜀 — {bin_why}.) ")
          + ("(구간은 같은 학습 표로 lightgbm.Dataset 을 다시 만들어 확인했습니다 — 학습은 하지 않음.) " if nb else "")
          + f"그 후보 가운데 이 나무에 뽑힌 행·열에서 잎마다 행 ≥ {p['min_child_samples']} 를 지키며 <b>분할 이득</b>이 가장 큰 곳을 고릅니다: 이득 = G_L² ÷ (n_L + λ) + G_R² ÷ (n_R + λ) − G² ÷ (n + λ) "
          "(G = 그 마디 행들의 잔차 합, n = 행 수; λ = 0 이면 그 마디를 둘로 나눠 잔차 제곱합이 줄어드는 양과 같음). λ = 1 인 이 식이 LightGBM 의 split_gain 과 같은지는 다음 화면에서 맞춰 봅니다. "
          f"첫 나무의 잎 값에는 시작값 F₀ = {pct(init, 3)} 가 더해져 있습니다. '행 수' 는 이 나무에 뽑힌 행 수입니다(전체 {n_tr}행 중 {roots[0]}행).",
          code=C(TC, [src.line(TC, 'out["lgbm"] = {"dump": bst.dump_model()')], src.line(TC, 'out["lgbm"] = {"dump": bst.dump_model()'), "capture · LightGBM 기록", 3, 5),
          say=("첫 나무를 펼쳐 봅니다. 마디마다 어떤 재료의 z가 기준 이하인지 묻고, 그렇다면 왼쪽으로 갑니다. "
               + f"라이트지비엠은 학습 표 {n_tr}행을 열마다 여러 구간으로 미리 묶고, 그 경계만 기준 후보로 씁니다. "
               f"값 0의 자리도 늘 경계라서, 이 폴드의 기준 가운데 {n_zero}개가 0입니다. "
               + "그 후보 가운데, 이 나무에 뽑힌 행과 열에서 분할 이득이 가장 큰 곳을 고릅니다. "
               f"첫 질문은 {nms(fn[j0r])}의 z가 {r0['threshold']:+.3f} 이하인지 묻습니다. 이 기준은 원래 단위로 {orig_say(j0r, r0['threshold'])}입니다."),
          notes={src.line(TC, 'out["lgbm"] = {"dump": bst.dump_model()'): f"booster.dump_model(): 나무 {nT}그루의 구조(마디의 열·기준·분할 이득·행 수, 잎 값)를 LightGBM 에서 그대로 꺼냄 → 아래 표"},
          stack=stk_wf + ["lightgbm · booster.dump_model() — 나무 0"], nodes=["fit"],
          tensors=[{"type": "table", "title": "나무 1 (z 단위 기준값, 원래 단위 병기)", "columns": ["마디", "원래 단위", "분할 이득", "잎 값", "행 수"], "rows": node_rows(r0)}],
          checks=[chk(f"마디 {n_nodes}개의 기준 = LightGBM 구간 경계: 이웃한 두 구간 사이의 한가운데({n_mid}) 또는 0 경계({n_zero})", n_mid + n_zero, n_nodes, 0, kind="재실행") if Bn is not None else
                  skip(f"마디 {n_nodes}개의 기준 = LightGBM 구간 경계 (이웃한 두 구간 사이의 한가운데 또는 0 경계)", bin_why, "재실행")]
                 + ([chk(f"열 {K}개 모두 0 자리에 빈 구간이 있음 (lightgbm.Dataset)", zero_gap, K, 0, kind="재실행"),
                     chk(f"구간 번호로 센 행 수의 합 = {n_tr} (열마다) · 3행 미만 구간은 모두 양 끝이나 0 구간 옆 (lightgbm.Dataset)", [float(bin_sum_ok), float(bin_small_ok)], [1.0, 1.0], 0, kind="재실행")] if Bn is not None else
                    [skip(f"열 {K}개 모두 0 자리에 빈 구간이 있음 (lightgbm.Dataset)", bin_why, "재실행"),
                     skip(f"구간 번호로 센 행 수의 합 = {n_tr} (열마다) · 3행 미만 구간은 모두 양 끝이나 0 구간 옆 (lightgbm.Dataset)", bin_why, "재실행")]))
        # 잎 값·분할 이득 공식 검증: 뽑힌 행 수 = 그 잎의 학습 행 수인 잎만(모두 뽑힌 잎) — 어느 행이 빠졌는지는 LightGBM 내부라 알 수 없음
        # 학습 행마다 나무 dump 를 직접 따라 내려가 도착 잎을 찾는다(LightGBM 의 pred_leaf 를 쓰지 않음 → 아래에서 pred_leaf 와 대조)
        def leaf_at(node, zrow):
            while "leaf_value" not in node:
                node = node["left_child"] if zrow[node["split_feature"]] <= node["threshold"] else node["right_child"]
            return node["leaf_index"]
        leaf_own = np.array([[leaf_at(t_["tree_structure"], Zt[i_]) for t_ in trees] for i_ in range(n_tr)])
        F = np.full(n_tr, init); ver_h, ver_m, n_all_l = [], [], 0; ex_rows = []; F_hist = []
        gain_h, gain_m, gain_rows, gain_lab = [], [], [], None
        for m_, tinfo in enumerate(trees):
            lv = leaves(tinfo["tree_structure"], {}); rows_ = leaf_own[:, m_]; full = set()
            for li, nd in lv.items():
                n_all_l += 1
                idx = np.flatnonzero(rows_ == li)
                if len(idx) == nd["leaf_count"] and len(idx) > 0:
                    full.add(li)
                    hv = (init if m_ == 0 else 0.0) + lr * np.sum(yt[idx] - F[idx]) / (len(idx) + lam)
                    ver_h.append(hv); ver_m.append(nd["leaf_value"])
                    if len(ex_rows) < 6: ex_rows.append([m_ + 1, li, len(idx), f"{np.sum(yt[idx] - F[idx]):+.6f}", f"{hv:+.8f}", f"{nd['leaf_value']:+.8f}"])
            for nd in inner(tinfo["tree_structure"], []):
                l_, r_ = nd["left_child"], nd["right_child"]
                if "leaf_value" in l_ and "leaf_value" in r_ and l_["leaf_index"] in full and r_["leaf_index"] in full:
                    il = np.flatnonzero(rows_ == l_["leaf_index"]); ir = np.flatnonzero(rows_ == r_["leaf_index"])
                    GL = np.sum(yt[il] - F[il]); GR = np.sum(yt[ir] - F[ir])
                    gh = GL ** 2 / (len(il) + lam) + GR ** 2 / (len(ir) + lam) - (GL + GR) ** 2 / (len(il) + len(ir) + lam)
                    gain_h.append(gh); gain_m.append(nd["split_gain"])
                    gain_rows.append([m_ + 1, f"{fn[nd['split_feature']]} ≤ {nd['threshold']:+.3f}", f"{len(il)} / {len(ir)}", f"{gh:.6e}", f"{nd['split_gain']:.6e}"])
                    if gain_lab is None and m_ > 0:
                        gain_lab = lab_leaf_gain(m_, nd, yt[il], stg[m_ - 1][il], yt[ir], stg[m_ - 1][ir], lr, lam, fn[nd["split_feature"]])
            F = F + np.array([lv[r]["leaf_value"] for r in rows_]) - (init if m_ == 0 else 0.0)
            F_hist.append(F.copy())
        S("6. LightGBM 학습",
          f"<b>잎 값 공식 확인</b>: 잎 값 = 학습률 × (잎에 든 행의 잔차 합) ÷ (행 수 + λ). 그런데 나무마다 행을 {p['subsample']:.0%} 비율로 뽑아 쓰고(이 폴드에서 나무마다 {min(roots)}~{max(roots)}행) "
          "<b>어느 행이 뽑혔는지는 LightGBM 내부라 기록되지 않습니다</b>. "
          f"그래서 '그 잎에 들어가는 학습 행이 모두 뽑힌' 잎(행 수가 같은 잎)에서만 손으로 확인할 수 있습니다: 전체 잎 {n_all_l}개 중 <b>{len(ver_h)}개</b>. "
          f"두 자식 잎이 모두 그런 마디 {len(gain_h)}개에서는 <b>분할 이득</b>도 손으로 계산해 맞춰 봅니다(두 번째 표).",
          code=C(TC, [src.line(TC, '"leaf_tr": bst.predict(cap["Xtr"][: cap["n_tr"]], pred_leaf=True)')], src.line(TC, '"leaf_tr": bst.predict(cap["Xtr"][: cap["n_tr"]], pred_leaf=True)'), "capture · LightGBM 기록", 4, 3),
          say=("잎 값 공식을 손으로 확인합니다. 잎 값은 잔차 합을, 행 수에 람다를 더한 값으로 나누고 학습률을 곱한 것입니다. "
               "그런데 나무마다 어느 행을 뽑았는지는 기록되지 않습니다. "
               f"그래서 학습 행이 모두 뽑힌 잎에서만 확인할 수 있는데, 그런 잎은 {len(ver_h)}개입니다. 전체 잎은 {n_all_l}개입니다. "
               f"두 자식 잎이 모두 그런 마디 {len(gain_h)}개에서는 분할 이득도 계산했습니다. 둘 다 라이트지비엠의 값과 32비트 계산 오차 안에서 같습니다."),
          notes={src.line(TC, '"leaf_tr": bst.predict(cap["Xtr"][: cap["n_tr"]], pred_leaf=True)'): f"pred_leaf: 학습 행 {n_tr}개 모두가 나무마다 어느 잎에 도착하는지 (뽑힌 행만이 아님). 영상은 dump 를 직접 따라가 같은 잎 번호를 얻어 쓰고, 이 기록과 대조합니다 → 잎의 행 수와 같으면 '모두 뽑힌 잎'"},
          stack=["numpy · 잎 값·분할 이득 다시 계산 (나무 순서대로 F 를 갱신하며)"], nodes=["fit"],
          formula="잎 값_m = 학습률 × Σ_(i∈잎) (y_i − F_(m−1)(x_i)) ÷ (n_잎 + λ)   (+ F₀, 첫 나무만) · 이득 = G_L² ÷ (n_L + λ) + G_R² ÷ (n_R + λ) − (G_L + G_R)² ÷ (n_L + n_R + λ)",
          tensors=[{"type": "table", "title": "예: 확인한 잎 (처음 6개)", "columns": ["나무", "잎", "행 수", "잔차 합", "손 계산", "LightGBM"], "rows": ex_rows},
                   {"type": "table", "title": f"분할 이득을 확인한 마디 ({len(gain_rows)}개)", "columns": ["나무", "마디", "행 수 (왼 / 오)", "손 계산", "LightGBM"], "rows": gain_rows}],
          hint="제곱오차의 기울기는 g = F − y(잔차 y − F 와 부호가 반대)이고 헤세는 1 입니다. LightGBM 의 잎 값 = −학습률 × Σg ÷ (n + λ) 는 위의 '잔차 합' 식과 같습니다. "
               "LightGBM 은 이 기울기를 32비트 실수로 계산합니다. 그래서 64비트로 다시 계산한 값과 소수 9자리쯤에서 차이가 납니다(아래 잎 값 검증의 허용 오차 1e-8, 분할 이득은 상대 1e-5).",
          lab=gain_lab,
          checks=[chk(f"직접 따라간 도착 잎 번호 = LightGBM pred_leaf (학습 행 {n_tr} × 나무 {nT})", leaf_own.T, L["leaf_tr"].T, 0),
                  chk(f"잎 값 공식 = LightGBM 잎 값 ({len(ver_h)}개 잎)", ver_h, ver_m, 1e-8),
                  chk(f"분할 이득 공식 = LightGBM split_gain ({len(gain_h)}개 마디) (상대)", gain_h, gain_m, 1e-5, rel=True)])
        mse = [float(np.mean((yt - s_) ** 2)) for s_ in stg]; mse0 = float(np.mean((yt - init) ** 2)); r2_l = 1 - mse[-1] / mse0
        n_up = int(np.sum(np.diff([mse0] + mse) > 0))
        S("6. LightGBM 학습",
          f"나무가 늘수록 학습 표에서의 오차(평균제곱오차)가 줄어듭니다 — 이 폴드에서는 나무를 더할 때마다 " + ("매번 줄었습니다(늘어난 적 0번)" if n_up == 0 else f"줄었고, 늘어난 적은 {n_up}번입니다")
          + f": 나무 0개(평균만) {mse0:.2e} → {len(stg)}그루 {mse[-1]:.2e} (학습 R² {r2_l:.3f}). "
          "검증이 없으므로 몇 그루에서 멈출지는 설정값(n_estimators)이 정합니다 — 학습 오차가 계속 줄어드는 것은 일반화의 증거가 아닙니다. "
          "그래프의 값은 LightGBM 이 나무 앞쪽 몇 그루까지만 써서 낸 학습 행 예측으로 계산했고, 학습 행마다 나무 dump 를 직접 따라 내려가 잎 값을 쌓은 것과 같습니다(검증).",
          code=C(TC, [src.line(TC, '"staged_tr": np.array([bst.predict(cap["Xtr"][: cap["n_tr"]], num_iteration=i)')], src.line(TC, '"staged_tr": np.array([bst.predict(cap["Xtr"][: cap["n_tr"]], num_iteration=i)'), "capture · LightGBM 기록", 3, 4),
          say=(f"나무를 더할수록 학습 표의 평균제곱오차가 줄어듭니다. 평균만 쓸 때 {mse0:.5f}에서, {len(stg)}그루 뒤에는 {mse[-1]:.5f}가 되었고, 학습 R²는 {r2_l:.3f}입니다. "
               + ("이 폴드에서는 나무를 더할 때마다 매번 줄었습니다. " if n_up == 0 else "")
               + "하지만 검증 구간이 없으니 몇 그루에서 멈출지는 설정값이 정하고, 학습 오차가 줄었다고 새 자료에서도 잘 맞는다고 말할 수는 없습니다."),
          notes={src.line(TC, '"staged_tr": np.array([bst.predict(cap["Xtr"][: cap["n_tr"]], num_iteration=i)'): f"학습 행 {n_tr}개를 나무 1그루, 2그루, …, {len(stg)}그루까지만 써서 예측 → 그래프의 평균제곱오차"},
          stack=[f"lightgbm · predict(num_iteration = 1…{len(stg)}) — 학습 표"], nodes=["fit"],
          tensors=[{"type": "line", "title": "나무 수에 따른 학습 표 평균제곱오차", "x": [str(i) for i in range(0, len(stg) + 1)], "xlabel": "나무 수",
                    "series": [{"name": "학습 MSE", "color": "#3b6fb6", "y": [num(mse0, 5)] + [num(v_, 5) for v_ in mse]}], "height": 220}],
          checks=[chk(f"학습 행마다 dump 를 직접 따라가 쌓은 예측 = LightGBM predict (나무 1~{len(stg)}그루마다)", np.array(F_hist), stg, 1e-12)]
                 + ([chk("학습 MSE 가 늘어난 나무 수 = 0", n_up, 0, 0, kind="식")] if n_up == 0 else []))
        # 예측
        def walk(node):
            path = []
            while "leaf_value" not in node:
                j = node["split_feature"]; left = z[j] <= node["threshold"]
                path.append(f"{fn[j]} {z[j]:+.3f} {'≤' if left else '>'} " + ("0" if abs(node["threshold"]) < 1e-30 else f"{node['threshold']:+.2f}"))
                node = node["left_child"] if left else node["right_child"]
            return float(node["leaf_value"]), path
        tot = 0.0; prow = []; vals = []
        for m_, tinfo in enumerate(trees):
            lvv, path = walk(tinfo["tree_structure"]); tot += lvv; vals.append(lvv)
            if m_ < 6 or m_ >= len(trees) - 2: prow.append([m_ + 1, " → ".join(path), f"{lvv:+.6f}", pct(tot, 3)])
            elif m_ == 6: prow.append(["⋯", f"(나무 7 ~ {len(trees) - 2} 생략)", "", ""])
        same_pred = float(tot) == float(cap["pred"].loc[t])
        S("7. 예측",
          f"결정일 {t.date()} 의 z 로 {len(trees)}그루를 각각 따라 내려가 도착한 잎 값을 모두 더하면 예측입니다(첫 나무의 잎 값에 F₀ 포함). 예측 <b>{pct(tot, 3)}</b>, 실제 {pct(y.loc[t], 3)}.",
          code=C(MODEL, src.span(MODEL, "def predict(self, X):", "return self.est.predict_proba(X)[:, 1]"), src.line(MODEL, "return self.est.predict(X)"), "Model.predict", 2, 2),
          say=(f"이제 결정일 {t.date()}의 예측입니다. 표준화한 재료 z를 들고 {nT}그루를 하나씩 따라 내려가, 도착한 잎의 값을 모두 더합니다. 첫 나무의 잎 값에는 시작값이 들어 있습니다. "
               "표와 그림의 누적은 나무를 더할 때마다 쌓이는 합입니다. "
               f"이렇게 더한 예측은 {pct(tot, 3)}이고, 실제 수익률은 {pct(y.loc[t], 3)}였습니다. 손으로 더한 값은 엔진의 예측과 " + ("똑같습니다." if same_pred else "같습니다.")),
          notes={src.line(MODEL, "return self.est.predict(X)"): f"나무 {nT}그루의 잎 값 합 → {pct(tot, 4)}"},
          lab=lab_tree_sum(trees, z, t.date(), cap["pred"].loc[t]),
          stack=stk_wf + [f"{MODEL} · Model.predict"], nodes=["pred"], edges=[("fit", "pred")],
          formula=f"ŷ = Σ_m 잎값_m(z) = {pct(tot, 4)}",
          tensors=[{"type": "table", "title": "나무마다의 경로와 잎 값 (z 단위, 기준 0 = LightGBM 의 0 경계 ±1e-35)", "columns": ["나무", "질문과 답", "잎 값", "누적"], "rows": prow},
                   {"type": "line", "title": "누적 합 (나무 수에 따라)", "x": [str(i + 1) for i in range(len(vals))], "xlabel": "나무", "series": [{"name": "누적", "color": "#c8710f", "y": [num(v_) for v_ in np.cumsum(vals)]}], "height": 160}],
          checks=[chk("잎 값을 손으로 더한 예측 = 엔진 예측", tot, cap["pred"].loc[t], 1e-12)])
        ct = L["contrib_te"][-1]; base = ct[-1]; cc = ct[:-1]; oc = np.argsort(-np.abs(cc)); jc = int(oc[0])
        base_h = 0.0   # TreeSHAP 기준값 = 나무마다 잎 값을 '그 나무에 뽑힌 행 수' 로 가중평균한 것의 합
        for t_ in trees:
            rt = t_["tree_structure"]; nr = rt.get("internal_count", rt.get("leaf_count"))
            base_h += sum(nd["leaf_count"] / nr * nd["leaf_value"] for nd in leaves(rt, {}).values())
        mean_tr = float(np.mean(stg[-1]))
        # 민감도: 가장 크게 민 재료의 값을 학습 평균 바로 건너편(0.001%p 단위)으로 옮기면 예측이 얼마나 바뀌나 — 나무 dump 로 다시 계산(위에서 dump 합 = 엔진 예측 확인)
        def tree_sum(zz):
            s_ = 0.0
            for t_ in trees:
                n_ = t_["tree_structure"]
                while "leaf_value" not in n_:
                    n_ = n_["left_child"] if zz[n_["split_feature"]] <= n_["threshold"] else n_["right_child"]
                s_ += n_["leaf_value"]
            return s_
        nd_c = [nd for t_ in trees for nd in inner(t_["tree_structure"], []) if nd["split_feature"] == jc]
        nd_c0 = [nd for nd in nd_c if abs(nd["threshold"]) < 1e-30]
        x_c, mu_c = float(xt[jc]), float(mu[jc])
        x_alt = (np.ceil(mu_c * 1e5) / 1e5 if z[jc] <= 0 else np.floor(mu_c * 1e5) / 1e5)
        x_alt = x_alt if (x_alt - mu_c) * (x_c - mu_c) < 0 else x_alt + (1e-5 if z[jc] <= 0 else -1e-5)
        z_alt = z.copy(); z_alt[jc] = (x_alt - mu_c) / sd[jc]; pred_alt = tree_sum(z_alt)
        crossed = [nd["threshold"] for nd in nd_c if min(z[jc], z_alt[jc]) < nd["threshold"] <= max(z[jc], z_alt[jc])]   # 두 값 사이에 놓인 기준
        fragile = bool(nd_c0) and abs(z[jc]) < 0.1 and crossed and all(abs(th_) < 1e-30 for th_ in crossed)
        S("7. 예측",
          "어느 재료가 이 예측을 얼마나 밀었나: LightGBM 의 <b>TreeSHAP</b>(pred_contrib)은 예측을 '기준값 + 재료별 기여' 로 정확히 나눕니다(나무 구조로 계산한 섀플리 값). "
          f"기준값 {pct(base, 3)} 는 나무마다 잎 값을 '그 나무에 뽑힌 행 수' 로 가중평균해 {nT}그루를 더한 값입니다(검증). 뽑힌 행으로 가중하므로 학습 표 {n_tr}행 전체의 평균 예측({pct(mean_tr, 3)})과 조금 다릅니다. "
          "나무는 값의 크기가 아니라 기준보다 큰지 작은지만 보므로, z 가 0 에 가까운 재료도 기여가 클 수 있습니다. 기여의 단위는 %p(퍼센트포인트)입니다."
          + (f" <b>민감도</b>: {fn[jc]} 로 나눈 마디 {len(nd_c)}개 중 {len(nd_c0)}개가 0 경계(= 학습 평균 {pct(mu_c, 4)}) 입니다. 이 날 값 {pct(x_c, 3)} 는 평균보다 아주 조금 작아(z = {z[jc]:+.4f}) "
             f"그 {len(nd_c0)}개 마디에서 모두 '예' 쪽으로 갔습니다. 값이 {pct(x_alt, 3)}(평균보다 조금 큼, z = {z_alt[jc]:+.5f}) 였다면 예측은 {pct(pred_alt, 4)} 가 됩니다"
             f"(나무 dump 로 다시 계산 — 원래 값과 바꾼 값 사이에 있는 기준은 0 경계 {len(crossed)}개뿐, 검증). 시작값 F₀({pct(init, 4)})와 비슷하지만, 같아야 할 구조적 이유는 찾지 못했습니다." if fragile else ""),
          code=C(TC, [src.line(TC, '"raw_te": bst.predict(Xte, raw_score=True), "contrib_te": bst.predict(Xte, pred_contrib=True),')],
                 src.line(TC, '"raw_te": bst.predict(Xte, raw_score=True), "contrib_te": bst.predict(Xte, pred_contrib=True),'), "capture · LightGBM 기록", 2, 3),
          say=("트리 샵(TreeSHAP)은 이 예측을 기준값과 재료마다의 기여로 정확히 나눕니다. "
               f"기준값 {pct(base, 3)}는 나무마다 잎 값을 그 나무에 뽑힌 행 수로 가중평균해 더한 값이라, 학습 표 전체의 평균 예측 {pct(mean_tr, 3)}와 조금 다릅니다. "
               f"가장 크게 {'위로' if cc[jc] > 0 else '아래로'} 민 재료는 {nms(fn[jc], '로')}, 기여가 {cc[jc] * 100:+.2f} 퍼센트포인트입니다. "
               + (f"그런데 이 날 그 값의 z는 {z[jc]:+.3f}로 평소와 거의 같습니다. 그래도 기여가 큰 것은, 나무가 값의 크기가 아니라 기준의 어느 쪽인지만 보기 때문입니다. "
                  if abs(z[jc]) < 0.1 else f"이 날 그 값의 z는 {z[jc]:+.3f}입니다. ")
               + (f"이 재료로 나눈 마디 {len(nd_c)}개 가운데 {len(nd_c0)}개가 평균 자리, 곧 0 경계에 있어서, "
                  f"값이 {pct(x_c, 3)}가 아니라 {pct(x_alt, 3)}였다면 예측은 {pct(pred_alt, 4)}로 바뀝니다. 이 예측은 이렇게 작은 차이에 민감합니다." if fragile else "")),
          notes={src.line(TC, '"raw_te": bst.predict(Xte, raw_score=True), "contrib_te": bst.predict(Xte, pred_contrib=True),'): f"pred_contrib=True: 재료 {K}개의 기여 + 맨 끝에 기준값 (TreeSHAP) → 아래 표"},
          stack=["lightgbm · predict(pred_contrib=True)"], nodes=["pred"],
          formula="ŷ = 기준값 + Σ_j 기여_j   (TreeSHAP, Lundberg et al. 2020) · 기준값 = Σ_m Σ_잎 (n_잎 ÷ n_나무) × 잎 값_m   (n = 그 나무에 뽑힌 행 수)",
          tensors=[{"type": "table", "title": "재료별 기여 (크기 순, 상위 12개 — 단위 %p)", "columns": ["열", "뜻", "z", "기여"], "rows": [[fn[j], dsc(fn[j]), f"{z[j]:+.3f}", f"{cc[j] * 100:+.2f}%p"] for j in oc[:12]]}]
                   + ([{"type": "num", "title": f"민감도: {fn[jc]} 하나만 바꾸면", "items": [{"label": "이 날 값 (z)", "value": f"{pct(x_c, 3)} ({z[jc]:+.4f})"}, {"label": "학습 평균", "value": pct(mu_c, 4)},
                                                                                     {"label": "바꾼 값 (z)", "value": f"{pct(x_alt, 3)} ({z_alt[jc]:+.5f})"}, {"label": "예측", "value": f"{pct(tot, 3)} → {pct(pred_alt, 4)}"}]}] if fragile else []),
          checks=[chk("기준값 + 기여 합 = 예측 (TreeSHAP 의 덧셈 성질)", base + cc.sum(), tot, 1e-12, kind="식"), chk("기준값 = Σ_나무 Σ_잎 (뽑힌 행 수 비율 × 잎 값)", base_h, base, 1e-12)]
                 + ([chk(f"민감도: 원래 값과 바꾼 값 사이의 {fn[jc]} 기준 {len(crossed)}개가 모두 0 경계 (나머지 마디의 답은 그대로)", len([c_ for c_ in crossed if abs(c_) < 1e-30]), len(nd_c0), 0, kind="식")] if fragile else []))
        gain_s = np.zeros(K); split_s = np.zeros(K)
        for t_ in trees:
            for nd in inner(t_["tree_structure"], []):
                gain_s[nd["split_feature"]] += nd["split_gain"]; split_s[nd["split_feature"]] += 1
        gi = L["importance_gain"]; og = np.argsort(-gi); unused = [fn[j] for j in range(K) if L["importance_split"][j] == 0]
        S("7. 예측",
          f"<b>중요도(gain)</b>: 이 폴드 모델의 변수 중요도 = 그 열로 나눈 마디의 분할 이득을 {nT}그루 전체에서 더한 값입니다('쓰인 횟수' = 그 열로 나눈 마디 수). "
          "위의 SHAP 기여(이 날 하나)와 달리 학습 전체의 요약이고, 방향(+/−)은 없습니다. "
          + (f"이 폴드에서 한 번도 쓰이지 않은 열: {', '.join(unused)}. " if unused else "")
          + f"나무 dump 에서 직접 더해 엔진 값과 맞춰 봅니다. 결과 화면의 변수 중요도는 이 값을 폴드 {len(af)}개에서 평균한 것입니다(다음 화면).",
          code=C(MODEL, src.span(MODEL, "def feature_importance(self):", 'return self.est.booster_.feature_importance(importance_type="gain").astype(float)'), src.line(MODEL, 'return self.est.booster_.feature_importance(importance_type="gain").astype(float)'), "feature_importance", 2, 2),
          say=("이번에는 이 폴드 모델 전체의 중요도입니다. 재료마다, 그 재료로 나눈 마디의 분할 이득을 "
               f"{nT}그루에서 모두 더합니다. 이 폴드에서는 {nms(fn[og[0]], '이')} {gi[og[0]]:.3f}로 가장 크고, 마디 {int(L['importance_split'][og[0]])}개에서 쓰였습니다. "
               "앞의 기여와 달리 이 값은 학습 전체의 요약이고, 예측을 올렸는지 내렸는지는 알려 주지 않습니다."
               + (f" {nms(unused[0], '는')} 이 폴드에서 한 번도 쓰이지 않았습니다." if len(unused) == 1 else "")),
          notes={src.line(MODEL, 'return self.est.booster_.feature_importance(importance_type="gain").astype(float)'): f"importance_type = gain → 열마다 분할 이득의 합 (이 폴드 1위 {fn[og[0]]} {gi[og[0]]:.3e})"},
          stack=[f"{MODEL} · feature_importance"], nodes=["fit"],
          tensors=[{"type": "table", "title": "분할 이득 합 (상위 15)", "columns": ["열", "뜻", "이득 합", "쓰인 횟수"], "rows": [[fn[j], dsc(fn[j]), f"{gi[j]:.3e}", int(L["importance_split"][j])] for j in og[:15]], "split": 1}],
          checks=[chk("dump 에서 더한 이득 = feature_importance(gain) (상대)", gain_s, gi, 1e-6, rel=True, kind="식"),
                  chk("dump 에서 센 마디 수 = feature_importance(split)", split_s, L["importance_split"], 0, kind="식")])
    # ================================================================ 8. 모든 폴드
    af = cap["all_folds"]
    dts = [pd.Timestamp(x["test"][-1]) for x in af]; pp = np.array([x["pred"][-1] for x in af]); yy = y.reindex(dts).values
    ok = np.isfinite(pp) & np.isfinite(yy)
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    hit = lambda q: float(np.mean(np.sign(q[ok]) == np.sign(yy[ok])))
    if is_ridge:
        W = np.array([x["coef"] for x in af]); top = np.argsort(-np.mean(np.abs(W), 0))[:6]; cols6 = ["#3b6fb6", "#c8710f", "#2f8f5b", "#6a3fa0", "#b3261e", "#8a93a3"]
        flips = [int(np.sum(np.sign(W[:, j]) != np.sign(np.median(W[:, j])))) for j in top]
        S("8. 모든 폴드",
          f"폴드마다 다시 학습하므로 계수도 폴드마다 다릅니다({len(af)}개 폴드). 평균 크기 상위 6개 재료의 계수입니다 — 부호가 바뀌는 재료(괄호: 중앙값과 부호가 다른 폴드 수)는 '같은 방향의 관계' 라고 말하기 어렵습니다.",
          say=(f"폴드마다 다시 학습하므로 계수도 폴드마다 달라집니다. 그림은 평균 크기가 큰 재료 여섯 개의 폴드별 계수입니다. "
               f"범례 괄호 안의 숫자는 부호가 중앙값과 반대인 폴드 수입니다. "
               + (f"예를 들어 {nm(feats[top[int(np.argmax(flips))]], '는')} {len(af)}개 폴드 중 {max(flips)}개에서 부호가 반대입니다. " if max(flips) else "여기 여섯 개는 부호가 한 번도 바뀌지 않았습니다. ")
               + "부호가 자주 바뀌는 재료는 일정한 방향의 관계라고 말하기 어렵습니다."),
          stack=["tools/codewalk/table_capture.py · 폴드마다 같은 엔진 경로로 학습"], nodes=["allf"], edges=[("pred", "allf")],
          tensors=[{"type": "line", "title": "폴드별 계수", "x": [d.strftime("%Y-%m") for d in dts], "xlabel": "테스트 월",
                    "series": [{"name": f"{feats[j]} ({fl})", "color": cols6[n], "y": [num(v_) for v_ in W[:, j]]} for n, (j, fl) in enumerate(zip(top, flips))], "height": 250}])
    else:
        G = np.array([x["gain"] for x in af]); rank_top5 = np.zeros(K)
        for gr in G: rank_top5[np.argsort(-gr)[:5]] += 1
        o5 = np.argsort(-rank_top5)[:12]
        Gm = G.mean(0); rfi = np.asarray(res["feature_importance"], dtype=float) if res and res.get("feature_names") == feats and res.get("feature_importance") else None
        orr = np.argsort(-(rfi if rfi is not None else Gm))[:5]
        after_fi = src.line(EXP, "def _finish_forecast(") if mode == "forecast" else 0
        ln_fi = src.line(EXP, "im = m.feature_importance()"); ln_ap = src.line(EXP, "importances.append(imp_f)")
        ln_mean = src.line(EXP, "fi_mean = np.mean(importances, axis=0).tolist() if importances else None", after=after_fi)
        ln_save = src.line(EXP, '"feature_importance": fi_mean', after=after_fi)
        j5, jl = int(o5[0]), int(og[0])
        # 폴드 평균은 뒤쪽 폴드가 좌우: 폴드별 이득 합은 그 폴드 학습 정답의 제곱합(n × 분산)을 따라간다
        tot_g = G.sum(1); nq = 20 if len(af) >= 60 else max(1, len(af) // 3); n3 = len(af) // 3
        sse = []
        for x_ in af:
            i_ = prices.index.get_loc(pd.Timestamp(x_["test"][0]))
            tw = prices.index[i_ - tl: i_ - emb]; trw = X.loc[tw].join(y.loc[tw].rename("y")).dropna()
            assert len(trw) == x_["n_tr"]
            sse.append(float(np.sum((trw["y"] - trw["y"].mean()) ** 2)))
        sse = np.array(sse); g_ratio = tot_g[-nq:].sum() / tot_g[:nq].sum(); s_ratio = sse[-nq:].sum() / sse[:nq].sum(); g_corr = float(np.corrcoef(tot_g, sse)[0, 1])
        j1 = int(orr[0]); share = G[-n3:, j1].sum() / G[:, j1].sum()
        Gs = G / G.sum(1, keepdims=True); gs_m = Gs.mean(0); ors = np.argsort(-gs_m)   # 폴드마다 이득을 그 폴드 합으로 나눈 비중의 평균
        S("8. 모든 폴드",
          f"폴드마다 다시 학습하므로 중요한 재료도 폴드마다 다릅니다. {len(af)}개 폴드에서 각 재료가 '이득 상위 5' 에 든 횟수입니다 — 한두 재료가 꾸준히 위에 있는지, 폴드마다 바뀌는지를 봅니다. "
          + (f"마지막 폴드에서 이득이 가장 컸던 {fn[jl]} 는 {int(rank_top5[jl])}개 폴드에서 상위 5 에 들었습니다. " if jl != j5 else "")
          + f"<b>결과 화면의 변수 중요도</b>는 폴드 {len(af)}개의 이득(gain)을 열마다 평균한 값입니다(fi_mean) — 두 번째 표는 그 상위 5"
          + (" 이고, 결과 파일의 feature_importance 와 같습니다(검증)." if rfi is not None else " 입니다.")
          + f" <b>평균은 뒤쪽 폴드가 좌우합니다</b>: 마지막 {nq}개 폴드의 이득 합은 처음 {nq}개의 {g_ratio:.1f}배이고, 폴드별 이득 합은 그 폴드 학습 정답의 제곱합(Σ(y − ȳ)²)과 거의 비례합니다"
          f"(상관 {g_corr:.4f}; 마지막 {nq}개 폴드의 제곱합은 처음 {nq}개의 {s_ratio:.2f}배). {fn[j1]} 이득의 {share:.0%} 는 마지막 {n3}개 폴드(테스트 {af[-n3]['test'][0][:7]} ~)에서 나왔습니다. "
          + f"폴드마다 이득을 그 폴드의 합으로 나눠(비중) 평균하면 1위가 {fn[int(ors[0])]}({gs_m[ors[0]]:.3f}) > {fn[int(ors[1])]}({gs_m[ors[1]]:.3f}) 로 바뀝니다. "
          + (f"그래서 '상위 5 횟수' 1위({fn[j5]})와 평균 1위({fn[j1]})가 다를 수 있습니다." if j5 != j1 else ""),
          code=C(EXP, [ln_fi, ln_ap, ln_mean, ln_save], ln_mean, "_walkforward_predict · _run_experiment", 2, 3),
          say=(f"중요한 재료도 폴드마다 달라집니다. 표는 {len(af)}개 폴드에서 각 재료가 이득 상위 다섯 안에 든 횟수입니다. "
               f"{nms(fn[j5], '이')} {int(rank_top5[j5])}개 폴드로 가장 자주 들었고"
               + (f", 마지막 폴드에서 이득이 가장 컸던 {nms(fn[jl], '은')} {int(rank_top5[jl])}개 폴드에서만 들었습니다. " if jl != j5 else "습니다. ")
               + f"결과 화면의 변수 중요도는 이 이득의 폴드 평균이고, 1위는 {nms(fn[j1])}입니다. "
               + f"그런데 이 평균은 뒤쪽 폴드가 좌우합니다. 마지막 {nq}개 폴드의 이득 합이 처음 {nq}개의 {g_ratio:.1f}배입니다. "
               f"{nms(fn[j1], '은')} 이득의 {share:.0%}가 마지막 {n3}개 폴드에서 나왔습니다. "
               + (f"폴드마다 이득을 비중으로 바꿔 평균하면 1위가 {nms(fn[int(ors[0])], '로')} 바뀝니다. " if int(ors[0]) != j1 else "")
               + ("횟수 1위와 평균 1위가 다른 것도 이 때문으로 보입니다." if j5 != j1 else "")),
          notes={ln_fi: "폴드마다 학습한 모델의 gain 중요도(앞 화면의 값)를 모음",
                 ln_mean: f"폴드 {len(af)}개를 열마다 평균 → 결과 화면의 변수 중요도",
                 ln_save: "결과 파일(feature_importance)에 저장"},
          lab=(lab_imp_mean([fn[j] for j in orr], [G[:, j] for j in orr], rfi[orr], len(af)) if rfi is not None else None),
          stack=["tools/codewalk/table_capture.py · 폴드마다 같은 엔진 경로로 학습", "engine/experiment.py · fi_mean"], nodes=["allf"], edges=[("pred", "allf")],
          tensors=[{"type": "table", "title": f"이득 상위 5 에 든 폴드 수 (전체 {len(af)})", "columns": ["열", "뜻", "횟수", "비율"], "rows": [[fn[j], dsc(fn[j]), int(rank_top5[j]), f"{rank_top5[j] / len(af):.0%}"] for j in o5]},
                   {"type": "table", "title": f"결과 화면의 변수 중요도 상위 5 (폴드 {len(af)}개 평균)", "columns": ["열", "뜻", "폴드 평균 gain", "결과 파일"],
                    "rows": [[fn[j], dsc(fn[j]), f"{Gm[j]:.4e}", f"{rfi[j]:.4e}" if rfi is not None else "—"] for j in orr]}],
          checks=([chk(f"폴드 {len(af)}개 gain 평균 = 결과 파일 feature_importance ({K}개 열) (상대)", Gm, rfi, 1e-12, rel=True, kind="재실행")] if rfi is not None else []))
    if mode == "forecast":
        nv = (y.shift(k)).reindex(dts).values      # engine/calendar.py naive_forecast: 지난 k기간 변화 그대로
        ok2 = ok & np.isfinite(nv)
        code_rmse = (
            f"# {int(ok2.sum())}개 폴드의 실제 변화율, {KN} 예측, naive 예측 (지난 변화 그대로) — 실행 기록 그대로\n"
            f"actual = {py_list(yy[ok2])}\n"
            f"model = {py_list(pp[ok2])}\n"
            f"naive = {py_list(nv[ok2])}\n\n"
            "def rmse(pred):\n"
            "    s = 0.0\n"
            "    for p, a in zip(pred, actual):\n"
            "        s += (p - a) ** 2\n"
            "    return (s / len(actual)) ** 0.5\n\n"
            f"print(\"{KN}\", f\"{{rmse(model) * 100:.4f}}%\")\n"
            "print(\"naive\", f\"{rmse(naive) * 100:.4f}%\")\n"
            "print(\"0 예측\", f\"{rmse([0.0] * len(actual)) * 100:.4f}%\")\n")
        exp_rmse = (f"{KN} {rmse(pp[ok2] - yy[ok2]) * 100:.4f}%\nnaive {rmse(nv[ok2] - yy[ok2]) * 100:.4f}%\n0 예측 {rmse(yy[ok2]) * 100:.4f}%")
        lab_rmse = {"title": "RMSE 를 직접 계산", "kind": "식", "desc": "오차를 제곱해 평균하고 제곱근을 씌웁니다. 모델·naive·'항상 0' 을 같은 방법으로 비교합니다.",
                    **run_lab(code_rmse, exp_rmse, pure=True)}
        S("8. 모든 폴드",
          f"{len(af)}개 폴드의 한 달 예측을 실제와 나란히 놓았습니다. 기준 두 가지: '항상 0' 과 결과 화면의 naive('지난 변화 그대로' = y_(t−{k})). "
          f"RMSE: {KN} {pct(rmse(pp[ok2] - yy[ok2]), 3, False)}, naive {pct(rmse(nv[ok2] - yy[ok2]), 3, False)}, 0 예측 {pct(rmse(yy[ok2]), 3, False)}"
          + (" — <b>naive 보다 오차가 큽니다.</b>" if rmse(pp[ok2] - yy[ok2]) > rmse(nv[ok2] - yy[ok2]) else " — naive 보다 오차가 작습니다(유의성은 결과 화면의 DM 검정).")
          + " 공식 판정은 결과 화면의 naive 대비 배율·DM 검정입니다.",
          code=C(CAL, src.span(CAL, "def naive_forecast(y: pd.Series, k: int) -> pd.Series:", 'return y.shift(k).rename("naive")'), src.line(CAL, 'return y.shift(k).rename("naive")'), "naive_forecast", 0, 0),
          say=(f"{len(af)}개 폴드의 예측을 실제 값과 나란히 놓았습니다. 비교 기준은 두 가지입니다. 하나는 항상 0이라고 예측하는 것, "
               "다른 하나는 지난달의 변화가 그대로 이어진다고 보는 나이브 예측입니다. 오차의 크기인 RMSE는 "
               f"{KN} {pct(rmse(pp[ok2] - yy[ok2]), 3, False)}, 나이브 {pct(rmse(nv[ok2] - yy[ok2]), 3, False)}, 0 예측 {pct(rmse(yy[ok2]), 3, False)}입니다. "
               + ("나이브보다 오차가 큽니다. 자료 자체가 매끄러워서 지난달 변화만으로도 잘 맞기 때문입니다." if rmse(pp[ok2] - yy[ok2]) > rmse(nv[ok2] - yy[ok2]) else
                  "나이브보다 오차가 작습니다. 우연인지는 결과 화면의 DM 검정으로 봅니다.")),
          notes={src.line(CAL, 'return y.shift(k).rename("naive")'): f"정답을 {k}칸 미룸 = '지난 {k}개월 변화가 그대로' 라는 예측"},
          lab=lab_rmse,
          stack=["engine/calendar.py · naive_forecast"], nodes=["allf"],
          tensors=[{"type": "line", "title": "폴드별 예측과 실제", "x": [d.strftime("%Y-%m") for d in dts], "xlabel": "테스트 월",
                    "series": [{"name": "실제", "color": "#3b6fb6", "y": [num(v_) for v_ in yy], "dots": True}, {"name": KN, "color": "#c8710f", "y": [num(v_) for v_ in pp], "dots": True},
                               {"name": "naive", "color": "#8a93a3", "y": [num(v_) for v_ in nv]}], "height": 230},
                   {"type": "table", "title": f"요약 ({int(ok2.sum())}개 폴드)", "columns": ["예측", "RMSE", "방향 적중"],
                    "rows": [[KN, pct(rmse(pp[ok2] - yy[ok2]), 4, False), f"{np.mean(np.sign(pp[ok2]) == np.sign(yy[ok2])) * 100:.1f}%"],
                             ["naive (지난 변화 그대로)", pct(rmse(nv[ok2] - yy[ok2]), 4, False), f"{np.mean(np.sign(nv[ok2]) == np.sign(yy[ok2])) * 100:.1f}%"], ["항상 0", pct(rmse(yy[ok2]), 4, False), "—"]]}],
          checks=[chk("모든 폴드 예측의 마지막 = 마지막 폴드 캡처의 예측", pp[-1], cap["pred"].loc[t], 1e-14, kind="재실행")])
    else:
        thr = cfg["trade"]["threshold"]; sig = (pp > thr).astype(float)
        held = yy[ok & (sig == 1)]; out_ = yy[ok & (sig == 0)]
        m_held = np.mean(held) if len(held) else np.nan; m_out = np.mean(out_) if len(out_) else np.nan
        h_ = hit(pp) * 100; up_ = np.mean(yy[ok] > 0) * 100
        pm = (res or {}).get("prediction_metrics") or {}; pf = (res or {}).get("performance") or {}
        dmv, dmp = pm.get("dm_vs_zero"), pm.get("dm_vs_zero_p")
        has_dm = dmv is not None and dmp is not None and np.isfinite(dmv) and np.isfinite(dmp)
        dm_sig = has_dm and dmv > 0 and dmp < 0.05
        ln_sig = src.line(EXP, "signal = (pred > cfg.trade.threshold).astype(float).where(pred.notna())")   # 2026-11-12: 신호 변환이 _signal_from_prediction 함수로 옮겨짐   # 개발 구간 평가(봉인 아님) 갈래
        ln_bt = src.line(EXP, "daily_full, _ = run_backtest(ds.prices, signal, cost=cost, periods_per_year=ppy)")
        ln_dm = src.line(EXP, 'dm, dm_p = diebold_mariano(e.values, (0 - ev["target"]).values, h=k)')
        lab_tr = (lab_trade(pp[ok], yy[ok], thr, pf["exposure"], pm["rmse"], pm["dir_acc"]) if all(k_ in pf for k_ in ("exposure",)) and all(k_ in pm for k_ in ("rmse", "dir_acc")) else None)
        # 봉인 경계: 보유 구간(t+1 시가 → t+1+k 시가)이 봉인 구간 안에 있는 결정 — 엔진은 결정일만 봉인 전으로 자른다
        from engine.backtest import diebold_mariano
        sl_ = pd.Timestamp(sealed) if sealed else None
        in_seal = np.array([sl_ is not None and open_day(prices.index.get_loc(d) + 1) >= sl_ for d in dts])        # 진입부터 봉인 구간
        part_seal = [d for d in dts if sl_ is not None and open_day(prices.index.get_loc(d) + 1) < sl_ <= open_day(prices.index.get_loc(d) + 1 + k)]   # 청산만 봉인 구간
        keep = ok & ~in_seal
        held_k = yy[keep & (sig == 1)]; dm_k = diebold_mariano((pp - yy)[keep], (0 - yy)[keep], h=k)
        pq = ROOT / str(cap["config_file"]).replace(".json", "_daily.parquet")
        pq_d = pd.read_parquet(pq, columns=["position", "net_return", "cost"]) if pq.exists() and in_seal.any() else None   # 검증 줄에만 씀 — 없으면 그 줄만 건너뜀
        ev_p = pf.get("n_periods"); ev_rng = (res or {}).get("evaluation_period") or [None, None]
        S("8. 모든 폴드",
          f"{len(af)}개 폴드(각 1개월)의 예측입니다. 매매 규칙: 예측 > {thr} 이면 다음 달 첫 거래일 시가에 보유, 아니면 현금. "
          f"보유한 달 {len(held)}개의 실제 수익률 평균 {pct(m_held)}, 현금으로 둔 달 {len(out_)}개의 실제 평균 {pct(m_out)}. "
          f"방향 적중 {h_:.1f}% (항상 '오름' 이라고 했다면 {up_:.1f}%). "
          + (f"결과 파일의 DM 검정(모델 vs 항상 0, 제곱오차, h = {k}): 통계량 {dmv:+.2f}, p = {dmp:.3f} — 양수는 모델의 오차가 더 크다는 뜻"
             + (", 5% 수준에서 유의합니다. " if dmp < 0.05 else ", 5% 수준에서 유의하지 않습니다. ") if has_dm else "")
          + "수수료·슬리피지를 넣은 실제 성과와 다른 검정(부트스트랩·Deflated Sharpe)은 결과 화면에 있습니다."
          + (f" <b>봉인 경계</b>: 마지막 결정 " + ", ".join(str(d.date()) for d, f_ in zip(dts, in_seal) if f_) + f" 의 보유 구간({open_day(ti + 1).date()}→{open_day(ti + 1 + k).date()} 시가)은 전부 봉인 구간({sealed}~)이고, "
             f"그 달의 수익 {pct(y.loc[t])} 가 위의 평균·방향 적중·RMSE·DM 과 결과 화면의 성과에 들어 있습니다"
             + (f"(" + ", ".join(f"{d.date()} 결정은 청산 시가 {open_day(prices.index.get_loc(d) + 1 + k).date()} 가 봉인 구간 안" for d in part_seal) + ")" if part_seal else "")
             + f" — 엔진은 결정일만 봉인 전으로 자릅니다. 봉인 구간에서 시작하는 달을 빼면({int(keep.sum())}개월): 보유 {len(held_k)}개월 평균 {pct(np.mean(held_k))}, "
             f"RMSE {pct(rmse(pp[keep] - yy[keep]), 2, False)}, DM {dm_k[0]:+.2f}(p = {dm_k[1]:.3f}, 엔진 diebold_mariano 로 계산)." if in_seal.any() else ""),
          code=C(EXP, [ln_sig, ln_bt, ln_dm], ln_sig, "_run_experiment · 신호 변환 → 체결 → 예측 성능", 3, 3),
          say=(f"{len(af)}개 폴드의 예측으로 매매합니다. 예측이 {thr:g}보다 크면 다음 달 첫 거래일 시가에 사서 한 달 들고, 아니면 현금으로 둡니다. "
               f"보유한 {len(held)}개월의 실제 평균 수익률은 {pct(m_held)}, 현금 {len(out_)}개월은 {pct(m_out)}"
               + ("로, 들고 있지 않은 달이 평균으로 더 올랐습니다. " if m_out > m_held else "입니다. ")
               + f"방향 적중은 {h_:.1f}%로, 항상 오른다고 했을 때의 {up_:.1f}%보다 " + ("낮습니다. " if h_ < up_ else "높습니다. ")
               + (f"결과 파일의 DM 검정에서도, 모델의 예측 오차가 항상 0이라고 예측했을 때의 오차보다 5% 수준에서 유의하게 컸습니다. 통계량은 양수인 {abs(dmv):.2f}, p값은 약 {dmp:.2f}입니다. " if dm_sig else "")
               + "평균 수익률은 비용을 빼기 전입니다."),
          notes={ln_sig: f"예측 > {thr} → 1(보유), 아니면 0(현금) → {len(af)}개월 중 {len(held)}개월 보유",
                 ln_bt: "포지션 = 신호(1 보유 / 0 현금). 결정일 t 의 포지션은 t+1 시가 → t+2 시가 수익률을 얻고, 거래와 비용은 포지션이 바뀔 때만 생김",
                 **({ln_dm: f"DM 검정: 모델 오차 vs '항상 0' 오차 → 통계량 {dmv:+.2f}, p = {dmp:.3f} (결과 파일)"} if has_dm else {})},
          lab=lab_tr,
          stack=["engine/experiment.py · _run_experiment"], nodes=["allf"],
          tensors=[{"type": "line", "title": "폴드별 예측과 실제 (월 수익률)", "x": [d.strftime("%Y-%m") for d in dts], "xlabel": "결정 월(수익은 다음 달 첫 시가 → 그다음 달 첫 시가)",
                    "series": [{"name": "실제", "color": "#3b6fb6", "y": [num(v_) for v_ in yy], "dots": True}, {"name": KN, "color": "#c8710f", "y": [num(v_) for v_ in pp], "dots": True}], "height": 230},
                   {"type": "table", "title": "요약 (보유·현금 달 평균은 비용 전)", "columns": ["", "값"], "rows": [["폴드(월)", int(ok.sum())], ["보유한 달", int(len(held))], ["보유한 달 실제 평균", pct(m_held)],
                                                                   ["현금 달 실제 평균", pct(m_out)], ["방향 적중", f"{h_:.1f}%"], ["RMSE (모델 / 항상 0)", f"{pct(rmse(pp[ok] - yy[ok]), 2, False)} / {pct(rmse(yy[ok]), 2, False)}"]]
                    + ([["DM 통계량 (모델 vs 항상 0, 결과 파일)", f"{dmv:+.2f} (p = {dmp:.3f})"]] if has_dm else [])}],
          checks=[chk("모든 폴드 예측의 마지막 = 마지막 폴드 캡처의 예측", pp[-1], cap["pred"].loc[t], 1e-14, kind="재실행")]
                 + ([chk("영상의 RMSE·방향 적중 = 결과 파일 prediction_metrics", [rmse(pp[ok] - yy[ok]), np.mean((pp[ok] > 0) == (yy[ok] > 0))], [pm["rmse"], pm["dir_acc"]], 1e-12, kind="재실행")] if "rmse" in pm and "dir_acc" in pm else [])
                 + (([chk(f"봉인 구간 보유 달이 결과 화면 성과에 들어감: 결과 파일 {t.date()} position = 1 · 순수익 = 이 달 정답 {pct(y.loc[t])}(비용 0) · "
                          f"성과 구간 {ev_p}기간({ev_rng[0]} ~ {ev_rng[1]}, performance.n_periods) = 폴드 {len(af)}개",
                          [float(pq_d.loc[t, "position"]), float(pq_d.loc[t, "net_return"]), float(pq_d.loc[t, "cost"]), float(ev_p or -1), float(ev_rng[1] == str(t.date()))],
                          [1.0, float(y.loc[t]), 0.0, float(len(af)), 1.0], 1e-12, kind="재실행")] if pq_d is not None else
                     [skip(f"봉인 구간 보유 달이 결과 화면 성과에 들어감(결과 파일 {t.date()} position·순수익·성과 구간)", f"{pq.relative_to(ROOT)} 없음", "재실행")]) if in_seal.any() else []))
    mr = match_result(mode, cfg["name"], {d: v_ for d, v_ in zip(dts, pp)})
    if not mr:   # 이 단계의 대본(음성)이 결과 파일 비교를 말하므로, 결과 파일이 없으면 멈춘다(빌드 환경에 따라 단계·음성이 달라지지 않게)
        raise FileNotFoundError(f"results/{mode}/{cfg['name']}_*_daily.parquet 결과 파일이 없습니다 — '결과 파일과 비교' 단계의 대본이 이 값을 씁니다")
    if mr:
        same = mr["max_diff"] < 1e-12
        S("8. 모든 폴드",
          (f"이 영상의 {len(af)}개 예측을 결과 폴더의 같은 실험 결과 <b>{mr['file'].replace('_daily.parquet', '')}</b> 의 예측과 맞춰 봅니다: 겹치는 {mr['n']}개, 최대 차이 {mr['max_diff']:.1e}"
           + (" — <b>결과 화면의 숫자와 같은 계산</b>입니다." if same else " — 다릅니다. 이 결과 파일은 지금 엔진·데이터 이전에 만든 것이라 다시 돌린 결과(_v1107)와 비교해야 합니다.")),
          say=(f"마지막으로, 이 영상에서 계산한 {len(af)}개 예측을 결과 폴더에 저장된 같은 실험의 결과 파일과 비교합니다. "
               + (f"겹치는 {mr['n']}개 예측의 최대 차이는 0으로, 모두 똑같습니다. 결과 화면의 숫자와 같은 계산이라는 뜻입니다." if mr["max_diff"] == 0 else
                  f"겹치는 {mr['n']}개 예측의 최대 차이는 {mr['max_diff']:.1e}"
                  + ("로 사실상 0입니다. 결과 화면의 숫자와 같은 계산이라는 뜻입니다." if same else "로 서로 다릅니다. 이 결과 파일은 지금의 엔진과 데이터 이전에 만든 것입니다."))),
          stack=["results/" + mode + "/" + mr["file"]], nodes=["allf"],
          tensors=[{"type": "table", "title": "결과 파일별 비교", "columns": ["결과 파일", "겹친 예측 수", "최대 차이"], "rows": [[a_.replace("_daily.parquet", ""), n_, f"{d_:.1e}"] for a_, n_, d_ in mr["all"]]}],
          checks=[chk(f"영상의 예측 = 결과 파일 {mr['file'].replace('_daily.parquet', '')} 의 예측 ({mr['n']}개)", mr["max_diff"], 0.0, 1e-12, kind="재실행")] if same else [])
    # ================================================================ 9. 정리
    n_ok = sum(c["ok"] is True for c in checks_all); n_skip = sum(bool(c.get("skipped")) for c in checks_all)
    lim = []   # 공개 시점만으로 막을 수 없는 한계 (3. 외부 변수 화면)
    if vint: lim.append("최신 수정값 " + "·".join(vint))
    if unconf or t_unconf: lim.append("공개 시각 미확인 " + "·".join(unconf + t_unconf))
    lim_t = (" — 단, " + ", ".join(lim)) if lim else ""
    kc = Counter(c_["kind"] for c_ in checks_all)
    kind_t = " · ".join(f"{KIND[k_]} {kc.get(k_, 0)}" for k_ in KIND)
    tgt_lag = int(mst.loc[cfg["target_variable"], "lag_days"]) if mode == "forecast" and cfg.get("target_variable") in mst.index else None

    S("9. 정리",
      (f"정리: 외부 변수 {len(exo)}개는 결정일(대상 값 {cfg['target_variable']}_t 의 기준일)까지 공개된 값만 쓰고(변환 · 발표시차 · ffill{lim_t}), "
       f"기술지표 {len(tech)}개는 대상 자신의 그달 값 v_t 까지 써서 만들었습니다 — 대상 값이 공개된 뒤(기준일 + lag {tgt_lag}일, 변수 사전) 예측한다고 보는 설정입니다(forecast_info_time = {cfg.get('forecast_info_time')}). "
       if mode == "forecast" else
       f"정리: 결정일마다 공개일(첫 발표일)이 지난 값만 쓰도록 맞춘 재료 {K}개를 만들고(기술지표 · 변환 · 발표시차 · ffill{lim_t}), ")
      + f"학습 창 {cfg['validation']['train_len']}개월에서 비움 {emb}개월을 뺀 학습 표로 눈금을 맞춘 뒤 "
      + ("Ridge 의 닫힌 해로 계수를 정해 가중합으로 예측합니다." if is_ridge else f"LightGBM 이 얕은 나무 {p['n_estimators']}그루를 차례로 더해 예측합니다.")
      + f" 검증 {len(checks_all)}줄 — {kind_t} — 중 <b>{n_ok}줄 통과</b>" + (f", {n_skip}줄 건너뜀(이 빌드 환경에서 확인할 수 없음 — 표 참조)" if n_skip else "") + ".",
      say=((f"정리하면, 외부 변수 {len(exo)}개는 결정일, 곧 대상 값의 기준일까지 공개된 값만 썼고, 기술지표 {len(tech)}개는 대상 자신의 그달 값까지 써서 만들었습니다. "
            "대상 값이 공개된 뒤 예측한다고 보는 설정입니다. " if mode == "forecast" else
            f"정리하면, 재료 {K}개는 결정일마다 공개일이 지난 값만 쓰도록 맞췄습니다. ")
           + ((f"다만 {_att(vint_name, '은')} 발표 뒤 고쳐진 최신 값이고" if len(vint) == 1 else f"다만 경제 변수 {len(vint)}개는 발표 뒤 고쳐진 최신 값이고") if vint else "다만")
           + (f", {_att(tu_say, '은')} {tu_verb}. " if tu_say else "입니다. ")
           + f"학습 창 {tl}개월에서 끝 {emb_w}을 비운 학습 표로 눈금을 맞춘 뒤, "
           + ("릿지의 닫힌 해로 계수를 정해 가중합으로 예측합니다. " if is_ridge else f"라이트지비엠이 얕은 나무 {p['n_estimators']}그루를 차례로 더해 예측합니다. ")
           + f"이 영상의 검증 {len(checks_all)}줄은 모두 통과했습니다. 그중 {kc.get('독립', 0)}줄은 다른 경로로 대조했고, {kc.get('식', 0)}줄은 같은 식이나 구조를 확인했으며, "
           f"{kc.get('재실행', 0)}줄은 " + ("다시 돌린 기록이나 결과 파일과 맞췄습니다. " if is_ridge else "다시 돌리거나 다시 만든 기록, 또는 결과 파일과 맞췄습니다. ") + "단계마다 있는 '직접 실행해 보기' 코드로 같은 계산을 직접 돌려 볼 수 있습니다."),
      stack=["요약"], nodes=ALL,
      tensors=[{"type": "table", "title": f"검증 전체 ({kind_t})", "columns": [f"화면(n/{len(steps) + 1})", "무엇", "종류", "결과"],
                "rows": [[c["step"], c["what"], KIND[c["kind"]], "— 건너뜀" if c.get("skipped") else ("✓ 통과" if c["ok"] else "✗ 다름")] for c in checks_all]}])
    if n_ok + n_skip != len(checks_all):
        raise AssertionError(f"검증 실패: {[c for c in checks_all if c['ok'] is False]}")

    def clean(o):
        if isinstance(o, dict): return {k_: clean(v_) for k_, v_ in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v_) for v_ in o]
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        return o
    steps[:] = clean(steps)
    used = sorted({s_["code"]["file"] for s_ in steps if s_.get("code")})
    return {"title": f"{KN} 코드 따라가기 — 실제로 계산되는 순서", "model": model, "steps": steps, "sources": {fn_: src.text[fn_] for fn_ in used}, "diagram": diagram(is_ridge), "checks": clean(checks_all)}


def diagram(is_ridge) -> dict:
    labels = [("tgt", "대상·정답", "make_target"), ("tech", "기술지표", "features.py"), ("exo", "외부 변수", "변환·발표시차"), ("ds", "재료 표 X", "결정일 × 열"), ("fold", "폴드", "학습·비움·테스트"),
              ("scale", "눈금", "StandardScaler"), ("fit", "학습", "닫힌 해" if is_ridge else "부스팅 나무"), ("pred", "예측", "가중합" if is_ridge else "잎 값의 합"), ("allf", "모든 폴드", "안정성·기준 비교")]
    nodes = [{"id": i, "label": l, "x": 10 + 208 * k, "y": 14, "w": 186, "h": 52, "sub": s} for k, (i, l, s) in enumerate(labels)]
    E = [{"from": "tgt", "to": "tech"}, {"from": "tech", "to": "exo"}, {"from": "exo", "to": "ds"}, {"from": "ds", "to": "fold"}, {"from": "fold", "to": "scale"},
         {"from": "scale", "to": "fit"}, {"from": "fit", "to": "pred"}, {"from": "pred", "to": "allf"}]
    return {"width": 10 + 208 * len(labels), "height": 84, "nodes": nodes, "edges": E, "groups": []}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ridge", choices=["ridge", "lgbm"])
    ap.add_argument("--capture"); ap.add_argument("--out")
    a = ap.parse_args()
    cap = pickle.load(open(a.capture or ROOT / f"docs/guide_build/walk/{a.model}_capture.pkl", "rb"))
    sc = build(cap)
    out = build_html(sc, Path(a.out) if a.out else ROOT / f"docs/guide_build/walk/{a.model}.html")
    n_sk = sum(bool(c.get("skipped")) for c in sc["checks"])
    print("저장:", out, "· 단계", len(sc["steps"]), "· 검증", sum(c["ok"] is True for c in sc["checks"]), "/", len(sc["checks"]) - n_sk, (f"(건너뜀 {n_sk})" if n_sk else ""), "· 크기", round(out.stat().st_size / 1e6, 2), "MB")
