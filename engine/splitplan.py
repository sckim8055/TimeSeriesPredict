"""1회 분할 계획 — 비율(%)이나 날짜로 학습·검증·테스트를 나누면 '실제로' 몇 개씩, 어느 날짜부터 어느 날짜까지 쓰이는지 미리 보여 준다.

실험을 돌리지 않고(학습 없음) 엔진이 실행 때 쓰는 함수(prepare·make_folds·_embargo)와 같은 규칙으로 센다.
그래서 이 계획의 숫자는 실행 결과의 periods(폴드 기록 n_train·n_val·n_test)와 같아야 한다(tests/test_splitplan.py 가 확인).

비율 → 개수 규칙(원본 BIVA data_provider/data_loader.py 78~80줄과 같음):
    학습 = int(N × 학습%) · 테스트 = int(N × 테스트%) · 검증 = N − 학습 − 테스트 (나머지는 검증에)
    N = 정답(타겟)이 있는 결정일 수(봉인 전, 데이터 사용 기간 안). 검증% 가 0 이면 나머지는 학습에.
날짜는 정확히 그 날이 없어도 된다: 검증·테스트 시작일은 '그 날짜 이후 첫 결정일', 테스트 종료일은 '그 날짜 이전 마지막 결정일'로 맞춰진다.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from engine import calendar as cal

MIN_ROWS = 30  # 엔진 규칙: 학습 표본이 30개 미만인 폴드는 건너뛴다(_walkforward_predict)
MIN_VAL = 10   # 엔진 규칙: 검증이 10개 미만이거나 검증을 떼고 남는 학습이 10개 미만이면 검증 없이 전부 학습(_split_fit)


def ratio_counts(n: int, ratios) -> tuple[int, int, int]:
    """N 개를 학습·검증·테스트 비율(%)로 나눈 개수. 원본 BIVA 와 같이 학습·테스트는 내림, 나머지는 검증(검증% 0 이면 학습)."""
    a, b, c = (Fraction(str(x)) for x in ratios)
    if min(a, b, c) < 0 or a + b + c != 100:
        raise ValueError(f"비율은 0 이상이고 합이 100 이어야 합니다: {list(ratios)}")
    n_tr = int(n * a / 100)
    n_te = int(n * c / 100)
    n_va = n - n_tr - n_te
    if b == 0:
        n_tr, n_va = n_tr + n_va, 0
    return n_tr, n_va, n_te


def _d(x) -> str | None:
    return None if x is None else str(pd.Timestamp(x).date())


def _span(ix: pd.DatetimeIndex) -> dict:
    return {"first": _d(ix[0]) if len(ix) else None, "last": _d(ix[-1]) if len(ix) else None, "n": int(len(ix))}


def base_index(cfg, ds, k: int) -> pd.DatetimeIndex:
    """비율을 셀 바탕: 봉인 전·데이터 사용 기간 안에서 정답(타겟)이 있는 결정일."""
    from engine.experiment import seal_cut
    idx = cal.trading_calendar(ds.prices)
    sf = seal_cut(cfg, idx, k)
    if sf is not None:
        idx = idx[idx < sf]
    yv = ds.y.dropna().index
    return idx[idx.isin(yv)]


def split_plan(cfg_dict: dict, dv, reg, ratios=None) -> dict:
    """1회 분할 계획. ratios=[학습, 검증, 테스트](%) 를 주면 날짜를 계산해 채우고, 없으면 설정의 날짜를 그대로 쓴다."""
    from engine.experiment import ExperimentConfig, _embargo, make_folds, prepare

    cfg0 = ExperimentConfig(**cfg_dict)
    if cfg0.validation.mode != "split":
        cfg0 = cfg0.model_copy(update={"validation": cfg0.validation.model_copy(update={"mode": "split"})})
    cfg, plugin, params, _tt, ds = prepare(cfg0, dv, reg)
    k = int(cfg.target.k)
    emb = _embargo(cfg, k)
    shape = plugin.INPUT_SHAPE
    uses_val = bool(getattr(plugin.Model, "USES_VALIDATION", True)) and shape in ("table", "window")
    out: dict = {"model": cfg.model, "shape": shape, "embargo": emb, "uses_validation": uses_val,
                 "seq_len": int(cfg.seq_len) if shape == "window" else None, "notes": [], "blocking": []}

    base = base_index(cfg, ds, k)
    out["base"] = {**_span(base), "what": "정답(타겟)이 있는 결정일 — 봉인 전, 데이터 사용 기간 안"}
    v = cfg.validation
    if ratios is not None:
        n_tr, n_va, n_te = ratio_counts(len(base), ratios)
        if n_te < 1:
            raise ValueError(f"테스트가 0개입니다 — 결정일 {len(base)}개에 테스트 {ratios[2]}%")
        vs = base[n_tr] if (n_va > 0 and uses_val) else None
        ts = base[n_tr + n_va]
        te = base[-1]
        out["ratio"] = {"requested": [float(x) for x in ratios], "counts": [n_tr, n_va, n_te],
                        "rule": "학습 = int(N×학습%) · 테스트 = int(N×테스트%) · 검증 = 나머지 (원본 BIVA data_loader.py 와 같은 규칙)"}
        if n_va > 0 and not uses_val:
            out["notes"].append(f"{plugin.NAME} 은(는) 검증을 쓰지 않으므로 검증 몫 {n_va}개도 학습에 씁니다(검증 시작일 비움). "
                                "같은 날짜의 테스트로 다른 모델과 비교할 수 있게 테스트 시작일은 그대로 둡니다.")
        v = v.model_copy(update={"val_start": _d(vs), "test_start": _d(ts), "split_date": None, "test_end": _d(te)})
        cfg = cfg.model_copy(update={"validation": v})
    req = {"val_start": v.val_start, "test_start": v.test_start or v.split_date, "test_end": v.test_end}
    out["dates"] = req

    folds, _ = make_folds(cfg, ds, k)   # 실행과 같은 함수 — 날짜 오류·너무 짧은 구간은 여기서 같은 말로 멈춘다
    f = folds[0]
    ts_req = pd.Timestamp(req["test_start"])

    # ---- 날짜 맞춤(요청한 날짜가 결정일이 아닐 때)
    snap = []
    if req["val_start"]:
        a = f.train[f.train >= pd.Timestamp(req["val_start"])]
        if len(a) and _d(a[0]) != req["val_start"]:
            snap.append(f"검증 시작일 {req['val_start']} → {_d(a[0])} (그 날짜 이후 첫 결정일)")
    if _d(f.test[0]) != _d(ts_req):
        snap.append(f"테스트 시작일 {_d(ts_req)} → {_d(f.test[0])} (그 날짜 이후 첫 결정일)")
    if req["test_end"] and _d(f.test[-1]) != req["test_end"]:
        snap.append(f"테스트 종료일 {req['test_end']} → {_d(f.test[-1])} (그 날짜 이전 마지막 결정일)")
    out["snapped"] = snap

    fc = fold_counts(cfg, plugin, params, ds, f, emb, uses_val)
    out.update({k: fc[k] for k in ("rows", "used_pct", "steps")})
    out["blocking"] += fc["blocking"]
    out["notes"] += fc["notes"]
    if fc.get("window"):
        out["window"] = fc["window"]
    accepts_nan = fc["accepts_nan"]
    X = ds.X
    feats = list(ds.feature_names)
    notes = out["notes"]
    after = base[base > f.test[-1]]
    if len(after):
        notes.append(f"테스트 종료일 뒤 결정일 {len(after)}개({_d(after[0])}~{_d(after[-1])})는 쓰지 않습니다.")
    if cfg.validation.sealed_from:
        notes.append(f"봉인 시작일 {cfg.validation.sealed_from} 이후는 계산하지 않습니다(비율도 봉인 전 결정일로만 셉니다).")

    out["blanks"] = blanks_info(X, feats, f.train, f.test, shape, accepts_nan)
    return out


def blanks_info(X, feats, train_idx, test_idx, shape: str, accepts_nan: bool) -> dict:
    """변수별 빈 칸(학습·테스트 구간)과 그 모델이 빈 칸을 처리하는 방법."""
    how = ("빈 칸을 그대로 BRITS 에 넘김(BRITS 가 앞뒤 흐름·다른 변수로 추정). 중간에 빈 달은 엔진이 먼저 직전 값으로 채움" if accepts_nan else
           "직전 값으로 채움(ffill). 변수가 시작하기 전 앞쪽 빈 칸은 학습 구간 평균으로 채움" if shape == "window" else
           "빈 칸이 있는 행은 학습에서 빼고, 테스트에서는 그날 예측하지 않음" if shape == "table" else
           "이 모델은 입력 변수 대신 대상의 과거 값으로 예측")
    blanks = []
    for c in feats:
        s = X[c]
        fv = s.first_valid_index()
        n_tr_nan = int(s.loc[train_idx].isna().sum())
        n_te_nan = int(s.loc[test_idx].isna().sum())
        if n_tr_nan or n_te_nan:
            blanks.append({"var": str(c), "first_valid": _d(fv), "blank_train": n_tr_nan, "blank_test": n_te_nan})
    return {"how": how, "vars": blanks}


def fold_counts(cfg, plugin, params, ds, f, emb: int, uses_val: bool) -> dict:
    """폴드 하나에서 학습·검증·테스트로 실제로 쓰이는 개수를 실행 코드(_walkforward_predict·_split_fit)와 같은 순서로 센다.
    steps: 학습 창(또는 학습 구간)에서 무엇이 몇 개씩 빠져 '학습'이 남는지 순서대로."""
    shape = plugin.INPUT_SHAPE
    v = cfg.validation
    X, y = ds.X, ds.y
    feats = list(ds.feature_names)
    full = X.loc[f.train].join(y.loc[f.train].rename("y"))
    y_ok = full["y"].notna()
    complete = full.dropna()                              # 엔진의 tr: 입력·정답에 빈 칸이 하나도 없는 행
    te_rows = X.loc[f.test]
    accepts_nan = False
    if shape == "window":
        wn = getattr(plugin.Model, "wants_nan", None)
        accepts_nan = bool(getattr(plugin.Model, "ACCEPTS_NAN", False)) and bool(wn(params) if wn else False)
    blocking, notes, steps = [], [], []
    if len(complete) < MIN_ROWS:
        blocking.append(f"학습 구간({_d(f.train[0])}~{_d(f.train[-1])})에서 입력·정답이 모두 있는 행이 {len(complete)}개 (< {MIN_ROWS}) — 엔진이 이 폴드를 건너뜁니다. "
                        "늦게 시작하는 변수를 빼거나 시작일을 늦추세요.")
    if (len(te_rows) == 0) if not feats else te_rows.dropna().empty:
        blocking.append("테스트 구간에 입력이 모두 있는 행이 하나도 없어 엔진이 이 폴드를 건너뜁니다.")

    if v.mode == "walkforward":
        steps.append({"what": f"학습 창 {v.train_len}행 ({_d(f.train[0])} ~ 테스트 전날)", "n": len(f.train) + emb})
    else:
        steps.append({"what": f"테스트 시작 전 전부 ({_d(f.train[0])} ~ 테스트 전날)", "n": len(f.train) + emb})
    steps.append({"what": f"끝의 {emb}행 비움 — 학습 정답이 테스트 기간 값을 쓰지 않게", "n": len(f.train), "minus": emb})
    L = None
    if shape == "window":
        L = int(cfg.seq_len)
        Xf = full[feats].ffill()
        allnan = [c for c in feats if Xf[c].notna().sum() == 0]
        okv = np.isfinite(full["y"].to_numpy(float)[L - 1:])
        steps.append({"what": f"앞 {L - 1}행은 창({L}행)을 채울 과거가 없어 정답으로 못 씀(입력으로만)", "n": len(full) - (L - 1), "minus": L - 1})
        miss = int((~okv).sum())
        if miss:
            steps.append({"what": "정답(타겟)이 없는 행", "n": int(okv.sum()), "minus": miss})
        if not accepts_nan and allnan:
            okv[:] = False
            steps.append({"what": f"학습 구간 내내 빈 변수({', '.join(allnan)}) — 채울 평균도 없어 모든 창 제외", "n": 0, "minus": int(okv.size)})
        win_idx = full.index[L - 1:][okv]
        unit = "창(결정일)"
        idx_for_val = win_idx
    elif shape == "table":
        unit = "행"
        miss = int((~y_ok).sum())
        if miss:
            steps.append({"what": "정답(타겟)이 없는 행", "n": int(y_ok.sum()), "minus": miss})
        blank = int(y_ok.sum() - len(complete))
        if blank:
            steps.append({"what": "입력에 빈 칸이 있는 행(표 모델은 채우지 않고 뺌)", "n": len(complete), "minus": blank})
        idx_for_val = complete.index
    else:
        unit = "행"
        idx_for_val = complete.index if feats else full.index[y_ok.values]
    n_all = len(idx_for_val)
    if uses_val and shape in ("table", "window"):
        n_val = int((idx_for_val >= f.val_from).sum()) if f.val_from is not None else int(n_all * v.val_frac)
    else:
        n_val = 0
    has_val = uses_val and n_val >= MIN_VAL and n_all - n_val - emb >= MIN_VAL
    n_fit = n_all - n_val - emb if has_val else n_all
    if has_val:
        how_val = f"검증 시작일 이후 {n_val}개" if f.val_from is not None else f"뒤 {int(round(v.val_frac * 100))}% = int({n_all}×{v.val_frac}) = {n_val}개"
        steps.append({"what": f"검증으로 뗌 — {how_val} (언제 멈출지 정하는 데만 씀)", "n": n_all - n_val, "minus": n_val})
        steps.append({"what": f"학습과 검증 사이 {emb}개 비움", "n": n_fit, "minus": emb})
    steps.append({"what": "= 학습(가중치를 맞추는 데 쓰는 것)", "n": n_fit, "final": True})
    fit_idx = idx_for_val[:n_fit]
    val_idx = idx_for_val[n_all - n_val:] if has_val else idx_for_val[:0]
    test_pred = te_rows.dropna().index if shape == "table" else te_rows.index
    rows = [{"part": "학습(가중치를 맞춤)", **_span(fit_idx), "unit": unit}]
    if has_val:
        rows.append({"part": "비움(학습 정답이 검증 값을 쓰지 않게)", "n": emb, "unit": unit})
        rows.append({"part": "검증(언제 멈출지)", **_span(val_idx), "unit": unit})
    rows.append({"part": "비움(학습 정답이 테스트 값을 쓰지 않게)", "n": emb, "unit": "결정일"})
    rows.append({"part": "테스트(성적)", **_span(test_pred), "unit": "결정일"})
    n_used = len(fit_idx) + len(val_idx) + len(test_pred)
    used_pct = {"학습": round(100 * len(fit_idx) / max(n_used, 1), 1), "검증": round(100 * len(val_idx) / max(n_used, 1), 1),
                "테스트": round(100 * len(test_pred) / max(n_used, 1), 1)}

    if cfg.model == "lightgbm_model" and has_val and n_val <= MIN_VAL:
        notes.append(f"검증 {n_val}개 — 떼어 두기는 하지만 LightGBM 은 검증이 11개 이상일 때만 조기 종료를 합니다(나무를 n_estimators 개 모두 사용).")
    if uses_val and shape in ("table", "window") and not has_val:
        notes.append(f"검증 {n_val}개 < {MIN_VAL} 또는 검증을 떼고 남는 학습 < {MIN_VAL} — 검증 없이 학습 구간 전체로 학습합니다(조기 종료 없음)."
                     + (" BIVA 는 이때 학습 손실이 가장 낮던 시점의 가중치를 고릅니다." if cfg.model == "biva" else ""))
    if cfg.model == "biva" and has_val and len(val_idx) < int(params.get("batch_size", 1)):
        notes.append(f"검증 창 {len(val_idx)}개 < batch_size {params.get('batch_size')} — BIVA 는 검증 손실 대신 학습 손실로 최적 시점을 고릅니다.")
    if has_val:
        notes.append(f"학습과 검증 사이, 검증과 테스트 사이에 각각 {emb}개를 비웁니다(결정일의 정답이 {emb}기간 뒤의 값까지 쓰기 때문 — "
                     "비우지 않으면 학습 정답이 다음 구간의 값을 미리 보게 됩니다).")
    else:
        notes.append(f"학습과 테스트 사이에 {emb}개를 비웁니다(결정일의 정답이 {emb}기간 뒤의 값까지 쓰기 때문).")
    if L is not None:
        notes.append(f"창 길이 {L}: 테스트(검증)의 첫 창은 바로 앞 {L - 1}개 행을 입력으로 빌려 씁니다(이미 지난 값이라 누출 아님).")
    if shape == "table":
        d_te = int(len(te_rows) - len(test_pred))
        if d_te:
            notes.append(f"입력에 빈 칸이 있는 테스트 결정일 {d_te}개는 예측하지 않습니다(성적에서 빠짐).")
    return {"rows": rows, "used_pct": used_pct, "steps": steps, "blocking": blocking, "notes": notes, "accepts_nan": accepts_nan,
            "n_fit": n_fit, "n_val": len(val_idx), "n_test": len(test_pred), "unit": unit,
            "window": {"seq_len": L, "warmup": L - 1, "accepts_nan": accepts_nan} if L is not None else None}


def walk_plan(cfg_dict: dict, dv, reg) -> dict:
    """워크포워드 계획(학습 없음): 폴드 수, 첫·마지막 폴드의 날짜·개수와 학습 창을 자르는 순서, 건너뛰는 폴드."""
    from engine.experiment import ExperimentConfig, _embargo, make_folds, prepare

    cfg, plugin, params, _tt, ds = prepare(ExperimentConfig(**cfg_dict), dv, reg)
    k = int(cfg.target.k)
    emb = _embargo(cfg, k)
    shape = plugin.INPUT_SHAPE
    uses_val = bool(getattr(plugin.Model, "USES_VALIDATION", True)) and shape in ("table", "window")
    folds, _ = make_folds(cfg, ds, k)
    v = cfg.validation
    out: dict = {"mode": "walkforward", "model": cfg.model, "shape": shape, "embargo": emb, "uses_validation": uses_val,
                 "seq_len": int(cfg.seq_len) if shape == "window" else None,
                 "window": {"train_len": v.train_len, "test_len": v.test_len, "kind": "밀려가는 창(길이 고정) — 시작점도 함께 test_len 씩 이동"},
                 "n_folds": len(folds), "notes": [], "blocking": []}
    skipped = []
    for i, f in enumerate(folds):
        tr = ds.X.loc[f.train].join(ds.y.loc[f.train].rename("y")).dropna()
        te = ds.X.loc[f.test]
        if len(tr) < MIN_ROWS or (te.dropna().empty if len(ds.feature_names) else len(te) == 0):
            skipped.append(i + 1)
    out["skipped_folds"] = skipped
    if skipped:
        out["blocking"].append(f"폴드 {len(skipped)}개(번호 {skipped[0]}~{skipped[-1]})는 학습 창에 입력·정답이 모두 있는 행이 {MIN_ROWS}개 미만이라 엔진이 건너뜁니다 — "
                               "늦게 시작하는 변수 때문인 경우가 많습니다(아래 빈 칸 표의 '첫 값이 있는 날').")
    run_idx = [i for i in range(len(folds)) if i + 1 not in skipped]
    out["test_span"] = {"first": _d(folds[run_idx[0]].test[0]) if run_idx else None, "last": _d(folds[run_idx[-1]].test[-1]) if run_idx else None,
                        "n": int(sum(len(folds[i].test) for i in run_idx)), "n_folds_run": len(run_idx)}
    picks = [("첫 폴드", run_idx[0])] + ([("마지막 폴드", run_idx[-1])] if len(run_idx) > 1 else []) if run_idx else []
    out["folds"] = []
    accepts_nan = False
    for name, i in picks:
        fc = fold_counts(cfg, plugin, params, ds, folds[i], emb, uses_val)
        out["folds"].append({"name": f"{name} (폴드 {i + 1})", "rows": fc["rows"], "steps": fc["steps"], "used_pct": fc["used_pct"]})
        if name == "첫 폴드":
            out["notes"] += fc["notes"]
            accepts_nan = fc["accepts_nan"]
    if run_idx:
        f0, f1 = folds[run_idx[0]], folds[run_idx[-1]]
        out["blanks"] = blanks_info(ds.X, list(ds.feature_names), f0.train.union(f1.train), f0.test.union(f1.test), shape, accepts_nan)
        out["notes"].append(f"폴드마다 새 모델을 처음부터 학습하고 바로 뒤 {v.test_len}개만 예측합니다. 테스트 부분만 이어 붙인 "
                            f"{out['test_span']['first']}~{out['test_span']['last']} ({out['test_span']['n']}개)로 성적을 냅니다.")
        out["notes"].append(f"학습 창은 길이 {v.train_len} 그대로 {v.test_len}씩 밀려갑니다 — 시작점도 함께 이동해 창 밖으로 밀려난 오래된 자료는 쓰지 않습니다.")
    return out


def validation_plan(cfg_dict: dict, dv, reg, ratios=None) -> dict:
    """설정의 검증 방식에 맞는 계획. 비율을 주면 1회 분할로 날짜를 계산한다."""
    mode = (cfg_dict.get("validation") or {}).get("mode", "walkforward")
    if ratios is None and mode == "walkforward":
        return walk_plan(cfg_dict, dv, reg)
    return {"mode": "split", **split_plan(cfg_dict, dv, reg, ratios=ratios)}
