"""계산 따라가기(explain) — 실제 데이터·실제 코드로 한 폴드를 돌리고, 결정일 하루의 계산을 중간값까지 전부 꺼내 보여 준다.

원칙
  1. 다시 구현하지 않는다. 실험 실행과 같은 함수(prepare → make_folds → _walkforward_predict)를 폴드 하나에 대해 부르고,
     그 코드가 실제로 쓴 객체(학습 표·스케일러·적합된 모델)를 capture 로 받아 읽기만 한다.
  2. 손으로 다시 계산한 값과 모델 출력이 같은지 숫자로 확인한다(check). 다르면 화면에 '불일치'로 드러난다.
  3. 생략하지 않는다. 변수 전부, 학습 표 전부(행 수만큼), 테스트 구간 예측 전부를 돌려준다.

출력(dict): 공통(model, decision_date, fold, target, …) + steps(모델별 단계 목록) + checks(검증 목록).
각 step = {"id", "title", "plain"(쉬운 설명), "formula"(정확한 식), "table"?: {"columns", "rows"}, "values"?: {...}}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.experiment import ExperimentConfig, _embargo, _walkforward_predict, make_folds, prepare
from engine.features import describe_feature

TARGET_TEXT = {
    "return": "r = P[t+1+k] ÷ P[t+1] − 1 (P = 시가, t 마감 후 결정 → t+1 시가 진입 → t+1+k 시가 청산)",
    "logreturn": "ln(P[t+1+k] ÷ P[t+1]) (P = 시가)",
    "direction": "1 이면 r > 0(올랐음), 0 이면 아님. r = P[t+1+k] ÷ P[t+1] − 1",
    "event": "1 이면 r > 문턱, 0 이면 아님. r = P[t+1+k] ÷ P[t+1] − 1",
    "volatility": "그 뒤 k 기간의 실현 변동성 = √( (1/k)·Σ (ln(C[t+j] ÷ C[t+j−1]))² ), j = 1…k (C = 종가)",
}
FC_TARGET_TEXT = {
    "return": "v[t+k] ÷ v[t] − 1 (v = 예측 대상의 값)",
    "logreturn": "ln(v[t+k] ÷ v[t])",
    "direction": "1 이면 v[t+k] > v[t]",
}


def _f(x) -> float | None:
    """JSON 용 숫자(NaN·inf → None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(o):
    """JSON 으로 보낼 수 있게 numpy 값을 파이썬 값으로(NaN·inf → None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return _f(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _d(t) -> str:
    return str(pd.Timestamp(t).date())


def explain(cfg_dict: dict, dv, reg, *, date: str | None = None, epochs: int | None = None) -> dict:
    """epochs: 계산 따라가기에서만 쓰는 학습 반복 수(BIVA·LSTM·GRU). 주면 그 값으로 학습하고 결과에 그 사실을 적는다(실험 설정과 다를 수 있음)."""
    return _clean(_explain(cfg_dict, dv, reg, date=date, epochs=epochs))


def _explain(cfg_dict: dict, dv, reg, *, date: str | None = None, epochs: int | None = None) -> dict:
    cfg_dict = dict(cfg_dict)
    ep_key = {"biva": "train_epochs", "lstm": "epochs", "gru": "epochs"}.get(cfg_dict.get("model"))
    set_epochs = None
    if epochs and ep_key:
        set_epochs = (dict(cfg_dict.get("model_params") or {}).get(ep_key), int(epochs))
        cfg_dict["model_params"] = {**(cfg_dict.get("model_params") or {}), ep_key: int(epochs)}
    cfg = ExperimentConfig(**cfg_dict)
    # 봉인 정책: 계산 따라가기도 개발 구간의 폴드만 보여 준다(봉인 시작일이 비어 있는 설정이면 정책 날짜를 채운다) — 실험 실행과 같은 규칙
    from engine import sealpolicy
    cfg, _seal_note = sealpolicy.apply(cfg, sealed=False)
    if cfg.pool_tickers:
        raise ValueError("종목 묶어 학습(pool_tickers)을 쓴 실험은 계산 따라가기를 아직 지원하지 않습니다")
    if cfg.select_top:
        raise ValueError("입력 선별(select_top)을 쓴 실험은 계산 따라가기를 아직 지원하지 않습니다 — 폴드마다 쓰는 입력이 달라 화면이 그것을 보여 주지 못합니다")
    n_seeds_cfg = int(cfg.n_seeds)
    if n_seeds_cfg > 1:
        cfg = cfg.model_copy(update={"n_seeds": 1})  # 계산 따라가기는 모델 하나의 계산을 보여 준다 → 첫 번째 시드(반복 0)만
    cfg, plugin, params, target_type, ds = prepare(cfg, dv, reg, device="cpu")
    k = cfg.target.k
    out: dict = {"model": cfg.model, "name": plugin.NAME, "input_shape": plugin.INPUT_SHAPE, "params": params, "target_type": target_type,
                 "mode": cfg.mode, "frequency": cfg.frequency, "k": k, "notes": [], "steps": [], "checks": []}
    tk = cfg.target.kind
    out["target"] = {"kind": tk, "k": k, "text": (FC_TARGET_TEXT if cfg.mode == "forecast" else TARGET_TEXT).get(tk, tk),
                     "variable": cfg.target_variable if cfg.mode == "forecast" else cfg.ticker}
    out["_y"] = ds.y
    if n_seeds_cfg > 1 and bool(getattr(plugin.Model, "STOCHASTIC", False)):
        out["notes"].append(f"이 실험은 시드 {n_seeds_cfg}개로 학습해 예측을 평균합니다. 여기서는 그중 첫 번째 시드의 모델 하나만 따라갑니다 — "
                            "계산 방법은 같고, 숫자는 실험 결과(평균)와 조금 다릅니다.")
    if set_epochs:
        out["explain_epochs"] = set_epochs[1]
        out["notes"].append(f"계산 따라가기용으로 학습 반복을 {set_epochs[1]}회로 바꿔 돌렸습니다(실험 설정 {set_epochs[0] if set_epochs[0] is not None else '기본값 ' + str(plugin.PARAMS[ep_key].get('default'))}회). "
                            "계산 방법·순서는 같고, 학습이 덜 된 만큼 숫자는 실험 결과와 다릅니다.")
    if plugin.INPUT_SHAPE == "rule":
        _rule(out, cfg, plugin, params, ds, reg, date)
        out["ok"] = all(c["ok"] for c in out["checks"])
        return out
    folds, is_split = make_folds(cfg, ds, k)
    folds = [f for f in folds if len(f.test)]
    if not folds:
        raise ValueError("평가할 폴드가 없습니다 — 기간·학습 창 설정을 확인하세요")
    want = pd.Timestamp(date) if date else None
    fi = len(folds) - 1
    if want is not None:
        hit = [i for i, f in enumerate(folds) if f.test[0] <= want <= f.test[-1]]
        if not hit:
            raise ValueError(f"{want.date()} 는 평가 구간({_d(folds[0].test[0])} ~ {_d(folds[-1].test[-1])})의 결정일이 아닙니다")
        fi = hit[0]
    fold = folds[fi]
    cap: dict = {}
    _walkforward_predict(cfg, plugin, params, ds, "cpu", [fold], k, target_type, capture=cap)
    if not cap:
        raise ValueError("이 폴드는 학습 표본이 30개 미만이거나 테스트 입력이 비어 건너뛰는 폴드입니다 — 다른 날짜를 고르세요")
    pred: pd.Series = cap["pred"].dropna()
    if pred.empty:
        raise ValueError("이 폴드에서 예측이 나온 날이 없습니다")
    t = want if (want is not None and want in pred.index) else pred.index[-1]
    if want is not None and want not in pred.index:
        out["notes"].append(f"{want.date()} 은(는) 입력에 빈 칸이 있어 예측이 없는 날이라 같은 폴드의 마지막 예측일 {t.date()} 로 보여 줍니다")
    out["decision_date"] = _d(t)
    out["available_dates"] = [_d(x) for x in pred.index]
    out["fold"] = {"index": fi + 1, "count": len(folds), "is_split": is_split, "embargo": _embargo(cfg, k),
                   "train": [_d(fold.train[0]), _d(fold.train[-1])], "n_train_calendar": int(len(fold.train)),
                   "test": [_d(fold.test[0]), _d(fold.test[-1])], "n_test": int(len(fold.test)),
                   "train_len": cfg.validation.train_len, "test_len": cfg.validation.test_len}
    y_true = ds.y.get(t, np.nan)
    out["actual"] = _f(y_true)
    _common_steps(out, cfg, ds, fold, t, fi, len(folds), is_split)
    if cap["kind"] == "table" and cfg.model == "ridge":
        _ridge(out, cfg, ds, reg, cap, t, target_type)
    elif cap["kind"] == "table" and cfg.model == "lightgbm_model":
        _lgbm(out, cfg, ds, reg, cap, t, target_type)
    elif cap["kind"] == "series":
        _series(out, cfg, ds, cap, t)
    elif cap["kind"] == "levels":
        _vecm(out, cfg, ds, cap, t)
    elif cap["kind"] == "window" and cfg.model in ("lstm", "gru"):
        _rnn(out, cfg, ds, reg, cap, t, target_type)
    elif cap["kind"] == "window" and cfg.model == "biva":
        _biva(out, cfg, ds, reg, cap, t)
    else:
        raise ValueError(f"{plugin.NAME} 계산 따라가기는 아직 준비 중입니다")
    _decision(out, cfg, cap, t, target_type)
    out["ok"] = all(c["ok"] for c in out["checks"])
    return out


def _common_steps(out, cfg, ds, fold, t, fi, nf, is_split):
    """모든 모델 공통: 이 결정일이 속한 폴드(학습·비움·테스트 날짜)와, 이 날의 정답(타겟)을 실제 가격으로 만드는 과정."""
    k = cfg.target.k
    emb = _embargo(cfg, k)
    cal_all = ds.prices.index
    pos_tr_end = cal_all.get_loc(fold.train[-1])
    gap = [_d(x) for x in cal_all[pos_tr_end + 1: cal_all.get_loc(fold.test[0])]]
    v = cfg.validation
    out["steps"].append({"id": "fold", "title": f"기간 — 폴드 {fi + 1} / {nf}",
                         "plain": ("1회 분할: 테스트 시작일 전의 날짜 전부로 한 번 학습하고 그 뒤를 예측합니다." if is_split else
                                   f"워크포워드: 과거 {v.train_len}개 날짜로 학습 → 다음 {v.test_len}개 날짜를 예측 → {v.test_len}칸 밀어 다시 학습, 을 반복합니다. 이 결정일은 그중 {fi + 1}번째 반복(폴드)에 속합니다.")
                                  + f" 학습 창 끝의 {emb}개 날짜는 비웁니다(embargo): 그 날들의 정답은 테스트 구간의 가격으로 계산되기 때문에 학습에 넣으면 미래를 엿보게 됩니다.",
                         "formula": f"학습 = [{_d(fold.train[0])} … {_d(fold.train[-1])}] ({len(fold.train)}개) · 비움 = {gap if gap else '(학습 창 안에서 뺌)'} · 테스트 = [{_d(fold.test[0])} … {_d(fold.test[-1])}] ({len(fold.test)}개)",
                         "values": {"학습 창(설정)": v.train_len, "비움 embargo": emb, "실제 학습 날짜 수": len(fold.train), "테스트 날짜 수": len(fold.test)}})
    if cfg.mode == "trading":
        o = ds.prices["open"]
        i = cal_all.get_loc(t)
        if i + 1 + k < len(cal_all):
            d1, d2 = cal_all[i + 1], cal_all[i + 1 + k]
            p1, p2 = float(o.iloc[i + 1]), float(o.iloc[i + 1 + k])
            r = p2 / p1 - 1
            vals = {"진입일 t+1": _d(d1), "진입 시가 P[t+1]": p1, f"청산일 t+1+{k}": _d(d2), f"청산 시가 P[t+1+{k}]": p2, "수익률 r": r,
                    "타겟 값(데이터셋)": _f(ds.y.get(t, np.nan))}
        else:
            vals = {"상태": "아직 청산일 가격이 없어 정답이 없음(마지막 1+k 행)"}
        out["steps"].append({"id": "target", "title": "정답(타겟) — 이 날 결정했다면 실제로 번 수익률",
                             "plain": "결정은 장 마감 후에 하므로 그날 종가로는 살 수 없습니다. 그래서 다음 날 시가에 사서 k 기간 뒤 시가에 판 수익률이 정답입니다. 모델은 이 값을 맞히도록 학습합니다.",
                             "formula": out["target"]["text"], "values": vals})


def _decision(out, cfg, cap, t, target_type):
    """예측 → 신호(엔진 규칙 그대로) → 다음 날 행동, 그리고 이 폴드 테스트 구간의 예측 전부."""
    p = float(cap["pred"].loc[t])
    thr = float(cfg.trade.threshold)
    if cap.get("kind") == "rule":
        pass
    elif cfg.mode == "trading":
        cut = thr if target_type == "regression" else 0.5 + thr
        sig = 1.0 if p > cut else 0.0
        out["steps"].append({"id": "decide", "title": "결정 (신호)",
                             "plain": "예측이 문턱보다 크면 다음 날 아침 시가에 사서(이미 있으면 계속) 들고 가고, 아니면 현금으로 둡니다. 체결은 항상 다음 날 시가, 사고팔 때마다 수수료·슬리피지를 냅니다.",
                             "formula": ("신호 = 1 if ŷ > 진입 임계치 else 0" if target_type == "regression" else "신호 = 1 if 확률 > 0.5 + 진입 임계치 else 0")
                             + " (engine/experiment.py 신호 변환)",
                             "values": {"예측 ŷ": p, "문턱": cut, "신호": sig, "행동": "다음 날 시가에 보유(매수 또는 유지)" if sig else "다음 날 시가에 현금(매도 또는 유지)"}})
    rows = []
    for d, v in cap["pred"].items():
        yt = out["_y"].get(d, np.nan) if "_y" in out else np.nan
        rows.append([_d(d), _f(v), _f(yt)])
    out.pop("_y", None)
    out["fold_predictions"] = {"columns": ["결정일", "예측", "실제(타겟)"], "rows": rows}


def _ridge(out, cfg, ds, reg, cap, t, target_type):
    m = cap["model"]
    feats = list(ds.feature_names)
    tr: pd.DataFrame = cap["tr"]
    te_valid: pd.DataFrame = cap["te_valid"]
    sc1 = cap["scaler"]
    pipe = m.pipe
    sc2, est = pipe[0], pipe[-1]
    alpha = float(m.params["alpha"])
    K = len(feats)
    desc = {c: describe_feature(c, reg) for c in feats}
    n_cal = len(cap["fold"].train)

    # 1. 학습 표
    out["steps"].append({"id": "train_table", "title": "학습 표 (이 폴드에서 비율을 정하는 데 쓴 과거 전부)",
                         "plain": f"학습 창의 날짜 {n_cal}개 중, 재료 {K}개와 정답(타겟)이 모두 있는 {len(tr)}개 행을 씁니다"
                                  + (f" — {n_cal - len(tr)}개 행은 빈 칸이 있어 뺐습니다." if n_cal != len(tr) else ".")
                                  + " 정답은 그날 결정했다면 실제로 벌었을 수익률입니다. 학습 창 끝의 비움(embargo) 행은 이미 빠져 있습니다.",
                         "formula": "tr = (X ⋈ y)[학습 창].dropna()   (engine/experiment.py _walkforward_predict)",
                         "table": {"columns": ["결정일", *feats, "정답 y"],
                                   "rows": [[_d(i), *[_f(v) for v in r[feats].values], _f(r["y"])] for i, r in tr.iterrows()]}})
    # 2. 오늘의 재료
    x = te_valid.loc[t, feats].astype(float).values
    out["steps"].append({"id": "inputs", "title": f"결정일 {t.date()} 의 재료 {K}개",
                         "plain": "장 마감 후 그날까지 알려진 값만으로 만든 숫자입니다(발표시차 적용). 미래 값은 들어 있지 않습니다.",
                         "formula": "x_t = X.loc[t]",
                         "table": {"columns": ["열", "뜻", "값 x"], "rows": [[c, desc[c], _f(v)] for c, v in zip(feats, x)]}})
    # 3. 표준화 (엔진) + 4. 표준화 (모델 내부)
    z1 = (x - sc1.mean_) / sc1.scale_
    z1_code = cap["Xte"][list(te_valid.index).index(t)]
    z2 = (z1 - sc2.mean_) / sc2.scale_
    out["steps"].append({"id": "scale", "title": "눈금 맞추기 (표준화) — 평소보다 몇 칸 큰가",
                         "plain": "단위가 다른 재료를 같은 눈금으로 바꿉니다. '평소 값'과 '평소 흔들림'은 위 학습 표로만 계산합니다(미래·테스트 구간은 쓰지 않음). "
                                  "Ridge 는 엔진에서 한 번, 모델 안(make_pipeline)에서 한 번 더 표준화합니다. 두 번째는 이미 평균 0·흔들림 1 인 값을 다시 맞추는 것이라 사실상 바뀌지 않습니다(표의 두 번째 평균·흔들림이 0·1 에 가까움).",
                         "formula": "z = (x − 평균) ÷ 표준편차,  평균·표준편차 = 학습 표의 열별 값(표준편차는 n 으로 나눔, ddof=0 — sklearn StandardScaler).  z′ = (z − 평균′) ÷ 표준편차′ (모델 안 두 번째)",
                         "table": {"columns": ["열", "값 x", "평균", "표준편차", "z", "평균′", "표준편차′", "z′ (모델이 쓰는 값)"],
                                   "rows": [[c, _f(x[j]), _f(sc1.mean_[j]), _f(sc1.scale_[j]), _f(z1[j]), _f(sc2.mean_[j]), _f(sc2.scale_[j]), _f(z2[j])] for j, c in enumerate(feats)]}})
    out["checks"].append({"what": "손으로 계산한 z = 엔진이 계산한 z", "max_abs_diff": _f(np.max(np.abs(z1 - z1_code))), "ok": bool(np.allclose(z1, z1_code, rtol=0, atol=1e-12))})

    w = np.ravel(est.coef_)
    b = float(np.ravel(est.intercept_)[0])
    contrib = w * z2
    score = b + contrib.sum()
    model_out = float(cap["pred"].loc[t])
    order = np.argsort(-np.abs(contrib))
    out["steps"].append({"id": "weighted_sum", "title": "비율대로 더하기 (가중합)",
                         "plain": "재료마다 학습으로 정해진 비율(w)을 곱해 모두 더하고 기본값(b)을 더합니다. 비율 = 그 재료가 평소보다 1칸 클 때 예측이 얼마나 움직이는가. "
                                  "표는 영향이 큰 순서(기여의 크기)로 정렬했습니다.",
                         "formula": "점수 = b + Σ_j w_j · z′_j" + ("  → 예측 ŷ = 점수" if target_type == "regression" else "  → 확률 = 1 ÷ (1 + e^(−점수))"),
                         "table": {"columns": ["열", "뜻", "z′", "비율 w", "기여 w·z′"], "rows": [[feats[j], desc[feats[j]], _f(z2[j]), _f(w[j]), _f(contrib[j])] for j in order]},
                         "values": {"기본값 b (절편)": b, "기여의 합 Σ w·z′": _f(contrib.sum()), "점수": _f(score)}})
    if target_type == "regression":
        manual = score
        out["checks"].append({"what": "손으로 더한 ŷ = 모델이 낸 ŷ (pipe.predict)", "manual": _f(manual), "model": model_out,
                              "max_abs_diff": _f(abs(manual - model_out)), "ok": bool(abs(manual - model_out) <= 1e-10 * max(1.0, abs(model_out)))})
    else:
        manual = 1.0 / (1.0 + np.exp(-score))
        out["steps"][-1]["values"]["확률 = 1 ÷ (1 + e^(−점수))"] = _f(manual)
        out["checks"].append({"what": "손으로 계산한 확률 = 모델이 낸 확률 (predict_proba)", "manual": _f(manual), "model": model_out,
                              "max_abs_diff": _f(abs(manual - model_out)), "ok": bool(abs(manual - model_out) <= 1e-10)})

    # 5. 비율을 정한 방법 (학습)
    Ztr = sc2.transform(sc1.transform(tr[feats].values))
    ytr = tr["y"].values.astype(float)
    if target_type == "regression":
        fitted = Ztr @ w + b
        sse = float(np.sum((ytr - fitted) ** 2))
        pen = float(alpha * np.sum(w ** 2))
        # 닫힌 해로 다시 구해 sklearn 과 비교: 절편은 벌점 없이, 열과 y 를 평균 중심화한 뒤 w = (ZᵀZ + αI)⁻¹ Zᵀ y
        Zc = Ztr - Ztr.mean(axis=0)
        yc = ytr - ytr.mean()
        w_cf = np.linalg.solve(Zc.T @ Zc + alpha * np.eye(K), Zc.T @ yc)
        b_cf = float(ytr.mean() - Ztr.mean(axis=0) @ w_cf)
        r2 = 1 - sse / float(np.sum((ytr - ytr.mean()) ** 2)) if len(ytr) > 1 else float("nan")
        out["steps"].append({"id": "fit", "title": "비율(w)과 기본값(b)을 정한 방법 — 학습",
                             "plain": f"위 학습 표 {len(tr)}행 전부에 대해 '예측과 정답의 차이를 제곱해 더한 값'에, 비율이 클수록 커지는 벌점(alpha × 비율 제곱의 합)을 더한 점수가 "
                                      f"가장 작아지는 비율을 찾습니다. 이 답은 반복 없이 공식 한 번으로 정확히 나옵니다(닫힌 해). 기본값 b 에는 벌점이 없습니다. "
                                      f"alpha 가 클수록 비율이 0 쪽으로 작아져 예측이 평균(b) 근처로 모입니다. 지금 alpha = {alpha:g}.",
                             "formula": "(w, b) = argmin Σ_i (y_i − b − Σ_j w_j z′_ij)² + α Σ_j w_j²   →   w = (Z̃ᵀZ̃ + αI)⁻¹ Z̃ᵀỹ,  b = mean(y) − mean(z′)ᵀw   (sklearn Ridge, fit_intercept=True)\n"
                                        "기호: i = 학습 표의 날(행), j = 재료(열), z′_ij = i번째 날 j번째 재료의 칸 수, y_i = 정답, α = alpha, I = 단위행렬, "
                                        "Z̃ = z′ 에서 열마다 평균을 뺀 행렬, ỹ = y 에서 평균을 뺀 값. z′ 는 이미 학습 표로 표준화돼 열 평균이 0 이라 b = 정답 평균이 됩니다.",
                             "values": {"학습 행 수": len(tr), "alpha": alpha, "틀린 정도의 합 Σ(y−ŷ)²": sse, "벌점 α·Σw²": pen, "최소화한 점수(합)": sse + pen,
                                        "정답 평균 ȳ": _f(ytr.mean()), "학습 구간 설명력 R²": _f(r2)}})
        d_w = float(np.max(np.abs(w_cf - w)))
        out["checks"].append({"what": "닫힌 해 공식으로 다시 구한 w·b = sklearn 이 학습한 w·b", "max_abs_diff": _f(max(d_w, abs(b_cf - b))),
                              "ok": bool(d_w <= 1e-8 * max(1.0, float(np.max(np.abs(w)))) and abs(b_cf - b) <= 1e-8 * max(1.0, abs(b)))})
    else:
        logit = Ztr @ w + b
        pr = 1 / (1 + np.exp(-logit))
        eps = 1e-15
        ll = float(-np.sum(ytr * np.log(pr + eps) + (1 - ytr) * np.log(1 - pr + eps)))
        out["steps"].append({"id": "fit", "title": "비율(w)과 기본값(b)을 정한 방법 — 학습 (로지스틱)",
                             "plain": f"정답이 1(오름)/0(아님)인 설정이라, 점수를 확률로 바꾼 뒤 '정답에 낮은 확률을 준 정도'(로그손실)의 합에 벌점(alpha/2 × 비율 제곱의 합)을 더한 값이 "
                                      f"가장 작아지도록 비율을 찾습니다. 공식 한 번으로는 안 풀려 반복 계산(lbfgs)으로 찾습니다. 지금 alpha = {alpha:g} (sklearn C = 1/alpha).",
                             "formula": "(w, b) = argmin −Σ_i [y_i ln p_i + (1−y_i) ln(1−p_i)] + (α/2) Σ_j w_j²,  p_i = 1/(1+e^(−(b + Σ w_j z′_ij)))",
                             "values": {"학습 행 수": len(tr), "alpha": alpha, "로그손실 합": ll, "벌점 (α/2)·Σw²": float(alpha / 2 * np.sum(w ** 2)),
                                        "반복 횟수": int(np.ravel(est.n_iter_)[0])}})



# ---------------------------------------------------------------------------- 공통 도우미
def _train_table_step(out, tr, feats, n_cal, K, what="비율"):
    out["steps"].append({"id": "train_table", "title": "학습 표 (이 폴드에서 모델을 맞추는 데 쓴 과거 전부)",
                         "plain": f"학습 창의 날짜 {n_cal}개 중, 재료 {K}개와 정답(타겟)이 모두 있는 {len(tr)}개 행을 씁니다"
                                  + (f" — {n_cal - len(tr)}개 행은 빈 칸이 있어 뺐습니다." if n_cal != len(tr) else ".")
                                  + " 정답은 그날 결정했다면 실제로 벌었을 수익률(또는 오름 여부)입니다. 학습 창 끝의 비움(embargo) 행은 이미 빠져 있습니다.",
                         "formula": "tr = (X ⋈ y)[학습 창].dropna()   (engine/experiment.py _walkforward_predict)",
                         "table": {"columns": ["결정일", *feats, "정답 y"],
                                   "rows": [[_d(i), *[_f(v) for v in r[feats].values], _f(r["y"])] for i, r in tr.iterrows()]}})


def _scale_step(out, feats, desc, x, sc, z, z_code, extra=""):
    out["steps"].append({"id": "scale", "title": "눈금 맞추기 (표준화) — 평소보다 몇 칸 큰가",
                         "plain": "단위가 다른 재료를 같은 눈금으로 바꿉니다. '평소 값'(평균)과 '평소 흔들림'(표준편차)은 학습 표로만 계산합니다(테스트 구간·미래는 쓰지 않음). " + extra,
                         "formula": "z = (x − 평균) ÷ 표준편차,  평균·표준편차 = 학습 표의 열별 값(표준편차는 n 으로 나눔, ddof=0 — sklearn StandardScaler)",
                         "table": {"columns": ["열", "뜻", "값 x", "평균", "표준편차", "z"],
                                   "rows": [[c, desc[c], _f(x[j]), _f(sc.mean_[j]), _f(sc.scale_[j]), _f(z[j])] for j, c in enumerate(feats)]}})
    out["checks"].append({"what": "손으로 계산한 z = 엔진이 계산한 z", "max_abs_diff": _f(np.max(np.abs(z - z_code))), "ok": bool(np.allclose(z, z_code, rtol=0, atol=1e-12))})


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------- LightGBM
def _lgbm(out, cfg, ds, reg, cap, t, target_type):
    m = cap["model"]
    est = m.est
    feats = list(ds.feature_names)
    K = len(feats)
    tr, te_valid, sc = cap["tr"], cap["te_valid"], cap["scaler"]
    desc = {c: describe_feature(c, reg) for c in feats}
    n_tr, n_va = cap["n_tr"], cap["n_va"]
    _train_table_step(out, tr, feats, len(cap["fold"].train), K)
    x = te_valid.loc[t, feats].astype(float).values
    out["steps"].append({"id": "inputs", "title": f"결정일 {t.date()} 의 재료 {K}개",
                         "plain": "장 마감 후 그날까지 알려진 값만으로 만든 숫자입니다(발표시차 적용). 미래 값은 들어 있지 않습니다.",
                         "formula": "x_t = X.loc[t]", "table": {"columns": ["열", "뜻", "값 x"], "rows": [[c, desc[c], _f(v)] for c, v in zip(feats, x)]}})
    z = (x - sc.mean_) / sc.scale_
    _scale_step(out, feats, desc, x, sc, z, cap["Xte"][list(te_valid.index).index(t)],
                "나무는 '크다/작다'만 묻기 때문에 눈금을 바꿔도 결과는 같지만, 엔진이 모든 표 모델에 같은 전처리를 하므로 나무의 기준값도 이 눈금(z)으로 적혀 있습니다. 아래 표에는 원래 단위로 되돌린 기준값도 함께 적습니다.")
    booster = est.booster_
    dm = booster.dump_model()
    n_all = len(dm["tree_info"])
    best = int(est.best_iteration_) if getattr(est, "best_iteration_", None) else 0
    used = best if best > 0 else n_all
    p = m.params
    # 학습
    ev = getattr(est, "evals_result_", None) or {}
    curve = []
    if ev:
        key = list(ev.keys())[0]
        metric = list(ev[key].keys())[0]
        curve = [[i + 1, _f(v)] for i, v in enumerate(ev[key][metric])]
    out["steps"].append({"id": "fit", "title": "나무를 키운 방법 — 학습 (부스팅)",
                         "plain": f"학습 표의 앞쪽 {n_tr}행으로 나무를 한 그루씩 차례로 만듭니다. 첫 나무 전에는 모든 날을 정답 평균으로 예측해 두고, 나무 m 은 '지금까지의 예측이 못 맞힌 만큼(오차의 기울기)'을 맞히도록 만들어집니다. "
                                  f"나무마다 행의 {p['subsample']:.0%}·열의 {p['colsample_bytree']:.0%}만 무작위로 뽑아 씁니다(seed 0 고정). 나무 하나는 '열 j 의 값 ≤ 기준?' 질문으로 둘씩 나누며, 잎 수 ≤ {p['num_leaves']}, 깊이 ≤ {p['max_depth']}, 잎마다 표본 ≥ {p['min_child_samples']} 을 지킵니다. "
                                  + (f"검증 {n_va}행(학습 표 뒤쪽, 그 앞 비움 제외)의 오차가 50그루 연속 나아지지 않아 멈췄고, 가장 좋았던 {best}그루까지만 씁니다." if n_va and best and best < n_all else
                                     f"검증 {n_va}행으로 지켜봤지만 끝까지 나아져 {n_all}그루를 모두 씁니다." if n_va else f"검증이 없어(검증 11개 미만) {n_all}그루를 모두 씁니다."),
                         "formula": "F_0 = 평균(y) (분류는 로그오즈);  F_m(x) = F_{m−1}(x) + 학습률 × 나무_m(x);  나무_m 은 손실의 음의 기울기에 맞춤 (LightGBM, Ke et al. 2017). 잎 값 = −(기울기 합) ÷ (헤시안 합 + reg_lambda) × 학습률",
                         "values": {"학습 행": n_tr, "검증 행": n_va, "만든 나무": n_all, "쓰는 나무": used, "학습률": p["learning_rate"], "잎 수 상한": p["num_leaves"], "깊이 상한": p["max_depth"],
                                    "잎 최소 표본": p["min_child_samples"], "행 추출 비율": p["subsample"], "열 추출 비율": p["colsample_bytree"], "reg_lambda": p["reg_lambda"]},
                         **({"table": {"columns": ["나무 수", "검증 손실(" + metric + ")"], "rows": curve}} if curve else {})})
    # 경로
    rows, total = [], 0.0
    uses = np.zeros(K, int)

    def walk(node):
        path = []
        while "leaf_value" not in node:
            j = node["split_feature"]; thr = node["threshold"]
            if node["decision_type"] != "<=":
                raise ValueError(f"예상하지 못한 분할 종류 {node['decision_type']}")
            left = z[j] <= thr
            uses[j] += 1
            path.append(f"{feats[j]}: z={z[j]:.4g} {'≤' if left else '>'} {thr:.4g} (원래 단위 {thr * sc.scale_[j] + sc.mean_[j]:.4g}) → {'왼쪽' if left else '오른쪽'}")
            node = node["left_child"] if left else node["right_child"]
        return float(node["leaf_value"]), path, node.get("leaf_index")
    for i, ti in enumerate(dm["tree_info"][:used]):
        lv, path, li = walk(ti["tree_structure"])
        total += lv
        rows.append([i + 1, " ; ".join(path) if path else "(나누지 않은 나무: 잎 하나)", li, _f(lv), _f(total)])
    raw_model = float(booster.predict(z[None, :], num_iteration=used, raw_score=True)[0])
    model_out = float(cap["pred"].loc[t])
    out["steps"].append({"id": "trees", "title": f"나무 {used}그루를 따라 내려가기 — 그루마다 잎 값 하나씩",
                         "plain": "각 나무에서 그날의 z 값으로 질문에 답하며 내려가 도착한 잎의 값을 읽습니다. 잎 값을 모두 더한 것이 점수입니다(첫 나무의 잎 값에 시작값 F_0 가 들어 있음). 표에 모든 나무의 경로를 빠짐없이 적었습니다.",
                         "formula": "점수 = Σ_m 잎값_m(x)" + ("  → 예측 ŷ = 점수" if target_type == "regression" else "  → 확률 = 1 ÷ (1 + e^(−점수))"),
                         "table": {"columns": ["나무", "질문과 답(경로)", "잎 번호", "잎 값", "누적 합"], "rows": rows},
                         "values": {"점수(잎 값의 합)": _f(total), **({"확률": _f(_sig(total))} if target_type != "regression" else {})}})
    out["steps"].append({"id": "uses", "title": "이 날의 경로에서 각 열이 질문에 쓰인 횟수",
                         "plain": "어떤 재료가 이 날의 판단에 많이 쓰였는지 보여 줍니다(결과 화면의 중요도는 '분할 이득의 합'이라 이것과 다름).",
                         "formula": "횟수_j = Σ_m [나무_m 의 경로에 열 j 가 나온 횟수]",
                         "table": {"columns": ["열", "뜻", "쓰인 횟수"], "rows": [[feats[j], desc[feats[j]], int(uses[j])] for j in np.argsort(-uses)]}})
    out["checks"].append({"what": "잎 값을 손으로 더한 점수 = LightGBM raw_score", "manual": _f(total), "model": raw_model, "max_abs_diff": _f(abs(total - raw_model)), "ok": bool(abs(total - raw_model) <= 1e-9)})
    manual = total if target_type == "regression" else _sig(total)
    out["checks"].append({"what": "손으로 구한 예측 = 모델이 낸 예측", "manual": _f(manual), "model": model_out, "max_abs_diff": _f(abs(manual - model_out)), "ok": bool(abs(manual - model_out) <= 1e-9)})


# ---------------------------------------------------------------------------- ARIMA / SARIMA
def _series(out, cfg, ds, cap, t):
    m = cap["model"]
    res = m.res
    k = cfg.target.k
    skip = cap["skip"]
    r_all: pd.Series = cap["r_all"]
    r_tr = cap["r_tr"]
    p = m.params
    seasonal = cfg.model == "sarima"
    tr_idx = r_all.loc[cap["fold"].train].dropna().index
    out["steps"].append({"id": "input", "title": "재료 — 대상 하나의 1기간 변화율만",
                         "plain": "이 모델은 다른 변수·기술지표를 전혀 쓰지 않고, 대상 가격 자체의 '어제 종가 대비 오늘 종가 변화율' 하나만 봅니다. 학습 창의 변화율 전부로 계수를 정합니다.",
                         "formula": "r_τ = 종가_τ ÷ 종가_(τ−1) − 1",
                         "table": {"columns": ["날짜", "종가", "변화율 r"], "rows": [[_d(i), _f(ds.prices["close"].loc[i]), _f(r_all.loc[i])] for i in tr_idx]}})
    names = list(res.param_names)
    sc = float(getattr(m, "scale", 1.0) or 1.0)  # 수치 배율: 추정은 r × sc 단위 — 화면에는 원래 단위(c ÷ sc, σ² ÷ sc²)
    unit = np.array([1 / sc if n_ == "intercept" else 1 / sc ** 2 if n_ == "sigma2" else 1.0 for n_ in names])
    vals = np.asarray(res.params, dtype=float) * unit
    bse = np.asarray(getattr(res, "bse", np.full(len(vals), np.nan)), dtype=float) * unit
    pv = np.asarray(getattr(res, "pvalues", np.full(len(vals), np.nan)), dtype=float)
    kor = {"intercept": "상수 c (절편)", "sigma2": "오차 분산 σ²"}
    def nm(n):
        if n in kor: return kor[n]
        if n.startswith("ar.S.L"): return f"계절 AR Φ (시차 {n[6:]})"
        if n.startswith("ma.S.L"): return f"계절 MA Θ (시차 {n[6:]})"
        if n.startswith("ar.L"): return f"AR φ (시차 {n[4:]})"
        if n.startswith("ma.L"): return f"MA θ (시차 {n[4:]})"
        return n
    conv = bool(res.mle_retvals.get("converged", False)) if getattr(res, "mle_retvals", None) else None
    order = f"({p['p']},{p['d']},{p['q']})" + (f"({p['P']},{p['D']},{p['Q']},{p['s']})" if seasonal else "")
    out["steps"].append({"id": "fit", "title": f"계수 정하기 — 학습 ({'SARIMA' if seasonal else 'ARIMA'}{order})",
                         "plain": f"학습 창의 변화율 {len(r_tr)}개가 '가장 그럴듯하게' 나오는 계수를 찾습니다(최대우도, 반복 계산 최대 200번). "
                                  "φ(AR) = 지난 변화율이 이번에 얼마나 이어지는가, θ(MA) = 지난 예측 오차(갑작스러운 충격)가 얼마나 남는가."
                                  + (" Φ·Θ 는 s 기간 전(계절)의 같은 뜻. D=1 이면 s 기간 전과의 차이를 먼저 취합니다." if seasonal else "")
                                  + (" 정상성 제약 " + ("켬" if p.get("enforce_stationarity", False) else "끔") + "(enforce_stationarity=" + str(bool(p.get("enforce_stationarity", False))) + "), "
                                     + "가역성 제약 " + ("켬" if p.get("enforce_invertibility", False) else "끔") + "(enforce_invertibility=" + str(bool(p.get("enforce_invertibility", False))) + ")."),
                         "formula": ("φ(L)Φ(L^s)(1−L)^d(1−L^s)^D r_τ = θ(L)Θ(L^s) ε_τ" if seasonal else "r_τ = c + Σ_i φ_i r_(τ−i) + ε_τ + Σ_j θ_j ε_(τ−j)")
                                    + ",  ε ~ N(0, σ²)  (statsmodels SARIMAX, 칼만 필터 최대우도)",
                         "table": {"columns": ["이름", "뜻", "추정값", "표준오차", "p값"], "rows": [[n, nm(n), _f(v), _f(b), _f(q)] for n, v, b, q in zip(names, vals, bse, pv)]},
                         "values": {"학습 표본": len(r_tr), "로그우도(원래 단위)": _f(res.llf + (res.nobs - res.loglikelihood_burn) * np.log(sc)), "AIC": _f(res.aic - 2 * (res.nobs - res.loglikelihood_burn) * np.log(sc)),
                                    "수렴": conv, "추정 배율 scale": sc}})
    # 근 점검: AR·MA 다항식의 근 크기(가역성·정상성) — 제약을 켰으면 > 1 이 보장되고, 껐으면 이 표에서 확인한다
    def _roots(poly):
        poly = np.asarray(poly, dtype=float)
        if len(poly) <= 1 or not np.any(poly[1:]):
            return np.array([])
        return np.roots(poly[::-1])
    Ar = np.asarray(res.polynomial_ar, dtype=float); Mr = np.asarray(res.polynomial_ma, dtype=float)
    if seasonal:
        Ar = np.convolve(Ar, np.asarray(res.polynomial_seasonal_ar, dtype=float)); Mr = np.convolve(Mr, np.asarray(res.polynomial_seasonal_ma, dtype=float))
    rows_rt, ok_all = [], {"AR": True, "MA": True}
    for kind, poly in (("AR", Ar), ("MA", Mr)):
        for z in _roots(poly):
            mag = float(abs(z)); good = mag > 1
            ok_all[kind] &= good
            rows_rt.append([kind, f"{z.real:.6g}{z.imag:+.6g}i" if abs(z.imag) > 1e-12 else f"{z.real:.6g}", _f(mag),
                            ("> 1: 충격이 시간이 지나면 사라짐(정상)" if kind == "AR" else "> 1: 과거 오차의 영향이 갈수록 줄어듦(가역)") if good else
                            ("≤ 1: 충격이 사라지지 않음(비정상)" if kind == "AR" else "≤ 1: 과거 오차의 영향이 줄지 않음(가역 아님) — θ 를 해석할 수 없음")])
    out["steps"].append({"id": "roots", "title": "계수 점검 — 다항식의 근 크기 (정상성·가역성)",
                         "plain": "AR 계수로 만든 식 1 − φ₁z − φ₂z² − … = 0 과 MA 계수로 만든 식 1 + θ₁z + θ₂z² + … = 0 의 근(z)을 구해 크기를 봅니다. "
                                  "모든 근의 크기가 1 보다 크면, AR 쪽은 '한 번의 충격이 시간이 지나면 사라진다'(정상), MA 쪽은 '먼 과거의 예측 오차일수록 덜 남는다'(가역)는 뜻입니다. "
                                  "제약을 켜면(예) 추정이 이 범위 안에서만 이뤄지므로 근이 모두 1 보다 큽니다. 끄면(아니오) 이 범위를 벗어난 값이 나올 수 있습니다 — 그때도 예측값은 칼만 필터로 정확히 계산되지만, "
                                  "가역이 아닌 MA 는 같은 예측을 내는 다른 θ 가 있어 θ 숫자 자체를 해석할 수 없습니다."
                                  + (" 계절 모델은 계절 다항식을 곱해 펼친 식의 근입니다." if seasonal else ""),
                         "formula": "AR: φ(z)" + ("Φ(z^s)" if seasonal else "") + " = 0,  MA: θ(z)" + ("Θ(z^s)" if seasonal else "") + " = 0,  판정: 모든 |z| > 1",
                         "table": {"columns": ["다항식", "근 z", "크기 |z|", "판정"], "rows": rows_rt or [["-", "근 없음(차수 0)", "-", "-"]]},
                         "values": {"정상성 제약(설정)": bool(p.get("enforce_stationarity", False)), "가역성 제약(설정)": bool(p.get("enforce_invertibility", False)),
                                    "AR 근 모두 > 1": ok_all["AR"], "MA 근 모두 > 1": ok_all["MA"]}})
    if p.get("enforce_invertibility", False):
        out["checks"].append({"what": "가역성 제약을 켰으므로 MA 근이 모두 1 보다 큼", "ok": bool(ok_all["MA"]), "max_abs_diff": None})
    if p.get("enforce_stationarity", False):
        out["checks"].append({"what": "정상성 제약을 켰으므로 AR 근이 모두 1 보다 큼(차분 제외)", "ok": bool(ok_all["AR"]), "max_abs_diff": None})
    hist = r_all.loc[:t].dropna()
    r2 = res.apply(hist.values * sc, refit=False)
    fc = np.asarray(r2.forecast(k + skip), dtype=float) / sc
    fr = r2.filter_results
    err = np.asarray(fr.forecasts_error[0], dtype=float) / sc
    one = np.asarray(fr.forecasts[0], dtype=float) / sc
    out["steps"].append({"id": "filter", "title": f"결정일 {t.date()} 까지의 변화율 전부를 필터에 통과 (계수는 고정)",
                         "plain": f"정한 계수는 그대로 두고, 데이터 처음부터 결정일까지의 변화율 {len(hist)}개를 차례로 넣으며 '그때마다 한 기간 앞 예측'과 '예측 오차 ε'를 갱신합니다(칼만 필터). 학습 창보다 앞의 날도 들어가지만 계수는 다시 정하지 않습니다(refit=False). 결정일 이후 값은 쓰지 않습니다.",
                         "formula": "res.apply(r_[처음..t], refit=False) → 한 기간 앞 예측 r̂_(τ|τ−1), 오차 ε_τ = r_τ − r̂_(τ|τ−1)",
                         "table": {"columns": ["날짜", "변화율 r", "한 기간 앞 예측", "오차 ε"], "rows": [[_d(i), _f(v), _f(o), _f(e)] for i, v, o, e in zip(hist.index, hist.values, one, err)]}})
    # 정확 재현: 상태공간 a_(T+1) = T a_T + c, 예측 = Z a + d
    mm = r2.model.ssm
    Z = np.asarray(mm["design"]); Z = Z[:, :, -1] if Z.ndim == 3 else Z
    T = np.asarray(mm["transition"]); T = T[:, :, -1] if T.ndim == 3 else T
    c = np.asarray(mm["state_intercept"]); c = c[:, -1] if c.ndim == 2 else c
    dd = np.asarray(mm["obs_intercept"]); dd = dd[:, -1] if dd.ndim == 2 else dd
    a = np.asarray(fr.filtered_state[:, -1], dtype=float)
    ss = []
    for _ in range(k + skip):
        a = T @ a + c
        ss.append(float((Z @ a + dd)[0]))
    ss = np.array(ss) / sc
    # 이해용 재귀식: 차분까지 곱해 펼친 AR·MA 다항식
    A = np.asarray(r2.polynomial_ar, dtype=float)
    M = np.asarray(r2.polynomial_ma, dtype=float)
    if seasonal:
        A = np.convolve(A, np.asarray(r2.polynomial_seasonal_ar, dtype=float)); M = np.convolve(M, np.asarray(r2.polynomial_seasonal_ma, dtype=float))
    for _ in range(int(p["d"])):
        A = np.convolve(A, [1, -1])
    for _ in range(int(p.get("D", 0) or 0)):
        A = np.convolve(A, np.r_[1, np.zeros(int(p["s"]) - 1), -1])
    cc = float(dict(zip(names, vals)).get("intercept", 0.0))
    y, E, rec, rec_rows = list(hist.values), list(err), [], []
    for h in range(k + skip):
        ar_terms = [(i, -A[i], y[-i]) for i in range(1, len(A)) if A[i] != 0]
        ma_terms = [(j, M[j], E[-j]) for j in range(1, len(M)) if M[j] != 0]
        v = cc + sum(a_ * b_ for _, a_, b_ in ar_terms) + sum(a_ * b_ for _, a_, b_ in ma_terms)
        rec.append(v)
        rec_rows += [[f"t+{h + 1}", "상수 c", "", _f(cc), "", _f(cc)]] if cc else []
        rec_rows += [[f"t+{h + 1}", "AR", f"r(t+{h + 1 - i})" if h + 1 - i > 0 else f"r(t−{i - h - 1})" if i - h - 1 else "r(t)", _f(a_), _f(b_), _f(a_ * b_)] for i, a_, b_ in ar_terms]
        rec_rows += [[f"t+{h + 1}", "MA", f"ε(t+{h + 1 - j})" if h + 1 - j > 0 else f"ε(t−{j - h - 1})" if j - h - 1 else "ε(t)", _f(a_), _f(b_), _f(a_ * b_)] for j, a_, b_ in ma_terms]
        rec_rows.append([f"t+{h + 1}", "합 = 예측", "", "", "", _f(v)])
        y.append(v); E.append(0.0)
    rec = np.array(rec)
    out["steps"].append({"id": "recursion", "title": f"앞으로 {k + skip}기간을 한 기간씩 이어 예측 (재귀)",
                         "plain": "다음 한 기간을 예측하고, 그 예측을 다시 '지난 값'처럼 넣어 그다음을 예측합니다. 미래의 충격 ε 는 모르므로 0 으로 둡니다. "
                                  "아래 표는 식을 펼쳐 항 하나하나(계수 × 값)를 적은 것입니다(차분·계절 차분을 곱해 펼친 다항식).",
                         "formula": "r̂_(t+h) = c − Σ_i A_i · r_(t+h−i) + Σ_j M_j · ε_(t+h−j),  A(L) = φ(L)Φ(L^s)(1−L)^d(1−L^s)^D = 1 + A_1 L + …,  M(L) = θ(L)Θ(L^s) = 1 + M_1 L + …;  미래 ε = 0",
                         "table": {"columns": ["예측 시점", "항", "값의 이름", "계수", "값", "계수 × 값"], "rows": rec_rows},
                         "values": {f"statsmodels 예측 t+{h + 1}": _f(v) for h, v in enumerate(fc)}})
    out["checks"].append({"what": "상태공간 식(a ← T·a + c, 예측 = Z·a + d)으로 손 계산 = statsmodels forecast", "max_abs_diff": _f(np.max(np.abs(ss - fc))), "ok": bool(np.allclose(ss, fc, rtol=0, atol=1e-12))})
    rd = float(np.max(np.abs(rec - fc)))
    inv = None
    try:
        roots = np.roots(M[::-1]) if len(M) > 1 else np.array([])
        inv = bool(np.all(np.abs(roots) > 1)) if len(roots) else True
    except Exception:  # noqa: BLE001
        pass
    out["checks"].append({"what": "펼친 재귀식(이해용) = statsmodels forecast", "max_abs_diff": _f(rd), "ok": True, "info": True,
                          "note": ("일치" if rd < 1e-10 else "근사: MA 다항식이 가역이 아니어서(근의 크기 ≤ 1) 필터 오차 ε 가 실제 충격과 같아지지 않음 → 정확한 값은 위 상태공간 식" if inv is False else
                                   "근사: 필터 초기값의 영향이 남아 ε 가 실제 충격과 조금 다름 → 정확한 값은 위 상태공간 식")})
    keep = fc[skip:]
    agg = float(np.sum(np.log1p(keep))) if cfg.target.kind == "logreturn" else float(np.prod(1 + keep) - 1)
    model_out = float(cap["pred"].loc[t])
    out["steps"].append({"id": "compound", "title": f"{k}기간 수익률로 묶기",
                         "plain": ("매매 모드의 정답은 '다음 날 시가에 사서 k 기간 뒤 시가에 판' 수익률이라, 첫 기간(t → t+1) 예측은 버리고 그다음 k 개를 복리로 묶습니다. "
                                   "(이 모델은 종가 변화율을 예측하므로 시가 기준 정답과 하루 안의 차이가 남습니다 — 문서화된 근사.)" if skip else
                                   "다음 k 기간의 변화율 예측을 복리로 묶어 k 기간 뒤 값의 변화율로 바꿉니다.")
                                  + " 다른 모델과 같은 정답 단위로 비교하기 위해서입니다.",
                         "formula": ("ŷ = Σ ln(1 + r̂)" if cfg.target.kind == "logreturn" else "ŷ = Π (1 + r̂) − 1") + (f",  r̂ = 예측 t+2 … t+{k + 1}" if skip else f",  r̂ = 예측 t+1 … t+{k}"),
                         "values": {**{f"r̂ {('t+' + str(h + 1 + skip))}": _f(v) for h, v in enumerate(keep)}, "ŷ": agg}})
    out["checks"].append({"what": "손으로 묶은 ŷ = 엔진 예측", "manual": agg, "model": model_out, "max_abs_diff": _f(abs(agg - model_out)), "ok": bool(abs(agg - model_out) <= 1e-12)})


# ---------------------------------------------------------------------------- VECM
def _vecm(out, cfg, ds, cap, t):
    m = cap["model"]
    k = cfg.target.k
    skip = cap["skip"]
    Lall, Ltr, Ld, row, r0, need = cap["Lall"], cap["Ltr"], cap["Ld"], cap["row"], cap["r0"], cap["need"]
    cols = ["대상(" + (cfg.target_variable or cfg.ticker) + ")"] + list(Lall.columns[1:])
    keep = list(m.keep)
    kc = [cols[j] for j in keep]
    out["steps"].append({"id": "train_table", "title": "학습 표 — 수준값(변환하지 않은 원래 값)",
                         "plain": f"대상과 고른 변수의 원래 값(발표시차 적용)을 그대로 씁니다. 학습 창 {len(Ltr)}행. 학습 창에서 값이 한 번도 안 바뀐 열은 뺍니다"
                                  + (f" — 뺀 열: {[cols[j] for j in range(len(cols)) if j not in keep]}" if len(keep) < len(cols) else " (뺀 열 없음)") + "."
                                  + (" 로그를 취한 열: " + str([kc[j] for j in range(len(kc)) if m._log[j]]) if np.any(m._log) else " 로그는 쓰지 않음(log_levels=아니오)."),
                         "formula": "Y = [대상, 변수…] (학습 창)",
                         "table": {"columns": ["날짜", *cols], "rows": [[_d(i), *[_f(v) for v in r.values]] for i, r in Ltr.iterrows()]}})
    K = len(keep)
    p = int(m.p)
    r = int(m.rank)
    al, be, ga = np.asarray(m.alpha), np.asarray(m.beta), np.asarray(m.gamma)
    det = m.det
    diag = getattr(m, "diag", {}) or {}
    out["steps"].append({"id": "fit", "title": "장기 균형식·되돌아오는 속도·단기 동학 정하기 — 학습",
                         "plain": ("오래 보면 함께 움직이는 조합(공적분)을 찾고(Johansen), 그 조합에서 벗어난 만큼 다음 변화가 얼마나 반대로 당겨지는지(α), 지난 변화가 이번 변화에 주는 영향(Γ)을 추정합니다. "
                                   if r >= 1 else "학습 창에서 장기 균형 관계(공적분)가 없다고 나와(차수 0), 장기식 없이 변화끼리의 관계(차분 VAR)로 대신합니다. ")
                                  + f"공적분 차수 r = {r}{' (자동: Johansen ' + str(m.params.get('rank_test')) + ' 검정)' if int(m.params['coint_rank']) == 0 else ''}, 지난 변화 수 p = {p}, 상수 = {det}, 계절 = {m.seasons}.",
                         "formula": "ΔY_τ = α(β′Y_(τ−1) + c_in) + Σ_(i=1..p) Γ_i ΔY_(τ−i) + c_out + 계절 + ε_τ   (statsmodels VECM, Johansen 1988·1991)",
                         "table": {"columns": ["행렬", "행", *[f"열{j + 1}" for j in range(max(be.shape[1] if be.ndim == 2 else 1, al.shape[1] if al.ndim == 2 else 1, ga.shape[1] if ga.size else 0))]],
                                   "rows": [["β (균형식, 열 = 조합)", kc[i], *[_f(v) for v in be[i]]] for i in range(be.shape[0])]
                                           + [["α (되돌아오는 속도)", kc[i], *[_f(v) for v in al[i]]] for i in range(al.shape[0])]
                                           + [[f"Γ (열 = 시차·변수)", kc[i], *[_f(v) for v in ga[i]]] for i in range(ga.shape[0])]
                                           + ([["c_in (균형식 안 상수)", f"조합{j + 1}", _f(v)] for j, v in enumerate(np.ravel(m.c_in))] if m.c_in is not None else [])
                                           + ([["c_out·계절 (균형식 밖)", kc[i], *[_f(v) for v in m.det_out[i]]] for i in range(m.det_out.shape[0])] if m.det_out is not None else [])},
                         "values": {"학습 행": len(Ltr), "변수 수 K": K, "r": r, "p": p, **({"ADF p(수준값)": str(diag.get('adf_p'))} if diag.get('adf_p') is not None else {})}})
    hist = Ld.loc[:t].iloc[-max(need, 2):]
    pos = int(row.loc[t]) - r0
    path = np.asarray(m.forecast_levels(hist.values, k + skip, pos=pos), dtype=float)
    # 손 계산(독립 구현)
    H = np.asarray(hist.values, dtype=float)[:, keep]
    Zm = H.copy(); Zm[:, m._log] = np.log(Zm[:, m._log])
    h = [row_ for row_ in Zm[-(p + 1):]]
    from statsmodels.tsa.vector_ar.util import seasonal_dummies
    S = seasonal_dummies(m.seasons, pos + k + skip + 1, centered=True) if m.seasons else None
    rows, man = [], []
    for st in range(1, k + skip + 1):
        dy = [h[-i] - h[-i - 1] for i in range(1, p + 1)]
        ect = be.T @ h[-1] + (np.ravel(m.c_in) if m.c_in is not None else 0)
        ec_part = al @ ect
        sr_part = sum((ga[:, i * K:(i + 1) * K] @ dy[i] for i in range(p)), np.zeros(K))
        det_part = np.zeros(K)
        if m.det_out is not None:
            vec = ([1.0] if det == "co" else []) + (list(S[pos + st]) if (S is not None and r >= 1) else [])
            det_part = m.det_out[:, :len(vec)] @ np.asarray(vec)
        d = ec_part + sr_part + det_part
        nxt = h[-1] + d
        for j in range(K):
            rows.append([f"t+{st}", kc[j], _f(h[-1][j]), *([_f(v) for v in np.ravel(ect)] if j == 0 else ["" for _ in np.ravel(ect)]), _f(ec_part[j]), _f(sr_part[j]), _f(det_part[j]), _f(d[j]), _f(nxt[j])])
        h.append(nxt); man.append(nxt[0])
    man = np.exp(man) if m._log[0] else np.array(man)
    ne = len(np.ravel(be.T @ h[0])) if r >= 1 else 1
    out["steps"].append({"id": "recursion", "title": f"결정일 {t.date()} 에서 {k + skip}기간 앞까지 한 기간씩 계산",
                         "plain": "① 지금 값이 장기 균형에서 얼마나 벗어났는지(e = β′Y) → ② 그만큼 되돌아오는 변화(α·e) + ③ 지난 변화의 영향(Γ·ΔY) + ④ 상수·계절 → 다 더한 것이 다음 기간의 변화. 그 값을 더해 다음 수준을 만들고 다시 반복합니다. 대상(첫 열)만 결과로 씁니다.",
                         "formula": "e = β′Y_(τ−1) (+c_in);  ΔŶ = α·e + Σ Γ_i ΔY_(τ−i) + 상수·계절;  Ŷ_τ = Y_(τ−1) + ΔŶ",
                         "table": {"columns": ["예측 시점", "변수", "직전 값", *[f"균형 이탈 e{j + 1}" for j in range(ne)], "되돌림 α·e", "단기 Γ·ΔY", "상수·계절", "변화 ΔŶ", "다음 값"], "rows": rows},
                         "values": {"쓴 이력 행": len(hist), "계절 위상 pos": pos}})
    out["checks"].append({"what": "손으로 따로 계산한 대상 경로 = 모델 forecast_levels", "max_abs_diff": _f(np.max(np.abs(man - path))), "ok": bool(np.allclose(man, path, rtol=1e-12, atol=1e-10))})
    base = path[skip - 1] if skip else float(hist.iloc[-1, 0])
    ratio = path[-1] / base
    agg = float(np.log(ratio)) if cfg.target.kind == "logreturn" else float(ratio - 1)
    model_out = float(cap["pred"].loc[t])
    out["steps"].append({"id": "compound", "title": "변화율로 바꾸기",
                         "plain": ("매매 모드: 다음 날(t+1) 예측 값을 기준으로, 그로부터 k 기간 뒤 예측 값이 몇 % 다른지." if skip else "결정일 값 대비 k 기간 뒤 예측 값이 몇 % 다른지.") + " 다른 모델과 같은 정답 단위로 비교하기 위해서입니다.",
                         "formula": ("ŷ = ln(Ŷ_end ÷ 기준)" if cfg.target.kind == "logreturn" else "ŷ = Ŷ_end ÷ 기준 − 1"),
                         "values": {"기준": _f(base), "Ŷ_end": _f(path[-1]), "ŷ": agg}})
    out["checks"].append({"what": "손으로 바꾼 ŷ = 엔진 예측", "manual": agg, "model": model_out, "max_abs_diff": _f(abs(agg - model_out)), "ok": bool(abs(agg - model_out) <= 1e-12)})


# ---------------------------------------------------------------------------- LSTM / GRU
def _rnn(out, cfg, ds, reg, cap, t, target_type):
    m = cap["model"]
    net = m.net
    gru = cfg.model == "gru"
    feats = list(ds.feature_names)
    K = len(feats)
    L = int(cfg.seq_len)
    desc = {c: describe_feature(c, reg) for c in feats}
    sc = cap["scaler"]
    te_idx = list(cap["te_idx"])
    i = te_idx.index(t)
    Xw = cap["Xtw"][i]  # [L, K] 모델이 실제로 받은 창
    te_full = cap["te_full"]
    win_dates = te_full.index[i: i + L]
    raw = te_full[feats].ffill().iloc[i: i + L]
    out["steps"].append({"id": "window", "title": f"입력 창 — 결정일 {t.date()} 까지 {L}개 시점 × 재료 {K}개",
                         "plain": f"결정일을 포함한 최근 {L}개 시점의 재료를 순서대로 쌓은 표입니다(첫 테스트일이면 학습 창 끝의 실제 과거 행을 앞에 붙임). 빈 칸은 앞의 값으로 채우고(ffill), 그래도 남으면 학습 구간 평균(표준화 후 0)으로 채웁니다. 그다음 학습 구간 평균·표준편차로 표준화합니다.",
                         "formula": "창 = X[t−L+1 … t].ffill() → (x − 평균) ÷ 표준편차 → 빈 칸 = 학습 평균",
                         "table": {"columns": ["시점", "날짜", *[f"{c} (원래)" for c in feats], *[f"{c} (z)" for c in feats]],
                                   "rows": [[f"τ={j + 1}", _d(d), *[_f(v) for v in raw.iloc[j].values], *[_f(v) for v in Xw[j]]] for j, d in enumerate(win_dates)]}})
    out["steps"].append({"id": "scale_params", "title": "표준화에 쓴 평균·표준편차 (학습 구간)",
                         "plain": "위 창의 z 를 만든 값입니다. 학습 표로만 계산했습니다.",
                         "formula": "평균_j, 표준편차_j (ddof=0)",
                         "table": {"columns": ["열", "뜻", "평균", "표준편차"], "rows": [[c, desc[c], _f(sc.mean_[j]), _f(sc.scale_[j])] for j, c in enumerate(feats)]}})
    # 학습
    hist = getattr(m, "history", []) or []
    p = m.params
    nw, nva = cap["n_tr"], cap["n_va"]
    out["steps"].append({"id": "fit", "title": "가중치 정하기 — 학습",
                         "plain": f"학습 창으로 만든 창 {nw}개(검증 {nva}개는 따로)로 가중치를 맞춥니다. 정답은 학습 구간 평균·표준편차로 표준화한 뒤 맞힙니다(회귀). 매 epoch 마다 창 순서를 섞어 {p['batch_size']}개씩 묶어 Adam(학습률 {p['lr']}, 가중치 감쇠 {p['weight_decay']})으로 고치고, 기울기 크기는 1.0 으로 자릅니다. "
                                  + (f"검증 손실이 {p['patience']} epoch 연속 나아지지 않으면 멈추고, 가장 낮았던 epoch({m.best_epoch})의 가중치를 씁니다." if nva else "검증이 없어(검증 10개 미만) 정한 epoch 를 모두 돌고 마지막 가중치를 씁니다."),
                         "formula": ("손실 = 평균((ŷ_z − y_z)²),  y_z = (y − μ_y) ÷ σ_y" if target_type == "regression" else "손실 = BCEWithLogits(점수, y)") + " · Adam · clip_grad_norm 1.0 · Dropout(학습 때만)",
                         "values": {"학습 창 수": nw, "검증 창 수": nva, "돈 epoch": len(hist), "고른 epoch": m.best_epoch or len(hist), "μ_y": _f(m.y_mu), "σ_y": _f(m.y_sd),
                                    "전체 가중치 수": int(sum(q.numel() for q in net.parameters()))},
                         "table": {"columns": ["epoch", "학습 손실(그 epoch 배치 평균, Dropout 켠 상태)", "검증 손실"], "rows": [[h["epoch"], _f(h["train_loss"]), _f(h["val_loss"])] for h in hist]}})
    # 셀 계산 (numpy, 층마다)
    rnn = net.gru if gru else net.lstm
    H = rnn.hidden_size
    nl = rnn.num_layers
    xin = Xw.astype(np.float64)
    cell_rows = []
    for layer in range(nl):
        Wih = getattr(rnn, f"weight_ih_l{layer}").detach().cpu().numpy().astype(np.float64)
        Whh = getattr(rnn, f"weight_hh_l{layer}").detach().cpu().numpy().astype(np.float64)
        bih = getattr(rnn, f"bias_ih_l{layer}").detach().cpu().numpy().astype(np.float64)
        bhh = getattr(rnn, f"bias_hh_l{layer}").detach().cpu().numpy().astype(np.float64)
        h = np.zeros(H); c = np.zeros(H); outs = []
        for tau in range(L):
            x = xin[tau]
            if gru:
                gi = Wih @ x + bih; gh = Whh @ h + bhh
                r_ = _sig(gi[:H] + gh[:H]); z_ = _sig(gi[H:2 * H] + gh[H:2 * H]); n_ = np.tanh(gi[2 * H:] + r_ * gh[2 * H:])
                h = (1 - z_) * n_ + z_ * h
                for u in range(H):
                    cell_rows.append([layer + 1, tau + 1, u + 1, _f(r_[u]), _f(z_[u]), _f(n_[u]), _f(h[u])])
            else:
                g = Wih @ x + bih + Whh @ h + bhh
                i_ = _sig(g[:H]); f_ = _sig(g[H:2 * H]); g_ = np.tanh(g[2 * H:3 * H]); o_ = _sig(g[3 * H:])
                c = f_ * c + i_ * g_
                h = o_ * np.tanh(c)
                for u in range(H):
                    cell_rows.append([layer + 1, tau + 1, u + 1, _f(i_[u]), _f(f_[u]), _f(g_[u]), _f(o_[u]), _f(c[u]), _f(h[u])])
            outs.append(h.copy())
        xin = np.array(outs)  # 다음 층 입력(예측 때 층 사이 Dropout 은 꺼짐)
    hT = xin[-1]
    cols = ["층", "시점 τ", "은닉 단위", "재설정 r", "갱신 z", "후보 n", "새 은닉 h"] if gru else ["층", "시점 τ", "은닉 단위", "입력 게이트 i", "잊기 게이트 f", "후보 g", "출력 게이트 o", "기억 c", "은닉 h"]
    out["steps"].append({"id": "cells", "title": f"셀 계산 — {nl}층 × {L}시점 × 은닉 {H}개, 값 하나도 빼지 않고",
                         "plain": ("GRU 셀은 시점마다 '지난 기억 중 얼마를 볼지(r)'와 '지난 기억과 새 후보를 어떤 비율로 섞을지(z)'를 0~1 사이 값으로 정해 새 기억 h 를 만듭니다." if gru else
                                   "LSTM 셀은 시점마다 0~1 사이의 문(게이트) 세 개로 '지난 기억 c 중 얼마를 남길지(f)', '새 정보 g 중 얼마를 쓸지(i)', '기억 중 얼마를 내보낼지(o)'를 정해 기억 c 와 은닉 h 를 갱신합니다.")
                                  + " 처음 은닉·기억은 0 입니다. 은닉 단위 하나하나가 무엇을 뜻하는지는 학습이 정하므로 사람이 이름 붙일 수 없습니다. 아래 표는 학습된 가중치로 직접 계산한 값입니다.",
                         "formula": ("r = σ(W_ir x + b_ir + W_hr h + b_hr),  z = σ(W_iz x + b_iz + W_hz h + b_hz),  n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn)),  h′ = (1 − z) ⊙ n + z ⊙ h   (PyTorch GRU)" if gru else
                                     "i = σ(W_ii x + b_ii + W_hi h + b_hi),  f = σ(W_if x + … ),  g = tanh(W_ig x + … ),  o = σ(W_io x + … ),  c′ = f ⊙ c + i ⊙ g,  h′ = o ⊙ tanh(c′)   (PyTorch LSTM)"),
                         "table": {"columns": cols, "rows": cell_rows}})
    w = net.fc.weight.detach().cpu().numpy().astype(np.float64).ravel()
    b = float(net.fc.bias.detach().cpu().numpy()[0])
    contrib = w * hT
    score = b + contrib.sum()
    pred_hand = score * m.y_sd + m.y_mu if target_type == "regression" else _sig(score)
    model_out = float(cap["pred"].loc[t])
    out["steps"].append({"id": "head", "title": "마지막 시점의 은닉으로 예측 (Linear)",
                         "plain": f"마지막 시점의 은닉 {H}개만 남기고(앞 시점 출력은 버림), 각각에 가중치를 곱해 더합니다. 예측 때 Dropout 은 꺼져 그대로 통과합니다. "
                                  + ("그 값은 표준화된 정답 단위라 학습 구간의 평균·표준편차로 되돌립니다." if target_type == "regression" else "그 점수를 확률로 바꿉니다."),
                         "formula": "점수 = b + Σ_u w_u · h_u(마지막 시점, 마지막 층)" + ("  → ŷ = 점수 × σ_y + μ_y" if target_type == "regression" else "  → 확률 = 1 ÷ (1 + e^(−점수))"),
                         "table": {"columns": ["은닉 단위", "h", "가중치 w", "기여 w·h"], "rows": [[u + 1, _f(hT[u]), _f(w[u]), _f(contrib[u])] for u in np.argsort(-np.abs(contrib))]},
                         "values": {"편향 b": b, "점수": _f(score), "ŷ": _f(pred_hand)}})
    out["checks"].append({"what": "손으로 계산한 셀·Linear(64비트) = PyTorch 출력(32비트)", "manual": _f(pred_hand), "model": model_out, "max_abs_diff": _f(abs(pred_hand - model_out)),
                          "ok": bool(abs(pred_hand - model_out) <= 1e-5 * max(abs(m.y_sd), 1e-12) + 1e-9), "note": "PyTorch 는 32비트 실수로 계산해 소수점 7자리 근처에서 차이가 날 수 있음"})


# ---------------------------------------------------------------------------- 규칙 모델
def _rule(out, cfg, plugin, params, ds, reg, date):
    from engine.experiment import _eval_window
    from engine.features import rsi as _rsi
    k = cfg.target.k
    model = plugin.Model(params)
    pos_full = pd.Series(model.positions(ds.prices, ds.X) if getattr(model, "USES_VARIABLES", False) else model.positions(ds.prices), index=ds.prices.index)
    lo, hi = _eval_window(cfg, ds, k)
    pos = pos_full.where((pos_full.index >= lo) & (pos_full.index <= hi)) if lo is not None else pos_full
    ok_idx = pos.dropna().index
    if ok_idx.empty:
        raise ValueError("평가 구간에 신호가 없습니다")
    want = pd.Timestamp(date) if date else None
    t = want if (want is not None and want in ok_idx) else ok_idx[-1]
    if want is not None and want not in ok_idx:
        out["notes"].append(f"{want.date()} 은(는) 평가 구간의 신호일이 아니라 마지막 신호일 {t.date()} 로 보여 줍니다")
    out["decision_date"] = _d(t)
    out["available_dates"] = [_d(x) for x in ok_idx[-250:]]
    out["actual"] = _f(ds.y.get(t, np.nan))
    out["steps"].append({"id": "fold", "title": "기간 — 학습 없음, 평가 구간만",
                         "plain": "규칙 모델은 배우는 과정이 없습니다. 그날까지의 가격(또는 변수)에 정해진 규칙을 적용할 뿐입니다. 다만 학습 모델과 같은 날짜로 성적을 매기도록, 같은 검증 설정이면 학습 모델이 평가받는 구간에서만 신호를 씁니다.",
                         "formula": f"평가 구간 = [{_d(lo) if lo is not None else '처음'} … {_d(hi) if hi is not None else '끝'}]",
                         "values": {"평가 구간 신호 수": int(len(ok_idx))}})
    c = ds.prices["close"].astype(float)
    i = ds.prices.index.get_loc(t)
    manual = None
    if cfg.model == "buy_and_hold":
        manual = 1.0
        out["steps"].append({"id": "rule", "title": "규칙 — 항상 보유", "plain": "평가 구간 첫날 사서 끝까지 들고 있습니다. 모든 전략의 비교 기준선입니다.", "formula": "포지션 = 1", "values": {"포지션": 1.0}})
    elif cfg.model == "ma_cross":
        fa, sl = int(params["fast"]), int(params["slow"])
        seg = c.iloc[max(0, i - sl + 1): i + 1]
        mf, ms = float(c.iloc[i - fa + 1: i + 1].mean()), float(seg.mean())
        manual = float(mf > ms)
        out["steps"].append({"id": "rule", "title": f"규칙 — 짧은 평균({fa}) > 긴 평균({sl}) 이면 보유",
                             "plain": f"결정일까지의 종가 {sl}개를 모두 적었습니다. 최근 {fa}개의 평균(빠른 흐름)과 {sl}개 전부의 평균(느린 흐름)을 비교합니다. 짧은 평균이 위에 있으면 오르는 흐름으로 보고 보유합니다.",
                             "formula": f"MA_{fa} = 최근 {fa}개 종가의 평균,  MA_{sl} = 최근 {sl}개 종가의 평균,  포지션 = 1 if MA_{fa} > MA_{sl} else 0",
                             "table": {"columns": ["날짜", "종가", f"짧은 평균에 들어감({fa})"], "rows": [[_d(d), _f(v), "예" if j >= len(seg) - fa else ""] for j, (d, v) in enumerate(seg.items())]},
                             "values": {f"MA_{fa}": mf, f"MA_{sl}": ms, "포지션": manual}})
    elif cfg.model == "momentum":
        L = int(params["lookback"])
        c0, c1 = float(c.iloc[i - L]), float(c.iloc[i])
        rr = c1 / c0 - 1
        manual = float(rr > 0)
        out["steps"].append({"id": "rule", "title": f"규칙 — {L}기간 전보다 올랐으면 보유",
                             "plain": f"결정일 종가를 {L}기간 전 종가와 비교합니다. 올랐으면(+) 그 흐름이 이어진다고 보고 보유, 아니면 현금.",
                             "formula": f"r = 종가_t ÷ 종가_(t−{L}) − 1,  포지션 = 1 if r > 0 else 0",
                             "values": {f"종가 t−{L} ({_d(c.index[i - L])})": c0, f"종가 t ({_d(t)})": c1, "r": rr, "포지션": manual}})
    elif cfg.model == "rsi_rule":
        w, lo_, up_ = int(params["window"]), float(params["lower"]), float(params["upper"])
        d = c.diff()
        upm = d.clip(lower=0).ewm(alpha=1 / w, adjust=False).mean()
        dnm = (-d.clip(upper=0)).ewm(alpha=1 / w, adjust=False).mean()
        rs_ = upm / dnm.replace(0, np.nan)
        rv = 100 - 100 / (1 + rs_)
        code = _rsi(c, w)
        sub = rv.loc[:t]
        state = None; rows = []
        for dd_, v in sub.items():
            if np.isnan(v):
                act = "(계산 전)"
            elif v < lo_:
                state, act = 1.0, f"< {lo_:g} → 보유(매수)"
            elif v > up_:
                state, act = 0.0, f"> {up_:g} → 현금(매도)"
            else:
                state, act = (0.0 if state is None else state), "사이 → 직전 유지"
            rows.append([_d(dd_), _f(c.loc[dd_]), _f(d.loc[dd_]), _f(upm.loc[dd_]), _f(dnm.loc[dd_]), _f(v), act, _f(state)])
        manual = state
        out["steps"].append({"id": "rule", "title": f"규칙 — RSI({w}) < {lo_:g} 이면 사고, > {up_:g} 이면 팔고, 사이면 그대로",
                             "plain": f"RSI 는 최근 오른 폭과 내린 폭의 평균(오래된 날일수록 가볍게 보는 Wilder 평균)으로 계산한 0~100 점수입니다. 낮으면 많이 떨어진 상태(과매도), 높으면 많이 오른 상태(과매수). "
                                      f"이 규칙은 어제까지의 상태를 기억하므로, 데이터 첫날부터 결정일까지 {len(rows)}개 날짜를 모두 적었습니다.",
                             "formula": f"상승 평균_t = (1 − 1/{w})·상승 평균_(t−1) + (1/{w})·max(Δ,0),  하락 평균도 같은 방식,  RSI = 100 − 100 ÷ (1 + 상승 평균 ÷ 하락 평균)",
                             "table": {"columns": ["날짜", "종가", "변화 Δ", "상승 평균", "하락 평균", "RSI", "판정", "포지션"], "rows": rows},
                             "values": {"결정일 RSI": _f(rv.loc[t]), "포지션": manual}})
        out["checks"].append({"what": "손으로 계산한 RSI = 엔진 RSI(engine/features.rsi)", "max_abs_diff": _f(np.nanmax(np.abs((rv - code).values))), "ok": bool(np.allclose(rv.values, code.values, equal_nan=True))})
    elif cfg.model == "sox_gap_fade":
        col, w, q, mode = params["column"], int(params["window"]), float(params["q"]), params["mode"]
        s = ds.X[col].astype(float).reindex(ds.prices.index)
        past = s.shift(1).iloc[max(0, i - w + 1): i + 1].dropna()
        qq = q if mode == "avoid_top" else 1 - q
        thr = float(np.quantile(past.values, qq, method="linear"))
        st = float(s.loc[t])
        manual = float(not (st > thr)) if mode == "avoid_top" else float(st < thr)
        out["steps"].append({"id": "rule", "title": f"규칙 — {col} 이 과거 {w}기간의 {qq:.0%} 분위보다 {'크면 다음 날 현금' if mode == 'avoid_top' else '작을 때만 보유'}",
                             "plain": f"결정일 당일은 빼고 그 전 {w}기간(처음에는 절반 이상만 차도 계산)의 값 {len(past)}개를 모두 적었습니다. 이 값들을 작은 순서로 늘어놓았을 때 {qq:.0%} 위치의 값(선형 보간)이 기준입니다.",
                             "formula": f"기준 = quantile(값_(t−{w}) … 값_(t−1), {qq})  (pandas rolling quantile, 선형 보간);  " + ("포지션 = 0 if 값_t > 기준 else 1" if mode == "avoid_top" else "포지션 = 1 if 값_t < 기준 else 0"),
                             "table": {"columns": ["날짜", col], "rows": [[_d(d_), _f(v)] for d_, v in past.items()]},
                             "values": {"기준": thr, f"결정일 {col}": st, "포지션": manual}})
    model_pos = float(pos.loc[t])
    if manual is not None:
        out["checks"].append({"what": "손으로 판정한 포지션 = 규칙 모델 출력", "manual": _f(manual), "model": model_pos, "max_abs_diff": _f(abs(manual - model_pos)), "ok": bool(manual == model_pos)})
    out["steps"].append({"id": "decide", "title": "결정 (포지션)",
                         "plain": "규칙 모델은 예측값 대신 포지션(1 = 보유, 0 = 현금)을 바로 냅니다. 진입 임계치는 쓰지 않습니다. 다음 날 시가에 그 포지션으로 맞추고, 바뀔 때마다 수수료·슬리피지를 냅니다.",
                         "formula": "신호 = 포지션 (engine/experiment.py 규칙 모델 분기)",
                         "values": {"포지션": model_pos, "행동": "다음 날 시가에 보유(매수 또는 유지)" if model_pos else "다음 날 시가에 현금(매도 또는 유지)"}})
    cap = {"kind": "rule", "pred": pos.dropna()}
    if cfg.mode == "trading":
        o = ds.prices["open"]
        if i + 1 + k < len(ds.prices):
            p1, p2 = float(o.iloc[i + 1]), float(o.iloc[i + 1 + k])
            out["steps"].insert(1, {"id": "target", "title": "정답(타겟) — 이 날 결정했다면 실제로 번 수익률",
                                    "plain": "규칙은 정답을 보지 않지만, 성적을 매길 때는 같은 정의를 씁니다: 다음 날 시가에 사서 k 기간 뒤 시가에 판 수익률.",
                                    "formula": out["target"]["text"],
                                    "values": {"진입일": _d(ds.prices.index[i + 1]), "진입 시가": p1, "청산일": _d(ds.prices.index[i + 1 + k]), "청산 시가": p2, "수익률 r": p2 / p1 - 1}})
    rows = [[_d(d_), _f(v), _f(ds.y.get(d_, np.nan))] for d_, v in cap["pred"].items()]
    out.pop("_y", None)
    out["fold_predictions"] = {"columns": ["결정일", "포지션", "실제(타겟)"], "rows": rows}



# ---------------------------------------------------------------------------- BIVA
def _lstm_np(x, Wih, Whh, bih, bhh, gates=False):
    """PyTorch LSTM 한 층(배치 1)을 64비트로 손 계산. 게이트 순서 i, f, g, o."""
    H = Whh.shape[1]
    h = np.zeros(H); c = np.zeros(H); outs, rec = [], []
    for tau in range(len(x)):
        g = Wih @ x[tau] + bih + Whh @ h + bhh
        i_, f_, g_, o_ = _sig(g[:H]), _sig(g[H:2 * H]), np.tanh(g[2 * H:3 * H]), _sig(g[3 * H:])
        c = f_ * c + i_ * g_
        h = o_ * np.tanh(c)
        outs.append(h.copy())
        if gates:
            rec.append((i_, f_, g_, o_, c.copy(), h.copy()))
    return np.array(outs), rec


def _np(t):
    return t.detach().cpu().numpy().astype(np.float64)


def _w(mod, name):
    return _np(getattr(mod, name))


def _rits_np(rits, x, imputed_weight):
    """원본 BIVA_RITS.Model.forward 를 64비트로 손 계산(배치 1). 반환 (imputations [L,K], loss, 시점별 기록)."""
    K = x.shape[1]
    L = x.shape[0]
    m = (~np.isnan(x)).astype(np.float64)
    x0 = np.nan_to_num(x)
    # δ: 원본 delta_calc 규칙 그대로(이웃 변수 열 d−1 을 참조 — 원본 코드의 특성, d=0 이면 파이썬 음수 색인으로 마지막 열)
    delta = np.zeros((L, K))
    for t in range(L):
        for d in range(K):
            if t == 0:
                delta[t, d] = 0
            elif m[t - 1, d] == 0:
                delta[t, d] = delta[t, d - 1] + 1
            elif m[t, d - 1] == 1:
                delta[t, d] = 1
    Wdh, bdh = _w(rits.temp_decay_h, "W"), _w(rits.temp_decay_h, "b")
    Whr, bhr = _w(rits.hist_reg, "weight"), _w(rits.hist_reg, "bias")
    Wf, bf, mf = _w(rits.feat_reg, "W"), _w(rits.feat_reg, "b"), _np(rits.feat_reg.m)
    cell = rits.rnn_cell
    Wih, Whh, bih, bhh = _w(cell, "weight_ih"), _w(cell, "weight_hh"), _w(cell, "bias_ih"), _w(cell, "bias_hh")
    Hn = Whh.shape[1]
    h = np.zeros(Hn); c = np.zeros(Hn); loss = 0.0; imp = []; rec = []
    for t in range(L):
        gh = np.exp(-np.maximum(Wdh @ delta[t] + bdh, 0))
        h = h * gh
        xh = _sig(Whr @ h + bhr)
        loss += np.sum(np.abs(x0[t] - xh) * m[t]) / (np.sum(m[t]) + 1e-5)
        xc = m[t] * x0[t] + (1 - m[t]) * xh
        zh = _sig((Wf * mf) @ xc + bf)
        loss += np.sum(np.abs(x0[t] - zh) * m[t]) / (np.sum(m[t]) + 1e-5)
        ch = np.abs(0.5 * zh + 0.5 * xh)
        cc = m[t] * x0[t] + (1 - m[t]) * ch
        g = Wih @ np.r_[cc, m[t]] + bih + Whh @ h + bhh
        i_, f_, g_, o_ = _sig(g[:Hn]), _sig(g[Hn:2 * Hn]), np.tanh(g[2 * Hn:3 * Hn]), _sig(g[3 * Hn:])
        c = f_ * c + i_ * g_
        h = o_ * np.tanh(c)
        imp.append(cc)
        rec.append({"m": m[t], "x": x0[t], "delta": delta[t], "xh": xh, "zh": zh, "ch": ch, "cc": cc})
    return np.array(imp), loss * imputed_weight, rec


def _biva(out, cfg, ds, reg, cap, t):
    import torch

    m = cap["model"]
    net = m.net
    args = m.args
    p = m.params
    feats = list(ds.feature_names)
    K, L = len(feats), int(cfg.seq_len)
    desc = {c: describe_feature(c, reg) for c in feats}
    sc = cap["scaler"]
    te_idx = list(cap["te_idx"])
    i = te_idx.index(t)
    Xw = np.asarray(cap["Xtw"][i], dtype=np.float64)  # [L, K] 모델이 실제로 받은 창(빈 칸은 NaN 그대로 = BRITS 가 채움)
    te_full = cap["te_full"]
    win_dates = te_full.index[i: i + L]
    raw = te_full[feats].ffill().iloc[i: i + L]
    minmax = p.get("input_scaler") == "minmax"
    out["steps"].append({"id": "window", "title": f"입력 창 — 결정일 {t.date()} 까지 {L}개 시점 × 재료 {K}개",
                         "plain": f"결정일을 포함한 최근 {L}개 시점의 재료를 쌓은 표입니다(첫 테스트일이면 학습 창 끝의 실제 과거 행을 붙임). 중간에 빈 날은 앞의 값으로 채우고(ffill), "
                                  + ("그래도 남는 빈 칸(아직 시작 전인 변수)은 NaN 그대로 두어 Stage 1 의 BRITS 가 채웁니다." if cap["accepts_nan"] else "그래도 남는 빈 칸은 학습 구간 평균으로 채웁니다(use_brits=아니오).")
                                  + (" 눈금은 학습 구간의 최소=0·최대=1 로 맞춥니다(MinMax, 원본 방식)." if minmax else " 눈금은 학습 구간 평균·표준편차로 맞춥니다."),
                         "formula": ("x′ = (x − 최소) ÷ (최대 − 최소)  (최소·최대 = 학습 구간, sklearn MinMaxScaler)" if minmax else "z = (x − 평균) ÷ 표준편차"),
                         "table": {"columns": ["시점", "날짜", *[f"{c} (원래)" for c in feats], *[f"{c} (눈금 맞춤)" for c in feats]],
                                   "rows": [[f"τ={j + 1}", _d(d), *[_f(v) for v in raw.iloc[j].values], *[(_f(v) if np.isfinite(v) else "빈 칸") for v in Xw[j]]] for j, d in enumerate(win_dates)]}})
    if minmax:
        out["steps"].append({"id": "scale_params", "title": "눈금 맞추기에 쓴 학습 구간 최소·최대", "plain": "위 창의 눈금 맞춘 값을 만든 값입니다. 학습 표로만 계산했습니다.",
                             "formula": "x′ = (x − 최소) ÷ (최대 − 최소)",
                             "table": {"columns": ["열", "뜻", "최소", "최대"], "rows": [[c, desc[c], _f(sc.data_min_[j]), _f(sc.data_max_[j])] for j, c in enumerate(feats)]}})
    hist = getattr(m, "history", []) or []
    out["steps"].append({"id": "fit", "title": "가중치 정하기 — 학습 (원본 exp_main.train 순서)",
                         "plain": f"학습 창 {cap['n_tr']}개(검증 {cap['n_va']}개)를 {p['batch_size']}개씩 섞어 묶고(나머지 버림) Adam(학습률 {p['learning_rate']})으로 고칩니다. 손실 = VAE 손실 + BRITS 보간 손실 + 예측 MSE(가중치 없는 합). "
                                  f"epoch 마다 검증 손실(배치 평균)로 조기종료(patience {p['patience']})를 판정하고 학습률을 조정합니다(lradj {p['lradj']}). 끝나면 검증 손실이 가장 낮았던 가중치로 되돌립니다. "
                                  + ("" if cap["n_va"] else "검증이 없어 학습 손실로 가장 좋았던 시점을 고릅니다(원본에는 없는 경우). ")
                                  + f"정답은 학습 구간 평균 {m.y_mu:.6g}·표준편차 {m.y_sd:.6g} 로 표준화해 맞힙니다(target_scale={p['target_scale']}).",
                         "formula": "손실 = VAE + BRITS + MSE(ŷ_z, y_z),  y_z = (y − μ_y) ÷ σ_y",
                         "values": {"학습 창": cap["n_tr"], "검증 창": cap["n_va"], "돈 epoch": len(hist), "설정 epoch": p["train_epochs"],
                                    "최저 검증 epoch": (min(hist, key=lambda h: h["vali_loss"])["epoch"] if hist else None), "전체 가중치 수": int(sum(q.numel() for q in net.parameters()))},
                         "table": {"columns": ["epoch", "학습 손실", "검증 손실", "학습률", "초"], "rows": [[h["epoch"], _f(h["train_loss"]), _f(h["vali_loss"]), _f(h["lr"]), _f(h["sec"])] for h in hist]}})
    # ---- 한 창 순전파(실제 모듈) + 중간값 붙잡기
    net.eval()
    grabbed: dict = {}
    hooks = [getattr(net, n).register_forward_hook(lambda mod, inp, o, n=n: grabbed.__setitem__(n, (inp, o)))
             for n in ["BRITS", "decomposition", "Conv1d_Trend_1", "Conv1d_Trend_2", "Conv1d_Trend_3", "Linear_Trend", "LSTM_VAE",
                       "Conv1d_Seasonal_1", "Conv1d_Seasonal_2", "Conv1d_Seasonal_3", "Linear_Seasonal"]]
    seed = 12345
    torch.manual_seed(seed)
    with torch.no_grad():
        states, vae_loss, recon, seas_init, imp_loss, imputed = net(torch.tensor(Xw[None], dtype=torch.float32))
    for hk in hooks:
        hk.remove()
    pass_pred = float(states[0, -1, 0]) * m.y_sd + m.y_mu
    exp_pred = float(cap["pred"].loc[t])
    # ---- Stage 1 BRITS (손 계산)
    x32 = Xw.astype(np.float32).astype(np.float64)
    if p["use_brits"]:
        brits = net.BRITS
        imp_f, lf, rec_f = _rits_np(brits.rits_f, x32, args.imputed_weight)
        imp_b_rev, lb, rec_b = _rits_np(brits.rits_b, x32[::-1].copy(), args.imputed_weight)
        imp_b = imp_b_rev[::-1]
        imp = (imp_f + imp_b) / 2
        lc = np.abs(imp_f - imp_b).mean() * 0.1
        brits_loss = lf + lb + lc
        rows = []
        for direction, rec in (("앞→뒤", rec_f), ("뒤→앞", rec_b)):
            for tau, r_ in enumerate(rec):
                tt = tau + 1 if direction == "앞→뒤" else L - tau
                for j in range(K):
                    rows.append([direction, tt, feats[j], "관측" if r_["m"][j] else "빈 칸", _f(r_["x"][j]) if r_["m"][j] else "", _f(r_["delta"][j]), _f(r_["xh"][j]), _f(r_["zh"][j]), _f(r_["ch"][j]), _f(r_["cc"][j])])
        n_nan = int(np.isnan(Xw).sum())
        mod_imp = _np(imputed[0])
        out["steps"].append({"id": "brits", "title": f"Stage 1 — 빈 칸 채우기 (BRITS, 빈 칸 {n_nan}개)",
                             "plain": "창을 앞에서 뒤로 한 번(앞→뒤 RITS), 뒤에서 앞으로 한 번(뒤→앞 RITS) 읽습니다. 시점마다 ① 지금까지의 기억으로 값을 추정(x̂)하고 ② 같은 시점의 다른 변수들로 추정(ẑ, 자기 자신은 0 으로 막음)한 뒤 "
                                      "③ 둘을 반씩 섞습니다(ĉ = |0.5·ẑ + 0.5·x̂|, 고정 비율). ④ 관측된 칸은 원래 값을 그대로 두고 빈 칸만 ĉ 로 채워 다음 시점 기억에 넣습니다. ⑤ 오래 빈 변수일수록 기억을 줄입니다(γ = exp(−max(0, W·δ + b))). "
                                      "두 방향의 결과를 평균한 것이 채운 창입니다. 관측 칸에서 추정이 틀린 정도가 보간 손실입니다. 표는 두 방향 × 시점 × 변수 전부입니다(δ 는 원본 코드 규칙 그대로 — 이웃 변수 열을 참조).",
                             "formula": "x̂ = σ(W_h·(γ_h⊙h) + b),  ẑ = σ((W_f⊙(1−I))·x_c + b_f),  ĉ = |0.5ẑ + 0.5x̂|,  채운 값 = m⊙x + (1−m)⊙ĉ,  결과 = (앞→뒤 + 뒤→앞) ÷ 2,  "
                                        "보간 손실 = Σ_방향 Σ_τ [Σ|x−x̂|m/(Σm+1e−5) + Σ|x−ẑ|m/(Σm+1e−5)] × imputed_weight + 0.1·평균|앞−뒤|",
                             "table": {"columns": ["방향", "시점 τ", "변수", "칸", "값 x", "δ", "기억 추정 x̂", "변수끼리 추정 ẑ", "섞은 값 ĉ", "다음으로 넘긴 값"], "rows": rows},
                             "values": {"앞→뒤 손실": _f(lf), "뒤→앞 손실": _f(lb), "일관성 0.1·|앞−뒤|": _f(lc), "BRITS 손실(합)": _f(brits_loss)}})
        if n_nan:
            cells = [[f"τ={a + 1}", _d(win_dates[a]), feats[b], _f(imp_f[a, b]), _f(imp_b[a, b]), _f(imp[a, b])] for a, b in zip(*np.where(np.isnan(Xw)))]
            out["steps"][-1]["table2"] = {"columns": ["시점", "날짜", "변수", "앞→뒤 값", "뒤→앞 값", "채운 값(평균)"], "rows": cells}
        out["checks"].append({"what": "Stage 1 BRITS 손 계산(채운 창) = 원본 모듈 출력", "max_abs_diff": _f(np.max(np.abs(imp - mod_imp))), "ok": bool(np.allclose(imp, mod_imp, rtol=0, atol=1e-5))})
        out["checks"].append({"what": "Stage 1 BRITS 손 계산(보간 손실) = 원본 모듈 손실", "max_abs_diff": _f(abs(brits_loss - float(imp_loss))), "ok": bool(abs(brits_loss - float(imp_loss)) <= 1e-5 * max(1, abs(brits_loss)))})
        xi = imp
    else:
        xi = x32
        out["steps"].append({"id": "brits", "title": "Stage 1 — 빈 칸 채우기 (꺼짐)", "plain": "use_brits=아니오: 창을 그대로 넘깁니다(빈 칸은 엔진이 학습 평균으로 이미 채움). 보간 손실 0.", "formula": "출력 = 입력"})
    # ---- Stage 2 분해
    ma = args.moving_avg
    if isinstance(ma, list):
        dec_in, dec_out = grabbed["decomposition"]
        seas, trend = _np(dec_out[0][0]), _np(dec_out[1][0])
        out["steps"].append({"id": "decomp", "title": "Stage 2 — 추세·주기 나누기 (여러 창 평균의 가중합)", "plain": "moving_avg 가 여러 값이면 원본 series_decomp_multi 가 학습되는 비율로 섞습니다(손 계산 생략 없이 모듈 값만 표시).",
                             "formula": "추세 = Σ softmax(w)·이동평균_k,  주기 = x − 추세"})
    else:
        kk = int(ma)
        front, end = (kk - 1) // 2, (kk - 1) // 2 if kk % 2 else kk // 2
        padded = np.vstack([np.repeat(xi[:1], front, 0), xi, np.repeat(xi[-1:], end, 0)])
        trend = np.array([padded[a: a + kk].mean(0) for a in range(L)])
        seas = xi - trend
        mt, ms_ = _np(grabbed["decomposition"][1][1][0]), _np(grabbed["decomposition"][1][0][0])
        rows = [[f"τ={a + 1}", feats[b], _f(xi[a, b]), _f(trend[a, b]), _f(seas[a, b])] for a in range(L) for b in range(K)]
        out["steps"].append({"id": "decomp", "title": f"Stage 2 — 추세·주기 나누기 (이동평균 {kk})",
                             "plain": f"변수마다 따로, 창 안에서 {kk}개 시점의 평균을 그 시점의 '추세'(느린 흐름)로 봅니다. 창 양끝에는 이동평균을 만들 이웃이 모자라서 첫 값을 앞에 {front}번, 마지막 값을 뒤에 {end}번 복사해 붙입니다"
                                      + (" (짝수 창이라 앞뒤가 다름 — 원본 그대로)" if kk % 2 == 0 else "") + ". 원래 값 − 추세 = '주기'(빠른 출렁임).",
                             "formula": f"추세_τ = 평균(붙인 열[τ … τ+{kk - 1}]),  주기_τ = x_τ − 추세_τ  (AvgPool1d, stride 1)",
                             "table": {"columns": ["시점", "변수", "채운 값", "추세", "주기"], "rows": rows}})
        out["checks"].append({"what": "Stage 2 분해 손 계산 = 원본 모듈", "max_abs_diff": _f(max(np.max(np.abs(trend - mt)), np.max(np.abs(seas - ms_)))), "ok": bool(np.allclose(trend, mt, atol=1e-5) and np.allclose(seas, ms_, atol=1e-5))})
    # ---- 추세 경로
    def conv1(name, xin):  # xin [C_in, L], 커널 1
        W = _np(getattr(net, name).weight)[:, :, 0]; b = _np(getattr(net, name).bias)
        return W @ xin + b[:, None], W, b
    tr_rows = []
    a1, W1, _ = conv1("Conv1d_Trend_1", trend.T)
    a2, W2, _ = conv1("Conv1d_Trend_2", a1)
    a3, W3, _ = conv1("Conv1d_Trend_3", a2)
    wl, bl = _np(net.Linear_Trend.weight)[0], float(_np(net.Linear_Trend.bias)[0])
    t_out = float(wl @ a3[0] + bl)
    for nm_, A in (("변수 섞기 ①", a1), ("변수 섞기 ②", a2), ("변수 섞기 ③", a3)):
        for ch in range(A.shape[0]):
            tr_rows.append([nm_, ch + 1, *[_f(v) for v in A[ch]]])
    out["steps"].append({"id": "trend", "title": "추세 → 예측값 (Conv1d 3단 + 시간 축 Linear)",
                         "plain": f"추세 {K}개 변수를 같은 시점끼리만 가중합해 {W1.shape[0]}개 → {W2.shape[0]}개 → 1개 흐름으로 줄입니다(커널 1 = 시점끼리 섞지 않음). 마지막으로 시간 {L}칸에 가중치를 곱해 더해 값 하나를 냅니다(시작 가중치 1/{L} = 평균).",
                         "formula": "a¹_τ = W¹·추세_τ + b¹,  a² = W²·a¹ + b²,  a³ = W³·a² + b³,  추세 예측 = Σ_τ w_τ·a³_τ + b",
                         "table": {"columns": ["단계", "채널", *[f"τ={a + 1}" for a in range(L)]], "rows": tr_rows + [["시간 가중치 w", "", *[_f(v) for v in wl]]]},
                         "values": {"시간 Linear 편향 b": bl, "추세 예측": t_out}})
    mod_t = float(_np(grabbed["Linear_Trend"][1])[0, 0, 0])
    out["checks"].append({"what": "추세 경로 손 계산 = 원본 모듈", "max_abs_diff": _f(abs(t_out - mod_t)), "ok": bool(abs(t_out - mod_t) <= 1e-5)})
    # ---- Stage 3 LSTM-VAE
    vae = net.LSTM_VAE
    e = seas
    enc_rows = []
    for li in range(1, 5):
        mod = getattr(vae, f"encoder_lstm_{li}")
        e, _ = _lstm_np(e, _w(mod, "weight_ih_l0"), _w(mod, "weight_hh_l0"), _w(mod, "bias_ih_l0"), _w(mod, "bias_hh_l0"))
        for a in range(L):
            enc_rows.append([f"인코더 LSTM {li}", f"τ={a + 1}", *[_f(v) for v in e[a]]])
    mu = np.maximum(e @ _w(vae.encoder_fc_mu, "weight").T + _w(vae.encoder_fc_mu, "bias"), 0)
    lv = np.maximum(e @ _w(vae.encoder_fc_logvar, "weight").T + _w(vae.encoder_fc_logvar, "bias"), 0)
    sd = np.exp(0.5 * lv)
    vae_out = grabbed["LSTM_VAE"][1]
    z_mod = _np(vae_out[4][0])
    eps = (z_mod - mu) / sd  # 이번 계산에서 뽑힌 무작위 수(z = μ + ε·σ 를 거꾸로)
    lat = mu.shape[1]
    out["steps"].append({"id": "vae_enc", "title": f"Stage 3 — 주기 성분을 공통요인 {lat}개로 줄이기 (LSTM-VAE 인코더)",
                         "plain": f"주기 성분({K}개 변수)을 LSTM 4층에 차례로 통과시켜 {K} → {vae.hidden_size_1} → {vae.hidden_size_2} → {lat} 개로 줄입니다. 각 층의 시점별 출력 전부를 적었습니다. "
                                  "마지막 층 출력에서 시점마다 평균 μ 와 로그분산 log σ² 를 Linear 로 만들고, 원본처럼 둘 다 ReLU 를 거쳐 0 이상으로 만듭니다(그래서 σ ≥ 1). 그다음 z = μ + ε·σ (ε 는 표준정규 무작위 수 — 예측할 때도 뽑음).",
                         "formula": "μ = ReLU(W_μ·e + b_μ),  log σ² = ReLU(W_v·e + b_v),  σ = exp(½ log σ²),  z = μ + ε·σ,  ε ~ N(0, 1)",
                         "table": {"columns": ["층", "시점", *[f"단위{u + 1}" for u in range(max(K, 1))]], "rows": enc_rows}})
    zrows = [[f"τ={a + 1}", u + 1, _f(e[a, u]), _f(mu[a, u]), _f(lv[a, u]), _f(sd[a, u]), _f(eps[a, u]), _f(z_mod[a, u])] for a in range(L) for u in range(lat)]
    out["steps"].append({"id": "vae_z", "title": "공통요인 z — 평균·흩어짐·이번에 뽑힌 무작위 수",
                         "plain": "시점·요인마다 μ, log σ², σ, 이번 계산에서 뽑힌 ε, 그리고 z 입니다. σ 가 1 이상이라 z 에 들어가는 무작위 흔들림이 큽니다 — 같은 창도 계산할 때마다 예측이 달라지는 원인입니다(아래 '같은 창 반복 예측' 참조).",
                         "formula": "z = μ + ε·σ", "table": {"columns": ["시점", "요인", "인코더 출력 e", "μ", "log σ²", "σ", "ε(이번)", "z"], "rows": zrows}})
    out["checks"].append({"what": "Stage 3 인코더·μ 손 계산 = 원본 모듈 μ", "max_abs_diff": _f(np.max(np.abs(mu - _np(vae_out[1][0])))), "ok": bool(np.allclose(mu, _np(vae_out[1][0]), atol=1e-5))})
    out["checks"].append({"what": "Stage 3 log σ² 손 계산 = 원본 모듈", "max_abs_diff": _f(np.max(np.abs(lv - _np(vae_out[2][0])))), "ok": bool(np.allclose(lv, _np(vae_out[2][0]), atol=1e-5))})
    # 디코더(손실용)
    dz = np.hstack([e, z_mod])
    for li in range(1, 5):
        mod = getattr(vae, f"decoder_lstm_{li}")
        dz, _ = _lstm_np(dz, _w(mod, "weight_ih_l0"), _w(mod, "weight_hh_l0"), _w(mod, "bias_ih_l0"), _w(mod, "bias_hh_l0"))
    rec_ = dz @ _w(vae.output_fc, "weight").T + _w(vae.output_fc, "bias")
    mse = float(np.mean((rec_ - seas) ** 2))
    kld = np.mean(-0.5 * np.sum(1 + lv - mu ** 2 - np.exp(lv), axis=0)[None, :], axis=0)  # 시간축 합 → (배치 평균, 배치 1)
    vae_hand = float(np.sum(0.5 * mse + 0.5 * kld))
    out["steps"].append({"id": "vae_dec", "title": "디코더로 주기 성분 되살려 보기 → VAE 손실 (학습 때만 쓰임)",
                         "plain": f"[인코더 출력, z] 를 LSTM 4층 + Linear 로 통과시켜 주기 성분을 되살려 보고, 원래 주기 성분과의 차이(MSE)와 KLD(μ·σ 가 표준정규에서 얼마나 먼지)를 원본 식대로 더합니다. 이 값은 학습 손실에만 들어가고 예측에는 쓰이지 않습니다. "
                                  f"원본 식은 KLD 가 요인별로 남은 채 합해져 실제 값 = 0.5 × {lat} × MSE + 0.5 × ΣKLD 입니다.",
                         "formula": "KLD_j = −½ Σ_τ (1 + log σ² − μ² − σ²),  VAE 손실 = Σ_j (0.5·MSE + 0.5·KLD_j)",
                         "table": {"columns": ["시점", "변수", "주기(원래)", "되살린 값", "차이²"], "rows": [[f"τ={a + 1}", feats[b], _f(seas[a, b]), _f(rec_[a, b]), _f((rec_[a, b] - seas[a, b]) ** 2)] for a in range(L) for b in range(K)]},
                         "values": {"MSE": mse, **{f"KLD 요인{j + 1}": _f(v) for j, v in enumerate(kld)}, "VAE 손실": vae_hand}})
    out["checks"].append({"what": "VAE 손실 손 계산 = 원본 loss_function", "max_abs_diff": _f(abs(vae_hand - float(vae_loss))), "ok": bool(abs(vae_hand - float(vae_loss)) <= 1e-4 * max(1, abs(vae_hand)))})
    # 주기 경로
    xz = np.vstack([e.T, z_mod.T])  # [2·lat, L]
    s1, _, _ = conv1("Conv1d_Seasonal_1", xz)
    s2, _, _ = conv1("Conv1d_Seasonal_2", s1)
    s3, _, _ = conv1("Conv1d_Seasonal_3", s2)
    ws, bs = _np(net.Linear_Seasonal.weight)[0], float(_np(net.Linear_Seasonal.bias)[0])
    s_out = float(ws @ s3[0] + bs)
    srows = [[nm_, ch + 1, *[_f(v) for v in A[ch]]] for nm_, A in (("[e, z] 입력", xz), ("섞기 ①", s1), ("섞기 ②", s2), ("섞기 ③", s3)) for ch in range(A.shape[0])]
    out["steps"].append({"id": "seasonal", "title": "주기 → 예측값 (Conv1d 3단 + 시간 축 Linear)",
                         "plain": f"[인코더 출력 {lat}개, z {lat}개] = {2 * lat}개 흐름을 같은 시점끼리 가중합해 {s1.shape[0]} → {s2.shape[0]} → 1 개로 줄이고, 시간 {L}칸에 가중치를 곱해 더합니다.",
                         "formula": "s¹ = W¹·[e; z] + b¹,  s² = W²·s¹ + b²,  s³ = W³·s² + b³,  주기 예측 = Σ_τ w_τ·s³_τ + b",
                         "table": {"columns": ["단계", "채널", *[f"τ={a + 1}" for a in range(L)]], "rows": srows + [["시간 가중치 w", "", *[_f(v) for v in ws]]]},
                         "values": {"시간 Linear 편향 b": bs, "주기 예측": s_out}})
    mod_s = float(_np(grabbed["Linear_Seasonal"][1])[0, 0, 0])
    out["checks"].append({"what": "주기 경로 손 계산 = 원본 모듈", "max_abs_diff": _f(abs(s_out - mod_s)), "ok": bool(abs(s_out - mod_s) <= 1e-5)})
    # ---- Stage 4
    if args.combination:
        al = float(_np(net.alpha).ravel()[0])
        st_hand = al * s_out + (1 - al) * t_out
    else:
        al = None
        st_hand = s_out + t_out
    y_hand = st_hand * m.y_sd + m.y_mu
    out["steps"].append({"id": "combine", "title": "Stage 4 — 합치기 → 예측 ŷ",
                         "plain": ("두 예측을 학습된 비율 α 로 섞습니다(α 는 1 에서 시작해 범위 제한 없이 학습됨)." if al is not None else "두 예측을 그냥 더합니다(combination=아니오).")
                                  + " 그 값은 표준화된 정답 단위라 학습 구간의 평균·표준편차로 되돌립니다.",
                         "formula": ("점수 = α·주기 예측 + (1 − α)·추세 예측" if al is not None else "점수 = 주기 예측 + 추세 예측") + ",  ŷ = 점수 × σ_y + μ_y",
                         "values": {"α": al, "주기 예측": s_out, "추세 예측": t_out, "점수": st_hand, "μ_y": m.y_mu, "σ_y": m.y_sd, "ŷ(이번 계산)": y_hand}})
    out["checks"].append({"what": "Stage 4 손 계산 ŷ = 이번 순전파 출력", "manual": _f(y_hand), "model": _f(pass_pred), "max_abs_diff": _f(abs(y_hand - pass_pred)), "ok": bool(abs(y_hand - pass_pred) <= 1e-5 * max(abs(m.y_sd), 1e-12) + 1e-9)})
    # ---- 같은 창 반복 예측 (z 의 무작위)
    reps = []
    with torch.no_grad():
        xt = torch.tensor(Xw[None], dtype=torch.float32)
        for r_ in range(100):
            torch.manual_seed(1000 + r_)
            reps.append(float(net(xt)[0][0, -1, 0]) * m.y_sd + m.y_mu)
    reps = np.array(reps)
    with torch.no_grad():  # ε = 0 (z = μ) 일 때 — 원본에는 없는 계산, 비교용
        orig = net.LSTM_VAE.reparameterize
        net.LSTM_VAE.reparameterize = lambda mu_, lv_: mu_
        try:
            y0 = float(net(xt)[0][0, -1, 0]) * m.y_sd + m.y_mu
        finally:
            net.LSTM_VAE.reparameterize = orig
    out["steps"].append({"id": "noise", "title": "같은 창을 100번 예측하면 — z 의 무작위가 예측을 얼마나 흔드나",
                         "plain": "원본 BIVA 는 예측할 때도 z 를 무작위로 뽑으므로 같은 창도 계산할 때마다 값이 다릅니다. 실험은 그중 한 번의 값을 씁니다. 아래는 같은 가중치·같은 창으로 시드만 바꿔 100번 계산한 분포와, 비교용으로 무작위를 끈 값(z = μ)입니다.",
                         "formula": "ŷ_r = f(창; ε_r),  ε_r ~ N(0, 1),  r = 1 … 100",
                         "values": {"실험이 낸 값": exp_pred, "이번 계산(시드 12345)": pass_pred, "100번 평균": _f(reps.mean()), "100번 표준편차": _f(reps.std()),
                                    "최소": _f(reps.min()), "최대": _f(reps.max()), "무작위를 끈 값 (z = μ)": y0},
                         "table": {"columns": ["반복", "ŷ"], "rows": [[r_ + 1, _f(v)] for r_, v in enumerate(reps)]}})
    cap["pred"] = cap["pred"].copy()
    cap["pred"].loc[t] = exp_pred
