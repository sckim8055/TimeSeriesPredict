"""모델 흐름도용 '실제 모양' 계산 — 화면의 모델별 알고리즘 그림에 숫자를 채운다.

- 지금 실험 설정으로 데이터셋을 실제로 만들어 입력 변수 이름·개수·행 수·기간을 얻는다(실패하면 변수 수로 추정하고 note 에 적음).
- 딥러닝(LSTM·GRU·BIVA): 지금 파라미터로 신경망을 만들고 시험 입력을 한 번 통과시켜(forward hook) 모듈마다
  입력·출력 모양과 가중치 수를 기록한다 → 그림의 모양 표기가 코드와 어긋날 수 없다. 학습은 하지 않는다.
- 그 밖(Ridge·LightGBM·ARIMA·SARIMA·VECM·규칙): 파라미터와 데이터 크기로 계수·나무·시차 구조를 계산한다.
모양 표기의 첫 숫자(배치)는 화면에서 'B' 로 바꿔 보인다.
"""
from __future__ import annotations

import pandas as pd

import logging

import numpy as np

from engine.experiment import ExperimentConfig, build_dataset, load_plugin
from models.base import resolve_params

log = logging.getLogger("tsp.engine.flow")
B = 2  # 시험 입력 배치 크기


def _shape(o):
    import torch
    if torch.is_tensor(o):
        return list(o.shape)
    if isinstance(o, dict):
        return {k: _shape(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_shape(v) for v in o]
    return None


def _torch_trace(net, x, depth: int = 3) -> dict:
    """모듈 이름 → {type, in, out, params}. 첫 호출만 기록(RNN 셀처럼 시점마다 불리는 모듈)."""
    import torch
    rec: dict = {}
    hs = []
    for name, m in net.named_modules():
        if not name or name.count(".") >= depth:
            continue

        def hook(mod, inp, out, name=name):
            if name not in rec:
                rec[name] = {"type": type(mod).__name__, "in": _shape(inp[0]) if inp else None, "out": _shape(out),
                             "params": int(sum(p.numel() for p in mod.parameters()))}
        hs.append(m.register_forward_hook(hook))
    net.eval()
    with torch.no_grad():
        net(x)
    for h in hs:
        h.remove()
    rec["_total"] = {"params": int(sum(p.numel() for p in net.parameters()))}
    return rec


def model_flow(cfg_dict: dict, dv, reg) -> dict:
    cfg = ExperimentConfig(**cfg_dict)
    plugin = load_plugin(cfg.model)
    freq = reg.get(cfg.target_variable).freq if cfg.mode == "forecast" and cfg.target_variable else cfg.frequency
    params = resolve_params(plugin.PARAMS, cfg.model_params, freq)
    seq_len = cfg.seq_len or {"D": 20, "W": 13, "M": 12, **(getattr(plugin, "SEQ_LEN_DEFAULT", None) or {})}.get(freq, 20)
    out: dict = {"model": cfg.model, "name": plugin.NAME, "input_shape": plugin.INPUT_SHAPE, "params": params, "freq": freq,
                 "seq_len": seq_len if plugin.INPUT_SHAPE == "window" else None, "notes": []}
    feats, n_rows, period, levels_cols = None, None, None, None
    try:
        ds = build_dataset(cfg, dv, reg)
        feats = list(ds.feature_names)
        Xv = ds.X[feats].dropna(how="all") if feats else ds.X
        n_rows = int(len(ds.y.dropna()))
        period = [str(ds.y.dropna().index[0].date()), str(ds.y.dropna().index[-1].date())] if n_rows else None
        out["nan_cells"] = int(ds.X[feats].isna().sum().sum()) if feats else 0
        out["late_start"] = {c: str(ds.X[c].first_valid_index().date()) for c in feats
                             if ds.X[c].first_valid_index() is not None and ds.X[c].first_valid_index() > Xv.index[0]}
        if ds.levels is not None and len(ds.levels.columns):
            levels_cols = ["대상"] + list(ds.levels.columns)
    except Exception as e:  # noqa: BLE001 — 데이터 없이도 그림은 그린다
        out["notes"].append(f"데이터셋을 만들지 못해 변수 수로 추정: {e}")
        feats = list(cfg.variable_ids)
    k = max(1, len(feats))
    # 입력 열이 어디서 왔는지: 'A_' 로 시작하면 가격(대상)으로 만든 자체 지표, 나머지는 사용자가 고른 변수(변환 접미사 포함)
    from engine.features import describe_feature
    n_tech = sum(1 for c in feats if str(c).startswith("A_"))
    out["input"] = {"features": feats, "n_features": k, "n_rows": n_rows, "period": period,
                    "n_technical": n_tech, "n_selected": len(feats) - n_tech,
                    "feature_desc": {c: describe_feature(str(c), reg) for c in feats},
                    "target": {"kind": cfg.target.kind, "k": cfg.target.k, "variable": cfg.target_variable if cfg.mode == "forecast" else cfg.ticker},
                    "train_len": cfg.validation.train_len, "val_frac": cfg.validation.val_frac,
                    "test_len": cfg.validation.test_len, "threshold": cfg.trade.threshold if cfg.mode == "trading" else None}
    # 화면의 기술지표 개별 선택용: 이 주기·모드에서 만들 수 있는 전체 목록(고르지 않은 것 포함)과 묶음
    from engine.features import technical_columns, technical_group
    try:
        allt = technical_columns(freq, with_volume=(cfg.mode != "forecast"))
    except Exception:  # noqa: BLE001 — 기술지표가 정의되지 않은 주기(Q·H·Y)
        allt = []
    out["input"]["technical_all"] = [{"name": c, "group": technical_group(c), "desc": describe_feature(c, reg)} for c in allt]
    shape = plugin.INPUT_SHAPE
    # 한 폴드에서 실제로 학습에 쓰는 개수(엔진 규칙 그대로: 학습 창 안에서 비움 1+k(예측 모드 k)를 빼고,
    # 검증을 쓰는 모델은 뒤쪽 val_frac 를 떼되 검증 10개 미만이거나 남는 학습이 10개 미만이면 검증 없이 전부 학습).
    # 결측 행 제외·창 모델의 앞 seq_len−1 행 제외 전 값 — 실제 값은 결과의 periods(폴드 기록)에 남는다.
    from engine.experiment import _embargo
    emb = _embargo(cfg, int(cfg.target.k))
    rows = max(0, int(cfg.validation.train_len) - emb)
    if shape == "window":
        rows = max(0, rows - (int(seq_len) - 1))
    uses_val = bool(getattr(plugin.Model, "USES_VALIDATION", True)) and shape not in ("rule", "series", "levels")
    n_val = int(rows * float(cfg.validation.val_frac)) if uses_val else 0
    v = cfg.validation
    ts_ = v.test_start or v.split_date
    if v.mode == "split" and ts_ and n_rows:  # 1회 분할: 학습 = 테스트 시작 전 전부(− 비움), 검증 = 날짜 지정 또는 뒤쪽 val_frac
        yi = ds.y.dropna().index
        rows = max(0, int((yi < pd.Timestamp(ts_)).sum()) - emb - (int(seq_len) - 1 if shape == "window" else 0))
        if uses_val:
            n_val = int(((yi >= pd.Timestamp(v.val_start)) & (yi < pd.Timestamp(ts_))).sum()) - emb if v.val_start else int(rows * float(v.val_frac))
    has_val = uses_val and n_val >= 10 and rows - n_val - emb >= 10
    if cfg.model == "lightgbm_model" and n_val <= 10:  # LightGBM 은 검증이 11개 이상일 때만 조기종료
        has_val = False
    out["input"]["fit_plan"] = {"embargo": emb, "rows": rows, "uses_val": uses_val, "n_val": n_val if has_val else 0,
                                "n_fit": rows - n_val - emb if has_val else rows, "has_val": has_val}
    if uses_val and not has_val:
        out["notes"].append(f"검증 {n_val}개 < {11 if cfg.model == 'lightgbm_model' else 10} — 이 설정에서는 검증 없이 학습 창 전체로 학습하고 조기종료를 하지 않습니다"
                            + (" (마지막 epoch 가중치 사용)" if shape == "window" and cfg.model != "biva" else " (BIVA: 학습 손실로 최적 시점 선택)" if cfg.model == "biva" else " (나무를 n_estimators 개 모두 사용)"))
    try:
        if cfg.model in ("lstm", "gru"):
            import torch
            mod = __import__(f"models.{cfg.model}", fromlist=["_Net"])
            net = mod._Net(k, params["hidden"], params["layers"], params["dropout"])
            out["trace"] = _torch_trace(net, torch.rand(B, seq_len, k))
            rnn = net.lstm if cfg.model == "lstm" else net.gru
            out["weights"] = {n: list(p.shape) for n, p in rnn.named_parameters()}
        elif cfg.model == "biva":
            import torch
            m = plugin.Model(params, n_features=k, target_type="regression", device="cpu")
            net = m._build(seq_len)
            x = torch.rand(B, seq_len, k)
            out["trace"] = _torch_trace(net, x, depth=3)
            out["latent"] = int(params["vae_latent_size"])
        elif shape == "table":
            n_tr = out["input"]["fit_plan"]["n_fit"]
            if cfg.model == "lightgbm_model":
                out["trees"] = {"n": params["n_estimators"], "leaves": params["num_leaves"], "depth": params["max_depth"],
                                "rows_per_tree": int(round(n_tr * params["subsample"])), "cols_per_tree": max(1, int(round(k * params["colsample_bytree"]))),
                                "min_leaf": params["min_child_samples"], "n_train": n_tr}
                if params["min_child_samples"] * 2 > int(round(n_tr * params["subsample"])):
                    out["notes"].append(f"min_child_samples({params['min_child_samples']})×2 > 나무 하나의 표본({int(round(n_tr * params['subsample']))}) — 나눌 수 없어 상수 예측이 됩니다")
            else:
                out["coef"] = {"n": k, "n_train": n_tr}
        elif shape == "series":
            p = params
            ar = list(range(1, p["p"] + 1)); ma = list(range(1, p["q"] + 1))
            if cfg.model == "sarima":
                s = p["s"]
                ar += [s * i for i in range(1, p["P"] + 1)]; ma += [s * i for i in range(1, p["Q"] + 1)]
                ar += [s * i + j for i in range(1, p["P"] + 1) for j in range(1, p["p"] + 1)]  # 곱 항
                ma += [s * i + j for i in range(1, p["Q"] + 1) for j in range(1, p["q"] + 1)]
            # 추정 계수 = φ p개 + θ q개 + Φ P개 + Θ Q개 (+상수). 곱 항(s+j)의 계수는 φ·Φ 의 곱이라 따로 추정하지 않는다
            n_coef = p["p"] + p["q"] + (p.get("P", 0) + p.get("Q", 0) if cfg.model == "sarima" else 0) + (1 if p.get("trend", "n") == "c" else 0)
            out["lags"] = {"ar": sorted(set(ar)), "ma": sorted(set(ma)), "n_coef": n_coef}
        elif shape == "levels":
            K = len(levels_cols) if levels_cols else 1 + len(cfg.variable_ids)
            r = int(params["coint_rank"]); pk = int(params["k_ar_diff"])
            out["vecm"] = {"vars": levels_cols or (["대상"] + list(cfg.variable_ids)), "K": K, "r": r if r else "자동(Johansen)", "p": pk,
                           "alpha": [K, r or "r"], "beta": [K, r or "r"], "gamma": [K, K * pk], "n_train": out["input"]["fit_plan"]["n_fit"]}
    except Exception as e:  # noqa: BLE001
        out["notes"].append(f"모양 계산 실패: {type(e).__name__}: {e}")
        log.exception("model_flow 실패")
    log.debug("model_flow %s: 입력 %d개 · %s", cfg.model, k, {kk: vv for kk, vv in out.items() if kk in ("seq_len", "lags", "vecm", "trees", "coef")})
    return out
