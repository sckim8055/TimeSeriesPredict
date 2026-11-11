"""LSTM·GRU 실제 실행 붙잡기 — 엔진의 실제 경로(_walkforward_predict → models/lstm.py·gru.py)로 마지막 폴드를 학습·예측하며 기록한다.

  python -m tools.codewalk.rnn_capture --model lstm [--config configs/f11_lstm_macro_k1.json] [--out docs/guide_build/walk/lstm_capture.pkl]

기록(전부 실행에서 나온 값):
- 데이터: 학습 폴드 표, 변환·발표시차, 표준화 평균·표준편차, 빈 칸 채움값, 창·정답·결정일, 학습/검증 개수
- 학습: epoch 별 손실, 배치 나누기(randperm)의 실제 순서, 첫 갱신(손실·기울기 크기·잘라내기·Adam 걸음), epoch 마다 기울기 잘라내기 비율
- 순전파: 학습이 끝난 가중치로 마지막 epoch 첫 배치를 net 에 넣어 models/<모델>.py 의 실행 줄·텐서(sys.settrace), 드롭아웃 마스크
- 예측: 테스트 창, 예측값(원래 단위)과 실제 값
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.experiment import ExperimentConfig, _embargo, _walkforward_predict, make_folds, prepare  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402
from tools.codewalk.trace import LineTracer  # noqa: E402


def capture(model: str = "lstm", config: str = "configs/f11_lstm_macro_k1.json", data_root: Path | None = None, epochs: int | None = None, replay: bool = True) -> dict:
    """replay=True: 마지막 폴드 앞의 폴드들을 먼저 그대로 돌려(기록 없이) 난수 상태를 실제 실험과 같게 만든 뒤 마지막 폴드를 기록한다.
    엔진은 시드를 실험 시작에 한 번만 고정하므로, 마지막 폴드의 가중치 초기값·섞는 순서는 앞 폴드들이 쓴 난수 뒤의 값이다 —
    replay 없이 마지막 폴드만 새 시드로 돌리면 결과 파일과 다른 값이 나온다(2026-11-07 확인)."""
    import importlib
    data_root = data_root or ROOT / "data"
    raw_cfg = json.loads((ROOT / config).read_text(encoding="utf-8"))
    raw_cfg["model"] = model
    if epochs:
        raw_cfg["model_params"] = {**raw_cfg.get("model_params", {}), "epochs": int(epochs)}
    reg, _ = Registry.load_master(data_root / "variable_master.xlsx")
    cfg = ExperimentConfig(**raw_cfg)
    dv = DataVersion(data_root, cfg.data_version)
    cfg, plugin, params, target_type, ds = prepare(cfg, dv, reg, device="cpu")
    MM = importlib.import_module(f"models.{model}")
    k = cfg.target.k
    folds, _ = make_folds(cfg, ds, k)
    ne = [f for f in folds if len(f.test)]
    fold = ne[-1]
    n_replayed = 0
    if replay and len(ne) > 1:
        from engine.experiment import _seed
        _seed(cfg.seed)
        _walkforward_predict(cfg, plugin, params, ds, "cpu", ne[:-1], k, target_type)   # 기록 없이 앞 폴드들 — 난수 상태를 실제 실험과 같게
        n_replayed = len(ne) - 1

    rec: dict = {"perm": [], "steps": [], "clip": {}, "first": None, "last_epoch_first_batch": None}
    orig_randperm, orig_clip, orig_adam = torch.randperm, torch.nn.utils.clip_grad_norm_, torch.optim.Adam
    state = {"ep": 0, "step_in_ep": 0, "model": None}

    def randperm(n, *a, **kw):  # 원래 randperm 그대로 + epoch 마다 순서 기록
        out = orig_randperm(n, *a, **kw)
        state["ep"] += 1; state["step_in_ep"] = 0
        rec["perm"].append(out.clone())
        return out

    def clip(params_, max_norm, *a, **kw):  # 원래 잘라내기 그대로 + 자르기 전 크기 기록
        ps = [p_ for p_ in params_]
        if "pre_clip" not in rec and state["model"] is not None:  # 첫 갱신: 가중치별 자르기 전 기울기
            nm = {id(p_): n_ for n_, p_ in state["model"].net.named_parameters()}
            rec["pre_clip"] = {nm[id(p_)]: p_.grad.detach().clone() for p_ in ps if p_.grad is not None}
        tot = orig_clip(ps, max_norm, *a, **kw)
        rec["clip"].setdefault(state["ep"], []).append(float(tot))
        state["step_in_ep"] += 1
        return tot

    class RecAdam(orig_adam):
        _n = 0

        def step(self, closure=None):
            RecAdam._n += 1
            if RecAdam._n == 1:
                net = state["model"].net
                g = {n_: float(p_.grad.detach().norm()) for n_, p_ in net.named_parameters() if p_.grad is not None}
                before = {n_: p_.detach().clone() for n_, p_ in net.named_parameters()}
                grads = {n_: p_.grad.detach().clone() for n_, p_ in net.named_parameters()}
                out = super().step(closure)
                after = {n_: p_.detach().clone() for n_, p_ in net.named_parameters()}
                g_ = self.param_groups[0]
                rec["first"] = {"grad_norms": g, "before": before, "after": after, "grads": grads, "lr": g_["lr"], "betas": g_["betas"], "eps": g_["eps"], "wd": g_["weight_decay"]}
                return out
            return super().step(closure)

    orig_init = MM.Model.fit

    def fit(self, X, y, X_val=None, y_val=None):  # 원래 fit 그대로(모델 객체만 기억)
        state["model"] = self
        return orig_init(self, X, y, X_val, y_val)

    torch.randperm, torch.nn.utils.clip_grad_norm_, torch.optim.Adam, MM.Model.fit = randperm, clip, RecAdam, fit
    if not n_replayed:
        torch.manual_seed(cfg.seed)
    cap: dict = {}
    try:
        _walkforward_predict(cfg, plugin, params, ds, "cpu", [fold], k, target_type, capture=cap)
    finally:
        torch.randperm, torch.nn.utils.clip_grad_norm_, torch.optim.Adam, MM.Model.fit = orig_randperm, orig_clip, orig_adam, orig_init
    m = cap["model"]; net = m.net
    feats = list(ds.feature_names); L = int(cfg.seq_len)
    Xw, yw = cap["Xw"], cap["yw"]
    n_tr = cap["n_tr"]
    bs = int(params["batch_size"])
    # ---- 첫 갱신의 배치(첫 epoch randperm 의 앞 bs 개)와 손실
    p0 = rec["perm"][0].tolist()
    first_idx = p0[:bs]
    yz = (yw - m.y_mu) / m.y_sd
    # ---- 추적할 배치: 마지막 epoch 의 첫 배치(창 번호), 학습이 끝난 가중치로
    pl = rec["perm"][-1].tolist()
    b_idx = pl[:bs]
    xb = torch.tensor(Xw[b_idx], dtype=torch.float32)
    acts = {}
    hs = [net.drop.register_forward_hook(lambda mod, i, o: acts.update(drop_in=i[0].detach().clone(), drop_out=o.detach().clone()) if mod.training else None)]
    tracer = LineTracer([str(ROOT / "models")])
    torch.manual_seed(2026)
    net.train()  # 학습 때 모드(드롭아웃 켜짐)
    with tracer:
        out_tr = net(xb)
    net.eval()
    with torch.no_grad():
        out_ev = net(xb)
        acts_ev = {}
    for h in hs:
        h.remove()
    loss_tr = float(torch.nn.functional.mse_loss(out_tr.detach(), torch.tensor(yz[b_idx], dtype=torch.float32)))
    # ---- 예측
    pred = cap["pred"].dropna()
    te_idx = list(cap["te_idx"])
    t_dec = pred.index[-1]
    i_dec = te_idx.index(t_dec)
    Xtw = cap["Xtw"]
    with torch.no_grad():
        reps = [float(net(torch.tensor(Xtw[i_dec:i_dec + 1], dtype=torch.float32))[0]) * m.y_sd + m.y_mu for _ in range(3)]
    sc = cap["scaler"]
    full = ds.X.loc[fold.train]
    return {
        "model": model, "config": json.loads(cfg.model_dump_json()), "config_file": config, "params": params, "feats": feats,
        "feat_names": {c: (reg.get(c).name if c in reg.ids(usable_only=False) else c) for c in feats},
        "transforms": ds.transforms_used, "lags": ds.lag_days_used, "var_names": {v: reg.get(v).name for v in (ds.transforms_used or {})}, "seq_len": L, "k": k, "embargo": _embargo(cfg, k),
        "fold": {"train": [str(fold.train[0].date()), str(fold.train[-1].date())], "test": [str(fold.test[0].date()), str(fold.test[-1].date())],
                 "n_train_rows": int(len(fold.train)), "n_folds": len(folds), "val_frac": cfg.validation.val_frac,
                 "val_from": str(fold.val_from.date()) if getattr(fold, "val_from", None) is not None else None},
        "raw_train": full, "y_train": ds.y.loc[fold.train], "target_level": ds.prices["close"],
        "scaler_mean": sc.mean_.copy(), "scaler_scale": sc.scale_.copy(), "fill_val": np.asarray(cap["fill_val"]).copy(),
        "Xw": Xw, "yw": yw, "win_idx": [str(d.date()) for d in cap["win_idx"]], "n_tr": n_tr, "n_va": cap["n_va"],
        "y_mu": m.y_mu, "y_sd": m.y_sd, "history": m.history, "best_epoch": m.best_epoch,
        "perm_first": p0, "perm_last": pl, "first_idx": first_idx, "batch_idx": b_idx, "batch_size": bs,
        "clip": {e: v for e, v in rec["clip"].items()}, "first_step": {**rec["first"], "pre_clip": rec.get("pre_clip")},
        "batch": {"x": xb, "y_z": yz[b_idx].tolist(), "out_train": out_tr.detach(), "out_eval": out_ev.detach(), "loss_train": loss_tr,
                  "drop_in": acts.get("drop_in"), "drop_out": acts.get("drop_out")},
        "events": tracer.events,
        "predict": {"date": str(t_dec.date()), "pred": float(pred.loc[t_dec]), "actual": float(ds.y.get(t_dec, np.nan)),
                    "level": float(ds.prices["close"].loc[t_dec]), "window": Xtw[i_dec], "repeats": reps, "n_test": len(pred),
                    "window_dates": [str(d.date()) for d in cap["te_full"].index[i_dec:i_dec + L]]},
        "net_state": {k_: v.detach().clone() for k_, v in net.state_dict().items()},
        "n_params": int(sum(p_.numel() for p_ in net.parameters())),
        "n_replayed": n_replayed, "torch_threads": int(torch.get_num_threads()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--config", default="configs/f11_lstm_macro_k1.json")
    ap.add_argument("--out")
    a = ap.parse_args()
    cap = capture(a.model, a.config)
    out = Path(a.out or ROOT / f"docs/guide_build/walk/{a.model}_capture.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cap, open(out, "wb"))
    h = cap["history"]
    print("저장:", out, "· epoch", len(h), "· best", cap["best_epoch"], "· 학습/검증 창", cap["n_tr"], cap["n_va"], "· 예측", cap["predict"]["date"], round(cap["predict"]["pred"], 6))
