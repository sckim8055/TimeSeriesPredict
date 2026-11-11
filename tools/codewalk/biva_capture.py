"""BIVA 실제 실행 붙잡기 — 원본 BIVA(models/biva_src)를 실데이터·실제 엔진 경로로 학습시키고, 한 배치의 순전파를 줄 단위로 기록한다.

  python -m tools.codewalk.biva_capture [--epochs 20] [--config configs/repro_biva_orig.json]

기록하는 것(전부 실제 실행에서 나온 값, 손으로 만든 값 없음)
- 데이터: 학습 폴드의 원래 표(날짜 × 변수), 빈 칸, MinMax 최소·최대, 창(make_windows)과 각 창의 결정일·정답, 검증·비움·학습 개수, 정답 표준화 μ·σ
- 학습: epoch 별 학습·검증 손실·학습률, 첫 번째 학습 배치(창 번호·손실 3종), 첫 Adam 갱신(기울기 크기·α 값의 변화), 조기종료가 고른 epoch
- 순전파: 학습이 끝난 가중치로, 마지막 epoch 에 실제로 쓰인 '빈 칸이 있는 배치' 하나를 BIVA.forward 에 넣어 원본 코드의 실행 줄·만든 텐서를 차례대로
- 예측: 테스트 창의 예측(원래 단위)과 실제 값, 같은 창 반복 예측의 흔들림
결과는 pickle(텐서 포함) — tools/codewalk/biva_script.py 가 단계 대본(JSON)으로 바꾼다.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.experiment import ExperimentConfig, _embargo, _walkforward_predict, make_folds, prepare  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402
from tools.codewalk.trace import LineTracer  # noqa: E402


def capture(config: str = "configs/repro_biva_orig.json", epochs: int = 20, data_root: Path | None = None) -> dict:
    import models.biva as MB

    data_root = data_root or ROOT / "data"
    raw_cfg = json.loads((ROOT / config).read_text(encoding="utf-8"))
    raw_cfg["model_params"] = {**raw_cfg.get("model_params", {}), "train_epochs": int(epochs)}
    reg, _ = Registry.load_master(data_root / "variable_master.xlsx")
    cfg = ExperimentConfig(**raw_cfg)
    dv = DataVersion(data_root, cfg.data_version)
    cfg, plugin, params, target_type, ds = prepare(cfg, dv, reg, device="cpu")
    k = cfg.target.k
    folds, is_split = make_folds(cfg, ds, k)
    fold = [f for f in folds if len(f.test)][-1]

    rec: dict = {"batches_last_epoch": [], "first_step": None, "train_calls": 0}
    orig_build = MB.Model._build

    def _build(self, seq_len):  # 원래 _build 그대로 + 기록용 훅만 붙임(계산 불변)
        net = orig_build(self, seq_len)
        rec["init_alpha"] = float(net.alpha.detach().ravel()[0])

        def pre(mod, inp):
            if mod.training:
                rec["train_calls"] += 1
                ep = len(self.history) if hasattr(self, "history") else 0
                rec.setdefault("alpha_epoch", {}).setdefault(ep + 1, float(mod.alpha.detach().ravel()[0]))
                if ep == params["train_epochs"] - 1 or rec.get("_last_ep") == ep:
                    if rec.get("_last_ep") != ep:
                        rec["batches_last_epoch"] = []; rec["_last_ep"] = ep
                    rec["batches_last_epoch"].append(inp[0].detach().clone())

        def post(mod, inp, out):
            if mod.training and rec["first_step"] is None:
                states, vae_loss, _, _, imp_loss, _ = out
                rec["first_step"] = {"x": inp[0].detach().clone(), "states": states.detach().clone(),
                                     "vae_loss": float(vae_loss), "imp_loss": float(imp_loss)}
        net.register_forward_pre_hook(pre)
        net.register_forward_hook(post)

        def vae_stat(kind):  # 학습 배치마다 μ·log σ² 의 ReLU 앞 값이 양수인 비율(0 이면 ReLU 뒤 전부 0 → z = ε)
            def h(mod, inp, out):
                if mod.training:
                    ep = len(self.history) + 1 if hasattr(self, "history") else 1
                    st = rec.setdefault("vae_pos", {}).setdefault(ep, {"mu": 0, "lv": 0, "n": 0, "mu_max": -1e9, "lv_max": -1e9})
                    o = out.detach()
                    st[kind] += int((o > 0).sum()); st[kind + "_max"] = max(st[kind + "_max"], float(o.max()))
                    if kind == "mu":
                        st["n"] += o.numel()
            return h
        net.LSTM_VAE.encoder_fc_mu.register_forward_hook(vae_stat("mu"))
        net.LSTM_VAE.encoder_fc_logvar.register_forward_hook(vae_stat("lv"))
        rec["net"] = net
        return net

    orig_adam = MB.optim.Adam

    class RecAdam(orig_adam):  # 첫 갱신만 기록: 모듈별 기울기 크기, α 의 기울기·이전·이후 값
        _n = 0

        def step(self, closure=None):
            RecAdam._n += 1
            if RecAdam._n == 1:
                net = rec["net"]
                g = {}
                for name, mod in net.named_children():
                    gs = [p_.grad.detach().norm() ** 2 for p_ in mod.parameters() if p_.grad is not None]
                    if gs:
                        g[name] = float(torch.stack(gs).sum().sqrt())
                a_before = float(net.alpha.detach().ravel()[0])
                a_grad = float(net.alpha.grad.ravel()[0])
                out = super().step(closure)
                rec["first_step"].update(grad_norms=g, alpha_before=a_before, alpha_grad=a_grad, alpha_after=float(net.alpha.detach().ravel()[0]),
                                         lr=self.param_groups[0]["lr"], betas=self.param_groups[0]["betas"], eps=self.param_groups[0]["eps"])
                return out
            return super().step(closure)

    MB.Model._build = _build
    MB.optim.Adam = RecAdam
    torch.manual_seed(cfg.seed)
    cap: dict = {}
    try:
        _walkforward_predict(cfg, plugin, params, ds, "cpu", [fold], k, target_type, capture=cap)
    finally:
        MB.Model._build = orig_build
        MB.optim.Adam = orig_adam
    m = cap["model"]; net = m.net
    feats = list(ds.feature_names)
    L = int(cfg.seq_len)
    # ---- 첫 학습 배치가 어느 창인지(입력이 같은 창을 찾아 번호로)
    Xw, yw, win_idx = cap["Xw"], cap["yw"], cap["win_idx"]
    def idx_of(xb):
        out = []
        for w in xb.numpy():
            hit = [i for i in range(len(Xw)) if np.allclose(np.nan_to_num(Xw[i], nan=-9), np.nan_to_num(w, nan=-9), atol=1e-6)]
            out.append(hit[0] if hit else None)
        return out
    rec["first_step"]["win"] = idx_of(rec["first_step"]["x"])
    yz = (yw - m.y_mu) / m.y_sd
    fs = rec["first_step"]
    by = torch.tensor(yz[fs["win"]], dtype=torch.float32).view(-1, 1, 1)
    fs["mse"] = float(torch.nn.functional.mse_loss(fs["states"], by))
    fs["y_z"] = by.view(-1).tolist()
    # ---- 추적할 배치: 마지막 epoch 에 실제로 쓰인 배치 중 빈 칸이 가장 많은 것
    batches = rec["batches_last_epoch"]
    nn_ = [int(torch.isnan(b).sum()) for b in batches]
    bi = int(np.argmax(nn_)) if batches else None
    xb = batches[bi]
    b_win = idx_of(xb)
    tracer = LineTracer([str(ROOT / "models" / "biva_src")])
    torch.manual_seed(2026)
    net.train()  # 학습 때와 같은 모드(이 모델은 dropout·batchnorm 을 쓰지 않아 eval 과 값이 같음)
    with tracer:
        out = net(xb)
    states, vae_loss, recon, seas_init, imp_loss, imputed = out
    yb = torch.tensor(yz[b_win], dtype=torch.float32).view(-1, 1, 1)
    loss_states = torch.nn.functional.mse_loss(states, yb)
    # ---- 예측(테스트 창): 엔진이 낸 값 + 같은 창 반복 예측
    pred = cap["pred"].dropna()
    te_idx = list(cap["te_idx"])
    yv = ds.y.reindex(pred.index)
    t_dec = yv.dropna().index[-1]  # 실제 값을 아는 마지막 예측일
    i_dec = te_idx.index(t_dec)
    Xtw = cap["Xtw"]
    reps = []
    net.eval()
    with torch.no_grad():
        for r in range(100):
            torch.manual_seed(r)
            reps.append(float(net(torch.tensor(Xtw[i_dec:i_dec + 1], dtype=torch.float32))[0][0, -1, 0]) * m.y_sd + m.y_mu)
    # ---- 고른 가중치로 모든 창(학습+검증)의 μ·log σ² (ReLU 앞) — 몇 칸이 양수인가
    vv = {}
    hs = [net.LSTM_VAE.encoder_fc_mu.register_forward_hook(lambda m_, i_, o_: vv.__setitem__("mu", o_.detach())),
          net.LSTM_VAE.encoder_fc_logvar.register_forward_hook(lambda m_, i_, o_: vv.__setitem__("lv", o_.detach()))]
    with torch.no_grad():
        torch.manual_seed(0)
        net(torch.tensor(np.asarray(Xw, dtype=np.float32)))
    for h_ in hs:
        h_.remove()
    vae_all = {"n_windows": int(len(Xw)), "mu_pos": int((vv["mu"] > 0).sum()), "lv_pos": int((vv["lv"] > 0).sum()), "n": int(vv["mu"].numel()),
               "mu_max": float(vv["mu"].max()), "lv_max": float(vv["lv"].max())}
    tr_rows = ds.X.loc[fold.train]
    return {
        "config": json.loads(cfg.model_dump_json()), "params": params, "epochs": int(epochs), "feats": feats,
        "feat_names": {c: (reg.get(c).name if c in reg.ids(usable_only=False) else c) for c in feats},
        "seq_len": L, "k": k, "embargo": _embargo(cfg, k),
        "fold": {"train": [str(fold.train[0].date()), str(fold.train[-1].date())], "test": [str(fold.test[0].date()), str(fold.test[-1].date())],
                 "val_from": str(fold.val_from.date()) if fold.val_from is not None else None, "n_train_rows": int(len(fold.train))},
        "raw_train": tr_rows, "y_train": ds.y.loc[fold.train], "target_level": ds.prices["close"],
        "scaler_min": cap["scaler"].data_min_.copy(), "scaler_max": cap["scaler"].data_max_.copy(),
        "Xw": Xw, "yw": yw, "win_idx": [str(d.date()) for d in win_idx], "n_tr": cap["n_tr"], "n_va": cap["n_va"],
        "y_mu": m.y_mu, "y_sd": m.y_sd, "history": m.history, "alpha_epoch": rec.get("alpha_epoch", {}), "vae_pos": rec.get("vae_pos", {}), "init_alpha": rec["init_alpha"], "final_alpha": float(net.alpha.detach().ravel()[0]),
        "first_step": {k_: v for k_, v in fs.items() if k_ not in ("x", "states")}, "n_batches_epoch": len(batches),
        "batch": {"index_in_epoch": bi, "win": b_win, "x": xb, "y_z": yb.view(-1).tolist(), "n_nan": nn_[bi]},
        "events": tracer.events,
        "outputs": {"states": states.detach(), "vae_loss": float(vae_loss), "imp_loss": float(imp_loss), "mse": float(loss_states),
                    "recon": recon.detach(), "seasonal": seas_init.detach(), "imputed": imputed.detach()},
        "args": vars(m.args).copy(),
        "predict": {"date": str(t_dec.date()), "pred": float(pred.loc[t_dec]), "actual": float(ds.y.get(t_dec, np.nan)),
                    "level": float(ds.prices["close"].loc[t_dec]), "window": Xtw[i_dec], "window_dates": [str(d.date()) for d in cap["te_full"].index[i_dec:i_dec + L]],
                    "repeats": reps, "n_test": len(pred), "all_pred": pred, "y_all": ds.y},
        "net_state": {k_: v.detach().clone() for k_, v in net.state_dict().items()},
        "n_params": int(sum(p_.numel() for p_ in net.parameters())), "vae_all": vae_all,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--config", default="configs/repro_biva_orig.json")
    ap.add_argument("--out", default="docs/guide_build/walk/biva_capture.pkl")
    a = ap.parse_args()
    r = capture(a.config, a.epochs)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(r, f)
    print("저장:", out, "· 추적 줄", len(r["events"]), "· epoch", len(r["history"]))
