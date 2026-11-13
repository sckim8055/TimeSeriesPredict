"""다변량 GRU. LSTM 과 같은 구조에서 셀만 GRU(게이트 2개, 파라미터 약 25% 적음)로 바꾼 것 — 원 노트북(노트북3)의 GRU(1→32) 에 대응. window 입력 [N, seq_len, k] → 다음 타겟 1개.

구조: GRU(k→hidden, layers) → 마지막 시점 은닉 → Linear(hidden→1).
학습: Adam, MSE(회귀) 또는 BCEWithLogits(분류), 검증 손실 최저 시점 가중치 사용(조기종료).
표본이 작을 때(월간 수백 개) hidden·layers 를 줄인다 — 기존 노트북의 32/1층은 일간에서는 작고, BIVA 점검에서 본 과대 모델은 실패한다.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from models.base import BaseModel

log = logging.getLogger("tsp.model.gru")

NAME = "GRU"
INPUT_SHAPE = "window"
TARGET_TYPES = ["regression", "classification"]
PARAMS = {
    "hidden": {"default_by_freq": {"D": 32, "W": 16, "M": 8}, "type": "int", "default": 32, "min": 4, "max": 512, "desc": "GRU 은닉 크기"},
    "layers": {"type": "int", "default": 1, "min": 1, "max": 4, "desc": "GRU 층 수"},
    "dropout": {"type": "float", "default": 0.1, "min": 0.0, "max": 0.9, "desc": "드롭아웃 비율 — 마지막 은닉 → Linear 앞에 항상, 층이 2개 이상이면 층 사이에도"},
    "epochs": {"default_by_freq": {"D": 100, "W": 150, "M": 200}, "type": "int", "default": 100, "min": 1, "max": 5000, "desc": "학습 반복 횟수(상한)"},
    "lr": {"type": "float", "default": 1e-3, "min": 1e-6, "max": 1.0, "desc": "학습률(Adam)"},
    "batch_size": {"default_by_freq": {"D": 64, "W": 32, "M": 16}, "type": "int", "default": 64, "min": 1, "max": 4096, "desc": "미니배치 크기"},
    "patience": {"type": "int", "default": 15, "min": 1, "max": 1000, "desc": "검증 손실 개선 없을 때 조기종료 에폭 수"},
    "weight_decay": {"type": "float", "default": 1e-4, "min": 0.0, "max": 1.0, "desc": "L2 규제(가중치 감쇠)"},
}


class _Net(nn.Module):
    def __init__(self, k, hidden, layers, dropout):
        super().__init__()
        self.gru = nn.GRU(k, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1, :])).squeeze(-1)


class Model(BaseModel):
    STOCHASTIC = True  # 가중치 초기값·미니배치 순서·드롭아웃이 시드에 따라 달라진다

    def fit(self, X, y, X_val=None, y_val=None):
        p = self.params
        dev = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
        self.net = _Net(self.n_features, p["hidden"], p["layers"], p["dropout"]).to(dev)
        # 타겟 스케일(회귀): 학습 구간 통계로만
        if self.target_type == "regression":
            self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y) + 1e-12)
            loss_fn = nn.MSELoss()
        else:
            self.y_mu, self.y_sd = 0.0, 1.0
            loss_fn = nn.BCEWithLogitsLoss()
        ys = (y - self.y_mu) / self.y_sd
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        yt = torch.tensor(ys, dtype=torch.float32, device=dev)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            Xv = torch.tensor(X_val, dtype=torch.float32, device=dev)
            yv = torch.tensor((y_val - self.y_mu) / self.y_sd, dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        best, best_state, bad = float("inf"), None, 0
        n = len(Xt)
        self.history = []  # epoch 별 기록(계산 따라가기·진단용). 학습 계산에는 영향 없음
        self.best_epoch = None
        log.debug("fit 시작: X%s · 검증 %s · 파라미터 %s", tuple(X.shape), None if not has_val else len(y_val), p)
        for ep in range(p["epochs"]):
            self.net.train()
            perm = torch.randperm(n, device=dev)
            tl, tn = 0.0, 0
            for i in range(0, n, p["batch_size"]):
                idx = perm[i : i + p["batch_size"]]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tl += float(loss.detach()) * len(idx); tn += len(idx)
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    vl = loss_fn(self.net(Xv), yv).item()
                log.debug("epoch %d | 마지막 배치 학습 손실 %.6g | 검증 %.6g | 최저 %.6g", ep + 1, loss.item(), vl, min(best, vl))
                self.history.append({"epoch": ep + 1, "train_loss": tl / max(tn, 1), "val_loss": vl})
                if vl < best - 1e-7:
                    best, bad = vl, 0
                    self.best_epoch = ep + 1
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= p["patience"]:
                        break
            else:
                self.history.append({"epoch": ep + 1, "train_loss": tl / max(tn, 1), "val_loss": None})
        log.debug("fit 끝: %d epoch · 최저 검증 손실 %s", ep + 1, best if has_val else "검증 없음")
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        # 진단: 몇 번째 반복의 가중치를 쓰는가(검증 손실 최저 시점). 검증이 없으면 마지막 반복.
        self.diag = {"early_stopping": bool(has_val), "epochs_run": int(ep + 1), "epoch_used": int(self.best_epoch) if self.best_epoch else int(ep + 1),
                     "epochs_max": int(p["epochs"]), "best_val_loss": float(best) if has_val and np.isfinite(best) else None}

    def predict(self, X):
        dev = next(self.net.parameters()).device
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float32, device=dev)).cpu().numpy()
        if self.target_type == "regression":
            return out * self.y_sd + self.y_mu
        return 1.0 / (1.0 + np.exp(-out))
