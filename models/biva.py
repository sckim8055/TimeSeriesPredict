"""BIVA — 연구실 원본 소스를 그대로 쓰는 플러그인. window 입력 [N, seq_len, k].

원본 위치: D:/test/SVU_SCT_FIN-main/Model/BIVA/model  →  사본: models/biva_src/ (고친 곳은 [TSP 수정 #n] 표시, PATCHES.md)
  - 신경망: biva_src/BIVA.py(Model) · BIVA_BRITS.py · BIVA_RITS.py · BIVA_LSTM_VAE.py  — 원본 그대로
  - 학습 규칙: 원본 exp_main.Exp_Main.train / vali 를 이 파일의 fit / _vali 가 **같은 순서**로 재현
      Adam(learning_rate) · MSELoss · 손실 = VAE_loss + imputed_loss + MSE(예측)   (가중치 없는 합)
      매 epoch: 학습(shuffle, drop_last) → 검증 손실(배치 평균) → EarlyStopping(patience) → 멈춤이면 끝 → adjust_learning_rate(epoch+1)
      끝나면 검증 손실이 가장 낮았던 가중치로 되돌림
  - 원본과 다른 점(플랫폼에서 돌기 위해 필요한 것만)
      a. 데이터: 엔진이 만든 창을 받는다(원본 data_loader 대신). 정답은 '입력 마지막 행'이 아니라 k 기간 뒤 값(누출 제거).
         입력 스케일러는 학습 구간으로만 맞춘다(원본은 전체 기간으로 다시 맞춤 = 누출). 기본은 원본과 같은 0~1(MinMax).
      b. 체크포인트를 파일 대신 메모리에 보관(EarlyStopping.save_checkpoint 만 바꿈).
      c. 원본 학습 루프의 epoch 마다 test 손실 계산은 하지 않는다(출력용이었고, 테스트 구간을 학습 중에 보지 않기 위해).
      d. use_brits=아니오 이면 BRITS 대신 '입력을 그대로 넘기는' 모듈을 끼운다(빈 칸은 엔진이 학습 구간 평균으로 채움).
      e. target_scale: 대상 값을 학습 구간 평균·표준편차로 맞춰 학습하고 되돌린다(기본). 원본은 스케일 없이 지수 수준값을
         맞혔는데, 플랫폼의 대상(k 기간 변화율, 크기 약 0.003)은 그대로 두면 예측 MSE 가 VAE·보간 손실보다 수만 배 작아(F11 학습 창 1.8e-5 대 0.81)
         공유 가중치(BRITS·VAE)의 학습과 검증 손실에 예측 오차가 거의 반영되지 않는다. 'none' 으로 원본처럼 둘 수 있다.
"""
from __future__ import annotations

import copy
import logging
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from models.base import BaseModel
from models.biva_src import BIVA as BIVA_SRC
from models.biva_src.tools_orig import EarlyStopping, adjust_learning_rate

log = logging.getLogger("tsp.model.biva")

NAME = "BIVA"
INPUT_SHAPE = "window"
TARGET_TYPES = ["regression"]
SEQ_LEN_DEFAULT = {"M": 6}  # 원본 노트북 len_seq_list = [6] (월간). 일·주간은 엔진 기본
LRADJ = ["3", "none", "1", "2", "4", "5", "6", "dtype1", "dtype2", "type-e"]
# 기본값 = 원본 노트북 (BASE)BIVA_SEOUL_After_4_Local_1027.ipynb 의 args. 그룹(group)은 화면 배치·그림 표시용.
PARAMS = {
    "use_brits": {"type": "bool", "default": True, "group": "Stage 1 BRITS",
                  "desc": "결측 보간 BRITS 사용. 원본은 항상 사용(빈 칸을 양방향 RNN 으로 채우고 보간 손실을 학습에 더함). 끄면 빈 칸은 학습 구간 평균으로 채워 BRITS 없이 학습"},
    "rnn_hid_size": {"type": "int", "default": 128, "min": 1, "max": 1024, "group": "Stage 1 BRITS",
                     "desc": "BRITS 안 RNN(LSTMCell) 은닉 크기. 시점마다 지난 정보를 담는 기억의 크기. 원본 128"},
    "imputed_weight": {"type": "float", "default": 0.05, "min": 0.0, "max": 10.0, "group": "Stage 1 BRITS",
                       "desc": "보간 손실에 곱하는 계수. 클수록 '관측된 칸을 잘 복원하는 것'을 더 중시. 원본 0.05"},
    "moving_avg": {"type": "str", "default": "6", "group": "Stage 2 분해",
                   "desc": "추세/계절 분해 이동평균 창(개월 수). 정수 하나(예 6) 또는 쉼표로 여러 개(예 3,6,12 → 원본 series_decomp_multi 로 여러 창을 학습 가중 평균). 원본 6"},
    "vae_latent_size": {"type": "int", "default": 4, "min": 2, "max": 256, "group": "Stage 3 LSTM-VAE",
                        "desc": "계절 성분을 압축하는 잠재 변수 z 의 차원. 작을수록 강하게 요약. 2 미만이면 원본 Conv1d 채널이 0 이 되어 오류. 원본 4"},
    "conv1d": {"type": "bool", "default": True, "group": "추세 경로",
               "desc": "추세·계절 출력에 Conv1d 3단 사용. 원본은 항상 예 — '아니오'는 원본 forward 가 Conv1d 를 항상 호출해 오류(그래서 막음)"},
    "conv_kernal": {"type": "int", "default": 1, "min": 1, "max": 1, "group": "추세 경로",
                    "desc": "Conv1d 커널 크기. 원본은 패딩이 없어 1 이 아니면 길이가 줄어 뒤의 Linear(seq_len) 와 맞지 않아 오류 → 1 만 가능. 원본 1"},
    "combination": {"type": "bool", "default": True, "group": "Stage 4 합치기·출력",
                    "desc": "예측 = α·계절 + (1−α)·추세 에서 α 를 학습. 아니오면 단순 합(계절 + 추세). 원본 예"},
    "RIN": {"type": "bool", "default": False, "group": "입력",
            "desc": "창마다 평균·표준편차를 빼고 마지막 입력 열의 통계로 되돌리는 정규화(RevIN). 빈 칸(NaN)이 있으면 통계가 NaN 이 되므로 BRITS 와 함께 쓰기 어렵다. 원본 아니오"},
    "learning_rate": {"type": "float", "default": 0.01, "min": 1e-7, "max": 1.0, "group": "학습",
                      "desc": "Adam 학습률(한 번에 가중치를 얼마나 크게 고치나). 원본 0.01"},
    "lradj": {"type": "str", "default": "3", "choices": LRADJ, "group": "학습",
              "desc": "학습률 일정(원본 adjust_learning_rate). 3: 10 epoch 학습 뒤(11번째부터) ×0.1(원본) · 4/5/6: 15/25/5 epoch 뒤부터 ×0.1 · 1: 매 epoch ×0.98 · dtype1: 매 epoch ×0.5 · 2·dtype2: 정해진 표 · type-e: 원본 실험용 · none: 고정(플랫폼 추가)"},
    "batch_size": {"type": "int", "default": 3, "min": 1, "max": 4096, "group": "학습",
                   "desc": "한 번에 학습하는 창 수. 남는 창은 원본처럼 버리되(drop_last) 섞기 때문에 epoch 마다 다른 창이 빠진다. 원본 3"},
    "train_epochs": {"type": "int", "default": 500, "min": 1, "max": 100000, "group": "학습",
                     "desc": "학습 반복 횟수 상한. 원본 500"},
    "patience": {"type": "int", "default": 500, "min": 1, "max": 100000, "group": "학습",
                 "desc": "검증 손실이 이 횟수만큼 연속으로 나아지지 않으면 멈춤. 원본 500(= 사실상 끝까지 학습 후 최저 검증 손실 시점 선택)"},
    "input_scaler": {"type": "str", "default": "minmax", "choices": ["minmax", "standard"], "group": "입력",
                     "desc": "입력 스케일. minmax: 학습 구간 최소0·최대1(원본 방식 — BRITS 보간값이 sigmoid 0~1 출력이라 맞춰야 함). standard: 평균0·표준편차1"},
    "target_scale": {"type": "str", "default": "standard", "choices": ["standard", "none"], "group": "Stage 4 합치기·출력",
                     "desc": "대상 값 스케일. standard: 학습 구간 평균·표준편차로 맞춰 학습 후 되돌림(기본). none: 원본처럼 그대로 — 변화율 대상은 값이 매우 작아 예측 손실이 다른 손실에 묻힘"},
    "init_checkpoint": {"type": "str", "default": "", "group": "학습",
                        "desc": "시작 가중치 파일(.pth, 원본 last_chkpt). 비우면 무작위 초기화. 모양(변수 수·은닉 크기)이 같아야 함"},
}
PARAM_GROUPS = ["입력", "Stage 1 BRITS", "Stage 2 분해", "추세 경로", "Stage 3 LSTM-VAE", "Stage 4 합치기·출력", "학습"]


class _NoBRITS(nn.Module):
    """use_brits=아니오: 입력을 그대로 넘기고 보간 손실 0. (원본 BRITS 와 같은 반환 모양)"""

    def forward(self, x):
        return {"imputations": x, "loss": torch.zeros((), device=x.device)}


class _MemEarlyStopping(EarlyStopping):
    """원본 EarlyStopping 그대로, 체크포인트만 파일 대신 메모리에 저장."""

    best_state = None

    def save_checkpoint(self, val_loss, model, path, model_id):
        if self.verbose:
            log.debug("검증 손실 감소 (%.6f --> %.6f) 가중치 보관", self.val_loss_min, val_loss)
        self.best_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


def _parse_ma(s: str):
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    vals = [int(p) for p in parts]
    if not vals or any(v < 1 for v in vals):
        raise ValueError(f"moving_avg 는 1 이상의 정수(또는 쉼표 목록)여야 합니다: {s!r}")
    return vals[0] if len(vals) == 1 else vals


class Model(BaseModel):
    ACCEPTS_NAN = True
    STOCHASTIC = True  # 가중치 초기값·미니배치 순서·VAE 표본추출(예측 때도)이 시드에 따라 달라진다

    @classmethod
    def wants_nan(cls, params) -> bool:
        """엔진이 빈 칸을 NaN 그대로 넘길지(=BRITS 가 채움)."""
        return bool(params.get("use_brits", True))

    def _args(self, seq_len: int) -> SimpleNamespace:
        p = self.params
        if not p["conv1d"]:
            raise ValueError("BIVA 원본은 conv1d=아니오 를 지원하지 않습니다(원본 forward 가 Conv1d_Trend_1 을 항상 호출 → 오류)")
        if p["conv_kernal"] != 1:
            raise ValueError("BIVA 원본은 conv_kernal=1 만 동작합니다(패딩 없는 Conv1d → 길이가 줄어 Linear(seq_len) 와 불일치)")
        return SimpleNamespace(
            seq_len=seq_len, pred_len=1, label_len=1, channels=self.n_features,
            rnn_hid_size=p["rnn_hid_size"], vae_latent_size=p["vae_latent_size"], infer_hid_size=128,  # infer_hid_size 는 원본에서 저장만 하고 쓰지 않음
            conv_kernal=p["conv_kernal"], moving_avg=_parse_ma(p["moving_avg"]), batch_size=p["batch_size"],
            conv1d=p["conv1d"], RIN=p["RIN"], combination=p["combination"], imputed_weight=p["imputed_weight"],
            lradj=p["lradj"], learning_rate=p["learning_rate"],
        )

    def _build(self, seq_len: int):
        self.args = self._args(seq_len)
        net = BIVA_SRC.Model(self.args).float()  # 원본 _build_model 과 같음
        if self.params.get("init_checkpoint"):
            net.load_state_dict(torch.load(self.params["init_checkpoint"], map_location="cpu"))
            log.info("시작 가중치 불러옴: %s", self.params["init_checkpoint"])
        if not self.params["use_brits"]:
            net.BRITS = _NoBRITS()  # 생성은 원본과 같이 한 뒤 교체 → 나머지 층의 초기값은 BRITS 사용 때와 같다
        return net.to(self.device)

    def _tensor(self, X, y=None):
        Xt = torch.tensor(np.asarray(X, dtype=np.float32))
        if y is None:
            return Xt
        yt = torch.tensor(((np.asarray(y, dtype=np.float64) - self.y_mu) / self.y_sd).astype(np.float32)).view(-1, 1, 1)
        return Xt, yt

    def _run(self, bx):
        return self.net(bx.float().to(self.device))

    def _vali(self, loader, criterion) -> float:
        """원본 Exp_Main.vali 와 같음: eval, 배치마다 (VAE + 보간 + 예측 MSE) → 배치 평균."""
        total = []
        self.net.eval()
        with torch.no_grad():
            for bx, by in loader:
                states, VAE_loss, _, _, imputed_loss, _ = self._run(bx)
                loss = VAE_loss.detach().cpu() + imputed_loss.detach().cpu() + criterion(states.detach().cpu(), by)
                total.append(loss.item())
        return float(np.average(total)) if total else float("nan")

    def fit(self, X, y, X_val=None, y_val=None):
        p = self.params
        N, L, K = X.shape
        if p["target_scale"] == "standard":
            self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y) or 1.0)
        else:
            self.y_mu, self.y_sd = 0.0, 1.0
        self.net = self._build(L)
        n_par = sum(t.numel() for t in self.net.parameters())
        nan_x = int(np.isnan(X).sum())
        log.info("BIVA fit 시작: 창 %d × 길이 %d × 변수 %d (빈 칸 %d) · 검증 창 %s · 파라미터 %d · 장치 %s · use_brits=%s · 대상 스케일 μ=%.6g σ=%.6g",
                 N, L, K, nan_x, None if X_val is None else len(X_val), n_par, self.device, p["use_brits"], self.y_mu, self.y_sd)
        if N < p["batch_size"]:
            raise ValueError(f"학습 창({N})이 batch_size({p['batch_size']})보다 적어 drop_last 로 학습할 배치가 없습니다")
        tr = DataLoader(TensorDataset(*self._tensor(X, y)), batch_size=p["batch_size"], shuffle=True, drop_last=True)
        va = None
        if X_val is not None and len(X_val) >= p["batch_size"]:
            # 원본 data_factory: flag='val' 은 else 분기 → shuffle=True, drop_last=True
            va = DataLoader(TensorDataset(*self._tensor(X_val, y_val)), batch_size=p["batch_size"], shuffle=True, drop_last=True)
        else:
            log.warning("검증 창이 없거나 batch_size 보다 적음 → 학습 손실로 조기종료·최적 시점을 고름(원본에는 없는 경우)")
        opt = optim.Adam(self.net.parameters(), lr=p["learning_rate"])  # 원본 _select_optimizer
        criterion = nn.MSELoss()  # 원본 _select_criterion
        early = _MemEarlyStopping(patience=p["patience"], verbose=True)
        self.history = []
        for epoch in range(p["train_epochs"]):
            t0 = time.time()
            self.net.train()
            train_loss = []
            for bx, by in tr:
                opt.zero_grad()
                states, VAE_loss, _, _, imputed_loss, _ = self._run(bx)
                loss_states = criterion(states, by.to(self.device))
                loss = VAE_loss + imputed_loss + loss_states  # 원본: 가중치 없는 합
                train_loss.append(loss.item())
                loss.backward()
                opt.step()
            train_loss = float(np.average(train_loss))
            vali_loss = self._vali(va, criterion) if va is not None else train_loss
            lr_now = opt.param_groups[0]["lr"]
            self.history.append({"epoch": epoch + 1, "train_loss": train_loss, "vali_loss": vali_loss, "lr": lr_now, "sec": round(time.time() - t0, 3)})
            log.debug("epoch %d | train %.7f | vali %.7f | lr %.3g | %.2fs", epoch + 1, train_loss, vali_loss, lr_now, time.time() - t0)
            if not np.isfinite(train_loss):
                raise FloatingPointError(f"BIVA 손실이 {train_loss} (epoch {epoch + 1}) — 입력 스케일·학습률 확인")
            early(vali_loss, self.net, "", "")
            if early.early_stop:
                log.info("조기종료 epoch %d (patience %d)", epoch + 1, p["patience"])
                break
            if p["lradj"] != "none":
                adjust_learning_rate(opt, epoch + 1, self.args)
        if early.best_state is not None:
            self.net.load_state_dict(early.best_state)
        best = min(self.history, key=lambda h: h["vali_loss"]) if self.history else None
        self.diag = {"early_stopping": True, "epochs_run": len(self.history), "epoch_used": int(best["epoch"]) if best else len(self.history),
                     "epochs_max": int(p["train_epochs"]), "best_val_loss": float(best["vali_loss"]) if best else None}
        log.info("BIVA fit 끝: %d epoch · 최저 검증 손실 %.7f (epoch %s) · 총 %.1fs", len(self.history),
                 early.val_loss_min, best and best["epoch"], sum(h["sec"] for h in self.history))

    def predict(self, X):
        """eval 모드로 전체 창 예측(원본 test 의 drop_last 는 마지막 창을 버리는 문제라 쓰지 않음).
        원본 그대로 VAE 표본추출(z = μ + ε·σ)이 예측 때도 일어나므로 같은 시드여야 같은 값."""
        self.net.eval()
        out = []
        Xt = self._tensor(X)
        with torch.no_grad():
            for i in range(0, len(Xt), self.params["batch_size"]):
                states = self._run(Xt[i:i + self.params["batch_size"]])[0]
                out.append(states[:, -1, 0].detach().cpu().numpy())
        pred = np.concatenate(out).astype(np.float64) * self.y_sd + self.y_mu if out else np.zeros(0)
        log.debug("BIVA predict: 창 %d → 예측 평균 %.6g 표준편차 %.6g", len(X), float(np.mean(pred)) if len(pred) else float("nan"),
                  float(np.std(pred)) if len(pred) else float("nan"))
        return pred
