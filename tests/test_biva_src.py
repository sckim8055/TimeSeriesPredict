"""BIVA 원본 충실도 시험 (2026-11-06).

1. 사본(models/biva_src)의 모든 변경에 [TSP 수정] 표시가 있는가(표시 없는 수정 금지).
2. delta_calc 가 원본 글자 그대로인가(속도용 수정은 2026-11-12 되돌림).
3. 원본 모듈(수정 없는 파일)과 사본이 같은 가중치·같은 난수에서 같은 출력을 내는가 — 빈 칸 포함 입력.
   원본은 불러올 때 os.chdir('D:...') 와 .cuda() 를 하므로, 시험 안에서만 그 두 호출을 무력화해 불러온다(파일은 그대로).
4. 감싸개(models/biva.py)의 학습 규칙: 학습률 일정(lradj=3)·조기종료·use_brits 끄기·NaN 처리.
원본 위치가 없으면(공개 저장소 등) 2·4 만 돈다.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from biva_src_diff import check, orig_dir  # noqa: E402

OD = orig_dir()
needs_orig = pytest.mark.skipif(OD is None, reason="원본 BIVA 소스 없음")


def _args(ch=34, seq=6):
    return SimpleNamespace(seq_len=seq, pred_len=1, label_len=1, channels=ch, rnn_hid_size=16, vae_latent_size=4, infer_hid_size=128,
                           conv_kernal=1, moving_avg=6, batch_size=3, conv1d=True, RIN=False, combination=True, imputed_weight=0.05)


def _load_original():
    """수정 없는 원본 파일을 'model' 패키지로 불러온다(시험 안에서만 chdir·cuda 무력화)."""
    for k in [k for k in sys.modules if k == "model" or k.startswith("model.") or k in ("BIVA_RITS",)]:
        del sys.modules[k]
    pkg = types.ModuleType("model")
    pkg.__path__ = [str(OD)]
    sys.modules["model"] = pkg
    sys.path.insert(0, str(OD))  # 원본 BIVA_BRITS 의 'import BIVA_RITS' 용
    real_chdir, real_cuda, real_dwb = os.chdir, torch.Tensor.cuda, sys.dont_write_bytecode
    os.chdir = lambda *_a, **_k: None
    torch.Tensor.cuda = lambda self, *a, **k: self
    sys.dont_write_bytecode = True  # 원본 폴더에 __pycache__ 를 만들지 않는다(원본 수정 금지)
    try:
        return importlib.import_module("model.BIVA")
    finally:
        os.chdir, torch.Tensor.cuda, sys.dont_write_bytecode = real_chdir, real_cuda, real_dwb
        sys.path.remove(str(OD))


@needs_orig
def test_every_change_to_original_is_marked():
    assert check(OD) == []


@needs_orig
def test_delta_calc_is_the_original_code():
    """2026-11-12 사용자 결정으로 속도용 수정(#6)을 되돌렸다 — 사본의 delta_calc 는 원본 함수 글자 그대로."""
    import inspect
    from models.biva_src import BIVA_RITS
    src = (OD / "BIVA_RITS.py").read_text(encoding="utf-8")
    want = src[src.index("    def delta_calc(self, masks):"):src.index("    def forward(self, data, direct):")].rstrip()
    assert inspect.getsource(BIVA_RITS.Model.delta_calc).rstrip() == want and not hasattr(BIVA_RITS.Model, "delta_calc_orig")


@needs_orig
def test_copy_matches_original_forward_with_missing_values():
    orig = _load_original()
    from models.biva_src import BIVA as COPY
    torch.manual_seed(0)
    a = _args()
    net_o = orig.Model(a).float()
    torch.manual_seed(0)
    net_c = COPY.Model(a).float()
    assert net_o.state_dict().keys() == net_c.state_dict().keys()
    for k, v in net_o.state_dict().items():  # 같은 시드 → 같은 초기 가중치(생성 순서가 원본과 같다는 뜻)
        assert torch.equal(v, net_c.state_dict()[k]), k
    x = torch.rand(3, 6, 34)
    x[0, :3, 5] = float("nan")
    x[1, :, 7] = float("nan")
    x[2, 1::2, 30] = float("nan")
    real_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a_, **k: self  # 원본 RITS backward 의 .cuda()
    try:
        torch.manual_seed(5)
        out_o = net_o(x.clone())
    finally:
        torch.Tensor.cuda = real_cuda
    torch.manual_seed(5)
    out_c = net_c(x.clone())
    for i, (o, c) in enumerate(zip(out_o, out_c)):
        assert torch.equal(o, c), f"출력 {i} 다름"


@needs_orig
def test_copy_matches_original_training_steps():
    """학습까지 같은가: 원본과 사본에 같은 배치로 3번 (순전파 → 원본 손실식 → 역전파 → Adam) 을 돌려
    매 단계의 손실·모든 기울기·갱신된 가중치가 비트 단위로 같은지 본다. 빈 칸(NaN) 포함 입력."""
    orig = _load_original()
    from models.biva_src import BIVA as COPY
    a = _args()
    torch.manual_seed(0)
    net_o = orig.Model(a).float()
    torch.manual_seed(0)
    net_c = COPY.Model(a).float()
    opt_o = torch.optim.Adam(net_o.parameters(), lr=0.01)
    opt_c = torch.optim.Adam(net_c.parameters(), lr=0.01)
    mse = torch.nn.MSELoss()
    g = torch.Generator().manual_seed(1)
    real_cuda = torch.Tensor.cuda
    for step in range(3):
        x = torch.rand(3, 6, 34, generator=g)
        x[0, :3, 5] = float("nan")
        x[2, 1::2, 30] = float("nan")
        y = torch.rand(3, 1, 1, generator=g)  # 예측 states 모양 = (배치, pred_len, 1)
        losses = []
        for net, opt, patch in ((net_o, opt_o, True), (net_c, opt_c, False)):
            net.train()
            opt.zero_grad()
            if patch:
                torch.Tensor.cuda = lambda self, *a_, **k: self
            try:
                torch.manual_seed(100 + step)
                states, vae_loss, _, _, imputed_loss, _ = net(x.clone())
                loss = vae_loss + imputed_loss + mse(states, y)  # 원본 exp_main.train 의 손실식
                loss.backward()
            finally:
                torch.Tensor.cuda = real_cuda
            opt.step()
            losses.append(loss.detach())
        assert torch.equal(losses[0], losses[1]), f"{step} 단계 손실 다름"
        po, pc = dict(net_o.named_parameters()), dict(net_c.named_parameters())
        for k in po:
            if po[k].grad is not None or pc[k].grad is not None:
                assert torch.equal(po[k].grad, pc[k].grad), f"{step} 단계 기울기 다름: {k}"
            assert torch.equal(po[k], pc[k]), f"{step} 단계 가중치 다름: {k}"


def _toy(n=60, k=5, seq=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, seq, k)).astype(np.float32)
    y = X[:, -1, 0] * 0.5 + rng.normal(0, 0.01, n)
    return X, y


def test_wrapper_follows_original_training_rules():
    from models.base import resolve_params
    import models.biva as B
    X, y = _toy()
    p = resolve_params(B.PARAMS, {"train_epochs": 12, "patience": 50, "rnn_hid_size": 8, "batch_size": 8}, "M")
    torch.manual_seed(0)
    m = B.Model(p, n_features=X.shape[2], target_type="regression")
    m.fit(X[:45], y[:45], X[45:], y[45:])
    lrs = [h["lr"] for h in m.history]
    assert lrs[:10] == [0.01] * 10 and abs(lrs[10] - 0.001) < 1e-12, lrs  # lradj=3: epoch 10 이 끝난 뒤 ×0.1
    best = min(range(len(m.history)), key=lambda i: m.history[i]["vali_loss"])
    assert len(m.history) == 12 and best >= 0
    assert np.isfinite(m.predict(X[45:])).all()


def test_wrapper_without_brits_and_with_nan():
    from models.base import resolve_params
    import models.biva as B
    X, y = _toy(k=3)
    Xn = X.copy()
    Xn[:20, :, 1] = np.nan
    for use in (True, False):
        p = resolve_params(B.PARAMS, {"train_epochs": 3, "rnn_hid_size": 8, "batch_size": 8, "use_brits": use}, "M")
        m = B.Model(p, n_features=3, target_type="regression")
        Xi = Xn if use else np.nan_to_num(Xn, nan=0.5)  # 끄면 엔진이 평균으로 채워 넘김
        m.fit(Xi[:45], y[:45], Xi[45:], y[45:])
        assert np.isfinite(m.predict(Xi)).all()
        assert (type(m.net.BRITS).__name__ == "_NoBRITS") == (not use)


def test_unsupported_original_options_are_refused():
    from models.base import resolve_params
    import models.biva as B
    X, y = _toy()
    p = resolve_params(B.PARAMS, {"conv1d": False, "train_epochs": 1}, "M")
    with pytest.raises(ValueError, match="conv1d"):
        B.Model(p, n_features=5, target_type="regression").fit(X, y)


@needs_orig
def test_change_blocks_match_lock_and_taken_tools_are_original():
    """검토 11(낮음): '덩어리마다 표시 하나'만 보면, 표시가 붙은 덩어리 안에 표시 없는 수정이 숨을 수 있다 → 덩어리 전체의 내용 지문이 잠금 목록과 같아야 한다.
    tools_orig.py 의 두 개(adjust_learning_rate·EarlyStopping)는 원본 utils/tools.py 글자 그대로."""
    from biva_src_diff import hunk_fingerprints, read_lock, tools_problems
    assert hunk_fingerprints(OD) == read_lock() and len(read_lock()) == 13
    assert tools_problems(OD) == []
