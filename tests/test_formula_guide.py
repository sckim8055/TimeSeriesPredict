"""수식 읽는 법(docs/guide/35_수식_읽는_법.md)의 '작은 예로 직접 풀기' 숫자가 실제 코드·라이브러리 계산과 같은지 확인한다.
문서의 예시 숫자를 바꾸면 이 시험도 같이 고칠 것 — 교재 숫자가 틀리면 안 된다."""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "guide" / "35_수식_읽는_법.md"


def r(x, n=3):
    return round(float(x), n)


def test_doc_has_every_section_marker():
    s = DOC.read_text(encoding="utf-8")
    for key in ["symbols", "folds", "standardize", "minmax", "target", "decision", "ridge_sum", "ridge_fit", "logistic", "lgbm", "arima", "kalman", "sarima",
                "compound", "vecm", "nn_train", "lstm_cell", "gru_cell", "rnn_head", "brits", "moving_avg", "conv1d", "vae", "biva_combine", "noise",
                "rule_ma", "rule_momentum", "rule_rsi", "rule_quantile", "metrics", "dm"]:
        assert f"<!--f:{key}-->" in s, key


def test_standardize_and_minmax():
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    sc = StandardScaler().fit(np.array([[2.0], [4.0], [6.0]]))
    assert r(sc.mean_[0]) == 4 and r(sc.scale_[0]) == 1.633 and r(sc.transform([[7.0]])[0, 0]) == 1.837
    mm = MinMaxScaler().fit(np.array([[2.0], [4.0], [6.0]]))
    assert mm.transform([[5.0]])[0, 0] == 0.75 and mm.transform([[7.0]])[0, 0] == 1.25


def test_target_compound_cost():
    assert r(103 / 100 - 1) == 0.03 and r(math.log(1.03), 5) == 0.02956
    rr = np.array([0.01, -0.02, 0.03])
    assert r(np.prod(1 + rr) - 1, 6) == 0.019494 and r(np.log1p(rr).sum(), 5) == 0.01931
    from engine.backtest import CostModel
    assert round(1_000_000 * CostModel().one_way) == 650


def test_ridge_examples():
    from sklearn.linear_model import Ridge
    assert r(0.05 + 0.1 * 0.95 - 0.05 * 0.8) == 0.105
    z = np.array([[-1.0], [0.0], [1.0]]); y = np.array([1.0, 2.0, 6.0])
    m = Ridge(alpha=1.0).fit(z, y)
    assert r(m.coef_[0]) == 1.667 and r(m.intercept_) == 3.0
    assert r(Ridge(alpha=1e-12).fit(z, y).coef_[0]) == 2.5 and r(Ridge(alpha=100).fit(z, y).coef_[0]) == 0.049
    score = lambda w: float(((y - (3 + w * z[:, 0])) ** 2).sum() + w * w)
    assert r(score(1.5), 2) == 5.75 and r(score(5 / 3)) == 5.667 and r(score(2.0), 2) == 6.0
    sig = lambda a: 1 / (1 + math.exp(-a))
    assert sig(0) == 0.5 and r(sig(2), 4) == 0.8808 and r(sig(-2), 4) == 0.1192
    assert r(-math.log(0.88)) == 0.128 and r(-math.log(0.12)) == 2.120


def test_lgbm_example_matches_lightgbm():
    import lightgbm as lgb
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]]); y = np.array([1.0, 2.0, 6.0, 7.0])
    m = lgb.LGBMRegressor(n_estimators=1, learning_rate=0.1, num_leaves=2, min_child_samples=1, min_child_weight=0, reg_lambda=1.0,
                          min_data_in_bin=1, verbose=-1).fit(X, y)
    leaves = sorted(round(v, 4) for v in m.predict(X))
    assert leaves == [3.8333, 3.8333, 4.1667, 4.1667], leaves
    assert r(m.predict(np.array([[0.3]]))[0], 3) == 4.167
    g = 4 - y
    gain = lambda L: g[L].sum() ** 2 / (L.sum() + 1) + g[~L].sum() ** 2 / ((~L).sum() + 1) - g.sum() ** 2 / (len(g) + 1)
    assert r(gain(X[:, 0] <= 0), 2) == 16.67 and r(gain(X[:, 0] <= -1), 2) == 6.75


def test_arma_sarima_examples():
    c, phi, th = 0.001, 0.5, 0.3
    r1 = c + phi * 0.02 + th * 0.005; r2 = c + phi * r1; r3 = c + phi * r2
    assert (r(r1, 4), r(r2, 5), r(r3, 6), r(c / (1 - phi), 3)) == (0.0125, 0.00725, 0.004625, 0.002)
    assert (r(c + phi * 0.02), r(c + phi * 0.011, 4)) == (0.011, 0.0065)
    assert list(np.convolve([1, -0.5], [1, 0, 0, 0, -1])) == [1, -0.5, 0, 0, -1, 0.5]
    assert r(0.5 * 0.02 + 0.01 - 0.5 * 0.03) == 0.005


def test_vecm_example():
    beta = np.array([1.0, -2.0]); alpha = np.array([-0.25, 0.1]); y = np.array([104.0, 50.0])
    e = beta @ y; nxt = y + alpha * e
    assert e == 4 and list(nxt) == [103.0, 50.4] and r(beta @ nxt, 1) == 2.2


def test_nn_examples():
    sig = lambda a: 1 / (1 + math.exp(-a))
    i, f, g, o = sig(0), sig(1.5 * 1 + 0.4 * 0 + 0.5), math.tanh(0.5), sig(1)
    c = f * 0.5 + i * g; h = o * math.tanh(c)
    assert (r(f, 4), r(g, 4), r(o, 4), r(c, 4), r(h, 4)) == (0.8808, 0.4621, 0.7311, 0.6715, 0.4284)
    rr, zz = sig(0), sig(1); n = math.tanh(0.6 + rr * 0.8); hh = (1 - zz) * n + zz * 0.2
    assert (r(n, 4), r(hh, 4)) == (0.7616, 0.351)
    sc = 0.1 + 0.5 * 0.43 + 1.0 * -0.2
    assert r(sc) == 0.115 and r(sc * 0.02 + 0.001, 4) == 0.0033
    w = 1.0; grad = 2 * (2 * w - 3) * 2; w2 = w - 0.1 * grad
    assert grad == -4 and r(w2) == 1.4 and r((2 * w2 - 3) ** 2) == 0.04


def test_biva_examples_match_original_modules():
    import torch
    from models.biva_src.BIVA import moving_avg
    x = torch.tensor([1, 2, 3, 4, 5, 9], dtype=torch.float32).view(1, 6, 1)
    tr = moving_avg(6, stride=1)(x).view(-1).numpy()
    assert [round(float(v), 3) for v in tr] == [2.0, 2.667, 4.0, 5.333, 6.5, 7.5]
    assert r(abs(0.5 * 0.4 + 0.5 * 0.6), 2) == 0.5 and r(math.exp(-0.7), 4) == 0.4966
    s = math.exp(0.5 * 0.5)
    assert (r(s), r(0.4 * s, 4), r(-0.5 * (1 + 0.5 - 0 - math.exp(0.5)), 4)) == (1.284, 0.5136, 0.0744)
    assert r(0.1 + 0.5 * 0.2 - 1 * 0.1 + 2 * 0.3, 2) == 0.7 and r(0.8 * 0.3 + 0.2 * -0.1, 2) == 0.22 and r(0.22 * 0.02 + 0.001, 4) == 0.0054
    assert r(np.std([0.20, 0.22, 0.24]), 4) == 0.0163


def test_rule_examples():
    from engine.features import rsi
    c = pd.Series([10, 11, 12, 13, 14.0])
    assert c.iloc[-3:].mean() == 13 and c.mean() == 12
    v = rsi(pd.Series([10, 12, 11, 14.0]), 3).values
    assert np.isnan(v[1]) and r(v[2], 2) == 80.0 and r(v[3], 2) == 89.47
    assert r(np.quantile([1, 3, 2, 5, 4], 0.9), 2) == 4.6
    assert pd.Series([1, 3, 2, 5, 4, 0.0]).shift(1).rolling(5, min_periods=2).quantile(0.9).iloc[-1] == pytest.approx(4.6)


def test_metrics_and_dm():
    from engine.backtest import performance
    p = performance(pd.Series([0.01, -0.02, 0.03]))
    assert r(p["total_return"], 6) == 0.019494 and r(p["sharpe"], 2) == 4.21 and r(p["mdd"], 4) == -0.02
    assert r(1.019494 ** 0.5 - 1, 4) == 0.0097
    from scipy.stats import norm
    d = np.array([1, -1, 2.0]) ** 2 - np.array([2, -2, 1.0]) ** 2
    dm = d.mean() / math.sqrt(((d - d.mean()) ** 2).mean() / 3)
    assert list(d) == [-3, -3, 3] and r(dm) == -0.612 and r(2 * (1 - norm.cdf(abs(dm))), 2) == 0.54


def test_every_figure_placeholder_has_a_figure_and_svgs_parse():
    import re
    import xml.dom.minidom

    from tools.formula_figs import FIGS
    keys = re.findall(r"\{\{FIG:([a-z_]+)\}\}", DOC.read_text(encoding="utf-8"))
    assert keys and not [k for k in keys if k not in FIGS]
    for k, fn in FIGS.items():
        xml.dom.minidom.parseString(fn().render())


def test_walkforward_example():
    import engine.calendar as cal
    days = pd.bdate_range("2020-01-01", periods=11)
    pos = lambda ix: [days.get_loc(x) + 1 for x in ix]
    f10 = cal.walkforward_folds(days[:10], train_len=4, test_len=2, k=0, embargo=1)
    assert [(pos(f.train), pos(f.test)) for f in f10] == [([1, 2, 3], [5, 6]), ([3, 4, 5], [7, 8]), ([5, 6, 7], [9, 10])]
    f11 = cal.walkforward_folds(days, train_len=4, test_len=2, k=0, embargo=1)
    assert (pos(f11[-1].train), pos(f11[-1].test)) == ([7, 8, 9], [11])


# ------------------------------------------------------------------ 움직이는 그림(tools/formula_anims.py): 그림 속 숫자 = 실제 코드
def test_anim_svgs_have_valid_smil_and_static_final_state():
    """모든 움직이는 그림: keyTimes 가 0 에서 시작해 늘어나고 values 개수와 같다(SMIL 규칙). 같은 그림 안의 dur 는 하나."""
    import re
    import xml.dom.minidom

    from tools.formula_figs import FIGS
    anims = {k: fn for k, fn in FIGS.items() if k.endswith("_anim")}
    assert len(anims) >= 20
    for k, fn in anims.items():
        svg = fn().render()
        xml.dom.minidom.parseString(svg)
        durs = set(re.findall(r'dur="([\d.]+)s"', svg))
        assert len(durs) == 1 or k in ("lstm_anim",), (k, durs)
        for kt, vals in re.findall(r'keyTimes="([^"]+)"[^>]*values="([^"]+)"', svg):
            ks = [float(x) for x in kt.split(";")]
            assert ks[0] == 0 and all(b >= a for a, b in zip(ks, ks[1:])) and ks[-1] <= 1, (k, kt)
            assert len(ks) == len(vals.split(";")), (k, kt, vals)


def test_anim_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge

    from tools.formula_anims import RIDGE_FIT, compute_ridge_alpha, compute_ridge_sum
    assert r(compute_ridge_sum()["yhat"], 6) == 0.105
    z = np.array(RIDGE_FIT["z"]).reshape(-1, 1); y = np.array(RIDGE_FIT["y"])
    for row in compute_ridge_alpha():
        m = Ridge(alpha=row["alpha"]).fit(z, y)
        assert abs(m.coef_[0] - row["w"]) < 1e-9 and abs(m.intercept_ - row["b"]) < 1e-9
    assert [r(q["w"], 3) for q in compute_ridge_alpha()] == [2.5, 2.0, 1.667, 1.0, 0.417, 0.049]


def test_anim_lgbm_matches_lightgbm():
    import lightgbm as lgb

    from tools.formula_anims import LGBM, compute_lgbm
    c = compute_lgbm()
    X = np.array(LGBM["z"]).reshape(-1, 1)
    m = lgb.LGBMRegressor(n_estimators=1, learning_rate=LGBM["eta"], reg_lambda=LGBM["lam"], num_leaves=2, min_child_samples=1, min_child_weight=0,
                          min_split_gain=0, subsample=1.0, colsample_bytree=1.0, min_data_in_bin=1, verbose=-1).fit(X, np.array(LGBM["y"]))
    assert np.allclose(m.predict(X), c["new"]) and abs(m.predict(np.array([[LGBM["new_z"]]]))[0] - c["pred_new"]) < 1e-9
    tree = m.booster_.dump_model()["tree_info"][0]["tree_structure"]
    assert abs(tree["split_gain"] - c["gains"]["z ≤ 0"]) < 1e-4 and r(c["gains"]["z ≤ −1"], 2) == 6.75


def test_anim_kalman_and_sarima_follow_recursions():
    from tools.formula_anims import KALMAN, SARIMA, compute_kalman, compute_sarima
    k = compute_kalman(); c, phi, th = KALMAN["params"]; rr = KALMAN["r"]
    assert abs(k["one"][0] - c / (1 - phi)) < 1e-12  # 첫 예측 = 장기 평균(정상 초기값)
    # 칼만 필터의 한 기간 앞 예측과 '펼친 식'(c + φ·지난 실제 + θ·지난 충격)의 차이: 처음엔 있고(필터가 충격을 아직 확신 못 함) 점점 0 으로(그림 설명 문장의 근거)
    gaps = [abs(k["one"][t] - (c + phi * rr[t - 1] + th * k["err"][t - 1])) for t in range(1, len(rr))]
    assert gaps[0] > 1e-4 and all(b < a for a, b in zip(gaps, gaps[1:])) and gaps[-1] < 1e-5
    assert abs(k["forecast"][0] - (c + phi * rr[-1] + th * k["err"][-1])) < 1e-6 and abs(k["forecast"][1] - (c + phi * k["forecast"][0])) < 1e-12
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    s = compute_sarima()
    assert r(s["pred"], 6) == 0.005 and s["poly"] == [1, -0.5, 0, 0, -1, 0.5]
    rng = np.random.default_rng(0)
    series = np.r_[rng.normal(0, 0.01, 30), [SARIMA["r"][i] for i in range(-5, 1)]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(series, order=(1, 0, 0), seasonal_order=(0, 1, 0, 4)).filter(np.array([0.5, 1e-4]))
    assert abs(float(res.forecast(1)[0]) - s["pred"]) < 1e-7  # 계절 차분 상태의 근사 확산 초기값 때문에 1e-8 수준 차이


def test_anim_roots_vecm_numbers():
    from tools.formula_anims import compute_roots, compute_vecm
    rt = compute_roots()
    assert rt["ar"][0.5]["irf"] == [0.5 ** j for j in range(6)] and r(rt["ar"][1.2]["root"], 3) == 0.833
    assert rt["ma"][2.0]["inv"] == [(-2.0) ** j for j in range(6)] and rt["ma"][0.5]["root"] == 2 and rt["ma"][2.0]["root"] == 0.5
    v = compute_vecm()
    assert v[1]["y"] == [103.0, 50.4] and r(v[1]["e"], 6) == 2.2 and all(r(v[i + 1]["e"] / v[i]["e"], 6) == 0.55 for i in range(len(v) - 1))


def test_anim_gru_and_gd_match_torch():
    import torch

    from tools.formula_anims import GD, GRU, compute_gd, compute_gru
    cell = torch.nn.GRUCell(1, 1)
    with torch.no_grad():  # PyTorch 순서: r, z, n. 입력 x = 1, h = 0.2
        cell.weight_ih[:] = torch.tensor([[GRU["pre_r"]], [GRU["pre_z"]], [GRU["n_in"]]]); cell.bias_ih.zero_()
        cell.weight_hh[:] = torch.tensor([[0.0], [0.0], [GRU["n_hid"] / GRU["h"]]]); cell.bias_hh.zero_()
        h = cell(torch.tensor([[GRU["x"]]]), torch.tensor([[GRU["h"]]]))
    g = compute_gru()
    assert abs(float(h) - g["h"]) < 1e-6 and r(g["h"], 4) == 0.351
    w = torch.tensor(GD["w0"], requires_grad=True)
    opt = torch.optim.SGD([w], lr=GD["lr"])
    for q in compute_gd()[:-1]:
        assert abs(float(w) - q["w"]) < 1e-6
        opt.zero_grad(); loss = (w * GD["x"] - GD["y"]) ** 2; loss.backward()
        assert abs(float(w.grad) - q["grad"]) < 1e-5
        opt.step()


def test_anim_lstm_training_record_is_real_early_stopping():
    from tools.formula_anims import compute_nn_train
    t = compute_nn_train()
    h = t["history"]
    assert 1 <= t["best_epoch"] <= len(h) < t["epochs"]
    assert len(h) - t["best_epoch"] == t["patience"]  # 마지막 개선 뒤 patience 번 더 보고 멈춤
    assert min(q["val_loss"] for q in h) == h[t["best_epoch"] - 1]["val_loss"]


def test_anim_biva_parts_match_original_modules():
    import torch

    from tools.formula_anims import BRITS, compute_brits, compute_conv1d, compute_moving_avg, compute_vae
    b = compute_brits()
    for t, row in enumerate(BRITS["x"]):
        for d, v in enumerate(row):
            if v == v:
                assert abs(b["final"][t][d] - v) < 1e-6  # 관측 칸은 그대로
            else:
                assert abs(b["final"][t][d] - (b["fwd"][t][d] + b["bwd"][t][d]) / 2) < 1e-6  # 빈 칸 = 두 방향 평균
    from types import SimpleNamespace

    from models.biva_src import BIVA_RITS
    m = BIVA_RITS.Model(SimpleNamespace(seq_len=6, channels=2, rnn_hid_size=4, imputed_weight=1.0))
    mask = torch.tensor([b["mask"]])
    assert torch.equal(m.delta_calc(mask), m.delta_calc_orig(mask)) and b["delta_f"] == m.delta_calc_orig(mask)[0].tolist()
    mv = compute_moving_avg()
    assert [r(v, 3) for v in mv["trend"]] == [2, 2.667, 4, 5.333, 6.5, 7.5] and [r(v, 3) for v in mv["seasonal"]] == [-1, -0.667, -1, -1.333, -1.5, 1.5]
    cv = compute_conv1d()
    assert [r(v, 6) for v in cv["a"]] == [0.7, 0.5, 0.9] and r(cv["out"], 6) == 0.7
    va = compute_vae()
    assert va["mu"] == 0 and r(va["z"][0], 4) == 0.5136 and r(va["kld"], 4) == 0.0744 and va["sigma"] >= 1


def test_anim_rules_use_model_positions():
    import pandas as pd

    from tools.formula_anims import RULES, compute_rule
    ma = compute_rule("ma_cross"); c = pd.Series(ma["close"])
    f, s = c.rolling(RULES["ma_cross"]["fast"]).mean(), c.rolling(RULES["ma_cross"]["slow"]).mean()
    exp = [None if pd.isna(b) else float(a > b) for a, b in zip(f, s)]
    assert ma["pos"] == exp
    mo = compute_rule("momentum"); c = pd.Series(mo["close"]); L = RULES["momentum"]["lookback"]
    assert mo["pos"] == [None if i < L else float(c[i] / c[i - L] - 1 > 0) for i in range(len(c))]
    q = compute_rule("sox_gap_fade"); sox, thr = list(q["ind"].values())
    assert q["pos"] == [None if (t != t or v != v) else float(not v > t) for v, t in zip(sox, thr)]
    rs = compute_rule("rsi_rule"); rsi = list(rs["ind"].values())[0]
    state = None
    for v, p in zip(rsi, rs["pos"]):
        if v != v:
            assert p is None; continue
        state = 1.0 if v < 30 else 0.0 if v > 70 else (state if state is not None else 0.0)
        assert p == state
