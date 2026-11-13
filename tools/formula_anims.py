"""'수식 읽는 법'의 움직이는 그림(SVG 애니메이션) — 모델마다 계산이 진행되는 순서를 단계별로 보여 준다.

원칙
- 그림 속 숫자는 **실제 코드로 계산**한다: sklearn Ridge, LightGBM, statsmodels(SARIMAX·ArmaProcess), PyTorch(GRUCell),
  원본 BIVA 모듈(moving_avg·RITS), models/*.py 의 규칙 모델(positions), models/lstm.py 의 학습 기록. 문서의 작은 예와 같은 숫자는
  tests/test_formula_guide.py 가 같은 코드로 다시 확인한다(compute_* 함수).
- 정지 화면(애니메이션을 못 돌리는 보기·인쇄)에서는 **마지막 단계**가 보이도록 만든다(정적 속성 = 마지막 상태).
- 단계마다 아래쪽에 '단계 i/n' 과 한 줄 설명이 바뀐다. 전체가 끝나면 잠시 멈췄다가 처음부터 반복한다.

formula_figs.py 가 이 파일의 fig_* 를 FIGS 에 합친다(build_guide → docs/guide_build/figs/<키>.svg, 웹 /api/guide/fig/<키>).
"""
from __future__ import annotations

import math
import zlib
from types import SimpleNamespace

import numpy as np

from tools.formula_figs import C, Axes, Svg

INK, INK2 = C["ink"], C["ink2"]


# ============================================================================== 단계 애니메이션 도구
def _sub() -> Svg:
    return Svg(1, 1, "")


def _frag(sv: Svg) -> str:
    return "".join(sv.el)


class Anim:
    """n 단계 × step 초 + 마지막 hold 초를 반복. show(i, 조각) = i 단계부터(until 단계 전까지) 보임."""

    def __init__(self, s: Svg, n: int, step: float = 1.8, hold: float = 3.0):
        self.s, self.n, self.step, self.hold = s, n, step, hold
        self.dur = n * step + hold
        self._id = 0

    def f(self, i: float) -> float:
        return min(1.0, max(0.0, i * self.step / self.dur))

    def _vis(self, a: float, b: float | None) -> str:
        if b is None or b >= 1 - 1e-9:
            if a <= 1e-9:
                return ""
            kt, vals = f"0;{a:.4f}", "0;1"
        elif a <= 1e-9:
            kt, vals = f"0;{b:.4f}", "1;0"
        else:
            kt, vals = f"0;{a:.4f};{b:.4f}", "0;1;0"
        return f'<animate attributeName="opacity" dur="{self.dur:.2f}s" repeatCount="indefinite" calcMode="discrete" keyTimes="{kt}" values="{vals}"/>'

    def show(self, i: int, frag: str | Svg, until: int | None = None):
        """i 단계부터 보임. until 이 있으면 그 단계에서 사라짐(정지 화면에서는 안 보임)."""
        frag = _frag(frag) if isinstance(frag, Svg) else frag
        static = 1 if until is None else 0
        self.s.add(f'<g opacity="{static}">{self._vis(self.f(i), None if until is None else self.f(until))}{frag}</g>')

    def captions(self, x: float, y: float, texts: list[str], size: int = 12, width_chars: int = 0):
        """단계별 설명(한 번에 하나). 마지막 것은 정지 화면에도 남는다."""
        n = len(texts)
        for i, t in enumerate(texts):
            g = _sub()
            g.text(x, y, f"단계 {i + 1}/{n}", size, C["model"], weight="bold")
            g.text(x + 66, y, t, size, INK)
            self.show(i, g, None if i == n - 1 else i + 1)

    def move(self, frag: str | Svg, pos: list[tuple[float, float]], smooth: bool = False):
        """단계마다 위치(dx, dy) — pos[i] 는 i 단계의 위치. 정지 화면은 마지막 위치."""
        frag = _frag(frag) if isinstance(frag, Svg) else frag
        kts = [self.f(i) for i in range(len(pos))]
        vals = ";".join(f"{dx:.1f} {dy:.1f}" for dx, dy in pos)
        if smooth:
            kts.append(1.0); vals += f";{pos[-1][0]:.1f} {pos[-1][1]:.1f}"
        mode = "linear" if smooth else "discrete"
        kt = ";".join(f"{k:.4f}" for k in kts)
        self.s.add(f'<g transform="translate({pos[-1][0]:.1f} {pos[-1][1]:.1f})"><animateTransform attributeName="transform" type="translate" dur="{self.dur:.2f}s" '
                   f'repeatCount="indefinite" calcMode="{mode}" keyTimes="{kt}" values="{vals}"/>{frag}</g>')

    def reveal(self, frag: str | Svg, x0: float, y0: float, w: float, h: float, i_from: int, i_to: int):
        """왼쪽에서 오른쪽으로 서서히 드러냄(i_from 단계 시작 ~ i_to 단계 시작). 정지 화면은 전부."""
        frag = _frag(frag) if isinstance(frag, Svg) else frag
        self._id += 1
        cid = f"rv{self._id}_{zlib.crc32(self.s.title.encode()) % 99999}"
        a, b = self.f(i_from), self.f(i_to)
        self.s.add(f'<clipPath id="{cid}"><rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}">'
                   f'<animate attributeName="width" dur="{self.dur:.2f}s" repeatCount="indefinite" keyTimes="0;{a:.4f};{b:.4f};1" values="0;0;{w:.1f};{w:.1f}"/></rect></clipPath>'
                   f'<g clip-path="url(#{cid})">{frag}</g>')

    def slide(self, frag: str | Svg, dx: float, i_from: int, i_to: int):
        """i_from ~ i_to 단계 동안 오른쪽으로 dx 만큼 부드럽게 이동(커서). 정지 화면은 끝 위치."""
        frag = _frag(frag) if isinstance(frag, Svg) else frag
        a, b = self.f(i_from), self.f(i_to)
        self.s.add(f'<g transform="translate({dx:.1f} 0)"><animateTransform attributeName="transform" type="translate" dur="{self.dur:.2f}s" repeatCount="indefinite" '
                   f'keyTimes="0;{a:.4f};{b:.4f};1" values="0 0;0 0;{dx:.1f} 0;{dx:.1f} 0"/>{frag}</g>')


def _badge(s: Svg, x, y, text, color=C["model"], size=11, anchor="start"):
    s.text(x, y, text, size, color, anchor, "bold")


def _f(v, d=4):
    return f"{v:.{d}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.3g}"


# ============================================================================== 5. Ridge — 가중합이 쌓이는 순서
RIDGE_SUM = {"b": 0.05, "w": (0.1, -0.05), "z": (0.95, 0.8)}


def compute_ridge_sum():
    b, w, z = RIDGE_SUM["b"], RIDGE_SUM["w"], RIDGE_SUM["z"]
    contrib = [wi * zi for wi, zi in zip(w, z)]
    return {"b": b, "contrib": contrib, "yhat": b + sum(contrib)}


def fig_ridge_sum_anim():
    r = compute_ridge_sum()
    s = Svg(760, 300, "Ridge 가중합이 쌓이는 순서 (움직이는 그림)")
    A = Anim(s, 4, step=2.0)
    ax = Axes(s, 90, 30, 600, 190, (0, 4), (0, 0.16), xticks=[], yticks=[0, 0.04, 0.08, 0.12, 0.16], yfmt=lambda v: f"{v:g}")
    xs = [0.5, 1.5, 2.5, 3.5]
    labels = ["기본값 b", "재료 1 기여", "재료 2 기여", "예측 ŷ"]
    for x, lab in zip(xs, labels):
        s.text(ax.X(x), 238, lab, 12, INK2, "middle")
    bw = 90
    levels = [(0, r["b"]), (r["b"], r["b"] + r["contrib"][0]), (r["b"] + r["contrib"][0], r["yhat"]), (0, r["yhat"])]
    cols = [C["grey"], C["good"], C["bad"], C["data"]]
    texts = [f"b = {r['b']:g}", f"0.95 × 0.1 = +{r['contrib'][0]:g}", f"0.8 × (−0.05) = {r['contrib'][1]:g}", f"ŷ = {r['yhat']:.3f}"]
    for i, ((lo, hi), col, t) in enumerate(zip(levels, cols, texts)):
        g = _sub()
        y1, y2 = ax.Y(max(lo, hi)), ax.Y(min(lo, hi))
        g.rect(ax.X(xs[i]) - bw / 2, y1, bw, y2 - y1, col, col, 2)
        g.text(ax.X(xs[i]), y1 - 7, t, 12, col if i else INK, "middle", "bold")
        if 0 < i < 3:
            g.line(ax.X(xs[i - 1]) + bw / 2, ax.Y(lo), ax.X(xs[i]) - bw / 2, ax.Y(lo), C["grey"], 1, "3 3")
        if i == 3:
            g.line(ax.X(xs[2]) + bw / 2, ax.Y(r["yhat"]), ax.X(xs[3]) - bw / 2, ax.Y(r["yhat"]), C["grey"], 1, "3 3")
        A.show(i, g)
    A.captions(20, 272, ["기본값(절편) b = 0.05 에서 시작합니다",
                         "재료 1: 칸 수 z₁ 0.95 × 비율 w₁ 0.1 = +0.095 → 위로 쌓음 (비율이 + 라 예측을 올림)",
                         "재료 2: 칸 수 z₂ 0.8 × 비율 w₂ −0.05 = −0.04 → 아래로 (비율이 − 라 예측을 내림)",
                         "예측 ŷ = 0.05 + 0.095 − 0.04 = 0.105 — 재료가 18개면 막대가 18개 쌓일 뿐 방법은 같습니다"])
    return s


# ============================================================================== 6. Ridge — alpha 를 키우면 비율이 줄어든다
RIDGE_FIT = {"z": (-1.0, 0.0, 1.0), "y": (1.0, 2.0, 6.0), "alphas": (0.0, 0.5, 1.0, 3.0, 10.0, 100.0)}


def compute_ridge_alpha():
    """w = Σ z(y − ȳ) ÷ (Σ z² + α), b = ȳ (재료 평균 0). sklearn Ridge 와 같다(시험에서 확인)."""
    z, y = np.array(RIDGE_FIT["z"]), np.array(RIDGE_FIT["y"])
    out = []
    for a in RIDGE_FIT["alphas"]:
        w = float((z * (y - y.mean())).sum() / ((z ** 2).sum() + a))
        b = float(y.mean() - z.mean() * w)
        sse = float(((y - b - w * z) ** 2).sum())
        out.append({"alpha": a, "w": w, "b": b, "sse": sse, "penalty": a * w * w, "score": sse + a * w * w})
    return out


def fig_ridge_alpha_anim():
    rows = compute_ridge_alpha()
    n = len(rows)
    s = Svg(760, 320, "alpha 를 키우면 비율 w 가 0 쪽으로 (움직이는 그림)")
    A = Anim(s, n, step=2.0)
    ax = Axes(s, 60, 30, 330, 210, (-1.5, 1.5), (-1, 8), xlabel="재료 z (표준화)", ylabel="정답 y", xticks=[-1, 0, 1], yticks=[0, 2, 4, 6, 8])
    for zi, yi in zip(RIDGE_FIT["z"], RIDGE_FIT["y"]):
        s.circle(ax.X(zi), ax.Y(yi), 5, INK)
    s.line(ax.X(-1.5), ax.Y(3), ax.X(1.5), ax.Y(3), C["grey"], 1, "4 4")
    s.text(ax.X(1.5) - 2, ax.Y(3) + 14, "평균 3 (w = 0 이면 이 선)", 10, C["grey"], "end")
    bx = Axes(s, 470, 30, 250, 150, (0, 2.6), (0, 1), xticks=[0, 0.5, 1, 1.5, 2, 2.5], yticks=[])
    s.text(720, 60, "막대 = 비율 w", 10, INK2, "end")
    for i, r in enumerate(rows):
        g = _sub()
        g.line(ax.X(-1.5), ax.Y(r["b"] - 1.5 * r["w"]), ax.X(1.5), ax.Y(r["b"] + 1.5 * r["w"]), C["model"], 2.5)
        for zi, yi in zip(RIDGE_FIT["z"], RIDGE_FIT["y"]):
            p = r["b"] + r["w"] * zi
            g.line(ax.X(zi), ax.Y(yi), ax.X(zi), ax.Y(p), C["bad"], 1.2, "2 2")
        g.rect(bx.X(0), 80, bx.X(r["w"]) - bx.X(0), 40, C["model"], C["model"], 2)
        if r["w"] > 1.8:
            g.text(bx.X(r["w"]) - 6, 106, f"w = {r['w']:.3f}", 14, "#ffffff", "end", "bold")
        else:
            g.text(bx.X(r["w"]) + 6, 106, f"w = {r['w']:.3f}", 14, C["model"], weight="bold")
        g.text(470, 60, f"alpha = {r['alpha']:g}", 16, INK, weight="bold")
        g.text(470, 215, f"틀린 정도² 합 {r['sse']:.3f} + 벌점 {r['alpha']:g}×w² {r['penalty']:.3f}", 11, INK2)
        g.text(470, 232, f"= 점수 {r['score']:.3f} (이 alpha 에서 가장 작은 점수)", 11, INK2)
        A.show(i, g, None if i == n - 1 else i + 1)
    A.captions(20, 290, ["alpha 0: 벌점 없음 → 보통 최소제곱. w = 5 ÷ 2 = 2.5 (점에 가장 가깝게)",
                         "alpha 0.5: w = 5 ÷ 2.5 = 2.0 — 벌점 때문에 기울기가 줄어듦",
                         "alpha 1 (문서의 예): w = 5 ÷ 3 = 1.667",
                         "alpha 3: w = 5 ÷ 5 = 1.0",
                         "alpha 10: w = 5 ÷ 12 = 0.417 — 빨간 점선(틀린 정도)이 커져도 벌점을 줄이는 쪽",
                         "alpha 100: w = 5 ÷ 102 = 0.049 → 거의 평균선. 'alpha 가 크면 재료를 덜 믿는다' (sklearn Ridge 결과와 같음)"], size=11)
    return s


# ============================================================================== 8. LightGBM — 나무 하나를 만들고 더하는 순서
LGBM = {"z": (-1.0, -0.5, 0.5, 1.0), "y": (1.0, 2.0, 6.0, 7.0), "lam": 1.0, "eta": 0.1, "new_z": 0.3}


def compute_lgbm():
    z, y = np.array(LGBM["z"]), np.array(LGBM["y"])
    F0 = float(y.mean())
    g = F0 - y
    lam = LGBM["lam"]

    def gain(mask):
        GL, GR, HL, HR = g[mask].sum(), g[~mask].sum(), mask.sum(), (~mask).sum()
        return float(GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - g.sum() ** 2 / (len(g) + lam))
    cands = {"z ≤ −1": gain(z <= -1), "z ≤ 0": gain(z <= 0), "z ≤ 0.5": gain(z <= 0.5)}
    best = z <= 0
    leaf_l = float(-g[best].sum() / (best.sum() + lam)); leaf_r = float(-g[~best].sum() / ((~best).sum() + lam))
    new = np.where(best, F0 + LGBM["eta"] * leaf_l, F0 + LGBM["eta"] * leaf_r)
    pred_new = F0 + LGBM["eta"] * (leaf_l if LGBM["new_z"] <= 0 else leaf_r)
    return {"F0": F0, "g": g.tolist(), "gains": cands, "leaf": (leaf_l, leaf_r), "new": new.tolist(), "pred_new": float(pred_new)}


def fig_lgbm_anim():
    r = compute_lgbm()
    s = Svg(760, 360, "LightGBM 나무 하나 (움직이는 그림)")
    A = Anim(s, 7, step=2.2)
    ax = Axes(s, 70, 150, 600, 150, (0.3, 4.9), (0, 8), xticks=[], yticks=[0, 2, 4, 6, 8], ylabel="정답·예측")
    X = [ax.X(i + 1) for i in range(4)]
    for i in range(4):
        s.circle(X[i], ax.Y(LGBM["y"][i]), 5, INK)
        s.text(X[i], 318, f"{i + 1}일  z = {LGBM['z'][i]:g}", 11, INK2, "middle")
        s.text(X[i] + 9, ax.Y(LGBM["y"][i]) + 4, f"y={LGBM['y'][i]:g}", 10, INK)
    g0 = _sub()
    for i in range(4):
        g0.line(X[i] - 22, ax.Y(r["F0"]), X[i] + 22, ax.Y(r["F0"]), C["grey"], 3)
    g0.text(ax.X(4.9), ax.Y(r["F0"]) - 5, "F₀ = 평균 4", 11, C["grey"], "end")
    A.show(0, g0)
    g1 = _sub()
    for i in range(4):
        g1.line(X[i] - 30, ax.Y(r["F0"]), X[i] - 30, ax.Y(LGBM["y"][i]), C["bad"], 1.5, None, True)
        g1.text(X[i] - 36, (ax.Y(r["F0"]) + ax.Y(LGBM["y"][i])) / 2, f"g={r['g'][i]:+g}", 11, C["bad"], "end", "bold")
    A.show(1, g1, 4)
    split_x = {"z ≤ −1": (X[0] + X[1]) / 2, "z ≤ 0": (X[1] + X[2]) / 2}
    g2 = _sub(); g2.line(split_x["z ≤ −1"], 150, split_x["z ≤ −1"], 300, C["warmline"], 2, "6 4")
    g2.text(split_x["z ≤ −1"] + 4, 162, f"z ≤ −1 ? 이득 {r['gains']['z ≤ −1']:.2f}", 11, C["warmline"], weight="bold")
    A.show(2, g2, 3)
    g3 = _sub(); g3.line(split_x["z ≤ 0"], 150, split_x["z ≤ 0"], 300, C["model"], 2.5, "6 4")
    g3.text(split_x["z ≤ 0"] + 4, 162, f"z ≤ 0 ? 이득 {r['gains']['z ≤ 0']:.2f} ← 가장 큼", 11, C["model"], weight="bold")
    A.show(3, g3)
    gt = _sub()
    gt.box(290, 10, 180, 40, ["질문: z ≤ 0 ?"], C["warm"], C["warmline"], 12)
    gt.line(340, 50, 200, 82, arrow=True); gt.line(420, 50, 560, 82, arrow=True)
    gt.text(255, 70, "예", 11, C["good"], weight="bold"); gt.text(505, 70, "아니오", 11, C["bad"], weight="bold")
    A.show(3, gt)
    g4 = _sub()
    g4.box(90, 84, 220, 48, ["왼쪽 잎 (1·2일): G = 3 + 2 = 5", f"잎 값 = −5 ÷ (2 + 1) = {r['leaf'][0]:.3f}"], C["fill"], size=11, bold_first=False)
    g4.box(450, 84, 220, 48, ["오른쪽 잎 (3·4일): G = −5", f"잎 값 = +5 ÷ (2 + 1) = +{r['leaf'][1]:.3f}"], C["fill"], size=11, bold_first=False)
    A.show(4, g4)
    g5 = _sub()
    for i in range(4):
        g5.line(X[i] - 22, ax.Y(r["new"][i]), X[i] + 22, ax.Y(r["new"][i]), C["model"], 3)
        g5.text(X[i] + 26, ax.Y(r["new"][i]) + (12 if i < 2 else -4), f"{r['new'][i]:.3f}", 11, C["model"], weight="bold")
    A.show(5, g5)
    g6 = _sub()
    xn = (X[2] + X[1]) / 2 + 0.3 * (X[2] - X[1])
    g6.add(f'<polygon points="{xn},{ax.Y(r["pred_new"]) - 9} {xn + 8},{ax.Y(r["pred_new"]) + 6} {xn - 8},{ax.Y(r["pred_new"]) + 6}" fill="{C["data"]}"/>')
    g6.text(xn, 140, f"새 날 z = 0.3 → 아니오 → {r['pred_new']:.3f}", 11, C["data"], "middle", "bold")
    A.show(6, g6)
    A.captions(20, 344, ["처음엔 모든 날을 평균 4 로 예측 (F₀)",
                         "틀린 만큼 = 기울기 g = 예측 − 정답 (3, 2, −2, −3)",
                         "질문 후보 'z ≤ −1 ?' → 나눴을 때의 이득 6.75",
                         "질문 후보 'z ≤ 0 ?' → 이득 16.67 로 가장 큼 → 이 질문을 고름",
                         "칸마다 잎 값 = −(기울기 합) ÷ (날 수 + λ 1)",
                         "학습률 0.1 을 곱해 더함 → 3.833 / 4.167 (실제 LightGBM 출력과 같음)",
                         "예측: 새 날 z = 0.3 은 '아니오' 쪽 잎 → 4.167. 나무가 여러 개면 잎 값을 차례로 더함"], size=11)
    return s


# ============================================================================== 10. 칼만 필터 — 하루씩 예측하고 고치기
KALMAN = {"params": (0.001, 0.5, 0.3), "r": (0.004, 0.012, -0.006, 0.010, 0.020), "steps": 3}


def compute_kalman():
    """ARMA(1,1) c=0.001 φ=0.5 θ=0.3 — statsmodels SARIMAX 의 칼만 필터로 한 기간 앞 예측·오차·이어 예측."""
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    c, phi, th = KALMAN["params"]
    r = np.array(KALMAN["r"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(r, order=(1, 0, 1), trend="c").filter(np.array([c, phi, th, 1e-4]))
    one = np.asarray(res.forecasts[0], float); err = np.asarray(res.forecasts_error[0], float)
    fc = np.asarray(res.forecast(KALMAN["steps"]), float)
    return {"one": one.tolist(), "err": err.tolist(), "forecast": fc.tolist(), "mean": c / (1 - phi)}


def fig_kalman_anim():
    k = compute_kalman()
    r = KALMAN["r"]; n_obs, n_fc = len(r), KALMAN["steps"]
    s = Svg(760, 330, "칼만 필터: 예측 → 실제 → 고치기 (움직이는 그림)")
    A = Anim(s, n_obs + n_fc + 1, step=1.7)
    ax = Axes(s, 80, 30, 620, 200, (0.4, n_obs + n_fc + 0.6), (-0.012, 0.024), xticks=list(range(1, n_obs + n_fc + 1)),
              fmt=lambda v: (f"τ{int(v)}" if v <= n_obs else f"t+{int(v) - n_obs}"), yticks=[-0.01, 0, 0.01, 0.02], yfmt=lambda v: f"{v:+.2f}" if v else "0", ylabel="1기간 변화율")
    s.line(ax.X(0.4), ax.Y(k["mean"]), ax.X(n_obs + n_fc + 0.6), ax.Y(k["mean"]), C["grey"], 1, "4 4")
    s.text(ax.X(n_obs + n_fc + 0.6), ax.Y(k["mean"]) - 4, f"장기 평균 c ÷ (1 − φ) = {k['mean']:g}", 10, C["grey"], "end")
    s.line(ax.X(n_obs + 0.5), 30, ax.X(n_obs + 0.5), 230, C["bad"], 1, "3 3")
    s.text(ax.X(n_obs + 0.5) + 4, 42, "결정일 이후: 실제 값 없음", 10, C["bad"])
    for i in range(n_obs):
        g = _sub()
        x = ax.X(i + 1)
        g.circle(x, ax.Y(k["one"][i]), 6, C["model"], "#ffffff", 2)
        g.circle(x, ax.Y(r[i]), 5, C["data"])
        g.line(x + 9, ax.Y(k["one"][i]), x + 9, ax.Y(r[i]), C["bad"], 2)
        g.text(x + 13, (ax.Y(k["one"][i]) + ax.Y(r[i])) / 2 + 4, f"ε={k['err'][i]:+.4f}", 10, C["bad"], weight="bold")
        A.show(i, g)
    for h in range(n_fc):
        g = _sub()
        x = ax.X(n_obs + h + 1)
        g.circle(x, ax.Y(k["forecast"][h]), 6, C["model"])
        g.text(x + 9, ax.Y(k["forecast"][h]) + 4, f"{k['forecast'][h]:.5f}", 10, C["model"], weight="bold")
        A.show(n_obs + h, g)
    g = _sub()
    g.circle(96, 262, 6, C["model"], "#ffffff", 2); g.text(108, 266, "한 기간 앞 예측", 11, INK2)
    g.circle(226, 262, 5, C["data"]); g.text(238, 266, "실제", 11, INK2)
    g.line(290, 256, 290, 268, C["bad"], 2); g.text(298, 266, "충격 ε = 실제 − 예측", 11, INK2)
    g.circle(440, 262, 6, C["model"]); g.text(452, 266, "결정일 뒤 이어 예측 (미래 충격 = 0)", 11, INK2)
    s.add(_frag(g))
    caps = [f"τ1: 기억이 없어 장기 평균 {k['one'][0]:.4f} 로 예측 → 실제 {r[0]} → 충격 {k['err'][0]:+.4f}"]
    c_, phi_, th_ = KALMAN["params"]
    for i in range(1, n_obs):
        naive = c_ + phi_ * r[i - 1] + th_ * k["err"][i - 1]
        caps.append(f"τ{i + 1}: 필터 예측 {k['one'][i]:.5f} (식 c + φ·지난 실제 + θ·지난 충격 = {naive:.5f}, 차이 {k['one'][i] - naive:+.5f}) → 실제 {r[i]} → 충격 {k['err'][i]:+.4f}")
    caps += [f"t+{h + 1}: 실제 값이 없으므로 예측만 이어 감 → {k['forecast'][h]:.5f}" + (" (미래 충격은 0)" if h else "") for h in range(n_fc)]
    caps += ["처음 몇 기간은 필터가 지난 충격을 아직 확신하지 못해 식과 조금 다르고, 기간이 지날수록 차이가 0 으로(10절) — statsmodels SARIMAX 칼만 필터 값(계수 고정)"]
    A.captions(20, 304, caps, size=11)
    return s


# ============================================================================== 9-2. 근의 크기 — 충격이 사라지나, 오차를 되짚을 수 있나
ROOTS = {"ar": (0.5, 1.2), "ma": (0.5, 2.0), "lags": 6}


def compute_roots():
    from statsmodels.tsa.arima_process import ArmaProcess, arma2ar, arma_impulse_response
    L = ROOTS["lags"]
    ar = {phi: {"irf": arma_impulse_response([1, -phi], [1], L).tolist(), "root": float(abs(ArmaProcess([1, -phi], [1]).arroots[0]))} for phi in ROOTS["ar"]}
    ma = {th: {"inv": arma2ar([1], [1, th], L).tolist(), "root": float(abs(ArmaProcess([1], [1, th]).maroots[0]))} for th in ROOTS["ma"]}
    return {"ar": ar, "ma": ma}


def fig_roots_anim():
    r = compute_roots()
    L = ROOTS["lags"]
    s = Svg(760, 400, "근의 크기: 충격이 사라지나 · 오차를 되짚을 수 있나 (움직이는 그림)")
    A = Anim(s, L + 1, step=1.5)
    panels = [("AR(1): 오늘 충격 1 이 j 기간 뒤 남는 크기", "ar", ROOTS["ar"], "irf", 40), ("MA(1): 오늘 충격을 되짚는 무게 (j 기간 전 관측에 곱함)", "ma", ROOTS["ma"], "inv", 205)]
    for title, kind, vals, key, y0 in panels:
        s.text(40, y0 + 4, title, 12, INK, weight="bold")
        for col, v in enumerate(vals):
            x0 = 40 + col * 250
            good = r[kind][v]["root"] > 1
            sym = "φ" if kind == "ar" else "θ"
            s.text(x0, y0 + 24, f"{sym} = {v:g} → 근 |z| = {r[kind][v]['root']:.3f} {'> 1 (사라짐)' if good else '< 1 (안 사라짐)'}", 11, C["good"] if good else C["bad"], weight="bold")
            ax = Axes(s, x0 + 10, y0 + 40, 210, 90, (-0.6, L - 0.4), (-2.5, 2.5) if kind == "ma" else (0, 2.5), xticks=list(range(L)), fmt=lambda q: f"j={int(q)}",
                      yticks=[-2, 0, 2] if kind == "ma" else [0, 1, 2], yfmt=lambda q: f"{q:g}")
            for j in range(L):
                w = r[kind][v][key][j]
                h = max(-2.5, min(2.5, w))
                g = _sub()
                y_top, y_bot = ax.Y(max(h, 0)), ax.Y(min(h, 0))
                g.rect(ax.X(j) - 11, y_top, 22, max(y_bot - y_top, 1), C["good"] if good else C["bad"], "none", 1)
                if abs(w) > 2.5:  # 눈금 밖: 막대 안에 흰 글씨
                    g.text(ax.X(j), (y_top + 12) if w > 0 else (y_bot - 5), f"{w:g}", 9, "#ffffff", "middle", "bold")
                else:
                    g.text(ax.X(j), (y_top - 4) if w >= 0 else (y_bot + 11), f"{w:g}", 9, INK, "middle")
                A.show(j, g)
    g = _sub()
    g.rect(560, 40, 190, 150, "#fafbfc", C["line"], 4)
    g.text(570, 60, "근의 크기 |z| (1 이 경계)", 11, INK, weight="bold")
    nx = lambda v: 640 + v / 2.5 * 100
    g.line(nx(1), 70, nx(1), 172, C["ink2"], 1.2, "4 3"); g.text(nx(1), 184, "1", 10, INK2, "middle")
    g.text(nx(0.1), 184, "0", 10, INK2, "middle"); g.text(nx(2.5), 184, "2.5", 10, INK2, "middle")
    for i, (kind, v, lab) in enumerate([("ar", 0.5, "φ 0.5"), ("ar", 1.2, "φ 1.2"), ("ma", 0.5, "θ 0.5"), ("ma", 2.0, "θ 2")]):
        z = r[kind][v]["root"]; yy = 84 + i * 24
        g.text(570, yy + 4, lab, 10, INK2)
        g.line(nx(0), yy, nx(2.5), yy, "#e3e6ea", 1)
        g.circle(nx(z), yy, 5, C["good"] if z > 1 else C["bad"])
        g.text(nx(z) + (8 if z < 2 else -8), yy - 6, f"{z:.3g}", 9, C["good"] if z > 1 else C["bad"], "start" if z < 2 else "end")
    A.show(L, g)
    A.captions(20, 380, [f"j = {j}: " + ("시작 — 충격 1, 되짚는 무게 1" if j == 0 else
                         f"φ 0.5 → {r['ar'][0.5]['irf'][j]:g} · φ 1.2 → {r['ar'][1.2]['irf'][j]:g} | θ 0.5 → {r['ma'][0.5]['inv'][j]:g} · θ 2 → {r['ma'][2.0]['inv'][j]:g}") for j in range(L)]
               + ["판정: 근의 크기가 1 보다 크면(초록) 갈수록 줄고, 1 보다 작으면(빨강) 커짐 — '제약 켬'은 초록 범위에서만 찾음"], size=11)
    return s


# ============================================================================== 11. SARIMA — 펼친 계수로 다음 값 만들기
SARIMA = {"phi": 0.5, "s": 4, "r": {-5: 0.015, -4: 0.03, -3: 0.01, -2: -0.005, -1: 0.008, 0: 0.02}}


def compute_sarima():
    """(1 − 0.5L)(1 − L⁴) 를 펼친 계수와 다음 기간 예측. statsmodels SARIMAX 예측과 같다(시험에서 확인)."""
    A = np.convolve([1, -SARIMA["phi"]], np.r_[1, np.zeros(SARIMA["s"] - 1), -1])
    r = SARIMA["r"]
    terms = [(i, float(-A[i]), r[1 - i]) for i in range(1, len(A)) if A[i] != 0]
    return {"poly": A.tolist(), "terms": terms, "pred": float(sum(a * v for _, a, v in terms))}


def fig_sarima_anim():
    c = compute_sarima()
    s = Svg(760, 330, "SARIMA: 펼친 계수로 다음 값 (움직이는 그림)")
    A = Anim(s, 5, step=2.2)
    ax = Axes(s, 70, 100, 440, 150, (-5.6, 1.6), (-0.01, 0.035), xticks=list(range(-5, 2)), fmt=lambda v: "t" if v == 0 else (f"t+{int(v)}" if v > 0 else f"t{int(v)}"),
              yticks=[0, 0.01, 0.02, 0.03], yfmt=lambda v: f"{v:g}", ylabel="변화율")
    for t, v in SARIMA["r"].items():
        s.circle(ax.X(t), ax.Y(v), 5, C["data"])
        s.text(ax.X(t) + 8, ax.Y(v) + 4, f"{v:g}", 10, INK2)
    g0 = _sub()
    g0.text(20, 26, "(1 − 0.5L)(1 − L⁴) = 1 − 0.5L − L⁴ + 0.5L⁵", 14, INK, weight="bold")
    g0.text(20, 46, "→ r(t+1) = 0.5·r(t) + 1·r(t−3) − 0.5·r(t−4) + ε   (L = 한 기간 전, L⁴ = 4기간 전)", 12, INK2)
    A.show(0, g0)
    cols = [C["model"], C["good"], C["bad"]]
    names = {1: "r(t)", 4: "r(t−3)", 5: "r(t−4)"}
    s.rect(540, 96, 205, 150, "#fafbfc", C["line"], 4)
    s.text(552, 116, "다음 값 = 항목의 합", 12, INK, weight="bold")
    for k, (i_, a, v) in enumerate(c["terms"]):
        g = _sub()
        x0, y0 = ax.X(1 - i_), ax.Y(v)
        g.circle(x0, y0, 9, cols[k], "none", 2.5)
        g.add(f'<path d="M{x0:.1f},{y0 - 10:.1f} Q{(x0 + ax.X(1)) / 2:.1f},{62 + 8 * k:.1f} {ax.X(1):.1f},{ax.Y(c["pred"]) - 12:.1f}" fill="none" stroke="{cols[k]}" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#ah)"/>')
        g.text(552, 142 + 24 * k, f"{a:+g} × {names[i_]} {v:g} = {a * v:+.3f}", 12, cols[k], weight="bold")
        A.show(k + 1, g)
    g = _sub()
    g.circle(ax.X(1), ax.Y(c["pred"]), 7, C["model"])
    g.text(ax.X(1), ax.Y(c["pred"]) + 22, f"{c['pred']:.3f}", 12, C["model"], "middle", "bold")
    g.line(552, 210, 735, 210, C["line"], 1)
    g.text(552, 232, f"합 = {c['pred']:.3f}", 14, C["model"], weight="bold")
    A.show(4, g)
    A.captions(20, 312, ["두 괄호를 곱해 펼치면 '몇 기간 전 값에 얼마를 곱하나'의 목록이 됩니다 (s = 4, D = 1)",
                         "1기간 전 r(t) = 0.02 에 0.5 를 곱함 → +0.010",
                         "4기간 전(작년 같은 철) r(t−3) = 0.01 에 1 을 곱함 → +0.010",
                         "5기간 전 r(t−4) = 0.03 에 −0.5 를 곱함 → −0.015",
                         "더하면 다음 기간 예측 0.005 — statsmodels SARIMAX 예측과 소수 8자리까지 같음 (나머지 칸은 계수 0)"], size=11)
    return s


# ============================================================================== 13. VECM — 벌어진 만큼 되돌아오기
VECM = {"y0": (104.0, 50.0), "beta": (1.0, -2.0), "alpha": (-0.25, 0.1), "steps": 5}


def compute_vecm():
    y = np.array(VECM["y0"]); be, al = np.array(VECM["beta"]), np.array(VECM["alpha"])
    path = []
    for _ in range(VECM["steps"]):
        e = float(be @ y)
        path.append({"y": y.tolist(), "e": e, "adj": (al * e).tolist()})
        y = y + al * e
    path.append({"y": y.tolist(), "e": float(be @ y), "adj": None})
    return path


def fig_vecm_anim():
    p = compute_vecm()
    n = len(p)
    s = Svg(760, 330, "VECM: 균형에서 벗어난 만큼 되돌아오기 (움직이는 그림)")
    A = Anim(s, n, step=2.0)
    ax = Axes(s, 80, 30, 560, 210, (-0.4, n - 0.6), (99.5, 104.5), xticks=list(range(n)), fmt=lambda v: f"{int(v)}기간 뒤" if v else "지금",
              yticks=[100, 101, 102, 103, 104], yfmt=lambda v: f"{v:g}")
    for i, st in enumerate(p):
        g = _sub()
        x = ax.X(i)
        hy, cy2 = st["y"][0], 2 * st["y"][1]
        g.circle(x, ax.Y(hy), 6, C["data"]); g.rect(x - 6, ax.Y(cy2) - 6, 12, 12, C["model"], C["model"], 2)
        g.line(x + 12, ax.Y(hy), x + 12, ax.Y(cy2), C["bad"], 2.5)
        g.text(x + 16, (ax.Y(hy) + ax.Y(cy2)) / 2 + 4, f"e={st['e']:.3f}", 11, C["bad"], weight="bold")
        g.text(x, ax.Y(hy) - 10, f"{hy:.3f}", 10, C["data"], "middle")
        g.text(x, ax.Y(cy2) + 20, f"2×{st['y'][1]:.3f}", 10, C["model"], "middle")
        if i:
            q = p[i - 1]
            g.line(ax.X(i - 1), ax.Y(q["y"][0]), x, ax.Y(hy), C["data"], 1.5, "3 3")
            g.line(ax.X(i - 1), ax.Y(2 * q["y"][1]), x, ax.Y(cy2), C["model"], 1.5, "3 3")
        A.show(i, g)
    s.circle(660, 60, 6, C["data"]); s.text(672, 64, "집값", 11, INK2)
    s.rect(654, 76, 12, 12, C["model"], C["model"], 2); s.text(672, 86, "2 × 물가", 11, INK2)
    s.line(660, 98, 660, 112, C["bad"], 2.5); s.text(672, 109, "이탈 e", 11, INK2)
    s.text(650, 132, "균형: 집값 = 2×물가", 10, INK2)
    caps = []
    for i, st in enumerate(p):
        if st["adj"] is None:
            caps.append(f"{i}기간 뒤 e = {st['e']:.3f} — 매번 0.55 배(= 1 − 0.25 − 2×0.1)로 줄며 균형으로 돌아갑니다")
        else:
            caps.append(f"e = {st['y'][0]:.3f} − 2×{st['y'][1]:.3f} = {st['e']:.3f} → 되돌림 α·e = ({st['adj'][0]:+.3f}, {st['adj'][1]:+.3f}) 를 더함")
    A.captions(20, 304, caps, size=11)
    return s


# ============================================================================== 15-2. GRU 셀 — 문 두 개로 섞기
GRU = {"x": 1.0, "h": 0.2, "pre_r": 0.0, "pre_z": 1.0, "n_in": 0.6, "n_hid": 0.8}


def compute_gru():
    sig = lambda a: 1 / (1 + math.exp(-a))
    r, z = sig(GRU["pre_r"]), sig(GRU["pre_z"])
    n = math.tanh(GRU["n_in"] + r * GRU["n_hid"])
    h2 = (1 - z) * n + z * GRU["h"]
    return {"r": r, "z": z, "n": n, "h": h2, "part_new": (1 - z) * n, "part_old": z * GRU["h"]}


def fig_gru_anim():
    g_ = compute_gru()
    s = Svg(760, 330, "GRU 셀: 문 두 개 (움직이는 그림)")
    A = Anim(s, 5, step=2.2)
    s.box(20, 60, 120, 50, ["입력 x = 1", "(이번 시점 재료)"], C["fill"], size=11)
    s.box(20, 170, 120, 50, [f"지난 기억 h = {GRU['h']:g}"], "#e7eef8", C["data"], 11)
    boxes = [(200, 30, "재설정 문 r", f"σ(0) = {g_['r']:.4f}", "지난 기억을 얼마나 볼지"),
             (200, 130, "갱신 문 z", f"σ(1) = {g_['z']:.4f}", "새것 : 지난 것 = (1−z) : z"),
             (400, 80, "새 후보 n", f"tanh(0.6 + {g_['r']:.1f} × 0.8) = {g_['n']:.4f}", "입력 부분 0.6, 기억 부분 0.8")]
    for i, (x, y, t, v, d) in enumerate(boxes):
        g = _sub()
        g.box(x, y, 170 if i < 2 else 200, 64, [t, v, d], C["warm"], C["warmline"], 11)
        g.line(140, 85 if i != 1 else 195, x - 2, y + 32, C["ink2"], 1.2, None, True)
        if i == 2:
            g.line(370, 62, 398, 100, C["ink2"], 1.2, None, True)
        A.show(i + 1, g)
    g4 = _sub()
    W = 300; x0 = 400
    g4.text(x0, 196, "새 기억 h′ = (1 − z) × n + z × h", 12, INK, weight="bold")
    wn = W * (1 - g_["z"])
    g4.rect(x0, 206, wn, 30, C["model"], C["model"], 2); g4.rect(x0 + wn, 206, W - wn, 30, C["data"], C["data"], 2)
    g4.text(x0 + wn / 2, 226, f"{1 - g_['z']:.4f}", 11, "#ffffff", "middle", "bold")
    g4.text(x0 + wn + (W - wn) / 2, 226, f"z {g_['z']:.4f}", 11, "#ffffff", "middle", "bold")
    g4.text(x0, 254, f"{1 - g_['z']:.4f} × {g_['n']:.4f} = {g_['part_new']:.4f}  (새 후보 몫)", 11, C["model"])
    g4.text(x0, 270, f"{g_['z']:.4f} × {GRU['h']:g} = {g_['part_old']:.4f}  (지난 기억 몫)", 11, C["data"])
    g4.text(x0, 290, f"h′ = {g_['h']:.4f}", 15, INK, weight="bold")
    A.show(4, g4)
    g0 = _sub(); g0.add(f'<rect x="16" y="56" width="128" height="168" rx="6" fill="none" stroke="{C["model"]}" stroke-width="2.5"/>')
    A.show(0, g0, 1)
    A.captions(20, 318, ["재료: 이번 입력 x 와 지난 기억 h (첫 시점이면 h = 0)",
                         "재설정 문 r = σ(가중합) — 0 이면 지난 기억을 무시, 1 이면 그대로 봄",
                         "갱신 문 z = σ(가중합) — 지난 기억을 얼마나 그대로 남길지",
                         "새 후보 n = tanh(입력 부분 + r × 기억 부분)",
                         "섞기: h′ = (1 − z)·n + z·h = 0.2048 + 0.1462 = 0.3510 (PyTorch GRUCell 과 같음)"], size=11)
    return s


# ============================================================================== 14. 신경망 학습 — 기울기 반대로 조금씩 (작은 예)
GD = {"x": 2.0, "y": 3.0, "w0": 1.0, "lr": 0.1, "steps": 5}


def compute_gd():
    w, out = GD["w0"], []
    for _ in range(GD["steps"]):
        yh = w * GD["x"]; loss = (yh - GD["y"]) ** 2; grad = 2 * (yh - GD["y"]) * GD["x"]
        out.append({"w": w, "yhat": yh, "loss": loss, "grad": grad})
        w = w - GD["lr"] * grad
    out.append({"w": w, "yhat": w * GD["x"], "loss": (w * GD["x"] - GD["y"]) ** 2, "grad": 2 * (w * GD["x"] - GD["y"]) * GD["x"]})
    return out


def fig_gd_anim():
    st = compute_gd()
    n = len(st)
    s = Svg(760, 320, "가중치를 기울기 반대로 조금씩 (움직이는 그림)")
    A = Anim(s, n, step=1.8)
    ax = Axes(s, 70, 30, 420, 220, (0.6, 2.0), (0, 1.3), xlabel="가중치 w", ylabel="손실 (ŷ − y)²", xticks=[0.6, 1, 1.4, 1.8], yticks=[0, 0.5, 1])
    ws = np.linspace(0.6, 2.0, 80)
    s.poly([(ax.X(w), ax.Y((w * GD["x"] - GD["y"]) ** 2)) for w in ws if (w * GD["x"] - GD["y"]) ** 2 <= 1.3], C["grey"], 2)
    s.line(ax.X(1.5), ax.Y(0), ax.X(1.5), ax.Y(0.05), C["good"], 2)
    s.text(ax.X(1.5), ax.Y(0) - 10, "바닥 w = 1.5", 10, C["good"], "middle")
    for i, q in enumerate(st):
        g = _sub()
        X, Y = ax.X(q["w"]), ax.Y(q["loss"])
        g.circle(X, Y, 7, C["model"])
        dx = min(0.18, 0.3 / abs(q["grad"])) if q["grad"] else 0.18
        g.line(ax.X(q["w"] - dx), ax.Y(q["loss"] - q["grad"] * dx), ax.X(q["w"] + dx), ax.Y(q["loss"] + q["grad"] * dx), C["bad"], 1.5, "4 3")
        A.show(i, g, None if i == n - 1 else i + 1)
        t = _sub()
        t.text(520, 60, f"{i}번째 고침 뒤" if i else "처음", 13, INK, weight="bold")
        t.text(520, 86, f"w = {q['w']:.4f}", 13, C["model"], weight="bold")
        t.text(520, 108, f"ŷ = w × 2 = {q['yhat']:.4f}", 12, INK2)
        t.text(520, 128, f"손실 = ({q['yhat']:.4f} − 3)² = " + (f"{q['loss']:.5f}" if q["loss"] >= 1e-4 else f"{q['loss']:.1e}"), 12, INK2)
        t.text(520, 148, f"기울기 = 2 × (ŷ − y) × x = {q['grad']:.4f}", 12, C["bad"])
        if i < n - 1:
            t.text(520, 174, f"다음 w = {q['w']:.4f} − 0.1 × ({q['grad']:.4f})", 12, INK)
            t.text(520, 192, f"      = {st[i + 1]['w']:.4f}", 12, INK, weight="bold")
        A.show(i, t, None if i == n - 1 else i + 1)
    caps = [f"처음 w = 1: 손실 1, 기울기 −4 (빨간 점선 = 그 점의 기울기)"]
    caps += [f"w ← w − 학습률 0.1 × 기울기 → {st[i]['w']:.4f}, 손실 " + (f"{st[i]['loss']:.5f}" if st[i]["loss"] >= 1e-4 else f"{st[i]['loss']:.1e}") for i in range(1, n)]
    caps[-1] += " — 바닥(1.5)에 다가갈수록 기울기가 작아져 걸음도 작아짐"
    A.captions(20, 300, caps, size=11)
    return s


# ============================================================================== 14. 신경망 학습 — 실제 LSTM 학습 기록과 조기종료
NN_TRAIN = {"seed": 7, "n_train": 240, "n_val": 60, "seq": 6, "params": {"hidden": 8, "layers": 1, "dropout": 0.0, "lr": 0.003, "epochs": 120,
                                                                       "batch_size": 32, "patience": 12, "weight_decay": 1e-4}}


def compute_nn_train():
    """models/lstm.py 를 작은 합성 자료(사인파 + 잡음, 창 6)로 실제 학습시킨 epoch 별 손실·조기종료 기록."""
    import torch

    from models import lstm as M
    from models.base import resolve_params
    rng = np.random.default_rng(NN_TRAIN["seed"])
    T = NN_TRAIN["n_train"] + NN_TRAIN["n_val"] + NN_TRAIN["seq"] + 1
    t = np.arange(T)
    sig = np.sin(t / 5.0)
    x1 = sig + rng.normal(0, 0.6, T); x2 = rng.normal(0, 1, T)
    y_all = np.r_[sig[1:], 0] + rng.normal(0, 0.6, T)
    F = np.c_[x1, x2]
    L = NN_TRAIN["seq"]
    X = np.stack([F[i - L + 1:i + 1] for i in range(L - 1, T - 1)]).astype(np.float32)
    y = y_all[L - 1:T - 1].astype(np.float32)
    ntr = NN_TRAIN["n_train"]
    torch.manual_seed(NN_TRAIN["seed"]); np.random.seed(NN_TRAIN["seed"])
    p = resolve_params(M.PARAMS, NN_TRAIN["params"])
    m = M.Model(p, n_features=2, target_type="regression", device="cpu")
    m.fit(X[:ntr], y[:ntr], X[ntr:ntr + NN_TRAIN["n_val"]], y[ntr:ntr + NN_TRAIN["n_val"]])
    return {"history": m.history, "best_epoch": m.best_epoch, "patience": p["patience"], "epochs": p["epochs"]}


def fig_nn_train_anim():
    r = compute_nn_train()
    h = r["history"]; E = len(h)
    s = Svg(760, 330, "실제 LSTM 학습 기록과 조기종료 (움직이는 그림)")
    A = Anim(s, 4, step=2.6, hold=3.5)
    tr = [q["train_loss"] for q in h]; va = [q["val_loss"] for q in h]
    ymax = math.ceil(max(max(tr), max(va)) * 10) / 10; ymin = math.floor(min(min(tr), min(va)) * 10) / 10
    ax = Axes(s, 70, 30, 600, 220, (1, max(E, 2)), (ymin, ymax), xlabel="epoch (전체 학습 자료를 한 바퀴 본 횟수)", ylabel="손실 (표준화한 정답 기준 MSE)",
              xticks=[1] + list(range(10, E + 1, 10)), yticks=list(np.round(np.arange(ymin, ymax + 1e-9, 0.2), 1)), yfmt=lambda v: f"{v:.1f}")
    pts_tr = [(ax.X(i + 1), ax.Y(v)) for i, v in enumerate(tr)]
    pts_va = [(ax.X(i + 1), ax.Y(v)) for i, v in enumerate(va)]
    g = _sub(); g.poly(pts_tr, C["data"], 2); g.poly(pts_va, C["model"], 2)
    A.reveal(g, 70, 20, 610, 240, 0, 2)
    b = r["best_epoch"]
    g2 = _sub()
    g2.line(ax.X(b), 30, ax.X(b), 250, C["good"], 1.5, "5 4")
    g2.circle(ax.X(b), ax.Y(va[b - 1]), 7, C["good"])
    g2.text(ax.X(b) + 6, 44, f"검증 손실 최저: epoch {b} ({va[b - 1]:.3f})", 11, C["good"], weight="bold")
    A.show(2, g2)
    g3 = _sub()
    g3.line(ax.X(E), 30, ax.X(E), 250, C["bad"], 1.5, "2 3")
    g3.text(ax.X(E) - 4, 64, f"epoch {E}: {r['patience']}번 연속 나아지지 않아 멈춤", 11, C["bad"], "end", weight="bold")
    g3.text(ax.X(E) - 4, 80, f"→ epoch {b} 의 가중치로 되돌림", 11, C["bad"], "end")
    A.show(3, g3)
    s.line(330, 16, 350, 16, C["data"], 2); s.text(356, 20, "학습 손실(배치 평균)", 11, INK2)
    s.line(480, 16, 500, 16, C["model"], 2); s.text(506, 20, "검증 손실", 11, INK2)
    A.captions(20, 312, ["epoch 마다 학습 자료를 섞어 32개씩 넣고 가중치를 고침(Adam) → 학습 손실(파랑)이 내려감",
                         "검증 손실(주황)은 학습에 쓰지 않은 자료 — 어느 순간부터 더 나아지지 않음(외우기 시작)",
                         f"검증 손실이 가장 낮았던 epoch {b} 를 기억",
                         f"{r['patience']}번 연속 개선이 없으면 멈추고 epoch {b} 의 가중치로 되돌림 — models/lstm.py 를 실제로 돌린 기록"], size=11)
    return s


# ============================================================================== 17. BIVA Stage 1 — BRITS 가 빈 칸을 채우는 순서 (원본 모듈 그대로)
BRITS = {"x": [[0.2, float("nan")], [0.3, 0.5], [float("nan"), 0.4], [float("nan"), 0.6], [0.6, float("nan")], [0.7, 0.8]],
         "names": ("변수 A", "변수 B"), "hid": 4, "seed": 0}


def compute_brits():
    """원본 BIVA_BRITS(두 방향 RITS) 를 학습 전 가중치(시드 0)로 그대로 돌린 값. δ 는 원본 delta_calc 그대로."""
    import torch

    from models.biva_src import BIVA_BRITS
    torch.manual_seed(BRITS["seed"])
    x = torch.tensor([BRITS["x"]], dtype=torch.float32)
    T, D = x.shape[1], x.shape[2]
    args = SimpleNamespace(seq_len=T, channels=D, rnn_hid_size=BRITS["hid"], imputed_weight=1.0)
    m = BIVA_BRITS.Model(args).eval()
    with torch.no_grad():
        rf = m.rits_f(x, "forward")
        rb = m.rits_b(x, "backward")
        out = m(x)
    fwd = rf["imputations"][0].numpy()
    bwd = torch.flip(rb["imputations"], [1])[0].numpy()  # 뒤→앞 결과를 원래 시간 순서로
    mask = (~torch.isnan(x)).float()
    d_f = m.rits_f.delta_calc(mask)[0].numpy()
    d_b = torch.flip(m.rits_b.delta_calc(torch.flip(mask, [1])), [1])[0].numpy()
    return {"fwd": fwd.tolist(), "bwd": bwd.tolist(), "final": out["imputations"][0].numpy().tolist(), "mask": mask[0].numpy().tolist(),
            "delta_f": d_f.tolist(), "delta_b": d_b.tolist()}


def fig_brits_anim():
    r = compute_brits()
    X = BRITS["x"]; T, D = len(X), len(X[0])
    s = Svg(760, 380, "BRITS: 앞→뒤, 뒤→앞으로 읽으며 빈 칸 채우기 (움직이는 그림)")
    A = Anim(s, 2 * T + 1, step=1.2, hold=4.0)
    x0, y0, cw, ch = 150, 50, 90, 46
    for t in range(T):
        s.text(x0 + t * cw + cw / 2, y0 - 10, f"τ{t + 1}", 12, INK2, "middle")
    for d in range(D):
        s.text(x0 - 10, y0 + d * ch * 2 + 28, BRITS["names"][d], 12, INK, "end", "bold")
        s.text(x0 - 10, y0 + d * ch * 2 + 28 + ch, "δ 앞→뒤 / 뒤→앞", 9, INK2, "end")
        for t in range(T):
            xx, yy = x0 + t * cw, y0 + d * ch * 2
            miss = r["mask"][t][d] == 0
            s.rect(xx + 2, yy + 2, cw - 4, ch - 4, "#f3f4f6" if miss else "#ffffff", C["line"], 3)
            if not miss:
                s.text(xx + cw / 2, yy + 28, f"{X[t][d]:g}", 13, C["data"], "middle", "bold")
            else:
                s.text(xx + 8, yy + 16, "빈 칸", 9, C["grey"])
    # 앞→뒤
    for t in range(T):
        g = _sub()
        g.rect(x0 + t * cw, y0 - 2, cw, D * ch * 2 - 4, "none", C["model"], 3, 2.5)
        A.show(t, g, t + 1)
        g = _sub()
        for d in range(D):
            yy = y0 + d * ch * 2
            g.text(x0 + t * cw + 22, yy + ch + 26, f"{r['delta_f'][t][d]:g}", 11, C["model"], "middle")
            if r["mask"][t][d] == 0:
                g.text(x0 + t * cw + cw / 2, yy + 30, f"→{r['fwd'][t][d]:.3f}", 11, C["model"], "middle", "bold")
        A.show(t, g)
    # 뒤→앞
    for k, t in enumerate(range(T - 1, -1, -1)):
        g = _sub()
        g.rect(x0 + t * cw, y0 - 2, cw, D * ch * 2 - 4, "none", "#6a3fa0", 3, 2.5)
        A.show(T + k, g, T + k + 1)
        g = _sub()
        for d in range(D):
            yy = y0 + d * ch * 2
            g.text(x0 + t * cw + cw - 22, yy + ch + 26, f"/{r['delta_b'][t][d]:g}", 11, "#6a3fa0", "middle")
            if r["mask"][t][d] == 0:
                g.text(x0 + t * cw + cw / 2, yy + 42, f"←{r['bwd'][t][d]:.3f}", 11, "#6a3fa0", "middle", "bold")
        A.show(T + k, g)
    g = _sub()
    for t in range(T):
        for d in range(D):
            if r["mask"][t][d] == 0:
                yy = y0 + d * ch * 2
                g.rect(x0 + t * cw + 4, yy + 4, cw - 8, ch - 8, "#fff3df", C["model"], 3, 2)
                g.text(x0 + t * cw + cw / 2, yy + 22, f"{r['final'][t][d]:.3f}", 13, C["model"], "middle", "bold")
                g.text(x0 + t * cw + cw / 2, yy + 36, "평균", 9, INK2, "middle")
    A.show(2 * T, g)
    caps = [f"앞→뒤 τ{t + 1}: 기억으로 추정 x̂ 와 같은 시점 다른 변수로 추정 ẑ 를 반씩 섞어 빈 칸만 채움 (δ = 비어 있던 정도)" for t in range(T)]
    caps += [f"뒤→앞 τ{t + 1}: 거꾸로 같은 계산(별도 가중치)" for t in range(T - 1, -1, -1)]
    caps += ["최종 = 두 방향의 평균. 관측 칸은 원래 값 그대로. 숫자는 원본 BRITS 를 학습 전 가중치(시드 0)로 돌린 값 — 학습하면 달라짐"]
    A.captions(20, 362, caps, size=11)
    s.text(20, 290, "δ: 원본 delta_calc 규칙 그대로 계산(앞 변수의 δ 를 이어받는 원본의 특징 포함). 학습된 실제 값은 계산 따라가기 'Stage 1' 표.", 10, INK2)
    s.text(20, 306, "원본은 빈 칸 추정 c = |0.5·ẑ + 0.5·x̂| (절댓값, 학습되는 섞는 비율 α 는 계산만 하고 쓰지 않음)", 10, INK2)
    return s


# ============================================================================== 18. BIVA Stage 2 — 이동평균 창이 미끄러지며 추세를 만드는 순서
MAVG = {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 9.0], "k": 6}


def compute_moving_avg():
    """원본 BIVA.moving_avg(k, stride=1) 와 series_decomp 를 그대로 돌린 값."""
    import torch

    from models.biva_src.BIVA import series_decomp
    x = torch.tensor(MAVG["x"], dtype=torch.float32).view(1, -1, 1)
    res, trend = series_decomp(MAVG["k"])(x)
    k = MAVG["k"]
    front, end = (k - 1) // 2, (k - 1) // 2 if k % 2 else k // 2
    padded = [MAVG["x"][0]] * front + MAVG["x"] + [MAVG["x"][-1]] * end
    return {"trend": trend.view(-1).tolist(), "seasonal": res.view(-1).tolist(), "padded": padded, "front": front, "end": end}


def fig_moving_avg_anim():
    r = compute_moving_avg()
    x = MAVG["x"]; k = MAVG["k"]; T = len(x)
    s = Svg(760, 380, "BIVA 분해: 이동평균 창이 미끄러지며 추세 만들기 (움직이는 그림)")
    A = Anim(s, T + 1, step=1.7)
    cw, x0, y0 = 54, 80, 40
    P = r["padded"]
    for i, v in enumerate(P):
        pad = i < r["front"] or i >= r["front"] + T
        s.rect(x0 + i * cw, y0, cw - 4, 34, "#f3f4f6" if pad else "#ffffff", C["line"], 3)
        s.text(x0 + i * cw + (cw - 4) / 2, y0 + 22, f"{v:g}", 13, C["grey"] if pad else C["data"], "middle", "bold")
    s.text(x0 + (r["front"] * cw) / 2 - 2, y0 + 50, "앞에 복사", 9, C["grey"], "middle")
    s.text(x0 + (r["front"] + T) * cw + (r["end"] * cw) / 2 - 2, y0 + 50, "뒤에 복사", 9, C["grey"], "middle")
    win = _sub(); win.rect(x0 - 3, y0 - 4, k * cw + 2, 42, "none", C["model"], 4, 2.5)
    A.move(win, [(t * cw, 0) for t in range(T)] + [((T - 1) * cw, 0)])
    ax = Axes(s, 80, 110, 600, 170, (0.5, T + 0.5), (-2, 10), xticks=list(range(1, T + 1)), fmt=lambda v: f"τ{int(v)}", yticks=[-2, 0, 2, 4, 6, 8, 10])
    s.line(ax.X(0.5), ax.Y(0), ax.X(T + 0.5), ax.Y(0), C["ink2"], 0.8)
    for t in range(T):
        s.circle(ax.X(t + 1), ax.Y(x[t]), 5, C["data"])
        g = _sub()
        g.circle(ax.X(t + 1), ax.Y(r["trend"][t]), 6, C["model"])
        g.text(ax.X(t + 1) + 8, ax.Y(r["trend"][t]) - 6, f"{r['trend'][t]:.3f}", 10, C["model"], weight="bold")
        if t:
            g.line(ax.X(t), ax.Y(r["trend"][t - 1]), ax.X(t + 1), ax.Y(r["trend"][t]), C["model"], 1.5)
        A.show(t, g)
    g = _sub()
    for t in range(T):
        v = r["seasonal"][t]
        y1, y2 = ax.Y(max(v, 0)), ax.Y(min(v, 0))
        g.rect(ax.X(t + 1) + 12, y1, 14, y2 - y1, C["good"] if v > 0 else C["bad"], "none", 1)
        g.text(ax.X(t + 1) + 19, (y2 + 12) if v < 0 else (y1 - 4), f"{v:+.3f}", 9, INK, "middle")
    A.show(T, g)
    s.circle(110, 128, 5, C["data"]); s.text(120, 132, "원래 값", 10, INK2)
    s.circle(180, 128, 6, C["model"]); s.text(190, 132, "추세(평균)", 10, INK2)
    s.rect(260, 123, 10, 10, C["good"], "none", 1); s.text(276, 132, "주기 = 원래 − 추세", 10, INK2)
    caps = [f"τ{t + 1}: 창({k}칸)의 평균 = ({' + '.join(f'{v:g}' for v in P[t:t + k])}) ÷ {k} = {r['trend'][t]:.3f}" for t in range(T)]
    caps += ["주기 = 원래 − 추세 → 마지막 날의 급등(9)이 주기 +1.5 로 드러남. 숫자는 원본 moving_avg(AvgPool1d) 출력"]
    A.captions(20, 362, caps, size=11)
    return s


# ============================================================================== 19. BIVA 추세 경로 — 같은 시점의 변수 섞기(Conv1d, 커널 1) → 시간 평균
CONV = {"W": (0.5, -1.0, 2.0), "b": 0.1, "x": ((0.2, 0.1, 0.3), (0.4, 0.2, 0.2), (0.2, 0.1, 0.4))}


def compute_conv1d():
    """torch Conv1d(3→1, 커널 1) 와 Linear(시점 3→1, 가중치 1/3, 원본 초기값) 로 계산."""
    import torch
    conv = torch.nn.Conv1d(3, 1, kernel_size=1)
    lin = torch.nn.Linear(3, 1)
    with torch.no_grad():
        conv.weight[:] = torch.tensor(CONV["W"]).view(1, 3, 1); conv.bias[:] = CONV["b"]
        lin.weight[:] = 1 / 3; lin.bias[:] = 0.0
        x = torch.tensor(CONV["x"], dtype=torch.float32).T.unsqueeze(0)  # [1, 변수 3, 시점 3]
        a = conv(x)
        out = lin(a)
    return {"a": a.view(-1).tolist(), "out": float(out.view(-1)[0])}


def fig_conv1d_anim():
    r = compute_conv1d()
    s = Svg(760, 320, "Conv1d(커널 1): 같은 시점의 변수끼리 섞기 → 시간 평균 (움직이는 그림)")
    A = Anim(s, 4, step=2.3)
    x0, y0, cw, ch = 150, 40, 110, 34
    for t in range(3):
        s.text(x0 + t * cw + cw / 2, y0 - 8, f"시점 τ{t + 1}", 12, INK2, "middle")
    for v in range(3):
        s.text(x0 - 10, y0 + v * ch + 22, f"변수 {v + 1}  (W = {CONV['W'][v]:+g})", 11, INK, "end")
        for t in range(3):
            s.rect(x0 + t * cw + 2, y0 + v * ch + 2, cw - 4, ch - 4, "#ffffff", C["line"], 3)
            s.text(x0 + t * cw + cw / 2, y0 + v * ch + 22, f"{CONV['x'][t][v]:g}", 12, C["data"], "middle", "bold")
    s.text(x0 - 10, y0 + 3 * ch + 30, f"섞은 값 a  (+ b {CONV['b']:g})", 11, INK, "end", "bold")
    for t in range(3):
        g = _sub(); g.rect(x0 + t * cw, y0 - 1, cw, 3 * ch + 2, "none", C["model"], 3, 2.5)
        A.show(t, g, t + 1)
        g = _sub()
        xs = CONV["x"][t]
        g.rect(x0 + t * cw + 2, y0 + 3 * ch + 10, cw - 4, 32, "#fff3df", C["model"], 3)
        g.text(x0 + t * cw + cw / 2, y0 + 3 * ch + 31, f"{r['a'][t]:.3f}", 13, C["model"], "middle", "bold")
        A.show(t, g)
    g = _sub()
    g.line(x0, 232, x0 + 3 * cw, 232, C["line"], 1)
    g.text(x0 + 1.5 * cw, 256, f"시간 가중합(원본 초기값 1/3 씩, 학습되며 바뀜): ({r['a'][0]:.1f} + {r['a'][1]:.1f} + {r['a'][2]:.1f}) ÷ 3 = {r['out']:.3f}", 12, INK, "middle", "bold")
    A.show(3, g)
    s.text(500, 60, "원본 BIVA 의 추세 경로", 12, INK, weight="bold")
    s.text(500, 80, "Conv1d 3단(변수 V → V/4 → V/16 → 1)", 11, INK2)
    s.text(500, 96, "모두 커널 1 = 시점끼리는 섞지 않음, 사이에 활성화 없음", 11, INK2)
    s.text(500, 112, "→ Linear(창 길이 L → 1) 로 시점들을 가중합", 11, INK2)
    s.text(500, 128, "주기 경로도 같은 모양(입력 = 인코더 출력 + z)", 11, INK2)
    s.text(500, 150, "이 그림은 1단(3 → 1)만 작게 보임", 10, C["grey"])
    caps = [f"τ{t + 1}: {CONV['b']:g} + 0.5×{CONV['x'][t][0]:g} − 1×{CONV['x'][t][1]:g} + 2×{CONV['x'][t][2]:g} = {r['a'][t]:.3f} (같은 W 를 시점마다 그대로)" for t in range(3)]
    caps += [f"시점 3개를 가중합해 값 하나 {r['out']:.3f} — torch Conv1d·Linear 출력과 같음"]
    A.captions(20, 304, caps, size=11)
    return s


# ============================================================================== 20. BIVA Stage 3 — z 를 매번 새로 뽑는다 (σ ≥ 1)
VAE = {"pre_mu": -0.3, "pre_lv": 0.5, "eps": (0.4, -1.2, 0.9, -0.5, 1.7, -0.1, -1.6, 0.6)}


def compute_vae():
    import torch
    import torch.nn.functional as Fn
    mu = float(Fn.relu(torch.tensor(VAE["pre_mu"]))); lv = float(Fn.relu(torch.tensor(VAE["pre_lv"])))
    sd = math.exp(0.5 * lv)
    z = [mu + e * sd for e in VAE["eps"]]
    kld = -0.5 * (1 + lv - mu ** 2 - math.exp(lv))
    return {"mu": mu, "logvar": lv, "sigma": sd, "z": z, "kld": kld}


def fig_vae_anim():
    r = compute_vae()
    n = len(VAE["eps"])
    s = Svg(760, 340, "공통요인 z 뽑기 (움직이는 그림)")
    A = Anim(s, n + 2, step=1.4, hold=3.5)
    s.box(20, 40, 150, 56, ["인코더 출력 e", "(LSTM 4층)"], C["fill"], size=11)
    s.box(220, 20, 220, 44, [f"Linear → {VAE['pre_mu']:g} → ReLU → μ = {r['mu']:g}"], C["warm"], C["warmline"], 11, False)
    s.box(220, 74, 220, 44, [f"Linear → {VAE['pre_lv']:g} → ReLU → log σ² = {r['logvar']:g}"], C["warm"], C["warmline"], 11, False)
    s.line(170, 60, 218, 42, arrow=True); s.line(170, 76, 218, 96, arrow=True)
    s.box(490, 50, 240, 44, [f"σ = e^(½·{r['logvar']:g}) = {r['sigma']:.3f}  (ReLU 때문에 ≥ 1)"], "#e7eef8", C["data"], 11, False)
    s.line(440, 96, 488, 76, arrow=True)
    ax = Axes(s, 60, 150, 640, 110, (-4, 4), (0, 0.45), xticks=list(range(-4, 5)), yticks=[])
    s.text(712, 264, "z", 12, INK2)
    xs = np.linspace(-4, 4, 161)
    pdf = lambda v, m, sd: math.exp(-0.5 * ((v - m) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    s.poly([(ax.X(v), ax.Y(pdf(v, 0, 1))) for v in xs], C["grey"], 1.5, dash="4 3")
    s.poly([(ax.X(v), ax.Y(pdf(v, r["mu"], r["sigma"]))) for v in xs], C["model"], 2)
    s.text(ax.X(1.2), ax.Y(0.38), "N(0, 1)", 10, C["grey"]); s.text(ax.X(2.3), ax.Y(0.2), f"N(μ {r['mu']:g}, σ {r['sigma']:.3f}²)", 10, C["model"])
    for i, (e, z) in enumerate(zip(VAE["eps"], r["z"])):
        g = _sub()
        g.circle(ax.X(z), ax.Y(0.02) - (i % 3) * 9, 5, C["data"])
        A.show(i, g)
        t = _sub()
        t.text(380, 300, f"{i + 1}번째: ε = {e:+g} → z = {r['mu']:g} + ({e:+g}) × {r['sigma']:.3f} = {z:+.4f}", 13, C["data"], "middle", "bold")
        A.show(i, t, i + 1)
    t = _sub(); t.text(380, 300, f"KLD(한 칸) = −½ × (1 + {r['logvar']:g} − {r['mu']:g}² − e^{r['logvar']:g}) = {r['kld']:.4f}", 13, INK, "middle", "bold")
    A.show(n, t)
    caps = [f"ε 를 매번 새로 뽑아 z = μ + ε·σ — {i + 1}번째 점" for i in range(n)]
    caps[0] = "μ·log σ² 를 만들고 ε 를 뽑아 z = μ + ε·σ (첫 ε = 0.4 → z = 0.5136)"
    caps += ["KLD = 'N(0,1) 에서 얼마나 먼가'의 벌점. 원본은 예측 때도 z 를 뽑으므로 같은 창도 계산할 때마다 흔들림",
             "σ ≥ 1 이라 z 는 표준정규(점선)보다 넓게 퍼짐 — 원본 LSTM-VAE 의 ReLU 때문(22절 흔들림)"]
    A.captions(20, 326, caps, size=11)
    return s


# ============================================================================== 21. BIVA 전체 — 네 단계가 차례로 (구조도)
def fig_biva_anim():
    from models.biva import PARAMS as BP
    lat = int(BP["vae_latent_size"]["default"]); mav = BP["moving_avg"]["default"]
    s = Svg(780, 420, "BIVA 네 단계 (움직이는 그림)")
    A = Anim(s, 7, step=2.2, hold=3.5)
    B = lambda g, x, y, w, h, lines, fill=C["fill"], st=C["line"]: g.box(x, y, w, h, lines, fill, st, 11)
    stages = []
    g = _sub(); B(g, 10, 160, 120, 70, ["입력 창 [L × V]", "눈금 0~1 (학습 구간)", "빈 칸 = NaN"], C["fill"]); stages.append(g)
    g = _sub(); B(g, 160, 160, 120, 70, ["Stage 1 BRITS", "앞→뒤 + 뒤→앞", "빈 칸만 채움"], C["warm"], C["warmline"]); g.line(130, 195, 158, 195, arrow=True); stages.append(g)
    g = _sub(); B(g, 310, 160, 120, 70, ["Stage 2 분해", f"이동평균(기본 {mav})", "추세 / 주기"], C["warm"], C["warmline"]); g.line(280, 195, 308, 195, arrow=True); stages.append(g)
    g = _sub()
    B(g, 470, 60, 150, 76, ["추세 [L × V]", "Conv1d V→V/4→V/16→1", "(커널 1, 시점끼리 안 섞음)"])
    B(g, 650, 60, 120, 76, ["Linear L → 1", "(시간 가중합)", "→ 추세 예측"], C["out"], "#7fb894")
    g.line(430, 180, 468, 100, arrow=True); g.line(620, 98, 648, 98, arrow=True); stages.append(g)
    g = _sub()
    B(g, 470, 250, 150, 76, ["주기 [L × V]", "LSTM-VAE 인코더 4층", f"→ μ, log σ² (ReLU) [L × {lat}]"])
    B(g, 470, 336, 150, 50, [f"z = μ + ε·σ  [L × {lat}]", "(예측 때도 뽑음)"], "#e7eef8", C["data"])
    B(g, 650, 250, 120, 76, [f"[인코더 출력, z] 2×{lat}", f"Conv1d →{lat}→{lat // 2}→1", "Linear L → 1"], C["out"], "#7fb894")
    g.line(430, 210, 468, 288, arrow=True); g.line(620, 288, 648, 288, arrow=True); g.line(545, 326, 545, 334, C["data"], 1.2); stages.append(g)
    g = _sub()
    g.box(560, 160, 210, 60, ["Stage 4 합치기", "α·주기 예측 + (1 − α)·추세 예측", "→ 원래 단위로 되돌림 → ŷ"], C["out"], "#2f8f5b", 11)
    g.line(710, 136, 690, 158, C["good"], 1.5, None, True); g.line(710, 250, 690, 222, C["good"], 1.5, None, True); stages.append(g)
    g = _sub()
    g.rect(10, 20, 440, 110, "#fbe9e7", C["bad"], 4)
    g.text(22, 42, "학습할 때의 손실 (한 번에 더해 가중치를 고침, 따로 가중치 없음 — 원본)", 11, C["bad"], weight="bold")
    g.text(22, 64, "① 예측 오차 MSE(ŷ, 정답)", 11, INK)
    g.text(22, 82, "② BRITS 손실(관측 칸을 되살리는 오차 + 두 방향 일관성)", 11, INK)
    g.text(22, 100, "③ VAE 손실 = 0.5·재구성 MSE + 0.5·KLD", 11, INK)
    g.text(22, 118, "→ Adam, 조기종료(검증 손실), 학습률 일정, 가장 좋았던 가중치로 복원", 11, INK2)
    stages.append(g)
    for i, g in enumerate(stages):
        A.show(i, g)
    A.captions(20, 408, ["창: 최근 L 개 시점 × V 개 변수. 아직 시작 전인 변수의 칸은 비워 둠",
                             "Stage 1: BRITS 가 두 방향으로 읽으며 빈 칸만 채움(관측 칸은 그대로)",
                             "Stage 2: 변수마다 이동평균 = 추세, 원래 − 추세 = 주기",
                             "추세 경로: 변수를 섞어 줄이고(Conv1d) 시점을 가중합 → 추세 예측 하나",
                             f"주기 경로: LSTM-VAE 로 공통요인 z({lat}개)를 뽑고 인코더 출력과 이어 붙여 → 주기 예측 하나",
                             "Stage 4: 학습된 α 로 두 예측을 합쳐 ŷ (α 는 0~1 로 제한되지 않음)",
                             "학습: 세 손실의 합을 줄이도록 모든 가중치를 함께 고침 — 숫자는 원본 BIVA.py 구조 그대로"], size=11)
    return s


# ============================================================================== 23~26. 규칙 모델 — 실제 models/*.py 의 positions 로 계산
def _rule_prices(n=90, seed=3):
    """작은 합성 가격(오름 → 내림 → 오름). 규칙 그림 공용."""
    rng = np.random.default_rng(seed)
    drift = np.r_[np.full(30, 0.006), np.full(25, -0.008), np.full(n - 55, 0.006)]
    r = drift + rng.normal(0, 0.012, n)
    close = 100 * np.exp(np.cumsum(r))
    import pandas as pd
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"open": close * (1 + rng.normal(0, 0.002, n)), "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1.0}, index=idx)


RULES = {"ma_cross": {"fast": 5, "slow": 15}, "momentum": {"lookback": 10}, "rsi_rule": {"window": 6, "lower": 30.0, "upper": 70.0},
         "sox_gap_fade": {"q": 0.9, "window": 60, "mode": "avoid_top", "column": "SOX_pct"}}


def compute_rule(name: str) -> dict:
    import importlib

    import pandas as pd

    from models.base import resolve_params
    mod = importlib.import_module(f"models.{name}")
    p = resolve_params(mod.PARAMS, RULES[name])
    prices = _rule_prices(130 if name == "sox_gap_fade" else 90)
    c = prices["close"]
    out = {"close": c.tolist(), "params": p}
    if name == "sox_gap_fade":
        rng = np.random.default_rng(11)
        sox = pd.Series(rng.normal(0, 0.015, len(c)), index=c.index)
        sox.iloc[[70, 85, 96, 110, 121]] = [0.045, 0.05, 0.038, 0.055, 0.042]
        X = pd.DataFrame({"SOX_pct": sox})
        pos = mod.Model(p).positions(prices, X)
        thr = sox.shift(1).rolling(p["window"], min_periods=p["window"] // 2).quantile(p["q"])
        out.update(ind={"SOX 변화율": sox.tolist(), f"기준(과거 {p['window']}일 {p['q']:g} 분위)": thr.tolist()})
    else:
        pos = mod.Model(p).positions(prices)
        if name == "ma_cross":
            out.update(ind={f"짧은 평균 {p['fast']}일": c.rolling(p["fast"]).mean().tolist(), f"긴 평균 {p['slow']}일": c.rolling(p["slow"]).mean().tolist()})
        elif name == "momentum":
            out.update(ind={f"{p['lookback']}일 수익률": (c / c.shift(p["lookback"]) - 1).tolist()})
        else:
            out.update(ind={f"RSI({p['window']})": mod.rsi(c, p["window"]).tolist()})
    out["pos"] = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in pos]
    return out


def _rule_anim(name, title, caps, lower_panel: str):
    r = compute_rule(name)
    c = r["close"]; n = len(c); pos = r["pos"]
    H = 370 if lower_panel else 270
    s = Svg(760, H, title)
    A = Anim(s, 4, step=2.4, hold=3.5)
    lo, hi = min(c) * 0.97, max(c) * 1.03
    ax = Axes(s, 90, 20, 610, 150, (0, n - 1), (lo, hi), xticks=list(range(0, n, 10)), fmt=lambda v: f"{int(v)}일", yfmt=lambda v: f"{v:.0f}", ylabel="종가")
    g = _sub()
    g.poly([(ax.X(i), ax.Y(v)) for i, v in enumerate(c)], INK, 1.6)
    if name == "ma_cross":
        for (lab, vals), col in zip(r["ind"].items(), (C["model"], C["data"])):
            g.poly([(ax.X(i), ax.Y(v)) for i, v in enumerate(vals) if v == v], col, 1.8)
    if lower_panel:
        keys = list(r["ind"].keys())
        vals_all = [v for k in keys for v in r["ind"][k] if v == v]
        if name == "rsi_rule":
            yr = (0, 100)
        else:
            yr = (min(vals_all) * 1.1 if min(vals_all) < 0 else 0, max(vals_all) * 1.1)
        bx = Axes(s, 90, 200, 610, 80, (0, n - 1), yr, xticks=[], yticks=([30, 70] if name == "rsi_rule" else [0]), yfmt=lambda v: f"{v:g}")
        cols = (C["data"], C["model"])
        for (lab, vals), col in zip(r["ind"].items(), cols):
            if name == "sox_gap_fade" and col == C["data"]:
                for i, v in enumerate(vals):
                    if v == v:
                        g.line(bx.X(i), bx.Y(0), bx.X(i), bx.Y(v), col, 2)
            else:
                g.poly([(bx.X(i), bx.Y(v)) for i, v in enumerate(vals) if v == v], col, 1.6)
        if name == "rsi_rule":
            for lv in (30, 70):
                s.line(bx.X(0), bx.Y(lv), bx.X(n - 1), bx.Y(lv), C["grey"], 1, "4 3")
        if name == "momentum":
            s.line(bx.X(0), bx.Y(0), bx.X(n - 1), bx.Y(0), C["grey"], 1, "4 3")
    ys = 300 if lower_panel else 200
    for i, v in enumerate(pos):  # 포지션: t 결정 → t+1 시가부터 보유
        if v == 1 and i + 1 < n:
            g.rect(ax.X(i + 0.5), ys, ax.X(i + 1.5) - ax.X(i + 0.5), 14, C["good"], "none", 0)
        elif v == 0 and i + 1 < n:
            g.rect(ax.X(i + 0.5), ys, ax.X(i + 1.5) - ax.X(i + 0.5), 14, "#e3e6ea", "none", 0)
    A.reveal(g, 0, 10, 760, ys + 20, 0, 3)
    s.text(84, ys + 11, "포지션", 10, INK, "end")
    s.text(90, ys + 34, "띠: 초록 = 보유 · 회색 = 현금 (그날 결정 → 다음 날 시가부터)", 10, INK2)
    cur = _sub(); cur.line(ax.X(0), 18, ax.X(0), ys + 16, C["bad"], 1.2, "3 3")
    A.slide(cur, ax.X(n - 1) - ax.X(0), 0, 3)
    leg_x = 450
    for (lab, _), col in zip(r["ind"].items(), (C["model"], C["data"]) if name == "ma_cross" else (C["data"], C["model"])):
        s.line(leg_x, 18, leg_x + 18, 18, col, 2); s.text(leg_x + 22, 22, lab, 10, INK2); leg_x += 22 + 9 * len(lab) + 16
    A.captions(20, H - 10, caps, size=11)
    return s


def fig_rule_ma_anim():
    p = RULES["ma_cross"]
    return _rule_anim("ma_cross", "이동평균 교차 (움직이는 그림)", [
        "종가를 하루씩 따라가며 짧은 평균(5일)·긴 평균(15일)을 계산(그날까지의 종가만)",
        "짧은 평균 > 긴 평균 → 그날 장 마감 후 '보유' 결정 → 다음 날 시가에 삼(초록 띠는 하루 늦게 시작)",
        "짧은 평균 ≤ 긴 평균 → '현금' 결정. 긴 평균이 계산되기 전(처음 14일)은 결정 없음",
        f"초록 = 보유, 회색 = 현금 — models/ma_cross.py 의 positions 출력 그대로(fast {p['fast']}, slow {p['slow']})"], "")


def fig_rule_momentum_anim():
    p = RULES["momentum"]
    return _rule_anim("momentum", "모멘텀 (움직이는 그림)", [
        f"오늘 종가 ÷ {p['lookback']}일 전 종가 − 1 = {p['lookback']}일 수익률(아래 칸)",
        "수익률 > 0 → 다음 날 시가부터 보유",
        "수익률 ≤ 0 → 현금. 처음 10일은 비교할 과거가 없어 결정 없음",
        f"models/momentum.py 의 positions 출력 그대로(lookback {p['lookback']})"], "ind")


def fig_rule_rsi_anim():
    p = RULES["rsi_rule"]
    return _rule_anim("rsi_rule", "RSI 규칙 (움직이는 그림)", [
        f"RSI({p['window']}) = 오른 폭과 내린 폭의 Wilder 평균으로 0~100 (아래 칸, 점선 30·70)",
        "RSI < 30(많이 내림) → 사기로 결정 → 다음 날 시가에 삼",
        "RSI > 70(많이 오름) → 팔기로 결정. 30~70 사이는 어제 상태 유지",
        f"models/rsi_rule.py 의 positions 출력 그대로(창 {p['window']}, 문턱 {p['lower']:g}·{p['upper']:g})"], "ind")


def fig_rule_quantile_anim():
    p = RULES["sox_gap_fade"]
    return _rule_anim("sox_gap_fade", "분위수 규칙: SOX 급등 다음 날 쉬기 (움직이는 그림)", [
        f"SOX 하루 변화율(아래 막대)과 '어제까지 {p['window']}일' 의 {p['q']:g} 분위 기준(주황 선, 오늘은 포함 안 함)",
        "오늘 SOX 변화율이 기준보다 크면 → 다음 날은 현금(급등 뒤 되돌림 회피)",
        f"그 밖의 날은 보유. 처음 {p['window'] // 2}일은 기준을 만들 과거가 부족해 결정 없음",
        "models/sox_gap_fade.py 의 positions 출력 그대로(합성 SOX 자료)"], "ind")


# ============================================================================== 결정일 하루 — 엔진의 순서 (데이터 → 예측 → 체결 → 채점)
def fig_pipeline_anim():
    s = Svg(780, 430, "결정일 하루의 흐름 (움직이는 그림)")
    A = Anim(s, 7, step=2.4, hold=3.5)
    xs = {"t-1": 90, "t": 250, "t+1": 410, "t+2": 570, "t+1+k": 710}
    s.line(40, 300, 760, 300, C["ink2"], 1.2)
    for k, x in xs.items():
        s.line(x, 294, x, 306, C["ink2"]); s.text(x, 324, k if k != "t+1+k" else "t+1+k", 12, INK2, "middle")
    s.text(40, 346, "거래일 → (일간 기준. 주간·월간은 같은 순서를 주·월 단위로)", 10, INK2)
    steps = []
    g = _sub()
    g.box(20, 20, 220, 60, ["① 원자료", "가격(종가·시가) · 변수(발표일이 각자 다름)"], C["fill"], size=11)
    steps.append(g)
    g = _sub()
    g.box(20, 96, 220, 66, ["② 발표시차 맞춤", "변수 값은 '실제로 알려진 날'부터 씀", "(변수 사전 lag_days 만큼 뒤로)"], C["warm"], C["warmline"], 11)
    g.add(f'<path d="M{xs["t-1"]},{290} Q{xs["t-1"] + 70},{240} {xs["t"] - 6},{288}" fill="none" stroke="{C["warmline"]}" stroke-width="2" marker-end="url(#ah)"/>')
    g.text(xs["t-1"] + 70, 252, "발표 → 알려진 날", 10, C["warmline"], "middle")
    steps.append(g)
    g = _sub()
    g.box(260, 20, 230, 66, ["③ 결정일 t 의 한 행", "기술지표(t 종가까지) + 변수(t 까지 알려진 값)", "→ 학습 구간 통계로 눈금 맞춤(표준화·BIVA 는 0~1)"], C["fill"], size=10)
    g.circle(xs["t"], 300, 7, C["data"]); g.text(xs["t"] + 10, 316, "장 마감 후 결정", 10, C["data"], weight="bold")
    steps.append(g)
    g = _sub()
    g.box(260, 100, 230, 60, ["④ 모델 → 예측 ŷ", "(폴드의 학습 창으로 배운 모델)"], C["fill"], size=11)
    g.box(510, 20, 250, 60, ["⑤ 결정: ŷ > 임계치 → 보유(1)", "분류면 확률 > 0.5 + 임계치"], C["out"], "#7fb894", 11)
    g.line(490, 130, 508, 60, arrow=True)
    steps.append(g)
    g = _sub()
    g.rect(xs["t+1"], 262, xs["t+2"] - xs["t+1"], 22, C["good"], "none", 2, extra='fill-opacity="0.85"')
    g.text((xs["t+1"] + xs["t+2"]) / 2, 277, "보유 1기간", 11, "#ffffff", "middle", "bold")
    g.circle(xs["t+1"], 300, 7, C["model"]); g.text(xs["t+1"], 250, "t+1 시가 체결", 10, C["model"], "middle", "bold")
    g.box(510, 96, 250, 66, ["⑥ 매매 회계(엔진 backtest)", "수익 = 포지션 × (시가ₜ₊₂ ÷ 시가ₜ₊₁ − 1)", "− 포지션이 바뀔 때만 비용"], C["fill"], size=11)
    steps.append(g)
    g = _sub()
    g.add(f'<path d="M{xs["t+1"]},{334} Q{(xs["t+1"] + xs["t+1+k"]) / 2},{390} {xs["t+1+k"]},{334}" fill="none" stroke="{C["bad"]}" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#ah)"/>')
    g.text((xs["t+1"] + xs["t+1+k"]) / 2, 384, "정답 y = 시가ₜ₊₁₊ₖ ÷ 시가ₜ₊₁ − 1 (모델이 맞히는 값)", 10, C["bad"], "middle", "bold")
    g.box(260, 176, 500, 50, ["⑦ 채점: ŷ 와 y 비교(오차·DM) · 수익을 쌓아 성과(샤프·MDD) — 다음 날 t+1 이 새 결정일"], "#fbe9e7", C["bad"], 11, False)
    steps.append(g)
    for i, g in enumerate(steps):
        A.show(i, g)
    A.captions(20, 418, ["원자료: 가격과 변수. 변수마다 '언제 발표되나'가 다름",
                         "발표시차: 변수 값은 발표되어 알려진 날부터 씀(그 전 날짜에는 직전 값) — 미래 정보 누출 방지",
                         "결정일 t 장 마감 후, t 까지 알 수 있는 값만으로 한 행(창 모델은 최근 L 행)을 만들고 학습 구간 통계로 눈금을 맞춤",
                         "모델이 ŷ 를 내고 임계치와 비교해 보유/현금을 정함 (규칙 모델은 ③④ 없이 가격으로 바로 포지션)",
                         "체결은 다음 날(t+1) 시가. 엔진은 결정 하나로 1기간(t+1 → t+2 시가) 보유하고 매일 다시 판단",
                         "모델이 맞히는 정답은 t+1 시가 → t+1+k 시가 수익률. k > 1 이면 정답 기간과 보유 기간이 다름(문서화된 한계)",
                         "예측과 정답으로 오차·검정을, 수익으로 성과를 계산하고 다음 날로"], size=11)
    return s
