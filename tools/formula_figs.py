"""'수식 읽는 법'(docs/guide/35_수식_읽는_법.md)의 그림 — 작은 예시 숫자를 그대로 그린 SVG.

숫자는 문서의 '작은 예로 직접 풀기'와 같고(tests/test_formula_guide.py 가 같은 숫자를 코드로 확인), 그림도 여기서 다시 계산해 그린다.
build_all(out_dir) → <key>.svg 파일들. tools/build_guide.py 가 문서의 {{FIG:key}} 를 <img> 로 바꾼다. 일부는 움직이는 그림(SVG 애니메이션).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

FONT = "Pretendard, 'Noto Sans KR', 'Malgun Gothic', sans-serif"
C = {"ink": "#1c2027", "ink2": "#5b6372", "line": "#c9ced6", "data": "#3b6fb6", "model": "#d07a1a", "good": "#2f8f5b", "bad": "#b3261e",
     "fill": "#eef3f9", "warm": "#fdf6e3", "warmline": "#d6a63c", "grey": "#8f98a6", "out": "#e3f3f0"}


class Svg:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h, self.title, self.el = w, h, title, []

    def add(self, s: str):
        self.el.append(s)

    def text(self, x, y, t, size=12, color=C["ink"], anchor="start", weight="normal", italic=False):
        t = str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="{weight}"{" font-style=\"italic\"" if italic else ""}>{t}</text>')

    def line(self, x1, y1, x2, y2, color=C["ink2"], width=1.2, dash=None, arrow=False):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"'
                 + (f' stroke-dasharray="{dash}"' if dash else "") + (' marker-end="url(#ah)"' if arrow else "") + "/>")

    def poly(self, pts, color=C["data"], width=2, fill="none", dash=None):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.add(f'<polyline points="{p}" fill="{fill}" stroke="{color}" stroke-width="{width}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")

    def circle(self, x, y, r=4, color=C["data"], fill=None, width=1.5):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill or color}" stroke="{color}" stroke-width="{width}"/>')

    def rect(self, x, y, w, h, fill=C["fill"], stroke=C["line"], rx=4, width=1, extra=""):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" {extra}/>')

    def box(self, x, y, w, h, lines, fill=C["fill"], stroke=C["line"], size=12, bold_first=True):
        self.rect(x, y, w, h, fill, stroke)
        n = len(lines)
        y0 = y + h / 2 - (n - 1) * (size + 3) / 2 + size * 0.35
        for i, t in enumerate(lines):
            self.text(x + w / 2, y0 + i * (size + 3), t, size=size, anchor="middle", weight="bold" if (i == 0 and bold_first) else "normal")

    def render(self) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" font-family="{FONT}">'
                f'<title>{self.title}</title><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{C["ink2"]}"/></marker></defs><rect width="100%" height="100%" fill="#ffffff"/>' + "".join(self.el) + "</svg>")


class Axes:
    """x·y 값 → 화면 좌표. 눈금·격자·축 이름."""

    def __init__(self, s: Svg, x0, y0, w, h, xr, yr, xlabel="", ylabel="", xticks=None, yticks=None, fmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}"):
        self.s, self.x0, self.y0, self.w, self.h, self.xr, self.yr = s, x0, y0, w, h, xr, yr
        s.line(x0, y0 + h, x0 + w, y0 + h, C["ink2"])
        if h:
            s.line(x0, y0, x0, y0 + h, C["ink2"])
        for v in (xticks if xticks is not None else np.linspace(*xr, 5)):
            X = self.X(v); s.line(X, y0 + h, X, y0 + h + 4, C["ink2"]); s.text(X, y0 + h + 16, fmt(v), 11, C["ink2"], "middle")
        for v in ([] if h == 0 else (yticks if yticks is not None else np.linspace(*yr, 5))):  # h == 0: 수직선(눈금 없음)
            Y = self.Y(v); s.line(x0 - 4, Y, x0, Y, C["ink2"]); s.line(x0, Y, x0 + w, Y, "#eef0f3", 1); s.text(x0 - 7, Y + 4, yfmt(v), 11, C["ink2"], "end")
        if xlabel:
            s.text(x0 + w / 2, y0 + h + 32, xlabel, 12, C["ink2"], "middle")
        if ylabel:
            s.text(x0, y0 - 8, ylabel, 12, C["ink2"], "start")

    def X(self, v):
        return self.x0 + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def Y(self, v):
        return self.y0 + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h


def _sig(a):
    return 1 / (1 + math.exp(-a))


# ------------------------------------------------------------------------------------------------ 공통
def fig_standardize():
    s = Svg(720, 230, "표준화: 평소보다 몇 칸 큰가")
    x = np.array([2.0, 4, 6]); mu, sd = x.mean(), x.std()
    a = Axes(s, 60, 40, 620, 0, (0, 9), (0, 1), xticks=range(0, 10))
    s.text(60, 28, "원래 눈금 (학습 표 값 2·4·6, 테스트일 값 7)", 12, C["ink2"])
    s.rect(a.X(mu - sd), 30, a.X(mu + sd) - a.X(mu - sd), 20, "#e7eef8", "none", 2)
    s.text(a.X(mu), 72, f"평균 {mu:g} ± 표준편차 {sd:.3f}", 11, C["data"], "middle")
    for v in x:
        s.circle(a.X(v), 40, 5, C["data"])
    s.circle(a.X(7), 40, 6, C["model"])
    s.line(a.X(mu), 30, a.X(mu), 50, C["data"], 2)
    b = Axes(s, 60, 150, 620, 0, ((0 - mu) / sd, (9 - mu) / sd), (0, 1), xticks=[-2, -1, 0, 1, 2, 3], fmt=lambda v: f"{v:+g}칸" if v else "0칸")
    s.text(60, 138, "칸 눈금 z = (x − 평균) ÷ 표준편차", 12, C["ink2"])
    for v in x:
        s.circle(b.X((v - mu) / sd), 150, 5, C["data"])
        s.line(a.X(v), 46, b.X((v - mu) / sd), 144, C["line"], 1, "3 3")
    z7 = (7 - mu) / sd
    s.circle(b.X(z7), 150, 6, C["model"]); s.line(a.X(7), 46, b.X(z7), 144, C["model"], 1.2, "3 3")
    s.text(b.X(z7) - 8, 132, f"7 → z = (7 − 4) ÷ {sd:.3f} = {z7:.3f}칸", 12, C["model"], "end", "bold")
    s.text(60, 210, "파란 점 = 학습 표(평균·표준편차를 여기서만 계산) · 주황 점 = 테스트일(학습 표의 평균·표준편차로 바꿈)", 11, C["ink2"])
    return s


def fig_minmax():
    s = Svg(720, 200, "최소·최대 눈금")
    a = Axes(s, 60, 40, 620, 0, (1, 8), (0, 1), xticks=range(1, 9))
    s.text(60, 28, "원래 눈금 — 학습 최소 2, 최대 6", 12, C["ink2"])
    b = Axes(s, 60, 130, 620, 0, ((1 - 2) / 4, (8 - 2) / 4), (0, 1), xticks=[-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5], fmt=lambda v: f"{v:g}")
    s.text(60, 118, "0~1 눈금 x′ = (x − 2) ÷ (6 − 2)", 12, C["ink2"])
    s.rect(b.X(0), 122, b.X(1) - b.X(0), 16, "#e7eef8", "none", 2)
    for v, col in [(2, C["data"]), (6, C["data"]), (5, C["good"]), (7, C["model"])]:
        s.circle(a.X(v), 40, 5, col); s.circle(b.X((v - 2) / 4), 130, 5, col); s.line(a.X(v), 46, b.X((v - 2) / 4), 124, col, 1, "3 3")
        s.text(b.X((v - 2) / 4), 160, f"{v}→{(v - 2) / 4:g}", 11, col, "middle", "bold")
    s.text(60, 188, "학습 때 본 범위(파란 띠) 밖의 값은 0~1 밖으로 나갑니다(7 → 1.25)", 11, C["ink2"])
    return s


def fig_target():
    s = Svg(720, 170, "정답(타겟)")
    xs = [40, 270, 500]
    s.box(xs[0], 40, 180, 64, ["결정일 t", "장 마감 후 판단", "(종가에는 못 삼)"], C["fill"])
    s.box(xs[1], 40, 180, 64, ["t+1 시가 100", "산다(진입)"], C["warm"], C["warmline"])
    s.box(xs[2], 40, 180, 64, ["t+1+k 시가 103", "판다(청산), k = 1"], C["out"], "#7fb894")
    s.line(220, 72, 268, 72, arrow=True); s.line(450, 72, 498, 72, arrow=True)
    s.text(360, 140, "정답 r = 103 ÷ 100 − 1 = 0.03 (3%) · 로그 ln(1.03) = 0.02956 · 방향 = 1(오름)", 13, C["ink"], "middle", "bold")
    return s


def fig_decision():
    s = Svg(720, 190, "결정")
    s.box(30, 70, 150, 50, ["예측 ŷ = 0.0016"], C["fill"])
    s.add(f'<polygon points="260,95 340,55 420,95 340,135" fill="{C["warm"]}" stroke="{C["warmline"]}"/>')
    s.text(340, 92, "ŷ > 임계치(0)?", 12, anchor="middle", weight="bold"); s.text(340, 108, "확률이면 > 0.5+임계치", 10, C["ink2"], "middle")
    s.line(180, 95, 258, 95, arrow=True)
    s.box(500, 30, 190, 52, ["예 → 보유(1)", "다음 날 시가에 사거나 유지"], C["out"], "#7fb894", 11)
    s.box(500, 112, 190, 52, ["아니오 → 현금(0)", "다음 날 시가에 팔거나 유지"], "#f3f4f6", C["grey"], 11)
    s.line(420, 90, 498, 58, arrow=True); s.line(420, 100, 498, 136, arrow=True)
    s.text(360, 182, "포지션이 바뀌는 날만 비용: 100만 원 × (0.015% + 0.05%) = 650원", 11, C["ink2"], "middle")
    return s


def fig_compound():
    s = Svg(720, 140, "복리로 묶기")
    vals = [1.0]
    for r in (0.01, -0.02, 0.03):
        vals.append(vals[-1] * (1 + r))
    xs = [20, 190, 360, 530]
    for i, v in enumerate(vals):
        s.box(xs[i], 40, 120, 44, [f"{v:.6g}"], C["out"] if i == 3 else C["fill"], size=13)
    for i, r in enumerate((0.01, -0.02, 0.03)):
        s.line(xs[i] + 120, 62, xs[i + 1] - 2, 62, arrow=True); s.text((xs[i] + 120 + xs[i + 1]) / 2, 54, f"× {1 + r:g}", 12, C["model"], "middle", "bold")
    s.text(360, 118, f"묶은 수익률 = {vals[-1]:.6f} − 1 = {vals[-1] - 1:.6f} (그냥 더한 0.02 와 다름)", 13, anchor="middle", weight="bold")
    return s


# ------------------------------------------------------------------------------------------------ Ridge
def fig_ridge_sum():
    s = Svg(720, 240, "가중합(기여)")
    parts = [("기본값 b", 0.05), ("0.1 × 0.95칸", 0.095), ("−0.05 × 0.8칸", -0.04)]
    total = sum(v for _, v in parts)
    a = Axes(s, 80, 30, 600, 160, (0, 4), (-0.05, 0.15), ylabel="예측에 더해지는 값", xticks=[], yticks=[-0.05, 0, 0.05, 0.1, 0.15])
    cum = 0.0
    for i, (lab, v) in enumerate(parts + [("합계 = 예측", None)]):
        x = a.X(i + 0.15); w = a.X(i + 0.85) - x
        if v is None:
            y1, y0, col = a.Y(total), a.Y(0), C["good"]
            s.text(x + w / 2, a.Y(total) - 6, f"{total:.3f}", 12, C["good"], "middle", "bold")
        else:
            y0, y1 = a.Y(cum), a.Y(cum + v); col = C["data"] if v >= 0 else C["bad"]
            s.text(x + w / 2, min(y0, y1) - 6, f"{v:+.3f}", 12, col, "middle", "bold"); cum += v
        s.rect(x, min(y0, y1), w, abs(y1 - y0), col, "none", 2)
        s.text(x + w / 2, 222, lab, 11, C["ink2"], "middle")
    s.line(a.X(0), a.Y(0), a.X(4), a.Y(0), C["ink2"], 1)
    return s


def fig_ridge_fit():
    s = Svg(760, 320, "Ridge 학습")
    z = np.array([-1.0, 0, 1]); y = np.array([1.0, 2, 6]); yb = y.mean()
    a = Axes(s, 60, 40, 280, 200, (-1.5, 1.5), (0, 7), xlabel="재료 z (칸)", ylabel="정답 y", xticks=[-1, 0, 1], yticks=[0, 2, 4, 6])
    for w, col, lab, dash in [(2.5, C["grey"], "α=0: w=2.5", "5 4"), (5 / 3, C["model"], "α=1: w=1.667", None), (5 / 102, C["data"], "α=100: w=0.049", "2 3")]:
        s.poly([(a.X(-1.5), a.Y(yb + w * -1.5)), (a.X(1.5), a.Y(yb + w * 1.5))], col, 2, dash=dash)
        s.text(a.X(-1.45), {2.5: a.Y(6.6), 5 / 3: a.Y(6.0), 5 / 102: a.Y(5.4)}[w], lab, 11, col, "start", "bold")
    for zi, yi in zip(z, y):
        s.circle(a.X(zi), a.Y(yi), 5, C["ink"])
    s.text(200, 308, "예측선 = 3 + w × z (점 = 학습 표 3일)", 11, C["ink2"], "middle")
    b = Axes(s, 440, 40, 290, 200, (0, 3), (0, 16), xlabel="비율 w", ylabel="점수 = 틀린 정도² 합 + α·w²", xticks=[0, 0.5, 1, 1.5, 2, 2.5, 3], yticks=[0, 4, 8, 12, 16])
    ws = np.linspace(0, 3, 61)
    for alpha, col, dash in [(0, C["grey"], "5 4"), (1, C["model"], None)]:
        sc = [float(((y - (yb + w * z)) ** 2).sum() + alpha * w * w) for w in ws]
        s.poly([(b.X(w), b.Y(v)) for w, v in zip(ws, sc)], col, 2, dash=dash)
        wm = 5 / (2 + alpha); vm = float(((y - (yb + wm * z)) ** 2).sum() + alpha * wm * wm)
        s.circle(b.X(wm), b.Y(vm), 5, col); s.text(b.X(wm), b.Y(vm) + 18, f"α={alpha}: 바닥 w={wm:.3g}, 점수 {vm:.3g}", 11, col, "middle", "bold")
    s.text(585, 308, "그릇 모양의 바닥 = 답. α 를 키우면 바닥이 0 쪽으로", 11, C["ink2"], "middle")
    return s


def fig_logistic():
    s = Svg(560, 270, "시그모이드")
    a = Axes(s, 60, 30, 460, 190, (-6, 6), (0, 1), xlabel="점수", ylabel="확률", xticks=[-6, -4, -2, 0, 2, 4, 6], yticks=[0, 0.25, 0.5, 0.75, 1])
    xs = np.linspace(-6, 6, 121)
    s.poly([(a.X(v), a.Y(_sig(v))) for v in xs], C["data"], 2.5)
    for v in (-2, 0, 2):
        s.circle(a.X(v), a.Y(_sig(v)), 5, C["model"]); s.text(a.X(v) + 8, a.Y(_sig(v)) - 8 if v <= 0 else a.Y(_sig(v)) + 16, f"σ({v}) = {_sig(v):.4f}", 11, C["model"], weight="bold")
    s.line(a.X(-6), a.Y(0.5), a.X(6), a.Y(0.5), C["grey"], 1, "4 3")
    return s


# ------------------------------------------------------------------------------------------------ LightGBM
def fig_lgbm():
    s = Svg(760, 300, "LightGBM 나무 하나")
    s.box(250, 20, 200, 50, ["질문: z ≤ 0 ?", "이득 16.67 (z ≤ −1 은 6.75)"], C["warm"], C["warmline"], 11)
    s.box(80, 130, 220, 70, ["예 → 왼쪽 잎 (1·2일)", "기울기 합 G = 3 + 2 = 5", "잎 값 = −5 ÷ (2 + 1) = −1.667"], C["fill"], size=11)
    s.box(400, 130, 220, 70, ["아니오 → 오른쪽 잎 (3·4일)", "G = −2 − 3 = −5", "잎 값 = +5 ÷ (2 + 1) = +1.667"], C["fill"], size=11)
    s.line(320, 70, 200, 128, arrow=True); s.line(380, 70, 500, 128, arrow=True)
    s.text(240, 100, "예", 12, C["good"], weight="bold"); s.text(450, 100, "아니오", 12, C["bad"], weight="bold")
    a = Axes(s, 80, 225, 600, 50, (0.5, 4.5), (0, 8), xticks=[], yticks=[0, 4, 8])
    y = [1, 2, 6, 7]; new = [3.833, 3.833, 4.167, 4.167]
    for i in range(4):
        X = a.X(i + 1)
        s.circle(X, a.Y(y[i]), 4, C["ink"]); s.line(X - 18, a.Y(4), X + 18, a.Y(4), C["grey"], 2); s.line(X - 18, a.Y(new[i]), X + 18, a.Y(new[i]), C["model"], 2.5)
        s.text(X, 292, f"{i + 1}일 z={[-1, -0.5, 0.5, 1][i]}", 10, C["ink2"], "middle")
    s.text(700, 240, "● 정답", 11, C["ink"], "end"); s.text(700, 255, "— 시작 4", 11, C["grey"], "end"); s.text(700, 270, "— 나무 1 뒤 (×0.1)", 11, C["model"], "end")
    return s


# ------------------------------------------------------------------------------------------------ ARIMA / Kalman / SARIMA
def fig_arima():
    s = Svg(720, 270, "ARIMA 재귀 예측")
    c, phi, th = 0.001, 0.5, 0.3
    vals = [c + phi * 0.02 + th * 0.005]
    for _ in range(5):
        vals.append(c + phi * vals[-1])
    a = Axes(s, 70, 30, 610, 190, (-0.5, 6.5), (0, 0.022), xlabel="기간", ylabel="변화율", xticks=range(0, 7), yticks=[0, 0.005, 0.01, 0.015, 0.02], fmt=lambda v: f"{v:g}" if v < 1e-9 or v >= 1 else f"{v:.3f}")
    s.line(a.X(-0.5), a.Y(0.002), a.X(6.5), a.Y(0.002), C["good"], 1.2, "5 4"); s.text(a.X(1.2), a.Y(0.002) + 14, "장기 평균 c ÷ (1 − φ) = 0.002", 11, C["good"], "start", "bold")
    s.circle(a.X(0), a.Y(0.02), 6, C["data"]); s.text(a.X(0) + 9, a.Y(0.02) + 4, "오늘 r = 0.02 (충격 ε = 0.005)", 11, C["data"])
    pts = [(a.X(i + 1), a.Y(v)) for i, v in enumerate(vals)]
    s.poly([(a.X(0), a.Y(0.02))] + pts, C["model"], 2, dash="4 3")
    for i, (X, Y) in enumerate(pts):
        s.circle(X, Y, 5, C["model"]); s.text(X + 6, Y - 8, f"{vals[i]:.4g}", 11, C["model"])
    s.text(375, 262, "t+1 = 0.001 + 0.5×0.02 + 0.3×0.005 = 0.0125 → 이후 미래 충격 0 → 0.5배씩 평균 쪽으로", 11, C["ink2"], "middle")
    return s


def fig_kalman():
    s = Svg(720, 230, "칼만 필터")
    s.box(40, 80, 140, 56, ["상태 a", "(지난 값·충격)"], C["fill"])
    s.box(240, 20, 170, 56, ["예측 = Z·(T·a + c)", "다음 값을 미리 계산"], C["fill"], size=11)
    s.box(240, 140, 170, 56, ["실제 값이 들어옴", "오차 = 실제 − 예측"], C["warm"], C["warmline"], 11)
    s.box(470, 80, 210, 56, ["상태 고치기", "a ← a + K·오차"], C["out"], "#7fb894")
    s.line(180, 100, 238, 54, arrow=True); s.line(325, 76, 325, 138, arrow=True); s.line(410, 168, 468, 120, arrow=True); s.line(470, 108, 182, 108, C["grey"], 1.2, "5 4", True)
    s.text(325, 222, "결정일까지 반복 → 그 뒤(미래)는 실제 값이 없어 '예측 → 다음 상태'만 반복", 12, C["ink2"], "middle")
    return s


def fig_sarima():
    s = Svg(720, 220, "SARIMA 펼치기")
    s.text(20, 26, "(1 − 0.5L)(1 − L⁴) = 1 − 0.5L − L⁴ + 0.5L⁵  →  r̂(t+1) = 0.5·r(t) + 1·r(t−3) − 0.5·r(t−4)", 13, weight="bold")
    xs = {k: 70 + i * 110 for i, k in enumerate(["t−4", "t−3", "t−2", "t−1", "t", "t+1"])}
    vals = {"t−4": 0.03, "t−3": 0.01, "t−2": None, "t−1": None, "t": 0.02}
    for k, X in xs.items():
        s.box(X - 40, 130, 80, 40, [k, "" if vals.get(k) is None else f"{vals[k]}"] if k != "t+1" else ["t+1", "0.005"], C["out"] if k == "t+1" else C["fill"], size=11)
    for k, coef, col in [("t", "+0.5", C["data"]), ("t−3", "+1", C["data"]), ("t−4", "−0.5", C["bad"])]:
        X = xs[k]; Xe = xs["t+1"]
        s.add(f'<path d="M{X},128 Q{(X + Xe) / 2},{60 - (Xe - X) / 12} {Xe - 6},128" fill="none" stroke="{col}" stroke-width="1.6" marker-end="url(#ah)"/>')
        s.text(X + 6, 112, f"× {coef}", 12, col, "start", "bold")
    s.text(360, 205, "0.5 × 0.02 + 1 × 0.01 − 0.5 × 0.03 = 0.005", 13, anchor="middle", weight="bold")
    return s


# ------------------------------------------------------------------------------------------------ VECM
def fig_vecm():
    s = Svg(560, 330, "VECM 되돌림")
    a = Axes(s, 70, 30, 440, 240, (49, 52), (98, 106), xlabel="물가", ylabel="집값", xticks=[49, 50, 51, 52], yticks=[98, 100, 102, 104, 106])
    s.poly([(a.X(49), a.Y(98)), (a.X(52), a.Y(104))], C["good"], 2); s.text(a.X(49.7), a.Y(98.7), "균형선: 집값 = 2 × 물가 (e = 0)", 11, C["good"], "start", "bold")
    A, B = (50, 104), (50.4, 103)
    s.line(a.X(A[0]), a.Y(A[1]), a.X(A[0]), a.Y(100), C["bad"], 1.2, "3 3"); s.text(a.X(A[0]) - 6, a.Y(102), "e = 4", 12, C["bad"], "end", "bold")
    s.circle(a.X(A[0]), a.Y(A[1]), 6, C["bad"]); s.text(a.X(A[0]) + 8, a.Y(A[1]) - 6, "지금 (50, 104)", 11, C["bad"], weight="bold")
    s.line(a.X(A[0]), a.Y(A[1]), a.X(B[0]), a.Y(B[1]), C["model"], 2, None, True)
    s.circle(a.X(B[0]), a.Y(B[1]), 6, C["model"]); s.text(a.X(B[0]) + 8, a.Y(B[1]) + 14, "다음 (50.4, 103) · e = 2.2", 11, C["model"], weight="bold")
    s.text(290, 318, "되돌림 α·e = (−0.25 × 4, +0.1 × 4) = (−1, +0.4)", 12, anchor="middle")
    return s


# ------------------------------------------------------------------------------------------------ 신경망
def fig_nn_train():
    s = Svg(560, 300, "기울기로 가중치 고치기")
    a = Axes(s, 60, 30, 460, 200, (0.6, 2.4), (0, 3.5), xlabel="가중치 w", ylabel="손실 (2w − 3)²", xticks=[0.6, 1, 1.4, 2, 2.4], yticks=[0, 1, 2, 3])
    ws = np.linspace(0.6, 2.4, 91)
    s.poly([(a.X(w), a.Y((2 * w - 3) ** 2)) for w in ws], C["data"], 2.5)
    s.circle(a.X(1), a.Y(1), 6, C["bad"]); s.text(a.X(1) + 8, a.Y(1) - 8, "w=1: 손실 1, 기울기 −4", 11, C["bad"], weight="bold")
    s.line(a.X(1), a.Y(1) + 2, a.X(1.4), a.Y(0.04) - 6, C["model"], 2, None, True)
    s.circle(a.X(1.4), a.Y(0.04), 6, C["good"]); s.text(a.X(1.4) + 8, a.Y(0.04) - 10, "w=1.4: 손실 0.04", 11, C["good"], weight="bold")
    s.text(290, 292, "w ← w − 학습률(0.1) × 기울기(−4) = 1.4 — 기울기의 반대쪽으로 한 걸음 (바닥은 w = 1.5)", 11, C["ink2"], "middle")
    return s


def fig_lstm_cell():
    s = Svg(760, 300, "LSTM 셀")
    f, i, g, o = _sig(2), _sig(0), math.tanh(0.5), _sig(1)
    c = f * 0.5 + i * g; h = o * math.tanh(c)
    s.text(20, 24, "지난 기억 c = 0.5, 지난 은닉 h = 0, 입력 x = 1 일 때 (값은 문서 예시)", 12, C["ink2"])
    s.line(40, 70, 720, 70, C["ink2"], 2, None, True); s.text(46, 62, "기억 줄 c: 0.5", 12, weight="bold")
    s.text(700, 62, f"새 c = {c:.4f}", 12, C["good"], "end", "bold")
    for X, lab, val, col in [(150, "잊기 문 f", f, C["warmline"]), (300, "입력 문 i", i, C["warmline"]), (430, "후보 g", g, C["data"]), (590, "출력 문 o", o, C["warmline"])]:
        s.box(X - 55, 170, 110, 48, [lab, f"{'σ' if lab[-1] != 'g' else 'tanh'}(…) = {val:.4f}"], C["warm"] if lab[-1] != "g" else C["fill"], C["warmline"] if lab[-1] != "g" else C["line"], 11)
    s.circle(150, 70, 13, "#ffffff", 1.5); s.text(150, 75, "×", 14, anchor="middle", weight="bold"); s.line(150, 168, 150, 84, arrow=True)
    s.text(160, 110, f"0.5 × {f:.4f} = {f * 0.5:.4f}", 11, C["ink2"])
    s.circle(365, 70, 13, "#ffffff", 1.5); s.text(365, 75, "+", 14, anchor="middle", weight="bold")
    s.circle(365, 130, 11, "#ffffff", 1.5); s.text(365, 135, "×", 13, anchor="middle", weight="bold")
    s.line(300, 168, 360, 140, arrow=True); s.line(430, 168, 370, 140, arrow=True); s.line(365, 119, 365, 84, arrow=True)
    s.text(375, 112, f"{i:.1f} × {g:.4f} = {i * g:.4f}", 11, C["ink2"])
    s.line(520, 70, 520, 118, C["ink2"]); s.box(480, 118, 80, 26, [f"tanh(c)={math.tanh(c):.4f}"], "#ffffff", C["line"], 10, False)
    s.circle(640, 131, 11, "#ffffff", 1.5); s.text(640, 136, "×", 13, anchor="middle", weight="bold"); s.line(560, 131, 627, 131, arrow=True); s.line(600, 168, 635, 142, arrow=True)
    s.line(651, 131, 720, 131, C["good"], 2, None, True); s.text(722, 150, f"h = {h:.4f}", 13, C["good"], "end", "bold")
    s.text(380, 250, f"c′ = f×c + i×g = {f * 0.5:.4f} + {i * g:.4f} = {c:.4f}    h′ = o × tanh(c′) = {o:.4f} × {math.tanh(c):.4f} = {h:.4f}", 12, anchor="middle", weight="bold")
    s.text(380, 276, "f 의 (…) = 1.5×x + 0.4×h + 0.5 = 2 (나머지 문도 같은 방식, 각자 다른 가중치)", 11, C["ink2"], "middle")
    return s


def fig_gru_cell():
    s = Svg(720, 230, "GRU 셀")
    r, z = _sig(0), _sig(1); n = math.tanh(0.6 + r * 0.8); h = (1 - z) * n + z * 0.2
    s.box(30, 40, 150, 50, ["재설정 r = σ(0)", f"= {r:.4f}"], C["warm"], C["warmline"], 11)
    s.box(30, 130, 150, 50, ["갱신 z = σ(1)", f"= {z:.4f}"], C["warm"], C["warmline"], 11)
    s.box(250, 40, 210, 50, ["후보 n = tanh(0.6 + r×0.8)", f"= tanh(1.0) = {n:.4f}"], C["fill"], size=11)
    s.box(250, 130, 210, 50, ["지난 은닉 h = 0.2"], C["fill"], size=11)
    s.box(520, 80, 170, 60, ["새 h = (1−z)·n + z·h", f"= {(1 - z) * n:.4f} + {z * 0.2:.4f}", f"= {h:.4f}"], C["out"], "#7fb894", 11)
    s.line(180, 65, 248, 65, arrow=True); s.line(460, 65, 518, 100, arrow=True); s.line(460, 155, 518, 122, arrow=True); s.line(180, 155, 248, 155, C["grey"], 1, "4 3")
    s.text(360, 215, f"z 가 클수록 지난 기억을 많이 유지: 1 − z = {1 - z:.4f} 만큼만 새 후보", 12, C["ink2"], "middle")
    return s


def fig_rnn_head():
    s = Svg(720, 170, "마지막 은닉으로 예측")
    s.box(20, 30, 130, 40, ["h₁ = 0.43"], C["fill"]); s.box(20, 90, 130, 40, ["h₂ = −0.2"], C["fill"])
    s.box(220, 55, 170, 50, ["점수 = 0.1 + 0.5×0.43", "+ 1.0×(−0.2) = 0.115"], C["warm"], C["warmline"], 11)
    s.box(460, 55, 240, 50, ["ŷ = 0.115 × 0.02 + 0.001", "= 0.0033 (0.33%)"], C["out"], "#7fb894", 11)
    s.line(150, 50, 218, 72, arrow=True); s.line(150, 110, 218, 88, arrow=True); s.line(390, 80, 458, 80, arrow=True)
    s.text(184, 46, "×0.5", 11, C["model"], weight="bold"); s.text(184, 124, "×1.0", 11, C["model"], weight="bold")
    s.text(360, 158, "정답을 학습 구간 평균 0.001·표준편차 0.02 로 표준화해 배웠으므로 되돌림", 11, C["ink2"], "middle")
    return s


# ------------------------------------------------------------------------------------------------ BIVA
def fig_brits():
    s = Svg(760, 300, "BRITS 빈 칸 채우기")
    T, V = 6, 4
    miss = {(0, 2), (1, 2), (2, 2), (3, 0), (4, 3), (5, 3)}
    x0, y0, cw, ch = 140, 50, 70, 34
    for v in range(V):
        s.text(x0 - 10, y0 + v * ch + 22, f"변수 {v + 1}", 12, C["ink2"], "end")
    for t in range(T):
        s.text(x0 + t * cw + cw / 2, y0 - 10, f"τ={t + 1}", 11, C["ink2"], "middle")
        for v in range(V):
            m = (t, v) in miss
            s.rect(x0 + t * cw + 2, y0 + v * ch + 2, cw - 4, ch - 4, "#fff4e0" if m else C["fill"], C["warmline"] if m else C["line"], 3, 1.5 if m else 1)
            s.text(x0 + t * cw + cw / 2, y0 + v * ch + 22, "빈 칸 → ĉ" if m else "관측", 10, C["model"] if m else C["ink2"], "middle", "bold" if m else "normal")
    yb = y0 + V * ch + 18
    s.line(x0, yb, x0 + T * cw, yb, C["data"], 2, None, True); s.text(x0 + T * cw + 6, yb + 4, "앞→뒤 RITS", 11, C["data"], weight="bold")
    s.line(x0 + T * cw, yb + 22, x0, yb + 22, C["model"], 2, None, True); s.text(x0 + T * cw + 6, yb + 26, "뒤→앞 RITS", 11, C["model"], weight="bold")
    s.text(380, yb + 58, "시점마다: 기억 추정 x̂ 와 다른 변수 추정 ẑ 를 반씩 → ĉ = |0.5·ẑ + 0.5·x̂| (예: 0.4·0.6 → 0.5)", 12, anchor="middle")
    s.text(380, yb + 78, "관측 칸은 그대로 · 두 방향 결과의 평균이 최종 (예: 0.5 와 0.7 → 0.6)", 12, C["ink2"], "middle")
    return s


def fig_moving_avg():
    s = Svg(720, 340, "이동평균 분해")
    x = np.array([1, 2, 3, 4, 5, 9.0]); k = 6
    pad = np.r_[[x[0]] * 2, x, [x[-1]] * 3]
    tr = np.array([pad[i:i + k].mean() for i in range(6)])
    a = Axes(s, 60, 30, 620, 170, (-2.5, 8.5), (0, 10), ylabel="값", xticks=list(range(-2, 9)), yticks=[0, 2, 4, 6, 8, 10],
             fmt=lambda v: f"τ{int(v) + 1}" if 0 <= v <= 5 else "복사", yfmt=lambda v: f"{v:g}")
    for j, v in enumerate(pad):
        X = a.X(j - 2)
        s.circle(X, a.Y(v), 5, C["data"], "#ffffff" if (j < 2 or j > 7) else None)
    s.poly([(a.X(i), a.Y(v)) for i, v in enumerate(tr)], C["model"], 2.5)
    for i, v in enumerate(tr):
        s.text(a.X(i), a.Y(v) - 9, f"{v:.3g}", 10, C["model"], "middle", "bold")
    s.rect(a.X(-2) - 6, 28, a.X(3) - a.X(-2) + 12, 174, "none", C["warmline"], 4, 1.2, 'stroke-dasharray="4 3"'); s.text(a.X(0.5), 26, "τ1 의 평균에 쓰는 6칸", 10, C["warmline"], "middle")
    b = Axes(s, 60, 235, 620, 60, (-2.5, 8.5), (-2, 2), xticks=[], yticks=[-2, 0, 2])
    s.text(60, 228, "주기 = 원래 − 추세", 11, C["ink2"])
    for i, v in enumerate(x - tr):
        X = a.X(i); s.rect(X - 12, min(b.Y(v), b.Y(0)), 24, abs(b.Y(v) - b.Y(0)), C["bad"] if v < 0 else C["good"], "none", 1)
        s.text(X, b.Y(v) + (12 if v < 0 else -4), f"{v:+.2f}", 10, C["ink2"], "middle")
    s.text(360, 332, "○ 끝값 복사(앞 2·뒤 3) · ● 원래 값 · 주황 선 = 추세 · 아래 막대 = 주기", 11, C["ink2"], "middle")
    return s


def fig_conv1d():
    s = Svg(720, 220, "Conv1d(커널 1)과 시간 가중합")
    s.text(20, 22, "한 시점 τ 에서: 변수 3개 → 1개", 12, C["ink2"])
    for i, (v, w) in enumerate([(0.2, 0.5), (0.1, -1), (0.3, 2)]):
        s.box(30, 36 + i * 46, 110, 36, [f"변수{i + 1} = {v}"], C["fill"], size=11)
        s.line(140, 54 + i * 46, 238, 100, arrow=True); s.text(175, 52 + i * 46 + (8 if i else 0), f"×{w}", 11, C["model"], weight="bold")
    s.box(240, 76, 170, 48, ["0.1 + 0.1 − 0.1 + 0.6", "= 0.7"], C["warm"], C["warmline"], 11)
    s.text(470, 40, "시점마다 반복 → 시간 가중합", 12, C["ink2"])
    for i, v in enumerate([0.7, 0.5, 0.9]):
        s.box(450 + i * 80, 76, 66, 40, [f"τ{i + 1}: {v}"], C["fill"], size=11)
    s.box(480, 150, 180, 44, ["(0.7+0.5+0.9) × 1/3 = 0.7"], C["out"], "#7fb894", 11)
    s.line(570, 118, 570, 148, arrow=True); s.line(410, 100, 448, 96, arrow=True)
    s.text(240, 205, "커널 1 = 같은 시점끼리만 섞음(시점끼리는 안 섞음)", 11, C["ink2"])
    return s


def fig_vae():
    s = Svg(640, 280, "VAE 표본추출")
    a = Axes(s, 60, 30, 540, 180, (-4, 4), (0, 0.42), xlabel="z", xticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4], yticks=[0, 0.2, 0.4])
    sd = math.exp(0.25); xs = np.linspace(-4, 4, 161)
    s.poly([(a.X(v), a.Y(math.exp(-v * v / 2) / math.sqrt(2 * math.pi))) for v in xs], C["grey"], 1.5, dash="5 4")
    s.poly([(a.X(v), a.Y(math.exp(-v * v / (2 * sd * sd)) / (sd * math.sqrt(2 * math.pi)))) for v in xs], C["data"], 2.5)
    z = 0.4 * sd
    s.line(a.X(z), a.Y(0), a.X(z), a.Y(0.33), C["model"], 2); s.text(a.X(z) + 6, a.Y(0.33), f"이번 z = 0 + 0.4 × {sd:.3f} = {z:.4f}", 11, C["model"], weight="bold")
    s.text(a.X(-3.8), a.Y(0.39), f"파랑: μ = ReLU(−0.3) = 0, σ = e^(0.5/2) = {sd:.3f}", 11, C["data"], weight="bold")
    s.text(a.X(-3.8), a.Y(0.355), "회색 점선: 표준정규(μ=0, σ=1) — KLD 는 둘의 거리 = 0.0744", 11, C["ink2"])
    s.text(330, 268, "원본은 log σ² 에도 ReLU → σ ≥ 1: 예측 때도 z 가 이만큼 흔들림", 11, C["ink2"], "middle")
    return s


def fig_biva_combine():
    s = Svg(640, 200, "합치기")
    s.box(20, 30, 170, 46, ["주기 예측 0.3", "× α(0.8) = 0.24"], C["fill"], size=11)
    s.box(20, 110, 170, 46, ["추세 예측 −0.1", "× (1−α)(0.2) = −0.02"], C["fill"], size=11)
    s.box(260, 70, 120, 46, ["합 = 0.22"], C["warm"], C["warmline"])
    s.box(440, 60, 180, 66, ["ŷ = 0.22 × 0.02", "+ 0.001", "= 0.0054"], C["out"], "#7fb894", 11)
    s.line(190, 53, 258, 86, arrow=True); s.line(190, 133, 258, 100, arrow=True); s.line(380, 93, 438, 93, arrow=True)
    s.text(320, 190, "α 는 1 에서 시작해 학습되며 0~1 로 제한되지 않음", 11, C["ink2"], "middle")
    return s


def fig_noise():
    s = Svg(560, 150, "반복 예측의 흔들림")
    a = Axes(s, 60, 50, 440, 0, (0.18, 0.26), (0, 1), xticks=[0.18, 0.2, 0.22, 0.24, 0.26], fmt=lambda v: f"{v:.2f}%")
    m, sd = 0.22, float(np.std([0.20, 0.22, 0.24]))
    s.rect(a.X(m - sd), 36, a.X(m + sd) - a.X(m - sd), 28, "#e7eef8", "none", 3)
    for v in (0.20, 0.22, 0.24):
        s.circle(a.X(v), 50, 6, C["data"])
    s.line(a.X(m), 30, a.X(m), 70, C["model"], 2); s.text(a.X(m), 26, f"평균 {m:.2f}%, 표준편차 {sd:.4f}%p (파란 띠 = ±1 표준편차)", 11, C["model"], "middle", "bold")
    s.text(280, 120, "같은 창·같은 가중치인데 무작위 z 만 달라도 예측이 이렇게 퍼짐", 11, C["ink2"], "middle")
    return s


# ------------------------------------------------------------------------------------------------ 규칙
def fig_rule_ma():
    s = Svg(560, 250, "이동평균 교차")
    c = [10, 11, 12, 13, 14]
    a = Axes(s, 60, 30, 440, 160, (0.5, 5.5), (9, 15), xlabel="날", ylabel="종가", xticks=[1, 2, 3, 4, 5], yticks=[9, 11, 13, 15])
    s.poly([(a.X(i + 1), a.Y(v)) for i, v in enumerate(c)], C["ink2"], 1.5)
    for i, v in enumerate(c):
        s.circle(a.X(i + 1), a.Y(v), 4, C["ink"])
    s.line(a.X(1), a.Y(12), a.X(5), a.Y(12), C["data"], 2.5); s.text(a.X(1), a.Y(12) + 16, "긴 평균(5일) = 12", 11, C["data"], weight="bold")
    s.line(a.X(3), a.Y(13), a.X(5), a.Y(13), C["model"], 2.5); s.text(a.X(3), a.Y(13) - 8, "짧은 평균(3일) = 13", 11, C["model"], weight="bold")
    s.text(280, 240, "13 > 12 → 보유", 13, C["good"], "middle", "bold")
    return s


def fig_rule_momentum():
    s = Svg(560, 220, "모멘텀")
    a = Axes(s, 60, 30, 440, 130, (-125, 5), (95, 115), xlabel="날(결정일 = 0)", ylabel="종가", xticks=[-120, -90, -60, -30, 0], yticks=[95, 100, 105, 110, 115])
    s.circle(a.X(-120), a.Y(100), 6, C["data"]); s.circle(a.X(0), a.Y(110), 6, C["model"])
    s.line(a.X(-120), a.Y(100), a.X(0), a.Y(110), C["ink2"], 1.5, "5 4", True)
    s.text(a.X(-120) + 8, a.Y(100) + 16, "120일 전 100", 11, C["data"], weight="bold"); s.text(a.X(0) - 6, a.Y(110) - 10, "오늘 110", 11, C["model"], "end", "bold")
    s.text(280, 212, "r = 110 ÷ 100 − 1 = 0.1 > 0 → 보유", 13, C["good"], "middle", "bold")
    return s


def fig_rule_rsi():
    s = Svg(640, 340, "RSI")
    c = [10, 12, 11, 14]
    a = Axes(s, 60, 30, 520, 110, (-0.5, 3.5), (9, 15), ylabel="종가", xticks=[0, 1, 2, 3], yticks=[9, 12, 15], fmt=lambda v: f"{int(v) + 1}일")
    s.poly([(a.X(i), a.Y(v)) for i, v in enumerate(c)], C["ink2"], 1.5)
    for i, v in enumerate(c):
        s.circle(a.X(i), a.Y(v), 4, C["ink"])
        if i:
            d = v - c[i - 1]; s.text(a.X(i - 0.5), a.Y((v + c[i - 1]) / 2) - 8, f"Δ{d:+d}", 11, C["good"] if d > 0 else C["bad"], "middle", "bold")
    b = Axes(s, 60, 190, 520, 110, (-0.5, 3.5), (0, 100), ylabel="RSI(3)", xticks=[0, 1, 2, 3], yticks=[0, 30, 70, 100], fmt=lambda v: f"{int(v) + 1}일")
    s.rect(b.X(-0.5), b.Y(100), b.X(3.5) - b.X(-0.5), b.Y(70) - b.Y(100), "#fbe9e7", "none", 0)
    s.rect(b.X(-0.5), b.Y(30), b.X(3.5) - b.X(-0.5), b.Y(0) - b.Y(30), "#e3f3f0", "none", 0)
    s.text(b.X(-0.4), b.Y(85), "> 70 팔기(현금)", 10, C["bad"]); s.text(b.X(3.5) - 4, b.Y(12), "< 30 사기", 10, C["good"], "end")
    pts = [(2, 80.0), (3, 89.47)]
    s.poly([(b.X(i), b.Y(v)) for i, v in pts], C["model"], 2)
    for i, v in pts:
        s.circle(b.X(i), b.Y(v), 5, C["model"]); s.text(b.X(i), b.Y(v) + 18, f"{v:g}", 11, C["model"], "middle", "bold")
    s.text(b.X(1), b.Y(50), "하락 평균 0 → 계산 안 함", 10, C["ink2"], "middle")
    s.text(320, 332, "상승 평균 2 → 1.333 → 1.889, 하락 평균 0 → 0.333 → 0.222 (Wilder, 1/3 씩 새 값 반영)", 11, C["ink2"], "middle")
    return s


def fig_rule_quantile():
    s = Svg(560, 170, "분위수 기준")
    a = Axes(s, 60, 70, 440, 0, (0.5, 5.5), (0, 1), xticks=[1, 2, 3, 4, 5])
    for v in (1, 2, 3, 4, 5):
        s.circle(a.X(v), 70, 6, C["data"])
    s.line(a.X(4.6), 40, a.X(4.6), 90, C["model"], 2); s.text(a.X(4.6), 34, "90% 위치 = 4 + 0.6 × (5 − 4) = 4.6", 11, C["model"], "middle", "bold")
    s.circle(a.X(4.8), 70, 7, C["bad"], "#ffffff", 2.5); s.text(a.X(4.8) + 2, 112, "오늘 4.8", 11, C["bad"], weight="bold")
    s.text(280, 150, "정렬한 과거 5개(1~5), 위치 (5 − 1) × 0.9 = 3.6 → 4번째와 5번째 사이 · 4.8 > 4.6 → 다음 날 현금", 11, C["ink2"], "middle")
    return s


# ------------------------------------------------------------------------------------------------ 성적표·검정
def fig_metrics():
    s = Svg(560, 270, "자산곡선과 최대낙폭")
    eq = [1.0, 1.01, 1.01 * 0.98, 1.01 * 0.98 * 1.03]
    a = Axes(s, 70, 30, 440, 180, (-0.3, 3.3), (0.98, 1.03), xlabel="날", ylabel="자산(시작 1)", xticks=[0, 1, 2, 3], yticks=[0.98, 0.99, 1.0, 1.01, 1.02, 1.03], fmt=lambda v: f"{v:g}")
    s.poly([(a.X(i), a.Y(v)) for i, v in enumerate(eq)], C["data"], 2.5)
    for i, v in enumerate(eq):
        s.circle(a.X(i), a.Y(v), 4, C["data"]); s.text(a.X(i) + 6, a.Y(v) - 8, f"{v:.6g}", 10, C["data"])
    s.line(a.X(1), a.Y(1.01), a.X(2), a.Y(1.01), C["grey"], 1, "4 3")
    s.line(a.X(2), a.Y(1.01), a.X(2), a.Y(eq[2]), C["bad"], 2, None, True); s.text(a.X(2) - 8, a.Y(0.984), "최대낙폭 0.9898 ÷ 1.01 − 1 = −2%", 11, C["bad"], "end", "bold")
    s.text(290, 262, "총수익 1.95% · 샤프 = 0.006667 ÷ 0.02517 × √252 = 4.21 (3일이라 믿으면 안 되는 값)", 11, C["ink2"], "middle")
    return s


def fig_dm():
    s = Svg(560, 250, "DM 검정")
    d = [-3, -3, 3]
    a = Axes(s, 60, 30, 440, 160, (0.5, 3.5), (-4, 4), xlabel="날", ylabel="d = 오차1² − 오차2²", xticks=[1, 2, 3], yticks=[-4, -2, 0, 2, 4])
    for i, v in enumerate(d):
        X = a.X(i + 1); s.rect(X - 22, min(a.Y(v), a.Y(0)), 44, abs(a.Y(v) - a.Y(0)), C["good"] if v < 0 else C["bad"], "none", 2)
        s.text(X, a.Y(v) + (16 if v < 0 else -6), f"{v:+d}", 12, C["ink"], "middle", "bold")
    s.line(a.X(0.5), a.Y(-1), a.X(3.5), a.Y(-1), C["model"], 2, "6 4"); s.text(a.X(3.5) - 4, a.Y(-1) - 6, "평균 −1", 11, C["model"], "end", "bold")
    s.text(280, 242, "DM = −1 ÷ √(8 ÷ 3) = −0.612 → p = 0.54 (우연과 구별 안 됨)", 12, anchor="middle", weight="bold")
    return s


# ------------------------------------------------------------------------------------------------ 움직이는 그림
def fig_walkforward_anim():
    """워크포워드: 학습 창(파랑)·비움(회색)·테스트(초록)가 오른쪽으로 밀려 가며 반복 — SVG 애니메이션(6초 반복)."""
    s = Svg(720, 190, "워크포워드 (움직이는 그림)")
    x0, w = 40, 640
    s.rect(x0, 70, w, 36, "#f6f7f9", C["line"], 3)
    s.text(x0, 56, "시간 →", 12, C["ink2"]); s.text(x0 + w, 56, "봉인 시작일 ▶", 12, C["bad"], "end")
    s.rect(x0 + w - 4, 64, 4, 48, C["bad"], "none", 0)
    tr, emb, te, step = 260, 18, 80, 80
    n = 5
    for k in range(n):
        a_, b_ = k / n, (k + 1) / n  # 6초 중 이 폴드가 보이는 구간
        if k == 0:
            kt, vals = f"0;{b_ - 0.01:.3f};{b_:.3f};1", "1;1;0;0"
        elif k == n - 1:
            kt, vals = f"0;{a_:.3f};{a_ + 0.01:.3f};1", "0;0;1;1"
        else:
            kt, vals = f"0;{a_:.3f};{a_ + 0.01:.3f};{b_ - 0.01:.3f};{b_:.3f};1", "0;0;1;1;0;0"
        g = f'<g opacity="{1 if k == 0 else 0}"><animate attributeName="opacity" values="{vals}" keyTimes="{kt}" dur="6s" repeatCount="indefinite"/>'
        xs = x0 + k * step
        g += f'<rect x="{xs}" y="72" width="{tr - emb}" height="32" rx="3" fill="#cfe0f5" stroke="{C["data"]}"/>'
        g += f'<rect x="{xs + tr - emb}" y="72" width="{emb}" height="32" fill="#e3e6ea" stroke="{C["grey"]}"/>'
        g += f'<rect x="{xs + tr}" y="72" width="{min(te, x0 + w - 4 - xs - tr)}" height="32" rx="3" fill="#cdeedd" stroke="{C["good"]}"/>'
        g += f'<text x="{xs + (tr - emb) / 2}" y="93" font-size="12" text-anchor="middle" fill="{C["ink"]}">폴드 {k + 1} 학습</text>'
        g += f'<text x="{xs + tr + min(te, x0 + w - 4 - xs - tr) / 2}" y="93" font-size="11" text-anchor="middle" fill="{C["ink"]}">테스트{" (남은 날만큼)" if k == 4 else ""}</text></g>'
        s.add(g)
    s.text(x0, 140, "파랑 = 학습 창 · 회색 = 비움(embargo: 학습 정답이 테스트 가격을 쓰지 않게) · 초록 = 예측·채점", 11, C["ink2"])
    s.text(x0, 160, "테스트 길이만큼 밀어 다시 학습 → 마지막 폴드는 봉인 전날까지 남은 날만큼 · 각 초록 구간을 이어 붙인 것이 최종 성적", 11, C["ink2"])
    return s


def fig_lstm_anim():
    """LSTM 이 창을 시점 순서대로 읽는 모습 — 시점 상자가 차례로 켜지고 기억이 갱신됨."""
    s = Svg(720, 170, "LSTM 이 창을 읽는 순서 (움직이는 그림)")
    T = 6
    for t in range(T):
        x = 40 + t * 105
        s.add(f'<rect x="{x}" y="40" width="80" height="36" rx="4" fill="{C["fill"]}" stroke="{C["line"]}">'
              f'<animate attributeName="fill" values="{C["fill"]};#ffe8c2;{C["fill"]}" keyTimes="0;0.15;1" dur="{T}s" begin="{t}s" repeatCount="indefinite"/></rect>')
        s.text(x + 40, 63, f"τ={t + 1} 재료", 11, anchor="middle")
        s.add(f'<rect x="{x}" y="100" width="80" height="30" rx="4" fill="#ffffff" stroke="{C["data"]}">'
              f'<animate attributeName="fill" values="#ffffff;#cfe0f5;#ffffff" keyTimes="0;0.15;1" dur="{T}s" begin="{t}s" repeatCount="indefinite"/></rect>')
        s.text(x + 40, 120, f"기억 h{t + 1}, c{t + 1}", 10, C["data"], "middle")
        s.line(x + 40, 76, x + 40, 98, arrow=True)
        if t:
            s.line(x - 25, 115, x - 2, 115, C["data"], 1.2, None, True)
    s.text(40, 160, "첫 기억 = 0 → 시점마다 '그날 재료 + 지난 기억'으로 새 기억 → 마지막 기억(h6)으로 예측", 11, C["ink2"])
    return s


FIGS = {k[4:]: v for k, v in globals().items() if k.startswith("fig_") and callable(v)}
from tools import formula_anims as _anims  # noqa: E402  움직이는 그림(모델별 계산 순서) — 실제 코드로 계산한 숫자
FIGS.update({k[4:]: v for k, v in vars(_anims).items() if k.startswith("fig_") and callable(v)})


def build_all(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, fn in FIGS.items():
        p = out / f"{key}.svg"
        p.write_text(fn().render(), encoding="utf-8")
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in build_all(Path(__file__).resolve().parents[1] / "docs" / "guide_build" / "figs"):
        print(p)
