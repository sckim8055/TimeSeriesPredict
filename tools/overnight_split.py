"""미국 마감 → 한국 시가 시차 검증 (2026-11-16). 읽기만 하고 결과(results/)에 쓰지 않는다.

질문: 미국 반도체(SOX·NVDA·MU·TSM)의 지난밤 움직임이 091160 의 하루 중 어느 조각에 남는가?
  한국 날짜 D 의 조각:
    갭(gap)      = D 시가 ÷ 전날 종가 − 1     — 미국 지난밤 정보가 들어오는 곳. 시가에 사는 우리는 가질 수 없다
    장중(intra)  = D 종가 ÷ D 시가 − 1        — 시가에 사서 종가에 팔면 가질 수 있다(엔진에는 없는 매매 방식)
    다음 밤(next)= D+1 시가 ÷ D 종가 − 1      — 다음 미국 장의 정보(결정 때는 모름)가 들어감
    엔진 정답    = D+1 시가 ÷ D 시가 − 1      — 엔진 매매 모드 k=1 의 보유 수익(결정 행 = D 의 전 거래일)
  미국 쪽: 한국 D 시가(09:00 KST) 전에 끝난 마지막 미국 장 = 달력 날짜가 D 보다 앞선 마지막 미국 거래일 u 의 종가 변화율.
    그 u 를 전 한국 거래일이 이미 썼으면(미국 휴장) 새 정보가 없으므로 그날은 뺀다.
방법: 조각 = a + b × 미국 변화율 + 오차, 최소제곱 + Newey–West(시차 5) 표준오차. b = 미국 1% 움직임이 그 조각에 남기는 크기(%).
  기간은 두 절반(관측 개수로 나눔)으로도 본다 — 효과가 한 시기에만 있었는지. 봉인 시작일(seal_policy.json) 앞 자료만.
  간단한 규칙 확인(설명용, 매매 근거 아님): '미국이 올랐으면 시가에 사서 종가에 판다'의 1회 평균 수익과 왕복 비용(편도 0.065% × 2).
실행: python tools/overnight_split.py [데이터버전] > docs/analysis/overnight_split_<버전>.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.backtest import CostModel  # noqa: E402
from engine.datacheck import series_of  # noqa: E402
from engine.store import DataVersion  # noqa: E402

US = ["SOX", "NVDA", "MU", "TSM"]
TICKER = "091160"
HAC_LAGS = 5


def us_before_open(kr_index: pd.DatetimeIndex, us_ret: pd.Series) -> pd.Series:
    """한국 날짜 D 마다: 달력 날짜 < D 인 마지막 미국 거래일의 변화율. 전 한국 거래일과 같은 미국 날이면 NaN(새 정보 없음)."""
    us_ret = us_ret.dropna().sort_index()
    pos = us_ret.index.searchsorted(kr_index, side="left") - 1          # D 보다 엄격히 앞
    out = pd.Series(np.nan, index=kr_index)
    used = pd.Series(pd.NaT, index=kr_index)
    ok = pos >= 0
    out[ok] = us_ret.values[pos[ok]]
    used[ok] = us_ret.index[pos[ok]]
    out[used.eq(used.shift(1))] = np.nan
    return out


def segments(px: pd.DataFrame) -> pd.DataFrame:
    o, c = px["open"], px["close"]
    return pd.DataFrame({"gap": o / c.shift(1) - 1, "intra": c / o - 1, "next": o.shift(-1) / c - 1, "engine": o.shift(-1) / o - 1})


def ols_hac(y: pd.Series, x: pd.Series) -> dict:
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    m = sm.OLS(d["y"], sm.add_constant(d["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return {"n": int(len(d)), "b": float(m.params["x"]), "t": float(m.tvalues["x"]), "p": float(m.pvalues["x"]), "r2": float(m.rsquared),
            "corr": float(d["y"].corr(d["x"]))}


def main(ver: str | None = None) -> str:
    if ver is None:
        vs = sorted(p.name for p in (ROOT / "data" / "versions").iterdir() if p.is_dir())
        ver = vs[-1]
    dv = DataVersion(ROOT / "data", ver)
    seal = json.loads((ROOT / "seal_policy.json").read_text(encoding="utf-8"))
    sealed_from = pd.Timestamp(seal.get(TICKER) if isinstance(seal.get(TICKER), str) else seal.get(TICKER, {}).get("sealed_from", "2025-01-01"))
    px = dv.read_prices(TICKER).sort_index()
    px.index = pd.DatetimeIndex(px.index)
    px = px[px.index < sealed_from]
    seg = segments(px)
    seg = seg[seg.index < px.index[-1]]                                  # 마지막 날은 다음 시가가 없음
    long = dv.read_variables(US)
    usr = {}
    for v in US:
        s, _ = series_of(long[long["variable_id"] == v])
        usr[v] = us_before_open(seg.index, s.pct_change())
    X = pd.DataFrame(usr)
    lines = [f"# 미국 마감 → 한국 시가 시차 검증 — {TICKER}", "",
             f"데이터 버전 `{ver}` · 봉인 시작일 {sealed_from.date()} 앞 자료만 · 만든 도구 `tools/overnight_split.py` (읽기만, 결과 저장 없음) · Newey–West 시차 {HAC_LAGS}", "",
             "b = 미국이 1% 움직였을 때 그 조각이 평균 몇 % 움직였나. t 가 ±2 를 넘으면 우연으로 보기 어렵다(여러 번 본 것을 감안하면 더 엄격히).", ""]
    span = X.dropna(how="all").index
    lines += [f"기간 {span.min().date()} ~ {span.max().date()} (한국 거래일, 미국 새 정보가 있는 날)", ""]
    halves = {}
    idx = X["SOX"].dropna().index
    mid = idx[len(idx) // 2]
    for v in US:
        lines += [f"## {v}", "", "| 조각 | 뜻 | n | b | t | R² | 상관 |", "|---|---|---|---|---|---|---|"]
        for k, desc in (("gap", "밤사이 갭(전날 종가→시가) — 가질 수 없음"), ("intra", "장중(시가→종가)"), ("next", "다음 밤(종가→다음 시가)"), ("engine", "엔진 정답(시가→다음 시가)")):
            r = ols_hac(seg[k], X[v])
            lines.append(f"| {k} | {desc} | {r['n']} | {r['b']:.3f} | {r['t']:.2f} | {r['r2']:.3f} | {r['corr']:.3f} |")
        lines += ["", "기간을 둘로 나눠 본 장중 b (t):", ""]
        for name, sel in (("앞 절반", seg.index < mid), ("뒤 절반", seg.index >= mid)):
            r = ols_hac(seg["intra"][sel], X[v][sel]); rg = ols_hac(seg["gap"][sel], X[v][sel])
            halves[(v, name)] = (r, rg)
            lines.append(f"- {name}: 장중 b {r['b']:.3f} (t {r['t']:.2f}) · 갭 b {rg['b']:.3f} (t {rg['t']:.2f}) · n {r['n']}")
        lines.append("")
    # 설명용 규칙: SOX 가 올랐으면 시가 매수 → 종가 매도
    one_way = CostModel().one_way
    d = pd.concat([seg["intra"], seg["engine"], X["SOX"]], axis=1).dropna()
    up, dn = d[d["SOX"] > 0], d[d["SOX"] <= 0]
    lines += ["## 설명용 규칙 — 'SOX 가 지난밤 올랐으면 시가에 사서 종가에 판다' (매매 근거 아님)", "",
              "| 경우 | 날 수 | 장중 평균 | 시가→다음 시가 평균 |", "|---|---|---|---|",
              f"| SOX 상승한 밤 다음 날 | {len(up)} | {up['intra'].mean() * 100:.3f}% | {up['engine'].mean() * 100:.3f}% |",
              f"| SOX 하락·보합 밤 다음 날 | {len(dn)} | {dn['intra'].mean() * 100:.3f}% | {dn['engine'].mean() * 100:.3f}% |",
              f"| 모든 날 | {len(d)} | {d['intra'].mean() * 100:.3f}% | {d['engine'].mean() * 100:.3f}% |", "",
              f"왕복 비용(편도 {one_way * 100:.3f}% × 2) = {2 * one_way * 100:.3f}%. 상승한 밤 다음 날 장중 평균 − 왕복 비용 = {(up['intra'].mean() - 2 * one_way) * 100:.3f}%/회.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else None))
