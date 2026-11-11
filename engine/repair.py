"""가격 데이터 보수 — 두 출처를 합쳐 결측 거래일을 메운다. 모든 보정은 보고서로 남긴다.

merge_prices(primary, secondary):
  primary(예: Yahoo 원주가, 긴 기간)에 없는 거래일을 secondary(예: 네이버 수정주가)에서 가져온다.
  두 출처의 가격 스케일이 다를 수 있으므로(수정주가/원주가), 빠진 날 바로 앞의 공통 거래일에서
  ratio = primary_close / secondary_close 를 구해 secondary 행의 OHLC 에 곱한다. 거래량은 그대로.
"""
from __future__ import annotations

import pandas as pd


def merge_prices(primary: pd.DataFrame, secondary: pd.DataFrame, *, max_ratio_dev: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    p, s = primary.sort_index(), secondary.sort_index()
    missing = s.index.difference(p.index)
    missing = missing[(missing >= p.index.min()) & (missing <= p.index.max())]  # primary 기간 안의 결측만
    common = p.index.intersection(s.index)
    ratio = (p.loc[common, "close"] / s.loc[common, "close"])
    rows, log = [], []
    for d in missing:
        prev = common[common < d]
        if len(prev) == 0:
            continue
        r = float(ratio.loc[prev[-1]])
        if abs(r - 1) > max_ratio_dev:
            log.append({"date": d, "action": "skip", "ratio": r, "reason": "스케일 차이 과대"})
            continue
        row = s.loc[d].copy()
        for c in ("open", "high", "low", "close"):
            row[c] = row[c] * r
        rows.append(row.rename(d))
        log.append({"date": d, "action": "fill", "ratio": r, "from_prev_common": prev[-1]})
    merged = pd.concat([p, pd.DataFrame(rows)]).sort_index() if rows else p.copy()
    merged.index.name = "date"
    return merged, pd.DataFrame(log)
