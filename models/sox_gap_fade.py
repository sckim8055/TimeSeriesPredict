"""규칙: 미국 반도체 급등 다음날 회피 (2차 실험 H1 에서 나온 역추세 후보).

근거(docs/results_round2.md): SOX(t) 상승은 한국 개장가에 거의 다 반영되고(갭과 상관 +0.67), 그 뒤 t+1 시가→t+2 시가 수익은 음(−)의 상관.
규칙: 기본 롱. SOX(t) 일간수익률이 **과거 window 일의 상위 (1−q) 분위**를 넘으면 다음 날 현금.
    분위수는 t 이전 자료로만 계산한다(롤링, 당일 제외) — 미래 분위수 사용 금지.
입력: 설정의 variable_ids=["SOX"], variable_transforms={"SOX":"pct"}, lag_override={"SOX":0} (결정 시점 = t+1 개장 직전).
mode="avoid_top": 급등일 회피(기본) / "long_bottom": 급락일에만 롱.
"""
import numpy as np
import pandas as pd

from models.base import RuleModel

NAME = "SOX Gap Fade"
INPUT_SHAPE = "rule"
TARGET_TYPES = ["signal"]
PARAMS = {
    "q": {"type": "float", "default": 0.9, "min": 0.5, "max": 0.99, "desc": "회피 임계 분위 (0.9 = 상위 10% 급등일 회피)"},
    "window": {"default_by_freq": {"D": 252, "W": 104, "M": 60}, "type": "int", "default": 252, "min": 60, "max": 1260, "desc": "분위수 계산 창(기간 수: 일간 거래일·주간 주·월간 월). 처음에는 창의 절반만 차도 계산(min_periods = window/2)"},
    "mode": {"type": "str", "default": "avoid_top", "choices": ["avoid_top", "long_bottom"], "desc": "avoid_top: 기본 롱·급등일 회피 / long_bottom: 급락일만 롱"},
    "column": {"type": "str", "default": "SOX_pct", "desc": "X 에서 쓸 열 이름"},
}


class Model(RuleModel):
    USES_VARIABLES = True

    def positions(self, prices, X=None):
        if X is None or self.params["column"] not in X.columns:
            raise ValueError(f"X 에 {self.params['column']} 열이 필요합니다 (variable_ids=['SOX'], variable_transforms={{'SOX':'pct'}})")
        s = X[self.params["column"]].astype(float).reindex(prices.index)
        w, q = int(self.params["window"]), float(self.params["q"])
        past = s.shift(1)  # 당일 제외: t 의 임계는 t−1 까지의 분포
        if self.params["mode"] == "avoid_top":
            thr = past.rolling(w, min_periods=w // 2).quantile(q)
            pos = (~(s > thr)).astype(float)
        else:
            thr = past.rolling(w, min_periods=w // 2).quantile(1 - q)
            pos = (s < thr).astype(float)
        pos[thr.isna() | s.isna()] = np.nan  # 워밍업/결측: 결정 불가
        return pos.values
