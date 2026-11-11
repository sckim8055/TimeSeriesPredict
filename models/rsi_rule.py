"""기준선: RSI. RSI < lower 에서 매수(롱 진입), RSI > upper 에서 매도(현금). 그 사이는 유지."""
import numpy as np
import pandas as pd
from engine.features import rsi
from models.base import RuleModel

NAME = "RSI"
INPUT_SHAPE = "rule"
TARGET_TYPES = ["signal"]
PARAMS = {
    "window": {"default_by_freq": {"D": 14, "W": 8, "M": 6}, "type": "int", "default": 14, "min": 2, "max": 100, "desc": "RSI 계산 창(주기 수)"},
    "lower": {"type": "float", "default": 30.0, "min": 0, "max": 100, "desc": "이 값 아래로 내려가면 매수(과매도)"},
    "upper": {"type": "float", "default": 70.0, "min": 0, "max": 100, "desc": "이 값 위로 올라가면 매도(과매수)"},
}


class Model(RuleModel):
    def positions(self, prices, X=None):
        r = rsi(prices["close"].astype(float), self.params["window"])
        pos = pd.Series(np.nan, index=r.index)
        pos[r < self.params["lower"]] = 1.0
        pos[r > self.params["upper"]] = 0.0
        pos[r.isna()] = np.nan
        # 진입/청산 사이는 유지 → ffill; 워밍업 구간은 NaN 유지
        first = r.first_valid_index()
        if first is not None:
            pos.loc[first:] = pos.loc[first:].ffill().fillna(0.0)
        return pos.values
