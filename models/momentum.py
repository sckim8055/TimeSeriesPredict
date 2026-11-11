"""기준선: 시계열 모멘텀. 과거 lookback 수익률 > 0 이면 롱, 아니면 현금. (Moskowitz, Ooi & Pedersen 2012 의 단순형)"""
import numpy as np
from models.base import RuleModel

NAME = "Momentum"
INPUT_SHAPE = "rule"
TARGET_TYPES = ["signal"]
PARAMS = {"lookback": {"default_by_freq": {"D": 120, "W": 26, "M": 12}, "type": "int", "default": 120, "min": 2, "max": 500, "desc": "수익률 계산 기간"}}


class Model(RuleModel):
    def positions(self, prices, X=None):
        c = prices["close"].astype(float)
        r = c / c.shift(self.params["lookback"]) - 1
        pos = (r > 0).astype(float)
        pos[r.isna()] = np.nan
        return pos.values
