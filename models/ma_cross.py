"""기준선: 이동평균 교차. 단기 SMA > 장기 SMA 이면 롱, 아니면 현금. (골든/데드 크로스)"""
import numpy as np
from models.base import RuleModel

NAME = "MA Cross"
INPUT_SHAPE = "rule"
TARGET_TYPES = ["signal"]
PARAMS = {
    "fast": {"default_by_freq": {"D": 20, "W": 4, "M": 3}, "type": "int", "default": 20, "min": 2, "max": 250, "desc": "단기 이동평균 기간"},
    "slow": {"default_by_freq": {"D": 60, "W": 13, "M": 6}, "type": "int", "default": 60, "min": 3, "max": 500, "desc": "장기 이동평균 기간"},
}


class Model(RuleModel):
    def positions(self, prices, X=None):
        c = prices["close"].astype(float)
        fast = c.rolling(self.params["fast"]).mean()
        slow = c.rolling(self.params["slow"]).mean()
        pos = (fast > slow).astype(float)
        pos[slow.isna()] = np.nan  # 워밍업 구간: 결정 불가
        return pos.values
