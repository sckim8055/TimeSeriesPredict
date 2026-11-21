"""기준선: 섀넌 재조정(2026-11-15). ETF 비율을 정해 두고(기본 50%) 나머지는 현금, 정한 주기마다 그 비율로 다시 맞춘다.
예측도 지표도 쓰지 않는다. '판단 없이 늘 절반만 투자하면 이만큼'을 보여 주는 잣대 — 늘 100% 인 Buy&Hold 와 짝.
오르면 ETF 비율이 늘어 일부를 팔고, 내리면 줄어 일부를 산다(자동으로 '비쌀 때 팔고 쌀 때 사기').
출처: Claude Shannon 의 강의 아이디어('섀넌의 악마') — 정식 논문은 없고 강의·기록으로 전한다.
회계: 비율은 다음 재조정까지 가격을 따라 흘러가고, 다시 맞출 때만 비용을 낸다(engine.backtest.run_backtest 의 rebalance)."""
import numpy as np
from engine.backtest import period_end_mask
from models.base import RuleModel

NAME = "섀넌 재조정"
INPUT_SHAPE = "rule"
TARGET_TYPES = ["signal"]
PARAMS = {
    "weight": {"type": "float", "default": 0.5, "min": 0.05, "max": 0.95, "desc": "ETF 에 넣어 두는 비율(나머지는 현금). 0.5 = 절반"},
    "every": {"type": "str", "default": "M", "choices": ["D", "W", "M", "Q"],
              "desc": "다시 맞추는 주기 — D 매 기간, W 주마다, M 달마다(기본), Q 분기마다. 그 주기의 마지막 결정일에 정하고 다음 시가에 체결"},
}


class Model(RuleModel):
    REBALANCE = True   # engine 이 rebalance_days 를 run_backtest 에 넘긴다(흘러가는 비율 회계)

    def positions(self, prices, X=None):
        return np.full(len(prices), float(self.params["weight"]))

    def rebalance_days(self, prices):
        return period_end_mask(prices.index, self.params["every"])
