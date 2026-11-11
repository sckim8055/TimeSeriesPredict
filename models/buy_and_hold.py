"""기준선: Buy & Hold. 항상 롱. 모든 비교 화면의 기준."""
import numpy as np
from models.base import RuleModel

NAME = "Buy&Hold"
INPUT_SHAPE = "rule"
TARGET_TYPES = ["signal"]
PARAMS: dict = {}


class Model(RuleModel):
    def positions(self, prices, X=None):
        return np.ones(len(prices))
