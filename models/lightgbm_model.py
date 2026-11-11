"""LightGBM (그래디언트 부스팅 트리). 표 입력. 변수 중요도(gain) 제공 → ablation 뷰의 주력."""
import numpy as np
import lightgbm as lgb
from models.base import BaseModel

NAME = "LightGBM"
INPUT_SHAPE = "table"
TARGET_TYPES = ["regression", "classification"]
PARAMS = {
    "n_estimators": {"default_by_freq": {"D": 300, "W": 200, "M": 150}, "type": "int", "default": 300, "min": 10, "max": 5000, "desc": "트리 개수(부스팅 라운드 상한)"},
    "learning_rate": {"type": "float", "default": 0.03, "min": 1e-4, "max": 1.0, "desc": "학습률 — 작을수록 트리를 많이 필요"},
    "num_leaves": {"default_by_freq": {"D": 15, "W": 7, "M": 4}, "type": "int", "default": 15, "min": 2, "max": 512, "desc": "트리당 최대 잎 수(복잡도)"},
    "max_depth": {"default_by_freq": {"D": 4, "W": 3, "M": 2}, "type": "int", "default": 4, "min": -1, "max": 64, "desc": "트리 최대 깊이(-1 = 제한 없음)"},
    "min_child_samples": {"default_by_freq": {"D": 40, "W": 15, "M": 6}, "type": "int", "default": 40, "min": 1, "max": 1000, "desc": "잎 하나의 최소 표본 수 — 학습 표본이 적으면(월간 60개) 반드시 줄일 것, 아니면 분할이 안 되어 상수 예측(퇴화)"},
    "subsample": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "desc": "트리마다 쓰는 행 비율(배깅)"},
    "colsample_bytree": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "desc": "트리마다 쓰는 변수 비율"},
    "reg_lambda": {"type": "float", "default": 1.0, "min": 0.0, "max": 100.0, "desc": "L2 규제 강도"},
}


class Model(BaseModel):
    def fit(self, X, y, X_val=None, y_val=None):
        kw = dict(self.params, subsample_freq=1, verbose=-1, random_state=0, n_jobs=1)
        cls = lgb.LGBMRegressor if self.target_type == "regression" else lgb.LGBMClassifier
        self.est = cls(**kw)
        if X_val is not None and len(X_val) > 10:
            self.est.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            self.est.fit(X, y)

    def predict(self, X):
        if self.target_type == "regression":
            return self.est.predict(X)
        return self.est.predict_proba(X)[:, 1]

    def feature_importance(self):
        return self.est.booster_.feature_importance(importance_type="gain").astype(float)
