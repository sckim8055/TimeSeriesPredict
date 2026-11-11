"""Ridge 회귀 / 로지스틱 회귀 (L2). 표 입력. 선형 기준 ML."""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from models.base import BaseModel

NAME = "Ridge"
INPUT_SHAPE = "table"
TARGET_TYPES = ["regression", "classification"]
PARAMS = {"alpha": {"default_by_freq": {"D": 1.0, "W": 10.0, "M": 100.0}, "type": "float", "default": 1.0, "min": 1e-6, "max": 1e6, "desc": "L2 규제 강도"}}


class Model(BaseModel):
    USES_VALIDATION = False  # 검증 분할 불필요 → 학습 창 전체 사용

    def fit(self, X, y, X_val=None, y_val=None):
        a = self.params["alpha"]
        est = Ridge(alpha=a) if self.target_type == "regression" else LogisticRegression(C=1.0 / a, max_iter=2000)
        self.pipe = make_pipeline(StandardScaler(), est).fit(X, y)

    def predict(self, X):
        if self.target_type == "regression":
            return self.pipe.predict(X)
        return self.pipe.predict_proba(X)[:, 1]

    def feature_importance(self):
        est = self.pipe[-1]
        return np.abs(np.ravel(est.coef_))
