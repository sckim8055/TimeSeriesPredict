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
    STOCHASTIC = True  # subsample·colsample_bytree < 1 이면 행·열 표본추출이 시드에 따라 달라진다

    def fit(self, X, y, X_val=None, y_val=None):
        # 시드(2026-11-10): 그 전에는 random_state=0 고정이라 실험 설정의 seed 를 바꿔도 LightGBM 결과는 같았고, 시드에 따른 흔들림을 잴 수 없었다.
        # LightGBM 은 자체 난수기를 써서 엔진이 맞춘 전역 난수를 따르지 않는다 → 엔진이 이 폴드·이 반복의 시드로 맞춰 둔 numpy 전역 난수에서
        # 시드를 하나 뽑아 넘긴다. 모델 파일이 시드 값을 정하는 것이 아니다(절대 규칙 3) — 같은 실험 시드·같은 폴드면 같은 값이 뽑힌다.
        seed = int(np.random.randint(0, 2**31 - 1))
        kw = dict(self.params, subsample_freq=1, verbose=-1, random_state=seed, n_jobs=1)
        cls = lgb.LGBMRegressor if self.target_type == "regression" else lgb.LGBMClassifier
        self.est = cls(**kw)
        early = False
        if X_val is not None and len(X_val) > 10:
            early = True
            self.est.fit(X, y, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            self.est.fit(X, y)
        # 진단: 실제로 예측에 쓰이는 트리 수. 조기 종료가 켜지면 검증 손실이 가장 낮았던 라운드까지만 쓴다(LightGBM 의 predict 기본 동작).
        # 수익률처럼 신호가 약한 타겟에서는 1~2그루에서 멈추는 폴드가 많다 — 그때 예측은 '학습 평균 + 아주 작은 조정'에 가깝다.
        # 그 사실을 숨기지 않도록 폴드마다 남긴다(결과의 model_diag).
        best = getattr(self.est, "best_iteration_", None)
        built = int(self.est.booster_.current_iteration())
        self.diag = {"early_stopping": bool(early), "trees_built": built, "trees_used": int(best) if (early and best) else built,
                     "n_estimators": int(self.params["n_estimators"]), "lgb_seed": seed}

    def predict(self, X):
        if self.target_type == "regression":
            return self.est.predict(X)
        return self.est.predict_proba(X)[:, 1]

    def feature_importance(self):
        return self.est.booster_.feature_importance(importance_type="gain").astype(float)
