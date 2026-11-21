"""실험 실행기 — 설정(ExperimentConfig) 하나로 데이터 → 타겟 → 워크포워드 학습·예측 → 신호 → 백테스트 → 결과.

실험 1건 = 변수세트 × 모델 × 타겟 × 매매주기 × 매매규칙 × 검증설정 (기획서 2장).
모든 모델이 이 한 경로를 지나므로 비교 조건이 같다. 모델 파일은 fit/predict 만 한다.
"""
from __future__ import annotations

import hashlib
import os
import importlib
import json
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import logging

from engine import calendar as cal
from engine.logs import describe, step
from engine.backtest import performance, CostModel, bootstrap_excess_pvalue, clark_west, deflated_sharpe_pvalue, diebold_mariano, diebold_mariano_brier, return_moments, run_backtest, shannon_benchmark
from engine.features import select_technical, technical_block
from engine.registry import Registry
from engine.store import DataVersion
from models.base import BaseModel as PluginBase
from models.base import RuleModel, resolve_params

log_e = logging.getLogger("tsp.engine")
log_d = logging.getLogger("tsp.engine.data")
log_f = logging.getLogger("tsp.engine.fold")


# ---------------------------------------------------------------------------
# 설정 스키마 (CLI·API·화면이 같은 것을 쓴다)
# ---------------------------------------------------------------------------
class TargetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 모르는 키(오타)는 오류 — 조용히 기본값으로 돌지 않게(2026-11-10)
    # volatility(2026-11-12): 수익률이 아니라 그 뒤 k 기간의 실현 변동성을 맞힌다(매매 모드·학습 모델만). 매매 규칙은 '예측 변동성이 높으면 비중을 줄임'
    kind: Literal["return", "logreturn", "direction", "event", "volatility"] = Field("return", description="무엇을 맞힐지. return: k 기간 수익률 / logreturn: 로그 수익률 / direction: 오르면 1·아니면 0 / event: 수익률이 threshold 를 넘으면 1 / volatility: 뒤 k 기간 실현 변동성")
    k: int = Field(1, ge=1, description="예측 지평(주기 수)")
    threshold: float = Field(0.0, description="kind=event 일 때만: 수익률이 이 값을 넘으면 1(사건). 다른 kind 에서는 쓰지 않음")


class TradeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 모르는 키(오타)는 오류 — 조용히 기본값으로 돌지 않게(2026-11-10)
    threshold: float = Field(0.0, description="regression: 예상수익률 > threshold → 롱. classification: 확률 > 0.5+threshold → 롱")
    fee_rate: float = Field(0.00015, description="편도 수수료 비율 (0.00015 = 0.015%). 사고팔 때마다 거래 금액에 곱해 뺀다")
    slippage_bp: float = Field(5.0, description="편도 슬리피지(bp, 1bp = 0.01%) — 원하는 가격과 실제 체결 가격의 차이. 수수료와 더해 비용이 된다")
    # 투자 비율 정하기(2026-11-15, engine/sizing.py): threshold = 예측이 문턱을 넘으면 전부·아니면 현금(기존), kelly = 과거 결과로 보정한 켈리 비율
    sizing: Literal["threshold", "kelly"] = Field("threshold", description="threshold: 문턱 넘으면 전부 보유 / kelly: 과거에 같은 자신감일 때의 실제 수익으로 비율(0~1)")
    kelly_fraction: float = Field(0.5, gt=0.0, le=1.0, description="켈리 분수 — 0.5 = 절반 켈리(기본). 1 = 전부 켈리(추정이 틀리면 크게 잃을 수 있음)")
    kelly_min_history: int | None = Field(None, ge=20, description="켈리 준비 기간 — 결과가 나온 과거 예측이 이 개수가 될 때까지 현금. 비우면 주기별(일 250·주 52·월 24)")


class Validation(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 모르는 키(오타)는 오류 — 조용히 기본값으로 돌지 않게(2026-11-10)
    train_len: int = Field(756, description="학습 창 길이(주기 수). 일간 3년≈756")
    test_len: int = Field(21, description="재학습 주기(주기 수). 일간 1개월≈21")
    embargo: int | None = Field(None, description="비움(행 수) — 학습 끝과 테스트(검증) 시작 사이를 비우는 수. 비우면 매매 1+k · 예측 연구 k (결정일의 정답이 그만큼 뒤의 값을 쓰므로)")  # None → 1+k
    sealed_from: str | None = Field(None, description="봉인 시작일 'YYYY-MM-DD'. 이 날 이후는 개발 중 계산하지 않음(최종 확인용으로 남겨 둠)")
    val_frac: float = Field(0.15, description="검증 비율 — 학습 창(구간) 뒤쪽 이 비율을 조기 종료용 검증으로 뗌. 검증이 10개 미만이면 검증 없이 전부 학습. 검증을 쓰지 않는 모델은 무시")
    mode: Literal["walkforward", "split"] = Field("walkforward", description="walkforward: 창을 밀며 반복 재학습 / split: 1회 분할(노트북 방식, split_date 이전 학습·이후 테스트)")
    split_date: str | None = Field(None, description="(이전 이름) mode=split 의 테스트 시작일. test_start 가 있으면 test_start 를 쓴다")
    val_start: str | None = Field(None, description="mode=split 에서 검증 시작일 'YYYY-MM-DD'. 검증 구간 = [val_start, 테스트 시작 − embargo). 비우면 val_frac")
    test_start: str | None = Field(None, description="테스트(평가) 시작일. split: 이 날 이전이 학습. walkforward: 첫 테스트 폴드가 이 날 시작(그 전 폴드는 만들지 않음) — 모든 모델의 평가 기간을 맞출 때")
    test_end: str | None = Field(None, description="테스트(평가) 종료일(포함). 비우면 봉인 시작 전날 또는 데이터 끝")


def _norm_technical(tf: list[str] | None, use_technical: bool, freq: str, with_volume: bool) -> list[str] | None:
    """기술지표 선택 목록 정규화: 기술지표를 끄면 None, 중복 제거, 만드는 순서대로, 전부면 None. 없는 이름은 남겨 두어 실행 때 select_technical 이 오류로 알린다."""
    if tf is None or not use_technical:
        return None
    try:
        from engine.features import technical_columns
        allc = technical_columns(freq, with_volume=with_volume)
    except Exception:  # noqa: BLE001 — 기술지표가 정의되지 않은 주기: 그대로 두고 실행 때 오류로 드러나게
        return tf
    known = [c for c in allc if c in set(tf)]
    unknown = [c for c in dict.fromkeys(tf) if c not in set(allc)]
    if not unknown and len(known) == len(allc):
        return None
    return known + unknown  # 빈 목록은 실행 때 오류(전부 빼려면 use_technical 을 끈다)


class ExperimentConfig(BaseModel):
    # 모르는 키는 오류(전에는 무시 — 'frequncy' 같은 오타가 기본값으로 조용히 돌았다, 2026-11-10). '_' 로 시작하는 키(_comment)는 메모로 보고 버린다
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_comment_keys(cls, data):
        return {k: v for k, v in data.items() if not str(k).startswith("_")} if isinstance(data, dict) else data

    @model_validator(mode="after")
    def _normalise_technical(self):
        """technical_features 정규화 — 같은 입력이면 같은 해시가 되게(_norm_technical). 예측 연구 모드의 실제 주기(대상 변수의 주기)는 변수 사전이
        있어야 알 수 있어, 여기서는 설정의 frequency 로 한 번 하고 실행 때 fill_defaults 가 실제 주기로 다시 한다."""
        if self.technical_features is not None:
            object.__setattr__(self, "technical_features", _norm_technical(self.technical_features, self.use_technical, self.frequency, self.mode != "forecast"))
        # pool_tickers 에 같은 종목을 두 번: 결과는 같은데(실행 때 한 번만 씀) 설정 해시만 달라져 같은 후보가 둘로 세어진다 → 거절(2026-11-13 독립 검토 12).
        # 순서는 바꾸지 않는다(보탠 행의 순서가 LightGBM 학습에 영향을 줄 수 있어, 적은 순서가 곧 설정이다).
        dup = [t for i, t in enumerate(self.pool_tickers) if t in self.pool_tickers[:i]]
        if dup:
            raise ValueError(f"종목 묶어 학습(pool_tickers)에 같은 종목을 두 번 넣었습니다: {sorted(set(dup))} — 한 번씩만 넣어 주세요")
        return self

    name: str = Field("", description="결과 이름(메모). 계산·해시에는 들어가지 않음")
    mode: Literal["trading", "forecast"] = Field("trading", description="trading: 매매 시뮬레이션(ETF 가격, 체결·비용·B&H) / forecast: 예측 연구(임의 시계열, 오차 지표·naive 대비)")
    target_variable: str | None = Field(None, description="forecast 모드의 대상 변수 id (마스터). 주기는 그 변수의 원래 주기")
    forecast_info_time: Literal["stamp", "release"] = Field(
        "stamp", description="forecast 모드에서 외생변수를 '언제까지 알려진 것'으로 맞출지. stamp: 대상 값의 기준일 t(보수적, 기본). "
                             "release: 대상 값 v[t] 가 공개되는 시점 t+lag_대상 — 실제로 예측을 하는 시점의 정보 집합(현실적)")
    ticker: str = Field("091160", description="매매 모드의 대상 종목 코드(예: 091160 = KODEX 반도체). 가격으로 정답(수익률)과 체결을 계산")
    frequency: Literal["D", "W", "M"] = Field("D", description="결정 주기 — D 일간 / W 주간 / M 월간. 이 간격으로 판단하고 정답 지평 k 도 이 단위")
    start: str | None = Field(None, description="데이터 사용 시작일. 이 날 이전 데이터는 읽지 않는다(학습·검증·테스트 모두)")
    end: str | None = Field(None, description="데이터 사용 종료일. 이 날 이후 데이터는 읽지 않는다")
    target: TargetSpec = Field(TargetSpec(), description="맞힐 대상(정답)의 정의")
    variable_ids: list[str] = Field(default_factory=list, description="레지스트리 변수 ID. 비우면 A그룹(기술지표)만")
    use_technical: bool = Field(True, description="가격으로 만든 기술지표(A 그룹)를 입력에 넣을지")
    technical_features: list[str] | None = Field(None, description="쓸 기술지표 열 이름(A_…). 비우면(None) 전부 — 2026-11-09 추가. 해시에는 값이 있을 때만 들어간다")
    fill: Literal["ffill", "bfill", "interp"] = Field("ffill", description="변수를 결정일 달력에 맞출 때 빈 날 채우기. ffill: 직전 값(미래 안 봄) / bfill·interp: 뒤 값을 써서 미래가 섞임(경고)")
    apply_lag: bool = Field(True, description="발표 시차 적용 — 변수 값을 기준일이 아니라 공개일(기준일 + lag_days)부터 쓴다. 끄면 기준일부터(미래가 섞일 수 있음)")
    variable_transforms: dict[str, str] = Field(default_factory=dict, description="변수별 변환 재정의 {id: none|pct|pct{n}|diff|diff{n}|log|logdiff}. 비우면 마스터의 transform")
    lag_override: dict[str, int] = Field(default_factory=dict, description="변수별 lag_days 재정의(가설 검정용). 결과에 기록된다")
    model: str = Field(..., description="models/ 파일명 (확장자 없이)")
    model_params: dict[str, Any] = Field(default_factory=dict, description="모델 파라미터 중 기본값과 다르게 준 것만. 실제로 쓰인 전체 값은 결과의 '모델 파라미터'")
    seq_len: int | None = Field(None, description="window 입력 길이 (window 모델만). 비우면 주기별 기본값 D 20 / W 13 / M 12")
    trade: TradeRule = Field(TradeRule(), description="매매 규칙과 비용(매매 모드)")
    validation: Validation = Field(Validation(), description="학습·검증·테스트를 나누는 방법")
    data_version: str = Field("", description="쓴 데이터 버전(수집 시각). 비우면 실행 때 최신 버전 — 결과에는 실제 버전이 기록됨")
    seed: int = Field(2025, description="난수 시드 — 같은 시드면 같은 결과(재현). 시드에 따라 달라지는 모델만 영향")
    n_seeds: int = Field(1, ge=1, le=50, description="난수에 따라 결과가 달라지는 모델(LSTM·GRU·BIVA·LightGBM)을 폴드마다 시드를 바꿔 몇 번 학습할지. "
                                                     "예측은 그 평균을 쓰고 시드 사이 흩어짐을 결과에 남긴다. 결정적 모델(릿지·ARIMA 등)은 무시. 2026-11-10 추가, 해시에는 1 이 아닐 때만 들어간다")

    select_top: int | None = Field(None, ge=1, description="입력 선별(2026-11-12): 폴드마다 '학습 창 안에서' 정답과의 |순위상관|이 높은 입력 N개만 골라 학습한다. "
                                   "None = 고르지 않음(전부 사용, 순위만 결과에 남김). 표·창 입력 학습 모델만. 해시에는 None 이 아닐 때만 들어간다")

    pool_tickers: list[str] = Field(default_factory=list, description="묶어 학습(2026-11-12): 여기 적은 종목의 (입력, 정답) 행을 폴드마다 '학습에만' 보탠다. "
                                    "예측·평가는 대상 종목(ticker)만. 매매 모드·표 입력 모델(릿지·LightGBM)만. 해시에는 비어 있지 않을 때만 들어간다")

    def config_hash(self) -> str:
        # technical_features 는 2026-11-09 에 생긴 칸: 기본값(None = 전부)이면 해시에서 빼서 그 전 결과의 이름·재실행 연결을 그대로 둔다
        # n_seeds(2026-11-10)도 같다: 기본값 1 이면 해시에서 뺀다
        ex = {"name"} | ({"technical_features"} if self.technical_features is None else set()) | ({"n_seeds"} if self.n_seeds == 1 else set()) | ({"select_top"} if self.select_top is None else set()) | ({"pool_tickers"} if not self.pool_tickers else set())
        # 켈리 칸(2026-11-15)도 같다: 기본(문턱)이면 trade 의 세 칸을 해시에서 빼서 그 전 결과의 지문을 그대로 둔다
        exd: dict = {k_: True for k_ in ex}
        if self.trade.sizing == "threshold":
            exd["trade"] = {"sizing": True, "kelly_fraction": True, "kelly_min_history": True}
        return hashlib.sha1(self.model_dump_json(exclude=exd).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 플러그인 로딩
# ---------------------------------------------------------------------------
def load_plugin(name: str):
    mod = importlib.import_module(f"models.{name}")
    for attr in ("NAME", "INPUT_SHAPE", "TARGET_TYPES", "PARAMS", "Model"):
        if not hasattr(mod, attr):
            raise AttributeError(f"models/{name}.py 에 {attr} 선언이 없습니다")
    return mod


def pinned_model_params(model_file: str, recorded: dict | None, given: dict | None) -> tuple[dict, list[str]]:
    """옛 결과를 같은 실험으로 다시 돌릴 때의 model_params. 결과에 기록된 실제 파라미터(recorded)에 없는 키 중
    모델이 LEGACY_DEFAULTS 로 선언한 것은 '그때 쓰인 값'으로 채운다(기본값이 바뀐 파라미터 — 예: ARIMA 추정 제약, 2026-11-06).
    반환 (model_params, 채운 내용 설명)."""
    plugin = load_plugin(model_file)
    out = dict(given or {})
    notes = []
    if recorded is None:
        return out, notes
    for k, v in (getattr(plugin, "LEGACY_DEFAULTS", None) or {}).items():
        if k not in recorded and k not in out:
            out[k] = v
            notes.append(f"{k}={v} (이 결과를 낼 때의 값 — 결과 파일에 기록이 없어 그때 고정값으로 채움)")
    # 처음부터 있던 파라미터의 '기본값'이 바뀐 경우(PIN_RECORDED — 예: SARIMA 의 D, 2026-11-10 에 1 → 0): 옛 결과에는 그때 값이 기록돼 있지만
    # 설정에는 적혀 있지 않아(기본값으로 돌았으므로) 그대로 다시 돌리면 새 기본값이 들어간다 → 기록된 값이 지금 기본값과 다르면 그 값으로 고정한다.
    for k in getattr(plugin, "PIN_RECORDED", ()) or ():
        if k in recorded and k not in out and recorded[k] != (plugin.PARAMS.get(k) or {}).get("default"):
            out[k] = recorded[k]
            notes.append(f"{k}={recorded[k]} (이 결과를 낼 때 실제로 쓰인 값 — 그 뒤 기본값이 {plugin.PARAMS[k].get('default')} 로 바뀜)")
    return out, notes


def list_plugins() -> list[dict]:
    out = []
    for p in sorted(Path(__file__).resolve().parent.parent.joinpath("models").glob("*.py")):
        if p.stem in ("__init__", "base"):
            continue
        m = load_plugin(p.stem)
        uses_val = bool(getattr(getattr(m, "Model", None), "USES_VALIDATION", False)) if m.INPUT_SHAPE != "rule" else False
        out.append({"file": p.stem, "name": m.NAME, "input_shape": m.INPUT_SHAPE, "target_types": m.TARGET_TYPES, "params": m.PARAMS,
                    "param_groups": getattr(m, "PARAM_GROUPS", None), "uses_validation": uses_val, "seq_len_default": getattr(m, "SEQ_LEN_DEFAULT", None), "doc": (m.__doc__ or "").strip().split("\n")[0]})
    return out


# ---------------------------------------------------------------------------
# 데이터 조립
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    prices: pd.DataFrame  # 매매 주기로 리샘플된 OHLCV
    X: pd.DataFrame  # 결정일 인덱스 × 변수 (t 마감 시점 정보만)
    y: pd.Series  # 결정일 인덱스, 미래 타겟
    feature_names: list[str]
    leak_warning: bool
    lag_unconfirmed: list[str] = None  # lag_days 가 기본값(미확인)인 변수
    lag_days_used: dict | None = None
    transforms_used: dict | None = None  # 변수별 실제 적용 변환 (설정 재정의 > 마스터 기본)
    levels: pd.DataFrame | None = None  # 변환하지 않은 수준값(발표시차·정렬 적용). levels 모델(VECM) 전용
    transform_basis: dict | None = None  # 변수별 변환의 '한 칸': model_period(모델 주기) · native(변수의 원래 주기) · level(변화형 아님)
    pool: list | None = None  # 묶어 학습: [(종목, 그 종목의 Dataset)] — 학습 행만 보탠다(pool_training_rows)


_TR_RE = re.compile(r"^(pct|diff)([1-9]\d*)?$")   # 숫자는 1 이상, 앞에 0 없이(pct0·pct05 거절 — 2026-11-13 검토 8). 숫자를 비우면 1
_FREQ_RANK = {"D": 0, "W": 1, "M": 2, "Q": 3, "H": 4, "Y": 5}  # 작을수록 촘촘한 주기


def apply_transform(s: pd.Series, transform: str) -> pd.Series:
    """변수 변환. 변수의 원래 주기(관측 간격) 기준이므로 pct5 는 '5관측 전 대비' (일간이면 5거래일, 월간이면 5개월)."""
    t = (transform or "none").strip().lower()
    if t in ("", "none"):
        return s
    if t == "log":
        return np.log(s)
    if t == "logdiff":
        return np.log(s).diff()
    m = _TR_RE.match(t)
    if m:
        n = int(m.group(2) or 1)
        return s.pct_change(n) if m.group(1) == "pct" else s.diff(n)
    raise ValueError(f"알 수 없는 변환 '{transform}' (none|pct|pct{{n}}|diff|diff{{n}}|log|logdiff)")


# 변환 선택지 — 한 곳에서 정의한다(2026-11-14). 서버 meta(실험 화면·[값 보기] 고치기·변수 추가)와 도움말 표({{TRANSFORMS}})가 모두 이것을 읽는다.
# 엔진은 이 밖의 pct{n}·diff{n}(예: pct3)도 계산한다(_TR_RE) — 변수 사전에 그런 값이 있으면 화면이 그 값을 함께 보여 준다.
# n 은 '그 변수 자신의 관측 n개 전'이다(월간 pct12 = 12개월 전, 일간 pct5 = 5거래일 전).
TRANSFORM_OPTIONS: list[dict] = [
    {"value": "none", "name": "그대로", "formula": "x_t", "meaning": "받은 값을 바꾸지 않는다",
     "use": "이미 비율·성장률·흐름인 값(GDP 성장률, 순매수대금)", "caution": "가격·지수 같은 수준값에 쓰면 추세가 그대로 남는다"},
    {"value": "pct", "name": "변화율", "formula": "x_t ÷ x_(t−1) − 1", "meaning": "직전 관측 대비 몇 % 변했나",
     "use": "가격·지수·환율·잔액·통화량(보통 이것)", "caution": "0 이하 값이 있으면 그날은 계산되지 않음. % 단위 값(금리)에 쓰면 왜곡"},
    {"value": "pct4", "name": "4개 전 대비 변화율", "formula": "x_t ÷ x_(t−4) − 1", "meaning": "분기 자료면 전년 같은 분기 대비",
     "use": "계절성이 남은 분기 수준값", "caution": "처음 4개 관측은 값이 없다"},
    {"value": "pct5", "name": "5개 전 대비 변화율", "formula": "x_t ÷ x_(t−5) − 1", "meaning": "일간 자료면 한 주(5거래일) 변화",
     "use": "일간 가격의 주간 흐름", "caution": "이웃한 날의 값이 겹쳐 서로 닮는다"},
    {"value": "pct12", "name": "12개 전 대비 변화율", "formula": "x_t ÷ x_(t−12) − 1", "meaning": "월 자료면 전년 같은 달 대비",
     "use": "계절성이 남은 월 수준값(소비자물가·고용 원계열 등) — 자료 점검표가 권함", "caution": "처음 12개월은 값이 없다. 이웃한 달끼리 겹쳐 닮는다"},
    {"value": "pct20", "name": "20개 전 대비 변화율", "formula": "x_t ÷ x_(t−20) − 1", "meaning": "일간 자료면 약 한 달(20거래일) 변화",
     "use": "일간 가격의 한 달 흐름", "caution": "이웃한 날의 값이 많이 겹친다"},
    {"value": "diff", "name": "차이", "formula": "x_t − x_(t−1)", "meaning": "직전 관측보다 얼마나 올랐나(단위 그대로)",
     "use": "금리·비율(%)·확산지수·재고량", "caution": "단위가 큰 값(지수 수천)은 크기가 시기마다 달라진다 — 그런 값은 pct"},
    {"value": "diff4", "name": "4개 전과의 차이", "formula": "x_t − x_(t−4)", "meaning": "분기 자료면 전년 같은 분기와의 차이",
     "use": "계절성이 남은 분기 비율값", "caution": "처음 4개 관측은 값이 없다"},
    {"value": "diff5", "name": "5개 전과의 차이", "formula": "x_t − x_(t−5)", "meaning": "일간 자료면 한 주 동안의 변화",
     "use": "일간 금리의 주간 흐름", "caution": "이웃한 날의 값이 겹친다"},
    {"value": "diff12", "name": "12개 전과의 차이", "formula": "x_t − x_(t−12)", "meaning": "월 자료면 전년 같은 달과의 차이",
     "use": "계절성이 남은 월 비율값(연체율·고용률 원계열 등)", "caution": "처음 12개월은 값이 없다"},
    {"value": "diff20", "name": "20개 전과의 차이", "formula": "x_t − x_(t−20)", "meaning": "일간 자료면 약 한 달 동안의 변화",
     "use": "일간 금리의 한 달 흐름", "caution": "이웃한 날의 값이 많이 겹친다"},
    {"value": "log", "name": "로그", "formula": "ln x_t", "meaning": "수준을 로그 눈금으로(곱셈 변화를 덧셈으로)",
     "use": "수준 자체를 넣고 싶을 때만", "caution": "추세가 남아 정상성 검사에 걸리기 쉽다. 0 이하 값은 계산 안 됨"},
    {"value": "logdiff", "name": "로그 차이", "formula": "ln x_t − ln x_(t−1)", "meaning": "로그 수익률 — 작은 변화에서는 변화율과 거의 같다",
     "use": "가격·지수(변화율 대신)", "caution": "0 이하 값은 계산 안 됨"},
]


def build_dataset(cfg: ExperimentConfig, dv: DataVersion, reg: Registry) -> Dataset:
    if cfg.mode == "forecast":
        # 대상 = 마스터 변수(예: F11 주택매매가격지수). '가격' 틀을 빌려 쓴다: open=close=수준값, 거래량 없음.
        if not cfg.target_variable:
            raise ValueError("forecast 모드에는 target_variable 이 필요합니다")
        tv = reg.get(cfg.target_variable)
        if tv.freq not in ("D", "W", "M"):
            raise ValueError(f"대상 변수 주기 {tv.freq} 는 아직 지원하지 않습니다(D/W/M)")
        level = dv.wide([cfg.target_variable])[cfg.target_variable].dropna().astype(float)
        if cfg.start:
            level = level.loc[cfg.start :]
        if cfg.end:
            level = level.loc[: cfg.end]
        if (level <= 0).any():
            raise ValueError("대상 변수에 0 이하 값이 있어 변화율 타겟을 만들 수 없습니다")
        prices = pd.DataFrame({"open": level, "high": level, "low": level, "close": level, "volume": np.nan})
        prices.index = pd.DatetimeIndex(prices.index).normalize(); prices.index.name = "date"
        freq = tv.freq
    else:
        daily = dv.read_prices(cfg.ticker)
        if cfg.start:
            daily = daily.loc[cfg.start :]
        if cfg.end:
            daily = daily.loc[: cfg.end]
        prices = cal.resample_prices(daily, cfg.frequency)
        freq = cfg.frequency
    calendar = cal.trading_calendar(prices)
    # 외생변수를 맞추는 '정보 시점'. 매매 모드는 결정일 자체. 예측 모드 release 는 대상 값의 공개 시점(기준일 + 대상 lag).
    info_calendar = calendar
    if cfg.mode == "forecast" and cfg.forecast_info_time == "release":
        lag_t = int(cfg.lag_override.get(cfg.target_variable, reg.get(cfg.target_variable).lag_days))
        info_calendar = pd.DatetimeIndex(calendar + pd.Timedelta(days=lag_t), name=calendar.name)

    parts = []
    var_ids: list[str] = []
    level_cols: dict[str, pd.Series] = {}
    tr_basis: dict[str, str] = {}
    if cfg.use_technical:
        parts.append(select_technical(technical_block(prices, freq, with_volume=(cfg.mode != "forecast")), cfg.technical_features))
    if cfg.variable_ids:
        # 중복(duplicate_of) → 대표 ID 치환, active=N 은 빼고 기록(Registry.resolve_detail). 설정 파일의 ID 는 그대로 두고 여기서만 바꾼다.
        var_ids, notes = reg.resolve_many(cfg.variable_ids)
        for n in notes:
            print(f"  [변수] {n}")
        vt_map, n1 = reg.remap_overrides(cfg.variable_ids, cfg.variable_transforms)
        lo_map, n2 = reg.remap_overrides(cfg.variable_ids, cfg.lag_override)
        for n in n1 + n2:
            print(f"  [변수 재정의] {n}")
        wide = dv.wide(var_ids)
        cols = {}
        level_cols: dict[str, pd.Series] = {}
        for vid in var_ids:
            if vid not in wide.columns:
                raise KeyError(f"데이터 버전에 변수 {vid} 없음")
            v = reg.get(vid)
            tr = vt_map.get(vid, v.transform or "none")
            # 변화형 변환(pct·diff·logdiff, 간격을 따로 적지 않은 것)의 '한 칸':
            #   변수 주기가 모델 주기보다 촘촘하면(일간 변수 → 월간·주간 모델) 모델 주기 한 칸의 변화 = 결정일마다 알 수 있는 값(발표시차 적용)을 먼저 뽑고 그 사이의 변화.
            #   그 밖(같은 주기, 또는 월간 변수 → 일간 모델)은 변수의 원래 주기에서 변화를 구한 뒤 결정일에 맞춘다.
            # 2026-11-10 전에는 늘 원래 주기에서 먼저 변환해, 월간 모델에 들어간 일간 변수가 '그 달 마지막 하루의 변화'였다(점검 M4 — KOSPI 월간 행과 실제 월간 수익률의 상관 −0.06).
            # pct5·diff20 처럼 간격을 적은 변환은 '변수의 원래 주기로 n 칸'이라는 뜻 그대로 둔다.
            period_tr = tr.strip().lower() in ("pct", "diff", "logdiff") and _FREQ_RANK.get(str(v.freq), 0) < _FREQ_RANK.get(str(freq), 0)
            tr_basis[vid] = "model_period" if period_tr else ("native" if tr not in ("", "none", "log") else "level")
            raw = wide[vid].dropna() if period_tr else apply_transform(wide[vid].dropna(), tr)
            bad = raw.index[np.isinf(raw.values.astype(float))]
            if len(bad):
                raise ValueError(
                    f"변수 {vid} 에 변환 {tr} 을 적용하면 무한대/계산불가 값이 {len(bad)}개 생깁니다 (예: {bad[0].date()}). "
                    f"원값에 0 이나 음수가 있는 시리즈(이미 변화율·성장률인 값, 수지, 순매수)에는 pct/log 를 쓸 수 없습니다 — "
                    f"변수 사전의 transform 을 none 또는 diff 로 바꾸거나 실험 설정에서 재정의하세요")
            lag = lo_map.get(vid, v.lag_days)
            name = vid if tr in ("", "none") else f"{vid}_{tr}"
            cols[name] = cal.align_to_calendar(raw, info_calendar, lag_days=lag, fill=cfg.fill, apply_lag=cfg.apply_lag).set_axis(calendar)
            if period_tr:
                cols[name] = apply_transform(cols[name], tr)  # 결정일 사이(모델 주기 한 칸)의 변화
                bad2 = cols[name].index[np.isinf(cols[name].values.astype(float))]
                if len(bad2):
                    raise ValueError(f"변수 {vid} 에 변환 {tr} 을 적용하면 무한대 값이 {len(bad2)}개 생깁니다 (예: {bad2[0].date()}) — transform 을 none 또는 diff 로 바꾸세요")
            level_cols[vid] = cal.align_to_calendar(wide[vid].dropna(), info_calendar, lag_days=lag, fill=cfg.fill, apply_lag=cfg.apply_lag).set_axis(calendar)
            _al = cols[name]
            _chg = _al.dropna()
            _last_change = _chg.index[np.flatnonzero(np.r_[True, np.diff(_chg.values.astype(float)) != 0])[-1]] if len(_chg) else None
            _stale = int((calendar > _last_change).sum()) if _last_change is not None else None
            log_d.debug("변수 %s: 원자료 %s~%s(%d개, 주기 %s) → 변환 %s → 발표시차 %d일 → 달력 정렬 %s · 첫 사용일 %s · 마지막 값 변경 %s(그 뒤 %s행 같은 값)",
                        vid, raw.index[0].date() if len(raw) else None, raw.index[-1].date() if len(raw) else None, len(raw), v.freq, tr, lag,
                        describe(_al), _chg.index[0].date() if len(_chg) else None, _last_change.date() if _last_change is not None else None, _stale)
            _cal_days = {"D": 365 / 252, "W": 7, "M": 30.4}.get(cfg.frequency if cfg.mode == "trading" else (reg.get(cfg.target_variable).freq if cfg.target_variable else "M"), 30.4)
            _stale_max = 3 * {"D": 1, "W": 7, "M": 31, "Q": 92}.get(str(v.freq), 31) / _cal_days + 5  # 발표 주기 3번 이상 안 바뀌면 경고
            if _stale is not None and _stale > _stale_max:
                log_d.warning("변수 %s 의 값이 %s 이후 %d행 동안 바뀌지 않음 — 데이터가 끝났다면 그 뒤는 낡은 값이 이어진 입력입니다(데이터 사용 종료일을 앞당기세요)",
                              vid, _last_change.date(), _stale)
        parts.append(pd.DataFrame(cols))
    X = pd.concat(parts, axis=1).reindex(calendar) if parts else pd.DataFrame(index=calendar)
    if cfg.mode == "forecast":
        y = cal.make_forecast_target(prices["close"], k=cfg.target.k, kind=cfg.target.kind, threshold=cfg.target.threshold)
    else:
        y = cal.make_target(prices, k=cfg.target.k, kind=cfg.target.kind, threshold=cfg.target.threshold)
    if len(X.columns):
        cal.assert_no_target_in_inputs(X, y, cfg.target.k)
    vt_map, _ = reg.remap_overrides(cfg.variable_ids, cfg.variable_transforms)
    lo_map, _ = reg.remap_overrides(cfg.variable_ids, cfg.lag_override)
    used = {vid: lo_map.get(vid, reg.get(vid).lag_days) for vid in var_ids}
    trs = {vid: vt_map.get(vid, reg.get(vid).transform or "none") for vid in var_ids}
    unconf = [vid for vid in used if not reg.get(vid).lag_confirmed]
    if unconf:
        print(f"  [lag 미확인] {len(unconf)}/{len(used)}개 변수가 마스터에서 lag 확인 안 됨 (결과 json 의 lag_unconfirmed 참조)")
    levels = pd.DataFrame(level_cols).reindex(calendar) if cfg.variable_ids else pd.DataFrame(index=calendar)
    log_d.debug("데이터셋: %s · %s · 가격/대상 %s", describe(X, "입력 X"), describe(y, "타겟 y"), describe(prices["close"], "close"))
    return Dataset(prices, X, y, list(X.columns), cal.fill_leaks_future(cfg.fill), unconf, used, trs, levels, tr_basis)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
# 결과 계산 방식이 바뀐 날. 결과 json 에 'engine_rev' 로 남고, tools/rerun_stale.py 가 이 값으로 옛 결과를 찾는다.
# 2026-11-07: 창 입력 모델 눈금을 학습 폴드 전체 행으로 · 중복 변수 ID 의 변환·lag 설정을 대표 ID 로.
# 2026-11-10: 봉인 경계를 '쓰이는 가격' 기준으로(개발 평가가 h 행 일찍 끝남) · 모델보다 촘촘한 변수의 pct/diff 는 결정일 사이의 변화 ·
#             폴드마다 시드(날짜로 정함)·시드 반복(n_seeds)·LightGBM 이 엔진 시드를 씀 · SARIMA 기본 D=0+상수항 · ARIMA·SARIMA 여러 출발점 추정 ·
#             k>1 검정의 분산을 절단 핵+HLN 보정으로 · 학습 창 평균 기준 지표(cw_vs_mean 등)·기준 열(bench_mean) 추가.
#             → 그 전 결과는 전부 '옛 방식'(V·Holm·순위에서 제외). 다시 돌려야 한다(tools/rerun_stale.py).
ENGINE_REV = "2026-11-10"


def fold_seed(seed: int, test_start, rep: int = 0) -> int:
    """폴드 하나·반복 하나의 시드. 실험 시드, 그 폴드의 첫 테스트 결정일, 반복 번호(0부터)만으로 정한다.
    왜 폴드마다 따로 정하나(2026-11-10): 그 전에는 실험 시작에 한 번만 시드를 고정해, 어떤 폴드의 난수는 앞 폴드들이 난수를 얼마나 썼는지에 달려 있었다.
    그래서 (1) 폴드 하나만 다시 계산하면(계산 따라가기) 실험 때와 다른 값이 나왔고 (2) 이어 돌리기는 난수 상태를 통째로 저장해야 했고
    (3) 폴드 구성이 조금만 달라도 뒤 폴드 전부가 달라졌다. 이제 폴드는 자기 날짜로 시드가 정해져 어떤 순서·묶음으로 돌려도 같은 값이 나온다."""
    d = pd.Timestamp(test_start).strftime("%Y-%m-%d")
    return int(hashlib.sha256(f"{int(seed)}|{d}|{int(rep)}".encode()).hexdigest()[:8], 16) % (2**31 - 1)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _code_version() -> str:
    """결과를 만든 코드의 git 커밋. 커밋 안 된 수정이 있으면 '-dirty' 를 붙인다(그 결과는 커밋만으로 재현 불가하다는 표시).
    데이터·결과 폴더의 변경은 코드가 아니므로 dirty 판정에서 뺀다."""
    cwd = Path(__file__).resolve().parent
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"
    try:
        st = subprocess.check_output(["git", "status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)data/versions"],
                                     cwd=cwd.parent, stderr=subprocess.DEVNULL).decode()
        if any(l.strip() for l in st.splitlines()):
            h += "-dirty"
    except Exception:
        pass
    return h


def default_device() -> str:
    """CUDA 가 있으면 cuda. window 모델(LSTM, BIVA)만 영향."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _embargo(cfg, k: int) -> int:
    """학습 끝과 다음 구간(검증·테스트) 사이에 비우는 행 수. 결정일 t 의 타겟이 t+k(예측) / t+1+k(매매) 까지의 값을 쓰므로
    그만큼 비워야 학습 정답이 다음 구간의 값과 겹치지 않는다(purging)."""
    if cfg.validation.embargo is not None:
        return int(cfg.validation.embargo)
    return k if cfg.mode == "forecast" else 1 + k




def seal_horizon(cfg, k: int) -> int:
    """결정일 t 의 정답·수익이 쓰는 마지막 값이 몇 행 뒤인가: 매매 1+k(다음 날 시가에 사서 그 k 주기 뒤 시가까지), 예측 k."""
    return k if cfg.mode == "forecast" else 1 + k


def seal_cut(cfg, calendar_all: pd.DatetimeIndex, k: int) -> pd.Timestamp | None:
    """봉인 시작일(sealed_from: '이 날부터의 값은 개발 중 쓰지 않는다')을 결정일 기준 경계로 바꾼다. 반환 = 첫 봉인 결정일(없으면 None).

    결정일 t 의 정답(타겟)과 보유수익은 t+h 행까지의 값을 쓴다(h = seal_horizon). 그래서
      개발  = 그 마지막 값의 날짜가 봉인 시작 전인 결정일  (t+h 행의 날짜 < sealed_from)
      봉인  = 봉인 시작일 이후의 값이 하나라도 쓰이는 결정일 (t+h 행의 날짜 ≥ sealed_from)
    으로 나눈다. 빠지는 날·겹치는 날이 없고, 개발 구간의 어떤 수치도 봉인 시작일 이후의 값을 쓰지 않는다.
    2026-11-10 전에는 '결정일 < sealed_from' 으로만 잘라, 개발의 마지막 h 개 결정이 봉인 구간의 가격으로 채점됐다
    (월간이면 봉인 첫 달의 수익이 개발 성과에 들어가고 봉인 평가에서는 빠졌다 — 점검 T5)."""
    if not cfg.validation.sealed_from:
        return None
    sf = pd.Timestamp(cfg.validation.sealed_from)
    n0 = int(calendar_all.searchsorted(sf))  # sealed_from 전의 행 수
    c = max(0, n0 - seal_horizon(cfg, k))
    return calendar_all[c] if c < len(calendar_all) else sf


def make_folds(cfg, ds, k) -> tuple[list, bool]:
    """검증 설정으로 폴드 목록을 만든다(학습 모델 실행·규칙 모델 평가 구간·계산 따라가기(explain)가 모두 이 함수를 쓴다).
    1회 분할: 폴드 1개(테스트 시작 전 전부 − 비움 / 테스트 시작~종료). 워크포워드: calendar.walkforward_folds.
    반환 (folds, is_split)."""
    v = cfg.validation
    emb = _embargo(cfg, k)
    calendar_all = cal.trading_calendar(ds.prices)
    ts = pd.Timestamp(v.test_start or v.split_date) if (v.test_start or (v.mode == "split" and v.split_date)) else None
    te_end = pd.Timestamp(v.test_end) if v.test_end else None
    if v.mode == "split":
        if ts is None:
            raise ValueError("1회 분할에는 테스트 시작일(test_start)이 필요합니다")
        calendar_dev = calendar_all
        sf = seal_cut(cfg, calendar_all, k)  # 첫 봉인 결정일(가격 기준 경계를 결정일로 옮긴 것)
        if sf is not None:
            calendar_dev = calendar_dev[calendar_dev < sf]
        train_idx = calendar_dev[calendar_dev < ts]
        train_idx = train_idx[:-emb] if emb and len(train_idx) > emb else train_idx  # purging/embargo
        test_idx = calendar_dev[calendar_dev >= ts]
        if te_end is not None:
            test_idx = test_idx[test_idx <= te_end]
        vf = pd.Timestamp(v.val_start) if v.val_start else None
        if vf is not None and int((train_idx < vf).sum()) < 30:
            raise ValueError(f"검증 시작일({vf.date()}) 이전 학습 구간이 30개 미만입니다")
        if len(train_idx) < 30 or len(test_idx) == 0:
            raise ValueError("테스트 시작일 기준 학습/테스트 구간이 너무 짧습니다")
        return [cal.Fold(train_idx, test_idx, val_from=vf)], True
    start_row = int(calendar_all.searchsorted(ts)) if ts is not None else None
    folds = cal.walkforward_folds(calendar_all, train_len=v.train_len, test_len=v.test_len, k=k, embargo=emb, start=start_row,
                                  sealed_from=seal_cut(cfg, calendar_all, k))
    if ts is not None and folds and folds[0].test[0] > ts:
        print(f"  [기간] 테스트 시작 {ts.date()} 에는 학습 창({v.train_len})이 모자라 첫 테스트가 {folds[0].test[0].date()} 부터입니다", flush=True)
    if te_end is not None:
        folds = [cal.Fold(f.train, f.test[f.test <= te_end]) for f in folds if f.test[0] <= te_end]
    return folds, False

def _eval_window(cfg, ds, k):
    """학습 없이, 같은 검증 설정이면 학습 모델이 평가받는 첫날·마지막 날을 계산한다(규칙 모델의 평가 구간 맞춤용).
    make_folds 와 같은 폴드를 써서 첫 폴드 테스트 시작 ~ 마지막 폴드 테스트 끝을 돌려준다.
    계산할 수 없으면 (None, None) — 그때는 전체 기간을 그대로 쓴다."""
    try:
        folds, _ = make_folds(cfg, ds, k)
    except ValueError:
        return None, None
    folds = [f for f in folds if len(f.test)]
    return (folds[0].test[0], folds[-1].test[-1]) if folds else (None, None)


def _split_fit(m, X, y, n_val: int, emb: int):
    """검증을 쓰는 모델: 뒤쪽 n_val 개를 검증, 그 앞 emb 개는 버리고(학습 정답이 검증 구간의 값을 쓰지 않게 — purging) 나머지로 학습.
    검증이 너무 작거나 학습이 모자라면 검증 없이 전체 학습. (학습 개수, 검증 개수) 를 돌려준다."""
    n = len(X)
    t0 = time.perf_counter()
    if n_val >= 10 and n - n_val - emb >= 10:
        cut = n - n_val - emb
        m.fit(X[:cut], y[:cut], X[n - n_val:], y[n - n_val:])
        log_f.debug("fit: 학습 %s · %s · 검증 %d개(그 앞 %d개 비움) · %.2fs", describe(np.asarray(X[:cut]), "X"), describe(np.asarray(y[:cut]), "y"), n_val, emb, time.perf_counter() - t0)
        return cut, n_val
    m.fit(X, y)
    log_f.debug("fit: 학습 %s · %s · 검증 없음(검증 %d개 < 10 또는 학습 부족) · %.2fs", describe(np.asarray(X), "X"), describe(np.asarray(y), "y"), n_val, time.perf_counter() - t0)
    return n, 0


def _walkforward_predict(cfg, plugin, params, ds, device, folds, k, target_type, *, want_fit: bool = False, log: list | None = None,
                         capture: dict | None = None, diags: list | None = None):
    """폴드 목록을 돌며 학습→예측. (pred, n_folds, importances[, fit]). 워크포워드·1회 분할·봉인 평가가 같은 코드를 쓴다.
    want_fit=True 면 학습 구간 적합값(in-sample)도 돌려준다 — 1회 분할(노트북 방식) 차트용. 적합값은 성과·검정에 절대 쓰지 않는다.
    capture: dict 를 주면 마지막으로 처리한 폴드의 중간 객체(학습 표·스케일러·적합된 모델·테스트 입력 등)를 담는다 — 계산 따라가기
    (engine/explain.py)가 '같은 코드가 실제로 쓴 값'을 보여 주기 위한 것. 계산에는 영향 없음."""
    if not folds:
        raise ValueError("워크포워드 폴드가 0개입니다 — 기간이 train_len 보다 짧습니다")
    pred = pd.Series(np.nan, index=ds.prices.index, name="prediction")
    fit = pd.Series(np.nan, index=ds.prices.index, name="fit")
    importances: list[np.ndarray] = []
    ranks: list = []  # levels 모델: 폴드별 선택된 공적분 차수
    n_folds = 0
    emb_v = _embargo(cfg, k)
    uses_val = bool(getattr(plugin.Model, "USES_VALIDATION", True))
    # 시드 반복: 난수에 따라 달라지는 모델만. 결정적 모델은 여러 번 돌려도 같은 값이라 1번.
    n_rep = int(cfg.n_seeds) if bool(getattr(plugin.Model, "STOCHASTIC", False)) else 1

    def _fit_reps(f, X, y, n_val, X_te, X_fit=None, n_feat=None):
        """폴드 하나를 시드 n_rep 개로 학습·예측. 반환 (첫 시드 모델, 학습 수, 검증 수, 예측 평균, 적합값 평균, 중요도 평균, 폴드 진단).
        반복마다 fold_seed(실험 시드, 폴드 첫 테스트일, 반복 번호)로 전역 난수(random·numpy·torch)를 맞춘 뒤 모델을 만들고 학습한다
        (모델에는 시드를 넘기지 않는다 — 절대 규칙 3. 자체 난수기를 쓰는 LightGBM 은 이 numpy 전역 난수에서 시드를 뽑는다)."""
        ps, fits, imps, dgs = [], [], [], []
        m0 = None; n_tr = n_va = 0
        for r in range(n_rep):
            sd = fold_seed(cfg.seed, f.test[0], r)
            _seed(sd)
            m = plugin.Model(params, n_features=int(n_feat or len(ds.feature_names)), target_type=target_type, device=device)
            n_tr, n_va = _split_fit(m, X, y, n_val, emb_v)
            ps.append(np.asarray(m.predict(X_te), dtype=float))
            if X_fit is not None:
                fits.append(np.asarray(m.predict(X_fit), dtype=float))
            im = m.feature_importance()
            if im is not None:
                imps.append(np.asarray(im, dtype=float))
            dgs.append(dict(getattr(m, "diag", None) or {}, seed=sd))
            if m0 is None:
                m0 = m
        P = np.vstack(ps)
        fd = {"test_start": str(pd.Timestamp(f.test[0]).date()), "n_train": int(n_tr), "n_val": int(n_va), "n_seeds": n_rep, "reps": dgs,
              "pred_mean": float(np.mean(P)) if P.size else None,
              # 시드 사이 흩어짐: 같은 결정일의 예측이 시드에 따라 얼마나 다른가(표준편차)의 폴드 평균. 시드 1개면 없음
              "seed_sd": float(np.mean(np.std(P, axis=0, ddof=1))) if n_rep > 1 and P.shape[1] else None,
              # 시간 방향 흩어짐: 이 폴드의 예측(시드 평균)이 결정일에 따라 얼마나 다른가. 0 이면 이 폴드는 상수 예측
              "pred_sd": float(np.std(P.mean(axis=0))) if P.shape[1] > 1 else None,
              # 상수 판정은 최댓값 − 최솟값으로(표준편차는 같은 값들에도 반올림 오차로 0 이 아닐 수 있다)
              "pred_const": bool(np.ptp(P.mean(axis=0)) == 0.0) if P.shape[1] > 1 else None}
        if diags is not None:
            diags.append(fd)
        return m0, n_tr, n_va, P.mean(axis=0), (np.mean(fits, axis=0) if fits else None), (np.mean(imps, axis=0) if imps else None), fd

    def _nval(idx: pd.DatetimeIndex, f) -> int:  # 학습 표본(행 또는 창의 결정일) 중 검증에 쓸 개수
        if not uses_val:
            return 0
        if getattr(f, "val_from", None) is not None:
            return int((idx >= f.val_from).sum())
        return int(len(idx) * cfg.validation.val_frac)

    def _log(f, tr_idx, va_idx, te_idx):
        if log is None:
            return
        d = lambda ix: [str(ix[0].date()), str(ix[-1].date())] if len(ix) else None
        log.append({"train": d(tr_idx), "val": d(va_idx), "test": d(te_idx), "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)), "n_test": int(len(te_idx))})
        log_f.debug("폴드 %d/%d: 학습 %s(%d) · 검증 %s(%d) · 테스트 %s(%d)", len(log), len(folds), d(tr_idx), len(tr_idx), d(va_idx), len(va_idx), d(te_idx), len(te_idx))
    def _note(**kw):  # 마지막 폴드 기록에 덧붙임(기록만 — 계산은 바꾸지 않음, 2026-11-21 결과 꾸러미 '데이터 변화 단계')
        if log is not None and log:
            log[-1].update(kw)

    def _scale_rec(sc, cols):
        if isinstance(sc, MinMaxScaler):
            return {"kind": "minmax", "features": list(cols), "min": [float(x) for x in sc.data_min_], "max": [float(x) for x in sc.data_max_]}
        return {"kind": "standard", "features": list(cols), "mean": [float(x) for x in sc.mean_], "std": [float(x) for x in sc.scale_]}
    t_fold = time.time()
    last_print = time.time()
    for fi, f in enumerate(folds, 1):
        if fi > 1 and (time.time() - last_print > 10 or (fi - 1) % max(1, len(folds) // 10) == 0):
            el = time.time() - t_fold
            print(f"  폴드 {fi-1}/{len(folds)} 완료, 경과 {el/60:.1f}분, 예상 남은 {el/(fi-1)*(len(folds)-fi+1)/60:.1f}분", flush=True)
            last_print = time.time()
        tr = ds.X.loc[f.train].join(ds.y.loc[f.train].rename("y")).dropna()
        te = ds.X.loc[f.test]
        te_empty = (len(te) == 0) if not ds.feature_names else te.dropna().empty  # 변수 0개(series 모델)면 행만 본다 (.empty 는 열 0개도 True)
        if len(tr) < 30 or te_empty:
            continue
        if plugin.INPUT_SHAPE == "levels":
            # 다변량 수준값(VECM): [대상 수준, 변수 수준들]. 학습 창으로 적합 → 테스트 시점 t 마다 t 까지의 값으로 재귀 예측.
            Lall = pd.concat([ds.prices["close"].rename("__target__"), ds.levels], axis=1)
            Ltr = Lall.loc[f.train].dropna()
            if len(Ltr) < 30:
                continue
            m = plugin.Model(params, n_features=Lall.shape[1], target_type=target_type, device=device)
            m.fit(Ltr.values, np.array([]))
            _log(f, Ltr.index, pd.DatetimeIndex([]), te.index)
            skip = 0 if cfg.mode == "forecast" else 1
            need = int(getattr(m, "p", params.get("k_ar_diff", 2))) + 2
            Ld = Lall.dropna()
            row = pd.Series(np.arange(len(Ld)), index=Ld.index)
            r0 = int(row.loc[Ltr.index[0]])  # 학습 창 첫 행 = 0 (계절 더미 위상)
            for t in te.index:
                if t not in row.index:
                    continue
                hist = Ld.loc[:t].iloc[-max(need, 2):]
                if len(hist) < need:
                    continue
                path = np.asarray(m.forecast_levels(hist.values, k + skip, pos=int(row.loc[t]) - r0), dtype=float)
                base = path[skip - 1] if skip else float(hist.iloc[-1, 0])
                ratio = path[-1] / base
                pred.loc[t] = float(np.log(ratio)) if cfg.target.kind == "logreturn" else float(ratio - 1.0)
            if capture is not None:
                capture.update(kind="levels", fold=f, model=m, Lall=Lall, Ltr=Ltr, Ld=Ld, row=row, r0=r0, need=need, skip=skip, pred=pred.loc[te.index].copy())
            ranks.append(getattr(m, "rank", None))
            if log is not None and getattr(m, "diag", None):
                log[-1]["model_diag"] = m.diag
            n_folds += 1
            continue
        if plugin.INPUT_SHAPE == "series":
            # 단변량 통계 모델: 대상 시계열의 1기간 변화율만 사용. 결정일 t 까지의 변화율로 k 기간 앞을 예측해 복리로 묶는다.
            r_all = ds.prices["close"].pct_change()
            r_tr = r_all.loc[f.train].dropna().values
            if len(r_tr) < 30:
                continue
            m = plugin.Model(params, n_features=1, target_type=target_type, device=device)
            m.fit(r_tr, np.array([]))
            _log(f, r_all.loc[f.train].dropna().index, pd.DatetimeIndex([]), te.index)
            if getattr(m, "diag", None):  # 추정 진단(어느 출발점의 해를 썼나·로그우도) — 폴드 기록과 결과 요약에
                dg_ = {k_: v_ for k_, v_ in m.diag.items() if k_ != "tried"}
                if log is not None and log:
                    log[-1]["model_diag"] = dg_
                if diags is not None:
                    diags.append({"test_start": str(pd.Timestamp(f.test[0]).date()), "n_train": int(len(r_tr)), "n_val": 0, "n_seeds": 1, "reps": [dg_],
                                  "seed_sd": None, "pred_sd": None, "pred_const": None})
            def _agg(fc):  # 1기간 변화율 예측 k개 → 타겟 정의에 맞는 k기간 값
                fc = np.asarray(fc, dtype=float)[:k]
                return float(np.sum(np.log1p(fc))) if cfg.target.kind == "logreturn" else float(np.prod(1.0 + fc) - 1.0)
            # 매매 모드 타겟은 t+1 시가 진입이라 예측해야 할 구간이 [t+1, t+1+k] — 1기간 더 앞을 내다보고 첫 기간을 버린다.
            skip = 0 if cfg.mode == "forecast" else 1
            for t in te.index:
                hist = r_all.loc[:t].dropna().values  # t 마감까지 알려진 변화율 (미래 없음)
                pred.loc[t] = _agg(np.asarray(m.forecast(hist, k + skip), dtype=float)[skip:])
            if capture is not None:
                capture.update(kind="series", fold=f, model=m, r_all=r_all, r_tr=r_tr, skip=skip, te_index=te.index, pred=pred.loc[te.index].copy())
            if want_fit:
                for t in tr.index:
                    hist = r_all.loc[:t].dropna().values
                    if len(hist) < 10:  # 학습 창 맨 앞은 과거가 거의 없어 적합값을 만들지 않는다
                        continue
                    fit.loc[t] = _agg(np.asarray(m.forecast(hist, k + skip), dtype=float)[skip:])
            n_folds += 1
            continue
        # 스케일러: 학습 구간에만 fit (절대규칙 2)
        # input_scaler: 모델이 요구하면 MinMax(학습 구간 최소0·최대1). BIVA 의 BRITS 는 보간 추정치가 sigmoid 출력(0~1)이라
        # 원본처럼 0~1 로 맞춘 입력이어야 보간이 제 범위에 들어간다. 그 밖은 표준화.
        # 연관성 순위·입력 선별(2026-11-12): 이 폴드의 학습 행(tr)만으로 입력–정답의 |순위상관|을 잰다. select_top 이 있으면 상위 N 개 열만 쓴다.
        # 전체 자료로 변수를 고른 뒤 같은 자료로 평가하면 미래를 보고 고른 셈이 된다 — 그래서 폴드 안에서 한다.
        # 2026-11-13(독립 검토 13): 검증을 쓰는 폴드면 '검증 구간 + 그 앞 비움(emb_v)' 앞의 행으로만 잰다 — _split_fit 이 학습에 쓰는 행과 같다.
        # 검증 행은 조기 종료 판단에 쓰므로, 선별이 그 행을 보면 검증이 '처음 보는 자료'가 아니게 된다.
        # 창 입력 모델의 검증은 '창의 결정일' 단위인데, 창 목록은 정답이 있는 날로만 정해져(입력 열과 무관) 선별 전에 정확히 구할 수 있다.
        sel_tr = tr
        if uses_val:
            if plugin.INPUT_SHAPE == "window":
                _full0 = ds.X.loc[f.train].join(ds.y.loc[f.train].rename("y"))
                _yv = _full0["y"].to_numpy(float)[cfg.seq_len - 1:]
                _widx = _full0.index[cfg.seq_len - 1:][np.isfinite(_yv)]
                _nv = _nval(_widx, f)
                if _nv >= 10 and len(_widx) - _nv - emb_v >= 10:
                    sel_tr = tr[tr.index < _widx[len(_widx) - _nv - emb_v]]
            else:
                _nv = _nval(tr.index, f)
                if _nv >= 10 and len(tr) - _nv - emb_v >= 10:
                    sel_tr = tr.iloc[: len(tr) - _nv - emb_v]
        rel = fold_relevance(sel_tr[ds.feature_names].values, sel_tr["y"].values)
        keep = list(range(len(ds.feature_names)))
        if cfg.select_top:
            if cfg.select_top > len(ds.feature_names):
                raise ValueError(f"select_top({cfg.select_top})이 입력 변수 수({len(ds.feature_names)})보다 큽니다")
            keep = sorted(np.argsort(-rel, kind="stable")[: cfg.select_top].tolist())
        feat = [ds.feature_names[j] for j in keep]   # 이 폴드에서 쓰는 입력 열
        # 묶어 학습(2026-11-12, 표 입력 모델만): 다른 종목의 행을 학습 앞쪽에 보탠다. 검증은 대상 종목의 뒤쪽 행 그대로 두고,
        # 보태는 행은 정답이 쓰는 마지막 가격이 경계(검증을 쓰면 검증 시작일, 아니면 테스트 시작일) 앞인 것만(pool_training_rows).
        pool_X = pool_y = None
        n_val_t = _nval(tr.index, f)
        if ds.pool and plugin.INPUT_SHAPE == "table":
            if not (n_val_t >= 10 and len(tr) - n_val_t - emb_v >= 10):
                n_val_t = 0   # 대상 종목만으로 검증을 못 쓰는 폴드는 보탠 뒤에도 검증 없이(_split_fit 와 같은 조건)
            cut_d = tr.index[len(tr) - n_val_t] if n_val_t else f.test[0]
            prs = [pool_training_rows(pds, f.train[0], cut_d, seal_horizon(cfg, k)) for _, pds in ds.pool]
            prs = [r for r in prs if len(r)]
            if prs:
                pool_X = np.vstack([r[feat].values for r in prs]); pool_y = np.concatenate([r["y"].values for r in prs])
        fit_rows = tr[feat].values if pool_X is None else np.vstack([pool_X, tr[feat].values])   # 눈금을 맞추는 학습 행(보탠 행 포함)
        scaler = (MinMaxScaler() if params.get("input_scaler") == "minmax" else StandardScaler()).fit(fit_rows)
        fill_val = scaler.transform(fit_rows).mean(axis=0)  # 빈 칸을 채울 '학습 구간 평균'(표준화면 0)
        Xtr = scaler.transform(tr[feat].values)
        ytr = tr["y"].values
        te_valid = te.dropna()
        Xte = scaler.transform(te_valid[feat].values)
        if plugin.INPUT_SHAPE == "window":
            # window 는 연속 행이 필요 → 학습 구간 전체(X,y 정렬 유지)로 윈도우 생성
            full = ds.X.loc[f.train].join(ds.y.loc[f.train].rename("y"))
            # 빈 칸 처리: 앞의 값으로만 채운다(ffill). 변수가 늦게 시작해 남는 앞쪽 빈 칸은 '평균(표준화 0)'으로 둔다.
            # (2026-11-05 수정: 이전엔 bfill 로 첫 관측값을 과거로 끌어와 미래 정보를 썼다.) 결측을 스스로 보간하는 모델
            # (ACCEPTS_NAN, 예: BIVA 의 BRITS)은 빈 칸을 NaN 그대로 받는다.
            _wants = getattr(plugin.Model, "wants_nan", None)
            accepts_nan = bool(getattr(plugin.Model, "ACCEPTS_NAN", False)) and bool(_wants(params) if _wants else False)
            if accepts_nan and params.get("input_scaler", "standard") != "minmax" and fi == 1:
                print("  [경고] BRITS 보간을 켰는데 입력이 표준화(standard)입니다 — BRITS 보간값은 sigmoid(0~1) 출력이라 빈 칸이 0~1 로만 채워집니다. input_scaler=minmax 권장", flush=True)
            # 눈금: 창을 만드는 학습 폴드의 모든 행으로 맞춘다(빈 칸은 sklearn 이 무시). 2026-11-07 전에는 위의 '빈 칸 없는 행'(tr)으로만 맞춰
            # 늦게 시작하는 변수가 있으면 그보다 앞의 행이 학습 눈금 밖(MinMax 면 0~1 밖)으로 나갔다(repro_biva_orig: 135행 중 63행으로 맞춤 → −2.2~4.8).
            scaler = (MinMaxScaler() if params.get("input_scaler") == "minmax" else StandardScaler()).fit(full[feat].ffill().values)
            Xf = scaler.transform(full[feat].ffill().values)
            fill_val = np.nanmean(Xf, axis=0)  # 빈 칸을 채울 '학습 구간 평균'(표준화면 0)
            n_blank_fill = int(np.isnan(Xf).sum()) if not accepts_nan else 0
            if not accepts_nan:
                Xf = np.where(np.isnan(Xf), fill_val[None, :], Xf)
            Xw, yw = cal.make_windows(Xf, full["y"].values, cfg.seq_len)
            ok = np.isfinite(yw) & (np.ones(len(yw), bool) if accepts_nan else np.isfinite(Xw).all(axis=(1, 2)))
            win_idx = full.index[cfg.seq_len - 1:][: len(ok)][ok]  # 각 창의 결정일(창의 마지막 행)
            Xw, yw = Xw[ok], yw[ok]
            # 테스트: 학습 창 끝 seq_len-1 행을 앞에 붙여 첫 테스트일부터 윈도우 완성
            ctx = ds.X.loc[: f.test[0]].iloc[-cfg.seq_len:-1] if cfg.seq_len > 1 else ds.X.iloc[0:0]
            te_full = pd.concat([ctx, te])
            Xtf = scaler.transform(te_full[feat].ffill().values)
            if not accepts_nan:
                Xtf = np.where(np.isnan(Xtf), fill_val[None, :], Xtf)
            Xtw, _ = cal.make_windows(Xtf, np.zeros(len(Xtf)), cfg.seq_len)
            te_idx = te.index
            m, n_tr, n_va, p, fit_p, imp_f, fd = _fit_reps(f, Xw, yw, _nval(win_idx, f), Xtw, Xw if (want_fit and len(Xw)) else None, n_feat=len(feat))
            _log(f, win_idx[:n_tr], win_idx[len(win_idx) - n_va:] if n_va else win_idx[:0], te_idx)
            _note(scale=_scale_rec(scaler, feat), n_blank_filled=n_blank_fill, n_blank_left_nan=int(np.isnan(Xf).sum()) if accepts_nan else 0,
                  n_windows=int(len(win_idx)), n_windows_dropped=int((~ok).sum()))
            pred.loc[te_idx] = p[: len(te_idx)]
            if capture is not None:
                capture.update(kind="window", fold=f, model=m, scaler=scaler, fill_val=fill_val, Xw=Xw, yw=yw, win_idx=win_idx, n_tr=n_tr, n_va=n_va,
                               Xtw=Xtw, te_idx=te_idx, te_full=te_full, accepts_nan=accepts_nan, pred=pd.Series(p[: len(te_idx)], index=te_idx))
            if fit_p is not None:
                fit.loc[win_idx] = fit_p[: len(win_idx)]
        else:
            n_pool = 0 if pool_X is None else len(pool_X)
            Xfit_in, yfit_in = (Xtr, ytr) if not n_pool else (np.vstack([scaler.transform(pool_X), Xtr]), np.concatenate([pool_y, ytr]))
            m, n_tr, n_va, p, fit_p, imp_f, fd = _fit_reps(f, Xfit_in, yfit_in, n_val_t, Xte, Xtr if want_fit else None, n_feat=len(feat))
            if ds.pool:
                fd["n_pool"] = int(n_pool)
            _log(f, tr.index[:n_tr - n_pool], tr.index[len(tr) - n_va:] if n_va else tr.index[:0], te_valid.index)
            _note(scale=_scale_rec(scaler, feat), n_train_dropped_blank=int(len(ds.X.loc[f.train]) - len(tr)), n_test_not_predicted=int(len(te) - len(te_valid)))
            pred.loc[te_valid.index] = p
            if capture is not None:
                capture.update(kind="table", fold=f, model=m, scaler=scaler, tr=tr, te=te, te_valid=te_valid, Xtr=Xtr, ytr=ytr, Xte=Xte,
                               n_tr=n_tr, n_va=n_va, pred=pred.loc[te_valid.index].copy())
            if fit_p is not None:
                fit.loc[tr.index] = fit_p
        fd["relevance"] = rel.tolist()
        if cfg.select_top:
            fd["selected"] = keep
        if imp_f is not None:
            if len(keep) != len(ds.feature_names):   # 고르지 않은 입력의 중요도는 0 — 전체 변수 순서에 맞춰 놓는다
                full_imp = np.zeros(len(ds.feature_names)); full_imp[keep] = imp_f
                imp_f = full_imp
            importances.append(imp_f)
        if log is not None and log and fd.get("reps") and any(len(d) > 1 for d in fd["reps"]):
            log[-1]["model_diag"] = fd["reps"][0] if n_rep == 1 else fd["reps"]
        n_folds += 1
    if ranks:
        from collections import Counter

        print(f"  [VECM] 폴드별 공적분 차수 분포: {dict(Counter(ranks))} (0 = 공적분 없음 → 차분 VAR 로 대체)", flush=True)
        if log:
            ps = [d["model_diag"]["p"] for d in log if d.get("model_diag")]
            drops = sum(1 for d in log if d.get("model_diag", {}).get("dropped_constant_cols"))
            print(f"  [VECM] 폴드별 차분 시차 p 분포: {dict(Counter(ps))} · 상수 열을 뺀 폴드 {drops}개", flush=True)
    if want_fit:
        return pred, n_folds, importances, fit
    return pred, n_folds, importances


def _signal_from_prediction(cfg, pred: pd.Series, bench: pd.Series | None, target_type: str) -> pd.Series:
    """예측 → 목표 포지션(0~1).
    · 수익률·방향 타겟: 예측이 문턱을 넘으면 1(전부 보유), 아니면 0(현금).
    · 변동성 타겟(2026-11-12): 방향은 맞히지 않는다 → '살까 말까'가 아니라 '얼마나 실을까'. 비중 = 기준 변동성 ÷ 예측 변동성 을 0~1 로 자른 값.
      기준 변동성 = 그 결정일을 예측한 폴드의 학습 창 평균 변동성(bench — 결정일 이후 값을 쓰지 않고, 사용자가 숫자를 정할 필요가 없다).
      예측이 평소(학습 창 평균)보다 높으면 그 비율만큼 줄이고, 낮으면 전부 보유(빌려서 더 싣지는 않는다). 예측이 0 이하면 전부 보유."""
    if cfg.target.kind == "volatility":
        w = (bench / pred.clip(lower=1e-12)).clip(0.0, 1.0)
        return w.where(pred.notna() & bench.notna())
    if target_type == "regression":
        signal = (pred > cfg.trade.threshold).astype(float).where(pred.notna())
    else:
        signal = (pred > 0.5 + cfg.trade.threshold).astype(float).where(pred.notna())
    return signal


def _confidence(pred: pd.Series, ds, k: int):
    from engine.sizing import confidence_table, realized_hold_return
    return confidence_table(pred.dropna(), realized_hold_return(ds.prices, k))


def _apply_sizing(cfg, ds, pred: pd.Series, signal: pd.Series, k: int):
    """trade.sizing = kelly 면 문턱 신호 대신 켈리 비율(engine/sizing.py — 결정일에 결과가 나온 과거만). 아니면 그대로."""
    if cfg.trade.sizing != "kelly":
        return signal, None
    from engine.sizing import KELLY_MIN_HISTORY, kelly_positions, realized_hold_return
    mh = cfg.trade.kelly_min_history or KELLY_MIN_HISTORY.get(cfg.frequency, 250)
    w, info = kelly_positions(pred, realized_hold_return(ds.prices, k), k=k, fraction=cfg.trade.kelly_fraction, min_history=mh)
    return w.where(signal.notna()), info


def pool_training_rows(pds: "Dataset", lo, cut, h: int) -> pd.DataFrame:
    """묶어 학습에 보탤 행: 다른 종목 데이터셋 pds 에서 결정일이 lo 이상이고, 그 행의 정답이 쓰는 '마지막 가격의 날짜'(그 종목 달력으로 h = 1+k 행 뒤)가
    cut(대상 종목의 검증 또는 테스트 시작일)보다 앞인 행만. 빈 칸이 있는 행은 뺀다. 열 = 입력들 + 'y'.
    행 수가 아니라 날짜로 자르므로 거래일이 다른 종목이어도 경계 뒤의 가격이 학습에 들어오지 않는다."""
    cal_ = pds.prices.index
    pos = np.arange(len(cal_)) + int(h)
    last = pd.Series(pd.NaT, index=cal_, dtype="datetime64[ns]")
    ok = pos < len(cal_)
    last.iloc[np.flatnonzero(ok)] = cal_[pos[ok]]
    keep = (cal_ >= pd.Timestamp(lo)) & last.notna().values & (last.values < np.datetime64(pd.Timestamp(cut)))
    return pds.X.loc[keep, pds.feature_names].join(pds.y.loc[keep].rename("y")).dropna()


def fold_relevance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """입력 열마다 정답과의 |스피어만 순위상관|. 폴드의 '학습 행'만 받는다 — 테스트 구간의 값은 여기에 들어오지 않는다(절대규칙 2: 변수 선택은 학습 구간만으로).
    순위상관을 쓰는 이유: 크기가 큰 날 몇 개에 끌려가지 않고, 곧은 관계가 아니어도(한쪽으로 커지기만 하면) 잡는다. 정답이 0/1(분류)이어도 쓸 수 있다.
    값이 변하지 않는 열·빈 칸뿐인 열은 0. 빈 칸은 그 열에서만 빼고 계산한다."""
    from scipy.stats import rankdata

    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j]) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        a, b = rankdata(X[ok, j]), rankdata(y[ok])
        sa, sb = a.std(), b.std()
        if sa > 0 and sb > 0:
            out[j] = abs(float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)))
    return out


def relevance_summary(diags: list | None, names: list[str], select_top: int | None) -> dict | None:
    """폴드별 연관성(diags 의 relevance)을 변수별 순위표로. 순위는 '폴드마다 학습 창 안에서 잰 값'의 요약이라 전체 자료를 보고 매긴 순위가 아니다.
    - mean_abs_corr: 폴드 평균 |순위상관| · mean_rank: 폴드 평균 등수(1 = 가장 높음) · top_share: 상위 N 안에 든 폴드의 비율(선택 안정성)
    - selected_share: 실제로 골라 쓴 폴드의 비율(select_top 을 켰을 때만). 낮으면 폴드마다 다른 변수가 뽑혔다는 뜻 — 믿을 순위가 아니다."""
    rows = [d for d in (diags or []) if d.get("relevance") is not None]
    if not rows or not names:
        return None
    R = np.array([d["relevance"] for d in rows], dtype=float)
    if R.shape[1] != len(names):
        return None
    top_n = int(select_top) if select_top else min(10, len(names))
    rk = np.argsort(np.argsort(-R, axis=1, kind="stable"), axis=1, kind="stable") + 1   # 폴드별 등수(동률은 열 순서)
    feats = []
    for j, n in enumerate(names):
        f = {"name": n, "mean_abs_corr": float(R[:, j].mean()), "mean_rank": float(rk[:, j].mean()), "top_share": float((rk[:, j] <= top_n).mean())}
        if select_top:
            f["selected_share"] = float(np.mean([j in (d.get("selected") or []) for d in rows]))
        feats.append(f)
    feats.sort(key=lambda f: (f["mean_rank"], -f["mean_abs_corr"]))
    return {"method": "폴드마다 학습 창 안에서 잰 입력–정답의 |스피어만 순위상관|", "n_folds": len(rows), "top_n": top_n, "selected": bool(select_top), "features": feats}


def train_mean_benchmark(ds, folds) -> pd.Series:
    """결정일마다 '그 결정일을 예측한 폴드의 학습 창 타겟 평균'(분류면 학습 창의 양(+) 비율). '평균만 아는 예측' — Clark–West·DM 의 기준.
    학습 창(f.train)은 테스트 시작 앞 비움(embargo)까지 뺀 구간이라 이 평균에는 결정일 이후의 값이 들어가지 않는다. 모델과 무관하게 폴드와 타겟만으로 정해진다."""
    b = pd.Series(np.nan, index=ds.prices.index, name="bench_mean")
    for f in folds:
        if not len(f.test):
            continue
        ytr = ds.y.loc[f.train].dropna()
        if len(ytr):
            b.loc[f.test] = float(ytr.mean())
    return b


def mean_benchmark_metrics(y: np.ndarray, pred: np.ndarray, bench: np.ndarray, k: int) -> dict:
    """학습 창 평균(분류면 양(+) 비율)을 기준으로 한 지표(2026-11-10 독립 검토 F2).
    - cw_vs_mean(_p): Clark–West, 한쪽 p — "평균을 아는 것 말고 예측 정보가 더 있는가". 변수의 정보를 말할 때는 이것.
    - dm_vs_mean(_p): Diebold–Mariano, 양측 p — "이 방법이 '평균만 내는 예측'보다 정확한가". 음수 = 모델이 정확.
    - rmse_mean, r2_oos_vs_mean = 1 − Σ(y−ŷ)²/Σ(y−평균)² (Campbell·Thompson 의 표본 밖 R²)."""
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(bench)
    if ok.sum() < 10:
        return {}
    y, pred, bench = y[ok], pred[ok], bench[ok]
    cw, cw_p = clark_west(y, pred, bench, h=k)
    dm, dm_p = diebold_mariano(pred - y, bench - y, h=k)
    sse_b = float(((y - bench) ** 2).sum())
    return {"cw_vs_mean": cw, "cw_vs_mean_p": cw_p, "dm_vs_mean": dm, "dm_vs_mean_p": dm_p, "rmse_mean": float(np.sqrt(((y - bench) ** 2).mean())),
            "r2_oos_vs_mean": float(1 - ((y - pred) ** 2).sum() / sse_b) if sse_b > 0 else None, "n_vs_mean": int(ok.sum())}


def diag_summary(diags: list | None) -> dict | None:
    """폴드별 학습 진단(_walkforward_predict 의 diags)을 결과에 실을 요약으로. 계산에는 쓰지 않는다 — '모델이 실제로 학습됐는가'를 보여 주는 기록.
    - seeds: 시드 수, 시드 정하는 규칙, 시드 사이 예측 표준편차의 폴드 평균(시드 2개 이상일 때)
    - constant_folds: 테스트 예측이 한 폴드 안에서 전혀 변하지 않은 폴드 수(테스트가 2행 이상인 폴드 중)
    - trees_used / epoch_used: 조기 종료 뒤 실제로 쓰인 트리 수·반복 수의 분포(모델이 남길 때만). share_le2 = 2 이하인 비율"""
    if not diags:
        return None
    out: dict = {"n_folds": len(diags), "seeds": {"n_seeds": int(diags[0].get("n_seeds", 1)), "rule": "sha256('실험 시드|폴드 첫 테스트 결정일|반복 번호') 앞 8자리(16진) mod (2^31−1)"}}
    sds = [d["seed_sd"] for d in diags if d.get("seed_sd") is not None]
    if sds:
        out["seeds"]["pred_sd_across_seeds_mean"] = float(np.mean(sds))
    psd = [d["pred_sd"] for d in diags if d.get("pred_sd") is not None]
    if psd:
        out["constant_folds"] = int(sum(1 for d in diags if d.get("pred_const")))
        out["folds_checked_for_constant"] = len(psd)
        if sds and len(sds) == len(psd):
            out["seeds"]["pred_sd_within_fold_mean"] = float(np.mean(psd))
    reps = [r for d in diags for r in d.get("reps", [])]
    for key in ("trees_used", "epoch_used"):
        vals = np.array([r[key] for r in reps if r.get(key) is not None], dtype=float)
        if len(vals):
            q = np.percentile(vals, [0, 25, 50, 75, 100])
            out[key] = {"min": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "max": float(q[4]),
                        "share_le2": float(np.mean(vals <= 2)), "n": int(len(vals))}
    gains = [r["llf_gain"] for r in reps if r.get("llf_gain") is not None]
    if gains:  # 통계 모델의 여러 출발점 추정: 기본 출발점보다 로그우도가 높은 해를 찾은 폴드
        from collections import Counter
        out["multi_start"] = {"folds": len(gains), "improved_folds": int(sum(g > 0.01 for g in gains)), "max_llf_gain": float(max(gains)),
                              "start_used": dict(Counter(r.get("start_used") for r in reps if r.get("llf_gain") is not None)),
                              "not_converged_folds": int(sum(1 for r in reps if r.get("llf_gain") is not None and not r.get("converged")))}
    es = [bool(r["early_stopping"]) for r in reps if "early_stopping" in r]
    if es:
        out["early_stopping_share"] = float(np.mean(es))
    lim = [r.get("n_estimators") or r.get("epochs_max") for r in reps if (r.get("n_estimators") or r.get("epochs_max"))]
    if lim:
        out["limit"] = int(max(lim))
    return out


def sealed_folds(cfg, ds, k) -> list:
    """봉인 평가의 폴드(실행·미리보기가 같은 함수를 쓴다). 개발 평가와 같은 '규칙'으로, 테스트 시작만 봉인 시작일로 바꾼다.
    - 워크포워드: 봉인 첫날부터 테스트가 시작되는 폴드를 test_len 씩 밀며(실전과 같은 절차) 데이터 끝(설정의 end)까지. 마지막 폴드는 남은 날만큼.
      2026-11-06 전에는 처음부터 만든 폴드 중 테스트 시작이 봉인일 이후인 것만 골라, 봉인 첫 폴드 앞의 최대 test_len−1 일이 평가되지 않았다.
    - 1회 분할: '테스트 시작 전 전부 − 비움'으로 한 번 학습하는 개발 규칙에서 테스트 시작을 봉인 시작일로 바꾼 것 — 봉인 시작 전 전부 − 비움으로
      한 번 학습해 봉인 구간 전체를 예측한다. 검증 시작일(val_start)이 있으면 검증 = [val_start, 봉인 시작 − 비움). 2026-11-06 전에는 1회 분할도
      워크포워드로 봉인 구간을 평가했다(개발 규칙과 달랐음)."""
    v = cfg.validation
    if not v.sealed_from:
        raise ValueError("봉인 평가에는 validation.sealed_from(봉인 시작일)이 필요합니다")
    emb = _embargo(cfg, k)
    calendar_all = cal.trading_calendar(ds.prices)
    sf = seal_cut(cfg, calendar_all, k)  # 첫 봉인 결정일 — 정답·수익에 봉인 시작일 이후 값이 처음 쓰이는 결정일(가격 기준 봉인 시작일보다 h 행 앞)
    if int(calendar_all.searchsorted(pd.Timestamp(v.sealed_from))) >= len(calendar_all):
        raise ValueError(f"봉인 시작일({v.sealed_from}) 이후 데이터가 없습니다 — 데이터 끝({calendar_all[-1].date()})")
    if v.mode == "split":
        train_idx = calendar_all[calendar_all < sf]
        train_idx = train_idx[:-emb] if emb and len(train_idx) > emb else train_idx
        test_idx = calendar_all[calendar_all >= sf]
        vf = pd.Timestamp(v.val_start) if v.val_start else None
        if len(test_idx) == 0:
            raise ValueError(f"봉인 시작일({sf.date()}) 이후 데이터가 없습니다 — 데이터 끝({calendar_all[-1].date()})")
        return [cal.Fold(train_idx, test_idx, val_from=vf)]
    folds = cal.walkforward_folds(calendar_all, train_len=v.train_len, test_len=v.test_len, k=k, embargo=emb,
                                  start=int(calendar_all.searchsorted(sf)), sealed_from=None)
    folds = [f for f in folds if len(f.test) and f.test[0] >= sf]
    if not folds:
        raise ValueError(f"봉인 구간에 폴드가 없습니다 — 봉인 시작 {sf.date()}, 데이터 끝 {calendar_all[-1].date()}, 학습 창 {v.train_len}")
    return folds


def _predict_sealed(cfg, plugin, params, ds, device, sealed_from, k, log: list | None = None, diags: list | None = None):
    """봉인 구간 예측(sealed_folds 의 폴드로 학습→예측)."""
    target_type = "classification" if cfg.target.kind in ("direction", "event") else "regression"
    return _walkforward_predict(cfg, plugin, params, ds, device, sealed_folds(cfg, ds, k), k, target_type, log=log, diags=diags)


def run_experiment(cfg: ExperimentConfig, dv: DataVersion, reg: Registry, *, n_trials_so_far: int = 1, device: str | None = None, sealed: bool = False, trial_sr_var: float | None = None) -> dict:
    """sealed=True 면 봉인 구간(sealed_from 이후)만 평가한다. 봉인 평가는 후보당 1회가 원칙 — 결과에 sealed=True 로 기록된다.
    결과의 periods 에 실제로 쓴 데이터·학습·검증·테스트 기간(첫 폴드·마지막 폴드)을 남긴다 — 논문 표에 그대로 옮길 값."""
    from engine import runlog, sealpolicy
    # 봉인 정책(seal_policy.json): 이 대상에 봉인 시작일이 정해져 있으면 설정이 그것을 지키는지 본다 — 비어 있으면 채우고, 어기면 멈춘다(engine/sealpolicy.py)
    cfg, seal_note = sealpolicy.apply(cfg, sealed=sealed)
    if seal_note and seal_note["status"] == "applied":
        print(f"  [봉인 정책] 이 대상({seal_note['target']})의 봉인 시작일 {seal_note['policy_date']} 을 적용했습니다(설정에는 비어 있었음)", flush=True)
    elif seal_note and seal_note["status"] == "override":
        print(f"  [봉인 정책] 봉인 시작일 {seal_note['policy_date']} 을 일부러 넘깁니다 — 이유: {seal_note.get('reason')}. 이 결과는 봉인 구간을 본 것으로 기록됩니다", flush=True)
    _check_periods(cfg)
    started = runlog.now_utc()
    # 변수 치환·제외 내역(build_dataset 과 같은 규칙·같은 입력이라 같은 결과) — 결과의 run_record 에 남긴다
    var_detail = reg.resolve_detail(cfg.variable_ids) if cfg.variable_ids else None
    ov_notes = reg.remap_overrides(cfg.variable_ids, cfg.variable_transforms)[1] + reg.remap_overrides(cfg.variable_ids, cfg.lag_override)[1]
    fold_log: list = []
    diags: list = []
    with step(log_e, f"실험 {cfg.name or cfg.model} ({cfg.mode})", logging.INFO, 모델=cfg.model, 변수=cfg.variable_ids, 기간={"start": cfg.start, "end": cfg.end, **cfg.validation.model_dump()}):
        res = _run_experiment(cfg, dv, reg, n_trials_so_far=n_trials_so_far, device=device, sealed=sealed, trial_sr_var=trial_sr_var, fold_log=fold_log, diags=diags)
    if cfg.pool_tickers:
        npl = [d["n_pool"] for d in (diags or []) if "n_pool" in d]
        res["pool"] = {"tickers": list(dict.fromkeys(cfg.pool_tickers)), "n_folds": len(npl), "rows_mean": float(np.mean(npl)) if npl else 0.0,
                       "rows_min": int(min(npl)) if npl else 0, "rows_max": int(max(npl)) if npl else 0,
                       "note": "다른 종목의 행은 학습에만 보탰다. 예측·지표·검정은 대상 종목만"}
    res["feature_relevance"] = relevance_summary(diags, res.get("feature_names") or [], cfg.select_top)  # 입력 변수의 연관성 순위(폴드 안에서 잰 값)
    res["model_diag"] = diag_summary(diags)  # 학습 진단(시드 수·조기 종료 지점·상수 예측 폴드). 표 입력·창 입력 모델만
    # 자료 점검표 중 이 실험이 쓴 변수의 줄(2026-11-14). 계산에는 쓰지 않는 기록이다. 봉인 시작일이 있으면 그 앞의 값만으로 검사해
    # 봉인 구간 값이 점검에 섞이지 않게 한다. 점검이 실패해도 실험은 그대로(이유만 남김).
    try:
        from engine.datacheck import check_table
        used, _ = reg.resolve_many(cfg.variable_ids)
        if cfg.mode == "forecast" and cfg.target_variable and cfg.target_variable not in used:
            used = [cfg.target_variable] + list(used)
        tr_over, _ = reg.remap_overrides(cfg.variable_ids, cfg.variable_transforms)
        upto = cfg.validation.sealed_from
        res["data_checks"] = {"upto": upto, "rows": check_table(dv, reg, ids=list(used), transforms=tr_over, upto=upto)}
    except Exception as e:  # noqa: BLE001
        res["data_checks"] = {"upto": cfg.validation.sealed_from, "rows": [], "error": f"{type(e).__name__}: {e}"[:300]}
    md = res["model_diag"]
    # 퇴화 판정을 넓힌다(2026-11-10): 그 전에는 '중요도 합이 0'일 때만 잡아, 트리가 1그루라도 생기면(중요도 > 0) 사실상 상수 예측이어도 표시되지 않았다.
    # 이제 (가) 평가 구간 전체에서 예측이 한 값이거나 (나) 확인할 수 있는 모든 폴드에서 테스트 예측이 변하지 않으면 퇴화로 표시한다.
    pm_ = res.get("prediction_metrics") or {}
    if (pm_.get("pred_sd_ratio") is not None and pm_["pred_sd_ratio"] < 1e-12) or (md and md.get("folds_checked_for_constant") and md.get("constant_folds") == md["folds_checked_for_constant"]):
        if not res.get("degenerate_model"):
            print("  [경고] 예측이 변하지 않음(상수 예측) → 퇴화 모델로 표시", flush=True)
        res["degenerate_model"] = True
    if md:
        tu = md.get("trees_used") or md.get("epoch_used")
        if tu:
            print(f"  [학습 진단] 실제로 쓰인 {'트리' if 'trees_used' in md else '반복'} 수: 중앙값 {tu['median']:.0f} (최소 {tu['min']:.0f} ~ 최대 {tu['max']:.0f}, 상한 {md.get('limit')}) · 2 이하인 비율 {tu['share_le2']:.0%}", flush=True)
        if md.get("constant_folds"):
            print(f"  [학습 진단] 테스트 예측이 폴드 안에서 전혀 변하지 않은 폴드 {md['constant_folds']}/{md['folds_checked_for_constant']}개", flush=True)
        if md.get("multi_start"):
            ms_ = md["multi_start"]
            print(f"  [학습 진단] 여러 출발점 추정: {ms_['folds']}개 폴드 중 {ms_['improved_folds']}개에서 기본 출발점보다 로그우도가 높은 해(최대 +{ms_['max_llf_gain']:.3f}) · 쓰인 출발점 {ms_['start_used']} · 미수렴 {ms_['not_converged_folds']}개", flush=True)
        if md["seeds"].get("pred_sd_across_seeds_mean") is not None:
            print(f"  [학습 진단] 시드 {md['seeds']['n_seeds']}개 평균 · 시드 사이 예측 표준편차(폴드 평균) {md['seeds']['pred_sd_across_seeds_mean']:.6g}", flush=True)
    if sealed:
        # 봉인 수치는 기록(결과 파일·봉인 기록)이 저장된 뒤에만 보여 준다. 그 전에 로그에 찍으면, 저장 단계에서 실패한 실행이 '수치를 봤는데 개봉으로
        # 세지 않는' 구멍이 된다(2026-11-10 독립 검토 F7).
        log_e.info("결과: 폴드 %s · 예측 %s개 · (봉인 평가 — 지표는 기록을 저장한 뒤에 표시)", res.get("n_folds"), res.get("n_predictions"))
    else:
        log_e.info("결과: 폴드 %s · 예측 %s개 · 지표 %s", res.get("n_folds"), res.get("n_predictions"), describe(res.get("prediction_metrics") or {}))
    v = cfg.validation
    res["periods"] = {
        "data": res.get("period"),
        "requested": {"start": cfg.start, "end": cfg.end, "val_start": v.val_start, "test_start": v.test_start or (v.split_date if v.mode == "split" else None),
                      "test_end": v.test_end, "sealed_from": v.sealed_from},
        "embargo": _embargo(cfg, cfg.target.k),
        "validation_rule": ("date" if (v.mode == "split" and v.val_start) else f"val_frac={v.val_frac}"),
        "n_folds_logged": len(fold_log),
        "first_fold": fold_log[0] if fold_log else None,
        "last_fold": fold_log[-1] if fold_log else None,
        "test": [fold_log[0]["test"][0], fold_log[-1]["test"][1]] if fold_log and fold_log[0]["test"] and fold_log[-1]["test"] else None,
        "folds": fold_log,   # 모든 폴드의 학습·검증·테스트 날짜와 개수(2026-11-20 — 그 전 결과는 첫·마지막 폴드만)
    }
    res["seal_policy"] = seal_note  # 봉인 정책 적용 내역(정책이 없는 대상은 None)
    res["data_version"] = dv.id  # 실제로 읽은 버전(예측 모드는 설정에 적힌 값을 그대로 적어, --data-version 으로 돌리면 틀리거나 비었다 — 2026-11-10)
    rr = runlog.build_run_record(cfg=cfg, dv=dv, reg=reg, variables=var_detail, override_notes=ov_notes, started_utc=started, device=res.get("device"))
    rr["finished_utc"] = runlog.now_utc()
    res["run_record"] = rr
    for n in rr["notes"]:
        print(f"  [실행 기록] {n}")
    return res


def _check_periods(cfg: ExperimentConfig) -> None:
    """기간 설정의 순서 검사: 데이터 시작 ≤ 검증 시작 < 테스트 시작 ≤ 테스트 종료 < 봉인 시작 ≤ 데이터 끝."""
    v = cfg.validation
    T = lambda x: pd.Timestamp(x) if x else None
    st, en, vs = T(cfg.start), T(cfg.end), T(v.val_start)
    ts, te, sf = T(v.test_start or (v.split_date if v.mode == "split" else None)), T(v.test_end), T(v.sealed_from)
    errs = []
    if st and en and st >= en:
        errs.append(f"데이터 시작({st.date()})이 종료({en.date()})보다 늦습니다")
    if vs and v.mode != "split":
        errs.append("검증 시작일(val_start)은 1회 분할에서만 씁니다 — 워크포워드는 학습 창마다 뒤쪽 val_frac 를 검증으로 씁니다")
    if vs and ts and vs >= ts:
        errs.append(f"검증 시작({vs.date()})은 테스트 시작({ts.date()})보다 앞이어야 합니다")
    if vs and st and vs <= st:
        errs.append(f"검증 시작({vs.date()})은 데이터 시작({st.date()})보다 뒤여야 합니다")
    if ts and te and te < ts:
        errs.append(f"테스트 종료({te.date()})가 테스트 시작({ts.date()})보다 앞입니다")
    if te and sf and te >= sf:
        errs.append(f"테스트 종료({te.date()})는 봉인 시작({sf.date()}) 전이어야 합니다 — 봉인 구간은 개발 중 보지 않는 최종 확인용")
    if ts and sf and ts >= sf:
        errs.append(f"테스트 시작({ts.date()})이 봉인 시작({sf.date()}) 이후입니다")
    if ts and en and ts > en:
        errs.append(f"테스트 시작({ts.date()})이 데이터 끝({en.date()}) 이후입니다")
    if errs:
        raise ValueError("기간 설정 오류: " + " / ".join(errs))



def fill_defaults(cfg: ExperimentConfig, reg: Registry) -> ExperimentConfig:
    """실행 때 채워지는 값을 채운 설정. 결과 파일의 config·config_hash 는 이 설정의 것이다.
    · 창 길이 seq_len 기본값(주기별, 모델이 선언하면 그것)
    · 예측 연구 모드의 기술지표 선택 목록을 '대상 변수의 실제 주기'로 다시 정규화 — 설정의 frequency 칸은 예측 연구 모드에서 쓰이지 않는데 정규화가
      그 칸을 봐서, 같은 선택이 frequency 를 무엇으로 적었느냐에 따라 다른 해시가 됐다(2026-11-10 독립 검토 F18).
    데이터를 읽지 않으므로 실행 전에 '결과와 같은 해시'를 알 수 있다 — 봉인 평가의 예약 줄이 이 해시를 쓴다(F7:
    그 전에는 예약 줄에 채우기 전 해시를 적어, 중간에 죽은 실행이 같은 설정의 개봉 횟수에 잡히지 않았다)."""
    upd: dict = {}
    freq = cfg.frequency
    if cfg.mode == "forecast" and cfg.target_variable:
        try:
            freq = reg.get(cfg.target_variable).freq
        except Exception:  # noqa: BLE001 — 없는 대상 변수는 뒤에서 제 오류로 드러난다
            pass
        if cfg.technical_features is not None:
            nt = _norm_technical(cfg.technical_features, cfg.use_technical, freq, False)
            if nt != cfg.technical_features:
                upd["technical_features"] = nt
    if cfg.seq_len is None:
        plugin = load_plugin(cfg.model)
        dflt = {"D": 20, "W": 13, "M": 12, **(getattr(plugin, "SEQ_LEN_DEFAULT", None) or {})}  # 모델이 원본 기본을 선언하면 그것(BIVA 월간 6)
        upd["seq_len"] = dflt.get(freq, 20)
    return cfg.model_copy(update=upd) if upd else cfg


def prepare(cfg: ExperimentConfig, dv: DataVersion, reg: Registry, *, device: str = "cpu"):
    """실행 준비(실험 실행과 계산 따라가기가 같이 쓴다): 시드 고정 → 플러그인·타겟 종류 확인 → 주기별 기본값을 채운 실제 파라미터 →
    창 길이 기본값 → 데이터셋. 반환 (cfg, plugin, params, target_type, ds)."""
    _seed(cfg.seed)
    plugin = load_plugin(cfg.model)
    target_type = "classification" if cfg.target.kind in ("direction", "event") else "regression"
    if plugin.INPUT_SHAPE == "levels" and not cfg.variable_ids:
        raise ValueError(f"{plugin.NAME} 은(는) 입력 변수를 1개 이상 골라야 합니다 — 대상과 변수들의 수준값이 장기적으로 함께 움직이는 관계(공적분)를 쓰는 모델이라 "
                         "대상 하나만으로는 정의되지 않습니다. 기술지표(자체 지표)는 VECM 에 쓰이지 않습니다. 예: 월간 F11 이면 F7(CPI)·B1(금리) 등 2~5개")
    if plugin.INPUT_SHAPE != "rule" and target_type not in plugin.TARGET_TYPES:
        raise ValueError(f"{plugin.NAME} 은(는) {target_type} 타겟을 지원하지 않습니다")
    freq_for_defaults = (reg.get(cfg.target_variable).freq if cfg.mode == "forecast" and cfg.target_variable else cfg.frequency)
    params = resolve_params(plugin.PARAMS, cfg.model_params, freq_for_defaults)
    cfg = fill_defaults(cfg, reg)  # 창 길이 기본값
    log_e.info("모델 %s(%s) · 장치 %s · 파라미터(받은 값 %s → 기본값 채운 실제 값) %s · 창 길이 %s", plugin.NAME, plugin.INPUT_SHAPE, device,
               describe(cfg.model_params), describe(params), cfg.seq_len if plugin.INPUT_SHAPE == "window" else "-")
    with step(log_d, "데이터셋 만들기(읽기→변환→발표시차 정렬→타겟)"):
        if cfg.select_top and plugin.INPUT_SHAPE not in ("table", "window"):   # 2026-11-13 독립 검토 12 — 무시되면서 해시만 바꾸던 것
            raise ValueError("입력 선별(select_top)은 입력 변수로 학습하는 모델(릿지·LightGBM·LSTM·GRU·BIVA)에서만 쓸 수 있습니다 — "
                             f"{plugin.NAME} 은(는) 입력 변수를 고르지 않으므로 선별이 아무 일도 하지 않습니다. 선별 칸을 비우세요")
        ds = build_dataset(cfg, dv, reg)
        if cfg.pool_tickers:
            ds.pool = _build_pool(cfg, plugin, dv, reg, ds)
    return cfg, plugin, params, target_type, ds


def _build_pool(cfg: ExperimentConfig, plugin, dv: DataVersion, reg: Registry, ds: "Dataset") -> list:
    """묶어 학습할 종목마다 같은 설정(입력·타겟·주기)으로 데이터셋을 만든다. 기술지표는 그 종목의 가격으로, 변수(시장·거시)는 날짜로 맞춘 같은 값."""
    if cfg.mode != "trading":
        raise ValueError("종목 묶어 학습(pool_tickers)은 매매 모드에서만 쓸 수 있습니다")
    if plugin.INPUT_SHAPE != "table":
        raise ValueError("종목 묶어 학습(pool_tickers)은 표 입력 모델(릿지·LightGBM)에서만 쓸 수 있습니다 — "
                         "창 입력 모델(LSTM·GRU·BIVA)은 종목마다 창을 따로 만들어야 해 아직 지원하지 않고, 규칙·ARIMA 류는 다른 종목으로 학습하지 않습니다")
    out = []
    for t in dict.fromkeys(cfg.pool_tickers):
        if t == cfg.ticker:
            raise ValueError(f"종목 묶어 학습(pool_tickers)에 대상 종목 {t} 자신을 넣을 수 없습니다")
        try:
            pds = build_dataset(cfg.model_copy(update={"ticker": t, "pool_tickers": []}), dv, reg)
        except FileNotFoundError:
            raise ValueError(f"종목 묶어 학습(pool_tickers): 데이터 버전에 {t} 의 가격 자료가 없습니다 — 먼저 그 종목을 수집해야 합니다") from None
        if pds.feature_names != ds.feature_names:
            raise ValueError(f"종목 묶어 학습(pool_tickers): {t} 의 입력 열이 대상 종목과 다릅니다")
        out.append((t, pds))
    return out

def _run_experiment(cfg: ExperimentConfig, dv: DataVersion, reg: Registry, *, n_trials_so_far: int = 1, device: str | None = None, sealed: bool = False, trial_sr_var: float | None = None, fold_log: list | None = None, diags: list | None = None) -> dict:
    t0 = time.time()
    device = device or default_device()
    if cfg.target.kind == "volatility":
        if sealed:
            raise ValueError("변동성 타겟은 아직 봉인 평가를 할 수 없습니다 — 통과 기준(무엇을 이기면 통과인지)을 사전 분석 계획서에서 먼저 정해야 합니다")
        if cfg.mode == "forecast":
            raise ValueError("변동성 타겟은 매매 모드에서만 쓸 수 있습니다(예측 연구 모드의 대상은 가격이 아니라 지표 값입니다)")
        if load_plugin(cfg.model).INPUT_SHAPE not in ("table", "window"):
            raise ValueError("변동성 타겟은 입력 변수로 학습하는 모델(릿지·LightGBM·LSTM·GRU·BIVA)에서만 쓸 수 있습니다 — "
                             "규칙 모델과 ARIMA·SARIMA·VECM 은 수익률(가격)을 맞히도록 만들어져 있습니다")
    cfg, plugin, params, target_type, ds = prepare(cfg, dv, reg, device=device)
    k = cfg.target.k
    if cfg.trade.sizing == "kelly":
        if cfg.mode == "forecast":
            raise ValueError("켈리 투자 비율은 매매 모드에서만 쓸 수 있습니다")
        if plugin.INPUT_SHAPE == "rule":
            raise ValueError("켈리 투자 비율은 예측을 내는 모델에서만 쓸 수 있습니다 — 규칙 모델은 예측 없이 바로 비율을 냅니다")
        if cfg.target.kind == "volatility":
            raise ValueError("켈리 투자 비율은 변동성 타겟과 함께 쓸 수 없습니다 — 변동성 타겟은 이미 비율(기준 ÷ 예측 변동성)을 정합니다")

    pred = pd.Series(np.nan, index=ds.prices.index, name="prediction")
    importances: list[np.ndarray] = []
    n_folds = 0
    bench = None  # 학습 창 타겟 평균(기준 예측). 학습 모델만

    if cfg.mode == "forecast" and plugin.INPUT_SHAPE == "rule":
        raise ValueError("예측 연구 모드에는 규칙(매매) 모델을 쓸 수 없습니다")
    kelly_info = None  # 켈리 투자 비율을 쓴 경우의 정보(2026-11-15)
    rebal = None  # 재조정 날(섀넌 재조정처럼 '흘러가는 비율'을 쓰는 규칙만, 2026-11-15). None = 예전 회계
    if plugin.INPUT_SHAPE == "rule":
        model: RuleModel = plugin.Model(params)
        if getattr(model, "REBALANCE", False):
            rebal = pd.Series(model.rebalance_days(ds.prices), index=ds.prices.index)
        pos = pd.Series(model.positions(ds.prices, ds.X) if getattr(model, "USES_VARIABLES", False) else model.positions(ds.prices), index=ds.prices.index)
        if not sealed:
            # 규칙 모델은 학습이 없어 첫날부터 신호를 낼 수 있지만, 학습 모델과 같은 날짜로 성적을 매겨야 비교가 공정하다.
            # 같은 검증 설정이면 학습 모델이 평가받았을 구간(첫 폴드 테스트 시작 ~ 마지막 폴드 테스트 끝)만 남긴다.
            lo, hi = _eval_window(cfg, ds, k)
            if lo is not None:
                pos = pos.where((pos.index >= lo) & (pos.index <= hi))
                log_e.info("규칙 모델 평가 구간을 학습 모델과 맞춤: %s ~ %s", lo.date(), hi.date())
        signal = pos
        pred[:] = pos.values
    elif sealed:
        # 봉인 평가: 개발 구간은 다시 계산하지 않는다(결과에 쓰지 않으므로). 봉인 폴드만 학습→예측
        fit = None
        pred, n_folds, importances = _predict_sealed(cfg, plugin, params, ds, device, cfg.validation.sealed_from, k, log=fold_log, diags=diags)
        bench = train_mean_benchmark(ds, sealed_folds(cfg, ds, k))
        signal = _signal_from_prediction(cfg, pred, bench, target_type)
        signal, kelly_info = _apply_sizing(cfg, ds, pred, signal, k)
    else:
        folds, is_split = make_folds(cfg, ds, k)
        bench = train_mean_benchmark(ds, folds)
        fit = None
        if is_split:
            pred, n_folds, importances, fit = _walkforward_predict(cfg, plugin, params, ds, device, folds, k, target_type, want_fit=True, log=fold_log, diags=diags)
        else:
            pred, n_folds, importances = _walkforward_predict(cfg, plugin, params, ds, device, folds, k, target_type, log=fold_log, diags=diags)
        # 신호 변환
        signal = _signal_from_prediction(cfg, pred, bench, target_type)
        signal, kelly_info = _apply_sizing(cfg, ds, pred, signal, k)

    cost = CostModel(fee_rate=cfg.trade.fee_rate, slippage_bp=cfg.trade.slippage_bp)
    ppy = {"D": 252, "W": 52, "M": 12}[cfg.frequency]
    # 아래의 sealed_from 은 '첫 봉인 결정일'이다(설정의 sealed_from 은 값 기준 — seal_cut 설명 참고)
    _cal_all = cal.trading_calendar(ds.prices)
    sealed_from = seal_cut(cfg, _cal_all, k)
    seal_info = None if sealed_from is None else {
        "sealed_from": cfg.validation.sealed_from, "first_sealed_decision": str(sealed_from.date()), "horizon_rows": seal_horizon(cfg, k),
        "rule": "개발 = 정답·수익의 마지막 값 날짜 < sealed_from 인 결정일, 봉인 = 그 밖(빠짐·겹침 없음)"}
    if cfg.mode == "forecast":
        res = _finish_forecast(cfg, plugin, params, ds, device, pred, fit if cfg.validation.mode == "split" else None,
                               importances, n_folds, k, target_type, sealed, sealed_from, t0, fold_log=fold_log, bench=bench)
        res["seal_boundary"] = seal_info
        return res
    if sealed:
        if sealed_from is None:
            raise ValueError("봉인 평가에는 validation.sealed_from 이 필요합니다")
        if plugin.INPUT_SHAPE == "rule":  # 학습 없는 규칙: 봉인 구간의 포지션만 남긴다(학습 모델은 위에서 봉인 폴드로 이미 예측)
            signal = signal.where(signal.index >= sealed_from)
            pred = pred.where(pred.index >= sealed_from)
    elif sealed_from is not None:
        # 개발 구간 평가: 봉인 이후는 신호를 지운다 (규칙 모델도 봉인 구간을 쓰지 못하게)
        signal = signal.where(signal.index < sealed_from)
        pred = pred.where(pred.index < sealed_from)

    daily_full, _ = run_backtest(ds.prices, signal, cost=cost, periods_per_year=ppy, rebalance=rebal)
    daily_full["prediction"] = pred
    daily_full["target"] = ds.y
    if bench is not None:
        daily_full["bench_mean"] = bench.where(pred.notna())  # 기준 예측(학습 창 타겟 평균). 예측이 있는 결정일만
    if plugin.INPUT_SHAPE != "rule" and cfg.validation.mode == "split" and fit is not None:
        daily_full["fit"] = fit  # 학습 구간 적합값(차트용). 성과·검정에는 쓰지 않는다
    # 성과는 '신호가 있는 구간'(첫 신호~마지막 신호)에서만 계산한다. 그 밖은 B&H 와 비교할 수 없다.
    has = signal.dropna()
    if len(has) == 0:
        raise RuntimeError("신호가 하나도 없습니다")
    daily = daily_full.loc[has.index.min(): has.index.max()].copy()
    perf = performance(daily["net_return"], daily["hold_return"], periods_per_year=ppy)
    # 거래량: 예전 회계는 포지션 변화량, 흘러가는 비율 회계는 실제로 사고판 양(비율이 흘러간 것은 거래가 아님 — 2026-11-15)
    turnover = daily["position"].diff().abs().fillna(daily["position"].abs()) if rebal is None else daily["turnover"]
    perf.update(n_trades=int((turnover > 0).sum()), turnover=float(turnover.sum()), exposure=float(daily["position"].mean()), total_cost=float(daily["cost"].sum()))

    # 비교 기준 '섀넌 50:50'(2026-11-15): 같은 평가 구간·같은 비용으로 ETF 절반·현금 절반, 한 달마다 다시 맞춤.
    # B&H(늘 100%)와 짝 — '판단 없이 절반만 투자해도 이만큼'. 모델 성적에는 영향 없음(비교용 숫자만 추가).
    sh_ret, sh_met = shannon_benchmark(ds.prices, daily.index.min(), daily.index.max(), cost=cost, periods_per_year=ppy)
    daily_full["shannon_return"] = sh_ret          # 평가 구간만 값(그 밖 NaN) — 결과 *_daily.parquet 에 남아 손으로 다시 계산 가능
    perf["shannon"] = sh_met

    # 예측 성능 (예측이 있는 구간)
    ev = daily[["prediction", "target"]].dropna()
    pred_metrics: dict[str, float] = {}
    if plugin.INPUT_SHAPE != "rule" and len(ev) > 5:
        e = ev["prediction"] - ev["target"]
        if cfg.target.kind == "volatility":
            # 변동성은 늘 양수라 '방향'·'0 예측과의 비교'는 뜻이 없다. 기준은 둘: 학습 창 평균(아래 mean_benchmark_metrics)과
            # '최근 k 기간의 변동성이 그대로 이어진다'(naive — 변동성은 이어지는 성질이 강해 이것을 이기기가 어렵다).
            nv = cal.naive_volatility(ds.prices, k).reindex(ev.index)
            okn = nv.notna()
            dm, dm_p = diebold_mariano(e[okn].values, (nv[okn] - ev["target"][okn]).values, h=k)
            sd_t = float(ev["target"].std())
            pred_metrics = {"rmse": float(np.sqrt((e**2).mean())), "mae": float(e.abs().mean()),
                            "rmse_naive": float(np.sqrt(((nv[okn] - ev["target"][okn]) ** 2).mean())), "dm_vs_naive": dm, "dm_vs_naive_p": dm_p,
                            "corr": float(np.corrcoef(ev["prediction"], ev["target"])[0, 1]) if ev["prediction"].std() > 0 and sd_t > 0 else None,
                            "pred_sd_ratio": float(ev["prediction"].std() / sd_t) if sd_t > 0 else None}
        elif target_type == "regression":
            pred_metrics = {
                "rmse": float(np.sqrt((e**2).mean())),
                "mae": float(e.abs().mean()),
                "dir_acc": float(((ev["prediction"] > 0) == (ev["target"] > 0)).mean()),
            }
            # naive(0 수익률 예측) 대비 DM 검정
            dm, dm_p = diebold_mariano(e.values, (0 - ev["target"]).values, h=k)
            pred_metrics.update(dm_vs_zero=dm, dm_vs_zero_p=dm_p)
            # 0 예측은 모델에 내포된다 → '변수에 예측 정보가 있는가'는 Clark–West 로 본다(DM 은 '이 방법이 0예측보다 정확한가', 2026-11-10)
            cw, cw_p = clark_west(ev["target"].values, ev["prediction"].values, 0.0, h=k)
            pred_metrics.update(cw_vs_zero=cw, cw_vs_zero_p=cw_p)
            # 0 예측의 오차와 예측의 흩어짐(2026-11-10): rmse 만으로는 '모델이 거의 상수를 내는지' 알 수 없다.
            # pred_sd_ratio = 예측의 표준편차 ÷ 정답의 표준편차. 0 에 가까우면 예측이 날마다 거의 같다(조기 종료로 트리 1~2그루 등).
            sd_t = float(ev["target"].std())
            pred_metrics.update(rmse_zero=float(np.sqrt((ev["target"] ** 2).mean())), pred_sd_ratio=float(ev["prediction"].std() / sd_t) if sd_t > 0 else None)
        else:
            from sklearn.metrics import roc_auc_score

            yb = ev["target"].values
            pred_metrics = {"dir_acc": float(((ev["prediction"] > 0.5) == (yb > 0.5)).mean()),
                            "brier": float(((ev["prediction"].values - yb) ** 2).mean())}
            if len(np.unique(yb)) == 2:
                pred_metrics["auc"] = float(roc_auc_score(yb, ev["prediction"].values))
            # 무정보(0.5) 대비 DM — 회귀의 dm_vs_zero 와 같은 역할·같은 키 이름
            dm, dm_p = diebold_mariano_brier(ev["prediction"].values, yb, h=k)
            pred_metrics.update(dm_vs_zero=dm, dm_vs_zero_p=dm_p, dm_baseline="brier_vs_0.5")
            cw, cw_p = clark_west(yb, ev["prediction"].values, 0.5, h=k)
            pred_metrics.update(cw_vs_zero=cw, cw_vs_zero_p=cw_p)
        if "bench_mean" in daily:
            pred_metrics.update(mean_benchmark_metrics(ev["target"].values.astype(float), ev["prediction"].values.astype(float),
                                                       daily["bench_mean"].reindex(ev.index).values.astype(float), k))

    # 신뢰도
    valid = daily.dropna(subset=["net_return", "hold_return"])
    perf["bootstrap_p_excess"] = bootstrap_excess_pvalue(valid["net_return"], valid["hold_return"], block=max(5, 2 * k))
    sk, ku = return_moments(valid["net_return"])
    perf["deflated_sharpe_p"] = deflated_sharpe_pvalue(perf.get("sharpe", float("nan")), perf.get("n_periods", 0), n_trials_so_far,
                                                       skew=sk, kurt=ku, periods_per_year=ppy, trial_sr_var=trial_sr_var)
    perf["return_skew"], perf["return_kurt"] = sk, ku
    perf["trial_sr_var"] = trial_sr_var
    perf["n_trials_so_far"] = int(n_trials_so_far)

    fi_mean = np.mean(importances, axis=0).tolist() if importances else None
    # 퇴화 검사: 중요도가 전부 0 이면 모델이 입력을 전혀 쓰지 않은 것(= 폴드별 상수 예측). 그 dir_acc 는 "항상 롱"의 적중률일 뿐이다.
    degenerate = bool(fi_mean is not None and len(fi_mean) > 0 and float(np.nansum(np.abs(fi_mean))) == 0.0)
    if degenerate:
        print("  [경고] 모델이 어떤 입력도 쓰지 않음(중요도 전부 0) → 상수 예측. 파라미터(min_child_samples 등)가 학습 표본 수에 비해 큼")
    return {
        "config": json.loads(cfg.model_dump_json()),
        "config_hash": cfg.config_hash(),
        "model": {"file": cfg.model, "name": plugin.NAME, "input_shape": plugin.INPUT_SHAPE, "params": params},
        "data_version": dv.id,
        "code_version": _code_version(), "engine_rev": ENGINE_REV,
        "n_folds": n_folds,
        "n_features": len(ds.feature_names),
        "feature_names": ds.feature_names,
        "feature_importance": fi_mean,
        "trade_rule_used": ({"kind": "volatility_target", "rule": "비중 = 학습 창 평균 변동성 ÷ 예측 변동성 (0~1 로 자름)"} if cfg.target.kind == "volatility"
                            else {"kind": "kelly", "rule": "비율 = 분수 × μ_b ÷ σ²_b (결정일에 결과가 나온 과거 예측 중 같은 자신감 묶음의 실제 수익), 0~1 로 자름", **(kelly_info or {})} if kelly_info is not None
                            else {"kind": "rule" if plugin.INPUT_SHAPE == "rule" else "threshold", "threshold": cfg.trade.threshold}),
        # 자신감 확인표(2026-11-15): 예측 크기 순 5묶음의 실제 결과 — 진단용(매매 결정에는 안 씀). 예측을 내는 모델만
        "confidence_table": (None if plugin.INPUT_SHAPE == "rule" or cfg.target.kind == "volatility" else
                             _confidence(pred.loc[daily.index.min(): daily.index.max()], ds, k)),
        "degenerate_model": degenerate,
        "sealed_evaluation": bool(sealed),
        "evaluation_period": [str(daily.index[0].date()), str(daily.index[-1].date())],
        "leak_warning": ds.leak_warning,
        "lag_days_used": ds.lag_days_used,
        "lag_unconfirmed": ds.lag_unconfirmed,
        "transforms_used": ds.transforms_used, "transform_basis": ds.transform_basis,
        "prediction_metrics": pred_metrics,
        "performance": perf,
        "period": [str(daily.index[0].date()), str(daily.index[-1].date())],
        "prediction_period": [str(ev.index[0].date()), str(ev.index[-1].date())] if len(ev) else None,
        "n_predictions": int(len(ev)),
        "variable_first_date": {c: (str(ds.X[c].first_valid_index().date()) if ds.X[c].first_valid_index() is not None else None) for c in ds.feature_names if not c.startswith("A_")},
        "device": device,
        "seconds": round(time.time() - t0, 1),
        "seal_boundary": seal_info,
        "_daily": daily_full,  # 전체 기간(워밍업·봉인 포함). 성과는 evaluation_period 로 잘라 계산했다
    }


def forecast_level_metrics(daily: pd.DataFrame, kind: str) -> dict:
    """k기간 뒤 '값'(수준값)으로 환산한 예측 오차 — 예측력의 1차 지표(논문 표기용).
    모델 v[t]·g(ŷ), naive v[t]·g(y[t−k]), 변화없음 v[t] 를 실제 v[t+k] 와 비교. g = 1+x(return) / e^x(logreturn).
    평가 행 = 모델 예측·타겟·naive 가 모두 있는 행(봉인 구간은 이미 제외됨). 분류 타겟이면 빈 dict."""
    if kind not in ("return", "logreturn"):
        return {}
    g = (lambda x: 1 + x) if kind == "return" else np.exp
    d = daily.dropna(subset=["prediction", "target", "naive", "actual_future_level"])
    if d.empty:
        return {}
    act = d["actual_future_level"]
    out, ape = {}, {}
    for tag, lv in (("", d["level"] * g(d["prediction"])), ("_naive", d["level"] * g(d["naive"])), ("_zero", d["level"])):
        e = lv - act
        ape[tag] = (e.abs() / act.abs()).values
        out[f"level_rmse{tag}"] = float(np.sqrt((e**2).mean()))
        out[f"level_mape{tag}"] = float(ape[tag].mean())
    # 화면에 보이는 MAPE 와 같은 눈금의 개선율·검정(참고). 판정은 사전 등록대로 변화율 제곱오차 DM(dm_vs_naive).
    out["skill_mape_vs_naive"] = float(1 - out["level_mape"] / out["level_mape_naive"]) if out["level_mape_naive"] > 0 else None
    k = int(daily.attrs.get("k", 1)) if hasattr(daily, "attrs") else 1
    dm, p = diebold_mariano(ape[""], ape["_naive"], h=k, power=1)
    out.update(dm_ape_vs_naive=dm, dm_ape_vs_naive_p=p)
    return out


def forecast_metrics(daily_full: pd.DataFrame, kind: str, k: int, target_type: str) -> tuple[dict, pd.DataFrame]:
    """예측 연구 지표 — 저장된 daily(예측·타겟·naive·수준값)만으로 계산한다. 실험 마무리와 앙상블이 같은 함수를 쓴다."""
    ev = daily_full[["prediction", "target", "naive"]].dropna()
    if len(ev) < 10:
        raise RuntimeError("예측이 10개 미만입니다 — 학습 창·변수 시작일을 확인하세요")
    e = ev["prediction"] - ev["target"]
    e0 = 0 - ev["target"]
    en = ev["naive"] - ev["target"]
    pm: dict[str, float] = {}
    if target_type == "regression":
        pm = {
            "rmse": float(np.sqrt((e**2).mean())), "mae": float(e.abs().mean()),
            "rmse_zero": float(np.sqrt((e0**2).mean())), "rmse_naive": float(np.sqrt((en**2).mean())),
            "dir_acc": float(((ev["prediction"] > 0) == (ev["target"] > 0)).mean()),
            "dir_acc_naive": float(((ev["naive"] > 0) == (ev["target"] > 0)).mean()),
        }
        dm, p = diebold_mariano(e.values, e0.values, h=k); pm.update(dm_vs_zero=dm, dm_vs_zero_p=p)
        dm, p = diebold_mariano(e.values, en.values, h=k); pm.update(dm_vs_naive=dm, dm_vs_naive_p=p)
        cw, p = clark_west(ev["target"].values, ev["prediction"].values, 0.0, h=k); pm.update(cw_vs_zero=cw, cw_vs_zero_p=p)
        sd_t = float(ev["target"].std())
        pm["pred_sd_ratio"] = float(ev["prediction"].std() / sd_t) if sd_t > 0 else None  # 예측의 흩어짐 ÷ 정답의 흩어짐(0 에 가까우면 거의 상수 예측)
        daily_full.attrs["k"] = k
        pm.update(forecast_level_metrics(daily_full, kind))
        pm["skill_vs_naive"] = float(1 - pm["rmse"] / pm["rmse_naive"]) if pm.get("rmse_naive") else None
    else:
        from sklearn.metrics import roc_auc_score

        yb = ev["target"].values
        pm = {"dir_acc": float(((ev["prediction"] > 0.5) == (yb > 0.5)).mean()), "brier": float(((ev["prediction"].values - yb) ** 2).mean())}
        if len(np.unique(yb)) == 2:
            pm["auc"] = float(roc_auc_score(yb, ev["prediction"].values))
        dm, p = diebold_mariano_brier(ev["prediction"].values, yb, h=k); pm.update(dm_vs_zero=dm, dm_vs_zero_p=p, dm_baseline="brier_vs_0.5")
        cw, p = clark_west(yb, ev["prediction"].values, 0.5, h=k); pm.update(cw_vs_zero=cw, cw_vs_zero_p=p)
        nb = ev["naive"].clip(0, 1).values
        dm, p = diebold_mariano(ev["prediction"].values - yb, nb - yb, h=k); pm.update(dm_vs_naive=dm, dm_vs_naive_p=p)
    if "bench_mean" in daily_full:
        pm.update(mean_benchmark_metrics(ev["target"].values.astype(float), ev["prediction"].values.astype(float),
                                         daily_full["bench_mean"].reindex(ev.index).values.astype(float), k))
    return pm, ev


def _finish_forecast(cfg, plugin, params, ds, device, pred, fit, importances, n_folds, k, target_type, sealed, sealed_from, t0, fold_log: list | None = None, bench=None) -> dict:
    """예측 연구 모드 마무리: 체결·비용 없음. 오차 지표 + 0예측·naive(지속) 대비 DM. 봉인 처리는 매매 모드와 같은 규칙."""
    if sealed:
        if sealed_from is None:
            raise ValueError("봉인 평가에는 validation.sealed_from 이 필요합니다")
        pred = pred.where(pred.index >= sealed_from)  # 위(_run_experiment)에서 봉인 폴드로 이미 예측했다
    elif sealed_from is not None:
        pred = pred.where(pred.index < sealed_from)
    y = ds.y
    naive = cal.naive_forecast(y, k)
    level = ds.prices["close"]
    # 예측 수익률 → k기간 뒤 수준값: return 이면 v(1+ŷ), logreturn 이면 v·exp(ŷ). 분류(방향)는 수준값이 정의되지 않음.
    kind = cfg.target.kind
    if kind == "return":
        pfl = level * (1 + pred)
    elif kind == "logreturn":
        pfl = level * np.exp(pred)
    else:
        pfl = pd.Series(np.nan, index=level.index)
    daily_full = pd.DataFrame({"level": level, "target": y, "prediction": pred, "naive": naive,
                               "actual_future_level": level.shift(-k), "pred_future_level": pfl.where(pred.notna())})
    if bench is not None:
        daily_full["bench_mean"] = bench.where(pred.notna())  # 기준 예측(학습 창 타겟 평균). 예측이 있는 결정일만
    if fit is not None:
        daily_full["fit"] = fit
    pm, ev = forecast_metrics(daily_full, cfg.target.kind, k, target_type)
    fi_mean = np.mean(importances, axis=0).tolist() if importances else None
    degenerate = bool(fi_mean is not None and len(fi_mean) > 0 and float(np.nansum(np.abs(fi_mean))) == 0.0)
    if degenerate:
        print("  [경고] 모델이 어떤 입력도 쓰지 않음(중요도 전부 0) → 상수 예측")
    return {
        "config": json.loads(cfg.model_dump_json()), "config_hash": cfg.config_hash(),
        "model": {"file": cfg.model, "name": plugin.NAME, "input_shape": plugin.INPUT_SHAPE, "params": params},
        "data_version": cfg.data_version, "code_version": _code_version(), "engine_rev": ENGINE_REV, "n_folds": n_folds, "n_features": len(ds.feature_names),
        "feature_names": ds.feature_names, "feature_importance": fi_mean, "degenerate_model": degenerate,
        "sealed_evaluation": bool(sealed), "evaluation_period": [str(ev.index[0].date()), str(ev.index[-1].date())],
        "leak_warning": ds.leak_warning, "lag_days_used": ds.lag_days_used, "lag_unconfirmed": ds.lag_unconfirmed, "transforms_used": ds.transforms_used, "transform_basis": ds.transform_basis,
        "prediction_metrics": pm, "performance": {}, "period": [str(daily_full.index[0].date()), str(daily_full.index[-1].date())],
        "prediction_period": [str(ev.index[0].date()), str(ev.index[-1].date())], "n_predictions": int(len(ev)),
        "variable_first_date": {c: (str(ds.X[c].first_valid_index().date()) if ds.X[c].first_valid_index() is not None else None) for c in ds.feature_names if not c.startswith("A_")},
        "device": device, "seconds": round(time.time() - t0, 1), "_daily": daily_full,
    }


def trial_sharpe_variance(results_dir: str | Path, periods_per_year: int) -> float | None:
    """지금까지 저장된 실험들의 기간 샤프 분산 (Deflated Sharpe 의 V). 같은 주기(periods_per_year)의 결과만. 2개 미만이면 None.
    results_dir 는 모드별 폴더(results/trading)여야 한다 — 2026-11-04 이주 뒤 cli 가 results/ 바로 아래를 넘겨 V 가 늘 None(근사 1/(n−1))이었다(2026-11-06 수정).
    봉인 평가 결과는 '시도'가 아니므로 뺀다. 계산 방식이 지금과 다른 옛 결과(engine_rev 불일치)도 뺀다.
    화면에서 지운 결과(history/ 의 …_deleted)는 넣는다 — 지워도 '해 본 시도'라는 사실은 남고, 빼면 결과를 지워서 V 를 줄일 수 있었다(독립 검토 F21).
    같은 설정을 다시 돌려 밀려난 옛 판(history/ 의 나머지)은 넣지 않는다(같은 시도를 두 번 세지 않는다)."""
    srs = []
    d_ = Path(results_dir)
    files = list(d_.glob("*.json")) + (list((d_ / "history").glob("*_deleted*.json")) if (d_ / "history").exists() else [])
    for p in files:
        if p.name in ("experiment_counter.json", "sealed_log.json") or p.name.startswith("ablation_"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            ppy = {"D": 252, "W": 52, "M": 12}.get(r["config"]["frequency"])
            sh = r["performance"].get("sharpe")
            if r.get("sealed_evaluation"):
                continue
            if r.get("engine_rev") != ENGINE_REV:
                continue  # 옛 계산 방식(평가 구간·눈금이 다름)의 샤프는 지금 결과와 같은 분포가 아니다 — 섞지 않는다(2026-11-10)
            if ppy == periods_per_year and sh is not None and np.isfinite(sh):
                srs.append(sh / np.sqrt(ppy))
        except Exception:  # noqa: BLE001
            continue
    return float(np.var(srs, ddof=1)) if len(srs) >= 2 else None


def results_dir(root: str | Path, mode: str) -> Path:
    """모드별 결과 저장소: results/trading, results/forecast. 카운터·봉인 기록도 각각.
    2026-11-04 이전 결과(results/ 바로 아래)는 처음 호출 때 results/trading 으로 옮긴다(1회 이주)."""
    root = Path(root)
    d = root / ("forecast" if mode == "forecast" else "trading")
    d.mkdir(parents=True, exist_ok=True)
    # 카운터·봉인 기록은 옮기지 않는다: 모드 폴더의 것이 지워졌을 때 낡은 루트 파일이 그 자리를 차지하던 문제(2026-11-10 점검 S2)
    legacy = [p for p in root.iterdir() if p.is_file() and p.suffix in (".json", ".parquet", ".csv", ".md")
              and p.name not in ("experiment_counter.json", "sealed_log.json")]
    moved = 0
    if legacy:
        tdir = root / "trading"; tdir.mkdir(exist_ok=True)
        for p in legacy:
            target = tdir / p.name
            if not target.exists():
                p.rename(target); moved += 1
    if moved:
        print(f"  [이주] 옛 결과 {moved}개 → results/trading/")
    return d


def save_result(res: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{res['config']['name'] or res['model']['file']}_{res['config_hash']}"
    if res.get("sealed_evaluation"):  # 봉인 결과는 같은 설정의 개발 결과와 이름이 같아 덮어쓰지 않도록 따로(같은 설정을 다시 열면 번호)
        stem += "_sealed"
        n = 2
        while (out / f"{stem}.json").exists():
            stem = f"{stem.rsplit('_sealed', 1)[0]}_sealed{n}"; n += 1
    from engine import runlog
    # 저장은 한 묶음으로: ① 새 일별 자료를 임시 이름에 쓰고 ② 같은 이름의 옛 결과를 history/ 로 옮기고 ③ 새 json 을 임시 이름에 쓴 뒤 ④ 둘을 제 이름으로 바꾼다.
    # 어느 단계에서 실패해도 옛 결과는 제자리나 history/ 에 온전히 남는다(2026-11-10 — 전에는 중간 실패 시 옛 일별 자료를 덮을 수 있었다).
    final_j, final_p = out / f"{stem}.json", out / f"{stem}_daily.parquet"
    tmp_j, tmp_p = out / f".{stem}.json.tmp", out / f".{stem}_daily.parquet.tmp"
    daily = res.pop("_daily")
    archived: list[str] = []
    try:
        # 새 결과 두 파일을 임시 이름으로 '먼저 다' 만든다 — json 으로 못 바꾸는 값이나 디스크 오류는 옛 결과를 옮기기 전에 드러난다.
        # (2026-11-10 독립 검토 F17: 전에는 옛 결과를 옮긴 뒤에 새 json 을 써서, 거기서 실패하면 제 이름에 결과가 없었다.)
        daily.to_parquet(tmp_p)
        tmp_j.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        archived = runlog.archive_existing(out, stem)
        try:
            if archived:
                res.setdefault("run_record", {})["replaced_previous"] = archived
                print(f"  [실행 기록] 같은 이름의 옛 결과를 옮김: {', '.join(archived)}")
                tmp_j.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
            os.replace(tmp_p, final_p)
            os.replace(tmp_j, final_j)
        except BaseException:
            # 옮긴 뒤에 실패하면 옛 결과를 제자리로 되돌린다(새 파일이 반만 놓였으면 치운다)
            for f_ in (final_p, final_j):
                if f_.exists():
                    f_.unlink()
            for rel in archived:
                src = out / rel
                dst = final_p if rel.endswith("_daily.parquet") else final_j
                if src.exists():
                    src.rename(dst)
            raise
    finally:
        res["_daily"] = daily
        for t in (tmp_j, tmp_p):
            if t.exists():
                t.unlink()
    rr = res.get("run_record") or {}
    runlog.append_log(out, {"event": "saved", "run_id": rr.get("run_id"), "job_id": rr.get("job_id"), "result": stem,
                            "config_hash": res.get("config_hash"), "model": (res.get("model") or {}).get("file"),
                            "data_version": res.get("data_version"), "master_sha256": (rr.get("master") or {}).get("sha256"),
                            "sealed": bool(res.get("sealed_evaluation")), "replaced_previous": archived})
    return out / f"{stem}.json"
