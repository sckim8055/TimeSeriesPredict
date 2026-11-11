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
from engine.backtest import performance, CostModel, bootstrap_excess_pvalue, clark_west, deflated_sharpe_pvalue, diebold_mariano, diebold_mariano_brier, return_moments, run_backtest
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
    kind: Literal["return", "logreturn", "direction", "event"] = "return"
    k: int = Field(1, ge=1, description="예측 지평(주기 수)")
    threshold: float = 0.0


class TradeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 모르는 키(오타)는 오류 — 조용히 기본값으로 돌지 않게(2026-11-10)
    threshold: float = Field(0.0, description="regression: 예상수익률 > threshold → 롱. classification: 확률 > 0.5+threshold → 롱")
    fee_rate: float = 0.00015
    slippage_bp: float = 5.0


class Validation(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 모르는 키(오타)는 오류 — 조용히 기본값으로 돌지 않게(2026-11-10)
    train_len: int = Field(756, description="학습 창 길이(주기 수). 일간 3년≈756")
    test_len: int = Field(21, description="재학습 주기(주기 수). 일간 1개월≈21")
    embargo: int | None = None  # None → 1+k
    sealed_from: str | None = None  # 'YYYY-MM-DD' 이후는 봉인(폴드 생성 안 함)
    val_frac: float = 0.15  # 학습 창 뒤쪽 비율을 검증용(조기종료 등)으로
    mode: Literal["walkforward", "split"] = Field("walkforward", description="walkforward: 창을 밀며 반복 재학습 / split: 1회 분할(노트북 방식, split_date 이전 학습·이후 테스트)")
    split_date: str | None = Field(None, description="(이전 이름) mode=split 의 테스트 시작일. test_start 가 있으면 test_start 를 쓴다")
    val_start: str | None = Field(None, description="mode=split 에서 검증 시작일 'YYYY-MM-DD'. 검증 구간 = [val_start, 테스트 시작 − embargo). 비우면 val_frac")
    test_start: str | None = Field(None, description="테스트(평가) 시작일. split: 이 날 이전이 학습. walkforward: 첫 테스트 폴드가 이 날 시작(그 전 폴드는 만들지 않음) — 모든 모델의 평가 기간을 맞출 때")
    test_end: str | None = Field(None, description="테스트(평가) 종료일(포함). 비우면 봉인 시작 전날 또는 데이터 끝")


class ExperimentConfig(BaseModel):
    # 모르는 키는 오류(전에는 무시 — 'frequncy' 같은 오타가 기본값으로 조용히 돌았다, 2026-11-10). '_' 로 시작하는 키(_comment)는 메모로 보고 버린다
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _drop_comment_keys(cls, data):
        return {k: v for k, v in data.items() if not str(k).startswith("_")} if isinstance(data, dict) else data

    @model_validator(mode="after")
    def _normalise_technical(self):
        """technical_features 정규화 — 같은 입력이면 같은 해시가 되게: 기술지표를 끄면 None, 중복 제거, 만드는 순서대로, 전부면 None."""
        tf = self.technical_features
        if tf is None:
            return self
        if not self.use_technical:
            object.__setattr__(self, "technical_features", None)
            return self
        try:
            from engine.features import technical_columns
            allc = technical_columns(self.frequency, with_volume=(self.mode != "forecast"))
        except Exception:  # noqa: BLE001 — 기술지표가 정의되지 않은 주기: 그대로 두고 실행 때 오류로 드러나게
            return self
        known = [c for c in allc if c in set(tf)]
        unknown = [c for c in dict.fromkeys(tf) if c not in set(allc)]  # 없는 이름은 남겨 두어 실행 때 select_technical 이 오류로 알린다
        norm = known + unknown
        if not unknown and len(known) == len(allc):
            norm = None
        elif not norm:
            norm = []  # 빈 목록은 실행 때 오류(전부 빼려면 use_technical 을 끈다)
        object.__setattr__(self, "technical_features", norm)
        return self

    name: str = ""
    mode: Literal["trading", "forecast"] = Field("trading", description="trading: 매매 시뮬레이션(ETF 가격, 체결·비용·B&H) / forecast: 예측 연구(임의 시계열, 오차 지표·naive 대비)")
    target_variable: str | None = Field(None, description="forecast 모드의 대상 변수 id (마스터). 주기는 그 변수의 원래 주기")
    forecast_info_time: Literal["stamp", "release"] = Field(
        "stamp", description="forecast 모드에서 외생변수를 '언제까지 알려진 것'으로 맞출지. stamp: 대상 값의 기준일 t(보수적, 기본). "
                             "release: 대상 값 v[t] 가 공개되는 시점 t+lag_대상 — 실제로 예측을 하는 시점의 정보 집합(현실적)")
    ticker: str = "091160"
    frequency: Literal["D", "W", "M"] = "D"
    start: str | None = Field(None, description="데이터 사용 시작일. 이 날 이전 데이터는 읽지 않는다(학습·검증·테스트 모두)")
    end: str | None = Field(None, description="데이터 사용 종료일. 이 날 이후 데이터는 읽지 않는다")
    target: TargetSpec = TargetSpec()
    variable_ids: list[str] = Field(default_factory=list, description="레지스트리 변수 ID. 비우면 A그룹(기술지표)만")
    use_technical: bool = True
    technical_features: list[str] | None = Field(None, description="쓸 기술지표 열 이름(A_…). 비우면(None) 전부 — 2026-11-09 추가. 해시에는 값이 있을 때만 들어간다")
    fill: Literal["ffill", "bfill", "interp"] = "ffill"
    apply_lag: bool = True
    variable_transforms: dict[str, str] = Field(default_factory=dict, description="변수별 변환 재정의 {id: none|pct|pct{n}|diff|diff{n}|log|logdiff}. 비우면 마스터의 transform")
    lag_override: dict[str, int] = Field(default_factory=dict, description="변수별 lag_days 재정의(가설 검정용). 결과에 기록된다")
    model: str = Field(..., description="models/ 파일명 (확장자 없이)")
    model_params: dict[str, Any] = Field(default_factory=dict)
    seq_len: int | None = Field(None, description="window 입력 길이 (window 모델만). 비우면 주기별 기본값 D 20 / W 13 / M 12")
    trade: TradeRule = TradeRule()
    validation: Validation = Validation()
    data_version: str = ""
    seed: int = 2025

    def config_hash(self) -> str:
        # technical_features 는 2026-11-09 에 생긴 칸: 기본값(None = 전부)이면 해시에서 빼서 그 전 결과의 이름·재실행 연결을 그대로 둔다
        ex = {"name"} | ({"technical_features"} if self.technical_features is None else set())
        return hashlib.sha1(self.model_dump_json(exclude=ex).encode()).hexdigest()[:12]


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


_TR_RE = re.compile(r"^(pct|diff)(\d*)$")


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
            raw = apply_transform(wide[vid].dropna(), tr)  # 변환은 원래 주기에서, 정렬 전에
            bad = raw.index[np.isinf(raw.values.astype(float))]
            if len(bad):
                raise ValueError(
                    f"변수 {vid} 에 변환 {tr} 을 적용하면 무한대/계산불가 값이 {len(bad)}개 생깁니다 (예: {bad[0].date()}). "
                    f"원값에 0 이나 음수가 있는 시리즈(이미 변화율·성장률인 값, 수지, 순매수)에는 pct/log 를 쓸 수 없습니다 — "
                    f"변수 사전의 transform 을 none 또는 diff 로 바꾸거나 실험 설정에서 재정의하세요")
            lag = lo_map.get(vid, v.lag_days)
            name = vid if tr in ("", "none") else f"{vid}_{tr}"
            cols[name] = cal.align_to_calendar(raw, info_calendar, lag_days=lag, fill=cfg.fill, apply_lag=cfg.apply_lag).set_axis(calendar)
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
    return Dataset(prices, X, y, list(X.columns), cal.fill_leaks_future(cfg.fill), unconf, used, trs, levels)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
# 결과 계산 방식이 바뀐 날. 결과 json 에 'engine_rev' 로 남고, tools/rerun_stale.py 가 이 값으로 옛 결과를 찾는다.
# 2026-11-07: 창 입력 모델 눈금을 학습 폴드 전체 행으로 · 중복 변수 ID 의 변환·lag 설정을 대표 ID 로.
ENGINE_REV = "2026-11-07"


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
        sf = pd.Timestamp(v.sealed_from) if v.sealed_from else None
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
                                  sealed_from=pd.Timestamp(v.sealed_from) if v.sealed_from else None)
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
                         capture: dict | None = None):
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
        scaler = (MinMaxScaler() if params.get("input_scaler") == "minmax" else StandardScaler()).fit(tr[ds.feature_names].values)
        fill_val = scaler.transform(tr[ds.feature_names].values).mean(axis=0)  # 빈 칸을 채울 '학습 구간 평균'(표준화면 0)
        Xtr = scaler.transform(tr[ds.feature_names].values)
        ytr = tr["y"].values
        te_valid = te.dropna()
        Xte = scaler.transform(te_valid[ds.feature_names].values)
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
            scaler = (MinMaxScaler() if params.get("input_scaler") == "minmax" else StandardScaler()).fit(full[ds.feature_names].ffill().values)
            Xf = scaler.transform(full[ds.feature_names].ffill().values)
            fill_val = np.nanmean(Xf, axis=0)  # 빈 칸을 채울 '학습 구간 평균'(표준화면 0)
            if not accepts_nan:
                Xf = np.where(np.isnan(Xf), fill_val[None, :], Xf)
            Xw, yw = cal.make_windows(Xf, full["y"].values, cfg.seq_len)
            ok = np.isfinite(yw) & (np.ones(len(yw), bool) if accepts_nan else np.isfinite(Xw).all(axis=(1, 2)))
            win_idx = full.index[cfg.seq_len - 1:][: len(ok)][ok]  # 각 창의 결정일(창의 마지막 행)
            Xw, yw = Xw[ok], yw[ok]
            # 테스트: 학습 창 끝 seq_len-1 행을 앞에 붙여 첫 테스트일부터 윈도우 완성
            ctx = ds.X.loc[: f.test[0]].iloc[-cfg.seq_len:-1] if cfg.seq_len > 1 else ds.X.iloc[0:0]
            te_full = pd.concat([ctx, te])
            Xtf = scaler.transform(te_full[ds.feature_names].ffill().values)
            if not accepts_nan:
                Xtf = np.where(np.isnan(Xtf), fill_val[None, :], Xtf)
            Xtw, _ = cal.make_windows(Xtf, np.zeros(len(Xtf)), cfg.seq_len)
            te_idx = te.index
            m: PluginBase = plugin.Model(params, n_features=len(ds.feature_names), target_type=target_type, device=device)
            n_tr, n_va = _split_fit(m, Xw, yw, _nval(win_idx, f), emb_v)
            _log(f, win_idx[:n_tr], win_idx[len(win_idx) - n_va:] if n_va else win_idx[:0], te_idx)
            p = m.predict(Xtw)
            pred.loc[te_idx] = p[: len(te_idx)]
            if capture is not None:
                capture.update(kind="window", fold=f, model=m, scaler=scaler, fill_val=fill_val, Xw=Xw, yw=yw, win_idx=win_idx, n_tr=n_tr, n_va=n_va,
                               Xtw=Xtw, te_idx=te_idx, te_full=te_full, accepts_nan=accepts_nan, pred=pd.Series(p[: len(te_idx)], index=te_idx))
            if want_fit and len(Xw):
                fit.loc[win_idx] = m.predict(Xw)[: len(win_idx)]
        else:
            m = plugin.Model(params, n_features=len(ds.feature_names), target_type=target_type, device=device)
            n_tr, n_va = _split_fit(m, Xtr, ytr, _nval(tr.index, f), emb_v)
            _log(f, tr.index[:n_tr], tr.index[len(tr) - n_va:] if n_va else tr.index[:0], te_valid.index)
            pred.loc[te_valid.index] = m.predict(Xte)
            if capture is not None:
                capture.update(kind="table", fold=f, model=m, scaler=scaler, tr=tr, te=te, te_valid=te_valid, Xtr=Xtr, ytr=ytr, Xte=Xte,
                               n_tr=n_tr, n_va=n_va, pred=pred.loc[te_valid.index].copy())
            if want_fit:
                fit.loc[tr.index] = m.predict(Xtr)
        fi = m.feature_importance()
        if fi is not None:
            importances.append(np.asarray(fi, dtype=float))
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
    sf = pd.Timestamp(v.sealed_from)
    emb = _embargo(cfg, k)
    calendar_all = cal.trading_calendar(ds.prices)
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


def _predict_sealed(cfg, plugin, params, ds, device, sealed_from, k, log: list | None = None):
    """봉인 구간 예측(sealed_folds 의 폴드로 학습→예측)."""
    target_type = "classification" if cfg.target.kind in ("direction", "event") else "regression"
    return _walkforward_predict(cfg, plugin, params, ds, device, sealed_folds(cfg, ds, k), k, target_type, log=log)


def run_experiment(cfg: ExperimentConfig, dv: DataVersion, reg: Registry, *, n_trials_so_far: int = 1, device: str | None = None, sealed: bool = False, trial_sr_var: float | None = None) -> dict:
    """sealed=True 면 봉인 구간(sealed_from 이후)만 평가한다. 봉인 평가는 후보당 1회가 원칙 — 결과에 sealed=True 로 기록된다.
    결과의 periods 에 실제로 쓴 데이터·학습·검증·테스트 기간(첫 폴드·마지막 폴드)을 남긴다 — 논문 표에 그대로 옮길 값."""
    _check_periods(cfg)
    from engine import runlog
    started = runlog.now_utc()
    # 변수 치환·제외 내역(build_dataset 과 같은 규칙·같은 입력이라 같은 결과) — 결과의 run_record 에 남긴다
    var_detail = reg.resolve_detail(cfg.variable_ids) if cfg.variable_ids else None
    ov_notes = reg.remap_overrides(cfg.variable_ids, cfg.variable_transforms)[1] + reg.remap_overrides(cfg.variable_ids, cfg.lag_override)[1]
    fold_log: list = []
    with step(log_e, f"실험 {cfg.name or cfg.model} ({cfg.mode})", logging.INFO, 모델=cfg.model, 변수=cfg.variable_ids, 기간={"start": cfg.start, "end": cfg.end, **cfg.validation.model_dump()}):
        res = _run_experiment(cfg, dv, reg, n_trials_so_far=n_trials_so_far, device=device, sealed=sealed, trial_sr_var=trial_sr_var, fold_log=fold_log)
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
    }
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
    if cfg.seq_len is None:
        dflt = {"D": 20, "W": 13, "M": 12, **(getattr(plugin, "SEQ_LEN_DEFAULT", None) or {})}  # 모델이 원본 기본을 선언하면 그것(BIVA 월간 6)
        cfg = cfg.model_copy(update={"seq_len": dflt.get(freq_for_defaults, 20)})
    log_e.info("모델 %s(%s) · 장치 %s · 파라미터(받은 값 %s → 기본값 채운 실제 값) %s · 창 길이 %s", plugin.NAME, plugin.INPUT_SHAPE, device,
               describe(cfg.model_params), describe(params), cfg.seq_len if plugin.INPUT_SHAPE == "window" else "-")
    with step(log_d, "데이터셋 만들기(읽기→변환→발표시차 정렬→타겟)"):
        ds = build_dataset(cfg, dv, reg)
    return cfg, plugin, params, target_type, ds

def _run_experiment(cfg: ExperimentConfig, dv: DataVersion, reg: Registry, *, n_trials_so_far: int = 1, device: str | None = None, sealed: bool = False, trial_sr_var: float | None = None, fold_log: list | None = None) -> dict:
    t0 = time.time()
    device = device or default_device()
    cfg, plugin, params, target_type, ds = prepare(cfg, dv, reg, device=device)
    k = cfg.target.k

    pred = pd.Series(np.nan, index=ds.prices.index, name="prediction")
    importances: list[np.ndarray] = []
    n_folds = 0

    if cfg.mode == "forecast" and plugin.INPUT_SHAPE == "rule":
        raise ValueError("예측 연구 모드에는 규칙(매매) 모델을 쓸 수 없습니다")
    if plugin.INPUT_SHAPE == "rule":
        model: RuleModel = plugin.Model(params)
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
        pred, n_folds, importances = _predict_sealed(cfg, plugin, params, ds, device, cfg.validation.sealed_from, k, log=fold_log)
        if target_type == "regression":
            signal = (pred > cfg.trade.threshold).astype(float).where(pred.notna())
        else:
            signal = (pred > 0.5 + cfg.trade.threshold).astype(float).where(pred.notna())
    else:
        folds, is_split = make_folds(cfg, ds, k)
        fit = None
        if is_split:
            pred, n_folds, importances, fit = _walkforward_predict(cfg, plugin, params, ds, device, folds, k, target_type, want_fit=True, log=fold_log)
        else:
            pred, n_folds, importances = _walkforward_predict(cfg, plugin, params, ds, device, folds, k, target_type, log=fold_log)
        # 신호 변환
        if target_type == "regression":
            signal = (pred > cfg.trade.threshold).astype(float).where(pred.notna())
        else:
            signal = (pred > 0.5 + cfg.trade.threshold).astype(float).where(pred.notna())

    cost = CostModel(fee_rate=cfg.trade.fee_rate, slippage_bp=cfg.trade.slippage_bp)
    ppy = {"D": 252, "W": 52, "M": 12}[cfg.frequency]
    sealed_from = pd.Timestamp(cfg.validation.sealed_from) if cfg.validation.sealed_from else None
    if cfg.mode == "forecast":
        return _finish_forecast(cfg, plugin, params, ds, device, pred, fit if cfg.validation.mode == "split" else None,
                                importances, n_folds, k, target_type, sealed, sealed_from, t0, fold_log=fold_log)
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

    daily_full, _ = run_backtest(ds.prices, signal, cost=cost, periods_per_year=ppy)
    daily_full["prediction"] = pred
    daily_full["target"] = ds.y
    if plugin.INPUT_SHAPE != "rule" and cfg.validation.mode == "split" and fit is not None:
        daily_full["fit"] = fit  # 학습 구간 적합값(차트용). 성과·검정에는 쓰지 않는다
    # 성과는 '신호가 있는 구간'(첫 신호~마지막 신호)에서만 계산한다. 그 밖은 B&H 와 비교할 수 없다.
    has = signal.dropna()
    if len(has) == 0:
        raise RuntimeError("신호가 하나도 없습니다")
    daily = daily_full.loc[has.index.min(): has.index.max()].copy()
    perf = performance(daily["net_return"], daily["hold_return"], periods_per_year=ppy)
    turnover = daily["position"].diff().abs().fillna(daily["position"].abs())
    perf.update(n_trades=int((turnover > 0).sum()), turnover=float(turnover.sum()), exposure=float(daily["position"].mean()), total_cost=float(daily["cost"].sum()))

    # 예측 성능 (예측이 있는 구간)
    ev = daily[["prediction", "target"]].dropna()
    pred_metrics: dict[str, float] = {}
    if plugin.INPUT_SHAPE != "rule" and len(ev) > 5:
        e = ev["prediction"] - ev["target"]
        if target_type == "regression":
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
        "degenerate_model": degenerate,
        "sealed_evaluation": bool(sealed),
        "evaluation_period": [str(daily.index[0].date()), str(daily.index[-1].date())],
        "leak_warning": ds.leak_warning,
        "lag_days_used": ds.lag_days_used,
        "lag_unconfirmed": ds.lag_unconfirmed,
        "transforms_used": ds.transforms_used,
        "prediction_metrics": pred_metrics,
        "performance": perf,
        "period": [str(daily.index[0].date()), str(daily.index[-1].date())],
        "prediction_period": [str(ev.index[0].date()), str(ev.index[-1].date())] if len(ev) else None,
        "n_predictions": int(len(ev)),
        "variable_first_date": {c: (str(ds.X[c].first_valid_index().date()) if ds.X[c].first_valid_index() is not None else None) for c in ds.feature_names if not c.startswith("A_")},
        "device": device,
        "seconds": round(time.time() - t0, 1),
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
    return pm, ev


def _finish_forecast(cfg, plugin, params, ds, device, pred, fit, importances, n_folds, k, target_type, sealed, sealed_from, t0, fold_log: list | None = None) -> dict:
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
        "leak_warning": ds.leak_warning, "lag_days_used": ds.lag_days_used, "lag_unconfirmed": ds.lag_unconfirmed, "transforms_used": ds.transforms_used,
        "prediction_metrics": pm, "performance": {}, "period": [str(daily_full.index[0].date()), str(daily_full.index[-1].date())],
        "prediction_period": [str(ev.index[0].date()), str(ev.index[-1].date())], "n_predictions": int(len(ev)),
        "variable_first_date": {c: (str(ds.X[c].first_valid_index().date()) if ds.X[c].first_valid_index() is not None else None) for c in ds.feature_names if not c.startswith("A_")},
        "device": device, "seconds": round(time.time() - t0, 1), "_daily": daily_full,
    }


def trial_sharpe_variance(results_dir: str | Path, periods_per_year: int) -> float | None:
    """지금까지 저장된 실험들의 기간 샤프 분산 (Deflated Sharpe 의 V). 같은 주기(periods_per_year)의 결과만. 2개 미만이면 None.
    results_dir 는 모드별 폴더(results/trading)여야 한다 — 2026-11-04 이주 뒤 cli 가 results/ 바로 아래를 넘겨 V 가 늘 None(근사 1/(n−1))이었다(2026-11-06 수정).
    봉인 평가 결과는 '시도'가 아니므로 뺀다. 계산 방식이 지금과 다른 옛 결과(engine_rev 불일치)도 뺀다."""
    srs = []
    for p in Path(results_dir).glob("*.json"):
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
    try:
        daily.to_parquet(tmp_p)
        archived = runlog.archive_existing(out, stem)
        if archived:
            res.setdefault("run_record", {})["replaced_previous"] = archived
            print(f"  [실행 기록] 같은 이름의 옛 결과를 옮김: {', '.join(archived)}")
        tmp_j.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        os.replace(tmp_p, final_p)
        os.replace(tmp_j, final_j)
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
