"""데이터 변화 단계(2026-11-21) — 결과 하나에 들어간 입력 변수가 원자료에서 모델 입력이 되기까지 거친 단계를 결정일마다 남긴다.

변수마다(결정일 한 줄씩):
  사용한 원자료 기준일 → 공개일(기준일 + 발표 시차) → 원래 값 → ① 시차·채움 후 값(수준) → ② 변환 후 값 = 모델 입력
  + 새 값이 공개된 날인가, 빈칸이면 왜 빈칸인가.
원칙: 다시 구현하지 않는다 — build_dataset 과 같은 함수(apply_transform·align_to_calendar)·같은 재정의 규칙(remap_overrides)으로 만들고,
마지막 열(모델 입력)이 실제 데이터셋(ds.X)의 그 열과 같은지 확인해 '일치' 칸에 적는다. 다르면 감추지 않고 '불일치'로 드러난다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import calendar as cal


def _order_text(tr: str, lag: int, fill: str, basis: str, apply_lag: bool) -> str:
    lag_t = f"발표 시차 {lag}일" if apply_lag else "발표 시차 적용 안 함(apply_lag=false)"
    fill_t = {"ffill": "앞 값 유지(ffill)", "bfill": "다음 값 당겨오기(bfill — 미래 정보 포함)", "interp": "앞뒤 사이 보간(interp — 미래 정보 포함)"}.get(fill, fill)
    if basis == "model_period":
        return f"원래 값 → {lag_t} → 결정일에 맞춤·{fill_t} → 결정일 사이 변화({tr})"
    if basis == "native":
        return f"원래 값 → 원래 주기에서 변환({tr}) → {lag_t} → 결정일에 맞춤·{fill_t}"
    if tr == "log":
        return f"원래 값 → 로그 → {lag_t} → 결정일에 맞춤·{fill_t}"
    return f"원래 값 → {lag_t} → 결정일에 맞춤·{fill_t} (변환 없음)"


def variable_trace(cfg, dv, reg, ds) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """(변수별 요약 표, {입력 열 이름: 결정일별 단계 표})."""
    from engine.experiment import _FREQ_RANK, apply_transform
    calendar = pd.DatetimeIndex(ds.X.index)
    if not cfg.variable_ids:
        return pd.DataFrame(), {}
    if cfg.mode == "forecast":
        freq = reg.get(cfg.target_variable).freq
    else:
        freq = cfg.frequency
    info_cal = calendar
    release = cfg.mode == "forecast" and cfg.forecast_info_time == "release"
    if release:
        lag_t = int(cfg.lag_override.get(cfg.target_variable, reg.get(cfg.target_variable).lag_days))
        info_cal = pd.DatetimeIndex(calendar + pd.Timedelta(days=lag_t))
    var_ids, _ = reg.resolve_many(cfg.variable_ids)
    vt_map, _ = reg.remap_overrides(cfg.variable_ids, cfg.variable_transforms)
    lo_map, _ = reg.remap_overrides(cfg.variable_ids, cfg.lag_override)
    wide = dv.wide(var_ids)
    summary, tables = [], {}
    for vid in var_ids:
        v = reg.get(vid)
        tr = vt_map.get(vid, v.transform or "none")
        lag = int(lo_map.get(vid, v.lag_days))
        period_tr = tr.strip().lower() in ("pct", "diff", "logdiff") and _FREQ_RANK.get(str(v.freq), 0) < _FREQ_RANK.get(str(freq), 0)
        basis = "model_period" if period_tr else ("native" if tr not in ("", "none", "log") else "level")
        name = vid if tr in ("", "none") else f"{vid}_{tr}"
        raw0 = wide[vid].dropna().astype(float)
        raw0.index = pd.DatetimeIndex(raw0.index).normalize()
        raw0 = raw0[~raw0.index.duplicated(keep="last")].sort_index()
        al = lambda s, fill=cfg.fill: cal.align_to_calendar(s, info_cal, lag_days=lag, fill=fill, apply_lag=cfg.apply_lag).set_axis(calendar)  # noqa: E731
        # 원자료 기준일을 값으로 갖는 시리즈를 같은 규칙으로 맞추면 '그 결정일에 쓴 원자료의 기준일'이 나온다
        src_num = pd.Series(raw0.index.asi8.astype(float), index=raw0.index)
        use_fill = "bfill" if cfg.fill == "bfill" else "ffill"
        src = al(src_num, use_fill)
        src_d = pd.to_datetime(src, errors="coerce")
        orig = raw0.reindex(src_d.values).set_axis(calendar)
        level = al(raw0)
        if period_tr:
            final = apply_transform(level, tr)
        else:
            final = al(apply_transform(raw0, tr))
        actual = ds.X[name] if name in ds.X.columns else pd.Series(np.nan, index=calendar)
        same = bool(np.allclose(final.values.astype(float), actual.values.astype(float), rtol=0, atol=1e-12, equal_nan=True)) and bool((final.isna() == actual.isna()).all())
        new = src_d.ne(src_d.shift(1)) & src_d.notna()
        why = np.where(src_d.isna(), "첫 공개 전(아직 알 수 있는 값이 없음)",
                       np.where(final.isna(), "변환할 이전 값이 없음(변환의 첫 칸)", ""))
        df = pd.DataFrame({"결정일": calendar.date})
        if release:
            df["정보 시점(대상 값 공개일)"] = info_cal.date
        df["사용한 원자료 기준일"] = src_d.dt.date.values
        df["공개일(기준일+시차)"] = (src_d + pd.to_timedelta(lag if cfg.apply_lag else 0, unit="D")).dt.date.values
        if cfg.fill == "interp":
            nxt = pd.to_datetime(al(src_num, "bfill"), errors="coerce")
            nxt = nxt.where(src_d.notna() & (nxt != src_d) & level.notna())   # 실제로 두 공개값 사이를 보간한 날만
            df["보간에 쓴 다음 기준일(미래)"] = nxt.dt.date.values
        df["원래 값"] = orig.values
        df["① 시차·채움 후 값(수준)"] = level.values
        df[f"② 변환({tr}) 후 = 모델 입력 [{name}]"] = final.values
        df["새 값이 공개된 날"] = np.where(new.values, "예", "")
        df["빈칸 이유"] = why
        tables[name] = df
        chg = final.dropna()
        last_chg = chg.index[np.flatnonzero(np.r_[True, np.diff(chg.values.astype(float)) != 0])[-1]] if len(chg) else None
        summary.append({
            "변수 ID": vid, "이름": v.name, "모델 입력 열": name, "변수 주기": v.freq, "모델 주기": freq,
            "원자료 개수": len(raw0), "원자료 기간": f"{raw0.index[0].date()} ~ {raw0.index[-1].date()}" if len(raw0) else "",
            "변환": tr, "발표 시차(일)": lag, "채움": cfg.fill, "거친 순서": _order_text(tr, lag, cfg.fill, basis, cfg.apply_lag),
            "미래 정보 포함": "예(채움 방식)" if cal.fill_leaks_future(cfg.fill) else "",
            "첫 사용 결정일": chg.index[0].date() if len(chg) else "", "결정일 수": len(calendar), "빈칸 결정일 수": int(final.isna().sum()),
            "새 값 공개 횟수": int(new.sum()), "마지막 값 변경 결정일": last_chg.date() if last_chg is not None else "",
            "그 뒤 같은 값 행 수": int((calendar > last_chg).sum()) if last_chg is not None else "",
            "실제 입력과 일치": "일치" if same else "불일치 — 이 표를 만드는 규칙이 엔진과 다름(알려 주세요)",
        })
    return pd.DataFrame(summary), tables


def technical_trace(ds, var_cols, registry=None) -> pd.DataFrame:
    """가격에서 계산한 기술지표 열(변수에서 온 열이 아닌 입력 열): 뜻·첫 값이 있는 결정일·빈칸 수."""
    from engine.features import describe_feature, technical_group
    rows = []
    for c in ds.feature_names:
        if c in set(var_cols):
            continue
        g = technical_group(c)
        s = ds.X[c]
        nn = s.dropna()
        rows.append({"입력 열": c, "묶음": g, "뜻(계산 규칙)": describe_feature(c, registry), "첫 값 결정일": nn.index[0].date() if len(nn) else "",
                     "빈칸 결정일 수": int(s.isna().sum()), "비고": "그날 종가까지의 가격만으로 계산(engine/features.py)"})
    return pd.DataFrame(rows)


def pipeline_steps(r: dict, cfg: dict, ds, var_summary: pd.DataFrame, n_tech: int) -> list[tuple[str, str, str, str]]:
    """결과 하나가 거친 단계 전체: (단계, 무엇을 했나, 이 결과에서, 확인할 파일)."""
    v = cfg.get("validation") or {}
    t = cfg.get("target") or {}
    shape = (r.get("model") or {}).get("input_shape", "")
    params = (r.get("model") or {}).get("params") or {}
    folds = (r.get("periods") or {}).get("folds") or []
    drop = sum(int(f.get("n_train_dropped_blank") or 0) for f in folds)
    nopred = sum(int(f.get("n_test_not_predicted") or 0) for f in folds)
    scaler = params.get("input_scaler", "standard")
    S = [("1 원자료", "데이터 버전에 저장된 값을 그대로 읽음", f"데이터 버전 {r.get('data_version')} · 변수 {len(var_summary)}개", "1_사전_입력/변수_사전.csv, 데이터_버전.json")]
    if len(var_summary):
        S.append(("2 발표 시차·채움·변환", "변수마다 '그 결정일에 알 수 있었던 값'만 쓰게 시차를 두고, 결정일에 맞춰 채우고, 변환",
                  f"채움 {cfg.get('fill', 'ffill')} · 시차 적용 {cfg.get('apply_lag', True)} · 실제 입력과 일치 {int((var_summary['실제 입력과 일치'] == '일치').sum())}/{len(var_summary)}",
                  "1_사전_입력/데이터_변화_단계/요약.csv 와 변수별 CSV"))
    if n_tech:
        S.append(("3 기술지표", "가격에서 수익률·이동평균 비율·변동성·RSI·MACD 등을 계산", f"{n_tech}개 열", "1_사전_입력/데이터_변화_단계/기술지표.csv"))
    S.append(("4 정답 만들기", "결정일 t 의 정답 = 그 뒤 k 기간의 값(입력과 겹치지 않게)", f"{t.get('kind')} · k={t.get('k')}", "1_사전_입력/입력_자료.csv 의 정답_target"))
    S.append(("5 모델 입력 표", "결정일 × 입력 열로 묶음", f"결정일 {len(ds.X) if ds is not None else '?'}개 × 입력 {len(r.get('feature_names') or [])}개", "1_사전_입력/입력_자료.csv"))
    S.append(("6 학습·검증·테스트 나누기", "폴드마다 학습 창과 테스트 구간을 정하고, 사이에 엠바고를 둠",
              f"{v.get('mode', 'walkforward')} · 폴드 {r.get('n_folds')}개", "2_사중_진행/폴드별_기간.csv, 1_사전_입력/검증_계획.json"))
    if shape == "table":
        S.append(("7 빈칸 행 처리", "입력에 빈칸이 있는 학습 행은 빼고, 빈칸 있는 테스트 행은 예측하지 않음",
                  (f"뺀 학습 행 합 {drop} · 예측 못한 테스트 행 합 {nopred}" if folds and "n_train_dropped_blank" in folds[0] else "이 결과에는 기록 없음(2026-11-21 이전 결과)"), "2_사중_진행/폴드별_기간.csv"))
    elif shape == "window":
        S.append(("7 빈칸 처리", "앞 값으로 채운 뒤에도 남는 빈칸은 학습 구간 평균(표준화면 0)으로, BRITS 를 켠 BIVA 는 빈칸 그대로",
                  (f"채운 빈칸 합 {sum(int(f.get('n_blank_filled') or 0) for f in folds)}" if folds and "n_blank_filled" in folds[0] else "이 결과에는 기록 없음(2026-11-21 이전 결과)"), "2_사중_진행/폴드별_기간.csv"))
    if shape in ("table", "window"):
        S.append(("8 눈금 맞추기", f"폴드마다 학습 구간 값만으로 {('표준화(평균 0, 표준편차 1)' if scaler != 'minmax' else '최소-최대(0~1)')} 기준을 정해 학습·테스트에 같이 적용",
                  "폴드별 기준값 기록 있음" if folds and "scale" in folds[0] else "이 결과에는 기록 없음(2026-11-21 이전 결과)", "2_사중_진행/폴드별_눈금.csv"))
    if shape == "window":
        S.append(("9 창 만들기", "결정일마다 앞의 seq_len 개 시점을 한 묶음으로", f"seq_len={cfg.get('seq_len') or '주기 기본값'}", "1_사전_입력/검증_계획.json"))
    S.append(("10 학습·예측", "학습 창으로 모델을 맞추고 테스트 결정일마다 예측", f"{cfg.get('model')} · 시드 {cfg.get('n_seeds', 1)}개", "2_사중_진행/job_output.log, 3_사후_결과/일별_결과.csv"))
    if cfg.get("mode", "trading") == "trading":
        S.append(("11 매매·평가", "예측을 매매 규칙에 넣어 포지션을 정하고 비용을 빼서 성과 계산", f"수수료 {cfg.get('fee_rate')} · 슬리피지 {cfg.get('slippage_bp')}bp", "3_사후_결과/일별_결과.csv, 지표와_뜻.csv"))
    else:
        S.append(("11 평가", "예측과 정답·naive 기준선을 비교", "", "3_사후_결과/지표와_뜻.csv"))
    S.append(("12 계산 따라가기", "결정일 하루를 같은 코드로 다시 계산해 중간값을 모두 보여 주고, 결과의 예측과 같은지 확인", "", "2_사중_진행/계산_따라가기/"))
    return S
