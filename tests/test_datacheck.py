"""자료 점검표(2026-11-14) — 정상성·계절성·계절조정 표시·빈 기간·확인해 볼 날. 값을 바꾸지 않는 검사다.
시험을 먼저 쓰고(실패 확인) 만든다. 판정 규칙은 engine/datacheck.py 머리말과 도움말 20장 '자료 점검표'에 같은 말로 적는다."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from engine.registry import Registry, Variable
from engine.store import DataVersion


def _rw(n, seed=0, freq="B", start="2010-01-01"):
    r = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(r.normal(0, 1, n)), index=pd.date_range(start, periods=n, freq=freq))


def test_stationarity_verdicts():
    """ADF(H0 단위근)·KPSS(H0 정상) 두 검정을 같이 본다: ADF 기각 & KPSS 기각 못 함 → 정상, 반대 → 추세 남음,
    둘 다 기각·둘 다 기각 못 함 → 판단 어려움. 값 30개 미만 → 자료 부족. 상수 → 판단 불가."""
    from engine.datacheck import stationarity
    rw = _rw(800, seed=1)
    assert stationarity(rw)["verdict"] == "추세 남음"
    noise = pd.Series(np.random.default_rng(2).normal(size=800), index=rw.index)
    st = stationarity(noise)
    assert st["verdict"] == "정상" and st["adf_p"] < 0.05 and st["n"] == 800
    assert st["kpss_p_text"].startswith(">") or st["kpss_p"] > 0.05      # KPSS p 는 표 범위(0.01~0.1) 밖이면 경계값 — 글자로 그렇게 적는다
    assert stationarity(noise.iloc[:20])["verdict"] == "자료 부족"
    assert stationarity(pd.Series(5.0, index=rw.index[:100]))["verdict"] == "판단 불가"
    with warnings.catch_warnings():                                       # 검정이 내는 경고가 밖으로 새지 않는다
        warnings.simplefilter("error")
        stationarity(noise)


def test_seasonality_on_monthly_and_quarterly_only():
    """월·분기 자료만: 달(분기) 더미 회귀의 F 검정 p. p<0.01 → 계절성 남음, p<0.05 → 약한 계절성, 그 밖 → 없음.
    3년치 미만 → 자료 부족. 일·주 자료 → 해당 없음."""
    from engine.datacheck import seasonality
    idx = pd.date_range("2005-01-01", periods=180, freq="MS")
    r = np.random.default_rng(3)
    seas = pd.Series(np.tile([3, 1, 0, -1, -2, -3, -2, 0, 1, 2, 3, -2], 15) + r.normal(0, 0.5, 180), index=idx)
    s1 = seasonality(seas, "M")
    assert s1["verdict"] == "계절성 남음" and s1["p"] < 0.01
    assert seasonality(pd.Series(r.normal(size=180), index=idx), "M")["verdict"] == "없음"
    assert seasonality(seas.iloc[:30], "M")["verdict"] == "자료 부족"
    q = pd.Series(np.tile([2.0, -1, 0, -1], 20) + r.normal(0, 0.3, 80), index=pd.date_range("2000-01-01", periods=80, freq="QS"))
    assert seasonality(q, "Q")["verdict"] == "계절성 남음"
    assert seasonality(_rw(500), "D")["verdict"] == "해당 없음"


def test_seasonal_adjustment_label():
    """계절조정 표시: 변수 사전의 seasonal_adj 칸(SA/NSA)이 있으면 그것, 없으면 이름·설명·비고의 글자('계절조정'·'원계열'),
    그것도 없으면 '확인 필요'. 일·주 자료는 '해당 없음'(계절조정 개념을 쓰지 않는 금리·가격·수급)."""
    from engine.datacheck import sa_label
    assert sa_label(Variable(id="A1", freq="Q", description="실질 GDP 전기 대비 성장률(%, 계절조정)"))[0] == "조정됨"
    assert sa_label(Variable(id="K1", freq="M", name="취업자", description="원계열"))[0] == "조정 안 됨"
    # 비고는 읽지 않는다 — 실제 사전 K1 의 비고처럼 '원계열'과 '계절조정계열로 확인'이 함께 있으면 글자로 짐작할 수 없다
    lab, why = sa_label(Variable(id="K1b", freq="M", name="취업자", note="원계열 아님. 엑셀 고용 변수는 계절조정계열로 확인"))
    assert lab == "확인 필요" and "비고" in why
    assert sa_label(Variable(id="Kd", freq="M", description="계절조정 전 값"))[0] == "조정 안 됨"
    assert sa_label(Variable(id="Kc", freq="M", description="계절조정 전 원계열과 계절조정 계열 둘 다"))[0] == "확인 필요"
    assert sa_label(Variable(id="K2", freq="M", name="실업자"))[0] == "확인 필요"
    assert sa_label(Variable(id="K3", freq="M", name="고용률", seasonal_adj="NSA", description="계절조정"))[0] == "조정 안 됨"   # 칸이 글자보다 먼저
    assert sa_label(Variable(id="X", freq="D", name="코스피"))[0] == "해당 없음"
    assert sa_label(Variable(id="H3", freq="M", description="Actual Transaction Price Index for Apartment"))[0] == "확인 필요"   # 'Transaction' 안의 nsa 에 걸리면 안 됨
    assert sa_label(Variable(id="Z", freq="M", description="Industrial production (SA)"))[0] == "조정됨"
    with pytest.raises(ValueError):
        Variable(id="Y", freq="M", seasonal_adj="maybe")


def _dv(tmp_path):
    dv = DataVersion(tmp_path, "v1")
    r = np.random.default_rng(5)
    idx_m = pd.date_range("2008-01-01", periods=180, freq="MS")
    lvl = pd.Series(100 * np.exp(np.cumsum(0.003 + np.tile([0.02, 0.0, -0.01, 0, 0, -0.02, 0, 0.01, 0, 0, 0.01, -0.01], 15)
                                          + r.normal(0, 0.002, 180))), index=idx_m)
    rw = _rw(1500, seed=6, start="2015-01-01")
    for vid, s in (("SEAS", lvl), ("RW", rw)):
        dv.write_variables(s.rename("value").rename_axis("date").reset_index().assign(variable_id=vid), source=f"syn_{vid}")
    reg = Registry([Variable(id="SEAS", freq="M", name="계절 있는 지수", transform="pct", source="XLSX", source_code="df_M!SEAS"),
                    Variable(id="RW", freq="D", name="랜덤워크 가격", transform="none", source="XLSX", source_code="df_D!RW")])
    return dv, reg


def test_check_table_rows_and_signal(tmp_path):
    """변수 하나의 점검 줄: 변환한 값(모델이 받는 꼴, 원래 주기)으로 정상성·계절성을 보고, 원래 값으로 빈 기간·확인해 볼 날을 본다.
    신호: 초록 = 걸린 것 없음, 노랑 = 추세 남음·계절성 남음·확인해 볼 날·빈 기간 중 하나 이상(이유를 적음), 회색 = 자료 부족·판단 불가.
    upto 를 주면 그 날짜 앞의 값만 쓴다(봉인 구간을 보지 않게)."""
    from engine.datacheck import check_table
    dv, reg = _dv(tmp_path)
    rows = {r["id"]: r for r in check_table(dv, reg)}
    assert set(rows) == {"SEAS", "RW"}
    s, w = rows["SEAS"], rows["RW"]
    assert s["seasonality"]["verdict"] == "계절성 남음" and s["signal"] == "노랑" and any("계절" in x for x in s["reasons"])
    assert any("pct12" in a for a in s["advice"])                                            # 계절성이 남으면 전년 같은 달 대비를 권한다(바꾸지는 않는다)
    assert w["stationarity"]["verdict"] == "추세 남음" and w["signal"] == "노랑" and w["seasonality"]["verdict"] == "해당 없음"
    assert s["transform"] == "pct" and s["n"] == 180
    cut = check_table(dv, reg, upto="2012-01-01")
    assert {r["id"]: r["n"] for r in cut} == {"SEAS": 48, "RW": 0}           # RW 는 2015 년부터라 2012 앞에는 값이 없다
    assert {r["id"]: r["signal"] for r in cut}["RW"] == "회색"
    assert json.dumps(rows, ensure_ascii=False)                              # 결과 파일·화면에 그대로 실린다(JSON 가능)


def test_api_data_checks_and_value_view(tmp_path, monkeypatch):
    """화면: /api/data-checks = 전체 변수 점검표(사용 가능한 변수만, 데이터 버전 표시). [값 보기]에도 그 변수의 점검 줄이 붙는다."""
    import api.main as m
    data = tmp_path / "data"
    dv, reg = _dv(data)
    (data / "versions" / "v1" / "manifest.json").write_text(json.dumps({"start": "2006-01-01"}))
    monkeypatch.setattr(m, "DATA", data)
    monkeypatch.setattr(m, "_registry", lambda: (reg, []))
    t = m.data_checks()
    assert t["version"] == "v1" and {r["id"] for r in t["rows"]} == {"SEAS", "RW"} and t["rule"]
    vv = m.variable_values("SEAS")
    assert vv["datacheck"]["seasonality"]["verdict"] == "계절성 남음"


def test_experiment_result_records_checks_of_used_variables_before_seal(tmp_path):
    """실험 결과에 '쓴 변수'의 점검 줄이 남는다(data_checks). 봉인 시작일이 있으면 그 앞의 값만으로 검사한다 —
    봉인 구간 값을 바꿔도 점검 줄이 같다. 예측·성과 숫자에는 영향이 없다(점검은 계산에 쓰지 않는다)."""
    from engine.experiment import ExperimentConfig, Validation, run_experiment
    from tests.synthetic import predictable_ohlcv
    from tests.test_engine import _dv_with
    prices, s = predictable_ohlcv(700, seed=5)
    dv, reg = _dv_with(prices, s, tmp_path / "a")
    sf = str(prices.index[600].date())
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False,
                           validation=Validation(train_len=250, test_len=40, sealed_from=sf))
    r1 = run_experiment(cfg, dv, reg)
    dc = r1["data_checks"]
    assert [x["id"] for x in dc["rows"]] == ["S_signal"] and dc["upto"] == sf
    s2 = s.copy(); s2[s2.index >= sf] = s2[s2.index >= sf] * 50 + 7
    dv2, reg2 = _dv_with(prices, s2, tmp_path / "b")
    assert run_experiment(cfg, dv2, reg2)["data_checks"]["rows"] == dc["rows"]
    cfg0 = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, validation=Validation(train_len=250, test_len=40))
    a = run_experiment(cfg0, dv, reg)
    assert a["data_checks"]["upto"] is None


def _var(vid, freq, source="XLSX", dup=""):
    return Variable(id=vid, name=vid, group="t", freq=freq, source=source, source_code="x", duplicate_of=dup)


def test_cross_source_flags_only_and_picks_definition():
    """다른 출처와 대조(2026-11-14, 사용자 결정 '표시만'): duplicate_of 짝을 같은 기간끼리 맞춰 본다.
    ① 일간 ↔ 월 자료는 월말·월평균·월초 중 가장 잘 맞는 정의를 쓴다 ② 평소 차이에서 크게 벗어난 기간만 알린다(강건 z > 8, 바닥 ≈1%)
    ③ 한쪽에 값이 없으면 회색 ④ upto 앞 값만 ⑤ 입력 값을 바꾸지 않는다."""
    from engine.datacheck import cross_pair
    d = _rw(2600, seed=3).abs() + 50                                   # 일간(양수)
    m_avg = d.groupby(d.index.to_period("M")).mean()
    m_avg.index = m_avg.index.to_timestamp()                           # 월초 날짜로 적힌 '월평균' 엑셀 자료
    va, vb = _var("XM", "M", dup="YD"), _var("YD", "D", source="YF")
    keep_a, keep_b = m_avg.copy(), d.copy()
    r = cross_pair(va, m_avg, vb, d)
    assert r["signal"] == "초록" and r["definition"] == "평균" and r["n"] == len(m_avg) and r["flags"] == []
    assert r["change_corr"] > 0.99
    pd.testing.assert_series_equal(m_avg, keep_a); pd.testing.assert_series_equal(d, keep_b)
    # 한 달을 5% 틀리게 → 그 달만 걸리고, 노랑
    bad = m_avg.copy(); bad.iloc[40] *= 1.05
    r2 = cross_pair(va, bad, vb, d)
    assert r2["signal"] == "노랑" and [f["period"] for f in r2["flags"]] == [str(bad.index[40].to_period("M"))]
    assert abs(r2["flags"][0]["diff"] - np.log(1.05)) < 1e-9 and r2["flags"][0]["a"] == bad.iloc[40]
    # upto: 틀린 달보다 앞까지만 보면 걸리지 않는다
    assert cross_pair(va, bad, vb, d, upto=str(bad.index[30].date()))["flags"] == []
    # 거의 같은 두 출처(흩어짐 0) — 바닥(≈1%) 아래 차이는 걸리지 않고, 넘는 차이만 걸린다
    same = d.copy(); same.iloc[100] *= 1.005; same.iloc[200] *= 1.03
    r3 = cross_pair(_var("YD2", "D", "FRED", dup="YD"), same, vb, d)
    assert [f["period"] for f in r3["flags"]] == [str(d.index[200].date())]
    # 한쪽 값이 없으면 회색
    r4 = cross_pair(va, pd.Series(dtype=float), vb, d)
    assert r4["signal"] == "회색" and "XM" in r4["reasons"][0]


def test_cross_table_uses_duplicate_pairs(tmp_path):
    """cross_table 은 변수 사전의 duplicate_of 짝 전부를 본다(실험에서 빠지는 중복 변수도 포함). ids 를 주면 그 변수가 든 짝만."""
    from engine.datacheck import cross_table
    reg = Registry()
    reg.add(_var("YD", "D", "YF")); reg.add(_var("XM", "M", dup="YD")); reg.add(_var("Z", "D", "YF"))
    d = _rw(1500, seed=4).abs() + 50
    m = d.groupby(d.index.to_period("M")).last(); m.index = m.index.to_timestamp("M")
    long = pd.concat([pd.DataFrame({"variable_id": "YD", "date": d.index, "value": d.values}),
                      pd.DataFrame({"variable_id": "XM", "date": m.index, "value": m.values}),
                      pd.DataFrame({"variable_id": "Z", "date": d.index, "value": d.values})])
    class DV:
        def read_variables(self, ids):
            return long[long["variable_id"].isin(ids)]
    rows = cross_table(DV(), reg)
    assert [(r["id"], r["ref_id"]) for r in rows] == [("XM", "YD")] and rows[0]["signal"] == "초록" and rows[0]["definition"] == "말일"
    assert cross_table(DV(), reg, ids=["YD"]) and cross_table(DV(), reg, ids=["Z"]) == []
