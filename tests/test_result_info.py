"""결과 화면 보강(2026-11-20): 설정 모든 칸의 뜻, 결과별 로그 내려받기."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime

from fastapi.testclient import TestClient


def _paths(model, prefix=""):
    from pydantic import BaseModel
    out = []
    for k, f in model.model_fields.items():
        out.append(prefix + k)
        t = f.annotation
        if isinstance(t, type) and issubclass(t, BaseModel):
            out += _paths(t, prefix + k + ".")
    return out


def test_config_guide_covers_every_field_with_a_meaning():
    """설정 JSON 의 모든 칸(안쪽 칸 포함)이 안내에 있고, 뜻이 비어 있지 않다 — 새 칸을 만들면서 설명을 빠뜨리면 실패."""
    import api.main as m
    from engine.experiment import ExperimentConfig
    g = TestClient(m.app).get("/api/config-guide").json()["fields"]
    assert [f["path"] for f in g] == _paths(ExperimentConfig)
    empty = [f["path"] for f in g if not f["desc"].strip()]
    assert not empty, f"뜻이 없는 칸: {empty}"
    kinds = next(f for f in g if f["path"] == "target.kind")
    assert kinds["choices"] == ["return", "logreturn", "direction", "event", "volatility"] and kinds["default"] == "return"


def test_result_logs_zip_collects_this_run_only(tmp_path, monkeypatch):
    """로그 묶음: 결과·작업 설정·작업 출력·개발 로그(이 실행 태그 줄만)·실행 기록(이 run_id 줄만)·실행 시간대 접속 기록."""
    import api.jobs as jobs
    import api.main as m
    import engine.logs as lg
    res_dir = tmp_path / "results"
    (res_dir / "trading").mkdir(parents=True)
    jdir, cdir, ldir = tmp_path / "jobs", tmp_path / "jobs" / "configs", tmp_path / "logs"
    cdir.mkdir(parents=True); ldir.mkdir()
    monkeypatch.setattr(m, "RESULTS", res_dir)
    monkeypatch.setattr(jobs, "JOBS_DIR", jdir)
    monkeypatch.setattr(jobs, "CONFIG_DIR", cdir)
    monkeypatch.setattr(lg, "LOG_DIR", ldir)
    s_utc, f_utc = "2026-11-20T01:00:00+00:00", "2026-11-20T01:30:00+00:00"
    loc = lambda u: datetime.fromisoformat(u).astimezone().replace(tzinfo=None)
    t0, t1 = loc(s_utc), loc(f_utc)
    r = {"config": {"model": "ridge"}, "run_record": {"run_id": "RUN1", "job_id": "JOB1", "started_utc": s_utc, "finished_utc": f_utc}}
    (res_dir / "trading" / "x_ridge.json").write_text(json.dumps(r), encoding="utf-8")
    (cdir / "JOB1.json").write_text('{"model": "ridge"}', encoding="utf-8")
    (jdir / "JOB1.log").write_text("출력 전체\n", encoding="utf-8")
    day = t0.strftime("%Y%m%d")
    ts = t0.strftime("%Y-%m-%d %H:%M:%S")
    (ldir / f"tsp_{day}.log").write_text(
        f"{ts}.000 INFO    [run=JOB1] tsp.engine | 이 실행\nTraceback 이어지는 줄\n{ts}.000 INFO    [run=OTHER] tsp.engine | 다른 실행\n", encoding="utf-8")
    (ldir / f"access_{day}.log").write_text(
        f"{ts}.000 | 1.2.3.4 | 노트북 직접 | GET /api/x | 200\n2001-01-01 00:00:00.000 | 밖 시간\n", encoding="utf-8")
    (res_dir / "trading" / "run_log.jsonl").write_text('{"run_id": "RUN1", "event": "start"}\n{"run_id": "RUN9"}\n', encoding="utf-8")
    resp = TestClient(m.app).get("/api/results/x_ridge/logs.zip")
    assert resp.status_code == 200 and resp.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    assert set(z.namelist()) == {"result.json", "job_config.json", "job_output.log", "dev_log_this_run.log", "access_during_run.log", "run_log.jsonl", "README.txt"}
    dev = z.read("dev_log_this_run.log").decode()
    assert "이 실행" in dev and "Traceback 이어지는 줄" in dev and "다른 실행" not in dev
    assert "GET /api/x" in z.read("access_during_run.log").decode() and "밖 시간" not in z.read("access_during_run.log").decode()
    assert z.read("run_log.jsonl").decode().strip() == '{"run_id": "RUN1", "event": "start"}'
    assert TestClient(m.app).get("/api/results/..%2Fx/logs.zip").status_code == 404
    assert t1 >= t0


def test_result_keeps_every_fold_period(tmp_path):
    """결과의 periods.folds 에 모든 폴드의 학습·검증·테스트 날짜와 개수가 남는다(첫·마지막 폴드와 같다)."""
    from engine.experiment import ExperimentConfig, run_experiment
    from tests.test_splitplan import _data
    dv, reg = _data(tmp_path, late=0)
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False,
                           validation=dict(mode="walkforward", train_len=200, test_len=100))
    per = run_experiment(cfg, dv, reg)["periods"]
    assert len(per["folds"]) == per["n_folds_logged"] > 2
    assert per["folds"][0] == per["first_fold"] and per["folds"][-1] == per["last_fold"]


def test_result_package_has_before_during_after_and_report(tmp_path, monkeypatch):
    """결과 꾸러미(2026-11-21): 실제로 돌린 결과 하나로 — 사전(설정 전체와 뜻·모델 파라미터·입력 자료와 열 설명·검증 계획)·
    사중(실행 기록·폴드별 기간)·사후(결과·지표와 뜻·일별 결과) + 보고서. 입력 자료는 결과의 데이터 버전으로 다시 만들어 입력 목록과 같다."""
    import pandas as pd
    import api.jobs as jobs
    import api.main as m
    import engine.logs as lg
    from engine.experiment import ExperimentConfig, run_experiment, save_result
    from engine.report_pack import build_package, config_fields
    from tests.test_splitplan import _data
    dv, reg = _data(tmp_path, late=0)
    cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False,
                           validation=dict(mode="walkforward", train_len=200, test_len=100))
    res = run_experiment(cfg, dv, reg)
    p = save_result(res, tmp_path / "results" / "trading")
    data = build_package(p.stem, p, data_root=dv.root, registry=reg, log_dir=tmp_path / "logs", jobs_dir=tmp_path / "j", config_dir=tmp_path / "c")
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    need = {"README.txt", "1_사전_입력/설정_전체와_뜻.csv", "1_사전_입력/설정.json", "1_사전_입력/모델_파라미터와_뜻.csv", "1_사전_입력/변수_사전.csv",
            "1_사전_입력/입력_자료.csv", "1_사전_입력/입력_자료_열설명.csv", "1_사전_입력/검증_계획.json", "1_사전_입력/데이터_버전.json",
            "2_사중_진행/실행_기록.json", "2_사중_진행/폴드별_기간.csv", "2_사중_진행/dev_log_this_run.log",
            "3_사후_결과/result.json", "3_사후_결과/지표와_뜻.csv", "3_사후_결과/일별_결과.csv", "3_사후_결과/일별_결과_열설명.csv"}
    assert need <= names, need - names
    assert ("보고서.docx" in names) or ("보고서.html" in names)
    r = json.loads(z.read("3_사후_결과/result.json"))
    # 설정: 모든 칸이 한 줄씩, 뜻이 채워져 있다
    c = pd.read_csv(io.BytesIO(z.read("1_사전_입력/설정_전체와_뜻.csv")), encoding="utf-8-sig")
    assert list(c["칸"]) == [f["path"] for f in config_fields() if not f["group"]] and c["뜻"].notna().all()
    # 입력 자료: 결과의 입력 목록과 같은 열, 열 설명이 모든 열에
    x = pd.read_csv(io.BytesIO(z.read("1_사전_입력/입력_자료.csv")), encoding="utf-8-sig")
    assert [col for col in x.columns if col in r["feature_names"]] == r["feature_names"] and "정답_target" in x.columns
    cd = pd.read_csv(io.BytesIO(z.read("1_사전_입력/입력_자료_열설명.csv")), encoding="utf-8-sig")
    assert set(x.columns) == set(cd["열"])
    assert "같습니다" in json.loads(z.read("1_사전_입력/데이터_버전.json"))["입력 자료 확인"]
    # 폴드: 모든 폴드, 결과의 기록과 같다
    f = pd.read_csv(io.BytesIO(z.read("2_사중_진행/폴드별_기간.csv")), encoding="utf-8-sig")
    assert len(f) == r["n_folds"] == len(r["periods"]["folds"]) and f["테스트 시작"].iloc[0] == r["periods"]["folds"][0]["test"][0]
    # 지표: 결과의 값 그대로, 뜻이 비지 않음
    mt = pd.read_csv(io.BytesIO(z.read("3_사후_결과/지표와_뜻.csv")), encoding="utf-8-sig")
    sh = mt[(mt["구분"] == "매매 성과") & (mt["지표"] == "sharpe")]["값"].iloc[0]
    assert abs(float(sh) - r["performance"]["sharpe"]) < 1e-9
    assert mt[mt["구분"].isin(["예측 지표", "매매 성과"])]["뜻"].fillna("").str.len().gt(0).all()
    # 일별 결과: parquet 와 같은 행 수
    d = pd.read_csv(io.BytesIO(z.read("3_사후_결과/일별_결과.csv")), encoding="utf-8-sig")
    assert len(d) == len(pd.read_parquet(p.with_name(p.stem + "_daily.parquet")))
    # 화면 경로
    res_dir = tmp_path / "results"
    monkeypatch.setattr(m, "RESULTS", res_dir); monkeypatch.setattr(m, "DATA", dv.root)
    monkeypatch.setattr(m, "_registry", lambda: (reg, []))
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "j"); monkeypatch.setattr(jobs, "CONFIG_DIR", tmp_path / "c"); monkeypatch.setattr(lg, "LOG_DIR", tmp_path / "logs")
    resp = TestClient(m.app).get(f"/api/results/{p.stem}/package.zip")
    assert resp.status_code == 200 and resp.headers["content-type"] == "application/zip"
    assert TestClient(m.app).get("/api/results/..%2Fx/package.zip").status_code == 404


def test_report_falls_back_to_html_without_python_docx(monkeypatch):
    """python-docx 가 없으면 Word 대신 HTML 보고서(같은 내용)를 만들고 README 알림에 설치 방법을 적는다."""
    import builtins
    import engine.report_pack as rp
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda n, *a, **k: (_ for _ in ()).throw(ImportError(n)) if n.startswith("docx") else real(n, *a, **k))
    import pandas as pd
    ctx = dict(steps_all=[], var_sum=pd.DataFrame(), expl=None, stem="s", r={}, cfg={}, rr={}, mode="trading", model_name="Ridge", model_doc="", var_rows=[], data_check="", ds=None, plan=None,
               linfo={}, missing=[], notes=[])
    import pandas as pd
    ctx.update(cfg_df=pd.DataFrame(columns=["칸", "이 결과의 값", "기본값", "바꿈", "뜻"]), par_df=pd.DataFrame(columns=["파라미터", "실제로 쓰인 값", "기본값", "뜻"]),
               fold_df=pd.DataFrame(), met_df=pd.DataFrame(columns=["구분", "지표", "값", "뜻"]))
    try:
        rp._report_docx(ctx)
        raise AssertionError("ImportError 가 나야 함")
    except ImportError:
        pass
    assert "<h1>결과 보고서" in rp._report_html(ctx)


def test_package_traces_data_steps_and_reproduces_predictions(tmp_path):
    """데이터 변화 단계(2026-11-21): 변수별 단계표의 마지막 열 = 실제 모델 입력(시차·변환·채움 조합마다), 폴드별 눈금 = 그 폴드 학습 행의
    평균·표준편차, 계산 따라가기로 다시 계산한 폴드 예측 = 결과 파일의 예측."""
    import numpy as np
    import pandas as pd
    from engine.experiment import ExperimentConfig, run_experiment, save_result
    from engine.report_pack import build_package
    from tests.test_splitplan import _data
    dv, reg = _data(tmp_path, late=200)
    cases = [dict(frequency="D", fill="ffill", variable_transforms={"S_signal": "pct"}, lag_override={"S_signal": 3}),
             dict(frequency="W", fill="ffill", variable_transforms={"S_signal": "pct"}, validation=dict(mode="walkforward", train_len=60, test_len=20)),
             dict(frequency="D", fill="interp", lag_override={"S_signal": 2})]
    for i, kw in enumerate(cases):
        kw.setdefault("validation", dict(mode="walkforward", train_len=200, test_len=100))
        cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=(i == 0), **kw)
        res = run_experiment(cfg, dv, reg)
        p = save_result(res, tmp_path / f"r{i}" / "trading")
        z = zipfile.ZipFile(io.BytesIO(build_package(p.stem, p, data_root=dv.root, registry=reg, log_dir=tmp_path / "l", jobs_dir=tmp_path / "j", config_dir=tmp_path / "c")))
        rd = lambda n: pd.read_csv(io.BytesIO(z.read(n)), encoding="utf-8-sig")  # noqa: E731
        s = rd("1_사전_입력/데이터_변화_단계/요약.csv")
        assert list(s["실제 입력과 일치"]) == ["일치"], (kw, s.to_dict())
        col = s["모델 입력 열"].iloc[0]
        t = rd(f"1_사전_입력/데이터_변화_단계/{col}.csv")
        x = rd("1_사전_입력/입력_자료.csv")
        assert np.allclose(t.iloc[:, -3].values.astype(float), x[col].values.astype(float), equal_nan=True)   # 끝에서 셋째 = ② 변환 후 = 모델 입력
        assert (t["빈칸 이유"].fillna("") != "").sum() == x[col].isna().sum()
        if kw["fill"] == "interp":
            assert "보간에 쓴 다음 기준일(미래)" in t and s["미래 정보 포함"].iloc[0] == "예(채움 방식)"
        # 폴드별 눈금 = 그 폴드 학습 행(빈칸 행 뺀 것)의 평균·표준편차
        r = json.loads(z.read("3_사후_결과/result.json"))
        f0 = r["periods"]["folds"][0]
        sc = rd("2_사중_진행/폴드별_눈금.csv")
        sc0 = sc[sc["폴드"] == 1].set_index("입력 열")
        xi = x.set_index("결정일")
        rows = xi.loc[f0["train"][0]:f0["train"][1], r["feature_names"] + ["정답_target"]].dropna()
        assert len(rows) == f0["n_train"]
        assert np.allclose(sc0.loc[r["feature_names"], "평균"].values, rows[r["feature_names"]].mean().values, rtol=1e-9)
        assert "n_train_dropped_blank" in f0 and "n_test_not_predicted" in f0
        # 계산 따라가기: 확인 전부 통과, 다시 계산한 예측 = 결과 예측
        chk = rd("2_사중_진행/계산_따라가기/확인.csv")
        assert (chk["통과"] == "예").all()
        readme = z.read("README.txt").decode()
        assert "결과 파일의 예측과 최대" not in readme, readme
        assert "1_사전_입력/데이터_흐름_전체.csv" in z.namelist()


def test_window_fold_record_has_scale_and_blank_counts(tmp_path):
    """창 모델 폴드 기록: 눈금(학습 폴드 전체 행으로 맞춘 기준)·평균으로 채운 빈칸 수·창 수. 계산 따라가기는 넣지 않고 알림."""
    from engine.experiment import ExperimentConfig, run_experiment, save_result
    from engine.report_pack import build_package
    from tests.test_splitplan import _data
    dv, reg = _data(tmp_path, late=200)
    cfg = ExperimentConfig(ticker="TEST", model="lstm", variable_ids=["S_signal"], use_technical=False, model_params={"epochs": 2},
                           validation=dict(mode="walkforward", train_len=200, test_len=100))
    res = run_experiment(cfg, dv, reg)
    f0 = res["periods"]["folds"][0]
    assert f0["scale"]["kind"] == "standard" and f0["scale"]["features"] == res["feature_names"]
    assert f0["n_blank_filled"] > 0 and f0["n_windows"] > 0
    p = save_result(res, tmp_path / "r" / "trading")
    z = zipfile.ZipFile(io.BytesIO(build_package(p.stem, p, data_root=dv.root, registry=reg, log_dir=tmp_path / "l", jobs_dir=tmp_path / "j", config_dir=tmp_path / "c")))
    assert "2_사중_진행/폴드별_눈금.csv" in z.namelist()
    assert not any("계산_따라가기" in n for n in z.namelist()) and "창 모델" in z.read("README.txt").decode()


def test_package_saved_after_job_is_served_frozen_and_archived_with_result(tmp_path, monkeypatch):
    """실험 직후 저장(2026-11-21): 작업이 끝나면 결과 옆에 <stem>_package.zip 을 저장하고, 내려받기는 그 파일을 그대로(바이트까지 같게) 준다.
    저장본이 없는 결과는 누를 때 만들고 README 에 그렇게 적는다. 결과를 지우면(history 로 옮기면) 꾸러미도 함께 옮겨진다."""
    import api.jobs as jobs
    import api.main as m
    import engine.logs as lg
    from engine import runlog
    from engine.experiment import ExperimentConfig, run_experiment, save_result
    from engine.report_pack import MADE_AT_RUN, MADE_ON_DOWNLOAD
    from tests.test_splitplan import _data
    dv, reg = _data(tmp_path, late=0)
    res_dir = tmp_path / "results"
    monkeypatch.setattr(m, "RESULTS", res_dir); monkeypatch.setattr(m, "DATA", dv.root)
    monkeypatch.setattr(m, "_registry", lambda: (reg, []))
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "j"); monkeypatch.setattr(jobs, "CONFIG_DIR", tmp_path / "c"); monkeypatch.setattr(lg, "LOG_DIR", tmp_path / "logs")
    assert m.jobs.after_done is m._save_package_after_job
    mk = lambda seed: save_result(run_experiment(ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False, seed=seed,  # noqa: E731
                                                                  validation=dict(mode="walkforward", train_len=200, test_len=100)), dv, reg), res_dir / "trading")
    p = mk(1)
    pk = m._save_package_after_job({"result_path": str(p).replace("/", "\\\\")})   # 윈도 cli 출력처럼 역슬래시 두 겹이어도
    assert pk == p.with_name(p.stem + "_package.zip") and pk.exists()
    c = TestClient(m.app)
    assert c.get(f"/api/results/{p.stem}").json()["package_saved"] is True
    got = c.get(f"/api/results/{p.stem}/package.zip").content
    assert got == pk.read_bytes()                       # 저장본 그대로(다시 만들지 않음)
    assert MADE_AT_RUN in zipfile.ZipFile(io.BytesIO(got)).read("README.txt").decode()
    # 저장본이 없는 결과: 누를 때 만들고 그렇게 적는다
    q = mk(2)
    assert c.get(f"/api/results/{q.stem}").json()["package_saved"] is False
    z2 = zipfile.ZipFile(io.BytesIO(c.get(f"/api/results/{q.stem}/package.zip").content))
    assert MADE_ON_DOWNLOAD in z2.read("README.txt").decode()
    assert not q.with_name(q.stem + "_package.zip").exists()   # 내려받기만으로는 저장하지 않는다
    # 결과를 history 로 옮기면 꾸러미도 함께 — 같은 이름의 새 결과에 옛 꾸러미가 붙지 않게
    moved = runlog.archive_existing(p.parent, p.stem, tag="deleted")
    assert any(x.endswith("_package.zip") for x in moved) and not pk.exists()
    assert m._save_package_after_job({"result_path": str(tmp_path / "nope.json")}) is None
