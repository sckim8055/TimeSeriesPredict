"""로그 날짜별 파일·다시 열기·접속 기록(2026-11-17).
사용자 화면에서 본 문제: 서버를 켠 뒤 드라이브가 잠깐 끊기자 로그 쓰기가 OSError 22 로 계속 실패하고 긴 예외 추적이 콘솔을 덮었다.
서버를 며칠 켜 두면 날짜가 바뀌어도 첫날 파일에 계속 썼다."""
from __future__ import annotations

import logging
from datetime import datetime

from engine.logs import DailyFileHandler


def _logger(h, name):
    lg = logging.getLogger(name)
    lg.handlers[:] = [h]
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(message)s"))
    return lg


def test_new_file_after_midnight(tmp_path):
    now = {"t": datetime(2026, 11, 16, 23, 59, 59)}
    h = DailyFileHandler(tmp_path, "tsp", clock=lambda: now["t"])
    lg = _logger(h, "t_daily_a")
    lg.info("밤 11시 59분")
    now["t"] = datetime(2026, 11, 17, 0, 0, 1)
    lg.info("자정 지나 첫 기록")
    lg.info("같은 날 두 번째")
    assert (tmp_path / "tsp_20261116.log").read_text(encoding="utf-8") == "밤 11시 59분\n"
    assert (tmp_path / "tsp_20261117.log").read_text(encoding="utf-8") == "자정 지나 첫 기록\n같은 날 두 번째\n"
    h.close()


def test_reopens_after_os_error(tmp_path, capsys):
    """파일 연결이 망가져 쓰기·비우기가 OSError(22)로 실패해도, 닫고 다시 열어 그 줄을 남긴다."""
    h = DailyFileHandler(tmp_path, "tsp", clock=lambda: datetime(2026, 11, 17, 9, 0))
    lg = _logger(h, "t_daily_b")
    lg.info("첫 줄")

    class Broken:
        def write(self, s): raise OSError(22, "Invalid argument")
        def flush(self): raise OSError(22, "Invalid argument")
        def close(self): raise OSError(22, "Invalid argument")
    h._f = Broken()
    lg.info("끊긴 뒤 줄")
    assert (tmp_path / "tsp_20261117.log").read_text(encoding="utf-8") == "첫 줄\n끊긴 뒤 줄\n"
    assert h.n_failed == 0 and capsys.readouterr().err == ""
    h.close()


def test_persistent_failure_warns_once_not_traceback(tmp_path, capsys):
    """다시 열어도 계속 실패하면 줄은 잃지만, 콘솔에는 10분에 한 번 짧은 알림만(긴 예외 추적 없음)."""
    bad = tmp_path / "file_not_dir"
    bad.write_text("x")                                     # 폴더 자리에 파일 → 열기 실패
    h = DailyFileHandler(bad, "tsp", clock=lambda: datetime(2026, 11, 17, 9, 0))
    lg = _logger(h, "t_daily_c")
    for i in range(5):
        lg.info("줄 %d", i)
    err = capsys.readouterr().err
    assert h.n_failed == 5 and err.count("[로그]") == 1 and "Traceback" not in err
    h.close()


def test_access_log_records_origin(tmp_path, monkeypatch):
    """접속 기록: 보낸 주소·노트북 직접/외부(Funnel)·방법 주소·결과·브라우저 정보. 암호 머리글 값은 남기지 않는다."""
    import api.main as m
    from fastapi.testclient import TestClient
    h = DailyFileHandler(tmp_path, "access", clock=lambda: datetime(2026, 11, 17, 9, 0))
    h.setFormatter(logging.Formatter("%(message)s"))
    old = list(m.acclog.handlers)
    m.acclog.handlers[:] = [h]
    try:
        c = TestClient(m.app)
        c.get("/api/meta", headers={"user-agent": "TestBrowser/1.0"})
        c.get("/api/meta?x=1", headers={"tailscale-funnel-request": "?1", "user-agent": "scanner", "x-tsp-password": "SECRETVALUE"})
    finally:
        m.acclog.handlers[:] = old
        h.close()
    lines = (tmp_path / "access_20261117.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert " | 노트북 직접 | GET /api/meta | 200 | " in lines[0] and lines[0].endswith("| TestBrowser/1.0")
    assert " | 외부(Funnel) | GET /api/meta?x=1 | 200 | " in lines[1] and "SECRETVALUE" not in lines[1]
    assert lines[0].split(" | ")[0]                          # 보낸 주소 칸이 있다(시험 클라이언트는 'testclient')


def test_logs_api_kind_and_date_checked():
    import api.main as m
    from fastapi.testclient import TestClient
    c = TestClient(m.app)
    assert c.get("/api/logs?kind=access").status_code == 200
    assert c.get("/api/logs?kind=../x").status_code == 400
    assert c.get("/api/logs?date=..%2F..%2Fx").status_code == 400
