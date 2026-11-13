"""코드 따라가기(tools/codewalk) — 실제 실행 기록으로 만든 대본이 원본 코드·손 계산과 맞는지.

1) 짧은 학습(2 epoch)으로 실제 데이터에서 붙잡은 기록 → 대본: 손 계산 검증이 모두 맞고, 단계마다 가리키는 코드 줄이 원본에 그대로 있다.
   (원본 BIVA 소스나 그 호출 줄이 바뀌면 여기서 멈춘다 → 영상이 낡지 않는다)
2) 저장된 기록(docs/guide_build/walk/biva_capture_e500.pkl)이 있으면 그것으로 만든 대본도 같은 검사를 통과한다.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_OK = (ROOT / "data" / "variable_master.xlsx").exists()


def _check_script(sc: dict) -> None:
    assert sc["steps"], "단계 없음"
    bad = [c for c in sc["checks"] if c["ok"] is False or (c["ok"] is None and not c.get("skipped"))]   # 건너뜀(ok=None, skipped)은 실패가 아님
    assert not bad, f"손 계산 불일치: {bad}"
    for i, st in enumerate(sc["steps"], 1):
        cd = st.get("code")
        if not cd:
            continue
        src = sc["sources"][cd["file"]]
        assert 1 <= cd["focus"] <= len(src), f"단계 {i}: 줄 번호 범위 밖"
        assert cd["focus"] in cd["lines"] or cd["lines"], f"단계 {i}: 강조 줄 없음"
        # 대본 안에 넣은 원본 줄 = 지금 원본 파일의 줄
        now = (ROOT / cd["file"]).read_text(encoding="utf-8").split("\n")
        assert now[cd["focus"] - 1] == src[cd["focus"] - 1], f"단계 {i}: {cd['file']}:{cd['focus']} 이 바뀜"
        assert src[cd["focus"] - 1].strip(), f"단계 {i}: 빈 줄을 가리킴"


@pytest.mark.skipif(not DATA_OK, reason="실데이터 없음")
def test_biva_walk_from_short_capture():
    from tools.codewalk.biva_capture import capture
    from tools.codewalk.biva_script import build

    cap = capture(epochs=2)
    assert len(cap["history"]) == 2
    assert cap["batch"]["n_nan"] > 0  # 빈 칸이 있는 배치를 골랐다(BRITS 가 실제로 채우는 장면)
    assert cap["vae_all"]["n"] == cap["vae_all"]["n_windows"] * cap["seq_len"] * cap["args"]["vae_latent_size"]
    sc = build(cap)
    _check_script(sc)
    titles = {s["chapter"] for s in sc["steps"]}
    for ch in ("7. BRITS 앞→뒤", "8. BRITS 뒤→앞", "12. VAE 인코더 (LSTM 4층)", "18. 손실 · 역전파 · Adam", "19. 예측"):
        assert ch in titles


@pytest.mark.skipif(not (ROOT / "docs/guide_build/walk/biva_capture_e500.pkl").exists(), reason="저장된 500 epoch 기록 없음")
def test_biva_walk_from_saved_capture():
    from tools.codewalk.biva_script import build

    cap = pickle.load(open(ROOT / "docs/guide_build/walk/biva_capture_e500.pkl", "rb"))
    _check_script(build(cap))


def test_player_placeholders():
    h = (ROOT / "tools/codewalk/player.html").read_text(encoding="utf-8")
    assert "/*__SCRIPT__*/null" in h and "__TITLE__" in h


# ---------------------------------------------------------------- 2026-11-07: 나머지 모델의 코드 따라가기
WALK = ROOT / "docs/guide_build/walk"
SAVED = [("series_script", "arima"), ("series_script", "sarima"), ("table_script", "ridge"), ("table_script", "lgbm"), ("vecm_script", "vecm"),
         ("rule_script", "rule"), ("rnn_script", "lstm"), ("rnn_script", "gru")]


@pytest.mark.parametrize("script,name", SAVED)
def test_walk_from_saved_capture(script, name):
    """저장된 실행 기록으로 만든 대본: 손 계산 검증이 모두 맞고, 가리키는 코드 줄이 지금 원본과 같다(원본이 바뀌면 여기서 멈춤)."""
    import importlib

    p = WALK / f"{name}_capture.pkl"
    if not p.exists():
        pytest.skip(f"{p.name} 없음")
    cap = pickle.load(open(p, "rb"))
    sc = importlib.import_module(f"tools.codewalk.{script}").build(cap)
    _check_script(sc)
    assert len(sc["checks"]) >= 8


@pytest.mark.skipif(not DATA_OK, reason="실데이터 없음")
def test_table_walk_live_capture_ridge():
    """Ridge: 지금 엔진·데이터로 다시 붙잡은 기록도 검증을 모두 통과(데이터셋 만들기·발표시차·닫힌 해·예측)."""
    from tools.codewalk.table_capture import capture
    from tools.codewalk.table_script import build

    cap = capture("ridge")
    assert cap["n_va"] == 0 and len(cap["all_folds"]) == cap["n_folds"]
    _check_script(build(cap))


@pytest.mark.skipif(not DATA_OK, reason="실데이터 없음")
def test_vecm_walk_live_capture():
    """VECM: Johansen(β·α·Γ)을 Lütkepohl 식으로 다시 계산한 값 = statsmodels, 재귀 예측 = 엔진."""
    from tools.codewalk.vecm_capture import capture
    from tools.codewalk.vecm_script import build

    _check_script(build(capture()))


def test_rnn_capture_replays_earlier_folds():
    """창 모델은 시드를 실험 시작에 한 번만 고정 → 마지막 폴드를 결과 파일과 같게 기록하려면 앞 폴드를 먼저 돌려야 한다(rnn_capture replay)."""
    import inspect

    from tools.codewalk import rnn_capture

    src = inspect.getsource(rnn_capture.capture)
    assert "replay" in src and "ne[:-1]" in src


def test_match_result_stops_when_result_files_change(monkeypatch):
    """대본이 결과 파일 수·차이를 말하므로, 결과 폴더의 파일이 대본을 쓸 때와 달라지면 조용히 다른 문장을 만들지 않고 멈춘다."""
    from tools.codewalk import common
    key = ("forecast", "f11_arima_k1")
    monkeypatch.setitem(common.RESULT_FILES, key, ["f11_arima_k1_v1107_29387d348071_daily.parquet"])
    if common.result_files(*key) == common.RESULT_FILES[key]:
        pytest.skip("결과 폴더 구성이 다름")
    with pytest.raises(RuntimeError, match="결과 파일 목록"):
        common.match_result(*key, {"2023-12-01": 0.0})


def test_trace_realigns_shifted_lines_by_code_text(tmp_path):
    """기록 뒤에 위쪽 코드가 줄어 줄 번호가 밀려도, 기록된 줄 글자로 지금 줄을 찾는다. 지워진 코드는 stale 표시(2026-11-12)."""
    from tools.codewalk.common import _realign
    from tools.codewalk.trace import Event
    lines = (ROOT / "models/biva_src/BIVA_RITS.py").read_text(encoding="utf-8").split("\n")
    n = next(i + 1 for i, s in enumerate(lines) if "def forward(self, data, direct):" in s)
    moved = Event(seq=0, file="/x/tsp/models/biva_src/BIVA_RITS.py", line=n + 20, func="forward", code=lines[n - 1], depth=1)
    gone = Event(seq=1, file="/x/tsp/models/biva_src/BIVA_RITS.py", line=n, func="delta_calc", code="        prev_missing = masks[:, :-1, :] == 0", depth=2)
    a, b = _realign([moved, gone])
    assert a.line == n and b.meta.get("stale") is True and b.line == n
