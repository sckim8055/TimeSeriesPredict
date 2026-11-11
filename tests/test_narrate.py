"""narrate.py 의 '새로 만들 단계' 고르기 — 음성을 바꾸면 이전 음성 단계가 남지 않는지(중간에 끊겨도). 네트워크 없이 가짜 TTS 로."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import tools.narrate as N

ROOT = Path(__file__).resolve().parents[1]


def _args(**kw):
    a = dict(engine="elevenlabs", voice=None, stability=0.5, similarity=0.75, speed=1.0, model="eleven_multilingual_v2", seed=2025,
             steps=None, force=False, dry_run=False, confirm=False, model_dir="")
    a.update(kw)
    return SimpleNamespace(**a)


def _setup(tmp_path, monkeypatch):
    walk = tmp_path / "walk"; walk.mkdir()
    shutil.copy(ROOT / "docs" / "guide_build" / "walk" / "vecm.html", walk / "t.html")
    monkeypatch.setattr(N, "WALK", walk)
    monkeypatch.setattr(N, "relink", lambda name: None)
    monkeypatch.setattr(N, "remaining_credits", lambda key: (10 ** 7, {}))
    monkeypatch.setattr(N, "resolve_voice", lambda key, v: (v, f"name-{v}"))
    made = []

    def fake_tts(key, voice_id, text, **kw):
        made.append(voice_id)
        n = len(text)
        return b"ID3", {"characters": list(text), "character_start_times_seconds": [k * 0.05 for k in range(n)],
                        "character_end_times_seconds": [k * 0.05 + 0.05 for k in range(n)]}
    monkeypatch.setattr(N, "tts_eleven", fake_tts)
    return walk, made


def test_voice_change_rerecords_every_step(tmp_path, monkeypatch):
    walk, made = _setup(tmp_path, monkeypatch)
    env = {"ELEVENLABS_API_KEY": "x", "ELEVENLABS_VOICE_ID": "A" * 20}
    N.run("t", _args(), env)
    n = len(made); assert n > 3 and set(made) == {"A" * 20}
    made.clear(); N.run("t", _args(), env)
    assert made == []                                                     # 같은 음성·대본이면 아무것도 안 함
    env["ELEVENLABS_VOICE_ID"] = "B" * 20
    N.run("t", _args(steps="0"), env)                                     # 새 음성으로 0단계만(시험 듣기)
    assert made == ["B" * 20]
    man = json.loads((walk / "audio" / "t" / "manifest.json").read_text(encoding="utf-8"))
    assert man["steps"]["0"]["sig"]["voice_id"] == "B" * 20 and man["steps"]["1"]["sig"]["voice_id"] == "A" * 20
    made.clear(); N.run("t", _args(), env)                                # 나머지는 이전 음성이므로 모두 다시
    assert len(made) == n - 1 and set(made) == {"B" * 20}
    man = json.loads((walk / "audio" / "t" / "manifest.json").read_text(encoding="utf-8"))
    assert {e["sig"]["voice_id"] for e in man["steps"].values()} == {"B" * 20}


def test_old_manifest_without_step_sig(tmp_path, monkeypatch):
    """예전 기록(단계별 sig 없음)은 화면 전체 설정을 그 단계의 설정으로 본다."""
    walk, made = _setup(tmp_path, monkeypatch)
    env = {"ELEVENLABS_API_KEY": "x", "ELEVENLABS_VOICE_ID": "A" * 20}
    N.run("t", _args(), env)
    mf = walk / "audio" / "t" / "manifest.json"
    man = json.loads(mf.read_text(encoding="utf-8"))
    for e in man["steps"].values():
        e.pop("sig", None)
    mf.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    made.clear(); N.run("t", _args(), env)
    assert made == []
    env["ELEVENLABS_VOICE_ID"] = "B" * 20
    N.run("t", _args(), env)
    assert len(made) == len(man["steps"])


def test_ranges_for_messages():
    assert N._ranges([1, 2, 3, 5, 7, 8]) == "1~3, 5, 7~8" and N._ranges([4]) == "4" and N._ranges([]) == ""
