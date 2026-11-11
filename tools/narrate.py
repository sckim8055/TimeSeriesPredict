"""코드 따라가기 설명 음성 만들기 — 단계마다 mp3 한 개와 자막 줄 시각(언제 어느 줄을 읽는지).

노트북(Windows)에서:
  narrate.bat (더블클릭)            음성 목록·남은 크레딧을 보여 주고, 음성 ID 가 있으면 화면 이름을 물은 뒤 확인을 받고 만든다
  narrate.bat ridge                 ElevenLabs 로 ridge 화면의 모든 단계 음성을 만든다(이미 만든 단계는 건너뜀)
  narrate.bat ridge lgbm            여러 화면
  narrate.bat --list-voices         내 ElevenLabs 음성 목록(이름 · ID) — 쓸 음성의 ID 를 .env 에 넣는다
  narrate.bat ridge --dry-run       만들 글자 수와 남은 크레딧만 보여 주고 끝
처음 한 번: 프로젝트 폴더의 .env 에 두 줄을 넣는다(채팅·문서에 키를 붙이지 말 것)
  ELEVENLABS_API_KEY=...
  ELEVENLABS_VOICE_ID=...           (또는 --voice "음성 이름")

결과: docs/guide_build/walk/audio/<화면>/s000.mp3 … + s000.json(글자별 시각 원본) + manifest.json(대본 지문·음성·모델·설정).
마지막에 docs/guide_build/walk/<화면>.html 에 음성을 연결한다 → 화면의 '음성 설명' 이 녹음을 재생하며 자막을 바꾼다.
영상(mp4)에 넣기는 python -m tools.codewalk.render_mp4 docs/guide_build/walk/<화면>.html (클라우드 세션에서 한다).

대본: 화면을 만들 때 단계마다 정해진 '읽을 문장'(speech, tools/codewalk/speech.py — 숫자를 한글로 풀어 씀)을 그대로 보낸다.
대본이 바뀐 단계만 다시 만든다(지문 = speech 의 sha1). 같은 음성·모델·설정·seed 면 ElevenLabs 도 대체로 같은 소리를 낸다(완전히 같다는 보장은 없음).
--engine offline 은 클라우드 시험용(무료 오프라인 음성 sherpa-onnx, 품질 낮음) — 파이프라인 시험에만 쓴다.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WALK = ROOT / "docs" / "guide_build" / "walk"
API = "https://api.elevenlabs.io"


def load_env() -> dict:
    env = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return {**env, **{k: v for k, v in os.environ.items() if k.startswith("ELEVENLABS_")}}


def load_script(html: Path) -> dict:
    h = html.read_text(encoding="utf-8")
    i = h.index("const S = ") + len("const S = ")
    j = h.index(";\nconst $ =")
    return json.loads(h[i:j])


def step_texts(S: dict) -> list[list[str]]:
    from tools.codewalk.speech import prepare
    return [st["speech"] if st.get("speech") else prepare(st)["speech"] for st in S["steps"]]


# ---------------------------------------------------------------- ElevenLabs
def _req(method: str, path: str, key: str, body: dict | None = None, timeout: int = 180):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method, headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:600]
            if e.code == 401:
                sys.exit(f"ElevenLabs 가 키를 거부했습니다(401). .env 의 ELEVENLABS_API_KEY 를 확인하세요. 응답: {msg}")
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                wait = 5 * (attempt + 1)
                print(f"  잠시 뒤 다시 시도 ({e.code}, {wait}초)", flush=True); time.sleep(wait); continue
            raise RuntimeError(f"ElevenLabs {e.code}: {msg}") from None
        except urllib.error.URLError as e:
            if attempt < 5:
                print(f"  연결 실패, 다시 시도 ({e.reason})", flush=True); time.sleep(5 * (attempt + 1)); continue
            raise
    raise RuntimeError("ElevenLabs 요청 실패")


def list_voices(key: str) -> list[dict]:
    """내 음성 목록. show_legacy=true — 예전 기본 음성(예: Daniel onwK4e9ZLuTAKqWW03F9)은 이게 없으면 목록에서 빠진다(ElevenLabs 기본값 false)."""
    return _req("GET", "/v1/voices?show_legacy=true", key)["voices"]


def resolve_voice(key: str, voice: str) -> tuple[str, str]:
    """음성 ID 면 그 음성을 직접 조회(목록에 없어도 쓸 수 있는 음성이 있다), 아니면 이름으로 목록에서 찾는다."""
    if len(voice) == 20 and voice.isalnum():
        try:
            v = _req("GET", f"/v1/voices/{voice}", key)
            return v["voice_id"], v.get("name") or voice
        except RuntimeError as e:
            sys.exit(f"음성 ID '{voice}' 를 쓸 수 없습니다: {e}")
    for v in list_voices(key):
        if voice in (v["voice_id"], v["name"]) or v["name"].split(" - ")[0] == voice:
            return v["voice_id"], v["name"]
    sys.exit(f"음성 '{voice}' 를 찾지 못했습니다. narrate.bat 을 더블클릭해 목록의 이름·ID 를 확인하세요.")


def remaining_credits(key: str):
    s = _req("GET", "/v1/user/subscription", key)
    return s.get("character_limit", 0) - s.get("character_count", 0), s


def tts_eleven(key: str, voice_id: str, text: str, *, model: str, settings: dict, seed: int, prev: str | None, nxt: str | None) -> tuple[bytes, dict]:
    body = {"text": text, "model_id": model, "voice_settings": settings, "seed": seed}
    if prev:
        body["previous_text"] = prev
    if nxt:
        body["next_text"] = nxt
    try:
        r = _req("POST", f"/v1/text-to-speech/{voice_id}/with-timestamps?output_format=mp3_44100_128", key, body)
    except RuntimeError as e:
        if ("previous_text" in str(e) or "next_text" in str(e)) and (prev or nxt):   # 앞뒤 문맥을 받지 않는 모델
            body.pop("previous_text", None); body.pop("next_text", None)
            r = _req("POST", f"/v1/text-to-speech/{voice_id}/with-timestamps?output_format=mp3_44100_128", key, body)
        else:
            raise
    al = r.get("alignment") or r.get("normalized_alignment")
    return base64.b64decode(r["audio_base64"]), al


# ---------------------------------------------------------------- 오프라인(클라우드 시험용)
_OFF = {}


def tts_offline(model_dir: str, segs: list[str], speed: float) -> tuple[bytes, dict]:
    """sherpa-onnx VITS(한국어 KSS) — 자막 줄마다 따로 합성해 0.25초 쉬고 이어 붙인다. 글자 시각은 줄 안에서 고르게 나눈다(줄 경계는 정확)."""
    import numpy as np
    import sherpa_onnx
    if "tts" not in _OFF:
        d = Path(model_dir)
        onnx = next(d.glob("*.onnx"))
        cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(onnx), tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data")), num_threads=2))
        _OFF["tts"] = sherpa_onnx.OfflineTts(cfg)
    tts = _OFF["tts"]
    parts, chars, st_, en_, t = [], [], [], [], 0.0
    sr = 22050
    for k, s in enumerate(segs):
        a = tts.generate(s, sid=0, speed=speed); sr = a.sample_rate
        x = np.asarray(a.samples, dtype=np.float32); dur = len(x) / sr
        for q, ch in enumerate(s):
            chars.append(ch); st_.append(t + dur * q / len(s)); en_.append(t + dur * (q + 1) / len(s))
        parts.append(x); t += dur
        if k < len(segs) - 1:
            gap = np.zeros(int(sr * 0.25), np.float32); parts.append(gap); chars.append(" "); st_.append(t); en_.append(t + 0.25); t += 0.25
    x = np.concatenate(parts)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    mp3 = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3", "pipe:1"],
                         input=pcm, capture_output=True, check=True).stdout
    return mp3, {"characters": chars, "character_start_times_seconds": st_, "character_end_times_seconds": en_}


# ---------------------------------------------------------------- 공통
def seg_times(segs: list[str], al: dict) -> tuple[list[list[float]], float, bool]:
    """자막 줄마다 [시작, 끝] 초. 요청 문장 = 줄을 공백 하나로 이은 것 → 글자 위치로 찾는다. 글자가 맞지 않으면 비율로(exact=False)."""
    text = " ".join(segs)
    ch, s0, s1 = al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]
    total = float(max(s1)) if s1 else 0.0
    exact = "".join(ch) == text
    out, o = [], 0
    for s in segs:
        a, b = o, o + len(s) - 1
        if exact:
            out.append([round(float(s0[a]), 3), round(float(s1[b]), 3)])
        else:
            n = max(1, len(text) - 1)
            out.append([round(total * a / n, 3), round(total * b / n, 3)])
        o += len(s) + 1
    return out, round(total, 3), exact


def write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def relink(name: str) -> None:
    """화면 HTML 에 녹음을 연결(대본 지문이 맞는 단계만)."""
    from tools.codewalk.common import attach_speech
    html = WALK / f"{name}.html"
    h = html.read_text(encoding="utf-8")
    i = h.index("const S = ") + len("const S = "); j = h.index(";\nconst $ =")
    S = attach_speech(json.loads(h[i:j]), name)
    js = json.dumps(S, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    tmp = html.with_suffix(".html.tmp"); tmp.write_text(h[:i] + js + h[j:], encoding="utf-8"); os.replace(tmp, html)
    a = S.get("audio") or {}
    mf = WALK / "audio" / name / "manifest.json"
    rec = json.loads(mf.read_text(encoding="utf-8")).get("steps", {}) if mf.exists() else {}
    miss = a.get("missing") or []
    todo = [i for i in miss if str(i) not in rec]                 # 아직 녹음하지 않은 단계
    stale = [i for i in miss if str(i) in rec]                    # 녹음은 있으나 그 뒤 대본이 바뀐(또는 파일이 없는) 단계
    print(f"  화면에 연결: {html.name} · 녹음 {len(a.get('steps', {}))}/{len(S['steps'])} 단계"
          + (f" · 아직 녹음 안 한 단계 {len(todo)}개({_ranges(todo)})" if todo else "")
          + (f" · 녹음 뒤 대본이 바뀌어 다시 녹음할 단계 {len(stale)}개({_ranges(stale)})" if stale else "")
          + (f"\n  → 나머지를 녹음하려면: narrate.bat {name}" if miss else ""))


def _ranges(xs: list[int]) -> str:
    """[1,2,3,5] → '1~3, 5'"""
    out, xs = [], sorted(xs)
    for x in xs:
        if out and x == out[-1][1] + 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return ", ".join(f"{a}~{b}" if b > a else f"{a}" for a, b in out)


def run(name: str, a, env: dict) -> None:
    from tools.codewalk.common import speech_hash
    html = WALK / f"{name}.html"
    if not html.exists():
        sys.exit(f"{html} 가 없습니다")
    S = load_script(html)
    texts = step_texts(S)
    out = WALK / "audio" / name; out.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.json"
    man = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"walk": name, "steps": {}}
    if a.engine == "elevenlabs":
        key = env.get("ELEVENLABS_API_KEY")
        if not key:
            sys.exit(".env 에 ELEVENLABS_API_KEY=... 가 없습니다")
        vid = a.voice or env.get("ELEVENLABS_VOICE_ID")
        if not vid:
            sys.exit(".env 에 ELEVENLABS_VOICE_ID=... 가 없습니다(또는 --voice). narrate.bat --list-voices 로 확인")
        voice_id, voice_name = resolve_voice(key, vid)
        settings = {"stability": a.stability, "similarity_boost": a.similarity, "style": 0.0, "use_speaker_boost": True, "speed": a.speed}
        sig = {"engine": "elevenlabs", "voice_id": voice_id, "voice_name": voice_name, "model": a.model, "settings": settings, "seed": a.seed}
    else:
        sig = {"engine": "offline", "voice_id": "kss_low", "voice_name": "sherpa-onnx vits-mimic3 ko_KO kss_low", "model": Path(a.model_dir).name, "settings": {"speed": a.speed}, "seed": None}
    SIGK = ("engine", "voice_id", "model", "settings", "seed")      # 이 값이 다르면 다른 녹음(음성 이름은 바뀔 수 있어 빼고 봄)
    def step_sig(e: dict) -> dict:                                  # 단계마다 녹음한 설정(예전 기록은 화면 전체 설정)
        return {k: (e.get("sig") or man).get(k) for k in SIGK}
    want = {k: sig[k] for k in SIGK}
    todo = []
    for i, sp in enumerate(texts):
        h = speech_hash(sp)
        e = man["steps"].get(str(i))
        same = e and e.get("hash") == h and step_sig(e) == want and (out / e["file"]).exists()
        if a.force or not same:
            todo.append(i)
    if a.steps:
        lo, _, hi = a.steps.partition("-")
        keep = set(range(int(lo), int(hi or lo) + 1))
        todo = [i for i in todo if i in keep]
    chars = sum(len(" ".join(texts[i])) for i in todo)
    print(f"[{name}] 단계 {len(texts)} · 새로 만들 단계 {len(todo)} · 글자 {chars:,}" + (f" · 음성 {sig['voice_name']} ({sig['voice_id']}) · 모델 {sig['model']}" if sig["engine"] == "elevenlabs" else " · 오프라인 시험 음성"))
    if sig["engine"] == "elevenlabs":
        rem, sub = remaining_credits(key)
        print(f"  남은 크레딧(글자) {rem:,} / {sub.get('character_limit', 0):,} · 요금제 {sub.get('tier')}")
        if a.dry_run:
            return
        if chars > rem and not a.force:
            sys.exit(f"  크레딧이 모자랍니다(필요 {chars:,} > 남음 {rem:,}). 일부만 하려면 --steps 0-5 처럼 고르세요.")
    elif a.dry_run:
        return
    if getattr(a, "confirm", False) and todo and input(f"  단계 {len(todo)}개, 글자 {chars:,} 를 만듭니다(크레딧 사용). 진행할까요? (y/n): ").strip().lower() not in ("y", "yes", "ㅛ"):
        print("  취소했습니다"); return
    old = [i for i, e in man["steps"].items() if step_sig(e) != want]
    if old:
        print(f"  음성·모델·설정이 다른 이전 녹음 {len(old)}단계 — 이번에 고른 단계({len(todo)}개)를 새 설정으로 만듭니다"
              + (". 나머지 이전 녹음은 그대로 남아, 화면 하나에 음성이 섞입니다(모두 바꾸려면 --steps 없이 다시)" if a.steps else ""))
    for e in man["steps"].values():                                  # 이전 단계에 그때 설정을 남겨 둔다(중간에 끊겨도 섞인 것을 알아봄)
        e.setdefault("sig", {k: man.get(k) for k in SIGK})
    man.update(sig)
    for n, i in enumerate(todo, 1):
        sp = texts[i]; text = " ".join(sp)
        if sig["engine"] == "elevenlabs":
            prev = " ".join(texts[i - 1])[-300:] if i > 0 else None
            nxt = " ".join(texts[i + 1])[:300] if i + 1 < len(texts) else None
            mp3, al = tts_eleven(key, sig["voice_id"], text, model=a.model, settings=sig["settings"], seed=a.seed, prev=prev, nxt=nxt)
        else:
            mp3, al = tts_offline(a.model_dir, sp, a.speed)
        segs, dur, exact = seg_times(sp, al)
        f = f"s{i:03d}.mp3"
        (out / (f + ".tmp")).write_bytes(mp3); os.replace(out / (f + ".tmp"), out / f)
        write_json(out / f"s{i:03d}.json", {"text": text, "segments": sp, "alignment": al, "exact_alignment": exact})
        man["steps"][str(i)] = {"file": f, "hash": speech_hash(sp), "chars": len(text), "segs": segs, "duration": dur, "exact_alignment": exact,
                                "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "sig": dict(want), "voice_name": sig["voice_name"]}
        write_json(mf, man)
        print(f"  단계 {i + 1}/{len(texts)} · {dur:.1f}초 · 글자 {len(text)}" + ("" if exact else " · 글자 시각이 문장과 달라 비율로 나눔") + f"  ({n}/{len(todo)})", flush=True)
    relink(name)


def main():
    ap = argparse.ArgumentParser(description="코드 따라가기 설명 음성 만들기")
    ap.add_argument("names", nargs="*", help="화면 이름(biva lstm gru arima sarima ridge lgbm vecm rule)")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="이미 만든 단계도 다시 / 크레딧 확인 무시")
    ap.add_argument("--steps", help="이 범위의 단계만(0부터): 예 0-5")
    ap.add_argument("--engine", default="elevenlabs", choices=["elevenlabs", "offline"])
    ap.add_argument("--voice", help="음성 ID 또는 이름(.env 의 ELEVENLABS_VOICE_ID 대신)")
    ap.add_argument("--model", default="eleven_multilingual_v2")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--stability", type=float, default=0.5)
    ap.add_argument("--similarity", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--model-dir", help="--engine offline: sherpa-onnx 모델 폴더")
    a = ap.parse_args()
    env = load_env()
    if a.list_voices:
        key = env.get("ELEVENLABS_API_KEY") or sys.exit(".env 에 ELEVENLABS_API_KEY=... 가 없습니다")
        for v in list_voices(key):
            print(f"{v['name']:<30} {v['voice_id']}  ({v.get('category', '')}{', 예전 기본 음성' if v.get('is_legacy') else ''})")
        return
    if not a.names:          # narrate.bat 을 더블클릭: 음성 목록·크레딧을 보여 주고, 음성 ID 가 있으면 화면 이름을 묻는다
        key = env.get("ELEVENLABS_API_KEY")
        if not key:
            sys.exit(".env 에 ELEVENLABS_API_KEY=... 가 없습니다")
        print("내 ElevenLabs 음성 (이름 · ID · 종류):")
        for v in list_voices(key):
            print(f"  {v['name']:<30} {v['voice_id']}  ({v.get('category', '')}{', 예전 기본 음성' if v.get('is_legacy') else ''})")
        rem, sub = remaining_credits(key)
        print(f"남은 크레딧(글자) {rem:,} / {sub.get('character_limit', 0):,} · 요금제 {sub.get('tier')}")
        if not (a.voice or env.get("ELEVENLABS_VOICE_ID")):
            print("\n위 목록에서 쓸 음성의 ID 를 .env 에 한 줄 더 넣고 다시 실행하세요:  ELEVENLABS_VOICE_ID=<ID>")
            return
        name = input("\n만들 화면 이름 (biva lstm gru arima sarima ridge lgbm vecm rule — 여러 개는 띄어 쓰기, 빈칸 = ridge): ").strip() or "ridge"
        a.names = name.split(); a.confirm = True
    for n in a.names:
        run(n, a, env)


if __name__ == "__main__":
    main()
