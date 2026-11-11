"""코드 따라가기 대본 공용 도구: 텐서 → 화면용 숫자, 실행 기록(Event) 찾기, 원본 줄 번호 찾기, 플레이어 HTML 만들기."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def num(v, sig: int = 4):
    """화면용 숫자: NaN → None, 유효숫자 sig 자리."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(v) or math.isinf(v):
        return None
    if v == 0:
        return 0.0
    return float(f"{v:.{sig}g}")


def arr(x, sig: int = 4):
    """텐서/배열 → 중첩 리스트(NaN → None)."""
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=float)
    if a.ndim == 0:
        return num(a, sig)
    return [arr(r, sig) for r in a] if a.ndim > 1 else [num(v, sig) for v in a]


def f(v, d: int = 4) -> str:
    """설명 문장 속 숫자."""
    v = float(v)
    if math.isnan(v):
        return "NaN"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f"{v:.3e}"
    return f"{v:.{d}f}".rstrip("0").rstrip(".") if d else f"{v:.0f}"


class Trace:
    """실행 기록 찾기. 기록은 tools/codewalk/trace.py 의 Event 목록."""

    def __init__(self, events):
        self.ev = events

    def find(self, file: str | None = None, contains: str | None = None, func: str | None = None, call: bool | None = None, **meta):
        out = []
        for e in self.ev:
            if file and not e.file.endswith(file):
                continue
            if func and e.func != func:
                continue
            if contains and contains not in e.code:
                continue
            if call is not None and bool(e.meta.get("call")) != call:
                continue
            if any(e.meta.get(k) != v for k, v in meta.items()):
                continue
            out.append(e)
        return out

    def one(self, *a, nth: int = 0, **k):
        r = self.find(*a, **k)
        if len(r) <= nth:
            raise KeyError(f"실행 기록에 없음: {a} {k} (찾은 {len(r)}개)")
        return r[nth]

    def p(self, name: str, *a, nth: int = 0, **k):
        e = self.one(*a, nth=nth, **k)
        if name not in e.produced:
            raise KeyError(f"{e.file}:{e.line} 이(가) {name} 을(를) 만들지 않음: {list(e.produced)}")
        return e.produced[name]


class Sources:
    """원본 파일 줄(대본 안에 그대로 넣어 플레이어가 보여 줌)과 줄 번호 찾기."""

    def __init__(self, files: list[str]):
        self.text = {fn: (ROOT / fn).read_text(encoding="utf-8").split("\n") for fn in files}

    def line(self, fn: str, contains: str, nth: int = 0, after: int = 0) -> int:
        hits = [i + 1 for i, s in enumerate(self.text[fn]) if contains in s and i + 1 > after]
        if len(hits) <= nth:
            raise KeyError(f"{fn} 에 '{contains}' 없음 — 원본이 바뀌었으면 대본을 고쳐야 합니다")
        return hits[nth]

    def lines(self, fn: str, *contains: str, after: int = 0) -> list[int]:
        return [self.line(fn, c, after=after) for c in contains]

    def span(self, fn: str, start: str, end: str, after: int = 0) -> list[int]:
        a = self.line(fn, start, after=after)
        b = self.line(fn, end, after=a - 1)
        return list(range(a, b + 1))


def speech_hash(speech: list[str]) -> str:
    """단계 음성 대본의 지문 — 녹음 파일이 지금 대본으로 만든 것인지 확인(tools/narrate.py 와 같은 함수)."""
    import hashlib
    return hashlib.sha1("\n".join(speech).encode("utf-8")).hexdigest()[:16]


def attach_speech(script: dict, name: str) -> dict:
    """단계마다 자막 줄(captions)·읽을 문장(speech)을 붙이고, 녹음(docs/guide_build/walk/audio/<name>/manifest.json)이 지금 대본과 맞으면 S.audio 로 연결."""
    from tools.codewalk.speech import josa_display, prepare
    for st in script["steps"]:
        for k in ("narr", "hint"):            # 화면 글의 조사도 읽는 소리에 맞춤(0.1655 을 → 0.1655 를)
            if st.get(k):
                st[k] = josa_display(st[k])
        c = st.get("code")
        if isinstance(c, dict) and c.get("notes"):
            c["notes"] = {k: josa_display(v) for k, v in c["notes"].items()}
        st.update(prepare(st))
    man = ROOT / "docs" / "guide_build" / "walk" / "audio" / name / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        ok = {}
        for i, st in enumerate(script["steps"]):
            a = m.get("steps", {}).get(str(i))
            if a and a.get("hash") == speech_hash(st["speech"]) and (man.parent / a["file"]).exists():
                ok[str(i)] = {"file": a["file"], "segs": a["segs"], "duration": a["duration"]}
        vids = sorted({((m["steps"][i].get("sig") or m).get("voice_id") or "") for i in ok})      # 연결된 단계들의 음성(섞였는지)
        script["audio"] = {"dir": f"audio/{name}/", "engine": m.get("engine"), "voice": m.get("voice_name") or m.get("voice_id"), "steps": ok,
                           "missing": [i for i in range(len(script["steps"])) if str(i) not in ok], "voices": vids}
        if len(vids) > 1:
            print(f"  ⚠ {name}: 녹음 음성이 섞여 있습니다 {vids} — narrate.bat {name} 로 모두 같은 음성으로 다시 만드세요")
    return script


def build_html(script: dict, out: Path) -> Path:
    attach_speech(script, Path(out).stem)
    tpl = (Path(__file__).parent / "player.html").read_text(encoding="utf-8")
    js = json.dumps(script, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = tpl.replace("/*__SCRIPT__*/null", js).replace("__TITLE__", script["title"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


# 대본을 쓸 때 결과 폴더에 있던 파일(설정 이름별). 대본(음성)이 결과 파일 수·차이를 말하므로, 파일이 새로 생기거나 없어지면
# 조용히 다른 문장을 만들지 않고 빌드를 멈춘다(2026-11-09 — 녹음은 음성 지문에 묶여 있음). 결과를 새로 돌렸다면 대본을 고친 뒤 여기를 갱신.
RESULT_FILES = {
    ("forecast", "f11_lstm_macro_k1"): ["f11_lstm_macro_k1_24716c9991cf_daily.parquet", "f11_lstm_macro_k1_v1105_a490118990b1_daily.parquet",
                                        "f11_lstm_macro_k1_v1107_6bcd24db12da_daily.parquet"],
    ("forecast", "f11_arima_k1"): ["f11_arima_k1_62140656dbf1_daily.parquet", "f11_arima_k1_v1107_29387d348071_daily.parquet"],
    ("forecast", "f11_sarima_k1"): ["f11_sarima_k1_326b08f39e00_daily.parquet", "f11_sarima_k1_v1107_fbc71152d943_daily.parquet"],
    ("forecast", "repro_biva_orig"): ["repro_biva_orig_16d88d2dd3f1_daily.parquet", "repro_biva_orig_9d68fbbfb7e3_daily.parquet",
                                      "repro_biva_orig_v1107_50dd5d24d5ba_daily.parquet"],
    ("forecast", "f11_ridge_macro_k1"): ["f11_ridge_macro_k1_c4dfb6c66693_daily.parquet", "f11_ridge_macro_k1_v1105_088044cf2a3f_daily.parquet"],
    ("trading", "12_lgbm_ABCD_M_v1107"): ["12_lgbm_ABCD_M_v1107_0b61d18f4d03_daily.parquet"],
    ("forecast", "f11_vecm_k1"): [],
}


def result_files(mode: str, name: str) -> list[str]:
    """results/<mode>/<name>[_태그]_*_daily.parquet 목록(다른 실험 이름이 같은 접두어로 시작하는 경우 제외)."""
    out = []
    for fp in sorted((ROOT / "results" / mode).glob(f"{name}_*_daily.parquet")):
        stem = fp.name[len(name) + 1:]
        if stem.count("_") == 1 or stem.startswith("v1"):
            out.append(fp.name)
    return out


def match_result(mode: str, name: str, preds: dict) -> dict | None:
    """결과 폴더(results/<mode>/<name>[_태그]_*_daily.parquet)에서 이 예측들과 가장 잘 맞는 결과 파일을 찾는다.
    preds: {날짜: 예측}. 반환 {file, n, max_diff, all: [(file, n, max_diff), …]} — 결과 파일이 없으면 None.
    RESULT_FILES 에 이 이름이 있으면 지금 폴더의 목록이 그것과 같은지 확인하고, 다르면 멈춘다(대본이 바뀌지 않게)."""
    import pandas as pd
    found = result_files(mode, name)
    if (mode, name) in RESULT_FILES and sorted(RESULT_FILES[(mode, name)]) != found:
        want = set(RESULT_FILES[(mode, name)])
        raise RuntimeError(f"결과 파일 목록이 대본을 쓸 때와 다릅니다({mode}/{name}): 새로 생김 {sorted(set(found) - want)}, 없어짐 {sorted(want - set(found))}"
                           " — 대본의 결과 파일 설명을 고친 뒤 tools/codewalk/common.py 의 RESULT_FILES 를 갱신하세요(녹음은 바뀐 단계만 다시).")
    out = []
    for fn in found:
        fp = ROOT / "results" / mode / fn
        d = pd.read_parquet(fp, columns=["prediction"])["prediction"].dropna()   # 못 읽으면 멈춤(조용히 건너뛰면 대본이 바뀜)
        s = pd.Series(preds, dtype=float)
        s.index = pd.DatetimeIndex(s.index)
        common = d.index.intersection(s.index)
        if len(common):
            out.append((fp.name, int(len(common)), float(np.max(np.abs(d.loc[common].values - s.loc[common].values)))))
    if not out:
        return None
    best = min(out, key=lambda r: (r[2], -r[1]))
    return {"file": best[0], "n": best[1], "max_diff": best[2], "all": out}


def _near(a: str, b: str, rtol: float, atol: float) -> bool:
    """두 출력이 숫자만 조금 다른지: 숫자 아닌 글자는 같고, 숫자는 |a−b| ≤ atol + rtol·|b|."""
    import re as _re
    pat = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    if _re.sub(pat, "#", a) != _re.sub(pat, "#", b):
        return False
    xa, xb = [float(v) for v in _re.findall(pat, a)], [float(v) for v in _re.findall(pat, b)]
    return len(xa) == len(xb) and all(abs(u - v) <= atol + rtol * abs(v) for u, v in zip(xa, xb))


def run_lab(code: str, expect: str, *, pure: bool, timeout: int = 120, rtol: float | None = None, atol: float = 1e-9, tutor: bool | None = None) -> dict:
    """'직접 실행해 보기' 코드: 혼자 도는 파이썬 코드를 실제로 실행해 출력이 기대 출력(엔진 값으로 만든 문자열)과 같은지 확인한다.
    pure=True 면 표준 라이브러리(math)만 써야 한다 — Python Tutor(한 줄씩 실행 화면)에서 돌아가게. tutor=False 는 순수 파이썬이라도 Python Tutor 의
    단계 한도(1,000)를 넘을 만큼 반복이 많아 복사해서 실행하게 한다(기본: pure 와 같음). 반환: 플레이어용 lab 사전(ok 포함)."""
    import re as _re
    import subprocess
    import sys as _sys
    if pure:
        bad = [m for m in _re.findall(r"^\s*(?:import|from)\s+(\w+)", code, flags=_re.M) if m != "math"]
        if bad:
            raise AssertionError(f"순수 파이썬 실험 코드에 외부 모듈: {bad}")
    r = subprocess.run([_sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    exact = r.returncode == 0 and out == expect.strip()
    ok = exact or (r.returncode == 0 and rtol is not None and _near(out, expect.strip(), rtol, atol))
    return {"code": code, "expect": expect.strip(), "got": out if not exact else None, "err": r.stderr[-800:] if r.returncode else None, "pure": pure, "tutor": pure if tutor is None else bool(tutor and pure), "ok": ok,
            "tol": rtol if (ok and not exact) else None}


def py_list(vals, sig: int = 17) -> str:
    """실험 코드에 넣을 숫자 목록(값을 그대로 — 17자리)."""
    return "[" + ", ".join(repr(float(v)) if sig >= 17 else f"{float(v):.{sig}g}" for v in vals) + "]"
