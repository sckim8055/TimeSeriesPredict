"""음성 대본 점검 — 대본(say)의 숫자가 그 단계 화면(설명·식·힌트·표·코드 줄 설명)에 있는 숫자인지, 읽을 문장에 영문·숫자가 남지 않았는지.

  python -m tools.codewalk.check_say docs/guide_build/walk/biva.html [--show]

대본은 화면 문장을 소리 내 읽기 좋게 고쳐 쓴 것이라, 숫자가 화면과 어긋나면 안 된다(교수 심사에서 영상과 화면 숫자가 다르면 전체 신뢰가 흔들림).
규칙: 대본의 숫자 x 마다, 같은 단계 화면의 숫자 y 가운데 x 의 표시 자릿수로 반올림해 같은 것이 있어야 한다(지수 표기는 유효숫자 기준).
화면에 없는 숫자(대본에서 처음 나오는 수)는 '출처 없음' 으로 모두 보여 준다 — 0 개여야 통과. 구조적인 수(배치 3, 시점 6 등)도 화면에 있어야 한다.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.codewalk.speech import leftovers, prepare  # noqa: E402

NUM = re.compile(r"(?<![A-Za-z_])[-−+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?")
DATE = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")


def nums(text: str) -> list[str]:
    t = DATE.sub(lambda m: " " + m.group(0).replace("-", " ") + " ", text)   # 날짜는 연·월·일 숫자로
    return [m.group(0).replace("−", "-").replace(",", "") for m in NUM.finditer(t)]


def walk(o, out: list):
    if isinstance(o, dict):
        for v in o.values():
            walk(v, out)
    elif isinstance(o, list):
        for v in o:
            walk(v, out)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        if math.isfinite(o):
            out.append(repr(float(o)))
    elif isinstance(o, str):
        out.extend(nums(o))


def matches(x: str, pool: list[float]) -> bool:
    v = float(x)
    if "e" in x.lower():
        sig = len(re.sub(r"[^0-9]", "", x.lower().split("e")[0]).lstrip("0")) or 1
        return any(y != 0 and abs(y - v) <= 0.5 * 10 ** (math.floor(math.log10(abs(y))) - sig + 1) * 1.0001 for y in pool) or (v == 0 and 0.0 in pool)
    d = len(x.split(".")[1]) if "." in x else 0
    tol = 0.5 * 10 ** (-d) * 1.0001
    return any(abs(y - v) <= tol for y in pool)


def check(html: Path, show: bool = False) -> int:
    h = html.read_text(encoding="utf-8")
    i = h.index("const S = ") + len("const S = "); j = h.index(";\nconst $ =")
    S = json.loads(h[i:j])
    bad_total = 0
    for k, st in enumerate(S["steps"]):
        say = st.get("say")
        p = prepare(st)
        if not say:
            print(f"[{k}] 대본 없음(화면 설명을 그대로 읽음)")
            continue
        pool_txt: list = []
        for key in ("narr", "formula", "hint"):
            pool_txt.extend(nums(st.get(key) or ""))
        walk(st.get("tensors"), pool_txt)
        walk((st.get("code") or {}).get("notes"), pool_txt)
        walk([c["what"] for c in st.get("checks") or []], pool_txt)
        pool = [float(v) for v in pool_txt]
        miss = [x for x in nums(say) if not matches(x, pool)]
        left = leftovers(" ".join(p["speech"]))
        if miss or left or show:
            print(f"[{k}] {st['chapter']}" + (f"  · 출처 없음: {miss}" if miss else "") + (f"  · 읽을 문장에 남은 영문·숫자: {left}" if left else ""))
            if show:
                for c, s_ in zip(p["captions"], p["speech"]):
                    print("     자막:", c); print("     음성:", s_)
        bad_total += len(miss) + len(left)
    print(f"점검: 단계 {len(S['steps'])} · 문제 {bad_total}")
    return bad_total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("html"); ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    sys.exit(1 if check(Path(a.html), a.show) else 0)
