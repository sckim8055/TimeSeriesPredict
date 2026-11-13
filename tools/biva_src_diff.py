"""원본 BIVA 소스와 플랫폼 사본(models/biva_src)의 차이를 보여주고, 표시 없는 수정이 있는지 검사한다.

  python tools/biva_src_diff.py                      # 원본 위치: legacy/biva_orig/model (없으면 BIVA_ORIG 환경변수 또는 D:/test/SVU_SCT_FIN-main/Model/BIVA/model)
  python tools/biva_src_diff.py --check              # 모든 변경 덩어리에 [TSP 수정] 표시가 있는지만 검사(테스트에서 사용)
  python tools/biva_src_diff.py --lock               # 지금의 변경 덩어리 목록을 models/biva_src/PATCHES_LOCK.txt 에 다시 쓴다
                                                     # (사본을 고친 뒤에만 — PATCHES.md 에 그 수정을 먼저 적을 것)

규칙: 사본의 모든 변경 덩어리(hunk)는 '[TSP 수정 #n]' 주석을 포함해야 한다. 원본 줄은 '# 원본:' 으로 남긴다.
잠금(2026-11-13 독립 검토 11): 표시 하나가 붙은 덩어리 안에 표시 없는 수정이 함께 들어가도 위 규칙은 통과한다. 그래서 변경 덩어리 전체의
'내용 지문'(덩어리의 -/+ 줄만으로 만든 sha256 — 줄 번호는 넣지 않는다)을 PATCHES_LOCK.txt 에 적어 두고, 시험이 지금 덩어리와 똑같은지 본다.
tools_orig.py(원본 utils/tools.py 에서 가져온 두 개)는 함수·클래스 글자가 원본과 같은지 본다.
"""
from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["BIVA.py", "BIVA_BRITS.py", "BIVA_RITS.py", "BIVA_LSTM_VAE.py"]
LOCK = ROOT / "models" / "biva_src" / "PATCHES_LOCK.txt"
TOOLS_TAKEN = ["adjust_learning_rate", "EarlyStopping"]   # tools_orig.py 가 원본 utils/tools.py 에서 그대로 가져온 것


def orig_dir() -> Path | None:
    for c in [os.environ.get("BIVA_ORIG"), ROOT / "legacy" / "biva_orig" / "model", "D:/test/SVU_SCT_FIN-main/Model/BIVA/model"]:
        if c and (Path(c) / "BIVA.py").exists():
            return Path(c)
    return None


def hunks(a: list[str], b: list[str]):
    cur = []
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if line.startswith("@@"):
            if cur:
                yield cur
            cur = [line]
        elif cur:
            cur.append(line)
    if cur:
        yield cur


def check(od: Path) -> list[str]:
    problems = []
    for f in FILES:
        a = (od / f).read_text(encoding="utf-8").splitlines()
        b = (ROOT / "models" / "biva_src" / f).read_text(encoding="utf-8").splitlines()
        for h in hunks(a, b):
            if not any("[TSP 수정" in x for x in h):
                problems.append(f"{f} {h[0]}: 표시 없는 변경\n" + "\n".join(h[1:8]))
    return problems


def hunk_fingerprints(od: Path) -> list[str]:
    """변경 덩어리마다 '파일 이름 sha256(덩어리의 -/+ 줄)' 한 줄. 줄 번호(@@ 머리)는 넣지 않아 위쪽이 밀려도 같다."""
    import hashlib
    out = []
    for f in FILES:
        a = (od / f).read_text(encoding="utf-8").splitlines()
        b = (ROOT / "models" / "biva_src" / f).read_text(encoding="utf-8").splitlines()
        for h in hunks(a, b):
            body = "\n".join(x for x in h[1:] if x[:1] in "+-")
            out.append(f"{f} {hashlib.sha256(body.encode('utf-8')).hexdigest()}")
    return out


def read_lock() -> list[str]:
    return [ln.strip() for ln in LOCK.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")] if LOCK.exists() else []


def tools_problems(od: Path) -> list[str]:
    """tools_orig.py 의 가져온 함수·클래스가 원본 utils/tools.py 의 글자와 같은가."""
    import ast
    src = od / "utils" / "tools.py"
    if not src.exists():
        return [f"원본 {src} 가 없어 tools_orig.py 를 비교하지 못했습니다"]
    o, c = src.read_text(encoding="utf-8"), (ROOT / "models" / "biva_src" / "tools_orig.py").read_text(encoding="utf-8")
    def seg(t, name):
        n = next((x for x in ast.parse(t).body if getattr(x, "name", None) == name), None)
        return None if n is None else ast.get_source_segment(t, n)
    return [f"tools_orig.py 의 {nm} 가 원본과 다릅니다" for nm in TOOLS_TAKEN if seg(o, nm) is None or seg(o, nm) != seg(c, nm)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--lock", action="store_true")
    a = ap.parse_args()
    od = orig_dir()
    if od is None:
        sys.exit("원본 BIVA 소스를 찾지 못했습니다(BIVA_ORIG 환경변수로 지정)")
    if a.lock:
        fp = hunk_fingerprints(od)
        LOCK.write_text("# BIVA 사본의 변경 덩어리 지문 — tools/biva_src_diff.py --lock 으로만 다시 쓴다. 바꾸기 전에 PATCHES.md 에 그 수정을 적을 것.\n"
                        + "\n".join(fp) + "\n", encoding="utf-8")
        print(f"{len(fp)}개 덩어리를 {LOCK} 에 적었습니다")
        return
    if a.check:
        p = check(od)
        print("\n\n".join(p) if p else f"OK — 모든 변경에 [TSP 수정] 표시 (원본: {od})")
        sys.exit(1 if p else 0)
    for f in FILES:
        a_ = (od / f).read_text(encoding="utf-8").splitlines()
        b_ = (ROOT / "models" / "biva_src" / f).read_text(encoding="utf-8").splitlines()
        d = list(difflib.unified_diff(a_, b_, f"원본/{f}", f"사본/{f}", lineterm="", n=1))
        print("\n".join(d) if d else f"== {f}: 원본과 같음")


if __name__ == "__main__":
    main()
