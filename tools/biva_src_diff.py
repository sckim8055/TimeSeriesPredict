"""원본 BIVA 소스와 플랫폼 사본(models/biva_src)의 차이를 보여주고, 표시 없는 수정이 있는지 검사한다.

  python tools/biva_src_diff.py                      # 원본 위치: legacy/biva_orig/model (없으면 BIVA_ORIG 환경변수 또는 D:/test/SVU_SCT_FIN-main/Model/BIVA/model)
  python tools/biva_src_diff.py --check              # 모든 변경 덩어리에 [TSP 수정] 표시가 있는지만 검사(테스트에서 사용)

규칙: 사본의 모든 변경 덩어리(hunk)는 '[TSP 수정 #n]' 주석을 포함해야 한다. 원본 줄은 '# 원본:' 으로 남긴다.
"""
from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["BIVA.py", "BIVA_BRITS.py", "BIVA_RITS.py", "BIVA_LSTM_VAE.py"]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    od = orig_dir()
    if od is None:
        sys.exit("원본 BIVA 소스를 찾지 못했습니다(BIVA_ORIG 환경변수로 지정)")
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
