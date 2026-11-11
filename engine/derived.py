"""파생 변수(source=DERIVED) 계산.

마스터의 source_code 에 적힌 식을 그대로 계산한다. 예: `US10Y - US2Y`, `log(KOSPI)`, `SOX / KOSPI`.
- 식에 등장하는 이름은 마스터의 다른 변수 id 여야 한다 (대표 id. 중복 id 는 허용하지 않는다).
- 날짜는 등장 변수들의 교집합. 주기가 다르면 교집합이 작아지므로, 같은 주기끼리만 조합할 것.
- 허용 함수: log, exp, abs, sqrt, diff, pct, shift(n), roll_mean(n), roll_std(n).
"""
from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd

from engine.registry import Registry
from engine.store import LONG_COLS, DataVersion

_FUNCS = {
    "log": np.log,
    "exp": np.exp,
    "abs": np.abs,
    "sqrt": np.sqrt,
    "diff": lambda s, n=1: s.diff(n),
    "pct": lambda s, n=1: s.pct_change(n),
    "shift": lambda s, n=1: s.shift(n),
    "roll_mean": lambda s, n=20: s.rolling(n).mean(),
    "roll_std": lambda s, n=20: s.rolling(n).std(),
}
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def referenced_ids(expr: str) -> list[str]:
    return [n for n in dict.fromkeys(_NAME_RE.findall(expr)) if n not in _FUNCS]


def _check_ast(expr: str) -> None:
    """식에 허용된 구문만 있는지 (이름, 숫자, 사칙연산, 허용 함수 호출)."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Load,
                             ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            continue
        raise ValueError(f"파생식에 허용되지 않는 구문: {ast.dump(node)[:40]} in '{expr}'")


def compute_derived(reg: Registry, dv: DataVersion) -> tuple[pd.DataFrame, list[str]]:
    """마스터의 DERIVED 변수를 모두 계산해 롱 테이블로 돌려준다. (롱, 메시지)."""
    out = []
    notes: list[str] = []
    derived = [reg.get(v) for v in reg.ids(usable_only=False) if reg.get(v).source == "DERIVED"]
    if not derived:
        return pd.DataFrame(columns=LONG_COLS), notes
    need = sorted({i for v in derived for i in referenced_ids(v.source_code)})
    wide = dv.wide(need)
    for v in derived:
        expr = v.source_code.strip()
        try:
            _check_ast(expr)
            missing = [i for i in referenced_ids(expr) if i not in wide.columns]
            if missing:
                notes.append(f"{v.id}: 입력 변수 없음 {missing} → 건너뜀")
                continue
            env = {**_FUNCS, **{c: wide[c] for c in wide.columns}}
            s = eval(compile(expr, "<derived>", "eval"), {"__builtins__": {}}, env)  # noqa: S307 — 구문 검사 통과한 식만
            s = pd.Series(s, index=wide.index).dropna()
            out.append(pd.DataFrame({"variable_id": v.id, "date": s.index, "value": s.values}))
            notes.append(f"{v.id} = {expr}: {len(s)}행")
        except Exception as e:  # noqa: BLE001
            notes.append(f"{v.id}: 계산 실패 — {e}")
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=LONG_COLS)), notes
