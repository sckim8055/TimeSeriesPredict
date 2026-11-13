"""변수 사전(`data/variable_master.xlsx`) 검사 — 데이터를 받기 전에 '한 변수 = 한 출처'를 지킨다(2026-11-09).

원칙(사용자 결정): 사용자는 엑셀 데이터 시트에 값을 넣고 '데이터 모으기'만 한다. 판단은 프로그램이 하고,
멈출 때는 고칠 곳(시트·열) 하나와 할 일 한 가지만 쉬운 말로 알려 준다.

자동 처리(멈추지 않음, 기록만):
  · 새 열(variables 에 없는 이름) → variables 에 자동 등록(source=XLSX, 주기는 시트로, lag 는 주기별 보수적 기본값 · '확인 안 됨').
  · API 로 받는 변수(source 가 XLSX 가 아닌 변수)의 열이 데이터 시트에 있음 → 그 엑셀 열은 쓰지 않고 API 값을 쓴다(안내 한 줄).
    전에는 둘 다 저장되어 어느 값이 쓰일지가 파일 이름 순서로 정해졌다(2026-11-09 확인한 문제).
  · 머리글은 비었지만 값도 없는 열(서식만 남은 열) → 무시.
멈춤(오류) — 엑셀 모양이 잘못되어 프로그램이 정할 수 없는 경우만:
  · 머리글이 빈 열에 값이 있음 / 같은 이름의 열이 두 번(같은 시트·다른 시트) / date 열 없음
  · 등록된 엑셀 변수가 다른 주기 시트에 있음 / 열 이름에 띄어쓰기·기호(밑줄 _ 만 됨) / A_ 로 시작(기술지표 전용) / _pct·_diff·_log 로 끝남(변환 열 이름과 겹침)
  · (관리자용 variables 시트) 같은 API 코드를 대표 변수 둘이 가리킴 / API 변수의 source_code 빈칸
관리자 참고(수집 기록에만 남김): 엑셀 변수인데 값이 없음, 같은 이름(name)의 대표 변수 둘, 거의 같은 시계열 쌍(near_duplicates).
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from engine.registry import Registry

DATA_SHEETS = (("df_D", "D"), ("df_W", "W"), ("df_M", "M"), ("df_Q", "Q"), ("df_H", "H"), ("df_Y", "Y"))
API_SOURCES = ("YF", "FRED", "ECOS", "KRX")
ID_RE = re.compile(r"^\w+$")  # 영문·숫자·한글·밑줄(띄어쓰기·기호 안 됨)
SUFFIX_RE = re.compile(r"_(pct|diff|log|logdiff)\d*$", re.IGNORECASE)


@dataclass
class CheckResult:
    errors: list[str] = field(default_factory=list)    # 멈춤 — 사용자에게 보임
    notices: list[str] = field(default_factory=list)   # 자동 처리한 것 — 사용자에게 한 줄씩 보임
    admin: list[str] = field(default_factory=list)     # 관리자 참고 — 수집 기록(manifest)에만
    new_vars: list[dict] = field(default_factory=list)  # 자동 등록할 행
    ignored: list[str] = field(default_factory=list)   # 쓰지 않을 엑셀 열(API 변수)

    def as_dict(self) -> dict:
        return {"errors": self.errors, "notices": self.notices, "admin": self.admin,
                "new_vars": [r["id"] for r in self.new_vars], "ignored": self.ignored}


def sheet_headers(master_path: str | Path) -> dict[str, tuple[list, set[int]]]:
    """데이터 시트별 (첫 줄 머리글 — 엑셀 그대로, 값이 하나라도 있는 열 번호(0부터)).
    pandas 는 중복 머리글에 '.1' 을 붙여 숨기므로 openpyxl 로 직접 읽는다."""
    import openpyxl

    wb = openpyxl.load_workbook(master_path, read_only=True, data_only=True)
    out: dict[str, tuple[list, set[int]]] = {}
    try:
        for sh, _ in DATA_SHEETS:
            if sh not in wb.sheetnames:
                continue
            wb[sh].reset_dimensions()  # 시트의 크기 표시가 낡아 있어도 실제 줄을 끝까지 읽는다(engine/store.read_data_sheets 와 같은 이유)
            rows = wb[sh].iter_rows(values_only=True)
            hdr = list(next(rows, ()))
            filled: set[int] = set()
            for row in rows:
                for i, x in enumerate(row):
                    if x is not None and str(x).strip() != "":
                        filled.add(i)
            out[sh] = (hdr, filled)
    finally:
        wb.close()
    return out


def _blank(x) -> bool:
    return x is None or str(x).strip() == ""


def check_master(reg: Registry, master_path: str | Path) -> CheckResult:
    r = CheckResult()
    seen: dict[str, str] = {}  # 열 이름 → 처음 나온 시트
    hs = sheet_headers(master_path)
    # 시트 이름 오타(df_m 등): 알림 없이 무시되던 것을 멈춤으로(2026-11-10)
    import openpyxl

    _wb = openpyxl.load_workbook(master_path, read_only=True)
    try:
        valid = {sh for sh, _ in DATA_SHEETS}
        for name in _wb.sheetnames:
            fixed = next((sh for sh in valid if sh.lower() == name.strip().lower()), None)
            if fixed and name != fixed:
                r.errors.append(f"시트 이름 '{name}' 을 '{fixed}' 로 고쳐 주세요(대소문자·빈칸까지 같아야 읽습니다).")
    finally:
        _wb.close()
    try:
        from engine.store import read_data_sheets
        _, cell_issues = read_data_sheets(master_path)
    except Exception as e:  # noqa: BLE001 — 칸 검사를 못 해도 머리글 검사는 계속
        cell_issues = [{"kind": "column_stats", "numeric_cells": {}, "formula_cells": []}]
        r.admin.append(f"데이터 시트 칸 검사를 하지 못함: {type(e).__name__}: {e}")
    stats = cell_issues[-1]
    lower_ids = {i.lower(): i for i in reg.ids(usable_only=False)}
    read_cols: set[str] = set()  # 실제로 값을 읽을 열(등록된 엑셀 변수 + 새로 등록할 열)
    for sh, fq in DATA_SHEETS:
        if sh not in hs:
            continue
        hdr, filled = hs[sh]
        cols = [i for i in range(max(len(hdr), max(filled) + 1 if filled else 0))]
        heads = {i: (None if i >= len(hdr) or _blank(hdr[i]) else str(hdr[i])) for i in cols}
        named = [i for i in cols if heads[i] is not None]
        date_ix = [i for i in named if heads[i].strip().lower() == "date"]
        if [i for i in named if i not in date_ix] and not date_ix:
            r.errors.append(f"{sh} 시트: 맨 왼쪽 열의 제목이 'date' 여야 합니다. A1 칸에 date 라고 적어 주세요.")
        for i in cols:
            cell = f"{sh} 시트 {_col_letter(i + 1)}열"
            raw = heads[i]
            if raw is None:
                if i in filled:
                    r.errors.append(f"{cell}: 값은 있는데 제목(1행)이 비어 있습니다. 1행에 변수 이름을 적거나 이 열을 지워 주세요.")
                continue
            h = raw.strip()
            if h.lower() == "date":
                continue
            if raw != h:
                r.notices.append(f"{cell} '{raw}': 제목 앞뒤 공백을 지우고 '{h}' 로 읽었습니다.")
            if h in seen:
                where = "이 시트에 두 번" if seen[h] == sh else f"{seen[h]} 시트에도"
                r.errors.append(f"{cell} '{h}': 같은 이름의 열이 {where} 있습니다. 하나만 남겨 주세요.")
                continue
            seen[h] = sh
            if h in reg:
                v = reg.get(h)
                if v.source != "XLSX":
                    r.ignored.append(h)
                    r.notices.append(f"{cell} '{h}': 자동으로 받아 오는 변수({v.source})라 엑셀 값은 쓰지 않고 받아 온 값을 씁니다.")
                    continue
                if str(v.freq) != fq:
                    r.errors.append(f"{cell} '{h}': 이 변수는 {_FREQ_KO.get(str(v.freq), v.freq)} 자료로 등록되어 있습니다. "
                                    f"{_SHEET_OF.get(str(v.freq), 'df_' + str(v.freq))} 시트로 옮겨 주세요.")
                    continue
                if v.source_code and v.source_code.strip() != f"{sh}!{h}":
                    r.admin.append(f"{h}: source_code '{v.source_code}' 를 실제 위치 {sh}!{h} 로 읽었습니다")
                read_cols.add(h)
                continue
            if h.lower() in lower_ids:
                r.errors.append(f"{cell} '{h}': 이미 있는 변수 '{lower_ids[h.lower()]}' 와 대소문자만 다릅니다. 같은 변수면 이름을 똑같이 적어 주세요.")
                continue
            if stats["numeric_cells"].get(f"{sh}!{h}", 0) == 0:
                # 숫자가 하나도 없는 새 열(비고·단위 같은 메모)은 변수로 등록하지 않는다 — 전에는 글자 열도 사전에 등록됐다(2026-11-10 점검 N13)
                r.notices.append(f"{cell} '{h}': 숫자가 없어 변수로 읽지 않습니다(메모 열로 봅니다).")
                continue
            bad = _bad_id(h)
            if bad:
                r.errors.append(f"{cell} '{h}': 이름에 {bad}. 이름을 고쳐 주세요.")
                continue
            read_cols.add(h)
            r.new_vars.append({"id": h, "name": h, "freq": fq, "source": "XLSX", "source_code": f"{sh}!{h}", "active": "Y",
                               "default_fill": "ffill", "note": f"데이터 시트 {sh} 에서 자동 등록 — lag_days·group·설명 확인 필요(관리자)"})
            r.notices.append(f"{cell} '{h}': 새 변수로 자동 등록했습니다(공개 시차는 안전한 기본값으로 둡니다).")
    # 칸 단위 문제: 읽을 열에 못 읽는 날짜·글자 값·계산값 없는 수식·같은 날짜 중복이 있으면 멈춘다 — 전에는 그런 줄·칸이 조용히 빠졌다(2026-11-10 점검 N1·N2)
    _MSG = {"bad_date": "날짜를 읽을 수 없습니다('{text}'). 2020-01-31 처럼 적어 주세요.",
            "bad_value": "숫자가 아닌 값('{text}')이 있습니다. 숫자로 고치거나 칸을 비워 주세요.",
            "formula_no_value": "수식의 계산값이 없습니다. 엑셀에서 이 파일을 열어 다시 저장하거나, 값으로 붙여 넣어 주세요.",
            "dup_date": "같은 날짜 {text} 가 두 번 있습니다. 한 줄만 남겨 주세요."}
    grouped: dict[tuple, list[dict]] = {}
    for x in cell_issues[:-1]:
        if x["kind"] == "bad_date" or x["column"] in read_cols:
            grouped.setdefault((x["sheet"], x["column"], x["kind"]), []).append(x)
    for (sh, col, kind), xs in grouped.items():
        first = xs[0]
        more = f" (같은 문제가 이 열에 {len(xs) - 1}칸 더 있습니다)" if len(xs) > 1 else ""
        where = f"{sh} 시트 {first['cell']}" + ("" if col == "date" else f" ('{col}' 열)")
        r.errors.append(f"{where}: " + _MSG[kind].format(text=first["text"]) + more)
    if r.new_vars and stats.get("formula_cells"):
        # 새 열을 등록하려면 프로그램이 엑셀을 다시 저장해야 하는데, 그때 수식의 계산값이 지워진다(2026-11-10 점검 N3)
        fc = stats["formula_cells"]
        r.errors.append(f"{fc[0].replace('!', ' 시트 ')} 에 수식이 있습니다" + (f"(외 {len(fc) - 1}칸)" if len(fc) > 1 else "")
                        + ". 새 열을 등록하면 수식의 값이 지워지므로, 수식 칸을 '값으로 붙여넣기' 한 뒤 다시 실행해 주세요.")
    for vid in reg.ids(usable_only=False):
        v = reg.get(vid)
        bad = _bad_id(vid)
        if bad:
            r.errors.append(f"variables 시트 '{vid}': 이름에 {bad}. (관리자) 이름을 고쳐 주세요.")
        if v.source in API_SOURCES and not (v.source_code or "").strip():
            r.errors.append(f"variables 시트 '{vid}': 받아 올 코드(source_code)가 비어 있습니다. (관리자) 코드를 적어 주세요.")
        if v.source == "XLSX" and vid not in seen:
            r.admin.append(f"{vid}: 엑셀 변수인데 데이터 시트에 열이 없음(값 없음)")
    groups: dict[tuple, list[str]] = {}
    for vid in reg.ids(usable_only=False):
        v = reg.get(vid)
        if v.source in API_SOURCES and (v.source_code or "").strip():
            groups.setdefault((v.source, v.source_code.strip()), []).append(vid)
    for (src, code), ids in groups.items():
        reps = [i for i in ids if not reg.get(i).duplicate_of]
        if len(reps) > 1:
            r.errors.append(f"variables 시트: {', '.join(reps)} 가 같은 값({src} {code})을 두 번 받습니다. "
                            f"(관리자) {reps[1]} 의 duplicate_of 에 {reps[0]} 를 적어 주세요.")
    names: dict[str, list[str]] = {}
    for vid in reg.ids(usable_only=False):
        v = reg.get(vid)
        nm = (v.name or "").strip()
        if nm and not v.duplicate_of:
            names.setdefault(nm, []).append(vid)
    for nm, ids in names.items():
        if len(ids) > 1:
            r.admin.append(f"이름 '{nm}' 인 대표 변수 둘 이상: {', '.join(ids)}")
    return r


_FREQ_KO = {"D": "일간", "W": "주간", "M": "월간", "Q": "분기", "H": "반기", "Y": "연간"}
_SHEET_OF = {fq: sh for sh, fq in DATA_SHEETS}


def _bad_id(vid: str) -> str:
    if not ID_RE.match(vid):
        return "띄어쓰기나 기호가 있습니다(글자·숫자·밑줄 _ 만 됩니다)"
    if vid.startswith("A_"):
        return "'A_' 로 시작할 수 없습니다(기술지표 이름과 겹침)"
    if SUFFIX_RE.search(vid):
        return "_pct·_diff·_log 로 끝날 수 없습니다(프로그램이 만드는 이름과 겹침)"
    return ""


def near_duplicates(dv, reg: Registry, *, corr_min: float = 0.99, scale: tuple[float, float] = (0.9, 1.11), min_obs: int = 24) -> list[str]:
    """수집한 값으로 '거의 같은 시계열' 쌍을 찾는다(경고만). 월말 값 → 월간 변화의 상관, 중앙값 크기 비.
    이미 duplicate_of 로 이어진 쌍(한쪽이 다른 쪽을 가리키거나 같은 대표를 가리킴)은 뺀다."""
    w = dv.wide()
    if w.empty:
        return []
    m = w.resample("ME").last()
    out = []
    for a, b in itertools.combinations(m.columns, 2):
        if a in reg and b in reg and reg.resolve(a) == reg.resolve(b):
            continue
        d = pd.concat([m[a], m[b]], axis=1).dropna()
        if len(d) < min_obs:
            continue
        c = d.diff().dropna()
        if len(c) < min_obs - 1 or (c.std() == 0).any():
            continue
        rho = float(c.corr().iloc[0, 1])
        ratio = float((d[a].abs().median() + 1e-12) / (d[b].abs().median() + 1e-12))
        if rho >= corr_min and scale[0] <= ratio <= scale[1]:
            out.append(f"{a} 와 {b}: 월간 변화 상관 {rho:.4f}, 크기 비 {ratio:.3f}({len(d)}개월) — 같은 값일 수 있음(관리자 확인)")
    return out


def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, rem = divmod(n - 1, 26)
        s = chr(65 + rem) + s
    return s


def register_new_columns(reg: Registry, master_path: str | Path) -> list[str]:
    """새 열을 variables 에 등록하고 안내문을 돌려준다(check_master 의 new_vars 를 그대로 씀). 오류가 있으면 아무것도 쓰지 않는다."""
    from engine.registry import append_master_rows

    r = check_master(reg, master_path)
    if r.errors or not r.new_vars:
        return []
    append_master_rows(master_path, r.new_vars)
    return [n for n in r.notices if "자동 등록" in n]
