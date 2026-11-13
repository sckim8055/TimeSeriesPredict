"""변수 레지스트리 — 모든 입력 변수의 메타데이터 (기획서 3-2, 3-6).

기존 BIVA 엑셀의 `df_var_info` 시트 구조를 그대로 받아 확장한다. 기존 파일은 수정 없이 들어와야 한다.
레지스트리 1행 = 변수 1개. 데이터 자체는 여기 없고 (engine/store.py 의 롱 테이블), 여기는 '설명서'만.
"""
from __future__ import annotations

import hashlib
import io

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

Freq = Literal["D", "W", "M", "Q", "H", "Y"]
Source = Literal["KRX", "YF", "ECOS", "FRED", "XLSX", "DERIVED"]
Fill = Literal["ffill", "bfill", "interp"]

# 기존 엑셀 FEQ 표기 → 표준
_FREQ_MAP = {"D": "D", "W": "W", "M": "M", "Q": "Q", "H": "H", "S": "H", "Y": "Y", "A": "Y", "d": "D", "w": "W", "m": "M", "q": "Q", "h": "H", "y": "Y", "a": "Y"}  # S/H=반기, A/Y=연간
# 기존 LAG(월 단위)를 일 단위로 바꿀 때의 계수
_MONTH_DAYS = 30
# lag_days 가 비어 있을 때의 주기별 보수적 기본값 (DECISIONS 2026-11-03). 0 은 가장 낙관적이라 기본값으로 쓰지 않는다.
LAG_DEFAULT = {"D": 1, "W": 14, "M": 45, "Q": 120, "H": 225, "Y": 455}


class Variable(BaseModel):
    id: str = Field(description="고유 ID. 기존 엑셀 ID 유지 (예: F11, K1)")
    name: str = ""
    description: str = ""
    group: str = ""
    freq: Freq = "M"
    lag_days: int = Field(0, ge=0, description="기준일(기간 시작일)→공개일 일수. 변수별로 정한다")
    lag_confirmed: bool = Field(True, description="False 면 마스터에 lag_days 가 비어 있어 주기별 보수적 기본값을 쓴 것 (미확인)")
    transform: str = "none"
    source: Source = "XLSX"
    source_code: str = Field("", description="어댑터 조회 코드. XLSX는 '시트!열' 형식")
    default_fill: Fill = "ffill"
    unit: str = ""
    domain: str = Field("", description="적용 도메인 (예: common, semicon, realestate)")
    duplicate_of: str = Field("", description="같은 변수의 대표 ID. 값이 있으면 실험에서 자동 제외되고 대표로 치환")
    active: bool = Field(True, description="사용자 스위치. N이면 모든 실험에서 제외 (duplicate_of 와 별개)")
    note: str = Field("", description="마스터 비고. lag/변환 기본값의 근거가 적혀 있다 (엔진은 쓰지 않음, 화면·문서 표시용)")

    @property
    def usable(self) -> bool:
        """실험 후보가 되려면 active 이고 중복이 아니어야 한다 (AND)."""
        return self.active and not self.duplicate_of

    @field_validator("freq", mode="before")
    @classmethod
    def _norm_freq(cls, v: str) -> str:
        v = str(v).strip()
        if v not in _FREQ_MAP:
            raise ValueError(f"알 수 없는 주기 '{v}'")
        return _FREQ_MAP[v]


class Registry:
    def __init__(self, variables: list[Variable] | None = None):
        self._vars: dict[str, Variable] = {}
        self.source_path: str = ""  # 읽은 변수 사전 파일(결과의 run_record 에 남김)
        self.source_sha256: str = ""  # 읽은 그 바이트의 sha256 — 읽은 뒤 파일이 바뀌어도 '실제로 쓴 사전'을 가리킨다(2026-11-10)
        self.source_bytes: bytes | None = None  # 읽은 그 바이트(실험 기록이 사본을 만들 때 디스크를 다시 읽지 않도록)
        for v in variables or []:
            self.add(v)

    # -- 기본 조작 -----------------------------------------------------------
    def add(self, v: Variable, *, replace: bool = False) -> None:
        if v.id in self._vars and not replace:
            raise KeyError(f"중복 ID: {v.id}")
        self._vars[v.id] = v

    def get(self, vid: str) -> Variable:
        return self._vars[vid]

    def __contains__(self, vid: str) -> bool:
        return vid in self._vars

    def __len__(self) -> int:
        return len(self._vars)

    def ids(self, group: str | None = None, usable_only: bool = True) -> list[str]:
        """실험 후보 ID. usable_only=True 면 active=N 과 duplicate_of 있는 행을 모두 제외."""
        return [
            v.id
            for v in self._vars.values()
            if (group is None or v.group == group) and (not usable_only or v.usable)
        ]

    def groups(self) -> list[str]:
        return sorted({v.group for v in self._vars.values()})

    # -- 중복 처리 -----------------------------------------------------------
    def resolve(self, vid: str) -> str:
        """duplicate_of 를 따라 대표 ID 를 돌려준다 (체인은 validate 에서 막음)."""
        v = self.get(vid)
        return v.duplicate_of if v.duplicate_of else vid

    def resolve_detail(self, vids: list[str]) -> dict:
        """요청 ID 목록을 실제로 쓸 ID 로 바꾸고, 무엇을 왜 바꾸거나 뺐는지 함께 돌려준다(결과 파일에 그대로 기록).

        규칙(2026-11-09 사용자 확정 — 도움말 2-3절 표와 같음):
          duplicate_of 값 있음 + active Y/N → 자기 대신 대표 변수로 바꿈(자기 active 는 보지 않음)
          duplicate_of 값 없음 + active Y   → 그대로 포함
          duplicate_of 값 없음 + active N   → 빼고 excluded 에 기록(예전에는 오류로 멈춤)
          대표로 바꾼 뒤 그 대표가 active N  → 빼고 excluded 에 기록
          고른 변수가 모두 빠지면 → 오류(설정과 달리 변수 없이 도는 것을 막는다)
        반환 {requested, used, substituted[{from,to}], merged[{from,to}], excluded[{id,requested_as,reason}]}."""
        used: list[str] = []
        sub, merged, excl = [], [], []
        for vid in vids:
            rep = self.resolve(vid)
            if rep != vid:
                sub.append({"from": vid, "to": rep})
            if not self.get(rep).active:
                excl.append({"id": rep, "requested_as": vid,
                             "reason": "active=N" if rep == vid else f"대표 {rep} 가 active=N"})
                continue
            if rep in used:
                merged.append({"from": vid, "to": rep})
                continue
            used.append(rep)
        if vids and not used:
            raise ValueError("고른 변수 " + ", ".join(vids) + " 가 모두 active=N(또는 대표가 active=N)이라 쓸 변수가 없습니다. "
                             "마스터에서 Y 로 바꾸거나 다른 변수를 고르세요.")
        return {"requested": list(vids), "used": used, "substituted": sub, "merged": merged, "excluded": excl}

    def resolve_many(self, vids: list[str]) -> tuple[list[str], list[str]]:
        """요청 ID 목록 → (실제로 쓸 대표 ID 목록(중복 제거, 순서 유지), 안내 메시지). 규칙은 resolve_detail."""
        d = self.resolve_detail(vids)
        notes = [f"{x['from']} → {x['to']} (중복, 대표로 치환)" for x in d["substituted"]]
        notes += [f"{x['from']}: 이미 포함된 {x['to']} 와 같아 한 번만 사용" for x in d["merged"]]
        notes += [f"{x['requested_as']}: 제외 ({x['reason']})" for x in d["excluded"]]
        return d["used"], notes

    def remap_overrides(self, requested: list[str], overrides: dict) -> tuple[dict, list[str]]:
        """설정의 변수별 재정의(variable_transforms·lag_override)를 대표 ID 기준으로 옮긴다.
        중복(duplicate_of)으로 치환된 변수에 원래 ID 로 적은 재정의가 있으면 대표 ID 에 적용한다(2026-11-07 — 그 전에는 버려져
        마스터 기본값이 쓰였다. 예: repro_biva_orig 의 L1:none → USDKRW 가 마스터 기본 pct 로 들어감).
        대표 ID 로도 적혀 있으면 대표 ID 쪽을 쓰고 안내한다. 치환과 무관한 키(대상 변수 등)는 그대로 둔다."""
        out = dict(overrides or {})
        notes: list[str] = []
        for vid in requested or []:
            rep = self.resolve(vid)
            if rep == vid or vid not in out:
                continue
            val = out.pop(vid)
            if rep in (overrides or {}):
                if overrides[rep] != val:
                    notes.append(f"{vid}(→{rep}) 의 재정의 {val!r} 는 {rep} 에 직접 적은 {overrides[rep]!r} 와 달라 {rep} 쪽을 씀")
                continue
            out[rep] = val
            notes.append(f"{vid} 에 적은 재정의 {val!r} 를 대표 {rep} 에 적용")
        return out, notes

    def validate(self) -> list[str]:
        """마스터 일관성 검사. 오류 문자열 목록 (비어 있으면 통과)."""
        errs: list[str] = []
        for v in self._vars.values():
            d = v.duplicate_of
            if not d:
                continue
            if d == v.id:
                errs.append(f"{v.id}: duplicate_of 가 자기 자신")
            elif d not in self._vars:
                errs.append(f"{v.id}: duplicate_of={d} 가 마스터에 없음")
            elif self._vars[d].duplicate_of:
                errs.append(f"{v.id}: duplicate_of={d} 도 중복 행 (체인 금지 — 대표를 직접 가리킬 것)")
        return errs

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([v.model_dump() for v in self._vars.values()]).set_index("id")

    # -- 저장/로드 (CSV; DB 도입 전까지) ------------------------------------
    def save_csv(self, path: str | Path) -> None:
        self.to_frame().to_csv(path, encoding="utf-8-sig")

    @classmethod
    def load_csv(cls, path: str | Path) -> "Registry":
        raw = Path(path).read_bytes()
        df = pd.read_csv(io.BytesIO(raw), index_col="id", dtype=str, keep_default_na=False)
        reg = cls()
        reg.source_path = str(path)
        reg.source_sha256, reg.source_bytes = hashlib.sha256(raw).hexdigest(), raw
        for vid, row in df.iterrows():
            d = row.to_dict()
            d["id"] = str(vid)
            d["lag_days"] = int(float(d.get("lag_days") or 0))
            # active 빈칸 = Y(도움말 2-3절의 기본값). 전에는 빈칸이 꺼짐으로 읽혀, 손으로 추가한 행이 실험에서 말없이 빠질 수 있었다(2026-11-10)
            d["active"] = _to_bool(d["active"]) if str(d.get("active") or "").strip() else True
            d["lag_confirmed"] = _to_bool(d.get("lag_confirmed")) if str(d.get("lag_confirmed") or "").strip() else False  # 빈칸 = 확인 안 됨(2026-11-09)
            reg.add(Variable(**{k: v for k, v in d.items() if k in Variable.model_fields}))
        return reg

    # -- 변수 사전 엑셀 (data/variable_master.xlsx) --------------------------
    @classmethod
    def load_master(cls, path: str | Path, *, sheet: str = "variables") -> tuple["Registry", list[str]]:
        """사용자가 관리하는 단일 변수 사전을 읽는다. (레지스트리, 경고 목록).

        - 엔진이 쓰는 열: id, group, domain, freq, lag_days, source, source_code, default_fill, duplicate_of, active
        - lag_days 빈칸 → 0 으로 두고 경고 ('미확인')
        - validate() 오류는 예외로 올린다 (잘못된 중복 지정은 조용히 넘어가지 않는다)
        """
        raw = Path(path).read_bytes()  # 한 번만 읽는다 — 해시·사본·해석이 모두 같은 바이트에서 나오게
        # keep_default_na=False: 'NA'·'N/A' 같은 글자를 빈칸으로 바꾸지 않는다(id 가 NA 인 변수가 사라지던 문제)
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, dtype=str, keep_default_na=False).fillna("")
        df.columns = [str(c).strip() for c in df.columns]
        if "id" not in df.columns:
            raise ValueError(f"{path}: 'id' 열이 없습니다")
        reg = cls()
        reg.source_path = str(path)
        reg.source_sha256, reg.source_bytes = hashlib.sha256(raw).hexdigest(), raw
        warns: list[str] = []
        for _, r in df.iterrows():
            d = {k: str(r[k]).strip() for k in df.columns if k in Variable.model_fields}
            if not d.get("id"):
                continue
            if not d.get("lag_days"):
                fq = _FREQ_MAP.get(str(d.get("freq", "M")).strip(), "M")
                d["lag_days"] = LAG_DEFAULT[fq]
                d["lag_confirmed"] = False
                warns.append(f"{d['id']}: lag_days 빈칸 → 주기 {fq} 기본값 {LAG_DEFAULT[fq]}일 (미확인)")
            else:
                d["lag_days"] = int(float(d["lag_days"]))
                # lag_confirmed 는 Y 라고 적은 것만 확인됨. 빈칸(또는 열 없음) = 확인 안 됨(2026-11-09 사용자 결정 — 전에는 빈칸 = Y)
                d["lag_confirmed"] = _to_bool(d["lag_confirmed"]) if str(d.get("lag_confirmed") or "").strip() else False
            d["active"] = _to_bool(d["active"]) if str(d.get("active") or "").strip() else True  # 빈칸 = Y(위 load_csv 와 같은 규칙)
            for k in ("source", "default_fill", "transform", "freq"):
                if k in d and not d[k]:
                    d.pop(k)  # 빈칸은 기본값
            reg.add(Variable(**d))
        errs = reg.validate()
        if errs:
            raise ValueError("variable_master 오류:\n  " + "\n  ".join(errs))
        return reg, warns


def append_master_rows(path: str | Path, rows: list[dict], *, sheet: str = "variables") -> int:
    """마스터 엑셀 끝에 행을 추가한다 (업로드 파일에서 새 열을 발견했을 때). 서식·다른 시트는 건드리지 않는다.
    이미 있는 id 는 건너뛴다. 추가한 행 수를 돌려준다."""
    from openpyxl import load_workbook

    wb = load_workbook(path)
    ws = wb[sheet]
    hdr = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
    if "id" not in hdr:
        raise ValueError("마스터에 id 열이 없습니다")
    existing = {str(r[hdr.index("id")]).strip() for r in ws.iter_rows(min_row=2, values_only=True) if r[hdr.index("id")]}
    n = 0
    for row in rows:
        if row.get("id") in existing:
            continue
        ws.append([row.get(h, "") for h in hdr])
        existing.add(row["id"])
        n += 1
    if n:
        wb.save(path)
    return n


def update_master_row(path: str | Path, vid: str, changes: dict, *, sheet: str = "variables") -> None:
    """마스터 엑셀에서 id 가 vid 인 줄의 칸들을 고친다(2026-11-13, 화면에서 발표시차·변환 고치기). 서식·다른 줄·다른 시트는 건드리지 않는다.
    머리글에 없는 칸(예: 오래된 사전에 note 열이 없음)은 머리글 끝에 열을 만든다. 없는 id 면 KeyError."""
    from openpyxl import load_workbook

    wb = load_workbook(path)
    ws = wb[sheet]
    hdr = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
    if "id" not in hdr:
        raise ValueError("마스터에 id 열이 없습니다")
    for k in changes:
        if k not in hdr:
            ws.cell(row=1, column=len(hdr) + 1, value=k)
            hdr.append(k)
    col_id = hdr.index("id") + 1
    for r in range(2, ws.max_row + 1):
        if str(ws.cell(row=r, column=col_id).value or "").strip() == vid:
            for k, val in changes.items():
                ws.cell(row=r, column=hdr.index(k) + 1, value=val)
            wb.save(path)
            return
    raise KeyError(vid)


def _to_bool(s: object) -> bool:
    return str(s).strip().lower() in ("true", "1", "y", "yes")


# ---------------------------------------------------------------------------
# 기존 엑셀(df_var_info) 흡수
# ---------------------------------------------------------------------------
class IngestReport(BaseModel):
    """엑셀 업로드 검증 보고서 — 거부하지 않고 사실만 적는다 (기획서 3-6)."""

    file: str
    sheets_read: list[str]
    sheets_ignored: list[str]
    variables_added: int
    duplicate_ids: list[str] = []
    missing_in_data: list[str] = []
    data_not_in_var_info: list[str] = []
    freq_mismatch: list[str] = []
    notes: list[str] = []


def ingest_var_info(
    xlsx: str | Path, registry: Registry | None = None, *, source: Source = "XLSX"
) -> tuple[Registry, IngestReport]:
    """`df_var_info` 시트를 읽어 레지스트리에 넣는다. 데이터 시트(df_D/df_W/df_M/df_Q)는 열 이름 대조만 한다.

    - 기존 열: ID, Group, Variable, description, 단위, FEQ, LAG(월), transform
    - 선택 열(있으면 사용): source, source_code, lag_days
    - 중복 ID: 첫 행을 쓰고 나머지는 보고서에 기록
    """
    xlsx = Path(xlsx)
    reg = registry or Registry()
    xl = pd.ExcelFile(xlsx)
    known = {"df_var_info", "df_D", "df_W", "df_M", "df_Q", "df_Y"}
    sheets_read = [s for s in xl.sheet_names if s in known]
    sheets_ignored = [s for s in xl.sheet_names if s not in known]
    rep = IngestReport(file=xlsx.name, sheets_read=sheets_read, sheets_ignored=sheets_ignored, variables_added=0)
    if "df_var_info" not in xl.sheet_names:
        rep.notes.append("df_var_info 시트 없음 — 데이터 시트 열 이름으로만 등록(메타 없음)")
        info = pd.DataFrame()
    else:
        info = xl.parse("df_var_info", dtype=str)
        info.columns = [str(c).strip() for c in info.columns]
        if "Group" in info.columns:  # 엑셀 병합셀: 그룹명이 첫 행에만 있음 → 아래로 채움
            info["Group"] = info["Group"].ffill()
        info = info.fillna("")

    # 데이터 시트의 열 → 주기
    data_cols: dict[str, str] = {}
    for sh, fq in (("df_D", "D"), ("df_W", "W"), ("df_M", "M"), ("df_Q", "Q"), ("df_H", "H"), ("df_Y", "Y")):
        if sh in xl.sheet_names:
            cols = [str(c) for c in xl.parse(sh, nrows=0).columns if str(c).lower() != "date"]
            for c in cols:
                data_cols.setdefault(c, fq)

    added = 0
    seen: set[str] = set()
    for _, r in info.iterrows():
        vid = str(r.get("ID", "")).strip()
        if not vid:
            continue
        if vid in seen or vid in reg:
            rep.duplicate_ids.append(vid)
            continue
        seen.add(vid)
        freq_raw = str(r.get("FEQ", "")).strip() or data_cols.get(vid, "M")
        lag_days = r.get("lag_days", "")
        if str(lag_days).strip():
            lag = int(float(lag_days))
        else:
            try:
                lag = int(float(r.get("LAG", "0") or 0)) * _MONTH_DAYS
            except ValueError:
                lag = 0
        sheet = {"D": "df_D", "W": "df_W", "M": "df_M", "Q": "df_Q", "H": "df_H", "Y": "df_Y"}.get(_FREQ_MAP.get(freq_raw, "M"), "df_M")
        v = Variable(
            id=vid,
            name=str(r.get("Variable", "")),
            description=str(r.get("description", "")),
            group=str(r.get("Group", "")).replace("\n", ""),
            freq=freq_raw,
            lag_days=lag,
            transform=str(r.get("transform", "") or "none"),
            source=str(r.get("source", "") or source),  # type: ignore[arg-type]
            source_code=str(r.get("source_code", "") or f"{sheet}!{vid}"),
            unit=str(r.get("단위", "")),
        )
        if vid in data_cols and data_cols[vid] != v.freq:
            rep.freq_mismatch.append(f"{vid}: FEQ={v.freq}, 시트={data_cols[vid]}")
        if vid not in data_cols:
            rep.missing_in_data.append(vid)
        reg.add(v)
        added += 1
    for c in data_cols:
        if c not in reg:
            rep.data_not_in_var_info.append(c)
    rep.variables_added = added
    return reg, rep
