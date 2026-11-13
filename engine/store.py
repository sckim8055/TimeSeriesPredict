"""데이터 저장소 — 모든 출처의 변수 값을 한 가지 모양(롱 테이블)으로 보관한다 (기획서 3-6, 9장).

롱 테이블: columns = [variable_id, date, value]. 주기 정보는 레지스트리가 가지고 있으므로 시트 구분이 사라진다.
수집 1회 = 데이터 버전 1개 (`data/versions/<version_id>/`). 실험은 버전 ID를 기록해 재현한다.

가격(OHLCV)은 변수와 별도 테이블로 둔다: `prices_<ticker>.parquet`, columns = [date, open, high, low, close, volume].
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.registry import Registry

LONG_COLS = ["variable_id", "date", "value"]
PRICE_COLS = ["open", "high", "low", "close", "volume"]


def list_versions(data_root: str | Path) -> list[dict]:
    """완성된(manifest.json 이 있는) 데이터 버전 목록, 오래된 순. incomplete = 수집 때 요청한 항목 일부를 받지 못한 버전(받지 못한 항목은 직전 버전 값 그대로)."""
    out = []
    vd = Path(data_root) / "versions"
    for p in sorted(vd.glob("*")) if vd.exists() else []:
        m = p / "manifest.json"
        if not m.exists():
            continue
        try:
            j = json.loads(m.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            j = {}
        out.append({"id": p.name, "incomplete": bool(j.get("incomplete")), "manifest": j})
    return out


def latest_version_id(data_root: str | Path) -> str | None:
    """설정에 데이터 버전을 적지 않았을 때 쓰는 '최신' 버전: 가장 새 '온전한' 버전. 일부를 받지 못한 버전(incomplete)은 자동으로 고르지 않는다 —
    그런 버전을 쓰려면 설정에 그 id 를 직접 적는다. 온전한 버전이 하나도 없으면 가장 새 버전(2026-11-10 독립 검토 F1)."""
    vs = list_versions(data_root)
    ok = [v["id"] for v in vs if not v["incomplete"]]
    return ok[-1] if ok else (vs[-1]["id"] if vs else None)


class DataVersion:
    """`data/versions/<id>/` 디렉터리 하나."""

    def __init__(self, root: str | Path, version_id: str | None = None):
        self.root = Path(root)
        if version_id is not None and not str(version_id).strip():
            raise ValueError("데이터 버전 id 가 비어 있습니다 — 새 버전을 만들려면 id 를 주지 말고(None), 있는 버전을 읽으려면 그 id 를 주세요")  # 빈 글자가 조용히 새 버전이 되던 것(F22)
        self.id = version_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if any(c in self.id for c in "/\\:") or ".." in self.id or not self.id.strip():
            raise ValueError(f"데이터 버전 id 가 올바르지 않습니다: {self.id!r}")  # 경로 문자가 든 id 로 저장소 밖에 폴더가 생기던 것 방지(2026-11-10 점검 S12)
        self.dir = self.root / "versions" / self.id
        if version_id is None:
            self.dir.mkdir(parents=True, exist_ok=True)  # 새 버전만 만든다. 있는 버전을 '읽으려고' 연 것만으로 폴더가 생기지 않게(오타 id → 빈 버전 폴더)

    # -- 변수 ------------------------------------------------------------------
    def write_variables(self, long_df: pd.DataFrame, *, source: str) -> Path:
        long_df = _validate_long(long_df)
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"variables_{source}.parquet"
        long_df.to_parquet(path, index=False)
        return path

    def read_variables(self, ids: list[str] | None = None) -> pd.DataFrame:
        parts = [pd.read_parquet(p) for p in sorted(self.dir.glob("variables_*.parquet"))]
        if not parts:
            return pd.DataFrame(columns=LONG_COLS)
        df = pd.concat(parts, ignore_index=True)
        if ids is not None:
            df = df[df["variable_id"].isin(ids)]
        return df.sort_values(["variable_id", "date"]).reset_index(drop=True)

    def wide(self, ids: list[str] | None = None) -> pd.DataFrame:
        """롱 → 와이드(date × variable_id). 주기가 섞여 있으면 빈 칸은 NaN (정렬은 calendar.align_to_calendar가 담당)."""
        df = self.read_variables(ids)
        if df.empty:
            return pd.DataFrame()
        return df.pivot_table(index="date", columns="variable_id", values="value", aggfunc="last").sort_index()

    # -- 가격 ------------------------------------------------------------------
    def write_prices(self, ticker: str, ohlcv: pd.DataFrame) -> Path:
        df = ohlcv.copy()
        df.columns = [str(c).lower() for c in df.columns]
        missing = [c for c in PRICE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"가격 데이터에 열이 없습니다: {missing}")
        df = df[PRICE_COLS].astype(float)
        df.index = pd.DatetimeIndex(df.index).normalize()
        df.index.name = "date"
        df = df[~df.index.duplicated(keep="last")].sort_index()
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"prices_{ticker}.parquet"
        df.to_parquet(path)
        return path

    def read_prices(self, ticker: str) -> pd.DataFrame:
        return pd.read_parquet(self.dir / f"prices_{ticker}.parquet")

    # -- 메타 ------------------------------------------------------------------
    def write_manifest(self, extra: dict | None = None) -> Path:
        files = sorted(p.name for p in self.dir.glob("*.parquet"))
        checksums = {f: _sha256(self.dir / f) for f in files}
        now = datetime.now(timezone.utc).isoformat()
        old = self.manifest() if (self.dir / "manifest.json").exists() else {}
        manifest = {
            "version_id": self.id,
            # 처음 만든 시각은 유지하고, 이어 받은 시각은 updated_utc 로(2026-11-09 — 전에는 이어 받으면 created_utc 를 덮어씀)
            "created_utc": old.get("created_utc", now),
            "updated_utc": now,
            "files": checksums,
            **(extra or {}),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def manifest(self) -> dict:
        return json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))


def _validate_long(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in LONG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"롱 테이블 열 누락: {missing}")
    out = df[LONG_COLS].copy()
    out["variable_id"] = out["variable_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"]).drop_duplicates(["variable_id", "date"], keep="last")
    return out.sort_values(["variable_id", "date"]).reset_index(drop=True)


def sha256_file(path: str | Path) -> str:
    """파일 내용의 sha256 (manifest·수집 기록용)."""
    return _sha256(Path(path))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 엑셀 데이터 시트 → 롱 테이블. 2026-11-09 부터 읽는 엑셀은 data/variable_master.xlsx 하나(데이터 시트 df_D/df_W/df_M/df_Q/df_H/df_Y)
# ---------------------------------------------------------------------------
DATA_SHEETS = (("df_D", "D"), ("df_W", "W"), ("df_M", "M"), ("df_Q", "Q"), ("df_H", "H"), ("df_Y", "Y"))
_MISSING_TEXT = {"", "-", "–", "—", ".", "na", "n/a", "nan", "null", "none", "#n/a"}
_DATE_PATTERNS = [
    (re.compile(r"^(\d{4})$"), lambda m: (int(m[1]), 1, 1)),
    (re.compile(r"^(\d{4})[-/.](\d{1,2})$"), lambda m: (int(m[1]), int(m[2]), 1)),
    (re.compile(r"^(\d{4})(\d{2})$"), lambda m: (int(m[1]), int(m[2]), 1)),
    (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\.?$"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})[ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?$"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),  # 글자로 저장된 '2020-01-31 00:00:00'
    (re.compile(r"^(\d{4})\s*년\s*(\d{1,2})\s*월$"), lambda m: (int(m[1]), int(m[2]), 1)),
    (re.compile(r"^(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일$"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"^(\d{4})\s*년$"), lambda m: (int(m[1]), 1, 1)),
    (re.compile(r"^(\d{4})\s*[-/]?\s*[Qq]([1-4])$"), lambda m: (int(m[1]), 3 * int(m[2]), 1)),
    (re.compile(r"^(\d{4})\s*년\s*([1-4])\s*분기$"), lambda m: (int(m[1]), 3 * int(m[2]), 1)),
]


def parse_date_cell(x) -> pd.Timestamp | None:
    """데이터 시트의 날짜 칸 하나를 날짜로. 못 읽으면 None(조용히 버리지 않고 호출한 쪽이 문제로 보고한다).

    받는 꼴: 엑셀 날짜 · 2020-01-31 · 2020/1/31 · 2020.01.31 · 20200131 · 2020-01 · 202001 · 2020 · 2020년 1월(1일) · 2020Q1 · 2020년 1분기.
    월·연 단위는 그 기간의 첫날(2024-03 → 2024-03-01, 2024 → 2024-01-01). 분기는 도움말 2-4절의 약속대로 분기 마지막 달의 1일(2024Q1 → 2024-03-01, 지금 A1 자료의 규약).
    숫자로 적힌 2019 는 연도, 202001 은 연월, 20200131 은 날짜로 읽는다 — 전에는 pandas 가 1970-01-01 뒤의 나노초로 읽어 모든 줄이 한 날짜가 됐다(2026-11-10 점검 N1).
    43831 같은 엑셀 일련번호는 날짜인지 값인지 알 수 없어 읽지 않는다(칸 서식을 날짜로 바꾸면 읽힌다). 1900~2200 년 밖은 읽지 않는다."""
    import datetime as _dt

    if x is None:
        return None
    if isinstance(x, _dt.time):
        return None  # 시각만 있는 칸은 날짜가 아니다
    if isinstance(x, (pd.Timestamp, _dt.datetime, _dt.date)):
        t = pd.Timestamp(x)
        return t.normalize() if 1900 <= t.year <= 2200 else None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (x != x or x != int(x)):
            return None
        x = str(int(x))
    s = str(x).strip()
    for rx, f in _DATE_PATTERNS:
        m = rx.match(s)
        if m:
            y, mo, d = f(m)
            if not (1900 <= y <= 2200):
                return None
            try:
                return pd.Timestamp(year=y, month=mo, day=d)
            except ValueError:
                return None
    # 위의 꼴에 맞지 않으면 읽지 않는다. 2026-11-10 전에는 여기서 pandas 가 짐작해 읽었는데, 뜻이 둘인 글자를 조용히 받아들였다(독립 검토 F6):
    # '20-01-31' → 2031-01-20, '12:00'·'now'·'today' → 오늘 날짜, '1Q20' → 2020-01-01(분기 약속과 다름), '1/2/2020' 은 1월 2일인데 '31/01/2020' 은 1월 31일.
    return None


def parse_value_cell(x) -> tuple[float | None, bool]:
    """값 칸 하나 → (숫자 또는 None, 문제 여부). 빈칸·'-'·'N/A' 는 결측(문제 아님). '1,234.5' 는 1234.5.
    '3.5%' 처럼 뜻이 둘로 읽히는 글자(3.5 인가 0.035 인가)나 그 밖의 글자는 문제로 돌려준다 — 짐작하지 않는다."""
    if x is None:
        return None, False
    if isinstance(x, bool):
        return None, True
    if isinstance(x, (int, float)):
        if x != x:
            return None, False
        return (None, True) if x in (float("inf"), float("-inf")) else (float(x), False)
    s = str(x).strip()
    if s.lower() in _MISSING_TEXT:
        return None, False
    t = s.replace(" ", "")
    # 쉼표는 '천 단위 구분'일 때만 받는다(1,234 · 12,345.67 · -1,234,567). '1,5'·'1.234,5'(소수점 쉼표)는 뜻이 둘이라 문제로 돌려준다 —
    # 2026-11-10 전에는 쉼표를 그냥 지워 '1,5' 가 15 로, '1.234,5' 가 1.2345 로 읽혔다(독립 검토 F19).
    if "," in t:
        if not re.fullmatch(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?", t):
            return None, True
        t = t.replace(",", "")
    try:
        v = float(t)
    except ValueError:
        return None, True
    return (None, True) if v != v or v in (float("inf"), float("-inf")) else (v, False)  # 'inf'·'nan' 글자는 값이 아니다


def read_data_sheets(xlsx: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """데이터 시트(df_D … df_Y)를 칸 단위로 읽는다. 반환 (롱 테이블[sheet, variable_id, date, value], 문제 목록).

    문제 목록의 kind: bad_date(값은 있는데 날짜를 못 읽음) · bad_value(글자 값) · formula_no_value(수식인데 계산값이 없음 — 프로그램이 엑셀을
    다시 저장하면 수식의 계산값이 사라진다. 엑셀에서 열어 다시 저장하거나 값으로 붙여 넣어야 한다) · dup_date(한 변수에 같은 날짜가 두 번).
    각 문제에 sheet·cell(예: C5)·column(변수 이름)·text(칸 내용)가 있다. 문제 있는 칸은 롱 테이블에 넣지 않는다 — 호출한 쪽이 멈출지 정한다.
    열 통계는 문제 목록 맨 끝의 kind='column_stats' 한 항목: numeric_cells {'시트!변수': 숫자 칸 수}, formula_cells ['시트!칸', …](수식이 든 칸)."""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    try:
        wf = openpyxl.load_workbook(xlsx, read_only=True, data_only=False)
    except Exception:  # noqa: BLE001
        wf = None
    rows_out: list[tuple] = []
    issues: list[dict] = []
    stats: dict[str, int] = {}
    formulas: list[str] = []
    try:
        for sh, _ in DATA_SHEETS:
            if sh not in wb.sheetnames:
                continue
            # 시트에 적힌 크기 표시(dimension)를 믿지 않고 실제 줄을 끝까지 읽는다 — 다른 프로그램이 내보낸 파일은 이 표시가 낡아 있을 수 있고,
            # 그러면 표시 밖의 줄을 조용히 버렸다(독립 검토 F5: 120줄 시트에 표시가 A1:B50 이면 49줄만 읽음).
            wb[sh].reset_dimensions()
            it = wb[sh].iter_rows(values_only=True)
            hdr = [None if h is None or str(h).strip() == "" else str(h).strip() for h in next(it, ())]
            di = next((i for i, h in enumerate(hdr) if h is not None and h.lower() == "date"), None)
            if di is None:
                continue
            if wf is not None and sh in wf.sheetnames:
                wf[sh].reset_dimensions()
            fit = wf[sh].iter_rows(min_row=2, values_only=True) if wf is not None and sh in wf.sheetnames else None
            seen: dict[tuple, int] = {}
            for rno, row in enumerate(it, start=2):
                frow = next(fit, None) if fit is not None else None
                vals = [(i, row[i]) for i in range(len(row)) if i != di and i < len(hdr) and hdr[i] is not None]
                has_formula = [(i, frow[i]) for i, _ in vals if frow is not None and i < len(frow) and isinstance(frow[i], str) and frow[i].startswith("=")]
                if all(v is None or str(v).strip() == "" for _, v in vals) and not has_formula:
                    continue  # 값이 없는 줄(날짜만 있거나 완전히 빈 줄)
                d = parse_date_cell(row[di] if di < len(row) else None)
                if d is None:
                    issues.append({"sheet": sh, "cell": f"{_col_letter(di + 1)}{rno}", "column": "date", "kind": "bad_date", "text": str(row[di] if di < len(row) else "")[:30]})
                    continue
                fmap = dict(has_formula)
                for i, v in vals:
                    cell = f"{_col_letter(i + 1)}{rno}"
                    if i in fmap:
                        formulas.append(f"{sh}!{cell}")
                    if v is None and i in fmap:
                        issues.append({"sheet": sh, "cell": cell, "column": hdr[i], "kind": "formula_no_value", "text": str(fmap[i])[:30]})
                        continue
                    x, bad = parse_value_cell(v)
                    if bad:
                        issues.append({"sheet": sh, "cell": cell, "column": hdr[i], "kind": "bad_value", "text": str(v)[:30]})
                        continue
                    if x is None:
                        continue
                    key = (hdr[i], d)
                    if key in seen:
                        issues.append({"sheet": sh, "cell": cell, "column": hdr[i], "kind": "dup_date", "text": f"{d.date()} (앞서 {seen[key]}행)"})
                        continue
                    seen[key] = rno
                    stats[f"{sh}!{hdr[i]}"] = stats.get(f"{sh}!{hdr[i]}", 0) + 1
                    rows_out.append((sh, hdr[i], d, x))
    finally:
        wb.close()
        if wf is not None:
            wf.close()
    issues.append({"kind": "column_stats", "numeric_cells": stats, "formula_cells": formulas})
    return pd.DataFrame(rows_out, columns=["sheet", "variable_id", "date", "value"]), issues


def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, rem = divmod(n - 1, 26)
        s = chr(65 + rem) + s
    return s


def excel_to_long(xlsx: str | Path, registry: Registry | None = None) -> pd.DataFrame:
    """df_D/df_W/df_M/df_Q/df_H/df_Y 시트를 읽어 롱 테이블로(다른 시트 — variables·설명·groups — 는 무시).
    칸 단위로 읽는다(read_data_sheets): 날짜는 parse_date_cell, 값은 parse_value_cell.
    읽을 변수(등록부를 주면 source=XLSX 변수만)에 못 읽는 날짜·글자 값·계산값 없는 수식·같은 날짜 중복이 있으면 ValueError — 전에는 그런 줄·칸을 조용히 버렸다(2026-11-10 점검 N1·N2).
    수집(collect)은 그 전에 master_check 가 같은 문제를 칸 주소와 함께 쉬운 말로 알려 주고 멈춘다."""
    long, issues = read_data_sheets(xlsx)
    keep = None
    if registry is not None:
        # source=XLSX 로 등록된 변수만. API 변수의 열은 읽지 않는다(한 변수 = 한 출처, engine/master_check)
        keep = {i for i in registry.ids(usable_only=False) if registry.get(i).source == "XLSX"}
        long = long[long["variable_id"].isin(keep)]
    bad = [x for x in issues if x["kind"] != "column_stats" and (x["kind"] == "bad_date" or keep is None or x["column"] in keep)]
    if bad:
        ex = "; ".join(f"{x['sheet']}!{x['cell']} {x['kind']} '{x['text']}'" for x in bad[:5])
        raise ValueError(f"{xlsx}: 데이터 시트에 읽을 수 없는 칸이 {len(bad)}개 있습니다 — {ex}")
    if long.empty:
        return pd.DataFrame(columns=LONG_COLS)
    return _validate_long(long[["variable_id", "date", "value"]].reset_index(drop=True))
