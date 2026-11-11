"""KRX 정보데이터시스템에서 내려받은 '투자자별 거래실적(일별)' CSV 들 → 변수 사전 data/variable_master.xlsx 의 df_D 시트 열.

사용:
  python tools/krx_csv_to_upload.py <CSV 폴더> 091160
  → data/variable_master.xlsx 의 df_D 시트에 열 FLOW_FOREIGN_091160, FLOW_INST_091160, FLOW_INDIV_091160 을 넣는다
    (2026-11-09 부터 직접 넣는 데이터는 변수 사전 파일 하나에 둔다 — data/uploads/ 는 읽지 않음). 같은 이름의 열이 있으면 바꿔 쓴다.
  이후 python collect.py --sources xlsx 로 새 데이터 버전에 흡수(받지 않는 출처는 직전 버전에서 이어받음 — 2026-11-10. variables 시트에 행이 없으면 자동 등록).
  같은 변수가 variables 에 KRX(API)로 등록되어 있으면 source 를 XLSX 로 바꾼다(한 변수 = 한 출처 — 이제 CSV 값을 씀).
  엑셀이 열려 있으면 저장이 실패하니 닫고 실행.

KRX CSV 는 EUC-KR 인 경우가 많고, 열 이름이 화면마다 조금씩 다르다('외국인', '외국인합계', '기관', '기관합계', '개인').
숫자는 천단위 쉼표 포함 문자열. 날짜 열은 '일자' 또는 '날짜'.
값이 '순매수 거래대금' 인지 확인한다 — 화면에서 '거래대금/순매수' 로 받았어야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

COLMAP = {"외국인": "FOREIGN", "외국인합계": "FOREIGN", "기관": "INST", "기관합계": "INST", "개인": "INDIV"}


def read_krx_csv(p: Path) -> pd.DataFrame:
    for enc in ("euc-kr", "cp949", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(p, encoding=enc, thousands=",")
            break
        except Exception:  # noqa: BLE001
            df = None
    if df is None:
        raise RuntimeError(f"CSV 를 읽지 못함: {p}")
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if c in ("일자", "날짜", "date", "Date")), None)
    if date_col is None:
        raise RuntimeError(f"{p.name}: 날짜 열('일자'/'날짜')이 없음. 열: {list(df.columns)}")
    df["date"] = pd.to_datetime(df[date_col].astype(str).str.replace("/", "-"), errors="coerce")
    df = df.dropna(subset=["date"])
    out = pd.DataFrame({"date": df["date"]})
    found = []
    for c in df.columns:
        key = c.replace(" ", "")
        if key in COLMAP and COLMAP[key] not in found:
            out[COLMAP[key]] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
            found.append(COLMAP[key])
    missing = [k for k in ("FOREIGN", "INST", "INDIV") if k not in found]
    if missing:
        raise RuntimeError(f"{p.name}: 열 부족 {missing}. 받은 열: {list(df.columns)}")
    return out


def main(csv_dir: str, ticker: str, master: str = "data/variable_master.xlsx") -> Path:
    files = sorted(Path(csv_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"CSV 가 없습니다: {csv_dir}")
    parts = [read_krx_csv(f) for f in files]
    df = pd.concat(parts).drop_duplicates("date", keep="last").sort_values("date").set_index("date")
    df.columns = [f"FLOW_{c}_{ticker}" for c in df.columns]
    out = Path(master)
    from openpyxl import load_workbook

    wb = load_workbook(out)
    old = pd.read_excel(out, sheet_name="df_D") if "df_D" in wb.sheetnames else pd.DataFrame(columns=["date"])
    old.columns = [("date" if str(c).strip().lower() == "date" else str(c).strip()) for c in old.columns]  # 'Date'·앞뒤 공백 허용
    if "date" not in old.columns:
        raise SystemExit(f"{out} 의 df_D 시트 맨 왼쪽 열 제목이 'date' 여야 합니다")
    if len(old.columns):
        old["date"] = pd.to_datetime(old["date"])
        old = old.set_index("date").drop(columns=[c for c in df.columns if c in old.columns])
    merged = old.join(df, how="outer").sort_index() if len(old) else df
    pos = wb.sheetnames.index("df_D") if "df_D" in wb.sheetnames else None
    if "df_D" in wb.sheetnames:
        del wb["df_D"]
    ws = wb.create_sheet("df_D", pos)  # 원래 자리(시트 순서 유지)
    ws.append(["date", *merged.columns])
    for d, r in merged.iterrows():
        ws.append([d.to_pydatetime(), *[None if pd.isna(x) else float(x) for x in r.values]])
    # 한 변수 = 한 출처: 이 열들의 값은 이제 엑셀(CSV)에서 온다. variables 에 KRX(API)로 등록되어 있으면 XLSX 로 바꾼다.
    # 그대로 두면 collect 가 '같은 이름의 API 변수'로 보고 이 열을 읽지 않는다(engine/master_check).
    switched = []
    vs = wb["variables"]
    hdr = [str(c.value).strip() if c.value is not None else "" for c in vs[1]]
    ci, cs, cc = hdr.index("id"), hdr.index("source"), hdr.index("source_code")
    for row in vs.iter_rows(min_row=2):
        vid = str(row[ci].value or "").strip()
        if vid in df.columns and str(row[cs].value or "").strip() != "XLSX":
            switched.append(f"{vid}({row[cs].value} → XLSX)")
            row[cs].value, row[cc].value = "XLSX", f"df_D!{vid}"
    wb.save(out)
    if switched:
        print("  [알림] 이 변수들은 이제 엑셀(CSV) 값을 씁니다:", ", ".join(switched))
    gaps = df.index.to_series().diff().dt.days
    print(f"{len(files)}개 파일 → {out} (df_D 시트): {df.index.min().date()}~{df.index.max().date()} {len(df)}행, 7일 이상 빈 구간 {int((gaps > 7).sum())}곳")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
