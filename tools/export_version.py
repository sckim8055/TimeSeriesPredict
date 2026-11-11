"""데이터 버전의 값을 엑셀로 내보내 보기(보기 전용) — API 로 받은 값까지 한 파일에서 눈으로 확인할 때.

사용:  python tools/export_version.py            # 가장 최근 데이터 버전
       python tools/export_version.py 20261103T141116Z
결과:  data/exports/<버전>_보기전용.xlsx
       - '안내' 시트: 이 파일은 프로그램이 읽지 않는다는 안내, 버전·만든 시각
       - '변수' 시트: id·이름·출처·주기·첫날·마지막 날·값 개수
       - 가격_<종목> 시트: 실험이 읽는 가격(조립본)
       - 주기별 시트(일간·주간·월간·분기·반기·연간): 날짜 × 변수 값
이 파일을 고쳐도 아무 데도 반영되지 않는다. 엑셀로 넣을 값은 variable_master.xlsx 의 데이터 시트에 넣는다.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402

FREQ_SHEET = {"D": "일간", "W": "주간", "M": "월간", "Q": "분기", "H": "반기", "Y": "연간"}


def export(version: str | None = None, data_dir: Path = ROOT / "data") -> Path:
    vers = sorted(p.name for p in (data_dir / "versions").glob("*") if (p / "manifest.json").exists())
    if not vers:
        raise SystemExit("데이터 버전이 없습니다 — 먼저 collect.py 를 실행하세요")
    vid = version or vers[-1]
    if vid not in vers:
        raise SystemExit(f"데이터 버전 {vid} 가 없습니다. 있는 버전: {', '.join(vers[-5:])}")
    dv = DataVersion(data_dir, vid)
    reg, _ = Registry.load_master(data_dir / "variable_master.xlsx")
    long = dv.read_variables()
    out_dir = data_dir / "exports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{vid}_보기전용.xlsx"
    rows = []
    for vid_, g in long.dropna(subset=["value"]).groupby("variable_id"):
        v = reg.get(vid_) if vid_ in reg else None
        rows.append({"id": vid_, "이름": v.name if v else "", "출처": v.source if v else "(사전에 없음)", "주기": v.freq if v else "",
                     "첫날": g["date"].min().date(), "마지막 날": g["date"].max().date(), "값 개수": len(g)})
    info = pd.DataFrame({"항목": ["안내", "데이터 버전", "만든 시각(UTC)"],
                         "내용": ["보기 전용 파일입니다. 프로그램은 이 파일을 읽지 않습니다 — 고쳐도 반영되지 않습니다. "
                                "엑셀로 넣을 값은 data/variable_master.xlsx 의 데이터 시트(df_D … df_Y)에 넣으세요.",
                                vid, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")]})
    with pd.ExcelWriter(out) as xw:
        info.to_excel(xw, sheet_name="안내", index=False)
        pd.DataFrame(rows).to_excel(xw, sheet_name="변수", index=False)
        for pf in sorted(dv.dir.glob("prices_*.parquet")):
            t = pf.stem[len("prices_"):]
            if "_" in t:  # 출처별 원본(prices_<t>_datago 등)은 빼고 실험이 읽는 조립본만
                continue
            pr = pd.read_parquet(pf)
            pr.index = pd.to_datetime(pr.index).date
            pr.rename_axis("date").to_excel(xw, sheet_name=f"가격_{t}"[:31])
        for fq, name in FREQ_SHEET.items():
            ids = [r["id"] for r in rows if r["주기"] == fq]
            if ids:
                w = dv.wide(ids)
                w.index = w.index.date
                w.rename_axis("date").to_excel(xw, sheet_name=name)
    print(f"보기 전용 엑셀: {out}")
    return out


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else None)
