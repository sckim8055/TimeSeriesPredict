"""기존 데이터 버전의 변수 ID 를 마스터 ID 로 바꿔 새 버전을 만든다 (재수집 없이).

배경: 1차 수집(20261102T134628Z)은 C_/B_ 접두어 ID 를 썼고, variable_master.xlsx 는 접두어 없는 ID 를 쓴다.
사용: python tools/migrate_ids.py 20261102T134628Z
  → data/versions/<새 id>/ 에 가격은 그대로, 변수는 ID 치환 + DERIVED 계산 + registry.csv 포함.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.derived import compute_derived  # noqa: E402
from engine.registry import Registry  # noqa: E402
from engine.store import DataVersion  # noqa: E402

RENAME = {
    "C_USDKRW": "USDKRW", "C_KOSPI": "KOSPI", "B_SOX": "SOX", "B_NVDA": "NVDA", "B_MU": "MU", "B_TSM": "TSM",
    "B_SAMSUNG": "SAMSUNG", "B_HYNIX": "HYNIX", "C_US10Y_YF": "US10Y_YF", "C_VIX_YF": "VIX_YF",
    "C_US10Y": "US10Y", "C_VIX": "VIX", "C_FEDFUNDS": "FEDFUNDS", "C_US2Y": "US2Y",
    "C_BOK_RATE": "BOK_RATE", "C_KR10Y": "KR10Y", "C_KR3Y": "KR3Y",
}


def main(old_id: str, data_dir: Path = ROOT / "data") -> str:
    old = DataVersion(data_dir, old_id)
    new = DataVersion(data_dir)
    new.dir.mkdir(parents=True, exist_ok=True)
    reg, warns = Registry.load_master(data_dir / "variable_master.xlsx")
    for w in warns:
        print("  [마스터]", w)
    known = set(reg.ids(usable_only=False))
    for p in sorted(old.dir.glob("*")):
        if p.name.startswith("prices_") or p.name.startswith("repair_log_"):
            shutil.copy(p, new.dir / p.name)
        elif p.name.startswith("variables_"):
            df = pd.read_parquet(p)
            df["variable_id"] = df["variable_id"].map(lambda v: RENAME.get(v, v))
            unknown = sorted(set(df["variable_id"]) - known)
            if unknown:
                print(f"  [경고] {p.name}: 마스터에 없는 ID {unknown} → 제외")
                df = df[df["variable_id"].isin(known)]
            df.to_parquet(new.dir / p.name, index=False)
            print(f"  {p.name}: {df['variable_id'].nunique()}개 변수")
    long, notes = compute_derived(reg, new)
    for n in notes:
        print("   ", n)
    if not long.empty:
        new.write_variables(long, source="derived")
    reg.save_csv(new.dir / "registry.csv")
    collected = set(new.read_variables()["variable_id"].unique())
    missing = [v for v in reg.ids(usable_only=False) if v not in collected]
    new.write_manifest({"migrated_from": old_id, "rename": RENAME, "master": "variable_master.xlsx",
                        "variables_collected": len(collected), "variables_missing": missing, **{k: old.manifest().get(k) for k in ("start", "end", "tickers", "collect_log")}})
    print(f"새 버전 {new.id}: 변수 {len(collected)}개, 미수집 {missing}")
    return new.id


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "20261102T134628Z")
