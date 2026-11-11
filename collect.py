"""데이터 수집 — 노트북(또는 네트워크가 열린 환경)에서 실행.

수집 목록의 유일한 기준은 `data/variable_master.xlsx` 다. 여기 적힌 source/source_code 대로 가져온다.
  - YF     : source_code = 야후 심볼 (KRW=X, ^SOX, 005930.KS ...)
  - FRED   : source_code = 시리즈 ID (DGS10 ...)
  - ECOS   : source_code = 통계코드/항목코드/주기 (722Y001/0101000/M)
  - XLSX   : 변수 사전 파일 자체의 데이터 시트(df_D/df_W/df_M/df_Q/df_H/df_Y)의 열 (source_code = 시트!열, 예: df_M!B1).
             2026-11-09 부터 직접 넣는 데이터도 variable_master.xlsx 한 파일에 둔다 — data/uploads/ 의 엑셀은 읽지 않는다.
             새 열은 variables 에 자동 등록, API 변수와 같은 이름의 열은 읽지 않고 API 값을 씀(한 변수 = 한 출처).
             프로그램이 짐작할 수 없는 경우(같은 열이 두 곳 등)만 멈춤. 규칙 전체는 engine/master_check.py
  - KRX    : 투자자별 수급 (source_code = flow:티커:투자자). .env 에 KRX_ID/KRX_PW 필요
  - DERIVED: 다른 변수로 계산 (source_code = 식). 수집이 끝난 뒤 계산
  - active=N 이나 duplicate_of 가 있는 행도 수집은 한다 (데이터는 두고, 실험에서만 제외)
가격 우선순위(2026-11-03 확정): 공공데이터포털(공식, 2020~) → 그 이전은 Yahoo 원주가(공식과 종가 100% 일치 확인) → 결측 거래일은 네이버로 비율 보정.
  실험이 읽는 파일은 prices_<ticker>.parquet 하나. 출처별 원본은 prices_<ticker>_datago/_yf/_naver 로 보존.

사용법 (프로젝트 루트에서):
  python collect.py                 # .env 의 키로 가능한 출처 전부 → data/versions/<id>/
  python collect.py --sources yf    # 특정 출처만
  python collect.py --start 2006-01-01
  python collect.py --sources datago --version 20261103T141116Z   # 실패한 출처만 기존 버전에 다시 받고 가격 재조립

출처별 필요 키(.env): DATAGO_API_KEY(공공데이터포털), ECOS_API_KEY, FRED_API_KEY. yf·pykrx 는 키 없음.
결과: data/versions/<id>/prices_<ticker>.parquet, variables_<source>.parquet, registry.csv(마스터 사본), manifest.json
API 원본 응답: data/raw/<출처>/<키>_<UTC>.json (받을 때마다 새 파일, 지우지 않음) · 받은 이력: data/raw/fetch_log.jsonl (한 줄 = 한 번 받기, 추가만)
manifest.json 의 collect_runs = 이 버전에 수집을 실행한 기록(같은 버전에 이어 받으면 덧붙임 — 덮어쓰지 않음)
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from adapters.base import RawCache  # noqa: E402
from engine.derived import compute_derived  # noqa: E402
from engine.registry import Registry, append_master_rows  # noqa: E402
from engine.store import DataVersion, excel_to_long  # noqa: E402

TICKERS = ["091160", "091230"]  # KODEX 반도체, TIGER 반도체
MASTER = "variable_master.xlsx"




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--sources", default="datago,yf,pykrx,fred,ecos,krx,xlsx,derived", help="쉼표 구분")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--refresh", action="store_true", help="--version 재사용 시 이미 있는 변수도 다시 받는다")
    ap.add_argument("--version", default=None, help="기존 버전 ID 를 주면 그 폴더에 이어서 받고(실패한 출처 재시도) 가격을 다시 조립한다")
    a = ap.parse_args()
    sources = {s.strip() for s in a.sources.split(",")}
    data_dir = Path(a.data_dir)
    master_path = data_dir / MASTER
    if not master_path.exists():
        print(f"변수 사전이 없습니다: {master_path}")
        return 2

    # 0) 변수 사전의 데이터 시트 스캔 → variables 보강 → 마스터 로드. data/uploads/ 는 2026-11-09 부터 읽지 않는다(있으면 경고)
    stale = [f.name for f in sorted((data_dir / "uploads").glob("*.xlsx")) if not f.name.startswith("~$")] if (data_dir / "uploads").exists() else []
    if stale:
        print(f"  [경고] data/uploads/ 의 엑셀 {stale} 은 읽지 않습니다 — 직접 넣는 데이터는 {MASTER} 의 데이터 시트(df_D/df_M …)에 넣으세요")
    # 한 변수 = 한 출처(engine/master_check). 판단은 프로그램이 한다: 새 열은 자동 등록, API 로 받는 변수의 엑셀 열은 쓰지 않음.
    # 엑셀 모양이 잘못되어 짐작할 수 없을 때만 멈춘다 — 그때는 엑셀도 고치지 않고 아무것도 받지 않는다
    from engine.master_check import check_master, near_duplicates
    reg, warns = Registry.load_master(master_path)
    chk = check_master(reg, master_path)
    if chk.errors:
        print(f"엑셀 파일 {MASTER} 에서 고칠 곳이 {len(chk.errors)}군데 있습니다. 고친 뒤 다시 실행해 주세요(아무것도 받지 않았습니다).")
        for e in chk.errors:
            print("  ·", e)
        return 2
    if chk.new_vars:  # 새 열 자동 등록 → 다시 읽기(발표시차 빈칸 → 주기별 보수적 기본값 · 확인 안 됨)
        append_master_rows(master_path, chk.new_vars)
        reg, warns = Registry.load_master(master_path)
    for n in chk.notices:
        print("  [알림]", n)
    for w in warns:
        print("  [마스터]", w)
    by_src: dict[str, list] = {}
    for vid in reg.ids(usable_only=False):
        by_src.setdefault(reg.get(vid).source, []).append(reg.get(vid))
    print("마스터:", len(reg), "변수 /", {k: len(v) for k, v in by_src.items()})

    ALL_SOURCES = {"datago", "yf", "pykrx", "fred", "ecos", "krx", "xlsx", "derived"}
    carry_from = None
    if not a.version and not ALL_SOURCES <= sources:
        # 일부 출처만 받는 새 버전: 나머지는 직전 버전에서 이어받는다. 전에는 받은 것만 든 버전이 '최신'이 되어
        # (예: --sources xlsx → 가격 없는 버전) 그 뒤 실험이 돌지 않았다(2026-11-10 점검 N6)
        done = sorted(p.name for p in (data_dir / "versions").glob("*") if (p / "manifest.json").exists()) if (data_dir / "versions").exists() else []
        carry_from = done[-1] if done else None
        if carry_from and "derived" not in sources:
            sources.add("derived")  # 파생 변수는 입력이 바뀌었을 수 있으니 항상 다시 계산
    dv = DataVersion(data_dir, a.version)
    dv.dir.mkdir(parents=True, exist_ok=True)
    prev = dv.manifest() if a.version and (dv.dir / "manifest.json").exists() else {}
    if prev:
        print(f"  [경고] 이미 완성된 버전 {dv.id} 을 고칩니다 — 이 버전으로 낸 결과의 재현 근거(manifest 지문)가 달라집니다. 새 버전을 만들려면 --version 을 빼세요.")
    carried: dict[str, str] = {}
    if carry_from:
        import shutil
        from engine.store import sha256_file as _sha

        src_dir = data_dir / "versions" / carry_from
        file_src = {"yf": "YF", "fred": "FRED", "ecos": "ECOS", "krx": "KRX", "derived": "DERIVED"}
        price_src = {"datago": "datago", "yf": "yf", "pykrx": "naver"}
        for f in sorted(src_dir.glob("prices_*_*.parquet")):
            tag = f.stem.rsplit("_", 1)[-1]
            if any(price_src[k] == tag for k in price_src if k not in sources):
                shutil.copy2(f, dv.dir / f.name); carried[f.name] = _sha(f)
        for f in sorted(src_dir.glob("variables_*.parquet")):
            key = f.stem[len("variables_"):]
            fam = "xlsx" if key.startswith("xlsx") else key
            if fam in sources or fam == "derived":
                continue
            d = pd.read_parquet(f)
            want = "XLSX" if fam == "xlsx" else file_src.get(fam)
            # 사전에서 출처가 바뀌었거나 없어진 변수는 이어받지 않는다(한 변수 = 한 출처)
            keep = d[d["variable_id"].map(lambda v: v in reg and reg.get(v).source == want)] if want else d
            if len(keep) == len(d):
                shutil.copy2(f, dv.dir / f.name)
            elif len(keep):
                keep.to_parquet(dv.dir / f.name, index=False)
            if len(keep):
                carried[f.name] = _sha(f) + ("" if len(keep) == len(d) else f" (변수 {d['variable_id'].nunique() - keep['variable_id'].nunique()}개 제외)")
        print(f"  [이어받음] 받지 않는 출처의 자료 {len(carried)}개 파일을 직전 버전 {carry_from} 에서 가져왔습니다")
    log: dict[str, str] = dict(prev.get("collect_log", {}))
    run_log: dict[str, str] = {}   # 이번 실행에서 시도한 것만(collect_runs 에 덧붙임)
    items: dict[str, dict] = {}    # 변수별 받은 줄 수·첫날·마지막 날(이번 실행)
    superseded: list[str] = []     # 이번 실행으로 읽지 않게 된 옛 엑셀 자료 파일
    t_start = pd.Timestamp.now(tz="UTC").isoformat()
    print(f"데이터 버전 {dv.id} → {dv.dir}")

    def attempt(name, fn):
        try:
            fn()
            log[name] = run_log[name] = "ok"
            print(f"  [ok] {name}")
        except Exception as e:  # noqa: BLE001
            log[name] = run_log[name] = f"fail: {e.__class__.__name__}: {e}"
            print(f"  [fail] {name}: {e}")
            traceback.print_exc(limit=1)

    already = set(dv.read_variables()["variable_id"].unique()) if a.version and not a.refresh else set()

    def collect_rows(vars_, fetch, src_name):
        rows = []
        existing = dv.read_variables([v.id for v in vars_ if v.id in already]) if already else None
        if existing is not None and not existing.empty:
            rows.append(existing)
            print(f"  [유지] {src_name}: 이미 있는 {existing['variable_id'].nunique()}개 변수는 다시 받지 않음 (--refresh 로 강제)")
        for v in vars_:
            if v.id in already:
                continue
            def f(v=v):
                s = fetch(v)
                rows.append(pd.DataFrame({"variable_id": v.id, "date": s.index, "value": s.values}))
                ss = s.dropna()
                items[v.id] = {"source": v.source, "source_code": v.source_code, "n": int(len(ss)),
                               "first": str(pd.Timestamp(ss.index.min()).date()) if len(ss) else None,
                               "last": str(pd.Timestamp(ss.index.max()).date()) if len(ss) else None}
            attempt(f"{src_name} {v.id} ({v.source_code})", f)
        if rows:
            dv.write_variables(pd.concat(rows), source=src_name)

    # 1) 가격 — 출처별로 전부 받아 둔다 (prices_<t>_datago / _yf / _naver). 2) 에서 합친다.
    prices_done: set[str] = set()
    have: dict[str, set[str]] = {t: set() for t in TICKERS}
    if "datago" in sources and os.environ.get("DATAGO_API_KEY"):
        from adapters import datago_adapter as dg

        for t in TICKERS:
            def f(t=t):
                dv.write_prices(f"{t}_datago", dg.fetch_prices(t, a.start, a.end, cache=RawCache(data_dir, "datago")))
                have[t].add("datago")
            attempt(f"datago prices {t} (공식, 2020~)", f)
    if "yf" in sources:
        from adapters import yf_adapter as yf

        for t in TICKERS:
            def f(t=t):
                dv.write_prices(f"{t}_yf", yf.fetch_prices(t, a.start, a.end, cache=RawCache(data_dir, "yf")))
                have[t].add("yf")
            attempt(f"yf prices {t} (원주가, 2009~; 공식과 종가 100% 일치 확인 2026-11-03)", f)
        collect_rows(by_src.get("YF", []), lambda v: yf.fetch_series(v.source_code, a.start, a.end, cache=RawCache(data_dir, "yf")), "yf")
    if "pykrx" in sources:
        from adapters import pykrx_adapter as pk

        for t in TICKERS:
            def f(t=t):
                dv.write_prices(f"{t}_naver", pk.fetch_prices(t, a.start, a.end, cache=RawCache(data_dir, "pykrx")))
                have[t].add("naver")
            attempt(f"pykrx prices {t} (네이버 수정주가, 교차검증용)", f)

    # 2) 주 시리즈 조립: datago(공식) 를 기준으로, 그 이전 기간은 yf 로 잇고, 남는 결측 거래일은 naver 로 비율 보정.
    #    결과: prices_<t>.parquet (= 실험이 읽는 파일), repair_log_<t>.csv (어느 날을 어느 출처로 메웠는지)
    from engine.repair import merge_prices

    for t in TICKERS:
        for src in ("datago", "yf", "naver"):  # 이전 실행에서 받아 둔 파일도 조립에 쓴다
            if (dv.dir / f"prices_{t}_{src}.parquet").exists():
                have[t].add(src)
        def f(t=t):
            parts = []
            if "datago" in have[t]:
                d = dv.read_prices(f"{t}_datago")
                parts.append(("datago", d))
            if "yf" in have[t]:
                y = dv.read_prices(f"{t}_yf")
                if parts:
                    y = y[y.index < parts[0][1].index.min()]  # 공식 데이터 이전 기간만
                parts.append(("yf", y))
            if not parts:
                raise RuntimeError("가격 주 출처(datago/yf) 없음")
            main = pd.concat([p for _, p in parts]).sort_index()
            main = main[~main.index.duplicated(keep="first")]
            src_note = " + ".join(f"{n}({p.index.min().date()}~{p.index.max().date()}, {len(p)}일)" for n, p in parts if len(p))
            rep = pd.DataFrame()
            if "naver" in have[t]:
                main, rep = merge_prices(main, dv.read_prices(f"{t}_naver"))
            elif "yf" in have[t] and "datago" in have[t]:
                main, rep = merge_prices(main, dv.read_prices(f"{t}_yf"))
            dv.write_prices(t, main)
            rep.to_csv(dv.dir / f"repair_log_{t}.csv", index=False, encoding="utf-8-sig")
            prices_done.add(t)
            print(f"      {t}: {src_note}; 결측 거래일 보정 {len(rep)}일")
        attempt(f"assemble prices {t}", f)

    # 3) 변수
    if "fred" in sources and os.environ.get("FRED_API_KEY"):
        from adapters import fred_adapter as fr

        collect_rows(by_src.get("FRED", []), lambda v: fr.fetch_series(v.source_code, a.start, a.end, cache=RawCache(data_dir, "fred")), "fred")
    if "ecos" in sources and os.environ.get("ECOS_API_KEY"):
        from adapters import ecos_adapter as ec

        def fetch_ecos(v):
            from engine.api_spec import ecos_period  # 기간 문자열 규칙은 한 곳(변수 사전의 '데이터 받는 법'과 같음)

            st, it, fq = v.source_code.split("/")
            # 2026-11-06 수정: 분기(Q)는 ECOS 가 2006Q1 형식을 요구하는데 '%Y'(2006)로 보내고 있었다(현재 분기 변수는 없어 영향 없음)
            return ec.fetch_series(st, it, fq, ecos_period(pd.Timestamp(a.start), fq), ecos_period(pd.Timestamp(a.end or pd.Timestamp.today()), fq), cache=RawCache(data_dir, "ecos"))

        collect_rows(by_src.get("ECOS", []), fetch_ecos, "ecos")
    if "krx" in sources and by_src.get("KRX"):
        if os.environ.get("KRX_ID") and os.environ.get("KRX_PW"):
            from adapters import krx_adapter as kx

            collect_rows(by_src.get("KRX", []), lambda v: kx.fetch_series(v.source_code, a.start, a.end, cache=RawCache(data_dir, "krx")), "krx")
        else:
            print("  [건너뜀] KRX 변수가 마스터에 있지만 .env 에 KRX_ID/KRX_PW 가 없습니다")
    if "xlsx" in sources:
        def fx():
            long = excel_to_long(master_path, reg)
            dv.write_variables(long, source="xlsx_master")
            # 같은 버전에 남아 있는 옛 엑셀 자료 파일은 읽히지 않게 이름을 바꾼다 — 전에는 새 파일과 함께 읽혀 엑셀에서 지운 값이 계속 쓰였다(2026-11-10 점검 N7)
            for old_f in sorted(dv.dir.glob("variables_xlsx_*.parquet")):
                if old_f.name != "variables_xlsx_master.parquet":
                    old_f.rename(old_f.with_name(old_f.name + ".superseded"))
                    superseded.append(old_f.name)
                    print(f"      옛 엑셀 자료 {old_f.name} → 쓰지 않음(.superseded)")
            for vid, g in long.dropna(subset=["value"]).groupby("variable_id"):
                items[vid] = {"source": "XLSX", "source_code": reg.get(vid).source_code, "n": int(len(g)),
                              "first": str(pd.Timestamp(g["date"].min()).date()), "last": str(pd.Timestamp(g["date"].max()).date())}
        attempt(f"xlsx {MASTER} (데이터 시트)", fx)
    if "derived" in sources:
        def f():
            long, notes = compute_derived(reg, dv)
            for n in notes:
                print("    ", n)
            if not long.empty:
                dv.write_variables(long, source="derived")
        attempt("derived", f)

    # 4) 레지스트리 사본을 버전에 포함 (실험 재현용: 그때의 lag_days·중복 지정)
    reg.save_csv(dv.dir / "registry.csv")
    reg.save_csv(data_dir / "registry.csv")
    collected = set(dv.read_variables()["variable_id"].unique())
    src_map = {"yf": "YF", "fred": "FRED", "ecos": "ECOS", "krx": "KRX", "xlsx": "XLSX", "derived": "DERIVED"}
    requested = {src_map[s] for s in sources if s in src_map}
    missing = [v for v in reg.ids(usable_only=False) if v not in collected and reg.get(v).source in requested]
    if missing:
        print(f"  [경고] 요청한 출처인데 수집 안 된 변수 {len(missing)}개: {missing}")
    if not collected and not prices_done:
        import shutil

        shutil.rmtree(dv.dir, ignore_errors=True)
        print("수집된 것이 없어 데이터 버전을 만들지 않았습니다.")
        return 1
    from engine.store import sha256_file
    import importlib.metadata as _md
    try:
        near = near_duplicates(dv, reg)  # 수집한 값 기준 '거의 같은 시계열'(관리자 참고)
    except Exception as e:  # noqa: BLE001 — 참고용 검사가 실패해도 수집 기록(manifest)은 남긴다
        near = [f"검사 실패: {type(e).__name__}: {e}"]
    for w in near:
        print("  [관리자 참고]", w)
    # 한 변수가 두 자료 파일에 있으면 어느 값이 쓰일지 정해지지 않는다 — 있으면 기록하고 알린다
    seen_in: dict[str, list[str]] = {}
    for f in sorted(dv.dir.glob("variables_*.parquet")):
        for vid in pd.read_parquet(f, columns=["variable_id"])["variable_id"].unique():
            seen_in.setdefault(str(vid), []).append(f.name)
    multi = {k: v for k, v in seen_in.items() if len(v) > 1}
    if multi:
        print(f"  [경고] 두 자료 파일에 함께 있는 변수 {len(multi)}개: {dict(list(multi.items())[:5])} — 한 출처만 남기세요")
    from engine import runlog as _runlog
    try:
        msnap = _runlog.snapshot_master(master_path, data_dir, raw=reg.source_bytes, sha=reg.source_sha256)  # 수집 때의 사전 사본(나중에 비교·재현용)
    except OSError as e:
        msnap = {"sha256": reg.source_sha256, "snapshot": None, "error": str(e)}

    def _ver(p):
        try:
            return _md.version(p)
        except _md.PackageNotFoundError:
            return None
    run = {"started_utc": t_start, "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(), "sources": sorted(sources),
           "start": a.start, "end": a.end, "refresh": bool(a.refresh), "master_sha256": msnap["sha256"], "master_snapshot": msnap.get("snapshot"),
           "carried_from": ({"version": carry_from, "files": carried} if carry_from else None), "superseded": superseded, "multi_source_ids": multi,
           "library_versions": {p: _ver(p) for p in ("pandas", "yfinance", "pykrx", "fredapi", "requests", "openpyxl")},
           "log": run_log, "items": items, "checks": {**chk.as_dict(), "near_duplicates": near}}
    dv.write_manifest({"collect_log": log, "collect_runs": list(prev.get("collect_runs", [])) + [run],
                       "start": a.start, "end": a.end, "tickers": sorted(prices_done),
                       "master": MASTER, "master_sha256": run["master_sha256"], "variables_collected": len(collected), "variables_missing": missing})
    print("완료. manifest:", dv.dir / "manifest.json")
    return 0 if prices_done else 1


if __name__ == "__main__":
    sys.exit(main())
