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
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from adapters.base import RawCache  # noqa: E402
from engine.derived import compute_derived  # noqa: E402
from engine.registry import Registry, append_master_rows  # noqa: E402
from engine.store import DataVersion, excel_to_long  # noqa: E402

from engine import tickers as tickers_mod  # noqa: E402

TICKERS = [t["id"] for t in tickers_mod.DEFAULT]  # 기본 종목(파일이 없을 때). 실제 목록은 data/tickers.json — main() 에서 읽는다
MASTER = "variable_master.xlsx"




def main() -> int:
    """종료 코드: 0 = 모두 받음 · 1 = 종목 가격을 새로 받지 못함(받은 것이 아예 없으면 새 버전도 만들지 않음) ·
    2 = 받지 못한 항목이 있음(그 항목은 직전 값, 버전은 '일부 받지 못함') · 3 = 시작하기 전에 멈춤(변수 사전 없음·엑셀 고칠 곳·없는 --version id).
    (argparse 가 잘못된 인자에 쓰는 2 는 화면이 인자를 정해 주므로 생기지 않는다.)"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--sources", default="datago,yf,pykrx,fred,ecos,krx,xlsx,derived", help="쉼표 구분")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--refresh", action="store_true", help="--version 재사용 시 이미 있는 변수도 다시 받는다")
    ap.add_argument("--version", default=None, help="기존 버전 ID 를 주면 그 폴더에 이어서 받고(실패한 출처 재시도) 가격을 다시 조립한다")
    ap.add_argument("--allow-shrink", action="store_true", help="새로 받은 자료가 직전 버전보다 짧아도(줄 수 감소·시작일이 늦어짐) 그대로 쓴다. 기본은 직전 값을 두고 실패로 기록")
    a = ap.parse_args()
    sources = {s.strip() for s in a.sources.split(",")}
    data_dir = Path(a.data_dir)
    master_path = data_dir / MASTER
    if not master_path.exists():
        print(f"변수 사전이 없습니다: {master_path}")
        return 3  # 시작하기 전에 멈춤(아무것도 받지 않음) — '받지 못한 항목 있음'(2)과 구분(2026-11-13 독립 검토 5)

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
        return 3  # 시작하기 전에 멈춤(아무것도 받지 않음) — '받지 못한 항목 있음'(2)과 구분(2026-11-13 독립 검토 5)
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

    carry_from = None
    if a.version and not (data_dir / "versions" / a.version / "manifest.json").exists():
        # --version 은 '있는 버전에 이어 받기'다. 없는 id(오타)를 주면 새 버전 폴더가 조용히 생겼다(독립 검토 F22)
        print(f"데이터 버전 {a.version} 이(가) 없습니다 — --version 은 이미 있는 버전에 이어 받을 때만 씁니다. 새 버전을 만들려면 --version 을 빼세요.")
        return 3  # 시작하기 전에 멈춤(아무것도 받지 않음) — '받지 못한 항목 있음'(2)과 구분(2026-11-13 독립 검토 5)
    if not a.version:
        # 새 버전은 '직전 버전의 전부'에서 시작하고, 이번에 받는 데 성공한 것만 새 값으로 바꾼다(2026-11-10 독립 검토 F1).
        # 그 전에는 요청한 출처는 이어받지 않아, 그 출처가 실패하면(접속 불가·키 없음) 자료가 빠진 버전이 '최신'이 되고 종료 코드도 0 이었다
        # — 가격이 2007년부터가 아니라 2020년부터인 버전으로 실험이 조용히 돌았다(폴드 149 → 23).
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
            if tag in price_src.values():  # 출처별 원 가격 파일 전부(요청한 출처도 — 받는 데 성공하면 새 파일로 바뀐다)
                shutil.copy2(f, dv.dir / f.name); carried[f.name] = _sha(f)
        for f in sorted(src_dir.glob("variables_*.parquet")):
            key = f.stem[len("variables_"):]
            fam = "xlsx" if key.startswith("xlsx") else key
            if fam == "xlsx" and "xlsx" in sources:
                continue  # 엑셀 자료는 사전 파일에서 그대로 다시 읽는다(통신이 없어 '받지 못함'이 없다. 읽지 못하면 위의 엑셀 검사에서 이미 멈춘다)
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
        print(f"  [이어받음] 직전 버전 {carry_from} 의 자료 {len(carried)}개 파일에서 시작합니다 — 이번에 받는 데 성공한 것만 새 값으로 바뀝니다")
    log: dict[str, str] = dict(prev.get("collect_log", {}))
    run_log: dict[str, str] = {}   # 이번 실행에서 시도한 것만(collect_runs 에 덧붙임)
    items: dict[str, dict] = {}    # 변수별 받은 줄 수·첫날·마지막 날(이번 실행)
    superseded: list[str] = []     # 이번 실행으로 읽지 않게 된 옛 엑셀 자료 파일
    kept_previous: dict[str, str] = {}  # 받지 못해(또는 받은 것이 더 짧아) 직전 버전 값을 그대로 둔 항목 → 이유
    source_notes: dict[str, str] = {}   # 실패는 아니지만 남겨 둘 처리(예: 교차검증 출처의 이동 창을 직전 자료와 이음)
    prev_start = None
    if carry_from:
        try:
            prev_start = json.loads((data_dir / "versions" / carry_from / "manifest.json").read_text(encoding="utf-8")).get("start")
        except Exception:  # noqa: BLE001
            prev_start = None
    # 줄어듦 검사: 수집 시작일(--start)이 직전과 같을 때만 뜻이 있다(시작일을 늦추면 짧아지는 것이 정상)
    check_shrink = bool(carry_from) and not a.allow_shrink and (prev_start is None or str(prev_start) == str(a.start))

    def _shrunk(new_n: int, new_first, old_n: int, old_first) -> str | None:
        """새로 받은 것이 직전보다 짧으면 이유 글자, 아니면 None."""
        if not check_shrink or not old_n:
            return None
        if new_n < old_n:
            return f"받은 줄 수 {new_n} < 직전 {old_n}"
        if new_first is not None and old_first is not None and pd.Timestamp(new_first) > pd.Timestamp(old_first):
            return f"받은 자료의 시작 {pd.Timestamp(new_first).date()} 이 직전 {pd.Timestamp(old_first).date()} 보다 늦음"
        return None
    t_start = pd.Timestamp.now(tz="UTC").isoformat()
    print(f"데이터 버전 {dv.id} → {dv.dir}")

    def attempt(name, fn):
        try:
            fn()
            log[name] = run_log[name] = "ok"
            print(f"  [ok] {name}")
        except Exception as e:  # noqa: BLE001
            # 오류 문구에는 요청 주소(접속 키가 든)가 들어 있을 수 있다 → 키를 지운 글자만 기록·출력한다(adapters.base.scrub_secrets)
            from adapters.base import scrub_secrets
            why = scrub_secrets(f"{e.__class__.__name__}: {e}")
            log[name] = run_log[name] = f"fail: {why}"
            print(f"  [fail] {name}: {why}")
            print("    " + scrub_secrets(traceback.format_exc(limit=1)).strip().replace("\n", "\n    "))

    already = set(dv.read_variables()["variable_id"].unique()) if a.version and not a.refresh else set()
    already -= set(prev.get("kept_previous") or {})  # 지난번에 받지 못해 직전 값을 그대로 둔 변수는 '이미 있음'이 아니다 — 다시 받는다

    def collect_rows(vars_, fetch, src_name):
        rows = []
        got: set[str] = set()
        pf = dv.dir / f"variables_{src_name}.parquet"
        prev_df = pd.read_parquet(pf) if (carry_from and pf.exists()) else None  # 직전 버전에서 이어받은 이 출처의 자료
        existing = dv.read_variables([v.id for v in vars_ if v.id in already]) if already else None
        if existing is not None and not existing.empty:
            rows.append(existing)
            print(f"  [유지] {src_name}: 이미 있는 {existing['variable_id'].nunique()}개 변수는 다시 받지 않음 (--refresh 로 강제)")
        for v in vars_:
            if v.id in already:
                continue
            def f(v=v):
                s = fetch(v)
                ss = s.dropna()
                if prev_df is not None:
                    old = prev_df[prev_df["variable_id"] == v.id].dropna(subset=["value"])
                    why = _shrunk(len(ss), ss.index.min() if len(ss) else None, len(old), old["date"].min() if len(old) else None)
                    if why:
                        raise RuntimeError(f"받은 자료가 직전 버전보다 짧습니다({why}) — 직전 값을 그대로 둡니다. 일부러 줄인 것이면 --allow-shrink")
                rows.append(pd.DataFrame({"variable_id": v.id, "date": s.index, "value": s.values}))
                got.add(v.id)
                items[v.id] = {"source": v.source, "source_code": v.source_code, "n": int(len(ss)),
                               "first": str(pd.Timestamp(ss.index.min()).date()) if len(ss) else None,
                               "last": str(pd.Timestamp(ss.index.max()).date()) if len(ss) else None}
            attempt(f"{src_name} {v.id} ({v.source_code})", f)
        if prev_df is not None:
            # 이번에 받지 못한 변수는 직전 버전의 값을 그대로 둔다(빠뜨리지 않는다). 무엇을 그대로 두었는지는 기록한다
            for v in vars_:
                if v.id in got or v.id in already:
                    continue
                old = prev_df[prev_df["variable_id"] == v.id]
                if len(old):
                    rows.append(old[["variable_id", "date", "value"]])
                    kept_previous[v.id] = run_log.get(f"{src_name} {v.id} ({v.source_code})", "받지 못함")
        if rows:
            dv.write_variables(pd.concat(rows), source=src_name)

    def write_prices_checked(name: str, df):
        """받은 가격이 직전 버전의 같은 출처 파일보다 짧으면 쓰지 않고 오류(직전 파일이 그대로 남는다)."""
        pf = dv.dir / f"prices_{name}.parquet"
        if carry_from and pf.exists():
            old = pd.read_parquet(pf)
            why = _shrunk(len(df), df.index.min() if len(df) else None, len(old), old.index.min() if len(old) else None)
            if why and name.endswith("_naver") and len(df):
                # 네이버 수정주가(빠진 날 메우기·교차검증용)는 '최근 N일'만 주는 이동 창이라 받을 때마다 시작일이 밀린다 — 실패로 보지 않는다(2026-11-12).
                # 겹치는 날을 보고 셋 중 하나(2026-11-13 독립 검토 13):
                #  ① 값이 같다 → 직전 앞부분 + 새 자료로 잇는다.
                #  ② 시가·고가·저가·종가가 모두 '같은 배수'로 달라졌다(배당·분할로 과거 전체를 다시 조정 — 수정주가의 정의) →
                #     직전 앞부분에 그 배수를 곱해 새 기준으로 맞춘 뒤 잇는다(전에는 직전 파일을 그대로 둬, 다음 실행에서도 또 달라 영영 갱신되지 않았다).
                #     거래량은 고치지 않는다(빠진 날 메우기는 가격 비율만 쓴다).
                #  ③ 배수가 날마다·열마다 다르거나 겹치는 날이 없다 → 설명할 수 없으니 직전 파일을 그대로 두고 그 이유를 적는다.
                common = old.index.intersection(df.index)
                cols = [c for c in ("open", "high", "low", "close") if c in old.columns and c in df.columns]
                o_, n_ = old.loc[common, cols].to_numpy(float), df.loc[common, cols].to_numpy(float)
                key = f"prices {name}"
                if len(common) == 0:
                    source_notes[key] = f"이동 창({why}) — 직전 파일과 겹치는 날이 없어 이을 수 없었다(직전 파일을 그대로 두었다)"
                elif np.allclose(o_, n_, rtol=1e-9, atol=1e-9, equal_nan=True):
                    merged = pd.concat([old[old.index < df.index.min()], df]).sort_index()
                    dv.write_prices(name, merged[~merged.index.duplicated(keep="last")])
                    source_notes[key] = f"이동 창({why}) — 겹치는 {len(common)}일 값이 같아 직전 앞부분에 새 자료를 이었다"
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = n_ / o_
                    fin = ratio[np.isfinite(ratio)]
                    fac = float(np.median(fin)) if len(fin) else float("nan")
                    if len(fin) == ratio.size and np.isfinite(fac) and fac > 0 and np.all(np.abs(fin / fac - 1) < 1e-6):
                        pre = old[old.index < df.index.min()].copy()
                        pre[cols] = pre[cols] * fac
                        merged = pd.concat([pre, df]).sort_index()
                        dv.write_prices(name, merged[~merged.index.duplicated(keep="last")])
                        source_notes[key] = (f"이동 창({why}) — 겹치는 {len(common)}일의 가격이 모두 같은 배수 {fac:.6g} 로 바뀌어(수정주가 소급 조정) "
                                             f"직전 앞부분({len(pre)}일)에 그 배수를 곱해 새 기준으로 맞춘 뒤 이었다")
                    else:
                        source_notes[key] = (f"이동 창({why}) — 겹치는 {len(common)}일 값이 날마다·열마다 다른 비율로 바뀌어(배수 {np.nanmin(fin) if len(fin) else float('nan'):.4g}~"
                                             f"{np.nanmax(fin) if len(fin) else float('nan'):.4g}) 설명할 수 없다 — 직전 파일을 그대로 두었다")
                print(f"  [참고] {name}: {source_notes[key]}")
                return
            if why:
                kept_previous[f"prices {name}"] = why
                raise RuntimeError(f"받은 가격이 직전 버전보다 짧습니다({why}) — 직전 값을 그대로 둡니다. 일부러 줄인 것이면 --allow-shrink")
        dv.write_prices(name, df)

    def no_key(src: str, env: str, what: str):
        """요청한 출처인데 접속 키가 없다 — 조용히 건너뛰지 않고 '받지 못함'으로 기록한다(직전 버전 값은 그대로)."""
        msg = f"fail: .env 에 {env} 가 없어 받지 못함"
        log[f"{src} ({what})"] = run_log[f"{src} ({what})"] = msg
        print(f"  [fail] {src}: .env 파일에 {env} 가 없어 {what}을(를) 받지 못했습니다" + (" — 직전 버전 값을 그대로 둡니다" if carry_from else ""))

    # 1) 가격 — 출처별로 전부 받아 둔다 (prices_<t>_datago / _yf / _naver). 2) 에서 합친다.
    prices_done: set[str] = set()
    tick_rows = tickers_mod.load(data_dir)
    TICKERS = [t["id"] for t in tick_rows]                       # noqa: N806 — 종목 목록 파일(data/tickers.json)
    ETFS = [t["id"] for t in tick_rows if t["kind"] == "etf"]    # noqa: N806 — 공식 출처는 ETF 시세 API 라 ETF 에만 묻는다
    have: dict[str, set[str]] = {t: set() for t in TICKERS}
    if "datago" in sources and not os.environ.get("DATAGO_API_KEY"):
        no_key("datago", "DATAGO_API_KEY", "공식 가격(2020~)")
    if "datago" in sources and os.environ.get("DATAGO_API_KEY"):
        from adapters import datago_adapter as dg

        for t in ETFS:
            def f(t=t):
                write_prices_checked(f"{t}_datago", dg.fetch_prices(t, a.start, a.end, cache=RawCache(data_dir, "datago")))
                have[t].add("datago")
            attempt(f"datago prices {t} (공식, 2020~)", f)
    if "yf" in sources:
        from adapters import yf_adapter as yf

        for t in TICKERS:
            def f(t=t):
                write_prices_checked(f"{t}_yf", yf.fetch_prices(t, a.start, a.end, cache=RawCache(data_dir, "yf")))
                have[t].add("yf")
            attempt(f"yf prices {t} (원주가, 2009~; 공식과 종가 100% 일치 확인 2026-11-03)", f)
        collect_rows(by_src.get("YF", []), lambda v: yf.fetch_series(v.source_code, a.start, a.end, cache=RawCache(data_dir, "yf")), "yf")
    if "pykrx" in sources:
        from adapters import pykrx_adapter as pk

        for t in TICKERS:
            def f(t=t):
                write_prices_checked(f"{t}_naver", pk.fetch_prices(t, a.start, a.end, cache=RawCache(data_dir, "pykrx")))
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
    if "fred" in sources and by_src.get("FRED") and not os.environ.get("FRED_API_KEY"):
        no_key("fred", "FRED_API_KEY", f"FRED 변수 {len(by_src['FRED'])}개")
    if "ecos" in sources and by_src.get("ECOS") and not os.environ.get("ECOS_API_KEY"):
        no_key("ecos", "ECOS_API_KEY", f"ECOS 변수 {len(by_src['ECOS'])}개")
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
            no_key("krx", "KRX_ID/KRX_PW", f"KRX 변수 {len(by_src['KRX'])}개")
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
    # 요청한 항목 가운데 받지 못한 것(지금까지의 실행을 통틀어 마지막 상태가 실패인 것). 하나라도 있으면 이 버전은 '일부 받지 못함'(incomplete)이다:
    # 실험이 데이터 버전을 적지 않았을 때 자동으로 고르는 '최신'에서 빠지고(engine.store.latest_version_id), 종료 코드는 2.
    failed_items = sorted(k for k, v in log.items() if str(v).startswith("fail"))
    incomplete = bool(failed_items)  # 값이 하나도 없는 변수(missing)는 사전의 상태이지 '받기 실패'가 아니다 — 위에서 경고만 한다
    run.update(kept_previous=kept_previous, notes=source_notes, failed_items=sorted(k for k, v in run_log.items() if str(v).startswith("fail")))
    dv.write_manifest({"collect_log": log, "collect_runs": list(prev.get("collect_runs", [])) + [run],
                       "start": a.start, "end": a.end, "tickers": sorted(prices_done),
                       "master": MASTER, "master_sha256": run["master_sha256"], "variables_collected": len(collected), "variables_missing": missing,
                       "incomplete": incomplete, "failed_items": failed_items, "kept_previous": kept_previous, "notes": source_notes})
    print("완료. manifest:", dv.dir / "manifest.json")
    if incomplete:
        from engine.store import latest_version_id
        use = latest_version_id(data_dir)
        print(f"\n[주의] 받지 못한 항목이 {len(failed_items)}개 있습니다.")
        for k in failed_items[:20]:
            print(f"  · {k}: {str(log[k])[:160]}")
        if kept_previous:
            print(f"  → 그중 {len(kept_previous)}개는 직전 버전({carry_from})의 값을 그대로 두었습니다: {sorted(kept_previous)[:15]}")
        print(f"  이 버전({dv.id})은 '일부 받지 못함'으로 표시되어, 데이터 버전을 적지 않은 실험은 이 버전을 쓰지 않습니다"
              + (f"(지금 자동으로 쓰이는 버전: {use})." if use and use != dv.id else "(온전한 버전이 없어 이 버전이 쓰입니다)."))
        print(f"  받지 못한 원인(접속·키)을 고친 뒤 같은 버전에 이어 받으세요: python collect.py --version {dv.id}")
        return 2
    return 0 if prices_done else 1


if __name__ == "__main__":
    sys.exit(main())
