"""검증 전 lag·변환으로 계산된 '낡은' 결과를 찾아 같은 설정으로 다시 돌린다.

낡음 판정: 결과에 기록된 변수별 lag(lag_days_used)·변환(transforms_used)이 지금 마스터(+그 설정의 재정의)로 계산한 값과 다르거나,
기록 자체가 없으면(2026-11-04 이전 결과) 낡은 것으로 본다. 창 입력 모델(LSTM·GRU·BIVA)은 결과의 engine_rev 가 2026-11-07 보다 앞서면
(창 눈금 수정 전) 낡은 것으로 본다. 기술지표만 쓰는 실험·규칙 모델은 변수 정렬과 무관하므로 제외.

  python tools/rerun_stale.py                 # 목록만(아무것도 돌리지 않음)
  python tools/rerun_stale.py --run           # CPU 로 빠른 것(Ridge·LightGBM·ARIMA·SARIMA)만 재실행
  python tools/rerun_stale.py --run --all     # LSTM·GRU·BIVA 포함(노트북 GPU 권장)
  python tools/rerun_stale.py --run --mode forecast

[2026-11-10 주의] 엔진 계산 방식이 2026-11-10 에 크게 바뀌어(봉인 경계·시드·SARIMA 기본·변환 등, engine.experiment.ENGINE_REV) 그 전 결과는 '전부' 옛 방식이다.
이 도구는 'lag·변환이 달라진 결과'만 찾고, ARIMA·SARIMA 는 '그때의 옛 기본값'으로 고정해 다시 돌린다 — 논문용 재실행에는 맞지 않는다.
논문용 재실행은 사전 분석 계획서(docs/PREREG_초안_2026-11.md)의 설정 목록으로 한다. 그래서 --run 은 --old-defaults 를 함께 줄 때만 돈다.

재실행은 정식 실험으로 기록된다(실험 카운터 +1) — 같은 가설을 데이터 교정 후 다시 본 것이므로 숨기지 않는 쪽이 정직하다.
옛 결과 파일은 지우지 않는다(비교용). 새 결과 이름 끝에 '_v1107' 을 붙여 구분한다(전에 붙인 '_v1105' 는 떼고).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.registry import Registry  # noqa: E402

try:  # 1차 실험(2026-11-02)의 옛 변수 ID(B_SOX 등) → 현재 ID
    from tools.migrate_ids import RENAME as OLD_IDS  # noqa: E402
except Exception:  # noqa: BLE001
    OLD_IDS = {}

SLOW = {"lstm", "gru", "biva"}
TAG, OLD_TAGS = "_v1107", ("_v1105",)
WINDOW_REV = "2026-11-07"  # 창 입력 모델의 눈금 계산이 바뀐 날(engine.experiment.ENGINE_REV)


BIVA_RENAME = {"epochs": "train_epochs", "lr": "learning_rate", "use_brits_impute": "use_brits"}
BIVA_DROPPED = {"w_pred", "w_vae", "w_imp", "grad_clip", "brits_always"}


def biva_params_now(mp: dict) -> dict:
    out = {}
    for k, v in mp.items():
        if k in BIVA_DROPPED:
            continue
        out[BIVA_RENAME.get(k, k)] = str(v) if k == "moving_avg" else v
    return out


def current_used(cfg: dict, reg: Registry) -> tuple[dict, dict]:
    ids, _ = reg.resolve_many(cfg.get("variable_ids") or [])
    lo, _ = reg.remap_overrides(cfg.get("variable_ids") or [], cfg.get("lag_override") or {})
    tr, _ = reg.remap_overrides(cfg.get("variable_ids") or [], cfg.get("variable_transforms") or {})
    return ({v: int(lo.get(v, reg.get(v).lag_days)) for v in ids}, {v: tr.get(v, reg.get(v).transform or "none") for v in ids})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--all", action="store_true", help="딥러닝 모델 포함")
    ap.add_argument("--mode", choices=["trading", "forecast", "both"], default="both")
    ap.add_argument("--match", default="", help="결과 파일 이름에 이 글자(쉼표로 여러 개)가 든 것만")
    ap.add_argument("--old-defaults", action="store_true", help="옛 기본값(ARIMA·SARIMA 추정 설정 등)으로 고정해 다시 돌리는 것임을 알고 실행")
    a = ap.parse_args()
    if a.run and not a.old_defaults:
        print("이 도구의 재실행은 ARIMA·SARIMA 를 '옛 기본값'으로 고정합니다 — 2026-11-10 엔진의 논문용 재실행에는 사전 분석 계획서의 설정 목록을 쓰세요.\n"
              "그래도 옛 기본값으로 다시 돌리려면 --old-defaults 를 함께 주세요. (목록만 보려면 --run 을 빼세요)")
        return
    reg, _ = Registry.load_master(ROOT / "data" / "variable_master.xlsx")
    stale = []
    newest: dict = {}  # (모드, 이름(태그 뗀 것), 모델) → 가장 최근 결과 파일의 시각(낡지 않은 것 포함)
    modes = ["trading", "forecast"] if a.mode == "both" else [a.mode]
    for mode in modes:
        for p in sorted((ROOT / "results" / mode).glob("*.json")):
            if p.name in ("experiment_counter.json", "sealed_log.json") or p.name.startswith("ablation_"):
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            cfg = r["config"]
            nm0 = cfg.get("name") or p.stem.rsplit("_", 1)[0]
            for t in (TAG, *OLD_TAGS):
                nm0 = nm0[: -len(t)] if nm0.endswith(t) else nm0
            k0 = (mode, nm0, r["model"].get("file"))
            newest[k0] = max(newest.get(k0, 0.0), p.stat().st_mtime)
            old_ids = [v for v in (cfg.get("variable_ids") or []) if v in OLD_IDS]
            if old_ids:  # 옛 ID 를 현재 ID 로 바꿔 같은 실험을 재현(설정의 재정의 키도 함께)
                cfg = dict(cfg)
                cfg["variable_ids"] = [OLD_IDS.get(v, v) for v in cfg["variable_ids"]]
                for key in ("lag_override", "variable_transforms"):
                    cfg[key] = {OLD_IDS.get(k, k): x for k, x in (cfg.get(key) or {}).items()}
            if r["model"].get("file") == "biva":  # 2026-11-06 BIVA 원본 복원: 옛 파라미터 이름 → 원본 이름, 원본에 없는 것은 버림
                cfg = dict(cfg)
                cfg["model_params"] = biva_params_now(cfg.get("model_params") or {})
            if r["model"].get("file") in ("arima", "sarima"):  # 2026-11-06 추정 제약 기본 변경: 옛 결과는 그때 값(False)으로 재현
                from engine.experiment import pinned_model_params
                cfg = dict(cfg)
                cfg["model_params"], _ = pinned_model_params(r["model"]["file"], r["model"].get("params"), cfg.get("model_params"))
            if a.match and not any(m_ in p.stem for m_ in a.match.split(",")):
                continue
            win_old = r["model"].get("input_shape") == "window" and (r.get("engine_rev") or "") < WINDOW_REV
            if cfg.get("name", "").endswith(TAG) or r["model"].get("input_shape") == "rule":
                continue
            if not cfg.get("variable_ids"):  # 기술지표만: 변수 정렬과 무관 — 창 모델이면 눈금 수정만 따진다
                if win_old:
                    stale.append((mode, p, cfg, r["model"]["file"], [], {}, {}, True))
                continue
            try:
                lag_now, tr_now = current_used(cfg, reg)
            except Exception as e:  # noqa: BLE001
                print(f"  건너뜀 {p.name}: {e}")
                continue
            lag_old = {k: int(v) for k, v in (r.get("lag_days_used") or {}).items()}
            tr_old = r.get("transforms_used")
            diff = [v for v in lag_now if lag_old.get(v) != lag_now[v]]
            if old_ids or tr_old is None or tr_old != tr_now or diff or win_old:
                stale.append((mode, p, cfg, r["model"]["file"], diff, {OLD_IDS.get(k, k): v for k, v in lag_old.items()}, lag_now, win_old))
    # 같은 실험(모드·이름·모델)을 여러 번 저장했으면 가장 최근 것만 다시 돌린다
    latest = {}
    for item in sorted(stale, key=lambda x: x[1].stat().st_mtime):
        nm = item[2].get("name") or item[1].stem.rsplit("_", 1)[0]
        for t in OLD_TAGS:  # 'x' 와 'x_v1105' 는 같은 실험 — 가장 최근 것만
            nm = nm[: -len(t)] if nm.endswith(t) else nm
        latest[(item[0], nm, item[3])] = item
    n_all = len(stale)
    # 그 실험의 가장 최근 결과가 낡지 않았으면(이미 다시 돌림) 건너뛴다
    stale = [it for k, it in latest.items() if it[1].stat().st_mtime >= newest.get(k, 0.0)]
    print(f"낡은 결과 {n_all}건 → 중복 제외 {len(stale)}개 실험 (변수 정렬이 지금 마스터와 다름)\n")
    for mode, p, cfg, model, diff, lo, ln, win_old in stale:
        tag = "  [GPU 권장]" if model in SLOW else ""
        ch = ", ".join(f"{v} {lo.get(v, '?')}→{ln[v]}" for v in diff[:6]) + (" …" if len(diff) > 6 else "")
        print(f"- [{mode}] {p.stem}  모델 {model}{tag}\n    lag 변경: {ch or '(lag 기록 없음/변환 차이)'}" + ("\n    창 눈금: 2026-11-07 수정 전 계산" if win_old else ""))
    if not a.run:
        print("\n목록만 출력했습니다. 재실행: --run (딥러닝까지: --run --all)")
        return
    py = sys.executable
    for mode, p, cfg, model, *_ in stale:
        if model in SLOW and not a.all:
            continue
        c = dict(cfg)
        base = c.get("name") or p.stem.rsplit("_", 1)[0]
        for t in OLD_TAGS:
            base = base[: -len(t)] if base.endswith(t) else base
        c["name"] = base + TAG
        # 결과가 쓴 데이터 버전에 지금의 변수 ID 가 없으면(2026-11-03 변수 ID 정리 전 버전) 최신 데이터 버전으로 — 무엇을 썼는지는 결과 json 의 data_version 에 남는다
        dv_id = c.get("data_version")
        if dv_id and c.get("variable_ids"):
            from engine.store import DataVersion
            try:
                have = set(DataVersion(ROOT / "data", dv_id).wide(None).columns)
            except Exception:  # noqa: BLE001
                have = set()
            miss = [v for v in c["variable_ids"] if v not in have]
            if miss:
                from engine.store import latest_version_id
                latest = latest_version_id(ROOT / "data")  # cli.latest_version 과 같은 규칙(가장 새 온전한 버전)
                print(f"  [데이터 버전] {dv_id} 에 {', '.join(miss[:5])}{' …' if len(miss) > 5 else ''} 없음(옛 ID 시절 버전) → 최신 데이터 버전 {latest} 로 실행")
                c["data_version"] = latest
                tk = c.get("ticker") or ""
                if tk.endswith("_merged") and not (ROOT / "data" / "versions" / latest / f"prices_{tk}.parquet").exists():
                    # 옛 버전의 '합친 시세'(prices_<코드>_merged)는 2026-11-03 버전부터 prices_<코드> 가 그 역할(manifest: assemble prices)
                    c["ticker"] = tk[: -len("_merged")]
                    print(f"  [시세] {tk} → {c['ticker']} (최신 버전에서 합친 시세의 파일 이름)")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(c, f, ensure_ascii=False)
        print(f"\n▶ 재실행 {c['name']} ({model})", flush=True)
        rc = subprocess.call([py, str(ROOT / "cli.py"), "run", f.name], cwd=str(ROOT))
        print(f"  종료 코드 {rc}")


if __name__ == "__main__":
    main()
