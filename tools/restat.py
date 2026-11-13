"""통계 재계산 — 저장된 결과의 일별 예측으로 검정을 다시 계산해 '한 기준'의 표를 만든다. 다시 학습하지 않고, 결과 파일을 고치지 않는다(2026-11-10).

왜 필요한가
  · 결과마다 과최적화 보정(Deflated Sharpe)에 쓴 시도 횟수 N 과 분산 V 가 '그 실험을 돌린 순간'의 값이라 서로 다르다(N 2~98). 서로 비교하려면 같은 N·V 로 다시 계산해야 한다.
  · 0예측 대비 비교에 Clark–West 가 없던 때의 결과에는 cw 값이 없다.
  · 옛 계산 방식의 결과(engine_rev 가 지금과 다름, 평가 구간에 봉인 구간이 섞인 것 포함)를 표에서 구분해야 한다.

하는 일
  1. results/<mode>/*.json 과 *_daily.parquet 을 읽는다(봉인 평가 결과·ablation 표 제외).
  2. 평가 구간의 (예측, 실제)로 DM(0예측 대비)을 다시 계산해 저장된 값과 맞는지 본다 — 맞지 않으면 그 줄에 '불일치'를 적는다(이 도구가 엔진과 같은 표본을 쓰는지의 자기 점검).
  3. Clark–West(0예측 대비)를 계산한다.
  4. 매매 모드: 원장의 개발 시도 수 N 하나, 지금 엔진 결과들의 샤프 분산 V 하나(주기별)로 Deflated Sharpe 를 다시 계산한다.
  5. Holm 보정(engine/multitest.py): 예측 모드는 'naive 보다 낫다'의 한쪽 p, 두 모드 모두 CW(평균 기준·0 기준)의 한쪽 p.
     가족 = 같은 대상·주기·k·타겟 종류(·정보 시점)의 지금 엔진 결과. 모드 전체를 한 가족으로 본 값도 같이 낸다(가족 정의에 대한 민감도).
  6. results/restat/restat_<UTC>.json · .md 로 저장한다(원 결과는 그대로).

쓰기:  python tools/restat.py            (두 모드 모두)
       python tools/restat.py --mode trading
한계: N 은 원장 값이다(원장 시작 전의 횟수는 옛 카운터 값). '서로 다른 설정 수'가 아니라 '돌린 횟수'라 보수적이다. V 는 지금 엔진 결과가 2개 미만이면 계산하지 못하고
      Deflated Sharpe 는 근사(1/(n−1))로 떨어진다 — 그 경우 표에 적는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import ledger  # noqa: E402
from engine.backtest import clark_west, deflated_sharpe_pvalue, diebold_mariano, diebold_mariano_brier  # noqa: E402
from engine.experiment import ENGINE_REV, mean_benchmark_metrics  # noqa: E402
from engine.multitest import better_one_sided, holm  # noqa: E402

PPY = {"D": 252, "W": 52, "M": 12}
SKIP = ("experiment_counter.json", "sealed_log.json")


def _load(d: Path) -> list[tuple[Path, dict]]:
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name in SKIP or p.name.startswith("ablation_"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, dict) and "config" in r and not r.get("sealed_evaluation"):
            out.append((p, r))
    return out


def _eval_rows(p: Path, r: dict) -> pd.DataFrame | None:
    q = p.with_name(p.stem + "_daily.parquet")
    if not q.exists():
        return None
    d = pd.read_parquet(q)
    if "prediction" not in d or "target" not in d:
        return None
    ev = r.get("evaluation_period") or r.get("prediction_period")
    if ev:
        d = d.loc[str(ev[0]):str(ev[1])]
    return d.dropna(subset=["prediction", "target"])


def _is_stale(r: dict) -> bool:
    return r.get("engine_rev") != ENGINE_REV


def _row(p: Path, r: dict) -> dict:
    c, pm, pf = r["config"], r.get("prediction_metrics") or {}, r.get("performance") or {}
    k = int((c.get("target") or {}).get("k", 1))
    cls = (c.get("target") or {}).get("kind") in ("direction", "event")
    row = {"file": p.stem, "name": c.get("name") or r["model"]["file"], "model": r["model"]["file"], "mode": c.get("mode", "trading"), "ticker": c.get("ticker"),
           "frequency": c.get("frequency"), "k": k, "target_kind": (c.get("target") or {}).get("kind"), "target_variable": c.get("target_variable"),
           "info_time": c.get("forecast_info_time", "stamp"),
           "engine_rev": r.get("engine_rev"), "stale": _is_stale(r), "code_version": r.get("code_version"),
           "evaluation_period": r.get("evaluation_period") or r.get("prediction_period"),
           "sealed_inclusive": c.get("mode", "trading") != "forecast" and not r.get("evaluation_period"),
           "dir_acc": pm.get("dir_acc"), "dm_vs_zero_stored": pm.get("dm_vs_zero"), "dm_vs_zero_p_stored": pm.get("dm_vs_zero_p"),
           "dm_vs_naive": pm.get("dm_vs_naive"), "dm_vs_naive_p": pm.get("dm_vs_naive_p"),
           "sharpe": pf.get("sharpe"), "n_periods": pf.get("n_periods"), "bootstrap_p_excess": pf.get("bootstrap_p_excess"),
           "deflated_p_stored": pf.get("deflated_sharpe_p"), "n_trials_stored": pf.get("n_trials_so_far"), "trial_sr_var_stored": pf.get("trial_sr_var"),
           "return_skew": pf.get("return_skew"), "return_kurt": pf.get("return_kurt")}
    ev = _eval_rows(p, r)
    if ev is None or len(ev) < 10 or pm.get("dm_vs_zero") is None:
        row["note"] = "예측값 없음(규칙 모델·보유 등) — DM·CW 해당 없음" if pm.get("dm_vs_zero") is None else "일별 자료 없음"
        return row
    y, f = ev["target"].to_numpy(float), ev["prediction"].to_numpy(float)
    if cls:
        dm, dm_p = diebold_mariano_brier(f, y, h=k)
        cw, cw_p = clark_west(y, f, 0.5, h=k)
    else:
        dm, dm_p = diebold_mariano(f - y, 0 - y, h=k)
        cw, cw_p = clark_west(y, f, 0.0, h=k)
    row.update(n_eval=int(len(ev)), dm_vs_zero=dm, dm_vs_zero_p=dm_p, cw_vs_zero=cw, cw_vs_zero_p=cw_p,
               dm_matches_stored=bool(np.isclose(dm, pm["dm_vs_zero"], rtol=1e-6, atol=1e-9)))
    # 학습 창 평균 기준(변수의 정보를 말할 때 쓰는 검정). 일별 자료에 기준 열(bench_mean)이 있는 결과만 — 2026-11-10 전 결과에는 없어 다시 돌려야 한다.
    if "bench_mean" in ev:
        row.update(mean_benchmark_metrics(y, f, ev["bench_mean"].to_numpy(float), k))
    else:
        row["note_mean"] = "학습 창 평균 기준 없음(2026-11-10 전 결과) — CW(평균 기준)는 다시 돌려야 나옴"
    row["dm_vs_naive_p_better"] = better_one_sided(row.get("dm_vs_naive"), row.get("dm_vs_naive_p"))  # 'naive 보다 낫다'의 한쪽 p
    if not row["dm_matches_stored"]:
        row["note"] = "DM 재계산이 저장값과 다름 — 이 줄의 CW 는 엔진과 다른 표본일 수 있음(확인 필요)"
    return row


def restat(mode: str, results_root: Path) -> dict:
    d = results_root / mode
    rows = [_row(p, r) for p, r in _load(d)]
    n_trials = ledger.count(d)
    info: dict = {"mode": mode, "engine_rev": ENGINE_REV, "n_results": len(rows), "n_current": sum(not r["stale"] for r in rows),
                  "n_stale": sum(r["stale"] for r in rows), "n_sealed_inclusive": sum(bool(r["sealed_inclusive"]) for r in rows),
                  "n_trials_ledger": n_trials, "ledger": ledger.verify(d) if (d / ledger.LEDGER).exists() else {"ok": None, "why": "원장 없음 — 옛 카운터 값 사용"}}
    if mode == "trading":
        v_by: dict[str, float | None] = {}
        for fq, ppy in PPY.items():
            srs = [r["sharpe"] / np.sqrt(ppy) for r in rows if not r["stale"] and r["frequency"] == fq and r.get("sharpe") is not None and np.isfinite(r["sharpe"])]
            v_by[fq] = float(np.var(srs, ddof=1)) if len(srs) >= 2 else None
        info["trial_sr_var_by_frequency"] = v_by
        for r in rows:
            if r.get("sharpe") is None or not r.get("n_periods"):
                continue
            ppy = PPY.get(r["frequency"], 252)
            r["deflated_p_unified"] = deflated_sharpe_pvalue(r["sharpe"], int(r["n_periods"]), max(1, n_trials),
                                                             skew=r.get("return_skew") or 0.0, kurt=r.get("return_kurt") or 3.0,
                                                             periods_per_year=ppy, trial_sr_var=v_by.get(r["frequency"]))
            # 표시(F14): V 가 없어 근사(1/(n−1))로 떨어졌는지, 왜도·첨도가 저장돼 있지 않아 정규 가정(0·3)을 썼는지
            r["deflated_basis"] = {"n_trials": n_trials, "trial_sr_var": v_by.get(r["frequency"]), "v_fallback": v_by.get(r["frequency"]) is None,
                                   "moments_assumed": r.get("return_skew") is None or r.get("return_kurt") is None}
    else:
        holm(rows, "dm_vs_naive_p_better", "holm_dm_naive_p", skip=lambda r: r.get("stale"))
    holm(rows, "cw_vs_zero_p", "holm_cw_zero_p", skip=lambda r: r.get("stale"))
    holm(rows, "cw_vs_mean_p", "holm_cw_mean_p", skip=lambda r: r.get("stale"))
    info["dm_recalc_mismatch"] = [r["file"] for r in rows if r.get("dm_matches_stored") is False]
    return {"info": info, "rows": rows}


def _h(r: dict, key: str) -> str:
    if r.get(key) is None:
        return "—"
    return f"{_fmt(r[key])} ({r.get(key + '_m')}) · {_fmt(r.get(key + '_all'))} ({r.get(key + '_all_m')})"


def _dir(dm) -> str:
    return "—" if dm is None or not np.isfinite(dm) else ("나음" if dm < 0 else "나쁨")


def _dsr_mark(r: dict) -> str:
    b = r.get("deflated_basis") or {}
    return ("†" if b.get("v_fallback") else "") + ("‡" if b.get("moments_assumed") else "")


def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def to_markdown(res: dict) -> str:
    i, rows = res["info"], res["rows"]
    L = [f"# 통계 재계산 — {i['mode']} (엔진 {i['engine_rev']})", "",
         f"- 결과 {i['n_results']}건: 지금 엔진 {i['n_current']}건 · 옛 계산 방식 {i['n_stale']}건(그중 봉인 구간이 섞인 평가 {i['n_sealed_inclusive']}건)",
         f"- 개발 시도 수 N = {i['n_trials_ledger']} (원장: {i['ledger']})",
         f"- DM 재계산이 저장값과 다른 결과: {i['dm_recalc_mismatch'] or '없음'}", ""]
    if i["mode"] == "trading":
        L += [f"- 샤프 분산 V(주기별, 지금 엔진 결과만): {i['trial_sr_var_by_frequency']}", "",
              "| 결과 | 주기 | 옛 방식 | 방향 적중 | DM vs 0 (p) | CW vs 평균 (한쪽 p) | Holm p 가족 (m) · 전체 (m) | CW vs 0 (한쪽 p) | Holm p 가족 (m) · 전체 (m) | 샤프 | 초과수익 p | DSR p(저장, 당시 N) | DSR p(통일 N·V) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['name']} | {r['frequency']} | {'봉인 섞임' if r['sealed_inclusive'] else ('예' if r['stale'] else '')} | {_fmt(r.get('dir_acc'))} | "
                     f"{_fmt(r.get('dm_vs_zero'), 2)} ({_fmt(r.get('dm_vs_zero_p'))}) | {_fmt(r.get('cw_vs_mean'), 2)} ({_fmt(r.get('cw_vs_mean_p'))}) | {_h(r, 'holm_cw_mean_p')} | "
                     f"{_fmt(r.get('cw_vs_zero'), 2)} ({_fmt(r.get('cw_vs_zero_p'))}) | {_h(r, 'holm_cw_zero_p')} | {_fmt(r.get('sharpe'), 2)} | {_fmt(r.get('bootstrap_p_excess'))} | "
                     f"{_fmt(r.get('deflated_p_stored'))} (N={r.get('n_trials_stored')}) | {_fmt(r.get('deflated_p_unified'))}{_dsr_mark(r)} |")
    else:
        L += ["| 결과 | 대상 | k | 옛 방식 | 방향 적중 | DM vs naive (양측 p) | 방향 | 'naive 보다 낫다' 한쪽 p | Holm p 가족 (m) · 전체 (m) | CW vs 평균 (한쪽 p) | Holm p 가족 (m) · 전체 (m) | CW vs 0 (한쪽 p) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['name']} | {r.get('target_variable')} | {r['k']} | {'예' if r['stale'] else ''} | {_fmt(r.get('dir_acc'))} | "
                     f"{_fmt(r.get('dm_vs_naive'), 2)} ({_fmt(r.get('dm_vs_naive_p'))}) | {_dir(r.get('dm_vs_naive'))} | {_fmt(r.get('dm_vs_naive_p_better'))} | {_h(r, 'holm_dm_naive_p')} | "
                     f"{_fmt(r.get('cw_vs_mean'), 2)} ({_fmt(r.get('cw_vs_mean_p'))}) | {_h(r, 'holm_cw_mean_p')} | {_fmt(r.get('cw_vs_zero'), 2)} ({_fmt(r.get('cw_vs_zero_p'))}) |")
    L += ["", "읽는 법",
          "- DM vs 0 / vs naive = 이 방법·이 학습 창이 그 기준보다 정확한가(음수면 정확, 추정 잡음 포함, 양측 p). 방향 칸이 '나쁨'이면 p 가 작아도 '유의하게 나쁘다'는 뜻이다.",
          "- CW vs 평균 = 학습 창 평균(분류면 양(+) 비율)만 아는 예측보다 예측 정보가 더 있는가(추정 잡음을 뺀 비교, 한쪽 p). 변수의 정보를 말할 때는 이 칸을 쓴다. 2026-11-10 전 결과에는 기준 열이 없어 비어 있다(다시 돌려야 나옴).",
          "- CW vs 0 = 예측이 실제와 같은 쪽으로 움직이는가(평균 포함). 타겟 평균이 0 이 아니면 '평균만 맞히는' 예측도 유의하게 나온다 — 변수의 정보의 증거로 쓰지 않는다.",
          "- Holm = 다중비교 보정. '가족' = 같은 대상·주기·k·타겟 종류(·정보 시점)의 지금 엔진 결과, '전체' = 이 모드의 지금 엔진 결과 전부. 둘이 다르면 결론이 가족 정의에 달려 있다는 뜻이다. 옛 방식 결과는 보정·V 에 넣지 않았다.",
          "- DSR p(통일) 뒤의 표시: † = 같은 주기의 지금 엔진 결과가 2개 미만이라 V 를 근사(1/(n−1))로 씀 · ‡ = 왜도·첨도가 저장돼 있지 않아 정규 가정(0·3)을 씀.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["trading", "forecast", "both"], default="both")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None, help="저장 폴더(기본 results/restat)")
    a = ap.parse_args()
    root = Path(a.results)
    out = Path(a.out) if a.out else root / "restat"
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for mode in (("trading", "forecast") if a.mode == "both" else (a.mode,)):
        res = restat(mode, root)
        (out / f"restat_{mode}_{ts}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        (out / f"restat_{mode}_{ts}.md").write_text(to_markdown(res), encoding="utf-8")
        i = res["info"]
        print(f"[{mode}] 결과 {i['n_results']}건(지금 엔진 {i['n_current']} · 옛 방식 {i['n_stale']}) · N={i['n_trials_ledger']} · DM 불일치 {len(i['dm_recalc_mismatch'])}건 → {out / f'restat_{mode}_{ts}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
