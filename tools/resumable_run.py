"""실험을 폴드 단위로 이어서 돌린다 — 긴 실행(예: 일간 BIVA)이 중간에 끊겨도 마지막으로 끝난 폴드 다음부터 계속한다.

  python tools/resumable_run.py configs/10_biva_ABC_e20.json            (끊기면 같은 명령을 다시 실행)
  python tools/resumable_run.py <설정> --stop-after 3                     (시험용: 이번 실행에서 폴드 3개만 하고 멈춤)

결과는 끊김 없이 한 번에 돌린 것과 같다. 폴드 사이에 넘기는 상태가 예측·중요도·폴드 기록·학습 진단·난수 상태(random·numpy·torch)뿐이다.
그래서 폴드가 끝날 때마다 이것들을 저장해 두고, 다시 시작하면 되돌린 뒤 다음 폴드부터 같은 함수로 돌린다.
(2026-11-10 부터 엔진이 폴드마다 시드를 그 폴드의 날짜로 다시 정하므로(engine.experiment.fold_seed) 학습 모델은 난수 상태를 되돌리지 않아도
같은 값이 나온다. 난수 상태 저장은 통계 모델 등 폴드 시드를 쓰지 않는 경로를 위해 그대로 둔다.) 저장 파일: logs/ckpt/<실험 이름>.pkl (설정·폴드 구성이 다르면 쓰지 않음).
끝까지 돌면 cli.py run 과 똑같이 결과를 저장하고(실험 카운터 포함) 저장 파일을 지운다.
시험: tests/test_resumable.py — 중간에 멈췄다 이어 돌린 예측 = 한 번에 돌린 예측.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path

import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine.experiment as E  # noqa: E402

_orig = E._walkforward_predict


class _Stop(Exception):
    """--stop-after 로 일부러 멈춤(시험용)."""


def _rng_get():
    st = {"py": random.getstate(), "np": np.random.get_state()}
    try:
        import torch
        st["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            st["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return st


def _rng_set(st):
    random.setstate(st["py"]); np.random.set_state(st["np"])
    if "torch" in st:
        import torch
        torch.set_rng_state(st["torch"])
        if "cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda"])


def make_resumable(ckpt: Path, stop_after: int | None = None):
    """E._walkforward_predict 를 폴드 단위 저장·재개 판으로 바꾼다(want_fit·capture 를 쓰는 호출은 원래 함수 그대로)."""

    def wf(cfg, plugin, params, ds, device, folds, k, target_type, *, want_fit=False, log=None, capture=None, diags=None):
        if want_fit or capture is not None or len(folds) < 2:
            return _orig(cfg, plugin, params, ds, device, folds, k, target_type, want_fit=want_fit, log=log, capture=capture, diags=diags)
        # 지문에 입력 자료 자체(X·y 의 값과 날짜)와 코드 판을 넣는다 — 중단과 재개 사이에 변수 사전(lag·변환·active)이나 데이터 버전이 바뀌면
        # 앞 폴드와 뒤 폴드가 서로 다른 입력으로 계산돼 섞였다(2026-11-10 점검 N9). 이제는 지문이 달라져 처음부터 다시 돈다.
        # 가격·수준값도 넣는다(ARIMA·VECM 은 X·y 가 아니라 가격·수준값으로 학습한다). 코드 판은 git 이 없으면 늘 "nogit" 이라 코드가 바뀌어도 같았다
        # → 엔진·모델 소스 파일 내용의 해시를 따로 넣는다(2026-11-10 독립 검토 F20).
        lv = getattr(ds, "levels", None)
        data_fp = hashlib.sha1(pd.util.hash_pandas_object(ds.X, index=True).values.tobytes()
                               + pd.util.hash_pandas_object(ds.y, index=True).values.tobytes()
                               + pd.util.hash_pandas_object(ds.prices, index=True).values.tobytes()
                               + (pd.util.hash_pandas_object(lv, index=True).values.tobytes() if lv is not None and len(lv) else b"")
                               + "|".join(map(str, ds.X.columns)).encode()).hexdigest()
        src_fp = hashlib.sha1(b"".join(q.read_bytes() for q in sorted((ROOT / "engine").glob("*.py")) + sorted((ROOT / "models").glob("*.py")))).hexdigest()
        fp = hashlib.sha1(json.dumps([cfg.model_dump_json(), params, device, data_fp, src_fp, E._code_version(), E.ENGINE_REV,
                                      [[str(f.train[0]), str(f.train[-1]), str(f.test[0]) if len(f.test) else "", str(f.test[-1]) if len(f.test) else ""] for f in folds]],
                                     default=str).encode()).hexdigest()
        pred = None; n_folds = 0; imps: list = []; done = 0; spent = 0.0
        if ckpt.exists():
            c = pickle.load(open(ckpt, "rb"))
            if c.get("fp") == fp:
                pred, n_folds, imps, done, spent = c["pred"], c["n"], c["imp"], c["done"], c.get("spent", 0.0)
                if log is not None:
                    log[:] = c["log"]
                if diags is not None:
                    diags[:] = c.get("diags") or []
                _rng_set(c["rng"])
                print(f"  [이어 돌리기] {ckpt.name}: 폴드 {done}/{len(folds)} 까지 끝난 상태에서 계속(앞서 쓴 시간 {spent/60:.1f}분)", flush=True)
            else:
                print(f"  [이어 돌리기] {ckpt.name} 은 다른 설정·폴드의 저장 파일 — 처음부터", flush=True)
        ran = 0
        for i in range(done, len(folds)):
            t0 = time.time()
            p_i, n_i, imp_i = _orig(cfg, plugin, params, ds, device, [folds[i]], k, target_type, log=log, diags=diags)
            pred = p_i if pred is None else pred.where(p_i.isna(), p_i)
            n_folds += n_i; imps += list(imp_i); spent += time.time() - t0
            c = {"fp": fp, "done": i + 1, "pred": pred, "n": n_folds, "imp": imps, "log": list(log) if log is not None else None, "diags": list(diags) if diags is not None else None, "rng": _rng_get(),
                 "spent": spent}
            tmp = ckpt.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump(c, fh)
            os.replace(tmp, ckpt)
            ran += 1
            print(f"  [이어 돌리기] 폴드 {i + 1}/{len(folds)} 저장 · 이 폴드 {(time.time() - t0)/60:.1f}분 · 누적 {spent/60:.1f}분", flush=True)
            if stop_after is not None and ran >= stop_after and i + 1 < len(folds):
                raise _Stop(f"폴드 {i + 1}/{len(folds)} 에서 멈춤(--stop-after {stop_after})")
        if pred is None:
            pred = pd.Series(np.nan, index=ds.prices.index, name="prediction")
        print(f"  [이어 돌리기] 폴드 {len(folds)}개 모두 끝 · 폴드 학습·예측에 쓴 시간 합계 {spent/60:.1f}분(끊긴 실행 포함)", flush=True)
        return pred, n_folds, imps

    E._walkforward_predict = wf
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--stop-after", type=int); ap.add_argument("--ckpt")
    a = ap.parse_args()
    raw = json.loads(Path(a.config).read_text(encoding="utf-8"))
    name = raw.get("name") or Path(a.config).stem
    ckpt = Path(a.ckpt) if a.ckpt else ROOT / "logs" / "ckpt" / f"{name}.pkl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    make_resumable(ckpt, a.stop_after)
    import cli
    sys.argv = ["cli.py", "run", a.config]
    try:
        cli._main_with_log()
    except _Stop as e:
        print(f"[이어 돌리기] {e} — 같은 명령을 다시 실행하면 이어서 돕니다", flush=True)
        sys.exit(3)
    if ckpt.exists():
        ckpt.unlink()


if __name__ == "__main__":
    main()
