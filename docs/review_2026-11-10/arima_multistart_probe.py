"""E6 조사: ARIMA·SARIMA 가능도 다중 극대 — 여러 출발점 중 최대 가능도를 고르면 기본 출발점보다 얼마나 자주·얼마나 높아지나(F11 월간 폴드)."""
import sys, json, warnings, time
sys.path.insert(0, "/home/claude/tsp"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from engine import experiment as ex
from engine.store import DataVersion
from engine.registry import Registry
ROOT = Path("/home/claude/tsp")

def starts(mod, base):
    names = list(mod.param_names); out = [("default", None)]
    def vec(ar, ma, sar=0.0, sma=0.0):
        v = np.array(base, dtype=float).copy()
        for i, n in enumerate(names):
            if n.startswith("ar.S."): v[i] = sar
            elif n.startswith("ma.S."): v[i] = sma
            elif n.startswith("ar.L"): v[i] = ar if n.endswith("L1") else 0.0
            elif n.startswith("ma.L"): v[i] = ma if n.endswith("L1") else 0.0
        return v
    out += [("zero", vec(0, 0)), ("ar+", vec(0.5, 0.0, 0.3, 0.0)), ("ar-", vec(-0.5, 0.0, -0.3, 0.0)), ("ma+", vec(0.0, 0.5, 0.0, 0.3)), ("ma-", vec(0.0, -0.5, 0.0, -0.3))]
    return out

def fit_multi(r, order, seas, trend, scale=100.0):
    mod = SARIMAX(r * scale, order=order, seasonal_order=seas, trend=trend, enforce_stationarity=True, enforce_invertibility=True)
    res = []
    base = mod.start_params
    for nm, sp in starts(mod, base):
        try:
            f = mod.fit(start_params=sp, disp=False, maxiter=200)
            res.append((nm, float(f.llf), bool(f.mle_retvals.get("converged")), f))
        except Exception as e:
            res.append((nm, float("-inf"), False, None))
    try:
        f = mod.fit(method="nm", disp=False, maxiter=2000); f = mod.fit(start_params=f.params, disp=False, maxiter=200)
        res.append(("nm", float(f.llf), bool(f.mle_retvals.get("converged")), f))
    except Exception:
        pass
    return res

for name in sys.argv[1:]:
    cfg = ex.ExperimentConfig(**json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8")))
    dv = DataVersion(ROOT / "data", cfg.data_version); reg, _ = Registry.load_master(ROOT / "data" / "variable_master.xlsx")
    c, pl, pa, tt, ds = ex.prepare(cfg, dv, reg)
    folds, _ = ex.make_folds(c, ds, 1); folds = [f for f in folds if len(f.test)]
    r_all = ds.prices["close"].pct_change()
    order = (pa["p"], pa["d"], pa["q"]); seas = (pa["P"], pa["D"], pa["Q"], pa["s"]) if "P" in pa else (0, 0, 0, 0)
    tr = pa.get("trend", "c"); tr = ("c" if (pa["d"] == 0 and pa.get("D", 0) == 0) else "n") if tr == "auto" else tr
    t0 = time.time(); gain = []; win = {}; nconv = 0; fc_diff = []
    for f in folds[::4]:   # 4 폴드마다 하나(시간)
        r = r_all.loc[f.train].dropna().values
        rs = fit_multi(r, order, seas, tr)
        d = rs[0][1]; best = max(rs, key=lambda x: x[1])
        gain.append(best[1] - d); win[best[0]] = win.get(best[0], 0) + 1; nconv += rs[0][2]
        if best[3] is not None and rs[0][3] is not None:
            fc_diff.append(abs(best[3].forecast(1)[0] - rs[0][3].forecast(1)[0]) / 100.0)
    g = np.array(gain)
    print(name, "folds", len(g), "order", order, seas, tr, "| default converged", nconv, "| llf gain >0.01:", int((g > 0.01).sum()), ">0.5:", int((g > 0.5).sum()), "max", round(float(g.max()), 3),
          "| best start", win, "| mean |Δ1-step forecast|", float(np.mean(fc_diff)), "vs r sd", float(np.nanstd(r_all)), "| sec", round(time.time() - t0))
