"""E2 조사: LightGBM 조기 종료가 폴드마다 몇 그루에서 멈추는가. 파일을 고치지 않고 플러그인을 감싸서 잰다."""
import sys, json, warnings
sys.path.insert(0, "/home/claude/tsp")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from engine import experiment as ex
from engine.store import DataVersion
from engine.registry import Registry
import models.lightgbm_model as lg
import lightgbm as lgb

ROOT = Path("/home/claude/tsp")
rec = []
orig_fit = lg.Model.fit
def fit(self, X, y, X_val=None, y_val=None):
    orig_fit(self, X, y, X_val, y_val)
    bi = getattr(self.est, "best_iteration_", None)
    p_tr = self.predict(X)
    rec.append({"n_tr": len(X), "n_va": 0 if X_val is None else len(X_val), "best_it": bi, "n_trees": self.est.booster_.num_trees(),
                "used": self.est.booster_.current_iteration() if not bi else bi, "pred_std": float(np.std(p_tr)), "y_std": float(np.std(y))})
lg.Model.fit = fit

out = {}
for name in sys.argv[1:]:
    cfg = ex.ExperimentConfig(**json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8")))
    dv = DataVersion(ROOT / "data", cfg.data_version)
    reg, _ = Registry.load_master(ROOT / "data" / "variable_master.xlsx")
    rec.clear()
    c2, plugin, params, tt, ds = ex.prepare(cfg, dv, reg)
    folds, is_split = ex.make_folds(c2, ds, c2.target.k)
    pred, n, imp = ex._walkforward_predict(c2, plugin, params, ds, "cpu", folds, c2.target.k, tt)
    d = pd.DataFrame(rec)
    y = ds.y.reindex(pred.index)
    ok = pred.notna() & y.notna()
    out[name] = {"folds": len(d), "n_estimators": params["n_estimators"], "used_quantiles": d["used"].quantile([0, .1, .25, .5, .75, .9, 1]).round(0).tolist(),
                 "share_used_le2": float((d["used"] <= 2).mean()), "share_used_le5": float((d["used"] <= 5).mean()),
                 "n_va_median": float(d["n_va"].median()), "pred_std_over_y_std_median": float((d["pred_std"] / d["y_std"]).median()),
                 "oos_pred_std": float(pred[ok].std()), "oos_y_std": float(y[ok].std()),
                 "oos_corr": float(np.corrcoef(pred[ok], y[ok])[0, 1]) if pred[ok].std() > 0 else None,
                 "oos_rmse": float(np.sqrt(((pred[ok] - y[ok]) ** 2).mean())), "oos_rmse_zero": float(np.sqrt((y[ok] ** 2).mean()))}
    print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
Path("/tmp/e2/probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
