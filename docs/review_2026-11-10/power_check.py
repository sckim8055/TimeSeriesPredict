"""플랫폼 검출력: 심은 신호의 크기(R²)를 바꿔 가며, 플랫폼의 판정이 신호를 얼마나 잡는가(합성 자료, Ridge 일간)."""
import sys, json, tempfile, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from engine.store import DataVersion
from engine.registry import Registry, Variable
from engine.experiment import ExperimentConfig, Validation, run_experiment
from scipy import stats

def make(n, seed, r2, sigma=0.02):
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n): s[t] = 0.7 * s[t-1] + rng.normal(0, 1)
    b = 0.0 if r2 == 0 else sigma * np.sqrt(r2 / ((1 - r2) * (1 / (1 - 0.49))))
    r_open = b * np.r_[0, 0, s[:-2]] + rng.normal(0, sigma, n)
    open_ = 10000 * np.exp(np.cumsum(r_open)); close = open_ * (1 + rng.normal(0, sigma/2, n))
    idx = pd.bdate_range("2010-01-01", periods=n)
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close)*1.002, "low": np.minimum(open_, close)*0.998, "close": close, "volume": rng.lognormal(12, .3, n)}, index=idx)
    return df, pd.Series(s, index=idx, name="S_signal")

def cw(y, f, h=1):  # Clark–West: 0예측(내포) 대비, 한쪽
    adj = y**2 - ((y - f)**2 - f**2)
    m = adj.mean(); se = adj.std(ddof=1) / np.sqrt(len(adj))
    t = m / se; return t, 1 - stats.norm.cdf(t)

out = []
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3600
for r2 in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05):
    for seed in range(int(sys.argv[2]) if len(sys.argv) > 2 else 12):
        prices, s = make(N, 1000 + seed, r2)
        tmp = tempfile.mkdtemp()
        dv = DataVersion(tmp, "t"); dv.write_prices("TEST", prices)
        reg = Registry(); reg.add(Variable(id="S_signal", freq="D", source="DERIVED"))
        dv.write_variables(s.rename("value").rename_axis("date").reset_index().assign(variable_id="S_signal"), source="syn")
        cfg = ExperimentConfig(ticker="TEST", model="ridge", variable_ids=["S_signal"], use_technical=False)
        res = run_experiment(cfg, dv, reg)
        d = res["_daily"].dropna(subset=["prediction", "target"]) if "target" in res["_daily"] else None
        pm, pf = res["prediction_metrics"], res["performance"]
        t, p = cw(d["target"].values, d["prediction"].values) if d is not None else (np.nan, np.nan)
        out.append(dict(r2=r2, seed=seed, n=len(d) if d is not None else 0, dir=pm.get("dir_acc"), dm=pm.get("dm_vs_zero"), dm_p=pm.get("dm_vs_zero_p"),
                        cw_t=t, cw_p=p, sharpe=pf.get("sharpe"), bh=pf.get("bh_sharpe") or (pf.get("benchmark") or {}).get("sharpe"),
                        boot_p=pf.get("bootstrap_p_excess"), dsr_p=pf.get("deflated_sharpe_p")))
        if seed == 0 and r2 == 0.0:
            print("keys pm:", list(pm)); print("keys pf:", list(pf)); print("daily cols:", list(res["_daily"].columns)); sys.stdout.flush()
json.dump(out, open(__import__("pathlib").Path(__file__).with_name("power_check_result.json"), "w"))
df = pd.DataFrame(out)
print(df.groupby("r2").agg(n=("n","mean"), dir=("dir","mean"), dm_mean=("dm","mean"), cw_t=("cw_t","mean")).round(3))
df["dm_pass"] = (df.dm < 0) & (df.dm_p < 0.05)      # 플랫폼 기준 1: 0예측보다 유의하게 나음
df["dm_bad"] = (df.dm > 0) & (df.dm_p < 0.05)       # "0예측보다 유의하게 나쁨" 배지
df["cw_pass"] = df.cw_p < 0.05
df["boot_pass"] = df.boot_p < 0.05
df["dsr_pass"] = df.dsr_p < 0.05
df["all3"] = df.dm_pass & df.boot_pass & df.dsr_pass
print(df.groupby("r2").agg(runs=("seed","count"), dir=("dir","mean"), dm_pass=("dm_pass","mean"), dm_bad=("dm_bad","mean"), cw_pass=("cw_pass","mean"),
      boot_pass=("boot_pass","mean"), dsr_pass=("dsr_pass","mean"), all3=("all3","mean"), sharpe=("sharpe","mean")).round(3).to_string())
