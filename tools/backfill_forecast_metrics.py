"""저장된 예측 연구 결과(results/forecast/*.json)에 수준값 지표(모델·naive·변화없음)와 naive 대비 개선율을 채운다.
모델을 다시 돌리지 않고 저장된 *_daily.parquet 에서 결정적으로 계산한다(값이 바뀌지 않는 파생 지표만 추가)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.experiment import forecast_level_metrics  # noqa: E402

for p in sorted((ROOT / "results" / "forecast").glob("*.json")):
    if p.name in ("experiment_counter.json", "sealed_log.json"):
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    dp = p.with_name(p.stem + "_daily.parquet")
    if not dp.exists():
        print("daily 없음:", p.name); continue
    d = pd.read_parquet(dp)
    d.attrs["k"] = r["config"]["target"]["k"]
    pm = r["prediction_metrics"]
    add = forecast_level_metrics(d, r["config"]["target"]["kind"])
    if pm.get("rmse_naive"):
        add["skill_vs_naive"] = 1 - pm["rmse"] / pm["rmse_naive"]
    pm.update(add)
    p.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{p.name}: MAPE {pm.get('level_mape', float('nan')):.4%} naive {pm.get('level_mape_naive', float('nan')):.4%} 변화없음 {pm.get('level_mape_zero', float('nan')):.4%} 개선율(MAPE) {pm.get('skill_mape_vs_naive', float('nan')):+.1%} p {pm.get('dm_ape_vs_naive_p', float('nan')):.3f}")
