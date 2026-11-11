"""results/*.json → 리더보드 (results/leaderboard.csv, .md).  python report.py"""
import glob, json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
rows = []
for f in sorted(glob.glob(str(ROOT / "results" / "trading" / "*.json"))):
    if f.endswith("experiment_counter.json"):
        continue
    r = json.load(open(f, encoding="utf-8"))
    c, p, m = r["config"], r["performance"], r["prediction_metrics"]
    rows.append({
        "실험": c["name"], "모델": r["model"]["name"], "주기": c["frequency"], "변수": "+".join((["A"] if c["use_technical"] else []) + sorted({(v.split("_")[0] if "_" in v else "D") for v in c["variable_ids"]})),
        "k": c["target"]["k"], "폴드": r["n_folds"], "방향정확도": m.get("dir_acc"), "DM(vs0)": m.get("dm_vs_zero"), "DM p": m.get("dm_vs_zero_p"),
        "CAGR": p.get("cagr"), "샤프": p.get("sharpe"), "MDD": p.get("mdd"), "거래수": p.get("n_trades"), "노출": p.get("exposure"),
        "B&H CAGR": p.get("benchmark_cagr"), "B&H MDD": p.get("benchmark_mdd"), "초과수익 p": p.get("bootstrap_p_excess"), "Deflated샤프 p": p.get("deflated_sharpe_p"),
        "예측구간": "~".join(r.get("prediction_period") or ["",""]), "예측수": r.get("n_predictions"), "초": r["seconds"], "장치": r.get("device"),
    })
df = pd.DataFrame(rows).sort_values("샤프", ascending=False)
df.to_csv(ROOT / "results" / "trading" / "leaderboard.csv", index=False, encoding="utf-8-sig")
fmt = df.copy()
for c in ("방향정확도", "CAGR", "MDD", "노출", "B&H CAGR", "B&H MDD"):
    fmt[c] = fmt[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
for c in ("샤프", "DM(vs0)", "DM p", "초과수익 p", "Deflated샤프 p"):
    fmt[c] = fmt[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
(ROOT / "results" / "trading" / "leaderboard.md").write_text(fmt.drop(columns=["초", "장치"]).to_markdown(index=False), encoding="utf-8")
print(fmt.drop(columns=["초", "장치", "B&H MDD", "Deflated샤프 p"]).to_string(index=False))
