"""가이드 문서 생성 — docs/guide/*.md 의 자리표시자를 코드에서 뽑은 표로 채운다.

  {{PARAMS:<모델파일>}}   models/<파일>.py 의 PARAMS → 표 (이름 / 뜻 / 기본값 / 범위)
  {{VARIABLE_TABLE}}     data/variable_master.xlsx → 변수 표
  {{FEATURE_PRESETS}}    engine/features.PRESETS → 주기별 창 길이 표
  {{FIG:<그림키>}}       tools/formula_figs.py 의 SVG 그림(수식 읽는 법)
  {{FLOW:<모델파일>}}     웹 도움말에서 그 모델의 흐름도 그림(실험 화면과 같은 ModelFlow)으로 바뀌는 자리

출력: docs/guide_build/<같은 이름>.md (화면·API 가 읽음) + docs/GUIDE.md (전체 합본, 논문·인쇄용)
사용: python tools/build_guide.py   (API 시작 시에도 자동 실행)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "docs" / "guide"
OUT = ROOT / "docs" / "guide_build"


def params_table(model_file: str) -> str:
    from engine.experiment import load_plugin

    m = load_plugin(model_file)
    if not m.PARAMS:
        return "(파라미터 없음)"
    by_freq = any(isinstance(p.get("default_by_freq"), dict) for p in m.PARAMS.values())
    if by_freq:
        rows = ["| 파라미터 | 뜻 | 기본값 일(D) | 주(W) | 월(M) | 범위/선택 |", "|---|---|---|---|---|---|"]
    else:
        rows = ["| 파라미터 | 뜻 | 기본값 | 범위/선택 |", "|---|---|---|---|"]
    for k, p in m.PARAMS.items():
        rng = ", ".join(map(str, p["choices"])) if p.get("choices") else (f"{p.get('min', '')} ~ {p.get('max', '')}" if "min" in p or "max" in p else "")
        if by_freq:
            d = p.get("default_by_freq") or {}
            cells = " | ".join(str(d.get(f, p.get("default"))) for f in ("D", "W", "M"))
            rows.append(f"| `{k}` | {p.get('desc', '')} | {cells} | {rng} |")
        else:
            rows.append(f"| `{k}` | {p.get('desc', '')} | {p.get('default')} | {rng} |")
    if by_freq:
        rows.append("")
        rows.append("기본값은 주기(일/주/월)에 따라 다릅니다. 표본 수가 적은 월간에는 더 작고 보수적인 값을 씁니다. 설정에서 값을 주면 그 값이 우선하고, 실제 사용값은 결과 json 의 `model.params` 에 남습니다.")
    return "\n".join(rows)


def variable_table() -> str:
    import pandas as pd

    df = pd.read_excel(ROOT / "data" / "variable_master.xlsx", sheet_name="variables", dtype=str).fillna("")
    rows = ["| id | 이름 | 그룹 | 도메인 | 주기 | 변환 | lag | 확인 | 출처 | 상태 | 기본값 근거 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        if not r["id"]:
            continue
        status = []
        if r.get("active", "Y").strip().upper() in ("N", "FALSE", "0"):
            status.append("active=N")
        if r.get("duplicate_of", ""):
            status.append(f"중복→{r['duplicate_of']}")
        note = str(r.get("note", "")).replace("|", "/")
        rows.append(f"| {r['id']} | {r['name']} | {r['group']} | {r['domain']} | {r['freq']} | {r.get('transform', '') or 'none'} | {r['lag_days'] or '(기본값)'} | {r.get('lag_confirmed', '') or 'N'} | {r['source']} | {', '.join(status)} | {note} |")
    return "\n".join(rows)


def lag_evidence_table() -> str:
    """tools/lag_evidence.py 의 근거 표 → 가이드 표. 마스터의 실제 lag 와 다르면 표에 ⚠ 로 표시(문서·코드 불일치 방지)."""
    import sys as _s
    import pandas as pd

    _s.path.insert(0, str(ROOT / "tools"))
    from lag_evidence import E, VAR, evidence_table

    ev = evidence_table()
    m = pd.read_excel(ROOT / "data" / "variable_master.xlsx", sheet_name="variables", dtype=str).fillna("").set_index("id")
    rows = ["| id | 이름 | 주기 | 공표 자료 | 관측 표본 | 공개일−기준일(일) | lag = 최장+여유 | 확인 | 수정(revision) |", "|---|---|---|---|---|---|---|---|---|"]
    for vid, r in ev.items():
        name = m.loc[vid, "name"] if vid in m.index else ""
        fq = m.loc[vid, "freq"] if vid in m.index else ""
        mlag = m.loc[vid, "lag_days"] if vid in m.index else ""
        warn = "" if str(mlag).split(".")[0] == str(r["lag"]) else f" ⚠마스터 {mlag}"
        if r["group"] == "rule":
            rows.append(f"| {vid} | {name} | {fq} | (정의상) | – | – | {r['lag']}{warn} | {'Y' if r['confirmed'] else 'N'} | {r['note'].split('] ', 1)[1].replace('|', '/')} |")
        else:
            g = E[r["group"]]
            rows.append(f"| {vid} | {name} | {fq} | {r['group']} | {r['n_samples']}건 | {r['min_observed']}~{r['max_observed']} | {r['max_observed']}+{g['buffer']}={r['lag']}{warn} | {'Y' if r['confirmed'] else 'N'} | {g['rev']} |")
    rows.append("")
    rows.append("공표 자료 그룹별 관측 표본(기준일 → 공개일):")
    rows.append("")
    for gk, g in E.items():
        from datetime import date as _d
        smp = ", ".join(f"{a[:7]}→{b[5:]}({(_d.fromisoformat(b) - _d.fromisoformat(a)).days})" for a, b in g["samples"])
        rows.append(f"- **{gk}** — {smp}. 출처: {g['src']}")
    return "\n".join(rows)


def feature_presets() -> str:
    from engine.features import PRESETS

    rows = ["| 주기 | 수익률 lag | 이동평균 창 | 변동성 창 | RSI | MACD(fast, slow, signal) | 거래량 창 |", "|---|---|---|---|---|---|---|"]
    for f, P in PRESETS.items():
        rows.append(f"| {f} | {P['lags']} | {P['ma']} | {P['vol']} | {P['rsi']} | {P['macd']} | {P['volume']} |")
    return "\n".join(rows)


def build() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    outs = []
    combined = ["# TimeSeriesPredict 가이드 (자동 생성 합본)\n"]
    for p in sorted(SRC.glob("*.md")):
        s = p.read_text(encoding="utf-8")
        s = re.sub(r"\{\{PARAMS:([a-z_]+)\}\}", lambda m: params_table(m.group(1)), s)
        # 모델 흐름도 자리: 웹 도움말은 이 표시를 실제 그림(ModelFlow, /api/flow 로 잰 값)으로 바꾼다. 정적 문서에는 안내 문구만.
        s = re.sub(r"\{\{FLOW:([a-z_]+)\}\}", lambda m: f"<!--flow:{m.group(1)}-->\n*(그림: 웹 화면 도움말에서 보입니다 — 실험 화면의 흐름도와 같은 그림)*", s)
        s = s.replace("{{LAG_EVIDENCE}}", lag_evidence_table()) if "{{LAG_EVIDENCE}}" in s else s
        s = s.replace("{{VARIABLE_TABLE}}", variable_table()).replace("{{FEATURE_PRESETS}}", feature_presets())
        # 수식 그림: tools/formula_figs.py 가 만든 SVG. 웹은 /api/guide/fig/ 로, 합본 GUIDE.md 는 상대 경로로
        s = re.sub(r"\{\{FIG:([a-z_]+)\}\}", lambda m: f'<img class="fig" src="/api/guide/fig/{m.group(1)}.svg" alt="그림: {m.group(1)}">', s)
        o = OUT / p.name
        o.write_text(s, encoding="utf-8")
        outs.append(o)
        combined.append(s.replace("/api/guide/fig/", "guide_build/figs/").replace("/api/guide/walk/", "guide_build/walk/") + "\n\n---\n")
    (ROOT / "docs" / "GUIDE.md").write_text("\n".join(combined), encoding="utf-8")
    from tools.formula_figs import build_all
    build_all(OUT / "figs")
    return outs


if __name__ == "__main__":
    for o in build():
        print("built", o.name)
