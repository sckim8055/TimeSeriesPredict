"""가이드 문서 생성 — docs/guide/*.md 의 자리표시자를 코드에서 뽑은 표로 채운다.

  {{PARAMS:<모델파일>}}   models/<파일>.py 의 PARAMS → 표 (이름 / 뜻 / 기본값 / 범위)
  {{VARIABLE_TABLE}}     data/variable_master.xlsx → 변수 표
  {{FEATURE_PRESETS}}    engine/features.PRESETS → 주기별 창 길이 표
  {{TRANSFORMS}}         engine.experiment.TRANSFORM_OPTIONS → 변환 표(화면과 같은 목록)
  {{TECH_GROUPS}} · {{TECH_COLUMNS}} · {{TECH_EXAMPLE}} · {{TECH_ORIGIN}}  engine.features 의 기술지표 설명·열 표·예시·출처(변수 사전 화면과 같은 글)
  {{TRANSFORM_EXAMPLES}} 엔진 apply_transform 을 돌린 변환 예시(월간 지수·금리·성장률)
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


def transform_table() -> str:
    """{{TRANSFORMS}} — engine.experiment.TRANSFORM_OPTIONS(화면 세 곳과 같은 목록)를 표로. 2026-11-14."""
    from engine.experiment import TRANSFORM_OPTIONS
    rows = ["| 변환 | 이름 | 계산식 | 뜻 | 맞는 자료 | 주의 |", "|---|---|---|---|---|---|"]
    for o in TRANSFORM_OPTIONS:
        rows.append(f"| `{o['value']}` | {o['name']} | {o['formula']} | {o['meaning']} | {o['use']} | {o.get('caution', '')} |")
    return "\n".join(rows)


def transform_examples() -> str:
    """{{TRANSFORM_EXAMPLES}} — 엔진 apply_transform 을 실제로 돌린 예시(숫자를 손으로 적지 않는다). 2026-11-14."""
    import numpy as np
    import pandas as pd
    from engine.experiment import TRANSFORM_OPTIONS, apply_transform
    # 월간 지수 25개월(마지막 달 = t). 앞쪽 값은 아래 '쓴 값' 줄에 적은 것만 계산에 쓰인다.
    s = pd.Series([100.0] * 5 + [108.0] + [120.0] * 7 + [150.0] + [155.0] * 5 + [160.0, 165.0, 170.0, 172.0, 175.0, 180.0])
    x = {k: float(s.iloc[-1 - k]) for k in (0, 1, 4, 5, 12, 20)}
    def g(v):
        return f"{v:g}"
    calc = {"none": f"{g(x[0])}", "log": f"ln {g(x[0])}", "logdiff": f"ln {g(x[0])} − ln {g(x[1])}"}
    for n in (1, 4, 5, 12, 20):
        sfx = "" if n == 1 else str(n)
        calc[f"pct{sfx}"] = f"{g(x[0])} ÷ {g(x[n])} − 1"
        calc[f"diff{sfx}"] = f"{g(x[0])} − {g(x[n])}"
    out = ["**예 1 — 월간 지수(가격·물가지수 같은 수준값).** 마지막 달 x_t = "
           f"{g(x[0])}, 1개월 전 {g(x[1])}, 4개월 전 {g(x[4])}, 5개월 전 {g(x[5])}, 12개월 전 {g(x[12])}, 20개월 전 {g(x[20])}. "
           "아래 결과는 엔진(`apply_transform`)이 실제로 계산한 값입니다.", "",
           "| 변환 | 계산 | 결과 | 읽는 법 |", "|---|---|---|---|"]
    for o in TRANSFORM_OPTIONS:
        v = o["value"]
        r = float(apply_transform(s, v).iloc[-1])
        if v.startswith("pct"):
            read = f"{r * 100:.2f}% 올랐다"
        elif v.startswith("diff"):
            read = f"{r:g} 올랐다(지수 단위)"
        elif v == "logdiff":
            read = f"로그 수익률 {r:.4f} ≈ 변화율 {np.expm1(r) * 100:.2f}%"
        elif v == "log":
            read = "수준의 로그 — 변화가 아니다"
        else:
            read = "받은 값 그대로"
        out.append(f"| `{v}` | {calc.get(v, '')} | {r:.4f} | {read} |")
    # 예 2: 금리(%) — diff 가 맞고 pct 는 왜곡
    rate = pd.Series([3.25, 3.50])
    d, p_ = float(apply_transform(rate, "diff").iloc[-1]), float(apply_transform(rate, "pct").iloc[-1])
    out += ["", f"**예 2 — 금리(% 단위).** 기준금리 3.25% → 3.50%: `diff` = {d:g} (0.25%포인트 올랐다 — 맞음). "
                f"`pct` = {p_:.4f} = {p_ * 100:.2f}% (금리가 '7.69% 올랐다'는 뜻이 되어 크기가 금리 수준에 따라 달라짐 — 금리에는 쓰지 않음)."]
    # 예 3: 이미 성장률인 값 — pct 는 0 에서 깨진다
    gr = pd.Series([1.2, 0.0, 0.8])
    pv = apply_transform(gr, "pct").tolist()
    out += ["", f"**예 3 — 이미 성장률인 값(GDP 전기 대비 %).** 1.2% → 0.0% → 0.8%: `none` 으로 그대로 씁니다. `pct` 를 쓰면 "
                f"'성장률의 변화율'이 되어 뜻이 없고, 0% 다음 분기에서 {pv[2]} (나눗셈 0 → 무한대)가 나옵니다 — 실제로 A1 이 그래서 `none` 으로 바뀌었습니다(2026-11-04)."]
    return "\n".join(out)


def tech_groups_doc() -> str:
    """{{TECH_GROUPS}} — 묶음별 '왜 있나·비유·읽는 법·계산·주의'(engine.features.TECH_GROUP_DOCS, 변수 사전 화면과 같은 글). 2026-11-14."""
    from engine.features import TECH_GROUP_DOCS
    out = []
    for i, d in enumerate(TECH_GROUP_DOCS, 1):
        out += [f"#### 5-{i}. {d['group']} (`{d['prefix']}…`)", "",
                f"- **왜 있나**: {d['why']}",
                f"- **쉽게 말하면**: {d['analogy']}",
                f"- **읽는 법**: {d['read']}",
                f"- **계산**: {d['formula']}",
                f"- **주의**: {d['caution']}",
                f"- **누가 만들었나·기간은 어디서**: {d['origin']}", ""]
    return "\n".join(out).rstrip()


def tech_origin_doc() -> str:
    """{{TECH_ORIGIN}} — 18개 묶음과 주기별 창 길이를 누가 정했나 + 창 길이를 대략의 주(週) 수로 바꾼 표(engine.features.TECH_ORIGIN·preset_calendar_weeks,
    변수 사전 화면과 같은 글). 2026-11-14."""
    from engine.features import TECH_ORIGIN, preset_calendar_weeks
    fmt = lambda c: " · ".join(f"{p}(≈{w:g}주)" for p, w in zip(c["periods"], c["weeks"]))
    rows = ["| 묶음 | 일간(D) 창 | 주간(W) 창 | 월간(M) 창 |", "|---|---|---|---|"]
    for r in preset_calendar_weeks():
        rows.append(f"| {r['label']} | {fmt(r['D'])} | {fmt(r['W'])} | {fmt(r['M'])} |")
    return "\n".join([f"- {TECH_ORIGIN['summary']}", f"- {TECH_ORIGIN['not_standard']}", f"- {TECH_ORIGIN['weekly_monthly']}", "",
                      "창 길이를 대략의 주(週) 수로 바꾸면(일 ÷ 5, 월 × 52/12):", ""] + rows)


def tech_columns_table() -> str:
    """{{TECH_COLUMNS}} — 일간 18개 열의 이름·묶음·계산식(engine.features 의 설명 규칙 그대로). 2026-11-14."""
    from engine.features import feature_description, technical_columns, technical_group
    rows = ["| 열 이름 | 묶음 | 계산 |", "|---|---|---|"]
    for c in technical_columns("D"):
        rows.append(f"| `{c}` | {technical_group(c)} | {feature_description(c)} |")
    return "\n".join(rows)


def tech_example_doc() -> str:
    """{{TECH_EXAMPLE}} — 엔진 함수로 계산한 예시(engine.features.technical_example). 2026-11-14."""
    from engine.features import technical_example
    ex = technical_example()
    v, c, vol = ex["values"], ex["close"], ex["volume"]
    head = "| 날 | " + " | ".join(str(i) for i in range(1, len(c) + 1)) + " |"
    out = [f"10거래일 종가와 거래량에 엔진 함수를 그대로 적용한 **마지막 날(10일째)** 값입니다. 창 길이: {ex['windows']}.", "",
           head, "|---|" + "---|" * len(c),
           "| 종가 | " + " | ".join(f"{x:g}" for x in c) + " |",
           "| 거래량 | " + " | ".join(f"{x:g}" for x in vol) + " |", "",
           "| 지표 | 계산 | 결과 | 읽는 법 |", "|---|---|---|---|",
           f"| `A_ret_1` | {c[-1]:g} ÷ {c[-2]:g} − 1 | {v['A_ret_1']:.4f} | 하루에 {v['A_ret_1'] * 100:.2f}% 올랐다 |",
           f"| `A_ret_5` | {c[-1]:g} ÷ {c[-6]:g} − 1 | {v['A_ret_5']:.4f} | 5일 동안 {v['A_ret_5'] * 100:.2f}% 올랐다 |",
           f"| `A_ma_ratio_5` | {c[-1]:g} ÷ 평균({', '.join(f'{x:g}' for x in c[-5:])} = {sum(c[-5:]) / 5:g}) − 1 | {v['A_ma_ratio_5']:.4f} | 최근 5일 평균보다 {v['A_ma_ratio_5'] * 100:.2f}% 위 |",
           f"| `A_vol_ratio_5` | {vol[-1]:g} ÷ 평균({', '.join(f'{x:g}' for x in vol[-5:])} = {sum(vol[-5:]) / 5:g}) − 1 | {v['A_vol_ratio_5']:.4f} | 거래가 평소의 {1 + v['A_vol_ratio_5']:.2f}배 |",
           f"| `A_vol_5` | 최근 5일 로그수익률의 표준편차 × √252 | {v['A_vol_5']:.4f} | 1년으로 환산하면 약 {v['A_vol_5'] * 100:.0f}% 출렁이는 정도 |",
           f"| `A_rsi_5` | 100 − 100 ÷ (1 + 상승폭 평균 ÷ 하락폭 평균) | {v['A_rsi_5']:.1f} | {'70 이상 — 관례상 과열 쪽' if v['A_rsi_5'] >= 70 else ('30 이하 — 관례상 침체 쪽' if v['A_rsi_5'] <= 30 else '30~70 — 중간')} |",
           f"| `A_macd` | (3일 지수평균 − 6일 지수평균) ÷ 종가 | {v['A_macd']:.4f} | {'짧은 평균이 긴 평균보다 위 — 오름세' if v['A_macd'] > 0 else '짧은 평균이 긴 평균보다 아래 — 내림세'} |",
           f"| `A_macd_hist` | MACD − 시그널 | {v['A_macd_hist']:.4f} | {'오름세가 강해지는 중' if v['A_macd_hist'] > 0 else '오름세가 약해지는 중'} |"]
    return "\n".join(out)


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
        for key, fn in (("{{TECH_GROUPS}}", tech_groups_doc), ("{{TECH_COLUMNS}}", tech_columns_table), ("{{TECH_EXAMPLE}}", tech_example_doc), ("{{TECH_ORIGIN}}", tech_origin_doc)):
            if key in s:
                s = s.replace(key, fn())
        if "{{TRANSFORMS}}" in s:
            s = s.replace("{{TRANSFORMS}}", transform_table())
        if "{{TRANSFORM_EXAMPLES}}" in s:
            s = s.replace("{{TRANSFORM_EXAMPLES}}", transform_examples())
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
