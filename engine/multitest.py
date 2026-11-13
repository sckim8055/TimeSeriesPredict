"""다중비교 보정(Holm–Bonferroni)과 그 '가족' 정의 — 결과 화면(api)과 통계 재계산(tools/restat.py)이 같은 함수를 쓴다.

가족(같은 문제) = 모드 · 대상(매매는 종목, 예측 연구는 대상 변수) · 주기(매매) · k · 타겟 종류 · 정보 시점.
평가 구간은 가족 정의에 넣지 않는다. 2026-11-10 전에는 평가 구간이 글자 그대로 같아야 한 가족이었는데, 같은 종목·같은 주기·같은 k 의 결과가
평가 시작일이 며칠 다르다는 이유로 혼자 한 가족(m=1)이 되어 보정이 사라졌다(독립 검토 F3: 09_lstm 의 CW p=0.0058 이 m=7 로 보고됐지만
같은 일간·같은 종목 전체로는 m=8, 그 모드의 모든 검정으로는 m=10 이었다). 그래서 가족 안 보정값과 함께 '그 모드의 지금 엔진 결과 전부'를
한 가족으로 본 보정값(…_all)도 같이 낸다 — 가족을 어떻게 잡느냐에 결론이 달라지는지 보이기 위해서다.

한쪽 p값: DM 은 양측 검정이라 p 가 작아도 '더 나쁘다'일 수 있다. Holm 은 '모델이 기준보다 낫다'는 한쪽 p 에 적용한다(F13).
"""
from __future__ import annotations

import math


def family_key(r: dict) -> tuple:
    mode = r.get("mode", "trading")
    if mode == "forecast":
        return (mode, r.get("target_variable"), None, r.get("k"), r.get("target_kind"), r.get("info_time", "stamp"))
    t = r.get("ticker")
    return (mode, str(t).split("_")[0] if t else None, r.get("frequency"), r.get("k"), r.get("target_kind"), None)  # '091160_yf' 도 같은 종목


def better_one_sided(stat, p_two):
    """양측 p(통계량이 음수일수록 모델이 나음 — DM 의 약속) → '모델이 기준보다 낫다'의 한쪽 p. 통계량이 0 이상이면 0.5 이상이 된다."""
    if stat is None or p_two is None or not (math.isfinite(stat) and math.isfinite(p_two)):
        return None
    return p_two / 2 if stat < 0 else 1 - p_two / 2


def _adjust(members: list[dict], pkey: str, out: str) -> None:
    m = len(members)
    members.sort(key=lambda r: r[pkey])
    run = 0.0
    for i, r in enumerate(members):
        run = max(run, min(1.0, (m - i) * r[pkey]))  # p_(1) ≤ … ≤ p_(m): adj_(i) = max_{j≤i} min(1, (m−j+1)·p_(j))  (Holm 1979)
        r[out], r[out + "_m"] = run, m


def holm(rows: list[dict], pkey: str, out: str, *, skip=None) -> None:
    """rows 의 pkey(한쪽 p)에 Holm 보정을 적용해 out(가족 안), out+'_m'(가족 크기), out+'_all', out+'_all_m'(모드 전체를 한 가족으로)을 채운다.
    skip(r) 이 참이거나 p 가 없으면 건너뛴다(옛 계산 방식·봉인 평가·오류 줄)."""
    ok = [r for r in rows if not (skip and skip(r)) and r.get(pkey) is not None and math.isfinite(r[pkey])]
    fam: dict[tuple, list[dict]] = {}
    for r in ok:
        fam.setdefault(family_key(r), []).append(r)
    for members in fam.values():
        _adjust(members, pkey, out)
    by_mode: dict[str, list[dict]] = {}
    for r in ok:
        by_mode.setdefault(r.get("mode", "trading"), []).append(r)
    for members in by_mode.values():
        _adjust(members, pkey, out + "_all")
