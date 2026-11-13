"""봉인 정책 — 대상(종목·대상 변수)마다 '이 날 이후는 봉인 구간'을 한 곳(seal_policy.json — 프로젝트 폴더 바로 아래)에 정해 두고, 모든 실행이 지키게 한다(2026-11-10).

왜 필요한가(독립 검토 F8)
  봉인 시작일은 실험 설정마다 따로 적는 칸(validation.sealed_from)이었다. 그 칸을 비우고 돌리면 보통 실험이 봉인 구간까지 채점했고,
  봉인 기록(sealed_log)에는 아무것도 남지 않았다. 실제로 repro_biva_* 결과 5건은 봉인 시작일 없이 F11 을 2025-02 까지 평가했는데,
  같은 대상의 f11_* 실험은 2024-01-01 부터를 봉인으로 두고 있었다. 봉인이 '설정 안의 약속'이면 약속을 안 적은 설정에는 봉인이 없다.

규칙 (정책 파일에 그 대상의 날짜 D 가 있을 때만 적용. 없으면 전과 같다)
  개발 실행
    · 설정의 봉인 시작일이 비어 있으면 → D 를 채워서 돌린다(결과의 config 와 seal_policy 에 남는다). 단 데이터 사용 종료일(end)이 D 보다 앞이면 그대로.
    · 설정의 봉인 시작일이 D 보다 뒤면 → 멈춘다(그 사이 [D, 설정값)는 봉인 구간인데 개발 평가가 채점하게 된다).
    · D 와 같거나 앞이면 그대로.
  봉인 평가(sealed=True)
    · 봉인 시작일이 비어 있으면 D. D 보다 앞이면 멈춘다(개발 구간을 봉인 평가로 채점하게 된다).
  일부러 넘기기
    · 원 논문의 분할을 그대로 재현하는 실험처럼 봉인 구간을 봐야만 하는 경우: 환경변수 TSP_SEAL_OVERRIDE 에 이유를 적는다(cli.py run --past-seal "이유").
      그러면 설정 그대로 돌리고, 결과(seal_policy.status = "override")와 시도 원장(kind = seal_override)에 이유가 남는다.
      그 결과는 '봉인 구간을 본 것'이므로, 그 대상의 봉인 평가는 더 이상 '처음 보는 기간'의 평가가 아니다 — 논문에 그대로 적어야 한다.

정책 파일 형식: {"trading": {"<종목>": "YYYY-MM-DD"}, "forecast": {"<대상 변수 id>": "YYYY-MM-DD"}}. '_' 로 시작하는 키는 메모.
다른 파일을 쓰려면 환경변수 TSP_SEAL_POLICY 에 경로(시험용). 파일이 없으면 정책 없음.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "seal_policy.json"


def policy_path() -> Path:
    return Path(os.environ.get("TSP_SEAL_POLICY") or DEFAULT_PATH)


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else policy_path()
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    out: dict = {}
    for mode in ("trading", "forecast"):
        out[mode] = {str(k): str(pd.Timestamp(v).date()) for k, v in (d.get(mode) or {}).items() if not str(k).startswith("_")}
    return out


def base_ticker(t) -> str | None:
    """종목 id 의 바탕: '091160_yf'·'091160_datago'·'091160_merged'(출처별·옛 이름의 가격 파일)도 같은 종목 '091160' 이다 —
    이름 끝만 바꾼 설정으로 봉인 정책을 비켜 가지 못하게 한다."""
    return str(t).split("_")[0] if t else None


def target_of(cfg) -> tuple[str, str | None]:
    return (cfg.mode, cfg.target_variable) if cfg.mode == "forecast" else (cfg.mode, base_ticker(cfg.ticker))


def policy_date(cfg, path: str | Path | None = None) -> str | None:
    mode, key = target_of(cfg)
    return (load(path).get(mode) or {}).get(str(key)) if key else None


def apply(cfg, *, sealed: bool, path: str | Path | None = None):
    """정책을 적용한 설정과 기록을 돌려준다: (cfg, note | None). 어기면 ValueError(쉬운 말로 무엇을 고칠지)."""
    mode, key = target_of(cfg)
    D = policy_date(cfg, path)
    if not D:
        return cfg, None
    note = {"target": key, "mode": mode, "policy_date": D, "configured": cfg.validation.sealed_from}
    dD = pd.Timestamp(D)
    sf = pd.Timestamp(cfg.validation.sealed_from) if cfg.validation.sealed_from else None
    reason = (os.environ.get("TSP_SEAL_OVERRIDE") or "").strip()

    def _with(d: str):
        return cfg.model_copy(update={"validation": cfg.validation.model_copy(update={"sealed_from": d})})

    if sealed:
        if sf is None:
            return _with(D), {**note, "status": "applied"}
        if sf < dD and not reason:
            raise ValueError(f"봉인 평가를 할 수 없습니다 — 설정의 봉인 시작일({sf.date()})이 이 대상({key})의 봉인 시작일({D})보다 앞입니다. "
                             f"그대로 돌리면 개발 구간({sf.date()} ~ {D})까지 봉인 평가로 채점합니다. 설정의 봉인 시작일을 {D} 로 맞추세요.")
        return cfg, {**note, "status": "override" if (sf < dD) else "ok", **({"reason": reason} if sf < dD else {})}
    if sf is not None and sf <= dD:
        return cfg, {**note, "status": "ok"}
    if sf is None and cfg.end and pd.Timestamp(cfg.end) < dD:
        return cfg, {**note, "status": "ok_end_before_seal"}
    if reason:
        return cfg, {**note, "status": "override", "reason": reason}
    if sf is None:
        return _with(D), {**note, "status": "applied"}
    raise ValueError(f"실험을 돌릴 수 없습니다 — 이 대상({key})은 {D} 부터가 봉인 구간인데, 설정의 봉인 시작일이 {sf.date()} 입니다. "
                     f"그대로 돌리면 봉인 구간의 일부({D} ~ {sf.date()})를 개발 평가가 채점합니다. 봉인 시작일을 {D} 로 바꾸세요. "
                     f"(봉인 구간을 일부러 봐야 하는 재현 실험이면: python cli.py run <설정> --past-seal \"이유\")")
