"""수집 어댑터 규격. 어댑터 1개 = 파일 1개. 응답 원본(raw)을 먼저 저장하고 그 다음 표준 모양으로 바꾼다 (CLAUDE.md 8-2).

표준 출력
- 가격: DataFrame[date index, open, high, low, close, volume]
- 변수: DataFrame[variable_id, date, value]  (롱 테이블)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class RawCache:
    """API 원본 응답 저장(JSON). data/raw/<adapter>/<key>_<utc>.json + 받은 이력 data/raw/fetch_log.jsonl (추가만)."""

    def __init__(self, root: str | Path, adapter: str):
        self.dir = Path(root) / "raw" / adapter
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, payload) -> Path:
        """원본을 JSON 으로 저장하고 data/raw/fetch_log.jsonl 에 한 줄(시각·출처·키·파일·sha256·줄 수)을 덧붙인다.
        지우거나 덮어쓰지 않는다 — 같은 키를 다시 받으면 새 파일이 생긴다(언제 무엇을 받았는지와 공급처 수정값을 나중에 비교할 수 있게)."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(ch if ch.isalnum() or ch in "-_.^=" else "_" for ch in str(key))
        p = self.dir / f"{safe}_{ts}.json"
        n = 1
        while p.exists():  # 같은 초에 두 번 받아도 덮어쓰지 않음
            p = self.dir / f"{safe}_{ts}_{n}.json"; n += 1
        rows = len(payload) if isinstance(payload, (pd.DataFrame, list)) else None
        if isinstance(payload, pd.DataFrame):
            payload = json.loads(payload.to_json(orient="split", date_format="iso"))
        text = json.dumps(payload, ensure_ascii=False)
        p.write_text(text, encoding="utf-8")
        rec = {"utc": now.isoformat(), "adapter": self.dir.name, "key": str(key), "file": f"{self.dir.name}/{p.name}",
               "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "bytes": len(text.encode("utf-8")), "rows": rows}
        with open(self.dir.parent / "fetch_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return p


def to_price_frame(df: pd.DataFrame, mapping: dict[str, str], date_col: str) -> pd.DataFrame:
    """열 이름 매핑 → 표준 가격 프레임."""
    out = df.rename(columns=mapping)
    out["date"] = pd.to_datetime(out[date_col] if date_col in out.columns else out["date"])
    out = out.set_index("date").sort_index()
    out = out[["open", "high", "low", "close", "volume"]].astype(float)
    return out[~out.index.duplicated(keep="last")]


def scrub_secrets(text: object) -> str:
    """글자에서 접속 키를 지운다(*** 로). 통신 오류 문구에는 요청 주소가 통째로 들어 있고 그 주소에 키가 있다 — 그대로 화면·로그·manifest 에 쓰면 키가 새어 나간다(2026-11-12).
    지우는 것: ① 환경 변수 가운데 이름이 KEY·TOKEN·SECRET·PW·PASSWORD 로 끝나거나 그 말을 담은 것의 값(8자 이상) ② 주소의 api_key=·serviceKey=·key=·token= 뒤의 값."""
    import os
    import re

    s = str(text)
    for k, v in os.environ.items():
        if v and len(v) >= 8 and re.search(r"(KEY|TOKEN|SECRET|PASSWORD|_PW)", k.upper()):
            s = s.replace(v, "***")
    return re.sub(r"(?i)\b(api_key|apikey|servicekey|service_key|key|token|password)=([^&\s'\")]+)", r"\1=***", s)
