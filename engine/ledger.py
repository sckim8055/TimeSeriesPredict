"""시도 원장 — 개발 시도 횟수 N(Deflated Sharpe 의 N)의 근거를 지울 수 없게 남긴다(2026-11-10).

전에는 results/<mode>/experiment_counter.json 의 숫자 하나였다: 무엇을 셌는지 알 수 없고, 평문이라 고치거나 지울 수 있었고(화면 서버의 삭제 API 로도),
같은 설정을 다시 돌려도 +1 이었다. 이제는 results/<mode>/trial_ledger.jsonl 에 시도마다 한 줄을 덧붙인다.

  · 줄마다 seq(차례), utc, kind, config_hash, name, model, run_id, result, prev(바로 앞 줄의 sha256) — 앞 줄을 고치거나 빼면 뒤 줄의 prev 가 맞지 않게 된다(verify).
  · kind: genesis(원장 시작 — 그 전의 횟수는 옛 카운터 값으로만 알 수 있음) · trial(실험) · ablation(변수 제거 실험) · ensemble · sealed(봉인 평가 — 개발 시도로 세지 않음) · deleted(결과 삭제 기록)
  · 개발 시도 수 N = genesis 의 count + (trial·ablation·ensemble 줄 수). 같은 설정의 재실행도 한 줄이다(다시 본 것도 본 것).
    서로 다른 설정의 수는 distinct_configs() 로 따로 센다(원장 시작 뒤의 것만 알 수 있음).
  · experiment_counter.json 은 화면 표시용 사본으로 계속 쓰지만 근거는 원장이다. 사본이 지워져도 N 은 변하지 않는다.

보강(2026-11-10 독립 검토 F9·F15)
  · 끝 줄을 지우면 사슬만으로는 알 수 없다(사슬은 '앞 줄'만 지킨다). 그래서 원장의 머리(마지막 줄의 차례·sha256)를 두 곳에 따로 적는다:
    화면용 사본(experiment_counter.json 의 ledger_seq·ledger_head)과 각 결과 파일(run_record.ledger_before — 그 실험을 시작할 때의 머리).
    verify() 가 이 둘과 원장을 맞춰 본다: 사본이 가리키는 줄이 없거나 다르면, 결과 파일이 가리키는 줄이 없거나 다르면, 결과의 실행 번호(run_id)가
    원장에 없으면 '잘림·바뀜'으로 알린다.
  · 덧붙이기는 잠금 파일(trial_ledger.lock)을 잡고 한다 — 두 실행이 동시에 쓰면 같은 차례 번호가 두 번 나와 verify 가 영영 실패했다.
  · 줄은 ASCII 로만 쓴다(이름에 줄 나눔 문자가 들어가도 한 줄). 읽을 때는 '\n' 으로만 나눈다. 덧붙이기 전에 파일이 줄 끝으로 끝나는지 보고 아니면 줄을 바꾼다.
  · 읽을 수 없는 줄(쓰다 끊긴 줄 등)은 '시도 1회'로 센다 — N 이 줄어드는 쪽으로 틀리지 않게. verify 는 그 줄을 알린다.

한계(정직하게): 로컬 파일이라 원장 전체를 지우고 새로 만들면 막을 수 없다. 그 경우 genesis 줄의 시각이 새로 찍히고, git 에 올린 원장과 달라져 드러난다.
즉 '고칠 수 없다'가 아니라 '고치면 드러난다'이다. 원장은 git 으로 관리할 것.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

LEDGER = "trial_ledger.jsonl"
COUNTER = "experiment_counter.json"
LOCK = "trial_ledger.lock"
DEV_KINDS = ("trial", "ablation", "ensemble")
CORRUPT = "corrupt"  # 읽을 수 없는 줄 — 개발 시도 1회로 센다(보수적)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _read(res_dir: str | Path) -> list[str]:
    p = Path(res_dir) / LEDGER
    if not p.exists():
        return []
    # '\n' 으로만 나눈다 — splitlines() 는 U+2028·NEL 같은 문자에서도 줄을 나눠, 이름에 그런 문자가 있으면 한 줄이 둘로 갈렸다
    return [ln.rstrip("\r") for ln in p.read_text(encoding="utf-8").split("\n") if ln.strip()]


def _parse(ln: str) -> dict:
    try:
        e = json.loads(ln)
        return e if isinstance(e, dict) else {"kind": CORRUPT, "raw": ln[:200]}
    except Exception:  # noqa: BLE001
        return {"kind": CORRUPT, "raw": ln[:200]}


def entries(res_dir: str | Path) -> list[dict]:
    return [_parse(ln) for ln in _read(res_dir)]


@contextlib.contextmanager
def _locked(res_dir: Path, timeout: float = 30.0, stale: float = 120.0):
    """원장 쓰기 잠금(잠금 파일을 '없을 때만 만들기'로 잡는다 — 윈도우·리눅스 공통). stale 초보다 오래된 잠금은 죽은 실행의 것으로 보고 치운다."""
    res_dir.mkdir(parents=True, exist_ok=True)
    lk = res_dir / LOCK
    t0 = time.time()
    while True:
        try:
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {_now()}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lk.stat().st_mtime > stale:
                    lk.unlink()
                    continue
            except OSError:
                continue
            if time.time() - t0 > timeout:
                raise TimeoutError(f"원장 잠금({lk})을 {timeout:.0f}초 동안 잡지 못했습니다 — 다른 실행이 쓰는 중이거나 죽은 실행의 잠금 파일이 남았습니다")
            time.sleep(0.02)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            lk.unlink()


def _append_unlocked(res_dir: Path, entry: dict) -> dict:
    p = res_dir / LEDGER
    lines = _read(res_dir)
    e = {"seq": len(lines), "utc": _now(), **entry, "prev": _sha(lines[-1]) if lines else None}
    with open(p, "ab") as f:
        if p.stat().st_size and not p.read_bytes().endswith(b"\n"):
            f.write(b"\n")  # 앞 줄이 쓰다 끊겼으면 새 줄에서 시작한다(끊긴 줄에 이어 붙이지 않게)
        f.write((json.dumps(e, ensure_ascii=True, default=str) + "\n").encode("ascii"))
    return e


def _append(res_dir: Path, entry: dict) -> dict:
    with _locked(res_dir):
        return _append_unlocked(res_dir, entry)


def ensure(res_dir: str | Path) -> None:
    """원장이 없으면 genesis 줄로 시작한다. 그 전의 횟수는 옛 카운터 값(없으면 0)이고, 그때 폴더에 있던 결과 이름을 함께 적는다."""
    d = Path(res_dir)
    if (d / LEDGER).exists():
        return
    with _locked(d):
        if (d / LEDGER).exists():  # 잠금을 기다리는 사이 다른 실행이 만들었다
            return
        c = d / COUNTER
        unreadable = False
        try:
            n0 = int(json.loads(c.read_text(encoding="utf-8")).get("count", 0)) if c.exists() else 0
        except Exception:  # noqa: BLE001 — 깨진 카운터는 0 으로 두고 그 사실을 적는다
            n0, unreadable = 0, True
        stems = sorted(p.stem for p in d.glob("*.json") if p.name not in (COUNTER, "sealed_log.json")) if d.exists() else []
        note = ("원장 시작. 옛 카운터 파일을 읽을 수 없어 이전 시도 수를 0 으로 두었다 — 실제 이전 시도 수는 알 수 없다(아래 결과 목록보다 적지 않다)." if unreadable else
                "원장 시작. 이전 시도 수는 experiment_counter.json 의 값이며 개별 내역은 없다(같은 설정의 재실행·지운 결과 포함).")
        _append_unlocked(d, {"kind": "genesis", "count": n0, "counter_unreadable": unreadable, "note": note, "results_on_disk": stems})


def count(res_dir: str | Path) -> int:
    """개발 시도 수 N. 원장이 없으면 옛 카운터 값(읽기만 — 원장을 만들지 않는다). 읽을 수 없는 줄은 시도 1회로 센다."""
    d = Path(res_dir)
    es = entries(d)
    if not es:
        c = d / COUNTER
        try:
            return int(json.loads(c.read_text(encoding="utf-8")).get("count", 0)) if c.exists() else 0
        except Exception:  # noqa: BLE001
            return 0
    base = next((int(e.get("count", 0)) for e in es if e.get("kind") == "genesis"), 0)
    return base + sum(1 for e in es if e.get("kind") in DEV_KINDS or e.get("kind") == CORRUPT)


def head(res_dir: str | Path) -> dict | None:
    """원장의 머리: 마지막 줄의 차례 번호와 sha256. 원장이 없으면 None. 결과 파일(run_record.ledger_before)과 화면용 사본에 적어 두는 값."""
    lines = _read(res_dir)
    return {"seq": len(lines) - 1, "sha": _sha(lines[-1])} if lines else None


def record(res_dir: str | Path, kind: str, **fields) -> int:
    """원장에 한 줄을 덧붙이고 개발 시도 수 N 을 돌려준다. 화면용 사본(experiment_counter.json)에 N 과 원장의 머리를 같이 적는다."""
    d = Path(res_dir)
    ensure(d)
    with _locked(d):
        _append_unlocked(d, {"kind": kind, **fields})
        n = count(d)
        h = head(d) or {}
        try:
            (d / COUNTER).write_text(json.dumps({"count": n, "ledger_seq": h.get("seq"), "ledger_head": h.get("sha")}), encoding="utf-8")
        except OSError:
            pass  # 사본은 표시용 — 못 써도 원장이 근거
    return n


def distinct_configs(res_dir: str | Path) -> int:
    """원장 시작 뒤 돌린 서로 다른 설정(config_hash)의 수 — 같은 설정의 재실행은 한 번으로."""
    return len({e.get("config_hash") for e in entries(res_dir) if e.get("kind") in DEV_KINDS and e.get("config_hash")})


def verify(res_dir: str | Path) -> dict:
    """원장 검사.
    ① 사슬: 각 줄의 prev 가 바로 앞 줄의 sha256 과 같은지, seq 가 0부터 이어지는지, 읽을 수 없는 줄이 없는지.
    ② 머리(끝 줄을 지운 경우를 잡는다): 화면용 사본이 적어 둔 머리(ledger_seq·ledger_head)가 원장의 그 줄과 같은지.
    ③ 결과 파일(지금 폴더와 history/): run_record.ledger_before 가 가리키는 줄이 원장에 그대로 있는지, 그 결과의 run_id 가 원장에 있는지.
    하나라도 어긋나면 ok=False 와 이유. ②③ 은 그 정보가 적힌 것(2026-11-10 뒤의 사본·결과)만 본다."""
    d = Path(res_dir)
    lines = _read(d)
    for i, ln in enumerate(lines):
        e = _parse(ln)
        if e.get("kind") == CORRUPT:
            return {"ok": False, "lines": len(lines), "broken_at": i, "why": "줄을 읽을 수 없음"}
        if e.get("seq") != i:
            return {"ok": False, "lines": len(lines), "broken_at": i, "why": f"차례 번호가 {e.get('seq')} (기대 {i}) — 줄이 빠졌거나 끼어듦"}
        want = _sha(lines[i - 1]) if i else None
        if e.get("prev") != want:
            return {"ok": False, "lines": len(lines), "broken_at": i, "why": "앞 줄이 바뀌었음(prev 불일치)"}
    out = {"ok": True, "lines": len(lines), "n_dev_trials": count(d)}
    if not lines:
        return out

    def _line_ok(seq, sha) -> bool:
        return isinstance(seq, int) and 0 <= seq < len(lines) and _sha(lines[seq]) == sha

    c = d / COUNTER
    try:
        cj = json.loads(c.read_text(encoding="utf-8")) if c.exists() else {}
    except Exception:  # noqa: BLE001
        cj = {}
    if cj.get("ledger_head") is not None and not _line_ok(cj.get("ledger_seq"), cj.get("ledger_head")):
        return {**out, "ok": False, "why": f"화면용 사본이 적어 둔 원장 머리(차례 {cj.get('ledger_seq')})가 원장에 없거나 다름 — 원장의 끝 줄이 지워졌거나 바뀌었음"}
    run_ids = {e.get("run_id") for e in map(_parse, lines) if e.get("run_id")}
    bad = []
    for p in list(d.glob("*.json")) + list((d / "history").glob("*.json") if (d / "history").exists() else []):
        if p.name in (COUNTER, "sealed_log.json"):
            continue
        try:
            rr = (json.loads(p.read_text(encoding="utf-8")) or {}).get("run_record") or {}
        except Exception:  # noqa: BLE001
            continue
        lb = rr.get("ledger_before")
        if not lb:
            continue
        if not _line_ok(lb.get("seq"), lb.get("sha")):
            bad.append(f"{p.name}: 시작할 때의 원장 머리(차례 {lb.get('seq')})가 원장에 없거나 다름")
        elif rr.get("run_id") and rr["run_id"] not in run_ids:
            bad.append(f"{p.name}: 이 결과의 실행 번호 {rr['run_id']} 가 원장에 없음")
    if bad:
        return {**out, "ok": False, "why": "결과 파일과 원장이 맞지 않음 — 원장의 줄이 지워졌거나 바뀌었음", "mismatch": bad[:20], "n_mismatch": len(bad)}
    return out
