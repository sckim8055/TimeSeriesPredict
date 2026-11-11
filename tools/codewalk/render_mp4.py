"""코드 따라가기 HTML → MP4(1920×1080, 30fps).

  python -m tools.codewalk.render_mp4 docs/guide_build/walk/ridge.html [--out ….mp4] [--silent]

녹음이 있으면(HTML 의 S.audio — tools/narrate.py 가 연결) 설명 음성 + 큰 자막:
  단계마다 자막 줄 수만큼 화면을 찍고(?film=1&cap=1, setCap(k)), 그 줄을 읽는 동안 보여 준다(줄의 시작 시각 = 녹음의 글자 시각).
  단계 길이 = 앞 여백 0.6초 + 녹음 길이 + 뒤 여백 1.0초. 녹음이 없는 단계는 자막 줄을 고르게 나눈 시간 + 무음.
녹음이 없으면(또는 --silent) 예전처럼 단계마다 한 장 + 무음, 단계 시간 = 4초 + 설명 글자 × 0.085초 + … (8~40초).
방법: 단계마다 장면 파일(같은 크기·fps·코덱·음성 형식)을 만들고 concat 으로 잇는다(다시 인코딩 없음). 장면 파일은 .part.mp4 로 쓰고 ffmpeg 가 끝나면 이름을 바꾼다.
--work DIR: 찍은 화면·장면 파일을 DIR 에 남겨, 끊긴 뒤 같은 명령으로 다시 돌리면 이미 만든 단계는 건너뛴다(HTML 이 바뀌면 DIR 을 비우고 처음부터).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEAD, TAIL = 0.6, 1.0
VENC = ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", "30"]
AENC = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "1"]


def step_seconds(st: dict) -> float:
    if st.get("dur"):
        return float(st["dur"])
    t = lambda s: len(re.sub(r"<[^>]+>", "", s or ""))
    return max(8.0, min(40.0, 4 + 0.085 * t(st.get("narr")) + 0.03 * t(st.get("formula")) + 0.03 * t(st.get("hint"))))


def load_script(html: Path) -> dict:
    h = html.read_text(encoding="utf-8")
    i = h.index("const S = ") + len("const S = ")
    j = h.index(";\nconst $ =")
    return json.loads(h[i:j])


async def shoot(html: Path, outdir: Path, plan: list[int], cap: bool) -> None:
    """plan[i] = 단계 i 에서 찍을 장 수(자막 줄 수, 자막을 안 쓰면 1). 이미 다 찍힌 단계는 건너뛴다."""
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1920, "height": 1080})
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(html.resolve().as_uri() + ("?film=1&cap=1" if cap else "?film=1"))
        await pg.wait_for_timeout(600)
        for i, n in enumerate(plan):
            if all((outdir / f"s{i:03d}_{k:02d}.jpg").exists() for k in range(n)):
                continue
            await pg.evaluate(f"goto({i})")
            await pg.wait_for_timeout(180)
            for k in range(n):
                if cap:
                    await pg.evaluate(f"setCap({k})")
                    await pg.wait_for_timeout(40)
                tmpf = outdir / f"s{i:03d}_{k:02d}.part.jpg"
                await pg.screenshot(path=str(tmpf), type="jpeg", quality=92)
                tmpf.rename(outdir / f"s{i:03d}_{k:02d}.jpg")
        await b.close()
        if errs:
            raise RuntimeError("플레이어 오류: " + " / ".join(errs))


def _scene(imgs: list[tuple[Path, float]], audio: Path | None, total: float, out: Path) -> None:
    """그림(장마다 보여 줄 초) + 음성(앞 여백 LEAD 뒤에 시작) → 장면 파일 하나."""
    lst = out.with_suffix(".txt")
    lines = []
    for f, d in imgs:
        lines += [f"file '{f.as_posix()}'", f"duration {d:.3f}"]
    lines.append(f"file '{imgs[-1][0].as_posix()}'")          # concat 은 마지막 그림의 duration 을 쓰려면 한 번 더 적어야 한다
    lst.write_text("\n".join(lines), encoding="utf-8")
    part = out.with_name(out.stem + ".part.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if audio is not None:
        ms = int(LEAD * 1000)
        cmd += ["-i", str(audio), "-filter_complex", f"[0:v]fps=30,format=yuv420p,fade=t=in:st=0:d=0.25[v];[1:a]adelay={ms}:all=1,apad,atrim=0:{total:.3f}[a]",
                "-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-filter_complex", "[0:v]fps=30,format=yuv420p,fade=t=in:st=0:d=0.25[v]", "-map", "[v]", "-map", "1:a"]
    cmd += ["-t", f"{total:.3f}", *VENC, *AENC, str(part)]
    subprocess.run(cmd, check=True)
    part.rename(out)


def render(html: Path, out: Path, silent: bool = False, work: Path | None = None) -> Path:
    import hashlib
    S = load_script(html)
    steps = S["steps"]
    A = (S.get("audio") or {}) if not silent else {}
    use = bool(A.get("steps"))
    adir = html.parent / A.get("dir", "")
    plan = [max(1, len(st.get("captions") or [])) if use else 1 for st in steps]
    if work:
        tmp = Path(work).resolve(); tmp.mkdir(parents=True, exist_ok=True)        # concat 목록은 목록 파일 기준 경로라 절대 경로로
        fp = hashlib.sha1(html.read_bytes() + (b"silent" if silent else b"")).hexdigest()
        if (tmp / "fingerprint").exists() and (tmp / "fingerprint").read_text() != fp:      # HTML 이 바뀜 → 처음부터
            shutil.rmtree(tmp); tmp.mkdir(parents=True)
        (tmp / "fingerprint").write_text(fp)
    else:
        tmp = Path(tempfile.mkdtemp(prefix="walk_"))
    try:
        asyncio.run(shoot(html, tmp, plan, cap=use))
        lst, total_all, narrated = [], 0.0, 0
        for i, st in enumerate(steps):
            a = A.get("steps", {}).get(str(i)) if use else None
            n = plan[i]
            if a:
                total = LEAD + float(a["duration"]) + TAIL
                starts = [0.0] + [LEAD + float(g[0]) for g in a["segs"][1:]]
                ends = starts[1:] + [total]
                imgs = [(tmp / f"s{i:03d}_{k:02d}.jpg", max(0.04, ends[k] - starts[k])) for k in range(n)]
                if not (tmp / f"p{i:03d}.mp4").exists():
                    _scene(imgs, adir / a["file"], total, tmp / f"p{i:03d}.mp4")
                narrated += 1
            else:
                total = step_seconds(st)
                imgs = [(tmp / f"s{i:03d}_{k:02d}.jpg", total / n) for k in range(n)]
                if not (tmp / f"p{i:03d}.mp4").exists():
                    _scene(imgs, None, total, tmp / f"p{i:03d}.mp4")
            lst.append(f"file '{(tmp / f'p{i:03d}.mp4').as_posix()}'"); total_all += total
        (tmp / "list.txt").write_text("\n".join(lst), encoding="utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        out_tmp = out.with_name(out.stem + ".part.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"), "-c", "copy",
                        "-movflags", "+faststart", str(out_tmp)], check=True)
        out_tmp.replace(out)
        print(f"저장: {out} · 단계 {len(steps)} · 음성 {narrated}/{len(steps)}" + (f" ({A.get('engine')} · {A.get('voice')})" if use else " (무음)")
              + f" · 길이 {int(total_all // 60)}분 {int(total_all % 60)}초 · {out.stat().st_size / 1e6:.1f} MB")
        return out
    finally:
        if not work:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("--out")
    ap.add_argument("--silent", action="store_true", help="녹음이 있어도 무음·자막 없이(예전 방식)")
    ap.add_argument("--work", help="중간 파일을 남길 폴더(끊기면 같은 명령으로 이어서)")
    a = ap.parse_args()
    h = (ROOT / a.html) if not Path(a.html).is_absolute() else Path(a.html)
    render(h, Path(a.out) if a.out else h.with_suffix(".mp4"), a.silent, Path(a.work) if a.work else None)
