"""줄 단위 실행 추적기 — 지정한 폴더의 파이썬 코드가 '어느 줄을 어떤 순서로' 실행했고 '그 줄이 어떤 텐서를 만들었는지'를 기록한다.

sys.settrace 로 그 폴더의 함수에 들어갈 때만 줄 이벤트를 켜고(PyTorch 내부는 추적하지 않음), 줄이 바뀔 때마다 지역 변수 중
새로 생기거나 바뀐 텐서(같은 이름이라도 다른 객체이거나 _version 이 바뀐 것)를 '방금 실행한 줄'의 결과로 붙인다.
계산에는 손대지 않는다(읽기만). 결과: Event 목록 — file, line, func, code(원본 줄), locals_meta(t·direct 같은 반복 정보), produced{이름: 텐서 복사본}.
"""
from __future__ import annotations

import linecache
import os
import sys
from dataclasses import dataclass, field

import torch


@dataclass
class Event:
    seq: int
    file: str
    line: int
    func: str
    code: str
    depth: int
    meta: dict = field(default_factory=dict)
    produced: dict = field(default_factory=dict)
    ret: object = None


class LineTracer:
    def __init__(self, roots: list[str], meta_names=("t", "direct", "epoch")):
        self.roots = [os.path.abspath(r) for r in roots]
        self.meta_names = meta_names
        self.events: list[Event] = []
        self._snap: dict[int, dict] = {}
        self._last: dict[int, Event] = {}
        self._depth = 0

    def _want(self, fn: str) -> bool:
        fn = os.path.abspath(fn)
        return any(fn.startswith(r) for r in self.roots)

    @staticmethod
    def _key(v):
        return (id(v), getattr(v, "_version", 0)) if isinstance(v, torch.Tensor) else None

    def _diff(self, frame) -> dict:
        fid = id(frame)
        old = self._snap.get(fid, {})
        new, changed = {}, {}
        for k, v in frame.f_locals.items():
            kk = self._key(v)
            if kk is None:
                continue
            new[k] = kk
            if old.get(k) != kk:
                changed[k] = v.detach().clone()
        self._snap[fid] = new
        return changed

    def _local(self, frame, event, arg):
        fid = id(frame)
        if event in ("line", "return"):
            ch = self._diff(frame)
            if fid in self._last and ch:
                self._last[fid].produced.update(ch)
            if event == "return":
                if fid in self._last:
                    self._last[fid].ret = arg
                self._snap.pop(fid, None); self._last.pop(fid, None)
                self._depth -= 1
                return self._local
            co = frame.f_code
            meta = {n: frame.f_locals[n] for n in self.meta_names if n in frame.f_locals and isinstance(frame.f_locals[n], (int, str))}
            ev = Event(len(self.events), os.path.abspath(co.co_filename), frame.f_lineno, co.co_name,
                       linecache.getline(co.co_filename, frame.f_lineno).rstrip("\n"), self._depth, meta)
            self.events.append(ev)
            self._last[fid] = ev
        return self._local

    def _global(self, frame, event, arg):
        if event == "call" and self._want(frame.f_code.co_filename):
            self._depth += 1
            co = frame.f_code
            args = {k: v.detach().clone() for k, v in frame.f_locals.items() if isinstance(v, torch.Tensor)}
            meta = {"call": True, **{k: v for k, v in frame.f_locals.items() if k != "self" and isinstance(v, (int, str, float))}}
            self.events.append(Event(len(self.events), os.path.abspath(co.co_filename), co.co_firstlineno, co.co_name,
                                     linecache.getline(co.co_filename, co.co_firstlineno).rstrip("\n"), self._depth, meta, args))
            self._snap[id(frame)] = {k: self._key(v) for k, v in frame.f_locals.items() if isinstance(v, torch.Tensor)}
            return self._local
        return None

    def __enter__(self):
        sys.settrace(self._global)
        return self

    def __exit__(self, *a):
        sys.settrace(None)
