# -*- coding: utf-8 -*-
"""控制台输出：分节横幅、KV 表、指标表。同时把日志写到 run.log。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_USE_COLOR = True
_C = {
    "head": "\033[1;36m", "ok": "\033[1;32m", "warn": "\033[1;33m",
    "err": "\033[1;31m", "dim": "\033[2;37m", "hi": "\033[1;37m",
    "agent": "\033[1;35m", "end": "\033[0m",
}

_LOG_PATH: Path | None = None
_T0 = time.time()


def set_log(path: Path) -> None:
    global _LOG_PATH
    _LOG_PATH = path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _emit(line: str, color: str = "") -> None:
    plain = line
    if _USE_COLOR and color:
        line = _C[color] + line + _C["end"]
    print(line, flush=True)
    if _LOG_PATH:
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(plain + "\n")


def banner(title: str, sub: str = "") -> None:
    _emit("")
    _emit("=" * 78, "head")
    _emit("  " + title, "hi")
    if sub:
        _emit("  " + sub, "dim")
    _emit("=" * 78, "head")


def stage(n: int, name: str, desc: str = "") -> None:
    _emit("")
    _emit("+" + "-" * 77, "head")
    _emit("| STAGE %d  %s" % (n, name), "head")
    if desc:
        _emit("| " + desc, "dim")
    _emit("+" + "-" * 77, "head")


def agent(name: str, msg: str) -> None:
    _emit("  [%s] %s" % (name, msg), "agent")


def info(msg: str) -> None:
    _emit("      " + msg)


def ok(msg: str) -> None:
    _emit("      [OK] " + msg, "ok")


def warn(msg: str) -> None:
    _emit("      [WARN] " + msg, "warn")


def err(msg: str) -> None:
    _emit("      [ERR] " + msg, "err")


def kv(key: str, value, width: int = 26) -> None:
    _emit("      %-*s : %s" % (width, key, value))


def table(headers, rows, widths=None) -> None:
    if widths is None:
        widths = []
        for i, h in enumerate(headers):
            w = len(str(h))
            for r in rows:
                w = max(w, len(str(r[i])))
            widths.append(min(w + 2, 46))
    line = "      " + "".join(str(h).ljust(w) for h, w in zip(headers, widths))
    _emit(line, "hi")
    _emit("      " + "-" * sum(widths), "dim")
    for r in rows:
        _emit("      " + "".join(str(c).ljust(w) for c, w in zip(r, widths)))


def elapsed() -> str:
    return "%.1fs" % (time.time() - _T0)
