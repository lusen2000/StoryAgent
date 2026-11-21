# -*- coding: utf-8 -*-
"""分块并行续传下载器：ModelScope CDN 对单连接限速约 3 MB/s，
对同一文件开多个 Range 连接可叠加带宽。支持从已有 .part 前缀继续。

    python tools/fetch_model_fast.py
"""
from __future__ import annotations

import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "Z-Image-Turbo"
BASE = "https://www.modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo/resolve/master"

BIG = [
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]
CHUNK = 256 * 1024 * 1024      # 单块 256MB
WORKERS = 8
_lock = threading.Lock()


def log(m):
    with _lock:
        print("[FAST] " + m, flush=True)


def url_of(rel):
    return BASE + "/" + rel


def probe_total(url):
    r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(20, 60))
    cr = r.headers.get("Content-Range", "")
    r.close()
    if "/" not in cr:
        raise RuntimeError("CDN 不支持 Range: " + cr)
    return int(cr.rsplit("/", 1)[1])


def get_chunk(url, start, end, path, retries=4):
    want = end - start + 1
    for i in range(1, retries + 1):
        try:
            with requests.get(url, headers={"Range": "bytes=%d-%d" % (start, end)},
                              stream=True, timeout=(20, 300)) as r:
                if r.status_code != 206:
                    raise IOError("期望 206，得到 %s" % r.status_code)
                with open(path, "wb") as f:
                    for c in r.iter_content(1 << 20):
                        if c:
                            f.write(c)
            got = path.stat().st_size
            if got != want:
                raise IOError("短读 %d/%d" % (got, want))
            return want
        except Exception as e:
            log("  retry %s %d/%d: %s" % (path.name, i, retries, e))
            time.sleep(2 * i)
    raise RuntimeError("分块失败 " + str(path))


def finish_file(rel):
    url = url_of(rel)
    out = DEST / rel
    part = out.with_suffix(out.suffix + ".part")
    total = probe_total(url)
    if out.exists() and out.stat().st_size == total:
        log("SKIP %-58s %.2f GB" % (rel, total / 1e9))
        return total
    have = part.stat().st_size if part.exists() else 0
    if have >= total:
        part.replace(out)
        return total
    if have:
        log("续传 %-58s 已有 %.2f / %.2f GB" % (rel, have / 1e9, total / 1e9))
    ranges = []
    s = have
    while s < total:
        ranges.append((s, min(s + CHUNK - 1, total - 1)))
        s += CHUNK
    log("START %-58s %.2f GB 待下, %d 块" %
        (rel, (total - have) / 1e9, len(ranges)))
    tmp = DEST / ".chunks"
    tmp.mkdir(exist_ok=True)
    t0 = time.time()
    jobs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (a, b) in enumerate(ranges):
            p = tmp / ("%s.chunk%03d" % (out.name, i))
            jobs.append((i, p, ex.submit(get_chunk, url, a, b, p)))
        for i, p, f in jobs:
            f.result()
    with open(part, "ab") as f:
        for i, p, _ in jobs:
            with open(p, "rb") as c:
                while True:
                    buf = c.read(1 << 22)
                    if not buf:
                        break
                    f.write(buf)
            p.unlink()
    got = part.stat().st_size
    if got != total:
        raise RuntimeError("%s 大小不符 %d/%d" % (rel, got, total))
    part.replace(out)
    dt = max(time.time() - t0, .01)
    log("DONE  %-58s %.2f GB  %.1fs  %.1f MB/s" %
        (rel, total / 1e9, dt, (total - have) / 1e6 / dt))
    return total


def main():
    t0 = time.time()
    total_all = 0
    for rel in BIG:
        try:
            total_all += finish_file(rel)
        except Exception as e:
            log("FAILED %s: %s" % (rel, e))
            return 1
    log("全部就绪 %.2f GB，用时 %.1f 分钟" % (total_all / 1e9, (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
