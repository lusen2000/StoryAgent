# -*- coding: utf-8 -*-
"""Z-Image-Turbo 权重下载（ModelScope CDN，多线程 + 断点重下 + 进度打印）"""
from __future__ import annotations

import sys
import time
import threading
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
SMALL = [
    "model_index.json", "configuration.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/generation_config.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/merges.txt", "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
]
_lock = threading.Lock()
_done = {}


def log(m):
    with _lock:
        print("[FETCH] " + m, flush=True)


def fetch(rel, retries=5):
    url = BASE + "/" + rel
    out = DEST / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    err = None
    for i in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                if out.exists() and total and out.stat().st_size == total:
                    log("SKIP  %-62s %8.1f MB" % (rel, total / 1e6))
                    return rel, total
                if out.exists() and not total:
                    return rel, out.stat().st_size
                t0 = time.time()
                got = 0
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)
                            got += len(chunk)
                if total and got < total:
                    raise IOError("短读 %d/%d" % (got, total))
                part.replace(out)
                dt = max(time.time() - t0, 0.01)
                log("DONE  %-62s %8.1f MB  %6.1fs  %5.1f MB/s"
                    % (rel, got / 1e6, dt, got / 1e6 / dt))
                return rel, got
        except Exception as e:
            err = e
            log("RETRY %-62s %d/%d  %s: %s" % (rel, i, retries, type(e).__name__, e))
            time.sleep(2 * i)
    raise RuntimeError("%s -> %s" % (rel, err))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    log("目标: %s" % DEST)
    for rel in SMALL:
        try:
            fetch(rel, retries=3)
        except Exception as e:
            log("小文件失败 %s: %s" % (rel, e))
    log("启动 7 个大权重并行下载 ...")
    t0 = time.time()
    ok, fail = 0, []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch, r): r for r in BIG}
        for f in as_completed(futs):
            try:
                f.result()
                ok += 1
            except Exception as e:
                fail.append(futs[f])
                log("FAILED %s: %s" % (futs[f], e))
    log("完成 %d/%d, 耗时 %.1f 分钟" % (ok, len(BIG), (time.time() - t0) / 60))
    if fail:
        log("缺失: " + ", ".join(fail))
        return 1
    sz = sum(p.stat().st_size for p in DEST.rglob("*.safetensors")) / 1e9
    log("权重总量 %.2f GB，就绪" % sz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
