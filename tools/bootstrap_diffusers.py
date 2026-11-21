# -*- coding: utf-8 -*-
"""重建项目私有 vendor/ 里的 diffusers 0.36.0
（首个内置 ZImagePipeline 的版本；conda 环境里通常是更旧的版本）。

不改动任何 conda 环境，也不依赖 pip 解析依赖树，只下载一个 4.6 MB 的 wheel。

    python tools/bootstrap_diffusers.py
"""
from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
CACHE = ROOT / ".cache" / "wheels"
WHEEL = "diffusers-0.36.0-py3-none-any.whl"
SHA_PATH = "35/50/281f92cb1f83854dbd79b6e958b3bc5018607e2542971d41604ba7a14b2f"
MIRRORS = [
    "https://mirrors.huaweicloud.com/repository/pypi/packages/" + SHA_PATH + "/" + WHEEL,
    "https://mirrors.aliyun.com/pypi/packages/" + SHA_PATH + "/" + WHEEL,
    "https://files.pythonhosted.org/packages/" + SHA_PATH + "/" + WHEEL,
]
EXPECT = 4597884


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / WHEEL
    if not (dst.exists() and dst.stat().st_size == EXPECT):
        for url in MIRRORS:
            print("[BOOTSTRAP] 下载:", url.split("/")[2])
            try:
                subprocess.run(["curl", "-sL", "-m", "180", "-o", str(dst), url],
                               check=True, timeout=200)
            except Exception as e:
                print("[BOOTSTRAP] 失败:", e)
                continue
            if dst.exists() and dst.stat().st_size == EXPECT:
                break
            print("[BOOTSTRAP] 文件不完整，换源重试")
    if not (dst.exists() and dst.stat().st_size == EXPECT):
        print("[BOOTSTRAP] 所有镜像均失败")
        return 1
    print("[BOOTSTRAP] wheel %.2f MB" % (dst.stat().st_size / 1e6))
    VENDOR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(dst.read_bytes())) as z:
        z.extractall(VENDOR)
    print("[BOOTSTRAP] 解压到:", VENDOR)
    sys.path.insert(0, str(VENDOR))
    import diffusers
    from diffusers import ZImagePipeline
    print("[BOOTSTRAP] diffusers", diffusers.__version__, "ZImagePipeline OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
