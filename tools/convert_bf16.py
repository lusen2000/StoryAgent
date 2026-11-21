# -*- coding: utf-8 -*-
"""把 Z-Image-Turbo 的 F32 权重预转换为 BF16，写入独立目录。

动机：官方仓库是 F32 32.83GB，加载 24.6GB 的 DiT 分片时本机 16GB 内存
会 SIGSEGV。预先转 BF16 后磁盘/内存占用减半（约 16.4GB），且推理精度
与 diffusers 里 torch_dtype=bfloat16 完全一致。

转换用 safetensors 的 safe_open 逐张量流式读写，峰值内存 ≈ 单个分片
的 BF16 大小 + 最大张量，不会把整个 F32 分片读进内存。

    python tools/convert_bf16.py
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "Z-Image-Turbo"
DST = ROOT / "models" / "Z-Image-Turbo-bf16"


def convert_one(src: Path, dst: Path) -> tuple[int, int, int]:
    out = {}
    n_conv = 0
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            # 只把 float32 权重降为 bfloat16；整型/布尔/索引张量原样保留
            if t.is_floating_point() and t.dtype == torch.float32:
                t = t.to(torch.bfloat16)
                n_conv += 1
            out[k] = t.contiguous()
            del t
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst), metadata={"format": "pt"})
    del out
    return src.stat().st_size, dst.stat().st_size, n_conv


def fix_index_size(path: Path, actual: dict[str, int]) -> None:
    """更新分片索引里的 total_size 元数据（有些加载器会核对）。"""
    if not path.exists():
        return
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        meta = d.get("metadata", {})
        sizes = set()
        for w, fn in d.get("weight_map", {}).items():
            if fn in actual:
                sizes.add(actual[fn])
        if sizes:
            meta["total_size"] = sum(actual.values())
            d["metadata"] = meta
            path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print("[BF16] index 更新跳过:", e)


def main() -> int:
    if not SRC.exists():
        print("[BF16] 源目录不存在:", SRC)
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s_total = d_total = 0
    actual: dict[str, int] = {}

    weights = sorted(p for p in SRC.rglob("*.safetensors")
                     if ".chunks" not in p.parts)
    others = sorted(p for p in SRC.rglob("*")
                    if p.is_file() and p.suffix != ".safetensors"
                    and ".chunks" not in p.parts and p.suffix != ".part")
    print("[BF16] 源 %s -> 目标 %s" % (SRC, DST))
    print("[BF16] 权重文件 %d 个 / 其他文件 %d 个" % (len(weights), len(others)))

    for p in others:
        rel = p.relative_to(SRC)
        q = DST / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    print("[BF16] 配置/分词器/scheduler 已复制")

    for p in weights:
        rel = p.relative_to(SRC)
        q = DST / rel
        if q.exists() and q.stat().st_size > 0 and abs(q.stat().st_size - p.stat().st_size / 2) < p.stat().st_size * 0.15:
            print("[BF16] SKIP %-62s (已转换)" % rel.as_posix())
            s, d = p.stat().st_size, q.stat().st_size
        else:
            print("[BF16] 转换 %-62s %.2f GB ..." % (rel.as_posix(), p.stat().st_size / 1e9), flush=True)
            s, d, n = convert_one(p, q)
            print("[BF16]   -> %.2f GB  (%d 个张量转 bf16)" % (d / 1e9, n), flush=True)
        s_total += s
        d_total += d
        actual[rel.name] = d

    for idx in DST.rglob("*index.json"):
        fix_index_size(idx, actual)

    print("[BF16] 完成: %.2f GB -> %.2f GB (%.1f%%)，用时 %.1f 分钟"
          % (s_total / 1e9, d_total / 1e9, d_total / s_total * 100,
             (time.time() - t0) / 60))
    print("[BF16] 目录:", DST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
