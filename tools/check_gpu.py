# -*- coding: utf-8 -*-
"""Z-Image-Turbo 在 RTX 5060 Ti 16GB 上的 GPU 可行性验证。

分两步：
  1) 静态核算：权重体积 vs 显存容量，判断全量驻留是否可行
  2) 实测（权重齐备时）：加载 + 生成 1 张，打印 GPU 峰值显存/耗时，
     用 torch.cuda.memory_allocated 证明确实在 GPU 上执行，而不是 CPU 回退。

    python tools/check_gpu.py            # 静态 + 实测
    python tools/check_gpu.py --no-run   # 只做静态核算
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from story_agent.backends import ZImageBackend  # noqa: E402
from story_agent import console  # noqa: E402


def line(t):
    print("\n" + "=" * 74 + "\n  " + t + "\n" + "=" * 74, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    ap.add_argument("--mode", default="offload",
                    choices=["offload", "sequential", "manual", "full_gpu", "cpu"])
    ap.add_argument("--size", default="1024x576", help="生成分辨率，如 768x432 / 512x288")
    ap.add_argument("--sequential", action="store_true", help="用更省内存的 sequential offload")
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.lower().split("x"))
    config.GEN["width"], config.GEN["height"] = w, h

    line("STEP 1  硬件与运行时")
    import torch
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    vram = props.total_memory / 1e9 if props else 0
    print("  torch          :", torch.__version__, "CUDA", torch.version.cuda)
    print("  GPU            :", props.name if props else "无")
    print("  显存           : %.2f GB" % vram)
    print("  算力           : sm_%d%d %s" %
          (*torch.cuda.get_device_capability(0), "(Blackwell, 需 CUDA>=12.8)") if props else "N/A")
    print("  CUDA 可用      :", torch.cuda.is_available())

    line("STEP 2  Z-Image-Turbo 权重体积核算")
    files = sorted(config.MODEL_DIR.rglob("*.safetensors"))
    total = sum(p.stat().st_size for p in files) / 1e9
    print("  权重目录       :", config.MODEL_DIR)
    print("  .safetensors  : %d 个 / %.2f GB" % (len(files), total))
    for p in files:
        print("     %-62s %7.2f GB" % (p.relative_to(config.MODEL_DIR).as_posix(),
                                      p.stat().st_size / 1e9))
    from story_agent.backends import _WEIGHT_FILES
    complete = all((config.MODEL_DIR / f).exists() for f in _WEIGHT_FILES)
    converted = complete and total < 25.0    # BF16 预转换目录约 16.4 GB
    disk = total if complete else 33.0
    # F32 官方权重加载时转 bf16 显存减半；BF16 预转换目录本身就是最终精度
    bf16 = total if converted else disk / 2
    need = bf16 * 1.2        # 激活/临时缓冲经验系数
    if not complete:
        print("  (权重尚未下完，以下按官方完整 F32 33 GB 估算)")
    else:
        print("  权重类型        : %s" %
              ("BF16 预转换（磁盘/内存占用已减半）" if converted else "F32 官方"))
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
    except Exception:
        ram = 0
    print("\n  磁盘权重          : %.2f GB %s" %
          (disk, "" if complete else "(估算)"))
    print("  bf16 模型显存     : %.2f GB %s" %
          (bf16, "(磁盘已是 bf16)" if converted else "(= F32 / 2)"))
    print("  含激活缓冲估算    : ≈ %.2f GB" % need)
    print("  本机可用显存      : %.2f GB  |  物理内存 %.1f GB" % (vram, ram))
    verdict_full = "可行" if need < vram * 0.92 else "不可行（会 OOM）"
    print("  -> 全量 GPU 驻留 : %s" % verdict_full)
    print("  -> CPU offload   : DiT 约 %.1f GB 在 GPU，文本编码器约 %.1f GB 留内存"
          % (bf16 * 0.75, bf16 * 0.25))

    ok, why = ZImageBackend().available()
    print("\n  权重完整性      :", ("OK - " if ok else "NOT READY - ") + why)

    if a.no_run or not ok:
        line("结论（静态）")
        print("  Z-Image-Turbo = 6B DiT + Qwen3 文本编码器。仓库以 F32 存储共约 33 GB，")
        print("  以 bfloat16 加载后模型约 16.5 GB，仍超过 16GB 显卡的可用显存。")
        print("  官网所说「16G 能跑」指 enable_model_cpu_offload()：DiT 在 GPU 上推理，")
        print("  文本编码器/VAE 留内存按需换入，而不是全量 GPU 驻留。")
        print("  注意：本机物理内存也只有 %.1f GB，必须用 low_cpu_mem_usage=True" % ram)
        print("  （逐层加载 + safetensors mmap），否则 F32 权重会先把内存打爆。")
        print("  若要全量 GPU 驻留，需 FP8 / GGUF 量化版（约 8-10 GB）。")
        if not ok:
            print("\n  权重尚未下载完整，未执行实测。下载完成后重跑本脚本即可实测。")
        return 0

    line("STEP 3  实测加载 + 生成 1 张")
    z = ZImageBackend(mode="sequential" if a.sequential else a.mode)
    z.load()
    torch.cuda.reset_peak_memory_stats()
    try:
        import psutil
        vm = psutil.virtual_memory()
        print("  推理前: 显存 %.2f GB 已分配 / 内存可用 %.2f GB" %
              (torch.cuda.memory_allocated() / 1e9, vm.available / 1e9))
    except Exception:
        pass
    prompt = ("cinematic historical epic film still, Qin dynasty China 209 BC, "
              "a Chinese peasant leader raising a wooden staff on a mound at dawn, "
              "crowd of conscripts behind him, dramatic chiaroscuro, film grain, "
              "russet and steel-blue tones, 2.39:1 widescreen")
    print("  prompt:", prompt[:100] + " ...")
    t0 = time.time()
    img, info = z.generate(prompt, 2090209, type("S", (), {
        "shot_id": 0, "title": "GPU 验证", "characters": [],
        "shot_size": "全景", "camera": "", "lighting": "", "emotion": ""})())
    out = ROOT / "output" / "gpu_probe.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print("\n  生成耗时       : %.1f s" % info["seconds"])
    print("  GPU 峰值显存   : %.2f GB / %.2f GB" % (info["peak_vram_gb"], vram))
    print("  运行模式       :", info["mode"])
    print("  出图           :", out, img.size)
    print("  生成结束时显存占用: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

    line("结论（实测）")
    used_gpu = info["mode"] != "cpu" and info["peak_vram_gb"] > 0.05
    if used_gpu:
        print("  [通过] Z-Image-Turbo 确实在本机 GPU 上执行推理（峰值显存 %.2f GB），"
              % info["peak_vram_gb"])
        print("         不是 CPU 回退。16GB 显存可用，方式为：%s。" % info["mode"])
        if info["mode"] == "sequential_offload_gpu":
            print("         sequential 逐子模块换入 GPU，峰值显存天然很低，属正常现象。")
        print("         单张 %dx%d / %d steps 耗时 %.1fs，8 镜预计 %.1f 分钟。" %
              (config.GEN["width"], config.GEN["height"], config.GEN["steps"],
               info["seconds"], info["seconds"] * 8 / 60))
    else:
        print("  [未通过] 未观察到有效 GPU 显存占用，实际走了 CPU 路径。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
