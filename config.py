# -*- coding: utf-8 -*-
"""全局配置：路径、模型、生成参数、验收阈值。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 私有 vendor/ 里放 diffusers 0.36.0（首个内置 ZImagePipeline 的版本），
# 优先于 conda 环境的旧版 diffusers，且不改动用户环境。
_VENDOR = ROOT / "vendor"
if _VENDOR.exists() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# 默认官方 F32 权重；设置 SB_MODEL_DIR 可切到预转换的 BF16 目录
MODEL_DIR = Path(os.environ.get(
    "SB_MODEL_DIR", str(ROOT / "models" / "Z-Image-Turbo")))
OUTPUT_ROOT = ROOT / "output"
STORY_DIR = ROOT / "stories"
CACHE_DIR = ROOT / ".cache"

# 所有缓存写 E: 盘，不占 C: 系统盘
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hf" / "hub"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))

MODEL_REPO = "Tongyi-MAI/Z-Image-Turbo"

# ---- T2I 生成参数（RTX 5060 Ti 16GB）----
GEN = {
    "width": 1024,
    "height": 576,          # 16:9，比 1024x1024 省显存
    "steps": 9,             # Turbo: 9 步 = 8 次 DiT 前向
    "guidance": 0.0,        # Turbo 蒸馏模型必须 0.0
    "dtype": "bfloat16",
    "seed": 2090209,
    "offload": True,        # 16GB 显存用 CPU offload
    # 官方示例用 False，但本机物理内存仅 16GB 而 F32 权重有 33GB，
    # 必须 True（meta device 逐层加载 + safetensors mmap）才不会把内存打爆。
    "low_cpu_mem_usage": True,
}

# ---- 验收阈值 ----
THRESHOLDS = {
    "overall": 0.60,
    "clip": 0.40,
    "dino": 0.42,
    "aesthetic": 0.42,
    "vlm": 0.55,
    "max_repair": 2,        # 单镜最多重画次数
}

# ---- 可选外部服务（无则自动降级到规则/代理实现）----
VLM_ENDPOINT = os.getenv("SB_VLM_ENDPOINT", "")
VLM_API_KEY = os.getenv("SB_VLM_API_KEY", "")
VLM_MODEL = os.getenv("SB_VLM_MODEL", "qwen-vl-max")
CLOUD_IMAGE_API = os.getenv("SB_CLOUD_IMAGE_API", "")   # 云侧 T2I 兜底开关

FONTS = {
    "regular": r"C:\Windows\Fonts\msyh.ttc",
    "bold": r"C:\Windows\Fonts\msyhbd.ttc",
    "serif": r"C:\Windows\Fonts\simsun.ttc",
}

GLOBAL_STYLE = (
    "cinematic historical epic film still, Qin dynasty China 209 BC, "
    "dramatic chiaroscuro lighting, realistic oil-painting texture, "
    "wuxia color grading with russet and steel-blue tones, film grain, "
    "shallow depth of field, anamorphic lens, high contrast, epic scale, "
    "2.39:1 cinematic framing"
)
NEGATIVE = (
    "modern buildings, cars, plastic, neon sign, watermark, text overlay, "
    "logo, deformed hands, extra limbs, blurry, low quality, cartoon"
)
