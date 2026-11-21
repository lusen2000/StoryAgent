# -*- coding: utf-8 -*-
"""T2I 后端层：Z-Image-Turbo(GPU) / 本地程序化渲染(降级) / 云侧接口(可选)。"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import config
from story_agent import console

_REQUIRED = [
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "vae/config.json",
    "tokenizer/tokenizer.json",
]
_WEIGHT_FILES = [
    "text_encoder/model-00001-of-00003.safetensors",
    "text_encoder/model-00002-of-00003.safetensors",
    "text_encoder/model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]


class BaseBackend:
    name = "base"

    def available(self) -> Tuple[bool, str]:
        raise NotImplementedError

    def load(self) -> None:
        pass

    def generate(self, prompt: str, seed: int, shot) -> Tuple[Image.Image, Dict]:
        raise NotImplementedError


class ZImageBackend(BaseBackend):
    """官方 Tongyi-MAI/Z-Image-Turbo，diffusers ZImagePipeline。

    16GB 显存策略：
      - offload (默认): enable_model_cpu_offload()，GPU 峰值约 8-12GB，可跑
      - full_gpu       : pipe.to('cuda')，全量 bf16 约 20GB，16GB 卡会 OOM
      - cpu            : 纯 CPU 推理（仅作可行性对照，极慢）
    """

    name = "zimage"

    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.pipe = None
        self.torch = None
        self.info: Dict = {}

    def available(self) -> Tuple[bool, str]:
        paths = _REQUIRED + _WEIGHT_FILES
        missing = [f for f in paths if not (config.MODEL_DIR / f).exists()]
        total = sum((config.MODEL_DIR / f).stat().st_size
                    for f in _WEIGHT_FILES
                    if (config.MODEL_DIR / f).exists()) / 1e9
        if missing:
            return False, "权重不完整，缺 %s（已落盘 %.2f GB）" % (
                ", ".join(missing[:3]), total)
        if total < 10:
            return False, "权重仅 %.2f GB，疑似损坏" % total
        kind = "BF16 预转换" if total < 25 else "F32 官方"
        # 强制 --backend zimage 时也要拦截 F32 + 小内存的组合。
        # 这不是显存不够，而是系统内存装不下 24.6GB DiT 分片；
        # offload 首步去噪会在 native 层 CUDA OOM/SIGSEGV，Python 抓不到。
        if total > 28.0:
            try:
                import psutil
                ram = psutil.virtual_memory().total / 1e9
                if ram < 24.0:
                    return False, (
                        "F32 权重 %.1f GB + 系统内存 %.1f GB < 24 GB："
                        "model_cpu_offload 首步去噪会 CUDA OOM/SIGSEGV。"
                        "请先运行 tools/convert_bf16.py，设置 SB_MODEL_DIR 指向 "
                        "models/Z-Image-Turbo-bf16，并使用 --mode sequential"
                        % (total, ram))
            except Exception:
                pass
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA 不可用"
            from diffusers import ZImagePipeline  # noqa: F401
        except Exception as e:
            return False, "依赖不可用: %s" % e
        return True, "%s权重 %.2f GB / CUDA OK / %s" % (
            kind, total, config.MODEL_DIR.name)

    def load(self) -> None:
        import torch
        from diffusers import ZImagePipeline
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = True

        low = config.GEN.get("low_cpu_mem_usage", True)
        console.info("from_pretrained(%s)  dtype=%s  low_cpu_mem_usage=%s"
                     % (config.MODEL_DIR, config.GEN["dtype"], low))
        try:
            import psutil
            vm = psutil.virtual_memory()
            disk = sum(p.stat().st_size for p in
                       config.MODEL_DIR.rglob("*.safetensors")) / 1e9
            console.info("系统内存 %.1f GB / 可用 %.1f GB（本地权重 %.1f GB，逐层加载）"
                         % (vm.total / 1e9, vm.available / 1e9, disk))
        except Exception:
            pass
        t0 = time.time()
        try:
            self.pipe = ZImagePipeline.from_pretrained(
                str(config.MODEL_DIR),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=low,
            )
        except Exception as e:
            if not low:
                raise
            console.warn("逐层加载失败 (%s)，改用官方示例的 low_cpu_mem_usage=False"
                         % str(e)[:110])
            self.pipe = ZImagePipeline.from_pretrained(
                str(config.MODEL_DIR),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            )
        console.ok("模型加载完成 %.1fs" % (time.time() - t0))

        mode = self.mode
        if mode == "auto":
            mode = "offload" if config.GEN["offload"] else "full_gpu"
        self.info["requested_mode"] = mode

        if mode == "cpu":
            self.pipe = self.pipe.to("cpu")
            self.info["mode"] = "cpu"
        elif mode == "full_gpu":
            try:
                self.pipe = self.pipe.to("cuda")
                self.info["mode"] = "full_gpu"
                self.info["vram_after_load_gb"] = round(
                    torch.cuda.memory_allocated() / 1e9, 2)
                console.ok("全量驻留 GPU 成功，占用 %.2f GB"
                           % self.info["vram_after_load_gb"])
            except torch.cuda.OutOfMemoryError as e:
                console.warn("全量 GPU OOM（%s），自动切换 CPU offload" % str(e)[:90])
                torch.cuda.empty_cache()
                mode = "offload"
        if mode == "offload":
            self.pipe.enable_model_cpu_offload()
            self.info["mode"] = "cpu_offload_gpu"
            console.ok("启用 CPU offload：DiT 在 GPU 上推理，其余模块按需换入")
        elif mode == "sequential":
            self.pipe.enable_sequential_cpu_offload()
            self.info["mode"] = "sequential_offload_gpu"
            console.ok("启用 sequential offload：逐子模块换入 GPU，内存/显存占用最低")
        elif mode == "manual":
            self._manual_setup()
            self.info["mode"] = "manual_gpu"

        self.info["gpu"] = torch.cuda.get_device_name(0)
        self.info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        self.info["cuda"] = torch.version.cuda
        self.info["torch"] = torch.__version__

    def _manual_setup(self) -> None:
        """绕过 accelerate offload hook：text_encoder 留 CPU，DiT + VAE 放 GPU。

        Windows + 16GB 内存下 enable_model_cpu_offload 在首步前向会段错误，
        手动分配设备可以避免 hook 反复换入换出，显存峰值也更可控。
        """
        import torch
        from diffusers import DiffusionPipeline

        DiffusionPipeline._execution_device = property(
            lambda self: torch.device("cuda"))
        self.pipe.text_encoder.to("cpu")
        self.pipe.transformer.to("cuda")
        self.pipe.vae.to("cuda")
        console.ok("手动设备分配: text_encoder@CPU, transformer+vae@GPU")
        console.info("常驻显存 %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

    def generate(self, prompt: str, seed: int, shot) -> Tuple[Image.Image, Dict]:
        import torch
        if self.pipe is None:
            self.load()
        g = config.GEN
        # breakpoint()
        # CPU generator 在 CPU-offload 模式下更安全：diffusers 的 randn_tensor
        # 会把噪声搬到当前模块所在 device，不会产生 generator/张量设备冲突。
        gen = torch.Generator(device="cpu").manual_seed(seed)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        call = dict(
            height=g["height"],
            width=g["width"],
            num_inference_steps=g["steps"],
            guidance_scale=g["guidance"],
            generator=gen,
        )
        if self.info.get("mode") == "manual_gpu":
            # 预计算文本向量（CPU），再把 DiT 需要的所有输入放到 GPU
            embeds_cpu = self.pipe._encode_prompt(
                prompt, device=torch.device("cpu"))
            embeds = [e.to("cuda", dtype=torch.bfloat16) for e in embeds_cpu]
            call.update(prompt=None, prompt_embeds=embeds)
            console.info("prompt embedding 已在 CPU 预计算: %d 段 -> GPU" % len(embeds))
        else:
            call["prompt"] = prompt
        out = self.pipe(**call)
        dt = time.time() - t0
        img = out.images[0].convert("RGB")
        info = {
            "backend": "Z-Image-Turbo",
            "mode": self.info.get("mode"),
            "seconds": round(dt, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "steps": g["steps"],
            "size": "%dx%d" % img.size,
            "seed": seed,
        }
        return img, info


# ---------------------------------------------------------------------------
# 降级后端：纯 PIL 程序化电影分镜渲染（保证三大流程始终可完整跑通）
# ---------------------------------------------------------------------------
_PALETTES = [
    ((38, 45, 58), (196, 138, 78)),      # 1 鸿鹄之志 暖金
    ((28, 38, 52), (96, 118, 138)),      # 2 大雨误期 冷蓝
    ((26, 22, 24), (214, 128, 54)),      # 3 营帐密谋 烛火
    ((32, 30, 42), (168, 96, 62)),       # 4 鱼腹藏书 幽紫褐
    ((18, 26, 34), (176, 112, 48)),      # 5 篝火狐鸣 暗夜橙
    ((46, 24, 24), (198, 74, 52)),       # 6 怒杀校尉 血红
    ((52, 44, 40), (214, 168, 92)),      # 7 王侯将相 破晓金
    ((44, 52, 62), (206, 142, 82)),      # 8 揭竿而起 晨光
]


def _font(size: int, bold: bool = False):
    p = config.FONTS["bold"] if bold else config.FONTS["regular"]
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()


class MockBackend(BaseBackend):
    """不依赖任何深度学习权重，绘制可阅读的电影分镜概念图（降级 / 冒烟测试用）。"""

    name = "local-render"

    def __init__(self):
        self.info = {}

    def available(self) -> Tuple[bool, str]:
        return True, "纯 PIL，无外部依赖"

    def generate(self, prompt: str, seed: int, shot) -> Tuple[Image.Image, Dict]:
        w, h = config.GEN["width"], config.GEN["height"]
        rng = np.random.default_rng(seed)
        idx = (shot.shot_id - 1) % len(_PALETTES)
        top, glow = _PALETTES[idx]

        # 天空渐变
        yy = np.linspace(0, 1, h)[:, None, None]
        base = np.array(top, dtype=np.float32)[None, None, :]
        horizon = np.clip(np.array(glow, dtype=np.float32) * 0.45
                          + np.array(top, dtype=np.float32) * 0.55, 0, 255)
        arr = base * (1 - yy ** 1.5) + horizon[None, None, :] * (yy ** 1.5)
        arr = np.repeat(arr, w, axis=1)

        # 主光源辉光
        lx = w * (0.28 + 0.44 * ((shot.shot_id * 7 % 10) / 10.0))
        ly = h * (0.32 + 0.12 * ((shot.shot_id * 3 % 5) / 5.0))
        xs = np.arange(w)[None, :]
        ys = np.arange(h)[:, None]
        d = np.sqrt(((xs - lx) / (w * 0.55)) ** 2 + ((ys - ly) / (h * 0.75)) ** 2)
        light = np.exp(-d ** 2 * 3.2)[:, :, None] * np.array(glow, np.float32)[None, None, :] * 0.55
        arr = np.clip(arr + light, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(img, "RGBA")

        # 远山 / 城墙剪影
        ridge = h * 0.66
        pts = [(0, h)]
        for x in range(0, w + 40, 40):
            pts.append((x, ridge - 26 * math.sin(x / 190.0 + idx)
                        - 14 * math.sin(x / 71.0 + idx * 2)))
        pts += [(w, h)]
        draw.polygon(pts, fill=(int(top[0] * .55), int(top[1] * .58), int(top[2] * .62), 255))
        draw.rectangle([0, int(h * .74), w, h],
                       fill=(int(top[0] * .42), int(top[1] * .44), int(top[2] * .48), 255))

        # 雨丝
        if any(k in shot.title + shot.emotion + shot.lighting for k in ("雨", "冷", "绝望")):
            for _ in range(240):
                x, y = rng.integers(0, w), rng.integers(0, int(h * .8))
                draw.line([(x, y), (x - 8, y + 26)], fill=(190, 205, 220, 42), width=1)

        # 人物剪影（数量按景别）
        n = {"特写": 1, "近景": 2, "中景": 3, "全景": 7, "远景": 11}.get(shot.shot_size, 4)
        n = min(n, max(1, len(shot.characters) + 4))
        for i in range(n):
            if n == 1:
                cx, scale = w * .38, 1.7
            else:
                cx = w * (.12 + .76 * i / max(n - 1, 1)) + rng.normal(0, 8)
                scale = 0.85 + 0.35 * rng.random()
            if "陈胜" in shot.characters and i == n // 2:
                scale *= 1.35
            hh = 78 * scale
            cy = h * .72
            col = (16, 15, 18, 235)
            draw.ellipse([cx - 11 * scale, cy - hh, cx + 11 * scale, cy - hh + 22 * scale], fill=col)
            draw.polygon([(cx - 17 * scale, cy - hh + 20 * scale),
                          (cx + 17 * scale, cy - hh + 20 * scale),
                          (cx + 13 * scale, cy), (cx - 13 * scale, cy)], fill=col)
            arm_up = "王侯" in shot.title or "揭竿" in shot.title or "怒杀" in shot.title
            if arm_up and (i % 2 == 0):
                draw.line([(cx + 12 * scale, cy - hh + 30 * scale),
                           (cx + 30 * scale, cy - hh - 18 * scale)], fill=col, width=int(7 * scale))
                draw.line([(cx + 30 * scale, cy - hh - 18 * scale),
                           (cx + 34 * scale, cy - hh - 58 * scale)], fill=(70, 52, 38, 255),
                          width=int(4 * scale))
            else:
                draw.line([(cx - 12 * scale, cy - hh + 30 * scale),
                           (cx - 20 * scale, cy - 8 * scale)], fill=col, width=int(6 * scale))

        # 旗帜 / 火把
        if "揭竿" in shot.title or "王侯" in shot.title:
            for bx in (w * .2, w * .62):
                draw.line([(bx, h * .72), (bx, h * .38)], fill=(60, 44, 32, 255), width=5)
                draw.polygon([(bx, h * .38), (bx + 58, h * .40), (bx + 46, h * .47), (bx, h * .45)],
                             fill=(150, 44, 40, 220))
        if "篝火" in shot.title or "密谋" in shot.title or "狐鸣" in shot.title:
            fx, fy = w * .76, h * .68
            for r, a in ((70, 28), (44, 55), (22, 90)):
                draw.ellipse([fx - r, fy - r, fx + r, fy + r], fill=(230, 140, 50, a))
            draw.polygon([(fx - 16, fy + 12), (fx, fy - 34), (fx + 16, fy + 12)],
                         fill=(250, 200, 90, 220))

        # 雾 / 暗角 / 颗粒
        fog = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        fd = ImageDraw.Draw(fog)
        for j in range(5):
            fd.rectangle([0, h * (.58 + j * .055), w, h * (.60 + j * .055)],
                         fill=(200, 205, 210, 10))
        img = Image.alpha_composite(img.convert("RGBA"), fog.filter(ImageFilter.GaussianBlur(14)))
        vig = np.zeros((h, w), np.float32)
        cxv, cyv = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        vig = np.clip(1 - .55 * (cxv ** 2 + cyv ** 2) ** 1.3, .35, 1)
        a = np.asarray(img).astype(np.float32)
        a[:, :, :3] *= vig[:, :, None]
        grain = rng.normal(0, 3.2, (h, w, 1))
        img = Image.fromarray(np.clip(a[:, :, :3] + grain, 0, 255).astype(np.uint8), "RGB")

        # 分镜信息条
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, 0, w, 34], fill=(10, 12, 16, 190))
        d.text((16, 8), "STORYBOARD AGENT  /  SHOT %02d  /  LOCAL-RENDER FALLBACK"
               % shot.shot_id, font=_font(15, True), fill=(235, 230, 220, 255))
        d.rectangle([0, h - 76, w, h], fill=(10, 12, 16, 205))
        d.text((18, h - 70), "%02d  %s" % (shot.shot_id, shot.title),
               font=_font(23, True), fill=(245, 226, 190, 255))
        d.text((18, h - 38), "%s  |  %s  |  %s  |  %s"
               % (shot.shot_size, shot.camera, shot.lighting[:38], shot.emotion),
               font=_font(14), fill=(186, 192, 198, 255))
        d.rectangle([0, 0, w - 1, h - 1], outline=(255, 255, 255, 55), width=2)

        return img, {"backend": "local-render (降级)", "mode": "cpu-pil",
                     "seconds": 0.0, "peak_vram_gb": 0.0,
                     "steps": 0, "size": "%dx%d" % (w, h), "seed": seed}


class CloudBackend(BaseBackend):
    """OpenAI 兼容 images/generations 云侧接口（需配置环境变量）。"""

    name = "cloud"

    def __init__(self):
        # 不在构造时导入 requests：rule + mock 的最小路径不需要它。
        self.url = os.getenv("SB_CLOUD_IMAGE_URL", "")
        self.key = os.getenv("SB_CLOUD_IMAGE_KEY", "")
        self.model = os.getenv("SB_CLOUD_IMAGE_MODEL", "z-image-turbo")

    def available(self) -> Tuple[bool, str]:
        return (bool(self.url and self.key),
                "云侧接口 %s" % (self.model if self.url else "未配置"))

    def generate(self, prompt: str, seed: int, shot) -> Tuple[Image.Image, Dict]:
        import requests
        t0 = time.time()
        r = requests.post(self.url,
                          headers={"Authorization": "Bearer " + self.key},
                          json={"model": self.model, "prompt": prompt,
                                "size": "1024x1024", "n": 1, "seed": seed},
                          timeout=180)
        r.raise_for_status()
        data = r.json()["data"][0]
        raw = base64.b64decode(data["b64_json"]) if data.get("b64_json") \
            else requests.get(data["url"], timeout=120).content
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img, {"backend": "cloud:" + self.model, "mode": "remote",
                     "seconds": round(time.time() - t0, 2),
                     "peak_vram_gb": 0.0, "steps": -1,
                     "size": "%dx%d" % img.size, "seed": seed}


def build_backend(prefer: str = "auto") -> BaseBackend:
    """auto: Z-Image GPU > 云侧 > 本地渲染；可强制指定。"""
    if prefer in ("zimage", "auto"):
        z = ZImageBackend()
        good, why = z.available()
        console.info("Z-Image 可用性检查: %s" % why)
        # 关键预检：官方权重是 F32 33GB，加载 DiT 分片时若物理内存不足，
        # safetensors/mmap 会触发 SIGSEGV（Python 无法捕获），必须提前拦截。
        if prefer == "auto":
            try:
                import psutil
                ram = psutil.virtual_memory().total / 1e9
                disk = sum(p.stat().st_size for p in
                           config.MODEL_DIR.rglob("*.safetensors")) / 1e9
                if disk > 28.0 and ram < 24.0:
                    good = False
                    why = ("物理内存 %.1f GB < 24 GB，加载 %.1f GB F32 权重会段错误"
                           % (ram, disk))
                    console.warn("本机内存预检未通过: %s" % why)
                    console.warn("这不是显卡不支持 —— GPU/CUDA 正常，瓶颈是系统内存")
            except Exception:
                pass
        if good:
            return z
        console.warn("跳过 Z-Image -> %s" % why)
        if prefer == "zimage":
            raise RuntimeError("强制使用 Z-Image 但不可用: " + why)
    if prefer in ("cloud", "auto"):
        c = CloudBackend()
        good, why = c.available()
        if good:
            console.ok("使用云侧接口: " + why)
            return c
        if prefer == "cloud":
            raise RuntimeError("云侧接口未配置")
        console.info("云侧接口未配置，跳过")
    console.warn("降级到本地程序化渲染（流程仍完整执行，但非神经网络出图）")
    return MockBackend()
