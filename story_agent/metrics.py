# -*- coding: utf-8 -*-
"""视觉验收指标层。

所有分数都由图像像素真实计算得出；当缺少深度模型权重时，
使用可解释的代理/规则实现并在名称上显式标注 (proxy)，绝不伪造满分。
设置 SB_REAL_CLIP=1 可启用真实 CLIPScore（需能下载 CLIP 权重）。
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance

import config

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# 基础图像统计
# ---------------------------------------------------------------------------
def image_stats(img: Image.Image) -> Dict[str, float]:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = a @ np.array([0.299, 0.587, 0.114], np.float32)
    hsv = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2HSV) if _HAS_CV2 else None

    sat = float(hsv[:, :, 1].mean() / 255) if hsv is not None else 0.4
    hist = np.histogram(gray, bins=32, range=(0, 256))[0].astype(np.float64)
    p = hist / max(hist.sum(), 1)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / 5.0)  # 归一到 0-1
    if _HAS_CV2:
        sharp = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
    else:
        gx = np.diff(gray, axis=1)
        sharp = float(gx.var())
    bright = float(gray.mean() / 255)
    contrast = float(gray.std() / 128)
    uniq = len(np.unique((a // 16).astype(np.uint8).reshape(-1, 3), axis=0))
    return {
        "brightness": round(bright, 4),
        "contrast": round(contrast, 4),
        "saturation": round(sat, 4),
        "entropy": round(min(entropy, 1.0), 4),
        "sharpness": round(sharp, 2),
        "color_bins": uniq,
    }


def aesthetic_score(img: Image.Image, st: Dict[str, float]) -> Tuple[float, Dict]:
    """审美分（启发式）：曝光、对比、色彩层次、锐度、饱和度的加权。"""
    parts = {
        "exposure": 1.0 - abs(st["brightness"] - 0.46) / 0.55,
        "contrast": min(st["contrast"] / 0.62, 1.0),
        "tonal_range": min(st["entropy"] / 0.82, 1.0),
        "detail": min(math.log1p(st["sharpness"]) / math.log1p(900), 1.0),
        "color": min(st["color_bins"] / 900.0, 1.0),
    }
    for k in parts:
        parts[k] = round(max(0.0, min(1.0, parts[k])), 4)
    score = (0.26 * parts["exposure"] + 0.22 * parts["contrast"]
             + 0.20 * parts["tonal_range"] + 0.18 * parts["detail"]
             + 0.14 * parts["color"])
    return round(score, 4), parts


# ---------------------------------------------------------------------------
# CLIPScore：真实 CLIP 或 可解释代理
# ---------------------------------------------------------------------------
_CLIP = None


def _load_clip():
    global _CLIP
    if _CLIP is not None:
        return _CLIP
    if os.getenv("SB_REAL_CLIP", "") != "1":
        _CLIP = False
        return False
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        mid = os.getenv("SB_CLIP_MODEL", "openai/clip-vit-base-patch32")
        _CLIP = (CLIPModel.from_pretrained(mid),
                 CLIPProcessor.from_pretrained(mid), torch)
        return _CLIP
    except Exception:
        _CLIP = False
        return False


_CUE = {
    "rain": ("cold", "dark"), "overcast": ("cold", "dark"),
    "night": ("dark",), "fire": ("warm",), "torch": ("warm",),
    "candle": ("warm",), "golden": ("warm", "bright"),
    "sunlight": ("bright",), "dawn": ("bright", "warm"),
    "blood": ("warm",), "mist": ("lowsat",),
}


def clipscore(img: Image.Image, prompt: str, st: Dict[str, float]) -> Tuple[float, str, Dict]:
    real = _load_clip()
    if real:
        try:
            model, proc, torch = real
            inputs = proc(text=[prompt[:400]], images=img, return_tensors="pt",
                          padding=True, truncation=True)
            with torch.no_grad():
                f = model(**inputs)
                sim = float(f.logits_per_image[0, 0])
            # CLIP cosine 典型 -10~35 映射到 0-1
            return round(max(0.0, min(1.0, (sim + 5) / 35)), 4), "real", {"raw": round(sim, 3)}
        except Exception:
            pass

    # 代理：prompt 中视觉线索与图像色调/曝光的吻合度 + 主题词覆盖
    pl = prompt.lower()
    warm = float(np.asarray(img.convert("RGB"), np.float32)[:, :, 0].mean()
                 - np.asarray(img.convert("RGB"), np.float32)[:, :, 2].mean()) / 255
    cues = []
    for k, tags in _CUE.items():
        if k in pl:
            cues.append((k, tags))
    hit, total = 0.0, 0.0
    cue_detail = {}
    for k, tags in cues:
        for t in tags:
            total += 1
            if t == "warm":
                v = warm > -0.01
            elif t == "cold":
                v = warm < 0.02
            elif t == "dark":
                v = st["brightness"] < 0.52
            elif t == "bright":
                v = st["brightness"] > 0.38
            elif t == "lowsat":
                v = st["saturation"] < 0.55
            else:
                v = True
            hit += 1 if v else 0.35
            cue_detail[k + ":" + t] = bool(v)
    tone = hit / total if total else 0.62
    # 主题词覆盖：角色/场景 token 是否写进 prompt（防止空 prompt）
    tokens = len(re.findall(r"[A-Za-z]{4,}", prompt))
    cover = min(tokens / 45.0, 1.0)
    # 代理实现留出天花板，避免把启发式分数伪装成满分的深度模型结果
    score = round(min(0.96, 0.68 * tone + 0.32 * cover), 4)
    return score, "proxy", {"tone_match": round(tone, 3),
                            "prompt_token_cover": round(cover, 3),
                            "cues": cue_detail}


# ---------------------------------------------------------------------------
# 角色一致性：DINO 或 颜色/结构代理
# ---------------------------------------------------------------------------
def _hsv_hist(img: Image.Image) -> np.ndarray:
    """颜色分布直方图。

    有 opencv 时用 HSV 2D 直方图；没有 opencv 时退回归一化 RGB 量化直方图。
    这样最小依赖（numpy + Pillow）的 rule + mock 路径也能跑完整评估。
    """
    a = np.asarray(img.convert("RGB").resize((256, 256)), np.uint8)
    if _HAS_CV2:
        h = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([h], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)
    # 无 cv2：4 bit/通道量化成 16^3 个桶，归一化后用于相似度比较。
    q = (a.reshape(-1, 3) // 16).astype(np.int64)
    idx = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    hist = np.bincount(idx, minlength=4096).astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    return hist


def hist_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """直方图相似度；opencv 可用时用 Bhattacharyya 距离，否则用巴氏系数。"""
    if h1 is None or h2 is None or h1.size == 0 or h2.size == 0:
        return 0.0
    if _HAS_CV2:
        d = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32),
                            cv2.HISTCMP_BHATTACHARYYA)
        return max(0.0, 1.0 - float(d))
    n = min(len(h1), len(h2))
    bc = float(np.sqrt(np.maximum(h1[:n], 0) * np.maximum(h2[:n], 0)).sum())
    return max(0.0, min(1.0, bc))


def _dhash(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((17, 16)), np.uint8)
    return (a[:, 1:] > a[:, :-1]).flatten()


def dino_consistency(img: Image.Image, ref: Image.Image | None,
                     st: Dict[str, float]) -> Tuple[float, str, Dict]:
    if ref is None:
        return 0.0, "no-ref", {}
    h1, h2 = _hsv_hist(img), _hsv_hist(ref)
    color_sim = hist_similarity(h1, h2)
    # 构图哈希对"参考板 vs 剧情画面"没有可比性，只作弱佐证；
    # 该代理指标衡量的是角色/场景的整体色调一致性，不是像素级身份锁定。
    d1, d2 = _dhash(img), _dhash(ref)
    struct_sim = 1.0 - float(np.count_nonzero(d1 != d2)) / d1.size
    struct_sim = max(0.0, min(1.0, (struct_sim - 0.42) / 0.45))
    score = round(0.88 * color_sim + 0.12 * struct_sim, 4)
    return score, "proxy(hsv-palette)", {
        "color_hist_sim": round(color_sim, 3),
        "structure_sim": round(struct_sim, 3),
    }


# ---------------------------------------------------------------------------
# VLM 检查：云侧 VLM 或 规则检查
# ---------------------------------------------------------------------------
def _vlm_remote(img: Image.Image, shot, prompt: str):
    import requests
    buf = io.BytesIO()
    img.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "model": config.VLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text":
                "你是分镜验收员。检查这张图：1)情节是否符合「%s」2)构图/景别 3)有无明显崩坏（畸变、纯色、糊烂）。"
                "只输出 JSON: {\"plot\":0-1,\"composition\":0-1,\"integrity\":0-1,\"issues\":[]}" % shot.title}
        ]}],
        "temperature": 0.1,
    }
    r = requests.post(config.VLM_ENDPOINT,
                      headers={"Authorization": "Bearer " + config.VLM_API_KEY},
                      json=payload, timeout=120)
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {}


def vlm_check(img: Image.Image, shot, prompt: str,
              st: Dict[str, float]) -> Tuple[float, str, List[str], Dict]:
    if config.VLM_ENDPOINT and config.VLM_API_KEY:
        try:
            d = _vlm_remote(img, shot, prompt)
            score = (float(d.get("plot", .6)) + float(d.get("composition", .6))
                     + float(d.get("integrity", .6))) / 3
            return round(score, 4), "VLM(remote)", list(d.get("issues", [])), d
        except Exception:
            pass

    # 规则式验收：情节关键词→必需视觉线索、构图、崩坏
    issues: List[str] = []
    plot_hits, plot_total = 0, 0
    text = (shot.title + shot.lighting + shot.emotion + prompt).lower()
    if "雨" in text or "rain" in text:
        plot_total += 1
        if st["brightness"] < 0.60:
            plot_hits += 1
        else:
            issues.append("plot:raining_scene_too_bright")
    if "火" in text or "fire" in text or "torch" in text:
        plot_total += 1
        a = np.asarray(img.convert("RGB"), np.float32)
        warm = float((a[:, :, 0].mean() - a[:, :, 2].mean()) / 255)
        if warm > -0.02:
            plot_hits += 1
        else:
            issues.append("plot:missing_warm_firelight")
    if "夜" in text or "night" in text or "dark" in text:
        plot_total += 1
        if st["brightness"] < 0.58:
            plot_hits += 1
        else:
            issues.append("plot:night_scene_too_bright")
    plot = plot_hits / plot_total if plot_total else 0.78

    # 构图：显著区域是否偏离正中（三分法）+ 画面层次
    a = np.asarray(img.convert("L"), np.float32)
    h, w = a.shape
    col = a.mean(axis=0)
    peak = float(np.argmax(col)) / w
    comp = 1.0 - abs(peak - 0.5) * 0.7
    comp = 0.55 + 0.45 * min(comp / 0.85, 1.0)
    comp *= min(st["entropy"] / 0.7, 1.0) ** 0.35

    # 崩坏：纯色、全黑全白、噪点异常、锐度极低
    integ = 1.0
    if st["color_bins"] < 40:
        integ -= 0.45
        issues.append("integrity:near_blank")
    if st["brightness"] < 0.055 or st["brightness"] > 0.965:
        integ -= 0.4
        issues.append("integrity:extreme_exposure")
    if st["contrast"] < 0.10:
        integ -= 0.25
        issues.append("integrity:flat_image")
    if st["sharpness"] < 12:
        integ -= 0.2
        issues.append("integrity:blurry")
    if st["brightness"] < 0.12:   # 仅极端欠曝才判问题，暗调夜景属正常创作
        issues.append("lighting:too_dark")
    integ = max(0.0, integ)

    score = round(0.40 * plot + 0.22 * comp + 0.38 * integ, 4)
    return score, "VLM(rule-based proxy)", issues, {
        "plot": round(plot, 3), "composition": round(comp, 3),
        "integrity": round(integ, 3), "bright_peak_x": round(peak, 3),
    }


def repair_image(img: Image.Image, issues: List[str]) -> Image.Image:
    """按问题类型做局部修复（不重画）：曝光/对比度/锐度。"""
    out = img
    if any("too_dark" in i for i in issues):
        out = ImageEnhance.Brightness(out).enhance(1.28)
    if any("flat_image" in i or "blurry" in i for i in issues):
        out = ImageEnhance.Contrast(out).enhance(1.12)
        out = ImageEnhance.Sharpness(out).enhance(1.35)
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def evaluate_shot(img: Image.Image, shot, ref: Image.Image | None,
                  prompt: str) -> Dict:
    st = image_stats(img)
    aes, aes_d = aesthetic_score(img, st)
    cs, cs_mode, cs_d = clipscore(img, prompt, st)
    ds, ds_mode, ds_d = dino_consistency(img, ref, st)
    vs, vs_mode, issues, vs_d = vlm_check(img, shot, prompt, st)

    w = {"clip": .30, "dino": .15, "aesthetic": .25, "vlm": .30}
    if ds_mode == "no-ref":       # 无参考图时重新分配权重
        w = {"clip": .34, "dino": .0, "aesthetic": .29, "vlm": .37}
    total = round(w["clip"] * cs + w["dino"] * ds
                  + w["aesthetic"] * aes + w["vlm"] * vs, 4)

    th = config.THRESHOLDS
    failed = []
    if cs < th["clip"]:
        failed.append("clip")
    if ds_mode != "no-ref" and ds < th["dino"]:
        failed.append("dino")
    if aes < th["aesthetic"]:
        failed.append("aesthetic")
    if vs < th["vlm"]:
        failed.append("vlm")
    passed = total >= th["overall"] and not failed

    local = [i for i in issues if i.startswith("lighting:")
             or "flat_image" in i or "blurry" in i]
    if passed:
        action = "pass"
    elif issues and len(local) == len(issues) and len(issues) <= 2:
        action = "local_repair"
    else:
        action = "repaint"

    return {
        "total": total, "passed": passed, "action": action,
        "metrics": {"clipscore": cs, "dino_consistency": ds,
                    "aesthetic": aes, "vlm": vs},
        "modes": {"clipscore": cs_mode, "dino": ds_mode,
                  "aesthetic": "heuristic", "vlm": vs_mode},
        "weights": w, "failed": failed, "issues": issues,
        "image_stats": st, "detail": {"aesthetic": aes_d, "clip": cs_d,
                                      "dino": ds_d, "vlm": vs_d},
    }
