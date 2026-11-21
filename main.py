# -*- coding: utf-8 -*-
"""StoryBoard Agent — 多智能体故事分镜生成系统入口。

用法:
    python main.py                          # 自动选后端，完整跑陈胜起义 8 镜
    python main.py --backend zimage         # 强制 Z-Image
    python main.py --backend mock           # 强制本地渲染（冒烟测试）
    python main.py --shots 4                # 只跑 4 镜
    python main.py --mode full_gpu          # 尝试全量驻留 GPU（16G 会 OOM）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from story_agent import console
from story_agent.agents import (ConsistencyAgent, RenderAgent, ReviewerAgent,
                                ScriptAgent, StoryboardAgent)
from story_agent.backends import build_backend
from story_agent.blackboard import Blackboard


def parse_args():
    p = argparse.ArgumentParser(description="StoryBoard Agent: Multi-Agent Consistent Story Visualization")
    p.add_argument("--story", default="chensheng",
                   help="故事文件名(不含 .txt) 或任意 .txt 路径")
    p.add_argument("--script-mode", default="rule", choices=["rule", "llm", "auto"],
                   help="预处理模式：rule=规则文本处理(默认)，llm=结构化抽取，auto=有LLM配置先用LLM再回退")
    p.add_argument("--backend", default="auto", choices=["auto", "zimage", "mock", "cloud"])
    p.add_argument("--mode", default="auto",
                   choices=["auto", "offload", "sequential", "manual",
                            "full_gpu", "cpu"])
    p.add_argument("--shots", type=int, default=8)
    p.add_argument("--size", default="", help="覆盖分辨率，如 512x288（16 的倍数）")
    p.add_argument("--steps", type=int, default=0, help="覆盖推理步数（Turbo 推荐 9）")
    p.add_argument("--no-ref", action="store_true", help="跳过角色参考图生成")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.size:
        w, h = (int(x) for x in args.size.lower().split("x"))
        config.GEN["width"], config.GEN["height"] = w, h
    if args.steps:
        config.GEN["steps"] = args.steps
    t0 = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")

    # ---------- 读入故事：内置故事名 或 任意 .txt 路径 ----------
    candidate = Path(args.story)
    story_path = candidate if candidate.suffix.lower() == ".txt" and candidate.exists() \
        else config.STORY_DIR / (args.story + ".txt")
    if not story_path.exists():
        raise FileNotFoundError("找不到故事文件: %s" % story_path)
    # 输出目录只用文件名 stem，避免绝对路径中的盘符/分隔符造成非法目录名
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", story_path.stem)[:48] or "story"

    outdir = config.OUTPUT_ROOT / ("%s_%s" % (safe_name, run_id))
    outdir.mkdir(parents=True, exist_ok=True)
    console.set_log(outdir / "run.log")

    console.banner(
        "StoryBoard Agent · 多智能体故事分镜生成系统",
        "A Multi-Agent Storyboard Generation System for Narrative-Consistent Visual Storytelling")
    console.kv("运行目录", outdir)
    console.kv("故事", str(story_path))
    console.kv("后端策略", args.backend)

    story_text = story_path.read_text(encoding="utf-8").strip()
    bb = Blackboard(title=story_path.stem, story_text=story_text)
    console.kv("故事文本", "%d 字 / %s" % (len(story_text), story_path.name))
    console.kv("预处理模式", args.script_mode)

    # ---------- 后端选择 ----------
    console.stage(0, "环境探测 / 后端选择")
    from story_agent.backends import ZImageBackend
    z = ZImageBackend()
    good, why = z.available()
    console.kv("Z-Image 权重", why)
    try:
        import torch
        if torch.cuda.is_available():
            console.kv("GPU", "%s  %.1f GB  sm_%d%d  CUDA %s  torch %s" %
                       (torch.cuda.get_device_name(0),
                        torch.cuda.get_device_properties(0).total_memory / 1e9,
                        *torch.cuda.get_device_capability(0),
                        torch.version.cuda, torch.__version__))
        else:
            console.kv("GPU", "CUDA 不可用")
    except Exception as e:
        console.kv("GPU", "torch 不可用: %s" % e)

    backend = build_backend(args.backend)
    if isinstance(backend, ZImageBackend):
        backend.mode = args.mode
        backend.load()
        bb.meta["device"] = "%s / %s" % (backend.info.get("gpu"), backend.info.get("mode"))
        bb.meta["mode"] = backend.info.get("mode")
        console.ok("后端就绪: Z-Image-Turbo (GPU: %s)" % backend.info.get("mode"))
    else:
        bb.meta["device"] = "CPU / PIL"
        bb.meta["mode"] = backend.name
        console.warn("后端就绪: %s（降级）" % backend.name)
    bb.meta["backend"] = backend.name
    bb.meta["run_id"] = run_id

    # ================= STAGE 1 预处理 =================
    console.stage(1, "预处理 PREPROCESS",
                  "Script Agent 抽取角色/场景/情绪 → Storyboard Agent 编排分镜表")
    ScriptAgent(bb, outdir).run(mode=args.script_mode)
    shots = StoryboardAgent(bb, outdir).run(max_shots=args.shots)
    console.ok("预处理完成：%d 个角色 / %d 个场景 / %d 个节拍 / %d 个镜头"
               % (len(bb.characters), len(bb.scenes), len(bb.beats), len(shots)))

    # ================= STAGE 2 生成 =================
    console.stage(2, "生成 GENERATION",
                  "Render Agent 按分镜拼 prompt，复用角色参考图，调用 T2I 逐镜出图")
    render = RenderAgent(bb, outdir, backend)
    if not args.no_ref:
        render.prepare_reference()
    else:
        console.info("跳过角色参考图（--no-ref）")

    gen_rows = []
    peak = 0.0
    for s in shots:
        console.info("-" * 62)
        console.info("SHOT %02d  《%s》  %s / %s" %
                     (s.shot_id, s.title, s.shot_size, s.camera))
        console.info("  prompt: %s ..." % render.build_prompt(s)[:150].replace("\n", " "))
        _, info = render.render_one(s)
        peak = max(peak, float(info.get("peak_vram_gb", 0)))
        console.ok("出图 %s  %.1fs  峰值显存 %.2f GB  seed=%d"
                   % (info["size"], info["seconds"], info["peak_vram_gb"], s.seed))
        gen_rows.append([s.shot_id, s.title[:12], info["backend"][:20],
                         "%.1fs" % info["seconds"], "%.2f" % info["peak_vram_gb"]])
    bb.meta["peak_vram_gb"] = peak
    console.table(["镜", "标题", "后端", "耗时", "显存GB"], gen_rows,
                  [4, 16, 24, 10, 10])
    console.ok("生成完成：%d 张图片，显存峰值 %.2f GB，累计 %.1fs"
               % (len(shots), peak, time.time() - t0))

    # ================= STAGE 3 评估 =================
    console.stage(3, "评估 EVALUATION",
                  "Reviewer Agent 指标+VLM 验收 → Consistency Agent 跨镜一致性 → 分镜册")
    ReviewerAgent(bb, outdir, render).run()
    g = ConsistencyAgent(bb, outdir).run()

    console.banner("运行完成 SUMMARY")
    console.kv("总耗时", "%.1f s" % (time.time() - t0))
    console.kv("后端 / 设备", "%s / %s" % (bb.meta["backend"], bb.meta["device"]))
    console.kv("显存峰值", "%.2f GB" % peak)
    console.kv("分镜数 / 通过", "%d / %d" % (len(shots), g["review_passed"]))
    console.kv("跨镜色彩一致性", "%.3f" % g["cross_shot_palette_consistency"])
    console.kv("分镜册", str(outdir / "storyboard.html"))
    console.kv("预览长图", str(outdir / "preview.png"))
    console.kv("控制台日志", str(outdir / "run.log"))

    # ---------- 最后打印输入的故事文本 ----------
    console.banner("输入的故事文本 INPUT STORY TEXT")
    print(bb.story_text, flush=True)
    if console._LOG_PATH:
        with console._LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(bb.story_text + "\n")
    console.banner("END")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.err("用户中断")
        sys.exit(130)
    except Exception as e:
        console.err("运行失败: %s" % e)
        traceback.print_exc()
        sys.exit(1)
