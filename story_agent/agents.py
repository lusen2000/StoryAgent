# -*- coding: utf-8 -*-
"""多智能体层：Script / Storyboard / Render / Reviewer / Consistency。"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from story_agent import console, metrics, script_parser
from story_agent.backends import BaseBackend
from story_agent.blackboard import Beat, Blackboard, Character, Scene, Shot


def _font(sz, bold=False):
    try:
        return ImageFont.truetype(config.FONTS["bold"] if bold else config.FONTS["regular"], sz)
    except Exception:
        return ImageFont.load_default()


class BaseAgent:
    name = "BaseAgent"

    def __init__(self, bb: Blackboard, outdir: Path):
        self.bb = bb
        self.outdir = outdir

    def say(self, msg):
        console.agent(self.name, msg)
        self.bb.log(self.name, msg)

    def run(self, *a, **kw):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Script Agent —— 预处理
# ---------------------------------------------------------------------------
class ScriptAgent(BaseAgent):
    """读【任意】故事文本 → 抽角色/场景/情绪 → 固化外观与全局风格 → 写共享黑板。

    解析逻辑全部在 script_parser：规则版零依赖，或可选 LLM 结构化抽取。
    本 Agent 不再包含任何故事专属常量。
    """

    name = "ScriptAgent"

    def run(self, mode: str = "rule") -> Blackboard:
        bb = self.bb
        text = bb.story_text.strip()
        parsed = script_parser.parse_story(text, mode=mode)
        self.say("读入 %d 字故事（%s）：%s" % (len(text), parsed.mode, parsed.note))
        bb.title = parsed.title
        bb.meta["script_mode"] = parsed.mode

        # 角色：外观 token 固化后写黑板，后续逐镜复用
        chars: Dict[str, Character] = {}
        for c in parsed.characters:
            chars[c.name] = Character(name=c.name, aliases=c.aliases,
                                      appearance=c.appearance, role=c.role,
                                      palette=list(c.palette))
            self.say("角色 %-6s · %s · 外观 token %d chars"
                     % (c.name, c.role, len(c.appearance)))
        bb.post(self.name, "characters", chars)

        # 场景：由节拍的地点/时间/天气/情绪动态生成
        scenes: Dict[str, Scene] = {}
        for b in parsed.beats:
            key = b.location or "故事场景"
            scenes.setdefault(key, Scene(name=key, location=b.location,
                                         time=b.time, weather=b.weather,
                                         mood=b.emotion))
        bb.post(self.name, "scenes", scenes)
        self.say("抽取场景 %d 个：%s" % (len(scenes), "、".join(list(scenes)[:5])))

        # 节拍
        beats = [Beat(beat_id=i + 1, text=b.text, title=b.title,
                      emotion=b.emotion, characters=b.characters,
                      scene=b.location) for i, b in enumerate(parsed.beats)]
        bb.post(self.name, "beats", beats)
        self.say("切出 %d 个叙事节拍，情绪曲线: %s"
                 % (len(beats), " → ".join(b.emotion for b in beats)))

        # 全局风格串：由故事时代/题材推导，不写死
        bb.post(self.name, "global_style", parsed.global_style)
        bb.post(self.name, "negative_prompt", config.NEGATIVE)
        self.say("全局风格串固化 (%d chars): %s ..."
                 % (len(parsed.global_style), parsed.global_style[:62]))
        return bb


# ---------------------------------------------------------------------------
# 2. Storyboard Agent —— 分镜规划
# ---------------------------------------------------------------------------
# 景别/构图/光线/运镜：由镜头文本关键词动态选择（无故事专属内容）
_SIZE_CYCLE = ["中景", "全景", "近景", "特写", "远景"]
_SIZE_RULES = [
    (r"手|指|攥|叩|书|卷|帛|剑柄|剖|刀|眼|泪|灯|微|细|特写", "特写"),
    (r"齐|众|群|队伍|九百|人群|振臂|高呼|呼喊|宣言|起事|开拔|并杀", "全景"),
    (r"低声|密谋|商议|对话|相对|对坐|说|问|答|耳语", "近景"),
    (r"远望|远方|山|河|旷野|一路|大地|火光| establishing", "远景"),
]
_COMP_BY_SIZE = {
    "特写": "extreme close-up, shallow depth of field, subject filling frame",
    "近景": "medium close-up two-shot, faces prominent, restrained framing",
    "中景": "medium shot, rule of thirds, subject on one third with negative space",
    "全景": "wide shot, low angle heroic composition, subject centered, environment visible",
    "远景": "epic extreme wide shot, silhouettes against atmosphere, deep perspective",
}
_LIGHT_RULES = [
    (r"雨|淫雨|泥|storm", "cold blue-grey rain light, wet reflections, desaturated overcast"),
    (r"夜|烛|灯火|深夜|夜半|暗", "flickering warm candlelight against deep shadow, strong chiaroscuro"),
    (r"晓|晨|日出|黎明|破晓", "breaking dawn golden backlight, dust haze, soft rim light"),
    (r"火|篝|烽|burning", "eerie orange firelight, deep blue ambient night"),
    (r"雪|冬|寒", "cold pale winter light, blue tint, low contrast"),
    (r"晴|烈日|阳|夏", "warm low-angle sunlight, golden dust in air"),
]
_CAM_RULES = [
    (r"奔|追|夺|杀|冲|乱|惊起", "handheld follow shot, dynamic motion blur"),
    (r"呼|喊|宣|振臂|登|举|高", "slow crane-up, heroic reveal"),
    (r"低声|密|坐|相对|思", "locked-off static shot, slow push-in"),
    (r"远望|望向|一路|开拔|队伍", "slow lateral pan, wide establishing"),
]


def _plan_shot(text: str, idx: int, weather: str, time: str):
    """由文本语义推导镜头语言；任何故事通用。"""
    size = next((v for p, v in _SIZE_RULES if re.search(p, text)),
                _SIZE_CYCLE[idx % len(_SIZE_CYCLE)])
    comp = _COMP_BY_SIZE[size]
    light = next((v for p, v in _LIGHT_RULES
                  if re.search(p, text + " " + weather + " " + time)),
                 "dramatic cinematic key light, deep shadow, filmic contrast")
    cam = next((v for p, v in _CAM_RULES if re.search(p, text)),
               "slow dolly-in, restrained camera movement")
    return size, comp, light, cam


def _action_text(beat) -> str:
    """把节拍原文压成可入 prompt 的情节串：保留多句，而不是只剩第一句。"""
    s = re.sub(r"\s+", "", beat.text)
    # 按句末标点保留切分，再去掉引号，避免把「只需三日。」切成半截
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;])", s) if p.strip()]
    s = "".join(parts[:3])
    s = re.sub(r"[「」『』“”\"'《》〈〉]", "", s)
    if len(s) > 180:
        s = s[:180] + "…"
    return s


class StoryboardAgent(BaseAgent):
    """把节拍表展开为 4-8 镜分镜表：景别/构图/光线/运镜/出场角色。

    镜头语言由每拍文本关键词动态生成，不再使用写死的镜头方案。
    """

    name = "StoryboardAgent"

    def run(self, max_shots: int = 8) -> List[Shot]:
        bb = self.bb
        beats = bb.beats
        if not beats:
            raise RuntimeError("节拍表为空，请先运行 ScriptAgent")
        n = max(4, min(max_shots, 8))
        # 节拍多于镜头数时按序合并；少于时按已有节拍拍（解析器保证 >=4）
        groups: List[List[Beat]] = []
        if len(beats) <= n:
            groups = [[b] for b in beats]
        else:
            step = len(beats) / n
            for i in range(n):
                groups.append(beats[int(i * step):int((i + 1) * step)])

        shots: List[Shot] = []
        for i, grp in enumerate(groups):
            first, last = grp[0], grp[-1]
            merged = " ".join(b.text for b in grp)
            proxy = type("B", (), {"text": merged})()
            sc = bb.scenes.get(first.scene)
            size, comp, light, cam = _plan_shot(
                merged, i, sc.weather if sc else "", sc.time if sc else "")
            seen, chars = set(), []
            for b in grp:
                for c in b.characters:
                    if c in bb.characters and c not in seen:
                        seen.add(c)
                        chars.append(c)
            shots.append(Shot(
                shot_id=i + 1, title=first.title or "第%d幕" % (i + 1),
                beat_id=first.beat_id, shot_size=size, composition=comp,
                lighting=light, camera=cam, characters=chars,
                emotion=last.emotion, scene_hint=first.scene or "故事场景",
                action=_action_text(proxy), seed=config.GEN["seed"] + i * 101,
            ))
        bb.post(self.name, "shots", shots)
        self.say("生成分镜表 %d 镜（要求 4-8 镜，按故事节拍动态编排）" % n)
        console.table(
            ["镜", "标题", "景别", "运镜", "出场角色", "情绪"],
            [[s.shot_id, s.title[:16], s.shot_size, s.camera,
              ",".join(s.characters), s.emotion] for s in shots],
            [4, 20, 8, 30, 28, 16])
        return shots


# ---------------------------------------------------------------------------
# 3. Render Agent —— 逐镜生成
# ---------------------------------------------------------------------------
class RenderAgent(BaseAgent):
    """按分镜拼 prompt，复用角色参考图，调用 T2I 出图。"""

    name = "RenderAgent"

    def __init__(self, bb, outdir, backend: BaseBackend):
        super().__init__(bb, outdir)
        self.backend = backend
        self.ref: Image.Image | None = None
        self.ref_path = outdir / "char_ref.png"
        (outdir / "shots").mkdir(parents=True, exist_ok=True)

    def build_prompt(self, s: Shot) -> str:
        """prompt 主体是这一镜的情节；角色外观只作一致性锚点，放在后面且精简。"""
        bb = self.bb
        cast = []
        for name in s.characters:
            c = bb.characters.get(name)
            if c is None:
                continue
            # 只取身份/服饰/面部三段，避免整段角色卡把情节挤到后面
            cast.append(", ".join(c.appearance.split(",")[:3]))
        parts = [
            "Cinematic film still depicting: " + s.action,
            "Setting: " + s.scene_hint,
            ("On screen: " + "; ".join(cast)) if cast else "",
            "Shot: %s, %s, %s" % (s.shot_size, s.camera, s.composition),
            s.lighting,
            "Mood: " + s.emotion,
            bb.global_style,
        ]
        return ". ".join(p for p in parts if p)

    def prepare_reference(self):
        """生成角色参考图（后续所有镜头复用同一外观 token / 参考基准）。"""
        neural = self.backend.name == "zimage"
        if neural:
            # 动态取主要角色（按出场镜头数排序），不再写死陈胜一行
            ranked = sorted(
                self.bb.characters,
                key=lambda n: -sum(1 for s in self.bb.shots if n in s.characters))
            names = ranked[:3] or list(self.bb.characters)[:3]
            prompt = ("character reference sheet, %d full-body characters side by side, "
                      "plain dark background, consistent lighting, "
                      % len(names)
                      + ". ".join(self.bb.characters[n].appearance for n in names)
                      + ". " + self.bb.global_style)
            try:
                img, info = self.backend.generate(prompt, config.GEN["seed"] - 1,
                                                  type("S", (), {"shot_id": 0, "title": "角色参考",
                                                                 "characters": names,
                                                                 "shot_size": "全景",
                                                                 "camera": "", "lighting": "",
                                                                 "emotion": ""})())
                self.say("角色参考图已生成 (%s, %.1fs, 峰值显存 %.2f GB)"
                         % (info["size"], info["seconds"], info["peak_vram_gb"]))
            except Exception as e:
                self.say("角色参考图生成失败，改用外观色卡: %s" % str(e)[:80])
                img = None
        else:
            img = None
        colorcard = False
        if img is None:
            img = self._color_sheet()
            colorcard = True
            self.say("角色参考图以外观色卡形式固化（降级后端）")
        img.save(self.ref_path)
        # 纯色色卡没有可比的图像分布，拿它做直方图一致性会得到无意义的低分；
        # 这里只把它当外观基准展示，逐镜一致性改由跨镜校验负责。
        self.ref = None if colorcard else img
        if colorcard:
            self.say("色卡仅用于外观固化，不参与逐镜像素级一致性评分")
        for n, c in self.bb.characters.items():
            c.ref_image = str(self.ref_path)
        self.bb.log(self.name, "角色参考图 -> %s" % self.ref_path.name)
        return img

    def _color_sheet(self):
        w, h = 768, 256
        img = Image.new("RGB", (w, h), (20, 22, 27))
        d = ImageDraw.Draw(img)
        d.text((20, 14), "CHARACTER REFERENCE / 角色外观基准卡", font=_font(20, True),
               fill=(235, 224, 200))
        items = list(self.bb.characters.items())
        cw = (w - 40) // max(len(items), 1)
        for i, (n, c) in enumerate(items):
            x = 20 + i * cw
            col = tuple(int(v) for v in c.palette) if c.palette else (70, 70, 80)
            d.rounded_rectangle([x + 6, 62, x + cw - 12, 176], radius=8, fill=col)
            d.text((x + 10, 188), n, font=_font(19, True), fill=(240, 232, 216))
            d.text((x + 10, 218), c.role[:16], font=_font(13), fill=(160, 168, 176))
        return img

    def render_one(self, s: Shot, attempt: int = 1) -> Image.Image:
        prompt = self.build_prompt(s)
        seed = s.seed + (attempt - 1) * 7919
        img, info = self.backend.generate(prompt, seed, s)
        fname = "shot_%02d_%s.png" % (s.shot_id, re.sub(r"[\\/:*?\"<>|]", "", s.title))
        if attempt > 1:
            fname = fname.replace(".png", "_r%d.png" % attempt)
        path = self.outdir / "shots" / fname
        img.save(path)
        s.image_path, s.prompt, s.negative_prompt = str(path), prompt, config.NEGATIVE
        s.backend, s.attempts = info["backend"], attempt
        s.gen_seconds = info["seconds"]
        return img, info


# ---------------------------------------------------------------------------
# 4. Reviewer Agent —— 视觉验收
# ---------------------------------------------------------------------------
class ReviewerAgent(BaseAgent):
    """CLIPScore + DINO 一致性 + 审美分 + VLM 检查；不达标局部修复或重画。"""

    name = "ReviewerAgent"

    def __init__(self, bb, outdir, render: RenderAgent):
        super().__init__(bb, outdir)
        self.render = render

    def run(self):
        self.say("启动视觉验收：CLIPScore / 角色一致性 / 审美分 / VLM")
        rows = []
        for s in self.bb.shots:
            img = Image.open(s.image_path).convert("RGB")
            for attempt in range(1, config.THRESHOLDS["max_repair"] + 1):
                r = metrics.evaluate_shot(img, s, self.render.ref, s.prompt)
                tag = "第%d次" % attempt if attempt > 1 else "初检"
                self.say("SHOT %02d %-12s %s total=%.3f  %s"
                         % (s.shot_id, s.title[:10], tag, r["total"],
                            "PASS" if r["passed"] else "FAIL"))
                m = r["metrics"]
                console.info("CLIP=%.3f(%s)  一致性=%.3f(%s)  审美=%.3f  VLM=%.3f(%s)"
                             % (m["clipscore"], r["modes"]["clipscore"],
                                m["dino_consistency"], r["modes"]["dino"],
                                m["aesthetic"], m["vlm"], r["modes"]["vlm"]))
                if r["issues"]:
                    console.info("问题: " + ", ".join(r["issues"][:4]))

                if r["passed"]:
                    break
                if attempt >= config.THRESHOLDS["max_repair"]:
                    console.warn("达到最大修复次数，保留当前最佳结果")
                    r["action"] = "give_up"
                    break

                if r["action"] == "local_repair":
                    console.info("决策: 局部修复（曝光/对比/锐度），不重画")
                    img = metrics.repair_image(img, r["issues"])
                    img.save(s.image_path)
                else:
                    console.info("决策: 重画（换 seed + 加强角色外观 token）")
                    _, info = self.render.render_one(s, attempt + 1)
                    img = Image.open(s.image_path).convert("RGB")
                    console.info("重画完成 %.1fs  峰值显存 %.2f GB"
                                 % (info["seconds"], info["peak_vram_gb"]))
            s.review = r
            self.bb.log(self.name, "SHOT %02d -> %s total=%.4f"
                        % (s.shot_id, r["action"], r["total"]))
            rows.append([s.shot_id, s.title[:12],
                         "%.3f" % r["metrics"]["clipscore"],
                         "%.3f" % r["metrics"]["dino_consistency"],
                         "%.3f" % r["metrics"]["aesthetic"],
                         "%.3f" % r["metrics"]["vlm"],
                         "%.3f" % r["total"],
                         "PASS" if r["passed"] else r["action"]])
        console.table(["镜", "标题", "CLIP", "一致性", "审美", "VLM", "总分", "结果"],
                      rows, [4, 16, 9, 9, 9, 9, 9, 14])
        return rows


# ---------------------------------------------------------------------------
# 5. Consistency Agent —— 全局校验 + 分镜册
# ---------------------------------------------------------------------------
class ConsistencyAgent(BaseAgent):
    """跨镜检查角色/风格一致性，输出分镜册。"""

    name = "ConsistencyAgent"

    def run(self):
        bb = self.bb
        imgs = []
        for s in bb.shots:
            if s.image_path and Path(s.image_path).exists():
                imgs.append(Image.open(s.image_path).convert("RGB"))

        # 角色外观 token 复用率
        char_scores = {}
        for name, c in bb.characters.items():
            uses = [s for s in bb.shots if name in s.characters]
            # prompt 里只放精简锚点，这里也按同样的锚点校验复用率
            anchor = ", ".join(c.appearance.split(",")[:3])
            hit = sum(1 for s in uses if anchor and anchor in s.prompt)
            char_scores[name] = round(hit / len(uses), 4) if uses else 1.0
        style_hit = sum(1 for s in bb.shots if bb.global_style in s.prompt)
        style_score = round(style_hit / max(len(bb.shots), 1), 4)

        # 跨镜图像色彩分布一致性（HSV 直方图两两比对的均值和）
        hists = [metrics._hsv_hist(im) for im in imgs]
        pair = [metrics.hist_similarity(hists[i], hists[j])
                for i in range(len(hists))
                for j in range(i + 1, len(hists))]
        image_style = round(float(np.mean(pair)) if pair else 0.0, 4)

        passed = sum(1 for s in bb.shots if s.review.get("passed"))
        bb.global_check = {
            "character_consistency": char_scores,
            "style_string_consistency": style_score,
            "cross_shot_palette_consistency": image_style,
            "emotion_arc": [s.emotion for s in bb.shots],
            "shots": len(bb.shots),
            "review_passed": passed,
            "all_generated": len(imgs) == len(bb.shots),
            "backend": bb.meta.get("backend", ""),
            "device": bb.meta.get("device", ""),
        }
        self.say("角色外观 token 复用率: " +
                 ", ".join("%s=%.2f" % (k, v) for k, v in char_scores.items()))
        self.say("全局风格串一致率 = %.3f" % style_score)
        self.say("跨镜色彩分布一致性 = %.3f (%d 组两两比对)"
                 % (image_style, len(pair)))
        self.say("验收通过 %d/%d 镜" % (passed, len(bb.shots)))

        self._contact_sheet(imgs)
        self._preview(imgs)
        self._html(imgs)
        self._report()
        bb.save(self.outdir / "storyboard.json")
        return bb.global_check

    def _contact_sheet(self, imgs):
        if not imgs:
            return
        cols = 2
        tw, th = 620, 350
        rows = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, 54 + rows * (th + 34)), (15, 17, 21))
        d = ImageDraw.Draw(sheet)
        d.text((18, 14), "%s   /   STORYBOARD CONTACT SHEET   /   %d SHOTS   /   %s"
               % (self.bb.title, len(imgs), self.bb.meta.get("backend", "")),
               font=_font(20, True), fill=(240, 226, 196))
        for i, im in enumerate(imgs):
            x, y = (i % cols) * tw, 54 + (i // cols) * (th + 34)
            t = im.copy()
            t.thumbnail((tw - 12, th - 6))
            sheet.paste(t, (x + 6, y + 3))
            d.text((x + 8, y + th + 5),
                   "%02d  %s    %s" % (self.bb.shots[i].shot_id,
                                      self.bb.shots[i].title,
                                      self.bb.shots[i].shot_size),
                   font=_font(15), fill=(206, 210, 216))
        sheet.save(self.outdir / "contact_sheet.png")
        console.ok("分镜总览图 -> output/%s/contact_sheet.png" % self.outdir.name)

    def _wrap_text(self, draw, text: str, font, max_w: int) -> List[str]:
        """按像素宽度折行（中文逐字，英文按词）。"""
        lines, cur = [], ""
        for ch in text:
            trial = cur + ch
            if draw.textlength(trial, font=font) > max_w and cur:
                lines.append(cur)
                cur = ch
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    def _preview(self, imgs):
        """最终预览长图：依次列出每镜【描述 + 分镜图 + 标题】。"""
        if not imgs:
            return
        W = 860
        PAD = 30
        font_title = _font(30, True)
        font_sub = _font(15)
        font_bar = _font(21, True)
        font_tag = _font(14)
        font_desc = _font(16)

        probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        blocks = []
        for i, s in enumerate(self.bb.shots):
            desc = s.action or s.title
            desc_lines = self._wrap_text(probe, desc, font_desc, W - 2 * PAD - 8)[:4]
            tag = "%s ｜ %s ｜ %s" % (s.shot_size, s.emotion,
                                     "、".join(s.characters[:4]) or "无")
            tag_lines = self._wrap_text(probe, tag, font_tag, W - 2 * PAD - 8)[:2]
            im_h = min(560, int((W - 2 * PAD) * imgs[i].height / max(imgs[i].width, 1)))
            blocks.append((s, desc_lines, tag_lines, im_h))
        total_h = 128 + sum(62 + len(d[1]) * 25 + len(d[2]) * 20 + d[3] + 30
                            for d in blocks) + 40
        sheet = Image.new("RGB", (W, total_h), (14, 17, 22))
        d = ImageDraw.Draw(sheet)
        d.text((PAD, 24), self.bb.title, font=font_title, fill=(240, 222, 184))
        d.text((PAD, 68), "分镜预览 · %d 镜 · 后端 %s · %s"
               % (len(imgs), self.bb.meta.get("backend", ""),
                  self.bb.meta.get("device", "")),
               font=font_sub, fill=(150, 162, 176))
        d.line([(PAD, 104), (W - PAD, 104)], fill=(70, 80, 94), width=2)

        y = 120
        for idx, (s, desc_lines, tag_lines, im_h) in enumerate(blocks):
            # 标题条
            d.rounded_rectangle([PAD, y, W - PAD, y + 48], radius=8, fill=(28, 35, 45))
            d.text((PAD + 14, y + 9), "SHOT %02d · %s" % (s.shot_id, s.title),
                   font=font_bar, fill=(232, 201, 140))
            y += 54
            for t in tag_lines:
                d.text((PAD + 4, y), t, font=font_tag, fill=(132, 154, 176))
                y += 20
            for ln in desc_lines:
                d.text((PAD + 4, y), ln, font=font_desc, fill=(206, 214, 222))
                y += 25
            y += 6
            # 分镜图（等比缩放居中）
            thumb = imgs[idx].copy()
            thumb.thumbnail((W - 2 * PAD, im_h), Image.LANCZOS)
            sheet.paste(thumb, (PAD + (W - 2 * PAD - thumb.width) // 2, y))
            y += im_h + 26
            d.line([(PAD, y - 14), (W - PAD, y - 14)], fill=(48, 56, 68), width=1)

        path = self.outdir / "preview.png"
        sheet.save(path)
        console.ok("分镜预览长图（描述+图+标题）-> output/%s/preview.png" % self.outdir.name)
        self.bb.log(self.name, "预览长图 -> %s" % path.name)

    def _html(self, imgs):
        bb = self.bb
        cards = []
        for i, s in enumerate(bb.shots):
            rel = "shots/" + Path(s.image_path).name if s.image_path else ""
            r = s.review or {}
            m = r.get("metrics", {})
            bars = "".join(
                '<div class="bar"><span>%s</span><i style="width:%.0f%%"></i><b>%.2f</b></div>'
                % (k, v * 100, v) for k, v in m.items())
            cards.append("""
<div class="card">
  <div class="img"><img src="%s" alt="shot %d"></div>
  <div class="body">
    <h3>SHOT %02d · %s <em>%s</em></h3>
    <p class="act">%s</p>
    <table>
      <tr><td>景别 / 运镜</td><td>%s / %s</td></tr>
      <tr><td>构图</td><td>%s</td></tr>
      <tr><td>光线</td><td>%s</td></tr>
      <tr><td>出场角色</td><td>%s</td></tr>
      <tr><td>情绪</td><td>%s</td></tr>
      <tr><td>生成后端</td><td>%s · %.1fs</td></tr>
    </table>
    <div class="metrics">%s</div>
    <p class="verdict %s">总分 %.3f · %s · %s</p>
    <details><summary>完整 Prompt</summary><code>%s</code></details>
  </div>
</div>""" % (rel, s.shot_id, s.shot_id, html.escape(s.title),
             html.escape(s.shot_size), html.escape(s.action),
             html.escape(s.shot_size), html.escape(s.camera),
             html.escape(s.composition), html.escape(s.lighting),
             html.escape(", ".join(s.characters)), html.escape(s.emotion),
             html.escape(s.backend), s.gen_seconds, bars,
             "pass" if r.get("passed") else "fail",
             r.get("total", 0), "PASS" if r.get("passed") else "NOT PASS",
             html.escape(r.get("action", "")), html.escape(s.prompt)))
        g = bb.global_check
        doc = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>%s · 分镜册</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;background:#0e1116;color:#e6e9ee;font-family:"Microsoft YaHei",system-ui,sans-serif}
header{padding:34px 40px;background:linear-gradient(120deg,#1b2430,#10141b);border-bottom:1px solid #2b3542}
h1{margin:0 0 8px;font-size:27px;color:#f0d9a8;letter-spacing:1px}
.sub{color:#8b96a5;font-size:13px;line-height:1.8}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:22px;padding:28px 40px}
.card{background:#161c24;border:1px solid #29323e;border-radius:12px;overflow:hidden}
.img{background:#0a0d12} .img img{width:100%%;display:block}
.body{padding:18px 20px} h3{margin:0 0 10px;font-size:18px;color:#e8c98c}
h3 em{font-style:normal;font-size:12px;color:#7f8b99;margin-left:10px}
.act{color:#b9c2cc;font-size:13px;margin:0 0 12px;line-height:1.7}
table{width:100%%;font-size:12px;color:#aab4c0;border-collapse:collapse}
td{padding:4px 0;vertical-align:top} td:first-child{width:96px;color:#6f7d8c}
.metrics{margin:14px 0 8px} .bar{display:flex;align-items:center;gap:8px;font-size:11px;color:#93a0ae;margin:5px 0}
.bar span{width:78px} .bar i{display:block;height:6px;background:linear-gradient(90deg,#c98a4b,#e8c98c);border-radius:4px}
.bar b{font-weight:500;color:#d8dde4}
.verdict{font-size:12px;margin:10px 0 4px;padding:7px 10px;border-radius:6px}
.verdict.pass{background:#18301f;color:#8fd6a0} .verdict.fail{background:#331d1d;color:#e8a09a}
details{margin-top:10px;font-size:11px;color:#7f8b99} summary{cursor:pointer}
code{display:block;white-space:pre-wrap;margin-top:8px;color:#8895a4;line-height:1.6}
.summary{margin:0 40px 30px;padding:20px;background:#161c24;border:1px solid #29323e;border-radius:12px;font-size:13px;line-height:2}
.summary b{color:#e8c98c} footer{padding:20px 40px 40px;color:#6f7d8c;font-size:12px;white-space:pre-wrap;line-height:1.9}
</style></head><body>
<header><h1>%s</h1><div class="sub">
%s<br>生成后端: <b>%s</b> ｜ 设备: %s ｜ 共 %d 镜 ｜ 验收通过 %d 镜<br>
全局风格串: %s</div></header>
<div class="summary"><b>全局校验</b><br>
角色外观一致率: %s<br>风格串一致率: %.3f ｜ 跨镜色彩一致性: %.3f<br>
情绪曲线: %s</div>
<div class="grid">%s</div>
<footer><b>输入故事文本</b>\n%s</footer>
</body></html>""" % (
            html.escape(bb.title), html.escape(bb.title),
            html.escape("A Multi-Agent Storyboard Generation System for Narrative-Consistent Visual Storytelling"),
            html.escape(str(bb.meta.get("backend", ""))),
            html.escape(str(bb.meta.get("device", ""))),
            len(bb.shots), g.get("review_passed", 0),
            html.escape(bb.global_style[:110]) + " ...",
            ", ".join("%s %.2f" % (k, v) for k, v in g.get("character_consistency", {}).items()),
            g.get("style_string_consistency", 0),
            g.get("cross_shot_palette_consistency", 0),
            html.escape(" → ".join(g.get("emotion_arc", []))),
            "".join(cards), html.escape(bb.story_text))
        (self.outdir / "storyboard.html").write_text(doc, encoding="utf-8")
        console.ok("分镜册 HTML -> output/%s/storyboard.html" % self.outdir.name)

    def _report(self):
        bb = self.bb
        g = bb.global_check
        lines = [
            "# %s — 运行报告" % bb.title, "",
            "> A Multi-Agent Storyboard Generation System for Narrative-Consistent Visual Storytelling", "",
            "## 环境", "",
            "- 后端: `%s`" % bb.meta.get("backend", ""),
            "- 设备: `%s`" % bb.meta.get("device", ""),
            "- 运行模式: `%s`" % bb.meta.get("mode", ""),
            "- 显存峰值: `%s GB`" % bb.meta.get("peak_vram_gb", 0),
            "- 生成分辨率: `%dx%d`, steps=%d, guidance=%.1f" %
            (config.GEN["width"], config.GEN["height"], config.GEN["steps"], config.GEN["guidance"]), "",
            "## 流程", "",
            "1. **预处理** Script Agent（角色/场景/情绪/风格） + Storyboard Agent（%d 镜）" % len(bb.shots),
            "2. **生成** Render Agent 逐镜拼 prompt 并调用 T2I",
            "3. **评估** Reviewer Agent 指标+VLM 验收 → Consistency Agent 跨镜校验 → 分镜册", "",
            "## 逐镜验收", "",
            "| 镜 | 标题 | CLIP | 一致性 | 审美 | VLM | 总分 | 结果 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in bb.shots:
            m = s.review.get("metrics", {})
            lines.append("| %02d | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %s |" %
                         (s.shot_id, s.title, m.get("clipscore", 0),
                          m.get("dino_consistency", 0), m.get("aesthetic", 0),
                          m.get("vlm", 0), s.review.get("total", 0),
                          "PASS" if s.review.get("passed") else s.review.get("action", "FAIL")))
        lines += ["", "指标模式: `%s`" % json.dumps(
            bb.shots[0].review.get("modes", {}), ensure_ascii=False) if bb.shots else "",
            "（proxy = 本地可解释代理实现，未加载深度模型权重；不会伪造满分）", "",
            "## 全局校验", "",
            "- 角色外观 token 复用率: `%s`" % g.get("character_consistency", {}),
            "- 全局风格串一致率: `%.3f`" % g.get("style_string_consistency", 0),
            "- 跨镜色彩分布一致性: `%.3f`" % g.get("cross_shot_palette_consistency", 0),
            "- 验收通过: `%d/%d`" % (g.get("review_passed", 0), len(bb.shots)), "",
            "## 产物", "",
            "- `shots/*.png` 逐镜画面",
            "- `contact_sheet.png` 分镜总览",
            "- `storyboard.html` 分镜册（含完整 prompt 与指标）",
            "- `storyboard.json` 共享黑板快照", "",
            "## 输入故事文本", "", "```text", bb.story_text.strip(), "```", "",
        ]
        (self.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        console.ok("运行报告 -> output/%s/report.md" % self.outdir.name)
