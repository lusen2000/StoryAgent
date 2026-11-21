# -*- coding: utf-8 -*-
"""共享黑板 (Blackboard)：所有 Agent 的唯一事实来源，可序列化落盘。"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Character:
    """固化后的角色外观描述：Render Agent 逐镜复用，Reviewer 用于一致性比对。"""
    name: str
    aliases: List[str] = field(default_factory=list)
    appearance: str = ""            # 英文外观 token，保证跨镜一致
    role: str = ""
    ref_image: str = ""             # 角色参考图路径
    palette: List[float] = field(default_factory=list)   # 参考图主色(HSV 均值)


@dataclass
class Scene:
    name: str
    location: str = ""
    time: str = ""
    weather: str = ""
    mood: str = ""


@dataclass
class Beat:
    """叙事节拍：Script Agent 从故事文本抽取。"""
    beat_id: int
    text: str
    title: str = ""
    emotion: str = ""
    characters: List[str] = field(default_factory=list)
    scene: str = ""


@dataclass
class Shot:
    """分镜表的一行：Storyboard Agent 产出。"""
    shot_id: int
    title: str = ""
    beat_id: int = 0
    shot_size: str = ""            # 景别
    composition: str = ""          # 构图
    lighting: str = ""             # 光线
    camera: str = ""               # 运镜
    characters: List[str] = field(default_factory=list)
    emotion: str = ""
    scene_hint: str = ""
    action: str = ""
    prompt: str = ""               # Render Agent 拼装后回填
    negative_prompt: str = ""
    seed: int = 0
    image_path: str = ""
    backend: str = ""
    gen_seconds: float = 0.0
    attempts: int = 1
    review: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Blackboard:
    title: str = ""
    story_text: str = ""
    global_style: str = ""
    negative_prompt: str = ""
    characters: Dict[str, Character] = field(default_factory=dict)
    scenes: Dict[str, Scene] = field(default_factory=dict)
    beats: List[Beat] = field(default_factory=list)
    shots: List[Shot] = field(default_factory=list)
    global_check: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    # ---- 黑板读写 ----
    def post(self, agent: str, key: str, value: Any) -> None:
        setattr(self, key, value)
        self.log(agent, "写入黑板 <- %s" % key)

    def log(self, agent: str, msg: str) -> None:
        self.events.append("[%s] [%s] %s" % (time.strftime("%H:%M:%S"), agent, msg))

    def char(self, name: str) -> Optional[Character]:
        return self.characters.get(name)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["characters"] = {k: asdict(v) for k, v in self.characters.items()}
        d["scenes"] = {k: asdict(v) for k, v in self.scenes.items()}
        d["beats"] = [asdict(b) for b in self.beats]
        d["shots"] = [asdict(s) for s in self.shots]
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Blackboard":
        d = json.loads(path.read_text(encoding="utf-8"))
        bb = cls()
        for k in ("title", "story_text", "global_style", "negative_prompt",
                  "global_check", "meta", "events", "created_at"):
            if k in d:
                setattr(bb, k, d[k])
        bb.characters = {k: Character(**v) for k, v in d.get("characters", {}).items()}
        bb.scenes = {k: Scene(**v) for k, v in d.get("scenes", {}).items()}
        bb.beats = [Beat(**b) for b in d.get("beats", [])]
        bb.shots = [Shot(**s) for s in d.get("shots", [])]
        return bb
