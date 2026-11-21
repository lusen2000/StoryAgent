# -*- coding: utf-8 -*-
"""通用故事解析器：把任意故事文本变成结构化"剧本"。

两种模式：
  rule —— 零依赖规则版（默认）。中文人名/群体/场景/情绪/节拍抽取。
  llm  —— 调用 OpenAI 兼容的 /chat/completions 做结构化抽取，失败自动回退 rule。

对外只暴露 parse_story(text, mode) -> ParsedStory，ScriptAgent 消费它即可，
任何故事都能走同一条流水线。
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 数据结构（与 blackboard 解耦，ScriptAgent 负责搬运）
# ---------------------------------------------------------------------------
@dataclass
class ParsedCharacter:
    name: str
    aliases: List[str] = field(default_factory=list)
    appearance: str = ""
    role: str = ""
    palette: List[float] = field(default_factory=list)


@dataclass
class ParsedBeat:
    text: str
    title: str = ""
    emotion: str = ""
    characters: List[str] = field(default_factory=list)
    location: str = ""
    time: str = ""
    weather: str = ""


@dataclass
class ParsedStory:
    title: str = ""
    global_style: str = ""
    characters: List[ParsedCharacter] = field(default_factory=list)
    beats: List[ParsedBeat] = field(default_factory=list)
    mode: str = "rule"
    note: str = ""


# ---------------------------------------------------------------------------
# 规则词典
# ---------------------------------------------------------------------------
_SURNAME = (
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄"
    "和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁"
    "杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍"
    "虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚"
    "程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓"
    "牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙"
    "叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻"
    "莘党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀郏浦尚农温"
    "别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡"
    "国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾"
    "毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公"
)
_DOUBLE_SURNAME = ("司马", "欧阳", "上官", "诸葛", "夏侯", "东方", "独孤",
                   "南宫", "万俟", "闻人", "公孙", "慕容", "长孙", "宇文",
                   "司徒", "鲜于", "太叔", "申屠", "轩辕", "令狐", "皇甫")
_NAME_STOP = {
    "王侯", "将军", "天下", "中国", "成败", "生死", "富贵", "鸿鹄", "燕雀",
    "众人", "士卒", "戍卒", "校尉", "都尉", "屯长", "高台", "丛祠", "大楚",
    "士人", "丈夫", "壮士", "将军", "足下", "将军", "王侯将相", "秦军",
    "连绵", "都督", "诸将", "吴军", "蜀军", "曹军", "大军", "连营", "江上",
    "束草", "万支", "伏兵", "弓弩", "满载", "人间", "天下", "此天",
    "林木", "白帝", "麦城", "夷陵", "巫峡", "江陵", "柴桑", "赤壁",
}
# 3 字候选的尾字若是常见虚词/身份后缀，多半是"姓名+语法字"误切
_NAME_TAIL = set("王自与等者曰云谓为将于乃即亦皆而之以其说时后中上下也矣焉哉乎则此其为以")
# 通用身份群体（任何古代/现代叙事都可能出现）
_TITLE_GROUPS = [
    ("军官", r"校尉|将尉|都尉|军官|县令|县丞|官吏|警长|队长|军官们",
     "an authoritative uniformed officer in period clothing, hard cold expression, standing upright",
     (40, 44, 54)),
]

# 群体角色（故事里反复出现的"一群人"）
_GROUPS = [
    ("戍卒们", "戍卒|徒属|士卒|九百人|九百戍卒|士兵|卒伍",
     "a crowd of exhausted ancient Chinese conscript soldiers in coarse hemp clothes, "
     "bamboo hats, thin tired faces", (96, 88, 76)),
    ("众人", "众人|众人皆|诸人|大家|围观者|百姓|群众|村民|人群",
     "a crowd of common ancient Chinese townspeople in simple rough clothes, "
     "curious anxious faces", (104, 96, 84)),
    ("将士", "将士|群臣|诸将|军士|骑兵|兵士|部下|随从",
     "a group of ancient Chinese armed soldiers and retainers in uniform armor", (60, 66, 78)),
]

# 身份/外貌线索（上下文命中即写入外观 token）
_ROLE_HINTS = [
    # 具体身份在前，泛指士兵放最后；避免"军令状/伏兵"里的单字把谋士误判成士兵
    (r"僧|寺|和尚|道人|道士|尼", "monk", "simple grey robe, shaved head or topknot, serene face"),
    (r"书生|儒|学者|先生|博士|军师|谋士", "scholar", "long pale robe, scholar's cap, thoughtful expression"),
    (r"将|尉|校尉|都尉|将军|都督|统帅|将领", "army officer", "dark lamellar armor, helmet, stern hard expression"),
    (r"王|侯|帝|君|主公|贵族|陛下", "noble lord", "layered silk robe, ornate hairpin, dignified bearing"),
    (r"女|妇|她|娘|夫人|少女", "woman", "traditional long dress, hair in a bun, delicate features"),
    (r"老|翁|叟|老者", "elderly person", "grey hair, deep wrinkles, weathered face"),
    (r"少年|少时|童|孩", "young person", "youthful face, short topknot, lean build"),
    (r"农|耕|田|锄|垄|农夫|农民", "peasant farmer", "coarse hemp tunic, sun-tanned weathered face, calloused hands"),
    (r"戍卒|士卒|士兵|兵士|军士|卒伍|行伍|当兵", "soldier", "worn hemp tunic, leather belt, tired determined expression"),
]
_AGE_HINT = [(r"老|翁|叟|老者", "elderly"), (r"少年|少时|童|孩", "young"), (r"壮| muscular", "muscular adult")]
_EMOTION_BUCKETS = [
    (r"绝望|死|斩|杀|亡|恐惧|怕|惊", "绝望与恐惧"),
    (r"怒|愤|激|拔剑|夺|反抗|暴|厉", "暴烈与反抗"),
    (r"密谋|计|誓|低声|窃|策|谋", "决绝与密谋"),
    (r"神|鬼|狐|异|怪|兆|符|书", "神秘与期待"),
    (r"敬|畏|骚动|惊起|拜|服", "敬畏与骚动"),
    (r"志|望|叹|嗟|鸿鹄|不甘|怅", "不甘与志向"),
    (r"呼|应|举|起|振|昂|胜|凯|歌", "激昂与昂扬"),
    (r"悲|哭|泣|哀|别|离", "悲伤与离别"),
    (r"喜|笑|乐|欢|贺", "喜悦"),
    (r"疑|惑|奇|异|惊疑", "惊疑"),
]
_LOCATION_WORDS = ("田野", "田垄", "驿道", "营帐", "营地", "灶台", "丛祠", "祠", "校场",
                   "高台", "土台", "宫殿", "宫", "殿", "街", "市", "酒肆", "客栈", "府",
                   "宅", "船上", "船", "桥", "山", "河", "林", "寺", "庙", "关", "城",
                   "塞", "寨", "洞", "谷", "海边", "篮球场", "教室", "办公室", "车站")
_TIME_WORDS = ("夜半", "深夜", "当夜", "夜", "拂晓", "破晓", "清晨", "次日", "黎明",
               "黄昏", "傍晚", "午后", "正午", "白天", "日出")
_WEATHER_WORDS = ("大雨", "淫雨", "雨", "雪", "风", "雾", "雷电", "雷", "沙尘", "晴", "阴")
_ERA_WORDS = [(r"三国|赤壁|曹操|诸葛亮|周瑜", "Three Kingdoms era China"),
              (r"秦二世|秦朝|秦军|陈胜|吴广|战国", "Qin dynasty China"),
              (r"汉朝|刘邦|汉武帝|大汉", "Han dynasty China"),
              (r"唐朝|李世民|长安", "Tang dynasty China"),
              (r"宋朝|赵匡胤|汴京", "Song dynasty China"),
              (r"明朝|朱元璋|大明", "Ming dynasty China"),
              (r"清朝|康熙|乾隆|大清", "Qing dynasty China")]
_BASE_STYLE = ("cinematic historical epic film still, %s, dramatic chiaroscuro lighting, "
               "realistic oil-painting texture, wuxia color grading with russet and "
               "steel-blue tones, film grain, shallow depth of field, anamorphic lens, "
               "high contrast, epic scale, 2.39:1 cinematic framing")


# ---------------------------------------------------------------------------
# 规则抽取
# ---------------------------------------------------------------------------
def _split_beats(text: str, target_min: int = 4, target_max: int = 8) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paras) == 1:
        paras = [p.strip() for p in re.split(r"(?<=[。！？!?；;])\s*", text) if p.strip()]
    if len(paras) > target_max:
        # 均分合并
        out, step = [], len(paras) / target_max
        i = 0.0
        while i < len(paras):
            j = min(int(i + step), len(paras))
            out.append(" ".join(paras[int(i):j]))
            i += step
        paras = out
    if len(paras) < target_min:
        # 按句再切
        parts = []
        for p in paras:
            ss = [s.strip() for s in re.split(r"(?<=[。！？!?])", p) if s.strip()]
            parts.extend(ss if ss else [p])
        if len(parts) >= target_min:
            step = len(parts) / min(target_max, len(parts))
            out, i = [], 0.0
            while i < len(parts):
                j = min(int(i + step), len(parts))
                out.append(" ".join(parts[int(i):j]))
                i += step
            paras = out
    return paras[:target_max]


def _detect_names(text: str, max_n: int = 8) -> List[Tuple[str, int]]:
    """中文人名：姓 + 1~2 字，按出现次数排序；取说话人身份加成。"""
    surname_class = "[%s]" % re.escape(_SURNAME)
    # 单姓取 2 字名（周瑜/陈胜），复姓取 2-3 字（诸葛亮/司马光）。
    # 若单姓贪婪取 3 字会把语法字吞进去（周瑜欲→周瑜欲），故单姓固定 1 字。
    pattern = re.compile(
        r"(?:(?:%s)[\u4e00-\u9fa5]|%s[\u4e00-\u9fa5])"
        % ("|".join(_DOUBLE_SURNAME), surname_class))
    freq: Dict[str, int] = {}
    for m in pattern.finditer(text):
        w = m.group(0)
        if w in _NAME_STOP or len(w) < 2:
            continue
        freq[w] = freq.get(w, 0) + 1
    # 说话人加权，减少"王侯"之类误检
    for m in re.finditer(r"([\u4e00-\u9fa5]{2,4})(?:曰|说道|大声说|低声说|喊道|问道|答道)\s*[「“']", text):
        w = m.group(1)
        if w in freq:
            freq[w] += 3
        elif re.match(pattern, w) and w not in _NAME_STOP:
            freq[w] = 3
    # 纠正"姓名+语法字"误切：诸葛亮度 -> 诸葛亮；陈胜王 -> 陈胜
    fixed: Dict[str, int] = {}
    for n, c in freq.items():
        # 复姓全名（诸葛亮/司马光）不砍尾字
        if n.startswith(_DOUBLE_SURNAME) and len(n) >= 3:
            fixed[n] = fixed.get(n, 0) + c
        elif len(n) >= 3 and n[-1] in _NAME_TAIL:
            fixed[n[:-1]] = fixed.get(n[:-1], 0) + c
        else:
            fixed[n] = fixed.get(n, 0) + c
    # 取出现次数 >=2 的候选；明显说话人（后接曰/说）即使只 1 次也保留
    speakers = set(m.group(1) for m in re.finditer(
        r"([\u4e00-\u9fa5]{2,3})(?:曰|说道|大声说|低声说|喊道|问道|答道)\s*[「“']", text))
    cand = [(n, c) for n, c in fixed.items() if c >= 2 or n in speakers]
    cand.sort(key=lambda x: -x[1])
    # "陈涉字涉"类别名归一：字X 里的 X 合并到原名，避免重复角色
    alias_map: Dict[str, str] = {}
    for m in re.finditer(r"([\u4e00-\u9fa5]{2,3})[，,]?\s*字([\u4e00-\u9fa5]{1,2})", text):
        alias_map[m.group(2)] = m.group(1)
    merged: Dict[str, int] = {}
    for n, c in cand:
        key = alias_map.get(n, n)
        merged[key] = merged.get(key, 0) + c
    out = sorted(merged.items(), key=lambda x: -x[1])
    return out[:max_n]


def _appearance(name: str, context: str, era: str, freq: int) -> Tuple[str, str, List[float]]:
    """从姓名与上下文合成稳定的英文外观 token（跨镜复用）。"""
    role, cloth = "ancient Chinese person", "simple period clothing, neat hair"
    # 身份头衔通常在名字紧邻处；只看局部窗口，避免整段 context 里的
    # "士卒/伏兵/军令状"把具名角色（如诸葛亮）误判成士兵
    local_parts = []
    for m in re.finditer(re.escape(name), context):
        local_parts.append(context[max(0, m.start() - 12):m.end() + 4])
    local = " ".join(local_parts) or context[:60]
    for pat, r, c in _ROLE_HINTS:
        if re.search(pat, name) or re.search(pat, local):
            role, cloth = r, c
            break
    age = next((a for p, a in _AGE_HINT if re.search(p, context)), "adult")
    is_female = bool(re.search(r"女|妇|她|娘|夫人|少女", context))
    if is_female and role == "ancient Chinese person":
        role = "ancient Chinese woman"
    tone = "calm"
    if re.search(r"怒|愤|激|厉|杀", context):
        tone = "fierce determined"
    elif re.search(r"悲|绝望|哭|哀", context):
        tone = "sorrowful"
    elif re.search(r"笑|喜|乐", context):
        tone = "warm gentle"
    appearance = (
        "a %s %s, %s, %s face, expressive eyes, full body visible, "
        "consistent character design" % (age, role, cloth, tone))
    # 稳定调色板：按名字哈希，保证每镜角色色一致
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) % 360
    palette = [float((h + i * 17) % 256) for i in range(3)]
    return appearance, role, palette


def _pick(words, text, default=""):
    for w in words:
        if w in text:
            return w
    return default


def _clean_title(sentence: str, idx: int) -> str:
    """从节拍文本里取一个 4~12 字的短标题。"""
    s = re.sub(r"[「」『』“”\"'《》〈〉]", "", sentence).strip()
    # 优先引号中的台词/关键词
    quoted = re.findall(r"[「“]([^」”]{2,14})[」”]", sentence)
    if quoted:
        s = quoted[0]
    else:
        s = re.split(r"[，,。；;：:！!？?]", s)[0]
    s = s.strip(" 　·—-")
    if len(s) > 12:
        s = s[:12]
    return s if len(s) >= 2 else "第%d幕" % idx


def parse_rule(text: str) -> ParsedStory:
    beats_text = _split_beats(text)
    era = next((v for p, v in _ERA_WORDS if re.search(p, text[:1200])),
               "a dramatic cinematic setting")
    style = _BASE_STYLE % era

    names = _detect_names(text)
    chars: List[ParsedCharacter] = []
    # 具名角色
    for name, freq in names:
        ctx = " ".join(b for b in beats_text if name in b)[:500] or text[:500]
        app, role, pal = _appearance(name, ctx, era, freq)
        aliases = []
        m = re.search(re.escape(name) + r"[，,]?\s*字([\u4e00-\u9fa5]{1,2})", text)
        if m:
            aliases.append(m.group(1))
        chars.append(ParsedCharacter(name, aliases, app,
                                     "%s · 出现%d次" % (role, freq), pal))
    # 群体角色 + 身份群体
    for gname, pat, app, pal in _GROUPS + _TITLE_GROUPS:
        if re.search(pat, text) and gname not in [c.name for c in chars]:
            chars.append(ParsedCharacter(gname, [], app, "群演 / 具有身份的角色", list(pal)))
    if not chars:  # 没有任何角色时的兜底
        chars.append(ParsedCharacter(
            "主角", [], "a central figure in period clothing, expressive face, full body",
            "主角", [70, 74, 82]))

    beats: List[ParsedBeat] = []
    for i, b in enumerate(beats_text):
        present = [c.name for c in chars
                   if c.name in b or any(a and a in b for a in c.aliases)]
        # 群体通过正文模式命中
        for gname, pat, _, _ in _GROUPS + _TITLE_GROUPS:
            if re.search(pat, b) and gname not in present:
                present.append(gname)
        emo = next((v for p, v in _EMOTION_BUCKETS if re.search(p, b)), "叙事推进")
        beats.append(ParsedBeat(
            text=b, title=_clean_title(b, i + 1), emotion=emo, characters=present,
            location=_pick(_LOCATION_WORDS, b, "故事场景"),
            time=_pick(_TIME_WORDS, b, "白天"),
            weather=_pick(_WEATHER_WORDS, b, "常态"),
        ))
    # 未识别到中国朝代的故事，去掉外观 token 里写死的"Chinese"，避免误导 T2I
    if era == "a dramatic cinematic setting":
        for c in chars:
            c.appearance = c.appearance.replace("ancient Chinese ", "period-costumed ")
            c.appearance = c.appearance.replace("Chinese ", "")

    first = beats[0].text if beats else text
    return ParsedStory(title=_clean_title(first, 1) + " · 故事分镜",
                       global_style=style, characters=chars, beats=beats,
                       mode="rule", note="规则抽取：姓名字典+群体词+情绪/场景关键词")


# ---------------------------------------------------------------------------
# LLM 抽取（OpenAI 兼容 /chat/completions，stdlib urllib，无新依赖）
# ---------------------------------------------------------------------------
_LLM_SYSTEM = """你是专业影视剧本分析师。把用户给的故事结构化输出为严格 JSON（不要 Markdown 代码块）：
{
  "title": "简短故事标题",
  "global_style": "一段英文视觉风格串，用于文生图，包含时代/地域/色调/镜头语言",
  "characters": [{"name":"角色名","aliases":[],"appearance":"英文外貌描述，稳定可复用","role":"身份","palette":[r,g,b]}],
  "beats": [{"text":"原文节拍","title":"4-12字镜头标题","emotion":"情绪","characters":["角色名"],"location":"地点","time":"时间","weather":"天气"}]
}
要求：beats 4-8 个，按叙事顺序；characters 覆盖所有具名角色和主要群体；appearance 必须具体（年龄/服饰/面部/气质）。
只输出 JSON。"""


def _llm_call(text: str) -> dict:
    base = os.getenv("SB_LLM_BASE", "https://api.openai.com/v1").rstrip("/")
    key = os.getenv("SB_LLM_KEY", "")
    model = os.getenv("SB_LLM_MODEL", "gpt-4o-mini")
    timeout = float(os.getenv("SB_LLM_TIMEOUT", "60"))
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": _LLM_SYSTEM},
                     {"role": "user", "content": text[:12000]}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        base + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
    return json.loads(content)


def parse_llm(text: str) -> ParsedStory:
    """LLM 结构化抽取；任何异常由调用方回退规则版。"""
    d = _llm_call(text)
    chars = []
    for c in d.get("characters", []):
        pal = c.get("palette") or [70, 74, 82]
        chars.append(ParsedCharacter(
            name=str(c.get("name", "角色")),
            aliases=[str(x) for x in c.get("aliases", [])],
            appearance=str(c.get("appearance", "")),
            role=str(c.get("role", "")),
            palette=[float(x) for x in pal[:3]],
        ))
    beats = []
    for b in d.get("beats", []):
        beats.append(ParsedBeat(
            text=str(b.get("text", "")), title=str(b.get("title", "")),
            emotion=str(b.get("emotion", "")),
            characters=[str(x) for x in b.get("characters", [])],
            location=str(b.get("location", "")), time=str(b.get("time", "")),
            weather=str(b.get("weather", "")),
        ))
    return ParsedStory(
        title=str(d.get("title", "故事分镜")),
        global_style=str(d.get("global_style") or _BASE_STYLE % "a cinematic historical setting"),
        characters=chars, beats=beats, mode="llm",
        note="LLM 结构化抽取")


def parse_story(text: str, mode: str = "auto") -> ParsedStory:
    """mode: rule | llm | auto（auto = 有 LLM 配置则先试 LLM，失败回退规则）。"""
    want_llm = mode == "llm" or (mode == "auto" and
                                 bool(os.getenv("SB_LLM_KEY") or os.getenv("SB_LLM_BASE")))
    if want_llm:
        try:
            return parse_llm(text)
        except Exception as e:
            fallback = parse_rule(text)
            fallback.note = "LLM 不可用(%s) → 已回退规则抽取" % str(e)[:90]
            return fallback
    return parse_rule(text)
