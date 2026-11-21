# StoryBoard Agent · 模块职责与实现说明

> 面向组员的工程说明文档：讲清楚每个模块**做什么、为什么这么做、具体怎么做**。
> 配套快速上手见 `README.md`；本文聚焦"内部原理与模块协作"。
>
> 项目位置：`E:/PyProj/t2i-story-agent`

---

## 0. 一句话定位

把**任意故事文本**自动变成一套可审阅的电影分镜：

```text
故事文本 → 结构化剧本/角色/风格 → 分镜表 → 逐镜出图 → 指标+VLM 验收 → 跨镜一致性检查 → 分镜册
```

核心设计目标有三个：

1. **叙事一致性**：同一角色在不同镜头里长得一样、风格串统一；
2. **完全通用**：不写死任何故事常量，换故事只换输入文本；
3. **始终可跑**：没有 GPU / 模型 / 网络时，用纯 PIL 降级后端也能完整走完三大流程。

---

## 1. 整体架构

```text
                      ┌──────────── Blackboard 共享黑板 ─────────────┐
                      │ title / global_style / characters / scenes   │
                      │ beats / shots / global_check / meta / events  │
                      └──────────────────────────────────────────────┘
                            ▲写           ▲写           ▲写        ▲写
                            │             │             │          │
  .txt 故事                ScriptAgent  StoryboardAgent RenderAgent Reviewer/
    │                      (剧本抽取)     (分镜编排)     (逐镜出图)  Consistency
    ▼                           │             │             │          │
 main.py ──► build_backend ─────┴─────────────┴─────────────┴──────────┘
 (编排)      ├ zimage  本地 GPU 真实出图
             ├ cloud   云侧 HTTP 接口
             └ mock    纯 PIL 降级渲染
                            │
                            ▼
   output/<故事>_<时间>/  shots/*.png  preview.png  contact_sheet.png
                          storyboard.html  storyboard.json  report.md  run.log
```

**流水线分三段**（`main.py` 里用 `console.stage()` 打印）：

- **STAGE 1 预处理**：ScriptAgent + StoryboardAgent
- **STAGE 2 生成**：RenderAgent + 后端
- **STAGE 3 评估**：ReviewerAgent + ConsistencyAgent

五个 Agent **只通过黑板交换状态**，不互相直接调用数据；黑板最终落盘为 `storyboard.json`，任何阶段都可回溯。

---

## 2. 数据与状态层

### 2.1 `story_agent/blackboard.py` — 共享黑板

**职责**：定义系统内所有数据结构，充当唯一的"事实来源"（single source of truth）。

**为什么需要它**：多智能体系统最怕各模块各存一份状态、互相不一致。黑板让每个 Agent 从同一处读、往同一处写，并且可以整体序列化落盘，便于审计和复现。

**核心数据结构**：

| 结构 | 关键字段 | 说明 |
|---|---|---|
| `Character` | `name / aliases / appearance / role / ref_image / palette` | 角色固化后的英文外观 token，跨镜复用；`palette` 是按名字哈希得到的稳定配色 |
| `Scene` | `name / location / time / weather / mood` | 场景的地点/时间/天气/情绪 |
| `Beat` | `beat_id / text / title / emotion / characters / scene` | 叙事节拍，ScriptAgent 从原文抽取 |
| `Shot` | `shot_id / title / shot_size / composition / lighting / camera / characters / emotion / scene_hint / action / prompt / seed / image_path / attempts / review` | 分镜表的一行，贯穿生成与验收全过程 |
| `Blackboard` | 上述集合 + `global_style / global_check / meta / events` | 顶层容器，`events` 记录每个 Agent 的动作日志 |

**做法**：

- `post(agent, key, value)` 写字段并自动记事件；`log(agent, msg)` 追加带时间戳的事件；
- `to_dict()` / `save()` / `load()` 负责 JSON 序列化与反序列化，把 dataclass 展开成字典再还原；
- `load()` 用 `Character(**v)` 等方式精确还原类型，保证读回后仍是 dataclass。

### 2.2 `config.py` — 全局配置

**职责**：集中管理路径、生成参数、验收阈值、可选服务开关。

**关键做法**：

- **私有 vendor 注入**：把项目内 `vendor/`（放 diffusers 0.36.0，首个内置 `ZImagePipeline` 的版本）插到 `sys.path` 最前，不改用户的 conda 环境；
- **权重路径**：`MODEL_DIR` 默认指向 F32 目录，可用环境变量 `SB_MODEL_DIR` 切到 BF16 目录；
- **缓存重定向**：`HF_HOME` / `TORCH_HOME` 等默认写项目 `.cache/`，避免占 C 盘系统盘；
- **生成参数** `GEN`：默认 1024×576（16:9 省显存）、`steps=9`（Turbo 推荐）、`guidance=0.0`（蒸馏模型必须为 0）、`dtype=bfloat16`、`low_cpu_mem_usage=True`（逐层加载 + safetensors mmap，防止 F32 权重打爆内存）；
- **验收阈值** `THRESHOLDS`：`overall=0.60`、`clip=0.40`、`dino=0.42`、`aesthetic=0.42`、`vlm=0.55`、`max_repair=2`；
- **可选服务**：`SB_VLM_*` / `SB_CLOUD_IMAGE_*` 等，无配置时自动降级到规则/代理实现。

> 注：`GLOBAL_STYLE` 是保留的静态风格模板常量；当前**实际生效**的全局风格串由 `script_parser` 按故事题材动态生成后写入黑板，`NEGATIVE` 则用于回填每张图的 `negative_prompt`。

---

## 3. 入口编排层

### 3.1 `main.py` — 主控入口

**职责**：解析参数、读入故事、选择后端、按三阶段调用 Agent、打印总结。

**具体做法**：

1. **参数解析**：`--story`（故事名或 `.txt` 路径）、`--script-mode`（rule/llm/auto）、`--backend`（auto/zimage/mock/cloud）、`--mode`（显存策略）、`--shots`、`--size`、`--steps`、`--no-ref`；
2. **故事读入**：如果 `--story` 是存在的 `.txt` 路径就直接读，否则当作 `stories/` 下的故事名；输出目录名只取文件 stem 并过滤非法字符，避免绝对路径里的盘符/分隔符造成 Windows 非法目录；
3. **环境探测**：Stage 0 打印权重检查结果、GPU 型号/显存/CUDA/torch 版本，然后 `build_backend()` 选后端；Z-Image 时把设备与模式写入 `bb.meta`；
4. **三阶段执行**：依次调用 ScriptAgent → StoryboardAgent → RenderAgent（逐镜循环）→ ReviewerAgent → ConsistencyAgent；
5. **总结与回显**：打印总耗时、显存峰值、通过镜数、各产物路径；**最后把输入的故事全文再打印一遍**（并追加到 `run.log`），方便核对输入。

**异常处理**：`KeyboardInterrupt` 返回 130，其他异常打印 traceback 返回 1，便于批处理判断成败。

### 3.2 `story_agent/console.py` — 控制台与日志

**职责**：统一输出样式，同时写屏幕和 `run.log`。

**做法**：`_emit()` 先 `print(flush=True)`，再以追加模式写日志文件（写的是去掉 ANSI 颜色的纯文本）；提供 `banner / stage / agent / info / ok / warn / err / kv / table` 等语义化函数，让整份日志结构清晰、可直接发人。

---

## 4. 预处理层：故事 → 剧本 → 分镜表

### 4.1 `story_agent/script_parser.py` — 通用故事解析器

**职责**：把任意故事文本解析成结构化的 `ParsedStory`（角色、节拍、风格），供 ScriptAgent 消费。**这是"去写死化"的核心模块。**

**对外接口**：`parse_story(text, mode) -> ParsedStory`，`mode` 为 `rule` / `llm` / `auto`。

#### rule 模式（默认，零依赖）

**原理**：纯中文文本规则 + 词频统计，不依赖 LLM 或网络。

**具体做法**：

- **节拍切分** `_split_beats`：优先按空行分段；只有一段时按句末标点切；超过上限就均分合并，不足 4 拍就再按句细切，最终保证 4–8 拍；
- **人名识别** `_detect_names`：`百家姓` + 21 个复姓构成正则，单姓取 2 字、复姓取 2–3 字（避免"周瑜欲"这种把语法字吞进去的误切）；用"说话人 + 曰/说 + 引号"加权；命中停用词表（地名、身份、常见词如"王侯/白帝/都督"）则排除；`_NAME_TAIL` 纠正"陈胜王→陈胜"这类尾字误切；`字X` 别名归一避免同一人被识别成两个角色；最终取出现次数 ≥2 或明确说话人的候选，最多 8 个；
- **群体角色**：`戍卒/众人/将士` 及"校尉/军官"等身份群体用独立词表识别，避免把主角误判成士兵；
- **外观合成** `_appearance`：按名字**局部窗口**（前后各十几字）匹配身份线索（僧/书生/将军/王侯/女子/老者/农夫/士兵等），合成稳定的英文外观 token；年龄、性别、语气分别由关键词决定；配色按名字哈希计算，保证跨镜一致；
  - 关键修复：身份线索只看名字**局部窗口**，不再扫整段 context，否则"军令状/伏兵"里的"军/士"会把诸葛亮、周瑜全判成士兵；
- **场景/情绪**：地点、时间、天气、情绪各有词表动态命中；标题优先取引号内台词，否则取首句，截成 4–12 字；
- **时代风格**：从文本前 1200 字识别秦/三国/汉/唐/宋/明/清，生成对应英文风格串；未识别到中国朝代时，去掉外观 token 里写死的 "Chinese"，避免误导 T2I。

#### llm 模式（可选）

**原理**：调用任意 OpenAI 兼容的 `/chat/completions`，用固定 JSON schema 让 LLM 做结构化抽取。用标准库 `urllib` 实现，不引入新依赖。

**做法**：system prompt 明确要求输出 `title / global_style / characters / beats` 的严格 JSON；`temperature=0.2`、`response_format=json_object`；对返回的 Markdown 代码块标记做清洗后再 `json.loads`。

**回退机制**：`auto` 模式下若有 `SB_LLM_KEY` 或 `SB_LLM_BASE` 就先试 LLM，**任何异常都会回退 rule** 并在 `ParsedStory.note` 里写明原因，绝不中断流程。

### 4.2 `ScriptAgent` — 剧本 Agent

**职责**：把 `parse_story` 的结果搬进黑板并固化。

**做法**：解析后，把角色（含外观 token 与配色）写入 `bb.characters`；从节拍地点/时间/天气/情绪动态生成 `bb.scenes`；把节拍转成 `Beat` 列表写入；把风格串和 `config.NEGATIVE` 写入黑板。全程通过 `bb.post()`，每个动作都记事件日志。**Agent 内不含任何故事专属常量。**

### 4.3 `StoryboardAgent` — 分镜编排 Agent

**职责**：把节拍表展开成 4–8 镜的分镜表，给每镜定景别、构图、光线、运镜和 seed。

**做法**：

- 镜数 `n` 限制在 4–8；节拍多于镜数就按序均分合并，少于就一拍一镜；
- **镜头语言由文本关键词动态推导**（`_plan_shot`）：
  - 景别 `_SIZE_RULES`：出现"手/指/剑柄/眼/灯"等取特写，"众/群/振臂/高呼"取全景，"低声/密谋/对话"取近景，"远望/山河/旷野"取远景，否则按 `中景→全景→近景→特写→远景` 轮转；
  - 构图 `_COMP_BY_SIZE`：按景别映射英文构图描述（三分法、低角度英雄构图、浅景深等）；
  - 光线 `_LIGHT_RULES`：雨→冷蓝，夜/烛→烛火明暗，破晓→金色逆光，火→橙光，雪→冷白等；
  - 运镜 `_CAM_RULES`：奔/杀/冲→手持跟拍，呼/喊/振臂→缓慢升镜，密谋/思→固定机位缓推，远望/开拔→横移；
- **情节串 `_action_text`**：按句末标点切分，保留**前 3 句**（180 字内），去掉中英文引号，避免把一句台词切成半截。这是后来按"分镜要生成情节内容"的要求改的——**情节是 prompt 主体**；
- seed 按 `config.seed + 镜头序号×101` 分配，保证可复现且各镜不同。

---

## 5. 生成层：prompt → 图像

### 5.1 `story_agent/backends.py` — 三种出图后端

**职责**：把"生成一张图"这件事抽象成统一接口 `generate(prompt, seed, shot) -> (Image, info)`，上层不关心底层是 GPU 还是 PIL。

#### ZImageBackend（本地真实 GPU）

**原理**：用 diffusers 的 `ZImagePipeline` 加载通义 Z-Image-Turbo（6B DiT + Qwen3 文本编码器），在 GPU 上做 9 步去噪。

**做法**：

- `available()` 做完整性检查：核对 `_REQUIRED` 配置文件和 7 个 `_WEIGHT_FILES` 分片，统计权重体积；**关键守卫**：若权重 >28GB（F32）且系统内存 <24GB，直接判定不可用并给出"转 BF16 + sequential"的修复指引——因为这组合会在加载 DiT 分片时 native 层 SIGSEGV，Python 的 try/except 抓不到；
- `load()`：`from_pretrained(torch_dtype=bfloat16, low_cpu_mem_usage=True)`，失败时回退 `low_cpu_mem_usage=False`；随后按 `mode` 设置设备策略：
  - `full_gpu`：全量驻留，20.5GB → 16GB 卡 OOM；
  - `offload`（`enable_model_cpu_offload`）：整模块换入，DiT 12GB 一次到位 → 碎片化 OOM；
  - `sequential`（`enable_sequential_cpu_offload`）：**逐 transformer block 换入，峰值仅 0.78GB**，是 16GB 卡唯一稳定方案；
  - `manual`：绕过 accelerate hook，text_encoder 留 CPU、transformer+vae 放 GPU（实测仍 OOM）；
  - `cpu`：纯 CPU，仅作对照；
- `generate()`：用 **CPU generator** 固定 seed（避免 offload 模式下 generator/张量设备冲突）；生成前 `reset_peak_memory_stats()`，生成后从 `max_memory_allocated()` 读真实峰值显存写入 `info`；`manual` 模式额外在 CPU 预计算 prompt embedding 再搬到 GPU。

#### MockBackend（纯 PIL 降级）

**职责**：无权重、无网络时保证流程完整可跑（冒烟测试）。

**做法**：按镜头序号取 8 套配色，用 numpy 画天空渐变和主光源辉光，再叠远山剪影、雨丝、人物剪影（数量按景别）、旗帜/火把、雾/暗角/颗粒，最后盖一条分镜信息条。**0 秒出图、0 显存**，但画面是程序化概念图，不是神经网络生成。

#### CloudBackend（云侧接口）

**职责**：调用 OpenAI 兼容的 `images/generations` 接口。做法：从环境变量读 URL/Key/Model，`requests.post` 传 prompt 与 seed，返回 base64 或 URL 图像。**构造函数不导入 requests**，避免污染 rule+mock 的最小路径。

#### build_backend（选择策略）

`auto` 模式按 **Z-Image → 云侧 → mock** 依次降级；强制 `zimage` 但不可用时直接抛异常，让问题尽早暴露。

### 5.2 `RenderAgent` — 渲染 Agent

**职责**：把分镜拼成 prompt，生成/复用角色参考图，逐镜调用后端出图。

#### prompt 结构（重点）

prompt 的主体是**这一镜的情节**，角色外观只是后置的一致性锚点：

```text
Cinematic film still depicting: <本镜情节 action>
Setting: <场景>
On screen: <出场角色外观，只取 appearance 前 3 段>
Shot: <景别>, <运镜>, <构图>
<光线描述>
Mood: <情绪>
<全局风格串>
```

这是按"分镜要生成情节内容，不是单一角色"的要求重构的：角色外观只取身份/服饰/面部三段放 `On screen:`，避免整段角色卡把情节挤到后面。中文情节由 Z-Image 的 Qwen3 文本编码器理解，英文片段作一致性锚点。

#### 角色参考图

- ZImage 后端：按出场镜头数排序取前 3 个角色，拼一张 "character reference sheet" 生成真实参考图；
- 生成失败（如 OOM）就降级为"外观色卡"（按 `palette` 画的色块 + 角色名）；
- **色卡模式会把 `self.ref` 设为 None**，不参与逐镜像素级一致性评分——因为纯色卡和剧情画面的直方图没有可比性，硬比会得到无意义的低分。

#### render_one

拼 prompt → 算 seed（`s.seed + (attempt-1)*7919`，重画自动换 seed）→ 调后端 → 按镜号/标题命名存到 `shots/`（重画加 `_rN` 后缀）→ 把图片路径、prompt、后端、耗时回填到 `Shot`。

---

## 6. 评估层：验收 → 修复 → 一致性 → 分镜册

### 6.1 `story_agent/metrics.py` — 视觉指标

**职责**：从图像像素真实计算分数；没有深度模型权重时用**可解释的代理实现**并显式标注 `proxy`，绝不伪造满分。

| 指标 | 真实实现 | 代理实现（默认） |
|---|---|---|
| CLIPScore | `SB_REAL_CLIP=1` 时加载 CLIP 模型算图文余弦相似度 | 从 prompt 视觉线索（雨/夜/火/金色等）与图像色调/曝光的吻合度 + prompt token 覆盖率算分，上限 0.96 |
| 角色一致性 | 可替换真实 DINOv2 | 角色参考图 vs 镜头图的 HSV 直方图相似度（0.88 权重）+ dHash 结构弱佐证（0.12）；无参考图返回 `no-ref` |
| 审美分 | — | 曝光/对比/色调层次/细节/色彩数五项加权 |
| VLM 检查 | 配 `SB_VLM_ENDPOINT/KEY` 后调多模态模型给 plot/composition/integrity 分 | 规则式：情节关键词→必需视觉线索、构图偏离度、崩坏检测（纯色/极曝/平图/糊图） |

**基础统计** `image_stats`：亮度、对比度、饱和度、熵（色调层次）、锐度（Laplacian 或 numpy 近似）、色彩桶数；opencv 可选，没装也有 numpy 回退，保证最小依赖可跑。

**修复** `repair_image`：针对 `too_dark/flat_image/blurry` 做局部提亮、提对比、锐化，不重画。

**汇总** `evaluate_shot`：权重 `CLIP 0.30 / 一致性 0.15 / 审美 0.25 / VLM 0.30`；无参考图时重新分配为 `0.34 / 0 / 0.29 / 0.37`。总分达 `overall` 且四项都不低于各自阈值才 PASS；决策分 `pass / local_repair / repaint`。

### 6.2 `ReviewerAgent` — 视觉验收 Agent

**职责**：逐镜验收，不达标就局部修复或重画。

**做法**：对每镜最多 `max_repair=2` 次循环：调 `evaluate_shot` → 打印各项分数与模式 → 若 PASS 跳出；若达到最大次数则 `give_up` 保留当前结果；若问题全是曝光/对比/锐度类（`local_repair`）就调用 `repair_image` 原地修复；否则调用 `render_one(attempt+1)` 换 seed 重画。所有结果写入 `Shot.review` 并打表输出。

### 6.3 `ConsistencyAgent` — 全局一致性 + 产物 Agent

**职责**：跨镜校验一致性，并生成全部交付产物。

**做法**：

- **角色 token 复用率**：对每个角色，检查其外观锚点（appearance 前 3 段）是否出现在所有出场镜头的 prompt 里——与 RenderAgent 的精简锚点保持一致，避免假性归零；
- **风格串一致率**：`global_style` 出现在多少镜的 prompt 中；
- **跨镜色彩一致性**：所有镜头两两计算 HSV 直方图相似度，取均值；
- 把结果写入 `bb.global_check`，并记录情绪曲线、通过镜数、后端/设备；
- **生成产物**：
  - `contact_sheet.png`：2 列网格总览图；
  - `preview.png`：**预览长图**，按镜号依次绘制"标题条 → 景别/情绪/角色标签 → 画面描述 → 分镜图"，方便直接滚动审阅；
  - `storyboard.html`：分镜册，每镜一张卡片含图、镜头信息、指标条、完整 prompt，末尾附输入故事全文；
  - `report.md`：运行报告（环境、流程、逐镜验收表、全局校验、产物清单）；
  - `storyboard.json`：黑板快照，供复现/审计。

---

## 7. 工具层 `tools/`

| 脚本 | 职责 | 关键做法 |
|---|---|---|
| `bootstrap_diffusers.py` | 提供 ZImagePipeline 所需的 diffusers 0.36.0 | 从华为/阿里/PyPI 三镜像下载 4.6MB wheel，校验大小后解压到私有 `vendor/`，不碰 conda 环境 |
| `fetch_model.py` | 下载 F32 权重 | ModelScope CDN，7 个大文件 4 线程 + 小文件串行，`.part` 断点 + 重试 |
| `fetch_model_fast.py` | 加速下载 | CDN 单连接限速约 3MB/s，对同一文件开 8 个 256MB Range 连接叠加带宽，支持从已有 `.part` 续传 |
| `convert_bf16.py` | F32 → BF16 预转换 | 用 safetensors `safe_open` **逐张量流式读写**（峰值内存≈单分片），只转 float32 张量、整型原样保留；跳过已转换文件；修正索引 `total_size`。32.83GB → 20.52GB，加载 33s → 0.8s |
| `check_gpu.py` | GPU 可行性验证 | 先静态核算（权重体积 vs 显存），再实测加载+生成 1 张，用 `torch.cuda.memory_allocated` 和峰值显存证明确实走 GPU；支持 `--mode sequential --size` |

---

## 8. 关键技术决策与原理

### 8.1 为什么 prompt 以情节为主体

早期 prompt 以角色外观卡开头，情节被挤到后面，生成的是"人物展示"而不是"剧情画面"。现改为 `Cinematic film still depicting: <情节>` 开头，角色外观后置且精简。角色抽取保留下来不是为了分词，而是给纯 T2I 的 Z-Image 提供跨镜一致性锚点（它没有原生 IP-Adapter）。

### 8.2 一致性如何保证

Z-Image-Turbo 是纯 T2I，没有身份锁定机制。系统用三招组合：① 固化角色外观 token 并逐镜复用；② 固定 seed 便于复现；③ 生成角色参考图/色卡供验收比对。这是"近似一致性"，不是像素级身份锁定。

### 8.3 16GB 显卡的显存账（重点）

BF16 权重实测分布：

```text
transformer (DiT)  12 GB   ← 去噪主模块
text_encoder      7.5 GB
vae               0.16 GB
合计             ≈20.5 GB   ← 标称 16GB 卡装不下
```

各模式实测结论：

| 模式 | 机制 | 峰值显存 | 结果 |
|---|---|---|---|
| `full_gpu` | 全部常驻 GPU | ~20.5 GB | 直接 OOM |
| `offload` | 整个 DiT 一次换入 GPU | ~13–14 GB | 实测 OOM：15.93GiB 卡上 PyTorch 占 8.60GiB、仅剩 5.94GiB，连 742MiB 都分配不出（碎片化） |
| `sequential` | 逐 transformer block 换入 | **0.78 GB** | 稳定出图，代价是每块都要过 PCIe，单张 512×288 约 2.5–5 分钟 |
| `cpu` | 纯 CPU | 0 | 极慢，仅对照 |

**为什么 offload 装不下**：12GB 不是稳定占用，而是必须一次性拿出的连续显存块；加上激活值、CUDA context 和 allocator 碎片，瞬时峰值逼近 14GB。而 sequential 把 DiT 拆成几十个 block 逐个算，峰值只有 0.78GB——这就是它在 16GB 卡上唯一稳定的原因，也是它慢的本质。F32 权重（DiT 分片 24.6GB）更糟，系统内存仅 17GB，加载阶段就 SIGSEGV，所以必须先转 BF16。

### 8.4 指标为什么分"真实"与"proxy"

CLIP / DINO / VLM 的真实模型都要额外下载权重或配置云接口。为了让最小路径零依赖，默认用可解释的启发式代理，并在日志里标注 `proxy` / `rule-based`，分数**不是写死的**。配置对应环境变量后即可切到真实模型。已知局限：代理一致性指标会把"参考板 vs 剧情画面"的构图差异误判为角色不一致（曾导致火烧连营 SHOT 03 被 `give_up`），接真实 DINOv2/VLM 可缓解。

---

## 9. 产物清单

运行后在 `output/<故事名>_<时间>/` 得到：

| 文件 | 内容 |
|---|---|
| `shots/*.png` | 逐镜画面（重画带 `_rN` 后缀） |
| `char_ref.png` | 角色参考图，或降级外观色卡 |
| `preview.png` | 预览长图（描述 + 图 + 标题） |
| `contact_sheet.png` | 分镜总览网格图 |
| `storyboard.html` | 分镜册（含完整 prompt 与指标） |
| `storyboard.json` | 共享黑板快照 |
| `report.md` | 运行报告 |
| `run.log` | 控制台日志 |

---

## 10. 运行方法速查

### 10.1 最小路径（rule + mock，零依赖）

```bash
python -m pip install -r requirements.txt
python main.py --story chensheng --script-mode rule --backend mock --shots 4 --size 512x288
# 任意文本文件（相对/绝对均可）
python main.py --story D:/stories/my_story.txt --script-mode rule --backend mock
```

### 10.2 可选 LLM 预处理

```bash
set SB_LLM_BASE=https://your-endpoint/v1
set SB_LLM_KEY=sk-xxxx
set SB_LLM_MODEL=qwen-plus
python main.py --story D:/stories/my_story.txt --script-mode llm --backend mock
```

### 10.3 真实 Z-Image（BF16 + sequential，16GB 卡）

```bat
set PY=E:\Anaconda\envs\showo2\python.exe
set SB_MODEL_DIR=E:\PyProj\t2i-story-agent\models\Z-Image-Turbo-bf16
%PY% main.py --story huoshao --script-mode rule --backend zimage --mode sequential --shots 7 --size 512x288
```

> **切记**：只写 `--backend zimage` 而不设 `SB_MODEL_DIR` / `--mode sequential`，会默认加载 F32 32.83GB + `cpu_offload`，启动时就会被守卫拦下并打印修复提示。

### 10.4 参数一览

| 参数 | 说明 |
|---|---|
| `--story NAME\|路径.txt` | 内置故事名或任意 `.txt` 路径 |
| `--script-mode rule\|llm\|auto` | 剧本/分镜来源 |
| `--backend auto\|zimage\|mock\|cloud` | 出图后端 |
| `--mode auto\|offload\|sequential\|full_gpu\|cpu\|manual` | Z-Image 显存策略 |
| `--shots 4..8` | 分镜数 |
| `--size 宽x高` | 分辨率，建议 16 的倍数 |
| `--steps N` | 推理步数，Turbo 推荐 9 |
| `--no-ref` | 跳过角色参考图，少一次模型调用 |

---

## 11. 已知限制

- Z-Image-Turbo 无原生身份锁定，角色一致性靠 token 复用 + 固定 seed + 参考图验收近似实现；
- 代理一致性指标存在误判，建议接真实 DINOv2 / VLM；
- rule 版人名基于姓氏字典 + 词频，生僻复姓或只出现一次的角色可能漏抽，可用 `--script-mode llm`；
- sequential 模式在 16GB 卡上稳定但慢；要真正提速需 FP8/GGUF 量化 DiT（可整体驻留 GPU）、换 24GB+ 显卡、换小模型或走云侧接口；
- 预览与 HTML 依赖中文字体，缺失时自动回退 PIL 默认字体，不影响流程。
