# meeting-mind — 项目全貌（给 AI/开发者）

长会议音频转写全套方案：ASR 转写 + 说话人分离 + 声纹注册/识别 + 热词增强 + RAG 知识库 + 章节摘要 + 关键人物专项分析 + 会议纪要。

> 面向使用者的精简版见 [README.md](README.md)。
> 注：本仓库为**脱敏后的公共版本**，config/ 下数据与 prompt 均为示例内容，请按实际业务替换。

## 目录结构

```
meeting-mind/
├── src/                          # 全部源码
│   ├── pipeline.py               # 一键总控（8 阶段）
│   ├── audio_splitter.py         # ffmpeg 音频分割（自动转 16kHz 单声道）
│   ├── oss_uploader.py           # 阿里云 OSS 上传，生成签名 URL
│   ├── batch_transcribe.py       # 非实时 ASR 批量转写（qwen-audio-3.0-asr-flash-filetrans）
│   ├── speaker_normalizer.py     # 跨段说话人归一化（Union-Find 传递闭包）
│   ├── merge_transcript.py       # 合并转写 + 输出多种格式 + 置信度标记
│   ├── speaker_recognizer.py     # 声纹引擎（CAM++）：库管理 + 说话人重识别
│   ├── enroll_speaker.py         # 声纹注册 CLI
│   ├── summarizer.py             # 章节检测 + 子段拆分 + 差异化摘要 + 关键人物分析
│   ├── meeting_summarizer.py     # 会议纪要生成（参照章节概要 + RAG + 热词）
│   ├── transcript_cleaner.py     # 正则口语清洗
│   └── knowledge.py              # RAG 知识库加载 + 检索
├── webapp/                       # 配置管理 Web 工具（纯标准库，零依赖）
│   ├── server.py                 # 后端：静态文件 + hotwords/knowledge 读写 API
│   └── static/index.html         # 前端：热词/知识库两个 tab，分类+搜索+增删改
├── config/                       # 配置与提示词（示例内容，按需替换）
│   ├── oss_config.json           # OSS 连接信息（AK/SK 走环境变量，不入库）
│   ├── oss_config.example.json   # OSS 配置模板（无真实凭证，可入库）
│   ├── hotwords.json             # ASR 热词库（示例）
│   ├── knowledge_base.json       # RAG 知识库（示例）
│   ├── prompt_chapter.md         # 章节子段摘要 prompt
│   ├── prompt_leader.md          # 关键人物发言专项提取 prompt
│   └── prompt_template.md        # 会议纪要 prompt（基于章节概要）
├── speakers/                     # 说话人声纹库 + 注册样本（不入库，个人数据）
├── meeting_audio/                # 会议音频（不入库）
│   ├── raw_audio/                # 原始会议音频（.m4a 等）
│   └── processed_audio/          # （可选）已预处理音频，管道会自动转 16kHz 单声道
├── segments/                     # 临时分段缓存（可删，运行时重建）
├── output/                       # 每场会议独立子文件夹
├── requirements.txt
└── .gitignore                    # 忽略音频/分段/输出/声纹库/oss 配置
```

## 环境

- Python ≥ 3.10（conda 或系统 python，依赖见 requirements.txt）
- 关键包：dashscope、oss2、funasr、numpy、torch
- ffmpeg：系统工具，须在 PATH 中
- 环境变量：`DASHSCOPE_API_KEY`、`OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`、（可选）`DASHSCOPE_WORKSPACE_ID`

### 运行

```powershell
# PowerShell
$env:PATH = "<ffmpeg安装目录>\bin;" + $env:PATH
cd <项目根目录>
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --speaker-count 10 --skip-minutes
```

```bash
# Git Bash
export PATH="/<ffmpeg安装目录>/bin:$PATH"
cd <项目根目录>
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --speaker-count 10 --skip-minutes
```

## 管道流程（8 阶段）

```
Audio → [1]ffmpeg 分割(自动16kHz单声道) → [2]OSS 上传 → [3]ASR 批量转写(说话人分离+热词)
     → [4]说话人归一化 → [5]合并(去重叠+置信度) → [6]声纹识别(SPK_XX→真实姓名)
     → [7]正则清洗 + 章节检测+子段摘要 → [8]会议纪要(参照章节概要 + RAG+热词)
```

阶段名：`split` / `upload` / `transcribe` / `normalize` / `merge` / `identify` / `chapters` / `minutes`。

- `--start-from X`：从 X 阶段开始，**自动跳过 X 之前的阶段**。
- `--skip-X`：跳过 X 阶段（用于跳过 X 之后想省略的阶段）。

## ASR 模型

- 非实时转写模型：`qwen-audio-3.0-asr-flash-filetrans`（阿里云百炼录音文件识别）
- 支持说话人分离、热词增强、长音频（≤12h，单文件 ≤2GB，仅公网 URL 输入）
- 说话人分离仅支持单声道 → 分割阶段自动转 16kHz 单声道
- 热词表**用完即删**：每次运行自动创建 → 转写 → 删除，避免占满每账号 10 个热词表配额

## 声纹系统

### 模型

CAM++（`iic/speech_campplus_sv_zh-cn_16k-common`），192 维声纹向量，`funasr.AutoModel` 本地加载。

### 注册

```bash
# 单文件夹注册（文件夹名=拼音，--name 指定中文显示名）
python src/enroll_speaker.py --folder "speakers/zhangsan" --name "张三"

# 批量注册（会用目录英文名当姓名，慎用）
python src/enroll_speaker.py --batch "speakers"

# 查看/删除/调阈值
python src/enroll_speaker.py --list
python src/enroll_speaker.py --remove "张三"
python src/enroll_speaker.py --threshold 0.78
```

### 识别流程（pipeline 阶段 6）

1. 加载合并后 transcript.json 的 sentences，按 speaker_id 分组
2. 过滤 < 2000ms 单段、累计 < 5000ms 的说话人（MIN_TOTAL_MS=5000）
3. ffmpeg 切片段 → CAM++ 提取 embedding → 平均 → L2 归一化
4. 与声纹库余弦相似度匹配，阈值 0.78 以上分配真实姓名
5. 重写所有输出文件中的 speaker 标签

### 关键设计决策

- **后置识别**：不改变云端 API 调用，在 merge 之后做声纹匹配。原因：DashScope ASR API 不返回 speaker embedding。
- **中文路径处理**：MinGW ffmpeg 无法处理非 ASCII 路径 → `SpeakerRecognizer._get_safe_audio_path()` 自动复制到 ASCII 临时文件（识别结束后 `cleanup()` 清理）。
- **阈值选 0.78**：实测调优，平衡误判与漏判。
- **声纹库格式**：JSON + base64 编码 numpy float32 embedding，可读可扩展。

## 章节摘要系统

### 两层检测

1. **粗粒度章节检测**（`ChapterDetector.detect`）：滑动窗口 + LLM 判断话题切换点
2. **子段拆分**（`ChapterDetector.detect_sub_segments`）：每章内 LLM 识别 2-5 个语义转折，标注类型

### 差异化摘要策略（`ChapterSummarizer`）

| 类型 | 模型 | 策略 |
|------|------|------|
| `report`（单人汇报） | qwen-max | 简略，2-3 句提炼核心数据和结论 |
| `discussion`（多人讨论） | qwen-max | **详细**，保留各发言人观点/论据/分歧/共识，不压缩 |
| `leader`（关键人物发言） | qwen-max + 专用 prompt | 引用原话 → 灵活标注含义，不强分类、不凑字段 |
| `transition`（过渡） | qwen-max | 一句话概括 |

### 关键人物特殊处理（`prompt_leader.md`）

- 引用驱动：先摘原文 → 再标注含义，原文在场防止 LLM 编造
- 灵活标注：只有一层就标一层，兼有多层就自然描述，不做强制分类

### 置信度标记（`merge_transcript.py`）

- 从 ASR 词级 confidence 计算句级平均
- 低于 0.5 的句子在 transcript.txt 中加 `[⚠低置信]` 前缀
- 摘要 LLM 被告知该段质量低，不要依赖细节

## 会议纪要（阶段 8）

- 输入主体 = **章节概要**（chapter_summaries.md，干净浓缩，避免原始转写无效信息稀释注意力）
- RAG 检索 = **完整清洗转写**（transcript_clean.txt，背景知识命中更全，`summarize(rag_text=...)` 传入）
- 生成失败抛错（非零退出）；章节概要为空/缺失也抛错

## 命名规范

| 对象 | 规则 | 示例 |
|------|------|------|
| 注册人文件夹 | pinyin，全小写 | `zhangsan/`、`lisi/` |
| 注册音频文件 | `sample_01.wav` 递增 | `sample_01.wav` |
| 待转写音频 | 英文 + 数字 | `meeting_20260713.m4a` |
| 显示姓名 | `--name` 指定中文 | `--name "张三"` |

## 已知问题 & 注意事项

1. **文件名必须纯 ASCII**：Git Bash → Python sys.argv 传参时中文路径编码会损坏，音频/文件夹名用英文+数字。
2. **切换音频后清缓存**：`rm segments/manifest.json segments/urls.json output/{name}/segments/*.json`。
3. **声纹识别需先注册**：voiceprint_profiles.json 为空时识别阶段自动跳过，transcript 里是 SPK_XX。
4. **子段检测偶有 JSON 解析失败**：LLM 不输出 JSON 而是分析文本，会兜底为空摘要。
5. **热词 prefix 限制**：DashScope API 要求 prefix ≤10 字符、仅英文+数字，当前用 `mrec`。

## 常用命令

```bash
# 全流程（跳过纪要）
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --speaker-count 10 --skip-minutes

# 仅转写（已有 URL）
python src/pipeline.py --input meeting.mp3 --start-from transcribe

# 仅后处理（已有转写，从合并开始）
python src/pipeline.py --input meeting.mp3 --start-from merge

# 仅声纹识别（已有合并转写）
python src/pipeline.py --input meeting.mp3 --start-from identify

# 仅章节概要
python src/pipeline.py --input meeting.mp3 --start-from chapters --skip-minutes

# 仅纪要（已有章节概要）
python src/pipeline.py --input meeting.mp3 --start-from minutes
```
