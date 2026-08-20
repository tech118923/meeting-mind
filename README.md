# meeting-mind

长会议音频转写工具：把一段会议录音，自动转成**文字转写 + 说话人识别 + 章节摘要 + 会议纪要**。

## 环境要求

- Python ≥ 3.10
- ffmpeg（系统工具，需在 PATH 中）
- 依赖：`pip install -r requirements.txt`

需要设置的环境变量：

| 变量 | 用途 |
|------|------|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key（转写 + 摘要） |
| `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` | 阿里云 OSS 上传凭证 |

## 管道流程（8 阶段）

```
Audio → [1]ffmpeg 分割(自动16kHz单声道) → [2]OSS 上传 → [3]ASR 批量转写(说话人分离+热词)
     → [4]说话人归一化 → [5]合并(去重叠+置信度) → [6]声纹识别(SPK_XX→真实姓名)
     → [7]正则清洗 + 章节检测+子段摘要 → [8]会议纪要(参照章节概要 + RAG+热词)
```

阶段名：`split` / `upload` / `transcribe` / `normalize` / `merge` / `identify` / `chapters` / `minutes`。

- `--start-from X`：从 X 阶段开始，**自动跳过 X 之前的阶段**。
- `--skip-X`：跳过 X 阶段（用于跳过 X 之后想省略的阶段）。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 OSS（复制模板并填入你的 bucket 名）
#    config/oss_config.example.json → config/oss_config.json

# 3. 跑一次（直接用原始音频即可）
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --speaker-count 10 --skip-minutes
```

## 音频放哪

```
meeting_audio/
├── raw_audio/        # 原始会议音频（.m4a / .mp3 / .wav，放这里）
└── processed_audio/  # （可选）已预处理音频，一般不用管
```

直接把会议录音丢进 `raw_audio/`，然后 `--input` 指向它即可。多声道音频不需要手动转单声道，程序会自动处理。

## 说话人显示中文名

转写结果里显示中文名（如「张三」），需要先**注册声纹**：

```bash
# 注册：speakers/ 下建英文文件夹放该人的音频样本，--name 指定中文显示名
python src/enroll_speaker.py --folder "speakers/zhangsan" --name "张三"

# 查看已注册
python src/enroll_speaker.py --list
```

> 目录名用英文拼音（`zhangsan`），显示的中文名靠 `--name` 指定。

## 输出在哪

每次运行后，结果在 `output/<音频文件名>/` 下：

| 文件 | 内容 |
|------|------|
| `transcript.txt` | 文字转写（带时间戳和说话人） |
| `transcript.json` | 结构化转写 |
| `chapter_summaries.md` | 章节摘要 |
| `meeting_minutes_*.md` | 会议纪要 |

## 配置管理（热词 / 知识库）

用网页界面管理热词和知识库（分类、搜索、增删改），不用手改 JSON：

```bash
python webapp/server.py
```

运行后自动打开浏览器（默认 http://127.0.0.1:8787），界面分两个 tab：

- **热词**：管理 ASR 热词（分类/权重/语言），用于转写纠错
- **知识库**：管理 RAG 背景知识和术语（类型/优先级/关键词）

改完点「保存」写回 `config/` 下的 JSON 文件（保存前自动备份 `.bak`）。仅本机访问，不暴露外网；纯 Python 标准库，无第三方依赖。

## 常见问题

- **转写里是 SPK_XX 不是姓名**：没注册声纹，或该说话人没匹配到。注册声纹后再跑 `--start-from identify`。
- **切换音频后结果不对**：清缓存 `rm segments/manifest.json segments/urls.json output/<名>/segments/*.json`。
- **控制台中文乱码**：Windows 显示问题，数据本身正常；加 `PYTHONIOENCODING=utf-8` 运行即可。
- **音频文件名及路径用英文+数字**：不要用中文命名音频文件。

## 附录：常用命令

> 都在项目根目录下运行。

### 管道（pipeline.py）

> 下面命令里的 `xxx.m4a` 换成你的音频文件名；音频统一放在 `meeting_audio/raw_audio/` 下。

```bash
# 全流程（含纪要）
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --speaker-count 10

# 全流程（不生成纪要）
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --speaker-count 10 --skip-minutes

# 只转写（已有上传好的音频）
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --start-from transcribe

# 只做声纹识别（已有转写，把 SPK_XX 换成姓名）
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --start-from identify

# 只生成章节摘要
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --start-from chapters --skip-minutes

# 只生成纪要（已有章节摘要）
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --start-from minutes

# 从某阶段继续（某阶段报错时，日志会提示恢复命令）
python src/pipeline.py --input "meeting_audio/raw_audio/xxx.m4a" --start-from <阶段名>
```

### 声纹注册（enroll_speaker.py）

```bash
python src/enroll_speaker.py --folder "speakers/zhangsan" --name "张三"   # 注册
python src/enroll_speaker.py --list                                     # 查看
python src/enroll_speaker.py --remove "张三"                            # 删除
python src/enroll_speaker.py --threshold 0.78                           # 调匹配阈值
```
