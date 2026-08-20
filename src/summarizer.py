"""
会议摘要模块：章节检测 + 章节摘要 + 关键人物发言专项分析。

用法：
    from summarizer import ChapterDetector, ChapterSummarizer

    detector = ChapterDetector()
    chapters = detector.detect(transcript, sentences)

    summarizer = ChapterSummarizer()
    results = summarizer.summarize_all(chapters, transcript, speaker_names)
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import dashscope
from knowledge import load_knowledge_base, retrieve_context

# ── 配置 ──────────────────────────────────────────────

MODEL_CHAPTER_DETECT = 'qwen-plus'   # 章节检测：轻量
MODEL_CHAPTER_SUMMARY = 'qwen-max'   # 章节摘要 + 关键人物分析：深度推理
MAX_CHARS_PER_WINDOW = 4000          # 滑动窗口大小（字符），约 3-5 分钟口语
MAX_CHARS_PER_CHAPTER = 15000        # 单章最大送入字符

LEADER_NAMES = {'张三'}  # 关键人物姓名集合，可扩展


# ── Prompt 模板加载 ─────────────────────────────────

def _load_prompt(name: str) -> str:
    """加载 config/ 下的 prompt 模板文件。"""
    path = os.path.join(PROJECT_ROOT, 'config', f'prompt_{name}.md')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    print(f'[警告] prompt 模板不存在: {path}')
    return ''


def _call_llm(prompt: str, model: str = MODEL_CHAPTER_SUMMARY,
              expect_json: bool = True) -> str | None:
    """统一 LLM 调用，返回文本内容。失败返回 None。"""
    if not dashscope.api_key:
        dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not dashscope.api_key:
            print('[LLM] 未设置 DASHSCOPE_API_KEY')
            return None

    kwargs: dict = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
    }
    if expect_json:
        kwargs['result_format'] = 'message'

    try:
        response = dashscope.Generation.call(**kwargs)
        if response.status_code != 200:
            print(f'[LLM] 调用失败: {response.code} {response.message}')
            return None
        if expect_json:
            return response.output.choices[0].message.content
        else:
            return response.output.text or response.output.choices[0].message.content
    except Exception as e:
        print(f'[LLM] 异常: {e}')
        return None


def _parse_json_safe(text: str) -> dict:
    """安全解析 LLM 返回的 JSON，剥离可能的 markdown 代码块包装。"""
    text = text.strip()
    # 移除 ```json ... ``` 包装
    if text.startswith('```'):
        end = text.rfind('```')
        if end > 0:
            text = text[text.index('\n') + 1:end].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试修复尾逗号等常见小问题
        try:
            import re
            fixed = re.sub(r',\s*}', '}', text)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            print(f'[JSON] 解析失败: {text[:200]}...')
            return {}


# ── 章节检测 ────────────────────────────────────────

class ChapterDetector:
    """基于滑动窗口 + LLM 的章节边界检测。

    不再用脆弱的字符串匹配，而是让 LLM 判断每个时间窗口处是否有话题切换。
    """

    def __init__(self, window_chars: int = MAX_CHARS_PER_WINDOW):
        self.window_chars = window_chars

    def detect(self, transcript: str, sentences: list[dict] = None) -> list[dict]:
        """检测章节边界。

        Args:
            transcript: 完整转写文本 [HH:MM:SS] Name: text 格式
            sentences: sentence 列表（可选，用于获取时间戳）

        Returns:
            [{title, start_ms, end_ms, start_line, end_line}]
        """
        lines = transcript.strip().split('\n')
        if len(lines) < 5:
            return self._single_chapter(lines, sentences)

        # 策略：按行数滑动而非字符数，保持行完整
        total_lines = len(lines)
        # 估算每 30 行约 3-5 分钟内容
        step = max(10, min(40, total_lines // 5))

        boundaries = [0]  # 行索引边界
        prev_summaries = []

        for i in range(step, total_lines - step, step):
            # 取当前窗口前后的文本
            before = '\n'.join(lines[max(0, i - step):i])
            after = '\n'.join(lines[i:i + step])

            is_boundary, brief = self._check_boundary(before, after, prev_summaries)
            if is_boundary:
                boundaries.append(i)
                prev_summaries.append(brief)

        boundaries.append(total_lines)

        # 合并过近的边界（< 10 行）
        merged = [boundaries[0]]
        for b in boundaries[1:]:
            if b - merged[-1] < 10:
                merged[-1] = b
            else:
                merged.append(b)

        # 构建章节
        chapters = []
        for idx in range(len(merged) - 1):
            start_line = merged[idx]
            end_line = merged[idx + 1]
            chapter_lines = lines[start_line:end_line]
            chapter_text = '\n'.join(chapter_lines)

            # 提取时间范围
            start_ms, end_ms = self._get_time_range(
                chapter_lines, sentences)

            # 生成标题
            title = self._generate_title(chapter_text)

            chapters.append({
                'index': idx,
                'title': title,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'start_line': start_line,
                'end_line': end_line,
                'text': chapter_text,
            })

        return chapters

    def _check_boundary(self, before: str, after: str,
                        prev: list[str]) -> tuple[bool, str]:
        """调用 LLM 判断两段文本之间是否是话题切换点。"""
        prompt = f"""你是会议分析助手。判断以下两段会议内容之间是否是话题切换点。

话题切换的特征：
- 讨论的具体事务发生变化（如从"数据分析"切换到"云平台部署"）
- 发言人明确引出新议题
- 汇报人发生变化且内容无关联

不是话题切换的特征：
- 同一议题下的补充讨论
- 不同人对同一话题的回应
- 语气词、过渡语

前文：
{before[:2000]}

后文：
{after[:2000]}

此前已识别的话题：{'; '.join(prev[-3:]) if prev else '无'}

只回答 JSON：{{"is_switch": true/false, "brief": "如果是切换点，用5字概括新话题；否则为空"}}"""

        result = _call_llm(prompt, model=MODEL_CHAPTER_DETECT)
        if result:
            parsed = _parse_json_safe(result)
            return parsed.get('is_switch', False), parsed.get('brief', '')
        return False, ''

    def _generate_title(self, chapter_text: str) -> str:
        """为一段章节文本生成标题。"""
        # 取前 20 行就够了
        head = '\n'.join(chapter_text.strip().split('\n')[:20])[:2000]
        prompt = f"""为以下会议片段生成一个标题（≤15字），概括核心讨论议题。
只输出标题文本，不要其他内容。

会议片段：
{head}"""

        result = _call_llm(prompt, model=MODEL_CHAPTER_DETECT, expect_json=False)
        if result:
            title = result.strip().strip('"').strip("'")
            return title[:30]
        return '未命名章节'

    def _get_time_range(self, chapter_lines: list[str],
                        sentences: list[dict] = None) -> tuple[float, float]:
        """从章节文本行提取起止时间戳。"""
        return self._get_time_range_static(chapter_lines)

    @staticmethod
    def _get_time_range_static(chapter_lines: list[str]) -> tuple[float, float]:
        """静态版本，供 ChapterSummarizer 复用。"""
        start_ms, end_ms = 0.0, 0.0
        for line in chapter_lines:
            m = re.match(r'\[(\d+):(\d+):(\d+)\]', line)
            if m:
                ms = int(m.group(1)) * 3600000 + int(m.group(2)) * 60000 + int(m.group(3)) * 1000
                if start_ms == 0.0:
                    start_ms = ms
                end_ms = ms
        return start_ms, end_ms

    def _single_chapter(self, lines: list[str],
                        sentences: list[dict] = None) -> list[dict]:
        """文本太短，返回单章。"""
        text = '\n'.join(lines)
        start_ms, end_ms = self._get_time_range(lines, sentences)
        return [{
            'index': 0, 'title': '会议全文',
            'start_ms': start_ms, 'end_ms': end_ms,
            'start_line': 0, 'end_line': len(lines),
            'text': text,
        }]

    def detect_sub_segments(self, chapter_text: str) -> list[dict]:
        """在章节内检测语义子段（汇报/讨论/关键人物/过渡）。

        Returns:
            [{start_line, end_line, type, brief}]
        """
        lines = chapter_text.strip().split('\n')
        if len(lines) < 20:
            return [{'start_line': 0, 'end_line': len(lines),
                     'type': 'discussion', 'brief': ''}]

        prompt = f"""你是会议结构分析专家。将以下会议文本划分为 2-5 个连续子段。
每个子段根据内容性质标注类型：

类型定义：
- report: 单人持续汇报/陈述，无打断无讨论
- discussion: 多人讨论、观点交锋、问答互动
- leader: 关键人物发言占主要篇幅
- transition: 过渡语/会议组织/简短总结

输出严格 JSON 数组，每项必须覆盖连续的行范围且不重不漏：
```json
[
  {{"start_line": 0, "type": "report", "brief": "5字概括"}},
  {{"start_line": 8, "type": "discussion", "brief": "5字概括"}},
  ...
]
```

会议文本（共 {len(lines)} 行）：
{chapter_text[:6000]}"""

        result = _call_llm(prompt, model=MODEL_CHAPTER_DETECT)
        if not result:
            return [{'start_line': 0, 'end_line': len(lines),
                     'type': 'discussion', 'brief': ''}]

        parsed = _parse_json_safe(result)
        if not isinstance(parsed, list) or len(parsed) == 0:
            return [{'start_line': 0, 'end_line': len(lines),
                     'type': 'discussion', 'brief': ''}]

        # 补全 end_line，并夹紧到有效行范围内
        sub_segments = []
        for i, seg in enumerate(parsed):
            start = max(0, min(len(lines), seg.get('start_line', 0)))
            if i + 1 < len(parsed):
                end = max(0, min(len(lines), parsed[i + 1].get('start_line', len(lines))))
            else:
                end = len(lines)
            # 行号越界导致空切片时跳过
            if start >= end:
                continue
            sub_segments.append({
                'start_line': start,
                'end_line': end,
                'type': seg.get('type', 'discussion'),
                'brief': seg.get('brief', ''),
            })
        # 全部越界时兜底为整段
        if not sub_segments:
            sub_segments = [{'start_line': 0, 'end_line': len(lines),
                             'type': 'discussion', 'brief': ''}]
        return sub_segments


# ── 章节摘要 ────────────────────────────────────────

class ChapterSummarizer:
    """用 qwen-max 对每个章节生成密集信息摘要，支持子段拆分和差异化处理。"""

    def __init__(self):
        self.chapter_prompt = _load_prompt('chapter')
        self.leader_prompt = _load_prompt('leader')

    def summarize_all(self, chapters: list[dict], speaker_names: dict) -> list[dict]:
        """对所有章节做子段级摘要。"""
        kb_entries = load_knowledge_base()

        detector = ChapterDetector()

        results = []
        for i, ch in enumerate(chapters):
            title = ch.get('title', '...')
            chapter_text = ch.get('text', '')
            chapter_lines = chapter_text.strip().split('\n')

            if len(chapter_text) > MAX_CHARS_PER_CHAPTER:
                chapter_text = chapter_text[:MAX_CHARS_PER_CHAPTER]

            print(f'[摘要] 章节 {i+1}/{len(chapters)}: {title}')

            # 检测子段
            sub_segments = detector.detect_sub_segments(chapter_text)
            print(f'  → {len(sub_segments)} 个子段')

            sub_results = []
            for j, seg in enumerate(sub_segments):
                seg_lines = chapter_lines[seg['start_line']:seg['end_line']]
                seg_text = '\n'.join(seg_lines)
                seg_type = seg['type']
                seg_start_ms, seg_end_ms = ChapterDetector._get_time_range_static(seg_lines)

                # 按关键词检索相关知识（替代全量注入）
                seg_knowledge, _ = retrieve_context(seg_text, kb_entries)

                type_label = {'report': '汇报', 'discussion': '讨论',
                              'leader': '关键人物发言', 'transition': '过渡'}.get(seg_type, seg_type)

                # 关键人物子段走专项
                leader_analysis = ''
                if seg_type == 'leader':
                    leader_text = self._filter_leader(seg_text)
                    if leader_text:
                        print(f'    [{j+1}] {type_label}: 关键人物专项分析...')
                        leader_analysis = self._extract_leader(
                            leader_text, title, seg_knowledge)

                # 非关键人物子段生成摘要（关键人物段也生成简要摘要作为上下文）
                print(f'    [{j+1}] {type_label}: {seg.get("brief", "")}')
                summary = self._summarize_sub_segment(
                    seg_text, seg_type, seg_knowledge)

                sub_results.append({
                    'type': seg_type,
                    'type_label': type_label,
                    'title': summary.get('title', ''),
                    'summary': summary.get('summary', ''),
                    'decisions': summary.get('decisions', []),
                    'action_items': summary.get('action_items', []),
                    'start_ms': seg_start_ms,
                    'end_ms': seg_end_ms,
                    'leader_analysis': leader_analysis,
                })

            results.append({
                'index': i,
                'title': title,
                'start_ms': ch.get('start_ms', 0),
                'end_ms': ch.get('end_ms', 0),
                'sub_segments': sub_results,
            })
        return results

    def _summarize_sub_segment(self, seg_text: str, seg_type: str,
                                knowledge: str) -> dict:
        """对单个子段生成摘要，根据类型调整粒度。"""
        detail_hint = {
            'report': '该段为单人汇报，简略提炼核心议题和关键数据即可，2-3句。',
            'discussion': '该段为多人讨论，必须详细处理。保留每个发言人的观点、论据、分歧、共识，不压缩信息量。用连贯段落呈现。',
            'transition': '该段为过渡内容，一句话概括即可。',
            'leader': '简要概括上下文即可，关键人物发言会专项处理。',
        }.get(seg_type, '')

        prompt = self.chapter_prompt.replace(
            '{knowledge_context}', knowledge or '无').replace(
            '{chapter_text}', seg_text)

        # 在 prompt 中插入粒度提示
        if detail_hint:
            prompt = prompt.replace(
                '## 会议转写',
                f'## 摘要粒度\n{detail_hint}\n\n## 会议转写')

        result = _call_llm(prompt, model=MODEL_CHAPTER_SUMMARY)
        if result:
            return _parse_json_safe(result)
        return {'title': '', 'summary': seg_text[:200],
                'decisions': [], 'action_items': []}

    def _summarize_one(self, chapter_text: str, knowledge: str) -> dict:
        """对单个章节文本生成摘要。"""
        prompt = self.chapter_prompt.replace(
            '{knowledge_context}', knowledge or '无').replace(
            '{chapter_text}', chapter_text)

        result = _call_llm(prompt, model=MODEL_CHAPTER_SUMMARY)
        if result:
            return _parse_json_safe(result)
        return {'title': '', 'summary': chapter_text[:200],
                'decisions': [], 'action_items': []}

    def _filter_leader(self, chapter_text: str) -> str:
        """从章节文本中提取关键人物发言行。"""
        lines = chapter_text.split('\n')
        leader_lines = []
        for line in lines:
            for name in LEADER_NAMES:
                if name in line:
                    leader_lines.append(line)
                    break
        return '\n'.join(leader_lines) if leader_lines else ''

    def _extract_leader(self, leader_text: str, topic: str,
                        knowledge: str) -> str:
        """对关键人物发言做专项标注分析。"""
        prompt = self.leader_prompt.replace(
            '{knowledge_context}', knowledge or '无').replace(
            '{leader_text}', leader_text)

        # 在 prompt 前加话题标题引导
        prompt = prompt.replace(
            '## 会议转写（含关键人物发言）',
            f'当前话题：{topic}\n\n## 会议转写（含关键人物发言）')

        result = _call_llm(prompt, model=MODEL_CHAPTER_SUMMARY,
                           expect_json=False)
        return result or ''


# ── 格式化输出 ──────────────────────────────────────

def format_timestamp(ms: float) -> str:
    """毫秒 → MM:SS 格式。"""
    total_sec = int(ms / 1000)
    return f'{total_sec // 60:02d}:{total_sec % 60:02d}'


def format_chapter_markdown(chapters: list[dict]) -> str:
    """将章节摘要格式化为 Markdown，含子段级时间戳和类型标签。"""
    lines = ['## 章节详情', '']

    all_decisions = []
    all_actions = []

    for ch in chapters:
        title = ch.get('title', '未命名')
        ts_start = format_timestamp(ch.get('start_ms', 0))
        ts_end = format_timestamp(ch.get('end_ms', 0))

        lines.append(f'### {ts_start}-{ts_end} {title}')
        lines.append('')

        sub_segments = ch.get('sub_segments', [])
        if not sub_segments:
            # 兼容旧格式
            summary = ch.get('summary', '')
            if summary:
                lines.append(summary)
                lines.append('')
            continue

        for seg in sub_segments:
            seg_ts = format_timestamp(seg.get('start_ms', 0))
            seg_type = seg.get('type_label', '')
            seg_title = seg.get('title', '')
            seg_summary = seg.get('summary', '')

            # 子段标题行
            label = f'**[{seg_type}]** ' if seg_type else ''
            lines.append(f'**{seg_ts} {label}{seg_title}**')
            lines.append('')

            if seg_summary:
                lines.append(seg_summary)
                lines.append('')

            # 关键人物专项分析
            leader = seg.get('leader_analysis', '')
            if leader:
                lines.append(leader)
                lines.append('')

            # 收集决定和行动项
            all_decisions.extend(seg.get('decisions', []))
            all_actions.extend(seg.get('action_items', []))

    # 汇总决定和行动项
    if all_decisions:
        lines.append('---')
        lines.append('')
        lines.append('## 结论/决定汇总')
        for d in all_decisions:
            lines.append(f'- {d}')
        lines.append('')

    if all_actions:
        lines.append('## 后续行动汇总')
        for a in all_actions:
            lines.append(f'- {a}')
        lines.append('')

    return '\n'.join(lines)


# ── 顶层接口 ────────────────────────────────────────

def generate_chapter_summaries(transcript_path: str,
                               sentences: list[dict] = None,
                               speaker_names: dict = None) -> tuple[list[dict], str]:
    """从转写文件生成章节摘要 + 关键人物分析。

    Args:
        transcript_path: transcript.txt 路径
        sentences: sentence 列表（从 JSON 获取，用于时间戳）
        speaker_names: {speaker_id: display_name}

    Returns:
        (chapters_list, markdown_text)
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = f.read()

    # 1. 检测章节边界
    print('[章节检测] 正在识别话题切换点...')
    detector = ChapterDetector()
    chapters = detector.detect(transcript, sentences)
    print(f'[章节检测] 识别到 {len(chapters)} 个章节')

    # 2. 各章节摘要 + 关键人物分析
    print('[章节摘要] 正在生成摘要...')
    summarizer = ChapterSummarizer()
    results = summarizer.summarize_all(chapters, speaker_names or {})

    # 3. 格式化
    md = format_chapter_markdown(results)

    return results, md
