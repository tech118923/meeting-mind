"""
会议纪要生成模块：基于章节概要生成正式会议纪要。

用法：
    python meeting_summarizer.py --outline output/<会议名>/chapter_summaries.md

功能：
    - 调用千问大模型 + RAG知识库生成正式会议纪要
    - 输入为章节概要（由 pipeline 阶段 7 生成）
    - 输出 Markdown 格式纪要
"""

import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


def load_prompt_template(path: str = None) -> str:
    """加载会议纪要Prompt模板。"""
    if path is None:
        path = os.path.join(PROJECT_ROOT, 'config', 'prompt_template.md')

    if not os.path.exists(path):
        print(f'[纪要] 模板文件不存在: {path}')
        return _default_prompt()

    with open(path, 'r', encoding='utf-8') as f:
        template = f.read()

    if '{chapter_outline}' not in template:
        print('[纪要] 警告：模板缺少 {chapter_outline} 占位符')

    return template


def _default_prompt() -> str:
    """默认Prompt模板（config/prompt_template.md 缺失时使用）。"""
    return """你是一个专业的会议纪要助手。请根据以下会议章节概要，整理为正式的会议纪要。

核心原则：章节概要中没有出现的信息就是不存在。人名、日期、数据一律以章节概要为准。

## 输出结构

开头段：会议重点围绕【xxx、xxx...】等事项展开讨论。具体要求如下：

整体要求：提取会议中不绑定特定议题的总体判断或战略方向。

分议题纪要：按议题逐一展开，每个议题以"一、""二、"开头。

主要参会人：从章节概要中整理可确认的参会人员。

## 语言风格
- 聚焦会议实际达成的结论
- 动词区分："会议指出："、"会议明确："、"会议强调："、"会议要求："
- 优先用完整段落整合

{rag_context}

## 章节概要
{chapter_outline}
"""


def summarize(chapter_outline: str, prompt_template: str = None,
              knowledge_base_entries: list = None,
              hotwords: list = None,
              api_key: str = None,
              model: str = 'qwen-plus',
              rag_text: str = None) -> str | None:
    """生成会议纪要（参照章节概要）。

    Args:
        chapter_outline: 章节概要文本（chapter_summaries.md 内容），作为纪要生成主体
        prompt_template: Prompt模板（含 {chapter_outline} 和 {rag_context} 占位符）
        knowledge_base_entries: RAG知识库条目列表
        hotwords: 热词列表（用于增强LLM对业务术语的理解）
        api_key: API Key
        model: 模型名称
        rag_text: RAG 检索用文本（如完整清洗转写）；None 则回退用 chapter_outline

    Returns:
        Markdown格式的会议纪要文本，失败返回 None
    """
    import dashscope
    from dashscope import Generation

    if api_key:
        dashscope.api_key = api_key
    else:
        dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not dashscope.api_key:
            print('[纪要] 未设置 DASHSCOPE_API_KEY')
            return None

    # 加载 Prompt 模板
    if prompt_template is None:
        prompt_template = load_prompt_template()

    # RAG 检索（用完整清洗转写做关键词匹配，背景知识命中更全；默认回退章节概要）
    rag_context = None
    if knowledge_base_entries:
        try:
            from knowledge import retrieve_context
            rag_context, matched_ids = retrieve_context(rag_text or chapter_outline, knowledge_base_entries)
            if matched_ids:
                print(f'[纪要] RAG匹配到 {len(matched_ids)} 条背景知识: {", ".join(matched_ids)}')
        except ImportError:
            pass

    # 构建热词提示
    hotword_hint = ''
    if hotwords:
        hotword_texts = []
        for hw in hotwords[:20]:
            if isinstance(hw, dict):
                hotword_texts.append(hw.get('text', ''))
            else:
                hotword_texts.append(str(hw))
        if hotword_texts:
            hotword_hint = f'\n\n会议中涉及的关键术语（供参考，不要在纪要中额外解释这些术语）：{", ".join(hotword_texts)}'

    # 填充模板
    prompt = prompt_template.replace('{rag_context}', rag_context or '（无相关背景信息）')
    prompt = prompt.replace('{chapter_outline}', chapter_outline)
    prompt += hotword_hint

    # 截断过长的prompt（千问上下文窗口）
    if len(prompt) > 28000:
        print(f'[纪要] Prompt过长({len(prompt)}字符)，截断章节概要')
        # 保留模板 + RAG + 尽可能多的章节概要
        base = prompt_template.replace('{rag_context}', rag_context or '') + hotword_hint
        max_outline_len = 28000 - len(base)
        truncated = chapter_outline[:max_outline_len] + '\n\n[章节概要过长，后续内容已截断]'
        prompt = prompt_template.replace('{rag_context}', rag_context or '（无相关背景信息）')
        prompt = prompt.replace('{chapter_outline}', truncated)
        prompt += hotword_hint

    print(f'[纪要] Prompt长度: {len(prompt)} 字符')
    print(f'[纪要] 正在调用 {model}...')

    try:
        response = Generation.call(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
        )

        if response.status_code == 200:
            output = response.output
            if output is None:
                print('[纪要] API返回output为空')
                return None
            if hasattr(output, 'text') and output.text:
                return output.text
            if (hasattr(output, 'choices') and output.choices
                    and output.choices[0].message
                    and output.choices[0].message.content):
                return output.choices[0].message.content
            print(f'[纪要] 无法解析响应: {output}')
            return None
        else:
            print(f'[纪要] API调用失败: code={response.status_code}, message={response.message}')
            return None
    except Exception as e:
        print(f'[纪要] 异常: {e}')
        return None


def save_minutes(content: str, output_dir: str = None, filename: str = None) -> str:
    """保存会议纪要文件。"""
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        filename = f'meeting_minutes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'

    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'[纪要] 已保存: {filepath}')
    return filepath


def main():
    import argparse

    parser = argparse.ArgumentParser(description='生成会议纪要')
    parser.add_argument('--outline', required=True, help='章节概要路径 (chapter_summaries.md)')
    parser.add_argument('--rag-text', default=None, help='RAG 检索用文本路径（如完整清洗转写），默认用章节概要')
    parser.add_argument('--output-dir', default=None, help='输出目录')
    parser.add_argument('--prompt-template', default=None, help='Prompt模板路径')
    parser.add_argument('--knowledge-base', default=None, help='知识库JSON路径')
    parser.add_argument('--hotwords', default=None, help='热词JSON路径')
    parser.add_argument('--model', default='qwen-plus', help='千问模型名')
    args = parser.parse_args()

    # 加载章节概要
    with open(args.outline, 'r', encoding='utf-8') as f:
        chapter_outline = f.read()

    # 可选：加载 RAG 检索用文本
    rag_text = None
    if args.rag_text:
        with open(args.rag_text, 'r', encoding='utf-8') as f:
            rag_text = f.read()

    # 加载资源
    prompt_template = load_prompt_template(args.prompt_template)

    kb_entries = None
    kb_path = args.knowledge_base or os.path.join(PROJECT_ROOT, 'config', 'knowledge_base.json')
    try:
        from knowledge import load_knowledge_base
        kb_entries = load_knowledge_base(kb_path)
    except ImportError:
        pass

    hotwords = None
    hw_path = args.hotwords or os.path.join(PROJECT_ROOT, 'config', 'hotwords.json')
    if os.path.exists(hw_path):
        with open(hw_path, 'r', encoding='utf-8') as f:
            hotwords = json.load(f)
        print(f'[纪要] 加载 {len(hotwords)} 个热词')

    # 生成纪要
    minutes = summarize(
        chapter_outline,
        prompt_template=prompt_template,
        knowledge_base_entries=kb_entries,
        hotwords=hotwords,
        model=args.model,
        rag_text=rag_text,
    )

    if minutes:
        # 保存
        if args.output_dir is None:
            args.output_dir = os.path.join(PROJECT_ROOT, 'output')
        save_minutes(minutes, args.output_dir)

        print('\n' + '=' * 60)
        print('【会议纪要】')
        print('=' * 60)
        print(minutes[:2000])
        if len(minutes) > 2000:
            print(f'\n... (共 {len(minutes)} 字符)')
    else:
        print('[纪要] 生成失败')
        sys.exit(1)


if __name__ == '__main__':
    main()
