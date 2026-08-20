"""
RAG（检索增强生成）知识库模块。

提供：
- load_knowledge_base(): 加载并校验知识库 JSON 文件
- retrieve_context():  根据转写文本检索匹配的背景知识和术语解释
"""
import json
import os

# ============================================================
# 默认路径
# ============================================================
DEFAULT_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'knowledge_base.json'
)


def load_knowledge_base(path=None):
    """加载并校验知识库 JSON 文件。

    Args:
        path: 知识库文件路径，默认为 config/knowledge_base.json

    Returns:
        list[dict]: 校验通过的知识条目列表；文件不存在或解析失败时返回 []
    """
    if path is None:
        path = DEFAULT_KB_PATH

    if not os.path.exists(path):
        print(f'[RAG] 知识库文件 {path} 不存在，RAG功能将跳过。')
        return []

    try:
        with open(path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
    except json.JSONDecodeError as e:
        print(f'[RAG] 知识库JSON解析错误: {e}，RAG功能将跳过。')
        return []
    except Exception as e:
        print(f'[RAG] 读取知识库文件失败: {e}，RAG功能将跳过。')
        return []

    if not isinstance(entries, list):
        print('[RAG] 知识库格式错误（应为JSON数组），RAG功能将跳过。')
        return []

    # 逐条校验
    valid_entries = []
    required_fields = {'id', 'type', 'title', 'content', 'keywords', 'priority'}

    for entry in entries:
        if not isinstance(entry, dict):
            print(f'[RAG] 跳过非对象条目: {entry}')
            continue

        missing = required_fields - set(entry.keys())
        if missing:
            print(f'[RAG] 条目缺少必填字段 {missing}，跳过: {entry.get("id", "unknown")}')
            continue

        if entry.get('priority') not in ('high', 'normal', 'low'):
            print(f'[RAG] 条目 priority 无效: {entry.get("priority")}，跳过: {entry["id"]}')
            continue

        keywords = entry.get('keywords', [])
        if not isinstance(keywords, list) or len(keywords) == 0:
            print(f'[RAG] 条目 keywords 为空或格式错误，跳过: {entry["id"]}')
            continue

        valid_entries.append(entry)

    if not valid_entries:
        print('[RAG] 知识库中无有效条目，RAG功能将跳过。')
    else:
        print(f'[RAG] 知识库加载成功（{len(valid_entries)} 条记录）。')
        bg_count = sum(1 for e in valid_entries if e.get('type') == 'background')
        term_count = sum(1 for e in valid_entries if e.get('type') == 'term')
        always_on = sum(1 for e in valid_entries if e.get('priority') == 'high')
        print(f'[RAG]   - 背景条目: {bg_count} | 术语条目: {term_count} | 始终注入: {always_on}')

    return valid_entries


def retrieve_context(transcript, kb_entries):
    """根据转写文本检索匹配的背景知识和术语解释。

    检索逻辑：
    - priority='high' 的条目始终包含（如公司业务背景）
    - priority='normal'/'low' 的条目仅在 keywords 中任意关键词出现在转写文本中时才包含
    - 关键词匹配为大小写不敏感的子串匹配

    Args:
        transcript: 转写文本（字符串）
        kb_entries: load_knowledge_base() 返回的知识条目列表

    Returns:
        tuple[str, list[str]]: (格式化的上下文文本, 匹配到的条目ID列表)
    """
    if not kb_entries or not transcript:
        return None, []

    transcript_lower = transcript.lower()
    background_entries = []
    term_entries = []
    matched_ids = []

    for entry in kb_entries:
        priority = entry.get('priority', 'normal')
        keywords = entry.get('keywords', [])

        # priority='high' 始终包含
        if priority == 'high':
            if entry.get('type') == 'background':
                background_entries.append(entry)
            else:
                term_entries.append(entry)
            matched_ids.append(entry['id'])
            continue

        # priority='normal': 关键词匹配
        matched = any(
            kw.lower() in transcript_lower
            for kw in keywords
        )
        if matched:
            if entry.get('type') == 'background':
                background_entries.append(entry)
            else:
                term_entries.append(entry)
            matched_ids.append(entry['id'])

    if not matched_ids:
        return None, []

    # 组装上下文文本
    lines = ['## 参考背景信息\n']

    if background_entries:
        lines.append('### 公司业务背景')
        for entry in background_entries:
            lines.append(entry['content'])
            lines.append('')

    if term_entries:
        lines.append('### 相关术语解释')
        for entry in term_entries:
            lines.append(f"**{entry['title']}**：{entry['content']}")
            lines.append('')

    context_text = '\n'.join(lines).strip()
    return context_text, matched_ids
