"""
转写文本清洗模块。

在 ASR 转写文本送入 LLM 之前进行轻量清洗，提升可读性和信息密度。

用法：
    from transcript_cleaner import clean_transcript
    cleaned = clean_transcript(raw_transcript_text)

功能：
- 去除填充词（呃、嗯等语气词）
- 修复 ASR 结巴（"好好好方案" → "好方案"）
- 合并冗余口语短语
- 段落重建
"""

import re


def clean_transcript(text: str) -> str:
    """清洗 ASR 转写文本的主入口。

    清洗流程按顺序执行，每一步都有标注。
    清洗策略偏保守——宁可不删也不误删有意义的内容。

    Returns:
        清洗后的文本
    """
    if not text:
        return text

    text = _fix_stuttering(text)
    text = _remove_fillers(text)
    text = _remove_redundant_phrases(text)
    text = _reconstruct_paragraphs(text)
    text = _normalize_whitespace(text)

    return text


def _fix_stuttering(text: str) -> str:
    """修复 ASR 结巴：连续 3 次及以上的同一字符压缩为 1 次。

    "好好好方案" → "好方案"
    "不不不会" → "不会"
    "我我我们" → "我们"（注：连续 3 个"我"会被压缩为 1 个）
    """
    # 中文字符连续重复 3 次及以上 → 保留 1 次
    # 范围：CJK 统一表意文字 U+4E00–U+9FFF
    text = re.sub(r'([一-鿿])\1{2,}', r'\1', text)
    return text


def _remove_fillers(text: str) -> str:
    """去除无意义的语气填充词。

    处理以下模式：
    - "呃""嗯"在任何位置出现，直接删除
    - "啊"在非句末位置删除（句末保留，如"好的啊"）
    """
    # "呃"和"嗯"：几乎都是填充词，直接删
    text = re.sub(r'[呃嗯]', '', text)

    # "啊"在句中（后面还有内容）时删除
    # "啊，" → "，"
    # "啊。" → "。"
    text = re.sub(r'啊([，。！？,\.!\?\s])', r'\1', text)
    # "啊"在行首或前面有标点
    text = re.sub(r'([，。！？,\.!\?\s])啊', r'\1', text)

    return text


def _remove_redundant_phrases(text: str) -> str:
    """去除口语中高频出现的冗余短语。

    这些短语在口语中是正常的停顿/衔接，但写入正式文本会降低信息密度。
    """
    # "就是说" — 口语解释标记，保留后面内容
    text = re.sub(r'就是说[，,]?', '', text)
    # "就是，" → ""
    text = re.sub(r'就是[，,](?=\s*就是)', '', text)  # 连续"就是，就是"
    # 单个"就是[，,]" → 保留（可能是有意义的）
    # "然后呢[，,]" → ""
    text = re.sub(r'然后呢[，,]?', '', text)
    # "对吧[，,。]?" → ""
    text = re.sub(r'对吧[，,。]?', '', text)
    # "对不对[？?]?" → ""
    text = re.sub(r'对不对[？?]?', '', text)
    # "是不是[？?]?" → ""
    text = re.sub(r'是不是[？?]?', '', text)
    # "你比方说" → "比如"
    text = re.sub(r'你比方说', '比如', text)
    # "我这边" → ""（在句首或逗号后）
    text = re.sub(r'([，,。\s])我这边', r'\1', text)
    # "那个"在句首 → ""
    text = re.sub(r'^那个[，,]?\s*', '', text, flags=re.MULTILINE)

    return text


def _reconstruct_paragraphs(text: str) -> str:
    """重建段落：在语义转折处插入空行。

    判定规则：
    - "那么"开头的句子 → 可能是话题过渡，前面加空行
    - "另外""再有""还有" → 新增要点，前面加空行
    - "第一个""第二个" → 序号标记，前面加空行
    - "我这边"、"我这边要讲" → 汇报人切换标记
    """

    # 话题/逻辑过渡词
    transition_patterns = [
        (r'([。！？])(那么[，,]?\s*)', r'\1\n\n\2'),
        (r'([。！？])(另外[，,]?\s*)', r'\1\n\n\2'),
        (r'([。！？])(再有[，,]?\s*)', r'\1\n\n\2'),
        (r'([。！？])(还有[，,]?\s*)', r'\1\n\n\2'),
        (r'([。！？])(所以[，,]?\s*)', r'\1\n\n\2'),
        (r'([。！？])(最后[，,]?\s*)', r'\1\n\n\2'),
        (r'([。！？])(首先[，,]?\s*)', r'\1\n\n\2'),
    ]

    for pattern, repl in transition_patterns:
        text = re.sub(pattern, repl, text)

    return text


def _normalize_whitespace(text: str) -> str:
    """规范化空白字符。"""
    # 连续 3 个及以上换行 → 2 个
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 行尾空格
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    # 行首空格
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
    # 空行内的空格
    text = re.sub(r'\n[ \t]+\n', '\n\n', text)
    # 行内多个连续空格 → 1 个
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()
