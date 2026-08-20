"""
转写合并模块：合并各分段转写结果，应用全局说话人映射，输出统一说话人标签的完整转写。

用法：
    python merge_transcript.py --results output/segments/index.json \
                               --mapping output/speaker_mapping.json \
                               --manifest segments/manifest.json

输出格式：
    - full_transcript.json  — 完整结构化数据
    - full_transcript.txt   — 人类可读文本 [HH:MM:SS] 说话人N: ...
    - transcript_by_speaker.txt — 按说话人分组
"""

import argparse
import json
import os
import sys
from typing import Any


def format_timestamp(ms: float) -> str:
    """将毫秒格式化为 [HH:MM:SS] 格式。"""
    total_sec = int(ms / 1000)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    return f'[{hours:02d}:{minutes:02d}:{seconds:02d}]'


def build_global_timeline(segment_results: dict[int, dict],
                          speaker_mapping: dict,
                          manifest: dict,
                          dedup_buffer_sec: float = 5.0) -> list[dict]:
    """构建全局统一时间线的句子列表。

    Args:
        segment_results: {seg_index: parsed_transcript_json}
        speaker_mapping: {(seg_index, local_spk): global_spk} 或 {f"{seg}_{spk}": global_spk}
        manifest: 分段清单
        dedup_buffer_sec: 去重缓冲区（秒），重叠区末尾的N秒内容在下一段中去除

    Returns:
        按时间排序的全局句子列表
    """
    all_sentences = []
    segments = manifest['segments']

    # 解析说话人映射格式
    # mapping 格式可能是 {'seg_spk': gid} 或包含 'mapping_tuples' 的 dict
    if isinstance(speaker_mapping, dict) and 'mapping_tuples' in speaker_mapping:
        spk_map = speaker_mapping['mapping_tuples']
    elif isinstance(speaker_mapping, dict) and 'mapping' in speaker_mapping:
        # 字符串key格式: '0_1' = seg0_spk1
        raw_map = speaker_mapping['mapping']
        spk_map = {}
        for key, gid in raw_map.items():
            parts = key.split('_', 1)
            if len(parts) == 2:
                spk_map[(int(parts[0]), int(parts[1]))] = gid
    else:
        spk_map = speaker_mapping

    for seg in segments:
        seg_index = seg['index']
        seg_start_ms = seg['start_time'] * 1000  # 全局时间偏移

        res = segment_results.get(seg_index)
        if res is None:
            print(f'[合并] 分段 {seg_index} 结果缺失，跳过')
            continue

        # 计算去重边界（该段重叠区末尾需去除）
        cutoff_global_ms = None
        if not seg.get('is_last', False):
            cutoff_global_ms = (seg['end_time'] - dedup_buffer_sec) * 1000

        for transcript in res.get('transcripts', []):
            for sentence in transcript.get('sentences', []):
                local_spk = sentence.get('speaker_id', -1)
                global_spk = spk_map.get((seg_index, local_spk), local_spk)

                begin_local = sentence.get('begin_time', 0)
                end_local = sentence.get('end_time', 0)
                text = sentence.get('text', '').strip()
                if not text:
                    continue

                # 计算句级平均置信度
                words = sentence.get('words', [])
                confidence = None
                if words:
                    confs = [w.get('confidence', 0) for w in words if 'confidence' in w]
                    if confs:
                        confidence = sum(confs) / len(confs)

                begin_global = begin_local + seg_start_ms
                end_global = end_local + seg_start_ms

                # 去重检查：跳过下一段中属于重叠缓冲区的句子
                if cutoff_global_ms is not None and begin_global >= cutoff_global_ms:
                    continue

                all_sentences.append({
                    'speaker': f'SPK_{global_spk:02d}',
                    'speaker_id': global_spk,
                    'start_ms': begin_global,
                    'end_ms': end_global,
                    'text': text,
                    'confidence': confidence,
                })

    # 按时间排序
    all_sentences.sort(key=lambda s: s['start_ms'])

    # 合并相邻的同说话人短句（间距<1秒且同一说话人）
    merged = []
    for s in all_sentences:
        if merged and merged[-1]['speaker_id'] == s['speaker_id']:
            gap = s['start_ms'] - merged[-1]['end_ms']
            if gap < 1000:  # 1秒以内合并
                merged[-1]['end_ms'] = s['end_ms']
                merged[-1]['text'] += s['text']
                # 合并置信度（取平均）
                c1 = merged[-1].get('confidence')
                c2 = s.get('confidence')
                if c1 is not None and c2 is not None:
                    merged[-1]['confidence'] = (c1 + c2) / 2
                elif c2 is not None:
                    merged[-1]['confidence'] = c2
                continue
        merged.append(s)

    return merged


def generate_speaker_labels(sentences: list[dict]) -> dict[int, dict]:
    """统计各说话人的发言信息，用于生成标签。"""
    speakers = {}
    for s in sentences:
        spk_id = s['speaker_id']
        if spk_id not in speakers:
            speakers[spk_id] = {
                'label': s['speaker'],
                'utterance_count': 0,
                'total_duration_ms': 0,
                'sample_text': s['text'][:100],
            }
        speakers[spk_id]['utterance_count'] += 1
        speakers[spk_id]['total_duration_ms'] += (s['end_ms'] - s['start_ms'])
    return speakers


def save_full_json(sentences: list[dict], speakers: dict, metadata: dict,
                   output_path: str):
    """保存完整结构化JSON。"""
    output = {
        'metadata': metadata,
        'speakers': speakers,
        'sentences': sentences,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'[输出] 完整JSON → {output_path}')


LOW_CONF_THRESHOLD = 0.5


def save_readable_transcript(sentences: list[dict], output_path: str):
    """保存人类可读文本格式。低置信度句子前加 [⚠低置信] 标记。"""
    lines = []
    for s in sentences:
        ts = format_timestamp(s['start_ms'])
        conf = s.get('confidence')
        marker = ''
        if conf is not None and conf < LOW_CONF_THRESHOLD:
            marker = '[⚠低置信] '
        lines.append(f'{ts} {marker}{s["speaker"]}: {s["text"]}')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[输出] 可读文本 → {output_path}')


def save_by_speaker(sentences: list[dict], output_path: str):
    """保存按说话人分组的文本。"""
    # 按说话人分组
    by_speaker = {}
    for s in sentences:
        spk = s['speaker']
        if spk not in by_speaker:
            by_speaker[spk] = []
        by_speaker[spk].append(s)

    lines = []
    for spk in sorted(by_speaker.keys()):
        lines.append(f'\n=== {spk} ===')
        for s in by_speaker[spk]:
            ts = format_timestamp(s['start_ms'])
            lines.append(f'{ts} {s["text"]}')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[输出] 按说话人分组 → {output_path}')


def main():
    parser = argparse.ArgumentParser(description='合并分段转写结果')
    parser.add_argument('--results', required=True,
                        help='转写结果索引文件 (output/segments/index.json)')
    parser.add_argument('--mapping', required=True,
                        help='说话人映射文件 (output/speaker_mapping.json)')
    parser.add_argument('--manifest', required=True,
                        help='分段清单 (segments/manifest.json)')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录，默认 output/')
    parser.add_argument('--source', default=None,
                        help='原始音频文件名（用于元数据）')
    args = parser.parse_args()

    # 输出目录
    if args.output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output_dir = os.path.join(project_root, 'output')
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    with open(args.results, 'r', encoding='utf-8') as f:
        index = json.load(f)
    with open(args.mapping, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # 加载所有分段的转写JSON
    segment_results = {}
    results_dir = os.path.dirname(args.results)
    for seg_str, result_path in index.get('results', {}).items():
        seg_index = int(seg_str)
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                segment_results[seg_index] = json.load(f)

    # 构建全局时间线
    sentences = build_global_timeline(segment_results, mapping, manifest)
    print(f'[合并] 共 {len(sentences)} 个句子')

    # 统计说话人
    speakers = generate_speaker_labels(sentences)
    print(f'[合并] 共 {len(speakers)} 个说话人:')
    for spk_id, info in sorted(speakers.items()):
        duration_min = info['total_duration_ms'] / 60000
        print(f'  {info["label"]}: {info["utterance_count"]} 句话, '
              f'约 {duration_min:.1f} 分钟, '
              f'示例: "{info["sample_text"]}..."')

    # 元数据
    metadata = {
        'source_audio': args.source or manifest.get('source_file', 'unknown'),
        'total_duration_sec': manifest.get('total_duration', 0),
        'total_sentences': len(sentences),
        'total_speakers': len(speakers),
    }

    # 输出各格式
    save_full_json(sentences, speakers, metadata,
                   os.path.join(args.output_dir, 'full_transcript.json'))
    save_readable_transcript(sentences,
                             os.path.join(args.output_dir, 'full_transcript.txt'))
    save_by_speaker(sentences,
                    os.path.join(args.output_dir, 'transcript_by_speaker.txt'))

    print(f'\n[合并] 全部输出完成！')


if __name__ == '__main__':
    main()
