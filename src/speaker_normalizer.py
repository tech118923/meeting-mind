"""
说话人归一化模块：跨分段统一说话人ID。

利用音频分段之间的重叠区域，通过文本相似度匹配，
将各分段独立编号的说话人映射到全局统一的说话人空间。

用法：
    python speaker_normalizer.py --results output/segments/index.json --manifest segments/manifest.json

算法：
    1. 提取相邻段重叠区域的句子
    2. 时间窗口对齐 + Jaccard文本相似度匹配句子对
    3. 建立跨段说话人共现矩阵
    4. 贪心映射 + 传递闭包传播
    5. 输出全局说话人映射表
"""

import argparse
import json
import os
import sys


def jaccard_similarity(t1: str, t2: str) -> float:
    """计算两个文本的Jaccard字符相似度。"""
    if not t1 or not t2:
        return 0.0
    s1, s2 = set(t1), set(t2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def extract_overlap_sentences(seg_result: dict, overlap_start_local: float,
                              overlap_end_local: float) -> list[dict]:
    """从转写结果中提取指定时间范围内的句子。

    Args:
        seg_result: 单个分段的转写JSON（已解析为dict）
        overlap_start_local: 重叠区在该段中的起始时间（ms）
        overlap_end_local: 重叠区在该段中的结束时间（ms）

    Returns:
        重叠区内的句子列表，每项含 speaker_id, begin_time, end_time, text
    """
    sentences = []
    transcripts = seg_result.get('transcripts', [])

    for transcript in transcripts:
        for sentence in transcript.get('sentences', []):
            begin = sentence.get('begin_time', 0)
            end = sentence.get('end_time', 0)

            # 句子与重叠区有交集
            if end > overlap_start_local and begin < overlap_end_local:
                sentences.append({
                    'speaker_id': sentence.get('speaker_id', -1),
                    'begin_time': begin,
                    'end_time': end,
                    'text': sentence.get('text', '').strip(),
                })

    return sentences


def match_speakers_between_segments(sentences_a: list[dict], sentences_b: list[dict],
                                    time_offset_b_to_a: float,
                                    similarity_threshold: float = 0.3,
                                    time_tolerance_ms: float = 5000) -> dict:
    """在两个相邻分段之间建立说话人映射。

    Args:
        sentences_a: 段A重叠区的句子
        sentences_b: 段B重叠区的句子
        time_offset_b_to_a: 段B的本地时间转换为段A本地时间的偏移（ms）
                            B_local + offset = A_local
        similarity_threshold: 文本相似度最低阈值
        time_tolerance_ms: 时间对齐容差（ms），默认±5秒

    Returns:
        {speaker_id_in_A: speaker_id_in_B} 映射表
    """
    # 构建共现矩阵: cooccur[spk_a][spk_b] = 匹配次数
    cooccur = {}

    for sa in sentences_a:
        spk_a = sa['speaker_id']
        if spk_a not in cooccur:
            cooccur[spk_a] = {}

        best_similarity = 0.0
        best_spk_b = None

        for sb in sentences_b:
            spk_b = sb['speaker_id']

            # 时间对齐检查
            b_time_in_a = sb['begin_time'] + time_offset_b_to_a
            time_diff = abs(sa['begin_time'] - b_time_in_a)
            if time_diff > time_tolerance_ms:
                continue

            # 文本相似度
            sim = jaccard_similarity(sa['text'], sb['text'])
            if sim > best_similarity:
                best_similarity = sim
                best_spk_b = spk_b

        if best_spk_b is not None and best_similarity >= similarity_threshold:
            cooccur[spk_a][best_spk_b] = cooccur[spk_a].get(best_spk_b, 0) + 1

    # 贪心映射：每个段A说话人选择共现最多的段B说话人
    mapping = {}
    for spk_a, spk_b_counts in cooccur.items():
        if not spk_b_counts:
            continue
        # 选共现次数最多的
        best_spk_b = max(spk_b_counts, key=spk_b_counts.get)
        best_count = spk_b_counts[best_spk_b]
        total_matches = sum(spk_b_counts.values())
        # 需要超过半数匹配
        if best_count >= total_matches * 0.5:
            mapping[spk_a] = best_spk_b

    return mapping


def transitive_closure(pairwise_mappings: list[dict]) -> dict:
    """通过传递闭包将所有分段的说话人映射到全局空间。

    使用并查集（Union-Find）算法，将传递关联的说话人合并为同一全局ID。

    Args:
        pairwise_mappings: 每对相邻分段的映射列表 [{seg_N_spk: seg_N+1_spk}, ...]

    Returns:
        {(seg_index, local_spk_id): global_spk_id}
    """
    # 收集所有 (seg_index, local_spk)
    all_nodes = set()
    for seg_idx, mapping in enumerate(pairwise_mappings):
        for spk_a, spk_b in mapping.items():
            all_nodes.add((seg_idx, spk_a))
            all_nodes.add((seg_idx + 1, spk_b))

    # 并查集
    parent = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 初始化
    for node in all_nodes:
        parent[node] = node

    # 合并映射关联的节点
    for seg_idx, mapping in enumerate(pairwise_mappings):
        for spk_a, spk_b in mapping.items():
            union((seg_idx, spk_a), (seg_idx + 1, spk_b))

    # 分配全局ID
    component_ids = {}
    global_spk_id = 0
    result = {}

    for node in sorted(all_nodes):
        root = find(node)
        if root not in component_ids:
            component_ids[root] = global_spk_id
            global_spk_id += 1
        result[node] = component_ids[root]

    return result, global_spk_id


class SpeakerNormalizer:
    """跨段说话人ID归一化器。"""

    def __init__(self, similarity_threshold: float = 0.3,
                 time_tolerance_ms: float = 5000):
        self.similarity_threshold = similarity_threshold
        self.time_tolerance_ms = time_tolerance_ms

    def normalize(self, segment_results: dict[int, dict],
                  manifest: dict) -> dict:
        """执行说话人归一化。

        Args:
            segment_results: {seg_index: parsed_transcript_json}
            manifest: 分段清单（含 start_time, duration, overlap 等）

        Returns:
            {
                'global_speaker_count': int,
                'mapping': {(seg_index, local_spk): global_spk},
                'stats': {...}
            }
        """
        segments = manifest['segments']
        overlap_ms = manifest['overlap'] * 1000  # 转换为毫秒

        if len(segments) < 2:
            # 单分段，直接映射
            result = segment_results.get(segments[0]['index'], {})
            local_speakers = self._collect_speakers(result)
            mapping = {f'{segments[0]["index"]}_{spk}': spk for spk in local_speakers}
            return {
                'global_speaker_count': len(local_speakers),
                'mapping': mapping,
                'stats': {'segments': 1, 'total_local_speakers': len(local_speakers),
                          'mapped_pairs': 0, 'unmapped_speakers': 0},
            }

        # 逐对处理相邻分段
        pairwise_mappings = []
        match_stats = []

        for i in range(len(segments) - 1):
            seg_a = segments[i]
            seg_b = segments[i + 1]

            res_a = segment_results.get(seg_a['index'])
            res_b = segment_results.get(seg_b['index'])
            if res_a is None or res_b is None:
                print(f'[警告] 分段 {seg_a["index"]} 或 {seg_b["index"]} 结果缺失，跳过')
                pairwise_mappings.append({})
                continue

            # 重叠区在段A的本地时间：段A的最后 overlap_ms
            overlap_start_a = (seg_a['duration'] - overlap_ms / 1000) * 1000
            overlap_end_a = seg_a['duration'] * 1000

            # 重叠区在段B的本地时间：段B的最初 overlap_ms
            overlap_start_b = 0
            overlap_end_b = overlap_ms

            # 时间偏移：将段B本地时间转换为段A本地时间
            # B_local(0) 对应 A的 (duration - overlap)
            time_offset_b_to_a = (seg_a['duration'] * 1000 - overlap_ms)

            sentences_a = extract_overlap_sentences(res_a, overlap_start_a, overlap_end_a)
            sentences_b = extract_overlap_sentences(res_b, overlap_start_b, overlap_end_b)

            print(f'\n[归一化] 分段 {seg_a["index"]} <-> {seg_b["index"]}: '
                  f'重叠区 {len(sentences_a)}+{len(sentences_b)} 个句子')

            mapping = match_speakers_between_segments(
                sentences_a, sentences_b,
                time_offset_b_to_a,
                similarity_threshold=self.similarity_threshold,
                time_tolerance_ms=self.time_tolerance_ms,
            )

            pairwise_mappings.append(mapping)
            match_stats.append({
                'seg_a': seg_a['index'],
                'seg_b': seg_b['index'],
                'sentences_a': len(sentences_a),
                'sentences_b': len(sentences_b),
                'mappings_found': len(mapping),
                'mapping': {str(k): v for k, v in mapping.items()},
            })

            print(f'  映射: {len(mapping)} 个说话人匹配')

        # 传递闭包
        global_mapping, global_count = transitive_closure(pairwise_mappings)

        # 处理未映射的说话人：分配新全局ID
        all_local_speakers = set()
        for seg in segments:
            res = segment_results.get(seg['index'])
            if res:
                speakers = self._collect_speakers(res)
                for spk in speakers:
                    all_local_speakers.add((seg['index'], spk))

        unmapped = 0
        for node in all_local_speakers:
            if node not in global_mapping:
                global_mapping[node] = global_count
                global_count += 1
                unmapped += 1

        return {
            'global_speaker_count': global_count,
            'mapping': {f'{seg}_{spk}': gid for (seg, spk), gid in global_mapping.items()},
            'mapping_tuples': {(seg, spk): gid for (seg, spk), gid in global_mapping.items()},
            'stats': {
                'segments': len(segments),
                'total_local_speakers': len(all_local_speakers),
                'mapped_pairs': sum(s['mappings_found'] for s in match_stats),
                'unmapped_speakers': unmapped,
                'match_details': match_stats,
            },
        }

    @staticmethod
    def _collect_speakers(seg_result: dict) -> set[int]:
        """收集分段结果中所有出现的说话人ID。"""
        speakers = set()
        for transcript in seg_result.get('transcripts', []):
            for sentence in transcript.get('sentences', []):
                spk = sentence.get('speaker_id')
                if spk is not None:
                    speakers.add(spk)
        return speakers


def main():
    parser = argparse.ArgumentParser(description='跨分段说话人ID归一化')
    parser.add_argument('--results', required=True,
                        help='转写结果索引文件 (output/segments/index.json)')
    parser.add_argument('--manifest', required=True,
                        help='分段清单 (segments/manifest.json)')
    parser.add_argument('--output', default=None,
                        help='输出路径，默认 output/speaker_mapping.json')
    parser.add_argument('--similarity-threshold', type=float, default=0.3,
                        help='文本相似度阈值，默认0.3')
    parser.add_argument('--time-tolerance', type=float, default=5000,
                        help='时间对齐容差（ms），默认5000')
    args = parser.parse_args()

    # 加载转写结果索引
    with open(args.results, 'r', encoding='utf-8') as f:
        index = json.load(f)

    # 加载所有分段的转写JSON
    segment_results = {}
    results_dir = os.path.dirname(args.results)
    for seg_str, result_path in index.get('results', {}).items():
        seg_index = int(seg_str)
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                segment_results[seg_index] = json.load(f)
        else:
            print(f'[警告] 分段 {seg_index} 结果文件不存在: {result_path}')

    if not segment_results:
        print('[错误] 没有可用的转写结果')
        sys.exit(1)

    # 加载分段清单
    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # 执行归一化
    normalizer = SpeakerNormalizer(
        similarity_threshold=args.similarity_threshold,
        time_tolerance_ms=args.time_tolerance,
    )
    result = normalizer.normalize(segment_results, manifest)

    # 输出
    if args.output is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output = os.path.join(project_root, 'output', 'speaker_mapping.json')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        # 序列化时转换 tuple key 为字符串
        output_data = {
            'global_speaker_count': result['global_speaker_count'],
            'mapping': result['mapping'],
            'stats': result['stats'],
        }
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f'\n{"="*60}')
    print(f'说话人归一化完成')
    print(f'{"="*60}')
    print(f'  总分段数: {result["stats"]["segments"]}')
    print(f'  分段内说话人总数: {result["stats"]["total_local_speakers"]}')
    print(f'  全局说话人数: {result["global_speaker_count"]}')
    print(f'  成功映射: {result["stats"]["mapped_pairs"]} 对')
    print(f'  未映射（新增）: {result["stats"]["unmapped_speakers"]} 个')
    print(f'  结果已保存至: {args.output}')


if __name__ == '__main__':
    main()
