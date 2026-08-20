"""
meeting-mind 统一管道：串联音频分割→上传→转写→归一化→合并→清洗→章节概要→会议纪要。

用法：
    # 一键全流程
    python src/pipeline.py --input meeting.mp3

    # 仅转写（已有URL）
    python src/pipeline.py --input meeting.mp3 --start-from transcribe --skip-split --skip-upload

    # 仅生成纪要（已有转写结果）
    python src/pipeline.py --input meeting.mp3 --start-from minutes \\
        --skip-split --skip-upload --skip-transcribe --skip-normalize --skip-merge

阶段说明:
    split       - ffmpeg 音频分割
    upload      - OSS上传，获取公网URL
    transcribe  - FunASR 批量转写（含说话人分离 + 热词增强）
    normalize   - 跨段说话人ID归一化
    merge       - 合并分段转写，输出4种格式
    identify    - 声纹识别：将SPK_XX映射为注册说话人真实姓名
    chapters    - LLM识别章节 + 生成章节概要
    minutes     - 生成完整会议纪要（RAG + 热词增强）
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# Windows 中文环境 stdout 默认 GBK，重配置为 UTF-8 以支持 ✓/✗ 等符号
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


class FlashPipeline:
    """meeting-mind 统一管道。"""

    def __init__(self, input_path: str, segment_duration: float = 5280,
                 overlap: float = 300, speaker_count: int = 15,
                 language: str = 'zh', voiceprint_db: str = None):
        self.input_path = input_path
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.speaker_count = speaker_count
        self.language = language

        # 路径 — 每场会议独立文件夹
        meeting_name = os.path.splitext(os.path.basename(input_path))[0]
        self.meeting_name = meeting_name
        self.segments_dir = os.path.join(PROJECT_ROOT, 'segments')
        self.output_dir = os.path.join(PROJECT_ROOT, 'output', meeting_name)
        self.config_dir = os.path.join(PROJECT_ROOT, 'config')
        self.speakers_dir = os.path.join(PROJECT_ROOT, 'speakers')
        self.voiceprint_db = voiceprint_db or os.path.join(
            self.speakers_dir, 'voiceprint_profiles.json')

        self.manifest_path = os.path.join(self.segments_dir, 'manifest.json')
        self.urls_path = os.path.join(self.segments_dir, 'urls.json')
        self.results_index = os.path.join(self.output_dir, 'segments', 'index.json')
        self.mapping_path = os.path.join(self.output_dir, 'segments', 'speaker_map.json')
        self.transcript_json = os.path.join(self.output_dir, 'transcript.json')
        self.transcript_txt = os.path.join(self.output_dir, 'transcript.txt')
        self.by_speaker_txt = os.path.join(self.output_dir, 'by_speaker.txt')
        self.voiceprint_json = os.path.join(self.output_dir, 'voiceprint.json')

        for d in [self.segments_dir, self.output_dir,
                  os.path.join(self.output_dir, 'segments')]:
            os.makedirs(d, exist_ok=True)

    # ── 管线 ──────────────────────────────────────────────

    def run(self, start_from: str = 'split', skip_stages: set = None,
            ffmpeg_bin: str = '', generate_chapters: bool = True,
            generate_minutes: bool = True):
        skip = skip_stages or set()
        stages = [
            ('split',       self._split),
            ('upload',      self._upload),
            ('transcribe',  self._transcribe),
            ('normalize',   self._normalize),
            ('merge',       self._merge),
            ('identify',    self._identify),
            ('chapters',    self._chapters),
            ('minutes',     self._minutes),
        ]

        started = False
        for name, handler in stages:
            if not started and name != start_from:
                print(f'[管道] 跳过: {name}')
                continue
            started = True
            if name in skip:
                print(f'[管道] 跳过: {name}')
                continue
            try:
                handler()
            except Exception as e:
                print(f'\n[管道] "{name}" 阶段失败: {e}')
                print(f'[管道] 恢复命令: --start-from {name}')
                raise

        self._print_summary()

    # ── 各阶段实现 ────────────────────────────────────────

    def _split(self):
        print(f'\n{"="*60}\n阶段 1/8: 音频分割\n{"="*60}')
        # 检查缓存是否匹配当前输入文件
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r', encoding='utf-8') as f:
                    old = json.load(f)
                if old.get('source_file') == self.input_path:
                    print('[分割] 清单已存在且匹配当前文件，跳过')
                    return
                else:
                    print('[分割] 清单属于其他文件，重新分割')
            except Exception:
                pass
        from audio_splitter import split_audio, get_audio_duration
        total = get_audio_duration(self.input_path)
        print(f'[分割] 时长: {total/3600:.1f}h, 段长: {self.segment_duration/60:.0f}min, 重叠: {self.overlap}s')
        split_audio(self.input_path, self.segment_duration, self.overlap, self.segments_dir)

    def _upload(self):
        print(f'\n{"="*60}\n阶段 2/8: OSS上传\n{"="*60}')
        if os.path.exists(self.urls_path):
            try:
                with open(self.urls_path, 'r', encoding='utf-8') as f:
                    old = json.load(f)
                # 检查URL是否属于当前manifest
                if os.path.exists(self.manifest_path):
                    with open(self.manifest_path, 'r', encoding='utf-8') as fm:
                        m = json.load(fm)
                    if old.get('source_file') == m.get('source_file'):
                        print('[上传] URL清单已存在且匹配，跳过')
                        return
            except Exception:
                pass
            print('[上传] URL清单过期，重新上传')
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f'请先执行分割阶段: {self.manifest_path}')
        from oss_uploader import OSSUploader, load_oss_config
        cfg = load_oss_config()
        up = OSSUploader(cfg)
        up.connect()
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        url_list = []
        for seg in manifest['segments']:
            if not os.path.exists(seg['file']):
                print(f'[跳过] {seg["file"]} 不存在')
                continue
            key = f"meeting_segments/{os.path.basename(seg['file'])}"
            url = up.upload_file(seg['file'], key)
            url_list.append({'index': seg['index'], 'local_file': seg['file'],
                             'oss_key': key, 'url': url,
                             'start_time': seg['start_time'], 'end_time': seg['end_time']})
        with open(self.urls_path, 'w', encoding='utf-8') as f:
            json.dump({'source_file': self.input_path,
                       'segments': url_list,
                       'generated_at': datetime.now().isoformat()},
                      f, indent=2, ensure_ascii=False)
        print(f'[上传] 完成 {len(url_list)} 个文件')

    def _transcribe(self):
        print(f'\n{"="*60}\n阶段 3/8: ASR批量转写\n{"="*60}')
        if not os.path.exists(self.urls_path):
            raise FileNotFoundError(f'请先执行上传阶段: {self.urls_path}')
        import dashscope
        dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
        if not dashscope.api_key:
            raise RuntimeError('未设置 DASHSCOPE_API_KEY')
        ws = os.environ.get('DASHSCOPE_WORKSPACE_ID')
        if ws:
            dashscope.base_http_api_url = f'https://{ws}.cn-beijing.maas.aliyuncs.com/api/v1'

        from batch_transcribe import BatchTranscriber, load_hotwords, setup_vocabulary, delete_vocabulary, ASR_MODEL
        hotwords = load_hotwords()
        vid = None
        if hotwords:
            vid = setup_vocabulary(ASR_MODEL, hotwords)
        try:
            with open(self.urls_path, 'r', encoding='utf-8') as f:
                urls = json.load(f)
            tc = BatchTranscriber(vocabulary_id=vid, speaker_count=self.speaker_count,
                                  language_hints=[self.language])
            tc.submit_all(urls['segments'], os.path.join(self.output_dir, 'segments'))
            results = tc.wait_all(os.path.join(self.output_dir, 'segments'))
            # 保存转写结果索引（normalize/merge 阶段依赖此文件）
            index_path = os.path.join(self.output_dir, 'segments', 'index.json')
            with open(index_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'results': {str(k): v for k, v in results.items()},
                    'completed_at': datetime.now().isoformat(),
                }, f, indent=2, ensure_ascii=False)
            print(f'[转写] 索引已保存至 {index_path}')
        finally:
            if vid:
                delete_vocabulary(vid)

    def _normalize(self):
        print(f'\n{"="*60}\n阶段 4/8: 说话人归一化\n{"="*60}')
        for p in [self.results_index, self.manifest_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f'缺少: {p}')
        from speaker_normalizer import SpeakerNormalizer
        with open(self.results_index, 'r', encoding='utf-8') as f:
            idx = json.load(f)
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        seg_results = {}
        for k, v in idx.get('results', {}).items():
            if os.path.exists(v):
                with open(v, 'r', encoding='utf-8') as f:
                    seg_results[int(k)] = json.load(f)
        n = SpeakerNormalizer()
        result = n.normalize(seg_results, manifest)
        with open(self.mapping_path, 'w', encoding='utf-8') as f:
            json.dump({'global_speaker_count': result['global_speaker_count'],
                       'mapping': result['mapping'], 'stats': result['stats']},
                      f, indent=2, ensure_ascii=False)
        print(f'[归一化] {result["global_speaker_count"]} 个全局说话人')

    def _merge(self):
        print(f'\n{"="*60}\n阶段 5/8: 合并转写\n{"="*60}')
        for p in [self.results_index, self.mapping_path, self.manifest_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f'缺少: {p}')
        from merge_transcript import (
            build_global_timeline, generate_speaker_labels,
            save_full_json, save_readable_transcript, save_by_speaker,
        )
        with open(self.results_index, 'r', encoding='utf-8') as f:
            idx = json.load(f)
        with open(self.mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        seg_results = {}
        for k, v in idx.get('results', {}).items():
            if os.path.exists(v):
                with open(v, 'r', encoding='utf-8') as f:
                    seg_results[int(k)] = json.load(f)
        sentences = build_global_timeline(seg_results, mapping, manifest)
        speakers = generate_speaker_labels(sentences)
        meta = {'source_audio': self.input_path,
                'total_duration_sec': manifest.get('total_duration', 0),
                'total_sentences': len(sentences), 'total_speakers': len(speakers)}
        save_full_json(sentences, speakers, meta, self.transcript_json)
        save_readable_transcript(sentences, self.transcript_txt)
        save_by_speaker(sentences, self.by_speaker_txt)
        print(f'[合并] {len(sentences)} 句, {len(speakers)} 个说话人')

    def _identify(self):
        print(f'\n{"="*60}\n阶段 6/8: 声纹识别\n{"="*60}')

        # 检查声纹库
        if not os.path.exists(self.voiceprint_db):
            print(f'[声纹] 声纹库不存在 ({self.voiceprint_db})，跳过')
            return

        try:
            with open(self.voiceprint_db, 'r', encoding='utf-8') as f:
                db = json.load(f)
        except Exception:
            print('[声纹] 无法读取声纹库，跳过')
            return

        if not db.get('profiles'):
            print('[声纹] 声纹库为空，跳过')
            return

        print(f'[声纹] 已注册说话人: {", ".join(db["profiles"].keys())}')

        # 加载合并后的转写
        merged_json_path = self.transcript_json
        if not os.path.exists(merged_json_path):
            raise FileNotFoundError(f'缺少合并转写: {merged_json_path}')
        with open(merged_json_path, 'r', encoding='utf-8') as f:
            merged = json.load(f)

        sentences = merged.get('sentences', [])
        if not sentences:
            print('[声纹] 无转写内容，跳过')
            return

        # 执行声纹识别
        from speaker_recognizer import identify_speakers, set_ffmpeg_bin_dir
        # 尝试复用 audio_splitter 的 ffmpeg 路径
        try:
            import audio_splitter
            if audio_splitter.FFMPEG_BIN_DIR:
                set_ffmpeg_bin_dir(audio_splitter.FFMPEG_BIN_DIR)
        except Exception:
            pass

        result = identify_speakers(
            original_audio=self.input_path,
            sentences=sentences,
            voiceprint_db_path=self.voiceprint_db,
        )

        speaker_names = result['speaker_names']

        # 打印结果
        print(f'\n[声纹] 识别结果:')
        for m in result.get('matched', []):
            print(f'  ✓ SPK_{m["speaker_id"]:02d} --> {m["name"]} '
                  f'(置信度: {m["confidence"]:.3f})')
        for u in result.get('unmatched', []):
            print(f'  ? {u["label"]} 未匹配 '
                  f'(最近: {u.get("best_match", "N/A")} @ {u["confidence"]:.3f})')
        for s in result.get('skipped', []):
            print(f'  - {s.get("label", "SPK_"+str(s.get("speaker_id","?")))} 跳过 '
                  f'({s.get("reason", "unknown")})')

        # 更新所有输出文件
        from merge_transcript import (
            generate_speaker_labels,
            save_full_json, save_readable_transcript, save_by_speaker,
        )

        for s in sentences:
            sid = s['speaker_id']
            if sid in speaker_names:
                s['speaker'] = speaker_names[sid]

        speakers = generate_speaker_labels(sentences)
        meta = merged.get('metadata', {})

        save_full_json(sentences, speakers, meta, self.transcript_json)
        save_readable_transcript(sentences, self.transcript_txt)
        save_by_speaker(sentences, self.by_speaker_txt)

        # 保存声纹映射审计文件
        vm_path = self.voiceprint_json
        with open(vm_path, 'w', encoding='utf-8') as f:
            json.dump({
                'voiceprint_db': self.voiceprint_db,
                'threshold': result.get('threshold', 0.7),
                'speaker_names': {str(k): v for k, v in speaker_names.items()},
                'matched': result.get('matched', []),
                'unmatched': result.get('unmatched', []),
                'skipped': result.get('skipped', []),
            }, f, indent=2, ensure_ascii=False)
        print(f'[声纹] 映射已保存至 {vm_path}')
        print(f'[声纹] 已更新所有输出文件中的说话人标签')

    def _chapters(self):
        print(f'\n{"="*60}\n阶段 7/8: 章节概要 + 关键人物专项分析\n{"="*60}')
        if not os.path.exists(self.transcript_txt):
            raise FileNotFoundError(f'缺少转写文本: {self.transcript_txt}')

        # 生成章节概要前先正则清洗，清洗版写入 transcript_clean.txt（原始保留）
        clean_path = os.path.join(self.output_dir, 'transcript_clean.txt')
        from transcript_cleaner import clean_transcript
        with open(self.transcript_txt, 'r', encoding='utf-8') as f:
            raw_transcript = f.read()
        cleaned = clean_transcript(raw_transcript)
        with open(clean_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f'[章节] 文本清洗: {len(raw_transcript)} → {len(cleaned)} 字符，已保存至 {clean_path}')

        # 加载 transcript JSON 获取 sentences（用于时间戳）和 speaker_names
        sentences = None
        speaker_names = None
        if os.path.exists(self.transcript_json):
            with open(self.transcript_json, 'r', encoding='utf-8') as f:
                tj = json.load(f)
            sentences = tj.get('sentences', [])
            speakers = tj.get('speakers', {})
            speaker_names = {
                int(k): v.get('label', f'SPK_{int(k):02d}')
                for k, v in speakers.items()}

        from summarizer import generate_chapter_summaries

        chapters, md = generate_chapter_summaries(
            clean_path, sentences, speaker_names)

        chapter_md_path = os.path.join(self.output_dir, 'chapter_summaries.md')
        chapter_json_path = os.path.join(self.output_dir, 'chapter_summaries.json')

        with open(chapter_md_path, 'w', encoding='utf-8') as f:
            f.write(md)
        with open(chapter_json_path, 'w', encoding='utf-8') as f:
            json.dump(chapters, f, indent=2, ensure_ascii=False)

        print(f'[章节] 共 {len(chapters)} 个章节，已保存至 {chapter_md_path}')

    def _minutes(self):
        print(f'\n{"="*60}\n阶段 8/8: 会议纪要\n{"="*60}')
        # 纪要参照章节概要生成
        chapter_md_path = os.path.join(self.output_dir, 'chapter_summaries.md')
        if not os.path.exists(chapter_md_path):
            raise FileNotFoundError(
                f'缺少章节概要: {chapter_md_path}\n'
                '[纪要] 请先运行章节概要阶段 (--start-from chapters)')
        from meeting_summarizer import summarize, save_minutes
        with open(chapter_md_path, 'r', encoding='utf-8') as f:
            chapter_outline = f.read()
        if not chapter_outline.strip():
            raise ValueError(f'章节概要为空: {chapter_md_path}')
        # 读取完整清洗转写，作为 RAG 检索源（背景知识命中更全）
        rag_text = None
        clean_path = os.path.join(self.output_dir, 'transcript_clean.txt')
        if os.path.exists(clean_path):
            with open(clean_path, 'r', encoding='utf-8') as f:
                rag_text = f.read()
        # 知识库
        kb = None
        kb_path = os.path.join(self.config_dir, 'knowledge_base.json')
        if os.path.exists(kb_path):
            try:
                from knowledge import load_knowledge_base
                kb = load_knowledge_base(kb_path)
            except ImportError:
                pass
        # 热词
        hotwords = None
        hw_path = os.path.join(self.config_dir, 'hotwords.json')
        if os.path.exists(hw_path):
            with open(hw_path, 'r', encoding='utf-8') as f:
                hotwords = json.load(f)
        minutes = summarize(chapter_outline, knowledge_base_entries=kb, hotwords=hotwords, rag_text=rag_text)
        if minutes:
            save_minutes(minutes, self.output_dir)
        else:
            raise RuntimeError('[纪要] 生成失败（summarize 返回空）')

    def _print_summary(self):
        print(f'\n{"="*60}')
        print('管道执行完毕！输出文件:')
        print(f'{"="*60}')
        for f in ['transcript.json', 'transcript.txt',
                  'by_speaker.txt', 'voiceprint.json',
                  'chapter_summaries.md', 'chapter_summaries.json']:
            p = os.path.join(self.output_dir, f)
            print(f'  {"✓" if os.path.exists(p) else "✗"} {f}')


def main():
    parser = argparse.ArgumentParser(
        description='meeting-mind — 长会议音频转写 + 说话人分离 + 章节/会议概要',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例:\n'
               '  python src/pipeline.py --input meeting.mp3\n'
               '  python src/pipeline.py --input meeting.mp3 --start-from transcribe --skip-split --skip-upload')
    parser.add_argument('--input', required=True, help='音频文件路径')
    parser.add_argument('--segment-duration', type=float, default=5280, help='分段时长(秒), 默认5280(88min)')
    parser.add_argument('--overlap', type=float, default=300, help='重叠时长(秒), 默认300(5min)')
    parser.add_argument('--speaker-count', type=int, default=15, help='说话人数量参考值')
    parser.add_argument('--language', default='zh', help='语言代码')
    parser.add_argument('--ffmpeg-bin', default='', help='ffmpeg bin目录路径')
    parser.add_argument('--start-from', default='split',
                        choices=['split', 'upload', 'transcribe', 'normalize', 'merge',
                                'identify', 'chapters', 'minutes'],
                        help='从指定阶段开始')
    parser.add_argument('--voiceprint-db', default=None,
                        help='声纹库路径，默认 speakers/voiceprint_profiles.json')
    for s in ['split', 'upload', 'transcribe', 'normalize', 'merge', 'identify', 'chapters', 'minutes']:
        parser.add_argument(f'--skip-{s}', action='store_true', help=f'跳过 {s} 阶段')
    args = parser.parse_args()

    skip = {s for s in ['split', 'upload', 'transcribe', 'normalize', 'merge',
                        'identify', 'chapters', 'minutes']
            if getattr(args, f'skip_{s}', False)}

    pipeline = FlashPipeline(
        input_path=args.input,
        segment_duration=args.segment_duration,
        overlap=args.overlap,
        speaker_count=args.speaker_count,
        language=args.language,
        voiceprint_db=args.voiceprint_db,
    )
    pipeline.run(start_from=args.start_from, skip_stages=skip, ffmpeg_bin=args.ffmpeg_bin)


if __name__ == '__main__':
    main()
