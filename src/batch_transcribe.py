"""
批量ASR转写模块：提交音频分段到FunASR进行非实时语音识别（含说话人分离）。

用法：
    python batch_transcribe.py --urls segments/urls.json --output-dir output/segments/

流程：
    1. 读取URL清单
    2. 并行提交所有分段的ASR任务
    3. 轮询等待所有任务完成
    4. 下载转写JSON结果
    5. 支持断点续传（跳过已有结果的分段）
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from http import HTTPStatus

import dashscope
from dashscope.audio.asr import Transcription

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 当前使用的非实时 ASR 模型
ASR_MODEL = 'qwen-audio-3.0-asr-flash-filetrans'


def load_hotwords(hotwords_path: str = None) -> list:
    """从JSON文件加载热词列表（兼容fun_asr_Cdmeeting格式）。"""
    if hotwords_path is None:
        hotwords_path = os.path.join(PROJECT_ROOT, 'config', 'hotwords.json')

    if not os.path.exists(hotwords_path):
        print(f'[热词] 文件 {hotwords_path} 不存在，跳过热词功能')
        return []

    try:
        with open(hotwords_path, 'r', encoding='utf-8') as f:
            hotwords = json.load(f)
        if not isinstance(hotwords, list):
            return []
        valid = [hw for hw in hotwords if isinstance(hw, dict) and 'text' in hw]
        if valid:
            print(f'[热词] 加载 {len(valid)} 个热词')
        return valid
    except Exception as e:
        print(f'[热词] 加载失败: {e}')
        return []


def setup_vocabulary(target_model: str, vocabulary: list) -> str | None:
    """创建热词表，返回 vocabulary_id；失败返回 None。"""
    if not vocabulary:
        return None
    try:
        from dashscope.audio.asr import VocabularyService
        service = VocabularyService()
        # 过滤掉 category 等非 dashscope 字段，避免 API 拒绝未知字段
        clean_vocabulary = [
            {k: hw[k] for k in ('text', 'weight', 'lang') if k in hw}
            for hw in vocabulary
        ]
        vid = service.create_vocabulary(
            prefix='mrec',
            target_model=target_model,
            vocabulary=clean_vocabulary
        )
        status = service.query_vocabulary(vid)
        if status.get('status') == 'OK':
            print(f'[热词] 热词表创建成功，ID: {vid}')
            return vid
        else:
            print(f'[热词] 热词表状态异常: {status}')
            return vid
    except Exception as e:
        print(f'[热词] 创建热词表失败: {e}，降级为不使用热词继续识别')
        return None


def delete_vocabulary(vocabulary_id: str) -> bool:
    """删除热词表，释放配额（用完即删，避免堆满每账号 10 个的上限）。"""
    if not vocabulary_id:
        return False
    try:
        from dashscope.audio.asr import VocabularyService
        VocabularyService().delete_vocabulary(vocabulary_id)
        print(f'[热词] 热词表已删除: {vocabulary_id}')
        return True
    except Exception as e:
        print(f'[热词] 删除热词表失败: {e}')
        return False


def submit_task(file_url: str, vocabulary_id: str = None,
                speaker_count: int = None, language_hints: list = None) -> str:
    """提交单个ASR任务，返回 task_id。

    Args:
        file_url: 音频文件公网URL
        vocabulary_id: 热词表ID（可选）
        speaker_count: 说话人数量参考值（可选）
        language_hints: 语种提示（可选），默认 ['zh']

    Returns:
        task_id 字符串
    """
    kwargs = {
        'model': ASR_MODEL,
        'file_urls': [file_url],
        'diarization_enabled': True,
    }

    if vocabulary_id:
        kwargs['vocabulary_id'] = vocabulary_id
    if speaker_count:
        kwargs['speaker_count'] = speaker_count
    if language_hints:
        kwargs['language_hints'] = language_hints
    else:
        kwargs['language_hints'] = ['zh']

    response = Transcription.async_call(**kwargs)

    if response.status_code != HTTPStatus.OK:
        error_msg = response.message or '未知错误'
        raise RuntimeError(f'提交任务失败: HTTP {response.status_code}, {error_msg}')

    task_id = response.output.task_id
    print(f'[提交] task_id={task_id}')
    return task_id


def poll_task(task_id: str, poll_interval: int = 30,
              max_wait: int = 7200) -> dict:
    """轮询等待单个任务完成。

    Args:
        task_id: 任务ID
        poll_interval: 轮询间隔（秒），默认30
        max_wait: 最大等待时间（秒），默认2小时

    Returns:
        任务完成的 TranscriptionResponse.output (dict)
    """
    start_time = time.time()
    status = 'PENDING'

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise TimeoutError(f'任务 {task_id} 超时（>{max_wait}s），当前状态: {status}')

        response = Transcription.fetch(task=task_id)
        status = response.output.task_status

        if status == 'SUCCEEDED':
            elapsed_str = f'{elapsed:.0f}s'
            print(f'[完成] task_id={task_id} 耗时 {elapsed_str}')
            return response.output

        if status == 'FAILED':
            # 获取失败原因
            results = getattr(response.output, 'results', [])
            error_codes = []
            for r in results:
                if hasattr(r, 'subtask_status') and r.subtask_status == 'FAILED':
                    code = getattr(r, 'code', 'unknown')
                    msg = getattr(r, 'message', 'unknown')
                    error_codes.append(f'{code}: {msg}')
            raise RuntimeError(f'任务 {task_id} 失败: {"; ".join(error_codes) or "未知错误"}')

        print(f'[轮询] task_id={task_id} 状态={status} 已等待 {elapsed:.0f}s')
        time.sleep(poll_interval)


def download_transcript(task_output, output_dir: str, seg_index: int) -> str:
    """下载转写JSON结果到本地。

    Args:
        task_output: TranscriptionOutput 对象（任务完成后的 output）
        output_dir: 输出目录
        seg_index: 分段索引

    Returns:
        本地JSON文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'segment_{seg_index:02d}.json')

    # 提取 transcription_url 并下载
    results = task_output.get('results', []) if isinstance(task_output, dict) else task_output.results
    if not results:
        raise RuntimeError(f'分段 {seg_index} 的转写结果为空')

    transcription_url = results[0].get('transcription_url') if isinstance(results[0], dict) else getattr(results[0], 'transcription_url', None)
    if not transcription_url:
        raise RuntimeError(f'分段 {seg_index} 缺少 transcription_url')

    # 下载JSON
    import urllib.request
    print(f'[下载] 分段 {seg_index}: {transcription_url[:80]}...')
    urllib.request.urlretrieve(transcription_url, output_path)

    # 补充元信息
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['_meta'] = {
        'segment_index': seg_index,
        'task_id': task_output.get('task_id', '') if isinstance(task_output, dict) else getattr(task_output, 'task_id', ''),
        'downloaded_at': datetime.now().isoformat(),
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f'[下载] 分段 {seg_index} → {output_path}')
    return output_path


class BatchTranscriber:
    """批量ASR转写器。"""

    def __init__(self, vocabulary_id: str = None, speaker_count: int = None,
                 language_hints: list = None, poll_interval: int = 30,
                 max_wait_per_task: int = 7200):
        self.vocabulary_id = vocabulary_id
        self.speaker_count = speaker_count
        self.language_hints = language_hints or ['zh']
        self.poll_interval = poll_interval
        self.max_wait_per_task = max_wait_per_task
        self.tasks = {}  # seg_index -> task_id
        self.existing = {}  # seg_index -> 已有结果路径（断点续传）

    def submit_all(self, url_list: list[dict], output_dir: str = None) -> None:
        """并行提交所有分段的ASR任务；已有结果的分段跳过提交（断点续传）。

        Args:
            url_list: URL清单中的 segments 列表
            output_dir: 转写结果输出目录，用于判断是否已有结果
        """
        print(f'\n{"="*60}')
        print(f'提交 {len(url_list)} 个ASR任务...')
        print(f'{"="*60}')

        for item in url_list:
            seg_index = item['index']
            url = item['url']
            output_path = os.path.join(output_dir, f'segment_{seg_index:02d}.json') if output_dir else None
            if output_path and os.path.exists(output_path):
                print(f'\n[断点续传] 分段 {seg_index} 已有结果，跳过提交')
                self.existing[seg_index] = output_path
                continue
            print(f'\n[分段 {seg_index}] 提交: {os.path.basename(item.get("local_file", url))}')
            task_id = submit_task(
                file_url=url,
                vocabulary_id=self.vocabulary_id,
                speaker_count=self.speaker_count,
                language_hints=self.language_hints,
            )
            self.tasks[seg_index] = task_id

        print(f'\n[提交] 全部 {len(self.tasks)} 个任务已提交（{len(self.existing)} 个跳过）')

    def wait_all(self, output_dir: str) -> dict:
        """轮询等待所有任务完成并下载结果。

        Returns:
            {seg_index: local_json_path}
        """
        print(f'\n{"="*60}')
        print(f'等待 {len(self.tasks)} 个任务完成...')
        print(f'{"="*60}')

        results = dict(self.existing)  # 断点续传的已有结果
        failed = []

        for seg_index, task_id in self.tasks.items():
            print(f'\n[分段 {seg_index}] 等待 task_id={task_id}')
            try:
                task_output = poll_task(
                    task_id,
                    poll_interval=self.poll_interval,
                    max_wait=self.max_wait_per_task,
                )
                # task_output 可能是 TranscriptionOutput 对象或 dict
                output_dict = task_output if isinstance(task_output, dict) else vars(task_output)
                local_path = download_transcript(output_dict, output_dir, seg_index)
                results[seg_index] = local_path
            except Exception as e:
                print(f'[分段 {seg_index}] 失败: {e}')
                failed.append({'index': seg_index, 'task_id': task_id, 'error': str(e)})

        if failed:
            print(f'\n[警告] {len(failed)} 个任务失败:')
            for f in failed:
                print(f'  分段 {f["index"]}: {f["error"]}')

        print(f'\n[完成] 成功: {len(results)}/{len(self.tasks)}')
        return results


def main():
    parser = argparse.ArgumentParser(description='批量提交FunASR录音文件识别任务')
    parser.add_argument('--urls', required=True,
                        help='URL清单JSON路径 (segments/urls.json)')
    parser.add_argument('--output-dir', default=None,
                        help='转写结果输出目录，默认 output/segments/')
    parser.add_argument('--speaker-count', type=int, default=None,
                        help='说话人数量参考值（2-100）')
    parser.add_argument('--language', default='zh',
                        help='语言代码，默认 zh')
    parser.add_argument('--poll-interval', type=int, default=30,
                        help='轮询间隔（秒），默认30')
    parser.add_argument('--max-wait', type=int, default=7200,
                        help='单个任务最大等待时间（秒），默认7200（2h）')
    parser.add_argument('--hotwords', default=None,
                        help='热词JSON文件路径')
    parser.add_argument('--vocabulary-id', default=None,
                        help='已有的热词表ID（跳过创建）')
    args = parser.parse_args()

    # dashscope 配置
    dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY')
    if not dashscope.api_key:
        print('[错误] 未设置 DASHSCOPE_API_KEY 环境变量')
        sys.exit(1)

    # 可选：设置业务空间专属域名
    workspace_id = os.environ.get('DASHSCOPE_WORKSPACE_ID')
    if workspace_id:
        dashscope.base_http_api_url = f'https://{workspace_id}.cn-beijing.maas.aliyuncs.com/api/v1'

    # 输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, 'output', 'segments')

    # 加载URL清单
    with open(args.urls, 'r', encoding='utf-8') as f:
        url_data = json.load(f)
    url_list = url_data['segments']

    # 热词处理（记录是否本次新建，用于用完即删）
    vocabulary_id = args.vocabulary_id
    created_vocabulary = False
    if not vocabulary_id:
        hotwords = load_hotwords(args.hotwords)
        if hotwords:
            vocabulary_id = setup_vocabulary(ASR_MODEL, hotwords)
            created_vocabulary = vocabulary_id is not None

    # 提交并等待
    transcriber = BatchTranscriber(
        vocabulary_id=vocabulary_id,
        speaker_count=args.speaker_count,
        language_hints=[args.language],
        poll_interval=args.poll_interval,
        max_wait_per_task=args.max_wait,
    )

    try:
        transcriber.submit_all(url_list, args.output_dir)
        results = transcriber.wait_all(args.output_dir)

        # 保存转写结果索引
        index_path = os.path.join(args.output_dir, 'index.json')
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': {str(k): v for k, v in results.items()},
                'url_list': url_list,
                'completed_at': datetime.now().isoformat(),
            }, f, indent=2, ensure_ascii=False)

        print(f'\n转写结果索引已保存至 {index_path}')
    finally:
        if created_vocabulary and vocabulary_id:
            delete_vocabulary(vocabulary_id)


if __name__ == '__main__':
    main()
