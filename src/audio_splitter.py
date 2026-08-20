"""
音频分割模块：使用 ffmpeg 将长音频切分为带重叠的短片段。

用法：
    python audio_splitter.py --input meeting_7h.mp3 --segment-duration 5400 --overlap 60

输出：
    - segments/ 目录下的分段音频文件
    - segments/manifest.json 分段清单
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ffmpeg 可执行文件路径，可通过 --ffmpeg-bin-dir 参数覆盖
FFMPEG_BIN_DIR = ''

# 降级重编码时，按输出扩展名选择兼容的编码器
_FALLBACK_CODEC = {
    '.wav': 'pcm_s16le',
    '.m4a': 'aac',
    '.mp4': 'aac',
    '.aac': 'aac',
    '.mp3': 'libmp3lame',
    '.flac': 'flac',
}


def _ffmpeg(tool: str) -> str:
    """返回 ffmpeg/ffprobe 的完整路径。"""
    if FFMPEG_BIN_DIR:
        exe = os.path.join(FFMPEG_BIN_DIR, f'{tool}.exe')
        if os.path.exists(exe):
            return exe
    return tool


def get_audio_duration(input_path: str) -> float:
    """使用 ffprobe 获取音频时长（秒）。"""
    cmd = [
        _ffmpeg('ffprobe'), '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'json',
        input_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return float(info['format']['duration'])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        print(f'[错误] 无法获取音频时长: {e}')
        print('[提示] 请确认 ffmpeg/ffprobe 已安装并在 PATH 中')
        sys.exit(1)


def get_audio_format(input_path: str) -> str:
    """获取音频文件扩展名（用于确定容器格式）。"""
    return Path(input_path).suffix.lower()


def split_audio(input_path: str, segment_duration: float, overlap: float,
                output_dir: str) -> list[dict]:
    """
    将音频分割为带重叠的片段。

    Args:
        input_path: 输入音频文件路径
        segment_duration: 每段时长（秒），不含重叠
        overlap: 相邻段重叠时长（秒）
        output_dir: 输出目录

    Returns:
        分段清单 list[dict]，每项含 file, start_time, end_time, duration
    """
    total_duration = get_audio_duration(input_path)
    ext = get_audio_format(input_path)
    base_name = Path(input_path).stem

    os.makedirs(output_dir, exist_ok=True)

    segments = []
    seg_index = 0
    current_start = 0.0

    while current_start < total_duration:
        # 每段实际时长 = 基础时长 + 重叠（最后一段不加重叠）
        is_last = (current_start + segment_duration) >= total_duration
        actual_duration = (total_duration - current_start) if is_last else (segment_duration + overlap)

        seg_name = f'{base_name}_seg{seg_index:02d}_{int(current_start)}s{ext}'
        seg_path = os.path.join(output_dir, seg_name)

        # ffmpeg 命令：从 current_start 开始截取 actual_duration 秒，
        # 统一转 16kHz 单声道（ASR 说话人分离要求单声道），codec 按输出扩展名匹配
        codec = _FALLBACK_CODEC.get(ext, 'aac')
        cmd = [
            _ffmpeg('ffmpeg'), '-y',  # -y 覆盖已有文件
            '-ss', str(current_start),
            '-t', str(actual_duration),
            '-i', input_path,
            '-ar', '16000', '-ac', '1',
            '-acodec', codec,
        ]
        if codec != 'pcm_s16le':  # 有损/压缩编码才指定码率；pcm 为固定比特率
            cmd += ['-b:a', '192k']
        cmd.append(seg_path)

        print(f'[分割] 片段 {seg_index}: {current_start:.0f}s ~ {current_start + actual_duration:.0f}s '
              f'(时长 {actual_duration:.0f}s)')

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        segment_info = {
            'index': seg_index,
            'file': seg_path,
            'start_time': current_start,
            'end_time': current_start + actual_duration,
            'duration': actual_duration,
            'is_last': is_last,
        }
        segments.append(segment_info)

        # 下一段起点 = 当前起点 + 基础段长（不含重叠部分）
        current_start += segment_duration
        seg_index += 1

    # 写入清单
    manifest_path = os.path.join(output_dir, 'manifest.json')
    manifest = {
        'source_file': input_path,
        'total_duration': total_duration,
        'segment_duration': segment_duration,
        'overlap': overlap,
        'segment_count': len(segments),
        'segments': segments,
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'\n[分割] 完成！共 {len(segments)} 个片段，清单已保存至 {manifest_path}')
    return segments


def main():
    parser = argparse.ArgumentParser(description='将长音频分割为带重叠的短片段')
    parser.add_argument('--input', required=True, help='输入音频文件路径')
    parser.add_argument('--segment-duration', type=float, default=5280,
                        help='每段基础时长（秒），默认5280（88分钟）')
    parser.add_argument('--overlap', type=float, default=300,
                        help='相邻段重叠时长（秒），默认300（5分钟）')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录，默认为项目根目录下的 segments/')
    parser.add_argument('--ffmpeg-bin-dir', default='',
                        help='ffmpeg/ffprobe 可执行文件所在目录')
    args = parser.parse_args()

    global FFMPEG_BIN_DIR
    FFMPEG_BIN_DIR = args.ffmpeg_bin_dir

    if args.output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output_dir = os.path.join(project_root, 'segments')

    split_audio(args.input, args.segment_duration, args.overlap, args.output_dir)


if __name__ == '__main__':
    main()
