"""
说话人声纹注册CLI工具。

用法：
    # 按文件夹批量注册（文件夹名=姓名，平均目录内所有WAV）
    python enroll_speaker.py --folder "speakers/zhangsan" --name "张三"

    # 单文件注册（手动指定姓名）
    python enroll_speaker.py --name "张三" --audio speaker1.wav --audio speaker2.wav

    # 批量注册父目录下所有子文件夹
    python enroll_speaker.py --batch "speakers"

    # 查看已注册列表
    python enroll_speaker.py --list

    # 删除
    python enroll_speaker.py --remove "张三"

    # 调整阈值
    python enroll_speaker.py --threshold 0.75
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from speaker_recognizer import (
    VoiceprintProfiles,
    SpeakerRecognizer,
    set_ffmpeg_bin_dir,
)

DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, 'speakers', 'voiceprint_profiles.json')


def _scan_wav_files(folder: str) -> list[str]:
    """扫描目录中所有 .wav / .WAV 文件，按文件名排序。"""
    wavs = []
    if not os.path.isdir(folder):
        print(f'[错误] 目录不存在: {folder}')
        sys.exit(1)
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith('.wav'):
            wavs.append(os.path.join(folder, fname))
    return wavs


def enroll_speaker(name: str, audio_files: list[str],
                   db_path: str = DEFAULT_DB_PATH) -> None:
    """注册单个说话人：从多个WAV文件提取声纹并平均。

    Args:
        name: 说话人姓名
        audio_files: WAV文件路径列表
        db_path: 声纹库路径
    """
    if not audio_files:
        print(f'[错误] 未提供注册音频文件')
        sys.exit(1)

    # 验证文件存在
    for f in audio_files:
        if not os.path.exists(f):
            print(f'[错误] 文件不存在: {f}')
            sys.exit(1)

    print(f'\n{"="*60}')
    print(f'注册说话人: {name}')
    print(f'{"="*60}')
    print(f'  音频文件: {len(audio_files)} 个')

    # 提取声纹
    recognizer = SpeakerRecognizer()
    embeddings = []
    for i, f in enumerate(audio_files):
        fname = os.path.basename(f)
        print(f'  [{i+1}/{len(audio_files)}] 提取: {fname}')
        try:
            emb = recognizer.extract_embedding(f)
            embeddings.append(emb)
            print(f'      维度: {emb.shape}, 范围: [{emb.min():.4f}, {emb.max():.4f}]')
        except Exception as e:
            print(f'      失败: {e}')
            continue

    if not embeddings:
        print(f'\n[错误] 所有音频文件提取失败，注册中止')
        sys.exit(1)

    # 平均
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-10)  # L2归一化

    # 计算两两相似度（质量检查）
    if len(embeddings) >= 2:
        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dot = np.dot(embeddings[i], embeddings[j])
                denom = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                sims.append(float(dot / denom) if denom > 0 else 0.0)
        avg_sim = sum(sims) / len(sims)
        min_sim = min(sims)
        print(f'\n  样本间相似度: 平均={avg_sim:.4f}, 最低={min_sim:.4f}')
        if min_sim < 0.5:
            print(f'  [警告] 最低相似度偏低，建议检查音频质量或剔除异常样本')

    # 存入声纹库
    profiles = VoiceprintProfiles(db_path)
    profiles.add_profile(name, avg_emb, audio_files)

    print(f'\n[完成] {name} 已注册到 {db_path}')


def enroll_folder(folder: str, name: str = None,
                  db_path: str = DEFAULT_DB_PATH) -> None:
    """按文件夹注册：扫描目录内所有WAV，文件夹名作为说话人姓名。

    Args:
        folder: 包含WAV文件的目录
        name: 说话人姓名（None=使用文件夹名）
        db_path: 声纹库路径
    """
    wavs = _scan_wav_files(folder)
    if not wavs:
        print(f'[错误] 目录中未找到 .wav 文件: {folder}')
        sys.exit(1)

    if name is None:
        name = os.path.basename(os.path.normpath(folder))

    print(f'[文件夹注册] {folder}')
    print(f'  姓名: {name}')
    print(f'  文件: {len(wavs)} 个')
    for w in wavs:
        print(f'    - {os.path.basename(w)}')

    enroll_speaker(name, wavs, db_path)


def batch_enroll(parent_dir: str, db_path: str = DEFAULT_DB_PATH) -> None:
    """批量注册父目录下所有子文件夹（每个子文件夹=一个说话人）。

    Args:
        parent_dir: 包含多个说话人子文件夹的父目录
        db_path: 声纹库路径
    """
    if not os.path.isdir(parent_dir):
        print(f'[错误] 目录不存在: {parent_dir}')
        sys.exit(1)

    subdirs = []
    for entry in sorted(os.listdir(parent_dir)):
        full = os.path.join(parent_dir, entry)
        if os.path.isdir(full):
            subdirs.append(full)

    if not subdirs:
        print(f'[错误] 未找到子文件夹: {parent_dir}')
        sys.exit(1)

    print(f'\n批量注册: {parent_dir}')
    print(f'  发现 {len(subdirs)} 个子文件夹:')
    for d in subdirs:
        print(f'    - {os.path.basename(d)}')

    for folder in subdirs:
        name = os.path.basename(folder)
        wavs = _scan_wav_files(folder)
        if not wavs:
            print(f'\n[跳过] {name}: 无 .wav 文件')
            continue
        enroll_speaker(name, wavs, db_path)

    # 汇总
    profiles = VoiceprintProfiles(db_path)
    print(f'\n{"="*60}')
    print(f'批量注册完成！共 {len(profiles.profiles)} 人')
    print(f'{"="*60}')
    for n in profiles.list_names():
        p = profiles.get_profile(n)
        print(f'  {n}: {p["sample_count"]} 个样本')


def list_speakers(db_path: str = DEFAULT_DB_PATH) -> None:
    """列出所有已注册说话人。"""
    profiles = VoiceprintProfiles(db_path)

    if not profiles.profiles:
        print('声纹库为空')
        return

    print(f'\n已注册说话人 ({len(profiles.profiles)} 人):')
    print(f'{"─"*60}')
    print(f'{"姓名":<12} {"样本数":<8} {"注册时间":<22} {"模型"}')
    print(f'{"─"*60}')
    for name in profiles.list_names():
        p = profiles.get_profile(name)
        print(f'{name:<12} {p["sample_count"]:<8} '
              f'{p.get("enrolled_at", "N/A")[:19]:<22} '
              f'{profiles.model_name.split("/")[-1]}')
    print(f'{"─"*60}')
    print(f'模型: {profiles.model_name}')
    print(f'阈值: {profiles.threshold}')
    print(f'库文件: {db_path}')


def remove_speaker(name: str, db_path: str = DEFAULT_DB_PATH) -> None:
    """删除说话人。"""
    profiles = VoiceprintProfiles(db_path)
    if profiles.remove_profile(name):
        print(f'已删除: {name}')
    else:
        print(f'未找到: {name}')
        sys.exit(1)


def set_threshold(threshold: float, db_path: str = DEFAULT_DB_PATH) -> None:
    """更新匹配阈值。"""
    profiles = VoiceprintProfiles(db_path)
    profiles.threshold = threshold
    profiles._save()
    print(f'阈值已更新为: {threshold}')


# ── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='meeting-mind 声纹注册工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例:\n'
               '  python enroll_speaker.py --folder "speakers/zhangsan" --name "张三"\n'
               '  python enroll_speaker.py --batch "speakers"\n'
               '  python enroll_speaker.py --name "张三" --audio a.wav --audio b.wav\n'
               '  python enroll_speaker.py --list')
    parser.add_argument('--name', default=None, help='说话人姓名（单文件注册时使用）')
    parser.add_argument('--audio', action='append', default=None,
                        help='WAV文件路径（可重复，用于单文件注册）')
    parser.add_argument('--folder', default=None,
                        help='文件夹路径（文件夹名=姓名，扫描目录内所有WAV）')
    parser.add_argument('--batch', default=None,
                        help='父目录路径（批量注册所有子文件夹）')
    parser.add_argument('--db', default=DEFAULT_DB_PATH,
                        help=f'声纹库路径，默认 {DEFAULT_DB_PATH}')
    parser.add_argument('--list', action='store_true', help='列出已注册说话人')
    parser.add_argument('--remove', default=None, help='删除指定说话人')
    parser.add_argument('--threshold', type=float, default=None,
                        help='设置匹配阈值 (0-1)')
    parser.add_argument('--ffmpeg-bin-dir', default='',
                        help='ffmpeg可执行文件目录')

    args = parser.parse_args()

    # ffmpeg 路径
    if args.ffmpeg_bin_dir:
        set_ffmpeg_bin_dir(args.ffmpeg_bin_dir)

    # 分发命令
    if args.list:
        list_speakers(args.db)
    elif args.remove:
        remove_speaker(args.remove, args.db)
    elif args.threshold is not None:
        set_threshold(args.threshold, args.db)
    elif args.batch:
        batch_enroll(args.batch, args.db)
    elif args.folder:
        enroll_folder(args.folder, name=args.name, db_path=args.db)
    elif args.name and args.audio:
        enroll_speaker(args.name, args.audio, args.db)
    else:
        parser.print_help()
        print('\n[提示] 请指定 --folder / --batch / --name+--audio / --list / --remove')


if __name__ == '__main__':
    main()
