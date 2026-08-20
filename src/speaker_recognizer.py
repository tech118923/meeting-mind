"""
声纹识别核心模块：声纹注册库管理 + CAM++声纹提取 + 说话人重识别。

用法：
    # 识别（pipeline 调用）
    from speaker_recognizer import identify_speakers
    result = identify_speakers(original_audio, sentences, voiceprint_db_path)

    # 声纹库管理
    from speaker_recognizer import VoiceprintProfiles, SpeakerRecognizer
    profiles = VoiceprintProfiles('speakers/voiceprint_profiles.json')
    recognizer = SpeakerRecognizer()
    emb = recognizer.extract_embedding('speaker.wav')
    profiles.add_profile('张三', emb, ['speaker.wav'])
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import numpy as np

# ── 常量 ──────────────────────────────────────────────

MIN_SEGMENT_MS = 2000      # 单段最低时长 (ms)，短于此值的片段跳过
MIN_TOTAL_MS = 5000         # 说话人累计最低发言时长 (ms)，低于此值跳过匹配（5秒以上尝试匹配）
DEFAULT_MODEL = 'iic/speech_campplus_sv_zh-cn_16k-common'
DEFAULT_THRESHOLD = 0.7

# ffmpeg 可执行文件路径，沿用 audio_splitter 的查找方式
_FFMPEG_BIN_DIR = ''


def _ffmpeg(tool: str) -> str:
    """返回 ffmpeg/ffprobe 的完整路径。"""
    if _FFMPEG_BIN_DIR:
        exe = os.path.join(_FFMPEG_BIN_DIR, f'{tool}.exe')
        if os.path.exists(exe):
            return exe
    return tool


def set_ffmpeg_bin_dir(bin_dir: str) -> None:
    """设置 ffmpeg 可执行文件目录（跨模块共享）。"""
    global _FFMPEG_BIN_DIR
    if bin_dir and os.path.isdir(bin_dir):
        _FFMPEG_BIN_DIR = bin_dir


# ── 声纹库管理 ─────────────────────────────────────────

class VoiceprintProfiles:
    """JSON文件持久化的声纹注册库。

    存储格式:
        {
          "model": "iic/speech_campplus_sv_zh-cn_16k-common",
          "threshold": 0.7,
          "profiles": {
            "张三": {
              "embedding_b64": "AAAA...==",
              "enrollment_audio": ["speaker1.wav"],
              "sample_count": 1,
              "enrolled_at": "2026-07-27T16:00:00"
            }
          }
        }

    内存中的 profile['embedding'] 为 numpy float32 数组，不持久化到文件。
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model_name = DEFAULT_MODEL
        self.threshold = DEFAULT_THRESHOLD
        self.profiles: dict[str, dict] = {}  # name -> {embedding, enrollment_audio, ...}
        self._load()

    # ── 文件读写 ──────────────────────────────────

    def _load(self) -> None:
        """从JSON文件加载声纹库，自动base64解码embedding。"""
        if not os.path.exists(self.db_path):
            print(f'[声纹库] 文件不存在，初始化空库: {self.db_path}')
            return

        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f'[声纹库] 读取失败 ({e})，使用空库')
            return

        self.model_name = data.get('model', DEFAULT_MODEL)
        self.threshold = data.get('threshold', DEFAULT_THRESHOLD)

        for name, profile in data.get('profiles', {}).items():
            b64 = profile.get('embedding_b64', '')
            if not b64:
                print(f'[声纹库] 跳过 {name}: 缺少 embedding_b64')
                continue
            try:
                embedding = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
            except Exception as e:
                print(f'[声纹库] 跳过 {name}: base64解码失败 ({e})')
                continue

            self.profiles[name] = {
                'embedding': embedding,
                'enrollment_audio': profile.get('enrollment_audio', []),
                'sample_count': profile.get('sample_count', 0),
                'enrolled_at': profile.get('enrolled_at', ''),
            }

    def _save(self) -> None:
        """将声纹库编码为base64后写入JSON文件。"""
        profiles_out = {}
        for name, profile in self.profiles.items():
            emb = profile.get('embedding')
            if emb is None:
                continue
            b64 = base64.b64encode(emb.astype(np.float32).tobytes()).decode('ascii')
            profiles_out[name] = {
                'embedding_b64': b64,
                'enrollment_audio': profile.get('enrollment_audio', []),
                'sample_count': profile.get('sample_count', 0),
                'enrolled_at': profile.get('enrolled_at', ''),
            }

        data = {
            'model': self.model_name,
            'threshold': self.threshold,
            'profiles': profiles_out,
        }

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── 公共接口 ──────────────────────────────────

    def add_profile(self, name: str, embedding: np.ndarray,
                    enrollment_audio: list[str] = None) -> None:
        """添加或更新说话人声纹。

        Args:
            name: 说话人姓名
            embedding: 192维声纹向量 (np.float32)
            enrollment_audio: 注册音频文件路径列表
        """
        self.profiles[name] = {
            'embedding': embedding.astype(np.float32).copy(),
            'enrollment_audio': enrollment_audio or [],
            'sample_count': len(enrollment_audio) if enrollment_audio else 1,
            'enrolled_at': datetime.now().isoformat(),
        }
        self._save()
        print(f'[声纹库] 已注册: {name}')

    def get_profile(self, name: str) -> dict | None:
        """获取说话人声纹信息（含 decoded embedding）。"""
        return self.profiles.get(name)

    def remove_profile(self, name: str) -> bool:
        """删除说话人声纹。返回 True 表示删除成功。"""
        if name in self.profiles:
            del self.profiles[name]
            self._save()
            print(f'[声纹库] 已删除: {name}')
            return True
        return False

    def list_names(self) -> list[str]:
        """返回所有已注册说话人姓名（排序）。"""
        return sorted(self.profiles.keys())

    def match(self, query_embedding: np.ndarray,
              threshold: float = None) -> tuple[str | None, float]:
        """将查询声纹与库中所有声纹做余弦相似度比对。

        Args:
            query_embedding: 查询声纹向量 (192,)
            threshold: 匹配阈值，None则使用库默认值

        Returns:
            (best_name, best_score) — best_name为None表示无匹配
        """
        if threshold is None:
            threshold = self.threshold

        if not self.profiles:
            return None, 0.0

        best_name = None
        best_score = -1.0

        # L2归一化查询向量
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        for name, profile in self.profiles.items():
            ref_emb = profile.get('embedding')
            if ref_emb is None:
                continue
            ref_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-10)
            sim = float(np.dot(query_norm, ref_norm))

            if sim > best_score:
                best_score = sim
                best_name = name

        if best_score >= threshold:
            return best_name, best_score
        return None, best_score


# ── CAM++ 声纹提取 ─────────────────────────────────────

class SpeakerRecognizer:
    """CAM++声纹模型包装器，提取192维speaker embedding。

    模型首次使用时自动下载（约30MB），之后缓存在ModelScope目录。
    """

    def __init__(self, model_name: str = None, device: str = 'cpu'):
        self.model_name = model_name or DEFAULT_MODEL
        self.device = device
        self._model = None
        self._safe_path_cache: dict[str, str] = {}  # 中文路径 → ASCII临时副本
        self._temp_files: list[str] = []  # 追踪生成的临时副本（cleanup 用）

    def _ensure_model(self):
        """懒加载CAM++模型。"""
        if self._model is not None:
            return

        try:
            from funasr import AutoModel
        except ImportError:
            print('[错误] 缺少 funasr 库，请运行: pip install funasr')
            sys.exit(1)

        print(f'[CAM++] 加载模型: {self.model_name} (首次使用将自动下载约30MB)...')
        t0 = time.time()
        self._model = AutoModel(
            model=self.model_name,
            device=self.device,
            disable_pbar=True,
        )
        print(f'[CAM++] 模型就绪，耗时 {time.time() - t0:.1f}s')

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """从WAV音频文件提取192维声纹向量。

        Args:
            audio_path: 16kHz单声道WAV文件路径

        Returns:
            np.ndarray shape (192,) dtype float32
        """
        self._ensure_model()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f'音频文件不存在: {audio_path}')

        result = self._model.generate(input=audio_path)
        if not result or len(result) == 0:
            raise RuntimeError(f'声纹提取失败: {audio_path} (模型返回空结果)')

        emb = result[0].get('spk_embedding')
        if emb is None:
            raise RuntimeError(f'声纹提取失败: {audio_path} (缺少 spk_embedding)')

        return np.array(emb, dtype=np.float32).flatten()

    def _get_safe_audio_path(self, source_audio: str) -> str | None:
        """获取ffmpeg兼容的音频路径（处理中文路径问题）。

        MinGW编译的ffmpeg无法处理含非ASCII字符的路径。
        此方法检测路径是否含中文等字符，若是则复制到ASCII临时文件。
        同一文件的副本会被缓存，后续调用直接复用。
        """
        abs_path = os.path.abspath(source_audio)
        if abs_path in self._safe_path_cache:
            return self._safe_path_cache[abs_path]

        # 检测是否含非ASCII字符
        try:
            abs_path.encode('ascii')
            # 纯ASCII，直接使用
            self._safe_path_cache[abs_path] = abs_path
            return abs_path
        except UnicodeEncodeError:
            pass

        # 含非ASCII字符，复制到临时ASCII路径
        import shutil
        suffix = os.path.splitext(source_audio)[1]
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, prefix='fp_safe_', delete=False)
        safe_path = tmp.name
        tmp.close()
        try:
            print(f'[ffmpeg兼容] 复制音频到: {os.path.basename(safe_path)}')
            shutil.copy2(abs_path, safe_path)
            self._safe_path_cache[abs_path] = safe_path
            self._temp_files.append(safe_path)
            return safe_path
        except Exception as e:
            print(f'[ffmpeg兼容] 复制失败: {e}')
            try:
                os.unlink(safe_path)
            except OSError:
                pass
            return None

    def cleanup(self) -> None:
        """删除所有生成的临时音频副本。"""
        for p in self._temp_files:
            try:
                os.unlink(p)
            except OSError:
                pass
        self._temp_files.clear()
        self._safe_path_cache.clear()

    def extract_embedding_from_segment(
            self, source_audio: str, start_ms: float, end_ms: float,
            temp_dir: str = None) -> np.ndarray | None:
        """从长音频的指定时间片段中提取声纹向量。

        Args:
            source_audio: 原始音频文件路径
            start_ms: 片段起始时间（毫秒）
            end_ms: 片段结束时间（毫秒）
            temp_dir: 临时文件目录（None=系统临时目录）

        Returns:
            np.ndarray (192,) 或 None（片段太短或提取失败）
        """
        duration_ms = end_ms - start_ms
        if duration_ms < MIN_SEGMENT_MS:
            return None

        start_sec = start_ms / 1000.0
        duration_sec = duration_ms / 1000.0

        # 处理中文路径：ffmpeg MinGW编译版无法处理非ASCII路径，
        # 复制源文件到ASCII临时路径（仅首次）
        safe_source = self._get_safe_audio_path(source_audio)
        if safe_source is None:
            return None

        # 创建临时WAV文件（16kHz mono PCM，CAM++要求）
        tmp = tempfile.NamedTemporaryFile(
            suffix='.wav', dir=temp_dir, delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            cmd = [
                _ffmpeg('ffmpeg'), '-y',
                '-ss', str(start_sec),
                '-t', str(duration_sec),
                '-i', safe_source,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-loglevel', 'error',
                tmp_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True, check=True,
                           encoding='utf-8', errors='replace')

            emb = self.extract_embedding(tmp_path)
            return emb
        except subprocess.CalledProcessError as e:
            print(f'[ffmpeg] 切分失败 ({start_ms}-{end_ms}ms): {e.stderr.strip() if e.stderr else e}')
            return None
        except Exception as e:
            print(f'[声纹] 提取失败 ({start_ms}-{end_ms}ms): {e}')
            return None
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── 识别编排 ───────────────────────────────────────────

def identify_speakers(
        original_audio: str,
        sentences: list[dict],
        voiceprint_db_path: str,
        threshold: float = None,
        temp_dir: str = None,
        sample_limit: int = 20,
) -> dict:
    """对转写结果中的每个说话人做声纹识别。

    Args:
        original_audio: 原始会议音频文件路径
        sentences: 转写句子列表，每项含 speaker, speaker_id, start_ms, end_ms, text
        voiceprint_db_path: 声纹库JSON文件路径
        threshold: 匹配阈值覆盖（None=使用库默认值）
        temp_dir: 临时文件目录
        sample_limit: 每个说话人最多采样的片段数（控制耗时）

    Returns:
        {
            'speaker_names': {speaker_id_int: 'display_name'},
            'matched': [{'speaker_id': int, 'name': str, 'confidence': float}, ...],
            'unmatched': [{'speaker_id': int, 'label': str, 'best_match': str|null, 'confidence': float}, ...],
            'skipped': [{'speaker_id': int, 'reason': str}, ...],
            'threshold': float,
        }
    """
    # ── 加载声纹库 ──
    profiles = VoiceprintProfiles(voiceprint_db_path)
    if not profiles.profiles:
        print('[识别] 声纹库为空，跳过')
        return {
            'speaker_names': {},
            'matched': [], 'unmatched': [], 'skipped': [],
            'threshold': threshold or profiles.threshold,
        }

    # ── 按speaker_id分组 ──
    speaker_segments: dict[int, list[dict]] = {}
    for s in sentences:
        sid = s.get('speaker_id')
        if sid is None:
            continue
        if sid not in speaker_segments:
            speaker_segments[sid] = []
        speaker_segments[sid].append({
            'start_ms': s['start_ms'],
            'end_ms': s['end_ms'],
        })

    # ── 对每个说话人提取声纹并匹配 ──
    recognizer = SpeakerRecognizer()
    matched = []
    unmatched = []
    skipped = []
    speaker_names: dict[int, str] = {}

    total_speakers = len(speaker_segments)
    try:
        for idx, (sid, segments) in enumerate(sorted(speaker_segments.items())):
            label = f'SPK_{sid:02d}'

            # 筛选足够长的片段
            valid_segs = [seg for seg in segments
                          if (seg['end_ms'] - seg['start_ms']) >= MIN_SEGMENT_MS]
            total_ms = sum(seg['end_ms'] - seg['start_ms'] for seg in valid_segs)

            if total_ms < MIN_TOTAL_MS:
                print(f'[{idx+1}/{total_speakers}] {label}: 发言时长不足 '
                      f'({total_ms/1000:.1f}s < {MIN_TOTAL_MS/1000:.0f}s)，跳过')
                skipped.append({
                    'speaker_id': sid,
                    'label': label,
                    'reason': 'insufficient_audio',
                    'total_ms': total_ms,
                    'segment_count': len(valid_segs),
                })
                speaker_names[sid] = label
                continue

            # 采样：取最长的 sample_limit 个片段
            valid_segs.sort(key=lambda s: s['end_ms'] - s['start_ms'], reverse=True)
            sample_segs = valid_segs[:sample_limit]

            print(f'[{idx+1}/{total_speakers}] {label}: 提取声纹 '
                  f'({len(sample_segs)}/{len(valid_segs)} 个片段, '
                  f'总时长 {total_ms/1000:.1f}s)...')

            embeddings = []
            for seg in sample_segs:
                emb = recognizer.extract_embedding_from_segment(
                    original_audio, seg['start_ms'], seg['end_ms'], temp_dir)
                if emb is not None:
                    embeddings.append(emb)

            if not embeddings:
                print(f'  → 未能提取有效声纹，跳过')
                skipped.append({
                    'speaker_id': sid,
                    'label': label,
                    'reason': 'extraction_failed',
                    'total_ms': total_ms,
                })
                speaker_names[sid] = label
                continue

            # 平均 + L2归一化
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-10)

            # 匹配
            best_name, best_score = profiles.match(avg_emb, threshold)

            if best_name is not None:
                print(f'  ✓ 匹配: {best_name} (置信度: {best_score:.4f})')
                matched.append({
                    'speaker_id': sid,
                    'name': best_name,
                    'confidence': round(best_score, 4),
                    'segment_count': len(embeddings),
                })
                speaker_names[sid] = best_name
            else:
                close_name = None
                close_score = 0.0
                # 找最接近的（即使未达阈值）
                for name, profile in profiles.profiles.items():
                    ref = profile.get('embedding')
                    if ref is None:
                        continue
                    ref_norm = ref / (np.linalg.norm(ref) + 1e-10)
                    sim = float(np.dot(avg_emb, ref_norm))
                    if sim > close_score:
                        close_score = sim
                        close_name = name

                print(f'  ? 未匹配 (最近: {close_name} @ {close_score:.4f}, '
                      f'阈值: {threshold or profiles.threshold})')
                unmatched.append({
                    'speaker_id': sid,
                    'label': label,
                    'best_match': close_name,
                    'confidence': round(close_score, 4),
                    'segment_count': len(embeddings),
                })
                speaker_names[sid] = label
    finally:
        recognizer.cleanup()

    return {
        'speaker_names': speaker_names,
        'matched': matched,
        'unmatched': unmatched,
        'skipped': skipped,
        'threshold': threshold or profiles.threshold,
    }
