"""
OSS上传模块：将音频分段上传到阿里云OSS并生成公网访问URL。

用法：
    python oss_uploader.py --manifest segments/manifest.json

配置：
    config/oss_config.json — OSS连接信息
    环境变量 OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET 可覆盖配置文件中的凭证
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'oss_config.json')


def load_oss_config(config_path: str = None) -> dict:
    """加载OSS配置，环境变量优先于配置文件。"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        print(f'[错误] OSS配置文件 {config_path} 不存在')
        print('[提示] 请创建 config/oss_config.json，参考模板：')
        print('  {')
        print('    "endpoint": "oss-cn-beijing.aliyuncs.com",')
        print('    "bucket_name": "fun-asr-meeting",')
        print('    "access_key_id": "your-key",')
        print('    "access_key_secret": "your-secret"')
        print('  }')
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 环境变量覆盖
    config['access_key_id'] = os.environ.get('OSS_ACCESS_KEY_ID', config.get('access_key_id', ''))
    config['access_key_secret'] = os.environ.get('OSS_ACCESS_KEY_SECRET', config.get('access_key_secret', ''))

    required = ['endpoint', 'bucket_name', 'access_key_id', 'access_key_secret']
    for key in required:
        if not config.get(key):
            print(f'[错误] OSS配置缺少必填项: {key}')
            print('[提示] 请在 config/oss_config.json 中填写或设置环境变量')
            sys.exit(1)

    return config


class OSSUploader:
    """阿里云OSS上传器，支持进度回调。"""

    def __init__(self, config: dict):
        self.endpoint = config['endpoint']
        self.bucket_name = config['bucket_name']
        self.access_key_id = config['access_key_id']
        self.access_key_secret = config['access_key_secret']
        self.bucket = None

    def connect(self):
        """建立OSS连接。"""
        try:
            import oss2
            auth = oss2.Auth(self.access_key_id, self.access_key_secret)
            self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)
            print(f'[OSS] 已连接，Bucket: {self.bucket_name}')
        except ImportError:
            print('[错误] 缺少 oss2 库，请运行: pip install oss2')
            sys.exit(1)
        except Exception as e:
            print(f'[错误] OSS连接失败: {e}')
            print('[提示] 请检查 config/oss_config.json 中的配置是否正确')
            sys.exit(1)

    def upload_file(self, local_path: str, oss_key: str = None,
                    progress_callback=None) -> str:
        """上传单个文件到OSS，返回公网签名URL。

        Args:
            local_path: 本地文件路径
            oss_key: OSS中的对象名，默认使用文件名
            progress_callback: 进度回调 (bytes_uploaded, total_bytes) -> None

        Returns:
            带签名的公网URL（有效期48小时）
        """
        if oss_key is None:
            oss_key = os.path.basename(local_path)

        file_size = os.path.getsize(local_path)
        print(f'[OSS] 上传: {os.path.basename(local_path)} → oss://{self.bucket_name}/{oss_key} '
              f'({file_size / 1024 / 1024:.1f} MB)')

        start_time = time.time()

        def _progress_callback(bytes_consumed, total_bytes):
            if progress_callback:
                progress_callback(bytes_consumed, total_bytes)
            elif total_bytes:
                pct = bytes_consumed * 100 // total_bytes
                if pct % 20 == 0:
                    elapsed = time.time() - start_time
                    speed = bytes_consumed / elapsed / 1024 if elapsed > 0 else 0
                    print(f'  进度: {pct}% ({bytes_consumed / 1024 / 1024:.1f}/{total_bytes / 1024 / 1024:.1f} MB, '
                          f'{speed:.0f} KB/s)')

        try:
            self.bucket.put_object_from_file(
                oss_key, local_path,
                progress_callback=_progress_callback
            )
        except Exception as e:
            print(f'[错误] 上传失败: {e}')
            raise

        elapsed = time.time() - start_time
        print(f'[OSS] 上传完成，耗时 {elapsed:.0f}s')

        # 生成签名URL（48小时 = 172800秒）
        url = self.bucket.sign_url('GET', oss_key, 172800)
        return url


def main():
    parser = argparse.ArgumentParser(description='上传音频分段到OSS')
    parser.add_argument('--manifest', required=True,
                        help='分段清单JSON路径 (segments/manifest.json)')
    parser.add_argument('--config', default=None,
                        help='OSS配置文件路径，默认 config/oss_config.json')
    parser.add_argument('--output', default=None,
                        help='URL清单输出路径，默认 segments/urls.json')
    args = parser.parse_args()

    # 加载分段清单
    if not os.path.exists(args.manifest):
        print(f'[错误] 清单文件 {args.manifest} 不存在')
        sys.exit(1)

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # 加载OSS配置
    oss_config = load_oss_config(args.config)

    # 上传
    uploader = OSSUploader(oss_config)
    uploader.connect()

    url_list = []
    for seg in manifest['segments']:
        local_path = seg['file']
        if not os.path.exists(local_path):
            print(f'[跳过] 文件不存在: {local_path}')
            continue

        oss_key = f"meeting_segments/{os.path.basename(local_path)}"
        url = uploader.upload_file(local_path, oss_key)
        url_list.append({
            'index': seg['index'],
            'local_file': local_path,
            'oss_key': oss_key,
            'url': url,
            'start_time': seg['start_time'],
            'end_time': seg['end_time'],
        })

    # 保存URL清单
    if args.output is None:
        output_dir = os.path.dirname(args.manifest)
        args.output = os.path.join(output_dir, 'urls.json')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({'segments': url_list, 'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')},
                  f, indent=2, ensure_ascii=False)

    print(f'\n[OSS] 全部上传完成！URL清单已保存至 {args.output}')
    print(f'[OSS] URL有效期: 48小时')


if __name__ == '__main__':
    main()
