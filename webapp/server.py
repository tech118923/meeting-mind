"""
meeting-mind 配置管理 Web 工具（纯 Python 标准库，零依赖）。

管理 config/hotwords.json 与 config/knowledge_base.json：
    - 热词：分类(category) / 权重 / 语言 / 搜索 / 增删改
    - 知识库：类型 / 优先级 / 关键词 / 搜索 / 增删改

用法：
    python webapp/server.py            # 启动后自动打开浏览器
    python webapp/server.py --port 9000  # 指定端口

仅监听 127.0.0.1，供本机使用，不暴露到外网。
"""

import argparse
import json
import os
import shutil
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# 各配置文件：读/写路径 + 写回时缩进（与原文件保持一致，减小 diff）
FILES = {
    'hotwords': {'path': os.path.join(CONFIG_DIR, 'hotwords.json'), 'indent': 2},
    'knowledge': {'path': os.path.join(CONFIG_DIR, 'knowledge_base.json'), 'indent': 4},
}


def _read_json(key: str):
    path = FILES[key]['path']
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_json(key: str, data) -> str:
    """写回 JSON，先备份 .bak，返回文件路径。"""
    cfg = FILES[key]
    if os.path.exists(cfg['path']):
        try:
            shutil.copy2(cfg['path'], cfg['path'] + '.bak')
        except OSError:
            pass
    with open(cfg['path'], 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=cfg['indent'])
    return cfg['path']


class Handler(BaseHTTPRequestHandler):
    server_version = 'meeting-mind/1.0'

    # ── 工具 ──────────────────────────────────────

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text, status=200, ctype='text/plain; charset=utf-8'):
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict | None:
        try:
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length)
            return json.loads(raw)
        except Exception as e:
            print(f'[错误] 解析请求体失败: {e}')
            return None

    def log_message(self, fmt, *args):
        print(f'[server] {self.address_string()} {fmt % args}')

    # ── 路由 ──────────────────────────────────────

    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/':
            self._serve_index()
        elif path == '/api/hotwords':
            self._send_json(_read_json('hotwords'))
        elif path == '/api/knowledge':
            self._send_json(_read_json('knowledge'))
        else:
            self._send_text('404 Not Found', status=404)

    def do_POST(self):
        path = self.path.split('?')[0]
        if path == '/api/hotwords':
            data = self._read_body()
            if data is None or not isinstance(data, list):
                self._send_json({'ok': False, 'error': '数据格式错误，应为数组'}, status=400)
                return
            try:
                saved = _write_json('hotwords', data)
            except Exception as e:
                self._send_json({'ok': False, 'error': str(e)}, status=500)
                return
            self._send_json({'ok': True, 'path': saved, 'count': len(data)})
        elif path == '/api/knowledge':
            data = self._read_body()
            if data is None or not isinstance(data, list):
                self._send_json({'ok': False, 'error': '数据格式错误，应为数组'}, status=400)
                return
            try:
                saved = _write_json('knowledge', data)
            except Exception as e:
                self._send_json({'ok': False, 'error': str(e)}, status=500)
                return
            self._send_json({'ok': True, 'path': saved, 'count': len(data)})
        else:
            self._send_json({'ok': False, 'error': '404 Not Found'}, status=404)

    def _serve_index(self):
        idx = os.path.join(STATIC_DIR, 'index.html')
        if not os.path.exists(idx):
            self._send_text('index.html 缺失', status=500)
            return
        with open(idx, 'r', encoding='utf-8') as f:
            self._send_text(f.read(), ctype='text/html; charset=utf-8')


def main():
    parser = argparse.ArgumentParser(description='meeting-mind 配置管理 Web 工具（热词/知识库）')
    parser.add_argument('--port', type=int, default=8787, help='端口，默认 8787')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    args = parser.parse_args()

    url = f'http://127.0.0.1:{args.port}'
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print('=' * 50)
    print('meeting-mind 配置管理')
    print(f'  打开: {url}')
    print(f'  管理: config/hotwords.json + config/knowledge_base.json')
    print('  按 Ctrl+C 停止')
    print('=' * 50)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n已停止')
        sys.exit(0)


if __name__ == '__main__':
    main()
