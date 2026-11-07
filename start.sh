#!/usr/bin/env bash
# macOS / Linux 启动脚本：bash start.sh
set -e
cd "$(dirname "$0")"

PY=""
if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python; fi
if [ -z "$PY" ]; then
  echo "[!] 没找到 Python。先装 3.11 或更新版本：https://www.python.org/downloads/" >&2
  exit 1
fi

if [ ! -f config.json ]; then
  cp config.example.json config.json
  echo "[i] 已从示例生成 config.json（默认 mock 模式：不联网、不用 API key）"
fi

echo "[i] 启动中，按 Ctrl+C 停止"
exec "$PY" -u -m autonomous_agent