#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────
# run-local.sh —— 一条命令在本地把整个 demo 跑起来
#
# 1) 装 Python 依赖
# 2) 启 mock 后端 :8888 （不需要 GPU，回放预先缓存的真模型输出）
# 3) 启前端静态服务 :7777
# 4) 自动打开浏览器到 prompt-puppet.html
#
# Ctrl-C 停止全部进程。
# ────────────────────────────────────────────────────────────────────────
set -e

PY=${PYTHON:-python3}
PORT_BACK=${PORT_BACK:-8888}
PORT_FRONT=${PORT_FRONT:-7777}
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

echo "[1/4] installing Python deps (fastapi/uvicorn/numpy)…"
$PY -m pip install --quiet --disable-pip-version-check \
  fastapi 'uvicorn[standard]' numpy pydantic 2>/dev/null || \
$PY -m pip install --user --quiet --disable-pip-version-check \
  fastapi 'uvicorn[standard]' numpy pydantic

echo "[2/4] starting mock backend on :$PORT_BACK …"
PORT=$PORT_BACK $PY mock_server.py > /tmp/mds-backend.log 2>&1 &
BACK_PID=$!
trap "kill $BACK_PID 2>/dev/null; kill \$FRONT_PID 2>/dev/null; exit" INT TERM
sleep 1
if ! kill -0 $BACK_PID 2>/dev/null; then
  echo "  backend failed to start. log:"
  cat /tmp/mds-backend.log
  exit 1
fi

echo "[3/4] starting frontend static server on :$PORT_FRONT …"
( cd frontend && $PY -m http.server $PORT_FRONT ) > /tmp/mds-frontend.log 2>&1 &
FRONT_PID=$!
sleep 1

URL="http://localhost:$PORT_FRONT/prompt-puppet.html?api=http://localhost:$PORT_BACK/api/generate"
echo "[4/4] opening $URL"
if   command -v open       >/dev/null; then open "$URL"
elif command -v xdg-open   >/dev/null; then xdg-open "$URL"
elif command -v start      >/dev/null; then start "$URL"
fi

echo
echo "ready · backend pid=$BACK_PID · frontend pid=$FRONT_PID"
echo "Ctrl-C to stop."
wait $BACK_PID $FRONT_PID
