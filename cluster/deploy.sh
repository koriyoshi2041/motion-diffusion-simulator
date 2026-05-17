#!/usr/bin/env bash
# 一键把项目部署到 qiyuan 集群（只需在 macOS 本地跑一次）
#
# 步骤：
#   1) rsync 把项目同步到集群 NFS （/nfsdata/wxu/<user>/motion_diffusion_simulator/）
#   2) 进集群 48（能上外网），跑 01_setup_data.sh 装依赖+下权重
#   3) 启动 tmux 会话，用户随时 attach
#
# 使用：
#   bash cluster/deploy.sh
#   bash cluster/deploy.sh --skip-data    # 已经装过权重时
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-wxu}"
REMOTE_HOST="${REMOTE_HOST:-qiyuan}"        # ssh alias 已配 (172.16.1.48)
REMOTE_DIR="${REMOTE_DIR:-/nfsdata/wxu/motion_diffusion_simulator}"
SKIP_DATA=false

for arg in "$@"; do
  case $arg in
    --skip-data) SKIP_DATA=true ;;
  esac
done

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "[deploy] local : ${LOCAL_DIR}"
echo "[deploy] remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

# 1) sync code (skip baselines/ckpts/runs/data — 都很大或集群独有)
rsync -av --progress \
  --exclude '.git' \
  --exclude 'baselines' \
  --exclude 'ckpts' \
  --exclude 'data' \
  --exclude 'runs' \
  --exclude 'venv' \
  --exclude '__pycache__' \
  "${LOCAL_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# 2) remote setup
ssh "${REMOTE_USER}@${REMOTE_HOST}" "
  set -euo pipefail
  cd ${REMOTE_DIR}
  source /nfsdata/wxu/miniconda3/etc/profile.d/conda.sh
  conda activate linear
  if [ '${SKIP_DATA}' = 'false' ]; then
    bash scripts/01_setup_data.sh
  fi
  pip install -r requirements.txt
"

# 3) start tmux session for user to attach later
ssh -t "${REMOTE_USER}@${REMOTE_HOST}" "
  tmux new-session -d -s motion -c ${REMOTE_DIR} || true
  tmux send-keys -t motion 'source /nfsdata/wxu/miniconda3/etc/profile.d/conda.sh && conda activate linear' C-m
  echo 'tmux session [motion] ready. attach via:  ssh ${REMOTE_HOST} -t tmux a -t motion'
"

echo
echo "✅ 部署完成。下一步:"
echo "   ssh ${REMOTE_HOST} -t tmux a -t motion"
echo "   python scripts/02_run_baselines.py --baselines hy_motion mdm --prompts prompts/chinese_demo.txt --translate"
