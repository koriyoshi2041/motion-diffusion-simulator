#!/usr/bin/env bash
# 多机多卡分布式 fine-tune HY-Motion 1.0 在 Motion-X++ 上
#
# 假设：
#   - 已 ssh qiyuan，cd ${REMOTE_DIR}
#   - 6 台机器（172.16.1.{35,36,37,39,40,48}），每台 8× A800-80GB
#   - NFS 共享路径：/nfsdata/wxu/<user>/motion_diffusion_simulator
#   - 数据 Motion-X++ 已放 /nfsdata/wxu/datasets/motion-x-plus-plus/
#
# 我们用 torchrun 多机多卡。MASTER_ADDR 用 35 号，其它机器作为 worker。
set -euo pipefail

NNODES=${NNODES:-2}                 # 默认先两台测试
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-172.16.1.35}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}            # 在每台机器上设置 0,1,2,...

DATA_ROOT=${DATA_ROOT:-/nfsdata/wxu/datasets/motion-x-plus-plus}
OUT_DIR=${OUT_DIR:-/nfsdata/wxu/checkpoints/hy_motion_ft}

source /nfsdata/wxu/miniconda3/etc/profile.d/conda.sh
conda activate linear

# 训练脚本调用 HY-Motion 内部 trainer (我们提供 wrapper)
torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  scripts/05_finetune_hy.py \
    --data_root  ${DATA_ROOT} \
    --output_dir ${OUT_DIR} \
    --base_ckpt  baselines/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_steps 50000 \
    --eval_every 2000 \
    --log_every 50 \
    --save_every 5000

echo "✅ rank=${NODE_RANK} done."
