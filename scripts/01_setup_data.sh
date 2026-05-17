#!/usr/bin/env bash
# 集群一键准备：克隆 4 个 baseline 仓库、下载权重、准备 HumanML3D
# 在 qiyuan 集群上 ssh qiyuan (172.16.1.48 能访问外网) 上执行

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
mkdir -p "${PROJECT_ROOT}/baselines"
mkdir -p "${PROJECT_ROOT}/data"
cd "${PROJECT_ROOT}/baselines"

# ------------------------------------------------------------------
# 1) HY-Motion 1.0 (主 SOTA)
# ------------------------------------------------------------------
if [[ ! -d HY-Motion-1.0 ]]; then
  echo "[1/4] cloning HY-Motion-1.0 ..."
  git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git
fi
cd HY-Motion-1.0
pip install -r requirements.txt

# 模型权重（约 30GB）— 用 huggingface-cli
mkdir -p ckpts
huggingface-cli download tencent/HY-Motion-1.0       --include "HY-Motion-1.0/*"      --local-dir ckpts/tencent
huggingface-cli download tencent/HY-Motion-1.0       --include "HY-Motion-1.0-Lite/*" --local-dir ckpts/tencent
huggingface-cli download openai/clip-vit-large-patch14            --local-dir ckpts/clip-vit-large-patch14/
huggingface-cli download Qwen/Qwen3-8B                            --local-dir ckpts/Qwen3-8B
huggingface-cli download Text2MotionPrompter/Text2MotionPrompter  --local-dir ckpts/Text2MotionPrompter || true

cd "${PROJECT_ROOT}/baselines"

# ------------------------------------------------------------------
# 2) MoMask (CVPR 2024 强基线)
# ------------------------------------------------------------------
if [[ ! -d MoMask ]]; then
  echo "[2/4] cloning MoMask ..."
  git clone https://github.com/EricGuo5513/momask-codes.git MoMask
fi
cd MoMask
pip install -r requirements.txt || true
bash prepare/download_models.sh   # 上游脚本 — 自动下载 evaluator + checkpoint
cd "${PROJECT_ROOT}/baselines"

# ------------------------------------------------------------------
# 3) MDM (ICLR 2023 经典)
# ------------------------------------------------------------------
if [[ ! -d motion-diffusion-model ]]; then
  echo "[3/4] cloning MDM ..."
  git clone https://github.com/GuyTevet/motion-diffusion-model.git
fi
cd motion-diffusion-model
pip install git+https://github.com/openai/CLIP.git || true
python -m spacy download en_core_web_sm || true
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
# 训练好的 ckpt
mkdir -p save/humanml_trans_enc_512 && cd save/humanml_trans_enc_512
wget -nc https://huggingface.co/GuyTevet/motion-diffusion-model/resolve/main/humanml_trans_enc_512.zip
unzip -n humanml_trans_enc_512.zip && rm humanml_trans_enc_512.zip
cd "${PROJECT_ROOT}/baselines"

# ------------------------------------------------------------------
# 4) HumanML3D 数据 (训练 / 评测共用)
# ------------------------------------------------------------------
cd "${PROJECT_ROOT}/data"
if [[ ! -d HumanML3D ]]; then
  echo "[4/4] cloning HumanML3D dataset repo ..."
  git clone https://github.com/EricGuo5513/HumanML3D.git
fi
echo
echo "提示：HumanML3D 内容受 AMASS 协议限制，需要按其 README 注册下载 AMASS 后用脚本生成。"
echo "    集群 NFS 上 /nfsdata/wxu/datasets/ 可能已有共享拷贝，先 ls 看一下。"

cd "${PROJECT_ROOT}"
echo
echo "✅ 全部 baseline 仓库与权重准备完成"
echo "   下一步：python scripts/02_run_baselines.py --baselines hy_motion mdm --gpu 0"
