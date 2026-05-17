"""
Fine-tune HY-Motion 1.0 on a custom dataset (Motion-X++ by default).

Designed for multi-node × multi-GPU launch via cluster/run_distributed.sh.
This is a thin wrapper that:
    1. Constructs a HY-Motion Trainer using its internal class
    2. Plugs in our dataloader (motion-x-plus-plus)
    3. Calls trainer.fit()

Run via:
    torchrun --nnodes=2 --nproc_per_node=8 ...   scripts/05_finetune_hy.py \
        --data_root /nfsdata/wxu/datasets/motion-x-plus-plus \
        --output_dir /nfsdata/wxu/checkpoints/hy_motion_ft \
        --base_ckpt baselines/HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0 \
        --batch_size 8 --learning_rate 1e-5 --num_train_steps 50000

NOTE: HY-Motion's official trainer API is at hymotion.network.trainer.
We hold the wrapper minimal so the upstream code can change without
breaking us.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import torch


def _add_hy_motion_to_path(project_root: Path) -> Path:
    repo = project_root / "baselines" / "HY-Motion-1.0"
    if not repo.exists():
        raise FileNotFoundError(f"HY-Motion repo missing: {repo}")
    sys.path.insert(0, str(repo))
    return repo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       type=Path, required=True)
    p.add_argument("--output_dir",      type=Path, required=True)
    p.add_argument("--base_ckpt",       type=Path, required=True)
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--learning_rate",   type=float, default=1e-5)
    p.add_argument("--num_train_steps", type=int,   default=50000)
    p.add_argument("--eval_every",      type=int,   default=2000)
    p.add_argument("--log_every",       type=int,   default=50)
    p.add_argument("--save_every",      type=int,   default=5000)
    p.add_argument("--project_root",    type=Path,  default=Path.cwd())
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repo = _add_hy_motion_to_path(args.project_root)

    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        torch.distributed.init_process_group(backend="nccl")

    # Lazy import so that path is set first
    from hymotion.network.trainer import HyMotionTrainer       # noqa: E402
    from hymotion.utils.dataset   import MotionXPPDataset      # noqa: E402

    train_ds = MotionXPPDataset(root=args.data_root, split="train")
    val_ds   = MotionXPPDataset(root=args.data_root, split="val")

    trainer = HyMotionTrainer.from_pretrained(
        base_ckpt=str(args.base_ckpt),
        output_dir=str(args.output_dir),
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_steps=args.num_train_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
