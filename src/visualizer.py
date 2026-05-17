"""
─────────────────────────────────────────────────────────────────────────
visualizer.py —— 4 种动作可视化（作业第 10 条「自然语言 → 示意图」）
─────────────────────────────────────────────────────────────────────────

输入：(T, 22, 3) 关节坐标张量（HumanML3D / SMPL-22 顺序，单位米）
输出：4 种与数据对应、可解释的「示意图」：

    1. 3D 火柴人动画（mp4/gif）—— 主可视化
    2. 关节强度热图（22 关节 × 时间）—— 看哪个肢体动得最多
    3. 俯视轨迹图（root xz + facing 箭头）—— 看路径
    4. 运动能量曲线（每帧 ‖Δjoint‖² 之和） —— 看动作的爆发点

为什么是这 4 种？
    它们覆盖了 (空间, 时间, 部位) 三个维度，让评委只看一张图就能判断
    「模型有没有正确响应 prompt」。
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# HumanML3D 22-joint SMPL kinematic chain
KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],         # right leg
    [0, 1, 4, 7, 10],         # left leg
    [0, 3, 6, 9, 12, 15],     # spine + head
    [9, 14, 17, 19, 21],      # right arm
    [9, 13, 16, 18, 20],      # left arm
]


def render_stickfigure_video(
    joints: np.ndarray,
    out_path: str | Path,
    fps: int = 20,
    title: str = "",
    dpi: int = 100,
    figsize=(6, 6),
):
    """Render a (T, 22, 3) sequence to mp4 (or gif if path ends .gif)."""
    out_path = Path(out_path)
    T = joints.shape[0]

    mins = joints.reshape(-1, 3).min(0) - 0.2
    maxs = joints.reshape(-1, 3).max(0) + 0.2

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=15, azim=-70)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[2], maxs[2])
    ax.set_zlim(mins[1], maxs[1])
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title)
    ax.grid(False)

    def _frame(t):
        ax.clear()
        ax.view_init(elev=15, azim=-70)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[2], maxs[2])
        ax.set_zlim(mins[1], maxs[1])
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"{title}  t={t:>3}/{T}")
        ax.grid(False)
        for chain in KINEMATIC_CHAIN:
            xs = joints[t, chain, 0]
            ys = joints[t, chain, 2]
            zs = joints[t, chain, 1]
            ax.plot(xs, ys, zs, lw=3)
        ax.scatter(joints[t, :, 0], joints[t, :, 2], joints[t, :, 1], s=15)

    anim = FuncAnimation(fig, _frame, frames=T, interval=1000 / fps)
    if out_path.suffix == ".gif":
        anim.save(out_path, writer=PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)


def render_joint_angle_heatmap(joints: np.ndarray, out_path: str | Path, title: str = ""):
    """Approximate "joint angle" by per-joint velocity magnitude across time."""
    vel = np.linalg.norm(np.diff(joints, axis=0), axis=-1)        # (T-1, 22)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(vel.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("frame")
    ax.set_ylabel("joint id (0–21, SMPL order)")
    ax.set_title(f"Joint motion intensity — {title}")
    fig.colorbar(im, ax=ax, label="‖Δjoint‖")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_topdown_trajectory(joints: np.ndarray, out_path: str | Path, title: str = ""):
    """Top-down view of root joint (x, z) with facing-direction arrows."""
    root = joints[:, 0, :]                                        # (T, 3)
    x, z = root[:, 0], root[:, 2]
    # facing: vector from left-hip (1) to right-hip (2), rotated 90 deg in xz
    hips = joints[:, 2, :] - joints[:, 1, :]
    facing = np.stack([hips[:, 2], -hips[:, 0]], axis=-1)
    facing /= np.linalg.norm(facing, axis=-1, keepdims=True) + 1e-8

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, z, "-", lw=1.5, alpha=0.8)
    step = max(1, len(x) // 20)
    ax.quiver(x[::step], z[::step], facing[::step, 0], facing[::step, 1],
              scale=18, width=0.005, color="crimson")
    ax.scatter(x[0], z[0], c="green", s=80, marker="o", label="start")
    ax.scatter(x[-1], z[-1], c="red", s=80, marker="X", label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"Top-down trajectory — {title}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_energy_spectrogram(joints: np.ndarray, out_path: str | Path, title: str = ""):
    """Per-frame total kinetic 'energy' = sum of squared joint velocities."""
    vel = np.diff(joints, axis=0)                                  # (T-1, 22, 3)
    energy = (vel ** 2).sum(axis=(1, 2))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(energy, lw=1.6, color="darkorange")
    ax.fill_between(np.arange(len(energy)), energy, alpha=0.3, color="darkorange")
    ax.set_xlabel("frame")
    ax.set_ylabel("Σ‖Δjoint‖²")
    ax.set_title(f"Motion-energy curve — {title}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_all(joints: np.ndarray, out_dir: str | Path, tag: str = "demo", fps: int = 20):
    """One-shot: produce all four visualisations into `out_dir/<tag>_*`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_stickfigure_video(joints, out_dir / f"{tag}_stickfigure.mp4", fps=fps, title=tag)
    render_joint_angle_heatmap(joints, out_dir / f"{tag}_jointmap.png", title=tag)
    render_topdown_trajectory(joints, out_dir / f"{tag}_topdown.png", title=tag)
    render_energy_spectrogram(joints, out_dir / f"{tag}_energy.png", title=tag)
    return {
        "video":   str(out_dir / f"{tag}_stickfigure.mp4"),
        "joints":  str(out_dir / f"{tag}_jointmap.png"),
        "topdown": str(out_dir / f"{tag}_topdown.png"),
        "energy":  str(out_dir / f"{tag}_energy.png"),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fake = np.cumsum(rng.standard_normal((90, 22, 3)) * 0.05, axis=0)
    print(render_all(fake, "/tmp/motion_demo", tag="dummy"))
