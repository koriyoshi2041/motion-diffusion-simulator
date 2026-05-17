"""
─────────────────────────────────────────────────────────────────────────
data_utils.py —— HumanML3D 263 维特征 ↔ (T, 22, 3) 关节坐标 互转
─────────────────────────────────────────────────────────────────────────

HumanML3D 把每个动作存成 (T, 263) 的特征向量。本模块把它「解码」回 22 个
3D 关节坐标，给 visualizer / 度量模块用。

263 维的精确切分（来自 HumanML3D 论文 & EricGuo5513/HumanML3D 代码）：

    索引          含义
    [0]           root_y_rotational_velocity （根关节绕 y 轴的角速度，rad）
    [1:3]         root_xz_linear_velocity     （根关节在地面的速度）
    [3]           root_y_height               （根关节的离地高度）
    [4:67]        21 个非根关节的相对位置 （21 × 3）
    [67:193]      21 个非根关节的 6D 旋转 （21 × 6）
    [193:259]    22 个关节的线速度         （22 × 3）
    [259:263]    4 个脚-地接触二元 flag

解码思路（与官方 recover_from_ric 一致）：
    1. 把根关节的角速度积分 → 全局朝向
    2. 把根关节的 xz 速度按 当前朝向 旋转 → 全局位移
    3. 21 个非根关节的相对位置叠加上根位姿 → 全局关节坐标
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np


def quat_from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    half = angle / 2.0
    s = np.sin(half)
    return np.stack([np.cos(half), axis[..., 0]*s, axis[..., 1]*s, axis[..., 2]*s], axis=-1)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    tx = 2 * (qy*vz - qz*vy)
    ty = 2 * (qz*vx - qx*vz)
    tz = 2 * (qx*vy - qy*vx)
    rx = vx + qw*tx + (qy*tz - qz*ty)
    ry = vy + qw*ty + (qz*tx - qx*tz)
    rz = vz + qw*tz + (qx*ty - qy*tx)
    return np.stack([rx, ry, rz], axis=-1)


def recover_joints_from_263(features: np.ndarray) -> np.ndarray:
    """(T, 263) -> (T, 22, 3) joint xyz, integrating root pose over time."""
    if features.ndim != 2 or features.shape[1] != 263:
        raise ValueError(f"Expected (T, 263), got {features.shape}")
    T = features.shape[0]

    rot_vel = features[:, 0]
    lin_vel_xz = features[:, 1:3]
    root_y = features[:, 3]
    rel_joints = features[:, 4:67].reshape(T, 21, 3)

    # Integrate root yaw and (x, z) translation.
    yaws = np.cumsum(rot_vel)
    yaws = np.concatenate([[0.0], yaws[:-1]])
    cos, sin = np.cos(yaws), np.sin(yaws)
    rot_xz = np.stack([
        np.stack([cos,  sin], axis=-1),
        np.stack([-sin, cos], axis=-1),
    ], axis=-2)                                                   # (T, 2, 2)

    delta_world = np.einsum("tij,tj->ti", rot_xz, lin_vel_xz)
    root_xz = np.cumsum(delta_world, axis=0)
    root_xz = np.concatenate([np.zeros((1, 2)), root_xz[:-1]], axis=0)

    root = np.stack([root_xz[:, 0], root_y, root_xz[:, 1]], axis=-1)  # (T, 3)

    # Rotate the 21 relative joints by the root yaw.
    full_rot = np.zeros((T, 3, 3))
    full_rot[:, 0, 0] = cos
    full_rot[:, 0, 2] = sin
    full_rot[:, 1, 1] = 1.0
    full_rot[:, 2, 0] = -sin
    full_rot[:, 2, 2] = cos
    rotated = np.einsum("tij,tnj->tni", full_rot, rel_joints)

    joints = np.zeros((T, 22, 3))
    joints[:, 0] = root
    joints[:, 1:] = rotated + root[:, None, :]
    return joints


def feature_summary_per_motion(features: np.ndarray) -> np.ndarray:
    """Compress (T, 263) -> (D,) summary statistic for KDE-based KL/JS."""
    mean = features.mean(0)
    std  = features.std(0)
    minv = features.min(0)
    maxv = features.max(0)
    return np.concatenate([mean, std, minv, maxv])


if __name__ == "__main__":
    fake = np.random.RandomState(0).randn(60, 263) * 0.05
    j = recover_joints_from_263(fake)
    print("joints:", j.shape)
    print("summary:", feature_summary_per_motion(fake).shape)
