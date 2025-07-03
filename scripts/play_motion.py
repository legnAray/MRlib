#!/usr/bin/env python3
# from isaacgym.torch_utils import *
# 在导入任何库之前设置环境变量
import os

os.environ["OMP_NUM_THREADS"] = "1"
import sys

sys.path.append(os.getcwd())

import torch
import numpy as np
import pickle  # 使用Python标准库pickle
import glob
import time
from collections import Counter

# mrlib_core imports
import mrlib.poselib.core.rotation3d as pRot
import mrlib.utils.rotation_conversions as sRot
from mrlib.utils import torch_utils
from mrlib.poselib.skeleton.skeleton3d import (
    SkeletonTree,
    SkeletonMotion,
    SkeletonState,
)
from mrlib.smpllib.smpl_parser import SMPL_Parser
from mrlib.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
)

# project-specific imports
from mrlib.utils.torch_humanoid_batch import Humanoid_Batch

# third-party imports
from scipy.spatial.transform import Rotation as scipyRot
import hydra
from omegaconf import DictConfig
import mujoco
import mujoco.viewer


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )


def print_dataset_structure(motion_data):
    """打印数据集结构信息"""
    print(f"\n{'='*60}")
    print(f"📊 数据集结构分析")
    print(f"{'='*60}")

    print(f"🗂️  数据集类型: {type(motion_data).__name__}")

    if isinstance(motion_data, dict):
        print(f"📁 动作数量: {len(motion_data)} 个")
        print(
            f"📝 动作列表: {list(motion_data.keys())[:5]}{'...' if len(motion_data) > 5 else ''}"
        )

        # 分析第一个动作的结构
        first_key = list(motion_data.keys())[0]
        first_motion = motion_data[first_key]
        print(f"\n🎯 以 '{first_key}' 为例分析数据结构:")

    else:
        # 单个动作数据
        first_motion = motion_data
        print(f"📁 单个动作数据")
        print(f"\n🎯 数据结构分析:")

    print(f"📋 数据字段:")
    for key, value in first_motion.items():
        if isinstance(value, np.ndarray):
            print(f"   {key:25s}: {str(value.shape):15s} {value.dtype}")
        elif isinstance(value, dict):
            print(f"   {key:25s}: dict (包含 {len(value)} 个字段)")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, np.ndarray):
                    print(
                        f"     └─ {sub_key:20s}: {str(sub_value.shape):15s} {sub_value.dtype}"
                    )
                else:
                    print(f"     └─ {sub_key:20s}: {type(sub_value).__name__}")
        elif isinstance(value, (int, float)):
            print(f"   {key:25s}: {type(value).__name__} = {value}")
        else:
            print(f"   {key:25s}: {type(value).__name__}")

    # 分析动作时长信息
    if isinstance(motion_data, dict):
        total_frames = sum(
            motion["dof"].shape[0] for motion in motion_data.values() if "dof" in motion
        )
        fps = first_motion.get("fps", 30)
        total_duration = total_frames / fps

        print(f"\n⏱️  时长统计:")
        print(f"   总帧数: {total_frames} 帧")
        print(f"   总时长: {total_duration:.2f} 秒")
        print(f"   帧率: {fps} FPS")

        # 显示各动作的时长
        print(f"\n📏 各动作时长:")
        for i, (key, motion) in enumerate(motion_data.items()):
            if i >= 10:  # 只显示前10个
                print(f"   ... (还有 {len(motion_data) - 10} 个动作)")
                break
            frames = motion["dof"].shape[0] if "dof" in motion else 0
            duration = frames / fps
            print(f"   {key:30s}: {frames:4d} 帧 ({duration:5.2f}s)")
    else:
        frames = first_motion["dof"].shape[0] if "dof" in first_motion else 0
        fps = first_motion.get("fps", 30)
        duration = frames / fps
        print(f"\n⏱️  时长信息:")
        print(f"   帧数: {frames} 帧")
        print(f"   时长: {duration:.2f} 秒")
        print(f"   帧率: {fps} FPS")

    # 检查特殊字段
    special_fields = []
    if "task_info" in first_motion:
        special_fields.append("✅ task_info (任务信息)")
    if "joint_local_velocities" in first_motion:
        special_fields.append("✅ joint_local_velocities (关节角速度)")
    if "base_lin_vel_local" in first_motion:
        special_fields.append("✅ base_lin_vel_local (机器人局部线速度)")
    if "default_joint_angles" in first_motion:
        special_fields.append("✅ default_joint_angles (默认关节角度)")
    if "task_vel" in first_motion:
        special_fields.append("✅ task_vel (目标线速度)")

    if special_fields:
        print(f"\n🎁 特殊字段:")
        for field in special_fields:
            print(f"   {field}")

    print(f"{'='*60}\n")


def analyze_motion_stats(motion_data, current_motion_key=None):
    """分析动作序列的统计信息

    Args:
        motion_data: 运动数据字典
        current_motion_key: 如果指定，只分析当前动作；否则分析所有动作
    """
    print("\n" + "=" * 80)
    if current_motion_key:
        print(f"📊 当前动作统计分析: {current_motion_key}")
    else:
        print("📊 所有动作序列统计分析")
    print("=" * 80)

    # 根据参数决定分析哪些动作
    if current_motion_key and current_motion_key in motion_data:
        motions_to_analyze = {current_motion_key: motion_data[current_motion_key]}
    else:
        motions_to_analyze = motion_data

    for motion_key, motion in motions_to_analyze.items():
        print(f"\n🎬 动作: {motion_key}")
        print("-" * 60)

        # 检查数据可用性
        has_task_vel = "task_vel" in motion
        has_task_info = "task_info" in motion
        has_base_vel = "base_lin_vel_local" in motion

        print(f"📋 数据可用性:")
        print(f"   task_vel: {'✅' if has_task_vel else '❌'}")
        print(f"   task_info: {'✅' if has_task_info else '❌'}")
        print(f"   base_lin_vel_local: {'✅' if has_base_vel else '❌'}")

        # 优先使用task_vel，其次使用task_info
        target_vel = None
        if has_task_vel:
            # 新格式：使用task_vel (只有前进和横向，没有角速度)
            task_vel_2d = motion["task_vel"]  # [N, 2]
            # 如果有task_info，补充角速度；否则填0
            if has_task_info:
                task_info = motion["task_info"]
                angular_vel = task_info["target_vel"][:, 2:3]  # [N, 1]
                target_vel = np.hstack([task_vel_2d, angular_vel])  # [N, 3]
            else:
                # 补充0角速度
                target_vel = np.hstack(
                    [task_vel_2d, np.zeros((len(task_vel_2d), 1))]
                )  # [N, 3]
            print(f"🎯 使用task_vel数据源")
        elif has_task_info:
            # 兼容旧格式：使用task_info
            task_info = motion["task_info"]
            target_vel = task_info["target_vel"]
            print(f"🎯 使用task_info数据源")
        else:
            print("❌ 无速度数据，跳过速度分析")

        # 分析目标速度
        if target_vel is not None:
            # 计算最大速度
            min_vel_x = np.min(target_vel[:, 0])
            max_vel_x = np.max(target_vel[:, 0])
            min_vel_y = np.min(target_vel[:, 1])
            max_vel_y = np.max(target_vel[:, 1])
            min_vel_z = np.min(target_vel[:, 2])
            max_vel_z = np.max(target_vel[:, 2])

            print(f"\n🎯 目标速度分析:")
            print(f"   X轴(前进): 范围=[{min_vel_x:.3f}, {max_vel_x:.3f}] m/s")
            print(f"   Y轴(横向): 范围=[{min_vel_y:.3f}, {max_vel_y:.3f}] m/s")
            print(f"   Z轴(转向): 范围=[{min_vel_z:.3f}, {max_vel_z:.3f}] rad/s")

            # 计算真实平均值（带符号）和绝对值统计
            avg_vel_x = np.mean(target_vel[:, 0])
            avg_vel_y = np.mean(target_vel[:, 1])
            avg_vel_z = np.mean(target_vel[:, 2])

            # 计算绝对值最大值（用于显示最大绝对速度）
            max_abs_vel_x = np.max(np.abs(target_vel[:, 0]))
            max_abs_vel_y = np.max(np.abs(target_vel[:, 1]))
            max_abs_vel_z = np.max(np.abs(target_vel[:, 2]))

            # 计算总速度范围
            total_speeds = np.sqrt(target_vel[:, 0] ** 2 + target_vel[:, 1] ** 2)
            max_total_speed = np.max(total_speeds)
            avg_total_speed = np.mean(total_speeds)

            print(
                f"   X轴(前进): 最大绝对值={max_abs_vel_x:6.3f} m/s, 平均={avg_vel_x:+6.3f} m/s"
            )
            print(
                f"   Y轴(横向): 最大绝对值={max_abs_vel_y:6.3f} m/s, 平均={avg_vel_y:+6.3f} m/s"
            )
            print(
                f"   Z轴(转向): 最大绝对值={max_abs_vel_z:6.3f} rad/s, 平均={avg_vel_z:+6.3f} rad/s"
            )
            print(
                f"   总速度:    最大={max_total_speed:6.3f} m/s, 平均={avg_total_speed:6.3f} m/s"
            )

            # 分析速度分布
            total_frames = len(target_vel)
            high_speed_frames = np.sum(total_speeds > 1.0)
            medium_speed_frames = np.sum((total_speeds > 0.3) & (total_speeds <= 1.0))
            low_speed_frames = np.sum(total_speeds <= 0.3)

            print(f"\n🚀 目标速度分布:")
            print(
                f"   高速(>1.0m/s):   {high_speed_frames:4d}帧 ({high_speed_frames/total_frames*100:5.1f}%)"
            )
            print(
                f"   中速(0.3-1.0m/s): {medium_speed_frames:4d}帧 ({medium_speed_frames/total_frames*100:5.1f}%)"
            )
            print(
                f"   低速(<=0.3m/s):  {low_speed_frames:4d}帧 ({low_speed_frames/total_frames*100:5.1f}%)"
            )

            # 分析转向
            strong_turn_frames = np.sum(np.abs(target_vel[:, 2]) > 0.8)
            medium_turn_frames = np.sum(
                (np.abs(target_vel[:, 2]) > 0.3) & (np.abs(target_vel[:, 2]) <= 0.8)
            )
            weak_turn_frames = np.sum(
                (np.abs(target_vel[:, 2]) > 0.1) & (np.abs(target_vel[:, 2]) <= 0.3)
            )
            straight_frames = np.sum(np.abs(target_vel[:, 2]) <= 0.1)

            print(f"\n🔄 目标转向分布:")
            print(
                f"   明显转向(>0.8rad/s):   {strong_turn_frames:4d}帧 ({strong_turn_frames/total_frames*100:5.1f}%)"
            )
            print(
                f"   中等转向(0.3-0.8rad/s): {medium_turn_frames:4d}帧 ({medium_turn_frames/total_frames*100:5.1f}%)"
            )
            print(
                f"   轻微转向(0.1-0.3rad/s): {weak_turn_frames:4d}帧 ({weak_turn_frames/total_frames*100:5.1f}%)"
            )
            print(
                f"   基本直行(<=0.1rad/s):  {straight_frames:4d}帧 ({straight_frames/total_frames*100:5.1f}%)"
            )

        # 分析机器人实际速度
        if has_base_vel:
            base_vel = motion["base_lin_vel_local"]  # [N, 3]

            print(f"\n🚀 机器人实际速度分析:")

            # 计算速度范围
            min_base_x = np.min(base_vel[:, 0])
            max_base_x = np.max(base_vel[:, 0])
            min_base_y = np.min(base_vel[:, 1])
            max_base_y = np.max(base_vel[:, 1])
            min_base_z = np.min(base_vel[:, 2])
            max_base_z = np.max(base_vel[:, 2])

            print(f"   X轴(前进): 范围=[{min_base_x:.3f}, {max_base_x:.3f}] m/s")
            print(f"   Y轴(横向): 范围=[{min_base_y:.3f}, {max_base_y:.3f}] m/s")
            print(f"   Z轴(垂直): 范围=[{min_base_z:.3f}, {max_base_z:.3f}] m/s")

            # 计算平均值和最大绝对值
            avg_base_x = np.mean(base_vel[:, 0])
            avg_base_y = np.mean(base_vel[:, 1])
            avg_base_z = np.mean(base_vel[:, 2])

            max_abs_base_x = np.max(np.abs(base_vel[:, 0]))
            max_abs_base_y = np.max(np.abs(base_vel[:, 1]))
            max_abs_base_z = np.max(np.abs(base_vel[:, 2]))

            # 计算水平和总速度
            base_horizontal_speeds = np.sqrt(base_vel[:, 0] ** 2 + base_vel[:, 1] ** 2)
            base_total_speeds = np.sqrt(np.sum(base_vel**2, axis=1))

            max_base_horizontal = np.max(base_horizontal_speeds)
            avg_base_horizontal = np.mean(base_horizontal_speeds)
            max_base_total = np.max(base_total_speeds)
            avg_base_total = np.mean(base_total_speeds)

            print(
                f"   X轴(前进): 最大绝对值={max_abs_base_x:6.3f} m/s, 平均={avg_base_x:+6.3f} m/s"
            )
            print(
                f"   Y轴(横向): 最大绝对值={max_abs_base_y:6.3f} m/s, 平均={avg_base_y:+6.3f} m/s"
            )
            print(
                f"   Z轴(垂直): 最大绝对值={max_abs_base_z:6.3f} m/s, 平均={avg_base_z:+6.3f} m/s"
            )
            print(
                f"   水平速度:  最大={max_base_horizontal:6.3f} m/s, 平均={avg_base_horizontal:6.3f} m/s"
            )
            print(
                f"   总速度:    最大={max_base_total:6.3f} m/s, 平均={avg_base_total:6.3f} m/s"
            )

            # 如果有目标速度，对比分析
            if target_vel is not None and len(target_vel) == len(base_vel):
                print(f"\n📊 目标vs实际速度对比:")
                # 计算差异
                diff_x = base_vel[:, 0] - target_vel[:, 0]
                diff_y = base_vel[:, 1] - target_vel[:, 1]

                rmse_x = np.sqrt(np.mean(diff_x**2))
                rmse_y = np.sqrt(np.mean(diff_y**2))
                mae_x = np.mean(np.abs(diff_x))
                mae_y = np.mean(np.abs(diff_y))

                print(f"   X轴差异: RMSE={rmse_x:.3f} m/s, MAE={mae_x:.3f} m/s")
                print(f"   Y轴差异: RMSE={rmse_y:.3f} m/s, MAE={mae_y:.3f} m/s")

                # 计算相关性
                corr_x = np.corrcoef(target_vel[:, 0], base_vel[:, 0])[0, 1]
                corr_y = np.corrcoef(target_vel[:, 1], base_vel[:, 1])[0, 1]

                print(f"   相关性: X轴={corr_x:.3f}, Y轴={corr_y:.3f}")

        # 分析机器人姿态
        if "root_rot" in motion:
            root_rot = motion["root_rot"]  # [N, 4] 四元数

            print(f"\n🧭 机器人姿态分析:")

            # 转换所有四元数为欧拉角
            rotations = scipyRot.from_quat(root_rot)
            euler_angles = rotations.as_euler("xyz", degrees=True)  # [N, 3] 度数

            # 分析欧拉角范围
            roll_range = [np.min(euler_angles[:, 0]), np.max(euler_angles[:, 0])]
            pitch_range = [np.min(euler_angles[:, 1]), np.max(euler_angles[:, 1])]
            yaw_range = [np.min(euler_angles[:, 2]), np.max(euler_angles[:, 2])]

            print(
                f"   Roll (横滚):  范围=[{roll_range[0]:+7.2f}°, {roll_range[1]:+7.2f}°]"
            )
            print(
                f"   Pitch (俯仰): 范围=[{pitch_range[0]:+7.2f}°, {pitch_range[1]:+7.2f}°]"
            )
            print(
                f"   Yaw (偏航):   范围=[{yaw_range[0]:+7.2f}°, {yaw_range[1]:+7.2f}°]"
            )

            # 计算平均值和标准差
            roll_mean = np.mean(euler_angles[:, 0])
            pitch_mean = np.mean(euler_angles[:, 1])
            yaw_mean = np.mean(euler_angles[:, 2])

            roll_std = np.std(euler_angles[:, 0])
            pitch_std = np.std(euler_angles[:, 1])
            yaw_std = np.std(euler_angles[:, 2])

            print(f"   Roll 统计:  平均={roll_mean:+7.2f}°, 标准差={roll_std:6.2f}°")
            print(f"   Pitch 统计: 平均={pitch_mean:+7.2f}°, 标准差={pitch_std:6.2f}°")
            print(f"   Yaw 统计:   平均={yaw_mean:+7.2f}°, 标准差={yaw_std:6.2f}°")

            # 分析姿态稳定性
            roll_stable = np.sum(np.abs(euler_angles[:, 0]) < 5.0)  # Roll小于5度的帧数
            pitch_stable = np.sum(
                np.abs(euler_angles[:, 1]) < 5.0
            )  # Pitch小于5度的帧数

            total_frames_rot = len(euler_angles)
            print(f"\n   姿态稳定性分析:")
            print(
                f"   稳定Roll (<5°):   {roll_stable:4d}帧 ({roll_stable/total_frames_rot*100:5.1f}%)"
            )
            print(
                f"   稳定Pitch (<5°):  {pitch_stable:4d}帧 ({pitch_stable/total_frames_rot*100:5.1f}%)"
            )

            # 分析转向幅度
            yaw_changes = np.abs(np.diff(euler_angles[:, 2]))
            # 处理跨越±180度的情况
            yaw_changes = np.minimum(yaw_changes, 360 - yaw_changes)

            large_turns = np.sum(yaw_changes > 10.0)  # 大转向（>10度/帧）
            medium_turns = np.sum(
                (yaw_changes > 3.0) & (yaw_changes <= 10.0)
            )  # 中等转向
            small_turns = np.sum(yaw_changes <= 3.0)  # 小转向

            total_turn_frames = len(yaw_changes)
            print(f"\n   转向幅度分析:")
            print(
                f"   大转向 (>10°/帧):   {large_turns:4d}帧 ({large_turns/total_turn_frames*100:5.1f}%)"
            )
            print(
                f"   中转向 (3-10°/帧):  {medium_turns:4d}帧 ({medium_turns/total_turn_frames*100:5.1f}%)"
            )
            print(
                f"   小转向 (≤3°/帧):   {small_turns:4d}帧 ({small_turns/total_turn_frames*100:5.1f}%)"
            )

    print("\n" + "=" * 80)


def key_call_back(keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys, info_display_interval, motion_data, motion_id, playback_speed
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif chr(keycode) == "Q":
        print("previous")
        motion_id = max(0, motion_id - 1)
        curr_motion_key = motion_data_keys[motion_id]
        print(f"切换到运动: {curr_motion_key}")
    elif chr(keycode) == "E":
        print("next")
        motion_id = min(len(motion_data_keys) - 1, motion_id + 1)
        curr_motion_key = motion_data_keys[motion_id]
        print(f"切换到运动: {curr_motion_key}")
    elif chr(keycode) == "I":
        # 切换信息显示频率
        if info_display_interval == 10:
            info_display_interval = 30  # 每秒显示一次
            print("📊 Task Info显示频率: 每30帧(1秒)")
        elif info_display_interval == 30:
            info_display_interval = 1  # 每帧显示
            print("📊 Task Info显示频率: 每帧")
        else:
            info_display_interval = 10  # 每10帧显示
            print("📊 Task Info显示频率: 每10帧")
    elif chr(keycode) == "S":
        # 显示当前动作统计信息
        curr_motion_key = motion_data_keys[motion_id]
        print(f"📊 显示当前动作统计信息: {curr_motion_key}")
        analyze_motion_stats(motion_data, curr_motion_key)
    elif chr(keycode) == "A":
        # 显示所有动作统计信息
        print("📊 显示所有动作统计信息...")
        analyze_motion_stats(motion_data)
    # 新增播放速度控制
    elif chr(keycode) == "1":
        playback_speed = 0.25
        print(f"🎬 播放速度: {playback_speed}x (1/4倍速)")
    elif chr(keycode) == "2":
        playback_speed = 0.5
        print(f"🎬 播放速度: {playback_speed}x (1/2倍速)")
    elif chr(keycode) == "3":
        playback_speed = 1.0
        print(f"🎬 播放速度: {playback_speed}x (正常倍速)")
    elif chr(keycode) == "4":
        playback_speed = 2.0
        print(f"🎬 播放速度: {playback_speed}x (2倍速)")
    elif chr(keycode) == "5":
        playback_speed = 4.0
        print(f"🎬 播放速度: {playback_speed}x (4倍速)")
    elif chr(keycode) == "+":
        # 增加播放速度
        if playback_speed < 0.5:
            playback_speed = min(0.5, playback_speed + 0.25)
        elif playback_speed < 2.0:
            playback_speed = min(2.0, playback_speed + 0.5)
        else:
            playback_speed = min(8.0, playback_speed + 1.0)
        print(f"🎬 播放速度: {playback_speed}x")
    elif chr(keycode) == "-":
        # 减少播放速度
        if playback_speed > 2.0:
            playback_speed = max(2.0, playback_speed - 1.0)
        elif playback_speed > 0.5:
            playback_speed = max(0.5, playback_speed - 0.5)
        else:
            playback_speed = max(0.1, playback_speed - 0.1)
        print(f"🎬 播放速度: {playback_speed}x")
    else:
        print("未映射的按键:", chr(keycode))


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys, info_display_interval, motion_data, playback_speed
    device = torch.device("cpu")
    humanoid_xml = cfg.robot.asset.assetFileName
    sk_tree = SkeletonTree.from_mjcf(humanoid_xml)

    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = (
        0,
        1,
        0,
        set(),
        0,
        1 / 30,
        False,
    )
    info_display_interval = 10  # 信息显示间隔帧数
    playback_speed = 1.0  # 播放速度倍数

    # 默认查找最新的运动文件，用户也可以通过配置指定
    if "motion_file" in cfg:
        motion_file = cfg.motion_file
    else:
        # 自动查找最新的pkl文件
        pkl_pattern = f"{cfg.output_path}/{cfg.robot.humanoid_type}/*.pkl"
        pkl_files = glob.glob(pkl_pattern)
        if not pkl_files:
            print(
                f"错误: 在 {cfg.output_path}/{cfg.robot.humanoid_type}/ 中未找到运动文件"
            )
            print("请先运行retarget_motion.py生成运动数据")
            return
        # 使用最新的文件
        motion_file = max(pkl_files, key=os.path.getmtime)

    if not os.path.exists(motion_file):
        print(f"错误: 运动文件不存在: {motion_file}")
        print("请先运行retarget_motion.py生成运动数据")
        return

    print(f"加载运动文件: {motion_file}")

    # 使用pickle加载运动数据（统一期望多运动格式）
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)

    print(f"运动数据类型: {type(motion_data)}")

    # 统一处理为多运动格式
    if isinstance(motion_data, dict):
        # 检查是否是单个运动数据（直接包含运动字段）
        if "root_trans_offset" in motion_data:
            # 这是旧格式的单个运动数据，包装为多运动格式
            filename = os.path.splitext(os.path.basename(motion_file))[0]
            motion_data = {filename: motion_data}
            print(f"  检测到旧格式单个运动数据，自动包装为: {filename}")

        motion_data_keys = list(motion_data.keys())
    else:
        print(f"错误: 不支持的运动数据格式: {type(motion_data)}")
        return

    print(f"找到 {len(motion_data_keys)} 个运动")

    # 打印数据集结构
    print_dataset_structure(motion_data)

    # 启动时分析所有运动统计信息
    # analyze_motion_stats(motion_data)
    print("💡 提示: 播放时按S键查看当前动作的详细统计，按A键查看所有动作统计")

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)

    RECORDING = False

    mj_model.opt.timestep = dt

    print("控制说明:")
    print("  空格键: 暂停/继续")
    print("  R键: 重置到开始")
    print("  Q键: 上一个运动")
    print("  E键: 下一个运动")
    print("  I键: 切换Task Info显示频率 (每10帧/每30帧/每帧)")
    print("  S键: 显示当前动作统计信息")
    print("  A键: 显示所有动作统计信息")
    print("")
    print("🎬 播放速度控制:")
    print("  1键: 0.25x (1/4倍速)")
    print("  2键: 0.5x (1/2倍速)")
    print("  3键: 1.0x (正常倍速)")
    print("  4键: 2.0x (2倍速)")
    print("  5键: 4.0x (4倍速)")
    print("  +键: 增加播放速度")
    print("  -键: 减少播放速度")
    print("")
    print("🔄 Task Info显示间隔: 每10帧")
    print(f"🎬 当前播放速度: {playback_speed}x")

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_call_back
    ) as viewer:
        for _ in range(25):
            add_visual_capsule(
                viewer.user_scn,
                np.zeros(3),
                np.array([0.001, 0, 0]),
                0.03,
                np.array([1, 0, 0, 1]),
            )

        while viewer.is_running():
            step_start = time.time()
            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]

            max_frames = curr_motion["dof"].shape[0]
            curr_time = int(time_step / dt) % max_frames

            try:
                mj_data.qpos[:3] = curr_motion["root_trans_offset"][curr_time]
                mj_data.qpos[3:7] = curr_motion["root_rot"][curr_time][[3, 0, 1, 2]]
                # mj_data.qpos[7:] = curr_motion['dof'][curr_time]
                mj_data.qpos[7:] = (
                    curr_motion["dof_increment"][curr_time]
                    + curr_motion["default_joint_angles"]
                )
            except IndexError as e:
                print(
                    f"数据索引错误: {e}, curr_time={curr_time}, max_frames={max_frames}"
                )
                time_step = 0
                continue

            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt * playback_speed  # 根据播放速度调整时间步进

            # 根据设置的间隔显示task_info
            if curr_time % info_display_interval == 0:
                print(f"\n{'='*50}")
                print(f"🎬 动作: {curr_motion_key} | 播放速度: {playback_speed}x")
                print(f"📍 帧: {curr_time}/{max_frames} | 时间: {curr_time/30:.2f}s")

                # 显示目标速度信息 (优先使用task_vel，其次使用task_info)
                target_vel_displayed = False
                if "task_vel" in curr_motion and curr_time < len(
                    curr_motion["task_vel"]
                ):
                    task_vel_2d = curr_motion["task_vel"][
                        curr_time
                    ]  # [2] - 只有前进和横向

                    print(f"🎯 目标速度 (task_vel):")
                    print(
                        f"   前进: {task_vel_2d[0]:+6.3f} m/s | 横向: {task_vel_2d[1]:+6.3f} m/s"
                    )

                    # 如果还有task_info，显示角速度
                    if "task_info" in curr_motion and curr_time < len(
                        curr_motion["task_info"]["target_vel"]
                    ):
                        angular_vel = curr_motion["task_info"]["target_vel"][curr_time][
                            2
                        ]
                        print(f"   转向: {angular_vel:+6.3f} rad/s")
                        target_vel_3d = np.array(
                            [task_vel_2d[0], task_vel_2d[1], angular_vel]
                        )
                    else:
                        target_vel_3d = np.array([task_vel_2d[0], task_vel_2d[1], 0.0])

                    # 计算总速度
                    speed = np.sqrt(task_vel_2d[0] ** 2 + task_vel_2d[1] ** 2)
                    print(f"   水平总速度: {speed:6.3f} m/s")
                    target_vel_displayed = True

                elif "task_info" in curr_motion and curr_time < len(
                    curr_motion["task_info"]["target_vel"]
                ):
                    # 回退到旧格式
                    task_info = curr_motion["task_info"]
                    target_vel_3d = task_info["target_vel"][curr_time]

                    print(f"🎯 目标速度 (task_info):")
                    print(
                        f"   前进: {target_vel_3d[0]:+6.3f} m/s | 横向: {target_vel_3d[1]:+6.3f} m/s | 转向: {target_vel_3d[2]:+6.3f} rad/s"
                    )

                    # 计算总速度
                    speed = np.sqrt(target_vel_3d[0] ** 2 + target_vel_3d[1] ** 2)
                    print(f"   水平总速度: {speed:6.3f} m/s")
                    target_vel_displayed = True
                else:
                    print("❌ 无目标速度数据")
                    target_vel_3d = None

                # =============== 显示机器人当前线速度 ===============
                if "base_lin_vel_local" in curr_motion and curr_time < len(
                    curr_motion["base_lin_vel_local"]
                ):
                    base_vel_local = curr_motion["base_lin_vel_local"][curr_time]
                    print(f"\n🚀 机器人实际线速度 (局部坐标系):")
                    print(
                        f"   前进: {base_vel_local[0]:+6.3f} m/s | 横向: {base_vel_local[1]:+6.3f} m/s | 垂直: {base_vel_local[2]:+6.3f} m/s"
                    )

                    # 计算水平面总速度
                    horizontal_speed = np.sqrt(
                        base_vel_local[0] ** 2 + base_vel_local[1] ** 2
                    )
                    total_speed = np.sqrt(np.sum(base_vel_local**2))
                    print(
                        f"   水平速度: {horizontal_speed:6.3f} m/s | 三维总速度: {total_speed:6.3f} m/s"
                    )

                    # 比较目标速度和当前速度（如果都存在）
                    if target_vel_displayed and target_vel_3d is not None:
                        vel_diff_x = base_vel_local[0] - target_vel_3d[0]
                        vel_diff_y = base_vel_local[1] - target_vel_3d[1]
                        print(
                            f"   与目标差异: Δx={vel_diff_x:+6.3f} m/s | Δy={vel_diff_y:+6.3f} m/s"
                        )

                        # 计算误差大小
                        error_magnitude = np.sqrt(vel_diff_x**2 + vel_diff_y**2)
                        print(f"   误差大小: {error_magnitude:6.3f} m/s")
                else:
                    print("\n❌ 无base_lin_vel_local数据")

                # =============== 显示关节速度 ===============
                if "joint_local_velocities" in curr_motion and curr_time < len(
                    curr_motion["joint_local_velocities"]
                ):
                    joint_velocities = curr_motion["joint_local_velocities"][curr_time]
                    print(f"\n⚙️  关节角速度:")
                    print(f"   前8个关节: [", end="")
                    for i in range(min(8, len(joint_velocities))):
                        print(
                            f"{joint_velocities[i]:+5.2f}",
                            end="" if i == min(7, len(joint_velocities) - 1) else ", ",
                        )
                    print("] rad/s")

                    # 显示速度最大的几个关节
                    abs_velocities = np.abs(joint_velocities)
                    max_indices = np.argsort(abs_velocities)[-3:][::-1]  # 取最大的3个
                    print(f"   最大角速度关节: ", end="")
                    for i, idx in enumerate(max_indices):
                        print(
                            f"关节{idx}({joint_velocities[idx]:+5.2f})",
                            end="" if i == 2 else ", ",
                        )
                    print(" rad/s")

                    # 统计分析
                    max_speed = np.max(abs_velocities)
                    avg_speed = np.mean(abs_velocities)
                    active_joints = np.sum(
                        abs_velocities > 0.1
                    )  # 速度大于0.1rad/s的活跃关节数
                    print(
                        f"   统计: 最大={max_speed:.2f}, 平均={avg_speed:.2f}, 活跃关节={active_joints}/{len(joint_velocities)}"
                    )
                else:
                    print("\n❌ 无joint_local_velocities数据")

                # =============== 显示机器人姿态信息 ===============
                # 获取当前四元数并转换为欧拉角
                curr_quat = curr_motion["root_rot"][curr_time]  # [x, y, z, w] 格式

                # 转换为欧拉角 (Roll, Pitch, Yaw)
                rotation = scipyRot.from_quat(curr_quat)
                euler_angles = rotation.as_euler("xyz", degrees=True)  # 返回度数
                roll, pitch, yaw = euler_angles

                print(f"\n🧭 机器人姿态 (欧拉角):")
                print(
                    f"   Roll (横滚):  {roll:+7.2f}° | Pitch (俯仰): {pitch:+7.2f}° | Yaw (偏航): {yaw:+7.2f}°"
                )

                # 显示四元数原始值
                print(
                    f"   四元数: [{curr_quat[0]:+6.3f}, {curr_quat[1]:+6.3f}, {curr_quat[2]:+6.3f}, {curr_quat[3]:+6.3f}]"
                )

                # 显示机器人朝向（单位向量）
                forward_vector = rotation.apply([1, 0, 0])  # 机器人前方向
                right_vector = rotation.apply([0, -1, 0])  # 机器人右方向
                up_vector = rotation.apply([0, 0, 1])  # 机器人上方向

                print(
                    f"   前方向量: [{forward_vector[0]:+6.3f}, {forward_vector[1]:+6.3f}, {forward_vector[2]:+6.3f}]"
                )
                print(
                    f"   右方向量: [{right_vector[0]:+6.3f}, {right_vector[1]:+6.3f}, {right_vector[2]:+6.3f}]"
                )
                print(
                    f"   上方向量: [{up_vector[0]:+6.3f}, {up_vector[1]:+6.3f}, {up_vector[2]:+6.3f}]"
                )

                if cfg.get("debug", False):
                    print(
                        f"🤖 根位置: [{mj_data.qpos[0]:.3f}, {mj_data.qpos[1]:.3f}, {mj_data.qpos[2]:.3f}]"
                    )
                    if "root_height" in curr_motion:
                        print(f"📐 根高度: {curr_motion['root_height'][curr_time]:.3f}")

                print(f"{'='*50}")

            joint_gt = motion_data[curr_motion_key]["smpl_joints"]
            max_joints = min(joint_gt.shape[1], len(viewer.user_scn.geoms))
            for i in range(max_joints):
                if i < len(viewer.user_scn.geoms):
                    viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]

            viewer.sync()

            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
