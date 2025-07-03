#!/usr/bin/env python3
import os
import sys
import pickle
import time
import glob

# 确保在导入 torch/numpy 等库之前设置
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import mujoco
import mujoco.viewer

# 从您提供的代码中导入必要的科学计算库
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

# --- 1. 运动数据处理模块 ---
class MotionProcessor:
    """封装运动数据重采样逻辑的类"""

    def __init__(self, target_fps: int = 50):
        """
        初始化处理器。
        Args:
            target_fps (int): 重采样后的目标帧率。
        """
        self.target_fps = target_fps
        self.simulation_dt = 1.0 / target_fps
        print(f"✅ 运动处理器已初始化，目标帧率: {self.target_fps} FPS (dt={self.simulation_dt:.4f}s)")

    def _resample_data_Rn(self, data: np.ndarray, original_keyframes, target_keyframes) -> np.ndarray:
        """使用线性插值重采样普通向量数据 (例如位置)。"""
        # data 的形状是 (T, D)
        f = interp1d(original_keyframes, data, axis=0)
        return f(target_keyframes)

    def _resample_data_SO3(self, raw_quaternions: np.ndarray, original_keyframes, target_keyframes) -> np.ndarray:
        rotations = Rotation.from_quat(raw_quaternions)
        slerp = Slerp(original_keyframes, rotations)
        resampled_rotations = slerp(target_keyframes)
        return resampled_rotations.as_quat()

    def _process_single_motion(self, motion_data: dict) -> dict:
        """处理单个运动序列，将其重采样到目标帧率。"""
        # 提取核心数据
        joint_positions_raw = motion_data["dof_increment"]
        root_positions_raw = motion_data["root_trans_offset"]
        root_quats_raw = motion_data["root_rot"]  # 格式: xyzw
        original_fps = motion_data["fps"]

        # 计算原始和目标时间序列
        dt_orig = 1.0 / original_fps
        T_orig = len(joint_positions_raw)
        t_orig = np.linspace(0, (T_orig - 1) * dt_orig, T_orig)

        T_new = int(T_orig * dt_orig / 0.02)
        t_new = np.linspace(0, (T_orig - 1) * dt_orig, T_new)

        # 重采样所有需要的数据
        resampled_joint_positions = self._resample_data_Rn(joint_positions_raw, t_orig, t_new)
        resampled_base_positions = self._resample_data_Rn(root_positions_raw, t_orig, t_new)
        resampled_base_orientations = self._resample_data_SO3(root_quats_raw, t_orig, t_new)

        # 构建新的、只包含核心播放数据的字典
        new_motion = {
            "dof_increment": resampled_joint_positions,
            "root_trans_offset": resampled_base_positions,
            "root_rot": resampled_base_orientations,
            "fps": self.target_fps,
            "default_joint_angles": motion_data.get("default_joint_angles", 0) # 保留默认角度
        }
        return new_motion

    def process_dataset(self, original_dataset: dict) -> dict:
        """
        处理整个数据集，对其中每个动作进行重采样。
        """
        print("\n" + "="*50)
        print("🚀 开始处理整个数据集...")
        resampled_dataset = {}
        for motion_key, motion_data in original_dataset.items():
            print(f"  处理动作: '{motion_key}'")
            try:
                new_motion = self._process_single_motion(motion_data)
                resampled_dataset[motion_key] = new_motion
            except Exception as e:
                print(f"    ❌ 处理 '{motion_key}' 时发生错误: {e}")

        print("🎉 数据集处理完成!")
        print("="*50 + "\n")
        return resampled_dataset


# --- 2. 简化的播放器 ---
# 全局变量用于键盘回调
paused = False
motion_id = 0
time_step = 0.0
motion_data_keys = []
motion_data_resampled = {}


def key_callback(keycode):
    """简化的键盘回调函数"""
    global paused, motion_id, time_step
    key_char = chr(keycode)

    if key_char == ' ':
        paused = not paused
        print("播放已暂停" if paused else "播放已继续")
    elif key_char == 'R':
        time_step = 0
        print("重置当前动作")
    elif key_char == 'Q':
        motion_id = max(0, motion_id - 1)
        time_step = 0
        print(f"切换到上一个动作: {motion_data_keys[motion_id]}")
    elif key_char == 'E':
        motion_id = min(len(motion_data_keys) - 1, motion_id + 1)
        time_step = 0
        print(f"切换到下一个动作: {motion_data_keys[motion_id]}")
    else:
        # 其他按键可以保留或移除，这里只保留最核心的
        pass

@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    global paused, motion_id, time_step, motion_data_keys, motion_data_resampled

    # --- A. 加载和处理数据 ---

    # 自动查找最新的原始运动文件
    pkl_pattern = f"{cfg.output_path}/{cfg.robot.humanoid_type}/*.pkl"
    pkl_files = glob.glob(pkl_pattern)
    if not pkl_files:
        print(f"错误: 在 {cfg.output_path}/{cfg.robot.humanoid_type}/ 中未找到原始运动文件。")
        return

    original_motion_file = max(pkl_files, key=os.path.getmtime)
    print(f"📖 正在加载原始运动文件: {original_motion_file}")

    with open(original_motion_file, "rb") as f:
        motion_data_original = pickle.load(f)

    # 实例化处理器并处理数据
    TARGET_FPS = 50
    processor = MotionProcessor(target_fps=TARGET_FPS)
    motion_data_resampled = processor.process_dataset(motion_data_original)

    # 保存处理后的文件
    output_filename = os.path.join(
        os.path.dirname(original_motion_file),
        f"{os.path.splitext(os.path.basename(original_motion_file))[0]}_resampled_{TARGET_FPS}fps.pkl"
    )
    print(f"💾 正在保存重采样后的数据集到: {output_filename}")
    with open(output_filename, "wb") as f:
        pickle.dump(motion_data_resampled, f)
    print("保存成功!")

    motion_data_keys = list(motion_data_resampled.keys())
    if not motion_data_keys:
        print("错误: 重采样后的数据集中没有任何动作。")
        return

    # --- B. 初始化MuJoCo播放器 ---
    humanoid_xml = cfg.robot.asset.assetFileName
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)

    # 设置模拟器的 timestep 以匹配我们新的帧率
    dt = 1.0 / TARGET_FPS
    mj_model.opt.timestep = dt

    mj_data = mujoco.MjData(mj_model)

    print("\n" + "="*50)
    print("🎮 播放器控制:")
    print("  空格: 暂停/继续")
    print("  R:   重置当前动作")
    print("  Q:   上一个动作")
    print("  E:   下一个动作")
    print("="*50 + "\n")

    # --- C. 启动播放器和主循环 ---
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 获取当前动作
            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data_resampled[curr_motion_key]

            max_frames = len(curr_motion["root_rot"])
            curr_frame_idx = int(time_step / dt) % max_frames

            # 更新模型状态
            mj_data.qpos[:3] = curr_motion["root_trans_offset"][curr_frame_idx]
            # MuJoCo四元数顺序为 w,x,y,z; 我们的数据是 x,y,z,w
            mj_data.qpos[3:7] = curr_motion["root_rot"][curr_frame_idx][[3, 0, 1, 2]]

            # 关节角度 = 增量 + 默认值
            default_angles = curr_motion.get("default_joint_angles", 0)
            mj_data.qpos[7:] = curr_motion["dof_increment"][curr_frame_idx] + default_angles

            mujoco.mj_forward(mj_model, mj_data)

            # 简单的信息打印
            if not paused and curr_frame_idx % TARGET_FPS == 0: # 每秒打印一次
                print(f"  ▶️ 播放中: {curr_motion_key} | 帧: {curr_frame_idx}/{max_frames}", end='\r')

            # 更新时间步
            if not paused:
                time_step += dt

            # 同步渲染
            viewer.sync()

            # 控制播放速度以匹配真实时间
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()