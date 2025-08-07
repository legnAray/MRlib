#!/usr/bin/env python3

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_dataset(pkl_file):
    """加载pkl数据集文件"""
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"文件不存在: {pkl_file}")
    
    print(f"加载数据集: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # 统一处理为多动作格式
    if isinstance(data, dict):
        if "root_trans_offset" in data:
            # 旧格式的单个运动数据
            filename = os.path.splitext(os.path.basename(pkl_file))[0]
            data = {filename: data}
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")
    
    return data

def plot_velocity_curves(motion_data, motion_key):
    """绘制线速度曲线"""
    if motion_key not in motion_data:
        raise KeyError(f"动作 '{motion_key}' 不存在于数据集中")
    
    motion = motion_data[motion_key]
    
    # 检查必要的数据字段
    has_base_lin_vel = "base_lin_vel" in motion and motion["base_lin_vel"] is not None
    has_base_lin_vel_local = "base_lin_vel_local" in motion and motion["base_lin_vel_local"] is not None
    has_base_lin_vel_local_50window = "base_lin_vel_local_50window" in motion and motion["base_lin_vel_local_50window"] is not None
    
    if not has_base_lin_vel and not has_base_lin_vel_local and not has_base_lin_vel_local_50window:
        raise ValueError(f"动作 '{motion_key}' 中没有找到速度数据")
    
    # 获取数据
    fps = motion.get("fps", 30)
    
    # 创建图形
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Linear Velocity Curves - {motion_key}", fontsize=16)
    
    # 绘制base_lin_vel（世界坐标系）
    if has_base_lin_vel:
        base_lin_vel = motion["base_lin_vel"]
        n_frames = len(base_lin_vel)
        time_axis = np.arange(n_frames) / fps
        
        axes[0, 0].plot(time_axis, base_lin_vel[:, 0], 'r-', linewidth=2, label='X')
        axes[0, 0].set_title('base_lin_vel - X axis (World Frame)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Velocity (m/s)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].plot(time_axis, base_lin_vel[:, 1], 'g-', linewidth=2, label='Y')
        axes[0, 1].set_title('base_lin_vel - Y axis (World Frame)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[0, 2].plot(time_axis, base_lin_vel[:, 2], 'b-', linewidth=2, label='Z')
        axes[0, 2].set_title('base_lin_vel - Z axis (World Frame)')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Velocity (m/s)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # 打印统计信息
        print(f"\n=== base_lin_vel (World Frame) Statistics ===")
        print(f"Data shape: {base_lin_vel.shape}")
        print(f"X axis: Range=[{np.min(base_lin_vel[:, 0]):.3f}, {np.max(base_lin_vel[:, 0]):.3f}] m/s, Mean={np.mean(base_lin_vel[:, 0]):+.3f} m/s")
        print(f"Y axis: Range=[{np.min(base_lin_vel[:, 1]):.3f}, {np.max(base_lin_vel[:, 1]):.3f}] m/s, Mean={np.mean(base_lin_vel[:, 1]):+.3f} m/s")
        print(f"Z axis: Range=[{np.min(base_lin_vel[:, 2]):.3f}, {np.max(base_lin_vel[:, 2]):.3f}] m/s, Mean={np.mean(base_lin_vel[:, 2]):+.3f} m/s")
    else:
        for i in range(3):
            axes[0, i].text(0.5, 0.5, 'base_lin_vel data not available', ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].set_title(f'base_lin_vel - {"XYZ"[i]} axis (World Frame)')
    
    # 绘制base_lin_vel_local（局部坐标系）
    if has_base_lin_vel_local:
        base_lin_vel_local = motion["base_lin_vel_local"]
        n_frames = len(base_lin_vel_local)
        time_axis = np.arange(n_frames) / fps
        
        axes[1, 0].plot(time_axis, base_lin_vel_local[:, 0], 'r-', linewidth=2, label='X')
        axes[1, 0].set_title('base_lin_vel_local - X axis (Local Frame)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Velocity (m/s)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].plot(time_axis, base_lin_vel_local[:, 1], 'g-', linewidth=2, label='Y')
        axes[1, 1].set_title('base_lin_vel_local - Y axis (Local Frame)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Velocity (m/s)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        axes[1, 2].plot(time_axis, base_lin_vel_local[:, 2], 'b-', linewidth=2, label='Z')
        axes[1, 2].set_title('base_lin_vel_local - Z axis (Local Frame)')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Velocity (m/s)')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        # 打印统计信息
        print(f"\n=== base_lin_vel_local (Local Frame) Statistics ===")
        print(f"Data shape: {base_lin_vel_local.shape}")
        print(f"X axis: Range=[{np.min(base_lin_vel_local[:, 0]):.3f}, {np.max(base_lin_vel_local[:, 0]):.3f}] m/s, Mean={np.mean(base_lin_vel_local[:, 0]):+.3f} m/s")
        print(f"Y axis: Range=[{np.min(base_lin_vel_local[:, 1]):.3f}, {np.max(base_lin_vel_local[:, 1]):.3f}] m/s, Mean={np.mean(base_lin_vel_local[:, 1]):+.3f} m/s")
        print(f"Z axis: Range=[{np.min(base_lin_vel_local[:, 2]):.3f}, {np.max(base_lin_vel_local[:, 2]):.3f}] m/s, Mean={np.mean(base_lin_vel_local[:, 2]):+.3f} m/s")
    else:
        for i in range(3):
            axes[1, i].text(0.5, 0.5, 'base_lin_vel_local data not available', ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f'base_lin_vel_local - {"XYZ"[i]} axis (Local Frame)')
    
    # 绘制base_lin_vel_local_50window（50窗口滑动平均）
    if has_base_lin_vel_local_50window:
        base_lin_vel_local_50window = motion["base_lin_vel_local_50window"]
        n_frames = len(base_lin_vel_local_50window)
        time_axis = np.arange(n_frames) / fps
        
        axes[2, 0].plot(time_axis, base_lin_vel_local_50window[:, 0], 'r-', linewidth=2, label='X')
        axes[2, 0].set_title('base_lin_vel_local_50window - X axis (Local Frame, 50-window smoothed)')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Velocity (m/s)')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend()
        
        axes[2, 1].plot(time_axis, base_lin_vel_local_50window[:, 1], 'g-', linewidth=2, label='Y')
        axes[2, 1].set_title('base_lin_vel_local_50window - Y axis (Local Frame, 50-window smoothed)')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Velocity (m/s)')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].legend()
        
        axes[2, 2].plot(time_axis, base_lin_vel_local_50window[:, 2], 'b-', linewidth=2, label='Z')
        axes[2, 2].set_title('base_lin_vel_local_50window - Z axis (Local Frame, 50-window smoothed)')
        axes[2, 2].set_xlabel('Time (s)')
        axes[2, 2].set_ylabel('Velocity (m/s)')
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].legend()
        
        # 打印统计信息
        print(f"\n=== base_lin_vel_local_50window (Local Frame, 50-window smoothed) Statistics ===")
        print(f"Data shape: {base_lin_vel_local_50window.shape}")
        print(f"X axis: Range=[{np.min(base_lin_vel_local_50window[:, 0]):.3f}, {np.max(base_lin_vel_local_50window[:, 0]):.3f}] m/s, Mean={np.mean(base_lin_vel_local_50window[:, 0]):+.3f} m/s")
        print(f"Y axis: Range=[{np.min(base_lin_vel_local_50window[:, 1]):.3f}, {np.max(base_lin_vel_local_50window[:, 1]):.3f}] m/s, Mean={np.mean(base_lin_vel_local_50window[:, 1]):+.3f} m/s")
        print(f"Z axis: Range=[{np.min(base_lin_vel_local_50window[:, 2]):.3f}, {np.max(base_lin_vel_local_50window[:, 2]):.3f}] m/s, Mean={np.mean(base_lin_vel_local_50window[:, 2]):+.3f} m/s")
    else:
        for i in range(3):
            axes[2, i].text(0.5, 0.5, 'base_lin_vel_local_50window data not available', ha='center', va='center', transform=axes[2, i].transAxes)
            axes[2, i].set_title(f'base_lin_vel_local_50window - {"XYZ"[i]} axis (Local Frame, 50-window smoothed)')
    
    plt.tight_layout()
    plt.show()

def list_motions(motion_data):
    """列出数据集中的所有动作"""
    print(f"\n数据集包含 {len(motion_data)} 个动作:")
    for i, (key, motion) in enumerate(motion_data.items()):
        # 获取帧数信息
        if "dof" in motion:
            frames = motion["dof"].shape[0]
        elif "root_trans_offset" in motion:
            frames = motion["root_trans_offset"].shape[0]
        else:
            frames = 0
        
        fps = motion.get("fps", 30)
        duration = frames / fps
        
        # 检查速度数据
        has_base_lin_vel = "base_lin_vel" in motion and motion["base_lin_vel"] is not None
        has_base_lin_vel_local = "base_lin_vel_local" in motion and motion["base_lin_vel_local"] is not None
        has_base_lin_vel_local_50window = "base_lin_vel_local_50window" in motion and motion["base_lin_vel_local_50window"] is not None
        
        print(f"  {i+1}. {key}")
        print(f"      帧数: {frames}, 时长: {duration:.2f}s, 帧率: {fps}fps")
        print(f"      速度数据: base_lin_vel={'✓' if has_base_lin_vel else '✗'}, base_lin_vel_local={'✓' if has_base_lin_vel_local else '✗'}, base_lin_vel_local_50window={'✓' if has_base_lin_vel_local_50window else '✗'}")

def select_motion_by_index(motion_data, index):
    """通过序号选择动作"""
    motion_keys = list(motion_data.keys())
    if index < 1 or index > len(motion_keys):
        raise ValueError(f"序号 {index} 超出范围，应该在 1-{len(motion_keys)} 之间")
    return motion_keys[index - 1]

def main():
    parser = argparse.ArgumentParser(description='可视化数据集中的线速度曲线')
    parser.add_argument('pkl_file', help='输入的pkl文件路径')
    parser.add_argument('--motion', '-m', type=str, default=None, help='指定要绘制的动作名称')
    parser.add_argument('--index', '-i', type=int, default=None, help='通过序号选择动作（从1开始）')
    parser.add_argument('--list', '-l', action='store_true', help='列出数据集中的所有动作')
    
    args = parser.parse_args()
    
    # 加载数据集
    try:
        motion_data = load_dataset(args.pkl_file)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return
    
    # 列出所有动作
    if args.list:
        list_motions(motion_data)
        return
    
    # 确定要绘制的动作
    motion_key = None
    
    if args.index:
        # 通过序号选择
        try:
            motion_key = select_motion_by_index(motion_data, args.index)
            print(f"选择动作 [{args.index}]: {motion_key}")
        except ValueError as e:
            print(f"错误: {e}")
            list_motions(motion_data)
            return
    elif args.motion:
        # 通过名称选择
        if args.motion not in motion_data:
            print(f"错误: 动作 '{args.motion}' 不存在于数据集中")
            list_motions(motion_data)
            return
        motion_key = args.motion
    else:
        # 如果只有一个动作，自动选择
        if len(motion_data) == 1:
            motion_key = list(motion_data.keys())[0]
            print(f"自动选择动作: {motion_key}")
        else:
            print("数据集包含多个动作，请使用 --index 参数指定序号或 --motion 参数指定名称")
            list_motions(motion_data)
            return
    
    # 绘制速度曲线
    try:
        plot_velocity_curves(motion_data, motion_key)
    except Exception as e:
        print(f"绘制失败: {e}")
        return

if __name__ == "__main__":
    main()
