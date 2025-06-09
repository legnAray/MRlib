import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from mrlib.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import Counter

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def analyze_motion_stats(motion_data, current_motion_key=None):
    """分析动作序列的统计信息
    
    Args:
        motion_data: 运动数据字典
        current_motion_key: 如果指定，只分析当前动作；否则分析所有动作
    """
    print("\n" + "="*80)
    if current_motion_key:
        print(f"📊 当前动作统计分析: {current_motion_key}")
    else:
        print("📊 所有动作序列统计分析")
    print("="*80)
    
    motion_type_names = {
        0: "原地踏步", 1: "左转行走", 2: "右转行走", 3: "直线行走",
        4: "左转跑步", 5: "右转跑步", 6: "直线跑步", 
        7: "站立", 8: "蹲下", 9: "未知"
    }
    
    # 根据参数决定分析哪些动作
    if current_motion_key and current_motion_key in motion_data:
        motions_to_analyze = {current_motion_key: motion_data[current_motion_key]}
    else:
        motions_to_analyze = motion_data
    
    for motion_key, motion in motions_to_analyze.items():
        print(f"\n🎬 动作: {motion_key}")
        print("-" * 60)
        
        if 'task_info' not in motion:
            print("❌ 无task_info数据")
            continue
            
        task_info = motion['task_info']
        target_vel = task_info['target_vel']
        motion_types = task_info['motion_type']
        
        # 计算最大速度
        min_vel_x = np.min(target_vel[:, 0])
        max_vel_x = np.max(target_vel[:, 0])
        print(f"X轴(前进): 范围=[{min_vel_x:.3f}, {max_vel_x:.3f}] m/s")
        min_vel_y = np.min(target_vel[:, 1])
        max_vel_y = np.max(target_vel[:, 1])
        print(f"Y轴(横向): 范围=[{min_vel_y:.3f}, {max_vel_y:.3f}] m/s")
        min_vel_z = np.min(target_vel[:, 2])
        max_vel_z = np.max(target_vel[:, 2])
        print(f"Z轴(转向): 范围=[{min_vel_z:.3f}, {max_vel_z:.3f}] rad/s")
        
        # 计算真实平均值（带符号）和绝对值统计
        avg_vel_x = np.mean(target_vel[:, 0])
        avg_vel_y = np.mean(target_vel[:, 1])
        avg_vel_z = np.mean(target_vel[:, 2])
        
        # 计算绝对值最大值（用于显示最大绝对速度）
        max_abs_vel_x = np.max(np.abs(target_vel[:, 0]))
        max_abs_vel_y = np.max(np.abs(target_vel[:, 1]))
        max_abs_vel_z = np.max(np.abs(target_vel[:, 2]))
        
        # 计算总速度范围
        total_speeds = np.sqrt(target_vel[:, 0]**2 + target_vel[:, 1]**2)
        max_total_speed = np.max(total_speeds)
        avg_total_speed = np.mean(total_speeds)
        
        print(f"🎯 速度分析:")
        print(f"   X轴(前进): 最大绝对值={max_abs_vel_x:6.3f} m/s, 平均={avg_vel_x:+6.3f} m/s")
        print(f"   Y轴(横向): 最大绝对值={max_abs_vel_y:6.3f} m/s, 平均={avg_vel_y:+6.3f} m/s")
        print(f"   Z轴(转向): 最大绝对值={max_abs_vel_z:6.3f} rad/s, 平均={avg_vel_z:+6.3f} rad/s")
        print(f"   总速度:    最大={max_total_speed:6.3f} m/s, 平均={avg_total_speed:6.3f} m/s")
        
        # 统计运动类型占比
        type_counts = Counter(motion_types)
        total_frames = len(motion_types)
        
        print(f"🏃 运动类型占比 (总帧数: {total_frames}):")
        for motion_type in sorted(type_counts.keys()):
            count = type_counts[motion_type]
            percentage = (count / total_frames) * 100
            type_name = motion_type_names.get(motion_type, f"类型{motion_type}")
            print(f"   {motion_type:2d} ({type_name:8s}): {count:4d}帧 ({percentage:5.1f}%)")
        
        # 分析速度分布
        high_speed_frames = np.sum(total_speeds > 1.0)
        medium_speed_frames = np.sum((total_speeds > 0.3) & (total_speeds <= 1.0))
        low_speed_frames = np.sum(total_speeds <= 0.3)
        
        print(f"🚀 速度分布:")
        print(f"   高速(>1.0m/s):   {high_speed_frames:4d}帧 ({high_speed_frames/total_frames*100:5.1f}%)")
        print(f"   中速(0.3-1.0m/s): {medium_speed_frames:4d}帧 ({medium_speed_frames/total_frames*100:5.1f}%)")
        print(f"   低速(<=0.3m/s):  {low_speed_frames:4d}帧 ({low_speed_frames/total_frames*100:5.1f}%)")
        
        # 分析转向，使用与分类逻辑一致的阈值
        strong_turn_frames = np.sum(np.abs(target_vel[:, 2]) > 0.8)  # 明显转向
        medium_turn_frames = np.sum((np.abs(target_vel[:, 2]) > 0.3) & (np.abs(target_vel[:, 2]) <= 0.8))  # 中等转向
        weak_turn_frames = np.sum((np.abs(target_vel[:, 2]) > 0.1) & (np.abs(target_vel[:, 2]) <= 0.3))  # 轻微转向
        straight_frames = np.sum(np.abs(target_vel[:, 2]) <= 0.1)  # 基本直行
        
        print(f"🔄 转向分布 (新阈值: 明显转向>0.8rad/s):")
        print(f"   明显转向(>0.8rad/s):   {strong_turn_frames:4d}帧 ({strong_turn_frames/total_frames*100:5.1f}%)")
        print(f"   中等转向(0.3-0.8rad/s): {medium_turn_frames:4d}帧 ({medium_turn_frames/total_frames*100:5.1f}%)")
        print(f"   轻微转向(0.1-0.3rad/s): {weak_turn_frames:4d}帧 ({weak_turn_frames/total_frames*100:5.1f}%)")
        print(f"   基本直行(<=0.1rad/s):  {straight_frames:4d}帧 ({straight_frames/total_frames*100:5.1f}%)")
        
        # 新增：分析机器人坐标系下的运动方向
        print(f"🧭 运动方向分析:")
        forward_frames = np.sum(target_vel[:, 0] > 0.1)   # 前进
        backward_frames = np.sum(target_vel[:, 0] < -0.1)  # 后退
        left_frames = np.sum(target_vel[:, 1] > 0.1)      # 左移
        right_frames = np.sum(target_vel[:, 1] < -0.1)    # 右移
        static_frames = np.sum((np.abs(target_vel[:, 0]) <= 0.1) & (np.abs(target_vel[:, 1]) <= 0.1))  # 静止
        
        print(f"   前进(+X>0.1m/s):     {forward_frames:4d}帧 ({forward_frames/total_frames*100:5.1f}%)")
        print(f"   后退(-X<-0.1m/s):    {backward_frames:4d}帧 ({backward_frames/total_frames*100:5.1f}%)")
        print(f"   左移(+Y>0.1m/s):     {left_frames:4d}帧 ({left_frames/total_frames*100:5.1f}%)")
        print(f"   右移(-Y<-0.1m/s):    {right_frames:4d}帧 ({right_frames/total_frames*100:5.1f}%)")
        print(f"   静止(|XY|<=0.1m/s):  {static_frames:4d}帧 ({static_frames/total_frames*100:5.1f}%)")
    
    print("\n" + "="*80)

def key_call_back(keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys, info_display_interval, motion_data, motion_id
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
            info_display_interval = 1   # 每帧显示
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
    else:
        print("未映射的按键:", chr(keycode))
    
    
@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys, info_display_interval, motion_data
    device = torch.device("cpu")
    humanoid_xml = cfg.robot.asset.assetFileName
    sk_tree = SkeletonTree.from_mjcf(humanoid_xml)
    
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False
    info_display_interval = 10  # 信息显示间隔帧数
    
    motion_file = cfg.get("motion_file", f"{cfg.output_path}/{cfg.robot.humanoid_type}/motion_im.p")
    
    if not os.path.exists(motion_file):
        print(f"错误: 运动文件不存在: {motion_file}")
        print("请先运行retarget_motion.py生成运动数据")
        return

    print(f"加载运动文件: {motion_file}")
    motion_data = joblib.load(motion_file)
    print(f"运动数据类型: {type(motion_data)}")
    
    if isinstance(motion_data, dict):
        motion_data_keys = list(motion_data.keys())
    else:
        motion_data_keys = ["single_motion"]
        motion_data = {"single_motion": motion_data}
    
    print(f"找到 {len(motion_data_keys)} 个运动")
    
    # 启动时分析所有运动统计信息
    analyze_motion_stats(motion_data)
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
    print("🔄 Task Info显示间隔: 每10帧")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(25):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.03, np.array([1, 0, 0, 1]))
        
        while viewer.is_running():
            step_start = time.time()
            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]
            
            max_frames = curr_motion['dof'].shape[0]
            curr_time = int(time_step/dt) % max_frames
            
            try:
                mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
                mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]
                # mj_data.qpos[7:] = curr_motion['dof'][curr_time]
                mj_data.qpos[7:] = curr_motion['dof_increment'][curr_time] + curr_motion['default_joint_angles']
            except IndexError as e:
                print(f"数据索引错误: {e}, curr_time={curr_time}, max_frames={max_frames}")
                time_step = 0
                continue
                
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt

            # 根据设置的间隔显示task_info
            if curr_time % info_display_interval == 0:
                print(f"\n{'='*50}")
                print(f"🎬 动作: {curr_motion_key}")
                print(f"📍 帧: {curr_time}/{max_frames} | 时间: {curr_time/30:.2f}s")
                
                # 显示task_info信息
                if 'task_info' in curr_motion:
                    task_info = curr_motion['task_info']
                    if curr_time < len(task_info['target_vel']):
                        target_vel = task_info['target_vel'][curr_time]
                        motion_type = task_info['motion_type'][curr_time]
                        
                        # 运动类型名称映射
                        motion_type_names = {
                            0: "原地踏步", 1: "左转行走", 2: "右转行走", 3: "直线行走",
                            4: "左转跑步", 5: "右转跑步", 6: "直线跑步", 
                            7: "站立", 8: "蹲下", 9: "未知"
                        }
                        type_name = motion_type_names.get(motion_type, f"类型{motion_type}")
                        
                        print(f"🎯 当前帧速度:")
                        print(f"   前进: {target_vel[0]:+6.3f} m/s | 横向: {target_vel[1]:+6.3f} m/s | 转向: {target_vel[2]:+6.3f} rad/s")
                        
                        # 计算总速度
                        speed = np.sqrt(target_vel[0]**2 + target_vel[1]**2)
                        print(f"   总速度: {speed:6.3f} m/s")
                        
                        print(f"🏃 运动类型: {motion_type} ({type_name})")
                        
                        # 显示这个动作的整体统计 (简化版)
                        all_vel = task_info['target_vel']
                        all_types = task_info['motion_type']
                        max_speed = np.max(np.sqrt(all_vel[:, 0]**2 + all_vel[:, 1]**2))
                        max_turn = np.max(np.abs(all_vel[:, 2]))
                        
                        type_counts = Counter(all_types)
                        most_common_type = type_counts.most_common(1)[0]
                        most_common_name = motion_type_names.get(most_common_type[0], f"类型{most_common_type[0]}")
                        
                        print(f"📊 整体统计: 最大速度={max_speed:.2f}m/s | 最大转速={max_turn:.2f}rad/s | 主要类型={most_common_name}({most_common_type[1]}帧)")
                        
                else:
                    print("❌ 无task_info数据")
                    
                if cfg.get("debug", False):
                    print(f"🤖 根位置: [{mj_data.qpos[0]:.3f}, {mj_data.qpos[1]:.3f}, {mj_data.qpos[2]:.3f}]")
                    if 'root_height' in curr_motion:
                        print(f"📐 根高度: {curr_motion['root_height'][curr_time]:.3f}")
                
                print(f"{'='*50}")

            joint_gt = motion_data[curr_motion_key]['smpl_joints']
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

