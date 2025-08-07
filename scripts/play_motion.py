#!/usr/bin/env python3
import os
import sys
import time
import glob
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import hydra
from omegaconf import DictConfig

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

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

# 全局变量
motion_id = 0
time_step = 0.0
paused = False
motion_data = None
motion_data_keys = []
selected_motions = None
playback_speed = 1.0
motion_file_paths = {}  # 存储motion_key到原始文件路径的映射

def key_callback(keycode):
    """键盘回调函数"""
    global motion_id, time_step, paused, motion_data_keys, playback_speed, motion_file_paths
    
    if chr(keycode) == " ":
        paused = not paused
        print("暂停" if paused else "继续")
    elif chr(keycode) == "R":
        time_step = 0.0
        print("重置到开始")
    elif chr(keycode) == "Q":
        motion_id = max(0, motion_id - 1)
        print(f"切换到运动: {motion_data_keys[motion_id]}")
        time_step = 0.0
    elif chr(keycode) == "E":
        motion_id = min(len(motion_data_keys) - 1, motion_id + 1)
        print(f"切换到运动: {motion_data_keys[motion_id]}")
        time_step = 0.0
    elif chr(keycode) == "=":  # 加号键（不需要按Shift）
        playback_speed = min(5.0, playback_speed + 0.25)
        print(f"播放速度: {playback_speed:.2f}x")
    elif chr(keycode) == "-":  # 减号键
        playback_speed = max(0.25, playback_speed - 0.25)
        print(f"播放速度: {playback_speed:.2f}x")
    elif chr(keycode) == "S":  # S键保存当前运动
        save_current_motion_to_good_list()

def save_current_motion_to_good_list():
    """保存当前播放的运动到good_motion.txt文件"""
    global motion_id, motion_data_keys, motion_file_paths
    
    if not motion_data_keys or motion_id >= len(motion_data_keys):
        print("错误: 没有有效的运动数据")
        return
    
    current_motion_key = motion_data_keys[motion_id]
    
    # 获取原始文件路径
    if current_motion_key in motion_file_paths:
        original_file_path = motion_file_paths[current_motion_key]
        good_motion_file = "good_motion.txt"
        
        # 检查是否已经存在
        existing_entries = []
        if os.path.exists(good_motion_file):
            with open(good_motion_file, 'r', encoding='utf-8') as f:
                existing_entries = [line.strip() for line in f.readlines()]
        
        if original_file_path not in existing_entries:
            # 添加到文件
            with open(good_motion_file, 'a', encoding='utf-8') as f:
                f.write(f"{original_file_path}\n")
            print(f"✓ 已保存到good_motion.txt: {original_file_path}")
        else:
            print(f"⚠ 已存在于good_motion.txt: {original_file_path}")
    else:
        print(f"警告: 无法找到原始文件路径 for {current_motion_key}")

def load_motion_data(cfg, selected_npy_files=None):
    """加载运动数据"""
    if selected_npy_files is not None:
        # 如果指定了npy文件列表，直接从原始数据加载
        return load_motion_data_from_npy_files(selected_npy_files)
    
    if "motion_file" in cfg:
        motion_file = cfg.motion_file
    else:
        # 自动查找最新的pkl文件
        pkl_pattern = f"{cfg.output_path}/{cfg.robot.humanoid_type}/*.pkl"
        pkl_files = glob.glob(pkl_pattern)
        if not pkl_files:
            print(f"错误: 在 {cfg.output_path}/{cfg.robot.humanoid_type}/ 中未找到运动文件")
            return None
        motion_file = max(pkl_files, key=os.path.getmtime)
    
    if not os.path.exists(motion_file):
        print(f"错误: 运动文件不存在: {motion_file}")
        return None
    
    print(f"加载运动文件: {motion_file}")
    
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
    
    # 统一处理为多运动格式
    if isinstance(motion_data, dict):
        if "root_trans_offset" in motion_data:
            # 旧格式的单个运动数据
            filename = os.path.splitext(os.path.basename(motion_file))[0]
            motion_data = {filename: motion_data}
    else:
        print(f"错误: 不支持的运动数据格式: {type(motion_data)}")
        return None
    
    return motion_data

def load_motion_data_from_npy_files(npy_files):
    """从原始npy文件直接加载运动数据"""
    global motion_file_paths
    motion_data = {}
    motion_file_paths = {}  # 重置映射
    
    for npy_file in npy_files:
        if not os.path.exists(npy_file):
            print(f"警告: 文件不存在: {npy_file}")
            continue
        
        try:
            # 加载npy数据
            data = np.load(npy_file)
            print(f"数据形状: {data.shape} - {npy_file}")
            
            # 验证数据结构
            if data.shape[1] < 36:
                print(f"警告: 数据列数不足36列 ({data.shape[1]}列): {npy_file}")
                continue
            
            # 生成motion key
            relative_path = npy_file.replace('/media/ray/Data/Retargeted_AMASS_for_robotics/g1/', '')
            motion_key = relative_path.replace('_poses_120_jpos.npy', '').replace('_poses_60_jpos.npy', '')
            motion_key = motion_key.replace('/', '_')
            
            # 保存原始文件路径映射
            motion_file_paths[motion_key] = relative_path
            
            # 转换为运动数据格式
            motion_info = convert_npy_to_motion_format(data, motion_key)
            if motion_info is not None:
                motion_data[motion_key] = motion_info
                print(f"已加载: {motion_key} (帧数: {data.shape[0]}, 120fps, Z+0.8)")
        
        except Exception as e:
            print(f"加载失败 {npy_file}: {e}")
    
    return motion_data

def convert_npy_to_motion_format(data, motion_key):
    """将npy数据转换为播放器需要的格式"""
    try:
        # 数据格式: 0:3 root position, 3:7 root quaternion (xyzw), 7:36 joint positions
        num_frames = data.shape[0]
        
        # 提取根节点位置并增加Z轴偏移
        root_positions = data[:, :3].copy()
        root_positions[:, 2] += 0.8  # Z坐标加高0.8
        
        # 提取四元数并转换顺序 (从xyzw转换为wxyz，MuJoCo使用wxyz顺序)
        quat_xyzw = data[:, 3:7]  # 原始数据是xyzw
        quat_wxyz = np.zeros_like(quat_xyzw)
        quat_wxyz[:, 0] = quat_xyzw[:, 3]  # w
        quat_wxyz[:, 1] = quat_xyzw[:, 0]  # x  
        quat_wxyz[:, 2] = quat_xyzw[:, 1]  # y
        quat_wxyz[:, 3] = quat_xyzw[:, 2]  # z
        
        motion_info = {
            "fps": 120,  # 设置为120帧率
            "root_trans_offset": root_positions,  # 根节点位置（已添加Z偏移）
            "root_rot": quat_wxyz,  # 根节点四元数 (转换为wxyz顺序)
            "dof": data[:, 7:36],  # 关节角度 (7:36，共29个关节)
            "default_joint_angles": np.zeros(29),  # 默认关节角度 (29个关节)
        }
        
        return motion_info
    
    except Exception as e:
        print(f"转换数据格式失败 {motion_key}: {e}")
        return None

def load_selected_motions(select_file):
    """从筛选结果文件中加载指定的运动序列文件路径"""
    if not os.path.exists(select_file):
        print(f"错误: 筛选文件不存在: {select_file}")
        return None
    
    selected_npy_files = []
    base_dir = "/media/ray/Data/Retargeted_AMASS_for_robotics/g1"
    print(f"从筛选文件加载: {select_file}")
    
    with open(select_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析文件，查找直线运动文件列表
    in_linear_section = False
    for line in lines:
        line = line.strip()
        if line == "=== 直线运动文件 ===":
            in_linear_section = True
            continue
        elif line.startswith("===") and in_linear_section:
            break
        elif in_linear_section and line and not line.startswith("  "):
            # 这是一个文件路径行
            if line.endswith('_poses_120_jpos.npy') or line.endswith('_poses_60_jpos.npy'):
                # 构建完整的文件路径
                full_path = os.path.join(base_dir, line)
                if os.path.exists(full_path):
                    selected_npy_files.append(full_path)
                else:
                    print(f"警告: 文件不存在: {full_path}")
    
    print(f"从筛选文件中找到 {len(selected_npy_files)} 个直线运动序列文件")
    return selected_npy_files

@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    global motion_id, time_step, paused, motion_data, motion_data_keys, selected_motions, playback_speed
    
    # 使用 Hydra 的 override 方式获取参数
    # 这些参数通过 +key=value 的方式传递
    select_file = getattr(cfg, 'select', None)
    playback_speed = getattr(cfg, 'speed', 1.0)
    
    # 如果 select 参数为空字符串，使用默认文件
    if select_file == "":
        select_file = "linear_motion_results.txt"
    
    print(f"筛选文件: {select_file if select_file else '无 (播放所有运动)'}")
    print(f"播放速度: {playback_speed}x")
    
    # 加载筛选的运动文件列表
    selected_npy_files = None
    if select_file is not None:
        selected_npy_files = load_selected_motions(select_file)
        if selected_npy_files is None or len(selected_npy_files) == 0:
            print("错误: 未找到任何有效的运动文件")
            return
    
    # 加载运动数据
    motion_data = load_motion_data(cfg, selected_npy_files)
    if motion_data is None:
        return
    
    motion_data_keys = list(motion_data.keys())
    print(f"找到 {len(motion_data_keys)} 个运动")
    
    # 初始化MuJoCo
    humanoid_xml = cfg.robot.asset.assetFileName
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    
    # 设置时间步长
    fps = motion_data[motion_data_keys[0]]["fps"]
    mj_model.opt.timestep = 1.0 / fps
    dt = mj_model.opt.timestep
    
    print(f"使用帧率: {fps} FPS")
    print(f"时间步长: {dt:.6f} 秒")
    
    print("\n控制说明:")
    print("  空格键: 暂停/继续")
    print("  R键: 重置到开始") 
    print("  Q键: 上一个运动")
    print("  E键: 下一个运动")
    print("  +键: 加速播放")
    print("  -键: 减速播放")
    print("  S键: 保存当前运动到good_motion.txt")
    print(f"  当前播放速度: {playback_speed:.2f}x")
    
    # 如果使用了select，显示good_motion.txt相关信息
    if select_file is not None:
        print(f"\n📁 正在使用筛选模式，按S键可保存好的运动到 good_motion.txt")
    print()
    
    # 启动MuJoCo查看器
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        # 初始化可视化胶囊
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
            
            # 获取当前运动数据
            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]
            
            # 计算当前帧
            max_frames = curr_motion["dof"].shape[0]
            curr_frame = int(time_step / dt) % max_frames
            
            try:
                # 设置机器人状态
                mj_data.qpos[:3] = curr_motion["root_trans_offset"][curr_frame]
                mj_data.qpos[3:7] = curr_motion["root_rot"][curr_frame]
                
                # 使用增量关节角度
                if "dof_increment" in curr_motion:
                    mj_data.qpos[7:] = (curr_motion["dof_increment"][curr_frame] + 
                                       curr_motion["default_joint_angles"])
                else:
                    mj_data.qpos[7:] = curr_motion["dof"][curr_frame]
                
                # 重置速度
                mj_data.qvel[:] = 0.0
                
            except IndexError:
                print(f"帧索引超出范围，重置到开始")
                time_step = 0.0
                continue
            
            # 前向动力学
            mujoco.mj_forward(mj_model, mj_data)
            
            # 更新时间步长
            if not paused:
                time_step += dt * playback_speed
            
            # 显示简单信息
            if curr_frame % 30 == 0:  # 每秒显示一次
                print(f"运动: {curr_motion_key} | 帧: {curr_frame}/{max_frames} | 时间: {curr_frame/fps:.2f}s | 速度: {playback_speed:.2f}x")
            
            # 显示关节位置（如果存在smpl_joints数据）
            if "smpl_joints" in curr_motion:
                joint_gt = curr_motion["smpl_joints"]
                max_joints = min(joint_gt.shape[1], len(viewer.user_scn.geoms))
                for i in range(max_joints):
                    if i < len(viewer.user_scn.geoms):
                        viewer.user_scn.geoms[i].pos = joint_gt[curr_frame, i]
            
            # 同步显示
            viewer.sync()
            
            # 控制帧率（考虑播放速度）
            time_until_next_step = (dt / playback_speed) - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
