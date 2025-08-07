import os
import numpy as np
from pathlib import Path
import pickle
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import argparse

dataset_dir = "/media/ray/Data/g1_data_web/walk_straight"
default_joint_angles = np.array(
        [
            -0.12,
            0,
            0,
            0.45,
            -0.31,
            0,  # 左腿6个关节：hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
            -0.12,
            0,
            0,
            0.45,
            -0.31,
            0,  # 右腿6个关节：hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
            0,
            0,
            0,  # 腰部3个关节：waist_yaw, waist_roll, waist_pitch
            0,
            0.25,
            0,
            1.04,
            0,
            0,
            0,  # 左臂7个关节：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
            0,
            -0.25,
            0,
            1.04,
            0,
            0,
            0,  # 右臂7个关节：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        ]
    )

def ensure_quaternion_continuity(quaternions):
    """确保四元数序列的连续性，避免w分量跳跃"""
    continuous_quat = quaternions.copy()
    for i in range(1, len(quaternions)):
        # 检查当前四元数与前一个四元数的点积
        if np.dot(continuous_quat[i-1], continuous_quat[i]) < 0:
            continuous_quat[i] = -continuous_quat[i]
    return continuous_quat

def sliding_window_average(data, window_size):
    """
    计算滑动窗口平均，保持原始数据长度
    边界处根据实际可用点数计算平均，不人为补零
    
    Args:
        data: 输入数据 (N, D)
        window_size: 窗口大小
    
    Returns:
        滑动平均后的数据，形状与输入相同
    """
    if window_size <= 1:
        return data.copy()
    
    n_frames, n_dims = data.shape
    smoothed_data = np.zeros_like(data)
    
    # 计算窗口的半径（向下取整）
    half_window = window_size // 2
    
    # 对每一帧计算滑动平均
    for i in range(n_frames):
        # 确定当前帧的窗口范围
        start_idx = max(0, i - half_window)
        end_idx = min(n_frames, i + half_window + 1)
        
        # 计算当前窗口内的平均值
        smoothed_data[i] = np.mean(data[start_idx:end_idx], axis=0)
    
    return smoothed_data

def interpolate_motion_data(root_trans_offset, root_rot, dof, original_fps=120, target_fps=50):
    """
    对运动数据进行插值
    
    Args:
        root_trans_offset: 根部位置偏移 (N, 3)
        root_rot: 根部旋转四元数 (N, 4)
        dof: 关节角度 (N, 29)
        original_fps: 原始帧率
        target_fps: 目标帧率
    
    Returns:
        插值后的数据
    """
    original_length = len(root_trans_offset)
    
    # 创建时间轴
    original_duration = original_length / original_fps
    new_length = int(original_duration * target_fps)
    
    original_time = np.linspace(0, original_duration, original_length)
    new_time = np.linspace(0, original_duration, new_length)
    
    # 线性插值 - 根部位置
    f_trans = interp1d(original_time, root_trans_offset, axis=0, kind='linear')
    root_trans_offset_interp = f_trans(new_time)
    
    # 线性插值 - 关节角度
    f_dof = interp1d(original_time, dof, axis=0, kind='linear')
    dof_interp = f_dof(new_time)
    
    # 球面插值 - 四元数
    continuous_quat = ensure_quaternion_continuity(root_rot)
    rotations = Rotation.from_quat(continuous_quat)
    slerp = Slerp(original_time, rotations)
    root_rot_interp = slerp(new_time).as_quat()
    
    print(f"  插值: {original_length} 帧 ({original_fps}fps) → {new_length} 帧 ({target_fps}fps)")
    
    return root_trans_offset_interp, root_rot_interp, dof_interp

def calculate_velocities(root_trans_offset, dof, root_rot, fps):
    """
    使用前向差分计算速度
    
    Args:
        root_trans_offset: 根部位置偏移 (N, 3)
        dof: 关节角度 (N, 29)
        root_rot: 根部旋转四元数 (N, 4)
        fps: 帧率
    
    Returns:
        base_lin_vel: 世界坐标系线速度
        joint_velocities: 关节速度
        base_lin_vel_local: 局部坐标系线速度
    """
    dt = 1.0 / fps
    n_frames = len(root_trans_offset)
    
    # 计算世界坐标系线速度
    base_lin_vel = np.zeros_like(root_trans_offset)
    base_lin_vel[:-1] = (root_trans_offset[1:] - root_trans_offset[:-1]) / dt
    base_lin_vel[-1] = base_lin_vel[-2]  # 最后一帧使用前一帧的速度
    
    # 计算关节速度
    joint_velocities = np.zeros_like(dof)
    joint_velocities[:-1] = (dof[1:] - dof[:-1]) / dt
    joint_velocities[-1] = joint_velocities[-2]  # 最后一帧使用前一帧的速度
    
    # 计算局部坐标系线速度 - 向量化处理
    rotations = Rotation.from_quat(root_rot[:,[1,2,3,0]])
    base_lin_vel_local = rotations.inv().apply(base_lin_vel)

    return base_lin_vel, joint_velocities, base_lin_vel_local

def create_mirror_data(root_trans_offset, root_rot, dof):
    """
    创建镜像数据
    
    Args:
        root_trans_offset: 根部位置偏移 (N, 3)
        root_rot: 根部旋转四元数 (N, 4) - [w, x, y, z]
        dof: 关节角度 (N, 29)
    
    Returns:
        镜像后的数据
    """
    # 位置镜像：X,Y,Z -> X,-Y,Z
    mirror_trans = root_trans_offset.copy()
    mirror_trans[:, 1] = -mirror_trans[:, 1]  # Y坐标翻转
    
    # 四元数镜像：w,x,y,z -> w,-x,y,-z
    mirror_rot = root_rot.copy()
    mirror_rot[:, 1] = -mirror_rot[:, 1]  # x分量翻转
    mirror_rot[:, 3] = -mirror_rot[:, 3]  # z分量翻转
    
    # 关节角度镜像
    mirror_dof = np.zeros_like(dof)
    
    # 关节索引定义（基于dof_names的顺序）
    # 左腿 (0-5): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    # 右腿 (6-11): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    # 腰部 (12-14): waist_yaw, waist_roll, waist_pitch
    # 左臂 (15-21): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
    # 右臂 (22-28): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
    
    # 左腿 -> 右腿
    mirror_dof[:, 6] = dof[:, 0]      # left_hip_pitch -> right_hip_pitch
    mirror_dof[:, 7] = -dof[:, 1]     # left_hip_roll -> -right_hip_roll
    mirror_dof[:, 8] = -dof[:, 2]     # left_hip_yaw -> -right_hip_yaw
    mirror_dof[:, 9] = dof[:, 3]      # left_knee -> right_knee
    mirror_dof[:, 10] = dof[:, 4]     # left_ankle_pitch -> right_ankle_pitch
    mirror_dof[:, 11] = -dof[:, 5]    # left_ankle_roll -> -right_ankle_roll
    
    # 右腿 -> 左腿
    mirror_dof[:, 0] = dof[:, 6]      # right_hip_pitch -> left_hip_pitch
    mirror_dof[:, 1] = -dof[:, 7]     # right_hip_roll -> -left_hip_roll
    mirror_dof[:, 2] = -dof[:, 8]     # right_hip_yaw -> -left_hip_yaw
    mirror_dof[:, 3] = dof[:, 9]      # right_knee -> left_knee
    mirror_dof[:, 4] = dof[:, 10]     # right_ankle_pitch -> left_ankle_pitch
    mirror_dof[:, 5] = -dof[:, 11]    # right_ankle_roll -> -left_ankle_roll
    
    # 腰部
    mirror_dof[:, 12] = -dof[:, 12]   # waist_yaw -> -waist_yaw
    mirror_dof[:, 13] = -dof[:, 13]   # waist_roll -> -waist_roll
    mirror_dof[:, 14] = dof[:, 14]    # waist_pitch -> waist_pitch
    
    # 左臂 -> 右臂
    mirror_dof[:, 22] = dof[:, 15]    # left_shoulder_pitch -> right_shoulder_pitch
    mirror_dof[:, 23] = -dof[:, 16]   # left_shoulder_roll -> -right_shoulder_roll
    mirror_dof[:, 24] = -dof[:, 17]   # left_shoulder_yaw -> -right_shoulder_yaw
    mirror_dof[:, 25] = dof[:, 18]    # left_elbow -> right_elbow
    mirror_dof[:, 26] = -dof[:, 19]   # left_wrist_roll -> -right_wrist_roll
    mirror_dof[:, 27] = dof[:, 20]    # left_wrist_pitch -> right_wrist_pitch
    mirror_dof[:, 28] = -dof[:, 21]   # left_wrist_yaw -> -right_wrist_yaw
    
    # 右臂 -> 左臂
    mirror_dof[:, 15] = dof[:, 22]    # right_shoulder_pitch -> left_shoulder_pitch
    mirror_dof[:, 16] = -dof[:, 23]   # right_shoulder_roll -> -left_shoulder_roll
    mirror_dof[:, 17] = -dof[:, 24]   # right_shoulder_yaw -> -left_shoulder_yaw
    mirror_dof[:, 18] = dof[:, 25]    # right_elbow -> left_elbow
    mirror_dof[:, 19] = -dof[:, 26]   # right_wrist_roll -> -left_wrist_roll
    mirror_dof[:, 20] = dof[:, 27]    # right_wrist_pitch -> left_wrist_pitch
    mirror_dof[:, 21] = -dof[:, 28]   # right_wrist_yaw -> -left_wrist_yaw
    
    return mirror_trans, mirror_rot, mirror_dof

def merge_datasets(dataset_dir, output_path=None, target_fps=50, enable_mirror=False):
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_dir}")
    
    npy_files = list(dataset_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"在目录 {dataset_dir} 中未找到.npy文件")
        return {}
    
    print(f"找到 {len(npy_files)} 个.npy文件")
    print(f"目标帧率: {target_fps}fps")
    if enable_mirror:
        print("启用镜像数据")
    
    dataset = {}
    
    for npy_file in npy_files:
        key = npy_file.stem
        try:
            data = np.load(npy_file)
            print(f"加载文件 {npy_file.name}: 形状 {data.shape}")
            if len(data.shape) != 2:
                print(f"警告: 文件 {npy_file.name} 的形状不是二维 (N_steps, root + dof)")
                continue
            
            # 提取各个部分的数据
            root_trans_offset = data[:, 0:3]  # 前3列
            root_trans_offset[:, 2] += 0.793
            root_rot = data[:, [6, 3, 4, 5]]  # 第6、3、4、5列
            dof = data[:, -29:]  # 后29列
            
            # 插值到目标帧率
            root_trans_offset_interp, root_rot_interp, dof_interp = interpolate_motion_data(
                root_trans_offset, root_rot, dof, original_fps=120, target_fps=target_fps
            )
            
            # 计算关节角度增量
            dof_increment = dof_interp - default_joint_angles
            
            # 计算速度
            base_lin_vel, joint_velocities, base_lin_vel_local = calculate_velocities(
                root_trans_offset_interp, dof_interp, root_rot_interp, target_fps
            )
            
            # 计算base_lin_vel_local的50窗口滑动平均
            base_lin_vel_local_50window = sliding_window_average(base_lin_vel_local, 50)
            
            # 构建新的数据结构
            dataset[key] = {
                'root_trans_offset': root_trans_offset_interp,
                'root_rot': root_rot_interp,
                'dof': dof_interp,
                'default_joint_angles': default_joint_angles,
                'dof_increment': dof_increment,
                'joint_velocities': joint_velocities,
                'base_lin_vel': base_lin_vel,
                'base_lin_vel_local': base_lin_vel_local,
                'base_lin_vel_local_50window': base_lin_vel_local_50window,
                'fps': target_fps
            }
            
            # 如果启用镜像，添加镜像数据
            if enable_mirror:
                mirror_trans, mirror_rot, mirror_dof = create_mirror_data(
                    root_trans_offset_interp, root_rot_interp, dof_interp
                )
                
                # 计算镜像数据的增量和速度
                mirror_dof_increment = mirror_dof - default_joint_angles
                mirror_base_lin_vel, mirror_joint_velocities, mirror_base_lin_vel_local = calculate_velocities(
                    mirror_trans, mirror_dof, mirror_rot, target_fps
                )
                
                # 计算镜像数据的base_lin_vel_local的50窗口滑动平均
                mirror_base_lin_vel_local_50window = sliding_window_average(mirror_base_lin_vel_local, 50)
                
                # 添加镜像数据集
                mirror_key = f"{key}_mirror"
                dataset[mirror_key] = {
                    'root_trans_offset': mirror_trans,
                    'root_rot': mirror_rot,
                    'dof': mirror_dof,
                    'default_joint_angles': default_joint_angles,
                    'dof_increment': mirror_dof_increment,
                    'joint_velocities': mirror_joint_velocities,
                    'base_lin_vel': mirror_base_lin_vel,
                    'base_lin_vel_local': mirror_base_lin_vel_local,
                    'base_lin_vel_local_50window': mirror_base_lin_vel_local_50window,
                    'fps': target_fps
                }
                print(f"  添加镜像数据: {mirror_key}")
            
        except Exception as e:
            print(f"加载文件 {npy_file.name} 时出错: {e}")
    
    print(f"成功加载 {len(dataset)} 个数据文件")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"数据集已保存到: {output_path}")
    
    return dataset

def analyze_dataset(dataset):
    print("\n=== 数据集分析 ===")
    print(f"数据集包含 {len(dataset)} 个序列")
    
    total_steps = 0
    
    for key, data_dict in dataset.items():
        # 使用root_trans_offset来获取步数信息
        steps = data_dict['root_trans_offset'].shape[0]
        print(f"序列 '{key}': {steps} 步")
        print(f"  - root_trans_offset: {data_dict['root_trans_offset'].shape}")
        print(f"  - root_rot: {data_dict['root_rot'].shape}")
        print(f"  - dof: {data_dict['dof'].shape}")
        print(f"  - joint_velocities: {data_dict['joint_velocities'].shape}")
        print(f"  - base_lin_vel: {data_dict['base_lin_vel'].shape}")
        print(f"  - base_lin_vel_local: {data_dict['base_lin_vel_local'].shape}")
        print(f"  - base_lin_vel_local_50window: {data_dict['base_lin_vel_local_50window'].shape}")
        print(f"  - fps: {data_dict['fps']}")
        total_steps += steps
    
    print(f"\n总步数: {total_steps}")
    print(f"数据字段: root_trans_offset, root_rot, dof, default_joint_angles, dof_increment, joint_velocities, base_lin_vel, base_lin_vel_local, base_lin_vel_local_50window, fps")
    
    return total_steps

def main():
    parser = argparse.ArgumentParser(description='合并运动数据集并进行插值处理')
    parser.add_argument('--dataset_dir', '-d', type=str, default=dataset_dir,
                        help='数据集目录路径')
    parser.add_argument('--output_path', '-o', type=str, default=None,
                        help='输出文件路径')
    parser.add_argument('--target_fps', '-f', type=int, default=50,
                        help='目标帧率 (默认: 50)')
    parser.add_argument('--auto_save', '-s', action='store_true',
                        help='自动保存，不询问用户')
    parser.add_argument('--mirror', '-m', action='store_true',
                        help='启用镜像数据，将每个动作的镜像翻转也添加到数据集中')
    
    args = parser.parse_args()
    
    print(f"数据集目录: {args.dataset_dir}")
    print(f"目标帧率: {args.target_fps}")
    if args.mirror:
        print("镜像模式: 启用")

    dataset = merge_datasets(args.dataset_dir, target_fps=args.target_fps, enable_mirror=args.mirror)
    
    if dataset:
        analyze_dataset(dataset)
        
        if args.auto_save:
            save_choice = 'y'
        else:
            save_choice = input("\n是否保存合并后的数据集？(y/n): ")
            
        if save_choice.lower() in ['y', 'yes', '是']:
            if args.output_path:
                output_path = args.output_path
            else:
                output_path = input("请输入保存路径 (默认: ./output/merged_dataset.pkl): ")
                if not output_path:
                    output_path = "./output/merged_dataset.pkl"
            
            merge_datasets(args.dataset_dir, output_path, args.target_fps, args.mirror)
    else:
        print("未找到任何数据文件")

if __name__ == "__main__":
    main()
