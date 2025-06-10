# from isaacgym.torch_utils import *
# 在导入任何库之前设置环境变量
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import joblib
import glob

# mrlib_core imports
import mrlib.poselib.core.rotation3d as pRot
import mrlib.utils.rotation_conversions as sRot
from mrlib.utils import torch_utils
from mrlib.smpllib.smpl_parser import SMPL_Parser
from mrlib.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES, 
)
from mrlib.utils.rotation_conversions import axis_angle_to_matrix, fix_continous_dof
from mrlib.utils.smoothing_utils import gaussian_filter_1d_batch

# project-specific imports
from mrlib.utils.torch_humanoid_batch import Humanoid_Batch

# third-party imports
from scipy.spatial.transform import Rotation as scipyRot
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import open3d as o3d

# 配置常量
MOTION_FLAG = 1  # 替代全局变量
FPS = 30
PAST_FRAMES = 2
FUTURE_FRAMES = 7
TOTAL_FRAMES = PAST_FRAMES + FUTURE_FRAMES
TIME_SPAN = TOTAL_FRAMES / FPS

# 运动检测阈值常量
LATERAL_SPEED_THRESHOLD = 0.6
CONSISTENCY_THRESHOLD = 0.6  
PERSISTENCE_THRESHOLD = 0.8
MIN_CURRENT_VY = 0.4
TURN_THRESHOLD = 0.8  # rad/s

# 运动类型编码
MOTION_TYPES = {
    0: "walk_in_place",
    1: "left_turn_walk", 
    2: "right_turn_walk",
    3: "straight_walk",
    4: "left_turn_run",
    5: "right_turn_run", 
    6: "straight_run",
    7: "stand",
    8: "squat",
    9: "unknown"
}

def detect_turning_with_lateral_velocity(target_vels, frame_idx, window_size=9):
    """
    基于侧向速度的转向检测，比角速度更准确
    
    Args:
        target_vels: 速度序列 [[vx, vy, wz], ...], vy是侧向速度
        frame_idx: 当前帧索引
        window_size: 时间窗口大小（帧数），默认9帧=0.3秒
    
    Returns:
        tuple: (is_turning, turn_direction, confidence)
            - is_turning: bool, 是否在转向
            - turn_direction: int, 转向方向 (1=左转(+vy), -1=右转(-vy), 0=不转向)
            - confidence: float, 置信度 [0, 1]
    """
    # 确保输入是numpy数组
    if not isinstance(target_vels, np.ndarray):
        target_vels = np.array(target_vels)
    
    half_window = window_size // 2
    start_idx = max(0, frame_idx - half_window)
    end_idx = min(len(target_vels), frame_idx + half_window + 1)
    
    # 提取时间窗口内的速度数据
    window_vels = target_vels[start_idx:end_idx]
    
    if len(window_vels) < 3:  # 窗口太小，无法判断
        return False, 0, 0.0
    
    # 提取侧向速度vy（第1列）
    window_vy = window_vels[:, 1]
    
    # 方法1：平均侧向速度强度
    avg_vy = np.mean(window_vy)
    abs_avg_vy = abs(avg_vy)
    
    # 方法2：一致性检测（标准差相对于均值的比例）
    std_vy = np.std(window_vy)
    consistency = 1.0 - min(1.0, std_vy / (abs_avg_vy + 0.1))  # 避免除零
    
    # 方法3：持续性检测（同向侧向速度的占比）
    if abs_avg_vy > 0.05:  # 有明显侧向运动趋势
        same_direction_frames = np.sum(np.sign(window_vy) == np.sign(avg_vy))
        persistence = same_direction_frames / len(window_vy)
    else:
        persistence = 0.0
    
    # 使用全局常量替代硬编码值
    
    # 预检查：当前帧侧向速度必须足够大
    center_idx = len(window_vy) // 2
    current_frame_vy = abs(window_vy[center_idx]) if len(window_vy) > center_idx else abs_avg_vy
    if current_frame_vy < MIN_CURRENT_VY:
        return False, 0, 0.0
    
    # 多重条件判断
    conditions_met = 0
    
    # 条件1：平均侧向速度超过阈值
    if abs_avg_vy >= LATERAL_SPEED_THRESHOLD:
        conditions_met += 1
    
    # 条件2：侧向速度一致性高
    if consistency >= CONSISTENCY_THRESHOLD:
        conditions_met += 1
        
    # 条件3：持续性好
    if persistence >= PERSISTENCE_THRESHOLD:
        conditions_met += 1
    
    # 需要满足至少2个条件才认为是转向
    is_turning = conditions_met >= 2
    
    # 确定转向方向: +vy=左转, -vy=右转
    if is_turning:
        turn_direction = 1 if avg_vy > 0 else -1
    else:
        turn_direction = 0
    
    # 计算置信度 (0-1)
    confidence = min(1.0, (
        0.4 * min(1.0, abs_avg_vy / LATERAL_SPEED_THRESHOLD) +
        0.3 * consistency +
        0.3 * persistence
    ))
    
    return is_turning, turn_direction, confidence

def get_motion_type_with_temporal_analysis(base_motion_flag, vx, vy, target_vels, frame_idx):
    """
    基于时间窗口分析的运动类型判断（新版本）
    
    Args:
        base_motion_flag: 基础运动标志
        vx, vy: 当前帧的线速度
        target_vels: 完整的速度序列 [[vx, vy, wz], ...]
        frame_idx: 当前帧索引
    """
    speed = np.sqrt(vx**2 + vy**2)
    
    # 基础运动类型映射
    base_types = {
        0: "walk",
        1: "run", 
        2: "stand",
        3: "squat",
    }
    
    base_type = base_types.get(base_motion_flag, "unknown")
    
    # 使用时间窗口检测转向（基于侧向速度）
    is_turning, turn_direction, confidence = detect_turning_with_lateral_velocity(
        target_vels, frame_idx
    )
    
    # 根据运动参数细分
    if base_type == "walk":
        if speed < 0.05:  # 几乎静止
            return 0  # walk_in_place
        elif is_turning and confidence > 0.5:  # 高置信度转向
            if turn_direction > 0:
                return 1  # left_turn_walk  
            else:
                return 2  # right_turn_walk
        else:
            return 3  # straight_walk
            
    elif base_type == "run":
        if is_turning and confidence > 0.5:  # 高置信度转向
            if turn_direction > 0:
                return 4  # left_turn_run
            else:
                return 5  # right_turn_run
        else:
            return 6  # straight_run
            
    elif base_type == "stand":
        return 7  # stand
        
    elif base_type == "squat":
        return 8  # squat
        
    else:
        return 9  # unknown/other

def compute_yaw_from_quaternion(quat):
    """从四元数计算yaw角度 - 使用scipy"""
    # 使用scipy的标准转换，避免重复实现
    return scipyRot.from_quat(quat).as_euler('xyz')[2]

def compute_task_info(root_trans_offset, root_rot_quat, base_motion_flag):
    """
    计算task_info: target_vel和motion_type
    使用前2帧+后7帧的中心差分方法计算速度，更平滑且响应更快
    关键改进：将世界坐标系下的速度转换为机器人局部坐标系下的速度
    """
    
    target_vels = []
    angular_velocities = []  # 收集所有角速度用于时间窗口分析
    
    # 第一遍：计算所有帧的速度
    for i in range(len(root_trans_offset)):
        # 计算前2帧+后7帧的平均线速度和角速度
        if i >= PAST_FRAMES and i + FUTURE_FRAMES < len(root_trans_offset):
            # 有完整的前2帧和后7帧
            future_pos = root_trans_offset[i + FUTURE_FRAMES]    # t+7
            past_pos = root_trans_offset[i - PAST_FRAMES]        # t-2
            
            # 计算世界坐标系下的位移向量
            delta_pos_world = future_pos - past_pos
            
            # 计算世界坐标系下的速度
            vel_world = delta_pos_world / TIME_SPAN
            
            # *** 智能坐标变换：根据运动轨迹选择最佳参考坐标系 ***
            
            # 使用机器人朝向作为参考坐标系
            start_time_index = i - PAST_FRAMES
            reference_quat = root_rot_quat[start_time_index]
            reference_rotation = scipyRot.from_quat(reference_quat)
            vel_local = reference_rotation.inv().apply(vel_world)
            
            # 机器人坐标系下的速度
            vel_x = vel_local[0]  # 前进/后退方向
            vel_y = vel_local[1]  # 左/右方向
            
            # 计算角速度
            past_yaw = compute_yaw_from_quaternion(root_rot_quat[i - PAST_FRAMES])
            future_yaw = compute_yaw_from_quaternion(root_rot_quat[i + FUTURE_FRAMES])
            
            # 处理角度跳跃 (-π 到 π)
            delta_yaw = future_yaw - past_yaw
            if delta_yaw > np.pi:
                delta_yaw -= 2 * np.pi
            elif delta_yaw < -np.pi:
                delta_yaw += 2 * np.pi
                
            vel_z = delta_yaw / TIME_SPAN
            
        else:
            # 对于前2帧和后7帧，无法计算完整的窗口信息，设为0
            # 这些帧会在后续截断中被移除
            vel_x = vel_y = vel_z = 0.0
        
        target_vel = [vel_x, vel_y, vel_z]
        target_vels.append(target_vel)
        angular_velocities.append(vel_z)
    
    # 转换为numpy数组
    angular_velocities = np.array(angular_velocities)
    
    # *** 智能优化：基于整体轨迹重新计算直线运动的速度 ***
    start_pos = root_trans_offset[0]
    end_pos = root_trans_offset[-1]
    total_displacement = end_pos - start_pos
    trajectory_distance = np.linalg.norm(total_displacement[:2])
    
    use_trajectory_reference = False
    if trajectory_distance > 1.0:  # 有明显位移的运动
        # 计算轨迹偏离度
        mid_pos = root_trans_offset[len(root_trans_offset)//2]
        line_direction = total_displacement[:2] / trajectory_distance
        start_to_mid = mid_pos - start_pos
        projection_length = np.dot(start_to_mid[:2], line_direction)
        projection_point = start_pos[:2] + projection_length * line_direction
        trajectory_deviation = np.linalg.norm(mid_pos[:2] - projection_point)
        
        if trajectory_deviation < 0.2:
            # 直线运动：使用轨迹方向重新计算速度
            use_trajectory_reference = True
            trajectory_direction_2d = total_displacement[:2] / trajectory_distance
            trajectory_yaw = np.arctan2(trajectory_direction_2d[1], trajectory_direction_2d[0])
            trajectory_rotation = scipyRot.from_euler('z', trajectory_yaw)
            
            # 重新计算所有速度
            for i in range(len(root_trans_offset)):
                if i >= PAST_FRAMES and i + FUTURE_FRAMES < len(root_trans_offset):
                    future_pos = root_trans_offset[i + FUTURE_FRAMES]
                    past_pos = root_trans_offset[i - PAST_FRAMES]
                    delta_pos_world = future_pos - past_pos
                    vel_world = delta_pos_world / TIME_SPAN
                    
                    # 使用轨迹坐标系
                    vel_local = trajectory_rotation.inv().apply(vel_world)
                    target_vels[i] = [vel_local[0], vel_local[1], target_vels[i][2]]  # 保持角速度不变
            
            print(f"使用轨迹坐标系优化直线运动 (偏离={trajectory_deviation:.3f}m)")
    
    # 第二遍：基于完整的角速度序列计算运动类型
    motion_types = []
    for i in range(len(target_vels)):
        vel_x, vel_y, vel_z = target_vels[i]
        
        # 使用新的时间窗口转向检测方法（基于侧向速度）
        motion_type = get_motion_type_with_temporal_analysis(
            base_motion_flag, vel_x, vel_y, target_vels, i
        )
        motion_types.append(motion_type)
    
    return {
        "target_vel": np.array(target_vels),
        "motion_type": np.array(motion_types),
        "base_motion_flag": base_motion_flag
    }

def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not 'mocap_framerate' in  entry_data:
        return

    framerate = entry_data['mocap_framerate']
    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis = -1) # 72 total dof
    betas = entry_data['betas']
    gender = entry_data['gender']
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }
    
def process_motion(key_names, key_name_to_pkls, cfg):
    device = torch.device("cpu")
    
    # 默认关节角度（unitree_g1_29dof的标准姿态）
    # 顺序：左腿6个 + 右腿6个 + 腰部3个 + 左臂7个 + 右臂7个 = 29个关节
    default_joint_angles = np.array([
        -0.12, 0, 0, 0.45, -0.31, 0,    # 左腿6个关节：hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        -0.12, 0, 0, 0.45, -0.31, 0,    # 右腿6个关节：hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll  
        0, 0, 0,                        # 腰部3个关节：waist_yaw, waist_roll, waist_pitch
        0, 0.62, 0, 1.04, 0, 0, 0,      # 左臂7个关节：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        0, -0.62, 1.04, 0, 0, 0, 0      # 右臂7个关节：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
    ])
    
    humanoid_fk = Humanoid_Batch(cfg.robot) # load forward kinematics model
    num_augment_joint = len(cfg.robot.extend_config)

    #### Define corresonpdances between robots and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [robot_joint_names_augment.index(j) for j in robot_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]
    
    smpl_parser_n = SMPL_Parser(model_path=cfg.smpl_model_path, gender="neutral")
    shape_file = f"{cfg.output_path}/{cfg.robot.humanoid_type}/shape_optimized.pkl"
    shape_new, scale = joblib.load(shape_file) 
    
    all_data = {}
    total_files = len(key_names)
    for idx, data_key in enumerate(key_names):
        # print(f"正在处理 {idx+1}/{total_files}: {data_key}")
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None: 
            continue

        skip = int(amass_data['fps']//30) # 30 fps
        trans = torch.from_numpy(amass_data['trans'][::skip]) # root translation
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(amass_data['pose_aa'][::skip]).float() # 72 dof
        
        if N < 10:
            print("too short")
            continue

        with torch.no_grad():
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            root_pos = joints[:, 0:1]
            joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        joints[..., 2] -= verts[0, :, 2].min().item()
        
        offset = joints[:, 0] - trans
        root_trans_offset = (trans + offset).clone()

        gt_root_rot_quat = torch.from_numpy((scipyRot.from_rotvec(pose_aa_walk[:, :3]) * scipyRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).float() # can't directly use this 
        gt_root_rot = torch.from_numpy(scipyRot.from_quat(torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()).float() # so only use the heading. 
        
        dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))

        dof_pos_new = dof_pos.clone().requires_grad_(True)
        root_rot_new = gt_root_rot.clone().requires_grad_(True)
        root_pos_offset = torch.zeros(1, 3, requires_grad=True)
        optimizer = torch.optim.Adam([dof_pos_new, root_rot_new, root_pos_offset],lr=0.02)

        kernel_size = 5  # Size of the Gaussian kernel
        sigma = 0.75  # Standard deviation of the Gaussian kernel
        B, T, J, D = dof_pos_new.shape

        for iteration in range(cfg.get("fitting_iterations", 500)):
            pose_aa_robot_new = torch.cat([
                root_rot_new[None, :, None], 
                humanoid_fk.dof_axis * dof_pos_new, 
                torch.zeros((1, N, num_augment_joint, 3)).to(device)
            ], axis = 2)
            fk_return = humanoid_fk.fk_batch(pose_aa_robot_new, root_trans_offset[None, ] + root_pos_offset )
            
            if num_augment_joint > 0:
                diff = fk_return.global_translation_extend[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            else:
                diff = fk_return.global_translation[:, :, robot_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
                
            loss_g = diff.norm(dim = -1).mean() + 0.01 * torch.mean(torch.square(dof_pos_new))
            loss = loss_g
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])

            # if iteration % 100 == 0:  # 每100次迭代打印一次进度
            #     print(f"  {data_key} - 迭代 {iteration}/500, 损失: {loss.item() * 1000:.3f}")
            dof_pos_new.data = gaussian_filter_1d_batch(dof_pos_new.squeeze().transpose(1, 0)[None, ], kernel_size, sigma).transpose(2, 1)[..., None]

        dof_pos_new.data.clamp_(humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None])
        pose_aa_robot_new = torch.cat([root_rot_new[None, :, None], humanoid_fk.dof_axis * dof_pos_new, torch.zeros((1, N, num_augment_joint, 3)).to(device)], axis = 2)

        root_trans_offset_dump = (root_trans_offset + root_pos_offset ).clone()

        # move to ground
        # 1.using the lowest body pos in motion
        # height_diff = fk_return.global_translation[..., 2].min().item() 

        # 2.using the lowest point of mesh in motion
        combined_mesh = humanoid_fk.mesh_fk(pose_aa_robot_new[:, :1].detach(), root_trans_offset_dump[None, :1].detach())
        if isinstance(combined_mesh, o3d.geometry.TriangleMesh):
            height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()
        elif isinstance(combined_mesh, torch.Tensor):
            # Fallback if mesh loading failed and fk_results are returned instead
            # logging.warning("Received tensor instead of mesh, using joint positions for height calculation.")
            height_diff = combined_mesh[..., 2].min().item()
        else:
            height_diff = 0 # Default if something unexpected is returned
        
        root_trans_offset_dump[..., 2] -= height_diff
        joints_dump = joints.numpy().copy()
        joints_dump[..., 2] -= height_diff
        
        root_trans_offset_np = root_trans_offset_dump.squeeze().detach().numpy()
        root_trans_vel_np = np.diff(root_trans_offset_np, axis=0) * FPS

        root_trans_offset_np=root_trans_offset_np[:-1]
        dof_np = dof_pos_new.squeeze().detach().numpy()[:-1]
        root_rot_np = scipyRot.from_rotvec(root_rot_new.detach().numpy()).as_quat()[:-1]
        joints_np = joints_dump.copy()[:-1]

        lowest_height = np.min(joints_np[:,:,2])
        
        joints_np[..., 2] -= lowest_height
        root_trans_offset_np[..., 2] -= lowest_height

        # 计算增量关节角度（当前角度 - 默认角度）
        dof_increment = dof_np - default_joint_angles[None, :]
        
        # 先在完整数据上计算task_info，然后再截断
        # 检查数据长度是否足够计算task_info（需要前2帧+后7帧，至少10帧）
        if len(root_trans_offset_np) < 10:
            print(f"⚠️  数据太短({len(root_trans_offset_np)}帧)，跳过task_info计算")
            task_info = None
            # 不截断，保持原始数据
        else:
            # 先在完整数据上计算task_info（这样有足够的前后帧）
            task_info_full = compute_task_info(root_trans_offset_np, root_rot_np, MOTION_FLAG)
            
            # 然后截断前2帧和后7帧（包括task_info）
            start_idx = 2  # 跳过前2帧
            end_idx = len(root_trans_offset_np) - 7  # 跳过后7帧
            
            # 截断所有相关数组
            root_trans_offset_np = root_trans_offset_np[start_idx:end_idx]
            dof_np = dof_np[start_idx:end_idx] 
            root_rot_np = root_rot_np[start_idx:end_idx]
            joints_np = joints_np[start_idx:end_idx]
            dof_increment = dof_increment[start_idx:end_idx]
            
            # 截断task_info
            task_info = {
                "target_vel": task_info_full["target_vel"][start_idx:end_idx],
                "motion_type": task_info_full["motion_type"][start_idx:end_idx],
                "base_motion_flag": task_info_full["base_motion_flag"]
            }
            
            truncate_length = end_idx - start_idx
            print(f"✅ 计算task_info (前2+后7帧方法)，截断后长度: {truncate_length}帧")

        data_dump = {
                    "root_trans_offset": root_trans_offset_np,
                    # "pose_aa": pose_aa_robot_new.squeeze().detach().numpy(),
                    "root_height": root_trans_offset_np[...,2],
                    # "root_trans_vel": root_trans_vel_np,
                    "dof": dof_np, 
                    "dof_increment": dof_increment,  
                    "default_joint_angles": default_joint_angles,  
                    "root_rot": root_rot_np,
                    "smpl_joints": joints_np, # 用于可视化
                    "fps": FPS
                    }
        
        # 添加task_info到数据中
        if task_info is not None:
            data_dump["task_info"] = task_info
        all_data[data_key] = data_dump
        print(f"✅ 完成处理: {data_key}")
    return all_data
        
@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg: DictConfig):

    amass_root = cfg.amass_path
    motion_name = os.path.basename(os.path.normpath(amass_root))
    
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    if not all_pkls:
        print(f"Error: No .npz files found in {amass_root}. Please check the amass_path.")
        return

    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}
    key_names = ["_".join(data_path.split("/")[split_len:]).replace(".npz", "") for data_path in all_pkls]
    
    from multiprocessing import Pool
    print(f"找到 {len(key_names)} 个运动文件")
    
    # 可以通过配置选择是否使用多进程
    use_multiprocessing = cfg.get("use_multiprocessing", True)
    
    if not use_multiprocessing or len(key_names) <= 5:
        print("使用单进程处理...")
        all_data = process_motion(key_names, key_name_to_pkls, cfg)
    else:
        print("使用多进程并行处理...")
        jobs = key_names
        num_jobs = min(30, len(key_names))  # 不要超过文件数量
        chunk = np.ceil(len(jobs)/num_jobs).astype(int)
        jobs= [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
        job_args = [(jobs[i], key_name_to_pkls, cfg) for i in range(len(jobs))]
        
        try:
            pool = Pool(num_jobs)   # multi-processing
            all_data_list = pool.starmap(process_motion, job_args)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print("用户中断处理")
            pool.terminate()
            pool.join()
            return
        
        print("合并处理结果...")
        all_data = {}
        for data_dict in all_data_list:
            all_data.update(data_dict)
    # 保存处理后的运动数据
    os.makedirs(f'{cfg.output_path}/{cfg.robot.humanoid_type}', exist_ok=True)
    
    if len(all_data) == 1:
        # 如果只有一个运动文件，保存为motion_im.p（兼容旧版本）
        joblib.dump(all_data[list(all_data.keys())[0]], f'{cfg.output_path}/{cfg.robot.humanoid_type}/motion_im.p')
        print(f"运动重定向完成，保存到 {cfg.output_path}/{cfg.robot.humanoid_type}/motion_im.p")
    else:
        # 如果有多个运动文件，保存到以数据集名称命名的子文件夹
        os.makedirs(f'{cfg.output_path}/{cfg.robot.humanoid_type}/{motion_name}', exist_ok=True)
        joblib.dump(all_data, f'{cfg.output_path}/{cfg.robot.humanoid_type}/{motion_name}/{motion_name}.pkl')
        print(f"运动重定向完成，处理了{len(all_data)}个运动文件，保存到 {cfg.output_path}/{cfg.robot.humanoid_type}/{motion_name}/{motion_name}.pkl")
    
    print(f"数据包含：关节角度(dof)、增量角度(dof_increment)、默认角度(default_joint_angles)、任务信息(task_info)等")
    print(f"task_info包含：target_vel(目标速度[vx,vy,wz])、motion_type(运动类型0-9)、base_motion_flag(原始标志)")

if __name__ == "__main__":
    main()
