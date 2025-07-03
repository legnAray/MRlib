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
MOTION_FLAG = 0  # 替代全局变量
FPS = 30
PAST_FRAMES = 2
FUTURE_FRAMES = 7
TOTAL_FRAMES = PAST_FRAMES + FUTURE_FRAMES
TIME_SPAN = TOTAL_FRAMES / FPS


def compute_raw_derivative(data: np.ndarray, dt: float) -> np.ndarray:
    """计算数据的导数（简单差分）"""
    d = (data[1:] - data[:-1]) / dt
    return np.vstack([d, d[-1:]])


def compute_yaw_from_quaternion(quat):
    """从四元数计算yaw角度 - 使用scipy"""
    # 使用scipy的标准转换，避免重复实现
    return scipyRot.from_quat(quat).as_euler("xyz")[2]


def compute_task_info(root_trans_offset, root_rot_quat):
    """
    计算task_info: 只计算target_vel
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
            future_pos = root_trans_offset[i + FUTURE_FRAMES]  # t+7
            past_pos = root_trans_offset[i - PAST_FRAMES]  # t-2

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
        mid_pos = root_trans_offset[len(root_trans_offset) // 2]
        line_direction = total_displacement[:2] / trajectory_distance
        start_to_mid = mid_pos - start_pos
        projection_length = np.dot(start_to_mid[:2], line_direction)
        projection_point = start_pos[:2] + projection_length * line_direction
        trajectory_deviation = np.linalg.norm(mid_pos[:2] - projection_point)

        if trajectory_deviation < 0.2:
            # 直线运动：使用轨迹方向重新计算速度
            use_trajectory_reference = True
            trajectory_direction_2d = total_displacement[:2] / trajectory_distance
            trajectory_yaw = np.arctan2(
                trajectory_direction_2d[1], trajectory_direction_2d[0]
            )
            trajectory_rotation = scipyRot.from_euler("z", trajectory_yaw)

            # 重新计算所有速度
            for i in range(len(root_trans_offset)):
                if i >= PAST_FRAMES and i + FUTURE_FRAMES < len(root_trans_offset):
                    future_pos = root_trans_offset[i + FUTURE_FRAMES]
                    past_pos = root_trans_offset[i - PAST_FRAMES]
                    delta_pos_world = future_pos - past_pos
                    vel_world = delta_pos_world / TIME_SPAN

                    # 使用轨迹坐标系
                    vel_local = trajectory_rotation.inv().apply(vel_world)
                    target_vels[i] = [
                        vel_local[0],
                        vel_local[1],
                        target_vels[i][2],
                    ]  # 保持角速度不变

            print(f"使用轨迹坐标系优化直线运动 (偏离={trajectory_deviation:.3f}m)")

    return {"target_vel": np.array(target_vels)}


def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not "mocap_framerate" in entry_data:
        return

    framerate = entry_data["mocap_framerate"]
    root_trans = entry_data["trans"]
    pose_aa = np.concatenate(
        [entry_data["poses"][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1
    )  # 72 total dof
    betas = entry_data["betas"]
    gender = entry_data["gender"]
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


def process_motion(key_names, key_name_to_pkls, cfg):
    device = torch.device("cpu")

    # 默认关节角度（unitree_g1_29dof的标准姿态）
    # 顺序：左腿6个 + 右腿6个 + 腰部3个 + 左臂7个 + 右臂7个 = 29个关节
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
            0.62,
            0,
            1.04,
            0,
            0,
            0,  # 左臂7个关节：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
            0,
            -0.62,
            0,
            1.04,
            0,
            0,
            0,  # 右臂7个关节：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        ]
    )

    humanoid_fk = Humanoid_Batch(cfg.robot)  # load forward kinematics model
    num_augment_joint = len(cfg.robot.extend_config)

    #### Define corresonpdances between robots and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [
        robot_joint_names_augment.index(j) for j in robot_joint_pick
    ]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path=cfg.smpl_model_path, gender="neutral")

    # 使用pickle加载形状参数
    shape_file = f"{cfg.output_path}/{cfg.robot.humanoid_type}/shape_optimized.pkl"
    if not os.path.exists(shape_file):
        raise FileNotFoundError(f"Shape file not found: {shape_file}")

    with open(shape_file, "rb") as f:
        shape_new, scale = pickle.load(f)

    all_data = {}
    total_files = len(key_names)
    for idx, data_key in enumerate(key_names):
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None:
            continue

        skip = int(amass_data["fps"] // 30)  # 30 fps
        trans = torch.from_numpy(amass_data["trans"][::skip])  # root translation
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(amass_data["pose_aa"][::skip]).float()  # 72 dof

        if N < 10:
            print("too short")
            continue

        with torch.no_grad():
            verts, joints = smpl_parser_n.get_joints_verts(
                pose_aa_walk, shape_new, trans
            )
            root_pos = joints[:, 0:1]
            joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        joints[..., 2] -= verts[0, :, 2].min().item()

        offset = joints[:, 0] - trans
        root_trans_offset = (trans + offset).clone()

        gt_root_rot_quat = torch.from_numpy(
            (
                scipyRot.from_rotvec(pose_aa_walk[:, :3])
                * scipyRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
            ).as_quat()
        ).float()  # can't directly use this
        gt_root_rot = torch.from_numpy(
            scipyRot.from_quat(
                torch_utils.calc_heading_quat(gt_root_rot_quat)
            ).as_rotvec()
        ).float()  # so only use the heading.

        dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))

        dof_pos_new = dof_pos.clone().requires_grad_(True)
        root_rot_new = gt_root_rot.clone().requires_grad_(True)
        root_pos_offset = torch.zeros(1, 3, requires_grad=True)
        optimizer = torch.optim.Adam(
            [dof_pos_new, root_rot_new, root_pos_offset], lr=0.02
        )

        kernel_size = 5  # Size of the Gaussian kernel
        sigma = 0.75  # Standard deviation of the Gaussian kernel
        B, T, J, D = dof_pos_new.shape

        for iteration in range(cfg.get("fitting_iterations", 500)):
            pose_aa_robot_new = torch.cat(
                [
                    root_rot_new[None, :, None],
                    humanoid_fk.dof_axis * dof_pos_new,
                    torch.zeros((1, N, num_augment_joint, 3)).to(device),
                ],
                axis=2,
            )
            fk_return = humanoid_fk.fk_batch(
                pose_aa_robot_new, root_trans_offset[None,] + root_pos_offset
            )

            if num_augment_joint > 0:
                diff = (
                    fk_return.global_translation_extend[:, :, robot_joint_pick_idx]
                    - joints[:, smpl_joint_pick_idx]
                )
            else:
                diff = (
                    fk_return.global_translation[:, :, robot_joint_pick_idx]
                    - joints[:, smpl_joint_pick_idx]
                )

            loss_g = diff.norm(dim=-1).mean() + 0.01 * torch.mean(
                torch.square(dof_pos_new)
            )
            loss = loss_g

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dof_pos_new.data.clamp_(
                humanoid_fk.joints_range[:, 0, None],
                humanoid_fk.joints_range[:, 1, None],
            )

            dof_pos_new.data = gaussian_filter_1d_batch(
                dof_pos_new.squeeze().transpose(1, 0)[None,], kernel_size, sigma
            ).transpose(2, 1)[..., None]

        dof_pos_new.data.clamp_(
            humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None]
        )
        pose_aa_robot_new = torch.cat(
            [
                root_rot_new[None, :, None],
                humanoid_fk.dof_axis * dof_pos_new,
                torch.zeros((1, N, num_augment_joint, 3)).to(device),
            ],
            axis=2,
        )

        root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()

        # move to ground
        # 1.using the lowest body pos in motion
        # height_diff = fk_return.global_translation[..., 2].min().item()

        # 2.using the lowest point of mesh in motion
        combined_mesh = humanoid_fk.mesh_fk(
            pose_aa_robot_new[:, :1].detach(), root_trans_offset_dump[None, :1].detach()
        )
        if isinstance(combined_mesh, o3d.geometry.TriangleMesh):
            height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()
        elif isinstance(combined_mesh, torch.Tensor):
            # Fallback if mesh loading failed and fk_results are returned instead
            # logging.warning("Received tensor instead of mesh, using joint positions for height calculation.")
            height_diff = combined_mesh[..., 2].min().item()
        else:
            height_diff = 0  # Default if something unexpected is returned

        root_trans_offset_dump[..., 2] -= height_diff
        joints_dump = joints.numpy().copy()
        joints_dump[..., 2] -= height_diff

        root_trans_offset_np = root_trans_offset_dump.squeeze().detach().numpy()

        dof_np = dof_pos_new.squeeze().detach().numpy()
        root_rot_np = scipyRot.from_rotvec(root_rot_new.detach().numpy()).as_quat()
        joints_np = joints_dump.copy()

        lowest_height = np.min(joints_np[:, :, 2])

        joints_np[..., 2] -= lowest_height
        root_trans_offset_np[..., 2] -= lowest_height

        # 计算增量关节角度（当前角度 - 默认角度）
        dof_increment = dof_np - default_joint_angles[None, :]

        # ===================== 计算机器人线速度 =====================
        # 使用您提供的方法计算机器人线速度
        dt = 1.0 / FPS  # 时间步长

        # 计算世界坐标系下的线速度
        base_lin_vel = compute_raw_derivative(root_trans_offset_np, dt)  # [N, 3]

        # 转换为局部坐标系下的线速度
        base_orientations = scipyRot.from_quat(root_rot_np)
        base_lin_vel_local = np.stack(
            [R.as_matrix().T @ v for R, v in zip(base_orientations, base_lin_vel)]
        )

        print(f"✅ 计算机器人线速度 (局部坐标系)")

        # ===================== 计算关节局部速度 =====================
        joint_velocities = []  # 存储每帧各关节的角速度

        # 逐帧计算关节角速度 - 使用中心差分法提高精度
        for i in range(len(dof_np)):
            if i == 0:
                # 第一帧：使用向前差分
                joint_angle_diff = dof_np[i + 1] - dof_np[i]
                joint_angular_vel = joint_angle_diff * FPS
            elif i == len(dof_np) - 1:
                # 最后一帧：使用向后差分
                joint_angle_diff = dof_np[i] - dof_np[i - 1]
                joint_angular_vel = joint_angle_diff * FPS
            else:
                # 中间帧：使用中心差分（更高精度）
                joint_angle_diff = dof_np[i + 1] - dof_np[i - 1]
                joint_angular_vel = (
                    joint_angle_diff * FPS / 2.0
                )  # 除以2因为跨越了2个时间步

            joint_velocities.append(joint_angular_vel)

        # 转换为numpy数组
        joint_velocities = np.array(joint_velocities)  # [N, num_joints]

        if len(root_trans_offset_np) < 10:
            print(f"⚠️  数据太短({len(root_trans_offset_np)}帧)，跳过task_info计算")
            task_info = None
            # 不截断，保持原始数据，但仍然需要将base_lin_vel_local添加到数据中
        else:
            task_info_full = compute_task_info(root_trans_offset_np, root_rot_np)

            # 然后截断前2帧和后7帧（包括task_info）
            start_idx = 2  # 跳过前2帧
            end_idx = len(root_trans_offset_np) - 7  # 跳过后7帧

            # 截断所有相关数组
            root_trans_offset_np = root_trans_offset_np[start_idx:end_idx]
            dof_np = dof_np[start_idx:end_idx]
            root_rot_np = root_rot_np[start_idx:end_idx]
            joints_np = joints_np[start_idx:end_idx]
            dof_increment = dof_increment[start_idx:end_idx]

            # 截断关节速度和机器人线速度
            joint_velocities = joint_velocities[start_idx:end_idx]
            base_lin_vel_local = base_lin_vel_local[start_idx:end_idx]

            # 截断task_info
            task_info = {
                "target_vel": task_info_full["target_vel"][start_idx:end_idx],
            }

            truncate_length = end_idx - start_idx
            print(f"✅ 计算task_info (前2+后7帧方法)，截断后长度: {truncate_length}帧")

        data_dump = {
            "root_trans_offset": root_trans_offset_np,
            "root_rot": root_rot_np,
            "root_height": root_trans_offset_np[..., 2],
            "dof": dof_np,
            "dof_increment": dof_increment,
            "default_joint_angles": default_joint_angles,
            "joint_velocities": joint_velocities,  # 各关节局部角速度 [N, num_joints]
            "base_lin_vel_local": base_lin_vel_local,  # 机器人局部线速度 [N, 3]
            # "pose_aa": pose_aa_robot_new.squeeze().detach().numpy(),
            "smpl_joints": joints_np,  # 用于可视化
            "fps": FPS,
            "base_motion_flag": MOTION_FLAG,
        }

        if task_info is not None:
            data_dump["task_vel"] = task_info["target_vel"][:, :2]  # 不要角速度

        all_data[data_key] = data_dump
        print(f"✅ 完成处理: {data_key}")
    return all_data


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg: DictConfig):

    amass_root = cfg.amass_path
    motion_name = os.path.basename(os.path.normpath(amass_root))

    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    if not all_pkls:
        print(
            f"Error: No .npz files found in {amass_root}. Please check the amass_path."
        )
        return

    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {
        "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path
        for data_path in all_pkls
    }
    key_names = [
        "_".join(data_path.split("/")[split_len:]).replace(".npz", "")
        for data_path in all_pkls
    ]

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
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        jobs = [jobs[i : i + chunk] for i in range(0, len(jobs), chunk)]
        job_args = [(jobs[i], key_name_to_pkls, cfg) for i in range(len(jobs))]

        try:
            pool = Pool(num_jobs)  # multi-processing
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
    # 保存处理后的运动数据（统一使用多运动格式）
    os.makedirs(f"{cfg.output_path}/{cfg.robot.humanoid_type}", exist_ok=True)

    # 统一保存为多运动格式的字典，即使只有一个运动
    output_path = f"{cfg.output_path}/{cfg.robot.humanoid_type}/{motion_name}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(all_data, f)
    print(f"运动重定向完成，处理了{len(all_data)}个运动文件，保存到 {output_path}")

    print(
        f"数据包含：关节角度(dof)、增量角度(dof_increment)、默认角度(default_joint_angles)、任务信息(task_info)等"
    )
    print(f"task_info包含：target_vel(目标速度[vx,vy,wz])、base_motion_flag(原始标志)")
    print(f"新增数据字段：")
    print(f"  - joint_velocities: 各关节角速度 [N, num_joints]")
    print(f"  - base_lin_vel_local: 机器人局部线速度 [N, 3]")


if __name__ == "__main__":
    main()
