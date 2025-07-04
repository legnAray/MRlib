# 在导入任何库之前设置环境变量
import os

os.environ["OMP_NUM_THREADS"] = "1"
import sys

sys.path.append(os.getcwd())

import torch
import numpy as np
from mrlib.utils import torch_utils
from mrlib.poselib.skeleton.skeleton3d import (
    SkeletonTree,
    SkeletonMotion,
    SkeletonState,
)
import mrlib.poselib.core.rotation3d as pRot
import mrlib.utils.rotation_conversions as sRot
from mrlib.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

from scipy.spatial.transform import Rotation as scipyRot
import pickle  # 使用Python标准库pickle
from tqdm import tqdm
from mrlib.utils.rotation_conversions import axis_angle_to_matrix, fix_continous_dof
import hydra
from hydra.utils import to_absolute_path
from mrlib.smpllib.smpl_joint_names import (
    SMPL_MUJOCO_NAMES,
    SMPL_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
import yaml
from mrlib.utils.torch_humanoid_batch import Humanoid_Batch
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg: DictConfig) -> None:

    humanoid_fk = Humanoid_Batch(cfg.robot)  # load forward kinematics model

    #### Define corresonpdances between robots and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in cfg.robot.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.robot.joint_matches]
    robot_joint_pick_idx = [
        robot_joint_names_augment.index(j) for j in robot_joint_pick
    ]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    #### Preparing fitting varialbes
    device = torch.device("cpu")
    pose_aa_robot = np.repeat(
        np.repeat(
            scipyRot.identity().as_rotvec()[
                None,
                None,
                None,
            ],
            humanoid_fk.num_bodies,
            axis=2,
        ),
        1,
        axis=1,
    )
    pose_aa_robot = torch.from_numpy(pose_aa_robot).float()

    ###### prepare SMPL default pose
    pose_aa_stand = np.zeros((1, 72))
    pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)

    for modifiers in cfg.robot.smpl_pose_modifier:
        modifier_key = list(modifiers.keys())[0]
        modifier_value = list(modifiers.values())[0]
        pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index(modifier_key)] = (
            scipyRot.from_euler("xyz", eval(modifier_value), degrees=False).as_rotvec()
        )

    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))
    smpl_parser_n = SMPL_Parser(model_path=cfg.smpl_model_path, gender="neutral")

    ###### Shape fitting
    trans = torch.zeros([1, 3])
    beta = torch.zeros([1, 10])
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans)
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset

    fk_return = humanoid_fk.fk_batch(pose_aa_robot[None,], root_trans_offset[None, 0:1])

    shape_new = torch.zeros([1, 10], device=device, requires_grad=True)
    scale = torch.ones([1], device=device, requires_grad=True)
    optimizer_shape = torch.optim.Adam([shape_new, scale], lr=0.1)

    train_iterations = 3000
    print("start fitting shapes")
    pbar = tqdm(range(train_iterations))
    for iteration in pbar:
        verts, joints = smpl_parser_n.get_joints_verts(
            pose_aa_stand, shape_new, trans[0:1]
        )  # fitted smpl shape
        root_pos = joints[:, 0]
        joints = (joints - joints[:, 0]) * scale + root_pos
        if len(cfg.robot.extend_config) > 0:
            diff = (
                fk_return.global_translation_extend[:, :, robot_joint_pick_idx]
                - joints[:, smpl_joint_pick_idx]
            )
        else:
            diff = (
                fk_return.global_translation[:, :, robot_joint_pick_idx]
                - joints[:, smpl_joint_pick_idx]
            )

        # loss_g = diff.norm(dim = -1).mean()
        loss_g = diff.norm(dim=-1).square().sum()

        loss = loss_g
        pbar.set_description_str(f"{iteration} - Loss: {loss.item() * 1000}")

        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()

    # print the fitted shape and scale parameters
    print("shape:", shape_new.detach())
    print("scale:", scale)

    if cfg.get("vis", False):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt

        j3d = (
            fk_return.global_translation_extend[0, :, robot_joint_pick_idx, :]
            .detach()
            .numpy()
        )
        j3d = j3d - j3d[:, 0:1]
        j3d_joints = joints[:, smpl_joint_pick_idx].detach().numpy()
        j3d_joints = j3d_joints - j3d_joints[:, 0:1]
        idx = 0
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(90, 0)
        ax.scatter(
            j3d[idx, :, 0],
            j3d[idx, :, 1],
            j3d[idx, :, 2],
            label="Humanoid Shape",
            c="blue",
        )
        ax.scatter(
            j3d_joints[idx, :, 0],
            j3d_joints[idx, :, 1],
            j3d_joints[idx, :, 2],
            label="Fitted Shape",
            c="red",
        )

        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")
        drange = 1
        ax.set_xlim(-drange, drange)
        ax.set_ylim(-drange, drange)
        ax.set_zlim(-drange, drange)
        ax.legend()
        plt.show()

    output_dir = f"{cfg.output_path}/{cfg.robot.humanoid_type}"
    os.makedirs(output_dir, exist_ok=True)

    # 使用pickle保存形状参数
    with open(f"{output_dir}/shape_optimized.pkl", "wb") as f:
        pickle.dump((shape_new.detach(), scale), f)
    print(f"形状参数已保存为pickle格式: {output_dir}/shape_optimized.pkl")


if __name__ == "__main__":
    main()
