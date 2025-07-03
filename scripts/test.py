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
        1 / 50,
        False,
    )

    default_joint_angles =[
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
    # default_joint_angles = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    pose1 =[-0.62303371,  0.22637329,  0.53801364,  0.10554837,  0.27690181,
            0.        ,  0.12103113, -0.01303962, -0.14333688,  0.20302998,
            0.04823458,  0.        ,  0.28510013,  0.04176901, -0.0518946 ,
            0.42876944, -0.5719352 ,  0.05238509,  0.06529101, -0.02787924,
            0.14837351,  0.        , -0.18098742, 0.50347577 , -0.42569768,
            -0.49031202, -0.00954138,  0.07539803,  0.        ]
    pose2 =pose1
    pose11 = [x + y for x, y in zip(default_joint_angles, pose1)]
    print(pose11)
    pose22 = [x + y for x, y in zip(default_joint_angles, pose2)]

    pose3=[-0.7662,  0.3328,  0.5377,  0.6938, -0.0479,  0.0000, -0.1205,  0.0595,
            -0.0333,  0.6888, -0.1910,  0.0000,  0.3969,  0.0361, -0.0798,  0.3849,
            0.0288,  0.1168,  1.0726, -0.0040,  0.1496,  0.0000, -0.0609, -0.1192,
            -0.2938,  0.4893, -0.0029,  0.0634,  0.0000]
    pose4=[-0.7942,  0.3220,  0.5766,  0.6203, -0.0407,  0.0000, -0.0729,  0.0507,
            -0.0414,  0.7036, -0.1892,  0.0000,  0.3973,  0.0384, -0.0883,  0.4075,
            0.0302,  0.1741,  1.0966,  0.0097,  0.1552,  0.0000, -0.0809, -0.1512,
            -0.2789,  0.4492, -0.0015,  0.0550,  0.0000]

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)

    RECORDING = False

    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        t = 0
        while viewer.is_running():
            step_start = time.time()
            if t % 2 == 0:
                mj_data.qpos[7:] = pose11
            else:
                mj_data.qpos[7:] = pose22
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            t += 1
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
