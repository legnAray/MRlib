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

def key_call_back(keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
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
    else:
        print("未映射的按键:", chr(keycode))
    
    
@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    device = torch.device("cpu")
    humanoid_xml = cfg.robot.asset.assetFileName
    sk_tree = SkeletonTree.from_mjcf(humanoid_xml)
    
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False
    
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
    
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    
    RECORDING = False
    
    mj_model.opt.timestep = dt
    
    print("控制说明:")
    print("  空格键: 暂停/继续")
    print("  R键: 重置到开始")
    print("  Q键: 上一个运动")
    print("  E键: 下一个运动")
    
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

            if cfg.get("debug", False):
                print(f"时间 {curr_time}: 根位置 {mj_data.qpos[:3]}")
                print(f"时间 {curr_time}: 根高度 {curr_motion['root_height'][curr_time]}")

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

