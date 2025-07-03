import torch
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from scipy.ndimage import gaussian_filter1d


def quat_correct(quat):
    """Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order)"""
    for q in range(1, quat.shape[0]):
        if np.linalg.norm(quat[q - 1] - quat[q], axis=0) > np.linalg.norm(
            quat[q - 1] + quat[q], axis=0
        ):
            quat[q] = -quat[q]
    return quat


def quat_smooth_window(quats, sigma=5):
    """Smooth quaternions using Gaussian filter"""
    quats = quat_correct(quats)
    quats = gaussian_filter1d(quats, sigma, axis=0)
    quats /= np.linalg.norm(quats, axis=1)[:, None]
    return quats


def smooth_smpl_quat_window(pose_aa, sigma=5):
    """
    Smooth SMPL quaternions using window-based filtering
    From SMPLSim/smpl_sim/utils/transform_utils.py
    """
    batch = pose_aa.shape[0]
    pose_quat = sRot.from_rotvec(pose_aa.reshape(-1, 3)).as_quat().reshape(batch, -1, 4)
    pose_quat = pose_quat[:, :, [1, 2, 3, 0]].copy()

    quats_all = []
    for i in range(pose_quat.shape[1]):
        quats = pose_quat[:, i, :].copy()
        quats_all.append(quat_smooth_window(quats, sigma))

    pose_quat_smooth = np.stack(quats_all, axis=1)[:, :, [3, 0, 1, 2]]

    pose_rot_vec = (
        sRot.from_quat(pose_quat_smooth.reshape(-1, 4))
        .as_rotvec()
        .reshape(batch, -1, 3)
    )
    return pose_rot_vec


def smooth_smpl_quat_tensor(pose_aa, sigma=5):
    """
    Tensor version of smooth_smpl_quat_window that can handle both numpy and torch tensors
    """
    if isinstance(pose_aa, torch.Tensor):
        pose_aa_np = pose_aa.detach().numpy()
        return_torch = True
    else:
        pose_aa_np = pose_aa
        return_torch = False

    result = smooth_smpl_quat_window(pose_aa_np, sigma)

    if return_torch:
        return torch.from_numpy(result).float()
    else:
        return result


def fix_continous_dof(dof):
    """
    Fix continuous DOF to prevent large jumps in joint angles
    From SMPLSim/smpl_sim/utils/pytorch3d_transforms.py

    This function is not perfect. For instance, it does not fix the wrap around problem.
    """
    if isinstance(dof, torch.Tensor):
        was_torch = True
        device = dof.device
        dof_np = dof.detach().cpu()
    else:
        was_torch = False
        dof_np = torch.from_numpy(dof) if isinstance(dof, np.ndarray) else dof

    assert (
        len(dof_np.shape) == 3
    ), f"Expected 3D tensor (T, J, 3), got {dof_np.shape}"  # T, J, 3
    T = dof_np.shape[0] - 1
    for t in range(1, T):
        diff = dof_np[t] - dof_np[t - 1]
        times = 0
        while diff.abs().max().item() >= 3:
            change_joints = diff.abs().numpy().sum(axis=-1) >= 3
            dof_change = dof_np[t][change_joints].clone()
            dof_change[:, 0] = np.pi + dof_change[:, 0]
            dof_change[:, 1] = np.pi - dof_change[:, 1]
            dof_change[:, 2] = np.pi + dof_change[:, 2]
            dof_change[dof_change > np.pi] -= np.pi * 2
            dof_change[dof_change < -np.pi] += np.pi * 2
            dof_np[t][change_joints] = dof_change
            diff = dof_np[t] - dof_np[t - 1]
            times += 1
            if times > 1:
                break

    if was_torch:
        return dof_np.to(device)
    else:
        return dof_np.numpy() if isinstance(dof, np.ndarray) else dof_np


def fix_continous_smpl_dof(dof):
    """
    Alias for fix_continous_dof for SMPL-specific usage
    """
    return fix_continous_dof(dof)
