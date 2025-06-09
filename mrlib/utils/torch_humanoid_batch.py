import glob
import os
import sys
import pdb
import os.path as osp
import torch 
from collections import defaultdict


import numpy as np
# import smpl_sim.utils.rotation_conversions as tRot
from . import rotation_conversions as tRot
from scipy.spatial.transform import Rotation as sRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import mrlib.poselib.core.rotation3d as pRot
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO
import copy
from collections import OrderedDict
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from stl import mesh
import logging
import open3d as o3d

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Humanoid_Batch:

    def __init__(self, cfg, device = torch.device("cpu")):
        self.cfg = cfg
        self.mjcf_file = cfg.asset.assetFileName
        
        parser = XMLParser(remove_blank_text=True)

        mjcf_path = os.path.join(hydra.utils.get_original_cwd(), self.mjcf_file)
        if not os.path.exists(mjcf_path):
            raise FileNotFoundError(f"Could not find mjcf file: {mjcf_path} (resolved from {self.mjcf_file})")

        tree = parse(BytesIO(open(mjcf_path, "rb").read()), parser=parser,)
        self.dof_axis = []
        joints = sorted([j.attrib['name'] for j in tree.getroot().find("worldbody").findall('.//joint')])
        motors = sorted([m.attrib['name'] for m in tree.getroot().find("actuator").getchildren()])
        assert len(motors) > 0, "No motors found in the mjcf file"
        
        self.num_dof = len(motors) 
        self.num_extend_dof = self.num_dof
        
        self.mjcf_data = mjcf_data = self.from_mjcf(self.mjcf_file)
        self.body_names = copy.deepcopy(mjcf_data['node_names'])
        self._parents = mjcf_data['parent_indices']
        self.body_names_augment = copy.deepcopy(mjcf_data['node_names'])
        self._offsets = mjcf_data['local_translation'][None, ].to(device)
        self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
        self.actuated_joints_idx = np.array([self.body_names.index(k) for k, v in mjcf_data['body_to_joint'].items()])
        
        for m in motors:
            if not m in joints:
                print(m)
        
        if "type" in tree.getroot().find("worldbody").findall('.//joint')[0].attrib and tree.getroot().find("worldbody").findall('.//joint')[0].attrib['type'] == "free":
            for j in tree.getroot().find("worldbody").findall('.//joint')[1:]:
                self.dof_axis.append([int(i) for i in j.attrib['axis'].split(" ")])
            self.has_freejoint = True
        elif not "type" in tree.getroot().find("worldbody").findall('.//joint')[0].attrib:
            for j in tree.getroot().find("worldbody").findall('.//joint'):
                self.dof_axis.append([int(i) for i in j.attrib['axis'].split(" ")])
            self.has_freejoint = True
        else:
            for j in tree.getroot().find("worldbody").findall('.//joint')[6:]:
                self.dof_axis.append([int(i) for i in j.attrib['axis'].split(" ")])
            self.has_freejoint = False
        
        self.dof_axis = torch.tensor(self.dof_axis)

        for extend_config in cfg.extend_config:
            self.body_names_augment += [extend_config.joint_name]
            self._parents = torch.cat([self._parents, torch.tensor([self.body_names.index(extend_config.parent_name)]).to(device)], dim = 0)
            self._offsets = torch.cat([self._offsets, torch.tensor([[extend_config.pos]]).to(device)], dim = 1)
            self._local_rotation = torch.cat([self._local_rotation, torch.tensor([[extend_config.rot]]).to(device)], dim = 1)
            self.num_extend_dof += 1
            
        self.num_bodies = len(self.body_names)
        self.num_bodies_augment = len(self.body_names_augment)
        

        self.joints_range = mjcf_data['joints_range'].to(device)
        self._local_rotation_mat = tRot.quaternion_to_matrix(self._local_rotation).float() # w, x, y ,z
        self.load_mesh()
        
    def from_mjcf(self, path):
        # function from Poselib: 
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        xml_joint_root = xml_body_root.find("joint")
        
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        body_to_joint = OrderedDict()

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint") # joints need to remove the first 6 joints
            if len(all_joints) == 6:
                all_joints = all_joints[6:]
            
            for joint in all_joints:
                if not joint.attrib.get("range") is None: 
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
                else:
                    if not joint.attrib.get("type") == "free":
                        joints_range.append([-np.pi, np.pi])
            for joint_node in xml_node.findall("joint"):
                body_to_joint[node_name] = joint_node.attrib.get("name")
                
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)

            
            return node_index
        
        _add_xml_node(xml_body_root, -1, 0)
        assert len(joints_range) == self.num_dof 
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range)),
            "body_to_joint": body_to_joint
        }

        
    def fk_batch(self, pose, trans, convert_to_mat=True, return_full = False, dt=1/30):
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2]
        pose = pose[..., :len(self._parents), :] # H1 fitted joints might have extra joints
        
        if convert_to_mat:
            pose_quat = tRot.axis_angle_to_quaternion(pose.clone())
            pose_mat = tRot.quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
            
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1  # Exclude root
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        
        return_dict = EasyDict()
        
        
        wbody_rot = tRot.wxyz_to_xyzw(tRot.matrix_to_quaternion(wbody_mat))
        if len(self.cfg.extend_config) > 0:
            if return_full:
                return_dict.global_velocity_extend = self._compute_velocity(wbody_pos, dt) 
                return_dict.global_angular_velocity_extend = self._compute_angular_velocity(wbody_rot, dt)
                
            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_mat_extend = wbody_mat.clone()
            return_dict.global_rotation_extend = wbody_rot
            
            wbody_pos = wbody_pos[..., :self.num_bodies, :]
            wbody_mat = wbody_mat[..., :self.num_bodies, :, :]
            wbody_rot = wbody_rot[..., :self.num_bodies, :]

        
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot
        if return_full:
            rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)  # Isaac gym is [x, y, z, w]. All the previous functions are [w, x, y, z]
            rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
            return_dict.local_rotation = tRot.wxyz_to_xyzw(pose_quat)
            return_dict.global_root_velocity = rigidbody_linear_velocity[..., 0, :]
            return_dict.global_root_angular_velocity = rigidbody_angular_velocity[..., 0, :]
            return_dict.global_angular_velocity = rigidbody_angular_velocity
            return_dict.global_velocity = rigidbody_linear_velocity
            
            if len(self.cfg.extend_config) > 0:
                return_dict.dof_pos = pose.sum(dim = -1)[..., 1:self.num_bodies] # you can sum it up since unitree's each joint has 1 dof. Last two are for hands. doesn't really matter. 
            else:
                if not len(self.actuated_joints_idx) == len(self.body_names):
                    return_dict.dof_pos = pose.sum(dim = -1)[..., self.actuated_joints_idx]
                else:
                    return_dict.dof_pos = pose.sum(dim = -1)[..., 1:]
            
            dof_vel = ((return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1] )/dt)
            return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim = 1)
            return_dict.fps = int(1/dt)
        
        return return_dict
    
    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """
        
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))
        # print(expanded_offsets.shape, J)

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]])
                rot_mat = torch.matmul(rotations_world[self._parents[i]], torch.matmul(self._local_rotation_mat[:,  (i):(i + 1)], rotations[:, :, (i - 1):i, :]))
                # rot_mat = torch.matmul(rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :])
                # print(rotations[:, :, (i - 1):i, :].shape, self._local_rotation_mat.shape)
                
                positions_world.append(jpos)
                rotations_world.append(rot_mat)
        
        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world
    
    @staticmethod
    def _compute_velocity(p, time_delta, guassian_filter=True):
        # assume the second last dimension is the time axis
        p_prev = torch.cat([p[:, 0:1], p[:, :-1]], dim = 1)
        v = (p - p_prev) / time_delta
        if guassian_filter:
            v = filters.gaussian_filter1d(v.numpy(), 2, axis = 1)
            v = torch.from_numpy(v)
        return v
    
    @staticmethod
    def _compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # assume the second last dimension is the time axis

        r_prev = torch.cat([r[:, 0:1], r[:, :-1]], dim = 1)
        q_diff = tRot.quaternion_multiply(r, tRot.quaternion_invert(r_prev))
        r_vel = 2.0 / time_delta * q_diff[..., 1:]
        
        if guassian_filter:
            r_vel = filters.gaussian_filter1d(r_vel.numpy(), 2, axis = 1)
            r_vel = torch.from_numpy(r_vel)
        return r_vel
    
    def load_mesh(self):
        self.body_mesh = defaultdict(dict)
        mjcf_abs_path = os.path.join(hydra.utils.get_original_cwd(), self.mjcf_file)
        mjcf_dir = os.path.dirname(mjcf_abs_path)
        
        tree = ETree.parse(mjcf_abs_path)
        xml_doc_root = tree.getroot()
        
        # Find mesh directory from compiler tag
        compiler = xml_doc_root.find("compiler")
        mesh_dir = ""
        if compiler is not None and "meshdir" in compiler.attrib:
            mesh_dir = compiler.attrib["meshdir"]
        
        asset_dir = os.path.join(mjcf_dir, mesh_dir)

        xml_asset = xml_doc_root.find("asset")
        if xml_asset is None:
            return
        
        all_mesh = xml_asset.findall("mesh")
        
        def find_body(root, name):
            for body in root.iter("body"):
                if body.attrib.get("name") == name:
                    return body
            return None

        world_body = xml_doc_root.find("worldbody")
        for asset in all_mesh:
            body_name = "_".join(asset.attrib['name'].split("_")[1:])
            body = find_body(world_body, body_name)
            
            if not body is None and 'mesh' in body.find("geom").attrib: 
                geom = body.find("geom")
                
                scale = np.fromstring(asset.attrib.get("scale", "1 1 1"), dtype=float, sep=" ")

                if 'file' not in asset.attrib:
                    continue
                
                file_path = os.path.join(asset_dir, asset.attrib['file'])

                if not os.path.exists(file_path): 
                    logging.warning(f"Mesh file not found for {body_name} at {file_path}")
                    continue

                mesh_obj = mesh.Mesh.from_file(file_path)
                
                pos = np.fromstring(geom.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
                quat = np.fromstring(geom.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
                
                mat = sRot.from_quat(quat[[1,2,3,0]]).as_matrix()
                self.body_mesh[body_name] = {
                    "vertices": (mesh_obj.vectors @ mat.T * scale + pos),
                    "faces": np.arange(len(mesh_obj.vectors.reshape(-1, 3))).reshape(-1, 3)
                }

    def mesh_fk(self, pose = None, trans = None):
        B, T = pose.shape[:2]
        fk_res = self.fk_batch(pose, trans)
        global_trans = fk_res.global_translation_extend
        global_rot = fk_res.global_rotation_mat_extend
        
        
        all_vertices = []
        all_faces = []
        face_count = 0

        if not self.body_mesh:
            # logging.warning("body_mesh is empty in mesh_fk. Cannot compute mesh vertices. Returning FK translations.")
            # Fallback: if no mesh, return the joint translations directly. This might not be perfect for ground contact.
            return fk_res.global_translation_extend

        for b in range(B):
            for t in range(T):
                for idx, body_name in enumerate(self.body_names_augment):
                    if body_name in self.body_mesh:
                        vertices = self.body_mesh[body_name]['vertices']
                        faces = self.body_mesh[body_name]['faces']
                        
                        trans_vertices = vertices @ global_rot[b,t,idx].T.numpy() + global_trans[b,t,idx].numpy()
                        all_vertices.append(trans_vertices)
                        all_faces.append(faces + face_count)
                        face_count += len(vertices)

        if not all_vertices:
            raise ValueError("Could not generate any mesh vertices. Check if STL files are correctly referenced and loaded.")

        all_vertices = np.concatenate(all_vertices)
        all_faces = np.concatenate(all_faces)

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(all_vertices),
            o3d.utility.Vector3iVector(all_faces)
        )

        if not isinstance(mesh, o3d.geometry.TriangleMesh):
             # Fallback for when mesh calculation fails but we need a result.
             # This creates a dummy mesh object to prevent crashes, but results will be incorrect.
            logging.error("Final mesh object is not a TriangleMesh. Returning FK translations.")
            return fk_res.global_translation_extend

        return mesh