#!/usr/bin/env python3
"""
合并数据集脚本 - 用于合并通过retarget_motion.py生成的数据集

支持的数据格式：
1. 单个运动文件：motion_im.p
2. 多个运动文件：{dataset_name}/{dataset_name}.pkl

使用方法：
python scripts/merge_datasets.py dataset1_path dataset2_path output_path [--prefix1 name1] [--prefix2 name2]
"""

import os
import argparse
import joblib
import numpy as np
from collections import defaultdict


def load_dataset(dataset_path):
    """
    加载数据集文件
    
    Args:
        dataset_path: 数据集文件路径
        
    Returns:
        dict: 加载的数据集字典
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    print(f"📂 加载数据集: {dataset_path}")
    data = joblib.load(dataset_path)
    
    # 检查数据格式
    if isinstance(data, dict):
        # 检查是否是单个运动数据（包含root_trans_offset等直接字段）
        if "root_trans_offset" in data:
            # 单个运动数据，包装成字典格式
            filename = os.path.splitext(os.path.basename(dataset_path))[0]
            data = {filename: data}
            print(f"  ✓ 检测到单个运动文件，包装为: {filename}")
        else:
            print(f"  ✓ 检测到多运动数据集，包含 {len(data)} 个运动")
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")
    
    return data


def validate_motion_data(motion_key, motion_data):
    """
    验证单个运动数据的完整性
    
    Args:
        motion_key: 运动数据的键名
        motion_data: 运动数据字典
        
    Returns:
        bool: 验证是否通过
    """
    required_fields = [
        "root_trans_offset", "dof", "dof_increment", 
        "default_joint_angles", "root_rot", "smpl_joints", "fps"
    ]
    
    missing_fields = []
    for field in required_fields:
        if field not in motion_data:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"  ⚠️  运动 {motion_key} 缺少字段: {missing_fields}")
        return False
    
    # 检查数组长度一致性
    try:
        n_frames = len(motion_data["root_trans_offset"])
        arrays_to_check = ["dof", "dof_increment", "root_rot", "smpl_joints"]
        
        for array_name in arrays_to_check:
            if array_name in motion_data:
                if len(motion_data[array_name]) != n_frames:
                    print(f"  ⚠️  运动 {motion_key} 的 {array_name} 长度不匹配: {len(motion_data[array_name])} vs {n_frames}")
                    return False
        
        # 检查task_info
        if "task_info" in motion_data:
            task_info = motion_data["task_info"]
            if "target_vel" in task_info and len(task_info["target_vel"]) != n_frames:
                print(f"  ⚠️  运动 {motion_key} 的 task_info.target_vel 长度不匹配")
                return False
        
        print(f"  ✓ 运动 {motion_key}: {n_frames} 帧，数据完整")
        return True
        
    except Exception as e:
        print(f"  ❌ 运动 {motion_key} 验证失败: {e}")
        return False


def check_compatibility(dataset1, dataset2):
    """
    检查两个数据集的兼容性
    
    Args:
        dataset1: 第一个数据集
        dataset2: 第二个数据集
        
    Returns:
        bool: 是否兼容
    """
    print("🔍 检查数据集兼容性...")
    
    # 随机选择一个运动数据检查格式
    sample1 = next(iter(dataset1.values()))
    sample2 = next(iter(dataset2.values()))
    
    # 检查FPS是否一致
    fps1 = sample1.get("fps", 30)
    fps2 = sample2.get("fps", 30)
    if fps1 != fps2:
        print(f"  ⚠️  FPS不一致: 数据集1={fps1}, 数据集2={fps2}")
    
    # 检查关节数量是否一致
    dof1_shape = sample1["dof"].shape[-1] if "dof" in sample1 else 0
    dof2_shape = sample2["dof"].shape[-1] if "dof" in sample2 else 0
    if dof1_shape != dof2_shape:
        print(f"  ❌ 关节数量不匹配: 数据集1={dof1_shape}, 数据集2={dof2_shape}")
        return False
    
    # 检查默认关节角度是否一致
    default1 = sample1.get("default_joint_angles")
    default2 = sample2.get("default_joint_angles")
    if default1 is not None and default2 is not None:
        if not np.allclose(default1, default2, atol=1e-6):
            print(f"  ⚠️  默认关节角度不完全一致，可能来自不同机器人配置")
    
    print(f"  ✓ 数据集兼容，关节数量: {dof1_shape}")
    return True


def resolve_key_conflicts(dataset1, dataset2, prefix1="", prefix2=""):
    """
    解决键名冲突
    
    Args:
        dataset1: 第一个数据集
        dataset2: 第二个数据集
        prefix1: 第一个数据集的前缀
        prefix2: 第二个数据集的前缀
        
    Returns:
        tuple: 处理后的两个数据集
    """
    keys1 = set(dataset1.keys())
    keys2 = set(dataset2.keys())
    conflicts = keys1 & keys2
    
    if conflicts:
        print(f"🔀 检测到 {len(conflicts)} 个键名冲突: {list(conflicts)[:5]}...")
        
        # 如果没有指定前缀，自动生成
        if not prefix1:
            prefix1 = "dataset1_"
        if not prefix2:
            prefix2 = "dataset2_"
        
        print(f"  使用前缀解决冲突: '{prefix1}' 和 '{prefix2}'")
        
        # 重命名冲突的键
        new_dataset1 = {}
        new_dataset2 = {}
        
        for key, value in dataset1.items():
            if key in conflicts:
                new_key = prefix1 + key
                new_dataset1[new_key] = value
                print(f"    {key} → {new_key}")
            else:
                new_dataset1[key] = value
        
        for key, value in dataset2.items():
            if key in conflicts:
                new_key = prefix2 + key
                new_dataset2[new_key] = value
                print(f"    {key} → {new_key}")
            else:
                new_dataset2[key] = value
        
        return new_dataset1, new_dataset2
    else:
        print("✓ 无键名冲突")
        return dataset1, dataset2


def merge_datasets(dataset1, dataset2, prefix1="", prefix2=""):
    """
    合并两个数据集
    
    Args:
        dataset1: 第一个数据集
        dataset2: 第二个数据集
        prefix1: 第一个数据集的前缀
        prefix2: 第二个数据集的前缀
        
    Returns:
        dict: 合并后的数据集
    """
    print("🔀 开始合并数据集...")
    
    # 验证数据完整性
    print("📋 验证数据集1...")
    valid_count1 = 0
    for key, motion_data in dataset1.items():
        if validate_motion_data(key, motion_data):
            valid_count1 += 1
    
    print("📋 验证数据集2...")
    valid_count2 = 0
    for key, motion_data in dataset2.items():
        if validate_motion_data(key, motion_data):
            valid_count2 += 1
    
    print(f"数据集1: {valid_count1}/{len(dataset1)} 个有效运动")
    print(f"数据集2: {valid_count2}/{len(dataset2)} 个有效运动")
    
    # 检查兼容性
    if not check_compatibility(dataset1, dataset2):
        raise ValueError("数据集不兼容，无法合并")
    
    # 解决键名冲突
    dataset1, dataset2 = resolve_key_conflicts(dataset1, dataset2, prefix1, prefix2)
    
    # 合并数据
    merged_dataset = {}
    merged_dataset.update(dataset1)
    merged_dataset.update(dataset2)
    
    print(f"✅ 合并完成！总共 {len(merged_dataset)} 个运动")
    return merged_dataset


def save_merged_dataset(merged_dataset, output_path):
    """
    保存合并后的数据集
    
    Args:
        merged_dataset: 合并后的数据集
        output_path: 输出路径
    """
    print(f"💾 保存合并数据集到: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存数据
    joblib.dump(merged_dataset, output_path)
    
    # 计算统计信息
    total_frames = 0
    total_duration = 0
    fps_list = []
    
    for motion_data in merged_dataset.values():
        frames = len(motion_data["root_trans_offset"])
        fps = motion_data.get("fps", 30)
        total_frames += frames
        total_duration += frames / fps
        fps_list.append(fps)
    
    avg_fps = np.mean(fps_list) if fps_list else 30
    
    print(f"📊 数据集统计:")
    print(f"  运动数量: {len(merged_dataset)}")
    print(f"  总帧数: {total_frames}")
    print(f"  总时长: {total_duration:.1f} 秒")
    print(f"  平均FPS: {avg_fps:.1f}")
    print(f"✅ 保存完成!")


def main():
    parser = argparse.ArgumentParser(description="合并retarget motion生成的数据集")
    parser.add_argument("dataset1", help="第一个数据集路径")
    parser.add_argument("dataset2", help="第二个数据集路径") 
    parser.add_argument("output", help="输出文件路径")
    parser.add_argument("--prefix1", default="", help="第一个数据集的前缀（解决键名冲突）")
    parser.add_argument("--prefix2", default="", help="第二个数据集的前缀（解决键名冲突）")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的输出文件")
    
    args = parser.parse_args()
    
    # 检查输出文件是否已存在
    if os.path.exists(args.output) and not args.force:
        response = input(f"输出文件 {args.output} 已存在，是否覆盖？(y/N): ")
        if response.lower() != 'y':
            print("操作取消")
            return
    
    try:
        # 加载数据集
        dataset1 = load_dataset(args.dataset1)
        dataset2 = load_dataset(args.dataset2)
        
        # 合并数据集
        merged_dataset = merge_datasets(dataset1, dataset2, args.prefix1, args.prefix2)
        
        # 保存结果
        save_merged_dataset(merged_dataset, args.output)
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 