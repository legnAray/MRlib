#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增量关节角度功能
验证保存的数据是否包含正确的增量角度信息
"""

import os
import joblib
import numpy as np
import glob

def test_increment_angles(data_path):
    """测试增量角度数据"""
    print(f"正在测试数据文件: {data_path}")
    
    # 加载数据
    if data_path.endswith('.pkl'):
        # 多运动文件格式
        all_data = joblib.load(data_path)
        # 取第一个运动进行测试
        first_key = list(all_data.keys())[0]
        motion_data = all_data[first_key]
        print(f"测试运动: {first_key}")
    else:
        # 单运动文件格式
        motion_data = joblib.load(data_path)
        print("测试单个运动文件")
    
    # 检查数据结构
    print(f"数据字段: {list(motion_data.keys())}")
    
    # 检查增量角度相关字段
    if 'dof_increment' in motion_data:
        print("✅ 包含增量角度数据 (dof_increment)")
        dof = motion_data['dof']
        dof_increment = motion_data['dof_increment']
        default_angles = motion_data['default_joint_angles']
        
        print(f"关节角度形状: {dof.shape}")
        print(f"增量角度形状: {dof_increment.shape}")
        print(f"默认角度长度: {len(default_angles)}")
        
        # 验证增量计算是否正确
        calculated_increment = dof - default_angles[None, :]
        is_correct = np.allclose(dof_increment, calculated_increment, atol=1e-6)
        print(f"增量计算验证: {'✅ 正确' if is_correct else '❌ 错误'}")
        
        # 显示默认角度
        print(f"默认关节角度: {default_angles}")
        
        # 显示一些统计信息
        print(f"原始角度范围: [{dof.min():.3f}, {dof.max():.3f}]")
        print(f"增量角度范围: [{dof_increment.min():.3f}, {dof_increment.max():.3f}]")
        print(f"增量角度均值: {dof_increment.mean():.3f}")
        print(f"增量角度标准差: {dof_increment.std():.3f}")
        
    else:
        print("❌ 未找到增量角度数据 (dof_increment)")
    
    if 'default_joint_angles' in motion_data:
        print("✅ 包含默认角度参考 (default_joint_angles)")
    else:
        print("❌ 未找到默认角度参考 (default_joint_angles)")

def main():
    """主函数"""
    print("=== 增量关节角度功能测试 ===\n")
    
    # 查找输出目录中的数据文件
    output_base = "output"
    
    # 查找所有可能的数据文件
    data_files = []
    data_files.extend(glob.glob(f"{output_base}/**/motion_im.p", recursive=True))
    data_files.extend(glob.glob(f"{output_base}/**/*.pkl", recursive=True))
    
    if not data_files:
        print("未找到任何数据文件，请先运行运动重定向")
        print("查找路径: output/**/motion_im.p 或 output/**/*.pkl")
        return
    
    print(f"找到 {len(data_files)} 个数据文件：")
    for i, file in enumerate(data_files, 1):
        print(f"{i}. {file}")
    
    print("\n开始测试...")
    for file in data_files:
        print("\n" + "="*50)
        try:
            test_increment_angles(file)
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n" + "="*50)
    print("测试完成！")

if __name__ == "__main__":
    main() 