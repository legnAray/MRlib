#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查运动数据文件的完整性和结构
"""

import joblib
import numpy as np
import sys
import os

def check_motion_data(motion_file):
    """检查运动数据文件"""
    print(f"检查运动文件: {motion_file}")
    
    if not os.path.exists(motion_file):
        print(f"❌ 文件不存在: {motion_file}")
        return False
    
    try:
        # 加载数据
        motion_data = joblib.load(motion_file)
        print(f"✅ 文件加载成功")
        print(f"数据类型: {type(motion_data)}")
        
        # 处理不同的数据格式
        if isinstance(motion_data, dict):
            if len(motion_data) == 0:
                print("❌ 字典为空")
                return False
            
            print(f"包含 {len(motion_data)} 个运动:")
            for key in list(motion_data.keys())[:5]:  # 只显示前5个
                print(f"  - {key}")
            if len(motion_data) > 5:
                print(f"  ... 还有 {len(motion_data) - 5} 个运动")
            
            # 检查第一个运动
            first_key = list(motion_data.keys())[0]
            single_motion = motion_data[first_key]
            print(f"\n检查运动: {first_key}")
        else:
            single_motion = motion_data
            print("单个运动数据")
        
        # 检查单个运动的结构
        print(f"运动数据字段: {list(single_motion.keys())}")
        
        # 检查关键字段
        required_fields = ['dof', 'root_trans_offset', 'root_rot']
        for field in required_fields:
            if field in single_motion:
                shape = single_motion[field].shape
                print(f"✅ {field}: {shape}")
            else:
                print(f"❌ 缺少字段: {field}")
        
        # 检查增量角度相关字段
        if 'dof_increment' in single_motion:
            print(f"✅ dof_increment: {single_motion['dof_increment'].shape}")
        else:
            print("⚠️  没有增量角度数据 (dof_increment)")
            
        if 'default_joint_angles' in single_motion:
            shape = single_motion['default_joint_angles'].shape
            print(f"✅ default_joint_angles: {shape}")
            print(f"默认角度值: {single_motion['default_joint_angles']}")
        else:
            print("⚠️  没有默认角度数据 (default_joint_angles)")
        
        # 检查数据一致性
        dof_shape = single_motion['dof'].shape
        print(f"\n数据一致性检查:")
        print(f"dof 形状: {dof_shape}")
        
        # 检查所有时间相关字段的长度是否一致
        time_fields = ['dof', 'root_trans_offset', 'root_rot']
        if 'dof_increment' in single_motion:
            time_fields.append('dof_increment')
            
        time_lengths = []
        for field in time_fields:
            if field in single_motion:
                length = single_motion[field].shape[0]
                time_lengths.append(length)
                print(f"  {field}: {length} 帧")
        
        if len(set(time_lengths)) == 1:
            print("✅ 所有时间序列长度一致")
        else:
            print("❌ 时间序列长度不一致!")
            
        # 检查增量角度计算是否正确
        if 'dof_increment' in single_motion and 'default_joint_angles' in single_motion:
            dof = single_motion['dof']
            dof_increment = single_motion['dof_increment']
            default_angles = single_motion['default_joint_angles']
            
            # 验证计算
            calculated_increment = dof - default_angles[None, :]
            if np.allclose(dof_increment, calculated_increment, atol=1e-6):
                print("✅ 增量角度计算正确")
            else:
                print("❌ 增量角度计算错误!")
                print(f"最大差异: {np.abs(dof_increment - calculated_increment).max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) > 1:
        motion_file = sys.argv[1]
    else:
        motion_file = "output/g1_29dof/run/run.pkl"
    
    print("=== 运动数据完整性检查 ===\n")
    success = check_motion_data(motion_file)
    
    if success:
        print("\n✅ 检查完成，数据结构正常")
    else:
        print("\n❌ 检查失败，请修复数据问题")

if __name__ == "__main__":
    main() 