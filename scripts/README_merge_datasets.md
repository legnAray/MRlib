# 数据集合并工具使用说明

这个工具用于合并通过`retarget_motion.py`生成的运动数据集。

## 📁 支持的数据格式

1. **单个运动文件**: `motion_im.p` (包含单个运动的数据)
2. **多运动数据集**: `{dataset_name}.pkl` (包含多个运动的字典)

## 🚀 使用方法

### 基本命令

```bash
python scripts/merge_datasets.py dataset1_path dataset2_path output_path
```

### 完整参数

```bash
python scripts/merge_datasets.py dataset1_path dataset2_path output_path \
    --prefix1 "walk_" \
    --prefix2 "run_" \
    --force
```

### 参数说明

- `dataset1_path`: 第一个数据集文件路径
- `dataset2_path`: 第二个数据集文件路径
- `output_path`: 输出合并数据集的路径
- `--prefix1`: 第一个数据集的前缀（解决键名冲突）
- `--prefix2`: 第二个数据集的前缀（解决键名冲突）
- `--force`: 强制覆盖已存在的输出文件

## 📝 使用示例

### 示例1: 合并两个多运动数据集

```bash
python scripts/merge_datasets.py \
    data/unitree_g1_29dof/walking/walking.pkl \
    data/unitree_g1_29dof/running/running.pkl \
    data/unitree_g1_29dof/merged/walking_running.pkl \
    --prefix1 "walk_" \
    --prefix2 "run_"
```

### 示例2: 合并单个运动文件

```bash
python scripts/merge_datasets.py \
    data/unitree_g1_29dof/motion_im.p \
    data/unitree_g1_29dof/motion_im2.p \
    data/unitree_g1_29dof/merged_motions.pkl
```

### 示例3: 使用示例脚本

```bash
python scripts/merge_example.py
```

## 🔧 功能特性

### ✅ 数据验证
- 自动检查数据完整性
- 验证必需字段是否存在
- 检查数组长度一致性
- 验证task_info数据格式

### 🔄 兼容性检查
- 检查FPS是否一致
- 验证关节数量匹配
- 比较默认关节角度
- 确保数据格式兼容

### 🏷️ 键名冲突处理
- 自动检测键名冲突
- 支持自定义前缀解决冲突
- 智能重命名策略

### 📊 统计信息
- 显示运动数量
- 计算总帧数和时长
- 显示平均FPS
- 验证结果报告

## 📋 数据结构

### 输入数据格式

```python
# 单个运动数据
{
    "root_trans_offset": np.array,    # 根部位置偏移
    "dof": np.array,                  # 关节角度
    "dof_increment": np.array,        # 增量关节角度
    "default_joint_angles": np.array, # 默认关节角度
    "root_rot": np.array,             # 根部旋转
    "smpl_joints": np.array,          # SMPL关节位置
    "fps": int,                       # 帧率
    "task_info": {                    # 任务信息
        "target_vel": np.array,       # 目标速度
        "motion_type": np.array,      # 运动类型
        "base_motion_flag": int       # 基础运动标志
    }
}

# 多运动数据集
{
    "motion_key1": {单个运动数据},
    "motion_key2": {单个运动数据},
    ...
}
```

### 输出数据格式

合并后的数据集始终为多运动格式，即使输入是单个运动文件也会被包装成字典格式。

## ⚠️ 注意事项

1. **数据兼容性**: 只能合并关节数量相同的数据集
2. **键名冲突**: 建议使用前缀参数避免运动名称冲突
3. **内存使用**: 大数据集合并可能需要较多内存
4. **备份数据**: 建议在合并前备份原始数据

## 🐛 常见问题

### Q: 提示"关节数量不匹配"
A: 确保两个数据集使用相同的机器人配置(如都是unitree_g1_29dof)

### Q: 提示"数据集文件不存在"
A: 检查文件路径是否正确，确保已通过retarget_motion.py生成数据

### Q: 键名冲突如何处理？
A: 使用`--prefix1`和`--prefix2`参数为冲突的键添加前缀

### Q: 如何查看合并结果？
A: 可以使用joblib.load()加载合并后的文件，查看包含的运动列表

## 📖 相关工具

- `retarget_motion.py`: 生成运动数据集
- `merge_example.py`: 合并使用示例
- 其他分析和可视化工具

## 🔗 更多信息

运行`python scripts/merge_datasets.py --help`查看完整参数说明。 