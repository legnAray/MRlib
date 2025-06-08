# MRlib - Motion Retarget Library
MRlib是一个用于将AMASS人体运动数据集重定向到机器人关节的Python库。

## 功能特性

- ✅ 支持AMASS数据集的运动重定向
- ✅ 支持宇树G1机器人（29自由度）
- ✅ 内置可视化工具
- ✅ 基于Hydra的配置管理系统

## 系统要求

- Python 3.8+
- CUDA支持的GPU（推荐）
- Linux操作系统（推荐）

## 安装步骤

### 1. 下载AMASS数据集

访问 [AMASS官方下载页面](https://amass.is.tue.mpg.de/download.php) 注册并下载所需的数据集。

### 2. 克隆仓库

```bash
git clone <仓库地址>
cd MRlib
```

### 3. 创建Conda环境

```bash
conda create -n mrlib python=3.8
conda activate mrlib
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 准备SMPL模型

确保您有SMPL模型文件，并将其放置在项目目录下的`smpl`文件夹中。

## 使用方法

### 第一步：形状重定向

运行以下命令进行形状匹配：

```bash
python retarget_shape.py \
    robot=unitree_g1_29dof \
    +smpl_model_path=/path/to/your/smpl \
    +amass_path=/path/to/your/amass/data \
    output_path=output \
    +fit_all=True
```

参数说明：
- `robot`: 目标机器人类型（目前支持`unitree_g1_29dof`）
- `smpl_model_path`: SMPL模型文件路径
- `amass_path`: AMASS数据集路径
- `output_path`: 输出文件夹路径
- `fit_all`: 是否处理所有数据

### 第二步：运动重定向

运行以下命令进行运动重定向：

```bash
python retarget_motion.py \
    robot=unitree_g1_29dof \
    +smpl_model_path=/path/to/your/smpl \
    +amass_path=/path/to/your/amass/data \
    output_path=output \
    +fit_all=True
```

### 第三步：可视化结果

使用以下命令可视化重定向后的运动：

```bash
python play_motion.py +motion_file=output/g1_29dof/test/all_data.pkl
```

## 项目结构

```
MRlib/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖列表
├── retarget_shape.py         # 形状重定向脚本
├── retarget_motion.py        # 运动重定向脚本
├── play_motion.py            # 运动可视化脚本
├── test_installation.py     # 安装测试脚本
├── cfg/                      # 配置文件目录
│   ├── config.yaml          # 主配置文件
│   └── robot/               # 机器人配置文件
├── robot/                    # 机器人模型定义
│   └── unitree_g1_29dof/    # 宇树G1机器人配置
├── mrlib/                    # 核心库
│   ├── utils/               # 工具函数
│   ├── smpllib/             # SMPL相关功能
│   └── poselib/             # 姿态处理功能
└── smpl/                     # SMPL模型文件（需要用户提供）
```

## 支持的机器人

目前支持的机器人平台：
- **宇树G1人形机器人（29自由度）**: `unitree_g1_29dof`

## 依赖说明

详细依赖列表请参考`requirements.txt`文件。

## 许可证

本项目使用BSD 3-Clause许可证。请参考项目根目录下的LICENSE文件。

### 代码来源声明

本项目的部分代码来源于以下开源项目：

- **PHC (Perpetual Humanoid Control)**: [https://github.com/ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC)
  - 作者: Zhengyi Luo, Jinkun Cao, Alexander W. Winkler, Kris Kitani, Weipeng Xu
  - 许可证: BSD 3-Clause License
  - 引用: 主要使用了其中的运动重定向相关代码，并进行了修改以适配本项目需求

如果您在学术研究中使用本项目，请考虑引用原始PHC论文：

```bibtex
@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}            
```

## 贡献

欢迎提交Issues和Pull Requests来改进这个项目！ 