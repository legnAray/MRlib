# MRlib - Motion Retarget Library
MRlib是一个用于将AMASS人体运动数据集重定向到机器人关节的Python仓库。

## 功能特性

- ✅ 支持AMASS数据集的运动重定向
- ✅ 支持宇树G1机器人（29自由度）
- ✅ 支持可视化重定向动作
- ✅ 基于Hydra的配置管理系统

## 系统要求

- Python 3.8+
- Linux操作系统（推荐）

## 安装步骤

### 1. 克隆仓库

```bash
git clone <仓库地址>
cd MRlib
```

### 2. 下载AMASS数据集

访问 [AMASS官方下载页面](https://amass.is.tue.mpg.de/download.php) 注册并下载所需的数据集。

### 3. 下载SMPL模型

访问https://smpl.is.tue.mpg.de/download.php

将其放置在项目目录下的`smpl`文件夹中。

### 4. 创建Conda环境

```bash
conda create -n mrlib python=3.8
conda activate mrlib
```

### 5. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 第一步：SMPL模型参数辨识


```bash
python scripts/retarget_shape.py \
    robot=unitree_g1_29dof \
    +smpl_model_path=/path/to/your/smpl\
    output_path=output \
    +vis=True
```

### 第二步：运动重定向

```bash
python scripts/retarget_motion.py \
    robot=unitree_g1_29dof \
    +smpl_model_path=/path/to/your/smpl \
    +amass_path=/path/to/your/amass/data \
    output_path=output
```

### 第三步：可视化结果

```bash
python scripts/play_motion.py +motion_file=output/g1_29dof/test/all_data.pkl
```

## 支持的机器人

目前支持的机器人平台：
- **宇树G1人形机器人（29自由度）**: `unitree_g1_29dof`

## 许可证

本项目使用BSD 3-Clause许可证。请参考项目根目录下的LICENSE文件。

### 代码来源声明

本项目的部分代码来源于以下开源项目：

- **PHC (Perpetual Humanoid Control)**: [https://github.com/ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC)
  - 作者: Zhengyi Luo, Jinkun Cao, Alexander W. Winkler, Kris Kitani, Weipeng Xu
  - 许可证: BSD 3-Clause License
  - 引用: 主要使用了其中的运动重定向相关代码，并进行了修改以适配本项目需求

如果您在学术研究中使用本项目，请引用原始论文：

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