import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.spatial.distance import euclidean

dir = "/media/ray/Data/Retargeted_AMASS_for_robotics/g1"

class MotionClassifier:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_motion_data(self, file_path):
        """加载运动数据"""
        try:
            data = np.load(file_path)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def analyze_root_motion(self, data):
        """分析根节点运动特征"""
        if data is None or len(data) == 0:
            return None
            
        # 提取根节点位置 (前3列)
        root_positions = data[:, :3]
        
        # 计算运动轨迹
        trajectory = np.diff(root_positions, axis=0)
        speeds = np.linalg.norm(trajectory, axis=1)
        
        # 只分析直线运动的关键特征
        features = {
            'avg_speed': np.mean(speeds),
            'total_distance': np.sum(speeds),
            'trajectory_linearity': self.calculate_linearity(root_positions),
            'forward_motion_ratio': self.calculate_forward_motion_ratio(trajectory),
        }
        
        return features
    
    def calculate_linearity(self, positions):
        """计算轨迹的直线性"""
        if len(positions) < 3:
            return 0
            
        # 计算起点到终点的直线距离
        straight_distance = euclidean(positions[0], positions[-1])
        
        # 计算实际轨迹长度
        actual_distance = 0
        for i in range(1, len(positions)):
            actual_distance += euclidean(positions[i-1], positions[i])
        
        if actual_distance == 0:
            return 0
            
        # 直线性 = 直线距离 / 实际轨迹长度
        linearity = straight_distance / actual_distance
        return linearity
    
    def calculate_forward_motion_ratio(self, trajectory):
        """计算前进运动的比例"""
        if len(trajectory) == 0:
            return 0
        
        # 计算总体位移向量（起点到终点的方向）
        total_displacement = np.sum(trajectory, axis=0)
        
        # 如果总位移几乎为零，说明没有明显的前进方向
        total_displacement_magnitude = np.linalg.norm(total_displacement)
        if total_displacement_magnitude < 1e-6:
            return 0
        
        # 归一化总位移向量，得到主要前进方向
        main_direction = total_displacement / total_displacement_magnitude
        
        # 计算每一步的运动在主要方向上的投影
        forward_projections = np.dot(trajectory, main_direction)
        
        # 计算前进运动的比例（正向投影的比例）
        forward_steps = np.sum(forward_projections > 0)
        forward_ratio = forward_steps / len(trajectory)
        
        return forward_ratio
    
    def is_linear_motion(self, features):
        """判断是否为直线运动"""
        if features is None:
            return False
        
        linearity = features['trajectory_linearity']
        forward_ratio = features['forward_motion_ratio']
        avg_speed = features['avg_speed']
        total_distance = features['total_distance']
        
        # 更严格的直线运动判断条件
        is_linear = (
            linearity > 0.8 and          # 轨迹必须非常直线 (从0.5提高到0.8)
            forward_ratio > 0.8 and      # 前进运动比例必须很高 (从0.6提高到0.8)
            avg_speed > 0.005 and        # 速度必须更明显 (从0.001提高到0.005)
            total_distance > 0.5         # 总距离必须更长 (从0.1提高到0.5)
        )
        
        return is_linear
    
    def scan_dataset(self, output_file=None, visualize=False):
        """扫描整个数据集并筛选直线运动"""
        linear_motion_files = []
        other_files = []
        
        print(f"正在扫描数据集: {self.data_dir}")
        
        # 遍历所有.npy文件
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('_poses_120_jpos.npy') or file.endswith('_poses_60_jpos.npy'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.data_dir)
                    
                    print(f"分析: {relative_path}")
                    
                    # 加载和分析数据
                    data = self.load_motion_data(file_path)
                    features = self.analyze_root_motion(data)
                    is_linear = self.is_linear_motion(features)
                    
                    if is_linear:
                        linear_motion_files.append((relative_path, features))
                        print(f"  -> 直线运动 ✓")
                    else:
                        other_files.append((relative_path, features))
                        print(f"  -> 其他运动")
        
        # 输出结果
        print(f"\n筛选结果:")
        print(f"直线运动: {len(linear_motion_files)} 个")
        print(f"其他运动: {len(other_files)} 个")
        
        # 保存结果到文件
        if output_file:
            self.save_classification_results(linear_motion_files, other_files, output_file)
        
        # 可视化分析
        if visualize:
            self.visualize_classification_results(linear_motion_files, other_files)
        
        return linear_motion_files, other_files
    
    def save_classification_results(self, linear_motion_files, other_files, output_file):
        """保存分类结果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== 人形机器人直线运动筛选结果 ===\n\n")
            
            f.write(f"总计:\n")
            f.write(f"  直线运动: {len(linear_motion_files)} 个\n")
            f.write(f"  其他运动: {len(other_files)} 个\n\n")
            
            f.write("=== 直线运动文件 ===\n")
            for file_path, features in linear_motion_files:
                f.write(f"{file_path}\n")
                if features:
                    f.write(f"  平均速度: {features['avg_speed']:.4f}\n")
                    f.write(f"  轨迹直线性: {features['trajectory_linearity']:.4f}\n")
                    f.write(f"  前进比例: {features['forward_motion_ratio']:.4f}\n")
                    f.write(f"  总距离: {features['total_distance']:.4f}\n")
                f.write("\n")
        
        print(f"筛选结果已保存到: {output_file}")
    
    def visualize_classification_results(self, linear_motion_files, other_files):
        """可视化分类结果"""
        # 提取特征用于可视化
        def extract_features(file_list):
            speeds = []
            linearities = []
            forward_ratios = []
            for _, features in file_list:
                if features:
                    speeds.append(features['avg_speed'])
                    linearities.append(features['trajectory_linearity'])
                    forward_ratios.append(features['forward_motion_ratio'])
            return speeds, linearities, forward_ratios
        
        linear_speeds, linear_linearities, linear_forward_ratios = extract_features(linear_motion_files)
        other_speeds, other_linearities, other_forward_ratios = extract_features(other_files)
        
        # 创建可视化图表
        plt.figure(figsize=(15, 10))
        
        # 速度 vs 直线性
        plt.subplot(2, 3, 1)
        plt.scatter(linear_speeds, linear_linearities, alpha=0.6, label='Linear Motion', color='green', s=50)
        plt.scatter(other_speeds, other_linearities, alpha=0.3, label='Other Motion', color='gray', s=30)
        plt.xlabel('Average Speed')
        plt.ylabel('Trajectory Linearity')
        plt.title('Speed vs Linearity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 前进比例 vs 直线性
        plt.subplot(2, 3, 2)
        plt.scatter(linear_forward_ratios, linear_linearities, alpha=0.6, label='Linear Motion', color='green', s=50)
        plt.scatter(other_forward_ratios, other_linearities, alpha=0.3, label='Other Motion', color='gray', s=30)
        plt.xlabel('Forward Motion Ratio')
        plt.ylabel('Trajectory Linearity')
        plt.title('Forward Ratio vs Linearity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 速度分布
        plt.subplot(2, 3, 3)
        plt.hist(linear_speeds, alpha=0.7, label='Linear Motion', bins=20, color='green')
        plt.hist(other_speeds, alpha=0.5, label='Other Motion', bins=20, color='gray')
        plt.xlabel('Average Speed')
        plt.ylabel('Count')
        plt.title('Speed Distribution')
        plt.legend()
        
        # 直线性分布
        plt.subplot(2, 3, 4)
        plt.hist(linear_linearities, alpha=0.7, label='Linear Motion', bins=20, color='green')
        plt.hist(other_linearities, alpha=0.5, label='Other Motion', bins=20, color='gray')
        plt.xlabel('Trajectory Linearity')
        plt.ylabel('Count')
        plt.title('Linearity Distribution')
        plt.legend()
        
        # 前进比例分布
        plt.subplot(2, 3, 5)
        plt.hist(linear_forward_ratios, alpha=0.7, label='Linear Motion', bins=20, color='green')
        plt.hist(other_forward_ratios, alpha=0.5, label='Other Motion', bins=20, color='gray')
        plt.xlabel('Forward Motion Ratio')
        plt.ylabel('Count')
        plt.title('Forward Ratio Distribution')
        plt.legend()
        
        # 统计饼图
        plt.subplot(2, 3, 6)
        labels = ['Linear Motion', 'Other Motion']
        sizes = [len(linear_motion_files), len(other_files)]
        colors = ['green', 'gray']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('Motion Type Distribution')
        
        plt.tight_layout()
        plt.savefig('linear_motion_classification.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='人形机器人直线运动筛选工具')
    parser.add_argument('--data_dir', type=str, default=dir, 
                       help='数据集根目录')
    parser.add_argument('--output', type=str, default='linear_motion_results.txt',
                       help='输出结果文件名')
    parser.add_argument('--visualize', action='store_true',
                       help='是否显示可视化结果')
    parser.add_argument('--linearity_threshold', type=float, default=0.5,
                       help='轨迹直线性阈值 (0-1)')
    parser.add_argument('--forward_threshold', type=float, default=0.6,
                       help='前进运动比例阈值 (0-1)')
    
    args = parser.parse_args()
    
    # 创建分类器
    classifier = MotionClassifier(args.data_dir)
    
    # 可以调整阈值
    if hasattr(args, 'linearity_threshold') or hasattr(args, 'forward_threshold'):
        print(f"使用自定义阈值: 直线性>{args.linearity_threshold}, 前进比例>{args.forward_threshold}")
    
    # 执行筛选
    linear_motion_files, other_files = classifier.scan_dataset(
        output_file=args.output,
        visualize=args.visualize
    )
    
    print(f"\n筛选完成! 找到:")
    print(f"  直线运动: {len(linear_motion_files)} 个")
    print(f"  其他运动: {len(other_files)} 个")
    
    # 显示前几个结果作为示例
    print(f"\n直线运动示例:")
    for i, (file_path, features) in enumerate(linear_motion_files[:10]):
        print(f"  {i+1}. {file_path}")
        if features:
            print(f"      直线性: {features['trajectory_linearity']:.3f}, "
                  f"前进比例: {features['forward_motion_ratio']:.3f}, "
                  f"平均速度: {features['avg_speed']:.4f}")

if __name__ == "__main__":
    main()