#!/usr/bin/env python3

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import stats

def load_dataset(pkl_file):
    """Load pkl dataset file"""
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"File not found: {pkl_file}")
    
    print(f"Loading dataset: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Unify to multi-motion format
    if isinstance(data, dict):
        if "root_trans_offset" in data:
            # Old format single motion data
            filename = os.path.splitext(os.path.basename(pkl_file))[0]
            data = {filename: data}
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    
    return data

def collect_all_base_lin_vel_local_50window(motion_data):
    """Collect base_lin_vel_local_50window data from all motion sequences"""
    all_velocities = []
    motion_labels = []
    
    print(f"\nCollecting data from {len(motion_data)} motion sequences:")
    
    for motion_key, motion in motion_data.items():
        if "base_lin_vel_local_50window" not in motion or motion["base_lin_vel_local_50window"] is None:
            print(f"  Skipping '{motion_key}': no base_lin_vel_local_50window data")
            continue
        
        base_lin_vel_local_50window = motion["base_lin_vel_local_50window"]
        n_frames = len(base_lin_vel_local_50window)
        
        print(f"  Loading '{motion_key}': {n_frames} frames")
        
        # Collect data
        all_velocities.append(base_lin_vel_local_50window)
        motion_labels.extend([motion_key] * n_frames)
    
    if not all_velocities:
        raise ValueError("No motion sequences with base_lin_vel_local_50window data found")
    
    # Merge all data
    all_velocities = np.vstack(all_velocities)
    motion_labels = np.array(motion_labels)
    
    print(f"\nTotal collected {len(all_velocities)} frames of data")
    return all_velocities, motion_labels

def plot_velocity_distribution(velocities, motion_labels, save_dir=None):
    """Plot base_lin_vel_local_50window distribution"""
    
    # Set font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create main figure with a new 3x3 layout
    fig = plt.figure(figsize=(21, 15))
    
    # Color mapping
    unique_motions = np.unique(motion_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_motions)))
    motion_color_map = dict(zip(unique_motions, colors))
    
    # Process X, Y, Z axes separately
    axis_names = ['X-axis (Forward/Backward)', 'Y-axis (Left/Right)', 'Z-axis (Up/Down)']
    
    for axis_idx in range(3):
        axis_data = velocities[:, axis_idx]
        
        # 1. Histogram distribution (top row)
        # New subplot position: 3x3 grid, first row
        ax1 = plt.subplot(3, 3, axis_idx + 1)
        
        # Plot overall distribution of all motions
        n, bins, patches = plt.hist(axis_data, bins=50, alpha=0.7, color='skyblue', 
                                   density=True, label='Overall Distribution')
        
        # Plot each motion's distribution (different colors)
        for motion in unique_motions:
            mask = motion_labels == motion
            motion_data = axis_data[mask]
            if len(motion_data) > 0:
                plt.hist(motion_data, bins=30, alpha=0.3, 
                        color=motion_color_map[motion], density=True, 
                        label=f'{motion}')
        
        plt.title(f'{axis_names[axis_idx]} - Velocity Distribution Histogram')
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        if axis_idx == 0:  # Show legend only in the first subplot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Cumulative distribution function (middle row)
        # New subplot position: 3x3 grid, second row
        ax2 = plt.subplot(3, 3, axis_idx + 4)
        
        # Overall CDF
        sorted_data = np.sort(axis_data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, y, linewidth=2, color='blue', label='Overall CDF')
        
        # CDF for each motion
        for motion in unique_motions:
            mask = motion_labels == motion
            motion_data = axis_data[mask]
            if len(motion_data) > 0:
                sorted_motion = np.sort(motion_data)
                y_motion = np.arange(1, len(sorted_motion) + 1) / len(sorted_motion)
                plt.plot(sorted_motion, y_motion, alpha=0.7, 
                        color=motion_color_map[motion], label=f'{motion}')
        
        plt.title(f'{axis_names[axis_idx]} - Cumulative Distribution Function')
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        
        # 3. Box plot (bottom row) - REMOVED
        # The code block for box plots that was here has been deleted.

    # 4. Speed magnitude distribution (bottom row, first plot)
    # New subplot position: 3x3 grid, position 7
    ax4 = plt.subplot(3, 3, 7)
    speed_magnitude = np.linalg.norm(velocities, axis=1)
    
    # Overall speed magnitude distribution
    plt.hist(speed_magnitude, bins=50, alpha=0.7, color='orange', density=True, label='Overall')
    
    # Speed magnitude distribution for each motion
    for motion in unique_motions:
        mask = motion_labels == motion
        motion_speed = speed_magnitude[mask]
        if len(motion_speed) > 0:
            plt.hist(motion_speed, bins=30, alpha=0.3, 
                    color=motion_color_map[motion], density=True, label=motion)
    
    plt.title('Speed Magnitude Distribution')
    plt.xlabel('Speed Magnitude (m/s)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # 5. 2D velocity distribution (bottom row, second plot)
    # New subplot position: 3x3 grid, position 8
    ax5 = plt.subplot(3, 3, 8)
    
    # Plot X-Y plane velocity distribution
    plt.scatter(velocities[:, 0], velocities[:, 1], alpha=0.1, s=1, c='blue')
    
    # Plot distribution centers for each motion
    for motion in unique_motions:
        mask = motion_labels == motion
        if np.sum(mask) > 0:
            motion_vel = velocities[mask]
            mean_x = np.mean(motion_vel[:, 0])
            mean_y = np.mean(motion_vel[:, 1])
            plt.scatter(mean_x, mean_y, color=motion_color_map[motion], 
                       s=100, alpha=0.8, marker='o', edgecolor='black', linewidth=1)
            plt.annotate(motion, (mean_x, mean_y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    plt.title('X-Y Velocity Distribution Scatter Plot')
    plt.xlabel('X-axis Velocity (m/s)')
    plt.ylabel('Y-axis Velocity (m/s)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # 6. Statistics information table (bottom row, third plot)
    # New subplot position: 3x3 grid, position 9
    ax6 = plt.subplot(3, 3, 9)
    ax6.axis('off')
    
    # Calculate statistics
    stats_text = "Statistics Summary:\n\n"
    
    for axis_idx, axis_name in enumerate(['X-axis', 'Y-axis', 'Z-axis']):
        axis_data = velocities[:, axis_idx]
        stats_text += f"{axis_name}:\n"
        stats_text += f"  Mean: {np.mean(axis_data):+.3f} m/s\n"
        stats_text += f"  Std: {np.std(axis_data):.3f} m/s\n"
        stats_text += f"  Range: [{np.min(axis_data):.3f}, {np.max(axis_data):.3f}] m/s\n"
        stats_text += f"  95% Percentile: [{np.percentile(axis_data, 2.5):.3f}, {np.percentile(axis_data, 97.5):.3f}] m/s\n\n"
    
    # Speed magnitude statistics
    speed_magnitude = np.linalg.norm(velocities, axis=1)
    stats_text += f"Speed Magnitude:\n"
    stats_text += f"  Mean: {np.mean(speed_magnitude):.3f} m/s\n"
    stats_text += f"  Std: {np.std(speed_magnitude):.3f} m/s\n"
    stats_text += f"  Max: {np.max(speed_magnitude):.3f} m/s\n"
    
    plt.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'base_lin_vel_local_50window_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()

def print_detailed_statistics(velocities, motion_labels):
    """Print detailed statistical information"""
    print("\n" + "="*80)
    print("BASE_LIN_VEL_LOCAL_50WINDOW DETAILED STATISTICAL ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print(f"\n[Overall Statistics]")
    print(f"Total data points: {len(velocities)}")
    print(f"Number of motions: {len(np.unique(motion_labels))}")
    
    axis_names = ['X-axis (Forward/Backward)', 'Y-axis (Left/Right)', 'Z-axis (Up/Down)']
    
    for axis_idx, axis_name in enumerate(axis_names):
        axis_data = velocities[:, axis_idx]
        print(f"\n[{axis_name}]")
        print(f"  Mean: {np.mean(axis_data):+.6f} m/s")
        print(f"  Median: {np.median(axis_data):+.6f} m/s")
        print(f"  Std: {np.std(axis_data):.6f} m/s")
        print(f"  Min: {np.min(axis_data):+.6f} m/s")
        print(f"  Max: {np.max(axis_data):+.6f} m/s")
        print(f"  25th percentile: {np.percentile(axis_data, 25):+.6f} m/s")
        print(f"  75th percentile: {np.percentile(axis_data, 75):+.6f} m/s")
        print(f"  95% CI: [{np.percentile(axis_data, 2.5):+.6f}, {np.percentile(axis_data, 97.5):+.6f}] m/s")
        
        # Normality test
        _, p_value = stats.normaltest(axis_data)
        print(f"  Normality test p-value: {p_value:.6f} ({'Normal' if p_value > 0.05 else 'Non-normal'})")
    
    # Speed magnitude statistics
    speed_magnitude = np.linalg.norm(velocities, axis=1)
    print(f"\n[Speed Magnitude]")
    print(f"  Average speed: {np.mean(speed_magnitude):.6f} m/s")
    print(f"  Median speed: {np.median(speed_magnitude):.6f} m/s")
    print(f"  Max speed: {np.max(speed_magnitude):.6f} m/s")
    print(f"  Speed std: {np.std(speed_magnitude):.6f} m/s")
    
    # Individual motion statistics
    print(f"\n[Individual Motion Statistics]")
    unique_motions = np.unique(motion_labels)
    
    for motion in unique_motions:
        mask = motion_labels == motion
        motion_vel = velocities[mask]
        if len(motion_vel) == 0:
            continue
            
        print(f"\n  Motion: {motion}")
        print(f"    Data points: {len(motion_vel)}")
        print(f"    Duration: {len(motion_vel)/30:.2f}s (assuming 30fps)")
        
        motion_speed = np.linalg.norm(motion_vel, axis=1)
        print(f"    Average speed: {np.mean(motion_speed):.6f} m/s")
        print(f"    Max speed: {np.max(motion_speed):.6f} m/s")
        
        for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
            axis_data = motion_vel[:, axis_idx]
            print(f"    {axis_name}-axis: Mean={np.mean(axis_data):+.6f}, Std={np.std(axis_data):.6f} m/s")

def main():
    parser = argparse.ArgumentParser(description='Analyze base_lin_vel_local_50window distribution in dataset')
    parser.add_argument('pkl_file', help='Input pkl file path')
    parser.add_argument('--save', '-s', type=str, default=None, help='Directory path to save figures')
    parser.add_argument('--no-plot', action='store_true', help='Only output statistics, do not show plots')
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        motion_data = load_dataset(args.pkl_file)
        
        # Collect all base_lin_vel_local_50window data
        velocities, motion_labels = collect_all_base_lin_vel_local_50window(motion_data)
        
        # Print detailed statistics
        print_detailed_statistics(velocities, motion_labels)
        
        # Plot distribution
        if not args.no_plot:
            plot_velocity_distribution(velocities, motion_labels, args.save)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())