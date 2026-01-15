"""
数据准备脚本
解压和处理SCUT-FBP5500数据集
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def extract_zip(zip_path: str, extract_to: str = None):
    """
    解压zip文件
    
    Args:
        zip_path: zip文件路径
        extract_to: 解压目标目录，如果为None则解压到同目录
    """
    if extract_to is None:
        extract_to = str(Path(zip_path).parent)
    
    print(f"正在解压: {zip_path}")
    print(f"目标目录: {extract_to}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print("解压完成！")


def check_dataset_structure(dataset_root: str):
    """
    检查数据集结构
    
    Args:
        dataset_root: 数据集根目录
    """
    print("\n检查数据集结构...")
    dataset_path = Path(dataset_root)
    
    # 检查常见的数据集目录结构
    possible_dirs = [
        'SCUT-FBP-5500_v2.1',
        'SCUT-FBP5500_v2',
        'train_test_files',
        'All_Ratings.xlsx',
        'All_Ratings.txt'
    ]
    
    found_dirs = []
    for item in possible_dirs:
        path = dataset_path / item
        if path.exists():
            found_dirs.append(item)
            print(f"  ✓ 找到: {item}")
        else:
            print(f"  ✗ 未找到: {item}")
    
    # 检查图像目录
    image_count = 0
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        images = list(dataset_path.rglob(f'*{ext}'))
        image_count += len(images)
    
    if image_count > 0:
        print(f"\n找到 {image_count} 张图像")
    else:
        print("\n警告: 未找到图像文件")
    
    return found_dirs


def create_split_files(dataset_root: str, train_ratio: float = 0.6):
    """
    创建训练/测试集划分文件（如果不存在）
    
    Args:
        dataset_root: 数据集根目录
        train_ratio: 训练集比例
    """
    dataset_path = Path(dataset_root)
    
    # 检查是否已有划分文件
    split_dir = dataset_path / 'train_test_files' / 'split_of_60%_training_and_40%_testing'
    if split_dir.exists() and (split_dir / 'train_1.txt').exists():
        print("划分文件已存在，跳过创建")
        return
    
    # 查找评分文件
    score_file = None
    for possible_file in ['All_Ratings.xlsx', 'All_Ratings.txt', 'ratings.txt']:
        possible_path = dataset_path / possible_file
        if possible_path.exists():
            score_file = possible_path
            break
    
    if score_file is None:
        print("警告: 未找到评分文件，无法创建划分文件")
        print("请手动创建 train_1.txt 和 test_1.txt 文件")
        return
    
    print(f"从评分文件创建划分: {score_file}")
    
    # 读取评分文件
    if score_file.suffix == '.xlsx':
        df = pd.read_excel(score_file)
    else:
        df = pd.read_csv(score_file, sep='\t' if '\t' in open(score_file).read() else ',')
    
    # 假设第一列是文件名，第二列是评分
    filenames = df.iloc[:, 0].values
    scores = df.iloc[:, 1].values
    
    # 随机划分
    indices = np.random.permutation(len(filenames))
    split_idx = int(len(filenames) * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # 创建输出目录
    output_dir = dataset_path / 'train_test_files' / 'split_of_60%_training_and_40%_testing'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 写入训练集文件
    with open(output_dir / 'train_1.txt', 'w') as f:
        for idx in train_indices:
            f.write(f"{filenames[idx]}\t{scores[idx]:.2f}\n")
    
    # 写入测试集文件
    with open(output_dir / 'test_1.txt', 'w') as f:
        for idx in test_indices:
            f.write(f"{filenames[idx]}\t{scores[idx]:.2f}\n")
    
    print(f"创建完成: {len(train_indices)} 训练样本, {len(test_indices)} 测试样本")


def main():
    parser = argparse.ArgumentParser(description='准备SCUT-FBP5500数据集')
    parser.add_argument('--zip', type=str, default='dataset/SCUT-FBP5500_v2.zip',
                       help='zip文件路径')
    parser.add_argument('--extract-to', type=str, default='dataset',
                       help='解压目标目录')
    parser.add_argument('--check-only', action='store_true',
                       help='只检查数据集结构，不解压')
    parser.add_argument('--create-split', action='store_true',
                       help='创建训练/测试集划分文件')
    
    args = parser.parse_args()
    
    # 解压
    if not args.check_only:
        if os.path.exists(args.zip):
            extract_zip(args.zip, args.extract_to)
        else:
            print(f"警告: zip文件不存在: {args.zip}")
    
    # 检查数据集结构
    dataset_root = args.extract_to
    if not os.path.exists(dataset_root):
        # 尝试在解压目录中查找数据集文件夹
        possible_roots = [
            os.path.join(args.extract_to, 'SCUT-FBP-5500_v2.1'),
            os.path.join(args.extract_to, 'SCUT-FBP5500_v2'),
            args.extract_to
        ]
        for root in possible_roots:
            if os.path.exists(root):
                dataset_root = root
                break
    
    check_dataset_structure(dataset_root)
    
    # 创建划分文件
    if args.create_split:
        import numpy as np
        create_split_files(dataset_root)
    
    print("\n数据准备完成！")
    print(f"\n数据集路径: {dataset_root}")
    print("请在config.py中设置 DATASET_ROOT 为上述路径")


if __name__ == '__main__':
    main()
