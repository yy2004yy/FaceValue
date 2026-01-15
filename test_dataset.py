"""
测试数据集加载
验证数据集是否能正确加载
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_data_loader
from config import Config


def test_dataset_loading():
    """测试数据集加载"""
    print("=" * 60)
    print("测试数据集加载")
    print("=" * 60)
    
    # 加载配置
    config = Config()
    
    print(f"\n数据集路径: {config.DATASET_ROOT}")
    print(f"训练集文件: {config.TRAIN_SPLIT_FILE}")
    print(f"测试集文件: {config.TEST_SPLIT_FILE}")
    
    # 检查路径是否存在
    if not os.path.exists(config.DATASET_ROOT):
        print(f"\n错误: 数据集路径不存在: {config.DATASET_ROOT}")
        return
    
    # 测试训练集加载
    print("\n" + "-" * 60)
    print("测试训练集加载...")
    print("-" * 60)
    
    try:
        train_loader = get_data_loader(
            root_dir=config.DATASET_ROOT,
            split_file=config.TRAIN_SPLIT_FILE,
            mode='train',
            model_type=config.MODEL_NAME,
            batch_size=4,  # 使用小batch用于测试
            num_workers=0  # Windows上使用0避免多进程问题
        )
        
        print(f"训练集大小: {len(train_loader.dataset)}")
        
        # 加载一个batch测试
        if len(train_loader) > 0:
            images, scores = next(iter(train_loader))
            print(f"Batch大小: {images.shape}")
            print(f"评分范围: {scores.min().item():.2f} - {scores.max().item():.2f}")
            print(f"评分示例: {scores[:5].tolist()}")
            print("✓ 训练集加载成功！")
        else:
            print("⚠ 训练集为空")
            
    except Exception as e:
        print(f"✗ 训练集加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试测试集加载
    print("\n" + "-" * 60)
    print("测试测试集加载...")
    print("-" * 60)
    
    try:
        test_loader = get_data_loader(
            root_dir=config.DATASET_ROOT,
            split_file=config.TEST_SPLIT_FILE,
            mode='test',
            model_type=config.MODEL_NAME,
            batch_size=4,
            num_workers=0
        )
        
        print(f"测试集大小: {len(test_loader.dataset)}")
        
        # 加载一个batch测试
        if len(test_loader) > 0:
            images, scores = next(iter(test_loader))
            print(f"Batch大小: {images.shape}")
            print(f"评分范围: {scores.min().item():.2f} - {scores.max().item():.2f}")
            print(f"评分示例: {scores[:5].tolist()}")
            print("✓ 测试集加载成功！")
        else:
            print("⚠ 测试集为空")
            
    except Exception as e:
        print(f"✗ 测试集加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 检查数据样本
    print("\n" + "-" * 60)
    print("检查数据样本...")
    print("-" * 60)
    
    try:
        dataset = train_loader.dataset
        if len(dataset) > 0:
            img_path, score = dataset.data_list[0]
            print(f"示例图像: {img_path}")
            print(f"示例评分: {score:.4f}")
            print(f"图像文件存在: {os.path.exists(img_path)}")
    except Exception as e:
        print(f"检查数据样本时出错: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    test_dataset_loading()
