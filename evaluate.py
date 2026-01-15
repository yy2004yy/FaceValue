"""
评估脚本
计算模型在测试集上的PC, MAE, RMSE指标
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from models import get_model
from dataset import get_data_loader
from utils.metrics import calculate_metrics


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    verbose: bool = True
) -> dict:
    """
    评估模型
    
    Returns:
        包含PC, MAE, RMSE的评估结果
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='评估中') if verbose else dataloader
        for images, scores in pbar:
            images = images.to(device)
            scores = scores.to(device)
            
            outputs = model(images)
            
            predictions.extend(outputs.cpu().numpy())
            targets.extend(scores.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = calculate_metrics(predictions, targets)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='评估颜值预测模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default=None, help='数据集根目录')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--split-file', type=str, default=None, help='测试集划分文件')
    
    args = parser.parse_args()
    
    # 加载checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"错误: checkpoint文件不存在: {args.checkpoint}")
        sys.exit(1)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 获取配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = Config(**config_dict)
    else:
        config = Config()
    
    # 命令行参数覆盖
    if args.dataset:
        config.DATASET_ROOT = args.dataset
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.split_file:
        config.TEST_SPLIT_FILE = args.split_file
    
    # 设置设备
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    print(f"加载模型: {args.checkpoint}")
    
    # 创建模型
    model_name = config.MODEL_NAME if 'config' in checkpoint else 'resnet18'
    model = get_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型: {model_name}")
    
    # 创建测试数据加载器
    test_loader = get_data_loader(
        root_dir=config.DATASET_ROOT,
        split_file=config.TEST_SPLIT_FILE or args.split_file,
        mode='test',
        model_type=model_name,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 评估
    print("\n开始评估...")
    metrics = evaluate_model(model, test_loader, device)
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    print(f"Pearson Correlation (PC): {metrics['pc']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.4f}")
    print("="*50)
    
    # 如果checkpoint中有验证集指标，也打印出来对比
    if 'val_stats' in checkpoint:
        val_stats = checkpoint['val_stats']
        print("\n训练时的验证集指标（参考）:")
        print(f"  PC: {val_stats.get('pc', 'N/A'):.4f}")
        print(f"  MAE: {val_stats.get('mae', 'N/A'):.4f}")
        print(f"  RMSE: {val_stats.get('rmse', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
