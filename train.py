"""
训练脚本
基于SCUT-FBP5500论文的训练流程
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from models import get_model
from dataset import get_data_loader
from utils.metrics import calculate_metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> dict:
    """
    训练一个epoch
    
    Returns:
        包含损失和指标的训练统计信息
    """
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    for batch_idx, (images, scores) in enumerate(pbar):
        images = images.to(device)
        scores = scores.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, scores)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(scores.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
        
        # 记录到tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        if writer:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
    
    # 计算平均损失和指标
    avg_loss = running_loss / len(dataloader)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = calculate_metrics(predictions, targets)
    
    if writer:
        writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Train/PC', metrics['pc'], epoch)
        writer.add_scalar('Train/MAE', metrics['mae'], epoch)
        writer.add_scalar('Train/RMSE', metrics['rmse'], epoch)
    
    return {
        'loss': avg_loss,
        **metrics
    }


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> dict:
    """
    验证模型
    
    Returns:
        包含损失和指标的验证统计信息
    """
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
        for images, scores in pbar:
            images = images.to(device)
            scores = scores.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, scores)
            
            running_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets.extend(scores.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # 计算平均损失和指标
    avg_loss = running_loss / len(dataloader)
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    metrics = calculate_metrics(predictions, targets)
    
    if writer:
        writer.add_scalar('Val/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Val/PC', metrics['pc'], epoch)
        writer.add_scalar('Val/MAE', metrics['mae'], epoch)
        writer.add_scalar('Val/RMSE', metrics['rmse'], epoch)
    
    return {
        'loss': avg_loss,
        **metrics
    }


def main():
    parser = argparse.ArgumentParser(description='训练颜值预测模型')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径（JSON格式）')
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['alexnet', 'resnet18', 'resnext50'],
                       help='模型类型')
    parser.add_argument('--dataset', type=str, default=None, help='数据集根目录')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config(**config_dict)
    else:
        config = Config()
    
    # 命令行参数覆盖配置
    if args.model:
        config.MODEL_NAME = args.model
    if args.dataset:
        config.DATASET_ROOT = args.dataset
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    
    # 设置设备
    if config.DEVICE == 'cuda' and torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    print(f"模型: {config.MODEL_NAME}")
    print(f"数据集: {config.DATASET_ROOT}")
    
    # 创建数据加载器
    print("加载数据...")
    train_loader = get_data_loader(
        root_dir=config.DATASET_ROOT,
        split_file=config.TRAIN_SPLIT_FILE,
        mode='train',
        model_type=config.MODEL_NAME,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = get_data_loader(
        root_dir=config.DATASET_ROOT,
        split_file=config.TEST_SPLIT_FILE,
        mode='test',
        model_type=config.MODEL_NAME,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    print(f"创建模型: {config.MODEL_NAME}...")
    model = get_model(config.MODEL_NAME, pretrained=config.PRETRAINED)
    model = model.to(device)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度器（根据迭代次数，而非epoch）
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_DECAY_STEP // len(train_loader),  # 转换为epoch数
        gamma=config.LR_DECAY_FACTOR
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    # 恢复训练
    start_epoch = 0
    best_metric = float('inf')  # 用于保存最佳模型（基于MAE）
    
    if args.resume:
        print(f"恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint.get('best_metric', float('inf'))
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        # 训练
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # 验证
        val_stats = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Train - Loss: {train_stats['loss']:.4f}, "
              f"PC: {train_stats['pc']:.4f}, MAE: {train_stats['mae']:.4f}, "
              f"RMSE: {train_stats['rmse']:.4f}")
        print(f"Val   - Loss: {val_stats['loss']:.4f}, "
              f"PC: {val_stats['pc']:.4f}, MAE: {val_stats['mae']:.4f}, "
              f"RMSE: {val_stats['rmse']:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # 保存checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_stats': train_stats,
            'val_stats': val_stats,
            'config': config.__dict__,
            'best_metric': best_metric
        }
        
        # 保存最新模型
        latest_path = os.path.join(config.CHECKPOINT_DIR, f'{config.MODEL_NAME}_latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型（基于验证集MAE）
        if val_stats['mae'] < best_metric:
            best_metric = val_stats['mae']
            checkpoint['best_metric'] = best_metric
            best_path = os.path.join(config.CHECKPOINT_DIR, f'{config.MODEL_NAME}_best.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型 (MAE: {best_metric:.4f})")
        
        # 定期保存
        if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
            epoch_path = os.path.join(config.CHECKPOINT_DIR, 
                                    f'{config.MODEL_NAME}_epoch_{epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
    
    writer.close()
    print("\n训练完成！")


if __name__ == '__main__':
    main()
