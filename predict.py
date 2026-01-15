"""
使用训练好的深度学习模型进行颜值预测
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from models import get_model
from config import Config


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        device: 设备 ('cpu' 或 'cuda')
        
    Returns:
        (model, model_name, config)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型名称和配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        model_name = config_dict.get('MODEL_NAME', 'resnet18')
    else:
        model_name = 'resnet18'
    
    # 创建模型
    model = get_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, model_name, checkpoint.get('config', {})


def preprocess_image(image_path: str, model_name: str = 'resnet18'):
    """
    预处理图像
    
    Args:
        image_path: 图像路径
        model_name: 模型名称
        
    Returns:
        预处理后的图像tensor
    """
    # 根据模型类型确定裁剪尺寸
    if model_name.lower() == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def predict(image_path: str, checkpoint_path: str, device: str = 'cpu'):
    """
    预测单张图像的颜值分数
    
    Args:
        image_path: 图像路径
        checkpoint_path: 模型checkpoint路径
        device: 设备
        
    Returns:
        预测分数 (1-5分)
    """
    # 加载模型
    model, model_name, config = load_model(checkpoint_path, device)
    
    # 预处理图像
    image_tensor = preprocess_image(image_path, model_name)
    image_tensor = image_tensor.to(device)
    
    # 预测
    with torch.no_grad():
        score = model(image_tensor).item()
    
    # 限制分数在1-5范围内
    score = max(1.0, min(5.0, score))
    
    return score, model_name


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型预测颜值')
    parser.add_argument('image', type=str, help='输入图像路径')
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs/checkpoints/resnet18_best.pth',
                       help='模型checkpoint路径')
    parser.add_argument('--gpu', action='store_true', help='使用GPU')
    parser.add_argument('--batch', type=str, default=None, 
                       help='批量预测：包含图像路径列表的文本文件')
    
    args = parser.parse_args()
    
    # 设置设备
    device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'
    
    # 批量预测
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"错误: 批量文件不存在: {args.batch}")
            sys.exit(1)
        
        print(f"批量预测模式")
        print(f"模型: {args.checkpoint}")
        print(f"设备: {device}\n")
        
        results = []
        with open(args.batch, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
        
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"警告: 图像不存在，跳过: {image_path}")
                continue
            
            try:
                score, model_name = predict(image_path, args.checkpoint, device)
                results.append((image_path, score))
                print(f"{image_path}: {score:.2f}/5.0")
            except Exception as e:
                print(f"错误: 处理 {image_path} 时出错: {e}")
        
        # 保存结果
        output_file = args.batch.replace('.txt', '_results.txt')
        with open(output_file, 'w') as f:
            for path, score in results:
                f.write(f"{path}\t{score:.2f}\n")
        
        print(f"\n结果已保存到: {output_file}")
    
    # 单张预测
    else:
        if not os.path.exists(args.image):
            print(f"错误: 图像文件不存在: {args.image}")
            sys.exit(1)
        
        print(f"预测图像: {args.image}")
        print(f"模型: {args.checkpoint}")
        print(f"设备: {device}\n")
        
        try:
            score, model_name = predict(args.image, args.checkpoint, device)
            print("=" * 50)
            print(f"预测结果:")
            print(f"  图像: {args.image}")
            print(f"  模型: {model_name}")
            print(f"  颜值分数: {score:.2f}/5.0")
            print("=" * 50)
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
