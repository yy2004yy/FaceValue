"""
便捷的颜值预测脚本
使用训练好的深度学习模型进行预测
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from predict import predict


def main():
    """主函数 - 简单的预测接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='使用深度学习模型预测颜值')
    parser.add_argument('image', type=str, help='输入图像路径')
    parser.add_argument(
        '--checkpoint', 
        type=str,
        default='outputs/checkpoints/resnet18_best.pth',
        help='模型checkpoint路径（默认: outputs/checkpoints/resnet18_best.pth）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='使用设备（auto: 自动选择, cpu: CPU, cuda: GPU）'
    )
    
    args = parser.parse_args()
    
    # 检查图像文件
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        sys.exit(1)
    
    # 检查checkpoint文件
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型文件不存在: {args.checkpoint}")
        print(f"请先训练模型或检查路径是否正确")
        sys.exit(1)
    
    # 自动选择设备
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # 执行预测
    try:
        print(f"正在预测: {args.image}")
        print(f"使用模型: {args.checkpoint}")
        print(f"设备: {device}\n")
        
        score, score_100, model_name = predict(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            device=device
        )
        
        print("=" * 50)
        print(f"预测结果:")
        print(f"  图像: {args.image}")
        print(f"  模型: {model_name}")
        print(f"  颜值分数: {score:.2f}/5.0 ({score_100:.1f}/100分)")
        print("=" * 50)
        
    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
