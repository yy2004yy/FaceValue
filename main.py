"""
颜值打分系统 - 主程序入口
支持输入照片进行人脸分析和评分
支持两种评分方式：
1. 几何特征评分（基于三庭五眼、对称性等）
2. 深度学习模型评分（基于训练好的神经网络模型）
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from analyzer import FaceAnalyzer
from scorer import FaceScorer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='颜值打分系统 - 输入照片进行人脸分析和评分',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用几何特征评分（默认）
  python main.py path/to/image.jpg
  python main.py path/to/image.jpg --output report.txt
  
  # 使用深度学习模型评分
  python main.py path/to/image.jpg --model outputs/checkpoints/resnet18_best.pth
  python main.py path/to/image.jpg --model outputs/checkpoints/resnet18_best.pth --device cuda
        """
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='输入照片路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出报告文件路径（可选，默认输出到控制台）'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细信息'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='使用深度学习模型评分（提供checkpoint路径，如: outputs/checkpoints/resnet18_best.pth）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='深度学习模型使用的设备（auto: 自动选择）'
    )
    
    args = parser.parse_args()
    
    # 检查图像文件是否存在
    if not os.path.exists(args.image_path):
        print(f"错误：图像文件不存在: {args.image_path}")
        sys.exit(1)
    
    try:
        print("=" * 50)
        print("颜值打分系统")
        print("=" * 50)
        
        # 如果指定了深度学习模型，使用模型预测
        if args.model:
            if not os.path.exists(args.model):
                print(f"错误: 模型文件不存在: {args.model}")
                sys.exit(1)
            
            print(f"\n使用深度学习模型进行预测")
            print(f"模型: {args.model}")
            
            # 导入predict模块
            from predict import predict
            import torch
            
            # 自动选择设备
            if args.device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = args.device
            
            print(f"设备: {device}\n")
            
            # 执行预测
            score, model_name = predict(
                image_path=args.image_path,
                checkpoint_path=args.model,
                device=device
            )
            
            # 生成报告
            report = f"""
=== 颜值评分报告（深度学习模型） ===

【模型信息】
模型: {model_name}
设备: {device}

【预测结果】
颜值分数: {score:.2f}/5.0

【评分说明】
- 使用在SCUT-FBP5500数据集上训练的深度学习模型
- 评分范围: 1.0 - 5.0
- 基于5500张人脸数据训练的模型
"""
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\n报告已保存到: {args.output}\n")
            else:
                print(report)
            
            print("=" * 50)
            print("预测完成！")
            print("=" * 50)
        
        else:
            # 使用几何特征评分（原有功能）
            print(f"\n使用几何特征进行评分")
            print(f"正在分析图像: {args.image_path}\n")
            
            # 初始化分析器和评分器
            print("初始化分析器...")
            analyzer = FaceAnalyzer()
            scorer = FaceScorer()
            
            # 执行分析
            print("正在检测人脸和提取特征...")
            analysis_result = analyzer.analyze(args.image_path)
            
            if args.verbose:
                print(f"\n检测到人脸:")
                print(f"  位置: ({analysis_result['face_bbox']['x']}, {analysis_result['face_bbox']['y']})")
                print(f"  尺寸: {analysis_result['face_bbox']['width']}x{analysis_result['face_bbox']['height']}")
                print(f"  置信度: {analysis_result['face_bbox']['confidence']:.3f}")
            
            # 执行评分
            print("正在计算评分...")
            scores = scorer.score(analysis_result)
            
            # 输出报告
            report = scores['report']
            
            if args.output:
                # 保存到文件
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\n报告已保存到: {args.output}\n")
            else:
                # 输出到控制台
                print(report)
            
            print("=" * 50)
            print("分析完成！")
            print("=" * 50)
        
    except ValueError as e:
        print(f"\n错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生未预期的错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
