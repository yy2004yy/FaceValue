"""
预测功能集成示例
演示如何在你的代码中使用预测功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from predict import predict


def example_single_prediction():
    """示例1: 单张图片预测"""
    print("=" * 60)
    print("示例1: 单张图片预测")
    print("=" * 60)
    
    score, score_100, model_name = predict(
        image_path='your_image.jpg',  # 替换为实际图像路径
        checkpoint_path='outputs/checkpoints/resnet18_best.pth',
        device='cuda'  # 或 'cpu'
    )
    print(f"颜值分数: {score:.2f}/5.0 ({score_100:.1f}/100分)")
    print(f"使用模型: {model_name}")


def example_batch_prediction():
    """示例2: 批量预测"""
    print("\n" + "=" * 60)
    print("示例2: 批量预测")
    print("=" * 60)
    
    image_paths = [
        'data/photo1.jpg',
        'data/photo2.jpg',
        'data/photo3.jpg'
    ]
    
    checkpoint_path = 'outputs/checkpoints/resnet18_best.pth'
    device = 'cuda'
    
    results = []
    for img_path in image_paths:
        try:
            score, score_100, model_name = predict(img_path, checkpoint_path, device)
            results.append((img_path, score, score_100))
            print(f"{img_path}: {score:.2f}/5.0 ({score_100:.1f}/100分)")
        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")
    
    # 保存结果
    with open('prediction_results.txt', 'w') as f:
        for path, score in results:
            f.write(f"{path}\t{score:.2f}\n")
    
    print(f"\n结果已保存到: prediction_results.txt")


def example_integration():
    """示例3: 集成到你的应用"""
    print("\n" + "=" * 60)
    print("示例3: 集成到应用")
    print("=" * 60)
    
    class BeautyPredictor:
        """颜值预测器类"""
        
        def __init__(self, checkpoint_path, device='cuda'):
            self.checkpoint_path = checkpoint_path
            self.device = device
            print(f"初始化预测器，模型: {checkpoint_path}")
        
        def predict_score(self, image_path):
            """预测单张图片的颜值分数"""
            score, score_100, model_name = predict(
                image_path=image_path,
                checkpoint_path=self.checkpoint_path,
                device=self.device
            )
            return {
                'score': score,
                'score_100': score_100,
                'model': model_name,
                'image': image_path
            }
        
        def predict_batch(self, image_paths):
            """批量预测"""
            results = []
            for img_path in image_paths:
                result = self.predict_score(img_path)
                results.append(result)
            return results
    
    # 使用示例
    predictor = BeautyPredictor(
        checkpoint_path='outputs/checkpoints/resnet18_best.pth',
        device='cuda'
    )
    
    # 预测单张图片
    result = predictor.predict_score('data/test_photo.jpg')
    print(f"预测结果: {result}")
    
    # 批量预测
    image_list = ['data/photo1.jpg', 'data/photo2.jpg']
    results = predictor.predict_batch(image_list)
    for r in results:
        print(f"{r['image']}: {r['score']:.2f}/5.0 ({r['score_100']:.1f}/100分)")


def example_web_api():
    """示例4: Web API集成（Flask示例）"""
    print("\n" + "=" * 60)
    print("示例4: Web API集成（代码示例）")
    print("=" * 60)
    
    code_example = '''
from flask import Flask, request, jsonify
from predict import predict
import os

app = Flask(__name__)
CHECKPOINT = 'outputs/checkpoints/resnet18_best.pth'
DEVICE = 'cuda'  # 或 'cpu'

@app.route('/predict', methods=['POST'])
def predict_beauty():
    """API端点：预测颜值"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400
    
    image_file = request.files['image']
    image_path = f'temp/{image_file.filename}'
    image_file.save(image_path)
    
    try:
        score, model_name = predict(image_path, CHECKPOINT, DEVICE)
        
        return jsonify({
            'success': True,
            'score': round(score, 2),
            'model': model_name,
            'range': '1.0-5.0'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
'''
    print(code_example)


if __name__ == '__main__':
    print("预测功能集成示例\n")
    print("注意: 请先训练模型或使用已训练好的checkpoint文件\n")
    
    # 运行示例（注释掉你不想运行的）
    # example_single_prediction()
    # example_batch_prediction()
    # example_integration()
    example_web_api()
    
    print("\n" + "=" * 60)
    print("提示: 取消注释上面的函数调用以运行示例")
    print("=" * 60)
