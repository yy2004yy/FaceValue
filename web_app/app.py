"""
颜值评分系统 - Web应用
基于Flask的在线颜值评分平台
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from analyzer import FaceAnalyzer
from scorer import FaceScorer
from predict import predict as dl_predict

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
app.config['UPLOAD_FOLDER'] = 'web_app/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量存储分析器和模型
face_analyzer = None
face_scorer = None
dl_model_cache = {}


def init_analyzers():
    """初始化分析器和评分器"""
    global face_analyzer, face_scorer
    if face_analyzer is None:
        face_analyzer = FaceAnalyzer()
    if face_scorer is None:
        face_scorer = FaceScorer()


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def image_to_base64(image_path):
    """将图片转换为base64编码"""
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        return base64.b64encode(img_data).decode('utf-8')
    except Exception:
        return None


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/score', methods=['POST'])
def score_image():
    """评分API接口"""
    try:
        # 检查是否有文件
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 获取评分模式
        mode = request.form.get('mode', 'geometric')  # geometric, dl, both
        model_path = request.form.get('model_path', 'outputs/resnet18/checkpoints/resnet18_best.pth')
        
        if not file or not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = {}
        
        # 几何特征评分
        if mode in ['geometric', 'both']:
            try:
                init_analyzers()
                analysis_result = face_analyzer.analyze(filepath)
                
                if not analysis_result.get('face_detected', False):
                    return jsonify({'error': '未检测到人脸，请上传包含清晰人脸的图片'}), 400
                
                scores = face_scorer.score(analysis_result)
                results['geometric'] = {
                    'overall': scores['overall'],  # 保留完整的overall字典
                    'overall_score': scores['overall']['overall_score'],
                    'three_regions': scores['three_regions']['three_regions_score'],
                    'five_eyes': scores['five_eyes']['five_eyes_score'],
                    'symmetry': scores['symmetry']['symmetry_score'],
                    'report': scores['report']
                }
            except Exception as e:
                return jsonify({'error': f'几何特征分析失败: {str(e)}'}), 500
        
        # 深度学习评分
        if mode in ['dl', 'both']:
            if not TORCH_AVAILABLE:
                return jsonify({'error': 'PyTorch未安装，无法使用深度学习模式'}), 400
            
            if not os.path.exists(model_path):
                return jsonify({'error': f'模型文件不存在: {model_path}'}), 400
            
            try:
                device = 'cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu'
                score, score_100, model_name = dl_predict(filepath, model_path, device)
                
                results['dl'] = {
                    'score': score,
                    'score_100': score_100,
                    'model_name': model_name,
                    'device': device
                }
            except Exception as e:
                return jsonify({'error': f'深度学习预测失败: {str(e)}'}), 500
        
        # 计算最终得分
        if mode == 'both':
            final_score = (results['geometric']['overall_score'] + results['dl']['score']) / 2
        elif mode == 'geometric':
            final_score = results['geometric']['overall_score']
        else:  # dl
            final_score = results['dl']['score']
        
        # 将图片转换为base64用于前端显示
        image_base64 = image_to_base64(filepath)
        
        # 清理临时文件
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'final_score': final_score,
            'final_score_100': final_score * 20,
            'mode': mode,
            'results': results,
            'image_base64': image_base64
        })
    
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用的模型列表"""
    models = []
    checkpoint_dir = 'outputs'
    
    # 扫描outputs目录下的所有best.pth文件
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith('_best.pth'):
                model_path = os.path.join(root, file)
                models.append({
                    'name': file.replace('_best.pth', ''),
                    'path': model_path
                })
    
    return jsonify({'models': models})


if __name__ == '__main__':
    print("=" * 50)
    print("颜值评分系统 - Web应用")
    print("=" * 50)
    print(f"访问地址: http://localhost:5000")
    print(f"PyTorch可用: {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        import torch
        print(f"CUDA可用: {torch.cuda.is_available()}")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
