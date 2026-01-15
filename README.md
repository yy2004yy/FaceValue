# 颜值打分系统 (Appearance Score System)

一个基于计算机视觉和几何分析的颜值评估系统，能够分析人脸特征并进行多维度评分。

## 项目结构

```
Appearance_score/
├── analyzer/              # 面部解析模块 (The Analyzer)
│   ├── __init__.py
│   └── face_analyzer.py  # 人脸检测和特征提取
├── scorer/               # 评分模块 (The Scorer)
│   ├── __init__.py
│   └── face_scorer.py    # 多维度评分系统（几何特征）
├── beautifier/           # 美化模块 (The Beautifier) - 第三阶段
│   └── __init__.py
├── dataset/              # 数据集模块
│   ├── __init__.py
│   └── dataset_loader.py # SCUT-FBP5500数据集加载器
├── models/               # 模型定义目录
│   ├── __init__.py
│   └── neural_models.py  # 深度学习模型（AlexNet, ResNet-18, ResNeXt-50）
├── utils/                # 工具函数模块
│   ├── __init__.py
│   ├── image_utils.py    # 图像处理工具
│   ├── geometry_utils.py # 几何计算工具
│   └── metrics.py        # 评估指标计算
├── outputs/              # 训练输出目录
│   ├── checkpoints/      # 模型检查点
│   └── logs/             # TensorBoard日志
├── dataset/              # 数据集存放目录
├── tests/                # 测试文件目录
├── config.py            # 训练配置文件
├── train.py             # 训练脚本
├── evaluate.py          # 评估脚本
├── predict.py           # 模型预测脚本
├── prepare_data.py      # 数据准备脚本
├── main.py              # 主程序入口（几何特征评分）
├── requirements.txt     # 依赖文件
└── README.md           # 项目说明
```

## 功能特性

### 第一阶段：面部解析与特征提取 ✅

- **人脸检测**：使用MediaPipe检测照片中的人脸
- **关键点提取**：提取468个人脸关键点
- **几何特征计算**：
  - 三庭比例（上庭、中庭、下庭）
  - 五眼比例（脸宽与眼睛宽度的关系）
  - 面部对称性分析

### 第二阶段：多维评分系统 ✅

- **三庭评分**：基于三庭比例的评分（0-5分）
- **五眼评分**：基于五眼比例的评分（0-5分）
- **对称性评分**：基于面部对称性的评分（0-5分）
- **综合评分**：加权平均的综合颜值评分
- **详细报告**：生成包含各维度评分的详细报告

### 第二阶段（深度学习）：基于SCUT-FBP5500的颜值预测模型 ✅

- **深度学习模型**：实现AlexNet、ResNet-18、ResNeXt-50三种模型
- **数据加载**：SCUT-FBP5500数据集加载和预处理
- **模型训练**：基于论文的训练配置（ImageNet预训练、L2损失、学习率衰减等）
- **模型评估**：计算Pearson Correlation (PC)、MAE、RMSE指标
- **模型推理**：使用训练好的模型进行颜值预测

### 第三阶段：AIGC美化功能（规划中）

- 基于Stable Diffusion的图像生成
- ControlNet约束的人脸美化
- 保留特征的微调

## 安装指南

### 1. 环境要求

- Python 3.8+
- pip

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import cv2, numpy, mediapipe; print('依赖安装成功')"
```

## 使用方法

### 命令行使用

基本用法：

```bash
python main.py <图片路径>
```

示例：

```bash
python main.py data/test_face.jpg
```

保存报告到文件：

```bash
python main.py data/test_face.jpg --output report.txt
```

显示详细信息：

```bash
python main.py data/test_face.jpg --verbose
```

### Python API 使用

```python
from analyzer import FaceAnalyzer
from scorer import FaceScorer

# 初始化
analyzer = FaceAnalyzer()
scorer = FaceScorer()

# 分析图像
analysis_result = analyzer.analyze('path/to/image.jpg')

# 计算评分
scores = scorer.score(analysis_result)

# 查看结果
print(scores['report'])
print(f"综合评分: {scores['overall']['overall_score']}/5.0")
```

## 评分说明

### 评分维度

1. **三庭比例评分**（权重35%）
   - 上庭：发际线到眉间的比例
   - 中庭：眉间到鼻底的比例
   - 下庭：鼻底到下巴的比例
   - 理想比例约为 1:1:1

2. **五眼比例评分**（权重30%）
   - 脸宽应该等于五个眼睛的宽度
   - 理想比例：脸宽 / (5 × 平均眼宽) = 1.0

3. **对称性评分**（权重35%）
   - 基于左右眼、眉毛、嘴角的对称程度
   - 对称性越高，分数越高

### 评分标准

- **0-5分制**：每个维度独立评分
- **综合评分**：加权平均（三庭35% + 五眼30% + 对称性35%）
- **骨相评分**：基于几何特征的评分

## 技术栈

### 基础功能
- **OpenCV**：图像处理
- **MediaPipe**：人脸检测和关键点提取（468点）
- **NumPy**：数值计算

### 深度学习（基于SCUT-FBP5500）
- **PyTorch**：深度学习框架
- **TorchVision**：预训练模型和图像变换
- **TensorBoard**：训练过程可视化
- **SCUT-FBP5500数据集**：5500张人脸照片，1-5分颜值评分

### 编程语言
- **Python 3.8+**

## 深度学习模型训练

本项目支持基于SCUT-FBP5500数据集训练深度学习模型进行颜值预测。

### 1. 数据准备

#### 下载数据集
从[SCUT-FBP5500 GitHub仓库](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)下载数据集：
- 中国用户：[百度网盘](https://pan.baidu.com/s/1Ff2W2VLJ1ZbWSeV5JbF0lw) (密码: if7p)
- 其他用户：[Google Drive](https://drive.google.com/open?id=1w0TorBfTlqbquQVd6k3h_77ypnrvfGwf)

#### 准备数据
```bash
# 解压数据集
python prepare_data.py --zip dataset/SCUT-FBP5500_v2.zip --extract-to dataset

# 检查数据集结构
python prepare_data.py --check-only --extract-to dataset/SCUT-FBP5500_v2

# 创建训练/测试集划分文件（如果需要）
python prepare_data.py --extract-to dataset/SCUT-FBP5500_v2 --create-split
```

### 2. 训练模型

#### 基本训练
```bash
# 使用ResNet-18训练（推荐）
python train.py --model resnet18 --dataset dataset/SCUT-FBP5500_v2

# 使用AlexNet训练
python train.py --model alexnet --dataset dataset/SCUT-FBP5500_v2

# 使用ResNeXt-50训练
python train.py --model resnext50 --dataset dataset/SCUT-FBP5500_v2
```

#### 自定义训练参数
```bash
python train.py \
    --model resnet18 \
    --dataset dataset/SCUT-FBP5500_v2 \
    --batch-size 16 \
    --lr 0.001 \
    --epochs 100 \
    --gpu 0
```

#### 恢复训练
```bash
python train.py --resume outputs/checkpoints/resnet18_best.pth
```

### 3. 评估模型

```bash
# 评估最佳模型
python evaluate.py --checkpoint outputs/checkpoints/resnet18_best.pth --dataset dataset/SCUT-FBP5500_v2

# 评估指定checkpoint
python evaluate.py --checkpoint outputs/checkpoints/resnet18_epoch_50.pth
```

### 4. 训练配置

训练配置基于SCUT-FBP5500论文：
- **图像尺寸**：256×256，随机裁剪（AlexNet: 227×227, ResNet: 224×224）
- **预训练模型**：ImageNet预训练权重
- **优化器**：SGD (momentum=0.9, weight_decay=5e-4/1e-4)
- **学习率**：0.001，每5000次迭代衰减10倍
- **损失函数**：L2-norm distance loss (MSE)
- **批次大小**：16
- **评估指标**：Pearson Correlation (PC), MAE, RMSE

### 5. 预期结果

根据SCUT-FBP5500论文，预期性能指标：

| 模型 | PC | MAE | RMSE |
|------|----|----|------|
| AlexNet | 0.8634 | 0.2651 | 0.3481 |
| ResNet-18 | 0.89 | 0.2419 | 0.3166 |
| ResNeXt-50 | 0.8997 | 0.2291 | 0.3017 |

### 6. 使用训练好的模型进行预测

#### 命令行预测
```bash
# 预测单张图像
python predict.py path/to/image.jpg --checkpoint outputs/checkpoints/resnet18_best.pth

# 使用GPU
python predict.py path/to/image.jpg --checkpoint outputs/checkpoints/resnet18_best.pth --gpu

# 批量预测
python predict.py dummy.jpg --checkpoint outputs/checkpoints/resnet18_best.pth --batch image_list.txt
```

#### Python API使用
```python
from models import get_model
import torch
from PIL import Image
from torchvision import transforms

# 加载模型
model = get_model('resnet18', pretrained=False)
checkpoint = torch.load('outputs/checkpoints/resnet18_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预处理图像
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 预测
image = Image.open('path/to/image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    score = model(image_tensor).item()
    print(f"预测颜值分数: {score:.2f}/5.0")
```

### 7. TensorBoard可视化

```bash
# 启动TensorBoard
tensorboard --logdir outputs/logs

# 在浏览器中打开 http://localhost:6006
```

## 开发计划

- [x] 第一阶段：面部解析与特征提取
- [x] 第二阶段：多维评分系统（几何特征）
- [x] 第二阶段：深度学习模型训练（SCUT-FBP5500）
- [ ] 第三阶段：AIGC美化功能
  - [ ] Stable Diffusion集成
  - [ ] ControlNet约束
  - [ ] 局部重绘功能

## 注意事项

1. **输入图像要求**：
   - 支持常见图像格式（jpg, png等）
   - 图像中应包含清晰可见的人脸
   - 建议正面照，光线充足

2. **性能**：
   - 首次运行可能较慢（模型加载）
   - 处理单张图像通常需要1-3秒

3. **精度说明**：
   - 评分基于几何特征，仅供参考
   - 不同角度、光线会影响结果
   - 评分标准基于传统美学比例

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎提出Issue。
