# 快速开始指南

## 一、环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "import torch; import cv2; import mediapipe; print('所有依赖安装成功！')"
```

## 二、使用几何特征评分（无需训练）

### 快速测试

```bash
# 准备一张包含人脸的图片
python main.py data/your_photo.jpg

# 保存报告
python main.py data/your_photo.jpg --output report.txt
```

## 三、使用深度学习模型（需要训练）

### 1. 准备数据集

```bash
# 步骤1: 下载SCUT-FBP5500数据集
# 从 https://github.com/HCIILAB/SCUT-FBP5500-Database-Release 下载

# 步骤2: 解压数据集
python prepare_data.py --zip dataset/SCUT-FBP5500_v2.zip --extract-to dataset

# 步骤3: 检查数据集结构
python prepare_data.py --check-only --extract-to dataset/SCUT-FBP5500_v2
```

### 2. 训练模型

```bash
# 使用ResNet-18训练（推荐，效果最好）
python train.py --model resnet18 --dataset dataset/SCUT-FBP5500_v2 --gpu 0

# 使用AlexNet训练（速度最快）
python train.py --model alexnet --dataset dataset/SCUT-FBP5500_v2 --gpu 0

# 使用ResNeXt-50训练（精度最高）
python train.py --model resnext50 --dataset dataset/SCUT-FBP5500_v2 --gpu 0
```

### 3. 评估模型

```bash
# 评估最佳模型
python evaluate.py --checkpoint outputs/checkpoints/resnet18_best.pth --dataset dataset/SCUT-FBP5500_v2
```

### 4. 使用模型预测

```bash
# 预测单张图像
python predict.py data/test_face.jpg --checkpoint outputs/checkpoints/resnet18_best.pth

# 使用GPU加速
python predict.py data/test_face.jpg --checkpoint outputs/checkpoints/resnet18_best.pth --gpu
```

## 四、完整工作流程示例

### 场景1: 快速测试（无需训练）

```bash
# 1. 输入一张照片
python main.py data/my_photo.jpg

# 2. 查看评分报告
# 输出包含三庭、五眼、对称性等详细评分
```

### 场景2: 训练和使用深度学习模型

```bash
# 1. 准备数据
python prepare_data.py --zip dataset/SCUT-FBP5500_v2.zip --extract-to dataset

# 2. 训练模型（约需几个小时，取决于GPU）
python train.py --model resnet18 --dataset dataset/SCUT-FBP5500_v2 --gpu 0

# 3. 训练过程中可以查看进度
tensorboard --logdir outputs/logs

# 4. 评估模型性能
python evaluate.py --checkpoint outputs/checkpoints/resnet18_best.pth --dataset dataset/SCUT-FBP5500_v2

# 5. 使用模型预测新照片
python predict.py data/new_photo.jpg --checkpoint outputs/checkpoints/resnet18_best.pth
```

## 五、常见问题

### Q1: 训练时内存不足？

**A:** 减小batch_size：
```bash
python train.py --model resnet18 --batch-size 8 --dataset dataset/SCUT-FBP5500_v2
```

### Q2: 找不到数据集？

**A:** 检查数据集路径，并更新config.py中的DATASET_ROOT：
```python
# config.py
DATASET_ROOT = 'dataset/SCUT-FBP5500_v2'  # 修改为你的路径
```

### Q3: 训练太慢？

**A:** 
- 使用GPU: `python train.py ... --gpu 0`
- 使用更小的模型: `--model alexnet`
- 减少epoch数（仅用于测试）

### Q4: 如何使用自己的数据集？

**A:** 修改`dataset/dataset_loader.py`中的`_load_data_list`方法，或创建符合SCUT-FBP5500格式的标注文件。

## 六、预期结果

### 几何特征评分
- 无需训练，即时可用
- 基于传统美学标准
- 适用于快速评估

### 深度学习模型（ResNet-18）
- Pearson Correlation (PC): ~0.89
- Mean Absolute Error (MAE): ~0.24
- Root Mean Square Error (RMSE): ~0.32

## 七、下一步

- 查看完整文档: `README.md`
- 了解模型架构: `models/neural_models.py`
- 自定义训练配置: `config.py`
- 查看训练日志: `tensorboard --logdir outputs/logs`
