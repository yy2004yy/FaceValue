"""
训练配置文件
基于SCUT-FBP5500论文的训练配置
"""

import os
from pathlib import Path


class Config:
    """训练配置类"""
    
    # 数据集配置
    DATASET_ROOT = 'dataset/SCUT-FBP5500_v2'  # 解压后的数据集路径
    SPLIT_MODE = '60_split'  # '60_split' 或 '5fold_cv'
    TRAIN_SPLIT_FILE = None  # 如果为None，会自动查找
    TEST_SPLIT_FILE = None   # 如果为None，会自动查找
    
    # 模型配置
    MODEL_NAME = 'resnet18'  # 'alexnet', 'resnet18', 'resnext50'
    PRETRAINED = True  # 是否使用ImageNet预训练权重
    
    # 训练配置（基于论文）
    BATCH_SIZE = 16
    NUM_EPOCHS = 100  # 最大epoch数（论文中基于迭代次数）
    MAX_ITERATIONS = 20000  # 最大迭代次数（论文配置）
    
    # 优化器配置
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4  # AlexNet: 5e-4, ResNet/ResNeXt: 1e-4
    LR_DECAY_STEP = 5000  # 每5000次迭代衰减
    LR_DECAY_FACTOR = 0.1  # 学习率衰减因子
    
    # 损失函数
    LOSS_FN = 'mse'  # L2-norm distance loss (MSE)
    
    # 训练设置
    NUM_WORKERS = 4
    PIN_MEMORY = True
    # 自动检测GPU（Windows和Linux都兼容）
    try:
        import torch
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        DEVICE = 'cpu'
    
    # 图像尺寸配置
    IMAGE_SIZE = 256  # 输入图像尺寸
    ALEXNET_CROP = 227  # AlexNet裁剪尺寸
    RESNET_CROP = 224   # ResNet/ResNeXt裁剪尺寸
    
    # 输出配置（按模型名分类，更整洁）
    OUTPUT_DIR = 'outputs'
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')  # 默认路径（向后兼容）
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # 评估指标
    EVAL_METRICS = ['pc', 'mae', 'rmse']  # Pearson correlation, MAE, RMSE
    
    # 模型保存
    SAVE_BEST = True
    SAVE_EVERY_N_EPOCHS = 10
    
    def __init__(self, **kwargs):
        """允许通过kwargs覆盖默认配置"""
        for key, value in kwargs.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
            else:
                setattr(self, key, value)
        
        # 创建输出目录（按模型名分类）
        # 使用更整洁的目录结构：outputs/{MODEL_NAME}/checkpoints/
        model_checkpoint_dir = os.path.join(self.OUTPUT_DIR, self.MODEL_NAME, 'checkpoints')
        model_log_dir = os.path.join(self.OUTPUT_DIR, self.MODEL_NAME, 'logs')
        
        # 更新checkpoint和log目录路径
        self.CHECKPOINT_DIR = model_checkpoint_dir
        self.LOG_DIR = model_log_dir
        
        # 创建目录
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        
        # 根据模型类型调整weight_decay
        if self.MODEL_NAME.lower() == 'alexnet':
            self.WEIGHT_DECAY = 5e-4
        else:
            self.WEIGHT_DECAY = 1e-4
        
        # 自动查找划分文件
        if self.TRAIN_SPLIT_FILE is None or self.TEST_SPLIT_FILE is None:
            self._find_split_files()
    
    def _find_split_files(self):
        """自动查找训练/测试集划分文件"""
        base_path = Path(self.DATASET_ROOT)
        
        # 查找split文件目录（按优先级）
        # 优先级1: 60%训练和40%测试划分（实际文件夹名，有空格）
        split_dir_60 = base_path / 'train_test_files' / 'split_of_60%training and 40%testing'
        # 优先级2: 下划线版本
        split_dir_60_alt = base_path / 'train_test_files' / 'split_of_60%_training_and_40%_testing'
        # 优先级3: 5折交叉验证
        cv_dir = base_path / 'train_test_files' / '5_folders_cross_validations_files' / 'cross_validation_1'
        
        # 尝试60%划分
        for split_dir in [split_dir_60, split_dir_60_alt]:
            if split_dir.exists():
                train_file = split_dir / 'train.txt'
                test_file = split_dir / 'test.txt'
                
                if train_file.exists():
                    self.TRAIN_SPLIT_FILE = str(train_file)
                if test_file.exists():
                    self.TEST_SPLIT_FILE = str(test_file)
                
                if self.TRAIN_SPLIT_FILE and self.TEST_SPLIT_FILE:
                    return
        
        # 尝试5折交叉验证
        if cv_dir.exists():
            train_file = cv_dir / 'train_1.txt'
            test_file = cv_dir / 'test_1.txt'
            
            if train_file.exists():
                self.TRAIN_SPLIT_FILE = str(train_file)
            if test_file.exists():
                self.TEST_SPLIT_FILE = str(test_file)
            
            if self.TRAIN_SPLIT_FILE and self.TEST_SPLIT_FILE:
                return
        
        # 如果都没找到，打印警告
        if not self.TRAIN_SPLIT_FILE or not self.TEST_SPLIT_FILE:
            print(f"警告: 未找到训练/测试集划分文件")
            print(f"请检查数据集路径: {self.DATASET_ROOT}")
            print(f"预期路径: {split_dir_60}")
            print(f"或: {cv_dir}")
