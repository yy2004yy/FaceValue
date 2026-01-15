"""
SCUT-FBP5500数据集加载器
根据GitHub文档实现数据加载和预处理
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional, List
import json


class SCUTFBP5500Dataset(Dataset):
    """
    SCUT-FBP5500数据集类
    数据集包含5500张人脸照片，每张图片有1-5分的颜值评分
    """
    
    def __init__(
        self,
        root_dir: str,
        split_file: Optional[str] = None,
        mode: str = 'train',
        model_type: str = 'alexnet',
        transform: Optional[transforms.Compose] = None
    ):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录
            split_file: 训练/测试集划分文件（可选）
            mode: 'train' 或 'test'
            model_type: 'alexnet', 'resnet' 或 'resnext'
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.mode = mode
        self.model_type = model_type.lower()
        
        # 默认变换
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # 加载数据列表
        self.data_list = self._load_data_list(split_file)
        
    def _get_default_transform(self) -> transforms.Compose:
        """
        获取默认的图像变换
        根据模型类型返回不同的变换
        """
        # 基础变换：resize到256x256
        base_transforms = [
            transforms.Resize((256, 256)),
        ]
        
        if self.mode == 'train':
            # 训练时添加数据增强
            if self.model_type == 'alexnet':
                # AlexNet: 随机裁剪到227x227
                base_transforms.append(transforms.RandomCrop(227))
            else:
                # ResNet/ResNeXt: 随机裁剪到224x224
                base_transforms.append(transforms.RandomCrop(224))
            
            base_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # 测试时中心裁剪
            if self.model_type == 'alexnet':
                base_transforms.append(transforms.CenterCrop(227))
            else:
                base_transforms.append(transforms.CenterCrop(224))
            
            base_transforms.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        return transforms.Compose(base_transforms)
    
    def _load_data_list(self, split_file: Optional[str]) -> List[Tuple[str, float]]:
        """
        加载数据列表
        
        Args:
            split_file: 划分文件路径
            
        Returns:
            数据列表 [(图像路径, 评分), ...]
        """
        data_list = []
        
        # 如果提供了划分文件
        if split_file and os.path.exists(split_file):
            data_list = self._load_from_annotation_file(split_file)
        else:
            # 如果没有划分文件，尝试自动查找
            # 优先级1: split_of_60%training and 40%testing (实际文件夹名，有空格)
            split_dir_60 = os.path.join(self.root_dir, 'train_test_files', 
                                       'split_of_60%training and 40%testing')
            # 优先级2: split_of_60%_training_and_40%_testing (下划线版本)
            split_dir_60_alt = os.path.join(self.root_dir, 'train_test_files',
                                           'split_of_60%_training_and_40%_testing')
            
            annotation_file = None
            
            # 尝试查找60%划分文件
            for split_dir in [split_dir_60, split_dir_60_alt]:
                if os.path.exists(split_dir):
                    file_name = 'train.txt' if self.mode == 'train' else 'test.txt'
                    annotation_file = os.path.join(split_dir, file_name)
                    if os.path.exists(annotation_file):
                        break
            
            # 如果没找到，尝试5折交叉验证的文件
            if annotation_file is None or not os.path.exists(annotation_file):
                cv_dir = os.path.join(self.root_dir, 'train_test_files',
                                     '5_folders_cross_validations_files',
                                     'cross_validation_1')
                if os.path.exists(cv_dir):
                    file_name = f'train_1.txt' if self.mode == 'train' else f'test_1.txt'
                    annotation_file = os.path.join(cv_dir, file_name)
            
            if annotation_file and os.path.exists(annotation_file):
                data_list = self._load_from_annotation_file(annotation_file)
            else:
                # 尝试加载评分文件
                score_file = os.path.join(self.root_dir, 'All_Ratings.xlsx')
                if not os.path.exists(score_file):
                    # 尝试在SCUT-FBP5500_v2.1目录下查找
                    score_file = os.path.join(self.root_dir, '..', 'SCUT-FBP5500_v2.1', 'All_Ratings.xlsx')
                
                if os.path.exists(score_file):
                    data_list = self._load_from_score_file(score_file)
                else:
                    # 最后尝试扫描图像目录
                    print(f"警告: 未找到划分文件，尝试扫描图像目录...")
                    data_list = self._scan_image_directory()
        
        return data_list
    
    def _load_from_annotation_file(self, annotation_file: str) -> List[Tuple[str, float]]:
        """从标注文件加载数据"""
        data_list = []
        images_dir = os.path.join(self.root_dir, 'Images')
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 支持空格或制表符分隔
                    parts = line.split('\t') if '\t' in line else line.split()
                    if len(parts) >= 2:
                        filename = parts[0]  # 只包含文件名，如 CM148.jpg
                        score = float(parts[1])
                        
                        # 图像在Images文件夹下
                        full_path = os.path.join(images_dir, filename)
                        
                        # 如果Images目录下找不到，尝试直接在root_dir下找
                        if not os.path.exists(full_path):
                            full_path = os.path.join(self.root_dir, filename)
                        
                        if os.path.exists(full_path):
                            data_list.append((full_path, score))
                        else:
                            # 记录缺失的文件（用于调试）
                            if len(data_list) < 10:  # 只记录前10个缺失的文件
                                print(f"警告: 图像文件不存在: {full_path}")
        
        if len(data_list) == 0:
            print(f"错误: 未能从 {annotation_file} 加载任何数据")
            print(f"请检查数据集路径: {self.root_dir}")
            print(f"图像目录: {images_dir}")
        
        return data_list
    
    def _load_from_score_file(self, score_file: str) -> List[Tuple[str, float]]:
        """从Excel评分文件加载数据"""
        data_list = []
        images_dir = os.path.join(self.root_dir, 'Images')
        
        try:
            # 读取Excel文件
            df = pd.read_excel(score_file)
            # 假设第一列是文件名，最后一列是平均评分（通常是Rating或平均分）
            # 需要找到包含评分的列
            score_col = None
            filename_col = 0  # 第一列通常是文件名
            
            # 尝试找到评分列（可能名称不同）
            for col in df.columns:
                if 'rating' in str(col).lower() or 'score' in str(col).lower() or 'mean' in str(col).lower():
                    score_col = col
                    break
            
            # 如果没找到，使用第二列
            if score_col is None and len(df.columns) > 1:
                score_col = df.columns[1]
            
            if score_col is None:
                print(f"警告: 无法在 {score_file} 中找到评分列")
                return data_list
            
            for _, row in df.iterrows():
                filename = str(row.iloc[filename_col])
                try:
                    score = float(row[score_col])
                    
                    # 图像在Images文件夹下
                    img_path = os.path.join(images_dir, filename)
                    if not os.path.exists(img_path):
                        img_path = os.path.join(self.root_dir, filename)
                    
                    if os.path.exists(img_path):
                        data_list.append((img_path, score))
                except (ValueError, KeyError) as e:
                    continue  # 跳过无效行
                    
        except Exception as e:
            print(f"加载评分文件时出错: {e}")
            import traceback
            traceback.print_exc()
        
        return data_list
    
    def _scan_image_directory(self) -> List[Tuple[str, float]]:
        """扫描图像目录（最后手段，需要配合All_labels.txt或All_Ratings.xlsx）"""
        data_list = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # 尝试从All_labels.txt加载评分
        labels_file = os.path.join(self.root_dir, 'train_test_files', 'All_labels.txt')
        score_dict = {}
        
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split('\t') if '\t' in line else line.split()
                            if len(parts) >= 2:
                                filename = parts[0]
                                score = float(parts[1])
                                score_dict[filename] = score
            except Exception as e:
                print(f"读取All_labels.txt时出错: {e}")
        
        # 扫描Images目录
        images_dir = os.path.join(self.root_dir, 'Images')
        if not os.path.exists(images_dir):
            images_dir = self.root_dir
        
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    img_path = os.path.join(root, file)
                    filename = os.path.basename(file)
                    
                    # 从score_dict获取评分，如果没有则使用默认值（不推荐）
                    if filename in score_dict:
                        score = score_dict[filename]
                        data_list.append((img_path, score))
                    else:
                        # 如果没有评分信息，跳过（避免使用无意义的默认值）
                        if len(data_list) < 10:
                            print(f"警告: 未找到 {filename} 的评分信息，跳过")
        
        return data_list
    
    def _extract_score_from_filename(self, filename: str) -> Optional[float]:
        """从文件名提取评分（如果需要）"""
        # 这里可以根据实际文件名格式提取
        # SCUT-FBP5500数据集的评分通常在单独的标注文件中
        return None
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据项
        
        Returns:
            (图像tensor, 评分tensor)
        """
        img_path, score = self.data_list[idx]
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"读取图像失败 {img_path}: {e}")
            # 返回黑色图像作为fallback
            image = Image.new('RGB', (256, 256), color='black')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 转换为tensor
        score_tensor = torch.tensor(score, dtype=torch.float32)
        
        return image, score_tensor


def get_data_loader(
    root_dir: str,
    split_file: Optional[str] = None,
    mode: str = 'train',
    model_type: str = 'alexnet',
    batch_size: int = 16,
    shuffle: Optional[bool] = None,
    num_workers: int = 4
) -> DataLoader:
    """
    获取数据加载器
    
    Args:
        root_dir: 数据集根目录
        split_file: 划分文件路径
        mode: 'train' 或 'test'
        model_type: 'alexnet', 'resnet' 或 'resnext'
        batch_size: 批次大小
        shuffle: 是否打乱（默认：训练时True，测试时False）
        num_workers: 数据加载线程数
        
    Returns:
        DataLoader对象
    """
    if shuffle is None:
        shuffle = (mode == 'train')
    
    dataset = SCUTFBP5500Dataset(
        root_dir=root_dir,
        split_file=split_file,
        mode=mode,
        model_type=model_type
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader
