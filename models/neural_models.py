"""
神经网络模型定义
基于SCUT-FBP5500论文中的模型架构
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class AlexNetBeauty(nn.Module):
    """
    AlexNet用于颜值预测
    修改最后的全连接层输出单个回归值（1-5分）
    """
    
    def __init__(self, pretrained: bool = True):
        super(AlexNetBeauty, self).__init__()
        
        # 加载预训练的AlexNet
        alexnet = models.alexnet(pretrained=pretrained)
        
        # 使用AlexNet的特征提取部分
        self.features = alexnet.features
        
        # 修改分类器部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1)  # 输出单个回归值
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x.squeeze(1)  # 移除最后一个维度


class ResNet18Beauty(nn.Module):
    """
    ResNet-18用于颜值预测
    修改最后的全连接层输出单个回归值（1-5分）
    """
    
    def __init__(self, pretrained: bool = True):
        super(ResNet18Beauty, self).__init__()
        
        # 加载预训练的ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        
        # 使用ResNet的特征提取部分
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        
        # 修改最后的全连接层
        self.fc = nn.Linear(resnet.fc.in_features, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


class ResNeXt50Beauty(nn.Module):
    """
    ResNeXt-50用于颜值预测
    修改最后的全连接层输出单个回归值（1-5分）
    """
    
    def __init__(self, pretrained: bool = True):
        super(ResNeXt50Beauty, self).__init__()
        
        # 加载预训练的ResNeXt-50
        resnext = models.resnext50_32x4d(pretrained=pretrained)
        
        # 使用ResNeXt的特征提取部分
        self.features = nn.Sequential(
            resnext.conv1,
            resnext.bn1,
            resnext.relu,
            resnext.maxpool,
            resnext.layer1,
            resnext.layer2,
            resnext.layer3,
            resnext.layer4,
            resnext.avgpool
        )
        
        # 修改最后的全连接层
        self.fc = nn.Linear(resnext.fc.in_features, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    获取模型实例
    
    Args:
        model_name: 模型名称 ('alexnet', 'resnet18', 'resnext50')
        pretrained: 是否使用预训练权重
        
    Returns:
        模型实例
    """
    model_name = model_name.lower()
    
    if model_name == 'alexnet':
        return AlexNetBeauty(pretrained=pretrained)
    elif model_name == 'resnet18' or model_name == 'resnet-18':
        return ResNet18Beauty(pretrained=pretrained)
    elif model_name == 'resnext50' or model_name == 'resnext-50':
        return ResNeXt50Beauty(pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型: {model_name}。支持: 'alexnet', 'resnet18', 'resnext50'")
