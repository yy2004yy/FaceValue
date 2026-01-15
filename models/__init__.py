"""
模型定义模块
包含AlexNet、ResNet-18、ResNeXt-50等模型
"""

from .neural_models import (
    AlexNetBeauty,
    ResNet18Beauty,
    ResNeXt50Beauty,
    get_model
)

__all__ = [
    'AlexNetBeauty',
    'ResNet18Beauty',
    'ResNeXt50Beauty',
    'get_model'
]
