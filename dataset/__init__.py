"""
数据集模块
负责SCUT-FBP5500数据集的加载和处理
"""

from .dataset_loader import SCUTFBP5500Dataset, get_data_loader

__all__ = ['SCUTFBP5500Dataset', 'get_data_loader']
