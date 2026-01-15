"""
图像处理工具函数
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    加载图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        BGR格式的图像数组
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return image


def preprocess_image(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    预处理图像
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)，如果为None则不缩放
        
    Returns:
        预处理后的图像
    """
    if target_size is not None:
        image = cv2.resize(image, target_size)
    return image


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    保存图像
    
    Args:
        image: 图像数组
        output_path: 输出路径
        
    Returns:
        是否保存成功
    """
    return cv2.imwrite(output_path, image)
