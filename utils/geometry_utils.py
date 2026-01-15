"""
几何计算工具函数
"""

import numpy as np
from typing import List, Tuple, Dict


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    计算两点间的欧氏距离
    
    Args:
        p1: 点1坐标
        p2: 点2坐标
        
    Returns:
        距离
    """
    return np.linalg.norm(p1 - p2)


def calculate_ratios(landmarks: np.ndarray, indices: List[int]) -> Dict[str, float]:
    """
    计算关键点之间的比例
    
    Args:
        landmarks: 关键点坐标
        indices: 关键点索引列表
        
    Returns:
        比例字典
    """
    points = landmarks[indices]
    # 实现具体的比例计算逻辑
    pass


def calculate_symmetry(left_points: np.ndarray, right_points: np.ndarray, center_x: float) -> float:
    """
    计算左右对称性
    
    Args:
        left_points: 左侧点集
        right_points: 右侧点集
        center_x: 中心线x坐标
        
    Returns:
        对称性分数 (0-1)
    """
    if len(left_points) != len(right_points):
        return 0.0
    
    # 计算左侧点到中心线的距离
    left_distances = np.abs(left_points[:, 0] - center_x)
    # 计算右侧点到中心线的距离（镜像）
    right_distances = np.abs(right_points[:, 0] - center_x)
    
    # 计算差异
    differences = np.abs(left_distances - right_distances)
    avg_diff = np.mean(differences)
    
    # 归一化到0-1（差异越小，对称性越高）
    symmetry_score = max(0, 1 - avg_diff / 50.0)
    
    return symmetry_score
