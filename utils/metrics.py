"""
评估指标计算
计算PC (Pearson Correlation), MAE, RMSE
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        predictions: 预测值数组
        targets: 真实值数组
        
    Returns:
        包含PC, MAE, RMSE的字典
    """
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # 确保长度一致
    min_len = min(len(predictions), len(targets))
    predictions = predictions[:min_len]
    targets = targets[:min_len]
    
    # Pearson Correlation
    pc, _ = pearsonr(predictions, targets)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    return {
        'pc': float(pc),
        'mae': float(mae),
        'rmse': float(rmse)
    }
