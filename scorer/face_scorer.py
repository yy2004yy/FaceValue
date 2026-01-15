"""
人脸评分器
基于几何特征进行多维度评分
"""

import numpy as np
from typing import Dict, List


class FaceScorer:
    """人脸评分器类"""
    
    def __init__(self):
        """初始化评分器"""
        # 理想比例标准（基于黄金比例和美学标准）
        self.ideal_three_regions = {
            'upper': 0.33,   # 上庭理想比例
            'middle': 0.34,  # 中庭理想比例
            'lower': 0.33    # 下庭理想比例
        }
        self.ideal_five_eyes_ratio = 1.0  # 五眼理想比例
        
    def score_three_regions(self, three_regions: Dict[str, float]) -> Dict[str, float]:
        """
        三庭评分
        
        Args:
            three_regions: 三庭比例字典
            
        Returns:
            三庭评分字典（0-5分）
        """
        upper = three_regions['upper_ratio']
        middle = three_regions['middle_ratio']
        lower = three_regions['lower_ratio']
        
        # 计算与理想比例的差异
        upper_diff = abs(upper - self.ideal_three_regions['upper'])
        middle_diff = abs(middle - self.ideal_three_regions['middle'])
        lower_diff = abs(lower - self.ideal_three_regions['lower'])
        
        # 差异越小，分数越高（0-5分制）
        # 差异小于0.05为5分，差异大于0.15为0分
        def diff_to_score(diff):
            if diff < 0.02:
                return 5.0
            elif diff < 0.05:
                return 4.5 - (diff - 0.02) / 0.03 * 0.5
            elif diff < 0.10:
                return 4.0 - (diff - 0.05) / 0.05 * 1.0
            elif diff < 0.15:
                return 3.0 - (diff - 0.10) / 0.05 * 1.5
            else:
                return max(0, 1.5 - (diff - 0.15) / 0.10 * 1.5)
        
        upper_score = diff_to_score(upper_diff)
        middle_score = diff_to_score(middle_diff)
        lower_score = diff_to_score(lower_diff)
        
        # 综合三庭分数（加权平均）
        avg_score = (upper_score + middle_score + lower_score) / 3
        
        return {
            'upper_score': round(upper_score, 2),
            'middle_score': round(middle_score, 2),
            'lower_score': round(lower_score, 2),
            'three_regions_score': round(avg_score, 2),
            'upper_ratio': round(upper, 3),
            'middle_ratio': round(middle, 3),
            'lower_ratio': round(lower, 3)
        }
    
    def score_five_eyes(self, five_eyes: Dict[str, float]) -> Dict[str, float]:
        """
        五眼评分
        
        Args:
            five_eyes: 五眼比例字典
            
        Returns:
            五眼评分字典（0-5分）
        """
        ratio = five_eyes['five_eyes_ratio']
        
        # 理想比例是1.0（脸宽 = 5个眼睛宽度）
        diff = abs(ratio - self.ideal_five_eyes_ratio)
        
        # 差异小于0.05为5分
        if diff < 0.05:
            score = 5.0
        elif diff < 0.10:
            score = 4.5 - (diff - 0.05) / 0.05 * 0.5
        elif diff < 0.15:
            score = 4.0 - (diff - 0.10) / 0.05 * 1.0
        elif diff < 0.20:
            score = 3.0 - (diff - 0.15) / 0.05 * 1.5
        else:
            score = max(0, 1.5 - (diff - 0.20) / 0.15 * 1.5)
        
        return {
            'five_eyes_score': round(score, 2),
            'five_eyes_ratio': round(ratio, 3),
            'face_width': round(five_eyes['face_width'], 2),
            'eye_width': round(five_eyes['eye_width'], 2)
        }
    
    def score_symmetry(self, symmetry: Dict[str, float]) -> Dict[str, float]:
        """
        对称性评分
        
        Args:
            symmetry: 对称性字典
            
        Returns:
            对称性评分字典（0-5分）
        """
        symmetry_score = symmetry['symmetry_score']
        
        # 将0-1的对称性分数转换为0-5分
        score = symmetry_score * 5
        
        return {
            'symmetry_score': round(score, 2),
            'symmetry_score_raw': round(symmetry_score, 3),
            'avg_symmetry_error': round(symmetry['avg_symmetry_error'], 2)
        }
    
    def calculate_overall_score(self, scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        计算综合评分
        
        Args:
            scores: 各维度评分字典
            
        Returns:
            综合评分字典
        """
        # 各维度权重
        weights = {
            'three_regions': 0.35,  # 三庭权重
            'five_eyes': 0.30,      # 五眼权重
            'symmetry': 0.35        # 对称性权重
        }
        
        # 计算加权平均
        three_regions_score = scores['three_regions']['three_regions_score']
        five_eyes_score = scores['five_eyes']['five_eyes_score']
        symmetry_score = scores['symmetry']['symmetry_score']
        
        overall_score = (
            three_regions_score * weights['three_regions'] +
            five_eyes_score * weights['five_eyes'] +
            symmetry_score * weights['symmetry']
        )
        
        # 计算骨相分（基于几何特征）
        bone_score = (three_regions_score + five_eyes_score + symmetry_score) / 3
        
        return {
            'overall_score': round(overall_score, 2),
            'bone_score': round(bone_score, 2),
            'weights': weights
        }
    
    def generate_report(self, analysis_result: Dict, scores: Dict) -> str:
        """
        生成评分报告文本
        
        Args:
            analysis_result: 分析结果
            scores: 评分结果
            
        Returns:
            报告文本
        """
        overall = scores['overall']
        three_regions = scores['three_regions']
        five_eyes = scores['five_eyes']
        symmetry = scores['symmetry']
        
        report = f"""
=== 颜值评分报告 ===

【综合评分】{overall['overall_score']:.2f}/5.0 分
【骨相评分】{overall['bone_score']:.2f}/5.0 分

【详细评分】

1. 三庭比例评分：{three_regions['three_regions_score']:.2f}/5.0
   - 上庭：{three_regions['upper_score']:.2f}分 (比例: {three_regions['upper_ratio']:.3f})
   - 中庭：{three_regions['middle_score']:.2f}分 (比例: {three_regions['middle_ratio']:.3f})
   - 下庭：{three_regions['lower_score']:.2f}分 (比例: {three_regions['lower_ratio']:.3f})
   
2. 五眼比例评分：{five_eyes['five_eyes_score']:.2f}/5.0
   - 五眼比例：{five_eyes['five_eyes_ratio']:.3f} (理想值: 1.0)
   - 脸宽：{five_eyes['face_width']:.1f}px
   - 平均眼宽：{five_eyes['eye_width']:.1f}px

3. 对称性评分：{symmetry['symmetry_score']:.2f}/5.0
   - 对称性指数：{symmetry['symmetry_score_raw']:.3f}
   - 平均对称误差：{symmetry['avg_symmetry_error']:.2f}px

【评分说明】
- 三庭比例：上庭（发际线到眉间）、中庭（眉间到鼻底）、下庭（鼻底到下巴）的理想比例约为1:1:1
- 五眼比例：脸宽应该等于五个眼睛的宽度
- 对称性：基于左右眼、眉毛、嘴角的对称程度评估

【权重分配】
- 三庭比例：35%
- 五眼比例：30%
- 对称性：35%
"""
        return report
    
    def score(self, analysis_result: Dict) -> Dict:
        """
        完整评分流程
        
        Args:
            analysis_result: 分析结果字典
            
        Returns:
            评分结果字典
        """
        # 各维度评分
        three_regions_score = self.score_three_regions(analysis_result['three_regions'])
        five_eyes_score = self.score_five_eyes(analysis_result['five_eyes'])
        symmetry_score = self.score_symmetry(analysis_result['symmetry'])
        
        scores = {
            'three_regions': three_regions_score,
            'five_eyes': five_eyes_score,
            'symmetry': symmetry_score
        }
        
        # 综合评分
        overall_score = self.calculate_overall_score(scores)
        scores['overall'] = overall_score
        
        # 生成报告
        report = self.generate_report(analysis_result, scores)
        scores['report'] = report
        
        return scores
