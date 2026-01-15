"""
使用示例
演示如何使用颜值打分系统
"""

from analyzer import FaceAnalyzer
from scorer import FaceScorer


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===\n")
    
    # 初始化
    analyzer = FaceAnalyzer()
    scorer = FaceScorer()
    
    # 分析图像（请替换为你的图像路径）
    image_path = "data/test_face.jpg"  # 请替换为实际路径
    
    try:
        # 执行分析
        print(f"正在分析图像: {image_path}")
        analysis_result = analyzer.analyze(image_path)
        
        # 计算评分
        print("正在计算评分...")
        scores = scorer.score(analysis_result)
        
        # 输出结果
        print("\n" + scores['report'])
        
        # 查看详细评分
        print("\n=== 详细评分数据 ===")
        print(f"综合评分: {scores['overall']['overall_score']}/5.0")
        print(f"骨相评分: {scores['overall']['bone_score']}/5.0")
        print(f"三庭评分: {scores['three_regions']['three_regions_score']}/5.0")
        print(f"五眼评分: {scores['five_eyes']['five_eyes_score']}/5.0")
        print(f"对称性评分: {scores['symmetry']['symmetry_score']}/5.0")
        
    except FileNotFoundError:
        print(f"错误：找不到图像文件: {image_path}")
        print("请将图像路径替换为实际存在的文件路径")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


def example_api_usage():
    """API 使用示例"""
    print("\n=== API 使用示例 ===\n")
    
    analyzer = FaceAnalyzer()
    scorer = FaceScorer()
    
    image_path = "data/test_face.jpg"  # 请替换为实际路径
    
    try:
        # 分析
        analysis = analyzer.analyze(image_path)
        
        # 只获取分析结果（不评分）
        print("分析结果:")
        print(f"  检测到人脸: {analysis['face_detected']}")
        print(f"  关键点数量: {len(analysis['landmarks'])}")
        print(f"  三庭比例: {analysis['three_regions']}")
        print(f"  五眼比例: {analysis['five_eyes']}")
        print(f"  对称性: {analysis['symmetry']}")
        
        # 评分
        scores = scorer.score(analysis)
        print(f"\n综合评分: {scores['overall']['overall_score']}/5.0")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    print("颜值打分系统 - 使用示例\n")
    print("注意：请将示例中的图像路径替换为实际存在的文件路径\n")
    
    # 运行基本示例
    example_basic_usage()
    
    # 运行API示例
    # example_api_usage()
