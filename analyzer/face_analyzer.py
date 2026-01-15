"""
人脸分析器
使用MediaPipe进行人脸检测和关键点提取
计算三庭五眼、对称性等几何特征
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import os


class FaceAnalyzer:
    """人脸分析器类"""
    
    def __init__(self):
        """初始化MediaPipe人脸检测和关键点模型"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 初始化人脸检测
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0:近距离, 1:远距离
            min_detection_confidence=0.5
        )
        
        # 初始化人脸网格（468个关键点）
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # 使用468个关键点
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测人脸
        
        Args:
            image: 输入的BGR图像
            
        Returns:
            包含人脸位置信息的字典，如果没有检测到则返回None
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = image.shape[:2]
            
            return {
                'x': int(bbox.xmin * w),
                'y': int(bbox.ymin * h),
                'width': int(bbox.width * w),
                'height': int(bbox.height * h),
                'confidence': detection.score[0]
            }
        return None
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取人脸关键点
        
        Args:
            image: 输入的BGR图像
            
        Returns:
            468x2的关键点坐标数组，如果没有检测到则返回None
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = landmark.x * w
                y = landmark.y * h
                landmarks.append([x, y])
            
            return np.array(landmarks)
        return None
    
    def get_key_landmark_indices(self) -> Dict[str, List[int]]:
        """
        获取关键特征点的索引
        MediaPipe 468点关键索引
        """
        return {
            # 面部轮廓 (下巴到发际线)
            'jawline': list(range(0, 17)) + [234, 227, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
            
            # 左眉毛
            'left_eyebrow': [107, 55, 65, 52, 53, 46],
            # 右眉毛
            'right_eyebrow': [336, 296, 334, 293, 300, 276],
            
            # 左眼
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            # 右眼
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            
            # 鼻子
            'nose_tip': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 290, 305, 290, 305],
            'nose_bridge': [6, 51, 48, 115, 131, 134],
            
            # 嘴巴
            'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'mouth_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324],
            
            # 关键点（用于计算比例）
            'forehead_center': [10],  # 额头中心
            'chin_center': [152],  # 下巴中心
            'nose_tip_center': [1],  # 鼻尖
            'left_eye_center': [33],  # 左眼中心
            'right_eye_center': [263],  # 右眼中心
            'left_mouth_corner': [61],
            'right_mouth_corner': [291],
        }
    
    def calculate_three_regions(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        计算三庭比例
        上庭：发际线到眉间
        中庭：眉间到鼻底
        下庭：鼻底到下巴
        
        Args:
            landmarks: 关键点坐标
            
        Returns:
            三庭比例字典
        """
        indices = self.get_key_landmark_indices()
        
        # 使用关键点估算
        # 上庭：额头中心到眉毛中心
        forehead = landmarks[indices['forehead_center'][0]]
        
        # 眉毛中心（左右眉毛中点）
        left_eyebrow = landmarks[indices['left_eyebrow']]
        right_eyebrow = landmarks[indices['right_eyebrow']]
        eyebrow_center = (np.mean(left_eyebrow, axis=0) + np.mean(right_eyebrow, axis=0)) / 2
        
        # 鼻底（使用鼻子关键点）
        nose_bridge = landmarks[indices['nose_bridge']]
        nose_bottom = np.mean(nose_bridge[-2:], axis=0)
        
        # 下巴
        chin = landmarks[indices['chin_center'][0]]
        
        # 计算距离
        upper_region = np.linalg.norm(forehead - eyebrow_center)
        middle_region = np.linalg.norm(eyebrow_center - nose_bottom)
        lower_region = np.linalg.norm(nose_bottom - chin)
        
        total = upper_region + middle_region + lower_region
        
        return {
            'upper_ratio': upper_region / total if total > 0 else 0,
            'middle_ratio': middle_region / total if total > 0 else 0,
            'lower_ratio': lower_region / total if total > 0 else 0,
            'upper_region': upper_region,
            'middle_region': middle_region,
            'lower_region': lower_region
        }
    
    def calculate_five_eyes(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        计算五眼比例
        理想比例：脸宽应该等于五个眼睛的宽度
        
        Args:
            landmarks: 关键点坐标
            
        Returns:
            五眼比例字典
        """
        indices = self.get_key_landmark_indices()
        
        # 获取左右眼的关键点
        left_eye = landmarks[indices['left_eye']]
        right_eye = landmarks[indices['right_eye']]
        
        # 计算眼睛宽度（使用外眼角到内眼角）
        left_eye_width = np.max(left_eye[:, 0]) - np.min(left_eye[:, 0])
        right_eye_width = np.max(right_eye[:, 0]) - np.min(right_eye[:, 0])
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        # 计算脸宽（使用下巴两侧关键点）
        jawline = landmarks[indices['jawline']]
        face_width = np.max(jawline[:, 0]) - np.min(jawline[:, 0])
        
        # 计算五眼比例
        five_eyes_width = avg_eye_width * 5
        ratio = face_width / five_eyes_width if five_eyes_width > 0 else 0
        
        return {
            'face_width': face_width,
            'eye_width': avg_eye_width,
            'five_eyes_ratio': ratio,
            'left_eye_width': left_eye_width,
            'right_eye_width': right_eye_width
        }
    
    def calculate_symmetry(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        计算面部对称性
        
        Args:
            landmarks: 关键点坐标
            
        Returns:
            对称性评分字典
        """
        indices = self.get_key_landmark_indices()
        
        # 计算面部中心线（使用鼻子中线和下巴中心）
        nose_bridge = landmarks[indices['nose_bridge']]
        chin = landmarks[indices['chin_center'][0]]
        face_center_x = (np.min(nose_bridge[:, 0]) + np.max(nose_bridge[:, 0])) / 2
        
        # 计算左右对称点的差异
        symmetry_errors = []
        
        # 左右眼对称
        left_eye_center = np.mean(landmarks[indices['left_eye']], axis=0)
        right_eye_center = np.mean(landmarks[indices['right_eye']], axis=0)
        eye_symmetry = abs((left_eye_center[0] - face_center_x) - (face_center_x - right_eye_center[0]))
        symmetry_errors.append(eye_symmetry)
        
        # 左右眉毛对称
        left_eyebrow = np.mean(landmarks[indices['left_eyebrow']], axis=0)
        right_eyebrow = np.mean(landmarks[indices['right_eyebrow']], axis=0)
        eyebrow_symmetry = abs((left_eyebrow[0] - face_center_x) - (face_center_x - right_eyebrow[0]))
        symmetry_errors.append(eyebrow_symmetry)
        
        # 嘴角对称
        left_mouth = landmarks[indices['left_mouth_corner'][0]]
        right_mouth = landmarks[indices['right_mouth_corner'][0]]
        mouth_symmetry = abs((left_mouth[0] - face_center_x) - (face_center_x - right_mouth[0]))
        symmetry_errors.append(mouth_symmetry)
        
        # 计算平均对称误差
        avg_error = np.mean(symmetry_errors)
        
        # 归一化到0-1（对称性分数，越高越好）
        # 使用经验值：误差小于10像素算完美对称
        symmetry_score = max(0, 1 - avg_error / 50.0)
        
        return {
            'symmetry_score': symmetry_score,
            'eye_symmetry_error': eye_symmetry,
            'eyebrow_symmetry_error': eyebrow_symmetry,
            'mouth_symmetry_error': mouth_symmetry,
            'avg_symmetry_error': avg_error
        }
    
    def analyze(self, image_path: str) -> Dict:
        """
        完整分析流程
        
        Args:
            image_path: 图像路径
            
        Returns:
            包含所有分析结果的字典
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 检测人脸
        face_bbox = self.detect_face(image)
        if face_bbox is None:
            raise ValueError("未检测到人脸")
        
        # 提取关键点
        landmarks = self.extract_landmarks(image)
        if landmarks is None:
            raise ValueError("无法提取人脸关键点")
        
        # 计算各种特征
        three_regions = self.calculate_three_regions(landmarks)
        five_eyes = self.calculate_five_eyes(landmarks)
        symmetry = self.calculate_symmetry(landmarks)
        
        return {
            'face_detected': True,
            'face_bbox': face_bbox,
            'landmarks': landmarks.tolist(),  # 转换为列表便于序列化
            'three_regions': three_regions,
            'five_eyes': five_eyes,
            'symmetry': symmetry,
            'image_shape': image.shape
        }
