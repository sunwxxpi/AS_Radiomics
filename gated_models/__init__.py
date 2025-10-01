"""Gated Fusion Models for Radiomics + DL Feature Fusion

이 패키지는 Radiomics와 DL embedding features를 Gated Mechanism으로 융합하는 모델과 학습/평가 파이프라인을 제공합니다.

Modules:
    - gated_model: Gated Fusion Layer와 Classifier 모델
    - gated_trainer: Gated Fusion 모델 학습
    - gated_feature_extractor: 학습된 모델로 fused features 추출
    - gated_pipeline: Gated Fusion 전체 분석 파이프라인
"""

from .gated_model import GatedFusionLayer, GatedFusionClassifier
from .gated_trainer import GatedFusionTrainer
from .gated_feature_extractor import GatedFeatureExtractor
from .gated_pipeline import run_gated_fusion_analysis

__all__ = [
    'GatedFusionLayer',
    'GatedFusionClassifier',
    'GatedFusionTrainer',
    'GatedFeatureExtractor',
    'run_gated_fusion_analysis'
]