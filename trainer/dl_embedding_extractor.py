import os
import sys
import torch
import numpy as np
import nibabel as nib
from monai.transforms import Compose, Resize

sys.path.append('/home/psw/AS_Radiomics')
from DL_Classification.dl_cls_model import CustomModel, nnUNetClassificationModel
from DL_Classification.dl_cls_dataset import CTNormalization, load_nifti_image


class DLEmbeddingExtractor:
    """Deep Learning 모델에서 embedding feature를 추출하는 클래스"""
    
    def __init__(self, model_path, model_type='custom', nnunet_config=None, img_size=None, device=None):
        self.model_path = model_path
        self.model_type = model_type
        self.nnunet_config = nnunet_config
        self.img_size = img_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.full_model = None
        self.embedding_dim = None
        
        # CT 정규화 설정
        self.ct_normalization = CTNormalization(
            mean_intensity=363.5522766113281,
            std_intensity=249.69992065429688,
            lower_bound=130.0,
            upper_bound=1298.0
        )
        
        # 전처리 파이프라인 설정
        self.transform = Compose([
            self.ct_normalization,
            Resize(self.img_size, mode='trilinear')
        ])
        
        self._load_model()
    
    def _load_image(self, image_path):
        """NIfTI 이미지 로딩 및 전처리 - dl_cls_dataset의 공통 함수 재사용"""
        try:
            # 공통 이미지 로딩 함수 사용
            img_tensor = load_nifti_image(image_path)
            
            # 전처리 적용
            if self.transform is not None:
                img_tensor = self.transform(img_tensor)
            
            return img_tensor
            
        except Exception as e:
            raise RuntimeError(f"이미지 로딩 실패: {e}")
    
    def _load_model(self):
        """모델 로드 및 키 매핑 처리"""
        print(f"  DL 모델 로딩 중: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            print(f"    오류: 모델 파일이 존재하지 않음: {self.model_path}")
            return
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # 체크포인트에서 num_classes 추출
            num_classes = self._extract_num_classes_from_checkpoint(checkpoint)
            
            # 모델 타입에 따라 모델 생성
            if self.model_type == 'custom':
                self.full_model = CustomModel(num_classes=num_classes)
            elif self.model_type == 'nnunet':
                if self.nnunet_config is None:
                    raise ValueError("nnUNet 모델에는 nnunet_config가 필요합니다.")
                
                self.full_model = nnUNetClassificationModel(num_classes=num_classes, pretrained_encoder_path=self.nnunet_config)
            else:
                raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
            
            # 체크포인트에서 가중치 추출 및 키 매핑
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            mapped_state_dict = self._map_state_dict_keys(state_dict)
            
            # 모델 가중치 로드
            self.full_model.load_state_dict(mapped_state_dict, strict=True)
            
            # embedding 차원 계산 및 모델 설정
            self._calculate_embedding_dimension()
            self.full_model.to(self.device)
            self.full_model.eval()
            
            print(f"    DL 모델 로딩 완료 (Device: {self.device}, Classes: {num_classes}, Embedding Dim: {self.embedding_dim})")
            
        except Exception as e:
            print(f"    ⚠️ 오류: DL 모델 로딩 실패 - {e}")
            self.full_model = None
    
    def _extract_num_classes_from_checkpoint(self, checkpoint):
        """체크포인트에서 classifier의 구조를 분석하여 num_classes 추출"""
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Custom 모델의 경우
        if self.model_type == 'custom':
            # classifier.weight 또는 fc.weight에서 out_features 추출
            classifier_keys = [key for key in state_dict.keys() if 'classifier.weight' in key or 'fc.weight' in key]
            if classifier_keys:
                classifier_weight = state_dict[classifier_keys[0]]
                num_classes = classifier_weight.shape[0]  # out_features
                print(f"    Custom 모델에서 추출된 num_classes: {num_classes}")
                return num_classes
            
            # net.fc.weight 형태의 키도 확인
            net_fc_keys = [key for key in state_dict.keys() if 'net.fc.weight' in key]
            if net_fc_keys:
                fc_weight = state_dict[net_fc_keys[0]]
                num_classes = fc_weight.shape[0]
                print(f"    Custom 모델에서 추출된 num_classes (net.fc): {num_classes}")
                return num_classes
        
        # nnUNet 모델의 경우
        elif self.model_type == 'nnunet':
            classifier_keys = [key for key in state_dict.keys() if 'classifier' in key and 'weight' in key]
            if classifier_keys:
                # 가장 마지막 classifier layer 찾기 (multi-layer classifier의 경우)
                last_classifier_key = None
                for key in sorted(classifier_keys):
                    if 'weight' in key:
                        last_classifier_key = key
                
                if last_classifier_key:
                    classifier_weight = state_dict[last_classifier_key]
                    num_classes = classifier_weight.shape[0]
                    print(f"    nnUNet 모델에서 추출된 num_classes: {num_classes}")
                    return num_classes

    def _map_state_dict_keys(self, state_dict):
        """체크포인트의 키를 현재 모델 구조에 맞게 매핑"""
        mapped_state_dict = {}
        
        if self.model_type == 'custom':
            for key, value in state_dict.items():
                if key.startswith('net.'):
                    new_key = key.replace('net.', '')
                    if new_key.startswith('fc.'):
                        new_key = new_key.replace('fc.', 'classifier.')
                    else:
                        new_key = f'backbone.{new_key}'
                    mapped_state_dict[new_key] = value
                elif key.startswith(('classifier.', 'backbone.')):
                    mapped_state_dict[key] = value
                else:
                    mapped_state_dict[key] = value
        elif self.model_type == 'nnunet':
            for key, value in state_dict.items():
                mapped_state_dict[key] = value
        
        return mapped_state_dict
    
    def _calculate_embedding_dimension(self):
        """Embedding 차원 계산"""
        if self.full_model is None:
            return
        
        try:
            if self.model_type == 'custom':
                if hasattr(self.full_model, 'classifier') and hasattr(self.full_model.classifier, 'in_features'):
                    self.embedding_dim = self.full_model.classifier.in_features
                elif hasattr(self.full_model, 'in_features'):
                    self.embedding_dim = self.full_model.in_features
                else:
                    self.embedding_dim = 2048
            elif self.model_type == 'nnunet':
                if hasattr(self.full_model, 'classifier') and hasattr(self.full_model.classifier, 'in_features'):
                    self.embedding_dim = self.full_model.classifier.in_features
        except Exception:
            self.embedding_dim = 2048 if self.model_type == 'custom' else None
    
    def extract_features_for_case(self, image_path, case_id):
        """단일 케이스에 대한 DL embedding 특징 추출"""
        try:
            # 이미지 로딩 및 전처리
            img_tensor = self._load_image(image_path)
            
            # 배치 차원 추가 및 device로 이동
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # 모델을 통해 특징 추출
            with torch.no_grad():
                features = self.full_model.encoder(img_tensor) if hasattr(self.full_model, 'encoder') else self.full_model.backbone(img_tensor)
                
                # Global Average Pooling 적용 (필요한 경우)
                if len(features.shape) > 2:
                    spatial_dims = tuple(range(2, features.ndim))
                    features = torch.mean(features, dim=spatial_dims)
                
                # CPU로 이동하고 numpy로 변환
                features = features.cpu().numpy().flatten()
            
            # DL embedding 특징을 딕셔너리로 반환
            dl_features = {}
            for i, feature_value in enumerate(features, 1):
                dl_features[f'dl_embedding_feature_{i}'] = float(feature_value)
            
            # 파일명과 함께 성공 메시지 출력
            image_filename = os.path.basename(image_path)
            print(f"      DL embedding 추출 성공: {image_filename} ({len(features)} dim)")
            return dl_features
            
        except Exception as e:
            print(f"      DL embedding 추출 오류 ({case_id}): {e}")
            return {}
    
    def get_embedding_dimension(self):
        """Embedding 차원 반환"""
        return self.embedding_dim