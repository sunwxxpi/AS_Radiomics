import os
import sys
import torch
import numpy as np
import nibabel as nib
from monai.transforms import Compose, Resize

sys.path.append('/home/psw/AS_Radiomics')
from DL_Classification.dl_cls_model import CustomModel, nnUNetClassificationModel


class CTNormalization:
    """CT 이미지 정규화를 위한 클래스"""
    
    def __init__(self, mean_intensity=None, std_intensity=None, lower_bound=None, upper_bound=None, target_dtype=np.float32):
        self.mean_intensity = mean_intensity
        self.std_intensity = std_intensity
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.target_dtype = target_dtype
    
    def run(self, image: np.ndarray) -> np.ndarray:
        assert all(v is not None for v in [self.mean_intensity, self.std_intensity, self.lower_bound, self.upper_bound]), \
            "CTNormalization requires all intensity parameters: mean_intensity, std_intensity, lower_bound, upper_bound"
        
        image = image.astype(self.target_dtype, copy=False)
        np.clip(image, self.lower_bound, self.upper_bound, out=image)
        image -= self.mean_intensity
        image /= max(self.std_intensity, 1e-8)
        return image
    
    def __call__(self, data):
        """MONAI transform 호환을 위한 호출 메서드"""
        return torch.from_numpy(self.run(data.numpy()))


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
        """NIfTI 이미지 로딩 및 전처리"""
        try:
            # NIfTI 파일 로드
            nii_img = nib.load(image_path)
            img_data = nii_img.get_fdata().astype(np.float32)
            
            # NIfTI 기본 순서 (W, H, D)를 PyTorch 표준 (D, H, W)로 transpose
            img_data = np.transpose(img_data, (2, 1, 0))  # (W, H, D) -> (D, H, W)
            
            # 채널 차원 추가 (D, H, W) -> (1, D, H, W)
            if len(img_data.shape) == 3:
                img_data = img_data[np.newaxis, ...]  # 첫 번째 차원에 채널 추가
            
            # torch tensor로 변환
            img_tensor = torch.from_numpy(img_data)
            
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
            
            # 모델 타입에 따라 모델 생성
            if self.model_type == 'custom':
                class_num = checkpoint.get('class_num', 3)
                self.full_model = CustomModel(class_num=class_num)
            elif self.model_type == 'nnunet':
                if self.nnunet_config is None:
                    raise ValueError("nnUNet 모델에는 nnunet_config가 필요합니다.")
                class_num = checkpoint.get('class_num', 3)
                self.full_model = nnUNetClassificationModel(
                    class_num=class_num,
                    pretrained_encoder_path=self.nnunet_config
                )
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
            
            print(f"    DL 모델 로딩 완료 (Device: {self.device}, Embedding Dim: {self.embedding_dim})")
            
        except Exception as e:
            print(f"    오류: DL 모델 로딩 실패 - {e}")
            self.full_model = None
    
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