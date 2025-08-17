import os
import sys
import glob
import re
import json
import pandas as pd
import numpy as np
import torch
from glob import glob
from torch.utils.data.dataset import Dataset
from monai.transforms import Compose, RandFlip, RandAffine, RandAdjustContrast, RandGaussianNoise, Resize

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Config
from data.loader import DataLoader as ASDataLoader
from utils.data_splitter import DataSplitter


def load_intensity_properties_from_plans(plans_file_path):
    """nnUNet plans 파일에서 intensity properties를 로드하는 함수"""
    if not os.path.exists(plans_file_path):
        print(f"⚠️ Warning: Plans file not found at {plans_file_path}")
        return None
    
    try:
        with open(plans_file_path, 'r') as f:
            plans_data = json.load(f)
        
        # foreground_intensity_properties_per_channel에서 첫 번째 채널 정보 추출
        intensity_props = plans_data.get('foreground_intensity_properties_per_channel', {})
        
        if '0' in intensity_props:
            channel_0_props = intensity_props['0']
            return {
                'mean_intensity': float(channel_0_props.get('mean', 340.6403503417969)),
                'std_intensity': float(channel_0_props.get('std', 239.483154296875)),
                'lower_bound': float(channel_0_props.get('percentile_00_5', 130.0)),
                'upper_bound': float(channel_0_props.get('percentile_99_5', 1272.0))
            }
        else:
            print(f"⚠️ Warning: Channel '0' not found in intensity properties")
            return None
            
    except Exception as e:
        print(f"⚠️ Warning: Failed to load intensity properties from {plans_file_path}: {e}")
        return None


class CTNormalization:
    """CT 이미지 정규화를 위한 클래스"""
    
    def __init__(self, mean_intensity=None, std_intensity=None, lower_bound=None, upper_bound=None, target_dtype=np.float32):
        self.mean_intensity = mean_intensity
        self.std_intensity = std_intensity
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.target_dtype = target_dtype
        self.leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False
    
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
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


def load_nifti_image(image_path):
    """NIfTI 이미지를 로딩하고 PyTorch tensor로 변환하는 공통 함수"""
    import nibabel as nib
    nii_img = nib.load(image_path)
    img_data = nii_img.get_fdata()
    
    # numpy array를 torch tensor로 변환하기 위해 float32로 변환
    img_data = img_data.astype(np.float32)
    
    # NIfTI 기본 순서 (W, H, D)를 PyTorch 표준 (D, H, W)로 transpose
    img_data = np.transpose(img_data, (2, 1, 0))  # (W, H, D) -> (D, H, W)
    
    # 채널 차원 추가 (D, H, W) -> (1, D, H, W)
    if len(img_data.shape) == 3:
        img_data = img_data[np.newaxis, ...]  # 첫 번째 차원에 채널 추가
    
    # torch tensor로 변환
    img_tensor = torch.from_numpy(img_data)
    
    return img_tensor


class ASDataset(Dataset):
    """AS Radiomics 데이터를 위한 Dataset 클래스"""
    
    def __init__(self, image_files, labels, label_to_idx, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.encoded_labels = np.array([label_to_idx[label] for label in labels], dtype=np.int64)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_path = self.image_files[index]
        label = self.encoded_labels[index]
        
        # 공통 이미지 로딩 함수 사용
        img_data = load_nifti_image(image_path)
        
        if self.transform is not None:
            img_data = self.transform(img_data)
        
        # 파일명 추출
        file_name = os.path.basename(image_path)
        
        return {
            'imgs': img_data,
            'labels': label,
            'names': file_name
        }


def prepare_as_data(data_split_mode='fix', data_split_random_state=42, test_size_ratio=0.4):
    """AS 데이터 준비 - 설정에 따른 train/test 분할
    
    Args:
        data_split_mode: 분할 모드 ('fix' 또는 'random')
        data_split_random_state: 랜덤 시드 (random 모드에서만 사용)
        test_size_ratio: 테스트 데이터 비율 (random 모드에서만 사용)
    """
    
    # 기존 데이터 로더 사용
    data_loader = ASDataLoader(Config.LABEL_FILE)
    severity_map = data_loader.get_severity_mapping(mode=Config.CLASSIFICATION_MODE)
    
    print(f"이미지 파일 수집 중...")
    print(f"  Train 이미지 디렉토리: {Config.IMAGE_TR_DIR}")
    print(f"  Val 이미지 디렉토리: {Config.IMAGE_VAL_DIR}")
    
    # 모든 디렉토리에서 데이터 수집
    all_files, all_labels = [], []
    
    # TR 디렉토리에서 데이터 수집
    if os.path.exists(Config.IMAGE_TR_DIR):
        nii_files = glob(os.path.join(Config.IMAGE_TR_DIR, "*.nii.gz"))
        for file_path in nii_files:
            filename = os.path.basename(file_path)
            match = re.match(r'([A-Za-z0-9\.\-]+)_(\d{4,})_0000\.nii\.gz', filename)
            if match:
                patient_id = match.group(1).strip()
                if patient_id in severity_map:
                    all_files.append(file_path)
                    all_labels.append(severity_map[patient_id])
    
    # VAL 디렉토리에서 데이터 수집
    if os.path.exists(Config.IMAGE_VAL_DIR):
        nii_files = glob(os.path.join(Config.IMAGE_VAL_DIR, "*.nii.gz"))
        for file_path in nii_files:
            filename = os.path.basename(file_path)
            match = re.match(r'([A-Za-z0-9\.\-]+)_(\d{4,})_0000\.nii\.gz', filename)
            if match:
                patient_id = match.group(1).strip()
                if patient_id in severity_map:
                    all_files.append(file_path)
                    all_labels.append(severity_map[patient_id])
    
    if not all_files:
        print("오류: 레이블과 매칭되는 이미지 파일이 없습니다.")
        sys.exit(1)
    
    print(f"  전체 파일 수: {len(all_files)}")
    
    # 레이블 인코딩
    unique_labels = sorted(list(set(all_labels)))
    if Config.CLASSIFICATION_MODE == 'multi':
        desired_order = ['normal', 'nonsevere', 'severe']
        unique_labels = [cls for cls in desired_order if cls in unique_labels]
        remaining_labels = sorted([cls for cls in set(all_labels) if cls not in desired_order])
        unique_labels.extend(remaining_labels)
    
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"  클래스 매핑: {label_to_idx}")
    print(f"  클래스별 샘플 수:")
    for label, idx in label_to_idx.items():
        count = sum(1 for l in all_labels if l == label)
        print(f"    {label}: {count}")
    
    if data_split_mode == 'fix':
        # 고정 분할: 디렉토리 기반으로 분할
        print(f"\n  고정 분할 모드: 디렉토리 기반 분할")
        
        # TR 디렉토리 데이터를 train으로, VAL 디렉토리 데이터를 test로 사용
        train_files, train_labels = [], []
        test_files, test_labels = [], []
        
        # TR 디렉토리에서 데이터 수집 (train)
        if os.path.exists(Config.IMAGE_TR_DIR):
            nii_files = glob(os.path.join(Config.IMAGE_TR_DIR, "*.nii.gz"))
            for file_path in nii_files:
                filename = os.path.basename(file_path)
                match = re.match(r'([A-Za-z0-9\.\-]+)_(\d{4,})_0000\.nii\.gz', filename)
                if match:
                    patient_id = match.group(1).strip()
                    if patient_id in severity_map:
                        train_files.append(file_path)
                        train_labels.append(severity_map[patient_id])
        
        # VAL 디렉토리에서 데이터 수집 (test)
        if os.path.exists(Config.IMAGE_VAL_DIR):
            nii_files = glob(os.path.join(Config.IMAGE_VAL_DIR, "*.nii.gz"))
            for file_path in nii_files:
                filename = os.path.basename(file_path)
                match = re.match(r'([A-Za-z0-9\.\-]+)_(\d{4,})_0000\.nii\.gz', filename)
                if match:
                    patient_id = match.group(1).strip()
                    if patient_id in severity_map:
                        test_files.append(file_path)
                        test_labels.append(severity_map[patient_id])
        
        train_df = pd.DataFrame({
            'image_path': train_files,
            'severity': train_labels
        })
        
        test_df = pd.DataFrame({
            'image_path': test_files,
            'severity': test_labels
        })
        
        print(f"    Train 데이터: {len(train_df)} 샘플")
        print(f"    Test 데이터: {len(test_df)} 샘플")
        
    elif data_split_mode == 'random':
        # 랜덤 분할: 전체 데이터를 랜덤하게 분할
        print(f"\n  랜덤 분할 모드: 전체 데이터 랜덤 분할 (test_ratio={test_size_ratio}, random_state={data_split_random_state})")
        
        # 전체 데이터를 DataFrame으로 생성
        all_df = pd.DataFrame({
            'image_path': all_files,
            'severity': all_labels
        })
        
        # DataSplitter를 사용하여 train/test 분할 (랜덤 모드로 설정)
        original_mode = Config.DATA_SPLIT_MODE
        original_random_state = Config.DATA_SPLIT_RANDOM_STATE
        original_test_ratio = Config.TEST_SIZE_RATIO
        
        # 임시로 설정 변경
        Config.DATA_SPLIT_MODE = 'random'
        Config.DATA_SPLIT_RANDOM_STATE = data_split_random_state
        Config.TEST_SIZE_RATIO = test_size_ratio
        
        splitter = DataSplitter()
        train_df, test_df = splitter.split_data(all_df, mode=Config.CLASSIFICATION_MODE)
        
        # 원래 설정 복원
        Config.DATA_SPLIT_MODE = original_mode
        Config.DATA_SPLIT_RANDOM_STATE = original_random_state
        Config.TEST_SIZE_RATIO = original_test_ratio
        
        print(f"    Train 데이터: {len(train_df)} 샘플")
        print(f"    Test 데이터: {len(test_df)} 샘플")
        
    else:
        raise ValueError(f"지원하지 않는 data_split_mode: {data_split_mode}")
    
    return train_df, test_df, label_to_idx, idx_to_label, unique_labels


def get_as_dataset(img_size, mode='train', data_split_mode='fix', data_split_random_state=42, test_size_ratio=0.4):
    """AS 데이터셋을 위한 새로운 함수
    
    Args:
        img_size: 이미지 크기
        mode: 데이터셋 모드 ('train', 'test', 'val')
        data_split_mode: 분할 모드 ('fix' 또는 'random')
        data_split_random_state: 랜덤 시드 (random 모드에서만 사용)
        test_size_ratio: 테스트 데이터 비율 (random 모드에서만 사용)
    """
    
    # 데이터 준비
    train_df, test_df, label_to_idx, idx_to_label, unique_labels = prepare_as_data(
        data_split_mode=data_split_mode,
        data_split_random_state=data_split_random_state,
        test_size_ratio=test_size_ratio
    )
    
    # nnUNet plans 파일에서 intensity properties 로드 시도
    intensity_props = None
    if hasattr(Config, 'DL_NNUNET_CONFIG') and Config.DL_NNUNET_CONFIG:
        plans_file = Config.DL_NNUNET_CONFIG.get('plans_file')
        if plans_file:
            intensity_props = load_intensity_properties_from_plans(plans_file)
    
    # Fallback으로 기본값 사용
    if intensity_props is None:
        print("ℹ️ Using default intensity normalization values")
        intensity_props = {
            'mean_intensity': 363.5522766113281,
            'std_intensity': 249.69992065429688,
            'lower_bound': 130.0,
            'upper_bound': 1298.0
        }
    else:
        print(f"✓ Loaded intensity properties from nnUNet plans file:")
        print(f"  Mean: {intensity_props['mean_intensity']:.2f}")
        print(f"  Std: {intensity_props['std_intensity']:.2f}")
        print(f"  Lower bound: {intensity_props['lower_bound']:.2f}")
        print(f"  Upper bound: {intensity_props['upper_bound']:.2f}\n")
    ct_normalization = CTNormalization(**intensity_props)
    
    # img_size가 튜플이 아닌 경우 튜플로 변환
    if isinstance(img_size, (int, float)):
        img_size = (int(img_size), int(img_size), int(img_size))
    elif isinstance(img_size, (tuple, list)) and len(img_size) == 3:
        img_size = tuple(int(x) for x in img_size)
    else:
        raise ValueError(f"img_size는 정수 또는 3개 요소의 튜플이어야 합니다. 받은 값: {img_size}")
    
    # Cardiac CT에 적합한 Augmentation
    train_transform = Compose([
        # 좌우 반전만 유지 (심장은 좌우 대칭성이 있음)
        RandFlip(prob=0.5, spatial_axis=2),  # 좌우 반전만
        # 회전 범위 크게 줄임 (심장의 해부학적 방향성 보존)
        RandAffine(prob=0.5, rotate_range=(np.pi/30, np.pi/30, np.pi/30), 
                  scale_range=(0.05, 0.05, 0.05), padding_mode='zeros'),
        # 약간의 밝기/대비 조정만 유지
        RandAdjustContrast(prob=0.3, gamma=(0.9, 1.1)),
        # 가우시안 노이즈 크게 줄임 (칼슘의 고밀도 특성 보존)
        RandGaussianNoise(prob=0.3, std=0.01),
        ct_normalization,
        Resize(img_size, mode='trilinear'),
    ])
    
    test_transform = Compose([
        ct_normalization,
        Resize(img_size, mode='trilinear')
    ])
    
    if mode == 'train':
        # 훈련 데이터 반환
        return ASDataset(
            image_files=train_df['image_path'].tolist(),
            labels=train_df['severity'].tolist(),
            label_to_idx=label_to_idx,
            transform=train_transform
        ), label_to_idx, idx_to_label, unique_labels
    
    elif mode == 'test' or mode == 'val':
        # 검증 데이터 반환
        return ASDataset(
            image_files=test_df['image_path'].tolist(),
            labels=test_df['severity'].tolist(),
            label_to_idx=label_to_idx,
            transform=test_transform
        ), label_to_idx, idx_to_label, unique_labels
    
    else:
        raise ValueError(f"지원하지 않는 모드: {mode}")