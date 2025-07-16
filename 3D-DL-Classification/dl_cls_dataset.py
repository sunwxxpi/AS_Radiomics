import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import torch
from glob import glob
from torch.utils.data.dataset import Dataset
from monai.transforms import Compose, RandFlip, RandAffine, RandAdjustContrast, RandGaussianNoise, Resize
from config import Config
from data.loader import DataLoader as ASDataLoader
from utils.data_splitter import DataSplitter


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
        
        # MONAI를 사용하여 NIfTI 파일 로드
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
        img_data = torch.from_numpy(img_data)
        
        if self.transform is not None:
            img_data = self.transform(img_data)
        
        # 파일명 추출
        file_name = os.path.basename(image_path)
        
        return {
            'imgs': img_data,
            'labels': label,
            'names': file_name
        }


def prepare_as_data():
    """AS 데이터 준비 - DataSplitter를 사용한 train/test 분할"""
    
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
    
    # 전체 데이터를 DataFrame으로 생성
    all_df = pd.DataFrame({
        'image_path': all_files,
        'severity': all_labels
    })
    
    # DataSplitter를 사용하여 train/test 분할
    print("\n  DataSplitter를 사용한 train/test 분할...")
    splitter = DataSplitter()
    train_df, test_df = splitter.split_data(all_df, mode=Config.CLASSIFICATION_MODE)
    
    return train_df, test_df, label_to_idx, idx_to_label, unique_labels


def get_as_dataset(img_size, mode='train'):
    """AS 데이터셋을 위한 새로운 함수"""
    
    # 데이터 준비
    train_df, test_df, label_to_idx, idx_to_label, unique_labels = prepare_as_data()
    
    ct_normalization = CTNormalization(
        mean_intensity=363.5522766113281,
        std_intensity=249.69992065429688,
        lower_bound=130.0,
        upper_bound=1298.0
    )
    
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
        # Resize((img_size/2, img_size, img_size), mode='trilinear'),
        Resize((img_size/5, img_size, img_size), mode='trilinear'),
    ])
    
    test_transform = Compose([
        ct_normalization,
        # Resize((img_size/2, img_size, img_size), mode='trilinear')
        Resize((img_size/5, img_size, img_size), mode='trilinear')
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