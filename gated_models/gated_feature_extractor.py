import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from gated_models.gated_model import GatedFusionClassifier


class GatedFeatureExtractor:
    """학습된 Gated Fusion 모델로 fused features 추출

    전통적 ML 분류기 학습을 위한 특징 추출기
    """

    def __init__(self, model_path, model_config, scaler_radiomics=None, scaler_dl=None, device=None):
        """
        Args:
            model_path (str): 학습된 모델 체크포인트 경로
            model_config (dict): 모델 설정
            scaler_radiomics (StandardScaler): Radiomics 정규화 스케일러
            scaler_dl (StandardScaler): DL 정규화 스케일러
            device (torch.device): 디바이스
        """
        self.model_path = model_path
        self.model_config = model_config
        self.scaler_radiomics = scaler_radiomics if scaler_radiomics else StandardScaler()
        self.scaler_dl = scaler_dl if scaler_dl else StandardScaler()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self._load_model()

    def _load_model(self):
        """학습된 모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 모델 생성
        self.model = GatedFusionClassifier(
            radiomics_dim=self.model_config['radiomics_dim'],
            dl_dim=self.model_config['dl_dim'],
            num_classes=self.model_config['num_classes'],
            fusion_dim=self.model_config.get('fusion_dim'),
            hidden_dims=self.model_config.get('hidden_dims', [256, 128]),
            dropout=0.0  # Inference에서는 dropout 비활성화
        )

        # 가중치 로드
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"학습된 Gated Fusion 모델 로드 완료: {self.model_path}")
        print(f"  - Validation Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"  - Validation Accuracy: {checkpoint.get('val_acc', 'N/A')}")

    def extract_features_from_dataframe(self, features_df, use_fitted_scaler=False):
        """DataFrame에서 fused features 추출

        Args:
            features_df (pd.DataFrame): Radiomics + DL features
            use_fitted_scaler (bool): 학습된 scaler 사용 여부 (True: transform, False: fit_transform)

        Returns:
            pd.DataFrame: Fused features DataFrame
        """
        # 특징 컬럼 분리
        radiomics_cols = [col for col in features_df.columns
                          if col.startswith(('original_', 'wavelet_', 'log_', 'square_', 'squareroot_',
                                           'exponential_', 'logarithm_', 'gradient_', 'lbp-'))]
        dl_cols = [col for col in features_df.columns
                   if col.startswith('dl_embedding_feature_')]

        if not radiomics_cols or not dl_cols:
            raise ValueError("Radiomics 또는 DL embedding 특징을 찾을 수 없습니다.")

        # 특징 추출
        X_radiomics = features_df[radiomics_cols].values
        X_dl = features_df[dl_cols].values

        # 정규화
        if use_fitted_scaler:
            X_radiomics_scaled = self.scaler_radiomics.transform(X_radiomics)
            X_dl_scaled = self.scaler_dl.transform(X_dl)
        else:
            X_radiomics_scaled = self.scaler_radiomics.fit_transform(X_radiomics)
            X_dl_scaled = self.scaler_dl.fit_transform(X_dl)

        # Fused features 추출
        fused_features = self.extract_features(X_radiomics_scaled, X_dl_scaled)

        # DataFrame 생성
        fused_cols = [f'gated_fused_feature_{i+1}' for i in range(fused_features.shape[1])]
        fused_df = pd.DataFrame(fused_features, columns=fused_cols, index=features_df.index)

        # Metadata 컬럼 추가 (case_id, severity, data_source)
        meta_cols = ['case_id', 'severity', 'data_source']
        for col in meta_cols:
            if col in features_df.columns:
                fused_df[col] = features_df[col].values

        return fused_df

    def extract_features(self, radiomics_features, dl_features):
        """Numpy array로 fused features 추출

        Args:
            radiomics_features (np.ndarray): [N, radiomics_dim]
            dl_features (np.ndarray): [N, dl_dim]

        Returns:
            np.ndarray: [N, fusion_dim]
        """
        radiomics_tensor = torch.FloatTensor(radiomics_features).to(self.device)
        dl_tensor = torch.FloatTensor(dl_features).to(self.device)

        with torch.no_grad():
            fused = self.model.extract_fused_features(radiomics_tensor, dl_tensor)

        return fused.cpu().numpy()