import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from trainer.feature_selector import FeatureSelector

class DataPreprocessor:
    """데이터 전처리를 담당하는 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = FeatureSelector(config)
    
    def prepare_data(self, train_df, val_df=None):
        """전체 데이터 전처리 파이프라인"""
        print("--- 데이터 전처리 시작 ---")
        
        # 1. 특징과 레이블 분리
        X_train, y_train = self._split_features_labels(train_df)
        X_val, y_val = self._split_features_labels(val_df) if val_df is not None and not val_df.empty else (pd.DataFrame(), pd.Series())
        
        # 2. 레이블 인코딩
        y_train_encoded = self._encode_labels(y_train)
        y_val_encoded = self._encode_labels(y_val, fit=False) if not y_val.empty else np.array([])
        
        # 3. 결측값 처리
        X_train_imputed = self._handle_missing_values(X_train)
        X_val_imputed = self._handle_missing_values(X_val, fit=False) if not X_val.empty else pd.DataFrame()
        
        # 4. 특징 스케일링
        X_train_scaled = self._scale_features(X_train_imputed)
        X_val_scaled = self._scale_features(X_val_imputed, fit=False) if not X_val_imputed.empty else pd.DataFrame()
        
        # 5. 특징 선택
        X_train_selected, X_val_selected = self.feature_selector.select_features(X_train_scaled, y_train_encoded, X_val_scaled)
        
        print("--- 데이터 전처리 완료 ---\n")
        
        return {
            'X_train': X_train_selected,
            'y_train': y_train_encoded,
            'X_val': X_val_selected,
            'y_val': y_val_encoded
        }
    
    def _split_features_labels(self, df):
        """특징과 레이블 분리"""
        if df is None or df.empty:
            return pd.DataFrame(), pd.Series()
        
        if 'severity' not in df.columns:
            raise ValueError("DataFrame에 'severity' 컬럼이 없습니다.")
        
        X = df.drop('severity', axis=1) 
        if 'as_grade' in X.columns:
            X = X.drop('as_grade', axis=1)
            
        y = df['severity'].copy()
        
        return X, y
    
    def _encode_labels(self, y, fit=True):
        """레이블 인코딩"""
        if y.empty:
            return np.array([])
        
        if fit:
            print("  레이블 인코딩 시작...")
            self.label_encoder.fit(y)
            print(f"    매핑: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        try:
            y_encoded = self.label_encoder.transform(y)
            print(f"    인코딩된 레이블 분포: {np.bincount(y_encoded)}")
            return y_encoded
        except ValueError as e:
            print(f"    레이블 인코딩 오류: {e}")
            return np.array([])
    
    def _handle_missing_values(self, X, fit=True):
        """결측값 처리"""
        if X.empty:
            return pd.DataFrame()
        
        print(f"  결측값 처리 {'(fit)' if fit else '(transform)'}...")
        
        if fit:
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_imputed = pd.DataFrame(
                self.imputer.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"    {missing_count}개 결측값을 평균으로 대체")
        
        return X_imputed
    
    def _scale_features(self, X, fit=True):
        """특징 스케일링"""
        if X.empty:
            return pd.DataFrame()
        
        print(f"  특징 스케일링 {'(fit)' if fit else '(transform)'}...")
        
        if fit:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled