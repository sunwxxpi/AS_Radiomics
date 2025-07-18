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
        x_train, y_train = self._split_features_labels(train_df)
        x_val, y_val = self._split_features_labels(val_df) if val_df is not None and not val_df.empty else (pd.DataFrame(), pd.Series())
        
        # 2. 레이블 인코딩
        y_train_encoded = self._encode_labels(y_train)
        y_val_encoded = self._encode_labels(y_val, fit=False) if not y_val.empty else np.array([])
        
        # 3. 결측값 처리
        x_train_imputed = self._handle_missing_values(x_train)
        x_val_imputed = self._handle_missing_values(x_val, fit=False) if not x_val.empty else pd.DataFrame()
        
        # 4. 특징 스케일링
        x_train_scaled = self._scale_features(x_train_imputed)
        x_val_scaled = self._scale_features(x_val_imputed, fit=False) if not x_val_imputed.empty else pd.DataFrame()
        
        # 5. 특징 선택
        x_train_selected, x_val_selected = self.feature_selector.select_features(x_train_scaled, y_train_encoded, x_val_scaled)
        
        print("--- 데이터 전처리 완료 ---\n")
        
        return {
            'x_train': x_train_selected,
            'y_train': y_train_encoded,
            'x_val': x_val_selected,
            'y_val': y_val_encoded
        }
    
    def get_lasso_analysis(self):
        """LASSO 분석 결과 반환"""
        if hasattr(self.feature_selector, 'lasso_analysis'):
            return self.feature_selector.lasso_analysis
        return None
    
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
            
            # 다중 분류의 경우 클래스 순서 지정
            if self.config.CLASSIFICATION_MODE == 'multi':
                # 원하는 순서: normal(0), nonsevere(1), severe(2)
                desired_order = ['normal', 'nonsevere', 'severe']
                unique_classes = y.unique()
                # 실제 데이터에 존재하는 클래스만 원하는 순서로 정렬
                classes_to_fit = [cls for cls in desired_order if cls in unique_classes]
                # 원하는 순서에 없는 클래스가 있다면 뒤에 추가 (알파벳 순서)
                remaining_classes = sorted([cls for cls in unique_classes if cls not in desired_order])
                classes_to_fit.extend(remaining_classes)
                
                # 명시적으로 클래스 순서 지정하여 fit
                self.label_encoder.classes_ = np.array(classes_to_fit, dtype=object)
            else:
                # 이진 분류의 경우 기본 동작 유지
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
            x_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            x_imputed = pd.DataFrame(
                self.imputer.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"    {missing_count}개 결측값을 평균으로 대체")
        
        return x_imputed
    
    def _scale_features(self, X, fit=True):
        """특징 스케일링"""
        if X.empty:
            return pd.DataFrame()
        
        print(f"  특징 스케일링 {'(fit)' if fit else '(transform)'}...")
        
        if fit:
            x_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            x_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return x_scaled