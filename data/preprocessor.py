import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

class DataPreprocessor:
    """데이터 전처리를 담당하는 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = None
    
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
        X_train_selected, X_val_selected = self._select_features(X_train_scaled, y_train_encoded, X_val_scaled)
        
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
    
    def _select_features(self, X_train, y_train, X_val=None):
        """특징 선택 (LassoCV)"""
        print("--- 특징 선택 시작 ---")
        
        if X_train.empty or len(y_train) == 0:
            print("  경고: 데이터가 비어있어 특징 선택을 건너뜁니다.")
            return X_train, X_val if X_val is not None else pd.DataFrame()
        
        min_class_count = np.min(np.bincount(y_train)) if len(np.unique(y_train)) > 1 else 0
        cv_folds = min(self.config.CV_FOLDS, max(2, min_class_count))
        
        if min_class_count < 2 or X_train.shape[0] < cv_folds:
            print("  경고: 샘플 수 부족으로 특징 선택을 건너뜁니다.")
            return X_train, X_val if X_val is not None else pd.DataFrame()
        
        try:
            print(f"  LassoCV 특징 선택 시작 (CV folds: {cv_folds})...")
            
            lasso_cv = LassoCV(
                cv=cv_folds,
                random_state=self.config.RANDOM_STATE,
                max_iter=10000,
                n_jobs=-1,
                tol=self.config.LASSO_TOLERANCE,
                n_alphas=self.config.LASSO_ALPHA_COUNT
            )
            
            self.feature_selector = SelectFromModel(
                estimator=lasso_cv,
                threshold=self.config.FEATURE_THRESHOLD
            )
            
            self.feature_selector.fit(X_train, y_train)
            
            # 선택된 특징으로 변환
            selected_features = self.feature_selector.get_support(indices=True)
            selected_feature_names = X_train.columns[selected_features]
            
            if len(selected_feature_names) == 0:
                print("  경고: 선택된 특징이 없어 모든 특징을 사용합니다.")
                return X_train, X_val if X_val is not None else pd.DataFrame()
            
            X_train_selected = pd.DataFrame(
                self.feature_selector.transform(X_train),
                columns=selected_feature_names,
                index=X_train.index
            )
            
            X_val_selected = pd.DataFrame()
            if X_val is not None and not X_val.empty:
                X_val_selected = pd.DataFrame(
                    self.feature_selector.transform(X_val),
                    columns=selected_feature_names,
                    index=X_val.index
                )
            
            print(f"  특징 선택 완료: {X_train.shape[1]} → {len(selected_feature_names)}")
            return X_train_selected, X_val_selected
            
        except Exception as e:
            print(f"  특징 선택 오류: {e}")
            print("  모든 특징을 사용합니다.")
            return X_train, X_val if X_val is not None else pd.DataFrame()