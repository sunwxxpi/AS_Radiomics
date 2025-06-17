import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    """다양한 특징 선택 방법을 제공하는 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.method = config.FEATURE_SELECTION_METHOD
        self.selector = None
        
    def select_features(self, x_train, y_train, x_val=None):
        """특징 선택 메인 함수"""
        print(f"--- 특징 선택 시작 (방법: {self.method}) ---")
        
        if x_train.empty or len(y_train) == 0:
            print("  경고: 데이터가 비어있어 특징 선택을 건너뜁니다.")
            return x_train, x_val if x_val is not None else pd.DataFrame()
        
        if self.method == 'none':
            print("  특징 선택을 건너뜁니다.")
            return x_train, x_val if x_val is not None else pd.DataFrame()
        
        # 데이터 유효성 검사
        if not self._validate_data(x_train, y_train):
            return x_train, x_val if x_val is not None else pd.DataFrame()
        
        try:
            # 특징 선택 방법에 따라 선택기 생성
            self.selector = self._create_selector(x_train, y_train)
            
            if self.selector is None:
                print("  특징 선택기 생성 실패. 모든 특징을 사용합니다.")
                return x_train, x_val if x_val is not None else pd.DataFrame()
            
            # 특징 선택 수행
            return self._apply_selection(x_train, x_val, y_train)
            
        except Exception as e:
            print(f"  특징 선택 오류: {e}")
            print("  모든 특징을 사용합니다.")
            return x_train, x_val if x_val is not None else pd.DataFrame()
    
    def _validate_data(self, x_train, y_train):
        """데이터 유효성 검사"""
        min_class_count = np.min(np.bincount(y_train)) if len(np.unique(y_train)) > 1 else 0
        cv_folds = min(self.config.CV_FOLDS, max(2, min_class_count))
        
        if min_class_count < 2 or x_train.shape[0] < cv_folds:
            print("  경고: 샘플 수 부족으로 특징 선택을 건너뜁니다.")
            return False
        return True
    
    def _create_selector(self, x_train, y_train):
        """특징 선택 방법에 따른 선택기 생성"""
        min_class_count = np.min(np.bincount(y_train))
        cv_folds = min(self.config.CV_FOLDS, max(2, min_class_count))
        
        if self.method == 'lasso':
            return self._create_lasso_selector(cv_folds)
        elif self.method == 'rfe':
            return self._create_rfe_selector(x_train.shape[1])
        elif self.method == 'univariate':
            return self._create_univariate_selector(x_train.shape[1])
        elif self.method == 'mutual_info':
            return self._create_mutual_info_selector(x_train.shape[1])
        elif self.method == 'random_forest':
            return self._create_rf_selector()
        else:
            print(f"  경고: 알 수 없는 특징 선택 방법: {self.method}")
            return None
    
    def _create_lasso_selector(self, cv_folds):
        """Lasso 기반 특징 선택기 생성"""
        print(f"  LassoCV 특징 선택 시작 (CV folds: {cv_folds})...")
        
        lasso_cv = LassoCV(
            cv=cv_folds,
            random_state=self.config.RANDOM_STATE,
            max_iter=10000,
            n_jobs=-1,
            tol=self.config.LASSO_TOLERANCE,
            n_alphas=self.config.LASSO_ALPHA_COUNT
        )
        
        return SelectFromModel(
            estimator=lasso_cv,
            threshold=self.config.FEATURE_THRESHOLD
        )
    
    def _create_rfe_selector(self, n_features):
        """RFE 기반 특징 선택기 생성"""
        print("  RFE 특징 선택 시작...")
        
        # 선택할 특징 수 계산
        target_features = int(n_features * self.config.RFE_N_FEATURES_RATIO)
        n_features_to_select = max(
            self.config.RFE_MIN_FEATURES,
            min(self.config.RFE_MAX_FEATURES, target_features)
        )
        
        estimator = RandomForestClassifier(
            random_state=self.config.RANDOM_STATE,
            n_estimators=self.config.RFE_ESTIMATOR_N_ESTIMATORS,
            n_jobs=-1
        )
        
        print(f"    선택할 특징 수: {n_features_to_select}")
        
        return RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=self.config.RFE_STEP
        )
    
    def _create_univariate_selector(self, n_features):
        """Univariate 통계 기반 특징 선택기 생성"""
        print("  Univariate F-test 특징 선택 시작...")
        
        # 선택할 특징 수 계산
        target_k = int(n_features * self.config.UNIVARIATE_K_RATIO)
        k_features = max(
            self.config.UNIVARIATE_MIN_K,
            min(self.config.UNIVARIATE_MAX_K, target_k)
        )
        
        print(f"    선택할 특징 수: {k_features}")
        
        return SelectKBest(
            score_func=f_classif,
            k=k_features
        )
    
    def _create_mutual_info_selector(self, n_features):
        """Mutual Information 기반 특징 선택기 생성"""
        print("  Mutual Information 특징 선택 시작...")
        
        # 선택할 특징 수 계산
        target_k = int(n_features * self.config.MUTUAL_INFO_K_RATIO)
        k_features = max(
            self.config.MUTUAL_INFO_MIN_K,
            min(self.config.MUTUAL_INFO_MAX_K, target_k)
        )
        
        print(f"    선택할 특징 수: {k_features}")
        
        return SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=self.config.MUTUAL_INFO_RANDOM_STATE),
            k=k_features
        )
    
    def _create_rf_selector(self):
        """Random Forest 중요도 기반 특징 선택기 생성"""
        print("  Random Forest 중요도 기반 특징 선택 시작...")
        
        rf = RandomForestClassifier(
            random_state=self.config.RANDOM_STATE,
            n_estimators=self.config.RF_FEATURE_N_ESTIMATORS,
            n_jobs=-1
        )
        
        return SelectFromModel(
            estimator=rf,
            threshold=self.config.RF_FEATURE_THRESHOLD
        )
    
    def _apply_selection(self, x_train, x_val, y_train):
        """특징 선택 적용"""
        # 선택기 학습
        self.selector.fit(x_train, y_train)
        
        # 선택된 특징 확인
        if hasattr(self.selector, 'get_support'):
            selected_features = self.selector.get_support(indices=True)
            selected_feature_names = x_train.columns[selected_features]
        else:
            # RFE의 경우
            selected_feature_names = x_train.columns[self.selector.support_]
        
        if len(selected_feature_names) == 0:
            print("  경고: 선택된 특징이 없어 모든 특징을 사용합니다.")
            return x_train, x_val if x_val is not None else pd.DataFrame()
        
        # 특징 변환
        x_train_selected = pd.DataFrame(
            self.selector.transform(x_train),
            columns=selected_feature_names,
            index=x_train.index
        )
        
        x_val_selected = pd.DataFrame()
        if x_val is not None and not x_val.empty:
            x_val_selected = pd.DataFrame(
                self.selector.transform(x_val),
                columns=selected_feature_names,
                index=x_val.index
            )
        
        print(f"  특징 선택 완료: {x_train.shape[1]} → {len(selected_feature_names)}")
        print(f"  선택된 특징 예시: {list(selected_feature_names[:5])}")
        
        # 선택 방법별 추가 정보 출력
        self._print_selection_info(x_train, selected_feature_names)
        
        return x_train_selected, x_val_selected
    
    def _print_selection_info(self):
        """선택 방법별 추가 정보 출력"""
        if self.method == 'lasso' and hasattr(self.selector.estimator_, 'alpha_'):
            print(f"    최적 alpha: {self.selector.estimator_.alpha_:.6f}")
        
        elif self.method == 'rfe' and hasattr(self.selector, 'ranking_'):
            print(f"    특징 순위 범위: {np.min(self.selector.ranking_)} ~ {np.max(self.selector.ranking_)}")
        
        elif self.method in ['univariate', 'mutual_info'] and hasattr(self.selector, 'scores_'):
            selected_scores = self.selector.scores_[self.selector.get_support()]
            print(f"    선택된 특징 점수 범위: {np.min(selected_scores):.4f} ~ {np.max(selected_scores):.4f}")
        
        elif self.method == 'random_forest' and hasattr(self.selector.estimator_, 'feature_importances_'):
            selected_importances = self.selector.estimator_.feature_importances_[self.selector.get_support()]
            print(f"    선택된 특징 중요도 범위: {np.min(selected_importances):.4f} ~ {np.max(selected_importances):.4f}")
    
    def get_available_methods(self):
        """사용 가능한 특징 선택 방법 목록 반환"""
        return ['lasso', 'rfe', 'univariate', 'mutual_info', 'random_forest', 'none']
