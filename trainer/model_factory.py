from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class ModelFactory:
    """다양한 머신러닝 모델을 생성하는 팩토리 클래스"""
    
    def __init__(self, config):
        self.config = config
    
    def create_models(self, model_names=None):
        """지정된 모델들을 생성"""
        if model_names is None:
            model_names = self.config.CLASSIFICATION_MODELS
        
        models = {}
        for model_name in model_names:
            model = self._create_single_model(model_name)
            if model is not None:
                models[model_name] = model
            else:
                print(f"  경고: 알 수 없는 모델 '{model_name}'을 건너뜁니다.")
        
        return models
    
    def _create_single_model(self, model_name):
        """단일 모델 생성"""
        model_configs = {
            'LR': self._create_logistic_regression,
            'SVM': self._create_svm,
            'RF': self._create_random_forest,
            'GB': self._create_gradient_boosting,
            'KNN': self._create_knn,
            'NB': self._create_naive_bayes
        }
        
        if model_name in model_configs:
            return model_configs[model_name]()
        else:
            return None
    
    def _create_logistic_regression(self):
        """로지스틱 회귀 모델 생성"""
        return LogisticRegression(
            random_state=self.config.RANDOM_STATE,
            max_iter=self.config.LR_MAX_ITER,
            solver=self.config.LR_SOLVER,
            C=self.config.LR_C
        )
    
    def _create_svm(self):
        """SVM 모델 생성"""
        return SVC(
            probability=self.config.SVM_PROBABILITY,
            random_state=self.config.RANDOM_STATE,
            C=self.config.SVM_C,
            kernel=self.config.SVM_KERNEL,
            gamma=self.config.SVM_GAMMA
        )
    
    def _create_random_forest(self):
        """랜덤 포레스트 모델 생성"""
        return RandomForestClassifier(
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
            n_estimators=self.config.RF_N_ESTIMATORS,
            max_depth=self.config.RF_MAX_DEPTH,
            min_samples_split=self.config.RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=self.config.RF_MIN_SAMPLES_LEAF,
            max_features=self.config.RF_MAX_FEATURES
        )
    
    def _create_gradient_boosting(self):
        """그래디언트 부스팅 모델 생성"""
        return GradientBoostingClassifier(
            random_state=self.config.RANDOM_STATE,
            n_estimators=self.config.GB_N_ESTIMATORS,
            learning_rate=self.config.GB_LEARNING_RATE,
            max_depth=self.config.GB_MAX_DEPTH,
            min_samples_split=self.config.GB_MIN_SAMPLES_SPLIT,
            min_samples_leaf=self.config.GB_MIN_SAMPLES_LEAF,
            subsample=self.config.GB_SUBSAMPLE
        )
    
    def _create_knn(self):
        """K-최근접 이웃 모델 생성"""
        return KNeighborsClassifier(
            n_neighbors=self.config.KNN_N_NEIGHBORS,
            weights=self.config.KNN_WEIGHTS,
            algorithm=self.config.KNN_ALGORITHM,
            p=self.config.KNN_P,
            n_jobs=-1
        )
    
    def _create_naive_bayes(self):
        """나이브 베이즈 모델 생성"""
        return GaussianNB(
            var_smoothing=self.config.NB_VAR_SMOOTHING
        )
    
    def get_available_models(self):
        """사용 가능한 모델 목록 반환"""
        return ['LR', 'SVM', 'RF', 'GB', 'KNN', 'NB']
