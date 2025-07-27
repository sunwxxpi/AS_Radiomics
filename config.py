import os
import datetime

class Config:
    """프로젝트 설정을 관리하는 클래스"""
    
    # 경로 설정
    BASE_DIR = '/home/psw/AVS-Diagnosis/Dataset001_KMU_Cardiac_AVC'
    # BASE_DIR = '/home/psw/AVS-Diagnosis/Dataset002_KMU_Chest_AVC'
    LABEL_FILE = './data/AS_CRF_radiomics.csv'
    BASE_OUTPUT_DIR = './radiomics_analysis_results'
    
    IMAGE_TR_DIR = os.path.join(BASE_DIR, 'imagesTr')
    LABEL_TR_DIR = os.path.join(BASE_DIR, 'labelsTr')
    IMAGE_VAL_DIR = os.path.join(BASE_DIR, 'imagesVal')
    LABEL_VAL_DIR = os.path.join(BASE_DIR, 'labelsVal')
    
    # 분류 모드 설정 (binary 또는 multi)
    CLASSIFICATION_MODE = 'multi'  # 기본값은 multi 분류
    
    # DL Embedding 특징 설정
    ENABLE_DL_EMBEDDING = True          # DL embedding 특징 사용 여부

    DL_MODEL_TYPE = 'nnunet'            # 'custom' 또는 'nnunet'
    DL_IMG_SIZE = (32, 384, 320)        # DL 모델 입력 이미지 크기 (D, H, W) / nnUNet : (16, 112, 128), (32, 384, 320), Med3D : (56, 448, 448)
    IMG_SIZE_DEPTH, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH = DL_IMG_SIZE
    DL_COMMENT_WRITER = f'{DL_MODEL_TYPE}_{IMG_SIZE_DEPTH}_{IMG_SIZE_HEIGHT}_{IMG_SIZE_WIDTH}'
    FOLD = None                         # None이면 1~5 모든 fold 사용, 숫자면 해당 fold만 사용

    DL_MODEL_PATH = f'./DL_Classification/weights/{DL_COMMENT_WRITER}/{FOLD}/best_model.pth'
    
    # nnUNet 관련 설정 (DL_MODEL_TYPE이 'nnunet'인 경우)
    DL_NNUNET_CONFIG = {
        'plans_file': './DL_Classification/nnUNet/Dataset001_COCA/nnUNetResEncUNetLPlans.json',
        'dataset_json_file': './DL_Classification/nnUNet/Dataset001_COCA/dataset.json',
        'checkpoint_file': './DL_Classification/nnUNet/Dataset001_COCA/checkpoint_final.pth',
        'configuration': '3d_fullres'
    }
    
    # Dilation 설정
    ENABLE_DILATION = True   # Dilation 사용 여부
    DILATION_ITERATIONS = 1   # Dilation 반복 횟수
    
    # 데이터 분할 설정
    DATA_SPLIT_RANDOM_STATE = 42  # 데이터 분할을 위한 랜덤 시드
    TEST_SIZE_RATIO = 0.4         # 테스트 데이터 비율 (0.0 ~ 1.0)
    
    @classmethod
    def get_dl_model_paths(cls):
        """FOLD 설정에 따라 DL 모델 경로 딕셔너리 반환"""
        if cls.FOLD is None:
            # 모든 fold (1~5) 사용
            return {
                fold: f'./DL_Classification/weights/{cls.DL_COMMENT_WRITER}/{fold}/best_model.pth'
                for fold in range(1, 6)
            }
        else:
            # 특정 fold만 사용
            return {
                cls.FOLD: f'./DL_Classification/weights/{cls.DL_COMMENT_WRITER}/{cls.FOLD}/best_model.pth'
            }

    # 모델 하이퍼파라미터
    RANDOM_STATE = 42
    MAX_ITER = 2000
    N_ESTIMATORS = 100
    CV_FOLDS = 5
    
    # ===============================
    # 특징 선택 설정
    # ===============================
    # 사용 가능한 특징 선택 방법들:
    # - 'lasso': L1 정규화 기반 선형 모델 (희소성 유도, 선형 관계)
    # - 'rfe': Recursive Feature Elimination (재귀적 특징 제거, 비선형 관계 고려)
    # - 'univariate': 단변량 통계 검정 (F-test, 빠르고 간단)
    # - 'mutual_info': Mutual Information (상호 정보량, 비선형 관계 포착)
    # - 'random_forest': Random Forest 중요도 기반 (앙상블 안정성)
    # - 'none': 특징 선택 없음 (모든 특징 사용)
    FEATURE_SELECTION_METHOD = 'lasso'
    
    # Lasso 관련 파라미터
    LASSO_ALPHA_COUNT = 100        # 알파 값 후보 개수
    LASSO_TOLERANCE = 1e-3         # 수렴 허용 오차
    FEATURE_THRESHOLD = 1e-5       # 특징 선택 임계값
    
    # RFE (Recursive Feature Elimination) 관련 파라미터
    RFE_N_FEATURES_RATIO = 0.3     # 전체 특징의 30% 선택
    RFE_MIN_FEATURES = 10          # 최소 선택 특징 수
    RFE_MAX_FEATURES = 50          # 최대 선택 특징 수
    RFE_STEP = 1                   # RFE 스텝 크기
    RFE_ESTIMATOR_N_ESTIMATORS = 50  # RFE용 Random Forest 트리 수
    
    # Univariate 통계 검정 관련 파라미터
    UNIVARIATE_K_RATIO = 0.3       # 전체 특징의 30% 선택
    UNIVARIATE_MIN_K = 10          # 최소 선택 특징 수
    UNIVARIATE_MAX_K = 50          # 최대 선택 특징 수
    
    # Mutual Information 관련 파라미터
    MUTUAL_INFO_K_RATIO = 0.3      # 전체 특징의 30% 선택
    MUTUAL_INFO_MIN_K = 10         # 최소 선택 특징 수
    MUTUAL_INFO_MAX_K = 50         # 최대 선택 특징 수
    MUTUAL_INFO_RANDOM_STATE = 42  # 재현성을 위한 랜덤 시드
    
    # Random Forest 중요도 기반 특징 선택 관련 파라미터
    RF_FEATURE_N_ESTIMATORS = 100  # Random Forest 트리 수
    RF_FEATURE_THRESHOLD = 'mean'  # 특징 선택 임계값 ('mean', 'median', 숫자값)
    
    # ===============================
    # 분류 모델 설정
    # ===============================
    # 사용 가능한 분류 모델들:
    # - 'LR': Logistic Regression (로지스틱 회귀)
    # - 'SVM': Support Vector Machine (서포트 벡터 머신)
    # - 'RF': Random Forest (랜덤 포레스트)
    # - 'GB': Gradient Boosting (그래디언트 부스팅)
    # - 'KNN': K-Nearest Neighbors (K-최근접 이웃)
    # - 'NB': Naive Bayes (나이브 베이즈)
    CLASSIFICATION_MODELS = ['LR', 'SVM', 'RF']  # 사용할 모델들을 리스트로 지정
    
    # ===============================
    # 개별 모델 하이퍼파라미터
    # ===============================
    
    # Logistic Regression 파라미터
    LR_MAX_ITER = 2000            # 최대 반복 횟수
    LR_SOLVER = 'liblinear'       # 최적화 알고리즘
    LR_C = 1.0                    # 정규화 강도 역수
    
    # SVM 파라미터
    SVM_C = 1.0                   # 정규화 매개변수
    SVM_KERNEL = 'rbf'            # 커널 함수
    SVM_GAMMA = 'scale'           # 커널 계수
    SVM_PROBABILITY = True        # 확률 예측 활성화
    
    # Random Forest 파라미터
    RF_N_ESTIMATORS = 100         # 트리 개수
    RF_MAX_DEPTH = None           # 트리 최대 깊이
    RF_MIN_SAMPLES_SPLIT = 2      # 분할을 위한 최소 샘플 수
    RF_MIN_SAMPLES_LEAF = 1       # 리프 노드 최소 샘플 수
    RF_MAX_FEATURES = 'sqrt'      # 분할 시 고려할 특징 수
    
    # Gradient Boosting 파라미터
    GB_N_ESTIMATORS = 100         # 부스팅 단계 수
    GB_LEARNING_RATE = 0.1        # 학습률
    GB_MAX_DEPTH = 3              # 트리 최대 깊이
    GB_MIN_SAMPLES_SPLIT = 2      # 분할을 위한 최소 샘플 수
    GB_MIN_SAMPLES_LEAF = 1       # 리프 노드 최소 샘플 수
    GB_SUBSAMPLE = 1.0            # 서브샘플링 비율
    
    # K-Nearest Neighbors 파라미터
    KNN_N_NEIGHBORS = 5           # 이웃 수
    KNN_WEIGHTS = 'uniform'       # 가중치 ('uniform', 'distance')
    KNN_ALGORITHM = 'auto'        # 알고리즘 ('auto', 'ball_tree', 'kd_tree', 'brute')
    KNN_P = 2                     # 거리 계산 파라미터 (1: 맨하탄, 2: 유클리드)
    
    # Naive Bayes 파라미터
    NB_VAR_SMOOTHING = 1e-9       # 분산 스무딩 파라미터
    
    @classmethod
    def _get_dataset_type(cls):
        """BASE_DIR 경로에 따라 데이터셋 종류 결정"""
        if 'Chest' in cls.BASE_DIR:
            return 'chest'
        elif 'Cardiac' in cls.BASE_DIR:
            return 'cardiac'
        else:
            return 'unknown'
    
    @classmethod
    def ensure_output_dir(cls):
        """특징 선택 방법과 실행 시간, 분류 모드, dilation 설정, DL embedding에 따른 출력 디렉토리 생성"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_suffix = "binary" if cls.CLASSIFICATION_MODE == 'binary' else "multi"
        
        # Dilation 정보 추가
        dilation_suffix = ""
        if cls.ENABLE_DILATION:
            dilation_suffix = f"_dil{cls.DILATION_ITERATIONS}"
        
        # DL embedding 정보 추가
        dl_suffix = ""
        if cls.ENABLE_DL_EMBEDDING:
            dl_suffix = f"_dl{cls.DL_MODEL_TYPE}_{cls.IMG_SIZE_DEPTH}_{cls.IMG_SIZE_HEIGHT}_{cls.IMG_SIZE_WIDTH}"
        
        # DL과 dilation이 모두 비활성화된 경우 default 접두사 사용
        if not cls.ENABLE_DL_EMBEDDING and not cls.ENABLE_DILATION:
            final_dir_name = f"default_av_roi_cropped_{timestamp}"
        else:
            final_dir_name = f"{dl_suffix}{dilation_suffix}_av_roi_cropped_{timestamp}".lstrip('_')
        
        # 데이터셋 타입에 따라 하위 디렉토리 결정
        dataset_type = cls._get_dataset_type()
        
        # 4단계 디렉토리 구조: base/dataset_type/feature_method/mode/final_name
        output_dir = os.path.join(
            cls.BASE_OUTPUT_DIR,
            dataset_type,
            cls.FEATURE_SELECTION_METHOD,
            mode_suffix,
            final_dir_name
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    @classmethod
    def get_available_feature_methods(cls):
        """사용 가능한 특징 선택 방법 목록 반환"""
        return ['lasso', 'rfe', 'univariate', 'mutual_info', 'random_forest', 'none']
    
    @classmethod
    def get_available_classification_models(cls):
        """사용 가능한 분류 모델 목록 반환"""
        return ['LR', 'SVM', 'RF', 'GB', 'KNN', 'NB']
    
    @classmethod
    def get_available_classification_modes(cls):
        """사용 가능한 분류 모드 목록 반환"""
        return ['binary', 'multi']
    
    @classmethod
    def print_config_summary(cls):
        """현재 설정 요약 출력"""
        print("=== 현재 설정 요약 ===")
        print(f"분류 모드: {cls.CLASSIFICATION_MODE}")
        print(f"특징 선택 방법: {cls.FEATURE_SELECTION_METHOD}")
        print(f"분류 모델: {cls.CLASSIFICATION_MODELS}")
        print(f"Dilation 사용: {cls.ENABLE_DILATION}")
        if cls.ENABLE_DILATION:
            print(f"Dilation 반복 횟수: {cls.DILATION_ITERATIONS}")
        print(f"DL Embedding 사용: {cls.ENABLE_DL_EMBEDDING}")
        if cls.ENABLE_DL_EMBEDDING:
            print(f"DL 모델 타입: {cls.DL_MODEL_TYPE}")
            print(f"DL 모델 경로: {cls.DL_MODEL_PATH}")
        print(f"랜덤 시드: {cls.RANDOM_STATE}")
        print(f"CV 폴드 수: {cls.CV_FOLDS}")
        print("========================")