import os
import datetime

class Config:
    """프로젝트 설정을 관리하는 클래스"""
    
    # 경로 설정
    BASE_DIR = './data/datasets/Dataset004_mix_KUDH0467rm_cropped'
    LABEL_FILE = './data/AS_CRF.csv'
    BASE_OUTPUT_DIR = './radiomics_analysis_results'
    
    IMAGE_TR_DIR = os.path.join(BASE_DIR, 'imagesTr')
    LABEL_TR_DIR = os.path.join(BASE_DIR, 'labelsTr')
    IMAGE_VAL_DIR = os.path.join(BASE_DIR, 'imagesVal')
    LABEL_VAL_DIR = os.path.join(BASE_DIR, 'labelsVal')
    
    # 데이터 분할 설정
    DATA_SPLIT_MODE = 'fix'    # 분할 모드: 'random' (병합 후 랜덤 분할) 또는 'fix' (디렉토리 기반 고정 분할)
    DATA_SPLIT_RANDOM_STATE = 42  # 데이터 분할을 위한 랜덤 시드 (random 모드에서만 사용)
    TEST_SIZE_RATIO = 0.2         # 테스트 데이터 비율 (random 모드에서만 사용, 0.0 ~ 1.0)
    
    # 분류 모드 설정
    CLASSIFICATION_MODE = 'multi'       # 'binary' 또는 'multi'

    # 특징 융합 설정
    ENABLE_DL_EMBEDDING = False          # DL embedding 사용 여부
    USE_ENSEMBLE = False                 # Soft Voting Ensemble 사용 여부
    USE_GATED_FUSION = False             # True: Gated Fusion, False: 일반 Concat

    # DL 모델 설정
    DL_MODEL_TYPE = 'nnunet'            # 'nnunet' 또는 'custom'
    DL_IMG_SIZE = (32, 384, 320)        # 입력 이미지 크기 (D, H, W): nnUNet(32,384,320), Med3D(56,448,448)
    IMG_SIZE_DEPTH, IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH = DL_IMG_SIZE
    DL_NUM_FOLDS = 5                    # DL cross-fitting fold 수. dl_cls_train.py 의 --fold 가 다른 값이면 시작 전에 멈춘다
    # 옛 split 으로 학습한 산출물과 가중치 경로가 겹치지 않게 데이터셋 이름을 붙인다.
    DL_DATASET_TAG = os.path.basename(os.path.normpath(BASE_DIR))
    DL_COMMENT_WRITER = f'{DL_MODEL_TYPE}_{IMG_SIZE_DEPTH}_{IMG_SIZE_HEIGHT}_{IMG_SIZE_WIDTH}_{DL_DATASET_TAG}'
    # DL 학습이 가중치를 저장하는 곳과 파이프라인이 읽는 곳이 갈리지 않게 한 자리에서 정한다.
    DL_WEIGHTS_ROOT = './DL_Classification/weights'
    
    # nnUNet 관련 설정 (DL_MODEL_TYPE이 'nnunet'인 경우)
    DL_NNUNET_CONFIG = {
        'plans_file_arch': './DL_Classification/nnUNet/COCA_nnUNetResEncUNetLPlans.json',
        'configuration': '3d_fullres',
        'dataset_json_file': './DL_Classification/nnUNet/COCA_dataset.json',
        'checkpoint_file': './DL_Classification/nnUNet/COCA_checkpoint_final.pth',
        'plans_file_norm': './DL_Classification/nnUNet/AVC_nnUNetResEncUNetLPlans.json'
    }
    
    # Radiomics 추출 설정
    # PyRadiomics 의 resampledPixelSpacing 은 [x, y, z] 순서 (nnUNet plans 의 [z, y, x] 와 반대).
    # in-plane 값은 이 코호트의 segmentation 에 쓴 nnU-Net planner 가 결정한 spacing 이고,
    # z 는 전 증례 3.0 mm 로 동일해 보간 오차만 늘리므로 원본을 유지한다.
    RESAMPLED_SPACING = [0.3828125, 0.3828125, 3.0]   # None 이면 리샘플링 미적용
    RESAMPLE_INTERPOLATOR = 'sitkLinear'              # 석회화 경계에서 B-spline 은 overshoot 을 만든다

    # Dilation 설정
    ENABLE_DILATION = False   # Dilation 사용 여부
    DILATION_ITERATIONS = 1   # Dilation 반복 횟수

    # 리샘플링 후 선·점으로 축퇴해 PyRadiomics 가 거부하는 마스크만 팽창시켜 한 번 재시도한다.
    # 0 이면 그대로 실패시킨다. ENABLE_DILATION 과 달리 실패한 케이스에만 걸린다.
    DEGENERATE_MASK_DILATION_ITERATIONS = 1
    
    @classmethod
    def get_dl_model_paths(cls):
        """fold 별 가중치 경로를 `{fold 번호: 경로}` 로 만든다.

        development 행의 OOF embedding 은 케이스마다 자기를 검증으로 뺀 fold 를 골라 쓰므로 `DL_NUM_FOLDS` 개가 다 있어야 한다.
        """
        return {
            fold: f'{cls.DL_WEIGHTS_ROOT}/{cls.DL_COMMENT_WRITER}/{fold}/best_model.pth'
            for fold in range(1, cls.DL_NUM_FOLDS + 1)
        }

    @classmethod
    def get_dl_refit_model_path(cls):
        """development 322 전체로 다시 학습한 가중치 경로.

        test 추론과 test 행 embedding 은 이 하나만 쓴다. development 행은 그 케이스를 학습에 안 쓴 fold 모델로 뽑는다.
        """
        return f'{cls.DL_WEIGHTS_ROOT}/{cls.DL_COMMENT_WRITER}/refit/refit_model.pth'

    @classmethod
    def get_dl_fold_assignment_path(cls):
        """development 케이스가 어느 fold 의 검증이었는지 적힌 CSV 경로.

        OOF embedding 은 이 배정으로 케이스마다 모델을 고르므로, DL 학습이 남긴 파일과 짝이 맞지 않으면 조용히 다른 분할이 된다.
        """
        return f'{cls.DL_WEIGHTS_ROOT}/{cls.DL_COMMENT_WRITER}/cls_fold_assignment.csv'

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
    # - 'MLP1': Multilayer Perceptron, 은닉층 1개
    # - 'MLP2': Multilayer Perceptron, 은닉층 2개
    # - 'SVM': Support Vector Machine (서포트 벡터 머신)
    # - 'RF': Random Forest (랜덤 포레스트)
    # - 'GB': Gradient Boosting (그래디언트 부스팅)
    # - 'KNN': K-Nearest Neighbors (K-최근접 이웃)
    # - 'NB': Naive Bayes (나이브 베이즈)
    CLASSIFICATION_MODELS = ['LR', 'MLP1', 'MLP2']  # 사용할 모델들을 리스트로 지정
    
    # ===============================
    # 개별 모델 하이퍼파라미터
    # ===============================
    
    # Logistic Regression 파라미터
    LR_MAX_ITER = 2000            # 최대 반복 횟수
    LR_SOLVER = 'liblinear'       # 최적화 알고리즘
    LR_C = 1.0                    # 정규화 강도 역수
    
    # MLP 파라미터 (MLP1 · MLP2 공용, 은닉층 구성만 다르다)
    # sklearn 은 조기 종료용 검증 분할을 가중치 초기화 뒤 난수 상태로 자르므로, 층 구성이 다르면 두 후보의 검증 행이 갈린다.
    # 조기 종료를 끈 것은 세 후보가 같은 행을 보고 학습하게 하려는 것이고, 후보 선택은 development CV 가 한다.
    MLP1_HIDDEN_LAYER_SIZES = (128,)       # 은닉층 1개
    MLP2_HIDDEN_LAYER_SIZES = (128, 128)   # 은닉층 2개. 폭을 MLP1 과 맞춰 두 후보의 구조 차이를 은닉층 수 하나로 줄인다
    MLP_ACTIVATION = 'relu'       # 은닉층 활성화 함수
    MLP_SOLVER = 'adam'           # 최적화 알고리즘
    MLP_ALPHA = 1.0               # L2 정규화 강도. 튜닝한 값이 아니라 두 후보에 같이 거는 고정값이다
    MLP_LEARNING_RATE_INIT = 1e-3 # 초기 학습률
    MLP_MAX_ITER = 2000           # 최대 epoch 수
    MLP_EARLY_STOPPING = False    # 켜면 학습 데이터에서 검증분을 떼어 조기 종료
    MLP_VALIDATION_FRACTION = 0.1 # 조기 종료용 검증 비율, MLP_EARLY_STOPPING 이 True 일 때만 쓰인다
    MLP_N_ITER_NO_CHANGE = 20     # 손실이 이만큼 개선 없이 지나면 종료 (조기 종료를 켜면 검증 점수 기준)

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
            return cls.BASE_DIR.split('_', 1)[1]
    
    @classmethod
    def ensure_output_dir(cls):
        """실행 설정에 따른 출력 디렉토리 생성

        디렉토리 구조: base/dataset_type/feature_method/mode/final_name
        예시: radiomics_analysis_results/total/lasso/multi/dlnnunet_32_384_320_gated_20250930_123456/

        Returns:
            str: 생성된 출력 디렉토리 경로
        """
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
            # Ensemble 사용 시 _ensemble 접미사 추가
            if cls.USE_ENSEMBLE:
                dl_suffix += "_ensemble"
            # Gated fusion 사용 시 _gated 접미사 추가
            if cls.USE_GATED_FUSION:
                dl_suffix += "_gated"

        # DL embedding과 Dilation이 모두 비활성화된 경우 default 접두사 사용
        if not cls.ENABLE_DL_EMBEDDING and not cls.ENABLE_DILATION:
            final_dir_name = f"default_{timestamp}"
        else:
            final_dir_name = f"{dl_suffix}{dilation_suffix}_{timestamp}".lstrip('_')
        
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
        return ['LR', 'MLP1', 'MLP2', 'SVM', 'RF', 'GB', 'KNN', 'NB']
    
    @classmethod
    def get_available_classification_modes(cls):
        """사용 가능한 분류 모드 목록 반환"""
        return ['binary', 'multi']
    
    @classmethod 
    def get_available_data_split_modes(cls):
        """사용 가능한 데이터 분할 모드 목록 반환"""
        return ['random', 'fix']
    
    @classmethod
    def print_config_summary(cls):
        """현재 설정 요약 출력"""
        print("=== 현재 설정 요약 ===")
        print(f"분류 모드: {cls.CLASSIFICATION_MODE}")
        print(f"데이터 분할 모드: {cls.DATA_SPLIT_MODE}")
        if cls.DATA_SPLIT_MODE == 'random':
            print(f"테스트 데이터 비율: {cls.TEST_SIZE_RATIO}")
            print(f"랜덤 시드: {cls.DATA_SPLIT_RANDOM_STATE}")
        print(f"특징 선택 방법: {cls.FEATURE_SELECTION_METHOD}")
        print(f"분류 모델: {cls.CLASSIFICATION_MODELS}")
        if cls.RESAMPLED_SPACING is None:
            print("Radiomics 리샘플링: 미적용 (원본 spacing)")
        else:
            print(f"Radiomics 리샘플링: {cls.RESAMPLED_SPACING} mm [x, y, z], interpolator={cls.RESAMPLE_INTERPOLATOR}")
        print(f"Dilation 사용: {cls.ENABLE_DILATION}")
        if cls.ENABLE_DILATION:
            print(f"Dilation 반복 횟수: {cls.DILATION_ITERATIONS}")
        if cls.DEGENERATE_MASK_DILATION_ITERATIONS > 0:
            print(f"축퇴 마스크 팽창 재시도: {cls.DEGENERATE_MASK_DILATION_ITERATIONS}회")
        else:
            print("축퇴 마스크 팽창 재시도: 미적용")
        print(f"DL Embedding 사용: {cls.ENABLE_DL_EMBEDDING}")
        if cls.ENABLE_DL_EMBEDDING:
            fusion_type = "Gated Fusion" if cls.USE_GATED_FUSION else "일반 Concat"
            print(f"융합 방식: {fusion_type}")
            print(f"DL 모델 타입: {cls.DL_MODEL_TYPE}")
            print(f"DL 이미지 크기: {cls.DL_IMG_SIZE}")
            print("DL embedding 출처: development=fold 1-5 OOF, test=refit")
        print(f"Soft Voting Ensemble 사용: {cls.USE_ENSEMBLE}")
        print(f"CV 폴드 수: {cls.CV_FOLDS}")
        print("========================")