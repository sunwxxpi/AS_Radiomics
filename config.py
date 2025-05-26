import os

class Config:
    """프로젝트 설정을 관리하는 클래스"""
    
    # 경로 설정
    BASE_DIR = '/home/psw/AVS-Diagnosis/Dataset003_KMU_Cardiac_AVC'
    LABEL_FILE = './data/AS_CRF_radiomics.csv'
    OUTPUT_DIR = './radiomics_analysis_results/'
    
    IMAGE_TR_DIR = os.path.join(BASE_DIR, 'imagesTr')
    LABEL_TR_DIR = os.path.join(BASE_DIR, 'labelsTr')
    IMAGE_VAL_DIR = os.path.join(BASE_DIR, 'imagesVal')
    LABEL_VAL_DIR = os.path.join(BASE_DIR, 'labelsVal')
    
    # 모델 하이퍼파라미터
    RANDOM_STATE = 42
    MAX_ITER = 2000
    N_ESTIMATORS = 100
    CV_FOLDS = 5
    
    # 특징 선택 설정
    LASSO_ALPHA_COUNT = 100
    LASSO_TOLERANCE = 1e-3
    FEATURE_THRESHOLD = 1e-5
    
    @classmethod
    def ensure_output_dir(cls):
        """출력 디렉토리 생성"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return cls.OUTPUT_DIR