# AS_Radiomics

**대동맥 판막 협착증(Aortic Stenosis) 진단을 위한 하이브리드 의료 영상 분석 시스템**

Radiomics 특징 추출과 딥러닝 임베딩을 결합하여 3D 심장 영상 데이터를 분석하고 AS 중증도를 분류하는 프로젝트입니다.

## 주요 특징

- **하이브리드 접근법**: Handcrafted Radiomics + 딥러닝 임베딩 결합
- **Multi-fold DL 임베딩**: fold 별 DL 가중치를 바꿔 5회 반복 평가 (데이터 분할은 고정 hold-out — 교차검증 아님)
- **이중 분류 모드**: Binary (nonsevere/severe) 및 Multi-class (normal/nonsevere/severe) 분류 지원
- **nnUNet 통합**: 사전 훈련된 nnUNet 인코더 활용 가능
- **Gated Fusion**: Learnable gate를 통한 adaptive feature fusion
- **Soft Voting Ensemble**: DL + ML 모델 앙상블
- **유연한 특징 선택**: LASSO, RFE, Univariate, Mutual Info, Random Forest 지원

## 빠른 시작

### 메인 파이프라인 실행

```bash
# Multi-class 분류 모드로 전체 파이프라인 실행
python main.py
```

### 딥러닝 분류 모델 학습

```bash
# nnUNet 인코더 사용 (권장) — 경로가 cwd 상대이므로 저장소 루트에서 실행
python DL_Classification/dl_cls_train.py --model_type nnunet --img_size "(32, 384, 320)"

# Custom ResNet50 사용
python DL_Classification/dl_cls_train.py --model_type custom --img_size "(56, 448, 448)"
```

### 딥러닝 모델 테스트 및 시각화

```bash
# Grad-CAM 시각화 포함 테스트 — 저장소 루트에서 실행
python DL_Classification/dl_cls_test.py --model_type nnunet --img_size "(32, 384, 320)" --enable_cam
```

## 핵심 설정 (`config.py`)

### 분류 모드

```python
CLASSIFICATION_MODE = 'multi'  # 'binary' 또는 'multi'
```

### DL Embedding 설정

```python
ENABLE_DL_EMBEDDING = False    # DL embedding 사용 여부 (기본값 False; 사용하려면 True)
DL_MODEL_TYPE = 'nnunet'       # 'nnunet' 또는 'custom'
DL_IMG_SIZE = (32, 384, 320)   # nnUNet 권장: (32, 384, 320)
```

### 특징 융합 방식

```python
USE_GATED_FUSION = False       # True: Gated Fusion, False: 일반 Concat
USE_ENSEMBLE = False           # Soft Voting Ensemble 사용 여부
```

### 특징 선택 방법

```python
FEATURE_SELECTION_METHOD = 'lasso'  # 'lasso', 'rfe', 'univariate', 'mutual_info', 'random_forest', 'none'
```

### 데이터 분할 설정

```python
DATA_SPLIT_MODE = 'fix'        # 'random' 또는 'fix' (디렉토리 기반 고정 분할)
TEST_SIZE_RATIO = 0.2          # random 모드에서만 사용
DATA_SPLIT_RANDOM_STATE = 42   # random 모드에서만 사용
```

## 프로젝트 구조

```
AS_Radiomics/
├── config.py                          # 전역 설정 관리
├── main.py                            # 메인 파이프라인
├── data/                              # 데이터 로딩 및 전처리
│   ├── loader.py
│   ├── preprocessor.py
│   └── AS_CRF.csv                     # 환자 레이블 파일
├── trainer/                           # 특징 추출 및 모델 학습
│   ├── features_extractor.py
│   ├── dl_embedding_extractor.py
│   ├── feature_selector.py
│   ├── model_factory.py
│   └── train.py
├── DL_Classification/                 # 딥러닝 분류 모듈
│   ├── dl_cls_train.py                # DL 모델 학습
│   ├── dl_cls_test.py                 # DL 모델 테스트
│   ├── dl_cls_cam.py                  # Grad-CAM 시각화
│   ├── dl_cls_model.py                # 3D CNN 모델 정의
│   ├── dl_cls_dataset.py              # 데이터로더
│   ├── dl_cls_config.py               # 설정 및 파싱
│   ├── dl_cls_valid.py                # 성능 평가
│   └── nnUNet/                        # nnUNet 설정 파일
├── gated_models/                      # Gated Fusion 모델
│   ├── gated_model.py                 # Gated Fusion 레이어 및 분류기
│   ├── gated_trainer.py               # 학습 스크립트
│   ├── gated_feature_extractor.py     # 특징 추출기
│   ├── gated_pipeline.py              # 파이프라인
│   └── README.md                      # Gated Fusion 상세 문서
├── utils/                             # 유틸리티 모듈
│   ├── plotter.py                     # 결과 시각화
│   ├── file_handler.py                # 파일 저장 및 관리
│   ├── logger.py                      # 로깅 시스템
│   ├── data_splitter.py               # 데이터 분할
│   └── ensemble.py                    # Soft Voting Ensemble
├── docs/                              # 상세 문서 (데이터셋/아키텍처/워크플로우/설정/결과)
├── CLAUDE.md                          # 코딩 에이전트용 하드 컨스트레인트
└── radiomics_analysis_results/        # 분석 결과 저장
```

## 워크플로우

### 1. Radiomics 특징 추출

- PyRadiomics를 사용하여 handcrafted 특징 추출
- imagesTr과 imagesVal 디렉토리에서 독립적으로 추출 후 병합
- Dilation 옵션 지원

### 2. DL Embedding 추출 (선택)

- Fold별 사전 훈련된 DL 모델에서 고차원 특징 추출
- nnUNet 또는 Custom ResNet50 모델 지원

### 3. 특징 융합

#### 일반 Concat 방식 (`USE_GATED_FUSION = False`)
- Radiomics + DL features를 단순 concatenation

#### Gated Fusion 방식 (`USE_GATED_FUSION = True`)
- Learnable gate를 통한 adaptive fusion
- Two-stage learning:
  1. Stage 1: Gated Fusion Layer + MLP Classifier 학습 → imagesVal(test set)로 MLP 최종 성능 평가
  2. Stage 2: Fused features 추출 → 전통적 ML 분류기 학습
     (`fusion_dim` 미지정 시 radiomics + DL 차원 합. nnUNet 기본 설정에서는 107 + 320 = 427)
- Gated Fusion 이 켜져 있으면 Ensemble 은 실행되지 않는다 (`main.py:208-210` 에서 조기 반환)

### 4. 특징 선택

- **LASSO**: L1 정규화 기반 (희소성 유도)
- **RFE**: Recursive Feature Elimination
- **Univariate**: F-test 기반 단변량 검정
- **Mutual Info**: 상호 정보량 기반
- **Random Forest**: 특징 중요도 기반

### 5. 모델 학습 및 평가

- 전통적 ML 분류기: LR, SVM, RF, GB, KNN, NB (기본값은 `CLASSIFICATION_MODELS = ['LR', 'SVM', 'RF']`)
- 학습/평가는 train/test hold-out 1회. `CV_FOLDS = 5` 는 LassoCV 등 특징 선택 내부 CV 에만 쓰인다
- 성능 메트릭: Accuracy, F1-Score, AUC, AP

### 6. Soft Voting Ensemble (선택)

- DL 모델과 ML 모델들의 확률값 결합
- DL+LR, DL+RF, DL+SVM 조합
- Macro-average AUC, AP 계산

## 주요 기능

### 1. 자동 클래스 가중치 계산

클래스 불균형 문제를 자동으로 해결:
```python
# Cross Entropy Loss에 자동 적용
weights = total_samples / (num_classes * class_counts)
```

### 2. 레이블 순서 고정

일관된 클래스 순서 보장:
- **Multi-class**: ['normal', 'nonsevere', 'severe'] (0, 1, 2)
- **Binary**: ['nonsevere', 'severe'] (0, 1)

### 3. Macro-Average AUC 계산

Multi-class 분류에서 One-vs-Rest 방식 사용:
```python
# 각 클래스를 이진 분류 문제로 변환
y_true_bin = label_binarize(y_true, classes=range(n_classes))
auc_score = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
```

### 4. Grad-CAM 시각화

모델 해석 가능성 향상 (`--enable_cam` 지정 시, 최대 20 샘플 × 3장):
- `{sample}_key_slices.png`: CAM 반응이 큰 상위 12개 슬라이스
- `{sample}_all_slices.png`: 전체 슬라이스 그리드
- `{sample}_3d_projection.png`: Axial/Coronal/Sagittal 3방향 최대 강도 투영(MIP)
- 저장 위치: `DL_Classification/results/{writer_comment}/cam_visualization/fold_{n}/`

## 데이터 구조

### 환자 레이블 파일 (`data/AS_CRF.csv`)

총 120개 컬럼이지만 코드가 읽는 것은 아래 3개뿐이다 (`data/loader.py:38`).

```csv
1차년도연구번호,...,AV_binaryclassification,AS ,...
patient001,...,nonsevere,none,...
patient002,...,severe,severe,...
patient003,...,nonsevere,mild,...
```

- `AS ` 는 컬럼명 **끝에 공백이 하나** 붙어 있고, `1차년도연구번호` 는 파일 선두에 BOM 이 있다. 컬럼명 비교 시 주의.

### 파일 명명 규칙

- **이미지 파일**: `{patient_id}_{sequence}_0000.nii.gz`
- **레이블 파일**: `{patient_id}_{sequence}.nii.gz`

### 레이블 매핑

**Binary 모드**:
```python
# AV_binaryclassification 컬럼 사용
'nonsevere' → 0
'severe' → 1
```

**Multi-class 모드**:
```python
# AS 컬럼 변환
'none', 'no' → 'normal' (0)
'mild', 'moderate', 'pseudosevere' → 'nonsevere' (1)
'severe', 'very severe' → 'severe' (2)
```

## 결과 구조

```
radiomics_analysis_results/
└── total/
    └── lasso/
        └── multi/
            ├── dlnnunet_32_384_320_gated_20250930_123456/   # USE_GATED_FUSION = True
            │   ├── log.txt                                  # 실행 로그 (최상단에만 생성)
            │   ├── fold_1/
            │   │   ├── fold_1_best_model.pth                # Gated Fusion 모델
            │   │   ├── gated_training.log                   # Gated 학습 로그
            │   │   ├── gated_fusion_predictions_fold_1.csv  # MLP 예측 결과
            │   │   ├── gated_fused_features_all.csv         # fused features (+ _train/_test)
            │   │   ├── model_validation_summary.csv         # MLP/LR/RF/SVM 성능 요약
            │   │   ├── {model}_confusion_matrix.png         # Confusion Matrix
            │   │   ├── test_cases_prediction_results.csv    # 예측 결과
            │   │   └── lasso_feature_analysis.csv           # LASSO 분석
            │   ├── fold_2/
            │   └── ...
            └── dlnnunet_32_384_320_ensemble_20250930_123456/  # USE_ENSEMBLE = True (Gated 와 배타)
                └── fold_1/
                    ├── radiomics_features_all.csv           # (+ _train/_test)
                    ├── model_validation_summary.csv
                    ├── test_cases_prediction_results.csv
                    └── ensemble/                            # Ensemble 결과
                        ├── ensemble_results_fold_1.csv
                        └── ensemble_model_validation_summary.csv
```

## 고급 기능

### Gated Fusion 모델

Radiomics와 DL features를 adaptive하게 융합:
```
h = tanh(W_h [Radiomics; Deep Learning] + b_h)
g = σ(w_g [Radiomics; Deep Learning] + b_g)
F_fused = g ⊗ h
```

자세한 내용은 [gated_models/README.md](gated_models/README.md) 참조

### Soft Voting Ensemble

DL과 ML 모델의 확률값을 평균하여 최종 예측:
```python
# DL+LR 앙상블 예시
ensemble_proba = (DL_proba + LR_proba) / 2
predicted_class = argmax(ensemble_proba)
```

## 의존성

`requirements.txt` / `pyproject.toml` 은 없다. 아래 라이브러리를 직접 설치해야 한다.

- `torch`, `tensorboard`: 딥러닝 프레임워크 및 학습 로깅
- `monai`: 의료 영상 딥러닝 (Custom ResNet50 백본)
- `nnunetv2`: nnUNet encoder 로딩
- `pyradiomics`, `nibabel`, `scipy`: Radiomics 특징 추출 및 마스크 Dilation
- `scikit-learn`: 전통적 ML 및 평가
- `pandas`, `numpy`: 데이터 처리
- `matplotlib`, `seaborn`: 시각화
- `tqdm`, `natsort`: 진행 표시 및 파일 정렬

## 성능 평가 메트릭

### Multi-class (3-class)
- **Accuracy**: 전체 정확도
- **F1-Score**: Macro-average (모든 클래스 동등)
- **AUC**: One-vs-Rest Macro-average
- **AP**: Average Precision (Macro-average)

### Binary (2-class)
- **Accuracy**: 전체 정확도
- **F1-Score**: Binary 방식
- **AUC**: 양성 클래스(severe) 기준
- **AP**: 양성 클래스(severe) 기준

## 문제 해결

### DL 모델 경로 오류
```python
# config.py의 get_dl_model_paths()가 {fold: 경로} 딕셔너리를 반환한다.
# 포맷: ./DL_Classification/weights/{DL_COMMENT_WRITER}/{fold}/best_model.pth
Config.get_dl_model_paths()
```

### CUDA Out of Memory
```python
# Gated Fusion: gated_models/gated_pipeline.py 의 train_config 에서 배치 크기 감소
train_config = {
    'batch_size': 8,  # 16 → 8로 감소
    ...
}
```
DL Classification 학습은 CLI 로 조정한다: `--batch_size 1` (기본값 2).

### Ensemble/Gated Fusion 사용 시
```python
# DL Embedding이 활성화되어야 함
ENABLE_DL_EMBEDDING = True
USE_ENSEMBLE = True  # 또는 USE_GATED_FUSION = True
```

## 참고 자료

- 상세 문서 인덱스: [docs/README.md](docs/README.md)
  - 데이터셋 카탈로그·클래스 분포: [docs/DATASET.md](docs/DATASET.md)
  - 모듈 의존성·데이터 흐름: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
  - 실행 시나리오·트러블슈팅: [docs/WORKFLOWS.md](docs/WORKFLOWS.md)
  - `Config` 전수 레퍼런스: [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)
  - 결과 디렉토리/CSV 사양: [docs/RESULTS_LAYOUT.md](docs/RESULTS_LAYOUT.md)
  - 실험 설계 검토 노트 (제안 단계, 코드 미반영): [docs/notes/](docs/notes/)
- 코딩 에이전트용 규약: [CLAUDE.md](CLAUDE.md)
- Gated Fusion 상세: [gated_models/README.md](gated_models/README.md)
- DL Classification: `DL_Classification/` 디렉토리

## 라이선스

이 프로젝트는 연구 목적으로만 사용됩니다.

## 문의

프로젝트 관련 문의사항은 이슈를 통해 남겨주세요.
