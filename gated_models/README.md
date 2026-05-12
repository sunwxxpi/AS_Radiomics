# Gated Fusion Models

Radiomics와 DL embedding features를 **Gated Mechanism**으로 융합하는 모델 구현체입니다.

## 개요

단순히 Radiomics와 DL features를 concatenate하는 대신, learnable gate를 통해 두 modality를 adaptive하게 융합합니다.

### Gated Fusion 수식

```
h = tanh(W_h [Radiomics; Deep Learning] + b_h)
g = σ(w_g [Radiomics; Deep Learning] + b_g)
F_fused = g ⊗ h
```

여기서:
- `[Radiomics; Deep Learning]`: 두 feature의 concatenation
- `h`: Transform된 표현
- `g`: Gate 값 (0~1 사이의 adaptive weight)
- `⊗`: Element-wise multiplication

## 모델 아키텍처

### 1. GatedFusionLayer (융합 레이어)

**입력**:
- `radiomics_features`: [batch_size, 107] - Radiomics 특징
- `dl_features`: [batch_size, 320] - DL embedding 특징

**내부 구조**:
```
Concatenation: [107] + [320] → [427]
    ↓
Transform Layer (W_h): Linear(427, 427)
    ↓ tanh
h: [batch_size, 427]

Gate Layer (w_g): Linear(427, 427)
    ↓ sigmoid
g: [batch_size, 427]

Element-wise Multiplication: g ⊗ h
    ↓
Dropout(0.3)
```

**출력**:
- `fused_features`: [batch_size, 427] - 융합된 특징

**주요 특징**:
- `fusion_dim=None` (기본값): 출력 차원 = 입력 차원 (427)
- `fusion_dim=N` (지정 시): 출력 차원 = N (차원 축소/확장 가능)
- Xavier uniform 초기화로 학습 안정성 확보

---

### 2. GatedFusionClassifier (Stage 1 학습용 전체 모델)

**입력**:
- `radiomics_features`: [batch_size, 107]
- `dl_features`: [batch_size, 320]

**내부 구조**:
```
GatedFusionLayer: [107] + [320] → [427]
    ↓
MLP Classifier (분류 헤드):
    Linear(427, 256) → BatchNorm1d(256) → ReLU → Dropout(0.3)
    ↓ [batch_size, 256]
    Linear(256, 128) → BatchNorm1d(128) → ReLU → Dropout(0.3)
    ↓ [batch_size, 128]
    Linear(128, 3)
    ↓ [batch_size, 3]
```

**출력**:
- `logits`: [batch_size, 3] - 클래스별 분류 점수 (multi-class 기준)
  - Binary 모드: [batch_size, 2]

**역할**:
- **Stage 1**: Cross Entropy Loss로 end-to-end 학습
  - MLP가 제공하는 supervision 신호로 Gated Fusion Layer 학습
  - 학습 후 모델 저장: `fold_{N}_best_model.pth`
- **Stage 2**: Gated Fusion Layer만 추출하여 특징 생성
  - MLP는 사용하지 않음 (버려짐)
  - 융합된 [427] 특징을 LASSO + ML 분류기로 분석

**파라미터 수**:
```
GatedFusionLayer:
  - Transform: 427 × 427 + 427 = 182,706
  - Gate: 427 × 427 + 427 = 182,706
  - 소계: 365,412

MLP Classifier:
  - Layer 1: 427 × 256 + 256 = 109,568
  - Layer 2: 256 × 128 + 128 = 32,896
  - Layer 3: 128 × 3 + 3 = 387
  - 소계: 142,851

전체: 508,263 파라미터
```

---

### 3. Dimension 흐름 요약

#### Stage 1: Gated Fusion 학습
```
Radiomics [107] ─┐
                 ├─ Concat [427] ─ GatedFusionLayer ─ Fused [427] ─ MLP ─ Logits [3]
DL Embed [320] ──┘                                                            ↓
                                                                    Cross Entropy Loss
```

#### Stage 2: 특징 추출 및 ML 학습
```
Radiomics [107] ─┐
                 ├─ Concat [427] ─ GatedFusionLayer ─ Fused [427]
DL Embed [320] ──┘                  (학습된)              ↓
                                                   LASSO Selection
                                                          ↓
                                                   Selected [~19]
                                                          ↓
                                        ML Classifiers (LR, SVM, RF, GB, KNN, NB)
```

**차원 변화**:
- 입력: Radiomics (107) + DL (320) = **427**
- Gated Fusion 출력: **427** (fusion_dim=None 기본값)
- LASSO 선택 후: **약 19** (데이터에 따라 변동)
- 최종 분류: **3** (multi) 또는 **2** (binary)

## 파일 구조

```
gated_models/
├── __init__.py                    # 패키지 초기화
├── README.md                      # 이 파일
├── gated_model.py                 # Gated Fusion Layer + Classifier 모델
├── gated_trainer.py               # 학습 스크립트
├── gated_feature_extractor.py     # Fused features 추출기
└── gated_pipeline.py              # Gated Fusion 분석 파이프라인
```

## 사용 방법

### 1. 메인 파이프라인 실행

`config.py`에서 설정을 변경한 후 실행:

```python
# config.py
CLASSIFICATION_MODE = 'multi'  # 'binary' 또는 'multi'
USE_GATED_FUSION = True        # True: gated fusion, False: 일반 concat
ENABLE_DL_EMBEDDING = True     # DL embedding 필수
```

```bash
# 설정 후 실행
python main.py
```

이 스크립트는 다음을 자동으로 수행합니다:
1. Radiomics + DL features 추출
2. Fold별 Gated Fusion 모델 학습 (`--use_gated_fusion` 사용 시)
3. Fused features 추출
4. 전통적 ML 분류기 (LR, SVM, RF, GB, KNN, NB) 학습 및 평가
5. LASSO 특징 선택 결과 저장

### 2. 개별 모듈 사용

#### Gated Fusion 모델 생성

```python
from gated_models import GatedFusionClassifier

model = GatedFusionClassifier(
    radiomics_dim=107,      # Radiomics 특징 차원
    dl_dim=320,             # DL embedding 차원
    num_classes=3,          # 분류 클래스 수
    fusion_dim=None,        # None이면 radiomics_dim + dl_dim 사용
    hidden_dims=[256, 128], # Classifier hidden layers
    dropout=0.3
)
```

#### Gated Fusion 학습

```python
from gated_models import GatedFusionTrainer

# 모델 설정
model_config = {
    'radiomics_dim': 107,
    'dl_dim': 320,
    'num_classes': 3,
    'fusion_dim': None,
    'hidden_dims': [256, 128],
    'dropout': 0.3
}

# 학습 설정
train_config = {
    'learning_rate': 1e-4,
    'weight_decay': 1e-2,
    'min_lr': 1e-6,
    'batch_size': 16,
    'epochs': 100,
    'patience': 50
}

# Trainer 초기화 및 학습
trainer = GatedFusionTrainer(
    model_config=model_config,
    train_config=train_config,
    output_dir='./gated_results',
    random_seed=42
)

# 단일 fold 학습 (n_folds=1, 내부적으로 train/val split)
fold_results = trainer.train_with_cv(
    features_df,
    n_folds=1,
    external_fold_idx=1  # fold 번호 지정
)

# 또는 K-fold 교차검증 학습
fold_results = trainer.train_with_cv(features_df, n_folds=5)
```

#### Fused Features 추출

```python
from gated_models import GatedFeatureExtractor

# 학습된 모델 로드
extractor = GatedFeatureExtractor(
    model_path='./gated_results/fold_1_best_model.pth',
    model_config=model_config
)

# Fused features 추출
fused_df = extractor.extract_features_from_dataframe(
    features_df,
    use_fitted_scaler=False
)
```

## 설정 변경

`config.py`에서 다음 설정을 조정:

```python
# config.py

# 필수 설정 (Gated Fusion 사용 시)
ENABLE_DL_EMBEDDING = True     # DL embedding 사용 활성화
USE_GATED_FUSION = True        # Gated Fusion 사용 (False면 일반 concat)

# 분류 모드
CLASSIFICATION_MODE = 'multi'  # 'binary' 또는 'multi'

# DL 모델 설정
DL_MODEL_TYPE = 'nnunet'       # 'nnunet' 또는 'custom'
DL_IMG_SIZE = (32, 384, 320)   # DL 모델 입력 크기

# 특징 선택 방법
FEATURE_SELECTION_METHOD = 'lasso'  # LASSO 특징 선택 사용
```

## 학습 전략

### Two-Stage Learning

1. **Stage 1: Gated Fusion 학습**
   - DL backbone은 frozen (사전 학습된 가중치 사용)
   - Gated Fusion Layer + Classifier만 학습
   - Cross Entropy Loss 최소화 (자동 class weight 적용)
   - 5-fold 교차검증 (DL Classification과 동일한 split 방식)
   - Fold별 독립 학습 (재현성 보장)
   - **Test Set 평가**: imagesVal 데이터로 최종 성능 평가
     - MLP 모델 결과를 model_validation_summary.csv에 자동 병합
     - Confusion Matrix 이미지 생성
     - 예측 확률 포함한 결과 저장

2. **Stage 2: 전통적 ML 분류기 학습**
   - 학습된 Gated Fusion으로 fused features 추출 (427개)
   - LASSO로 특징 선택 수행 (427개 → 약 19개)
   - LR, SVM, RF 등 전통적 분류기 학습 및 비교
   - ML 분류기 결과를 model_validation_summary.csv에 병합
   - 최종 결과: MLP, LR, RF, SVM 순서로 정렬

### 학습 최적화 기법

- **Optimizer**: AdamW (weight decay 적용)
- **Scheduler**: Cosine Annealing (학습률 점진적 감소)
- **Loss Function**: Cross Entropy Loss
  - **자동 클래스 가중치 계산**: 학습 데이터셋의 클래스 분포를 기반으로 가중치 자동 계산
  - 가중치 = 전체 샘플 수 / (클래스 수 × 클래스별 샘플 수)
  - 클래스 불균형 문제 자동 해결
- **Label Smoothing**: 0.1 (과적합 방지)
- **Regularization**: Dropout (0.3), Weight Decay (1e-2)
- **Early Stopping**: Validation loss 기준 (patience=50)
- **재현성**: 모든 random seed 고정 (random_seed=42)

### 재현성 보장

모든 fold에서 완전히 동일한 결과를 보장하기 위해:
- Python random, NumPy, PyTorch random seed 고정 (seed=42)
- CUDA deterministic mode 활성화
- cuDNN benchmark mode 비활성화
- 각 fold 시작 시 seed 재설정 (완전한 재현성)
- StratifiedKFold 사용 (DL Classification과 동일한 분할 방식)

### 레이블 인코딩

클래스 순서를 고정하여 일관성 보장:
- **Multi-class**: ['normal', 'nonsevere', 'severe'] (0, 1, 2)
- **Binary**: ['nonsevere', 'severe'] (0, 1)
- desired_order 기반 자동 정렬
- 데이터에 없는 클래스는 자동 제외

## 결과 저장

학습 결과는 다음 구조로 저장됩니다:

```
radiomics_analysis_results/
└── total/
    └── lasso/
        └── multi/
            └── dlnnunet_32_384_320_gated_YYYYMMDD_HHMMSS/
                ├── fold_1/
                │   ├── fold_1_best_model.pth              # 학습된 Gated Fusion 모델
                │   ├── gated_training.log                 # 학습 로그
                │   ├── gated_fused_features_all.csv       # 전체 fused features
                │   ├── gated_fused_features_train.csv     # 학습용 fused features
                │   ├── gated_fused_features_test.csv      # 테스트용 fused features
                │   ├── lasso_feature_analysis.csv         # LASSO 특징 분석 결과
                │   ├── confusion_matrix_*.png             # Confusion matrix
                │   ├── test_cases_prediction_results.csv  # 예측 결과
                │   └── model_validation_summary.csv       # 성능 요약
                ├── fold_2/
                │   └── ...
                ├── fold_3/
                │   └── ...
                ├── fold_4/
                │   └── ...
                └── fold_5/
                    └── ...
```

### 주요 결과 파일

1. **fold_{N}_best_model.pth**: 학습된 Gated Fusion 모델 체크포인트
2. **gated_training.log**: 학습 과정 상세 로그
3. **model_validation_summary.csv**: MLP 및 ML 분류기 성능 요약 (병합)
   - MLP: Gated Fusion + Classifier (Test Set 평가)
   - LR, RF, SVM: 전통적 ML 분류기 (Validation Set 평가)
4. **MLP_confusion_matrix.png**: MLP 모델의 Confusion Matrix 이미지
5. **gated_fusion_predictions_fold_{N}.csv**: MLP 예측 결과 (확률 포함)
6. **lasso_feature_analysis.csv**: LASSO 특징 선택 결과
7. **gated_fused_features_*.csv**: Fused features 데이터

## 디렉토리 명명 규칙

- **일반 concat 방식**: `dlnnunet_32_384_320_YYYYMMDD_HHMMSS`
- **Gated fusion 방식**: `dlnnunet_32_384_320_gated_YYYYMMDD_HHMMSS`

`_gated` 접미사로 두 방식을 구분합니다.

## 코드 정리

현재 구현은 실제로 사용되는 코드만 포함하고 있습니다:

### 사용 중인 클래스
- `GatedFusionLayer`: 기본 gated fusion 레이어
- `GatedFusionClassifier`: Gated fusion + MLP 분류기
- `GatedFusionTrainer`: 학습 파이프라인
- `GatedFeatureExtractor`: 특징 추출기

### 제거된 실험용 코드
- `MultiHeadGatedFusion`: Multi-head attention 스타일 (미사용)
- `SimpleGatedFusionClassifier`: 단순 버전 (미사용)
- 관련 파라미터: `use_multihead`, `num_heads`, `simple`

## 성능 모니터링

학습 중 다음을 통해 모니터링 가능:

1. **콘솔 출력**: 실시간 loss, accuracy 출력 (tqdm progress bar)
2. **로그 파일**: `gated_training.log`에 상세 기록 (fold별 독립 파일)
3. **모델 체크포인트**: Best validation loss 기준 자동 저장

## 문제 해결

### ImportError 발생 시

```bash
# 프로젝트 루트에서 실행
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python main_gated.py
```

### CUDA Out of Memory

`train_config`에서 배치 크기 감소:

```python
train_config = {
    'batch_size': 8,  # 16 → 8로 감소
    ...
}
```

### BatchNorm 오류 (batch_size=1)

학습 DataLoader에서 `drop_last=True` 설정되어 있어 자동으로 방지됩니다.

### DL 모델 경로 오류

`config.py`에서 DL 모델 경로 확인:

```python
# DL 모델이 학습되어 있는지 확인
DL_MODEL_PATH = f'./DL_Classification/weights/{DL_COMMENT_WRITER}/{FOLD}/best_model.pth'
```

### Fold 번호 불일치

- main.py는 외부 fold 번호를 `external_fold_idx`로 전달
- 모델 파일명, 로그 등이 모두 올바른 fold 번호로 저장됨
- 예: fold_3 디렉토리 → `fold_3_best_model.pth`, "Fold 3" 로그

### 성능 평가 메트릭

**Multi-class 분류** (3-class):
- **AUC**: One-vs-Rest Macro-Average 방식 사용
  - 각 클래스를 이진 분류 문제로 변환
  - 모든 클래스의 AUC 평균값 계산
- **F1-Score**: Macro-Average (모든 클래스 동등하게 취급)
- **AP (Average Precision)**: Macro-Average 방식 사용

**Binary 분류** (2-class):
- AUC, AP: 양성 클래스(severe) 기준 계산
- F1-Score: Binary 방식

## 통합된 파이프라인

`main.py`는 이제 일반 concat 방식과 gated fusion 방식을 모두 지원합니다:

- **일반 Concat 방식**: `config.py`에서 `USE_GATED_FUSION = False` (기본값)
- **Gated Fusion 방식**: `config.py`에서 `USE_GATED_FUSION = True`

두 방식 모두 동일한 코드베이스를 사용하며, `config.py`의 설정만 변경하면 됩니다.

## 참고 자료

- 통합 파이프라인: `main.py`
- DL Classification: `DL_Classification/`
- 전체 프로젝트 문서: `CLAUDE.md`