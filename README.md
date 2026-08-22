# AS_Radiomics

심장 CT 로 대동맥 판막 협착증(AS) 중증도를 `normal` / `nonsevere` / `severe` 3클래스로 분류한다.
nnU-Net 으로 대동맥 판막 석회화(AVC) 마스크를 얻고, 그 마스크 ROI 에서 뽑은 handcrafted radiomics 와 3D CNN 임베딩을 융합해 분류기를 학습한다.

## 사용법

모든 스크립트는 저장소 루트에서 실행한다. 기본 경로가 cwd 상대라 하위 디렉토리에서 실행하면 어긋난다.

```bash
python main.py    # Radiomics + ML 파이프라인. 설정은 config.py 를 직접 고친다 (CLI 인자 없음)

python DL_Classification/dl_cls_train.py
python DL_Classification/dl_cls_test.py --enable_cam

tensorboard --logdir DL_Classification/logs/{writer_comment}
```

`{writer_comment}` 는 `config.py` 의 `Config.DL_COMMENT_WRITER` 하나가 정한다 — `{model_type}_{D}_{H}_{W}_{BASE_DIR 폴더 이름}` 이다.
`main.py` 도 이 이름으로 DL 산출물을 찾는다. DL 스크립트의 모델·입력 크기·경로·fold 수·분할 인자는 `config.py` 의 같은 값을 기본값으로 받고, 다르게 주면 시작 전에 멈춘다.

`dl_cls_train.py` 는 5-fold 를 돌린 뒤 development 전체로 refit 까지 한다 — GPU 학습 6회다. `--stage folds` / `--stage refit` 로 끊어 돌릴 수 있다.
refit 종료 epoch 은 fold 별 best epoch 의 중앙값이라, `weights/{writer_comment}/fold_best_epochs.csv` 에 fold 가 `DL_NUM_FOLDS` 만큼 다 있어야 `--stage refit` 이 돈다.
`--resume` 은 그 기록과 `best_model.pth` 가 둘 다 있는 fold 만 건너뛴다. 기록에 적힌 학습 인자나 `cls_fold_assignment.csv` 의 배정이 지금과 다르면 덮어쓰기 전에 멈춘다.
이어 돌릴 때는 fold 를 돌릴 때 쓴 인자를 그대로 다시 넘기고, 처음부터 돌릴 때는 그 가중치 디렉토리를 먼저 지운다.

`dl_cls_test.py` 도 같은 `--stage` 를 받는다. 융합 갈래가 읽는 DL 확률은 refit 것(`results/{writer_comment}/probs/refit.csv`) 하나다.
그 가중치가 없으면 그 자리에서 멈춘다. fold 5개 평가는 DL 팔 내부 점검용이라 가중치가 없는 fold 만 건너뛴다.

`main.py` 의 갈래는 `config.py` 의 세 플래그로 정한다.

| 갈래 | `ENABLE_DL_EMBEDDING` | `USE_GATED_FUSION` | `USE_ENSEMBLE` | 선행 조건 |
| --- | --- | --- | --- | --- |
| Radiomics 단독 | False | False | False | 없음 |
| Concat fusion | True | False | False | fold 5개 + refit 가중치 + fold 배정 CSV |
| Gated fusion | True | True | False | fold 5개 + refit 가중치 + fold 배정 CSV |
| Soft voting ensemble | True | False | True | 위에 더해 `dl_cls_test.py` 가 만든 `probs/refit.csv` |

- DL 임베딩은 케이스마다 출처가 다르다. development 행은 그 행을 검증으로 뺀 fold 모델(OOF)로, test 행은 refit 모델로 뽑는다. 임베딩은 한 벌이다.
  development 행에 그 행을 학습에 쓴 모델의 임베딩을 주면 융합 분류기가 test 에는 없을 과적합된 표현 위에서 학습된다.
  배정은 `weights/{writer_comment}/cls_fold_assignment.csv` 를 따르고, 산출물이 하나라도 없거나 배정과 어긋나는 케이스가 있으면 추출 전에 멈춘다.
  전부 없을 때도 갈래를 바꾸지 않는다. Radiomics 단독으로 돌릴 생각이면 `ENABLE_DL_EMBEDDING` 을 직접 내린다.
- 데이터 분할은 고정 hold-out 이고 교차검증이 아니다.
- Gated 를 켜면 Ensemble 은 실행되지 않는다 — `main.py` 가 gated 분석 직후 반환한다.
- Gated 는 two-stage 다. Stage 1 이 `GatedFusionLayer` + MLP 를 학습하고, Stage 2 가 fused feature(radiomics 107 + DL 320 = 427)로 LR/MLP1/MLP2 를 학습한다.
  Stage 1 이 조기 종료를 판단할 검증 fold 가 필요해 gated 갈래만 자기 5-fold 를 돌아 결과가 다섯 벌 나온다. 나머지 갈래는 한 벌이다.
  `model_validation_summary.csv` 한 표에 두 stage 결과가 섞여 있다 — `GatedMLP` 이 Stage 1 의 torch 헤드고, `LR`/`MLP1`/`MLP2` 는 Stage 2 의 sklearn 분류기다.

## 데이터

영상·마스크는 `data/datasets/`, 원본 DICOM 은 `data/datasets_raw/` 에 있고 둘 다 git 밖이다.
`Config.BASE_DIR` 이 가리키는 폴더가 파이프라인이 읽는 데이터셋이다.

### 디렉토리와 파일명

```
Dataset{NNN}_{name}/
├── imagesTr/   {patient_id}_{sequence}_0000.nii.gz   # 1채널 CT, _0000 은 입력 채널 0
├── labelsTr/   {patient_id}_{sequence}.nii.gz        # binary mask (foreground=1)
├── imagesVal/, labelsVal/                            # 동일 패턴
└── crop_window.csv                                   # cropped 데이터셋만. 창 좌표와 마스크 손실 기록
```

`patient_id` 는 `KUDH0001` 형식, `sequence` 는 4자리 이상 숫자다.
같은 환자라도 데이터셋마다 시퀀스 번호가 다르므로 데이터셋 간 조인은 파일명이 아니라 `patient_id` 로 한다.

분류에서 `imagesVal` 은 validation 이 아니라 **hold-out test** 다. `DATA_SPLIT_MODE='fix'` 는 디렉토리 위치로만 분할을 정한다.

### 데이터셋

| 폴더 | imagesTr | imagesVal | 마스크 | 용도 |
| --- | ---: | ---: | --- | --- |
| `Dataset004_mix_KUDH0467rm_cropped` | 322 | 83 | 예측 | **분류 메인 데이터셋** (예측 마스크 centroid 창으로 crop) |
| `Dataset004_gt_cropped` | 167 | 83 | GT | 마스크 출처 ablation 의 GT 팔 (GT centroid 창, 250건) |
| `Dataset004_mix_KUDH0467rm`, `Dataset004_gt` | | | 예측 / GT | 위 두 벌의 crop 전 원본 해상도 |
| `Dataset001_KMU_Cardiac_AVC_TRAIN_ONLY` | 250 | — | GT | GT 마스크 250건의 원본. GT 팔이 여기서 나온다 |

`_cropped` 두 벌은 마스크뿐 아니라 `images*` 도 서로 다르다 — 팔마다 자기 마스크의 centroid 로 창을 잡아 영상과 마스크를 함께 잘랐기 때문이다.
crop 은 `data/dataprep/utils/crop_avc.py` 가 centroid 기준 고정 크기 `(160, 160, 32)` 박스로 수행한다.

`Dataset003_*` 와 `Dataset001_*_TOTAL*` 은 이전 기준(406건, `fold=all` 마스크)이라 지금 실험과 섞어 쓸 수 없다.

이 데이터셋들은 `data/dataprep/organize_ablation_dataset.py` 와 `data/dataprep/utils/` 의 스크립트가 만든다.
파이프라인이 호출하지 않는 1회성 스크립트이고, 실행하면 곧바로 파일을 만들거나 지운다.

### 클래스 분포 (405건)

| severity | development (`imagesTr`) | test (`imagesVal`) | 합계 |
| --- | ---: | ---: | ---: |
| normal | 69 | 18 | 87 |
| nonsevere | 88 | 23 | 111 |
| severe | 165 | 42 | 207 |
| **합계** | **322** | **83** | **405** |

### GT 마스크 보유 (segmentation 학습셋 250건)

| severity | development | test | 계 | 405 중 커버리지 |
| --- | ---: | ---: | ---: | ---: |
| normal | 69 | 18 | 87 | 100% |
| nonsevere | 1 | 23 | 24 | 21.6% |
| severe | 97 | 42 | 139 | 67.1% |
| **합계** | **167** | **83** | **250** | 61.7% |

test 83 은 GT 보유 환자 안에서만 뽑아 전원 GT 를 갖는다. 분류가 쓰는 405개 마스크는 이 250건으로 학습한 5-fold 의 cross-fitting 예측물이다.

segmentation 학습 자체는 이 저장소에서 돌지 않는다.
같은 250건이 nnU-Net 워크스페이스(`/home/psw/nnUNet/data/`)에 ID 003 으로 등록돼 있고, `nnUNetv2_*` 에 넘기는 번호는 003 이다.

### 태스크별 분할

| 태스크 | train | val | test |
| --- | ---: | --- | ---: |
| Segmentation (nnU-Net) | fold 별 200 | fold 별 50 | 공통 83 (OOF 예측으로 평가) |
| Classification (ML) `main.py` | 322 | 322 내부 CV (특징 선택 전용) | 83 |
| Classification (DL) `dl_cls_train.py` | fold 별 257~258 | fold 별 64~65 | 83 |
| Classification (DL refit) `dl_cls_train.py` | 322 | 없음 (종료 epoch 은 5-fold 에서 온다) | 83 |

## 설정 (`config.py:Config`)

`main.py` 파이프라인의 설정 소스다. 예외가 하나 있다 — Gated Stage 1 의 학습 하이퍼파라미터(learning rate, batch size, epoch, patience)와 모델 구조(`hidden_dims=[256, 128]`, `dropout=0.3`)는
`gated_models/gated_pipeline.py` 안에 하드코딩돼 있고, 그쪽 seed 는 `GatedFusionTrainer` 기본값 42 가 아니라 50 이다.
DL 학습/평가는 `DL_Classification/dl_cls_config.py` 의 argparse 를 쓰지만,
`LABEL_FILE` · `CLASSIFICATION_MODE` · `IMAGE_TR_DIR`/`IMAGE_VAL_DIR` · `DL_NNUNET_CONFIG` 는 그쪽에서도 `Config` 를 직접 읽어 CLI 로 바꿀 수 없다.

| 속성 | 기본값 | 의미 |
| --- | --- | --- |
| `BASE_DIR` | `./data/datasets/Dataset004_mix_KUDH0467rm_cropped` | 데이터셋 루트. `IMAGE_*_DIR`/`LABEL_*_DIR` 이 여기서 파생된다 |
| `CLASSIFICATION_MODE` | `'multi'` | `'multi'` \| `'binary'` |
| `DATA_SPLIT_MODE` | `'fix'` | `'fix'`=디렉토리 기반, `'random'`=`TEST_SIZE_RATIO`(0.2) stratified |
| `ENABLE_DL_EMBEDDING` | `False` | DL 임베딩 결합 여부. `USE_*` 를 쓰려면 직접 켜야 한다 |
| `USE_GATED_FUSION` / `USE_ENSEMBLE` | `False` / `False` | 융합 방식 선택 |
| `DL_MODEL_TYPE` | `'nnunet'` | `'nnunet'` \| `'custom'`(MONAI ResNet50) |
| `DL_IMG_SIZE` | `(32, 384, 320)` | (D, H, W). nnUNet 사전학습 patch size. custom 은 `(56, 448, 448)` |
| `DL_NUM_FOLDS` | `5` | DL cross-fitting fold 수. `dl_cls_train.py --fold` 기본값이고 다른 값을 주면 시작 전에 멈춘다 |
| `DL_COMMENT_WRITER` | `f'{type}_{D}_{H}_{W}_{DL_DATASET_TAG}'` | 가중치·결과 디렉토리 이름. `DL_DATASET_TAG` 가 `BASE_DIR` 의 폴더 이름이라 데이터셋을 바꾸면 이름도 같이 바뀐다 |
| `RESAMPLED_SPACING` | `[0.3828125, 0.3828125, 3.0]` | radiomics 추출 전 목표 spacing `[x, y, z]` mm. `None` 이면 원본 |
| `ENABLE_DILATION` / `DILATION_ITERATIONS` | `False` / `1` | 마스크 팽창. 켜면 결과 디렉토리 이름에 `_dil{N}` 이 붙는다 |
| `FEATURE_SELECTION_METHOD` | `'lasso'` | `'lasso'`/`'rfe'`/`'univariate'`/`'mutual_info'`/`'random_forest'`/`'none'` |
| `CLASSIFICATION_MODELS` | `['LR', 'MLP1', 'MLP2']` | `MLP1`/`MLP2` 는 은닉층 1개/2개 `MLPClassifier`. `'SVM'`/`'RF'`/`'GB'`/`'KNN'`/`'NB'` 추가 가능 |
| `CV_FOLDS` | `5` | 특징 선택 내부 CV 전용. 학습/평가는 hold-out 1회다 |

방법별·모델별 세부 파라미터(`RFE_*`, `LASSO_*`, `SVM_*` 등)는 `config.py` 에 그대로 있다.

nnUNet 사전학습 자산은 `DL_NNUNET_CONFIG` 로 묶여 있다 — 아키텍처 plans(COCA), 정규화 통계 plans(AVC), 체크포인트.
**`DL_NNUNET_CONFIG` 의 파일이 하나라도 없으면 `dl_cls_train.py` 가 학습 시작 전에 멈춘다** — 체크포인트가 없어도 random init 으로 폴백하지 않는다.

## 결과

```
radiomics_analysis_results/{dataset_type}/{feature_method}/{mode}/{run_name}/[fold_{N}/]
├── log.txt                                       # 실행 로그. run_name 최상단에만 생긴다
├── model_validation_summary.csv                  # 모델별 Accuracy / F1 / AUC / AP
├── test_cases_prediction_results.csv
├── radiomics_features_{all,train,test}.csv       # Gated 는 gated_fused_features_*.csv
├── fold_{N}_best_model.pth, gated_training.log   # Gated 전용. Stage 1 체크포인트와 학습 로그
├── gated_fusion_predictions_fold_{N}.csv         # Gated 전용. GatedMLP 예측 확률
├── lasso_feature_analysis.csv
└── {model}_confusion_matrix.png, {model}_multiclass_{ROC,PR}_curve.png

DL_Classification/
├── weights/{writer_comment}/cls_fold_assignment.csv # development 322 의 patient_id → fold
├── weights/{writer_comment}/fold_best_epochs.csv   # fold 별 best epoch. refit 종료 epoch 의 근거
├── weights/{writer_comment}/{1..5}/best_model.pth   # 융합 갈래 세 개의 선행 조건
├── weights/{writer_comment}/refit/refit_model.pth   # development 322 전체로 다시 학습한 모델
├── logs/{writer_comment}/{1..5}/, logs/{writer_comment}/refit/  # TensorBoard
├── results/{writer_comment}/probs/refit.csv         # test 확률. ensemble 의 선행 조건
└── results/{writer_comment}/probs/fold_{N}.csv      # fold 별 test 확률. DL 팔 내부 점검용
```

`dataset_type` 은 `BASE_DIR` 이름에서 유도된다.
`run_name` 은 `default_` 또는 `dl{type}_{D}_{H}_{W}_` 에 `_ensemble` · `_gated` · `_dil{N}` 이 붙고 끝에 `_YYYYMMDD_HHMMSS` 가 온다.
DL 임베딩을 쓰면 `fold_oof/` 하나가 생기고(gated 는 stage 1 의 CV fold 마다 `fold_{N}/`), Radiomics 단독은 `run_name/` 직속이다.

## 규약

- **클래스 라벨 순서 고정** — multi=`['normal','nonsevere','severe']`, binary=`['nonsevere','severe']`.
  `data/preprocessor.py`, `DL_Classification/dl_cls_dataset.py`, `utils/ensemble.py`, `gated_models/` 가 이 순서를 가정한다.
- **파일명은 위 nnU-Net 패턴을 지킨다.** 어긋난 케이스는 에러 없이 건너뛴다.
- **DL 확률 CSV 의 컬럼명은 `proba_{class}`** 다. `utils/ensemble.py` 가 이 이름으로 DL/ML 확률을 합친다.
  Gated 의 `gated_fusion_predictions.csv` 만 `prob_{class}` 를 쓰고 `case_id` 컬럼이 없다.
- **메트릭은 multi=One-vs-Rest macro, binary=양성 `severe`** 기준이다. 새 평가 코드도 `trainer/train.py:_calculate_metrics` 와 맞춘다.
  `gated_trainer.py:evaluate_final_performance` 만 예외로 binary 에서도 F1 을 macro 로 계산하므로, binary 모드의 `GatedMLP` 행은 같은 표의 다른 행과 직접 비교되지 않는다.
- **증강 시드는 두 곳에서 주입된다** — `seed_torch` 의 `monai.utils.set_determinism` 이 `Compose` 가 자식 transform 에 넣을 시드를 정하고,
  loader 의 `worker_init_fn` 이 worker·epoch 마다 그 시드를 다시 뿌린다.
  `set_determinism` 을 transform 생성 뒤에 부르거나 `worker_init_fn` 을 빼면 같은 학습이 두 번 나오지 않는다.
- **`RESAMPLED_SPACING` 은 `[x, y, z]` 순서**다. nnU-Net plans.json 은 `[z, y, x]` 라 그대로 옮기면 z 축을 잘못 리샘플링한다.
- 새 하이퍼파라미터는 `Config` 클래스 속성으로 추가하고 `print_config_summary()` 에 반영한다.