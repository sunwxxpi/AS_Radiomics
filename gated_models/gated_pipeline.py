import os
import pandas as pd
import numpy as np
from config import Config
from data.preprocessor import DataPreprocessor
from utils.data_splitter import DataSplitter
from utils.plotter import ResultPlotter
from utils.file_handler import FileHandler
from trainer.train import ModelTrainer
from gated_models.gated_trainer import GatedFusionTrainer
from gated_models.gated_feature_extractor import GatedFeatureExtractor


def run_gated_fusion_analysis(combined_features_df, fold_name, mode, fold_output_dir):
    """Gated Fusion 방식으로 분석 수행

    Args:
        combined_features_df (pd.DataFrame): Radiomics + DL features (imagesTr + imagesVal)
        fold_name (int): Fold 번호
        mode (str): 분류 모드
        fold_output_dir (str): Fold 출력 디렉토리
    """
    print(f"\n  [Fold {fold_name}] Gated Fusion 모델 학습 중...")

    # DL Classification과 동일하게: Train (data_source='train')만 사용하여 5-fold split
    if 'data_source' not in combined_features_df.columns:
        print("  경고: data_source 컬럼이 없습니다. 전체 데이터로 학습합니다.")
        train_features_df = combined_features_df.copy()
        test_features_df = None
    else:
        train_features_df = combined_features_df[combined_features_df['data_source'] == 'train'].copy()
        test_features_df = combined_features_df[combined_features_df['data_source'] == 'val'].copy()

        print(f"  Train 데이터 (imagesTr): {len(train_features_df)} 샘플")
        print(f"  Test 데이터 (imagesVal): {len(test_features_df)} 샘플")

    # 모델 설정
    model_config = {
        'radiomics_dim': None,  # 자동 추출
        'dl_dim': None,  # 자동 추출
        'num_classes': None,  # 자동 추출
        'fusion_dim': None,  # radiomics_dim + dl_dim 사용
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

    # Gated Fusion 학습 (Train 데이터만 사용)
    gated_trainer = GatedFusionTrainer(
        model_config=model_config,
        train_config=train_config,
        output_dir=fold_output_dir,
        random_seed=50
    )

    # DL Classification과 동일한 5-fold 중 해당 fold만 학습
    gated_trainer.train_with_cv(
        train_features_df,
        n_folds=5,
        external_fold_idx=fold_name
    )

    print(f"\n  [Fold {fold_name}] Gated Fusion 학습 완료")

    # ======================================================================
    # Test Set 평가 (imagesVal, data_source='val')
    # ======================================================================
    gated_model_path = os.path.join(fold_output_dir, f'fold_{fold_name}_best_model.pth')

    if not os.path.exists(gated_model_path):
        print(f"  경고: Fold {fold_name} Gated 모델을 찾을 수 없음: {gated_model_path}")
        return

    from gated_models.gated_trainer import GatedFusionDataset
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import label_binarize, LabelEncoder

    if test_features_df is not None and len(test_features_df) > 0:
        print(f"\n  [Fold {fold_name}] Test Set (imagesVal) 평가 중...")

        # Test 데이터 전처리
        test_data = gated_trainer.prepare_data(test_features_df)
        X_rad_test = test_data['X_radiomics']
        X_dl_test = test_data['X_dl']
        y_test = test_data['y']

        # Test dataset 생성
        test_dataset = GatedFusionDataset(X_rad_test, X_dl_test, y_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=4
        )

        # Test 평가 수행
        test_results = gated_trainer.evaluate_final_performance(
            test_loader,
            gated_model_path,
            fold_name
        )

        if test_results:
            # AUC와 AP 계산
            y_true = test_results['labels']
            y_proba = test_results['probabilities']
            n_classes = len(test_results['class_names'])

            if n_classes == 2:
                # Binary classification
                auc_score = roc_auc_score(y_true, y_proba[:, 1])
                ap_score = average_precision_score(y_true, y_proba[:, 1])
            else:
                # Multi-class classification (macro average)
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                auc_score = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
                ap_score = average_precision_score(y_true_bin, y_proba, average='macro')

            # MLP 모델 결과를 model_validation_summary.csv에 병합
            mlp_results = {
                'MLP': {
                    'Accuracy': test_results['accuracy'],
                    'F1': test_results['f1_macro'],
                    'AUC': auc_score,
                    'AP': ap_score
                }
            }

            # 기존 파일이 있으면 읽어서 병합
            summary_path = os.path.join(fold_output_dir, 'model_validation_summary.csv')
            if os.path.exists(summary_path):
                existing_df = pd.read_csv(summary_path, index_col='Model')
                mlp_df = pd.DataFrame(mlp_results).T
                mlp_df.index.name = 'Model'
                combined_df = pd.concat([existing_df, mlp_df]).sort_index()
                combined_df.to_csv(summary_path)
                print(f"  MLP 결과를 model_validation_summary.csv에 병합: {summary_path}")
            else:
                # 기존 파일이 없으면 새로 생성
                file_handler = FileHandler(fold_output_dir, 'gated_fusion')
                file_handler.save_model_summary(mlp_results, 'model_validation_summary.csv')
                print(f"  MLP 결과를 model_validation_summary.csv에 저장: {summary_path}")

            # Confusion Matrix 이미지 생성 (CSV 파일은 생성하지 않음)
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(test_results['class_names'])

            plotter = ResultPlotter(fold_output_dir, label_encoder)

            # metrics 딕셔너리 생성 (plotter가 필요로 하는 형식)
            metrics = {
                'Accuracy': test_results['accuracy'],
                'F1': test_results['f1_macro'],
                'AUC': auc_score,
                'AP': ap_score,
                'Confusion Matrix': test_results['confusion_matrix']
            }

            plotter._plot_confusion_matrix('gated_fusion', 'MLP', test_results['confusion_matrix'], metrics)
            print(f"  Confusion Matrix 이미지 생성: {os.path.join(fold_output_dir, 'MLP_confusion_matrix.png')}")

            # Predictions 저장 (예측 확률 포함)
            pred_df = pd.DataFrame({
                'true_label': [test_results['class_names'][i] for i in test_results['labels']],
                'predicted_label': [test_results['class_names'][i] for i in test_results['predictions']]
            })

            # Probabilities 추가
            for i, class_name in enumerate(test_results['class_names']):
                pred_df[f'prob_{class_name}'] = test_results['probabilities'][:, i]

            pred_path = os.path.join(fold_output_dir, f'gated_fusion_predictions_fold_{fold_name}.csv')
            pred_df.to_csv(pred_path, index=False)
            print(f"  예측 결과 저장: {pred_path}")

        print(f"\n  [Fold {fold_name}] Test Set 평가 완료!")
    else:
        print(f"\n  [Fold {fold_name}] Test Set이 없어 평가를 건너뜁니다.")

    # 학습된 Gated Fusion으로 fused features 추출
    print(f"\n  [Fold {fold_name}] Fused Features 추출 중...")

    # 학습된 모델 로드
    gated_model_path = os.path.join(fold_output_dir, f'fold_{fold_name}_best_model.pth')

    if not os.path.exists(gated_model_path):
        print(f"  경고: Fold {fold_name} Gated 모델을 찾을 수 없음: {gated_model_path}")
        return

    # Radiomics와 DL feature 차원 추출
    radiomics_cols = [col for col in combined_features_df.columns
                    if col.startswith(('original_', 'wavelet_', 'log_', 'square_', 'squareroot_',
                                     'exponential_', 'logarithm_', 'gradient_', 'lbp-'))]
    dl_cols = [col for col in combined_features_df.columns
              if col.startswith('dl_embedding_feature_')]

    model_config['radiomics_dim'] = len(radiomics_cols)
    model_config['dl_dim'] = len(dl_cols)
    model_config['num_classes'] = combined_features_df['severity'].nunique()

    gated_extractor = GatedFeatureExtractor(
        model_path=gated_model_path,
        model_config=model_config
    )

    fused_features_df = gated_extractor.extract_features_from_dataframe(
        combined_features_df,
        use_fitted_scaler=False
    )

    print(f"  [Fold {fold_name}] Fused Features 추출 완료: {fused_features_df.shape}")

    # Fused features로 전통적 ML 분류기 학습
    print(f"\n  [Fold {fold_name}] 전통적 ML 분류기 학습 중...")

    # case_id를 인덱스로 변환
    if 'case_id' in fused_features_df.columns:
        fused_features_df_processed = fused_features_df.set_index('case_id')
    else:
        fused_features_df_processed = fused_features_df.copy()

    # 1. 데이터 분할
    print(f"\n  --- Fold {fold_name} 데이터 분할 ---")
    data_splitter = DataSplitter()
    train_features_df, val_features_df = data_splitter.split_data(fused_features_df_processed, mode)

    # 2. 특징 및 분할 정보 저장
    print(f"\n  --- Fold {fold_name} 특징 및 분할 정보 저장 ---")
    file_handler = FileHandler(fold_output_dir, 'gated_fusion')

    # 전체 데이터셋 저장
    file_handler.save_features_to_csv(fused_features_df, 'gated_fused_features_all.csv', f"전체 (Fold {fold_name})")

    # 분할된 데이터셋 저장
    file_handler.save_split_data(train_features_df, val_features_df, 'gated_fused_features', mode)

    # 3. 데이터 전처리
    print(f"\n  --- Fold {fold_name} 데이터 전처리 ---")
    preprocessor = DataPreprocessor(Config)
    processed_data = preprocessor.prepare_data(train_features_df, val_features_df)

    # LASSO 분석 결과 저장
    if Config.FEATURE_SELECTION_METHOD == 'lasso':
        lasso_analysis = preprocessor.get_lasso_analysis()
        if lasso_analysis is not None:
            file_handler.save_lasso_analysis(lasso_analysis, 'lasso_feature_analysis.csv')

    # 4. 모델 학습 및 평가
    print(f"\n  --- Fold {fold_name} 모델 학습 및 평가 ---")
    trainer = ModelTrainer(Config, preprocessor.label_encoder)
    results, prediction_results = trainer.train_and_evaluate(
        processed_data['x_train'], processed_data['y_train'],
        processed_data['x_val'], processed_data['y_val']
    )

    # 5. 결과 시각화
    print(f"\n  --- Fold {fold_name} 결과 시각화 ---")
    plotter = ResultPlotter(fold_output_dir, preprocessor.label_encoder)
    plotter.plot_all_results('gated_fusion', trainer.trained_models, prediction_results, results)

    # 6. 결과 저장
    print(f"\n  --- Fold {fold_name} 결과 저장 ---")
    file_handler.save_prediction_results(prediction_results, 'test_cases_prediction_results.csv')

    # ML 분류기 결과를 기존 model_validation_summary.csv에 병합
    summary_path = os.path.join(fold_output_dir, 'model_validation_summary.csv')
    if os.path.exists(summary_path):
        # 기존 파일 읽기 (MLP 포함)
        existing_df = pd.read_csv(summary_path, index_col='Model')

        # ML 분류기 결과 추가
        ml_df = pd.DataFrame({
            model_name: {
                'Accuracy': res.get('Accuracy', float('nan')),
                'F1': res.get('F1', float('nan')),
                'AUC': res.get('AUC', float('nan')),
                'AP': res.get('AP', float('nan'))
            } for model_name, res in results.items() if isinstance(res, dict)
        }).T
        ml_df.index.name = 'Model'

        # 병합 및 정렬 (MLP, LR, RF, SVM 순서)
        combined_df = pd.concat([existing_df, ml_df])
        # 중복 제거 (같은 모델이 있으면 나중 것 사용)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

        # 원하는 순서로 정렬
        desired_order = ['MLP', 'LR', 'RF', 'SVM']
        available_models = [m for m in desired_order if m in combined_df.index]
        other_models = sorted([m for m in combined_df.index if m not in desired_order])
        final_order = available_models + other_models

        combined_df = combined_df.reindex(final_order)
        combined_df.to_csv(summary_path)
        print(f"  ML 분류기 결과를 model_validation_summary.csv에 병합: {summary_path}")
    else:
        # 기존 파일이 없으면 그냥 저장
        file_handler.save_model_summary(results, 'model_validation_summary.csv')

    print(f"  Fold {fold_name} 분석 완료!")