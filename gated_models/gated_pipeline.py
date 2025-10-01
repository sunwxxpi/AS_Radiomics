import os

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
        combined_features_df (pd.DataFrame): Radiomics + DL features
        fold_name (int): Fold 번호
        mode (str): 분류 모드
        fold_output_dir (str): Fold 출력 디렉토리
    """
    print(f"\n  [Fold {fold_name}] Gated Fusion 모델 학습 중...")

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
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'min_lr': 1e-6,
        'batch_size': 16,
        'epochs': 100,
        'patience': 15
    }

    # Gated Fusion 학습
    gated_trainer = GatedFusionTrainer(
        model_config=model_config,
        train_config=train_config,
        output_dir=fold_output_dir,
        random_seed=42  # 재현성을 위한 고정 seed
    )

    # 단일 fold만 학습 (내부적으로 train/val split)
    fold_results = gated_trainer.train_with_cv(combined_features_df, n_folds=1, external_fold_idx=fold_name)

    print(f"\n  [Fold {fold_name}] Gated Fusion 학습 완료")

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
    file_handler.save_model_summary(results, 'model_validation_summary.csv')

    print(f"  Fold {fold_name} 분석 완료!")