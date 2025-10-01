import sys
import os
import pandas as pd

# 프로젝트 루트를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from utils.data_splitter import DataSplitter
from utils.logger import setup_logging, close_logging
from utils.plotter import ResultPlotter
from utils.file_handler import FileHandler
from trainer.features_extractor import RadiomicsExtractor
from trainer.train import ModelTrainer
from gated_models import run_gated_fusion_analysis
from utils.ensemble import run_ensemble_for_fold

def run_pipeline():
    """Radiomics 분석 메인 파이프라인

    워크플로우:
        1. Radiomics 특징 추출 (한 번만)
        2. Fold별 DL embedding 추가
        3. 특징 융합 (일반 Concat 또는 Gated Fusion)
        4. 전통적 ML 분류기 학습 및 평가
    """
    # 설정 및 로깅 초기화
    output_dir = Config.ensure_output_dir()
    logger = setup_logging(output_dir)

    try:
        print("--- Radiomics 분석 파이프라인 시작 ---\n")

        # Ensemble 사전 검증
        if Config.USE_ENSEMBLE and not Config.ENABLE_DL_EMBEDDING:
            print("오류: Ensemble을 사용하려면 DL Embedding이 활성화되어야 합니다.")
            print("config.py에서 ENABLE_DL_EMBEDDING = True로 설정해주세요.")
            return
        
        # Gated Fusion 사전 검증
        if Config.USE_GATED_FUSION and not Config.ENABLE_DL_EMBEDDING:
            print("오류: Gated Fusion을 사용하려면 DL Embedding이 활성화되어야 합니다.")
            print("config.py에서 ENABLE_DL_EMBEDDING = True로 설정해주세요.")
            return

        # 설정 요약 출력
        Config.print_config_summary()

        mode = Config.CLASSIFICATION_MODE
        
        # 1. 데이터 로딩
        print("\n--- 1. 데이터 로딩 ---")
        data_loader = DataLoader(Config.LABEL_FILE)
        patient_info_map = data_loader.load_labels(mode)
        
        # 2. DL 모델 경로 확인
        dl_model_paths = {}
        if Config.ENABLE_DL_EMBEDDING:
            dl_model_paths = Config.get_dl_model_paths()
            print(f"  DL Embedding 활성화: {Config.DL_MODEL_TYPE} 모델 사용")
            
            missing_models = []
            for fold, path in dl_model_paths.items():
                if not os.path.exists(path):
                    missing_models.append(f"Fold {fold}: {path}")
            
            if missing_models:
                print(f"  경고: 다음 DL 모델 파일들을 찾을 수 없습니다:")
                for missing in missing_models:
                    print(f"    - {missing}")
                
                if len(missing_models) == len(dl_model_paths):
                    print("  모든 DL 모델이 없으므로 DL embedding 없이 계속 진행합니다.")
                    Config.ENABLE_DL_EMBEDDING = False
                    dl_model_paths = {}
        else:
            print("  DL Embedding 비활성화")
            
        if Config.ENABLE_DILATION:
            print(f"  Dilation 활성화: {Config.DILATION_ITERATIONS}회 팽창 적용")
        else:
            print("  원본 마스크 사용 (Dilation 비활성화)")
        
        # 3. Radiomics 특징 추출기 초기화
        print("\n--- 2. Radiomics 특징 추출기 초기화 ---")
        extractor = RadiomicsExtractor(
            enable_dl_embedding=Config.ENABLE_DL_EMBEDDING,
            dl_model_paths=dl_model_paths,
            dl_model_type=Config.DL_MODEL_TYPE if Config.ENABLE_DL_EMBEDDING else 'custom',
            dl_nnunet_config=Config.DL_NNUNET_CONFIG if Config.ENABLE_DL_EMBEDDING and Config.DL_MODEL_TYPE == 'nnunet' else None,
            enable_dilation=Config.ENABLE_DILATION,
            dilation_iterations=Config.DILATION_ITERATIONS
        )
        
        # 4. Radiomics 특징 추출 (한 번만)
        print("\n--- 3. Radiomics 특징 추출 ---")
        print("\n  Training 디렉토리에서 Radiomics 특징 추출 중...")
        train_radiomics_df = extractor.extract_radiomics_features_for_set(
            Config.IMAGE_TR_DIR, Config.LABEL_TR_DIR, "Train", patient_info_map, mode
        )
        
        print("\n  Validation 디렉토리에서 Radiomics 특징 추출 중...")
        val_radiomics_df = extractor.extract_radiomics_features_for_set(
            Config.IMAGE_VAL_DIR, Config.LABEL_VAL_DIR, "Validation", patient_info_map, mode
        )
        
        print("\n  TR_DIR과 VAL_DIR Radiomics 특징 병합 중...")
        radiomics_df = pd.concat([train_radiomics_df, val_radiomics_df], axis=0)
        print(f"  총 {len(radiomics_df)} 개의 샘플 병합됨 (TR: {len(train_radiomics_df)}, VAL: {len(val_radiomics_df)})")

        # 5. Fold별 DL embedding 특징 추가 및 분석
        if Config.ENABLE_DL_EMBEDDING and dl_model_paths:
            print("\n--- 4. Fold별 DL Embedding 특징 추가 및 분석 ---")
            
            # 각 fold에 대해 독립적인 분석 수행
            for fold in sorted(dl_model_paths.keys()):
                print(f"\n=== Fold {fold} 분석 시작 ===")
                
                # 해당 fold의 DL embedding을 Radiomics에 추가
                combined_features_df = extractor.add_dl_features_to_radiomics(radiomics_df, fold)
                
                # 클래스 분포 확인
                if 'severity' in combined_features_df.columns:
                    print(f"\n--- Fold {fold} 데이터셋 'severity' 분포 ---")
                    severity_counts = combined_features_df['severity'].value_counts(dropna=False)
                    
                    if mode == 'multi':
                        class_order = ['normal', 'nonsevere', 'severe']
                        ordered_counts = pd.Series({cls: severity_counts.get(cls, 0) for cls in class_order if cls in severity_counts.index})
                        print(ordered_counts)
                    else:
                        severity_distribution = severity_counts.sort_index()
                        print(severity_distribution)
                    print(f"총 데이터 케이스 수: {len(combined_features_df)}")
                
                # fold별 전체 분석 파이프라인 실행
                run_fold_analysis(combined_features_df, fold, mode, output_dir)
        else:
            print("\n--- 4. Radiomics 전용 분석 ---")
            # DL embedding 없이 Radiomics만으로 분석
            radiomics_only_df = radiomics_df.drop(columns=['image_path'], errors='ignore')

            if 'severity' in radiomics_only_df.columns:
                print("\n--- Radiomics 전용 데이터셋 'severity' 분포 ---")
                severity_counts = radiomics_only_df['severity'].value_counts(dropna=False)

                if mode == 'multi':
                    class_order = ['normal', 'nonsevere', 'severe']
                    ordered_counts = pd.Series({cls: severity_counts.get(cls, 0) for cls in class_order if cls in severity_counts.index})
                    print(ordered_counts)
                else:
                    severity_distribution = severity_counts.sort_index()
                    print(severity_distribution)
                print(f"총 데이터 케이스 수: {len(radiomics_only_df)}")

            run_fold_analysis(radiomics_only_df, None, mode, output_dir)
        
        print(f"\n{mode} 모드 분석 과정 완료. 결과는 '{output_dir}' 폴더에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        close_logging()
        return logger

def run_fold_analysis(features_df, fold_name, mode, base_output_dir):
    """Fold별 전체 분석 파이프라인

    프로세스:
        1. 데이터 분할 (train/validation)
        2. 특징 정규화 및 선택 (LASSO 등)
        3. ML 분류기 학습 (LR, SVM, RF, GB, KNN, NB)
        4. 성능 평가 및 시각화
        5. 결과 저장

    Args:
        features_df (pd.DataFrame): Radiomics + DL features (또는 Gated Fused features)
        fold_name (int or None): Fold 번호 (None이면 'Radiomics_Only')
        mode (str): 'binary' 또는 'multi'
        base_output_dir (str): 출력 디렉토리 경로
    """
    fold_name = fold_name if fold_name else 'Radiomics_Only'

    print(f"\n  === {fold_name} 분석 실행 ===")

    # fold별 독립 출력 디렉토리 생성 (fold_name이 'Radiomics_Only'이면 base_output_dir 사용)
    if fold_name != 'Radiomics_Only':
        fold_output_dir = os.path.join(base_output_dir, f"fold_{fold_name}")
        os.makedirs(fold_output_dir, exist_ok=True)
    else:
        fold_output_dir = base_output_dir

    try:
        # Gated fusion 사용 시 별도 처리
        if Config.USE_GATED_FUSION:
            run_gated_fusion_analysis(features_df, fold_name, mode, fold_output_dir)
            return
        # case_id를 인덱스로 변환 (전처리 호환성)
        if 'case_id' in features_df.columns:
            features_df_processed = features_df.set_index('case_id')
        else:
            features_df_processed = features_df.copy()
        
        # 1. 데이터 분할
        print(f"\n  --- {fold_name} 데이터 분할 ---")
        data_splitter = DataSplitter()
        train_features_df, val_features_df = data_splitter.split_data(features_df_processed, mode)
        
        # 2. 특징 및 분할 정보 저장
        print(f"\n  --- {fold_name} 특징 및 분할 정보 저장 ---")
        file_handler = FileHandler(fold_output_dir, Config.FEATURE_SELECTION_METHOD)
        
        # 전체 데이터셋 저장
        file_handler.save_features_to_csv(features_df, 'radiomics_features_all.csv', f"전체 ({fold_name})")
        
        # 분할된 데이터셋 저장
        file_handler.save_split_data(train_features_df, val_features_df, 'radiomics_features', mode)
        
        # 3. 데이터 전처리
        print(f"\n  --- {fold_name} 데이터 전처리 ---")
        preprocessor = DataPreprocessor(Config)
        processed_data = preprocessor.prepare_data(train_features_df, val_features_df)
        
        # LASSO 분석 결과 저장
        if Config.FEATURE_SELECTION_METHOD == 'lasso':
            lasso_analysis = preprocessor.get_lasso_analysis()
            if lasso_analysis is not None:
                file_handler.save_lasso_analysis(lasso_analysis, 'lasso_feature_analysis.csv')
        
        # 4. 모델 학습 및 평가
        print(f"\n  --- {fold_name} 모델 학습 및 평가 ---")
        trainer = ModelTrainer(Config, preprocessor.label_encoder)
        results, prediction_results = trainer.train_and_evaluate(
            processed_data['x_train'], processed_data['y_train'],
            processed_data['x_val'], processed_data['y_val']
        )
        
        # 5. 결과 시각화
        print(f"\n  --- {fold_name} 결과 시각화 ---")
        plotter = ResultPlotter(fold_output_dir, preprocessor.label_encoder)
        plotter.plot_all_results(Config.FEATURE_SELECTION_METHOD, trainer.trained_models, prediction_results, results)
        
        # 6. 결과 저장
        print(f"\n  --- {fold_name} 결과 저장 ---")
        file_handler.save_prediction_results(prediction_results, 'test_cases_prediction_results.csv')
        file_handler.save_model_summary(results, 'model_validation_summary.csv')

        # 7. Ensemble 수행 (옵션)
        if Config.USE_ENSEMBLE and Config.ENABLE_DL_EMBEDDING and fold_name != 'Radiomics_Only':
            print(f"\n  --- {fold_name} Soft Voting Ensemble 수행 ---")
            try:
                # DL 결과 디렉토리 경로
                dl_results_dir = f'./DL_Classification/results/{Config.DL_COMMENT_WRITER}'

                # Ensemble 수행
                run_ensemble_for_fold(
                    fold=fold_name,
                    dl_results_dir=dl_results_dir,
                    radiomics_results_dir=fold_output_dir,
                    classification_mode=mode,
                    models=Config.CLASSIFICATION_MODELS,
                    feature_selection_method=Config.FEATURE_SELECTION_METHOD
                )

                print(f"  {fold_name} Ensemble 완료!")
                print(f"  결과: {os.path.join(fold_output_dir, 'ensemble', f'ensemble_results_fold_{fold_name}.csv')}")

            except Exception as ensemble_error:
                print(f"  {fold_name} Ensemble 중 오류 발생: {ensemble_error}")
                import traceback
                traceback.print_exc()

        print(f"  {fold_name} 분석 완료!")

    except Exception as e:
        print(f"  {fold_name} 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()