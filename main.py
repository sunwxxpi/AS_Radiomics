import sys
import os
import pandas as pd

# 프로젝트 루트를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from utils.logger import setup_logging, close_logging
from utils.plotter import ResultPlotter
from utils.file_handler import FileHandler
from trainer.features_extractor import RadiomicsExtractor
from trainer.train import ModelTrainer

def run_pipeline(mode):
    """지정된 모드로 전체 파이프라인 실행"""
    # 설정 및 로깅 초기화
    Config.CLASSIFICATION_MODE = mode
    
    output_dir = Config.ensure_output_dir()
    logger = setup_logging(output_dir)
    
    try:
        print(f"--- Radiomics 분석 파이프라인 시작 (모드: {mode}) ---\n")
        
        # 설정 요약 출력
        Config.print_config_summary()
        
        # 1. 데이터 로딩
        print("\n--- 1. 데이터 로딩 ---")
        data_loader = DataLoader(Config.LABEL_FILE)
        patient_info_map = data_loader.load_labels(mode)
        
        # 2. Radiomics 특징 추출
        print("\n--- 2. Radiomics 특징 추출 ---")
        if Config.ENABLE_DILATION:
            print(f"  Dilation 활성화: {Config.DILATION_ITERATIONS}회 팽창 적용")
        else:
            print("  원본 마스크 사용 (Dilation 비활성화)")
            
        extractor = RadiomicsExtractor(
            enable_dilation=Config.ENABLE_DILATION,
            dilation_iterations=Config.DILATION_ITERATIONS
        )
        
        train_features_df = extractor.extract_features_for_set(
            Config.IMAGE_TR_DIR, Config.LABEL_TR_DIR, "Train", patient_info_map, mode
        )
        
        val_features_df = extractor.extract_features_for_set(
            Config.IMAGE_VAL_DIR, Config.LABEL_VAL_DIR, "Validation", patient_info_map, mode
        )
        
        if train_features_df.empty:
            print("\n오류: 학습 세트에서 특징 추출 실패. 프로그램을 종료합니다.")
            return

        # 실제 데이터셋의 'severity' 분포 출력
        if not train_features_df.empty and 'severity' in train_features_df.columns:
            print("\n--- 학습 데이터셋 'severity' 분포 ---")
            train_severity_counts = train_features_df['severity'].value_counts(dropna=False)
            
            # 다중 분류인 경우 원하는 순서로 정렬
            if mode == 'multi':
                class_order = ['normal', 'nonsevere', 'severe']
                ordered_counts = pd.Series({cls: train_severity_counts.get(cls, 0) for cls in class_order if cls in train_severity_counts.index})
                print(ordered_counts)
            else:
                train_severity_distribution = train_severity_counts.sort_index()
                print(train_severity_distribution)
            print(f"총 학습 케이스 수: {len(train_features_df)}")

        if not val_features_df.empty and 'severity' in val_features_df.columns:
            print("\n--- 검증 데이터셋 'severity' 분포 ---")
            val_severity_counts = val_features_df['severity'].value_counts(dropna=False)
            
            # 다중 분류인 경우 원하는 순서로 정렬
            if mode == 'multi':
                class_order = ['normal', 'nonsevere', 'severe']
                ordered_counts = pd.Series({cls: val_severity_counts.get(cls, 0) for cls in class_order if cls in val_severity_counts.index})
                print(ordered_counts)
            else:
                val_severity_distribution = val_severity_counts.sort_index()
                print(val_severity_distribution)
            print(f"총 검증 케이스 수: {len(val_features_df)}")
        
        # 3. 특징 저장
        print("\n--- 3. 특징 저장 ---")
        file_handler = FileHandler(output_dir, Config.FEATURE_SELECTION_METHOD)
        file_handler.save_features_to_csv(
            train_features_df, f'extracted_radiomics_features_train_{mode}.csv', "학습"
        )
        
        if not val_features_df.empty:
            file_handler.save_features_to_csv(
                val_features_df, f'extracted_radiomics_features_val_{mode}.csv', "검증"
            )
        
        # 4. 데이터 전처리
        print("\n--- 4. 데이터 전처리 ---")
        preprocessor = DataPreprocessor(Config)
        processed_data = preprocessor.prepare_data(train_features_df, val_features_df)
        
        # 5. 모델 학습 및 평가
        print("\n--- 5. 모델 학습 및 평가 ---")
        trainer = ModelTrainer(Config, preprocessor.label_encoder)
        results, prediction_results = trainer.train_and_evaluate(
            processed_data['x_train'], processed_data['y_train'],
            processed_data['x_val'], processed_data['y_val']
        )
        
        # 6. 결과 시각화
        print("\n--- 6. 결과 시각화 ---")
        plotter = ResultPlotter(output_dir, preprocessor.label_encoder)
        plotter.plot_all_results(Config.FEATURE_SELECTION_METHOD, trainer.trained_models, prediction_results, results)
        
        # 7. 결과 저장
        print("\n--- 7. 결과 저장 ---")
        file_handler.save_prediction_results(
            prediction_results, f'test_cases_prediction_results_{mode}.csv'
        )
        file_handler.save_model_summary(results, f'model_validation_summary_{mode}.csv')
        
        print(f"\n{mode} 모드 분석 과정 완료. 결과는 '{output_dir}' 폴더에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        close_logging()
        return logger

def main():
    # 두 가지 모드로 파이프라인 실행
    print("===== Binary 분류 모드로 파이프라인 실행 =====")
    binary_logger = run_pipeline('binary')
    
    print("\n\n===== Multi-class 분류 모드로 파이프라인 실행 =====")
    run_pipeline('multi')
    
    if binary_logger:
        sys.stdout = binary_logger.terminal

if __name__ == "__main__":
    main()