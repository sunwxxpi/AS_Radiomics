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
        
        # TR_DIR에서 특징 추출
        print("\n  Training 디렉토리에서 특징 추출 중...")
        train_features_df = extractor.extract_features_for_set(
            Config.IMAGE_TR_DIR, Config.LABEL_TR_DIR, "Train", patient_info_map, mode
        )
        
        # VAL_DIR에서 특징 추출
        print("\n  Validation 디렉토리에서 특징 추출 중...")
        val_features_df = extractor.extract_features_for_set(
            Config.IMAGE_VAL_DIR, Config.LABEL_VAL_DIR, "Validation", patient_info_map, mode
        )
        
        print("\n  TR_DIR과 VAL_DIR 특징 병합 중...")
        features_df = pd.concat([train_features_df, val_features_df], axis=0)
        print(f"  총 {len(features_df)} 개의 샘플 병합됨 (TR: {len(train_features_df)}, VAL: {len(val_features_df)})")

        # 실제 데이터셋의 'severity' 분포 출력
        if 'severity' in features_df.columns:
            print("\n--- 전체 데이터셋 'severity' 분포 ---")
            severity_counts = features_df['severity'].value_counts(dropna=False)
            
            # 다중 분류인 경우 원하는 순서로 정렬
            if mode == 'multi':
                class_order = ['normal', 'nonsevere', 'severe']
                ordered_counts = pd.Series({cls: severity_counts.get(cls, 0) for cls in class_order if cls in severity_counts.index})
                print(ordered_counts)
            else:
                severity_distribution = severity_counts.sort_index()
                print(severity_distribution)
            print(f"총 데이터 케이스 수: {len(features_df)}")
        
        # 3. 데이터 분할 (Config에 설정된 비율, 클래스 비율 유지)
        print("\n--- 3. 데이터 분할 ---")
        data_splitter = DataSplitter()
        train_features_df, val_features_df = data_splitter.split_data(features_df, mode)
        
        # 4. 특징 및 분할 정보 저장
        print("\n--- 4. 특징 및 분할 정보 저장 ---")
        file_handler = FileHandler(output_dir, Config.FEATURE_SELECTION_METHOD)
        
        # 전체 데이터셋 저장
        file_handler.save_features_to_csv(
            features_df, f'extracted_radiomics_features_all_{mode}.csv', "전체"
        )
        
        # 분할된 데이터셋 저장
        file_handler.save_split_data(
            train_features_df, val_features_df, f'radiomics_features_{mode}', mode
        )
        
        # 5. 데이터 전처리
        print("\n--- 5. 데이터 전처리 ---")
        preprocessor = DataPreprocessor(Config)
        processed_data = preprocessor.prepare_data(train_features_df, val_features_df)
        
        # 6. 모델 학습 및 평가
        print("\n--- 6. 모델 학습 및 평가 ---")
        trainer = ModelTrainer(Config, preprocessor.label_encoder)
        results, prediction_results = trainer.train_and_evaluate(
            processed_data['x_train'], processed_data['y_train'],
            processed_data['x_val'], processed_data['y_val']
        )
        
        # 7. 결과 시각화
        print("\n--- 7. 결과 시각화 ---")
        plotter = ResultPlotter(output_dir, preprocessor.label_encoder)
        plotter.plot_all_results(Config.FEATURE_SELECTION_METHOD, trainer.trained_models, prediction_results, results)
        
        # 8. 결과 저장
        print("\n--- 8. 결과 저장 ---")
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