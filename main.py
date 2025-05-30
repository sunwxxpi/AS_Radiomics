import sys
import os

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

def main():
    # 설정 및 로깅 초기화
    config = Config()
    output_dir = config.ensure_output_dir()
    setup_logging(output_dir)
    
    try:
        print("--- Radiomics 분석 파이프라인 시작 ---\n")
        
        # 설정 요약 출력
        config.print_config_summary()
        
        print(f"\n사용 가능한 특징 선택 방법: {config.get_available_feature_methods()}")
        print(f"사용 가능한 분류 모델: {config.get_available_classification_models()}")
        
        # 1. 데이터 로딩
        print("\n--- 1. 데이터 로딩 ---")
        data_loader = DataLoader(config.LABEL_FILE)
        patient_info_map = data_loader.load_labels()
        
        # 2. 특징 추출
        print("\n--- 2. Radiomics 특징 추출 ---")
        extractor = RadiomicsExtractor()
        
        train_features_df = extractor.extract_features_for_set(
            config.IMAGE_TR_DIR, config.LABEL_TR_DIR, "Train", patient_info_map
        )
        
        val_features_df = extractor.extract_features_for_set(
            config.IMAGE_VAL_DIR, config.LABEL_VAL_DIR, "Validation", patient_info_map
        )
        
        if train_features_df.empty:
            print("\n오류: 학습 세트에서 특징 추출 실패. 프로그램을 종료합니다.")
            return

        # 실제 데이터셋의 'AS_grade' 분포 출력
        if not train_features_df.empty and 'as_grade' in train_features_df.columns:
            print("\n--- 학습 데이터셋 'AS_grade' 분포 ---")
            train_as_distribution = train_features_df['as_grade'].value_counts(dropna=False).sort_index()
            print(train_as_distribution)
            print(f"총 학습 케이스 수: {len(train_features_df)}")

        if not val_features_df.empty and 'as_grade' in val_features_df.columns:
            print("\n--- 검증 데이터셋 'AS_grade' 분포 ---")
            val_as_distribution = val_features_df['as_grade'].value_counts(dropna=False).sort_index()
            print(val_as_distribution)
            print(f"총 검증 케이스 수: {len(val_features_df)}")

        # 전체 데이터셋 분포 (학습 + 검증)
        if not train_features_df.empty or not val_features_df.empty:
            all_features_list = []
            if not train_features_df.empty and 'as_grade' in train_features_df.columns:
                all_features_list.extend(train_features_df['as_grade'].tolist())
            if not val_features_df.empty and 'as_grade' in val_features_df.columns:
                all_features_list.extend(val_features_df['as_grade'].tolist())
            
            if all_features_list:
                import pandas as pd
                combined_as_series = pd.Series(all_features_list)
                print("\n--- 전체 데이터셋 (학습+검증) 'AS_grade' 분포 ---")
                combined_as_distribution = combined_as_series.value_counts(dropna=False).sort_index()
                print(combined_as_distribution)
                print(f"전체 케이스 수: {len(combined_as_series)}")
        
        # 3. 특징 저장
        file_handler = FileHandler(output_dir, config.FEATURE_SELECTION_METHOD)
        file_handler.save_features_to_csv(
            train_features_df, 'extracted_radiomics_features_train.csv', "학습"
        )
        
        if not val_features_df.empty:
            file_handler.save_features_to_csv(
                val_features_df, 'extracted_radiomics_features_val.csv', "검증"
            )
        
        # 4. 데이터 전처리
        print("\n--- 3. 데이터 전처리 ---")
        preprocessor = DataPreprocessor(config)
        processed_data = preprocessor.prepare_data(train_features_df, val_features_df)
        
        # 5. 모델 학습 및 평가
        print("\n--- 4. 모델 학습 및 평가 ---")
        trainer = ModelTrainer(config, preprocessor.label_encoder)
        results, prediction_results = trainer.train_and_evaluate(
            processed_data['X_train'], processed_data['y_train'],
            processed_data['X_val'], processed_data['y_val']
        )
        
        # 6. 시각화
        print("\n--- 5. 결과 시각화 ---")
        plotter = ResultPlotter(output_dir, preprocessor.label_encoder)
        plotter.plot_all_results(trainer.trained_models, prediction_results, results)
        
        # 7. 결과 저장
        print("\n--- 6. 결과 저장 ---")
        file_handler.save_prediction_results(
            prediction_results, 'test_cases_prediction_results.csv'
        )
        file_handler.save_model_summary(results)
        
        print(f"\n모든 분석 과정 완료. 결과는 '{output_dir}' 폴더에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        close_logging()

if __name__ == "__main__":
    main()