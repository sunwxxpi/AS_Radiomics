import os
import numpy as np
import pandas as pd
from natsort import natsorted

class FileHandler:
    """파일 저장을 담당하는 클래스"""
    
    def __init__(self, output_dir, feature_selection_method='unknown'):
        self.output_dir = output_dir
        self.feature_selection_method = feature_selection_method
    
    def save_features_to_csv(self, df, filename, df_name):
        """특징 DataFrame을 CSV로 저장"""
        print(f"\n  {df_name} 특징 저장 시도: {filename}")
        
        if df.empty:
            print(f"  {df_name} DataFrame이 비어있어 저장하지 않습니다.")
            return
        
        df_to_save = df.reset_index()
        
        # 컬럼 순서 정렬
        if 'severity' in df_to_save.columns:
            other_cols = [col for col in df_to_save.columns if col not in ['case_id', 'severity']]
            new_cols_order = ['case_id', 'severity'] + other_cols
        else:
            other_cols = [col for col in df_to_save.columns if col != 'case_id']
            new_cols_order = ['case_id'] + other_cols
            print(f"  경고: {df_name} 저장 시 'severity' 열을 찾을 수 없습니다.")
        
        df_to_save = df_to_save[new_cols_order]
        
        # case_id 기준으로 오름차순 정렬
        if 'case_id' in df_to_save.columns:
            df_to_save = df_to_save.sort_values(by='case_id')
            print(f"  {df_name} 데이터를 case_id 기준으로 오름차순 정렬했습니다.")
        
        file_path = os.path.join(self.output_dir, filename)
        df_to_save.to_csv(file_path, index=False)
        print(f"  {df_name} 특징 저장 완료.")
    
    def save_prediction_results(self, model_predictions, filename):
        """각 테스트 케이스별 예측 결과를 CSV로 저장"""
        print(f"\n  예측 결과 저장 시도: {filename}")
        
        if not model_predictions or not any(model_predictions.values()):
            print("  저장할 예측 결과가 없습니다.")
            return
        
        result_df = pd.DataFrame()
        
        for model_name, prediction_dict in model_predictions.items():
            if not prediction_dict:
                continue
            
            if result_df.empty:
                # 첫 모델의 결과로 기본 DataFrame 생성
                result_df = pd.DataFrame({
                    'case_id': prediction_dict['case_ids'],
                    'actual_label': prediction_dict['actual_labels_str']
                })
            
            # 각 모델의 예측 결과 추가
            result_df[f'{model_name}_predicted'] = prediction_dict['predicted_labels_str']
            
            # 모든 클래스에 대한 확률 추가
            for class_label in [key.replace('proba_', '') for key in prediction_dict.keys() if key.startswith('proba_')]:
                if f'proba_{class_label}' in prediction_dict:
                    result_df[f'{model_name}_{class_label}_probability'] = prediction_dict[f'proba_{class_label}']
        
        if not result_df.empty:
            # case_id 기준으로 오름차순 정렬
            if 'case_id' in result_df.columns:
                result_df = result_df.sort_values(by='case_id')
                print(f"  예측 결과를 case_id 기준으로 오름차순 정렬했습니다.")
                
            file_path = os.path.join(self.output_dir, filename)
            result_df.to_csv(file_path, index=False)
            print(f"  예측 결과 저장 완료: {result_df.shape[0]} 케이스, {result_df.shape[1]} 열")
        else:
            print("  저장할 예측 결과가 없습니다.")
    
    def save_model_summary(self, results, filename=None):
        """모델 성능 요약을 CSV로 저장"""
        if not results:
            print("  요약할 모델 평가 결과가 없습니다.")
            return
        
        if filename is None:
            filename = 'model_validation_summary.csv'
        
        results_summary_df = pd.DataFrame({
            model_name: {
                'Accuracy': res.get('Accuracy', float('nan')),
                'F1': res.get('F1', float('nan')),
                'AUC': res.get('AUC', float('nan')),
                'AP': res.get('AP', float('nan'))
            } for model_name, res in results.items() if isinstance(res, dict)
        }).T.sort_index()
        
        results_summary_df.index.name = 'Model'
        
        print("\n  --- 전체 모델 검증 성능 요약 ---")
        print(results_summary_df)
        
        file_path = os.path.join(self.output_dir, filename)
        results_summary_df.to_csv(file_path)
        print(f"  모델 성능 요약 저장 완료: {file_path}")
    
    def save_split_data(self, train_df, test_df, base_filename, mode='binary'):
        """분할된 train/test 데이터를 CSV로 저장"""
        print(f"\n  분할된 데이터 저장: {base_filename}")
        
        if train_df.empty or test_df.empty:
            print("  저장할 분할 데이터가 없습니다.")
            return
        
        # case_id 기준으로 오름차순 정렬
        if 'case_id' in train_df.index.names or 'case_id' in train_df.columns:
            if 'case_id' in train_df.index.names:
                train_df = train_df.reset_index()
            train_df = train_df.sort_values(by='case_id')
            print(f"  학습 데이터를 case_id 기준으로 오름차순 정렬했습니다.")
            
        if 'case_id' in test_df.index.names or 'case_id' in test_df.columns:
            if 'case_id' in test_df.index.names:
                test_df = test_df.reset_index()
            test_df = test_df.sort_values(by='case_id')
            print(f"  테스트 데이터를 case_id 기준으로 오름차순 정렬했습니다.")
        
        # 파일 경로
        train_path = os.path.join(self.output_dir, f"{base_filename}_train.csv")
        test_path = os.path.join(self.output_dir, f"{base_filename}_test.csv")
        
        # 저장
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"  학습 데이터 저장 완료: {train_path} ({len(train_df)} 샘플)")
        print(f"  테스트 데이터 저장 완료: {test_path} ({len(test_df)} 샘플)")
    
    def save_lasso_analysis(self, lasso_analysis, filename=None):
        """LASSO 특징 선택 분석 결과를 CSV로 저장"""
        if lasso_analysis is None:
            print("  LASSO 분석 결과가 없습니다.")
            return
        
        if filename is None:
            filename = 'lasso_feature_analysis.csv'
        
        # 특징별 분석 결과 DataFrame 생성
        analysis_df = pd.DataFrame({
            'feature_name': lasso_analysis['feature_names'],
            'coefficient': lasso_analysis['coefficients'],
            'abs_coefficient': np.abs(lasso_analysis['coefficients']),
            'selection_status': ['L1_regularized' if abs(coef) < 1e-10 
                               else 'below_threshold' if abs(coef) < lasso_analysis['threshold']
                               else 'selected' 
                               for coef in lasso_analysis['coefficients']]
        })
        
        # 상태별로 분리하여 각각 정렬
        selected_df = analysis_df[analysis_df['selection_status'] == 'selected'].copy()
        below_threshold_df = analysis_df[analysis_df['selection_status'] == 'below_threshold'].copy()
        l1_regularized_df = analysis_df[analysis_df['selection_status'] == 'L1_regularized'].copy()
        
        # 각 그룹별로 정렬
        # 선택된 특징: 계수 절댓값 기준 내림차순
        if not selected_df.empty:
            selected_df = selected_df.sort_values('abs_coefficient', ascending=False)
        
        # Threshold 미달 특징: 계수 절댓값 기준 내림차순
        if not below_threshold_df.empty:
            below_threshold_df = below_threshold_df.sort_values('abs_coefficient', ascending=False)
        
        # L1 정규화된 특징: 특징명 기준 오름차순
        if not l1_regularized_df.empty:
            natural_sorted_indices = natsorted(
                range(len(l1_regularized_df)), 
                key=lambda i: l1_regularized_df.iloc[i]['feature_name']
            )
            l1_regularized_df = l1_regularized_df.iloc[natural_sorted_indices].reset_index(drop=True)
        
        # 정렬된 데이터프레임들을 다시 합침
        final_df = pd.concat([selected_df, below_threshold_df, l1_regularized_df], ignore_index=True)
        
        file_path = os.path.join(self.output_dir, filename)
        final_df.to_csv(file_path, index=False)
        
        print(f"  LASSO 특징 분석 결과 저장 완료: {file_path}")