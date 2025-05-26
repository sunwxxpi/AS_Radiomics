import os
import pandas as pd

class FileHandler:
    """파일 저장을 담당하는 클래스"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
    
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
            
            # 확률 값 추가
            if ('predicted_probas' in prediction_dict and 
                prediction_dict['predicted_probas'] is not None):
                result_df[f'{model_name}_severe_probability'] = prediction_dict['predicted_probas']
        
        if not result_df.empty:
            file_path = os.path.join(self.output_dir, filename)
            result_df.to_csv(file_path, index=False)
            print(f"  예측 결과 저장 완료: {result_df.shape[0]} 케이스, {result_df.shape[1]} 열")
        else:
            print("  저장할 예측 결과가 없습니다.")
    
    def save_model_summary(self, results, filename):
        """모델 성능 요약을 CSV로 저장"""
        if not results:
            print("  요약할 모델 평가 결과가 없습니다.")
            return
        
        results_summary_df = pd.DataFrame({
            model_name: {
                'Accuracy': res.get('Accuracy', float('nan')),
                'AUC': res.get('AUC', float('nan')),
                'AP': res.get('AP', float('nan'))
            } for model_name, res in results.items() if isinstance(res, dict)
        }).T.sort_values(by='AUC', ascending=False)
        
        results_summary_df.index.name = 'Model'
        
        print("\n  --- 전체 모델 검증 성능 요약 ---")
        print(results_summary_df)
        
        file_path = os.path.join(self.output_dir, filename)
        results_summary_df.to_csv(file_path)
        print(f"  모델 성능 요약 저장 완료: {file_path}")