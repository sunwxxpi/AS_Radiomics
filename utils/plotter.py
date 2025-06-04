import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

class ResultPlotter:
    """결과 시각화를 담당하는 클래스"""
    
    def __init__(self, output_dir, label_encoder):
        self.output_dir = output_dir
        self.label_encoder = label_encoder
    
    def plot_all_results(self, feature_selection_method, trained_models, prediction_results, results):
        """모든 모델의 시각화 결과 생성"""
        print("  시각화 결과 생성 중...")
        
        for model_name in trained_models.keys():
            if model_name in prediction_results and model_name in results:
                self._plot_model_results(feature_selection_method, model_name, prediction_results[model_name], results[model_name])
    
    def _plot_model_results(self, feature_selection_method, model_name, predictions, metrics):
        """단일 모델의 시각화 결과 생성"""
        # severe 클래스 확률 확인
        proba_severe = None
        for key in predictions.keys():
            if key == 'proba_severe':
                proba_severe = predictions[key]
                break
        
        # ROC 곡선과 PR 곡선
        if proba_severe is not None:
            self._plot_roc_curve(feature_selection_method, model_name, predictions, metrics['AUC'], proba_severe)
            self._plot_pr_curve(feature_selection_method, model_name, predictions, metrics['AP'], proba_severe)
        
        # Confusion Matrix
        if 'Confusion Matrix' in metrics and metrics['Confusion Matrix'].size > 0:
            self._plot_confusion_matrix(feature_selection_method, model_name, metrics['Confusion Matrix'])
    
    def _plot_roc_curve(self, feature_selection_method, model_name, predictions, auc_score, proba_severe):
        """ROC 곡선 플롯"""
        if len(np.unique(predictions['actual_labels'])) <= 1:
            return
        
        try:
            fpr, tpr, _ = roc_curve(predictions['actual_labels'], proba_severe)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'{model_name} ROC (AUC = {auc_score:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title(f'ROC Curve - {feature_selection_method}_{model_name}', fontsize=16)
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(alpha=0.3)
            
            filename = os.path.join(self.output_dir, f'{model_name}_ROC_curve.png')
            plt.savefig(filename)
            plt.close()
            
        except Exception as e:
            print(f"      ROC 곡선 생성 오류 ({model_name}): {e}")
    
    def _plot_pr_curve(self, feature_selection_method, model_name, predictions, ap_score, proba_severe):
        """PR 곡선 플롯"""
        if len(np.unique(predictions['actual_labels'])) <= 1:
            return
        
        try:
            precision, recall, _ = precision_recall_curve(
                predictions['actual_labels'], proba_severe
            )
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'{model_name} PR (AP = {ap_score:.4f})')
            plt.xlabel('Recall (Sensitivity)', fontsize=14)
            plt.ylabel('Precision', fontsize=14)
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'Precision-Recall Curve - {feature_selection_method}_{model_name}', fontsize=16)
            plt.legend(loc="best", fontsize=12)
            plt.grid(alpha=0.3)
            
            filename = os.path.join(self.output_dir, f'{model_name}_PR_curve.png')
            plt.savefig(filename)
            plt.close()
            
        except Exception as e:
            print(f"      PR 곡선 생성 오류 ({model_name}): {e}")
    
    def _plot_confusion_matrix(self, feature_selection_method, model_name, conf_matrix):
        """혼동 행렬 플롯"""
        try:
            # 원래 클래스 가져오기
            original_class_names = self.label_encoder.classes_
            
            # 원하는 순서 정의
            desired_order = ['normal', 'nonsevere', 'severe']
            
            # 원래 순서에 있는 클래스만 포함시키기 (모든 클래스가 항상 존재하지는 않을 수 있음)
            target_names = [c for c in desired_order if c in original_class_names]
            
            # 만약 원하는 순서에 없는 클래스가 있다면 마지막에 추가
            for c in original_class_names:
                if c not in target_names:
                    target_names.append(c)
            
            # 클래스 순서에 맞게 confusion matrix 재배열
            idx_map = [list(original_class_names).index(c) for c in target_names]
            conf_matrix = conf_matrix[idx_map, :][:, idx_map]
            
            # 원본 혼동 행렬
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=target_names, yticklabels=target_names,
                       annot_kws={"size": 18})
            plt.title(f'Confusion Matrix (Raw Counts) - {feature_selection_method}_{model_name}', fontsize=16)
            plt.xlabel('Predicted Label', fontsize=14)
            plt.ylabel('True Label', fontsize=14)
            plt.tight_layout()
            
            filename = os.path.join(self.output_dir, f'{model_name}_CM_raw.png')
            plt.savefig(filename)
            plt.close()
            
            # 정규화된 혼동 행렬
            conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=target_names, yticklabels=target_names,
                       annot_kws={"size": 18}, vmin=0.0, vmax=1.0)
            plt.title(f'Confusion Matrix (Normalized) - {feature_selection_method}_{model_name}', fontsize=16)
            plt.xlabel('Predicted Label', fontsize=14)
            plt.ylabel('True Label', fontsize=14)
            plt.tight_layout()
            
            filename = os.path.join(self.output_dir, f'{model_name}_CM_normalized.png')
            plt.savefig(filename)
            plt.close()
            
        except Exception as e:
            print(f"      혼동 행렬 생성 오류 ({feature_selection_method}_{model_name}): {e}")