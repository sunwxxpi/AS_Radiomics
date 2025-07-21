import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve, precision_recall_curve

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
        # 클래스 확률 확인
        class_probas = {}
        for key in predictions.keys():
            if key.startswith('proba_'):
                class_name = key.replace('proba_', '')
                class_probas[class_name] = predictions[key]
        
        # 클래스 개수 확인
        n_classes = len(np.unique(predictions['actual_labels']))
        
        # ROC 곡선과 PR 곡선
        if class_probas and n_classes > 0:
            # 이진 분류인 경우
            if n_classes == 2:
                severe_class = None
                for class_name in class_probas.keys():
                    if class_name == 'severe' or class_name == '1':
                        severe_class = class_name
                        break
                
                if severe_class and f'proba_{severe_class}' in predictions:
                    self._plot_roc_curve(feature_selection_method, model_name, predictions, metrics['AUC'], predictions[f'proba_{severe_class}'])
                    self._plot_pr_curve(feature_selection_method, model_name, predictions, metrics['AP'], predictions[f'proba_{severe_class}'])
            # 다중 클래스인 경우
            else:
                self._plot_multiclass_roc_curve(feature_selection_method, model_name, predictions, metrics['AUC'], class_probas)
                self._plot_multiclass_pr_curve(feature_selection_method, model_name, predictions, metrics['AP'], class_probas)
        
        # Confusion Matrix
        if 'Confusion Matrix' in metrics and metrics['Confusion Matrix'].size > 0:
            self._plot_confusion_matrix(feature_selection_method, model_name, metrics['Confusion Matrix'], metrics)
    
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
            precision, recall, _ = precision_recall_curve(predictions['actual_labels'], proba_severe)
            
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
    
    def _plot_multiclass_roc_curve(self, feature_selection_method, model_name, predictions, auc_score, class_probas):
        """다중 클래스 ROC 곡선 플롯"""
        if len(np.unique(predictions['actual_labels'])) <= 1:
            return
        
        try:
            plt.figure(figsize=(10, 8))
            
            # 각 클래스별로 ROC 커브 그리기
            for i, class_name in enumerate(class_probas.keys()):
                # 해당 클래스에 대한 이진 레이블 생성 (one-vs-rest)
                y_true_bin = np.array(predictions['actual_labels']) == i
                
                if np.sum(y_true_bin) == 0:  # 해당 클래스가 없으면 건너뜀
                    continue
                    
                fpr, tpr, _ = roc_curve(y_true_bin, class_probas[class_name])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.4f})')
            
            # 기준선 및 설정
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title(f'Multi-class ROC Curve - {feature_selection_method}_{model_name}\nMacro-average AUC = {auc_score:.4f}', fontsize=16)
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(alpha=0.3)
            
            filename = os.path.join(self.output_dir, f'{model_name}_multiclass_ROC_curve.png')
            plt.savefig(filename)
            plt.close()
            
        except Exception as e:
            print(f"      다중 클래스 ROC 곡선 생성 오류 ({model_name}): {e}")
    
    def _plot_multiclass_pr_curve(self, feature_selection_method, model_name, predictions, ap_score, class_probas):
        """다중 클래스 PR 곡선 플롯"""
        if len(np.unique(predictions['actual_labels'])) <= 1:
            return
        
        try:
            plt.figure(figsize=(10, 8))
            
            # 각 클래스별로 PR 커브 그리기
            for i, class_name in enumerate(class_probas.keys()):
                # 해당 클래스에 대한 이진 레이블 생성 (one-vs-rest)
                y_true_bin = np.array(predictions['actual_labels']) == i
                
                if np.sum(y_true_bin) == 0:  # 해당 클래스가 없으면 건너뜀
                    continue
                    
                precision, recall, _ = precision_recall_curve(y_true_bin, class_probas[class_name])
                pr_auc = auc(recall, precision)
                
                plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {pr_auc:.4f})')
            
            # 설정
            plt.xlabel('Recall (Sensitivity)', fontsize=14)
            plt.ylabel('Precision', fontsize=14)
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'Multi-class Precision-Recall Curve - {feature_selection_method}_{model_name}\nMacro-average AP = {ap_score:.4f}', fontsize=16)
            plt.legend(loc="best", fontsize=12)
            plt.grid(alpha=0.3)
            
            filename = os.path.join(self.output_dir, f'{model_name}_multiclass_PR_curve.png')
            plt.savefig(filename)
            plt.close()
            
        except Exception as e:
            print(f"      다중 클래스 PR 곡선 생성 오류 ({model_name}): {e}")
    
    def _plot_confusion_matrix(self, feature_selection_method, model_name, conf_matrix, metrics):
        """혼동 행렬 플롯 (정규화 비율과 원본 개수를 함께 표시)"""
        try:
            target_names = self.label_encoder.classes_
            
            # 정규화된 혼동 행렬 계산
            conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            
            # 각 셀에 표시할 텍스트 생성 (비율과 개수 함께)
            combined_text = np.empty_like(conf_matrix, dtype=object)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    combined_text[i, j] = f'{conf_matrix_normalized[i, j]:.3f}\n({conf_matrix[i, j]})'
            
            # 성능 지표 문자열 생성
            metrics_text = f"Accuracy: {metrics.get('Accuracy'):.4f} | F1-Score: {metrics.get('F1'):.4f} | AUC: {metrics.get('AUC'):.4f} | AP: {metrics.get('AP'):.4f}"
            
            plt.figure(figsize=(10, 10))
            sns.heatmap(conf_matrix_normalized, annot=combined_text, fmt='',
                       xticklabels=target_names, yticklabels=target_names,
                       cmap='Blues', annot_kws={"size": 20}, vmin=0.0, vmax=1.0,
                       cbar=False, square=True)
            plt.title(f'Confusion Matrix - {feature_selection_method}_{model_name}\n\n{metrics_text}\n(Values: Normalized Ratio (Raw Count))', 
                     fontsize=20, pad=20)
            plt.xlabel('Predicted Label', fontsize=18)
            plt.ylabel('True Label', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.tight_layout()
            
            filename = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
            plt.savefig(filename, dpi=200)
            plt.close()
            
        except Exception as e:
            print(f"      혼동 행렬 생성 오류 ({feature_selection_method}_{model_name}): {e}")