import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from trainer.model_factory import ModelFactory

class ModelTrainer:
    """모델 학습 및 평가를 담당하는 클래스"""
    
    def __init__(self, config, label_encoder):
        self.config = config
        self.label_encoder = label_encoder
        self.model_factory = ModelFactory(config)
        self.models = self.model_factory.create_models()
        self.trained_models = {}
        self.results = {}
        self.prediction_results = {}
    
    def train_and_evaluate(self, x_train, y_train, x_val=None, y_val=None):
        """모든 모델 학습 및 평가"""
        print("--- 모델 학습 시작 ---")
        print(f"  사용할 모델: {list(self.models.keys())}")
        
        if x_train.empty or len(y_train) == 0:
            print("  오류: 학습 데이터가 비어있습니다.")
            return self.results, self.prediction_results
        
        num_unique_classes = len(np.unique(y_train))
        if num_unique_classes < 2:
            print(f"  경고: 클래스가 {num_unique_classes}개만 존재합니다.")
        
        for model_name, model in self.models.items():
            print(f"\n  '{model_name}' 모델 처리 중...")
            
            # 모델 학습
            success = self._train_model(model_name, model, x_train, y_train)
            if not success:
                continue
            
            # 모델 평가 (검증 데이터가 있는 경우)
            if x_val is not None and not x_val.empty and len(y_val) > 0:
                self._evaluate_model(model_name, model, x_val, y_val)
            else:
                print(f"    '{model_name}' 검증 데이터가 없어 평가를 건너뜁니다.")
                self.results[model_name] = self._empty_result()
        
        print("--- 모델 학습 완료 ---\n")
        return self.results, self.prediction_results
    
    def _train_model(self, model_name, model, x_train, y_train):
        """단일 모델 학습"""
        try:
            if x_train.shape[0] < 2 or len(np.unique(y_train)) < 2:
                print(f"    '{model_name}' 학습 건너뛰기 (데이터 부족)")
                self.results[model_name] = self._empty_result()
                return False
            
            model.fit(x_train, y_train)
            self.trained_models[model_name] = model
            print(f"    '{model_name}' 학습 완료")
            return True
            
        except Exception as e:
            print(f"    '{model_name}' 학습 오류: {e}")
            self.results[model_name] = self._empty_result()
            return False
    
    def _evaluate_model(self, model_name, model, x_val, y_val):
        """단일 모델 평가"""
        try:
            print(f"    '{model_name}' 평가 시작...")
            
            # 예측
            y_pred = model.predict(x_val)
            all_class_probas = None

            # 확률 예측 (가능한 경우)
            if hasattr(model, "predict_proba"):
                all_class_probas = model.predict_proba(x_val)
            
            # 메트릭 계산
            metrics = self._calculate_metrics(y_val, y_pred, all_class_probas)
            # 예측 결과 저장
            self._save_predictions(model_name, x_val, y_val, y_pred, all_class_probas)
            # 결과 저장
            self.results[model_name] = metrics
            
            # 결과 출력
            self._print_evaluation_results(model_name, metrics)
        except Exception as e:
            print(f"    '{model_name}' 평가 오류: {e}")
            self.results[model_name] = self._empty_result()
    
    def _calculate_metrics(self, y_val, y_pred, y_pred_proba):
        """평가 메트릭 계산"""
        accuracy = accuracy_score(y_val, y_pred)

        # F1-Score 계산
        if len(np.unique(y_val)) <= 2:  # 이진 분류 또는 단일 클래스
            f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
        else:  # 다중 분류
            f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        # AUC와 AP 초기화
        auc = float('nan')
        ap = float('nan')

        # 클래스 수 확인
        n_classes = len(np.unique(y_val))

        # 모델이 확률을 제공하고, 클래스가 2개 이상일 때
        if y_pred_proba is not None and n_classes > 1:
            try:
                # 이진 분류인 경우
                if n_classes == 2:
                    # 양성 클래스(1)의 확률을 사용
                    auc = roc_auc_score(y_val, y_pred_proba[:, 1])
                    ap = average_precision_score(y_val, y_pred_proba[:, 1])
                # 다중 분류인 경우
                else:
                    # 다중 클래스 AUC 계산 (One-vs-Rest 방식)
                    auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='macro')

                    # 다중 클래스 AP 계산 (Macro Average)
                    ap_scores = []
                    for i in range(n_classes):
                        # 현재 클래스를 양성(1), 나머지를 음성(0)으로 변환
                        y_binary = (y_val == i).astype(int)
                        ap_scores.append(average_precision_score(y_binary, y_pred_proba[:, i]))
                    ap = np.mean(ap_scores)
            except Exception as e:
                print(f"      AUC/AP 계산 오류: {e}")
        
        # 분류 리포트와 혼동 행렬
        labels = self.label_encoder.transform(self.label_encoder.classes_)
        target_names = self.label_encoder.classes_
        
        class_report = classification_report(
            y_val, y_pred,
            target_names=target_names,
            labels=labels,
            output_dict=True,
            zero_division=0
        )
        
        conf_matrix = confusion_matrix(y_val, y_pred, labels=labels)
        
        return {
            'Accuracy': accuracy,
            'F1': f1,
            'AUC': auc,
            'AP': ap,
            'Classification Report': class_report,
            'Confusion Matrix': conf_matrix
        }
    
    def _save_predictions(self, model_name, x_val, y_val, y_pred, all_class_probas=None):
        """예측 결과 저장"""
        prediction_dict = {
            'case_ids': list(x_val.index),
            'actual_labels': y_val.tolist(),
            'actual_labels_str': [self.label_encoder.classes_[label] for label in y_val],
            'predicted_labels': y_pred.tolist(),
            'predicted_labels_str': [self.label_encoder.classes_[label] for label in y_pred],
        }
        
        # 모든 클래스 확률 저장
        if all_class_probas is not None:
            for i, class_name in enumerate(self.label_encoder.classes_):
                prediction_dict[f'proba_{class_name}'] = all_class_probas[:, i].tolist()
        
        self.prediction_results[model_name] = prediction_dict
    
    def _print_evaluation_results(self, model_name, metrics):
        """평가 결과 출력"""
        print(f"\n    --- '{model_name}' 검증 결과 ---")
        print(f"    Accuracy: {metrics['Accuracy']:.4f}")
        print(f"    F1-Score: {metrics['F1']:.4f}")
        print(f"    Area Under Curve: {metrics['AUC']:.4f}")
        print(f"    Average Precision: {metrics['AP']:.4f}")
    
    def _empty_result(self):
        """빈 결과 딕셔너리 반환"""
        return {
            'Accuracy': float('nan'),
            'F1': float('nan'),
            'AUC': float('nan'),
            'AP': float('nan'),
            'Classification Report': {},
            'Confusion Matrix': np.array([])
        }