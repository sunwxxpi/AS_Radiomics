import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.preprocessing import label_binarize, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class SoftVotingEnsemble:
    """딥러닝 모델과 전통적 ML 모델들의 확률값을 결합하여 Soft Voting Ensemble 수행"""

    def __init__(self, classification_mode: str = 'multi'):
        """
        Args:
            classification_mode (str): 분류 모드 ('binary' 또는 'multi')
        """
        self.classification_mode = classification_mode

        # 클래스 레이블 정의
        if classification_mode == 'binary':
            self.class_labels = ['nonsevere', 'severe']
        else:  # multi
            self.class_labels = ['normal', 'nonsevere', 'severe']

        # 레이블 매핑 (pd.Series.map()에서 사용)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.class_labels)}

        # LabelEncoder 초기화
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_labels)

    def load_dl_probs(self, dl_probs_path: str) -> pd.DataFrame:
        """딥러닝 모델의 확률값 로드

        Args:
            dl_probs_path (str): DL classification probs CSV 파일 경로

        Returns:
            pd.DataFrame: case_id를 인덱스로 하는 확률값 데이터프레임
        """
        df = pd.read_csv(dl_probs_path)

        # 필요한 컬럼만 추출
        prob_cols = [f'proba_{label}' for label in self.class_labels]
        df_probs = df[['case_id', 'true_label_str'] + prob_cols].copy()
        df_probs.set_index('case_id', inplace=True)

        return df_probs

    def load_radiomics_probs(self, radiomics_probs_path: str, models: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Radiomics 분석 결과에서 ML 모델들의 확률값 로드

        Args:
            radiomics_probs_path (str): Radiomics 분석 결과 CSV 파일 경로
            models (List[str]): 사용할 모델 리스트 (예: ['LR', 'SVM', 'RF'])

        Returns:
            Tuple[pd.DataFrame, List[str]]: (확률값 데이터프레임, 모델 리스트)
        """
        df = pd.read_csv(radiomics_probs_path)

        if models is None:
            models = ['LR', 'SVM', 'RF']

        # case_id를 인덱스로 설정
        df.set_index('case_id', inplace=True)

        return df, models

    def perform_soft_voting(
        self,
        dl_probs_path: str,
        radiomics_probs_path: str,
        models: List[str] = None,
        output_path: str = None,
        feature_selection_method: str = 'lasso'
    ) -> Tuple[pd.DataFrame, Dict]:
        """Soft Voting Ensemble 수행

        Args:
            dl_probs_path (str): DL classification probs CSV 파일 경로
            radiomics_probs_path (str): Radiomics 분석 결과 CSV 파일 경로
            models (List[str]): 사용할 ML 모델 리스트
            output_path (str): 결과 저장 경로 (None이면 저장하지 않음)
            feature_selection_method (str): 특징 선택 방법 (시각화에 사용)

        Returns:
            Tuple[pd.DataFrame, Dict]: (앙상블 결과 데이터프레임, 성능 메트릭 딕셔너리)
        """
        logger.info("=== Soft Voting Ensemble 시작 ===")

        # 1. DL 확률값 로드
        logger.info(f"DL 확률값 로드 중: {dl_probs_path}")
        df_dl = self.load_dl_probs(dl_probs_path)

        # 2. Radiomics 확률값 로드
        logger.info(f"Radiomics 확률값 로드 중: {radiomics_probs_path}")
        df_radiomics, models = self.load_radiomics_probs(radiomics_probs_path, models)

        # 3. case_id 교집합 확인
        common_cases = df_dl.index.intersection(df_radiomics.index)
        logger.info(f"공통 케이스 수: {len(common_cases)}")

        if len(common_cases) == 0:
            raise ValueError("DL과 Radiomics 결과에 공통 케이스가 없습니다.")

        # 4. 공통 케이스만 필터링
        df_dl = df_dl.loc[common_cases]
        df_radiomics = df_radiomics.loc[common_cases]

        # 5. Ensemble 결과 저장용 데이터프레임 초기화
        results = pd.DataFrame(index=common_cases)
        results['true_label_str'] = df_dl['true_label_str']
        results['true_label'] = results['true_label_str'].map(self.label_to_idx)

        # 6. DL 확률값 추가
        for label in self.class_labels:
            results[f'DL_proba_{label}'] = df_dl[f'proba_{label}']

        # 7. 각 ML 모델의 확률값 추가
        for model in models:
            for label in self.class_labels:
                col_name = f'{model}_{label}_probability'
                if col_name in df_radiomics.columns:
                    results[f'{model}_proba_{label}'] = df_radiomics[col_name]
                else:
                    logger.warning(f"컬럼을 찾을 수 없음: {col_name}")

        # 8. 다양한 앙상블 조합 계산
        ensemble_configs = self._create_ensemble_configs(models)

        for ensemble_name, model_list in ensemble_configs.items():
            for label in self.class_labels:
                prob_cols = [f'{model}_proba_{label}' for model in model_list]
                # 존재하는 컬럼만 선택
                available_cols = [col for col in prob_cols if col in results.columns]
                if available_cols:
                    results[f'{ensemble_name}_proba_{label}'] = results[available_cols].mean(axis=1)

        # 9. 각 앙상블의 최종 예측
        all_ensembles = list(ensemble_configs.keys())
        for ensemble_name in all_ensembles:
            ensemble_prob_cols = [f'{ensemble_name}_proba_{label}' for label in self.class_labels]
            if all(col in results.columns for col in ensemble_prob_cols):
                results[f'{ensemble_name}_predicted_idx'] = results[ensemble_prob_cols].values.argmax(axis=1)
                results[f'{ensemble_name}_predicted'] = results[f'{ensemble_name}_predicted_idx'].map(self.idx_to_label)

        # 10. 개별 모델의 예측 결과 추가 (원본 예측 사용)
        # DL 예측
        dl_prob_cols = [f'DL_proba_{label}' for label in self.class_labels]
        if all(col in results.columns for col in dl_prob_cols):
            results['DL_predicted_idx'] = results[dl_prob_cols].values.argmax(axis=1)
            results['DL_predicted'] = results['DL_predicted_idx'].map(self.idx_to_label)

        # ML 모델 예측 (원본 CSV에서 가져오기)
        for model in models:
            pred_col = f'{model}_predicted'
            if pred_col in df_radiomics.columns:
                results[pred_col] = df_radiomics.loc[common_cases, pred_col]

        # 11. 성능 메트릭 계산
        metrics = self._calculate_metrics(results, all_ensembles, models)

        # 12. 결과 저장
        if output_path:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)

            # 전체 결과 CSV 저장
            results.to_csv(output_path)
            logger.info(f"Ensemble 결과 저장: {output_path}")

            # model_validation_summary.csv 형식으로 저장
            self._save_model_summary(metrics, output_dir)

            # 시각화 생성
            self._generate_visualizations(results, metrics, output_dir, feature_selection_method, all_ensembles, models)

        logger.info("=== Soft Voting Ensemble 완료 ===")

        return results, metrics

    def _create_ensemble_configs(self, models: List[str]) -> Dict[str, List[str]]:
        """앙상블 구성 생성

        Args:
            models (List[str]): ML 모델 리스트

        Returns:
            Dict[str, List[str]]: {앙상블 이름: 모델 리스트}
        """
        configs = {}

        # 2-모델 조합: DL + 각 ML 모델
        for model in models:
            configs[f'DL+{model}'] = ['DL', model]

        return configs

    def _calculate_metrics(self, results: pd.DataFrame, ensemble_names: List[str], ml_models: List[str]) -> Dict:
        """각 모델 및 앙상블의 성능 메트릭 계산

        Args:
            results (pd.DataFrame): 앙상블 결과 데이터프레임
            ensemble_names (List[str]): 평가할 앙상블 이름 리스트
            ml_models (List[str]): ML 모델 리스트

        Returns:
            Dict: 모델별 성능 메트릭
        """
        y_true = results['true_label_str']
        y_true_encoded = results['true_label']
        metrics = {}

        # 평가할 모든 모델 리스트
        all_models_to_eval = ['DL'] + ml_models + ensemble_names

        for model_name in all_models_to_eval:
            pred_col = f'{model_name}_predicted'
            if pred_col not in results.columns:
                continue

            y_pred = results[pred_col]

            # 확률값 가져오기
            prob_cols = [f'{model_name}_proba_{label}' for label in self.class_labels]
            if all(col in results.columns for col in prob_cols):
                y_proba = results[prob_cols].values
            else:
                continue

            metrics[model_name] = self._compute_single_metrics(
                y_true, y_pred, y_true_encoded, y_proba
            )

        # 메트릭 로깅
        logger.info("\n=== 모델별 성능 비교 ===")
        for model_name, model_metrics in metrics.items():
            logger.info(f"{model_name}: Accuracy={model_metrics['Accuracy']:.4f}, "
                       f"F1={model_metrics['F1']:.4f}, "
                       f"AUC={model_metrics['AUC']:.4f}, "
                       f"AP={model_metrics['AP']:.4f}")

        return metrics

    def _compute_single_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_true_encoded: pd.Series,
        y_proba: np.ndarray
    ) -> Dict:
        """단일 모델의 성능 메트릭 계산

        Args:
            y_true (pd.Series): 실제 레이블 (문자열)
            y_pred (pd.Series): 예측 레이블 (문자열)
            y_true_encoded (pd.Series): 실제 레이블 (인코딩된 정수)
            y_proba (np.ndarray): 클래스별 확률값

        Returns:
            Dict: 성능 메트릭
        """
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=self.class_labels)

        # 기본 메트릭
        accuracy = accuracy_score(y_true, y_pred)

        # F1-Score 계산 (trainer.py와 동일한 방식)
        if len(self.class_labels) <= 2:  # Binary classification
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        else:  # Multi-class classification
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # AUC 계산 (multi-class의 경우 macro-average)
        try:
            if len(self.class_labels) == 2:
                # Binary classification
                auc_score = roc_auc_score(y_true_encoded, y_proba[:, 1])
            else:
                # Multi-class classification (one-vs-rest macro-average)
                y_true_bin = label_binarize(y_true_encoded, classes=range(len(self.class_labels)))
                auc_score = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        except Exception as e:
            logger.warning(f"AUC 계산 오류: {e}")
            auc_score = 0.0

        # AP (Average Precision) 계산
        try:
            if len(self.class_labels) == 2:
                # Binary classification
                ap_score = average_precision_score(y_true_encoded, y_proba[:, 1])
            else:
                # Multi-class classification (macro-average)
                y_true_bin = label_binarize(y_true_encoded, classes=range(len(self.class_labels)))
                ap_score = average_precision_score(y_true_bin, y_proba, average='macro')
        except Exception as e:
            logger.warning(f"AP 계산 오류: {e}")
            ap_score = 0.0

        return {
            'Accuracy': accuracy,
            'F1': f1,
            'AUC': auc_score,
            'AP': ap_score,
            'Confusion Matrix': conf_matrix,
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }

    def _save_model_summary(self, metrics: Dict, output_dir: str):
        """model_validation_summary.csv 형식으로 저장

        Args:
            metrics (Dict): 모델별 메트릭
            output_dir (str): 출력 디렉토리
        """
        # 저장 순서 정의: DL, LR, RF, SVM, DL+LR, DL+RF, DL+SVM
        model_order = ['DL', 'LR', 'RF', 'SVM', 'DL+LR', 'DL+RF', 'DL+SVM']

        summary_data = []

        # 정의된 순서대로 추가
        for model_name in model_order:
            if model_name in metrics:
                model_metrics = metrics[model_name]
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': model_metrics['Accuracy'],
                    'F1': model_metrics['F1'],
                    'AUC': model_metrics['AUC'],
                    'AP': model_metrics['AP']
                })

        # 순서에 없는 모델이 있다면 마지막에 추가
        for model_name, model_metrics in metrics.items():
            if model_name not in model_order:
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': model_metrics['Accuracy'],
                    'F1': model_metrics['F1'],
                    'AUC': model_metrics['AUC'],
                    'AP': model_metrics['AP']
                })

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'ensemble_model_validation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"모델 성능 요약 저장: {summary_path}")

    def _generate_visualizations(
        self,
        results: pd.DataFrame,
        metrics: Dict,
        output_dir: str,
        feature_selection_method: str,
        ensemble_names: List[str],
        ml_models: List[str]
    ):
        """Confusion Matrix, ROC Curve, PR Curve 시각화 생성

        Args:
            results (pd.DataFrame): 앙상블 결과
            metrics (Dict): 모델별 메트릭
            output_dir (str): 출력 디렉토리
            feature_selection_method (str): 특징 선택 방법
            ensemble_names (List[str]): 앙상블 이름 리스트
            ml_models (List[str]): ML 모델 리스트
        """
        from utils.plotter import ResultPlotter

        plotter = ResultPlotter(output_dir, self.label_encoder)

        # 평가할 모든 모델
        all_models = ['DL'] + ml_models + ensemble_names

        for model_name in all_models:
            if model_name not in metrics:
                continue

            # 해당 모델의 예측 결과 구성
            predictions = {
                'actual_labels': results['true_label'].values,
            }

            # 확률값 추가
            for label in self.class_labels:
                prob_col = f'{model_name}_proba_{label}'
                if prob_col in results.columns:
                    predictions[f'proba_{label}'] = results[prob_col].values

            # 시각화 생성
            try:
                plotter._plot_model_results(
                    feature_selection_method,
                    f'ensemble_{model_name}',
                    predictions,
                    metrics[model_name]
                )
                logger.info(f"{model_name} 시각화 생성 완료")
            except Exception as e:
                logger.warning(f"{model_name} 시각화 생성 오류: {e}")


def run_ensemble_for_fold(
    fold: int,
    dl_results_dir: str,
    radiomics_results_dir: str,
    classification_mode: str,
    models: List[str] = None,
    feature_selection_method: str = 'lasso'
) -> Tuple[pd.DataFrame, Dict]:
    """특정 fold에 대한 ensemble 수행

    Args:
        fold (int): Fold 번호
        dl_results_dir (str): DL 결과 디렉토리 (probs 폴더가 있는 경로)
        radiomics_results_dir (str): Radiomics 결과 디렉토리
        classification_mode (str): 분류 모드
        models (List[str]): 사용할 ML 모델 리스트
        feature_selection_method (str): 특징 선택 방법

    Returns:
        Tuple[pd.DataFrame, Dict]: (앙상블 결과, 성능 메트릭)
    """
    # DL probs 경로
    dl_probs_path = os.path.join(dl_results_dir, 'probs', f'fold_{fold}.csv')

    # Radiomics probs 경로
    radiomics_probs_path = os.path.join(radiomics_results_dir, 'test_cases_prediction_results.csv')

    # 파일 존재 확인
    if not os.path.exists(dl_probs_path):
        raise FileNotFoundError(f"DL probs 파일을 찾을 수 없습니다: {dl_probs_path}")

    if not os.path.exists(radiomics_probs_path):
        raise FileNotFoundError(f"Radiomics probs 파일을 찾을 수 없습니다: {radiomics_probs_path}")

    # Ensemble 수행
    ensemble = SoftVotingEnsemble(classification_mode=classification_mode)

    # ensemble 디렉토리 생성 및 출력 경로 설정
    ensemble_dir = os.path.join(radiomics_results_dir, 'ensemble')
    os.makedirs(ensemble_dir, exist_ok=True)
    output_path = os.path.join(ensemble_dir, f'ensemble_results_fold_{fold}.csv')

    results, metrics = ensemble.perform_soft_voting(
        dl_probs_path=dl_probs_path,
        radiomics_probs_path=radiomics_probs_path,
        models=models,
        output_path=output_path,
        feature_selection_method=feature_selection_method
    )

    return results, metrics