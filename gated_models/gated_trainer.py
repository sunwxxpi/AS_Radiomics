import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm
from gated_models.gated_model import GatedFusionClassifier


def set_seed(seed=42):
    """완전한 재현성을 위한 모든 random seed 고정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def calculate_class_weights(data_loader, num_classes):
    """학습 데이터셋의 클래스별 분포를 기반으로 가중치를 자동 계산

    Args:
        data_loader (DataLoader): 학습 데이터 로더
        num_classes (int): 클래스 개수

    Returns:
        torch.Tensor: 클래스별 가중치
    """
    class_counts = torch.zeros(num_classes)

    # 전체 데이터에서 클래스별 개수 계산
    for batch in data_loader:
        labels = batch['label']
        for label in labels:
            class_counts[label] += 1

    # 역가중치 계산 (전체 샘플 수 / (클래스 수 × 클래스별 샘플 수))
    total_samples = class_counts.sum()
    weights = total_samples / (num_classes * class_counts)

    # 가중치가 inf가 되는 경우 방지
    weights = torch.where(class_counts == 0, torch.tensor(1.0), weights)

    return weights


class GatedFusionDataset(Dataset):
    """Radiomics + DL Features Dataset"""

    def __init__(self, radiomics_features, dl_features, labels):
        """
        Args:
            radiomics_features (np.ndarray): [N, radiomics_dim]
            dl_features (np.ndarray): [N, dl_dim]
            labels (np.ndarray): [N]
        """
        self.radiomics_features = torch.FloatTensor(radiomics_features)
        self.dl_features = torch.FloatTensor(dl_features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'radiomics': self.radiomics_features[idx],
            'dl': self.dl_features[idx],
            'label': self.labels[idx]
        }


class GatedFusionTrainer:
    """Gated Fusion Model 학습 클래스

    Args:
        model_config (dict): 모델 설정
        train_config (dict): 학습 설정
        output_dir (str): 결과 저장 디렉토리
    """

    def __init__(self, model_config, train_config, output_dir, random_seed=42):
        self.model_config = model_config
        self.train_config = train_config
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(output_dir, exist_ok=True)

        self.scaler_radiomics = StandardScaler()
        self.scaler_dl = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Random seed 고정
        set_seed(self.random_seed)

        self._setup_logging()

    def _setup_logging(self):
        """로깅 설정 (fold별 독립적인 로거)"""
        log_file = os.path.join(self.output_dir, 'gated_training.log')

        # 기존 핸들러 제거 (fold별 독립 로깅)
        logger = logging.getLogger(f'gated_trainer_{self.output_dir}')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # 포매터 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, mode='w')  # 덮어쓰기 모드
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self.logger = logger

    def prepare_data(self, features_df):
        """데이터 전처리 및 분할

        Args:
            features_df (pd.DataFrame): Radiomics + DL features DataFrame
                - 'case_id', 'severity', 'data_source' 컬럼 포함
                - radiomics features: 'original_*', 'wavelet_*' 등
                - dl features: 'dl_embedding_feature_*'

        Returns:
            dict: 전처리된 데이터
        """
        self.logger.info("데이터 전처리 시작...")

        # 특징 컬럼 분리
        radiomics_cols = [col for col in features_df.columns
                          if col.startswith(('original_', 'wavelet_', 'log_', 'square_', 'squareroot_', 'exponential_', 'logarithm_', 'gradient_', 'lbp-'))]
        dl_cols = [col for col in features_df.columns
                   if col.startswith('dl_embedding_feature_')]

        if not radiomics_cols:
            raise ValueError("Radiomics 특징을 찾을 수 없습니다.")
        if not dl_cols:
            raise ValueError("DL embedding 특징을 찾을 수 없습니다.")

        self.logger.info(f"Radiomics 특징 수: {len(radiomics_cols)}")
        self.logger.info(f"DL embedding 특징 수: {len(dl_cols)}")

        # 특징 추출
        X_radiomics = features_df[radiomics_cols].values
        X_dl = features_df[dl_cols].values
        y = features_df['severity'].values

        # 레이블 인코딩
        desired_order = ['normal', 'nonsevere', 'severe']
        unique_classes = np.unique(y)
        classes_to_fit = [cls for cls in desired_order if cls in unique_classes]
        remaining_classes = sorted([cls for cls in unique_classes if cls not in desired_order])
        classes_to_fit.extend(remaining_classes)

        self.label_encoder.classes_ = np.array(classes_to_fit, dtype=object)
        y_encoded = self.label_encoder.transform(y)
        self.logger.info(f"클래스 매핑: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # 정규화
        X_radiomics_scaled = self.scaler_radiomics.fit_transform(X_radiomics)
        X_dl_scaled = self.scaler_dl.fit_transform(X_dl)

        return {
            'X_radiomics': X_radiomics_scaled,
            'X_dl': X_dl_scaled,
            'y': y_encoded,
            'radiomics_dim': len(radiomics_cols),
            'dl_dim': len(dl_cols),
            'num_classes': len(self.label_encoder.classes_)
        }

    def train_fold(self, train_loader, val_loader, model, fold_idx):
        """단일 fold 학습

        Args:
            train_loader (DataLoader): 학습 데이터 로더
            val_loader (DataLoader): 검증 데이터 로더
            model (nn.Module): Gated Fusion 모델
            fold_idx (int): Fold 번호

        Returns:
            dict: 학습 결과
        """
        # Loss function - 자동 가중치 계산
        num_classes = self.model_config['num_classes']
        weights = calculate_class_weights(train_loader, num_classes).to(self.device)
        self.logger.info(f"Calculated weights for CrossEntropyLoss: {weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_config['epochs'],
            eta_min=self.train_config['min_lr']
        )

        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(self.train_config['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            with tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch+1}/{self.train_config['epochs']}") as pbar:
                for batch in pbar:
                    radiomics = batch['radiomics'].to(self.device)
                    dl = batch['dl'].to(self.device)
                    labels = batch['label'].to(self.device)

                    optimizer.zero_grad()
                    outputs = model(radiomics, dl)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                    pbar.set_postfix({'loss': loss.item(), 'acc': 100. * train_correct / train_total})

            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total

            # Validation
            val_loss, val_acc = self._validate(model, val_loader, criterion)

            # Scheduler step
            scheduler.step()

            self.logger.info(
                f"Fold {fold_idx} Epoch {epoch+1}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Early stopping & model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0

                # Save best model
                model_save_path = os.path.join(self.output_dir, f'fold_{fold_idx}_best_model.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, model_save_path)
                self.logger.info(f"Best model saved: Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= self.train_config['patience']:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

        return {
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc
        }

    def _validate(self, model, val_loader, criterion):
        """검증"""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                radiomics = batch['radiomics'].to(self.device)
                dl = batch['dl'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(radiomics, dl)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train_with_cv(self, features_df, n_folds=5, external_fold_idx=None):
        """K-fold 교차검증 학습 (DL Classification과 동일한 split 방식)

        Args:
            features_df (pd.DataFrame): Radiomics + DL features
            n_folds (int): Fold 개수 (기본값: 5)
            external_fold_idx (int): 특정 fold만 학습 (1~5), None이면 전체 fold 학습

        Returns:
            list: Fold별 결과
        """
        # 데이터 전처리
        data = self.prepare_data(features_df)

        X_radiomics = data['X_radiomics']
        X_dl = data['X_dl']
        y = data['y']

        fold_results = []

        # DL Classification과 동일한 StratifiedKFold 사용
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(skf.split(X_radiomics, y))

        # external_fold_idx가 지정된 경우 해당 fold만 학습
        if external_fold_idx is not None:
            if external_fold_idx < 1 or external_fold_idx > n_folds:
                raise ValueError(f"external_fold_idx는 1~{n_folds} 사이여야 합니다.")

            # 해당 fold만 선택 (fold_idx는 1부터 시작)
            selected_split = [(external_fold_idx, splits[external_fold_idx - 1])]
        else:
            # 전체 fold 학습
            selected_split = [(i + 1, split) for i, split in enumerate(splits)]

        for fold_idx, (train_idx, val_idx) in selected_split:
            # 각 fold 시작 시 seed 재설정 (완전한 재현성)
            set_seed(42)
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Fold {fold_idx}/{n_folds if n_folds > 1 else 'Single'} 학습 시작")
            self.logger.info(f"{'='*50}")

            # Split data
            X_rad_train, X_rad_val = X_radiomics[train_idx], X_radiomics[val_idx]
            X_dl_train, X_dl_val = X_dl[train_idx], X_dl[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create datasets
            train_dataset = GatedFusionDataset(X_rad_train, X_dl_train, y_train)
            val_dataset = GatedFusionDataset(X_rad_val, X_dl_val, y_val)

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_config['batch_size'],
                shuffle=True,
                num_workers=4,
                drop_last=True  # BatchNorm 오류 방지
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.train_config['batch_size'],
                shuffle=False,
                num_workers=4,
                drop_last=False  # Validation은 모든 샘플 사용
            )

            # model_config 업데이트 (자동 설정)
            self.model_config['radiomics_dim'] = data['radiomics_dim']
            self.model_config['dl_dim'] = data['dl_dim']
            self.model_config['num_classes'] = data['num_classes']

            # Create model
            model = GatedFusionClassifier(
                radiomics_dim=data['radiomics_dim'],
                dl_dim=data['dl_dim'],
                num_classes=data['num_classes'],
                fusion_dim=self.model_config.get('fusion_dim'),
                hidden_dims=self.model_config.get('hidden_dims', [256, 128]),
                dropout=self.model_config.get('dropout', 0.3)
            )

            model = model.to(self.device)

            # Train fold
            result = self.train_fold(train_loader, val_loader, model, fold_idx)
            fold_results.append(result)

            self.logger.info(f"Fold {fold_idx} 완료: Val Loss {result['best_val_loss']:.4f}, Val Acc {result['best_val_acc']:.2f}%")

        # Summary
        avg_val_loss = np.mean([r['best_val_loss'] for r in fold_results])
        avg_val_acc = np.mean([r['best_val_acc'] for r in fold_results])

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"전체 교차검증 결과")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Average Val Loss: {avg_val_loss:.4f}")
        self.logger.info(f"Average Val Acc: {avg_val_acc:.2f}%")

        return fold_results

    def evaluate_final_performance(self, val_loader, model_path, fold_idx):
        """학습 완료 후 최종 성능 평가

        Args:
            val_loader (DataLoader): Validation 데이터 로더
            model_path (str): 저장된 best model 경로
            fold_idx (int): Fold 번호

        Returns:
            dict: 평가 결과
        """
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Fold {fold_idx} 최종 성능 평가 시작")
        self.logger.info(f"{'='*50}")

        # Best model 로드
        if not os.path.exists(model_path):
            self.logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return None

        checkpoint = torch.load(model_path, map_location=self.device)

        # 모델 생성
        radiomics_dim = self.model_config['radiomics_dim']
        dl_dim = self.model_config['dl_dim']
        num_classes = self.model_config['num_classes']

        model = GatedFusionClassifier(
            radiomics_dim=radiomics_dim,
            dl_dim=dl_dim,
            num_classes=num_classes,
            fusion_dim=self.model_config.get('fusion_dim'),
            hidden_dims=self.model_config.get('hidden_dims', [256, 128]),
            dropout=0.0  # Evaluation에서는 dropout 비활성화
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        self.logger.info(f"Best model 로드 완료 (Epoch {checkpoint['epoch']})")

        # Validation set에 대한 예측
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold_idx} Evaluation"):
                radiomics = batch['radiomics'].to(self.device)
                dl = batch['dl'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(radiomics, dl)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 성능 지표 계산
        accuracy = accuracy_score(all_labels, all_preds)

        # Multi-class: macro average
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        # Per-class metrics
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)

        # 결과 로깅
        self.logger.info(f"\n=== Fold {fold_idx} 최종 성능 ===")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"F1-Score (macro): {f1_macro:.4f}")
        self.logger.info(f"Precision (macro): {precision_macro:.4f}")
        self.logger.info(f"Recall (macro): {recall_macro:.4f}")

        self.logger.info(f"\n=== Per-Class Metrics ===")
        for i, class_name in enumerate(self.label_encoder.classes_):
            self.logger.info(f"{class_name}:")
            self.logger.info(f"  F1: {f1_per_class[i]:.4f}, Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}")

        self.logger.info(f"\n=== Confusion Matrix ===")
        self.logger.info(f"\n{cm}")

        # Classification Report
        self.logger.info(f"\n=== Classification Report ===")
        report = classification_report(all_labels, all_preds, target_names=self.label_encoder.classes_, zero_division=0)
        self.logger.info(f"\n{report}")

        # 결과 딕셔너리
        results = {
            'fold': fold_idx,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'class_names': self.label_encoder.classes_
        }

        # Per-class metrics 추가
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f'f1_{class_name}'] = f1_per_class[i]
            results[f'precision_{class_name}'] = precision_per_class[i]
            results[f'recall_{class_name}'] = recall_per_class[i]

        return results