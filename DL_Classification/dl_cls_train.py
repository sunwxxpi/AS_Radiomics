import os
import random
import math
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dl_cls_dataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from monai.data import worker_init_fn
from monai.utils import set_determinism
from dl_cls_config import load_config
from dl_cls_model import create_model
from dl_cls_valid import validate
# dl_cls_dataset 이 부모 디렉토리를 sys.path 에 넣은 뒤라야 찾을 수 있다.
from config import Config


def seed_torch(seed=1):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.enabled = True
    # MONAI 의 Rand* 는 위 시드가 닿지 않는 자기 RandomState 를 들고 있다. Compose 를 만들기 전에 불러야 값이 심긴다.
    set_determinism(seed)
    
    
def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)

    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)

    return conf_matrix


def save_results(model_save_path, filename, epoch, loss, val_acc, f1score, auc, spe, sen, pre, mode='a'):
    with open(os.path.join(model_save_path, filename), mode) as f:
        f.write(f'Result: (Epoch {epoch})\n')
        f.write('Loss: %f, Acc: %f, F1 Score: %f, AUC: %f, Spe: %f, Sen: %f, Pre: %f' % (loss, val_acc, f1score, auc, spe, sen, pre))


def check_fold_assignment_match(recorded, assignment, csv_path):
    """지금 계산한 fold 배정이 기록해 둔 배정과 같은지 본다.

    데이터 파일이 하나만 늘거나 줄어도 배정이 통째로 바뀌어, 재사용한 옛 가중치는 자기가 학습에 쓴 케이스를 검증으로 받는다.
    그러면 OOF embedding 이 조용히 샌다.
    """
    absent = sorted({'patient_id', 'fold'} - set(recorded.columns))
    if absent:
        raise ValueError(f"{csv_path}: fold 배정 기록에 {absent} 컬럼이 없다 — 5-fold 를 다시 돌린다")

    recorded_fold = {str(row.patient_id): int(row.fold) for row in recorded.itertuples()}
    current_fold = {str(row.patient_id): int(row.fold) for row in assignment.itertuples()}

    added = sorted(set(current_fold) - set(recorded_fold))
    removed = sorted(set(recorded_fold) - set(current_fold))
    moved = sorted(pid for pid in set(recorded_fold) & set(current_fold) if recorded_fold[pid] != current_fold[pid])
    if not (added or removed or moved):
        return

    def brief(patient_ids):
        return f"{len(patient_ids)}건 {patient_ids[:10]}"

    raise ValueError(
        f"{csv_path}: 지금 계산한 fold 배정이 기록과 다르다 "
        f"(새로 생긴 케이스 {brief(added)}, 없어진 케이스 {brief(removed)}, fold 가 바뀐 케이스 {brief(moved)}) — "
        f"이미 있는 가중치는 이 배정으로 학습된 것이 아니다 — 새로 돌릴 거면 그 가중치 디렉토리를 지운다"
    )


def save_fold_assignment(dataset, splits, save_path, require_match=False):
    """5-fold 배정을 `patient_id` 단위로 CSV 에 남긴다.

    `fold` 는 그 케이스가 검증으로 들어간 fold 번호이고, fold k 의 학습셋은 `fold != k` 인 행 전부다.
    배정이 정렬 없는 `glob` 순서에 걸려 있어 이 파일 말고는 radiomics 쪽에서 같은 분할을 재현할 길이 없다.
    파일명이 규약과 어긋나거나 배정이 케이스마다 한 번씩 걸리지 않으면 GPU 시간을 쓰기 전에 멈춘다.
    `require_match` 면 이미 있는 배정과 달라도 덮어쓰기 전에 멈춘다 — 남아 있는 가중치가 어느 분할로 학습된 것인지 알 수 없게 된다.
    """
    fold_of = np.zeros(len(dataset.image_files), dtype=np.int64)
    for fold, (_, val_idx) in enumerate(splits, start=1):
        overlapped = np.count_nonzero(fold_of[val_idx])
        if overlapped:
            raise ValueError(f"fold {fold}: 다른 fold 의 검증에 이미 들어간 케이스가 {overlapped}건 있다")
        fold_of[val_idx] = fold

    unassigned = np.count_nonzero(fold_of == 0)
    if unassigned:
        raise ValueError(f"어느 fold 의 검증에도 안 들어간 케이스가 {unassigned}건 있다")

    rows = []
    for index, image_path in enumerate(dataset.image_files):
        image_file = os.path.basename(image_path)
        match = re.match(r'([A-Za-z0-9\.\-]+)_(\d{4,})_0000\.nii\.gz', image_file)
        if not match:
            raise ValueError(f"파일명 규약 불일치: {image_file}")
        patient_id = match.group(1).strip()
        rows.append({
            'patient_id': patient_id,
            'case_id': f"{patient_id}_{match.group(2).strip()}",
            'severity': dataset.labels[index],
            'label_idx': int(dataset.encoded_labels[index]),
            'fold': int(fold_of[index]),
            'dataset_index': index,
        })

    assignment = pd.DataFrame(rows).sort_values('patient_id', ignore_index=True)
    duplicated = sorted(set(assignment.loc[assignment['patient_id'].duplicated(), 'patient_id']))
    if duplicated:
        raise ValueError(f"patient_id 가 겹쳐 조인 키로 못 쓴다: {duplicated}")

    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, 'cls_fold_assignment.csv')
    if require_match and os.path.exists(csv_path):
        try:
            recorded = pd.read_csv(csv_path, dtype={'patient_id': str})
        except Exception as e:
            raise ValueError(f"{csv_path}: fold 배정 기록을 읽을 수 없다 ({e}) — 5-fold 를 다시 돌린다") from e
        check_fold_assignment_match(recorded, assignment, csv_path)
    assignment.to_csv(csv_path, index=False)
    counts = assignment['fold'].value_counts().sort_index().to_dict()
    print(f"✓ fold 배정 저장: {csv_path} (fold 별 검증 건수 {counts})")
    return assignment


# 인자 이름이 frame 의 `fold` 컬럼과 겹쳐 접두사를 붙인다.
ARG_COLUMN_PREFIX = 'arg_'

# 단계마다 정당하게 달라지는 인자와 load_config 가 이미 Config 와 맞춰 둔 경로 인자. 나머지 학습 인자는 전부 기록하고 견준다.
UNCOMPARED_ARG_KEYS = ('stage', 'resume', 'enable_cam', 'save_model', 'model_path', 'writer_comment')


def compared_arg_keys(config):
    """기록하고 견줄 학습 인자 이름"""
    return sorted(key for key in vars(config) if key not in UNCOMPARED_ARG_KEYS)


def check_training_args(frame, config, csv_path):
    """기록의 학습 인자가 지금 인자와 같은지 본다.

    writer_comment 가 model_type·img_size·데이터셋만 담아 나머지 인자를 바꿔도 같은 디렉토리를 가리키므로,
    이 검사가 없으면 다른 인자로 뽑은 best epoch 이 그대로 refit 종료 epoch 이 된다.
    """
    keys = compared_arg_keys(config)
    recorded_columns = {column for column in frame.columns if column.startswith(ARG_COLUMN_PREFIX)}
    expected_columns = {ARG_COLUMN_PREFIX + key for key in keys}
    absent = sorted(expected_columns - recorded_columns)
    unknown = sorted(recorded_columns - expected_columns)
    if absent or unknown:
        raise ValueError(
            f"{csv_path}: 기록된 학습 인자 목록이 지금과 다르다 (없는 컬럼 {absent}, 모르는 컬럼 {unknown}) — 5-fold 를 다시 돌린다"
        )

    # CSV 왕복에서 타입이 바뀌므로 문자열로 견준다.
    mismatched = []
    for key in keys:
        recorded = sorted({str(value) for value in frame[ARG_COLUMN_PREFIX + key].tolist()})
        current = str(getattr(config, key))
        if recorded != [current]:
            mismatched.append(f"{key}: 기록 {recorded} vs 지금 {current}")

    if mismatched:
        raise ValueError(
            f"{csv_path}: 기록된 학습 인자가 지금 인자와 다르다 ({'; '.join(mismatched)}) — "
            f"같은 인자로 다시 부르거나, 인자를 바꿀 생각이면 5-fold 를 다시 돌린다"
        )


def fold_weight_path(config, fold):
    """fold 별 best 가중치 경로"""
    return os.path.join(config.model_path, config.writer_comment, str(fold), 'best_model.pth')


def merge_recorded_folds(fold_best, recorded_folds, config):
    """이번 실행에서 아직 안 돈 fold 는 가중치가 남아 있을 때만 기록을 이어 붙인다.

    저장이 `fold_best` 만 쓰면 뒤쪽 fold 의 기록이 지워져, 가중치가 멀쩡한데도 다시 학습하게 된다.
    가중치 없는 fold 의 기록까지 남기면 반대로 CSV 만 다 돈 것처럼 보여 `--stage refit` 이 없는 모델 위에서 그냥 돈다.
    """
    processed = {row['fold'] for row in fold_best}
    pending = [{'fold': fold, 'best_epoch': int(row['best_epoch']), 'best_val_loss': float(row['best_val_loss'])}
               for fold, row in sorted(recorded_folds.items())
               if fold not in processed and os.path.exists(fold_weight_path(config, fold))]
    return fold_best + pending


def save_fold_best_epochs(fold_best, save_path, config):
    """fold 별 best epoch 과 그때의 val loss 를 학습 인자와 함께 CSV 로 남긴다.

    refit 종료 epoch 이 이 값들의 중앙값이라 `--stage refit` 로 따로 돌릴 때는 이 파일이 유일한 근거다.
    fold 하나가 끝날 때마다 다시 써서 중간에 죽어도 앞선 fold 의 기록이 남는다.
    """
    frame = pd.DataFrame(fold_best).sort_values('fold', ignore_index=True)
    for key in compared_arg_keys(config):
        frame[ARG_COLUMN_PREFIX + key] = str(getattr(config, key))

    os.makedirs(save_path, exist_ok=True)
    csv_path = os.path.join(save_path, 'fold_best_epochs.csv')
    frame.to_csv(csv_path, index=False)
    print(f"✓ fold best epoch 저장: {csv_path} ({frame['best_epoch'].tolist()})")
    return frame


def load_fold_best_epochs(save_path, config, require_all_folds=True):
    """기록해 둔 fold 별 best epoch 을 읽는다.

    기록의 학습 인자가 지금 인자와 다르면 읽지 않고 멈춘다.
    `require_all_folds` 면 fold 1..config.fold 가 한 번씩 다 들어 있어야 한다.
    fold 하나가 끝날 때마다 저장하므로 중간에 죽어 일부만 적힌 파일도 그대로 남고, 그 중앙값은 5-fold 를 다 돌린 종료 epoch 과 다르다.
    """
    csv_path = os.path.join(save_path, 'fold_best_epochs.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"fold best epoch 기록이 없다: {csv_path} — 5-fold 를 먼저 돌린다")

    frame = pd.read_csv(csv_path)
    check_training_args(frame, config, csv_path)

    if require_all_folds:
        expected = list(range(1, config.fold + 1))
        recorded = sorted(int(value) for value in frame['fold'].tolist())
        if recorded != expected:
            missing = sorted(set(expected) - set(recorded))
            raise ValueError(
                f"{csv_path}: fold 기록이 {expected} 와 다르다 (있는 fold {recorded}, 빠진 fold {missing}) — "
                f"5-fold 를 마저 돌린다"
            )

    return frame


def refit_end_epoch(best_epochs):
    """5-fold best epoch 들의 중앙값을 refit 의 종료 epoch 으로 쓴다.

    평균이 아닌 이유는 분포가 한쪽으로 늘어지기 때문이다 (옛 split 실측 46 / 28 / 77 / 47 / 45).
    fold 수가 짝수면 중앙값이 정수가 아니므로 내림한다.
    """
    if len(best_epochs) == 0:
        raise ValueError("best epoch 이 하나도 없어 refit 종료 epoch 을 정할 수 없다")

    return int(np.median(best_epochs))


def freeze_backbone(model):
    """Freeze backbone parameters for head-only training"""
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen for head warming-up")
    elif hasattr(model, 'module') and hasattr(model.module, 'backbone'):
        for param in model.module.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen for head warming-up (DataParallel)")


def unfreeze_backbone(model):
    """Unfreeze backbone parameters for full model training"""
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen for full model training")
    elif hasattr(model, 'module') and hasattr(model.module, 'backbone'):
        for param in model.module.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen for full model training (DataParallel)")


def setup_optimizer(model, config, phase='head_warmup'):
    """Setup optimizer for different training phases"""
    if phase == 'head_warmup':
        # Only optimize classification head parameters
        if hasattr(model, 'classifier'):
            params = model.classifier.parameters()
        elif hasattr(model, 'module') and hasattr(model.module, 'classifier'):
            params = model.module.classifier.parameters()
        else:
            raise ValueError("Cannot find classifier in model")
        
        lr = config.head_warmup_lr
        print(f"✓ Head warming-up optimizer with lr={lr}")
        
    elif phase == 'full_training':
        # Differential learning rates for backbone and classifier
        if hasattr(model, 'backbone') and hasattr(model, 'classifier'):
            backbone_params = model.backbone.parameters()
            classifier_params = model.classifier.parameters()
        elif hasattr(model, 'module'):
            backbone_params = model.module.backbone.parameters()
            classifier_params = model.module.classifier.parameters()
        else:
            raise ValueError("Cannot find backbone or classifier in model")
        
        params = [
            {'params': backbone_params, 'lr': config.lr * config.backbone_lr_ratio},
            {'params': classifier_params, 'lr': config.lr}
        ]
        lr = config.lr
        print(f"✓ Full training optimizer with backbone_lr={config.lr * config.backbone_lr_ratio}, classifier_lr={config.lr}")
    
    else:
        raise ValueError(f"Unknown training phase: {phase}")
    
    optimizer_class = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD
    }[config.optimizer]
    
    return optimizer_class(params, lr=lr)


def setup_scheduler(optimizer, config, phase='head_warmup', total_epochs=None):
    """Setup scheduler for different training phases"""
    if phase == 'head_warmup':
        # Head warmup phase에서는 고정된 학습률 사용 (스케줄러 없음)
        return None
    
    elif phase == 'full_training':
        remaining_epochs = total_epochs - config.head_warmup_epochs
        
        if config.scheduler == 'cosine':
            # 기존 warmup 설정을 사용한 cosine annealing with warmup
            lr_lambda = lambda epoch: (epoch * (1 - config.warmup_decay) / config.warmup_epochs + config.warmup_decay) \
                if epoch < config.warmup_epochs else \
                (1 - config.min_lr / config.lr) * 0.5 * (math.cos((epoch - config.warmup_epochs) / (remaining_epochs - config.warmup_epochs) * math.pi) + 1) + config.min_lr / config.lr
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        elif config.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=config.gamma)
    
    else:
        raise ValueError(f"Unknown training phase: {phase}")


def train_phase(config, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, 
                writer, run_label, phase='head_warmup', start_epoch=1, end_epoch=None):
    """Train model for a specific phase

    `val_loader` 가 None 이면 검증을 건너뛴다 — refit 은 development 전체로 학습해 남는 검증 fold 가 없다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    phase_name = "Head Warming-up" if phase == 'head_warmup' else "Full Training"
    print(f"\n=== {phase_name} Phase Started ===")
    
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_loss = 0
        cm = torch.zeros((config.num_classes, config.num_classes))

        with tqdm(total=len(train_loader), desc=f"{phase_name} Epoch {epoch}/{end_epoch}", unit='Batch') as pbar:
            for pack in train_loader:
                images = pack['imgs'].to(device)
                labels = pack['labels'].to(device)

                output = model(images=images)
                loss = criterion(output, labels)
                
                pred = output.argmax(dim=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                cm = confusion_matrix(pred.detach(), labels.detach(), cm)

                pbar.set_postfix(Loss=loss.item())
                pbar.update(1)

        # Head warmup phase에서는 스케줄러가 None일 수 있음
        if lr_scheduler is not None:
            lr_scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_acc = cm.diag().sum() / cm.sum()
        print('%s, %s Epoch [%d/%d] - Avg Train Loss: %.4f' % 
              (run_label, phase_name, epoch, end_epoch, avg_epoch_loss))

        # Log training metrics with phase prefix
        phase_prefix = 'HeadWarmup' if phase == 'head_warmup' else 'FullTraining'
        writer.add_scalar(f'{phase_prefix}/Train/Avg Epoch Loss', avg_epoch_loss, global_step=epoch)
        writer.add_scalar(f'{phase_prefix}/Train/Acc', train_acc, global_step=epoch)
        writer.add_scalar(f'{phase_prefix}/Train/LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)

        if val_loader is not None and (epoch % config.log_step == 0 or epoch == end_epoch):
            result = validate(config, model, val_loader, criterion)
            val_loss, val_acc, f1score, auc, spe, sen, pre = result
            
            # Log validation metrics with phase prefix
            writer.add_scalar(f'{phase_prefix}/Validation/Val Loss', val_loss, global_step=epoch)
            writer.add_scalar(f'{phase_prefix}/Validation/Acc', val_acc, global_step=epoch)
            writer.add_scalar(f'{phase_prefix}/Validation/F1 Score', f1score, global_step=epoch)
            writer.add_scalar(f'{phase_prefix}/Validation/AUC', auc, global_step=epoch)
            writer.add_scalar(f'{phase_prefix}/Validation/Spe', spe, global_step=epoch)
            writer.add_scalar(f'{phase_prefix}/Validation/Sen', sen, global_step=epoch)
            writer.add_scalar(f'{phase_prefix}/Validation/Pre', pre, global_step=epoch)

    print(f"=== {phase_name} Phase Completed ===\n")
    return model


def calculate_class_weights(data_loader, num_classes):
    """학습 데이터셋의 클래스별 분포를 기반으로 가중치를 자동 계산"""
    class_counts = torch.zeros(num_classes)
    
    # 전체 데이터에서 클래스별 개수 계산
    for batch in data_loader:
        labels = batch['labels']
        for label in labels:
            class_counts[label] += 1
    
    # 역가중치 계산 (전체 샘플 수 / (클래스 수 × 클래스별 샘플 수))
    total_samples = class_counts.sum()
    weights = total_samples / (num_classes * class_counts)
    
    # 가중치가 inf가 되는 경우 방지
    weights = torch.where(class_counts == 0, torch.tensor(1.0), weights)
    
    return weights


def train(config, train_loader, val_loader, fold):
    """fold 하나를 학습하고 `best_val_loss` 가 가리킨 epoch 과 그 loss 를 돌려준다.

    다섯 fold 가 돌려준 epoch 의 중앙값이 refit 의 종료 epoch 이 된다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    model = create_model(config)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # LOSS FUNCTION - 자동 가중치 계산
    weights = calculate_class_weights(train_loader, config.num_classes).to(device)
    print(f"Calculated weights for CrossEntropyLoss: {weights.cpu().numpy()}\n")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights).to(device) if config.loss_function == 'CE' else None

    # TensorBoard WRITER
    writer = SummaryWriter(log_dir=f'./DL_Classification/logs/{args.writer_comment}/{str(fold)}')

    ckpt_path = os.path.join(config.model_path)
    model_save_path = os.path.join(ckpt_path, args.writer_comment, str(fold))
    
    best_val_loss = float('inf')
    best_epoch = None

    # PHASE 1: Classification Head Warming-up
    if config.head_warmup_epochs > 0:
        freeze_backbone(model)
        
        head_optimizer = setup_optimizer(model, config, phase='head_warmup')
        head_scheduler = setup_scheduler(head_optimizer, config, phase='head_warmup')
        
        model = train_phase(
            config, model, train_loader, val_loader, criterion, 
            head_optimizer, head_scheduler, writer, f'Fold [{fold}/{config.fold}]',
            phase='head_warmup', 
            start_epoch=1, 
            end_epoch=config.head_warmup_epochs
        )
        
        unfreeze_backbone(model)

    # PHASE 2: Full Model Training
    full_optimizer = setup_optimizer(model, config, phase='full_training')
    full_scheduler = setup_scheduler(full_optimizer, config, phase='full_training', total_epochs=config.epochs)
    
    start_epoch = config.head_warmup_epochs + 1
    end_epoch = config.epochs
    
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_loss = 0
        cm = torch.zeros((config.num_classes, config.num_classes))

        with tqdm(total=len(train_loader), desc=f"Full Training Epoch {epoch}/{config.epochs}", unit='Batch') as pbar:
            for pack in train_loader:
                images = pack['imgs'].to(device)
                labels = pack['labels'].to(device)

                output = model(images=images)
                loss = criterion(output, labels)
                
                pred = output.argmax(dim=1)

                full_optimizer.zero_grad()
                loss.backward()
                full_optimizer.step()

                epoch_loss += loss.item()
                cm = confusion_matrix(pred.detach(), labels.detach(), cm)

                pbar.set_postfix(Loss=loss.item())
                pbar.update(1)

        full_scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_acc = cm.diag().sum() / cm.sum()
        print('Fold [%d/%d], Full Training Epoch [%d/%d] - Avg Train Loss: %.4f' % 
              (fold, config.fold, epoch, config.epochs, avg_epoch_loss))

        # Log training metrics
        writer.add_scalar('FullTraining/Train/Avg Epoch Loss', avg_epoch_loss, global_step=epoch)
        writer.add_scalar('FullTraining/Train/Acc', train_acc, global_step=epoch)
        writer.add_scalar('FullTraining/Train/LR_Backbone', full_optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        writer.add_scalar('FullTraining/Train/LR_Classifier', full_optimizer.state_dict()['param_groups'][1]['lr'], global_step=epoch)

        if epoch % config.log_step == 0 or epoch == config.epochs:
            result = validate(config, model, val_loader, criterion)

            # Log validation metrics
            val_loss, val_acc, f1score, auc, spe, sen, pre = result
            writer.add_scalar('FullTraining/Validation/Val Loss', val_loss, global_step=epoch)
            writer.add_scalar('FullTraining/Validation/Acc', val_acc, global_step=epoch)
            writer.add_scalar('FullTraining/Validation/F1 Score', f1score, global_step=epoch)
            writer.add_scalar('FullTraining/Validation/AUC', auc, global_step=epoch)
            writer.add_scalar('FullTraining/Validation/Spe', spe, global_step=epoch)
            writer.add_scalar('FullTraining/Validation/Sen', sen, global_step=epoch)
            writer.add_scalar('FullTraining/Validation/Pre', pre, global_step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                print("=> saved best model")

                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)

                if config.save_model:
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
                    else:
                        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))

                save_results(model_save_path, 'result_best.txt', epoch, val_loss, val_acc, f1score, auc, spe, sen, pre, 'w')

            if epoch == config.epochs:
                if config.save_model:
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.state_dict(), os.path.join(model_save_path, 'last_epoch_model.pth'))
                    else:
                        torch.save(model.state_dict(), os.path.join(model_save_path, 'last_epoch_model.pth'))

                save_results(model_save_path, 'result_last_epoch.txt', epoch, val_loss, val_acc, f1score, auc, spe, sen, pre, 'a')

            writer.flush()
    writer.close()

    if best_epoch is None:
        raise RuntimeError(f"Fold {fold}: 검증이 한 번도 안 돌아 best epoch 이 없다")

    return best_epoch, best_val_loss


def refit(config, train_loader, end_epoch):
    """development 전체로 다시 학습해 test 추론에 쓸 모델 하나를 만든다.

    검증 fold 가 없어 종료 epoch 을 밖에서 받고, 같은 값이 cosine 스케줄의 끝점으로도 들어가 lr 곡선이 잘리지 않는다.
    클래스 가중치는 넘겨받은 loader 에서 다시 세므로 fold 가 아니라 development 전체 분포를 따른다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    warmup_span = config.head_warmup_epochs + config.warmup_epochs
    if config.scheduler == 'cosine' and end_epoch <= warmup_span:
        raise ValueError(f"refit 종료 epoch {end_epoch} 이 warmup 구간 {warmup_span} 을 넘지 않아 cosine 스케줄이 성립하지 않는다")

    # MODEL
    model = create_model(config)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # LOSS FUNCTION - 자동 가중치 계산
    weights = calculate_class_weights(train_loader, config.num_classes).to(device)
    print(f"Calculated weights for CrossEntropyLoss: {weights.cpu().numpy()}\n")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights).to(device) if config.loss_function == 'CE' else None

    # TensorBoard WRITER
    writer = SummaryWriter(log_dir=f'./DL_Classification/logs/{config.writer_comment}/refit')

    model_save_path = os.path.join(config.model_path, config.writer_comment, 'refit')
    os.makedirs(model_save_path, exist_ok=True)

    # PHASE 1: Classification Head Warming-up
    if config.head_warmup_epochs > 0:
        freeze_backbone(model)

        head_optimizer = setup_optimizer(model, config, phase='head_warmup')
        head_scheduler = setup_scheduler(head_optimizer, config, phase='head_warmup')

        model = train_phase(
            config, model, train_loader, None, criterion,
            head_optimizer, head_scheduler, writer, 'Refit',
            phase='head_warmup',
            start_epoch=1,
            end_epoch=config.head_warmup_epochs
        )

        unfreeze_backbone(model)

    # PHASE 2: Full Model Training
    full_optimizer = setup_optimizer(model, config, phase='full_training')
    full_scheduler = setup_scheduler(full_optimizer, config, phase='full_training', total_epochs=end_epoch)

    model = train_phase(
        config, model, train_loader, None, criterion,
        full_optimizer, full_scheduler, writer, 'Refit',
        phase='full_training',
        start_epoch=config.head_warmup_epochs + 1,
        end_epoch=end_epoch
    )

    if config.save_model:
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, os.path.join(model_save_path, 'refit_model.pth'))
        print(f"=> saved refit model ({os.path.join(model_save_path, 'refit_model.pth')})")

    with open(os.path.join(model_save_path, 'refit_info.txt'), 'w') as f:
        f.write(f'End epoch: {end_epoch}\n')
        f.write(f'Train samples: {len(train_loader.dataset)}\n')
        f.write(f'Class weights: {weights.cpu().numpy()}\n')

    writer.flush()
    writer.close()


if __name__ == '__main__':
    seed_torch(42)
    args = load_config()

    # checkpoint 가 없으면 random init 으로 폴백해 성능이 폭락하고, 정규화 상수 파일이 없으면 다른 상수로 조용히 학습된다.
    if args.model_type == 'nnunet':
        required_files = ['plans_file_arch', 'dataset_json_file', 'checkpoint_file', 'plans_file_norm']
        absent = [f"{key}={Config.DL_NNUNET_CONFIG.get(key)}" for key in required_files
                  if not (Config.DL_NNUNET_CONFIG.get(key) and os.path.exists(Config.DL_NNUNET_CONFIG[key]))]
        if absent:
            raise FileNotFoundError(
                f"nnUNet 설정 파일이 없다: {', '.join(absent)} — "
                f"경로가 전부 상대경로라 저장소 루트에서 실행해야 한다"
            )

    cv = StratifiedKFold(n_splits=args.fold, random_state=42, shuffle=True)

    # AS Train 데이터셋 로드 (분할 설정 적용)
    print("AS Train 데이터셋 로딩...")
    train_set, label_to_idx, _, unique_labels = dl_cls_dataset.get_as_dataset(
        args.img_size, 
        mode='train',
        data_split_mode=args.data_split_mode,
        data_split_random_state=args.data_split_random_state,
        test_size_ratio=args.test_size_ratio
    )

    # 검증은 증강 없는 사본으로 본다. train_set 을 그대로 물리면 val loss 가 흔들려 best epoch 이 제비뽑기가 된다.
    val_set = dl_cls_dataset.make_eval_dataset(train_set, args.img_size)

    # 실제 클래스 수로 업데이트
    args.num_classes = len(unique_labels)
    print(f"AS Train 데이터셋 로드 완료. 클래스 수: {args.num_classes}")
    print(f"클래스 매핑: {label_to_idx}")
    print(f"데이터 분할 모드: {args.data_split_mode}")
    if args.data_split_mode == 'random':
        print(f"  - 테스트 데이터 비율: {args.test_size_ratio}")
        print(f"  - 랜덤 시드: {args.data_split_random_state}")
    print()

    print(vars(args))
    args_path = os.path.join(args.model_path, args.writer_comment)

    if not os.path.exists(args_path):
        os.makedirs(args_path)
    with open(os.path.join(args_path, f'model_info_{args.stage}.txt'), 'w') as f:
        f.write(str(vars(args)))

    print("START TRAINING")
    fold_best = []

    if args.stage in ('all', 'folds'):
        recorded_folds = {}
        if args.resume and os.path.exists(os.path.join(args_path, 'fold_best_epochs.csv')):
            # 중간에 죽은 뒤 이어 돌리는 자리라 일부 fold 만 적힌 기록도 받는다.
            recorded = load_fold_best_epochs(args_path, args, require_all_folds=False)
            recorded_folds = {int(row['fold']): row for _, row in recorded.iterrows()}
            assignment_path = os.path.join(args_path, 'cls_fold_assignment.csv')
            if not os.path.exists(assignment_path):
                raise FileNotFoundError(
                    f"fold 배정 기록이 없다: {assignment_path} — "
                    f"재사용할 가중치가 어느 분할로 학습된 것인지 확인할 수 없으니 --resume 없이 다시 돌린다"
                )

        train_labels = [train_set[i]['labels'] for i in range(len(train_set))]
        splits = list(cv.split(train_set, train_labels))
        # 옛 가중치가 남은 채 배정만 새로 쓰이면 그 fold 모델이 자기가 학습에 쓴 케이스를 검증으로 배정받는다.
        kept_weights = any(os.path.exists(fold_weight_path(args, fold)) for fold in range(1, args.fold + 1))
        save_fold_assignment(train_set, splits, args_path, require_match=args.resume or kept_weights)

        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            # fold 마다 같은 자리에서 시작한다. 전역 RNG 를 이어받으면 앞 fold 가 난수를 몇 번 뽑았는지에 초기 가중치가 걸린다.
            seed_torch(42 + fold)
            print(f"\nCross Validation Fold {fold}")

            record = recorded_folds.get(fold)
            weight_file = fold_weight_path(args, fold)
            if record is not None and os.path.exists(weight_file):
                print(f"fold {fold} 은 기록과 가중치가 다 있어 건너뛴다 (best epoch {int(record['best_epoch'])}, val loss {float(record['best_val_loss']):.4f})")
                fold_best.append({'fold': fold, 'best_epoch': int(record['best_epoch']),
                                  'best_val_loss': float(record['best_val_loss'])})
                continue

            train_sampler = SubsetRandomSampler(train_idx)
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                                      num_workers=6, worker_init_fn=worker_init_fn)
            # 검증은 순서를 고정한다. sampler 를 물리면 epoch 마다 배치 구성이 바뀐다.
            val_loader = DataLoader(Subset(val_set, val_idx), batch_size=args.batch_size, shuffle=False)

            best_epoch, best_val_loss = train(args, train_loader, val_loader, fold)
            fold_best.append({'fold': fold, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss})
            save_fold_best_epochs(merge_recorded_folds(fold_best, recorded_folds, args), args_path, args)

        save_fold_best_epochs(merge_recorded_folds(fold_best, recorded_folds, args), args_path, args)

    if args.stage in ('all', 'refit'):
        best_epochs = ([row['best_epoch'] for row in fold_best] if fold_best
                       else load_fold_best_epochs(args_path, args)['best_epoch'].tolist())
        end_epoch = refit_end_epoch(best_epochs)
        print(f"\nSTART REFIT - development {len(train_set)}, 종료 epoch {end_epoch} (fold best epochs {best_epochs})")

        # fold 가 쓰고 남긴 전역 RNG 를 이어받지 않는다. `--stage all` 의 refit 과 `--stage refit` 단독이 같은 초기 가중치에서 시작해야 한다.
        seed_torch(42)

        # 증강은 fold 학습과 같게 두고 sampler 만 뗀다. 322 전체가 매 epoch 한 번씩 들어간다.
        refit_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6,
                                  worker_init_fn=worker_init_fn)
        refit(args, refit_loader, end_epoch)