import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import dl_cls_dataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from dl_cls_config import load_config
from dl_cls_model import create_model
from dl_cls_valid import validate


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
                writer, fold, phase='head_warmup', start_epoch=1, end_epoch=None):
    """Train model for a specific phase"""
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
        print('Fold [%d/%d], %s Epoch [%d/%d] - Avg Train Loss: %.4f' % 
              (fold, config.fold, phase_name, epoch, end_epoch, avg_epoch_loss))

        # Log training metrics with phase prefix
        phase_prefix = 'HeadWarmup' if phase == 'head_warmup' else 'FullTraining'
        writer.add_scalar(f'{phase_prefix}/Train/Avg Epoch Loss', avg_epoch_loss, global_step=epoch)
        writer.add_scalar(f'{phase_prefix}/Train/Acc', train_acc, global_step=epoch)
        writer.add_scalar(f'{phase_prefix}/Train/LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)

        if epoch % config.log_step == 0 or epoch == end_epoch:
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


def train(config, train_loader, val_loader, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    model = create_model(config)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # LOSS FUNCTION
    weight = torch.tensor([1.0/0.209, 1.0/0.096, 1.0/0.696], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weight).to(device) if config.loss_function == 'CE' else None

    # TensorBoard WRITER
    writer = SummaryWriter(log_dir=f'./DL_Classification/logs/{args.writer_comment}/{str(fold)}')

    ckpt_path = os.path.join(config.model_path)
    model_save_path = os.path.join(ckpt_path, args.writer_comment, str(fold))
    
    best_val_loss = float('inf')

    # PHASE 1: Classification Head Warming-up
    if config.head_warmup_epochs > 0:
        freeze_backbone(model)
        
        head_optimizer = setup_optimizer(model, config, phase='head_warmup')
        head_scheduler = setup_scheduler(head_optimizer, config, phase='head_warmup')
        
        model = train_phase(
            config, model, train_loader, val_loader, criterion, 
            head_optimizer, head_scheduler, writer, fold,
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


if __name__ == '__main__':
    seed_torch(42)
    args = load_config()

    cv = StratifiedKFold(n_splits=args.fold, random_state=42, shuffle=True)
    
    # AS 데이터셋 사용 (분할 설정 적용)
    train_set, label_to_idx, idx_to_label, unique_labels = dl_cls_dataset.get_as_dataset(
        args.img_size, 
        mode='train',
        data_split_mode=args.data_split_mode,
        data_split_random_state=args.data_split_random_state,
        test_size_ratio=args.test_size_ratio
    )
    # 실제 클래스 수로 업데이트
    args.num_classes = len(unique_labels)
    print(f"AS 데이터셋 로드 완료. 클래스 수: {args.num_classes}")
    print(f"클래스 매핑: {label_to_idx}")
    print(f"데이터 분할 모드: {args.data_split_mode}")
    if args.data_split_mode == 'random':
        print(f"  - 테스트 데이터 비율: {args.test_size_ratio}")
        print(f"  - 랜덤 시드: {args.data_split_random_state}")

    print(vars(args))
    args_path = os.path.join(args.model_path, args.writer_comment)

    if not os.path.exists(args_path):
        os.makedirs(args_path)
    with open(os.path.join(args_path, 'model_info.txt'), 'w') as f:
        f.write(str(vars(args)))

    print("START TRAINING")
    fold = 1
    train_labels = [train_set[i]['labels'] for i in range(len(train_set))]
    
    for train_idx, val_idx in cv.split(train_set, train_labels):
        print(f"\nCross Validation Fold {fold}")

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers=6)
        val_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)

        train(args, train_loader, val_loader, fold)

        fold += 1