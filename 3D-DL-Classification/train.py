import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import dataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from dl_cls_config import load_config
from model import CustomModel
from valid import validate


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


def train(config, train_loader, val_loader, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    model = CustomModel(class_num=config.class_num)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device) if config.loss_function == 'CE' else None

    # OPTIMIZER
    optimizer = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'SGD': torch.optim.SGD
    }[config.optimizer](model.parameters(), lr=config.lr)

    # SCHEDULER
    if config.scheduler == 'cosine':
        lr_lambda = lambda epoch: (epoch * (1 - config.warmup_decay) / config.warmup_epochs + config.warmup_decay) \
            if epoch < config.warmup_epochs else \
            (1 - config.min_lr / config.lr) * 0.5 * (math.cos((epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs) * math.pi) + 1) + config.min_lr / config.lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif config.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=config.gamma)

    # TensorBoard WRITER
    writer = SummaryWriter(log_dir=f'./3D-DL-Classification/logs/{args.writer_comment}/{str(fold)}')

    ckpt_path = os.path.join(config.model_path)
    model_save_path = os.path.join(ckpt_path, args.writer_comment, str(fold))
    
    best_val_loss = float('inf')

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0
        cm = torch.zeros((config.class_num, config.class_num))

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{config.epochs}", unit='Batch') as pbar:
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

        lr_scheduler.step()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_acc = cm.diag().sum() / cm.sum()
        print('Fold [%d/%d], Epoch [%d/%d] - Avg Train Loss: %.4f' % (fold, config.fold, epoch, config.epochs, avg_epoch_loss))

        # Log training metrics
        writer.add_scalar('Train/Avg Epoch Loss', avg_epoch_loss, global_step=epoch)
        writer.add_scalar('Train/Acc', train_acc, global_step=epoch)
        writer.add_scalar('Train/LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)

        if epoch % config.log_step == 0 or epoch == config.epochs:
            result = validate(config, model, val_loader, criterion)

            # Log validation metrics
            val_loss, val_acc, f1score, auc, spe, sen, pre = result
            writer.add_scalar('Validation/Val Loss', val_loss, global_step=epoch)
            writer.add_scalar('Validation/Acc', val_acc, global_step=epoch)
            writer.add_scalar('Validation/F1 Score', f1score, global_step=epoch)
            writer.add_scalar('Validation/AUC', auc, global_step=epoch)
            writer.add_scalar('Validation/Spe', spe, global_step=epoch)
            writer.add_scalar('Validation/Sen', sen, global_step=epoch)
            writer.add_scalar('Validation/Pre', pre, global_step=epoch)

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
    
    # AS 데이터셋 사용
    train_set, label_to_idx, idx_to_label, unique_labels = dataset.get_as_dataset(args.img_size, mode='train')
    # 실제 클래스 수로 업데이트
    args.class_num = len(unique_labels)
    print(f"AS 데이터셋 로드 완료. 클래스 수: {args.class_num}")
    print(f"클래스 매핑: {label_to_idx}")

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