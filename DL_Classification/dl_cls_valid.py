import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)

    n_class = conf_matrix.shape[0]
    idx = labels * n_class + preds

    counts = torch.bincount(idx, minlength=n_class**2)
    conf_matrix += counts.reshape(n_class, n_class)

    return conf_matrix


def calculate_metrics(cm, num_classes):
    specificity = []
    sensitivity = []
    precision = []

    for i in range(num_classes):
        true_negative = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        false_positive = cm[:, i].sum() - cm[i, i]
        false_negative = cm[i, :].sum() - cm[i, i]
        true_positive = cm[i, i]

        specificity.append(true_negative / (true_negative + false_positive + 1e-6))
        sensitivity.append(true_positive / (true_positive + false_negative + 1e-6))
        precision.append(true_positive / (true_positive + false_positive + 1e-6))

    avg_specificity = sum(specificity) / num_classes
    avg_sensitivity = sum(sensitivity) / num_classes
    avg_precision = sum(precision) / num_classes
    avg_recall = avg_sensitivity
    f1score = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-6)

    return f1score, avg_specificity, avg_sensitivity, avg_precision


def validate(config, model, val_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()

    print("START VALIDATION")

    epoch_loss = 0
    y_true, y_score = [], []

    cm = torch.zeros((config.num_classes, config.num_classes))

    with tqdm(total=len(val_loader), desc="Validation", unit='Batch') as pbar:
        with torch.no_grad():
            for pack in val_loader:
                images = pack['imgs'].to(device)
                labels = pack['labels'].to(device)

                output = model(images=images)

                loss = criterion(output, labels)
                epoch_loss += loss.item() * images.size(0)  # accumulate loss over batch

                pred = output.argmax(dim=1)
                y_true.extend(labels.cpu().numpy())
                y_score.extend(output.softmax(dim=1).cpu().numpy())  # Apply softmax to get probabilities

                cm = confusion_matrix(pred.cpu(), labels.cpu(), cm)

                # Update progress bar
                pbar.update(1)

    avg_epoch_loss = epoch_loss / len(val_loader.dataset)
    print(f'Validation - Avg Loss: {avg_epoch_loss:.4f}')

    # Calculate metrics
    acc = cm.diag().sum() / cm.sum()
    f1score, avg_specificity, avg_sensitivity, avg_precision = calculate_metrics(cm, config.num_classes)

    # Compute AUC for each class and average
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if config.num_classes > 2:
        # For multiclass, y_true should be in shape (n_samples,) and y_score in shape (n_samples, n_classes)
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    else:
        # For binary, y_true should be in shape (n_samples,) and y_score in shape (n_samples,)
        y_true = y_true.flatten()
        y_score = y_score[:, 1]  # Use the probability of the positive class
        auc = roc_auc_score(y_true, y_score)

    return [avg_epoch_loss, acc, f1score, auc, avg_specificity, avg_sensitivity, avg_precision]