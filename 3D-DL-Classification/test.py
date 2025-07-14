import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from model import CustomModel
from dl_cls_config import load_config


def plot_confusion_matrix(conf_matrix, class_names, accuracy, f1, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Normalized Confusion Matrix\nAccuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}')
    
    plt.savefig(output_path)
    plt.close()


def load_model(model_path, config):
    model = CustomModel(class_num=config.class_num)
    
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model


def evaluate_single_fold(labels, probs, class_names, config, fold_number):
    """
    Evaluate single fold results.
    """
    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    
    class_report = classification_report(labels, preds, target_names=class_names)
    conf_matrix = confusion_matrix(labels, preds, normalize='true')

    print(f'Fold {fold_number} Results')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(class_report)

    # Confusion matrix 저장 경로
    output_path = os.path.join('./3D-DL-Classification', './result', config.writer_comment)
    conf_matrix_plot_path = os.path.join(output_path, 'conf_matrix', f'fold_{fold_number}.png')

    # Confusion matrix 그리기
    plot_confusion_matrix(conf_matrix, class_names, accuracy, f1, conf_matrix_plot_path)


def evaluate_ensemble(all_labels, all_probs, class_names, config):
    """
    Evaluate ensemble results using both soft voting and hard voting.
    """
    # 각 Fold의 예측 확률을 합산합니다.
    # all_probs의 형태: (fold 수, 샘플 수, 클래스 수)
    
    # Soft Voting: 모든 fold의 평균 확률로 최종 클래스 결정
    ensemble_probs = np.mean(all_probs, axis=0)  # (샘플 수, 클래스 수)
    ensemble_preds_soft = np.argmax(ensemble_probs, axis=1)

    # Hard Voting: 각 모델의 클래스 예측을 다수결로 결정
    all_preds = np.array([np.argmax(prob, axis=1) for prob in all_probs])  # (fold 수, 샘플 수)
    ensemble_preds_hard = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, all_preds)

    # 평가 및 결과 출력
    for voting_type, ensemble_preds in [('Soft Voting', ensemble_preds_soft), ('Hard Voting', ensemble_preds_hard)]:
        accuracy = accuracy_score(all_labels, ensemble_preds)
        f1 = f1_score(all_labels, ensemble_preds, average='macro')
        class_report = classification_report(all_labels, ensemble_preds, target_names=class_names)
        conf_matrix = confusion_matrix(all_labels, ensemble_preds, normalize='true')

        print(f'{voting_type} Results')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(class_report)

        # Confusion matrix 저장 경로
        output_path = os.path.join('./3D-DL-Classification', './result', config.writer_comment)
        conf_matrix_plot_path = os.path.join(output_path, 'conf_matrix', f'ensemble_{voting_type.lower().replace(" ", "_")}.png')

        # Confusion matrix 그리기
        plot_confusion_matrix(conf_matrix, class_names, accuracy, f1, conf_matrix_plot_path)


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AS 데이터셋을 한 번만 로드
    print("데이터셋 로딩 중...")
    test_dataset, label_to_idx, idx_to_label, unique_labels = dataset.get_as_dataset(config.img_size, mode='test')
    # 실제 클래스 수로 업데이트
    config.class_num = len(unique_labels)
    print(f"AS 데이터셋 로드 완료. 클래스 수: {config.class_num}")
    print(f"클래스 매핑: {label_to_idx}")
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=6)

    # 앙상블 예측을 위한 변수를 초기화합니다.
    all_labels = []  # 모든 fold에서 동일한 레이블을 저장합니다.
    all_probs = []  # 각 fold의 예측 확률을 저장할 리스트

    for fold in range(1, config.fold + 1):
        print(f"\n=== Fold {fold} 평가 시작 ===")
        model_path = os.path.join(config.model_path, config.writer_comment, str(fold), 'best_model.pth')
        
        if not os.path.exists(model_path):
            print(f"Model for fold {fold} not found at {model_path}")
            continue
        
        model = load_model(model_path, config)
        model = model.to(device)
        model.eval()

        fold_probs = []  # 각 fold의 예측 확률을 저장할 리스트
        fold_labels = []  # 각 fold의 레이블을 저장할 리스트

        with torch.no_grad():
            for pack in tqdm(test_loader, desc=f'Testing Fold {fold}', unit='batch'):
                images = pack['imgs'].to(device)
                labels = pack['labels'].to(device)

                output = model(images=images)
                probs = torch.softmax(output, dim=1)  # 각 클래스에 대한 확률 계산
                fold_probs.append(probs.cpu().numpy())

                # 각 배치에서 레이블을 추가합니다.
                fold_labels.extend(labels.cpu().numpy())

        # 현재 fold의 레이블 및 예측 확률을 저장합니다.
        fold_probs = np.concatenate(fold_probs, axis=0)
        all_probs.append(fold_probs)

        # 첫 번째 fold에서 레이블을 저장합니다.
        if fold == 1:
            all_labels = fold_labels

        # 각 fold의 성능 평가 (클래스 이름 사용)
        evaluate_single_fold(fold_labels, fold_probs, unique_labels, config, fold)

    # 앙상블 평가 (클래스 이름 사용)
    print(f"\n=== 앙상블 평가 시작 ===")
    evaluate_ensemble(all_labels, all_probs, unique_labels, config)


if __name__ == '__main__':
    main()