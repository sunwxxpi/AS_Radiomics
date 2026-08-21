import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dl_cls_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from dl_cls_model import create_model
from dl_cls_config import load_config
from dl_cls_cam import generate_cam_for_sample


def plot_confusion_matrix(conf_matrix_raw, class_names, accuracy, f1, auc_score, ap_score, output_path):
    """혼동행렬과 성능지표를 시각화하여 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 정규화된 혼동행렬 계산 및 텍스트 어노테이션 생성
    conf_matrix_normalized = conf_matrix_raw.astype('float') / conf_matrix_raw.sum(axis=1)[:, np.newaxis]
    
    annot_text = np.empty_like(conf_matrix_raw, dtype=object)
    for i in range(conf_matrix_raw.shape[0]):
        for j in range(conf_matrix_raw.shape[1]):
            annot_text[i, j] = f'{conf_matrix_normalized[i, j]:.3f}\n({conf_matrix_raw[i, j]})'
    
    # 성능지표 제목 생성
    metrics_text = f"Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f} | AUC: {auc_score:.4f} | AP: {ap_score:.4f}"
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix_normalized, annot=annot_text, fmt='',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', annot_kws={"size": 20}, vmin=0.0, vmax=1.0,
                cbar=False, square=True)
    plt.title(f'Confusion Matrix\n\n{metrics_text}\n(Values: Normalized Ratio (Raw Count))', 
              fontsize=20, pad=20)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200)
    plt.close()


def calculate_metrics(labels, probs, class_names):
    """예측 확률로부터 모든 성능지표 계산"""
    preds = np.argmax(probs, axis=1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    
    # 이진/다중클래스 AUC 계산
    try:
        if len(class_names) == 2:
            auc = roc_auc_score(labels, probs[:, 1])
            ap = average_precision_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            ap = average_precision_score(labels, probs, average='macro')
    except ValueError:
        auc = ap = 0.0
    
    return accuracy, f1, auc, ap


def evaluate_single_run(labels, probs, class_names, config, run_label):
    """가중치 하나의 성능 평가 및 혼동행렬 저장"""
    accuracy, f1, auc, ap = calculate_metrics(labels, probs, class_names)
    preds = np.argmax(probs, axis=1)
    
    class_report = classification_report(labels, preds, target_names=class_names)
    conf_matrix = confusion_matrix(labels, preds)

    print(f'\n=== {run_label} Results ===')
    print(f'Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}')
    print(f'\nClassification Report:')
    print(class_report)

    # 혼동행렬 시각화 저장
    output_path = os.path.join('./DL_Classification', 'results', config.writer_comment)
    conf_matrix_plot_path = os.path.join(output_path, 'conf_matrix', f'{run_label}.png')
    plot_confusion_matrix(conf_matrix, class_names, accuracy, f1, auc, ap, conf_matrix_plot_path)


def evaluate_ensemble(all_labels, all_probs, class_names, config):
    """Soft/Hard voting을 이용한 앙상블 성능 평가"""
    ensemble_probs = np.mean(all_probs, axis=0)  # Soft voting
    ensemble_preds_soft = np.argmax(ensemble_probs, axis=1)

    # Hard voting: 각 fold 예측의 다수결
    all_preds = np.array([np.argmax(prob, axis=1) for prob in all_probs])
    ensemble_preds_hard = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, all_preds)

    for voting_type, ensemble_preds in [('Soft Voting', ensemble_preds_soft), ('Hard Voting', ensemble_preds_hard)]:
        accuracy, f1, auc, ap = calculate_metrics(all_labels, ensemble_probs, class_names)
        
        class_report = classification_report(all_labels, ensemble_preds, target_names=class_names)
        conf_matrix = confusion_matrix(all_labels, ensemble_preds)

        print(f'\n=== {voting_type} Ensemble Results ===')
        print(f'Accuracy: {accuracy:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | AP: {ap:.4f}')
        print(f'\nClassification Report:')
        print(class_report)

        # 앙상블 혼동행렬 저장
        output_path = os.path.join('./DL_Classification', 'results', config.writer_comment)
        plot_path = os.path.join(output_path, 'conf_matrix', f'ensemble_{voting_type.lower().replace(" ", "_")}.png')
        plot_confusion_matrix(conf_matrix, class_names, accuracy, f1, auc, ap, plot_path)


def generate_cam_for_samples(model, cam_data_list, class_names, cam_save_dir):
    """수집된 샘플들에 대한 CAM 시각화 생성"""
    print(f"\n--- {len(cam_data_list)}개 샘플 CAM 생성 시작 ---")
    device = next(model.parameters()).device
    successful_count = 0
    
    for i, data in enumerate(cam_data_list):
        image_tensor = data['image_tensor'].to(device)
        true_label = data['true_label']
        sample_name = data['sample_name']
        
        print(f"샘플 {i + 1}/{len(cam_data_list)}: {sample_name} (실제: {class_names[true_label]})")
        
        try:
            cam_result = generate_cam_for_sample(
                model=model,
                image_tensor=image_tensor,
                target_class=None,
                class_names=class_names,
                save_dir=cam_save_dir,
                sample_name=sample_name.replace('_0000.nii.gz', ''),
                true_label=true_label
            )
            
            if cam_result and cam_result[0] is not None:
                _, predicted_class = cam_result
                print(f"-> 예측: {class_names[predicted_class]} | CAM 생성 성공")
                successful_count += 1
            else:
                print(f"-> CAM 생성 실패")
        
        except Exception as e:
            print(f"-> CAM 생성 오류: {e}")
            
    print(f"--- CAM 생성 완료: {successful_count}/{len(cam_data_list)}개 성공 ---")


def save_probabilities_csv(case_ids, labels, probs, class_names, config, run_label):
    """확률값을 CSV 파일로 저장 (앙상블용)

    Args:
        case_ids: 케이스 ID 리스트
        labels: 실제 레이블 리스트
        probs: 예측 확률 배열 (N x num_classes)
        class_names: 클래스 이름 리스트
        config: 설정 객체
        run_label: 파일 이름에 붙일 갈래 이름 ('fold_1' 또는 'refit')
    """
    # probs 디렉토리 생성
    probs_dir = os.path.join('./DL_Classification', 'results', config.writer_comment, 'probs')
    os.makedirs(probs_dir, exist_ok=True)

    # DataFrame 생성
    data = {
        'case_id': case_ids,
        'true_label': labels,
        'true_label_str': [class_names[label] for label in labels]
    }

    # 각 클래스별 확률 추가
    for i, class_name in enumerate(class_names):
        data[f'proba_{class_name}'] = probs[:, i]

    df = pd.DataFrame(data)

    # case_id 기준으로 정렬
    df = df.sort_values(by='case_id').reset_index(drop=True)

    output_path = os.path.join(probs_dir, f'{run_label}.csv')
    df.to_csv(output_path, index=False)


def run_inference(model, test_loader, device, run_label, max_cam_samples=10):
    """모델 추론을 실행하고, 결과 및 CAM 생성용 데이터를 반환"""
    all_probs = []
    all_labels = []
    all_case_ids = []
    cam_data_list = []

    model.eval()
    try:
        with torch.no_grad():
            for pack in tqdm(test_loader, desc=f'Testing {run_label}', unit='batch'):
                images = pack['imgs'].to(device)
                labels = pack['labels'].to(device)
                names = pack['names']

                output = model(images=images)
                probs = torch.softmax(output, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # case_id 수집 (파일명에서 _0000.nii.gz 제거)
                for name in names:
                    case_id = name.replace('_0000.nii.gz', '')
                    all_case_ids.append(case_id)

                # CAM 생성용 샘플 데이터 수집
                if len(cam_data_list) < max_cam_samples:
                    for i in range(images.shape[0]):
                        if len(cam_data_list) >= max_cam_samples:
                            break
                        cam_data_list.append({
                            'image_tensor': images[i].clone().cpu(),
                            'true_label': labels[i].item(),
                            'sample_name': names[i]
                        })
    finally:
        torch.cuda.empty_cache()

    return all_labels, np.concatenate(all_probs, axis=0), all_case_ids, cam_data_list


def evaluate_weights(config, model_path, test_loader, device, run_label, class_names, should_save_cam):
    """가중치 하나로 test 를 추론해 혼동행렬·확률 CSV·CAM 을 남기고 (labels, probs) 를 돌려준다.

    가중치 파일이 없으면 None 을 돌려준다.
    """
    if not os.path.exists(model_path):
        print(f"{run_label} 모델 파일 없음. 건너뜀: {model_path}")
        return None

    model = create_model(config)
    model.load_state_dict(torch.load(model_path))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    labels, probs, case_ids, cam_samples = run_inference(
        model, test_loader, device, run_label, max_cam_samples=20
    )

    evaluate_single_run(labels, probs, class_names, config, run_label)
    save_probabilities_csv(case_ids, labels, probs, class_names, config, run_label)
    print(f"{run_label} 확률값 저장 완료.")

    if should_save_cam and cam_samples:
        cam_save_dir = os.path.join('./DL_Classification', 'results', config.writer_comment, 'cam_visualization', run_label)
        os.makedirs(cam_save_dir, exist_ok=True)
        generate_cam_for_samples(model, cam_samples, class_names, cam_save_dir)

    return labels, probs


def main():
    """메인 평가 파이프라인"""
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # AS Test 데이터셋 로드 (분할 설정 적용)
    print("AS Test 데이터셋 로딩...")
    test_dataset, label_to_idx, _, class_names = dl_cls_dataset.get_as_dataset(
        config.img_size, 
        mode='test',
        data_split_mode=config.data_split_mode,
        data_split_random_state=config.data_split_random_state,
        test_size_ratio=config.test_size_ratio
    )
    
    # 실제 클래스 수로 업데이트
    config.num_classes = len(class_names)
    print(f"AS Test 데이터셋 로드 완료. 클래스 수: {config.num_classes}")
    print(f"클래스 매핑: {label_to_idx}")
    print(f"데이터 분할 모드: {config.data_split_mode}")
    if config.data_split_mode == 'random':
        print(f"  - 테스트 데이터 비율: {config.test_size_ratio}")
        print(f"  - 랜덤 시드: {config.data_split_random_state}")
    print()

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6)

    # 앙상블 평가용 변수
    all_fold_labels = []
    all_fold_probs = []

    # CAM 설정
    should_save_cam = config.enable_cam
    print(f"CAM 생성: {'활성화' if should_save_cam else '비활성화'}")

    # 각 fold 평가
    if config.stage in ('all', 'folds'):
        for fold in range(1, config.fold + 1):
            print(f"\n{'='*20} Fold {fold} 평가 시작 {'='*20}")
            model_path = os.path.join(config.model_path, config.writer_comment, str(fold), 'best_model.pth')

            result = evaluate_weights(config, model_path, test_loader, device, f'fold_{fold}', class_names, should_save_cam)
            if result is None:
                continue

            fold_labels, fold_probs = result
            if not all_fold_labels:
                all_fold_labels = fold_labels
            all_fold_probs.append(fold_probs)

        # 앙상블 평가
        if all_fold_probs:
            print(f"\n{'='*20} 앙상블 평가 시작 {'='*20}")
            evaluate_ensemble(all_fold_labels, np.array(all_fold_probs), class_names, config)

    # refit 평가. 논문 2.4 가 test 추론을 refit 모델 하나로 못박으므로 DL-only 성적과 ensemble 확률은 여기서 나온다.
    if config.stage in ('all', 'refit'):
        print(f"\n{'='*20} refit 평가 시작 {'='*20}")
        model_path = os.path.join(config.model_path, config.writer_comment, 'refit', 'refit_model.pth')
        evaluate_weights(config, model_path, test_loader, device, 'refit', class_names, should_save_cam)


if __name__ == '__main__':
    main()