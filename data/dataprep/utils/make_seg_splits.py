"""segmentation 5-fold cross-fitting 용 splits_final.json 을 severity stratified 로 생성한다.

첫 nnUNetv2_train 실행 전에 써 넣어야 한다. 파일이 없으면 nnU-Net 이 stratified 아닌 KFold 로
자동 생성해 고정하고 (nnUNetTrainer.do_split), 이후 fold 들이 그 파일을 재사용하므로
나중에 바꾸려면 이미 돌린 학습을 버려야 한다.
nonsevere 가 250건 중 24건뿐이라 stratify 없이 자르면 fold 별 nonsevere 수가 크게 흔들린다.
"""

import argparse
import json
import os
import sys

from sklearn.model_selection import StratifiedKFold

# organize_total_stratified_dataset 는 data/dataprep/ 에 있고 이 파일은 그 아래 utils/ 다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from organize_total_stratified_dataset import extract_patient_id, load_severity_mapping

PREPROCESSED_DIR = '/home/psw/nnUNet/data/nnUNet_preprocessed/Dataset003_KMU_Cardiac_AVC_TRAIN_ONLY'
RESULTS_DIR = '/home/psw/nnUNet/data/nnUNet_results/Dataset003_KMU_Cardiac_AVC_TRAIN_ONLY'
SEVERITY_CSV = '/home/psw/AS_Radiomics/data/AS_CRF.csv'
CLASS_ORDER = ['normal', 'nonsevere', 'severe']


class VerificationError(Exception):
    """검증 실패. 잘못된 split 이 고정되지 않도록 즉시 중단시킨다."""


def load_identifiers(preprocessed_dir):
    """전처리된 케이스 식별자 목록. do_split 이 조회하는 키와 같은 문자열이어야 한다."""
    dataset_json = os.path.join(preprocessed_dir, 'dataset.json')
    if not os.path.isfile(dataset_json):
        raise VerificationError(f"dataset.json 없음: {dataset_json} — 전처리 먼저 돌려야 한다")
    with open(dataset_json) as f:
        file_ending = json.load(f)['file_ending']

    gt_dir = os.path.join(preprocessed_dir, 'gt_segmentations')
    if not os.path.isdir(gt_dir):
        raise VerificationError(f"gt_segmentations 없음: {gt_dir}")
    identifiers = sorted(f[:-len(file_ending)] for f in os.listdir(gt_dir)
                         if f.endswith(file_ending))
    if not identifiers:
        raise VerificationError(f"{gt_dir} 에 케이스가 없다")

    verify_config_folders(preprocessed_dir, identifiers)
    return identifiers


def verify_config_folders(preprocessed_dir, identifiers):
    """전처리 산출물 폴더의 식별자가 gt_segmentations 와 같은지 확인한다."""
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    checked = []
    for name in sorted(os.listdir(preprocessed_dir)):
        folder = os.path.join(preprocessed_dir, name)
        if name == 'gt_segmentations' or not os.path.isdir(folder):
            continue
        found = set(infer_dataset_class(folder).get_identifiers(folder))
        if found != set(identifiers):
            missing = sorted(set(identifiers) - found)[:5]
            extra = sorted(found - set(identifiers))[:5]
            raise VerificationError(
                f"{name} 의 식별자가 gt_segmentations 와 다르다 "
                f"(누락 {len(set(identifiers) - found)}건 {missing}, 초과 {len(found - set(identifiers))}건 {extra})")
        checked.append(f"{name}({len(found)})")
    print(f"  전처리 폴더 일치 확인: {', '.join(checked) if checked else '없음 (gt_segmentations 만 사용)'}")


def label_cases(identifiers, severity_map):
    """식별자별 severity 라벨. patient_id 는 식별자 앞부분에서 뽑는다."""
    labels, missing = [], []
    for case_id in identifiers:
        patient_id = extract_patient_id(case_id)
        if patient_id not in severity_map:
            missing.append(case_id)
            continue
        labels.append(severity_map[patient_id])
    if missing:
        raise VerificationError(f"severity 없는 케이스 {len(missing)}건: {missing[:10]}")
    return labels


def make_splits(identifiers, labels, n_splits, seed):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in kfold.split(identifiers, labels):
        splits.append({'train': [identifiers[i] for i in train_idx],
                       'val': [identifiers[i] for i in val_idx]})
    return splits


def verify_splits(splits, identifiers, labels, n_splits):
    if len(splits) != n_splits:
        raise VerificationError(f"fold 수 불일치: {len(splits)} != {n_splits}")

    label_of = dict(zip(identifiers, labels))
    all_cases = set(identifiers)
    seen_val = []
    for fold, split in enumerate(splits):
        train, val = set(split['train']), set(split['val'])
        if train & val:
            raise VerificationError(f"fold {fold}: train/val 교집합 {len(train & val)}건")
        if train | val != all_cases:
            raise VerificationError(f"fold {fold}: train+val 이 전체 {len(all_cases)}건과 다르다")
        for cls in CLASS_ORDER:
            if not any(label_of[c] == cls for c in val):
                raise VerificationError(f"fold {fold} val 에 {cls} 가 없다")
        seen_val.extend(split['val'])

    if len(seen_val) != len(all_cases) or set(seen_val) != all_cases:
        raise VerificationError(
            f"val 합집합이 전체와 다르다 ({len(seen_val)}건, 중복 {len(seen_val) - len(set(seen_val))}건) "
            "— 케이스마다 자기가 held-out 인 fold 가 정확히 하나여야 3번 추론이 성립한다")


def report(splits, identifiers, labels):
    label_of = dict(zip(identifiers, labels))
    header = ' '.join(f"{cls:>10}" for cls in CLASS_ORDER)
    print(f"\n{'fold':>4} {'train':>6} {'val':>4} | val {header}")
    for fold, split in enumerate(splits):
        counts = ' '.join(f"{sum(label_of[c] == cls for c in split['val']):>10d}" for cls in CLASS_ORDER)
        print(f"{fold:>4} {len(split['train']):>6} {len(split['val']):>4} |     {counts}")
    total = ' '.join(f"{labels.count(cls):>10d}" for cls in CLASS_ORDER)
    print(f"{'전체':>4} {'':>6} {len(identifiers):>4} |     {total}")


def warn_if_trained(results_dir, n_splits):
    """이미 학습한 fold 가 있으면 split 을 바꾸는 순간 그 모델의 val 셋 의미가 달라진다."""
    trained = []
    for config in sorted(os.listdir(results_dir)):
        for fold in range(n_splits):
            if os.path.isdir(os.path.join(results_dir, config, f"fold_{fold}")):
                trained.append(f"{config}/fold_{fold}")
    if trained:
        print(f"\n  경고: 학습된 fold 가 이미 있다 ({', '.join(trained)}). "
              "split 을 바꾸면 그 모델들은 버려야 한다.")


def main():
    parser = argparse.ArgumentParser(description='severity stratified 5-fold splits_final.json 생성')
    parser.add_argument('--preprocessed_dir', default=PREPROCESSED_DIR)
    parser.add_argument('--results_dir', default=RESULTS_DIR, help='학습 이력 경고용')
    parser.add_argument('--severity_csv', default=SEVERITY_CSV)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true', help='쓰지 않고 분포만 출력')
    parser.add_argument('--force', action='store_true', help='기존 splits_final.json 덮어쓰기')
    args = parser.parse_args()

    out_path = os.path.join(args.preprocessed_dir, 'splits_final.json')
    if os.path.exists(out_path) and not (args.force or args.dry_run):
        raise VerificationError(
            f"이미 있다: {out_path} — 학습을 이미 돌렸다면 이 파일이 그 fold 구성이다. "
            "덮어쓰려면 --force")

    severity_map = load_severity_mapping(args.severity_csv, mode='multi')
    print(f"\n전처리 디렉토리: {args.preprocessed_dir}")
    identifiers = load_identifiers(args.preprocessed_dir)
    labels = label_cases(identifiers, severity_map)
    print(f"  케이스 {len(identifiers)}건")

    splits = make_splits(identifiers, labels, args.n_splits, args.seed)
    verify_splits(splits, identifiers, labels, args.n_splits)
    report(splits, identifiers, labels)

    if os.path.isdir(args.results_dir):
        warn_if_trained(args.results_dir, args.n_splits)

    if args.dry_run:
        print(f"\n[dry-run] 쓰지 않음: {out_path}")
        return

    with open(out_path, 'w') as f:
        json.dump(splits, f, indent=4)
    with open(out_path) as f:
        verify_splits(json.load(f), identifiers, labels, args.n_splits)
    print(f"\n저장: {out_path} (재검증 통과)")


if __name__ == '__main__':
    main()
