"""Dataset004_baseline 생성 — 406건에 cross-fitting 예측 마스크를 붙인다.

각 케이스는 자기를 학습하지 않은 fold 모델 하나로만 추론한다. 앙상블을 쓰면 GT 보유 250건은
자기 fold 모델 1개밖에 못 쓰는 반면 미보유 156건만 더 좋은 마스크를 받아, train/test 마스크
품질이 갈라진다.
GT 보유 250건에는 학습 때 만들어진 fold_{X}/validation 출력이 이미 있지만 재사용하지 않는다 —
156건은 어차피 추론해야 하고, 경로가 둘로 갈리면 그 차이가 GT 보유 여부(=클래스와 상관)를
따라 붙는다. 대신 그 출력은 fold 배정 검증용 대조군으로 쓴다.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd
import SimpleITK as sitk

# organize_total_stratified_dataset 는 data/dataprep/ 에 있고 이 파일은 그 아래 utils/ 다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from organize_total_stratified_dataset import extract_patient_id

GT_DATASET_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset004_gt'
PRED_DATASET_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset004_baseline'
INFO_CSV_NAME = 'Dataset004_info.csv'
SPLITS_JSON = ('/home/psw/nnUNet/data/nnUNet_preprocessed/'
               'Dataset003_KMU_Cardiac_AVC_TRAIN_ONLY/splits_final.json')
RAW_IMAGE_DIR = ('/home/psw/nnUNet/data/nnUNet_raw/'
                 'Dataset003_KMU_Cardiac_AVC_TRAIN_ONLY/imagesTr')
MODEL_DIR = ('/home/psw/nnUNet/data/nnUNet_results/Dataset003_KMU_Cardiac_AVC_TRAIN_ONLY/'
             'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres')

DATASET_ID = '3'
CONFIGURATION = '3d_fullres'
PLANS = 'nnUNetResEncUNetLPlans'
TRAINER = 'nnUNetTrainer'
N_FOLDS = 5
CLASS_ORDER = ['normal', 'nonsevere', 'severe']
GEOMETRY_TOLERANCE = 1e-5
# 학습 때 검증 출력과 비트 단위로 일치하는 것이 정상이라 (실측 Dice 1.000000) 둘 다 여유가 크다.
# fold 를 잘못 배정하면 그 fold 50건이 통째로 낮은 Dice 로 떨어진다.
MIN_MEAN_DICE_VS_VALIDATION = 0.95
MAX_LOW_DICE_CASES = 5
ASSIGNMENT_COLUMNS = ['patient_id', 'severity', 'split', 'has_gt',
                      'new_img_file', 'new_label_file', 'fold', 'assign_source']


class VerificationError(Exception):
    """배정·추론·수집 검증 실패. 잘못된 마스크 세트가 남지 않도록 즉시 중단시킨다."""


def image_subdir(split):
    return 'imagesTr' if split == 'train' else 'imagesVal'


def label_subdir(split):
    return 'labelsTr' if split == 'train' else 'labelsVal'


def case_id_of(new_img_file):
    """이미지 파일명에서 nnU-Net 케이스 식별자. 추론 출력이 이 이름으로 나온다."""
    return new_img_file[:-len('_0000.nii.gz')]


def load_splits(splits_json):
    with open(splits_json) as f:
        splits = json.load(f)
    if len(splits) != N_FOLDS:
        raise VerificationError(f"fold 수 불일치: {len(splits)} != {N_FOLDS}")
    return splits


def build_assignment(info_csv, splits):
    """406건에 fold 를 배정한다. GT 보유는 자기가 held-out 인 fold, 미보유는 라운드로빈."""
    info = pd.read_csv(info_csv).sort_values('patient_id').reset_index(drop=True)
    if len(info) != 406:
        raise VerificationError(f"info CSV 가 {len(info)}행 (기대 406행)")

    fold_of_gt = {}
    val_case_of = {}
    for fold, split in enumerate(splits):
        for case in split['val']:
            patient_id = extract_patient_id(case)
            if patient_id in fold_of_gt:
                raise VerificationError(f"splits_final 에 중복 patient_id: {patient_id}")
            fold_of_gt[patient_id] = fold
            val_case_of[patient_id] = case

    gt_patients = set(info.loc[info['has_gt'], 'patient_id'])
    if gt_patients != set(fold_of_gt):
        only_info = sorted(gt_patients - set(fold_of_gt))[:10]
        only_splits = sorted(set(fold_of_gt) - gt_patients)[:10]
        raise VerificationError(
            f"GT 보유 환자 집합이 splits_final 과 다르다 "
            f"(info 만 {len(gt_patients - set(fold_of_gt))}명 {only_info}, "
            f"splits 만 {len(set(fold_of_gt) - gt_patients)}명 {only_splits})")

    rows = []
    no_gt_index = 0
    for _, row in info.iterrows():
        if row['has_gt']:
            fold, source = fold_of_gt[row['patient_id']], 'held_out'
        else:
            fold, source = no_gt_index % N_FOLDS, 'round_robin'
            no_gt_index += 1
        rows.append({
            'patient_id': row['patient_id'],
            'severity': row['severity'],
            'split': row['split'],
            'has_gt': bool(row['has_gt']),
            'new_img_file': row['new_img_file'],
            'new_label_file': row['new_label_file'],
            'fold': fold,
            'assign_source': source,
        })
    return pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS), val_case_of


def sync_assignment(assignment, path):
    """배정표를 쓰거나, 이미 있으면 같은지 대조한다.

    두 터미널이 fold 를 나눠 돌 때 이 단계를 동시에 밟으므로 원자적으로 쓴다.
    """
    payload = assignment.to_csv(index=False)
    if os.path.exists(path):
        with open(path) as f:
            existing = f.read()
        if existing != payload:
            raise VerificationError(
                f"기존 배정표와 다르다: {path} — 이전 실행이 다른 입력으로 만든 것이다. "
                "이미 추론했다면 그 마스크의 fold 배정이 이 표와 어긋난다")
        print(f"  배정표 일치 확인: {path}")
        return

    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(payload)
    os.replace(tmp, path)
    print(f"  배정표 저장: {path} ({len(assignment)}행)")


def report_assignment(assignment):
    print("\n=== fold 배정 ===")
    header = f"{'fold':>4} {'계':>5} {'held_out':>9} {'round_robin':>12} | " \
             + ' '.join(f"{c:>10}" for c in CLASS_ORDER)
    print(header)
    for fold in range(N_FOLDS):
        sub = assignment[assignment['fold'] == fold]
        dist = Counter(sub['severity'])
        print(f"{fold:>4} {len(sub):>5} {(sub['assign_source'] == 'held_out').sum():>9} "
              f"{(sub['assign_source'] == 'round_robin').sum():>12} | "
              + ' '.join(f"{dist[c]:>10}" for c in CLASS_ORDER))
    dist = Counter(assignment['severity'])
    print(f"{'계':>4} {len(assignment):>5} {(assignment['assign_source'] == 'held_out').sum():>9} "
          f"{(assignment['assign_source'] == 'round_robin').sum():>12} | "
          + ' '.join(f"{dist[c]:>10}" for c in CLASS_ORDER))


def verify_assignment_sets(assignment, splits):
    """fold 별 배정 patient_id 집합이 splits_final 의 val 과 같은지 — 임계값 없는 정확 검사"""
    for fold, split in enumerate(splits):
        expected = {extract_patient_id(c) for c in split['val']}
        actual = set(assignment[(assignment['fold'] == fold) & assignment['has_gt']]['patient_id'])
        if actual != expected:
            raise VerificationError(
                f"fold {fold}: GT 보유 배정이 splits_final val 과 다르다 "
                f"(누락 {len(expected - actual)}, 초과 {len(actual - expected)})")
    print(f"검증 통과 (a): GT 보유 250건의 fold 배정 == splits_final val")


def md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image_identity(assignment, gt_dataset_dir, raw_image_dir):
    """GT 보유 250건의 이미지가 학습에 쓰인 파일과 바이트 동일한지 — patient_id 조인 검사"""
    raw_by_patient = {}
    for name in os.listdir(raw_image_dir):
        if name.endswith('.nii.gz'):
            raw_by_patient[extract_patient_id(name)] = os.path.join(raw_image_dir, name)

    checked = 0
    for _, row in assignment[assignment['has_gt']].iterrows():
        raw_path = raw_by_patient.get(row['patient_id'])
        if raw_path is None:
            raise VerificationError(f"{row['patient_id']}: 학습 데이터셋에 이미지가 없다")
        new_path = os.path.join(gt_dataset_dir, image_subdir(row['split']), row['new_img_file'])
        if md5(new_path) != md5(raw_path):
            raise VerificationError(
                f"{row['patient_id']}: 이미지가 학습 파일과 다르다 "
                f"({row['new_img_file']} vs {os.path.basename(raw_path)})")
        checked += 1
    print(f"검증 통과 (b): 이미지 {checked}건이 학습 파일과 바이트 동일")


def link_fold_inputs(assignment, fold, gt_dataset_dir, input_dir):
    """해당 fold 케이스의 이미지를 symlink 로 모은다. 파일명을 유지해야 출력이 자동으로 맞는다."""
    os.makedirs(input_dir, exist_ok=True)
    sub = assignment[assignment['fold'] == fold]
    for _, row in sub.iterrows():
        src = os.path.join(gt_dataset_dir, image_subdir(row['split']), row['new_img_file'])
        if not os.path.isfile(src):
            raise VerificationError(f"이미지 없음: {src}")
        dst = os.path.join(input_dir, row['new_img_file'])
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    stale = set(os.listdir(input_dir)) - set(sub['new_img_file'])
    if stale:
        raise VerificationError(f"fold {fold} 입력에 배정 밖 파일 {len(stale)}개: {sorted(stale)[:5]}")
    return len(sub)


def run_predict(fold, input_dir, output_dir, device, force, trainer):
    if force and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cmd = ['nnUNetv2_predict', '-i', input_dir, '-o', output_dir,
           '-d', DATASET_ID, '-c', CONFIGURATION, '-p', PLANS, '-tr', trainer,
           '-f', str(fold)]
    if not force:
        cmd.append('--continue_prediction')

    env = os.environ.copy()
    if device is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(device)

    print(f"\n$ {' '.join(cmd)}"
          + (f"   (CUDA_VISIBLE_DEVICES={device})" if device is not None else ''))
    subprocess.run(cmd, check=True, env=env)


def verify_fold_output(assignment, fold, output_dir):
    expected = {f"{case_id_of(f)}.nii.gz" for f in assignment[assignment['fold'] == fold]['new_img_file']}
    actual = {f for f in os.listdir(output_dir) if f.endswith('.nii.gz')}
    if actual != expected:
        raise VerificationError(
            f"fold {fold} 출력 불일치 (누락 {len(expected - actual)}건 "
            f"{sorted(expected - actual)[:5]}, 초과 {len(actual - expected)}건)")
    print(f"  fold {fold}: 마스크 {len(actual)}건 확인")


def collect(assignment, work_dir, gt_dataset_dir, pred_dataset_dir):
    """이미지는 _gt 에서 하드링크, 마스크는 fold 출력에서 복사"""
    for sub in ('imagesTr', 'imagesVal', 'labelsTr', 'labelsVal'):
        os.makedirs(os.path.join(pred_dataset_dir, sub), exist_ok=True)

    for fold in range(N_FOLDS):
        output_dir = os.path.join(work_dir, f'fold_{fold}', 'output')
        if not os.path.isdir(output_dir):
            raise VerificationError(f"fold {fold} 추론 출력이 없다: {output_dir}")
        verify_fold_output(assignment, fold, output_dir)

    for _, row in assignment.iterrows():
        img_src = os.path.join(gt_dataset_dir, image_subdir(row['split']), row['new_img_file'])
        img_dst = os.path.join(pred_dataset_dir, image_subdir(row['split']), row['new_img_file'])
        if os.path.exists(img_dst):
            os.remove(img_dst)
        os.link(img_src, img_dst)

        mask_src = os.path.join(work_dir, f"fold_{row['fold']}", 'output',
                                f"{case_id_of(row['new_img_file'])}.nii.gz")
        mask_dst = os.path.join(pred_dataset_dir, label_subdir(row['split']), row['new_label_file'])
        shutil.copy2(mask_src, mask_dst)

    print(f"\n수집 완료: 이미지 406건 하드링크, 마스크 406건 복사")


def verify_output(assignment, gt_dataset_dir, pred_dataset_dir):
    """개수와 이미지-마스크 헤더 일치"""
    expected_counts = {
        'imagesTr': (assignment['split'] == 'train').sum(),
        'imagesVal': (assignment['split'] == 'val').sum(),
        'labelsTr': (assignment['split'] == 'train').sum(),
        'labelsVal': (assignment['split'] == 'val').sum(),
    }
    for sub, expected in expected_counts.items():
        actual = len([f for f in os.listdir(os.path.join(pred_dataset_dir, sub))
                      if f.endswith('.nii.gz')])
        if actual != expected:
            raise VerificationError(f"{sub}: {actual}개 (기대 {expected}개)")

    reader = sitk.ImageFileReader()
    for _, row in assignment.iterrows():
        metas = []
        for path in (os.path.join(gt_dataset_dir, image_subdir(row['split']), row['new_img_file']),
                     os.path.join(pred_dataset_dir, label_subdir(row['split']), row['new_label_file'])):
            reader.SetFileName(path)
            reader.ReadImageInformation()
            metas.append((reader.GetSize(), reader.GetSpacing(),
                          reader.GetOrigin(), reader.GetDirection()))
        img_meta, lbl_meta = metas
        if img_meta[0] != lbl_meta[0]:
            raise VerificationError(
                f"{row['patient_id']}: size 불일치 {img_meta[0]} vs {lbl_meta[0]}")
        for idx, field in enumerate(('spacing', 'origin', 'direction'), start=1):
            if any(abs(a - b) > GEOMETRY_TOLERANCE for a, b in zip(img_meta[idx], lbl_meta[idx])):
                raise VerificationError(
                    f"{row['patient_id']}: {field} 불일치 {img_meta[idx]} vs {lbl_meta[idx]}")

    print(f"검증 통과 (d): {dict(expected_counts)}, 이미지-마스크 헤더 406건 일치")


def dice(a, b):
    """둘 다 비면 1.0 — 빈 마스크끼리는 일치로 본다"""
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return 1.0
    return 2.0 * int(np.logical_and(a, b).sum()) / total


def collect_mask_stats(assignment, pred_dataset_dir, stats_csv):
    """케이스별 복셀 수·연결성분 수. 빈 마스크는 9번 pairing 을 깨므로 따로 집계한다."""
    records = []
    for _, row in assignment.iterrows():
        path = os.path.join(pred_dataset_dir, label_subdir(row['split']), row['new_label_file'])
        image = sitk.ReadImage(path)
        array = sitk.GetArrayFromImage(image)

        values = set(np.unique(array).tolist())
        if not values <= {0, 1}:
            raise VerificationError(f"{row['patient_id']}: 라벨값이 {{0,1}} 밖 — {sorted(values)}")

        n_voxel = int((array == 1).sum())
        if n_voxel:
            components = sitk.RelabelComponent(sitk.ConnectedComponent(image == 1))
            n_component = int(sitk.GetArrayFromImage(components).max())
        else:
            n_component = 0
        spacing = image.GetSpacing()

        records.append({
            'patient_id': row['patient_id'],
            'severity': row['severity'],
            'split': row['split'],
            'has_gt': row['has_gt'],
            'fold': row['fold'],
            'new_label_file': row['new_label_file'],
            'n_voxel': n_voxel,
            'volume_mm3': n_voxel * spacing[0] * spacing[1] * spacing[2],
            'n_component': n_component,
            'is_empty': n_voxel == 0,
        })

    stats = pd.DataFrame(records)
    stats.to_csv(stats_csv, index=False)
    print(f"\n마스크 통계 저장: {stats_csv} ({len(stats)}행)")
    return stats


def report_empty_masks(stats):
    """빈 예측 마스크 — PyRadiomics 가 케이스를 통째로 빼므로 9번의 paired 설계가 여기 걸린다"""
    empty = stats[stats['is_empty']]
    print(f"\n=== 빈 예측 마스크 {len(empty)}건 / {len(stats)}건 ===")
    if empty.empty:
        return
    for split in ('train', 'val'):
        sub = empty[empty['split'] == split]
        dist = Counter(sub['severity'])
        print(f"  {split:<6} {len(sub):>3}건  " + ' '.join(f"{c}={dist[c]}" for c in CLASS_ORDER))
    print("  test(val) 케이스: "
          + (', '.join(sorted(empty[empty['split'] == 'val']['patient_id'])) or '없음'))


def verify_vs_validation(assignment, val_case_of, pred_dataset_dir, model_dir):
    """새 예측을 학습 때 나온 held-out 출력과 대조 — fold 배정·파일명 매핑의 최종 그물"""
    scores = []
    for _, row in assignment[assignment['has_gt']].iterrows():
        ref_path = os.path.join(model_dir, f"fold_{row['fold']}", 'validation',
                                f"{val_case_of[row['patient_id']]}.nii.gz")
        if not os.path.isfile(ref_path):
            raise VerificationError(f"학습 시 검증 출력이 없다: {ref_path}")
        new_path = os.path.join(pred_dataset_dir, label_subdir(row['split']), row['new_label_file'])

        ref = sitk.GetArrayFromImage(sitk.ReadImage(ref_path)) == 1
        new = sitk.GetArrayFromImage(sitk.ReadImage(new_path)) == 1
        if ref.shape != new.shape:
            raise VerificationError(
                f"{row['patient_id']}: 마스크 shape 불일치 {ref.shape} vs {new.shape}")
        scores.append({'patient_id': row['patient_id'], 'fold': row['fold'],
                       'dice': dice(ref, new)})

    frame = pd.DataFrame(scores)
    mean_dice = frame['dice'].mean()
    print(f"\n=== 학습 시 held-out 출력과 대조 ({len(frame)}건) ===")
    print(f"  mean {mean_dice:.4f} / median {frame['dice'].median():.4f} "
          f"/ min {frame['dice'].min():.4f}")
    low = frame[frame['dice'] < 0.9].sort_values('dice')
    print(f"  Dice < 0.9: {len(low)}건"
          + (f" — {', '.join(f'{r.patient_id}({r.dice:.2f})' for r in low.head(10).itertuples())}"
             if len(low) else ''))

    if mean_dice < MIN_MEAN_DICE_VS_VALIDATION or len(low) > MAX_LOW_DICE_CASES:
        raise VerificationError(
            f"평균 Dice {mean_dice:.4f} (>= {MIN_MEAN_DICE_VS_VALIDATION} 이어야 함), "
            f"Dice < 0.9 가 {len(low)}건 (<= {MAX_LOW_DICE_CASES} 이어야 함) — "
            "fold 배정이나 파일명 매핑이 틀렸을 가능성이 크다")
    print(f"검증 통과 (c): 평균 Dice {mean_dice:.4f}, Dice < 0.9 {len(low)}건")
    return frame


def main():
    parser = argparse.ArgumentParser(
        description='Dataset004_baseline 생성 (fold 배정 -> 추론 -> 수집·검증)')
    parser.add_argument('--gt_dataset_dir', default=GT_DATASET_DIR)
    parser.add_argument('--pred_dataset_dir', default=PRED_DATASET_DIR)
    parser.add_argument('--splits_json', default=SPLITS_JSON)
    parser.add_argument('--raw_image_dir', default=RAW_IMAGE_DIR)
    parser.add_argument('--model_dir', default=MODEL_DIR)
    parser.add_argument('--trainer', default=TRAINER,
                        help='--model_dir 의 trainer 와 반드시 같아야 한다')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(N_FOLDS)),
                        help='추론할 fold. GPU 를 나눠 쓸 때 두 번 실행한다')
    parser.add_argument('--device', default=None,
                        help='CUDA_VISIBLE_DEVICES 에 넣을 GPU 번호')
    parser.add_argument('--collect_only', action='store_true',
                        help='추론 없이 수집·검증만 — 모든 fold 가 끝난 뒤 한 번 실행한다')
    parser.add_argument('--force', action='store_true',
                        help='기존 추론 출력을 지우고 다시 추론한다')
    parser.add_argument('--dry_run', action='store_true', help='배정만 계산하고 출력')
    args = parser.parse_args()

    invalid = [f for f in args.folds if f not in range(N_FOLDS)]
    if invalid:
        raise VerificationError(f"fold 범위 밖: {invalid}")

    # 둘이 어긋나면 A 모델로 추론하고 B 모델의 검증 출력과 대조하게 된다
    expected_dir = f"{args.trainer}__{PLANS}__{CONFIGURATION}"
    if os.path.basename(args.model_dir.rstrip('/')) != expected_dir:
        raise VerificationError(
            f"--trainer 와 --model_dir 불일치: {args.trainer} 는 {expected_dir} 를 기대하는데 "
            f"{os.path.basename(args.model_dir.rstrip('/'))} 가 왔다")

    info_csv = os.path.join(args.gt_dataset_dir, INFO_CSV_NAME)
    work_dir = os.path.join(args.pred_dataset_dir, '_work')

    print("=== Dataset004_baseline 생성 ===")
    print(f"이미지 소스: {args.gt_dataset_dir}")
    print(f"모델: {args.model_dir}")
    print(f"출력: {args.pred_dataset_dir}")
    mode = 'collect' if args.collect_only else f"predict folds={args.folds}"
    print(f"모드: {mode}{'  [DRY RUN]' if args.dry_run else ''}")

    splits = load_splits(args.splits_json)
    assignment, val_case_of = build_assignment(info_csv, splits)
    verify_assignment_sets(assignment, splits)
    report_assignment(assignment)

    if args.dry_run:
        print("\nDRY RUN — 배정표를 쓰지 않고 종료")
        return

    os.makedirs(args.pred_dataset_dir, exist_ok=True)
    sync_assignment(assignment, os.path.join(args.pred_dataset_dir, 'fold_assignment.csv'))

    if not args.collect_only:
        verify_image_identity(assignment, args.gt_dataset_dir, args.raw_image_dir)
        for fold in args.folds:
            fold_dir = os.path.join(work_dir, f'fold_{fold}')
            input_dir, output_dir = os.path.join(fold_dir, 'input'), os.path.join(fold_dir, 'output')
            n_case = link_fold_inputs(assignment, fold, args.gt_dataset_dir, input_dir)
            print(f"\n--- fold {fold}: {n_case}건 ---")
            run_predict(fold, input_dir, output_dir, args.device, args.force, args.trainer)
            verify_fold_output(assignment, fold, output_dir)
        print(f"\n추론 완료 (folds={args.folds}). 전 fold 가 끝나면 --collect_only 로 수집한다")
        return

    collect(assignment, work_dir, args.gt_dataset_dir, args.pred_dataset_dir)
    verify_image_identity(assignment, args.gt_dataset_dir, args.raw_image_dir)
    verify_output(assignment, args.gt_dataset_dir, args.pred_dataset_dir)
    verify_vs_validation(assignment, val_case_of, args.pred_dataset_dir, args.model_dir)
    stats = collect_mask_stats(assignment, args.pred_dataset_dir,
                               os.path.join(args.pred_dataset_dir, 'pred_mask_stats.csv'))
    report_empty_masks(stats)

    print("\n=== 완료 ===")


if __name__ == '__main__':
    main()
