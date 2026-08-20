"""Dataset004 변형에서 특정 케이스를 빼고 새 데이터셋을 만든다.

빈 마스크 케이스를 예외 핸들러가 조용히 드롭하는 데 맡기지 않고 데이터셋 구조에서 없애는 것이 목적이다.
`features_extractor.py:191` 이 PyRadiomics 의 `ValueError` 를 잡아 케이스를 빼기 때문에, 제외를 코드에
두면 보고되는 N 이 우연의 산물이 되고 그 핸들러가 바뀌는 순간 파이프라인이 깨진다.
디렉토리 자체에서 빼면 radiomics·DL·crop 이 전부 같은 케이스 집합을 본다.
"""

import argparse
import os
import shutil
import sys

import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ablation_pred_masks import (VerificationError, image_subdir, label_subdir,
                                      collect_mask_stats, report_empty_masks)

DATASETS_DIR = '/home/psw/AS_Radiomics/data/datasets'
SOURCE = 'Dataset004_mix'
DEST = 'Dataset004_mix_KUDH0467rm'
GT_DATASET_DIR = f'{DATASETS_DIR}/Dataset004_gt'
# Tversky·baseline 5 fold 개별 + 앙상블 10개 가중치 전부에서 전경 확률 0.0000. 어떤 후처리로도 못 살린다.
EXCLUDE = {'KUDH0467': '406건 중 유일한 조영증강 스캔 (HU>130 이 24.0%, 2위 9.1%) — 12개 조합 전부 빈 마스크'}
SIDECAR_CSVS = ['fold_assignment.csv', 'mask_source.csv']


def build(source_dir, dest_dir, gt_dataset_dir, exclude):
    assignment = pd.read_csv(os.path.join(source_dir, 'fold_assignment.csv'))
    missing = set(exclude) - set(assignment['patient_id'])
    if missing:
        raise VerificationError(f"원본에 없는 제외 대상: {sorted(missing)}")

    kept = assignment[~assignment['patient_id'].isin(exclude)].reset_index(drop=True)
    for sub in ('imagesTr', 'imagesVal', 'labelsTr', 'labelsVal'):
        os.makedirs(os.path.join(dest_dir, sub), exist_ok=True)

    for _, row in kept.iterrows():
        img_src = os.path.join(gt_dataset_dir, image_subdir(row['split']), row['new_img_file'])
        img_dst = os.path.join(dest_dir, image_subdir(row['split']), row['new_img_file'])
        if os.path.exists(img_dst):
            os.remove(img_dst)
        os.link(img_src, img_dst)
        shutil.copy2(os.path.join(source_dir, label_subdir(row['split']), row['new_label_file']),
                     os.path.join(dest_dir, label_subdir(row['split']), row['new_label_file']))

    for name in SIDECAR_CSVS:
        path = os.path.join(source_dir, name)
        if not os.path.isfile(path):
            continue
        frame = pd.read_csv(path)
        frame[~frame['patient_id'].isin(exclude)].to_csv(os.path.join(dest_dir, name), index=False)

    dropped = assignment[assignment['patient_id'].isin(exclude)].copy()
    dropped['reason'] = dropped['patient_id'].map(exclude)
    dropped.to_csv(os.path.join(dest_dir, 'excluded.csv'), index=False)
    print(f"\n수집 완료: {len(kept)}건 (원본 {len(assignment)}건에서 {len(dropped)}건 제외)")
    return assignment, kept, dropped


def verify_counts(assignment, kept, dest_dir):
    expected = {
        'imagesTr': int((kept['split'] == 'train').sum()),
        'imagesVal': int((kept['split'] == 'val').sum()),
        'labelsTr': int((kept['split'] == 'train').sum()),
        'labelsVal': int((kept['split'] == 'val').sum()),
    }
    for sub, want in expected.items():
        got = len([f for f in os.listdir(os.path.join(dest_dir, sub)) if f.endswith('.nii.gz')])
        if got != want:
            raise VerificationError(f"{sub}: {got}개 (기대 {want}개)")
    print(f"검증 통과 (a): {expected}")


def verify_masks_identical(kept, source_dir, dest_dir):
    """남긴 케이스의 마스크가 원본과 바이트 동일한지 — 제외 말고 달라진 게 없어야 한다"""
    for _, row in kept.iterrows():
        paths = [os.path.join(d, label_subdir(row['split']), row['new_label_file'])
                 for d in (source_dir, dest_dir)]
        with open(paths[0], 'rb') as f0, open(paths[1], 'rb') as f1:
            if f0.read() != f1.read():
                raise VerificationError(f"{row['patient_id']}: 마스크가 원본과 다르다")
    print(f"검증 통과 (b): 남긴 {len(kept)}건 마스크가 원본과 바이트 동일")


def verify_excluded_absent(dropped, dest_dir):
    for _, row in dropped.iterrows():
        for sub, name in ((image_subdir(row['split']), row['new_img_file']),
                          (label_subdir(row['split']), row['new_label_file'])):
            path = os.path.join(dest_dir, sub, name)
            if os.path.exists(path):
                raise VerificationError(f"제외 대상이 남아 있다: {path}")
    print(f"검증 통과 (c): 제외 {len(dropped)}건의 이미지·마스크 모두 부재")


def verify_no_empty_mask(kept, dest_dir):
    """제외의 목적 자체 — 빈 마스크가 하나도 남지 않아야 한다"""
    empty = []
    for _, row in kept.iterrows():
        path = os.path.join(dest_dir, label_subdir(row['split']), row['new_label_file'])
        if int((sitk.GetArrayFromImage(sitk.ReadImage(path)) == 1).sum()) == 0:
            empty.append(row['patient_id'])
    if empty:
        raise VerificationError(f"빈 마스크가 남았다: {empty}")
    print(f"검증 통과 (d): 빈 마스크 0건")


def report(assignment, kept, dropped):
    print("\n=== 제외 ===")
    for _, row in dropped.iterrows():
        print(f"  {row['patient_id']} ({row['severity']}, {row['split']}, "
              f"has_gt={row['has_gt']}): {row['reason']}")
    print("\n=== 클래스 분포 ===")
    order = ['normal', 'nonsevere', 'severe']
    print(f"{'split':>6} {'전':>5} {'후':>5} | " + ' '.join(f"{c:>10}" for c in order))
    for split in ('train', 'val'):
        before, after = assignment[assignment['split'] == split], kept[kept['split'] == split]
        counts = ' '.join(f"{int((after['severity'] == c).sum()):>10}" for c in order)
        print(f"{split:>6} {len(before):>5} {len(after):>5} | {counts}")
    print(f"{'계':>6} {len(assignment):>5} {len(kept):>5}")


def main():
    parser = argparse.ArgumentParser(description='Dataset004 변형에서 특정 케이스를 뺀 데이터셋 생성')
    parser.add_argument('--datasets_dir', default=DATASETS_DIR)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--dest', default=DEST)
    parser.add_argument('--gt_dataset_dir', default=GT_DATASET_DIR)
    parser.add_argument('--exclude', nargs='+', default=None,
                        help='제외할 patient_id. 생략하면 EXCLUDE 상수를 쓴다')
    args = parser.parse_args()

    exclude = {p: '커맨드라인 지정' for p in args.exclude} if args.exclude else EXCLUDE
    source_dir = os.path.join(args.datasets_dir, args.source)
    dest_dir = os.path.join(args.datasets_dir, args.dest)
    print(f"=== {args.dest} 생성 ===")
    print(f"원본: {source_dir}")
    print(f"제외: {sorted(exclude)}")

    assignment, kept, dropped = build(source_dir, dest_dir, args.gt_dataset_dir, exclude)
    verify_counts(assignment, kept, dest_dir)
    verify_masks_identical(kept, source_dir, dest_dir)
    verify_excluded_absent(dropped, dest_dir)
    verify_no_empty_mask(kept, dest_dir)
    report(assignment, kept, dropped)

    stats = collect_mask_stats(kept, dest_dir, os.path.join(dest_dir, 'pred_mask_stats.csv'))
    report_empty_masks(stats)
    print("\n=== 완료 ===")


if __name__ == '__main__':
    main()
