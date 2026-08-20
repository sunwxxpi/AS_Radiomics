"""두 segmentation 학습의 fold validation 출력을 맞대 본다 — 판정 기준은 Dice 가 아니라 빈 마스크와 과분할이다.

nnU-Net 이 학습 끝에 남기는 `fold_X/validation/` 을 읽는다. 재추론하지 않으므로 GPU 가 필요 없다.
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd

# organize_total_stratified_dataset 는 data/dataprep/ 에 있고 이 파일은 그 아래 utils/ 다
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from organize_total_stratified_dataset import extract_patient_id, load_severity_mapping

RESULTS_DIR = '/home/psw/nnUNet/data/nnUNet_results/Dataset003_KMU_Cardiac_AVC_TRAIN_ONLY'
GT_DIR = ('/home/psw/nnUNet/data/nnUNet_preprocessed/'
          'Dataset003_KMU_Cardiac_AVC_TRAIN_ONLY/gt_segmentations')
SEVERITY_CSV = '/home/psw/AS_Radiomics/data/AS_CRF.csv'
BASELINE = 'nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres'
CANDIDATE = 'nnUNetTrainerTverskyCE_500epochs__nnUNetResEncUNetLPlans_smallpatch__3d_fullres'
CLASS_ORDER = ['normal', 'nonsevere', 'severe']
OVERSEG_RATIO = 2.0


class VerificationError(Exception):
    """읽을 것이 없으면 빈 표를 내는 대신 중단한다."""


def dice(a, b):
    n_a, n_b = int(a.sum()), int(b.sum())
    if n_a == 0 and n_b == 0:
        return 1.0
    return 2 * int((a & b).sum()) / (n_a + n_b)


def load_mask(path):
    return np.asanyarray(nib.load(path).dataobj) > 0


def collect_run(run_dir, folds, severity_map):
    """run 하나의 fold validation 출력을 케이스별로 읽는다."""
    rows = []
    for fold in folds:
        val_dir = os.path.join(run_dir, f'fold_{fold}', 'validation')
        if not os.path.isdir(val_dir):
            raise VerificationError(f"validation 출력 없음: {val_dir} — 학습이 아직 안 끝났다")
        cases = sorted(f[:-len('.nii.gz')] for f in os.listdir(val_dir) if f.endswith('.nii.gz'))
        if not cases:
            raise VerificationError(f"{val_dir} 에 마스크가 없다")
        for case in cases:
            gt_path = os.path.join(GT_DIR, f'{case}.nii.gz')
            if not os.path.isfile(gt_path):
                raise VerificationError(f"GT 없음: {gt_path}")
            pred, gt = load_mask(os.path.join(val_dir, f'{case}.nii.gz')), load_mask(gt_path)
            if pred.shape != gt.shape:
                raise VerificationError(f"{case}: shape 불일치 {pred.shape} vs {gt.shape}")
            n_gt, n_pred = int(gt.sum()), int(pred.sum())
            rows.append({'fold': fold, 'case': case,
                         'patient_id': extract_patient_id(case),
                         'severity': severity_map.get(extract_patient_id(case), '?'),
                         'n_gt': n_gt, 'n_pred': n_pred,
                         'dice': dice(pred, gt),
                         'is_empty': n_pred == 0,
                         'ratio': n_pred / n_gt if n_gt else np.nan})
    return pd.DataFrame(rows)


def summarize(df, label):
    empty = df[df.is_empty]
    overseg = df[(df.n_gt > 0) & (df.ratio > OVERSEG_RATIO)]
    print(f"\n[{label}] {len(df)}건")
    print(f"  mean Dice {df.dice.mean():.4f} / median {df.dice.median():.4f} / Dice=0 {(df.dice == 0).sum()}건")
    print(f"  빈 마스크 {len(empty)}건" + (f" — {', '.join(empty.patient_id)}" if len(empty) else ""))
    print(f"  과분할(pred > GT×{OVERSEG_RATIO:g}) {len(overseg)}건, "
          f"pred/GT 중앙 {df.ratio.median():.2f} · 총 pred 복셀 {df.n_pred.sum():,}")


def compare(base, cand):
    """같은 케이스에 대해 두 run 이 어떻게 달라졌는지. 빈 마스크의 이동 방향이 판정의 핵심이다."""
    merged = base.merge(cand, on=['fold', 'case', 'patient_id', 'severity'], suffixes=('_base', '_cand'))
    if len(merged) != len(base) or len(merged) != len(cand):
        raise VerificationError(f"케이스 집합이 다르다: base {len(base)} · cand {len(cand)} · 교집합 {len(merged)}")

    fixed = merged[merged.is_empty_base & ~merged.is_empty_cand]
    broken = merged[~merged.is_empty_base & merged.is_empty_cand]
    print(f"\n=== 빈 마스크 이동 ===")
    print(f"  살아남 {len(fixed)}건 / 새로 빔 {len(broken)}건")
    for _, r in pd.concat([fixed, broken]).iterrows():
        state = '복구' if r.n_pred_cand > 0 else '악화'
        print(f"  {state} {r.patient_id} ({r.severity}, GT {r.n_gt_base}): "
              f"pred {r.n_pred_base} -> {r.n_pred_cand}, Dice {r.dice_base:.3f} -> {r.dice_cand:.3f}")

    print(f"\n=== Dice 변화 ===")
    delta = merged.dice_cand - merged.dice_base
    print(f"  mean {merged.dice_base.mean():.4f} -> {merged.dice_cand.mean():.4f} ({delta.mean():+.4f})")
    print(f"  개선 {(delta > 0.01).sum()}건 / 악화 {(delta < -0.01).sum()}건 / 유지 {(delta.abs() <= 0.01).sum()}건")

    print(f"\n=== GT 크기 구간별 ===")
    bins = [0, 50, 100, 500, 2000, np.inf]
    labels = ['0~50', '50~100', '100~500', '500~2000', '2000~']
    merged['bin'] = pd.cut(merged.n_gt_base, bins=bins, labels=labels, right=False)
    print(f"{'GT voxel':>10} {'n':>4} {'base Dice':>10} {'cand Dice':>10} {'base 빈':>7} {'cand 빈':>7} "
          f"{'base 과분할':>11} {'cand 과분할':>11}")
    for name, g in merged.groupby('bin', observed=True):
        print(f"{name:>10} {len(g):>4} {g.dice_base.mean():>10.3f} {g.dice_cand.mean():>10.3f} "
              f"{g.is_empty_base.sum():>7} {g.is_empty_cand.sum():>7} "
              f"{((g.n_gt_base > 0) & (g.ratio_base > OVERSEG_RATIO)).sum():>11} "
              f"{((g.n_gt_cand > 0) & (g.ratio_cand > OVERSEG_RATIO)).sum():>11}")

    print(f"\n=== severity 별 빈 마스크 ===")
    for cls in CLASS_ORDER:
        g = merged[merged.severity == cls]
        if len(g):
            print(f"  {cls:>10} {len(g):>4}건: base {g.is_empty_base.sum()} -> cand {g.is_empty_cand.sum()}")
    return merged


def main():
    parser = argparse.ArgumentParser(description='두 학습의 fold validation 출력 비교 (빈 마스크·과분할 중심)')
    parser.add_argument('--results_dir', default=RESULTS_DIR)
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--candidate', default=CANDIDATE)
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--severity_csv', default=SEVERITY_CSV)
    parser.add_argument('--out_csv', default=None, help='케이스별 비교표 저장 경로')
    args = parser.parse_args()

    severity_map = load_severity_mapping(args.severity_csv, mode='multi')
    base_dir = os.path.join(args.results_dir, args.baseline)
    cand_dir = os.path.join(args.results_dir, args.candidate)
    print(f"\nbaseline : {base_dir}")
    print(f"candidate: {cand_dir}")
    print(f"fold     : {args.folds}")

    base = collect_run(base_dir, args.folds, severity_map)
    cand = collect_run(cand_dir, args.folds, severity_map)
    summarize(base, 'baseline')
    summarize(cand, 'candidate')
    merged = compare(base, cand)

    if args.out_csv:
        merged.to_csv(args.out_csv, index=False)
        print(f"\n저장: {args.out_csv} ({len(merged)}행)")


if __name__ == '__main__':
    main()
