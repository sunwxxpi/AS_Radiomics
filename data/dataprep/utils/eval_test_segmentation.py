"""test 83 의 분할 성능 산출 — Supplementary Table 1 과 Supplementary Figure 1 의 케이스 선정.

예측은 `Dataset004_mix_KUDH0467rm` 의 out-of-fold 마스크이고 정답은 `Dataset004_gt` 의 GT 마스크다.
이 코호트의 Dice 는 양봉이라 mean ± SD 만 적으면 오해를 만들어 median(IQR) 과 실패 건수를 같이 낸다.
"""

import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd

PRED_DATASET_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset004_mix_KUDH0467rm'
GT_DATASET_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset004_gt'
CLASS_ORDER = ['normal', 'nonsevere', 'severe']
CLASS_LABEL = {'normal': 'Normal', 'nonsevere': 'Non-severe AS', 'severe': 'Severe AS'}
FAILURE_THRESHOLD = 0.5


class VerificationError(Exception):
    """케이스가 빠지거나 geometry 가 어긋나면 빈 표를 내는 대신 중단한다."""


def load_mask(path):
    return np.asanyarray(nib.load(path).dataobj) > 0


def dice(pred, gt):
    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    if n_pred == 0 and n_gt == 0:
        return 1.0
    return 2 * int((pred & gt).sum()) / (n_pred + n_gt)


def jaccard(pred, gt):
    union = int((pred | gt).sum())
    if union == 0:
        return 1.0
    return int((pred & gt).sum()) / union


def is_in_sample(row):
    """예측에 쓴 fold 모델이 그 환자를 학습했는지. nnU-Net fold k 는 fold k 를 빼고 학습한다."""
    if not row['has_gt']:
        return False
    if row['mask_source'] == 'gt':
        return True
    return row['source_fold'] != row['assigned_fold']


def collect(pred_dataset_dir, gt_dataset_dir):
    """test 83 의 케이스별 DSC · IoU. 파일명은 두 데이터셋이 같은 재분할 번호를 쓴다."""
    source = pd.read_csv(os.path.join(pred_dataset_dir, 'mask_source.csv'))
    assignment = pd.read_csv(os.path.join(pred_dataset_dir, 'fold_assignment.csv'))
    test = source[source['split'] == 'val'].merge(
        assignment[['patient_id', 'new_label_file']], on='patient_id', validate='one_to_one')
    if len(test) != 83:
        raise VerificationError(f"test 가 83건이 아니다: {len(test)}건")
    if not test['has_gt'].all():
        raise VerificationError("test 에 GT 미보유 케이스가 있다")

    rows = []
    for _, row in test.iterrows():
        pred_path = os.path.join(pred_dataset_dir, 'labelsVal', row['new_label_file'])
        gt_path = os.path.join(gt_dataset_dir, 'labelsVal', row['new_label_file'])
        for path in (pred_path, gt_path):
            if not os.path.isfile(path):
                raise VerificationError(f"{row['patient_id']}: 마스크 없음 — {path}")
        pred, gt = load_mask(pred_path), load_mask(gt_path)
        if pred.shape != gt.shape:
            raise VerificationError(f"{row['patient_id']}: shape 불일치 {pred.shape} vs {gt.shape}")
        if not int(gt.sum()):
            raise VerificationError(f"{row['patient_id']}: GT 가 비었다")
        rows.append({'patient_id': row['patient_id'], 'severity': row['severity'],
                     'label_file': row['new_label_file'],
                     'dsc': dice(pred, gt), 'iou': jaccard(pred, gt),
                     'gt_voxels': int(gt.sum()), 'pred_voxels': int(pred.sum()),
                     'assigned_fold': row['assigned_fold'], 'source_fold': row['source_fold'],
                     'mask_source': row['mask_source'], 'in_sample': is_in_sample(row)})
    return pd.DataFrame(rows)


def summarize(group):
    q1, q3 = group['dsc'].quantile([0.25, 0.75])
    return {'n': len(group),
            'dsc_mean': group['dsc'].mean(), 'dsc_sd': group['dsc'].std(ddof=1),
            'dsc_median': group['dsc'].median(), 'dsc_q1': q1, 'dsc_q3': q3,
            'iou_mean': group['iou'].mean(), 'iou_sd': group['iou'].std(ddof=1),
            'n_zero': int((group['dsc'] == 0).sum()),
            'n_fail': int((group['dsc'] < FAILURE_THRESHOLD).sum())}


def summary_frame(df):
    rows = [{'group': 'Overall', **summarize(df)}]
    for cls in CLASS_ORDER:
        rows.append({'group': CLASS_LABEL[cls], **summarize(df[df['severity'] == cls])})
    return pd.DataFrame(rows)


def print_table(summary):
    """Supplementary Table 1 의 본문 행을 그대로 낸다 — 손으로 옮기다 자릿수가 어긋나는 것을 막는다."""
    print('\n=== Supplementary Table 1 ===')
    print('| Group | n | DSC (mean ± SD) | DSC median (IQR) | IoU (mean ± SD) | DSC = 0, n | DSC < 0.5, n |')
    print('| --- | ---: | --- | --- | --- | ---: | ---: |')
    for _, r in summary.iterrows():
        print(f"| {r['group']} | {r['n']} | {r['dsc_mean']:.3f} ± {r['dsc_sd']:.3f} | "
              f"{r['dsc_median']:.3f} ({r['dsc_q1']:.3f}–{r['dsc_q3']:.3f}) | "
              f"{r['iou_mean']:.3f} ± {r['iou_sd']:.3f} | {r['n_zero']} | {r['n_fail']} |")


def print_figure_candidates(df, n_show):
    """Supplementary Figure 1 후보 — severity 별로 자기 그룹 median 에 가까운 순.

    in-sample 케이스는 대표 예시로 쓰면 안 되므로 표시해서 뺀다.
    """
    print('\n=== Supplementary Figure 1 후보 (그룹 median 근접순) ===')
    for cls in CLASS_ORDER:
        group = df[df['severity'] == cls].copy()
        median = group['dsc'].median()
        group['gap'] = (group['dsc'] - median).abs()
        print(f"\n[{CLASS_LABEL[cls]}] n={len(group)} · median DSC {median:.3f}")
        for _, r in group.sort_values(['in_sample', 'gap']).head(n_show).iterrows():
            flag = ' ← in-sample, 제외' if r['in_sample'] else ''
            print(f"  {r['patient_id']}  DSC {r['dsc']:.3f}  IoU {r['iou']:.3f}  "
                  f"GT {r['gt_voxels']:>5} · pred {r['pred_voxels']:>5} 복셀{flag}")


def print_diagnostics(df):
    print(f"\n=== 진단 ===")
    print(f"  DSC 0 {int((df.dsc == 0).sum())}건 · 0.5 미만 {int((df.dsc < FAILURE_THRESHOLD).sum())}건 "
          f"· 0.8 이상 {int((df.dsc >= 0.8).sum())}건")
    print(f"  빈 예측 {int((df.pred_voxels == 0).sum())}건")
    leaked = df[df.in_sample]
    print(f"  out-of-fold 아님 {len(leaked)}건"
          + (f" — {', '.join(f'{r.patient_id}(DSC {r.dsc:.3f})' for r in leaked.itertuples())}" if len(leaked) else ""))
    print(f"\n  실패 케이스 (DSC < {FAILURE_THRESHOLD})")
    for r in df[df.dsc < FAILURE_THRESHOLD].sort_values('dsc').itertuples():
        print(f"    {r.patient_id} ({r.severity}) DSC {r.dsc:.3f} · GT {r.gt_voxels} · pred {r.pred_voxels}")


def main():
    parser = argparse.ArgumentParser(description='test 83 분할 성능 (Supplementary Table 1)')
    parser.add_argument('--pred_dataset_dir', default=PRED_DATASET_DIR)
    parser.add_argument('--gt_dataset_dir', default=GT_DATASET_DIR)
    parser.add_argument('--out_csv', default=None, help='케이스별 DSC·IoU 저장 경로')
    parser.add_argument('--out_summary_csv', default=None, help='그룹별 요약 저장 경로')
    parser.add_argument('--n_figure_candidates', type=int, default=5)
    args = parser.parse_args()

    print(f"예측 : {args.pred_dataset_dir}/labelsVal")
    print(f"정답 : {args.gt_dataset_dir}/labelsVal")

    df = collect(args.pred_dataset_dir, args.gt_dataset_dir).sort_values('patient_id')
    summary = summary_frame(df)
    print_table(summary)
    print_diagnostics(df)
    print_figure_candidates(df, args.n_figure_candidates)

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\n케이스별 저장: {args.out_csv}")
    if args.out_summary_csv:
        summary.to_csv(args.out_summary_csv, index=False)
        print(f"요약 저장: {args.out_summary_csv}")


if __name__ == '__main__':
    main()
