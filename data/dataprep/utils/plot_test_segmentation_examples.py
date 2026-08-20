"""Supplementary Figure 1 — test 대표 케이스의 분할 예시를 그린다.

케이스는 `eval_test_segmentation.py` 가 고른 severity 별 median 근접 케이스이고 DSC 도 그 산출물에서 읽는다.
표시는 원본 해상도 마스크로 하되 판막 주변 60 mm 만 잘라 보여준다 — 파이프라인의 `(160,160,32)` crop 과는 무관한 표시용 창이다.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

PRED_DATASET_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset004_mix_KUDH0467rm'
GT_DATASET_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset004_gt'
ROW_ORDER = [('normal', 'Normal'), ('nonsevere', 'Non-severe AS'), ('severe', 'Severe AS')]
DEFAULT_CASES = {'normal': 'KUDH0563', 'nonsevere': 'KUDH0084', 'severe': 'KUDH0086'}

# 종격동 창 — 석회는 포화되지만 판막 주변 연부조직 경계가 남아 병변 위치를 읽을 수 있다.
WINDOW_LEVEL, WINDOW_WIDTH = 40.0, 400.0
VIEW_MM = 60.0
SCALE_BAR_MM = 10.0
REF_COLOR, PRED_COLOR = '#00b0f0', '#ff7f0e'
OVERLAP_COLORS = {'tp': '#2ca02c', 'fn': '#00b0f0', 'fp': '#ff7f0e'}


class VerificationError(Exception):
    """케이스나 마스크가 없으면 빈 그림을 내는 대신 중단한다."""


def load_case(pred_dataset_dir, gt_dataset_dir, row):
    image = nib.load(os.path.join(gt_dataset_dir, 'imagesVal', row['new_img_file']))
    gt = nib.load(os.path.join(gt_dataset_dir, 'labelsVal', row['new_label_file']))
    pred = nib.load(os.path.join(pred_dataset_dir, 'labelsVal', row['new_label_file']))
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    gt_mask, pred_mask = np.asanyarray(gt.dataobj) > 0, np.asanyarray(pred.dataobj) > 0
    if not (volume.shape == gt_mask.shape == pred_mask.shape):
        raise VerificationError(f"{row['patient_id']}: shape 불일치")
    if nib.aff2axcodes(image.affine) != ('L', 'P', 'S'):
        raise VerificationError(f"{row['patient_id']}: LPS 가 아니다 — 표시 방향을 다시 잡아야 한다")
    return volume, gt_mask, pred_mask, image.header.get_zooms()[:2]


def display_slice(volume, gt_mask, pred_mask, spacing):
    """GT 면적이 가장 큰 axial slice 를 판막 중심 60 mm 로 자른다. 반환은 (행=A→P, 열=R→L) 이다."""
    z = int(np.argmax(gt_mask.sum(axis=(0, 1))))
    both = gt_mask[:, :, z] | pred_mask[:, :, z]
    if not both.any():
        raise VerificationError('선택한 slice 에 마스크가 없다')
    rows, cols = np.where(both)
    center = (int(round(rows.mean())), int(round(cols.mean())))
    half = (int(round(VIEW_MM / 2 / spacing[0])), int(round(VIEW_MM / 2 / spacing[1])))
    limits = [(max(0, c - h), min(s, c + h)) for c, h, s in zip(center, half, both.shape)]
    (x0, x1), (y0, y1) = limits
    crop = lambda arr: arr[x0:x1, y0:y1, z].T
    return crop(volume), crop(gt_mask), crop(pred_mask)


def draw_mask(ax, mask, color, alpha=0.35):
    overlay = np.zeros(mask.shape + (4,))
    overlay[mask] = matplotlib.colors.to_rgba(color, alpha)
    ax.imshow(overlay, interpolation='nearest')
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.2)


def draw_scale_bar(ax, width_px, spacing_x):
    length = SCALE_BAR_MM / spacing_x
    x0, y0 = width_px * 0.06, ax.get_ylim()[0] * 0.94
    ax.add_patch(Rectangle((x0, y0), length, max(1.0, width_px * 0.012), color='white'))
    ax.text(x0 + length / 2, y0 - width_px * 0.03, f'{SCALE_BAR_MM:g} mm',
            color='white', fontsize=6, ha='center', va='bottom')


def render(cases, out_paths):
    vmin, vmax = WINDOW_LEVEL - WINDOW_WIDTH / 2, WINDOW_LEVEL + WINDOW_WIDTH / 2
    titles = ['CT', 'Reference', 'Prediction', 'Overlap']
    fig, axes = plt.subplots(len(cases), 4, figsize=(8.0, 2.05 * len(cases)))
    axes = np.atleast_2d(axes)

    for r, case in enumerate(cases):
        image, gt_mask, pred_mask = case['slice']
        for c, ax in enumerate(axes[r]):
            ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, interpolation='bilinear')
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 1:
                draw_mask(ax, gt_mask, REF_COLOR)
            elif c == 2:
                draw_mask(ax, pred_mask, PRED_COLOR)
            elif c == 3:
                draw_mask(ax, gt_mask & pred_mask, OVERLAP_COLORS['tp'])
                draw_mask(ax, gt_mask & ~pred_mask, OVERLAP_COLORS['fn'])
                draw_mask(ax, pred_mask & ~gt_mask, OVERLAP_COLORS['fp'])
            if r == 0:
                ax.set_title(titles[c], fontsize=9, pad=4)
        draw_scale_bar(axes[r][0], image.shape[1], case['spacing'][0])
        axes[r][0].set_ylabel(f"{case['label']}\n{case['patient_id']}  DSC {case['dsc']:.3f}",
                              fontsize=8, labelpad=6)

    handles = [Patch(facecolor=REF_COLOR, alpha=0.6, edgecolor=REF_COLOR, label='Reference'),
               Patch(facecolor=PRED_COLOR, alpha=0.6, edgecolor=PRED_COLOR, label='Prediction'),
               Patch(facecolor=OVERLAP_COLORS['tp'], alpha=0.6, edgecolor=OVERLAP_COLORS['tp'],
                     label='Overlap (both)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.94, bottom=0.07, wspace=0.02, hspace=0.04)

    for path in out_paths:
        fig.savefig(path, dpi=300)
        print(f'저장: {path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Supplementary Figure 1 (대표 분할 예시)')
    parser.add_argument('--pred_dataset_dir', default=PRED_DATASET_DIR)
    parser.add_argument('--gt_dataset_dir', default=GT_DATASET_DIR)
    parser.add_argument('--performance_csv', default=None,
                        help='기본값은 pred_dataset_dir 의 test_seg_performance.csv')
    parser.add_argument('--out', nargs='+', required=True, help='저장 경로 (확장자로 형식 결정)')
    parser.add_argument('--case', action='append', default=[], metavar='SEVERITY=PATIENT_ID',
                        help='기본 케이스를 바꾼다')
    args = parser.parse_args()

    chosen = dict(DEFAULT_CASES)
    for item in args.case:
        severity, patient_id = item.split('=', 1)
        chosen[severity] = patient_id

    performance_csv = args.performance_csv or os.path.join(args.pred_dataset_dir, 'test_seg_performance.csv')
    performance = pd.read_csv(performance_csv).set_index('patient_id')
    assignment = pd.read_csv(os.path.join(args.pred_dataset_dir, 'fold_assignment.csv')).set_index('patient_id')

    cases = []
    for severity, label in ROW_ORDER:
        patient_id = chosen[severity]
        if patient_id not in performance.index:
            raise VerificationError(f'{patient_id}: test 83 에 없다')
        record = performance.loc[patient_id]
        if record['severity'] != severity:
            raise VerificationError(f"{patient_id}: severity 가 {record['severity']} 라 {severity} 행에 못 쓴다")
        if record['in_sample']:
            raise VerificationError(f'{patient_id}: out-of-fold 가 아니라 대표 예시로 못 쓴다')
        volume, gt_mask, pred_mask, spacing = load_case(
            args.pred_dataset_dir, args.gt_dataset_dir, assignment.loc[patient_id])
        cases.append({'patient_id': patient_id, 'label': label, 'dsc': float(record['dsc']),
                      'spacing': spacing, 'slice': display_slice(volume, gt_mask, pred_mask, spacing)})
        print(f"{label:>14}: {patient_id} DSC {record['dsc']:.3f} · spacing {spacing[0]:.4f} mm")

    render(cases, args.out)


if __name__ == '__main__':
    main()
