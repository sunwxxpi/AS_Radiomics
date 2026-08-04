"""crop 크기 결정을 위한 실측 — 마스크 extent, 필요 박스 크기, 후보별 보존율.

crop 창은 예측 마스크의 weighted centroid 로 잡히므로 (crop_avc.py), 필요한 크기는
"GT 마스크 자체 크기 + centroid 오차" 다. 둘을 따로 보고해 크기로 고칠 수 있는 실패와
중심이 어긋나 크기로는 못 고치는 실패를 가른다.
후보 박스는 복셀이 아니라 mm 로 받는다 — in-plane spacing 이 0.28~0.80 mm 로 흩어져 있어
고정 복셀 창은 케이스마다 물리 크기가 달라진다.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import chi2_contingency
import SimpleITK as sitk

MASK_FILE = re.compile(r'([A-Za-z0-9\.\-]+)_(\d{4,})\.nii\.gz')
DEFAULT_BOXES_MM = [(40, 40), (50, 50), (60, 60), (60, 96), (70, 70),
                    (73.5, 72), (80, 80), (98, 96), (100, 100), (120, 96)]
DEFAULT_VOXELS = [128, 160, 192, 256]
FINE_SPACING_MM = 0.45  # 프로토콜 2군의 경계. 히스토그램 골짜기에서 잡았다.


def index_masks(dirs):
    """마스크 디렉토리들에서 patient_id -> 경로 맵을 만든다"""
    by_patient = {}
    for d in dirs:
        for f in sorted(os.listdir(d)):
            m = MASK_FILE.match(f)
            if not m:
                continue
            pid = m.group(1)
            if pid in by_patient:
                raise ValueError(f"중복 patient_id: {pid} ({by_patient[pid]} vs {d}/{f})")
            by_patient[pid] = os.path.join(d, f)
    return by_patient


def load_mask(path):
    """(마스크 bool 배열, spacing[x,y,z] mm) 를 numpy 인덱스 순서로 반환"""
    img = sitk.ReadImage(path)
    # GetArrayFromImage 는 (z,y,x) 로 준다. spacing 은 (x,y,z) 이므로 축을 맞춘다.
    arr = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))
    return arr == 1, np.array(img.GetSpacing(), dtype=float)


def measure_case(gt_path, pred_path):
    """한 케이스의 마스크 기하 지표. 예측이 비었거나 GT 가 비면 None."""
    pred, spacing = load_mask(pred_path)
    gt, _ = load_mask(gt_path)
    if pred.sum() == 0 or gt.sum() == 0:
        return None

    pc = np.array(np.where(pred))
    gc = np.array(np.where(gt))
    centroid = pc.mean(axis=1)
    gt_centroid = gc.mean(axis=1)

    labeled, n_cc = ndimage.label(pred)
    cc_sizes = ndimage.sum(pred, labeled, range(1, n_cc + 1))
    largest_cc_centroid = pc[:, labeled[pred] == (np.argmax(cc_sizes) + 1)].mean(axis=1)

    row = {
        'pred_n': int(pred.sum()), 'gt_n': int(gt.sum()),
        'n_cc': n_cc, 'largest_cc_frac': cc_sizes.max() / pred.sum(),
        'centroid_err_mm': float(np.linalg.norm((gt_centroid - centroid) * spacing)),
        'centroid_err_cc_mm': float(np.linalg.norm((gt_centroid - largest_cc_centroid) * spacing)),
    }
    for i, axis in enumerate('xyz'):
        row[f'spacing_{axis}'] = spacing[i]
        # GT 자체 반경 — GT centroid 기준이라 중심 오차가 섞이지 않는다
        row[f'gt_radius_{axis}'] = float(np.abs(gc[i] - gt_centroid[i]).max() * spacing[i])
        # 예측 centroid 기준으로 GT 를 100% 담는 데 필요한 박스 변
        row[f'need_{axis}'] = float(2 * np.abs(gc[i] - centroid[i]).max() * spacing[i])

    dev_gt = np.abs(gc - centroid[:, None]) * spacing[:, None]
    dev_pred = np.abs(pc - centroid[:, None]) * spacing[:, None]
    return row, dev_gt, dev_pred


def add_retention(row, dev_gt, dev_pred, boxes_mm):
    """후보 박스별 GT/예측 마스크 복셀 보존율을 row 에 채운다"""
    for in_plane, depth in boxes_mm:
        half = np.array([in_plane / 2, in_plane / 2, depth / 2])
        key = f'{in_plane:g}x{depth:g}'
        row[f'gt_keep_{key}'] = float((dev_gt <= half[:, None]).all(axis=0).mean())
        row[f'pred_keep_{key}'] = float((dev_pred <= half[:, None]).all(axis=0).mean())


def collect(gt_map, pred_map, boxes_mm):
    shared = sorted(set(gt_map) & set(pred_map))
    print(f"GT {len(gt_map)}건 / 예측 {len(pred_map)}건 / 공통 {len(shared)}건")

    rows, skipped = [], []
    for pid in shared:
        measured = measure_case(gt_map[pid], pred_map[pid])
        if measured is None:
            skipped.append(pid)
            continue
        row, dev_gt, dev_pred = measured
        add_retention(row, dev_gt, dev_pred, boxes_mm)
        row['patient_id'] = pid
        rows.append(row)

    if skipped:
        print(f"제외 {len(skipped)}건 (예측 또는 GT 마스크가 빔): {', '.join(skipped)}")
    return pd.DataFrame(rows).set_index('patient_id'), skipped


def collect_cohort(gt_map, pred_map, severity_map):
    """예측 마스크 전건의 spacing·GT 보유·severity. 헤더만 읽는다."""
    rows = []
    for pid, path in sorted(pred_map.items()):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        rows.append({'patient_id': pid, 'spacing_x': reader.GetSpacing()[0],
                     'has_gt': pid in gt_map,
                     'severity': (severity_map or {}).get(pid)})
    return pd.DataFrame(rows).set_index('patient_id')


def report_cohort(cohort):
    """고정 복셀 창의 FOV 산포와, 그 배율이 라벨·GT 보유와 엮이는지."""
    print(f"\n=== 고정 복셀 창의 물리 크기 산포 (in-plane), 예측 전건 n={len(cohort)} ===")
    for n_vox in DEFAULT_VOXELS:
        fov = cohort.spacing_x * n_vox
        outside = (np.abs(fov / fov.median() - 1) > .2).mean() * 100
        print(f"  {n_vox:4d} 복셀: {fov.min():.0f}~{fov.max():.0f} mm "
              f"(중앙 {fov.median():.0f}), 중앙 대비 ±20% 밖 {outside:.0f}%")

    fine = (cohort.spacing_x < FINE_SPACING_MM).rename('fine_spacing')
    for column, label in [('severity', 'severity'), ('has_gt', 'GT 보유')]:
        sub = cohort.dropna(subset=[column])
        if sub[column].nunique() < 2:
            continue
        table = pd.crosstab(fine.loc[sub.index], sub[column])
        chi2, p, _, _ = chi2_contingency(table)
        print(f"\n=== spacing 군({FINE_SPACING_MM} mm 기준) × {label} (n={len(sub)}) ===")
        print(table)
        print((table.div(table.sum(axis=1), axis=0) * 100).round(1))
        print(f"  chi2={chi2:.2f}  p={p:.4g}")
    print("  유의하면 고정 복셀 창의 FOV 배율이 라벨 대리변수가 된다 — mm 기준 crop 이 필요하다.")


def report(df, boxes_mm):
    pd.set_option('display.width', 220)
    q = [.5, .9, .95, .99, 1.0]

    print(f"\n=== GT 마스크 자체 반경 (mm) — 순수 해부학적 요구치, n={len(df)} ===")
    print(df[[f'gt_radius_{a}' for a in 'xyz']].quantile(q).round(1))
    box = [df[f'gt_radius_{a}'].max() * 2 for a in 'xyz']
    print(f"  → GT 만 담으면 되면 {box[0]:.0f}x{box[1]:.0f}x{box[2]:.0f} mm 에 전건 포함")

    print("\n=== 예측 centroid 기준 GT 를 100% 담는 박스 변 (mm) ===")
    need = pd.DataFrame({'in_plane': df[['need_x', 'need_y']].max(axis=1), 'z': df['need_z']})
    print(need.quantile(q).round(1))

    print("\n=== 예측 centroid 의 GT centroid 대비 오차 (mm) ===")
    print(df[['centroid_err_mm', 'centroid_err_cc_mm']].quantile(q).round(2))
    print("  centroid_err_cc_mm 은 최대 연결성분 centroid 를 중심으로 쓸 때의 값")

    print("\n=== 예측 마스크 연결성분 ===")
    print(df[['n_cc', 'largest_cc_frac']].describe(percentiles=[.05, .5, .95]).round(3))

    print("\n=== 후보 박스별 보존율 ===")
    print(f"{'box (mm)':>16} | {'GT 전건포함':>10} {'GT 5%분위':>10} | "
          f"{'예측 전건포함':>12} | {'미포함 건수':>10} {'그중 보존율 중앙':>16}")
    for in_plane, depth in boxes_mm:
        key = f'{in_plane:g}x{depth:g}'
        gt_keep, pred_keep = df[f'gt_keep_{key}'], df[f'pred_keep_{key}']
        partial = gt_keep[gt_keep < 0.999]
        median_partial = f"{partial.median() * 100:.1f}%" if len(partial) else "-"
        print(f"{in_plane:6g}x{in_plane:<4g}x{depth:<4g} | {(gt_keep >= 0.999).mean() * 100:9.1f}% "
              f"{gt_keep.quantile(.05) * 100:9.1f}% | {(pred_keep >= 0.999).mean() * 100:11.1f}% | "
              f"{len(partial):9d} {median_partial:>16}")
    print("  '미포함 건수' 의 보존율 중앙값이 낮으면 크기가 모자란 게 아니라 창이 딴 데 있는 것이다.")


def main():
    parser = argparse.ArgumentParser(description='crop 크기 결정용 마스크 기하 실측')
    parser.add_argument('--gt_dir', default='/home/psw/AS_Radiomics/data/datasets/Dataset001_KMU_Cardiac_AVC_TRAIN_ONLY/labelsTr')
    parser.add_argument('--pred_dir', nargs='+',
                        default=['/home/psw/AS_Radiomics/data/datasets/Dataset003_total/labelsTr',
                                 '/home/psw/AS_Radiomics/data/datasets/Dataset003_total/labelsVal'])
    parser.add_argument('--out_csv', default=None, help='케이스별 측정치 저장 경로')
    parser.add_argument('--box_mm', nargs='+', default=None,
                        metavar='IN_PLANE:DEPTH', help='후보 박스 (예: 70:70 73.5:72)')
    parser.add_argument('--severity_csv', default='/home/psw/AS_Radiomics/data/AS_CRF.csv',
                        help='severity 연관 검정용. 없으면 해당 절 생략')
    args = parser.parse_args()

    boxes_mm = DEFAULT_BOXES_MM
    if args.box_mm:
        boxes_mm = [tuple(float(v) for v in str(b).split(':')) for b in args.box_mm]

    severity_map = None
    if args.severity_csv and os.path.exists(args.severity_csv):
        # organize_total_stratified_dataset 는 저장소 루트에 있고 이 파일은 utils/ 아래다
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from organize_total_stratified_dataset import load_severity_mapping
        severity_map = load_severity_mapping(args.severity_csv, mode='multi')

    gt_map = index_masks([args.gt_dir])
    pred_map = index_masks(args.pred_dir)

    df, _ = collect(gt_map, pred_map, boxes_mm)
    report(df, boxes_mm)
    # 코호트 절은 예측 전건이 필요하다 — GT 보유 여부가 공통 부분집합에서는 항상 참이다
    report_cohort(collect_cohort(gt_map, pred_map, severity_map))

    if args.out_csv:
        df.to_csv(args.out_csv)
        print(f"\n케이스별 측정치 저장: {args.out_csv}")


if __name__ == '__main__':
    main()
