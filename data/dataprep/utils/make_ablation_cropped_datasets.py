"""cropped 2벌 생성 — 팔마다 자기 마스크의 weighted centroid 로 창을 잡아 영상과 마스크를 함께 자른다.

두 팔에 같은 창을 쓰면 crop 영상이 바이트 동일이 되어 deep feature 가 마스크 출처에 반응하지 않는다.
crop 후에도 원본 affine 을 그대로 두므로 (crop_avc.py 와 동일) 영상과 마스크를 다른 창에서 조합해도
PyRadiomics 의 geometry 검사를 통과한다 — 저장물을 원본에서 다시 잘라 대조하는 것이 유일한 그물이다.
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ablation_pred_masks import VerificationError, image_subdir, label_subdir

REPO_ROOT = '/home/psw/AS_Radiomics'
DATASETS_DIR = f'{REPO_ROOT}/data/datasets'
CROP_SIZE = (160, 160, 32)
SUBDIRS = ('imagesTr', 'imagesVal', 'labelsTr', 'labelsVal')
CASE_COLUMNS = ['patient_id', 'severity', 'split', 'has_gt', 'new_img_file', 'new_label_file']
WINDOW_CSV = 'crop_window.csv'

ARMS = {
    'pred': {'source': 'Dataset004_mix_KUDH0467rm', 'dest': 'Dataset004_mix_KUDH0467rm_cropped',
             'cases_csv': 'fold_assignment.csv', 'only_with_gt': False, 'n_case': 405},
    'gt': {'source': 'Dataset004_gt', 'dest': 'Dataset004_gt_cropped',
           'cases_csv': 'Dataset004_info.csv', 'only_with_gt': True, 'n_case': 250},
}
# 5-a 실측. 원본 마스크가 1~7복셀이라 리샘플링 뒤 차원이 무너지는 케이스들이고 전부 train 이다.
EXPECTED_RADIOMICS_FAILURES = {
    'pred': {'KUDH0694'},
    'gt': {'KUDH0662', 'KUDH0694', 'KUDH0713', 'KUDH0817', 'KUDH0875'},
}


def load_cases(arm, source_dir):
    """팔이 자를 케이스 목록. GT 팔은 라벨이 있는 250건뿐이라 이미지도 그만큼만 만든다."""
    frame = pd.read_csv(os.path.join(source_dir, arm['cases_csv']))
    if arm['only_with_gt']:
        frame = frame[frame['has_gt']]
    frame = frame[CASE_COLUMNS].sort_values('patient_id').reset_index(drop=True)

    if len(frame) != arm['n_case']:
        raise VerificationError(f"케이스 {len(frame)}건 (기대 {arm['n_case']}건)")
    missing = [row['patient_id'] for _, row in frame.iterrows()
               if not (os.path.isfile(os.path.join(source_dir, image_subdir(row['split']),
                                                   row['new_img_file']))
                       and os.path.isfile(os.path.join(source_dir, label_subdir(row['split']),
                                                       row['new_label_file'])))]
    if missing:
        raise VerificationError(f"원본에 영상 또는 마스크가 없다: {missing}")
    return frame


def compute_window(mask, image_shape, crop_size):
    """crop_avc.py 와 같은 창 — 마스크 weighted centroid 중심, 이미지 경계에서는 안쪽으로 밀어 넣는다."""
    if mask.any():
        coords = np.where(mask)
        centroid = [float(np.mean(coords[i])) for i in range(3)]
    else:
        centroid = [image_shape[i] / 2 for i in range(3)]

    start = [int(centroid[i] - crop_size[i] // 2) for i in range(3)]
    end = [start[i] + crop_size[i] for i in range(3)]
    adjusted = False
    for i in range(3):
        if start[i] < 0:
            start[i], end[i], adjusted = 0, crop_size[i], True
        elif end[i] > image_shape[i]:
            end[i], start[i], adjusted = image_shape[i], image_shape[i] - crop_size[i], True
        start[i] = max(start[i], 0)
        end[i] = min(end[i], image_shape[i])
    return centroid, start, end, adjusted


def crop_case(row, source_dir, crop_size):
    """영상·마스크를 같은 창으로 자른 결과와 창 정보"""
    img_path = os.path.join(source_dir, image_subdir(row['split']), row['new_img_file'])
    label_path = os.path.join(source_dir, label_subdir(row['split']), row['new_label_file'])
    img_nii, label_nii = nib.load(img_path), nib.load(label_path)
    image, label = img_nii.get_fdata(), label_nii.get_fdata()
    if image.shape != label.shape:
        raise VerificationError(f"{row['patient_id']}: 영상 {image.shape} 마스크 {label.shape}")

    mask = label == 1
    centroid, start, end, adjusted = compute_window(mask, image.shape, crop_size)
    window = tuple(slice(start[i], end[i]) for i in range(3))
    cropped_image, cropped_label = image[window], label[window]

    n_source = int(mask.sum())
    n_cropped = int((cropped_label == 1).sum())
    record = {
        'patient_id': row['patient_id'], 'severity': row['severity'], 'split': row['split'],
        'has_gt': row['has_gt'],
        'new_img_file': row['new_img_file'], 'new_label_file': row['new_label_file'],
        'source_empty': n_source == 0, 'boundary_adjusted': adjusted,
        'n_voxel_source': n_source, 'n_voxel_cropped': n_cropped,
        'voxel_loss_ratio': 0.0 if n_source == 0 else (n_source - n_cropped) / n_source,
        'fully_contained': n_source > 0 and n_cropped == n_source,
    }
    for axis, name in enumerate('xyz'):
        record[f'image_shape_{name}'] = image.shape[axis]
        record[f'centroid_{name}'] = centroid[axis]
        record[f'crop_start_{name}'] = start[axis]
        record[f'crop_end_{name}'] = end[axis]
        record[f'cropped_shape_{name}'] = cropped_image.shape[axis]
    return record, (img_nii, cropped_image), (label_nii, cropped_label)


def build(cases, source_dir, dest_dir, crop_size, dry_run):
    if not dry_run:
        for sub in SUBDIRS:
            os.makedirs(os.path.join(dest_dir, sub), exist_ok=True)

    records = []
    for _, row in tqdm(cases.iterrows(), total=len(cases), desc='crop', unit='건'):
        record, (img_nii, cropped_image), (label_nii, cropped_label) = \
            crop_case(row, source_dir, crop_size)
        records.append(record)
        if dry_run:
            continue
        nib.save(nib.Nifti1Image(cropped_image, img_nii.affine, img_nii.header),
                 os.path.join(dest_dir, image_subdir(row['split']), row['new_img_file']))
        nib.save(nib.Nifti1Image(cropped_label, label_nii.affine, label_nii.header),
                 os.path.join(dest_dir, label_subdir(row['split']), row['new_label_file']))

    windows = pd.DataFrame(records)
    if not dry_run:
        windows.to_csv(os.path.join(dest_dir, WINDOW_CSV), index=False)
        print(f"\ncrop 창 저장: {os.path.join(dest_dir, WINDOW_CSV)} ({len(windows)}행)")
    return windows


def verify_counts(cases, dest_dir):
    expected = {'imagesTr': int((cases['split'] == 'train').sum()),
                'imagesVal': int((cases['split'] == 'val').sum())}
    expected['labelsTr'], expected['labelsVal'] = expected['imagesTr'], expected['imagesVal']
    for sub in SUBDIRS:
        actual = len([f for f in os.listdir(os.path.join(dest_dir, sub)) if f.endswith('.nii.gz')])
        if actual != expected[sub]:
            raise VerificationError(f"{sub}: {actual}개 (기대 {expected[sub]}개)")
    print(f"검증 통과 (a): {expected}")


def matches_saved(path, array):
    """저장할 때 원본 헤더의 정수 dtype 으로 되돌아가므로 (crop_avc.py 와 동일) 되읽은 값이 조금 밀린다.

    허용치는 양자화 한 칸 + NIfTI 헤더가 scl_slope/scl_inter 를 float32 로만 담아서 생기는 오차다.
    창이 한 복셀만 어긋나도 HU 가 수십~수백씩 벌어지므로 이 허용치로 갈린다.
    """
    saved_nii = nib.load(path)
    saved = saved_nii.get_fdata()
    step = abs(getattr(saved_nii.dataobj, 'slope', 1.0) or 1.0)
    inter = abs(getattr(saved_nii.dataobj, 'inter', 0.0) or 0.0)
    tolerance = step + (inter + 1.0) * 1e-6
    return saved.shape == array.shape and np.allclose(saved, array, rtol=1e-6, atol=tolerance)


def verify_same_window(cases, windows, source_dir, dest_dir, crop_size):
    """저장물을 원본에서 다시 잘라 대조 — 영상과 마스크가 같은 창에서 나왔음을 보이는 유일한 검사"""
    recorded = windows.set_index('patient_id')
    for _, row in tqdm(cases.iterrows(), total=len(cases), desc='창 대조', unit='건'):
        want, (_, image), (_, label) = crop_case(row, source_dir, crop_size)
        got = recorded.loc[row['patient_id']]
        for name in 'xyz':
            for field in (f'crop_start_{name}', f'crop_end_{name}'):
                if int(got[field]) != want[field]:
                    raise VerificationError(
                        f"{row['patient_id']}: {field} 기록 {int(got[field])} vs 재계산 {want[field]}")

        for array, sub, filename in ((image, image_subdir(row['split']), row['new_img_file']),
                                     (label, label_subdir(row['split']), row['new_label_file'])):
            if not matches_saved(os.path.join(dest_dir, sub, filename), array):
                raise VerificationError(f"{row['patient_id']}: {filename} 이 창 밖에서 나왔다")
    print(f"검증 통과 (b): {len(cases)}건의 영상·마스크가 기록된 같은 창의 산물")


def verify_masks(cases, dest_dir):
    """빈 마스크와 라벨값 — 빈 마스크는 PyRadiomics 가 케이스를 통째로 버려 paired 설계를 깬다.

    PyRadiomics 와 같은 SimpleITK 로 읽는다. nibabel 로 읽으면 저장 시 붙은 스케일 때문에 1 이
    0.99999995 로 돌아와 라벨값 검사가 헛돈다.
    """
    empty, off_label = [], []
    for _, row in cases.iterrows():
        array = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(dest_dir, label_subdir(row['split']), row['new_label_file'])))
        values = set(np.unique(array).tolist())
        if not values <= {0.0, 1.0}:
            off_label.append((row['patient_id'], sorted(values)))
        if int((array == 1).sum()) == 0:
            empty.append(row['patient_id'])
    if off_label:
        raise VerificationError(f"라벨값이 {{0,1}} 밖: {off_label}")
    if empty:
        raise VerificationError(f"crop 후 빈 마스크: {empty}")
    print(f"검증 통과 (c): 빈 마스크 0건, 라벨값 ⊆ {{0,1}}")


def verify_radiomics(cases, dest_dir, expected_failures):
    """crop 후 PyRadiomics 통과 여부. 개수·바이트 검증으로는 안 걸리는 실패가 여기서만 드러난다."""
    sys.path.insert(0, REPO_ROOT)
    from config import Config
    from trainer.features_extractor import RadiomicsExtractor

    extractor = RadiomicsExtractor(resampled_spacing=Config.RESAMPLED_SPACING,
                                   resample_interpolator=Config.RESAMPLE_INTERPOLATOR)
    failures = {}
    for _, row in tqdm(cases.iterrows(), total=len(cases), desc='radiomics', unit='건'):
        paths = (os.path.join(dest_dir, image_subdir(row['split']), row['new_img_file']),
                 os.path.join(dest_dir, label_subdir(row['split']), row['new_label_file']))
        try:
            extractor.extractor.execute(*paths, label=1)
        except Exception as error:
            failures[row['patient_id']] = (row['split'], f'{type(error).__name__}: {error}')

    for patient_id, (split, message) in sorted(failures.items()):
        print(f"  실패 {patient_id} ({split}): {message}")

    test_failures = sorted(p for p, (split, _) in failures.items() if split == 'val')
    if test_failures:
        raise VerificationError(f"test 케이스가 PyRadiomics 를 통과하지 못했다: {test_failures}")
    if set(failures) != expected_failures:
        print(f"  경고: 실패 목록이 5-a 실측과 다르다 "
              f"(초과 {sorted(set(failures) - expected_failures)}, "
              f"누락 {sorted(expected_failures - set(failures))})")
    print(f"검증 통과 (d): test 전원 통과, train 실패 {len(failures)}건")
    return failures


def report(windows, crop_size):
    print("\n=== crop 결과 ===")
    print(f"  마스크가 창 안에 온전히 담김: {windows['fully_contained'].mean() * 100:.1f}%"
          f" ({int(windows['fully_contained'].sum())}/{len(windows)}건)")
    print(f"  복셀 손실 중앙값 {windows['voxel_loss_ratio'].median() * 100:.2f}%,"
          f" 최대 {windows['voxel_loss_ratio'].max() * 100:.1f}%")
    print(f"  경계 조정: {int(windows['boundary_adjusted'].sum())}건")

    shapes = windows[[f'cropped_shape_{n}' for n in 'xyz']]
    undersized = windows[(shapes != list(crop_size)).any(axis=1)]
    if len(undersized):
        print(f"  원본이 창보다 작아 {crop_size} 미만인 케이스 {len(undersized)}건: "
              + ', '.join(f"{r['patient_id']}"
                          f"({r['cropped_shape_x']},{r['cropped_shape_y']},{r['cropped_shape_z']})"
                          for _, r in undersized.iterrows()))

    worst = windows.nlargest(5, 'voxel_loss_ratio')
    print("  손실 상위 5건: " + ', '.join(
        f"{r['patient_id']} {r['voxel_loss_ratio'] * 100:.1f}%" for _, r in worst.iterrows()))


def process(name, arm, datasets_dir, crop_size, dry_run, force):
    source_dir = os.path.join(datasets_dir, arm['source'])
    dest_dir = os.path.join(datasets_dir, arm['dest'])
    print(f"\n{'=' * 70}\n=== {name} 팔: {arm['source']} → {arm['dest']} ===")

    existing = [sub for sub in SUBDIRS
                if os.path.isdir(os.path.join(dest_dir, sub)) and os.listdir(os.path.join(dest_dir, sub))]
    if existing and not (force or dry_run):
        raise VerificationError(f"{dest_dir} 에 이미 파일이 있다 ({existing}). 덮어쓰려면 --force")

    cases = load_cases(arm, source_dir)
    print(f"케이스 {len(cases)}건 (train {int((cases['split'] == 'train').sum())} / "
          f"val {int((cases['split'] == 'val').sum())}), crop {crop_size}")

    windows = build(cases, source_dir, dest_dir, crop_size, dry_run)
    report(windows, crop_size)
    if dry_run:
        print("\n[dry-run] 파일을 쓰지 않았다")
        return

    verify_counts(cases, dest_dir)
    verify_same_window(cases, windows, source_dir, dest_dir, crop_size)
    verify_masks(cases, dest_dir)
    verify_radiomics(cases, dest_dir, EXPECTED_RADIOMICS_FAILURES[name])


def main():
    parser = argparse.ArgumentParser(description='마스크 출처별 cropped 데이터셋 생성')
    parser.add_argument('--datasets_dir', default=DATASETS_DIR)
    parser.add_argument('--arms', nargs='+', choices=sorted(ARMS), default=sorted(ARMS))
    parser.add_argument('--crop_size', nargs=3, type=int, default=list(CROP_SIZE))
    parser.add_argument('--dry_run', action='store_true', help='창만 계산하고 파일을 쓰지 않는다')
    parser.add_argument('--force', action='store_true', help='기존 출력 디렉토리를 덮어쓴다')
    args = parser.parse_args()

    crop_size = tuple(args.crop_size)
    for name in args.arms:
        process(name, ARMS[name], args.datasets_dir, crop_size, args.dry_run, args.force)

    print("\n=== 완료 ===")


if __name__ == '__main__':
    main()
