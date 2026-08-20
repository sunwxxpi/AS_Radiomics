"""Dataset004_gt 생성 — test 83건을 전원 GT 마스크 보유 환자로 재분할한다.

파일 번호는 406건 기준으로 여기서 한 번 확정되고, 이후 예측 마스크 데이터셋(_pred)도 같은 번호를 쓴다.
GT 마스크는 파일명이 아니라 patient_id 로 조인한다 — 4자리 시퀀스 번호가 데이터셋별 자체 일련번호라
파일명으로 맞추면 406건 중 3건만 붙고 에러 없이 끝난다.
"""

import argparse
import math
import os
import shutil
from collections import Counter, defaultdict

import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

from organize_total_stratified_dataset import (
    extract_patient_id,
    generate_nnunet_filename,
    load_severity_mapping,
)

TOTAL_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset001_KMU_Cardiac_AVC_TOTAL'
GT_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset001_KMU_Cardiac_AVC_TRAIN_ONLY'
CLASS_ORDER = ['normal', 'nonsevere', 'severe']
GEOMETRY_TOLERANCE = 1e-5


class VerificationError(Exception):
    """분할·복사 검증 실패. 잘못된 데이터셋이 남지 않도록 즉시 중단시킨다."""


def collect_cases(total_dir, gt_dir, severity_map):
    """406건 이미지에 severity 와 GT 마스크 경로(있으면)를 붙여 수집"""
    image_dir = os.path.join(total_dir, 'imagesVal')
    label_dir = os.path.join(gt_dir, 'labelsTr')

    gt_label_by_patient = {}
    for label_file in sorted(os.listdir(label_dir)):
        if not label_file.endswith('.nii.gz'):
            continue
        patient_id = extract_patient_id(label_file)
        if patient_id in gt_label_by_patient:
            raise VerificationError(f"GT 라벨에 중복 patient_id: {patient_id}")
        gt_label_by_patient[patient_id] = os.path.join(label_dir, label_file)

    cases = []
    seen = set()
    missing_severity = []
    for img_file in sorted(os.listdir(image_dir)):
        if not img_file.endswith('.nii.gz'):
            continue
        patient_id = extract_patient_id(img_file)
        if patient_id in seen:
            raise VerificationError(f"이미지에 중복 patient_id: {patient_id}")
        seen.add(patient_id)

        if patient_id not in severity_map:
            missing_severity.append(patient_id)
            continue

        cases.append({
            'patient_id': patient_id,
            'severity': severity_map[patient_id],
            'img_file': img_file,
            'img_path': os.path.join(image_dir, img_file),
            'gt_label_path': gt_label_by_patient.get(patient_id),
        })

    if missing_severity:
        raise VerificationError(
            f"CRF 에 severity 가 없는 환자 {len(missing_severity)}명: {sorted(missing_severity)[:10]}")

    unmatched_gt = sorted(set(gt_label_by_patient) - seen)
    if unmatched_gt:
        raise VerificationError(f"406건에 없는 GT patient_id {len(unmatched_gt)}명: {unmatched_gt[:10]}")

    print(f"\n수집 완료: 이미지 {len(cases)}건, 그중 GT 보유 "
          f"{sum(1 for c in cases if c['gt_label_path'])}건")
    return cases


def select_test_patients(cases, random_state):
    """클래스별 test 인원을 GT 보유 환자 안에서만 추출

    n_test = ceil(0.2 x 클래스 전체) 로 sklearn train_test_split(test_size=0.2) 의 반올림과 맞춘다.
    """
    by_class = defaultdict(list)
    gt_pool = defaultdict(list)
    for case in cases:
        by_class[case['severity']].append(case['patient_id'])
        if case['gt_label_path']:
            gt_pool[case['severity']].append(case['patient_id'])

    test_patients = set()
    print("\n=== 클래스별 test 추출 ===")
    for severity in CLASS_ORDER:
        total = len(by_class[severity])
        pool = sorted(gt_pool[severity])
        n_test = math.ceil(0.2 * total)

        if n_test > len(pool):
            raise VerificationError(
                f"{severity}: test {n_test}명이 필요한데 GT 보유가 {len(pool)}명뿐")
        if n_test == len(pool):
            selected = list(pool)
        else:
            _, selected = train_test_split(pool, test_size=n_test, random_state=random_state)

        test_patients.update(selected)
        print(f"  {severity:<10} 전체 {total:>3} / GT pool {len(pool):>3} -> test {n_test:>2}")

    return test_patients


def assign_filenames(cases, test_patients):
    """split 별로 patient_id 정렬 후 0001 부터 번호를 다시 매긴다"""
    train_rows = sorted([c for c in cases if c['patient_id'] not in test_patients],
                        key=lambda c: c['patient_id'])
    val_rows = sorted([c for c in cases if c['patient_id'] in test_patients],
                      key=lambda c: c['patient_id'])

    for rows, split in ((train_rows, 'train'), (val_rows, 'val')):
        for i, case in enumerate(rows):
            new_img_file = generate_nnunet_filename(case['patient_id'], i + 1)
            case['split'] = split
            case['new_img_file'] = new_img_file
            case['new_label_file'] = new_img_file.replace('_0000.nii.gz', '.nii.gz')

    return train_rows, val_rows


def verify_split(train_rows, val_rows):
    """복사 전 검증 — 개수·클래스 분포·GT 보유·파일명 규칙"""
    expected_test = {'normal': 18, 'nonsevere': 23, 'severe': 42}
    expected_train = {'normal': 69, 'nonsevere': 89, 'severe': 165}

    test_dist = Counter(c['severity'] for c in val_rows)
    train_dist = Counter(c['severity'] for c in train_rows)
    if dict(test_dist) != expected_test:
        raise VerificationError(f"test 클래스 분포 불일치: {dict(test_dist)} != {expected_test}")
    if dict(train_dist) != expected_train:
        raise VerificationError(f"train 클래스 분포 불일치: {dict(train_dist)} != {expected_train}")

    no_gt = [c['patient_id'] for c in val_rows if not c['gt_label_path']]
    if no_gt:
        raise VerificationError(f"test 에 GT 미보유 {len(no_gt)}명: {no_gt[:10]}")

    all_rows = train_rows + val_rows
    n_gt = sum(1 for c in all_rows if c['gt_label_path'])
    if n_gt != 250:
        raise VerificationError(f"GT 보유 합계 {n_gt} != 250")

    for rows, name in ((train_rows, 'imagesTr'), (val_rows, 'imagesVal')):
        names = [c['new_img_file'] for c in rows]
        if len(set(names)) != len(names):
            raise VerificationError(f"{name} 에 중복 파일명")
        expected = [generate_nnunet_filename(c['patient_id'], i + 1) for i, c in enumerate(rows)]
        if names != expected:
            raise VerificationError(f"{name} 번호가 정렬 순서와 어긋남")

    print("\n검증 통과: test 83 전원 GT 보유, 분포 18/23/42 · 69/89/165")


def verify_geometry(cases):
    """GT 마스크와 이미지의 헤더가 같은지 확인 — patient_id 조인이 틀리면 여기서 걸린다"""
    checked = 0
    for case in cases:
        if not case['gt_label_path']:
            continue
        reader = sitk.ImageFileReader()
        reader.SetFileName(case['img_path'])
        reader.ReadImageInformation()
        img_meta = (reader.GetSize(), reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection())

        reader.SetFileName(case['gt_label_path'])
        reader.ReadImageInformation()
        lbl_meta = (reader.GetSize(), reader.GetSpacing(), reader.GetOrigin(), reader.GetDirection())

        if img_meta[0] != lbl_meta[0]:
            raise VerificationError(
                f"{case['patient_id']}: size 불일치 {img_meta[0]} vs {lbl_meta[0]}")
        for idx, field in enumerate(('spacing', 'origin', 'direction'), start=1):
            if any(abs(a - b) > GEOMETRY_TOLERANCE for a, b in zip(img_meta[idx], lbl_meta[idx])):
                raise VerificationError(
                    f"{case['patient_id']}: {field} 불일치 {img_meta[idx]} vs {lbl_meta[idx]}")
        checked += 1

    print(f"검증 통과: GT {checked}건 이미지-마스크 헤더 일치")


def materialize(train_rows, val_rows, output_dir):
    """이미지 406건 전량 + GT 라벨 250건을 새 파일명으로 복사"""
    for sub in ('imagesTr', 'imagesVal', 'labelsTr', 'labelsVal'):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    for rows, img_sub, lbl_sub in ((train_rows, 'imagesTr', 'labelsTr'),
                                   (val_rows, 'imagesVal', 'labelsVal')):
        print(f"\n{img_sub} 복사: {len(rows)}건")
        for case in rows:
            shutil.copy2(case['img_path'], os.path.join(output_dir, img_sub, case['new_img_file']))
            if case['gt_label_path']:
                shutil.copy2(case['gt_label_path'],
                             os.path.join(output_dir, lbl_sub, case['new_label_file']))
        n_lbl = sum(1 for c in rows if c['gt_label_path'])
        print(f"{lbl_sub} 복사: {n_lbl}건")


def verify_output(train_rows, val_rows, output_dir):
    """복사 후 검증 — 파일 개수와 바이트 크기"""
    expected_counts = {
        'imagesTr': len(train_rows),
        'imagesVal': len(val_rows),
        'labelsTr': sum(1 for c in train_rows if c['gt_label_path']),
        'labelsVal': sum(1 for c in val_rows if c['gt_label_path']),
    }
    for sub, expected in expected_counts.items():
        actual = len([f for f in os.listdir(os.path.join(output_dir, sub)) if f.endswith('.nii.gz')])
        if actual != expected:
            raise VerificationError(f"{sub}: {actual}개 (기대 {expected}개)")

    for rows, img_sub, lbl_sub in ((train_rows, 'imagesTr', 'labelsTr'),
                                   (val_rows, 'imagesVal', 'labelsVal')):
        for case in rows:
            dst_img = os.path.join(output_dir, img_sub, case['new_img_file'])
            if os.path.getsize(dst_img) != os.path.getsize(case['img_path']):
                raise VerificationError(f"{case['new_img_file']}: 이미지 크기 불일치")
            if case['gt_label_path']:
                dst_lbl = os.path.join(output_dir, lbl_sub, case['new_label_file'])
                if os.path.getsize(dst_lbl) != os.path.getsize(case['gt_label_path']):
                    raise VerificationError(f"{case['new_label_file']}: 라벨 크기 불일치")

    print(f"\n검증 통과: {expected_counts}")


def write_info_csv(train_rows, val_rows, output_dir, csv_name):
    """번호 매핑을 CSV 로 저장 — _pred 라벨을 채울 때 같은 매핑을 쓴다"""
    records = []
    for case in train_rows + val_rows:
        records.append({
            'patient_id': case['patient_id'],
            'original_img_file': case['img_file'],
            'original_label_file': (os.path.basename(case['gt_label_path'])
                                    if case['gt_label_path'] else ''),
            'new_img_file': case['new_img_file'],
            'new_label_file': case['new_label_file'],
            'severity': case['severity'],
            'split': case['split'],
            'has_gt': bool(case['gt_label_path']),
        })

    csv_path = os.path.join(output_dir, csv_name)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"분할 정보 저장: {csv_path} ({len(records)}행)")


def print_summary(train_rows, val_rows):
    print("\n=== 최종 분할 ===")
    header = f"{'':<8}" + ''.join(f"{c:>12}" for c in CLASS_ORDER) + f"{'계':>8}{'GT':>6}"
    print(header)
    for rows, name in ((train_rows, 'train'), (val_rows, 'test')):
        dist = Counter(c['severity'] for c in rows)
        n_gt = sum(1 for c in rows if c['gt_label_path'])
        print(f"{name:<8}" + ''.join(f"{dist[c]:>12}" for c in CLASS_ORDER)
              + f"{len(rows):>8}{n_gt:>6}")

    print("\n파일명 예시")
    for rows, sub in ((train_rows, 'imagesTr'), (val_rows, 'imagesVal')):
        print(f"  {sub}: {rows[0]['new_img_file']} ... {rows[-1]['new_img_file']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/home/psw/AS_Radiomics/data/AS_CRF.csv')
    parser.add_argument('--output_dir', default='/home/psw/AS_Radiomics/data/datasets/Dataset004_gt')
    parser.add_argument('--info_csv_name', default='Dataset004_info.csv')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true',
                        help='분할과 파일명만 계산하고 복사는 하지 않는다')
    args = parser.parse_args()

    print("=== Dataset004_gt 생성 ===")
    print(f"이미지 소스: {TOTAL_DIR}/imagesVal")
    print(f"GT 마스크 소스: {GT_DIR}/labelsTr")
    print(f"출력: {args.output_dir}")
    print(f"랜덤 시드: {args.random_state}{'  [DRY RUN]' if args.dry_run else ''}")

    severity_map = load_severity_mapping(args.csv_path, mode='multi')
    cases = collect_cases(TOTAL_DIR, GT_DIR, severity_map)

    test_patients = select_test_patients(cases, args.random_state)
    train_rows, val_rows = assign_filenames(cases, test_patients)

    verify_split(train_rows, val_rows)
    verify_geometry(cases)
    print_summary(train_rows, val_rows)

    if args.dry_run:
        print("\nDRY RUN — 복사하지 않고 종료")
        return

    materialize(train_rows, val_rows, args.output_dir)
    verify_output(train_rows, val_rows, args.output_dir)
    write_info_csv(train_rows, val_rows, args.output_dir, args.info_csv_name)

    print("\n=== 완료 ===")


if __name__ == '__main__':
    main()
