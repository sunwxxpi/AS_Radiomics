"""Dataset004_mix 생성 — Tversky 마스크에서 빈 7건만 다른 출처로 갈아끼운다.

빈 마스크는 PyRadiomics 가 케이스를 통째로 버리게 만들어 paired 설계를 깬다. 배정 fold 가 아닌
가중치를 쓰므로 GT 보유 케이스에는 leakage 가 섞인다 — 어느 케이스가 무엇으로 대체됐는지는
mask_source.csv 에 남는다.
출처는 11건 × 12조합 프로브로 정했고 근거는 OVERRIDES 의 note 에 적었다.
"""

import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ablation_pred_masks import (VerificationError, case_id_of, collect_mask_stats,
                                      image_subdir, label_subdir, report_empty_masks,
                                      verify_output)

GT_DATASET_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset004_gt'
TVERSKY_DATASET_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset004_tversky'
MIX_DATASET_DIR = '/home/psw/AS_Radiomics/data/datasets/Dataset004_mix'
DATASET_ID = '3'
CONFIGURATION = '3d_fullres'
PLANS = 'nnUNetResEncUNetLPlans'
TRAINERS = {'tversky': 'nnUNetTrainerTverskyCE', 'baseline': 'nnUNetTrainer'}

# fold 는 배정 fold 가 아니라 프로브에서 고른 fold 다. gt 는 모델 출력이 아니라 정답 마스크를 쓴다.
OVERRIDES = {
    'KUDH0200': ('tversky', 2, 'Tversky 중 GT Dice 최고 0.633 (배정 fold4 는 확률 0.0001)'),
    'KUDH0291': ('tversky', 1, 'Tversky 중 GT Dice 최고 0.932'),
    'KUDH0694': ('gt', None, 'Tversky·baseline 10개 가중치 전부 GT Dice 0 — 병변을 찾은 모델이 없다'),
    'KUDH0457': ('tversky', 3, 'GT 없음. 비지 않은 3 fold 가 서로 겹치지 않아 최대확률 0.9994 로 선택'),
    'KUDH0491': ('tversky', 3, 'GT 없음. 비지 않은 Tversky fold 가 이것뿐'),
    'KUDH0507': ('tversky', 0, 'GT 없음. tv_f1 과 Dice 0.588 로 겹쳐 2 fold 가 지지 (tv_f4 는 고립)'),
}
# 10개 가중치 전부 전경 확률 0.0000. 406건 중 유일한 조영증강 스캔(HU>130 이 24%, 2위 9.1%)이다.
UNRECOVERABLE = ['KUDH0467']


def predict_cases(rows, trainer, fold, work_dir, gt_dataset_dir, device):
    """같은 (trainer, fold) 를 쓰는 케이스들을 한 번에 추론한다."""
    tag = f'{trainer}_f{fold}'
    input_dir, output_dir = os.path.join(work_dir, tag, 'input'), os.path.join(work_dir, tag, 'output')
    for path in (input_dir, output_dir):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
    for row in rows:
        os.symlink(os.path.join(gt_dataset_dir, image_subdir(row['split']), row['new_img_file']),
                   os.path.join(input_dir, row['new_img_file']))

    cmd = ['nnUNetv2_predict', '-i', input_dir, '-o', output_dir, '-d', DATASET_ID,
           '-c', CONFIGURATION, '-p', PLANS, '-tr', TRAINERS[trainer], '-f', str(fold)]
    env = os.environ.copy()
    if device is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(device)
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
    return output_dir


def build(assignment, gt_dataset_dir, tversky_dataset_dir, mix_dataset_dir, device):
    """기본은 Tversky 마스크 복사, OVERRIDES 에 걸린 케이스만 다른 출처로 덮는다."""
    for sub in ('imagesTr', 'imagesVal', 'labelsTr', 'labelsVal'):
        os.makedirs(os.path.join(mix_dataset_dir, sub), exist_ok=True)
    work_dir = os.path.join(mix_dataset_dir, '_work')

    by_model = {}
    for _, row in assignment[assignment['patient_id'].isin(OVERRIDES)].iterrows():
        source, fold, _ = OVERRIDES[row['patient_id']]
        if source != 'gt':
            by_model.setdefault((source, fold), []).append(row)
    predicted = {(s, f): predict_cases(rows, s, f, work_dir, gt_dataset_dir, device)
                 for (s, f), rows in sorted(by_model.items())}

    records = []
    for _, row in assignment.iterrows():
        img_src = os.path.join(gt_dataset_dir, image_subdir(row['split']), row['new_img_file'])
        img_dst = os.path.join(mix_dataset_dir, image_subdir(row['split']), row['new_img_file'])
        if os.path.exists(img_dst):
            os.remove(img_dst)
        os.link(img_src, img_dst)

        patient_id = row['patient_id']
        if patient_id in OVERRIDES:
            source, fold, note = OVERRIDES[patient_id]
            if source == 'gt':
                mask_src = os.path.join(gt_dataset_dir, label_subdir(row['split']), row['new_label_file'])
            else:
                mask_src = os.path.join(predicted[(source, fold)],
                                        f"{case_id_of(row['new_img_file'])}.nii.gz")
        else:
            source, fold, note = 'tversky', row['fold'], ''
            mask_src = os.path.join(tversky_dataset_dir, label_subdir(row['split']), row['new_label_file'])

        if not os.path.isfile(mask_src):
            raise VerificationError(f"{patient_id}: 마스크 원본이 없다 — {mask_src}")
        shutil.copy2(mask_src, os.path.join(mix_dataset_dir, label_subdir(row['split']),
                                            row['new_label_file']))
        records.append({'patient_id': patient_id, 'severity': row['severity'],
                        'split': row['split'], 'has_gt': row['has_gt'],
                        'assigned_fold': row['fold'], 'mask_source': source,
                        'source_fold': fold, 'is_override': patient_id in OVERRIDES,
                        'note': note})

    sources = pd.DataFrame(records)
    sources.to_csv(os.path.join(mix_dataset_dir, 'mask_source.csv'), index=False)
    print(f"\n수집 완료: 이미지 {len(assignment)}건 하드링크, 마스크 {len(assignment)}건 복사")
    return sources


def verify_overrides(assignment, tversky_dataset_dir, mix_dataset_dir):
    """대체한 케이스만 Tversky 와 다르고 나머지는 바이트 동일한지"""
    changed = []
    for _, row in assignment.iterrows():
        paths = [os.path.join(d, label_subdir(row['split']), row['new_label_file'])
                 for d in (tversky_dataset_dir, mix_dataset_dir)]
        with open(paths[0], 'rb') as f0, open(paths[1], 'rb') as f1:
            if f0.read() != f1.read():
                changed.append(row['patient_id'])

    expected = set(OVERRIDES)
    if set(changed) != expected:
        raise VerificationError(
            f"Tversky 와 달라진 케이스가 대체 목록과 다르다 "
            f"(초과 {sorted(set(changed) - expected)}, 누락 {sorted(expected - set(changed))})")
    print(f"검증 통과 (e): Tversky 대비 달라진 케이스 {len(changed)}건 == 대체 목록")


def verify_override_masks(assignment, mix_dataset_dir):
    """대체한 케이스가 실제로 비지 않았는지 — 대체의 목적 자체"""
    for patient_id in OVERRIDES:
        row = assignment[assignment['patient_id'] == patient_id].iloc[0]
        path = os.path.join(mix_dataset_dir, label_subdir(row['split']), row['new_label_file'])
        n_voxel = int((sitk.GetArrayFromImage(sitk.ReadImage(path)) == 1).sum())
        if n_voxel == 0:
            raise VerificationError(f"{patient_id}: 대체했는데도 마스크가 비었다")
        print(f"  {patient_id}: {n_voxel}복셀 ({OVERRIDES[patient_id][0]}"
              + (f" fold{OVERRIDES[patient_id][1]})" if OVERRIDES[patient_id][1] is not None else ")"))
    print(f"검증 통과 (f): 대체 {len(OVERRIDES)}건 전부 비어 있지 않다")


def main():
    parser = argparse.ArgumentParser(description='Dataset004_mix 생성')
    parser.add_argument('--gt_dataset_dir', default=GT_DATASET_DIR)
    parser.add_argument('--tversky_dataset_dir', default=TVERSKY_DATASET_DIR)
    parser.add_argument('--mix_dataset_dir', default=MIX_DATASET_DIR)
    parser.add_argument('--device', default=None, help='CUDA_VISIBLE_DEVICES 에 넣을 GPU 번호')
    args = parser.parse_args()

    assignment = pd.read_csv(os.path.join(args.tversky_dataset_dir, 'fold_assignment.csv'))
    missing = set(OVERRIDES) - set(assignment['patient_id'])
    if missing:
        raise VerificationError(f"배정표에 없는 대체 대상: {sorted(missing)}")

    print("=== Dataset004_mix 생성 ===")
    print(f"기본 마스크: {args.tversky_dataset_dir}")
    print(f"출력: {args.mix_dataset_dir}")
    print(f"대체 {len(OVERRIDES)}건, 복구 불가 {len(UNRECOVERABLE)}건 {UNRECOVERABLE}")

    sources = build(assignment, args.gt_dataset_dir, args.tversky_dataset_dir,
                    args.mix_dataset_dir, args.device)
    verify_output(assignment, args.gt_dataset_dir, args.mix_dataset_dir)
    verify_overrides(assignment, args.tversky_dataset_dir, args.mix_dataset_dir)
    verify_override_masks(assignment, args.mix_dataset_dir)

    print("\n=== 마스크 출처 ===")
    print(sources.groupby(['mask_source']).size().to_string())
    stats = collect_mask_stats(assignment, args.mix_dataset_dir,
                               os.path.join(args.mix_dataset_dir, 'pred_mask_stats.csv'))
    report_empty_masks(stats)

    print("\n=== 완료 ===")


if __name__ == '__main__':
    main()
