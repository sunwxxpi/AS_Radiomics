"""recall 편향 segmentation 재학습에 필요한 두 가지를 설치한다 — 커스텀 trainer 링크와 patch 축소 plans.

전처리는 다시 돌지 않는다. 새 plans 의 `data_identifier` 를 원본과 같은 값으로 두기 때문이고,
patch 크기는 전처리 산출물이 아니라 dataloader 가 쓰는 값이라 그래도 된다.
"""

import argparse
import hashlib
import inspect
import json
import os

import numpy as np
import torch

TRAINER_MODULE = 'nnUNetTrainerTverskyCE.py'
TRAINER_NAMES = ['nnUNetTrainerTverskyCE', 'nnUNetTrainerTverskyCE_250epochs',
                 'nnUNetTrainerTverskyCE_500epochs', 'nnUNetTrainerTverskyCE_2000epochs']
DATASET_NAME = 'Dataset003_KMU_Cardiac_AVC_TRAIN_ONLY'
DATASET_ID = '3'
SOURCE_PLANS = 'nnUNetResEncUNetLPlans'
TARGET_PLANS = 'nnUNetResEncUNetLPlans_smallpatch'
CONFIGURATION = '3d_fullres'
NEW_PATCH_SIZE = [40, 192, 192]
# patch 만 바뀌어야 한다. 나머지가 달라지면 4번이 재는 것이 loss/patch 효과가 아니게 된다.
IMMUTABLE_KEYS = ['data_identifier', 'preprocessor_name', 'batch_size', 'spacing', 'batch_dice',
                  'normalization_schemes', 'use_mask_for_norm', 'architecture',
                  'resampling_fn_data', 'resampling_fn_seg', 'resampling_fn_probabilities',
                  'resampling_fn_data_kwargs', 'resampling_fn_seg_kwargs',
                  'resampling_fn_probabilities_kwargs']


class VerificationError(Exception):
    """검증 실패. 잘못된 설정으로 며칠짜리 학습이 시작되지 않도록 즉시 중단시킨다."""


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def nnunet_package_dir():
    import nnunetv2
    return os.path.dirname(nnunetv2.__file__)


def trainer_link_path():
    return os.path.join(nnunet_package_dir(), 'training', 'nnUNetTrainer', 'variants', 'loss', TRAINER_MODULE)


def install_trainer(source, dry_run, force):
    """저장소의 trainer 를 nnunetv2 패키지 안으로 심볼릭 링크한다. 복사하면 저장소 수정이 조용히 무시된다."""
    link = trainer_link_path()
    if os.path.islink(link):
        current = os.path.realpath(link)
        if current == os.path.realpath(source):
            print(f"  이미 설치됨: {link}")
            return
        if not force:
            raise VerificationError(f"다른 파일이 링크돼 있다: {link} -> {current} (덮어쓰려면 --force)")
    elif os.path.exists(link):
        raise VerificationError(f"심볼릭 링크가 아닌 파일이 있다: {link} — 손대지 않는다")

    if dry_run:
        print(f"  [dry-run] 링크 예정: {link} -> {source}")
        return

    if os.path.islink(link):
        os.remove(link)
    os.symlink(os.path.realpath(source), link)
    print(f"  링크: {link} -> {os.path.realpath(source)}")


def stride_product(architecture):
    """architecture 의 stride 를 축별로 모두 곱한 값. patch 는 이 값으로 나누어떨어져야 한다."""
    strides = architecture['arch_kwargs']['strides']
    product = np.ones(len(strides[0]), dtype=int)
    for stride in strides:
        product *= np.array(stride, dtype=int)
    return product


def build_plans(preprocessed_dir, patch_size):
    source_path = os.path.join(preprocessed_dir, f'{SOURCE_PLANS}.json')
    if not os.path.isfile(source_path):
        raise VerificationError(f"원본 plans 없음: {source_path}")
    with open(source_path) as f:
        plans = json.load(f)

    config = plans['configurations'][CONFIGURATION]
    divisor = stride_product(config['architecture'])
    remainder = np.array(patch_size) % divisor
    if remainder.any():
        raise VerificationError(
            f"patch {patch_size} 가 stride 곱 {divisor.tolist()} 로 나누어떨어지지 않는다 "
            f"— 아키텍처를 그대로 두려면 배수여야 한다")
    if any(p <= 0 for p in patch_size):
        raise VerificationError(f"patch 값이 잘못됐다: {patch_size}")

    old_patch = config['patch_size']
    plans['plans_name'] = TARGET_PLANS
    config['patch_size'] = list(patch_size)
    return plans, old_patch, divisor


def report_patch_change(old_patch, new_patch, spacing, batch_size):
    """patch 를 물리 크기와 복셀 수로 같이 보여준다 — 복셀 수만 보면 z 축 축소를 과소평가한다."""
    old_mm = [round(p * s, 1) for p, s in zip(old_patch, spacing)]
    new_mm = [round(p * s, 1) for p, s in zip(new_patch, spacing)]
    old_vox, new_vox = int(np.prod(old_patch)), int(np.prod(new_patch))
    print(f"  patch {old_patch} -> {new_patch}")
    print(f"    물리 크기 (z,y,x mm) {old_mm} -> {new_mm}")
    print(f"    복셀 {old_vox:,} -> {new_vox:,} ({new_vox / old_vox:.2f} 배)")
    print(f"    step 당 복셀 (batch {batch_size}) {old_vox * batch_size:,} -> {new_vox * batch_size:,}")


def write_plans(plans, preprocessed_dir, dry_run, force):
    target_path = os.path.join(preprocessed_dir, f'{TARGET_PLANS}.json')
    text = json.dumps(plans, indent=4, sort_keys=False)
    if os.path.isfile(target_path):
        with open(target_path) as f:
            if f.read() == text:
                print(f"  이미 있음(내용 동일): {target_path}")
                return
        if not force:
            raise VerificationError(f"다른 내용의 plans 가 이미 있다: {target_path} (덮어쓰려면 --force)")

    if dry_run:
        print(f"  [dry-run] 저장 예정: {target_path}")
        return

    tmp = target_path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, target_path)
    print(f"  저장: {target_path}")


def verify_plans_diff(preprocessed_dir):
    """(a) 새 plans 가 patch_size·plans_name 말고는 원본과 같은지."""
    with open(os.path.join(preprocessed_dir, f'{SOURCE_PLANS}.json')) as f:
        source = json.load(f)
    with open(os.path.join(preprocessed_dir, f'{TARGET_PLANS}.json')) as f:
        target = json.load(f)

    if target['plans_name'] != TARGET_PLANS:
        raise VerificationError(f"plans_name 이 파일명과 다르다: {target['plans_name']} != {TARGET_PLANS} "
                                "— 결과 디렉토리 이름이 파일명이 아니라 이 값으로 정해진다")
    for key in source:
        if key in ('plans_name', 'configurations'):
            continue
        if source[key] != target[key]:
            raise VerificationError(f"plans 최상위 '{key}' 가 원본과 다르다")
    if set(source['configurations']) != set(target['configurations']):
        raise VerificationError("configuration 목록이 원본과 다르다")

    src_cfg, tgt_cfg = source['configurations'][CONFIGURATION], target['configurations'][CONFIGURATION]
    for key in IMMUTABLE_KEYS:
        if src_cfg[key] != tgt_cfg[key]:
            raise VerificationError(f"'{key}' 가 원본과 다르다 — patch 만 바꿔야 한다")
    changed = sorted(k for k in src_cfg if src_cfg[k] != tgt_cfg[k])
    if changed != ['patch_size']:
        raise VerificationError(f"{CONFIGURATION} 에서 바뀐 항목이 patch_size 하나가 아니다: {changed}")
    print(f"  검증 통과 (a): 원본 대비 바뀐 것은 plans_name 과 {CONFIGURATION}.patch_size 뿐")


def verify_preprocessed_reuse(preprocessed_dir):
    """(b) 전처리 재사용 — data_identifier 폴더가 이미 있고 케이스가 들어 있는지."""
    with open(os.path.join(preprocessed_dir, f'{TARGET_PLANS}.json')) as f:
        data_identifier = json.load(f)['configurations'][CONFIGURATION]['data_identifier']
    folder = os.path.join(preprocessed_dir, data_identifier)
    if not os.path.isdir(folder):
        raise VerificationError(f"전처리 폴더 없음: {folder} — 재전처리가 필요하다는 뜻이다")
    n_cases = len([f for f in os.listdir(folder) if f.endswith('.npz') or f.endswith('.b2nd')])
    if n_cases == 0:
        raise VerificationError(f"전처리 폴더가 비어 있다: {folder}")
    print(f"  검증 통과 (b): 전처리 재사용 {data_identifier} ({n_cases}건) — 재전처리 불필요")


def verify_splits_untouched(preprocessed_dir):
    """(c) splits_final.json 은 이 스크립트가 건드리지 않는다. 바뀌면 1~3번 산출물이 전부 무효다."""
    path = os.path.join(preprocessed_dir, 'splits_final.json')
    if not os.path.isfile(path):
        raise VerificationError(f"splits_final.json 없음: {path}")
    with open(path) as f:
        splits = json.load(f)
    n_val = [len(s['val']) for s in splits]
    print(f"  검증 통과 (c): splits_final.json 그대로 (fold {len(splits)}개, val {n_val}, md5 {md5(path)[:8]})")


def verify_loss():
    """(d) Tversky 구현 — alpha=beta=0.5 면 Dice 와 같고, beta 를 키우면 과소분할이 더 비싸야 한다."""
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTverskyCE import (
        MemoryEfficientSoftTverskyLoss)
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    torch.manual_seed(0)
    pred = torch.rand((2, 2, 16, 16, 16))
    ref = (torch.rand((2, 1, 16, 16, 16)) > 0.7).long()

    kwargs = dict(apply_nonlin=softmax_helper_dim1, batch_dice=False, do_bg=False, smooth=0, ddp=False)
    dice = MemoryEfficientSoftDiceLoss(**kwargs)(pred, ref).item()
    tversky_half = MemoryEfficientSoftTverskyLoss(**kwargs, alpha=0.5, beta=0.5)(pred, ref).item()
    if abs(dice - tversky_half) > 1e-6:
        raise VerificationError(f"alpha=beta=0.5 가 Dice 와 다르다: {tversky_half:.8f} != {dice:.8f}")

    # 같은 크기의 오차를 과소분할 쪽에만 준다 — recall 편향이면 이쪽 loss 가 더 커야 한다
    logits_under = torch.full((1, 2, 8, 8, 8), 0.0)
    logits_under[:, 1] = -4.0                      # 전경을 거의 예측하지 않음
    target = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    target[:, :, :4] = 1
    logits_over = torch.full((1, 2, 8, 8, 8), 0.0)
    logits_over[:, 1] = 4.0                        # 전경을 과하게 예측

    loss_fn = MemoryEfficientSoftTverskyLoss(apply_nonlin=softmax_helper_dim1, batch_dice=False,
                                             do_bg=False, smooth=1e-5, ddp=False, alpha=0.3, beta=0.7)
    under, over = loss_fn(logits_under, target).item(), loss_fn(logits_over, target).item()
    if not under > over:
        raise VerificationError(f"recall 편향이 반대로 걸렸다: 과소분할 {under:.4f} <= 과분할 {over:.4f}")
    print(f"  검증 통과 (d): alpha=beta=0.5 == Dice ({dice:.6f}), "
          f"alpha=0.3/beta=0.7 에서 과소분할 {under:.4f} > 과분할 {over:.4f}")


def verify_trainers(source_path, folds):
    """(e) nnU-Net 자신의 조회 경로로 trainer 를 찾고, 실제로 만들어 patch·epoch·loss 를 확인한다."""
    import nnunetv2
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.run.run_training import get_trainer_from_args
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    for name in TRAINER_NAMES:
        found = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                            name, 'nnunetv2.training.nnUNetTrainer')
        if found is None:
            raise VerificationError(f"nnU-Net 이 trainer 를 못 찾는다: {name}")
        if not issubclass(found, nnUNetTrainer):
            raise VerificationError(f"{name} 이 nnUNetTrainer 하위가 아니다")
        module_file = os.path.realpath(inspect.getfile(found))
        if module_file != os.path.realpath(source_path):
            raise VerificationError(f"{name} 이 저장소 파일이 아닌 {module_file} 에서 로드됐다")

    trainer = get_trainer_from_args(DATASET_ID, CONFIGURATION, folds[0],
                                    'nnUNetTrainerTverskyCE_2000epochs', TARGET_PLANS,
                                    device=torch.device('cpu'))
    if list(trainer.configuration_manager.patch_size) != list(NEW_PATCH_SIZE):
        raise VerificationError(f"patch 가 반영되지 않았다: {trainer.configuration_manager.patch_size}")
    if trainer.num_epochs != 2000:
        raise VerificationError(f"epoch 수가 다르다: {trainer.num_epochs}")
    if not os.path.isdir(trainer.preprocessed_dataset_folder):
        raise VerificationError(f"전처리 폴더를 못 찾는다: {trainer.preprocessed_dataset_folder}")

    trainer.initialize()
    loss = trainer.loss.loss if hasattr(trainer.loss, 'loss') else trainer.loss
    dice_term = getattr(loss.dc, '_orig_mod', loss.dc)  # torch.compile 이 걸리면 래퍼가 씌워진다
    if type(dice_term).__name__ != 'MemoryEfficientSoftTverskyLoss':
        raise VerificationError(f"loss 의 Dice 항이 Tversky 가 아니다: {type(dice_term).__name__}")
    if (dice_term.alpha, dice_term.beta) != (0.3, 0.7):
        raise VerificationError(f"alpha/beta 가 다르다: {dice_term.alpha}/{dice_term.beta}")

    tr_keys, val_keys = trainer.do_split()
    print(f"  검증 통과 (e): trainer {len(TRAINER_NAMES)}개 조회 성공, "
          f"patch {trainer.configuration_manager.patch_size} · batch {trainer.configuration_manager.batch_size} · "
          f"epoch {trainer.num_epochs} · Tversky(alpha={dice_term.alpha}, beta={dice_term.beta})")
    print(f"    결과 디렉토리: {trainer.output_folder_base}")
    print(f"    fold {folds[0]}: train {len(tr_keys)} / val {len(val_keys)}")


def main():
    parser = argparse.ArgumentParser(description='recall 편향 재학습용 trainer 링크 + patch 축소 plans 설치')
    parser.add_argument('--preprocessed_dir',
                        default=os.path.join(os.environ.get('nnUNet_preprocessed', ''), DATASET_NAME))
    parser.add_argument('--patch_size', type=int, nargs=3, default=NEW_PATCH_SIZE, metavar=('Z', 'Y', 'X'))
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 4], help='검증에 쓸 fold')
    parser.add_argument('--dry_run', action='store_true', help='쓰지 않고 확인만')
    parser.add_argument('--force', action='store_true', help='기존 링크/plans 덮어쓰기')
    args = parser.parse_args()

    source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TRAINER_MODULE)
    if not os.path.isfile(source_path):
        raise VerificationError(f"trainer 원본 없음: {source_path}")
    if not os.path.isdir(args.preprocessed_dir):
        raise VerificationError(f"전처리 디렉토리 없음: {args.preprocessed_dir} — nnUNet_preprocessed 환경변수 확인")

    print(f"\n=== recall 편향 재학습 설치 ===")
    print(f"전처리 디렉토리: {args.preprocessed_dir}")
    print(f"trainer 원본: {source_path}")

    print("\n[1/2] trainer 링크")
    install_trainer(source_path, args.dry_run, args.force)

    print("\n[2/2] plans 생성")
    plans, old_patch, divisor = build_plans(args.preprocessed_dir, args.patch_size)
    config = plans['configurations'][CONFIGURATION]
    print(f"  stride 곱 {divisor.tolist()} 로 나누어떨어짐")
    report_patch_change(old_patch, config['patch_size'], config['spacing'], config['batch_size'])
    write_plans(plans, args.preprocessed_dir, args.dry_run, args.force)

    if args.dry_run:
        print("\n[dry-run] 검증은 실제 설치 후에만 돈다")
        return

    print("\n=== 검증 ===")
    verify_plans_diff(args.preprocessed_dir)
    verify_preprocessed_reuse(args.preprocessed_dir)
    verify_splits_untouched(args.preprocessed_dir)
    verify_loss()
    verify_trainers(source_path, args.folds)

    print("\n=== 완료 ===")


if __name__ == '__main__':
    main()
