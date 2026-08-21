import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Config


def parse_img_size(img_size_str):
    """이미지 크기 문자열을 튜플로 변환하는 함수"""
    if isinstance(img_size_str, (tuple, list)):
        return tuple(img_size_str)
    
    # 문자열인 경우 파싱
    if isinstance(img_size_str, str):
        # 괄호와 공백 제거
        img_size_str = img_size_str.strip().strip('()')
        # 쉼표로 분리하여 정수 튜플로 변환 (공백도 제거)
        try:
            sizes = [int(x.strip()) for x in img_size_str.split(',') if x.strip()]
            if len(sizes) == 1:
                # 단일 값인 경우 3D로 확장
                return (sizes[0], sizes[0], sizes[0])
            elif len(sizes) == 3:
                return tuple(sizes)
            else:
                raise ValueError("img_size는 1개 또는 3개의 값이어야 합니다.")
        except ValueError as e:
            raise ValueError(f"img_size 파싱 오류: {e}")
    
    # 정수인 경우 3D로 확장
    if isinstance(img_size_str, int):
        return (img_size_str, img_size_str, img_size_str)
    
    raise ValueError(f"지원하지 않는 img_size 형태: {type(img_size_str)}")


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=Config.DL_WEIGHTS_ROOT)
    parser.add_argument('--save_model', type=bool, default=True)

    # MODEL PARAMETER
    parser.add_argument('--img_size', type=str, default=str(Config.DL_IMG_SIZE)) # nnUNet : (32, 384, 320), Med3D : (56, 448, 448)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--fold', type=int, default=Config.DL_NUM_FOLDS)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--stage', type=str, default='all', choices=['all', 'folds', 'refit'],
                       help="학습·평가 대상. all: 5-fold 와 refit 둘 다, folds: 5-fold 만, "
                            "refit: refit 만 (학습은 fold_best_epochs.csv 에서 종료 epoch 을 읽는다)")
    parser.add_argument('--resume', action='store_true',
                       help="fold_best_epochs.csv 에 기록이 있고 best_model.pth 도 있는 fold 는 다시 돌리지 않고 기록된 값을 쓴다")
    parser.add_argument('--log_step', type=int, default=1)

    # MODEL TYPE SELECTION
    parser.add_argument('--model_type', type=str, default=Config.DL_MODEL_TYPE, choices=['custom', 'nnunet'], 
                       help='Model type: custom (MONAI ResNet50) or nnunet (nnUNet encoder)')
    
    # CAM GENERATION CONTROL
    parser.add_argument('--enable_cam', action='store_true')
    
    # DATA SPLIT PARAMETERS
    parser.add_argument('--data_split_mode', type=str, default=Config.DATA_SPLIT_MODE, choices=['random', 'fix'])
    parser.add_argument('--data_split_random_state', type=int, default=Config.DATA_SPLIT_RANDOM_STATE)
    parser.add_argument('--test_size_ratio', type=float, default=Config.TEST_SIZE_RATIO)
        
    # LEARNING RATE PARAMETERS
    parser.add_argument('--loss_function', type=str, default='CE')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--lr', type=float, default=5e-5)
    
    parser.add_argument('--head_warmup_lr', type=float, default=1e-7)
    parser.add_argument('--head_warmup_epochs', type=int, default=0)
    parser.add_argument('--backbone_lr_ratio', type=float, default=1.0)
    
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_decay', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.9)

    config = parser.parse_args()
    
    config.img_size = parse_img_size(config.img_size)

    # 산출물 경로의 이름은 radiomics 파이프라인이 읽는 값 하나로만 정한다.
    config.writer_comment = Config.DL_COMMENT_WRITER

    if config.stage == 'refit' and config.resume:
        raise ValueError("--stage refit 은 fold 학습을 안 해 --resume 이 할 일이 없다: --stage all 이나 --stage folds 로 돌린다")

    # 학습·평가는 여기 값으로 경로와 분할을 정하고 radiomics 파이프라인은 Config 를 읽는다.
    # 어긋나면 writer_comment 가 같아 같은 자리에 다른 분할·다른 fold 수로 학습한 가중치를 덮어쓴다.
    mismatched = []
    if config.model_type != Config.DL_MODEL_TYPE:
        mismatched.append(f"model_type={config.model_type} vs DL_MODEL_TYPE={Config.DL_MODEL_TYPE}")
    if config.img_size != tuple(Config.DL_IMG_SIZE):
        mismatched.append(f"img_size={config.img_size} vs DL_IMG_SIZE={tuple(Config.DL_IMG_SIZE)}")
    if os.path.realpath(config.model_path) != os.path.realpath(Config.DL_WEIGHTS_ROOT):
        mismatched.append(f"model_path={config.model_path} vs DL_WEIGHTS_ROOT={Config.DL_WEIGHTS_ROOT}")
    if config.fold != Config.DL_NUM_FOLDS:
        mismatched.append(f"fold={config.fold} vs DL_NUM_FOLDS={Config.DL_NUM_FOLDS}")
    if config.data_split_mode != Config.DATA_SPLIT_MODE:
        mismatched.append(f"data_split_mode={config.data_split_mode} vs DATA_SPLIT_MODE={Config.DATA_SPLIT_MODE}")
    if config.data_split_random_state != Config.DATA_SPLIT_RANDOM_STATE:
        mismatched.append(f"data_split_random_state={config.data_split_random_state} vs "
                          f"DATA_SPLIT_RANDOM_STATE={Config.DATA_SPLIT_RANDOM_STATE}")
    if config.test_size_ratio != Config.TEST_SIZE_RATIO:
        mismatched.append(f"test_size_ratio={config.test_size_ratio} vs TEST_SIZE_RATIO={Config.TEST_SIZE_RATIO}")

    if mismatched:
        raise ValueError(
            f"CLI 인자가 config.py 와 다르다 ({'; '.join(mismatched)}) — config.py 쪽 값도 같이 바꾼다"
        )

    return config
