import argparse


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


def generate_writer_comment(model_type, img_size):
    """model_type과 img_size를 기반으로 writer_comment 자동 생성"""
    if isinstance(img_size, (tuple, list)) and len(img_size) == 3:
        depth, height, width = img_size
        return f"{model_type}_{depth}_{height}_{width}"


def setup_nnunet_paths(nnunet_config):
    """nnunet_config를 기반으로 nnUNet 관련 경로들을 자동 설정"""
    base_path = f'./DL_Classification/nnUNet/{nnunet_config}'
    
    return {
        'plans_file': f'{base_path}/nnUNetResEncUNetLPlans.json',
        'dataset_json': f'{base_path}/dataset.json',
        'checkpoint': f'{base_path}/checkpoint_final.pth'
    }


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./DL_Classification/weights')
    parser.add_argument('--writer_comment', type=str, default=None)
    parser.add_argument('--save_model', type=bool, default=True)

    # MODEL PARAMETER
    parser.add_argument('--img_size', type=str, default='(32, 384, 320)') # nnUNet : (32, 384, 320), Med3D : (56, 448, 448)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--log_step', type=int, default=1)

    # MODEL TYPE SELECTION
    parser.add_argument('--model_type', type=str, default='nnunet', choices=['custom', 'nnunet'], 
                       help='Model type: custom (MONAI ResNet50) or nnunet (nnUNet encoder)')

    # nnUNet SPECIFIC PARAMETERS
    parser.add_argument('--nnunet_config', type=str, default='Dataset001_COCA')
    parser.add_argument('--nnunet_plans_file', type=str, default=None)
    parser.add_argument('--nnunet_dataset_json', type=str, default=None)
    parser.add_argument('--nnunet_checkpoint', type=str, default=None)
    parser.add_argument('--nnunet_configuration', type=str, default='3d_fullres')
    
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
    
    # nnUNet 경로 자동 설정
    if config.model_type == 'nnunet':
        nnunet_paths = setup_nnunet_paths(config.nnunet_config)
        
        # 사용자가 직접 지정하지 않은 경우만 자동 설정
        if config.nnunet_plans_file is None:
            config.nnunet_plans_file = nnunet_paths['plans_file']
        if config.nnunet_dataset_json is None:
            config.nnunet_dataset_json = nnunet_paths['dataset_json']
        if config.nnunet_checkpoint is None:
            config.nnunet_checkpoint = nnunet_paths['checkpoint']
    
    if config.writer_comment is None:
        config.writer_comment = generate_writer_comment(config.model_type, config.img_size)
    
    return config