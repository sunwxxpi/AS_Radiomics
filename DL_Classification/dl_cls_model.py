import os
import sys
import json
import torch
import torch.nn as nn
import monai
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Config

class CustomModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomModel, self).__init__()
        
        self.backbone = monai.networks.nets.resnet50(
            pretrained=True,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,
            feed_forward=False,  # FC layer 제외
            shortcut_type='B',
            bias_downsample=False
        )
        
        # Feature dimension 추출
        self.in_features = self.backbone.in_planes
        
        # Classification head 별도 정의
        self.classifier = nn.Linear(self.in_features, num_classes)
        
    def forward(self, images=None):
        features = self.backbone(images)
        logits = self.classifier(features)
        
        return logits


class nnUNetEncoder(nn.Module):
    """nnU-Net에서 encoder 부분만 추출한 모델"""
    def __init__(self, full_model):
        super().__init__()
        
        if hasattr(full_model, 'encoder'):
            self.encoder_module = full_model.encoder
            print(f"✓ Successfully extracted encoder from {type(full_model).__name__}")
            print(f"Encoder type: {type(self.encoder_module).__name__}")
            
            total_params = sum(p.numel() for p in self.encoder_module.parameters())
            trainable_params = sum(p.numel() for p in self.encoder_module.parameters() if p.requires_grad)
            print(f"Encoder parameters: {total_params:,} total, {trainable_params:,} trainable")
        else:
            raise ValueError("Cannot find encoder in the model structure")
    
    def forward(self, x):
        features = self.encoder_module(x)
        
        if isinstance(features, (list, tuple)):
            bottleneck_features = features[-1]
        else:
            bottleneck_features = features
        
        spatial_dims = tuple(range(2, bottleneck_features.ndim))
        pooled_features = torch.mean(bottleneck_features, dim=spatial_dims)
        
        return pooled_features


class nnUNetClassificationModel(nn.Module):
    """nnU-Net Encoder + Classification Head"""
    def __init__(self, num_classes=2, pretrained_encoder_path=None):
        super().__init__()
        
        if pretrained_encoder_path:
            # Load pretrained nnUNet backbone
            self.backbone = self._load_pretrained_backbone(pretrained_encoder_path)
            
            # Feature dimension 직접 추출 (nnUNet encoder의 마지막 stage 출력 채널 수)
            if hasattr(self.backbone.encoder_module, 'stages'):
                # 마지막 stage의 출력 채널 수 추출
                last_stage = self.backbone.encoder_module.stages[-1]
                
                if hasattr(last_stage, 'blocks'):
                    # ResidualBlock의 출력 채널 수
                    feature_dim = last_stage.blocks[-1].conv2.all_modules[0].out_channels
            
            print(f"Extracted feature dimension from nnUNet encoder: {feature_dim}")
        else:
            raise ValueError("pretrained_encoder_path is required for nnUNetClassificationModel")
        
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def _load_pretrained_backbone(self, encoder_config):
        """사전학습된 nnUNet backbone 을 로드한다.

        체크포인트가 없거나, 로드에 실패하거나, 키가 아키텍처와 어긋나 encoder 가 채워지지 않으면 예외로 멈춘다.
        무작위 초기화로 이어가면 사전학습 없이 학습된 사실이 로그에만 남아 성능 저하의 원인을 놓치게 된다.
        """
        plans_file_arch = encoder_config.get('plans_file_arch')
        dataset_json_file = encoder_config.get('dataset_json_file')
        checkpoint_file = encoder_config.get('checkpoint_file')
        configuration = encoder_config.get('configuration')
        
        # JSON 파일들 로드
        with open(dataset_json_file, 'r') as f:
            dataset_json = json.load(f)

        # PlansManager 및 ConfigurationManager 로드
        plans_manager = PlansManager(plans_file_arch)
        config_manager = plans_manager.get_configuration(configuration)
        label_manager = plans_manager.get_label_manager(dataset_json)

        # 네트워크 아키텍처 빌드
        model = get_network_from_plans(
            config_manager.network_arch_class_name,
            config_manager.network_arch_init_kwargs,
            config_manager.network_arch_init_kwargs_req_import,
            1,  # num_input_channels
            label_manager.num_segmentation_heads,  # num_output_channels
            allow_init=True,
            deep_supervision=True
        )

        if not checkpoint_file:
            raise ValueError("No checkpoint file specified in the nnUNet encoder config")

        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_file}")

        print(f"Loading checkpoint from: {checkpoint_file}")
        try:
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {checkpoint_file}: {e}") from e

        # Checkpoint 유효성 검증
        if 'network_weights' not in checkpoint:
            raise ValueError("Invalid checkpoint: 'network_weights' key not found")

        network_weights = {k.replace('module.', ''): v for k, v in checkpoint['network_weights'].items()}

        # 가중치 로드
        # segmentation head 를 쓰지 않아 strict=False 가 필요하므로, 키 어긋남은 반환값으로만 잡을 수 있다.
        load_result = model.load_state_dict(network_weights, strict=False)

        model_keys = list(model.state_dict().keys())
        missing_keys = list(load_result.missing_keys)
        unexpected_keys = list(load_result.unexpected_keys)
        matched_count = len(model_keys) - len(missing_keys)

        if matched_count == 0:
            raise RuntimeError(
                f"Checkpoint {checkpoint_file} matched none of the {len(model_keys)} model parameters: "
                f"{len(missing_keys)} missing (e.g. {missing_keys[:5]}), "
                f"{len(unexpected_keys)} unexpected (e.g. {unexpected_keys[:5]})"
            )

        encoder_missing = [key for key in missing_keys if key.startswith('encoder.')]
        if encoder_missing:
            raise RuntimeError(
                f"Checkpoint {checkpoint_file} left {len(encoder_missing)} encoder parameters unloaded "
                f"(e.g. {encoder_missing[:5]}); {len(unexpected_keys)} checkpoint keys were unexpected "
                f"(e.g. {unexpected_keys[:5]})"
            )

        print(f"✓ Successfully loaded pre-trained nnU-Net model! "
              f"(matched {matched_count}/{len(model_keys)} parameters, {len(unexpected_keys)} unexpected keys)")

        # Encoder 추출
        encoder = nnUNetEncoder(model)
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return encoder
    
    def forward(self, images=None):
        features = self.backbone(images)
        logits = self.classifier(features)
        
        return logits


def create_model(config):
    if config.model_type == 'nnunet':
        # config.py의 DL_NNUNET_CONFIG 사용
        encoder_config = Config.DL_NNUNET_CONFIG.copy()
        model = nnUNetClassificationModel(num_classes=config.num_classes, pretrained_encoder_path=encoder_config)
        print(f"  Plans file (arch): {encoder_config.get('plans_file_arch')}")
        print(f"  Configuration: {encoder_config.get('configuration')}")
        print(f"  Dataset JSON: {encoder_config.get('dataset_json_file')}")
        print(f"  Checkpoint: {encoder_config.get('checkpoint_file')}")
        print(f"  Plans file (norm): {encoder_config.get('plans_file_norm')}")
        print(f"✓ Using nnUNet encoder model with {config.num_classes} classes\n")
    else:
        model = CustomModel(num_classes=config.num_classes)
        print(f"✓ Using custom MONAI ResNet50 model with {config.num_classes} classes\n")

    return model