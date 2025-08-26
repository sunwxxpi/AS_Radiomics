import os
import json
import torch
import torch.nn as nn
import monai
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    

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
        """Load nnUNet backbone (with or without pretrained weights)"""
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

        # 체크포인트 파일이 있고 존재하는 경우만 가중치 로드
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from: {checkpoint_file}")
            try:
                checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)

                # Checkpoint 유효성 검증
                if 'network_weights' not in checkpoint:
                    raise ValueError("Invalid checkpoint: 'network_weights' key not found")

                network_weights = {k.replace('module.', ''): v for k, v in checkpoint['network_weights'].items()}
                
                # 가중치 로드
                model.load_state_dict(network_weights, strict=False)
                print("✓ Successfully loaded pre-trained nnU-Net model!")
                
            except Exception as e:
                print(f"⚠️ Warning: Failed to load checkpoint ({e}). Using randomly initialized weights.")
        else:
            if checkpoint_file:
                print(f"⚠️ Warning: Checkpoint file not found at {checkpoint_file}. Using randomly initialized weights.")
            else:
                print("⚠️ No checkpoint file specified. Using randomly initialized nnU-Net architecture.")
        
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