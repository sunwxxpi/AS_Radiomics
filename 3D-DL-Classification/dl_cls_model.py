import json
import torch
import torch.nn as nn
import monai
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    

class CustomModel(nn.Module):
    def __init__(self, class_num=2):
        super(CustomModel, self).__init__()
        
        self.num_classes = class_num
        
        self.net = monai.networks.nets.resnet50(
            pretrained=True,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=class_num,
            feed_forward=False,
            shortcut_type='B',
            bias_downsample=False
        )
    
        self.in_features = self.net.in_planes
        self.net.fc = nn.Linear(self.in_features, class_num)
        
    def forward(self, images=None):
        return self.net(images)


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
    def __init__(self, class_num=2, pretrained_encoder_path=None):
        super().__init__()
        self.num_classes = class_num
        
        if pretrained_encoder_path:
            # Load pretrained nnUNet encoder
            self.encoder = self._load_pretrained_encoder(pretrained_encoder_path)
            
            # Feature dimension 자동 계산
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, 64, 64, 64)
                dummy_output = self.encoder(dummy_input)
                feature_dim = dummy_output.shape[1]
                print(f"Detected feature dimension: {feature_dim}")
        else:
            raise ValueError("pretrained_encoder_path is required for nnUNetClassificationModel")
        
        # Classification head
        """ self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, class_num)
        ) """
        self.classifier = nn.Linear(feature_dim, class_num)
    
    def _load_pretrained_encoder(self, encoder_config):
        """Load pretrained nnUNet encoder"""
        plans_file = encoder_config.get('plans_file', './3D-DL-Classification/nnUNet/nnUNetResEncUNetLPlans.json')
        dataset_json_file = encoder_config.get('dataset_json_file', './3D-DL-Classification/nnUNet/dataset.json')
        checkpoint_file = encoder_config.get('checkpoint_file', './3D-DL-Classification/nnUNet/checkpoint_final.pth')
        configuration = encoder_config.get('configuration', '3d_fullres')
        
        # JSON 파일들 로드
        with open(dataset_json_file, 'r') as f:
            dataset_json = json.load(f)

        # PlansManager 및 ConfigurationManager 로드
        plans_manager = PlansManager(plans_file)
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

        # 체크포인트에서 가중치 불러오기
        print(f"Loading checkpoint from: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)

        # Checkpoint 유효성 검증
        if 'network_weights' not in checkpoint:
            raise ValueError("Invalid checkpoint: 'network_weights' key not found")

        network_weights = {k.replace('module.', ''): v for k, v in checkpoint['network_weights'].items()}
        
        # 가중치 로드
        model.load_state_dict(network_weights, strict=False)
        print("✓ Successfully loaded pre-trained nnU-Net model!")
        
        # Encoder 추출
        encoder = nnUNetEncoder(model)
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return encoder
    
    def forward(self, images=None):
        features = self.encoder(images)
        logits = self.classifier(features)
        return logits