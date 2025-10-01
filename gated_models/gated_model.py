import torch
import torch.nn as nn


class GatedFusionLayer(nn.Module):
    """Gated Mechanism을 통한 Radiomics와 DL feature의 adaptive fusion

    수식:
        h = tanh(W_h [Radiomics; Deep Learning] + b_h)
        g = σ(w_g [Radiomics; Deep Learning] + b_g)
        F_fused = g ⊗ h  (element-wise multiplication)

    Args:
        radiomics_dim (int): Radiomics 특징 차원
        dl_dim (int): DL embedding 특징 차원
        fusion_dim (int): Fusion 후 출력 차원 (기본값: radiomics_dim + dl_dim)
        dropout (float): Dropout 비율 (기본값: 0.3)
    """

    def __init__(self, radiomics_dim, dl_dim, fusion_dim=None, dropout=0.3):
        super(GatedFusionLayer, self).__init__()

        self.radiomics_dim = radiomics_dim
        self.dl_dim = dl_dim
        self.input_dim = radiomics_dim + dl_dim
        self.fusion_dim = fusion_dim if fusion_dim else self.input_dim

        # Transformation layer: h = tanh(W_h * [Radiomics; DL] + b_h)
        self.transform_layer = nn.Linear(self.input_dim, self.fusion_dim)

        # Gate layer: g = sigmoid(w_g * [Radiomics; DL] + b_g)
        self.gate_layer = nn.Linear(self.input_dim, self.fusion_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 초기화"""
        nn.init.xavier_uniform_(self.transform_layer.weight)
        nn.init.zeros_(self.transform_layer.bias)

        nn.init.xavier_uniform_(self.gate_layer.weight)
        nn.init.zeros_(self.gate_layer.bias)

    def forward(self, radiomics_features, dl_features):
        """Forward pass

        Args:
            radiomics_features (torch.Tensor): Radiomics 특징 [batch_size, radiomics_dim]
            dl_features (torch.Tensor): DL embedding 특징 [batch_size, dl_dim]

        Returns:
            torch.Tensor: Fused 특징 [batch_size, fusion_dim]
        """
        # Concatenate radiomics and DL features
        concat_features = torch.cat([radiomics_features, dl_features], dim=1)

        # Transformation: h = tanh(W_h * concat + b_h)
        h = torch.tanh(self.transform_layer(concat_features))

        # Gate: g = sigmoid(w_g * concat + b_g)
        g = torch.sigmoid(self.gate_layer(concat_features))

        # Gated fusion: F_fused = g ⊗ h
        fused_features = g * h

        # Apply dropout
        fused_features = self.dropout(fused_features)

        return fused_features


class GatedFusionClassifier(nn.Module):
    """Gated Fusion + Classification Head

    Two-stage 학습을 위한 모델:
    1. Stage 1: 이 모델로 Gated Fusion layer 학습
    2. Stage 2: 학습된 Gated Fusion으로 fused features 추출 → 전통적 ML 분류기 학습

    Args:
        radiomics_dim (int): Radiomics 특징 차원
        dl_dim (int): DL embedding 특징 차원
        num_classes (int): 분류 클래스 수
        fusion_dim (int): Fusion 후 출력 차원 (기본값: None, radiomics_dim + dl_dim 사용)
        hidden_dims (list): 분류 헤드의 hidden layer 차원 리스트 (기본값: [256, 128])
        dropout (float): Dropout 비율 (기본값: 0.3)
    """

    def __init__(
        self,
        radiomics_dim,
        dl_dim,
        num_classes,
        fusion_dim=None,
        hidden_dims=[256, 128],
        dropout=0.3
    ):
        super(GatedFusionClassifier, self).__init__()

        self.radiomics_dim = radiomics_dim
        self.dl_dim = dl_dim
        self.num_classes = num_classes

        # Gated Fusion Layer
        self.gated_fusion = GatedFusionLayer(
            radiomics_dim, dl_dim, fusion_dim, dropout
        )
        self.fusion_dim = fusion_dim if fusion_dim else (radiomics_dim + dl_dim)

        # Classification Head (MLP)
        layers = []
        input_dim = self.fusion_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 초기화"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, radiomics_features, dl_features):
        """Forward pass

        Args:
            radiomics_features (torch.Tensor): Radiomics 특징 [batch_size, radiomics_dim]
            dl_features (torch.Tensor): DL embedding 특징 [batch_size, dl_dim]

        Returns:
            torch.Tensor: Classification logits [batch_size, num_classes]
        """
        # Gated Fusion
        fused_features = self.gated_fusion(radiomics_features, dl_features)

        # Classification
        logits = self.classifier(fused_features)

        return logits

    def extract_fused_features(self, radiomics_features, dl_features):
        """Fused features 추출 (전통적 ML 분류기 학습용)

        Args:
            radiomics_features (torch.Tensor): [batch_size, radiomics_dim]
            dl_features (torch.Tensor): [batch_size, dl_dim]

        Returns:
            torch.Tensor: [batch_size, fusion_dim]
        """
        with torch.no_grad():
            fused_features = self.gated_fusion(radiomics_features, dl_features)
        return fused_features