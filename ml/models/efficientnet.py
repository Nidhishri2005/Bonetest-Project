"""EfficientNet-B3 architecture with sex embedding fusion for pediatric bone-age prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB3SexFusion(nn.Module):
    """ImageNet-pretrained EfficientNet-B3 with sex embedding fusion.
    
    Architecture:
    - Input: 384x384 image (grayscale replicated to 3 channels: (B, 3, 384, 384))
    - Backbone: torchvision.models.efficientnet_b3 (features: 1536-dim)
    - Global Average Pooling: (B, 1536)
    - Sex Embedding: nn.Embedding(2, 32) mapping male (1) or female (0) -> 32-dim
    - Feature Fusion: Concatenation -> (B, 1536 + 32 = 1568)
    - Regression Head:
        Dropout(p=0.3) -> Linear(1568, 256) -> BatchNorm1d(256) -> ReLU()
        Dropout(p=0.2) -> Linear(256, 64) -> BatchNorm1d(64) -> ReLU()
        Linear(64, 1) -> Output scalar standardized age
    """

    def __init__(
        self,
        pretrained: bool = False,
        dropout: float = 0.3,
        sex_embedding_dim: int = 32,
        sex_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        if sex_embed_dim is not None:
            sex_embedding_dim = sex_embed_dim
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.sex_embedding = nn.Embedding(2, sex_embedding_dim)

        in_features = 1536 + sex_embedding_dim  # 1536 for EfficientNet-B3
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.67),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Image tensor of shape (B, 1, H, W) or (B, 3, H, W).
               If 1-channel, automatically replicated to 3 channels.
            sex: Sex tensor of shape (B,) or (B, 1), integer 0 (female) or 1 (male).
                 If float/bool, cast to long indices.
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)

        if sex.dim() > 1:
            sex = sex.squeeze(-1)
        sex_idx = sex.long()
        sex_emb = self.sex_embedding(sex_idx)

        fused = torch.cat([feat, sex_emb], dim=1)
        out = self.head(fused).squeeze(-1)
        return out

    def get_gradcam_target_layer(self) -> nn.Module:
        """Target layer for Grad-CAM++ (last convolutional block of features)."""
        return self.features[-1]
