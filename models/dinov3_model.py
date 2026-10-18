"""
DINOv3 + TS-TCC encoder/classifier for SpeechOcean762 Log-Mel tensors.

Expected input: (B, 64, 1536) or (B, 64, T)
Returns: (logits, features), where features has shape (B, 128, 14)
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


class DINOv3_Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        model_id = getattr(
            configs, "dinov3_model_id",
            "facebook/dinov3-vits16-pretrain-lvd1689m"
        )
        self.backbone = AutoModel.from_pretrained(model_id)
        hidden_size = getattr(self.backbone.config, "hidden_size", 384)
        self.freeze_backbone = bool(getattr(configs, "dinov3_freeze_backbone", True))
        self.image_size = int(getattr(configs, "dinov3_image_size", 224))
        self.grid_size = self.image_size // 16
        self.feature_channels = int(configs.final_out_channels)

        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_size, self.feature_channels),
            nn.LayerNorm(self.feature_channels),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(float(getattr(configs, "dropout", 0.35)))
        self.classifier = nn.Linear(
            self.feature_channels * self.grid_size,
            int(configs.num_classes)
        )

        # DINOv3 is frozen initially to reduce VRAM and stabilize training.
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen pretrained backbone in eval mode.
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _to_rgb_images(self, x):
        # Input Log-Mel: (B, mel_bins, frames). Convert to 1-channel image.
        if x.ndim != 3:
            raise ValueError(
                f"Expected input (B, mel_bins, frames), got shape {tuple(x.shape)}"
            )
        x = x.unsqueeze(1)  # (B, 1, mel_bins, frames)
        x = F.interpolate(
            x, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        )
        x = x.repeat(1, 3, 1, 1)

        # Per-sample standardization is safer for non-natural spectrogram images
        # than assuming raw Log-Mel values are already in [0, 1].
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        x = (x - mean) / std
        return x

    def forward(self, x_in):
        images = self._to_rgb_images(x_in)

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=images)
        else:
            outputs = self.backbone(pixel_values=images)

        # DINOv3 ViT-S/16: sequence usually contains CLS + register tokens + 196 patches.
        tokens = outputs.last_hidden_state
        patch_tokens = tokens[:, 1 + 4:, :]  # exclude CLS and 4 register tokens

        expected_patches = self.grid_size * self.grid_size
        if patch_tokens.shape[1] != expected_patches:
            raise RuntimeError(
                f"Expected {expected_patches} patch tokens for image size "
                f"{self.image_size}, got {patch_tokens.shape[1]}. "
                "Check the selected DINOv3 model and patch size."
            )

        b = patch_tokens.shape[0]
        patch_grid = patch_tokens.reshape(
            b, self.grid_size, self.grid_size, patch_tokens.shape[-1]
        )

        # Average over the frequency-like grid axis, leaving a sequence over time.
        sequence = patch_grid.mean(dim=1)            # (B, 14, hidden_size)
        sequence = self.feature_projection(sequence) # (B, 14, 128)
        features = sequence.transpose(1, 2).contiguous()  # (B, 128, 14)

        logits = self.classifier(self.dropout(features.flatten(start_dim=1)))
        return logits, features
