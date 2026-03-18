"""Vision Transformer (ViT) classifier using timm."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ViTClassifier(nn.Module):
    """ViT classifier built on the ``timm`` library.

    Features
    --------
    - Pretrained weights from timm model zoo
    - Configurable classification head
    - Attention weight extraction for interpretability
    """

    def __init__(
        self,
        num_classes: int = 100,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if model_name.startswith("torchvision/"):
            import torchvision.models as models
            base_name = model_name.replace("torchvision/", "")
            # Map common names
            name_map = {"vit_b_16": models.vit_b_16, "vit_l_16": models.vit_l_16}
            model_fn = name_map.get(base_name, models.vit_b_16)
            
            from collections import OrderedDict
            weights = "DEFAULT" if pretrained else None
            self.model = model_fn(weights=weights)
            # Replace head - use OrderedDict to match 'heads.head' naming
            in_features = self.model.heads.head.in_features
            self.model.heads = nn.Sequential(OrderedDict([
                ('head', nn.Linear(in_features, num_classes))
            ]))
            self._is_torchvision = True
        else:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove default head
                drop_rate=dropout,
            )
            embed_dim = self.model.embed_dim

            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(embed_dim, num_classes),
            )
            self._is_torchvision = False

        self._attention_weights: list[torch.Tensor] = []
        self._hooks: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_torchvision:
            return self.model(x)
            
        features = self.model.forward_features(x)  # (B, N, D)
        cls_token = features[:, 0]  # CLS token
        return self.head(cls_token)

    # ------------------------------------------------------------------
    # Attention extraction
    # ------------------------------------------------------------------

    def register_attention_hooks(self) -> None:
        """Register hooks on all transformer blocks to capture attention maps."""
        self._attention_weights.clear()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        if self._is_torchvision:
            # Torchvision: model.encoder.layers[i].self_attention
            for layer in self.model.encoder.layers:
                hook = layer.self_attention.register_forward_hook(self._attn_hook_torchvision)
                self._hooks.append(hook)
        else:
            # Timm: model.blocks[i].attn
            for block in self.model.blocks:
                hook = block.attn.register_forward_hook(self._attn_hook_timm)
                self._hooks.append(hook)

    def _attn_hook_timm(self, module, input, output):
        """Hook for timm Attention module."""
        # Generic way to get attention from timm (works for most ViTs)
        # B, N, C = input[0].shape
        # Re-compute attention is safer across timm versions
        qkv = module.qkv(input[0])
        B, N, C3 = qkv.shape
        C = C3 // 3
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        self._attention_weights.append(attn.detach().cpu())

    def _attn_hook_torchvision(self, module, input, output):
        """Hook for torchvision MultiheadAttention module."""
        # input[0] is (N, B, E) usually for torchvision's MHA if batch_first=False
        # But vision_transformer uses batch_first=True in recent versions
        # Let's check batch_first
        x = input[0]
        if getattr(module, "batch_first", False):
            # (B, N, E)
            pass
        else:
            # (N, B, E) -> (B, N, E)
            x = x.transpose(0, 1)
        
        B, N, E = x.shape
        num_heads = module.num_heads
        head_dim = E // num_heads
        
        # In torchvision, self_attention is a MultiheadAttention
        # We can't easily get the weights without re-computing if stay_weights=False (default)
        # So we re-compute from in_proj_weight
        q, k, v = torch.nn.functional.linear(x, module.in_proj_weight, module.in_proj_bias).chunk(3, dim=-1)
        
        q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        self._attention_weights.append(attn.detach().cpu())

    def get_attention_maps(self) -> list[torch.Tensor]:
        """Return captured attention maps from all layers.

        Each element has shape (B, num_heads, N, N).
        """
        return self._attention_weights

    def remove_attention_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._attention_weights.clear()
