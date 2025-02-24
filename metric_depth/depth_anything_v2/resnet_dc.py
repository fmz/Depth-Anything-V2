import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, List


#####################################################
#  Helper: Initialize weights (Kaiming normal)
#####################################################
def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


#####################################################
# 1. ResNet-34 Encoder
#    Returns a list of features: [layer4, layer3, layer2, layer1, conv1]
#####################################################
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(pretrained=pretrained)

        # First conv for RGB
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            [layer4, layer3, layer2, layer1, conv1]
            corresponding to [1/32, 1/16, 1/8, 1/4, 1/2] scales.
        """
        c1 = self.conv1(x)      # [B, 64,  H/2,  W/2 ]
        p1 = self.maxpool(c1)   # [B, 64,  H/4,  W/4 ]
        l1 = self.layer1(p1)    # [B, 64,  H/4,  W/4 ]
        l2 = self.layer2(l1)    # [B, 128, H/8,  W/8 ]
        l3 = self.layer3(l2)    # [B, 256, H/16, W/16]
        l4 = self.layer4(l3)    # [B, 512, H/32, W/32]
        return [l4, l3, l2, l1, c1]


#####################################################
# 2. Simple Depth Encoder
#    Returns: [d3, d2, d1] at scales [1/8, 1/4, 1/2]
#####################################################
class DepthEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )  # -> [B, 32,  H/2,  W/2 ]

        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )  # -> [B, 64,  H/4,  W/4 ]

        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )  # -> [B, 128, H/8,  W/8 ]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        d1 = self.conv1(x)  # [B, 32,  H/2,  W/2 ]
        d2 = self.conv2(d1) # [B, 64,  H/4,  W/4 ]
        d3 = self.conv3(d2) # [B, 128, H/8,  W/8 ]
        return [d3, d2, d1]


#####################################################
# 3. Gated Cross-Attention w/ optional mask
#    - self-attention on the RGB query
#    - cross-attention from depth (key/value)
#    - optional mask to ignore invalid key positions
#####################################################
class GatedCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        img_feat: torch.Tensor,   # [B, C, H, W]
        depth_feat: torch.Tensor, # [B, C, H, W]
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            img_feat:   [B, C, H, W] (query for attention)
            depth_feat: [B, C, H, W] (key/value for cross-attention)
            mask: Optional[Torch.Tensor] of shape [B, H, W] or [B, 1, H, W].
                  If provided, `True` means "ignore this position" in the depth_feat.
                  We'll flatten it to [B, N] to pass as key_padding_mask.

        Returns:
            fused: [B, C, H, W]
        """
        B, C, H, W = img_feat.shape
        N = H * W

        # Flatten spatially: [B, N, C]
        query = img_feat.view(B, C, N).permute(0, 2, 1)
        key   = depth_feat.view(B, C, N).permute(0, 2, 1)
        value = key

        # If we have a mask, reshape to [B, N], where `True` = ignore
        # for the key/value positions.
        key_padding_mask = None
        if mask is not None:
            # mask could be [B,1,H,W] or [B,H,W], flatten to [B,N]
            if mask.dim() == 4:
                mask = mask.squeeze(1)  # drop channel dim if it exists
            mask = mask.view(B, -1)  # [B, N]
            key_padding_mask = mask.bool()  # ensure bool if not already

        # Self-attention on the image features
        self_attn_out, _ = self.self_attn(
            query, query, query,
            key_padding_mask=key_padding_mask  # optionally also ignore positions in query
        )
        # Cross-attention: image queries depth
        cross_attn_out, _ = self.mha(
            query, key, value,
            key_padding_mask=key_padding_mask
        )

        # Compute gating from the original image feature
        gate = torch.sigmoid(self.gate_conv(img_feat))  # [B, C, H, W]
        gate = gate.view(B, C, N).permute(0, 2, 1)       # [B, N, C]

        # Fuse self-attention and cross-attention
        fused = gate * cross_attn_out + (1.0 - gate) * self_attn_out
        fused = self.norm(fused)  # [B, N, C]

        # Reshape back
        fused = fused.permute(0, 2, 1).view(B, C, H, W)
        return fused


#####################################################
# 4. UpBlock: (ConvTranspose) -> concat -> conv
#####################################################
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.iconv = nn.Conv2d(
            out_channels * 2, out_channels,
            kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: upsampled feature map
            skip: skip-connection feature map to concatenate
        """
        x = self.relu(self.up(x))
        # Ensure shape alignment
        # Example assertion (optional):
        # assert x.shape[2:] == skip.shape[2:], f"Mismatch in spatial size: {x.shape} vs {skip.shape}"

        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.iconv(x))
        return x


#####################################################
# 5. Multi-scale Decoder
#    Expects a list of encoder feats with channels:
#      [512, 256, 256, 64, 64]
#####################################################
class MultiScaleDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        """
        Example:
            encoder_channels = [512, 256, 256, 64, 64]
            decoder_channels = [256, 256, 64, 64]
        """
        super().__init__()
        self.up1 = UpBlock(encoder_channels[0], decoder_channels[0])  # 512 -> 256, skip 256
        self.up2 = UpBlock(encoder_channels[1], decoder_channels[1])  # 256 -> 128, skip 256
        self.up3 = UpBlock(encoder_channels[2], decoder_channels[2])  # 128 -> 64,  skip 64
        self.up4 = UpBlock(encoder_channels[3], decoder_channels[3])  # 64 -> 32,   skip 64

        self.out_conv = nn.Conv2d(decoder_channels[3], 1, kernel_size=3, padding=1)
        self.apply(init_weights)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        feats = [layer4, layer3, layer2, layer1, conv1]
          with channels [512, 256, 256, 64, 64].
        """
        x = feats[0]                  # [B, 512, H/32, W/32]
        x = self.up1(x, feats[1])     # => [B, 256, H/16, W/16]
        x = self.up2(x, feats[2])     # => [B, 128, H/8,  W/8 ]
        x = self.up3(x, feats[3])     # => [B, 64,  H/4,  W/4 ]
        x = self.up4(x, feats[4])     # => [B, 32,  H/2,  W/2 ]

        depth = self.out_conv(x)      # => [B, 1,   H/2,  W/2]
        # Upsample to match full resolution
        depth = F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=False)
        return depth


#####################################################
# 6. Spatial Propagation Refinement
#    Iterative local smoothing guided by RGB features
#####################################################
class SpatialPropagationRefinement(nn.Module):
    def __init__(self, num_iterations=3, kernel_size=3):
        super().__init__()
        self.num_iterations = num_iterations
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Guidance conv: from 64->(K*K) weights
        self.guidance_conv = nn.Conv2d(64, kernel_size*kernel_size, kernel_size=3, padding=1)

        # Pre-create an Unfold layer to extract patches
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)

        # Final activation block
        self.activation = nn.ReLU(inplace=True)

        self.apply(init_weights)

    def forward(self, depth: torch.Tensor, guidance_feat: torch.Tensor) -> torch.Tensor:
        """
        depth:         [B, 1,   H, W]
        guidance_feat: [B, 64,  H, W]
        """
        # Compute guidance weights => [B, K*K, H, W]
        guidance = self.guidance_conv(guidance_feat)
        # Softmax over channel dimension => sum of weights in each patch = 1
        guidance = F.softmax(guidance, dim=1)

        B, _, H, W = depth.shape
        for _ in range(self.num_iterations):
            # Extract patches from depth => [B, K*K, H*W]
            patches = self.unfold(depth)  # each vector is the local patch
            # Flatten guidance similarly => [B, K*K, H*W]
            guidance_flat = guidance.view(B, self.kernel_size * self.kernel_size, -1)
            # Weighted sum => [B, 1, H, W]
            refined = (patches * guidance_flat).sum(dim=1).view(B, 1, H, W)

            # Combine old depth with refined
            depth = 0.5 * depth + 0.5 * refined

        depth = self.activation(depth)

        return depth


#####################################################
# 7. Depth Completion Model
#####################################################
class DepthCompletionModel(nn.Module):
    def __init__(self, num_heads=4, attn_embed_dim=256):
        """
        A multi-scale depth-completion model with:
          - ResNet34 encoder for RGB
          - Depth encoder for sparse depth
          - Cross-attention block for mid-level fusion
          - U-Net style decoder with skip connections
          - Optional spatial propagation refinement
        """
        super().__init__()
        self.rgb_encoder = ResNetEncoder(pretrained=True)
        self.depth_encoder = DepthEncoder(in_channels=1, base_channels=32)

        # Project mid-level features (128 channels -> 256) for cross-attn
        self.rgb_proj = nn.Conv2d(128, attn_embed_dim, kernel_size=1)
        self.depth_proj = nn.Conv2d(128, attn_embed_dim, kernel_size=1)

        # Gated cross-attention
        self.gated_attn = GatedCrossAttention(embed_dim=attn_embed_dim, num_heads=num_heads)

        # Decoder: note the middle skip is now 256 channels after fusion
        self.decoder = MultiScaleDecoder(
            encoder_channels=[512, 256, 256, 64, 64],
            decoder_channels=[256, 256, 64, 64]
        )

        # Spatial refinement
        self.refine = SpatialPropagationRefinement(num_iterations=3, kernel_size=3)

        self.apply(init_weights)

    def forward(
        self,
        rgb: torch.Tensor,            # [B,3,H,W]
        sparse_depth: torch.Tensor,   # [B,1,H,W]
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            rgb:         [B, 3, H, W]     Input RGB
            sparse_depth:[B, 1, H, W]     Sparse depth map
            attn_mask:   Optional mask for attention, shape [B,H,W] or [B,1,H,W].
                         If True => ignore that spatial position in cross-attention.

        Returns:
            depth_refined: [B, 1, H, W]   Completed depth
        """
        # 1) Encode RGB
        rgb_feats = self.rgb_encoder(rgb)
        # => [l4(512), l3(256), l2(128), l1(64), c1(64)]

        # 2) Encode Depth
        depth_feats = self.depth_encoder(sparse_depth)
        # => [d3(128), d2(64), d1(32)]

        # 3) Mid-level fusion (layer2 & d3) => both are [B,128,H/8,W/8]
        rgb_mid = rgb_feats[2]      # [B,128,H/8,W/8]
        depth_mid = depth_feats[0]  # [B,128,H/8,W/8]

        # Project to attn_embed_dim => [B,256,H/8,W/8]
        rgb_proj = self.rgb_proj(rgb_mid)
        depth_proj = self.depth_proj(depth_mid)

        # 4) Cross-attention
        fused_feat = self.gated_attn(rgb_proj, depth_proj, mask=attn_mask)
        # fused_feat => [B,256,H/8,W/8]

        # 5) Replace original l2(128ch) with fused(256ch)
        rgb_feats[2] = fused_feat  # shape is now [B,256,H/8,W/8]

        # 6) Decode
        depth_initial = self.decoder(rgb_feats)  # => [B,1,H,W]

        # 7) Spatial propagation refinement
        #    Guidance from early conv1 => [B,64,H/2,W/2], upsample to [H,W].
        guidance = rgb_feats[4]  # [B,64,H/2,W/2]
        guidance = F.interpolate(
            guidance,
            size=depth_initial.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        depth_refined = self.refine(depth_initial, guidance)
        return depth_refined


# #####################################################
# # Example Usage
# #####################################################
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = DepthCompletionModel(num_heads=4, attn_embed_dim=256).to(device)
#     model.eval()

#     B, H, W = 2, 256, 512
#     rgb = torch.randn(B, 3, H, W).to(device)

#     # Create sparse depth
#     sparse_depth = torch.zeros(B, 1, H, W).to(device)
#     for b in range(B):
#         num_valid = int(0.05 * H * W)  # 5% valid
#         idx = torch.randperm(H * W)[:num_valid]
#         sparse_depth.view(B, -1)[b, idx] = torch.rand(num_valid).to(device) * 9.5 + 0.5

#     # Optional: build an attention mask (True => ignore)
#     # Example: ignore positions where depth==0
#     # shape: [B,H,W]
#     attn_mask = (sparse_depth == 0.0).squeeze(1)

#     with torch.no_grad():
#         output = model(rgb, sparse_depth, attn_mask=attn_mask)
#     print("Output depth shape:", output.shape)  # [B, 1, H, W]
