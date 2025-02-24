import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.gate_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, rgb_feat, depth_feat, mask=None):
        """
        rgb_feat:   [B, C, H, W]
        depth_feat: [B, C, H, W]
        mask: optional [B, H, W], True => ignore that key position
        """
        B, C, H, W = rgb_feat.shape
        N = H * W

        # Flatten
        query = rgb_feat.view(B, C, N).permute(0,2,1)   # [B, N, C]
        key   = depth_feat.view(B, C, N).permute(0,2,1) # [B, N, C]
        value = key

        # Optional mask => [B, N] with True=ignore
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask.view(B, -1).bool()

        # Self-attention on RGB
        self_attn_out, _ = self.self_attn(query, query, query, key_padding_mask=key_padding_mask)

        # Cross-attention: RGB queries Depth
        cross_attn_out, _ = self.cross_attn(query, key, value, key_padding_mask=key_padding_mask)

        # Gating
        gate = torch.sigmoid(self.gate_conv(rgb_feat))   # [B, C, H, W]
        gate = gate.view(B, C, N).permute(0,2,1)         # [B, N, C]

        # Fuse
        fused = gate * cross_attn_out + (1.0 - gate) * self_attn_out
        fused = self.norm(fused)   # [B, N, C]

        # Reshape
        fused = fused.permute(0,2,1).view(B, C, H, W)
        return fused

class DepthEncoder(nn.Module):
    """A lightweight CNN-based depth encoder. Replace with a Transformer if desired."""
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )  # -> [B, base_ch, H/2, W/2]
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True)
        )  # -> [B, 2*base_ch, H/4, W/4]
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True)
        )  # -> [B, 4*base_ch, H/8, W/8]

    def forward(self, x):
        """
        x: [B,1,H,W] depth input
        returns list: [feat3, feat2, feat1] from deepest to shallow
        """
        f1 = self.conv1(x)  # [B, base_ch,   H/2, W/2]
        f2 = self.conv2(f1) # [B, base_ch*2, H/4, W/4]
        f3 = self.conv3(f2) # [B, base_ch*4, H/8, W/8]
        return [f3, f2, f1]

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.iconv = nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.relu(self.up(x))
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.iconv(x))
        return x

class MultiScaleDecoder(nn.Module):
    """
    Simple top-down decoder that merges upsampled features with skip connections.
    """
    def __init__(self, ch_in=[512,256,128], ch_out=[256,128,64]):
        """
        Example:
          ch_in = [512,256,128] from deepest to shallow
          ch_out= [256,128,64]
        """
        super().__init__()
        self.up1 = UpBlock(ch_in[0], ch_out[0])
        self.up2 = UpBlock(ch_out[0], ch_out[1])
        self.up3 = UpBlock(ch_out[1], ch_out[2])
        self.out_conv = nn.Conv2d(ch_out[2], 1, kernel_size=3, padding=1)

    def forward(self, feats):
        # feats: [f0, f1, f2], from deep to shallow
        x = feats[0]                # deepest
        x = self.up1(x, feats[1])   # mid
        x = self.up2(x, feats[2])   # shallow
        x = self.up3(x, None)       # if we have only 3 scales
        depth = self.out_conv(x)
        return depth

class DepthAnythingCrossAttention(nn.Module):
    def __init__(
        self,
        encoder='vitl',            # e.g. 'vitl' -> ViT Large
        intermediate_idx=None,     # which layers to extract
        max_depth=20.0,
        num_heads=4,
        attn_embed_dim=256,
    ):
        super().__init__()
        if intermediate_idx is None:
            # For 'vitl', your code used [4, 11, 17, 23]
            self.intermediate_idx = [4, 11, 17, 23]
        else:
            self.intermediate_idx = intermediate_idx

        self.max_depth = max_depth

        # 1) DINOv2 as the RGB encoder
        self.rgb_encoder = DINOv2(model_name=encoder)

        # 2) Depth encoder
        self.depth_encoder = DepthEncoder(in_ch=1, base_ch=32)
        # or replace with a Transformer-based approach

        # 3) Cross attention blocks (example: multi-scale or single-scale)
        # We'll do cross-attn at two scales: depth_encoder outputs [f3, f2, f1].
        # Let's fuse the deepest two:
        #   - depth f3(4*base_ch=128) with DINO mid-late layer
        #   - depth f2(64) with DINO mid-early layer
        # We'll project them to attn_embed_dim
        self.rgb_proj1 = nn.Conv2d(attn_embed_dim, attn_embed_dim, kernel_size=1)
        self.dep_proj1 = nn.Conv2d(128, attn_embed_dim, kernel_size=1)
        self.attn1 = GatedCrossAttention(attn_embed_dim, num_heads)

        self.rgb_proj2 = nn.Conv2d(attn_embed_dim, attn_embed_dim, kernel_size=1)
        self.dep_proj2 = nn.Conv2d(64, attn_embed_dim, kernel_size=1)
        self.attn2 = GatedCrossAttention(attn_embed_dim, num_heads)

        # 4) A multi-scale decoder
        # Suppose the final fused deep feature is attn_embed_dim=256,
        # next scale is also 256, shallow scale is e.g. 64 or 32. We'll keep it simple.
        self.decoder = MultiScaleDecoder(
            ch_in=[256, 256, 32],  # for example
            ch_out=[256, 128, 64]
        )

    def forward(self, rgb, depth):
        """
        rgb:   [B,3,H,W]
        depth: [B,1,H,W]
        returns: depth_pred: [B,1,H,W]
        """
        B, _, H, W = rgb.shape

        # 1) Get DINOv2 intermediate features (tokens)
        features = self.rgb_encoder.get_intermediate_layers(
            rgb, self.intermediate_idx, return_class_token=True
        )
        # features is a list of length len(self.intermediate_idx).
        # each element might be (tokens, cls_token) if return_class_token=True.

        # For illustration, let's use only 2 from the 4 features
        # (the 2 deeper ones, for instance).
        # Each tokens array is [B, N, C], where N=patch_h*patch_w (plus maybe a class token).
        # We'll pick e.g. features[-2], features[-1].
        f1_tokens, f1_cls = features[-2]  # second to last
        f2_tokens, f2_cls = features[-1]  # last

        # Convert tokens -> 2D feature maps. We need patch_h & patch_w:
        # If your input is multiple of patch size (14?), you can do:
        # patch_h = rgb.shape[2] // 14
        # patch_w = rgb.shape[3] // 14
        patch_h = rgb.shape[2] // 14
        patch_w = rgb.shape[3] // 14

        # Reshape
        # f1_tokens: [B, N, C] => [B, patch_h*patch_w, C]
        # => [B, C, patch_h, patch_w]
        f1_map = f1_tokens.permute(0,2,1).reshape(B, f1_tokens.shape[-1], patch_h, patch_w)
        f2_map = f2_tokens.permute(0,2,1).reshape(B, f2_tokens.shape[-1], patch_h, patch_w)

        # (Optional) project them to a uniform embed size for cross-attn
        # For simplicity, let's do 1×1 conv to get [B,256,H/14,W/14]
        # We'll just define them in-ch -> out-ch in the constructor
        # Actually, we haven't defined them above specifically. Let's do it here for demonstration:
        f1_map = nn.Conv2d(f1_map.shape[1], 256, 1).to(f1_map.device)(f1_map)
        f2_map = nn.Conv2d(f2_map.shape[1], 256, 1).to(f2_map.device)(f2_map)

        # 2) Depth encoder => [d3(128, H/8, W/8), d2(64, H/4, W/4), d1(32, H/2, W/2)]
        dfeats = self.depth_encoder(depth)
        d3, d2, d1 = dfeats

        # Suppose we do cross-attn:
        #   f2_map (deep, 256) with d3 (128)
        #   f1_map (mid, 256) with d2 (64)
        # Must unify shapes. f2_map is at [H/14, W/14], while d3 is [H/8, W/8].
        # We can downsample d3 to match or upsample f2_map. Let's upsample f2_map to [H/8, W/8].
        # This is somewhat approximate. The scale mismatch is a tricky point with DINO patches.
        # For a correct approach, you might ensure your input is sized so that H/14 == H/8, etc.
        # Below is just a demonstration:
        f2_map_up = F.interpolate(f2_map, size=d3.shape[-2:], mode='bilinear', align_corners=False)
        f1_map_up = F.interpolate(f1_map, size=d2.shape[-2:], mode='bilinear', align_corners=False)

        # Project depth feats to 256
        d3_proj = self.dep_proj1(d3)  # => [B,256,H/8,W/8]
        d2_proj = self.dep_proj2(d2)  # => [B,256,H/4,W/4]

        # Cross-attention
        fused_d3 = self.attn1(f2_map_up, d3_proj)
        fused_d2 = self.attn2(f1_map_up, d2_proj)

        # 3) Now we have [fused_d3(256,H/8,W/8), fused_d2(256,H/4,W/4), and maybe d1(32,H/2,W/2)]
        # We feed these to a multi-scale decoder. We'll just do 3 scales:
        #   deep => fused_d3
        #   mid  => fused_d2
        #   shallow => d1 (32)
        decoded = self.decoder([fused_d3, fused_d2, d1])  # => [B,1,H/2,W/2]

        # Upsample to full resolution if needed:
        depth_pred = F.interpolate(decoded, size=(H, W), mode='bilinear', align_corners=False)
        depth_pred = depth_pred * self.max_depth  # scale to e.g. 0..20m
        return depth_pred

    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        """
        Similar to your original infer_image, but feeding to the new pipeline.
        """
        image, (orig_h, orig_w) = self.image2tensor(raw_image, input_size)
        depth_pred = self.forward(image, torch.zeros_like(image[:,0:1]))  # if you have real depth, pass it
        depth_pred = F.interpolate(depth_pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return depth_pred[0,0].cpu().numpy()

    def image2tensor(self, raw_image, input_size=518):
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image = image.to(device, dtype=torch.float32)
        return image, (h, w)