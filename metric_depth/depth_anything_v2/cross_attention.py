import torch
import torch.nn as nn
import torch.nn.functional as F
import math

##############################
# Positional Encoding Module #
##############################
def get_2d_sincos_pos_embed(embed_dim, H, W, device):
    """
    Generate 2D sine-cosine positional embeddings.
    Returns a tensor of shape [H*W, embed_dim].
    """
    grid_y, grid_x = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                                    torch.arange(W, dtype=torch.float32, device=device), indexing='ij')
    grid = torch.stack([grid_y, grid_x], dim=-1)  # [H, W, 2]
    grid = grid.reshape(-1, 2)  # [H*W, 2]

    # Compute the positional embedding for each coordinate.
    assert embed_dim % 4 == 0, "Embed dim must be divisible by 4 for sine-cosine encoding."
    dim_each = embed_dim // 2
    omega = torch.arange(dim_each // 2, dtype=torch.float32, device=device)
    omega = 1. / (10000 ** (omega / (dim_each // 2)))

    # Expand grid coordinates to match frequency dimensions.
    pos_x = grid[:, 1:2] * omega  # [H*W, dim_each//2]
    pos_y = grid[:, 0:2] * omega  # [H*W, dim_each//2]

    # Compute sine and cosine embeddings.
    pos_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=1)  # [H*W, dim_each]
    pos_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=1)  # [H*W, dim_each]

    pos_embed = torch.cat([pos_y, pos_x], dim=1)  # [H*W, embed_dim]
    return pos_embed

######################################
# Fusion Transformer Block Component #
######################################
class FusionTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # Cross-attention: queries come from RGB tokens; keys/values from depth tokens.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

        # Self-attention on the fused tokens.
        self.norm2 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

        # Feed-forward network.
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value):
        # Cross-attention with residual connection.
        q_norm = self.norm1(query)
        cross_out, _ = self.cross_attn(query=q_norm, key=key, value=value)
        query = query + cross_out

        # Self-attention on the updated query tokens.
        q2 = self.norm2(query)
        self_out, _ = self.self_attn(query=q2, key=q2, value=q2)
        query = query + self_out

        # Feed-forward network.
        q3 = self.norm3(query)
        ffn_out = self.ffn(q3)
        output = query + ffn_out
        return output

##############################################################
# Extended Cross-Attention Depth Completion (Version 2) Model #
##############################################################
class CrossAttentionDepthCompletionV2(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_fusion_layers=3, dropout=0.1):
        super().__init__()
        # --- Encoders ---
        # RGB encoder: extract features from the RGB image.
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        # Depth encoder: extract features from the sparse depth image.
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # --- Fusion Transformer Blocks ---
        self.fusion_blocks = nn.ModuleList([
            FusionTransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_fusion_layers)
        ])

        # --- Decoder ---
        # A simple decoder that upsamples the fused features to produce a dense depth map.
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 1, kernel_size=1)
        )

    def forward(self, rgb, sparse_depth, depth_mask):
        """
        Args:
            rgb:         [B, 3, H, W] RGB image.
            sparse_depth:[B, 1, H, W] Sparse depth image (with zeros or dummy values where missing).
            depth_mask:  [B, 1, H, W] Binary mask (1 for valid depth, 0 for missing).
        """
        device = rgb.device
        # --- Feature Extraction ---
        rgb_feat = self.rgb_encoder(rgb)  # [B, C, H, W]
        depth_feat = self.depth_encoder(sparse_depth)  # [B, C, H, W]
        # Zero out invalid depth features.
        depth_feat = depth_feat * depth_mask

        B, C, H, W = rgb_feat.shape

        # --- Flatten Spatial Dimensions ---
        # Convert to token sequences of shape [B, H*W, C]
        rgb_tokens = rgb_feat.view(B, C, -1).transpose(1, 2)
        depth_tokens = depth_feat.view(B, C, -1).transpose(1, 2)

        # --- Add Positional Encodings ---
        pos_embed = get_2d_sincos_pos_embed(C, H, W, device)  # [H*W, C]
        # Expand for batch dimension.
        pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)
        rgb_tokens = rgb_tokens + pos_embed
        depth_tokens = depth_tokens + pos_embed

        # --- Fusion via Transformer Blocks ---
        # Here, we iteratively refine the RGB tokens by attending to depth tokens.
        for block in self.fusion_blocks:
            rgb_tokens = block(rgb_tokens, depth_tokens, depth_tokens)

        # --- Reshape and Fuse ---
        # Reshape refined tokens back to spatial feature maps.
        fused_feat = rgb_tokens.transpose(1, 2).view(B, C, H, W)
        # Fuse with original RGB features via a residual connection.
        fused_feat = fused_feat + rgb_feat

        # --- Decode to Dense Depth ---
        dense_depth = self.decoder(fused_feat)
        return dense_depth

# #####################################
# # Example Usage of the Extended Model
# #####################################
# if __name__ == '__main__':
#     B, H, W = 2, 256, 256
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Create dummy inputs.
#     rgb = torch.randn(B, 3, H, W, device=device)
#     sparse_depth = torch.randn(B, 1, H, W, device=device)
#     # Simulate a binary mask with ~30% valid depth values.
#     depth_mask = (torch.rand(B, 1, H, W, device=device) > 0.7).float()

#     model = CrossAttentionDepthCompletionV2(embed_dim=64, num_heads=4, num_fusion_layers=3, dropout=0.1).to(device)
#     output = model(rgb, sparse_depth, depth_mask)
#     print("Output dense depth shape:", output.shape)
