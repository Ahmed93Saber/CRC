import torch
import torch.nn as nn
import math
from nystrom_attention import NystromAttention


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]

        # Reshape sequence to pseudo-2D image for convolutional position encoding
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)

        # Flatten back to sequence
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransLayer(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            num_landmarks=256,  # Number of landmarks for approximation
            pinv_iterations=6,  # Moore-Penrose iterations for approximating pseudo-inverse
            residual=True,  # Residual connection on values
            dropout=dropout
        )

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out = self.attn(x_norm)
        return x + attn_out


class TransMILEncoder(nn.Module):
    def __init__(self, input_feature_dim=768, hidden_dim=512, dropout=0.1, n_heads=8):
        super().__init__()
        # Initial projection if the incoming features don't match the hidden dimension
        self.fc = nn.Linear(input_feature_dim, hidden_dim) if input_feature_dim != hidden_dim else nn.Identity()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.layer1 = TransLayer(dim=hidden_dim, num_heads=n_heads, dropout=dropout)
        self.pos_layer = PPEG(dim=hidden_dim)
        self.layer2 = TransLayer(dim=hidden_dim, num_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, return_raw_attention=False):
        if isinstance(x, dict):
            x = x['features']

        x = self.fc(x)
        B, N, C = x.shape

        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.layer1(x)

        # Calculate dimensions for PPEG and pad if sequence length is not a perfect rectangle
        H = int(math.ceil(math.sqrt(N)))
        W = int(math.ceil(N / H))
        pad_len = H * W - N

        if pad_len > 0:
            pad = torch.zeros(B, pad_len, C, device=x.device)
            x = torch.cat((x, pad), dim=1)

        x = self.pos_layer(x, H, W)

        # Strip padding after PPEG
        if pad_len > 0:
            x = x[:, :-(pad_len)]

        x = self.layer2(x)
        x = self.norm(x)

        # Aggregate bag representations using the class token
        features = x[:, 0]

        if return_raw_attention:
            return features, None

        return features