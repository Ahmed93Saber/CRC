import torch.nn as nn
from src.trident_load import ABMILSlideEncoder
from mammoth import Mammoth


import torch.nn as nn
from src.trident_load import ABMILSlideEncoder
from src.transmil import TransMILEncoder
from mammoth import Mammoth


class ClassificationModel(nn.Module):
    def __init__(self, input_feature_dim=768, moe_args=None, n_heads=1, head_dim=512, dropout=0.,
                 gated=True, hidden_dim=256, output_dim=2, encoder_type="abmil"):
        super().__init__()
        self.encoder_type = encoder_type.lower()

        if moe_args is None:
            moe_args = {}
        if moe_args and moe_args.get('num_experts', 0) > 0:
            self.fc = Mammoth(**moe_args)
        else:
            self.fc = nn.Linear(input_feature_dim, hidden_dim)

        if self.encoder_type == "abmil":
            self.feature_encoder = ABMILSlideEncoder(
                freeze=False,
                input_feature_dim=hidden_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                dropout=dropout,
                gated=gated
            )
        elif self.encoder_type == "transmil":
            # TransMIL typically benefits from multiple heads in self-attention
            transmil_heads = 8 if n_heads == 1 else n_heads
            self.feature_encoder = TransMILEncoder(
                input_feature_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                n_heads=transmil_heads
            )
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, return_raw_attention=False):
        x_feats = x.get('features') if isinstance(x, dict) else x
        x_proj = self.fc(x_feats)

        if return_raw_attention:
            features, attn = self.feature_encoder(x_proj, return_raw_attention=True)
        else:
            features = self.feature_encoder(x_proj)

        logits = self.classifier(features).squeeze(1)

        if return_raw_attention:
            return logits, attn

        return logits


class BinaryClassificationModel(nn.Module):
    def __init__(self, input_feature_dim=768, n_heads=1, head_dim=512, dropout=0.,
                 gated=True, hidden_dim=256, output_dim=2, encoder_type="abmil"):
        super().__init__()
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "abmil":
            self.feature_encoder = ABMILSlideEncoder(
                freeze=False,
                input_feature_dim=input_feature_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                dropout=dropout,
                gated=gated
            )
        elif self.encoder_type == "transmil":
            transmil_heads = 8 if n_heads == 1 else n_heads
            self.feature_encoder = TransMILEncoder(
                input_feature_dim=input_feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                n_heads=transmil_heads
            )
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        clf_input_dim = hidden_dim if self.encoder_type == "transmil" else input_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, return_raw_attention=False):
        x_feats = x.get('features') if isinstance(x, dict) else x

        if return_raw_attention:
            features, attn = self.feature_encoder(x_feats, return_raw_attention=True)
        else:
            features = self.feature_encoder(x_feats)

        logits = self.classifier(features).squeeze(1)

        if return_raw_attention:
            return logits, attn

        return logits


class ClassicClassificationModel(nn.Module):
    def __init__(self, input_feature_dim=768, moe_args=None, n_heads=1, head_dim=512, dropout=0.,
                 gated=True, hidden_dim=256, output_dim=2):
        super().__init__()

        if moe_args is None:
            moe_args = {}
        if moe_args and moe_args.get('num_experts', 0) > 0:
            self.fc = Mammoth(**moe_args)
        else:
            self.fc = nn.Linear(input_feature_dim, hidden_dim)

        self.feature_encoder = ABMILSlideEncoder(
            freeze=False,
            input_feature_dim=hidden_dim,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=gated
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, return_raw_attention=False):
        x = self.fc(x)
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)
        logits = self.classifier(features).squeeze(1)

        if return_raw_attention:
            return logits, attn

        return logits





# class BinaryClassificationModel(nn.Module):
#     def __init__(self, input_feature_dim=768, n_heads=1, head_dim=512, dropout=0.,
#                  gated=True, hidden_dim=256, output_dim=2):
#         super().__init__()
#         self.feature_encoder = ABMILSlideEncoder(
#             freeze=False,
#             input_feature_dim=input_feature_dim,
#             n_heads=n_heads,
#             head_dim=head_dim,
#             dropout=dropout,
#             gated=gated
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(input_feature_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )
#
#     def forward(self, x, return_raw_attention=False):
#         if return_raw_attention:
#             features, attn = self.feature_encoder(x, return_raw_attention=True)
#         else:
#             features = self.feature_encoder(x)
#         logits = self.classifier(features).squeeze(1)
#
#         if return_raw_attention:
#             return logits, attn
#
#         return logits

