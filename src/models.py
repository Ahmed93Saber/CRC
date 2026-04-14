import torch.nn as nn
from src.trident_load import ABMILSlideEncoder
from mammoth import Mammoth


class ClassificationModel(nn.Module):
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
        x = self.fc(x.get('features'))
        if return_raw_attention:
            features, attn = self.feature_encoder(x, return_raw_attention=True)
        else:
            features = self.feature_encoder(x)
        logits = self.classifier(features).squeeze(1)

        if return_raw_attention:
            return logits, attn

        return logits


