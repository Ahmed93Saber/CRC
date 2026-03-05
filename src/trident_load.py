import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from typing import Optional, Tuple, Dict, Any
from abc import abstractmethod


class ABMIL(nn.Module):
    """
    Multi-headed attention network with optional gating. Uses tanh-attention and sigmoid-gating as in ABMIL (https://arxiv.org/abs/1802.04712).
    Note that this is different from canonical attention in that the attention scores are computed directly by a linear layer rather than by a dot product between queries and keys.

    Args:
        feature_dim (int): Input feature dimension
        head_dim (int): Hidden layer dimension for each attention head. Defaults to 256.
        n_heads (int): Number of attention heads. Defaults to 8.
        dropout (float): Dropout probability. Defaults to 0.
        n_branches (int): Number of attention branches. Defaults to 1, but can be set to n_classes to generate one set of attention scores for each class.
        gated (bool): If True, sigmoid gating is applied. Otherwise, the simple attention mechanism is used.
    """

    def __init__(self, feature_dim=1024, head_dim=256, n_heads=8, dropout=0., n_branches=1, gated=False):
        super().__init__()
        self.gated = gated
        self.n_heads = n_heads

        # Initialize attention head(s)
        self.attention_heads = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, head_dim),
                                                            nn.Tanh(),
                                                            nn.Dropout(dropout)) for _ in range(n_heads)])

        # Initialize gating layers if gating is used
        if self.gated:
            self.gating_layers = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, head_dim),
                                                              nn.Sigmoid(),
                                                              nn.Dropout(dropout)) for _ in range(n_heads)])

        # Initialize branching layers
        self.branching_layers = nn.ModuleList([nn.Linear(head_dim, n_branches) for _ in range(n_heads)])

        # Initialize condensing layer if multiple heads are used
        if n_heads > 1:
            self.condensing_layer = nn.Linear(n_heads * feature_dim, feature_dim)

    def forward(self, features, attn_mask=None):
        """
        Forward pass

        Args:
            features (torch.Tensor): Input features, acting as queries and values. Shape: batch_size x num_images x feature_dim
            attn_mask (torch.Tensor): Attention mask to enforce zero attention on empty images. Defaults to None. Shape: batch_size x num_images

        Returns:
            aggregated_features (torch.Tensor): Attention-weighted features aggregated across heads. Shape: batch_size x n_branches x feature_dim
        """

        assert features.dim() == 3, f'Input features must be 3-dimensional (batch_size x num_images x feature_dim). Got {features.shape} instead.'
        if attn_mask is not None:
            assert attn_mask.dim() == 2, f'Attention mask must be 2-dimensional (batch_size x num_images). Got {attn_mask.shape} instead.'
            assert features.shape[
                       :2] == attn_mask.shape, f'Batch size and number of images must match between features and mask. Got {features.shape[:2]} and {attn_mask.shape} instead.'

        # Get attention scores for each head
        head_attentions = []
        head_features = []
        for i in range(len(self.attention_heads)):
            attention_vectors = self.attention_heads[i](
                features)  # Main attention vectors (shape: batch_size x num_images x head_dim)

            if self.gated:
                gating_vectors = self.gating_layers[i](
                    features)  # Gating vectors (shape: batch_size x num_images x head_dim)
                attention_vectors = attention_vectors.mul(
                    gating_vectors)  # Element-wise multiplication to apply gating vectors

            attention_scores = self.branching_layers[i](
                attention_vectors)  # Attention scores for each branch (shape: batch_size x num_images x n_branches)

            # Set attention scores for empty images to -inf
            if attn_mask is not None:
                attention_scores = attention_scores.masked_fill(~attn_mask.unsqueeze(-1),
                                                                -1e9)  # Mask is automatically broadcasted to shape: batch_size x num_images x n_branches

            # Softmax attention scores over num_images
            attention_scores_softmax = F.softmax(attention_scores, dim=1)  # Shape: batch_size x num_images x n_branches

            # Multiply features by attention scores
            weighted_features = torch.einsum('bnr,bnf->brf', attention_scores_softmax,
                                             features)  # Shape: batch_size x n_branches x feature_dim

            head_attentions.append(attention_scores)
            head_features.append(weighted_features)

        # Concatenate multi-head outputs and condense
        aggregated_features = torch.cat(head_features,
                                        dim=-1)  # Shape: batch_size x n_branches x (n_heads * feature_dim)
        if self.n_heads > 1:
            aggregated_features = self.condensing_layer(
                aggregated_features)  # Shape: batch_size x n_branches x feature_dim

        # Stack attention scores
        head_attentions = torch.stack(head_attentions, dim=-1)  # Shape: batch_size x num_images x n_branches x n_heads
        head_attentions = rearrange(head_attentions,
                                    'b n r h -> b r h n')  # Shape: batch_size x n_branches x n_heads x num_images

        return aggregated_features, head_attentions


class BaseSlideEncoder(torch.nn.Module):

    def __init__(self, freeze: bool = True, **build_kwargs: Dict[str, Any]) -> None:
        """
        Parent class for all pretrained slide encoders.
        """
        super().__init__()
        self.enc_name = None
        self.model, self.precision, self.embedding_dim = self._build(**build_kwargs)

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Can be overwritten if model requires special forward pass.
        """
        z = self.model(batch)
        return z

    @abstractmethod
    def _build(self, **build_kwargs: Dict[str, Any]) -> Tuple[torch.nn.Module, torch.dtype, int]:
        """
        Initialization method, must be defined in child class.
        """
        pass


class ABMILSlideEncoder(BaseSlideEncoder):

    def __init__(self, **build_kwargs: Dict[str, Any]):
        """
        ABMIL initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
            self,
            input_feature_dim: int,
            n_heads: int,
            head_dim: int,
            dropout: float,
            gated: bool,
            pretrained: bool = False
    ) -> Tuple[torch.nn.ModuleDict, torch.dtype, int]:


        self.enc_name = 'abmil'

        assert pretrained is False, "ABMILSlideEncoder has no corresponding pretrained models. Please load with pretrained=False."

        pre_attention_layers = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        image_pooler = ABMIL(
            n_heads=n_heads,
            feature_dim=input_feature_dim,
            head_dim=head_dim,
            dropout=dropout,
            n_branches=1,
            gated=gated
        )

        post_attention_layers = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        model = nn.ModuleDict({
            'pre_attention_layers': pre_attention_layers,
            'image_pooler': image_pooler,
            'post_attention_layers': post_attention_layers
        })

        precision = torch.float32
        embedding_dim = input_feature_dim
        return model, precision, embedding_dim

    def forward(self, batch, device='cuda', return_raw_attention=False):
        image_features = self.model['pre_attention_layers'](batch['features'].to(device))
        image_features, attn = self.model['image_pooler'](
            image_features)  # Features shape: (b n_branches f), where n_branches = 1. Branching is not used in this implementation.
        image_features = rearrange(image_features, 'b 1 f -> b f')
        image_features = self.model['post_attention_layers'](
            image_features)  # Attention scores shape: (b r h n), where h is number of attention heads
        if return_raw_attention:
            return image_features, attn
        return image_features