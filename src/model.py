# ViT (Vision Transformer) Model

# This implementation defines the Vision Transformer (ViT) model architecture. ViT is a transformer-based model
# for image classification, where the input image is divided into fixed-size patches and processed using a
# transformer architecture. The model consists of a Patch Embedding layer, a stack of Transformer Encoder Blocks,
# and a Classification Head.

# The code includes classes for Patch Embedding, Multi-Head Attention, Residual Addition, FeedForward Block,
# Transformer Encoder Block, Transformer Encoder, Classification Head, and the overall ViT model.

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    """
    Patch Embedding class for Vision Transformer (ViT) model.

    Args:
        in_channels (int): Number of input channels in the image.
        patch_size (int): Size of each patch in the image.
        emb_size (int): Dimensionality of the embedding for each patch.
        img_size (int): Size of the input image.

    Attributes:
        patch_size (int): Size of each patch in the image.
        projection (nn.Sequential): Sequential module for patch embedding.
        cls_token (nn.Parameter): Learnable parameter representing the class token.
        positions (nn.Parameter): Learnable parameter representing the positions of the patches.

    Methods:
        forward(x: Tensor) -> Tensor:
            Forward pass through the patch embedding layer.

    """  
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super().__init__()  # Move this line up
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
    
    def forward(self, x: Tensor):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention class for Vision Transformer (ViT) model.

    Args:
        emb_size (int): Dimensionality of the embedding for each token.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.

    Attributes:
        emb_size (int): Dimensionality of the embedding for each token.
        num_heads (int): Number of attention heads.
        keys (nn.Linear): Linear layer for computing keys.
        queries (nn.Linear): Linear layer for computing queries.
        values (nn.Linear): Linear layer for computing values.
        att_drop (nn.Dropout): Dropout layer for attention weights.
        projection (nn.Linear): Linear layer for projecting output.
        scaling (float): Scaling factor for attention scores.

    Methods:
        forward(x: Tensor, mask: Tensor = None) -> Tensor:
            Forward pass through the multi-head attention layer.

    """  
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    """
    Residual Addition module for adding residual connections.

    Args:
        fn: Sub-module to which the residual connection is added.

    Methods:
        forward(x, **kwargs) -> Tensor:
            Forward pass through the residual addition layer.

    """  
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

  class FeedForwardBlock(nn.Sequential):
    """
    FeedForward Block class for Vision Transformer (ViT) model.

    Args:
        emb_size (int): Dimensionality of the embedding for each token.
        expansion (int): Expansion factor for the feedforward block.
        drop_p (float): Dropout probability.

    """    
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(nn.Linear(emb_size, expansion * emb_size),
                         nn.GELU(),
                         nn.Dropout(drop_p),
                         nn.Linear(expansion * emb_size, emb_size))

class TransformerEncoderBlock(nn.Sequential):
    """
    Transformer Encoder Block class for Vision Transformer (ViT) model.

    Args:
        emb_size (int): Dimensionality of the embedding for each token.
        drop_p (float): Dropout probability.
        forward_expansion (int): Expansion factor for the feedforward block.
        forward_drop_p (float): Dropout probability for the feedforward block.
        **kwargs: Additional arguments passed to MultiHeadAttention.

    """  
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0,
                 ** kwargs):
        super().__init__(ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size),
                                                   MultiHeadAttention(emb_size, **kwargs),
                                                   nn.Dropout(drop_p))),
                         ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size),
                                                   FeedForwardBlock(emb_size,
                                                                    expansion=forward_expansion,
                                                                    drop_p=forward_drop_p),
                                                   nn.Dropout(drop_p))))

class TransformerEncoder(nn.Sequential):
    """
    Transformer Encoder class for stacking Transformer Encoder Blocks.

    Args:
        depth (int): Number of Transformer Encoder Blocks to stack.
        **kwargs: Additional arguments passed to TransformerEncoderBlock.

    """  
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    """
    Classification Head class for Vision Transformer (ViT) model.

    Args:
        emb_size (int): Dimensionality of the embedding for each token.
        n_classes (int): Number of output classes.

    """  
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(Reduce('b n e -> b e', reduction='mean'),
                         nn.LayerNorm(emb_size), 
                         nn.Linear(emb_size, n_classes))

class ViT(nn.Sequential):
    """
    Vision Transformer (ViT) model class.

    Args:
        in_channels (int): Number of input channels in the image.
        patch_size (int): Size of each patch in the image.
        emb_size (int): Dimensionality of the embedding for each patch.
        img_size (int): Size of the input image.
        depth (int): Number of Transformer Encoder Blocks in the model.
        n_classes (int): Number of output classes.
        **kwargs: Additional arguments passed to PatchEmbedding and TransformerEncoder.

    """  
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
                  
        super().__init__(PatchEmbedding(in_channels,
                                        patch_size,
                                        emb_size,
                                        img_size),
                         TransformerEncoder(depth,
                                            emb_size=emb_size,
                                            **kwargs),
                         ClassificationHead(emb_size,
                                            n_classes))
 