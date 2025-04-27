import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from models.convnext import DropPath, LayerNorm

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x

class Downsample(nn.Module):
    def __init__(self,
                 dim,
                 keep_dim=False
                 ):
        pass


    def forward(self, x: Tensor) -> Tensor:
        pass


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_chans=3,
                 in_dim=64,
                 dim=96
                 ):
        pass


    def forward(self, x: Tensor) -> Tensor:
        pass


class ConvBlock(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=0.0,
                 layer_scale=None,
                 kernel_size=3
                 ):
        pass


    def forward(self, x: Tensor) -> Tensor:
        pass


class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
        ):
        pass

    def forward(self, hidden_states):
        pass


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0
                 ):
        pass


    def forward(self, x: Tensor) -> Tensor:
        pass


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass


class MambaVisionLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        pass


    def forward(self, x: Tensor) -> Tensor:
        pass


class MambaVision(nn.Module):
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        pass


    def _init_weights(self, m):
        pass


    def forward_features(self, x: Tensor) -> Tensor:
        pass


    def forward(self, x: Tensor) -> Tensor:
        pass


    def _load(self,
              pretrained: bool=False
              ):
        pass

