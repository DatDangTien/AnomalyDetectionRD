import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from models.convnext import DropPath, LayerNorm
from models.convnext import Block as ConvNeXtBlock
from models.resnet import AttnBottleneck
import math
from functools import partial
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from typing import Type, Any, List, Tuple



model_urls = {
    'mambavision_t' : 'https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar',
    'mambavision_s' : 'https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar',
    'mambavision_b' : 'https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',
    'mambavision_l' : 'https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar',
    'mambavision_b21k': 'https://huggingface.co/nvidia/MambaVision-B-21K/resolve/main/mambavision_base_21k.pth.tar',
    'mambavision_l21k': 'https://huggingface.co/nvidia/MambaVision-L-21K/resolve/main/mambavision_large_21k.pth.tar',
}


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
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=3, stride=2, padding=1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduction(x)
        return x

class Upsample(nn.Module):
    def __init__(self,
                 dim,
                 keep_dim=False
                 ):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = dim // 2
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim_out, kernel_size=3, stride=2, padding=1, output_padding=1),
            # LayerNorm(dim_out),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim_out, kernel_size=3, stride=2, padding=1),
            # LayerNorm(dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Special layer 4,4 -> 7,7
        if x.shape[2] == 4:
            x = self.upsample2(x)
        else:
            x = self.upsample1(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_chans=3,
                 in_dim=64,
                 dim=96
                 ):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(),
        )


    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=0.0,
                 layer_scale=None,
                 kernel_size=3
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


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
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )


    def forward(self, hidden_states):
        """
        :param hidden_states: (B, L, D)
        :return: same shape as hidden_states
        """
        # print('Mixer', hidden_states.shape)
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        # print('x,z', x.shape, z.shape)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias,
                            padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias,
                            padding='same', groups=self.d_inner//2))
        x_db1 = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        # print('x_db1', x_db1.shape)
        dt, B, C = torch.split(x_db1, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)
        # print('y', y.shape)
        y = torch.cat([y,z], dim=1)
        # print('y_cat', y.shape)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        # print('Mixer_out', out.shape)
        return out

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.q_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k ,v,
                dropout_p=self.attn_drop.p
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2). reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias,) * 2 if not isinstance(bias, (tuple, list)) else bias
        drop_probs = (drop,) * 2 if not isinstance(drop, (tuple, list)) else drop
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


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
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=3,
                expand=1,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



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
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """
        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([
                ConvBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale_conv,
                ) for i in range(depth)
            ])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=dim,
                    counter=i,
                    transformer_blocks=transformer_blocks,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale,
                ) for i in range(depth)
            ])
            self.transformer_block = True

        self.downsample = Downsample(dim=dim) if downsample else None
        self.do_gt = False
        self.window_size = window_size


    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape

        # Window transform
        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size)  % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x , (0, pad_r, 0, pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            # print('Pad:', self.window_size, Hp, Wp)
            x = window_partition(x, self.window_size)
        # print('Window:', x.shape)
        # Feature extract
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        # print('Block:', x.shape)

        # Reverse window transform
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        x = self.downsample(x) if self.downsample is not None else x
        # print('Downsample:', x.shape)
        return x

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
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            in_dim=in_dim,
            dim=dim
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList([])
        for i in range(len(depths)):
            conv = True if (i < 2) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 3),
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(math.ceil(depths[i] / 2),depths[i])),
                                     )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


    def forward_features(self, x: Tensor) -> List[Tensor]:
        """
        Mambavison-B:
        Dim: 3,224,224 -> 128,56,56 -> [256,28,26 -> 512,14,14 -> 1024,7,7->1024,7,7]
        Last layer has no downsample.
        """


        feature = []
        x = self.patch_embed(x)
        # print('----------------')
        # print(x.shape)
        for level in self.levels[:3]:
            x = level(x)
            # print(x.shape)
            feature.append(x)
        # print('----------------')
        # x = self.norm(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # print([f.shape for f in feature])
        return feature


    def forward(self, x: Tensor) -> List[Tensor]:
        # x = self.forward_features(x)
        # x = self.head(x)
        # return x
        return self.forward_features(x)


    def load(self,
              model_url: str
              ):
        state_dict = load_state_dict_from_url(model_url)
        # print(state_dict['state_dict'].keys())
        self.load_state_dict(state_dict['state_dict'])


class BN_layer(nn.Module):
    def __init__(self,
                 dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 num_stages = 3,
                 drop_path_rate=0.2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        super().__init__()
        self.mff = nn.ModuleList()
        norm_eps = 1e-5
        for i in range(num_stages):
            conv = []
            for j in range(i + 1, num_stages):
                conv.append(nn.Conv2d(dim * 2 ** j, dim * 2 ** (j+1),
                                      kernel_size=3, stride=2, padding=1))
                conv.append(LayerNorm(dim * 2 ** (j+1), eps=norm_eps))
                conv.append(nn.GELU())
            self.mff.append(nn.Sequential(*conv))

        # C = 1024 * 3 -> 2048
        self.downsample = nn.Sequential(
            nn.Conv2d(dim * (2 ** num_stages) * num_stages, dim * 2 ** (num_stages + 1), kernel_size=3, stride=2, padding=1),
            LayerNorm(dim * 2 ** (num_stages + 1), eps=norm_eps),
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths[-1])]
        self.oce = nn.Sequential(
            *[ConvNeXtBlock(
                dim=dim * 2 ** (num_stages + 1),
                drop_path=dpr[i],
                layer_scale_init_value=0 if layer_scale_conv is None else layer_scale_conv,
                norm_eps=norm_eps,
            ) for i in range (depths[-1])]
        )

    # Multiscale feature fusion
    def forward(self, x: List[Tensor]) -> Tensor:
        # print('BN_________________')
        x = [self.mff[i](x[i]) for i in range(len(x))]
        x = torch.cat(x, dim=1)
        # print(x.shape)
        x = self.downsample(x)
        # print(x.shape)
        x = self.oce(x)
        # print(x.shape)
        return x


class BN_layer_resnet(nn.Module):
    def __init__(self,
                 dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 num_stages = 3,
                 drop_path_rate=0.2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        super().__init__()
        self.mff = nn.ModuleList()
        norm_eps = 1e-5
        for i in range(num_stages):
            conv = []
            for j in range(i + 1, num_stages):
                conv.append(nn.Conv2d(dim * 2 ** j, dim * 2 ** (j+1),
                                      kernel_size=3, stride=2, padding=1))
                conv.append(LayerNorm(dim * 2 ** (j+1), eps=norm_eps))
                conv.append(nn.GELU())
            self.mff.append(nn.Sequential(*conv))

        self.norm = nn.BatchNorm2d
        downsample = nn.Sequential(
            nn.Conv2d(dim * (2 ** num_stages) * num_stages, dim * 2 ** (num_stages + 1), kernel_size=1, stride=2),
            self.norm(dim * 2 ** (num_stages + 1))
        )
        # C = 1024 * 3 -> 1024 * 2

        layers = []
        layers.append(AttnBottleneck(dim * (2 ** num_stages) * num_stages, #1024 * 3
                                     planes=dim * 2 ** (num_stages - 1),    #512
                                     stride=2,
                                     downsample=downsample,
                                     base_width=128,
                                     norm_layer=self.norm))
        for _ in range(1, depths[-1]):
            layers.append(AttnBottleneck(dim * 2 ** (num_stages + 1),       #2048
                                         planes=dim * 2 ** (num_stages - 1), #512
                                         base_width=128,
                                         norm_layer=self.norm))
        self.oce = nn.Sequential(*layers)


    # Multiscale feature fusion
    def forward(self, x: List[Tensor]) -> Tensor:
        # print('BN_________________')
        x = [self.mff[i](x[i]) for i in range(len(x))]
        x = torch.cat(x, dim=1)
        # print('MFF ',x.shape)
        x = self.oce(x)
        # print('OCE ',x.shape)
        return x

class BN_layer_mamba(nn.Module):
    def __init__(self,
                 dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 num_stages = 3,
                 drop_path_rate=0.2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        super().__init__()
        self.mff = nn.ModuleList()
        norm_eps = 1e-5
        for i in range(num_stages):
            conv = []
            for j in range(i + 1, num_stages):
                conv.append(nn.Conv2d(dim * 2 ** j, dim * 2 ** (j+1),
                                      kernel_size=3, stride=2, padding=1))
                conv.append(LayerNorm(dim * 2 ** (j+1), eps=norm_eps))
                conv.append(nn.GELU())
            self.mff.append(nn.Sequential(*conv))

        # C = 1024 * 3 -> 1024
        self.downsample = nn.Sequential(
            nn.Conv2d(dim * (2 ** num_stages) * num_stages, dim * 2 ** num_stages, kernel_size=3, stride=2, padding=1),
            LayerNorm(dim * 2 ** (num_stages + 1), eps=norm_eps),
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths[-1])]
        self.oce = nn.Sequential(
            *[MambaVisionLayer(
                dim=int(dim * 2 ** num_stages),
                depth=depths[-1],
                num_heads=num_heads[-1],
                window_size=window_size[-1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=False,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                # drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=False,
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=list(range(math.ceil(depths[-1] / 2), depths[-1])),
                ) for _ in range(depths[-1])]
        )


    # Multiscale feature fusion
    def forward(self, x: List[Tensor]) -> Tensor:
        # print('BN_________________')
        x = [self.mff[i](x[i]) for i in range(len(x))]
        x = torch.cat(x, dim=1)
        print('MFF: ', x.shape)
        x = self.downsample(x)
        print('Downsample: ', x.shape)
        x = self.oce(x)
        print('OCE: ', x.shape)
        return x


class DeMambaVisionLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 upsample=True,
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
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """
        super().__init__()
        self.upsample = Upsample(dim=dim) if upsample else None
        dim = dim // 2 if upsample else dim
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([
                ConvBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale_conv,
                ) for i in range(depth)
            ])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([
                Block(
                    dim=dim,
                    counter=i,
                    transformer_blocks=transformer_blocks,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    layer_scale=layer_scale,
                ) for i in range(depth)
            ])
            self.transformer_block = True

        self.do_gt = False
        self.window_size = window_size


    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x) if self.upsample is not None else x

        _, _, H, W = x.shape

        # Window transform
        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size)  % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = F.pad(x , (0, pad_r, 0, pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            # print('Pad:', self.window_size, Hp, Wp)
            x = window_partition(x, self.window_size)
        # print('Window:', x.shape)
        # Feature extract
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        # print('Block:', x.shape)

        # Reverse window transform
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        return x

class DeMambaVision(nn.Module):
    def __init__(self,
                 dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        print(depths)
        # Remove final stage
        depths = depths[:-1]
        # num_heads = num_heads[:-1]
        # window_size = window_size[:-1]
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths))]
        self.levels = nn.ModuleList([])
        for i in range(len(depths)):
            conv = True if (i < 2) else False
            level = DeMambaVisionLayer(
                    # dim=int(dim * 2 ** (len(depths) - i + 1)) if i > 0  else (dim * 2 ** len(depths)),
                    dim=int(dim * 2 ** (len(depths) - i + 1)),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    conv=conv,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    # drop_path=dpr[sum(depths[:i+1])-1:sum(depths[:i])-1 : -1],
                    # drop_path=dpr[sum(depths[:i]): sum(depths[:i+1])][::-1],
                    drop_path=dpr[sum(depths[:i]): sum(depths[:i+1])],
                    upsample=(i > 0),
                    layer_scale=layer_scale,
                    layer_scale_conv=layer_scale_conv,
                    transformer_blocks=list(range(math.ceil(depths[i] / 2),depths[i])),
                    )
            self.levels.append(level)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


    def forward_features(self, x: Tensor) -> List[Tensor]:
        """
        Mambavison-B:
        Dim: 3,224,224 -> 128,56,56 -> [256,28,26 -> 512,14,14 -> 1024,7,7->1024,7,7]
        Last layer has no downsample.
        """
        feature = []
        # print('----------------')
        # print(x.shape)
        # feature.append(x)
        for level in self.levels[:3]:
            x = level(x)
            # print(x.shape)
            feature.append(x)
        # print('----------------')

        # print([f.shape for f in feature[::-1]])
        return feature[::-1]


    def forward(self, x: Tensor) -> List[Tensor]:
        return self.forward_features(x)


def mambavision_t(
        pretrained=False,
        **kwargs
):
    depths = kwargs.pop('depths', [1, 3, 8, 4])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 80)
    in_dim = kwargs.pop("in_dim", 32)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    model = MambaVision(
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dim=dim,
        in_dim=in_dim,
        mlp_ratio=mlp_ratio,
        resolution=resolution,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    if pretrained:
        model.load(model_urls['mambavision_t'])
    bn = BN_layer(
        dim=dim,
        depths=depths,
        window_size=window_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
    )
    return model, bn

def mambavision_s(
        pretrained=False,
        **kwargs
):
    depths = kwargs.pop("depths", [3, 3, 7, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 96)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    model = MambaVision(
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dim=dim,
        in_dim=in_dim,
        mlp_ratio=mlp_ratio,
        resolution=resolution,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    if pretrained:
        model.load(model_urls['mambavision_s'])

    bn = BN_layer_mamba(
        dim=dim,
        depths=depths,
        window_size=window_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
    )
    return model, bn

def mambavision_b(
        pretrained=False,
        **kwargs
):
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 128)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    model = MambaVision(
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dim=dim,
        in_dim=in_dim,
        mlp_ratio=mlp_ratio,
        resolution=resolution,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
        **kwargs
    )
    if pretrained:
        model.load(model_urls['mambavision_b'])
    bn = BN_layer_mamba(
        dim=dim,
        depths=depths,
        window_size=window_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
    )

    return model, bn

def mambavision_l(
        pretrained=False,
        **kwargs
):
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    model = MambaVision(
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dim=dim,
        in_dim=in_dim,
        mlp_ratio=mlp_ratio,
        resolution=resolution,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
        **kwargs
    )
    if pretrained:
        model.load(model_urls['mambavision_l'])
    bn = BN_layer(
        dim=dim,
        depths=depths,
        window_size=window_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
    )
    return model

def mambavision_b21k(
        pretrained=False,
        **kwargs
):
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 128)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    model = MambaVision(
        # num_classes=21841,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dim=dim,
        in_dim=in_dim,
        mlp_ratio=mlp_ratio,
        resolution=resolution,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
        **kwargs
    )
    if pretrained:
        model.load(model_urls['mambavision_b21k'])
    bn = BN_layer(
        dim=dim,
        depths=depths,
        window_size=window_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
    )
    return model, bn

def mambavision_l21k(
        pretrained=False,
        **kwargs
):
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    model = MambaVision(
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dim=dim,
        in_dim=in_dim,
        mlp_ratio=mlp_ratio,
        resolution=resolution,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
        **kwargs
    )
    if pretrained:
        model.load(model_urls['mambavision_l21k'])
    bn = BN_layer(
        dim=dim,
        depths=depths,
        window_size=window_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
    )

    return model, bn


def demambavision_t(
        pretrained=False,
        **kwargs
):
    depths = kwargs.pop('depths', [1, 3, 8, 4])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 80)
    in_dim = kwargs.pop("in_dim", 32)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    model = DeMambaVision(
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dim=dim,
        in_dim=in_dim,
        mlp_ratio=mlp_ratio,
        resolution=resolution,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    return model


def demambavision_s(
        pretrained=False,
        **kwargs
):
    depths = kwargs.pop("depths", [3, 3, 7, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 96)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    model = DeMambaVision(
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dim=dim,
        in_dim=in_dim,
        mlp_ratio=mlp_ratio,
        resolution=resolution,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    return model


def demambavision_b(
        pretrained=False,
        **kwargs
):
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 128)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    model = DeMambaVision(
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dim=dim,
        in_dim=in_dim,
        mlp_ratio=mlp_ratio,
        resolution=resolution,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
        **kwargs
    )
    return model


def demambavision_l(
        pretrained=False,
        **kwargs
):
    depths = kwargs.pop("depths", [3, 3, 10, 5])
    num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 196)
    in_dim = kwargs.pop("in_dim", 64)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    model = DeMambaVision(
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        dim=dim,
        in_dim=in_dim,
        mlp_ratio=mlp_ratio,
        resolution=resolution,
        drop_path_rate=drop_path_rate,
        layer_scale=layer_scale,
        **kwargs
    )
    return model
