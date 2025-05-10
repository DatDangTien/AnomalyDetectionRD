import torch
from torch import nn
from torch import Tensor
from torch.hub import load_state_dict_from_url
from typing import Type, Any, List, Tuple


# Pretrained models on ImageNet-1K
model_urls = {
    "convnext_t": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_s": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_b": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_l": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
}

class LayerNorm(nn.Module):
    """Apply Layer norm across C_dim in [N, C, H, W].
    """
    def __init__(self,
                 normalized_shape,
                 eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape


    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(dim=1, keepdim=True)
        v = ((x - u) ** 2).mean(dim=1, keepdim=True)
        x = (x - u) / (v + self.eps) ** 0.5
        x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class DropPath(nn.Module):
    def __init__(self,
                 drop_rate: float = .0):
        super().__init__()
        self.drop_rate = drop_rate


    def forward(self, x: Tensor) -> Tensor:
        if self.drop_rate == 0. or not self.training:
            return x
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(mask_shape, device=x.device) > self.drop_rate).float()
        return x / (1 - self.drop_rate) * mask


class Block(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=.0,
                 layer_scale_init_value=1e-6,
                 norm_eps=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * 4, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = identity + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 num_states=4,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.0,
                 norm_eps=1e-6):
        super().__init__()
        self.num_states = num_states
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=norm_eps)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=norm_eps),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        # Stochastic depth
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        norm_eps=norm_eps) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=norm_eps)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    # Collect feature maps from each stage
    def forward(self, x: Tensor) -> List[Tensor]:
        """Dim: 3,224,224 -> [96,56,56 -> 192,28,28 -> 384,14,14 -> 768,7,7]"""
        feature = []
        # print('Encoder__________________')
        for i in range(self.num_states):
            # print(x.shape)
            x = self.downsample_layers[i](x)
            # print(x.shape)
            x = self.stages[i](x)
            # print(x.shape)
            feature.append(x)
        # print(len(feature))
        return feature

def _convnext(arch: str,
              pretrained: bool = False,
              progress: bool = True,
              depths: List[int] = [3, 3, 9, 3],
              **kwargs) -> ConvNeXt:
    model = ConvNeXt(depths=depths, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict["model"])
    return model

class BN_layer(nn.Module):
    def __init__(self,
                 block,
                 dims=[96, 192, 384, 768],
                 bn_depth=3,
                 num_states=4,
                 drop_path_rate=.0,
                 layer_scale_init_value=1e-6,
                 norm_eps=1e-6):
        super().__init__()
        self.num_states = num_states
        self.mff = nn.ModuleList()
        for i in range(num_states):
            conv = []
            for j in range(i, num_states - 1):
                conv.append(nn.Conv2d(dims[j], dims[j+1],
                                      kernel_size=3, stride=2, padding=1))
                conv.append(LayerNorm(dims[j+1], eps=norm_eps))
                conv.append(nn.GELU())
            self.mff.append(nn.Sequential(*conv))

        self.downsample = nn.Sequential(
            nn.Conv2d(dims[num_states-1] * num_states, dims[num_states-1] * 2, kernel_size=2, stride=2),
            LayerNorm(dims[num_states-1] * 2, eps=norm_eps),
        )
        # Stochasic depth
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, bn_depth)]
        self.oce = nn.Sequential(
            *[block(dims[num_states-1] * 2, drop_path=dp_rates[i],
                    layer_scale_init_value=layer_scale_init_value,
                    norm_eps=norm_eps) for i in range(bn_depth)],
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

class De_ConvNeXt(nn.Module):
    def __init__(self,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 num_states=4,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 norm_eps=1e-6):
        super().__init__()
        self.num_states = num_states
        # Reverse dims
        dims = dims[:num_states]
        dims = dims[::-1]   # Reverse dims -> [768, 384, 192, 96]
        self.upsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.upsample_layers.append(nn.Sequential(
            nn.ConvTranspose2d(dims[0]*2, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=norm_eps)
        ))
        for i in range(num_states-1):
            self.upsample_layers.append(nn.Sequential(
                nn.ConvTranspose2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                LayerNorm(dims[i + 1], eps=norm_eps),
            ))

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        # Stochastic depth
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(num_states):
            stage = nn.Sequential(
                *[Block(dim=dims[i]*2, drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        norm_eps=norm_eps) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    # Collect feature maps from each stage
    def forward(self, x: Tensor) -> List[Tensor]:
        """Dim: 3,224,224 -> [96,56,56 -> 192,28,28 -> 384,14,14 -> 768,7,7]"""
        feature = []
        # print('Decoder__________________')
        for i in range(self.num_states):
            # print(x.shape)
            x = self.stages[i](x)
            # print(x.shape)
            x = self.upsample_layers[i](x)
            # print(x.shape)
            feature.append(x)
        # print(len(feature))
        return feature[::-1]    # Reverse -> [96, 192, 384, 768]


def convnext_tiny(pretrained: bool = False,
                 progress: bool = True,
                 **kwargs) -> Tuple[ConvNeXt, BN_layer]:
    """ConvNeXt-Tiny model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_.
    """
    depths = [3, 3, 9, 3]
    kwargs['dims'] = [96, 192, 384, 768]
    kwargs['drop_path_rate'] = 0.1
    kwargs['num_states'] = 3
    return _convnext('convnext_t', pretrained, progress, **kwargs), BN_layer(Block, **kwargs)


def convnext_small(pretrained: bool = False,
                   progress: bool = True,
                   **kwargs) -> Tuple[ConvNeXt, BN_layer]:
    """ConvNeXt-Small model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_.
    """
    depths = [3, 3, 27, 3]
    kwargs['dims'] = [96, 192, 384, 768]
    kwargs['drop_path_rate'] = 0.4
    kwargs['num_states'] = 3
    return _convnext('convnext_s', pretrained, progress, depths, **kwargs), BN_layer(Block, **kwargs)


def convnext_base(pretrained: bool = False,
                  progress: bool = True,
                  **kwargs) -> Tuple[ConvNeXt, BN_layer]:
    """ConvNeXt-Base model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_.
    """
    depths = [3, 3, 27, 3]
    kwargs['dims'] = [128, 256, 512, 1024]
    kwargs['drop_path_rate'] = 0.5
    kwargs['num_states'] = 3
    return _convnext('convnext_b', pretrained, progress, depths, **kwargs), BN_layer(Block, **kwargs)


def convnext_large(pretrained: bool = False,
                   progress: bool = True,
                   **kwargs) -> Tuple[ConvNeXt, BN_layer]:
    """ConvNeXt-Large model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_.
    """
    depths = [3, 3, 27, 3]
    kwargs['dims'] = [192, 384, 768, 1536]
    kwargs['drop_path_rate'] = 0.5
    kwargs['num_states'] = 3
    return _convnext('convnext_l', pretrained, progress, depths, **kwargs), BN_layer(Block, **kwargs)


def de_convnext_tiny(progress: bool = True,
                     pretrained: bool = False,
                     **kwargs) -> De_ConvNeXt:
    """De-ConvNeXt-Tiny model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_.
    """
    depths = [3, 3, 9, 3]
    kwargs['dims'] = [96, 192, 384, 768]
    kwargs['drop_path_rate'] = 0.1
    kwargs['num_states'] = 3
    return De_ConvNeXt(depths=depths, **kwargs)


def de_convnext_small(progress: bool = True,
                      pretrained: bool = False,
                      **kwargs) -> De_ConvNeXt:
    """De-ConvNeXt-Small model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_.
    """
    depths = [3, 3, 27, 3]
    kwargs['dims'] = [96, 192, 384, 768]
    kwargs['drop_path_rate'] = 0.4
    kwargs['num_states'] = 3
    return De_ConvNeXt(depths=depths, **kwargs)


def de_convnext_base(progress: bool = True,
                     pretrained: bool = False,
                     **kwargs) -> De_ConvNeXt:
        """De-ConvNeXt-Base model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_.
        """
        depths = [3, 3, 27, 3]
        kwargs['dims'] = [128, 256, 512, 1024]
        kwargs['drop_path_rate'] = 0.5
        kwargs['num_states'] = 3
        return De_ConvNeXt(depths=depths, **kwargs)


def de_convnext_large(progress: bool = True,
                      pretrained: bool = False,
                      **kwargs) -> De_ConvNeXt:
        """De-ConvNeXt-Large model from `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_.
        """
        depths = [3, 3, 27, 3]
        kwargs['dims'] = [192, 384, 768, 1536]
        kwargs['drop_path_rate'] = 0.5
        kwargs['num_states'] = 3
        return De_ConvNeXt(depths=depths, **kwargs)

