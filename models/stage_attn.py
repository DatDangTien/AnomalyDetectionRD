import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
class AdaptiveStagesFusion(nn.Module):
    def __init__(self,
                 num_stages=4,
                 w_init: float = 1.0,
                 w_alpha = 1.0,
                 trainable:bool = False,
                 scale: bool = True,
                 inverse: bool = False,
                 device = 'cpu'):
        super().__init__()
        self.num_stages = num_stages
        self.trainable = trainable
        self.scale = scale
        self.inverse = inverse
        self.w_alpha = w_alpha
        self.weight = nn.Parameter(torch.full((num_stages,), w_init))
        self.linears = nn.ModuleList()
        self.act = nn.Softplus()
        self.device = device

    def forward(self, x) -> torch.Tensor:
        # If not trainable, return no grad weight.
        if len(self.linears) == 0:
            self._init_linears(x)

        fusion_scores = []
        for i in range(len(x)):
            # print(x[i].shape)
            # [B,C,H,W] -> [B,C,1,1] -> [B,C] -> [B,1] -> [B,N]
            # if x[i].isnan().any():
            # print('Decoder error: ')
            max_pool = F.adaptive_max_pool2d(x[i], output_size=1).squeeze(-1).squeeze(-1)
            # print(max_pool.shape)
            fusion_score = self.act(self.linears[i](max_pool).squeeze(-1))
            # print(fusion_score)
            fusion_scores.append(fusion_score)
        fusion_scores = torch.stack(fusion_scores, dim=1)
        # fusion_scores = fusion_scores.max(dim=0)
        if not self.trainable:
            fusion_scores = fusion_scores.detach()
        # print('fusion scores: ',fusion_scores)
        w = self.weight if self.trainable else self.weight.detach()
        # W-Alpha scale
        w = w * self.w_alpha

        # Inverse
        if self.inverse:
            w = 1.0 / (w.clamp(min=1e-4))

        # Feature scale
        if self.trainable:
            w = (w * fusion_scores)
        else:
            # Expand to [B, N]
            w = w.expand(x[0].shape[0], -1)

        # Normalize
        w = w.softmax(dim=1)
        # Scale
        if self.scale:
            w = w * self.num_stages
        return w


    def _init_linears(self, x):
        for feat in x:
            self.linears.append(nn.Sequential(
                # nn.LayerNorm(feat.shape[1], device=self.device),
                nn.Linear(feat.shape[1], 1, device=self.device)
            ))

        for block in self.linears:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=.02)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.weight, 1.0)



    def get_weight(self, x) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


    def freeze(self) -> None:
        self.weight.requires_grad = False
        self.trainable = False

    def unfreeze(self) -> None:
        self.weight.requires_grad = True
        self.trainable = True

    def set_inverse(self) -> None:
        self.inverse = not self.inverse

    def set_trainable(self, state: bool) -> None:
        self.trainable = state



def adap_loss_function(a, b, w_module=None,
                       loss_type='cosine',
                       w_entropy=0.01,
                       device='cpu'):
    cos_loss = torch.nn.CosineSimilarity()

    # w: Tensor(B, N)
    if w_module is None:
        w = torch.ones(len(a)).float()
    else:
        w = w_module(b)

    print(w[0])

    loss = torch.tensor(0.0, device=device)
    for item in range(len(a)):
        stage_loss = torch.mean(w[:, item] * (1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                                           b[item].view(b[item].shape[0], -1))))
        # loss = loss + w[item] * stage_loss
        loss = loss + stage_loss

    # Entropy penalty
    # gini = 1 - torch.sum((w / len(w)) ** 2)
    # penalty = 1.0 / gini
    penalty = torch.mean(torch.sum((w / w.shape[1]) ** 2, dim=1), dim=0)

    # Weight loss with entropy
    loss = loss + w_entropy * penalty

    # Debug
    if loss.item() > 2:
        print('Penalty: ', w_entropy * penalty)

    return loss


def cal_anomaly_map(a,b, w_module=None, out_size=224, amap_mode='mul'):
    if w_module is None:
        w = torch.ones(len(a))
    else:
        w = w_module(b)

    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(a)):
        a_map = 1 - F.cosine_similarity(a[i], b[i])
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        # Adaptive stage weight
        a_map = a_map * w[:, i]
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list
