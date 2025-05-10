import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
class AdaptiveStagesFusion(nn.Module):
    def __init__(self,
                 num_stages=4,
                 w_init: float = 1.0,
                 trainable:bool = False,
                 scale: bool = True,
                 inverse: bool = False,
                 device = 'cpu'):
        super().__init__()
        self.num_stages = num_stages
        self.trainable = trainable
        self.scale = scale
        self.inverse = inverse
        self.weight = nn.Parameter(torch.full((num_stages,), w_init))
        self.linears = None
        self.device = device

    def forward(self, x) -> torch.Tensor:
        # If not trainable, return no grad weight.
        if not self.linears:
            self._init_linears(x)

        fusion_scores = []
        for i in range(len(x)):
            # [B,C,H,W] -> [B,C,1,1] -> [B,C] -> [B,1] -> [1]
            max_pool = F.adaptive_max_pool2d(x[i], output_size=1).squeeze(-1).squeeze(-1)
            fusion_score = torch.mean(self.linears[i](max_pool))
            fusion_scores.append(fusion_score)
        fusion_scores = torch.stack(fusion_scores)
        print('fusion scores: ',fusion_scores)


        w = self.weight
        if not self.trainable:
            with torch.no_grad():
                w = torch.ones_like(self.weight)
        # Inverse
        if self.inverse:
            w = 1.0 / (w + 1e-8)

        # Normalize
        w = (w * fusion_scores).softmax(dim=0)
        # Scale
        if self.scale:
            w = w * self.num_stages
        return w

    def _init_linears(self, x):
        self.linears = nn.ModuleList([
            nn.Linear(feat.shape[1], 1, device=self.device) for feat in x
        ])

    def get_weight(self) -> torch.Tensor:
        with torch.no_grad():
            return self.forward()


    def freeze(self) -> None:
        self.weight.requires_grad = False

    def unfreeze(self) -> None:
        self.weight.requires_grad = True

    def set_inverse(self) -> None:
        self.inverse = not self.inverse

    def set_trainable(self, state: bool) -> None:
        self.trainable = state



def adap_loss_function(a, b, w_module=None,
                       loss_type='cosine',
                       w_entropy=0.01,
                       device='cpu'):
    cos_loss = torch.nn.CosineSimilarity()

    if w_module is None:
        w = torch.ones(len(a)).float()
    else:
        w = w_module(b)

    loss = torch.tensor(0.0, device=device)
    for item in range(len(a)):
        stage_loss = torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                             b[item].view(b[item].shape[0], -1)))
        loss = loss + w[item] * stage_loss

    # Entropy penalty
    gini = 1 - torch.sum((w / len(a)) ** 2)
    penalty = 1.0 / gini

    # Weight loss with entropy
    return loss + w_entropy * penalty


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
        a_map = a_map * w[i]
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list
