import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
class AdaptiveStages(nn.Module):
    def __init__(self,
                 num_stages=4,
                 w_init: float = 1.0,
                 scale: bool = True,
                 inverse: bool = False,):
        super().__init__()
        self.num_stages = num_stages
        self.scale = scale
        self.inverse = inverse
        self.weight = nn.Parameter(torch.full((num_stages,), w_init))

    def forward(self) -> torch.Tensor:
        # Inverse
        if self.inverse:
            w = 1.0 / (self.weight + 1e-8)
        else:
            w = self.weight
        # Normalize
        w = w.softmax(dim=0)
        # Scale
        if self.scale:
            w = w * self.num_stages
        return w

    def get_weight(self) -> torch.Tensor:
        with torch.no_grad():
            return self.forward()


def adap_loss_function(a, b, w=None,
                       loss_type='cosine',
                       w_entropy=0.01,
                       device='cpu'):
    cos_loss = torch.nn.CosineSimilarity()

    if w is None:
        w = torch.ones(len(a)).float()

    loss = torch.tensor(0.0, device=device)
    for item in range(len(a)):
        stage_loss = torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                             b[item].view(b[item].shape[0], -1)))
        loss = loss + w[item] * stage_loss

    gini = 1 - torch.sum((w / len(a)) ** 2)

    # Weight loss with entropy
    return loss + 2 * w_entropy * gini


def cal_anomaly_map(a,b, w=None, out_size=224, amap_mode='mul'):
    if w is None:
        w = torch.ones(len(a))

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