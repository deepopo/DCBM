import math
import torch
import torch.nn as nn

class Decoupling(nn.Module):
    def __init__(self, dim_explicit, dim_implicit, hidden_dim=100):
        super(Decoupling, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_explicit, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim_implicit))

    def forward(self, x):
        return self.layers(x)

class MINE(nn.Module):
    def __init__(self, dim_explicit, dim_implicit, hidden_dim=50) :
        super(MINE, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear((dim_explicit + dim_implicit), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, e_explicit, e_implicit):
        batch_size = e_explicit.size(0)
        tiled_x = torch.cat([e_explicit, e_explicit], dim=0)
        idx = torch.randperm(batch_size)

        shuffled_y = e_implicit[idx]
        concat_y = torch.cat([e_implicit, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        mi = math.log2(math.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        return mi