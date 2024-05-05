import torch.nn as nn
import torch.nn.functional as F
import torch


class QNet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(QNet, self).__init__()
        q_net_dim = 256
        self.fc1 = nn.Linear(input_dim, q_net_dim)
        self.fc21 = nn.Linear(q_net_dim, latent_dim)
        self.fc22 = nn.Linear(q_net_dim, latent_dim)

    def forward(self, x):
        e = F.relu(self.fc1(x))
        mu = self.fc21(e)
        lv = self.fc22(e)
        # return mu,lv.mul(0.5).exp_()
        return mu, torch.tensor(0.75).cuda()
