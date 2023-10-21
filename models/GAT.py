import torch.nn.functional as F
import torch.nn as nn
from dgl.nn import GATConv

import os
os.environ["DGLBACKEND"] = "pytorch"


class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, o_feats, heads):
        super(GAT, self).__init__()
        assert (h_feats % heads == 0)
        h_feats = int(h_feats / heads)
        self.bn = nn.BatchNorm1d(in_feats)
        self.conv1 = GATConv(in_feats, h_feats, heads)
        self.conv2 = GATConv(h_feats*heads, h_feats, heads)
        self.conv3 = GATConv(h_feats*heads, h_feats, heads)
        self.conv4 = GATConv(h_feats*heads, o_feats, 1)

    def forward(self, g, in_feat):
        h = self.bn(in_feat)
        h = self.conv1(g, h)
        h = h.view(h.shape[0], -1)
        h = F.relu(h)

        h = self.conv2(g, h)
        h = h.view(h.shape[0], -1)
        h = F.relu(h)

        h = self.conv3(g, h)
        h = h.view(h.shape[0], -1)
        h = F.relu(h)

        h = self.conv4(g, h)
        h = h.view(h.shape[0], -1)
        return h
