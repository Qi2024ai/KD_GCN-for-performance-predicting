import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from han import SemanticAttention
import random 
import numpy as np
import warnings
 
warnings.filterwarnings("ignore")
from num import num
device = "cuda" if torch.cuda.is_available() else "cpu"
num=num.to(device)


class GCNLayer(nn.Module):
    def __init__(self,in_size, hidden_size,residual =False,activation=None):
        super(GCNLayer, self).__init__()
        self.activation = activation
        self.gcn = GraphConv(in_size, hidden_size, activation=None)
        self.residual = residual
        if residual:
            self.linear = nn.Linear(in_size,hidden_size) if in_size != hidden_size else nn.Identity()

    def forward(self,g,h):
        gh = self.gcn(g,h)
        if self.residual:
            gh += self.linear(h)
        return self.activation(gh)

class HCANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size):
        super(HCANLayer, self).__init__()
        self.gcn_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gcn_layers.append(GCNLayer(in_size, out_size, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size,hidden_size=out_size)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gcn_layers[i](g, h))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HCANSingleLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size):
        super(HCANSingleLayer, self).__init__()
        self.gcn_layers = GCNLayer(in_size, out_size, activation=F.elu)
        self.semantic_attention = SemanticAttention(in_size=out_size,hidden_size=out_size)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gcn_layers(g, h))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings) 

class HCAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size,num_layers, out_size, dropout, num_heads=None):
        super(HCAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HCANLayer(num_meta_paths, in_size, hidden_size))
        for l in range(1, num_layers):
            self.layers.append(HCANLayer(num_meta_paths, hidden_size,hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.predict_layer1 = nn.Sequential(nn.Linear(hidden_size,250), nn.ReLU(True))
        self.predict_layer2 = nn.Sequential(nn.Linear(250, 75), nn.ReLU(True))
        self.predict_layer3 = nn.Sequential(nn.Linear(75, out_size))
        
    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g,h)
        h = self.dropout(h)
        scale = torch.floor_divide(torch.arange(h.shape[1]) ,torch.floor_divide(h.shape[1] , 10))
        h *= num[:, scale]
        h=self.predict_layer1(h)
        h=self.predict_layer2(h)
        h=self.predict_layer3(h)
        return h

class HCANSingle(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size,num_layers, out_size, dropout, num_heads=None):
        super(HCANSingle, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HCANSingleLayer(num_meta_paths, in_size, hidden_size))
        for l in range(1, num_layers):
            self.layers.append(HCANSingleLayer(num_meta_paths, hidden_size,hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.predict = nn.Linear(hidden_size, out_size)
        
    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g,h)
        h = self.dropout(h)
        return self.predict(h)