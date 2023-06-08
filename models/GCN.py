import torch
import torch.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv

from .model import register_model

class GraphConvolutionModel():
  """Graph Convolutional Network"""
  def __init__(self, dim_in, dim_h, dim_out, **kwargs):
    super().__init__()
    self.gcn1 = GCNConv(dim_in, dim_h)
    self.gcn2 = GCNConv(dim_h, dim_out)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.01,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.5, training=self.training)
    h = self.gcn1(h, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.gcn2(h, edge_index)
    return h, F.log_softmax(h, dim=1)
  

@register_model
def GCN(pretrained = False, dim_in = 1, dim_h = 16, dim_out = 5, **kwargs):
    model = GraphConvolutionModel(dim_in, dim_h, dim_out, **kwargs)
    return model