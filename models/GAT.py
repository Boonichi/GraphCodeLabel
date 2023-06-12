import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv

from .model import register_model

class GraphAttentionModel(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8, **kwargs):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)

  def forward(self, data):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    
    h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index, edge_attr)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index, edge_attr)

    return h, F.log_softmax(h, dim=1)

@register_model
def GAT(pretrained = False, dim_in = 1, dim_h = 4, dim_out = 5, **kwargs):
    model = GraphAttentionModel(dim_in, dim_h, dim_out, **kwargs)
    
    if pretrained == True:
       pass
    
    return model