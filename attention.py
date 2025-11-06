import torch
from torch import nn as nn
import math

class self_attn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        # x: bs, seq, dim
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, self.dim, dim=-1)
        # q, k, v: bs, seq, dim
        # attn_weights: bs, seq, seq
        attn_weight = q @ k.transpose(-1,-2) / math.sqrt(self.dim)
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask == 0, float('1e-10'))
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_drop(attn_weight)
        attn_value = attn_weight @ v
        attn_value = self.out_proj(attn_value)

        return attn_value

x = torch.rand(2, 3, 4)
mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
# print(mask.shape)
# print(mask)
mask = mask.unsqueeze(1)
# print(mask.shape)
# print(mask)
mask = mask.repeat(1, 3, 1)
# print(mask.shape)
# print(mask)

self_attn = self_attn(4)
output = self_attn(x, mask)
print(output)
        
