import torch
from torch import nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head_nums):
        super().__init__()
        self.dim = dim
        self.head_nums = head_nums
        assert dim % head_nums == 0, f"dim({dim})必须能被head_nums({head_nums})整除"
        self.head_dim = dim // head_nums

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.atten_drop = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        batch, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch, seq_len, self.head_nums, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.head_nums, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.head_nums, self.head_dim).transpose(1, 2)

        atten_value = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if mask is not None:
            atten_value = atten_value.masked_fill(mask==0, float('-inf'))
        atten_weight = torch.softmax(atten_value, dim=-1)
        atten_weight = self.atten_drop(atten_weight)

        atten_weight = atten_weight @ v
        atten_weight = atten_weight.transpose(1, 2).reshape(batch, seq_len, self.dim)
        atten_weight = self.out_proj(atten_weight)
        return atten_weight

x = torch.rand(2, 3, 4)
mask = torch.tensor([[1, 1, 0], [1, 0, 0]]) # (2, 3)
mask = mask.unsqueeze(1).repeat(1, 3, 1) # (2, 1, 3)->(2, 3, 3)
mha = MultiHeadAttention(4, 2)
res = mha(x, mask)
print(res)