import torch
from torch import nn as nn
import math

class SimpleDecoder(nn.Module):
    def __init__(self, dim, head_nums):
        super().__init__()
        self.dim = dim
        self.head_nums = head_nums
        assert dim % head_nums == 0, f"dim({dim})必须能被head_nums({head_nums})整除"
        self.head_dim = dim // head_nums

        self.atten_layerNorm = nn.LayerNorm(dim, eps=1e-5)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(0.1)

        self.up = nn.Linear(dim, dim * 4)
        self.down = nn.Linear(dim * 4, dim)
        self.fnn_layernorm2 = nn.LayerNorm(dim, eps=1e-5)
        self.act_fn = nn.GELU()
        self.fnn_drop = nn.Dropout(0.1)

    def atten_output(self, q, k, v, mask=None):
        atten_value = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.tril()
            atten_value = atten_value.masked_fill(mask==0, float('-inf'))
        else:
            mask = torch.ones_like(atten_value).tril()
            atten_value = atten_value.masked_fill(mask==0, float('-inf'))
        
        atten_weight = atten_value @ v
        batch, head_nums, seq_len, head_dim = atten_weight.size()
        atten_weight = atten_weight.transpose(1, 2).reshape(batch, seq_len, self.dim)
        atten_weight = self.out_proj(atten_weight)
        return atten_weight
    
    def block_block(self, x, mask=None):
        batch, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.q_proj(x)
        v = self.q_proj(x)

        q = q.reshape(batch, seq_len, self.head_nums, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.head_nums, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.head_nums, self.head_dim).transpose(1, 2)

        atten = self.atten_output(q, k, v, mask)
        atten = atten + x
        atten = self.atten_layerNorm(atten)
        return atten
    
    def fnn_block(self, x):
        up = self.up(x)
        up = self.act_fn(up)
        down = self.down(up)
        return self.fnn_layernorm2(down + x)
    
    def forward(self, x, mask=None):
        x = self.block_block(x, mask)
        x = self.fnn_block(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, dim, head_nums, layers):
        super().__init__()
        self.layer_list = nn.ModuleList([SimpleDecoder(dim, head_nums) for _ in range(layers)])
        self.emb = nn.Embedding(100, dim)
        self.out = nn.Linear(dim, 100)
    
    def forward(self, x, mask=None):
        x = self.emb(x)
        for i, layer in enumerate(self.layer_list):
            x = layer(x, mask)
        print(x.shape)
        x = self.out(x)
        return torch.softmax(x, dim=-1)

x = torch.rand(3, 4, 64)
net = SimpleDecoder(64, 8)
mask = (
    torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]]) # (3, 4)
    .unsqueeze(1)
    .unsqueeze(2) # (3, 1, 1, 4)
    .repeat(1, 8, 4, 1)
)
output = net(x, mask)
print(output.shape)
