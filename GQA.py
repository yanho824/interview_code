class GroupQueryAttention(nn.Module):
    def __init__(self, dim, head_nums, kv_nums):
        super().__init__()
        self.dim = dim
        self.head_nums = head_nums
        self.kv_nums = kv_nums
        self.head_dim = dim // head_nums
        assert dim % head_nums == 0
        assert head_nums % kv_nums == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, kv_nums * self.head_dim)
        self.v_proj = nn.Linear(dim, kv_nums * self.head_dim)
        self.out_proj = nn.Linear(dim, dim)

        self.atten_drop = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch, seq_len, self.head_nums, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, seq_len, self.kv_nums, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, seq_len, self.kv_nums, self.head_dim).transpose(1, 2)
        
        k = k.repeat_interleave(self.head_nums // self.kv_nums, dim=1)
        v = v.repeat_interleave(self.head_nums // self.kv_nums, dim=1)

        atten_value = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if mask is not None:
            atten_value = atten_value.masked_fill(mask == 0, float('1e-10'))
        atten_weight = torch.softmax(atten_value, dim=-1)
        atten_weight = self.atten_drop(atten_weight)
        atten_weight = atten_weight @ v
        atten_weight = atten_weight.transpose(1, 2).reshape(batch, seq_len, self.dim)
        atten_weight = self.out_proj(atten_weight)
        return atten_weight
