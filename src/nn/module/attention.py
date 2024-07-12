import torch
from torch import nn
from torch_geometric.nn import SimpleConv

from src.nn.module.layer import *
from src.nn.module.block import *
from src.utils.utils import *

class SGFormer(base_Module):
    def __init__(self, in_features, out_features, num_heads, use_weight=True, res=True, output_attn=False):
        super(SGFormer, self).__init__()
        apply_args(self)

        self.qs = nn.Linear(in_features, out_features * num_heads)
        self.ks = nn.Linear(in_features, out_features * num_heads)
        self.vs = nn.Linear(in_features, out_features * num_heads)

        self.fc = MLP(out_features, out_features)

        self._reset_parameters(self)

    def forward(self, x, A=None):
        query = self.qs(x).reshape(-1, self.num_heads, self.out_features)
        key = self.ks(x).reshape(-1, self.num_heads, self.out_features)

        if self.use_weight:
            value = self.vs(x).reshape(-1, self.num_heads, self.out_features)
        else:
            value = x.reshape(-1, 1, self.out_features)

        final_output = self.attention(query, key, value)
        if self.res:
            final_output += value.squeeze(1) if self.num_heads <= 1 else value.mean(dim=1).squeeze(1)

        return self.fc(final_output).unsqueeze(0)

    def attention(self, qs, ks, vs):
        """
        qs: query tensor [N, H, M]
        ks: key tensor [L, H, M]
        vs: value tensor [L, H, D]

        return output [N, H, D]
        """
        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape)
        )  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = (attention_num / attention_normalizer).mean(dim=1)  # [N, H, D]

        # compute attention for visualization if needed
        if self.output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        if self.output_attn:
            return attn_output, attention

        else:
            return attn_output

class P_SGFormer(SGFormer):
    def __init__(self, in_features, out_features, num_heads, use_weight=True, res=True, output_attn=False):
        super(P_SGFormer, self).__init__(**args2dict())
        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')

    def forward(self, x, A=None):
        query = self.qs(x).reshape(-1, self.num_heads, self.out_features)
        key = self.ks(x).reshape(-1, self.num_heads, self.out_features)

        if self.use_weight:
            value = self.vs(x).reshape(-1, self.num_heads, self.out_features)
        else:
            value = x.reshape(-1, 1, self.out_features)

        final_output = self.attention(query, key, value)
        if self.res:
            final_output += value.squeeze(1) if self.num_heads <= 1 else value.mean(dim=1).squeeze(1)

        if not self.training and not (a is not None or len(a) > 0):
            final_output = self.conv(final_output, a)

        return self.fc(final_output).unsqueeze(0)



__all__ = auto_all()

if __name__ == "__main__":
    _in = torch.rand((64, 10))
    a = SGFormer(10, 1024, 1)(_in, [[]])
    print(a.shape)

    a = P_SGFormer(10, 1024, 1)(_in, [[]])
    print(a.shape)