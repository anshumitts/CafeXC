from xc.libs.utils import mean_pooling
import torch.nn as nn
import numpy as np
import torch


def elu_feature_map(mat):
    return nn.functional.elu(mat) + 1


class Residual(nn.Module):
    def __init__(self, input_dims, dropout=0.1):
        super(Residual, self).__init__()
        drop = nn.Dropout(p=dropout)
        activation = nn.GELU()
        trans = Projection(input_dims, input_dims)
        self.transform = nn.Sequential(trans, drop, activation)

    def forward(self, embs):
        # type: (Tensor) -> Tensor
        return self.transform(embs) + embs


def Projection(num_embeddings, embedding_dim, residual=False):
    if residual:
        return Residual(num_embeddings)

    linear = nn.Linear(num_embeddings, embedding_dim)
    if num_embeddings == embedding_dim:
        torch.nn.init.eye_(linear.weight)
    else:
        torch.nn.init.kaiming_uniform_(linear.weight)
    linear.bias.data.fill_(0)
    linear = nn.utils.spectral_norm(linear)
    return linear


class QuadraticAttention(nn.Module):
    def __init__(self, input_dims, dropout=0.2):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask):
        E = q.size(-1)
        scr = torch.einsum("nlhe,nshe->nhls", q/np.sqrt(E), k)  # BxhxT2xT1
        scr.masked_fill_(mask == 0, -float("inf"))
        scr = self.activation(scr)
        scr = self.drop(scr)  # BxhxT2xT1
        return torch.einsum("nhls,nshe->nlhe", scr, v).contiguous(), scr


class MultiHeadAtttention(nn.Module):
    def __init__(self, attention, input_dims, n_heads, project=True, dropout=0.2, project_dim=None):
        super(MultiHeadAtttention, self).__init__()
        project_dim = project_dim if project_dim is not None else input_dims
        assert input_dims % n_heads == 0, f"Input dims {input_dims} must be divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.dim_per_head = input_dims//n_heads
        self.project_dim = project_dim
        self.plin_q = Projection(input_dims, project_dim)
        self.plin_k = Projection(input_dims, project_dim)
        self.plin_v = Projection(input_dims, project_dim)
        self.attention = attention
        self.plin_o = Projection(project_dim, input_dims)
        self.norm = nn.LayerNorm(input_dims)

    def forward(self, query, key, mask):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        bs, k_sq, D = key.size()
        bq, q_sq, _ = query.size()
        q, k, v = self.plin_q(query), self.plin_k(key), self.plin_v(key)
        k = k.view(bs, k_sq, self.n_heads, self.dim_per_head)  # BxTxh1xD//h
        v = v.view(bs, k_sq, self.n_heads, self.dim_per_head)  # BxT1xhxD//h
        q = q.view(bq, q_sq, self.n_heads, self.dim_per_head)  # BxT2xhxD//h
        mask = mask.view(bs, 1, 1, k_sq).repeat(1, self.n_heads, q_sq, 1)
        out, attn = self.attention(q, k, v, mask)
        out = out.view(bq, q_sq, self.project_dim)
        out = self.plin_o(out)
        return self.norm(out + query), attn

    def extra_repr(self):
        return f"(nheads): {self.n_heads}"


class ReverseTransformerBlock(nn.Module):
    def __init__(self, input_dims, head_dims=None, n_heads=1, project=True,
                 dropout=0.2, linear_attn=False):
        super(ReverseTransformerBlock, self).__init__()
        head_dims = input_dims if head_dims is None else head_dims
        attention = QuadraticAttention(input_dims, dropout=dropout)
        self.attented = MultiHeadAtttention(attention, input_dims, n_heads,
                                            project, dropout=dropout)

    def forward(self, contexts, _input, mask):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        contexts, attn_wts = self.attented(contexts, _input, mask)
        return contexts, attn_wts


class ReverseTransformer(nn.Module):
    def __init__(self, input_dims, head_dims=None, n_heads=1,
                 n_layers=1, project=True, dropout=0.1):
        super(ReverseTransformer, self).__init__()
        self.setup()
        self.layers = nn.ModuleList([
            ReverseTransformerBlock(input_dims, head_dims,
                                    n_heads, project, dropout,
                                    self.linear_attn)
            for i in range(n_layers)
        ])

    def setup(self):
        self.linear_attn = False

    def forward(self, contexts, context_mask, apply_pooling=True, output_attn_wts=False):
        # type: (Tensor, Tensor, bool) -> Tuple[Tensor, Tensor]
        for module in self.layers:
            contexts, attn_wts = module(contexts, contexts, context_mask)

        if apply_pooling:
            contexts = mean_pooling(contexts, context_mask)
            context_mask = torch.ones(
                (contexts.size(0), 1), device=contexts.device)

        if not output_attn_wts:
            attn_wts = None
        return contexts, context_mask, attn_wts


class RTEncoder(ReverseTransformer):
    def __init__(self, input_dims, head_dims=None, n_heads=1,
                 n_layers=1, project=True, dropout=0.1):
        super(RTEncoder, self).__init__(input_dims, head_dims, n_heads,
                                        n_layers, project, dropout)


class RTDecoder(ReverseTransformer):
    def __init__(self, input_dims, head_dims=None, n_heads=1,
                 n_layers=1, project=True, dropout=0.1):
        self.n_heads = n_heads
        super(RTDecoder, self).__init__(input_dims, head_dims, n_heads,
                                        n_layers, project, dropout)

    def setup(self):
        self.linear_attn = False

    def forward(self, contexts, context_mask, sequenence,
                sequenence_mask, apply_pooling=True, output_attn_wts=False):
        # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tuple[Tensor, Tensor, Tensor]
        bs, Sk, k_sq, D = sequenence.size()
        _, q_sq, _ = contexts.size()
        contexts = contexts.repeat_interleave(Sk, dim=0)
        sequenence = sequenence.view(bs*Sk, k_sq, D)
        sequenence_mask = sequenence_mask.view(bs*Sk, k_sq)
        for module in self.layers:
            contexts, attn_wts = module(contexts, sequenence, sequenence_mask)
        contexts = contexts.view(bs, Sk, q_sq, D)
        if apply_pooling:
            contexts = mean_pooling(contexts, context_mask.unsqueeze(1))
            context_mask = torch.ones((bs, Sk, 1), device=contexts.device)

        if not output_attn_wts:
            attn_wts = None
        else:
            attn_wts = attn_wts.view(bs, Sk, self.n_heads, q_sq, k_sq)
        return contexts, context_mask, attn_wts
