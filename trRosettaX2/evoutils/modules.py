import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint
from functools import partial


def apply_dropout(*, tensor, rate, is_training, broadcast_dim=None):
    """Applies dropout to a tensor."""
    if is_training and rate != 0.0:
        shape = list(tensor.shape)
        if broadcast_dim is not None:
            shape[broadcast_dim] = 1
        keep_rate = 1.0 - rate
        keep = torch.bernoulli(keep_rate * torch.ones(shape, device=tensor.device))
        return keep * tensor / keep_rate
    else:
        return tensor


class Symm(nn.Module):
    def __init__(self, pattern):
        super(Symm, self).__init__()
        self.pattern = pattern

    def forward(self, x):
        return (x + Rearrange(self.pattern)(x)) / 2


class InputEmbedder(nn.Module):
    def __init__(self, n_feat_tar, n_feat_msa):
        super(InputEmbedder, self).__init__()
        self.linear_tar = nn.Linear(n_feat_tar, 128 * 3)
        self.linear_msa = nn.Linear(n_feat_msa, 128)
        self.relpos = relpos()

    def forward(self, target_feat, res_id, msa_feat):
        """
        :param target_feat:b,L,d
        :param res_id: b,L
        :param msa_feat: b,n,l,d
        :return:
        """
        a, b, ft = torch.chunk(self.linear_tar(target_feat), chunks=3, dim=-1)
        z = a[:, None, :, :] + b[:, :, None, :]
        z = z + self.relpos(res_id)
        m = self.linear_msa(msa_feat) + ft
        return m, z


class relpos(nn.Module):

    def __init__(self):
        super(relpos, self).__init__()
        self.linear = nn.Linear(65, 128)

    def forward(self, res_id):
        device = res_id.device
        bin_values = torch.arange(-32, 33, device=device)
        d = res_id[:, :, None] - res_id[:, None, :]
        bdy = torch.tensor(32, device=device)
        d = torch.minimum(torch.maximum(-bdy, d), bdy)
        d_onehot = (d[..., None] == bin_values).float()
        assert d_onehot.sum(dim=-1).min() == 1
        p = self.linear(d_onehot)
        return p


class DropoutWrapper(nn.Module):
    def __init__(self, module, wise='row', rate=.15):
        super(DropoutWrapper, self).__init__()
        self.module = module
        self.wise = wise
        self.rate = rate

    def forward(self, *input_act, is_training=True):
        wise, dropout_rate = self.wise, self.rate
        residual = self.module(*input_act)
        if wise == 'row':
            broadcast_dim = 0
        elif wise == 'col':
            broadcast_dim = 1
        else:
            raise ValueError(f'unk wise {wise}')
        residual = apply_dropout(tensor=residual,
                                 rate=dropout_rate,
                                 is_training=is_training,
                                 broadcast_dim=broadcast_dim)

        return residual


class EvoformerStack(nn.Module):
    def __init__(
            self,
            n_block=48,
            dim=384,
            in_dim=128,
            dim_msa=32,
            dim_outer=32,
            dim_pair_multi=128,
            dim_pair_attn=32,
            dropout_rate_msarow=.15,
            dropout_rate_pair=.25,
    ):
        super(EvoformerStack, self).__init__()
        self.blocks = nn.ModuleList(
            [
                EvoformerBlock(in_dim=in_dim,
                               dim_msa=dim_msa,
                               dim_outer=dim_outer,
                               dim_pair_multi=dim_pair_multi,
                               dim_pair_attn=dim_pair_attn,
                               dropout_rate_msarow=dropout_rate_msarow,
                               dropout_rate_pair=dropout_rate_pair,
                               )
                for _ in range(n_block)
            ]
        )
        self.to_single = nn.Linear(in_dim, dim)

    def forward(self, m, z, is_training=True):
        for block in self.blocks:
            m, z = block(m, z, is_training=is_training)
        s = self.to_single(m[:, 0])
        return m, z, s


class EvoformerBlock(nn.Module):
    def __init__(
            self,
            in_dim=128,
            dim_msa=32,
            dim_outer=32,
            dim_pair_multi=128,
            dim_pair_attn=32,
            dropout_rate_msarow=.15,
            dropout_rate_pair=.25,
    ):
        super(EvoformerBlock, self).__init__()
        self.msa_row_attn = DropoutWrapper(MSARowAttention(in_dim=in_dim, dim=dim_msa), wise='row',
                                           rate=dropout_rate_msarow)
        self.msa_col_attn = MSAColAttention(in_dim=in_dim, dim=dim_msa)
        self.msa_trans = MSATransition(dim=in_dim)

        self.msa2pair = OuterProductMean(in_dim=in_dim, dim=dim_outer)

        self.pair_multi_out = DropoutWrapper(
            TriangleMultiplication(in_dim=in_dim, dim=dim_pair_multi, direct='outgoing'), rate=dropout_rate_pair)
        self.pair_multi_in = DropoutWrapper(
            TriangleMultiplication(in_dim=in_dim, dim=dim_pair_multi, direct='incoming'), rate=dropout_rate_pair)
        self.pair_row_attn = DropoutWrapper(TriangleAttention(in_dim=in_dim, dim=dim_pair_attn, wise='row'),
                                            rate=dropout_rate_pair)
        self.pair_col_attn = DropoutWrapper(TriangleAttention(in_dim=in_dim, dim=dim_pair_attn, wise='col'), wise='col',
                                            rate=dropout_rate_pair)

        self.pair_trans = PairTransition(dim=in_dim)

    def forward(self, m, z, is_training=True):
        msa_row_attn = partial(self.msa_row_attn, is_training=is_training)
        m = m + checkpoint(msa_row_attn, m, z)
        msa_col_attn = self.msa_col_attn
        m = m + checkpoint(msa_col_attn, m)
        m = m + self.msa_trans(m)

        z = z + self.msa2pair(m)

        z = z + self.pair_multi_out(z, is_training=is_training)
        z = z + self.pair_multi_in(z, is_training=is_training)
        pair_row_attn = partial(self.pair_row_attn, is_training=is_training)
        z = z + checkpoint(pair_row_attn, z)
        pair_col_attn = partial(self.pair_col_attn, is_training=is_training)
        z = z + checkpoint(pair_col_attn, z)
        z = z + self.pair_trans(z)

        return m, z


class MSARowAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=8):
        super(MSARowAttention, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, dim * n_heads),
            nn.Sigmoid()
        )
        self.linear_out = nn.Linear(dim * n_heads, in_dim)
        self.n_heads = n_heads

    def forward(self, *inputs):
        m, z = inputs
        m = self.norm(m)
        q, k, v = torch.chunk(self.to_qkv(m), 3, -1)
        q, k, v = map(lambda x: rearrange(x, 'b i j (h d)->b i j h d', h=self.n_heads), (q, k, v))
        b = self.for_pair(z)
        b = rearrange(b, 'b i j h->b h i j')
        gate = self.to_gate(m)
        gate = rearrange(gate, 'b i j (h d)->b i j h d', h=self.n_heads)
        scale = q.size(-1) ** .5

        attn = (torch.einsum('brihd,brjhd->bhij', q, k) / scale + b).softmax(-1)
        out = torch.einsum('bhij,brjhd->brihd', attn, v)
        out = rearrange(gate * out, 'b r i h d->b r i (h d)')
        m_ = self.linear_out(out)
        return m_


class MSAColAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=8):
        super(MSAColAttention, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, dim * n_heads),
            nn.Sigmoid()
        )
        self.linear_out = nn.Linear(dim * n_heads, in_dim)
        self.n_heads = n_heads

    def forward(self, m):
        m = self.norm(m)
        q, k, v = torch.chunk(self.to_qkv(m), 3, -1)
        q, k, v = map(lambda x: rearrange(x, 'b i j (h d)->b i j h d', h=self.n_heads), (q, k, v))
        gate = self.to_gate(m)
        gate = rearrange(gate, 'b i j (h d)->b i j h d', h=self.n_heads)
        scale = q.size(-1) ** .5
        attn = (torch.einsum('bilhd,bjlhd->bhijl', q, k) / scale).softmax(-2)
        out = torch.einsum('bhijl,bjlhd->bilhd', attn, v)
        out = rearrange(gate * out, 'b i l h d->b i l (h d)')
        m_ = self.linear_out(out)
        return m_


class MSATransition(nn.Module):
    def __init__(self, dim=128, n=4):
        super(MSATransition, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * n)
        self.linear2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(dim * n, dim)
        )

    def forward(self, m):
        m = self.norm(m)
        m = self.linear1(m)
        m = self.linear2(m)
        return m


class OuterProductMean(nn.Module):
    def __init__(self, in_dim=128, dim=32):
        super(OuterProductMean, self).__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, dim * 2)
        self.linear_out = nn.Linear(dim ** 2, in_dim)

    def forward(self, m):
        nrow = m.size(1)
        m = self.norm(m)
        a, b = torch.chunk(self.linear(m), 2, -1)

        out = torch.einsum('bric,brjd->bijcd', a, b) / nrow
        out = rearrange(out, 'b i j c d->b i j (c d)')
        z = self.linear_out(out)
        return z


class TriangleMultiplication(nn.Module):
    def __init__(self, in_dim=128, dim=128, direct='outgoing'):
        super(TriangleMultiplication, self).__init__()
        self.direct = direct
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, dim * 2)
        self.linear2 = nn.Sequential(
            nn.Linear(in_dim, dim * 2),
            nn.Sigmoid()
        )
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.linear_out = nn.Linear(dim, in_dim)
        # self.linear_out.weight.data.fill_(0.)
        # self.linear_out.bias.data.fill_(0.)
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            self.linear_out
        )

    def forward(self, z):
        direct = self.direct
        z = self.norm(z)
        a, b = torch.chunk(self.linear2(z) * self.linear1(z), 2, -1)
        gate = self.to_gate(z)
        if direct == 'outgoing':
            prod = torch.einsum('bikd,bjkd->bijd', a, b)
        elif direct == 'incoming':
            prod = torch.einsum('bkid,bkjd->bijd', a, b)
        else:
            raise ValueError('direct should be outgoing or incoming!')
        out = gate * self.to_out(prod)
        return out


class TriangleAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=4, wise='row'):
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.to_out = nn.Linear(n_heads * dim, in_dim)
        # self.to_out.weight.data.fill_(0.)
        # self.to_out.bias.data.fill_(0.)

    def forward(self, z):
        wise = self.wise
        z = self.norm(z)
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)
        q, k, v = map(lambda x: rearrange(x, 'b i j (h d)->b i j h d', h=self.n_heads), (q, k, v))
        b = self.linear_for_pair(z)
        gate = self.to_gate(z)
        scale = q.size(-1) ** .5
        if wise == 'row':
            eq_attn = 'brihd,brjhd->brijh'
            eq_multi = 'brijh,brjhd->brihd'
            b = rearrange(b, 'b i j (r h)->b r i j h', r=1)
            softmax_dim = 3
        elif wise == 'col':
            eq_attn = 'bilhd,bjlhd->bijlh'
            eq_multi = 'bijlh,bjlhd->bilhd'
            b = rearrange(b, 'b i j (l h)->b i j l h', l=1)
            softmax_dim = 2

        else:
            raise ValueError('wise should be col or row!')
        attn = (torch.einsum(eq_attn, q, k) / scale + b).softmax(softmax_dim)
        out = torch.einsum(eq_multi, attn, v)
        out = gate * rearrange(out, 'b i j h d-> b i j (h d)')
        z_ = self.to_out(out)
        return z_


class PairTransition(nn.Module):
    def __init__(self, dim=128, n=4):
        super(PairTransition, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * n)
        self.linear2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim * n, dim)
        )

    def forward(self, z):
        z = self.norm(z)
        a = self.linear1(z)
        z = self.linear2(a)
        return z


class DistHead(nn.Module):
    def __init__(self, dim=128):
        super(DistHead, self).__init__()
        self.to_distograms = nn.ModuleDict({
            'dist': nn.Sequential(
                Symm('b i j d->b j i d'),
                nn.Linear(dim, 37)
            ),
            'omega': nn.Sequential(
                Symm('b i j d->b j i d'),
                nn.Linear(dim, 25)
            ),
            'phi': nn.Linear(dim, 13),
            'theta': nn.Linear(dim, 25)
        })

    def forward(self, logits):
        pred_distograms = {}
        for k, model in self.to_distograms.items():
            pred_distograms[k] = model(logits).softmax(-1)[0]
        return pred_distograms
