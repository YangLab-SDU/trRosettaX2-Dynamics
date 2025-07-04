import math

import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
from itertools import islice, cycle
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from math import sqrt
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from .dropout import *

from .modules import TriangleAttention, TriangleMultiplication, PairTransition


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class PreNormCross(nn.Module):
    def __init__(self, dim1, dim2, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim1)
        self.norm_context = nn.LayerNorm(dim2)

    def forward(self, x, context, *args, **kwargs):
        x = self.norm(x)
        context = self.norm_context(context)
        return self.fn(x, context, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.feed_forward(x)


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, dilation=1, baseWidth=26, scale=4, stype='normal', expansion=4,
                 shortcut=True, norm=nn.InstanceNorm2d):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        self.expansion = expansion

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1)
        self.bn1 = norm(inplanes, affine=True)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation))
            bns.append(norm(width, affine=True))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1)
        self.bn3 = norm(width * scale, affine=True)

        # self.conv_st = nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1)

        self.relu = nn.ELU(inplace=True)
        self.stype = stype
        self.scale = scale
        self.width = width
        self.shortcut = shortcut

    def forward(self, x, mask=None):
        if mask is None:
            mask2d = torch.ones_like(x[:, 0:1, :, :])
        else:
            mask2d = mask.permute(0, 3, 1, 2)
        residual = x
        if mask is not None:
            out = self.bn1(x, mask=mask)
        else:
            out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            if mask is not None:
                bni = partial(self.bns[i], mask=mask)
            else:
                bni = self.bns[i]
            sp = self.relu(bni(sp))
            sp = self.convs[i](sp * mask2d)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        if self.stype == 'stage':
            residual = self.conv_st(residual)
        if mask is not None:
            bn3 = partial(self.bn3, mask=mask)
        else:
            bn3 = self.bn3
        out = bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.shortcut:
            out += residual

        return out * mask2d


class TriUpdate(nn.Module):
    def __init__(
            self,
            in_dim=128,
            # dim_outer=32,
            n_heads=4,
            dim_pair_multi=128,
            dim_pair_attn=32,
            dropout_rate_pair=.10,
    ):
        super(TriUpdate, self).__init__()

        self.ps_dropout_row_layer = DropoutRowwise(dropout_rate_pair)
        self.ps_dropout_col_layer = DropoutColumnwise(dropout_rate_pair)

        self.pair_multi_out = TriangleMultiplication(in_dim=in_dim, dim=dim_pair_multi, direct='outgoing')
        self.pair_multi_in = TriangleMultiplication(in_dim=in_dim, dim=dim_pair_multi, direct='incoming')
        self.pair_row_attn = TriangleAttention(in_dim=in_dim, dim=dim_pair_attn, n_heads=n_heads, wise='row')
        self.pair_col_attn = TriangleAttention(in_dim=in_dim, dim=dim_pair_attn, n_heads=n_heads, wise='col')

        self.pair_trans = PairTransition(dim=in_dim)

        self.conv_stem = nn.ModuleList(
            [
                nn.Sequential(
                    Rearrange('b i j d->b d i j'),
                    Bottle2neck(in_dim, in_dim, expansion=1, dilation=1, shortcut=False),
                    Rearrange('b d i j->b i j d'),
                )
                for _ in range(4)
            ]
        )

    def forward(self, z):
        z = z + self.ps_dropout_row_layer(self.pair_multi_out(z)) + self.conv_stem[0](z)
        z = z + self.ps_dropout_row_layer(self.pair_multi_in(z)) + self.conv_stem[1](z)
        pair_row_attn = self.pair_row_attn

        if z.requires_grad:
            z = z + self.ps_dropout_row_layer(checkpoint(pair_row_attn, z, use_reentrant=True)) + self.conv_stem[2](z)
        else:
            z = z + self.ps_dropout_row_layer(pair_row_attn(z)) + self.conv_stem[2](z)
        pair_col_attn = self.pair_col_attn
        if z.requires_grad:
            z = z + self.ps_dropout_row_layer(checkpoint(pair_col_attn, z, use_reentrant=True)) + self.conv_stem[3](z)
        else:
            z = z + self.ps_dropout_row_layer(pair_col_attn(z)) + self.conv_stem[3](z)
        z = z + self.pair_trans(z)

        return z


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_pair=None,
            seq_len=None,
            heads=8,
            dim_head=64,
            dropout=0.,
            tie_attn_dim=None,
            soft_tied=False,
            use_pair=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        if use_pair:
            self.pair_norm = nn.LayerNorm(dim_pair)
            self.pair_linear = nn.Linear(dim_pair, heads, bias=False)

            self.for_pair = nn.Sequential(
                self.pair_norm, self.pair_linear
            )

        self.dropout = nn.Dropout(dropout)

        self.tie_attn_dim = tie_attn_dim
        self.soft_tied = soft_tied
        if soft_tied:
            self.seq_weight = PositionalWiseWeight(n_heads=heads)

    def forward(self, *args, context=None, tie_attn_dim=None, return_attn=False):
        if len(args) == 2:
            x, pair_bias = args
        elif len(args) == 1:
            x, pair_bias = args[0], None
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)
        # orig: (B*R, L, D)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # for tying row-attention, for MSA axial self-attention

        if exists(tie_attn_dim):
            q, k, v = map(lambda t: rearrange(t, '(b r) h n d -> b r h n d', r=tie_attn_dim), (q, k, v))
            if self.soft_tied:
                w = self.seq_weight(rearrange(x, '(b r) l d -> b r l d', r=tie_attn_dim))  # b, L, H, R
                dots = einsum('b i h r, b r h i d, b r h j d -> b h i j', w, q, k) * self.scale
            else:
                dots = einsum('b r h i d, b r h j d -> b h i j', q, k) * self.scale * (tie_attn_dim ** -0.5)

        else:
            # q, k, v = map(lambda t: rearrange(t, '(b r) h n d -> b r h n d', r=tie_attn_dim), (q, k, v))

            #  SA:(B R H L D), (B R H L D) -> (B H R L L)
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # b=R

        # attention
        if pair_bias is not None:
            dots += rearrange(self.for_pair(pair_bias), 'b i j h -> b h i j')  # b=1
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate

        if exists(tie_attn_dim):
            out = einsum('b h i j, b r h j d -> b r h i d', attn, v)
            out = rearrange(out, 'b r h n d -> (b r) h n d')
        else:
            out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # combine heads and project out
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if return_attn:
            return rearrange(out, '(b r) n d -> b r n d', b=1), attn
        else:
            return rearrange(out, '(b r) n d -> b r n d', b=1)


class MSAAttention(nn.Module):
    def __init__(
            self,
            tie_row_attn=True,
            attn_class=SelfAttention,
            dim=64,
            **kwargs
    ):
        super().__init__()

        self.tie_row_attn = tie_row_attn  # tie the row attention, from the paper 'MSA Transformer'

        self.attn_width = attn_class(dim, **kwargs)
        self.attn_height = attn_class(dim, use_pair=True, **kwargs)

    def forward(self, *args, return_attn=False):
        if len(args) == 2:
            x, pair_bias = args
        if len(args) == 1:
            x, pair_bias = args[0], None

        # col-wise
        w_x = rearrange(x, 'b h w d -> (b w) h d')
        if w_x.requires_grad:
            w_out = checkpoint(self.attn_width, w_x, use_reentrant=True)
        else:
            w_out = self.attn_width(w_x)

        # row-wise
        tie_attn_dim = x.shape[1] if self.tie_row_attn else None
        h_x = rearrange(x, 'b h w d -> (b h) w d')
        attn_height = partial(self.attn_height, tie_attn_dim=tie_attn_dim, return_attn=return_attn)
        if h_x.requires_grad:
            h_out = checkpoint(attn_height, h_x, pair_bias, use_reentrant=True)
        else:
            h_out = attn_height(h_x, pair_bias)
        if return_attn:
            h_out, attn = h_out

        out = w_out.permute(0, 2, 1, 3) + h_out
        out /= 2
        if return_attn:
            return out, attn
        return out


class PositionalWiseWeight(nn.Module):
    def __init__(self, d_msa=128, n_heads=4):
        super(PositionalWiseWeight, self).__init__()
        self.to_q = nn.Linear(d_msa, d_msa)
        self.to_k = nn.Linear(d_msa, d_msa)
        self.n_heads = n_heads

    def forward(self, m):
        q = self.to_q(m[:, 0:1, :, :])  # b,1,L,d
        k = self.to_k(m)  # b,L,L,d

        q = rearrange(q, 'b i j (h d) -> b j h i d', h=self.n_heads)
        k = rearrange(k, 'b i j (h d) -> b j h i d', h=self.n_heads)
        scale = q.size(-1) ** .5
        attn = torch.einsum('bjhud,bjhid->bjhi', q, k) / scale
        return attn.softmax(dim=-1)  # b, L, H, R


class UpdateX(nn.Module):
    def __init__(self, in_dim=128, dim_msa=32, dim=128):
        super(UpdateX, self).__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa ** 2, dim)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x, m):
        m = self.proj_down1(m)  # b,r,l,d
        nrows = m.shape[1]
        outer_product = torch.einsum('brid,brjc -> bijcd', m / nrows, m)
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        outer_product = self.proj_down2(outer_product)
        pair_feats = x + outer_product

        return pair_feats


class UpdateM(nn.Module):
    def __init__(self, in_dim=128, pair_dim=128, n_heads=8):
        super(UpdateM, self).__init__()
        self.norm1 = nn.LayerNorm(pair_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(pair_dim, n_heads)
        self.linear2 = nn.Linear(in_dim, in_dim // n_heads)
        self.ff = FeedForward(in_dim, dropout=.1)
        self.n_heads = n_heads

    def forward(self, x, m):
        pair_feats = (x + rearrange(x, 'b i j d->b j i d')) / 2
        pair_feats = self.norm1(pair_feats)
        attn = self.linear1(pair_feats).softmax(-2)  # b i j h
        values = self.norm2(m)
        values = self.linear2(values)  # b r l d
        attn_out = torch.einsum('bijh,brjd->brihd', attn, values)
        attn_out = rearrange(attn_out, 'b r l h d -> b r l (h d)')
        out = m + attn_out
        residue = self.norm3(out)
        return out + self.ff(residue)


class relpos(nn.Module):

    def __init__(self, dim=128):
        super(relpos, self).__init__()
        self.linear = nn.Linear(65, dim)

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


class InputEmbedder(nn.Module):
    def __init__(self, dim=128):
        super(InputEmbedder, self).__init__()
        self.relpos = relpos(dim=dim)

    def forward(self, z, res_id):
        z = z + self.relpos(res_id)
        return z

# main class
class SequentialSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(
            self,
            x,
            m,
    ):
        layer_ind = 0
        for attn2d, msa_attn, cross_attn, msa_ff, msa_cross_attn in self.blocks:
            layer_ind += 1

            m_out = msa_attn(m, x, return_attn=False)
            m = m_out + m
            m = m + msa_ff(m)

            # cross attention
            x = cross_attn(x, m)
            x = attn2d(x)

            m = msa_cross_attn(x, m)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return x, m


class Predictor2D(nn.Module):
    def __init__(
            self,
            *,
            dim,
            in_dim=526,
            max_seq_len=2048,
            depth=6,
            heads=8,
            dim_head=64,
            num_tokens=21,
            attn_dropout=0.,
            ff_dropout=0.,
            msa_tie_row_attn=False,
            gpu_lst=[0, 0]
    ):
        super().__init__()

        self.gpu_lst = gpu_lst

        self.bn1 = nn.InstanceNorm2d(in_dim, affine=True)
        self.elu1 = nn.ELU(inplace=True)
        self.conv1 = nn.Conv2d(in_dim, dim, 1)
        self.linear1 = nn.Sequential(
            self.bn1,
            self.elu1,
            self.conv1
        )
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.linear_emb = nn.Linear(768, dim)
        self.input_emb = InputEmbedder(dim=dim)

        # main trunk modules

        layers = nn.ModuleList([])

        for layer_ind in range(depth):
            prenorm = partial(PreNorm, dim)

            layers.append(nn.ModuleList([
                TriUpdate(in_dim=dim),  # for seq
                prenorm(MSAAttention(dim=dim, dim_pair=dim, seq_len=max_seq_len, heads=heads, dim_head=dim_head,
                                     dropout=attn_dropout, tie_row_attn=msa_tie_row_attn,
                                     )),
                UpdateX(in_dim=dim, dim=dim),
                prenorm(FeedForward(dim=dim, dropout=ff_dropout)),
                UpdateM(in_dim=dim, pair_dim=dim),

            ]))

        self.net = SequentialSequence(layers)

        # to distogram output

        self.to_dist_logits = nn.Conv2d(dim, 37, 1)
        self.to_theta_logits = nn.Conv2d(dim, 25, 1)
        self.to_omega_logits = nn.Conv2d(dim, 25, 1)
        self.to_phi_logits = nn.Conv2d(dim, 13, 1)

    def forward(
            self,
            f2d,
            msa=None,
            res_id=None,
            msa_emb=None,
            preprocess=True,
            return_repr=False,
            to_prob=False,
            rec_reprs=None
    ):
        n, device = f2d.shape[1], f2d.device
        if res_id is None:
            res_id = torch.arange(n, device=device)
        res_id = res_id.view(1, n)

        if preprocess:
            x = f2d.permute(0, 3, 1, 2)
            x = self.linear1(x).permute(0, 2, 3, 1)

            m = self.token_emb(msa)
            if exists(msa_emb):
                m += self.linear_emb(msa_emb)
        else:
            x, m = f2d, msa_emb
        if rec_reprs is not None:
            m[:, 0] = m[:, 0] + rec_reprs['single']
            x = x + rec_reprs['pair']
        x = self.input_emb(x, res_id)
        x, m = self.net(x, m)
        x = x.permute(0, 3, 1, 2)

        trunk_embeds = (x + rearrange(x, 'b d i j -> b d j i')) * 0.5  # symmetrize
        dist_logits = self.to_dist_logits(trunk_embeds)
        theta_logits = self.to_theta_logits(x)
        omega_logits = self.to_omega_logits(trunk_embeds)
        phi_logits = self.to_phi_logits(x)

        pred2d = {
            'dist': dist_logits.permute(0, 2, 3, 1),
            'phi': phi_logits.permute(0, 2, 3, 1),
            'theta': theta_logits.permute(0, 2, 3, 1),
            'omega': omega_logits.permute(0, 2, 3, 1)
        }
        if to_prob:
            pred2d = {k: v.softmax(-1) for (k, v) in pred2d.items()}

        if return_repr:
            reprs = {'pair': x, 'msa': m}
            return pred2d, reprs
        return pred2d
