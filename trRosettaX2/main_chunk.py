import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from trRosettaX2.evoutils.attn_conv_e2e import Predictor2D
from trRosettaX2.strutils.structure_module import StructureModuleFullAtom
from trRosettaX2.strutils.utils_3d.rigid_utils import Rigid


def one_hot(x, bin_values=torch.arange(2, 20.5, .5)):
    bin_values = bin_values.to(x.device)
    n_bins = len(bin_values)
    bin_values = bin_values.view([1] * x.ndim + [-1])
    binned = (bin_values <= x[..., None]).sum(-1)
    binned = torch.where(binned > n_bins - 1, n_bins - 1, binned)
    onehot = (torch.arange(n_bins, device=x.device) == binned[..., None]).float()
    return onehot


class InputEmbedder(nn.Module):
    def __init__(self):
        super(InputEmbedder, self).__init__()

    def forward(self, msa, emb_out=None):
        with torch.no_grad():
            f2d, f1d, msa_emb = self.get_f2d(msa, emb_out=emb_out)
        return {'pair': f2d, 'msa': msa_emb}

    def get_f2d(self, msa, emb_out):
        device = msa.device
        nrow, ncol = msa.shape[-2], msa.shape[-1] - 1
        # with torch.no_grad():
        #     emb_out = emb_pretrain(msa[:, :500, :], repr_layers=[12], need_head_weights=True)
        emb_repr = emb_out['representations'][12][:, :, 1:]
        seq_emb = emb_repr[0, 0]
        row_attn = emb_out['row_attentions'][:, :, :, 1:, 1:]
        seq_emb = torch.cat([
            seq_emb[None, ...].repeat(ncol, 1, 1),
            seq_emb[:, None, ...].repeat(1, ncol, 1),
        ], dim=-1)[None, ...]  # 1,L,L,2*768
        row_attn = Rearrange('b l h m n -> b m n (l h)')(row_attn)  # 1, L,L,144

        msa1hot = (torch.arange(31, device=device) == msa[0, :, 1:, None]).float()
        w = self.reweight(msa1hot, .8)

        # 2D features
        f2d_dca = self.fast_dca(msa1hot, w) if nrow > 1 else torch.zeros([ncol, ncol, 962], device=device)

        # f2d = torch.cat([f1d[:, None, :].repeat([1, ncol, 1]),
        #                  f1d[None, :, :].repeat([ncol, 1, 1]),
        #                  f2d_dca], dim=-1)
        f2d = f2d_dca.view([1, ncol, ncol, 962])
        return torch.cat([seq_emb, row_attn, f2d], dim=-1), msa[:, 0, 1:], emb_repr

    @staticmethod
    def msa2pssm(msa1hot, w):
        beff = w.sum()
        f_i = (w[:, None, None] * msa1hot).sum(axis=0) / beff + 1e-9
        h_i = (-f_i * torch.log(f_i)).sum(axis=1)
        return torch.cat([f_i, h_i[:, None]], dim=1)

    @staticmethod
    def reweight(msa1hot, cutoff):
        id_min = msa1hot.size(1) * cutoff
        id_mtx = torch.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])
        id_mask = id_mtx > id_min
        w = 1.0 / id_mask.sum(axis=-1)
        return w

    @staticmethod
    def fast_dca(msa1hot, weights, penalty=4.5):
        device = msa1hot.device
        nr, nc, ns = msa1hot.size()
        try:
            x = msa1hot.view(nr, nc * ns)
        except RuntimeError:
            x = msa1hot.contiguous().view(nr, nc * ns)
        num_points = weights.sum() - torch.sqrt(weights.mean())
        mean = (x * weights[:, None]).sum(dim=0, keepdims=True) / num_points
        x = (x - mean) * torch.sqrt(weights[:, None])
        cov = torch.matmul(x.permute(1, 0), x) / num_points

        cov_reg = cov + torch.eye(nc * ns, device=device) * penalty / torch.sqrt(weights.sum())
        inv_cov = torch.inverse(cov_reg)

        x1 = inv_cov.view(nc, ns, nc, ns)
        x2 = x1.permute(0, 2, 1, 3)
        features = x2.reshape(nc, nc, ns * ns)
        nc_eye = torch.eye(nc, device=device)
        x3 = torch.sqrt(torch.square(x1[:, :-1, :, :-1]).sum((1, 3))) * (1 - nc_eye)
        apc = x3.sum(axis=0, keepdims=True) * x3.sum(axis=1, keepdims=True) / x3.sum()
        contacts = (x3 - apc) * (1 - nc_eye)

        return torch.cat([features, contacts[:, :, None]], dim=2)


class RecyclingEmbedder(nn.Module):
    def __init__(self, dim=128):
        super(RecyclingEmbedder, self).__init__()
        self.linear = nn.Linear(37, dim)
        self.norm_pair = nn.LayerNorm(dim)
        self.norm_msa = nn.LayerNorm(dim)

    def forward(self, reprs_prev, x=None):
        if x is None:
            x = reprs_prev['x']
        d = torch.cdist(x, x)
        d = one_hot(d)
        d = self.linear(d)
        pair = self.norm_pair(reprs_prev['pair']) + d
        single = self.norm_msa(reprs_prev['single'])
        return single, pair


class Folding(nn.Module):
    def __init__(self, dim_2d=128, dim_3d=32, layers_3d=8, dropout=.1, config={}):
        super(Folding, self).__init__()
        self.config = config
        self.input_embedder = InputEmbedder()
        self.recycle_embedder = RecyclingEmbedder(dim=dim_2d)
        self.layers_3d = layers_3d
        self.net2d = Predictor2D(in_dim=1680 + 962, dim=128, depth=12, num_tokens=31, msa_tie_row_attn=True,
                                 attn_dropout=dropout, ff_dropout=dropout)

        self.structure_module = StructureModuleFullAtom(
            **config['structure_module']
        )
        self.to_plddt = nn.Sequential(
            nn.LayerNorm(dim_3d),
            nn.Linear(dim_3d, dim_3d),
            nn.ReLU(),
            nn.Linear(dim_3d, dim_3d),
            nn.ReLU(),
            nn.Linear(dim_3d, 50),
        )

    def forward(self, raw_seq, msa, msa_filtered=None, emb_out=None, res_id=None, n_recycle=3,
                device='cuda:0', msa_cutoff=500):
        config = self.config
        reprs_prev = None
        outputs_all = {}
        L = len(raw_seq)
        if msa_filtered is None: msa_filtered = msa
        feats = self.input_embedder(msa, emb_out=emb_out)
        for c in range(1 + n_recycle):
            with torch.no_grad():
                with torch.amp.autocast(device_type=str(device).split(':')[0]):
                    if reprs_prev is None:
                        reprs_prev = {
                            'pair': torch.zeros((1, L, L, 128), device=device),
                            'single': torch.zeros((1, L, 128), device=device),
                            'x': torch.zeros((1, L, 3), device=device),
                        }
                        t = reprs_prev['x']

                    rec_single, rec_pair = self.recycle_embedder(reprs_prev, t)
                    # reprs['msa'][:, 0] = reprs['msa'][:, 0] + rec_single
                    # reprs['pair'] = reprs['pair'] + rec_pair
                    rec_reprs = {'single': rec_single, 'pair': rec_pair}

                    pred_gemos, reprs = self.net2d(f2d=feats['pair'], msa=msa_filtered[:, :msa_cutoff, 1:],
                                                   msa_emb=feats['msa'][:, :msa_cutoff, :],
                                                   rec_reprs=rec_reprs,
                                                   res_id=res_id, preprocess=True, to_prob=True, return_repr=True)
                    reprs['pair'] = rearrange(reprs['pair'], 'b d i j -> b i j d')

                _reprs = {"single": reprs['msa'][:, 0], "pair": reprs['pair']}
                rigids = None

                outputs = self.structure_module.forward(raw_seq, _reprs, rigids=rigids, return_mid=c == n_recycle)
                # ['frames', 'unnormalized_angles', 'angles', 'single', 'cord_tns_pred', "cords_c1'"]

                rots = []
                tsls = []
                for frames in outputs['scaled_frames']:
                    rigid = Rigid.from_tensor_7(frames, normalize_quats=True)
                    fram = rigid.to_tensor_4x4()
                    rots.append(fram[:, :, :3, :3])
                    tsls.append(fram[:, :, :3, 3:].squeeze(-1))
                outputs['frames'] = (torch.stack(rots, dim=0), torch.stack(tsls, dim=0))
                outputs['geoms'] = pred_gemos

                reprs_prev = {
                    'single': reprs['msa'][:, 0, :, :].detach(),
                    'pair': reprs['pair'].detach(),
                    "x": outputs["frames"][1][-1].detach(),
                }
            plddt_prob = self.to_plddt(outputs['single'][-1]).softmax(-1)
            plddt = torch.einsum('bik,k->bi', plddt_prob, torch.arange(0.01, 1.01, 0.02, device=device))
            outputs['plddt_prob'] = plddt_prob
            outputs['plddt'] = plddt

            outputs_all[c] = (outputs)
            torch.cuda.empty_cache()
        return outputs_all, outputs
