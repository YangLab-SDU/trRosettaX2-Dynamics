#!/usr/bin/env /public/home/wangwenkai/miniconda3/envs/server/bin/python

import resource
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.insert(0, '/home/wangwk')
sys.path.insert(0, '/home/wangwk/trRosetta_bin')
sys.path.insert(0, '/home/wangwenkai')
sys.path.insert(0, '/home/wangwenkai/trRosetta_bin')
import string
import os
import numpy as np
import pandas
import torch
import torch.nn as nn
from collections import defaultdict
from time import time, sleep

from einops.layers.torch import Rearrange

from utils_wwk.deeplearn import get_UUID, find_empty_gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sys.path.insert(0, '/public/home/yangserver/trRosetta/bin')
sys.path.insert(1, '/public/home/wangwenkai')
#os.environ["CUDA_VISIBLE_DEVICES"] = get_UUID(str(find_empty_gpu()))

import esm
from evoutils.attn_conv import Predictor2D

a3m_file = str(sys.argv[1])
npz_file = str(sys.argv[2])
out_file = str(sys.argv[3])

mname = 'attn_conv_crop_msatrf_afdb'
ncpu = 1
use_msatrf = True
filter_msa = True
window = 300
shift = 250

# MDIR = '/library/weights/'
MDIR = '/home/wangwenkai/a7d2/simply/params'
# esm_location = '/library/weights/esm_msa1_t12_100M_UR50S.pt'
esm_location = '/home/wangwenkai/a7d2/esm_models/esm_msa1_t12_100M_UR50S.pt'

torch.set_num_threads(ncpu)


# model_name = args.mname
# if f'{model_name}.index' not in os.listdir(MDIR):
#     raise FileNotFoundError(f'{model_name} not in MDIR, please check!')


def mymsa_to_esmmsa(msa, input_type='msa', in_torch=False):
    if in_torch:
        import torch
        device = msa.device
        token = torch.tensor([5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8,
                              11, 22, 19, 7, 30], device=device)
        cls = torch.zeros_like(msa[..., 0:1], device=device)
        eos = 2 * torch.ones_like(msa[..., 0:1], device=device)
        # token, cls, eos = map(lambda x: torch.from_numpy(x), [token, cls, eos])
        if input_type == 'fasta':
            return torch.cat([cls, token[msa], eos], dim=-1)
        else:
            return torch.cat([cls, token[msa]], dim=-1)
    else:
        token = np.array([5, 10, 17, 13, 23, 16, 9, 6, 21, 12, 4, 15, 20, 18, 14, 8,
                          11, 22, 19, 7, 30])
        cls = np.zeros_like(msa[..., 0:1])
        eos = 2 * np.ones_like(msa[..., 0:1])
        if input_type == 'fasta':
            return np.concatenate([cls, token[msa], eos], axis=-1)
        else:
            return np.concatenate([cls, token[msa]], axis=-1)


def parse_a3m(filename, limit=20000):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    i = 0
    for line in open(filename, "r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))
            i += 1
            if i == limit:
                break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa


class DistPredictorBaseline(nn.Module):
    def __init__(self):
        super(DistPredictorBaseline, self).__init__()
        self.net = Predictor2D(dim=128, depth=12, msa_tie_row_attn=True)

    def forward(self, msa, msa_cutoff=500, res_id=None):
        with torch.no_grad():
            f2d, f1d = self.get_f2d(msa[0])

        pred_distograms, reprs = self.net(f2d, msa[:, :msa_cutoff, :].long(), res_id=res_id,
                                          return_repr=True)  # dict((1, L, L, n_bins))
        for k in pred_distograms:
            pred_distograms[k] = pred_distograms[k].softmax(-1)
        return pred_distograms, reprs

    def get_f2d(self, msa):
        device = msa.device
        nrow, ncol = msa.size()[-2:]

        msa1hot = (torch.arange(21, device=device) == msa[..., None]).float()
        w = self.reweight(msa1hot, .8)

        # 1D features
        f1d_seq = msa1hot[0, :, :20]
        f1d_pssm = self.msa2pssm(msa1hot, w)

        f1d = torch.cat([f1d_seq, f1d_pssm], dim=1)

        # 2D features
        f2d_dca = self.fast_dca(msa1hot, w) if nrow > 1 else torch.zeros([ncol, ncol, 442], device=device)

        f2d = torch.cat([f1d[:, None, :].repeat([1, ncol, 1]),
                         f1d[None, :, :].repeat([ncol, 1, 1]),
                         f2d_dca], dim=-1)
        f2d = f2d.view([1, ncol, ncol, 442 + 2 * 42])
        return f2d, msa[0:1]

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


if use_msatrf:
    class DistPredictor(DistPredictorBaseline):
        def __init__(self):
            super(DistPredictor, self).__init__()
            self.net = Predictor2D(in_dim=1680 + 962, dim=128, depth=12, num_tokens=31, msa_tie_row_attn=True,
                                   gpu_lst=[0, 0])

        def forward(self, msa, msa_filtered, res_id=None):
            with torch.no_grad():
                f2d, f1d, msa_emb = self.get_f2d(msa, msa_filtered=msa_filtered)

            pred_distograms, reprs = self.net(f2d,
                                              msa_filtered[..., :500, 1:], msa_emb=msa_emb[None],
                                              res_id=res_id, return_repr=True)  # dict((1, L, L, n_bins))
            for k in pred_distograms:
                pred_distograms[k] = pred_distograms[k].softmax(-1)
            return pred_distograms, reprs

        def get_f2d(self, msa, msa_filtered=None):
            if msa_filtered is None:
                msa_filtered = msa
            device = msa.device
            nrow, ncol = msa.shape[-2], msa.shape[-1] - 1
            with torch.no_grad():
                emb_out = emb_pretrain(msa_filtered[..., :500, :], repr_layers=[12], need_head_weights=True)
            emb_repr = emb_out['representations'][12][0, :, 1:]
            seq_emb = emb_repr[0]
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

            f2d = f2d_dca.view([1, ncol, ncol, 962])
            return torch.cat([seq_emb, row_attn, f2d], dim=-1), msa[:, 0, 1:], emb_repr
else:
    DistPredictor = DistPredictorBaseline


def load_checkpoint(*models, pt, esm=False):
    model_CKPT = torch.load(pt, map_location=device)
    if esm:
        model, esm_model = models
        model.load_state_dict(model_CKPT['state_dict'])
        esm_model.load_state_dict(model_CKPT['esm_state_dict'])
        return model, esm_model
    else:
        model = models[0]
        model.load_state_dict(model_CKPT['state_dict'])
        return model


def main():
    if filter_msa:
        a3m_filtered = f'{a3m_file}_filter500'
        os.system(f'/home/wangwenkai/bin/hhsuite32/bin/hhfilter -i {a3m_file} -o {a3m_filtered} -diff 500')
    else:
        a3m_filtered = a3m_file

    msa = parse_a3m(a3m_file)
    msa_filtered = parse_a3m(a3m_filtered)
    if use_msatrf:
        msa = mymsa_to_esmmsa(msa)
        msa_filtered = mymsa_to_esmmsa(msa_filtered)

    msa = torch.from_numpy(msa).long().to(device)
    msa_filtered = torch.from_numpy(msa_filtered).long().to(device)

    seq = open(a3m_filtered).readlines()[1].strip()
    L = len(seq)
    with torch.no_grad():
        if L > window * 2:
            pred_dict = {
                # 'msa': torch.zeros((L, 128)),
                # 'pair': torch.zeros((L, L, 128)),
                'dist': torch.zeros((L, L, 37), device=msa.device),
                'omega': torch.zeros((L, L, 25), device=msa.device),
                'theta': torch.zeros((L, L, 25), device=msa.device),
                'phi': torch.zeros((L, L, 13), device=msa.device),
            }
            count_1d = torch.zeros((L), device=msa.device)
            count_2d = torch.zeros((L, L), device=msa.device)
            #
            grids = np.arange(0, L - window + shift, shift)
            ngrids = grids.shape[0]
            print("ngrid:     ", ngrids)
            print("grids:     ", grids)
            print("windows:   ", window)

            idx_pdb = torch.arange(L, device=msa.device).long().view(1, L)
            for i in range(ngrids):
                for j in range(i, ngrids):
                    start_1 = grids[i]
                    end_1 = min(grids[i] + window, L)
                    start_2 = grids[j]
                    end_2 = min(grids[j] + window, L)
                    sel = np.zeros((L)).astype(np.bool)
                    sel[start_1:end_1] = True
                    sel[start_2:end_2] = True

                    if use_msatrf:
                        input_msa = msa[:, np.array([True] + list(sel))]
                        mask = torch.sum(input_msa == 30, dim=-1) < 0.5 * sel.sum()  # remove too gappy sequences
                        input_msa_filtered = msa_filtered[:, np.array([True] + list(sel))]
                        mask_filtered = torch.sum(input_msa_filtered == 30,
                                                  dim=-1) < 0.5 * sel.sum()  # remove too gappy sequences
                    else:
                        input_msa = msa[:, sel]
                        mask = torch.sum(input_msa == 20, dim=-1) < 0.5 * sel.sum()  # remove too gappy sequences
                        input_msa_filtered = msa_filtered[:, sel]
                        mask_filtered = torch.sum(input_msa_filtered == 20,
                                                  dim=-1) < 0.5 * sel.sum()  # remove too gappy sequences

                    input_msa = input_msa[mask]
                    input_msa_filtered = input_msa_filtered[mask_filtered]
                    input_idx = idx_pdb[:, sel]

                    print("running crop: %d-%d/%d-%d" % (start_1, end_1, start_2, end_2), input_msa.shape)
                    with torch.cuda.amp.autocast():
                        # pred_distograms = model(input_msa, input_seq, input_idx, t1d=input_t1d, t2d=input_t2d)
                        pred_gemos, reprs = model(input_msa[None], msa_filtered=input_msa_filtered[None],
                                                  res_id=input_idx.to(input_msa.device))
                    # reprs['pair'] = Rearrange('b d i j->(b i) j d')(reprs['pair'])
                    # reprs['msa'] = reprs['msa'][0, 0]
                    weight = 1
                    sub_idx = input_idx[0].cpu()
                    sub_idx_2d = np.ix_(sub_idx, sub_idx)
                    count_2d[sub_idx_2d] += weight
                    count_1d[sub_idx] += weight

                    # pred_dict['pair'][sub_idx_2d] += weight * reprs['pair']
                    # pred_dict['msa'][sub_idx] += weight * reprs['msa']
                    pred_dict['dist'][sub_idx_2d] += weight * pred_gemos['dist'][0]
                    pred_dict['theta'][sub_idx_2d] += weight * pred_gemos['theta'][0]
                    pred_dict['omega'][sub_idx_2d] += weight * pred_gemos['omega'][0]
                    pred_dict['phi'][sub_idx_2d] += weight * pred_gemos['phi'][0]

            # pred_dict['msa'] = pred_dict['msa'] / count_1d[:, None]
            # pred_dict['pair'] = pred_dict['pair'] / count_2d[:, :, None]
            pred_dict['dist'] = pred_dict['dist'] / count_2d[:, :, None]
            pred_dict['theta'] = pred_dict['theta'] / count_2d[:, :, None]
            pred_dict['omega'] = pred_dict['omega'] / count_2d[:, :, None]
            pred_dict['phi'] = pred_dict['phi'] / count_2d[:, :, None]
            # pred_ca = model(msa[None], reprs=pred_dict)

        else:
            pred_gemos, reprs = model(msa[None], msa_filtered[None])
            pred_dict = {}
            # pred_dict['pair'] = Rearrange('b d i j->(b i) j d')(reprs['pair'])
            # pred_dict['msa'] = reprs['msa'][0, 0]
            pred_dict['dist'] = pred_gemos['dist'][0]
            pred_dict['theta'] = pred_gemos['theta'][0]
            pred_dict['omega'] = pred_gemos['omega'][0]
            pred_dict['phi'] = pred_gemos['phi'][0]
            # pred_ca = model(msa[None], reprs=pred_dict)
        for k in pred_dict:
            pred_dict[k] = pred_dict[k].detach().cpu().numpy()

    np.savez_compressed(npz_file, **pred_dict)

    # save CASP-style contacts
    w = np.sum(pred_dict['dist'][:, :, 1:13], axis=-1)
    L = w.shape[0]
    idx = np.array([[i + 1, j + 1, 0, 8, w[i, j]] for i in range(L) for j in range(i + 5, L)])
    out = idx[np.flip(np.argsort(idx[:, 4]))]

    data = [out[:, 0].astype(int), out[:, 1].astype(int), out[:, 2].astype(int), out[:, 3].astype(int),
            out[:, 4].astype(float)]
    df = pandas.DataFrame(data)
    df = df.transpose()
    df[0] = df[0].astype(int)
    df[1] = df[1].astype(int)
    df.columns = ["i", "j", "d1", "d2", "p"]
    df.to_csv(out_file, sep=' ', index=False)


if __name__ == '__main__':
    model = DistPredictor()
    model = load_checkpoint(model, pt=f'{MDIR}/{mname}.pth.tar', esm=False)
    model = model.to(device)
    model.eval()

    if use_msatrf:
        emb_pretrain, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_location)
        emb_pretrain = emb_pretrain.to(device)
        emb_pretrain.eval()

    main()
