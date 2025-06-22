import sys, os
import tempfile

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
from trRosettaX2.main_chunk import Folding
from trRosettaX2 import esm
from trRosettaX2.strutils.utils_3d.prot_converter import ProtConverter
from utils_trX2dy.utils import read_json, parse_a3m, mymsa_to_esmmsa

parser = ArgumentParser()
parser.add_argument('-i',
                    '--msa',
                    required=True,
                    help='(required) input MSA file')
parser.add_argument('-o',
                    '--out_dir',
                    required=True,
                    help='(required) output directory')
parser.add_argument('-mdir', '--model_pth',
                    default=f'model_pth/trX2_orig',
                    help='pretrained params directory')
parser.add_argument('-mname',
                    '--model_name',
                    default='model_1',
                    help='model name')
parser.add_argument('-nrows',
                    '--nrows',
                    default=500, type=int,
                    help='maximum number of rows in the MSA repr (default: 500).')
parser.add_argument('-nrec',
                    '--num_recycle',
                    default=3, type=int,
                    help='number of recyclings (default: 3).')
parser.add_argument('-gpu',
                    '--gpu',
                    default=0,
                    type=int,
                    help='use which gpu')
parser.add_argument('-cpu',
                    '--cpu',
                    default=4, type=int,
                    help='number of CPUs to use')

args = parser.parse_args()


def get_esm_emb(msa, res_id, C):
    with torch.no_grad() and torch.amp.autocast(device_type=str(device).split(':')[0]):
        emb_out = emb_pretrain(msa, repr_layers=[12], need_head_weights=True,
                               res_idx=res_id.repeat((C, 1)).to(device) if res_id is not None else res_id)
        emb_repr = emb_out['representations'][12].detach().contiguous()
        row_attn = emb_out['row_attentions'].detach().contiguous()
        del emb_out
        torch.cuda.empty_cache()
    return emb_repr, row_attn


def predict(model, seq, msa, ss):
    with torch.no_grad():
        L = len(seq)
        res_id = torch.arange(L, device=device).view(1, L)

        if ss is not None:
            ss = ss.squeeze().to(device)
            if not (ss.shape[-1] == msa.shape[-1] == L):
                raise ValueError(
                    f'Length mismatch: seq length {L}, ss length {ss.shape[-1]}, msa length {msa.shape[-1]}!')
            ss = ss.view(1, L, L)
        msa = msa.view(1, -1, L)
        outputs_all, outputs = model(seq, msa, ss, res_id=res_id.to(device), num_recycle=args.num_recycle,
                                     msa_cutoff=args.nrows, config=config)

    outputs_tosave_all = {}
    for c in outputs_all:
        outputs = outputs_all[c]
        outputs_tosave = {}
        for k in outputs:
            if isinstance(outputs[k], torch.Tensor):
                outputs_tosave[k] = outputs[k].cpu().detach().numpy()
            elif k == 'frames':
                outputs_tosave[k] = {
                    'R': outputs[k][0].cpu().detach().numpy(), 't': outputs[k][1].cpu().detach().numpy()
                }
            elif k == 'frames_allatm':
                outputs_tosave[k] = {
                    fname: {'R': tup[0].cpu().detach().numpy(), 't': tup[1].cpu().detach().numpy()}
                    for fname, tup in outputs[k].items()
                }
            elif k == 'geoms':
                pred_dict = outputs['geoms']['inter_labels']
                for kk in pred_dict:
                    for kkk in pred_dict[kk]:
                        pred_dict[kk][kkk] = pred_dict[kk][kkk].cpu().detach().numpy()
                outputs_tosave['inter_labels'] = pred_dict
        outputs_tosave_all[c] = outputs_tosave

    return outputs_tosave_all, outputs_all


if __name__ == '__main__':

    torch.set_num_threads(args.cpu)
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')
    print(f'run on {device}')
    py = sys.executable

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print('---------Load model and config---------')
    config = read_json(f'{args.model_pth}/{args.model_name}.json')
    model_ckpt = torch.load(f'{args.model_pth}/{args.model_name}.pth.tar', map_location=device, weights_only=True)

    model = Folding(dim_2d=config['dim_pair'], dim_3d=config['dim_str'], config=config).to(device)
    model.load_state_dict(model_ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if config['use_esm_msa']:
        esm_location = f'{args.model_pth}/esm_msa1_t12_100M_UR50S.pt'
        emb_pretrain, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_location)
        emb_pretrain.eval()
        emb_pretrain = emb_pretrain.to(device)
        for param in emb_pretrain.parameters():
            param.require_grad = False

    print('------------Read input file------------')
    cwd = os.getcwd()
    msa = parse_a3m(args.msa, limit=30000)
    if len(msa) == 1:
        msa = np.concatenate([msa, msa], axis=0)

    if len(msa) > 1.1 * args.nrows:
        a3m_filtered = f'{out_dir}/filter{args.nrows}.a3m'
        if not os.path.isfile(a3m_filtered):
            os.system(f'hhfilter -i {args.msa} -o {a3m_filtered} -diff {args.nrows}')
        msa_filtered = parse_a3m(a3m_filtered, limit=args.nrows)
    else:
        msa_filtered = msa

    if config['use_esm_msa']:
        msa = mymsa_to_esmmsa(msa)
        msa_filtered = mymsa_to_esmmsa(msa_filtered)
    msa = torch.from_numpy(msa).to(device)
    msa_filtered = torch.from_numpy(msa_filtered).to(device)

    raw_seq = open(args.msa).readlines()[1].strip().replace('-', '')
    L = len(raw_seq)
    res_id = torch.arange(L).view(1, L)

    print('----------------Predict----------------')
    if config['use_esm_msa']:
        if config['change_idx']:
            emb_repr, row_attn = get_esm_emb(msa_filtered[None], res_id, 1)
        else:
            emb_repr, row_attn = get_esm_emb(msa_filtered[None], None, 1)
    else:
        emb_out = None
    if emb_repr.max() == 0 or row_attn.max() == 0:
        raise ValueError(f'ESM-MSA fails to generate embeddings! Please check the input MSA file: {args.msa}.')
    emb_out = {'representations': {12: emb_repr}, 'row_attentions': row_attn}

    _, outputs = model.forward(raw_seq, msa[None], msa_filtered=msa_filtered[None],
                               res_id=res_id.to(msa.device),
                               emb_out=emb_out,
                               n_recycle=config['max_recycle'],
                               msa_cutoff=args.nrows,
                               device=device)
    print('-------------Save results--------------')

    out_npz_file = f'{out_dir}/{args.model_name}_results.npz'
    plddt = outputs['plddt'][-1].squeeze().cpu().numpy()

    np.savez_compressed(out_npz_file, plddt=plddt,
                        **{k: v.squeeze().cpu().numpy() for k, v in outputs['geoms'].items()})

    unrelaxed_model = f'{out_dir}/{args.model_name}.pdb'
    # refined_model = os.path.abspath(f'{out_dir}/{args.model_name}_relaxed{args.refine_steps}.pdb')
    cords_prot = outputs['cords_allatm'][-1].squeeze(0).permute(1, 0, 2)

    lines_prot = ProtConverter.export_pdb_file(raw_seq,
                                               cords_prot.squeeze(0).data.cpu().numpy(),
                                               path=None, chain_id='A', ca_only=False,
                                               confidence=plddt,
                                               )
    os.makedirs(os.path.dirname(unrelaxed_model), exist_ok=True)
    with open(unrelaxed_model, 'w') as f:
        f.write(''.join(lines_prot))

    table = pd.DataFrame()

    for i in range(len(raw_seq)):
        table.loc[i + 1, 'pLDDT'] = plddt[i]
    table.index.names = ['Residue_Index']
    table.to_csv(os.path.abspath(f'{out_dir}/plddt.csv'))

    print('done!')
    plddt_global = plddt.mean()
    print(f'pLDDT: {plddt_global:.3f}')
