import itertools
import os
import re
import string
import subprocess
from collections import defaultdict
import sys

import numpy as np
import scipy
from Bio import PDB
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from Bio.PDB import PPBuilder,PDBParser
from concurrent.futures import ThreadPoolExecutor
from trRosettaX2.evoutils.attn_conv import Predictor2D

# * Processing PDB to 2D geometries:dist, omega, theta, phi

# region
res_name_dict = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "PHD": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "UNK": "X",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "ASX": "B",
    "GLX": "Z",
    "XLE": "J",
    "XAA": "X",
}
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
retain_all_res = False


def get_dihedrals(a, b, c, d):
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.arctan2(y, x)

def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, None]

    x = np.sum(v * w, axis=1)

    return np.arccos(x)

def get_neighbors(xyzs, seq, dmax, many_seq=[]):
    nres = len(xyzs["CA"])
    # three anchor atoms
    N = xyzs["N"]
    Ca = xyzs["CA"]
    C = xyzs["C"]
    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
    Many_seq = 0
    if Cb.size - np.isnan(Cb).sum() <= 3:
        return True, "wrong", "wrong", "wrong", "wrong"
    if nres != len(seq):

        return True, "wrong", "wrong", "wrong", "wrong"
    if len(seq) <= 20:
        many_seq.append(Many_seq)
        print(len(many_seq))
        return True, "wrong", "wrong", "wrong", "wrong"
    for i in range(nres):
        if seq[i] != "G":
            try:
                Cb[i] = np.array(xyzs["CB"][i])
            except KeyError:
                pass

    # fast neighbors search
    nan_indices = np.isnan(Cb).any(axis=1)
    Cb = Cb[~nan_indices]
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array(
        [[i, j] for i in range(len(indices)) for j in indices[i] if i != j]
    ).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d = np.zeros((nres, nres))
    dist6d[idx0, idx1] = np.linalg.norm(Cb[idx1] - Cb[idx0], axis=-1)


    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0, idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])


    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0, idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])


    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0, idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    key = False
    return key, dist6d, omega6d, theta6d, phi6d

def pros(Dist, Omega=None, Theta_asym=None, Phi_asym=None, angle=False):
    Sdist, Somega, Stheta, Sphi = [], [], [], []
    SSdist, SSomega, SStheta, SSphi = [], [], [], []
    for i in range(len(Dist)):
        dist = Dist[i]
        Tdist = dist.reshape(-1, 1)
        Adist = np.array([np.arange(2, 20.5, 0.5).tolist()])
        Jdist = np.array([(Adist < Tdist).sum(axis=1)]).reshape(
            dist.shape[0], dist.shape[1]
        )
        Jdist = np.where(Jdist == 0, 0, Jdist)
        Jdist = np.where(Jdist >= 37, 0, Jdist)
        sdist = np.eye(37)[Jdist]
        Sdist.append(dist)
        SSdist.append(sdist)
        if angle:
            omega = Omega[i]
            Tomega = omega.reshape(-1, 1)
            Aomega = np.array([np.arange(-np.pi, np.pi, np.pi / 12).tolist()])
            Jomega = np.array([(Aomega < Tomega).sum(axis=1)]).reshape(
                omega.shape[0], omega.shape[1]
            )
            Jomega = np.where(Jdist == 0, 0, Jomega)
            Jomega = np.where(Jdist >= 37, 0, Jomega)
            somega = np.eye(25)[Jomega]
            Somega.append(omega)
            SSomega.append(somega)

            theta_asym = Theta_asym[i]
            Ttheta_asym = theta_asym.reshape(-1, 1)
            Atheta_asym = np.array([np.arange(-np.pi, np.pi, np.pi / 12).tolist()])
            Jtheta_asym = np.array([(Atheta_asym < Ttheta_asym).sum(axis=1)]).reshape(
                theta_asym.shape[0], theta_asym.shape[1]
            )
            Jtheta_asym = np.where(Jdist == 0, 0, Jtheta_asym)
            Jtheta_asym = np.where(Jdist >= 37, 0, Jtheta_asym)
            stheta_asym = np.eye(25)[Jtheta_asym]
            Stheta.append(theta_asym)
            SStheta.append(stheta_asym)

            phi_asym = Phi_asym[i]
            Tphi_asym = theta_asym.reshape(-1, 1)
            Aphi_asym = np.array([np.arange(0, np.pi, np.pi / 12).tolist()])
            Jphi_asym = np.array([(Aphi_asym < Tphi_asym).sum(axis=1)]).reshape(
                phi_asym.shape[0], phi_asym.shape[1]
            )
            Jphi_asym = np.where(Jdist == 0, 0, Jphi_asym)
            Jphi_asym = np.where(Jdist >= 37, 0, Jphi_asym)
            sphi_asym = np.eye(13)[Jphi_asym]
            Sphi.append(phi_asym)
            SSphi.append(sphi_asym)
    Sdist = [p.reshape(1, p.shape[0], p.shape[1]) for p in Sdist]
    SSdist = [p.reshape(1, p.shape[0], p.shape[1], p.shape[2]) for p in SSdist]
    if angle:
        Somega = [p.reshape(1, p.shape[0], p.shape[1]) for p in Somega]
        SSomega = [p.reshape(1, p.shape[0], p.shape[1], p.shape[2]) for p in SSomega]

        Stheta = [p.reshape(1, p.shape[0], p.shape[1]) for p in Stheta]
        SStheta = [p.reshape(1, p.shape[0], p.shape[1], p.shape[2]) for p in SStheta]

        Sphi = [p.reshape(1, p.shape[0], p.shape[1]) for p in Sphi]
        SSphi = [p.reshape(1, p.shape[0], p.shape[1], p.shape[2]) for p in SSphi]
        return np.array(SSdist), np.array(SStheta), np.array(SSomega), np.array(SSphi)
    else:
        return np.array(SSdist)

def get_atom_positions_pdb(pdb_file, model=0, retain_all_res=True):
    # load PDB
    pp = PDB.PDBParser(QUIET=True)
    structure = pp.get_structure("", pdb_file)[model]
    xyzs_all = defaultdict(list)
    # al_seq_pdb = []
    for chain in structure.child_list:
        residues = [
            res for res in chain.child_list if PDB.is_aa(res) and not res.id[0].strip()
        ]
        try:
            seq_pdb = "".join(res_name_dict[res.resname.strip()] for res in residues)
            # al_seq_pdb.append(seq_pdb)
        except KeyError:
            print("error")
        if retain_all_res:
            L = residues[-1].id[1]
            res_id = np.arange(L)
        else:
            L = len(residues)
            res_id = np.array([int(res.id[1]) for res in residues])

        xyzs = {}
        for atom in atom_types:
            xyzs[atom] = np.nan * np.zeros((L, 3))

        for i, res in enumerate(residues):
            for atom in atom_types:
                try:
                    atom_name = atom
                    coord = res[atom_name].coord
                    if retain_all_res:
                        xyzs[atom][res.id[1] - 1] = coord
                    else:
                        xyzs[atom][i] = coord
                except KeyError:
                    continue
        for atom in xyzs:
            xyzs_all[atom].append(xyzs[atom])
    # all_seq_pdb = ''.join(al_seq_pdb)
    for atom in xyzs_all:
        xyzs_all[atom] = np.concatenate(xyzs_all[atom], axis=0)
    return xyzs_all, res_id, seq_pdb

def get_distribution_from_pdb(pred_pdb_dir):
    xyzs, res_id, seq_pdb = get_atom_positions_pdb(
        pred_pdb_dir, model=0, retain_all_res=retain_all_res
    )
    key, dist, omega, theta_asym, phi_asym = get_neighbors(xyzs, seq_pdb, 20)
    if key:
        pass
    else:
        dist, omega, theta_asym, phi_asym = (
            dist[np.newaxis, :],
            omega[np.newaxis, :],
            theta_asym[np.newaxis, :],
            phi_asym[np.newaxis, :],
        )

    pros_fact = pros(dist, omega, theta_asym, phi_asym, angle=True)
    fact_dist, fact_theta, fact_omega, fact_phi = (
        pros_fact[0][0, 0, :, :, :],
        pros_fact[1][0, 0, :, :, :],
        pros_fact[2][0, 0, :, :, :],
        pros_fact[3][0, 0, :, :, :],
    )
    return fact_dist, fact_theta, fact_omega, fact_phi

# endregion

# * Generate new 2D geometries and structures based on the predicted 2D geometries and structures

# region

def params(flag):
    if flag == "0HHD":
        return 0, 0, 0.3, 0.03, 0.72
    if flag == "0LD":
        return 0, 0, 0.5, 0.07, 0.50
    if flag == "0HD":
        return 0, 0, 0.5, 0.05, 0.50
    if flag == "0LLD":
        return 0, 0, 0.7, 0.1, 0.42

def calculate_phi_psi(structure):
    phi_psi_list = []
    ppb = PPBuilder()

    for model in structure:
        for chain in model:
            polypeptides = ppb.build_peptides(chain)
            for poly in polypeptides:
                phi_psi = poly.get_phi_psi_list()
                phi_psi_list.extend(phi_psi)

    phi_psi_list = [(phi, psi) for phi, psi in phi_psi_list if phi is not None and psi is not None]
    return phi_psi_list

def ramachandran_score(phi_psi_list):
    allowed_region_count = 0

    for phi, psi in phi_psi_list:
        if (-180 <= phi <= 0) and (-180 <= psi <= 180):
            allowed_region_count += 1

    if len(phi_psi_list) == 0:
        return 0

    return allowed_region_count / len(phi_psi_list)

def calculate_reliability_score(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    # Calculate Ramachandran score
    phi_psi_list = calculate_phi_psi(structure)
    ramachandran_score_value = ramachandran_score(phi_psi_list)
    return ramachandran_score_value

def gaussian_smoothing(data, sigma):
    return gaussian_filter(data, sigma)

def process_distribution_with_pred_distribution(
    unprocessed_dist, fact_dist, norm=True, smooth=True, sigma=1.0
):
    tmp = np.copy(unprocessed_dist)
    processed_dist = np.copy(unprocessed_dist)
    
    backward, forward, P, pcut, decay_rate = params("0HD")
    mask = unprocessed_dist.max(axis=-1) < P

    
    for i, j in np.argwhere(mask):
        tmp1 = fact_dist[i, j]
        idx = np.argmax(tmp1)
        bw = backward if idx - backward >= 0 else idx
        fw = forward if idx + 1 + forward <= tmp1.size - 1 else tmp1.size - 1 - 1 - idx
        tmp2 = tmp[i, j][idx - bw : idx + 1 + fw]
        tmp[i, j][idx - bw : idx + 1 + fw] = np.where(
            tmp2 < pcut, tmp2, tmp2 * decay_rate
        )
        processed_dist[i, j] = tmp[i, j] / np.sum(tmp[i, j])
        if smooth:
            processed_dist[i, j] = gaussian_smoothing(processed_dist[i, j], sigma)
    if norm:
        return processed_dist
    else:
        return tmp

def get_npz_from_pred_pdb(
    unprocessed_npz_dir, pred_pdb_dir, tmp=False, simga=1.0, angle=True
):
    unprocessed_npz = np.load(unprocessed_npz_dir)
    if angle:
        unprocessed_dist, unprocessed_omega, unprocessed_theta, unprocessed_phi = (
            unprocessed_npz["dist"],
            unprocessed_npz["omega"],
            unprocessed_npz["theta"],
            unprocessed_npz["phi"],
        )

        # form pred_pdb get dist,omega,phi,theta
    else:
        unprocessed_dist = unprocessed_npz["dist"]
    fact_dist, fact_theta, fact_omega, fact_phi = get_distribution_from_pdb(
        pred_pdb_dir
    )

    if tmp:
        try:
            unprocessed_tmp = unprocessed_npz["tmp"]
        except KeyError:
            unprocessed_tmp = unprocessed_npz["dist"]
        processed_tmp = process_distribution_with_pred_distribution(
            unprocessed_dist=unprocessed_tmp,
            fact_dist=fact_dist,
            norm=False,
        )
        return processed_tmp
    # todo:
    # with the pred_distribution, process unprocessed_npz
    # highly_dynamic：P=0.3,pcut=0.03,decay_rate=0.72 low_dynamic:P=0.3,pcut=0.05,decay_rate=0.7
    if angle:
        processed_dist = process_distribution_with_pred_distribution(
            unprocessed_dist=unprocessed_dist,
            fact_dist=fact_dist,
            norm=True,
            smooth=True,
            sigma=simga,
        )
        processed_omega = process_distribution_with_pred_distribution(
            unprocessed_dist=unprocessed_omega,
            fact_dist=fact_omega,
            norm=True,
            smooth=True,
            sigma=simga,
        )
        processed_theta = process_distribution_with_pred_distribution(
            unprocessed_dist=unprocessed_theta,
            fact_dist=fact_theta,
            norm=True,
            smooth=True,
            sigma=simga,
        )
        processed_phi = process_distribution_with_pred_distribution(
            unprocessed_dist=unprocessed_phi,
            fact_dist=fact_phi,
            norm=True,
            smooth=True,
            sigma=simga,
        )
    
        return processed_dist, processed_omega, processed_theta, processed_phi
    else:
        processed_dist = process_distribution_with_pred_distribution(
            unprocessed_dist=unprocessed_dist,
            fact_dist=fact_dist,
            norm=True,
            smooth=True,
            sigma=simga,
        )
        return processed_dist

#endregion

# * Generate structure based on the predicted 2D geometries

# region

def folding_with_pred_npz(base_npz="../output/1TNQ/pred_npz/1TNQ_NMR.npz",
                base_fasta="../data/1TNQ-33_A.fasta",
                base_out="../output/1TNQ/pred_pdb/",
                out_name="pred_1TNQ",
                options="-m 2 -r no-idp --orient",
                repeat=0,
                start_id=0):
    base_command = f'{sys.executable} "./folding/folding.py"'
 
    os.makedirs(base_out, exist_ok=True)
        
    def run_command(i):
        out_file = os.path.join(base_out,f"{out_name}{i}.pdb")
        command = f"{base_command} -NPZ {base_npz} -FASTA {base_fasta} -OUT {out_file} {options}"
        subprocess.run(command, shell=True)
        print(f"Executed: {command}")
        

    if repeat:
        with ThreadPoolExecutor() as executor:
            executor.map(run_command, range(start_id, repeat + start_id))
    else:
        run_command('')

# endregion

# *cluster Generate structures

# region

def get_tmscore_and_rmsd(pred1,pred2):
    command = ['./bin/TMscore', pred1, pred2]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    result = result.stdout
    if "Structure1:" in result and "Structure2:" in result:
        rmsd = float(re.search(r"RMSD of  the common residues=\s+([\d.]+)", result).group(1))
        tm_score = float(re.search(r"TM-score    =\s+([\d.]+)", result).group(1))
        return tm_score, rmsd
    else:
        return "no_tm_score", "no_rmsd"

def get_tmscore_and_rmsd_matrix(pdb_dir):
    pdb_files = [i for i in os.listdir(pdb_dir) if i.endswith('.pdb')]
    tmscore_matrix = np.zeros((len(pdb_files), len(pdb_files)))
    rmsd_matrix = np.zeros((len(pdb_files), len(pdb_files)))
    for i, j in itertools.product(range(len(pdb_files)), repeat=2):
        if i <= j:
            continue
        pdb_file1 = pdb_files[i]
        pdb_file2 = pdb_files[j]
        pred1 = os.path.join(pdb_dir, pdb_file1)
        pred2 = os.path.join(pdb_dir, pdb_file2)
        tm_score, rmsd = get_tmscore_and_rmsd(pred1, pred2)
        tmscore_matrix[i][j] = tm_score
        rmsd_matrix[i][j] = rmsd
    return tmscore_matrix+tmscore_matrix.T, rmsd_matrix+rmsd_matrix.T,pdb_files

def get_glocon_matrix(pdb_dir):
    pdb_files = [i for i in os.listdir(pdb_dir) if i.endswith('.pdb')]
    glocon_matrix = np.zeros((len(pdb_files), len(pdb_files)))


    pdb_data = {}
    for file in pdb_files:
        pdb_file = os.path.join(pdb_dir, file)
        xyzs, res_id, seq = get_atom_positions_pdb(pdb_file)
        key, dist, omega, theta_asym, phi_asym = get_neighbors(xyzs, seq, 20)
        pdb_data[file] = dist
    
    for i, j in itertools.product(range(len(pdb_files)), repeat=2):
        if i <= j:
            continue
        pdb_file1 = pdb_files[i]
        pdb_file2 = pdb_files[j]
        dist1 = pdb_data[pdb_file1]
        dist2 = pdb_data[pdb_file2]

        dist_diff = np.abs(dist1 - dist2)
        dist_diff[dist_diff <= 3] = 0
        score = np.sum(np.triu(dist_diff))/(len(dist_diff)*(len(dist_diff)-1)/2)
        glocon_matrix[i][j] = score

    return glocon_matrix+glocon_matrix.T, pdb_files

def kmeans_clustering(glocon_matrix, pdb_files, n_clusters=10, draw=False):
    kmeans = KMeans(n_clusters=n_clusters,n_init=10, random_state=0).fit(glocon_matrix)
    labels = kmeans.labels_

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(pdb_files[i])

    if draw:
        plt.figure(figsize=(10, 7), dpi=100)
        plt.scatter(range(len(pdb_files)), [0]*len(pdb_files), c=labels, cmap='viridis', marker='o')
        plt.title('K-Means Clustering')
        plt.xlabel('PDB Files')
        plt.ylabel('Cluster Label')
        plt.xticks(range(len(pdb_files)), [i.split('.')[0] for i in pdb_files], rotation=90)
        plt.yticks(range(n_clusters))
        plt.show()

    return clusters

def save_cluster_result(pdb_dir,n_clusters=10,n_files=5,output_dir=None,mode='glocon'):
    if mode == 'glocon':
        glocon_matrix, pdb_files = get_glocon_matrix(pdb_dir)
    elif mode == 'tmscore':
        glocon_matrix,rmsd_matrix, pdb_files = get_tmscore_and_rmsd_matrix(pdb_dir)
    elif mode == 'rmsd':
        tmscore_matrix, glocon_matrix, pdb_files = get_tmscore_and_rmsd_matrix(pdb_dir)
    else:
        print("mode error")
        return 0
    if output_dir is None:
        output_dir = os.path.join(pdb_dir, "clusters_result")
    os.makedirs(output_dir, exist_ok=True)

    try:
        
        clusters = kmeans_clustering(glocon_matrix, pdb_files,n_clusters=n_clusters)
    except ValueError:
        return "no_cluster"
    for label, files in clusters.items():
        for i in range(n_files):
            try:
                os.system(f"cp {pdb_dir}/{files[i]} {output_dir}")
            except IndexError:
                break

# endregion

# *Predict 2D geometry using trRosettaX2

# region


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

def load_weights(model, weight_file,device,mode='weights_only'):

    if mode == 'all_model':
        mainnet = torch.load(weight_file, map_location=device,weights_only=False)
        weights = mainnet.state_dict()
    elif mode == 'weights_only':
        weights = torch.load(weight_file, map_location=device,weights_only=True)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    model_state_dict = model.state_dict()

    unloaded_layers = []

    for layer_name, _ in model_state_dict.items():
        if layer_name in weights:
            model_state_dict[layer_name] = weights[layer_name]
        else:
            unloaded_layers.append(layer_name)

    model.load_state_dict(model_state_dict)

    if unloaded_layers != []:
        print("Layers not load the weights:", unloaded_layers)
        return unloaded_layers
    else:
        print("All layers successfully load the weights")
        return model.to(device)

def parse_a3m(filename, limit=20000):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase + '*'))
    try:
        seq_len = len(open(filename, "r").readlines()[1].strip())
    except IndexError:
        pass
    # read file line by line
    count = 0
    for line in open(filename, "r"):
        # skip labels
        if line == ' ':
            continue
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            line = line.rstrip().translate(table)
            if len(line) != seq_len:
                continue
            seqs.append(line.rstrip().translate(table))
            count += 1
            if count >= limit:
                break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa

def pred_2d_geometry(
    trX2_model_pth, msa_file, save_dir=None, save_name=None, device=None
):
    mainnetwork = DistPredictorBaseline()
    trX2_model = load_weights(mainnetwork, trX2_model_pth, device, mode="weights_only")
    trX2_model.eval()
    if msa_file.split(".")[-1] == "a3m":
        msa = parse_a3m(msa_file)[None]
    if msa_file.split(".")[-1] == "npz":
        msa = np.load(msa_file)["msa"]
        if msa.ndim == 2:
            msa = msa[None]
    msa = torch.tensor(msa).to(device)
    with torch.no_grad():
        pred_2d, reprs = trX2_model(msa)
    dist = pred_2d["dist"].cpu().numpy()
    theta = pred_2d["theta"].cpu().numpy()
    phi = pred_2d["phi"].cpu().numpy()
    omega = pred_2d["omega"].cpu().numpy()
    labels = {}
    labels.update(
        {
            "dist": dist[0],
            "theta": theta[0],
            "omega": omega[0],
            "phi": phi[0],
        }
    )
    np.savez_compressed(os.path.join(save_dir, save_name), **labels)
    pass

# endregion
