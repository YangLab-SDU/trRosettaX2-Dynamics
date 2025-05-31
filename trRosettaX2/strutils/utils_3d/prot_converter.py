import os
from time import time

import torch
from torch import nn

from a7d2.e2e.models.protein_constants import *
from RNA.models.utils_rhofold.rigid_utils import Rigid, Rotation, calc_rot_tsl


def calc_angls_prot(coords, seq, psi_angle_atoms=psi_angle_atoms, chi_angles_atoms=chi_angles_atoms):
    """ note: the first three pos in dim2 is invalid """
    L = (coords["CA"]).shape[-2]
    if seq is None:
        seq = 'X'*L
    rots = torch.nan * torch.ones((L, 8, 3, 3), device=coords["CA"].device)
    angls = torch.nan * torch.ones((L, 8, 2), device=coords["CA"].device)

    rots[:, 0] = torch.eye(3, device=coords["CA"].device)
    angls[:, 0, 0] = 0  # sin
    angls[:, 0, 1] = 1  # cos

    for restype in restype_3to1:
        angle_atoms = [psi_angle_atoms] + chi_angles_atoms[restype]
        indices = [ind for ind, aa in enumerate(seq) if aa == restype_3to1[restype]]
        for i, atms in enumerate(angle_atoms):
            # rot_default, tsl_default = _make_rigid_transformation_rt(
            #     ex=coords[atms[2]][indices] - coords[atms[1]][indices],
            #     ey=coords[atms[0]][indices] - coords[atms[1]][indices],
            #     translation=coords[atms[2]][indices])
            rot_default, tsl_default = calc_rot_tsl(
                coords[atms[0]].squeeze(0)[indices],
                coords[atms[1]].squeeze(0)[indices],
                coords[atms[2]].squeeze(0)[indices]
            )
            # rot_cur, tsl_cur = _make_rigid_transformation_rt(
            #     ex=coords[atms[2]][indices] - coords[atms[1]][indices],
            #     ey=coords[atms[3]][indices] - coords[atms[2]][indices],
            #     translation=coords[atms[2]][indices])

            rot_cur, tsl_cur = calc_rot_tsl(
                coords[atms[3]].squeeze(0)[indices],
                coords[atms[2]].squeeze(0)[indices],
                coords[atms[2]].squeeze(0)[indices] +
                (coords[atms[2]].squeeze(0)[indices] - coords[atms[1]].squeeze(0)[indices])
            )

            rot_cur_to_default = torch.einsum('...ji,...jk->...ik', rot_default, rot_cur)
            rots[indices, i + 3] = rot_cur_to_default

            #         print(np.round(rot_cur_to_prev,3))
            cos = rot_cur_to_default[:, 1, 1]
            sin = rot_cur_to_default[:, 2, 1]

            angls[indices, i + 3, 0] = sin
            angls[indices, i + 3, 1] = cos
    return rots, angls


def torsion_angles_to_frames(
        r: Rigid,
        alpha: torch.Tensor,
        aatype: torch.Tensor,
        rrgdf: torch.Tensor,
):
    t0 = time()
    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)
    t1 = time()
    # print(f'default from 4x4:{t1-t0:.4f}')
    # [*, N, 8, 2]
    if alpha.size(-2) == 1:  # psi only
        bb_alpha = torch.zeros((*alpha.size()[:-2], 8, 2), device=alpha.device)
        bb_alpha[..., 1] = 1  # cos
        bb_alpha[..., 3:4, :] = alpha
        alpha = bb_alpha
    if alpha.size(-2) == 5:
        bb_alpha = torch.zeros((*alpha.size()[:-2], 3, 2), device=alpha.device)
        bb_alpha[..., 1] = 1  # cos
        alpha = torch.cat([bb_alpha, alpha], dim=-2)

    if alpha.size(-2) == 7:
        bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
        bb_rot[..., 1] = 1  # zero angle

        alpha = torch.cat(
            [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
        )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    t2 = time()
    # print(f'default_r.get_rots().get_rot_mats():{t2-t1:.4f}')

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    t3 = time()
    # print(f'Rigid(Rotation(rot_mats=all_rots), None):{t3-t2:.4f}')

    all_frames = default_r.compose(all_rots)
    # all_frames = all_rots.compose(default_r)

    chi2_frame_to_frame = all_frames[..., 5]  # chi2->chi1
    chi3_frame_to_frame = all_frames[..., 6]  # chi3->chi2
    chi4_frame_to_frame = all_frames[..., 7]  # chi4->chi3

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    t4 = time()
    # print(f'compose:{t4 - t3:.4f}')

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
        r: Rigid,
        aatype: torch.Tensor,
        default_frames,
        group_idx,
        atom_mask,
        lit_positions,
):
    # [*, N, 14, 4, 4]
    default_4x4 = default_frames[aatype, ...]

    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.nansum(x, dim=-1)
    )

    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions, atom_mask


class ProtConverter:
    def __init__(
            self,
            trans_scale_factor=10
    ):

        super(ProtConverter, self).__init__()

        # To be lazily initialized later
        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None
        self.trans_scale_factor = trans_scale_factor

    def build_cords(
            self,
            seq,
            scaled_rigids,
            angles,
    ):
        """
        Args:
            angles:
                [*, N, 7, 2]
            aatype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        aatype = torch.Tensor([restype_order[res] for res in seq]).long().to(angles.device)

        t0 = time()
        all_frames_to_global = self.torsion_angles_to_frames(
            scaled_rigids,
            angles,
            aatype,
        )  # torsion angle $ default frame -> frame to global

        t1 = time()
        # print(f'ang to frame: {t1-t0:.2f}')
        pred_xyz, atom_mask = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            aatype,
        )  # the xyz of specific atoms for each residue (*,L,14,3)
        t2 = time()
        # print(f'frame to atom14: {t2-t1:.2f}')
        return pred_xyz, atom_mask, all_frames_to_global

    def _init_residue_constants(self, float_dtype, device):
        if self.default_frames is None:
            self.default_frames = torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None:
            self.group_idx = torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None:
            self.atom_mask = torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None:
            self.lit_positions = torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        # t0 = time()
        self._init_residue_constants(alpha.dtype, alpha.device)
        # t1 = time()
        # print(f'init:{t1-t0:.3f}')
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
            self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

    @staticmethod
    def export_pdb_file(seq, atom_cords, path=None, atom_names=restype_name_to_atom14_names,
                        n_key_atoms=14, atom_masks=None, lens=None, ca_only=False, bb_only=False,
                        idx_resd_pdb=None, confidence=None, chain_id=None, logger=None):
        """Export a PDB file."""

        # configurations
        i_code = ' '
        # chain_id = '0' if chain_id is None else chain_id
        occupancy = 1.0
        cord_min = -999.0
        cord_max = 999.0
        seq_len = len(seq)
        if lens is None: lens = [seq_len]

        # take all the atom coordinates as valid, if not specified
        if atom_masks is None:
            atom_masks = np.ones(atom_cords.shape[:-1], dtype=np.int8)

        # determine the set of atom names (per residue)
        if atom_cords.ndim == 2:
            if atom_cords.shape[0] == seq_len * n_key_atoms:
                atom_cords = np.reshape(atom_cords, [seq_len, n_key_atoms, 3])
                atom_masks = np.reshape(atom_masks, [seq_len, n_key_atoms])
            else:
                raise ValueError('atom coordinates\' shape does not match the sequence length')

        elif atom_cords.ndim == 3:
            assert atom_cords.shape[0] == seq_len
            atom_cords = atom_cords
            atom_masks = atom_masks

        else:
            raise ValueError('atom coordinates must be a 2D or 3D np.ndarray')

        # reset invalid values in atom coordinates
        atom_cords = np.clip(atom_cords, cord_min, cord_max)
        atom_cords[np.isnan(atom_cords)] = 0.0
        atom_cords[np.isinf(atom_cords)] = 0.0

        lines = []
        n_atoms = 0
        for ich, Lch in enumerate(lens):
            start = 0 if ich == 0 else sum(lens[:ich])
            end = sum(lens[:ich + 1])
            for idx_resd in range(start, end):
                resd_name = restype_1to3[seq[idx_resd]]
                # for idx_resd, resd_name in enumerate(seq):
                if ca_only:
                    atom_names[resd_name] = ['CA']
                if resd_name not in atom_names: continue
                if bb_only:
                    atom_names[resd_name] = atom_names[resd_name][:4]
                for idx_atom, atom_name in enumerate(atom_names[resd_name]):
                    if len(atom_name) > 0:
                        temp_factor = 0.0 if confidence is None else \
                            float(100 * confidence.reshape([seq_len])[idx_resd - 1])

                        if atom_masks[idx_resd, idx_atom] == 0:
                            continue
                        n_atoms += 1
                        charge = atom_name[0]
                        if idx_resd_pdb is not None:
                            ires = idx_resd_pdb[idx_resd]
                        else:
                            ires = int(idx_resd - start)
                        chainID = ich if chain_id is None else chain_id if isinstance(chain_id, str) else chain_id[ich]
                        line_str = ''.join([
                            'ATOM  ',
                            '%5d' % n_atoms,
                            '  ' + atom_name + ' ' * (3 - len(atom_name)),
                            ' %s' % resd_name,
                            ' %s' % str(chainID)[0],
                            ' ' * (4 - len(str(ires + 1))),
                            '%s' % str(ires + 1),
                            '%s   ' % i_code,
                            '%8.3f' % atom_cords[idx_resd, idx_atom, 0],
                            '%8.3f' % atom_cords[idx_resd, idx_atom, 1],
                            '%8.3f' % atom_cords[idx_resd, idx_atom, 2],
                            '%6.2f' % occupancy,
                            '%6.2f' % temp_factor,
                            ' ' * 10,
                            '%2s' % charge,
                            '%2s' % ' ',
                        ])
                        assert len(line_str) == 80, 'line length must be exactly 80 characters: ' + line_str
                        lines.append(line_str + '\n')
        if path is not None:
            # export the 3D structure to a PDB file
            os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
            with open(path, 'w') as o_file:
                o_file.write(''.join(lines))
        if logger is not None:
            logger.info(f'    Export PDB file to {path}')
        return lines


if __name__ == '__main__':
    from Bio import PDB
    from e2e.models.protein_constants import restype_3to1


    def get_atom_positions_pdb(pdb_file, atom_types, retain_all_res=False):
        pp = PDB.PDBParser(QUIET=True)
        structure = pp.get_structure('', pdb_file)[0]
        xyzs_all = []
        for chain in structure.child_list:
            residues = [res for res in chain.child_list if not res.id[0].strip()]
            if retain_all_res:
                L = residues[-1].id[1]
            else:
                L = len(residues)
            seq = ['X'] * L

            xyzs_all = {atm: np.nan * np.zeros((L, 3)) for atm in atom_types}

            for i, res in enumerate(residues):
                for atom in atom_types:
                    try:
                        atom_name = atom
                        coord = res[atom_name].coord
                        if retain_all_res:
                            xyzs_all[atom][res.id[1] - 1] = coord
                            seq[res.id[1] - 1] = restype_3to1[res.resname.strip()]
                        else:
                            xyzs_all[atom][i] = coord
                            seq[i] = restype_3to1[res.resname.strip()]
                        continue
                    except KeyError:
                        continue
        return xyzs_all, ''.join(seq)


    def _make_rigid_transformation_4x4(ex, ey, translation):
        """Create a rigid 4x4 transformation matrix from two axes and transl."""
        if isinstance(ex, np.ndarray):
            ex = torch.from_numpy(ex)
        if isinstance(ey, np.ndarray):
            ey = torch.from_numpy(ey)
        if isinstance(translation, np.ndarray):
            translation = torch.from_numpy(translation)

        # Normalize ex.
        ex_normalized = ex / ex.norm(dim=-1, keepdim=True)

        # make ey perpendicular to ex
        ey_normalized = ey - torch.einsum('...d,...d->...', ey, ex_normalized)[..., None] * ex_normalized
        ey_normalized /= ey_normalized.norm(dim=-1, keepdim=True)

        # compute ez as cross product
        eznorm = torch.cross(ex_normalized, ey_normalized)
        #     return np.stack(
        #         [ex_normalized, ey_normalized, eznorm],axis=-1
        #     ),translation
        m = torch.stack(
            [ex_normalized, ey_normalized, eznorm, translation], dim=-1
        )

        tensor = torch.zeros((*m.shape[:-2], 4, 4), device=ex.device)
        tensor[..., :3, :] = m
        tensor[..., 3, :3] = 0
        tensor[..., 3, 3] = 1
        return tensor


    def _make_rigid_transformation_rt(ex, ey, translation):
        """Create a rigid 4x4 transformation matrix from two axes and transl."""
        # Normalize ex.
        ex_normalized = ex / np.linalg.norm(ex, axis=-1, keepdims=True)

        # make ey perpendicular to ex
        ey_normalized = ey - np.einsum('...d,...d->...', ey, ex_normalized)[..., None] * ex_normalized
        ey_normalized /= np.linalg.norm(ey_normalized, axis=-1, keepdims=True)

        # compute ez as cross product
        eznorm = np.cross(ex_normalized, ey_normalized)
        return np.stack(
            [ex_normalized, ey_normalized, eznorm], axis=-1
        ), translation


    #     m = np.stack(
    #         [ex_normalized, ey_normalized, eznorm, translation]
    #     ).transpose()
    #     m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    #     return m

    ATOM_INFOS_PER_RESD = rigid_group_atom_positions

    TRANS_DICT_PER_RESD = restype_rigid_group_default_frame

    coords, seq = get_atom_positions_pdb('/home/wangwenkai/db/casp14/pdb/T1091-D1.pdb', atom_types,
                                         retain_all_res=False)

    from RNA.models.utils_rhofold.rigid_utils import Rigid

    # resname = 'PHE'
    # indices = np.array(list(seq)) == restype_3to1[resname]
    # atom_positions = {atm:coords[atm][indices] for atm in coords}
    atom_positions = {atm: torch.from_numpy(coords[atm]).float() for atm in coords}

    # bb4x4 = _make_rigid_transformation_4x4(ex=atom_positions["C"] - atom_positions["CA"],
    #                                        ey=atom_positions["N"] - atom_positions["CA"],
    #                                        translation=atom_positions["CA"])
    bb_rot, bb_tsl = calc_rot_tsl(atom_positions["N"], atom_positions["CA"], atom_positions["C"])

    # bb_rigids = Rigid.from_tensor_4x4(bb4x4)
    bb_rigids = Rigid(rots=Rotation(rot_mats=bb_rot), trans=bb_tsl)

    rots, angls = calc_angls_prot(atom_positions, seq)
    # angls = torch.from_numpy(angls)
    prot_converter = ProtConverter()
    xyz = prot_converter.build_cords(seq, bb_rigids, angls[..., -5:, :])[0]
    prot_converter.export_pdb_file(seq, xyz, '/home/wangwenkai/RPI/see/prot_fullatm.pdb')
