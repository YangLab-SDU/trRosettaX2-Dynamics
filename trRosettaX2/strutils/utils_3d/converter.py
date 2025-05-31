# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
from collections import defaultdict
from a7d2.e2e.models.utils_rhofold.constants import RNA_CONSTANTS, ANGL_INFOS_PER_RESD, TRANS_DICT_PER_RESD
from a7d2.e2e.models.utils_rhofold.rigid_utils import Rigid, calc_rot_tsl, calc_angl_rot_tsl, merge_rot_tsl


def calc_angls(coords, seq, angl_infos=ANGL_INFOS_PER_RESD):
    L = len(coords["P"].squeeze(0))

    angls = {f'angl_{i}': torch.nan * torch.ones((L, 2), device=coords["P"].device) for i in range(4)}
    for res, angl_info in angl_infos.items():
        indices = [ind for ind, aa in enumerate(seq) if aa == res]
        for angl, _, atms in angl_info:
            atm1_prev, atm2_prev, atm3_prev = atms[:3]
            atm1, atm2, atm3 = atms[-1], atms[1], atms[2]
            rot_prev, tsl_prev = calc_rot_tsl(coords[atm1_prev].squeeze(0).float()[indices],
                                              coords[atm3_prev].squeeze(0).float()[indices],
                                              coords[atm2_prev].squeeze(0).float()[indices])
            rot_cur, tsl_cur = calc_rot_tsl(coords[atm1].squeeze(0).float()[indices],
                                            coords[atm3].squeeze(0).float()[indices],
                                            coords[atm2].squeeze(0).float()[indices])

            rot_cur_to_prev = torch.einsum('...ji,...jk->...ik', rot_prev, rot_cur)
            #         print(np.round(rot_cur_to_prev,3))
            cos = rot_cur_to_prev[:, 1, 1]
            sin = rot_cur_to_prev[:, 2, 1]
            norm = (cos ** 2 + sin ** 2) ** .5
            if (torch.abs(norm - 1) > 0.1).any():
                raise ValueError(f'cos^2 + sin^2!=1')
            cos /= norm
            sin /= norm
            angls[angl][indices, 0] = cos
            angls[angl][indices, 1] = sin
    return angls


def calc_angls_from_base(coords, seq, angl_infos=ANGL_INFOS_PER_RESD, trans_dict=TRANS_DICT_PER_RESD,
                         ignore_norm_err=False):
    L = len(coords["P"].squeeze(0))
    device = coords["P"].device
    angls = {f'angl_{i}': torch.nan * torch.ones((L, 2), device=device) for i in range(4)}
    for res, angl_info in angl_infos.items():
        indices = [ind for ind, aa in enumerate(seq) if aa == res]
        fram_dict = {}
        atm1, atm2, atm3 = ["C4'", "N9", "C1'"] if res in ['A', 'G'] else ["C4'", "N1", "C1'"]
        fram_dict['main'] = calc_rot_tsl(coords[atm1].squeeze(0).float()[indices],
                                         coords[atm3].squeeze(0).float()[indices],
                                         coords[atm2].squeeze(0).float()[indices])
        for angl, _, atms in angl_info:
            if angl in ['omega', 'phi', 'angl_0', 'angl_1']:
                angl_prev = 'main'
            else:
                angl_prev = 'angl_%d' % (int(angl[-1]) - 1)
            atm1, atm2, atm3 = atms[-1], atms[1], atms[2]
            rot_curr, tsl_curr = calc_rot_tsl(coords[atm1].squeeze(0).float()[indices],
                                              coords[atm3].squeeze(0).float()[indices],
                                              coords[atm2].squeeze(0).float()[indices])
            fram_dict[angl] = (rot_curr, tsl_curr)

            rot_prev, tsl_prev = fram_dict[angl_prev]
            rot_base, tsl_vec_base = trans_dict[res]['%s-%s' % (angl, angl_prev)]
            rot_base = torch.from_numpy(rot_base).float().to(device)
            tsl_base = torch.from_numpy(tsl_vec_base).float().to(device)

            rot_base, tsl_base = merge_rot_tsl(rot_prev, tsl_prev, rot_base, tsl_base)
            rot_addi = torch.einsum('...ji,...jk->...ik', rot_base, rot_curr)

            cos = rot_addi[:, 1, 1]
            sin = rot_addi[:, 2, 1]
            norm = (cos ** 2 + sin ** 2) ** .5
            if not ignore_norm_err and (torch.abs(norm - 1) > 0.1).any():
                raise ValueError(f'cos^2 + sin^2!=1')
            cos /= norm
            sin /= norm
            angls[angl][indices, 0] = cos
            angls[angl][indices, 1] = sin
    return angls
