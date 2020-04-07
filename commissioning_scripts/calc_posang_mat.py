"""."""

import numpy as np
from pymodels import tb, ts
import pyaccel

KICK = 1e-5


def get_tb_posang_respm(model=None):
    """Return TB position and angle correction matrix."""
    if model is None:
        model, _ = tb.create_accelerator()
    fam_data = tb.families.get_family_data(model)

    ch_idx = [fam_data['CH']['index'][-1], fam_data['InjSept']['index'][0]]
    cv_idx = [fam_data['CV']['index'][-2], fam_data['CV']['index'][-1]]

    return _calc_posang_matrices(model, ch_idx, cv_idx)


def get_ts_posang_respm(corrs_type='CH-Sept', model=None):
    """Return TS position and angle correction matrix."""
    if model is None:
        model, _ = ts.create_accelerator()
    fam_data = ts.families.get_family_data(model)

    if corrs_type == 'CH-Sept':
        ch_idx = [fam_data['CH']['index'][-1],
                  fam_data['InjSeptF']['index'][0]]
    else:
        ch_idx = [fam_data['InjSeptG']['index'][0],
                  fam_data['InjSeptG']['index'][1],
                  fam_data['InjSeptF']['index'][0]]
    cv_idx = [fam_data['CV']['index'][-2], fam_data['CV']['index'][-1]]

    return _calc_posang_matrices(model, ch_idx, cv_idx)


def _calc_posang_matrices(model, ch_idx, cv_idx):
    ini = ch_idx[0][0]
    mat_h_aux = np.zeros((2, len(ch_idx)))
    for idx in range(len(ch_idx)):
        pyaccel.lattice.set_attribute(
            model, 'hkick_polynom', ch_idx[idx], KICK/2/len(ch_idx[idx]))
        pos1, *_ = pyaccel.tracking.line_pass(model[ini:len(model)], 6*[0])
        pyaccel.lattice.set_attribute(
            model, 'hkick_polynom', ch_idx[idx], -KICK/2/len(ch_idx[idx]))
        pos2, *_ = pyaccel.tracking.line_pass(model[ini:len(model)], 6*[0])
        pyaccel.lattice.set_attribute(
            model, 'hkick_polynom', ch_idx[idx], 0)
        mat_h_aux[:, idx] = (pos1[0:2] - pos2[0:2])/KICK

    if len(ch_idx) == 3:
        mat_h = [[mat_h_aux[0, 0]+mat_h_aux[0, 1], mat_h_aux[0, 2]],
                 [mat_h_aux[1, 0]+mat_h_aux[1, 1], mat_h_aux[1, 2]]]
    else:
        mat_h = mat_h_aux

    ini = cv_idx[0][0]
    mat_v = np.zeros((2, 2))
    for idx in range(len(cv_idx)):
        pyaccel.lattice.set_attribute(
            model, 'vkick_polynom', cv_idx[idx], KICK/2/len(cv_idx[idx]))
        pos1, *_ = pyaccel.tracking.line_pass(model[ini:len(model)], 6*[0])
        pyaccel.lattice.set_attribute(
            model, 'vkick_polynom', cv_idx[idx], -KICK/2/len(cv_idx[idx]))
        pos2, *_ = pyaccel.tracking.line_pass(model[ini:len(model)], 6*[0])
        pyaccel.lattice.set_attribute(
            model, 'vkick_polynom', cv_idx[idx], 0)
        mat_v[:, idx] = (pos1[2:4] - pos2[2:4])/KICK

    return mat_h, mat_v
