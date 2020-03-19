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

    ch = [fam_data['CH']['index'][-1], fam_data['InjSept']['index'][0]]
    cv = [fam_data['CV']['index'][-2], fam_data['CV']['index'][-1]]

    return _calc_posang_matrices(model, ch, cv)


def get_ts_posang_respm(corrs_type='CH-Sept', model=None):
    """Return TS position and angle correction matrix."""
    if model is None:
        model, _ = ts.create_accelerator()
    fam_data = ts.families.get_family_data(model)

    if corrs_type == 'CH-Sept':
        ch = [fam_data['CH']['index'][-1],
              fam_data['InjSeptF']['index'][0]]
    else:
        ch = [fam_data['InjSeptG']['index'][0],
              fam_data['InjSeptG']['index'][1],
              fam_data['InjSeptF']['index'][0]]
    cv = [fam_data['CV']['index'][-2], fam_data['CV']['index'][-1]]

    return _calc_posang_matrices(model, ch, cv)


def _calc_posang_matrices(model, ch, cv):
    ini = ch[0][0]
    mat_h_aux = np.zeros((2, len(ch)))
    for i in range(len(ch)):
        pyaccel.lattice.set_attribute(
            model, 'hkick_polynom', ch[i], KICK/2/len(ch[i]))
        p1 = pyaccel.tracking.line_pass(model[ini:len(model)], 6*[0])
        pyaccel.lattice.set_attribute(
            model, 'hkick_polynom', ch[i], -KICK/2/len(ch[i]))
        p2 = pyaccel.tracking.line_pass(model[ini:len(model)], 6*[0])
        pyaccel.lattice.set_attribute(
            model, 'hkick_polynom', ch[i], 0)
        mat_h_aux[:, i] = (p1[0][0:2] - p2[0][0:2])/KICK

    if len(ch) == 3:
        mat_h = [[mat_h_aux[0, 0]+mat_h_aux[0, 1], mat_h_aux[0, 2]],
                 [mat_h_aux[1, 0]+mat_h_aux[1, 1], mat_h_aux[1, 2]]]
    else:
        mat_h = mat_h_aux

    ini = cv[0][0]
    mat_v = np.zeros((2, 2))
    for i in range(len(cv)):
        pyaccel.lattice.set_attribute(
            model, 'vkick_polynom', cv[i], KICK/2/len(cv[i]))
        p1 = pyaccel.tracking.line_pass(model[ini:len(model)], 6*[0])
        pyaccel.lattice.set_attribute(
            model, 'vkick_polynom', cv[i], -KICK/2/len(cv[i]))
        p2 = pyaccel.tracking.line_pass(model[ini:len(model)], 6*[0])
        pyaccel.lattice.set_attribute(
            model, 'vkick_polynom', cv[i], 0)
        mat_v[:, i] = (p1[0][2:4] - p2[0][2:4])/KICK

    return mat_h, mat_v
