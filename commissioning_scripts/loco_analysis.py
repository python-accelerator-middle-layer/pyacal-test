"""."""

from copy import deepcopy as _dcopy
import numpy as np
import pyaccel

from pymodels import si
from apsuite.commissioning_scripts.loco import LOCOUtils, LOCOConfigSI, LOCO


def get_famidx(config):
    """."""
    famidx = []
    for fam_name in config.famname_quadset:
        famidx.append(config.respm.fam_data[fam_name]['index'])
    return famidx


def errors_generate_gauss(avg, std, cut, size):
    """."""
    sel = np.array(size*[True])
    delta = np.zeros(size)
    while sel.any():
        delta[sel] = np.random.normal(avg, std, np.sum(sel))
        sel = np.abs(delta - avg) > cut*std
    return delta


def errors_add_quads_family(config, avg, std, cut, quad_fams=None):
    """."""
    # initializations
    config.update()
    if quad_fams is None:
        quad_fams = config.famname_quadset

    # generate errors
    # avg, std, cut = 0.0, 0.1/100, 3.0
    delta = errors_generate_gauss(avg, std, cut, len(quad_fams))

    # apply errors
    famidx = get_famidx(config)
    idx = []
    for sub in quad_fams:
        idx.append(quad_fams.index(sub))
    kquads_nom = LOCOUtils.get_quads_strengths(config.model, famidx)
    for i, ind in enumerate(idx):
        kdelta = [k[0]*delta[i] for k in kquads_nom[ind]]
        LOCOUtils.set_quadset_kdelta(
            config.model, famidx[ind], kquads_nom[ind], kdelta[0])

    return delta


def errors_add_quads_individual(config, avg, std, cut):
    """."""
    # initializations
    config.update()

    tunex = np.nan
    tuney = np.nan
    while np.isnan(tunex) or np.isnan(tuney):
        qidx = config.respm.fam_data['QN']['index']
        # generate errors
        delta = errors_generate_gauss(avg, std, cut, len(qidx))
        # avg = 0 # std = 0.05/100 # cutoff = 3

        # apply errors
        kquads_nom = LOCOUtils.get_quads_strengths(config.model, qidx)
        kdelta = np.zeros(len(qidx))
        for i, ind in enumerate(qidx):
            kdelta[i] = kquads_nom[i]*delta[i]
            LOCOUtils.set_quadmag_kdelta(
                config.model, ind, [kquads_nom[i]], kdelta[i])
        twi, *_ = pyaccel.optics.calc_twiss(config.model, indices='open')
        tunex = twi.mux[-1]/2/np.pi
        tuney = twi.muy[-1]/2/np.pi
        print('Tune x {0:0.6f}, Tune y {1:0.6f}'.format(
            tunex, tuney))

        return delta


def add_errors_gains(config, gtype, avg, std, cut):
    """."""
    # initializations
    config.update()

    if gtype == 'bpm':
        # generate errors
        size = 2*config.nr_bpm
        gains = errors_generate_gauss(avg, std, cut, size)
        # apply errors
        for idx in range(size):
            config.matrix[idx, :] *= gains[idx]
    elif gtype == 'corr':
        # generate errors
        size = config.nr_corr
        gains = errors_generate_gauss(avg, std, cut, size)
        # apply errors
        for idx in range(size):
            config.matrix[:, idx] *= gains[idx]

    return gains


def analysis_orbrespm_add_errors(config):
    """."""
    config = _dcopy(config)

    errors = dict()

    # quadrupole errors
    avg, cut = 0.0, 3.0
    if config.use_families:
        std = 0.1/100
        quad_fams = config.famname_quadset
        delta = errors_add_quads_family(
            config, avg, std, cut, quad_fams=quad_fams)
        errors['quadrupole_family'] = delta
    else:
        std = 0.05/100
        delta = errors_add_quads_individual(config, avg, std, cut)
        errors['quadrupole_individual'] = delta

    # bpm gain errors
    avg, std, cut = 1.0, 10.0/100, 3.0
    gains = add_errors_gains(config, 'bpm', avg, std, cut)
    errors['gain_bpm'] = gains

    # corr gain errors
    avg, std, cut = 1.0, 10.0/100, 3.0
    gains = add_errors_gains(config, 'corr', avg, std, cut)
    errors['gain_corr'] = gains

    # calc orbrespm
    config.update()
    orbrespm = config.matrix

    return orbrespm, errors


def analysis_create_config():
    """."""
    config = LOCOConfigSI()

    config.model = si.create_accelerator()
    config.dim = '6d'

    config.use_disp = True
    config.use_families = True
    config.use_coupling = False
    config.rf_freq = 499662430.0

    config.gain_bpm = None
    config.gain_corr = None
    config.roll_bpm = None
    config.roll_corr = None

    config.fit_quadrupoles = True
    config.fit_gain_bpm = True
    config.fit_roll_bpm = True
    config.fit_gain_corr = True

    config.svd_method = LOCOConfigSI.SVD_METHOD_THRESHOLD
    config.svd_thre = 1e-6

    return config


def analysis_run():
    """."""
    print('[create loco configuration]')
    config = analysis_create_config()
    print('')

    print('[calculate orbrespm for a perturbed model]')
    orbrespm, errors = analysis_orbrespm_add_errors(config)
    config.goalmat = orbrespm
    config.update()
    print('')

    print('[create loco object]')
    loco = LOCO(config)
    loco.update(fname_jloco_k=None)
    print('')

    return loco

    print('[run fit]')
    loco.run_fit(niter=1)
    print('')


analysis_run()
