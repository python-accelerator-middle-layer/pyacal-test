#!/usr/bin/env python-sirius
"""."""

from copy import deepcopy as _dcopy
import pickle as _pickle
import numpy as np
import pyaccel
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat


class LOCOUtils:
    """LOCO utils."""

    @staticmethod
    def save_data(fname, jloco):
        """."""
        data = dict(jloco_kmatrix=jloco)
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'wb') as fil:
            _pickle.dump(data, fil)

    @staticmethod
    def load_data(fname):
        """."""
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'rb') as fil:
            data = _pickle.load(fil)
        return data

    @staticmethod
    def respm_calc(model, respm, use_disp):
        """."""
        respm.model = _dcopy(model)
        matrix = respm.get_respm()
        if not use_disp:
            matrix[:, -1] *= 0
        return matrix

    @staticmethod
    def apply_bpm_gain(matrix, gain):
        """."""
        return gain[:, None] * matrix

    @staticmethod
    def apply_bpm_roll(matrix, roll):
        """."""
        cos_mat = np.diag(np.cos(roll))
        sin_mat = np.diag(np.sin(roll))
        r_alpha = np.hstack((cos_mat, sin_mat))
        r_alpha = np.vstack((r_alpha, np.hstack((-sin_mat, cos_mat))))
        return np.dot(r_alpha, matrix)

    @staticmethod
    def apply_corr_gain(matrix, gain):
        """."""
        matrix[:, :-1] = matrix[:, :-1] * gain[None, :]
        return matrix

    @staticmethod
    def apply_all_gain(matrix, gain_bpm, roll_bpm, gain_corr):
        """."""
        matrix = LOCOUtils.apply_bpm_gain(matrix, gain_bpm)
        matrix = LOCOUtils.apply_bpm_roll(matrix, roll_bpm)
        matrix = LOCOUtils.apply_corr_gain(matrix, gain_corr)
        return matrix

    @staticmethod
    def apply_bpm_weight(matrix, weight_bpm):
        """."""
        return weight_bpm[:, None] * matrix

    @staticmethod
    def apply_corr_weight(matrix, weight_corr):
        """."""
        return matrix * weight_corr[None, :]

    @staticmethod
    def apply_all_weight(matrix, weight_bpm, weight_corr):
        """."""
        matrix = LOCOUtils.apply_bpm_weight(matrix, weight_bpm)
        matrix = LOCOUtils.apply_corr_weight(matrix, weight_corr)
        return matrix

    @staticmethod
    def remove_coupling(matrix_in, nr_bpm, nr_ch, nr_cv):
        """."""
        matrix_out = np.zeros(matrix_in.shape)
        matrix_out[:nr_bpm, :nr_ch] = matrix_in[:nr_bpm, :nr_ch]
        matrix_out[nr_bpm:, nr_ch:nr_ch+nr_cv] = \
            matrix_in[nr_bpm:, nr_ch:nr_ch+nr_cv]
        matrix_out[:nr_bpm, -1] = matrix_in[:nr_bpm, -1]
        return matrix_out

    @staticmethod
    def get_quads_strengths(model, indices):
        """."""
        kquads = []
        for qidx in indices:
            kquads.append(pyaccel.lattice.get_attribute(
                model, 'K', qidx))
        return kquads

    @staticmethod
    def set_quadmag_kdelta(model, idx_mag, kvalues, kdelta):
        """."""
        pyaccel.lattice.set_attribute(
            model, 'K', idx_mag, kvalues + kdelta)

    @staticmethod
    def set_dipmag_kdelta(model, idx_mag, kvalues, kdelta):
        """."""
        ktotal = np.sum(kvalues)
        if ktotal:
            pyaccel.lattice.set_attribute(
                model, 'K', idx_mag, kvalues*(1+kdelta/ktotal))
        else:
            pyaccel.lattice.set_attribute(
                model, 'K', idx_mag, kvalues + kdelta/len(idx_mag))

    @staticmethod
    def set_quadmag_ksdelta(model, idx_mag, ksvalues, ksdelta):
        """."""
        pyaccel.lattice.set_attribute(
            model, 'Ks', idx_mag, ksvalues + ksdelta)

    @staticmethod
    def set_dipmag_ksdelta(model, idx_mag, ksvalues, ksdelta):
        """."""
        kstotal = np.sum(ksvalues)
        if kstotal:
            ksvalues /= kstotal
            pyaccel.lattice.set_attribute(
                model, 'Ks', idx_mag, ksvalues*(1 + ksdelta))
        else:
            pyaccel.lattice.set_attribute(
                model, 'Ks', idx_mag, ksvalues + ksdelta/len(idx_mag))

    @staticmethod
    def set_dipmag_kick(model, idx_mag, kick_values, kick_delta):
        """."""
        angle = np.array(
            pyaccel.lattice.get_attribute(model, 'angle', idx_mag))
        angle /= np.sum(angle)
        pyaccel.lattice.set_attribute(
            model, 'hkick_polynom', idx_mag, kick_values + kick_delta * angle)

    @staticmethod
    def set_quadset_kdelta(model, idx_set, kvalues, kdelta):
        """."""
        for idx, idx_mag in enumerate(idx_set):
            LOCOUtils.set_quadmag_kdelta(
                model, idx_mag, kvalues[idx], kdelta)

    @staticmethod
    def jloco_calc_linear(config, matrix):
        """."""
        nbpm = config.nr_bpm
        nch = config.nr_ch
        ncv = config.nr_cv
        ncorr = nch + ncv
        shape0 = matrix.shape[0]
        shape1 = matrix.shape[1]

        if shape0 != 2*nbpm:
            raise Exception('Problem with BPM number in matrix')
        if shape1 not in (ncorr, ncorr + 1):
            raise Exception('Problem with correctors number in matrix')

        if shape1 < ncorr + 1 and config.use_disp:
            raise Exception('There is no dispersion line in the matrix')

        g_bpm = np.ones(2*nbpm)
        alpha_bpm = np.zeros(nbpm)
        cos_mat = np.diag(np.cos(alpha_bpm))
        sin_mat = np.diag(np.sin(alpha_bpm))

        r_alpha = np.hstack((cos_mat, sin_mat))
        r_alpha = np.vstack((r_alpha, np.hstack((-sin_mat, cos_mat))))

        dr_alpha = np.hstack((-sin_mat, cos_mat))
        dr_alpha = np.vstack((dr_alpha, np.hstack((-cos_mat, sin_mat))))

        dmdg_bpm = np.zeros((shape0*shape1, 2*nbpm))
        for n in range(shape0):
            kron = LOCOUtils.kronecker(n, n, shape0)
            dB = np.dot(r_alpha, kron)
            dmdg_bpm[:, n] = np.dot(dB, matrix).flatten()

        dmdalpha_bpm = np.zeros((shape0*shape1, nbpm))
        for idx in range(shape0//2):
            kron = LOCOUtils.kronecker(idx, idx, shape0//2)
            kron = np.tile(kron, (2, 2))
            deltaR = np.dot(kron, dr_alpha)
            deltaB = deltaR * g_bpm[:, None]
            dmdalpha_bpm[:, idx] = np.dot(deltaB, matrix).flatten()

        dmdg_corr = np.zeros((shape0*shape1, ncorr))
        for idx in range(ncorr):
            kron = LOCOUtils.kronecker(idx, idx, shape1)
            dmdg_corr[:, idx] = np.dot(matrix, kron).flatten()

        return dmdg_bpm, dmdalpha_bpm, dmdg_corr

    @staticmethod
    def jloco_calc_k_quad(config, model):
        """."""
        if config.use_families:
            kindices = []
            for fam_name in config.famname_quadset:
                kindices.append(config.respm.fam_data[fam_name]['index'])
            kvalues = LOCOUtils.get_quads_strengths(
                model, kindices)
            set_quad_kdelta = LOCOUtils.set_quadset_kdelta
        else:
            kindices = config.respm.fam_data['QN']['index']
            kvalues = np.array(
                pyaccel.lattice.get_attribute(model, 'K', kindices))
            set_quad_kdelta = LOCOUtils.set_quadmag_kdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_disp)

        kmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(kindices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(kindices):
            set_quad_kdelta(
                model_this, idx_set, kvalues[idx], config.DEFAULT_DELTA_K)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_disp)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_K
            kmatrix[:, idx] = dmatrix.flatten()
            set_quad_kdelta(model_this, idx_set, kvalues[idx], 0)
        return kmatrix

    @staticmethod
    def jloco_calc_k_dipoles(config, model, dipole_name):
        """."""
        dip_indices = config.respm.fam_data[dipole_name]['index']
        dip_kvalues = np.array(
            pyaccel.lattice.get_attribute(model, 'K', dip_indices))
        set_quad_kdelta = LOCOUtils.set_dipmag_kdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_disp)

        dip_kmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(dip_indices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(dip_indices):
            set_quad_kdelta(
                model_this, idx_set, dip_kvalues[idx], config.DEFAULT_DELTA_K)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_disp)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_K
            dip_kmatrix[:, idx] = dmatrix.flatten()
            set_quad_kdelta(model_this, idx_set, dip_kvalues[idx], 0)
        return dip_kmatrix

    @staticmethod
    def jloco_calc_k_sextupoles(config, model):
        """."""
        sn_indices = config.respm.fam_data['SN']['index']
        sn_kvalues = np.array(
            pyaccel.lattice.get_attribute(model, 'K', sn_indices))
        set_quad_kdelta = LOCOUtils.set_quadmag_kdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_disp)

        sn_kmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(sn_indices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(sn_indices):
            set_quad_kdelta(
                model_this, idx_set, sn_kvalues[idx], config.DEFAULT_DELTA_K)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_disp)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_K
            sn_kmatrix[:, idx] = dmatrix.flatten()
            set_quad_kdelta(model_this, idx_set, sn_kvalues[idx], 0)
        return sn_kmatrix

    @staticmethod
    def jloco_calc_ks_quad(config, model):
        """."""
        kindices = config.respm.fam_data['QN']['index']
        ksvalues = np.array(
            pyaccel.lattice.get_attribute(model, 'Ks', kindices))
        set_quad_ksdelta = LOCOUtils.set_quadmag_ksdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_disp)

        ksmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(kindices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(kindices):
            set_quad_ksdelta(
                model_this, idx_set, ksvalues[idx], config.DEFAULT_DELTA_KS)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_disp)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_KS
            ksmatrix[:, idx] = dmatrix.flatten()
            set_quad_ksdelta(model_this, idx_set, ksvalues[idx], 0)
        return ksmatrix

    @staticmethod
    def jloco_calc_ks_dipoles(config, model, dipole_name):
        """."""
        dip_indices = config.respm.fam_data[dipole_name]['index']
        dip_ksvalues = np.array(
            pyaccel.lattice.get_attribute(model, 'Ks', dip_indices))
        set_quad_ksdelta = LOCOUtils.set_dipmag_ksdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_disp)

        dip_ksmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(dip_indices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(dip_indices):
            set_quad_ksdelta(
                model_this, idx_set, dip_ksvalues[idx],
                config.DEFAULT_DELTA_KS)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_disp)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_KS
            dip_ksmatrix[:, idx] = dmatrix.flatten()
            set_quad_ksdelta(model_this, idx_set, dip_ksvalues[idx], 0)
        return dip_ksmatrix

    @staticmethod
    def jloco_calc_ks_sextupoles(config, model):
        """."""
        sn_indices = config.respm.fam_data['SN']['index']
        sn_ksvalues = np.array(
            pyaccel.lattice.get_attribute(model, 'Ks', sn_indices))
        set_quad_ksdelta = LOCOUtils.set_quadmag_ksdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_disp)

        sn_ksmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(sn_indices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(sn_indices):
            set_quad_ksdelta(
                model_this, idx_set, sn_ksvalues[idx], config.DEFAULT_DELTA_KS)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_disp)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_KS
            sn_ksmatrix[:, idx] = dmatrix.flatten()
            set_quad_ksdelta(model_this, idx_set, sn_ksvalues[idx], 0)
        return sn_ksmatrix

    @staticmethod
    def jloco_calc_kick_dipoles(config, model, dipole_name):
        """."""
        dip_indices = config.respm.fam_data[dipole_name]['index']
        dip_kick_values = np.array(
            pyaccel.lattice.get_attribute(model, 'hkick_polynom', dip_indices))
        set_dip_kick = LOCOUtils.set_dipmag_kick
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_disp)

        dip_kick_matrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], 1))

        delta_kick = config.DEFAULT_DELTA_DIP_KICK

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(dip_indices):
            set_dip_kick(
                model_this, idx_set,
                dip_kick_values[idx], delta_kick)
        matrix_this = LOCOUtils.respm_calc(
            model_this, config.respm, config.use_disp)
        dmatrix = (matrix_this - matrix_nominal) / delta_kick
        dip_kick_matrix[:, 0] = dmatrix.flatten()

        for idx, idx_set in enumerate(dip_indices):
            set_dip_kick(model_this, idx_set, dip_kick_values[idx], 0)
        return dip_kick_matrix

    @staticmethod
    def jloco_merge_linear(
            config, kmatrix, ksmatrix, dmdg_bpm, dmdalpha_bpm, dmdg_corr,
            kick_dip):
        """."""
        nbpm = config.nr_bpm
        nch = config.nr_ch
        ncv = config.nr_cv
        nfam = kmatrix.shape[1]
        jloco = np.zeros((kmatrix.shape[0], 2*nfam + 3*nbpm + nch + ncv + 3))
        jloco[:, :nfam] = kmatrix
        jloco[:, nfam:2*nfam] = ksmatrix
        jloco[:, 2*nfam:2*nfam+2*nbpm] = dmdg_bpm
        jloco[:, 2*nfam+2*nbpm:2*nfam+3*nbpm] = dmdalpha_bpm
        jloco[:, 2*nfam+3*nbpm:2*nfam+3*nbpm+nch+ncv] = dmdg_corr
        jloco[:, 2*nfam+3*nbpm+nch+ncv:] = kick_dip
        return jloco

    @staticmethod
    def jloco_param_delete(config, jloco):
        """."""
        idx = 0
        # NORMAL
        quad_nrsets = len(config.quad_indices)
        if not config.fit_quadrupoles:
            jloco = np.delete(jloco, slice(idx, idx + quad_nrsets), axis=1)
            print('removing quadrupoles...')
        else:
            idx += quad_nrsets

        sext_nrsets = len(config.sext_indices)
        if not config.fit_sextupoles:
            jloco = np.delete(jloco, slice(idx, idx + sext_nrsets), axis=1)
            print('removing sextupoles...')
        else:
            idx += sext_nrsets

        b1_nrsets = len(config.b1_indices)
        if not config.fit_b1:
            jloco = np.delete(jloco, slice(idx, idx + b1_nrsets), axis=1)
            print('removing B1...')
        else:
            idx += b1_nrsets

        b2_nrsets = len(config.b2_indices)
        if not config.fit_b2:
            jloco = np.delete(jloco, slice(idx, idx + b2_nrsets), axis=1)
            print('removing B2...')
        else:
            idx += b2_nrsets

        bc_nrsets = len(config.bc_indices)
        if not config.fit_bc:
            jloco = np.delete(jloco, slice(idx, idx + bc_nrsets), axis=1)
            print('removing BC...')
        else:
            idx += bc_nrsets

        # SKEW
        if not config.fit_quadrupoles_coupling:
            jloco = np.delete(jloco, slice(idx, idx + quad_nrsets), axis=1)
            print('removing quadrupoles coupling (skew)...')
        else:
            idx += quad_nrsets

        sext_nrsets = len(config.sext_indices)
        if not config.fit_sextupoles_coupling:
            jloco = np.delete(jloco, slice(idx, idx + sext_nrsets), axis=1)
            print('removing sextupoles coupling (skew)...')
        else:
            idx += sext_nrsets

        b1_nrsets = len(config.b1_indices)
        if not config.fit_b1_coupling:
            jloco = np.delete(jloco, slice(idx, idx + b1_nrsets), axis=1)
            print('removing B1 coupling (skew)...')
        else:
            idx += b1_nrsets

        b2_nrsets = len(config.b2_indices)
        if not config.fit_b2_coupling:
            jloco = np.delete(jloco, slice(idx, idx + b2_nrsets), axis=1)
            print('removing B2 coupling (skew)...')
        else:
            idx += b2_nrsets

        bc_nrsets = len(config.bc_indices)
        if not config.fit_bc_coupling:
            jloco = np.delete(jloco, slice(idx, idx + bc_nrsets), axis=1)
            print('removing BC coupling (skew)...')
        else:
            idx += bc_nrsets

        if not config.fit_gain_bpm:
            jloco = np.delete(jloco, slice(
                idx, idx + 2*config.nr_bpm), axis=1)
            print('removing bpm gain...')
        else:
            idx += 2*config.nr_bpm
        if not config.use_coupling:
            jloco = np.delete(jloco, slice(
                idx, idx + config.nr_bpm), axis=1)
            print('removing bpm roll...')
        else:
            idx += config.nr_bpm
        if not config.fit_gain_corr:
            jloco = np.delete(jloco, slice(idx, idx + config.nr_corr), axis=1)
            print('removing corrector gain...')
        else:
            idx += config.nr_corr
        if not config.fit_kick_b1:
            jloco = np.delete(jloco, slice(idx, idx + 1), axis=1)
            print('removing kick b1...')
        else:
            idx += 1
        if not config.fit_kick_b2:
            jloco = np.delete(jloco, slice(idx, idx + 1), axis=1)
            print('removing kick b2...')
        else:
            idx += 1
        if not config.fit_kick_bc:
            jloco = np.delete(jloco, slice(idx, idx + 1), axis=1)
            print('removing kick bc...')
        else:
            idx += 1
        return jloco

    @staticmethod
    def jloco_apply_weight(jloco, weight_bpm, weight_corr):
        """."""
        weight = (weight_bpm[:, None] * weight_corr[None, :]).flatten()
        return weight[:, None] * jloco

    @staticmethod
    def param_select(config, param):
        """."""
        idx = 0
        param_dict = dict()
        if config.fit_quadrupoles:
            size = len(config.quad_indices)
            param_dict['quadrupoles'] = param[idx:idx+size]
            idx += size
        if config.fit_sextupoles:
            size = len(config.sext_indices)
            param_dict['sextupoles'] = param[idx:idx+size]
            idx += size
        if config.fit_b1:
            size = len(config.b1_indices)
            param_dict['b1'] = param[idx:idx+size]
            idx += size
        if config.fit_b2:
            size = len(config.b2_indices)
            param_dict['b2'] = param[idx:idx+size]
            idx += size
        if config.fit_bc:
            size = len(config.bc_indices)
            param_dict['bc'] = param[idx:idx+size]
            idx += size
        if config.fit_quadrupoles_coupling:
            size = len(config.quad_indices)
            param_dict['quadrupoles_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_sextupoles_coupling:
            size = len(config.sext_indices)
            param_dict['sextupoles_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_b1_coupling:
            size = len(config.b1_indices)
            param_dict['b1_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_b2_coupling:
            size = len(config.b2_indices)
            param_dict['b2_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_bc_coupling:
            size = len(config.bc_indices)
            param_dict['bc_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_gain_bpm:
            size = 2*config.nr_bpm
            param_dict['gain_bpm'] = param[idx:idx+size]
            idx += size
        if config.use_coupling:
            size = config.nr_bpm
            param_dict['roll_bpm'] = param[idx:idx+size]
            idx += size
        if config.fit_gain_corr:
            size = config.nr_corr
            param_dict['gain_corr'] = param[idx:idx+size]
            idx += size
        if config.fit_kick_b1:
            size = 1
            param_dict['kick_b1'] = param[idx:idx+size]
            idx += size
        if config.fit_kick_b2:
            size = 1
            param_dict['kick_b2'] = param[idx:idx+size]
            idx += size
        if config.fit_kick_bc:
            size = 1
            param_dict['kick_bc'] = param[idx:idx+size]
            idx += size

        return param_dict

    @staticmethod
    def kronecker(i, j, size):
        """."""
        kron = np.zeros((size, size))
        if i == j:
            kron[i, i] = 1
        else:
            kron[i, j] = 1
            kron[j, i] = 1
        return kron


class LOCOConfig:
    """SI LOCO configuration."""

    SVD_METHOD_SELECTION = 0
    SVD_METHOD_THRESHOLD = 1

    DEFAULT_DELTA_K = 1e-6  # [1/m^2]
    DEFAULT_DELTA_KS = 1e-6  # [1/m^2]
    DEFAULT_DELTA_DIP_KICK = 1e-6  # [rad]
    DEFAULT_DELTA_RF = 100  # [Hz]
    DEFAULT_SVD_THRESHOLD = 1e-6

    FAMNAME_RF = 'SRFCav'

    def __init__(self, **kwargs):
        """."""
        self.model = None
        self.dim = None
        self.respm = None
        self.goalmat = None
        self.delta_kickx_meas = None
        self.delta_kicky_meas = None
        self.delta_frequency_meas = None
        self.use_disp = None
        self.use_coupling = None
        self.use_families = None
        self.svd_method = None
        self.svd_sel = None
        self.svd_thre = None
        self.fit_quadrupoles = None
        self.fit_sextupoles = None
        self.fit_b1 = None
        self.fit_b2 = None
        self.fit_bc = None
        self.fit_quadrupoles_coupling = None
        self.fit_sextupoles_coupling = None
        self.fit_b1_coupling = None
        self.fit_b2_coupling = None
        self.fit_bc_coupling = None
        self.fit_gain_bpm = None
        self.fit_gain_corr = None
        self.fit_kick_b1 = None
        self.fit_kick_b2 = None
        self.fit_kick_bc = None
        self.cavidx = None
        self.matrix = None
        self.idx_cav = None
        self.idx_bpm = None
        self.gain_bpm = None
        self.gain_corr = None
        self.roll_bpm = None
        self.roll_corr = None
        self.vector = None
        self.quad_indices = None
        self.sext_indices = None
        self.dip_indices = None
        self.b1_indices = None
        self.b2_indices = None
        self.bc_indices = None
        self.k_nrsets = None
        self.weight_bpm = None
        self.weight_corr = None

        self._process_input(kwargs)

    @property
    def acc(self):
        """."""
        raise NotImplementedError

    @property
    def nr_bpm(self):
        """."""
        raise NotImplementedError

    @property
    def nr_ch(self):
        """."""
        raise NotImplementedError

    @property
    def nr_cv(self):
        """."""
        raise NotImplementedError

    @property
    def nr_corr(self):
        """."""
        return self.nr_ch + self.nr_cv

    @property
    def famname_quadset(self):
        """."""
        raise NotImplementedError

    def update(self):
        """."""
        self.update_model(self.model, self.dim)
        self.update_matrix(self.use_disp)
        self.update_goalmat(self.goalmat, self.use_disp, self.use_coupling)
        self.update_gain()
        self.update_weight()
        self.update_quad_knobs(self.use_families)
        self.update_sext_knobs()
        self.update_dip_knobs()
        self.update_svd(self.svd_method, self.svd_sel, self.svd_thre)

    def update_model(self, model, dim):
        """."""
        self.dim = dim
        self.model = _dcopy(model)
        self.model.cavity_on = dim == '6d'
        self.model.radiation_on = dim == '6d'
        self.respm = OrbRespmat(model=self.model, acc=self.acc, dim=self.dim)
        self._create_indices()

    def update_svd(self, svd_method, svd_sel=None, svd_thre=None):
        """."""
        self.svd_sel = svd_sel
        self.svd_thre = svd_thre
        if svd_method == LOCOConfig.SVD_METHOD_SELECTION:
            # if svd_sel is not None:
            #     print(
            #         'svd_selection: {:d} values will be used.'.format(self.svd_sel))
            # else:
            #     print('svd_selection: all values will be used.')
            pass
        if svd_method == LOCOConfig.SVD_METHOD_THRESHOLD:
            if svd_thre is None:
                self.svd_thre = LOCOConfig.DEFAULT_SVD_THRESHOLD
            # print('svd_threshold: {:f}'.format(self.svd_thre))

    def update_goalmat(self, goalmat, use_disp, use_coupling):
        """."""
        # init goalmat
        if goalmat is None:
            goalmat = _dcopy(self.matrix)

        # coupling
        self.use_coupling = use_coupling
        if not use_coupling:
            self.goalmat = LOCOUtils.remove_coupling(
                goalmat, self.nr_bpm, self.nr_ch, self.nr_cv)
        else:
            self.goalmat = _dcopy(goalmat)

        # dispersion
        self.use_disp = use_disp
        if not self.use_disp:
            self.goalmat[:, -1] *= 0

    def update_matrix(self, use_disp):
        """."""
        self.matrix = LOCOUtils.respm_calc(
            self.model, self.respm, use_disp)

    def update_gain(self,
                    gain_bpm=None, gain_corr=None,
                    roll_bpm=None, roll_corr=None):
        """."""
        # bpm
        if gain_bpm is None:
            if self.gain_bpm is None:
                self.gain_bpm = np.ones(2*self.nr_bpm)
        else:
            if isinstance(gain_bpm, (int, float)):
                self.gain_bpm = np.ones(2*self.nr_bpm) * gain_bpm
            else:
                print('setting initial bpm gain...')
                self.gain_bpm = gain_bpm
        if roll_bpm is None:
            if self.roll_bpm is None:
                self.roll_bpm = np.zeros(self.nr_bpm)
        else:
            if isinstance(roll_bpm, (int, float)):
                self.roll_bpm = np.ones(self.nr_bpm) * roll_bpm
            else:
                print('setting initial bpm roll...')
                self.roll_bpm = roll_bpm
        # corr
        if gain_corr is None:
            if self.gain_corr is None:
                self.gain_corr = np.ones(self.nr_corr)
        else:
            if isinstance(gain_corr, (int, float)):
                self.gain_bpm = np.ones(self.nr_corr) * gain_corr
            else:
                print('setting initial corrector gain...')
                self.gain_corr = gain_corr
        if roll_corr is None:
            if self.roll_corr is None:
                self.roll_corr = np.zeros(self.nr_corr)
        else:
            if isinstance(roll_corr, (int, float)):
                self.roll_corr = np.ones(self.nr_bpm) * roll_corr
            else:
                self.roll_corr = roll_corr

        self.matrix = LOCOUtils.apply_all_gain(
            matrix=self.matrix,
            gain_bpm=self.gain_bpm,
            roll_bpm=self.roll_bpm,
            gain_corr=self.gain_corr)
        self.vector = self.matrix.flatten()

    def update_weight(self):
        """."""
        # bpm
        if self.weight_bpm is None:
            self.weight_bpm = np.ones(2*self.nr_bpm)
        elif isinstance(self.weight_bpm, (int, float)):
            self.weight_bpm = np.ones(2*self.nr_bpm) * \
                self.weight_bpm / 2 / self.nr_bpm
        # corr
        if self.weight_corr is None:
            self.weight_corr = np.ones(self.nr_corr + 1)
        elif isinstance(self.weight_corr, (int, float)):
            self.weight_corr = np.ones(self.nr_corr + 1) * \
                self.weight_corr / (self.nr_corr + 1)

    def update_quad_knobs(self, use_families):
        """."""
        self.use_families = use_families
        if use_families:
            self.quad_indices = [None] * len(self.famname_quadset)
            for idx, fam_name in enumerate(self.famname_quadset):
                self.quad_indices[idx] = self.respm.fam_data[fam_name]['index']
        else:
            self.quad_indices = self.respm.fam_data['QN']['index']

    def update_sext_knobs(self):
        """."""
        self.sext_indices = self.respm.fam_data['SN']['index']

    def update_b1_knobs(self):
        """."""
        self.b1_indices = self.respm.fam_data['B1']['index']

    def update_b2_knobs(self):
        """."""
        self.b2_indices = self.respm.fam_data['B2']['index']

    def update_bc_knobs(self):
        """."""
        self.bc_indices = self.respm.fam_data['BC']['index']

    def update_dip_knobs(self):
        """."""
        self.update_b1_knobs()
        self.update_b2_knobs()
        self.update_bc_knobs()
        self.dip_indices = self.b1_indices + self.b2_indices + self.bc_indices

    def _process_input(self, kwargs):
        for key, value in kwargs.items():
            if key == 'model' and 'dim' in kwargs:
                model, dim = kwargs['model'], kwargs['dim']
                self.update_model(model, dim)
            elif key == 'dim':
                pass
            elif key == 'svd_method' and ('svd_sel' in kwargs or
                                          'svd_thre' in kwargs):
                svd_method = kwargs['svd_method']
                svd_sel = kwargs['svd_sel'] if 'svd_sel' in kwargs else None
                svd_thre = kwargs['svd_thre'] if 'svd_thre' in kwargs else None
                self.update_svd(svd_method, svd_sel, svd_thre)
            setattr(self, key, value)

    def _create_indices(self):
        """."""
        self.idx_cav = pyaccel.lattice.find_indices(
            self.model, 'fam_name', self.FAMNAME_RF)[0]
        self.idx_bpm = pyaccel.lattice.find_indices(
            self.model, 'fam_name', 'BPM')


class LOCOConfigSI(LOCOConfig):
    """."""

    @property
    def acc(self):
        """."""
        return 'SI'

    @property
    def nr_bpm(self):
        """."""
        return 160

    @property
    def nr_ch(self):
        """."""
        return 120

    @property
    def nr_cv(self):
        """."""
        return 160

    @property
    def famname_dipoles(self):
        """."""
        return [
            'B1', 'B2', 'BC']

    @property
    def famname_quadset(self):
        """."""
        return [
            'QFA', 'QDA', 'QDB2', 'QFB', 'QDB1', 'QDP2', 'QFP', 'QDP1',
            'Q1', 'Q2', 'Q3', 'Q4']

    @property
    def famname_sextset(self):
        """."""
        return ['SDA0', 'SDB0', 'SDP0', 'SDA1', 'SDB1', 'SDP1',
                'SDA2', 'SDB2', 'SDP2', 'SDA3', 'SDB3', 'SDP3',
                'SFA0', 'SFB0', 'SFP0', 'SFA1', 'SFB1', 'SFP1',
                'SFA2', 'SFB2', 'SFP2', ]


class LOCOConfigBO(LOCOConfig):
    """."""

    @property
    def acc(self):
        """."""
        return 'BO'

    @property
    def nr_bpm(self):
        """."""
        return 50

    @property
    def nr_ch(self):
        """."""
        return 25

    @property
    def nr_cv(self):
        """."""
        return 25

    @property
    def famname_dipoles(self):
        """."""
        return ['B']

    @property
    def famname_quadset(self):
        """."""
        return ['QF', 'QD']

    @property
    def famname_sextset(self):
        """."""
        return ['SF', 'SD']


class LOCO:
    """LOCO."""

    UTILS = LOCOUtils
    DEFAULT_TOL = 1e-16
    DEFAULT_REDUC_THRESHOLD = 5/100

    def __init__(self, config=None):
        """."""
        if config is not None:
            self.config = config
        else:
            self.config = LOCOConfig()

        self._model = None
        self._matrix = None
        self._nr_k_sets = None

        self._jloco_gain_bpm = None
        self._jloco_roll_bpm = None
        self._jloco_gain_corr = None
        self._jloco_k = None
        self._jloco_k_quad = None
        self._jloco_k_sext = None
        self._jloco_k_dip = None
        self._jloco_k_b1 = None
        self._jloco_k_b2 = None
        self._jloco_k_bc = None

        self._jloco_ks = None
        self._jloco_ks_quad = None
        self._jloco_ks_sext = None
        self._jloco_ks_dip = None
        self._jloco_ks_b1 = None
        self._jloco_ks_b2 = None
        self._jloco_ks_bc = None

        self._jloco_kick_b1 = None
        self._jloco_kick_b2 = None
        self._jloco_kick_bc = None
        self._jloco_kick_dip = None

        self._jloco = None
        self._jloco_inv = None
        self._jtjloco_u = None
        self._jtjloco_s = None
        self._jtjloco_v = None
        self._jtjloco_inv = None

        self._quad_k_inival = None
        self._quad_k_deltas = None
        self._sext_k_inival = None
        self._sext_k_deltas = None
        self._b1_k_inival = None
        self._b1_k_deltas = None
        self._b2_k_inival = None
        self._b2_k_deltas = None
        self._bc_k_inival = None
        self._bc_k_deltas = None

        self._quad_ks_inival = None
        self._quad_ks_deltas = None
        self._sext_ks_inival = None
        self._sext_ks_deltas = None
        self._b1_ks_inival = None
        self._b1_ks_deltas = None
        self._b2_ks_inival = None
        self._b2_ks_deltas = None
        self._bc_ks_inival = None
        self._bc_ks_deltas = None

        self._gain_bpm_inival = self.config.gain_bpm
        self._gain_bpm_delta = None
        self._roll_bpm_inival = self.config.roll_bpm
        self._roll_bpm_delta = None
        self._gain_corr_inival = self.config.gain_corr
        self._gain_corr_delta = None

        self._chi_init = None
        self._chi = None
        self._tol = None
        self._reduc_threshold = None

    def update(self,
               fname_jloco_k=None,
               fname_inv_jloco_k=None,
               fname_jloco_k_quad=None,
               fname_jloco_k_sext=None,
               fname_jloco_k_dip=None,
               fname_jloco_ks_quad=None,
               fname_jloco_ks_sext=None,
               fname_jloco_ks_dip=None,
               fname_jloco_kick_dip=None):
        """."""
        print('update config...')
        self.update_config()
        if fname_inv_jloco_k is not None:
            print('setting jloco inverse input...')
            self._jloco_inv = LOCOUtils.load_data(fname=fname_inv_jloco_k)
        else:
            print('update jloco...')
            self.update_jloco(
                fname_jloco_k=fname_jloco_k,
                fname_jloco_k_quad=fname_jloco_k_quad,
                fname_jloco_k_sext=fname_jloco_k_sext,
                fname_jloco_k_dip=fname_jloco_k_dip,
                fname_jloco_ks_quad=fname_jloco_ks_quad,
                fname_jloco_ks_sext=fname_jloco_ks_sext,
                fname_jloco_ks_dip=fname_jloco_ks_dip,
                fname_jloco_kick_dip=fname_jloco_kick_dip)
            print('update svd...')
            self.update_svd()
        print('update fit...')
        self.update_fit()

    def update_config(self):
        """."""
        self.config.update()
        # reset model
        self._model = _dcopy(self.config.model)
        self._matrix = _dcopy(self.config.matrix)

    def update_jloco(self,
                     fname_jloco_k=None,
                     fname_jloco_k_quad=None,
                     fname_jloco_k_sext=None,
                     fname_jloco_k_dip=None,
                     fname_jloco_ks_quad=None,
                     fname_jloco_ks_sext=None,
                     fname_jloco_ks_dip=None,
                     fname_jloco_kick_dip=None):
        """."""
        # calc jloco linear parts
        self._jloco_gain_bpm, self._jloco_roll_bpm, self._jloco_gain_corr = \
            LOCOUtils.jloco_calc_linear(self.config, self._matrix)

        if fname_jloco_k is not None:
            self._jloco_k = LOCOUtils.load_data(fname_jloco_k)['jloco_kmatrix']
        else:
        # calc jloco kick part for dipole
            case_b1 = False
            case_b2 = False
            case_bc = False
            if not self.config.fit_kick_b1:
                self._jloco_kick_b1 = np.zeros(
                    (self._matrix.size, 1))
                case_b1 = True

            if not self.config.fit_kick_b2:
                self._jloco_kick_b2 = np.zeros(
                    (self._matrix.size, 1))
                case_b2 = True

            if not self.config.fit_kick_bc:
                self._jloco_kick_bc = np.zeros(
                    (self._matrix.size, 1))
                case_bc = True

            case = case_b1
            case &= case_b2
            case &= case_bc

            if not case:
                if fname_jloco_kick_dip is None:
                    print('calculating B1 kick matrix...')
                    self._jloco_kick_b1 = LOCOUtils.jloco_calc_kick_dipoles(
                        self.config, self._model, 'B1')
                    print('calculating B2 kick matrix...')
                    self._jloco_kick_b2 = LOCOUtils.jloco_calc_kick_dipoles(
                        self.config, self._model, 'B2')
                    print('calculating BC kick matrix...')
                    self._jloco_kick_bc = LOCOUtils.jloco_calc_kick_dipoles(
                        self.config, self._model, 'BC')
                    case = True
                else:
                    print('loading dipole kick matrix...')
                    self._jloco_kick_dip = LOCOUtils.load_data(
                        fname_jloco_kick_dip)['jloco_kmatrix']

            if case:
                self._jloco_kick_dip = np.hstack((
                    self._jloco_kick_b1, self._jloco_kick_b2))
                self._jloco_kick_dip = np.hstack((
                    self._jloco_kick_dip, self._jloco_kick_bc))

            # calc jloco Ks part for quadrupole
            if not self.config.fit_quadrupoles_coupling:
                self._jloco_ks_quad = np.zeros(
                    (self._matrix.size, len(self.config.quad_indices)))
            elif fname_jloco_ks_quad is None:
                print('calculating quadrupoles ksmatrix...')
                self._jloco_ks_quad = LOCOUtils.jloco_calc_ks_quad(
                    self.config, self._model)
            else:
                print('loading quadrupoles ksmatrix...')
                self._jloco_ks_quad = LOCOUtils.load_data(
                    fname_jloco_ks_quad)['jloco_kmatrix']

            # calc jloco Ks part for dipole
            case = False
            if not self.config.fit_b1_coupling:
                self._jloco_ks_b1 = np.zeros(
                    (self._matrix.size, len(self.config.b1_indices)))
                case = True

            if not self.config.fit_b2_coupling:
                self._jloco_ks_b2 = np.zeros(
                    (self._matrix.size, len(self.config.b2_indices)))
                case = True

            if not self.config.fit_bc_coupling:
                self._jloco_ks_bc = np.zeros(
                    (self._matrix.size, len(self.config.bc_indices)))
                case = True

            if not case:
                if fname_jloco_ks_dip is None:
                    print('calculating B1 ksmatrix...')
                    self._jloco_ks_b1 = LOCOUtils.jloco_calc_ks_dipoles(
                        self.config, self._model, 'B1')
                    print('calculating B2 ksmatrix...')
                    self._jloco_ks_b2 = LOCOUtils.jloco_calc_ks_dipoles(
                        self.config, self._model, 'B2')
                    print('calculating BC ksmatrix...')
                    self._jloco_ks_bc = LOCOUtils.jloco_calc_ks_dipoles(
                        self.config, self._model, 'BC')
                    case = True
                else:
                    print('loading dipole ksmatrix...')
                    self._jloco_ks_dip = LOCOUtils.load_data(
                        fname_jloco_ks_dip)['jloco_kmatrix']

            if case:
                self._jloco_ks_dip = np.hstack((
                    self._jloco_ks_b1, self._jloco_ks_b2))
                self._jloco_ks_dip = np.hstack((
                    self._jloco_ks_dip, self._jloco_ks_bc))

            # calc jloco Ks part for sextupole
            if not self.config.fit_sextupoles_coupling:
                self._jloco_ks_sext = np.zeros(
                    (self._matrix.size, len(self.config.sext_indices)))
            elif fname_jloco_ks_sext is None:
                print('calculating sextupoles ksmatrix...')
                self._jloco_ks_sext = LOCOUtils.jloco_calc_ks_sextupoles(
                    self.config, self._model)
            else:
                print('loading sextupoles ksmatrix...')
                self._jloco_ks_sext = LOCOUtils.load_data(
                    fname_jloco_ks_sext)['jloco_kmatrix']

            # calc jloco K part for quadrupole
            if not self.config.fit_quadrupoles:
                self._jloco_k_quad = np.zeros(
                    (self._matrix.size, len(self.config.quad_indices)))
            elif fname_jloco_k_quad is None:
                print('calculating quadrupoles kmatrix...')
                self._jloco_k_quad = LOCOUtils.jloco_calc_k_quad(
                    self.config, self._model)
            else:
                print('loading quadrupoles kmatrix...')
                self._jloco_k_quad = LOCOUtils.load_data(
                    fname_jloco_k_quad)['jloco_kmatrix']

            # calc jloco K part for dipole
            case = False
            if not self.config.fit_b1_coupling:
                self._jloco_k_b1 = np.zeros(
                    (self._matrix.size, len(self.config.b1_indices)))
                case = True

            if not self.config.fit_b2_coupling:
                self._jloco_k_b2 = np.zeros(
                    (self._matrix.size, len(self.config.b2_indices)))
                case = True

            if not self.config.fit_bc_coupling:
                self._jloco_k_bc = np.zeros(
                    (self._matrix.size, len(self.config.bc_indices)))
                case = True

            if not case:
                if fname_jloco_k_dip is None:
                    print('calculating B1 kmatrix...')
                    self._jloco_k_b1 = LOCOUtils.jloco_calc_k_dipoles(
                        self.config, self._model, 'B1')
                    print('calculating B2 kmatrix...')
                    self._jloco_k_b2 = LOCOUtils.jloco_calc_k_dipoles(
                        self.config, self._model, 'B2')
                    print('calculating BC kmatrix...')
                    self._jloco_k_bc = LOCOUtils.jloco_calc_k_dipoles(
                        self.config, self._model, 'BC')
                    case = True
                else:
                    print('loading dipole kmatrix...')
                    self._jloco_k_dip = LOCOUtils.load_data(
                        fname_jloco_k_dip)['jloco_kmatrix']

            if case:
                self._jloco_k_dip = np.hstack((
                    self._jloco_k_b1, self._jloco_k_b2))
                self._jloco_k_dip = np.hstack((
                    self._jloco_k_dip, self._jloco_k_bc))

            # calc jloco K part for sextupole
            if not self.config.fit_sextupoles:
                self._jloco_k_sext = np.zeros(
                    (self._matrix.size, len(self.config.sext_indices)))
            elif fname_jloco_k_sext is None:
                print('calculating sextupoles kmatrix...')
                self._jloco_k_sext = LOCOUtils.jloco_calc_k_sextupoles(
                    self.config, self._model)
            else:
                print('loading sextupoles kmatrix...')
                self._jloco_k_sext = LOCOUtils.load_data(
                    fname_jloco_k_sext)['jloco_kmatrix']

            self._jloco_k = np.hstack((self._jloco_k_quad, self._jloco_k_sext))
            self._jloco_k = np.hstack((self._jloco_k, self._jloco_k_dip))

            self._jloco_ks = np.hstack(
                (self._jloco_ks_quad, self._jloco_ks_sext))
            self._jloco_ks = np.hstack(
                (self._jloco_ks, self._jloco_ks_dip))

        # merge J submatrices
        self._jloco = LOCOUtils.jloco_merge_linear(
            self.config, self._jloco_k, self._jloco_ks,
            self._jloco_gain_bpm, self._jloco_roll_bpm,
            self._jloco_gain_corr, self._jloco_kick_dip)

        # filter jloco
        self._jloco = LOCOUtils.jloco_param_delete(self.config, self._jloco)

        # apply weight
        self._jloco = LOCOUtils.jloco_apply_weight(
            self._jloco, self.config.weight_bpm, self.config.weight_corr)

        # calc jloco inv
        self._jtjloco_u, self._jtjloco_s, self._jtjloco_v = \
            np.linalg.svd(self._jloco.T @ self._jloco, full_matrices=False)

    def jloco_calc_energy_shift(self):
        """."""
        matrix0 = _dcopy(self._matrix)
        energy_shift = np.zeros((self.config.nr_corr, 1))
        delta_energy = 1e-8
        dm_energy_shift = np.zeros((matrix0.size, self.config.nr_corr))
        for c in range(self.config.nr_corr):
            energy_shift[c] = delta_energy
            matrix_shift = energy_shift[:, None] * self.disp_meas[None, :]
            dm_energy_shift[:, c] = matrix_shift.flatten()
            energy_shift[c] = 0
        return dm_energy_shift



    def update_svd(self, svd_thre=None, svd_sel=None):
        """."""
        u, s, v = self._jtjloco_u, self._jtjloco_s, self._jtjloco_v
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0

        if svd_thre is None:
            svd_thre = self.config.svd_thre
        if svd_sel is None:
            svd_sel = self.config.svd_sel

        if self.config.svd_method == self.config.SVD_METHOD_THRESHOLD:
            bad_sv = s/np.max(s) < svd_thre
            # print('removing {:d} bad singular values...'.format(
            #     np.sum(bad_sv)))
            inv_s[bad_sv] = 0
        elif self.config.svd_method == self.config.SVD_METHOD_SELECTION:
            inv_s[svd_sel:] = 0

        self._jtjloco_inv = np.dot(v.T * inv_s[None, :], u.T)

    def update_fit(self):
        """."""
        # k inival and deltas
        if self.config.use_families:
            self._quad_k_inival = LOCOUtils.get_quads_strengths(
                model=self._model, indices=self.config.quad_indices)
        else:
            self._quad_k_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'K', self.config.quad_indices))

        self._sext_k_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'K', self.config.sext_indices))
        self._b1_k_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'K', self.config.b1_indices))
        self._b2_k_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'K', self.config.b2_indices))
        self._bc_k_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'K', self.config.bc_indices))

        self._quad_ks_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'Ks', self.config.quad_indices))
        self._sext_ks_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'Ks', self.config.sext_indices))
        self._b1_ks_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'Ks', self.config.b1_indices))
        self._b2_ks_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'Ks', self.config.b2_indices))
        self._bc_ks_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'Ks', self.config.bc_indices))

        self._b1_kick_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'hkick_polynom', self.config.b1_indices))
        self._b2_kick_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'hkick_polynom', self.config.b2_indices))
        self._bc_kick_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'hkick_polynom', self.config.bc_indices))

        self._quad_k_deltas = np.zeros(len(self.config.quad_indices))
        self._sext_k_deltas = np.zeros(len(self.config.sext_indices))
        self._b1_k_deltas = np.zeros(len(self.config.b1_indices))
        self._b2_k_deltas = np.zeros(len(self.config.b2_indices))
        self._bc_k_deltas = np.zeros(len(self.config.bc_indices))

        self._quad_ks_deltas = np.zeros(len(self.config.quad_indices))
        self._sext_ks_deltas = np.zeros(len(self.config.sext_indices))
        self._b1_ks_deltas = np.zeros(len(self.config.b1_indices))
        self._b2_ks_deltas = np.zeros(len(self.config.b2_indices))
        self._bc_ks_deltas = np.zeros(len(self.config.bc_indices))

        self._b1_kick_deltas = np.zeros(len(self.config.b1_indices))
        self._b2_kick_deltas = np.zeros(len(self.config.b2_indices))
        self._bc_kick_deltas = np.zeros(len(self.config.bc_indices))

        # bpm inival and deltas
        if self._gain_bpm_inival is None:
            self._gain_bpm_inival = np.ones(2*self.config.nr_bpm)
        if self._roll_bpm_inival is None:
            self._roll_bpm_inival = np.zeros(self.config.nr_bpm)
        self._gain_bpm_delta = np.zeros(2*self.config.nr_bpm)
        self._roll_bpm_delta = np.zeros(self.config.nr_bpm)

        # corr inival and deltas
        if self._gain_corr_inival is None:
            self._gain_corr_inival = np.ones(self.config.nr_corr)
        self._gain_corr_delta = np.zeros(self.config.nr_corr)

        check_case = self._gain_bpm_inival is not None
        check_case |= self._roll_bpm_inival is not None
        check_case |= self._gain_corr_inival is not None

        if check_case:
            self._matrix = LOCOUtils.apply_all_gain(
                matrix=self._matrix,
                gain_bpm=self._gain_bpm_inival,
                roll_bpm=self._roll_bpm_inival,
                gain_corr=self._gain_corr_inival)

        self._chi = self.calc_chi()
        self._chi_init = self._chi
        print('chi_init: {0:.4f} um'.format(self._chi_init))

        self._tol = LOCO.DEFAULT_TOL
        self._reduc_threshold = LOCO.DEFAULT_REDUC_THRESHOLD

    def run_fit(self, niter=1):
        """."""
        self._chi = self._chi_init
        for _iter in range(niter):
            print('iter # {}/{}'.format(_iter+1, niter))


            matrix_diff = self.config.goalmat - self._matrix
            matrix_diff = LOCOUtils.apply_all_weight(
                matrix_diff, self.config.weight_bpm, self.config.weight_corr)
            param_new = np.dot(
                self._jtjloco_inv, np.dot(
                    self._jloco.T, matrix_diff.flatten()))
            param_new = param_new.flatten()
            model_new, matrix_new = self._calc_model_matrix(param_new)
            chi_new = self.calc_chi(matrix_new)
            print('chi: {0:.4f} um'.format(chi_new))
            if np.isnan(chi_new):
                print('matrix deviation is NaN!')
                break
            if chi_new < self._chi:
                self._update_state(model_new, matrix_new, chi_new)
            else:
                # print('recalculating jloco...')
                # self.update_jloco()
                # self.update_svd()
                factor = \
                    self._try_refactor_param(param_new)
                if factor <= self._reduc_threshold:
                    # could not converge at current iteration!
                    break
            if self._chi < self._tol:
                break

        print('Finished!')

    def calc_chi(self, matrix=None):
        """."""
        if matrix is None:
            matrix = self._matrix
        dmatrix = matrix - self.config.goalmat
        dmatrix[:, :self.config.nr_ch] *= self.config.delta_kickx_meas
        dmatrix[:, self.config.nr_ch:-1] *= self.config.delta_kicky_meas
        dmatrix[:, -1] *= self.config.delta_frequency_meas
        chi = np.linalg.norm(dmatrix)/np.sqrt(dmatrix.size)
        return chi * 1e6

    @property
    def jloco_k(self):
        """."""
        if self._jloco_k is None:
            self._jloco_k = LOCOUtils.jloco_calc_k_quad(
                self.config, self._model)
        return self._jloco_k

    def save_jloco_k(self, fname):
        """."""
        np.savetxt(fname, self.jloco_k)

    def save_jloco(self, fname):
        """."""
        np.savetxt(fname, self._jloco)

    def _calc_model_matrix(self, param):
        """."""
        model = _dcopy(self._model)
        config = self.config
        param_dict = LOCOUtils.param_select(config, param)
        param_names = {
            'quadrupoles',
            'sextupoles',
            'b1', 'b2', 'bc'}

        if bool(param_names.intersection(set(param_dict.keys()))):
            if 'quadrupoles' in param_dict:
                # update quadrupole delta
                self._quad_k_deltas += param_dict['quadrupoles']
                # update local model
                if self.config.use_families:
                    set_quad_kdelta = LOCOUtils.set_quadset_kdelta
                else:
                    set_quad_kdelta = LOCOUtils.set_quadmag_kdelta
                for idx, idx_set in enumerate(config.quad_indices):
                    set_quad_kdelta(
                        model, idx_set,
                        self._quad_k_inival[idx], self._quad_k_deltas[idx])
            if 'sextupoles' in param_dict:
                # update sextupole delta
                self._sext_k_deltas += param_dict['sextupoles']
                # update local model
                set_quad_kdelta = LOCOUtils.set_quadmag_kdelta
                for idx, idx_set in enumerate(config.sext_indices):
                    set_quad_kdelta(
                        model, idx_set,
                        self._sext_k_inival[idx], self._sext_k_deltas[idx])
            if 'b1' in param_dict:
                # update b1 delta
                self._b1_k_deltas += param_dict['b1']
                # update local model
                for idx, idx_set in enumerate(config.b1_indices):
                    LOCOUtils.set_dipmag_kdelta(
                        model, idx_set,
                        self._b1_k_inival[idx], self._b1_k_deltas[idx])
            if 'b2' in param_dict:
                # update b2 delta
                self._b2_k_deltas += param_dict['b2']
                # update local model
                for idx, idx_set in enumerate(config.b2_indices):
                    LOCOUtils.set_dipmag_kdelta(
                        model, idx_set,
                        self._b2_k_inival[idx], self._b2_k_deltas[idx])
            if 'bc' in param_dict:
                # update bc delta
                self._bc_k_deltas += param_dict['bc']
                # update local model
                for idx, idx_set in enumerate(config.bc_indices):
                    LOCOUtils.set_dipmag_kdelta(
                        model, idx_set,
                        self._bc_k_inival[idx], self._bc_k_deltas[idx])
            if 'quadrupoles_coupling' in param_dict:
                # update quadrupole Ks delta
                self._quad_ks_deltas += param_dict['quadrupoles_coupling']
                # update local model
                set_quad_ksdelta = LOCOUtils.set_quadmag_ksdelta
                for idx, idx_set in enumerate(config.quad_indices):
                    set_quad_ksdelta(
                        model, idx_set,
                        self._quad_ks_inival[idx], self._quad_ks_deltas[idx])
            if 'sextupoles_coupling' in param_dict:
                # update sextupole Ks delta
                self._sext_ks_deltas += param_dict['sextupoles_coupling']
                # update local model
                set_quad_ksdelta = LOCOUtils.set_quadmag_ksdelta
                for idx, idx_set in enumerate(config.sext_indices):
                    set_quad_ksdelta(
                        model, idx_set,
                        self._sext_ks_inival[idx], self._sext_ks_deltas[idx])
            if 'b1_coupling' in param_dict:
                # update b1 Ks delta
                self._b1_ks_deltas += param_dict['b1_coupling']
                # update local model
                for idx, idx_set in enumerate(config.b1_indices):
                    LOCOUtils.set_dipmag_ksdelta(
                        model, idx_set,
                        self._b1_ks_inival[idx], self._b1_ks_deltas[idx])
            if 'b2_coupling' in param_dict:
                # update b2 Ks delta
                self._b2_ks_deltas += param_dict['b2_coupling']
                # update local model
                for idx, idx_set in enumerate(config.b2_indices):
                    LOCOUtils.set_dipmag_ksdelta(
                        model, idx_set,
                        self._b2_ks_inival[idx], self._b2_ks_deltas[idx])
            if 'bc_coupling' in param_dict:
                # update bc delta
                self._bc_ks_deltas += param_dict['bc_coupling']
                # update local model
                for idx, idx_set in enumerate(config.bc_indices):
                    LOCOUtils.set_dipmag_ksdelta(
                        model, idx_set,
                        self._bc_ks_inival[idx], self._bc_ks_deltas[idx])
            if 'kick_b1' in param_dict:
                # update b1 kick delta
                self._b1_kick_deltas += np.repeat(
                    param_dict['kick_b1'], len(self.config.b1_indices))
                # update local model
                for idx, idx_set in enumerate(config.b1_indices):
                    LOCOUtils.set_dipmag_kick(
                        model, idx_set,
                        self._b1_kick_inival[idx], self._b1_kick_deltas[idx])
            if 'kick_b2' in param_dict:
                # update b2 kick delta
                self._b2_kick_deltas += np.repeat(
                    param_dict['kick_b2'], len(self.config.b2_indices))
                # update local model
                for idx, idx_set in enumerate(config.b2_indices):
                    LOCOUtils.set_dipmag_kick(
                        model, idx_set,
                        self._b2_kick_inival[idx], self._b2_kick_deltas[idx])
            if 'kick_bc' in param_dict:
                # update bc kick delta
                self._bc_kick_deltas += np.repeat(
                    param_dict['kick_bc'], len(self.config.bc_indices))
                # update local model
                for idx, idx_set in enumerate(config.bc_indices):
                    LOCOUtils.set_dipmag_kick(
                        model, idx_set,
                        self._bc_kick_inival[idx], self._bc_kick_deltas[idx])
            matrix = LOCOUtils.respm_calc(model, config.respm, config.use_disp)
        else:
            matrix = _dcopy(self.config.matrix)

        if 'gain_bpm' in param_dict:
            # update gain delta
            self._gain_bpm_delta += param_dict['gain_bpm']
            gain = self._gain_bpm_inival + self._gain_bpm_delta
            matrix = LOCOUtils.apply_bpm_gain(matrix, gain)

        if 'roll_bpm' in param_dict:
            # update roll delta
            self._roll_bpm_delta += param_dict['roll_bpm']
            roll = self._roll_bpm_inival + self._roll_bpm_delta
            matrix = LOCOUtils.apply_bpm_roll(matrix, roll)

        if 'gain_corr' in param_dict:
            # update gain delta
            self._gain_corr_delta += param_dict['gain_corr']
            gain = self._gain_corr_inival + self._gain_corr_delta
            matrix = LOCOUtils.apply_corr_gain(matrix, gain)

        return model, matrix

    def _try_refactor_param(self, param_new):
        """."""
        factor = 0.5
        _iter = 1
        while factor > self._reduc_threshold:
            print('chi was increased! Trial {0:d}'.format(_iter))
            print('applying {0:0.4f} %'.format(100*factor))
            model_new, matrix_new = \
                self._calc_model_matrix(factor*param_new)
            chi_new = self.calc_chi(matrix_new)
            print('chi: {0:.4f} um'.format(chi_new))
            if chi_new < self._chi:
                self._update_state(model_new, matrix_new, chi_new)
                break
            factor /= 2
            _iter += 1
        return factor

    def _update_state(self, model_new, matrix_new, chi_new):
        """."""
        self._model = model_new
        self._matrix = matrix_new
        self._chi = chi_new
