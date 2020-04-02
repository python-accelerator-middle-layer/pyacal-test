"""."""

from copy import deepcopy as _dcopy
import time
import pickle as _pickle
import numpy as np
import pyaccel
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat
from siriuspy.namesys import SiriusPVName as _PVName


class LOCOUtils:
    """LOCO utils."""

    @staticmethod
    def save_data(fname, jlocodict):
        """."""
        data = jlocodict
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
    def respm_calc(model, respm, use_dispersion):
        """."""
        respm.model = _dcopy(model)
        matrix = respm.get_respm()
        if not use_dispersion:
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
    def add_dispersion_to_respm(matrix, energy_shift, dispersion):
        """."""
        matrix_out = _dcopy(matrix)
        matrix_out[:, :-1] += dispersion[:, None] * energy_shift[None, :]
        return matrix_out

    @staticmethod
    def get_quads_strengths(model, indices):
        """."""
        kquads = []
        for qidx in indices:
            kquads.append(pyaccel.lattice.get_attribute(
                model, 'KL', qidx))
        return np.array(kquads)

    @staticmethod
    def set_quadmag_kdelta(model, idx_mag, kvalues, kdelta):
        """."""
        for idx, idx_seg in enumerate(idx_mag):
            pyaccel.lattice.set_attribute(
                model, 'KL', idx_seg, kvalues[idx] + kdelta/len(idx_mag))

    @staticmethod
    def set_quadset_kdelta(model, idx_set, kvalues, kdelta):
        """."""
        for idx, idx_mag in enumerate(idx_set):
            LOCOUtils.set_quadmag_kdelta(
                model, idx_mag, kvalues[idx], kdelta)

    @staticmethod
    def set_dipmag_kdelta(model, idx_mag, kvalues, kdelta):
        """."""
        ktotal = np.sum(kvalues)
        if ktotal:
            newk = [kval*(1+kdelta/ktotal) for kval in kvalues]
            pyaccel.lattice.set_attribute(
                model, 'KL', idx_mag, newk)
        else:
            newk = [kval + kdelta/len(idx_mag) for kval in kvalues]
            pyaccel.lattice.set_attribute(
                model, 'KL', idx_mag, kvalues + kdelta/len(idx_mag))

    @staticmethod
    def set_dipset_kdelta(model, idx_set, kvalues, kdelta):
        """."""
        for idx, idx_mag in enumerate(idx_set):
            LOCOUtils.set_dipmag_kdelta(
                model, idx_mag, kvalues[idx], kdelta)

    @staticmethod
    def set_quadmag_ksdelta(model, idx_mag, ksvalues, ksdelta):
        """."""
        pyaccel.lattice.set_attribute(
            model, 'KsL', idx_mag, ksvalues + ksdelta)

    @staticmethod
    def set_dipmag_ksdelta(model, idx_mag, ksvalues, ksdelta):
        """."""
        kstotal = np.sum(ksvalues)
        if kstotal:
            newks = [ksval*(1+ksdelta/kstotal) for ksval in ksvalues]
            pyaccel.lattice.set_attribute(
                model, 'KsL', idx_mag, newks)
        else:
            newks = [ksval + ksdelta/len(idx_mag) for ksval in ksvalues]
            pyaccel.lattice.set_attribute(
                model, 'KsL', idx_mag, newks)

    @staticmethod
    def set_dipmag_kick(model, idx_mag, kick_values, kick_delta):
        """."""
        angle = np.array(
            pyaccel.lattice.get_attribute(model, 'angle', idx_mag))
        angle /= np.sum(angle)
        pyaccel.lattice.set_attribute(
            model, 'hkick_polynom', idx_mag, kick_values + kick_delta * angle)

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

        if shape1 < ncorr + 1 and config.use_dispersion:
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
        if config.use_quad_families:
            kindices = []
            for fam_name in config.famname_quadset:
                kindices.append(config.respm.fam_data[fam_name]['index'])
            kvalues = LOCOUtils.get_quads_strengths(
                model, kindices)
            set_quad_kdelta = LOCOUtils.set_quadset_kdelta
        else:
            kindices = config.respm.fam_data['QN']['index']
            kvalues = np.array(
                pyaccel.lattice.get_attribute(model, 'KL', kindices))
            set_quad_kdelta = LOCOUtils.set_quadmag_kdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        kmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(kindices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(kindices):
            set_quad_kdelta(
                model_this, idx_set,
                kvalues[idx], config.DEFAULT_DELTA_K)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_dispersion)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_K
            kmatrix[:, idx] = dmatrix.flatten()
            set_quad_kdelta(model_this, idx_set, kvalues[idx], 0)
        return kmatrix

    @staticmethod
    def jloco_calc_k_dipoles(config, model):
        """."""
        if config.use_dip_families:
            dip_indices = []
            for fam_name in config.famname_dipset:
                dip_indices.append(config.respm.fam_data[fam_name]['index'])
            dip_kvalues = LOCOUtils.get_quads_strengths(
                model, dip_indices)
            set_quad_kdelta = LOCOUtils.set_dipset_kdelta
        else:
            dip_indices = config.respm.fam_data['BN']['index']
            dip_kvalues = np.array(
                pyaccel.lattice.get_attribute(model, 'KL', dip_indices))
            set_quad_kdelta = LOCOUtils.set_dipmag_kdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        dip_kmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(dip_indices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(dip_indices):
            set_quad_kdelta(
                model_this, idx_set,
                dip_kvalues[idx], config.DEFAULT_DELTA_K)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_dispersion)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_K
            dip_kmatrix[:, idx] = dmatrix.flatten()
            set_quad_kdelta(model_this, idx_set, dip_kvalues[idx], 0)
        return dip_kmatrix

    @staticmethod
    def jloco_calc_k_sextupoles(config, model):
        """."""
        sn_indices = config.respm.fam_data['SN']['index']
        sn_kvalues = np.array(
            pyaccel.lattice.get_attribute(model, 'KL', sn_indices))
        set_quad_kdelta = LOCOUtils.set_quadmag_kdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        sn_kmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(sn_indices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(sn_indices):
            set_quad_kdelta(
                model_this, idx_set, sn_kvalues[idx], config.DEFAULT_DELTA_K)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_dispersion)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_K
            sn_kmatrix[:, idx] = dmatrix.flatten()
            set_quad_kdelta(model_this, idx_set, sn_kvalues[idx], 0)
        return sn_kmatrix

    @staticmethod
    def jloco_calc_ks_quad(config, model):
        """."""
        kindices = config.respm.fam_data['QN']['index']
        ksvalues = np.array(
            pyaccel.lattice.get_attribute(model, 'KsL', kindices))
        set_quad_ksdelta = LOCOUtils.set_quadmag_ksdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        ksmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(kindices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(kindices):
            set_quad_ksdelta(
                model_this, idx_set, ksvalues[idx], config.DEFAULT_DELTA_KS)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_dispersion)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_KS
            ksmatrix[:, idx] = dmatrix.flatten()
            set_quad_ksdelta(model_this, idx_set, ksvalues[idx], 0)
        return ksmatrix

    @staticmethod
    def jloco_calc_ks_dipoles(config, model):
        """."""
        dip_indices = config.respm.fam_data['BN']['index']
        dip_ksvalues = np.array(
            pyaccel.lattice.get_attribute(model, 'KsL', dip_indices))
        set_quad_ksdelta = LOCOUtils.set_dipmag_ksdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        dip_ksmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(dip_indices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(dip_indices):
            set_quad_ksdelta(
                model_this, idx_set, dip_ksvalues[idx],
                config.DEFAULT_DELTA_KS)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_dispersion)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_KS
            dip_ksmatrix[:, idx] = dmatrix.flatten()
            set_quad_ksdelta(model_this, idx_set, dip_ksvalues[idx], 0)
        return dip_ksmatrix

    @staticmethod
    def jloco_calc_ks_sextupoles(config, model):
        """."""
        sn_indices = config.respm.fam_data['SN']['index']
        sn_ksvalues = np.array(
            pyaccel.lattice.get_attribute(model, 'KsL', sn_indices))
        set_quad_ksdelta = LOCOUtils.set_quadmag_ksdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        sn_ksmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(sn_indices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(sn_indices):
            set_quad_ksdelta(
                model_this, idx_set, sn_ksvalues[idx], config.DEFAULT_DELTA_KS)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_dispersion)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_KS
            sn_ksmatrix[:, idx] = dmatrix.flatten()
            set_quad_ksdelta(model_this, idx_set, sn_ksvalues[idx], 0)
        return sn_ksmatrix

    @staticmethod
    def jloco_calc_kick_dipoles(config, model):
        """."""
        dip_indices = config.respm.fam_data['BN']['index']
        dip_kick_values = np.array(
            pyaccel.lattice.get_attribute(model, 'hkick_polynom', dip_indices))
        set_dip_kick = LOCOUtils.set_dipmag_kick
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        dip_kick_matrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], 1))

        delta_kick = config.DEFAULT_DELTA_DIP_KICK

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(dip_indices):
            set_dip_kick(
                model_this, idx_set,
                dip_kick_values[idx], delta_kick)
            nmags = len(idx_set)
        matrix_this = LOCOUtils.respm_calc(
            model_this, config.respm, config.use_dispersion)
        dmatrix = (matrix_this - matrix_nominal) / delta_kick / nmags
        dip_kick_matrix[:, 0] = dmatrix.flatten()

        for idx, idx_set in enumerate(dip_indices):
            set_dip_kick(model_this, idx_set, dip_kick_values[idx], 0)
        return dip_kick_matrix

    @staticmethod
    def jloco_calc_energy_shift(config, model):
        """."""
        matrix0 = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)
        energy_shift = np.zeros(config.nr_corr + 1)
        dm_energy_shift = np.zeros((matrix0.size, config.nr_corr))
        for c in range(config.nr_corr):
            energy_shift[c] = 1
            matrix_shift = config.measured_dispersion[:, None] * \
                energy_shift[None, :]
            dm_energy_shift[:, c] = matrix_shift.flatten()
            energy_shift[c] = 0
        return dm_energy_shift

    @staticmethod
    def jloco_merge_linear(
            config, km_quad, km_sext, km_dip,
            ksm_quad, ksm_sext, ksm_dip,
            dmdg_bpm, dmdalpha_bpm, dmdg_corr,
            kick_dip, energy_shift):
        """."""
        nbpm = config.nr_bpm
        nch = config.nr_ch
        ncv = config.nr_cv
        knobs_k = 0
        knobs_ks = 0
        knobs_linear = 0

        if km_quad is not None:
            knobs_k += km_quad.shape[1]
        if km_sext is not None:
            knobs_k += km_sext.shape[1]
        if km_dip is not None:
            knobs_k += km_dip.shape[1]
        if ksm_quad is not None:
            knobs_ks += ksm_quad.shape[1]
        if ksm_sext is not None:
            knobs_ks += ksm_sext.shape[1]
        if ksm_dip is not None:
            knobs_ks += ksm_dip.shape[1]

        if config.fit_gain_bpm:
            knobs_linear += 2*nbpm
        if config.fit_roll_bpm:
            knobs_linear += nbpm
        if config.fit_gain_corr:
            knobs_linear += nch + ncv
        if config.fit_energy_shift:
            knobs_linear += nch + ncv
        if config.fit_dipoles_kick:
            knobs_linear += 3

        nknobs = knobs_k + knobs_ks + knobs_linear
        jloco = np.zeros(
            (2*nbpm*(nch+ncv+1), nknobs))
        idx = 0
        if config.fit_quadrupoles:
            n = km_quad.shape[1]
            jloco[:, idx:idx+n] = km_quad
            idx += n
        if config.fit_sextupoles:
            n = km_sext.shape[1]
            jloco[:, idx:idx+n] = km_sext
            idx += n
        if config.fit_dipoles:
            n = km_dip.shape[1]
            jloco[:, idx:idx+n] = km_dip
            idx += n
        if config.fit_quadrupoles_coupling:
            n = ksm_quad.shape[1]
            jloco[:, idx:idx+n] = ksm_quad
            idx += n
        if config.fit_sextupoles_coupling:
            n = ksm_sext.shape[1]
            jloco[:, idx:idx+n] = ksm_sext
            idx += n
        if config.fit_dipoles_coupling:
            n = ksm_dip.shape[1]
            jloco[:, idx:idx+n] = ksm_dip
            idx += n
        if config.fit_gain_bpm:
            n = dmdg_bpm.shape[1]
            jloco[:, idx:idx+n] = dmdg_bpm
            idx += n
        if config.fit_roll_bpm:
            n = dmdalpha_bpm.shape[1]
            jloco[:, idx:idx+n] = dmdalpha_bpm
            idx += n
        if config.fit_gain_corr:
            n = dmdg_corr.shape[1]
            jloco[:, idx:idx+n] = dmdg_corr
            idx += n
        if config.fit_dipoles_kick:
            n = kick_dip.shape[1]
            jloco[:, idx:idx+n] = kick_dip
            idx += n
        if config.fit_energy_shift:
            n = energy_shift.shape[1]
            jloco[:, idx:idx+n] = energy_shift
            idx += n
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
            param_dict['quadrupoles_gradient'] = param[idx:idx+size]
            idx += size
        if config.fit_sextupoles:
            size = len(config.sext_indices)
            param_dict['sextupoles_gradient'] = param[idx:idx+size]
            idx += size
        if config.fit_dipoles:
            size = len(config.dip_indices)
            param_dict['dipoles_gradient'] = param[idx:idx+size]
            idx += size
        if config.fit_quadrupoles_coupling:
            size = len(config.quad_indices_ks)
            param_dict['quadrupoles_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_sextupoles_coupling:
            size = len(config.sext_indices)
            param_dict['sextupoles_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_dipoles_coupling:
            size = len(config.dip_indices_ks)
            param_dict['dipoles_coupling'] = param[idx:idx+size]
            idx += size
        if config.fit_gain_bpm:
            size = 2*config.nr_bpm
            param_dict['gain_bpm'] = param[idx:idx+size]
            idx += size
        if config.fit_roll_bpm:
            size = config.nr_bpm
            param_dict['roll_bpm'] = param[idx:idx+size]
            idx += size
        if config.fit_gain_corr:
            size = config.nr_corr
            param_dict['gain_corr'] = param[idx:idx+size]
            idx += size
        if config.fit_dipoles_kick:
            size = len(config.dip_indices)
            param_dict['dipoles_kick'] = param[idx:idx+size]
            idx += size
        if config.fit_energy_shift:
            size = config.nr_corr
            param_dict['energy_shift'] = param[idx:idx+size]
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
        self.measured_dispersion = None
        self.delta_kickx_meas = None
        self.delta_kicky_meas = None
        self.delta_frequency_meas = None
        self.fitting_method = None
        self.lambda_lm = None
        self.use_dispersion = None
        self.use_coupling = None
        self.use_quad_families = None
        self.dipoles_to_fit = None
        self.quadrupoles_to_fit = None
        self.sextupoles_to_fit = None
        self.use_dip_families = None
        self.svd_method = None
        self.svd_sel = None
        self.svd_thre = None
        self.fit_quadrupoles = None
        self.fit_sextupoles = None
        self.fit_dipoles = None
        self.fit_quadrupoles_coupling = None
        self.fit_sextupoles_coupling = None
        self.fit_dipoles_coupling = None
        self.fit_gain_bpm = None
        self.fit_roll_bpm = None
        self.fit_gain_corr = None
        self.fit_dipoles_kick = None
        self.fit_energy_shift = None
        self.constraint_deltak = None
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
        self.quad_indices_ks = None
        self.sext_indices = None
        self.dip_indices = None
        self.dip_indices_ks = None
        self.b1_indices = None
        self.b2_indices = None
        self.bc_indices = None
        self.k_nrsets = None
        self.weight_bpm = None
        self.weight_corr = None
        self.weight_deltak = None

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

    @property
    def famname_sextset(self):
        """."""
        raise NotImplementedError

    @property
    def famname_dipset(self):
        """."""
        raise NotImplementedError

    def update(self):
        """."""
        self.update_model(self.model, self.dim)
        self.update_matrix(self.use_dispersion)
        self.update_goalmat(
            self.goalmat, self.use_dispersion, self.use_coupling)
        self.update_gain()
        self.update_quad_knobs(self.use_quad_families)
        self.update_sext_knobs()
        self.update_dip_knobs(self.use_dip_families)
        self.update_weight()
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

    def update_goalmat(self, goalmat, use_dispersion, use_coupling):
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
        self.use_dispersion = use_dispersion
        if not self.use_dispersion:
            self.goalmat[:, -1] *= 0

    def update_matrix(self, use_dispersion):
        """."""
        self.matrix = LOCOUtils.respm_calc(
            self.model, self.respm, use_dispersion)

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

        nquads = len(self.quad_indices)
        print(nquads)
        # delta K
        if self.weight_deltak is None:
            self.weight_deltak = np.ones(nquads)
        elif isinstance(self.weight_deltak, (int, float)):
            self.weight_deltak = np.ones(nquads) * \
                self.weight_deltak / (nquads)

    def update_quad_knobs(self, use_families):
        """."""
        if self.quadrupoles_to_fit is None:
            self.quadrupoles_to_fit = self.famname_quadset
        else:
            setquadfit = set(self.quadrupoles_to_fit)
            setquadall = set(self.famname_quadset)
            if not setquadfit.issubset(setquadall):
                raise Exception('invalid quadrupole name used to fit!')
        self.use_quad_families = use_families
        if self.use_quad_families:
            self.quad_indices = [None] * len(self.quadrupoles_to_fit)
            self.quad_indices_ks = []
            for idx, fam_name in enumerate(self.quadrupoles_to_fit):
                self.quad_indices[idx] = self.respm.fam_data[fam_name]['index']
                self.quad_indices_ks += self.quad_indices[idx]
                self.quad_indices_ks.sort()
        else:
            self.quad_indices = []
            for fam_name in self.quadrupoles_to_fit:
                self.quad_indices += self.respm.fam_data[fam_name]['index']
                self.quad_indices.sort()
                self.quad_indices_ks = self.quad_indices

    def update_sext_knobs(self):
        """."""
        if self.sextupoles_to_fit is None:
            self.sext_indices = self.respm.fam_data['SN']['index']
        else:
            setsextfit = set(self.sextupoles_to_fit)
            setsextall = set(self.famname_sextset)
            if not setsextfit.issubset(setsextall):
                raise Exception('invalid sextupole name used to fit!')
            else:
                self.sext_indices = []
                for fam_name in self.sextupoles_to_fit:
                    self.sext_indices += self.respm.fam_data[fam_name]['index']
                    self.sext_indices.sort()

    def update_b1_knobs(self):
        """."""
        self.b1_indices = self.respm.fam_data['B1']['index']

    def update_b2_knobs(self):
        """."""
        self.b2_indices = self.respm.fam_data['B2']['index']

    def update_bc_knobs(self):
        """."""
        self.bc_indices = self.respm.fam_data['BC']['index']

    def update_dip_knobs(self, use_families):
        """."""
        if self.dipoles_to_fit is None:
            self.dipoles_to_fit = self.famname_dipset
        else:
            setdipfit = set(self.dipoles_to_fit)
            setdipall = set(self.famname_dipset)
            if not setdipfit.issubset(setdipall):
                raise Exception('invalid dipole name used to fit!')
        self.use_dip_families = use_families
        if self.use_dip_families:
            self.dip_indices = [None] * len(self.dipoles_to_fit)
            self.dip_indices_ks = []
            for idx, fam_name in enumerate(self.dipoles_to_fit):
                self.dip_indices[idx] = self.respm.fam_data[fam_name]['index']
                self.dip_indices_ks += self.dip_indices[idx]
                self.dip_indices_ks.sort()
        else:
            self.dip_indices = []
            for fam_name in self.dipoles_to_fit:
                self.dip_indices += self.respm.fam_data[fam_name]['index']
                self.dip_indices.sort()
                self.dip_indices_ks = self.dip_indices

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
    def famname_dipset(self):
        """."""
        return [
            'B1', 'B2', 'BC']

    @property
    def famname_quadset(self):
        """."""
        return ['QFA', 'QDA', 'QDB2', 'QFB', 'QDB1', 'QDP2', 'QFP', 'QDP1',
                'Q1', 'Q2', 'Q3', 'Q4']

    @property
    def famname_sextset(self):
        """."""
        return ['SDA0', 'SDB0', 'SDP0', 'SDA1', 'SDB1', 'SDP1',
                'SDA2', 'SDB2', 'SDP2', 'SDA3', 'SDB3', 'SDP3',
                'SFA0', 'SFB0', 'SFP0', 'SFA1', 'SFB1', 'SFP1',
                'SFA2', 'SFB2', 'SFP2']


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
    def famname_dipset(self):
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
    DEFAULT_TOL = 1e-3
    DEFAULT_REDUC_THRESHOLD = 5/100
    DEFAULT_LAMBDA_LM = 1e-3
    DEFAULT_DELTAK_NORMALIZATION = 1
    JLOCO_INVERSION = 'normal'

    def __init__(self, config=None):
        """."""
        if config is not None:
            self.config = config
        else:
            self.config = LOCOConfig()

        if self.config.fitting_method == 'Levenberg-Marquadt':
            if self.config.lambda_lm is None:
                self.config.lambda_lm = LOCO.DEFAULT_LAMBDA_LM

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

        self._jloco_energy_shift = None

        self._jloco = None
        self._jloco_inv = None
        self._jtjloco_u = None
        self._jtjloco_s = None
        self._jtjloco_v = None
        self._jtjloco_inv = None
        self._jloco_u = None
        self._jloco_s = None
        self._jloco_v = None

        self._dip_k_inival = None
        self._dip_k_deltas = None
        self._quad_k_inival = None
        self._quad_k_deltas = None
        self._sext_k_inival = None
        self._sext_k_deltas = None

        self._dip_ks_inival = None
        self._dip_ks_deltas = None
        self._quad_ks_inival = None
        self._quad_ks_deltas = None
        self._sext_ks_inival = None
        self._sext_ks_deltas = None

        self._dip_kick_inival = None
        self._dip_kick_deltas = None

        self._energy_shift_inival = None
        self._energy_shift_deltas = None

        self._gain_bpm_inival = self.config.gain_bpm
        self._gain_bpm_delta = None
        self._roll_bpm_inival = self.config.roll_bpm
        self._roll_bpm_delta = None
        self._gain_corr_inival = self.config.gain_corr
        self._gain_corr_delta = None

        self._chi_init = None
        self._chi = None
        self._chi_history = []
        self._tol = None
        self._reduc_threshold = None

        self.fitmodel = None
        self.chi_history = None
        self.bpm_gain = None
        self.bpm_roll = None
        self.corr_gain = None
        self.energy_shift = None

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

    def _handle_dip_fit_k(self, fname_jloco_k_dip):
        # calculate K jacobian for dipole
        if self.config.fit_dipoles:
            if fname_jloco_k_dip is None:
                print('calculating dipoles kmatrix...')
                self._jloco_k_dip = LOCOUtils.jloco_calc_k_dipoles(
                    self.config, self._model)
            else:
                print('loading dipole kmatrix...')
                jloco_k_dip_dict = LOCOUtils.load_data(
                    fname_jloco_k_dip)
                self._jloco_k_dip = self._convert_dict2array(
                    jloco_k_dip_dict, 'dipole')

    def _handle_quad_fit_k(self, fname_jloco_k_quad):
        # calculate K jacobian for quadrupole
        if self.config.fit_quadrupoles:
            if fname_jloco_k_quad is None:
                print('calculating quadrupoles kmatrix...')
                t0 = time.time()
                self._jloco_k_quad = LOCOUtils.jloco_calc_k_quad(
                    self.config, self._model)
                dt = time.time() - t0
                print('it took {} min to calculate'.format(dt/60))
            else:
                print('loading quadrupoles kmatrix...')
                jloco_k_quad_dict = LOCOUtils.load_data(
                    fname_jloco_k_quad)
                self._jloco_k_quad = self._convert_dict2array(
                    jloco_k_quad_dict, 'quadrupole')

    def _convert_dict2array(self, jlocodict, magtype, is_normal=True):
        jloco = []
        if magtype == 'dipole':
            if self.config.dipoles_to_fit is not None:
                magstofit = self.config.dipoles_to_fit
            else:
                magstofit = self.config.famname_dipset
        elif magtype == 'quadrupole':
            if self.config.quadrupoles_to_fit is not None:
                magstofit = self.config.quadrupoles_to_fit
            else:
                magstofit = self.config.famname_quadset
        elif magtype == 'sextupole':
            if self.config.sextupoles_to_fit is not None:
                magstofit = self.config.sextupoles_to_fit
            else:
                magstofit = self.config.famname_sextset
        quadfam = self.config.use_quad_families
        quadfam &= magtype == 'quadrupole'
        quadfam &= is_normal
        dipfam = self.config.use_dip_families
        dipfam &= magtype == 'dipole'
        dipfam &= is_normal
        if dipfam or quadfam:
            for quad in magstofit:
                famcols = [
                    val for key, val in jlocodict.items() if quad in key]
                jloco.append(sum(famcols))
        else:
            for key in jlocodict:
                key = _PVName(key)
                if key.dev in magstofit:
                    jloco.append(jlocodict[key])
        return np.array(jloco).T

    def _handle_sext_fit_k(self, fname_jloco_k_sext):
        # calculate K jacobian for sextupole
        if self.config.fit_sextupoles:
            if fname_jloco_k_sext is None:
                print('calculating sextupoles kmatrix...')
                t0 = time.time()
                self._jloco_k_sext = LOCOUtils.jloco_calc_k_sextupoles(
                    self.config, self._model)
                dt = time.time() - t0
                print('it took {} min to calculate'.format(dt/60))
            else:
                print('loading sextupoles kmatrix...')
                jloco_k_sext_dict = LOCOUtils.load_data(
                    fname_jloco_k_sext)
                self._jloco_k_sext = self._convert_dict2array(
                    jloco_k_sext_dict, 'sextupole')

    def _handle_dip_fit_ks(self, fname_jloco_ks_dip):
        # calculate Ks jacobian for dipole
        if self.config.fit_dipoles_coupling:
            if fname_jloco_ks_dip is None:
                print('calculating dipoles ksmatrix...')
                self._jloco_ks_dip = LOCOUtils.jloco_calc_ks_dipoles(
                    self.config, self._model)
            else:
                print('loading dipole ksmatrix...')
                jloco_ks_dip_dict = LOCOUtils.load_data(
                    fname_jloco_ks_dip)
                self._jloco_ks_dip = self._convert_dict2array(
                    jloco_ks_dip_dict, 'dipole', is_normal=False)

    def _handle_quad_fit_ks(self, fname_jloco_ks_quad):
        # calculate Ks jacobian for quadrupole
        if self.config.fit_quadrupoles_coupling:
            if fname_jloco_ks_quad is None:
                print('calculating quadrupoles ksmatrix...')
                t0 = time.time()
                self._jloco_ks_quad = LOCOUtils.jloco_calc_ks_quad(
                    self.config, self._model)
                dt = time.time() - t0
                print('it took {} min to calculate'.format(dt/60))
            else:
                print('loading quadrupoles ksmatrix...')
                jloco_ks_quad_dict = LOCOUtils.load_data(
                    fname_jloco_ks_quad)
                self._jloco_ks_quad = self._convert_dict2array(
                    jloco_ks_quad_dict, 'quadrupole', is_normal=False)

    def _handle_sext_fit_ks(self, fname_jloco_ks_sext):
        # calculate Ks jacobian for sextupole
        if self.config.fit_sextupoles_coupling:
            if fname_jloco_ks_sext is None:
                print('calculating sextupoles ksmatrix...')
                t0 = time.time()
                self._jloco_ks_sext = LOCOUtils.jloco_calc_ks_sextupoles(
                    self.config, self._model)
                dt = time.time() - t0
                print('it took {} min to calculate'.format(dt/60))
            else:
                print('loading sextupoles ksmatrix...')
                jloco_ks_sext_dict = LOCOUtils.load_data(
                    fname_jloco_ks_sext)
                self._jloco_ks_sext = self._convert_dict2array(
                    jloco_ks_sext_dict, 'sextupole', is_normal=False)

    def _handle_dip_fit_kick(self, fname_jloco_kick_dip=None):
        # calculate kick jacobian for dipole
        if self.config.fit_dipoles_kick:
            if fname_jloco_kick_dip is None:
                print('calculating dipoles kick matrix...')
                self._jloco_kick_dip = LOCOUtils.jloco_calc_kick_dipoles(
                    self.config, self._model)
            else:
                print('loading dipole kick matrix...')
                self._jloco_kick_dip = LOCOUtils.load_data(
                    fname_jloco_kick_dip)['jloco_kmatrix']

    def create_new_jacobian_dict(self, jloco, idx, sub):
        """."""
        newjloco = dict()
        for num, ind in enumerate(idx):
            name = 'SI-'
            name += sub[num]
            name += ':PS-'
            name += self._model[ind[0]].fam_name
            newjloco[name] = jloco[:, num]
        return newjloco

    def save_jacobian(self):
        idxQN = self.config.respm.fam_data['QN']['index']
        subQN = self.config.respm.fam_data['QN']['subsection']

        idxSN = self.config.respm.fam_data['SN']['index']
        subSN = self.config.respm.fam_data['SN']['subsection']

        idxBN = self.config.respm.fam_data['BN']['index']
        subBN = self.config.respm.fam_data['BN']['subsection']

        jloco_k_dip = self.create_new_jacobian_dict(
            self._jloco_k_dip, idxBN, subBN)
        jloco_k_quad = self.create_new_jacobian_dict(
            self._jloco_k_quad, idxQN, subQN)
        jloco_k_sext = self.create_new_jacobian_dict(
            self._jloco_k_sext, idxSN, subSN)

        print('saving jacobian K')
        LOCOUtils.save_data(
            '4d_KL_dipoles', jloco_k_dip)
        LOCOUtils.save_data(
            '4d_KL_quadrupoles_trims', jloco_k_quad)
        LOCOUtils.save_data(
            '4d_KL_sextupoles', jloco_k_sext)

        jloco_ks_dip = self.create_new_jacobian_dict(
            self._jloco_ks_dip, idxBN, subBN)
        jloco_ks_quad = self.create_new_jacobian_dict(
            self._jloco_ks_quad, idxQN, subQN)
        jloco_ks_sext = self.create_new_jacobian_dict(
            self._jloco_ks_sext, idxSN, subSN)

        print('saving jacobian Ks')
        LOCOUtils.save_data(
            '4d_KsL_dipoles', jloco_ks_dip)
        LOCOUtils.save_data(
            '4d_KsL_quadrupoles_trims', jloco_ks_quad)
        LOCOUtils.save_data(
            '4d_KsL_sextupoles', jloco_ks_sext)

    def update_jloco(self,
                     fname_jloco_k=None,
                     fname_jloco_k_dip=None,
                     fname_jloco_k_quad=None,
                     fname_jloco_k_sext=None,
                     fname_jloco_ks_dip=None,
                     fname_jloco_ks_quad=None,
                     fname_jloco_ks_sext=None,
                     fname_jloco_kick_dip=None):
        """."""
        # calc jloco linear parts
        self._jloco_gain_bpm, self._jloco_roll_bpm, self._jloco_gain_corr = \
            LOCOUtils.jloco_calc_linear(self.config, self._matrix)

        if fname_jloco_k is not None:
            self._jloco_k = LOCOUtils.load_data(fname_jloco_k)['jloco_kmatrix']
        else:
            self._handle_dip_fit_kick(fname_jloco_kick_dip)

            self._handle_dip_fit_k(fname_jloco_k_dip)
            self._handle_quad_fit_k(fname_jloco_k_quad)
            self._handle_sext_fit_k(fname_jloco_k_sext)

            self._handle_dip_fit_ks(fname_jloco_ks_dip)
            self._handle_quad_fit_ks(fname_jloco_ks_quad)
            self._handle_sext_fit_ks(fname_jloco_ks_sext)

            # self.save_jacobian()

        if self.config.fit_energy_shift:
            print('calculating energy shift matrix...')
            self._jloco_energy_shift = LOCOUtils.jloco_calc_energy_shift(
                self.config, self._model)
        else:
            self._jloco_energy_shift = np.zeros((
                self._matrix.size, self.config.nr_corr))

        # merge J submatrices
        self._jloco = LOCOUtils.jloco_merge_linear(
            self.config,
            self._jloco_k_quad, self._jloco_k_sext, self._jloco_k_dip,
            self._jloco_ks_quad, self._jloco_ks_sext, self._jloco_ks_dip,
            self._jloco_gain_bpm, self._jloco_roll_bpm,
            self._jloco_gain_corr, self._jloco_kick_dip,
            self._jloco_energy_shift)

        # apply weight
        self._jloco = LOCOUtils.jloco_apply_weight(
            self._jloco, self.config.weight_bpm, self.config.weight_corr)

        if not self.config.use_dispersion:
            jloco_temp = np.reshape(
                self._jloco, (2*self.config.nr_bpm, self.config.nr_corr+1, -1))
            jloco_temp[:, -1, :] *= 0

        if self.config.constraint_deltak:
            jloco_deltak = self.calc_jloco_deltak_constraint()
            self._jloco = np.vstack((self._jloco, jloco_deltak))

        # calc jloco inv
        if LOCO.JLOCO_INVERSION == 'transpose':
            print('svd decomposition Jt * J')
            if self.config.fitting_method == 'Gauss-Newton':
                matrix2invert = self._jloco.T @ self._jloco
            elif self.config.fitting_method == 'Levenberg-Marquadt':
                D = self.calc_d_matrix()
                matrix2invert = self._jloco.T @ self._jloco
                matrix2invert += self.config.lambda_lm * D.T @ D
            self._jtjloco_u, self._jtjloco_s, self._jtjloco_v = \
                np.linalg.svd(matrix2invert, full_matrices=False)
        elif LOCO.JLOCO_INVERSION == 'normal':
            print('svd decomposition J')
            self._jloco_u, self._jloco_s, self._jloco_v = \
                np.linalg.svd(self._jloco, full_matrices=False)

    def calc_d_matrix(self):
        """."""
        ncols = self._jloco.shape[1]
        dmat = np.zeros(ncols)
        for col in range(ncols):
            dmat[col] = np.linalg.norm(self._jloco[:, col])
        return np.diag(dmat)

    def calc_jloco_deltak_constraint(self):
        """."""
        sigma_deltak = LOCO.DEFAULT_DELTAK_NORMALIZATION
        ncols = self._jloco.shape[1]
        nknobs = len(self.config.quad_indices)
        deltak_mat = np.zeros((nknobs, ncols))
        for knb in range(nknobs):
            deltak_mat[knb, knb] = self.config.weight_deltak[knb]/sigma_deltak
        return deltak_mat

    def update_svd(self, svd_thre=None, svd_sel=None):
        """."""
        if LOCO.JLOCO_INVERSION == 'transpose':
            u, s, v = self._jtjloco_u, self._jtjloco_s, self._jtjloco_v
        elif LOCO.JLOCO_INVERSION == 'normal':
            u, s, v = self._jloco_u, self._jloco_s, self._jloco_v

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

        if LOCO.JLOCO_INVERSION == 'transpose':
            self._jtjloco_inv = np.dot(v.T * inv_s[None, :], u.T)
        elif LOCO.JLOCO_INVERSION == 'normal':
            self._jloco_inv = np.dot(v.T * inv_s[None, :], u.T)

    def update_fit(self):
        """."""
        # k inival and deltas
        if self.config.use_quad_families:
            self._quad_k_inival = LOCOUtils.get_quads_strengths(
                model=self._model, indices=self.config.quad_indices)
        else:
            self._quad_k_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'KL', self.config.quad_indices))

        if self.config.use_dip_families:
            self._dip_k_inival = LOCOUtils.get_quads_strengths(
                model=self._model, indices=self.config.dip_indices)
        else:
            self._dip_k_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'KL', self.config.dip_indices))

        self._sext_k_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'KL', self.config.sext_indices))

        if self.config.use_coupling:
            self._quad_ks_inival = np.array(
                    pyaccel.lattice.get_attribute(
                        self._model, 'KsL', self.config.quad_indices_ks))
            self._sext_ks_inival = np.array(
                    pyaccel.lattice.get_attribute(
                        self._model, 'KsL', self.config.sext_indices))
            self._dip_ks_inival = np.array(
                    pyaccel.lattice.get_attribute(
                        self._model, 'KsL', self.config.dip_indices_ks))
            self._quad_ks_deltas = np.zeros(len(self.config.quad_indices_ks))
            self._sext_ks_deltas = np.zeros(len(self.config.sext_indices))
            self._dip_ks_deltas = np.zeros(len(self.config.dip_indices_ks))

        # self._dip_kick_inival = np.array(
        #         pyaccel.lattice.get_attribute(
        #             self._model, 'hkick_polynom', self.config.dip_indices_ks))

        self._energy_shift_inival = np.zeros(self.config.nr_corr)
        self._energy_shift_deltas = np.zeros(self.config.nr_corr)

        self._quad_k_deltas = np.zeros(len(self.config.quad_indices))
        self._sext_k_deltas = np.zeros(len(self.config.sext_indices))
        self._dip_k_deltas = np.zeros(len(self.config.dip_indices))

        # self._dip_kick_deltas = np.zeros(len(self.config.dip_indices_ks))

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

    def _calc_residue(self):
        matrix_diff = self.config.goalmat - self._matrix
        matrix_diff = LOCOUtils.apply_all_weight(
            matrix_diff, self.config.weight_bpm, self.config.weight_corr)
        res = matrix_diff.flatten()
        if self.config.constraint_deltak:
            res = np.hstack((res, self._quad_k_deltas))
        return res

    def run_fit(self, niter=1):
        """."""
        self._chi = self._chi_init
        for _iter in range(niter):
            self._chi_history.append(self._chi)
            print('iter # {}/{}'.format(_iter+1, niter))
            res = self._calc_residue()
            if LOCO.JLOCO_INVERSION == 'transpose':
                param_new = np.dot(
                    self._jtjloco_inv, np.dot(
                        self._jloco.T, res))
            elif LOCO.JLOCO_INVERSION == 'normal':
                param_new = np.dot(self._jloco_inv, res)
            param_new = param_new.flatten()
            model_new, matrix_new = self._calc_model_matrix(param_new)
            chi_new = self.calc_chi(matrix_new)
            print('chi: {0:.4f} um'.format(chi_new))
            if np.isnan(chi_new):
                print('chi is NaN!')
                break
            if chi_new < self._chi:
                if np.abs(chi_new - self._chi) < self._tol:
                    print('chi reduction is lower than tolerance...')
                    break
                else:
                    self._update_state(model_new, matrix_new, chi_new)
            else:
                # print('recalculating jloco...')
                # self.update_jloco()
                # self.update_svd()
                if self.config.fitting_method == 'Gauss-Newton':
                    factor = \
                        self._try_refactor_param(param_new)
                    if factor <= self._reduc_threshold:
                        # could not converge at current iteration!
                        break
                elif self.config.fitting_method == 'Levenberg-Marquadt':
                    self.config.lambda_lm = self._try_refactor_lambda(chi_new)
                    if self.config.lambda_lm >= 1e2:
                        break
            if self._chi < self._tol:
                print('chi is lower than specified tolerance!')
                break
        self._create_output_vars()
        print('Finished!')

    def calc_chi(self, matrix=None):
        """."""
        if matrix is None:
            matrix = self._matrix
        dmatrix = matrix - self.config.goalmat
        dmatrix = LOCOUtils.apply_all_weight(
                dmatrix, self.config.weight_bpm, self.config.weight_corr)
        dmatrix[:, :self.config.nr_ch] *= self.config.delta_kickx_meas
        dmatrix[:, self.config.nr_ch:-1] *= self.config.delta_kicky_meas
        dmatrix[:, -1] *= self.config.delta_frequency_meas
        chi = np.linalg.norm(dmatrix)/np.sqrt(dmatrix.size)
        # if self.config.constraint_deltak:
        #     deltak_term = self.config.weight_deltak @ self._quad_k_deltas
        #     deltak_term /= LOCO.DEFAULT_DELTAK_NORMALIZATION
        #     chi += deltak_term
        return chi * 1e6

    def _create_output_vars(self):
        """."""
        self.fitmodel = _dcopy(self._model)
        self.chi_history = self._chi_history
        self.bpm_gain = self._gain_bpm_inival + self._gain_bpm_delta
        self.bpm_roll = self._roll_bpm_inival + self._roll_bpm_delta
        self.corr_gain = self._gain_corr_inival + self._gain_corr_delta
        self.energy_shift = self._energy_shift_inival + \
            self._energy_shift_deltas

    def _calc_model_matrix(self, param):
        """."""
        model = _dcopy(self._model)
        config = self.config
        param_dict = LOCOUtils.param_select(config, param)
        param_names = {
            'dipoles_gradient',
            'quadrupoles_gradient',
            'sextupoles_gradient'}

        if bool(param_names.intersection(set(param_dict.keys()))):
            if 'dipoles_gradient' in param_dict:
                self._dip_k_deltas += param_dict['dipoles_gradient']
                # update local model
                if self.config.use_dip_families:
                    set_dip_kdelta = LOCOUtils.set_dipset_kdelta
                else:
                    set_dip_kdelta = LOCOUtils.set_dipmag_kdelta
                for idx, idx_set in enumerate(config.dip_indices):
                    set_dip_kdelta(
                        model, idx_set,
                        self._dip_k_inival[idx], self._dip_k_deltas[idx])
            if 'quadrupoles_gradient' in param_dict:
                # update quadrupole delta
                self._quad_k_deltas += param_dict['quadrupoles_gradient']
                # update local model
                if self.config.use_quad_families:
                    set_quad_kdelta = LOCOUtils.set_quadset_kdelta
                else:
                    set_quad_kdelta = LOCOUtils.set_quadmag_kdelta
                for idx, idx_set in enumerate(config.quad_indices):
                    set_quad_kdelta(
                        model, idx_set,
                        self._quad_k_inival[idx], self._quad_k_deltas[idx])
            if 'sextupoles_gradient' in param_dict:
                # update sextupole delta
                self._sext_k_deltas += param_dict['sextupoles_gradient']
                # update local model
                set_quad_kdelta = LOCOUtils.set_quadmag_kdelta
                for idx, idx_set in enumerate(config.sext_indices):
                    set_quad_kdelta(
                        model, idx_set,
                        self._sext_k_inival[idx], self._sext_k_deltas[idx])
            if 'quadrupoles_coupling' in param_dict:
                # update quadrupole Ks delta
                self._quad_ks_deltas += param_dict['quadrupoles_coupling']
                # update local model
                set_quad_ksdelta = LOCOUtils.set_quadmag_ksdelta
                for idx, idx_set in enumerate(config.quad_indices_ks):
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
            if 'dipoles_coupling' in param_dict:
                # update dipoles Ks delta
                self._dip_ks_deltas += param_dict['dipoles_coupling']
                # update local model
                for idx, idx_set in enumerate(config.dip_indices_ks):
                    LOCOUtils.set_dipmag_ksdelta(
                        model, idx_set,
                        self._dip_ks_inival[idx], self._dip_ks_deltas[idx])
            if 'dipoles_kicks' in param_dict:
                # update dipoles kick delta
                self._dip_kick_deltas += param_dict['dipoles_kick']
                # update local model
                for idx, idx_set in enumerate(config.dip_indices):
                    LOCOUtils.set_dipmag_kick(
                        model, idx_set,
                        self._dip_kick_inival[idx], self._dip_kick_deltas[idx])
            matrix = LOCOUtils.respm_calc(
                model, config.respm, config.use_dispersion)
            if 'energy_shift' in param_dict:
                # update energy shift
                self._energy_shift_deltas += param_dict['energy_shift']
                matrix = LOCOUtils.add_dispersion_to_respm(
                    matrix,
                    self._energy_shift_inival + self._energy_shift_deltas,
                    config.measured_dispersion)
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

    def _try_refactor_lambda(self, chi_new):
        """."""
        lambda_lm = _dcopy(self.config.lambda_lm)
        _iter = 0
        while chi_new > self._chi and lambda_lm < 1e2:
            lambda_lm *= 10
            print('chi was increased! Trial {0:d}'.format(_iter))
            print('applying lambda {0:0.4f}'.format(lambda_lm))
            dmat = self.calc_d_matrix()
            matrix2invert = self._jloco.T @ self._jloco
            matrix2invert += lambda_lm * dmat.T @ dmat
            self._jtjloco_u, self._jtjloco_s, self._jtjloco_v = \
                np.linalg.svd(matrix2invert, full_matrices=False)
            inv_s = 1/self._jtjloco_s
            inv_s[np.isnan(inv_s)] = 0
            inv_s[np.isinf(inv_s)] = 0

            if self.config.svd_method == self.config.SVD_METHOD_THRESHOLD:
                bad_sv = self._jtjloco_s/np.max(self._jtjloco_s) < \
                    self.config.svd_thre
                inv_s[bad_sv] = 0
            elif self.config.svd_method == self.config.SVD_METHOD_SELECTION:
                inv_s[self.config.svd_sel:] = 0
            self._jtjloco_inv = np.dot(
                self._jtjloco_v.T * inv_s[None, :], self._jtjloco_u.T)
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
            if chi_new < self._chi:
                self._update_state(model_new, matrix_new, chi_new)
                break
            _iter += 1
        return lambda_lm

    def _update_state(self, model_new, matrix_new, chi_new):
        """."""
        self._model = model_new
        self._matrix = matrix_new
        self._chi = chi_new
