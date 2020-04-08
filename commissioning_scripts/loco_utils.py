"""."""

from copy import deepcopy as _dcopy
import pickle as _pickle
import numpy as np
import pyaccel


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
        for num in range(shape0):
            kron = LOCOUtils.kronecker(num, num, shape0)
            dbmat = np.dot(r_alpha, kron)
            dmdg_bpm[:, num] = np.dot(dbmat, matrix).flatten()

        dmdalpha_bpm = np.zeros((shape0*shape1, nbpm))
        for idx in range(shape0//2):
            kron = LOCOUtils.kronecker(idx, idx, shape0//2)
            kron = np.tile(kron, (2, 2))
            drmat = np.dot(kron, dr_alpha)
            dbmat = drmat * g_bpm[:, None]
            dmdalpha_bpm[:, idx] = np.dot(dbmat, matrix).flatten()

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
        for cnum in range(config.nr_corr):
            energy_shift[cnum] = 1
            matrix_shift = config.measured_dispersion[:, None] * \
                energy_shift[None, :]
            dm_energy_shift[:, cnum] = matrix_shift.flatten()
            energy_shift[cnum] = 0
        return dm_energy_shift

    @staticmethod
    def jloco_calc_ks_skewquad(config, model):
        """."""
        qsindices = config.respm.fam_data['QS']['index']
        ksvalues = np.array(
            pyaccel.lattice.get_attribute(model, 'KsL', qsindices))
        set_quad_ksdelta = LOCOUtils.set_quadmag_ksdelta
        matrix_nominal = LOCOUtils.respm_calc(
            model, config.respm, config.use_dispersion)

        ksmatrix = np.zeros((
            matrix_nominal.shape[0]*matrix_nominal.shape[1], len(qsindices)))

        model_this = _dcopy(model)
        for idx, idx_set in enumerate(qsindices):
            set_quad_ksdelta(
                model_this, idx_set, ksvalues[idx], config.DEFAULT_DELTA_KS)
            matrix_this = LOCOUtils.respm_calc(
                model_this, config.respm, config.use_dispersion)
            dmatrix = (matrix_this - matrix_nominal)/config.DEFAULT_DELTA_KS
            ksmatrix[:, idx] = dmatrix.flatten()
            set_quad_ksdelta(model_this, idx_set, ksvalues[idx], 0)
        return ksmatrix

    @staticmethod
    def jloco_merge_linear(
            config, km_quad, km_sext, km_dip,
            ksm_quad, ksm_sext, ksm_dip,
            dmdg_bpm, dmdalpha_bpm, dmdg_corr,
            kick_dip, energy_shift, ks_skewquad):
        """."""
        nbpm = config.nr_bpm
        nch = config.nr_ch
        ncv = config.nr_cv
        knobs_k = 0
        knobs_ks = 0
        knobs_linear = 0
        knobs_skewquad = 0

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
        if ks_skewquad is not None:
            knobs_skewquad += ks_skewquad.shape[1]

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

        nknobs = knobs_k + knobs_ks + knobs_linear + knobs_skewquad
        jloco = np.zeros(
            (2*nbpm*(nch+ncv+1), nknobs))
        idx = 0
        if config.fit_quadrupoles:
            num = km_quad.shape[1]
            jloco[:, idx:idx+num] = km_quad
            idx += num
        if config.fit_sextupoles:
            num = km_sext.shape[1]
            jloco[:, idx:idx+num] = km_sext
            idx += num
        if config.fit_dipoles:
            num = km_dip.shape[1]
            jloco[:, idx:idx+num] = km_dip
            idx += num
        if config.fit_quadrupoles_coupling:
            num = ksm_quad.shape[1]
            jloco[:, idx:idx+num] = ksm_quad
            idx += num
        if config.fit_sextupoles_coupling:
            num = ksm_sext.shape[1]
            jloco[:, idx:idx+num] = ksm_sext
            idx += num
        if config.fit_dipoles_coupling:
            num = ksm_dip.shape[1]
            jloco[:, idx:idx+num] = ksm_dip
            idx += num
        if config.fit_gain_bpm:
            num = dmdg_bpm.shape[1]
            jloco[:, idx:idx+num] = dmdg_bpm
            idx += num
        if config.fit_roll_bpm:
            num = dmdalpha_bpm.shape[1]
            jloco[:, idx:idx+num] = dmdalpha_bpm
            idx += num
        if config.fit_gain_corr:
            num = dmdg_corr.shape[1]
            jloco[:, idx:idx+num] = dmdg_corr
            idx += num
        if config.fit_dipoles_kick:
            num = kick_dip.shape[1]
            jloco[:, idx:idx+num] = kick_dip
            idx += num
        if config.fit_energy_shift:
            num = energy_shift.shape[1]
            jloco[:, idx:idx+num] = energy_shift
            idx += num
        if config.fit_skew_quadrupoles:
            num = knobs_skewquad
            jloco[:, idx:idx+num] = ks_skewquad
            idx += num
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
        if config.fit_skew_quadrupoles:
            size = len(config.skew_quad_indices)
            param_dict['skew_quadrupoles'] = param[idx:idx+size]
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
