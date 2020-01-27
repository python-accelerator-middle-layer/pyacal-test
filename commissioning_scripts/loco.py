#!/usr/bin/env python-sirius
"""."""

from copy import deepcopy as _dcopy
import numpy as np
import pyaccel
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat


class LOCOUtils:
    """LOCO utils."""

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
        return np.dot(np.diag(gain), matrix)

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
        matrix[:, :-1] = np.dot(matrix[:, :-1], np.diag(gain))
        return matrix

    @staticmethod
    def apply_all_gain(matrix, gain_bpm, roll_bpm, gain_corr):
        """."""
        matrix = LOCOUtils.apply_bpm_gain(matrix, gain_bpm)
        matrix = LOCOUtils.apply_bpm_roll(matrix, roll_bpm)
        matrix = LOCOUtils.apply_corr_gain(matrix, gain_corr)
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
    def get_quads_strengths(model, k_indices):
        """."""
        kquads = []
        for qidx in k_indices:
            kquads.append(pyaccel.lattice.get_attribute(
                model, 'K', qidx))
        return kquads

    @staticmethod
    def set_quadmag_kdelta(model, idx_mag, kvalues, kdelta):
        """."""
        for idx, idx_seg in enumerate(idx_mag):
            pyaccel.lattice.set_attribute(
                model, 'K', idx_seg, kvalues[idx] +
                kdelta)

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
            deltaB = np.dot(deltaR, np.diag(g_bpm))
            dmdalpha_bpm[:, idx] = np.dot(deltaB, matrix).flatten()

        dmdg_corr = np.zeros((shape0*shape1, ncorr))
        for idx in range(ncorr):
            kron = LOCOUtils.kronecker(idx, idx, shape1)
            dmdg_corr[:, idx] = np.dot(matrix, kron).flatten()

        return dmdg_bpm, dmdalpha_bpm, dmdg_corr

    @staticmethod
    def jloco_calc_k(config, model):
        """."""
        if config.use_families:
            kindices = []
            for fam_name in config.famname_quadset:
                kindices.append(config.respm.fam_data[fam_name]['index'])
            kvalues = LOCOUtils.get_quads_strengths(
                config.respm.model, kindices)
            set_quad_kdelta = LOCOUtils.set_quadset_kdelta
        else:
            kindices = config.respm.fam_data['QN']['index']
            kvalues = np.array(
                pyaccel.lattice.get_attribute(
                    config.respm.model, 'K', kindices))
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
    def jloco_merge_linear(
            config, kmatrix, dmdg_bpm, dmdalpha_bpm, dmdg_corr):
        """."""
        nbpm = config.nr_bpm
        nch = config.nr_ch
        ncv = config.nr_cv
        nfam = kmatrix.shape[1]
        jloco = np.zeros((kmatrix.shape[0], nfam + 3*nbpm + nch + ncv))
        jloco[:, :nfam] = kmatrix
        jloco[:, nfam:nfam+2*nbpm] = dmdg_bpm
        jloco[:, nfam+2*nbpm:nfam+3*nbpm] = dmdalpha_bpm
        jloco[:, nfam+3*nbpm:nfam+3*nbpm+nch+ncv] = dmdg_corr
        return jloco

    @staticmethod
    def jloco_param_delete(config, jloco):
        """."""
        idx = 0
        k_nrsets = len(config.k_indices)
        if not config.fit_quadrupoles:
            jloco = np.delete(jloco, slice(idx, idx + k_nrsets), axis=1)
            print('removing quadrupoles...')
        else:
            idx += k_nrsets
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
        return jloco

    @staticmethod
    def param_select(config, param):
        """."""
        idx = 0
        param_dict = dict()
        if config.fit_quadrupoles:
            size = len(config.k_indices)
            param_dict['k'] = param[idx:idx+size]
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

    @staticmethod
    def chi2_calc(matrix1, matrix2):
        """."""
        dmatrix = matrix1 - matrix2
        # chi2 = np.mean(dmatrix**2)
        chi2 = np.linalg.norm(dmatrix)**2/dmatrix.size
        return chi2

class LOCOConfig:
    """SI LOCO configuration."""

    SVD_METHOD_SELECTION = 0
    SVD_METHOD_THRESHOLD = 1

    DEFAULT_DELTA_K = 1e-6  # [1/m^2]
    DEFAULT_DELTA_RF = 100  # [Hz]
    DEFAULT_SVD_THRESHOLD = 1e-6

    FAMNAME_RF = 'SRFCav'

    def __init__(self, **kwargs):
        """."""
        self.model = None
        self.dim = None
        self.respm = None
        self.goalmat = None
        self.use_disp = None
        self.use_coupling = None
        self.rf_freq = None
        self.use_families = None
        self.svd_method = None
        self.svd_sel = None
        self.svd_thre = None
        self.fit_quadrupoles = None
        self.fit_gain_bpm = None
        self.fit_gain_corr = None
        self.cavidx = None
        self.matrix = None
        self.idx_cav = None
        self.idx_bpm = None
        self.gain_bpm = None
        self.gain_corr = None
        self.roll_bpm = None
        self.roll_corr = None
        self.vector = None
        self.k_indices = None
        self.k_nrsets = None

        self._process_input(kwargs)

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
        self.update_rf(self.rf_freq)
        self.update_gain()
        self.update_quad_knobs(self.use_families)
        self.update_svd(self.svd_method, self.svd_sel, self.svd_thre)

    def update_model(self, model, dim):
        """."""
        self.dim = dim
        self.model = _dcopy(model)
        if not self.model.cavity_on and dim == '6d':
            self.model.cavity_on = True
        if not model.radiation_on:
            self.model.radiation_on = True
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

    def update_rf(self, rf_freq=None):
        """."""
        self.cavidx = pyaccel.lattice.find_indices(
            self.model, 'fam_name', self.FAMNAME_RF)[0]
        if rf_freq is None:
            self.rf_freq = self.model[self.cavidx].frequency
        else:
            self.rf_freq = rf_freq

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
                self.gain_bpm = gain_bpm
        if roll_bpm is None:
            if self.roll_bpm is None:
                self.roll_bpm = np.zeros(self.nr_bpm)
        else:
            if isinstance(roll_bpm, (int, float)):
                self.roll_bpm = np.ones(self.nr_bpm) * roll_bpm
            else:
                self.roll_bpm = roll_bpm
        # corr
        if gain_corr is None:
            if self.gain_corr is None:
                self.gain_corr = np.ones(self.nr_corr)
        else:
            if isinstance(gain_corr, (int, float)):
                self.gain_bpm = np.ones(self.nr_corr) * gain_corr
            else:
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

    def update_quad_knobs(self, use_families):
        """."""
        self.use_families = use_families
        if use_families:
            self.k_indices = [None] * len(self.famname_quadset)
            for idx, fam_name in enumerate(self.famname_quadset):
                self.k_indices[idx] = self.respm.fam_data[fam_name]['index']
        else:
            self.k_indices = self.respm.fam_data['QN']['index']

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
    def famname_quadset(self):
        """."""
        return [
            'QFA', 'QDA', 'QDB2', 'QFB', 'QDB1', 'QDP2', 'QFP', 'QDP1',
            'Q1', 'Q2', 'Q3', 'Q4']


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
    def famname_quadset(self):
        """."""
        return ['QF', 'QD']


class LOCO:
    """LOCO."""

    UTILS = LOCOUtils
    DEFAULT_TOL = 1e-16
    DEFAULT_REDUC_THRESHOLD = 0.05/100

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
        self._jloco = None
        self._jloco_u = None
        self._jloco_s = None
        self._jloco_v = None
        self._jloco_inv = None

        self._k_inival = None
        self._k_deltas = None
        self._gain_bpm_inival = None
        self._gain_bpm_delta = None
        self._roll_bpm_inival = None
        self._roll_bpm_delta = None
        self._gain_corr_inival = None
        self._gain_corr_delta = None

        self._chi2_init = None
        self._chi2 = None
        self._tol = None
        self._reduc_threshold = None

    def update(self, fname_jloco_k=None):
        """."""
        print('update config...')
        self.update_config()
        print('update jloco...')
        self.update_jloco(fname_jloco_k)
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

    def update_jloco(self, fname_jloco_k):
        """."""
        # calc jloco linear parts
        self._jloco_gain_bpm, self._jloco_roll_bpm, self._jloco_gain_corr = \
            LOCOUtils.jloco_calc_linear(self.config, self._matrix)

        # calc jloco K part
        if fname_jloco_k is None:
            self._jloco_k = LOCOUtils.jloco_calc_k(self.config, self._model)
        else:
            self._jloco_k = np.loadtxt(fname_jloco_k)

        # merge J submatrices
        self._jloco = LOCOUtils.jloco_merge_linear(
            self.config, self._jloco_k,
            self._jloco_gain_bpm, self._jloco_roll_bpm,
            self._jloco_gain_corr)

        # filter jloco
        self._jloco = LOCOUtils.jloco_param_delete(self.config, self._jloco)

        # calc jloco inv
        self._jloco_u, self._jloco_s, self._jloco_v = \
            np.linalg.svd(self._jloco, full_matrices=False)

    def update_svd(self):
        """."""
        u, s, v = self._jloco_u, self._jloco_s, self._jloco_v
        inv_s = 1/s
        inv_s[np.isnan(inv_s)] = 0
        inv_s[np.isinf(inv_s)] = 0
        inv_s = np.diag(inv_s)

        if self.config.svd_method == self.config.SVD_METHOD_THRESHOLD:
            bad_sv = s/np.max(s) < self.config.svd_thre
            print('removing {:d} bad singular values...'.format(
                np.sum(bad_sv)))
            inv_s[bad_sv] = 0
        elif self.config.svd_method == self.config.SVD_METHOD_SELECTION:
            if self.config.svd_sel is not None:
                inv_s[self.config.svd_sel:] = 0

        self._jloco_inv = np.dot(np.dot(v.T, inv_s), u.T)

    def update_fit(self):
        """."""
        # k inival and deltas
        if self.config.use_families:
            self._k_inival = LOCOUtils.get_quads_strengths(
                model=self._model, k_indices=self.config.k_indices)
        else:
            self._k_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'K', self.config.k_indices))

        self._k_deltas = np.zeros(len(self.config.k_indices))

        # bpm inival and deltas
        self._gain_bpm_inival = np.ones(2*self.config.nr_bpm)
        self._roll_bpm_inival = np.zeros(self.config.nr_bpm)
        self._gain_bpm_delta = np.zeros(2*self.config.nr_bpm)
        self._roll_bpm_delta = np.zeros(self.config.nr_bpm)

        # corr inival and deltas
        self._gain_corr_inival = np.ones(self.config.nr_corr)
        self._gain_corr_delta = np.zeros(self.config.nr_corr)

        self._chi2 = self.calc_chi2()
        self._chi2_init = self._chi2
        print('chi2_init: {:.6e}'.format(self._chi2_init))

        self._tol = LOCO.DEFAULT_TOL
        self._reduc_threshold = LOCO.DEFAULT_REDUC_THRESHOLD

    def run_fit(self, niter=1):
        """."""
        self._chi2 = self._chi2_init
        for _iter in range(niter):
            print('iter # {}/{}'.format(_iter+1, niter))

            matrix_diff = self.config.goalmat - self._matrix
            param_new = np.dot(self._jloco_inv, matrix_diff.flatten())
            model_new, matrix_new = self._calc_model_matrix(param_new)
            chi2_new = self.calc_chi2(matrix_new)
            print('chi2: {0:.6e}'.format(chi2_new))
            if np.isnan(chi2_new):
                print('matrix deviation is NaN!')
                break
            if chi2_new < self._chi2:
                self._update_state(model_new, matrix_new, chi2_new)
            else:
                factor = \
                    self._try_refactor_param(param_new)
                if factor <= self._reduc_threshold:
                    # could not converge at current iteration!
                    break
            if self._chi2 < self._tol:
                break

        print('Finished!')

    def calc_chi2(self, matrix=None):
        """."""
        if matrix is None:
            matrix = self._matrix
        chi2 = LOCOUtils.chi2_calc(
            self.config.goalmat, matrix)
        return chi2

    @property
    def jloco_k(self):
        """."""
        if self._jloco_k is None:
            self._jloco_k = LOCOUtils.jloco_calc_k(
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

        if 'k' in param_dict:
            # update delta
            self._k_deltas += param_dict['k']
            # update local model
            if self.config.use_families:
                set_quad_kdelta = LOCOUtils.set_quadset_kdelta
            else:
                set_quad_kdelta = LOCOUtils.set_quadmag_kdelta
            for idx, idx_set in enumerate(config.k_indices):
                set_quad_kdelta(
                    model, idx_set,
                    self._k_inival[idx], self._k_deltas[idx])
            # calc orbrespm
            matrix = LOCOUtils.respm_calc(
                model, config.respm, config.use_disp)
        else:
            matrix = _dcopy(self.config.matrix)

        if 'bpm_gain' in param_dict:
            # update delta
            self._gain_bpm_delta += param_dict['bpm_gain']
            gain = self._gain_bpm_inival + self._gain_bpm_delta
            matrix = LOCOUtils.apply_bpm_gain(matrix, gain)

        if 'bpm_roll' in param_dict:
            # update delta
            self._roll_bpm_delta += param_dict['bpm_roll']
            roll = self._roll_bpm_inival + self._roll_bpm_delta
            matrix = LOCOUtils.apply_bpm_roll(matrix, roll)

        if 'corr_gain' in param_dict:
            # update delta
            self._gain_corr_delta += param_dict['corr_gain']
            gain = self._gain_corr_inival + self._gain_corr_delta
            matrix = LOCOUtils.apply_corr_gain(matrix, gain)

        return model, matrix

    def _try_refactor_param(self, param_new):
        """."""
        factor = 0.5
        _iter = 1
        while factor > self._reduc_threshold:
            print('chi2 was increased! Trial {0:d}'.format(_iter))
            print('applying {0:0.4f} %'.format(100*factor))
            model_new, matrix_new = \
                self._calc_model_matrix(factor*param_new)
            chi2_new = self.calc_chi2(matrix_new)
            print('chi2: {0:.6e}'.format(chi2_new))
            if chi2_new < self._chi2:
                self._update_state(model_new, matrix_new, chi2_new)
                break
            factor /= 2
            _iter += 1
        return factor

    def _update_state(self, model_new, matrix_new, chi2_new):
        """."""
        self._model = model_new
        self._matrix = matrix_new
        self._chi2 = chi2_new
