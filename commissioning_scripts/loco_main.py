"""."""

from copy import deepcopy as _dcopy
import time
import numpy as np
from siriuspy.namesys import SiriusPVName as _PVName
import pyaccel
from apsuite.commissioning_scripts.loco_utils import LOCOUtils
from apsuite.commissioning_scripts.loco_config import LOCOConfig


class LOCO:
    """Main LOCO algorithm."""

    UTILS = LOCOUtils
    DEFAULT_TOL = 1e-3
    DEFAULT_REDUC_THRESHOLD = 5/100
    DEFAULT_LAMBDA_LM = 1e-3
    DEFAULT_DELTAK_NORMALIZATION = 1.0
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
        self._jloco_ks_skew_quad = None

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
        self._skew_quad_ks_inival = None
        self._skew_quad_ks_deltas = None

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
               fname_jloco_kick_dip=None,
               fname_jloco_ks_skewquad=None):
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
                fname_jloco_kick_dip=fname_jloco_kick_dip,
                fname_jloco_ks_skewquad=fname_jloco_ks_skewquad)
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
                time0 = time.time()
                self._jloco_k_quad = LOCOUtils.jloco_calc_k_quad(
                    self.config, self._model)
                dtime = time.time() - time0
                print('it took {} min to calculate'.format(dtime/60))
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
        elif magtype == 'skew_quadrupole':
            magstofit = self.config.famname_skewquadset
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
                time0 = time.time()
                self._jloco_k_sext = LOCOUtils.jloco_calc_k_sextupoles(
                    self.config, self._model)
                dtime = time.time() - time0
                print('it took {} min to calculate'.format(dtime/60))
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

    def _handle_skewquad_fit_ks(self, fname_jloco_ks_skewquad):
        # calculate Ks jacobian for skew quadrupoles
        if self.config.fit_skew_quadrupoles:
            if fname_jloco_ks_skewquad is None:
                print('calculating skew quadrupoles ksmatrix...')
                time0 = time.time()
                self._jloco_ks_skew_quad = LOCOUtils.jloco_calc_ks_skewquad(
                    self.config, self._model)
                dtime = time.time() - time0
                print('it took {} min to calculate'.format(dtime/60))
            else:
                print('loading skew quadrupoles ksmatrix...')
                jloco_ks_skewquad_dict = LOCOUtils.load_data(
                    fname_jloco_ks_skewquad)
                self._jloco_ks_skew_quad = self._convert_dict2array(
                    jloco_ks_skewquad_dict, 'skew_quadrupole', is_normal=False)

    def _handle_quad_fit_ks(self, fname_jloco_ks_quad):
        # calculate Ks jacobian for quadrupole
        if self.config.fit_quadrupoles_coupling:
            if fname_jloco_ks_quad is None:
                print('calculating quadrupoles ksmatrix...')
                time0 = time.time()
                self._jloco_ks_quad = LOCOUtils.jloco_calc_ks_quad(
                    self.config, self._model)
                dtime = time.time() - time0
                print('it took {} min to calculate'.format(dtime/60))
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
                time0 = time.time()
                self._jloco_ks_sext = LOCOUtils.jloco_calc_ks_sextupoles(
                    self.config, self._model)
                dtime = time.time() - time0
                print('it took {} min to calculate'.format(dtime/60))
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
        """."""
        idx_qn = self.config.respm.fam_data['QN']['index']
        sub_qn = self.config.respm.fam_data['QN']['subsection']

        idx_sn = self.config.respm.fam_data['SN']['index']
        sub_sn = self.config.respm.fam_data['SN']['subsection']

        idx_bn = self.config.respm.fam_data['BN']['index']
        sub_bn = self.config.respm.fam_data['BN']['subsection']

        idx_qs = self.config.respm.fam_data['QS']['index']
        sub_qs = self.config.respm.fam_data['QS']['subsection']

        jloco_k_dip = self.create_new_jacobian_dict(
            self._jloco_k_dip, idx_bn, sub_bn)
        jloco_k_quad = self.create_new_jacobian_dict(
            self._jloco_k_quad, idx_qn, sub_qn)
        jloco_k_sext = self.create_new_jacobian_dict(
            self._jloco_k_sext, idx_sn, sub_sn)

        print('saving jacobian K')
        LOCOUtils.save_data(
            '4d_KL_dipoles', jloco_k_dip)
        LOCOUtils.save_data(
            '4d_KL_quadrupoles_trims', jloco_k_quad)
        LOCOUtils.save_data(
            '4d_KL_sextupoles', jloco_k_sext)

        jloco_ks_dip = self.create_new_jacobian_dict(
            self._jloco_ks_dip, idx_bn, sub_bn)
        jloco_ks_quad = self.create_new_jacobian_dict(
            self._jloco_ks_quad, idx_qn, sub_qn)
        jloco_ks_sext = self.create_new_jacobian_dict(
            self._jloco_ks_sext, idx_sn, sub_sn)
        jloco_ks_skewquad = self.create_new_jacobian_dict(
            self._jloco_ks_skew_quad, idx_qs, sub_qs)

        print('saving jacobian Ks')
        LOCOUtils.save_data(
            '4d_KsL_dipoles', jloco_ks_dip)
        LOCOUtils.save_data(
            '4d_KsL_quadrupoles_trims', jloco_ks_quad)
        LOCOUtils.save_data(
            '4d_KsL_sextupoles', jloco_ks_sext)
        LOCOUtils.save_data(
            '4d_KsL_skew_quadrupoles', jloco_ks_skewquad)

    def update_jloco(self,
                     fname_jloco_k=None,
                     fname_jloco_k_dip=None,
                     fname_jloco_k_quad=None,
                     fname_jloco_k_sext=None,
                     fname_jloco_ks_dip=None,
                     fname_jloco_ks_quad=None,
                     fname_jloco_ks_sext=None,
                     fname_jloco_kick_dip=None,
                     fname_jloco_ks_skewquad=None):
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

            self._handle_skewquad_fit_ks(fname_jloco_ks_skewquad)

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
            self._jloco_energy_shift, self._jloco_ks_skew_quad)

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
                dmat = self.calc_d_matrix()
                matrix2invert = self._jloco.T @ self._jloco
                matrix2invert += self.config.lambda_lm * dmat.T @ dmat
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
            umat, smat, vmat = self._jtjloco_u, self._jtjloco_s, \
                self._jtjloco_v
        elif LOCO.JLOCO_INVERSION == 'normal':
            umat, smat, vmat = self._jloco_u, self._jloco_s, self._jloco_v

        ismat = 1/smat
        ismat[np.isnan(ismat)] = 0
        ismat[np.isinf(ismat)] = 0

        if svd_thre is None:
            svd_thre = self.config.svd_thre
        if svd_sel is None:
            svd_sel = self.config.svd_sel

        if self.config.svd_method == self.config.SVD_METHOD_THRESHOLD:
            bad_sv = smat/np.max(smat) < svd_thre
            # print('removing {:d} bad singular values...'.format(
            #     np.sum(bad_sv)))
            ismat[bad_sv] = 0
        elif self.config.svd_method == self.config.SVD_METHOD_SELECTION:
            ismat[svd_sel:] = 0

        if LOCO.JLOCO_INVERSION == 'transpose':
            self._jtjloco_inv = np.dot(vmat.T * ismat[None, :], umat.T)
        elif LOCO.JLOCO_INVERSION == 'normal':
            self._jloco_inv = np.dot(vmat.T * ismat[None, :], umat.T)

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
            self._skew_quad_ks_inival = np.array(
                pyaccel.lattice.get_attribute(
                    self._model, 'KsL', self.config.skew_quad_indices))
            self._quad_ks_deltas = np.zeros(len(self.config.quad_indices_ks))
            self._sext_ks_deltas = np.zeros(len(self.config.sext_indices))
            self._dip_ks_deltas = np.zeros(len(self.config.dip_indices_ks))
            self._skew_quad_ks_deltas = np.zeros(
                len(self.config.skew_quad_indices))

        self._energy_shift_inival = np.zeros(self.config.nr_corr)
        self._energy_shift_deltas = np.zeros(self.config.nr_corr)

        self._quad_k_deltas = np.zeros(len(self.config.quad_indices))
        self._sext_k_deltas = np.zeros(len(self.config.sext_indices))
        self._dip_k_deltas = np.zeros(len(self.config.dip_indices))

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
        chi2 = np.sum(dmatrix*dmatrix)/(dmatrix.size)
        return np.sqrt(chi2) * 1e6

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
            if 'skew_quadrupoles' in param_dict:
                # update skew quadrupoles
                self._skew_quad_ks_deltas += param_dict['skew_quadrupoles']
                # update local model
                set_quad_ksdelta = LOCOUtils.set_quadmag_ksdelta
                for idx, idx_set in enumerate(config.skew_quad_indices):
                    set_quad_ksdelta(
                        model, idx_set,
                        self._skew_quad_ks_inival[idx],
                        self._skew_quad_ks_deltas[idx])
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
