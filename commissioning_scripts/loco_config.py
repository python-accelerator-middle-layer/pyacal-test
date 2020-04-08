"""."""

from copy import deepcopy as _dcopy
import numpy as np
import pyaccel
from apsuite.commissioning_scripts.calc_orbcorr_mat import OrbRespmat
from apsuite.commissioning_scripts.loco_utils import LOCOUtils


class LOCOConfig:
    """LOCO configuration template."""

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
        self.fit_skew_quadrupoles = None
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
        self.skew_quad_indices = None
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
        self.update_skew_quad_knobs()
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
            if svd_sel is not None:
                print(
                    'svd_selection: {:d} values will be used.'.format(
                        self.svd_sel))
            else:
                print('svd_selection: all values will be used.')
        if svd_method == LOCOConfig.SVD_METHOD_THRESHOLD:
            if svd_thre is None:
                self.svd_thre = LOCOConfig.DEFAULT_SVD_THRESHOLD
            print('svd_threshold: {:f}'.format(self.svd_thre))

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

    def update_skew_quad_knobs(self):
        """."""
        self.skew_quad_indices = self.respm.fam_data['QS']['index']

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
    """Sirius Storage Ring LOCO configuration."""

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

    @property
    def famname_skewquadset(self):
        """."""
        return ['SFA0', 'SDB0', 'SDP0', 'FC2', 'SDA2', 'SDB2',
                'SDP2', 'SDA3', 'SDB3', 'SDP3']


class LOCOConfigBO(LOCOConfig):
    """Sirius Booster LOCO configuration."""

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

    @property
    def famname_skewquadset(self):
        """."""
        return ['QS']
