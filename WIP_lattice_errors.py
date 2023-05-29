#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt

import pyaccel as _pyaccel
import pymodels as _pymodels
from apsuite.orbcorr import OrbitCorr, CorrParams
from apsuite.optics_analysis import TuneCorr, OpticsCorr, CouplingCorr
from apsuite.commisslib.measure_bba import BBAParams
from mathphys.functions import save_pickle, load_pickle


class ConfigErrors:

    def __init__(self):
        self._fam_names = []
        self._sigma_x = 0
        self._sigma_y = 0
        self._sigma_roll = 0
        self._sigma_pitch = 0
        self._sigma_yaw = 0
        self._sigmas = dict()
        self._um = 1e-6
        self._mrad = 1e-3
        self._percent = 1e-2

    @property
    def error_types(self):
        return self._error_types

    @property
    def fam_names(self):
        return self._fam_names

    @fam_names.setter
    def fam_names(self, value):
        self._fam_names = value

    @property
    def sigma_x(self):
        return self._sigma_x

    @sigma_x.setter
    def sigma_x(self, value):
        self._sigma_x = value * self._um

    @property
    def sigma_y(self):
        return self._sigma_y

    @sigma_y.setter
    def sigma_y(self, value):
        self._sigma_y = value * self._um

    @property
    def sigma_roll(self):
        return self._sigma_roll

    @sigma_roll.setter
    def sigma_roll(self, value):
        self._sigma_roll = value * self._mrad

    @property
    def sigma_pitch(self):
        return self._sigma_pitch

    @sigma_pitch.setter
    def sigma_pitch(self, value):
        self._sigma_pitch = value * self._mrad

    @property
    def sigma_yaw(self):
        return self._sigma_yaw

    @sigma_yaw.setter
    def sigma_yaw(self, value):
        self._sigma_yaw = value * self._mrad

    @property
    def sigmas(self):
        return self._sigmas

    @sigmas.setter
    def sigmas(self, value):
        self._sigmas = value


class MultipolesErrors(ConfigErrors):

    def __init__(self):
        super().__init__()
        self._sigma_excit = 0
        self._r0 = 12e-3
        self._multipoles_dict = None
        self._normal_multipoles_order = []
        self._skew_multipoles_order = []
        self._sigma_multipoles_n = []
        self._sigma_multipoles_s = []

    @property
    def sigma_excit(self):
        return self._sigma_excit

    @sigma_excit.setter
    def sigma_excit(self, value):
        self._sigma_excit = value * self._percent

    @property
    def r0(self):
        return self._r0

    @r0.setter
    def r0(self, value):
        self._r0 = value

    @property
    def normal_multipoles_order(self):
        return self._normal_multipoles_order

    @normal_multipoles_order.setter
    def normal_multipoles_order(self, value):
        self._normal_multipoles_order = value

    @property
    def skew_multipoles_order(self):
        return self._skew_multipoles_order

    @skew_multipoles_order.setter
    def skew_multipoles_order(self, value):
        self._skew_multipoles_order = value

    @property
    def sigma_multipoles_n(self):
        return self._sigma_multipoles_n

    @sigma_multipoles_n.setter
    def sigma_multipoles_n(self, value):
        self._sigma_multipoles_n = value

    @property
    def sigma_multipoles_s(self):
        return self._sigma_multipoles_s

    @sigma_multipoles_s.setter
    def sigma_multipoles_s(self, value):
        self._sigma_multipoles_s = value

    @property
    def multipoles_dict(self):
        return self._multipoles_dict

    @multipoles_dict.setter
    def multipoles_dict(self, value):
        self._multipoles_dict = value

    def create_multipoles_dict(self):
        n_multipoles_order_dict = dict()
        s_multipoles_order_dict = dict()
        for i, order in enumerate(self.normal_multipoles_order):
            n_multipoles_order_dict[order] = self.sigma_multipoles_n[i]
        for i, order in enumerate(self.skew_multipoles_order):
            s_multipoles_order_dict[order] = self.sigma_multipoles_s[i]
        self.multipoles_dict = dict()
        self.multipoles_dict['normal'] = n_multipoles_order_dict
        self.multipoles_dict['skew'] = n_multipoles_order_dict
        self.multipoles_dict['r0'] = self._r0


class DipolesErrors(MultipolesErrors):

    def __init__(self):
        super().__init__()
        self._sigma_kdip = 0
        self._set_default_dipole_config()

    @property
    def sigma_kdip(self):
        return self._sigma_kdip

    @sigma_kdip.setter
    def sigma_kdip(self, value):
        self._sigma_kdip = value * self._percent

    def _set_default_dipole_config(self):
        self.fam_names = ['B1', 'B2', 'BC']
        self.sigma_x = 40
        self.sigma_y = 40
        self.sigma_roll = 0.30
        self.sigma_pitch = 0
        self.sigma_yaw = 0
        self.sigma_excit = 0.05
        self.sigma_kdip = 0.10
        self.sigma_multipoles_n = _np.ones(4)*1.5e-4
        self.sigma_multipoles_s = _np.ones(4)*0.5e-4
        self.normal_multipoles_order = [3, 4, 5, 6]
        self.skew_multipoles_order = [3, 4, 5, 6]
        self.create_multipoles_dict()

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['pitch'] = self.sigma_pitch
        sigmas['yaw'] = self.sigma_yaw
        sigmas['excit'] = self.sigma_excit
        sigmas['kdip'] = self.sigma_kdip
        sigmas['multipoles'] = self.multipoles_dict
        self.sigmas = sigmas


class QuadsErrors(MultipolesErrors):

    def __init__(self):
        super().__init__()
        self._set_default_quad_config()

    def _set_default_quad_config(self):
        self.fam_names = ['QFA', 'QDA', 'Q1', 'Q2', 'Q3', 'Q4', 'QDB1',
                          'QFB',  'QDB2', 'QDP1', 'QFP', 'QDP2']
        self.sigma_x = 40
        self.sigma_y = 40
        self.sigma_roll = 0.30
        self.sigma_pitch = 0
        self.sigma_yaw = 0
        self.sigma_excit = 0.05
        self.sigma_multipoles_n = _np.ones(4)*1.5e-4
        self.sigma_multipoles_s = _np.ones(4)*0.5e-4
        self.normal_multipoles_order = [3, 4, 5, 6]
        self.skew_multipoles_order = [3, 4, 5, 6]
        self.create_multipoles_dict()

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['pitch'] = self.sigma_pitch
        sigmas['yaw'] = self.sigma_yaw
        sigmas['excit'] = self.sigma_excit
        sigmas['multipoles'] = self.multipoles_dict
        self.sigmas = sigmas


class QuadsSkewErrors(MultipolesErrors):

    def __init__(self):
        super().__init__()
        self._set_default_quad_config()

    def _set_default_quad_config(self):
        self.fam_names = ['QS']
        self.sigma_x = 0
        self.sigma_y = 0
        self.sigma_roll = 0
        self.sigma_pitch = 0
        self.sigma_yaw = 0
        self.sigma_excit = 0.05
        self.sigma_multipoles_n = _np.ones(4)*1.5e-4
        self.sigma_multipoles_s = _np.ones(4)*0.5e-4
        self.normal_multipoles_order = [3, 4, 5, 6]
        self.skew_multipoles_order = [3, 4, 5, 6]
        self.create_multipoles_dict()

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['pitch'] = self.sigma_pitch
        sigmas['yaw'] = self.sigma_yaw
        sigmas['excit'] = self.sigma_excit
        sigmas['multipoles'] = self.multipoles_dict
        self.sigmas = sigmas


class SextsErrors(MultipolesErrors):

    def __init__(self):
        super().__init__()
        self._set_default_sext_config()

    def _set_default_sext_config(self):
        self.fam_names = ['SFA0', 'SDA0', 'SDA1', 'SFA1', 'SDA2', 'SDA3',
                          'SFA2', 'SFB2', 'SDB3', 'SDB2', 'SFB1', 'SDB1',
                          'SDB0', 'SFB0', 'SFP2', 'SDP3', 'SDP2', 'SFP1',
                          'SDP1', 'SDP0', 'SFP0']
        self.sigma_x = 40
        self.sigma_y = 40
        self.sigma_roll = 0.17
        self.sigma_pitch = 0
        self.sigma_yaw = 0
        self.sigma_excit = 0.05
        self.sigma_multipoles_n = _np.ones(4)*1.5e-4
        self.sigma_multipoles_s = _np.ones(4)*0.5e-4
        self.normal_multipoles_order = [4, 5, 6, 7]
        self.skew_multipoles_order = [4, 5, 6, 7]
        self.create_multipoles_dict()

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['pitch'] = self.sigma_pitch
        sigmas['yaw'] = self.sigma_yaw
        sigmas['excit'] = self.sigma_excit
        sigmas['multipoles'] = self.multipoles_dict
        self.sigmas = sigmas


class GirderErrors(ConfigErrors):

    def __init__(self):
        super().__init__()
        self._set_default_girder_config()

    def _set_default_girder_config(self):
        self.fam_names = ['girder']
        self.sigma_x = 80
        self.sigma_y = 80
        self.sigma_roll = 0.30
        self.sigma_pitch = 0
        self.sigma_yaw = 0

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['pitch'] = self.sigma_pitch
        sigmas['yaw'] = self.sigma_yaw
        self.sigmas = sigmas


class BpmsErrors(ConfigErrors):

    def __init__(self):
        super().__init__()
        self._set_default_bpm_config()

    def _set_default_bpm_config(self):
        self.fam_names = ['BPM']
        self.sigma_x = 20
        self.sigma_y = 20
        self.sigma_roll = 0.30
        self.sigma_pitch = 0
        self.sigma_yaw = 0

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['pitch'] = self.sigma_pitch
        sigmas['yaw'] = self.sigma_yaw
        self.sigmas = sigmas


class ManageErrors():

    def __init__(self):
        self._nr_mach = 20
        self._seed = 131071
        self._cutoff = 1
        self._error_configs = []
        self._famdata = None
        self._spos = None
        self._fam_errors = None
        self._bba_idcs = None
        self._nominal_model = None
        self._models = []
        self.functions = {'posx': _pyaccel.lattice.add_error_misalignment_x,
                          'posy': _pyaccel.lattice.add_error_misalignment_y,
                          'roll': _pyaccel.lattice.add_error_rotation_roll,
                          'pitch': _pyaccel.lattice.add_error_rotation_pitch,
                          'yaw': _pyaccel.lattice.add_error_rotation_yaw,
                          'excit': _pyaccel.lattice.add_error_excitation_main,
                          'kdip': _pyaccel.lattice.add_error_excitation_kdip}
        self._ocorr_params = None
        self._orbcorr = None
        self._nominal_orbrespm = None
        self.use_nominal_jacobian = True
        self.do_bba = True

    @property
    def nr_mach(self):
        return self._nr_mach

    @nr_mach.setter
    def nr_mach(self, value):
        if isinstance(value, int):
            self._nr_mach = value
        else:
            raise ValueError('Number of machines must be an integer')

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    @property
    def error_configs(self):
        return self._error_configs

    @error_configs.setter
    def error_configs(self, value):
        self._error_configs = value

    @property
    def famdata(self):
        return self._famdata

    @famdata.setter
    def famdata(self, value):
        self._famdata = value

    @property
    def spos(self):
        return self._spos

    @spos.setter
    def spos(self, value):
        self._spos = value

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value

    @property
    def fam_errors(self):
        return self._fam_errors

    @fam_errors.setter
    def fam_errors(self, value):
        self._fam_errors = value

    @property
    def bba_idcs(self):
        return self._bba_idcs

    @bba_idcs.setter
    def bba_idcs(self, value):
        self._bba_idcs = value

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, value):
        self._models = value

    @property
    def nominal_model(self):
        return self._nominal_model

    @nominal_model.setter
    def nominal_model(self, value):
        self._nominal_model = value

    @property
    def nominal_orbrespm(self):
        return self._nominal_orbrespm

    @nominal_orbrespm.setter
    def nominal_orbrespm(self, value):
        self._nominal_orbrespm = value

    @property
    def orbcorr(self):
        return self._orbcorr

    @orbcorr.setter
    def orbcorr(self, value):
        self._orbcorr = value

    @property
    def ocorr_params(self):
        return self._ocorr_params

    @ocorr_params.setter
    def ocorr_params(self, value):
        self._ocorr_params = value

    def generate_normal_dist(self, sigma, dim, mean=0):
        _np.random.seed = self.seed
        dist = _np.random.normal(loc=mean, scale=sigma, size=dim)
        while _np.any(_np.abs(dist) > self.cutoff*sigma):
            idx = _np.argwhere(_np.abs(dist) > self.cutoff*sigma)
            for i in idx:
                mach = i[0]
                element = i[1]
                dist[mach][element] = _np.random.normal(
                    loc=mean, scale=sigma, size=1)
        return dist

    def generate_errors(self, save_errors=False):
        fam_errors = dict()
        for config in self.error_configs:
            for fam_name in config.fam_names:
                idcs = _np.array(self.famdata[fam_name]['index'],
                                 dtype="object")
                error_type_dict = dict()
                for error_type, sigma in config.sigmas.items():
                    if error_type == 'multipoles':
                        error = dict()
                        multipole_dict_n = dict()
                        multipole_dict_s = dict()
                        for order, mp_value in sigma['normal'].items():
                            error_ = self.generate_normal_dist(
                                        sigma=mp_value, dim=(self.nr_mach,
                                                             len(idcs)))
                            multipole_dict_n[order] = error_
                        for order, mp_value in sigma['skew'].items():
                            error_ = self.generate_normal_dist(
                                        sigma=mp_value, dim=(self.nr_mach,
                                                             len(idcs)))
                            multipole_dict_s[order] = error_
                        error['normal'] = multipole_dict_n
                        error['skew'] = multipole_dict_s
                        error['r0'] = sigma['r0']
                    else:
                        error = self.generate_normal_dist(
                            sigma=sigma, dim=(self.nr_mach, len(idcs)))
                    error_type_dict[error_type] = error
                error_type_dict['index'] = idcs
                fam_errors[fam_name] = error_type_dict
        self.fam_errors = fam_errors
        if save_errors:
            self.save_error_file()
        return fam_errors

    def save_error_file(self, filename=None):
        if filename is None:
            filename = 'errors'  # Need to change this
        save_pickle(self.fam_errors, filename, overwrite=True)

    def load_error_file(self, filename):
        self.fam_errors = load_pickle(filename)
        fams = list(self.fam_errors.keys())
        nr_mach = len(self.fam_errors[fams[0]]['posx'])
        self.nr_mach = nr_mach
        return self.fam_errors

    def create_models(self):
        models_ = list()
        for _ in range(self.nr_mach):
            model = _pymodels.si.create_accelerator()
            model.cavity_on = True
            model.radiation_on = 1
            models_.append(model)
        self.models = models_

    def get_bba_idcs(self):
        quaddevnames = list(BBAParams.QUADNAMES)
        quads = [famname for famname in self.famdata.keys()
                 if famname[0] == 'Q' and famname[1] != 'N']
        quads_idcs = list()
        for quadfam in quads:
            for idx, devname in zip(self.famdata[quadfam]['index'],
                                    self.famdata[quadfam]['devnames']):
                if devname in quaddevnames:
                    quads_idcs.append(idx)
        bba_idcs = _np.array(quads_idcs).ravel()
        self.bba_idcs = _np.sort(bba_idcs)

    def apply_errors(self, nr_steps, mach):

        print('Applying errors...', end='')
        for fam, family in self.fam_errors.items():
            inds = family['index']
            error_types = [err for err in family.keys() if err != 'index']
            for error_type in error_types:
                if error_type != 'multipoles':
                    errors = family[error_type]
                    self.functions[error_type](
                        self.models[mach], inds, errors[mach]/nr_steps)
                # else:
                #     mag_key = fam[0] if fam != 'QS' else fam
                #     main_monom = _np.ones(len(inds))*main_monoms[mag_key]
                #     r0 = family[error_type]['r0']
                #     polb_order = list(family[error_type]['normal'].keys())
                #     pola_order = list(family[error_type]['skew'].keys())
                #     Bn_norm = _np.zeros((len(inds), max(polb_order)+1))
                #     An_norm = _np.zeros((len(inds), max(pola_order)+1))
                #     for n in polb_order:
                #         Bn_norm[:, n] = family[error_type]['normal'][n][mach]/nr_steps
                #     for n in pola_order:
                #         An_norm[:, n] = family[error_type]['skew'][n][mach]/nr_steps
                #     _pyaccel.lattice.add_error_multipoles(
                #         self.models[mach], inds, r0,
                #         main_monom, Bn_norm, An_norm)
        print('Done!')

    def apply_multipoles_errors(self, mach):
        error_type = 'multipoles'
        for fam, family in self.fam_errors.items():
            inds = family['index']
            if error_type in family.keys():
                main_monoms = {'B': 1, 'Q': 2, 'S': 3, 'QS': -2}
                mag_key = fam[0] if fam != 'QS' else fam
                main_monom = _np.ones(len(inds))*main_monoms[mag_key]
                r0 = family[error_type]['r0']
                polb_order = list(family[error_type]['normal'].keys())
                pola_order = list(family[error_type]['skew'].keys())
                Bn_norm = _np.zeros((len(inds), max(polb_order)+1))
                An_norm = _np.zeros((len(inds), max(pola_order)+1))
                for n in polb_order:
                    Bn_norm[:, n] = family[error_type]['normal'][n][mach]
                for n in pola_order:
                    An_norm[:, n] = family[error_type]['skew'][n][mach]
                _pyaccel.lattice.add_error_multipoles(
                    self.models[mach], inds, r0,
                    main_monom, Bn_norm, An_norm)

    def get_girder_errors(self, nr_steps, step, idcs, mach):
        girder_errorx = list()
        girder_errory = list()
        for i, girder in enumerate(self.fam_errors['girder']['index']):
            for idx in girder:
                if _np.any(idcs == idx):
                    girder_errorx.append(
                        self.fam_errors['girder']['posx'][mach][i])
                    girder_errory.append(
                        self.fam_errors['girder']['posy'][mach][i])
        girder_errors_idcs = _np.array(girder_errorx + girder_errory).ravel()
        return step*girder_errors_idcs/nr_steps

    def simulate_bba(self, nr_steps, step, mach):
        bpms = _np.array(self.famdata['BPM']['index']).ravel()
        orb0 = _np.zeros(2*len(bpms))
        orb0[:len(bpms)] += _pyaccel.lattice.get_error_misalignment_x(
                self.models[mach], self.bba_idcs).ravel()
        orb0[:len(bpms)] += _pyaccel.lattice.get_error_misalignment_x(
                self.models[mach], bpms).ravel()
        orb0[len(bpms):] += _pyaccel.lattice.get_error_misalignment_y(
                self.models[mach], self.bba_idcs).ravel()
        orb0[len(bpms):] += _pyaccel.lattice.get_error_misalignment_y(
                self.models[mach], bpms).ravel()

        if 'girder' in self.fam_errors.keys():
            bpm_girder_errors = self.get_girder_errors(
                nr_steps, step, bpms, mach)
            orb0 -= bpm_girder_errors

        return orb0

    def config_orb_corr(self):
        self.orbcorr = OrbitCorr(
                self.nominal_model, 'SI', params=self.ocorr_params)
        self.nominal_orbrespm = self.orbcorr.get_jacobian_matrix()

    def correct_orbit(self, orb0, mach):
        print('Correcting orbit...', end='')
        jac = None
        if self.use_nominal_jacobian:
            jac = self.nominal_orbrespm
        self.orbcorr.respm.model = self.models[mach]
        if not self.orbcorr.correct_orbit(
                jacobian_matrix=jac, goal_orbit=orb0):
            print('Could not achieve tolerance!')
        else:
            print('Done!\n')

        return self.orbcorr.get_orbit()

    def config_tune_corr(self):
        self.tunecorr = TuneCorr(
                            self.nominal_model,
                            'SI', method='Proportional',
                            grouping='TwoKnobs')
        self.tunemat = self.tunecorr.calc_jacobian_matrix()
        self.goal_tunes = self.tunecorr.get_tunes()
        print('Nominal tunes: {:.4f} {:.4f}'.format(
            self.goal_tunes[0], self.goal_tunes[1]))

    def correct_tunes(self, mach):
        self.tunecorr.correct_parameters(
            model=self.models[mach],
            goal_parameters=self.goal_tunes,
            jacobian_matrix=self.tunemat)

    def calc_coupling(self, mach):
        ed_tang, *_ = _pyaccel.optics.calc_edwards_teng(self.models[mach])
        min_tunesep, ratio =\
            _pyaccel.optics.estimate_coupling_parameters(ed_tang)

        return min_tunesep

    def config_coupling_corr(self):
        self.coup_corr = CouplingCorr(self.nominal_model, 'SI')
        self.coupmat = self.coup_corr.calc_jacobian_matrix(
            model=self.nominal_model)

    def correct_coupling(self, mach):
        self.coup_corr.model = self.models[mach]
        self.coup_corr.coupling_correction(
                    jacobian_matrix=self.coupmat,
                    tol=1e-8, weight_dispy=1)

    def config_optics_corr(self):
        self.opt_corr = OpticsCorr(self.nominal_model, 'SI')
        self.optmat = self.opt_corr.calc_jacobian_matrix()

    def correct_optics(self, mach):
        self.opt_corr.model = self.models[mach]
        return self.opt_corr.optics_corr_loco(goal_model=self.nominal_model,
                                              jacobian_matrix=self.optmat)

    def generate_machines(self, nr_steps=10):

        # Get quadrupoles near BPMs indexes
        self.get_bba_idcs()

        # Config orbit correction
        print('Configuring orbit correction...')
        self.config_orb_corr()

        # Config tune correction
        # print('Configuring tune correction...')
        # self.config_tune_corr()

        # Config coupling correction
        # print('Configuring coupling correction...')
        # self.config_coupling_corr()

        # Config optics correction
        # print('Configuring optics correction...')
        # self.config_optics_corr()

        self.create_models()
        self.orb0, self.orbf = list(), list()

        for mach in range(self.nr_mach):
            print('Machine ', mach)

            fig, axs = _plt.subplots(2)

            for step in range(nr_steps):
                print('Step ', step+1)

                self.apply_errors(nr_steps, mach)

                #  Orbit setted by BBA or setted to zero
                if self.do_bba:
                    orb0_ = self.simulate_bba(nr_steps, step+1, mach)
                else:
                    orb0_ = _np.zeros(2*len(self.bba_idcs))

                orbf_ = self.correct_orbit(orb0_, mach)

                fig.suptitle('Orbref ad orbfinal')
                axs[0].plot(1e6*orb0_[:160], label=step)
                axs[1].plot(1e6*orbf_[:160], label=step)
                axs[0].legend()
                axs[1].legend()
                _plt.show()

            # Correct tunes
            # tunes = self.tunecorr.get_tunes(model=self.models[mach])
            # print('Old tunes: {:.4f} {:.4f}'.format(tunes[0], tunes[1]))
            # for i in range(2):
            #     self.correct_tunes(mach)
            #     tunes = self.tunecorr.get_tunes(model=self.models[mach])
            #     print('iter # {} - New tunes: {:.4f} {:.4f}'.format(
            #         i+1, tunes[0], tunes[1]))

            # Correct coupling
            # mintune = self.calc_coupling(mach)
            # print(
            #     'Minimum tune separation before correction: {:.3f} %'.format(
            #         100*mintune))
            # self.correct_coupling(mach)
            # mintune = self.calc_coupling(mach)
            # print(
            #     'Minimum tune separation after correction: {:.3f} %'.format(
            #         100*mintune))

            # Symmetrize optics
            # res = self.correct_optics(mach)
            # print(res)

            self.apply_multipoles_errors(mach)

            self.orb0.append(orb0_)
            self.orbf.append(orbf_)
        return self.orb0, self.orbf
