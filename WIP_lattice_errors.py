#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt
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
        self.sigma_pitch = 0.10
        self.sigma_yaw = 0.10
        self.sigma_excit = 0.05
        self.sigma_kdip = 0.10
        self.sigma_multipoles_n = [0, 0, 0]
        self.sigma_multipoles_s = [0, 0, 0]
        self.normal_multipoles_order = [2, 3, 4]
        self.skew_multipoles_order = [2, 3, 4]
        self.create_multipoles_dict()

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['pitch'] = self.sigma_pitch
        sigmas['yaw'] = self.sigma_yaw
        sigmas['excitation'] = self.sigma_excit
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
        self.sigma_pitch = 0.10
        self.sigma_yaw = 0.10
        self.sigma_excit = 0.05
        self.sigma_multipoles_n = [0, 0, 0]
        self.sigma_multipoles_s = [0, 0, 0]
        self.normal_multipoles_order = [3, 4, 5]
        self.skew_multipoles_order = [3, 4, 5]
        self.create_multipoles_dict()

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['pitch'] = self.sigma_pitch
        sigmas['yaw'] = self.sigma_yaw
        sigmas['excitation'] = self.sigma_excit
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
        self.sigma_pitch = 0.10
        self.sigma_yaw = 0.10
        self.sigma_excit = 0.05
        self.sigma_multipoles_n = [0, 0, 0]
        self.sigma_multipoles_s = [0, 0, 0]
        self.normal_multipoles_order = [4, 5, 6]
        self.skew_multipoles_order = [4, 5, 6]
        self.create_multipoles_dict()

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['pitch'] = self.sigma_pitch
        sigmas['yaw'] = self.sigma_yaw
        sigmas['excitation'] = self.sigma_excit
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
        self.sigma_pitch = 0.10
        self.sigma_yaw = 0.10

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
        self.sigma_pitch = 0.10
        self.sigma_yaw = 0.10

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
        self._fam_errors = None

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

    def save_error_file(self):
        save_pickle(self.fam_errors, 'errors', overwrite=True)

    def load_error_file(self):
        self.fam_errors = load_pickle('errors')
        return self.fam_errors
