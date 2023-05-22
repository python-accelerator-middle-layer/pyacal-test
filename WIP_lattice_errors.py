#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt
from mathphys.functions import save_pickle


class config_errors:

    def __init__(self):
        self._fam_names = []
        self._sigma_x = 0
        self._sigma_y = 0
        self._sigma_roll = 0
        self._error_types = ['posx', 'posy', 'roll']
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
    def sigmas(self):
        return self._sigmas

    @sigmas.setter
    def sigmas(self, value):
        self._sigmas = value


class dipoles_errors(config_errors):

    def __init__(self):
        super().__init__()
        self._sigma_excit = 0
        self._sigma_kdip = 0
        self._sigma_multipoles_n = []
        self._sigma_multipoles_s = []
        self._error_types = ['posx', 'posy', 'roll', 'excitation', 'kdip',
                             'normal_multipoles', 'skew_multipoles']
        self._set_default_dipole_config()

    @property
    def sigma_excit(self):
        return self._sigma_excit

    @sigma_excit.setter
    def sigma_excit(self, value):
        self._sigma_excit = value * self._percent

    @property
    def sigma_kdip(self):
        return self._sigma_kdip

    @sigma_kdip.setter
    def sigma_kdip(self, value):
        self._sigma_kdip = value * self._percent

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

    def _set_default_dipole_config(self):
        self.fam_names = ['B1', 'B2', 'BC']
        self.sigma_x = 40
        self.sigma_y = 40
        self.sigma_roll = 0.30
        self.sigma_excit = 0.05
        self.sigma_kdip = 0.10
        self.sigma_multipoles_n = [0, 0, 0]
        self.sigma_multipoles_s = [0, 0, 0]

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['excitation'] = self.sigma_excit
        sigmas['kdip'] = self.sigma_kdip
        sigmas['normal_multipoles'] = self.sigma_multipoles_n
        sigmas['skew_multipoles'] = self.sigma_multipoles_s
        self.sigmas = sigmas


class quads_errors(config_errors):

    def __init__(self):
        super().__init__()
        self._sigma_excit = 0
        self._error_types = ['posx', 'posy', 'roll', 'excitation',
                             'normal_multipoles', 'skew_multipoles']
        self._sigma_multipoles_n = []
        self._sigma_multipoles_s = []
        self._set_default_quad_config()

    @property
    def sigma_excit(self):
        return self._sigma_excit

    @sigma_excit.setter
    def sigma_excit(self, value):
        self._sigma_excit = value * self._percent

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

    def _set_default_quad_config(self):
        self.fam_names = ['QFA', 'QDA', 'Q1', 'Q2', 'Q3', 'Q4', 'QDB1',
                          'QFB',  'QDB2', 'QDP1', 'QFP', 'QDP2']
        self.sigma_x = 40
        self.sigma_y = 40
        self.sigma_roll = 0.30
        self.sigma_excit = 0.05
        self.sigma_multipoles_n = [0, 0, 0]
        self.sigma_multipoles_s = [0, 0, 0]

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['excitation'] = self.sigma_excit
        sigmas['normal_multipoles'] = self.sigma_multipoles_n
        sigmas['skew_multipoles'] = self.sigma_multipoles_s
        self.sigmas = sigmas


class sexts_errors(config_errors):

    def __init__(self):
        super().__init__()
        self._sigma_excit = 0
        self._error_types = ['posx', 'posy', 'roll', 'excitation',
                             'normal_multipoles', 'skew_multipoles']
        self._sigma_multipoles_n = []
        self._sigma_multipoles_s = []
        self._set_default_sext_config()

    @property
    def sigma_excit(self):
        return self._sigma_excit

    @sigma_excit.setter
    def sigma_excit(self, value):
        self._sigma_excit = value * self._percent

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

    def _set_default_sext_config(self):
        self.fam_names = ['SFA0', 'SDA0', 'SDA1', 'SFA1', 'SDA2', 'SDA3',
                          'SFA2', 'SFB2', 'SDB3', 'SDB2', 'SFB1', 'SDB1',
                          'SDB0', 'SFB0', 'SFP2', 'SDP3', 'SDP2', 'SFP1',
                          'SDP1', 'SDP0', 'SFP0']
        self.sigma_x = 40
        self.sigma_y = 40
        self.sigma_roll = 0.17
        self.sigma_excit = 0.05
        self.sigma_multipoles_n = [0, 0, 0]
        self.sigma_multipoles_s = [0, 0, 0]

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        sigmas['excitation'] = self.sigma_excit
        sigmas['normal_multipoles'] = self.sigma_multipoles_n
        sigmas['skew_multipoles'] = self.sigma_multipoles_s
        self.sigmas = sigmas


class girder_errors(config_errors):

    def __init__(self):
        super().__init__()
        self._set_default_girder_config()

    def _set_default_girder_config(self):
        self.fam_names = ['girder']
        self.sigma_x = 80
        self.sigma_y = 80
        self.sigma_roll = 0.30

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        self.sigmas = sigmas


class bpms_errors(config_errors):

    def __init__(self):
        super().__init__()
        self._set_default_girder_config()

    def _set_default_girder_config(self):
        self.fam_names = ['BPM']
        self.sigma_x = 20
        self.sigma_y = 20
        self.sigma_roll = 0.30

        sigmas = dict()
        sigmas['posx'] = self.sigma_x
        sigmas['posy'] = self.sigma_y
        sigmas['roll'] = self.sigma_roll
        self.sigmas = sigmas


class manage_errors():

    @staticmethod
    def generate_normal_dist(sigma, cutoff, dim, mean=0, seed=131071):
        _np.random.seed(seed=seed)
        if not isinstance(sigma, list):
            dist = _np.random.normal(loc=mean, scale=sigma, size=dim)
            dist = _np.where(_np.abs(dist) < cutoff*sigma, dist, 0)
        else:
            dist = _np.ones((dim[0], dim[1], len(sigma)))
            for i, sigma_ in enumerate(sigma):
                dist[:, :, i] = _np.random.normal(
                    loc=mean, scale=sigma_, size=dim)
                dist[:, :, i] = _np.where(
                    _np.abs(dist[:, :, i]) < cutoff*sigma_, dist[:, :, i], 0)
        return dist

    @staticmethod
    def generate_errors(error_configs, nr_mach, cutoff, fam_data, seed=131071):
        fam_errors = dict()
        for config in error_configs:
            for fam_name in config.fam_names:
                idcs = _np.array(fam_data[fam_name]['index'], dtype="object")
                error_type_dict = dict()
                for error_type in config.error_types:
                    sigma = config.sigmas[error_type]
                    error = manage_errors.generate_normal_dist(
                        sigma=sigma, cutoff=cutoff,
                        dim=(nr_mach, len(idcs)), seed=seed)
                    error_type_dict[error_type] = error
                error_type_dict['index'] = idcs
                fam_errors[fam_name] = error_type_dict
        return fam_errors
