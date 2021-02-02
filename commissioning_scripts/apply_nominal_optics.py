"""."""
from collections import OrderedDict as _OrderedDict
import numpy as _np
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupply as _PowerSupply, SOFB as _SOFB
import pymodels as _pymod
from .base import BaseClass as _BaseClass


class SetOptics(_BaseClass):
    """."""

    TB_QUADS = [
        'LI-Fam:PS-QF2',
        'LI-01:PS-QD2',
        'LI-01:PS-QF3',
        'TB-01:PS-QD1',
        'TB-01:PS-QF1',
        'TB-02:PS-QD2A',
        'TB-02:PS-QD2B',
        'TB-02:PS-QF2A',
        'TB-02:PS-QF2B',
        'TB-03:PS-QD3',
        'TB-03:PS-QF3',
        'TB-04:PS-QD4',
        'TB-04:PS-QF4',
        ]

    TS_QUADS = [
        'TS-01:PS-QF1A',
        'TS-01:PS-QF1B',
        'TS-02:PS-QD2',
        'TS-02:PS-QF2',
        'TS-03:PS-QF3',
        'TS-04:PS-QD4A',
        'TS-04:PS-QD4B',
        'TS-04:PS-QF4',
        ]

    def __init__(self, acc, optics_mode=None, optics_data=None):
        """."""
        super().__init__()
        self.acc = acc
        self.optics_mode = optics_mode
        self.optics_data = optics_data
        self.model = None
        self.quad_list = []
        self.sext_list = []
        self.devices = _OrderedDict()
        self.applied_optics = _OrderedDict()
        self._select_model()
        self.devices['sofb'] = _SOFB(self._sofbdev)
        self._select_magnets()
        self._create_devices()
        self.model = self._pymodpack.create_accelerator()
        self._create_optics_data()
        self.famdata = self._pymodpack.get_family_data(self.model)
        # convert to integrated strengths
        for key in self.optics_data:
            if key in self.famdata:
                idx = self.famdata[key]['index'][0][0]
                self.optics_data[key] *= self.model[idx].length

    def _select_model(self):
        if self.acc == 'TB':
            self._pymodpack = _pymod.tb
            self._sofbdev = _SOFB.DEVICES.TB
        elif self.acc == 'BO':
            self._pymodpack = _pymod.bo
            self._sofbdev = _SOFB.DEVICES.BO
        elif self.acc == 'TS':
            self._pymodpack = _pymod.ts
            self._sofbdev = _SOFB.DEVICES.TS
        elif self.acc == 'SI':
            self._pymodpack = _pymod.si
            self._sofbdev = _SOFB.DEVICES.SI

    def _select_magnets(self):
        slist = []
        pvstr = ''
        if self.acc == 'TB':
            qlist = SetOptics.TB_QUADS
        elif self.acc == 'TS':
            qlist = SetOptics.TS_QUADS
        else:
            pvstr = self.acc + '-Fam:PS-'
            qlist = self._pymodpack.families.families_quadrupoles()
            slist = self._pymodpack.families.families_sextupoles()
        self.quad_list = [_PVName(pvstr+mag) for mag in qlist]
        self.sext_list = [_PVName(pvstr+mag) for mag in slist]
        sofb = self.devices['sofb']
        self.ch_list = [_PVName(mag) for mag in sofb.data.ch_names]
        self.cv_list = [_PVName(mag) for mag in sofb.data.cv_names]

    def _create_devices(self):
        all_mags = self.quad_list
        all_mags += self.sext_list
        all_mags += self.ch_list
        all_mags += self.cv_list
        for mag in all_mags:
            if mag not in self.devices:
                self.devices[mag] = _PowerSupply(mag)

    def _create_optics_data(self):
        if self.optics_data is None:
            optmode = self.optics_mode or self._pymodpack.default_optics_mode
            self.optics_data = self._pymodpack.lattice.get_optics_mode(
                optics_mode=optmode)
            if 'T' in self.acc:
                self.optics_data = self.optics_data[0]
                self.model = self.model[0]
        self.optics_data = _OrderedDict(
            [(key.upper(), val) for key, val in self.optics_data.items()])

    def _check_magtype(self, magtype):
        mags = _OrderedDict()
        if magtype == 'quadrupole':
            for key in self.devices:
                if 'Q' not in key:
                    continue
                else:
                    mags[key] = self.devices[key]
        elif magtype == 'sextupole':
            if 'T' in self.acc:
                raise ValueError('transport lines do not have sextupoles')
            for key in self.devices:
                if 'S' not in key:
                    continue
                else:
                    mags[key] = self.devices[key]
        else:
            raise ValueError('magtype must be quadrupole or sextupole')
        return mags

    def get_applied_strength(self, magnets=None):
        """."""
        magnets = magnets or self.devices
        for mag in magnets:
            magdev = mag.dev
            if self.acc == 'TB' and 'LI' in mag:
                magdev += 'L'
            self.applied_optics[mag.dev] = magnets[mag].strength

    def apply_strengths(
            self, magtype, init=None, average=None, factor=0, apply=False):
        """."""
        mags = self._check_magtype(magtype)
        initv = []
        goalv = []
        for mag in mags:
            initv.append(mags[mag].strength)
            magdev = mag.dev
            if self.acc == 'TB' and 'LI' in mag:
                magdev += 'L'
            goalv.append(self.optics_data[magdev])
        initv = _np.asarray(initv)
        goalv = _np.asarray(goalv)
        init = initv if init is None else init

        dperc = SetOptics.print_current_status(
            magnets=mags, goal_strength=goalv)
        average = _np.mean(dperc) if average is None else average
        print('average desired: {:+.4f} %'.format(average))
        print('average obtained: {:+.4f} %'.format(_np.mean(dperc)))
        print()

        ddif = _np.asarray(dperc) - average
        dimp_perc = factor/100 * (-ddif)
        implem = (1 + dimp_perc/100) * init

        SetOptics.print_strengths_implemented(
            factor=factor, magnets=mags,
            init_strength=init, implem_strength=implem)

        if apply:
            for mag, imp in zip(mags, implem):
                mags[mag].strength = imp
            print('\n applied!')
        return init

    @staticmethod
    def print_current_status(magnets, goal_strength):
        """."""
        diff = []
        print(
            '{:17s}  {:9s}  {:9s}  {:9s}%'.format(
                '', ' applied', ' goal', ' diff'))
        for mag, stren in zip(magnets, goal_strength):
            diff.append((magnets[mag].strength-stren)/stren*100)
            print('{:17s}: {:9.6f}  {:9.6f}  {:9.6f}%'.format(
                magnets[mag].devname, magnets[mag].strength, stren, diff[-1]))
        print()
        return diff

    @staticmethod
    def print_strengths_implemented(
            factor, magnets, init_strength, implem_strength):
        """."""
        print('-- to be implemented --')
        print('factor: {:5.1f}%'.format(factor))
        for mag, ini, imp in zip(magnets, init_strength, implem_strength):
            perc = (imp - ini) / ini * 100
            print(
                '{:17s}:  {:9.4f} -> {:9.4f}  [{:7.4}%]'.format(
                    magnets[mag].devname, ini, imp, perc))
