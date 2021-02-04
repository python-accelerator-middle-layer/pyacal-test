"""."""
from collections import OrderedDict as _OrderedDict
import numpy as _np
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupply as _PowerSupply
from siriuspy.search import PSSearch as _PSSearch
import pymodels as _pymod
from .base import BaseClass as _BaseClass


class Utils:
    """."""

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

    @staticmethod
    def implement_changes(magnets, strengths):
        """."""
        for mag, stren in zip(magnets, strengths):
            magnets[mag].strength = stren
            print('\n applied!')

    @staticmethod
    def create_devices(devices, devices_names):
        """."""
        for mag in devices_names:
            if mag not in devices:
                devices[mag] = _PowerSupply(mag)


class SetOpticsFamilies(_BaseClass):
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
        """Apply strengths of families to the machine for a given optics.

        Arguments:
        - acc: TB, BO, TS or SI.
        - optics_mode: available modes in pymodels. If None, default
        optics_mode for the accelerator will be used (optional).
        - optics_data: dictionary with quadrupoles and sextupoles, if applies
        for the accelerator, strengths not integrated (optional).
        """
        super().__init__()
        self.acc = acc.upper()
        self.optics_mode = optics_mode
        if optics_data is not None:
            if optics_data not in (dict, _OrderedDict):
                raise ValueError(
                    'optics_data must be a dictionary or OrderedDict')
        self.optics_data = optics_data
        self.model = None
        self.quad_list = []
        self.sext_list = []
        self.devices = _OrderedDict()
        self.applied_optics = _OrderedDict()
        self._select_model()
        self._select_magnets()
        Utils.create_devices(
            devices=self.devices,
            devices_names=self.quad_list + self.sext_list)
        self.model = self._pymodpack.create_accelerator()
        self._create_optics_data()
        self.famdata = self._pymodpack.get_family_data(self.model)
        # convert to integrated strengths
        for key in self.optics_data:
            if key in self.famdata:
                idx = self.famdata[key]['index'][0][0]
                self.optics_data[key] *= self.model[idx].length

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
        if init is not None:
            if len(init) != len(mags):
                raise ValueError(
                    'initial strength vector length is incompatible with \
                    number of magnets')
        init = initv if init is None else init

        dperc = Utils.print_current_status(
            magnets=mags, goal_strength=goalv)
        average = _np.mean(dperc) if average is None else average
        print('average desired: {:+.4f} %'.format(average))
        print('average obtained: {:+.4f} %'.format(_np.mean(dperc)))
        print()

        ddif = average - _np.asarray(dperc)
        dimp_perc = ddif * (factor/100)
        implem = (1 + dimp_perc/100) * init

        Utils.print_strengths_implemented(
            factor=factor, magnets=mags,
            init_strength=init, implem_strength=implem)

        if apply:
            Utils.implement_changes(magnets=mags, strengths=implem)
        return init

    # private methods
    def _select_model(self):
        if self.acc == 'TB':
            self._pymodpack = _pymod.tb
        elif self.acc == 'BO':
            self._pymodpack = _pymod.bo
        elif self.acc == 'TS':
            self._pymodpack = _pymod.ts
        elif self.acc == 'SI':
            self._pymodpack = _pymod.si

    def _select_magnets(self):
        slist = []
        pvstr = ''
        if self.acc == 'TB':
            qlist = SetOpticsFamilies.TB_QUADS
        elif self.acc == 'TS':
            qlist = SetOpticsFamilies.TS_QUADS
        else:
            pvstr = self.acc + '-Fam:PS-'
            qlist = self._pymodpack.families.families_quadrupoles()
            slist = self._pymodpack.families.families_sextupoles()
        self.quad_list = [_PVName(pvstr+mag) for mag in qlist]
        self.sext_list = [_PVName(pvstr+mag) for mag in slist]

    def _create_devices(self):
        for mag in self.quad_list + self.sext_list:
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


class ChangeCorretors(_BaseClass):
    """."""

    def __init__(self, acc):
        """."""
        super().__init__()
        self.acc = acc.upper()
        self._get_corr_names()
        self.devices = _OrderedDict()
        Utils.create_devices(
            devices=self.devices,
            devices_names=self.ch_list+self.cv_list)
        self.applied_strength = _OrderedDict()

    def get_applied_strength(self, magnets=None):
        """."""
        magnets = magnets or self.ch_list + self.cv_list
        for mag in magnets:
            self.applied_strength[mag] = self.devices[mag].strength

    def apply_factor(
            self, magtype=None, factor=1, apply=False):
        """."""
        mags, init = self._get_initial_state(magtype)
        implem = factor * init
        print(
            'Factor {:9.3f} will be applied in kicks of {:10s} magnets'.format(
                magtype, factor))
        if apply:
            Utils.implement_changes(magnets=mags, strengths=implem)
        return init

    def change_average_kicks(
            self, magtype=None, average=None, percentage=5, apply=False):
        """."""
        mags, init = self._get_initial_state(magtype)
        curr_ave = _np.mean(init)
        goal_ave = curr_ave if average is None else average
        diff = (curr_ave - goal_ave) * percentage/100
        implem = init - diff
        print('actual average : {:+.4f} urad'.format(curr_ave))
        print('goal average: {:+.4f} urad'.format(goal_ave))
        print('percentage of application: {:5.1f}%'.format(percentage))
        if apply:
            Utils.implement_changes(magnets=mags, strengths=implem)
        return init

    def apply_delta_kicks(
            self, delta_kicks, magtype=None, percentage=5, apply=False):
        """."""
        mags, init = self._get_initial_state(magtype)
        dkicks = _np.asarray(delta_kicks)
        if len(dkicks) != len(mags):
            raise ValueError(
                'delta kick vector length is incompatible with \
                number of magnets')
        implem = init + dkicks * (percentage/100)
        if apply:
            Utils.implement_changes(magnets=mags, strengths=implem)
        return init

    def apply_kicks(self, kicks, magtype=None, percentage=5, apply=False):
        """."""
        _, init = self._get_initial_state(magtype)
        dkicks = kicks - init
        self.apply_delta_kicks(
            delta_kicks=dkicks, magtype=magtype,
            percentage=percentage, apply=apply)

    def _get_corr_names(self):
        ch_names = _PSSearch.get_psnames(
            {'sec': self.acc, 'dis': 'PS', 'dev': 'CH'})
        cv_names = _PSSearch.get_psnames(
            {'sec': self.acc, 'dis': 'PS', 'dev': 'CV'})
        self.ch_list = [_PVName(mag) for mag in ch_names]
        self.cv_list = [_PVName(mag) for mag in cv_names]

    def _create_devices(self):
        for mag in self.ch_list + self.cv_list:
            if mag not in self.devices:
                self.devices[mag] = _PowerSupply(mag)

    def _check_magtype(self, magtype):
        if magtype in ('CH', 'CV'):
            mags = {
                key: val for key, val in self.devices.items()
                if magtype in key}
            mags = _OrderedDict(mags)
        else:
            raise ValueError('magtype must be CH or CV.')
        return magtype

    def _get_initial_state(self, magtype):
        mags = self._check_magtype(magtype)
        init = _np.asarray([mags[mag].strength for mag in mags])
        return mags, init


class SetOpticsIndividual(_BaseClass):
    """."""

    def __init__(self, model=None):
        """."""
        super().__init__()
        self.model = _pymod.si.create_accelerator() or model
        self.fam_data = _pymod.si.get_family_data(self.model)
        self.devices = _OrderedDict()
        self.quad_names = list()
        self.skewquad_names = list()
        self._get_quad_names()
        self._get_skewquad_names()
        Utils.create_devices(
            devices=self.devices,
            devices_names=self.quad_names+self.skewquad_names)

    def apply_strengths(self, magtype, strengths, percentage=5, apply=False):
        """."""
        mags, init = self._get_initial_state(magtype)
        stren = _np.asarray(strengths)
        if len(stren) != len(mags):
            raise ValueError(
                'strength vector length is incompatible with \
                number of magnets')
        dstren = stren - init
        _ = self.apply_delta_strengths(
            magtype=magtype, delta_strengths=dstren,
            percentage=percentage, apply=apply)
        return init

    def apply_delta_strengths(
            self, magtype, delta_strengths, percentage=5, apply=False):
        """."""
        mags, init = self._get_initial_state(magtype)
        dstren = _np.asarray(delta_strengths)
        if len(dstren) != len(mags):
            raise ValueError(
                'delta strength vector length is incompatible with \
                number of magnets')
        implem = init + dstren * (percentage/100)
        if apply:
            Utils.implement_changes(magnets=mags, strengths=implem)
        return init

    def _get_quad_names(self):
        """."""
        self.quads_idx = self.fam_data['QN']['index']
        self.quads_idx = _np.asarray(
            [idx[len(idx)//2] for idx in self.quads_idx])

        for qidx in self.quads_idx:
            name = self.model[qidx].fam_name
            idc = self.fam_data[name]['index'].index([qidx, ])
            sub = self.fam_data[name]['subsection'][idc]
            inst = self.fam_data[name]['instance'][idc]
            qname = f'SI-{sub}:PS-{name}-{inst}'
            self.quad_names.append(qname.strip('-'))

    def _get_skewquad_names(self):
        """."""
        self.skewquads_idx = self.fam_data['QS']['index']
        self.skewquads_idx = _np.asarray(
            [idx[len(idx)//2] for idx in self.skewquads_idx])

        for qidx in self.skewquads_idx:
            name = self.model[qidx].fam_name
            idc = self.fam_data[name]['index'].index([qidx, ])
            sub = self.fam_data[name]['subsection'][idc]
            inst = self.fam_data[name]['instance'][idc]
            qname = f'SI-{sub}:PS-QS-{inst}'
            self.skewquad_names.append(qname.strip('-'))

    def _create_devices(self):
        for mag in self.quad_names + self.skewquad_names:
            if mag not in self.devices:
                self.devices[mag] = _PowerSupply(mag)

    def _get_initial_state(self, magtype):
        mags = self._check_magtype(magtype)
        init = _np.asarray([mags[mag].strength for mag in mags])
        return mags, init

    def _check_magtype(self, magtype):
        if magtype == 'quadrupole':
            maglist = self.quad_names
        elif magtype == 'skew_quadrupole':
            maglist = self.skewquad_names
        else:
            raise ValueError('magtype must be quadrupole or skew_quadrupole')
        mags = _OrderedDict([
            {key: val for key, val in self.devices.items() if key in maglist}
            ])
        return mags
