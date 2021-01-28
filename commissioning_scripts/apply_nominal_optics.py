"""."""
import numpy as _np
from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupply as _PowerSupply
import pymodels as _pymod


class SetOptics:
    """."""

    TB_QUADS = [
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
        self.acc = acc
        self.optics_mode = optics_mode
        self.optics_data = optics_data
        self.model = None
        self.quad_list = []
        self.sext_list = []
        self.devices = dict()
        self.applied_optics = dict()
        self._select_model()
        self._select_magnets()
        self._create_devices()
        self.model = self.pymodpack.create_accelerator()

        if optics_data is None:
            optmode = self.optics_mode or self.pymodpack.default_optics_mode
            self.optics_data = self.pymodpack.lattice.get_optics_mode(
                optics_mode=optmode)
            if 'T' in self.acc:
                self.optics_data = self.optics_data[0]
                self.model = self.model[0]

        self.optics_data = dict(
            (key.upper(), val) for key, val in self.optics_data.items())
        self.famdata = self.pymodpack.get_family_data(self.model)

        for key in self.optics_data:
            if key in self.famdata:
                idx = self.famdata[key]['index'][0][0]
                self.optics_data[key] *= self.model[idx].length

    def _select_model(self):
        if self.acc == 'TB':
            self.pymodpack = _pymod.tb
        elif self.acc == 'BO':
            self.pymodpack = _pymod.bo
        elif self.acc == 'TS':
            self.pymodpack = _pymod.ts
        elif self.acc == 'SI':
            self.pymodpack = _pymod.si

    def _select_magnets(self):
        slist = []
        if self.acc == 'TB':
            qlist = SetOptics.TB_QUADS
        elif self.acc == 'TS':
            qlist = SetOptics.TS_QUADS
        else:
            pvstr = self.acc + '-Fam:PS-'
            qlist = self.pymodpack.families.families_quadrupoles()
            slist = self.pymodpack.families.families_sextupoles()

        self.quad_list = [_PVName(pvstr+mag) for mag in qlist]
        self.sext_list = [_PVName(pvstr+mag) for mag in slist]

    def _create_devices(self):
        for mag in self.quad_list + self.sext_list:
            if mag not in self.devices:
                self.devices[mag] = _PowerSupply(mag)

    def get_applied_optics(self):
        """."""
        for mag in self.devices:
            self.applied_optics[mag.dev] = self.devices[mag].strength

    def compare_applied_model_optics(self):
        """."""
        for mag in self.devices:
            diff = self.applied_optics[mag] - self.optics_data[mag]
            diff /= self.optics_data[mag]
            if 'Q' in mag:
                print(
                    '{0:30s}: KL_applied={1:9.4f}1/m, KL_model={2:9.4f}1/m, \
                        diff={3:9.4f}%'.format(
                        mag, self.applied_optics[mag],
                        self.optics_data[mag], diff*100))
            if 'S' in mag:
                print(
                    '{0:30s}: SL_applied={1:9.4f}1/m², SL_model={2:9.4f}1/m², \
                        diff={3:9.4f}%'.format(
                        mag, self.applied_optics[mag],
                        self.optics_data[mag], diff*100))

    def _check_magtype(self, magtype):
        if magtype == 'quadrupole':
            mags = dict()
            for key in self.devices:
                if 'Q' not in key:
                    continue
                else:
                    mags[key] = self.devices[key]
        elif magtype == 'sextupole':
            if 'T' in self.acc:
                raise ValueError('transport lines do not have sextupoles')
            mags = dict()
            for key in self.devices:
                if 'S' not in key:
                    continue
                else:
                    mags[key] = self.devices[key]
        else:
            raise ValueError('magtype must be quadrupole or sextupole')
        return mags

    def minimize_spread(
            self, magtype, init=None, average=None, factor=0, apply=False):
        """."""
        mags = self._check_magtype(magtype)
        init0 = []
        goal = []
        for mag in mags:
            init0.append(mags[mag].strength)
            goal.append(self.optics_data[mag.devname])
        init0 = _np.asarray(init0)
        goal = _np.asarray(goal)

        init = init0 if init is None else init
        dperc = []
        print(
            '{:17s}  {:9s}  {:9s}  {:9s}%'.format(
                '', 'applied', 'nominal', 'diff'))
        for mag, stren in zip(mags, goal):
            dperc.append((mags[mag].strength-stren)/stren*100)
            print('{:17s}: {:9.6f}  {:9.6f}  {:9.6f}%'.format(
                mag.devname, mags[mag].strength, stren, dperc[-1]))

        print()
        average = _np.mean(dperc) if average is None else average
        print('average desired: {:+.4f} %'.format(average))
        print('average obtained: {:+.4f} %'.format(_np.mean(dperc)))
        print()

        ddif = _np.asarray(dperc) - average
        dimp_perc = factor/100 * (-ddif)
        implem = (1 + dimp_perc/100) * init

        print('-- to be implemented --')
        print('factor: {:5.1f}%'.format(factor))
        for mag, ini, imp in zip(mags, init, implem):
            perc = (imp - ini) / ini * 100
            print(
                '{:17s}:  {:9.4f} -> {:9.4f}  [{:7.4}%]'.format(
                    mags[mag].devname, ini, imp, perc))

        if apply:
            for mag, imp in zip(mags, implem):
                mags[mag].strength = imp
            print('\n applied!')
        return init
