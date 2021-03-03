"""."""

import time as _time

import numpy as _np
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs

from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupplyPU, Screen, SOFB

from ..utils import ThreadedMeasBaseClass as _BaseClass


class Params:
    """."""

    def __init__(self):
        """."""
        self.delay_span = 2  # delay in [us]
        self.num_points = 10
        self.wait_for_pm = 3
        self.num_mean_scrn = 20
        self.wait_mean_scrn = 0.5
        self.analysis_slice = slice(None)
        self.analysis_bpm_idx = 0
        self.analysis_scrn_pos_thres = 14  # [mm]

    def __str__(self):
        """."""
        st = '{0:30s} = {1:7.3f}\n'.format('delay_span [us]', self.delay_span)
        st += '{0:30s} = {1:9.4f}\n'.format(
            'num_points', self.num_points)
        st += '{0:30s} = {1:9.4f}\n'.format(
            'num_mean_scrn', self.num_mean_scrn)
        st += '{0:30s} = {1:9.4f}\n'.format(
            'wait_for_pm [s]', self.wait_for_pm)
        st += '{0:30s} = {1:9.4f}\n'.format(
            'wait_mean_scrn [s]', self.wait_mean_scrn)
        st += '{0:30s} = {1:s}'.format(
            'analysis_slice', str(self.analysis_slice))
        st += '{0:30s} = {1:d}\n'.format(
            'analysis_bpm_idx [s]', self.analysis_bpm_idx)
        st += '{0:30s} = {1:9.4f}\n'.format(
            'analysis_scrn_pos_thres [s]', self.analysis_scrn_pos_thres)
        return st


class DelaysPMBeamBasedSearch(_BaseClass):
    """."""

    PULSED_MAGNETS = PowerSupplyPU.DEVICES
    SCREENS = Screen.DEVICES

    def __init__(self, pulsed_mag, screen, sofb_acc='BO'):
        """."""
        super().__init__(params=Params(), target=self._findmax_thread)
        self._magname = _PVName(pulsed_mag)
        self._scrname = _PVName(screen)
        self.devices[self._magname] = PowerSupplyPU(self._magname)
        self.devices[screen] = Screen(self._scrname)
        self.devices['sofb'] = SOFB(sofb_acc.upper() + '-Glob:AP-SOFB')
        self.data = dict(
            delay=[], trajx=[], trajy=[], posx=[], posy=[], sizex=[], sizey=[])

    @property
    def magnet_name(self):
        """."""
        return self._magname

    @property
    def screen_name(self):
        """."""
        return self._scrname

    @property
    def sofb(self):
        """."""
        return self.devices['sofb']

    @property
    def screen(self):
        """."""
        return self.devices[self._scrname]

    @property
    def magnet(self):
        """."""
        return self.devices[self._magname]

    @property
    def trajx(self):
        """."""
        return self.data['trajx']

    @property
    def trajy(self):
        """."""
        return self.data['trajy']

    @property
    def posx(self):
        """."""
        return self.data['posx']

    @property
    def posy(self):
        """."""
        return self.data['posy']

    @property
    def sizex(self):
        """."""
        return self.data['sizex']

    @property
    def sizey(self):
        """."""
        return self.data['sizey']

    def _findmax_thread(self):
        """."""
        print(self._magname)
        delta = self.params.delay_span

        origdelay = self.magnet.delay
        rangedelay = _np.linspace(-delta/2, delta/2, self.params.num_points)
        self.data['delay'] = []
        self.data['trajx'] = []
        self.data['trajy'] = []
        self.data['posx'] = []
        self.data['posy'] = []
        self.data['sizex'] = []
        self.data['sizey'] = []
        for dly in rangedelay:
            print('{0:9.4f}: Measuring'.format(origdelay+dly), end='')
            self.magnet.delay = origdelay + dly
            _time.sleep(self.params.wait_for_pm)
            for _ in range(self.params.num_mean_scrn):
                print('.', end='')
                self.data['delay'].append(self.magnet.delay)
                self.data['trajx'].append(self.sofb.trajx)
                self.data['trajy'].append(self.sofb.trajy)
                self.data['posx'].append(self.screen.centerx)
                self.data['posy'].append(self.screen.centery)
                self.data['sizex'].append(self.screen.sigmax)
                self.data['sizey'].append(self.screen.sigmay)
                _time.sleep(self.params.wait_mean_scrn)
                if self._stopevt.is_set():
                    break
            print('done!')
            if self._stopevt.is_set():
                break
        self.magnet.delay = origdelay
        for k, v in self.data.items():
            self.data[k] = _np.array(v)
        self.data['time'] = _time.time()
        print('Finished!')

    def process_data(self):
        """."""
        data = self.data
        slc = self.params.analysis_slice
        bpm_idx = self.params.analysis_bpm_idx
        thres = self.params.analysis_scrn_pos_thres

        delays = data['delay'][slc]
        posbx = _np.array(data['trajx'][slc, bpm_idx])/1e3
        posby = _np.array(data['trajy'][slc, bpm_idx])/1e3
        possx = _np.array(data['posx'][slc])
        possy = _np.array(data['posy'][slc])
        sizex = _np.array(data['sizex'][slc])
        sizey = _np.array(data['sizey'][slc])

        idx = (_np.abs(possx) < thres) & (_np.abs(possy) < thres)
        dtas = (posbx, posby, possx, possy, sizex, sizey)
        labs = ('posbx', 'posby', 'possx', 'possy', 'sizex', 'sizey')
        for lab, dta in zip(labs, dtas):
            dta = dta[idx]
            pol = _np.polynomial.polynomial.polyfit(delays, dta, 2)
            dly = -1*pol[1]/pol[2]/2
            fit = _np.polynomial.polynomial.polyval(pol, delays)
            self.analysis[f'pol_{lab}'] = pol
            self.analysis[f'delay_{lab}'] = dly
            self.analysis[f'fit_{lab}'] = fit

    def plot_data(self):
        """."""
        fig = _mplt.figure(figsize=(9.5, 12))
        gs = _mgs.GridSpec(3, 2)
        gs.update(left=0.12, right=0.88, top=0.94, wspace=0.15, vspace=0.2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
        ay1 = fig.add_subplot(gs[0, 1], sharex=ax1)
        ay2 = fig.add_subplot(gs[1, 1], sharex=ax1)
        ay3 = fig.add_subplot(gs[2, 1], sharex=ax1)

        data = self.data
        anl = self.analysis
        slc = self.params.analysis_slice
        bpm_idx = self.params.analysis_bpm_idx

        delays = data['delay'][slc]
        posbx = _np.array(data['trajx'][slc, bpm_idx])/1e3
        posby = _np.array(data['trajy'][slc, bpm_idx])/1e3
        possx = _np.array(data['posx'][slc])
        possy = _np.array(data['posy'][slc])
        sizex = _np.array(data['sizex'][slc])
        sizey = _np.array(data['sizey'][slc])

        fig.suptitle(self.magnet_name)

        ax1.plot(delays, posbx, 'bo', label='Data')
        ax1.plot(delays, anl['fit_posbx'], '-b', label='Fitting')
        ax1.axvline(anl['delay_posbx'], linestyle='--', color='k')
        ax1.set_ylabel(f'X centroid @ BPM {bpm_idx:d} [mm]')
        ax1.legend()

        ay1.plot(delays, posby, 'ro', label='Data')
        ay1.plot(delays, anl['fit_posby'], '-r', label='Fitting')
        ay1.set_ylabel(f'Y centroid @ BPM {bpm_idx:d} [mm]')
        ay1.yaxis.set_label_position("right")
        ay1.yaxis.tick_right()

        ax2.plot(delays, possx, 'bo', label='Data')
        ax2.plot(delays, anl['fit_possx'], '-b', label='Fitting')
        ax2.axvline(anl['delay_possx'], linestyle='--', color='k')
        ax2.set_ylabel(f'X centroid @ {self.screen_name:s} [mm]')

        ay2.plot(delays, possy, 'ro', label='Data')
        ay2.plot(delays, anl['fit_possy'], '-r', label='Fitting')
        ay2.set_ylabel(f'Y centroid @ {self.screen_name:s} [mm]')
        ay2.yaxis.set_label_position("right")
        ay2.yaxis.tick_right()

        ax3.plot(delays, sizex, 'bo', label='Data')
        ax3.plot(delays, anl['fit_sizex'], '-b', label='Fitting')
        ax3.axvline(anl['delay_sizex'], linestyle='--', color='k')
        ax3.set_ylabel(f'X size @ {self.screen_name:s} [mm]')

        ay3.plot(delays, sizey, 'ro', label='Data')
        ay3.plot(delays, anl['fit_sizey'], '-r', label='Fitting')
        ay3.set_ylabel(f'Y size @ {self.screen_name:s} [mm]')
        ay3.yaxis.set_label_position("right")
        ay3.yaxis.tick_right()

        ax3.set_xlabel('Delay [us]')
        ay3.set_xlabel('Delay [us]')
        _mplt.setp(ax1.get_xticklabels(), visible=False)
        _mplt.setp(ay1.get_xticklabels(), visible=False)
        _mplt.setp(ax2.get_xticklabels(), visible=False)
        _mplt.setp(ay2.get_xticklabels(), visible=False)
        return fig
