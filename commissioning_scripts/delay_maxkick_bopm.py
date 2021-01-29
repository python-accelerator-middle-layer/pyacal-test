"""."""

import time as _time
from threading import Thread as _Thread, Event as _Event

import numpy as _np

from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupplyPU, Screen, SOFB

from .base import BaseClass


class Params:
    """."""

    def __init__(self):
        """."""
        self.delay_span = 2  # delay in [us]
        self.num_points = 10
        self.wait_for_pm = 3
        self.num_mean_scrn = 20
        self.wait_mean_scrn = 0.5

    def __str__(self):
        """."""
        st = '{0:30s} = {1:7.3f}\n'.format('Delay Span [us]', self.delay_span)
        st += '{0:30s} = {1:9.4f}\n'.format(
            'Scan Num. of points', self.num_points)
        st += '{0:30s} = {1:9.4f}\n'.format(
            'Mean Num. of points', self.num_mean_scrn)
        st += '{0:30s} = {1:9.4f}\n'.format(
            'Wait for PM [s]', self.wait_for_pm)
        st += '{0:30s} = {1:9.4f}\n'.format(
            'Wait Time [s]', self.wait_mean_scrn)
        return st


class FindBeamBasedPMDelays(BaseClass):
    """."""

    def __init__(self, pulsed_mag, screen, sofb_acc='BO'):
        """."""
        super().__init__(params=Params())
        self._magname = _PVName(pulsed_mag)
        self._scrname = _PVName(screen)
        self.devices[self._magname] = PowerSupplyPU(self._magname)
        self.devices[screen] = Screen(self._scrname)
        self.devices['sofb'] = SOFB(sofb_acc.upper() + '-Glob:AP-SOFB')
        self.data = dict(
            delay=[], trajx=[], trajy=[], posx=[], posy=[], sizex=[], sizey=[])
        self._thread = _Thread(target=self._findmax_thread)
        self._stopped = _Event()

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

    def start(self):
        """."""
        if not self._thread.is_alive():
            self._thread = _Thread(
                target=self._findmax_thread, daemon=True)
            self._stopped.clear()
            self._thread.start()

    @property
    def ismeasuring(self):
        """."""
        return self._thread.is_alive()

    def stop(self):
        """."""
        self._stopped.set()

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
                if self._stopped.is_set():
                    break
                self.data['delay'].append(self.magnet.delay)
                self.data['trajx'].append(self.sofb.trajx)
                self.data['trajy'].append(self.sofb.trajy)
                self.data['posx'].append(self.screen.centerx)
                self.data['posy'].append(self.screen.centery)
                self.data['sizex'].append(self.screen.sigmax)
                self.data['sizey'].append(self.screen.sigmay)
                _time.sleep(self.params.wait_mean_scrn)
                if self._stopped.is_set():
                    break
            print('done!')
            if self._stopped.is_set():
                break
        self.magnet.delay = origdelay
        print('Finished!')
