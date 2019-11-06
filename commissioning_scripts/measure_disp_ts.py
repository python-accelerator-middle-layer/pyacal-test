#!/usr/bin/env python-sirius
"""."""

import time as _time
import numpy as np

from epics import PV
import pyaccel

from pymodels.middlelayer.devices import SOFB, RF
from apsuite.commissioning_scripts.base import BaseClass


class ParamsDisp:
    """."""

    def __init__(self):
        """."""
        self.energy_delta = 0.005  # in GeV
        self.wait_time = 40
        self.timeout_orb = 10
        self.num_points = 10
        self.delay2energy = 0.4923/489994.304  # [GeV/us]

    @property
    def ejection_delta(self):
        return self.energy_delta / self.delay2energy  # in us


class MeasureDispTBBO(BaseClass):
    """."""

    HARMONIC_NUM = 828

    def __init__(self):
        """."""
        super().__init__(ParamsDisp())
        self.devices = {
            'ts_sofb': SOFB('TS'),
            'rf': RF()
            }
        self.pvs = {
            'injsi': PV('AS-RaMO:TI-EVG:InjSIDelay-SP'),
            'digts': PV('AS-RaMO:TI-EVG:DigTSDelay-SP'),
            'update_evt': PV('AS-RaMO:TI-EVG:UpdateEvt-Cmd'),
            }

    @property
    def energy(self):
        """."""
        return self.params.delay2energy * self.pvs['injsi'].value

    @property
    def trajx(self):
        """."""
        return self.devices['ts_sofb'].trajx

    @property
    def trajy(self):
        """."""
        return self.devices['ts_sofb'].trajy

    @property
    def nr_points(self):
        """."""
        return self.devices['ts_sofb'].nr_points

    @nr_points.setter
    def nr_points(self, value):
        self.devices['ts_sofb'].nr_points = int(value)

    def wait(self, timeout=10):
        """."""
        self.devices['ts_sofb'].wait(timeout=timeout)

    def reset(self, wait=0):
        """."""
        _time.sleep(wait)
        self.devices['ts_sofb'].reset()
        _time.sleep(1)

    def update_events(self):
        self.pvs['update_evt'].value = 1

    def calc_delta(self, delta):
        # delta and revolution time in [us]
        t0 = self.HARMONIC_NUM/self.devices['rf'].frequency * 1e6
        return round(delta/t0)*t0

    def measure_dispersion(self):
        """."""
        self.nr_points = self.params.num_points
        delta = self.calc_delta(delta=self.params.ejection_delta)

        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_orb)
        orb = [-np.hstack([self.trajx, self.trajy]), ]
        ene0 = self.energy

        orig_delay = self.pvs['injsi'].value
        orig_delay_digts = self.pvs['digts'].value
        self.pvs['injsi'].value = orig_delay + delta
        self.pvs['digts'].value = orig_delay_digts + delta

        self.update_events()
        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_orb)

        orb.append(np.hstack([self.trajx, self.trajy]))
        ene1 = self.energy

        self.pvs['injsi'].value = orig_delay
        self.pvs['digts'].value = orig_delay_digts
        self.update_events()

        d_ene = ene1/ene0 - 1
        return np.array(orb).sum(axis=0) / d_ene


def calc_model_dispersionTBBO(model, bpms):
    """."""
    dene = 1e-3
    rout, *_ = pyaccel.tracking.line_pass(
        model,
        [[0, 0, 0, 0, dene/2, 0],
         [0, 0, 0, 0, -dene/2, 0]],
        bpms)
    dispx = (rout[0, 0, :] - rout[1, 0, :]) / dene
    dispy = (rout[0, 2, :] - rout[1, 2, :]) / dene
    return np.hstack([dispx, dispy])
