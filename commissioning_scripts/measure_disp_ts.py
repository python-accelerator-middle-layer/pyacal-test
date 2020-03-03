#!/usr/bin/env python-sirius
"""."""

import time as _time
import numpy as np

from epics import PV
import pyaccel

from siriuspy.devices import SOFB, RF
from apsuite.commissioning_scripts.base import BaseClass


class ParamsDisp:
    """."""

    def __init__(self):
        """."""
        self.energy_delta = 0.005  # in GeV
        self.wait_time = 2
        self.timeout_orb = 10
        self.num_points = 10
        self.delay2energy = 0.4923/48994.304  # [GeV/us]
        self.wait_update_events = 2.0  # [s]

    @property
    def ejection_delta(self):
        return self.energy_delta / self.delay2energy  # in us

    def __str__(self):
        """."""
        strt = ''
        strt += 'energy_delta: ' + str(self.energy_delta) + ' GeV \n'
        strt += 'wait_time: ' + str(self.wait_time) + ' s\n'
        strt += 'timeout_orb: ' + str(self.timeout_orb) + ' s\n'
        strt += 'num_points: ' + str(self.num_points) + '\n'
        strt += 'delay2energy: ' + str(self.delay2energy) + ' GeV/us\n'
        strt += 'wait_update_events: ' + str(self.wait_update_events) + ' s\n'
        return strt


class MeasureDispTS(BaseClass):
    """."""

    HARMONIC_NUM = 828

    def __init__(self):
        """."""
        super().__init__(ParamsDisp())
        self.devices = {
            'ts_sofb': SOFB('TS'),
            'bo_sofb': SOFB('BO'),
            'rf': RF()
            }
        self.pvs = {
            'injsi_sp': PV('AS-RaMO:TI-EVG:InjSIDelay-SP'),
            'injsi_rb': PV('AS-RaMO:TI-EVG:InjSIDelay-RB'),
            'digts_sp': PV('AS-RaMO:TI-EVG:DigTSDelay-SP'),
            'digts_rb': PV('AS-RaMO:TI-EVG:DigTSDelay-RB'),
            'update_evt': PV('AS-RaMO:TI-EVG:UpdateEvt-Cmd'),
            }

    @property
    def energy(self):
        """."""
        return self.params.delay2energy * self.injsi

    @property
    def trajx(self):
        """."""
        return self.devices['ts_sofb'].trajx

    @property
    def trajx_bo(self):
        """."""
        return self.devices['bo_sofb'].trajx_idx

    @property
    def trajy(self):
        """."""
        return self.devices['ts_sofb'].trajy

    @property
    def trajy_bo(self):
        """."""
        return self.devices['bo_sofb'].trajy_idx

    @property
    def injsi(self):
        """."""
        return self.pvs['injsi_rb'].value

    @injsi.setter
    def injsi(self, value):
        self.pvs['injsi_sp'].value = value

    @property
    def digts(self):
        """."""
        return self.pvs['digts_rb'].value

    @digts.setter
    def digts(self, value):
        """."""
        self.pvs['digts_sp'].value = value

    def update_events(self):
        """."""
        self.pvs['update_evt'].value = 1

    @property
    def nr_points(self):
        """."""
        return self.devices['ts_sofb'].nr_points

    @nr_points.setter
    def nr_points(self, value):
        self.devices['ts_sofb'].nr_points = int(value)
        self.devices['bo_sofb'].nr_points = int(value)

    def wait(self, timeout=10):
        """."""
        self.devices['ts_sofb'].wait(timeout=timeout)
        self.devices['bo_sofb'].wait(timeout=timeout)

    def reset(self, wait=0):
        """."""
        _time.sleep(wait)
        self.devices['ts_sofb'].reset()
        self.devices['bo_sofb'].reset()
        _time.sleep(1)

    def calc_delta(self, delta):
        """."""
        # delta and revolution time in [us]
        t0 = self.HARMONIC_NUM/self.devices['rf'].frequency * 1e6
        return round(delta/t0)*t0

    def measure_dispersion(self):
        """."""
        print('calc delta time')
        self.nr_points = self.params.num_points
        delta = self.calc_delta(delta=self.params.ejection_delta)

        print('reset sofb')
        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_orb)

        print('read orbit')
        orb = [-np.hstack([self.trajx, self.trajy]), ]
        orb_bo = [-np.hstack([self.trajx_bo, self.trajy_bo]), ]
        ene0 = self.energy

        print('set delays')
        orig_delay_injsi = self.injsi
        orig_delay_digts = self.digts
        self.injsi = orig_delay_injsi + delta
        self.digts = orig_delay_digts + delta
        _time.sleep(self.params.wait_update_events)

        print('update events')
        self.update_events()
        _time.sleep(self.params.wait_update_events)

        print('reset sofb')
        self.reset(self.params.wait_time)
        self.wait(self.params.timeout_orb)

        print('read orbit')
        orb.append(np.hstack([self.trajx, self.trajy]))
        orb_bo.append(np.hstack([self.trajx_bo, self.trajy_bo]))
        ene1 = self.energy

        print('restore original delays')
        self.injsi = orig_delay_injsi
        self.digts = orig_delay_digts
        _time.sleep(self.params.wait_update_events)
        self.update_events()
        _time.sleep(self.params.wait_update_events)

        d_ene = ene1/ene0 - 1
        return np.array(orb).sum(axis=0) / d_ene, \
               np.array(orb_bo).sum(axis=0) / d_ene


def calc_model_dispersionTS(model, bpms):
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
