#!/usr/bin/env python-sirius
"""."""

import time as _time
import numpy as np
from siriuspy.epics import PV
from siriuspy.devices import DCCT, SOFB

from apsuite.optimization import PSO, SimulAnneal


class Septum:
    """."""

    def __init__(self):
        """."""
        self.sp = 'TB-04:PM-InjSept:Kick-SP'
        self.rb = 'TB-04:PM-InjSept:Kick-RB'


class Kicker:
    """."""

    def __init__(self):
        """."""
        self.sp = 'BO-01D:PM-InjKckr:Kick-SP'
        self.rb = 'BO-01D:PM-InjKckr:Kick-RB'


class Corrs:
    """."""

    def __init__(self):
        """."""
        names = ['TB-04:MA-CH-1', 'TB-04:MA-CV-1', 'TB-04:MA-CV-2']
        self.sp = [c + ':Kick-SP' for c in names]
        self.rb = [c + ':Kick-RB' for c in names]


class Quads:
    """."""

    def __init__(self):
        """."""
        names = [
            'TB-02:MA-QF2A', 'TB-02:MA-QF2B',
            'TB-02:MA-QD2A', 'TB-02:MA-QD2B'
        ]
        self.sp = [c + ':KL-SP' for c in names]
        self.rb = [c + ':KL-RB' for c in names]


class Params:
    """."""

    def __init__(self):
        """."""
        self.deltas = {'Quads': 1, 'Corrs': 1000, 'InjSept': 2, 'InjKckr': 2}
        self.niter = 10
        self.nbuffer = 10
        self.nturns = 1
        self.nbpm = 50
        self.wait_change = 5
        self.dcct_nrsamples = 50
        self.dcct_period = 0.05
        self.dcct_timeout = 10
        self.freq = 2


class PSOInjection(PSO):
    """."""

    def __init__(self, save=False):
        """."""
        self.reference = []
        self.eyes = []
        self.hands = []
        self.f_init = 0
        self.params = Params()
        self.sofb = SOFB(SOFB.DEVICES.BO)
        self.dcct = DCCT(DCCT.DEVICES.BO)
        self.quads = Quads()
        self.corrs = Corrs()
        self.kckr = Kicker()
        self.sept = Septum()
        PSO.__init__(self, save=save)

    def initialization(self):
        """."""
        self.niter = self.params.niter
        self.nr_turns = self.params.nturns
        self.nr_bpm = self.params.nbpm
        self.bpm_idx = self.nr_bpm + 50 * (self.nr_turns - 1)

        self.get_pvs()

        while True:
            if self.check_connect():
                break

        self.sofb.nr_points = self.params.nbuffer

        quad_lim = np.ones(len(self.quads.sp)) * self.params.deltas['Quads']
        corr_lim = np.ones(len(self.corrs.sp)) * self.params.deltas['Corrs']
        sept_lim = np.array([self.params.deltas['InjSept']])
        kckr_lim = np.array([self.params.deltas['InjKckr']])

        up = np.concatenate((quad_lim, corr_lim, sept_lim, kckr_lim))
        down = -1 * up
        self.set_limits(upper=up, lower=down)

        self.dcct.cmd_turn_off(self.params.dcct_timeout)
        self.dcct.nrsamples = self.params.dcct_nrsamples
        self.dcct.period = self.params.dcct_period
        self.dcct.cmd_turn_on(self.params.dcct_timeout)

        self.reference = np.array([h.value for h in self.hands])
        # self.reset_wait_buffer()
        self.init_obj_func()

    def get_pvs(self):
        """."""
        # self.eyes = self.sofb.sum
        self.eyes = self.dcct.current_fast

        self.hands = [PV(c) for c in self.corrs.sp]
        self.hands.append(PV(self.kckr.sp))
        self.hands.append(PV(self.sept.sp))

    def check_connect(self):
        """."""
        conh = [h.connected for h in self.hands]
        cone = self.eyes.connected
        if cone and sum(conh) == len(conh):
            con = True
        else:
            con = False
        return con

    def get_change(self, part):
        """."""
        return self.reference + self.position[part, :]

    def set_change(self, change):
        """."""
        for k in range(len(self.hands)):
            self.hands[k].value = change[k]

    def reset_wait_buffer(self):
        """."""
        self.sofb.cmd_reset()
        self.sofb.wait_buffer()

    def init_obj_func(self):
        """."""
        # self.f_init = -np.sum(self.eyes.value[:self.bpm_idx])
        pulse_cnt = []
        for _ in range(self.params.nbuffer):
            pulse_cnt.append(np.mean(self.eyes))
            _time.sleep(1/self.params.freq)
        self.f_init = -np.mean(pulse_cnt)

    def calc_obj_fun(self):
        """."""
        f_out = np.zeros(self.nswarm)
        for i in range(self.nswarm):
            pulse_cnt = []
            self.set_change(self.get_change(i))
            _time.sleep(self.params.wait_change)
            for _ in range(self.params.nbuffer):
                pulse_cnt.append(np.mean(self.eyes))
                _time.sleep(1/self.params.freq)
            # self.reset_wait_buffer()
            # f_out[i] = np.sum(self.eyes.value[:self.bpm_idx])
            f_out[i] = np.mean(pulse_cnt)
            print(
                'Particle {:02d}/{:d} | Obj. Func. : {:f}'.format(
                    i+1, self.nswarm, f_out[i]))
        return - f_out


class SAInjection(SimulAnneal):
    """."""

    def __init__(self, save=False):
        """."""
        self.reference = []
        self.eyes = []
        self.hands = []
        self.f_init = 0
        self.params = Params()
        self.dcct = DCCT(DCCT.DEVICES.BO)
        self.sofb = SOFB(SOFB.DEVICES.BO)
        self.quads = Quads()
        self.corrs = Corrs()
        self.kckr = Kicker()
        self.sept = Septum()
        SimulAnneal.__init__(self, save=save)

    def initialization(self):
        """."""
        self.niter = self.params.niter
        self.nr_turns = self.params.nturns
        self.nr_bpm = self.params.nbpm
        self.bpm_idx = self.nr_bpm + 50 * (self.nr_turns - 1)

        self.get_pvs()

        while True:
            if self.check_connect():
                break

        quad_lim = np.ones(len(self.quads.sp)) * self.params.deltas['Quads']
        corr_lim = np.ones(len(self.corrs.sp)) * self.params.deltas['Corrs']
        sept_lim = np.array([self.params.deltas['InjSept']])
        kckr_lim = np.array([self.params.deltas['InjKckr']])

        up = np.concatenate((quad_lim, corr_lim, sept_lim, kckr_lim))
        # down = -1 * up
        infty = float('Inf')
        self.set_limits(upper=up*infty, lower=-1*up*infty)
        self.set_deltas(dmax=up)

        self.reference = np.array([h.value for h in self.hands])
        self.position = self.reference
        # self.reset_wait_buffer()
        self.init_obj_func()

    def get_pvs(self):
        """."""
        # self.eyes = self.sofb.sum
        self.eyes = self.dcct.current_fast

        self.hands = [PV(c) for c in self.corrs.sp]
        self.hands.append(PV(self.kckr.sp))
        self.hands.append(PV(self.sept.sp))

    def check_connect(self):
        """."""
        conh = [h.connected for h in self.hands]
        cone = self.eyes.connected
        if cone and sum(conh) == len(conh):
            con = True
        else:
            con = False
        return con

    def get_change(self):
        """."""
        return self.position

    def set_change(self, change):
        """."""
        for k in range(len(self.hands)):
            self.hands[k].value = change[k]

    def reset_wait_buffer(self):
        """."""
        self.sofb.cmd_reset()
        self.sofb.wait_buffer()

    def init_obj_func(self):
        """."""
        # self.f_init = -np.sum(self.eyes.value[:self.bpm_idx])
        pulse_cnt = []
        for _ in range(self.params.nbuffer):
            pulse_cnt.append(np.mean(self.eyes))
            _time.sleep(1/self.params.freq)
        self.f_init = -np.mean(pulse_cnt)

    def calc_obj_fun(self):
        """."""
        f_out = []
        pulse_cnt = []
        self.set_change(self.get_change())
        _time.sleep(self.params.wait_change)
        # self.reset_wait_buffer()
        for _ in range(self.params.nbuffer):
            pulse_cnt.append(np.mean(self.eyes))
            _time.sleep(1/self.params.freq)
        # f_out = np.sum(self.eyes.value[:self.bpm_idx])
        f_out = np.mean(pulse_cnt)
        return - f_out
