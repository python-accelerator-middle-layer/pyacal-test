#!/usr/bin/env python-sirius
"""."""

import time as _time
import numpy as np
from siriuspy.epics import PV
from apsuite.optimization import PSO


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


class SOFB:
    """."""

    def __init__(self):
        """."""
        self.nsample = 'BO-Glob:AP-SOFB:TrigNrSamplesPost-SP'
        self.sumsignal = 'BO-Glob:AP-SOFB:MTurnSum-Mon'
        self.nbuffer_sp = 'BO-Glob:AP-SOFB:SmoothNrPts-SP'
        self.nbuffer_rb = 'BO-Glob:AP-SOFB:SmoothNrPts-RB'
        self.nbuffer_mon = 'BO-Glob:AP-SOFB:BufferCount-Mon'
        self.nbuffer_reset = 'BO-Glob:AP-SOFB:SmoothReset-Cmd'


class Params:
    """."""

    def __init__(self):
        """."""
        self.deltas = {'Corrs': 1000, 'InjSept': 2, 'InjKckr': 2}
        self.niter = 10
        self.nbuffer = 10
        self.nturns = 1
        self.nbpm = 50
        self.wait_change = 5


class PSOInjection(PSO):
    """."""

    def __init__(self, save=False):
        """."""
        self.reference = []
        self.eyes = []
        self.hands = []
        self.pv_nr_pts_sp = []
        self.pv_nr_pts_rb = []
        self.pv_buffer_mon = []
        self.pv_buffer_reset = []
        self.pv_nr_sample = []
        self.f_init = 0
        self.params = Params()
        self.sofb = SOFB()
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

        self.pv_nr_pts_sp.value = self.params.nbuffer

        corrs = Corrs()
        corr_lim = np.ones(len(corrs.sp)) * self.params.deltas['Corrs']
        sept_lim = np.array([self.params.deltas['InjSept']])
        kckr_lim = np.array([self.params.deltas['InjKckr']])

        up = np.concatenate((corr_lim, sept_lim, kckr_lim))
        down = -1 * up
        self.set_limits(upper=up, lower=down)

        self.reference = np.array([h.value for h in self.hands])
        self.reset_wait_buffer()
        self.init_obj_func()

    def get_pvs(self):
        """."""
        self.pv_nr_sample = PV(self.sofb.nsample)
        _time.sleep(self.params.wait_change)
        self.pv_nr_sample.value = self.params.nturns

        self.eyes = PV(self.sofb.sumsignal, auto_monitor=True)

        self.hands = [PV(c) for c in self.corrs.sp]
        self.hands.append(PV(self.kckr.sp))
        self.hands.append(PV(self.sept.sp))

        self.pv_nr_pts_sp = PV(self.sofb.nbuffer_sp)
        self.pv_nr_pts_rb = PV(self.sofb.nbuffer_rb)
        self.pv_buffer_mon = PV(self.sofb.nbuffer_mon)
        self.pv_buffer_reset = PV(self.sofb.nbuffer_reset)

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
        self.pv_buffer_reset.value = 1
        _time.sleep(self.params.wait_change)

        while True:
            if self.pv_buffer_mon.value == self.pv_nr_pts_rb.value:
                break

    def init_obj_func(self):
        """."""
        self.f_init = -np.sum(self.eyes.value[:self.bpm_idx])

    def calc_obj_fun(self):
        """."""
        f_out = np.zeros(self.nswarm)

        for i in range(self.nswarm):
            self.set_change(self.get_change(i))
            _time.sleep(self.params.wait_change)
            self.reset_wait_buffer()
            f_out[i] = np.sum(self.eyes.value[:self.bpm_idx])
            print(
                'Particle {:02d}/{:d} | Obj. Func. : {:f}'.format(
                    i+1, self.nswarm, f_out[i]))
        return - f_out
