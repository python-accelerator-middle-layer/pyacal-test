#!/usr/bin/env python-sirius
"""."""

import time as _time
from epics import PV as _PV
import numpy as _np
import matplotlib.pyplot as plt
from optimization import PSO


class PSOInjection(PSO):
    """."""

    def __init__(self):
        """."""
        self.reference = []
        self._nswarm = []
        self.niter = []
        self.nr_turns = []
        self._name_hands = []
        self._name_quads = []
        self._name_corrs = []
        self._name_kckr = []
        self.eyes = []
        self.hands = []
        self.pv_nr_pts_sp = []
        self.pv_nr_pts_rb = []
        self.pv_buffer_mon = []
        self.pv_buffer_reset = []
        self.pv_nr_sample = []
        self._wait_change = []
        self._upper_limits = _np.array([])
        self._lower_limits = _np.array([])
        self._ndim = int(0)
        self.f_init = 0
        PSO.__init__(self)

    def initialization(self):
        """."""
        print('========================================================')
        p_quad = input('Set TB Quads Variation (Default = 10%): ')
        p_corr = input('Set TB Corrs Variation (Default = 25%): ')
        p_kick = input('Set Inj Kicker Variation (Default = 5%): ')
        print('========================================================')
        n_iter = input('Set Number of Iteractions (Default = 20): ')
        nr_pts = input('Set Buffer Size (SOFB) (Default = 10): ')
        nr_turns = input(
            'Set Number of Turns to measure Sum Signal (Default = 1): ')
        print('========================================================')

        if not n_iter:
            n_iter = 20
        if not nr_pts:
            nr_pts = 10
        if not nr_turns:
            nr_turns = 1
        if not nr_pts:
            nr_pts = 10
        if not p_quad:
            p_quad = 10
        if not p_corr:
            p_corr = 25
        if not p_kick:
            p_kick = 5

        self.niter = int(n_iter)
        self.nr_turns = int(nr_turns)
        self._wait_change = 1
        self.set_pvs()
        print('Waiting for PVs connection...')
        print('========================================================')
        while True:
            if self.check_connect():
                break

        self.pv_nr_pts_sp.value = int(nr_pts)

        quad_lim = _np.ones(len(self._name_quads)) * float(p_quad) / 100
        corr_lim = _np.ones(len(self._name_corrs)) * float(p_corr) / 100
        kckr_lim = _np.ones(len(self._name_kckr)) * float(p_kick) / 100
        self._upper_limits = _np.concatenate(
            (quad_lim, corr_lim, kckr_lim))
        self._lower_limits = -1 * self._upper_limits
        self._ndim = int(len(self._upper_limits))
        self._nswarm = 10 + 2 * int(_np.sqrt(self._ndim))

        print(
            'The script will take {:.2f} minutes to be finished'.format(
                (float(nr_pts)*0.5 + 3*self._wait_change) *
                self._nswarm * (self.niter+1) / 60))
        print('========================================================')

        self.reference = _np.array([h.value for h in self.hands])

        self.reset_wait_buffer()
        self.f_init = _np.sum(self.eyes.value)
        print('Inital Objective Function {:.5f}'.format(-self.f_init))
        print('========================================================')

    def set_pvs(self):
        """."""
        # Number of turns to measure Sum Signal
        self.pv_nr_sample = _PV('BO-Glob:AP-SOFB:TrigNrSamplesPost-SP')
        _time.sleep(self._wait_change)
        self.pv_nr_sample.value = int(self.nr_turns)

        # Diagnostic to calculate Objective Function
        self.eyes = _PV('BO-Glob:AP-SOFB:MTurnSum-Mon')

        self._name_quads = [
            'TB-01:MA-QF1', 'TB-01:MA-QD1',
            'TB-02:MA-QF2A', 'TB-02:MA-QD2A',
            'TB-02:MA-QF2B', 'TB-02:MA-QD2B',
            'TB-03:MA-QF3', 'TB-03:MA-QD3',
            'TB-04:MA-QF4', 'TB-04:MA-QD4'
        ]
        self._name_quads = [q + ':KL-SP' for q in self._name_quads]
        self._name_hands.extend(self._name_quads)

        self._name_corrs = [
            'TB-04:MA-CH-1', 'TB-04:MA-CV-1',
            'TB-04:MA-CH-2', 'TB-04:MA-CV-2',
        ]
        self._name_corrs = [c + ':Kick-SP' for c in self._name_corrs]
        self._name_hands.extend(self._name_corrs)

        self._name_kckr = ['BO-01D:PM-InjKckr:Kick-SP']
        self._name_hands.extend(self._name_kckr)

        # Actuator to change settings
        self.hands = [_PV(h) for h in self._name_hands]

        self.pv_nr_pts_sp = _PV('BO-Glob:AP-SOFB:SmoothNrPts-SP')
        self.pv_nr_pts_rb = _PV('BO-Glob:AP-SOFB:SmoothNrPts-RB')
        self.pv_buffer_mon = _PV('BO-Glob:AP-SOFB:BufferCount-Mon')
        self.pv_buffer_reset = _PV('BO-Glob:AP-SOFB:SmoothReset-Cmd')

    def check_connect(self):
        """."""
        con = [h.connected for h in self.hands]
        if sum(con) < len(con):
            con = False
        else:
            con = True
        return con

    def get_change(self, part):
        """."""
        return self.reference * (1 + 0*self._position[part, :])

    def set_change(self, change):
        """."""
        for k in range(len(self.hands)):
            self.hands[k].value = change[k]

    def reset_wait_buffer(self):
        """."""
        self.pv_buffer_reset.value = 1
        _time.sleep(self._wait_change)

        while True:
            if self.pv_buffer_mon.value == self.pv_nr_pts_rb.value:
                break

    def calc_merit_function(self):
        """."""
        f_out = _np.zeros(self._nswarm)

        for i in range(self._nswarm):
            chg = self.get_change(i)
            self.set_change(chg)
            _time.sleep(self._wait_change)
            self.reset_wait_buffer()
            f_out[i] = _np.sum(self.eyes.value)
            print(
                'Particle {:02d}/{:d} | Obj. Fun. : {:f}'.format(
                    i+1, self._nswarm, f_out[i]))
        print('========================================================')
        return - f_out

    def run(self):
        """."""
        pos, fig = self.start_optimization(niter=self.niter)
        plt.plot(fig, '-o')
        plt.xlabel('Number of Iteractions')
        plt.ylabel('Objective Function')
        plt.show()
        print(
            'The Objective Function changed from {:.5f} to {:.5f}'.format(
                self.f_init, -fig[-1]))
        return pos, fig


if __name__ == "__main__":
    opt_inj = PSOInjection()
    opt_inj.run()
