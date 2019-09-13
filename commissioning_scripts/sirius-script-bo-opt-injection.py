#!/usr/bin/env python-sirius
"""."""

import time as _time
from epics import PV as _PV
import numpy as _np
import matplotlib.pyplot as plt
from apsuite.optimization import PSO, SimulAnneal


class PSOInjection(PSO):
    """."""

    def __init__(self, save=False):
        """."""
        self.reference = []
        self.niter = 0
        self.nr_turns = 0
        self.nr_bpm = 0
        self.nswarm = 0
        self.bpm_idx = 0
        self._name_hands = []
        self._name_quads = []
        self._name_corrs = []
        self._name_sept = []
        self._name_kckr = []
        self.eyes = []
        self.hands = []
        self.pv_nr_pts_sp = []
        self.pv_nr_pts_rb = []
        self.pv_buffer_mon = []
        self.pv_buffer_reset = []
        self.pv_nr_sample = []
        self._wait_change = 0
        self.f_init = 0
        PSO.__init__(self, save=save)

    def initialization(self):
        """."""
        print('='*50)
        d_quad = input(
            'TB Quads KL Variation (Default = 1 [1/m]): ')
        d_corr = input(
            'TB Corrs Kick Variation (Default = 1000 [urad]): ')
        d_sept = input('InjSept Variation (Default = 2 [mrad]): ')
        d_kckr = input('InjKicker Variation (Default = 2 [mrad]): ')
        print('='*50)
        nr_iter = input('Number of Iteractions (Default = 10): ')
        nr_swarm = input('Swarm Size (Default = 10 + 2 * sqrt(D)): ')
        nr_pts = input('Buffer Size (SOFB) (Default = 10): ')
        nr_turns = input(
            'Number of Turns to Measure Sum Signal (Default = 1): ')
        nr_bpm = input(
            'Set the last BPM to Read ' +
            '(Default = 50, Range = [1,50]): ')
        print('='*50)

        if not nr_iter:
            nr_iter = 10
        if not nr_swarm:
            nr_swarm = 0
        if not nr_pts:
            nr_pts = 10
        if not nr_bpm:
            nr_bpm = 50
        if not nr_turns:
            nr_turns = 1
        if not nr_pts:
            nr_pts = 10
        if not d_quad:
            d_quad = 1
        if not d_corr:
            d_corr = 1000
        if not d_kckr:
            d_kckr = 2
        if not d_sept:
            d_sept = 2

        d_quad = float(d_quad)
        d_corr = float(d_corr)
        d_sept = float(d_sept)
        d_kckr = float(d_kckr)
        nr_iter = int(nr_iter)
        nr_swarm = int(nr_swarm)
        nr_pts = int(nr_pts)
        nr_turns = int(nr_turns)
        nr_bpm = int(nr_bpm)

        if not d_quad + d_corr + d_sept + d_kckr:
            raise Exception('You have set zero variation for all dimensions!')

        self.niter = nr_iter
        self.nswarm = nr_swarm
        self.nr_turns = nr_turns
        self.nr_bpm = nr_bpm
        self.bpm_idx = self.nr_bpm + 50 * (self.nr_turns - 1)
        self._wait_change = 1

        self.get_pvs()

        print('Waiting for PVs connection...')
        print('='*50)
        while True:
            if self.check_connect():
                break

        self.pv_nr_pts_sp.value = nr_pts

        quad_lim = _np.ones(len(self._name_quads)) * d_quad
        corr_lim = _np.ones(len(self._name_corrs)) * d_corr
        sept_lim = _np.ones(len(self._name_sept)) * d_sept
        kckr_lim = _np.ones(len(self._name_kckr)) * d_kckr

        up = _np.concatenate((quad_lim, corr_lim, sept_lim, kckr_lim))
        down = -1 * up
        self.set_limits(upper=up, lower=down)

        print(
            'The script will take {:.2f} minutes to be finished'.format(
                (nr_pts*0.5 + self._wait_change) *
                self.nswarm * self.niter / 60))
        print('='*50)

        self.reference = _np.array([h.value for h in self.hands])
        self.reset_wait_buffer()
        self.init_obj_func()
        print('Initial Objective Function {:.5f}'.format(self.f_init))
        print('='*50)

    def get_pvs(self):
        """."""
        # Number of turns to measure Sum Signal
        prefix = ''
        self.pv_nr_sample = _PV(
            prefix + 'BO-Glob:AP-SOFB:TrigNrSamplesPost-SP')
        _time.sleep(self._wait_change)
        self.pv_nr_sample.value = int(self.nr_turns)

        # Diagnostic to calculate Objective Function
        self.eyes = _PV(
            prefix + 'BO-Glob:AP-SOFB:MTurnSum-Mon', auto_monitor=True)

        self._name_quads = [
            # 'TB-01:MA-QF1', 'TB-01:MA-QD1',
            'TB-02:MA-QF2A', 'TB-02:MA-QD2A',
            'TB-02:MA-QF2B', 'TB-02:MA-QD2B',
            'TB-03:MA-QF3', 'TB-03:MA-QD3',
            'TB-04:MA-QF4', 'TB-04:MA-QD4',
        ]
        self._name_quads = [prefix + q + ':KL-SP' for q in self._name_quads]
        self._name_hands.extend(self._name_quads)

        self._name_corrs = [
            # 'TB-01:MA-CH-1', 'TB-01:MA-CV-1',
            # 'TB-01:MA-CH-2', 'TB-01:MA-CV-2',
            # 'TB-02:MA-CH-1', 'TB-02:MA-CV-1',
            # 'TB-02:MA-CH-2', 'TB-02:MA-CV-2',
            'TB-04:MA-CH-1', 'TB-04:MA-CV-1',
            # 'TB-04:MA-CH-2',  # Magnet transformed to QS
            'TB-04:MA-CV-2',
        ]
        self._name_corrs = [prefix + c + ':Kick-SP' for c in self._name_corrs]
        self._name_hands.extend(self._name_corrs)

        self._name_sept = [prefix + 'TB-04:PM-InjSept:Kick-SP']
        self._name_hands.extend(self._name_sept)

        self._name_kckr = [prefix + 'BO-01D:PM-InjKckr:Kick-SP']
        self._name_hands.extend(self._name_kckr)

        # Actuator to change settings
        self.hands = [_PV(h) for h in self._name_hands]

        self.pv_nr_pts_sp = _PV(prefix + 'BO-Glob:AP-SOFB:SmoothNrPts-SP')
        self.pv_nr_pts_rb = _PV(prefix + 'BO-Glob:AP-SOFB:SmoothNrPts-RB')
        self.pv_buffer_mon = _PV(prefix + 'BO-Glob:AP-SOFB:BufferCount-Mon')
        self.pv_buffer_reset = _PV(prefix + 'BO-Glob:AP-SOFB:SmoothReset-Cmd')

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
        _time.sleep(self._wait_change)

        while True:
            if self.pv_buffer_mon.value == self.pv_nr_pts_rb.value:
                break

    def save_bpms_sum(self):
        """."""
        with open('BPM_Sum.txt', 'a') as sbpm:
            sbpm.write('='*50)
            _np.savetxt(sbpm, self.eyes.value[:self.bpm_idx], fmt='%+.8e')

    def init_obj_func(self):
        """."""
        self.f_init = -_np.sum(self.eyes.value[:self.bpm_idx])

    def calc_obj_fun(self):
        """."""
        f_out = _np.zeros(self.nswarm)

        for i in range(self.nswarm):
            chg = self.get_change(i)
            self.set_change(chg)
            # _time.sleep(self._wait_change)
            self.reset_wait_buffer()
            self.save_bpms_sum()
            f_out[i] = _np.sum(self.eyes.value[:self.bpm_idx])
            print(
                'Particle {:02d}/{:d} | Obj. Func. : {:f}'.format(
                    i+1, self.nswarm, f_out[i]))
        print('='*50)

        return - f_out

    def run(self):
        """."""
        self.start()
        pos = self.best_positions_history
        fig = self.best_figures_history
        plt.plot(-fig, '-o')
        plt.xlabel('Number of Iteractions')
        plt.ylabel('Objective Function')
        plt.savefig('obj_fun_PSO.png')
        plt.show()
        print(
            'The Objective Function changed from {:.5f} to {:.5f}'.format(
                self.f_init, _np.abs(fig[-1])))
        if _np.abs(self.f_init) < _np.abs(fig[-1]):
            imp = (_np.abs(fig[-1]/self.f_init) - 1) * 100
            print('The script improved the system in {:.3f} %!'.format(imp))
            set_opt = input(
                'Do you want to set the best configuration found? (y or n):  ')
            if set_opt == 'y':
                _np.savetxt('initial_reference.txt', self.reference)
                _np.savetxt('initial_obj_fun.txt', self.f_init)
                best_setting = self.reference + pos[-1, :]
                self.set_change(best_setting)
                print('Best configuration found was set to the machine!')
            else:
                print('Ok... Setting initial reference!')
                self.set_change(self.reference)
        else:
            print('It was not possible to improve the system...')
            print('Setting initial reference.')
            self.set_change(self.reference)
        return pos, fig


class SAInjection(SimulAnneal):
    """."""

    def __init__(self, save=False):
        """."""
        self.reference = []
        self.niter = 0
        self.nr_turns = 0
        self.nr_bpm = 0
        self.bpm_idx = 0
        self.temperature = 0
        self._name_hands = []
        self._name_quads = []
        self._name_corrs = []
        self._name_sept = []
        self._name_kckr = []
        self.eyes = []
        self.hands = []
        self.pv_nr_pts_sp = []
        self.pv_nr_pts_rb = []
        self.pv_buffer_mon = []
        self.pv_buffer_reset = []
        self.pv_nr_sample = []
        self._wait_change = 0
        self.f_init = 0
        SimulAnneal.__init__(self, save=save)

    def initialization(self):
        """."""
        print('='*50)
        d_quad = input(
            'TB Quads KL Max Delta (Default = 0.5 [1/m]): ')
        d_corr = input(
            'TB Corrs Kick Max Delta (Default = 500 [urad]): ')
        d_sept = input('InjSept Variation (Default = 2 [mrad]): ')
        d_kckr = input('Inj Kicker Max Delta (Default = 2 [mrad]): ')
        temp = input('Initial Temperature (Default = 0): ')
        print('='*50)
        nr_iter = input('Number of Iteractions (Default = 100): ')
        nr_pts = input('Buffer Size (SOFB) (Default = 10): ')
        nr_turns = input(
            'Number of Turns to Measure BPM Sum Signal (Default = 1): ')
        nr_bpm = input(
            'Set the last BPM to Read ' +
            '(Default = 50, Range = [1,50]): ')
        print('='*50)

        if not nr_iter:
            nr_iter = 100
        if not nr_pts:
            nr_pts = 10
        if not nr_bpm:
            nr_bpm = 50
        if not nr_turns:
            nr_turns = 1
        if not nr_pts:
            nr_pts = 10
        if not d_quad:
            d_quad = 0.5
        if not d_corr:
            d_corr = 500
        if not d_kckr:
            d_kckr = 2
        if not d_sept:
            d_sept = 2
        if not temp:
            temp = 0

        d_quad = float(d_quad)
        d_corr = float(d_corr)
        d_sept = float(d_sept)
        d_kckr = float(d_kckr)
        temp = float(temp)
        nr_iter = int(nr_iter)
        nr_pts = int(nr_pts)
        nr_turns = int(nr_turns)
        nr_bpm = int(nr_bpm)

        if not d_quad + d_corr + d_sept + d_kckr:
            raise Exception('You have set zero variation for all dimensions!')

        self.niter = nr_iter
        self.nr_turns = nr_turns
        self.nr_bpm = nr_bpm
        self.bpm_idx = self.nr_bpm + 50 * (self.nr_turns - 1)
        self.temperature = temp
        self._wait_change = 1

        self.get_pvs()

        print('Waiting for PVs connection...')
        print('='*50)
        while True:
            if self.check_connect():
                break

        self.pv_nr_pts_sp.value = nr_pts

        quad_lim = _np.ones(len(self._name_quads)) * d_quad
        corr_lim = _np.ones(len(self._name_corrs)) * d_corr
        sept_lim = _np.ones(len(self._name_sept)) * d_sept
        kckr_lim = _np.ones(len(self._name_kckr)) * d_kckr

        delta = _np.concatenate((quad_lim, corr_lim, sept_lim, kckr_lim))
        self.set_deltas(dmax=delta)

        print(
            'The script will take {:.2f} minutes to be finished'.format(
                (nr_pts*0.5 + self._wait_change) * self.niter / 60))
        print('='*50)

        self.reference = _np.array([h.value for h in self.hands])
        self.position = self.reference
        self.reset_wait_buffer()
        self.init_obj_fun()

        print('Initial Objective Function {:.5f}'.format(self.f_init))
        print('='*50)

    def get_pvs(self):
        """."""
        prefix = ''
        # Number of turns to measure Sum Signal
        self.pv_nr_sample = _PV(
            prefix + 'BO-Glob:AP-SOFB:TrigNrSamplesPost-SP')
        _time.sleep(self._wait_change)
        self.pv_nr_sample.value = int(self.nr_turns)

        # Diagnostic to calculate Objective Function
        self.eyes = _PV(
            prefix + 'BO-Glob:AP-SOFB:MTurnSum-Mon', auto_monitor=True)

        self._name_quads = [
            # 'TB-01:MA-QF1', 'TB-01:MA-QD1',
            'TB-02:MA-QF2A', 'TB-02:MA-QD2A',
            'TB-02:MA-QF2B', 'TB-02:MA-QD2B',
            'TB-03:MA-QF3', 'TB-03:MA-QD3',
            'TB-04:MA-QF4', 'TB-04:MA-QD4',
        ]
        self._name_quads = [prefix + q + ':KL-SP' for q in self._name_quads]
        self._name_hands.extend(self._name_quads)

        self._name_corrs = [
            # 'TB-01:MA-CH-1', 'TB-01:MA-CV-1',
            # 'TB-01:MA-CH-2', 'TB-01:MA-CV-2',
            # 'TB-02:MA-CH-1', 'TB-02:MA-CV-1',
            # 'TB-02:MA-CH-2', 'TB-02:MA-CV-2',
            'TB-04:MA-CH-1', 'TB-04:MA-CV-1',
            # 'TB-04:MA-CH-2',  # Magnet transformed to QS
            'TB-04:MA-CV-2',
        ]
        self._name_corrs = [prefix + c + ':Kick-SP' for c in self._name_corrs]
        self._name_hands.extend(self._name_corrs)

        self._name_sept = [prefix + 'TB-04:PM-InjSept:Kick-SP']
        self._name_hands.extend(self._name_sept)

        self._name_kckr = [prefix + 'BO-01D:PM-InjKckr:Kick-SP']
        self._name_hands.extend(self._name_kckr)

        # Actuator to change settings
        self.hands = [_PV(h) for h in self._name_hands]

        self.pv_nr_pts_sp = _PV(prefix + 'BO-Glob:AP-SOFB:SmoothNrPts-SP')
        self.pv_nr_pts_rb = _PV(prefix + 'BO-Glob:AP-SOFB:SmoothNrPts-RB')
        self.pv_buffer_mon = _PV(prefix + 'BO-Glob:AP-SOFB:BufferCount-Mon')
        self.pv_buffer_reset = _PV(prefix + 'BO-Glob:AP-SOFB:SmoothReset-Cmd')

    def check_connect(self):
        """."""
        conh = [h.connected for h in self.hands]
        cone = self.eyes.connected
        if cone and sum(conh) == len(conh):
            con = False
        else:
            con = True
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
        self.pv_buffer_reset.value = 1
        _time.sleep(self._wait_change)

        while True:
            if self.pv_buffer_mon.value == self.pv_nr_pts_rb.value:
                break

    def init_obj_fun(self):
        """."""
        self.f_init = -_np.sum(self.eyes.value[:self.bpm_idx])

    def save_bpms_sum(self):
        """."""
        with open('BPM_Sum.txt', 'a') as sbpm:
            sbpm.write(print('='*50))
            _np.savetxt(sbpm, self.eyes.value[:self.bpm_idx], fmt='%+.8e')

    def calc_obj_fun(self):
        """."""
        f_out = []
        chg = self.get_change()
        self.set_change(chg)
        # _time.sleep(self._wait_change)
        self.reset_wait_buffer()
        self.save_bpms_sum()
        f_out = _np.sum(self.eyes.value[:self.bpm_idx])
        return - f_out

    def run(self):
        """."""
        self.start()
        pos = self.best_positions_history
        fig = self.best_figures_history
        plt.plot(-fig, '-o')
        plt.xlabel('Number of Iteractions')
        plt.ylabel('Objective Function')
        plt.savefig('obj_fun_SA.png')
        plt.show()
        print(
            'The Objective Function changed from {:.5f} to {:.5f}'.format(
                self.f_init, _np.abs(fig[-1])))
        if _np.abs(self.f_init) < _np.abs(fig[-1]):
            imp = (_np.abs(fig[-1]/self.f_init) - 1) * 100
            print(
                'The script improved the system in {:.3f} %!'.format(imp))
            set_opt = input(
                'Set the best configuration found? (y or n): ')
            if set_opt == 'y':
                _np.savetxt('initial_reference.txt', self.reference)
                _np.savetxt('initial_obj_fun.txt', self.f_init)
                self.set_change(pos[-1, :])
                print('Best configuration found was set to the machine!')
            else:
                print('Ok... Setting initial reference!')
                self.set_change(self.reference)
        return pos, fig


if __name__ == "__main__":
    import os
    print('='*50)
    print('OPTIMIZATION SCRIPT FOR BOOSTER INJECTION')
    print('='*50)
    for f in os.listdir('.'):
        if f.endswith('.txt'):
            os.system('rm *.txt')
        if f.endswith('.png'):
            os.system('rm *.png')
    FOLDER = input('Enter the folder name to save the data: ')
    MODE = input('Choose the optimization method (PSO or SA): ')
    if MODE == 'PSO':
        INJBO = PSOInjection(save=True)
        INJBO.run()
    elif MODE == 'SA':
        INJBO = SAInjection(save=True)
        INJBO.run()
    else:
        raise Exception('Invalid method!')
    CMD1 = 'mkdir ' + FOLDER
    os.system(CMD1)
    CMD2 = 'mv *.txt *.png ' + FOLDER
    os.system(CMD2)
