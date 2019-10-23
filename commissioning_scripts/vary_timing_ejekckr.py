import time as _time
import pickle as _pickle
import numpy as np

from pymodels.middlelayer.devices import SOFB, Kicker


class Params:

    def __init__(self):
        self.delay_ini = 0
        self.delay_fin = 1.7
        self.delay_step = 0.05
        self.kckr_voltage = 20
        self.nrpulses = 20
        self.sofb_timeout = 30
        self.wait_kckr = 1

    def __str__(self):
        st = '{0:30s}= {1:9.3f}\n'.format('initial delay [us]', self.delay_ini)
        st += '{0:30s}= {1:9.3f}\n'.format('final delay [us]', self.delay_fin)
        st += '{0:30s}= {1:9.3f}\n'.format('step delay [us]', self.delay_step)
        st += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        st += '{0:30s}= {1:9.3f}\n'.format('SOFB timeout', self.sofb_timeout)
        st += '{0:30s}= {1:9.3f}\n'.format('Wait Kicker', self.wait_kckr)
        return st


class ControlRF:
    def __init__(self):
        self.params = Params()
        self.sofb = SOFB('BO')
        self.kicker = Kicker('BO-48D:TI-EjeKckr')
        self.data_sum = []
        self.data_orbx = []
        self.data_orby = []

    @property
    def connected(self):
        conn = self.sofb.connected
        conn &= self.kicker.connected
        return conn

    @property
    def delay_spam(self):
        ini = self.params.delay_ini
        fin = self.params.delay_fin
        dlt = self.params.delay_step
        return self._calc_spam(ini, fin, dlt)

    @staticmethod
    def _calc_spam(ini, fin, dlt):
        npts = abs(int((fin - ini)/dlt)) + 1
        return np.linspace(ini, fin, npts)

    def do_scan(self):
        self.sofb.nr_points = self.params.nrpulses

        var_spam = self.delay_spam
        self.data_sum = []
        self.data_orbx = []
        self.data_orby = []
        self.kicker.voltage = self.params.kckr_voltage
        print('Starting Loop')
        for val in var_spam:
            self.kicker.delay = val
            self.kicker.turnoff_pulses()
            _time.sleep(self.params.wait_kckr)
            self.sofb.reset()
            self.sofb.wait(self.params.sofb_timeout)
            data_sum = [self.sofb.sum, ]
            data_orbx = [self.sofb.trajx, ]
            data_orby = [self.sofb.trajy, ]

            self.kicker.turnon_pulses()
            self.sofb.reset()
            self.sofb.wait(self.params.sofb_timeout)
            data_sum.append(self.sofb.sum)
            data_orbx.append(self.sofb.trajx)
            data_orby.append(self.sofb.trajy)

            self.data_sum.append(data_sum)
            self.data_orbx.append(data_orbx)
            self.data_orby.append(data_orby)
            std = np.std(data_orbx[1] - data_orbx[0])
            print('Delay [us]: {0:8.3f} -> orb std [um]: {1:8.3f}'.format(
                    self.kicker.delay, std))
        print('Finished!')

    def save_data(self, fname):
        data = dict(
            params=self.params,
            data_orbx=self.data_orbx,
            data_orby=self.data_orby,
            data_sum=self.data_sum,
            delay_spam=self.delay_spam,
            )
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'wb') as f:
            _pickle.dump(data, f)

    @staticmethod
    def load_data(fname):
        if not fname.endswith('.pickle'):
            fname += '.pickle'
        with open(fname, 'rb') as f:
            data = _pickle.load(f)
        return data
