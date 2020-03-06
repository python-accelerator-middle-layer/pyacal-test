import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


class CalcTune():

    TUNEX = 19.20433
    TUNEY = 7.31417
    NBPM = 50
    RING_LEN = 495.7045

    def __init__(self, file=None, plane=None, corr_num=None):
        self._file = file
        self.data = np.loadtxt(self._file, delimiter=',')
        self._plane = plane
        self._corr_num = corr_num
        self._aerror = 0
        self._perror = 0
        self.define_bpms()
        mml_folder = '/home/facs/repos/MatlabMiddleLayer/Release/'
        self._s_data = np.loadtxt(
            mml_folder + 'applications/lnls/+sirius_commis/spos.txt')

        if self.data.ndim > 1:
            self.ncorr = len(self.data[0, :])
            self.int_tune = np.zeros(self.ncorr)
            self.tune = np.zeros(self.ncorr)
            self.count_zeros_matrix()
            self.fit_tune_matrix()
        elif self.data.ndim == 1:
            self._shift_corr = -2 * self._corr_num
            self.data = np.roll(self.data, self._shift_corr)
            self.bpms = np.roll(self.bpms, self._shift_corr)
            self.count_zeros_vector()
            # self._s_data = np.roll(self._s_data, self._shift_corr)
            self._s_data[self._s_data < 0] += self.RING_LEN
            self.data, self.bpms, self._s_data = self._find_bpm_49(
                self.data, self.bpms, self._s_data)
            self.fit_tune_vector()

    def count_zeros_vector(self):
        nz = 0

        if np.sign(self.data[0]) != np.sign(self.data[-1]):
            nz += 1

        for i in range(len(self.data)-1):
            if np.sign(self.data[i]) != np.sign(self.data[i+1]):
                nz += 1

        print('The closest integer to the tune is ' + str(nz/2))
        self.int_tune = int(nz/2)
        return int(nz/2)

    def count_zeros_matrix(self):
        nz = np.zeros(self.ncorr)

        for k in range(self.ncorr):
            line = self.data[:, k]
            if np.sign(line[0]) != np.sign(line[-1]):
                nz[k] += 1

            for i in range(self.NBPM-1):
                if np.sign(line[i]) != np.sign(line[i+1]):
                    nz[k] += 1

            print('The closest integer to the tune is ' + str(nz[k]/2))
            self.int_tune[k] = int(nz[k]/2)

    def plot_data(self):
        plt.plot(self.bpms, self.data, '-o')
        plt.xticks(rotation=90)
        plt.grid(b=True)
        plt.xlabel('BPMs label')
        # plt.show()

    def fit_tune_vector(self):
        line = self.data
        if self._plane == 'x':
            par, par_cov = opt.curve_fit(
                self.fit_cosine, self._s_data, line,
                p0=[max(line, key=abs), self.TUNEX, 0])
        elif self._plane == 'y':
            par, par_cov = opt.curve_fit(
                self.fit_cosine, self._s_data, line,
                p0=[max(line, key=abs), self.TUNEY, 0])
        self.plot_data()
        plt.plot(
            self.fit_cosine(self._s_data, par[0], par[1], par[2]),
            '--o', label='Fit')

        print('====================================')
        print('           Fitted Parameters        ')
        print('====================================')
        print('|| Amplitude ||  Tune  ||  Phase  ||')
        print('|| {:.5f}  ||{:.5f} || {:.5f} ||'.format(
            par[0], par[1], par[2]))
        print('====================================')
        self.tune = par[1]

        if self._plane == 'x':
            print(
                'FITTED TUNE: {:.5f} || NOMINAL TUNE: {:.5f}'.format(
                    self.tune, self.TUNEX))
        if self._plane == 'y':
            print(
                'FITTED TUNE: {:.5f} || NOMINAL TUNE: {:.5f}'.format(
                    self.tune, self.TUNEY))

        self.calc_error()
        print(
            'DIFF. TO NOMINAL VALUE: {:.5f} ({:.5f} %)'.format(
                self._aerror, self._perror))
        plt.show()

    def fit_tune_matrix(self):

        for k in range(self.ncorr):
            line = self.data[:, k]
            shift = -2 * k
            line = np.roll(line, shift)
            bpm = np.roll(self.bpms, shift)
            spos = self._s_data
            # spos = np.roll(self._s_data, shift)
            spos[spos < 0] += self.RING_LEN
            line, bpm, spos = self._find_bpm_49(line, bpm, spos)
            if self._plane == 'x':
                par, par_cov = opt.curve_fit(
                    self.fit_cosine, spos, line,
                    p0=[max(line, key=abs), self.TUNEX, 0])
            elif self._plane == 'y':
                par, par_cov = opt.curve_fit(
                    self.fit_cosine, spos, line,
                    p0=[max(line, key=abs), self.TUNEY, 0])
            print('Fitted Parameters')
            print('=================')
            print('|| Amplitude || Tune || Phase ||')
            print(par[0], par[1], par[2])
            self.tune[k] = par[1]

        self.tune = np.mean(self.tune)
        if self._plane == 'x':
            print(
                'FITTED TUNE: {:.5f} || NOMINAL TUNE: {:.5f}'.format(
                    self.tune, self.TUNEX))
        if self._plane == 'y':
            print(
                'FITTED TUNE: {:.5f} || NOMINAL TUNE: {:.5f}'.format(
                    self.tune, self.TUNEY))
        self.calc_error()
        print(
            'DIFF. TO NOMINAL VALUE: {:.5f} ({:.5f} %)'.format(
                self._aerror, self._perror))

    def fit_cosine(self, s, A, MU, PHI):
        return A * np.cos(2 * np.pi * MU * s / self.RING_LEN + PHI)

    def _find_bpm_49(self, data, bpm, spos):
        idx = bpm.tolist().index('49')
        if idx < self.NBPM/2:
            new_bpm = bpm[idx+1:]
            new_spos = spos[idx+1:]
            new_data = data[idx+1:]
        else:
            new_bpm = bpm[:idx]
            new_spos = spos[:idx]
            new_data = data[:idx]
        return new_data, new_bpm, new_spos

    def define_bpms(self):
        self.bpms = list(range(2, self.NBPM+1))
        self.bpms.append(1)
        self.bpms = [str(b).zfill(2) for b in self.bpms]

    def calc_error(self):
        if self._plane == 'x':
            self._aerror = np.mean(self.tune) - self.TUNEX
            self._perror = 100 * self._aerror/self.TUNEX
        if self._plane == 'y':
            self._aerror = np.mean(self.tune) - self.TUNEY
            self._perror = 100 * self._aerror/self.TUNEY
