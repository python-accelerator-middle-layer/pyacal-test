"""."""

import time as _time
from threading import Thread, Event

import numpy as _np
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs
import matplotlib.cm as _mcm

from siriuspy.devices import RFCav, SOFB, DCCT, EVG
from apsuite.commissioning_scripts.base import BaseClass
import pyaccel as _pyaccel
from pymodels import si as _si, bo as _bo


class Params:
    """."""

    def __init__(self):
        """."""
        self._sweep_type = 'phase'
        self.phase_ini = -177.5
        self.phase_fin = 177.5
        self.phase_delta = 5
        self.voltage_ini = 50
        self.voltage_fin = 150
        self.voltage_delta = 1
        self.nrpulses = 20
        self.freq_pulses = 2
        self.sofb_timeout = 10
        self.rf_timeout = 10
        self.tim_timeout = 10
        self.wait_rf = 2

    @property
    def sweep_type(self):
        """."""
        return self._sweep_type

    @sweep_type.setter
    def sweep_type(self, val):
        if val.lower().startswith('pha'):
            self._sweep_type = 'phase'
        else:
            self._sweep_type = 'voltage'

    @property
    def isphasesweep(self):
        """."""
        return self._sweep_type.startswith('pha')

    def __str__(self):
        """."""
        strep = '{0:30s}= {1:s}\n'.format('sweep_type', self.sweep_type)
        if self.sweep_type.lower().startswith('pha'):
            strep += '{0:30s}= {1:9.3f}\n'.format(
                'initial phase [°]', self.phase_ini)
            strep += '{0:30s}= {1:9.3f}\n'.format(
                'final phase [°]', self.phase_fin)
            strep += '{0:30s}= {1:9.3f}\n'.format(
                'delta phase [°]', self.phase_delta)
        else:
            strep += '{0:30s}= {1:9.3f}\n'.format(
                'initial voltage [mV]', self.voltage_ini)
            strep += '{0:30s}= {1:9.3f}\n'.format(
                'final voltage [mV]', self.voltage_fin)
            strep += '{0:30s}= {1:9.3f}\n'.format(
                'delta voltage [mV]', self.voltage_delta)
        strep += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'pulses freq [Hz]', self.freq_pulses)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'SOFB timeout', self.sofb_timeout)
        strep += '{0:30s}= {1:9.3f}\n'.format('RF timeout', self.rf_timeout)
        strep += '{0:30s}= {1:9.3f}\n'.format('Wait RF', self.wait_rf)
        strep += '{0:30s}= {1:9.3f}\n'.format(
            'Timing timeout', self.tim_timeout)
        return strep


class ControlRF(BaseClass):
    """."""

    def __init__(self, acc=None):
        """."""
        super().__init__(Params())
        if acc in ('BO', 'SI'):
            self.acc = acc
        else:
            raise Exception('Set acc to BO or SI')

        if self.acc == 'BO':
            self._disp_avg = 220e3  # in um
            self.model = _bo.create_accelerator()
            self._mod_cavidx = _pyaccel.lattice.find_indices(
                self.model, 'fam_name', 'P5Cav')[0]
        else:
            self._disp_avg = 25e3  # in um
            self.model = _si.create_accelerator()
            self._mod_cavidx = _pyaccel.lattice.find_indices(
                self.model, 'fam_name', 'SRFCav')[0]
        self._mod_freq0 = self.model[self._mod_cavidx].frequency
        self.model.cavity_on = True
        self.model.radiation_on = True

        devname_rf, devname_sofb = ControlRF._get_devnames(acc)
        devname_dcct = ControlRF._get_devnames_dcct(acc)

        self.devices = {
            'tim': EVG(),
            'rf': RFCav(devname_rf),
            'sofb': SOFB(devname_sofb),
            'dcct': DCCT(devname_dcct)
            }
        self.data = {
            'phase': [],
            'voltage': [],
            'power': [],
            'sum': [],
            'orbx': [],
            'orby': [],
            'dcct': [],
            'time': None,
            }
        self._thread = Thread(target=self._do_scan)
        self._stopped = Event()

    def get_scan_param_span(self):
        """."""
        if self.params.isphasesweep:
            ini = self.params.phase_ini
            fin = self.params.phase_fin
            dlt = self.params.phase_delta
        else:
            ini = self.params.voltage_ini
            fin = self.params.voltage_fin
            dlt = self.params.voltage_delta

        npts = abs(int((fin - ini)/dlt)) + 1
        return _np.linspace(ini, fin, npts)

    def start(self):
        """."""
        if not self._thread.is_alive():
            self._thread = Thread(target=self._do_scan, daemon=True)
            self._stopped.clear()
            self._thread.start()

    def stop(self):
        """."""
        self._stopped.set()

    def plot_energy_vs_turns(self, npts=None, navg=1, cut_sum=1.8):
        """."""
        nbpms = self.devices['sofb'].data.nr_bpms
        if npts:
            npts = npts - (npts % navg)
        slc = slice(npts)

        data = self.data
        if self.params.isphasesweep:
            var_span = _np.mean(data['phase'], axis=1)
        else:
            var_span = _np.mean(data['voltage'], axis=1)

        fig = _mplt.figure(figsize=(7, 4))
        gs = _mgs.GridSpec(1, 1)
        gs.update(
            left=0.1, right=0.78, bottom=0.15, top=0.9, hspace=0.5,
            wspace=0.35)
        axx = fig.add_subplot(gs[0, 0])

        indcs = _np.linspace(0, 1, var_span.size)
        cmap = _mcm.brg(indcs)
        for i, (orb, sum_) in enumerate(zip(data['orbx'], data['sum'])):
            bpm_sum = _np.mean(sum_.reshape(-1, nbpms)[slc, :], axis=1)*1e6
            bpm_orbx = _np.mean(orb.reshape(-1, nbpms)[slc, :], axis=1)

            nturns = _np.arange(0, bpm_sum.size, navg)
            bpm_sum = _np.mean(bpm_sum.reshape(-1, navg), axis=1)
            bpm_orbx = _np.mean(bpm_orbx.reshape(-1, navg), axis=1)
            ind = bpm_sum > cut_sum
            axx.plot(
                nturns[ind], bpm_orbx[ind]/self._disp_avg*100, '-',
                color=cmap[i], label='{0:.1f}'.format(var_span[i]))
            axx.plot([0, 100], [0, -0.5/3e3*100*100], color='k', label='No RF')
        axx.grid(True)
        ncol = (var_span.size // 20) + 1
        axx.legend(
            loc='center left', bbox_to_anchor=(1., 0.5), fontsize='xx-small',
            ncol=ncol, )
        axx.set_title('Energy Variation vs Time')
        axx.set_ylabel(r'$\delta$ [%]')
        axx.set_xlabel('Number of Turns')
        return fig

    def plot_scan_param_vs_beam_intensity(self, isbpm=False):
        """."""
        fig = _mplt.figure(figsize=(7, 4))
        gs = _mgs.GridSpec(1, 1)
        gs.update(left=0.10, right=0.98, hspace=0, wspace=0.25)
        axs = fig.add_subplot(gs[0, 0])

        if self.params.isphasesweep:
            var = _np.array(self.data['phase']).flatten()
        else:
            var = _np.array(self.data['voltage']).flatten()
        if isbpm:
            current = _np.mean(_np.array(self.data['sum']), axis=1)
        else:
            current = _np.array(self.data['data']['dcct']).flatten()*1000
        pol = _np.polynomial.polynomial.polyfit(var, current, 2)

        axs.plot(var, current, 'o')
        axs.plot(var, _np.polynominal.polynomial.polyval(pol, var)*1000, '--')
        axs.grid(True)
        if isbpm:
            axs.set_ylabel('BPM Sum Signal [au]')
        else:
            axs.set_ylabel('DCCT Current [uA]')
        if self.params.isphasesweep:
            axs.set_title('RF Phase Scan')
            axs.set_xlabel('RF Phase [deg]')
        else:
            axs.set_title('RF Voltage Scan')
            axs.set_xlabel('RF Voltage [mV]')

        return fig

    def do_tracking(self, nturns, deltaf=0, voltage=1.4e6, npart=10):
        """."""
        mod = self.model
        idx = self._mod_cavidx
        mod[idx].voltage = voltage
        mod[idx].frequency = self._mod_freq0 + deltaf
        inn = _np.zeros((npart, 6))
        inn[:, 5] = _np.linspace(-0.30, 0.30, npart)
        inn[:, 4] = 0.00

        rout = []
        for _ in range(nturns):
            rou, *_ = _pyaccel.tracking.line_pass(mod, inn, indices=None)
            inn = rou[-npart:, :]
            inn[:, 5] -= 299792458*864/mod[idx].frequency - mod.length
            rout.append(rou)
        return _np.array(rout)

    @staticmethod
    def plot_tracking(rout, lims=None):
        """."""
        fig = _mplt.figure(figsize=(7, 5))
        gs = _mgs.GridSpec(2, 1)
        gs.update(left=0.10, right=0.98, hspace=0.2, wspace=0.25)

        axx = fig.add_subplot(gs[0, 0])
        lines = axx.plot(rout[:, :, 4]*100, '.')
        dener = -0.5/3e3*100*100
        axx.plot([0, 100], [0, dener], color='k', label='Without RF')
        indcs = _np.linspace(0, 1, len(lines))
        cmap = _mcm.brg(indcs)
        for cor, line in zip(cmap, lines):
            line.set_color(cor)
        axx.grid(True)
        axx.set_ylim([-3.5, 2] if lims is None else lims)

        axy = fig.add_subplot(gs[1, 0])
        lines = axy.plot(rout[:, :, 5]/60*360, rout[:, :, 4]*100, '.')
        for cor, line in zip(cmap, lines):
            line.set_color(cor)
        axy.grid(True)
        return fig

    def _do_scan(self):
        nrpul = self.params.nrpulses
        freq = self.params.freq_pulses

        self.devices['sofb'].nr_points = nrpul

        isphase = self.params.isphasesweep
        var_span = self.get_scan_param_span()
        self.data['phase'] = []
        self.data['voltage'] = []
        self.data['power'] = []
        self.data['sum'] = []
        self.data['orbx'] = []
        self.data['orby'] = []
        self.data['dcct'] = []
        print('Starting Loop')
        for val in var_span:
            print('Turning pulses off --> ', end='')
            self.devices['tim'].cmd_turn_off_pulses(self.params.tim_timeout)
            print(f'varying {self.params.sweep_type:s} --> ', end='')
            self._vary(val, isphase=isphase)
            _time.sleep(self.params.wait_rf)
            phase_data = _np.zeros(nrpul)
            voltage_data = _np.zeros(nrpul)
            power_data = _np.zeros(nrpul)
            dcct_data = _np.zeros(nrpul)
            print('turning pulses on --> ', end='')
            self.devices['tim'].cmd_turn_on_pulses(self.params.tim_timeout)
            print('Getting data ', end='')
            self.devices['sofb'].cmd_reset()
            for k in range(nrpul):
                print('.', end='')
                phase_data[k] = self.devices['rf'].dev_llrf.phase
                voltage_data[k] = self.devices['rf'].dev_llrf.voltage
                power_data[k] = self.devices['rf'].dev_rfpowmon.power
                dcct_data[k] = _np.mean(self.devices['dcct'].current_fast)
                _time.sleep(1/freq)
                if self._stopped.is_set():
                    break
            self.devices['sofb'].wait_buffer(self.params.sofb_timeout)
            self.data['phase'].append(phase_data)
            self.data['voltage'].append(voltage_data)
            self.data['power'].append(power_data)
            self.data['sum'].append(self.devices['sofb'].mt_sum)
            self.data['orbx'].append(self.devices['sofb'].mt_trajx)
            self.data['orby'].append(self.devices['sofb'].mt_trajy)
            self.data['dcct'].append(dcct_data)
            self.data['time'] = _time.time()
            if isphase:
                print('Phase [°]: {0:8.3f}'.format(
                    self.devices['rf'].dev_llrf.phase))
            else:
                print('Voltage [mV]: {0:8.3f}'.format(
                    self.devices['rf'].dev_llrf.voltage))
            if self._stopped.is_set():
                print('Stopped!')
                break
        self.devices['tim'].cmd_turn_off_pulses(self.params.tim_timeout)
        print('Finished!')

    def _vary(self, val, isphase=True):
        if isphase:
            self.devices['rf'].cmd_set_phase(
                val, timeout=self.params.rf_timeout)
        else:
            self.devices['rf'].cmd_set_voltage(
                val, timeout=self.params.rf_timeout)

    @staticmethod
    def _get_devnames(acc):
        if acc is None:
            devname_rf, devname_sofb = None, None
        elif acc.upper() == 'SI':
            devname_rf, devname_sofb = RFCav.DEVICES.SI, SOFB.DEVICES.SI
        elif acc.upper() == 'BO':
            devname_rf, devname_sofb = RFCav.DEVICES.BO, SOFB.DEVICES.BO
        else:
            devname_rf, devname_sofb = None, None
        return devname_rf, devname_sofb

    @staticmethod
    def _get_devnames_dcct(acc):
        if acc is None:
            devname_dcct = None
        elif acc.upper() == 'SI':
            devname_dcct = DCCT.DEVICES.SI_13C4
        elif acc.upper() == 'BO':
            devname_dcct = DCCT.DEVICES.BO
        else:
            devname_dcct = None
        return devname_dcct
