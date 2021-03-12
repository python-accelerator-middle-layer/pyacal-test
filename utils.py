"""."""
from threading import Thread as _Thread, Event as _Event
import logging as _log
import sys as _sys
import apsuite.commisslib as _commisslib

from mathphys.functions import save_pickle as _save_pickle, \
    load_pickle as _load_pickle


class DataBaseClass:
    """."""

    def __init__(self, params=None):
        """."""
        self.data = dict()
        self.params = params

    def save_data(self, fname, overwrite=False):
        """."""
        data = dict(params=self.params.to_dict(), data=self.data)
        _save_pickle(data, fname, overwrite=overwrite)

    def load_and_apply(self, fname):
        """."""
        data = self.load_data(fname)
        self.data = data['data']
        params = data['params']
        if params is not dict:
            params = params.to_dict()
        self.params.from_dict(params_dict=params)

    @staticmethod
    def load_data(fname):
        """."""
        try:
            data = _load_pickle(fname)
        except ModuleNotFoundError:
            _sys.modules['apsuite.commissioning_scripts'] = _commisslib
            data = _load_pickle(fname)
        return data


class ParamsBaseClass:
    """."""

    def to_dict(self):
        """."""
        return self.__dict__

    def from_dict(self, params_dict):
        """."""
        self.__dict__.update(params_dict)


class MeasBaseClass(DataBaseClass):
    """."""

    def __init__(self, params=None):
        """."""
        super().__init__(params=params)
        self.devices = dict()
        self.analysis = dict()
        self.pvs = dict()

    @property
    def connected(self):
        """."""
        conn = all([dev.connected for dev in self.devices.values()])
        conn &= all([pv.connected for pv in self.pvs.values()])
        return conn

    def wait_for_connection(self, timeout=None):
        """."""
        obs = list(self.devices.values()) + list(self.pvs.values())
        for dev in obs:
            if not dev.wait_for_connection(timeout=timeout):
                return False
        return True


class ThreadedMeasBaseClass(MeasBaseClass):
    """."""

    def __init__(self, params=None, target=None, isonline=True):
        """."""
        super().__init__(params=params)
        self.isonline = bool(isonline)
        self._target = target
        self._stopevt = _Event()
        self._thread = _Thread(target=self._run, daemon=True)

    def start(self):
        """."""
        if self.ismeasuring:
            _log.error('There is another measurement happening.')
            return
        self._stopevt.clear()
        self._thread = _Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """."""
        self._stopevt.set()

    @property
    def ismeasuring(self):
        """."""
        return self._thread.is_alive()

    def wait_measurement(self, timeout=None):
        """Wait for measurement to finish."""
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        else:
            return True
        return self._thread.is_alive()

    def _run(self):
        if self._target is not None:
            self._target()
