"""."""
import logging as _log
import sys as _sys
from copy import deepcopy as _dcopy
from threading import Event as _Event

from epics.ca import CAThread as _Thread
from mathphys.functions import load as _load, save as _save

import apsuite.commisslib as _commisslib


class DataBaseClass:
    """."""

    def __init__(self, params=None):
        """."""
        self.data = dict()
        self.params = params

    def to_dict(self) -> dict:
        """Dump all relevant object properties to dictionary.

        Returns:
            dict: contains all relevant properties of object.

        """
        return dict(data=self.data, params=self.params.to_dict())

    def from_dict(self, info: dict):
        """Update all relevant info from dictionary.

        Args:
            info (dict): dictionary with all relevant info.

        Returns:
            keys_not_used (set): Set containing keys not used by
                `self.params` object.

        """
        self.data = _dcopy(info['data'])
        params = info['params']
        if not isinstance(params, dict):
            params = params.to_dict()
        keys_not_used = self.params.from_dict(params)
        if keys_not_used:
            print('Some keys were not used in params')
        return keys_not_used

    def save_data(self, fname: str, overwrite=False, compress=False):
        """Save `data` and `params` to pickle or HDF5 file.

        Args:
            fname (str): name of the file. If extension is not provided,
                '.pickle' will be added and a pickle file will be assumed.
                If provided, must be '.pickle' for pickle files or
                {'.h5', '.hdf5', '.hdf', '.hd5'} for HDF5 files.
            overwrite (bool, optional): Whether to overwrite existing file.
                Defaults to False.
            compress (bool, optional): Whether to compress data before saving.

        """
        _save(self.to_dict(), fname, overwrite=overwrite, compress=compress)

    def load_and_apply(self, fname: str):
        """Load and apply `data` and `params` from pickle or HDF5 file.

        Args:
            fname (str): name of the pickle file. If extension is not provided,
                '.pickle' will be added and a pickle file will be assumed.
                If provided, must be {'.pickle', '.pkl'} for pickle files or
                {'.h5', '.hdf5', '.hdf', '.hd5'} for HDF5 files.

        """
        self.from_dict(self.load_data(fname))

    @staticmethod
    def load_data(fname: str):
        """Load and return `data` and `params` from pickle or HDF5 file.

        Args:
            fname (str): name of the pickle file. If extension is not provided,
                '.pickle' will be added and a pickle file will be assumed.
                If provided, must be {'.pickle', '.pkl'} for pickle files or
                {'.h5', '.hdf5', '.hdf', '.hd5'} for HDF5 files.

        Returns:
            data (dict): Dictionary with keys: `data` and `params`.

        """
        try:
            data = _load(fname)
        except ModuleNotFoundError:
            _sys.modules['apsuite.commissioning_scripts'] = _commisslib
            data = _load(fname)
        return data


class ParamsBaseClass:
    """."""

    def to_dict(self):
        """Dump all relevant object properties to dictionary.

        Returns:
            dict: contains all relevant properties of object.

        """
        return _dcopy(self.__dict__)

    def from_dict(self, params_dict):
        """Update all relevant info from dictionary.

        Args:
            params_dict (dict): dictionary with all relevant info.

        Returns:
            keys_not_used (set): Set containing keys not used.

        """
        keys_not_used = set()
        for k, v in params_dict.items():
            if k in self.__dict__:
                self.__dict__[k] = v
            else:
                keys_not_used.add(k)
        return keys_not_used


class MeasBaseClass(DataBaseClass):
    """."""

    def __init__(self, params=None, isonline=True):
        """."""
        super().__init__(params=params)
        self.isonline = bool(isonline)
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
        super().__init__(params=params, isonline=isonline)
        self._target = target
        self._stopevt = _Event()
        self._finished = _Event()
        self._finished.set()
        self._thread = _Thread(target=self._run, daemon=True)

    @property
    def target(self):
        """Target function to be executed by Thread."""
        return self._target

    @target.setter
    def target(self, func):
        if callable(func):
            self._target = func

    def start(self):
        """."""
        if self.ismeasuring:
            _log.error('There is another measurement happening.')
            return
        self._stopevt.clear()
        self._finished.clear()
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
        return self._finished.wait(timeout=timeout)

    def _run(self):
        if self._target is not None:
            self._target()
        self._finished.set()
