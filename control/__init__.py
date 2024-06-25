"""."""

import multiprocessing as _mp
import threading as _threading

from .. import CONTROL_SYSTEM, is_online
from ..utils import get_namedtuple as _get_namedtuple
from . import epics as _epics, tango as _tango, simulation as _simul

ControlSystemOptions = _get_namedtuple(
    "ControlSystemOptions", ("Epics", "Tango"))

# the following parameter is used to establish connections with PVs.
CONNECTION_TIMEOUT = 0.050  # [s]
GET_TIMEOUT = 5.0  # [s]


class PV:
    """."""

    def __init__(self, devname, propty, **kwargs):
        """."""
        self._control_system = CONTROL_SYSTEM
        if not is_online():
            self._obj = _simul.PV(devname, propty, **kwargs)
        elif CONTROL_SYSTEM == ControlSystemOptions.Epics:
            self._obj = _epics.PV(devname, propty, **kwargs)
        elif CONTROL_SYSTEM == ControlSystemOptions.Tango:
            # NOTE: tango equivalent of PV class in epics needs implementation
            raise ValueError('To be implemented.')

    @property
    def connected(self):
        """."""
        return self._obj.connected

    @property
    def value(self):
        """."""
        return self._obj.value

    @value.setter
    def value(self, value):
        self.put(value)

    def put(self, value, wait=False):
        """."""
        if self._control_system == ControlSystemOptions.Epics:
            self._obj.put(value, wait=wait)
        elif CONTROL_SYSTEM == ControlSystemOptions.Tango:
            raise ValueError('To be implemented.')

    def wait_for_connection(self, timeout=None):
        """."""
        return self._obj.wait_for_connection(timeout=timeout)


class Thread:
    """."""

    def __new__(cls, *args, **kwargs):
        """."""
        if CONTROL_SYSTEM == ControlSystemOptions.Epics:
            return _epics.CAThread(*args, **kwargs)
        elif CONTROL_SYSTEM == ControlSystemOptions.Tango:
            # NOTE: Does tango needs special classes for new processes?
            return _threading.Thread(*args, **kwargs)


class ProcessSpawn:
    """."""

    def __new__(cls, *args, **kwargs):
        """."""
        if CONTROL_SYSTEM == ControlSystemOptions.Epics:
            return _epics.CAProcessSpawn(*args, **kwargs)
        elif CONTROL_SYSTEM == ControlSystemOptions.Tango:
            # NOTE: Does tango needs special classes for new processes?
            return _mp.get_context("spawn").Process(*args, **kwargs)
